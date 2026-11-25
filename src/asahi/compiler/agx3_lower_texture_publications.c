/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx3_resource_ir.h"
#include "agx3_staging.h"
#include <string.h>

struct publication {
   agx_index coordinates, reference;
   uint32_t value;
   unsigned cost;
};

static bool
same_register_value(agx_index a, agx_index b)
{
   /* Cache/discard annotations describe individual reads, not values. */
   return a.type == b.type && a.value == b.value && a.size == b.size &&
          a.channels_m1 == b.channels_m1 && a.memory == b.memory &&
          a.abs == b.abs && a.neg == b.neg;
}

static bool
overlaps_register(agx_index a, agx_index b)
{
   return a.type == AGX_INDEX_REGISTER && b.type == AGX_INDEX_REGISTER &&
          a.memory == b.memory &&
          a.value < b.value + agx_index_size_16(b) &&
          b.value < a.value + agx_index_size_16(a);
}

/* Native register-only instructions preserve the sampler-private file. Keep
 * compound memory/texture operations, control flow, coverage changes and
 * arithmetic publications as region boundaries until their native resource
 * effects are represented. This classifies effects; it does not reorder them.
 */
static bool
preserves_publications(const agx_instr *I)
{
   if (agx_texture_resource(I))
      return true;
   const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
   if (I->agx3_tex_coord_stage || I->nr_dests != 1 ||
       I->dest[0].type != AGX_INDEX_REGISTER || I->dest[0].memory ||
       !info->can_eliminate || !info->can_reorder ||
       info->schedule_class != AGX_SCHEDULE_CLASS_NONE)
      return false;
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_INTERP_AGX3:
   case AGX_OPCODE_ITRCOORDS_AGX3:
   case AGX_OPCODE_RHW_INTERP_AGX3:
   case AGX_OPCODE_LDCF:
      return false;
   default:
      return true;
   }
}

/* This is the relocatable coordinate ABI, independent of shader identity or
 * instruction neighbourhood. Other native operand ABIs still have compound
 * lowering. Arithmetic-to-staging retains its helper-lane provenance.
 */
static bool
has_coordinate_publication(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || agx_texture_resource(I) ||
       I->agx3_tex_coords_staged || I->dim != AGX_DIM_2D || I->query_lod ||
       (I->offset && !I->offset_xyz) || I->nr_srcs != 6 ||
       I->src[0].type != AGX_INDEX_REGISTER || I->src[0].memory ||
       I->src[0].abs || I->src[0].neg || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 2)
      return false;

   if (I->shadow && (I->src[5].type != AGX_INDEX_REGISTER ||
                    I->src[5].memory || I->src[5].abs || I->src[5].neg ||
                    I->src[5].size != AGX_SIZE_32))
      return false;

   return I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
          ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
            I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
           I->src[1].type == AGX_INDEX_IMMEDIATE);
}

bool
agx3_lower_texture_publications(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   uint32_t next = 0;
   agx_foreach_instr_global(ctx, I)
      next = MAX3(next, agx_texture_resource(I), agx_publication_resource(I));

   bool progress = false;
   agx_foreach_block(ctx, block) {
      /* At most one four-half pair fits per entry. Reserving each available
       * value's aligned footprint bounds future-use overlap by the actual
       * register-file capacity, independently of the number of samples.
       * Eviction ends reuse eligibility; it never removes a required copy.
       */
      struct publication available[AGX3_STAGING_HALFS / 4];
      unsigned count = 0, reserved = 0;
      agx_foreach_instr_in_block_safe(block, I) {
         if (has_coordinate_publication(I)) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index reference = agx_null();
            if (I->shadow) {
               reference = I->src[5];
               reference.channels_m1 = 0;
            }
            unsigned match = count;
            for (unsigned j = 0; j < count; ++j) {
               if (same_register_value(available[j].coordinates, I->src[0]) &&
                   same_register_value(available[j].reference, reference)) {
                  match = j;
                  break;
               }
            }
            uint32_t value;
            if (match < count) {
               const struct publication hit = available[match];
               value = hit.value;
               memmove(&available[match], &available[match + 1],
                        (count - match - 1) * sizeof(*available));
               available[count - 1] = hit;
            } else {
               const unsigned cost = I->shadow ? 8 : 4;
               while (reserved + cost > AGX3_STAGING_HALFS) {
                  reserved -= available[0].cost;
                  memmove(available, available + 1, --count * sizeof(*available));
               }
               agx_publish_texture_agx3(&b, I->src[0], reference, ++next);
               value = next;
               available[count++] = (struct publication){
                  I->src[0], reference, value, cost,
               };
               reserved += cost;
            }
            /* The resource, not its original GPRs, feeds later samples. */
            I->src[0] = agx_get_vec_index(value, AGX_SIZE_32, I->shadow ? 3 : 2);
            I->src[0].type = AGX_INDEX_RESOURCE;
            if (I->shadow)
               I->src[5] = agx_null();
            progress = true;
         }

         if (!preserves_publications(I)) {
            count = reserved = 0;
            continue;
         }
         /* Any overlapping physical definition changes the input value,
          * including an asynchronous sample result or a half-register write.
          */
         for (unsigned j = 0; j < count;) {
            bool overwritten = false;
            agx_foreach_dest(I, d) {
               overwritten |= overlaps_register(I->dest[d], available[j].coordinates) ||
                              overlaps_register(I->dest[d], available[j].reference);
            }
            if (!overwritten) {
               ++j;
               continue;
            }
            reserved -= available[j].cost;
            memmove(&available[j], &available[j + 1],
                     (--count - j) * sizeof(*available));
         }
      }
   }
   return progress;
}

static bool
scalar_publication_source(agx_index src)
{
   return (src.type == AGX_INDEX_NORMAL || src.type == AGX_INDEX_REGISTER) &&
          !src.memory && !src.abs && !src.neg && src.size == AGX_SIZE_32 &&
          agx_channels(src) == 1;
}

static bool
scalar_cube_sample(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || I->nr_srcs != 6 ||
       I->nr_dests != 1 || I->dim != AGX_DIM_CUBE || I->shadow || I->offset ||
       I->query_lod || I->sparse || I->gather != AGX_GATHER_NONE ||
       I->lod_mode != AGX_LOD_MODE_AUTO_LOD || I->agx3_tex_coords_staged ||
       I->src[0].type != AGX_INDEX_NORMAL || I->src[0].memory ||
       I->src[0].size != AGX_SIZE_32 || agx_channels(I->src[0]) != 3 ||
       I->src[0].abs || I->src[0].neg ||
       I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].value != 0 ||
       !agx_is_null(I->src[5]))
      return false;
   for (unsigned q = 3; q <= 4; ++q) {
      if (I->src[q].type != AGX_INDEX_UNIFORM &&
          I->src[q].type != AGX_INDEX_IMMEDIATE)
         return false;
   }
   return true;
}

bool
agx3_lower_scalar_cube_coordinates(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return false;

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   BITSET_WORD *vector_uses =
      rzalloc_array(ctx, BITSET_WORD, BITSET_WORDS(ctx->alloc));
   if (!defs || !vector_uses)
      return false;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
      bool scalar_sample = scalar_cube_sample(I);
      agx_foreach_ssa_src(I, q) {
         if (q != 0 || !scalar_sample)
            BITSET_SET(vector_uses, I->src[q].value);
      }
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (!scalar_cube_sample(I))
         continue;
      if (BITSET_TEST(vector_uses, I->src[0].value))
         continue;
      agx_instr *collect = defs[I->src[0].value];
      if (!collect || collect->op != AGX_OPCODE_COLLECT ||
          collect->nr_srcs != 3)
         continue;
      bool eligible = true;
      for (unsigned c = 0; c < 3; ++c) {
         agx_index src = collect->src[c];
         eligible &= scalar_publication_source(src) &&
                     (src.type != AGX_INDEX_REGISTER || src.value <= 126);
      }
      if (!eligible)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_instr *sample = agx_alloc_instr(&b, I->op, 1, 8);
      memcpy(&sample->op, &I->op, sizeof(*I) - offsetof(agx_instr, op));
      sample->nr_srcs = 8;
      sample->dest[0] = I->dest[0];
      memcpy(sample->src, I->src, 6 * sizeof(agx_index));
      sample->src[0] = collect->src[0];
      sample->src[6] = collect->src[1];
      sample->src[7] = collect->src[2];
      sample->agx3_cube_scalar_coords = true;
      if (!ctx->collect_split_hints.data)
         util_dynarray_init(&ctx->collect_split_hints, ctx);
      agx_instr *hint = agx_alloc_instr(&b, AGX_OPCODE_COLLECT, 1, 3);
      hint->dest[0] = collect->dest[0];
      for (unsigned c = 0; c < 3; ++c)
         hint->src[c] = collect->src[c];
      util_dynarray_append(&ctx->collect_split_hints, hint);
      agx_builder_insert(&b.cursor, sample);
      agx_remove_instruction(I);
      progress = true;
   }
   return progress;
}

/* An M5 texel fetch stages its coordinates with one COPY per component, and a
 * multisampled or array read publishes its sample index or layer with
 * PACK_SMPIR, so the read never needs its coordinates as a contiguous vector.
 * A filtered 2D sample stages x and y with one COPY each as well; its level,
 * bias, gradient, offset and comparison operands stage on their own. A
 * collected vector per read costs a GPR move for every component that already
 * lives elsewhere: a depth pyramid that reads four samples at each of four
 * texels moves three words before each of its sixteen reads, and a filtered
 * shadow lookup moves both coordinates of every tap. Hand the read the
 * components themselves. */
static bool
split_coords_candidate(const agx_instr *I)
{
   if (I->nr_srcs != 6 || I->nr_dests != 1 || I->agx3_tex_coords_staged ||
       I->agx3_split_coords || I->agx3_cube_scalar_coords || I->query_lod ||
       I->sparse || I->src[0].type != AGX_INDEX_NORMAL || I->src[0].memory ||
       I->src[0].abs || I->src[0].neg || I->src[0].size != AGX_SIZE_32)
      return false;

   if (I->op == AGX_OPCODE_TEXTURE_SAMPLE)
      return I->dim == AGX_DIM_2D && agx_channels(I->src[0]) == 2;

   return I->op == AGX_OPCODE_TEXTURE_LOAD && !I->offset && !I->shadow &&
          I->gather == AGX_GATHER_NONE &&
          (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_2D_MS ||
           I->dim == AGX_DIM_2D_ARRAY) &&
          agx_channels(I->src[0]) == (I->dim == AGX_DIM_2D ? 2 : 3) &&
          agx_is_null(I->src[5]);
}

bool
agx3_split_texture_coordinates(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C || ctx->is_preamble)
      return false;

   /* G18G re-encodes the vector-coordinate fetch/sample forms. Its emitter
    * publishes components from consecutive registers and does not consume
    * the eight-source split-coordinate representation used by M5. Keep the
    * collect so register allocation preserves that input layout.
    */
   if (agx_is_g18g(&ctx->key->dev))
      return false;

   /* Every M5 stage stages its fetch and sample coordinates
    * (agx3_pack_binary.c). */
   switch (ctx->stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_FRAGMENT:
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_MESH:
   case MESA_SHADER_TASK:
      break;
   default:
      return false;
   }

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   if (!defs)
      return false;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (!split_coords_candidate(I))
         continue;
      const unsigned n = agx_channels(I->src[0]);
      agx_instr *collect = defs[I->src[0].value];
      if (!collect || collect->op != AGX_OPCODE_COLLECT ||
          collect->nr_srcs != n)
         continue;
      bool eligible = true;
      for (unsigned c = 0; c < n; ++c) {
         eligible &= collect->src[c].type == AGX_INDEX_NORMAL &&
                     scalar_publication_source(collect->src[c]);
         /* A component that is also another operand (a level, a reference
          * or an offset word) would name that operand's register: its own
          * staging may then discard the register before or after the
          * coordinate COPY reads it. Keep the vector. */
         for (unsigned s = 1; s < 6; ++s) {
            eligible &= I->src[s].type != AGX_INDEX_NORMAL ||
                        I->src[s].value != collect->src[c].value;
         }
      }
      if (!eligible)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_instr *split = agx_alloc_instr(&b, I->op, 1, 8);
      memcpy(&split->op, &I->op, sizeof(*I) - offsetof(agx_instr, op));
      split->nr_srcs = 8;
      split->dest[0] = I->dest[0];
      memcpy(split->src, I->src, 6 * sizeof(agx_index));
      for (unsigned c = 0; c < 3; ++c) {
         agx_index component = c < n ? collect->src[c] : agx_null();
         /* Liveness is recomputed by RA; drop the collect's annotations. */
         component.kill = component.cache = component.discard = false;
         split->src[c ? c + 5 : 0] = component;
      }
      split->agx3_split_coords = true;

      /* A fetch keeps the vector's placement as an RA affinity hint: where
       * the components can sit together, allocation matches the vector's. A
       * sample does not: filter taps pair each coordinate with several
       * others (a PCF footprint reads x0 with y0 and y1, x1 with both), so a
       * pair hint per tap only constrains where the coordinates may live. */
      if (I->op == AGX_OPCODE_TEXTURE_LOAD) {
         if (!ctx->collect_split_hints.data)
            util_dynarray_init(&ctx->collect_split_hints, ctx);
         agx_instr *hint = agx_alloc_instr(&b, AGX_OPCODE_COLLECT, 1, n);
         hint->dest[0] = collect->dest[0];
         for (unsigned c = 0; c < n; ++c)
            hint->src[c] = collect->src[c];
         util_dynarray_append(&ctx->collect_split_hints, hint);
      }

      agx_builder_insert(&b.cursor, split);
      agx_remove_instruction(I);
      progress = true;
   }
   return progress;
}

/* The packer computed every cube sample's CEM in one shared scratch above
 * the allocation, eight halves the file had to leave free. A cube array
 * stages CEM's face in a seven-bit PACK_SMPIR field
 * (agx3_tex_stage_cube_array_layer), so for cube arrays that scratch had to
 * sit below half 128 and capped the whole register file at 112 halves.
 *
 * Give each cube sample its own scratch instead: an undefined four-word
 * vector the sample reads, which RA places clear of every value live at the
 * sample, below half 128 for a cube array (texture_gpr). Its definition
 * emits no code. Run right before RA, so the scratch lives for this one
 * instruction. A bias with minimum also packs a word above the allocation
 * and keeps the shared scratch, as do scalar-coordinate cube samples. */
bool
agx3_reserve_cube_scratch(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE ||
          (I->dim != AGX_DIM_CUBE && I->dim != AGX_DIM_CUBE_ARRAY) ||
          I->nr_srcs != 6 || I->nr_dests != 1 ||
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index scratch = agx_vec_temp(ctx, AGX_SIZE_32, 4);
      agx_instr *collect = agx_collect_to(&b, scratch, 4);
      agx_foreach_src(collect, c)
         collect->src[c] = agx_undef(AGX_SIZE_32);

      agx_instr *sample = agx_alloc_instr(&b, I->op, 1, 7);
      memcpy(&sample->op, &I->op, sizeof(*I) - offsetof(agx_instr, op));
      sample->nr_srcs = 7;
      sample->dest[0] = I->dest[0];
      memcpy(sample->src, I->src, 6 * sizeof(agx_index));
      sample->src[6] = scratch;
      agx_builder_insert(&b.cursor, sample);
      agx_remove_instruction(I);
      progress = true;
   }
   return progress;
}

static bool
ssa_preserves_publications(const agx_instr *I)
{
   if (agx_texture_resource(I))
      return true;
   const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
   if (I->agx3_tex_coord_stage || !I->nr_dests ||
       !info->can_eliminate || !info->can_reorder ||
       info->schedule_class != AGX_SCHEDULE_CLASS_NONE)
      return false;
   /* SPLIT/collect are SSA wiring, not native resource effects. In particular,
    * splitting a texture result must not discard its still-live coordinates.
    */
   agx_foreach_dest(I, d) {
      if ((I->dest[d].type != AGX_INDEX_NORMAL &&
           I->dest[d].type != AGX_INDEX_NULL) || I->dest[d].memory)
         return false;
   }
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_INTERP_AGX3:
   case AGX_OPCODE_ITRCOORDS_AGX3:
   case AGX_OPCODE_RHW_INTERP_AGX3:
   case AGX_OPCODE_LDCF:
      return false;
   default:
      return true;
   }
}

bool
agx3_lower_scalar_texture_publications(agx_context *ctx)
{
   const bool m5 = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B) ||
       agx_is_g18g(&ctx->key->dev) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return false;

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   if (!defs)
      return false;
   uint32_t next = 0;
   agx_foreach_instr_global(ctx, I) {
      next = MAX3(next, agx_texture_resource(I), agx_publication_resource(I));
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }

   struct scalar_publication {
      agx_index x, y, reference;
      uint32_t value;
      unsigned cost;
   };
   bool progress = false;
   agx_foreach_block(ctx, block) {
      struct scalar_publication available[AGX3_STAGING_HALFS / 4];
      unsigned count = 0, reserved = 0;
      agx_foreach_instr_in_block_safe(block, I) {
         bool eligible =
            I->op == AGX_OPCODE_TEXTURE_SAMPLE && !agx_texture_resource(I) &&
            !I->agx3_tex_coords_staged && I->dim == (m5 ? AGX_DIM_3D : AGX_DIM_2D) && !I->query_lod &&
            (!I->offset || I->offset_xyz) && I->nr_srcs == 6 &&
            I->src[0].type == AGX_INDEX_NORMAL && !I->src[0].memory &&
            !I->src[0].abs && !I->src[0].neg && I->src[0].size == AGX_SIZE_32 &&
            agx_channels(I->src[0]) == (m5 ? 3 : 2) &&
            (!m5 || (I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_32 &&
                     !I->shadow && !I->offset && !I->sparse &&
                     I->gather == AGX_GATHER_NONE &&
                     I->lod_mode == AGX_LOD_MODE_AUTO_LOD)) &&
            (I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
             ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
               I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
              I->src[1].type == AGX_INDEX_IMMEDIATE));
         agx_instr *collect = eligible ? defs[I->src[0].value] : NULL;
         eligible &= collect && collect->op == AGX_OPCODE_COLLECT &&
                     collect->nr_srcs == (m5 ? 3 : 2) &&
                     (!m5 || scalar_publication_source(collect->src[2])) &&
                     scalar_publication_source(collect->src[0]) &&
                     scalar_publication_source(collect->src[1]) &&
                     (!I->shadow || scalar_publication_source(I->src[5]));
         if (eligible) {
            agx_index x = collect->src[0], y = collect->src[1];
            agx_index reference = m5 ? collect->src[2] :
                                  I->shadow ? I->src[5] : agx_null();
            unsigned match = count;
            for (unsigned j = 0; j < count; ++j) {
               if (same_register_value(available[j].x, x) &&
                   same_register_value(available[j].y, y) &&
                   same_register_value(available[j].reference, reference)) {
                  match = j;
                  break;
               }
            }
            uint32_t value;
            if (match < count) {
               const struct scalar_publication hit = available[match];
               value = hit.value;
               memmove(&available[match], &available[match + 1],
                       (count - match - 1) * sizeof(*available));
               available[count - 1] = hit;
            } else {
               const unsigned cost = (m5 || I->shadow) ? 8 : 4;
               while (reserved + cost > AGX3_STAGING_HALFS) {
                  reserved -= available[0].cost;
                  memmove(available, available + 1, --count * sizeof(*available));
               }
               agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
               /* The M5 publication stages x, y and z with the wide scalar
                * COPY, which reads the whole register file, so it reads
                * them where they live. */
               agx_publish_texture_scalar_agx3(&b, x, y, reference, ++next);
               value = next;
               available[count++] = (struct scalar_publication){
                  x, y, reference, value, cost,
               };
               reserved += cost;
            }
            I->src[0] = agx_get_vec_index(value, AGX_SIZE_32, (m5 || I->shadow) ? 3 : 2);
            I->src[0].type = AGX_INDEX_RESOURCE;
            if (I->shadow)
               I->src[5] = agx_null();
            progress = true;
         }
         if (!ssa_preserves_publications(I))
            count = reserved = 0;
      }
   }
   ralloc_free(defs);
   return progress;
}

static bool
texture_publications_fit(agx_context *ctx)
{
   unsigned count = 0, capacity = 0, use_capacity = 0, max_id = 0;
   agx_foreach_instr_global(ctx, I) {
      count++;
      capacity += I->agx3_tex_coord_start || agx_is_texture_publication(I);
      use_capacity += I->agx3_tex_coords_staged || agx_texture_resource(I);
      max_id = MAX3(max_id, agx_publication_resource(I), agx_texture_resource(I));
   }
   void *mem = ralloc_context(NULL);
   struct agx3_resource_value *values =
      rzalloc_array(mem, struct agx3_resource_value, capacity);
   struct agx3_resource_use *uses =
      rzalloc_array(mem, struct agx3_resource_use, use_capacity);
   struct agx3_slot_instr *base =
      rzalloc_array(mem, struct agx3_slot_instr, count);
   uint32_t *ids = rzalloc_array(mem, uint32_t, max_id + 1);
   uint16_t *placements = rzalloc_array(mem, uint16_t, capacity);
   bool ok = false;
   if (!mem || !values || !uses || !base || !ids || !placements)
      goto done;
   unsigned value_count = 0, use_count = 0, k = 0, active = UINT32_MAX;
   agx_foreach_instr_global(ctx, I) {
      base[k].is_async = I->agx3_tex_coord_start ||
         agx_is_texture_publication(I) || I->op == AGX_OPCODE_TEXTURE_SAMPLE;
      if (I->agx3_tex_coord_start) {
         if (active != UINT32_MAX)
            goto done;
         active = value_count++;
         values[active] = (struct agx3_resource_value){
            .producer = k, .halves = 4, .alignment = 4,
            .fixed_base = I->op == AGX_OPCODE_FFMA ? UINT16_MAX : 0,
         };
      }
      if (agx_is_texture_publication(I)) {
         const bool scalar = I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3;
         if (!I->imm || ids[I->imm])
            goto done;
         if (I->agx3_tex_coords_staged) {
            if (active == UINT32_MAX)
               goto done;
            ids[I->imm] = active + 1;
            active = UINT32_MAX;
         } else {
            ids[I->imm] = ++value_count;
            values[value_count - 1] = (struct agx3_resource_value){
               .producer = k,
               .halves = agx_is_null(I->src[scalar ? 2 : 1]) ? 4 : 6,
               .alignment = 4, .fixed_base = UINT16_MAX,
            };
         }
      } else if (I->agx3_tex_coords_staged) {
         if (active == UINT32_MAX)
            goto done;
         uses[use_count++] = (struct agx3_resource_use){
            .instruction = k, .value = active, .count = 4,
            .through_completion = true,
         };
         active = UINT32_MAX;
      }
      unsigned id = agx_texture_resource(I);
      if (id) {
         if (!ids[id])
            goto done;
         uses[use_count++] = (struct agx3_resource_use){
            .instruction = k, .value = ids[id] - 1,
            .count = agx_index_size_16(I->src[0]), .through_completion = true,
         };
      }
      k++;
   }
   ok = active == UINT32_MAX &&
      agx3_resource_lower(values, value_count, uses, use_count, base, count,
                           AGX3_STAGING_HALFS, placements, base);
done:
   ralloc_free(mem);
   return ok;
}

/* Non-published texture and image operations stage their operands at a fixed
 * sampler-private bank that the resource allocator does not reserve. */
static bool
staging_without_resource(const agx_instr *I)
{
   return (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
           I->op == AGX_OPCODE_TEXTURE_LOAD ||
           I->op == AGX_OPCODE_IMAGE_LOAD) &&
          !agx_texture_resource(I) && !I->agx3_tex_coords_staged;
}

static bool
fma_staging_producer(const agx_instr *I, bool immutable_uniforms)
{
   if (!I || I->op != AGX_OPCODE_FFMA || I->nr_srcs != 3 || I->nr_dests != 1 ||
       I->dest[0].type != AGX_INDEX_NORMAL || I->dest[0].size != AGX_SIZE_32 ||
       agx_channels(I->dest[0]) != 1 || I->saturate || I->agx3_tex_coord_stage)
      return false;
   unsigned special = 0;
   for (unsigned i = 0; i < 3; ++i) {
      const agx_index src = I->src[i];
      const bool immediate = src.type == AGX_INDEX_IMMEDIATE;
      if (src.size != (immediate ? AGX_SIZE_16 : AGX_SIZE_32) ||
          agx_channels(src) != 1 || src.memory || src.abs || src.neg)
         return false;
      if (immediate) {
         if (i == 2 || src.value >= 256)
            return false;
         special++;
      } else if (src.type == AGX_INDEX_UNIFORM) {
         if (!immutable_uniforms || i == 2 || src.value >= 256 ||
             (src.value & 1))
            return false;
         special++;
      } else if (src.type != AGX_INDEX_NORMAL)
         return false;
   }
   return special <= 1;
}

bool
agx3_fuse_fma_texture_publications(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies || list_length(&ctx->blocks) != 1)
      return false;
   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   unsigned *uses = rzalloc_array(ctx, unsigned, ctx->alloc);
   if (!defs || !uses)
      return false;
   bool immutable = true;
   agx_foreach_instr_global(ctx, I) {
      immutable &= I->op != AGX_OPCODE_UNIFORM_STORE &&
                   I->op != AGX_OPCODE_UNIFORM_FADD_AGX3 &&
                   I->op != AGX_OPCODE_UNIFORM_FMUL_AGX3 &&
                   I->op != AGX_OPCODE_UNIFORM_FFMA_AGX3;
      agx_foreach_ssa_src(I, i)
         uses[I->src[i].value]++;
      agx_foreach_ssa_dest(I, i)
         defs[I->dest[i].value] = I;
   }
   bool measured = false;
   bool progress = false;
   unsigned demand = 0;
   agx_foreach_instr_global_safe(ctx, publication) {
      if (publication->op != AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3 ||
          publication->agx3_tex_coords_staged ||
          publication->nr_srcs != 3 || !agx_is_null(publication->src[2]))
         continue;
      agx_instr *producer[2];
      bool valid = true;
      for (unsigned c = 0; c < 2; ++c) {
         const agx_index src = publication->src[c];
         if (src.type != AGX_INDEX_NORMAL || src.size != AGX_SIZE_32 ||
             agx_channels(src) != 1 || src.abs || src.neg || src.memory ||
             uses[src.value] != 1 ||
             !fma_staging_producer(defs[src.value], immutable)) {
            valid = false;
            break;
         }
         producer[c] = defs[src.value];
      }
      if (!valid || producer[0] == producer[1])
         continue;
      unsigned earlier = 0, found = 0;
      bool crosses_staging = false;
      agx_foreach_instr_global(ctx, I) {
         if (I == publication)
            break;
         if (I == producer[0] || I == producer[1]) {
            unsigned c = I == producer[1];
            if (!found)
               earlier = c;
            found |= 1u << c;
         } else if (found == 3 && staging_without_resource(I)) {
            crosses_staging = true;
         } else if (found &&
                    (I->op == AGX_OPCODE_BEGIN_CF ||
                     agx_opcodes_info[I->op].schedule_class ==
                        AGX_SCHEDULE_CLASS_INVALID ||
                     (agx_opcodes_info[I->op].schedule_class ==
                         AGX_SCHEDULE_CLASS_COVERAGE &&
                      (!I->nr_dests || !agx_opcodes_info[I->op].can_reorder)))) {
            valid = false;
            break;
         }
      }
      if (!valid || found != 3)
         continue;
      if (!measured) {
         agx_compute_liveness(ctx);
         demand = agx_calc_register_demand(ctx, false);
         measured = true;
      }
      struct list_head *next[2] = {producer[0]->link.next,
                                   producer[1]->link.next};
      struct list_head *publication_next = publication->link.next;
      agx_index dest[2] = {producer[0]->dest[0], producer[1]->dest[0]};
      agx_index src[2] = {publication->src[0], publication->src[1]};
      /* Hoisting the publication to its producers would keep it live across
       * that instruction's staging, so only sink the producers. */
      for (unsigned late = crosses_staging; late < 2; ++late) {
         struct list_head *anchor = late ? publication_next : next[earlier ^ 1];
         if (anchor == &publication->link)
            anchor = publication_next;
         list_del(&publication->link);
         for (unsigned c = 0; c < 2; ++c) {
            list_del(&producer[c]->link);
            list_addtail(&producer[c]->link, anchor);
            producer[c]->dest[0] = agx_null();
            producer[c]->dest[0].size = AGX_SIZE_32;
            publication->src[c] = agx_null();
            producer[c]->agx3_tex_coord_stage = true;
            producer[c]->agx3_tex_coord_component = c;
         }
         list_addtail(&publication->link, anchor);
         producer[0]->agx3_tex_coord_start = true;
         publication->agx3_tex_coords_staged = true;
         agx_compute_liveness(ctx);
         if (agx_calc_register_demand(ctx, false) <= demand &&
             texture_publications_fit(ctx)) {
            progress = true;
            break;
         }
         publication->agx3_tex_coords_staged = false;
         for (unsigned c = 0; c < 2; ++c) {
            producer[c]->dest[0] = dest[c];
            publication->src[c] = src[c];
            producer[c]->agx3_tex_coord_stage = false;
            producer[c]->agx3_tex_coord_start = false;
            producer[c]->agx3_tex_coord_component = 0;
         }
         list_del(&publication->link);
         list_addtail(&publication->link, publication_next);
         list_del(&producer[0]->link);
         list_del(&producer[1]->link);
         list_addtail(&producer[earlier ^ 1]->link, next[earlier ^ 1]);
         list_addtail(&producer[earlier]->link, next[earlier]);
         agx_compute_liveness(ctx);
      }
   }
   return progress;
}
