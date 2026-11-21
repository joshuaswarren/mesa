/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "agx_compile.h"
#include "agx_helpers.h"
#include "agx_pack.h"
#include "agx_uvs.h"
#include "agx3_vary_core.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

static bool
inline_uvs_index(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct agx_varyings_vs *varyings = data;
   if (intr->intrinsic != nir_intrinsic_load_uvs_index_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned slot = nir_intrinsic_io_semantics(intr).location;

   nir_def_replace(&intr->def, nir_imm_intN_t(b, varyings->slots[slot], 16));
   return true;
}

struct ctx {
   nir_def *layer, *viewport;
   nir_cursor after_layer_viewport;
   int layer_zero_root_offset;
   struct agx_unlinked_uvs_layout *layout;
   bool layer_as_varying;

   /* Relocatable layouts: each slot's placeholder UVS index. */
   uint8_t placeholder[VARYING_SLOT_MAX];
};

static enum uvs_group
group_for_varying(gl_varying_slot loc)
{
   switch (loc) {
   case VARYING_SLOT_POS:
      return UVS_POSITION;
   case VARYING_SLOT_PSIZ:
      return UVS_PSIZ;
   default:
      return UVS_VARYINGS;
   }
}

static nir_def *
upconvert_uvs_value(nir_builder *b, nir_intrinsic_instr *intr, nir_def *value)
{
   /* Native ST_VARY stores a 32-bit component.  A mediump producer may still
    * be 16-bit even when the logical store_output type is 32-bit, so perform
    * the type-correct conversion before replacing the generic IO intrinsic.
    * Passing the 16-bit value straight through makes the backend allocate a
    * 16-bit register, and the always-live ST_VARY pack assertion then fires
    * inside whatever process asked for the shader.
    */
   if (value->bit_size == 32)
      return value;

   assert(value->bit_size == 16);

   nir_alu_type type =
      nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));
   assert(type == nir_type_float || type == nir_type_int ||
          type == nir_type_uint);

   return nir_convert_to_bit_size(b, value, type, 32);
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct ctx *ctx = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned component = nir_intrinsic_component(intr);

   nir_def *value = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;

   value = upconvert_uvs_value(b, intr, value);

   /* If there is only 1 user varying, it is at the base of the varying section.
    * This saves us an indirection on simple separate shaders.
    */
   bool single_vary = util_is_power_of_two_nonzero64(ctx->layout->written);
   enum uvs_group group = group_for_varying(sem.location);

   nir_def *base;
   if (group == UVS_VARYINGS && ctx->layout->relocatable)
      base = nir_imm_intN_t(b, ctx->placeholder[sem.location], 16);
   else if ((group == UVS_VARYINGS) && !single_vary)
      base = nir_load_uvs_index_agx(b, .io_semantics = sem);
   else
      base = nir_imm_intN_t(b, ctx->layout->group_offs[group], 16);

   nir_def *index = nir_iadd(b, nir_iadd_imm(b, base, component),
                             nir_imul_imm(b, nir_u2u16(b, offset), 4));

   /* LAYER is consumed only by the shared fixed-function scalar. VIEWPORT
    * additionally needs an ordinary flat varying only when the next shader
    * reads it. nir_opt_varyings marks the fixed-function-only case with
    * no_varying; emitting both stores there disagrees with Apple's native UVS
    * layout and overstates the VertexProgramVariant output size. */
   if ((sem.location != VARYING_SLOT_LAYER || ctx->layer_as_varying) &&
       (sem.location != VARYING_SLOT_VIEWPORT || !sem.no_varying))
      nir_store_uvs_agx(b, value, index);

   /* Insert clip distance sysval writes, and gather layer/viewport writes so we
    * can accumulate their system value. These are still lowered like normal to
    * write them for the varying FS input.
    */
   if (sem.location == VARYING_SLOT_LAYER) {
      assert(ctx->layer == NULL && "only written once");
      ctx->layer = value;
      ctx->after_layer_viewport = nir_after_def(index);
   } else if (sem.location == VARYING_SLOT_VIEWPORT) {
      assert(ctx->viewport == NULL && "only written once");
      ctx->viewport = value;
      ctx->after_layer_viewport = nir_after_def(index);
   } else if (sem.location == VARYING_SLOT_CLIP_DIST0 ||
              sem.location == VARYING_SLOT_CLIP_DIST1) {

      unsigned clip_base = ctx->layout->group_offs[UVS_CLIP_DIST];
      unsigned c = 4 * (sem.location - VARYING_SLOT_CLIP_DIST0) + component;

      if (c < b->shader->info.clip_distance_array_size) {
         nir_def *clip_index = nir_iadd_imm(
            b, nir_imul_imm(b, nir_u2u16(b, offset), 4), clip_base + c);

         nir_store_uvs_agx(b, value, clip_index);
      }
   }

   /* Combined clip/cull used */
   assert(sem.location != VARYING_SLOT_CULL_DIST0);
   assert(sem.location != VARYING_SLOT_CULL_DIST1);

   return true;
}

static void
write_layer_viewport_sysval(struct ctx *ctx)
{
   nir_builder b = nir_builder_at(ctx->after_layer_viewport);

   nir_def *zero = nir_imm_intN_t(&b, 0, 16);
   nir_def *layer = ctx->layer ? nir_u2u16(&b, ctx->layer) : zero;
   nir_def *viewport = ctx->viewport ? nir_u2u16(&b, ctx->viewport) : zero;

   if (ctx->layer && ctx->layer_zero_root_offset >= 0) {
      nir_def *addr = nir_iadd_imm(&b, nir_load_root_agx(&b),
                                   ctx->layer_zero_root_offset);
      nir_def *flag =
         nir_load_global_constant(&b, 1, 16, addr, .align_mul = 2);
      layer = nir_bcsel(&b, nir_ine_imm(&b, flag, 0), zero, layer);
   }

   /* render_target_array_index is the low half and viewport_array_index the
    * high half on every generation, G17P included. */
   nir_def *packed = nir_pack_32_2x16_split(&b, layer, viewport);

   nir_store_uvs_agx(
      &b, packed,
      nir_imm_int(&b, ctx->layout->group_offs[UVS_LAYER_VIEWPORT]));
}

/* A user varying stored at a computed offset (an array indexed at run time)
 * has no constant UVS index to relocate: store it at each constant offset of
 * the array under a test of the computed one instead. */
static bool
lower_indirect_varying(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       group_for_varying(nir_intrinsic_io_semantics(intr).location) !=
          UVS_VARYINGS ||
       nir_src_is_const(intr->src[1]))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_def *offset = intr->src[1].ssa;
   nir_def *zero = nir_imm_int(b, 0);
   for (unsigned i = 0; i < sem.num_slots; ++i) {
      nir_io_semantics slot = sem;
      slot.location += i;
      slot.num_slots = 1;
      nir_push_if(b, nir_ieq_imm(b, offset, i));
      nir_intrinsic_instr *store =
         nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
      nir_intrinsic_set_io_semantics(store, slot);
      nir_builder_instr_insert(b, &store->instr);
      nir_src_rewrite(&store->src[1], zero);
      nir_pop_if(b, NULL);
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
gather_components(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct agx_unlinked_uvs_layout *layout = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned component = nir_intrinsic_component(intr);

   /* VIEWPORT remains a fixed-function output when it is not consumed by the
    * next shader. In that case only the packed layer/viewport scalar below is
    * allocated; it is not also a user varying. */
   if (sem.location == VARYING_SLOT_VIEWPORT && sem.no_varying)
      return false;

   if (nir_src_is_const(intr->src[1])) {
      unsigned loc = sem.location + nir_src_as_uint(intr->src[1]);
      layout->components[loc] = MAX2(layout->components[loc], component + 1);
   } else {
      for (unsigned i = 0; i < sem.num_slots; ++i) {
         layout->components[sem.location + i] = 4;
      }
   }

   return false;
}

bool
agx_nir_lower_default_layer(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (s->info.outputs_written & VARYING_BIT_LAYER)
      return nir_no_progress(impl);

   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                    .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_LAYER,
                                     .num_slots = 1},
                    .write_mask = 1);
   s->info.outputs_written |= VARYING_BIT_LAYER;
   return nir_progress(true, impl, nir_metadata_control_flow);
}

bool
agx_nir_lower_uvs(nir_shader *s, struct agx_unlinked_uvs_layout *layout,
                  bool layer_as_varying, bool relocatable,
                  int layer_zero_root_offset)
{
   bool progress = false;

   /* Scalarize up front so we can ignore vectors later */
   NIR_PASS(progress, s, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
            NULL);

   /* A shader compiled without its fragment shader would otherwise place its
    * varyings by the fragment shader's interpolation qualifiers, with UVS
    * indices loaded at run time. With relocatable, write each one at a
    * constant placeholder index instead, which linking it with the fragment
    * shader rewrites (agx_uvs_relocation_map()).
    */
   relocatable = relocatable && !s->info.known_interpolation_qualifiers;
   if (relocatable)
      NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_indirect_varying,
               nir_metadata_none, NULL);

   /* Determine the unlinked UVS layout */
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, gather_components,
            nir_metadata_control_flow, layout);

   unsigned sizes[UVS_NUM_GROUP] = {
      [UVS_POSITION] = 4,
      [UVS_PSIZ] = !!(s->info.outputs_written & VARYING_BIT_PSIZ),
      [UVS_LAYER_VIEWPORT] = !!(s->info.outputs_written &
                                (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT)),
      [UVS_CLIP_DIST] = s->info.clip_distance_array_size,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(layout->components); ++i) {
      if (i != VARYING_SLOT_POS && i != VARYING_SLOT_PSIZ &&
          (i != VARYING_SLOT_LAYER || layer_as_varying) &&
          layout->components[i]) {

         layout->written |= BITFIELD64_BIT(i);
         sizes[UVS_VARYINGS] += layout->components[i];
      }
   }

   struct ctx ctx = {
      .layout = layout,
      .layer_as_varying = layer_as_varying,
      .layer_zero_root_offset = layer_zero_root_offset,
   };
   layout->relocatable = relocatable;

   unsigned offs = 0;
   for (enum uvs_group g = 0; g < UVS_NUM_GROUP; ++g) {
      layout->group_offs[g] = offs;
      offs += sizes[g];
   }

   layout->size = offs;
   layout->user_size = sizes[UVS_VARYINGS];

   if (layout->relocatable) {
      struct agx_varyings_vs placeholder;
      agx_assign_uvs(&placeholder, layout, 0, 0);

      u_foreach_bit64(loc, layout->written) {
         assert(placeholder.slots[loc] + layout->components[loc] <=
                AGX3_UVS_RELOC_MAP_SIZE);
         ctx.placeholder[loc] = placeholder.slots[loc];
      }
   }

   /* Now lower in terms of the unlinked layout */
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower,
            nir_metadata_control_flow, &ctx);

   if (ctx.layer || ctx.viewport) {
      write_layer_viewport_sysval(&ctx);
   }

   /* Finally, pack what we can. It's much cheaper to do this at compile-time
    * than draw-time.
    */
   agx_pack(&layout->osel, OUTPUT_SELECT, cfg) {
      cfg.point_size = sizes[UVS_PSIZ];
      cfg.viewport_target = sizes[UVS_LAYER_VIEWPORT];
      cfg.render_target = cfg.viewport_target;

      cfg.clip_distance_plane_0 = sizes[UVS_CLIP_DIST] > 0;
      cfg.clip_distance_plane_1 = sizes[UVS_CLIP_DIST] > 1;
      cfg.clip_distance_plane_2 = sizes[UVS_CLIP_DIST] > 2;
      cfg.clip_distance_plane_3 = sizes[UVS_CLIP_DIST] > 3;
      cfg.clip_distance_plane_4 = sizes[UVS_CLIP_DIST] > 4;
      cfg.clip_distance_plane_5 = sizes[UVS_CLIP_DIST] > 5;
      cfg.clip_distance_plane_6 = sizes[UVS_CLIP_DIST] > 6;
      cfg.clip_distance_plane_7 = sizes[UVS_CLIP_DIST] > 7;
   }

   agx_pack(&layout->vdm, VDM_STATE_VERTEX_OUTPUTS, cfg) {
      cfg.output_count_1 = offs;
      cfg.output_count_2 = offs;
   }

   /* If we know the interpolation qualifiers in the vertex shader, we can
    * inline UVS indices to eliminate some indirection.
    */
   if (s->info.known_interpolation_qualifiers) {
      struct agx_varyings_vs v;
      uint64_t flat = ~(s->info.linear_varyings | s->info.perspective_varyings);

      agx_assign_uvs(&v, layout, flat, s->info.linear_varyings);

      nir_shader_intrinsics_pass(s, inline_uvs_index, nir_metadata_control_flow,
                                 &v);
   }

   return progress;
}

void
agx_assign_uvs(struct agx_varyings_vs *varyings,
               const struct agx_unlinked_uvs_layout *layout,
               uint64_t flat_mask, uint64_t linear_mask)
{
   *varyings = (struct agx_varyings_vs){0};

   /* Layer/viewport are always flat-shaded. Of the other special varyings
    * only the legacy GL ones can be: glShadeModel(GL_FLAT) makes the
    * colours flat (the state tracker lowers it to flat COL0/COL1 inputs),
    * and compatibility GLSL may redeclare them, the back colours, fog or the
    * texture coordinates flat. Counting such a slot as smooth gives a VARYING
    * COUNTS word that disagrees with its flat CF binding, which the HAL300
    * single-mode varying record cannot express. Position, point size and
    * clip/cull distances are never flat. */
   flat_mask &= BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32) |
                BITFIELD64_RANGE(VARYING_SLOT_COL0,
                                 VARYING_SLOT_TEX7 - VARYING_SLOT_COL0 + 1) |
                VARYING_BIT_BFC0 | VARYING_BIT_BFC1;
   flat_mask |= VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT;
   if (layout->mesh_primitive_page)
      flat_mask |= BITFIELD64_BIT(AGX_MESH_PRIMITIVE_PAGE_SLOT);

   /* The internal cull distance slots are always linearly-interpolated */
   linear_mask |= BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2);

   assert(!(flat_mask & linear_mask & layout->written));

   /* TODO: Link FP16 varyings */
   unsigned num_32_smooth = 0, num_32_flat = 0, num_32_linear = 0;
   struct {
      uint32_t *num;
      uint64_t mask;
   } parts[] = {
      {&num_32_smooth, ~flat_mask & ~linear_mask},
      {&num_32_flat, flat_mask},
      {&num_32_linear, linear_mask},
   };

   unsigned base = layout->group_offs[UVS_VARYINGS];

   for (unsigned p = 0; p < ARRAY_SIZE(parts); ++p) {
      u_foreach_bit64(loc, parts[p].mask & layout->written) {
         assert(loc < ARRAY_SIZE(varyings->slots));
         varyings->slots[loc] = base;

         base += layout->components[loc];
         (*parts[p].num) += layout->components[loc];
      }
   }

   agx_pack(&varyings->counts_32, VARYING_COUNTS, cfg) {
      cfg.smooth = num_32_smooth;
      cfg.flat = num_32_flat;
      cfg.linear = num_32_linear;
   }

   agx_pack(&varyings->counts_16, VARYING_COUNTS, cfg) {
      cfg.smooth = 0;
      cfg.flat = 0;
      cfg.linear = 0;
   }
}

static_assert(AGX3_UVS_RELOC_MAP_SIZE == 256, "agx_uvs.h");

void
agx_uvs_relocation_map(const struct agx_unlinked_uvs_layout *layout,
                       uint64_t flat_mask, uint64_t linear_mask,
                       uint8_t map[AGX3_UVS_RELOC_MAP_SIZE])
{
   for (unsigned i = 0; i < AGX3_UVS_RELOC_MAP_SIZE; ++i)
      map[i] = i;

   if (!layout->relocatable)
      return;

   /* The placeholders are the indices of the layout linked with no flat or
    * linear inputs (agx_nir_lower_uvs()). */
   struct agx_varyings_vs placeholder, linked;
   agx_assign_uvs(&placeholder, layout, 0, 0);
   agx_assign_uvs(&linked, layout, flat_mask, linear_mask);

   u_foreach_bit64(loc, layout->written) {
      for (unsigned c = 0; c < layout->components[loc]; ++c) {
         assert(placeholder.slots[loc] + c < AGX3_UVS_RELOC_MAP_SIZE);
         assert(linked.slots[loc] + c < AGX3_UVS_RELOC_MAP_SIZE);
         map[placeholder.slots[loc] + c] = linked.slots[loc] + c;
      }
   }
}

static inline enum agx_shade_model
translate_flat_shade_model(unsigned provoking_vertex)
{
   static_assert(AGX_SHADE_MODEL_FLAT_VERTEX_0 == 0, "hw");
   static_assert(AGX_SHADE_MODEL_FLAT_VERTEX_2 == 2, "hw");

   assert(provoking_vertex <= 2);

   if (provoking_vertex == 1)
      return AGX_SHADE_MODEL_FLAT_VERTEX_1;
   else
      return (enum agx_shade_model)provoking_vertex;
}

void
agx_link_varyings_vs_fs(void *out, struct agx_varyings_vs *vs,
                        unsigned nr_user_indices, struct agx_varyings_fs *fs,
                        unsigned provoking_vertex, uint8_t sprite_coord_enable,
                        bool *generate_primitive_id)
{
   assert(fs->nr_bindings > 0);

   *generate_primitive_id = false;

   struct agx_cf_binding_header_packed *header = out;
   struct agx_cf_binding_packed *bindings = (void *)(header + 1);

   unsigned user_base = 1 + (fs->reads_z ? 1 : 0);
   unsigned nr_slots = user_base + nr_user_indices;

   agx_pack(header, CF_BINDING_HEADER, cfg) {
      cfg.number_of_32_bit_slots = nr_slots;
      cfg.number_of_coefficient_registers = fs->nr_cf;
   }

   for (unsigned i = 0; i < fs->nr_bindings; ++i) {
      struct agx_cf_binding b = fs->bindings[i];

      agx_pack(bindings + i, CF_BINDING, cfg) {
         cfg.base_coefficient_register = b.cf_base;
         cfg.components = b.count;

         if (b.smooth) {
            cfg.shade_model = b.perspective ? AGX_SHADE_MODEL_PERSPECTIVE
                                            : AGX_SHADE_MODEL_LINEAR;
         } else {
            cfg.shade_model = translate_flat_shade_model(provoking_vertex);
         }

         if (b.slot == VARYING_SLOT_PNTC ||
             (b.slot >= VARYING_SLOT_TEX0 && b.slot <= VARYING_SLOT_TEX7 &&
              (sprite_coord_enable &
               BITFIELD_BIT(b.slot - VARYING_SLOT_TEX0)))) {

            assert(b.offset == 0);
            cfg.source = AGX_COEFFICIENT_SOURCE_POINT_COORD;
         } else if (b.slot == VARYING_SLOT_PRIMITIVE_ID &&
                    !vs->slots[VARYING_SLOT_PRIMITIVE_ID]) {
            cfg.source = AGX_COEFFICIENT_SOURCE_PRIMITIVE_ID;
            *generate_primitive_id = true;
         } else if (b.slot == VARYING_SLOT_POS) {
            assert(b.offset >= 2 && "gl_Position.xy are not varyings");
            assert(fs->reads_z || b.offset != 2);

            if (b.offset == 2) {
               cfg.source = AGX_COEFFICIENT_SOURCE_FRAGCOORD_Z;
               cfg.base_slot = 1;
            } else {
               assert(!b.perspective && "W must not be perspective divided");
            }
         } else {
            unsigned vs_index = vs->slots[b.slot];
            assert(b.offset < 4);

            /* Varyings not written by vertex shader are undefined but we can't
             * crash */
            if (vs_index) {
               assert(vs_index >= 4 &&
                      "gl_Position should have been the first 4 slots");

               cfg.base_slot = user_base + (vs_index - 4) + b.offset;
            }
         }

         assert(cfg.base_coefficient_register + cfg.components <= fs->nr_cf &&
                "overflowed coefficient registers");
      }
   }
}
