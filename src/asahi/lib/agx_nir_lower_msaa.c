/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2021 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_tilebuffer.h"
#include "layout.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"

struct lower_per_sample_ctx {
   bool force_early_pending;
};

/* Explicit sample interpolation is per-pixel, but its single-sample clamp
 * still depends on rasterizationSamples. It needs native specialization when
 * compiled without that state, just as a sample-shaded fragment does. */
bool
agx_nir_needs_sample_count(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_load_barycentric_at_sample ||
                op == nir_intrinsic_load_samples_log2_agx)
               return true;
         }
      }
   }
   return false;
}

static bool
specialize_sample_count(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_samples_log2_agx)
      return false;

   unsigned samples = *(const unsigned *)data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_imm_intN_t(b, util_logbase2(samples), intr->def.bit_size));
   return true;
}

bool
agx_nir_specialize_sample_count(nir_shader *shader, unsigned samples)
{
   assert(samples == 1 || samples == 2 || samples == 4 || samples == 8);
   return nir_shader_intrinsics_pass(shader, specialize_sample_count,
                                     nir_metadata_control_flow, &samples);
}

static bool
apply_api_mask_to_force_early(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   bool *done = data;
   if (*done || intr->intrinsic != nir_intrinsic_sample_mask_agx)
      return false;

   /* Fold the API mask into the one forced-early coverage update. A separate
    * prolog MOVMSK followed by the main shader's forced-early MOVMSK loses the
    * surviving samples on G17P; Apple's native program performs one combined
    * update before reading the post-test coverage for fragment side effects.
    * M5 also suppresses the separate prolog mask for forced-early shaders, so
    * its per-pixel program must include the API mask in this update. */
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *api = nir_load_api_sample_mask_agx(b);
   nir_src_rewrite(&intr->src[1],
                   nir_iand(b, intr->src[1].ssa,
                            nir_u2uN(b, api, nir_src_bit_size(intr->src[1]))));
   *done = true;
   return true;
}

bool
agx_nir_apply_api_mask_to_force_early(nir_shader *shader, enum agx_chip chip)
{
   const bool supported = agx_chip_native_agx3(chip) ||
      (agx_chip_hal300_path(chip) && !shader->info.fs.uses_sample_shading);
   if (!supported || !shader->info.fs.early_fragment_tests)
      return false;

   bool done = false;
   return nir_shader_intrinsics_pass(shader, apply_api_mask_to_force_early,
                                     nir_metadata_control_flow, &done);
}

static bool
lower_to_per_sample(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_per_sample_ctx *ctx = data;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_sample_id: {
      nir_def *mask = nir_u2u32(b, nir_load_active_samples_agx(b));
      nir_def *bit = nir_ufind_msb(b, mask);
      nir_def_replace(&intr->def, nir_u2uN(b, bit, intr->def.bit_size));
      return true;
   }

   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_store_zs_agx:
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_sample_mask_agx: {
      /* Explicit early tests are a per-pixel operation even when the shader
       * body is sample shaded.  Keep the all/all update inserted at the
       * function entry intact; the G17P static unroller hoists it ahead of
       * all sample copies below. */
      if (intr->intrinsic == nir_intrinsic_sample_mask_agx &&
          ctx->force_early_pending) {
         ctx->force_early_pending = false;
         return false;
      }

      /* Fragment I/O inside the loop should only affect active samples. */
      unsigned mask_index =
         (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) ? 1 : 0;

      nir_def *mask = intr->src[mask_index].ssa;
      nir_def *id_mask = nir_load_active_samples_agx(b);
      nir_def *converted = nir_u2uN(b, id_mask, mask->bit_size);

      nir_src_rewrite(&intr->src[mask_index], nir_iand(b, mask, converted));
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_to_per_sample_agx2(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_sample_id: {
      nir_def *mask = nir_u2u32(b, nir_load_active_samples_agx(b));
      nir_def *bit = nir_ufind_msb(b, mask);
      nir_def_replace(&intr->def, nir_u2uN(b, bit, intr->def.bit_size));
      return true;
   }

   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_store_zs_agx:
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_sample_mask_agx: {
      /* Fragment I/O inside the loop should only affect active samples. */
      unsigned mask_index =
         (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) ? 1 : 0;

      nir_def *mask = intr->src[mask_index].ssa;
      nir_def *id_mask = nir_load_active_samples_agx(b);
      nir_def *converted = nir_u2uN(b, id_mask, mask->bit_size);

      nir_src_rewrite(&intr->src[mask_index], nir_iand(b, mask, converted));
      return true;
   }

   default:
      return false;
   }
}

bool
agx_nir_lower_to_per_sample_agx2(nir_shader *shader)
{
   return nir_shader_intrinsics_pass(shader, lower_to_per_sample_agx2,
                                     nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_to_per_sample(nir_shader *shader)
{
   struct lower_per_sample_ctx ctx = {
      .force_early_pending = shader->info.fs.early_fragment_tests,
   };
   return nir_shader_intrinsics_pass(shader, lower_to_per_sample,
                                     nir_metadata_control_flow, &ctx);
}

static bool
lower_active_samples(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(&intr->def, data);
   return true;
}

/*
 * In a monolithic pixel shader, we wrap the fragment shader in a loop over
 * each sample, and then let optimizations (like loop unrolling) go to town.
 * This lowering is not compatible with fragment epilogues, which require
 * something similar at the binary level since the NIR is long gone by then.
 */
static bool
agx_nir_wrap_per_sample_loop(nir_shader *shader, uint8_t nr_samples)
{
   assert(nr_samples > 1);

   /* Get the original function */
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_cf_list list;
   nir_cf_extract(&list, nir_before_impl(impl), nir_after_impl(impl));

   /* Create a builder for the wrapped function */
   nir_builder b = nir_builder_at(nir_after_block(nir_start_block(impl)));

   nir_variable *i =
      nir_local_variable_create(impl, glsl_uintN_t_type(16), NULL);
   nir_store_var(&b, i, nir_imm_intN_t(&b, 1, 16), ~0);
   nir_def *bit = NULL;
   nir_def *end_bit = nir_imm_intN_t(&b, 1 << nr_samples, 16);

   /* Create a loop in the wrapped function */
   nir_loop *loop = nir_push_loop(&b);
   {
      bit = nir_load_var(&b, i);
      nir_break_if(&b, nir_uge(&b, bit, end_bit));

      b.cursor = nir_cf_reinsert(&list, b.cursor);
      nir_store_var(&b, i, nir_ishl_imm(&b, bit, 1), ~0);
   }
   nir_pop_loop(&b, loop);

   /* We've mucked about with control flow */
   nir_progress(true, impl, nir_metadata_none);

   /* Use the loop variable for the active sampple mask each iteration */
   nir_shader_intrinsics_pass(shader, lower_active_samples,
                              nir_metadata_control_flow, bit);
   return true;
}

struct g17p_sample_copy {
   nir_def *active_samples;
   nir_def *samples_log2;
   unsigned colour_offset;
   /* Export base of this sample's depth, or 0 to keep the ABI's. */
   unsigned zs_base;
};

static void
specialize_g17p_sample_copy(nir_cf_list *copy,
                            const struct g17p_sample_copy *sample)
{
   foreach_list_typed(nir_cf_node, node, node, &copy->list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_active_samples_agx:
               nir_def_rewrite_uses(&intr->def, sample->active_samples);
               nir_instr_remove(instr);
               break;

            case nir_intrinsic_load_samples_log2_agx:
               nir_def_rewrite_uses(&intr->def, sample->samples_log2);
               nir_instr_remove(instr);
               break;

            case nir_intrinsic_export_agx:
            case nir_intrinsic_load_exported_agx: {
               const unsigned base = nir_intrinsic_base(intr);
               if (base >= AGX_ABI_FOUT_COLOUR(0)) {
                  nir_intrinsic_set_base(intr, base + sample->colour_offset);
               } else if (sample->zs_base && (base == AGX_ABI_FOUT_Z ||
                                              base == AGX_ABI_FOUT_S)) {
                  nir_intrinsic_set_base(
                     intr, sample->zs_base + base - AGX_ABI_FOUT_Z);
               }
               break;
            }

            default:
               break;
            }
         }
      }
   }
}

/* Make each output an SSA value at the end of the sample body before cloning
 * it. Conditional depth/stencil writes need phis here: moving their exports
 * directly to the end would move uses outside the defining branch.
 */
void
agx_nir_lower_sample_exports_to_end(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_block *end = nir_cursor_current_block(nir_after_impl(impl));
   bool needs_lowering = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_export_agx &&
             block != end)
            needs_lowering = true;
      }
   }
   if (!needs_lowering)
      return;

   struct sample_export {
      unsigned base;
      nir_variable *var;
   };
   struct util_dynarray outputs;
   util_dynarray_init(&outputs, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_export_agx)
            continue;

         unsigned base = nir_intrinsic_base(intr);
         nir_def *value = intr->src[0].ssa;
         const struct glsl_type *type =
            glsl_vector_type(glsl_get_base_type(glsl_uintN_t_type(value->bit_size)),
                             value->num_components);
         nir_variable *var = NULL;
         util_dynarray_foreach(&outputs, struct sample_export, output) {
            if (output->base == base) {
               var = output->var;
               assert(var->type == type);
               break;
            }
         }
         if (!var) {
            var = nir_local_variable_create(impl, type, "sample_export");
            util_dynarray_append(&outputs, ((struct sample_export){base, var}));
         }
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_store_var(&b, var, value, BITFIELD_MASK(value->num_components));
         nir_instr_remove(instr);
      }
   }
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   util_dynarray_foreach(&outputs, struct sample_export, output)
      nir_export_agx(&b, nir_load_var(&b, output->var), .base = output->base);
   util_dynarray_fini(&outputs);
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_lower_vars_to_ssa(shader);
}

/*
 * G17P has no external sample loop around linked fragment parts. Apple's
 * applegpu_g17p fragments that read [[sample_id]] instead repeat the shader
 * body once per sample, with the sample folded in, and store each sample
 * separately. Do the same for one shader part already lowered to per-sample
 * form (agx_nir_lower_to_per_sample): clone the body per sample with that
 * sample active.
 *
 * Colours cross from the main fragment to its epilog in export registers, so
 * each sample gets its own: sample i of every location follows all locations
 * of the samples before it. Depth and stencil of every sample after the first
 * follow all colours, four halves per sample. Both parts must be unrolled with
 * the same count and the same colour layout.
 */
bool
agx_nir_unroll_g17p_samples_layout(nir_shader *shader, unsigned nr_samples,
                                   const struct agx_g17p_colour_layout *layout)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   assert(nr_samples == 2 || nr_samples == 4 || nr_samples == 8);

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   /* The forced early Z/S update belongs before the complete unrolled sample
    * body.  Cloning it once per sample runs the same tests repeatedly, which
    * violates the sample-mask contract and kills the later invocations. */
   bool force_early = false;
   if (shader->info.fs.early_fragment_tests) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
               nir_instr_remove(instr);
               force_early = true;
               break;
            }
         }
         if (force_early)
            break;
      }
   }

   agx_nir_lower_sample_exports_to_end(shader);

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   if (force_early)
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16),
                          nir_imm_intN_t(&b, 0xff, 16));

   nir_def *samples_log2 =
      nir_imm_intN_t(&b, util_logbase2(nr_samples), 16);

   for (unsigned i = 0; i < nr_samples; ++i) {
      const struct g17p_sample_copy sample = {
         .active_samples = nir_imm_intN_t(&b, BITFIELD_BIT(i), 16),
         .samples_log2 = samples_log2,
         .colour_offset = i * layout->stride,
         .zs_base =
            i ? agx_g17p_sample_zs_base_layout(layout, nr_samples, i) : 0,
      };

      nir_cf_list copy;
      nir_cf_list_clone(&copy, &body, &impl->cf_node, NULL);
      specialize_g17p_sample_copy(&copy, &sample);
      nir_cf_reinsert(&copy, nir_after_impl(impl));
   }

   nir_cf_delete(&body);

   /* The backend publishes exports after the block that is last when it
    * translates them, so a copy with control flow would leave its exports
    * ahead of the later copies' blocks. Every export's value dominates the
    * end of the function, so gather them all there. */
   struct util_dynarray exports;
   util_dynarray_init(&exports, NULL);
   nir_def *write_samples[8] = {NULL};
   unsigned write_sample_count = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_export_agx) {
            nir_intrinsic_instr *export = intr;

            /* A forced-early shader exports one software write mask.  The
             * per-sample unroll clones that export along with the body, but
             * the fragment ABI has only one FOUT_WRITE_SAMPLES half.  Fold
             * the clones into one mask: each invocation owns only its active
             * bit and leaves every other sample untouched. */
            if (nir_intrinsic_base(export) ==
                AGX_ABI_FOUT_WRITE_SAMPLES) {
               assert(write_sample_count < nr_samples);
               write_samples[write_sample_count++] = export->src[0].ssa;
               nir_instr_remove(instr);
            } else {
               util_dynarray_append(&exports, instr);
            }
         }
      }
   }

   if (write_sample_count) {
      assert(write_sample_count == nr_samples);
      b = nir_builder_at(nir_after_impl(impl));
      nir_def *combined = nir_imm_intN_t(&b, 0xff, 16);
      for (unsigned i = 0; i < nr_samples; ++i) {
         const unsigned bit = BITFIELD_BIT(i);
         nir_def *owned = nir_iand_imm(&b, write_samples[i], bit);
         nir_def *keep_others = nir_ior_imm(&b, owned, 0xff ^ bit);
         combined = nir_iand(&b, combined, keep_others);
      }
      nir_export_agx(&b, combined, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   }

   util_dynarray_foreach(&exports, nir_instr *, instr)
      nir_instr_move(nir_after_impl(impl), *instr);
   util_dynarray_fini(&exports);

   return nir_progress(true, impl, nir_metadata_none);
}

bool
agx_nir_unroll_g17p_samples(nir_shader *shader, unsigned nr_samples,
                            uint8_t loc_written)
{
   struct agx_g17p_colour_layout layout;
   agx_g17p_colour_layout_init(&layout, loc_written, 0, 0, false);

   return agx_nir_unroll_g17p_samples_layout(shader, nr_samples, &layout);
}

#define G17P_EQUIV_ALL_SAMPLES  0xFFu
#define G17P_EQUIV_MAX_PROBES   8u
#define G17P_EQUIV_REUSE_DWORDS 8u
#define G17P_EQUIV_MAX_DWORDS   16u

struct g17p_equiv_probe {
   /* The shading body's tilebuffer read, as specialize_g17p_equiv_copy()
    * matches it. */
   unsigned base;
   unsigned target;
   enum pipe_format format;
   uint8_t num_components;
   uint8_t bit_size;

   /* How the probe reads the same bytes to compare the samples. */
   enum pipe_format raw_format;
   uint8_t raw_components;
   uint8_t raw_bit_size;
};

/*
 * The probe only decides whether every sample of a pixel holds the same
 * bytes, and equal bytes decode to equal values in every format. An AGX3
 * tile load names a lane width, not a format (agx3_tile_format_width()):
 * 8-bit and 16-bit normalized lanes, sRGB bytes and the packed 32-bit
 * formats come back as raw lanes that the shader decodes afterwards
 * (normflt, the RGB10A2 unpack). Comparing decoded channels costs every
 * sample of every probe a decode, plus a conversion when a wide probe
 * narrows them to 16 bits.
 *
 * Compare the undecoded lanes instead. Four or two 8-bit lanes are read as
 * one 32-bit or 16-bit word at the same byte offset (pixel offsets are byte
 * offsets aligned to the lane width, and the lanes arrive in byte order
 * either way, as R11G11B10F's raw 32-bit word does). Four or two 16-bit
 * lanes at a 4-byte-aligned offset are read as two or one 32-bit words so
 * the probe compares half as many channels and folds directly into the
 * 32-bit difference accumulator; unaligned or odd-component 16-bit lanes
 * keep 16-bit integer reads, and a packed 32-bit format keeps its word.
 * Other shapes, one or three 8-bit lanes, keep the body's own read.
 *
 * Bytes that differ but decode alike (signed normalized -128 and -127, NaN
 * payloads) take the per-sample path, which is always correct.
 */
static void
g17p_equiv_set_raw_form(struct g17p_equiv_probe *p)
{
   p->raw_format = p->format;
   p->raw_components = p->num_components;
   p->raw_bit_size = p->bit_size;

   switch ((enum ail_isa_format)p->format) {
   case AIL_ISA_FORMAT_I8:
   case AIL_ISA_FORMAT_U8NORM:
   case AIL_ISA_FORMAT_S8NORM:
   case AIL_ISA_FORMAT_SRGBA8:
      if (p->num_components == 4 && (p->base % 4) == 0) {
         p->raw_format = PIPE_FORMAT_R32_UINT;
         p->raw_components = 1;
         p->raw_bit_size = 32;
      } else if (p->num_components == 2 && (p->base % 2) == 0) {
         p->raw_format = PIPE_FORMAT_R16_UINT;
         p->raw_components = 1;
         p->raw_bit_size = 16;
      }
      break;

   case AIL_ISA_FORMAT_I16:
   case AIL_ISA_FORMAT_F16:
   case AIL_ISA_FORMAT_U16NORM:
   case AIL_ISA_FORMAT_S16NORM:
      if (p->num_components == 4 && (p->base % 4) == 0) {
         p->raw_format = PIPE_FORMAT_R32_UINT;
         p->raw_components = 2;
         p->raw_bit_size = 32;
      } else if (p->num_components == 2 && (p->base % 4) == 0) {
         p->raw_format = PIPE_FORMAT_R32_UINT;
         p->raw_components = 1;
         p->raw_bit_size = 32;
      } else {
         p->raw_format = PIPE_FORMAT_R16_UINT;
         p->raw_bit_size = 16;
      }
      break;

   case AIL_ISA_FORMAT_RGB10A2:
   case AIL_ISA_FORMAT_RG11B10F:
   case AIL_ISA_FORMAT_RGB9E5:
      p->raw_format = PIPE_FORMAT_R32_UINT;
      p->raw_components = 1;
      p->raw_bit_size = 32;
      break;

   default:
      break;
   }
}

/* Two or four 32-bit words that one tile load reads for every probe it
 * covers. */
struct g17p_equiv_chunk {
   unsigned base;
   unsigned target;
   unsigned words;
};

static bool
g17p_equiv_probe_is_words(const struct g17p_equiv_probe *p)
{
   return p->raw_format == PIPE_FORMAT_R32_UINT && p->raw_bit_size == 32 &&
          (p->base % 4) == 0;
}

static bool
g17p_equiv_chunk_words_ok(unsigned words)
{
   /* The widths of the RG32 and RGBA32 targets' own reads. */
   return words == 2 || words == 4;
}

/*
 * A tile load names a byte offset and a lane width, not a render target
 * (agx3_emit_tile()), and every sample of a pixel holds its targets at the
 * same offsets. So probes whose raw words lie next to each other, usually of
 * different targets, can be compared with one load of two or four 32-bit
 * words per sample instead of one load per target. A merged read starts at a
 * probe's own offset, so at a target, and covers whole probes; a probe inside
 * another (the same target read in another shape) shares its read. Choose
 * the split of each run of adjacent probes that needs the fewest reads; any
 * probe not in a merged read keeps its own. Returns the number of merged
 * reads and sets chunk_of[k] to the read covering probe k, or -1.
 */
static unsigned
g17p_equiv_build_chunks(const struct g17p_equiv_probe *probes,
                        unsigned num_probes,
                        struct g17p_equiv_chunk *chunks, int *chunk_of)
{
   unsigned order[G17P_EQUIV_MAX_PROBES];
   unsigned n = 0;

   for (unsigned k = 0; k < num_probes; ++k) {
      chunk_of[k] = -1;
      if (!g17p_equiv_probe_is_words(&probes[k]))
         continue;

      /* Ascending offset, the wider read first at equal offsets. */
      unsigned i = n++;
      for (; i > 0; --i) {
         const struct g17p_equiv_probe *prev = &probes[order[i - 1]];
         if (prev->base < probes[k].base ||
             (prev->base == probes[k].base &&
              prev->raw_components >= probes[k].raw_components))
            break;
         order[i] = order[i - 1];
      }
      order[i] = k;
   }

   /* Units: probes not inside the previous unit. A unit that partly overlaps
    * the previous one is never merged. */
   unsigned unit_probe[G17P_EQUIV_MAX_PROBES];
   unsigned unit_of[G17P_EQUIV_MAX_PROBES];
   bool unit_alone[G17P_EQUIV_MAX_PROBES];
   unsigned num_units = 0;
   for (unsigned i = 0; i < n; ++i) {
      const struct g17p_equiv_probe *p = &probes[order[i]];
      if (num_units) {
         const struct g17p_equiv_probe *u = &probes[unit_probe[num_units - 1]];
         const unsigned end_u = u->base + 4 * u->raw_components;
         if (p->base + 4 * p->raw_components <= end_u) {
            unit_of[i] = num_units - 1;
            continue;
         }
         unit_alone[num_units] = p->base < end_u;
         if (unit_alone[num_units])
            unit_alone[num_units - 1] = true;
      } else {
         unit_alone[0] = false;
      }
      unit_probe[num_units] = order[i];
      unit_of[i] = num_units++;
   }

   /* Fewest reads covering units [u, num_units), by merged read or alone. */
   unsigned cost[G17P_EQUIV_MAX_PROBES + 1];
   unsigned last[G17P_EQUIV_MAX_PROBES];
   cost[num_units] = 0;
   for (int u = (int)num_units - 1; u >= 0; --u) {
      cost[u] = 1 + cost[u + 1];
      last[u] = u;
      unsigned words = probes[unit_probe[u]].raw_components;
      for (unsigned v = u + 1; v < num_units && !unit_alone[u]; ++v) {
         const struct g17p_equiv_probe *a = &probes[unit_probe[v - 1]];
         const struct g17p_equiv_probe *b = &probes[unit_probe[v]];
         if (unit_alone[v] || b->base != a->base + 4 * a->raw_components)
            break;
         words += b->raw_components;
         if (words > 4)
            break;
         /* At equal cost, prefer the longest read from the lowest offset. */
         if (g17p_equiv_chunk_words_ok(words) && 1 + cost[v + 1] <= cost[u]) {
            cost[u] = 1 + cost[v + 1];
            last[u] = v;
         }
      }
   }

   unsigned num_chunks = 0;
   int unit_chunk[G17P_EQUIV_MAX_PROBES];
   for (unsigned u = 0; u < num_units;) {
      const unsigned v = last[u];
      int c = -1;
      if (v > u) {
         const struct g17p_equiv_probe *p = &probes[unit_probe[u]];
         const struct g17p_equiv_probe *q = &probes[unit_probe[v]];
         c = num_chunks++;
         chunks[c] = (struct g17p_equiv_chunk){
            .base = p->base,
            .target = p->target,
            .words = (q->base - p->base) / 4 + q->raw_components,
         };
      }
      for (unsigned w = u; w <= v; ++w)
         unit_chunk[w] = c;
      u = v + 1;
   }

   for (unsigned i = 0; i < n; ++i)
      chunk_of[order[i]] = unit_chunk[unit_of[i]];

   return num_chunks;
}

static bool
g17p_equiv_probe_is_body_read(const struct g17p_equiv_probe *p)
{
   return p->raw_format == p->format &&
          p->raw_components == p->num_components &&
          p->raw_bit_size == p->bit_size;
}

/* A 32-bit raw probe of two or four 16-bit integer or float lanes holds the
 * exact bit pattern of sample 0's body read in packed 32-bit words, which
 * unpack into 16-bit half-registers with zero instructions (agx_subdivide_to)
 * instead of reloading the tilebuffer on the fast path. */
static bool
g17p_equiv_probe_can_unpack_v0(const struct g17p_equiv_probe *p)
{
   return p->raw_format == PIPE_FORMAT_R32_UINT && p->raw_bit_size == 32 &&
          p->raw_components * 2 == p->num_components &&
          (p->format == PIPE_FORMAT_R16_UINT ||
           p->format == PIPE_FORMAT_R16_FLOAT) &&
          (p->bit_size == 16 || p->bit_size == 32);
}

/* A 32-bit raw probe of four 8-bit normalized lanes holds the packed bytes a
 * tile read of the body's format returns, so sample 0's body read is decoded
 * from the probe word (agx_emit_unpack_local_pixel) instead of reloading the
 * tilebuffer on the fast path. */
static bool
g17p_equiv_probe_can_decode_v0(const struct g17p_equiv_probe *p)
{
   const enum ail_isa_format format = (enum ail_isa_format)p->format;
   return p->raw_format == PIPE_FORMAT_R32_UINT && p->raw_bit_size == 32 &&
          p->raw_components == 1 && p->num_components == 4 &&
          (format == AIL_ISA_FORMAT_U8NORM || format == AIL_ISA_FORMAT_S8NORM ||
           format == AIL_ISA_FORMAT_SRGBA8) &&
          (p->bit_size == 16 || p->bit_size == 32);
}

static bool
g17p_equiv_probe_reuses_v0(const struct g17p_equiv_probe *p)
{
   return g17p_equiv_probe_is_body_read(p) ||
          g17p_equiv_probe_can_unpack_v0(p) ||
          g17p_equiv_probe_can_decode_v0(p);
}

static bool
g17p_equiv_is_all_samples_iand(const nir_alu_instr *alu,
                               const nir_def *active_def)
{
   if (alu->op != nir_op_iand || alu->def.num_components != 1)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      nir_def *a = alu->src[i].src.ssa;
      nir_def *b = alu->src[1 - i].src.ssa;
      if (active_def && a != active_def)
         continue;
      nir_instr *ai = nir_def_instr(a);
      if (ai->type != nir_instr_type_intrinsic ||
          nir_instr_as_intrinsic(ai)->intrinsic !=
             nir_intrinsic_load_active_samples_agx)
         continue;
      if (nir_src_is_const(nir_src_for_ssa(b)) &&
          nir_src_as_uint(nir_src_for_ssa(b)) == G17P_EQUIV_ALL_SAMPLES)
         return true;
   }
   return false;
}

static bool
g17p_equiv_is_tile_mask_use(const nir_src *use)
{
   if (nir_src_is_if(use))
      return false;
   nir_instr *parent = nir_src_use_instr(use);
   if (parent->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);
   if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx)
      return use == &intr->src[0];
   if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx)
      return use == &intr->src[1];
   return false;
}

static bool
g17p_equiv_is_valid_tile_mask(const nir_def *def)
{
   const nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(instr)->intrinsic ==
          nir_intrinsic_load_active_samples_agx)
      return true;
   if (instr->type == nir_instr_type_alu)
      return g17p_equiv_is_all_samples_iand(nir_instr_as_alu(instr), NULL);
   return false;
}

static bool
g17p_equiv_active_samples_uses_valid(const nir_def *def)
{
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use))
         return false;
      if (g17p_equiv_is_tile_mask_use(use))
         continue;
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;
      const nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (!g17p_equiv_is_all_samples_iand(alu, def))
         return false;
      nir_foreach_use_including_if(alu_use, &alu->def) {
         if (!g17p_equiv_is_tile_mask_use(alu_use))
            return false;
      }
   }
   return true;
}

static int
g17p_equiv_find_probe(const struct g17p_equiv_probe *probes,
                      unsigned num_probes, const nir_intrinsic_instr *intr)
{
   const unsigned base = nir_intrinsic_base(intr);
   const unsigned target = nir_intrinsic_target(intr);
   const enum pipe_format format = nir_intrinsic_format(intr);
   for (unsigned k = 0; k < num_probes; ++k) {
      if (probes[k].base == base && probes[k].target == target &&
          probes[k].format == format &&
          probes[k].num_components == intr->def.num_components &&
          probes[k].bit_size == intr->def.bit_size)
         return (int)k;
   }
   return -1;
}

/* Intrinsics whose value depends on which sample, or which lanes of the
 * subgroup, run the body. Shading one sample for several changes them. */
static bool
g17p_equiv_sample_dependent(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_pos:
   case nir_intrinsic_load_samples_log2_agx:
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_exported_agx:
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_ballot:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_elect:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_quad_ballot_agx:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      return true;
   default:
      return false;
   }
}

static void
specialize_g17p_equiv_copy(nir_function_impl *impl, nir_cf_list *copy,
                           nir_def *load_samples, nir_def *store_samples,
                           const struct g17p_equiv_probe *probes,
                           nir_def *const *reuse_v0, unsigned num_probes)
{
   foreach_list_typed(nir_cf_node, node, node, &copy->list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_active_samples_agx) {
               nir_def_rewrite_uses(&intr->def, store_samples);
               nir_instr_remove(instr);
            } else if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx) {
               const int k =
                  reuse_v0 ? g17p_equiv_find_probe(probes, num_probes, intr)
                           : -1;
               assert(!reuse_v0 || k >= 0);
               if (k >= 0 && reuse_v0[k]) {
                  nir_def *val = reuse_v0[k];
                  const struct g17p_equiv_probe *p = &probes[k];
                  if (g17p_equiv_probe_can_decode_v0(p)) {
                     nir_builder b = nir_builder_create(impl);
                     b.cursor = nir_before_instr(instr);
                     val = nir_unpack_local_pixel_agx(
                        &b, p->num_components, p->bit_size, val,
                        .format = p->format);
                  } else if (!g17p_equiv_probe_is_body_read(p)) {
                     assert(g17p_equiv_probe_can_unpack_v0(p));
                     nir_builder b = nir_builder_create(impl);
                     b.cursor = nir_before_instr(instr);
                     nir_def *lanes[4];
                     for (unsigned i = 0; i < p->raw_components; ++i) {
                        nir_def *word = nir_channel(&b, val, i);
                        lanes[2 * i] = nir_unpack_32_2x16_split_x(&b, word);
                        lanes[2 * i + 1] = nir_unpack_32_2x16_split_y(&b, word);
                     }
                     val = nir_vec(&b, lanes, p->num_components);
                     if (p->bit_size == 32) {
                        val = p->format == PIPE_FORMAT_R16_FLOAT
                                 ? nir_f2f32(&b, val)
                                 : nir_u2u32(&b, val);
                     }
                  }
                  nir_def_rewrite_uses(&intr->def, val);
                  nir_instr_remove(instr);
               } else {
                  nir_src_rewrite(&intr->src[0], load_samples);
               }
            }
         }
      }
   }
}

static uint8_t
g17p_equiv_probe_bits(const struct g17p_equiv_probe *p, bool reuse_small)
{
   uint8_t probe_bits = p->raw_bit_size;
   if (g17p_equiv_probe_is_body_read(p) && !reuse_small && probe_bits == 32 &&
       (p->format == PIPE_FORMAT_R8_UNORM ||
        p->format == PIPE_FORMAT_R8_SNORM)) {
      probe_bits = 16;
   }
   return probe_bits;
}

static void
g17p_equiv_load_sample(nir_builder *b, const struct g17p_equiv_probe *probes,
                       unsigned num_probes,
                       const struct g17p_equiv_chunk *chunks,
                       unsigned num_chunks, const int *chunk_of,
                       bool reuse_small, nir_def *sample_mask,
                       nir_def **out_chunks, nir_def **out_probes)
{
   for (unsigned c = 0; c < num_chunks; ++c) {
      out_chunks[c] = nir_load_local_pixel_agx(
         b, chunks[c].words, 32, sample_mask, .base = chunks[c].base,
         .format = PIPE_FORMAT_R32_UINT, .target = chunks[c].target);
   }

   for (unsigned k = 0; k < num_probes; ++k) {
      if (chunk_of[k] >= 0) {
         const struct g17p_equiv_chunk *c = &chunks[chunk_of[k]];
         const unsigned first = (probes[k].base - c->base) / 4;
         out_probes[k] = nir_channels(
            b, out_chunks[chunk_of[k]],
            BITFIELD_MASK(probes[k].raw_components) << first);
      } else {
         const uint8_t probe_bits =
            g17p_equiv_probe_bits(&probes[k], reuse_small);
         out_probes[k] = nir_load_local_pixel_agx(
            b, probes[k].raw_components, probe_bits, sample_mask,
            .base = probes[k].base, .format = probes[k].raw_format,
            .target = probes[k].target);
      }
   }
}

static nir_def *
g17p_equiv_diff_sample(nir_builder *b, const struct g17p_equiv_probe *probes,
                       unsigned num_probes,
                       const struct g17p_equiv_chunk *chunks,
                       unsigned num_chunks, const int *chunk_of,
                       nir_def *const *a_chunks, nir_def *const *a_probes,
                       nir_def *const *b_chunks, nir_def *const *b_probes)
{
   nir_def *diff16 = NULL, *diff32 = NULL;

   for (unsigned c = 0; c < num_chunks; ++c) {
      nir_def *diff = nir_ixor(b, a_chunks[c], b_chunks[c]);
      for (unsigned i = 0; i < chunks[c].words; ++i) {
         nir_def *d = nir_channel(b, diff, i);
         diff32 = diff32 ? nir_ior(b, diff32, d) : d;
      }
   }

   for (unsigned k = 0; k < num_probes; ++k) {
      if (chunk_of[k] >= 0)
         continue;
      nir_def *diff = nir_ixor(b, a_probes[k], b_probes[k]);
      for (unsigned c = 0; c < probes[k].raw_components; ++c) {
         nir_def *d = nir_channel(b, diff, c);
         nir_def **acc = &diff16;
         if (d->bit_size != 16) {
            d = nir_u2u32(b, d);
            acc = &diff32;
         }
         *acc = *acc ? nir_ior(b, *acc, d) : d;
      }
   }

   if (diff16 && diff32)
      return nir_ior(b, diff32, nir_u2u32(b, diff16));
   if (diff32)
      return diff32;
   if (diff16)
      return nir_u2u32(b, diff16);
   return nir_imm_int(b, 0);
}

static void
g17p_equiv_inner_dedup(nir_builder *b, nir_function_impl *impl,
                       const struct g17p_equiv_probe *probes,
                       unsigned num_probes,
                       const struct g17p_equiv_chunk *chunks,
                       unsigned num_chunks, const int *chunk_of,
                       bool reuse_small, nir_def *const *cur_chunks,
                       nir_def *const *cur_probes, nir_def **store_mask_io,
                       nir_def **next_rem_io)
{
   nir_variable *var_store =
      nir_local_variable_create(impl, glsl_uint16_t_type(), "equiv_sm");
   nir_variable *var_rem =
      nir_local_variable_create(impl, glsl_uint16_t_type(), "equiv_nr");
   nir_store_var(b, var_store, *store_mask_io, 1);
   nir_store_var(b, var_rem, *next_rem_io, 1);

   nir_if *dedup_if = nir_push_if(b, nir_ine_imm(b, *next_rem_io, 0));
   {
      nir_variable *var_scan =
         nir_local_variable_create(impl, glsl_uint16_t_type(), "equiv_scan");
      nir_store_var(b, var_scan, *next_rem_io, 1);
      nir_loop *scan_loop = nir_push_loop(b);
      scan_loop->control = nir_loop_control_dont_unroll;
      {
         nir_def *scan = nir_load_var(b, var_scan);
         nir_break_if(b, nir_ieq_imm(b, scan, 0));
         nir_def *obit = nir_iand(b, scan, nir_ineg(b, scan));
         nir_store_var(b, var_scan, nir_ixor(b, scan, obit), 1);

         nir_def *o_chunks[G17P_EQUIV_MAX_PROBES];
         nir_def *o_probes[G17P_EQUIV_MAX_PROBES];
         g17p_equiv_load_sample(b, probes, num_probes, chunks, num_chunks,
                                chunk_of, reuse_small, obit, o_chunks,
                                o_probes);
         nir_def *odiff = g17p_equiv_diff_sample(
            b, probes, num_probes, chunks, num_chunks, chunk_of, cur_chunks,
            cur_probes, o_chunks, o_probes);
         nir_def *same = nir_ieq_imm(b, odiff, 0);
         nir_def *sm = nir_load_var(b, var_store);
         nir_def *nr = nir_load_var(b, var_rem);
         nir_store_var(b, var_store,
                       nir_bcsel(b, same, nir_ior(b, sm, obit), sm), 1);
         nir_store_var(b, var_rem,
                       nir_bcsel(b, same, nir_ixor(b, nr, obit), nr), 1);
      }
      nir_pop_loop(b, scan_loop);
   }
   nir_pop_if(b, dedup_if);

   *store_mask_io = nir_load_var(b, var_store);
   *next_rem_io = nir_load_var(b, var_rem);
}

/*
 * Monolithic MSAA local-read shaders often run on interior pixels where every
 * sample in the subgroup has full coverage and identical input tilebuffer
 * contents. Instead of unconditionally unrolling the entire shading body
 * nr_samples times, probe the input tilebuffer reads across samples, ballot
 * across the subgroup, and execute the body once with a whole-pixel tile store
 * when all active lanes are sample-equivalent. Only sample 0's probed values
 * remain live into the branches, keeping register pressure flat.
 */
bool
agx_nir_unroll_g17p_samples_equiv(nir_shader *shader, unsigned nr_samples,
                                  const struct agx_g17p_colour_layout *layout)
{
   (void)layout;
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   if ((nr_samples != 2 && nr_samples != 4) ||
       shader->info.fs.uses_discard || shader->info.writes_memory)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   struct g17p_equiv_probe probes[G17P_EQUIV_MAX_PROBES];
   unsigned num_probes = 0;
   unsigned total_dwords = 0;
   unsigned store_count = 0;
   bool saw_store = false;
   bool has_derivatives = false;
   nir_intrinsic_instr *force_early_instr = NULL;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type == nir_instr_type_tex) {
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               has_derivatives = true;
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_sample_mask_agx:
            if (shader->info.fs.early_fragment_tests && !force_early_instr &&
                block == nir_start_block(impl) && num_probes == 0 &&
                !saw_store && nir_src_is_const(intr->src[0]) &&
                nir_src_is_const(intr->src[1]) &&
                nir_src_as_uint(intr->src[0]) == G17P_EQUIV_ALL_SAMPLES &&
                nir_src_as_uint(intr->src[1]) == G17P_EQUIV_ALL_SAMPLES) {
               force_early_instr = intr;
               continue;
            }
            return false;

         case nir_intrinsic_load_active_samples_agx:
            if (!g17p_equiv_active_samples_uses_valid(&intr->def))
               return false;
            continue;

         case nir_intrinsic_load_local_pixel_agx: {
            if (saw_store || block->cf_node.parent != &impl->cf_node ||
                !g17p_equiv_is_valid_tile_mask(intr->src[0].ssa))
               return false;
            if (g17p_equiv_find_probe(probes, num_probes, intr) < 0) {
               if (num_probes >= G17P_EQUIV_MAX_PROBES)
                  return false;
               struct g17p_equiv_probe probe = {
                  .base = nir_intrinsic_base(intr),
                  .target = nir_intrinsic_target(intr),
                  .format = nir_intrinsic_format(intr),
                  .num_components = intr->def.num_components,
                  .bit_size = intr->def.bit_size,
               };
               g17p_equiv_set_raw_form(&probe);
               const unsigned dwords = DIV_ROUND_UP(
                  probe.raw_components * probe.raw_bit_size, 32);
               if (total_dwords + dwords > G17P_EQUIV_MAX_DWORDS)
                  return false;
               probes[num_probes++] = probe;
               total_dwords += dwords;
            }
            continue;
         }

         case nir_intrinsic_store_local_pixel_agx:
            if (!g17p_equiv_is_valid_tile_mask(intr->src[1].ssa))
               return false;
            saw_store = true;
            store_count++;
            continue;

         case nir_intrinsic_load_sample_mask:
            continue;

         default: {
            if (g17p_equiv_sample_dependent(intr->intrinsic))
               return false;
            const unsigned flags = nir_intrinsic_infos[intr->intrinsic].flags;
            if (!(flags & NIR_INTRINSIC_CAN_ELIMINATE))
               return false;
            if (nir_intrinsic_has_access(intr) &&
                (nir_intrinsic_access(intr) & ACCESS_VOLATILE))
               return false;
            if (flags & NIR_INTRINSIC_QUADGROUP) {
               has_derivatives = true;
               continue;
            }
            if (flags & NIR_INTRINSIC_SUBGROUP)
               return false;
            continue;
         }
         }
      }
   }

   if (num_probes == 0 || store_count == 0)
      return false;

   const bool force_early = force_early_instr != NULL;
   if (force_early_instr)
      nir_instr_remove(&force_early_instr->instr);

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   if (force_early)
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, G17P_EQUIV_ALL_SAMPLES, 16),
                          nir_imm_intN_t(&b, G17P_EQUIV_ALL_SAMPLES, 16));

   /* Sample coverage cannot change after force_early because discard, demote,
    * zs_emit and all other sample-mask mutations were rejected above. Read
    * native W16 SR39 once and derive the covered-sample masks for the probes,
    * fast path and slow path so agx_nir_lower_g17p_tile_samples() does not
    * emit a separate coverage read in every basic block. */
   const unsigned full_mask = BITFIELD_MASK(nr_samples);
   nir_def *cov_raw = nir_load_active_samples_agx(&b);
   nir_def *cov_low = nir_iand_imm(&b, cov_raw, full_mask);
   nir_def *sample_masks[4];
   for (unsigned s = 0; s < nr_samples; ++s)
      sample_masks[s] = nir_iand_imm(&b, cov_raw, BITFIELD_BIT(s));

   const bool reuse_small = total_dwords <= G17P_EQUIV_REUSE_DWORDS;

   nir_def *v0[G17P_EQUIV_MAX_PROBES];
   nir_def *reuse[G17P_EQUIV_MAX_PROBES];
   bool any_reuse = false;
   bool all_reusable = reuse_small;

   struct g17p_equiv_chunk chunks[G17P_EQUIV_MAX_PROBES];
   int chunk_of[G17P_EQUIV_MAX_PROBES];
   const unsigned num_chunks =
      g17p_equiv_build_chunks(probes, num_probes, chunks, chunk_of);
   nir_def *chunk_v0[G17P_EQUIV_MAX_PROBES];

   /* Emit probes in chunk-major order (all samples of each merged chunk, then
    * standalone probes) so each chunk's sample loads can batch together within
    * a tight register footprint. Track per-sample diffs against sample 0 so the
    * slow path can deduplicate samples matching sample 0 without extra probe
    * work on the fast path. */
   nir_def *diff16[4] = {NULL};
   nir_def *diff32[4] = {NULL};

   for (unsigned c = 0; c < num_chunks; ++c) {
      chunk_v0[c] = nir_load_local_pixel_agx(
         &b, chunks[c].words, 32, sample_masks[0], .base = chunks[c].base,
         .format = PIPE_FORMAT_R32_UINT, .target = chunks[c].target);

      for (unsigned s = 1; s < nr_samples; ++s) {
         nir_def *vs = nir_load_local_pixel_agx(
            &b, chunks[c].words, 32, sample_masks[s], .base = chunks[c].base,
            .format = PIPE_FORMAT_R32_UINT, .target = chunks[c].target);
         nir_def *diff = nir_ixor(&b, chunk_v0[c], vs);
         for (unsigned i = 0; i < chunks[c].words; ++i) {
            nir_def *d = nir_channel(&b, diff, i);
            diff32[s] = diff32[s] ? nir_ior(&b, diff32[s], d) : d;
         }
      }
   }

   for (unsigned k = 0; k < num_probes; ++k) {
      const bool can = reuse_small && g17p_equiv_probe_reuses_v0(&probes[k]);
      if (chunk_of[k] >= 0) {
         const struct g17p_equiv_chunk *c = &chunks[chunk_of[k]];
         const unsigned first = (probes[k].base - c->base) / 4;
         v0[k] = nir_channels(
            &b, chunk_v0[chunk_of[k]],
            BITFIELD_MASK(probes[k].raw_components) << first);
      } else {
         const uint8_t probe_bits =
            g17p_equiv_probe_bits(&probes[k], reuse_small);
         v0[k] = nir_load_local_pixel_agx(
            &b, probes[k].raw_components, probe_bits, sample_masks[0],
            .base = probes[k].base, .format = probes[k].raw_format,
            .target = probes[k].target);

         for (unsigned s = 1; s < nr_samples; ++s) {
            nir_def *vs = nir_load_local_pixel_agx(
               &b, probes[k].raw_components, probe_bits, sample_masks[s],
               .base = probes[k].base, .format = probes[k].raw_format,
               .target = probes[k].target);
            nir_def *diff = nir_ixor(&b, v0[k], vs);
            for (unsigned c = 0; c < probes[k].raw_components; ++c) {
               nir_def *d = nir_channel(&b, diff, c);
               nir_def **acc = &diff16[s];
               if (d->bit_size != 16) {
                  d = nir_u2u32(&b, d);
                  acc = &diff32[s];
               }
               *acc = *acc ? nir_ior(&b, *acc, d) : d;
            }
         }
      }
      reuse[k] = can ? v0[k] : NULL;
      any_reuse |= can;
      all_reusable &= can;
   }

   nir_def *ds[4] = {NULL};
   nir_def *diff_all = NULL;
   for (unsigned s = 1; s < nr_samples; ++s) {
      if (diff16[s] && diff32[s])
         ds[s] = nir_ior(&b, diff32[s], nir_u2u32(&b, diff16[s]));
      else if (diff32[s])
         ds[s] = diff32[s];
      else if (diff16[s])
         ds[s] = nir_u2u32(&b, diff16[s]);
      else
         ds[s] = nir_imm_int(&b, 0);
      diff_all = diff_all ? nir_ior(&b, diff_all, ds[s]) : ds[s];
   }

   /* A lane disqualifies the whole-pixel fast path if any sample is uncovered
    * (cov_low ^ full_mask != 0) or any probed bit differs across samples.
    * Fold all mismatch bits into one 32-bit word so the subgroup ballot lowers
    * directly to a single witnessed AGX3B 32-bit icmp_ballot(mismatch != 0). */
   nir_def *mismatch = nir_u2u32(
      &b, nir_ixor(&b, cov_low, nir_imm_intN_t(&b, full_mask, 16)));
   if (diff_all)
      mismatch = nir_ior(&b, mismatch, diff_all);

   nir_def *any_not_eq = nir_ballot(&b, 1, 32, nir_ine_imm(&b, mismatch, 0));
   nir_def *all_equiv = nir_ieq_imm(&b, any_not_eq, 0);

   nir_if *nif = nir_push_if(&b, all_equiv);
   {
      nir_cf_list fast_copy;
      nir_cf_list_clone(&fast_copy, &body, &impl->cf_node, NULL);
      specialize_g17p_equiv_copy(impl, &fast_copy, sample_masks[0], cov_low,
                                 probes, any_reuse ? reuse : NULL, num_probes);
      b.cursor = nir_cf_reinsert(&fast_copy, b.cursor);
   }
   nir_push_else(&b, nif);
   {
      if (has_derivatives) {
         /* Quad/derivative operations require uniform sample execution across
          * every 2x2 quad; keep the unrolled per-sample slow path instead of
          * a divergent per-lane deduplication loop. */
         for (unsigned i = 0; i < nr_samples; ++i) {
            nir_cf_list slow_copy;
            nir_cf_list_clone(&slow_copy, &body, &impl->cf_node, NULL);
            specialize_g17p_equiv_copy(impl, &slow_copy, sample_masks[i],
                                       sample_masks[i], probes,
                                       (i == 0 && any_reuse) ? reuse : NULL,
                                       num_probes);
            b.cursor = nir_cf_reinsert(&slow_copy, b.cursor);
         }
      } else {
         /* Deduplicate samples in the slow path: shade sample 0 once for all
          * covered samples matching sample 0 (reusing sample 0's probed words),
          * then loop over the remaining covered samples with a single coalesced
          * copy of the body (and inner deduplication among remaining samples
          * for 4x MSAA so 2-region edges shade twice instead of four times). */
         nir_def *eq0 = nir_imm_intN_t(&b, 1, 16);
         for (unsigned s = 1; s < nr_samples; ++s) {
            nir_def *same = nir_ieq_imm(&b, ds[s], 0);
            eq0 = nir_ior(&b, eq0,
                          nir_bcsel(&b, same,
                                    nir_imm_intN_t(&b, BITFIELD_BIT(s), 16),
                                    nir_imm_intN_t(&b, 0, 16)));
         }
         nir_def *eq0_mask = nir_bcsel(&b, nir_ine_imm(&b, sample_masks[0], 0),
                                       nir_iand(&b, eq0, cov_low),
                                       nir_imm_intN_t(&b, 0, 16));
         nir_def *rem = nir_ixor(&b, cov_low, eq0_mask);

         nir_cf_list pass0_copy;
         nir_cf_list_clone(&pass0_copy, &body, &impl->cf_node, NULL);
         specialize_g17p_equiv_copy(impl, &pass0_copy, sample_masks[0],
                                    eq0_mask, probes, any_reuse ? reuse : NULL,
                                    num_probes);
         b.cursor = nir_cf_reinsert(&pass0_copy, b.cursor);

         nir_variable *var_rem =
            nir_local_variable_create(impl, glsl_uint16_t_type(), "equiv_rem");
         nir_store_var(&b, var_rem, rem, 1);

         nir_loop *loop = nir_push_loop(&b);
         loop->control = nir_loop_control_dont_unroll;
         {
            nir_def *cur_rem = nir_load_var(&b, var_rem);
            nir_break_if(&b, nir_ieq_imm(&b, cur_rem, 0));

            nir_def *bit = nir_iand(&b, cur_rem, nir_ineg(&b, cur_rem));
            nir_def *next_rem = nir_ixor(&b, cur_rem, bit);
            nir_def *cur_store = bit;

            if (all_reusable) {
               nir_def *cur_chunks[G17P_EQUIV_MAX_PROBES];
               nir_def *cur_probes[G17P_EQUIV_MAX_PROBES];
               g17p_equiv_load_sample(&b, probes, num_probes, chunks,
                                      num_chunks, chunk_of, reuse_small, bit,
                                      cur_chunks, cur_probes);
               if (nr_samples > 2) {
                  g17p_equiv_inner_dedup(&b, impl, probes, num_probes, chunks,
                                         num_chunks, chunk_of, reuse_small,
                                         cur_chunks, cur_probes, &cur_store,
                                         &next_rem);
               }
               nir_store_var(&b, var_rem, next_rem, 1);
               nir_cf_list slow_copy;
               nir_cf_list_clone(&slow_copy, &body, &impl->cf_node, NULL);
               specialize_g17p_equiv_copy(impl, &slow_copy, bit, cur_store,
                                          probes, cur_probes, num_probes);
               b.cursor = nir_cf_reinsert(&slow_copy, b.cursor);
            } else {
               nir_store_var(&b, var_rem, next_rem, 1);
               nir_cf_list slow_copy;
               nir_cf_list_clone(&slow_copy, &body, &impl->cf_node, NULL);
               specialize_g17p_equiv_copy(impl, &slow_copy, bit, bit, probes,
                                          NULL, num_probes);
               b.cursor = nir_cf_reinsert(&slow_copy, b.cursor);
            }
         }
         nir_pop_loop(&b, loop);
      }
   }
   nir_pop_if(&b, nif);

   nir_cf_delete(&body);
   nir_progress(true, impl, nir_metadata_none);
   if (!has_derivatives)
      nir_lower_vars_to_ssa(shader);
   return true;
}

/*
 * AGX3B keeps track of which samples of a pixel hold the same tilebuffer
 * contents, and PIXINFO returns those sample groups for the fragment's coverage
 * (nir_intrinsic_load_sample_groups_agx). A monolithic multisample local-read
 * body already lowered to per-sample form then needs one shading per group, not
 * one per sample, and no software comparison of the samples:
 *
 *    w = groups(coverage)
 *    loop {
 *       load ..., mask w            ; a sample of group 0 of w
 *       ... shade ...
 *       store ..., mask coverage & w
 *       w >>= 4; break if w == 0
 *    }
 *
 * Every sample of a group holds the same bytes, so the group's shading is the
 * shading of each of its samples. The body must not depend on the sample index
 * or position, on other lanes (a divergent loop has no quads), or have side
 * effects other than tile stores, exactly as for the probed path
 * (agx_nir_unroll_g17p_samples_equiv()).
 */
bool
agx_nir_unroll_g17p_samples_groups(nir_shader *shader, unsigned nr_samples,
                                   enum agx_isa_encoding encoding)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   if (encoding != AGX_ISA_ENCODING_AGX3B ||
       (nr_samples != 2 && nr_samples != 4) ||
       shader->info.fs.uses_discard || shader->info.writes_memory)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   unsigned loads = 0, stores = 0;
   nir_intrinsic_instr *force_early_instr = NULL;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type == nir_instr_type_tex) {
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_sample_mask_agx:
            if (shader->info.fs.early_fragment_tests && !force_early_instr &&
                block == nir_start_block(impl) && loads == 0 && stores == 0 &&
                nir_src_is_const(intr->src[0]) &&
                nir_src_is_const(intr->src[1]) &&
                nir_src_as_uint(intr->src[0]) == G17P_EQUIV_ALL_SAMPLES &&
                nir_src_as_uint(intr->src[1]) == G17P_EQUIV_ALL_SAMPLES) {
               force_early_instr = intr;
               continue;
            }
            return false;

         case nir_intrinsic_load_active_samples_agx:
            if (!g17p_equiv_active_samples_uses_valid(&intr->def))
               return false;
            continue;

         case nir_intrinsic_load_local_pixel_agx:
            if (!g17p_equiv_is_valid_tile_mask(intr->src[0].ssa))
               return false;
            loads++;
            continue;

         case nir_intrinsic_store_local_pixel_agx:
            if (!g17p_equiv_is_valid_tile_mask(intr->src[1].ssa))
               return false;
            stores++;
            continue;

         case nir_intrinsic_load_sample_mask:
            continue;

         default: {
            if (g17p_equiv_sample_dependent(intr->intrinsic))
               return false;
            const unsigned flags = nir_intrinsic_infos[intr->intrinsic].flags;
            if (!(flags & NIR_INTRINSIC_CAN_ELIMINATE) ||
                (flags & (NIR_INTRINSIC_QUADGROUP | NIR_INTRINSIC_SUBGROUP)))
               return false;
            if (nir_intrinsic_has_access(intr) &&
                (nir_intrinsic_access(intr) & ACCESS_VOLATILE))
               return false;
            continue;
         }
         }
      }
   }

   if (loads == 0 || stores == 0)
      return false;

   if (force_early_instr)
      nir_instr_remove(&force_early_instr->instr);

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   if (force_early_instr)
      nir_sample_mask_agx(&b, nir_imm_intN_t(&b, G17P_EQUIV_ALL_SAMPLES, 16),
                          nir_imm_intN_t(&b, G17P_EQUIV_ALL_SAMPLES, 16));

   /* Coverage cannot change in the body: discard, demote, depth and stencil
    * writes and every other sample-mask update were rejected above. */
   /* The remaining groups are shifted as a 32-bit word: G17P's SRL takes an
    * immediate count only at 32 bits. Its low half is the 16-bit mask. */
   nir_def *coverage = nir_load_active_samples_agx(&b);
   nir_variable *groups =
      nir_local_variable_create(impl, glsl_uint_type(), "sample_groups");
   nir_store_var(&b, groups,
                 nir_u2u32(&b, nir_load_sample_groups_agx(&b, coverage)), 1);

   nir_loop *loop = nir_push_loop(&b);
   loop->control = nir_loop_control_dont_unroll;
   {
      nir_def *remaining = nir_load_var(&b, groups);
      nir_def *group = nir_u2u16(&b, remaining);
      nir_def *store_samples = nir_iand(&b, coverage, group);

      nir_cf_list copy;
      nir_cf_list_clone(&copy, &body, &impl->cf_node, NULL);
      specialize_g17p_equiv_copy(impl, &copy, group, store_samples, NULL,
                                 NULL, 0);
      b.cursor = nir_cf_reinsert(&copy, b.cursor);

      nir_def *rest = nir_ushr_imm(&b, remaining, 4);
      nir_store_var(&b, groups, rest, 1);
      nir_break_if(&b, nir_ieq_imm(&b, rest, 0));
   }
   nir_pop_loop(&b, loop);

   nir_cf_delete(&body);
   nir_progress(true, impl, nir_metadata_none);
   nir_lower_vars_to_ssa(shader);
   return true;
}

static bool movable_zs_source(nir_def *def);

/* Conservative set of bits a target can affect. Per-sample lowering ends
 * each target with AND(active_sample), even if algebraic folding has not
 * reduced its other operand to a constant yet. Unknown expressions may set
 * every bit, so they never prove independence from another nonempty target. */
static unsigned
possible_sample_targets(nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);
   if (def->num_components != 1 || def->bit_size != 16)
      return UINT_MAX;
   if (instr->type == nir_instr_type_load_const)
      return nir_src_as_uint(nir_src_for_ssa(def));
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_iand)
         return possible_sample_targets(alu->src[0].src.ssa) &
                possible_sample_targets(alu->src[1].src.ssa);
   }
   return UINT16_MAX;
}

static bool
movable_zs_src_cb(nir_src *src, void *data)
{
   return movable_zs_source(src->ssa);
}

static bool
movable_zs_source(nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu:
      return nir_foreach_src(instr, movable_zs_src_cb, NULL);
   case nir_instr_type_intrinsic:
      switch (nir_instr_as_intrinsic(instr)->intrinsic) {
      case nir_intrinsic_load_exported_agx:
      /* Alpha-coverage dithering reads FragCoord.xy, later lowered to pixel
       * coordinates. Both inputs are invariant across coverage/Z/S updates. */
      case nir_intrinsic_load_frag_coord:
      case nir_intrinsic_load_pixel_coord:
         return true;
      default:
         return false;
      }
   default:
      return false;
   }
}

struct move_zs_state {
   nir_cursor cursor;
   struct set *moved;
};

static bool
move_zs_source(nir_src *src, void *data)
{
   struct move_zs_state *state = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   if (_mesa_set_search(state->moved, instr))
      return true;

   nir_foreach_src(instr, move_zs_source, state);
   nir_instr_move(state->cursor, instr);
   state->cursor = nir_after_instr(instr);
   _mesa_set_add(state->moved, instr);
   return true;
}

/* A block the function executes exactly once: directly in the function body,
 * outside any if or loop. The unrolled per-sample copies of an epilog with
 * per-sample control flow (spilled render targets) start in such blocks. */
static bool
agx_block_runs_once(nir_function_impl *impl, nir_block *block)
{
   return block->cf_node.parent == &impl->cf_node;
}

/*
 * Coalesce each sample's kill into one coverage update. Keep a complete,
 * disjoint depth-only sample set in its original order: native G17P fragments
 * test a sample immediately before accessing that sample's colour. Hoisting
 * all depth operations ahead of all colour accesses loses that ordering.
 *
 * Other depth/stencil shapes retain the established hoist. The binary packer
 * recognizes the interleaved form and holds depth and colour classes together
 * until the final colour store; it never reacquires depth per sample while
 * holding colour classes.
 */
bool
agx_nir_hoist_g17p_zs_writes(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_intrinsic_instr *kills[8];
   nir_intrinsic_instr *writes[8];
   unsigned kill_count = 0, count = 0;
   bool progress = false;
   bool tests_samples = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_demote_samples)
            return false;

         if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            /* Pure kills may precede the depth transaction. Colour-only
             * epilogs may instead test their targeted samples here; admit
             * those only for provably disjoint target sets, checked below. */
            if (kill_count == ARRAY_SIZE(kills) ||
                !agx_block_runs_once(impl, block) ||
                !movable_zs_source(intr->src[0].ssa) ||
                !movable_zs_source(intr->src[1].ssa))
               return false;
            tests_samples |= !nir_src_is_const(intr->src[1]) ||
                             nir_src_as_uint(intr->src[1]) != 0;
            kills[kill_count++] = intr;
            continue;
         }

         if (intr->intrinsic != nir_intrinsic_store_zs_agx)
            continue;

         /* Static sample unrolling can prove that an API mask excludes this
          * sample. A zero-coverage Z/S store is no operation; retaining it
          * would instead make the backend issue a DEPTHF for that sample. */
         if (nir_src_is_const(intr->src[0]) &&
             nir_src_as_uint(intr->src[0]) == 0) {
            nir_instr_remove(instr);
            progress = true;
            continue;
         }

         if (count == ARRAY_SIZE(writes) || !agx_block_runs_once(impl, block))
            return false;

         for (unsigned i = 0; i < 3; ++i) {
            if (!movable_zs_source(intr->src[i].ssa))
               return false;
         }
         writes[count++] = intr;
      }
   }
   /* A sample's colour accesses must observe that sample's depth result.
    * Keep a complete, disjoint depth-only sample set in its original
    * sample order, including zero-colour epilogs. Kills still coalesce below.
    * The packer retains one shared
    * depth/colour transaction, so this does not reacquire class 1 per sample. */
   bool per_sample_depth = count == 2 || count == 4;
   unsigned targets = 0;
   for (unsigned w = 0; w < count; ++w) {
      const unsigned target = possible_sample_targets(writes[w]->src[0].ssa);
      per_sample_depth &= nir_intrinsic_base(writes[w]) == 1 &&
                          util_bitcount(target) == 1 && !(targets & target);
      targets |= target;
   }
   if (per_sample_depth && targets == BITFIELD_MASK(count))
      count = 0;

   if (tests_samples) {
      /* Do not combine testing updates with DEPTHF, or test one sample
       * twice. Independent targets permit LIVE to be masked by its own
       * TARGET before unioning; ORing the unmasked LIVE values is wrong. */
      if (count)
         return false;
      unsigned targets = 0;
      for (unsigned k = 0; k < kill_count; ++k) {
         unsigned target = possible_sample_targets(kills[k]->src[0].ssa);
         if (targets & target)
            return false;
         targets |= target;
      }
   }

   if ((count == 0 && kill_count <= 1) || (count == 1 && kill_count == 0))
      return nir_progress(progress, impl, nir_metadata_control_flow);

   struct move_zs_state state = {
      .cursor = nir_before_impl(impl),
      .moved = _mesa_pointer_set_create(NULL),
   };
   nir_def *affected = NULL, *surviving = NULL;
   for (unsigned k = 0; k < kill_count; ++k) {
      nir_foreach_src(&kills[k]->instr, move_zs_source, &state);
      nir_builder b = nir_builder_at(state.cursor);
      nir_def *target = kills[k]->src[0].ssa;
      affected = affected ? nir_ior(&b, affected, target) : target;
      if (tests_samples) {
         nir_def *live = nir_iand(&b, target, kills[k]->src[1].ssa);
         surviving = surviving ? nir_ior(&b, surviving, live) : live;
      }
      state.cursor = b.cursor;
      nir_instr_remove(&kills[k]->instr);
   }
   if (affected) {
      /* Kill-only updates commute even with overlapping targets. Testing
       * updates have disjoint targets, so each sample keeps its own LIVE bit
       * and runs tests at most once, before the shared colour transaction. */
      nir_builder b = nir_builder_at(state.cursor);
      if (!surviving)
         surviving = nir_imm_intN_t(&b, 0, affected->bit_size);
      nir_sample_mask_agx(&b, affected, surviving);
      state.cursor = b.cursor;
   }
   for (unsigned w = 0; w < count; ++w) {
      nir_foreach_src(&writes[w]->instr, move_zs_source, &state);
      nir_instr_move(state.cursor, &writes[w]->instr);
      state.cursor = nir_after_instr(&writes[w]->instr);
   }
   _mesa_set_destroy(state.moved, NULL);

   return nir_progress(true, impl, nir_metadata_control_flow);
}

/*
 * Lower a top-level whole-pixel depth-only or stencil-only store_zs_agx at
 * >= 2 samples on M5 into a single depth/coverage transaction looping over
 * covered samples with one DEPTHF per sample (AGX_ZS_COVERED_SAMPLE).
 * The loop runs at least once so lanes with empty coverage still balance the
 * transaction with an empty word. Combined depth+stencil and masked writes
 * are left unchanged.
 */
bool
agx_nir_lower_m5_zs_sample_loop(nir_shader *shader, unsigned nr_samples)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   if (nr_samples < 2)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_intrinsic_instr *store = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_zs_agx)
            continue;
         /* Exactly one write, of a supported shape, or none rewritten. */
         if (store)
            return false;
         store = intr;
      }
   }

   if (!store || !agx_block_runs_once(impl, store->instr.block))
      return false;

   const unsigned base = nir_intrinsic_base(store);
   const unsigned all = BITFIELD_MASK(nr_samples);
   if ((base != 1 && base != 2) || !nir_src_is_const(store->src[0]) ||
       (nir_src_as_uint(store->src[0]) & all) != all)
      return false;

   nir_builder b = nir_builder_at(nir_before_instr(&store->instr));

   /* Native clamps the depth once, ahead of the loop. The backend does not
    * clamp a covered-sample word again. */
   nir_def *depth = store->src[1].ssa;
   if (base == 1)
      depth = nir_fmin(&b, nir_fmax(&b, depth, nir_imm_float(&b, 0.0f)),
                       nir_imm_float(&b, 1.0f));
   nir_zs_transaction_agx(&b, .flags = 0);

   nir_variable *left =
      nir_local_variable_create(impl, glsl_uint16_t_type(), "zs_samples");
   nir_store_var(&b, left,
                 nir_iand_imm(&b, nir_u2u16(&b, nir_load_sample_mask(&b)), all),
                 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *samples = nir_load_var(&b, left);
      nir_def *sample = nir_iand(&b, samples, nir_ineg(&b, samples));
      nir_store_zs_agx(&b, sample, depth, store->src[2].ssa,
                       .base = base | AGX_ZS_COVERED_SAMPLE);
      nir_def *rest = nir_iand(&b, samples, nir_inot(&b, sample));
      nir_store_var(&b, left, rest, 1);
      nir_break_if(&b, nir_ieq_imm(&b, rest, 0));
   }
   nir_pop_loop(&b, loop);

   nir_zs_transaction_agx(&b, .flags = 1);
   nir_instr_remove(&store->instr);

   nir_progress(true, impl, nir_metadata_none);
   nir_lower_vars_to_ssa(shader);
   return true;
}

/* Whether an instruction may run once per pixel ahead of the sample loop:
 * the coverage update, the depth/stencil write and the pixel-rate values they
 * are computed from. */
static bool
m5_pixel_rate_instr(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return instr->type == nir_instr_type_alu ||
             instr->type == nir_instr_type_load_const ||
             instr->type == nir_instr_type_undef;

   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_sample_mask_agx:
   case nir_intrinsic_store_zs_agx:
   case nir_intrinsic_load_exported_agx:
   case nir_intrinsic_load_shader_part_tests_zs_agx:
   /* The coverage dither is per pixel. */
   case nir_intrinsic_load_frag_coord:
   case nir_intrinsic_load_pixel_coord:
      return true;
   default:
      return false;
   }
}

static bool
m5_coverage_or_zs(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
   return op == nir_intrinsic_sample_mask_agx ||
          op == nir_intrinsic_store_zs_agx;
}

/* The one top-level depth/stencil write, moved with its pixel-rate sources
 * to just after the last coverage update, ahead of any tile access. Epilogs
 * emit it last, after the blend. */
static void
m5_hoist_zs_write(nir_function_impl *impl)
{
   nir_intrinsic_instr *store = NULL;
   nir_cf_node *last_update = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
            if (store || !agx_block_runs_once(impl, block))
               return;
            store = intr;
         } else if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            /* A kill must reach the write first. */
            if (store)
               return;
            nir_cf_node *node = &block->cf_node;
            while (node->parent != &impl->cf_node)
               node = node->parent;
            last_update = node;
         }
      }
   }
   if (!store)
      return;
   for (unsigned i = 0; i < nir_intrinsic_infos[store->intrinsic].num_srcs; ++i) {
      if (!movable_zs_source(store->src[i].ssa))
         return;
   }
   /* Only where the whole part ahead of it stays at pixel rate. */
   if (last_update) {
      foreach_list_typed(nir_cf_node, node, node, &impl->body) {
         nir_foreach_block_in_cf_node(block, node) {
            nir_foreach_instr(instr, block) {
               if (!m5_pixel_rate_instr(instr))
                  return;
            }
         }
         if (node == last_update)
            break;
      }
   }

   struct move_zs_state state = {
      .cursor = last_update ? nir_after_cf_node(last_update)
                            : nir_before_impl(impl),
      .moved = _mesa_pointer_set_create(NULL),
   };
   nir_foreach_src(&store->instr, move_zs_source, &state);
   nir_instr_move(state.cursor, &store->instr);
   _mesa_set_destroy(state.moved, NULL);
}

/*
 * Find the end of the leading pixel-rate prefix of an M5 epilog containing
 * all coverage updates and depth/stencil writes (false if none or if other
 * work precedes the last one).
 *
 * Coverage updates and depth writes must complete at pixel rate before the
 * per-sample blend loop: a testing MOVMSK tests every currently covered
 * sample, so issuing it inside the sample loop would test and write depth for
 * samples killed by a later iteration, and leaving DEPTHF inside the loop
 * would interleave depth writes with per-sample colour blending.
 */
static bool
m5_pixel_rate_prefix(nir_shader *shader, nir_cursor *end)
{
   if (shader->info.fs.early_fragment_tests)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   m5_hoist_zs_write(impl);

   unsigned left = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block)
         left += m5_coverage_or_zs(instr);
   }
   if (!left)
      return false;

   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_loop)
         return false;
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block) {
            if (!m5_pixel_rate_instr(instr))
               return false;
            if (!m5_coverage_or_zs(instr) || --left)
               continue;
            /* The last one: end after it, or after the control flow
             * around it. */
            *end = node->type == nir_cf_node_block ? nir_after_instr(instr)
                                                   : nir_after_cf_node(node);
            return true;
         }
      }
   }
   UNREACHABLE("every update was counted");
}

/*
 * Lower a fragment shader into a monolithic pixel shader, with static sample
 * count, blend state, and tilebuffer formats in the shader key. For dynamic,
 * epilogs must be used, which have separate lowerings.
 */
static bool
lower_monolithic_msaa(nir_shader *shader, uint8_t nr_samples, bool m5)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   assert(nr_samples == 1 || nr_samples == 2 || nr_samples == 4 || nr_samples == 8);

   if (m5)
      agx_nir_lower_sample_mask_m5(shader);
   else
      agx_nir_lower_sample_mask(shader);

   /* In single sampled programs, interpolateAtSample needs to return the
    * center pixel.
    */
   if (nr_samples == 1) {
      nir_lower_single_sampled_options lss_opts = {
         .lower_sample_mask_in = true,
      };
      nir_lower_single_sampled(shader, &lss_opts);
   } else if (shader->info.fs.uses_sample_shading) {
      /* M5 keeps a pixel-rate coverage update and depth/stencil write ahead
       * of the sample loop: set them aside while the rest becomes
       * per-sample, then put them back in front of the loop. */
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      nir_cursor prefix_end;
      const bool split = m5 && m5_pixel_rate_prefix(shader, &prefix_end);
      nir_cf_list prefix;
      if (split)
         nir_cf_extract(&prefix, nir_before_impl(impl), prefix_end);

      agx_nir_lower_to_per_sample(shader);
      agx_nir_wrap_per_sample_loop(shader, nr_samples);

      if (split) {
         nir_cf_reinsert(&prefix, nir_before_impl(impl));
         nir_progress(true, impl, nir_metadata_none);
      }
   }

   return true;
}

/*
 * Lower a fragment shader into a monolithic pixel shader, with static sample
 * count, blend state, and tilebuffer formats in the shader key. For dynamic,
 * epilogs must be used, which have separate lowerings.
 */
bool
agx_nir_lower_monolithic_msaa_agx2(nir_shader *shader, uint8_t nr_samples)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   assert(nr_samples == 1 || nr_samples == 2 || nr_samples == 4);

   agx_nir_lower_sample_mask(shader);

   /* In single sampled programs, interpolateAtSample needs to return the
    * center pixel.
    */
   if (nr_samples == 1) {
      nir_lower_single_sampled_options lss_opts = {
         .lower_sample_mask_in = true,
      };
      nir_lower_single_sampled(shader, &lss_opts);
   } else if (shader->info.fs.uses_sample_shading) {
      agx_nir_lower_to_per_sample_agx2(shader);
      agx_nir_wrap_per_sample_loop(shader, nr_samples);
   }

   return true;
}

bool
agx_nir_lower_monolithic_msaa(nir_shader *shader, uint8_t nr_samples)
{
   return lower_monolithic_msaa(shader, nr_samples, false);
}

bool
agx_nir_lower_monolithic_msaa_m5(nir_shader *shader, uint8_t nr_samples)
{
   return lower_monolithic_msaa(shader, nr_samples, true);
}
