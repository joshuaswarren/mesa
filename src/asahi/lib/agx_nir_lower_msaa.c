/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2021 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_tilebuffer.h"
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
      (agx_chip_g17_m5(chip) && !shader->info.fs.uses_sample_shading);
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
   unsigned base;
   unsigned target;
   enum pipe_format format;
   uint8_t num_components;
   uint8_t bit_size;
};

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

static void
specialize_g17p_equiv_copy(nir_cf_list *copy, nir_def *load_samples,
                           nir_def *store_samples,
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
               if (reuse_v0) {
                  const int k = g17p_equiv_find_probe(probes, num_probes, intr);
                  assert(k >= 0);
                  nir_def_rewrite_uses(&intr->def, reuse_v0[k]);
                  nir_instr_remove(instr);
               } else {
                  nir_src_rewrite(&intr->src[0], load_samples);
               }
            }
         }
      }
   }
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
   nir_intrinsic_instr *force_early_instr = NULL;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
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
               const unsigned dwords =
                  DIV_ROUND_UP(intr->def.num_components * intr->def.bit_size,
                               32);
               if (total_dwords + dwords > G17P_EQUIV_MAX_DWORDS)
                  return false;
               probes[num_probes++] = (struct g17p_equiv_probe){
                  .base = nir_intrinsic_base(intr),
                  .target = nir_intrinsic_target(intr),
                  .format = nir_intrinsic_format(intr),
                  .num_components = intr->def.num_components,
                  .bit_size = intr->def.bit_size,
               };
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

         case nir_intrinsic_load_sample_id:
         case nir_intrinsic_load_sample_pos:
         case nir_intrinsic_load_samples_log2_agx:
         case nir_intrinsic_load_barycentric_sample:
         case nir_intrinsic_load_barycentric_at_sample:
         case nir_intrinsic_load_exported_agx:
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
            return false;

         default:
            if (!(nir_intrinsic_infos[intr->intrinsic].flags &
                  NIR_INTRINSIC_CAN_ELIMINATE))
               return false;
            continue;
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

   const unsigned full_mask = BITFIELD_MASK(nr_samples);
   nir_def *cov = nir_u2u16(&b, nir_load_sample_mask(&b));
   nir_def *lane_eq =
      nir_ieq_imm(&b, nir_iand_imm(&b, cov, full_mask), full_mask);

   /* Reuse sample 0's probed values across the branch when the live footprint
    * is small. For wider MRT local-read shaders (such as 4-MRT deferred
    * lighting passes), keeping all probed payloads live across the branch
    * inflates register pressure; drop the probe values after the ballot and
    * reload sample 0 in the shading body instead. Narrow 8-bit unorm/snorm
    * probes to f16 in that mode (injective for all 256 values) so the probe
    * preamble uses half the registers while the shading body still loads
    * exact f32 values. */
   const bool reuse_v0 = total_dwords <= G17P_EQUIV_REUSE_DWORDS;

   nir_def *v0[G17P_EQUIV_MAX_PROBES];
   for (unsigned k = 0; k < num_probes; ++k) {
      uint8_t probe_bits = probes[k].bit_size;
      if (!reuse_v0 && probe_bits == 32 &&
          (probes[k].format == PIPE_FORMAT_R8_UNORM ||
           probes[k].format == PIPE_FORMAT_R8_SNORM)) {
         probe_bits = 16;
      }

      v0[k] = nir_load_local_pixel_agx(
         &b, probes[k].num_components, probe_bits,
         nir_imm_intN_t(&b, BITFIELD_BIT(0), 16), .base = probes[k].base,
         .format = probes[k].format, .target = probes[k].target);

      for (unsigned s = 1; s < nr_samples; ++s) {
         nir_def *vs = nir_load_local_pixel_agx(
            &b, probes[k].num_components, probe_bits,
            nir_imm_intN_t(&b, BITFIELD_BIT(s), 16), .base = probes[k].base,
            .format = probes[k].format, .target = probes[k].target);
         nir_def *diff = nir_ixor(&b, v0[k], vs);
         nir_def *any_diff = nir_channel(&b, diff, 0);
         for (unsigned c = 1; c < probes[k].num_components; ++c)
            any_diff = nir_ior(&b, any_diff, nir_channel(&b, diff, c));
         lane_eq = nir_iand(&b, lane_eq, nir_ieq_imm(&b, any_diff, 0));
      }
   }

   nir_def *any_not_eq = nir_ballot(&b, 1, 32, nir_inot(&b, lane_eq));
   nir_def *all_equiv = nir_ieq_imm(&b, any_not_eq, 0);

   nir_if *nif = nir_push_if(&b, all_equiv);
   {
      nir_def *sample0_mask = nir_imm_intN_t(&b, BITFIELD_BIT(0), 16);
      nir_def *all_mask = nir_imm_intN_t(&b, G17P_EQUIV_ALL_SAMPLES, 16);
      nir_cf_list fast_copy;
      nir_cf_list_clone(&fast_copy, &body, &impl->cf_node, NULL);
      specialize_g17p_equiv_copy(&fast_copy, sample0_mask, all_mask, probes,
                                 reuse_v0 ? v0 : NULL, num_probes);
      b.cursor = nir_cf_reinsert(&fast_copy, b.cursor);
   }
   nir_push_else(&b, nif);
   {
      for (unsigned i = 0; i < nr_samples; ++i) {
         nir_def *sample_mask = nir_imm_intN_t(&b, BITFIELD_BIT(i), 16);
         nir_cf_list slow_copy;
         nir_cf_list_clone(&slow_copy, &body, &impl->cf_node, NULL);
         specialize_g17p_equiv_copy(&slow_copy, sample_mask, sample_mask,
                                    probes, (i == 0 && reuse_v0) ? v0 : NULL,
                                    num_probes);
         b.cursor = nir_cf_reinsert(&slow_copy, b.cursor);
      }
   }
   nir_pop_if(&b, nif);

   nir_cf_delete(&body);
   return nir_progress(true, impl, nir_metadata_none);
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
