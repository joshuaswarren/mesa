/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gallium/include/pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/nir/poly_nir.h"
#include "util/format/u_formats.h"
#include "agx_abi.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_lower_blend.h"
#include "shader_enums.h"

/* Legacy fragment epilog from upstream ab0ab220, retained under its MIT license. */
#define ALL_SAMPLES 0xFF
static int
map_fs_part_uniform(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_blend_const_color_r_float:
      return AGX_ABI_FUNI_BLEND_R;

   case nir_intrinsic_load_blend_const_color_g_float:
      return AGX_ABI_FUNI_BLEND_G;

   case nir_intrinsic_load_blend_const_color_b_float:
      return AGX_ABI_FUNI_BLEND_B;

   case nir_intrinsic_load_blend_const_color_a_float:
      return AGX_ABI_FUNI_BLEND_A;

   default:
      return -1;
   }
}

static bool
lower_non_monolithic_uniforms(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   int unif = map_fs_part_uniform(intr);

   if (unif >= 0) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *load = nir_load_preamble(b, 1, intr->def.bit_size, .base = unif);
      nir_def_rewrite_uses(&intr->def, load);
      return true;
   } else if (intr->intrinsic == nir_intrinsic_load_texture_handle_agx) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *offs =
         nir_imul_imm(b, nir_u2u32(b, intr->src[0].ssa), AGX_TEXTURE_LENGTH);
      nir_def_rewrite_uses(&intr->def,
                           nir_bindless_image_agx(b, offs, .desc_set = 0));
      return true;
   } else {
      return false;
   }
}

static bool
lower_tests_zs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *value = data;
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, *value ? 0xFF : 0, 16));
   return true;
}

static bool
lower_tests_zs(nir_shader *s, bool value)
{
   if (!s->info.fs.uses_discard)
      return false;

   return nir_shader_intrinsics_pass(s, lower_tests_zs_intr,
                                     nir_metadata_control_flow, &value);
}

static inline bool
blend_uses_2src(struct agx_blend_rt_key rt)
{
   assert(rt.advanced_blend == false);
   const struct agx_blend_standard blend = agx_unpack_blend_standard(rt.mode);

   enum pipe_blendfactor factors[] = {
      blend.rgb_src_factor,
      blend.rgb_dst_factor,
      blend.alpha_src_factor,
      blend.alpha_dst_factor,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      switch (factors[i]) {
      case PIPE_BLENDFACTOR_SRC1_COLOR:
      case PIPE_BLENDFACTOR_SRC1_ALPHA:
      case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
         return true;
      default:
         break;
      }
   }

   return false;
}

static void
copy_colour(nir_builder *b, const struct agx_fs_epilog_key *key,
            unsigned out_rt, unsigned in_loc, bool dual_src)
{
   unsigned size = (key->link.size_32 & BITFIELD_BIT(in_loc)) ? 32 : 16;

   nir_def *value =
      nir_load_exported_agx(b, 4, size, .base = AGX_ABI_FOUT_COLOUR(in_loc));

   if (key->link.loc0_w_1 && in_loc == 0) {
      value =
         nir_vector_insert_imm(b, value, nir_imm_floatN_t(b, 1.0, size), 3);
   }

   nir_store_output(b, value, nir_imm_int(b, 0),
                    .io_semantics.location = FRAG_RESULT_DATA0 + out_rt,
                    .io_semantics.dual_source_blend_index = dual_src,
                    .src_type = nir_type_float | size);
}

void
agx_nir_fs_epilog_agx2(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS epilog";

   /* First, construct a passthrough shader reading each colour and outputting
    * the value.
    */
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->remap); ++rt) {
      int location = key->remap[rt];

      /* Negative remaps indicate the attachment isn't written. */
      if (location >= 0 && key->link.loc_written & BITFIELD_BIT(location)) {
         copy_colour(b, key, rt, location, false);

         /* If this render target uses dual source blending, also copy the dual
          * source colour. While the copy_colour above is needed even for
          * missing attachments to handle alpha-to-coverage, this copy is only
          * for blending so should be suppressed for missing attachments to keep
          * the assert from blowing up on OpenGL.
          */
         if (!key->blend.rt[rt].advanced_blend &&
             blend_uses_2src(key->blend.rt[rt]) &&
             key->rt_formats[rt] != PIPE_FORMAT_NONE) {

            assert(location == 0);
            copy_colour(b, key, rt, 1, true);
         }
      }
   }

   /* Grab registers early, this has to happen in the first block. */
   nir_def *sample_id = NULL, *write_samples = NULL;
   if (key->link.sample_shading) {
      sample_id =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_SAMPLE_MASK);
   }

   if (key->link.sample_mask_after_force_early) {
      write_samples =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   }

   /* Now lower the resulting program using the key */
   struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(
      key->rt_formats, ARRAY_SIZE(key->rt_formats), key->nr_samples, true);

   if (key->force_small_tile)
      tib.tile_size = 16 * 16;

   bool force_translucent = false;
   nir_lower_blend_options opts = {
      .scalar_blend_const = true,
      .logicop_enable = key->blend.logicop_enable,
      .logicop_func = key->blend.logicop_func,
   };

   static_assert(ARRAY_SIZE(opts.rt) == 8, "max RTs out of sync");

   for (unsigned i = 0; i < 8; ++i) {
      if (key->blend.rt[i].advanced_blend) {
         const struct agx_blend_advanced blend =
            agx_unpack_blend_advanced(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->rt_formats[i],
            .advanced_blend = true,
            .colormask = key->blend.rt[i].colormask,
            .blend_mode = blend.op,
            .src_premultiplied = blend.src_premultiplied,
            .dst_premultiplied = blend.dst_premultiplied,
            .overlap = blend.overlap,
         };
      } else {
         const struct agx_blend_standard blend =
            agx_unpack_blend_standard(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->rt_formats[i],

            .rgb.src_factor = blend.rgb_src_factor,
            .rgb.dst_factor = blend.rgb_dst_factor,
            .rgb.func = blend.rgb_func,

            .alpha.src_factor = blend.alpha_src_factor,
            .alpha.dst_factor = blend.alpha_dst_factor,
            .alpha.func = blend.alpha_func,

            .colormask = key->blend.rt[i].colormask,
         };
      }
   }

   /* It's more efficient to use masked stores (with
    * agx_nir_lower_tilebuffer) than to emulate colour masking with
    * nir_lower_blend.
    */
   uint8_t colormasks[8] = {0};

   for (unsigned i = 0; i < 8; ++i) {
      if (key->rt_formats[i] == PIPE_FORMAT_NONE)
         continue;

      if (agx_tilebuffer_supports_mask(&tib, i)) {
         colormasks[i] = key->blend.rt[i].colormask;
         opts.rt[i].colormask = (uint8_t)BITFIELD_MASK(4);
      } else {
         colormasks[i] = (uint8_t)BITFIELD_MASK(4);
      }

      /* If not all bound RTs are fully written to, we need to force
       * translucent pass type. agx_nir_lower_tilebuffer will take
       * care of this for its own colormasks input.
       */
      unsigned comps = util_format_get_nr_components(key->rt_formats[i]);
      if ((opts.rt[i].colormask & BITFIELD_MASK(comps)) !=
          BITFIELD_MASK(comps)) {
         force_translucent = true;
      }
   }

   /* Alpha-to-coverage must be lowered before alpha-to-one */
   if (key->blend.alpha_to_coverage)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_coverage, false,
               NULL);

   /* Depth/stencil writes must be deferred until after all discards,
    * particularly alpha-to-coverage.
    */
   if (key->link.write_z || key->link.write_s) {
      nir_store_zs_agx(
         b, nir_imm_intN_t(b, 0xFF, 16),
         nir_load_exported_agx(b, 1, 32, .base = AGX_ABI_FOUT_Z),
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_S),
         .base = (key->link.write_z ? 1 : 0) | (key->link.write_s ? 2 : 0));

      if (key->link.write_z)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);

      if (key->link.write_s)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_STENCIL);
   }

   /* Alpha-to-one must be lowered before blending */
   if (key->blend.alpha_to_one)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_one);

   NIR_PASS(_, b->shader, nir_lower_blend, &opts);

   unsigned rt_spill = key->link.rt_spill_base;
   NIR_PASS(_, b->shader, agx_nir_lower_tilebuffer, &tib, colormasks, &rt_spill,
            write_samples, &force_translucent, false, false);
   NIR_PASS(_, b->shader, agx_nir_lower_texture);
   NIR_PASS(_, b->shader, agx_nir_lower_multisampled_image_store);

   /* If the API shader runs once per sample, then the epilog runs once per
    * sample as well, so we need to lower our code to run for a single sample.
    *
    * If the API shader runs once per pixel, then the epilog runs once per
    * pixel. So we run through the monolithic MSAA lowering, which wraps the
    * epilog in the sample loop if needed. This localizes sample shading
    * to the epilog, when sample shading is not used but blending is.
    */
   if (key->link.sample_shading) {
      /* Lower the resulting discards. Done in agx_nir_lower_monolithic_msaa_agx2 for
       * the pixel shaded path. Must be done before agx_nir_lower_to_per_sample_agx2
       * to avoid duplicating tests.
       */
      if (key->blend.alpha_to_coverage) {
         NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
      }

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample_agx2);
      NIR_PASS(_, b->shader, agx_nir_lower_fs_active_samples_to_register);

      /* Ensure the sample ID is preserved in register. We do this late since it
       * has to go in the last block, and the above passes might add control
       * flow when lowering.
       */
      b->cursor = nir_after_impl(b->impl);
      nir_export_agx(b, sample_id, .base = AGX_ABI_FIN_SAMPLE_MASK);
   } else if (!key->defer_monolithic_msaa) {
      NIR_PASS(_, b->shader, agx_nir_lower_monolithic_msaa_agx2, key->nr_samples);
   }

   /* Finally, lower uniforms according to our ABI */
   nir_shader_intrinsics_pass(b->shader, lower_non_monolithic_uniforms,
                              nir_metadata_control_flow, NULL);

   /* There is no shader part after the epilog, so we're always responsible for
    * running our own tests, unless the fragment shader forced early tests.
    */
   NIR_PASS(_, b->shader, lower_tests_zs, !key->link.already_ran_zs);

   b->shader->info.io_lowered = true;
   b->shader->info.fs.uses_fbfetch_output |= force_translucent;
   b->shader->info.fs.uses_sample_shading = key->link.sample_shading;
}
