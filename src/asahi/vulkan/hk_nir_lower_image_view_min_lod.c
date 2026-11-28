/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_image_view_min_lod.h"
#include "hk_native_descriptor.h"
#include "agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"

static bool
tex_needs_min_lod(const nir_tex_instr *tex)
{
   if (nir_tex_instr_is_query(tex))
      return false;

   /* Buffer textures don't have levels-of-detail */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;

   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   return true;
}

bool
hk_nir_needs_image_view_min_lod(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex &&
                tex_needs_min_lod(nir_instr_as_tex(instr)))
               return true;
         }
      }
   }
   return false;
}

static nir_def *
query_min_lod(nir_builder *b, nir_tex_instr *tex, bool int_coords)
{
   nir_alu_type T = int_coords ? nir_type_uint16 : nir_type_float16;
   return nir_build_texture_query(b, tex, nir_texop_image_min_lod_agx, 1, T,
                                  false, false);
}

static nir_def *
load_min_lod_post_desc(nir_builder *b, nir_tex_instr *tex, bool int_coords,
                       unsigned descriptor_generation)
{
   const int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle < 0)
      return NULL;

   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(descriptor_generation);
   assert(native.stride);

   nir_def *desc =
      nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa);
   /* Use 4-byte aligned 32-bit loads (matching lower_scaled_sample in
    * hk_shader.c) so narrow descriptor reads keep the folded uniform-base form:
    * min_lod (fp16) sits in the upper 16 bits of the 32-bit word at
    * native.min_lod - 2, and min_lod_uint (uint16) sits in the lower 16 bits
    * of the 32-bit word at native.min_lod_uint. */
   if (int_coords) {
      assert(((native.min_lod_uint - native.texture) & 3) == 0);
      nir_def *word = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, desc, native.min_lod_uint - native.texture),
         .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
      return nir_u2u16(b, word);
   } else {
      assert(((native.min_lod - 2 - native.texture) & 3) == 0);
      nir_def *word = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, desc, native.min_lod - 2 - native.texture),
         .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
      return nir_u2u16(b, nir_ushr_imm(b, word, 16));
   }
}

static bool
lower_min_lod_with(nir_builder *b, nir_tex_instr *tex,
                   unsigned descriptor_generation)
{
   if (!tex_needs_min_lod(tex))
      return false;

   bool int_coords = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                     tex->op == nir_texop_tg4;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *min_lod = descriptor_generation
                         ? load_min_lod_post_desc(b, tex, int_coords,
                                                  descriptor_generation)
                         : query_min_lod(b, tex, int_coords);
   if (!min_lod)
      return false;
   nir_def *other_min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

   if (int_coords) {
      assert(other_min_lod == NULL && "no API min lod for integer fetches");

      /* The descriptor stores max(floor(view minLod) - baseMipLevel, 0).
       * Integer Texel Coordinate Operations select baseMipLevel + Lod;
       * they do not clamp or offset Lod by minLod. Below the minimum,
       * robustImageAccess2 requires all components to be zero. Gathering
       * and multisample fetches implicitly use level zero.
       *
       * Keep this integer threshold out of nir_tex_src_min_lod, which is
       * floating point and describes a sampling clamp. Ordinary coordinate
       * and level robustness remains in the common texture lowering.
       */
      int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *oob = lod_idx >= 0
                       ? nir_ult(b, nir_u2u32(b, tex->src[lod_idx].src.ssa),
                                 nir_u2u32(b, min_lod))
                       : nir_ine_imm(b, min_lod, 0);
      b->cursor = nir_after_instr(&tex->instr);
      nir_def *old = &tex->def;
      nir_def *zero = nir_imm_zero(b, old->num_components, old->bit_size);
      nir_def *new_ = nir_bcsel(b, oob, zero, old);
      nir_def_rewrite_uses_after(old, new_);
   } else if (tex->op == nir_texop_txl) {
      assert(other_min_lod == NULL && "txl doesn't have an API min lod");

      nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
      if (lod) {
         min_lod = nir_fmax(b, nir_f2fN(b, lod, min_lod->bit_size), min_lod);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_lod, min_lod);
   } else {
      if (other_min_lod) {
         assert(!int_coords && "no API min lod");
         min_lod = nir_fmax(b, min_lod, nir_f2f16(b, other_min_lod));
      }

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, min_lod);
   }

   return true;
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   return lower_min_lod_with(b, tex, 0);
}

static bool
lower_min_lod_post_desc_cb(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const unsigned *gen = data;
   return lower_min_lod_with(b, tex, *gen);
}

bool
hk_nir_lower_image_view_min_lod(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower_min_lod, nir_metadata_none, NULL);
}

bool
hk_nir_lower_image_view_min_lod_post_desc(nir_shader *nir,
                                          unsigned descriptor_generation)
{
   return nir_shader_tex_pass(nir, lower_min_lod_post_desc_cb,
                              nir_metadata_none, &descriptor_generation);
}
