/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "util/format/u_format.h"
#include "agx_native_descriptor.h"
#include "agx_state.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * Lower binding table textures and images to texture state registers and (if
 * necessary) bindless access into an internal table mapped like additional
 * texture state registers. The following layout is used:
 *
 *    1. Textures
 *    2. Images (read/write interleaved; on G17P each image is followed by a
 *       software tail record, so images are options->image_records apart)
 */

static mesa_shader_stage
stage_for_descriptors(nir_shader *shader)
{
   mesa_shader_stage stage = shader->info.stage;

   if (stage == MESA_SHADER_VERTEX && shader->info.vs.tes_poly)
      stage = MESA_SHADER_TESS_EVAL;

   assert(stage < MESA_SHADER_STAGES);
   return stage;
}

static nir_def *
load_stage_array(nir_builder *b, nir_def *index, unsigned components,
                 unsigned bit_size, uintptr_t binding, unsigned stride,
                 unsigned align, unsigned count)
{
   const unsigned table = AGX_SYSVAL_STAGE(stage_for_descriptors(b->shader));
   nir_scalar scalar = nir_scalar_resolved(index, 0);

   if (nir_scalar_is_const(scalar)) {
      const unsigned slot = nir_scalar_as_uint(scalar);
      assert(slot < count);
      return nir_load_sysval_agx(b, components, bit_size, .desc_set = table,
                                 .binding = binding + (slot * stride));
   }

   /* Keep a bad dynamic GL index from escaping the fixed driver table. */
   index = nir_umin(b, index,
                    nir_imm_intN_t(b, count - 1,
                                   index->bit_size));

   struct agx_draw_uniforms *u = NULL;
   nir_def *table_base = nir_load_sysval_agx(
      b, 1, 64, .desc_set = AGX_SYSVAL_TABLE_ROOT,
      .binding = (uintptr_t)&u->tables[table]);
   nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, index, stride), binding);
   nir_def *address = nir_iadd(b, table_base, nir_u2u64(b, offset));
   return nir_load_global_constant(b, components, bit_size, address,
                                   .align_mul = align);
}

static nir_def *
sampler_index(nir_builder *b, nir_tex_instr *tex, bool native)
{
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset);
   if (offset_idx >= 0)
      return tex->src[offset_idx].src.ssa;

   const int handle_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (handle_idx >= 0) {
      nir_scalar handle =
         nir_scalar_chase_movs(nir_get_scalar(tex->src[handle_idx].src.ssa, 0));
      if (nir_scalar_is_alu(handle) &&
          nir_scalar_alu_op(handle) == nir_op_u2u64)
         handle = nir_scalar_chase_alu_src(handle, 0);

      nir_intrinsic_instr *load = nir_def_as_intrinsic_or_null(handle.def);
      if (handle.comp == 0 && load &&
          load->intrinsic == nir_intrinsic_load_sampler_handle_agx)
         return load->src[0].ssa;

      /* Texture lowering has already mapped a native API index i to i+1.
       * This also covers dynamic indices inside the immediate sampler bank,
       * which did not need load_sampler_handle_agx during binding lowering. */
      if (native)
         return nir_iadd_imm(b, nir_u2u16(b, tex->src[handle_idx].src.ssa), -1);
   }

   return nir_imm_int(b, tex->sampler_index);
}

static nir_def *
load_custom_border_sampler(nir_builder *b, nir_def *index)
{
   struct agx_stage_uniforms *s = NULL;
   return load_stage_array(
      b, index, 1, 16, (uintptr_t)&s->custom_border_sampler_handle,
      sizeof(s->custom_border_sampler_handle[0]), 2, PIPE_MAX_SAMPLERS);
}

static bool
lower_custom_border_query(nir_builder *b, nir_tex_instr *tex, bool native_index)
{
   if (tex->op != nir_texop_has_custom_border_color_agx &&
       tex->op != nir_texop_custom_border_color_agx)
      return false;

   nir_def *index = sampler_index(b, tex, native_index);
   nir_def *replacement;

   if (tex->op == nir_texop_has_custom_border_color_agx) {
      replacement = nir_ige_imm(b, load_custom_border_sampler(b, index), 0);
   } else {
      struct agx_stage_uniforms *s = NULL;
      nir_def *border = load_stage_array(
         b, index, 4, 32, (uintptr_t)&s->custom_border,
         sizeof(s->custom_border[0]), 4, PIPE_MAX_SAMPLERS);
      replacement = nir_convert_to_bit_size(
         b, border, nir_alu_type_get_base_type(tex->dest_type),
         tex->def.bit_size);
   }

   nir_def_replace(&tex->def, replacement);
   return true;
}

static bool
lower_sampler(nir_builder *b, nir_tex_instr *tex, bool wide_native_handle,
              bool clamp_to_zero, bool native_index, bool query_lod_bank)
{
   if (!nir_tex_instr_need_sampler(tex))
      return false;

   /* G18G uploads no G17P query-LOD bank; its LOD queries keep the ordinary
    * sampler handle. */

   /* Custom-border variants revisit binding lowering after the query's
    * indirect handle has already been built (and possibly optimized).
    * Preserve that handle, including a dynamic API sampler index. */
   if (query_lod_bank &&
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0)
      return false;

   nir_def *index = sampler_index(b, tex, native_index);
   nir_steal_tex_src(tex, nir_tex_src_sampler_offset);

   const bool noncompare =
      tex->backend_flags & AGX_TEXTURE_FLAG_BORDER_NONCOMPARE;

   if (clamp_to_zero || noncompare)
      nir_steal_tex_src(tex, nir_tex_src_sampler_handle);

   nir_def *handle = (clamp_to_zero || noncompare)
                       ? load_custom_border_sampler(b, index)
                       : nir_load_sampler_handle_agx(b, index);
   if (noncompare)
      handle = nir_iadd_imm(b, handle,
         AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE -
         AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE);
   if (query_lod_bank) {
      /* The ordinary table maps API i to native i+1. The query bank is
       * disjoint from both this bank and the custom-border companions.
       * G17G uploads no such bank, so it keeps the ordinary handle. */
      handle = nir_iadd_imm(b, handle, AGX_GALLIUM_G17P_QUERY_LOD_BASE - 1);
   }
   if (wide_native_handle)
      handle = nir_u2u64(b, handle);

   nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, handle);
   return true;
}

static bool
lower(nir_builder *b, nir_instr *instr, void *data)
{
   struct agx_lower_bindings_options *options = data;
   bool *uses_bindless_samplers = options->uses_bindless_samplers;
   bool progress = false;
   bool force_bindless = agx_nir_needs_texture_crawl(instr);
   if (options->g17p_plain_txf_crawl && instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      /* agx_nir_lower_texture() crawls a G17P integer fetch's descriptor to
       * predicate it: the explicit out-of-bounds path reads the size of a two-
       * or three-dimensional fetch, and the older robustness path reads the
       * descriptor for a fetch carrying a minimum level. A crawl needs the
       * descriptor, and only a bindless fetch has one, so every fetch that
       * lowering will crawl has to be made bindless here.
       *
       * agx_nir_needs_texture_crawl() already covers the shapes that crawl on
       * every chip -- a non-zero level, an array layer, a buffer texture --
       * so this adds the two that are particular to G17P. Keep it in step with
       * the conditions in agx_nir_lower_texture(): a fetch that reaches a
       * crawl without a descriptor asserts rather than falls back.
       */
      force_bindless |=
         tex->op == nir_texop_txf &&
         !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP) &&
         (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
          tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
          nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0);
   }
   b->cursor = nir_before_instr(instr);

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_atomic_swap:
      case nir_intrinsic_image_texel_address:
      case nir_intrinsic_load_image_block_size_agx:
         break;
      case nir_intrinsic_image_store_block_agx:
         if (!options->g17p_bindless_storage)
            return false;
         break;
      default:
         return false;
      }

      /* A buffer image's logical extent is not the width/height of its 2D
       * hardware view. AGX2 stores that extent in descriptor software bits;
       * Gen9 does not. Read the bound view's texel count from the stage table,
       * including for the bounds queries inserted by robustness lowering. */
      if (options->g17p_bindless_storage &&
          intr->intrinsic == nir_intrinsic_image_size &&
          nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF) {
         struct agx_stage_uniforms *s = NULL;
         nir_def *size = load_stage_array(
            b, intr->src[0].ssa, 1, 32, (uintptr_t)&s->image_buffer_size,
            sizeof(s->image_buffer_size[0]), 4, PIPE_MAX_SHADER_IMAGES);
         nir_def_replace(&intr->def, nir_u2uN(b, size, intr->def.bit_size));
         return true;
      }


      nir_def *index = intr->src[0].ssa;
      nir_scalar index_scalar = nir_scalar_resolved(index, 0);

      if (intr->intrinsic == nir_intrinsic_image_texel_address ||
          intr->intrinsic == nir_intrinsic_load_image_block_size_agx) {
         if (!options->image_address_mask)
            return false;

         STATIC_ASSERT(PIPE_MAX_SHADER_IMAGES <= 64);
         const unsigned count = b->shader->info.num_images;
         assert(count > 0 && count <= PIPE_MAX_SHADER_IMAGES);
         if (nir_scalar_is_const(index_scalar)) {
            const unsigned slot = nir_scalar_as_uint(index_scalar);
            assert(slot < count);
            *options->image_address_mask |= BITFIELD64_BIT(slot);
         } else {
            *options->image_address_mask |= BITFIELD64_MASK(count);
            index = nir_umin(b, index,
                            nir_imm_intN_t(b, count - 1, index->bit_size));
         }

         struct agx_stage_uniforms *s = NULL;
         struct agx_draw_uniforms *u = NULL;
         const unsigned table = AGX_SYSVAL_STAGE(stage_for_descriptors(b->shader));
         nir_def *table_base = nir_load_sysval_agx(
            b, 1, 64, .desc_set = AGX_SYSVAL_TABLE_ROOT,
            .binding = (uintptr_t)&u->tables[table]);
         nir_def *offset = nir_iadd_imm(
            b, nir_imul_imm(b, index, sizeof(s->image_address_metadata[0])),
            (uintptr_t)&s->image_address_metadata);
         nir_def *metadata = nir_iadd(b, table_base, nir_u2u64(b, offset));

         if (intr->intrinsic == nir_intrinsic_load_image_block_size_agx) {
            nir_def_replace(&intr->def, nir_load_global_constant(
               b, 1, 32, nir_iadd_imm(b, metadata, 36), .align_mul = 4));
            return true;
         }

         /* Even constant slots use ordinary loads. Pushing metadata for
          * every image can exhaust uniforms in legal image-heavy shaders. */
         nir_def *base = nir_load_global_constant(b, 4, 32, metadata,
                                                   .align_mul = 4);
         nir_def *layout = nir_load_global_constant(
            b, 4, 32, nir_iadd_imm(b, metadata, 16), .align_mul = 4);
         enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
         const bool msaa = dim == GLSL_SAMPLER_DIM_MS;
         nir_def *samples = msaa ? nir_load_global_constant(
            b, 1, 32, nir_iadd_imm(b, metadata, 32), .align_mul = 4) :
            nir_imm_int(b, 1);
         nir_def *bytes = nir_intrinsic_format(intr) == PIPE_FORMAT_NONE
            ? nir_load_global_constant(b, 1, 32, nir_iadd_imm(b, metadata, 36),
                                        .align_mul = 4)
            : nir_imm_int(b, util_format_get_blocksize(nir_intrinsic_format(intr)));
         const bool layered = nir_intrinsic_image_array(intr) ||
                              dim == GLSL_SAMPLER_DIM_CUBE ||
                              dim == GLSL_SAMPLER_DIM_3D;
         nir_def *address = libagx_bound_image_texel_address(
            b, base, layout, intr->src[1].ssa, samples,
            msaa ? intr->src[2].ssa : nir_imm_int(b, 0),
            bytes,
            nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D ||
                            dim == GLSL_SAMPLER_DIM_BUF),
            nir_imm_bool(b, layered));
         nir_def_replace(&intr->def, address);
         return true;
      }

      /* Robust buffer-image stores also issue imageSize queries. The native
       * descriptor has no AGX2 buffer_size_sw field to crawl. */
      if (options->native_buffer_metadata &&
          intr->intrinsic == nir_intrinsic_image_size &&
          nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF) {
         struct agx_stage_uniforms *s = NULL;
         nir_def *size = load_stage_array(
            b, index, 1, 32, (uintptr_t)&s->image_buffer_size,
            sizeof(s->image_buffer_size[0]), 4, PIPE_MAX_SHADER_IMAGES);
         nir_def_replace(&intr->def, nir_u2uN(b, size, intr->def.bit_size));
         return true;
      }

      /* Remap according to the driver layout */
      unsigned offset = BITSET_LAST_BIT(b->shader->info.textures_used);

      /* For reads and queries, we use the texture descriptor which is first.
       * Writes and atomics use the PBE descriptor.
       */
      if (intr->intrinsic != nir_intrinsic_image_load &&
          intr->intrinsic != nir_intrinsic_image_size &&
          intr->intrinsic != nir_intrinsic_image_samples) {
         offset++;

         if (options->image_pbe_mask) {
            STATIC_ASSERT(PIPE_MAX_SHADER_IMAGES <= 64);
            const unsigned count = b->shader->info.num_images;
            assert(count > 0 && count <= PIPE_MAX_SHADER_IMAGES);
            if (nir_scalar_is_const(index_scalar)) {
               const unsigned slot = nir_scalar_as_uint(index_scalar);
               assert(slot < count);
               *options->image_pbe_mask |= BITFIELD64_BIT(slot);
            } else {
               *options->image_pbe_mask |= BITFIELD64_MASK(count);
            }
         }
      }

      /* G17P has no texture state registers to name a PBE with: imgwci reads
       * its PBE from a heap -- a 64-bit uniform -- at a record index held in
       * a register. Bind stores and atomics through the stage texture table
       * so they carry that pair; agx_nir_lower_g17p_storage_records() turns
       * the byte offset into the record index once the table's uniform is
       * known.
       */
      if (options->g17p_bindless_storage &&
          ((intr->intrinsic != nir_intrinsic_image_load &&
            intr->intrinsic != nir_intrinsic_image_size &&
            intr->intrinsic != nir_intrinsic_image_samples) ||
           (intr->intrinsic == nir_intrinsic_image_load &&
            nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS)))
         force_bindless = true;

      /* If we can determine statically that the image fits in texture state
       * registers, avoid lowering to bindless access.
       */
      if (nir_scalar_is_const(index_scalar) && !force_bindless) {
         unsigned idx =
            (nir_scalar_as_uint(index_scalar) * options->image_records) +
            offset;

         if (idx < AGX_NUM_TEXTURE_STATE_REGS) {
            nir_src_rewrite(&intr->src[0], nir_imm_intN_t(b, idx, 16));
            return true;
         }
      }

      /* Otherwise, lower to bindless...
       *
       * The driver uploads enough null texture/PBE descriptors for robustness
       * given the shader limit, but we still need to clamp since we're lowering
       * to bindless so the hardware doesn't know the limit.
       *
       * The GL spec says out-of-bounds image indexing is undefined, but
       * faulting is not acceptable for robustness.
       */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_images - 1, index->bit_size));

      index = nir_iadd_imm(b, nir_imul_imm(b, index, options->image_records),
                           offset);

      nir_rewrite_image_intrinsic(intr, nir_load_texture_handle_agx(b, index),
                                  nir_image_intrinsic_type_bindless);
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (options->native_buffer_metadata &&
          (tex->op == nir_texop_txs || tex->op == nir_texop_txf) &&
          tex->sampler_dim == GLSL_SAMPLER_DIM_BUF &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) < 0) {
         struct agx_stage_uniforms *s = NULL;
         int offset = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
         nir_def *index = offset >= 0 ? tex->src[offset].src.ssa : NULL;
         index = index ? nir_iadd_imm(b, index, tex->texture_index)
                       : nir_imm_int(b, tex->texture_index);
         nir_def *size = load_stage_array(
            b, index, 1, 32, (uintptr_t)&s->texture_buffer_size,
            sizeof(s->texture_buffer_size[0]), 4,
            PIPE_MAX_SHADER_SAMPLER_VIEWS);
         if (tex->op == nir_texop_txs) {
            nir_def_replace(&tex->def, nir_u2uN(b, size, tex->def.bit_size));
            return true;
         }

         /* Buffer lowering generates another bounds query after this pass.
          * Carry the actual view metadata into that lowering, instead of
          * letting it decode software fields from a hardware descriptor. */
         nir_def *rgb32 = load_stage_array(
            b, index, 1, 32, (uintptr_t)&s->texture_buffer_rgb32,
            sizeof(s->texture_buffer_rgb32[0]), 4,
            PIPE_MAX_SHADER_SAMPLER_VIEWS);
         nir_def *address = load_stage_array(
            b, index, 1, 64, (uintptr_t)&s->texture_buffer_address,
            sizeof(s->texture_buffer_address[0]), 8,
            PIPE_MAX_SHADER_SAMPLER_VIEWS);
         assert(nir_tex_instr_src_index(tex, nir_tex_src_backend2) < 0);
         nir_tex_instr_add_src(
            tex, nir_tex_src_backend2,
            nir_vec4(b, size, rgb32,
                     nir_unpack_64_2x32_split_x(b, address),
                     nir_unpack_64_2x32_split_y(b, address)));
         tex->backend_flags |= AGX_TEXTURE_FLAG_BUFFER_METADATA;
         progress = true;
      }

      if (lower_custom_border_query(b, tex, options->g17p_shadow_gather))
         return true;

      const unsigned bindful_sampler_count =
         options->bindful_sampler_count ? options->bindful_sampler_count : 16;
      const bool dynamic_sampler =
         nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0;
      const bool clamp_to_zero =
         tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0;
      const bool noncompare =
         tex->backend_flags & AGX_TEXTURE_FLAG_BORDER_NONCOMPARE;
      const bool sampler_outside_bindful =
         tex->sampler_index >= bindful_sampler_count ||
         (dynamic_sampler &&
          BITSET_LAST_BIT(b->shader->info.samplers_used) >
             bindful_sampler_count);

      const bool query_lod = options->g17p_native_sampler_handles &&
                             !options->g18 && !options->table_index_crawls &&
                             tex->op == nir_texop_lod;
      if ((sampler_outside_bindful || clamp_to_zero || noncompare || query_lod) &&
          lower_sampler(b, tex, options->g17p_native_sampler_handles,
                        clamp_to_zero, options->g17p_shadow_gather,
                        query_lod)) {
         progress = true;
         if (!clamp_to_zero)
            *uses_bindless_samplers = true;
      }

      /* Nothing to do for "real" bindless */
      if (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0)
         return progress;

      /* Textures are mapped 1:1, so if we can prove it fits in a texture state
       * register, use the texture state register.
       */
      /* A G18G bindful texel fetch still needs the handle: its level/layer
       * clamp reads the descriptor (lower_regular_texture's txf robustness),
       * while the instruction keeps naming the texture state entry
       * (AGX_TEXTURE_FLAG_TABLE_INDEX, below). G17P and M5 make the fetches
       * they crawl bindless through g17p_plain_txf_crawl instead. */
      if (tex->texture_index < AGX_NUM_TEXTURE_STATE_REGS &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) == -1 &&
          !force_bindless && !(options->g18 && tex->op == nir_texop_txf))
         return progress;

      /* G17G's native fetches with a run-time level are bindful: the
       * instruction names the texture state entry and only the level and
       * layer clamps read the descriptor. Keep that shape here; the handle
       * added below feeds the crawl alone (AGX_TEXTURE_FLAG_TABLE_INDEX). */
      if (options->table_index_crawls &&
          tex->texture_index < AGX_NUM_TEXTURE_STATE_REGS &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) == -1)
         tex->backend_flags |= AGX_TEXTURE_FLAG_TABLE_INDEX;

      /* Otherwise, lower to bindless. Could be optimized. */
      nir_def *index = nir_steal_tex_src(tex, nir_tex_src_texture_offset);
      if (!index)
         index = nir_imm_int(b, tex->texture_index);

      /* As above */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_textures - 1, index->bit_size));

      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle,
                            nir_load_texture_handle_agx(b, index));
   }

   return true;
}

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex,
                                  nir_texop_has_custom_border_color_agx, 1,
                                  nir_type_bool1, false, false);
}

struct custom_border_options {
   bool g17g;
   /* G17P: shadow gathers through shadow_gather_border(). */
   bool shadow_gather;
   /* G18G: shadow gathers and filtered shadow samples through
    * custom_border_shadow_gather/filtered(); any other shadow operation
    * clears fully_supported. */
   bool g18;
   bool fully_supported;
};

static nir_tex_instr *
raw_depth_gather(nir_builder *b, const nir_tex_instr *tex, bool clamp_to_zero)
{
   nir_tex_instr *raw =
      nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   /* Source removal updates SSA use lists. Register the clone's uses before
    * removing its comparator, rather than moving unlinked sources. */
   nir_builder_instr_insert(b, &raw->instr);
   raw->is_shadow = false;
   raw->is_new_style_shadow = false;
   nir_tex_instr_remove_src(raw,
                           nir_tex_instr_src_index(raw, nir_tex_src_comparator));
   if (clamp_to_zero)
      raw->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;
   return raw;
}

static nir_def *
custom_border_shadow_gather(nir_builder *b, nir_tex_instr *tex,
                            nir_def *ordinary)
{
   /* Comparison results cannot identify the border: an interior depth can
    * compare identically to either native border value. Raw gathers with the
    * two companion samplers differ only on lanes outside the image. Compare
    * their bits so an interior NaN is not mistaken for the border. */
   nir_tex_instr *one = raw_depth_gather(b, tex, false);
   nir_tex_instr *zero = raw_depth_gather(b, tex, true);
   nir_def *interior = nir_ieq(b, &one->def, &zero->def);

   struct agx_stage_uniforms *s = NULL;
   nir_def *packed = load_stage_array(
      b, sampler_index(b, tex, false), 1, 16,
      (uintptr_t)&s->custom_border_compare,
      sizeof(s->custom_border_compare[0]), 2, PIPE_MAX_SAMPLERS);
   packed = nir_u2u32(b, packed);
   nir_def *func = nir_iand_imm(b, packed, 7);
   nir_def *fixed = nir_ine_imm(b, nir_iand_imm(b, packed, 8), 0);
   nir_def *ref = tex->src[nir_tex_instr_src_index(
      tex, nir_tex_src_comparator)].src.ssa;
   nir_def *depth = nir_channel(b, query_custom_border(b, tex), 0);

   /* OpenGL 4.6 section 8.23 clamps both operands for fixed-point depth;
    * floating-point depth preserves their full range. */
   ref = nir_bcsel(b, fixed, nir_fsat(b, ref), ref);
   depth = nir_bcsel(b, fixed, nir_fsat(b, depth), depth);
   nir_def *compared = nir_imm_false(b);
   for (unsigned f = PIPE_FUNC_LESS; f < PIPE_FUNC_ALWAYS; ++f) {
      nir_def *value = nir_compare_func(b, (enum compare_func)f, ref, depth);
      compared = nir_bcsel(b, nir_ieq_imm(b, func, f), value, compared);
   }
   compared = nir_ior(b, compared, nir_ieq_imm(b, func, PIPE_FUNC_ALWAYS));
   nir_def *border = nir_replicate(b, nir_b2f32(b, compared), 4);
   return nir_bcsel(b, interior, ordinary, border);
}

/* Keep the actual hardware filtering kernel. A fixed comparison reference
 * makes the border-zero and border-one probes differ only on border taps,
 * even when an interior floating depth is NaN or very large. */
static nir_def *
custom_border_compare_result(nir_builder *b, nir_def *func,
                             nir_def *ref, nir_def *depth)
{
   nir_def *result = nir_imm_false(b);
   for (unsigned f = PIPE_FUNC_LESS; f < PIPE_FUNC_ALWAYS; ++f) {
      nir_def *value = nir_compare_func(b, (enum compare_func)f, ref, depth);
      result = nir_bcsel(b, nir_ieq_imm(b, func, f), value, result);
   }
   return nir_ior(b, result, nir_ieq_imm(b, func, PIPE_FUNC_ALWAYS));
}

static nir_tex_instr *
custom_border_shadow_probe(nir_builder *b, const nir_tex_instr *tex,
                            nir_def *ref, bool clamp_to_zero)
{
   nir_tex_instr *probe = nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &probe->instr);
   int comparator = nir_tex_instr_src_index(probe, nir_tex_src_comparator);
   assert(comparator >= 0);
   nir_src_rewrite(&probe->src[comparator].src, ref);
   if (clamp_to_zero)
      probe->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;
   return probe;
}

static nir_def *
custom_border_shadow_filtered(nir_builder *b, nir_tex_instr *tex,
                               nir_def *ordinary)
{
   struct agx_stage_uniforms *s = NULL;
   nir_def *packed = load_stage_array(
      b, sampler_index(b, tex, false), 1, 16,
      (uintptr_t)&s->custom_border_compare,
      sizeof(s->custom_border_compare[0]), 2, PIPE_MAX_SAMPLERS);
   packed = nir_u2u32(b, packed);
   nir_def *func = nir_iand_imm(b, packed, 7);
   nir_def *fixed = nir_ine_imm(b, nir_iand_imm(b, packed, 8), 0);
   nir_def *ref = tex->src[nir_tex_instr_src_index(
      tex, nir_tex_src_comparator)].src.ssa;
   nir_def *depth = nir_f2fN(b, nir_channel(b, query_custom_border(b, tex), 0),
                            ref->bit_size);
   ref = nir_bcsel(b, fixed, nir_fsat(b, ref), ref);
   depth = nir_bcsel(b, fixed, nir_fsat(b, depth), depth);

   nir_def *equality = nir_ior(b, nir_ieq_imm(b, func, PIPE_FUNC_EQUAL),
                              nir_ieq_imm(b, func, PIPE_FUNC_NOTEQUAL));
   nir_def *probe_ref = nir_bcsel(b, equality,
      nir_imm_floatN_t(b, 0.0, ref->bit_size),
      nir_imm_floatN_t(b, 0.5, ref->bit_size));
   nir_tex_instr *one = custom_border_shadow_probe(b, tex, probe_ref, false);
   nir_tex_instr *zero = custom_border_shadow_probe(b, tex, probe_ref, true);
   nir_def *weight = nir_fabs(b, nir_fsub(b, &one->def, &zero->def));
   nir_def *custom = custom_border_compare_result(b, func, ref, depth);
   nir_def *native = custom_border_compare_result(
      b, func, ref, nir_imm_floatN_t(b, 1.0, ref->bit_size));
   nir_def *delta = nir_fsub(b, nir_b2fN(b, custom, ordinary->bit_size),
                              nir_b2fN(b, native, ordinary->bit_size));
   return nir_fsat(b, nir_ffma(b, weight, delta, ordinary));
}

static nir_def *
shadow_gather_border(nir_builder *b, nir_tex_instr *tex, nir_def *ordinary)
{
   /* Comparison results cannot identify border taps: for a reference outside
    * [0, 1], comparisons against both zero and one may be equal even though
    * the comparison against the API's floating-point border is different.
    * Gather raw depth with two non-comparing samplers instead. Interior taps
    * are bit-identical, border taps are zero versus one. */
   nir_def *probe[2];
   for (unsigned i = 0; i < 2; ++i) {
      nir_tex_instr *raw = nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
      raw->is_shadow = false;
      raw->is_new_style_shadow = false;
      raw->component = 0;
      nir_builder_instr_insert(b, &raw->instr);
      nir_steal_tex_src(raw, nir_tex_src_comparator);
      raw->backend_flags |= i ? AGX_TEXTURE_FLAG_BORDER_NONCOMPARE
                              : AGX_TEXTURE_FLAG_CLAMP_TO_0;
      probe[i] = &raw->def;
   }

   struct agx_stage_uniforms *s = NULL;
   nir_def *state = load_stage_array(
      b, sampler_index(b, tex, true), 1, 16,
      (uintptr_t)&s->custom_border_compare_func,
      sizeof(s->custom_border_compare_func[0]), 2, PIPE_MAX_SAMPLERS);
   nir_def *func = nir_iand_imm(b, state, 0xff);
   nir_def *clamp = nir_ine_imm(b, nir_iand_imm(b, state,
                                            AGX_BORDER_COMPARE_CLAMP_REF), 0);
   nir_def *ref = tex->src[nir_tex_instr_src_index(tex,
                                                nir_tex_src_comparator)].src.ssa;
   ref = nir_bcsel(b, clamp, nir_fsat(b, ref), ref);
   ref = nir_f2f32(b, ref);
   nir_def *border = nir_channel(b, nir_build_texture_query(
      b, tex, nir_texop_custom_border_color_agx, 4, nir_type_float32,
      false, false), 0);
   static_assert(PIPE_FUNC_LESS == (unsigned)COMPARE_FUNC_LESS, "compare order");
   static_assert(PIPE_FUNC_GEQUAL == (unsigned)COMPARE_FUNC_GEQUAL, "compare order");
   nir_def *compared = nir_ieq_imm(b, func, PIPE_FUNC_ALWAYS);
   for (unsigned f = PIPE_FUNC_LESS; f <= PIPE_FUNC_GEQUAL; ++f) {
      nir_def *value = nir_compare_func(b, (enum compare_func)f, ref, border);
      compared = nir_ior(b, compared,
                         nir_iand(b, nir_ieq_imm(b, func, f), value));
   }

   return nir_bcsel(b, nir_ieq(b, probe[0], probe[1]), ordinary,
                    nir_b2fN(b, compared, tex->def.bit_size));
}

static nir_tex_instr *
clone_tex(nir_builder *b, nir_tex_instr *tex, bool compare)
{
   nir_instr *instr = nir_instr_clone(b->shader, &tex->instr);
   nir_tex_instr *clone = nir_instr_as_tex(instr);
   /* Sources join their use lists on insertion. */
   nir_builder_instr_insert(b, instr);
   if (!compare) {
      nir_tex_instr_remove_src(
         clone, nir_tex_instr_src_index(clone, nir_tex_src_comparator));
      clone->is_shadow = false;
      clone->is_new_style_shadow = false;
      clone->component = 0;
   }
   return clone;
}

/*
 * The M5 has no custom border colour: the API sampler borders with opaque
 * white, so a depth comparison sees border depth 1.0 (gpu/gatherborder.c).
 * Its record leaves the comparison to the instruction, so a gather without
 * one returns the raw depths, the border's as 1.0, and one through the
 * clamp-to-zero bank returns the border's as 0.0: a texel whose two raw
 * depths differ is border. A border texel compares as
 * ref OP border == ref + (1 - border) OP 1.0, which a second comparing
 * gather with that reference computes against the opaque-white border
 * (dEQP-GLES31 texture.border_clamp.depth_compare_mode.depth32f_stencil8.
 * gather_size_*, whose float borders are not clamped).
 */
static bool
lower_custom_border_shadow_gather(nir_builder *b, nir_tex_instr *tex)
{
   const int comparator =
      nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   if (comparator < 0 || tex->def.num_components != 4)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);
   nir_def *ordinary = &clone_tex(b, tex, true)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced;
   {
      nir_def *raw_one = &clone_tex(b, tex, false)->def;
      nir_tex_instr *zero = clone_tex(b, tex, false);
      zero->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;
      nir_def *raw_zero = &zero->def;

      nir_def *ref = tex->src[comparator].src.ssa;
      nir_def *border = nir_channel(b, query_custom_border(b, tex), 0);
      if (border->bit_size != ref->bit_size)
         border = nir_f2fN(b, border, ref->bit_size);
      nir_def *shifted_ref = nir_fadd(b, ref, nir_fsub_imm(b, 1.0, border));

      nir_tex_instr *shifted_tex = clone_tex(b, tex, true);
      nir_src_rewrite(&shifted_tex->src[comparator].src, shifted_ref);
      nir_def *shifted = &shifted_tex->def;

      nir_def *comps[4];
      for (unsigned c = 0; c < 4; ++c) {
         nir_def *is_border = nir_fneu(b, nir_channel(b, raw_zero, c),
                                       nir_channel(b, raw_one, c));
         comps[c] = nir_bcsel(b, is_border, nir_channel(b, shifted, c),
                              nir_channel(b, ordinary, c));
      }
      replaced = nir_vec(b, comps, 4);
   }
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, replaced, ordinary);
   nir_def_replace(&tex->def, result);
   return true;
}

/* How the correctness-only custom border lowering treats depth-compare
 * gathers. G17P probes with raw comparison-disabled gathers; G17G's record
 * leaves the comparison to the instruction, so it shifts the reference
 * instead. */
struct custom_border_lowering {
   bool shadow_gathers;
   bool g17g_shadow_gathers;
};

static bool
lower_custom_border(nir_builder *b, nir_tex_instr *tex, void *data)
{
   struct custom_border_options *options = data;
   if (options->g17g && options->shadow_gather && tex->is_shadow &&
       tex->op == nir_texop_tg4 && nir_tex_instr_need_sampler(tex))
      return lower_custom_border_shadow_gather(b, tex);

   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex) ||
       (!options->g18 && tex->is_shadow &&
        (!options->shadow_gather || tex->op != nir_texop_tg4)))
      return false;

   const bool filtered_shadow = tex->op == nir_texop_tex ||
      tex->op == nir_texop_txb || tex->op == nir_texop_txl ||
      tex->op == nir_texop_txd;
   if (options->g18 && tex->is_shadow &&
       ((tex->op != nir_texop_tg4 && !filtered_shadow) ||
        nir_tex_instr_src_index(tex, nir_tex_src_comparator) < 0)) {
      /* Unknown shadow operations still need an explicit emulation path. */
      options->fully_supported = false;
      return false;
   }

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *ordinary_instr = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, ordinary_instr);
   nir_def *clamp_to_one = &nir_instr_as_tex(ordinary_instr)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced;
   if (tex->is_shadow) {
      if (options->g18) {
         replaced = tex->op == nir_texop_tg4
            ? custom_border_shadow_gather(b, tex, clamp_to_one)
            : custom_border_shadow_filtered(b, tex, clamp_to_one);
      } else {
         replaced = shadow_gather_border(b, tex, clamp_to_one);
      }
   } else {
      nir_instr *zero_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, zero_instr);
      nir_tex_instr *tex_zero = nir_instr_as_tex(zero_instr);
      nir_def *clamp_to_zero = &tex_zero->def;
      tex_zero->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;

      nir_def *border = query_custom_border(b, tex_zero);
      if (tex->op == nir_texop_tg4)
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);

      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {
         replaced = nir_flrp(b, clamp_to_zero, clamp_to_one, border);
         b->shader->info.flrp_lowered = false;
      } else {
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_zero, clamp_to_one),
                              clamp_to_zero, border);
      }
   }
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, replaced, clamp_to_one);
   nir_def_replace(&tex->def, result);
   return true;
}

bool
agx_nir_lower_custom_border_native(nir_shader *shader, bool shadow_gather,
                                   bool g18, bool *fully_supported)
{
   struct custom_border_options options = {
      .shadow_gather = shadow_gather,
      .g18 = g18,
      .fully_supported = true,
   };
   bool progress = nir_shader_tex_pass(shader, lower_custom_border,
                                       nir_metadata_none, &options);
   *fully_supported = options.fully_supported;

   if (progress) {
      bool uses_bindless_samplers = false;
      struct agx_lower_bindings_options options = {
         .uses_bindless_samplers = &uses_bindless_samplers,
         /* Native descriptor operands are represented by one 64-bit AIR
          * source, even though the value names a 16-bit table entry. */
         .g17p_native_sampler_handles = true,
         .g17p_shadow_gather = shadow_gather,
         .g18 = g18,
         .bindful_sampler_count = AGX_GALLIUM_G17P_API_SAMPLERS,
      };
      progress |= nir_shader_instructions_pass(
         shader, lower, nir_metadata_control_flow, &options);

      /* Binding lowering above can introduce descriptor-crawl handles again
       * for array samples. Texture coordinates are already lowered here, so
       * the sample itself must use the native G17P table index just as it does
       * after the initial texture-lowering pass.
       */
      if (!g18)
         progress |= agx_nir_rebind_g17p_texture_crawls(shader);

      /* Gallium variants apply this pass after the common AGX preprocess so
       * ordinary shaders do not pay for border emulation. Lower the small
       * amount of high-level NIR introduced above before handing the variant
       * to the backend. */
      NIR_PASS(_, shader, nir_lower_flrp, 16 | 32 | 64, false);
      NIR_PASS(_, shader, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(_, shader, nir_lower_all_phis_to_scalar);
      NIR_PASS(_, shader, nir_opt_dce);
   }

   return progress;
}

/* G17G has no comparison-disabled companion bank, so depth-compare gathers
 * use the reference-shifted form instead. The border sampling itself is the
 * same clamp-to-one/clamp-to-zero pair; the texture handles stay bindful for
 * the descriptor crawl (AGX_TEXTURE_FLAG_TABLE_INDEX). */
bool
agx_nir_lower_custom_border_g17g(nir_shader *shader)
{
   struct custom_border_options lowering = {
      .shadow_gather = true,
      .g17g = true,
      .fully_supported = true,
   };
   bool progress = nir_shader_tex_pass(shader, lower_custom_border,
                                       nir_metadata_none, &lowering);

   if (progress) {
      bool uses_bindless_samplers = false;
      struct agx_lower_bindings_options options = {
         .uses_bindless_samplers = &uses_bindless_samplers,
         .g17p_native_sampler_handles = true,
         .table_index_crawls = true,
         .bindful_sampler_count = AGX_GALLIUM_G17P_API_SAMPLERS,
      };
      progress |= nir_shader_instructions_pass(
         shader, lower, nir_metadata_control_flow, &options);

      /* Gallium variants apply this pass after the common AGX preprocess so
       * ordinary shaders do not pay for border emulation. Lower the small
       * amount of high-level NIR introduced above before handing the variant
       * to the backend. */
      NIR_PASS(_, shader, nir_lower_flrp, 16 | 32 | 64, false);
      NIR_PASS(_, shader, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(_, shader, nir_lower_all_phis_to_scalar);
      NIR_PASS(_, shader, nir_opt_dce);
   }

   return progress;
}

bool
agx_nir_lower_custom_border_g17p(nir_shader *shader, bool shadow_gather)
{
   bool fully_supported;
   return agx_nir_lower_custom_border_native(shader, shadow_gather, false,
                                             &fully_supported);
}

bool
agx_nir_lower_bindings(nir_shader *shader,
                       struct agx_lower_bindings_options *options)
{
   /* First lower index to offset so we can lower more naturally */
   bool progress = nir_lower_tex(
      shader, &(nir_lower_tex_options){.lower_index_to_offset = true});

   /* Next run constant folding so the constant optimizations above have a
    * chance.
    */
   progress |= nir_opt_constant_folding(shader);

   progress |= nir_shader_instructions_pass(
      shader, lower, nir_metadata_control_flow, options);
   return progress;
}

static nir_def *
load_g17p_texture_index(nir_builder *b, nir_def *index)
{
   struct agx_stage_uniforms *s = NULL;
   nir_scalar scalar = nir_scalar_resolved(index, 0);
   mesa_shader_stage stage = b->shader->info.stage;

   if (stage == MESA_SHADER_VERTEX && b->shader->info.vs.tes_poly)
      stage = MESA_SHADER_TESS_EVAL;

   assert(stage < MESA_SHADER_STAGES);
   const unsigned table = AGX_SYSVAL_STAGE(stage);

   if (nir_scalar_is_const(scalar)) {
      const unsigned slot = nir_scalar_as_uint(scalar);
      assert(slot < PIPE_MAX_SAMPLERS);
      return nir_u2u64(
         b, nir_load_sysval_agx(
               b, 1, 16, .desc_set = table,
               .binding = (uintptr_t)&s->texture_handle[slot]));
   }

   /* Dynamic Gallium sampler arrays select the matching texture table slot.
    * Load the stage-table address from the root, then index the compact
    * uint16_t identity table just like indirect sampler handles. */
   struct agx_draw_uniforms *u = NULL;
   nir_def *table_base = nir_load_sysval_agx(
      b, 1, 64, .desc_set = AGX_SYSVAL_TABLE_ROOT,
      .binding = (uintptr_t)&u->tables[table]);
   nir_def *offset = nir_iadd_imm(
      b, nir_imul_imm(b, index, sizeof(s->texture_handle[0])),
      (uintptr_t)&s->texture_handle);
   nir_def *address = nir_iadd(b, table_base, nir_u2u64(b, offset));
   return nir_u2u64(
      b, nir_load_global_constant(b, 1, 16, address, .align_mul = 2));
}

/*
 * G17G has no grounded texel fetch with a register texture index. An array
 * or cube image load is made bindless only for its layer clamp (a descriptor
 * crawl); once agx_nir_lower_texture() has made that crawl explicit, put a
 * statically bound load back on its immediate texture state index.
 */
static bool
rebind_g17g_image_load(nir_builder *b, nir_intrinsic_instr *intr,
                       UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_bindless_image_load)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   if (!handle || handle->intrinsic != nir_intrinsic_load_texture_handle_agx ||
       !nir_src_is_const(handle->src[0]) ||
       nir_src_as_uint(handle->src[0]) >= AGX_NUM_TEXTURE_STATE_REGS)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_rewrite_image_intrinsic(
      intr, nir_imm_intN_t(b, nir_src_as_uint(handle->src[0]), 16),
      nir_image_intrinsic_type_default);
   return true;
}

bool
agx_nir_rebind_g17g_image_loads(nir_shader *shader)
{
   bool progress = nir_shader_intrinsics_pass(
      shader, rebind_g17g_image_load, nir_metadata_control_flow, NULL);
   if (progress)
      NIR_PASS(_, shader, nir_opt_dce);
   return progress;
}

/*
 * A descriptor crawl needs an address, so binding lowering temporarily turns
 * an otherwise bindful texture into a bindless handle.  On AGX2 the texture
 * instruction itself can consume that base-plus-byte-offset pair.  G17P
 * cannot: its indirect operand is a native table index.
 *
 * Once texture lowering has made the crawl explicit, put a statically bound
 * access in the native texture table. Slots 0..15 use the immediate field;
 * larger or dynamically selected slots use a zero-extended native table
 * index. Keep the handle alive for independent
 * load_from_texture_handle_agx users which implement the Gen9 clamp. This
 * preserves the bounds operation without presenting AGX2's base-plus-byte-
 * offset pair to HAL200's fetch/sample instruction. Filtered array samples
 * need the same treatment because their layer clamp forces a crawl.
 */
bool
agx_nir_rebind_g17p_texture_crawls(nir_shader *shader)
{
   bool progress = false;

   /* Fold bound samplers' API index + 1 so only register samplers remain
    * non-constant. */
   NIR_PASS(progress, shader, nir_opt_constant_folding);

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);

            /* The cube pipeline has no form pairing an immediate texture with
             * a register sampler, which 2D has. A sampler past the bindful
             * range is a register, so index its texture the same way. */
            const int sampler_idx =
               nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
            const bool register_cube =
               tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && sampler_idx >= 0 &&
               !nir_src_is_const(tex->src[sampler_idx].src);

            const int handle_idx =
               nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
            if (handle_idx < 0) {
               if (register_cube &&
                   nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) <
                      0) {
                  b.cursor = nir_before_instr(instr);
                  nir_tex_instr_add_src(
                     tex, nir_tex_src_texture_offset,
                     load_g17p_texture_index(
                        &b, nir_imm_int(&b, tex->texture_index)));
                  tex->texture_index = 0;
                  impl_progress = true;
                  progress = true;
               }
               continue;
            }

            nir_intrinsic_instr *handle =
               nir_src_as_intrinsic(tex->src[handle_idx].src);
            if (!handle ||
                handle->intrinsic != nir_intrinsic_load_texture_handle_agx)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            if (!register_cube && nir_src_is_const(handle->src[0]) &&
                nir_src_as_uint(handle->src[0]) <
                   AGX_NUM_TEXTURE_STATE_REGS) {
               tex->texture_index = nir_src_as_uint(handle->src[0]);
            } else {
               tex->texture_index = 0;
               nir_tex_instr_add_src(
                  tex, nir_tex_src_texture_offset,
                  load_g17p_texture_index(&b, handle->src[0].ssa));
            }
            impl_progress = true;
            progress = true;
         }
      }

      if (impl_progress)
         nir_progress(true, impl, nir_metadata_control_flow);
   }

   if (progress)
      NIR_PASS(_, shader, nir_opt_dce);

   return progress;
}

/*
 * G17P's imgwci names its destination as a PBE heap -- a 64-bit uniform --
 * plus the index of a record in that heap, held in a register, and a bindless
 * image read names its texture the same way with a 64-bit index. Gallium's
 * bindless image handle is the stage texture table (the uniform) plus a byte
 * offset into it, and every descriptor in that table, texture or PBE, is one
 * record wide. Divide the offset by the record size for the stores that reach
 * the backend. This is the Gallium side of hk_g17p_lower_storage_records().
 *
 * It must run after the sysval layout has assigned the table's uniform to the
 * handle, and after texture lowering: atomics and multisampled stores are
 * rewritten there into texel addressing that reads the descriptor through the
 * byte-offset handle, so only genuine hardware accesses remain. Each access
 * gets its own handle rather than rewriting a shared one, so any remaining
 * descriptor reader keeps the byte offset.
 */
bool
agx_nir_lower_g17p_storage_records(nir_shader *shader, unsigned record_size_B)
{
   assert(util_is_power_of_two_nonzero(record_size_B));
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool load =
               intr->intrinsic == nir_intrinsic_bindless_image_load ||
               intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;
            if (intr->intrinsic != nir_intrinsic_bindless_image_store &&
                intr->intrinsic != nir_intrinsic_bindless_image_store_block_agx &&
                !load)
               continue;

            nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
            if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def *record = nir_ushr_imm(&b, handle->src[0].ssa,
                                           util_logbase2(record_size_B));
            /* A read names its texture descriptor the way a G17P texture
             * fetch does (agx_nir_rebind_g17p_texture_crawls()): a native
             * table index, zero-extended to the 64-bit register operand. */
            if (load)
               record = nir_u2u64(&b, record);
            nir_src_rewrite(&intr->src[0],
                            nir_bindless_image_agx(
                               &b, record,
                               .desc_set = nir_intrinsic_desc_set(handle)));
            impl_progress = true;
         }
      }

      if (impl_progress) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      }
   }

   return progress;
}
