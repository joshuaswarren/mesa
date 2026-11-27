/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Offline reproduction of the native-Gallium G17P blit crash, under the
 * no-op DRM shim -- no GPU, no window system, no submission.
 *
 * Core 983 caught the driver dereferencing a null `ctx->vs->bo` inside
 * agx_draw_vbo(), reached from util_blitter_blit() for a 4x8
 * BGRA8_UNORM -> RGBA8_UNORM region with uses_txf. The cause is that the AGX3
 * backend is fail-closed: agx_compile_shader_nir() returns false and zeroes
 * its output for a shader it cannot encode, so agx_compile_nir() produced a
 * compiled shader with binary_size == 0 and therefore no executable BO, and
 * the draw path used it anyway.
 *
 * Two things are asserted here, and they are separable:
 *
 *   1. The blit must never take the process down. A refused shader is a
 *      driver gap; it must surface as a skipped draw, not a SIGSEGV. This
 *      holds whether or not the shaders encode, and is the regression that
 *      guards the failure-propagation path.
 *
 *   2. With AGX_G17P_BLIT_REQUIRE_ENCODE=1, every shader the blit needs must
 *      actually encode. That is the stronger claim, and it is checked
 *      separately so a compiler regression cannot hide behind (1).
 *
 * The chip comes from ASAHI_SHIM_CHIP, so the same binary runs the G13G
 * control and the G17P case.
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drm-uapi/drm_fourcc.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/u_memory.h"
#include "util/driconf.h"
#include "util/xmlconfig.h"

#include "asahi/compiler/agx_compile.h"
#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx3_cdm_decode_core.h"
#include "asahi/lib/agx_tile_mip.h"
#include "asahi/lib/agx_chip_caps.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "compiler/glsl_types.h"

#include "asahi/vulkan/hk_g17p_meta_copy_core.h"
#include "agx_native_descriptor.h"
#include "agx_public.h"
#include "agx_state.h"

static bool
check_g17p_sampler_bias_record(void)
{
   /* applegpu-nt's HAL200 record for a compare-enabled sampler with a
    * nonzero bias. This is the native evidence that G17P consumes sampler
    * state bias in the descriptor rather than requiring nir_lower_tex to add
    * a second per-instruction bias. The heap stride is 32 bytes; only the
    * first eight are populated. */
   static const uint8_t expected[32] = {
      0x40, 0x00, 0xbe, 0x30, 0x5a, 0xfc, 0x3d, 0xc0,
   };
   uint8_t record[32] = {0};
   agx_pack(record, SAMPLER_HAL200, cfg) {
      cfg.minimum_lod = 1.0f;
      cfg.maximum_lod = 100.0f;
      cfg.lod_bias = -1.015625f;
      cfg.maximum_anisotropy = 8;
      cfg.magnify = AGX_FILTER_LINEAR;
      cfg.minify = AGX_FILTER_NEAREST;
      cfg.mip_filter = AGX_MIP_FILTER_LINEAR;
      cfg.wrap_s = AGX_WRAP_REPEAT;
      cfg.wrap_t = AGX_WRAP_MIRRORED_REPEAT;
      cfg.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
      cfg.compare_func = AGX_COMPARE_FUNC_LEQUAL;
      cfg.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
      cfg.pixel_coordinates = true;
      cfg.compare_enable = true;
      cfg.seamful_cube_maps = true;
   }

   if (memcmp(record, expected, sizeof(record)) != 0) {
      fprintf(stderr, "G17P HAL200 sampler bias record changed\n");
      return false;
   }
   return true;
}

static nir_shader *
build_es3_bitops_txf(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P ES3 bit operations plus texelFetch fragment shader");

   nir_def *frag = nir_load_frag_coord(&b);
   nir_def *word = nir_f2u32(&b, nir_channel(&b, frag, 0));
   nir_def *reversed = nir_bitfield_reverse(&b, word);
   nir_def *count = nir_bit_count(&b, reversed);
   nir_def *top = nir_ufind_msb(&b, nir_ior_imm(&b, reversed, 1));
   nir_def *lod = nir_iand_imm(&b, nir_iadd(&b, count, top), 3);
   nir_def *coord = nir_f2i32(&b, nir_channels(&b, frag, 0x3));

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def_init(&tex->instr, &tex->def, 4, 32);

   /* Keep each integer operation live in the colour. This is the GLSL ES
    * 3.x/WebGL2 family behind bitfieldReverse(), bitCount(), findMSB() and a
    * dynamic texelFetch() level, rather than four independent dead probes. */
   nir_def *bit_value = nir_iand_imm(
      &b, nir_ixor(&b, reversed, nir_ishl_imm(&b, count, 3)), 0xff);
   nir_def *scale = nir_fmul_imm(&b, nir_u2f32(&b, bit_value), 1.0f / 255.0f);
   /* This exact f16 -> f32 -> multiply chain is the remaining ext.frag
    * failure from Hyprland's built-in corpus. The widening must stay explicit
    * through the backend optimizer when the post-RA mixed multiply cannot
    * carry the combined producer wait and lifetime hints. */
   nir_def *rounded_half = nir_f2f16(&b, nir_channel(&b, &tex->def, 3));
   nir_def *mixed_product = nir_fmul(
      &b, nir_f2f32(&b, rounded_half), nir_fadd_imm(&b, scale, 1.0f));
   nir_def *colour = nir_fadd(
      &b, &tex->def,
      nir_vec4(&b, scale, scale, scale, mixed_product));
   nir_store_output(&b, colour, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);

   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   BITSET_SET(b.shader->info.textures_used, 0);
   BITSET_SET(b.shader->info.samplers_used, 0);
   return b.shader;
}

static nir_shader *
build_webgl_shadow_fs(enum glsl_sampler_dim dim, bool array)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      dim == GLSL_SAMPLER_DIM_CUBE
         ? (array ? "G17P WebGL cube-array shadow fragment shader"
                  : "G17P WebGL cube shadow fragment shader")
         : (array ? "G17P WebGL 2D-array shadow fragment shader"
                  : "G17P WebGL 2D shadow fragment shader"));
   nir_def *frag = nir_load_frag_coord(&b);

   /* Keep every desktop/WebGL comparison dimension in one API-realistic
    * family. All four forms are byte-grounded on G17P. */
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_tex;
   tex->sampler_dim = dim;
   tex->is_array = array;
   tex->is_shadow = true;
   tex->dest_type = nir_type_float32;
   const unsigned coord_components =
      dim == GLSL_SAMPLER_DIM_CUBE ? (array ? 4 : 3) : (array ? 3 : 2);
   tex->coord_components = coord_components;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_fmul_imm(&b, nir_channels(&b, frag,
                                    (nir_component_mask_t)BITFIELD_MASK(coord_components)),
                   1.0f / 64.0f));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                     nir_channel(&b, frag, 2));
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def_init(&tex->instr, &tex->def, 4, 32);

   nir_def *colour = nir_replicate(&b, nir_channel(&b, &tex->def, 0), 4);
   nir_store_output(&b, colour, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   BITSET_SET(b.shader->info.textures_used, 0);
   BITSET_SET(b.shader->info.samplers_used, 0);
   return b.shader;
}

static nir_shader *
build_sampler_boundary_fs(unsigned sampler_index)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P native sampler-table boundary fragment shader");
   nir_def *frag = nir_load_frag_coord(&b);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = sampler_index;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_fmul_imm(&b, nir_channels(&b, frag, 0x3), 1.0f / 64.0f));
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def_init(&tex->instr, &tex->def, 4, 32);

   nir_store_output(&b, &tex->def, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   BITSET_SET(b.shader->info.textures_used, 0);
   BITSET_SET(b.shader->info.samplers_used, sampler_index);
   return b.shader;
}

static nir_shader *
build_dynamic_sampler_boundary_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P dynamic native sampler-table fragment shader");
   nir_def *frag = nir_load_frag_coord(&b);
   nir_def *sampler = nir_iand_imm(
      &b, nir_f2u32(&b, nir_channel(&b, frag, 0)),
      AGX_GALLIUM_G17P_API_SAMPLERS - 1);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_fmul_imm(&b, nir_channels(&b, frag, 0x3), 1.0f / 64.0f));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_offset, sampler);
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_sampler_offset, sampler);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def_init(&tex->instr, &tex->def, 4, 32);

   nir_store_output(&b, &tex->def, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   for (unsigned i = 0; i < AGX_GALLIUM_G17P_API_SAMPLERS; ++i) {
      BITSET_SET(b.shader->info.textures_used, i);
      BITSET_SET(b.shader->info.samplers_used, i);
   }
   return b.shader;
}

static bool
check_sampler_boundary_compile(struct pipe_context *pctx)
{
   static const unsigned samplers[] = {
      AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS - 1,
      AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS,
      AGX_GALLIUM_G17P_API_SAMPLERS - 1,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(samplers); ++i) {
      struct pipe_shader_state cso = {
         .type = PIPE_SHADER_IR_NIR,
         .report_compile_error = true,
         .ir.nir = build_sampler_boundary_fs(samplers[i]),
      };
      struct agx_uncompiled_shader *shader =
         pctx->create_fs_state(pctx, &cso);
      agx_shader_wait_ready(shader);
      const bool indirect =
         samplers[i] >= AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS;

      if (!shader || cso.error_message ||
          shader->uses_bindless_samplers != indirect) {
         fprintf(stderr,
                 "G17P API sampler %u did not take the %s native-table "
                 "path: %s\n",
                 samplers[i], indirect ? "indirect" : "immediate",
                 cso.error_message ? cso.error_message : "no shader");
         free(cso.error_message);
         if (shader)
            pctx->delete_fs_state(pctx, shader);
         return false;
      }

      pctx->delete_fs_state(pctx, shader);
      free(cso.error_message);
   }

   struct pipe_shader_state dynamic_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .report_compile_error = true,
      .ir.nir = build_dynamic_sampler_boundary_fs(),
   };
   struct agx_uncompiled_shader *dynamic =
      pctx->create_fs_state(pctx, &dynamic_cso);
   agx_shader_wait_ready(dynamic);
   if (!dynamic || dynamic_cso.error_message ||
       !dynamic->uses_bindless_samplers) {
      fprintf(stderr, "G17P dynamic sampler table did not compile: %s\n",
              dynamic_cso.error_message ? dynamic_cso.error_message
                                        : "no shader");
      free(dynamic_cso.error_message);
      if (dynamic)
         pctx->delete_fs_state(pctx, dynamic);
      return false;
   }
   pctx->delete_fs_state(pctx, dynamic);
   free(dynamic_cso.error_message);

   return true;
}

static bool
check_query_lod_custom_border_bindings(void)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      nir_shader *nir = build_sampler_boundary_fs(0);
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
      nir_def *frag = nir_load_frag_coord(&b);
      nir_def *dynamic = nir_iand_imm(
         &b, nir_f2u32(&b, nir_channel(&b, frag, 0)), 31);
      nir_tex_instr *query = nir_tex_instr_create(nir, mode == 2 ? 2 : 1);
      query->op = nir_texop_lod;
      query->sampler_dim = GLSL_SAMPLER_DIM_2D;
      query->dest_type = nir_type_float32;
      query->coord_components = 2;
      query->sampler_index = mode == 1 ? 31 : 0;
      query->src[0] = nir_tex_src_for_ssa(
         nir_tex_src_coord, nir_channels(&b, frag, 3));
      if (mode == 2)
         query->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_offset, dynamic);
      nir_builder_instr_insert(&b, &query->instr);
      nir_def_init(&query->instr, &query->def, 2, 32);
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *sum = nir_fadd(&b, intr->src[0].ssa,
                                    nir_channel(&b, &query->def, 0));
            nir_src_rewrite(&intr->src[0], sum);
         }
      }
      BITSET_SET(nir->info.samplers_used, mode == 0 ? 0 : 31);
      bool bindless = false;
      struct agx_lower_bindings_options options = {
         .uses_bindless_samplers = &bindless,
         .g17p_native_sampler_handles = true,
         .bindful_sampler_count = AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS,
      };
      agx_nir_lower_bindings(nir, &options);
      const int handle_idx =
         nir_tex_instr_src_index(query, nir_tex_src_sampler_handle);
      assert(handle_idx >= 0);
      nir_def *handle = query->src[handle_idx].src.ssa;
      const bool lowered = agx_nir_lower_custom_border_g17p(nir, true);
      unsigned count = 0;
      for (unsigned i = 0; i < query->num_srcs; ++i) {
         if (query->src[i].src_type == nir_tex_src_sampler_handle) {
            ++count;
            if (query->src[i].src.ssa != handle)
               count += 100;
         }
      }
      const bool ok = lowered && bindless && count == 1;
      if (!ok)
         fprintf(stderr, "G17P query/custom-border mode %u changed query handle "
                         "or duplicated source (%u)\n", mode, count);
      if (ok)
         nir_validate_shader(nir, "query LOD with custom border");
      ralloc_free(nir);
      if (!ok)
         return false;
   }
   return true;
}

static bool
check_query_lod_compile(struct pipe_context *pctx)
{
   static const enum glsl_sampler_dim dimensions[] = {
      GLSL_SAMPLER_DIM_1D, GLSL_SAMPLER_DIM_2D,
      GLSL_SAMPLER_DIM_3D,
   };

   for (unsigned d = 0; d < ARRAY_SIZE(dimensions); ++d) {
      for (unsigned mode = 0; mode < 3; ++mode) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options,
            "G17P biased query LOD sampler bank");
         const unsigned components = d == 0 ? 1 : d == 1 ? 2 : 3;
         nir_def *frag = nir_load_frag_coord(&b);
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, mode == 2 ? 2 : 1);
         tex->op = nir_texop_lod;
         tex->sampler_dim = dimensions[d];
         tex->dest_type = nir_type_float32;
         tex->coord_components = components;
         tex->sampler_index = mode == 1 ? 31 : 0;
         tex->src[0] = nir_tex_src_for_ssa(
            nir_tex_src_coord, nir_trim_vector(&b, frag, components));
         if (mode == 2) {
            tex->src[1] = nir_tex_src_for_ssa(
               nir_tex_src_sampler_offset,
               nir_iand_imm(&b, nir_f2u32(&b, nir_channel(&b, frag, 0)), 31));
         }
         nir_builder_instr_insert(&b, &tex->instr);
         nir_def_init(&tex->instr, &tex->def, 2, 32);
         nir_def *result = nir_vec4(&b, nir_channel(&b, &tex->def, 0),
                                   nir_channel(&b, &tex->def, 1),
                                   nir_imm_float(&b, 0), nir_imm_float(&b, 1));
         nir_store_output(&b, result, nir_imm_int(&b, 0), .base = 0,
                          .write_mask = 0xf,
                          .io_semantics.location = FRAG_RESULT_DATA0);
         b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
         BITSET_SET(b.shader->info.textures_used, 0);
         BITSET_SET(b.shader->info.samplers_used, mode == 0 ? 0 : 31);
         struct pipe_shader_state cso = {
            .type = PIPE_SHADER_IR_NIR,
            .report_compile_error = true,
            .ir.nir = b.shader,
         };
         struct agx_uncompiled_shader *shader = pctx->create_fs_state(pctx, &cso);
         agx_shader_wait_ready(shader);
         const bool ok = shader && !cso.error_message &&
                         shader->info.uses_query_lod &&
                         shader->uses_bindless_samplers;
         if (!ok)
            fprintf(stderr, "G17P query LOD dimension %u mode %u: %s\n",
                    dimensions[d], mode,
                    cso.error_message ? cso.error_message : "missing query bank");
         if (shader)
            pctx->delete_fs_state(pctx, shader);
         free(cso.error_message);
         if (!ok)
            return false;
      }
   }
   return true;
}

static bool
check_es3_compile_contract(struct pipe_context *pctx)
{
   struct pipe_shader_state accepted = {
      .type = PIPE_SHADER_IR_NIR,
      .report_compile_error = true,
      .ir.nir = build_es3_bitops_txf(),
   };
   void *ok = pctx->create_fs_state(pctx, &accepted);
   if (!ok || accepted.error_message) {
      fprintf(stderr, "ES3 bitops/texelFetch shader did not compile: %s\n",
              accepted.error_message ? accepted.error_message : "no shader");
      free(accepted.error_message);
      if (ok)
         pctx->delete_fs_state(pctx, ok);
      return false;
   }
   pctx->delete_fs_state(pctx, ok);

   static const struct {
      enum glsl_sampler_dim dim;
      bool array;
      const char *name;
   } comparisons[] = {
      {GLSL_SAMPLER_DIM_2D, false, "sampler2DShadow"},
      {GLSL_SAMPLER_DIM_2D, true, "sampler2DArrayShadow"},
      {GLSL_SAMPLER_DIM_CUBE, false, "samplerCubeShadow"},
      {GLSL_SAMPLER_DIM_CUBE, true, "samplerCubeArrayShadow"},
   };

   bool compiled = true;
   for (unsigned i = 0; i < ARRAY_SIZE(comparisons); ++i) {
      struct pipe_shader_state shadow = {
         .type = PIPE_SHADER_IR_NIR,
         .report_compile_error = true,
         .ir.nir = build_webgl_shadow_fs(comparisons[i].dim,
                                         comparisons[i].array),
      };
      void *shader = pctx->create_fs_state(pctx, &shadow);
      if (!shader || shadow.error_message) {
         fprintf(stderr, "WebGL2 %s did not compile: %s\n",
                 comparisons[i].name,
                 shadow.error_message ? shadow.error_message : "no shader");
         compiled = false;
      }
      if (shader)
         pctx->delete_fs_state(pctx, shader);
      free(shadow.error_message);
   }
   return compiled;
}

static nir_shader *
build_image_store_cs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared storage image store test");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_images = 1;
   BITSET_SET(b.shader->info.images_used, 0);

   nir_image_store(&b, nir_imm_int(&b, 0),
                   nir_imm_ivec4(&b, 0, 0, 0, 0),
                   nir_imm_int(&b, 0),
                   nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f),
                   nir_imm_int(&b, 0),
                   .image_dim = GLSL_SAMPLER_DIM_2D,
                   .format = PIPE_FORMAT_R8G8B8A8_UNORM,
                   .access = ACCESS_NON_READABLE);
   return b.shader;
}

static bool
has_rejected_active_batch(struct agx_context *ctx)
{
   unsigned idx;
   BITSET_FOREACH_SET(idx, ctx->batches.active, AGX_MAX_BATCHES) {
      if (ctx->batches.slots[idx].native_rejected)
         return true;
   }
   return false;
}

static bool
check_interchange_and_shared_bo(struct pipe_context *pctx)
{
   struct pipe_screen *screen = pctx->screen;
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(screen);
   bool ok = true;

   const bool want_ix =
      agx_chip_has_cap(dev->chip, AGX_CAP_GEN9_2D_COMPRESSION);
   bool ext_only = true;
   const bool ix_bgra = screen->is_dmabuf_modifier_supported(
      screen, DRM_FORMAT_MOD_APPLE_INTERCHANGE, PIPE_FORMAT_B8G8R8A8_UNORM,
      &ext_only);
   if (ix_bgra != want_ix || (ix_bgra && ext_only)) {
      fprintf(stderr,
              "interchange BGRA8 support=%d ext_only=%d on %s (want %d, 0)\n",
              ix_bgra, ext_only, agx_chip_name(dev->chip), want_ix);
      ok = false;
   }

   ext_only = false;
   const bool comp_ok = screen->is_dmabuf_modifier_supported(
      screen, DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
      PIPE_FORMAT_B8G8R8A8_UNORM, &ext_only);
   const bool want_comp_ext = !agx_gallium_shared_compression_supported(
      agx_chip_descriptor_generation(dev->chip));
   if (!comp_ok || ext_only != want_comp_ext) {
      fprintf(stderr,
              "GPU_TILED_COMPRESSED support=%d ext_only=%d on %s (want 1, %d)\n",
              comp_ok, ext_only, agx_chip_name(dev->chip), want_comp_ext);
      ok = false;
   }

   const uint64_t display_mods[] = {
      DRM_FORMAT_MOD_APPLE_INTERCHANGE,
      DRM_FORMAT_MOD_LINEAR,
   };
   struct pipe_resource ix_templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_B8G8R8A8_UNORM,
      .width0 = 256,
      .height0 = 256,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_SCANOUT | PIPE_BIND_RENDER_TARGET |
              PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *ix = screen->resource_create_with_modifiers(
      screen, &ix_templ, display_mods, ARRAY_SIZE(display_mods));
   bool rt_rejected = false;
   bool comp_cs_rejected = false;
   if (!ix) {
      fprintf(stderr, "failed to allocate scanout image on %s\n",
              agx_chip_name(dev->chip));
      ok = false;
   } else {
      struct agx_resource *ix_rsrc = agx_resource(ix);
      const uint64_t want_mod = want_ix ? DRM_FORMAT_MOD_APPLE_INTERCHANGE
                                        : DRM_FORMAT_MOD_LINEAR;
      if (ix_rsrc->modifier != want_mod) {
         fprintf(stderr, "scanout modifier %#llx on %s (want %#llx)\n",
                 (unsigned long long)ix_rsrc->modifier,
                 agx_chip_name(dev->chip), (unsigned long long)want_mod);
         ok = false;
      }
      if (want_ix) {
         struct agx_bo *orig_bo = ix_rsrc->bo;
         if (!agx_resource_is_shared(ix_rsrc) || !ix_rsrc->layout.compressed) {
            fprintf(stderr,
                    "interchange scanout image not shared+compressed\n");
            ok = false;
         }
         /* Incompatible format view and direct agx_decompress() must never swap
          * the shared BO. */
         agx_legalize_compression(ctx, ix_rsrc, PIPE_FORMAT_R32_UINT);
         agx_decompress(ctx, ix_rsrc, "test shared");
         if (ix_rsrc->bo != orig_bo || !ix_rsrc->layout.compressed) {
            fprintf(stderr,
                    "shared interchange BO was reallocated on decompress path\n");
            ok = false;
         }

         /* Full-extent clear on compressed Interchange scanout must stay on
          * VDM+PBE and never convert to libagx_fast_clear() CDM. */
         pctx->flush(pctx, NULL, 0);
         struct pipe_framebuffer_state ix_fb = {
            .width = 256,
            .height = 256,
            .layers = 1,
            .samples = 1,
            .nr_cbufs = 1,
            .cbufs[0] = {.texture = ix, .format = PIPE_FORMAT_B8G8R8A8_UNORM},
         };
         pctx->set_framebuffer_state(pctx, &ix_fb);
         const union pipe_color_union ix_clear_col = {
            .f = {0.25f, 0.5f, 0.75f, 1.0f},
         };
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &ix_clear_col, 0.0,
                     0);
         struct agx_batch *ix_fc_batch = ctx->batch;
         if (!ix_fc_batch || agx_try_fast_clear_batch(ctx, ix_fc_batch) ||
             ix_fc_batch->vdm.bo == NULL || ix_fc_batch->cdm.bo != NULL ||
             has_rejected_active_batch(ctx)) {
            fprintf(stderr,
                    "interchange BGRA8 clear unexpectedly took fast-clear or "
                    "rejected (batch=%p vdm=%p cdm=%p rejected=%d)\n",
                    (void *)ix_fc_batch,
                    ix_fc_batch ? (void *)ix_fc_batch->vdm.bo : NULL,
                    ix_fc_batch ? (void *)ix_fc_batch->cdm.bo : NULL,
                    has_rejected_active_batch(ctx));
            ok = false;
         }
         pctx->flush(pctx, NULL, 0);
         struct pipe_framebuffer_state ix_empty_fb = {0};
         pctx->set_framebuffer_state(pctx, &ix_empty_fb);

         /* Rendering into an incompatible-format view of a shared compressed
          * target must fail closed via batch rejection without aborting or
          * swapping rsrc->bo. */
         pctx->flush(pctx, NULL, 0);
         struct pipe_surface bad_cbuf = {
            .texture = ix,
            .format = PIPE_FORMAT_R32_UINT,
            .first_layer = 0,
            .last_layer = 0,
            .level = 0,
         };
         struct pipe_framebuffer_state bad_fb = {
            .width = 256,
            .height = 256,
            .layers = 1,
            .samples = 1,
            .nr_cbufs = 1,
         };
         bad_fb.cbufs[0] = bad_cbuf;
         pctx->set_framebuffer_state(pctx, &bad_fb);
         const union pipe_color_union clear_u32 = {.ui = {1, 0, 0, 0}};
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &clear_u32, 0.0, 0);
         rt_rejected = has_rejected_active_batch(ctx);
         pctx->flush(pctx, NULL, 0);
         struct pipe_framebuffer_state empty_fb = {0};
         pctx->set_framebuffer_state(pctx, &empty_fb);
         if (!rt_rejected || ix_rsrc->bo != orig_bo ||
             !ix_rsrc->layout.compressed) {
            fprintf(stderr,
                    "incompatible shared interchange RT view: rejected=%d "
                    "bo_same=%d compressed=%d\n",
                    rt_rejected, ix_rsrc->bo == orig_bo,
                    ix_rsrc->layout.compressed);
            ok = false;
         }

         /* Dispatching a writable shader image store on a shared compressed
          * resource must reject the batch cleanly without swapping BO. */
         struct pipe_compute_state cs_templ = {
            .ir_type = PIPE_SHADER_IR_NIR,
            .prog = build_image_store_cs(),
         };
         void *store_cs = pctx->create_compute_state(pctx, &cs_templ);
         if (!store_cs) {
            fprintf(stderr, "failed to create image store CS on %s\n",
                    agx_chip_name(dev->chip));
            ok = false;
         } else {
            struct pipe_image_view iv = {
               .resource = ix,
               .format = PIPE_FORMAT_B8G8R8A8_UNORM,
               .shader_access = PIPE_IMAGE_ACCESS_WRITE,
            };
            const struct pipe_grid_info grid = {
               .block = {1, 1, 1},
               .grid = {1, 1, 1},
            };
            pctx->bind_compute_state(pctx, store_cs);
            pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &iv);
            const unsigned skipped_before_cs = ctx->skipped_no_program;
            pctx->launch_grid(pctx, &grid);
            comp_cs_rejected = has_rejected_active_batch(ctx);
            pctx->flush(pctx, NULL, 0);
            pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
            pctx->bind_compute_state(pctx, NULL);
            pctx->delete_compute_state(pctx, store_cs);
            if (!comp_cs_rejected ||
                ctx->skipped_no_program != skipped_before_cs ||
                ix_rsrc->bo != orig_bo || !ix_rsrc->layout.compressed) {
               fprintf(stderr,
                       "shared compressed writable image: rejected=%d "
                       "skipped=%u bo_same=%d compressed=%d\n",
                       comp_cs_rejected,
                       ctx->skipped_no_program - skipped_before_cs,
                       ix_rsrc->bo == orig_bo, ix_rsrc->layout.compressed);
               ok = false;
            }
         }
      }
      pipe_resource_reference(&ix, NULL);
   }

   /* Non-interchange uncompressed storage images (shared LINEAR/GPU_TILED 2D
    * and private 3D promoted via agx_decompress()) must keep upstream behavior
    * and must not be rejected on any chip. */
   const uint64_t non_ix_mods[] = {
      DRM_FORMAT_MOD_LINEAR,
      DRM_FORMAT_MOD_APPLE_GPU_TILED,
   };
   bool non_ix_cs_ok = true;
   {
      struct pipe_compute_state cs_templ = {
         .ir_type = PIPE_SHADER_IR_NIR,
         .prog = build_image_store_cs(),
      };
      void *store_cs = pctx->create_compute_state(pctx, &cs_templ);
      if (!store_cs) {
         fprintf(stderr, "failed to create non-ix image store CS on %s\n",
                 agx_chip_name(dev->chip));
         ok = false;
         non_ix_cs_ok = false;
      } else {
         const struct pipe_grid_info grid = {
            .block = {1, 1, 1},
            .grid = {1, 1, 1},
         };
         for (unsigned m = 0; m < ARRAY_SIZE(non_ix_mods); ++m) {
            struct pipe_resource sh2d_templ = {
               .target = PIPE_TEXTURE_2D,
               .format = PIPE_FORMAT_R8G8B8A8_UNORM,
               .width0 = 32,
               .height0 = 32,
               .depth0 = 1,
               .array_size = 1,
               .bind = PIPE_BIND_SHARED | PIPE_BIND_SAMPLER_VIEW,
            };
            struct pipe_resource *sh2d = screen->resource_create_with_modifiers(
               screen, &sh2d_templ, &non_ix_mods[m], 1);
            if (!sh2d) {
               fprintf(stderr, "failed to allocate shared 2D mod=%#llx on %s\n",
                       (unsigned long long)non_ix_mods[m],
                       agx_chip_name(dev->chip));
               ok = false;
               non_ix_cs_ok = false;
               continue;
            }
            struct agx_resource *sh2d_rsrc = agx_resource(sh2d);
            struct agx_bo *orig_bo = sh2d_rsrc->bo;
            struct pipe_image_view iv = {
               .resource = sh2d,
               .format = PIPE_FORMAT_R8G8B8A8_UNORM,
               .shader_access = PIPE_IMAGE_ACCESS_WRITE,
            };
            pctx->flush(pctx, NULL, 0);
            pctx->bind_compute_state(pctx, store_cs);
            pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &iv);
            pctx->launch_grid(pctx, &grid);
            const bool rejected = has_rejected_active_batch(ctx);
            pctx->flush(pctx, NULL, 0);
            pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
            if (rejected || sh2d_rsrc->bo != orig_bo ||
                !sh2d_rsrc->layout.writeable_image) {
               fprintf(stderr,
                       "shared uncompressed 2D mod=%#llx imageStore on %s: "
                       "rejected=%d bo_same=%d writeable=%d\n",
                       (unsigned long long)non_ix_mods[m],
                       agx_chip_name(dev->chip), rejected,
                       sh2d_rsrc->bo == orig_bo,
                       sh2d_rsrc->layout.writeable_image);
               ok = false;
               non_ix_cs_ok = false;
            }
            pipe_resource_reference(&sh2d, NULL);
         }

         /* Private 3D GPU-tiled image starts with writeable_image=false and
          * must still promote via agx_decompress() without batch rejection. */
         const uint64_t tiled_mod = DRM_FORMAT_MOD_APPLE_GPU_TILED;
         struct pipe_resource priv3d_templ = {
            .target = PIPE_TEXTURE_3D,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .width0 = 32,
            .height0 = 32,
            .depth0 = 2,
            .array_size = 1,
            .bind = PIPE_BIND_SAMPLER_VIEW,
         };
         struct pipe_resource *priv3d = screen->resource_create_with_modifiers(
            screen, &priv3d_templ, &tiled_mod, 1);
         if (!priv3d) {
            fprintf(stderr, "failed to allocate private 3D image on %s\n",
                    agx_chip_name(dev->chip));
            ok = false;
            non_ix_cs_ok = false;
         } else {
            struct agx_resource *priv3d_rsrc = agx_resource(priv3d);
            struct pipe_image_view iv = {
               .resource = priv3d,
               .format = PIPE_FORMAT_R8G8B8A8_UNORM,
               .shader_access = PIPE_IMAGE_ACCESS_WRITE,
               .u.tex = {.first_layer = 0, .last_layer = 0, .level = 0,
                         .single_layer_view = true},
            };
            pctx->flush(pctx, NULL, 0);
            pctx->bind_compute_state(pctx, store_cs);
            pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &iv);
            pctx->launch_grid(pctx, &grid);
            const bool rejected = has_rejected_active_batch(ctx);
            pctx->flush(pctx, NULL, 0);
            pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
            if (rejected || !priv3d_rsrc->layout.writeable_image) {
               fprintf(stderr,
                       "private 3D imageStore promotion on %s: rejected=%d "
                       "writeable=%d\n",
                       agx_chip_name(dev->chip), rejected,
                       priv3d_rsrc->layout.writeable_image);
               ok = false;
               non_ix_cs_ok = false;
            }
            pipe_resource_reference(&priv3d, NULL);
         }

         pctx->bind_compute_state(pctx, NULL);
         pctx->delete_compute_state(pctx, store_cs);
      }
   }

   printf("interchange/shared-BO safety on %s: want_ix=%d rt_rejected=%d "
          "comp_cs_rejected=%d non_ix_cs_ok=%d\n",
          agx_chip_name(dev->chip), want_ix, rt_rejected, comp_cs_rejected,
          non_ix_cs_ok);

   /* 10-bit RGB Interchange modifier advertising, scanout allocation, X2
    * clear+blended-draw compilation, and FP16/UINT non-advertisement. */
   static const enum pipe_format tenbit_fmts[] = {
      PIPE_FORMAT_B10G10R10A2_UNORM,
      PIPE_FORMAT_B10G10R10X2_UNORM,
      PIPE_FORMAT_R10G10B10A2_UNORM,
      PIPE_FORMAT_R10G10B10X2_UNORM,
   };
   static const enum pipe_format unsupported_ix_fmts[] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R10G10B10A2_UINT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(unsupported_ix_fmts); ++i) {
      bool u_ext = true;
      const bool u_ix = screen->is_dmabuf_modifier_supported(
         screen, DRM_FORMAT_MOD_APPLE_INTERCHANGE, unsupported_ix_fmts[i],
         &u_ext);
      if (u_ix) {
         fprintf(stderr, "unsupported interchange format %s advertised on %s\n",
                 util_format_name(unsupported_ix_fmts[i]),
                 agx_chip_name(dev->chip));
         ok = false;
      }
      struct pipe_resource u_templ = ix_templ;
      u_templ.format = unsupported_ix_fmts[i];
      struct pipe_resource *u_res = screen->resource_create_with_modifiers(
         screen, &u_templ, display_mods, ARRAY_SIZE(display_mods));
      if (!u_res || agx_resource(u_res)->modifier != DRM_FORMAT_MOD_LINEAR) {
         fprintf(stderr,
                 "unsupported interchange format %s did not fall back to linear on %s\n",
                 util_format_name(unsupported_ix_fmts[i]),
                 agx_chip_name(dev->chip));
         ok = false;
      }
      pipe_resource_reference(&u_res, NULL);
   }

   /* Prepare minimal VS/FS + blended state to verify 10-bit A2 and X2
    * Interchange clear + blended draw compile and emit without batch reject. */
   nir_builder vs_b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "interchange10 vs");
   nir_store_output(&vs_b, nir_imm_vec4(&vs_b, 0.0f, 0.0f, 0.0f, 1.0f),
                    nir_imm_int(&vs_b, 0), .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_shader_gather_info(vs_b.shader, nir_shader_get_entrypoint(vs_b.shader));
   struct pipe_shader_state vs_st = {.type = PIPE_SHADER_IR_NIR,
                                     .ir.nir = vs_b.shader};
   void *vs_cso = pctx->create_vs_state(pctx, &vs_st);

   nir_builder fs_b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "interchange10 fs");
   nir_store_output(&fs_b, nir_imm_vec4(&fs_b, 0.25f, 0.5f, 0.75f, 0.5f),
                    nir_imm_int(&fs_b, 0), .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   nir_shader_gather_info(fs_b.shader, nir_shader_get_entrypoint(fs_b.shader));
   struct pipe_shader_state fs_st = {.type = PIPE_SHADER_IR_NIR,
                                     .ir.nir = fs_b.shader};
   void *fs_cso = pctx->create_fs_state(pctx, &fs_st);

   struct pipe_blend_state blend = {0};
   blend.rt[0].blend_enable = 1;
   blend.rt[0].rgb_func = PIPE_BLEND_ADD;
   blend.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_SRC_ALPHA;
   blend.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   blend.rt[0].alpha_func = PIPE_BLEND_ADD;
   blend.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
   blend.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_ZERO;
   blend.rt[0].colormask = 0xf;
   void *blend_cso = pctx->create_blend_state(pctx, &blend);

   struct pipe_rasterizer_state rast = {
      .depth_clip_near = true,
      .depth_clip_far = true,
      .point_size = 1,
      .line_width = 1,
   };
   void *rast_cso = pctx->create_rasterizer_state(pctx, &rast);
   struct pipe_depth_stencil_alpha_state dsa = {0};
   void *dsa_cso = pctx->create_depth_stencil_alpha_state(pctx, &dsa);
   void *ve_cso = pctx->create_vertex_elements_state(pctx, 0, NULL);

   for (unsigned i = 0; i < ARRAY_SIZE(tenbit_fmts); ++i) {
      const enum pipe_format fmt = tenbit_fmts[i];
      bool tb_ext = true;
      const bool tb_ix = screen->is_dmabuf_modifier_supported(
         screen, DRM_FORMAT_MOD_APPLE_INTERCHANGE, fmt, &tb_ext);
      if (tb_ix != want_ix || (tb_ix && tb_ext)) {
         fprintf(stderr,
                 "interchange %s support=%d ext_only=%d on %s (want %d, 0)\n",
                 util_format_name(fmt), tb_ix, tb_ext, agx_chip_name(dev->chip),
                 want_ix);
         ok = false;
      }

      uint64_t qmods[8] = {0};
      unsigned qext[8] = {0};
      int qcount = 0;
      screen->query_dmabuf_modifiers(screen, fmt, ARRAY_SIZE(qmods), qmods,
                                     qext, &qcount);
      bool found_ix = false;
      for (int m = 0; m < qcount; ++m) {
         if (qmods[m] == DRM_FORMAT_MOD_APPLE_INTERCHANGE) {
            found_ix = true;
            if (qext[m]) {
               fprintf(stderr, "query_dmabuf_modifiers %s marked IX ext_only\n",
                       util_format_name(fmt));
               ok = false;
            }
         }
      }
      if (found_ix != want_ix) {
         fprintf(stderr, "query_dmabuf_modifiers %s found_ix=%d (want %d)\n",
                 util_format_name(fmt), found_ix, want_ix);
         ok = false;
      }

      struct pipe_resource tb_templ = ix_templ;
      tb_templ.format = fmt;
      struct pipe_resource *tb = screen->resource_create_with_modifiers(
         screen, &tb_templ, display_mods, ARRAY_SIZE(display_mods));
      if (!tb) {
         fprintf(stderr, "failed to allocate 10-bit scanout %s on %s\n",
                 util_format_name(fmt), agx_chip_name(dev->chip));
         ok = false;
         continue;
      }

      struct agx_resource *tb_rsrc = agx_resource(tb);
      const uint64_t want_mod = want_ix ? DRM_FORMAT_MOD_APPLE_INTERCHANGE
                                        : DRM_FORMAT_MOD_LINEAR;
      if (tb_rsrc->modifier != want_mod) {
         fprintf(stderr, "10-bit %s modifier %#llx on %s (want %#llx)\n",
                 util_format_name(fmt), (unsigned long long)tb_rsrc->modifier,
                 agx_chip_name(dev->chip), (unsigned long long)want_mod);
         ok = false;
      }

      if (want_ix) {
         struct agx_bo *orig_bo = tb_rsrc->bo;
         if (!agx_resource_is_shared(tb_rsrc) || !tb_rsrc->layout.compressed ||
             tb_rsrc->layout.tiling != AIL_TILING_INTERCHANGE) {
            fprintf(stderr, "10-bit %s not shared+compressed+interchange\n",
                    util_format_name(fmt));
            ok = false;
         }

         /* Full-extent clear-only batches on 10-bit Interchange scanout (both
          * A2 and void-alpha X2) must NOT convert to libagx_fast_clear() CDM:
          * fast clear packs void alpha as 00 (0x3fffffff) instead of opaque
          * 11 (0xffffffff) and bypasses PBE swizzle_a=3 + 1.0 padding. Test
          * both an initial clear and a second clear with data_valid set. */
         pctx->flush(pctx, NULL, 0);
         struct pipe_framebuffer_state fb = {
            .width = 256,
            .height = 256,
            .layers = 1,
            .samples = 1,
            .nr_cbufs = 1,
            .cbufs[0] = {.texture = tb, .format = fmt},
         };
         pctx->set_framebuffer_state(pctx, &fb);
         const union pipe_color_union clear_col = {.f = {0.1f, 0.2f, 0.3f, 1.0f}};
         for (unsigned pass = 0; pass < 2; ++pass) {
            pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &clear_col, 0.0,
                        0);
            struct agx_batch *fc_batch = ctx->batch;
            if (!fc_batch || agx_try_fast_clear_batch(ctx, fc_batch) ||
                fc_batch->vdm.bo == NULL || fc_batch->cdm.bo != NULL ||
                has_rejected_active_batch(ctx)) {
               fprintf(stderr,
                       "10-bit %s clear-only pass %u unexpectedly took "
                       "fast-clear or rejected on %s (batch=%p vdm=%p cdm=%p "
                       "rejected=%d)\n",
                       util_format_name(fmt), pass, agx_chip_name(dev->chip),
                       (void *)fc_batch,
                       fc_batch ? (void *)fc_batch->vdm.bo : NULL,
                       fc_batch ? (void *)fc_batch->cdm.bo : NULL,
                       has_rejected_active_batch(ctx));
               ok = false;
            }
            pctx->flush(pctx, NULL, 0);
            if (tb_rsrc->bo != orig_bo || !tb_rsrc->layout.compressed) {
               fprintf(stderr,
                       "10-bit %s clear-only pass %u mutated BO/compression\n",
                       util_format_name(fmt), pass);
               ok = false;
            }
         }

         /* Clear + blended draw must compile and submit cleanly (exercises
          * RGB10A2 unpack + x_alpha=1.0 padding + PACK_RGB10A2_AGX3 + Gen9
          * compressed Interchange PBE with swizzle_a=3 for X2). */
         pctx->bind_vs_state(pctx, vs_cso);
         pctx->bind_fs_state(pctx, fs_cso);
         pctx->bind_blend_state(pctx, blend_cso);
         pctx->bind_rasterizer_state(pctx, rast_cso);
         pctx->bind_depth_stencil_alpha_state(pctx, dsa_cso);
         pctx->bind_vertex_elements_state(pctx, ve_cso);
         struct pipe_viewport_state vp = {.scale = {128, 128, 0.5},
                                          .translate = {128, 128, 0.5}};
         pctx->set_viewport_states(pctx, 0, 1, &vp);
         pctx->set_sample_mask(pctx, ~0u);

         const unsigned skipped_before = ctx->skipped_no_program;
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &clear_col, 0.0, 0);
         const struct pipe_draw_info dinfo = {.mode = MESA_PRIM_TRIANGLES,
                                              .instance_count = 1};
         const struct pipe_draw_start_count_bias draw = {.count = 3};
         pctx->draw_vbo(pctx, &dinfo, 0, NULL, &draw, 1);
         const bool rejected = has_rejected_active_batch(ctx);
         const bool drew = ctx->batch && ctx->batch->draws;
         pctx->flush(pctx, NULL, 0);
         if (rejected || !drew || ctx->skipped_no_program != skipped_before ||
             tb_rsrc->bo != orig_bo || !tb_rsrc->layout.compressed) {
            fprintf(stderr,
                    "10-bit %s clear+blend draw failed on %s: rejected=%d "
                    "drew=%d skipped=%u bo_same=%d compressed=%d\n",
                    util_format_name(fmt), agx_chip_name(dev->chip), rejected,
                    drew, ctx->skipped_no_program - skipped_before,
                    tb_rsrc->bo == orig_bo, tb_rsrc->layout.compressed);
            ok = false;
         }

         /* Incompatible R32_UINT view on shared 10-bit Interchange must fail
          * closed without reallocating BO. */
         struct pipe_framebuffer_state bad_fb = {
            .width = 256,
            .height = 256,
            .layers = 1,
            .samples = 1,
            .nr_cbufs = 1,
            .cbufs[0] = {.texture = tb, .format = PIPE_FORMAT_R32_UINT},
         };
         pctx->set_framebuffer_state(pctx, &bad_fb);
         const union pipe_color_union clear_u32 = {.ui = {1, 0, 0, 0}};
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &clear_u32, 0.0, 0);
         const bool bad_rejected = has_rejected_active_batch(ctx);
         pctx->flush(pctx, NULL, 0);
         struct pipe_framebuffer_state empty_fb = {0};
         pctx->set_framebuffer_state(pctx, &empty_fb);
         if (!bad_rejected || tb_rsrc->bo != orig_bo ||
             !tb_rsrc->layout.compressed) {
            fprintf(stderr,
                    "10-bit %s incompatible RT view: rejected=%d bo_same=%d "
                    "compressed=%d\n",
                    util_format_name(fmt), bad_rejected, tb_rsrc->bo == orig_bo,
                    tb_rsrc->layout.compressed);
            ok = false;
         }
      }
      pipe_resource_reference(&tb, NULL);
   }

   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->delete_vs_state(pctx, vs_cso);
   pctx->delete_fs_state(pctx, fs_cso);
   pctx->delete_blend_state(pctx, blend_cso);
   pctx->delete_rasterizer_state(pctx, rast_cso);
   pctx->delete_depth_stencil_alpha_state(pctx, dsa_cso);
   pctx->delete_vertex_elements_state(pctx, ve_cso);

   return ok;
}

static struct pipe_resource *
make_2d(struct pipe_screen *screen, enum pipe_format format, unsigned w,
        unsigned h, unsigned bind)
{
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = format,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = bind,
   };

   return screen->resource_create(screen, &templ);
}

static struct pipe_resource *
make_buffer(struct pipe_screen *screen, unsigned size)
{
   struct pipe_resource templ = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      .width0 = size,
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_GLOBAL,
   };

   return screen->resource_create(screen, &templ);
}

static bool
check_direct_resolve_shapes(void)
{
   struct pipe_resource src = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 2,
      .nr_storage_samples = 2,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_resource dst = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_blit_info info = {
      .src = {
         .resource = &src,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.width = 8, .height = 8, .depth = 1},
      },
      .dst = {
         .resource = &dst,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.width = 8, .height = 8, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };

   if (!agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "exact offscreen 2x resolve shape was rejected\n");
      return false;
   }

   src.nr_samples = src.nr_storage_samples = 4;
   if (!agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "exact offscreen 4x resolve shape was rejected\n");
      return false;
   }
   src.nr_samples = src.nr_storage_samples = 2;

   src.format = dst.format = info.src.format = info.dst.format =
      PIPE_FORMAT_B8G8R8X8_UNORM;
   if (!agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "exact XRGB8 resolve shape was rejected\n");
      return false;
   }
   static const enum pipe_format accepted_formats[] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R16G16_FLOAT,
      PIPE_FORMAT_R16_FLOAT,
      PIPE_FORMAT_R16G16B16A16_UNORM,
      PIPE_FORMAT_R16G16B16A16_SNORM,
      PIPE_FORMAT_R10G10B10A2_UNORM,
      PIPE_FORMAT_R8G8B8A8_SNORM,
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R5G6B5_UNORM,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(accepted_formats); ++i) {
      src.format = dst.format = info.src.format = info.dst.format =
         accepted_formats[i];
      if (!agx_g17p_direct_resolve_blit_shape(&info, true)) {
         fprintf(stderr, "direct resolve format %s was rejected\n",
                 util_format_short_name(accepted_formats[i]));
         return false;
      }
   }
   static const enum pipe_format rejected_formats[] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R11G11B10_FLOAT,
      PIPE_FORMAT_R9G9B9E5_FLOAT,
      PIPE_FORMAT_R8G8B8A8_SRGB,
      PIPE_FORMAT_R8G8B8A8_UINT,
      PIPE_FORMAT_R16G16B16A16_SINT,
      PIPE_FORMAT_Z32_FLOAT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(rejected_formats); ++i) {
      src.format = dst.format = info.src.format = info.dst.format =
         rejected_formats[i];
      if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
         fprintf(stderr, "unsupported direct resolve format %s was accepted\n",
                 util_format_short_name(rejected_formats[i]));
         return false;
      }
   }
   src.format = dst.format = info.src.format = info.dst.format =
      PIPE_FORMAT_R8G8B8A8_UNORM;

   info.src.box.width = info.dst.box.width = 4;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "partial offscreen resolve was accepted\n");
      return false;
   }
   info.src.box.width = info.dst.box.width = 8;
   info.dst.format = PIPE_FORMAT_B8G8R8A8_UNORM;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "format-changing offscreen resolve was accepted\n");
      return false;
   }
   info.dst.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   dst.target = PIPE_TEXTURE_2D_ARRAY;
   dst.array_size = 2;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "array offscreen resolve was accepted\n");
      return false;
   }

   dst.target = PIPE_TEXTURE_2D;
   dst.array_size = 1;
   src.nr_samples = src.nr_storage_samples = 1;
   dst.nr_samples = dst.nr_storage_samples = 2;
   src.bind |= PIPE_BIND_SHARED;
   if (!agx_g17p_direct_resolve_blit_shape(&info, false)) {
      fprintf(stderr, "exact reverse WSI seed shape was rejected\n");
      return false;
   }
   src.bind &= ~PIPE_BIND_SHARED;
   if (agx_g17p_direct_resolve_blit_shape(&info, false)) {
      fprintf(stderr, "offscreen reverse seed shape was accepted\n");
      return false;
   }
   info.src.resource = NULL;
   if (agx_g17p_direct_resolve_blit_shape(&info, true)) {
      fprintf(stderr, "resolve without a source was accepted\n");
      return false;
   }

   return true;
}

/* A depth blit whose source and destination carry the same sample count is
 * the shape u_blitter shades once per sample: it turns on sample shading for
 * `src_samples > 1 && src_samples == dst_samples`. G17P has no per-sample
 * fragment execution -- agx_fast_link() refuses the AGX2 machine-code sample
 * loop -- so the fragment program fails to link and the draw is dropped, which
 * is what fails
 * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_depth_config_test.
 *
 * The body is unrolled over the samples instead, which is what Apple's own
 * compiler does, and the whole run of depth stores shares one pixel
 * transaction. Run the blit
 * through the shim, which builds the same programs without a GPU, and hold it
 * to producing one.
 */
static bool
check_msaa_depth_blit(struct pipe_context *pctx, unsigned samples)
{
   struct pipe_screen *screen = pctx->screen;
   struct agx_context *ctx = agx_context(pctx);

   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_Z32_FLOAT,
      .width0 = 32,
      .height0 = 32,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = samples,
      .nr_storage_samples = samples,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *src = screen->resource_create(screen, &templ);
   struct pipe_resource *dst = screen->resource_create(screen, &templ);
   if (!src || !dst) {
      fprintf(stderr, "msaa depth blit: %ux depth target unsupported\n",
              samples);
      pipe_resource_reference(&src, NULL);
      pipe_resource_reference(&dst, NULL);
      return true;
   }

   const struct pipe_blit_info info = {
      .src = {.resource = src,
              .format = PIPE_FORMAT_Z32_FLOAT,
              .level = 0,
              .box = {.width = 32, .height = 32, .depth = 1}},
      .dst = {.resource = dst,
              .format = PIPE_FORMAT_Z32_FLOAT,
              .level = 0,
              .box = {.width = 32, .height = 32, .depth = 1}},
      .mask = PIPE_MASK_Z,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };

   const unsigned skipped_before = ctx->skipped_no_program;
   pctx->blit(pctx, &info);
   const bool ok = ctx->skipped_no_program == skipped_before;

   printf("msaa depth blit: %ux to %ux %s\n", samples, samples,
          ok ? "built a program" : "DROPPED, no fragment program");

   pipe_resource_reference(&src, NULL);
   pipe_resource_reference(&dst, NULL);
   return ok;
}

/* KHR-GL30.framebuffer_blit.scissor_blit blits into a framebuffer holding a
 * colour texture and a depth texture, under a scissor that covers exactly one
 * tile, and the tile comes back zeroed. Build that shape through the shim --
 * which constructs the same background, draw and end-of-tile programs without
 * a GPU -- and hold it to producing every one of them.
 *
 * Every program is built, so the failure is in what they compute or in the
 * tile handling at execution, not in a missing or refused one. Keeping the
 * check means a future regression that does drop a program here is caught
 * without a GPU.
 */
static bool
check_scissored_blit(struct pipe_context *pctx)
{
   struct pipe_screen *screen = pctx->screen;
   struct agx_context *ctx = agx_context(pctx);

   struct pipe_resource *colour =
      make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 64,
              PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
   struct pipe_resource *source =
      make_2d(screen, PIPE_FORMAT_B10G10R10A2_UNORM, 64, 64,
              PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
                 PIPE_BIND_DISPLAY_TARGET);
   struct pipe_resource *depth =
      make_2d(screen, PIPE_FORMAT_Z16_UNORM, 64, 64,
              PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW);
   if (!colour || !source || !depth) {
      fprintf(stderr, "scissored blit: resources unsupported\n");
      pipe_resource_reference(&colour, NULL);
      pipe_resource_reference(&source, NULL);
      pipe_resource_reference(&depth, NULL);
      return true;
   }

   const unsigned skipped_before = ctx->skipped_no_program;

   /* Seed the destination the way the conformance case does, so the scissored
    * blit has existing contents to preserve outside its rectangle. */
   struct pipe_surface colour_surface = {
      .texture = colour, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .first_layer = 0, .last_layer = 0, .level = 0,
   };
   struct pipe_framebuffer_state fb = {
      .width = 64, .height = 64, .layers = 1, .samples = 1, .nr_cbufs = 1,
   };
   fb.cbufs[0] = colour_surface;
   pctx->set_framebuffer_state(pctx, &fb);
   const union pipe_color_union clear = {.f = {0.0f, 0.0f, 0.0f, 1.0f}};
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &clear, 0.0, 0);
   pctx->flush(pctx, NULL, 0);

   /* One tile's worth of scissor, exactly as the conformance case sets. */
   const struct pipe_blit_info info = {
      .src = {.resource = source,
              .format = PIPE_FORMAT_B10G10R10A2_UNORM,
              .level = 0,
              .box = {.width = 64, .height = 64, .depth = 1}},
      .dst = {.resource = colour,
              .format = PIPE_FORMAT_R8G8B8A8_UNORM,
              .level = 0,
              .box = {.width = 64, .height = 64, .depth = 1}},
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
      .scissor_enable = 1,
      .scissor = {.minx = 0, .miny = 0, .maxx = 32, .maxy = 32},
   };
   pctx->blit(pctx, &info);
   pctx->flush(pctx, NULL, 0);

   const bool ok = ctx->skipped_no_program == skipped_before;
   printf("scissored blit into a colour+depth target: %s\n",
          ok ? "built every program" : "DROPPED, a program was missing");

   pipe_resource_reference(&colour, NULL);
   pipe_resource_reference(&source, NULL);
   pipe_resource_reference(&depth, NULL);
   return ok;
}

static bool
check_stencil_blit_shapes(void)
{
   struct pipe_resource src = {.format = PIPE_FORMAT_S8_UINT};
   struct pipe_resource dst = {.format = PIPE_FORMAT_S8_UINT};
   struct pipe_blit_info info = {
      .src = {.resource = &src, .format = PIPE_FORMAT_S8_UINT},
      .dst = {.resource = &dst, .format = PIPE_FORMAT_S8_UINT},
      .mask = PIPE_MASK_S,
   };

   if (!agx_g17p_stencil_plane_blit_shape(&info)) {
      fprintf(stderr, "standalone stencil blit shape was rejected\n");
      return false;
   }

   info.src.format = PIPE_FORMAT_Z24_UNORM_S8_UINT;
   if (!agx_g17p_stencil_plane_blit_shape(&info)) {
      fprintf(stderr, "packed-to-standalone stencil blit was rejected\n");
      return false;
   }

   info.src.format = PIPE_FORMAT_Z32_FLOAT;
   if (agx_g17p_stencil_plane_blit_shape(&info)) {
      fprintf(stderr, "stencil blit without a source plane was accepted\n");
      return false;
   }
   info.src.format = PIPE_FORMAT_S8_UINT;
   info.mask = PIPE_MASK_S | PIPE_MASK_R;
   if (agx_g17p_stencil_plane_blit_shape(&info)) {
      fprintf(stderr, "mixed colour/stencil blit was accepted\n");
      return false;
   }
   info.mask = PIPE_MASK_ZS;
   if (agx_g17p_stencil_plane_blit_shape(&info)) {
      fprintf(stderr, "depth blit without depth planes was accepted\n");
      return false;
   }
   info.src.resource = NULL;
   if (agx_g17p_stencil_plane_blit_shape(&info)) {
      fprintf(stderr, "stencil blit without a source was accepted\n");
      return false;
   }

   return true;
}

static bool
check_staged_resolve_shapes(void)
{
   struct pipe_resource src = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 2,
      .nr_storage_samples = 2,
   };
   struct pipe_resource dst = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 1,
      .nr_storage_samples = 1,
   };
   struct pipe_blit_info info = {
      .src = {
         .resource = &src,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.width = 4, .height = 4, .depth = 1},
      },
      .dst = {
         .resource = &dst,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.width = 4, .height = 4, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };

   if (!agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "partial 2x resolve staging shape was rejected\n");
      return false;
   }
   src.nr_samples = src.nr_storage_samples = 4;
   if (!agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "partial 4x resolve staging shape was rejected\n");
      return false;
   }

   src.format = dst.format = info.src.format = info.dst.format =
      PIPE_FORMAT_B8G8R8X8_UNORM;
   if (!agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "partial XRGB8 resolve staging shape was rejected\n");
      return false;
   }
   src.format = dst.format = info.src.format = info.dst.format =
      PIPE_FORMAT_R16G16B16A16_FLOAT;
   if (!agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "partial RGBA16F resolve staging shape was rejected\n");
      return false;
   }
   src.format = dst.format = info.src.format = info.dst.format =
      PIPE_FORMAT_R32G32B32A32_FLOAT;
   if (agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "partial RGBA32F resolve staging shape was accepted\n");
      return false;
   }
   src.format = dst.format = info.src.format = info.dst.format =
      PIPE_FORMAT_R8G8B8A8_UNORM;

   info.sample0_only = true;
   if (agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "sample-zero copy was accepted as an average resolve\n");
      return false;
   }
   info.sample0_only = false;
   src.target = PIPE_TEXTURE_2D_ARRAY;
   src.array_size = 2;
   if (agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "array resolve was accepted by 2D staging\n");
      return false;
   }
   src.target = PIPE_TEXTURE_2D;
   src.array_size = 1;
   src.format = info.src.format = PIPE_FORMAT_R8G8B8A8_SRGB;
   if (agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "sRGB resolve was accepted before linear averaging\n");
      return false;
   }
   src.format = info.src.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   dst.nr_samples = dst.nr_storage_samples = 2;
   if (agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "multisample destination was accepted as resolved\n");
      return false;
   }
   dst.nr_samples = dst.nr_storage_samples = 1;
   info.mask = PIPE_MASK_Z;
   if (agx_g17p_staged_resolve_blit_shape(&info)) {
      fprintf(stderr, "depth resolve was accepted by colour staging\n");
      return false;
   }

   return true;
}

/* The official clip-distance coverage test captures gl_MaxClipDistances through
 * transform feedback. This follows the same software-VS/GS-copy path without a
 * GPU. Check command construction, not the buffer contents: the shim does not
 * execute shaders. */
static bool
check_xfb_copy_program(struct pipe_context *pctx, bool clip_distances,
                       unsigned interleaved, enum mesa_prim prim)
{
   struct pipe_screen *screen = pctx->screen;
   nir_builder vs = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "transform feedback constant eight");
   if (interleaved) {
      /* nir_create_passthrough_gs copies the vertex shader's output
       * variables, so a shape meant to exercise the copy program keeps its
       * variables; nir_lower_io runs in the driver. */
      nir_variable *pos = nir_variable_create(vs.shader, nir_var_shader_out,
                                              glsl_vec4_type(), "gl_Position");
      pos->data.location = VARYING_SLOT_POS;
      nir_store_var(&vs, pos, nir_imm_vec4(&vs, 0, 0, 0, 1), 0xf);
      for (unsigned i = 0; i < interleaved; ++i) {
         nir_variable *var = nir_variable_create(
            vs.shader, nir_var_shader_out, glsl_vec4_type(),
            ralloc_asprintf(vs.shader, "result_%u", i));
         var->data.location = VARYING_SLOT_VAR0 + i;
         var->data.driver_location = 1 + i;
         nir_store_var(&vs, var,
                       nir_imm_vec4(&vs, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3),
                       0xf);
      }
      /* Capture the position alongside every varying, as an interleaved
       * capture of a whole vertex does: the copy program then holds one more
       * vec4 live across its stores than there are user varyings. */
      nir_xfb_info *xfb =
         rzalloc_size(vs.shader, nir_xfb_info_size(interleaved + 1));
      xfb->buffers_written = 1;
      xfb->streams_written = 1;
      xfb->buffers[0].stride = 16 * (interleaved + 1);
      xfb->buffers[0].varying_count = interleaved + 1;
      xfb->output_count = interleaved + 1;
      xfb->outputs[0].buffer = 0;
      xfb->outputs[0].offset = 0;
      xfb->outputs[0].location = VARYING_SLOT_POS;
      xfb->outputs[0].component_mask = 0xf;
      for (unsigned i = 0; i < interleaved; ++i) {
         xfb->outputs[i + 1].buffer = 0;
         xfb->outputs[i + 1].offset = 16 * (i + 1);
         xfb->outputs[i + 1].location = VARYING_SLOT_VAR0 + i;
         xfb->outputs[i + 1].component_mask = 0xf;
      }
      vs.shader->xfb_info = xfb;
      vs.shader->info.xfb_stride[0] = 4 * (interleaved + 1);
      vs.shader->info.has_transform_feedback_varyings = true;
      vs.shader->info.known_interpolation_qualifiers = true;
      nir_shader_gather_info(vs.shader, nir_shader_get_entrypoint(vs.shader));
      goto vs_built;
   }
   nir_store_output(&vs, nir_imm_vec4(&vs, 0, 0, 0, 1),
                    nir_imm_int(&vs, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_store_output(&vs, nir_imm_int(&vs, 8), nir_imm_int(&vs, 0),
                    .base = 1, .write_mask = 1,
                    .io_semantics.location = VARYING_SLOT_VAR0,
                    .io_xfb.out[0].num_components = 1);
   vs.shader->info.xfb_stride[0] = 1;

   /* The functional clip-distance case also writes gl_ClipDistance and
    * captures it, so the copy program carries the clip distance group too.
    */
   if (clip_distances) {
      nir_store_output(&vs, nir_imm_vec2(&vs, 1.0f, -1.0f),
                       nir_imm_int(&vs, 0), .base = 2, .write_mask = 0x3,
                       .io_semantics.location = VARYING_SLOT_CLIP_DIST0,
                       .io_xfb.out[0].num_components = 2,
                       .io_xfb.out[0].buffer = 0, .io_xfb.out[0].offset = 1);
      vs.shader->info.xfb_stride[0] = 3;
      vs.shader->info.clip_distance_array_size = 2;
   }

   vs.shader->info.known_interpolation_qualifiers = true;
   nir_gather_xfb_info_from_intrinsics(vs.shader);
   nir_shader_gather_info(vs.shader, nir_shader_get_entrypoint(vs.shader));
vs_built:;
   struct pipe_shader_state vs_state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = vs.shader,
   };
   void *vs_cso = pctx->create_vs_state(pctx, &vs_state);

   nir_builder fs = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "transform feedback fragment");
   nir_store_output(&fs, nir_imm_vec4(&fs, 1, 0, 0, 1),
                    nir_imm_int(&fs, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   nir_shader_gather_info(fs.shader, nir_shader_get_entrypoint(fs.shader));
   struct pipe_shader_state fs_state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = fs.shader,
   };
   void *fs_cso = pctx->create_fs_state(pctx, &fs_state);
   struct pipe_rasterizer_state rast = {
      .rasterizer_discard = true,
      .depth_clip_near = true,
      .depth_clip_far = true,
      .clip_plane_enable = clip_distances ? 0x3 : 0,
      .point_size = 1,
      .line_width = 1,
   };
   struct pipe_blend_state blend = {.rt[0].colormask = 0xf};
   struct pipe_depth_stencil_alpha_state depth = {0};
   void *rast_cso = pctx->create_rasterizer_state(pctx, &rast);
   void *blend_cso = pctx->create_blend_state(pctx, &blend);
   void *depth_cso = pctx->create_depth_stencil_alpha_state(pctx, &depth);
   void *ve_cso = pctx->create_vertex_elements_state(pctx, 0, NULL);
   if (!vs_cso || !fs_cso || !rast_cso || !blend_cso || !depth_cso || !ve_cso)
      return false;

   struct pipe_framebuffer_state fb = {.width = 16, .height = 16, .layers = 1};
   struct pipe_viewport_state viewport = {
      .scale = {8, 8, 0.5}, .translate = {8, 8, 0.5},
   };
   pctx->set_framebuffer_state(pctx, &fb);
   pctx->set_viewport_states(pctx, 0, 1, &viewport);
   pctx->set_sample_mask(pctx, ~0u);
   pctx->bind_vs_state(pctx, vs_cso);
   pctx->bind_fs_state(pctx, fs_cso);
   pctx->bind_rasterizer_state(pctx, rast_cso);
   pctx->bind_blend_state(pctx, blend_cso);
   pctx->bind_depth_stencil_alpha_state(pctx, depth_cso);
   pctx->bind_vertex_elements_state(pctx, ve_cso);

   struct pipe_resource *buffer = make_buffer(screen, 256);
   if (!buffer)
      return false;
   struct pipe_stream_output_target *target =
      pctx->create_stream_output_target(pctx, buffer, 0, 256);
   if (!target)
      return false;
   unsigned offset = 0;
   pctx->set_stream_output_targets(pctx, 1, &target, &offset, prim);
   const struct pipe_draw_info info = {
      .mode = prim, .instance_count = 1,
   };
   const struct pipe_draw_start_count_bias draw = {
      .count = mesa_vertices_per_prim(prim),
   };
   struct agx_context *ctx = agx_context(pctx);
   bool ok = true;
   for (unsigned i = 0; i < 2; ++i) {
      pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
      ok &= ctx->gs && ctx->gs->gs_copy && !ctx->gs->gs_copy->failed &&
            ctx->skipped_no_program == 0 && ctx->batch && ctx->batch->draws;

      /* G17P has no helper program, so a shader that spills reads its reloads
       * back as zero with no refusal anywhere. The copy program of a wide
       * interleaved capture is the one that gets close to the register file,
       * so hold it to using no scratch at all.
       */
      ok &= ctx->gs && ctx->gs->gs_copy &&
            ctx->gs->gs_copy->b.info.scratch_size == 0;
   }
   if (!ok)
      fprintf(stderr, "XFB copy program%s%s%s: skipped=%u draws=%u scratch=%u\n",
              clip_distances ? " with clip distances" : "",
              interleaved ? " with interleaved varyings" : "",
              prim == MESA_PRIM_TRIANGLES ? " from triangles" : "",
              ctx->skipped_no_program, ctx->batch ? ctx->batch->draws : 0,
              ctx->gs && ctx->gs->gs_copy ? ctx->gs->gs_copy->b.info.scratch_size
                                          : 0);
   else
      printf("two transform-feedback draws%s%s%s built without a missing "
             "program or spill\n",
             clip_distances ? " with clip distances" : "",
             interleaved ? " with interleaved varyings" : "",
             prim == MESA_PRIM_TRIANGLES ? " from triangles" : "");

   pctx->set_stream_output_targets(pctx, 0, NULL, NULL, prim);
   pipe_so_target_reference(&target, NULL);
   pipe_resource_reference(&buffer, NULL);
   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->delete_vs_state(pctx, vs_cso);
   pctx->delete_fs_state(pctx, fs_cso);
   pctx->delete_rasterizer_state(pctx, rast_cso);
   pctx->delete_blend_state(pctx, blend_cso);
   pctx->delete_depth_stencil_alpha_state(pctx, depth_cso);
   pctx->delete_vertex_elements_state(pctx, ve_cso);
   return ok;
}


/* Whether every variant the driver has compiled for a bound vertex shader
 * state encoded. create_vs_state() precompiles the default hardware variant,
 * so a refusal is visible without a draw. */
static bool
vertex_state_encoded(void *cso)
{
   struct agx_uncompiled_shader *so = cso;
   agx_shader_wait_ready(so);
   bool ok = true;
   hash_table_foreach(so->variants, ent) {
      struct agx_compiled_shader *cs = ent->data;
      ok &= !cs->failed;
   }
   return ok;
}

static uint8_t
agx_mip_ref_avg4_u8(uint8_t a, uint8_t b, uint8_t c, uint8_t d)
{
   const unsigned sum = (unsigned)a + b + c + d;
   const unsigned q = sum >> 2;
   const unsigned r = sum & 3u;
   return (uint8_t)(q + (r > 2u || (r == 2u && q < 128u)));
}

static bool
agx_mip_ref_downsample(const uint8_t *src, uint32_t sw, uint32_t sh,
                       uint8_t *dst, uint32_t dw, uint32_t dh)
{
   const bool odd = (sw == dw * 2u + 1u) || (sh == dh * 2u + 1u);
   if (odd) {
      uint32_t *x_axis = malloc((size_t)(dw + dh) * sizeof(uint32_t));
      if (!x_axis)
         return false;
      uint32_t *y_axis = x_axis + dw;
      if (!agx_build_mip_axis(sw, dw, x_axis) ||
          !agx_build_mip_axis(sh, dh, y_axis)) {
         free(x_axis);
         return false;
      }
      for (uint32_t y = 0; y < dh; ++y) {
         const uint32_t ye = y_axis[y];
         const uint32_t sy0 = ye & 0xffffu;
         const uint32_t wy = (ye >> 16) & 0xffu;
         const uint32_t sy1 = sy0 + (wy != 0u);
         const uint32_t iy = 256u - wy;
         for (uint32_t x = 0; x < dw; ++x) {
            const uint32_t xe = x_axis[x];
            const uint32_t sx0 = xe & 0xffffu;
            const uint32_t wx = (xe >> 16) & 0xffu;
            const uint32_t sx1 = sx0 + (wx != 0u);
            const uint32_t ix = 256u - wx;
            const uint32_t w00 = ix * iy;
            const uint32_t w10 = wx * iy;
            const uint32_t w01 = ix * wy;
            const uint32_t w11 = wx * wy;
            for (unsigned c = 0; c < 4; ++c) {
               const uint32_t val =
                  src[(sy0 * sw + sx0) * 4 + c] * w00 +
                  src[(sy0 * sw + sx1) * 4 + c] * w10 +
                  src[(sy1 * sw + sx0) * 4 + c] * w01 +
                  src[(sy1 * sw + sx1) * 4 + c] * w11;
               const uint32_t fixed4 = (val + 2048u) >> 12;
               uint32_t out = fixed4 >> 4;
               const uint32_t rem = fixed4 & 15u;
               out += (rem > 8u || (rem == 8u && out < 128u));
               dst[(y * dw + x) * 4 + c] = (uint8_t)out;
            }
         }
      }
      free(x_axis);
      return true;
   }

   const uint32_t scx = (sw == dw) ? 1u : 2u;
   const uint32_t scy = (sh == dh) ? 1u : 2u;
   for (uint32_t y = 0; y < dh; ++y) {
      const uint32_t sy0 = y * scy;
      const uint32_t sy1 = sy0 + scy - 1u;
      for (uint32_t x = 0; x < dw; ++x) {
         const uint32_t sx0 = x * scx;
         const uint32_t sx1 = sx0 + scx - 1u;
         for (unsigned c = 0; c < 4; ++c) {
            dst[(y * dw + x) * 4 + c] = agx_mip_ref_avg4_u8(
               src[(sy0 * sw + sx0) * 4 + c], src[(sy0 * sw + sx1) * 4 + c],
               src[(sy1 * sw + sx0) * 4 + c], src[(sy1 * sw + sx1) * 4 + c]);
         }
      }
   }
   return true;
}

static bool
check_agx3_compute_mipmap_case(struct pipe_screen *screen,
                               struct pipe_context *pctx,
                               const char *name,
                               enum pipe_texture_target target,
                               uint32_t width, uint32_t height,
                               uint32_t layers, unsigned last_level,
                               bool expect_fused_2level, bool check_pixels)
{
   struct agx_device *mdev = agx_device(screen);
   struct agx_context *mctx = agx_context(pctx);
   const struct pipe_resource tmpl = {
      .target = target,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = width,
      .height0 = height,
      .depth0 = 1,
      .array_size = layers,
      .last_level = last_level,
      .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
   };
   const uint64_t uncomp_mod = DRM_FORMAT_MOD_APPLE_GPU_TILED;
   struct pipe_resource *tex =
      screen->resource_create_with_modifiers(screen, &tmpl, &uncomp_mod, 1);
   if (!tex) {
      fprintf(stderr, "mipmap %s: resource_create_with_modifiers failed\n",
              name);
      return false;
   }

   /* Upload deterministic non-constant RGBA8 data for every layer's level 0. */
   const size_t base_layer_bytes = (size_t)width * height * 4;
   uint8_t *ref_prev = malloc(base_layer_bytes * layers);
   if (!ref_prev) {
      pipe_resource_reference(&tex, NULL);
      return false;
   }
   for (uint32_t z = 0; z < layers; ++z) {
      uint8_t *layer_buf = ref_prev + z * base_layer_bytes;
      for (uint32_t y = 0; y < height; ++y) {
         for (uint32_t x = 0; x < width; ++x) {
            uint8_t *px = layer_buf + (y * width + x) * 4;
            px[0] = (uint8_t)((x * 13u + y * 7u + z * 37u) & 0xffu);
            px[1] = (uint8_t)((x * 5u + y * 19u + z * 53u + 17u) & 0xffu);
            px[2] = (uint8_t)(((x ^ y) * 11u + z * 29u + 125u) & 0xffu);
            px[3] = (uint8_t)((x * 3u + y * 23u + z * 41u + 190u) & 0xffu);
         }
      }
      struct pipe_box box = {
         .x = 0, .y = 0, .z = (int)z,
         .width = (int)width, .height = (int)height, .depth = 1,
      };
      struct pipe_transfer *xfer = NULL;
      uint8_t *map =
         pctx->texture_map(pctx, tex, 0, PIPE_MAP_WRITE, &box, &xfer);
      if (!map) {
         fprintf(stderr, "mipmap %s: level 0 map failed\n", name);
         free(ref_prev);
         pipe_resource_reference(&tex, NULL);
         return false;
      }
      for (uint32_t y = 0; y < height; ++y)
         memcpy(map + y * xfer->stride, layer_buf + y * width * 4, width * 4);
      pctx->texture_unmap(pctx, xfer);
   }
   pctx->flush(pctx, NULL, 0);

   if (!pctx->generate_mipmap ||
       !pctx->generate_mipmap(pctx, tex, PIPE_FORMAT_R8G8B8A8_UNORM, 0,
                              last_level, 0, layers - 1)) {
      fprintf(stderr, "mipmap %s: generate_mipmap failed\n", name);
      free(ref_prev);
      pipe_resource_reference(&tex, NULL);
      return false;
   }
   if (!mctx->batch || !mctx->batch->cdm.bo || mctx->batch->vdm.bo) {
      fprintf(stderr, "mipmap %s: did not stay in one CDM batch\n", name);
      free(ref_prev);
      pipe_resource_reference(&tex, NULL);
      return false;
   }

   /* Decode the emitted CDM records directly to confirm kernel selection. */
   const uint8_t *cdm_base = agx_bo_map(mctx->batch->cdm.bo);
   const size_t cdm_len = (size_t)(mctx->batch->cdm.current - cdm_base);
   size_t off = 0;
   unsigned launches = 0, fused_launches = 0;
   while (off < cdm_len) {
      struct agx3_cdm_decoded_record rec;
      if (!agx3_cdm_decode_one(mdev->chip, cdm_base + off, cdm_len - off,
                               &rec) ||
          !rec.length) {
         fprintf(stderr, "mipmap %s: CDM decode failed at +%zu/%zu\n", name,
                 off, cdm_len);
         free(ref_prev);
         pipe_resource_reference(&tex, NULL);
         return false;
      }
      if (rec.kind == AGX3_CDM_RECORD_LAUNCH) {
         ++launches;
         if (rec.data.launch.local_size[0] == 16 &&
             rec.data.launch.local_size[1] == 8 &&
             rec.data.launch.local_size[2] == 1)
            ++fused_launches;
      }
      off += rec.length;
   }
   if (expect_fused_2level &&
       (fused_launches < 2 * layers || launches >= last_level * layers)) {
      fprintf(stderr,
              "mipmap %s: expected fused 2-level (16x8) dispatches, got "
              "fused=%u total=%u (single-level=%u)\n",
              name, fused_launches, launches, last_level * layers);
      free(ref_prev);
      pipe_resource_reference(&tex, NULL);
      return false;
   }

   pctx->flush(pctx, NULL, 0);

   /* Map every generated level/layer; on real hardware verify bit-exact output
    * against the CPU reference matching libagx_mipmap_unorm8x4{,_odd}. */
   uint32_t prev_w = width, prev_h = height;
   for (unsigned lvl = 1; lvl <= last_level; ++lvl) {
      const uint32_t dw = u_minify(width, lvl);
      const uint32_t dh = u_minify(height, lvl);
      const size_t dst_layer_bytes = (size_t)dw * dh * 4;
      uint8_t *ref_cur = malloc(dst_layer_bytes * layers);
      if (!ref_cur) {
         free(ref_prev);
         pipe_resource_reference(&tex, NULL);
         return false;
      }
      for (uint32_t z = 0; z < layers; ++z) {
         const uint8_t *src_layer = ref_prev + (size_t)z * prev_w * prev_h * 4;
         uint8_t *dst_layer = ref_cur + (size_t)z * dst_layer_bytes;
         if (!agx_mip_ref_downsample(src_layer, prev_w, prev_h, dst_layer, dw,
                                     dh)) {
            free(ref_cur);
            free(ref_prev);
            pipe_resource_reference(&tex, NULL);
            return false;
         }

         struct pipe_box box = {
            .x = 0, .y = 0, .z = (int)z,
            .width = (int)dw, .height = (int)dh, .depth = 1,
         };
         struct pipe_transfer *xfer = NULL;
         const uint8_t *map =
            pctx->texture_map(pctx, tex, lvl, PIPE_MAP_READ, &box, &xfer);
         if (!map) {
            fprintf(stderr, "mipmap %s: readback map failed lvl=%u z=%u\n",
                    name, lvl, z);
            free(ref_cur);
            free(ref_prev);
            pipe_resource_reference(&tex, NULL);
            return false;
         }
         if (check_pixels) {
            for (uint32_t y = 0; y < dh; ++y) {
               const uint8_t *got_row = map + y * xfer->stride;
               const uint8_t *exp_row = dst_layer + y * dw * 4;
               if (memcmp(got_row, exp_row, dw * 4) != 0) {
                  for (uint32_t x = 0; x < dw; ++x) {
                     if (memcmp(got_row + x * 4, exp_row + x * 4, 4) != 0) {
                        fprintf(stderr,
                                "mipmap %s: mismatch lvl=%u z=%u (%u,%u) "
                                "got=(%u,%u,%u,%u) exp=(%u,%u,%u,%u)\n",
                                name, lvl, z, x, y, got_row[x * 4],
                                got_row[x * 4 + 1], got_row[x * 4 + 2],
                                got_row[x * 4 + 3], exp_row[x * 4],
                                exp_row[x * 4 + 1], exp_row[x * 4 + 2],
                                exp_row[x * 4 + 3]);
                        break;
                     }
                  }
                  pctx->texture_unmap(pctx, xfer);
                  free(ref_cur);
                  free(ref_prev);
                  pipe_resource_reference(&tex, NULL);
                  return false;
               }
            }
         }
         pctx->texture_unmap(pctx, xfer);
      }
      free(ref_prev);
      ref_prev = ref_cur;
      prev_w = dw;
      prev_h = dh;
   }

   free(ref_prev);
   pipe_resource_reference(&tex, NULL);
   return true;
}

/* Record which mip levels used the render blitter, forwarding the calls so
 * the shim checks the transition from rendering to compute as well. */
static void (*mip_blit)(struct pipe_context *, const struct pipe_blit_info *);
static unsigned rendered_levels;

static void
record_mip_blit(struct pipe_context *pipe, const struct pipe_blit_info *blit)
{
   assert(blit->dst.level == blit->src.level + 1);
   assert(blit->dst.resource == blit->src.resource);
   assert(blit->filter == PIPE_TEX_FILTER_LINEAR);
   rendered_levels |= BITFIELD_BIT(blit->dst.level);
   mip_blit(pipe, blit);
}

static bool
check_compressed_mip_chain(struct pipe_screen *screen,
                           struct pipe_context *pctx)
{
   struct agx_device *mdev = agx_device(screen);
   const struct pipe_resource big_tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 2560,
      .height0 = 1440,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 11,
      .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
   };
   const bool chain_compresses =
      agx_g17p_compressed_mip_chain_allowed(mdev, &big_tmpl);
   if (chain_compresses != agx_chip_gen9_color_compression(mdev->chip)) {
      fprintf(stderr, "compressed mip chain policy mismatch\n");
      return false;
   }

   struct pipe_resource *big_tex = screen->resource_create(screen, &big_tmpl);
   if (!big_tex) {
      fprintf(stderr, "big mip chain resource_create failed\n");
      return false;
   }
   struct agx_resource *brsrc = agx_resource(big_tex);
   if (brsrc->layout.compressed != chain_compresses ||
       (chain_compresses &&
        agx_generate_mipmap_agx3_supported(
           mdev, brsrc, PIPE_FORMAT_R8G8B8A8_UNORM, 0, 11, 0, 0))) {
      fprintf(stderr, "compressed mip chain allocation mismatch\n");
      pipe_resource_reference(&big_tex, NULL);
      return false;
   }

   /* On HAL200/AGX3B (G16G, G17P), exact 2x2 compressed levels stay in
    * compute via compressed block-mip (32x32 workgroups storing through the
    * compressed PBE) with zero render passes; chips without block-mip fall
    * back to rendering levels at or above the raw-compute crossover. */
   const bool block_mip =
      chain_compresses && hk_g17p_block_mip_supported(mdev->chip);
   unsigned expected_levels = 0;
   for (unsigned l = 0; l < big_tmpl.last_level; ++l) {
      const unsigned w = u_minify(big_tmpl.width0, l);
      const unsigned h = u_minify(big_tmpl.height0, l);
      if ((w | h) & 1 ||
          !(block_mip
               ? hk_compressed_mip_level_through_pbe(mdev->chip, w, h, w / 2,
                                                     h / 2)
               : hk_mip_level_through_pbe(mdev->chip, w / 2, h / 2)))
         break;
      expected_levels |= BITFIELD_BIT(l + 1);
   }
   if (block_mip && expected_levels != 0) {
      fprintf(stderr, "HAL200/AGX3B block-mip should render 0 exact-2x2 levels\n");
      pipe_resource_reference(&big_tex, NULL);
      return false;
   }

   struct agx_context *mctx = agx_context(pctx);
   mip_blit = pctx->blit;
   rendered_levels = 0;
   pctx->blit = record_mip_blit;
   bool generated = pctx->generate_mipmap(
      pctx, big_tex, PIPE_FORMAT_R8G8B8A8_UNORM, 0, 11, 0, 0);
   pctx->blit = mip_blit;
   if (!generated || !mctx->batch || !mctx->batch->cdm.bo ||
       (chain_compresses && rendered_levels != expected_levels)) {
      fprintf(stderr, "mipmap render levels %#x, expected %#x\n",
              rendered_levels, expected_levels);
      pipe_resource_reference(&big_tex, NULL);
      return false;
   }
   if (chain_compresses) {
      const uint8_t *cdm_base = agx_bo_map(mctx->batch->cdm.bo);
      const size_t cdm_len = (size_t)(mctx->batch->cdm.current - cdm_base);
      size_t off = 0;
      unsigned launches = 0, block_launches = 0, fused_launches = 0;
      while (off < cdm_len) {
         struct agx3_cdm_decoded_record rec;
         if (!agx3_cdm_decode_one(mdev->chip, cdm_base + off, cdm_len - off,
                                  &rec) ||
             !rec.length) {
            fprintf(stderr, "compressed mipchain CDM decode failed at +%zu/%zu\n",
                    off, cdm_len);
            pipe_resource_reference(&big_tex, NULL);
            return false;
         }
         if (rec.kind == AGX3_CDM_RECORD_LAUNCH) {
            ++launches;
            if (rec.data.launch.local_size[0] == 32 &&
                rec.data.launch.local_size[1] == 32 &&
                rec.data.launch.local_size[2] == 1)
               ++block_launches;
            if (rec.data.launch.local_size[0] == 16 &&
                rec.data.launch.local_size[1] == 8 &&
                rec.data.launch.local_size[2] == 1)
               ++fused_launches;
         }
         off += rec.length;
      }
      if (block_mip) {
         /* Levels 1..5 (1280x720..80x45) are exact 2x2 compressed block-mip
          * (32x32), followed by odd level 6 (40x22) and tail levels 7..11. */
         if (block_launches != 5 || launches < 11) {
            fprintf(stderr,
                    "compressed 2560x1440 mipchain did not take block-mip "
                    "(block=%u launches=%u)\n",
                    block_launches, launches);
            pipe_resource_reference(&big_tex, NULL);
            return false;
         }
      } else if (!fused_launches || launches >= 11) {
         fprintf(stderr,
                 "compressed 2560x1440 mipchain did not take precompiled tail "
                 "(fused=%u launches=%u)\n",
                 fused_launches, launches);
         pipe_resource_reference(&big_tex, NULL);
         return false;
      }
   }
   pctx->flush(pctx, NULL, 0);
   pipe_resource_reference(&big_tex, NULL);

   struct pipe_resource srgb_tmpl = big_tmpl;
   srgb_tmpl.format = PIPE_FORMAT_R8G8B8A8_SRGB;
   struct pipe_resource array_tmpl = big_tmpl;
   array_tmpl.target = PIPE_TEXTURE_2D_ARRAY;
   array_tmpl.array_size = 2;
   if (agx_g17p_compressed_mip_chain_allowed(mdev, &srgb_tmpl) ||
       agx_g17p_compressed_mip_chain_allowed(mdev, &array_tmpl)) {
      fprintf(stderr, "sRGB or layered mip chain admitted\n");
      return false;
   }
   return true;
}

static bool
check_agx3_compute_mipmap(struct pipe_screen *screen, struct pipe_context *pctx,
                          bool check_pixels)
{
   struct agx_device *mdev = agx_device(screen);
   if (agx_chip_descriptor_generation(mdev->chip) == AGX_DESCRIPTOR_GENERATION_10)
      return check_agx3_compute_mipmap_case(screen, pctx, "sampled_65x33",
                                            PIPE_TEXTURE_2D, 65, 33, 1, 6,
                                            false, check_pixels);

   if (!agx_chip_native_agx3(mdev->chip))
      return true;

   /* Verify predicate accept/reject matrix on an explicitly uncompressed
    * GPU-tiled mipmap chain. */
   const struct pipe_resource mip_tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 64,
      .height0 = 33,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 6,
      .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
   };
   const uint64_t uncomp_mod = DRM_FORMAT_MOD_APPLE_GPU_TILED;
   struct pipe_resource *mip_tex =
      screen->resource_create_with_modifiers(screen, &mip_tmpl, &uncomp_mod, 1);
   if (!mip_tex) {
      fprintf(stderr, "mip_tex resource_create_with_modifiers failed\n");
      return false;
   }
   struct agx_resource *mrsrc = agx_resource(mip_tex);
   if (!agx_generate_mipmap_agx3_supported(
          mdev, mrsrc, PIPE_FORMAT_R8G8B8A8_UNORM, 0, 6, 0, 0) ||
       agx_generate_mipmap_agx3_supported(
          mdev, mrsrc, PIPE_FORMAT_R8G8B8A8_SRGB, 0, 6, 0, 0) ||
       agx_generate_mipmap_agx3_supported(
          mdev, mrsrc, PIPE_FORMAT_R16G16B16A16_FLOAT, 0, 6, 0, 0) ||
       agx_generate_mipmap_agx3_supported(
          mdev, mrsrc, PIPE_FORMAT_R8G8B8A8_UNORM, 3, 3, 0, 0)) {
      fprintf(stderr, "AGX3 mipchain predicate matrix failed\n");
      pipe_resource_reference(&mip_tex, NULL);
      return false;
   }
   pipe_resource_reference(&mip_tex, NULL);

   /* Default private mipmapped 2D RGBA8 textures compress on G17P and are
    * rejected by the uncompressed CDM mipchain so util_gen_mipmap() preserves
    * compression. */
   struct pipe_resource *priv_mip_tex =
      screen->resource_create(screen, &mip_tmpl);
   if (!priv_mip_tex) {
      fprintf(stderr, "priv_mip_tex resource_create failed\n");
      return false;
   }
   struct agx_resource *priv_mrsrc = agx_resource(priv_mip_tex);
   const bool expect_comp = mdev->chip == AGX_CHIP_G17P;
   if (priv_mrsrc->layout.compressed != expect_comp ||
       (expect_comp && agx_generate_mipmap_agx3_supported(
                          mdev, priv_mrsrc, PIPE_FORMAT_R8G8B8A8_UNORM, 0, 6, 0,
                          0))) {
      fprintf(stderr, "AGX3 private mipchain compression guard failed\n");
      pipe_resource_reference(&priv_mip_tex, NULL);
      return false;
   }
   pipe_resource_reference(&priv_mip_tex, NULL);

   if (!check_agx3_compute_mipmap_case(screen, pctx, "even_64x64_fused",
                                       PIPE_TEXTURE_2D, 64, 64, 1, 6, true,
                                       check_pixels) ||
       !check_agx3_compute_mipmap_case(screen, pctx, "odd_65x33",
                                       PIPE_TEXTURE_2D, 65, 33, 1, 6, false,
                                       check_pixels) ||
       !check_agx3_compute_mipmap_case(screen, pctx, "strip_1x9",
                                       PIPE_TEXTURE_2D, 1, 9, 1, 3, false,
                                       check_pixels) ||
       !check_agx3_compute_mipmap_case(screen, pctx, "strip_9x1",
                                       PIPE_TEXTURE_2D, 9, 1, 1, 3, false,
                                       check_pixels) ||
       !check_agx3_compute_mipmap_case(screen, pctx, "array_33x17x4",
                                       PIPE_TEXTURE_2D_ARRAY, 33, 17, 4, 5,
                                       false, check_pixels) ||
       !check_agx3_compute_mipmap_case(screen, pctx, "cube_32x32x6_fused",
                                       PIPE_TEXTURE_CUBE, 32, 32, 6, 5, true,
                                       check_pixels) ||
       !check_compressed_mip_chain(screen, pctx))
      return false;

   printf("AGX3 compute generate_mipmap completed (check_pixels=%u)\n",
          (unsigned)check_pixels);
   return true;
}

/* The hardware vertex program shapes behind KHR-GL30.clip_distance.* and any
 * program with more than one user varying slot: position plus N clip
 * distances (the functional test, N = 1..8, written from gl_VertexID), the
 * coverage test's eight clip distances that the fragment shader reads back,
 * and two plain user varyings. Each is created as a vertex shader state and
 * the compiled variant checked for a refusal. */
static bool
check_vertex_output_shapes(struct pipe_context *pctx)
{
   bool ok = true;
   for (unsigned shape = 0; shape < 13; ++shape) {
      const unsigned clip_count = shape < 8 ? shape + 1 : shape == 8 ? 8 : 0;
      const bool two_varyings = shape == 9 || shape == 11;
      /* KHR-GL30.transform_feedback3.multiple_streams links no fragment
       * shader. Its geometry copy program writes position, two captured user
       * varyings and a point size; with no consumer the interpolation
       * qualifiers are known, so every UVS index is inlined. */
      const bool point_size = shape == 11;
      /* KHR-GL30.transform_feedback.capture_vertex_interleaved_test (and the
       * query and discard variants that reuse its program) write
       * min(MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS, MAX_VARYING_COMPONENTS)
       * / 4 - 1 constant vec4 varyings and select gl_Position by gl_VertexID.
       */
      const unsigned interleaved = shape == 10 ? 64 / 4 - 1 : 0;
      /* KHR-GL33.transform_feedback_overflow_query_ARB's program captures
       * four SCALAR float varyings and never writes gl_Position at all --
       * its vertex shader body is only `output1..4 = 1.0..4.0`.  Every other
       * shape here writes position, so nothing covered a positionless one,
       * and it is REFUSED: with no position the user outputs reach UVS index
       * 4 while the output count is only 4, which is below every qualified
       * shape in agx3_pack_binary.c's g17p_user_output_protocol (6, or >= 5
       * with terminal/leave-staging-alone, or the Firefox dense suffix), so
       * the `st_vary ... value >= 4` guard refuses it.  That is 5 of the 6
       * gl33 failures. Expected to fail until the shape is qualified. */
      const bool scalar_only = shape == 12;
      nir_builder vs = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "vertex output shape");
      nir_def *vid = nir_i2f32(&vs, nir_load_vertex_id(&vs));
      nir_def *pos = nir_vec4(&vs, nir_fmul_imm(&vs, vid, 0.5f),
                              nir_imm_float(&vs, 0), nir_imm_float(&vs, 0),
                              nir_imm_float(&vs, 1));
      if (interleaved) {
         nir_def *id = nir_load_vertex_id(&vs);
         const float e = 0.0625f;
         const float corners[4][2] = {
            {-1 + e, 1 - e}, {1 - e, 1 - e}, {-1 + e, -1 + e}, {1 - e, -1 + e}};
         pos = nir_imm_vec4(&vs, 0, 0, 0, 0);
         for (unsigned c = 0; c < 4; ++c) {
            pos = nir_bcsel(&vs, nir_ieq_imm(&vs, id, c),
                            nir_imm_vec4(&vs, corners[c][0], corners[c][1], 0, 1),
                            pos);
         }
         for (unsigned i = 0; i < interleaved; ++i) {
            nir_store_output(&vs,
                             nir_imm_vec4(&vs, i * 4, i * 4 + 1, i * 4 + 2,
                                          i * 4 + 3),
                             nir_imm_int(&vs, 0), .base = 1 + i, .write_mask = 0xf,
                             .io_semantics.location = VARYING_SLOT_VAR0 + i,
                             .io_semantics.num_slots = 1,
                             .src_type = nir_type_float32);
         }
      }
      if (scalar_only) {
         for (unsigned i = 0; i < 4; ++i) {
            nir_store_output(&vs, nir_imm_float(&vs, 1.0f + i),
                             nir_imm_int(&vs, 0), .base = 1 + i,
                             .write_mask = 1,
                             .io_semantics.location = VARYING_SLOT_VAR0 + i,
                             .io_semantics.num_slots = 1,
                             .src_type = nir_type_float32);
         }
      } else {
         nir_store_output(&vs, pos, nir_imm_int(&vs, 0), .base = 0,
                          .write_mask = 0xf,
                          .io_semantics.location = VARYING_SLOT_POS);
      }
      for (unsigned i = 0; i < clip_count; ++i) {
         nir_def *v = nir_fadd_imm(
            &vs,
            nir_fmul_imm(&vs, nir_fadd_imm(&vs, vid, 1.0f),
                         0.75f * (i + 1) / (clip_count * 3.0f)),
            0.25f);
         nir_store_output(&vs, v, nir_imm_int(&vs, 0), .base = 1 + i / 4,
                          .component = i % 4, .write_mask = 1,
                          .io_semantics.location = VARYING_SLOT_CLIP_DIST0 + i / 4,
                          .io_semantics.num_slots = 1,
                          .src_type = nir_type_float32);
      }
      if (two_varyings) {
         nir_store_output(&vs, nir_vec4(&vs, vid, vid, vid, vid),
                          nir_imm_int(&vs, 0), .base = 1, .write_mask = 0xf,
                          .io_semantics.location = VARYING_SLOT_VAR0);
         nir_store_output(&vs, nir_vec4(&vs, vid, vid, vid, vid),
                          nir_imm_int(&vs, 0), .base = 2, .write_mask = 0xf,
                          .io_semantics.location = VARYING_SLOT_VAR1);
      }
      if (point_size) {
         nir_store_output(&vs, nir_imm_float(&vs, 1.0f), nir_imm_int(&vs, 0),
                          .base = 3, .write_mask = 1,
                          .io_semantics.location = VARYING_SLOT_PSIZ,
                          .src_type = nir_type_float32);
      }
      vs.shader->info.clip_distance_array_size = clip_count;
      nir_shader_gather_info(vs.shader, nir_shader_get_entrypoint(vs.shader));

      /* A linked GL program arrives with its interpolation qualifiers known
       * (nir_opt_varyings), which lets the UVS lowering inline the varying
       * indices. A separable program does not, and keeps uniform indices.
       */
      const bool linked = getenv("AGX_TEST_SEPARABLE_VERTEX") == NULL;
      vs.shader->info.known_interpolation_qualifiers = linked;
      struct pipe_shader_state vs_state = {
         .type = PIPE_SHADER_IR_NIR,
         .ir.nir = vs.shader,
      };
      void *vs_cso = pctx->create_vs_state(pctx, &vs_state);
      const bool encoded = vs_cso && vertex_state_encoded(vs_cso);
      fprintf(stderr, "vertex output shape %u (clip=%u varyings=%u%s %s): %s\n",
              shape, clip_count, two_varyings ? 2 : interleaved,
              point_size ? " psiz" : "",
              linked ? "linked" : "separable",
              encoded ? "encoded" : "REFUSED");
      /* Shape 12 is a KNOWN, named gap rather than a regression: it is the
       * reproducer for the gl33 transform-feedback-overflow failures, and it
       * stays refused until a positionless four-output vertex shape is
       * qualified in g17p_user_output_protocol.  Report it, but do not fail
       * the suite on it -- a gate that is red by construction stops telling
       * anyone anything. Flip this to `ok &= encoded` once it is qualified,
       * and the shape becomes its regression test. */
      if (!scalar_only)
         ok &= encoded;
      if (vs_cso)
         pctx->delete_vs_state(pctx, vs_cso);
   }
   return ok;
}
static bool
check_discard_main_sample_key(struct pipe_context *pctx)
{
   struct agx_context *ctx = agx_context(pctx);
   nir_builder vs = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "sample key vertex");
   nir_store_output(&vs, nir_imm_vec4(&vs, 0, 0, 0, 1), nir_imm_int(&vs, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_shader_gather_info(vs.shader, nir_shader_get_entrypoint(vs.shader));
   struct pipe_shader_state vs_state = {.type = PIPE_SHADER_IR_NIR, .ir.nir = vs.shader};
   void *vs_cso = pctx->create_vs_state(pctx, &vs_state);
   nir_builder fs = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "discard main raster sample key");
   fs.shader->info.internal = false;
   const bool force_early = getenv("AGX_TEST_MAIN_SAMPLE_EARLY") != NULL;
   fs.shader->info.fs.early_fragment_tests = force_early;
   nir_def *x = nir_channel(&fs, nir_load_frag_coord(&fs), 0);
   nir_demote_if(&fs, nir_flt(&fs, x, nir_imm_float(&fs, 8.0)));
   nir_store_output(&fs, nir_imm_vec4(&fs, 1, 0, 0, 0.5), nir_imm_int(&fs, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   nir_shader_gather_info(fs.shader, nir_shader_get_entrypoint(fs.shader));
   struct pipe_shader_state fs_state = {.type = PIPE_SHADER_IR_NIR, .ir.nir = fs.shader};
   struct agx_uncompiled_shader *fs_cso = pctx->create_fs_state(pctx, &fs_state);
   struct pipe_rasterizer_state rast = {
      .depth_clip_near = true, .depth_clip_far = true, .multisample = true,
      .point_size = 1, .line_width = 1,
   };
   struct pipe_depth_stencil_alpha_state depth = {0};
   void *rast_cso = pctx->create_rasterizer_state(pctx, &rast);
   void *depth_cso = pctx->create_depth_stencil_alpha_state(pctx, &depth);
   void *ve_cso = pctx->create_vertex_elements_state(pctx, 0, NULL);
   if (!vs_cso || !fs_cso || !rast_cso || !depth_cso || !ve_cso)
      return false;
   agx_shader_wait_ready(fs_cso);
   if (fs_cso->info.needs_sample_count || fs_cso->info.uses_fbfetch ||
       fs_cso->info.uses_sample_shading)
      return false;
   pctx->bind_vs_state(pctx, vs_cso);
   pctx->bind_fs_state(pctx, fs_cso);
   pctx->bind_rasterizer_state(pctx, rast_cso);
   pctx->bind_depth_stencil_alpha_state(pctx, depth_cso);
   pctx->bind_vertex_elements_state(pctx, ve_cso);
   struct pipe_viewport_state viewport = {.scale = {8, 8, 0.5}, .translate = {8, 8, 0.5}};
   pctx->set_viewport_states(pctx, 0, 1, &viewport);
   pctx->set_sample_mask(pctx, ~0u);
   const unsigned sequence[] = {1, 4, 2, 1};
   struct agx_compiled_shader *variants[5] = {0};
   bool ok = true;
   for (unsigned alpha = 0; alpha < 4; ++alpha) {
      struct pipe_blend_state blend = {.rt[0].colormask = 0xf, .alpha_to_coverage = alpha & 1};
      void *blend_cso = pctx->create_blend_state(pctx, &blend);
      pctx->bind_blend_state(pctx, blend_cso);
      for (unsigned step = 0; step < ARRAY_SIZE(sequence); ++step) {
         const unsigned samples = sequence[step];
         struct pipe_resource templ = {
            .target = PIPE_TEXTURE_2D, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .width0 = 16, .height0 = 16, .depth0 = 1, .array_size = 1,
            .nr_samples = samples, .nr_storage_samples = samples,
            .bind = PIPE_BIND_RENDER_TARGET, .usage = PIPE_USAGE_DEFAULT,
         };
         struct pipe_resource *color = pctx->screen->resource_create(pctx->screen, &templ);
         if (!color) return false;
         struct pipe_framebuffer_state fb = {
            .width = 16, .height = 16, .layers = 1, .samples = samples, .nr_cbufs = 1,
            .cbufs[0] = {.texture = color, .format = templ.format},
         };
         pctx->set_framebuffer_state(pctx, &fb);
         const unsigned full_mask = (1u << samples) - 1;
         const unsigned mask = (alpha & 2) && samples > 1 ? (samples == 4 ? 5u : 2u) : full_mask;
         pctx->set_sample_mask(pctx, mask);
         const struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
         const struct pipe_draw_start_count_bias draw = {.count = 3};
         pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
         unsigned found = 0, keyed_samples = ~0u;
         hash_table_foreach(fs_cso->variants, ent) {
            if (ent->data == ctx->fs) {
               keyed_samples = ((const struct asahi_fs_shader_key *)ent->key)->nr_samples;
               found++;
            }
         }
         bool phase_ok = found == 1 && keyed_samples == samples && ctx->fs &&
                         !ctx->fs->failed && ctx->fs->b.info.main_size &&
                         ctx->skipped_no_program == 0 && ctx->batch && ctx->batch->draws;
         if (variants[samples]) phase_ok &= variants[samples] == ctx->fs;
         variants[samples] = ctx->fs;
         const bool main_may_test = force_early || (!(alpha & 1) && mask == full_mask);
         const bool actual_main_test = ctx->batch && ctx->batch->uniforms.no_epilog_discard;
         phase_ok &= actual_main_test == main_may_test;
         printf("test-owner alpha=%u mask=%x samples=%u main=%u expected=%u\n",
                alpha & 1, mask, samples, actual_main_test, main_may_test);
         printf("sample-key alpha=%u samples=%u actual=%u main_size=%u result=%s\n",
                alpha, samples, keyed_samples, ctx->fs ? ctx->fs->b.info.main_size : 0,
                phase_ok ? "PASS" : "FAIL");
         ok &= phase_ok;
         pipe_resource_reference(&color, NULL);
      }
      pctx->bind_blend_state(pctx, NULL);
      pctx->delete_blend_state(pctx, blend_cso);
   }
   ok &= variants[1] != variants[2] && variants[1] != variants[4] && variants[2] != variants[4];
   return ok;
}

static bool
check_invalidate_and_bgra_readback(struct pipe_screen *screen,
                                   struct pipe_context *pctx,
                                   bool check_pixels)
{
   struct agx_context *ctx = agx_context(pctx);

   /* 1. Flushed texture: invalidate_resource() must move previously valid
    * levels from data_valid into invalidated_levels without tainting never-
    * written levels, skip COLOR0 reload on the next batch, and clear
    * invalidated_levels[0] as soon as the next batch writes level 0. */
   struct pipe_resource tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 64,
      .height0 = 64,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *tex = screen->resource_create(screen, &tmpl);
   if (!tex) {
      fprintf(stderr, "invalidate test resource_create failed\n");
      return false;
   }
   struct agx_resource *rsrc = agx_resource(tex);

   struct pipe_framebuffer_state fb = {
      .width = 64,
      .height = 64,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 1,
   };
   fb.cbufs[0] = (struct pipe_surface){
      .texture = tex,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   pctx->set_framebuffer_state(pctx, &fb);

   const union pipe_color_union c0 = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &c0, 0.0, 0);
   pctx->flush(pctx, NULL, 0);

   if (!agx_resource_valid(rsrc, 0) || agx_resource_valid(rsrc, 1) ||
       BITSET_TEST(rsrc->invalidated_levels, 0) ||
       BITSET_TEST(rsrc->invalidated_levels, 1)) {
      fprintf(stderr, "unexpected valid/invalidated state after clear+flush\n");
      pipe_resource_reference(&tex, NULL);
      return false;
   }

   /* Unbind before invalidating (the try_pbo_upload_common() order). */
   struct pipe_framebuffer_state empty_fb = {0};
   pctx->set_framebuffer_state(pctx, &empty_fb);
   pctx->invalidate_resource(pctx, tex);

   if (agx_resource_valid(rsrc, 0) ||
       !BITSET_TEST(rsrc->invalidated_levels, 0) ||
       BITSET_TEST(rsrc->invalidated_levels, 1)) {
      fprintf(stderr,
              "expected invalidate_resource() to move level 0 to "
              "invalidated_levels without tainting level 1 (valid0=%d inv0=%d "
              "inv1=%d)\n",
              agx_resource_valid(rsrc, 0),
              BITSET_TEST(rsrc->invalidated_levels, 0),
              BITSET_TEST(rsrc->invalidated_levels, 1));
      pipe_resource_reference(&tex, NULL);
      return false;
   }

   /* Re-binding and initializing a new batch must skip COLOR0 reload, mark
    * level 0 valid again, and clear invalidated_levels[0]. */
   pctx->set_framebuffer_state(pctx, &fb);
   struct agx_batch *batch = agx_get_batch(ctx);
   agx_batch_init_state(batch);
   if ((batch->load & PIPE_CLEAR_COLOR0) || !agx_resource_valid(rsrc, 0) ||
       BITSET_TEST(rsrc->invalidated_levels, 0)) {
      fprintf(stderr,
              "post-invalidate batch init wrong (load=%#x valid0=%d inv0=%d)\n",
              batch->load, agx_resource_valid(rsrc, 0),
              BITSET_TEST(rsrc->invalidated_levels, 0));
      pipe_resource_reference(&tex, NULL);
      return false;
   }

   /* 2. Active initialized batch: invalidate_resource() must drop resolve and
    * dirty AGX_DIRTY_BLEND so a later draw in the same batch re-enables store,
    * while preserving data_valid and keeping invalidated_levels clear. */
   batch->resolve |= PIPE_CLEAR_COLOR0;
   ctx->dirty &= ~AGX_DIRTY_BLEND;
   pctx->invalidate_resource(pctx, tex);
   if ((batch->resolve & PIPE_CLEAR_COLOR0) ||
       !(ctx->dirty & AGX_DIRTY_BLEND) || !agx_resource_valid(rsrc, 0) ||
       BITSET_TEST(rsrc->invalidated_levels, 0)) {
      fprintf(stderr,
              "active-batch invalidate state wrong (resolve=%#x dirty=%#x "
              "valid=%d inv=%d)\n",
              batch->resolve, ctx->dirty, agx_resource_valid(rsrc, 0),
              BITSET_TEST(rsrc->invalidated_levels, 0));
      pipe_resource_reference(&tex, NULL);
      return false;
   }
   pctx->flush(pctx, NULL, 0);
   pctx->set_framebuffer_state(pctx, &empty_fb);
   pipe_resource_reference(&tex, NULL);

   /* 3. Separate stencil: active-batch invalidate dirties AGX_DIRTY_ZS, and
    * post-flush invalidate clears both depth and separate_stencil data_valid
    * into invalidated_levels. */
   struct pipe_resource *zs =
      make_2d(screen, PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, 64, 64,
              PIPE_BIND_DEPTH_STENCIL);
   if (!zs) {
      fprintf(stderr, "invalidate test zs resource_create failed\n");
      return false;
   }
   struct agx_resource *zrsrc = agx_resource(zs);
   struct pipe_framebuffer_state zs_fb = {
      .width = 64,
      .height = 64,
      .layers = 1,
      .samples = 1,
   };
   zs_fb.zsbuf = (struct pipe_surface){
      .texture = zs,
      .format = PIPE_FORMAT_Z32_FLOAT_S8X24_UINT,
   };
   pctx->set_framebuffer_state(pctx, &zs_fb);
   pctx->clear(pctx, PIPE_CLEAR_DEPTHSTENCIL, 0, 0, NULL, &c0, 0.5, 0x5a);

   struct agx_batch *zs_batch = agx_get_batch(ctx);
   zs_batch->resolve |= PIPE_CLEAR_DEPTHSTENCIL;
   ctx->dirty &= ~AGX_DIRTY_ZS;
   pctx->invalidate_resource(pctx, zs);
   if ((zs_batch->resolve & PIPE_CLEAR_DEPTHSTENCIL) ||
       !(ctx->dirty & AGX_DIRTY_ZS)) {
      fprintf(stderr,
              "active-batch ZS invalidate state wrong (resolve=%#x dirty=%#x)\n",
              zs_batch->resolve, ctx->dirty);
      pipe_resource_reference(&zs, NULL);
      return false;
   }

   pctx->flush(pctx, NULL, 0);
   pctx->set_framebuffer_state(pctx, &empty_fb);
   pctx->invalidate_resource(pctx, zs);
   if (agx_resource_valid(zrsrc, 0) ||
       !BITSET_TEST(zrsrc->invalidated_levels, 0) ||
       (zrsrc->separate_stencil &&
        (agx_resource_valid(zrsrc->separate_stencil, 0) ||
         !BITSET_TEST(zrsrc->separate_stencil->invalidated_levels, 0)))) {
      fprintf(stderr,
              "expected invalidate_resource() to move Z+S valid state into "
              "invalidated_levels\n");
      pipe_resource_reference(&zs, NULL);
      return false;
   }
   pipe_resource_reference(&zs, NULL);

   /* 4. BGRA8_UNORM -> RGBA8_UNORM blit readback (both uncompressed 32x32 and
    * compression-eligible 256x256). Under the no-op shim this exercises the
    * descriptor/shader path; on real hardware (check_pixels=true) it checks
    * exact R/G/B/A channel values. */
   const unsigned sizes[2] = {32, 256};
   const uint8_t want[4] = {64, 128, 191, 255};
   for (unsigned s = 0; s < ARRAY_SIZE(sizes); ++s) {
      const unsigned dim = sizes[s];
      struct pipe_resource *bgra =
         make_2d(screen, PIPE_FORMAT_B8G8R8A8_UNORM, dim, dim,
                 PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
      struct pipe_resource *rgba =
         make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, dim, dim,
                 PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
      if (!bgra || !rgba) {
         fprintf(stderr, "BGRA->RGBA %ux%u resource_create failed\n", dim, dim);
         pipe_resource_reference(&bgra, NULL);
         pipe_resource_reference(&rgba, NULL);
         return false;
      }

      struct pipe_framebuffer_state bfb = {
         .width = dim,
         .height = dim,
         .layers = 1,
         .samples = 1,
         .nr_cbufs = 1,
      };
      bfb.cbufs[0] = (struct pipe_surface){
         .texture = bgra,
         .format = PIPE_FORMAT_B8G8R8A8_UNORM,
      };
      pctx->set_framebuffer_state(pctx, &bfb);
      pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &c0, 0.0, 0);

      struct pipe_blit_info binfo = {
         .src = {
            .resource = bgra,
            .level = 0,
            .format = PIPE_FORMAT_B8G8R8A8_UNORM,
            .box = {.width = dim, .height = dim, .depth = 1},
         },
         .dst = {
            .resource = rgba,
            .level = 0,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .box = {.width = dim, .height = dim, .depth = 1},
         },
         .mask = PIPE_MASK_RGBA,
         .filter = PIPE_TEX_FILTER_NEAREST,
      };
      pctx->blit(pctx, &binfo);
      pctx->flush(pctx, NULL, 0);

      if (check_pixels) {
         struct pipe_transfer *xfer = NULL;
         uint8_t *map = pipe_texture_map(pctx, rgba, 0, 0, PIPE_MAP_READ, 0, 0,
                                         dim, dim, &xfer);
         if (!map || !xfer) {
            fprintf(stderr, "BGRA->RGBA %ux%u transfer_map failed\n", dim, dim);
            pipe_resource_reference(&bgra, NULL);
            pipe_resource_reference(&rgba, NULL);
            return false;
         }
         for (unsigned y = 0; y < dim; ++y) {
            const uint8_t *row = map + y * xfer->stride;
            for (unsigned x = 0; x < dim; ++x) {
               const uint8_t *px = row + x * 4;
               if (memcmp(px, want, 4) != 0) {
                  fprintf(stderr,
                          "BGRA->RGBA %ux%u mismatch at (%u,%u): got "
                          "(%u,%u,%u,%u) want (%u,%u,%u,%u)\n",
                          dim, dim, x, y, px[0], px[1], px[2], px[3], want[0],
                          want[1], want[2], want[3]);
                  pipe_texture_unmap(pctx, xfer);
                  pipe_resource_reference(&bgra, NULL);
                  pipe_resource_reference(&rgba, NULL);
                  return false;
               }
            }
         }
         pipe_texture_unmap(pctx, xfer);
      }

      pctx->set_framebuffer_state(pctx, &empty_fb);
      pipe_resource_reference(&bgra, NULL);
      pipe_resource_reference(&rgba, NULL);
   }

   printf("invalidate + BGRA8->RGBA8 readback check passed (check_pixels=%d)\n",
          check_pixels);
   return true;
}

static bool
check_rgba8_readback(struct pipe_context *pctx, struct pipe_resource *rsrc,
                     unsigned w, unsigned h, const uint8_t expected[4],
                     bool check_pixels, const char *label)
{
   struct pipe_transfer *xfer = NULL;
   struct pipe_box box = {.x = 0, .y = 0, .z = 0, .width = w, .height = h, .depth = 1};
   const uint8_t *map =
      pctx->texture_map(pctx, rsrc, 0, PIPE_MAP_READ, &box, &xfer);
   if (!map || !xfer) {
      fprintf(stderr, "%s readback map failed\n", label);
      return false;
   }

   bool ok = true;
   if (check_pixels) {
      for (unsigned y = 0; y < h && ok; ++y) {
         const uint8_t *row = map + y * xfer->stride;
         for (unsigned x = 0; x < w; ++x) {
            const uint8_t *px = row + x * 4;
            if (memcmp(px, expected, 4) != 0) {
               fprintf(stderr,
                       "%s pixel mismatch at (%u,%u): got (%u,%u,%u,%u), "
                       "expected (%u,%u,%u,%u)\n",
                       label, x, y, px[0], px[1], px[2], px[3], expected[0],
                       expected[1], expected[2], expected[3]);
               ok = false;
               break;
            }
         }
      }
   }
   pctx->texture_unmap(pctx, xfer);
   return ok;
}

static bool
check_fast_clear_batch(struct pipe_screen *screen, struct pipe_context *pctx,
                       bool check_pixels)
{
   struct agx_device *dev = agx_device(screen);
   struct agx_context *fc_ctx = agx_context(pctx);
   const union pipe_color_union colour = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   uint8_t expected_rgba[4] = {0};
   util_format_pack_rgba(PIPE_FORMAT_R8G8B8A8_UNORM, expected_rgba, colour.f, 1);

   /* 1. 1x compressed RGBA8 colour target: initial partial clear sets
    * data_valid so batch->load gets PIPE_CLEAR_COLOR0, then full-extent clear
    * with an active TIME_ELAPSED query must convert to CDM, flush, and read
    * back the exact clear colour. */
   struct pipe_resource *c1x =
      make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 256, 256,
              PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
   struct pipe_resource *c1x_rb =
      make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 16, 16,
              PIPE_BIND_RENDER_TARGET);
   if (!c1x || !c1x_rb) {
      fprintf(stderr, "fast-clear 1x resource_create failed\n");
      return false;
   }

   bool ok = true;
   unsigned cdm_cases = 0;
   const bool fc_4b_1x =
      agx_precomp_launchable(fc_ctx, LIBAGX_FAST_CLEAR(2, 0));
   const bool fc_4b_4x =
      agx_precomp_launchable(fc_ctx, LIBAGX_FAST_CLEAR(2, 2));
   const bool fc_2b_1x =
      agx_precomp_launchable(fc_ctx, LIBAGX_FAST_CLEAR(1, 0));
   const bool fc_1b_1x =
      agx_precomp_launchable(fc_ctx, LIBAGX_FAST_CLEAR(0, 0));
   if (fc_4b_1x && agx_resource(c1x)->layout.compressed) {
      struct pipe_framebuffer_state partial_fb = {
         .width = 16,
         .height = 16,
         .layers = 1,
         .samples = 1,
         .nr_cbufs = 1,
         .cbufs[0] = {.texture = c1x, .format = PIPE_FORMAT_R8G8B8A8_UNORM},
      };
      const union pipe_color_union seed = {.f = {0.9f, 0.1f, 0.1f, 1.0f}};
      pctx->set_framebuffer_state(pctx, &partial_fb);
      pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &seed, 0.0, 0);
      pctx->flush(pctx, NULL, 0);

      struct pipe_framebuffer_state full_fb = partial_fb;
      full_fb.width = 256;
      full_fb.height = 256;
      pctx->set_framebuffer_state(pctx, &full_fb);

      struct pipe_query *tq =
         pctx->create_query(pctx, PIPE_QUERY_TIME_ELAPSED, 0);
      if (tq)
         pctx->begin_query(pctx, tq);
      pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &colour, 0.0, 0);
      struct agx_batch *fc_batch = fc_ctx->batch;
      if (!fc_batch || !(fc_batch->load & PIPE_CLEAR_COLOR0) ||
          (tq && fc_batch->timestamps.size == 0) ||
          !agx_try_fast_clear_batch(fc_ctx, fc_batch) ||
          fc_batch->vdm.bo != NULL || fc_batch->cdm.bo == NULL) {
         fprintf(stderr,
                 "1x compressed clear did not convert to CDM "
                 "(batch=%p load=%#x ts=%u vdm=%p cdm=%p)\n",
                 (void *)fc_batch, fc_batch ? fc_batch->load : 0,
                 fc_batch ? fc_batch->timestamps.size : 0,
                 fc_batch ? (void *)fc_batch->vdm.bo : NULL,
                 fc_batch ? (void *)fc_batch->cdm.bo : NULL);
         ok = false;
      } else {
         cdm_cases++;
      }
      pctx->flush(pctx, NULL, 0);
      if (tq) {
         pctx->end_query(pctx, tq);
         union pipe_query_result qres = {0};
         if (check_pixels && !pctx->get_query_result(pctx, tq, true, &qres)) {
            fprintf(stderr, "TIME_ELAPSED get_query_result failed\n");
            ok = false;
         }
         pctx->destroy_query(pctx, tq);
      }

      struct pipe_blit_info sample_blit = {
         .src = {
            .resource = c1x,
            .level = 0,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .box = {.x = 0, .y = 0, .z = 0, .width = 16, .height = 16, .depth = 1},
         },
         .dst = {
            .resource = c1x_rb,
            .level = 0,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .box = {.x = 0, .y = 0, .z = 0, .width = 16, .height = 16, .depth = 1},
         },
         .mask = PIPE_MASK_RGBA,
         .filter = PIPE_TEX_FILTER_NEAREST,
      };
      pctx->blit(pctx, &sample_blit);
      pctx->flush(pctx, NULL, 0);
      ok &= check_rgba8_readback(pctx, c1x_rb, 16, 16, expected_rgba,
                                 check_pixels, "1x RGBA8 fast-clear");
   }
   pipe_resource_reference(&c1x, NULL);
   pipe_resource_reference(&c1x_rb, NULL);

   /* 2. 4x MSAA compressed colour target: full-extent clear -> CDM ->
    * resolve blit -> readback pixel compare. */
   struct pipe_resource msaa_tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 64,
      .height0 = 64,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = 4,
      .nr_storage_samples = 4,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *msaa_fc = screen->resource_create(screen, &msaa_tmpl);
   struct pipe_resource *msaa_rb =
      make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 64,
              PIPE_BIND_RENDER_TARGET);
   if (!msaa_fc || !msaa_rb) {
      fprintf(stderr, "fast-clear 4x MSAA resource_create failed\n");
      pipe_resource_reference(&msaa_fc, NULL);
      pipe_resource_reference(&msaa_rb, NULL);
      return false;
   }
   if (fc_4b_4x && agx_resource(msaa_fc)->layout.compressed) {
      struct pipe_framebuffer_state msaa_fb = {
         .width = 64,
         .height = 64,
         .layers = 1,
         .samples = 4,
         .nr_cbufs = 1,
         .cbufs[0] = {.texture = msaa_fc, .format = PIPE_FORMAT_R8G8B8A8_UNORM},
      };
      pctx->set_framebuffer_state(pctx, &msaa_fb);
      pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &colour, 0.0, 0);
      struct agx_batch *ms_batch = fc_ctx->batch;
      if (!ms_batch || !agx_try_fast_clear_batch(fc_ctx, ms_batch) ||
          ms_batch->vdm.bo != NULL || ms_batch->cdm.bo == NULL) {
         fprintf(stderr, "4x MSAA compressed clear did not convert to CDM\n");
         ok = false;
      } else {
         cdm_cases++;
      }
      pctx->flush(pctx, NULL, 0);

      struct pipe_blit_info msaa_resolve = {
         .src = {
            .resource = msaa_fc,
            .level = 0,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .box = {.x = 0, .y = 0, .z = 0, .width = 64, .height = 64, .depth = 1},
         },
         .dst = {
            .resource = msaa_rb,
            .level = 0,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .box = {.x = 0, .y = 0, .z = 0, .width = 64, .height = 64, .depth = 1},
         },
         .mask = PIPE_MASK_RGBA,
         .filter = PIPE_TEX_FILTER_NEAREST,
      };
      pctx->blit(pctx, &msaa_resolve);
      pctx->flush(pctx, NULL, 0);
      ok &= check_rgba8_readback(pctx, msaa_rb, 64, 64, expected_rgba,
                                 check_pixels, "4x MSAA RGBA8 fast-clear");
   }
   pipe_resource_reference(&msaa_fc, NULL);
   pipe_resource_reference(&msaa_rb, NULL);

   /* 3. Compressed depth (Z32_FLOAT and Z16_UNORM): full-extent clear -> CDM
    * -> readback compare. */
   const enum pipe_format z_fmts[2] = {
      PIPE_FORMAT_Z32_FLOAT,
      PIPE_FORMAT_Z16_UNORM,
   };
   const double z_vals[2] = {0.375, 0.25};
   for (unsigned i = 0; i < ARRAY_SIZE(z_fmts); ++i) {
      struct pipe_resource *zres =
         make_2d(screen, z_fmts[i], 256, 256,
                 PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW);
      if (!zres) {
         fprintf(stderr, "fast-clear depth resource_create failed\n");
         ok = false;
         continue;
      }
      const bool fc_z =
         z_fmts[i] == PIPE_FORMAT_Z32_FLOAT ? fc_4b_1x : fc_2b_1x;
      if (fc_z && agx_resource(zres)->layout.compressed) {
         struct pipe_framebuffer_state zfb = {
            .width = 256,
            .height = 256,
            .layers = 1,
            .samples = 1,
            .zsbuf = {.texture = zres, .format = z_fmts[i]},
         };
         pctx->set_framebuffer_state(pctx, &zfb);
         pctx->clear(pctx, PIPE_CLEAR_DEPTH, 0, 0, NULL, NULL, z_vals[i], 0);
         struct agx_batch *zbatch = fc_ctx->batch;
         if (!zbatch || !agx_try_fast_clear_batch(fc_ctx, zbatch) ||
             zbatch->vdm.bo != NULL || zbatch->cdm.bo == NULL) {
            fprintf(stderr, "compressed %s clear did not convert to CDM\n",
                    util_format_short_name(z_fmts[i]));
            ok = false;
         } else {
            cdm_cases++;
         }
         pctx->flush(pctx, NULL, 0);

         struct pipe_transfer *zxfer = NULL;
         struct pipe_box zbox = {
            .x = 0, .y = 0, .z = 0, .width = 16, .height = 16, .depth = 1,
         };
         const uint8_t *zmap =
            pctx->texture_map(pctx, zres, 0, PIPE_MAP_READ, &zbox, &zxfer);
         if (!zmap || !zxfer) {
            fprintf(stderr, "%s readback map failed\n",
                    util_format_short_name(z_fmts[i]));
            ok = false;
         } else {
            if (check_pixels) {
               for (unsigned y = 0; y < 16 && ok; ++y) {
                  const uint8_t *row = zmap + y * zxfer->stride;
                  for (unsigned x = 0; x < 16; ++x) {
                     if (z_fmts[i] == PIPE_FORMAT_Z32_FLOAT) {
                        float got = 0.0f;
                        memcpy(&got, row + x * 4, 4);
                        if (got != (float)z_vals[i]) {
                           fprintf(stderr,
                                   "Z32_FLOAT mismatch at (%u,%u): got %f, expected %f\n",
                                   x, y, got, z_vals[i]);
                           ok = false;
                           break;
                        }
                     } else {
                        uint16_t got = 0;
                        memcpy(&got, row + x * 2, 2);
                        const uint16_t want = (uint16_t)(z_vals[i] * 65535.0 + 0.5);
                        if (abs((int)got - (int)want) > 1) {
                           fprintf(stderr,
                                   "Z16_UNORM mismatch at (%u,%u): got %u, expected %u\n",
                                   x, y, got, want);
                           ok = false;
                           break;
                        }
                     }
                  }
               }
            }
            pctx->texture_unmap(pctx, zxfer);
         }
      }
      pipe_resource_reference(&zres, NULL);
   }

   /* 4. Combined COLOR|DEPTH|STENCIL on RGBA8 + Z32F_S8: takes CDM when all
    * three attachments (including separate S8_UINT on AGX2) are compressed,
    * and falls back cleanly to VDM when stencil is uncompressed (G17P). */
   struct pipe_resource *cds_c =
      make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM, 256, 256,
              PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
   struct pipe_resource *cds_zs =
      make_2d(screen, PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, 256, 256,
              PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW);
   if (!cds_c || !cds_zs) {
      fprintf(stderr, "fast-clear CDS resource_create failed\n");
      pipe_resource_reference(&cds_c, NULL);
      pipe_resource_reference(&cds_zs, NULL);
      return false;
   }
   struct agx_resource *zs_rsrc = agx_resource(cds_zs);
   struct agx_resource *s_rsrc =
      zs_rsrc->separate_stencil ? zs_rsrc->separate_stencil : zs_rsrc;
   const bool expect_cds_cdm =
      fc_4b_1x && fc_1b_1x && agx_resource(cds_c)->layout.compressed &&
      zs_rsrc->layout.compressed && s_rsrc->layout.compressed;
   struct pipe_framebuffer_state cds_fb = {
      .width = 256,
      .height = 256,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 1,
      .cbufs[0] = {.texture = cds_c, .format = PIPE_FORMAT_R8G8B8A8_UNORM},
      .zsbuf = {.texture = cds_zs, .format = PIPE_FORMAT_Z32_FLOAT_S8X24_UINT},
   };
   const double cds_depth = 0.625;
   const unsigned cds_stencil = 0x5a;
   pctx->set_framebuffer_state(pctx, &cds_fb);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0 | PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL,
               0xf, 0, NULL, &colour, cds_depth, cds_stencil);
   struct agx_batch *cds_batch = fc_ctx->batch;
   const bool cds_converted =
      cds_batch && agx_try_fast_clear_batch(fc_ctx, cds_batch);
   if (cds_converted != expect_cds_cdm) {
      fprintf(stderr,
              "CDS clear CDM conversion=%d, expected=%d (c=%d z=%d s=%d)\n",
              cds_converted, expect_cds_cdm,
              agx_resource(cds_c)->layout.compressed,
              zs_rsrc->layout.compressed, s_rsrc->layout.compressed);
      ok = false;
   } else if (cds_converted) {
      cdm_cases++;
   }
   pctx->flush(pctx, NULL, 0);

   ok &= check_rgba8_readback(pctx, cds_c, 16, 16, expected_rgba, check_pixels,
                              "CDS RGBA8 clear");
   struct pipe_transfer *zs_xfer = NULL;
   struct pipe_box zs_box = {
      .x = 0, .y = 0, .z = 0, .width = 16, .height = 16, .depth = 1,
   };
   const uint8_t *zs_map =
      pctx->texture_map(pctx, cds_zs, 0, PIPE_MAP_READ, &zs_box, &zs_xfer);
   if (!zs_map || !zs_xfer) {
      fprintf(stderr, "CDS Z32F_S8 readback map failed\n");
      ok = false;
   } else {
      if (check_pixels) {
         for (unsigned y = 0; y < 16 && ok; ++y) {
            const uint8_t *row = zs_map + y * zs_xfer->stride;
            for (unsigned x = 0; x < 16; ++x) {
               float got_z = 0.0f;
               memcpy(&got_z, row + x * 8, 4);
               const uint8_t got_s = row[x * 8 + 4];
               if (got_z != (float)cds_depth || got_s != cds_stencil) {
                  fprintf(stderr,
                          "CDS Z32F_S8 mismatch at (%u,%u): got (z=%f, s=%#x), "
                          "expected (z=%f, s=%#x)\n",
                          x, y, got_z, got_s, cds_depth, cds_stencil);
                  ok = false;
                  break;
               }
            }
         }
      }
      pctx->texture_unmap(pctx, zs_xfer);
   }
   pipe_resource_reference(&cds_c, NULL);
   pipe_resource_reference(&cds_zs, NULL);

   printf("fast-clear-batch chip=%s check_pixels=%d cdm_cases=%u cds_cdm=%d "
          "result=%s\n",
          agx_chip_name(dev->chip), check_pixels, cdm_cases, expect_cds_cdm,
          ok ? "PASS" : "FAIL");
   return ok;
}

int
main(void)
{
   const char *node = getenv("AGX_TEST_RENDER_NODE");
   int fd = open(node ? node : "/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      fprintf(stderr, "cannot open render node (is the drm-shim preloaded?)\n");
      return 77;
   }

   /* agx_screen_create() parses driconf unconditionally, so give it the
    * driver's own option descriptions rather than a null cache. */
   static const driOptionDescription asahi_driconf[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache option_info, option_cache;
   driParseOptionInfo(&option_info, asahi_driconf,
                      ARRAY_SIZE(asahi_driconf));

   struct pipe_screen_config config = {
      .options = &option_cache,
      .options_info = &option_info,
   };
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   if (!screen) {
      fprintf(stderr, "agx_screen_create failed\n");
      return 77;
   }

   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   if (!pctx) {
      fprintf(stderr, "context_create failed\n");
      return 1;
   }

   if (getenv("AGX_TEST_FAST_CLEAR_HW")) {
      const bool ok =
         check_fast_clear_batch(screen, pctx, getenv("ASAHI_SHIM_CHIP") == NULL);
      pctx->destroy(pctx);
      screen->destroy(screen);
      close(fd);
      return ok ? 0 : 1;
   }

   if (getenv("AGX_TEST_MAIN_SAMPLE_KEY")) {
      glsl_type_singleton_init_or_ref();
      const bool ok = check_discard_main_sample_key(pctx);
      pctx->destroy(pctx);
      screen->destroy(screen);
      glsl_type_singleton_decref();
      close(fd);
      return ok ? 0 : 1;
   }

   if (getenv("AGX_TEST_VERTEX_OUTPUT_SHAPES")) {
      const bool ok = check_vertex_output_shapes(pctx);
      pctx->destroy(pctx);
      screen->destroy(screen);
      close(fd);
      return ok ? 0 : 1;
   }

   if (getenv("AGX_TEST_MIPMAP_HW")) {
      const bool ok =
         check_agx3_compute_mipmap(screen, pctx, getenv("ASAHI_SHIM_CHIP") == NULL);
      pctx->destroy(pctx);
      screen->destroy(screen);
      close(fd);
      return ok ? 0 : 1;
   }

   if (getenv("AGX_TEST_BGRA_READBACK_HW")) {
      const bool ok = check_invalidate_and_bgra_readback(screen, pctx, true);
      pctx->destroy(pctx);
      screen->destroy(screen);
      close(fd);
      return ok ? 0 : 1;
   }

   if (getenv("AGX_TEST_INTERCHANGE_SHARED_BO")) {
      const bool ok = check_interchange_and_shared_bo(pctx);
      pctx->destroy(pctx);
      screen->destroy(screen);
      close(fd);
      return ok ? 0 : 1;
   }

   if (getenv("AGX_TEST_XFB_COPY_PROGRAM")) {
      glsl_type_singleton_init_or_ref();
      /* A triangle capture shades one of three candidate vertices, so its
       * copy program selects between three loads per varying. That is the
       * shape that runs out of registers, not the one-vertex point case. */
      const bool ok = check_xfb_copy_program(pctx, false, 0, MESA_PRIM_POINTS) &&
                      check_xfb_copy_program(pctx, true, 0, MESA_PRIM_POINTS) &&
                      check_xfb_copy_program(pctx, false, 15, MESA_PRIM_POINTS) &&
                      check_xfb_copy_program(pctx, false, 15, MESA_PRIM_TRIANGLES);
      glsl_type_singleton_decref();
      pctx->destroy(pctx);
      screen->destroy(screen);
      close(fd);
      return ok ? 0 : 1;
   }

   if (!check_g17p_sampler_bias_record())
      return 1;

   if (!check_direct_resolve_shapes())
      return 1;

   if (!check_stencil_blit_shapes())
      return 1;

   if (!check_msaa_depth_blit(pctx, 2) || !check_msaa_depth_blit(pctx, 4))
      return 1;

   if (!check_scissored_blit(pctx))
      return 1;

   if (!check_staged_resolve_shapes())
      return 1;

   if (!check_invalidate_and_bgra_readback(screen, pctx, false))
      return 1;

   if (!check_es3_compile_contract(pctx))
      return 1;
   printf("ES3 bitops/texelFetch and WebGL2 shadow compile contract passed\n");

   if (screen->shader_caps[MESA_SHADER_FRAGMENT].max_texture_samplers !=
          AGX_GALLIUM_G17P_API_SAMPLERS ||
       screen->shader_caps[MESA_SHADER_COMPUTE].max_texture_samplers !=
          AGX_GALLIUM_G17P_API_SAMPLERS ||
       !check_sampler_boundary_compile(pctx) ||
       !check_query_lod_compile(pctx) ||
       !check_query_lod_custom_border_bindings())
      return 1;
   printf("G17P native sampler slots 0..31 compile through the local table\n");

   /* WebGL2 RGB8UI render targets are represented by Gallium as the
    * four-lane RGBX8_UINT physical family. Exercise a complete clear/store
    * batch so format advertisement, tile layout, EOT and PBE agree. */
   struct pipe_resource *rgb8ui =
      make_2d(screen, PIPE_FORMAT_R8G8B8X8_UINT, 8, 8,
              PIPE_BIND_RENDER_TARGET);
   if (!rgb8ui) {
      fprintf(stderr, "RGB8UI render target was not advertised\n");
      return 1;
   }
   struct pipe_surface rgb8ui_surface = {
      .texture = rgb8ui,
      .format = PIPE_FORMAT_R8G8B8X8_UINT,
      .first_layer = 0,
      .last_layer = 0,
      .level = 0,
   };
   struct pipe_framebuffer_state rgb8ui_fb = {
      .width = 8,
      .height = 8,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 1,
   };
   rgb8ui_fb.cbufs[0] = rgb8ui_surface;
   pctx->set_framebuffer_state(pctx, &rgb8ui_fb);
   const union pipe_color_union rgb8ui_clear = {
      .ui = {17, 93, 201, 0},
   };
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &rgb8ui_clear, 0.0, 0);
   pctx->flush(pctx, NULL, 0);
   pipe_resource_reference(&rgb8ui, NULL);
   printf("G17P RGB8UI clear/store completed\n");

   struct pipe_resource xrgb8_msaa_template = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_B8G8R8X8_UNORM,
      .width0 = 8,
      .height0 = 8,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 2,
      .nr_storage_samples = 2,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_resource *xrgb8_msaa =
      screen->resource_create(screen, &xrgb8_msaa_template);
   struct pipe_resource *xrgb8_resolve =
      make_2d(screen, PIPE_FORMAT_B8G8R8X8_UNORM, 8, 8,
              PIPE_BIND_RENDER_TARGET);
   if (!xrgb8_msaa || !xrgb8_resolve) {
      fprintf(stderr, "XRGB8 2x render target was not advertised\n");
      return 1;
   }
   struct pipe_surface xrgb8_msaa_surface = {
      .texture = xrgb8_msaa,
      .format = PIPE_FORMAT_B8G8R8X8_UNORM,
      .first_layer = 0,
      .last_layer = 0,
      .level = 0,
   };
   struct pipe_framebuffer_state xrgb8_msaa_fb = {
      .width = 8,
      .height = 8,
      .layers = 1,
      .samples = 2,
      .nr_cbufs = 1,
      .resolve = xrgb8_resolve,
   };
   xrgb8_msaa_fb.cbufs[0] = xrgb8_msaa_surface;
   pctx->set_framebuffer_state(pctx, &xrgb8_msaa_fb);
   const union pipe_color_union xrgb8_clear = {
      .f = {0.125f, 0.5f, 0.875f, 1.0f},
   };
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &xrgb8_clear, 0.0, 0);
   pctx->flush(pctx, NULL, 0);
   xrgb8_msaa_fb.resolve = NULL;
   pctx->set_framebuffer_state(pctx, &xrgb8_msaa_fb);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &xrgb8_clear, 0.0, 0);
   const struct pipe_blit_info xrgb8_resolve_blit = {
      .src = {
         .resource = xrgb8_msaa,
         .format = PIPE_FORMAT_B8G8R8X8_UNORM,
         .box = {.width = 8, .height = 8, .depth = 1},
      },
      .dst = {
         .resource = xrgb8_resolve,
         .format = PIPE_FORMAT_B8G8R8X8_UNORM,
         .box = {.width = 8, .height = 8, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };
   pctx->blit(pctx, &xrgb8_resolve_blit);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &xrgb8_clear, 0.0, 0);
   pctx->blit(pctx, &xrgb8_resolve_blit);
   pctx->flush(pctx, NULL, 0);
   pipe_resource_reference(&xrgb8_msaa, NULL);
   pipe_resource_reference(&xrgb8_resolve, NULL);
   printf("G17P XRGB8 2x clear/direct-resolve completed\n");

   /* Exercise 4x RGBA16F direct resolve both preserving the MSAA source and
    * discarding it via invalidate_resource after attaching the resolve. */
   struct pipe_resource rgba16f_msaa_template = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R16G16B16A16_FLOAT,
      .width0 = 64,
      .height0 = 64,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 4,
      .nr_storage_samples = 4,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_resource *rgba16f_msaa =
      screen->resource_create(screen, &rgba16f_msaa_template);
   struct pipe_resource *rgba16f_resolve =
      make_2d(screen, PIPE_FORMAT_R16G16B16A16_FLOAT, 64, 64,
              PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW);
   if (!rgba16f_msaa || !rgba16f_resolve) {
      fprintf(stderr, "RGBA16F 4x render target was not advertised\n");
      return 1;
   }
   struct pipe_framebuffer_state rgba16f_msaa_fb = {
      .width = 64,
      .height = 64,
      .layers = 1,
      .samples = 4,
      .nr_cbufs = 1,
   };
   rgba16f_msaa_fb.cbufs[0] = (struct pipe_surface){
      .texture = rgba16f_msaa,
      .format = PIPE_FORMAT_R16G16B16A16_FLOAT,
      .first_layer = 0,
      .last_layer = 0,
      .level = 0,
   };
   const struct pipe_blit_info rgba16f_resolve_blit = {
      .src = {
         .resource = rgba16f_msaa,
         .format = PIPE_FORMAT_R16G16B16A16_FLOAT,
         .box = {.width = 64, .height = 64, .depth = 1},
      },
      .dst = {
         .resource = rgba16f_resolve,
         .format = PIPE_FORMAT_R16G16B16A16_FLOAT,
         .box = {.width = 64, .height = 64, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };
   pctx->set_framebuffer_state(pctx, &rgba16f_msaa_fb);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &xrgb8_clear, 0.0, 0);
   pctx->blit(pctx, &rgba16f_resolve_blit);
   pctx->flush(pctx, NULL, 0);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &xrgb8_clear, 0.0, 0);
   pctx->blit(pctx, &rgba16f_resolve_blit);
   {
      struct agx_context *actx = agx_context(pctx);
      struct agx_batch *rb =
         agx_writer_get(actx, agx_resource(rgba16f_msaa)->bo->handle);
      if (agx_chip_native_agx3(agx_device(screen)->chip)) {
         if (!rb || !rb->key.resolve || !rb->g17p_resolve_blit_attached ||
             !rb->g17p_resolve_preserve) {
            fprintf(stderr,
                    "RGBA16F blit-attached resolve batch did not preserve source\n");
            return 1;
         }
         pctx->invalidate_resource(pctx, rgba16f_msaa);
         if (rb->g17p_resolve_preserve ||
             (rb->resolve & PIPE_CLEAR_COLOR0) ||
             !(rb->clear & PIPE_CLEAR_COLOR0)) {
            fprintf(stderr,
                    "invalidate_resource left unexpected resolve/clear state on blit-attached batch\n");
            return 1;
         }
      } else {
         pctx->invalidate_resource(pctx, rgba16f_msaa);
      }
   }
   pctx->flush(pctx, NULL, 0);

   /* A DRI MSAA visual carries key.resolve on an active ctx->batch that keeps
    * taking draws after glInvalidateFramebuffer(). Invalidating it must NOT
    * clear g17p_resolve_preserve, or later draws before flush lose their MSAA
    * store. */
   rgba16f_msaa_fb.resolve = rgba16f_resolve;
   pctx->set_framebuffer_state(pctx, &rgba16f_msaa_fb);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &xrgb8_clear, 0.0, 0);
   pctx->invalidate_resource(pctx, rgba16f_msaa);
   {
      struct agx_context *actx = agx_context(pctx);
      struct agx_batch *dri_b = actx->batch;
      if (!dri_b || dri_b->key.resolve != rgba16f_resolve ||
          dri_b->g17p_resolve_blit_attached || !dri_b->g17p_resolve_preserve) {
         fprintf(stderr,
                 "invalidate_resource cleared g17p_resolve_preserve on active DRI MSAA batch\n");
         return 1;
      }
   }
   pctx->flush(pctx, NULL, 0);
   pipe_resource_reference(&rgba16f_msaa, NULL);
   pipe_resource_reference(&rgba16f_resolve, NULL);
   printf("G17P RGBA16F 4x clear/direct-resolve/invalidate completed\n");

   struct pipe_resource *mrt_xrgb =
      make_2d(screen, PIPE_FORMAT_R8G8B8X8_UNORM, 8, 8,
              PIPE_BIND_RENDER_TARGET);
   struct pipe_resource *mrt_bgra =
      make_2d(screen, PIPE_FORMAT_B8G8R8A8_UNORM, 8, 8,
              PIPE_BIND_RENDER_TARGET);
   if (!mrt_xrgb || !mrt_bgra) {
      fprintf(stderr, "RGBA8-alias MRT resources were not advertised\n");
      return 1;
   }
   struct pipe_surface mrt_surfaces[2] = {
      {
         .texture = mrt_xrgb,
         .format = PIPE_FORMAT_R8G8B8X8_UNORM,
         .first_layer = 0,
         .last_layer = 0,
         .level = 0,
      },
      {
         .texture = mrt_bgra,
         .format = PIPE_FORMAT_B8G8R8A8_UNORM,
         .first_layer = 0,
         .last_layer = 0,
         .level = 0,
      },
   };
   struct pipe_framebuffer_state alias_mrt_fb = {
      .width = 8,
      .height = 8,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 2,
   };
   alias_mrt_fb.cbufs[0] = mrt_surfaces[0];
   alias_mrt_fb.cbufs[1] = mrt_surfaces[1];
   pctx->set_framebuffer_state(pctx, &alias_mrt_fb);
   pctx->clear(pctx, PIPE_CLEAR_COLOR0 | PIPE_CLEAR_COLOR1, 0xf, 0, NULL,
               &xrgb8_clear, 0.0, 0);
   pctx->flush(pctx, NULL, 0);
   pipe_resource_reference(&mrt_xrgb, NULL);
   pipe_resource_reference(&mrt_bgra, NULL);
   printf("G17P XRGB8/BGRA8 MRT clear/store completed\n");

   /* Exactly the blit core 983 died on: 4x8, BGRA8_UNORM source, RGBA8_UNORM
    * destination, one level, one layer, nearest. u_blitter turns this into a
    * texel-fetch fragment shader over a pass-through vertex shader and a
    * triangle fan, which is where the null program was consumed. */
   struct pipe_resource *src =
      make_2d(screen, PIPE_FORMAT_B8G8R8A8_UNORM, 4, 8, PIPE_BIND_SAMPLER_VIEW);
   /* The native hardware oracle uses a 2408x1506 target. Unlike the old 4x8
    * fixture, that extent is compression-eligible, so it also reproduces the
    * real end-of-tile PBE descriptor path while the blit rectangle stays 4x8.
    */
   struct pipe_resource *dst = make_2d(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                       2408, 1506, PIPE_BIND_RENDER_TARGET);
   if (!src || !dst) {
      fprintf(stderr, "resource_create failed\n");
      return 1;
   }

   struct pipe_blit_info info = {
      .src = {
         .resource = src,
         .level = 0,
         .format = PIPE_FORMAT_B8G8R8A8_UNORM,
         .box = {.x = 0, .y = 0, .z = 0, .width = 4, .height = 8, .depth = 1},
      },
      .dst = {
         .resource = dst,
         .level = 0,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .box = {.x = 0, .y = 0, .z = 0, .width = 4, .height = 8, .depth = 1},
      },
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
      .scissor_enable = false,
      .render_condition_enable = false,
   };

   /* A clear and a flush first: no API shader is involved, so this is the
    * background/end-of-tile programs, the PBE render-target descriptor and
    * the batch submit path on their own. It used to abort the process in
    * agx_native_descriptor_rejected("end-of-tile PBE") before any draw
    * happened at all.
    */
   struct pipe_surface cbuf = {
      .texture = dst,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .first_layer = 0,
      .last_layer = 0,
      .level = 0,
   };
   struct pipe_framebuffer_state fb = {
      .width = 4,
      .height = 8,
      .layers = 1,
      .samples = 1,
      .nr_cbufs = 1,
   };
   fb.cbufs[0] = cbuf;
   pctx->set_framebuffer_state(pctx, &fb);

   const union pipe_color_union colour = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &colour, 0.0, 0);
   pctx->flush(pctx, NULL, 0);
   printf("clear and flush completed\n");

   /* Full-extent compressed clear-only batch (1x/4x colour, depth, and
    * combined colour+depth+stencil) must convert to libagx_fast_clear() CDM
    * dispatches, attach active TIME_ELAPSED queries, and read back exact
    * clear values when run on real hardware. */
   if (!check_fast_clear_batch(screen, pctx, getenv("ASAHI_SHIM_CHIP") == NULL))
      return 1;
   pctx->set_framebuffer_state(pctx, &fb);
   printf("full-extent compressed fast-clear CDM path verified\n");

   pctx->blit(pctx, &info);

   /* G17P's one admitted precompiled kernel is the vector buffer fill used
    * by clear_buffer. It feeds later graphics, so AGX_BARRIER_ALL must survive
    * native CDM lowering as the strong flags-5 cache barrier. The shim audits
    * the actual production stream and its nested Ioto argument source. */
   struct pipe_resource *filled = make_buffer(screen, 256);
   const uint32_t fill[4] = {
      UINT32_C(0x11223344), UINT32_C(0x55667788),
      UINT32_C(0x99aabbcc), UINT32_C(0xddeeff00),
   };
   if (!filled) {
      fprintf(stderr, "buffer resource_create failed\n");
      return 1;
   }
   pctx->clear_buffer(pctx, filled, 0, 256, fill, sizeof(fill));
   pctx->flush(pctx, NULL, 0);

   /* Native AGX3 compute mipchain generation: predicate matrix, CDM decode
    * verifying fused 2-level (16x8) selection, and even/odd/1xN/array/cube
    * readback (CPU-reference pixel verification when running on hardware). */
   if (!check_agx3_compute_mipmap(screen, pctx,
                                  getenv("ASAHI_SHIM_CHIP") == NULL))
      return 1;

   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(screen);
   const bool agx2 =
      agx_chip_command_stream_generation(dev->chip) == AGX_CS_GENERATION_AGX2;
   int status = 0;

   /* AGX2, Gen9 AGX3 and the M5 profile advertise blit-based texture
    * transfers; other Gen10 chips keep mapped CPU transfers while their PBE
    * encoder refuses sRGB, mipmapped (non-G18), and linear layered (G18)
    * transfer targets. */
   const enum agx_descriptor_generation desc_gen =
      agx_chip_descriptor_generation(dev->chip);
   const unsigned want_transfer_modes =
      (agx_gallium_descriptor_generation_is_agx2(desc_gen) ||
       desc_gen == AGX_DESCRIPTOR_GENERATION_9 || agx_chip_g17_m5(dev->chip))
         ? PIPE_TEXTURE_TRANSFER_BLIT
         : PIPE_TEXTURE_TRANSFER_DEFAULT;
   if (screen->caps.texture_transfer_modes != want_transfer_modes ||
       !screen->caps.invalidate_pbo_upload_destination) {
      fprintf(stderr,
              "texture transfer caps (%#x, inv_pbo=%d) on %s, expected (%#x, 1)\n",
              screen->caps.texture_transfer_modes,
              screen->caps.invalidate_pbo_upload_destination,
              agx_chip_name(dev->chip), want_transfer_modes);
      status = 1;
   }

   const bool vs_ok = ctx->vs != NULL && ctx->vs->bo != NULL;
   const bool fs_ok = ctx->fs != NULL && ctx->fs->bo != NULL;
   const bool linked_ok = ctx->linked.vs != NULL && ctx->linked.fs != NULL;

   printf("blit returned on %s; vs=%s fs=%s linked=%s\n",
          agx_chip_name(dev->chip), vs_ok ? "ok" : "refused",
          fs_ok ? "ok" : "refused", linked_ok ? "ok" : "refused");

   /* Gen9 colour-compression parts (G17P, G16G) and Gen10 ones
    * (AGX_CAP_GEN10_2D_COMPRESSION, the M5) compress this 2408x1506 RGBA8
    * render target, so the clear, the flush and the blit above took the
    * compressed end-of-tile PBE and background texture; an unencodable
    * descriptor would have aborted in agx_native_descriptor_rejected(). Other
    * AGX3 parts keep it plain. */
   if (!agx2 &&
       agx_resource(dst)->layout.compressed !=
          (agx_chip_gen9_color_compression(dev->chip) ||
           agx_chip_has_cap(dev->chip, AGX_CAP_GEN10_2D_COMPRESSION))) {
      fprintf(stderr,
              "AGX3 render target compression does not follow the colour "
              "compression policy\n");
      status = 1;
   }

   /* Verify DRM_FORMAT_MOD_APPLE_INTERCHANGE modifier queries, external_only
    * flags, scanout allocation, and shared-BO decompression safety. */
   if (!check_interchange_and_shared_bo(pctx))
      status = 1;

   /* AGX3-encoding parts must not take Gallium's AGX2 compute blitter.
    * The render blitter binds a vertex stage; the compute one never does, so
    * a null ctx->vs here means asahi_compute_blit() ran. */
   if (!agx2 && ctx->vs == NULL) {
      fprintf(stderr, "blit took the compute path on a non-AGX2 chip\n");
      status = 1;
   }

   /* A ratchet on what already works. u_blitter's pass-through vertex shader
    * moves a 16-bit uniform, which had no encoding until the AGX3b uniform
    * copy landed; if this regresses, every blit loses its vertex program
    * again. The fragment program has its own encode ratchet below so the
    * crash-safety check remains independently useful.
    */
   if (!agx2 && !vs_ok) {
      fprintf(stderr, "the blit's vertex program no longer encodes\n");
      status = 1;
   }

   /* Whatever the shaders did, a refused program must have been counted as a
    * skipped draw rather than silently rendering nothing. */
   if (!fs_ok && ctx->skipped_no_program == 0) {
      fprintf(stderr, "a refused program did not skip its draw\n");
      status = 1;
   }

   /* The stronger claim, kept separate so a compiler regression cannot hide
    * behind "at least it did not crash". */
   if (!agx2 && getenv("AGX_G17P_BLIT_REQUIRE_ENCODE")) {
      if (!vs_ok || !fs_ok || !linked_ok) {
         fprintf(stderr,
                 "blit shaders did not encode (vs=%d fs=%d linked=%d)\n",
                 vs_ok, fs_ok, linked_ok);
         status = 1;
      }
   }

   pipe_resource_reference(&src, NULL);
   pipe_resource_reference(&dst, NULL);
   pipe_resource_reference(&filled, NULL);
   pctx->destroy(pctx);
   screen->destroy(screen);
   close(fd);
   return status;
}
