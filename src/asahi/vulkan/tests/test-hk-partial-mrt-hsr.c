/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "hk_partial_mrt_hsr.h"
#include "util/u_math.h"

static unsigned failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static void
test_classify_clear_value(void)
{
   const uint32_t f_zero[4] = {0u, 0u, 0u, 0u};
   const uint32_t f_neg_zero[4] = {0x80000000u, 0u, 0u, 0u};
   const uint32_t f_black_opaque[4] = {0u, 0u, 0u, 0x3f800000u};
   const uint32_t f_white[4] = {0x3f800000u, 0x3f800000u, 0x3f800000u,
                                0x3f800000u};
   const uint32_t f_blue[4] = {0u, 0u, 0x3f800000u, 0x3f800000u};
   const uint32_t i_black_opaque[4] = {0u, 0u, 0u, 1u};
   const uint32_t i_white[4] = {1u, 1u, 1u, 1u};
   const uint32_t i_neg_one[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu,
                                  0xffffffffu};

   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R8G8B8A8_UNORM,
                                             f_zero) ==
         AGX_EPILOG_CLEAR_FILL_ZERO);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R16G16B16A16_FLOAT,
                                             f_black_opaque) ==
         AGX_EPILOG_CLEAR_FILL_BLACK_OPAQUE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_B8G8R8A8_SRGB,
                                             f_white) ==
         AGX_EPILOG_CLEAR_FILL_WHITE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R8G8B8A8_UNORM,
                                             f_blue) ==
         AGX_EPILOG_CLEAR_FILL_NONE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R16G16B16A16_FLOAT,
                                             f_neg_zero) ==
         AGX_EPILOG_CLEAR_FILL_NONE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_A8_UNORM,
                                             f_black_opaque) ==
         AGX_EPILOG_CLEAR_FILL_BLACK_OPAQUE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_A8_UNORM, f_zero) ==
         AGX_EPILOG_CLEAR_FILL_ZERO);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R32_UINT, f_zero) ==
         AGX_EPILOG_CLEAR_FILL_ZERO);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R32G32B32A32_UINT,
                                             i_black_opaque) ==
         AGX_EPILOG_CLEAR_FILL_BLACK_OPAQUE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R32G32B32A32_SINT,
                                             i_white) ==
         AGX_EPILOG_CLEAR_FILL_WHITE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_R32G32B32A32_SINT,
                                             i_neg_one) ==
         AGX_EPILOG_CLEAR_FILL_NONE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_Z32_FLOAT, f_zero) ==
         AGX_EPILOG_CLEAR_FILL_NONE);
   CHECK(hk_partial_mrt_classify_clear_value(PIPE_FORMAT_NONE, f_zero) ==
         AGX_EPILOG_CLEAR_FILL_NONE);
}

static void
test_pass_type_selection(void)
{
   uint8_t opq = 0;

   /* DONT_CARE unwritten RT4 (preserved_color_mask == 0x0f): promote to OPAQUE. */
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT, true, false, false, false, false, 0x0f,
            0x0f, &opq) == AGX_PASS_TYPE_OPAQUE);
   CHECK(opq == 0x0f);

   /* Live BG or prior user write on unwritten RT4 (preserved_color_mask == 0x1f):
    * must stay TRANSLUCENT so ISP does not skip BG or clobber RT4. */
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT, true, false, false, false, false, 0x0f,
            0x1f, &opq) == AGX_PASS_TYPE_TRANSLUCENT);
   CHECK(opq == 0);

   /* Cross-draw opaque conflict: Draw 1 writes {0,1} OPAQUE, Draw 2 writes {0}
    * only (even with DONT_CARE preserved mask) -> must flush as TRANSLUCENT. */
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_OPAQUE, false, false, false, false, false, 0x03, 0x00,
            &opq) == AGX_PASS_TYPE_OPAQUE);
   CHECK(opq == 0x03);
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT, true, false, false, false, false, 0x01,
            0x00, &opq) == AGX_PASS_TYPE_TRANSLUCENT);
   CHECK(opq == 0);

   /* Punch-through promotion and full-quad override. */
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH, true, false, false, false,
            false, 0x0f, 0x0f, &opq) == AGX_PASS_TYPE_PUNCH_THROUGH);
   CHECK(opq == 0x0f);
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH, true, false, false, false,
            true, 0x0f, 0x0f, &opq) == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH);
   CHECK(opq == 0);

   /* force_translucent (vkCmdClearAttachments) flushes pending opaque writes. */
   opq = 0x1f;
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_OPAQUE, false, false, false, true, false, 0x1f, 0x1f,
            &opq) == AGX_PASS_TYPE_TRANSLUCENT);
   CHECK(opq == 0);

   /* tag_write_disable and rasterizer_discard leave opaque_color_written intact. */
   opq = 0x0f;
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT, true, true, false, false, false, 0x01,
            0x1f, &opq) == AGX_PASS_TYPE_TRANSLUCENT);
   CHECK(opq == 0x0f);
   CHECK(hk_partial_mrt_hsr_select_pass_type(
            AGX_PASS_TYPE_TRANSLUCENT, true, false, true, false, false, 0x01,
            0x1f, &opq) == AGX_PASS_TYPE_TRANSLUCENT);
   CHECK(opq == 0x0f);
}

static unsigned
count_tile_stores(const nir_shader *s)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_store_local_pixel_agx)
               count++;
         }
      }
   }
   return count;
}

static void
test_epilog_clear_fill_nir(void)
{
   const unsigned opaque_blend = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   /* Mirror render.deferred 5-RT G-buffer layout: RT0..3 written by shader,
    * RT4 bound and cleared to (0,0,0,0). Test both 1x and 4x MSAA. */
   const unsigned sample_counts[2] = {1, 4};
   for (unsigned s = 0; s < 2; ++s) {
      struct agx_fs_epilog_key key = {
         .link.loc_written = 0x0f,
         .link.size_32 = 0x0f,
         .nr_samples = sample_counts[s],
         .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
         .native_imageblock = true,
         .max_bytes_per_tile = 32768,
         .max_samples_per_tile = 1024,
         .rt_formats = {
            PIPE_FORMAT_R8G8B8A8_UNORM,
            PIPE_FORMAT_R8G8B8A8_UNORM,
            PIPE_FORMAT_R16G16B16A16_FLOAT,
            PIPE_FORMAT_R8G8B8A8_UNORM,
            PIPE_FORMAT_R8G8B8A8_UNORM,
         },
         .remap = {0, 1, 2, 3, -1, -1, -1, -1},
      };
      for (unsigned rt = 0; rt < 5; ++rt) {
         key.blend.rt[rt] = (struct agx_blend_rt_key){
            .colormask = 0xf,
            .mode = opaque_blend,
         };
      }
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &key);

      /* Without clear_fill_rt: RT4 unwritten -> epilog forces translucent
       * solely because of the unwritten RT. */
      struct agx_fs_epilog_result res_plain = {0};
      nir_builder b_plain =
         nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                        "epilog_plain");
      agx_nir_fs_epilog_ex(&b_plain, &key, &res_plain);
      CHECK(b_plain.shader->info.fs.uses_fbfetch_output == true);
      CHECK(res_plain.translucent_only_for_unwritten_rts == true);
      CHECK(res_plain.color_rts_written == 0x0f);
      CHECK(count_tile_stores(b_plain.shader) == 4);
      ralloc_free(b_plain.shader);

      /* With clear_fill_rt on RT4 (ZERO): all 5 bound RTs stored -> OPAQUE. */
      key.clear_fill_rt = (uint16_t)(AGX_EPILOG_CLEAR_FILL_ZERO << (4 * 2));
      struct agx_fs_epilog_result res_fill = {0};
      nir_builder b_fill =
         nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                        "epilog_clear_fill");
      agx_nir_fs_epilog_ex(&b_fill, &key, &res_fill);
      CHECK(b_fill.shader->info.fs.uses_fbfetch_output == false);
      CHECK(res_fill.translucent_only_for_unwritten_rts == false);
      CHECK(res_fill.color_rts_written == 0x1f);
      CHECK(count_tile_stores(b_fill.shader) == 5);
      ralloc_free(b_fill.shader);
   }

   /* Verify that any other translucency cause clears
    * translucent_only_for_unwritten_rts even when an RT is unwritten. */
   const struct agx_fs_epilog_key base_partial = {
      .link.loc_written = 0x0f,
      .link.size_32 = 0x0f,
      .nr_samples = 1,
      .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
      .native_imageblock = true,
      .max_bytes_per_tile = 32768,
      .max_samples_per_tile = 1024,
      .rt_formats = {
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R16G16B16A16_FLOAT,
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
      },
      .remap = {0, 1, 2, 3, -1, -1, -1, -1},
      .blend.rt = {
         {.colormask = 0xf, .mode = opaque_blend},
         {.colormask = 0xf, .mode = opaque_blend},
         {.colormask = 0xf, .mode = opaque_blend},
         {.colormask = 0xf, .mode = opaque_blend},
         {.colormask = 0xf, .mode = opaque_blend},
      },
   };

   /* Blending on RT0 -> not solely unwritten RTs. */
   {
      struct agx_fs_epilog_key k = base_partial;
      k.blend.rt[0].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA,
         PIPE_BLENDFACTOR_INV_SRC_ALPHA, PIPE_BLEND_ADD,
         PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &k);
      struct agx_fs_epilog_result r = {0};
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "epilog_blend");
      agx_nir_fs_epilog_ex(&b, &k, &r);
      CHECK(b.shader->info.fs.uses_fbfetch_output == true);
      CHECK(r.translucent_only_for_unwritten_rts == false);
      ralloc_free(b.shader);
   }

   /* Partial colormask on RT0 -> not solely unwritten RTs. */
   {
      struct agx_fs_epilog_key k = base_partial;
      k.blend.rt[0].colormask = 0x7;
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &k);
      struct agx_fs_epilog_result r = {0};
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "epilog_mask");
      agx_nir_fs_epilog_ex(&b, &k, &r);
      CHECK(b.shader->info.fs.uses_fbfetch_output == true);
      CHECK(r.translucent_only_for_unwritten_rts == false);
      ralloc_free(b.shader);
   }

   /* Explicit preserve_rt -> not solely unwritten RTs. */
   {
      struct agx_fs_epilog_key k = base_partial;
      k.preserve_rt = BITFIELD_BIT(4);
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &k);
      struct agx_fs_epilog_result r = {0};
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "epilog_preserve");
      agx_nir_fs_epilog_ex(&b, &k, &r);
      CHECK(b.shader->info.fs.uses_fbfetch_output == true);
      CHECK(r.translucent_only_for_unwritten_rts == false);
      ralloc_free(b.shader);
   }

   /* Logic op XOR -> not solely unwritten RTs. */
   {
      struct agx_fs_epilog_key k = base_partial;
      k.blend.logicop_enable = true;
      k.blend.logicop_func = PIPE_LOGICOP_XOR;
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &k);
      struct agx_fs_epilog_result r = {0};
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "epilog_logicop");
      agx_nir_fs_epilog_ex(&b, &k, &r);
      CHECK(b.shader->info.fs.uses_fbfetch_output == true);
      CHECK(r.translucent_only_for_unwritten_rts == false);
      ralloc_free(b.shader);
   }

   /* Deferred API sample mask -> not solely unwritten RTs. */
   {
      struct agx_fs_epilog_key k = base_partial;
      k.defer_api_sample_mask = true;
      k.api_sample_mask = 0x5;
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &k);
      struct agx_fs_epilog_result r = {0};
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "epilog_sample_mask");
      agx_nir_fs_epilog_ex(&b, &k, &r);
      CHECK(b.shader->info.fs.uses_fbfetch_output == true);
      CHECK(r.translucent_only_for_unwritten_rts == false);
      ralloc_free(b.shader);
   }

   /* Pure-integer attachment clear-fill (BLACK_OPAQUE + WHITE) with non-opaque
    * app blend state ignored on the clear-filled RT. */
   struct agx_fs_epilog_key ikey = {
      .link.loc_written = 0x01,
      .link.size_32 = 0x01,
      .nr_samples = 1,
      .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
      .native_imageblock = true,
      .max_bytes_per_tile = 32768,
      .max_samples_per_tile = 1024,
      .rt_formats = {
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R32_UINT,
         PIPE_FORMAT_R16G16B16A16_SINT,
      },
      .remap = {0, -1, -1, -1, -1, -1, -1, -1},
      .clear_fill_rt =
         (uint16_t)((AGX_EPILOG_CLEAR_FILL_WHITE << (1 * 2)) |
                    (AGX_EPILOG_CLEAR_FILL_BLACK_OPAQUE << (2 * 2))),
   };
   ikey.blend.rt[0] = (struct agx_blend_rt_key){
      .colormask = 0xf,
      .mode = opaque_blend,
   };
   ikey.blend.rt[1] = (struct agx_blend_rt_key){.colormask = 0};
   ikey.blend.rt[2] = (struct agx_blend_rt_key){.colormask = 0};
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &ikey);

   nir_builder b_int = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "epilog_clear_fill_int");
   agx_nir_fs_epilog(&b_int, &ikey);
   CHECK(b_int.shader->info.fs.uses_fbfetch_output == false);
   CHECK(count_tile_stores(b_int.shader) == 3);
   ralloc_free(b_int.shader);
}

int
main(void)
{
   test_classify_clear_value();
   test_pass_type_selection();
   test_epilog_clear_fill_nir();
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
