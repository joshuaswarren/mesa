/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "agx_vdm_hal200_direct_draw_core.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
init_cmd(struct hk_device *dev, struct hk_cmd_buffer *cmd, enum agx_chip chip,
         bool robust)
{
   memset(dev, 0, sizeof(*dev));
   memset(cmd, 0, sizeof(*cmd));
   dev->dev.chip = chip;
   dev->vk.enabled_features.robustBufferAccess = robust;
   dev->vk.enabled_features.robustBufferAccess2 = robust;
   cmd->vk.base.device = &dev->vk;
}

int
main(void)
{
   struct hk_device dev;
   struct hk_cmd_buffer cmd;

   /* With robustness enabled, G17P (HAL200) and M5 (G17G/G17S/G17C) bound
    * aligned 16/32-bit indirect draws within 32-bit byte bounds in hardware;
    * oversized, unaligned, and 8-bit indirect draws take the software rewrite
    * path. */
   static const enum agx_chip native_chips[] = {AGX_CHIP_G17P, AGX_CHIP_G17G,
                                                AGX_CHIP_G17S, AGX_CHIP_G17C};
   for (unsigned i = 0; i < sizeof(native_chips) / sizeof(native_chips[0]);
        ++i) {
      assert(agx_chip_vdm_indirect_index_bounds(native_chips[i]));
      init_cmd(&dev, &cmd, native_chips[i], true);

      struct agx_draw ind_u32 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), UINT32_C(0xfffffffc),
         AGX_INDEX_SIZE_U32, false);
      assert(!hk_needs_index_robustness(&cmd, &ind_u32, NULL));

      struct agx_draw ind_u16 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), UINT32_C(0xfffffffc),
         AGX_INDEX_SIZE_U16, true);
      assert(!hk_needs_index_robustness(&cmd, &ind_u16, NULL));

      struct agx_draw ind_u16_unaligned = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), 6, AGX_INDEX_SIZE_U16,
         false);
      assert(hk_needs_index_robustness(&cmd, &ind_u16_unaligned, NULL));

      /* A 16-bit index buffer may start at 2 mod 4: the dword bound would
       * then cover the two bytes before it. */
      struct agx_draw ind_u16_base2 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000002), 8, AGX_INDEX_SIZE_U16,
         false);
      assert(hk_needs_index_robustness(&cmd, &ind_u16_base2, NULL));

      struct agx_draw ind_u8 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), 16, AGX_INDEX_SIZE_U8,
         false);
      assert(hk_needs_index_robustness(&cmd, &ind_u8, NULL));

      struct agx_draw ind_4gib = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), UINT64_C(1) << 32,
         AGX_INDEX_SIZE_U32, false);
      assert(hk_needs_index_robustness(&cmd, &ind_4gib, NULL));

      struct agx_draw ind_16gib = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), UINT64_C(1) << 34,
         AGX_INDEX_SIZE_U32, false);
      assert(hk_needs_index_robustness(&cmd, &ind_16gib, NULL));
   }

   /* Every other generation (including G18G, whose out-of-range hardware
    * index bounds are unmeasured) keeps the software rewrite for robust
    * indirect draws. */
   static const enum agx_chip rewrite_chips[] = {
      AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15G, AGX_CHIP_G15S,
      AGX_CHIP_G15C, AGX_CHIP_G16G, AGX_CHIP_G18G,
   };
   for (unsigned i = 0; i < sizeof(rewrite_chips) / sizeof(rewrite_chips[0]);
        ++i) {
      assert(!agx_chip_vdm_indirect_index_bounds(rewrite_chips[i]));
      init_cmd(&dev, &cmd, rewrite_chips[i], true);
      struct agx_draw ind_u32 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), UINT32_C(0xfffffffc),
         AGX_INDEX_SIZE_U32, false);
      assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL));
   }

   /* G18G has the HAL300 stream-link continuation for empty rewritten draws,
    * so oversized bindings without robustness route to the rewrite just like
    * M5, including an empty conditional direct record. */
   init_cmd(&dev, &cmd, AGX_CHIP_G18G, false);
   {
      struct agx_draw ind_16gib_p4 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000),
         (UINT64_C(1) << 34) + 4, AGX_INDEX_SIZE_U32, false);
      assert(hk_needs_index_robustness(&cmd, &ind_16gib_p4, NULL));
      struct agx_draw empty_direct = agx_draw_indexed(
         0, 1, 0, 0, 0, UINT64_C(0x100000000), (UINT64_C(1) << 34) + 4,
         AGX_INDEX_SIZE_U32, false);
      assert(hk_needs_index_robustness(&cmd, &ind_16gib_p4, &empty_direct));
   }

   /* All AGX3 path chips with stream-link continuations admit both
    * representable and oversized bindings into the rewrite. */
   {
      struct agx_draw representable = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), 0x1000,
         AGX_INDEX_SIZE_U32, false);
      struct agx_draw oversized = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000),
         (UINT64_C(1) << 34) + 4, AGX_INDEX_SIZE_U32, false);

      static const enum agx_chip admitted[] = {
         AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
         AGX_CHIP_G18G,
      };
      for (unsigned i = 0; i < sizeof(admitted) / sizeof(admitted[0]); ++i) {
         init_cmd(&dev, &cmd, admitted[i], true);
         assert(hk_robust_index_rewrite_admitted(&dev, representable,
                                                 AGX_PRIMITIVE_TRIANGLES));
         assert(hk_robust_index_rewrite_admitted(&dev, oversized,
                                                 AGX_PRIMITIVE_TRIANGLES));
      }
   }

   /* The bound the hardware reads is the size word both record families
    * emit: a 0x1000-byte range is 0x400 dwords, word 0x3ff, and a 2 mod 4
    * base adds the dword it straddles. */
   for (unsigned base_mod = 0; base_mod <= 2; base_mod += 2) {
      const uint64_t ib = UINT64_C(0x100000000) + base_mod;
      const uint32_t expect = base_mod ? 0x400 : 0x3ff;

      struct agx_draw d = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), ib, 0x1000, AGX_INDEX_SIZE_U16, false);
      uint32_t size_word = 0;
      assert(agx_draw_indirect_index_buffer_size_word(d, &size_word));
      assert(size_word == expect);

      /* HAL300 (M5): State Update, header, IB, A, Unk 2, then the size. */
      struct agx_vdm_index_list_indirect_words hal300;
      assert(agx_vdm_index_list_indirect_pack_words(
         AGX_CHIP_G17C, 3, true, false, AGX_VDM_INDEX_LIST_INDEX_SIZE_U16, ib,
         UINT64_C(0x200000000), 0xffff, size_word, &hal300));
      assert(hal300.count == 9 && hal300.words[6] == 7u &&
             hal300.words[7] == expect);

      /* HAL200 (G17P family): selector, State Update, header, IB, A, then
       * the size, with no Unk 2 word. */
      const struct agx_vdm_hal200_indirect_draw_fields fields = {
         .draw_state_context_offset = 4,
         .primitive = 3,
         .indexed = true,
         .index_size_B = 2,
         .restart_index = 0xffff,
         .index_buffer = ib,
         .index_buffer_range_B = 0x1000,
         .indirect_buffer = UINT64_C(0x200000000),
      };
      struct agx_vdm_hal200_indirect_draw hal200;
      assert(agx_vdm_hal200_indirect_draw_pack(AGX_CHIP_G17P, &fields,
                                               &hal200));
      uint32_t hal200_word;
      memcpy(&hal200_word, hal200.bytes + 4 + 4 * 6, 4);
      assert(hal200.length == AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH &&
             hal200_word == expect);
   }

   /* Oversized (> UINT32_MAX * 4) indirect and conditionally lowered direct
    * draws must always lower through the software rewrite even when no
    * robustness features are enabled, including a 1-point U32 draw at
    * firstIndex = 0 with a 16 GiB + 4 binding. */
   for (unsigned r = 0; r < 2; ++r) {
      const bool robust = r != 0;
      for (unsigned c = 0; c < 2; ++c) {
         const enum agx_chip chip = c == 0 ? AGX_CHIP_G17C : AGX_CHIP_G17P;
         init_cmd(&dev, &cmd, chip, robust);

         const uint64_t range_16gib_p4 = (UINT64_C(1) << 34) + 4;
         struct agx_draw cond_direct = agx_draw_indexed(
            1, 1, 0, 0, 0, UINT64_C(0x100000000), range_16gib_p4,
            AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), range_16gib_p4,
            AGX_INDEX_SIZE_U32, false);
         assert(hk_needs_index_robustness(&cmd, &cond_indirect, &cond_direct));
         assert(hk_needs_index_robustness(&cmd, &cond_indirect, NULL));

         const uint64_t range_16gib = UINT64_C(1) << 34;
         struct agx_draw cond_direct_16g = agx_draw_indexed(
            3, 1, 0, 0, 0, UINT64_C(0x100000000), range_16gib,
            AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect_16g = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), range_16gib,
            AGX_INDEX_SIZE_U32, false);
         assert(
            hk_needs_index_robustness(&cmd, &cond_indirect_16g, &cond_direct_16g));

         /* In-bounds encodable binding (4 GiB) whose accessed span stays within
          * 32-bit byte addressing skips rewrite. */
         struct agx_draw cond_direct_4g = agx_draw_indexed(
            1, 1, 0, 0, 0, UINT64_C(0x100000000), UINT64_C(1) << 32,
            AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect_4g = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), UINT64_C(1) << 32,
            AGX_INDEX_SIZE_U32, false);
         assert(
            !hk_needs_index_robustness(&cmd, &cond_indirect_4g, &cond_direct_4g));

         /* In-bounds 4 GiB + 4 binding with firstIndex at 4 GiB exceeds 32-bit
          * byte addressing from the unrelocated base and must take the 64-bit
          * software rewrite rather than the native indirect packet. */
         const uint64_t range_4gib_p4 = (UINT64_C(1) << 32) + 4;
         struct agx_draw cond_direct_4g_hi = agx_draw_indexed(
            1, 1, UINT32_C(0x40000000), 0, 0, UINT64_C(0x100000000),
            range_4gib_p4, AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect_4g_hi = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), range_4gib_p4,
            AGX_INDEX_SIZE_U32, false);
         assert(hk_needs_index_robustness(&cmd, &cond_indirect_4g_hi,
                                          &cond_direct_4g_hi));
      }
   }

   /* With VK_EXT_pipeline_robustness enabled on the device, index robustness
    * follows the bound vertex shader's per-pipeline vertex_inputs level
    * rather than pessimistically forcing robustness on every draw. */
   {
      struct hk_api_shader api_vs;
      struct agx_draw ind_u32 = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000), 0x1000,
         AGX_INDEX_SIZE_U32, false);
      struct agx_draw direct_oob = agx_draw_indexed(
         6, 1, 0, 0, 0, UINT64_C(0x100000000), 8, AGX_INDEX_SIZE_U32, false);
      struct agx_draw oversized = agx_draw_indexed_indirect(
         UINT64_C(0x200000000), UINT64_C(0x100000000),
         (UINT64_C(1) << 34) + 4, AGX_INDEX_SIZE_U32, false);

      init_cmd(&dev, &cmd, AGX_CHIP_G17P, false);
      dev.vk.enabled_features.pipelineRobustness = true;

      /* Where the VDM bounds indirect index fetches in hardware, an aligned
       * indirect draw never needs the software rewrite, robust or not; the
       * direct draw below still follows the robustness level. */
      const bool hw_index_bounds =
         agx_chip_vdm_indirect_index_bounds(AGX_CHIP_G17P);

      memset(&api_vs, 0, sizeof(api_vs));
      api_vs.vk.stage = MESA_SHADER_VERTEX;
      api_vs.variants[HK_VS_VARIANT_HW].info.stage = MESA_SHADER_VERTEX;
      api_vs.variants[HK_VS_VARIANT_HW].info.vs.has_vertex_robustness = true;
      api_vs.variants[HK_VS_VARIANT_HW].info.vs.vertex_robustness =
         AGX_ROBUSTNESS_DISABLED;
      cmd.state.gfx.shaders[MESA_SHADER_VERTEX] = &api_vs;

      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT) ==
             AGX_ROBUSTNESS_DISABLED);
      assert(!hk_needs_index_robustness(&cmd, &ind_u32, NULL));
      assert(!hk_needs_index_robustness(&cmd, &direct_oob, NULL));
      /* Oversized (> 16 GiB) bindings still require the software rewrite. */
      assert(hk_needs_index_robustness(&cmd, &oversized, NULL));

      /* Fast-linked GPL vertex input library robustness overrides the
       * pre-rasterization library's vertex shader default. */
      cmd.vk.dynamic_graphics_state.vertex_input_robustness =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT;
      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                cmd.vk.dynamic_graphics_state.vertex_input_robustness) ==
             AGX_ROBUSTNESS_D3D);
      assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL) ==
             !hw_index_bounds);
      assert(hk_needs_index_robustness(&cmd, &direct_oob, NULL));
      cmd.vk.dynamic_graphics_state.vertex_input_robustness =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT;

      /* Per-pipeline ROBUST_BUFFER_ACCESS / ROBUST_BUFFER_ACCESS_2 re-enables
       * index robustness even when device-wide robustness features are off. */
      api_vs.variants[HK_VS_VARIANT_HW].info.vs.vertex_robustness =
         AGX_ROBUSTNESS_GL;
      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT) ==
             AGX_ROBUSTNESS_GL);
      assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL) ==
             !hw_index_bounds);
      assert(hk_needs_index_robustness(&cmd, &direct_oob, NULL));

      api_vs.variants[HK_VS_VARIANT_HW].info.vs.vertex_robustness =
         AGX_ROBUSTNESS_D3D;
      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT) ==
             AGX_ROBUSTNESS_D3D);
      assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL) ==
             !hw_index_bounds);
      assert(hk_needs_index_robustness(&cmd, &direct_oob, NULL));

      /* Shader objects (has_vertex_robustness == false) derive robustness from
       * device-level robustBufferAccess/robustBufferAccess2, ignoring both
       * pipelineRobustness and any previously bound pipeline's VI robustness. */
      api_vs.variants[HK_VS_VARIANT_HW].info.vs.has_vertex_robustness = false;
      api_vs.variants[HK_VS_VARIANT_HW].info.vs.vertex_robustness =
         AGX_ROBUSTNESS_DISABLED;
      cmd.vk.dynamic_graphics_state.vertex_input_robustness =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT;
      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                cmd.vk.dynamic_graphics_state.vertex_input_robustness) ==
             AGX_ROBUSTNESS_DISABLED);
      assert(!hk_needs_index_robustness(&cmd, &ind_u32, NULL));
      assert(!hk_needs_index_robustness(&cmd, &direct_oob, NULL));

      dev.vk.enabled_features.robustBufferAccess = true;
      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                cmd.vk.dynamic_graphics_state.vertex_input_robustness) ==
             AGX_ROBUSTNESS_GL);
      assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL) ==
             !hw_index_bounds);
      assert(hk_needs_index_robustness(&cmd, &direct_oob, NULL));

      dev.vk.enabled_features.robustBufferAccess2 = true;
      assert(hk_vertex_robustness_level(
                &dev, &api_vs.variants[HK_VS_VARIANT_HW],
                cmd.vk.dynamic_graphics_state.vertex_input_robustness) ==
             AGX_ROBUSTNESS_D3D);
      assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL) ==
             !hw_index_bounds);
      assert(hk_needs_index_robustness(&cmd, &direct_oob, NULL));
   }

   puts("hk_index_robustness: ok");
   return 0;
}
