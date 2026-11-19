/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Copyright (C) 2026 Eryk Wieliczko */
#include <gtest/gtest.h>

#include "agx_chip.h"
#include "agx3_compute_program_core.h"
#include "asahi/genxml/agx3_cmdstream.h"

static drm_asahi_params_global
g16g_params()
{
   drm_asahi_params_global params = {};
   params.gpu_generation = 16;
   params.gpu_variant = 'G';
   params.chip_id = 0x8132;
   params.num_dies = 1;
   params.num_clusters_total = 1;
   params.usc_generation = AGX_ISA_AGX3;
   params.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   return params;
}

TEST(AGXChip, G16GRequiresExactKernelContract)
{
   const drm_asahi_params_global valid = g16g_params();
   ASSERT_EQ(agx_chip_from_params(&valid), AGX_CHIP_G16G);
   ASSERT_TRUE(agx_params_driver_supported(&valid));
   EXPECT_FALSE(agx_params_driver_supported(nullptr));

   for (unsigned field = 0; field < 7; ++field) {
      drm_asahi_params_global params = valid;
      switch (field) {
      case 0: params.gpu_generation = 17; break;
      case 1: params.gpu_variant = 'S'; break;
      case 2: params.chip_id = 0x6040; break;
      case 3: params.num_dies = 2; break;
      case 4: params.num_clusters_total = 2; break;
      case 5: params.usc_generation = AGX_ISA_AGX2; break;
      case 6: params.gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY; break;
      }
      EXPECT_FALSE(agx_params_driver_supported(&params)) << field;
   }

   for (auto chip : {AGX_CHIP_G16P, AGX_CHIP_G16S, AGX_CHIP_G16C}) {
      EXPECT_TRUE(agx_chip_codegen_supported(chip));
      EXPECT_FALSE(agx_chip_driver_supported(chip));
   }
   EXPECT_TRUE(agx_chip_driver_supported(AGX_CHIP_G15S));
   EXPECT_TRUE(agx_chip_driver_supported(AGX_CHIP_G17P));
}

TEST(AGXChip, G16GUsesSharedNativePathWithItsOwnCapabilities)
{
   EXPECT_TRUE(agx_chip_native_agx3(AGX_CHIP_G16G));
   EXPECT_EQ(agx_chip_isa_encoding(AGX_CHIP_G16G), AGX_ISA_ENCODING_AGX3B);
   EXPECT_EQ(agx_chip_record_family(AGX_CHIP_G16G), AGX_CHIP_G17P);
   EXPECT_EQ(agx_chip_command_stream_generation(AGX_CHIP_G16G),
             AGX_CS_GENERATION_HAL200);
   EXPECT_EQ(agx_chip_descriptor_generation(AGX_CHIP_G16G),
             AGX_DESCRIPTOR_GENERATION_9);
   EXPECT_EQ(agx_chip_sampler_generation(AGX_CHIP_G16G),
             AGX_SAMPLER_GENERATION_GEN4_HAL200);
   EXPECT_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G16G), 32u);
   EXPECT_EQ(agx_chip_usc_helper_table_slots(AGX_CHIP_G16G, 0), 10u);
   EXPECT_TRUE(agx_chip_native_indirect_compute(AGX_CHIP_G16G));
   EXPECT_TRUE(agx_chip_native_imageblock(AGX_CHIP_G16G));
   EXPECT_TRUE(agx_chip_explicit_robust_image_bounds(AGX_CHIP_G16G));
   EXPECT_FALSE(agx_chip_native_g15(AGX_CHIP_G16G));
   EXPECT_FALSE(agx_chip_has_native_render_context(AGX_CHIP_G16G));
   EXPECT_FALSE(agx_chip_native_preamble(AGX_CHIP_G16G));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_defect(AGX_CHIP_G16G));
   EXPECT_FALSE(agx_chip_tri_merge_across_side_effects(AGX_CHIP_G16G));
   EXPECT_FALSE(agx_chip_exact_fma32(AGX_CHIP_G16G));
   EXPECT_FALSE(agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_G16G));
}

TEST(AGXChip, TextureDescriptorSpareWordIsM5Gen10)
{
   for (enum agx_chip chip : {AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C})
      EXPECT_TRUE(agx_chip_texture_descriptor_spare_word(chip));
   /* AGX2 has 24-byte descriptors; Gen9 and G18G are not shown to ignore
    * the word. */
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15S,
                              AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G18G,
                              AGX_CHIP_UNKNOWN})
      EXPECT_FALSE(agx_chip_texture_descriptor_spare_word(chip));
}

TEST(AGXChip, UncoveredTileReloadDefect)
{
   /* M4 Pro keeps a 2x reload in the tilebuffer; 4x and A18 Pro spill. */
   EXPECT_FALSE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G17P, 0x6040, 2));
   EXPECT_FALSE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G17P, 0x6041, 2));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G17P, 0x6041, 4));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G17P, 0x6040, 4));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G17P, 0x8140, 2));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G17P, 0x8140, 4));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G16G, 0x8132, 2));
   EXPECT_TRUE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G15S, 0x6030, 2));
   EXPECT_FALSE(agx_chip_uncovered_tile_ms_reload_defect(AGX_CHIP_G13G, 0x8103, 2));
}

TEST(AGXChip, G16GSharedComputeLaunchBytes)
{
   agx3_g17p_native_compute_program_desc desc = {};
   desc.chip = AGX_CHIP_G16G;
   desc.program_va = UINT64_C(0x10000020000);
   desc.shader_va = UINT64_C(0x10000040000);
   agx3_compute_program program;
   ASSERT_TRUE(agx3_g17p_native_compute_program_bind(&desc, &program));

   uint8_t bytes[AGX3_NATIVE_INDIRECT_COMPUTE_DISPATCH_MAX_SIZE] = {};
   agx3_compute_dispatch_sink sink = {bytes, sizeof(bytes), 0};
   agx3_compute_direct_grid grid = {};
   grid.source = AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS;
   grid.size[0] = 3;
   grid.size[1] = 2;
   grid.size[2] = 1;
   grid.local_size[0] = 4;
   grid.local_size[1] = 2;
   grid.local_size[2] = 1;
   grid.dimensions = 2;
   ASSERT_TRUE(agx3_g17p_native_compute_dispatch_append(&program, &grid, &sink));
   const uint32_t direct[] = {0x00080000, 0x01000000, 0x00000800,
                              0x40000001, 12, 4, 1, 4, 2, 1, 0x60000160};
   ASSERT_EQ(sink.length, sizeof(direct));
   for (unsigned i = 0; i < ARRAY_SIZE(direct); ++i)
      EXPECT_EQ(agx3_cmdstream_load_le32(bytes + i * 4), direct[i]) << i;

   sink.length = 0;
   ASSERT_TRUE(agx3_g17p_native_indirect_compute_launch_append(
      &program, UINT64_C(0x4000008000), 5, &sink));
   const uint32_t indirect[] = {0x10080000, 0x01000000, 0x00000800,
                                0x40000001, 0x00000040, 0x00008000,
                                0x60001560};
   ASSERT_EQ(sink.length, sizeof(indirect));
   for (unsigned i = 0; i < ARRAY_SIZE(indirect); ++i)
      EXPECT_EQ(agx3_cmdstream_load_le32(bytes + i * 4), indirect[i]) << i;

   sink.length = 0;
   agx3_g17p_native_indirect_dispatch dispatch = {
      &program, UINT64_C(0x4000008000)};
   ASSERT_TRUE(agx3_g17p_native_indirect_compute_dispatch_append(
      &program, &dispatch, &sink));
   EXPECT_EQ(sink.length, 72u);
   EXPECT_EQ(agx3_native_indirect_compute_dispatch_size(AGX_CHIP_G16G), 72u);
   for (unsigned i = 0; i < 6; ++i)
      EXPECT_EQ(agx3_cmdstream_load_le32(bytes + 44 + i * 4), indirect[i]) << i;

   program.chip = AGX_CHIP_G16S;
   sink.length = 0;
   EXPECT_FALSE(agx3_g17p_native_indirect_compute_launch_append(
      &program, UINT64_C(0x4000008000), 5, &sink));
   EXPECT_EQ(sink.length, 0u);
}
