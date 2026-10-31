/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/libagx/libagx_dgc.h"
#include "asahi/lib/agx_m5_va.h"
#include "agx_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

#define CHECK_EQ(actual, expected)                                             \
   do {                                                                        \
      uint64_t a_ = (actual);                                                   \
      uint64_t e_ = (expected);                                                 \
      ++checks;                                                                 \
      if (a_ != e_) {                                                           \
         fprintf(stderr, "%s:%d: got 0x%llx, expected 0x%llx\n", __FILE__,   \
                 __LINE__, (unsigned long long)a_, (unsigned long long)e_);     \
         exit(1);                                                               \
      }                                                                         \
   } while (0)

static uint64_t
word64(const uint32_t *packed, unsigned index)
{
   uint64_t value;
   memcpy(&value, &packed[index * 2], sizeof(value));
   return value;
}

static void
test_m5_live_descriptors(void)
{
   /* Live AGXG17XFamilyTexture readout for a 256x128 RGBA8 texture. */
   static const uint8_t expected_texture[32] = {
      0x22, 0x0a, 0x88, 0xf6, 0x0f, 0xf8, 0x03, 0x20,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0x00, 0xa0, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_TEXTURE_GEN10 texture = {AGX_TEXTURE_GEN10_header};
   uint32_t texture_packed[AGX_TEXTURE_GEN10_LENGTH / 4] = {0};

   texture.dimension = AGX_TEXTURE_DIMENSION_2D;
   texture.layout = AGX_LAYOUT_GPU;
   texture.channels = AGX_CHANNELS_R8G8B8A8;
   texture.type = AGX_TEXTURE_TYPE_UNORM;
   texture.swizzle_r = AGX_CHANNEL_R;
   texture.swizzle_g = AGX_CHANNEL_G;
   texture.swizzle_b = AGX_CHANNEL_B;
   texture.swizzle_a = AGX_CHANNEL_A;
   texture.width = 256;
   texture.height = 128;
   texture.compressed = true;
   texture.address = 0x10000080000ull;
   texture.extended = true;
   texture.acceleration_buffer = 0x100000a0000ull;

   AGX_TEXTURE_GEN10_pack(texture_packed, &texture);
   ++checks;
   if (memcmp(texture_packed, expected_texture, sizeof(expected_texture))) {
      fprintf(stderr, "M5 live texture descriptor mismatch\n");
      exit(1);
   }

   static const uint8_t expected_pbe[32] = {
      0x22, 0x0a, 0xe4, 0xff, 0x80, 0x3f, 0x00, 0x20,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0x00, 0xa0, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_PBE_GEN10 pbe = {AGX_PBE_GEN10_header};
   uint32_t pbe_packed[AGX_PBE_GEN10_LENGTH / 4] = {0};

   pbe.dimension = AGX_TEXTURE_DIMENSION_2D;
   pbe.layout = AGX_LAYOUT_GPU;
   pbe.channels = AGX_CHANNELS_R8G8B8A8;
   pbe.type = AGX_TEXTURE_TYPE_UNORM;
   pbe.swizzle_r = AGX_CHANNEL_R;
   pbe.swizzle_g = AGX_CHANNEL_G;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;
   pbe.width = 256;
   pbe.height = 128;
   pbe.compressed = true;
   pbe.buffer = 0x10000080000ull;
   pbe.extended = true;
   pbe.acceleration_buffer = 0x100000a0000ull;

   AGX_PBE_GEN10_pack(pbe_packed, &pbe);
   ++checks;
   if (memcmp(pbe_packed, expected_pbe, sizeof(expected_pbe))) {
      fprintf(stderr, "M5 live PBE descriptor mismatch\n");
      exit(1);
   }
}

static void
test_generation_boundaries(void)
{
   struct AGX_PBE_GEN9 gen9 = {AGX_PBE_GEN9_header};
   struct AGX_PBE_GEN10 gen10 = {AGX_PBE_GEN10_header};
   uint32_t packed9[AGX_PBE_GEN9_LENGTH / 4] = {0};
   uint32_t packed10[AGX_PBE_GEN10_LENGTH / 4] = {0};

   gen9.width = gen10.width = 0x4000;
   gen9.height = gen10.height = 0x2345;
   gen9.samples = gen10.samples = AGX_SAMPLE_COUNT_8;
   gen9.buffer = gen10.buffer = 0xabcde123450ull;
   gen9.level = gen10.level = 3;
   gen9.levels = gen10.levels = 7;
   gen9.extended = gen10.extended = true;

   AGX_PBE_GEN9_pack(packed9, &gen9);
   AGX_PBE_GEN10_pack(packed10, &gen10);

   CHECK_EQ((word64(packed9, 0) >> 56) & 3, AGX_SAMPLE_COUNT_8);
   CHECK_EQ((word64(packed10, 0) >> 58) & 3, AGX_SAMPLE_COUNT_8);
   CHECK_EQ(word64(packed9, 1) & ((1ull << 44) - 1), gen9.buffer >> 4);
   CHECK_EQ(word64(packed10, 1) & ((1ull << 44) - 1), gen10.buffer >> 4);
   CHECK_EQ((word64(packed9, 2) >> 44) & 0xff, 0x63);
   CHECK_EQ((word64(packed10, 2) >> 44) & 0xff, 0x63);
   CHECK_EQ(word64(packed9, 3), 0);
   CHECK_EQ(word64(packed10, 3), 0);

   /* Width 0x4000 fits Gen9 exactly; Gen10 retains a fifteenth bit. */
   CHECK_EQ((word64(packed9, 0) >> 24) & 0x3fff, 0x3fff);
   CHECK_EQ((word64(packed10, 0) >> 24) & 0x7fff, 0x3fff);
}

static void
test_sampler_generations(void)
{
   struct AGX_SAMPLER_HAL200 hal200 = {AGX_SAMPLER_HAL200_header};
   struct AGX_SAMPLER_GEN5 gen5 = {AGX_SAMPLER_GEN5_header};
   uint32_t packed_hal200[AGX_SAMPLER_HAL200_LENGTH / 4] = {0};
   uint32_t packed_gen5[AGX_SAMPLER_GEN5_LENGTH / 4] = {0};

   hal200.maximum_lod = 100.0f;
   hal200.lod_bias = -1.015625f;
   hal200.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
   hal200.seamful_cube_maps = true;
   AGX_SAMPLER_HAL200_pack(packed_hal200, &hal200);

   gen5.maximum_lod = 100.0f;
   gen5.lod_bias = -1.015625f;
   gen5.reduction_mode = AGX_SAMPLER_REDUCTION_MODE_MINIMUM;
   gen5.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
   gen5.seamful_cube_maps = true;
   AGX_SAMPLER_GEN5_pack(packed_gen5, &gen5);

   CHECK_EQ((word64(packed_hal200, 0) >> 10) & 0x3ff, 0x380);
   CHECK_EQ((word64(packed_gen5, 0) >> 10) & 0x3ff, 0x3c0);
   CHECK_EQ((word64(packed_hal200, 0) >> 43) & 0x7ff, 0x7bf);
   CHECK_EQ((word64(packed_gen5, 0) >> 43) & 0x7ff, 0x7bf);
   CHECK_EQ((word64(packed_gen5, 0) >> 39) & 3,
            AGX_SAMPLER_REDUCTION_MODE_MINIMUM);
   CHECK_EQ((word64(packed_hal200, 0) >> 61) & 3,
            AGX_BORDER_COLOUR_OPAQUE_WHITE);
   CHECK_EQ((word64(packed_gen5, 0) >> 61) & 3,
            AGX_BORDER_COLOUR_OPAQUE_WHITE);
   CHECK_EQ(word64(packed_hal200, 0) >> 63, 1);
   CHECK_EQ(word64(packed_gen5, 0) >> 63, 1);

   memset(packed_gen5, 0, sizeof(packed_gen5));
   gen5.reduction_mode = AGX_SAMPLER_REDUCTION_MODE_MAXIMUM;
   gen5.lod_bias = -100.0f;
   AGX_SAMPLER_GEN5_pack(packed_gen5, &gen5);
   CHECK_EQ((word64(packed_gen5, 0) >> 39) & 3,
            AGX_SAMPLER_REDUCTION_MODE_MAXIMUM);
   CHECK_EQ((word64(packed_gen5, 0) >> 43) & 0x7ff, 0x400);

   memset(packed_gen5, 0, sizeof(packed_gen5));
   gen5.lod_bias = 100.0f;
   AGX_SAMPLER_GEN5_pack(packed_gen5, &gen5);
   CHECK_EQ((word64(packed_gen5, 0) >> 43) & 0x7ff, 0x3ff);
}

static void
test_agx3_cdm_launch(void)
{
   const enum agx_chip grounded[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   const uint8_t expected[AGX3_CDM_LAUNCH_SIZE] = {
      0x67, 0x45, 0x23, 0x01, 0xde, 0xbc, 0x0a, 0x01,
      0x23, 0x01, 0xef, 0x8d, 0xef, 0xbe, 0xad, 0xde,
      0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
      0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
   };

   struct agx3_cdm_launch_info input = {
      .word0_payload = 0x01234567,
      .word1_payload = 0x000abcde,
      .pipeline = UINT64_C(0x1237bc048c0),
      .opaque_word3 = 0xdeadbeef,
      .global_size = {3, 5, 7},
      .local_size = {8, 16, 32},
   };

   for (unsigned i = 0; i < sizeof(grounded) / sizeof(grounded[0]); ++i) {
      uint8_t packed[AGX3_CDM_LAUNCH_SIZE];
      input.chip = grounded[i];
      CHECK_EQ(agx3_cdm_launch_target_grounded(input.chip), true);
      CHECK_EQ(agx3_cdm_launch_pack(&input, packed), true);
      ++checks;
      if (memcmp(packed, expected, sizeof(expected))) {
         fprintf(stderr, "AGX3 CDM launch mismatch for chip %u\n", input.chip);
         for (unsigned b = 0; b < sizeof(packed); ++b)
            fprintf(stderr, "%02x%s", packed[b],
                    b + 1 == sizeof(packed) ? "\n" : " ");
         exit(1);
      }

      struct agx3_cdm_launch_info decoded;
      CHECK_EQ(agx3_cdm_launch_unpack(input.chip, packed, sizeof(packed),
                                      &decoded),
               true);
      CHECK_EQ(decoded.chip, input.chip);
      CHECK_EQ(decoded.word0_payload, input.word0_payload);
      CHECK_EQ(decoded.word1_payload, input.word1_payload);
      CHECK_EQ(decoded.pipeline, input.pipeline);
      CHECK_EQ(decoded.opaque_word3, input.opaque_word3);
      for (unsigned axis = 0; axis < 3; ++axis) {
         CHECK_EQ(decoded.global_size[axis], input.global_size[axis]);
         CHECK_EQ(decoded.local_size[axis], input.local_size[axis]);
      }
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      uint8_t packed[AGX3_CDM_LAUNCH_SIZE];
      input.chip = rejected[i];
      CHECK_EQ(agx3_cdm_launch_target_grounded(input.chip), false);
      CHECK_EQ(agx3_cdm_launch_pack(&input, packed), false);
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK_EQ(packed[b], 0);
   }

   uint8_t packed[AGX3_CDM_LAUNCH_SIZE];
   input.chip = AGX_CHIP_G17S;
   input.pipeline |= 1;
   CHECK_EQ(agx3_cdm_launch_pack(&input, packed), false);
   input.pipeline &= ~UINT64_C(0x3f);
   input.pipeline = UINT64_C(1) << 48;
   CHECK_EQ(agx3_cdm_launch_pack(&input, packed), false);
   input.pipeline = UINT64_C(0x1237bc048c0);
   input.word0_payload = UINT32_C(1) << 29;
   CHECK_EQ(agx3_cdm_launch_pack(&input, packed), false);
   input.word0_payload = 0x01234567;
   input.word1_payload = UINT32_C(1) << 22;
   CHECK_EQ(agx3_cdm_launch_pack(&input, packed), false);
   input.word1_payload = 0x000abcde;
   input.local_size[2] = 0x800;
   CHECK_EQ(agx3_cdm_launch_pack(&input, packed), false);
   input.local_size[2] = 32;
   CHECK_EQ(agx3_cdm_launch_pack(NULL, packed), false);
   CHECK_EQ(agx3_cdm_launch_pack(&input, NULL), false);

   CHECK_EQ(agx3_cdm_launch_pack(&input, packed), true);
   struct agx3_cdm_launch_info decoded;
   CHECK_EQ(agx3_cdm_launch_unpack(input.chip, packed, 39, &decoded), false);
   packed[3] |= 0x20;
   CHECK_EQ(agx3_cdm_launch_unpack(input.chip, packed, sizeof(packed),
                                   &decoded),
            false);
   CHECK_EQ(decoded.pipeline, 0);
}

static void
test_agx3_live_vdm_draw(void)
{
   CHECK_EQ(agx_primitive_restart_index(AGX_INDEX_SIZE_U8), UINT8_MAX);
   CHECK_EQ(agx_primitive_restart_index(AGX_INDEX_SIZE_U16), UINT16_MAX);
   CHECK_EQ(agx_primitive_restart_index(AGX_INDEX_SIZE_U32), UINT32_MAX);

   static const uint8_t nonindexed_g15[] = {
      0x00, 0x06, 0xc4, 0x61, 0x44, 0x33, 0x22, 0x11,
      0x02, 0x00, 0x00, 0x00, 0xdd, 0xcc, 0xbb, 0xaa,
   };
   static const uint8_t nonindexed_hal300[] = {
      0x00, 0x06, 0xc4, 0x69, 0x44, 0x33, 0x22, 0x11, 0x02, 0x00,
      0x00, 0x00, 0xdd, 0xcc, 0xbb, 0xaa, 0x07, 0x00, 0x00, 0x00,
   };
   static const uint8_t indexed_g15[] = {
      0x34, 0x09, 0xf5, 0x61, 0xbc, 0x9a, 0x78, 0x56, 0x04, 0x03,
      0x02, 0x01, 0x03, 0x00, 0x00, 0x00, 0xf9, 0xff, 0xff, 0xff,
      0x40, 0x30, 0x20, 0x10, 0x12, 0x00, 0x00, 0x00,
   };
   static const uint8_t indexed_hal300[] = {
      0x34, 0x09, 0xf5, 0x69, 0xbc, 0x9a, 0x78, 0x56, 0x04, 0x03, 0x02,
      0x01, 0x03, 0x00, 0x00, 0x00, 0xf9, 0xff, 0xff, 0xff, 0x07, 0x00,
      0x00, 0x00, 0x40, 0x30, 0x20, 0x10, 0x12, 0x00, 0x00, 0x00,
   };
   static const uint8_t indirect_nonindexed_g15[] = {
      0x00, 0x06, 0x04, 0x64, 0x45, 0x23,
      0x00, 0x00, 0xc0, 0xab, 0x89, 0x67,
   };
   static const uint8_t indirect_nonindexed_hal300[] = {
      0x00, 0x06, 0x04, 0x6c, 0x45, 0x23, 0x00, 0x00,
      0xc0, 0xab, 0x89, 0x67, 0x07, 0x00, 0x00, 0x00,
   };
   static const uint8_t indirect_indexed_g15[] = {
      0x01, 0x00, 0x00, 0x40, 0xff, 0xff, 0xff, 0xff,
      0x34, 0x09, 0x35, 0x64, 0xbc, 0x9a, 0x78, 0x56,
      0x45, 0x23, 0x00, 0x00, 0xc0, 0xab, 0x89, 0x67,
      0x3f, 0x30, 0x20, 0x10, 0x12, 0x00, 0x00, 0x00,
   };
   static const uint8_t indirect_indexed_hal300[] = {
      0x01, 0x00, 0x00, 0x40, 0xff, 0xff, 0xff, 0xff,
      0x34, 0x09, 0x35, 0x6c, 0xbc, 0x9a, 0x78, 0x56,
      0x45, 0x23, 0x00, 0x00, 0xc0, 0xab, 0x89, 0x67,
      0x07, 0x00, 0x00, 0x00, 0x3f, 0x30, 0x20, 0x10,
      0x12, 0x00, 0x00, 0x00,
   };
   const enum agx_chip chips[] = {AGX_CHIP_G15G, AGX_CHIP_G17S};

   for (unsigned i = 0; i < 2; ++i) {
      const bool hal300 = chips[i] == AGX_CHIP_G17S;
      uint32_t packed[9] = {0};
      const struct agx_draw nonindexed = {
         .b = agx_3d(UINT32_C(0x11223344), 2, 1),
         .start = UINT32_C(0xaabbccdd),
      };
      CHECK_EQ(
         agx_vdm_draw_supported(chips[i], nonindexed, (enum agx_primitive)6),
         true);
      CHECK_EQ(agx_vdm_draw_size(chips[i], nonindexed), hal300 ? 20 : 16);
      uint32_t *end =
         agx_vdm_draw(packed, chips[i], nonindexed, (enum agx_primitive)6, 0);
      CHECK_EQ((uintptr_t)end - (uintptr_t)packed, hal300 ? 20 : 16);
      ++checks;
      if (memcmp(packed, hal300 ? nonindexed_hal300 : nonindexed_g15,
                 hal300 ? sizeof(nonindexed_hal300) : sizeof(nonindexed_g15))) {
         fprintf(stderr, "AGX3 live nonindexed VDM mismatch\n");
         exit(1);
      }

      memset(packed, 0, sizeof(packed));
      const struct agx_draw indexed =
         agx_draw_indexed(UINT32_C(0x01020304), 3, 0, UINT32_C(0xfffffff9), 0,
                          UINT64_C(0x123456789abc), UINT32_C(0x4080c100),
                          AGX_INDEX_SIZE_U32, true);
      CHECK_EQ(agx_vdm_draw_supported(chips[i], indexed, (enum agx_primitive)9),
               true);
      CHECK_EQ(agx_vdm_draw_size(chips[i], indexed), hal300 ? 32 : 28);
      end = agx_vdm_draw(packed, chips[i], indexed, (enum agx_primitive)9,
                         UINT32_MAX);
      CHECK_EQ((uintptr_t)end - (uintptr_t)packed, hal300 ? 32 : 28);
      ++checks;
      if (memcmp(packed, hal300 ? indexed_hal300 : indexed_g15,
                 hal300 ? sizeof(indexed_hal300) : sizeof(indexed_g15))) {
         fprintf(stderr, "AGX3 live indexed VDM mismatch\n");
         exit(1);
      }

      memset(packed, 0, sizeof(packed));
      const struct agx_draw indirect_nonindexed =
         agx_draw_indirect(UINT64_C(0x23456789abc0));
      CHECK_EQ(agx_vdm_draw_supported(chips[i], indirect_nonindexed,
                                      (enum agx_primitive)6),
               true);
      CHECK_EQ(agx_vdm_draw_size(chips[i], indirect_nonindexed),
               hal300 ? 16 : 12);
      end = agx_vdm_draw(packed, chips[i], indirect_nonindexed,
                         (enum agx_primitive)6, 0);
      CHECK_EQ((uintptr_t)end - (uintptr_t)packed, hal300 ? 16 : 12);
      ++checks;
      if (memcmp(packed,
                 hal300 ? indirect_nonindexed_hal300
                        : indirect_nonindexed_g15,
                 hal300 ? sizeof(indirect_nonindexed_hal300)
                        : sizeof(indirect_nonindexed_g15))) {
         fprintf(stderr, "AGX3 live indirect nonindexed VDM mismatch\n");
         exit(1);
      }

      memset(packed, 0, sizeof(packed));
      const struct agx_draw indirect_indexed = agx_draw_indexed_indirect(
         UINT64_C(0x23456789abc0), UINT64_C(0x123456789abc),
         UINT32_C(0x4080c100), AGX_INDEX_SIZE_U32, true);
      CHECK_EQ(agx_vdm_draw_supported(chips[i], indirect_indexed,
                                      (enum agx_primitive)9),
               true);
      CHECK_EQ(agx_vdm_draw_size(chips[i], indirect_indexed),
               hal300 ? 36 : 32);
      end = agx_vdm_draw(packed, chips[i], indirect_indexed,
                         (enum agx_primitive)9, UINT32_MAX);
      CHECK_EQ((uintptr_t)end - (uintptr_t)packed, hal300 ? 36 : 32);
      ++checks;
      if (memcmp(packed,
                 hal300 ? indirect_indexed_hal300 : indirect_indexed_g15,
                 hal300 ? sizeof(indirect_indexed_hal300)
                        : sizeof(indirect_indexed_g15))) {
         fprintf(stderr, "AGX3 live indirect indexed VDM mismatch\n");
         exit(1);
      }

      /* VK_EXT_primitive_restart_index: a custom index, zero included,
       * replaces the type's all-ones value in the record's restart State
       * Update and nothing else. */
      uint32_t custom[9] = {0};
      agx_vdm_draw(custom, chips[i], indirect_indexed, (enum agx_primitive)9,
                   0);
      CHECK_EQ(custom[0], UINT32_C(0x40000001));
      CHECK_EQ(custom[1], 0);
      custom[1] = UINT32_MAX;
      ++checks;
      if (memcmp(custom, packed, hal300 ? 36 : 32)) {
         fprintf(stderr, "AGX3 custom restart index changed other words\n");
         exit(1);
      }
   }

   /* G17P's GPU-generated robust draw must match the complete native HAL200
    * indexed body.  A fresh render-state publication carries the preceding
    * context-relative selector; an unchanged state begins at the mandatory
    * restart State Update instead. */
   static const uint8_t robust_g17p[] = {
      0x01, 0x00, 0x00, 0x40, 0xff, 0xff, 0xff, 0xff,
      0x34, 0x09, 0xf5, 0x61, 0xbc, 0x9a, 0x78, 0x56,
      0x04, 0x03, 0x02, 0x01, 0x03, 0x00, 0x00, 0x00,
      0xf9, 0xff, 0xff, 0xff, 0x40, 0x30, 0x20, 0x10,
      0x12, 0x00, 0x00, 0x00,
   };
   static const uint8_t robust_g17p_with_state[] = {
      0x00, 0x80, 0x04, 0x00,
      0x01, 0x00, 0x00, 0x40, 0xff, 0xff, 0xff, 0xff,
      0x34, 0x09, 0xf5, 0x61, 0xbc, 0x9a, 0x78, 0x56,
      0x04, 0x03, 0x02, 0x01, 0x03, 0x00, 0x00, 0x00,
      0xf9, 0xff, 0xff, 0xff, 0x40, 0x30, 0x20, 0x10,
      0x12, 0x00, 0x00, 0x00,
   };
   const struct agx_draw robust_draw =
      agx_draw_indexed(UINT32_C(0x01020304), 3, 0, UINT32_C(0xfffffff9), 0,
                       UINT64_C(0x123456789abc), UINT32_C(0x4080c100),
                       AGX_INDEX_SIZE_U32, true);
   uint32_t robust_packed[10] = {0};
   CHECK_EQ(libagx_draw_robust_index_vdm_size(AGX_CHIP_G17P, false),
            sizeof(robust_g17p));
   uint32_t *robust_end = agx_vdm_draw_robust_index(
      robust_packed, AGX_CHIP_G17P, robust_draw, (enum agx_primitive)9,
      UINT32_MAX, 0, false, 0);
   CHECK_EQ((uintptr_t)robust_end - (uintptr_t)robust_packed,
            sizeof(robust_g17p));
   ++checks;
   if (memcmp(robust_packed, robust_g17p, sizeof(robust_g17p))) {
      fprintf(stderr, "G17P robust indexed body mismatch\n");
      exit(1);
   }

   memset(robust_packed, 0, sizeof(robust_packed));
   CHECK_EQ(libagx_draw_robust_index_vdm_size(AGX_CHIP_G17P, true),
            sizeof(robust_g17p_with_state));
   robust_end = agx_vdm_draw_robust_index(
      robust_packed, AGX_CHIP_G17P, robust_draw, (enum agx_primitive)9,
      UINT32_MAX, UINT32_C(0x48000), true, 0);
   CHECK_EQ((uintptr_t)robust_end - (uintptr_t)robust_packed,
            sizeof(robust_g17p_with_state));
   ++checks;
   if (memcmp(robust_packed, robust_g17p_with_state,
              sizeof(robust_g17p_with_state))) {
      fprintf(stderr, "G17P robust indexed state/body mismatch\n");
      exit(1);
   }

   /* The repaired draw keeps the application's custom restart index. */
   memset(robust_packed, 0, sizeof(robust_packed));
   agx_vdm_draw_robust_index(robust_packed, AGX_CHIP_G17P, robust_draw,
                             (enum agx_primitive)9, 1, UINT32_C(0x48000),
                             true, 0);
   CHECK_EQ(robust_packed[2], 1);
   robust_packed[2] = UINT32_MAX;
   ++checks;
   if (memcmp(robust_packed, robust_g17p_with_state,
              sizeof(robust_g17p_with_state))) {
      fprintf(stderr, "G17P robust custom restart index mismatch\n");
      exit(1);
   }

   /* The HAL200 framing is G17P-only; M3's existing GPU-generated packet is
    * byte-for-byte unchanged even if the caller's state flag is set. */
   memset(robust_packed, 0, sizeof(robust_packed));
   CHECK_EQ(libagx_draw_robust_index_vdm_size(AGX_CHIP_G15G, true),
            sizeof(indexed_g15));
   robust_end = agx_vdm_draw_robust_index(
      robust_packed, AGX_CHIP_G15G, robust_draw, (enum agx_primitive)9,
      UINT32_MAX, UINT32_C(0x48000), true, 0);
   CHECK_EQ((uintptr_t)robust_end - (uintptr_t)robust_packed,
            sizeof(indexed_g15));
   ++checks;
   if (memcmp(robust_packed, indexed_g15, sizeof(indexed_g15))) {
      fprintf(stderr, "M3 robust indexed body changed\n");
      exit(1);
   }

   /* The assertion-free device producer retains the exact AGX2 packet too. */
   uint32_t agx2_expected[8] = {0}, agx2_robust[8] = {0};
   struct agx_draw agx2_draw = robust_draw;
   agx2_draw.index_buffer = UINT64_C(0x3456789abc);
   uint32_t *agx2_expected_end =
      agx_vdm_draw(agx2_expected, AGX_CHIP_G13G, agx2_draw,
                   (enum agx_primitive)9, UINT32_MAX);
   uint32_t *agx2_robust_end =
      agx_vdm_draw_robust_index(agx2_robust, AGX_CHIP_G13G, agx2_draw,
                                (enum agx_primitive)9, UINT32_MAX,
                                UINT32_C(0x48000), true, 0);
   const size_t agx2_size =
      (uintptr_t)agx2_expected_end - (uintptr_t)agx2_expected;
   CHECK_EQ((uintptr_t)agx2_robust_end - (uintptr_t)agx2_robust, agx2_size);
   CHECK_EQ(libagx_draw_robust_index_vdm_size(AGX_CHIP_G13G, true),
            agx2_size);
   ++checks;
   if (memcmp(agx2_expected, agx2_robust, agx2_size)) {
      fprintf(stderr, "AGX2 robust indexed body changed\n");
      exit(1);
   }

   /* Indirect-count predication zeros every record above the runtime count.
    * A G17P direct indexed packet cannot encode either zero count, so the
    * device-side builder must skip its fixed reservation with the grounded
    * non-returning Stream Link instead of publishing a malformed draw. */
   static const uint32_t empty_jump_with_state[] = {
      UINT32_C(0x00048000), UINT32_C(0x8000005a), UINT32_C(0x12345678),
   };
   struct agx_draw empty_draw = robust_draw;
   empty_draw.b.count[0] = 0;
   memset(robust_packed, 0xcd, sizeof(robust_packed));
   robust_end = agx_vdm_draw_robust_index(
      robust_packed, AGX_CHIP_G17P, empty_draw,
      (enum agx_primitive)9, UINT32_MAX, UINT32_C(0x48000), true,
      UINT64_C(0x5a12345678));
   CHECK_EQ((uintptr_t)robust_end - (uintptr_t)robust_packed,
            sizeof(empty_jump_with_state));
   ++checks;
   if (memcmp(robust_packed, empty_jump_with_state,
              sizeof(empty_jump_with_state)) ||
       robust_packed[3] != UINT32_C(0xcdcdcdcd)) {
      fprintf(stderr, "G17P empty robust indexed jump mismatch\n");
      exit(1);
   }

   empty_draw = robust_draw;
   empty_draw.b.count[1] = 0;
   memset(robust_packed, 0xcd, sizeof(robust_packed));
   robust_end = agx_vdm_draw_robust_index(
      robust_packed, AGX_CHIP_G17P, empty_draw,
      (enum agx_primitive)9, UINT32_MAX, UINT32_C(0x48000), true,
      UINT64_C(0x5a12345678));
   CHECK_EQ((uintptr_t)robust_end - (uintptr_t)robust_packed,
            sizeof(empty_jump_with_state));
   ++checks;
   if (memcmp(robust_packed, empty_jump_with_state,
              sizeof(empty_jump_with_state)) ||
       robust_packed[3] != UINT32_C(0xcdcdcdcd)) {
      fprintf(stderr, "G17P zero-instance robust indexed jump mismatch\n");
      exit(1);
   }

   static const uint32_t empty_jump_without_state[] = {
      UINT32_C(0x8000005a), UINT32_C(0x12345678),
   };
   for (enum agx_chip chip = AGX_CHIP_G17G; chip <= AGX_CHIP_G17C; ++chip) {
      uint64_t compact;
      CHECK_EQ(agx_m5_vdm_compact(UINT64_C(0x5a12345678), &compact), true);
      CHECK_EQ(compact, UINT64_C(0x4a12345678));
      for (unsigned count = 0; count < 2; ++count) {
         struct agx_draw empty = robust_draw;
         empty.b.count[count] = 0;
         memset(robust_packed, 0xcd, sizeof(robust_packed));
         robust_end = agx_vdm_draw_robust_index(
            robust_packed, chip, empty, (enum agx_primitive)9,
            UINT32_MAX, UINT32_C(0x48000), true, compact);
         CHECK_EQ(robust_end - robust_packed, 2);
         CHECK_EQ(robust_packed[0], UINT32_C(0x8000004a));
         CHECK_EQ(robust_packed[1], UINT32_C(0x12345678));
         CHECK_EQ(robust_packed[2], UINT32_C(0xcdcdcdcd));
      }
   }
   memset(robust_packed, 0xcd, sizeof(robust_packed));
   robust_end = agx_vdm_draw_robust_index(
      robust_packed, AGX_CHIP_G17P, empty_draw,
      (enum agx_primitive)9, UINT32_MAX, 0, false, UINT64_C(0x5a12345678));
   CHECK_EQ((uintptr_t)robust_end - (uintptr_t)robust_packed,
            sizeof(empty_jump_without_state));
   ++checks;
   if (memcmp(robust_packed, empty_jump_without_state,
              sizeof(empty_jump_without_state)) ||
       robust_packed[2] != UINT32_C(0xcdcdcdcd)) {
      fprintf(stderr, "G17P reused-state empty indexed jump mismatch\n");
      exit(1);
   }

   /* The shadow-buffer calculation must never wrap either the 32-index
    * extension limit or the byte allocation passed to the GPU heap. */
   CHECK_EQ(agx_robust_index_clamped_count(64, UINT32_MAX - 16,
                                          AGX_PRIMITIVE_TRIANGLES), 64);
   CHECK_EQ(agx_robust_index_clamped_count(UINT32_MAX, UINT32_MAX - 16,
                                          AGX_PRIMITIVE_TRIANGLES),
            UINT32_MAX);
   CHECK_EQ(agx_robust_index_clamped_count(100, 12,
                                          AGX_PRIMITIVE_TRIANGLES), 44);
   /* With one accessible index, a 64-point draw still emits 64 points:
    * zero-index points affect blending and occlusion even when coincident. */
   CHECK_EQ(agx_robust_index_clamped_count(64, 1, AGX_PRIMITIVE_POINTS), 64);
   CHECK_EQ(agx_robust_index_clamped_count(100, 12, AGX_PRIMITIVE_POINTS), 100);
   CHECK_EQ(agx_robust_index_clamped_count(0, 1, AGX_PRIMITIVE_POINTS), 0);
   CHECK_EQ(agx_robust_index_clamped_count(UINT32_MAX, 1,
                                          AGX_PRIMITIVE_POINTS), UINT32_MAX);
   uint32_t payload_B = 0, allocation_B = 0;
   CHECK_EQ(agx_robust_index_shadow_sizes(
               (UINT32_MAX - 15) >> AGX_INDEX_SIZE_U32,
               AGX_INDEX_SIZE_U32, &payload_B, &allocation_B),
            true);
   CHECK_EQ(payload_B, UINT32_C(0xfffffff0));
   CHECK_EQ(allocation_B, UINT32_C(0xfffffff0));
   CHECK_EQ(agx_robust_index_shadow_sizes(
               ((UINT32_MAX - 15) >> AGX_INDEX_SIZE_U32) + 1,
               AGX_INDEX_SIZE_U32, &payload_B, &allocation_B),
            false);
   CHECK_EQ(agx_robust_index_shadow_sizes(
               1, (enum agx_index_size)3, &payload_B, &allocation_B),
            false);

   struct agx_draw rejected = {
      .b = agx_3d(1, 1, 1),
   };
   CHECK_EQ(
      agx_vdm_draw_supported(AGX_CHIP_G16P, rejected, (enum agx_primitive)6),
      false);
   rejected.b = agx_grid_indirect(UINT64_C(0x1002));
   CHECK_EQ(
      agx_vdm_draw_supported(AGX_CHIP_G17S, rejected, (enum agx_primitive)6),
      false);
   rejected = agx_draw_indexed_indirect(
      UINT64_C(0x1000), UINT64_C(0x2000), 0, AGX_INDEX_SIZE_U32, false);
   CHECK_EQ(
      agx_vdm_draw_supported(AGX_CHIP_G17S, rejected, (enum agx_primitive)6),
      false);
   struct agx_draw byte_indexed = agx_draw_indexed(1, 1, 0, 0, 0, UINT64_C(0x1000), 4,
                               AGX_INDEX_SIZE_U8, false);
   CHECK_EQ(
      agx_vdm_draw_supported(AGX_CHIP_G17S, byte_indexed, (enum agx_primitive)6),
      true);
}

static void
test_agx3_live_vdm_barrier(void)
{
   const enum agx_chip grounded[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };

   for (unsigned i = 0; i < sizeof(grounded) / sizeof(grounded[0]); ++i) {
      uint32_t packed[2] = {UINT32_C(0xdeadbeef), UINT32_C(0xfeedface)};
      uint32_t *end = agx_vdm_barrier(packed, grounded[i]);
      CHECK_EQ((uintptr_t)end - (uintptr_t)packed, sizeof(uint32_t));
      CHECK_EQ(packed[0], UINT32_C(0x21000960));
      CHECK_EQ(packed[1], UINT32_C(0xfeedface));
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G16P,
      AGX_CHIP_G17A,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      uint32_t packed = UINT32_C(0xdeadbeef);
      CHECK_EQ(agx_vdm_barrier(&packed, rejected[i]) == &packed, true);
      CHECK_EQ(packed, UINT32_C(0xdeadbeef));
   }
}

static void
test_agx3_cdm_stream_link(void)
{
   const enum agx_chip grounded[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   const uint8_t expected[AGX3_CDM_STREAM_LINK_SIZE] = {
      0xbc, 0x9a, 0x00, 0x30, 0x78, 0x56, 0x34, 0x12,
   };
   struct agx3_cdm_stream_link_info input = {
      .target = UINT64_C(0x9abc12345678),
      .with_return = true,
   };

   for (unsigned i = 0; i < sizeof(grounded) / sizeof(grounded[0]); ++i) {
      uint8_t packed[AGX3_CDM_STREAM_LINK_SIZE];
      input.chip = grounded[i];
      CHECK_EQ(agx3_cdm_stream_link_target_grounded(input.chip), true);
      CHECK_EQ(agx3_cdm_stream_link_pack(&input, packed), true);
      ++checks;
      if (memcmp(packed, expected, sizeof(expected))) {
         fprintf(stderr, "AGX3 CDM link mismatch for chip %u\n", input.chip);
         for (unsigned b = 0; b < sizeof(packed); ++b)
            fprintf(stderr, "%02x%s", packed[b],
                    b + 1 == sizeof(packed) ? "\n" : " ");
         exit(1);
      }

      struct agx3_cdm_stream_link_info decoded;
      CHECK_EQ(agx3_cdm_stream_link_unpack(input.chip, packed, sizeof(packed),
                                           &decoded),
               true);
      CHECK_EQ(decoded.target, input.target);
      CHECK_EQ(decoded.with_return, input.with_return);
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      uint8_t packed[AGX3_CDM_STREAM_LINK_SIZE];
      input.chip = rejected[i];
      CHECK_EQ(agx3_cdm_stream_link_target_grounded(input.chip), false);
      CHECK_EQ(agx3_cdm_stream_link_pack(&input, packed), false);
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK_EQ(packed[b], 0);
   }

   uint8_t packed[AGX3_CDM_STREAM_LINK_SIZE];
   input.chip = AGX_CHIP_G17S;
   input.target = UINT64_C(1) << 48;
   CHECK_EQ(agx3_cdm_stream_link_pack(&input, packed), false);
   input.target = UINT64_C(0x9abc12345678);
   CHECK_EQ(agx3_cdm_stream_link_pack(NULL, packed), false);
   CHECK_EQ(agx3_cdm_stream_link_pack(&input, NULL), false);

   CHECK_EQ(agx3_cdm_stream_link_pack(&input, packed), true);
   struct agx3_cdm_stream_link_info decoded;
   CHECK_EQ(agx3_cdm_stream_link_unpack(input.chip, packed, 7, &decoded),
            false);
   packed[2] = 1;
   CHECK_EQ(agx3_cdm_stream_link_unpack(input.chip, packed, sizeof(packed),
                                        &decoded),
            false);
   CHECK_EQ(decoded.target, 0);
}

static void
test_agx3_cdm_link_padding(void)
{
   const enum agx_chip grounded[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };

   for (unsigned i = 0; i < sizeof(grounded) / sizeof(grounded[0]); ++i) {
      for (uint8_t count = 0; count < 8; ++count) {
         struct agx3_cdm_link_padding_info info = {
            .chip = grounded[i],
            .count = count,
         };
         uint8_t packed[AGX3_CDM_LINK_PADDING_SIZE];
         struct agx3_cdm_link_padding_info decoded;
         CHECK_EQ(agx3_cdm_link_padding_pack(&info, packed), true);
         CHECK_EQ(agx3_cmdstream_load_le32(packed),
                  UINT32_C(0xa0000000) | count);
         CHECK_EQ(agx3_cdm_link_padding_unpack(
                     info.chip, packed, sizeof(packed), &decoded),
                  true);
         CHECK_EQ(decoded.chip, info.chip);
         CHECK_EQ(decoded.count, count);
      }
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      struct agx3_cdm_link_padding_info info = {
         .chip = rejected[i],
         .count = 0,
      };
      uint8_t packed[AGX3_CDM_LINK_PADDING_SIZE];
      CHECK_EQ(agx3_cdm_link_padding_pack(&info, packed), false);
      CHECK_EQ(agx3_cmdstream_load_le32(packed), 0);
   }

   struct agx3_cdm_link_padding_info info = {
      .chip = AGX_CHIP_G17G,
      .count = 8,
   };
   uint8_t packed[AGX3_CDM_LINK_PADDING_SIZE];
   CHECK_EQ(agx3_cdm_link_padding_pack(&info, packed), false);
   info.count = 0;
   CHECK_EQ(agx3_cdm_link_padding_pack(&info, packed), true);
   struct agx3_cdm_link_padding_info decoded;
   CHECK_EQ(agx3_cdm_link_padding_unpack(info.chip, packed, 3, &decoded),
            false);
   packed[1] = 1;
   CHECK_EQ(agx3_cdm_link_padding_unpack(
               info.chip, packed, sizeof(packed), &decoded),
            false);
   CHECK_EQ(decoded.count, 0);
   CHECK_EQ(agx3_cdm_link_padding_pack(NULL, packed), false);
   CHECK_EQ(agx3_cdm_link_padding_pack(&info, NULL), false);
}

static void
test_agx3_cdm_barrier(void)
{
   const enum agx_chip pre_gen7[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   };
   const uint8_t expected0[AGX3_CDM_BARRIER_SIZE] = {0x60, 0x01, 0x00, 0x60};
   const uint8_t expected5[AGX3_CDM_BARRIER_SIZE] = {0x60, 0x15, 0x00, 0x60};

   for (unsigned i = 0; i < sizeof(pre_gen7) / sizeof(pre_gen7[0]); ++i) {
      for (unsigned selection = 0; selection < 2; ++selection) {
         struct agx3_cdm_barrier_info info = {
            .chip = pre_gen7[i],
            .flags = selection ? 5 : 0,
         };
         uint8_t packed[AGX3_CDM_BARRIER_SIZE];
         struct agx3_cdm_barrier_info decoded;
         CHECK_EQ(agx3_cdm_barrier_pack(&info, packed), true);
         ++checks;
         if (memcmp(packed, selection ? expected5 : expected0,
                    sizeof(packed))) {
            fprintf(stderr, "AGX3 CDM pre-Gen7 barrier mismatch\n");
            exit(1);
         }
         CHECK_EQ(agx3_cdm_barrier_unpack(info.chip, packed, sizeof(packed),
                                          &decoded),
                  true);
         CHECK_EQ(decoded.chip, info.chip);
         CHECK_EQ(decoded.flags, info.flags);
      }

      for (uint8_t flags = 0; flags < 16; ++flags) {
         if (flags == 0 || flags == 5)
            continue;
         struct agx3_cdm_barrier_info rejected = {
            .chip = pre_gen7[i],
            .flags = flags,
         };
         uint8_t packed[AGX3_CDM_BARRIER_SIZE];
         CHECK_EQ(agx3_cdm_barrier_pack(&rejected, packed), false);
         CHECK_EQ(agx3_cmdstream_load_le32(packed), 0);
      }
   }

   const enum agx_chip gen7[] = {
      AGX_CHIP_G17G,
      AGX_CHIP_G17S,
      AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < sizeof(gen7) / sizeof(gen7[0]); ++i) {
      for (uint8_t flags = 0; flags < 16; ++flags) {
         struct agx3_cdm_barrier_info info = {
            .chip = gen7[i],
            .flags = flags,
         };
         uint8_t packed[AGX3_CDM_BARRIER_SIZE];
         struct agx3_cdm_barrier_info decoded;
         CHECK_EQ(agx3_cdm_barrier_pack(&info, packed), true);
         CHECK_EQ(agx3_cmdstream_load_le32(packed),
                  agx3_cdm_barrier_word(flags));
         CHECK_EQ(agx3_cdm_barrier_unpack(info.chip, packed, sizeof(packed),
                                          &decoded),
                  true);
         CHECK_EQ(decoded.flags, flags);
      }
   }

   /* Memory-scope record on HAL300 and G17P. Ordering flags alone do not
    * make writes visible through another virtual mapping of the same
    * memory. */
   const enum agx_chip memory_scope[] = {
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17P,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(memory_scope); ++i) {
      struct agx3_cdm_barrier_info full = {
         .chip = memory_scope[i], .flags = 5, .flush_memory = true,
      }, decoded;
      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      CHECK_EQ(agx3_cdm_barrier_flush_memory_grounded(full.chip), true);
      CHECK_EQ(agx3_cdm_barrier_pack(&full, packed), true);
      CHECK_EQ(agx3_cmdstream_load_le32(packed), 0x6000157c);
      CHECK_EQ(agx3_cdm_barrier_unpack(full.chip, packed, sizeof(packed), &decoded), true);
      CHECK_EQ(decoded.flush_memory, true);
      CHECK_EQ(decoded.flags, 5);
      /* Only the strong record takes the memory scope. */
      full.flags = 0;
      CHECK_EQ(agx3_cdm_barrier_pack(&full, packed), false);
      /* A partial scope is not a grounded record. */
      agx3_cmdstream_store_le32(packed, 0x60001570);
      CHECK_EQ(agx3_cdm_barrier_unpack(memory_scope[i], packed, sizeof(packed), &decoded), false);
   }
   const enum agx_chip no_memory_scope[] = {AGX_CHIP_G15S, AGX_CHIP_G16G};
   for (unsigned i = 0; i < ARRAY_SIZE(no_memory_scope); ++i) {
      struct agx3_cdm_barrier_info full = {
         .chip = no_memory_scope[i], .flags = 5, .flush_memory = true,
      };
      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      CHECK_EQ(agx3_cdm_barrier_flush_memory_grounded(full.chip), false);
      CHECK_EQ(agx3_cdm_barrier_pack(&full, packed), false);
   }

   const enum agx_chip unsupported[] = {
      AGX_CHIP_G13G, AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); ++i) {
      struct agx3_cdm_barrier_info info = {
         .chip = unsupported[i],
         .flags = 0,
      };
      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      CHECK_EQ(agx3_cdm_barrier_target_grounded(info.chip), false);
      CHECK_EQ(agx3_cdm_barrier_pack(&info, packed), false);
      CHECK_EQ(agx3_cmdstream_load_le32(packed), 0);
   }

   uint8_t packed[AGX3_CDM_BARRIER_SIZE];
   struct agx3_cdm_barrier_info info = {
      .chip = AGX_CHIP_G17S,
      .flags = 0,
   };
   CHECK_EQ(agx3_cdm_barrier_pack(&info, packed), true);
   struct agx3_cdm_barrier_info decoded;
   CHECK_EQ(agx3_cdm_barrier_unpack(info.chip, packed, 3, &decoded), false);
   packed[0] |= 1;
   CHECK_EQ(agx3_cdm_barrier_unpack(info.chip, packed, sizeof(packed),
                                    &decoded),
            false);
   CHECK_EQ(decoded.flags, 0);
   CHECK_EQ(agx3_cdm_barrier_pack(NULL, packed), false);
   CHECK_EQ(agx3_cdm_barrier_pack(&info, NULL), false);

   /* The wide-visibility form: the G15 parts on the native path and G16G
    * pack and decode it; other chips refuse it in both directions. */
   const enum agx_chip wide[] = {
      AGX_CHIP_G15G,
      AGX_CHIP_G15S,
      AGX_CHIP_G15C,
      AGX_CHIP_G16G,
   };
   for (unsigned i = 0; i < sizeof(wide) / sizeof(wide[0]); ++i) {
      struct agx3_cdm_barrier_info wide_info = {
         .chip = wide[i],
         .flags = 5,
         .wide_visibility = true,
      };
      CHECK_EQ(agx3_cdm_barrier_pack(&wide_info, packed), true);
      CHECK_EQ(agx3_cmdstream_load_le32(packed),
               agx3_cdm_barrier_word(5) | AGX3_CDM_BARRIER_WIDE_VISIBILITY);
      CHECK_EQ(agx3_cdm_barrier_unpack(wide[i], packed, sizeof(packed),
                                       &decoded),
               true);
      CHECK_EQ(decoded.wide_visibility, true);
      CHECK_EQ(decoded.flags, 5);
   }

   const enum agx_chip narrow[] = {
      AGX_CHIP_G15D,
      AGX_CHIP_G17P,
   };
   for (unsigned i = 0; i < sizeof(narrow) / sizeof(narrow[0]); ++i) {
      struct agx3_cdm_barrier_info narrow_info = {
         .chip = narrow[i],
         .flags = 5,
         .wide_visibility = true,
      };
      CHECK_EQ(agx3_cdm_barrier_pack(&narrow_info, packed), false);
      agx3_cmdstream_store_le32(packed, agx3_cdm_barrier_word(5) |
                                           AGX3_CDM_BARRIER_WIDE_VISIBILITY);
      CHECK_EQ(agx3_cdm_barrier_unpack(narrow[i], packed, sizeof(packed),
                                       &decoded),
               false);
   }
}

static void
test_agx3_cdm_substream_barrier(void)
{
   const enum agx_chip targets[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };

   for (unsigned i = 0; i < sizeof(targets) / sizeof(targets[0]); ++i) {
      const bool gen7 = targets[i] == AGX_CHIP_G17G ||
                        targets[i] == AGX_CHIP_G17S ||
                        targets[i] == AGX_CHIP_G17C;
      for (uint16_t substream = 0; substream < UINT8_MAX; ++substream) {
         for (uint8_t flags = 0; flags < 16; ++flags) {
            if (!gen7 && flags != 0 && flags != 5)
               continue;

            struct agx3_cdm_substream_barrier_info info = {
               .chip = targets[i],
               .flags = flags,
               .substream = (uint8_t)substream,
            };
            uint8_t packed[AGX3_CDM_BARRIER_SIZE];
            struct agx3_cdm_substream_barrier_info decoded;
            CHECK_EQ(agx3_cdm_substream_barrier_pack(&info, packed), true);
            CHECK_EQ(agx3_cmdstream_load_le32(packed),
                     agx3_cdm_barrier_word(flags) |
                        ((uint32_t)(substream + 1) << 21));
            CHECK_EQ(agx3_cdm_substream_barrier_unpack(
                        info.chip, packed, sizeof(packed), &decoded),
                     true);
            CHECK_EQ(decoded.chip, info.chip);
            CHECK_EQ(decoded.flags, info.flags);
            CHECK_EQ(decoded.substream, info.substream);
         }
      }
   }

   struct agx3_cdm_substream_barrier_info info = {
      .chip = AGX_CHIP_G17S,
      .flags = 0,
      .substream = UINT8_MAX,
   };
   uint8_t packed[AGX3_CDM_BARRIER_SIZE];
   CHECK_EQ(agx3_cdm_substream_barrier_pack(&info, packed), false);
   CHECK_EQ(agx3_cmdstream_load_le32(packed), 0);
   info.substream = 0;
   info.chip = AGX_CHIP_G17A;
   CHECK_EQ(agx3_cdm_substream_barrier_pack(&info, packed), false);
   info.chip = AGX_CHIP_G17S;
   CHECK_EQ(agx3_cdm_substream_barrier_pack(&info, packed), true);
   struct agx3_cdm_substream_barrier_info decoded;
   CHECK_EQ(agx3_cdm_substream_barrier_unpack(info.chip, packed, 3, &decoded),
            false);
   packed[2] ^= 1;
   CHECK_EQ(agx3_cdm_substream_barrier_unpack(
               info.chip, packed, sizeof(packed), &decoded),
            false);
   CHECK_EQ(decoded.substream, 0);
   CHECK_EQ(agx3_cdm_substream_barrier_pack(NULL, packed), false);
   CHECK_EQ(agx3_cdm_substream_barrier_pack(&info, NULL), false);
}

static void
test_agx3_cdm_secondary_barrier(void)
{
   const enum agx_chip grounded[] = {
      AGX_CHIP_G17G,
      AGX_CHIP_G17S,
      AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < sizeof(grounded) / sizeof(grounded[0]); ++i) {
      uint8_t packed[AGX3_CDM_SECONDARY_BARRIER_SIZE];
      CHECK_EQ(agx3_cdm_secondary_barrier_pack(grounded[i], packed), true);
      CHECK_EQ(agx3_cmdstream_load_le32(packed), UINT32_C(0x60000502));
      CHECK_EQ(agx3_cdm_secondary_barrier_validate(
                  grounded[i], packed, sizeof(packed)),
               true);
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G15G, AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17A,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      uint8_t packed[AGX3_CDM_SECONDARY_BARRIER_SIZE];
      CHECK_EQ(agx3_cdm_secondary_barrier_pack(rejected[i], packed), false);
      CHECK_EQ(agx3_cmdstream_load_le32(packed), 0);
   }

   uint8_t packed[AGX3_CDM_SECONDARY_BARRIER_SIZE];
   CHECK_EQ(agx3_cdm_secondary_barrier_pack(AGX_CHIP_G17G, packed), true);
   CHECK_EQ(agx3_cdm_secondary_barrier_validate(AGX_CHIP_G17G, packed, 3),
            false);
   packed[0] ^= 1;
   CHECK_EQ(agx3_cdm_secondary_barrier_validate(
               AGX_CHIP_G17G, packed, sizeof(packed)),
            false);
   CHECK_EQ(agx3_cdm_secondary_barrier_pack(AGX_CHIP_G17G, NULL), false);
}

static void
test_agx3_cdm_stream_terminate(void)
{
   const enum agx_chip grounded[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   const uint8_t expected[AGX3_CDM_STREAM_TERMINATE_SIZE] = {
      0x00, 0x00, 0x00, 0x40,
   };

   for (unsigned i = 0; i < sizeof(grounded) / sizeof(grounded[0]); ++i) {
      uint8_t packed[AGX3_CDM_STREAM_TERMINATE_SIZE];
      CHECK_EQ(agx3_cdm_stream_terminate_pack(grounded[i], packed), true);
      ++checks;
      if (memcmp(packed, expected, sizeof(expected))) {
         fprintf(stderr, "AGX3 CDM terminate mismatch\n");
         exit(1);
      }
      CHECK_EQ(agx3_cdm_stream_terminate_validate(
                  grounded[i], packed, sizeof(packed)),
               true);
   }

   const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      uint8_t packed[AGX3_CDM_STREAM_TERMINATE_SIZE];
      CHECK_EQ(agx3_cdm_stream_terminate_pack(rejected[i], packed), false);
      for (unsigned b = 0; b < sizeof(packed); ++b)
         CHECK_EQ(packed[b], 0);
   }

   uint8_t packed[AGX3_CDM_STREAM_TERMINATE_SIZE];
   CHECK_EQ(agx3_cdm_stream_terminate_pack(AGX_CHIP_G17S, packed), true);
   CHECK_EQ(agx3_cdm_stream_terminate_validate(AGX_CHIP_G17S, packed, 3),
            false);
   packed[0] ^= 1;
   CHECK_EQ(agx3_cdm_stream_terminate_validate(
               AGX_CHIP_G17S, packed, sizeof(packed)),
            false);
   CHECK_EQ(agx3_cdm_stream_terminate_pack(AGX_CHIP_G17S, NULL), false);
}

static void
test_robust_index_shadow(void)
{
   /* Exercise the actual host/device copy routine lane by lane. Short and
    * unaligned inputs used to underflow the vector-copy length, and uint32
    * extensions longer than 32 bytes were only partly initialized. */
   for (unsigned size = AGX_INDEX_SIZE_U8; size <= AGX_INDEX_SIZE_U32; ++size) {
      const unsigned width = 1u << size;
      CHECK_EQ(agx_robust_index_available(UINT32_MAX, 128, size), 0);
      CHECK_EQ(agx_robust_index_available(0x80000000u, 128, size), 0);
      CHECK_EQ(agx_robust_index_available(3, 3 * width, size), 0);
      CHECK_EQ(agx_robust_index_available(3, 4 * width - 1, size), 0);
      CHECK_EQ(agx_robust_index_available(3, 5 * width - 1, size), 1);

      struct agx_draw wrapped =
         agx_draw_indexed(UINT32_MAX, 1, 1, 0, 0, 0x1000, 128, size, false);
      CHECK_EQ(agx_direct_draw_overreads_indices(wrapped), true);
      wrapped.start = 0x80000000u;
      wrapped.b.count[0] = 1;
      CHECK_EQ(agx_direct_draw_overreads_indices(wrapped), true);

      for (unsigned offset = 0; offset < 16; ++offset) {
         for (unsigned available = 1; available <= 33; ++available) {
            uint8_t src[512], dst[512];
            for (unsigned i = 0; i < sizeof(src); ++i)
               src[i] = 1 + (i % 253);
            for (unsigned extra = 1; extra <= 32; ++extra) {
               memset(dst, 0xcd, sizeof(dst));
               const unsigned in_B = available * width;
               const unsigned out_B = (available + extra) * width;
               for (unsigned lane = 0; lane < 32; ++lane)
                  agx_robust_index_copy(dst + 16, src + offset, in_B, out_B,
                                        lane);

               for (unsigned i = 0; i < sizeof(dst); ++i) {
                  const uint8_t expected = i < 16 || i >= 16 + out_B ? 0xcd
                                           : i < 16 + in_B
                                              ? src[offset + i - 16]
                                              : 0;
                  CHECK_EQ(dst[i], expected);
               }
            }
         }
      }
   }
}

static void
test_xfb_vertex_count(void)
{
   CHECK_EQ(agx_xfb_vertex_count(0, 0, 32), 0);
   CHECK_EQ(agx_xfb_vertex_count(31, 0, 32), 0);
   CHECK_EQ(agx_xfb_vertex_count(32, 0, 32), 1);
   CHECK_EQ(agx_xfb_vertex_count(95, 0, 32), 2);
   CHECK_EQ(agx_xfb_vertex_count(96, 0, 32), 3);
   CHECK_EQ(agx_xfb_vertex_count(96, 32, 32), 2);
   CHECK_EQ(agx_xfb_vertex_count(32, 32, 32), 0);
   CHECK_EQ(agx_xfb_vertex_count(16, 32, 32), 0);
   CHECK_EQ(agx_xfb_vertex_count(UINT32_MAX, UINT32_MAX - 15, 16), 0);
   CHECK_EQ(agx_xfb_vertex_count(UINT32_MAX, UINT32_MAX - 16, 16), 1);
   CHECK_EQ(agx_xfb_vertex_count(UINT32_MAX, 0, 1), UINT32_MAX);
   CHECK_EQ(agx_xfb_vertex_count(UINT32_MAX, 0, 0), 0);
}

int
main(void)
{
   CHECK_EQ(AGX_TEXTURE_GEN9_LENGTH, 32);
   CHECK_EQ(AGX_TEXTURE_GEN10_LENGTH, 32);
   CHECK_EQ(AGX_PBE_GEN9_LENGTH, 32);
   CHECK_EQ(AGX_PBE_GEN10_LENGTH, 32);
   CHECK_EQ(AGX_SAMPLER_HAL200_LENGTH, 8);
   CHECK_EQ(AGX_SAMPLER_GEN5_LENGTH, 8);

   test_m5_live_descriptors();
   test_generation_boundaries();
   test_sampler_generations();
   test_agx3_cdm_launch();
   test_agx3_live_vdm_draw();
   test_agx3_live_vdm_barrier();
   test_agx3_cdm_stream_link();
   test_agx3_cdm_link_padding();
   test_agx3_cdm_barrier();
   test_agx3_cdm_substream_barrier();
   test_agx3_cdm_secondary_barrier();
   test_agx3_cdm_stream_terminate();
   test_robust_index_shadow();
   test_xfb_vertex_count();

   printf("agx3 packing: all %u checks passed\n", checks);
   return 0;
}
