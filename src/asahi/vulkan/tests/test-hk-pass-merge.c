/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "asahi/vulkan/hk_pass_merge.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
#define CHECK(x)                                                              \
   do {                                                                       \
      checks++;                                                               \
      if (!(x)) {                                                             \
         fprintf(stderr, "line %u: %s\n", __LINE__, #x);                      \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

int
main(void)
{
   static const unsigned sample_counts[] = {0, 1, 2, 4, 8, 16};
   static const unsigned sample_sizes[] = {0, 4, 8, 12, 16, 32, 64};

   for (unsigned c = 0; c <= AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = (enum agx_chip)c;
      const bool hal200_or_m5 =
         agx_chip_native_hal200(chip) || agx_chip_g17_m5(chip);

      CHECK(hk_pass_merge_layout_proof_supported(chip) == hal200_or_m5);

      for (unsigned si = 0; si < sizeof(sample_counts) / sizeof(sample_counts[0]);
           ++si) {
         const unsigned s = sample_counts[si];
         const bool valid_msaa = (s == 2 || s == 4);

         for (unsigned bi = 0;
              bi < sizeof(sample_sizes) / sizeof(sample_sizes[0]); ++bi) {
            const unsigned b = sample_sizes[bi];
            const bool wide =
               b > AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE;

            /* Generic chip_id = 0 (conservative for reload defect). */
            const bool expect_generic =
               hal200_or_m5 && valid_msaa &&
               !(wide && agx_chip_uncovered_tile_ms_reload_defect(chip, 0, s));
            CHECK(hk_msaa_pass_merge_supported(chip, 0, s, b) ==
                  expect_generic);
         }
      }
   }

   /* Concrete HAL200 SoCs: 1-8 B/sample merges at 2x and 4x everywhere. */
   static const uint32_t g17p_ids[] = {
      AGX_CHIP_ID_A18_PRO,
      AGX_CHIP_ID_M4_PRO,
      AGX_CHIP_ID_M4_MAX,
   };
   for (unsigned i = 0; i < sizeof(g17p_ids) / sizeof(g17p_ids[0]); ++i) {
      const uint32_t id = g17p_ids[i];
      CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 2, 0));
      CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 4, 0));
      CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 2, 8));
      CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 4, 8));
      CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 1, 8));
      CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 8, 8));
      /* Wide (> 8 B/sample) respects agx_chip_uncovered_tile_ms_reload_defect:
       * M4 Pro/Max have no 2x reload defect, while 4x and A18 Pro 2x/4x do. */
      CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 2, 16) ==
            (id != AGX_CHIP_ID_A18_PRO));
      CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G17P, id, 4, 16));
   }

   /* G16G (M4) HAL200: 1-8 B/sample merges at 2x/4x, wide stays off. */
   CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G16G, 0x8132, 2, 8));
   CHECK(hk_msaa_pass_merge_supported(AGX_CHIP_G16G, 0x8132, 4, 8));
   CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G16G, 0x8132, 2, 16));
   CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G16G, 0x8132, 4, 16));

   /* M5 (HAL300): merges 2x/4x at any sample size (no uncovered-tile defect). */
   static const enum agx_chip m5_chips[] = {
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < sizeof(m5_chips) / sizeof(m5_chips[0]); ++i) {
      CHECK(hk_msaa_pass_merge_supported(m5_chips[i], 0, 2, 8));
      CHECK(hk_msaa_pass_merge_supported(m5_chips[i], 0, 4, 8));
      CHECK(hk_msaa_pass_merge_supported(m5_chips[i], 0, 2, 16));
      CHECK(hk_msaa_pass_merge_supported(m5_chips[i], 0, 4, 32));
      CHECK(!hk_msaa_pass_merge_supported(m5_chips[i], 0, 8, 8));
   }

   /* Non-HAL200/non-M5 chips stay off. */
   CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G13G, 0, 2, 8));
   CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G14G, 0, 4, 8));
   CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G15G, 0, 2, 8));
   CHECK(!hk_msaa_pass_merge_supported(AGX_CHIP_G18G, 0, 4, 8));

   printf("hk_pass_merge policy: %u checks passed\n", checks);
   return 0;
}
