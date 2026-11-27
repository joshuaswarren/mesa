/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "lib/agx_chip.h"

int
main(void)
{
   /* The reduced word 0x1ff on the dies it was measured on: the G13X family
    * (the gen-13 S/C/D parts fold to it) and G14X at GPU generation 14. */
   assert(agx2_cdm_barrier_default_word(AGX_CHIP_G13G, 13) == 0x1ffu);
   assert(agx2_cdm_barrier_default_word(AGX_CHIP_G13X, 13) == 0x1ffu);
   assert(agx2_cdm_barrier_default_word(AGX_CHIP_G14X, 14) == 0x1ffu);

   /* The full word everywhere else: the gen-13/14 base parts and every
    * later chip, including a hypothetical other-generation G14X. */
   const enum agx_chip full_chips[] = {
      AGX_CHIP_G14G,
      AGX_CHIP_G15P, AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C,
      AGX_CHIP_G15D,
      AGX_CHIP_G16P, AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C,
      AGX_CHIP_G17P, AGX_CHIP_G17A, AGX_CHIP_G17G, AGX_CHIP_G17S,
      AGX_CHIP_G17C,
      AGX_CHIP_G18G,
   };
   for (unsigned i = 0; i < sizeof(full_chips) / sizeof(full_chips[0]); ++i)
      for (unsigned gen = 13; gen <= 18; ++gen)
         assert(agx2_cdm_barrier_default_word(full_chips[i], gen) ==
                AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_default_word(AGX_CHIP_G14X, 15) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_default_word(AGX_CHIP_G13X, 14) == 0x1ffu);

   /* The emit-time upgrade: a sparse resource or fullcdmbarrier selects the
    * full word; otherwise the chip's default word passes through. */
   assert(agx2_cdm_barrier_emit_word(0x1ffu, true, false) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_emit_word(0x1ffu, false, true) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_emit_word(0x1ffu, true, true) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_emit_word(0x1ffu, false, false) == 0x1ffu);
   assert(agx2_cdm_barrier_emit_word(AGX_CDM_BARRIER_FULL_WORD, false,
                                    false) == AGX_CDM_BARRIER_FULL_WORD);

   /* Pure compute-to-compute SSBO word: derived from the device's measured
    * default word (0x1ff -> 0x160; full word unchanged), still upgrading under
    * sparse/force_full. */
   assert(agx2_cdm_barrier_ssbo_word(0x1ffu) == AGX_CDM_BARRIER_SSBO_WORD);
   assert(agx2_cdm_barrier_ssbo_word(AGX_CDM_BARRIER_FULL_WORD) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_ssbo_word(
             agx2_cdm_barrier_default_word(AGX_CHIP_G14X, 14)) == 0x160u);
   assert(agx2_cdm_barrier_ssbo_word(
             agx2_cdm_barrier_default_word(AGX_CHIP_G13X, 13)) == 0x160u);
   assert(agx2_cdm_barrier_ssbo_word(
             agx2_cdm_barrier_default_word(AGX_CHIP_G14G, 14)) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_emit_word(0x160u, false, false) == 0x160u);
   assert(agx2_cdm_barrier_emit_word(0x160u, true, false) ==
          AGX_CDM_BARRIER_FULL_WORD);
   assert(agx2_cdm_barrier_emit_word(0x160u, false, true) ==
          AGX_CDM_BARRIER_FULL_WORD);

   return 0;
}
