/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/lib/agx_chip.h"
#include <stdbool.h>
#include <stdint.h>

/* Whether consecutive 2x/4x MSAA render-pass instances may continue in one
 * render stream.
 *
 * Both HAL200 (G16G, G17P) and HAL300/M5 (G17G, G17S, G17C) share the
 * partial-render per-sample store/reload programs required by render-pass
 * continuation. On chips whose hardware multisample background reload drops
 * sample bytes above AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE on
 * uncovered tiles (agx_chip_uncovered_tile_ms_reload_defect), wide
 * multisample tilebuffers stay unmerged so a continuation split or partial
 * render never reloads wide samples through bg.partial.
 */
static inline bool
hk_msaa_pass_merge_supported(enum agx_chip chip, uint32_t chip_id,
                             unsigned samples, unsigned sample_size_B)
{
   if (samples != 2 && samples != 4)
      return false;
   if (!agx_chip_native_hal200(chip) && !agx_chip_g17_m5(chip))
      return false;
   if (sample_size_B > AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE &&
       agx_chip_uncovered_tile_ms_reload_defect(chip, chip_id, samples))
      return false;
   return true;
}

/* Whether render-pass continuation may elide non-attachment layout transitions
 * and prove attachment-only layouts exclude descriptor reads. Shared across
 * HAL200 and M5. */
static inline bool
hk_pass_merge_layout_proof_supported(enum agx_chip chip)
{
   return agx_chip_native_hal200(chip) || agx_chip_g17_m5(chip);
}
