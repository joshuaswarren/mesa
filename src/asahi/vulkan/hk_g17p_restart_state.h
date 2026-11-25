/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Emit a HAL200 restart State Update only when the VDM stream does not already
 * hold `index` (each State Update drains the vertex pipeline, ~3.5 us each on
 * M4 Pro), or immediately after a fresh bind tail to preserve
 * selector/state/draw ordering.
 */
static inline bool
hk_g17p_restart_state_needed(bool tail_emitted, bool known,
                             uint32_t known_index, uint32_t index)
{
   return tail_emitted || !known || known_index != index;
}
