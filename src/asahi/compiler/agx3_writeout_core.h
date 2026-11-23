/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * 16-bit shift-add (`iadd`: dest = src0 + (src1 << f54)) and 1x coverage-word
 * builders for `movmsk` / `depthf`.
 *
 * At 1x, the staged coverage word's low half is `N | (T << 4) | (T << 8)`,
 * where T is 16-bit SR 2 and N is the live-sample nibble (15 for depth writes;
 * unconditional kill-all uses AGX3_ZS_KILL_ALL_SAMPLES = 0xFFFFFF00).
 * Multisampled coverage is built from SR 39 separately.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_WRITEOUT_MAX_LENGTH 10u

struct agx3_shift_add_desc {
   uint16_t dest;        /* 16-bit (half) register index */
   bool dest_hint;       /* f11 */
   bool src0_immediate;  /* src0 is an 8-bit immediate rather than a register */
   uint16_t src0;        /* half-register index, or the immediate value */
   bool src0_kill;       /* register src0 only */
   uint16_t src1;        /* half-register index of the shifted operand */
   bool src1_kill;       /* f42: last read of the shifted operand */
   uint8_t shift;        /* 0..4, witnessed range */
   uint16_t wait_mask;
};

/* dest(16) = src0 + (src1(16) << shift) */
bool agx3_shift_add16_encode(enum agx_isa_encoding enc,
                             const struct agx3_shift_add_desc *d,
                             uint8_t out[AGX3_WRITEOUT_MAX_LENGTH],
                             uint8_t *len);

#ifdef __cplusplus
}
#endif
