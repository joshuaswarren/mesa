/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The coverage word that `movmsk` and `depthf` consume, and the 16-bit
 * shift-add that builds it.
 *
 * ## AGX3 `iadd` is a shift-add
 *
 * PROVEN by a five-point probe ladder (`o[i] = a[i] + (b[i] << k)`, k = 0..4,
 * offline `applegpu-nt -arch applegpu_g17p`): the AGX3b form is
 *
 *     dest = src0 + (src1 << f54)
 *
 * with f54 carrying the shift directly, src1 always in f31, and src0 either a
 * register (f23 = 2, value in f16) or an 8-bit immediate (f23 = 1, value in
 * f15). Doubling compiles to `0 + (v << 1)`, which pins f31 as the shifted
 * operand rather than f16. Widths: f10 = destination, f24 = src0, f39 = src1;
 * 1 is 16-bit, 2 is 32-bit. f42 = 1 marks the 16-bit shifted operand, f27 = 1
 * accompanies a register src0, and f11 is the same one-bit destination hint
 * every AGX3 general-register destination carries.
 *
 * The pre-existing `iadd` entry in agx3_alu_table.h is 32-bit only, has no
 * shift and no immediate src0, so it cannot express any of this; rather than
 * widen a generated table that other tests pin, this is a separate encoder.
 *
 * ## The coverage word
 *
 * At 1x, both `movmsk` and `depthf` read a staged 32-bit word whose low half
 * Apple builds as
 *
 *     word = N + ((T + (T << 4)) << 4)   ==   N | (T << 4) | (T << 8)
 *
 * where T is special register 2 read 16-bit and N is the live-sample nibble:
 * the written mask for `[[sample_mask]] = K`, and always 15 for a `[[depth]]`
 * write. The high half is never written; Apple leaves it undefined. This is
 * witnessed identically in fdiscard (N from a register) and fdepth (N = 15),
 * and in every one of the eleven 1x depth probes.
 *
 * The unconditional-kill case is the one place Apple folds the whole word to a
 * constant, 0xFFFFFF00, and that constant already ships as
 * AGX3_ZS_KILL_ALL_SAMPLES.
 *
 * Multisampled coverage takes a different route (it rebuilds the word from
 * special register 39 with a shift and a mask instead of reading SR 2) and is
 * deliberately not modelled here.
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
