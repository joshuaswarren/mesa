/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 staging-register file writers (`ldimm.ep2`, staging/general `clr`).
 *
 * Destination classes: f9=2 GPR (f2), f9=3 uniform (f3), f9=4 TPS spill (f4),
 * f9=5 staging register (f5, kill f13, scoreboard slot f12). Staging writes
 * are asynchronous and signal f12 for consuming writeout instructions.
 * Currently AGX3B only; other encodings fail closed.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* copy is 10 bytes, ldimm.ep2 is 10 bytes, clr.ep0 is 4. */
#define AGX3_STAGE_MAX_LENGTH 10u

/* Scalar vertex export producers witnessed in native G17P terrain. FMA
 * allows a uniform in source zero; all other operands are general registers.
 * Indices are 16-bit halves. Live/discard hints belong to general sources. */
bool agx3_stage_vertex_alu(enum agx_isa_encoding enc, bool fma,
                           uint16_t dest, uint8_t slot,
                           const uint16_t src[3], bool uniform0,
                           const bool live[3], const bool discard[3],
                           uint16_t wait_mask, uint8_t out[16], uint8_t *len);

enum agx3_stage_fma_source {
   AGX3_STAGE_FMA_REGISTER,
   AGX3_STAGE_FMA_UNIFORM,
   AGX3_STAGE_FMA_IMMEDIATE,
};

bool agx3_stage_fma(enum agx_isa_encoding enc, uint16_t dest, uint8_t slot,
                    const uint16_t src[3], enum agx3_stage_fma_source source0,
                    const bool live[3], const bool discard[3],
                    uint16_t wait_mask, uint8_t out[16], uint8_t *len);

/* General-register source liveness is independent of staging liveness.
 * Vertex outputs may export the same GPR more than once; do not kill that
 * source merely because each staging value has a single UVSW consumer. */
bool agx3_stage_copy_source(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, bool src_kill, uint8_t slot,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len);

/* Set staging publication f13 independently of GPR last-use f27. Native
 * G17P DEPTHF producers set f13 on both coverage and depth staging words.
 * This must not force the GPR source dead if another instruction uses it. */
bool agx3_stage_copy_liveness(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src, bool stage_kill, bool src_kill,
                              uint8_t slot, uint16_t wait_mask,
                              uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len);

/* staging[dest] <- reg[src] with the operand's last-use bit f13 set. Apple
 * escalates to the ten-byte `copy.ep1` for this; the four-byte compact form
 * cannot carry the bit. */
bool agx3_stage_copy_kill(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, bool kill, uint8_t slot,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len);

/* reg[dest] <- 0. `half` selects the 16-bit form (f10 = 1), which cbits uses;
 * `hint` is the one-bit destination flag f11. */
bool agx3_clr(enum agx_isa_encoding enc, uint16_t dest, bool half, bool hint,
              uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len);

bool agx3_stage_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     uint8_t slot, uint16_t wait_mask,
                     uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
