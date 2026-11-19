/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Writers for the AGX3 *staging* register file.
 *
 * `movmsk`, `depthf`, `uvsw` and the texture coordinate operand do not name a
 * general register. Their operand field is selected by the destination-class
 * selector f9 = 5, and it indexes a separate file that only a small set of
 * instructions can write. agx3_tex_core has carried one such writer since the
 * sampler work (the coordinate `copy`); this header collects the whole family
 * so the coverage/depth writeout path can reach it too.
 *
 * The destination-class selector is shared by every AGX3 form:
 *
 *   f9 = 2  general register, index in f2, plus the one-bit hint f11
 *   f9 = 3  uniform / IOTO destination, index in f3
 *   f9 = 4  thread-parameter-store spill slot, index in f4
 *   f9 = 5  staging register, index in f5, with kill f13 and slot f12
 *
 * A staging write is asynchronous: it takes a scoreboard slot (f12) that the
 * consuming writeout instruction waits on. That is why `copy`, `ldimm.ep2` and
 * the staging `clr` all carry f12 while their general-register siblings do not.
 *
 * Evidence (G17P / AGX3b, offline `applegpu-nt -arch applegpu_g17p`, decoded
 * with the composer that also generates Mesa's own encoder tables):
 *
 *  - `ldimm.ep2` staging form: three distinct witnesses in every Apple vertex
 *    object of apple-g17p-isa.metalar, e.g. `0c 80 23 3e 00 00 00 0c 40 20`
 *    = staging[0] <- 0x3F800000 on slot 2, and the discard constant
 *    `0c 80 23 fe 1c 0c ff 0f 40 00` = staging[0] <- 0xFFFFFF00 on slot 1.
 *  - staging `clr`: `1b 00 20 04` = staging[2] <- 0, slot 1 (vmain).
 *  - general `clr`: `0b 00 00 00` / `1b 00 00 00` / `2b 00 00 00` /
 *    `3b 00 00 00` = r0/r2/r4/r6 <- 0 (uint4(0) store probe).
 *
 * AGX3c has all three forms but no witness, so every entry point here refuses
 * anything but AGX3b rather than guessing the longer encoding's constants.
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
