/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 `copy` register-file and uniform-file move encoders.
 *
 * Operand kind selectors use 1=immediate, 2=GPR, 3=uniform (slot A: f15/f16/
 * f17; slot B: f31/f32). Unsupported widths, modifiers, or wait combinations
 * fail closed.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Ordinary COPY is four bytes. Extended forms need up to ten bytes. */
#define AGX3_COPY_MAX_LENGTH 10u

/* The compact COPY's destination field (f2) is seven bits wide at EVERY
 * width, so no four-byte copy can name a destination at or above this half
 * index. G17P half copies refuse there; G17S uses its eight-byte ep1 half
 * copy, and the 32-bit uniform form uses ten-byte ep1. Register allocation
 * uses this boundary for G17P half-copy restrictions. The SOURCE fields are not
 * narrow this way: Apple's own ep0 copies carry sources well past it. */
#define AGX3_COPY_REG16_DEST_HALVES 128u

/*
 * Encode `copy dest32, uniform32` (half-register indices). Destinations below
 * half 128 use compact ep0 (7-bit f2); half 128+ use extended ep1.
 */
/*
 * Encode `copy dest16, src16` (ep0 below half 128, ep1 from half 128 to keep
 * the high destination bit), with scoreboard waits on f30 + (slot - 1).
 */
bool agx3_copy_encode_reg16(enum agx_isa_encoding encoding, uint16_t dest_half,
                            uint16_t src_half, bool dest_cache, bool src_cache,
                            bool src_discard, uint16_t wait_mask,
                            uint8_t out[AGX3_COPY_MAX_LENGTH], uint8_t *len);

bool agx3_copy_encode_reg32(enum agx_isa_encoding encoding, uint16_t dest_half,
                            uint16_t src_half, bool dest_cache, bool src_cache,
                            bool src_discard, uint16_t wait_mask,
                            uint8_t out[AGX3_COPY_MAX_LENGTH], uint8_t *len);

/*
 * Encode `copy uniform, reg32` (preamble `uniform_store`, f9=3, dest in f3):
 * 4-byte ep0 through half 126, 10-byte ep1 above. `kill` drives source
 * last-use bit f27 and must reflect actual liveness.
 */
bool agx3_copy_encode_reg_to_uniform(enum agx_isa_encoding encoding,
                                     uint16_t uniform_half, uint16_t src_half,
                                     bool kill, uint16_t wait_mask,
                                     uint8_t out[AGX3_COPY_MAX_LENGTH],
                                     uint8_t *len);

/*
 * G18G only (compiled with AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d
 * afterwards): the uniform write as the G18G layer emits it -- always the
 * eight-byte ep1 with f42 set, carrying its own scoreboard wait in
 * f30 + slot - 1. The other GPUs on the G17G-family encoding (agx3c) keep
 * agx3_copy_encode_reg_to_uniform().
 */
bool agx3_copy_encode_reg_to_uniform_g18g(enum agx_isa_encoding encoding,
                                          uint16_t uniform_half,
                                          uint16_t src_half, bool kill,
                                          uint16_t wait_mask,
                                          uint8_t out[AGX3_COPY_MAX_LENGTH],
                                          uint8_t *len);

bool agx3_copy_encode_uniform_to_reg32(enum agx_isa_encoding encoding,
                                       uint16_t dest_half,
                                       uint16_t uniform_half,
                                       uint8_t out[AGX3_COPY_MAX_LENGTH],
                                       uint8_t *len);

/*
 * Encode `copy dest16, uniform16` (f9=2, f10=1, f23=3, f24=1, f17=uniform).
 * Carries no wait bits; callers must retire pending producers before issuing.
 */
bool agx3_copy_encode_uniform_to_reg16(enum agx_isa_encoding encoding,
                                       uint16_t dest_half,
                                       uint16_t uniform_half, bool dest_cache,
                                       uint8_t out[AGX3_COPY_MAX_LENGTH],
                                       uint8_t *len);

#ifdef __cplusplus
}
#endif
