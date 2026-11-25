/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 f32 ALU with source/output modifiers (abs, neg, saturate).
 *
 * The grade-A no-modifier fadd/fmul/ffma are the compact hand-written packers.
 * When a modifier is present the compact forms cannot express it, so these
 * encoders emit the always-capable long form through the composer: both
 * commutative source slots (f35, f18) carry independent abs/neg bits, the ffma
 * addend (f52) carries its own, and f14 saturates the result. There is no slot
 * swap in those legacy interfaces. The runtime-aware ADD additionally tries
 * the equivalent commuted order, moving both modifiers and lifetime hints,
 * to keep subtraction in the shortest native form.
 *
 * All 96 witnessed Apple encodings are reproduced byte-exactly from the
 * baseline, the operands and the modifiers.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Operand widths (f43 for f35, f26 for f18, f60 for f52) compose orthogonally
 * with source modifiers (abs/neg) on the same encoding point.
 */
struct agx3_falu_mods {
   bool src0_abs; /* f47 on the f35 slot */
   bool src0_neg; /* f48 */
   bool src0_16;  /* f43 */
   bool src1_abs; /* f30 on the f18 slot */
   bool src1_neg; /* f31 */
   bool src1_16;  /* f26 */
   bool addend_abs; /* f64, ffma only */
   bool addend_neg; /* f65, ffma only */
   bool addend_16;  /* f60, ffma only */
   bool saturate;   /* f14 */
};

#define AGX3_FALU_MOD_MAX_LENGTH 16u

/* dest = sat?(srcmods(src0) OP srcmods(src1)). abs applies before neg. */
bool agx3_falu_add_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                       uint16_t src1, const struct agx3_falu_mods *m,
                       uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);
bool agx3_falu_mul_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                       uint16_t src1, const struct agx3_falu_mods *m,
                       uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);
/* dest = sat?(srcmods(src0) * srcmods(src1) + addendmods(addend)). */
bool agx3_falu_fma_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                       uint16_t src1, uint16_t addend,
                       const struct agx3_falu_mods *m,
                       uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

/* Actual post-RA hints, in API source order; no specimen lifetime defaults. */
struct agx3_falu_runtime {
   bool cache[3];
   bool discard[3];
   bool dest_cache;
   uint8_t wait_mask;
};

bool agx3_falu_add_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src0, uint16_t src1,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);

/* G17G only: dest = srcmods(src0) + minifloat (f25=1, constant in f17; used
 * for fmov.abs/neg via `fadd mod(x), -0.0`). Saturate is refused. */
bool agx3_falu_add_imm_runtime(uint16_t dest, uint16_t src0,
                               uint8_t minifloat,
                               const struct agx3_falu_mods *m,
                               const struct agx3_falu_runtime *runtime,
                               uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                               uint8_t *len);

/* G17G only: dest = srcmods(src0) * minifloat, the long f32mul with the
 * immediate-operand alternative. Apple folds a negated source into the
 * constant's sign, so only the absolute value is a source modifier here
 * (gpu/m5-xfb-alu4-reference.metal, -alu6-):
 *
 *   a * 2.0          29 00 05 10 22 80 ac 02 0b 00 20 00 00 10   f17 = 0x40
 *   fabs(a) * 2.0    29 00 05 10 22 80 ac 02 0b 00 20 08 00 10   f47
 *   -fabs(a) * 2.0   29 00 05 10 22 80 ad 02 0b 00 20 08 00 10   f17 = 0xc0
 *
 * all with a dead source waiting for the load in slot 8. */
bool agx3_falu_mul_imm_runtime(uint16_t dest, uint16_t src0,
                               uint8_t minifloat,
                               const struct agx3_falu_mods *m,
                               const struct agx3_falu_runtime *runtime,
                               uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                               uint8_t *len);

bool agx3_falu_mul_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src0, uint16_t src1,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);

/* f32 to f16 conversion, including odd destination halves. Runtime source0
 * is the sole real operand; the other input is exact negative zero. */
bool agx3_falu_narrow_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                         bool saturate, const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

/* minifloat - src, narrowed to a half. Runtime source0 describes src. */
bool agx3_falu_narrow_imm_sub_f32(enum agx_isa_encoding enc, uint16_t dest,
                                 uint8_t minifloat, uint16_t src,
                                 const struct agx3_falu_runtime *runtime,
                                 uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                                 uint8_t *len);

bool agx3_falu_fma_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t src0, uint16_t src1, uint16_t addend,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);

/* Native add/multiply with a single rounding to a scalar half result. */
bool agx3_falu_add_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
                            const struct agx3_falu_mods *mods,
                            const struct agx3_falu_runtime *runtime,
                            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);
bool agx3_falu_mul_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
                            const struct agx3_falu_mods *mods,
                            const struct agx3_falu_runtime *runtime,
                            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

/* Native F32 arithmetic with one final rounding directly to a half result. */
bool agx3_falu_fma_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
                            uint16_t addend, const struct agx3_falu_mods *m,
                            const struct agx3_falu_runtime *runtime,
                            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

/* C FMA32 also consumes half GPRs with an immediate addend. Preserve the
 * independent widths, modifiers, lifetime and saturation in one instruction. */
bool agx3_falu_fma_addend_imm_width_mod(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src0, uint16_t src1,
   uint8_t immediate, const struct agx3_falu_mods *m,
   const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_addend_imm_mod(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src0, uint16_t src1,
   uint8_t immediate, bool src0_abs, bool src0_neg, bool src1_abs,
   bool src1_neg,
   const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_two_imm(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, uint8_t multiplier, uint8_t addend,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_two_imm_narrow_m5(
   uint16_t dest, uint16_t src, uint8_t multiplier, uint8_t addend,
   bool src_abs, bool saturate, const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

/*
 * FMA with a minifloat immediate multiplier (f42=1, f34=imm) and register
 * multiplicand/addend carrying independent widths and modifiers. `m->src0_*`
 * are rejected; fold sign into `multiplier` directly.
 */
bool agx3_falu_fma_mul_imm_mod(enum agx_isa_encoding enc, uint16_t dest,
                              uint8_t multiplier, uint16_t src1,
                              uint16_t addend,
                              const struct agx3_falu_mods *m,
                              const struct agx3_falu_runtime *runtime,
                              uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                              uint8_t *len);
/* The same G17G form narrowing its result into a scalar half (f10=1). */
bool agx3_falu_fma_mul_imm_narrow_m5(uint16_t dest, uint8_t multiplier,
                                     uint16_t src1, uint16_t addend,
                                     const struct agx3_falu_mods *m,
                                     const struct agx3_falu_runtime *runtime,
                                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                                     uint8_t *len);

bool agx3_falu_add_imm_m5(uint16_t dest, uint16_t src, uint8_t immediate,
                         bool src_abs, bool src_neg, bool saturate,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_add_imm_narrow_m5(uint16_t dest, uint16_t src, uint8_t immediate,
   bool src_abs, bool src_neg, bool saturate,
   const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);
bool agx3_falu_fma_addend_imm_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
   uint8_t immediate, const struct agx3_falu_mods *mods,
   const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

bool agx3_falu_fma_addend_imm(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src0, uint16_t src1, uint8_t immediate,
                             const struct agx3_falu_runtime *runtime,
                             uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
