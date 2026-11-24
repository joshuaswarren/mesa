/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 `isel` (fused integer compare-and-select, AGX_OPCODE_ICMPSEL).
 *
 * Condition field f106 matches `enum agx_icond` (ult=1, ugt=2, eq=4, slt=5,
 * sgt=6; le/ge/ne swap the selected operands, condition 3 is refused).
 * Selected operands may independently be registers or 8-bit immediates;
 * scoreboard waits use f98..f103 on AGX3B (compact form supports a single slot
 * or slots 1+2). Destination cache is f12 on AGX3B and f11 on AGX3C.
 * Note: GPU0 forms are probe-verified; the separate AGX3C iseli32 encoder is
 * derived from the AGX3C tables.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_ISEL_MAX_LENGTH 14u

/* == enum agx_icond. Only the five Apple emits are encodable. */
enum agx3_isel_icond {
   AGX3_ISEL_ICOND_ULT = 1,
   AGX3_ISEL_ICOND_UGT = 2,
   AGX3_ISEL_ICOND_EQ = 4,
   AGX3_ISEL_ICOND_SLT = 5,
   AGX3_ISEL_ICOND_SGT = 6,
};

enum agx3_isel_operand {
   AGX3_ISEL_REGISTER = 0,
   AGX3_ISEL_IMMEDIATE = 1,
   /* An aligned 32-bit uniform (f66/f83 with class f72/f89 = 3). Only
    * encoded through the explicit-hint path, beside register comparands. */
   AGX3_ISEL_UNIFORM = 2,
};

struct agx3_isel_desc {
   uint16_t dest;      /* half-register units */
   bool dest_16;       /* 16-bit result, including odd destination halves */
   uint16_t cmp0;      /* register half, compared first */
   uint16_t cmp1;      /* register half or unsigned8 immediate */
   enum agx3_isel_icond cond;

   /* Native unsigned register-versus-u8 comparison. The four register
    * hints follow Mesa source order: both comparands, true, false; immediate
    * selected operands have no hint. Existing register/register forms retain
    * their separate native encoding. */
   bool cmp1_immediate;
   /* An aligned 32-bit uniform comparand (cmp0: f36 with f42 = 3; cmp1: f51
    * with f57 = 3), witnessed only against a 32-bit register comparand and
    * register/immediate selects. */
   bool cmp0_uniform;
   bool cmp1_uniform;
   bool cmp_16;

   /* G18G only (agx3_isel_encode_g18g): GPU1 widens a half-register second
    * comparand against a full-register first comparand inside the select:
    * f39 takes the immediate/narrow value and f43 selects sign extension.
    * Witnessed on applegpu_g17g from `uint >= uint(ushort)` (zero extension)
    * and `int < int(short)` (sign extension); Apple puts the wide comparand
    * first and reverses the condition when the source had them the other
    * way round. agx3_isel_encode refuses both. */
   bool cmp1_16;
   bool cmp1_sign_extend;
   bool cmp_mixed_width; /* M5: the second register has the opposite width. */
   bool explicit_hints;
   /* The result is consumed through the register cache: f12 on GPU0, f11 on
    * GPU1. */
   bool dest_cache;
   bool cache[4];
   bool discard[4];

   /* The selected operands: each a 32-bit register half or an 8-bit
    * immediate. sel_true is taken when (cmp0 cond cmp1) holds. */
   enum agx3_isel_operand true_kind;
   enum agx3_isel_operand false_kind;
   uint16_t sel_true;
   uint16_t sel_false;

   /* Bit i waits for scoreboard slot i + 1, as elsewhere in this backend. */
   uint16_t wait_mask;
};

bool agx3_isel_encode(enum agx_isa_encoding encoding,
                      const struct agx3_isel_desc *desc,
                      uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len);

/* G18G (M6) compiles as agx3c: the GPU1 integer select with cmp1_16 and
 * cmp1_sign_extend; uniform and cmp_mixed_width descriptors and every other
 * encoding go through agx3_isel_encode. */
bool agx3_isel_encode_g18g(enum agx_isa_encoding encoding,
                           const struct agx3_isel_desc *desc,
                           uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len);

/* G17P condition 3 is a nonzero mask test. Fresh mask1/2/4 witnesses pin
 * f49 and f106; selected-negation contrasts pin f78/f95. The quadrant
 * lowering uses these fused selects, not a branch or a separate bit test. */
struct agx3_bittest_desc {
   uint16_t dest;
   uint16_t src[3]; /* bits, true float, false float */
   uint8_t mask;
   bool negate[2];
   bool cache[3];
   bool discard[3];
   uint16_t wait_mask;
};

bool agx3_bittest_encode(enum agx_isa_encoding encoding,
                        const struct agx3_bittest_desc *desc,
                        uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
