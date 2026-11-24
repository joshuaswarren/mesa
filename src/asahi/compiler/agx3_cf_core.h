/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 control-flow encoders.
 *
 * AGX3 splits AGX2's fused compare-and-branch. A compare (icmp/fcmp) writes a
 * persistent per-lane flag slot; the exec-mask stack ops (cndst push, cndlt
 * loop/break test, cndef else, cndend pop) consume `flag[slot] ^ invert`; and
 * b/bn are the actual branches, carrying a signed byte displacement patched in
 * after block offsets are known. Conditions are AGX2's own icond/fcond enums.
 *
 * Structured control-flow lowering (flag-slot allocation and branch fixups)
 * lives in the backend.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AGX3 integer compare conditions (== enum agx_icond). le/ge/ne are expressed
 * as the inverse condition plus the consumer's invert bit. */
enum agx3_cf_icond {
   AGX3_CF_ICOND_ULT = 1,
   AGX3_CF_ICOND_UGT = 2,
   AGX3_CF_ICOND_EQ = 4,
   AGX3_CF_ICOND_SLT = 5,
   AGX3_CF_ICOND_SGT = 6,
};

/* AGX3 float compare conditions (== enum agx_fcond base values). With or_nan
 * set, lt/gt become the IEEE ordered-or-unordered variants Apple uses to build
 * ge/le/ne by inversion. */
enum agx3_cf_fcond {
   AGX3_CF_FCOND_EQ = 0,
   AGX3_CF_FCOND_LT = 1,
   AGX3_CF_FCOND_GT = 2,
};

/* Longest control-flow form is 14 bytes (`fcmp.ep2` with abs modifier). */
#define AGX3_CF_MAX_LENGTH    14u
#define AGX3_CF_MAX_FLAG_SLOT 5u

struct agx3_cf_source_hints {
   bool cache;
   bool discard;
};

bool agx3_cf_icmp_width_m5(enum agx3_cf_icond cond, unsigned slot,
                            const uint16_t src[2], const bool half[2],
                            const struct agx3_cf_source_hints hints[2],
                            uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

bool agx3_cf_icmp_reg16(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                        unsigned slot, const uint16_t src[2],
                        const struct agx3_cf_source_hints hints[2], uint16_t wait_mask,
                        uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Compares: write flag[slot] = (src0 cond src1). */
bool agx3_cf_icmp_reg(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                      unsigned slot, uint16_t src0, uint16_t src1,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
/* flag[slot] = (32-bit register src0 cond 32-bit uniform); `uniform` is an
 * even half index below 256. */
bool agx3_cf_icmp_reg_uniform32(enum agx_isa_encoding enc,
                                enum agx3_cf_icond cond, unsigned slot,
                                uint16_t src0, uint16_t uniform,
                                struct agx3_cf_source_hints hint,
                                uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
/* flag[slot] = (32-bit uniform cond imm); `uniform` is an even half index
 * below 256. */
bool agx3_cf_icmp_uniform_imm(enum agx_isa_encoding enc,
                              enum agx3_cf_icond cond, unsigned slot,
                              uint16_t uniform, uint8_t imm,
                              uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
bool agx3_cf_icmp_imm(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                      unsigned slot, uint16_t src0, uint8_t imm,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* 16-bit integer compare against an 8-bit immediate (0..255). Unsigned ordered
 * relations use the compact 6-byte form (no source-lifetime fields).
 */
bool agx3_cf_icmp_imm16_runtime(enum agx_isa_encoding enc,
                                enum agx3_cf_icond cond, unsigned slot,
                                uint16_t src, uint8_t imm,
                                struct agx3_cf_source_hints hint,
                                uint16_t wait_mask,
                                uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* G18G only: agx3_cf_icmp_imm16_runtime() as the G18G layer emits it on
 * AGX_ISA_ENCODING_AGX3C (re-encoded as agx3d afterwards). */
bool
agx3_cf_icmp_imm16_runtime_g18g(enum agx_isa_encoding enc,
                                enum agx3_cf_icond cond, unsigned slot,
                                uint16_t src, uint8_t imm,
                                struct agx3_cf_source_hints hint,
                                uint16_t wait_mask,
                                uint8_t out[AGX3_CF_MAX_LENGTH],
                                uint8_t *len);

bool agx3_cf_fcmp_reg(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                      bool or_nan, unsigned slot, uint16_t src0, uint16_t src1,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* G17P comparison of two half-float registers with their actual lifetimes. */
bool agx3_cf_fcmp_reg16(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                        bool or_nan, unsigned slot, const uint16_t src[2],
                        const struct agx3_cf_source_hints hints[2],
                        uint16_t wait_mask, uint8_t out[AGX3_CF_MAX_LENGTH],
                        uint8_t *len);

bool agx3_cf_fcmp_width_m5(enum agx3_cf_fcond cond, bool or_nan, unsigned slot,
                          const uint16_t src[2], const bool half[2],
                          const bool absolute[2],
                          const struct agx3_cf_source_hints hints[2],
                          uint16_t wait_mask,
                          uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* G17P register comparison with exactly one negated f32 source. */
bool agx3_cf_fcmp_reg_neg(enum agx_isa_encoding enc,
                          enum agx3_cf_fcond cond, bool or_nan, unsigned slot,
                          const uint16_t src[2], const bool neg[2],
                          const struct agx3_cf_source_hints hints[2],
                          uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* G17P register comparison with an absolute value on either f32 source. ABS
 * is a longer encoding point (six bytes -> fourteen, ep0 -> ep2) carrying f14
 * and f31, not a bit on the short form; see agx3_cf_core.c for the ladder. */
bool agx3_cf_fcmp_reg_abs(enum agx_isa_encoding enc,
                          enum agx3_cf_fcond cond, bool or_nan, unsigned slot,
                          const uint16_t src[2], const bool abs[2],
                          const struct agx3_cf_source_hints hints[2],
                          uint16_t wait_mask,
                          uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Float comparison against literal zero, retaining the actual source lifetime. */
bool agx3_cf_fcmp_zero(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                        bool or_nan, unsigned slot, uint16_t source, bool source_16,
                        struct agx3_cf_source_hints hint, uint16_t wait_mask,
                        uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* The same point comparing against a non-zero minifloat constant. f18 is the
 * second operand and used to be pinned to zero, so `if (x < 1.0)` had no
 * encoding at all -- the float twin of the icmp defect above. */
bool agx3_cf_fcmp_zero_imm(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                           bool or_nan, unsigned slot, uint16_t source,
                           bool source_16, uint8_t minifloat,
                           struct agx3_cf_source_hints hint,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Exec-mask stack. cndst pushes (an `if`), with push count 1 or 2 (2 = a loop
 * entry that reserves a nesting level). cndlt is the loop/break test (push
 * count fixed at 2). cndef is the `else` flip. cndend pops `count` levels. */
bool agx3_cf_cndst(enum agx_isa_encoding enc, unsigned slot, bool invert,
                   unsigned push_count, uint8_t out[AGX3_CF_MAX_LENGTH],
                   uint8_t *len);
bool agx3_cf_cndlt(enum agx_isa_encoding enc, unsigned slot, bool invert,
                   uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
bool agx3_cf_cndef(enum agx_isa_encoding enc, uint8_t out[AGX3_CF_MAX_LENGTH],
                   uint8_t *len);

/* cndcf deactivates lanes when (flag[slot] ^ invert) is true until
 * `level_count` exec-mask levels (inside the innermost loop, matching
 * `loop_nesting`) have been popped. Opposite predicate sense to cndlt.
 */
bool agx3_cf_cndcf(enum agx_isa_encoding enc, unsigned slot, bool invert,
                   unsigned level_count, uint8_t out[AGX3_CF_MAX_LENGTH],
                   uint8_t *len);
bool agx3_cf_cndend(enum agx_isa_encoding enc, unsigned pop_count,
                    uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* Branches. Emit with displacement 0 (a reserved 10-byte template), then patch
 * the byte displacement once the target offset is known. is_bn selects
 * branch-if-none-active (forward, an exec-mask skip) vs branch-if-any-active
 * (the backward loop edge). */
bool agx3_cf_branch(enum agx_isa_encoding enc, bool is_bn,
                    uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/* OR the signed byte displacement (target - branch_start, must be even) into a
 * reserved branch template in place. Returns false on an odd or out-of-range
 * displacement. */
bool agx3_cf_patch_branch(enum agx_isa_encoding enc, bool is_bn,
                          uint8_t *branch, int64_t displacement);

#define AGX3_CF_BRANCH_LENGTH 10u

bool agx3_cf_icmp_zero16_runtime(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                                 unsigned slot, uint16_t src,
                                 struct agx3_cf_source_hints hint, uint16_t wait_mask,
                                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
