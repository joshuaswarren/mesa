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
 * Every encoder here is grade A: it reproduces Apple's witnessed instruction
 * bytes exactly via the generated composer. What is NOT here is the lowering --
 * turning IR structured control flow into these sequences with flag-slot
 * allocation and branch fixups -- which lives in the backend.
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

/*
 * The longest control-flow form is fourteen bytes: `fcmp.ep2`, which is what
 * an absolute-valued comparison source moves to.  This was 10 -- the length
 * of `fcmp.ep1` -- so agx3_cf_fcmp_reg_abs() wrote four bytes past the end of
 * every caller's buffer, all of which are sized by this macro.  Nothing
 * caught it because the overrun lands in stack padding on the paths that
 * reach it; building a caller with a stack protector aborts on the spot.
 */
#define AGX3_CF_MAX_LENGTH 14u
/* `b`/`bn`, whose displacement agx3_cf_patch_branch() fills in later. This is
 * the size a caller must RESERVE for a forward branch, which is not the same
 * number as the buffer bound above. */
#define AGX3_CF_MAX_FLAG_SLOT 5u /* slots 0..5 are witnessed */

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
/* flag[slot] = (32-bit register src0 cond 32-bit uniform), the form Apple's
 * G17P and G17G loop tests use. `uniform` is an even half index below 256. */
bool agx3_cf_icmp_reg_uniform32(enum agx_isa_encoding enc,
                                enum agx3_cf_icond cond, unsigned slot,
                                uint16_t src0, uint16_t uniform,
                                struct agx3_cf_source_hints hint,
                                uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
/* flag[slot] = (32-bit uniform cond imm); `uniform` is an even half index
 * below 256. Apple's G17P/G17G loop-guard form. */
bool agx3_cf_icmp_uniform_imm(enum agx_isa_encoding enc,
                              enum agx3_cf_icond cond, unsigned slot,
                              uint16_t uniform, uint8_t imm,
                              uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);
bool agx3_cf_icmp_imm(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                      unsigned slot, uint16_t src0, uint8_t imm,
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len);

/*
 * The same encoding point with a non-zero immediate.  `int n; if (n < 2)` is
 * ordinary GLSL -- Hyprland's border, shadow and inner-glow shaders all reach
 * the packer that way -- and Mesa keeps a small int in a 16-bit half, so it
 * arrives here rather than at agx3_cf_icmp_imm().
 *
 * Witnessed with `metal-nt -arch applegpu_g17p` against the zero form as the
 * control, one kernel per immediate and nothing else changed.  Every witness
 * is the same ten-byte instruction with only f16 different, which is the
 * field agx3_cf_icmp_imm() already uses for the 32-bit source form:
 *
 *   short x;  if (x < 0)    0a 04 23 80 02 00 07 c2 02 00   (== the zero form)
 *   short x;  if (x < 2)    0a 04 23 82 02 00 07 c2 02 00
 *   short x;  if (x < 17)   0a 04 23 90 02 00 17 c2 02 00
 *   short x;  if (x < 255)  0a 04 33 fe 02 00 17 c2 02 00
 *   short x;  if (x > 5)    0a 04 23 84 02 00 16 c2 02 00
 *   short x;  if (x == 7)   0a 04 23 86 06 00 17 c2 02 00
 *   ushort x; if (x == 2)   0a 04 23 82 06 00 07 c2 02 00
 *
 * The two unsigned ordered relations use the compact six-byte encoding point.
 * Fresh dead/live witnesses for `ushort x < 4` are both
 * 0a 04 22 84 05 c2, so that form has no source-lifetime fields. At 256 and
 * above Apple changes the instruction, so the immediate is capped at 255.
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

/*
 * cndcf selects lanes to leave the current construct when flag[slot] XOR
 * invert is true. They stay inactive until `level_count` architectural
 * exec-mask levels have been popped. CNDLT instead retains lanes whose
 * predicate is true. These forms therefore require opposite predicate senses
 * when lowering the same BREAK_IF. A loop owns two levels; deeper breaks use
 * CNDCF's sixteen-bit level count.
 *
 * Witnessed with `metal-nt -arch applegpu_g17p` from MSL that breaks out of
 * a loop from inside 2..7 nested ifs, each condition reading a distinct
 * `device const uint *` element so the Metal compiler cannot flatten them.
 * Apple's bytes, and ours, byte for byte:
 *
 *   ifs=2  cndcf slot1 count3   8f 05 54 03 00 01
 *   ifs=3  cndcf slot1 count4   8f 05 54 04 00 01
 *   ifs=4  cndcf slot1 count5   8f 05 54 05 00 01
 *   ifs=5  cndcf slot1 count6   8f 05 54 06 00 01
 *   ifs=6  cndcf slot1 count7   8f 05 54 07 00 01
 *   ifs=7  cndcf slot1 count8   8f 05 54 08 00 01
 *   break in an else arm        8f 05 54 03 00 09   (invert = f0, bit 43)
 *   break in an INNER loop      8f 05 54 03 00 02   (slot 2, count still 3)
 *
 * The count is the number of levels open *inside the innermost loop*, loop
 * included: an outer loop's levels do not raise it, which is exactly how
 * Mesa's per-loop `loop_nesting` counter is defined. Negating the innermost
 * condition changes the compare and leaves these bytes identical, so the
 * break's sense lives in the compare, not in the count.
 *
 * The AGX3c constants mirror the sibling cnd encoders here (selector f3 = 28
 * from AGX3c.xml's `f3eq28` predicate, as cndlt uses 31 and cndst 33); AGX3c
 * parts have no captured witness for this operation.
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
