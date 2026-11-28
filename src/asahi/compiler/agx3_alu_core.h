/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Composer-driven AGX3 ALU encoder that carries scoreboard wait bits.
 *
 * The compact hand packers (agx3_select_core) cannot express a wait mask, but an
 * ALU instruction that consumes an async load result must wait its slot. This
 * encodes the same grade-A ALU forms through the composer so the per-mnemonic
 * wait-bit fields can be set. With wait_mask == 0 it reproduces the hand packers
 * byte-for-byte; the backend uses the compact packer when no wait is needed and
 * this when one is.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"
#include "agx3_alu_table.h"
#include "agx3_funop_core.h"
#ifdef __cplusplus
extern "C" {
#endif
#define AGX3_ALU_MAX_LENGTH 16u
bool agx3_alu_compose(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      uint16_t dest, uint16_t src0, uint16_t src1,
                      uint16_t src2, uint32_t imm, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Post-RA source hints. These are distinct from the SSA kill flag. */
enum agx3_alu_hint {
   AGX3_ALU_HINT_NONE = 0,
   AGX3_ALU_HINT_DISCARD = 1,
   AGX3_ALU_HINT_CACHE = 2,
};

/* Native C scalar rounding/log/exp with independent 16/32-bit widths,
 * source lifetime, modifiers, saturation and producer waits. */
bool agx3_alu_unary_m5(enum agx3_funop op, uint16_t dest, uint16_t src,
                       bool dest_half, bool src_half, bool src_abs,
                       bool src_neg, bool saturate, enum agx3_alu_hint hint,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native C fragment gradients (f32dfdx/f32dfdy) over the whole register
 * file, at 32 or 16 bits. */
bool agx3_alu_derivative_m5(bool dfdy, uint16_t dest, uint16_t src, bool half,
                            bool src_abs, bool src_neg, bool saturate,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native G17P population count of a scalar ushort or uint, with post-RA
 * source lifetime and producer waits. */
/* Native M5 bit reversal. A half source is zero-extended before reversing
 * all 32 bits, so its reversed bits occupy the high half of the result. */
bool agx3_alu_bitrev_m5(uint16_t dest, uint16_t src, bool src_half,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *len);

bool agx3_alu_popcount(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       bool half, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G18G (M6): the agx3c cbs form qualified on G18G: a live source is
 * cached and only the slot-1 and slot-8 waits are admitted. */
bool agx3_alu_popcount_g18g(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                            bool half, enum agx3_alu_hint hint, bool dest_cache,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Whether agx3_alu_compose()'s form for `op` marks source `i` (in
 * f_src0/f_src1/f_src2 order) dead. */
bool agx3_alu_form_ends_source(enum agx_isa_encoding enc, enum agx3_alu_op op,
                               unsigned i);

/* G18G (M6): also the agx3c RCP/I2F/U2F/F2I specimens. */
bool agx3_alu_form_ends_source_g18g(enum agx_isa_encoding enc, enum agx3_alu_op op,
                                    unsigned i);

/* G17P 32-bit AND/OR/XOR with actual post-RA operand hints. */
bool agx3_alu_bit_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      uint16_t dest, const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Opt-in caller: native compact/full forms with destination caching. */
bool agx3_alu_bit_reg_native(enum agx_isa_encoding enc, enum agx3_alu_op op,
                             uint16_t dest, const uint16_t src[2],
                             const enum agx3_alu_hint hints[2], bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* `reg & reg` into reg: a no-op carrying a scoreboard wait mask, which keeps
 * the register live whatever it holds. */
bool agx3_alu_band_identity_wait(enum agx_isa_encoding enc, uint16_t reg,
                                 uint16_t wait_mask,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH],
                                 uint8_t *len);

/* The AGX3c form of the same carrier that does keep `reg` live: both sources
 * carry Apple's live-source (cached) hint instead of the composer specimen's
 * last-use bit. */
bool agx3_alu_band_identity_live_gpu1(uint16_t reg, uint16_t wait_mask,
                                      uint8_t out[AGX3_ALU_MAX_LENGTH],
                                      uint8_t *len);

bool agx3_alu_bit_reg16(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_and_imm16_m5(uint16_t dest, uint16_t src, uint8_t immediate,
                         bool complement, bool immediate_first,
                         enum agx3_alu_hint hint,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_complement_reg16_m5(uint16_t dest, const uint16_t src[2],
                                const enum agx3_alu_hint hints[2],
                                bool disjunction, bool dest_cache,
                                uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * Bitwise AND/OR/XOR with an immediate second operand (f37=1, f29=imm).
 * A 16-bit result narrows f10 and f24, while f38 stays 2 for the immediate.
 */
bool agx3_alu_bit_imm(enum agx_isa_encoding enc, enum agx3_alu_op op,
                      bool narrow, uint16_t dest, uint16_t src,
                      uint8_t immediate, enum agx3_alu_hint hint,
                      uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 16-bit unary bitwise NOT (`inv`, width 1): dest cache f11, src f16
 * (cache/discard f25/f27), and producer slot waits f30..f35. */
bool agx3_alu_bit_not_reg16(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, enum agx3_alu_hint hint,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* AGX3C `inv` on a half or a 32-bit register. */
bool agx3_alu_bit_not_m5(uint16_t dest, uint16_t src, bool word,
                         enum agx3_alu_hint hint, bool dest_cache,
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 32-bit a & ~b.  Apple exposes this truth-table shape as bit_andc;
 * unlike AND it is not commutative, so the source order is preserved. */
bool agx3_alu_bit_andc(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

/* G17P 32-bit ~(a | b), Apple's bit_nor. */
bool agx3_alu_bit_nor(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

/* G17P 32-bit ~(a & b), ~(a ^ b) and a | ~b: Apple's bit_nand, bit_xnor and
 * bit_orc. bit_orc keeps its source order. */
bool agx3_alu_bit_nand(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2],
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);
bool agx3_alu_bit_xnor(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2],
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);
bool agx3_alu_bit_orc(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2],
                      const enum agx3_alu_hint hints[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

bool agx3_alu_shift_add16(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2], unsigned shift,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half saturating register addition, signed or unsigned. */
bool agx3_alu_add_sat16(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2],
                        const enum agx3_alu_hint hints[2], bool is_signed,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* AGX3C saturating 16-bit add/subtract (iadd32/isub32 with a half result). */
bool agx3_alu_addsub_sat16_m5(uint16_t dest, const uint16_t src[2],
                              const enum agx3_alu_hint hints[2],
                              bool is_signed, bool subtract, bool dest_cache,
                              uint16_t wait_mask,
                              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half register subtraction, src[0] - src[1]. */
bool agx3_alu_sub16(enum agx_isa_encoding enc, uint16_t dest,
                    const uint16_t src[2], const enum agx3_alu_hint hints[2],
                    bool dest_cache, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half integer negation, 0 - src. */
bool agx3_alu_neg16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P logical shift-left with register value and register shift count. */
bool agx3_alu_sll_reg(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                      bool dest_cache, const uint16_t src[2],
                      const bool src_16[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G18G (M6): the all-half agx3c register shift with the count's
 * cache hint. */
bool agx3_alu_sll_reg_g18g(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                           bool dest_cache, const uint16_t src[2],
                           const bool src_16[2],
                           const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half `1 << uint_register` form. */
bool agx3_alu_sll_one_reg32(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t shift, enum agx3_alu_hint shift_hint,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* AGX3C `1 << uint_register`: sll with immediate one and a 32-bit count, to
 * a half or 32-bit destination. */
bool agx3_alu_sll_one_m5(uint16_t dest, bool dest_16, uint16_t shift,
                         enum agx3_alu_hint shift_hint, bool dest_cache,
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P logical shift-right with register value and signed boundary. */
bool agx3_alu_srl_reg(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                      bool dest_cache, const uint16_t src[2],
                      const bool src_16[2],
                      const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G18G (M6): the all-half agx3c register shift with the count's
 * cache hint. */
bool agx3_alu_srl_reg_g18g(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                           bool dest_cache, const uint16_t src[2],
                           const bool src_16[2],
                           const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native signed 16-bit arithmetic shift-right by a 32-bit register count. */
bool agx3_alu_asr_reg16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t source, uint16_t shift,
                        const enum agx3_alu_hint hints[2], bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P 32-bit logical shift with an immediate count, avoiding a count GPR. */
bool agx3_alu_shift_imm32(enum agx_isa_encoding enc, bool left,
                          uint16_t dest, uint16_t src, unsigned shift,
                          enum agx3_alu_hint hint, bool dest_cache,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native 16-bit logical shift-right by an immediate bit count. */
bool agx3_alu_srl_imm16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, unsigned shift,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* The same, with slot B shifted left by `shift` first (f54). Witnessed for
 * ISUB at shifts 1..4; see agx3_alu_core.c for the byte ladder. */
bool agx3_alu_integer_reg_shift_cache(enum agx_isa_encoding enc,
                                      enum agx3_alu_op op, uint16_t dest,
                                      const uint16_t src[2],
                                      const enum agx3_alu_hint hints[2],
                                      uint8_t shift, bool dest_cache,
                                      uint16_t wait_mask,
                                      uint8_t out[AGX3_ALU_MAX_LENGTH],
                                      uint8_t *len);

/* Native signed/unsigned32 to float32 with post-RA source hints. */
bool agx3_alu_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, bool is_signed,
                            enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native signed/unsigned f32-to-i32 with post-RA source and destination cache
 * state. */
bool agx3_alu_float_to_int32(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, bool is_signed,
                             enum agx3_alu_hint hint, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native int8 or int16 to scalar-half or float32 conversion. A byte source
 * reads only the low eight bits of its half; unsigned8 converts only to
 * float32. */
bool agx3_alu_narrow_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                                  bool dest_16, uint16_t src, bool src_8,
                                  bool is_signed, enum agx3_alu_hint hint,
                                  bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

/* Native signed/unsigned32 to scalar-half conversion. */
bool agx3_alu_i32_to_f16(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, bool is_signed,
                         enum agx3_alu_hint hint, bool dest_cache,
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar float16 to signed/unsigned32 conversion, RTZ. */
bool agx3_alu_f16_to_int32(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, bool is_signed,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32 comparison with byte-sized integer alternatives and a16/32-bit result. */
bool agx3_alu_fcmp_select_zero(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[3], bool compare_16,
                             const enum agx3_alu_hint hints[3], unsigned condition,
                             bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_select_imm_width(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[2], const bool src_16[2],
                             const enum agx3_alu_hint hints[2],
                             unsigned condition, uint8_t if_true,
                             uint8_t if_false, bool dest_16, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * General four-operand `fcmpsel` (each operand carries an independent kind,
 * width, and value; kind 1=imm, 2=GPR, 3=uniform). Absolute compare sources
 * select a longer endpoint; GPR alternatives carry cache/discard hints
 * (f78/f80, f95/f97) while uniform alternatives carry none.
 */
enum agx3_fsel_kind {
   /* compare slots */
   AGX3_FSEL_CMP_IMMEDIATE = 1,
   AGX3_FSEL_CMP_REGISTER = 2,
   AGX3_FSEL_CMP_UNIFORM = 3,
   /* select slots */
   AGX3_FSEL_SEL_IMMEDIATE = 1,
   AGX3_FSEL_SEL_REGISTER = 2,
   AGX3_FSEL_SEL_UNIFORM = 3,
};

struct agx3_fsel_desc {
   uint16_t dest;
   bool dest_16, dest_cache;

   uint16_t cmp0;               /* register half, or a uniform half */
   bool cmp0_uniform;           /* f42 = 3, value in f36, no hints */
   enum agx3_alu_hint cmp0_hint;
   bool cmp0_abs;
   bool cmp0_neg;
   bool cmp0_16;

   bool cmp1_is_immediate;
   bool cmp1_uniform;           /* f59 = 3, value in f53, width f60, no hints */
   uint16_t cmp1;               /* register half, uniform half, or the immediate value */
   enum agx3_alu_hint cmp1_hint;
   bool cmp1_abs;
   bool cmp1_neg;
   bool cmp1_16;                /* register sources only; an immediate slot
                                 * always declares width 2 */

   uint8_t sel0_kind, sel1_kind; /* agx3_fsel_kind, select-slot numbering */
   uint16_t sel0, sel1;
   enum agx3_alu_hint sel0_hint, sel1_hint;
   bool sel_16;

   unsigned condition;
   uint16_t wait_mask;
};

bool agx3_alu_fcmp_select(enum agx_isa_encoding enc,
                          const struct agx3_fsel_desc *d,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native terrain's fmax(a, fabs(b)), full-width register operands only. */
bool agx3_alu_fmax_abs_f32(enum agx_isa_encoding enc, uint16_t dest,
                         const uint16_t src[2],
                         const enum agx3_alu_hint hints[2], bool dest_cache,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* The same select with an absolute value on either comparison source. ABS is
 * not a bit on the form above: it moves to a longer endpoint (ten bytes ->
 * fourteen) whose integer-select destination block is the same f3/f11/f12.
 * f47 and f64 are the two sources' bits and are separable. See
 * agx3_alu_core.c for the witnessed byte ladder. */
bool agx3_alu_fcmp_select_imm_width_abs(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   unsigned condition, uint8_t if_true, uint8_t if_false, bool dest_16,
   bool dest_cache, uint16_t wait_mask, const bool src_abs[2],
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* The same, with the negate on either comparator. There is no bit for a
 * negated FIRST source: Apple spells `-a > b` as `b < -a`, so this swaps the
 * operands and reverses the relation rather than refusing. Negating both is
 * the relation flip alone. */
bool agx3_alu_fcmp_select_imm_width_neg(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   const bool neg[2], unsigned condition, uint8_t if_true, uint8_t if_false,
   bool dest_16, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P f32-register == minifloat comparison selecting a half register or
 * immediate.  This is the exact vectorized native form used for half masks. */
bool agx3_alu_fcmp_minifloat_select_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, uint16_t if_true, uint8_t if_false,
   enum agx3_alu_hint compare_hint, enum agx3_alu_hint true_hint,
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Compare f32 against a minifloat and select the observed mixed register /
 * byte-immediate result shapes used by Hyprland and GTK shaders. The
 * half-result form carries any one of the six native scoreboard waits. */
bool agx3_alu_fcmp_minifloat_select_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_16, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Compare a scalar half against a minifloat and select f32 values, with at
 * most one byte-immediate alternative. */
bool agx3_alu_fcmp_half_minifloat_select_f32(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Plain f32 register FADD/FMUL/FMA with actual post-RA source hints and
 * scheduler waits. Compact witnessed forms only; unsupported wait combinations
 * return false so the caller can retire them explicitly before retrying. */
bool agx3_alu_float_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                         uint16_t dest, const uint16_t src[3],
                         const enum agx3_alu_hint hints[3], uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Write a scalar f32 add directly to the texture staging file.  Native
 * fragment shaders use this destination class for the last coordinate
 * arithmetic before an implicit sample; a later GPR-to-staging COPY loses
 * helper-lane derivative provenance at an odd hardware-scissor edge. */
bool agx3_alu_fadd_stage(enum agx_isa_encoding enc, uint16_t staging_dest,
                         uint8_t staging_slot, const uint16_t src[2],
                         const enum agx3_alu_hint hints[2],
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32 FMA with a minifloat multiplier and an optionally negated f32
 * register addend. The addend modifier remains fused and rounds only once. */
bool agx3_alu_fma_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t src, uint8_t immediate,
                                uint16_t addend, bool negate_addend, bool saturate,
                                enum agx3_alu_hint src_hint,
                                enum agx3_alu_hint addend_hint,
                                bool dest_cache, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH],
                                uint8_t *len);

bool agx3_alu_mov16(enum agx_isa_encoding enc, uint16_t dest, uint32_t bits,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* G17G half-register forms, witnessed byte-exact by Apple's own compiler
 * (gpu/m5-shift-add16-reference.metal, gpu/m5-fragment-zs-constant-
 * reference.metal). Waits are the eight-slot mask, bit i for slot i + 1. */
bool agx3_alu_mov16_gpu1(uint16_t dest, bool dest_cache, uint32_t bits,
                         uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                         uint8_t *len);
bool agx3_alu_fmul_mixed(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t f32, uint16_t f16,
                          enum agx3_alu_hint hint32, enum agx3_alu_hint hint16,
                          bool dest_cache, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native ABS(f32)-by-scalar-half multiply with an f32 result. */
bool agx3_alu_fmul_abs_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native unmodified f32 plus scalar-half register. */
bool agx3_alu_fadd_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32 ABS(register) plus scalar-half register. */
bool agx3_alu_fadd_abs_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native f32-by-f16 multiply rounded directly to a scalar half destination. */
bool agx3_alu_fmul_mixed_half(enum agx_isa_encoding enc, uint16_t dest,
                              uint16_t f32, uint16_t f16,
                              enum agx3_alu_hint hint32,
                              enum agx3_alu_hint hint16, bool dest_cache,
                              uint16_t wait_mask,
                              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P's unsigned16 GPR half -> float32, RTE. Every U16 is exactly
 * representable, so no precision is discarded. Native compiler witnesses
 * include both last-use and reused input, odd half indices and load waits.
 * This does not admit signed, byte, float16, vector, uniform or AGX3c forms.
 */
bool agx3_alu_s16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* RTZ float-to-16-bit-integer conversion. The source width (f24) and the
 * signed/unsigned mode (f41) are separate fields from the 16-bit result width
 * (f10), so a HALF source is the same instruction with one field changed; see
 * agx3_alu_core.c for the kernel pairs. */
bool agx3_alu_float_to_int16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, bool src_16, bool is_signed,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_add_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_sub_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, enum agx3_alu_hint hint,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_u16_to_f32(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17P's find-top-one operation.  This is the native operation
 * selected for llvm.agx2.ft1 and is the direct hardware implementation of
 * Mesa's ufind_msb/FFS instruction.  The zero input result is -1, matching
 * NIR.  Only the witnessed f32-width register form and its native
 * cache/discard and six dependency-wait axes are admitted. */
bool agx3_alu_ft1_u32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Native G17P f32 GPR + 8-bit minifloat, normal rounding, no modifiers.
 * Destination retention and the GPR's cache/discard state are independent.
 * The six-byte form carries no wait, one of six waits, or slots1+2 together.
 * Other wait combinations, operand files/widths and AGX3c are not admitted.
 */
bool agx3_alu_fadd_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, uint16_t immediate, bool dest_cache,
                            enum agx3_alu_hint hint, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G18G (M6): also the agx3c compact register+minifloat add. */
bool agx3_alu_fadd_minifloat_g18g(enum agx_isa_encoding enc, uint16_t dest,
                                 uint16_t src, uint16_t immediate, bool dest_cache,
                                 enum agx3_alu_hint hint, uint16_t wait_mask,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Same f32/minifloat inputs with GPR negation and/or output saturation.
 * Native ep2 carries saturation and all six independent wait bits; the
 * unsaturated ep1 retains the compact helper's bounded wait combinations.
 * Source hints apply to the GPR even when the native operand slots commute.
 */
bool agx3_alu_fadd_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t src, uint16_t immediate,
                                bool negate_src, bool saturate,
                                enum agx3_alu_hint hint, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G18G (M6): also the agx3c modified register+minifloat add. */
bool agx3_alu_fadd_minifloat_mod_g18g(enum agx_isa_encoding enc, uint16_t dest,
                                     uint16_t src, uint16_t immediate,
                                     bool negate_src, bool saturate,
                                     enum agx3_alu_hint hint, uint16_t wait_mask,
                                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Long G17P add form with an absolute register and minifloat operand. */
bool agx3_alu_fadd_abs_minifloat(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool negate_src, bool saturate,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Exact G17P 1.0 - abs(GPR) form used by the sky blend path. */
bool agx3_alu_fadd_one_absneg(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, bool dest_cache,
                             enum agx3_alu_hint hint, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fadd_abs_negzero(enum agx_isa_encoding enc, uint16_t dest,
                               uint16_t src, bool dest_cache,
                               enum agx3_alu_hint hint, uint16_t wait_mask,
                               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Full grounded G17P minifloat multiply form, including destination
 * retention. `src_abs` is retained as a fail-closed API guard but is always
 * rejected; lower ABS through fadd_abs_negzero followed by a plain FMUL. */
bool agx3_alu_fmul_minifloat_mod(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool src_abs,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * The same multiply with an independently selected destination width: f10
 * chooses a scalar half (1) or a float (2) result while f26 keeps the source
 * at 32 bits. `packHalf2x16` narrows a 32-bit product to a half, and the
 * optimizer folds that f2f16 into the multiply, so this mixed-width form is
 * what actually reaches the packer. See the witnesses in agx3_alu_core.c.
 *
 * G17P refuses ABS and saturation. G17G carries them directly in its longer
 * endpoint, independently of the result width and producer dependency waits.
 */
bool agx3_alu_fmul_minifloat_narrow(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool src_abs, bool dest_half,
   bool saturate,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half times minifloat with a float32 result. */
bool agx3_alu_fmul_half_minifloat_to_f32(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17G scalar-half times minifloat with a float32 result: the f32mul
 * minifloat form with a half source (f43=1), carrying ABS, saturation,
 * operand-cache hints and eight wait slots like the float-source form. */
bool agx3_alu_fmul_half_minifloat_m5(
   uint16_t dest, uint16_t src, uint16_t immediate, bool dest_cache,
   bool src_abs, bool saturate, enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* Unsigned G17P extract into a zero base, with a fixed1..31-bit width.
 * Native srl performs shift+mask in one instruction. Dynamic shifts and
 * immediate14, dead/live sources and slot6 waits are native witnesses. */
bool agx3_alu_bfeil_m5(uint16_t dest, uint16_t source, bool source_16,
                       uint16_t shift, bool shift_is_register, bool shift_16,
                       uint8_t width, const enum agx3_alu_hint hints[2],
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *length);
bool agx3_alu_bfeil(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                    uint16_t shift, bool shift_is_register, uint8_t width,
                    const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length);

/* Extract32 bits from a low/high register pair at an immediate offset1..31.
 * Native EXTRACT is one instruction and carries both source hints and waits. */
bool agx3_alu_extract_imm32(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t low_high[2], unsigned shift,
                            const enum agx3_alu_hint hints[2], bool dest_cache,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length);

/* Signed right shifts by an immediate, with native source lifetime/waits. */
bool agx3_alu_asr_reg_m5(uint16_t dest, uint16_t source, uint16_t shift,
                       bool dest_16, bool source_16,
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *length);

bool agx3_alu_asr_imm_m5(uint16_t dest, uint16_t source, bool dest_16,
                          bool source_16, unsigned shift,
                          enum agx3_alu_hint hint, bool dest_cache,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length);
bool agx3_alu_asr_imm32(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                        unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *length);
bool agx3_alu_asr_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                        unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *length);
bool agx3_alu_asr_imm16_to32(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t source, unsigned shift,
                            enum agx3_alu_hint hint, bool dest_cache,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length);

/*
 * G17P register/immediate FSEL (abs compare modifiers supported for f32
 * eq/lt/gt and single-abs ge/le with f32 selected values). Hardware only
 * encodes negate on the second comparator (f65); other placements rewrite via
 * operand swap, relation flip, or minifloat sign-bit folding.
 */
bool agx3_alu_fcmp_select_reg_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4], bool compare_zero,
                                  const bool compare_abs[2],
                                  const bool compare_neg[2],
                                  const bool immediate[2],
                                  const enum agx3_alu_hint hints[4], unsigned condition,
                                  bool dest_16, bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * Compare a 32-bit float against a minifloat immediate and select either of
 * two 32-bit registers -- what `x < k ? a : b` becomes once Mesa carries the
 * constant as a minifloat. The compared register may carry fabs; that native
 * form uses the long endpoint. Conditions are enum agx_fcond; eq, lt, gt, ge
 * and le are witnessed, ltn and gtn are not and are refused. An inverted
 * condition needs no encoding of its own: Apple spells `!=` as `==` with the
 * two selected operands exchanged.
 */
bool agx3_alu_fcmp_select_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  uint16_t src0, uint8_t minifloat,
                                  bool src0_abs,
                                  uint16_t src2, uint16_t src3,
                                  const enum agx3_alu_hint hints[4],
                                  unsigned condition, bool dest_cache,
                                  uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

/* Native scalar half comparison selecting either of two scalar halves. */
bool agx3_alu_fcmp_select_half(enum agx_isa_encoding enc, uint16_t dest,
                               const uint16_t src[4],
                               const enum agx3_alu_hint hints[4],
                               unsigned condition, bool dest_cache,
                               uint16_t wait_mask,
                               uint8_t out[AGX3_ALU_MAX_LENGTH],
                               uint8_t *len);

bool agx3_alu_fmin_half(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[4],
                        const enum agx3_alu_hint hints[4], bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *len);

bool agx3_alu_fmax_half(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[4],
                        const enum agx3_alu_hint hints[4], bool dest_cache,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *len);

/*
 * 32-bit tied fmin/fmax encoded as compact 6-byte `fsel` (ep7 register, ep2
 * minifloat immediate deposited twice). `wait_mask` carries at most one slot.
 */
bool agx3_alu_fmin_f32(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[4],
                       const enum agx3_alu_hint hints[4], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);

bool agx3_alu_fmax_f32(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[4],
                       const enum agx3_alu_hint hints[4], bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);

bool agx3_alu_fmin_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint8_t minifloat,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fmax_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint8_t minifloat,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native reciprocal with independently selected result/source widths and
 * source modifiers. Both 16- and 32-bit registers use the same f32rcp form. */
bool agx3_alu_rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  bool dest_half, bool src_half, bool src_abs, bool src_neg,
                  enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len);

/*
 * 32-bit reciprocal and reciprocal square root (same forms as the 16-bit
 * encoders with f10=2, f26=2).
 */
bool agx3_alu_f32rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/*
 * 32-bit reciprocal square root with separable source ABS (f30) and NEG (f31)
 * modifiers.
 */
bool agx3_alu_f32rsqrt_mod(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, enum agx3_alu_hint hint,
                           bool dest_cache, uint16_t wait_mask, bool src_neg,
                           bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                           uint8_t *len);

/* Native logarithm with independently selected scalar result/source widths.
 * Any half-width form is currently grounded for no wait or slot 6. */
bool agx3_alu_log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   bool dest_half, bool src_half, bool src_abs, bool src_neg,
                   enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len);

bool agx3_alu_f32exp2_mod(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, enum agx3_alu_hint hint,
                          bool dest_cache, uint16_t wait_mask, bool src_neg,
                          bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len);

/*
 * 32-bit rounding (`f32rint`, f10=2, f26=2): f43 selects mode (absent=rint,
 * 1=trunc, 2=ceil, 3=floor) with shared family lifetime and wait fields.
 */
bool agx3_alu_f32floor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, bool src_neg, bool src_abs,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_f32ceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, bool src_neg, bool src_abs,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * 32-bit round-to-nearest-even (`f32rint`, f43 absent) with full 8-bit
 * register fields, source modifiers, and scoreboard waits (unlike the 6-bit
 * source limit in `agx3_pack_f32_rint_reg`).
 */
bool agx3_alu_f32roundeven(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, enum agx3_alu_hint hint,
                           bool dest_cache, uint16_t wait_mask, bool src_neg,
                           bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                           uint8_t *len);

bool agx3_alu_fcmp_select_f32_f16(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src[4],
                                  const enum agx3_alu_hint hints[4],
                                  unsigned condition, bool dest_cache,
                                  uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

bool agx3_alu_fcmp_select_f16_f16_f32(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[4],
   const enum agx3_alu_hint hints[4], const bool src_abs[2],
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar half reciprocal square root. */
bool agx3_alu_hrsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Native scalar-half base-2 exponential with the recovered source negate. */
bool agx3_alu_hexp2_neg(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint,
                        bool dest_cache, uint16_t wait_mask, bool src_neg,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar half floor. */
bool agx3_alu_hfloor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Native scalar half ceiling. */
bool agx3_alu_hceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len);

/* Native scalar half round-to-nearest-even and truncate. */
bool agx3_alu_hroundeven(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, enum agx3_alu_hint hint,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_htrunc(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len);

/* Apple metal32023.850, applegpu_g17p:
 * sqrt(x)        2f015604030092400000
 * sqrt(-x)       2f015604030092400100 (f31)
 * sqrt(-fabs(x)) 2f015604030092c00100 (f30 and f31)
 * The following multiply retains the same original source modifiers.
 */
bool agx3_alu_sqrt_step_source_modifiers(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   bool src_abs, bool src_neg, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native square-root step with a scalar half source and float32 temporary. */
bool agx3_alu_sqrt_step_half(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, enum agx3_alu_hint hint,
                             bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fmul_narrow(enum agx_isa_encoding enc, uint16_t dest,
                           const uint16_t src[2], const enum agx3_alu_hint hints[2],
                           bool dest_cache, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hmul(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[2], const enum agx3_alu_hint hints[2],
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_hmul_abs(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2],
                       const enum agx3_alu_hint hints[2],
                       const bool src_abs[2], bool saturate, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar half register multiplied by an 8-bit minifloat. */
bool agx3_alu_hmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, uint16_t immediate,
                             enum agx3_alu_hint hint, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hadd(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[2], const enum agx3_alu_hint hints[2],
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Saturating scalar-half multiply and add: sat(a * b) and sat(a + b).
 * G17P spells the saturate as f14, which the compact ep0/ep1 points cannot
 * carry, so these land on the two-byte-longer ep2 form. Apple witnesses:
 *   saturate(a[i] * b[i])   10 03 1d 02 01 00 00 82  (vs 10 03 1d 02 00 c0)
 *   saturate(a[i] + b[i])   10 03 1c 02 01 00 00 82  (vs 10 03 1c 02 00 c0)
 * Their field vectors differ from the unsaturated ones in f14 alone. */
bool agx3_alu_hmul_saturate(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2],
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_hadd_saturate(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2],
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* G17G half addition with source negation bits and optional saturation. */
bool agx3_alu_hadd_neg(enum agx_isa_encoding enc, uint16_t dest,
                     const uint16_t src[2], const enum agx3_alu_hint hints[2],
                     unsigned neg_mask, bool saturate, bool dest_cache,
                     uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half saturate: sat(src + -0.0h). */
bool agx3_alu_hadd_saturate_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native -src + -0.0h used by half negate and matrix inverse lowering, and
 * the same form with saturation for saturate(half). */
bool agx3_alu_half_add_negzero_m5(
   uint16_t dest, uint16_t src, bool src_abs, bool src_neg, bool saturate,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hadd_neg_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar-half inverse: 1.0h + -src. */
bool agx3_alu_hadd_one_neg(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, enum agx3_alu_hint hint,
                           bool dest_cache, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hfma(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[3], const enum agx3_alu_hint hints[3],
                   bool product_neg, bool src2_neg, bool dest_cache,
                   uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hfma_saturate_m5(uint16_t dest, const uint16_t src[3],
              const enum agx3_alu_hint hints[3], bool product_neg,
              bool src2_neg, bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native scalar half FMA with an 8-bit minifloat addend. */
bool agx3_alu_hfma_addend_minifloat(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   uint16_t immediate, const enum agx3_alu_hint hints[2], bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_widen_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                         enum agx3_alu_hint hint, bool src_abs, bool src_neg,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_wait_supported(enum agx3_alu_op op);

bool agx3_alu_integer_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_integer_reg_shift(enum agx_isa_encoding enc, enum agx3_alu_op op,
                                uint16_t dest, const uint16_t src[2],
                                const enum agx3_alu_hint hints[2],
                                uint8_t shift, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH],
                                uint8_t *len);

bool agx3_alu_fcmp_select_imm(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[2],
                             const enum agx3_alu_hint hints[2],
                             unsigned condition, uint8_t if_true,
                             uint8_t if_false, bool dest_16, bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_select_imm_width_neg_second(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   unsigned condition, uint8_t if_true, uint8_t if_false, bool dest_16,
   bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fma_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src, uint8_t immediate, uint16_t addend,
                             enum agx3_alu_hint src_hint,
                             enum agx3_alu_hint addend_hint, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_f32_to_s16(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, enum agx3_alu_hint hint,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, uint16_t immediate, enum agx3_alu_hint hint,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hrcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len);

bool agx3_alu_f32rsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);

bool agx3_alu_f32log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

bool agx3_alu_f32exp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len);

bool agx3_alu_hexp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len);

bool agx3_alu_hlog2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len);

bool agx3_alu_sqrt_step(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_f32log2_mod(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, enum agx3_alu_hint hint,
                          bool dest_cache, uint16_t wait_mask, bool src_neg,
                          bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len);

bool agx3_alu_sqrt_step_mod(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, enum agx3_alu_hint hint,
                            bool dest_cache, uint16_t wait_mask, bool src_abs,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/*
 * G17G 32-bit float compare-select: two registers encode as 14-byte `fseli32`
 * (ep4, f77=76), and one register plus one minifloat encodes as 16-byte
 * `fself32` (ep4, f85=74); two selected immediates are refused. Comparands
 * carry no abs/neg; f11 is the destination-cache hint. Forms that read the
 * uniform file take the shortest endpoint that holds their fields, as the
 * native compiler does.
 */
struct agx3_fsel32_gpu1_desc {
   uint16_t dest;
   bool dest_cache;

   uint16_t cmp0; /* 32-bit register, first comparand */
   bool cmp0_16;
   enum agx3_alu_hint cmp0_hint;
   /* cmp0 names a uniform: an aligned 32-bit one, or a 16-bit one with
    * cmp0_16. See the encoder for the shapes it is admitted in. */
   bool cmp0_uniform;

   bool cmp1_immediate; /* the second comparand is an 8-bit minifloat */
   uint16_t cmp1;       /* 32-bit register, or the minifloat */
   enum agx3_alu_hint cmp1_hint;
   /* The second comparand is a 16-bit register, widened by the comparison
    * (f41 = 1). `fseli16` only; see agx3_alu_fcmp_select16_gpu1(). */
   bool cmp1_16;
   /* cmp1 names a uniform: an aligned 32-bit one, or a 16-bit one with
    * cmp1_16 (`fseli32`/`fself32` only). */
   bool cmp1_uniform;

   /* sel[0] is taken when the comparison holds. An immediate selected
    * operand is a minifloat and moves the instruction to `fself32`, unless
    * sel_integer says it is an 8-bit integer, which `fseli32` carries. */
   bool sel_immediate[2];
   uint16_t sel[2];
   enum agx3_alu_hint sel_hint[2];
   bool sel_integer;
   /* A selected operand names an aligned 32-bit uniform (`fseli32` and
    * `fself32` only). */
   bool sel_uniform[2];

   unsigned condition; /* enum agx_fcond: eq 0, lt 1, gt 2, ge 5, le 6 */
   uint16_t wait_mask;
};

bool agx3_alu_fcmp_select32_gpu1(const struct agx3_fsel32_gpu1_desc *d,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH],
                                 uint8_t *len);

/*
 * The same comparison with a 16-bit result (`fseli16`). Here a selected
 * register is a 16-bit half (any index) and a selected immediate is an 8-bit
 * integer, as in Mesa's fcmpsel; see agx3_alu_core.c for the witnesses.
 */
bool agx3_alu_fcmp_select_half_gpu1(uint16_t dest, const uint16_t src[4],
                                      const enum agx3_alu_hint hints[4],
                                      unsigned condition, bool invert,
                                      bool dest_cache, uint16_t wait_mask,
                                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hadd_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src, uint16_t immediate, bool src_neg,
                            bool saturate, enum agx3_alu_hint hint,
                            bool dest_cache, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

/* Native -src + -0.0h used by half negate and matrix inverse lowering. */
/* Native G17G/G18G f16add abs/negabs with a negative-zero immediate. */
bool agx3_alu_hadd_abs_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool src_neg, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_m5_ldimm(uint16_t dest, bool half, bool dest_cache, uint32_t bits,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_select16_gpu1(const struct agx3_fsel32_gpu1_desc *d,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* dest(16) = src0(16) & ~src1(16): Apple's ctz `bit_andc h3, h3, h2`
 * (gpu/m5-xfb-alu3-reference.metal). */
bool agx3_alu_bit_andc16_gpu1(uint16_t dest, bool dest_cache, uint16_t src0,
                              enum agx3_alu_hint hint0, uint16_t src1,
                              enum agx3_alu_hint hint1, uint16_t wait_mask,
                              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* dest(32) = src0(32) & ~src1(32): the same instruction over aligned pairs,
 * f10/f24/f38 = 2. Apple applegpu_g17g
 * (test/native/g17g/m5-libagx-reference.metal):
 *   x & ~y, both dead       23 00 06 80 32 30 a4 02 84 80
 *   x & ~y, both cached     23 00 06 80 32 30 a2 0a 82 80   into r2
 *   ~x & y, y dead          23 02 06 80 32 30 a0 0a 04 80   into r2 */
bool agx3_alu_bit_andc32_gpu1(uint16_t dest, bool dest_cache, uint16_t src0,
                              enum agx3_alu_hint hint0, uint16_t src1,
                              enum agx3_alu_hint hint1, uint16_t wait_mask,
                              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* dest(16) = src0(16) op src1(16) for AND, OR and XOR: the same operand
 * fields on each mnemonic's second encoding point, Apple applegpu_g17g
 * (gpu/m5-xfb-alu9-reference.metal):
 *   a & b   3b 03 07 00 30 20 a4 02 84 00
 *   a | b   3b 03 07 00 32 28 a4 02 84 00
 *   a ^ b   3b 03 06 00 32 28 a4 02 84 00   both sources dead, wait8 */
bool agx3_alu_bit_reg16_gpu1(enum agx3_alu_op op, uint16_t dest,
                             bool dest_cache, uint16_t src0,
                             enum agx3_alu_hint hint0, uint16_t src1,
                             enum agx3_alu_hint hint1, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* dest = src0 op src1 for the named two-register logic ops, by truth table:
 * ~(a | b) 0x1 bit_nor, a & ~b 0x2 bit_andc, ~(a & b) 0x7 bit_nand,
 * ~(a ^ b) 0x9 bit_xnor, a | ~b 0xB bit_orc. `word` selects aligned 32-bit
 * pairs, otherwise halves. Apple applegpu_g17g/g18g, both widths, one kernel
 * per table (g18ref-out/probe/m6-bitops): the same operand fields as the
 * AND/OR/XOR forms with f43 = 14/11/12/18/17. The one-sided forms keep their
 * operand order (~a & b is bit_andc b, a; ~a | b is bit_orc b, a). */
bool agx3_alu_bit_named_gpu1(unsigned truth_table, bool word, uint16_t dest,
                             bool dest_cache, uint16_t src0,
                             enum agx3_alu_hint hint0, uint16_t src1,
                             enum agx3_alu_hint hint1, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* dest(16) = src(16) & immediate. Only AND is witnessed on G17G. */
bool agx3_alu_bit_imm_gpu1(enum agx3_alu_op op, uint16_t dest, bool dest_cache,
                           uint16_t src, uint8_t immediate,
                           enum agx3_alu_hint hint, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* dest(32) = src + immediate with the register in slot A and an 8-bit
 * immediate in slot B, for the two shapes Apple emits that way: an unsigned
 * saturating add of a 32-bit register (addsat(x, 1u)) and the sign
 * extension of a half (int(short), `+ #0`). See agx3_alu_core.c. */
bool agx3_alu_iadd32_reg_imm_gpu1(uint16_t dest, bool dest_cache, uint16_t src,
                                  bool src_16, bool sign_extend,
                                  enum agx3_alu_hint hint, uint8_t immediate,
                                  bool saturate, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH],
                                  uint8_t *len);

bool agx3_alu_iadd16_gpu1(uint16_t dest, bool dest_cache, bool src0_immediate,
                          uint16_t src0, enum agx3_alu_hint src0_hint,
                          uint16_t src1, enum agx3_alu_hint src1_hint,
                          unsigned shift, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* G17G find-top-one (NIR ufind_msb): the index of the highest set bit of a
 * 32-bit register, or of a 16-bit half into a 16-bit result. Apple's clz
 * kernels (gpu/m5-xfb-alu3-reference.metal) keep the source cached and the
 * destination cached; only the slot-8 wait is witnessed. */
bool agx3_alu_ft1_gpu1(uint16_t dest, uint16_t src, bool half,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len);
/* G17G bit reverse (NIR bitfield_reverse) of a 32-bit register. */
bool agx3_alu_rev_gpu1(uint16_t dest, uint16_t src, enum agx3_alu_hint hint,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
/* G17G float compare selecting between two small 16-bit immediates
 * (`fseli16`): dest(16) = (src0(32) cond src1) ? if_true : if_false, where
 * src1 is an 8-bit minifloat immediate or a 32-bit register. Witnessed for
 * lt and ge, immediates 0..7, and the slot-8 wait
 * (gpu/m5-xfb-alu-reference.metal, -alu3-, -alu4-). */
/* G17G's mixed constant/register 16-bit float select; see agx3_alu_core.c. */
bool agx3_alu_fcmp_select16_const_reg_gpu1(
   uint16_t dest, uint16_t src0, enum agx3_alu_hint src0_hint,
   uint8_t compare_minifloat, uint8_t if_true_constant,
   uint16_t if_false_register, unsigned condition, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_fcmp_bool16_gpu1(uint16_t dest, uint16_t src0,
                               enum agx3_alu_hint src0_hint,
                               bool src1_is_register, uint16_t src1,
                               bool src1_kill, uint8_t if_true,
                               uint8_t if_false, unsigned condition,
                               uint16_t wait_mask,
                               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_isub_imm16_m5(uint16_t dest, uint16_t src, uint16_t immediate,
                           enum agx3_alu_hint hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_bit_half_to_word_m5(enum agx3_alu_op op, uint16_t dest,
                                  const uint16_t src[2],
                                  const enum agx3_alu_hint hints[2], bool dest_cache,
                                  uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_bit_mixed_m5(enum agx3_alu_op op, uint16_t dest, uint16_t word, uint16_t half,
                          enum agx3_alu_hint word_hint,
                          enum agx3_alu_hint half_hint, bool dest_cache,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_and_mixed_m5(uint16_t dest, uint16_t word, uint16_t half,
                          enum agx3_alu_hint word_hint,
                          enum agx3_alu_hint half_hint, bool dest_cache,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);


bool agx3_alu_bit_andc_width_m5(uint16_t dest, const uint16_t src[2],
                                const bool half[2],
                                const enum agx3_alu_hint hints[2],
                                bool dest_cache, uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool agx3_alu_hadd_signed(enum agx_isa_encoding enc, uint16_t dest,
                            const uint16_t src[2], const enum agx3_alu_hint hints[2],
                            const bool src_neg[2], bool dest_cache,
                            uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                            uint8_t *len);

/* Scalar half FMA with a register and two minifloat operands. */
bool agx3_alu_hfma_immediates_m5(uint16_t dest, uint16_t src,
   uint8_t multiplier, uint8_t addend, enum agx3_alu_hint hint,
   bool dest_cache, uint8_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
   uint8_t *len);

/* M5 scalar-half native modifier forms. */
bool agx3_alu_hadd_mod_m5(uint16_t dest, const uint16_t src[2],
                          const enum agx3_alu_hint hints[2],
                          const bool src_abs[2], const bool src_neg[2],
                          bool dest_cache, bool saturate, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_alu_half_add_imm_to_f32_m5(uint16_t dest, uint16_t src,
                                     uint8_t immediate, bool src_abs,
                                     bool src_neg, enum agx3_alu_hint hint,
                                     bool dest_cache, uint16_t wait_mask,
                                     uint8_t out[AGX3_ALU_MAX_LENGTH],
                                     uint8_t *len);

#ifdef __cplusplus
}
#endif
