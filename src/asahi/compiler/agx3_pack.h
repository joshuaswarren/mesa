/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Largest ldimm instruction emitted by the currently proven AGX3 subset. */
#define AGX3_LDIMM_MAX_LENGTH 8
#define AGX3_IADD_U32_IMM8_MAX_LENGTH 12
#define AGX3_IADD_U32_REG_REG_MAX_LENGTH 12
#define AGX3_FADD_F32_REG_REG_LENGTH 6
#define AGX3_FMUL_F32_REG_REG_LENGTH 6
#define AGX3_FMA_F32_REG_REG_REG_MAX_LENGTH 16
#define AGX3_FMIN_F32_REG_REG_MAX_LENGTH 14
#define AGX3_FMAX_F32_REG_REG_MAX_LENGTH 14
#define AGX3_I32_TO_F32_REG_MAX_LENGTH 10
#define AGX3_U32_TO_F32_REG_MAX_LENGTH 10
#define AGX3_F32_RINT_REG_LENGTH 10
#define AGX3_F32_RCP_REG_LENGTH 10
#define AGX3_F32_RSQRT_REG_LENGTH 10
#define AGX3_F32_TO_INT_RTZ_REG_LENGTH 10
#define AGX3_BIT_AND_U32_REG_REG_LENGTH 10
#define AGX3_BIT_OR_U32_REG_REG_LENGTH 10
#define AGX3_BIT_XOR_U32_REG_REG_LENGTH 10
#define AGX3_BIT_NOT_U32_REG_MAX_LENGTH 10
#define AGX3_ISUB_U32_REG_REG_MAX_LENGTH 12
#define AGX3_SLL_U32_REG_REG_MAX_LENGTH 14
#define AGX3_ASR_I32_REG_REG_MAX_LENGTH 12
#define AGX3_LSR_U32_REG_REG_MAX_LENGTH 14
#define AGX3_IMUL_U32_REG_REG_MAX_LENGTH 14
#define AGX3_IMADD_U32_REG_REG_REG_MAX_LENGTH 14
#define AGX3_THREADGROUP_BARRIER_LENGTH 6
#define AGX3_SCOPED_BARRIER_MAX_LENGTH 12
#define AGX3_SIMD_BROADCAST_F32_IMM_LENGTH 10
#define AGX3_INTERSECT_RAY_MAX_LENGTH 14
#define AGX3_CONTINUE_RAY_MAX_LENGTH 8
#define AGX3_CONTINUE_RAY_FIXED_FIRST_SITE_WITNESS_MAX_LENGTH 8
#define AGX3_INC_STK_FIXED_WITNESS_LENGTH 8
#define AGX3_END_LENGTH 4

struct agx3_encoded_ldimm {
   uint8_t bytes[AGX3_LDIMM_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_iadd_u32_imm8 {
   uint8_t bytes[AGX3_IADD_U32_IMM8_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_iadd_u32_reg_reg {
   uint8_t bytes[AGX3_IADD_U32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_fadd_f32_reg_reg {
   uint8_t bytes[AGX3_FADD_F32_REG_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_fmul_f32_reg_reg {
   uint8_t bytes[AGX3_FMUL_F32_REG_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_fma_f32_reg_reg_reg {
   uint8_t bytes[AGX3_FMA_F32_REG_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_fmin_f32_reg_reg {
   uint8_t bytes[AGX3_FMIN_F32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_fmax_f32_reg_reg {
   uint8_t bytes[AGX3_FMAX_F32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_i32_to_f32_reg {
   uint8_t bytes[AGX3_I32_TO_F32_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_u32_to_f32_reg {
   uint8_t bytes[AGX3_U32_TO_F32_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_f32_rint_reg {
   uint8_t bytes[AGX3_F32_RINT_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_f32_rcp_reg {
   uint8_t bytes[AGX3_F32_RCP_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_f32_rsqrt_reg {
   uint8_t bytes[AGX3_F32_RSQRT_REG_LENGTH];
   uint8_t length;
};

enum agx3_f32_to_int_rtz_mode {
   AGX3_F32_TO_I32_RTZ = 0,
   AGX3_F32_TO_U32_RTZ = 1,
};

struct agx3_encoded_f32_to_int_rtz_reg {
   uint8_t bytes[AGX3_F32_TO_INT_RTZ_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_bit_and_u32_reg_reg {
   uint8_t bytes[AGX3_BIT_AND_U32_REG_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_bit_or_u32_reg_reg {
   uint8_t bytes[AGX3_BIT_OR_U32_REG_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_bit_xor_u32_reg_reg {
   uint8_t bytes[AGX3_BIT_XOR_U32_REG_REG_LENGTH];
   uint8_t length;
};

struct agx3_encoded_bit_not_u32_reg {
   uint8_t bytes[AGX3_BIT_NOT_U32_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_isub_u32_reg_reg {
   uint8_t bytes[AGX3_ISUB_U32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_sll_u32_reg_reg {
   uint8_t bytes[AGX3_SLL_U32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_asr_i32_reg_reg {
   uint8_t bytes[AGX3_ASR_I32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_lsr_u32_reg_reg {
   uint8_t bytes[AGX3_LSR_U32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_imul_u32_reg_reg {
   uint8_t bytes[AGX3_IMUL_U32_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_imadd_u32_reg_reg_reg {
   uint8_t bytes[AGX3_IMADD_U32_REG_REG_REG_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_threadgroup_barrier {
   uint8_t bytes[AGX3_THREADGROUP_BARRIER_LENGTH];
   uint8_t length;
};

enum agx3_barrier_scope {
   AGX3_BARRIER_SCOPE_EXECUTION = 0,
   AGX3_BARRIER_SCOPE_THREADGROUP,
   AGX3_BARRIER_SCOPE_DEVICE,
   AGX3_BARRIER_SCOPE_TEXTURE,
   AGX3_BARRIER_SCOPE_DEVICE_FENCE,
   AGX3_BARRIER_SCOPE_TEXTURE_FENCE,
   AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT,
   /* AGX3C only: a memory-only texture fence at threadgroup scope. */
   AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP,
   /* Compiler ordering only: M5 subgroup shared memory needs no rendezvous. */
   AGX3_BARRIER_SCOPE_SUBGROUP_SHARED,
   /* Captured native mesh_amplified entry before the payload device load. */
   AGX3_BARRIER_SCOPE_MESH_PAYLOAD_ENTRY,
   /* Captured object.write_childcount after the API workgroup barrier. */
   AGX3_BARRIER_SCOPE_OBJECT_PAYLOAD_EXIT,
};

struct agx3_encoded_scoped_barrier {
   uint8_t bytes[AGX3_SCOPED_BARRIER_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_simd_broadcast_f32_imm {
   uint8_t bytes[AGX3_SIMD_BROADCAST_F32_IMM_LENGTH];
   uint8_t length;
};

struct agx3_encoded_intersect_ray {
   uint8_t bytes[AGX3_INTERSECT_RAY_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_continue_ray {
   uint8_t bytes[AGX3_CONTINUE_RAY_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_continue_ray_fixed_first_site_witness {
   uint8_t bytes[AGX3_CONTINUE_RAY_FIXED_FIRST_SITE_WITNESS_MAX_LENGTH];
   uint8_t length;
};

struct agx3_encoded_inc_stk_fixed_witness {
   uint8_t bytes[AGX3_INC_STK_FIXED_WITNESS_LENGTH];
   uint8_t length;
};

struct agx3_encoded_end {
   uint8_t bytes[AGX3_END_LENGTH];
   uint8_t length;
};

/*
 * Encode the exact register-source f32 `simdbcast` immediate-lane
 * shape recovered from the pinned G15G/G16G/G17S subgroup corpus.  Register
 * operands are even half-register indices.  AGX3b accepts 0..190 and AGX3c
 * deliberately stops at 254: the composer's ninth AGX3c register bit is not
 * exposed because no source-correlated specimen exercises it.
 *
 * Only immediate lanes 0 and 3 have direct `simdbcast` witnesses, so every
 * other immediate, register/uniform lane selection, element width, shuffle
 * mnemonic and scheduler-marker shape fails closed.  The fixed marker shape
 * here is the one carried by SB_bcast0/SB_bcast3.  The production backend may
 * lower only this exact bounded form; it does not imply wider subgroup
 * support or feature exposure.
 */
bool agx3_pack_simd_broadcast_f32_imm(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src, uint8_t lane,
   struct agx3_encoded_simd_broadcast_f32_imm *out);

/*
 * Encode the source-grounded 32-bit-register form of AGX3 `ldimm`.
 *
 * `dest` is a 16-bit half-register index, matching agx_index.value and the
 * register fields recovered from the rp_0..rp_64 probes.  A 32-bit destination
 * must therefore be even.  This deliberately does not expose the other
 * operand-kind forms, the 10-byte encoding points, or M5's unobserved ninth
 * destination bit.
 *
 * Returns false, without producing an instruction, for every unsupported or
 * malformed request.  In particular AGX2/UNKNOWN encodings, odd destinations,
 * and destinations outside Mesa's 256-half-register file fail closed.
 */
bool agx3_pack_ldimm_u32(enum agx_isa_encoding encoding, uint16_t dest,
                         uint32_t imm, struct agx3_encoded_ldimm *out);

/*
 * Encode the source-grounded register-plus-inline-u8 form of 32-bit
 * integer add.  `dest` and `src` are 16-bit half-register indices, so both
 * must be even.  The encoder tables limit this encoding point to r0..r95
 * on AGX3b (half-registers 0..190) and r0..r127 on AGX3c (0..254).
 *
 * The anonymous Apple fields were attached to their semantics by the pinned
 * add-immediate and register-pressure probes: f2 is the destination, f31 the
 * register source, and f15 the complete 8-bit immediate.  The same form has
 * 224 exact G15/G16 corpus witnesses and 218 exact G17 witnesses.  Cross-target
 * compiler replay is byte-identical within {G15*, G16*, G17P} and within
 * {G17G, G17S}; the two encodings are 10 and 12 bytes respectively.
 *
 * This function does not schedule G15G.  Its required standalone nop/stall
 * words remain a separate scheduling pass and are never folded into iadd.
 */
bool agx3_pack_iadd_u32_imm8(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src, uint8_t imm,
   struct agx3_encoded_iadd_u32_imm8 *out);

/*
 * Encode the source-grounded three-register form of 32-bit integer
 * add.  P2i_add identifies f2 as the destination and f16/f31 as its two
 * register sources: the immediately preceding lddev instructions define the
 * exact f16/f31 registers consumed by this instruction.  The independent
 * Q_iadd occurrence ladder reproduces the same form on G15G, G16G and G17S.
 *
 * All arguments are half-register indices and must therefore be even.  This
 * bounded API exposes r0..r95 on AGX3b and r0..r127 on AGX3c.  Apple's AGX3c
 * table contains a ninth register bit, but Mesa does not model that register
 * space and the pinned source probes do not exercise it, so it fails closed.
 * Only the exact anonymous-field selection emitted by the cross-target
 * volatile-load-plus-gid replay is accepted.  The other anonymous-field
 * selections visible in P2i_add and Q_iadd remain unsupported until they are
 * named.  G15G's standalone scheduling nops are deliberately outside this
 * packer.
 */
bool agx3_pack_iadd_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_iadd_u32_reg_reg *out);

/*
 * Encode the exact compact register-register f32 add selection common
 * to Apple's AGX3b and AGX3c tables.  A forced-live eight-load replay
 * independently correlates f2 with the result, f35 with `src0`, and f18 with
 * `src1` across G15G/S/D, G16P/G/S, G17P, and G17G/S.  Independent delayed-
 * use captures emit this six-byte selection on G15G, G16G, and G17S; high-
 * pressure G16G/G17S captures reproduce it while exercising register bit 7.
 *
 * All operands are even half-register indices in Mesa's modeled r0..r127
 * file.  This packer preserves the two physical source fields even though
 * fadd is commutative.  It accepts only the exact anonymous-field selection
 * in those compact captures.  Modifier-bearing, kill, wait/scheduler-extended
 * and twelve-byte forms, AGX3c's unprobed ninth register bits, and G15G's
 * standalone scheduling nops remain outside this API and fail closed.
 */
bool agx3_pack_fadd_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fadd_f32_reg_reg *out);

/*
 * Encode the exact compact register-register f32 multiply selections
 * emitted by the pinned compiler corpus.  P2f_mul correlates f2 with the
 * result, f35 with `src0`, and f18 with `src1`; the L_fmul and Q_fmul ladders
 * independently reproduce the operand fields.  The selected AGX3b form is
 * the fixed-0xc0 L/P2 selection witnessed across seven targets.  The selected
 * AGX3c form is G17S's fixed-0x20 delayed-use witness.  Those target gates
 * belong to the caller because this shared layer receives an encoding
 * generation rather than a chip.
 *
 * Operands are even half-register indices restricted to 0..126.  The compact
 * corpus does not exercise register bit 7, so it is deliberately not inferred
 * from the separate compact fadd mapping.  Alternate anonymous/context
 * selections (including the 0x20-trailer G15G/G16G Q captures), modifier-
 * bearing and fourteen-byte forms, and G15G's separate scheduling
 * instructions remain outside this exact API and fail closed.
 */
bool agx3_pack_fmul_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fmul_f32_reg_reg *out);

/*
 * Encode the exact four-register f32 fused multiply-add selection
 * emitted by the pinned Apple compiler corpus.  P3f_fma correlates f2 with
 * the result, f35/f18 with the two multiplicands, and f52 with the addend on
 * G15G, G16G and G17S.  The independent L_ffma ladder reproduces the same
 * anonymous/context selection byte-for-byte on G15G/S/D, G16P/G/S, G17P,
 * G17G and G17S.  It is eight bytes on AGX3b and sixteen on AGX3c.
 *
 * The two multiplicand fields are kept in physical order even though the
 * mathematical product is commutative.  Every operand is restricted to an
 * even half-register index in 0..126 because no direct f32fma artifact
 * exercises register bit 7.  Modifier-bearing selections, unfused-mad
 * semantics, unprobed high register bits, and G15G's standalone scheduling
 * instructions remain outside this exact API and fail closed.  Chip gates
 * belong to the caller.
 */
bool agx3_pack_fma_f32_reg_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t mul0,
   uint16_t mul1, uint16_t addend,
   struct agx3_encoded_fma_f32_reg_reg_reg *out);

/*
 * Encode the exact register-register f32 minimum selection emitted by
 * the pinned Apple compiler.  P2f_min correlates the result and both inputs;
 * the L_fmin ladder independently reproduces the same fixed selection on
 * G15G/S/D, G16P/G/S, G17P, G17G and G17S.  Apple's implementation ties each
 * input into both the comparison and selection operands.  This API keeps
 * those duplicated physical fields coherent.  The form is six bytes on
 * AGX3b (`fsel.ep7`) and fourteen on AGX3c (`fseli32.ep4`).
 *
 * Operands are restricted to even half-register indices in 0..126 because
 * direct fmin artifacts do not exercise register bit 7.  Other comparison
 * predicates, fmax, generic select, modifiers, alternate context selections,
 * unprobed registers, and scheduling/lowering policy remain outside this
 * exact API and fail closed.  In particular, this codec does not independently
 * claim IEEE NaN or signed-zero policy beyond the witnessed compiler form.
 */
bool agx3_pack_fmin_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fmin_f32_reg_reg *out);

/*
 * Encode the exact register-register f32 maximum selection emitted by
 * the pinned Apple compiler.  P2f_max correlates the result and both inputs;
 * the L_fmax ladder independently reproduces the same fixed selection on
 * G15G/S/D, G16P/G/S, G17P, G17G and G17S.  Apple's implementation ties each
 * input into both the comparison and selection operands.  This API keeps
 * those duplicated physical fields coherent.  The form is six bytes on
 * AGX3b (`fsel.ep7`) and fourteen on AGX3c (`fseli32.ep4`).
 *
 * Operands are restricted to even half-register indices in 0..126 because
 * direct fmax artifacts do not exercise register bit 7.  Other comparison
 * predicates, fmin, generic select, modifiers, alternate context selections,
 * unprobed registers, and scheduling/lowering policy remain outside this
 * exact API and fail closed.  In particular, this codec does not independently
 * claim IEEE NaN or signed-zero policy beyond the witnessed compiler form.
 */
bool agx3_pack_fmax_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fmax_f32_reg_reg *out);

/*
 * Encode the exact signed-i32-to-f32 register conversion emitted by
 * the pinned Apple compiler.  The L_i2f ladder reproduces this anonymous-
 * field selection byte-for-byte on G15G/S/D, G16P/G/S, G17P, G17G and G17S.
 * In the independent typical_kernel captures, an integer select writes f18's
 * register and the immediately following f32 add consumes f2's result, which
 * correlates f18 with the signed-i32 source and f2 with the f32 destination on
 * both AGX3b and AGX3c.  The form is eight bytes on AGX3b and ten on AGX3c.
 *
 * Operands are restricted to even half-register indices in 0..126 because
 * direct conversion artifacts do not exercise register bit 7.  Unsigned
 * conversion, float-to-integer conversion, alternate source selections,
 * modifiers, unprobed registers, and scheduling/lowering/runtime policy are
 * outside this exact API and fail closed.  Chip gates belong to the caller.
 */
bool agx3_pack_i32_to_f32_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_i32_to_f32_reg *out);

/*
 * Encode the exact unsigned-u32-to-f32 register conversion emitted by
 * the pinned Apple compiler.  The u2f source seed reproduces this anonymous-
 * field selection byte-for-byte on G15G/S/D, G16P/G/S, G17P, G17G and G17S.
 * Independent P2i_div dataflow correlates f18 with nonzero integer-select
 * results and f2 with the float consumers of each conversion result on both
 * AGX3b and AGX3c.  The form is eight bytes on AGX3b and ten on AGX3c.
 *
 * Operands are restricted to even half-register indices in 0..126 because
 * direct unsigned-conversion artifacts do not exercise register bit 7.  The
 * signed conversion is a distinct fixed selection.  Float-to-integer forms,
 * alternate sources, modifiers, unprobed registers, and scheduling/lowering/
 * runtime policy are outside this exact API and fail closed.  Chip gates
 * belong to the caller.
 */
bool agx3_pack_u32_to_f32_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_u32_to_f32_reg *out);

/*
 * Encode the exact ten-byte `f32rint` selection immediately preceding
 * the signed conversion in PC_f2ir.  A nine-identity Q_rint_pc replay
 * reproduces the AGX3b/AGX3c bytes, Q_rint_convert_kill reproduces the same
 * fixed context with nonzero registers, and Q_rint_convert_dataflow keeps the
 * input live to independently correlate f2 with the result and f18 with the
 * source.  Apple's composer reproduces all 4,096 even-register pairs from
 * those fixed fields on each table variant.
 *
 * Direct rint/trunc/ceil/floor sources distinguish selector f43 values
 * 0/1/2/3 respectively; this API exposes only the rint selection.  It does
 * not claim tie, exceptional-value, saturation, modifier, alternate-context,
 * scheduling, or lowering semantics.  Operands are even half-register
 * indices in 0..126.  Chip gates belong to the caller.
 */
bool agx3_pack_f32_rint_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_rint_reg *out);

/*
 * Encode the exact ten-byte scalar f32 reciprocal selected by the
 * pinned 1.0f/x source on G15G/S/D, G16P/G/S, G17P, and G17G/S.  A fresh
 * nine-target direct replay is byte-identical to the pinned frecip corpus.
 * Q_rcp_dataflow and Q_rcp_live8 keep each input live after the reciprocal,
 * independently identifying f2 as destination and f18 as source.  The first
 * nonzero Q_rcp_kill instruction reproduces the direct last-use/dependency
 * context exactly.  Apple's composer reproduces all 4,096 even-register
 * pairs from those fixed fields on each table variant.
 *
 * Only that fixed context is exposed.  Live-source or alternate dependency
 * contexts, rsqrt and other special functions, accuracy/refinement policy,
 * modifiers, high registers, scheduling, and lowering remain closed.
 * Operands are even half-register indices in 0..126.  Chip gates belong to
 * the caller.
 */
bool agx3_pack_f32_rcp_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_rcp_reg *out);

/*
 * Encode the exact ten-byte scalar f32 reciprocal-square-root selected
 * by the pinned rsqrt source on G15G/S/D, G16P/G/S, G17P, and G17G/S.  A
 * fresh nine-target direct replay is byte-identical to the pinned frsqrt
 * corpus.  Q_rsqrt_dataflow keeps its input live after the operation and the
 * Q_rsqrt_live8 last-use ladder independently identifies f2 as destination
 * and f18 as source.  Apple's composer reproduces all 4,096 even-register
 * pairs from those fixed fields on each table variant.
 *
 * Only the fixed direct last-use/dependency context is exposed.  Live-source
 * or alternate dependency contexts, reciprocal and other special functions,
 * accuracy/refinement policy, modifiers, high registers, scheduling, and
 * lowering remain closed.  Operands are even half-register indices in
 * 0..126.  Chip gates belong to the caller.
 */
bool agx3_pack_f32_rsqrt_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_rsqrt_reg *out);

/*
 * Encode the exact scalar f32-to-i32/u32 language-cast selections
 * emitted for the pinned fptosi/fptoui sources.  Both are ten-byte
 * flt2int.ep0 forms on AGX3b and AGX3c.  Direct f2i/f2u plus independent PC
 * probes reproduce the fixed signed/unsigned selectors on G15G/S/D,
 * G16P/G/S, G17P, and G17G/S.  P2f_ldexp and the compiler's division
 * expansions independently correlate f2 with the integer destination and
 * f16 with the float source across both table variants.
 *
 * The source casts establish round-toward-zero behavior for in-range values;
 * PC_f2ir emits a separate f32rint before the same signed cast.  Other
 * rounding selections, saturation, NaN/out-of-range behavior, source reuse/
 * kill variants, alternate contexts, high registers, and lowering policy are
 * not exposed.  Operands are even half-register indices in 0..126.  Chip
 * gates belong to the caller.
 */
bool agx3_pack_f32_to_int_rtz_reg(
   enum agx_isa_encoding encoding, enum agx3_f32_to_int_rtz_mode mode,
   uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_to_int_rtz_reg *out);

/*
 * Encode the source-grounded three-register u32 `bit_and` form.
 * P2i_and's two preceding lddev destinations correlate f2 with the result,
 * f16 with one input, and f30 with the other on both Apple table variants.
 * Q_iand_n1..n4 independently reproduce the encoding point on G15G, G16G,
 * and G17S.
 *
 * Registers use even half-register indices.  AGX3b's table covers the whole
 * modeled r0..r127 file.  AGX3c contains an unprobed ninth register bit, so
 * this API deliberately keeps all three operands in r0..r127.  Only the exact
 * anonymous-field selection from the nine-target volatile-load-and-gid replay
 * is accepted; other anonymous-field selections and G15G's standalone
 * scheduling nops remain outside this packer.
 */
bool agx3_pack_bit_and_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_and_u32_reg_reg *out);

/*
 * Encode the source-grounded three-register u32 `bit_or` form.
 * P2i_or's two preceding lddev destinations correlate f2 with the result,
 * f16 with one input, and f30 with the other on both Apple table variants.
 * Q_ior_n1..n4 independently reproduce the encoding point on G15G, G16G,
 * and G17S.
 *
 * Registers use even half-register indices.  AGX3b's table covers the whole
 * modeled r0..r127 file.  AGX3c contains an unprobed ninth register bit, so
 * this API deliberately keeps all three operands in r0..r127.  Only the exact
 * anonymous-field selection from the nine-target volatile-load-and-gid replay
 * is accepted; other anonymous-field selections and G15G's standalone
 * scheduling nops remain outside this packer.
 */
bool agx3_pack_bit_or_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_or_u32_reg_reg *out);

/*
 * Encode the source-grounded three-register u32 `bit_xor` form.
 * P2i_xor's two preceding lddev destinations correlate f2 with the result,
 * f16 with one input, and f30 with the other on both Apple table variants.
 * Q_ixor_n1..n4 independently reproduce the encoding point on G15G, G16G,
 * and G17S.
 *
 * Registers use even half-register indices.  AGX3b's table covers the whole
 * modeled r0..r127 file.  AGX3c contains an unprobed ninth register bit, so
 * this API deliberately keeps all three operands in r0..r127.  Only the exact
 * anonymous-field selection from the nine-target volatile-load-and-gid replay
 * is accepted; other anonymous-field selections and G15G's standalone
 * scheduling nops remain outside this packer.
 */
bool agx3_pack_bit_xor_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_xor_u32_reg_reg *out);

/*
 * Encode the source-grounded u32 register `bit_not` form, named `inv`
 * by Apple's tables.  The forced-live two-load replay independently
 * correlates f16 with the first lddev destination and f2 with the result
 * consumed by the following integer add.  It reproduces on every available
 * G15/G16/G17 compiler target; Q_iunot_n1..n4 separately establishes the
 * unary operation's occurrence count on G15G, G16G, and G17S.
 *
 * Registers use even half-register indices and are bounded to r0..r127.
 * AGX3c's ninth register bits, other anonymous-field selections, and G15G's
 * standalone scheduling nops remain unsupported and fail closed.
 */
bool agx3_pack_bit_not_u32_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_bit_not_u32_reg *out);

/*
 * Encode the source-grounded three-register u32 subtraction form,
 * named `isub` on AGX3b and `isub32` on AGX3c by Apple's tables.  The
 * forced-live two-load/three-store replay independently correlates f2 with
 * destination r3, f16 with minuend r2, and f31 with subtrahend r0 on every
 * available G15/G16/G17 compiler target.  Q_isub_n1..n4 independently
 * reproduces the operation on G15G, G16G, and G17S.
 *
 * All arguments are even half-register indices.  Apple's AGX3b encoding
 * point is limited to r0..r95; AGX3c is bounded here to the modeled
 * r0..r127 file, excluding its unprobed ninth register bits.  Other anonymous
 * selections and G15G's standalone scheduling nops remain unsupported.
 */
bool agx3_pack_isub_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_isub_u32_reg_reg *out);

/*
 * Encode the source-grounded three-register u32 variable shift-left
 * form, named `sll` by Apple's tables.  The forced-live four-store replay
 * independently correlates f2 with destination r3, f16 with value r5, and
 * f31 with masked shift count r0 on every available G15/G16/G17 target.
 * Q_ishl_n1..n4 independently reproduces the operation on G15G, G16G, and
 * G17S.
 *
 * All arguments are even half-register indices.  Apple's AGX3b encoding
 * point is limited to r0..r95; AGX3c is bounded here to modeled r0..r127,
 * excluding its unprobed ninth register bits.  This packs only `sll`, not the
 * separate count-masking operation.  Other anonymous selections and G15G's
 * standalone scheduling nops remain unsupported.
 */
bool agx3_pack_sll_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_sll_u32_reg_reg *out);

/*
 * Encode the source-grounded three-register i32 arithmetic variable
 * shift-right form, named `sra` by Apple's tables and `asr` by Mesa.  The
 * forced-live four-store replay independently correlates f2 with destination
 * r3, f16 with signed value r5, and f31 with masked shift count r0 on every
 * available G15/G16/G17 compiler target.  P2i_shr and Q_ishr_n1..n4
 * independently reproduce the arithmetic operation on G15G, G16G, and G17S.
 *
 * All arguments are even half-register indices.  Apple's AGX3b encoding
 * point is limited to r0..r95; AGX3c is bounded here to modeled r0..r127,
 * excluding its unprobed ninth register bits.  This packs only `sra`, not the
 * separate count-masking operation.  Other anonymous selections and G15G's
 * standalone scheduling nops remain unsupported.
 */
bool agx3_pack_asr_i32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_asr_i32_reg_reg *out);

/*
 * Encode the source-grounded three-register u32 logical variable
 * shift-right form, named `srl` by Apple's tables and `lsr` by Mesa.  The
 * forced-live four-store replay independently correlates f2 with destination
 * r3, f16 with unsigned value r5, and f31 with masked shift count r0 on every
 * available G15/G16/G17 compiler target.  P2u_shr and L_ushr_n2..n8
 * independently reproduce the logical operation across those generations.
 *
 * All arguments are even half-register indices.  Apple's AGX3b encoding
 * point is limited to r0..r95; AGX3c is bounded here to modeled r0..r127,
 * excluding its unprobed ninth register bits.  This packs only `srl`, not the
 * separate count-masking operation.  Other anonymous selections and G15G's
 * standalone scheduling nops remain unsupported.
 */
bool agx3_pack_lsr_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_lsr_u32_reg_reg *out);

/*
 * Encode the source-grounded three-register low-u32 multiply form.
 * Apple's tables name the containing operation `imadd` on AGX3b and
 * `imadd32` on AGX3c; this API accepts only the compiler's implicit-zero
 * addend selection, which is exactly a 32-bit multiply.  The forced-live
 * three-store replay independently correlates f2 with destination r3, f16
 * with one source r0, and f31 with the other source r2 on every available
 * G15/G16/G17 compiler target.  P2i_mul/P2u_mul and the Q_imul/L_imul
 * ladders independently reproduce this low-product operation across
 * generations.
 *
 * All arguments are even half-register indices.  Apple's AGX3b encoding
 * point is limited to r0..r95; AGX3c is bounded here to modeled r0..r127,
 * excluding its unprobed ninth register bits.  Other addend and anonymous
 * selections, signed/unsigned high multiply, and G15G's standalone scheduling
 * nops remain unsupported.
 */
bool agx3_pack_imul_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_imul_u32_reg_reg *out);

/*
 * Encode the source-grounded four-register low-u32 multiply-add form,
 * named `imadd` by Apple's AGX3b table and `imadd32` by AGX3c.  The forced-live
 * four-store replay independently correlates f2 with destination r5, f16 with
 * multiplier r2, f31 with multiplier r3, and f46 with real addend register r0
 * on every available G15/G16/G17 compiler target.  P3i/P3u and Q_i3mad
 * independently reproduce the same low-product-plus-addend operation.  Signed
 * and unsigned low forms are bit-identical modulo 2^32.
 *
 * All arguments are even half-register indices.  Apple's AGX3b encoding point
 * is limited to r0..r95; AGX3c is bounded here to modeled r0..r127, excluding
 * its unprobed ninth register bits.  This real-register-addend selection is
 * distinct from the implicit-zero imul selection.  Alternate selections,
 * signed/unsigned high multiply-add, and G15G's standalone scheduling nops
 * remain unsupported.
 */
bool agx3_pack_imadd_u32_reg_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t mul0,
   uint16_t mul1, uint16_t addend,
   struct agx3_encoded_imadd_u32_reg_reg_reg *out);

/*
 * Encode the fixed workgroup barrier emitted for air.wg.barrier.
 * Forced n=0..4 occurrence ladders reproduce the exact six-byte AGX3b form on
 * G15G/G15S/G16G/G17P and the distinct six-byte AGX3c form on G17S.  Apple's
 * independently extracted encoder tables identify each sequence as the
 * field-free `barrier` instruction.  Cross-generation byte strings are
 * rejected rather than being reinterpreted.
 */
bool agx3_pack_threadgroup_barrier(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_threadgroup_barrier *out);

/*
 * Encode the complete field-free AGX3 fence scope family recovered
 * from five independent Metal mem_flags contrasts on G15G/G16G/G17S.
 * Execution-only, threadgroup-memory and device-memory scopes are one six-byte
 * barrier. Texture memory is the exact ordered scope-4/scope-3 pair and is
 * therefore twelve bytes. Wait attachments remain outside this API.
 */
bool agx3_pack_scoped_barrier(enum agx_isa_encoding encoding,
                              enum agx3_barrier_scope scope,
                              struct agx3_encoded_scoped_barrier *out);

/*
 * AGX3C texture barriers carry their scoreboard wait in the first barrier
 * word itself (fields 16..23 name slots 1..8). G17G waits there on the result
 * slots of the image writes still in flight: Metal texture.fence(),
 * atomic_thread_fence(mem_texture) at every scope and
 * threadgroup_barrier(mem_texture) all do. Only the AGX3C texture scopes
 * accept a wait; a zero wait gives agx3_pack_scoped_barrier()'s bytes.
 */
bool agx3_pack_scoped_barrier_wait(enum agx_isa_encoding encoding,
                                   enum agx3_barrier_scope scope,
                                   uint16_t wait_mask,
                                   struct agx3_encoded_scoped_barrier *out);

/*
 * Encode the canonical unconditional shader terminator.  The exact
 * bytes 0e000000 terminate every decoded code region in the hash-pinned
 * G15/G16/G17 corpus.  Apple's tables expose additional anonymous end fields;
 * those noncanonical forms are deliberately rejected here.  The separately
 * proven wait-fence attachment fields are available only through the narrowly
 * named API below.
 */
bool agx3_pack_end(enum agx_isa_encoding encoding,
                   struct agx3_encoded_end *out);

/*
 * Encode the corpus-grounded hardware `intersect_ray` instruction.
 * The encoder tables identify f5 as a 13-bit half-register field; the
 * surrounding native stream makes its SCS/ray-state role likely, but that
 * semantic name remains an inference.  Two direct native-corpus pairs vary
 * only f5 and establish these exact values and bytes:
 *
 *   AGX3b: 0, 4
 *   AGX3c: 0, 8
 *
 * Every other anonymous field is held to the corresponding differential
 * pair's canonical value.  Values merely spellable by Apple's tables, other
 * intersect_ray specimens in the corpus, and every unwitnessed target remain
 * rejected.  This codec does not represent the required scoped barrier, SCS
 * layout, traversal lowering, scheduling, submission, or an API capability.
 */
bool agx3_pack_intersect_ray(enum agx_isa_encoding encoding,
                             uint16_t state_register,
                             struct agx3_encoded_intersect_ray *out);

/* Return an encoding only for the M3/M4/A18 Pro/M5 identities that reproduce
 * both exact pair members in the pinned cross-target native corpus.
 * ISA-equivalent identities without those bytes stay fail-closed. */
enum agx_isa_encoding
agx3_intersect_ray_witnessed_chip_encoding(enum agx_chip chip);

/*
 * Encode the canonical hardware `continue_ray` instruction with its
 * one varying Apple-table field, f5.  The field is a half-register index; its
 * exact bit placement is established by the AGX3b/AGX3c encoder tables.
 * Its role as the ray-state/SCS register is an inference from the surrounding
 * lowering, so callers must not treat this codec as a complete traversal
 * operation.
 *
 * This deliberately accepts only values present in the pinned five-kernel RT
 * corpus on every compiler target of the corresponding encoding:
 *
 *   AGX3b: 0, 2, 4, 6, 8, 10
 *   AGX3c: 0, 2, 8, 10, 12, 16, 18
 *
 * All selector, flag, wait, and hint fields remain at their corpus-canonical
 * values.  No intrinsic operands, SCS layout, barrier sequence, lowering,
 * scheduling, submission, or API capability is represented here.
 */
bool agx3_pack_continue_ray(enum agx_isa_encoding encoding,
                            uint16_t state_register,
                            struct agx3_encoded_continue_ray *out);

/* Return the encoding only for chip identities with a pinned native
 * `continue_ray` corpus.  Same-encoding but unwitnessed identities remain
 * rejected so API-facing bridges cannot broaden support accidentally. */
enum agx_isa_encoding
agx3_continue_ray_witnessed_chip_encoding(enum agx_chip chip);

/*
 * Encode only the exact first hardware `continue_ray` instruction
 * witnessed inside Apple's compiler lowering.  The pinned compiler emits the
 * six-byte AGX3b specimen on supported G15/G16/G17P targets and the distinct
 * eight-byte AGX3c specimen on G17G/G17S.  A forced-live 7-operand matrix
 * reproduces these same bytes without correlating any intrinsic operand to a
 * hardware field, so this API deliberately exposes no operands or anonymous
 * fields.
 *
 * The fixed-first-site-witness name is a hard boundary.  This instruction is
 * only one specimen inside a much larger compiler expansion: it excludes the
 * preceding barrier, the second `continue_ray`, shared-memory operations,
 * control flow, and generation-specific scheduling.  It must not be treated
 * as a standalone traversal operation or used for intrinsic lowering,
 * scheduling, or submission.  It does not enable codegen, runtime, or a device
 * target.  Every other instruction prefix, including the compiler's
 * second-site witness, fails closed.
 */
bool agx3_pack_continue_ray_fixed_first_site_witness(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_continue_ray_fixed_first_site_witness *out);

/*
 * Encode only the exact eight-byte `incStk` instruction observed once
 * in each pinned rt_rayquery_committed native binary.  Apple's tables decode
 * the sole nonzero operand as f24=4, but the corpus contains no value ladder
 * or direct differential for that field.  It is therefore deliberately not
 * exposed as an operand and the traversal-stack interpretation of the
 * mnemonic remains outside this byte-level contract.
 *
 * The encoding-level codec contains the exact AGX3b and AGX3c witnesses.
 * agx3_inc_stk_witnessed_chip_encoding() separately admits only scoped
 * M3/M4/A18 Pro/M5 identities that reproduce them.  Same-encoding identities
 * without a scoped corpus, as well as the ungrounded G16P marketing identity,
 * remain fail-closed at target-facing bridges.  This fixed witness supplies
 * no surrounding lowering, SCS layout, barriers, scheduling, command
 * emission, submission, device admission, or API capability.
 */
bool agx3_pack_inc_stk_fixed_witness(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_inc_stk_fixed_witness *out);

enum agx_isa_encoding
agx3_inc_stk_witnessed_chip_encoding(enum agx_chip chip);

#ifdef __cplusplus
}
#endif
