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
   /* A memory-only fence over threadgroup memory: the threadgroup-memory
    * barrier without its rendezvous (see agx3_pack_scoped_barrier()). */
   AGX3_BARRIER_SCOPE_THREADGROUP_FENCE,
   /* AGX3B only: a workgroup barrier over the compute imageblock, between a
    * workgroup's imageblock stores and its block store. */
   AGX3_BARRIER_SCOPE_IMAGEBLOCK,
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

/* Encode register-source f32 simdbcast with immediate lane (0 or 3).
 * Operands are even half-register indices (0..190 on AGX3B, 0..254 on AGX3C).
 */
bool agx3_pack_simd_broadcast_f32_imm(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src, uint8_t lane,
   struct agx3_encoded_simd_broadcast_f32_imm *out);

/* Encode 32-bit register ldimm. `dest` is an even half-register index < 256. */
bool agx3_pack_ldimm_u32(enum agx_isa_encoding encoding, uint16_t dest,
                         uint32_t imm, struct agx3_encoded_ldimm *out);

/* Encode 32-bit register + u8 immediate integer add. Operands are even
 * half-register indices (0..190 on AGX3B, 0..254 on AGX3C).
 */
bool agx3_pack_iadd_u32_imm8(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src, uint8_t imm,
   struct agx3_encoded_iadd_u32_imm8 *out);

/* Encode 32-bit three-register integer add. Operands are even half-register
 * indices (0..190 on AGX3B, 0..254 on AGX3C).
 */
bool agx3_pack_iadd_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_iadd_u32_reg_reg *out);

/* Encode compact 6-byte f32 register-register add (even half-registers 0..254).
 * Preserves physical source operand order.
 */
bool agx3_pack_fadd_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fadd_f32_reg_reg *out);

/* Encode compact 6-byte f32 register-register multiply (even half-registers
 * 0..126).
 */
bool agx3_pack_fmul_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fmul_f32_reg_reg *out);

/* Encode four-register f32 fused multiply-add (even half-registers 0..126).
 * Preserves physical multiplicand order.
 */
bool agx3_pack_fma_f32_reg_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t mul0,
   uint16_t mul1, uint16_t addend,
   struct agx3_encoded_fma_f32_reg_reg_reg *out);

/* Encode register-register f32 minimum via fsel/fseli32 (even half-registers
 * 0..126).
 */
bool agx3_pack_fmin_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fmin_f32_reg_reg *out);

/* Encode register-register f32 maximum via fsel/fseli32 (even half-registers
 * 0..126).
 */
bool agx3_pack_fmax_f32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_fmax_f32_reg_reg *out);

/* Encode signed i32-to-f32 register conversion (even half-registers 0..126). */
bool agx3_pack_i32_to_f32_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_i32_to_f32_reg *out);

/* Encode unsigned u32-to-f32 register conversion (even half-registers 0..126). */
bool agx3_pack_u32_to_f32_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_u32_to_f32_reg *out);

/* Encode scalar f32 round-to-nearest-even (even half-registers 0..126). */
bool agx3_pack_f32_rint_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_rint_reg *out);

/* Encode scalar f32 reciprocal (even half-registers 0..126). */
bool agx3_pack_f32_rcp_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_rcp_reg *out);

/* Encode scalar f32 reciprocal square root (even half-registers 0..126). */
bool agx3_pack_f32_rsqrt_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_rsqrt_reg *out);

/* Encode scalar f32-to-i32/u32 round-toward-zero conversion (even
 * half-registers 0..126).
 */
bool agx3_pack_f32_to_int_rtz_reg(
   enum agx_isa_encoding encoding, enum agx3_f32_to_int_rtz_mode mode,
   uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_to_int_rtz_reg *out);

/* Encode three-register u32 bitwise AND (even half-registers 0..254). */
bool agx3_pack_bit_and_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_and_u32_reg_reg *out);

/* Encode three-register u32 bitwise OR (even half-registers 0..254). */
bool agx3_pack_bit_or_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_or_u32_reg_reg *out);

/* Encode three-register u32 bitwise XOR (even half-registers 0..254). */
bool agx3_pack_bit_xor_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_xor_u32_reg_reg *out);

/* Encode u32 register bitwise NOT (`inv`, even half-registers 0..254). */
bool agx3_pack_bit_not_u32_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src,
   struct agx3_encoded_bit_not_u32_reg *out);

/* Encode three-register u32 subtraction (even half-registers 0..190 on AGX3B,
 * 0..254 on AGX3C).
 */
bool agx3_pack_isub_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_isub_u32_reg_reg *out);

/* Encode three-register u32 variable shift-left (`sll`, even half-registers
 * 0..190 on AGX3B, 0..254 on AGX3C). Does not mask the shift count.
 */
bool agx3_pack_sll_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_sll_u32_reg_reg *out);

/* Encode three-register i32 arithmetic shift-right (`sra`, even half-registers
 * 0..190 on AGX3B, 0..254 on AGX3C). Does not mask the shift count.
 */
bool agx3_pack_asr_i32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_asr_i32_reg_reg *out);

/* Encode three-register u32 logical shift-right (`srl`, even half-registers
 * 0..190 on AGX3B, 0..254 on AGX3C). Does not mask the shift count.
 */
bool agx3_pack_lsr_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_lsr_u32_reg_reg *out);

/* Encode three-register low-32-bit integer multiply (imadd with implicit zero
 * addend; even half-registers 0..190 on AGX3B, 0..254 on AGX3C).
 */
bool agx3_pack_imul_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_imul_u32_reg_reg *out);

/* Encode four-register low-32-bit integer multiply-add (even half-registers
 * 0..190 on AGX3B, 0..254 on AGX3C).
 */
bool agx3_pack_imadd_u32_reg_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t mul0,
   uint16_t mul1, uint16_t addend,
   struct agx3_encoded_imadd_u32_reg_reg_reg *out);

/* Encode a 6-byte workgroup barrier instruction. */
bool agx3_pack_threadgroup_barrier(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_threadgroup_barrier *out);

/* Encode a field-free AGX3 scoped barrier (6 bytes, or 12 bytes for the
 * two-instruction texture memory barrier).
 */
bool agx3_pack_scoped_barrier(enum agx_isa_encoding encoding,
                              enum agx3_barrier_scope scope,
                              struct agx3_encoded_scoped_barrier *out);

/* Encode an AGX3C texture barrier carrying a scoreboard wait mask (slots 1..8
 * in fields 16..23). A zero wait mask matches agx3_pack_scoped_barrier().
 */
bool agx3_pack_scoped_barrier_wait(enum agx_isa_encoding encoding,
                                   enum agx3_barrier_scope scope,
                                   uint16_t wait_mask,
                                   struct agx3_encoded_scoped_barrier *out);

/* Encode the canonical 4-byte unconditional shader terminator (`0e 00 00 00`). */
bool agx3_pack_end(enum agx_isa_encoding encoding,
                   struct agx3_encoded_end *out);

/* Encode hardware `intersect_ray` with state register `f5` (0 or 4 on AGX3B;
 * 0 or 8 on AGX3C). Unsupported values fail closed.
 */
bool agx3_pack_intersect_ray(enum agx_isa_encoding encoding,
                             uint16_t state_register,
                             struct agx3_encoded_intersect_ray *out);

/* Admitted: G15G/S/D, G16G/S, G17P (AGX3B), G17G/S (AGX3C); encoding-equivalent
 * chips without test rows (G16P, G17A, G18G) stay closed. */
enum agx_isa_encoding
agx3_intersect_ray_witnessed_chip_encoding(enum agx_chip chip);

/* Encode hardware `continue_ray` with state register `f5` (even half-registers
 * 0..10 on AGX3B; 0, 2, 8, 10, 12, 16, 18 on AGX3C).
 */
bool agx3_pack_continue_ray(enum agx_isa_encoding encoding,
                            uint16_t state_register,
                            struct agx3_encoded_continue_ray *out);

/* Admitted: G15G/S/D, G16P/G/S, G17P (AGX3B), G17G/S (AGX3C); encoding-equivalent
 * chips without test rows (G17A, G18G) stay closed. */
enum agx_isa_encoding
agx3_continue_ray_witnessed_chip_encoding(enum agx_chip chip);

/* Encode the canonical zero-register `continue_ray` instruction (6 bytes on
 * AGX3B, 8 bytes on AGX3C).
 */
bool agx3_pack_continue_ray_fixed_first_site_witness(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_continue_ray_fixed_first_site_witness *out);

/* Encode the fixed 8-byte `incStk` instruction (f24=4). */
bool agx3_pack_inc_stk_fixed_witness(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_inc_stk_fixed_witness *out);

/* Admitted: G15G/S/D, G16G/S, G17P (AGX3B), G17G/S (AGX3C); encoding-equivalent
 * chips without test rows (G16P, G17A, G18G) stay closed. */
enum agx_isa_encoding
agx3_inc_stk_witnessed_chip_encoding(enum agx_chip chip);

#ifdef __cplusplus
}
#endif
