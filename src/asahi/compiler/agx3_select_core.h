/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Standalone AGX3 instruction-selection core for the no-modifier 32-bit ALU,
 * move, immediate, and nullary terminator subset over normalized operands.
 * Unmapped opcodes, modifiers, non-32-bit sizes, and unsupported operand kinds
 * fail closed.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The AGX IR operations milestone one selects. The compiler shim maps
 * AGX_OPCODE_* onto these; the core never sees the generated opcode enum. */
enum agx3_sel_op {
   AGX3_SEL_OP_UNSUPPORTED = 0,
   AGX3_SEL_OP_END,       /* IR stop / shader terminator */
   AGX3_SEL_OP_BARRIER,   /* IR threadgroup_barrier */
   AGX3_SEL_OP_MOV_IMM,   /* IR mov_imm */
   AGX3_SEL_OP_FADD,      /* IR fadd (f32) */
   AGX3_SEL_OP_FMUL,      /* IR fmul (f32) */
   AGX3_SEL_OP_FFMA,      /* IR ffma (f32) */
   AGX3_SEL_OP_IADD,      /* IR iadd (u32), shift == 0 */
   AGX3_SEL_OP_IMAD,      /* IR imad (u32), shift == 0 */
   AGX3_SEL_OP_ASR,       /* IR asr (i32) -- arithmetic shift right */
   AGX3_SEL_OP_BITOP,     /* IR bitop; truth_table selects and/or/xor/inv */
   AGX3_SEL_OP_SLL,       /* logical shift left  (src0 << src1), from bfi */
   AGX3_SEL_OP_LSR,       /* logical shift right (src0 >> src1), from bfeil */
   AGX3_SEL_OP_F_TO_I32,  /* f32 -> i32, round toward zero */
   AGX3_SEL_OP_F_TO_U32,  /* f32 -> u32, round toward zero */
   AGX3_SEL_OP_I32_TO_F,  /* i32 -> f32, round to nearest even */
   AGX3_SEL_OP_U32_TO_F,  /* u32 -> f32, round to nearest even */
   AGX3_SEL_OP_ISUB,      /* u32 subtract, from iadd with a negated source */
   AGX3_SEL_OP_IMUL,      /* u32 multiply, from imad with a zero addend */
   AGX3_SEL_OP_F_RCP,     /* f32 reciprocal */
   AGX3_SEL_OP_F_RSQRT,   /* f32 reciprocal square root */
   AGX3_SEL_OP_F_RINT,    /* f32 round to nearest even (roundeven) */
   AGX3_SEL_OP_FMIN,      /* f32 minimum, from the fcmpsel min pattern */
   AGX3_SEL_OP_FMAX,      /* f32 maximum, from the fcmpsel max pattern */
};

/* A normalized source operand. `reg` is a 16-bit half-register index, the same
 * unit Apple's register fields use and the unit agx_index.value already
 * carries after RA. `is_reg` is false for immediate / uniform / null sources,
 * which milestone one refuses (a pre-RA legalize pass will materialize them
 * later). abs/neg mirror the IR float modifiers; milestone one refuses any set
 * modifier because the grounded no-modifier packers cannot express them. */
struct agx3_sel_src {
   uint16_t reg;
   bool is_reg;
   bool abs;
   bool neg;
   uint8_t size; /* AGX size: 0 = 16-bit, 1 = 32-bit, 2 = 64-bit */
   /* The register is read again after this instruction. Encoders that carry
    * no lifetime bits ignore it. */
   bool live;
};

struct agx3_sel_instr {
   enum agx3_sel_op op;
   uint8_t nr_srcs;
   bool has_dest;
   uint16_t dest;
   uint8_t dest_size; /* AGX size */
   struct agx3_sel_src src[3];
   uint32_t imm;      /* mov_imm immediate */
   uint32_t truth_table; /* bitop 4-bit truth table (UTIL_LUT2 convention) */
   uint8_t shift;     /* iadd / imad shift; milestone one requires 0 */
   bool saturate;
};

/* bitop truth tables, computed the same way Mesa does (UTIL_LUT2 with a=0xAA,
 * b=0xCC, masked to the low nibble). Milestone one encodes exactly these four;
 * every other truth table -- nand, nor, xnor, and-not, or-not -- fails closed
 * because Apple's inverting/complement bitops are not yet mapped. */
#define AGX3_SEL_TT_AND 0x8u
#define AGX3_SEL_TT_OR  0xEu
#define AGX3_SEL_TT_XOR 0x6u
#define AGX3_SEL_TT_NOT 0x5u /* ~a, independent of b */
#define AGX3_SEL_TT_MOV 0xAu /* a, independent of b -- register move */

/* AGX size code for a 32-bit value; the only width milestone one accepts. */
#define AGX3_SEL_SIZE_32 1u

/* The maximum bytes any milestone-one form emits (AGX3c fma is 16 B). */
#define AGX3_SEL_MAX_LENGTH 16u

/*
 * Encode one selected instruction. On success writes `*len` bytes (<=
 * AGX3_SEL_MAX_LENGTH) into `out` and returns true. Returns false -- writing
 * nothing meaningful -- for an unknown encoding, an unmapped op, or any
 * instruction outside the grounded domain. A false return must fail the
 * shader; it must never be treated as an empty instruction.
 */
bool agx3_select_encode(enum agx_isa_encoding encoding,
                        const struct agx3_sel_instr *instr,
                        uint8_t out[AGX3_SEL_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
