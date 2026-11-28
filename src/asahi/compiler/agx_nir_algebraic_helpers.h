/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_search.h"

/* nir_algebraic condition: the source is the scalar result of an
 * intrinsic, such as a single value loaded from memory, rather than one lane
 * of a vector (a four-byte load is one word on AGX3B, fold_w8_word_unpack())
 * or a value taken apart from a wider one by ALU instructions. */
static inline bool
is_scalar_intrinsic_result(UNUSED const nir_search_state *state,
                           const nir_alu_instr *instr, unsigned src,
                           UNUSED unsigned num_components,
                           const uint8_t *swizzle)
{
   nir_scalar s = nir_scalar_chase_movs(
      nir_get_scalar(instr->src[src].src.ssa, swizzle[0]));
   return nir_scalar_is_intrinsic(s) && s.def->num_components == 1;
}

/* True if every bit set in `mask` (which must be a contiguous low-bit mask
 * 2^k - 1) is provably zero in scalar `s`. Used to drop aligned addends under
 * low-bit masks: (a + b) & (2^k - 1) == b & (2^k - 1). */
static inline bool
agx_low_bits_zero(nir_scalar s, uint64_t mask, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s))
      return (nir_scalar_as_uint(s) & mask) == 0;
   if (depth == 0 || !nir_scalar_is_alu(s))
      return false;

   switch (nir_scalar_alu_op(s)) {
   case nir_op_iand:
      return agx_low_bits_zero(nir_scalar_chase_alu_src(s, 0), mask, depth - 1) ||
             agx_low_bits_zero(nir_scalar_chase_alu_src(s, 1), mask, depth - 1);
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_iadd:
   case nir_op_isub:
      return agx_low_bits_zero(nir_scalar_chase_alu_src(s, 0), mask, depth - 1) &&
             agx_low_bits_zero(nir_scalar_chase_alu_src(s, 1), mask, depth - 1);
   case nir_op_imul:
      return agx_low_bits_zero(nir_scalar_chase_alu_src(s, 0), mask, depth - 1) ||
             agx_low_bits_zero(nir_scalar_chase_alu_src(s, 1), mask, depth - 1);
   case nir_op_ishl: {
      nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (!nir_scalar_is_const(sh))
         return false;
      uint64_t shift = nir_scalar_as_uint(sh) & (s.def->bit_size - 1);
      return shift > 0 && (((UINT64_C(1) << shift) - 1) & mask) == mask;
   }
   case nir_op_imadshl_agx:
   case nir_op_imsubshl_agx: {
      bool mul_zero =
         agx_low_bits_zero(nir_scalar_chase_alu_src(s, 0), mask, depth - 1) ||
         agx_low_bits_zero(nir_scalar_chase_alu_src(s, 1), mask, depth - 1);
      if (!mul_zero)
         return false;
      nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 3));
      if (nir_scalar_is_const(sh)) {
         uint64_t shift = nir_scalar_as_uint(sh) & (s.def->bit_size - 1);
         if (shift > 0 && (((UINT64_C(1) << shift) - 1) & mask) == mask)
            return true;
      }
      return agx_low_bits_zero(nir_scalar_chase_alu_src(s, 2), mask, depth - 1);
   }
   default:
      return false;
   }
}

static inline bool
agx_src_low_bits_zero(const nir_alu_instr *instr, unsigned src,
                      const uint8_t *swizzle, uint64_t mask)
{
   return agx_low_bits_zero(
      nir_get_scalar(instr->src[src].src.ssa, swizzle[0]), mask, 6);
}

static inline bool
agx_is_multiple_of_2(UNUSED const nir_search_state *state,
                     const nir_alu_instr *instr, unsigned src,
                     UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 1);
}

static inline bool
agx_is_multiple_of_4(UNUSED const nir_search_state *state,
                     const nir_alu_instr *instr, unsigned src,
                     UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 3);
}

static inline bool
agx_is_multiple_of_8(UNUSED const nir_search_state *state,
                     const nir_alu_instr *instr, unsigned src,
                     UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 7);
}

static inline bool
agx_is_multiple_of_16(UNUSED const nir_search_state *state,
                      const nir_alu_instr *instr, unsigned src,
                      UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 15);
}

static inline bool
agx_is_multiple_of_32(UNUSED const nir_search_state *state,
                      const nir_alu_instr *instr, unsigned src,
                      UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 31);
}

static inline bool
agx_is_multiple_of_64(UNUSED const nir_search_state *state,
                      const nir_alu_instr *instr, unsigned src,
                      UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 63);
}

static inline bool
agx_is_multiple_of_128(UNUSED const nir_search_state *state,
                       const nir_alu_instr *instr, unsigned src,
                       UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 127);
}

static inline bool
agx_is_multiple_of_256(UNUSED const nir_search_state *state,
                       const nir_alu_instr *instr, unsigned src,
                       UNUSED unsigned num_components, const uint8_t *swizzle)
{
   return agx_src_low_bits_zero(instr, src, swizzle, 255);
}

static inline bool
agx_is_non_byte_mask16(UNUSED const nir_search_state *state,
                       const nir_alu_instr *instr, unsigned src,
                       unsigned num_components, const uint8_t *swizzle)
{
   for (unsigned i = 0; i < num_components; i++) {
      nir_scalar s = nir_scalar_chase_movs(
         nir_get_scalar(instr->src[src].src.ssa, swizzle[i]));
      if (!nir_scalar_is_const(s))
         return false;
      uint64_t m = nir_scalar_as_uint(s) & 0xffff;
      /* Leave 0xff for the 1-instruction AGX3C AGX_CONVERT_U8_TO_F path. */
      if (m == 0 || m == 0xff || m == 0xffff)
         return false;
   }
   return true;
}

static inline bool
agx_is_shift_1_to_15(UNUSED const nir_search_state *state,
                     const nir_alu_instr *instr, unsigned src,
                     unsigned num_components, const uint8_t *swizzle)
{
   for (unsigned i = 0; i < num_components; i++) {
      nir_scalar s = nir_scalar_chase_movs(
         nir_get_scalar(instr->src[src].src.ssa, swizzle[i]));
      if (!nir_scalar_is_const(s))
         return false;
      uint64_t sh = nir_scalar_as_uint(s);
      if (sh < 1 || sh > 15)
         return false;
   }
   return true;
}

static inline bool
agx_is_top_bits_ubfe16(const nir_alu_instr *instr)
{
   if (instr->op != nir_op_ubitfield_extract || instr->def.num_components != 1)
      return false;
   nir_scalar off = nir_scalar_chase_movs(
      nir_get_scalar(instr->src[1].src.ssa, instr->src[1].swizzle[0]));
   nir_scalar bits = nir_scalar_chase_movs(
      nir_get_scalar(instr->src[2].src.ssa, instr->src[2].swizzle[0]));
   if (!nir_scalar_is_const(off) || !nir_scalar_is_const(bits))
      return false;
   uint64_t o = nir_scalar_as_uint(off);
   uint64_t b = nir_scalar_as_uint(bits);
   return o >= 1 && o <= 15 && b >= 1 && o + b >= 16 && o + b <= 32;
}

static inline bool
agx_extract_iand_const(nir_scalar s, nir_scalar *base_out, uint64_t *mask_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_iand)
      return false;
   nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (nir_scalar_is_const(s1)) {
      *base_out = s0;
      *mask_out = nir_scalar_as_uint(s1);
      return true;
   }
   if (nir_scalar_is_const(s0)) {
      *base_out = s1;
      *mask_out = nir_scalar_as_uint(s0);
      return true;
   }
   return false;
}

static inline bool
agx_iadd_disjoint_masks(const nir_alu_instr *instr)
{
   if (instr->op != nir_op_iadd || instr->def.num_components != 1)
      return false;
   nir_scalar x0, x1;
   uint64_t m0 = 0, m1 = 0;
   if (!agx_extract_iand_const(
          nir_get_scalar(instr->src[0].src.ssa, instr->src[0].swizzle[0]),
          &x0, &m0) ||
       !agx_extract_iand_const(
          nir_get_scalar(instr->src[1].src.ssa, instr->src[1].swizzle[0]),
          &x1, &m1))
      return false;
   return nir_scalar_equal(x0, x1) && (m0 & m1) == 0;
}

static inline bool
agx_iadd_disjoint_masks_nested(const nir_alu_instr *instr)
{
   if (instr->op != nir_op_iadd || instr->def.num_components != 1)
      return false;
   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar x0;
      uint64_t m0 = 0;
      nir_scalar s_and = nir_get_scalar(instr->src[side].src.ssa,
                                        instr->src[side].swizzle[0]);
      nir_scalar s_add = nir_scalar_chase_movs(nir_get_scalar(
         instr->src[1 - side].src.ssa, instr->src[1 - side].swizzle[0]));
      if (!agx_extract_iand_const(s_and, &x0, &m0))
         continue;
      if (!nir_scalar_is_alu(s_add) || nir_scalar_alu_op(s_add) != nir_op_iadd)
         continue;
      bool any_match = false;
      bool all_disjoint = true;
      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar x1;
         uint64_t m1 = 0;
         if (agx_extract_iand_const(nir_scalar_chase_alu_src(s_add, k), &x1,
                                    &m1) &&
             nir_scalar_equal(x0, x1)) {
            any_match = true;
            if ((m0 & m1) != 0)
               all_disjoint = false;
         }
      }
      if (any_match && all_disjoint)
         return true;
   }
   return false;
}
