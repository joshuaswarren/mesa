/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_intrinsics.h"
#include "nir_opcodes.h"

struct match {
   nir_scalar base, offset;
   bool sign_extend;
   uint8_t shift;
   int16_t byte_offset;
};

struct lower_address_options {
   bool byte_indices;
   bool byte_offsets;
   bool checked_offsets;
   nir_shader *shader;
   struct hash_table *ranges;
};

/* Whether value = x << amount (or x * 2^amount), a 32-bit result, equals x
 * scaled by 2^amount in 64 bits after the access's extension: the product
 * must not wrap, unsigned for a zero-extended index or signed for a
 * sign-extended one. Either the instruction says so, or x is small enough.
 *
 * Where offsets are bounds-checked in software, the check tests value as it
 * is, wrapped or not, and the access must go where the check looked. A no-wrap
 * flag comes from the shader (SPIR-V NoUnsignedWrap/NoSignedWrap), and a
 * shader that breaks that promise must still not reach past its buffers, so
 * there only the range of x counts.
 */
static bool
scale_is_exact(const struct lower_address_options *options, nir_scalar value,
               nir_scalar x, unsigned amount, bool sign_extend)
{
   if (amount == 0)
      return true;

   nir_alu_instr *alu = nir_def_as_alu(value.def);
   if (!options->checked_offsets &&
       (sign_extend ? alu->no_signed_wrap : alu->no_unsigned_wrap))
      return true;

   /* A nonnegative x below 2^(31 - amount) also has no signed wrap. */
   const uint32_t limit = (sign_extend ? INT32_MAX : UINT32_MAX) >> amount;
   return nir_unsigned_upper_bound(options->shader, options->ranges, x) <=
          limit;
}

/* Take value = x << amount as an index at scale amount: x itself where that
 * is exact, and otherwise value >> amount, which is: the low amount bits of
 * value are zero, and a logical shift (arithmetic for a sign-extended index)
 * gives back ext(value) once the access scales it again. That costs one
 * 32-bit shift, shared by every access to value, instead of adding the
 * address in 64-bit ALU.
 */
static void
take_scaled(nir_builder *b, const struct lower_address_options *options,
            nir_scalar value, nir_scalar x, unsigned amount, bool sign_extend,
            nir_scalar *offset, uint8_t *shift)
{
   if (scale_is_exact(options, value, x, amount, sign_extend)) {
      *offset = x;
   } else {
      nir_def *v = nir_channel(b, value.def, value.comp);
      *offset = nir_get_scalar(
         sign_extend ? nir_ishr_imm(b, v, amount) : nir_ushr_imm(b, v, amount),
         0);
   }
   *shift = amount;
}

static void
peel_scaled_u32(nir_builder *b, const struct lower_address_options *options,
                nir_scalar *offset, uint8_t *shift, unsigned max_shift,
                bool sign_extend)
{
   nir_scalar value = nir_scalar_chase_movs(*offset);
   if (!nir_scalar_is_alu(value))
      return;

   const nir_op op = nir_scalar_alu_op(value);
   if (op == nir_op_ishl) {
      nir_scalar amount = nir_scalar_chase_alu_src(value, 1);
      if (nir_scalar_is_const(amount) &&
          nir_scalar_as_uint(amount) <= max_shift)
         take_scaled(b, options, value, nir_scalar_chase_alu_src(value, 0),
                     nir_scalar_as_uint(amount), sign_extend, offset, shift);
      return;
   }

   if (op != nir_op_imul)
      return;

   for (unsigned i = 0; i < 2; i++) {
      nir_scalar factor = nir_scalar_chase_alu_src(value, i);
      if (!nir_scalar_is_const(factor))
         continue;

      const uint64_t raw = nir_scalar_as_uint(factor);
      if (!util_is_power_of_two_nonzero(raw))
         continue;

      const unsigned amount = util_logbase2(raw);
      if (amount <= max_shift)
         take_scaled(b, options, value, nir_scalar_chase_alu_src(value, 1 - i),
                     amount, sign_extend, offset, shift);
      return;
   }
}

static nir_scalar
strip_zero_shift(nir_scalar value)
{
   value = nir_scalar_chase_movs(value);
   while (nir_scalar_is_alu(value) &&
          nir_scalar_alu_op(value) == nir_op_ishl) {
      nir_scalar amount = nir_scalar_chase_alu_src(value, 1);
      if (!nir_scalar_is_const(amount) || nir_scalar_as_uint(amount) != 0)
         break;
      value = nir_scalar_chase_alu_src(value, 0);
   }
   return value;
}

/* Recover the native 32-bit index from an address-sized zero extension.
 * agx_nir_fuse_lea may wrap this in a 64-bit LEA before this pass gets to
 * select the load's base-plus-index addressing mode.
 */
static bool
peel_u2u64_index(nir_builder *b, const struct lower_address_options *options,
                 nir_scalar wide, nir_scalar *narrow, uint8_t *shift,
                 unsigned max_shift)
{
   wide = strip_zero_shift(wide);
   if (!nir_scalar_is_alu(wide) || nir_scalar_alu_op(wide) != nir_op_u2u64)
      return false;

   *narrow = nir_scalar_chase_alu_src(wide, 0);
   if (narrow->def->bit_size != 32)
      return false;

   peel_scaled_u32(b, options, narrow, shift, max_shift, false);
   return true;
}

/* An index o = x << a whose shift is beyond the access's scale (a vec4 of
 * words at x * 128, say) is not taken as an index, and the access then adds
 * the whole 64-bit address in ALU. Since the low a bits of o are zero, index
 * with o >> (max_shift - shift) at the full scale instead: one 32-bit shift,
 * shared by every access to o, in place of a 64-bit add per access.
 *
 * The extension happens to o itself, so shift o rather than x: o wraps at
 * 2^32 (and a signed o changes sign at 2^31) exactly where the shader's own
 * 32-bit arithmetic does, and a logical shift for a zero-extended index or an
 * arithmetic one for a sign-extended index gives back ext(o) once the access
 * scales it by max_shift - shift again.
 */
static bool
rescale_wide_shift(nir_builder *b, nir_scalar *offset, unsigned *shift,
                   unsigned max_shift, bool sign_extend)
{
   nir_scalar value = nir_scalar_chase_movs(*offset);
   if (value.def->bit_size != 32 || !nir_scalar_is_alu(value) ||
       nir_scalar_alu_op(value) != nir_op_ishl)
      return false;

   nir_scalar amount = nir_scalar_chase_alu_src(value, 1);
   if (!nir_scalar_is_const(amount))
      return false;

   const unsigned a = nir_scalar_as_uint(amount) & 31;
   if (*shift + a <= max_shift || *shift >= max_shift)
      return false;

   nir_def *o = nir_channel(b, value.def, value.comp);
   const unsigned rest = max_shift - *shift;
   nir_def *index =
      sign_extend ? nir_ishr_imm(b, o, rest) : nir_ushr_imm(b, o, rest);
   *offset = nir_get_scalar(index, 0);
   *shift = max_shift;
   return true;
}

/* nir_lower_explicit_io scales each array index by its stride with amul. A
 * valid access has a nonnegative array index, so every such product is
 * nonnegative, and since UBOs and SSBOs are limited to INT32_MAX bytes their
 * sum stays below 2^31 (see the relaxed lea rules in agx_nir_algebraic.py).
 * With checked offsets no amul is left (see agx_nir_lower_address), since an
 * access a software bounds check let through need not be valid.
 */
static bool
is_amul_sum(nir_scalar s, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (depth > 8 || !nir_scalar_is_alu(s))
      return false;

   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_amul)
      return true;

   return op == nir_op_iadd &&
          is_amul_sum(nir_scalar_chase_alu_src(s, 0), depth + 1) &&
          is_amul_sum(nir_scalar_chase_alu_src(s, 1), depth + 1);
}

static bool
instr_dominates(nir_instr *a, nir_instr *b)
{
   if (a->block != b->block)
      return nir_block_dominates(a->block, b->block);

   for (nir_instr *it = a; it != NULL; it = nir_instr_next(it)) {
      if (it == b)
         return true;
   }

   return false;
}

/* Whether value is an index some memory access is addressed with */
static bool
used_as_index(nir_def *value)
{
   nir_foreach_use(use, value) {
      nir_instr *user = nir_src_use_instr(use);
      if (user->type == nir_instr_type_alu) {
         nir_op op = nir_instr_as_alu(user)->op;
         if (op == nir_op_ulea_agx || op == nir_op_ilea_agx)
            return true;
      } else if (user->type == nir_instr_type_intrinsic) {
         switch (nir_instr_as_intrinsic(user)->intrinsic) {
         case nir_intrinsic_load_agx:
         case nir_intrinsic_load_constant_agx:
         case nir_intrinsic_store_agx:
            return true;
         default:
            break;
         }
      }
   }

   return false;
}

/*
 * The index scale has room for up to two doublings beyond the element size,
 * and the bounds that let a power-of-two stride move into the scale hold for
 * any power-of-two factor of the stride. So an access to an array element at
 * index * stride may as well use index * (stride >> t), scaled by t more:
 * accesses to one element at different alignments then share one index
 * register. Only take an index * (stride >> t) that the program computes
 * before this access to address memory with already. A new one would be
 * merged with any equal value, and another value could be one computed far
 * away for another purpose, which would then stay live all the way here.
 */
static bool
existing_index_multiple(nir_intrinsic_instr *access, nir_scalar index,
                        uint64_t stride, unsigned max_t, unsigned *t_out,
                        nir_scalar *out)
{
   for (unsigned t = MIN2(max_t, ffsll(stride) - 1); t > 0; --t) {
      const uint64_t k = stride >> t;

      if (k == 1) {
         *t_out = t;
         *out = index;
         return true;
      }

      nir_foreach_use(use, index.def) {
         nir_instr *user = nir_src_use_instr(use);
         if (user->type != nir_instr_type_alu)
            continue;

         nir_alu_instr *alu = nir_instr_as_alu(user);
         if ((alu->op != nir_op_amul && alu->op != nir_op_imul) ||
             alu->def.num_components != 1 || alu->def.bit_size != 32)
            continue;

         nir_scalar mul = nir_get_scalar(&alu->def, 0);
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar x = nir_scalar_chase_alu_src(mul, i);
            nir_scalar c = nir_scalar_chase_alu_src(mul, 1 - i);
            if (x.def == index.def && x.comp == index.comp &&
                nir_scalar_is_const(c) && nir_scalar_as_uint(c) == k &&
                used_as_index(&alu->def) &&
                instr_dominates(user, &access->instr)) {
               *t_out = t;
               *out = mul;
               return true;
            }
         }
      }
   }

   return false;
}

/* AGX3C loads and stores add a signed 16-bit byte offset after scaling the
 * index, so a constant addend in the index -- a struct member or a constant
 * array element -- needs no ALU. The split must not change any valid address:
 *
 * - A sign-extended index gives sext(x + c) = sext(x) + c unless x + c
 *   overflows, which no valid signed index does.
 * - A zero-extended index gives zext(x + c) = zext(x) + c only if x + c does
 *   not wrap. An amul sum x is nonnegative and below 2^31 for a valid access,
 *   so this holds for c >= 0. Any other x may be negative (a reversed index
 *   N - i folds to c + -i), so leave it alone.
 * - An array index i + c scaled by amul is, for a valid access, nonnegative
 *   and below 2^31 / stride, so as signed integers (i + c) * stride equals
 *   sext(i) * stride + c * stride. The index becomes the 32-bit i * stride,
 *   which for c < 0 exceeds the valid offset by up to 32 KiB (the byte offset's
 *   range) and so can pass 2^31 in the last 32 KiB of a 2 GiB buffer, at any
 *   stride. Only where the access then scales the index (a 16-bit or wider
 *   element) is that product below 2^30 + 2^14, so require c >= 0 at a byte
 *   scale.
 *
 * The same bounds let a power-of-two stride move into the index scale.
 */
static void
fold_byte_offset(nir_builder *b, nir_intrinsic_instr *access,
                 struct match *match, unsigned format_shift,
                 unsigned max_shift)
{
   if (match->offset.def == NULL)
      return;

   const bool bytes = match->shift == AGX_NIR_MEMORY_INDEX_BYTES;
   const unsigned scale = bytes ? 0 : format_shift + match->shift;
   nir_scalar offset = nir_scalar_chase_movs(match->offset);
   int64_t byte_offset = 0;
   bool sign_extend = match->sign_extend;

   if (offset.def->bit_size != 32 || !nir_scalar_is_alu(offset))
      return;

   if (nir_scalar_alu_op(offset) == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar k = nir_scalar_chase_alu_src(offset, i);
         nir_scalar rest = nir_scalar_chase_alu_src(offset, 1 - i);
         if (!nir_scalar_is_const(k) || nir_scalar_is_const(rest))
            continue;

         const int64_t c = nir_scalar_as_int(k);
         if (!sign_extend && (c < 0 || !is_amul_sum(rest, 0)))
            return;

         byte_offset = c * (INT64_C(1) << scale);
         offset = nir_scalar_chase_movs(rest);
         break;
      }
   }

   /* amul(i + c, stride) or amul(i, stride) */
   nir_scalar index = offset;
   uint64_t stride = 0;
   if (nir_scalar_is_alu(offset) && nir_scalar_alu_op(offset) == nir_op_amul) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar d = nir_scalar_chase_alu_src(offset, i);
         if (nir_scalar_is_const(d) && nir_scalar_as_uint(d) <= 0x10000) {
            stride = nir_scalar_as_uint(d);
            index = nir_scalar_chase_movs(nir_scalar_chase_alu_src(offset, 1 - i));
            break;
         }
      }
   }

   bool index_folded = false;
   if (stride && nir_scalar_is_alu(index) &&
       nir_scalar_alu_op(index) == nir_op_iadd) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar k = nir_scalar_chase_alu_src(index, i);
         nir_scalar rest = nir_scalar_chase_alu_src(index, 1 - i);
         if (!nir_scalar_is_const(k) || nir_scalar_is_const(rest))
            continue;

         const int64_t c = nir_scalar_as_int(k);
         if (c < 0 && scale == 0)
            break;

         byte_offset += c * (int64_t)stride * (INT64_C(1) << scale);
         index = nir_scalar_chase_movs(rest);
         sign_extend = true;
         index_folded = true;
         break;
      }
   }

   if (byte_offset < INT16_MIN || byte_offset > INT16_MAX)
      return;

   unsigned t = 0;
   nir_scalar multiple;
   const bool rescale =
      stride && !bytes && (index_folded || !sign_extend) &&
      existing_index_multiple(access, index, stride, max_shift - scale, &t,
                              &multiple);

   if (rescale) {
      /* Either the array index lost its constant and what is left must be
       * rescaled, or an unchanged amul is nonnegative and does not wrap.
       */
      match->shift += t;
      match->offset = multiple;
   } else if (index_folded) {
      nir_def *x = nir_channel(b, index.def, index.comp);
      match->offset = nir_get_scalar(nir_imul_imm(b, x, stride), 0);
   } else {
      match->offset = offset;
   }

   match->byte_offset = byte_offset;
   match->sign_extend = sign_extend;
}

static enum pipe_format
format_for_bitsize(unsigned bitsize)
{
   switch (bitsize) {
   case 8:
      return PIPE_FORMAT_R8_UINT;
   case 16:
      return PIPE_FORMAT_R16_UINT;
   case 32:
      return PIPE_FORMAT_R32_UINT;
   default:
      UNREACHABLE("should have been lowered");
   }
}

static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global &&
       intr->intrinsic != nir_intrinsic_load_global_constant &&
       intr->intrinsic != nir_intrinsic_global_atomic &&
       intr->intrinsic != nir_intrinsic_global_atomic_swap &&
       intr->intrinsic != nir_intrinsic_store_global)
      return false;

   const struct lower_address_options *options = data;
   const bool byte_indices = options->byte_indices;
   b->cursor = nir_before_instr(&intr->instr);

   unsigned bitsize = intr->intrinsic == nir_intrinsic_store_global
                         ? nir_src_bit_size(intr->src[0])
                         : intr->def.bit_size;
   enum pipe_format format = format_for_bitsize(bitsize);
   unsigned format_shift = util_logbase2(util_format_get_blocksize(format));

   nir_src *orig_offset = nir_get_io_offset_src(intr);
   nir_scalar base = nir_scalar_resolved(orig_offset->ssa, 0);
   struct match match = {.base = base};
   bool shift_must_match =
      (intr->intrinsic == nir_intrinsic_global_atomic) ||
      (intr->intrinsic == nir_intrinsic_global_atomic_swap);
   unsigned max_shift = format_shift + (shift_must_match ? 0 : 2);

   if (nir_scalar_is_alu(base)) {
      nir_op op = nir_scalar_alu_op(base);
      if (op == nir_op_ulea_agx || op == nir_op_ilea_agx) {
         unsigned shift = nir_scalar_as_uint(nir_scalar_chase_alu_src(base, 2));
         if (shift <= max_shift) {
            nir_scalar offset = nir_scalar_chase_alu_src(base, 1);
            uint8_t extra_shift = 0;
            if (op != nir_op_ulea_agx ||
                !peel_u2u64_index(b, options, offset, &offset, &extra_shift,
                                  max_shift - shift))
               peel_scaled_u32(b, options, &offset, &extra_shift,
                               max_shift - shift, op == nir_op_ilea_agx);

            unsigned total_shift = shift + extra_shift;
            /* Only an unpeeled offset is the value the access extends. */
            if (options->byte_offsets && !shift_must_match &&
                total_shift < format_shift && extra_shift == 0)
               rescale_wide_shift(b, &offset, &total_shift, max_shift,
                                  op == nir_op_ilea_agx);
            if (total_shift >= format_shift ||
                (byte_indices && !shift_must_match && total_shift == 0 &&
                 op == nir_op_ulea_agx && offset.def->bit_size == 32)) {
               match = (struct match){
                  .base = nir_scalar_chase_alu_src(base, 0),
                  .offset = offset,
                  .shift = total_shift >= format_shift
                              ? total_shift - format_shift
                              : AGX_NIR_MEMORY_INDEX_BYTES,
                  .sign_extend = (op == nir_op_ilea_agx),
               };
            }
         }
      } else if (op == nir_op_iadd) {
         /* A load from base + u2u64(offset32 << shift) maps directly to AGX's
          * 64-bit base plus 32-bit register-index form when the scale covers
          * the element size. Keeping the add in 64-bit ALU form would
          * manufacture a carry chain that this address mode does not need.
          */
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar narrow;
            uint8_t shift = 0;
            if (!peel_u2u64_index(b, options, nir_scalar_chase_alu_src(base, i),
                                  &narrow, &shift, max_shift))
               continue;
            if (options->byte_offsets && !shift_must_match && shift == 0) {
               unsigned wide = shift;
               if (rescale_wide_shift(b, &narrow, &wide, max_shift, false))
                  shift = wide;
            }
            if (shift < format_shift &&
                !(byte_indices && !shift_must_match && shift == 0))
               continue;

            match = (struct match){
               .base = nir_scalar_chase_alu_src(base, 1 - i),
               .offset = narrow,
               .shift = shift >= format_shift
                           ? shift - format_shift
                           : AGX_NIR_MEMORY_INDEX_BYTES,
            };
            break;
         }

         if (match.offset.def != NULL)
            goto matched;

         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar const_scalar = nir_scalar_chase_alu_src(base, i);
            if (!nir_scalar_is_const(const_scalar))
               continue;

            /* Put scalar into form (k*2^n), clamping n at the maximum hardware
             * shift.
             */
            int64_t raw_scalar = nir_scalar_as_uint(const_scalar);
            assert(raw_scalar != 0 && "must have been optimized out");

            uint32_t shift = MIN2(__builtin_ctz(raw_scalar), max_shift);
            int64_t k = raw_scalar >> shift;

            /* See if the reduced scalar is from a sign extension. We must have
             * at least the format shift to avoid underflowing.
             */
            if (k > INT32_MAX || k < INT32_MIN || shift < format_shift)
               break;

            /* Match the constant */
            match = (struct match){
               .base = nir_scalar_chase_alu_src(base, 1 - i),
               .offset = nir_get_scalar(nir_imm_int(b, k), 0),
               .shift = shift - format_shift,
               .sign_extend = true,
            };

            break;
         }
      }
   }

matched:

   if (options->byte_offsets && !shift_must_match &&
       intr->intrinsic != nir_intrinsic_global_atomic &&
       intr->intrinsic != nir_intrinsic_global_atomic_swap) {
      fold_byte_offset(b, intr, &match, format_shift, max_shift);

      /* A constant added to the base, as to a pointer before it is indexed
       * ((p + 16) + i): the byte offset adds it after the index in the same
       * 64-bit sum, so the base needs no add of its own.
       */
      nir_scalar base_now = nir_scalar_chase_movs(match.base);
      if (match.offset.def != NULL && base_now.def->bit_size == 64 &&
          nir_scalar_is_alu(base_now) &&
          nir_scalar_alu_op(base_now) == nir_op_iadd) {
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar k = nir_scalar_chase_alu_src(base_now, i);
            if (!nir_scalar_is_const(k))
               continue;
            /* Compare before adding: the constant is any 64-bit value. */
            const int64_t c = nir_scalar_as_int(k);
            if (c >= INT16_MIN - (int64_t)match.byte_offset &&
                c <= INT16_MAX - (int64_t)match.byte_offset) {
               match.base = nir_scalar_chase_alu_src(base_now, 1 - i);
               match.byte_offset = match.byte_offset + c;
            }
            break;
         }
      }
   }

   const unsigned mem_base =
      match.shift | ((uint32_t)(uint16_t)match.byte_offset
                     << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);

   nir_def *offset = match.offset.def != NULL
                        ? nir_channel(b, match.offset.def, match.offset.comp)
                        : nir_imm_int(b, 0);

   nir_def *new_base = nir_channel(b, match.base.def, match.base.comp);

   nir_def *repl = NULL;
   bool has_dest = (intr->intrinsic != nir_intrinsic_store_global);
   unsigned num_components = has_dest ? intr->def.num_components : 0;
   unsigned bit_size = has_dest ? intr->def.bit_size : 0;

   if (intr->intrinsic == nir_intrinsic_load_global) {
      repl =
         nir_load_agx(b, num_components, bit_size, new_base, offset,
                      .access = nir_intrinsic_access(intr), .base = mem_base,
                      .format = format, .sign_extend = match.sign_extend);

   } else if (intr->intrinsic == nir_intrinsic_load_global_constant) {
      repl = nir_load_constant_agx(b, num_components, bit_size, new_base,
                                   offset, .access = nir_intrinsic_access(intr),
                                   .base = mem_base, .format = format,
                                   .sign_extend = match.sign_extend);
   } else if (intr->intrinsic == nir_intrinsic_global_atomic) {
      repl =
         nir_global_atomic_agx(b, bit_size, new_base, offset, intr->src[1].ssa,
                               .access = nir_intrinsic_access(intr),
                               .atomic_op = nir_intrinsic_atomic_op(intr),
                               .sign_extend = match.sign_extend);
   } else if (intr->intrinsic == nir_intrinsic_global_atomic_swap) {
      repl = nir_global_atomic_swap_agx(
         b, bit_size, new_base, offset, intr->src[1].ssa, intr->src[2].ssa,
         .access = nir_intrinsic_access(intr),
         .atomic_op = nir_intrinsic_atomic_op(intr),
         .sign_extend = match.sign_extend);
   } else {
      nir_store_agx(b, intr->src[0].ssa, new_base, offset,
                    .access = nir_intrinsic_access(intr), .base = mem_base,
                    .format = format, .sign_extend = match.sign_extend);
   }

   if (repl)
      nir_def_rewrite_uses(&intr->def, repl);

   nir_instr_remove(&intr->instr);
   return true;
}

/* amul promises that its product does not wrap, which only a valid access
 * keeps. A software bounds check tests the wrapped 32-bit offset instead, and
 * an index such as 0x40000001 into a uint[] passes it at offset 4. Where
 * offsets are checked, make every 32-bit amul the multiply the check saw, so
 * that the relaxed lea rules and the folds that rest on them cannot take the
 * unwrapped index: the access then stays at base + the checked offset. The
 * exact rules for imul and ishl (scale_is_exact) still take the index itself
 * wherever its range proves the product cannot wrap.
 */
static bool
amul_to_imul(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_amul || alu->def.bit_size != 32)
      return false;

   alu->op = nir_op_imul;
   return true;
}

bool
agx_nir_lower_address(nir_shader *nir, bool byte_indices, bool byte_offsets,
                      bool checked_offsets)
{
   struct lower_address_options options = {
      .byte_indices = byte_indices,
      .byte_offsets = byte_offsets,
      .checked_offsets = checked_offsets,
      .shader = nir,
   };
   bool progress = false;

   if (checked_offsets) {
      NIR_PASS(progress, nir, nir_shader_alu_pass, amul_to_imul,
               nir_metadata_control_flow, NULL);
   }

   /* First, clean up as much as possible. This will make fusing more effective.
    */
   do {
      progress = false;
      NIR_PASS(progress, nir, agx_nir_cleanup_amul);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   /* Then, fuse as many lea as possible */
   NIR_PASS(progress, nir, agx_nir_fuse_lea);

   /* Next, lower load/store using the lea's. nir_unsigned_upper_bound()
    * caches by SSA index, which every implementation numbers from zero, so
    * each implementation gets a table of its own.
    */
   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_dominance);
      options.ranges = _mesa_pointer_hash_table_create(NULL);
      progress |= nir_function_intrinsics_pass(
         impl, pass, nir_metadata_control_flow, (void *)&options);
      _mesa_hash_table_destroy(options.ranges, NULL);
   }

   /* Finally, lower any leftover lea instructions back to ALU to let
    * nir_opt_algebraic simplify them from here.
    */
   NIR_PASS(progress, nir, agx_nir_lower_lea);
   NIR_PASS(progress, nir, nir_opt_dce);

   return progress;
}
