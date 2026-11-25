/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"
#include "agx_nir.h"
#include "nir.h"

/*
 * Optimize 32-bit bitfield extracts on AGX3 when every consumer accepts a
 * zero-extended 16-bit half natively (iadd, imul, imadshl_agx, u2u16/u2u8):
 *
 * 1. Shared shift for paired low/high-half extracts from the same 32-bit word:
 *       lo = (p >> k) & mask1           (1 <= k <= 15, 1 <= w1 <= 8)
 *       hi = (p >> (k + 16)) [& mask2]  (full remaining bits or 1 <= w2 <= 8)
 *    becomes:
 *       q  = p >> k                     (1 IC op instead of 2)
 *       lo = u2u32(unpack_x(q) & mask1) (1 SCIB op, 0 IC)
 *       hi = u2u32(unpack_y(q) [& mask2]) (0 or 1 SCIB op, 0 IC)
 *
 * 2. High-half small-mask extract:
 *       (p >> 16) & ((1 << w) - 1)      (1 <= w <= 8)
 *    becomes:
 *       u2u32(unpack_y(p) & ((1 << w) - 1)) (1 SCIB op, 0 IC instead of bfeil)
 *
 * Keeping the mask at 16 bits avoids agx_nir_fold_extract16_bfe turning it
 * back into an IC-pipe bfeil while preserving single-instruction bfeil when
 * the result feeds non-widening consumers.
 *
 * p is one 32-bit component, of a scalar or of a vector: once
 * nir_opt_load_store_vectorize has merged a run of word loads (a row of
 * packed RGBA8 texels read from shared memory, say), each word is a channel
 * of one vec4 load and every extract reads it through a swizzle.
 */

static bool
is_small_bitmask(uint64_t mask, unsigned *bits_out)
{
   if (mask == 0 || mask > 0xff)
      return false;
   if (!util_is_power_of_two_nonzero(mask + 1))
      return false;
   *bits_out = util_bitcount64(mask);
   return true;
}

static bool
all_uses_fold_u16_zext(const nir_def *def)
{
   if (list_is_empty(&def->uses))
      return false;

   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use))
         return false;

      nir_instr *user = nir_src_use_instr(use);
      if (user->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *alu = nir_instr_as_alu(user);
      if (alu->def.num_components != 1)
         return false;

      switch (alu->op) {
      case nir_op_u2u8:
      case nir_op_u2u16:
         break;

      case nir_op_iadd: {
         if (alu->def.bit_size != 32)
            return false;
         unsigned idx = (use == &alu->src[0].src) ? 0 : 1;
         nir_src *other = &alu->src[1 - idx].src;
         /* Small immediate addends (<= 255) are folded into iadd imm forms
          * which agx_lower_agx3_widening_int widens; larger constants become
          * 32-bit registers and keep the native widening register add. */
         if (nir_src_is_const(*other) && nir_src_as_uint(*other) <= 255)
            return false;
         break;
      }

      case nir_op_imul:
         if (alu->def.bit_size != 32)
            return false;
         break;

      case nir_op_imadshl_agx:
         if (alu->def.bit_size != 32)
            return false;
         if (use != &alu->src[0].src && use != &alu->src[1].src)
            return false;
         if (!nir_src_is_const(alu->src[3].src) ||
             nir_src_as_uint(alu->src[3].src) != 0)
            return false;
         break;

      default:
         return false;
      }
   }

   return true;
}

static bool
is_word(nir_scalar s)
{
   return s.def->bit_size == 32;
}

static bool
match_lo_shifted_extract(nir_instr *instr, nir_scalar *src_out,
                         unsigned *shift_out, unsigned *bits_out)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;

   if (alu->op == nir_op_ubitfield_extract) {
      nir_scalar s0 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 1);
      nir_scalar s2 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 2);
      if (!is_word(s0) || !nir_scalar_is_const(s1) ||
          !nir_scalar_is_const(s2))
         return false;
      unsigned k = nir_scalar_as_uint(s1);
      unsigned w = nir_scalar_as_uint(s2);
      if (k >= 1 && k <= 15 && w >= 1 && w <= 8) {
         *src_out = s0;
         *shift_out = k;
         *bits_out = w;
         return true;
      }
      return false;
   }

   if (alu->op == nir_op_extract_u8) {
      nir_scalar s0 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 1);
      if (is_word(s0) && nir_scalar_is_const(s1) &&
          nir_scalar_as_uint(s1) == 1) {
         *src_out = s0;
         *shift_out = 8;
         *bits_out = 8;
         return true;
      }
      return false;
   }

   if (alu->op == nir_op_iand) {
      nir_scalar root = nir_get_scalar(&alu->def, 0);
      for (unsigned c = 0; c < 2; ++c) {
         nir_scalar mc = nir_scalar_chase_alu_src(root, c);
         unsigned w = 0;
         if (!nir_scalar_is_const(mc) ||
             !is_small_bitmask(nir_scalar_as_uint(mc), &w))
            continue;

         nir_scalar sh = nir_scalar_chase_alu_src(root, 1 - c);
         if (!nir_scalar_is_alu(sh) || nir_scalar_alu_op(sh) != nir_op_ushr ||
             !list_is_singular(&sh.def->uses))
            continue;

         nir_scalar p = nir_scalar_chase_alu_src(sh, 0);
         nir_scalar ks = nir_scalar_chase_alu_src(sh, 1);
         if (!is_word(p) || !nir_scalar_is_const(ks))
            continue;

         unsigned k = nir_scalar_as_uint(ks) & 31u;
         if (k >= 1 && k <= 15) {
            *src_out = p;
            *shift_out = k;
            *bits_out = w;
            return true;
         }
      }
   }

   return false;
}

static bool
same_scalar(nir_scalar a, nir_scalar b)
{
   return a.def == b.def && a.comp == b.comp;
}

static bool
match_hi_extract(nir_instr *instr, nir_scalar p, unsigned k,
                 unsigned *bits_out)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;

   nir_scalar root = nir_get_scalar(&alu->def, 0);
   const unsigned rem = 16u - k;

   if (alu->op == nir_op_ushr) {
      nir_scalar s0 = nir_scalar_chase_alu_src(root, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(root, 1);
      if (same_scalar(s0, p) && nir_scalar_is_const(s1) &&
          (nir_scalar_as_uint(s1) & 31u) == k + 16u) {
         *bits_out = rem;
         return true;
      }
      return false;
   }

   if (alu->op == nir_op_ubitfield_extract) {
      nir_scalar s0 = nir_scalar_chase_alu_src(root, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(root, 1);
      nir_scalar s2 = nir_scalar_chase_alu_src(root, 2);
      if (!same_scalar(s0, p) || !nir_scalar_is_const(s1) ||
          !nir_scalar_is_const(s2))
         return false;
      if (nir_scalar_as_uint(s1) != k + 16u)
         return false;
      unsigned w = nir_scalar_as_uint(s2);
      if (w >= rem) {
         *bits_out = rem;
         return true;
      }
      if (w >= 1 && w <= 8) {
         *bits_out = w;
         return true;
      }
      return false;
   }

   if (k == 8 && alu->op == nir_op_extract_u8) {
      nir_scalar s0 = nir_scalar_chase_alu_src(root, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(root, 1);
      if (same_scalar(s0, p) && nir_scalar_is_const(s1) &&
          nir_scalar_as_uint(s1) == 3) {
         *bits_out = 8;
         return true;
      }
      return false;
   }

   return false;
}

static bool
match_hi16_masked_extract(nir_instr *instr, nir_scalar *src_out,
                          unsigned *bits_out)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;

   nir_scalar root = nir_get_scalar(&alu->def, 0);

   if (alu->op == nir_op_ubitfield_extract) {
      nir_scalar s0 = nir_scalar_chase_alu_src(root, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(root, 1);
      nir_scalar s2 = nir_scalar_chase_alu_src(root, 2);
      if (is_word(s0) && nir_scalar_is_const(s1) && nir_scalar_is_const(s2) &&
          nir_scalar_as_uint(s1) == 16) {
         unsigned w = nir_scalar_as_uint(s2);
         if (w >= 1 && w <= 8) {
            *src_out = s0;
            *bits_out = w;
            return true;
         }
      }
      return false;
   }

   if (alu->op == nir_op_extract_u8) {
      nir_scalar s0 = nir_scalar_chase_alu_src(root, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(root, 1);
      if (is_word(s0) && nir_scalar_is_const(s1) &&
          nir_scalar_as_uint(s1) == 2) {
         *src_out = s0;
         *bits_out = 8;
         return true;
      }
      return false;
   }

   if (alu->op == nir_op_iand) {
      for (unsigned c = 0; c < 2; ++c) {
         nir_scalar mc = nir_scalar_chase_alu_src(root, c);
         unsigned w = 0;
         if (!nir_scalar_is_const(mc) ||
             !is_small_bitmask(nir_scalar_as_uint(mc), &w))
            continue;

         nir_scalar hi = nir_scalar_chase_alu_src(root, 1 - c);
         if (!nir_scalar_is_alu(hi))
            continue;

         nir_op op = nir_scalar_alu_op(hi);
         if (op == nir_op_extract_u16) {
            nir_scalar p = nir_scalar_chase_alu_src(hi, 0);
            nir_scalar idx = nir_scalar_chase_alu_src(hi, 1);
            if (is_word(p) && nir_scalar_is_const(idx) &&
                nir_scalar_as_uint(idx) == 1) {
               *src_out = p;
               *bits_out = w;
               return true;
            }
         } else if (op == nir_op_u2u32) {
            nir_scalar unp = nir_scalar_chase_alu_src(hi, 0);
            if (nir_scalar_is_alu(unp) &&
                nir_scalar_alu_op(unp) == nir_op_unpack_32_2x16_split_y) {
               nir_scalar p = nir_scalar_chase_alu_src(unp, 0);
               if (is_word(p)) {
                  *src_out = p;
                  *bits_out = w;
                  return true;
               }
            }
         } else if (op == nir_op_ushr) {
            nir_scalar p = nir_scalar_chase_alu_src(hi, 0);
            nir_scalar ks = nir_scalar_chase_alu_src(hi, 1);
            if (is_word(p) && nir_scalar_is_const(ks) &&
                (nir_scalar_as_uint(ks) & 31u) == 16) {
               *src_out = p;
               *bits_out = w;
               return true;
            }
         }
      }
   }

   return false;
}

bool
agx_nir_opt_half_bitfield_extract(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_index_instrs(impl);
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         /* Pass 1: share q = ushr(p, k) across paired low- and high-half
          * extracts at bit offsets k and k + 16 in the same block. */
         nir_foreach_instr_safe(instr, block) {
            nir_scalar p;
            unsigned k = 0, w1 = 0;
            if (!match_lo_shifted_extract(instr, &p, &k, &w1))
               continue;

            nir_alu_instr *lo_alu = nir_instr_as_alu(instr);
            if (!all_uses_fold_u16_zext(&lo_alu->def))
               continue;

            nir_instr *earliest = instr;
            bool found_hi = false;
            nir_foreach_use(use, p.def) {
               if (nir_src_is_if(use))
                  continue;
               nir_instr *cand = nir_src_use_instr(use);
               if (cand == instr || cand->block != block)
                  continue;
               unsigned w2 = 0;
               if (!match_hi_extract(cand, p, k, &w2))
                  continue;
               if (!all_uses_fold_u16_zext(&nir_instr_as_alu(cand)->def))
                  continue;
               found_hi = true;
               if (cand->index < earliest->index)
                  earliest = cand;
            }

            if (!found_hi)
               continue;

            b.cursor = nir_before_instr(earliest);
            nir_def *q = nir_ushr_imm(&b, nir_channel(&b, p.def, p.comp), k);
            nir_def *q_lo = nir_unpack_32_2x16_split_x(&b, q);
            nir_def *q_hi = nir_unpack_32_2x16_split_y(&b, q);

            b.cursor = nir_before_instr(instr);
            nir_def *lo_rep =
               nir_u2u32(&b, nir_iand_imm(&b, q_lo, (1u << w1) - 1));
            nir_def_rewrite_uses(&lo_alu->def, lo_rep);

            const unsigned rem = 16u - k;
            nir_foreach_use_safe(use, p.def) {
               if (nir_src_is_if(use))
                  continue;
               nir_instr *cand = nir_src_use_instr(use);
               if (cand->block != block)
                  continue;
               unsigned w2 = 0;
               if (!match_hi_extract(cand, p, k, &w2))
                  continue;
               nir_alu_instr *hi_alu = nir_instr_as_alu(cand);
               if (!all_uses_fold_u16_zext(&hi_alu->def))
                  continue;

               b.cursor = nir_before_instr(cand);
               nir_def *h = q_hi;
               if (w2 < rem)
                  h = nir_iand_imm(&b, h, (1u << w2) - 1);
               nir_def_rewrite_uses(&hi_alu->def, nir_u2u32(&b, h));
            }

            impl_progress = true;
         }

         /* Pass 2: lower standalone high-half small-mask extracts
          * ((p >> 16) & mask, mask <= 0xff) to 16-bit SCIB AND + u2u32 when
          * all consumers fold zero-extended halves natively. */
         nir_foreach_instr_safe(instr, block) {
            nir_scalar p;
            unsigned w = 0;
            if (!match_hi16_masked_extract(instr, &p, &w))
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (!all_uses_fold_u16_zext(&alu->def))
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def *hi =
               nir_unpack_32_2x16_split_y(&b, nir_channel(&b, p.def, p.comp));
            nir_def *rep = nir_u2u32(&b, nir_iand_imm(&b, hi, (1u << w) - 1));
            nir_def_replace(&alu->def, rep);
            impl_progress = true;
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}
