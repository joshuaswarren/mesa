/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * Narrow single-use packed 32-bit loads whose sole consumer extracts the
 * sub-word at the exact byte or halfword index that addressed the load.
 *
 * Shaders reading packed uint8/uint16 arrays through a uint32 SSBO (for
 * example `uint v = (buf[off >> 2] >> ((off & 3) * 8)) & 0xff`) lower to:
 *
 *    w = load_agx(base, ushr(off, 2), format=R32_UINT)
 *    v = ubitfield_extract(w, (off & 3) << 3, 8)
 *
 * On little-endian AGX, for every uint32_t `off`:
 *
 *    4ULL * (off >> 2) + (off & 3) == (uint64_t)off
 *    4ULL * (off >> 1) + 2ULL * (off & 1) == 2ULL * (uint64_t)off
 *
 * Because the narrowed load accesses a strict subset of the four bytes read by
 * the original 32-bit load at the exact same program point, it preserves
 * bounds safety and memory ordering while eliminating both the index shift and
 * the shift/mask/extract sequence.
 */

static nir_alu_instr *
single_scalar_alu_use(nir_def *def)
{
   if (!list_is_singular(&def->uses))
      return NULL;

   nir_src *use = list_first_entry(&def->uses, nir_src, use_link);
   if (nir_src_is_if(use))
      return NULL;

   nir_instr *user = nir_src_use_instr(use);
   if (user->type != nir_instr_type_alu)
      return NULL;

   nir_alu_instr *alu = nir_instr_as_alu(user);
   if (alu->def.num_components != 1)
      return NULL;

   return alu;
}

static bool
same_scalar(nir_scalar a, nir_scalar b)
{
   a = nir_scalar_chase_movs(a);
   b = nir_scalar_chase_movs(b);
   return a.def == b.def && a.comp == b.comp;
}

static bool
match_iand_const(nir_scalar s, nir_scalar *other_out, uint64_t *mask_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_iand)
      return false;

   nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
   nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
   if (nir_scalar_is_const(s1)) {
      *other_out = s0;
      *mask_out = nir_scalar_as_uint(s1);
      return true;
   }
   if (nir_scalar_is_const(s0)) {
      *other_out = s1;
      *mask_out = nir_scalar_as_uint(s0);
      return true;
   }
   return false;
}

static bool
match_scale_const(nir_scalar s, unsigned shift, nir_scalar *val_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s))
      return false;

   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl) {
      nir_scalar cnt = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(cnt) && (nir_scalar_as_uint(cnt) & 31u) == shift) {
         *val_out = nir_scalar_chase_alu_src(s, 0);
         return true;
      }
      return false;
   }

   if (op == nir_op_imul || op == nir_op_amul) {
      const uint64_t factor = UINT64_C(1) << shift;
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(s1) && nir_scalar_as_uint(s1) == factor) {
         *val_out = s0;
         return true;
      }
      if (nir_scalar_is_const(s0) && nir_scalar_as_uint(s0) == factor) {
         *val_out = s1;
         return true;
      }
   }

   return false;
}

/* Match index expressions that select the 32-bit word containing element `off`:
 *   - base == 0: ushr(off, 2) for byte, ushr(off, 1) for halfword
 *   - base == AGX_NIR_MEMORY_INDEX_BYTES: iand(off, ~3) or ishl(ushr(off, 2), 2)
 */
static bool
match_packed_index(nir_scalar idx, unsigned base, nir_scalar *off_out,
                   unsigned *elem_bytes_out)
{
   idx = nir_scalar_chase_movs(idx);
   if (idx.def->bit_size != 32 || !nir_scalar_is_alu(idx))
      return false;

   if (base == 0) {
      if (nir_scalar_alu_op(idx) != nir_op_ushr)
         return false;
      nir_scalar cnt = nir_scalar_chase_alu_src(idx, 1);
      if (!nir_scalar_is_const(cnt))
         return false;
      const uint64_t shift = nir_scalar_as_uint(cnt) & 31u;
      if (shift != 1 && shift != 2)
         return false;

      nir_scalar off = nir_scalar_chase_alu_src(idx, 0);
      if (off.def->bit_size != 32)
         return false;

      *off_out = off;
      *elem_bytes_out = shift == 2 ? 1u : 2u;
      return true;
   }

   if (base == AGX_NIR_MEMORY_INDEX_BYTES) {
      nir_scalar off;
      uint64_t mask = 0;
      if (match_iand_const(idx, &off, &mask) &&
          (mask & 0xffffffffu) == 0xfffffffcu && off.def->bit_size == 32) {
         *off_out = off;
         *elem_bytes_out = 1u;
         return true;
      }

      nir_scalar inner;
      if (match_scale_const(idx, 2, &inner)) {
         inner = nir_scalar_chase_movs(inner);
         if (nir_scalar_is_alu(inner) &&
             nir_scalar_alu_op(inner) == nir_op_ushr) {
            nir_scalar cnt = nir_scalar_chase_alu_src(inner, 1);
            nir_scalar src = nir_scalar_chase_alu_src(inner, 0);
            if (nir_scalar_is_const(cnt) &&
                (nir_scalar_as_uint(cnt) & 31u) == 2u &&
                src.def->bit_size == 32) {
               *off_out = src;
               *elem_bytes_out = 1u;
               return true;
            }
         }
      }
   }

   return false;
}

/* Match a bit-shift expression equal to:
 *   - elem_bytes == 1: (off & 3) * 8   (values 0, 8, 16, 24)
 *   - elem_bytes == 2: (off & 1) * 16  (values 0, 16)
 *
 * When `modulo32_ok` is true (e.g. the shift count of a 32-bit ushr), bare
 * `off << 3` / `off * 8` (or `off << 4` / `off * 16`) also matches because
 * 32-bit shifts mask the shift count by 31.
 */
static bool
match_subword_bit_shift(nir_scalar s, nir_scalar off, unsigned elem_bytes,
                        bool modulo32_ok)
{
   const unsigned scale_shift = elem_bytes == 1 ? 3u : 4u;
   const uint64_t lane_mask = elem_bytes == 1 ? 3u : 1u;
   const uint64_t shifted_mask = lane_mask << scale_shift; /* 24 or 16 */

   s = nir_scalar_chase_movs(s);
   if (s.def->bit_size != 32)
      return false;

   /* Outer mask: ((off << 3) & 24) or ((off << 3) & 31). */
   nir_scalar inner;
   uint64_t outer_mask = 0;
   if (match_iand_const(s, &inner, &outer_mask)) {
      if (outer_mask == shifted_mask || outer_mask == 31u)
         return match_subword_bit_shift(inner, off, elem_bytes, true);
      return false;
   }

   /* Scaled value: (off & 3) << 3 or (off << 3) when mod-32 is guaranteed. */
   nir_scalar scaled_src;
   if (!match_scale_const(s, scale_shift, &scaled_src))
      return false;

   if (modulo32_ok && same_scalar(scaled_src, off))
      return true;

   nir_scalar masked_src;
   uint64_t inner_mask = 0;
   if (match_iand_const(scaled_src, &masked_src, &inner_mask) &&
       inner_mask == lane_mask && same_scalar(masked_src, off))
      return true;

   return false;
}

static bool
opt_narrow_subword_load(nir_builder *b, nir_intrinsic_instr *load,
                        UNUSED void *data)
{
   if (load->intrinsic != nir_intrinsic_load_agx &&
       load->intrinsic != nir_intrinsic_load_constant_agx)
      return false;

   if (load->def.num_components != 1 || load->def.bit_size != 32 ||
       nir_intrinsic_format(load) != PIPE_FORMAT_R32_UINT ||
       nir_intrinsic_sign_extend(load))
      return false;

   const enum gl_access_qualifier access = nir_intrinsic_access(load);
   if (access & (ACCESS_COHERENT | ACCESS_VOLATILE))
      return false;

   const unsigned base = nir_intrinsic_base(load);
   if (base != 0 && base != AGX_NIR_MEMORY_INDEX_BYTES)
      return false;

   nir_scalar off;
   unsigned elem_bytes = 0;
   nir_scalar idx = nir_get_scalar(load->src[1].ssa, 0);
   if (!match_packed_index(idx, base, &off, &elem_bytes))
      return false;

   nir_alu_instr *use = single_scalar_alu_use(&load->def);
   if (!use)
      return false;

   nir_alu_instr *consumer = NULL;
   unsigned result_bits = 0;

   if (use->op == nir_op_ubitfield_extract) {
      nir_scalar src0 = nir_scalar_chase_alu_src(nir_get_scalar(&use->def, 0), 0);
      nir_scalar bit_off = nir_scalar_chase_alu_src(nir_get_scalar(&use->def, 0), 1);
      nir_scalar bits = nir_scalar_chase_alu_src(nir_get_scalar(&use->def, 0), 2);
      if (use->def.bit_size == 32 && src0.def == &load->def &&
          nir_scalar_is_const(bits) &&
          nir_scalar_as_uint(bits) == elem_bytes * 8u &&
          match_subword_bit_shift(bit_off, off, elem_bytes, false)) {
         consumer = use;
         result_bits = 32;
      }
   } else if (use->op == nir_op_ushr) {
      nir_scalar src0 = nir_scalar_chase_alu_src(nir_get_scalar(&use->def, 0), 0);
      nir_scalar shift = nir_scalar_chase_alu_src(nir_get_scalar(&use->def, 0), 1);
      if (use->def.bit_size == 32 && src0.def == &load->def &&
          match_subword_bit_shift(shift, off, elem_bytes, true)) {
         nir_alu_instr *mask_use = single_scalar_alu_use(&use->def);
         if (mask_use) {
            if (mask_use->op == nir_op_iand && mask_use->def.bit_size == 32) {
               nir_scalar other;
               uint64_t mask = 0;
               if (match_iand_const(nir_get_scalar(&mask_use->def, 0), &other,
                                    &mask) &&
                   same_scalar(other, nir_get_scalar(&use->def, 0)) &&
                   mask == (elem_bytes == 1 ? 0xffu : 0xffffu)) {
                  consumer = mask_use;
                  result_bits = 32;
               }
            } else if (elem_bytes == 1 && mask_use->op == nir_op_u2u8) {
               consumer = mask_use;
               result_bits = 8;
            } else if (elem_bytes == 2 && mask_use->op == nir_op_u2u16) {
               consumer = mask_use;
               result_bits = 16;
            }
         }
      }
   }

   if (!consumer)
      return false;

   /* Emit the narrowed load at the exact program point of the original load. */
   b->cursor = nir_before_instr(&load->instr);
   nir_def *off_def = nir_mov_scalar(b, off);
   const enum pipe_format fmt =
      elem_bytes == 1 ? PIPE_FORMAT_R8_UINT : PIPE_FORMAT_R16_UINT;

   nir_def *narrow =
      load->intrinsic == nir_intrinsic_load_constant_agx
         ? nir_load_constant_agx(b, 1, 16, load->src[0].ssa, off_def,
                                 .access = access, .base = 0, .format = fmt,
                                 .sign_extend = false)
         : nir_load_agx(b, 1, 16, load->src[0].ssa, off_def, .access = access,
                        .base = 0, .format = fmt, .sign_extend = false);

   b->cursor = nir_before_instr(&consumer->instr);
   nir_def *replacement = narrow;
   if (result_bits == 32)
      replacement = nir_u2u32(b, narrow);
   else if (result_bits == 8)
      replacement = nir_u2u8(b, narrow);

   nir_def_rewrite_uses(&consumer->def, replacement);
   return true;
}

bool
agx_nir_opt_narrow_subword_loads(nir_shader *shader)
{
   bool progress = nir_shader_intrinsics_pass(
      shader, opt_narrow_subword_load, nir_metadata_control_flow, NULL);
   if (progress)
      nir_opt_dce(shader);
   return progress;
}
