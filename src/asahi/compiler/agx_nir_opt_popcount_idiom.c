/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Population counts written out with masks and shifts, as shaders without a
 * bitCount() builtin do:
 *
 *    x = (x & 0x55555555) + ((x >> 1) & 0x55555555);   ... and so on to
 *    x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff);
 *
 * or the multiply form
 *
 *    x -= (x >> 1) & 0x55555555;
 *    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
 *    c = (((x + (x >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
 *
 * Both equal bitCount(x) for every 32-bit x. LLVM-based shader compilers
 * recognize them; M5 counts bits in one instruction.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op &&
          s.def->bit_size == 32;
}

static bool
is_imm(nir_scalar s, uint32_t v)
{
   return nir_scalar_is_const(s) && (uint32_t)nir_scalar_as_uint(s) == v;
}

static bool
same(nir_scalar a, nir_scalar b)
{
   return a.def == b.def && a.comp == b.comp;
}

/* s = v & m */
static bool
match_and(nir_scalar s, uint32_t m, nir_scalar *v)
{
   s = nir_scalar_chase_movs(s);
   if (!is_op(s, nir_op_iand))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      if (is_imm(nir_scalar_chase_alu_src(s, i), m)) {
         *v = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1 - i));
         return true;
      }
   }
   return false;
}

/* s = v >> sh */
static bool
match_shr(nir_scalar s, unsigned sh, nir_scalar *v)
{
   s = nir_scalar_chase_movs(s);
   if (!is_op(s, nir_op_ushr) || !is_imm(nir_scalar_chase_alu_src(s, 1), sh))
      return false;
   *v = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   return true;
}

/* s = (v & m) + ((v >> sh) & m) */
static bool
match_level(nir_scalar s, uint32_t m, unsigned sh, nir_scalar *v)
{
   s = nir_scalar_chase_movs(s);
   if (!is_op(s, nir_op_iadd))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar a, b, c;
      if (match_and(nir_scalar_chase_alu_src(s, i), m, &a) &&
          match_and(nir_scalar_chase_alu_src(s, 1 - i), m, &b) &&
          match_shr(b, sh, &c) && same(a, c)) {
         *v = a;
         return true;
      }
   }
   return false;
}

/* The tree up to and including the 0x00ff00ff level, from its input. */
static bool
match_tree4(nir_scalar s, nir_scalar *x)
{
   nir_scalar l3, l2, l1;
   return match_level(s, 0x00ff00ff, 8, &l3) &&
          match_level(l3, 0x0f0f0f0f, 4, &l2) &&
          match_level(l2, 0x33333333, 2, &l1) &&
          match_level(l1, 0x55555555, 1, x);
}

/* The last level of the tree: (y & 0xffff) + ((y >> 16) [& 0xffff]). */
static bool
match_tree(nir_scalar s, nir_scalar *x)
{
   nir_scalar y;
   if (match_level(s, 0xffff, 16, &y))
      return match_tree4(y, x);

   s = nir_scalar_chase_movs(s);
   if (!is_op(s, nir_op_iadd))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar a, b;
      if (match_and(nir_scalar_chase_alu_src(s, i), 0xffff, &a) &&
          match_shr(nir_scalar_chase_alu_src(s, 1 - i), 16, &b) && same(a, b))
         return match_tree4(a, x);
   }
   return false;
}

/* (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24 with
 * v = (u & 0x33..) + ((u >> 2) & 0x33..), u = x - ((x >> 1) & 0x55..). */
static bool
match_multiply(nir_scalar s, nir_scalar *x)
{
   nir_scalar p, q, v, w, u;
   if (!match_shr(s, 24, &p) || !is_op(p, nir_op_imul))
      return false;
   bool ok = false;
   for (unsigned i = 0; i < 2 && !ok; ++i) {
      if (is_imm(nir_scalar_chase_alu_src(p, i), 0x01010101)) {
         q = nir_scalar_chase_movs(nir_scalar_chase_alu_src(p, 1 - i));
         ok = true;
      }
   }
   if (!ok || !match_and(q, 0x0f0f0f0f, &q) || !is_op(q, nir_op_iadd))
      return false;
   ok = false;
   for (unsigned i = 0; i < 2 && !ok; ++i) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(q, i));
      if (match_shr(nir_scalar_chase_alu_src(q, 1 - i), 4, &w) && same(a, w)) {
         v = a;
         ok = true;
      }
   }
   if (!ok || !match_level(v, 0x33333333, 2, &u) || !is_op(u, nir_op_isub))
      return false;
   nir_scalar xx = nir_scalar_chase_movs(nir_scalar_chase_alu_src(u, 0));
   nir_scalar t;
   if (!match_and(nir_scalar_chase_alu_src(u, 1), 0x55555555, &t) ||
       !match_shr(t, 1, &t) || !same(t, xx))
      return false;
   *x = xx;
   return true;
}

static bool
pass(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_iadd && alu->op != nir_op_ushr) ||
       alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;

   nir_scalar s = nir_get_scalar(&alu->def, 0), x;
   if (!(alu->op == nir_op_iadd ? match_tree(s, &x) : match_multiply(s, &x)))
      return false;

   b->cursor = nir_after_instr(&alu->instr);
   nir_def *count = nir_bit_count(b, nir_channel(b, x.def, x.comp));
   nir_def_rewrite_uses(&alu->def, count);
   return true;
}

bool
agx_nir_opt_popcount_idiom(nir_shader *shader)
{
   return nir_shader_alu_pass(shader, pass, nir_metadata_control_flow, NULL);
}
