/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_byte_stencil.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"

#include "compiler/glsl_types.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "util/u_math.h"

/* Match a 32-bit constant-offset push-constant load and return its byte offset. */
static bool
match_push_const_u32(nir_def *def, uint16_t *offset_out)
{
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;

   if (nir_def_instr(def)->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(def));
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   if (!nir_src_is_const(intr->src[0]))
      return false;

   uint64_t off = (uint64_t)nir_intrinsic_base(intr) +
                  nir_src_as_uint(intr->src[0]);
   if ((off & 3u) != 0 || off + 4u > HK_MAX_PUSH_SIZE)
      return false;

   *offset_out = (uint16_t)off;
   return true;
}

static bool
same_push_const(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   uint16_t oa = 0, ob = 0;
   return match_push_const_u32(a, &oa) && match_push_const_u32(b, &ob) &&
          oa == ob;
}

/* Strip scalar move/copy wrappers. */
static nir_def *
strip_mov(nir_def *def)
{
   while (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov && alu->src[0].src.ssa->num_components == 1) {
         def = alu->src[0].src.ssa;
         continue;
      }
      break;
   }
   return def;
}

/* Match scalar component `comp` of an intrinsic `op` (handling direct vector
 * extract via nir_op_mov swizzle or vec/channel). */
static bool
match_sysval_comp(nir_def *def, nir_intrinsic_op op, unsigned *comp_out)
{
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;

   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov) {
         nir_def *src = alu->src[0].src.ssa;
         if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op) {
            *comp_out = alu->src[0].swizzle[0];
            return true;
         }
      }
   }

   return false;
}

static bool
match_sysval_alu_src(const nir_alu_src *asrc, nir_intrinsic_op op,
                     unsigned *comp_out)
{
   nir_def *src = asrc->src.ssa;
   if (!src || src->bit_size != 32)
      return false;
   if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op) {
      *comp_out = asrc->swizzle[0];
      return true;
   }
   return match_sysval_comp(src, op, comp_out);
}

static bool
match_scaled_sysval_comp(const nir_alu_src *asrc, nir_intrinsic_op op,
                         const uint32_t wg_size[3], unsigned comp)
{
   if (comp >= 3 || wg_size[comp] == 0)
      return false;

   unsigned sc = 0;
   if (wg_size[comp] == 1 && match_sysval_alu_src(asrc, op, &sc) && sc == comp)
      return true;

   nir_def *bm = strip_mov(asrc->src.ssa);
   if (!bm || nir_def_instr(bm)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(bm));
   if (mul->op == nir_op_imul) {
      for (unsigned ms = 0; ms < 2; ms++) {
         if (match_sysval_alu_src(&mul->src[ms], op, &sc) && sc == comp &&
             nir_src_is_const(mul->src[1 - ms].src) &&
             nir_src_comp_as_uint(mul->src[1 - ms].src,
                                  mul->src[1 - ms].swizzle[0]) ==
                wg_size[comp])
            return true;
      }
   } else if (mul->op == nir_op_ishl) {
      if (match_sysval_alu_src(&mul->src[0], op, &sc) && sc == comp &&
          nir_src_is_const(mul->src[1].src) &&
          (1u << nir_src_comp_as_uint(mul->src[1].src,
                                      mul->src[1].swizzle[0])) ==
             wg_size[comp])
         return true;
   } else if (mul->op == nir_op_iadd && wg_size[comp] == 2) {
      unsigned sc0 = 0, sc1 = 0;
      if (match_sysval_alu_src(&mul->src[0], op, &sc0) && sc0 == comp &&
          match_sysval_alu_src(&mul->src[1], op, &sc1) && sc1 == comp)
         return true;
   }
   return false;
}

/* Match global_invocation_id[comp], either directly, plus
 * base_global_invocation_id[comp] (or base_workgroup_id[comp] * wg_size[comp]),
 * or lowered to workgroup_id[comp] * wg_size[comp] + local_invocation_id[comp]. */
static bool
match_global_id_comp(nir_def *def, const uint32_t wg_size[3],
                     unsigned *comp_out)
{
   if (match_sysval_comp(def, nir_intrinsic_load_global_invocation_id,
                         comp_out))
      return true;

   /* When workgroup_size[c] == 1, nir_lower_system_values folds
    * local_invocation_id[c] == 0 away and leaves bare workgroup_id[c]
    * (or workgroup_id[c] + base_workgroup_id[c]). */
   unsigned wg_c = 0;
   if (match_sysval_comp(def, nir_intrinsic_load_workgroup_id, &wg_c) &&
       wg_c < 3 && wg_size[wg_c] == 1) {
      *comp_out = wg_c;
      return true;
   }

   /* Handle mov(iadd(load_global_invocation_id, load_base_global_invocation_id).c)
    * and mov(iadd(load_workgroup_id, load_base_workgroup_id).c) when wg_size[c] == 1. */
   if (def && def->num_components == 1 &&
       nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *m = nir_instr_as_alu(nir_def_instr(def));
      if (m->op == nir_op_mov && m->src[0].src.ssa->num_components > 1 &&
          nir_def_instr(m->src[0].src.ssa)->type == nir_instr_type_alu) {
         nir_alu_instr *vadd =
            nir_instr_as_alu(nir_def_instr(m->src[0].src.ssa));
         if (vadd->op == nir_op_iadd) {
            unsigned c = m->src[0].swizzle[0];
            nir_def *s0 = vadd->src[0].src.ssa;
            nir_def *s1 = vadd->src[1].src.ssa;
            if (nir_def_instr(s0)->type == nir_instr_type_intrinsic &&
                nir_def_instr(s1)->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op0 =
                  nir_instr_as_intrinsic(nir_def_instr(s0))->intrinsic;
               nir_intrinsic_op op1 =
                  nir_instr_as_intrinsic(nir_def_instr(s1))->intrinsic;
               if ((op0 == nir_intrinsic_load_global_invocation_id &&
                    op1 == nir_intrinsic_load_base_global_invocation_id) ||
                   (op1 == nir_intrinsic_load_global_invocation_id &&
                    op0 == nir_intrinsic_load_base_global_invocation_id)) {
                  *comp_out = c;
                  return true;
               }
               if (c < 3 && wg_size[c] == 1 &&
                   ((op0 == nir_intrinsic_load_workgroup_id &&
                     op1 == nir_intrinsic_load_base_workgroup_id) ||
                    (op1 == nir_intrinsic_load_workgroup_id &&
                     op0 == nir_intrinsic_load_base_workgroup_id))) {
                  *comp_out = c;
                  return true;
               }
            }
         }
      }
   }

   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(def));
   if (add->op != nir_op_iadd)
      return false;

   /* Scalarized global_invocation_id[c] + base_global_invocation_id[c],
    * global_invocation_id[c] + base_workgroup_id[c] * wg_size[c], or
    * workgroup_id[c] + base_workgroup_id[c] when wg_size[c] == 1. */
   for (unsigned side = 0; side < 2; side++) {
      unsigned gc = 0, bc = 0;
      if (match_sysval_alu_src(&add->src[side],
                               nir_intrinsic_load_global_invocation_id, &gc) &&
          gc < 3) {
         if ((match_sysval_alu_src(&add->src[1 - side],
                                   nir_intrinsic_load_base_global_invocation_id,
                                   &bc) &&
              bc == gc) ||
             match_scaled_sysval_comp(&add->src[1 - side],
                                      nir_intrinsic_load_base_workgroup_id,
                                      wg_size, gc)) {
            *comp_out = gc;
            return true;
         }
      }
      if (match_sysval_alu_src(&add->src[side],
                               nir_intrinsic_load_workgroup_id, &gc) &&
          gc < 3 && wg_size[gc] == 1) {
         if ((match_sysval_alu_src(&add->src[1 - side],
                                   nir_intrinsic_load_base_workgroup_id,
                                   &bc) &&
              bc == gc) ||
             (match_sysval_alu_src(&add->src[1 - side],
                                   nir_intrinsic_load_base_global_invocation_id,
                                   &bc) &&
              bc == gc)) {
            *comp_out = gc;
            return true;
         }
      }
   }

   for (unsigned side = 0; side < 2; side++) {
      unsigned loc_c = 0;
      if (!match_sysval_alu_src(&add->src[side],
                                nir_intrinsic_load_local_invocation_id, &loc_c))
         continue;
      if (loc_c >= 3)
         continue;

      if (match_scaled_sysval_comp(&add->src[1 - side],
                                   nir_intrinsic_load_workgroup_id, wg_size,
                                   loc_c)) {
         *comp_out = loc_c;
         return true;
      }
   }

   return false;
}

/* Match global_size[comp] = num_workgroups[comp] * wg_size[comp]. */
static bool
match_global_size_comp(nir_def *def, const uint32_t wg_size[3], unsigned comp)
{
   if (comp >= 3 || wg_size[comp] == 0)
      return false;

   unsigned nw_c = 0;
   if (wg_size[comp] == 1 &&
       match_sysval_comp(def, nir_intrinsic_load_num_workgroups, &nw_c) &&
       nw_c == comp)
      return true;

   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_imul) {
      for (unsigned s = 0; s < 2; s++) {
         if (!match_sysval_alu_src(&alu->src[s],
                                   nir_intrinsic_load_num_workgroups, &nw_c) ||
             nw_c != comp)
            continue;
         if (nir_src_is_const(alu->src[1 - s].src) &&
             nir_src_comp_as_uint(alu->src[1 - s].src,
                                  alu->src[1 - s].swizzle[0]) == wg_size[comp])
            return true;
         unsigned ws_c = 0;
         if (match_sysval_alu_src(&alu->src[1 - s],
                                  nir_intrinsic_load_workgroup_size, &ws_c) &&
             ws_c == comp)
            return true;
      }
   } else if (alu->op == nir_op_ishl) {
      if (match_sysval_alu_src(&alu->src[0], nir_intrinsic_load_num_workgroups,
                               &nw_c) &&
          nw_c == comp && nir_src_is_const(alu->src[1].src) &&
          (1u << nir_src_comp_as_uint(alu->src[1].src,
                                      alu->src[1].swizzle[0])) == wg_size[comp])
         return true;
   }

   return false;
}

/* Match isign(d) or its lowered form imin(imax(d, -1), 1) / imax(imin(d, 1), -1). */
static bool
match_isign_of_pc(nir_def *def, nir_def *pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_isign)
      return same_push_const(alu->src[0].src.ssa, pc);

   if (alu->op == nir_op_imin || alu->op == nir_op_imax) {
      int outer_c = (alu->op == nir_op_imin) ? 1 : -1;
      nir_op inner_op = (alu->op == nir_op_imin) ? nir_op_imax : nir_op_imin;
      int inner_c = -outer_c;
      for (unsigned s = 0; s < 2; s++) {
         if (!nir_src_is_const(alu->src[s].src) ||
             nir_src_as_int(alu->src[s].src) != outer_c)
            continue;
         nir_def *in = strip_mov(alu->src[1 - s].src.ssa);
         if (!in || nir_def_instr(in)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *ia = nir_instr_as_alu(nir_def_instr(in));
         if (ia->op != inner_op)
            continue;
         for (unsigned is = 0; is < 2; is++) {
            if (nir_src_is_const(ia->src[is].src) &&
                nir_src_as_int(ia->src[is].src) == inner_c &&
                same_push_const(ia->src[1 - is].src.ssa, pc))
               return true;
         }
      }
   }
   return false;
}

/* Match iabs(d) or its lowered form imax(d, ineg(d)). */
static bool
match_iabs_of_pc(nir_def *def, nir_def *pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iabs)
      return same_push_const(alu->src[0].src.ssa, pc);
   if (alu->op == nir_op_imax) {
      for (unsigned s = 0; s < 2; s++) {
         nir_def *a = strip_mov(alu->src[s].src.ssa);
         nir_def *b = strip_mov(alu->src[1 - s].src.ssa);
         if (same_push_const(a, pc) && b &&
             nir_def_instr(b)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(b))->op == nir_op_ineg &&
             same_push_const(
                nir_instr_as_alu(nir_def_instr(b))->src[0].src.ssa, pc))
            return true;
      }
   }
   return false;
}

static nir_def *
extract_pc_from_iabs(nir_def *def)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return NULL;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   uint16_t off = 0;
   if (alu->op == nir_op_iabs) {
      nir_def *d = strip_mov(alu->src[0].src.ssa);
      return match_push_const_u32(d, &off) ? d : NULL;
   }
   if (alu->op == nir_op_imax) {
      for (unsigned s = 0; s < 2; s++) {
         nir_def *a = strip_mov(alu->src[s].src.ssa);
         nir_def *b = strip_mov(alu->src[1 - s].src.ssa);
         if (match_push_const_u32(a, &off) && b &&
             nir_def_instr(b)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(b))->op == nir_op_ineg &&
             same_push_const(
                nir_instr_as_alu(nir_def_instr(b))->src[0].src.ssa, a))
            return a;
      }
   }
   return NULL;
}

/* Match radius = diameter / 2 (including algebraic lowering of signed idiv by 2
 * and logical/arithmetic shift right by 1; all agree for diameter >= 1). */
static bool
match_radius_from_diameter(nir_def *def, nir_def *diameter_pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_idiv) {
      return same_push_const(alu->src[0].src.ssa, diameter_pc) &&
             nir_src_is_const(alu->src[1].src) &&
             nir_src_as_int(alu->src[1].src) == 2;
   }

   if (alu->op == nir_op_imul) {
      /* nir_opt_algebraic lowers idiv(d, 2) to
       * imul(isign(d), ushr(iabs(d), 1)) (or ishr, with lowered isign/iabs). */
      for (unsigned s = 0; s < 2; s++) {
         nir_def *a = strip_mov(alu->src[s].src.ssa);
         nir_def *b = strip_mov(alu->src[1 - s].src.ssa);
         if (!a || !b || !match_isign_of_pc(a, diameter_pc) ||
             nir_def_instr(b)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *shr_alu = nir_instr_as_alu(nir_def_instr(b));
         if ((shr_alu->op != nir_op_ushr && shr_alu->op != nir_op_ishr) ||
             !nir_src_is_const(shr_alu->src[1].src) ||
             nir_src_as_uint(shr_alu->src[1].src) != 1)
            continue;
         if (match_iabs_of_pc(shr_alu->src[0].src.ssa, diameter_pc))
            return true;
      }
      return false;
   }

   if (alu->op == nir_op_ishr || alu->op == nir_op_ushr) {
      if (!nir_src_is_const(alu->src[1].src) ||
          nir_src_as_uint(alu->src[1].src) != 1)
         return false;

      nir_def *lhs = strip_mov(alu->src[0].src.ssa);
      if (same_push_const(lhs, diameter_pc))
         return true;

      /* nir_opt_algebraic lowers idiv(d, 2) to ishr(iadd(d, ushr(d, 31)), 1)
       * or ishr(isub(d, ishr(d, 31)), 1). */
      if (lhs && nir_def_instr(lhs)->type == nir_instr_type_alu) {
         nir_alu_instr *inner = nir_instr_as_alu(nir_def_instr(lhs));
         if (inner->op == nir_op_iadd) {
            for (unsigned s = 0; s < 2; s++) {
               nir_def *d = inner->src[s].src.ssa;
               nir_def *sgn = strip_mov(inner->src[1 - s].src.ssa);
               if (!same_push_const(d, diameter_pc))
                  continue;
               if (sgn && nir_def_instr(sgn)->type == nir_instr_type_alu) {
                  nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(sgn));
                  if (sa->op == nir_op_ushr &&
                      same_push_const(sa->src[0].src.ssa, diameter_pc) &&
                      nir_src_is_const(sa->src[1].src) &&
                      nir_src_as_uint(sa->src[1].src) == 31)
                     return true;
               }
            }
         } else if (inner->op == nir_op_isub) {
            nir_def *d = inner->src[0].src.ssa;
            nir_def *sgn = strip_mov(inner->src[1].src.ssa);
            if (same_push_const(d, diameter_pc) && sgn &&
                nir_def_instr(sgn)->type == nir_instr_type_alu) {
               nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(sgn));
               if (sa->op == nir_op_ishr &&
                   same_push_const(sa->src[0].src.ssa, diameter_pc) &&
                   nir_src_is_const(sa->src[1].src) &&
                   nir_src_as_uint(sa->src[1].src) == 31)
                  return true;
            }
         }
      }
   }

   return false;
}

/* Match `a < b` (signed 32-bit), returning true with `*negated` if inverted
 * (`a >= b`, i.e., `!(a < b)`). */
static bool
match_slt_or_sge(nir_def *cond, nir_def *a, nir_def *b, bool *is_ge_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub_ge = false;
      if (!match_slt_or_sge(alu->src[0].src.ssa, a, b, &sub_ge))
         return false;
      *is_ge_out = !sub_ge;
      return true;
   }

   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   bool match_ab = (s0 == a || same_push_const(s0, a)) &&
                   (s1 == b || same_push_const(s1, b));
   bool match_ba = (s0 == b || same_push_const(s0, b)) &&
                   (s1 == a || same_push_const(s1, a));

   if (alu->op == nir_op_ilt && match_ab) {
      *is_ge_out = false;
      return true;
   }
   if (alu->op == nir_op_ige && match_ab) {
      *is_ge_out = true;
      return true;
   }
   if (alu->op == nir_op_ige && match_ba) {
      /* b >= a is a <= b, not strictly a < b. */
      return false;
   }
   if (alu->op == nir_op_ilt && match_ba) {
      /* b < a is a > b, not a >= b. */
      return false;
   }

   return false;
}

/* Match signed `a < b` where `a` is an SSA def and `b` is a push constant,
 * extracting the push-constant def in `*bound_out`. */
static bool
match_iv_lt_pc(nir_def *cond, nir_def *iv, nir_def **bound_out, bool *is_ge_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub_ge = false;
      if (!match_iv_lt_pc(alu->src[0].src.ssa, iv, bound_out, &sub_ge))
         return false;
      *is_ge_out = !sub_ge;
      return true;
   }

   if (alu->op != nir_op_ilt && alu->op != nir_op_ige)
      return false;

   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   uint16_t off = 0;
   if (s0 == iv && match_push_const_u32(s1, &off)) {
      *bound_out = s1;
      *is_ge_out = (alu->op == nir_op_ige);
      return true;
   }

   return false;
}

/* Match `c < radius` (signed) where `radius = diameter / 2`. */
static bool
match_c_lt_radius(nir_def *cond, nir_def *c, nir_def *diameter_pc,
                  bool *negated_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub = false;
      if (!match_c_lt_radius(alu->src[0].src.ssa, c, diameter_pc, &sub))
         return false;
      *negated_out = !sub;
      return true;
   }

   if (alu->op != nir_op_ilt && alu->op != nir_op_ige)
      return false;

   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   if (s0 == c && match_radius_from_diameter(s1, diameter_pc)) {
      *negated_out = (alu->op == nir_op_ige);
      return true;
   }

   return false;
}

/* Match `c >= dim - radius` (or `c + radius >= dim`, or `!(c < dim - radius)`). */
static bool
match_c_ge_dim_minus_radius(nir_def *cond, nir_def *c, nir_def *dim_pc,
                            nir_def *diameter_pc, bool *negated_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub = false;
      if (!match_c_ge_dim_minus_radius(alu->src[0].src.ssa, c, dim_pc,
                                       diameter_pc, &sub))
         return false;
      *negated_out = !sub;
      return true;
   }

   if (alu->op != nir_op_ige && alu->op != nir_op_ilt)
      return false;

   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);

   /* Form 1: c >= (dim - radius)  or  c < (dim - radius) */
   if (s0 == c && s1 && nir_def_instr(s1)->type == nir_instr_type_alu) {
      nir_alu_instr *sub = nir_instr_as_alu(nir_def_instr(s1));
      if (sub->op == nir_op_isub &&
          same_push_const(sub->src[0].src.ssa, dim_pc) &&
          match_radius_from_diameter(sub->src[1].src.ssa, diameter_pc)) {
         *negated_out = (alu->op == nir_op_ilt);
         return true;
      }
      /* iadd(dim, ineg(radius)) */
      if (sub->op == nir_op_iadd) {
         for (unsigned side = 0; side < 2; side++) {
            nir_def *neg = strip_mov(sub->src[1 - side].src.ssa);
            if (same_push_const(sub->src[side].src.ssa, dim_pc) && neg &&
                nir_def_instr(neg)->type == nir_instr_type_alu &&
                nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg &&
                match_radius_from_diameter(
                   nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa,
                   diameter_pc)) {
               *negated_out = (alu->op == nir_op_ilt);
               return true;
            }
         }
      }
   }

   /* Form 2: (c + radius) >= dim  or  (c + radius) < dim */
   if (same_push_const(s1, dim_pc) && s0 &&
       nir_def_instr(s0)->type == nir_instr_type_alu) {
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(s0));
      if (add->op == nir_op_iadd) {
         for (unsigned side = 0; side < 2; side++) {
            if (strip_mov(add->src[side].src.ssa) == c &&
                match_radius_from_diameter(add->src[1 - side].src.ssa,
                                           diameter_pc)) {
               *negated_out = (alu->op == nir_op_ilt);
               return true;
            }
         }
      }
   }

   return false;
}

/* Match `in_border = (c < radius) || (c >= dim - radius)` (or its negation). */
static bool
match_border_cond(nir_def *cond, nir_def *c, nir_def *dim_pc,
                  nir_def *diameter_pc, bool *inverted_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub = false;
      if (!match_border_cond(alu->src[0].src.ssa, c, dim_pc, diameter_pc, &sub))
         return false;
      *inverted_out = !sub;
      return true;
   }

   nir_def *p0 = NULL, *p1 = NULL;
   bool is_and = false;

   if (alu->op == nir_op_ior) {
      p0 = alu->src[0].src.ssa;
      p1 = alu->src[1].src.ssa;
      is_and = false;
   } else if (alu->op == nir_op_iand) {
      p0 = alu->src[0].src.ssa;
      p1 = alu->src[1].src.ssa;
      is_and = true;
   } else if (alu->op == nir_op_bcsel) {
      /* bcsel(a, true, b) == a || b; bcsel(a, b, true) == !a || b */
      if (nir_src_is_const(alu->src[1].src) &&
          nir_src_as_bool(alu->src[1].src)) {
         p0 = alu->src[0].src.ssa;
         p1 = alu->src[2].src.ssa;
         is_and = false;
      } else if (nir_src_is_const(alu->src[2].src) &&
                 nir_src_as_bool(alu->src[2].src)) {
         /* !s0 || s1: handle by negating expectations on s0 below */
         bool neg_lt = false, neg_ge = false;
         if (match_c_lt_radius(alu->src[0].src.ssa, c, diameter_pc, &neg_lt) &&
             neg_lt &&
             match_c_ge_dim_minus_radius(alu->src[1].src.ssa, c, dim_pc,
                                         diameter_pc, &neg_ge) &&
             !neg_ge) {
            *inverted_out = false;
            return true;
         }
         if (match_c_ge_dim_minus_radius(alu->src[0].src.ssa, c, dim_pc,
                                         diameter_pc, &neg_ge) &&
             neg_ge &&
             match_c_lt_radius(alu->src[1].src.ssa, c, diameter_pc, &neg_lt) &&
             !neg_lt) {
            *inverted_out = false;
            return true;
         }
         return false;
      } else {
         return false;
      }
   } else {
      return false;
   }

   for (unsigned swap = 0; swap < 2; swap++) {
      nir_def *a = swap ? p1 : p0;
      nir_def *b = swap ? p0 : p1;
      bool neg_lt = false, neg_ge = false;
      if (match_c_lt_radius(a, c, diameter_pc, &neg_lt) &&
          match_c_ge_dim_minus_radius(b, c, dim_pc, diameter_pc, &neg_ge)) {
         if (!is_and && !neg_lt && !neg_ge) {
            *inverted_out = false;
            return true;
         }
         if (is_and && neg_lt && neg_ge) {
            /* (c >= radius) && (c < dim - radius) == !in_border */
            *inverted_out = true;
            return true;
         }
      }
   }

   return false;
}

/* Match an SSBO deref indexing a runtime array of elements of byte stride
 * `expected_stride` at struct member offset 0, returning (set, binding, index). */
static bool
match_ssbo_array_deref(nir_deref_instr *deref, unsigned expected_stride,
                       uint32_t *set_out, uint32_t *binding_out,
                       nir_def **index_out)
{
   if (!deref || !nir_deref_mode_is(deref, nir_var_mem_ssbo))
      return false;

   nir_def *elem_index = NULL;
   unsigned elem_stride = 0;

   while (deref) {
      if (deref->deref_type == nir_deref_type_array ||
          deref->deref_type == nir_deref_type_ptr_as_array) {
         if (elem_index != NULL)
            return false;
         elem_index = deref->arr.index.ssa;
         elem_stride = nir_deref_instr_array_stride(deref);
         deref = nir_deref_instr_parent(deref);
      } else if (deref->deref_type == nir_deref_type_struct) {
         nir_deref_instr *parent = nir_deref_instr_parent(deref);
         if (!parent)
            return false;
         if (glsl_get_struct_field_offset(parent->type, deref->strct.index) != 0)
            return false;
         deref = parent;
      } else if (deref->deref_type == nir_deref_type_cast) {
         if (nir_def_instr(deref->parent.ssa)->type == nir_instr_type_deref) {
            deref = nir_src_as_deref(deref->parent);
            continue;
         }
         /* Base must be load_vulkan_descriptor(vulkan_resource_index(0)). */
         if (nir_def_instr(deref->parent.ssa)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *lvd =
            nir_instr_as_intrinsic(nir_def_instr(deref->parent.ssa));
         if (lvd->intrinsic != nir_intrinsic_load_vulkan_descriptor)
            return false;
         if (nir_def_instr(lvd->src[0].ssa)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *vri =
            nir_instr_as_intrinsic(nir_def_instr(lvd->src[0].ssa));
         if (vri->intrinsic != nir_intrinsic_vulkan_resource_index)
            return false;
         if (nir_intrinsic_desc_type(vri) != nir_descriptor_type_storage_buffer)
            return false;
         if (!nir_src_is_const(vri->src[0]) ||
             nir_src_as_uint(vri->src[0]) != 0)
            return false;

         if (!elem_index || elem_stride != expected_stride)
            return false;
         if (elem_index->bit_size != 32 || elem_index->num_components != 1)
            return false;

         *set_out = nir_intrinsic_desc_set(vri);
         *binding_out = nir_intrinsic_binding(vri);
         *index_out = elem_index;
         return true;
      } else {
         return false;
      }
   }

   return false;
}

/* Linear decomposition of an integer index expression in terms of:
 *   c_x * x + c_ys * (y * stride) + c_r * radius + c_rs * (radius * stride)
 *   + c_i * i + c_is * (i * stride)
 * Handles arbitrary additive/subtractive reassociation and distributive
 * `(y - radius) * stride` or `(y - radius + i) * stride`. */
struct linear_idx {
   int c_x;
   int c_ys;
   int c_r;
   int c_rs;
   int c_i;
   int c_is;
};

static bool
decompose_linear_idx(nir_def *def, nir_def *x, nir_def *y, nir_def *stride_pc,
                     nir_def *diameter_pc, nir_def *loop_i,
                     struct linear_idx *out, int sign, unsigned depth)
{
   if (depth > 8)
      return false;

   def = strip_mov(def);
   if (!def)
      return false;

   if (def == x) {
      out->c_x += sign;
      return true;
   }
   if (loop_i && def == loop_i) {
      out->c_i += sign;
      return true;
   }
   if (match_radius_from_diameter(def, diameter_pc)) {
      out->c_r += sign;
      return true;
   }

   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iadd) {
      return decompose_linear_idx(alu->src[0].src.ssa, x, y, stride_pc,
                                  diameter_pc, loop_i, out, sign, depth + 1) &&
             decompose_linear_idx(alu->src[1].src.ssa, x, y, stride_pc,
                                  diameter_pc, loop_i, out, sign, depth + 1);
   }
   if (alu->op == nir_op_isub) {
      return decompose_linear_idx(alu->src[0].src.ssa, x, y, stride_pc,
                                  diameter_pc, loop_i, out, sign, depth + 1) &&
             decompose_linear_idx(alu->src[1].src.ssa, x, y, stride_pc,
                                  diameter_pc, loop_i, out, -sign, depth + 1);
   }
   if (alu->op == nir_op_ineg) {
      return decompose_linear_idx(alu->src[0].src.ssa, x, y, stride_pc,
                                  diameter_pc, loop_i, out, -sign, depth + 1);
   }
   if (alu->op == nir_op_imul) {
      for (unsigned side = 0; side < 2; side++) {
         if (!same_push_const(alu->src[side].src.ssa, stride_pc))
            continue;
         /* Factor * stride: decompose factor in terms of (y, radius, loop_i). */
         struct linear_idx sub = {0};
         /* Temporarily map `y` into `c_x` slot of `sub` to reuse recursion. */
         if (!decompose_linear_idx(alu->src[1 - side].src.ssa, y, NULL,
                                   stride_pc, diameter_pc, loop_i, &sub, sign,
                                   depth + 1))
            return false;
         if (sub.c_ys != 0 || sub.c_rs != 0 || sub.c_is != 0)
            return false;
         out->c_ys += sub.c_x;
         out->c_rs += sub.c_r;
         out->c_is += sub.c_i;
         return true;
      }
   }

   return false;
}

static bool
match_xy_index(nir_def *def, nir_def *x, nir_def *y, nir_def *stride_pc,
               nir_def *diameter_pc)
{
   struct linear_idx lin = {0};
   return decompose_linear_idx(def, x, y, stride_pc, diameter_pc, NULL, &lin, 1,
                               0) &&
          lin.c_x == 1 && lin.c_ys == 1 && lin.c_r == 0 && lin.c_rs == 0 &&
          lin.c_i == 0 && lin.c_is == 0;
}

/* Find the single push constant used as `stride` in `x + y * stride`. */
static bool
extract_stride_from_xy_index(nir_def *def, nir_def *x, nir_def *y,
                             nir_def **stride_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(def));
   if (add->op != nir_op_iadd)
      return false;

   for (unsigned side = 0; side < 2; side++) {
      if (strip_mov(add->src[side].src.ssa) != x)
         continue;
      nir_def *other = strip_mov(add->src[1 - side].src.ssa);
      if (!other || nir_def_instr(other)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(other));
      if (mul->op != nir_op_imul)
         continue;
      for (unsigned ms = 0; ms < 2; ms++) {
         uint16_t off = 0;
         if (strip_mov(mul->src[ms].src.ssa) == y &&
             match_push_const_u32(mul->src[1 - ms].src.ssa, &off)) {
            *stride_out = mul->src[1 - ms].src.ssa;
            return true;
         }
      }
   }

   return false;
}

/* Match a phi in `header` with 2 sources: one from outside `loop` (`init_out`)
 * and one from inside `loop` (`next_out`). */
static bool
match_loop_phi(nir_loop *loop, nir_phi_instr *phi, nir_def **init_out,
               nir_def **next_out)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_block *preheader = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_def *init = NULL, *next = NULL;
   unsigned count = 0;

   nir_foreach_phi_src(src, phi) {
      count++;
      if (src->pred == preheader) {
         init = src->src.ssa;
      } else if (src->pred->index >= header->index &&
                 src->pred->index <= nir_loop_last_block(loop)->index) {
         if (next && next != src->src.ssa)
            return false;
         next = src->src.ssa;
      } else {
         return false;
      }
   }

   if (!init || !next || count < 2)
      return false;
   *init_out = init;
   *next_out = next;
   return true;
}

/* Match a loop-header guard `if (iv >= bound) break;` or
 * `if (iv < bound) {} else { break; }` and return the `if` node. */
static nir_if *
match_loop_header_break(nir_loop *loop, nir_def *iv, nir_def **bound_out)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_cf_node *next_cf = nir_cf_node_next(&header->cf_node);
   if (!next_cf || next_cf->type != nir_cf_node_if)
      return NULL;

   nir_if *nif = nir_cf_node_as_if(next_cf);
   bool is_ge = false;
   if (!match_iv_lt_pc(nif->condition.ssa, iv, bound_out, &is_ge))
      return NULL;

   nir_block *then_blk = nir_if_first_then_block(nif);
   nir_block *else_blk = nir_if_first_else_block(nif);
   bool then_is_break =
      nir_if_first_then_block(nif) == nir_if_last_then_block(nif) &&
      nir_block_ends_in_break(then_blk) &&
      exec_list_is_singular(&then_blk->instr_list);
   bool else_is_break =
      nir_if_first_else_block(nif) == nir_if_last_else_block(nif) &&
      nir_block_ends_in_break(else_blk) &&
      exec_list_is_singular(&else_blk->instr_list);
   bool then_empty =
      nir_if_first_then_block(nif) == nir_if_last_then_block(nif) &&
      exec_list_is_empty(&then_blk->instr_list);
   bool else_empty =
      nir_if_first_else_block(nif) == nir_if_last_else_block(nif) &&
      exec_list_is_empty(&else_blk->instr_list);

   if (is_ge && then_is_break && else_empty)
      return nif;
   if (!is_ge && else_is_break && then_empty)
      return nif;

   return NULL;
}

/* Match `val = iadd(iv, step)` where `step` equals `expected_step` (or constant 1). */
static bool
match_iv_step_const1(nir_def *next, nir_def *iv)
{
   next = strip_mov(next);
   if (!next || nir_def_instr(next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(next));
   if (alu->op != nir_op_iadd)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (strip_mov(alu->src[s].src.ssa) == iv &&
          nir_src_is_const(alu->src[1 - s].src) &&
          nir_src_as_int(alu->src[1 - s].src) == 1)
         return true;
   }
   return false;
}

static bool
match_iv_step_pc(nir_def *next, nir_def *iv, nir_def *step_pc)
{
   next = strip_mov(next);
   if (!next || nir_def_instr(next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(next));
   if (alu->op != nir_op_iadd)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (strip_mov(alu->src[s].src.ssa) == iv &&
          same_push_const(alu->src[1 - s].src.ssa, step_pc))
         return true;
   }
   return false;
}

static bool
match_iv_step_global_size(nir_def *next, nir_def *iv, const uint32_t wg_size[3],
                          unsigned comp)
{
   next = strip_mov(next);
   if (!next || nir_def_instr(next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(next));
   if (alu->op != nir_op_iadd)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (strip_mov(alu->src[s].src.ssa) != iv)
         continue;
      unsigned nw_c = 0;
      if (comp < 3 && wg_size[comp] == 1 &&
          match_sysval_alu_src(&alu->src[1 - s],
                               nir_intrinsic_load_num_workgroups, &nw_c) &&
          nw_c == comp)
         return true;
      if (match_global_size_comp(alu->src[1 - s].src.ssa, wg_size, comp))
         return true;
   }
   return false;
}

/* Match u8 -> f32 conversion (i2f32(u2u32(b)), i2f32(u2u16(b)),
 * u2f32(u2u32(b)), u2f32(u2u16(b)), u2f32(b)).
 *
 * The tiled kernels widen source bytes as unsigned, so a signed conversion is
 * only accepted after an explicit zero-extension: i2f32 applied directly to
 * an 8-bit load reads the byte as int8_t ([-128, 127]) and must not match.
 */
static nir_intrinsic_instr *
match_u8_to_f32_load(nir_def *def)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return NULL;

   nir_alu_instr *cvt = nir_instr_as_alu(nir_def_instr(def));
   if (cvt->op != nir_op_i2f32 && cvt->op != nir_op_u2f32)
      return NULL;

   bool zero_extended = false;
   nir_def *cur = strip_mov(cvt->src[0].src.ssa);
   if (cur && nir_def_instr(cur)->type == nir_instr_type_alu) {
      nir_alu_instr *ext = nir_instr_as_alu(nir_def_instr(cur));
      if (ext->op == nir_op_u2u32 || ext->op == nir_op_u2u16) {
         cur = strip_mov(ext->src[0].src.ssa);
         zero_extended = true;
      }
   }

   if (cvt->op == nir_op_i2f32 && !zero_extended)
      return NULL;

   if (!cur || cur->bit_size != 8 || cur->num_components != 1)
      return NULL;
   if (nir_def_instr(cur)->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *load = nir_instr_as_intrinsic(nir_def_instr(cur));
   if (load->intrinsic != nir_intrinsic_load_deref)
      return NULL;
   return load;
}

/* Match f32 -> u8 store conversion: u2u8(f2i32(sum)) or u2u8(f2u32(sum)). */
static bool
match_f32_to_u8_store_val(nir_def *val, nir_def *sum_def,
                          enum hk_byte_stencil_store_conv *conv_out)
{
   val = strip_mov(val);
   if (!val || val->bit_size != 8 || val->num_components != 1)
      return false;
   if (nir_def_instr(val)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *trunc = nir_instr_as_alu(nir_def_instr(val));
   if (trunc->op != nir_op_u2u8 && trunc->op != nir_op_i2i8)
      return false;

   nir_def *mid = strip_mov(trunc->src[0].src.ssa);
   if (!mid || nir_def_instr(mid)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *f2i = nir_instr_as_alu(nir_def_instr(mid));
   if (f2i->op == nir_op_f2i32 && strip_mov(f2i->src[0].src.ssa) == sum_def) {
      *conv_out = HK_BYTE_STENCIL_CONV_F2I32;
      return true;
   }
   if (f2i->op == nir_op_f2u32 && strip_mov(f2i->src[0].src.ssa) == sum_def) {
      *conv_out = HK_BYTE_STENCIL_CONV_F2U32;
      return true;
   }

   return false;
}

static bool
populate_binding_info(uint32_t set_layout_count,
                      struct vk_descriptor_set_layout *const *set_layouts,
                      uint32_t set, uint32_t binding,
                      struct hk_byte_stencil_binding *out)
{
   if (set >= HK_MAX_SETS || set >= set_layout_count || !set_layouts ||
       !set_layouts[set])
      return false;

   const struct hk_descriptor_set_layout *layout =
      vk_to_hk_descriptor_set_layout(set_layouts[set]);
   if (layout->vk.flags &
       (VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT |
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT))
      return false;
   if (binding >= layout->binding_count)
      return false;

   const struct hk_descriptor_set_binding_layout *bl = &layout->binding[binding];
   if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || bl->array_size != 1)
      return false;
   if (bl->flags &
       (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT))
      return false;

   out->set = set;
   out->binding = binding;
   out->descriptor_offset_B = bl->offset;
   memcpy(out->set_layout_hash, layout->vk.blake3, sizeof(out->set_layout_hash));
   return true;
}

/* Match a grid-stride loop: single IV phi (init = global_id[comp], step =
 * iv + global_size[comp]), header break on `iv < dim_pc`. */
static bool
match_grid_stride_loop(nir_loop *loop, const uint32_t wg_size[3],
                       unsigned *comp_out, nir_def **iv_out,
                       nir_def **dim_pc_out, nir_if **break_if_out)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_phi_instr *iv_phi = NULL;

   nir_foreach_phi(phi, header) {
      if (iv_phi)
         return false; /* Expect only the coordinate IV phi. */
      iv_phi = phi;
   }
   if (!iv_phi || iv_phi->def.bit_size != 32 || iv_phi->def.num_components != 1)
      return false;

   nir_def *init = NULL, *next = NULL;
   if (!match_loop_phi(loop, iv_phi, &init, &next))
      return false;

   unsigned comp = 0;
   if (!match_global_id_comp(init, wg_size, &comp) || comp > 1)
      return false;

   if (!match_iv_step_global_size(next, &iv_phi->def, wg_size, comp))
      return false;

   nir_def *dim_pc = NULL;
   nir_if *break_if = match_loop_header_break(loop, &iv_phi->def, &dim_pc);
   if (!break_if)
      return false;

   *comp_out = comp;
   *iv_out = &iv_phi->def;
   *dim_pc_out = dim_pc;
   *break_if_out = break_if;
   return true;
}

/* Verify the entire shader contains ONLY the 3 matched loads and 2 matched
 * stores, and no barriers, atomics, images, textures, or other side effects. */
static bool
verify_shader_completeness(nir_shader *s,
                           const nir_intrinsic_instr *border_load,
                           const nir_intrinsic_instr *border_store,
                           const nir_intrinsic_instr *stencil_src_load,
                           const nir_intrinsic_instr *stencil_w_load,
                           const nir_intrinsic_instr *interior_store)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   unsigned seen_loads = 0, seen_stores = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;

         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref:
            if (intr == border_load || intr == stencil_src_load ||
                intr == stencil_w_load) {
               seen_loads++;
               break;
            }
            return false;
         case nir_intrinsic_store_deref:
            if (intr == border_store || intr == interior_store) {
               seen_stores++;
               break;
            }
            return false;
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            return false;
         }
      }
   }

   return seen_loads == 3 && seen_stores == 2;
}

/* Find the single loop inside a CF list. Other nodes are skipped here; any
 * unaccounted `if` among them is rejected later by verify_control_flow(). */
static nir_loop *
find_single_loop_in_list(struct exec_list *cf_list)
{
   nir_loop *found = NULL;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_loop) {
         if (found)
            return NULL;
         found = nir_cf_node_as_loop(node);
      }
   }
   return found;
}

static bool
cf_list_is_empty_block(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   if (n->type != nir_cf_node_block)
      return false;
   return exec_list_is_empty(&nir_cf_node_as_block(n)->instr_list);
}

static nir_block *
cf_list_tail_block(struct exec_list *cf_list)
{
   if (exec_list_is_empty(cf_list))
      return NULL;
   nir_cf_node *tail =
      exec_node_data(nir_cf_node, exec_list_get_tail(cf_list), node);
   return tail->type == nir_cf_node_block ? nir_cf_node_as_block(tail) : NULL;
}

/* A `return` here would end the whole invocation's grid-stride walk, which
 * the tiled kernel cannot reproduce, so only `continue` is accepted. */
static bool
cf_list_ends_in_continue(struct exec_list *cf_list)
{
   nir_block *blk = cf_list_tail_block(cf_list);
   nir_instr *last = blk ? nir_block_last_instr(blk) : NULL;
   return last && last->type == nir_instr_type_jump &&
          nir_instr_as_jump(last)->type == nir_jump_continue;
}

/* Every control-flow construct the matcher accounted for. Anything else in
 * the shader (an extra `if`, loop, or jump) could guard or cut short the
 * matched stores and is rejected by verify_control_flow(). */
struct byte_stencil_cf {
   nir_loop *loops[3];   /* outer grid-stride, inner grid-stride, stencil */
   nir_if *break_ifs[3]; /* the header `if (iv >= bound) break;` of each */
   nir_if *border_if;
   /* Block at the end of the border arm that may end in `continue`. */
   nir_block *continue_blk;
   /* Block at the end of an early bounds-guard arm that may end in `return`. */
   nir_block *return_blk;
};

static bool
jump_is_accounted(const nir_jump_instr *jump, nir_block *blk,
                  const struct byte_stencil_cf *cf)
{
   switch (jump->type) {
   case nir_jump_break: {
      nir_cf_node *parent = blk->cf_node.parent;
      if (!parent || parent->type != nir_cf_node_if)
         return false;
      for (unsigned i = 0; i < ARRAY_SIZE(cf->break_ifs); i++) {
         if (parent == &cf->break_ifs[i]->cf_node)
            return true;
      }
      return false;
   }
   case nir_jump_continue:
      return cf->continue_blk && blk == cf->continue_blk &&
             nir_block_last_instr(blk) == &jump->instr;
   case nir_jump_return:
      return cf->return_blk && blk == cf->return_blk &&
             nir_block_last_instr(blk) == &jump->instr;
   default:
      return false;
   }
}

static bool
verify_cf_list(struct exec_list *cf_list, const struct byte_stencil_cf *cf)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_jump &&
                !jump_is_accounted(nir_instr_as_jump(instr), blk, cf))
               return false;
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         bool known = nif == cf->border_if;
         for (unsigned i = 0; i < ARRAY_SIZE(cf->break_ifs); i++)
            known |= nif == cf->break_ifs[i];
         if (!known || !verify_cf_list(&nif->then_list, cf) ||
             !verify_cf_list(&nif->else_list, cf))
            return false;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         bool known = false;
         for (unsigned i = 0; i < ARRAY_SIZE(cf->loops); i++)
            known |= loop == cf->loops[i];
         if (!known || nir_loop_has_continue_construct(loop) ||
             !verify_cf_list(&loop->body, cf))
            return false;
         break;
      }
      default:
         return false;
      }
   }
   return true;
}

/* Reject any `if`, loop, or jump the matcher did not account for, anywhere in
 * the shader (function body, both grid-stride loop bodies, the border and
 * stencil arms, the stencil loop body, and the block after it). */
static bool
verify_control_flow(nir_function_impl *impl, const struct byte_stencil_cf *cf)
{
   return verify_cf_list(&impl->body, cf);
}

/* Find the border `if` and stencil `loop` inside a pixel body CF list. */
static bool
match_pixel_body(nir_shader *s, struct exec_list *body_cf_list,
                 nir_cf_node *start_after, nir_def *x, nir_def *y,
                 nir_def *width_pc, nir_def *height_pc,
                 const struct byte_stencil_cf *grid_cf,
                 uint32_t set_layout_count,
                 struct vk_descriptor_set_layout *const *set_layouts,
                 struct hk_byte_stencil_plan *plan)
{
   /* Walk CF nodes from `start_after` (or head of `body_cf_list`) to find the
    * border selection `nir_if` and an optional trailing `nir_loop` (when the
    * border arm terminates with `continue` or `return`). */
   nir_if *border_if = NULL;
   nir_loop *after_if_loop = NULL;
   nir_cf_node *cur =
      start_after ? nir_cf_node_next(start_after)
                  : exec_node_data(nir_cf_node, exec_list_get_head(body_cf_list),
                                   node);
   while (cur) {
      if (cur->type == nir_cf_node_if) {
         if (border_if)
            return false;
         border_if = nir_cf_node_as_if(cur);
      } else if (cur->type == nir_cf_node_loop) {
         if (!border_if || after_if_loop)
            return false;
         after_if_loop = nir_cf_node_as_loop(cur);
      }
      cur = nir_cf_node_next(cur);
   }
   if (!border_if)
      return false;

   /* Find the stencil loop in one of the arms (or after `border_if` when the
    * border arm ends in `continue` / `return`). */
   nir_loop *stencil_loop = NULL;
   struct exec_list *border_arm = NULL;
   bool border_in_then = true;

   nir_loop *then_loop = find_single_loop_in_list(&border_if->then_list);
   nir_loop *else_loop = find_single_loop_in_list(&border_if->else_list);

   if (then_loop && !else_loop && !after_if_loop) {
      stencil_loop = then_loop;
      border_arm = &border_if->else_list;
      border_in_then = false;
   } else if (else_loop && !then_loop && !after_if_loop) {
      stencil_loop = else_loop;
      border_arm = &border_if->then_list;
      border_in_then = true;
   } else if (!then_loop && !else_loop && after_if_loop) {
      if (cf_list_is_empty_block(&border_if->else_list) &&
          cf_list_ends_in_continue(&border_if->then_list)) {
         stencil_loop = after_if_loop;
         border_arm = &border_if->then_list;
         border_in_then = true;
      } else if (cf_list_is_empty_block(&border_if->then_list) &&
                 cf_list_ends_in_continue(&border_if->else_list)) {
         stencil_loop = after_if_loop;
         border_arm = &border_if->else_list;
         border_in_then = false;
      } else {
         return false;
      }
   } else {
      return false;
   }

   /* Step 1: Extract the border arm's single `dst[idx] = src[idx]` copy and
    * discover `stride_pc`, `src` (set, binding), `dst` (set, binding). */
   nir_intrinsic_instr *border_load = NULL;
   nir_intrinsic_instr *border_store = NULL;

   foreach_list_typed(nir_cf_node, n, node, border_arm) {
      if (n->type != nir_cf_node_block)
         return false;
      nir_block *blk = nir_cf_node_as_block(n);
      nir_foreach_instr(instr, blk) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            if (border_load)
               return false;
            border_load = intr;
         } else if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (border_store)
               return false;
            border_store = intr;
         }
      }
   }

   if (!border_load || !border_store)
      return false;
   if (border_load->def.bit_size != 8 || border_load->def.num_components != 1)
      return false;
   if (strip_mov(border_store->src[1].ssa) != &border_load->def)
      return false;

   uint32_t src_set = 0, src_bind = 0, dst_set = 0, dst_bind = 0;
   nir_def *border_src_idx = NULL, *border_dst_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(border_load->src[0]), 1,
                               &src_set, &src_bind, &border_src_idx))
      return false;
   if (!match_ssbo_array_deref(nir_src_as_deref(border_store->src[0]), 1,
                               &dst_set, &dst_bind, &border_dst_idx))
      return false;
   if (src_set == dst_set && src_bind == dst_bind)
      return false;

   nir_def *stride_pc = NULL;
   if (!extract_stride_from_xy_index(border_src_idx, x, y, &stride_pc))
      return false;

   /* Step 2: Inspect the stencil loop to discover `diameter_pc`, `weights`
    * (set, binding), `axis`, and verify the exact MAC accumulation. */
   nir_block *sl_header = nir_loop_first_block(stencil_loop);
   nir_phi_instr *i_phi = NULL, *sum_phi = NULL, *off_phi = NULL;

   nir_foreach_phi(phi, sl_header) {
      if (phi->def.bit_size != 32 || phi->def.num_components != 1)
         return false;
      nir_def *init = NULL, *next = NULL;
      if (!match_loop_phi(stencil_loop, phi, &init, &next))
         return false;

      if (nir_src_is_const(nir_src_for_ssa(init)) &&
          nir_src_as_int(nir_src_for_ssa(init)) == 0 &&
          match_iv_step_const1(next, &phi->def)) {
         if (i_phi)
            return false;
         i_phi = phi;
      } else if (nir_src_is_const(nir_src_for_ssa(init)) &&
                 nir_src_as_float(nir_src_for_ssa(init)) == 0.0) {
         /* Could be sum_phi (0.0f == int 0 bit pattern, so check float add/fma). */
         nir_def *nm = strip_mov(next);
         if (nm && nir_def_instr(nm)->type == nir_instr_type_alu) {
            nir_op op = nir_instr_as_alu(nir_def_instr(nm))->op;
            if (op == nir_op_fadd || op == nir_op_ffma) {
               if (sum_phi)
                  return false;
               sum_phi = phi;
               continue;
            }
         }
         return false;
      } else {
         if (off_phi)
            return false;
         off_phi = phi;
      }
   }

   if (!i_phi || !sum_phi)
      return false;

   nir_def *diameter_pc = NULL;
   nir_if *sl_break =
      match_loop_header_break(stencil_loop, &i_phi->def, &diameter_pc);
   if (!sl_break)
      return false;

   /* Now verify `border_dst_idx` and `border_src_idx` are both `x + y * stride`. */
   if (!match_xy_index(border_src_idx, x, y, stride_pc, diameter_pc) ||
       !match_xy_index(border_dst_idx, x, y, stride_pc, diameter_pc))
      return false;

   /* Step 3: Verify `border_if->condition` matches H or V border check! */
   enum hk_byte_stencil_axis axis;
   bool border_cond_inverted = false;
   if (match_border_cond(border_if->condition.ssa, x, width_pc, diameter_pc,
                         &border_cond_inverted)) {
      axis = HK_BYTE_STENCIL_HORIZ;
   } else if (match_border_cond(border_if->condition.ssa, y, height_pc,
                                diameter_pc, &border_cond_inverted)) {
      axis = HK_BYTE_STENCIL_VERT;
   } else {
      return false;
   }

   /* Border condition polarity must match which arm held the border copy. */
   if (border_cond_inverted == border_in_then)
      return false;

   /* Step 4: Verify stencil loop MAC: `sum_next = sum + float(src[...]) * w[i]`. */
   nir_def *sum_init = NULL, *sum_next = NULL;
   if (!match_loop_phi(stencil_loop, sum_phi, &sum_init, &sum_next))
      return false;

   sum_next = strip_mov(sum_next);
   if (!sum_next || nir_def_instr(sum_next)->type != nir_instr_type_alu)
      return false;

   /* The tiled kernels always emit ffma, so the accumulation must allow
    * contraction: reject NoContraction / precise (exact) arithmetic. */
   nir_alu_instr *acc = nir_instr_as_alu(nir_def_instr(sum_next));
   if (nir_alu_instr_is_exact(acc))
      return false;

   nir_def *mul_a = NULL, *mul_b = NULL;
   if (acc->op == nir_op_fadd) {
      nir_def *prod = NULL;
      if (strip_mov(acc->src[0].src.ssa) == &sum_phi->def)
         prod = strip_mov(acc->src[1].src.ssa);
      else if (strip_mov(acc->src[1].src.ssa) == &sum_phi->def)
         prod = strip_mov(acc->src[0].src.ssa);
      else
         return false;

      if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu ||
          nir_instr_as_alu(nir_def_instr(prod))->op != nir_op_fmul ||
          nir_alu_instr_is_exact(nir_instr_as_alu(nir_def_instr(prod))))
         return false;
      mul_a = nir_instr_as_alu(nir_def_instr(prod))->src[0].src.ssa;
      mul_b = nir_instr_as_alu(nir_def_instr(prod))->src[1].src.ssa;
   } else if (acc->op == nir_op_ffma) {
      if (strip_mov(acc->src[2].src.ssa) != &sum_phi->def)
         return false;
      mul_a = acc->src[0].src.ssa;
      mul_b = acc->src[1].src.ssa;
   } else {
      return false;
   }

   nir_intrinsic_instr *stencil_src_load = match_u8_to_f32_load(mul_a);
   nir_def *w_def = strip_mov(mul_b);
   if (!stencil_src_load) {
      stencil_src_load = match_u8_to_f32_load(mul_b);
      w_def = strip_mov(mul_a);
   }
   if (!stencil_src_load || !w_def ||
       nir_def_instr(w_def)->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *stencil_w_load =
      nir_instr_as_intrinsic(nir_def_instr(w_def));
   if (stencil_w_load->intrinsic != nir_intrinsic_load_deref ||
       stencil_w_load->def.bit_size != 32 ||
       stencil_w_load->def.num_components != 1)
      return false;

   /* Verify weights load is `weights[i]` (stride 4). */
   uint32_t w_set = 0, w_bind = 0;
   nir_def *w_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(stencil_w_load->src[0]), 4,
                               &w_set, &w_bind, &w_idx))
      return false;
   if (strip_mov(w_idx) != &i_phi->def)
      return false;
   if (w_set == dst_set && w_bind == dst_bind)
      return false;

   /* Verify stencil src load matches `src` binding and exact axis stride. */
   uint32_t sl_src_set = 0, sl_src_bind = 0;
   nir_def *sl_src_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(stencil_src_load->src[0]), 1,
                               &sl_src_set, &sl_src_bind, &sl_src_idx))
      return false;
   if (sl_src_set != src_set || sl_src_bind != src_bind)
      return false;

   sl_src_idx = strip_mov(sl_src_idx);
   if (off_phi && sl_src_idx == &off_phi->def) {
      nir_def *off_init = NULL, *off_next = NULL;
      if (!match_loop_phi(stencil_loop, off_phi, &off_init, &off_next))
         return false;
      struct linear_idx lin = {0};
      if (!decompose_linear_idx(off_init, x, y, stride_pc, diameter_pc, NULL,
                                &lin, 1, 0))
         return false;
      if (axis == HK_BYTE_STENCIL_HORIZ) {
         if (!(lin.c_x == 1 && lin.c_ys == 1 && lin.c_r == -1 &&
               lin.c_rs == 0 && lin.c_i == 0 && lin.c_is == 0))
            return false;
         if (!match_iv_step_const1(off_next, &off_phi->def))
            return false;
      } else {
         if (!(lin.c_x == 1 && lin.c_ys == 1 && lin.c_r == 0 &&
               lin.c_rs == -1 && lin.c_i == 0 && lin.c_is == 0))
            return false;
         if (!match_iv_step_pc(off_next, &off_phi->def, stride_pc))
            return false;
      }
   } else if (!off_phi) {
      struct linear_idx lin = {0};
      if (!decompose_linear_idx(sl_src_idx, x, y, stride_pc, diameter_pc,
                                &i_phi->def, &lin, 1, 0))
         return false;
      if (axis == HK_BYTE_STENCIL_HORIZ) {
         if (!(lin.c_x == 1 && lin.c_ys == 1 && lin.c_r == -1 &&
               lin.c_rs == 0 && lin.c_i == 1 && lin.c_is == 0))
            return false;
      } else {
         if (!(lin.c_x == 1 && lin.c_ys == 1 && lin.c_r == 0 &&
               lin.c_rs == -1 && lin.c_i == 0 && lin.c_is == 1))
            return false;
      }
   } else {
      return false;
   }

   /* Step 5: Verify the interior store right after `stencil_loop`. */
   nir_cf_node *after_loop_cf = nir_cf_node_next(&stencil_loop->cf_node);
   if (!after_loop_cf || after_loop_cf->type != nir_cf_node_block)
      return false;

   nir_block *after_blk = nir_cf_node_as_block(after_loop_cf);
   nir_intrinsic_instr *interior_store = NULL;
   nir_foreach_instr(instr, after_blk) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (interior_store)
               return false;
            interior_store = intr;
         }
      }
   }
   if (!interior_store)
      return false;

   uint32_t int_dst_set = 0, int_dst_bind = 0;
   nir_def *int_dst_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(interior_store->src[0]), 1,
                               &int_dst_set, &int_dst_bind, &int_dst_idx))
      return false;
   if (int_dst_set != dst_set || int_dst_bind != dst_bind)
      return false;
   if (!match_xy_index(int_dst_idx, x, y, stride_pc, diameter_pc))
      return false;

   /* The value stored must convert `sum_phi` (at loop exit) from f32 to u8. */
   enum hk_byte_stencil_store_conv store_conv;
   if (!match_f32_to_u8_store_val(interior_store->src[1].ssa, &sum_phi->def,
                                  &store_conv))
      return false;

   /* Step 6: Strict completeness check across the entire shader: no
    * unaccounted control flow, loads, stores, or side effects. */
   struct byte_stencil_cf cf = *grid_cf;
   cf.loops[2] = stencil_loop;
   cf.break_ifs[2] = sl_break;
   cf.border_if = border_if;
   cf.continue_blk = cf_list_tail_block(border_arm);
   if (!verify_control_flow(nir_shader_get_entrypoint(s), &cf))
      return false;

   if (!verify_shader_completeness(s, border_load, border_store,
                                   stencil_src_load, stencil_w_load,
                                   interior_store))
      return false;

   /* Step 7: Populate descriptor bindings & push constant offsets. */
   uint16_t width_off = 0, height_off = 0, stride_off = 0, diam_off = 0;
   if (!match_push_const_u32(width_pc, &width_off) ||
       !match_push_const_u32(height_pc, &height_off) ||
       !match_push_const_u32(stride_pc, &stride_off) ||
       !match_push_const_u32(diameter_pc, &diam_off))
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->src) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->dst) ||
       !populate_binding_info(set_layout_count, set_layouts, w_set, w_bind,
                              &plan->weights))
      return false;

   plan->axis = (uint8_t)axis;
   plan->store_conv = (uint8_t)store_conv;
   plan->elem_kind = HK_BYTE_STENCIL_ELEM_U8;
   plan->width_pc_offset_B = width_off;
   plan->height_pc_offset_B = height_off;
   plan->stride_pc_offset_B = stride_off;
   plan->diameter_pc_offset_B = diam_off;
   if (axis == HK_BYTE_STENCIL_HORIZ) {
      plan->tile_w = 128;
      plan->tile_h = 4;
   } else {
      plan->tile_w = 32;
      plan->tile_h = 64;
   }

   return true;
}

static bool
match_global_id_alu_src(const nir_alu_src *asrc, const uint32_t wg_size[3],
                        unsigned *comp_out)
{
   unsigned c = 0;
   if (match_sysval_alu_src(asrc, nir_intrinsic_load_global_invocation_id,
                            &c)) {
      *comp_out = c;
      return true;
   }
   if (match_sysval_alu_src(asrc, nir_intrinsic_load_workgroup_id, &c) &&
       c < 3 && wg_size[c] == 1) {
      *comp_out = c;
      return true;
   }

   nir_def *src = asrc->src.ssa;
   if (src && src->num_components > 1 &&
       nir_def_instr(src)->type == nir_instr_type_alu) {
      nir_alu_instr *vadd = nir_instr_as_alu(nir_def_instr(src));
      if (vadd->op == nir_op_iadd) {
         unsigned sw = asrc->swizzle[0];
         nir_def *s0 = vadd->src[0].src.ssa;
         nir_def *s1 = vadd->src[1].src.ssa;
         if (nir_def_instr(s0)->type == nir_instr_type_intrinsic &&
             nir_def_instr(s1)->type == nir_instr_type_intrinsic) {
            nir_intrinsic_op op0 =
               nir_instr_as_intrinsic(nir_def_instr(s0))->intrinsic;
            nir_intrinsic_op op1 =
               nir_instr_as_intrinsic(nir_def_instr(s1))->intrinsic;
            if ((op0 == nir_intrinsic_load_global_invocation_id &&
                 op1 == nir_intrinsic_load_base_global_invocation_id) ||
                (op1 == nir_intrinsic_load_global_invocation_id &&
                 op0 == nir_intrinsic_load_base_global_invocation_id)) {
               *comp_out = sw;
               return true;
            }
            if (sw < 3 && wg_size[sw] == 1 &&
                ((op0 == nir_intrinsic_load_workgroup_id &&
                  op1 == nir_intrinsic_load_base_workgroup_id) ||
                 (op1 == nir_intrinsic_load_workgroup_id &&
                  op0 == nir_intrinsic_load_base_workgroup_id))) {
               *comp_out = sw;
               return true;
            }
         }
      }
   }

   return match_global_id_comp(src, wg_size, comp_out);
}

static bool
same_global_coord_src(const nir_alu_src *asrc, const uint32_t wg_size[3],
                      unsigned expected_comp)
{
   unsigned comp = 0;
   return match_global_id_alu_src(asrc, wg_size, &comp) && comp == expected_comp;
}

static bool
extract_diameter_pc_from_radius(nir_def *def, nir_def **diameter_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   nir_def *cand = NULL;
   uint16_t off = 0;

   if (alu->op == nir_op_idiv) {
      cand = strip_mov(alu->src[0].src.ssa);
   } else if (alu->op == nir_op_ishr || alu->op == nir_op_ushr) {
      nir_def *lhs = strip_mov(alu->src[0].src.ssa);
      if (match_push_const_u32(lhs, &off)) {
         cand = lhs;
      } else if (lhs && nir_def_instr(lhs)->type == nir_instr_type_alu) {
         nir_alu_instr *inner = nir_instr_as_alu(nir_def_instr(lhs));
         if (inner->op == nir_op_iadd || inner->op == nir_op_isub) {
            if (match_push_const_u32(strip_mov(inner->src[0].src.ssa), &off))
               cand = strip_mov(inner->src[0].src.ssa);
            else if (inner->op == nir_op_iadd &&
                     match_push_const_u32(strip_mov(inner->src[1].src.ssa), &off))
               cand = strip_mov(inner->src[1].src.ssa);
         }
      }
   } else if (alu->op == nir_op_imul) {
      for (unsigned s = 0; s < 2 && !cand; s++) {
         nir_def *opnd = strip_mov(alu->src[s].src.ssa);
         if (!opnd || nir_def_instr(opnd)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(opnd));
         if (sa->op == nir_op_ishr || sa->op == nir_op_ushr) {
            nir_def *d = extract_pc_from_iabs(sa->src[0].src.ssa);
            if (d)
               cand = d;
         }
      }
   }

   if (!cand || !match_push_const_u32(cand, &off) ||
       !match_radius_from_diameter(def, cand))
      return false;

   *diameter_pc_out = cand;
   return true;
}

static bool
match_global_coord_cmp_pc(nir_def *cond, const uint32_t wg_size[3],
                          unsigned *comp_out, nir_def **dim_pc_out,
                          bool *is_ge_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub_ge = false;
      if (!match_global_coord_cmp_pc(alu->src[0].src.ssa, wg_size, comp_out,
                                     dim_pc_out, &sub_ge))
         return false;
      *is_ge_out = !sub_ge;
      return true;
   }

   if (alu->op != nir_op_ilt && alu->op != nir_op_ige &&
       alu->op != nir_op_ult && alu->op != nir_op_uge)
      return false;

   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   unsigned comp = 0;
   uint16_t off = 0;
   if (match_global_id_alu_src(&alu->src[0], wg_size, &comp) && comp <= 1 &&
       match_push_const_u32(s1, &off)) {
      *comp_out = comp;
      *dim_pc_out = s1;
      *is_ge_out = (alu->op == nir_op_ige || alu->op == nir_op_uge);
      return true;
   }

   return false;
}

/* Match `(x >= width || y >= height)` (`in_bounds_out = false`) or
 * `(x < width && y < height)` (`in_bounds_out = true`), including short-circuit
 * `bcsel` lowerings. */
static bool
match_bounds_guard_cond(nir_def *cond, const uint32_t wg_size[3],
                        nir_def **width_pc_out, nir_def **height_pc_out,
                        bool *in_bounds_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub = false;
      if (!match_bounds_guard_cond(alu->src[0].src.ssa, wg_size, width_pc_out,
                                   height_pc_out, &sub))
         return false;
      *in_bounds_out = !sub;
      return true;
   }

   nir_def *p0 = NULL, *p1 = NULL;
   bool is_and = false;
   bool invert_p0 = false;

   if (alu->op == nir_op_ior) {
      p0 = alu->src[0].src.ssa;
      p1 = alu->src[1].src.ssa;
      is_and = false;
   } else if (alu->op == nir_op_iand) {
      p0 = alu->src[0].src.ssa;
      p1 = alu->src[1].src.ssa;
      is_and = true;
   } else if (alu->op == nir_op_bcsel) {
      if (nir_src_is_const(alu->src[1].src) &&
          nir_src_as_bool(alu->src[1].src)) {
         /* a ? true : b == a || b */
         p0 = alu->src[0].src.ssa;
         p1 = alu->src[2].src.ssa;
         is_and = false;
      } else if (nir_src_is_const(alu->src[2].src) &&
                 nir_src_as_bool(alu->src[2].src)) {
         /* a ? b : true == !a || b */
         p0 = alu->src[0].src.ssa;
         p1 = alu->src[1].src.ssa;
         is_and = false;
         invert_p0 = true;
      } else if (nir_src_is_const(alu->src[2].src) &&
                 !nir_src_as_bool(alu->src[2].src)) {
         /* a ? b : false == a && b */
         p0 = alu->src[0].src.ssa;
         p1 = alu->src[1].src.ssa;
         is_and = true;
      } else if (nir_src_is_const(alu->src[1].src) &&
                 !nir_src_as_bool(alu->src[1].src)) {
         /* a ? false : b == !a && b */
         p0 = alu->src[0].src.ssa;
         p1 = alu->src[2].src.ssa;
         is_and = true;
         invert_p0 = true;
      } else {
         /* Short-circuit lowering without constant folding:
          *   bcsel(a, a, b)  == a || b
          *   bcsel(!a, b, a) == a || b
          *   bcsel(a, b, a)  == a && b
          *   bcsel(!a, a, b) == a && b */
         unsigned cc = 0, ct = 0, cf = 0;
         nir_def *dim_c = NULL, *dim_t = NULL, *dim_f = NULL;
         bool ge_c = false, ge_t = false, ge_f = false;
         if (!match_global_coord_cmp_pc(alu->src[0].src.ssa, wg_size, &cc,
                                        &dim_c, &ge_c) ||
             !match_global_coord_cmp_pc(alu->src[1].src.ssa, wg_size, &ct,
                                        &dim_t, &ge_t) ||
             !match_global_coord_cmp_pc(alu->src[2].src.ssa, wg_size, &cf,
                                        &dim_f, &ge_f))
            return false;

         if (cc == ct && same_push_const(dim_c, dim_t)) {
            if (ge_c == ge_t) {
               /* a ? a : b == a || b */
               p0 = alu->src[1].src.ssa;
               p1 = alu->src[2].src.ssa;
               is_and = false;
            } else {
               /* !a ? a : b == a && b */
               p0 = alu->src[1].src.ssa;
               p1 = alu->src[2].src.ssa;
               is_and = true;
            }
         } else if (cc == cf && same_push_const(dim_c, dim_f)) {
            if (ge_c != ge_f) {
               /* !a ? b : a == a || b */
               p0 = alu->src[2].src.ssa;
               p1 = alu->src[1].src.ssa;
               is_and = false;
            } else {
               /* a ? b : a == a && b */
               p0 = alu->src[2].src.ssa;
               p1 = alu->src[1].src.ssa;
               is_and = true;
            }
         } else {
            return false;
         }
      }
   } else {
      return false;
   }

   unsigned c0 = 0, c1 = 0;
   nir_def *dim0 = NULL, *dim1 = NULL;
   bool ge0 = false, ge1 = false;
   if (!match_global_coord_cmp_pc(p0, wg_size, &c0, &dim0, &ge0) ||
       !match_global_coord_cmp_pc(p1, wg_size, &c1, &dim1, &ge1) ||
       c0 == c1)
      return false;

   if (invert_p0)
      ge0 = !ge0;

   if (!is_and && ge0 && ge1) {
      *in_bounds_out = false;
   } else if (is_and && !ge0 && !ge1) {
      *in_bounds_out = true;
   } else {
      return false;
   }

   *width_pc_out = (c0 == 0) ? dim0 : dim1;
   *height_pc_out = (c0 == 1) ? dim0 : dim1;
   return true;
}

static bool
cf_list_is_single_return(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *blk = nir_cf_node_as_block(n);
   if (!exec_list_is_singular(&blk->instr_list))
      return false;
   nir_instr *instr = nir_block_first_instr(blk);
   return instr->type == nir_instr_type_jump &&
          nir_instr_as_jump(instr)->type == nir_jump_return;
}

static bool
match_c_minus_radius(nir_def *def, const uint32_t wg_size[3],
                     unsigned *comp_out, nir_def **diameter_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_isub) {
      nir_def *rhs = strip_mov(alu->src[1].src.ssa);
      nir_def *diam = NULL;
      if (!extract_diameter_pc_from_radius(rhs, &diam))
         return false;
      unsigned comp = 0;
      if (match_global_id_alu_src(&alu->src[0], wg_size, &comp) && comp <= 1) {
         *comp_out = comp;
         *diameter_pc_out = diam;
         return true;
      }
   } else if (alu->op == nir_op_iadd) {
      for (unsigned s = 0; s < 2; s++) {
         nir_def *neg = strip_mov(alu->src[1 - s].src.ssa);
         if (!neg || nir_def_instr(neg)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(neg))->op != nir_op_ineg)
            continue;
         nir_def *diam = NULL;
         if (!extract_diameter_pc_from_radius(
                nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa, &diam))
            continue;
         unsigned comp = 0;
         if (match_global_id_alu_src(&alu->src[s], wg_size, &comp) &&
             comp <= 1) {
            *comp_out = comp;
            *diameter_pc_out = diam;
            return true;
         }
      }
   }

   return false;
}

static bool
match_clamped_min(nir_def *def, const uint32_t wg_size[3], unsigned *comp_out,
                  nir_def **diameter_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_imax) {
      for (unsigned s = 0; s < 2; s++) {
         if (nir_src_is_const(alu->src[s].src) &&
             nir_src_as_int(alu->src[s].src) == 0 &&
             match_c_minus_radius(alu->src[1 - s].src.ssa, wg_size, comp_out,
                                  diameter_pc_out))
            return true;
      }
      return false;
   }

   if (alu->op == nir_op_bcsel) {
      nir_def *cond = strip_mov(alu->src[0].src.ssa);
      if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *cmp = nir_instr_as_alu(nir_def_instr(cond));
      nir_def *t = strip_mov(alu->src[1].src.ssa);
      nir_def *f = strip_mov(alu->src[2].src.ssa);
      nir_def *sub = NULL;
      bool zero_on_true = false;
      if (nir_src_is_const(alu->src[1].src) &&
          nir_src_as_int(alu->src[1].src) == 0) {
         zero_on_true = true;
         sub = f;
      } else if (nir_src_is_const(alu->src[2].src) &&
                 nir_src_as_int(alu->src[2].src) == 0) {
         zero_on_true = false;
         sub = t;
      } else {
         return false;
      }

      nir_def *c0 = strip_mov(cmp->src[0].src.ssa);
      nir_def *c1 = strip_mov(cmp->src[1].src.ssa);
      bool ok = false;
      if (zero_on_true) {
         /* (sub < 0) ? 0 : sub   or   (0 >= sub) ? 0 : sub */
         ok = (cmp->op == nir_op_ilt && c0 == sub &&
               nir_src_is_const(cmp->src[1].src) &&
               nir_src_as_int(cmp->src[1].src) == 0) ||
              (cmp->op == nir_op_ige && c1 == sub &&
               nir_src_is_const(cmp->src[0].src) &&
               nir_src_as_int(cmp->src[0].src) == 0);
      } else {
         /* (sub >= 0) ? sub : 0  or   (0 < sub) ? sub : 0 */
         ok = (cmp->op == nir_op_ige && c0 == sub &&
               nir_src_is_const(cmp->src[1].src) &&
               nir_src_as_int(cmp->src[1].src) == 0) ||
              (cmp->op == nir_op_ilt && c1 == sub &&
               nir_src_is_const(cmp->src[0].src) &&
               nir_src_as_int(cmp->src[0].src) == 0);
      }
      if (ok && match_c_minus_radius(sub, wg_size, comp_out, diameter_pc_out))
         return true;
   }

   return false;
}

static bool
match_c_plus_radius(nir_def *def, const uint32_t wg_size[3], unsigned comp,
                    nir_def *diameter_pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_iadd)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (same_global_coord_src(&alu->src[s], wg_size, comp) &&
          match_radius_from_diameter(alu->src[1 - s].src.ssa, diameter_pc))
         return true;
   }
   return false;
}

static bool
match_pc_minus_1(nir_def *def, nir_def *dim_pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_isub) {
      return same_push_const(alu->src[0].src.ssa, dim_pc) &&
             nir_src_is_const(alu->src[1].src) &&
             nir_src_as_int(alu->src[1].src) == 1;
   }
   if (alu->op == nir_op_iadd) {
      for (unsigned s = 0; s < 2; s++) {
         if (same_push_const(alu->src[s].src.ssa, dim_pc) &&
             nir_src_is_const(alu->src[1 - s].src) &&
             nir_src_as_int(alu->src[1 - s].src) == -1)
            return true;
      }
   }
   return false;
}

static bool
match_clamped_max(nir_def *def, const uint32_t wg_size[3], unsigned comp,
                  nir_def *dim_pc, nir_def *diameter_pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_imin) {
      for (unsigned s = 0; s < 2; s++) {
         if (match_pc_minus_1(alu->src[s].src.ssa, dim_pc) &&
             match_c_plus_radius(alu->src[1 - s].src.ssa, wg_size, comp,
                                 diameter_pc))
            return true;
      }
      return false;
   }

   if (alu->op == nir_op_bcsel) {
      nir_def *cond = strip_mov(alu->src[0].src.ssa);
      if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *cmp = nir_instr_as_alu(nir_def_instr(cond));
      nir_def *t = strip_mov(alu->src[1].src.ssa);
      nir_def *f = strip_mov(alu->src[2].src.ssa);

      nir_def *add_def = NULL, *dim_m1 = NULL;
      bool clamp_on_true = false;
      if (match_pc_minus_1(t, dim_pc) &&
          match_c_plus_radius(f, wg_size, comp, diameter_pc)) {
         clamp_on_true = true;
         dim_m1 = t;
         add_def = f;
      } else if (match_pc_minus_1(f, dim_pc) &&
                 match_c_plus_radius(t, wg_size, comp, diameter_pc)) {
         clamp_on_true = false;
         dim_m1 = f;
         add_def = t;
      } else {
         return false;
      }

      nir_def *c0 = strip_mov(cmp->src[0].src.ssa);
      nir_def *c1 = strip_mov(cmp->src[1].src.ssa);
      if (clamp_on_true) {
         /* (dim_m1 < add) or (add >= dim_pc) -> dim_m1 */
         return (cmp->op == nir_op_ilt &&
                 (c0 == dim_m1 || match_pc_minus_1(c0, dim_pc)) &&
                 (c1 == add_def ||
                  match_c_plus_radius(c1, wg_size, comp, diameter_pc))) ||
                (cmp->op == nir_op_ige &&
                 (c0 == add_def ||
                  match_c_plus_radius(c0, wg_size, comp, diameter_pc)) &&
                 same_push_const(c1, dim_pc));
      } else {
         /* (dim_m1 >= add) or (add < dim_pc) -> add */
         return (cmp->op == nir_op_ige &&
                 (c0 == dim_m1 || match_pc_minus_1(c0, dim_pc)) &&
                 (c1 == add_def ||
                  match_c_plus_radius(c1, wg_size, comp, diameter_pc))) ||
                (cmp->op == nir_op_ilt &&
                 (c0 == add_def ||
                  match_c_plus_radius(c0, wg_size, comp, diameter_pc)) &&
                 same_push_const(c1, dim_pc));
      }
   }

   return false;
}

static bool
match_iv_gt_bound(nir_def *cond, nir_def *iv, nir_def **bound_out,
                  bool *is_gt_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_inot) {
      bool sub_gt = false;
      if (!match_iv_gt_bound(alu->src[0].src.ssa, iv, bound_out, &sub_gt))
         return false;
      *is_gt_out = !sub_gt;
      return true;
   }

   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   /* NIR has no igt/ile: `iv > bound` is `ilt(bound, iv)` and `iv <= bound` is
    * `ige(bound, iv)`. */
   if (s1 == iv && (alu->op == nir_op_ilt || alu->op == nir_op_ige)) {
      *bound_out = s0;
      *is_gt_out = (alu->op == nir_op_ilt);
      return true;
   }

   return false;
}

static nir_if *
match_clamped_loop_break(nir_loop *loop, nir_def *c_iv, nir_def **c_max_out)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_cf_node *next_cf = nir_cf_node_next(&header->cf_node);
   if (!next_cf || next_cf->type != nir_cf_node_if)
      return NULL;

   nir_if *nif = nir_cf_node_as_if(next_cf);
   bool is_gt = false;
   if (!match_iv_gt_bound(nif->condition.ssa, c_iv, c_max_out, &is_gt))
      return NULL;

   nir_block *then_blk = nir_if_first_then_block(nif);
   nir_block *else_blk = nir_if_first_else_block(nif);
   bool then_is_break =
      nir_if_first_then_block(nif) == nir_if_last_then_block(nif) &&
      nir_block_ends_in_break(then_blk) &&
      exec_list_is_singular(&then_blk->instr_list);
   bool else_is_break =
      nir_if_first_else_block(nif) == nir_if_last_else_block(nif) &&
      nir_block_ends_in_break(else_blk) &&
      exec_list_is_singular(&else_blk->instr_list);

   /* Accept either an empty continuation arm (`if (c > c_max) break; body`) or
    * the loop body wrapped inside the non-break arm (`if (c <= c_max) { body }
    * else { break; }`, as spirv_to_nir emits for `while (c <= c_max)` loops);
    * verify_control_flow() rejects any unexpected CF inside the body arm. */
   if (is_gt && then_is_break && !else_is_break)
      return nif;
   if (!is_gt && else_is_break && !then_is_break)
      return nif;
   return NULL;
}

/* Match extraction of unsigned byte channel `ch` (0..3) from a 32-bit SSBO
 * pixel load, converted to f32. */
static nir_intrinsic_instr *
match_u32_byte_to_f32_load(nir_def *def, unsigned *ch_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return NULL;
   nir_alu_instr *cvt = nir_instr_as_alu(nir_def_instr(def));
   if (cvt->op != nir_op_u2f32 && cvt->op != nir_op_i2f32)
      return NULL;

   nir_def *e = strip_mov(cvt->src[0].src.ssa);
   if (!e || nir_def_instr(e)->type != nir_instr_type_alu)
      return NULL;
   nir_alu_instr *ea = nir_instr_as_alu(nir_def_instr(e));

   nir_def *pixel = NULL;
   unsigned ch = 0;

   if (ea->op == nir_op_extract_u8) {
      if (!nir_src_is_const(ea->src[1].src))
         return NULL;
      uint32_t idx = nir_src_as_uint(ea->src[1].src);
      if (idx > 3)
         return NULL;
      pixel = strip_mov(ea->src[0].src.ssa);
      ch = idx;
   } else if (ea->op == nir_op_ubitfield_extract) {
      if (!nir_src_is_const(ea->src[1].src) ||
          !nir_src_is_const(ea->src[2].src) ||
          nir_src_as_uint(ea->src[2].src) != 8)
         return NULL;
      uint32_t off = nir_src_as_uint(ea->src[1].src);
      if (off != 0 && off != 8 && off != 16 && off != 24)
         return NULL;
      pixel = strip_mov(ea->src[0].src.ssa);
      ch = off / 8;
   } else if (ea->op == nir_op_ushr) {
      if (!nir_src_is_const(ea->src[1].src))
         return NULL;
      uint32_t sh = nir_src_as_uint(ea->src[1].src);
      nir_def *s0 = strip_mov(ea->src[0].src.ssa);
      if (sh == 24) {
         pixel = s0;
         ch = 3;
      } else if ((sh == 8 || sh == 16) && s0 &&
                 nir_def_instr(s0)->type == nir_instr_type_alu &&
                 nir_instr_as_alu(nir_def_instr(s0))->op == nir_op_iand) {
         nir_alu_instr *and_a = nir_instr_as_alu(nir_def_instr(s0));
         uint32_t expect_mask = 0xffu << sh;
         for (unsigned side = 0; side < 2; side++) {
            if (nir_src_is_const(and_a->src[side].src) &&
                nir_src_as_uint(and_a->src[side].src) == expect_mask) {
               pixel = strip_mov(and_a->src[1 - side].src.ssa);
               ch = sh / 8;
               break;
            }
         }
      }
   } else if (ea->op == nir_op_iand) {
      nir_def *val = NULL;
      for (unsigned side = 0; side < 2; side++) {
         if (nir_src_is_const(ea->src[side].src) &&
             nir_src_as_uint(ea->src[side].src) == 0xffu) {
            val = strip_mov(ea->src[1 - side].src.ssa);
            break;
         }
      }
      if (!val)
         return NULL;
      if (nir_def_instr(val)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(val))->op == nir_op_ushr) {
         nir_alu_instr *shr = nir_instr_as_alu(nir_def_instr(val));
         if (!nir_src_is_const(shr->src[1].src))
            return NULL;
         uint32_t sh = nir_src_as_uint(shr->src[1].src);
         if (sh != 8 && sh != 16 && sh != 24)
            return NULL;
         nir_def *base = strip_mov(shr->src[0].src.ssa);
         if (sh == 8 && base &&
             nir_def_instr(base)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(base))->op == nir_op_extract_u16) {
            nir_alu_instr *eu16 = nir_instr_as_alu(nir_def_instr(base));
            if (!nir_src_is_const(eu16->src[1].src))
               return NULL;
            uint32_t w = nir_src_as_uint(eu16->src[1].src);
            if (w > 1)
               return NULL;
            pixel = strip_mov(eu16->src[0].src.ssa);
            ch = w * 2 + 1;
         } else {
            pixel = base;
            ch = sh / 8;
         }
      } else if (nir_def_instr(val)->type == nir_instr_type_alu &&
                 nir_instr_as_alu(nir_def_instr(val))->op ==
                    nir_op_extract_u16) {
         nir_alu_instr *eu16 = nir_instr_as_alu(nir_def_instr(val));
         if (!nir_src_is_const(eu16->src[1].src))
            return NULL;
         uint32_t w = nir_src_as_uint(eu16->src[1].src);
         if (w > 1)
            return NULL;
         pixel = strip_mov(eu16->src[0].src.ssa);
         ch = w * 2;
      } else {
         pixel = val;
         ch = 0;
      }
   }

   if (!pixel || pixel->bit_size != 32 || pixel->num_components != 1)
      return NULL;
   if (nir_def_instr(pixel)->type != nir_instr_type_intrinsic)
      return NULL;
   nir_intrinsic_instr *load = nir_instr_as_intrinsic(nir_def_instr(pixel));
   if (load->intrinsic != nir_intrinsic_load_deref)
      return NULL;

   *ch_out = ch;
   return load;
}

static bool
match_rgba8_pack_term(nir_def *def, nir_def *sum_defs[4], uint8_t *seen_mask,
                      int *conv_out, unsigned depth)
{
   if (depth > 8)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;

   if (nir_src_is_const(nir_src_for_ssa(def)) &&
       nir_src_as_uint(nir_src_for_ssa(def)) == 0)
      return true;

   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_iadd || alu->op == nir_op_ior) {
      return match_rgba8_pack_term(alu->src[0].src.ssa, sum_defs, seen_mask,
                                   conv_out, depth + 1) &&
             match_rgba8_pack_term(alu->src[1].src.ssa, sum_defs, seen_mask,
                                   conv_out, depth + 1);
   }

   /* Single channel term:
    *   ch 0: iand(f2u32/f2i32(sum[0]), 0xff)
    *   ch 1: ishl(iand(f2u32/f2i32(sum[1]), 0xff), 8)  or  iand(ishl(..., 8), 0xff00)
    *   ch 2: ishl(iand(f2u32/f2i32(sum[2]), 0xff), 16) or  iand(ishl(..., 16), 0xff0000)
    *   ch 3: ishl(iand(f2u32/f2i32(sum[3]), 0xff), 24) or  ishl(f2u32/f2i32(sum[3]), 24)
    */
   unsigned ch = 0;
   nir_def *cvt_def = NULL;

   if (alu->op == nir_op_ishl) {
      if (!nir_src_is_const(alu->src[1].src))
         return false;
      uint32_t sh = nir_src_as_uint(alu->src[1].src);
      if (sh != 8 && sh != 16 && sh != 24)
         return false;
      ch = sh / 8;
      nir_def *lhs = strip_mov(alu->src[0].src.ssa);
      if (!lhs || nir_def_instr(lhs)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *la = nir_instr_as_alu(nir_def_instr(lhs));
      if (la->op == nir_op_extract_u8) {
         if (nir_src_is_const(la->src[1].src) &&
             nir_src_as_uint(la->src[1].src) == 0)
            cvt_def = strip_mov(la->src[0].src.ssa);
      } else if (la->op == nir_op_iand) {
         for (unsigned s = 0; s < 2; s++) {
            if (nir_src_is_const(la->src[s].src) &&
                nir_src_as_uint(la->src[s].src) == 0xffu) {
               cvt_def = strip_mov(la->src[1 - s].src.ssa);
               break;
            }
         }
      } else if (sh == 24) {
         cvt_def = lhs;
      }
   } else if (alu->op == nir_op_extract_u8) {
      if (nir_src_is_const(alu->src[1].src) &&
          nir_src_as_uint(alu->src[1].src) == 0) {
         ch = 0;
         cvt_def = strip_mov(alu->src[0].src.ssa);
      }
   } else if (alu->op == nir_op_iand) {
      nir_def *lhs = NULL;
      uint32_t mask = 0;
      for (unsigned s = 0; s < 2; s++) {
         if (nir_src_is_const(alu->src[s].src)) {
            mask = nir_src_as_uint(alu->src[s].src);
            lhs = strip_mov(alu->src[1 - s].src.ssa);
            break;
         }
      }
      if (!lhs)
         return false;
      if (mask == 0xffu) {
         ch = 0;
         cvt_def = lhs;
      } else if ((mask == 0xff00u || mask == 0xff0000u) &&
                 nir_def_instr(lhs)->type == nir_instr_type_alu &&
                 nir_instr_as_alu(nir_def_instr(lhs))->op == nir_op_ishl) {
         nir_alu_instr *shl = nir_instr_as_alu(nir_def_instr(lhs));
         uint32_t expect_sh = (mask == 0xff00u) ? 8u : 16u;
         if (nir_src_is_const(shl->src[1].src) &&
             nir_src_as_uint(shl->src[1].src) == expect_sh) {
            ch = expect_sh / 8;
            cvt_def = strip_mov(shl->src[0].src.ssa);
         }
      }
   }

   if (!cvt_def || nir_def_instr(cvt_def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cvt = nir_instr_as_alu(nir_def_instr(cvt_def));
   int this_conv = -1;
   if (cvt->op == nir_op_f2u32)
      this_conv = HK_BYTE_STENCIL_CONV_F2U32;
   else if (cvt->op == nir_op_f2i32)
      this_conv = HK_BYTE_STENCIL_CONV_F2I32;
   else
      return false;

   if (strip_mov(cvt->src[0].src.ssa) != sum_defs[ch])
      return false;
   if (*seen_mask & BITFIELD_BIT(ch))
      return false;
   if (*conv_out >= 0 && *conv_out != this_conv)
      return false;

   *seen_mask |= BITFIELD_BIT(ch);
   *conv_out = this_conv;
   return true;
}

static bool
verify_rgba8_shader_completeness(nir_shader *s,
                                 const nir_intrinsic_instr *stencil_src_load,
                                 nir_intrinsic_instr *const *w_loads,
                                 unsigned w_load_count,
                                 const nir_intrinsic_instr *dst_store)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   unsigned seen_src = 0, seen_w = 0, seen_dst = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;

         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref: {
            if (intr == stencil_src_load) {
               seen_src++;
               break;
            }
            bool matched_w = false;
            for (unsigned i = 0; i < w_load_count; i++) {
               if (intr == w_loads[i]) {
                  matched_w = true;
                  seen_w++;
                  break;
               }
            }
            if (!matched_w)
               return false;
            break;
         }
         case nir_intrinsic_store_deref:
            if (intr == dst_store) {
               seen_dst++;
               break;
            }
            return false;
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            return false;
         }
      }
   }

   return seen_src == 1 && seen_w == w_load_count && seen_dst == 1;
}

/* Form B: 1-invocation-per-pixel clamped-window packed-RGBA8 (uint32_t) 1D
 * separable stencil (horizontal or vertical), e.g.:
 *   if (x >= width || y >= height) return;
 *   c_min = max(0, c - radius); c_max = min(dim - 1, c + radius);
 *   for (c_cur = c_min, i = 0; c_cur <= c_max; c_cur++, i++)
 *      rgba += unpack_u8x4(src[off]) * weights[i];
 *   dst[x + y * stride] = pack_u8x4(rgba);
 */
static bool
match_rgba8_clamped_stencil(nir_shader *s, uint32_t set_layout_count,
                            struct vk_descriptor_set_layout *const *set_layouts,
                            struct hk_byte_stencil_plan *plan)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   /* Find the single top-level bounds-guard `if` in impl->body. */
   nir_if *bounds_if = NULL;
   nir_loop *after_if_loop = NULL;
   foreach_list_typed(nir_cf_node, n, node, &impl->body) {
      if (n->type == nir_cf_node_if) {
         if (bounds_if)
            return false;
         bounds_if = nir_cf_node_as_if(n);
      } else if (n->type == nir_cf_node_loop) {
         if (!bounds_if || after_if_loop)
            return false;
         after_if_loop = nir_cf_node_as_loop(n);
      }
   }
   if (!bounds_if)
      return false;

   nir_def *width_pc = NULL, *height_pc = NULL;
   bool in_bounds = false;
   if (!match_bounds_guard_cond(bounds_if->condition.ssa, plan->orig_wg_size,
                                &width_pc, &height_pc, &in_bounds))
      return false;

   nir_loop *stencil_loop = NULL;
   nir_block *return_blk = NULL;
   nir_loop *then_loop = find_single_loop_in_list(&bounds_if->then_list);
   nir_loop *else_loop = find_single_loop_in_list(&bounds_if->else_list);

   if (in_bounds && then_loop && !else_loop && !after_if_loop &&
       cf_list_is_empty_block(&bounds_if->else_list)) {
      stencil_loop = then_loop;
   } else if (!in_bounds && else_loop && !then_loop && !after_if_loop &&
              cf_list_is_empty_block(&bounds_if->then_list)) {
      stencil_loop = else_loop;
   } else if (!in_bounds && !then_loop && !else_loop && after_if_loop &&
              cf_list_is_single_return(&bounds_if->then_list) &&
              cf_list_is_empty_block(&bounds_if->else_list)) {
      stencil_loop = after_if_loop;
      return_blk = cf_list_tail_block(&bounds_if->then_list);
   } else if (in_bounds && !then_loop && !else_loop && after_if_loop &&
              cf_list_is_single_return(&bounds_if->else_list) &&
              cf_list_is_empty_block(&bounds_if->then_list)) {
      stencil_loop = after_if_loop;
      return_blk = cf_list_tail_block(&bounds_if->else_list);
   } else {
      return false;
   }

   /* Inspect stencil loop header phis: expect i_phi, c_phi, optional off_phi,
    * and 4 FP32 channel accumulators. */
   nir_block *sl_header = nir_loop_first_block(stencil_loop);
   nir_phi_instr *i_phi = NULL, *c_phi = NULL, *off_phi = NULL;
   nir_phi_instr *sum_phis[4] = {NULL};
   unsigned sum_phi_count = 0;

   nir_phi_instr *other_int_phis[2] = {NULL};
   unsigned other_int_count = 0;

   nir_foreach_phi(phi, sl_header) {
      if (phi->def.bit_size != 32 || phi->def.num_components != 1)
         return false;
      nir_def *init = NULL, *next = NULL;
      if (!match_loop_phi(stencil_loop, phi, &init, &next))
         return false;

      if (nir_src_is_const(nir_src_for_ssa(init)) &&
          nir_src_as_int(nir_src_for_ssa(init)) == 0 &&
          match_iv_step_const1(next, &phi->def)) {
         if (i_phi)
            return false;
         i_phi = phi;
         continue;
      }

      if (nir_src_is_const(nir_src_for_ssa(init)) &&
          nir_src_as_float(nir_src_for_ssa(init)) == 0.0) {
         nir_def *nm = strip_mov(next);
         if (nm && nir_def_instr(nm)->type == nir_instr_type_alu) {
            nir_op op = nir_instr_as_alu(nir_def_instr(nm))->op;
            if (op == nir_op_fadd || op == nir_op_ffma) {
               if (sum_phi_count >= 4)
                  return false;
               sum_phis[sum_phi_count++] = phi;
               continue;
            }
         }
      }

      if (other_int_count >= 2)
         return false;
      other_int_phis[other_int_count++] = phi;
   }

   if (!i_phi || sum_phi_count != 4 || other_int_count < 1)
      return false;

   /* Identify c_phi via the loop header break `c_phi > c_max`. */
   nir_if *sl_break = NULL;
   nir_def *c_max_def = NULL;
   for (unsigned k = 0; k < other_int_count; k++) {
      nir_if *brk =
         match_clamped_loop_break(stencil_loop, &other_int_phis[k]->def,
                                  &c_max_def);
      if (brk) {
         if (sl_break)
            return false;
         sl_break = brk;
         c_phi = other_int_phis[k];
      } else {
         off_phi = other_int_phis[k];
      }
   }
   if (!sl_break || !c_phi)
      return false;

   /* Verify c_phi init = max(0, coord - radius), step = +1, and c_max =
    * min(dim - 1, coord + radius). */
   nir_def *c_init = NULL, *c_next = NULL;
   if (!match_loop_phi(stencil_loop, c_phi, &c_init, &c_next) ||
       !match_iv_step_const1(c_next, &c_phi->def))
      return false;

   unsigned axis_comp = 0;
   nir_def *diameter_pc = NULL;
   if (!match_clamped_min(c_init, plan->orig_wg_size, &axis_comp, &diameter_pc))
      return false;

   const enum hk_byte_stencil_axis axis =
      (axis_comp == 0) ? HK_BYTE_STENCIL_HORIZ : HK_BYTE_STENCIL_VERT;
   nir_def *axis_dim_pc = (axis_comp == 0) ? width_pc : height_pc;

   if (!match_clamped_max(c_max_def, plan->orig_wg_size, axis_comp, axis_dim_pc,
                          diameter_pc))
      return false;

   /* Verify all 4 channel accumulators read the 4 distinct bytes of the same
    * uint32_t `src` load and multiply by `weights[i_phi]`. */
   nir_intrinsic_instr *stencil_src_load = NULL;
   nir_intrinsic_instr *w_loads[4] = {NULL};
   unsigned w_load_count = 0;
   nir_def *sum_by_ch[4] = {NULL};
   uint32_t w_set = 0, w_bind = 0;

   for (unsigned k = 0; k < 4; k++) {
      nir_def *s_init = NULL, *s_next = NULL;
      if (!match_loop_phi(stencil_loop, sum_phis[k], &s_init, &s_next))
         return false;

      s_next = strip_mov(s_next);
      if (!s_next || nir_def_instr(s_next)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *acc = nir_instr_as_alu(nir_def_instr(s_next));
      if (nir_alu_instr_is_exact(acc))
         return false;

      nir_def *mul_a = NULL, *mul_b = NULL;
      if (acc->op == nir_op_fadd) {
         nir_def *prod = NULL;
         if (strip_mov(acc->src[0].src.ssa) == &sum_phis[k]->def)
            prod = strip_mov(acc->src[1].src.ssa);
         else if (strip_mov(acc->src[1].src.ssa) == &sum_phis[k]->def)
            prod = strip_mov(acc->src[0].src.ssa);
         else
            return false;
         if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(prod))->op != nir_op_fmul ||
             nir_alu_instr_is_exact(nir_instr_as_alu(nir_def_instr(prod))))
            return false;
         mul_a = nir_instr_as_alu(nir_def_instr(prod))->src[0].src.ssa;
         mul_b = nir_instr_as_alu(nir_def_instr(prod))->src[1].src.ssa;
      } else if (acc->op == nir_op_ffma) {
         if (strip_mov(acc->src[2].src.ssa) != &sum_phis[k]->def)
            return false;
         mul_a = acc->src[0].src.ssa;
         mul_b = acc->src[1].src.ssa;
      } else {
         return false;
      }

      unsigned ch = 0;
      nir_intrinsic_instr *px_load = match_u32_byte_to_f32_load(mul_a, &ch);
      nir_def *w_def = strip_mov(mul_b);
      if (!px_load) {
         px_load = match_u32_byte_to_f32_load(mul_b, &ch);
         w_def = strip_mov(mul_a);
      }
      if (!px_load || !w_def || ch >= 4 || sum_by_ch[ch] != NULL ||
          nir_def_instr(w_def)->type != nir_instr_type_intrinsic)
         return false;

      if (stencil_src_load && stencil_src_load != px_load)
         return false;
      stencil_src_load = px_load;
      sum_by_ch[ch] = &sum_phis[k]->def;

      nir_intrinsic_instr *wl = nir_instr_as_intrinsic(nir_def_instr(w_def));
      if (wl->intrinsic != nir_intrinsic_load_deref ||
          wl->def.bit_size != 32 || wl->def.num_components != 1)
         return false;
      uint32_t ws = 0, wb = 0;
      nir_def *wi = NULL;
      if (!match_ssbo_array_deref(nir_src_as_deref(wl->src[0]), 4, &ws, &wb,
                                  &wi) ||
          strip_mov(wi) != &i_phi->def)
         return false;
      if (w_load_count == 0) {
         w_set = ws;
         w_bind = wb;
      } else if (ws != w_set || wb != w_bind) {
         return false;
      }
      bool dup = false;
      for (unsigned j = 0; j < w_load_count; j++) {
         if (w_loads[j] == wl)
            dup = true;
      }
      if (!dup)
         w_loads[w_load_count++] = wl;
   }

   if (!stencil_src_load || !sum_by_ch[0] || !sum_by_ch[1] || !sum_by_ch[2] ||
       !sum_by_ch[3])
      return false;

   /* Verify src load deref (stride 4) and extract stride_pc. */
   uint32_t src_set = 0, src_bind = 0;
   nir_def *src_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(stencil_src_load->src[0]), 4,
                               &src_set, &src_bind, &src_idx))
      return false;

   /* Inspect the single dst store right after stencil_loop to extract
    * stride_pc and verify the packed RGBA8 output. */
   nir_cf_node *after_loop_cf = nir_cf_node_next(&stencil_loop->cf_node);
   if (!after_loop_cf || after_loop_cf->type != nir_cf_node_block)
      return false;
   nir_block *after_blk = nir_cf_node_as_block(after_loop_cf);
   nir_intrinsic_instr *dst_store = NULL;
   nir_foreach_instr(instr, after_blk) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (dst_store)
               return false;
            dst_store = intr;
         }
      }
   }
   if (!dst_store)
      return false;

   uint32_t dst_set = 0, dst_bind = 0;
   nir_def *dst_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(dst_store->src[0]), 4, &dst_set,
                               &dst_bind, &dst_idx))
      return false;
   if ((src_set == dst_set && src_bind == dst_bind) ||
       (w_set == dst_set && w_bind == dst_bind))
      return false;

   /* Extract stride_pc from `dst_idx = x + y * stride_pc` (accepting fresh
    * global_id copies if not CSE'd). */
   nir_def *stride_pc = NULL;
   nir_def *di = strip_mov(dst_idx);
   if (!di || nir_def_instr(di)->type != nir_instr_type_alu ||
       nir_instr_as_alu(nir_def_instr(di))->op != nir_op_iadd)
      return false;
   {
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(di));
      for (unsigned side = 0; side < 2 && !stride_pc; side++) {
         if (!same_global_coord_src(&add->src[side], plan->orig_wg_size, 0))
            continue;
         nir_def *other = strip_mov(add->src[1 - side].src.ssa);
         if (!other || nir_def_instr(other)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(other))->op != nir_op_imul)
            continue;
         nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(other));
         for (unsigned ms = 0; ms < 2; ms++) {
            uint16_t off = 0;
            if (same_global_coord_src(&mul->src[ms], plan->orig_wg_size, 1) &&
                match_push_const_u32(mul->src[1 - ms].src.ssa, &off)) {
               stride_pc = mul->src[1 - ms].src.ssa;
               break;
            }
         }
      }
   }
   if (!stride_pc)
      return false;

   /* Verify src_idx matches `c_min + y * stride` (+1/iter) for H or
    * `x + c_min * stride` (+stride/iter) for V. */
   src_idx = strip_mov(src_idx);
   nir_def *check_expr = NULL;
   nir_def *coord_var = NULL;
   if (off_phi && src_idx == &off_phi->def) {
      nir_def *off_init = NULL, *off_next = NULL;
      if (!match_loop_phi(stencil_loop, off_phi, &off_init, &off_next))
         return false;
      if (axis == HK_BYTE_STENCIL_HORIZ) {
         if (!match_iv_step_const1(off_next, &off_phi->def))
            return false;
      } else {
         if (!match_iv_step_pc(off_next, &off_phi->def, stride_pc))
            return false;
      }
      check_expr = strip_mov(off_init);
      coord_var = strip_mov(c_init);
   } else if (!off_phi) {
      check_expr = src_idx;
      coord_var = &c_phi->def;
   } else {
      return false;
   }

   if (!check_expr || nir_def_instr(check_expr)->type != nir_instr_type_alu ||
       nir_instr_as_alu(nir_def_instr(check_expr))->op != nir_op_iadd)
      return false;
   {
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(check_expr));
      bool idx_ok = false;
      for (unsigned side = 0; side < 2 && !idx_ok; side++) {
         nir_def *a = strip_mov(add->src[side].src.ssa);
         nir_def *b = strip_mov(add->src[1 - side].src.ssa);
         if (!b || nir_def_instr(b)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(b))->op != nir_op_imul)
            continue;
         nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(b));
         for (unsigned ms = 0; ms < 2 && !idx_ok; ms++) {
            if (!same_push_const(mul->src[1 - ms].src.ssa, stride_pc))
               continue;
            nir_def *row = strip_mov(mul->src[ms].src.ssa);
            if (axis == HK_BYTE_STENCIL_HORIZ) {
               if (a == coord_var &&
                   same_global_coord_src(&mul->src[ms], plan->orig_wg_size, 1))
                  idx_ok = true;
            } else {
               if (same_global_coord_src(&add->src[side], plan->orig_wg_size,
                                         0) &&
                   row == coord_var)
                  idx_ok = true;
            }
         }
      }
      if (!idx_ok)
         return false;
   }

   /* Verify packed RGBA8 output store value. */
   uint8_t seen_mask = 0;
   int store_conv = -1;
   if (!match_rgba8_pack_term(dst_store->src[1].ssa, sum_by_ch, &seen_mask,
                              &store_conv, 0) ||
       seen_mask != 0xfu || store_conv < 0)
      return false;

   /* Verify strict control flow & shader completeness. */
   const struct byte_stencil_cf cf = {
      .loops = {stencil_loop},
      .break_ifs = {sl_break},
      .border_if = bounds_if,
      .return_blk = return_blk,
   };
   if (!verify_control_flow(impl, &cf))
      return false;

   if (!verify_rgba8_shader_completeness(s, stencil_src_load, w_loads,
                                         w_load_count, dst_store))
      return false;

   uint16_t width_off = 0, height_off = 0, stride_off = 0, diam_off = 0;
   if (!match_push_const_u32(width_pc, &width_off) ||
       !match_push_const_u32(height_pc, &height_off) ||
       !match_push_const_u32(stride_pc, &stride_off) ||
       !match_push_const_u32(diameter_pc, &diam_off))
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->src) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->dst) ||
       !populate_binding_info(set_layout_count, set_layouts, w_set, w_bind,
                              &plan->weights))
      return false;

   plan->axis = (uint8_t)axis;
   plan->store_conv = (uint8_t)store_conv;
   plan->is_grid_stride = 0;
   plan->elem_kind = HK_BYTE_STENCIL_ELEM_RGBA8_CLAMPED;
   plan->width_pc_offset_B = width_off;
   plan->height_pc_offset_B = height_off;
   plan->stride_pc_offset_B = stride_off;
   plan->diameter_pc_offset_B = diam_off;
   if (axis == HK_BYTE_STENCIL_HORIZ) {
      plan->tile_w = 128;
      plan->tile_h = 4;
   } else {
      plan->tile_w = 16;
      plan->tile_h = 64;
   }

   return true;
}

/* Expand AGX-specific fused integer ops (e.g. imadshl_agx produced by
 * agx_preprocess_nir before hk_maybe_build_byte_stencil) back to standard NIR
 * ALU ops on the scratch clone so structural matching stays target-agnostic. */
static bool
expand_agx_ops_instr(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *rep = NULL;

   switch (alu->op) {
   case nir_op_imadshl_agx: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *bb = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 2);
      nir_def *d = nir_ssa_for_alu_src(b, alu, 3);
      rep = nir_iadd(b, nir_imul(b, a, bb), nir_ishl(b, c, d));
      break;
   }
   case nir_op_imsubshl_agx: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *bb = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 2);
      nir_def *d = nir_ssa_for_alu_src(b, alu, 3);
      rep = nir_isub(b, nir_imul(b, a, bb), nir_ishl(b, c, d));
      break;
   }
   case nir_op_amul: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *bb = nir_ssa_for_alu_src(b, alu, 1);
      rep = nir_imul(b, a, bb);
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&alu->def, rep);
   return true;
}

/* Hoist shader-invariant constant-offset push-constant loads and invocation
 * sysvals to the entry block of the scratch clone so nir_opt_peephole_select
 * can flatten glslang's short-circuit `||` and ternary min/max branches. */
static bool
hoist_matcher_invariants(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_block *start = nir_start_block(impl);
   nir_builder b = nir_builder_at(nir_before_block(start));
   bool moved = false;

   nir_foreach_block(block, impl) {
      if (block == start)
         continue;
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_push_constant:
            if (!nir_src_is_const(intr->src[0]))
               continue;
            b.cursor = nir_before_block(start);
            nir_def *off =
               nir_imm_intN_t(&b, nir_src_as_uint(intr->src[0]),
                              intr->src[0].ssa->bit_size);
            nir_src_rewrite(&intr->src[0], off);
            nir_instr_move(nir_after_def(off), instr);
            moved = true;
            break;
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_num_workgroups:
            nir_instr_move(nir_before_block(start), instr);
            moved = true;
            break;
         default:
            break;
         }
      }
   }
   return moved;
}

static bool
cf_list_has_any_loop(const struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if) {
         const nir_if *nif = nir_cf_node_as_if(node);
         if (cf_list_has_any_loop(&nif->then_list) ||
             cf_list_has_any_loop(&nif->else_list))
            return true;
      }
   }
   return false;
}

bool
hk_nir_match_byte_stencil(const nir_shader *orig_nir,
                          uint32_t set_layout_count,
                          struct vk_descriptor_set_layout *const *set_layouts,
                          struct hk_byte_stencil_plan *plan_out)
{
   if (!orig_nir || orig_nir->info.stage != MESA_SHADER_COMPUTE ||
       orig_nir->info.internal || orig_nir->info.shared_size != 0 ||
       orig_nir->info.workgroup_size_variable ||
       orig_nir->info.workgroup_size[0] == 0 ||
       orig_nir->info.workgroup_size[1] == 0 ||
       orig_nir->info.workgroup_size[2] == 0 ||
       orig_nir->info.ray_queries > 0 ||
       orig_nir->info.num_images > 0 || orig_nir->info.num_textures > 0 ||
       orig_nir->info.num_ubos > 0 || orig_nir->info.num_ssbos == 1)
      return false;

   nir_foreach_variable_in_shader(var, orig_nir) {
      if (var->data.mode & (nir_var_mem_shared | nir_var_image |
                            nir_var_uniform))
         return false;
   }

   if (!exec_list_is_singular(&orig_nir->functions))
      return false;

   const nir_function_impl *orig_impl = nir_shader_get_entrypoint(orig_nir);
   if (!orig_impl || !cf_list_has_any_loop(&orig_impl->body))
      return false;

   /* Canonicalize a scratch clone so we never mutate the caller's NIR. */
   nir_shader *s = nir_shader_clone(NULL, orig_nir);
   if (!s)
      return false;

   nir_lower_continue_constructs(s);
   nir_lower_vars_to_ssa(s);
   nir_lower_system_values(s);
   nir_lower_explicit_io(s, nir_var_mem_push_const,
                         nir_address_format_32bit_offset);
   nir_lower_alu_to_scalar(s, NULL, NULL);
   nir_shader_alu_pass(s, expand_agx_ops_instr, nir_metadata_control_flow, NULL);

   bool progress;
   unsigned iter = 0;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      progress |= hoist_matcher_invariants(s);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_undef);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      const nir_opt_peephole_select_options peep_opts = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, s, nir_opt_peephole_select, &peep_opts);
      NIR_PASS(progress, s, nir_opt_if, nir_opt_if_optimize_phi_true_false);
   } while (progress && ++iter < 8);

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_metadata_require(impl, nir_metadata_block_index);

   struct hk_byte_stencil_plan plan = {0};
   plan.orig_wg_size[0] = orig_nir->info.workgroup_size[0];
   plan.orig_wg_size[1] = orig_nir->info.workgroup_size[1];
   plan.orig_wg_size[2] = orig_nir->info.workgroup_size[2];

   bool matched = false;

   /* Form A: 2D nested grid-stride loops (outer y/x, inner x/y). */
   nir_loop *outer_loop = find_single_loop_in_list(&impl->body);
   if (outer_loop) {
      unsigned outer_comp = 0;
      nir_def *outer_iv = NULL, *outer_dim = NULL;
      nir_if *outer_break = NULL;
      if (match_grid_stride_loop(outer_loop, plan.orig_wg_size, &outer_comp,
                                 &outer_iv, &outer_dim, &outer_break)) {
         nir_loop *inner_loop = find_single_loop_in_list(&outer_loop->body);
         if (inner_loop) {
            unsigned inner_comp = 0;
            nir_def *inner_iv = NULL, *inner_dim = NULL;
            nir_if *inner_break = NULL;
            if (match_grid_stride_loop(inner_loop, plan.orig_wg_size,
                                       &inner_comp, &inner_iv, &inner_dim,
                                       &inner_break) &&
                outer_comp != inner_comp) {
               nir_def *x = (outer_comp == 0) ? outer_iv : inner_iv;
               nir_def *y = (outer_comp == 1) ? outer_iv : inner_iv;
               nir_def *width_pc = (outer_comp == 0) ? outer_dim : inner_dim;
               nir_def *height_pc = (outer_comp == 1) ? outer_dim : inner_dim;
               const struct byte_stencil_cf grid_cf = {
                  .loops = {outer_loop, inner_loop},
                  .break_ifs = {outer_break, inner_break},
               };
               plan.is_grid_stride = 1;
               matched = match_pixel_body(s, &inner_loop->body,
                                          &inner_break->cf_node, x, y,
                                          width_pc, height_pc, &grid_cf,
                                          set_layout_count, set_layouts, &plan);
            }
         }
      }
   }

   /* Form B: 1-invocation-per-pixel clamped packed-RGBA8 separable stencil. */
   if (!matched) {
      matched = match_rgba8_clamped_stencil(s, set_layout_count, set_layouts,
                                            &plan);
   }

   ralloc_free(s);

   if (!matched)
      return false;

   *plan_out = plan;
   return true;
}

static nir_def *
load_pc_i32(nir_builder *b, uint16_t offset_B)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = offset_B,
                                 .range = 4);
}

static nir_deref_instr *
build_ssbo_base_deref_aligned(nir_builder *b,
                              const struct hk_byte_stencil_binding *bind,
                              nir_address_format ssbo_addr_format,
                              const struct glsl_type *elem_type,
                              unsigned elem_stride, unsigned align_mul)
{
   unsigned comps = nir_address_format_num_components(ssbo_addr_format);
   unsigned bits = nir_address_format_bit_size(ssbo_addr_format);
   nir_def *ridx = nir_vulkan_resource_index(
      b, comps, bits, nir_imm_int(b, 0), .desc_set = bind->set,
      .binding = bind->binding, .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, comps, bits, ridx, .desc_type = nir_descriptor_type_storage_buffer);
   return nir_build_deref_cast_with_alignment(
      b, desc, nir_var_mem_ssbo, elem_type, elem_stride, align_mul, 0);
}

static nir_deref_instr *
build_ssbo_base_deref(nir_builder *b, const struct hk_byte_stencil_binding *bind,
                      nir_address_format ssbo_addr_format,
                      const struct glsl_type *elem_type, unsigned elem_stride)
{
   return build_ssbo_base_deref_aligned(b, bind, ssbo_addr_format, elem_type,
                                        elem_stride, 0);
}

static nir_def *
load_ssbo_u32(nir_builder *b, nir_deref_instr *base, nir_def *idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_ssbo_u32x4(nir_builder *b, nir_deref_instr *base_u32x4, nir_def *idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_u32x4, idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static void
store_ssbo_u32(nir_builder *b, nir_deref_instr *base, nir_def *idx,
               nir_def *val)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   nir_store_deref_with_access(b, elem, val, 0x1, ACCESS_NON_READABLE);
}

static void
store_ssbo_u32x4(nir_builder *b, nir_deref_instr *base_u32x4, nir_def *idx,
                 nir_def *val_u32x4)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_u32x4, idx);
   nir_store_deref_with_access(b, elem, val_u32x4, 0xf, ACCESS_NON_READABLE);
}

static nir_def *
load_ssbo_u8(nir_builder *b, nir_deref_instr *base, nir_def *idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_ssbo_u8x4(nir_builder *b, nir_deref_instr *base_u8x4, nir_def *byte_idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_u8x4, byte_idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static void
store_ssbo_u8(nir_builder *b, nir_deref_instr *base, nir_def *idx, nir_def *val)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   nir_store_deref_with_access(b, elem, val, 0x1, ACCESS_NON_READABLE);
}

static void
store_ssbo_u8x4(nir_builder *b, nir_deref_instr *base_u8x4, nir_def *byte_idx,
                nir_def *val_u8x4)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_u8x4, byte_idx);
   nir_store_deref_with_access(b, elem, val_u8x4, 0xf, ACCESS_NON_READABLE);
}

static nir_def *
load_ssbo_f32(nir_builder *b, nir_deref_instr *base, nir_def *idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_ssbo_f32x4(nir_builder *b, nir_deref_instr *base_v4, nir_def *idx4)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_v4, idx4);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
convert_sum_to_u8(nir_builder *b, nir_def *sum,
                  enum hk_byte_stencil_store_conv conv)
{
   nir_def *i = (conv == HK_BYTE_STENCIL_CONV_F2U32) ? nir_f2u32(b, sum)
                                                     : nir_f2i32(b, sum);
   return nir_u2u8(b, i);
}

static void
emit_shared_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

static nir_def *
load_sh_vec4(nir_builder *b, nir_deref_instr *arr_v4, nir_def *idx4)
{
   return nir_load_deref(b, nir_build_deref_array(b, arr_v4, idx4));
}

static void
store_sh_vec4(nir_builder *b, nir_deref_instr *arr_v4, nir_def *idx4,
              nir_def *val_v4)
{
   nir_store_deref(b, nir_build_deref_array(b, arr_v4, idx4), val_v4, 0xf);
}

static nir_def *
load_sh_vec4_scalar(nir_builder *b, nir_deref_instr *arr_v4, nir_def *idx)
{
   nir_deref_instr *vec =
      nir_build_deref_array(b, arr_v4, nir_ishr_imm(b, idx, 2));
   return nir_load_deref(b, nir_build_deref_array(b, vec, nir_iand_imm(b, idx, 3)));
}

static void
store_sh_vec4_scalar(nir_builder *b, nir_deref_instr *arr_v4, nir_def *idx,
                     nir_def *val)
{
   nir_deref_instr *vec =
      nir_build_deref_array(b, arr_v4, nir_ishr_imm(b, idx, 2));
   nir_store_deref(b, nir_build_deref_array(b, vec, nir_iand_imm(b, idx, 3)),
                   val, 0x1);
}

/* Emit a single-pixel scalar stencil from a vec4-packed shared tile + uniform
 * SSBO weights for rare border/edge pixels. */
static nir_def *
emit_edge_1px_stencil(nir_builder *b, nir_deref_instr *sh_tile_v4,
                      bool sh_is_f16, nir_deref_instr *w_base,
                      nir_def *base_idx, nir_def *diameter)
{
   nir_variable *i_var =
      nir_local_variable_create(b->impl, glsl_int_type(), "ei");
   nir_variable *sum_var =
      nir_local_variable_create(b->impl, glsl_float_type(), "esum");
   nir_store_var(b, i_var, nir_imm_int(b, 0), 0x1);
   nir_store_var(b, sum_var, nir_imm_float(b, 0.0f), 0x1);

   nir_loop *el = nir_push_loop(b);
   el->control = nir_loop_control_dont_unroll;
   {
      nir_def *i = nir_load_var(b, i_var);
      nir_push_if(b, nir_ige(b, i, diameter));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *sh_idx = nir_iadd(b, base_idx, i);
      nir_def *p = load_sh_vec4_scalar(b, sh_tile_v4, sh_idx);
      if (sh_is_f16)
         p = nir_f2f32(b, p);
      nir_def *w = load_ssbo_f32(b, w_base, i);
      nir_def *s = nir_ffma(b, p, w, nir_load_var(b, sum_var));
      nir_store_var(b, sum_var, s, 0x1);
      nir_store_var(b, i_var, nir_iadd_imm(b, i, 1), 0x1);
   }
   nir_pop_loop(b, NULL);

   return nir_load_var(b, sum_var);
}

/* Emit the vectorized 4- or 8-pixel-per-thread sliding-window stencil core
 * shared by both horizontal (4 px, FP32 vec4 shared) and vertical (8 px, FP16
 * vec4 shared) kernels. Each 4-tap iteration performs 1 uniform f32x4 weight
 * load + 1 aligned vec4 shared load for 4 * num_px FFMAs (16 horiz, 32 vert),
 * followed by a loop-free 0..3 tap tail. */
static void
emit_vec4_sliding_stencil(nir_builder *b, nir_deref_instr *sh_tile_v4,
                          bool sh_is_f16, nir_deref_instr *w_base,
                          nir_deref_instr *w_base_v4, nir_def *base_v4,
                          nir_def *diameter, unsigned num_px,
                          nir_def **out_sums)
{
   assert(num_px == 4 || num_px == 8);
   const unsigned n_vecs = num_px / 4;

   nir_def *n4 = nir_ishr_imm(b, diameter, 2);
   nir_def *rem = nir_iand_imm(b, diameter, 3);
   nir_def *i_tail = nir_iand_imm(b, diameter, ~3);

   nir_variable *i4_var =
      nir_local_variable_create(b->impl, glsl_int_type(), "si4");
   nir_variable *s_var[8];
   nir_variable *r_var[8];
   for (unsigned p = 0; p < num_px; p++) {
      s_var[p] = nir_local_variable_create(b->impl, glsl_float_type(), "ss");
      r_var[p] = nir_local_variable_create(b->impl, glsl_float_type(), "sr");
      nir_store_var(b, s_var[p], nir_imm_float(b, 0.0f), 0x1);
   }

   for (unsigned v = 0; v < n_vecs; v++) {
      nir_def *v_idx = v ? nir_iadd_imm(b, base_v4, v) : base_v4;
      nir_def *init_v = load_sh_vec4(b, sh_tile_v4, v_idx);
      if (sh_is_f16)
         init_v = nir_f2f32(b, init_v);
      for (unsigned p = 0; p < 4; p++)
         nir_store_var(b, r_var[v * 4 + p], nir_channel(b, init_v, p), 0x1);
   }
   nir_store_var(b, i4_var, nir_imm_int(b, 0), 0x1);

   /* 4-tap vectorized main loop: 1 uniform f32x4 + 1 aligned vec4 shared load
    * -> 4 * num_px FFMAs. Leave unrollable so runtime loop unrolling hides load
    * latency. */
   nir_push_loop(b);
   {
      nir_def *i4 = nir_load_var(b, i4_var);
      nir_push_if(b, nir_ige(b, i4, n4));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *wv = load_ssbo_f32x4(b, w_base_v4, i4);
      nir_def *nv = load_sh_vec4(
         b, sh_tile_v4, nir_iadd(b, base_v4, nir_iadd_imm(b, i4, n_vecs)));
      if (sh_is_f16)
         nv = nir_f2f32(b, nv);

      nir_def *rv[11];
      for (unsigned p = 0; p < num_px; p++)
         rv[p] = nir_load_var(b, r_var[p]);
      rv[num_px + 0] = nir_channel(b, nv, 0);
      rv[num_px + 1] = nir_channel(b, nv, 1);
      rv[num_px + 2] = nir_channel(b, nv, 2);

      nir_def *s[8];
      for (unsigned p = 0; p < num_px; p++)
         s[p] = nir_load_var(b, s_var[p]);

      for (unsigned t = 0; t < 4; t++) {
         nir_def *wt = nir_channel(b, wv, t);
         for (unsigned p = 0; p < num_px; p++)
            s[p] = nir_ffma(b, rv[t + p], wt, s[p]);
      }

      for (unsigned p = 0; p < num_px; p++)
         nir_store_var(b, s_var[p], s[p], 0x1);
      for (unsigned p = 0; p + 4 < num_px; p++)
         nir_store_var(b, r_var[p], rv[p + 4], 0x1);
      for (unsigned p = 0; p < 4; p++)
         nir_store_var(b, r_var[num_px - 4 + p], nir_channel(b, nv, p), 0x1);
      nir_store_var(b, i4_var, nir_iadd_imm(b, i4, 1), 0x1);
   }
   nir_pop_loop(b, NULL);

   /* Uniform 0..3 tap tail (no loop; tap 0 reuses r[0..num_px-1] in regs). */
   nir_push_if(b, nir_ine_imm(b, rem, 0));
   {
      nir_def *w0 = load_ssbo_f32(b, w_base, i_tail);
      for (unsigned p = 0; p < num_px; p++) {
         nir_def *s = nir_ffma(b, nir_load_var(b, r_var[p]), w0,
                               nir_load_var(b, s_var[p]));
         nir_store_var(b, s_var[p], s, 0x1);
      }

      nir_push_if(b, nir_ige_imm(b, rem, 2));
      {
         nir_def *nv = load_sh_vec4(
            b, sh_tile_v4, nir_iadd(b, base_v4, nir_iadd_imm(b, n4, n_vecs)));
         if (sh_is_f16)
            nv = nir_f2f32(b, nv);

         nir_def *w1 = load_ssbo_f32(b, w_base, nir_iadd_imm(b, i_tail, 1));
         for (unsigned p = 0; p < num_px; p++) {
            nir_def *tp = (p + 1 < num_px) ? nir_load_var(b, r_var[p + 1])
                                           : nir_channel(b, nv, 0);
            nir_def *s = nir_ffma(b, tp, w1, nir_load_var(b, s_var[p]));
            nir_store_var(b, s_var[p], s, 0x1);
         }

         nir_push_if(b, nir_ige_imm(b, rem, 3));
         {
            nir_def *w2 = load_ssbo_f32(b, w_base, nir_iadd_imm(b, i_tail, 2));
            for (unsigned p = 0; p < num_px; p++) {
               nir_def *tp = (p + 2 < num_px)
                                ? nir_load_var(b, r_var[p + 2])
                                : nir_channel(b, nv, p + 2 - num_px);
               nir_def *s = nir_ffma(b, tp, w2, nir_load_var(b, s_var[p]));
               nir_store_var(b, s_var[p], s, 0x1);
            }
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   for (unsigned p = 0; p < num_px; p++)
      out_sums[p] = nir_load_var(b, s_var[p]);
}

/* Build horizontal tiled kernel: 32x4 threads (128 threads/WG), 128x4 pixel
 * tile (4 adjacent horizontal pixels per thread), FP32 vec4 shared staging
 * (3 KiB -> full occupancy, zero f16->f32 conversions), uniform f32x4 weights,
 * and 128-byte coalesced u8x4 interior loads & stores. */
static void
build_horiz_tiled_kernel(nir_builder *b, const struct hk_byte_stencil_plan *plan,
                         nir_address_format ssbo_addr_format)
{
   const unsigned WG_X = 32;
   const unsigned WG_Y = 4;
   const unsigned TILE_W = 128;
   const unsigned SH_ROW_V4 = 48; /* 48 * 4 = 192 >= 128 + 2 * 28 = 184 */

   b->shader->info.workgroup_size[0] = WG_X;
   b->shader->info.workgroup_size[1] = WG_Y;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_tile_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vec4_type(), WG_Y * SH_ROW_V4, 0), "sh_tile");
   nir_deref_instr *sh_tile_v4 = nir_build_deref_var(b, sh_tile_var);

   nir_deref_instr *src_base = build_ssbo_base_deref(
      b, &plan->src, ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *src_base_u8x4 = build_ssbo_base_deref(
      b, &plan->src, ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *dst_base = build_ssbo_base_deref(
      b, &plan->dst, ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst_base_u8x4 = build_ssbo_base_deref(
      b, &plan->dst, ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *w_base = build_ssbo_base_deref(
      b, &plan->weights, ssbo_addr_format, glsl_float_type(), 4);
   nir_deref_instr *w_base_v4 = build_ssbo_base_deref(
      b, &plan->weights, ssbo_addr_format, glsl_vec4_type(), 16);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *diameter = load_pc_i32(b, plan->diameter_pc_offset_B);
   nir_def *radius = nir_ishr_imm(b, diameter, 1);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tx = nir_channel(b, lid, 0);
   nir_def *ly = nir_channel(b, lid, 1);

   nir_def *tile_x = nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W);
   nir_def *gy =
      nir_iadd(b, nir_imul_imm(b, nir_channel(b, wgid, 1), WG_Y), ly);
   nir_def *valid_y = nir_ilt(b, gy, height);
   nir_def *row_base = nir_imul(b, gy, stride);
   nir_def *sh_row_v4 = nir_imul_imm(b, ly, SH_ROW_V4);
   nir_def *sh_row_off = nir_ishl_imm(b, sh_row_v4, 2);

   nir_def *total_k = nir_iadd_imm(b, nir_ishl_imm(b, radius, 1), TILE_W);
   nir_def *x_start = nir_isub(b, tile_x, radius);
   nir_def *tile_in_x = nir_iand(b, nir_ige_imm(b, x_start, 0),
                                 nir_ige(b, width, nir_iadd(b, x_start, total_k)));

   /* Cooperative load of 128 + 2 * radius bytes per row into FP32 vec4 shared. */
   nir_push_if(b, valid_y);
   {
      nir_push_if(b, tile_in_x);
      {
         /* Fast branch-light interior tile: 128-byte coalesced u8x4 loads. */
         nir_def *k0 = nir_ishl_imm(b, tx, 2);
         nir_def *b0 =
            load_ssbo_u8x4(b, src_base_u8x4,
                           nir_iadd(b, row_base, nir_iadd(b, x_start, k0)));
         store_sh_vec4(b, sh_tile_v4, nir_iadd(b, sh_row_v4, tx),
                       nir_u2f32(b, b0));

         nir_def *k1 = nir_iadd_imm(b, k0, TILE_W);
         nir_push_if(b, nir_ige(b, total_k, nir_iadd_imm(b, k1, 4)));
         {
            nir_def *b1 =
               load_ssbo_u8x4(b, src_base_u8x4,
                              nir_iadd(b, row_base, nir_iadd(b, x_start, k1)));
            store_sh_vec4(b, sh_tile_v4,
                          nir_iadd(b, sh_row_v4, nir_iadd_imm(b, tx, WG_X)),
                          nir_u2f32(b, b1));
         }
         nir_push_else(b, NULL);
         {
            nir_push_if(b, nir_ilt(b, k1, total_k));
            {
               /* total_k = 128 + 2*radius is even, so at most 2 bytes remain. */
               nir_def *base_g = nir_iadd(b, row_base, nir_iadd(b, x_start, k1));
               nir_def *s0 = load_ssbo_u8(b, src_base, base_g);
               store_sh_vec4_scalar(b, sh_tile_v4, nir_iadd(b, sh_row_off, k1),
                                    nir_u2f32(b, s0));
               nir_push_if(b, nir_ilt(b, nir_iadd_imm(b, k1, 1), total_k));
               {
                  nir_def *s1 =
                     load_ssbo_u8(b, src_base, nir_iadd_imm(b, base_g, 1));
                  store_sh_vec4_scalar(
                     b, sh_tile_v4,
                     nir_iadd(b, sh_row_off, nir_iadd_imm(b, k1, 1)),
                     nir_u2f32(b, s1));
               }
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         /* Cold border tile: scalar bounded loads across 6 coalesced steps. */
         nir_variable *k_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "hk");
         nir_store_var(b, k_var, tx, 0x1);
         nir_loop *bl = nir_push_loop(b);
         bl->control = nir_loop_control_dont_unroll;
         {
            nir_def *k = nir_load_var(b, k_var);
            nir_push_if(b, nir_ige(b, k, total_k));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *gx = nir_iadd(b, x_start, k);
            nir_push_if(b, nir_iand(b, nir_ige_imm(b, gx, 0),
                                    nir_ilt(b, gx, width)));
            {
               nir_def *byte =
                  load_ssbo_u8(b, src_base, nir_iadd(b, row_base, gx));
               store_sh_vec4_scalar(b, sh_tile_v4, nir_iadd(b, sh_row_off, k),
                                    nir_u2f32(b, byte));
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, k_var, nir_iadd_imm(b, k, WG_X), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   nir_push_if(b, valid_y);
   {
      nir_def *lx0 = nir_ishl_imm(b, tx, 2);
      nir_def *x0 = nir_iadd(b, tile_x, lx0);
      nir_def *max_int_x = nir_isub(b, width, radius);
      nir_def *all_interior = nir_iand(
         b, nir_ige(b, x0, radius),
         nir_ilt(b, nir_iadd_imm(b, x0, 3), max_int_x));

      nir_push_if(b, all_interior);
      {
         nir_def *sums[4];
         emit_vec4_sliding_stencil(b, sh_tile_v4, false, w_base, w_base_v4,
                                   nir_iadd(b, sh_row_v4, tx), diameter, 4,
                                   sums);

         nir_def *outs[4];
         for (unsigned p = 0; p < 4; p++)
            outs[p] = convert_sum_to_u8(b, sums[p], plan->store_conv);
         store_ssbo_u8x4(b, dst_base_u8x4, nir_iadd(b, row_base, x0),
                         nir_vec(b, outs, 4));
      }
      nir_push_else(b, NULL);
      {
         /* Compact runtime loop for cold border/edge pixels (4 pixels/thread). */
         nir_variable *p_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "hp");
         nir_store_var(b, p_var, nir_imm_int(b, 0), 0x1);
         nir_loop *pl = nir_push_loop(b);
         pl->control = nir_loop_control_dont_unroll;
         {
            nir_def *p = nir_load_var(b, p_var);
            nir_push_if(b, nir_ige_imm(b, p, 4));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *xp = nir_iadd(b, x0, p);
            nir_def *lxp = nir_iadd(b, lx0, p);
            nir_push_if(b, nir_ilt(b, xp, width));
            {
               nir_def *is_border =
                  nir_ior(b, nir_ilt(b, xp, radius), nir_ige(b, xp, max_int_x));
               nir_push_if(b, is_border);
               {
                  nir_def *center_k =
                     nir_iadd(b, sh_row_off, nir_iadd(b, lxp, radius));
                  nir_def *orig_b = nir_u2u8(
                     b, nir_f2u32(b, load_sh_vec4_scalar(b, sh_tile_v4,
                                                         center_k)));
                  store_ssbo_u8(b, dst_base, nir_iadd(b, row_base, xp), orig_b);
               }
               nir_push_else(b, NULL);
               {
                  nir_def *sum = emit_edge_1px_stencil(
                     b, sh_tile_v4, false, w_base, nir_iadd(b, sh_row_off, lxp),
                     diameter);
                  store_ssbo_u8(b, dst_base, nir_iadd(b, row_base, xp),
                                convert_sum_to_u8(b, sum, plan->store_conv));
               }
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, p_var, nir_iadd_imm(b, p, 1), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Build vertical tiled kernel: 32x8 threads (256 threads/WG), 32x64 pixel tile
 * (8 vertical pixels per thread), 32-byte coalesced X loads & stores, column-
 * major FP16 vec4 shared staging (32 x 31 f16vec4 = 7.75 KiB -> 4 WGs = 1024
 * threads/core full occupancy, odd 8-byte stride = zero bank conflicts),
 * uniform f32x4 weights, and 8-pixel FP32 sliding register window (1 weight +
 * 1 shared load per 32 FFMAs). */
static void
build_vert_tiled_kernel(nir_builder *b, const struct hk_byte_stencil_plan *plan,
                        nir_address_format ssbo_addr_format)
{
   const unsigned WG_X = 32;
   const unsigned WG_Y = 8;
   const unsigned TILE_W = 32;
   const unsigned TILE_H = 64;
   const unsigned SH_COL_V4 = 31; /* 31 * 4 = 124 >= 64 + 2 * 28 = 120; odd -> 0 bank conflicts */
   const unsigned SH_COL_STRIDE = SH_COL_V4 * 4;

   b->shader->info.workgroup_size[0] = WG_X;
   b->shader->info.workgroup_size[1] = WG_Y;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_tile_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_FLOAT16, 4),
                      TILE_W * SH_COL_V4, 0),
      "sh_tile");
   nir_deref_instr *sh_tile_v4 = nir_build_deref_var(b, sh_tile_var);

   nir_deref_instr *src_base = build_ssbo_base_deref(
      b, &plan->src, ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst_base = build_ssbo_base_deref(
      b, &plan->dst, ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *w_base = build_ssbo_base_deref(
      b, &plan->weights, ssbo_addr_format, glsl_float_type(), 4);
   nir_deref_instr *w_base_v4 = build_ssbo_base_deref(
      b, &plan->weights, ssbo_addr_format, glsl_vec4_type(), 16);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *diameter = load_pc_i32(b, plan->diameter_pc_offset_B);
   nir_def *radius = nir_ishr_imm(b, diameter, 1);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tx = nir_channel(b, lid, 0);
   nir_def *ly = nir_channel(b, lid, 1);

   nir_def *gx = nir_iadd(b, nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W),
                          tx);
   nir_def *tile_y = nir_imul_imm(b, nir_channel(b, wgid, 1), TILE_H);
   nir_def *valid_x = nir_ilt(b, gx, width);

   nir_def *sh_col_v4 = nir_imul_imm(b, tx, SH_COL_V4);
   nir_def *sh_col_off = nir_imul_imm(b, tx, SH_COL_STRIDE);

   /* Cooperative 32-byte coalesced load of `64 + 2 * radius` rows into column-
    * major FP16 shared staging. Uniform interior-tile fast path omits per-row
    * bounds checks and multiplies. */
   nir_def *total_rows = nir_iadd_imm(b, nir_ishl_imm(b, radius, 1), TILE_H);
   nir_def *y_start = nir_isub(b, tile_y, radius);
   nir_def *tile_in_y = nir_iand(
      b, nir_ige_imm(b, y_start, 0),
      nir_ige(b, height, nir_iadd(b, y_start, total_rows)));

   nir_push_if(b, valid_x);
   {
      nir_push_if(b, tile_in_y);
      {
         nir_variable *r_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "vr");
         nir_variable *off_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "voff");
         nir_def *row_step = nir_ishl_imm(b, stride, 3);
         nir_store_var(b, r_var, ly, 0x1);
         nir_store_var(b, off_var,
                       nir_iadd(b, nir_imul(b, nir_iadd(b, y_start, ly), stride),
                                gx),
                       0x1);
         nir_push_loop(b);
         {
            nir_def *r = nir_load_var(b, r_var);
            nir_push_if(b, nir_ige(b, r, total_rows));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *off = nir_load_var(b, off_var);
            nir_def *byte = load_ssbo_u8(b, src_base, off);
            nir_def *f16 = nir_u2f16(b, nir_u2u16(b, byte));
            store_sh_vec4_scalar(b, sh_tile_v4, nir_iadd(b, sh_col_off, r), f16);

            nir_store_var(b, off_var, nir_iadd(b, off, row_step), 0x1);
            nir_store_var(b, r_var, nir_iadd_imm(b, r, WG_Y), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         nir_variable *r_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "vrb");
         nir_store_var(b, r_var, ly, 0x1);
         nir_loop *rbl = nir_push_loop(b);
         rbl->control = nir_loop_control_dont_unroll;
         {
            nir_def *r = nir_load_var(b, r_var);
            nir_push_if(b, nir_ige(b, r, total_rows));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *gy = nir_iadd(b, y_start, r);
            nir_push_if(b, nir_iand(b, nir_ige_imm(b, gy, 0),
                                    nir_ilt(b, gy, height)));
            {
               nir_def *byte = load_ssbo_u8(
                  b, src_base, nir_iadd(b, nir_imul(b, gy, stride), gx));
               nir_def *f16 = nir_u2f16(b, nir_u2u16(b, byte));
               store_sh_vec4_scalar(b, sh_tile_v4, nir_iadd(b, sh_col_off, r),
                                    f16);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, r_var, nir_iadd_imm(b, r, WG_Y), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   nir_push_if(b, valid_x);
   {
      nir_def *ly8 = nir_ishl_imm(b, ly, 3);
      nir_def *y0 = nir_iadd(b, tile_y, ly8);
      nir_def *max_int_y = nir_isub(b, height, radius);
      nir_def *all_interior = nir_iand(
         b, nir_ige(b, y0, radius),
         nir_ilt(b, nir_iadd_imm(b, y0, 7), max_int_y));

      nir_push_if(b, all_interior);
      {
         nir_def *sums[8];
         emit_vec4_sliding_stencil(
            b, sh_tile_v4, true, w_base, w_base_v4,
            nir_iadd(b, sh_col_v4, nir_ishl_imm(b, ly, 1)), diameter, 8, sums);

         nir_def *dst_off = nir_iadd(b, nir_imul(b, y0, stride), gx);
         for (unsigned p = 0; p < 8; p++) {
            nir_def *out_b = convert_sum_to_u8(b, sums[p], plan->store_conv);
            store_ssbo_u8(b, dst_base, dst_off, out_b);
            if (p < 7)
               dst_off = nir_iadd(b, dst_off, stride);
         }
      }
      nir_push_else(b, NULL);
      {
         /* Compact runtime loop for cold border/edge rows (8 rows/thread). */
         nir_variable *p_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "vp");
         nir_store_var(b, p_var, nir_imm_int(b, 0), 0x1);
         nir_loop *vpl = nir_push_loop(b);
         vpl->control = nir_loop_control_dont_unroll;
         {
            nir_def *p = nir_load_var(b, p_var);
            nir_push_if(b, nir_ige_imm(b, p, 8));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *yp = nir_iadd(b, y0, p);
            nir_push_if(b, nir_ilt(b, yp, height));
            {
               nir_def *is_border =
                  nir_ior(b, nir_ilt(b, yp, radius), nir_ige(b, yp, max_int_y));
               nir_def *dst_idx = nir_iadd(b, nir_imul(b, yp, stride), gx);
               nir_push_if(b, is_border);
               {
                  nir_def *center_r =
                     nir_iadd(b, nir_iadd(b, ly8, p), radius);
                  nir_def *center_sh = nir_iadd(b, sh_col_off, center_r);
                  nir_def *orig_b = nir_u2u8(
                     b, nir_f2u16(b, load_sh_vec4_scalar(b, sh_tile_v4,
                                                         center_sh)));
                  store_ssbo_u8(b, dst_base, dst_idx, orig_b);
               }
               nir_push_else(b, NULL);
               {
                  nir_def *base_sh =
                     nir_iadd(b, sh_col_off, nir_iadd(b, ly8, p));
                  nir_def *sum = emit_edge_1px_stencil(
                     b, sh_tile_v4, true, w_base, base_sh, diameter);
                  store_ssbo_u8(b, dst_base, dst_idx,
                                convert_sum_to_u8(b, sum, plan->store_conv));
               }
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, p_var, nir_iadd_imm(b, p, 1), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static nir_def *
decode_rgba8(nir_builder *b, nir_def *pixel)
{
   nir_def *channels[4];
   for (unsigned ch = 0; ch < 4; ch++)
      channels[ch] =
         nir_u2f32(b, nir_ubitfield_extract_imm(b, pixel, 8 * ch, 8));
   return nir_vec(b, channels, 4);
}

/* A byte is exactly representable as binary16. Convert once as each pixel
 * enters the shared tile and widen only the input to each FP32 FMA.
 */
static void
stage_rgba8(nir_builder *b, nir_deref_instr *tile, nir_def *index,
            nir_def *pixel)
{
   nir_def *c[4];
   for (unsigned i = 0; i < 4; i++)
      c[i] = nir_u2f16(b, nir_ubitfield_extract_imm(b, pixel, 8 * i, 8));

   nir_store_deref(b, nir_build_deref_array(b, tile, index), nir_vec(b, c, 4),
                   0xf);
}

static void
stage_rgba8x4_swizzled(nir_builder *b, nir_deref_instr *tile,
                       nir_def *sh_row_off, nir_def *col, nir_def *swiz,
                       nir_def *pixels)
{
   /* `col` is a multiple of 4 and `swiz` is 0 or 2, so each pair stays
    * 16-byte aligned and contiguous in shared memory. */
   nir_def *idx_lo = nir_iadd(b, sh_row_off, nir_ixor(b, col, swiz));
   nir_def *idx_hi =
      nir_iadd(b, sh_row_off, nir_ixor(b, nir_iadd_imm(b, col, 2), swiz));
   stage_rgba8(b, tile, idx_lo, nir_channel(b, pixels, 0));
   stage_rgba8(b, tile, nir_iadd_imm(b, idx_lo, 1), nir_channel(b, pixels, 1));
   stage_rgba8(b, tile, idx_hi, nir_channel(b, pixels, 2));
   stage_rgba8(b, tile, nir_iadd_imm(b, idx_hi, 1), nir_channel(b, pixels, 3));
}

static nir_def *
load_rgba8(nir_builder *b, nir_deref_instr *tile, nir_def *index)
{
   return nir_f2f32(b,
                    nir_load_deref(b, nir_build_deref_array(b, tile, index)));
}

static nir_def *
pack_rgba8_sum(nir_builder *b, nir_def *sum,
               enum hk_byte_stencil_store_conv conv)
{
   nir_def *i = conv == HK_BYTE_STENCIL_CONV_F2U32 ? nir_f2u32(b, sum)
                                                   : nir_f2i32(b, sum);
   return nir_pack_32_4x8(b, nir_u2u8(b, i));
}

static nir_def *
emit_rgba8_edge_1px_stencil(nir_builder *b, nir_deref_instr *tile,
                            nir_deref_instr *w_base, nir_def *sh_base,
                            nir_def *col_start, nir_def *swiz,
                            nir_def *n_taps, bool packed,
                            enum hk_byte_stencil_store_conv conv)
{
   nir_variable *index =
      nir_local_variable_create(b->impl, glsl_int_type(), "i");
   nir_variable *sum =
      nir_local_variable_create(b->impl, glsl_vec4_type(), "sum");
   nir_store_var(b, index, nir_imm_int(b, 0), 1);
   nir_store_var(b, sum, nir_imm_zero(b, 4, 32), 0xf);

   nir_loop *loop = nir_push_loop(b);
   loop->control = nir_loop_control_dont_unroll;
   {
      nir_def *i = nir_load_var(b, index);
      nir_push_if(b, nir_ige(b, i, n_taps));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *sample_index =
         swiz ? nir_iadd(b, sh_base, nir_ixor(b, nir_iadd(b, col_start, i), swiz))
              : nir_iadd(b, sh_base, i);
      nir_def *pixel = packed
                          ? decode_rgba8(b, load_sh_vec4_scalar(b, tile, sample_index))
                          : load_rgba8(b, tile, sample_index);
      nir_def *w = load_ssbo_f32(b, w_base, i);
      nir_store_var(b, sum, nir_ffma(b, pixel, w, nir_load_var(b, sum)), 0xf);
      nir_store_var(b, index, nir_iadd_imm(b, i, 1), 1);
   }
   nir_pop_loop(b, NULL);
   return pack_rgba8_sum(b, nir_load_var(b, sum), conv);
}

/* Walk decoded RGBA pixels once, distributing each sample to the four
 * neighboring outputs in strict ascending tap order while carrying the
 * trailing coefficients from the preceding 4-tap block. */
static void
rgba8_accumulate_decoded_block(nir_builder *b, nir_def *const pixels[4],
                               nir_def *prev, nir_def *weights, unsigned count,
                               unsigned samples, nir_variable *sums[4])
{
   for (unsigned k = 0; k < samples; k++) {
      nir_def *pixel = pixels[k];
      for (unsigned p = 0; p < 4; p++) {
         int t = (int)k - (int)p;
         if (t >= (int)count)
            continue;

         nir_def *w =
            t < 0 ? nir_channel(b, prev, 4 + t) : nir_channel(b, weights, t);
         nir_store_var(b, sums[p],
                       nir_ffma(b, pixel, w, nir_load_var(b, sums[p])), 0xf);
      }
   }
}

static void
emit_rgba8_horiz_streaming_stencil(nir_builder *b, nir_deref_instr *tile,
                                   nir_deref_instr *w_base,
                                   nir_deref_instr *w_base_v4,
                                   nir_def *sh_row_off, nir_def *lx0,
                                   nir_def *swiz, nir_def *tap_count,
                                   enum hk_byte_stencil_store_conv conv,
                                   nir_def *out_pixels[4])
{
   nir_def *n4 = nir_ishr_imm(b, tap_count, 2);
   nir_variable *i4_var =
      nir_local_variable_create(b->impl, glsl_int_type(), "group");
   nir_variable *trailing =
      nir_local_variable_create(b->impl, glsl_vec4_type(), "weights");
   nir_variable *sums[4];
   for (unsigned p = 0; p < 4; p++) {
      sums[p] = nir_local_variable_create(b->impl, glsl_vec4_type(), "sum");
      nir_store_var(b, sums[p], nir_imm_zero(b, 4, 32), 0xf);
   }
   /* Before the first tap the inputs are finite bytes and every sum is +0. */
   nir_store_var(b, trailing, nir_imm_zero(b, 4, 32), 0xf);
   nir_store_var(b, i4_var, nir_imm_int(b, 0), 0x1);

   /* `lx0` and `i4 << 2` are multiples of 4 and `swiz` is 0 or 2, so the
    * XOR bank swizzle distributes over `+ (i4 << 2)` and hoists out of the
    * inner tap loop. */
   nir_def *base_lo = nir_iadd(b, sh_row_off, nir_ixor(b, lx0, swiz));
   nir_def *base_hi =
      nir_iadd(b, sh_row_off, nir_ixor(b, nir_iadd_imm(b, lx0, 2), swiz));

   /* Walk each 4-pixel shared group once (two 16-byte loads instead of four),
    * issuing 24 FMAs against `trailing` before touching the freshly loaded
    * `wv` so global weight latency is hidden. */
   nir_loop *taps = nir_push_loop(b);
   taps->control = nir_loop_control_dont_unroll;
   {
      nir_def *i4 = nir_load_var(b, i4_var);
      nir_push_if(b, nir_ige(b, i4, n4));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *wv = load_ssbo_f32x4(b, w_base_v4, i4);
      nir_def *step = nir_ishl_imm(b, i4, 2);
      nir_def *idx_lo = nir_iadd(b, base_lo, step);
      nir_def *idx_hi = nir_iadd(b, base_hi, step);
      nir_def *pixels[4] = {
         load_rgba8(b, tile, idx_lo),
         load_rgba8(b, tile, nir_iadd_imm(b, idx_lo, 1)),
         load_rgba8(b, tile, idx_hi),
         load_rgba8(b, tile, nir_iadd_imm(b, idx_hi, 1)),
      };
      rgba8_accumulate_decoded_block(b, pixels, nir_load_var(b, trailing), wv,
                                     4, 4, sums);
      nir_store_var(b, trailing, wv, 0xf);
      nir_store_var(b, i4_var, nir_iadd_imm(b, i4, 1), 0x1);
   }
   nir_pop_loop(b, NULL);

   /* One or three coefficients remain. Only the three-tap tail reaches into
    * the next 16-byte shared pair (`col_t + 4`, `col_t + 5`). */
   nir_def *last = nir_iand_imm(b, tap_count, ~3);
   nir_def *tail_lo = nir_iadd(b, base_lo, last);
   nir_def *tail_hi = nir_iadd(b, base_hi, last);
   nir_def *pixels[4] = {
      load_rgba8(b, tile, tail_lo),
      load_rgba8(b, tile, nir_iadd_imm(b, tail_lo, 1)),
      load_rgba8(b, tile, tail_hi),
      load_rgba8(b, tile, nir_iadd_imm(b, tail_hi, 1)),
   };
   nir_def *prev = nir_load_var(b, trailing);
   nir_def *w0 = load_ssbo_f32(b, w_base, last);
   rgba8_accumulate_decoded_block(b, pixels, prev, w0, 1, 4, sums);
   nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, tap_count, 2), 0));
   {
      nir_def *end = nir_iadd_imm(b, tap_count, -1);
      nir_def *w1 =
         load_ssbo_f32(b, w_base, nir_umin(b, end, nir_iadd_imm(b, last, 1)));
      nir_def *w2 = load_ssbo_f32(b, w_base, end);
      nir_def *idx_next = nir_iadd_imm(b, tail_lo, 4);
      nir_def *six[6] = {
         pixels[0], pixels[1], pixels[2], pixels[3],
         load_rgba8(b, tile, idx_next),
         load_rgba8(b, tile, nir_iadd_imm(b, idx_next, 1)),
      };
      for (unsigned p = 0; p < 4; p++) {
         nir_def *sum = nir_ffma(b, six[p + 1], w1, nir_load_var(b, sums[p]));
         nir_store_var(b, sums[p], nir_ffma(b, six[p + 2], w2, sum), 0xf);
      }
   }
   nir_pop_if(b, NULL);

   for (unsigned p = 0; p < 4; p++)
      out_pixels[p] = pack_rgba8_sum(b, nir_load_var(b, sums[p]), conv);
}

/* Horizontal packed-RGBA8 tiled clamped stencil: 32x4 threads (128/WG),
 * 128x4 pixel tile (4 adjacent horizontal pixels/thread), FP16 RGBA shared
 * staging (8 KiB) with XOR 16-byte bank swizzle + 4-row-per-SIMDgroup compute
 * mapping for bank-conflict-free 16-byte shared reads, uniform f32x4 weights,
 * and 16-byte coalesced u32x4 interior loads & stores. */
static void
build_rgba8_horiz_tiled_kernel(nir_builder *b,
                               const struct hk_byte_stencil_plan *plan,
                               nir_address_format ssbo_addr_format)
{
   const unsigned WG_X = 32;
   const unsigned WG_Y = 4;
   const unsigned TILE_W = 128;
   const unsigned SH_ROW_STRIDE = TILE_W + 2 * HK_RGBA8_STENCIL_MAX_RADIUS;

   b->shader->info.workgroup_size[0] = WG_X;
   b->shader->info.workgroup_size[1] = WG_Y;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_tile_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_FLOAT16, 4),
                      WG_Y * SH_ROW_STRIDE, 0),
      "sh_tile");
   nir_deref_instr *tile = nir_build_deref_var(b, sh_tile_var);

   nir_deref_instr *src_base = build_ssbo_base_deref_aligned(
      b, &plan->src, ssbo_addr_format, glsl_uint_type(), 4, 4);
   nir_deref_instr *src_base_u4 = build_ssbo_base_deref_aligned(
      b, &plan->src, ssbo_addr_format, glsl_uvec4_type(), 4, 4);
   nir_deref_instr *dst_base = build_ssbo_base_deref_aligned(
      b, &plan->dst, ssbo_addr_format, glsl_uint_type(), 4, 4);
   nir_deref_instr *dst_base_u4 = build_ssbo_base_deref_aligned(
      b, &plan->dst, ssbo_addr_format, glsl_uvec4_type(), 4, 4);
   nir_deref_instr *w_base = build_ssbo_base_deref_aligned(
      b, &plan->weights, ssbo_addr_format, glsl_float_type(), 4, 4);
   nir_deref_instr *w_base_v4 = build_ssbo_base_deref_aligned(
      b, &plan->weights, ssbo_addr_format, glsl_vec4_type(), 16, 16);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *diameter = load_pc_i32(b, plan->diameter_pc_offset_B);
   nir_def *radius = nir_ishr_imm(b, diameter, 1);
   nir_def *tap_count = nir_ior_imm(b, nir_ishl_imm(b, radius, 1), 1);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tx = nir_channel(b, lid, 0);
   nir_def *ly = nir_channel(b, lid, 1);

   nir_def *tile_x = nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W);
   nir_def *tile_y = nir_imul_imm(b, nir_channel(b, wgid, 1), WG_Y);
   nir_def *gy = nir_iadd(b, tile_y, ly);
   nir_def *valid_y = nir_ilt(b, gy, height);
   nir_def *row_base = nir_imul(b, gy, stride);
   nir_def *sh_row_off = nir_imul_imm(b, ly, SH_ROW_STRIDE);
   nir_def *swiz = nir_ishl_imm(b, nir_iand_imm(b, ly, 1), 1);

   nir_def *total_k = nir_iadd_imm(b, nir_ishl_imm(b, radius, 1), TILE_W);
   nir_def *x_start = nir_isub(b, tile_x, radius);
   nir_def *tile_in_x = nir_iand(
      b, nir_ige_imm(b, x_start, 0),
      nir_ige(b, width, nir_iadd(b, x_start, total_k)));

   nir_push_if(b, valid_y);
   {
      nir_push_if(b, tile_in_x);
      {
         nir_def *k0 = nir_ishl_imm(b, tx, 2);
         nir_def *v0 = load_ssbo_u32x4(
            b, src_base_u4, nir_iadd(b, row_base, nir_iadd(b, x_start, k0)));
         stage_rgba8x4_swizzled(b, tile, sh_row_off, k0, swiz, v0);

         nir_def *k1 = nir_iadd_imm(b, k0, TILE_W);
         nir_push_if(b, nir_ige(b, total_k, nir_iadd_imm(b, k1, 4)));
         {
            nir_def *v1 = load_ssbo_u32x4(
               b, src_base_u4, nir_iadd(b, row_base, nir_iadd(b, x_start, k1)));
            stage_rgba8x4_swizzled(b, tile, sh_row_off, k1, swiz, v1);
         }
         nir_push_else(b, NULL);
         {
            nir_push_if(b, nir_ilt(b, k1, total_k));
            {
               /* total_k = 128 + 2*radius is even, so at most 2 pixels remain. */
               nir_def *base_g = nir_iadd(b, row_base, nir_iadd(b, x_start, k1));
               nir_def *idx_rem =
                  nir_iadd(b, sh_row_off, nir_ixor(b, k1, swiz));
               nir_def *s0 = load_ssbo_u32(b, src_base, base_g);
               stage_rgba8(b, tile, idx_rem, s0);
               nir_push_if(b, nir_ilt(b, nir_iadd_imm(b, k1, 1), total_k));
               {
                  nir_def *s1 =
                     load_ssbo_u32(b, src_base, nir_iadd_imm(b, base_g, 1));
                  stage_rgba8(b, tile, nir_iadd_imm(b, idx_rem, 1), s1);
               }
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         nir_variable *k_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "rhk");
         nir_store_var(b, k_var, tx, 0x1);
         nir_loop *bl = nir_push_loop(b);
         bl->control = nir_loop_control_dont_unroll;
         {
            nir_def *k = nir_load_var(b, k_var);
            nir_push_if(b, nir_ige(b, k, total_k));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *gx = nir_iadd(b, x_start, k);
            nir_push_if(b, nir_iand(b, nir_ige_imm(b, gx, 0),
                                    nir_ilt(b, gx, width)));
            {
               nir_def *px =
                  load_ssbo_u32(b, src_base, nir_iadd(b, row_base, gx));
               stage_rgba8(b, tile,
                           nir_iadd(b, sh_row_off, nir_ixor(b, k, swiz)), px);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, k_var, nir_iadd_imm(b, k, WG_X), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   /* Map each 32-lane SIMDgroup across all 4 rows x 8 horizontal quads
    * (`ly_c = tx & 3`, `tx_c = (ly << 3) | (tx >> 2)`). Combined with the
    * `col ^ ((ly_c & 1) << 1)` swizzle, each 16-byte shared load spans all 8
    * 16-byte bank groups equally (32 banks, 0 conflicts), while each row's 8
    * lanes still write a contiguous 128-byte cache line to global memory. */
   nir_def *ly_c = nir_iand_imm(b, tx, 3);
   nir_def *tx_c = nir_ior(b, nir_ishl_imm(b, ly, 3), nir_ushr_imm(b, tx, 2));
   nir_def *gy_c = nir_iadd(b, tile_y, ly_c);
   nir_def *valid_y_c = nir_ilt(b, gy_c, height);
   nir_def *row_base_c = nir_imul(b, gy_c, stride);
   nir_def *sh_row_c = nir_imul_imm(b, ly_c, SH_ROW_STRIDE);
   nir_def *swiz_c = nir_ishl_imm(b, nir_iand_imm(b, ly_c, 1), 1);

   nir_push_if(b, valid_y_c);
   {
      nir_def *lx0 = nir_ishl_imm(b, tx_c, 2);
      nir_def *x0 = nir_iadd(b, tile_x, lx0);
      nir_def *max_int_x = nir_isub(b, width, radius);
      nir_def *all_interior = nir_iand(
         b, nir_ige(b, x0, radius),
         nir_ilt(b, nir_iadd_imm(b, x0, 3), max_int_x));

      nir_push_if(b, all_interior);
      {
         nir_def *outs[4];
         emit_rgba8_horiz_streaming_stencil(
            b, tile, w_base, w_base_v4, sh_row_c, lx0, swiz_c, tap_count,
            plan->store_conv, outs);
         store_ssbo_u32x4(b, dst_base_u4, nir_iadd(b, row_base_c, x0),
                          nir_vec(b, outs, 4));
      }
      nir_push_else(b, NULL);
      {
         nir_def *w_m1 = nir_iadd_imm(b, width, -1);
         nir_variable *p_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "rhp");
         nir_store_var(b, p_var, nir_imm_int(b, 0), 0x1);
         nir_loop *pl = nir_push_loop(b);
         pl->control = nir_loop_control_dont_unroll;
         {
            nir_def *p = nir_load_var(b, p_var);
            nir_push_if(b, nir_ige_imm(b, p, 4));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *xp = nir_iadd(b, x0, p);
            nir_push_if(b, nir_ilt(b, xp, width));
            {
               nir_def *c_min =
                  nir_imax(b, nir_imm_int(b, 0), nir_isub(b, xp, radius));
               nir_def *c_max = nir_imin(b, w_m1, nir_iadd(b, xp, radius));
               nir_def *n_taps = nir_iadd_imm(b, nir_isub(b, c_max, c_min), 1);
               nir_def *col_start = nir_isub(b, c_min, x_start);
               nir_def *out_px = emit_rgba8_edge_1px_stencil(
                  b, tile, w_base, sh_row_c, col_start, swiz_c, n_taps, false,
                  plan->store_conv);
               store_ssbo_u32(b, dst_base, nir_iadd(b, row_base_c, xp), out_px);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, p_var, nir_iadd_imm(b, p, 1), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Walk the input pixels once, distributing each decoded sample to the four
 * neighboring outputs. At a block boundary the first samples still contribute
 * to the preceding weight block. Carry those coefficients across iterations.
 */
static void
rgba8_accumulate_block(nir_builder *b, nir_def *pixels, nir_def *prev,
                       nir_def *weights, unsigned count, unsigned samples,
                       nir_variable *sums[4])
{
   for (unsigned k = 0; k < samples; k++) {
      nir_def *pixel = decode_rgba8(b, nir_channel(b, pixels, k));
      for (unsigned p = 0; p < 4; p++) {
         int t = (int)k - (int)p;
         if (t >= (int)count)
            continue;

         nir_def *w =
            t < 0 ? nir_channel(b, prev, 4 + t) : nir_channel(b, weights, t);
         nir_store_var(b, sums[p],
                       nir_ffma(b, pixel, w, nir_load_var(b, sums[p])), 0xf);
      }
   }
}

static void
emit_rgba8_packed_stencil(nir_builder *b, nir_deref_instr *sh_tile_u4,
                          nir_deref_instr *w_base, nir_deref_instr *w_base_v4,
                          nir_def *base_v4, nir_def *tap_count,
                          enum hk_byte_stencil_store_conv conv,
                          nir_def *out_pixels[4])
{
   nir_def *n4 = nir_ishr_imm(b, tap_count, 2);
   nir_variable *i4_var =
      nir_local_variable_create(b->impl, glsl_int_type(), "ri4");
   nir_variable *trailing =
      nir_local_variable_create(b->impl, glsl_vec4_type(), "weights");
   nir_variable *sums[4];
   for (unsigned p = 0; p < 4; p++) {
      sums[p] = nir_local_variable_create(b->impl, glsl_vec4_type(), "sum");
      nir_store_var(b, sums[p], nir_imm_zero(b, 4, 32), 0xf);
   }
   /* Before the first tap the inputs are finite bytes and every sum is +0. */
   nir_store_var(b, trailing, nir_imm_zero(b, 4, 32), 0xf);
   nir_store_var(b, i4_var, nir_imm_int(b, 0), 0x1);

   /* Four taps across four pixels already expose independent work for all
    * 16 channel accumulators. Another copy increases register pressure. */
   nir_loop *taps = nir_push_loop(b);
   taps->control = nir_loop_control_dont_unroll;
   {
      nir_def *i4 = nir_load_var(b, i4_var);
      nir_push_if(b, nir_ige(b, i4, n4));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *wv = load_ssbo_f32x4(b, w_base_v4, i4);
      nir_def *pixels = load_sh_vec4(b, sh_tile_u4, nir_iadd(b, base_v4, i4));
      rgba8_accumulate_block(b, pixels, nir_load_var(b, trailing), wv, 4, 4,
                             sums);
      nir_store_var(b, trailing, wv, 0xf);
      nir_store_var(b, i4_var, nir_iadd_imm(b, i4, 1), 0x1);
   }
   nir_pop_loop(b, NULL);

   /* One or three coefficients remain. Only the three-tap tail reaches the
    * next shared vector. Read exactly the valid weights in either case. */
   nir_def *last = nir_iand_imm(b, tap_count, ~3);
   nir_def *pos = nir_iadd(b, base_v4, n4);
   nir_def *pixels = load_sh_vec4(b, sh_tile_u4, pos);
   nir_def *prev = nir_load_var(b, trailing);
   nir_def *w0 = load_ssbo_f32(b, w_base, last);
   rgba8_accumulate_block(b, pixels, prev, w0, 1, 4, sums);
   nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, tap_count, 2), 0));
   {
      nir_def *end = nir_iadd_imm(b, tap_count, -1);
      nir_def *w1 =
         load_ssbo_f32(b, w_base, nir_umin(b, end, nir_iadd_imm(b, last, 1)));
      nir_def *w2 = load_ssbo_f32(b, w_base, end);
      nir_def *pair[2] = {pixels,
                          load_sh_vec4(b, sh_tile_u4, nir_iadd_imm(b, pos, 1))};
      for (unsigned p = 0; p < 4; p++) {
         nir_def *x1 =
            decode_rgba8(b, nir_channel(b, pair[(p + 1) / 4], (p + 1) % 4));
         nir_def *x2 =
            decode_rgba8(b, nir_channel(b, pair[(p + 2) / 4], (p + 2) % 4));
         nir_def *sum = nir_ffma(b, x1, w1, nir_load_var(b, sums[p]));
         nir_store_var(b, sums[p], nir_ffma(b, x2, w2, sum), 0xf);
      }
   }
   nir_pop_if(b, NULL);

   for (unsigned p = 0; p < 4; p++)
      out_pixels[p] = pack_rgba8_sum(b, nir_load_var(b, sums[p]), conv);
}

/* Vertical packed-RGBA8 tiled clamped stencil: 16x16 threads (256/WG),
 * 16x64 pixel tile (4 vertical pixels/thread), 64-byte coalesced X loads &
 * stores, column-major uvec4 shared staging (16 x 49 uvec4 = 12.25 KiB, odd
 * column stride -> zero bank conflicts), uniform f32x4 weights, and 4-pixel
 * sliding register window. */
static void
build_rgba8_vert_tiled_kernel(nir_builder *b,
                              const struct hk_byte_stencil_plan *plan,
                              nir_address_format ssbo_addr_format)
{
   const unsigned WG_X = 16;
   const unsigned WG_Y = 16;
   const unsigned TILE_W = 16;
   const unsigned TILE_H = 64;
   const unsigned SH_COL_U4 =
      49; /* 49 * 4 = 196 >= 64 + 2 * 64; odd -> 0 bank conflicts */
   const unsigned SH_COL_STRIDE = SH_COL_U4 * 4;

   b->shader->info.workgroup_size[0] = WG_X;
   b->shader->info.workgroup_size[1] = WG_Y;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_tile_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_uvec4_type(), TILE_W * SH_COL_U4, 0), "sh_tile");
   nir_deref_instr *sh_tile_u4 = nir_build_deref_var(b, sh_tile_var);

   nir_deref_instr *src_base = build_ssbo_base_deref_aligned(
      b, &plan->src, ssbo_addr_format, glsl_uint_type(), 4, 4);
   nir_deref_instr *dst_base = build_ssbo_base_deref_aligned(
      b, &plan->dst, ssbo_addr_format, glsl_uint_type(), 4, 4);
   nir_deref_instr *w_base = build_ssbo_base_deref_aligned(
      b, &plan->weights, ssbo_addr_format, glsl_float_type(), 4, 4);
   nir_deref_instr *w_base_v4 = build_ssbo_base_deref_aligned(
      b, &plan->weights, ssbo_addr_format, glsl_vec4_type(), 16, 16);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *diameter = load_pc_i32(b, plan->diameter_pc_offset_B);
   nir_def *radius = nir_ishr_imm(b, diameter, 1);
   nir_def *tap_count = nir_ior_imm(b, nir_ishl_imm(b, radius, 1), 1);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tx = nir_channel(b, lid, 0);
   nir_def *ly = nir_channel(b, lid, 1);

   nir_def *gx =
      nir_iadd(b, nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W), tx);
   nir_def *tile_y = nir_imul_imm(b, nir_channel(b, wgid, 1), TILE_H);
   nir_def *valid_x = nir_ilt(b, gx, width);

   nir_def *sh_col_u4 = nir_imul_imm(b, tx, SH_COL_U4);
   nir_def *sh_col_off = nir_imul_imm(b, tx, SH_COL_STRIDE);

   nir_def *total_rows = nir_iadd_imm(b, nir_ishl_imm(b, radius, 1), TILE_H);
   nir_def *y_start = nir_isub(b, tile_y, radius);
   nir_def *tile_in_y = nir_iand(
      b, nir_ige_imm(b, y_start, 0),
      nir_ige(b, height, nir_iadd(b, y_start, total_rows)));

   nir_push_if(b, valid_x);
   {
      nir_push_if(b, tile_in_y);
      {
         nir_variable *r_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "rvr");
         nir_variable *off_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "rvoff");
         nir_def *row_step = nir_ishl_imm(b, stride, 4);
         nir_store_var(b, r_var, ly, 0x1);
         nir_store_var(b, off_var,
                       nir_iadd(b, nir_imul(b, nir_iadd(b, y_start, ly), stride),
                                gx),
                       0x1);
         /* The workgroup loads one coalesced row per iteration. Keep this
          * staging loop compact alongside the four-pixel stencil body. */
         nir_loop *rows = nir_push_loop(b);
         rows->control = nir_loop_control_dont_unroll;
         {
            nir_def *r = nir_load_var(b, r_var);
            nir_push_if(b, nir_ige(b, r, total_rows));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *off = nir_load_var(b, off_var);
            nir_def *px = load_ssbo_u32(b, src_base, off);
            store_sh_vec4_scalar(b, sh_tile_u4, nir_iadd(b, sh_col_off, r), px);

            nir_store_var(b, off_var, nir_iadd(b, off, row_step), 0x1);
            nir_store_var(b, r_var, nir_iadd_imm(b, r, WG_Y), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         nir_variable *r_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "rvrb");
         nir_store_var(b, r_var, ly, 0x1);
         nir_loop *rbl = nir_push_loop(b);
         rbl->control = nir_loop_control_dont_unroll;
         {
            nir_def *r = nir_load_var(b, r_var);
            nir_push_if(b, nir_ige(b, r, total_rows));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *gy = nir_iadd(b, y_start, r);
            nir_push_if(b, nir_iand(b, nir_ige_imm(b, gy, 0),
                                    nir_ilt(b, gy, height)));
            {
               nir_def *px = load_ssbo_u32(
                  b, src_base, nir_iadd(b, nir_imul(b, gy, stride), gx));
               store_sh_vec4_scalar(b, sh_tile_u4, nir_iadd(b, sh_col_off, r),
                                    px);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, r_var, nir_iadd_imm(b, r, WG_Y), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   nir_push_if(b, valid_x);
   {
      nir_def *ly4 = nir_ishl_imm(b, ly, 2);
      nir_def *y0 = nir_iadd(b, tile_y, ly4);
      nir_def *max_int_y = nir_isub(b, height, radius);
      nir_def *all_interior = nir_iand(
         b, nir_ige(b, y0, radius),
         nir_ilt(b, nir_iadd_imm(b, y0, 3), max_int_y));

      nir_push_if(b, all_interior);
      {
         nir_def *outs[4];
         emit_rgba8_packed_stencil(b, sh_tile_u4, w_base, w_base_v4,
                                   nir_iadd(b, sh_col_u4, ly), tap_count,
                                   plan->store_conv, outs);
         nir_def *dst_off = nir_iadd(b, nir_imul(b, y0, stride), gx);
         for (unsigned p = 0; p < 4; p++) {
            store_ssbo_u32(b, dst_base, dst_off, outs[p]);
            if (p < 3)
               dst_off = nir_iadd(b, dst_off, stride);
         }
      }
      nir_push_else(b, NULL);
      {
         nir_def *h_m1 = nir_iadd_imm(b, height, -1);
         nir_variable *p_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "rvp");
         nir_store_var(b, p_var, nir_imm_int(b, 0), 0x1);
         nir_loop *vpl = nir_push_loop(b);
         vpl->control = nir_loop_control_dont_unroll;
         {
            nir_def *p = nir_load_var(b, p_var);
            nir_push_if(b, nir_ige_imm(b, p, 4));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *yp = nir_iadd(b, y0, p);
            nir_push_if(b, nir_ilt(b, yp, height));
            {
               nir_def *c_min =
                  nir_imax(b, nir_imm_int(b, 0), nir_isub(b, yp, radius));
               nir_def *c_max = nir_imin(b, h_m1, nir_iadd(b, yp, radius));
               nir_def *n_taps = nir_iadd_imm(b, nir_isub(b, c_max, c_min), 1);
               nir_def *sh_start =
                  nir_iadd(b, sh_col_off, nir_isub(b, c_min, y_start));
               nir_def *out_px =
                  emit_rgba8_edge_1px_stencil(b, sh_tile_u4, w_base, sh_start,
                                              NULL, NULL, n_taps, true,
                                              plan->store_conv);
               store_ssbo_u32(
                  b, dst_base, nir_iadd(b, nir_imul(b, yp, stride), gx), out_px);
            }
            nir_pop_if(b, NULL);

            nir_store_var(b, p_var, nir_iadd_imm(b, p, 1), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

nir_shader *
hk_nir_build_tiled_byte_stencil(void *mem_ctx, const nir_shader *orig_nir,
                                const struct hk_byte_stencil_plan *plan,
                                nir_address_format ssbo_addr_format)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, orig_nir->options, "%s_tiled_stencil",
      orig_nir->info.name ? orig_nir->info.name : "byte_stencil");
   b.shader->info.internal = false;
   ralloc_steal(mem_ctx, b.shader);

   b.shader->info.zero_initialize_shared_memory =
      orig_nir->info.zero_initialize_shared_memory;

   if (plan->elem_kind == HK_BYTE_STENCIL_ELEM_RGBA8_CLAMPED) {
      if (plan->axis == HK_BYTE_STENCIL_HORIZ)
         build_rgba8_horiz_tiled_kernel(&b, plan, ssbo_addr_format);
      else
         build_rgba8_vert_tiled_kernel(&b, plan, ssbo_addr_format);
   } else {
      if (plan->axis == HK_BYTE_STENCIL_HORIZ)
         build_horiz_tiled_kernel(&b, plan, ssbo_addr_format);
      else
         build_vert_tiled_kernel(&b, plan, ssbo_addr_format);
   }

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "after hk_nir_build_tiled_byte_stencil");
   return b.shader;
}

static bool
resolve_ssbo_binding(const struct hk_descriptor_state *desc,
                     const struct hk_byte_stencil_binding *b,
                     struct hk_buffer_address *addr_out)
{
   if (b->set >= HK_MAX_SETS)
      return false;
   if (desc->buffer_active & BITFIELD_BIT(b->set))
      return false;

   if (desc->push_active & BITFIELD_BIT(b->set)) {
      const struct hk_push_descriptor_set *push = desc->push[b->set];
      if (!push)
         return false;
      if (memcmp(push->layout_hash, b->set_layout_hash,
                 sizeof(push->layout_hash)) != 0)
         return false;
      if ((uint64_t)b->descriptor_offset_B + sizeof(*addr_out) > push->size)
         return false;
      memcpy(addr_out, push->data + b->descriptor_offset_B, sizeof(*addr_out));
   } else {
      const struct hk_descriptor_set *ds = desc->sets[b->set];
      if (!ds || !ds->layout || !ds->mapped_ptr)
         return false;
      if (b->binding >= ds->layout->binding_count)
         return false;
      const struct hk_descriptor_set_binding_layout *bl =
         &ds->layout->binding[b->binding];
      if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          bl->offset != b->descriptor_offset_B)
         return false;
      if (bl->flags &
          (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT))
         return false;
      if ((uint64_t)b->descriptor_offset_B + sizeof(*addr_out) > ds->size)
         return false;
      memcpy(addr_out, (const uint8_t *)ds->mapped_ptr + b->descriptor_offset_B,
             sizeof(*addr_out));
   }

   if (addr_out->base_addr == 0 || addr_out->sparse_map_page != 0 ||
       addr_out->size == 0)
      return false;
   if (addr_out->base_addr > UINT64_MAX - addr_out->size)
      return false;

   return true;
}

static bool
va_ranges_disjoint(uint64_t a_base, uint64_t a_len, uint64_t b_base,
                   uint64_t b_len)
{
   if (a_base > UINT64_MAX - a_len || b_base > UINT64_MAX - b_len)
      return false;
   return (a_base + a_len <= b_base) || (b_base + b_len <= a_base);
}

static bool
read_written_push_i32(const struct hk_descriptor_state *desc, uint16_t offset_B,
                      int32_t *val_out)
{
   if ((offset_B & 3u) != 0 || offset_B + 4u > HK_MAX_PUSH_SIZE)
      return false;
   unsigned dword = offset_B >> 2;
   if (dword >= 64 || !(desc->push_written_dwords & (UINT64_C(1) << dword)))
      return false;
   memcpy(val_out, desc->root.push + offset_B, sizeof(*val_out));
   return true;
}

bool
hk_byte_stencil_validate_dispatch_core(
   bool alias_free, const struct hk_descriptor_state *desc,
   const struct hk_byte_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y)
{
   if (!alias_free || !desc || !plan)
      return false;

   if (base_x != 0 || base_y != 0 || base_z != 0)
      return false;
   if (groups_x == 0 || groups_y == 0 || groups_z != 1)
      return false;
   if (plan->tile_w == 0 || plan->tile_h == 0)
      return false;

   int32_t width = 0, height = 0, stride = 0, diameter = 0;
   if (!read_written_push_i32(desc, plan->width_pc_offset_B, &width) ||
       !read_written_push_i32(desc, plan->height_pc_offset_B, &height) ||
       !read_written_push_i32(desc, plan->stride_pc_offset_B, &stride) ||
       !read_written_push_i32(desc, plan->diameter_pc_offset_B, &diameter))
      return false;

   if (width < 1 || height < 1 || stride < width)
      return false;

   const bool is_rgba8 =
      plan->elem_kind == HK_BYTE_STENCIL_ELEM_RGBA8_CLAMPED;
   const uint32_t max_diam =
      is_rgba8 ? HK_RGBA8_STENCIL_MAX_DIAMETER : HK_BYTE_STENCIL_MAX_DIAMETER;
   const uint64_t elem_bytes = is_rgba8 ? 4u : 1u;

   /* The tiled kernels evaluate signed 32-bit coordinates up to one tile plus
    * one stencil diameter past the image edge (e.g. `x_start + total_k`,
    * `x0 + 3`, `y_start + total_rows`, `y0 + 7`). Bound each dimension so none
    * of these can wrap. */
   if ((uint32_t)width >
          (uint32_t)INT32_MAX - (plan->tile_w + max_diam) ||
       (uint32_t)height >
          (uint32_t)INT32_MAX - (plan->tile_h + max_diam))
      return false;
   if (diameter < 1 || (uint32_t)diameter > max_diam)
      return false;

   if (!plan->is_grid_stride) {
      if ((uint64_t)groups_x * plan->orig_wg_size[0] < (uint64_t)width ||
          (uint64_t)groups_y * plan->orig_wg_size[1] < (uint64_t)height)
         return false;
   }

   /* Ensure `y * stride + x` fits in signed 32-bit int without wraparound. */
   const uint64_t img_elems =
      (uint64_t)(height - 1) * (uint64_t)stride + (uint64_t)width;
   if (img_elems > (uint64_t)INT32_MAX)
      return false;
   const uint64_t img_bytes = img_elems * elem_bytes;
   const uint64_t w_elems =
      is_rgba8 ? (uint64_t)(((diameter / 2) * 2) + 1) : (uint64_t)diameter;
   const uint64_t w_bytes = w_elems * 4u;

   struct hk_buffer_address src_addr = {0}, dst_addr = {0}, w_addr = {0};
   if (!resolve_ssbo_binding(desc, &plan->src, &src_addr) ||
       !resolve_ssbo_binding(desc, &plan->dst, &dst_addr) ||
       !resolve_ssbo_binding(desc, &plan->weights, &w_addr))
      return false;

   if (src_addr.size < img_bytes || dst_addr.size < img_bytes ||
       w_addr.size < w_bytes || (w_addr.base_addr & 15u) != 0)
      return false;
   if (is_rgba8 && ((src_addr.base_addr | dst_addr.base_addr) & 3u) != 0)
      return false;

   /* Strict VA disjointness (combined with `alias_free` physical-to-VA
    * bijection) guarantees zero aliasing between dst and src/weights. */
   if (!va_ranges_disjoint(dst_addr.base_addr, img_bytes, src_addr.base_addr,
                           img_bytes) ||
       !va_ranges_disjoint(dst_addr.base_addr, img_bytes, w_addr.base_addr,
                           w_bytes))
      return false;

   *out_groups_x = DIV_ROUND_UP((uint32_t)width, (uint32_t)plan->tile_w);
   *out_groups_y = DIV_ROUND_UP((uint32_t)height, (uint32_t)plan->tile_h);
   return true;
}

bool
hk_byte_stencil_validate_dispatch(const struct hk_device *dev,
                                  const struct hk_descriptor_state *desc,
                                  const struct hk_byte_stencil_plan *plan,
                                  uint32_t base_x, uint32_t base_y,
                                  uint32_t base_z, uint32_t groups_x,
                                  uint32_t groups_y, uint32_t groups_z,
                                  uint32_t *out_groups_x,
                                  uint32_t *out_groups_y)
{
   if (!dev)
      return false;

   /* Guarantee 1:1 physical-to-VA mapping across all writable buffers so VA
    * disjointness implies physical non-aliasing. */
   const bool alias_free =
      p_atomic_read(&dev->va_alias_resources) == 0 &&
      p_atomic_read(&dev->alias_live) == 0 &&
      !hk_cdm_alias_scope(dev);

   return hk_byte_stencil_validate_dispatch_core(
      alias_free, desc, plan, base_x, base_y, base_z, groups_x, groups_y,
      groups_z, out_groups_x, out_groups_y);
}
