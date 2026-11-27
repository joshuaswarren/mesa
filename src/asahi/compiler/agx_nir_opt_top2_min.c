/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

/*
 * Simplify top-2 minimum (argmin2) loop recurrences:
 *
 * When a loop maintains two 32-bit integer running minima `(b1, b2)` with
 * `init(b1) <= init(b2)` (e.g. both initialized to INT_MAX) and updates them
 * each iteration via:
 *
 *    if (d < b1)      { b2 = b1; b1 = d; best_idx = j; }
 *    else if (d < b2) { b2 = d; }
 *
 * After peephole-select and algebraic folding, NIR expands this to ~11 scalar
 * ops per step including a redundant `imin(b1, b2)` and a tautological
 * `!(d >= b1 && d < b2)` mask on `d < min(b1, b2)`.
 *
 * Because `b1 <= b2` is a loop invariant (base: `init1 <= init2`; step:
 * `min(d, b1) <= min(max(d, b1), b2)` for every 32-bit integer `d`), this
 * collapses bit-exactly to:
 *
 *    lt1 = d < b1
 *    b1' = min(d, b1)
 *    b2' = min(max(d, b1), b2)
 *    best_idx' = lt1 ? j : best_idx
 */

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

static bool
match_bool_and(nir_def *def, nir_def **a_out, nir_def **b_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iand) {
      *a_out = strip_mov(alu->src[0].src.ssa);
      *b_out = strip_mov(alu->src[1].src.ssa);
      return true;
   }
   if (alu->op == nir_op_bcsel && nir_src_is_const(alu->src[2].src) &&
       !nir_src_as_bool(alu->src[2].src)) {
      *a_out = strip_mov(alu->src[0].src.ssa);
      *b_out = strip_mov(alu->src[1].src.ssa);
      return true;
   }
   return false;
}

static bool
match_lt(nir_def *def, nir_op lt_op, nir_def *lhs, nir_def *rhs)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   return alu->op == lt_op && strip_mov(alu->src[0].src.ssa) == lhs &&
          strip_mov(alu->src[1].src.ssa) == rhs;
}

static bool
match_ge(nir_def *def, nir_op ge_op, nir_op lt_op, nir_def *lhs, nir_def *rhs)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == ge_op && strip_mov(alu->src[0].src.ssa) == lhs &&
       strip_mov(alu->src[1].src.ssa) == rhs)
      return true;
   if (alu->op == nir_op_inot)
      return match_lt(alu->src[0].src.ssa, lt_op, lhs, rhs);
   return false;
}

static bool
match_min_of(nir_def *def, nir_op min_op, nir_def *a, nir_def *b)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != min_op)
      return false;
   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   return (s0 == a && s1 == b) || (s0 == b && s1 == a);
}

static bool
match_lt_b1_or_min(nir_def *def, nir_op lt_op, nir_op min_op, nir_def *d,
                   nir_def *b1, nir_def *b2)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != lt_op || strip_mov(alu->src[0].src.ssa) != d)
      return false;
   nir_def *rhs = strip_mov(alu->src[1].src.ssa);
   return rhs == b1 || match_min_of(rhs, min_op, b1, b2);
}

static bool
match_mid_range(nir_def *def, nir_op lt_op, nir_op ge_op, nir_def *d,
                nir_def *b1, nir_def *b2)
{
   nir_def *a = NULL, *b = NULL;
   if (!match_bool_and(def, &a, &b))
      return false;
   return (match_lt(a, lt_op, d, b2) && match_ge(b, ge_op, lt_op, d, b1)) ||
          (match_lt(b, lt_op, d, b2) && match_ge(a, ge_op, lt_op, d, b1));
}

static bool
match_is_new_b1(nir_def *def, nir_op lt_op, nir_op ge_op, nir_op min_op,
                nir_def *d, nir_def *b1, nir_def *b2, nir_def **lt_m_out)
{
   def = strip_mov(def);
   if (match_lt_b1_or_min(def, lt_op, min_op, d, b1, b2)) {
      *lt_m_out = def;
      return true;
   }
   nir_def *a = NULL, *b = NULL;
   if (!match_bool_and(def, &a, &b))
      return false;
   for (unsigned s = 0; s < 2; s++) {
      nir_def *x = s ? b : a;
      nir_def *y = s ? a : b;
      if (match_lt_b1_or_min(x, lt_op, min_op, d, b1, b2) &&
          nir_def_instr(y)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(y))->op == nir_op_inot &&
          match_mid_range(nir_instr_as_alu(nir_def_instr(y))->src[0].src.ssa,
                          lt_op, ge_op, d, b1, b2)) {
         *lt_m_out = x;
         return true;
      }
   }
   return false;
}

static bool
inits_ordered(nir_def *init1, nir_def *init2, bool is_signed)
{
   init1 = strip_mov(init1);
   init2 = strip_mov(init2);
   if (init1 == init2)
      return true;
   if (nir_src_is_const(nir_src_for_ssa(init1)) &&
       nir_src_is_const(nir_src_for_ssa(init2))) {
      if (is_signed)
         return nir_src_as_int(nir_src_for_ssa(init1)) <=
                nir_src_as_int(nir_src_for_ssa(init2));
      return nir_src_as_uint(nir_src_for_ssa(init1)) <=
             nir_src_as_uint(nir_src_for_ssa(init2));
   }
   return false;
}

struct phi_rec {
   nir_phi_instr *phi;
   nir_def *init;
   nir_def *next;
};

static bool
try_opt_top2_in_loop(nir_function_impl *impl, nir_loop *loop)
{
   nir_block *hdr = nir_loop_first_block(loop);
   nir_cf_node *prev_cf = nir_cf_node_prev(&loop->cf_node);
   if (!prev_cf || prev_cf->type != nir_cf_node_block)
      return false;
   nir_block *pre = nir_cf_node_as_block(prev_cf);
   nir_block *last = nir_loop_last_block(loop);

   struct phi_rec phis[32];
   unsigned n_phis = 0;

   nir_foreach_phi(phi, hdr) {
      if (phi->def.bit_size != 32 || phi->def.num_components != 1)
         continue;
      nir_def *init = NULL, *next = NULL;
      unsigned cnt = 0;
      bool ok = true;
      nir_foreach_phi_src(ps, phi) {
         cnt++;
         if (ps->pred == pre) {
            init = ps->src.ssa;
         } else if (ps->pred->index >= hdr->index &&
                    ps->pred->index <= last->index) {
            if (next) {
               ok = false;
               break;
            }
            next = ps->src.ssa;
         } else {
            ok = false;
            break;
         }
      }
      if (ok && cnt == 2 && init && next && n_phis < ARRAY_SIZE(phis)) {
         phis[n_phis++] = (struct phi_rec){phi, strip_mov(init), strip_mov(next)};
      }
   }

   if (n_phis < 2)
      return false;

   bool progress = false;

   for (unsigned i = 0; i < n_phis; i++) {
      nir_def *b1 = &phis[i].phi->def;
      nir_def *b1_next = phis[i].next;
      if (!b1_next || nir_def_instr(b1_next)->type != nir_instr_type_alu)
         continue;

      /* Candidate b1_next: `bcsel(c1, d, b1)` or `imin/umin(d, b1)`. */
      nir_alu_instr *b1_alu = nir_instr_as_alu(nir_def_instr(b1_next));
      nir_def *d = NULL;
      nir_def *c1 = NULL;
      bool b1_is_min = false;
      bool b1_signed_hint = true;

      if (b1_alu->op == nir_op_bcsel &&
          strip_mov(b1_alu->src[2].src.ssa) == b1) {
         c1 = strip_mov(b1_alu->src[0].src.ssa);
         d = strip_mov(b1_alu->src[1].src.ssa);
      } else if (b1_alu->op == nir_op_imin || b1_alu->op == nir_op_umin) {
         b1_is_min = true;
         b1_signed_hint = (b1_alu->op == nir_op_imin);
         nir_def *s0 = strip_mov(b1_alu->src[0].src.ssa);
         nir_def *s1 = strip_mov(b1_alu->src[1].src.ssa);
         if (s0 == b1)
            d = s1;
         else if (s1 == b1)
            d = s0;
         else
            continue;
      } else {
         continue;
      }

      if (!d || d == b1 || d->bit_size != 32 || d->num_components != 1)
         continue;

      for (unsigned j = 0; j < n_phis; j++) {
         if (i == j)
            continue;
         nir_def *b2 = &phis[j].phi->def;
         nir_def *b2_next = phis[j].next;
         if (!b2_next || nir_def_instr(b2_next)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *b2_alu = nir_instr_as_alu(nir_def_instr(b2_next));
         if (b2_alu->op != nir_op_bcsel)
            continue;

         for (int sign_idx = 0; sign_idx < 2; sign_idx++) {
            const bool is_signed = (sign_idx == 0);
            if (b1_is_min && is_signed != b1_signed_hint)
               continue;
            if (!inits_ordered(phis[i].init, phis[j].init, is_signed))
               continue;

            const nir_op lt_op = is_signed ? nir_op_ilt : nir_op_ult;
            const nir_op ge_op = is_signed ? nir_op_ige : nir_op_uge;
            const nir_op min_op = is_signed ? nir_op_imin : nir_op_umin;
            const nir_op max_op = is_signed ? nir_op_imax : nir_op_umax;

            nir_def *lt_m = NULL;
            if (c1 &&
                !match_is_new_b1(c1, lt_op, ge_op, min_op, d, b1, b2, &lt_m))
               continue;

            /* Match b2_next:
             * Form A: `bcsel(mid, d, bcsel(lt_m, b1, b2))`
             * Form B: `bcsel(is_new_b1, b1, bcsel(lt_b2, d, b2))`
             * Form C: `bcsel(is_new_b1, b1, min(d, b2))`
             * Form D: `bcsel(lt_b2, max(d, b1), b2)` */
            bool matched_b2 = false;
            nir_def *b2_c = strip_mov(b2_alu->src[0].src.ssa);
            nir_def *b2_t = strip_mov(b2_alu->src[1].src.ssa);
            nir_def *b2_e = strip_mov(b2_alu->src[2].src.ssa);
            nir_def *lt_m2 = NULL;

            if (b2_t == d && match_mid_range(b2_c, lt_op, ge_op, d, b1, b2) &&
                nir_def_instr(b2_e)->type == nir_instr_type_alu) {
               nir_alu_instr *ea = nir_instr_as_alu(nir_def_instr(b2_e));
               if (ea->op == nir_op_bcsel &&
                   strip_mov(ea->src[1].src.ssa) == b1 &&
                   strip_mov(ea->src[2].src.ssa) == b2 &&
                   match_lt_b1_or_min(ea->src[0].src.ssa, lt_op, min_op, d, b1,
                                      b2)) {
                  lt_m2 = strip_mov(ea->src[0].src.ssa);
                  matched_b2 = true;
               }
            } else if (b2_t == b1 &&
                       match_is_new_b1(b2_c, lt_op, ge_op, min_op, d, b1, b2,
                                       &lt_m2)) {
               if (match_min_of(b2_e, min_op, d, b2)) {
                  matched_b2 = true;
               } else if (nir_def_instr(b2_e)->type == nir_instr_type_alu) {
                  nir_alu_instr *ea = nir_instr_as_alu(nir_def_instr(b2_e));
                  if (ea->op == nir_op_bcsel &&
                      strip_mov(ea->src[1].src.ssa) == d &&
                      strip_mov(ea->src[2].src.ssa) == b2 &&
                      match_lt(ea->src[0].src.ssa, lt_op, d, b2)) {
                     matched_b2 = true;
                  }
               }
            } else if (b2_e == b2 && match_lt(b2_c, lt_op, d, b2) &&
                       match_min_of(b2_t, max_op, d, b1)) {
               matched_b2 = true;
            }

            if (!matched_b2)
               continue;

            /* Ensure `d` dominates both b1_next and b2_next (and c1/lt_m). */
            nir_instr *d_instr = nir_def_instr(d);
            if (!nir_block_dominates(d_instr->block,
                                     nir_def_instr(b1_next)->block) ||
                !nir_block_dominates(d_instr->block,
                                     nir_def_instr(b2_next)->block))
               continue;

            nir_builder b = nir_builder_create(impl);
            b.cursor = d_instr->type == nir_instr_type_phi
                          ? nir_after_phis(d_instr->block)
                          : nir_after_instr(d_instr);

            nir_def *lt1 = nir_build_alu2(&b, lt_op, d, b1);
            nir_def *new_b1 = nir_build_alu2(&b, min_op, d, b1);
            nir_def *new_b2 =
               nir_build_alu2(&b, min_op, nir_build_alu2(&b, max_op, d, b1), b2);

            if (c1 && c1 != lt1)
               nir_def_rewrite_uses(c1, lt1);
            if (lt_m && lt_m != c1 && lt_m != lt1)
               nir_def_rewrite_uses(lt_m, lt1);
            if (lt_m2 && lt_m2 != c1 && lt_m2 != lt_m && lt_m2 != lt1)
               nir_def_rewrite_uses(lt_m2, lt1);

            nir_def_rewrite_uses(b1_next, new_b1);
            nir_def_rewrite_uses(b2_next, new_b2);
            progress = true;
            goto next_i;
         }
      }
   next_i:;
   }

   return progress;
}

static bool
visit_cf_list(nir_function_impl *impl, struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= visit_cf_list(impl, &nif->then_list);
         progress |= visit_cf_list(impl, &nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= visit_cf_list(impl, &loop->body);
         progress |= try_opt_top2_in_loop(impl, loop);
      }
   }
   return progress;
}

bool
agx_nir_opt_top2_min(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance);
      bool impl_progress = visit_cf_list(impl, &impl->body);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }
   return progress;
}
