/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Strength-reduce `idiv`/`imod` of a 0-based unit-stepped loop induction
 * variable by a loop-invariant runtime divisor into a running (quotient,
 * remainder) phi recurrence, eliminating integer division from hot loops.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "compiler/nir/nir_range_analysis.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

static bool
def_outside_loop(nir_def *def, nir_loop *loop)
{
   for (nir_cf_node *node = nir_def_block(def)->cf_node.parent; node;
        node = node->parent) {
      if (node == &loop->cf_node)
         return false;
   }
   return true;
}

static void
collect_loops_rec(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops_rec(&nif->then_list, loops);
         collect_loops_rec(&nif->else_list, loops);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_loops_rec(&loop->body, loops);
         util_dynarray_append(loops, loop);
         break;
      }
      default:
         break;
      }
   }
}

static bool
block_is_lone_break(nir_block *block)
{
   nir_instr *last = nir_block_last_instr(block);
   return last && last == nir_block_first_instr(block) &&
          last->type == nir_instr_type_jump &&
          nir_instr_as_jump(last)->type == nir_jump_break;
}

static bool
branch_is_lone_break(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          block_is_lone_break(nir_cf_node_as_block(node));
}

static bool
branch_is_empty(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(node)->instr_list);
}

struct unit_iv_loop {
   nir_loop *loop;
   nir_block *preheader;
   nir_block *header;
   nir_block *latch;
   nir_phi_instr *iv;
   nir_scalar bound;
   bool is_unsigned_cmp;
};

/* Match a loop with a single back edge (header has exactly 2 predecessors:
 * preheader and latch), a single top exit `if (iv >= bound) break` or
 * `if (!(iv < bound)) break` (signed or unsigned), and a 32-bit header phi
 * `iv` initialized to 0 and incremented by +1 on the back edge. */
static bool
match_zero_unit_loop(nir_loop *loop, struct unit_iv_loop *out)
{
   if (!nir_loop_has_back_edge(loop) || nir_loop_has_continue_construct(loop))
      return false;

   nir_block *preheader = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   if (nir_block_num_preds(header) != 2 || nir_block_ends_in_jump(latch))
      return false;

   nir_cf_node *next = nir_cf_node_next(&header->cf_node);
   if (!next || next->type != nir_cf_node_if)
      return false;

   nir_if *nif = nir_cf_node_as_if(next);
   const bool break_in_then =
      branch_is_lone_break(&nif->then_list) && branch_is_empty(&nif->else_list);
   const bool break_in_else =
      branch_is_lone_break(&nif->else_list) && branch_is_empty(&nif->then_list);
   if (!break_in_then && !break_in_else)
      return false;

   nir_scalar cond = nir_scalar_resolved(nif->condition.ssa, 0);
   bool break_when_true = break_in_then;
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      cond = nir_scalar_chase_alu_src(cond, 0);
      break_when_true = !break_when_true;
   }
   if (!nir_scalar_is_alu(cond))
      return false;

   const nir_op op = nir_scalar_alu_op(cond);
   if (op != nir_op_ilt && op != nir_op_ige &&
       op != nir_op_ult && op != nir_op_uge)
      return false;

   const bool is_unsigned_cmp = (op == nir_op_ult || op == nir_op_uge);
   nir_scalar lhs = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar rhs = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   if (lhs.def->bit_size != 32 || rhs.def->bit_size != 32)
      return false;

   /* We need continue condition `iv < bound` (i.e. break on `iv >= bound`).
    * - break_when_true && ige/uge: break on lhs >= rhs -> iv = lhs, bound = rhs
    * - !break_when_true && ilt/ult: continue on lhs < rhs -> iv = lhs, bound = rhs
    */
   const bool want_ge = break_when_true;
   const bool is_ge = (op == nir_op_ige || op == nir_op_uge);
   if (want_ge != is_ge)
      return false;

   if (lhs.comp != 0 || nir_def_instr(lhs.def)->type != nir_instr_type_phi)
      return false;

   nir_phi_instr *iv = nir_instr_as_phi(nir_def_instr(lhs.def));
   if (iv->instr.block != header || iv->def.num_components != 1 ||
       iv->def.bit_size != 32)
      return false;

   nir_phi_src *init_src = nir_phi_get_src_from_block(iv, preheader);
   nir_phi_src *back_src = nir_phi_get_src_from_block(iv, latch);
   if (!init_src || !back_src)
      return false;

   nir_scalar init_s = nir_scalar_resolved(init_src->src.ssa, 0);
   if (!nir_scalar_is_const(init_s) || nir_scalar_as_uint(init_s) != 0)
      return false;

   /* Verify back_src == iv + 1. */
   nir_scalar back_s = nir_scalar_resolved(back_src->src.ssa, 0);
   if (!nir_scalar_is_alu(back_s) || nir_scalar_alu_op(back_s) != nir_op_iadd)
      return false;
   nir_scalar a0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(back_s, 0));
   nir_scalar a1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(back_s, 1));
   const bool step_ok =
      (a0.def == &iv->def && a0.comp == 0 &&
       nir_scalar_is_const(a1) && nir_scalar_as_uint(a1) == 1) ||
      (a1.def == &iv->def && a1.comp == 0 &&
       nir_scalar_is_const(a0) && nir_scalar_as_uint(a0) == 1);
   if (!step_ok)
      return false;

   if (!nir_scalar_is_const(rhs) && !def_outside_loop(rhs.def, loop))
      return false;

   *out = (struct unit_iv_loop){
      .loop = loop,
      .preheader = preheader,
      .header = header,
      .latch = latch,
      .iv = iv,
      .bound = rhs,
      .is_unsigned_cmp = is_unsigned_cmp,
   };
   return true;
}

/* --- Pass 2: Unit-stepped IV div/mod strength reduction --- */

static bool
scalar_equal_pair(nir_scalar a, nir_scalar b)
{
   a = nir_scalar_chase_movs(a);
   b = nir_scalar_chase_movs(b);
   return a.def == b.def && a.comp == b.comp;
}

/* True when a dominating `if` guard or range analysis proves 32-bit `target`
 * is nonnegative (0 <= target <= INT32_MAX) at `block`. */
static bool
divisor_is_known_nonneg(nir_shader *shader, struct hash_table *range_ht,
                        nir_scalar target, nir_block *block)
{
   target = nir_scalar_chase_movs(target);
   if (target.def->bit_size != 32)
      return false;
   if (nir_unsigned_upper_bound(shader, range_ht, target) <= INT32_MAX)
      return true;

   for (nir_cf_node *node = &block->cf_node; node->parent; node = node->parent) {
      if (node->parent->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node->parent);
      bool in_then = false;
      foreach_list_typed(nir_cf_node, child, node, &nif->then_list) {
         if (child == node) {
            in_then = true;
            break;
         }
      }
      nir_scalar cond = nir_scalar_resolved(nif->condition.ssa, 0);
      bool truth = in_then;
      while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
         cond = nir_scalar_chase_alu_src(cond, 0);
         truth = !truth;
      }
      if (!nir_scalar_is_alu(cond))
         continue;
      const nir_op op = nir_scalar_alu_op(cond);
      if (op != nir_op_ige && op != nir_op_ilt)
         continue;
      nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
      nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
      if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
         continue;
      if (op == nir_op_ige) {
         /* truth: s0 >= s1; !truth: s0 < s1 (s1 > s0) */
         if (truth && scalar_equal_pair(s0, target) &&
             nir_scalar_is_const(s1) && nir_scalar_as_int(s1) >= 0)
            return true;
         if (!truth && scalar_equal_pair(s1, target) &&
             nir_scalar_is_const(s0) && nir_scalar_as_int(s0) >= -1)
            return true;
      } else {
         /* ilt: truth: s0 < s1 (s1 > s0); !truth: s0 >= s1 */
         if (truth && scalar_equal_pair(s1, target) &&
             nir_scalar_is_const(s0) && nir_scalar_as_int(s0) >= -1)
            return true;
         if (!truth && scalar_equal_pair(s0, target) &&
             nir_scalar_is_const(s1) && nir_scalar_as_int(s1) >= 0)
            return true;
      }
   }
   return false;
}

struct divmod_group {
   nir_scalar divisor;
   bool is_unsigned;
   nir_phi_instr *q_phi;
   nir_phi_instr *r_phi;
   nir_def *d_def;
   nir_def *d_neg; /* NULL when unsigned or divisor provably >= 0 */
   nir_def *zero;
};

bool
agx_nir_opt_iv_divmod(nir_shader *shader)
{
   bool any = false;
   struct hash_table *range_ht = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_function_impl(impl, shader) {
      bool progress = false;
      struct util_dynarray loops;
      util_dynarray_init(&loops, NULL);
      collect_loops_rec(&impl->body, &loops);

      util_dynarray_foreach(&loops, nir_loop *, lp) {
         struct unit_iv_loop ul;
         if (!match_zero_unit_loop(*lp, &ul))
            continue;

         /* For signed idiv/imod/irem, we need 0 <= iv <= INT32_MAX inside the
          * loop body. A signed `<` exit (iv < bound) proves 0 <= iv < INT32_MAX
          * unconditionally; an unsigned `<` exit proves it when bound <= INT32_MAX. */
         const bool iv_nonneg_signed =
            !ul.is_unsigned_cmp ||
            nir_unsigned_upper_bound(shader, range_ht, ul.bound) <= INT32_MAX;

         /* Collect matching div/mod instructions inside `ul.loop`. */
         struct util_dynarray targets;
         util_dynarray_init(&targets, NULL);
         nir_foreach_block_in_cf_node(block, &ul.loop->cf_node) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->def.num_components != 1 || alu->def.bit_size != 32)
                  continue;
               const nir_op op = alu->op;
               const bool is_u = (op == nir_op_udiv || op == nir_op_umod);
               const bool is_s =
                  (op == nir_op_idiv || op == nir_op_imod || op == nir_op_irem);
               if (!is_u && !is_s)
                  continue;
               if (is_s && !iv_nonneg_signed)
                  continue;

               nir_scalar num = nir_scalar_resolved(alu->src[0].src.ssa,
                                                    alu->src[0].swizzle[0]);
               nir_scalar div = nir_scalar_resolved(alu->src[1].src.ssa,
                                                    alu->src[1].swizzle[0]);
               if (num.def != &ul.iv->def || num.comp != 0)
                  continue;
               /* Leave compile-time constant divisors to nir_opt_idiv_const. */
               if (nir_scalar_is_const(div) || !def_outside_loop(div.def, ul.loop))
                  continue;

               util_dynarray_append(&targets, alu);
            }
         }

         if (util_dynarray_num_elements(&targets, nir_alu_instr *) == 0) {
            util_dynarray_fini(&targets);
            continue;
         }

         struct divmod_group groups[8];
         unsigned ngroups = 0;

         nir_builder b_pre =
            nir_builder_at(nir_after_block_before_jump(ul.preheader));
         nir_builder b_latch =
            nir_builder_at(nir_after_block_before_jump(ul.latch));
         nir_builder b_use = nir_builder_create(impl);

         util_dynarray_foreach(&targets, nir_alu_instr *, alup) {
            nir_alu_instr *alu = *alup;
            const nir_op op = alu->op;
            const bool is_u = (op == nir_op_udiv || op == nir_op_umod);
            nir_scalar div = nir_scalar_resolved(alu->src[1].src.ssa,
                                                 alu->src[1].swizzle[0]);

            struct divmod_group *grp = NULL;
            for (unsigned g = 0; g < ngroups; ++g) {
               if (groups[g].is_unsigned == is_u &&
                   scalar_equal_pair(groups[g].divisor, div)) {
                  grp = &groups[g];
                  break;
               }
            }

            if (!grp) {
               if (ngroups >= ARRAY_SIZE(groups))
                  continue;
               grp = &groups[ngroups++];
               grp->divisor = div;
               grp->is_unsigned = is_u;
               grp->zero = nir_imm_int(&b_pre, 0);
               grp->d_def = nir_channel(&b_pre, div.def, div.comp);

               nir_def *abs_d = grp->d_def;
               grp->d_neg = NULL;
               if (!is_u &&
                   !divisor_is_known_nonneg(shader, range_ht, div, ul.header)) {
                  grp->d_neg = nir_ilt_imm(&b_pre, grp->d_def, 0);
                  abs_d = nir_bcsel(&b_pre, grp->d_neg,
                                    nir_ineg(&b_pre, grp->d_def), grp->d_def);
               }

               nir_phi_instr *q_phi = nir_phi_instr_create(shader);
               nir_def_init(&q_phi->instr, &q_phi->def, 1, 32);
               nir_phi_instr_add_src(q_phi, ul.preheader, grp->zero);

               nir_phi_instr *r_phi = nir_phi_instr_create(shader);
               nir_def_init(&r_phi->instr, &r_phi->def, 1, 32);
               nir_phi_instr_add_src(r_phi, ul.preheader, grp->zero);

               nir_def *r1 = nir_iadd_imm(&b_latch, &r_phi->def, 1);
               nir_def *wrap = nir_ieq(&b_latch, r1, abs_d);
               nir_def *r_next = nir_bcsel(&b_latch, wrap, grp->zero, r1);
               nir_def *q_next =
                  nir_bcsel(&b_latch, wrap,
                            nir_iadd_imm(&b_latch, &q_phi->def, 1),
                            &q_phi->def);

               nir_phi_instr_add_src(q_phi, ul.latch, q_next);
               nir_phi_instr_add_src(r_phi, ul.latch, r_next);

               nir_instr_insert(nir_before_block(ul.header), &q_phi->instr);
               nir_instr_insert(nir_before_block(ul.header), &r_phi->instr);

               grp->q_phi = q_phi;
               grp->r_phi = r_phi;
            }

            b_use.cursor = nir_before_instr(&alu->instr);
            nir_def *q = &grp->q_phi->def;
            nir_def *r = &grp->r_phi->def;
            nir_def *res = NULL;

            switch (op) {
            case nir_op_udiv:
               res = q;
               break;
            case nir_op_umod:
            case nir_op_irem:
               res = r;
               break;
            case nir_op_idiv:
               res = grp->d_neg
                        ? nir_bcsel(&b_use, grp->d_neg, nir_ineg(&b_use, q), q)
                        : q;
               break;
            case nir_op_imod:
               if (grp->d_neg) {
                  nir_def *neg_mod =
                     nir_bcsel(&b_use, nir_ieq_imm(&b_use, r, 0), grp->zero,
                               nir_iadd(&b_use, r, grp->d_def));
                  res = nir_bcsel(&b_use, grp->d_neg, neg_mod, r);
               } else {
                  res = r;
               }
               break;
            default:
               UNREACHABLE("unexpected op");
            }

            nir_def_replace(&alu->def, res);
            progress = true;
         }

         util_dynarray_fini(&targets);
      }

      util_dynarray_fini(&loops);
      if (progress) {
         nir_progress(true, impl, nir_metadata_control_flow);
         any = true;
      } else {
         nir_no_progress(impl);
      }
   }

   _mesa_hash_table_destroy(range_ht, NULL);

   if (any) {
      NIR_PASS(_, shader, nir_opt_copy_prop);
      NIR_PASS(_, shader, nir_opt_dce);
   }
   return any;
}
