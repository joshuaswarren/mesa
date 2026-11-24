/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Fold tautological bit-membership guards inside unit-stepped [0, N) loops
 * (N <= 32) where a 32-bit bitmask accumulator starts at 0 and only ever sets
 * bit `iv` on each iteration: `(acc & (1u << iv)) == 0` is provably true on
 * every body iteration.
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

/* --- Pass 1: Monotonic bit-insert branch folding --- */

static bool
is_iv_shift_amount(nir_scalar s, nir_def *iv_def)
{
   s = nir_scalar_chase_movs(s);
   if (s.def == iv_def && s.comp == 0)
      return true;
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iand) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (a.def == iv_def && a.comp == 0 &&
          nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 31)
         return true;
      if (b.def == iv_def && b.comp == 0 &&
          nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 31)
         return true;
   }
   return false;
}

static bool
is_iv_bitmask(nir_scalar s, nir_def *iv_def)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_ishl)
      return false;
   nir_scalar base = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   return nir_scalar_is_const(base) && nir_scalar_as_uint(base) == 1 &&
          is_iv_shift_amount(sh, iv_def);
}

/* Prove that `def` equals `acc` with at most bit `iv` additionally set. */
static bool
acc_only_sets_iv_bit(nir_def *def, nir_phi_instr *acc, nir_def *iv_def,
                     nir_loop *loop, unsigned depth)
{
   if (def == &acc->def)
      return true;
   if (!depth || def->num_components != 1 || def->bit_size != 32 ||
       def_outside_loop(def, loop))
      return false;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      /* Only allow non-header (inner diamond) phis so we cannot cycle. */
      if (phi->instr.block == nir_loop_first_block(loop))
         return false;
      nir_foreach_phi_src(ps, phi) {
         if (!acc_only_sets_iv_bit(ps->src.ssa, acc, iv_def, loop, depth - 1))
            return false;
      }
      return true;
   }

   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_mov:
      return acc_only_sets_iv_bit(alu->src[0].src.ssa, acc, iv_def, loop,
                                  depth - 1);
   case nir_op_bcsel:
      return acc_only_sets_iv_bit(alu->src[1].src.ssa, acc, iv_def, loop,
                                  depth - 1) &&
             acc_only_sets_iv_bit(alu->src[2].src.ssa, acc, iv_def, loop,
                                  depth - 1);
   case nir_op_ior: {
      nir_scalar s0 = nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
      nir_scalar s1 = nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
      if (is_iv_bitmask(s1, iv_def) &&
          acc_only_sets_iv_bit(s0.def, acc, iv_def, loop, depth - 1))
         return true;
      if (is_iv_bitmask(s0, iv_def) &&
          acc_only_sets_iv_bit(s1.def, acc, iv_def, loop, depth - 1))
         return true;
      return false;
   }
   default:
      return false;
   }
}

/* Match `ior(acc, 1u << iv)` (either operand order). */
static bool
is_acc_or_iv_bit(nir_scalar s, nir_def *acc_def, nir_def *iv_def)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_ior)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   return (a.def == acc_def && a.comp == 0 && is_iv_bitmask(b, iv_def)) ||
          (b.def == acc_def && b.comp == 0 && is_iv_bitmask(a, iv_def));
}

/* Match `iand(acc, 1u << iv)` or `iand(ushr(acc, iv), 1)`. */
static bool
is_acc_test_iv_bit(nir_scalar s, nir_def *acc_def, nir_def *iv_def)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_iand)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));

   if ((a.def == acc_def && a.comp == 0 && is_iv_bitmask(b, iv_def)) ||
       (b.def == acc_def && b.comp == 0 && is_iv_bitmask(a, iv_def)))
      return true;

   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar sh = side == 0 ? a : b;
      nir_scalar one = side == 0 ? b : a;
      if (nir_scalar_is_const(one) && nir_scalar_as_uint(one) == 1 &&
          nir_scalar_is_alu(sh) && nir_scalar_alu_op(sh) == nir_op_ushr) {
         nir_scalar v = nir_scalar_chase_movs(nir_scalar_chase_alu_src(sh, 0));
         nir_scalar amt = nir_scalar_chase_movs(nir_scalar_chase_alu_src(sh, 1));
         if (v.def == acc_def && v.comp == 0 && is_iv_shift_amount(amt, iv_def))
            return true;
      }
   }
   return false;
}

bool
agx_nir_opt_monotonic_bit_insert(nir_shader *shader)
{
   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      bool progress = false;
      struct util_dynarray loops;
      util_dynarray_init(&loops, NULL);
      collect_loops_rec(&impl->body, &loops);

      util_dynarray_foreach(&loops, nir_loop *, lp) {
         struct unit_iv_loop ul;
         if (!match_zero_unit_loop(*lp, &ul))
            continue;

         /* Must have constant bound <= 32 so `iv` visits distinct bit indices
          * 0 .. bound - 1 without wrapping mod 32. */
         if (!nir_scalar_is_const(ul.bound))
            continue;
         if (ul.is_unsigned_cmp) {
            uint64_t uval = nir_scalar_as_uint(ul.bound);
            if (uval == 0 || uval > 32)
               continue;
         } else {
            int64_t bval = nir_scalar_as_int(ul.bound);
            if (bval <= 0 || bval > 32)
               continue;
         }

         /* Collect qualifying monotonic 0-initialized bitmask accumulator phis. */
         nir_phi_instr *accs[16];
         unsigned naccs = 0;
         nir_foreach_phi(phi, ul.header) {
            if (phi == ul.iv || phi->def.num_components != 1 ||
                phi->def.bit_size != 32 || naccs >= ARRAY_SIZE(accs))
               continue;
            nir_phi_src *init_s = nir_phi_get_src_from_block(phi, ul.preheader);
            nir_phi_src *back_s = nir_phi_get_src_from_block(phi, ul.latch);
            if (!init_s || !back_s)
               continue;
            nir_scalar is = nir_scalar_resolved(init_s->src.ssa, 0);
            if (!nir_scalar_is_const(is) || nir_scalar_as_uint(is) != 0)
               continue;
            if (acc_only_sets_iv_bit(back_s->src.ssa, phi, &ul.iv->def, ul.loop, 16))
               accs[naccs++] = phi;
         }
         if (!naccs)
            continue;

         nir_builder b = nir_builder_create(impl);
         nir_foreach_block_in_cf_node(block, &ul.loop->cf_node) {
            /* Skip the header before the top exit break: on the exiting
             * iteration with bound == 32, iv == 32 so (iv & 31) == 0 wraps
             * back to bit 0 which may already be set. */
            if (block == ul.header)
               continue;
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if ((alu->op != nir_op_ine && alu->op != nir_op_ieq) ||
                   alu->def.num_components != 1 || alu->def.bit_size != 1)
                  continue;

               nir_scalar s0 = nir_scalar_resolved(alu->src[0].src.ssa,
                                                   alu->src[0].swizzle[0]);
               nir_scalar s1 = nir_scalar_resolved(alu->src[1].src.ssa,
                                                   alu->src[1].swizzle[0]);
               for (unsigned i = 0; i < naccs; ++i) {
                  nir_def *ad = &accs[i]->def;
                  bool folded = false;
                  bool val = false;

                  /* (acc | (1 << iv)) != acc -> true; == -> false */
                  if ((is_acc_or_iv_bit(s0, ad, &ul.iv->def) &&
                       s1.def == ad && s1.comp == 0) ||
                      (is_acc_or_iv_bit(s1, ad, &ul.iv->def) &&
                       s0.def == ad && s0.comp == 0)) {
                     folded = true;
                     val = (alu->op == nir_op_ine);
                  } else if ((is_acc_test_iv_bit(s0, ad, &ul.iv->def) &&
                              nir_scalar_is_const(s1) &&
                              nir_scalar_as_uint(s1) == 0) ||
                             (is_acc_test_iv_bit(s1, ad, &ul.iv->def) &&
                              nir_scalar_is_const(s0) &&
                              nir_scalar_as_uint(s0) == 0)) {
                     /* (acc & (1 << iv)) == 0 -> true; != 0 -> false */
                     folded = true;
                     val = (alu->op == nir_op_ieq);
                  }

                  if (folded) {
                     b.cursor = nir_before_instr(instr);
                     nir_def_replace(&alu->def, nir_imm_bool(&b, val));
                     progress = true;
                     break;
                  }
               }
            }
         }
      }

      util_dynarray_fini(&loops);
      if (progress) {
         nir_progress(true, impl, nir_metadata_control_flow);
         any = true;
      } else {
         nir_no_progress(impl);
      }
   }

   if (any) {
      NIR_PASS(_, shader, nir_opt_dead_cf);
      NIR_PASS(_, shader, nir_opt_copy_prop);
      NIR_PASS(_, shader, nir_opt_dce);
   }
   return any;
}
