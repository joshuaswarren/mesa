/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"

/*
 * Loop-invariant additive splitting and integer multiply-accumulate (MAC)
 * reduction factoring for inner loops:
 *
 * 1. Additive invariant splitting:
 *    Flattens single-use in-loop 32-bit `iadd` / `imadshl_agx(..., 0)` trees
 *    into loop-invariant and loop-variant summands, hoisting the combined
 *    invariant sum to the loop preheader so loop-invariant address terms like
 *    `(y * stride + border + x) + v * stride + u` collapse to a single
 *    in-loop `iadd(row_base, u)`.
 *
 * 2. Loop-invariant MAC reduction factoring:
 *    In a single-body top-tested loop, rewrites
 *
 *      acc = acc0;
 *      for (...) { acc += inv * val; }
 *
 *    into
 *
 *      sum = 0;
 *      for (...) { sum += val; }
 *      acc = acc0 + inv * sum;
 *
 *    In modular two's-complement integer arithmetic (mod 2^B), multiplication
 *    distributes over addition exactly regardless of signedness or wraparound.
 *
 * 3. Linear-IV MAC strength reduction (when a companion `sum += val` exists):
 *    For a unit-stepped induction variable `iv` (`iv_next = iv + 1`) in the
 *    same single-body top-tested loop, rewrites
 *
 *      m = m0;
 *      for (iv = iv0; cond(iv); iv += 1) { m += iv * val; sum += val; }
 *
 *    using the exact modular prefix-sum identity:
 *      w_0 = 0, s_0 = 0;
 *      w_{i+1} = w_i + s_i, s_{i+1} = s_i + val_i;
 *      sum_{i=0}^{N-1} (iv0 + i) * val_i == (iv_exit - 1) * s_N - w_N (mod 2^32)
 *
 *    This eliminates the per-iteration multiply on `m` and replaces it with
 *    `w += s` scheduled at the top of the loop body with zero dependency on
 *    `val` (hiding memory load latency). Exact for all N >= 0, including N = 0
 *    (where s_0 = w_0 = 0).
 */

#define MAX_ADD_TERMS 16u

static bool
def_dominates_block(nir_def *def, nir_block *blk)
{
   return nir_block_dominates(nir_def_block(def), blk);
}

static bool
expr_is_pure_loop_inv(nir_def *def, nir_block *pre, struct hash_table *memo)
{
   if (def_dominates_block(def, pre))
      return true;

   struct hash_entry *he = _mesa_hash_table_search(memo, def);
   if (he)
      return he->data != NULL;

   nir_instr *instr = nir_def_instr(def);
   bool ok = false;

   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      ok = true;
      break;

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      ok = intr->intrinsic == nir_intrinsic_load_preamble;
      break;
   }

   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->def.num_components != 1)
         break;
      switch (alu->op) {
      case nir_op_iadd:
      case nir_op_isub:
      case nir_op_imul:
      case nir_op_ineg:
      case nir_op_ishl:
      case nir_op_ushr:
      case nir_op_ishr:
      case nir_op_iand:
      case nir_op_ior:
      case nir_op_ixor:
      case nir_op_imadshl_agx:
      case nir_op_imsubshl_agx:
      case nir_op_mov:
      case nir_op_u2u32:
      case nir_op_i2i32:
         ok = true;
         break;
      default:
         ok = false;
         break;
      }
      if (ok) {
         unsigned nsrc = nir_op_infos[alu->op].num_inputs;
         for (unsigned i = 0; i < nsrc; i++) {
            if (!expr_is_pure_loop_inv(alu->src[i].src.ssa, pre, memo)) {
               ok = false;
               break;
            }
         }
      }
      break;
   }

   default:
      ok = false;
      break;
   }

   _mesa_hash_table_insert(memo, def, ok ? (void *)1 : NULL);
   return ok;
}

static nir_def *
materialize_inv_in_pre(nir_builder *b_pre, nir_def *def, nir_block *pre,
                       struct hash_table *remap)
{
   if (def_dominates_block(def, pre))
      return def;

   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;

   nir_instr *instr = nir_def_instr(def);
   nir_def *out = NULL;

   if (instr->type == nir_instr_type_load_const) {
      nir_instr *clone = nir_instr_clone(b_pre->shader, instr);
      nir_builder_instr_insert(b_pre, clone);
      out = nir_instr_def(clone);
   } else if (instr->type == nir_instr_type_undef) {
      nir_instr *clone = nir_instr_clone(b_pre->shader, instr);
      nir_builder_instr_insert(b_pre, clone);
      out = nir_instr_def(clone);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_instr *clone = nir_instr_clone(b_pre->shader, instr);
      nir_builder_instr_insert(b_pre, clone);
      out = nir_instr_def(clone);
   } else if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      unsigned nsrc = nir_op_infos[alu->op].num_inputs;
      nir_def *srcs[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < nsrc; i++) {
         nir_def *base =
            materialize_inv_in_pre(b_pre, alu->src[i].src.ssa, pre, remap);
         unsigned comps = nir_ssa_alu_instr_src_components(alu, i);
         if (comps == 1) {
            srcs[i] = nir_channel(b_pre, base, alu->src[i].swizzle[0]);
         } else {
            srcs[i] = base;
         }
      }
      out = nir_build_alu_src_arr(b_pre, alu->op, srcs);
   } else {
      UNREACHABLE("unexpected invariant instr");
   }

   _mesa_hash_table_insert(remap, def, out);
   return out;
}

static bool
is_scalar_iadd32_or_imad0(const nir_instr *instr, nir_alu_instr **out_alu)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.num_components != 1 || alu->def.bit_size != 32)
      return false;
   if (alu->op == nir_op_iadd) {
      *out_alu = alu;
      return true;
   }
   if (alu->op == nir_op_imadshl_agx &&
       nir_src_is_const(alu->src[3].src) &&
       nir_src_as_uint(alu->src[3].src) == 0) {
      *out_alu = alu;
      return true;
   }
   return false;
}

struct add_term {
   bool is_prod;
   nir_def *a;
   uint8_t swiz_a;
   nir_def *b;
   uint8_t swiz_b;
   bool is_inv;
   bool is_compound_in_loop;
};

static bool
collect_add_terms(nir_def *def, uint8_t swiz, nir_block *blk, nir_block *pre,
                  struct util_dynarray *terms, bool is_root)
{
   if (util_dynarray_num_elements(terms, struct add_term) >= MAX_ADD_TERMS)
      return false;

   nir_instr *instr = nir_def_instr(def);
   nir_alu_instr *alu = NULL;

   if (swiz == 0 && !def_dominates_block(def, pre) &&
       instr->block == blk &&
       (is_root || list_is_singular(&def->uses)) &&
       is_scalar_iadd32_or_imad0(instr, &alu)) {
      if (alu->op == nir_op_iadd) {
         return collect_add_terms(alu->src[0].src.ssa, alu->src[0].swizzle[0],
                                  blk, pre, terms, false) &&
                collect_add_terms(alu->src[1].src.ssa, alu->src[1].swizzle[0],
                                  blk, pre, terms, false);
      } else {
         /* imadshl_agx(a, b, c, 0) == (a * b) + c */
         if (util_dynarray_num_elements(terms, struct add_term) + 1 >= MAX_ADD_TERMS)
            return false;
         struct add_term prod = {
            .is_prod = true,
            .a = alu->src[0].src.ssa,
            .swiz_a = alu->src[0].swizzle[0],
            .b = alu->src[1].src.ssa,
            .swiz_b = alu->src[1].swizzle[0],
            .is_compound_in_loop = true,
         };
         util_dynarray_append(terms, prod);
         return collect_add_terms(alu->src[2].src.ssa, alu->src[2].swizzle[0],
                                  blk, pre, terms, false);
      }
   }

   struct add_term t = {
      .is_prod = false,
      .a = def,
      .swiz_a = swiz,
      .is_compound_in_loop =
         !def_dominates_block(def, pre) && instr->type == nir_instr_type_alu,
   };
   util_dynarray_append(terms, t);
   return true;
}

static bool
hoist_loop_invariant_iadd_terms(nir_loop *loop)
{
   if (!nir_loop_has_back_edge(loop))
      return false;

   nir_block *pre = nir_loop_predecessor_block(loop);
   void *mem_ctx = ralloc_context(NULL);
   struct hash_table *inv_memo = _mesa_pointer_hash_table_create(mem_ctx);
   struct hash_table *remap = _mesa_pointer_hash_table_create(mem_ctx);
   bool progress = false;

   nir_builder b_pre = nir_builder_at(nir_before_cf_node(&loop->cf_node));

   nir_foreach_block_in_cf_node(blk, &loop->cf_node) {
      /* Only rewrite in blocks that directly belong to this loop (inner loops
       * were already processed bottom-up). */
      if (blk->cf_node.parent != &loop->cf_node)
         continue;

      nir_foreach_instr_safe(instr, blk) {
         nir_alu_instr *root = NULL;
         if (!is_scalar_iadd32_or_imad0(instr, &root))
            continue;

         /* Skip non-root nodes whose sole use is another additive op in `blk`. */
         if (list_is_singular(&root->def.uses)) {
            nir_src *only = list_first_entry(&root->def.uses, nir_src, use_link);
            if (!nir_src_is_if(only)) {
               nir_instr *u = nir_src_use_instr(only);
               nir_alu_instr *ualu = NULL;
               if (u->block == blk && is_scalar_iadd32_or_imad0(u, &ualu)) {
                  if (ualu->op == nir_op_iadd ||
                      only == &ualu->src[2].src)
                     continue;
               }
            }
         }

         struct util_dynarray terms;
         util_dynarray_init(&terms, mem_ctx);
         if (!collect_add_terms(&root->def, 0, blk, pre, &terms, true))
            continue;

         unsigned n = util_dynarray_num_elements(&terms, struct add_term);
         if (n < 3)
            continue;

         unsigned inv_count = 0, var_count = 0;
         bool has_compound_inv = false;
         struct add_term *arr = terms.data;
         for (unsigned i = 0; i < n; i++) {
            if (arr[i].is_prod) {
               arr[i].is_inv =
                  expr_is_pure_loop_inv(arr[i].a, pre, inv_memo) &&
                  expr_is_pure_loop_inv(arr[i].b, pre, inv_memo);
            } else {
               arr[i].is_inv =
                  expr_is_pure_loop_inv(arr[i].a, pre, inv_memo);
            }
            if (arr[i].is_inv) {
               inv_count++;
               if (arr[i].is_compound_in_loop)
                  has_compound_inv = true;
            } else {
               var_count++;
            }
         }

         /* Profitable when we collapse >=2 loop-invariant summands into one
          * preheader value (or hoist an in-loop invariant product) while at
          * least one variant summand remains in the loop. */
         if (var_count == 0 || (inv_count < 2 && !has_compound_inv))
            continue;

         /* Materialize invariant sum in loop preheader. */
         nir_def *inv_sum = NULL;
         for (unsigned i = 0; i < n; i++) {
            if (!arr[i].is_inv)
               continue;
            nir_def *val;
            if (arr[i].is_prod) {
               nir_def *ma =
                  materialize_inv_in_pre(&b_pre, arr[i].a, pre, remap);
               nir_def *mb =
                  materialize_inv_in_pre(&b_pre, arr[i].b, pre, remap);
               val = nir_imul(&b_pre, nir_channel(&b_pre, ma, arr[i].swiz_a),
                              nir_channel(&b_pre, mb, arr[i].swiz_b));
            } else {
               nir_def *ma =
                  materialize_inv_in_pre(&b_pre, arr[i].a, pre, remap);
               val = nir_channel(&b_pre, ma, arr[i].swiz_a);
            }
            inv_sum = inv_sum ? nir_iadd(&b_pre, inv_sum, val) : val;
         }

         /* Rebuild variant terms + inv_sum at the root instruction. */
         nir_builder b = nir_builder_at(nir_before_instr(&root->instr));
         nir_def *cur = inv_sum;
         for (unsigned i = 0; i < n; i++) {
            if (arr[i].is_inv)
               continue;
            nir_def *val;
            if (arr[i].is_prod) {
               val = nir_imul(&b, nir_channel(&b, arr[i].a, arr[i].swiz_a),
                              nir_channel(&b, arr[i].b, arr[i].swiz_b));
            } else {
               val = nir_channel(&b, arr[i].a, arr[i].swiz_a);
            }
            cur = nir_iadd(&b, cur, val);
         }

         nir_def_replace(&root->def, cur);
         progress = true;
      }
   }

   ralloc_free(mem_ctx);
   return progress;
}

static bool
block_in_loop(nir_block *blk, nir_loop *loop)
{
   for (nir_cf_node *n = blk->cf_node.parent; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }
   return false;
}

static bool
block_has_pred(nir_block *blk, nir_block *pred)
{
   util_dynarray_foreach(&blk->predecessors, nir_block *, p) {
      if (*p == pred)
         return true;
   }
   return false;
}

/* Verify that `loop` is a canonical single-body top-tested loop:
 *   header -> if (exit) { break } else { empty } -> straight_line_body
 * with a single exit block dominated by header. */
static bool
match_simple_top_tested_loop(nir_loop *loop, nir_block **out_hdr,
                             nir_block **out_body, nir_block **out_exit)
{
   if (!nir_loop_has_back_edge(loop))
      return false;

   struct exec_node *n0 = exec_list_get_head_raw(&loop->body);
   if (exec_node_is_tail_sentinel(n0))
      return false;
   struct exec_node *n1 = n0->next;
   if (exec_node_is_tail_sentinel(n1))
      return false;
   struct exec_node *n2 = n1->next;
   if (exec_node_is_tail_sentinel(n2) || !exec_node_is_tail_sentinel(n2->next))
      return false;

   nir_cf_node *c0 = exec_node_data(nir_cf_node, n0, node);
   nir_cf_node *c1 = exec_node_data(nir_cf_node, n1, node);
   nir_cf_node *c2 = exec_node_data(nir_cf_node, n2, node);

   if (c0->type != nir_cf_node_block || c1->type != nir_cf_node_if ||
       c2->type != nir_cf_node_block)
      return false;

   nir_block *hdr = nir_cf_node_as_block(c0);
   nir_if *brk_if = nir_cf_node_as_if(c1);
   nir_block *body = nir_cf_node_as_block(c2);

   if (nir_block_ends_in_jump(hdr) || nir_block_ends_in_jump(body))
      return false;

   /* Both arms of brk_if must be single blocks; one ends in break (with no
    * other instructions) and the other is completely empty. */
   nir_cf_node *t0 = exec_node_data(nir_cf_node, exec_list_get_head(&brk_if->then_list), node);
   nir_cf_node *t_last = exec_node_data(nir_cf_node, exec_list_get_tail(&brk_if->then_list), node);
   nir_cf_node *e0 = exec_node_data(nir_cf_node, exec_list_get_head(&brk_if->else_list), node);
   nir_cf_node *e_last = exec_node_data(nir_cf_node, exec_list_get_tail(&brk_if->else_list), node);
   if (t0 != t_last || e0 != e_last ||
       t0->type != nir_cf_node_block || e0->type != nir_cf_node_block)
      return false;

   nir_block *tb = nir_cf_node_as_block(t0);
   nir_block *eb = nir_cf_node_as_block(e0);

   nir_block *brk_blk = NULL;
   nir_block *cont_blk = NULL;
   if (nir_block_ends_in_break(tb) && exec_list_is_empty(&eb->instr_list)) {
      brk_blk = tb;
      cont_blk = eb;
   } else if (nir_block_ends_in_break(eb) && exec_list_is_empty(&tb->instr_list)) {
      brk_blk = eb;
      cont_blk = tb;
   } else {
      return false;
   }
   (void)cont_blk;

   /* Break block must contain only the break jump itself. */
   if (exec_list_get_head(&brk_blk->instr_list) !=
       exec_list_get_tail(&brk_blk->instr_list))
      return false;

   nir_block *exit_blk = nir_loop_successor_block(loop);
   if (util_dynarray_num_elements(&exit_blk->predecessors, nir_block *) != 1 ||
       !block_has_pred(exit_blk, brk_blk))
      return false;

   /* Header must have exactly 2 predecessors: preheader and body. */
   nir_block *pre = nir_loop_predecessor_block(loop);
   if (util_dynarray_num_elements(&hdr->predecessors, nir_block *) != 2 ||
       !block_has_pred(hdr, pre) || !block_has_pred(hdr, body))
      return false;

   *out_hdr = hdr;
   *out_body = body;
   *out_exit = exit_blk;
   return true;
}

/* Match `acc_next = acc_phi + a * b` (either `iadd(acc_phi, imul(a, b))` or
 * `imadshl_agx(a, b, acc_phi, 0)`). */
static bool
match_mac_update(nir_def *acc_next, nir_def *acc_phi, nir_block *body,
                 nir_def **out_a, nir_def **out_b)
{
   nir_instr *instr = nir_def_instr(acc_next);
   if (instr->block != body || instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.num_components != 1 ||
       (alu->def.bit_size != 16 && alu->def.bit_size != 32))
      return false;

   if (alu->op == nir_op_imadshl_agx &&
       nir_src_is_const(alu->src[3].src) &&
       nir_src_as_uint(alu->src[3].src) == 0 &&
       alu->src[2].src.ssa == acc_phi &&
       alu->src[2].swizzle[0] == 0 &&
       alu->src[0].swizzle[0] == 0 &&
       alu->src[1].swizzle[0] == 0 &&
       alu->src[0].src.ssa->num_components == 1 &&
       alu->src[1].src.ssa->num_components == 1) {
      *out_a = alu->src[0].src.ssa;
      *out_b = alu->src[1].src.ssa;
      return true;
   }

   if (alu->op == nir_op_iadd &&
       alu->src[0].swizzle[0] == 0 && alu->src[1].swizzle[0] == 0) {
      nir_def *other = NULL;
      if (alu->src[0].src.ssa == acc_phi)
         other = alu->src[1].src.ssa;
      else if (alu->src[1].src.ssa == acc_phi)
         other = alu->src[0].src.ssa;
      else
         return false;

      nir_instr *oi = nir_def_instr(other);
      if (oi->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *mul = nir_instr_as_alu(oi);
      if (mul->op != nir_op_imul || mul->def.num_components != 1 ||
          mul->src[0].swizzle[0] != 0 || mul->src[1].swizzle[0] != 0 ||
          mul->src[0].src.ssa->num_components != 1 ||
          mul->src[1].src.ssa->num_components != 1)
         return false;
      *out_a = mul->src[0].src.ssa;
      *out_b = mul->src[1].src.ssa;
      return true;
   }

   return false;
}

static bool
phi_only_used_by_update_and_outside(nir_phi_instr *phi, nir_instr *update,
                                    nir_loop *loop, nir_block *exit_blk)
{
   bool saw_update = false;
   nir_foreach_use_including_if(src, &phi->def) {
      if (nir_src_is_if(src))
         return false;
      nir_instr *u = nir_src_use_instr(src);
      if (u == update) {
         if (saw_update)
            return false;
         saw_update = true;
      } else if (block_in_loop(u->block, loop)) {
         return false;
      } else if (u->type == nir_instr_type_phi && u->block == exit_blk) {
         return false;
      }
   }
   return saw_update;
}

static bool
def_only_used_by_phi(nir_def *def, nir_phi_instr *phi)
{
   bool saw = false;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src) || nir_src_use_instr(src) != &phi->instr)
         return false;
      saw = true;
   }
   return saw;
}

static bool
is_unit_step_iv(nir_phi_instr *phi, nir_block *body)
{
   if (phi->def.num_components != 1 || phi->def.bit_size != 32)
      return false;
   nir_phi_src *ls = nir_phi_get_src_from_block(phi, body);
   if (!ls)
      return false;
   nir_instr *ni = nir_def_instr(ls->src.ssa);
   if (ni->block != body || ni->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(ni);
   if (alu->op != nir_op_iadd || alu->def.num_components != 1)
      return false;

   for (unsigned i = 0; i < 2; i++) {
      if (alu->src[i].src.ssa == &phi->def && alu->src[i].swizzle[0] == 0 &&
          nir_src_is_const(alu->src[1 - i].src) &&
          nir_src_comp_as_int(alu->src[1 - i].src, alu->src[1 - i].swizzle[0]) == 1)
         return true;
   }
   return false;
}

static bool
is_trivial_const_factor(nir_def *def)
{
   nir_instr *i = nir_def_instr(def);
   if (i->type != nir_instr_type_load_const)
      return false;
   int64_t v = nir_src_as_int(nir_src_for_ssa(def));
   return v == 0 || v == 1;
}

struct sum_rec {
   nir_def *val;
   nir_phi_instr *sum_phi;
};

static nir_phi_instr *
create_zero_init_phi(nir_shader *shader, nir_block *pre, nir_block *hdr,
                     unsigned bit_size)
{
   nir_builder b_pre = nir_builder_at(nir_after_block_before_jump(pre));
   nir_def *zero = nir_imm_intN_t(&b_pre, 0, bit_size);

   nir_phi_instr *phi = nir_phi_instr_create(shader);
   nir_def_init(&phi->instr, &phi->def, 1, bit_size);
   nir_phi_instr_add_src(phi, pre, zero);
   nir_instr_insert(nir_before_block(hdr), &phi->instr);
   return phi;
}

static void
add_phi_latch_src(nir_phi_instr *phi, nir_block *body, nir_def *val)
{
   nir_phi_src *ps = nir_phi_instr_add_src(phi, body, val);
   list_addtail(&ps->src.use_link, &val->uses);
}

static bool
factor_loop_mac_reductions(nir_shader *shader, nir_loop *loop)
{
   nir_block *hdr = NULL, *body = NULL, *exit_blk = NULL;
   if (!match_simple_top_tested_loop(loop, &hdr, &body, &exit_blk))
      return false;

   nir_block *pre = nir_loop_predecessor_block(loop);
   bool progress = false;

   struct util_dynarray sums;
   util_dynarray_init(&sums, NULL);

   /* Record any pre-existing zero-initialized `sum += val` reductions. */
   nir_foreach_phi(phi, hdr) {
      if (phi->def.num_components != 1 || phi->def.bit_size != 32)
         continue;
      nir_phi_src *ps_pre = nir_phi_get_src_from_block(phi, pre);
      nir_phi_src *ps_body = nir_phi_get_src_from_block(phi, body);
      if (!ps_pre || !ps_body)
         continue;
      if (nir_def_instr(ps_pre->src.ssa)->type != nir_instr_type_load_const ||
          nir_src_as_uint(ps_pre->src) != 0)
         continue;
      nir_instr *ni = nir_def_instr(ps_body->src.ssa);
      if (ni->block != body || ni->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(ni);
      if (alu->op != nir_op_iadd || alu->def.num_components != 1)
         continue;
      nir_def *val = NULL;
      if (alu->src[0].src.ssa == &phi->def && alu->src[1].swizzle[0] == 0)
         val = alu->src[1].src.ssa;
      else if (alu->src[1].src.ssa == &phi->def && alu->src[0].swizzle[0] == 0)
         val = alu->src[0].src.ssa;
      if (val && val->num_components == 1) {
         struct sum_rec rec = { .val = val, .sum_phi = phi };
         util_dynarray_append(&sums, rec);
      }
   }

   /* Step 1: Factor loop-invariant MAC reductions `acc += inv * val`. */
   nir_foreach_phi_safe(acc_phi, hdr) {
      if (acc_phi->def.num_components != 1 ||
          (acc_phi->def.bit_size != 16 && acc_phi->def.bit_size != 32))
         continue;

      nir_phi_src *ps_pre = nir_phi_get_src_from_block(acc_phi, pre);
      nir_phi_src *ps_body = nir_phi_get_src_from_block(acc_phi, body);
      if (!ps_pre || !ps_body)
         continue;

      nir_def *acc_init = ps_pre->src.ssa;
      nir_def *acc_next = ps_body->src.ssa;

      nir_def *fa = NULL, *fb = NULL;
      if (!match_mac_update(acc_next, &acc_phi->def, body, &fa, &fb))
         continue;
      if (!def_only_used_by_phi(acc_next, acc_phi) ||
          !phi_only_used_by_update_and_outside(acc_phi, nir_def_instr(acc_next),
                                               loop, exit_blk))
         continue;

      bool a_inv = def_dominates_block(fa, pre);
      bool b_inv = def_dominates_block(fb, pre);
      if (a_inv == b_inv)
         continue;

      nir_def *inv = a_inv ? fa : fb;
      nir_def *val = a_inv ? fb : fa;
      if (is_trivial_const_factor(inv))
         continue;

      /* Reuse an existing sum_phi for `val` if available, else create one. */
      nir_phi_instr *sum_phi = NULL;
      util_dynarray_foreach(&sums, struct sum_rec, rec) {
         if (rec->val == val && rec->sum_phi->def.bit_size == acc_phi->def.bit_size) {
            sum_phi = rec->sum_phi;
            break;
         }
      }
      if (!sum_phi) {
         sum_phi = create_zero_init_phi(shader, pre, hdr, acc_phi->def.bit_size);
         nir_builder b_upd = nir_builder_at(nir_before_instr(nir_def_instr(acc_next)));
         nir_def *sum_next = nir_iadd(&b_upd, &sum_phi->def, val);
         add_phi_latch_src(sum_phi, body, sum_next);
         struct sum_rec rec = { .val = val, .sum_phi = sum_phi };
         util_dynarray_append(&sums, rec);
      }

      nir_builder b_exit = nir_builder_at(nir_before_block_after_phis(exit_blk));
      nir_def *acc_out =
         nir_iadd(&b_exit, acc_init, nir_imul(&b_exit, inv, &sum_phi->def));

      nir_foreach_use_including_if_safe(u, &acc_phi->def) {
         if (!nir_src_is_if(u) &&
             nir_src_use_instr(u) == nir_def_instr(acc_next))
            continue;
         nir_src_rewrite(u, acc_out);
      }

      nir_instr_remove(nir_def_instr(acc_next));
      nir_instr_remove(&acc_phi->instr);
      progress = true;
   }

   /* Step 2: Strength-reduce unit-stepped linear-IV MAC reductions
    * `m += iv * val` when a companion `sum += val` exists in `sums`. */
   if (util_dynarray_num_elements(&sums, struct sum_rec) > 0) {
      nir_foreach_phi_safe(m_phi, hdr) {
         if (m_phi->def.num_components != 1 || m_phi->def.bit_size != 32)
            continue;

         nir_phi_src *ps_pre = nir_phi_get_src_from_block(m_phi, pre);
         nir_phi_src *ps_body = nir_phi_get_src_from_block(m_phi, body);
         if (!ps_pre || !ps_body)
            continue;

         nir_def *m_init = ps_pre->src.ssa;
         nir_def *m_next = ps_body->src.ssa;

         nir_def *fa = NULL, *fb = NULL;
         if (!match_mac_update(m_next, &m_phi->def, body, &fa, &fb))
            continue;
         if (!def_only_used_by_phi(m_next, m_phi) ||
             !phi_only_used_by_update_and_outside(m_phi, nir_def_instr(m_next),
                                                  loop, exit_blk))
            continue;

         nir_phi_instr *iv_phi = NULL;
         nir_def *val = NULL;
         if (nir_def_instr(fa)->type == nir_instr_type_phi &&
             nir_def_block(fa) == hdr &&
             is_unit_step_iv(nir_instr_as_phi(nir_def_instr(fa)), body)) {
            iv_phi = nir_instr_as_phi(nir_def_instr(fa));
            val = fb;
         } else if (nir_def_instr(fb)->type == nir_instr_type_phi &&
                    nir_def_block(fb) == hdr &&
                    is_unit_step_iv(nir_instr_as_phi(nir_def_instr(fb)), body)) {
            iv_phi = nir_instr_as_phi(nir_def_instr(fb));
            val = fa;
         }
         if (!iv_phi || !val)
            continue;

         nir_phi_instr *sum_phi = NULL;
         util_dynarray_foreach(&sums, struct sum_rec, rec) {
            if (rec->val == val && rec->sum_phi->def.bit_size == 32) {
               sum_phi = rec->sum_phi;
               break;
            }
         }
         if (!sum_phi)
            continue;

         /* Add prefix accumulator w += sum_phi at the TOP of body (before the
          * load of `val`), then reconstruct:
          *   m_out = m_init + ((iv_exit - 1) * sum_exit - w_exit). */
         nir_phi_instr *w_phi = create_zero_init_phi(shader, pre, hdr, 32);
         nir_builder b_top = nir_builder_at(nir_before_block(body));
         nir_def *w_next = nir_iadd(&b_top, &w_phi->def, &sum_phi->def);
         add_phi_latch_src(w_phi, body, w_next);

         nir_builder b_exit =
            nir_builder_at(nir_before_block_after_phis(exit_blk));
         nir_def *last_iv = nir_iadd_imm(&b_exit, &iv_phi->def, -1);
         nir_def *prod = nir_imul(&b_exit, last_iv, &sum_phi->def);
         nir_def *m_out =
            nir_isub(&b_exit, nir_iadd(&b_exit, m_init, prod), &w_phi->def);

         nir_foreach_use_including_if_safe(u, &m_phi->def) {
            if (!nir_src_is_if(u) &&
                nir_src_use_instr(u) == nir_def_instr(m_next))
               continue;
            nir_src_rewrite(u, m_out);
         }

         nir_instr_remove(nir_def_instr(m_next));
         nir_instr_remove(&m_phi->instr);
         progress = true;
      }
   }

   util_dynarray_fini(&sums);
   return progress;
}

static bool
process_cf_list(nir_shader *shader, struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= process_cf_list(shader, &nif->then_list);
         progress |= process_cf_list(shader, &nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         /* Process inner loops first so hoisted invariant address terms
          * cascade cleanly out to enclosing loops. */
         progress |= process_cf_list(shader, &loop->body);
         progress |= hoist_loop_invariant_iadd_terms(loop);
         progress |= factor_loop_mac_reductions(shader, loop);
         break;
      }
      default:
         break;
      }
   }
   return progress;
}

bool
agx_nir_opt_factor_loop_mac(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_dominance);
      if (process_cf_list(shader, &impl->body)) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      } else {
         nir_no_progress(impl);
      }
   }

   return progress;
}
