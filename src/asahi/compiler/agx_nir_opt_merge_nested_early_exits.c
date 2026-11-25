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
 * Merge nested early-exit branches that execute identical side-effecting
 * fallback arms, fold redundant range-bound disjunctions exposed by the
 * combined condition, and hoist loop-invariant pure disjunction components
 * to the loop preheader:
 *
 *   if (A) {
 *      S;
 *   } else {
 *      pure_prelude;
 *      if (B) {
 *         S;
 *      } else {
 *         body;
 *      }
 *   }
 *
 * becomes:
 *
 *   pure_prelude;
 *   if (A || B) {
 *      S;
 *   } else {
 *      body;
 *   }
 *
 * Eliminating the extra control-flow nesting level removes a duplicate exit
 * store, cuts exec-mask push/else/pop overhead on every invocation, and
 * exposes loop-invariant and redundant range comparisons in `B` to preamble
 * hoisting, range-bound folding, and preheader hoisting.
 */

#define MAX_EXIT_ARM_INSTRS 8u
#define MAX_PRELUDE_INSTRS  32u
#define MAX_OR_LEAVES       32u

static bool
alu_is_pure_trap_free(const nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_idiv:
   case nir_op_udiv:
   case nir_op_imod:
   case nir_op_umod:
   case nir_op_irem:
      return false;
   default:
      return true;
   }
}

static bool
instr_allowed_in_prelude(const nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;

   case nir_instr_type_alu:
      return alu_is_pure_trap_free(nir_instr_as_alu(instr));

   case nir_instr_type_intrinsic: {
      const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_push_constant:
      case nir_intrinsic_load_sysval_agx:
         return true;
      case nir_intrinsic_load_constant_agx:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_ubo: {
         if (!nir_intrinsic_has_access(intr))
            return false;
         enum gl_access_qualifier acc = nir_intrinsic_access(intr);
         return (acc & ACCESS_CAN_SPECULATE) &&
                !(acc & (ACCESS_VOLATILE | ACCESS_COHERENT));
      }
      default:
         return false;
      }
   }

   default:
      return false;
   }
}

static bool
instr_allowed_in_exit_arm(const nir_instr *instr)
{
   if (instr_allowed_in_prelude(instr))
      return true;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_agx:
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared: {
      if (nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_COHERENT)))
         return false;
      return true;
   }
   default:
      return false;
   }
}

static bool
def_used_outside_block(const nir_def *def, const nir_block *block)
{
   nir_foreach_use_including_if(src, def) {
      if (nir_src_get_block(src) != block)
         return true;
   }
   return false;
}

static bool
ssa_matches(const nir_def *a, const nir_def *b, struct hash_table *local_map)
{
   if (a == b)
      return true;
   struct hash_entry *he = _mesa_hash_table_search(local_map, a);
   return he && he->data == b;
}

static bool
instrs_equivalent(const nir_instr *ia, const nir_instr *ib,
                  struct hash_table *local_map)
{
   if (ia->type != ib->type)
      return false;

   switch (ia->type) {
   case nir_instr_type_load_const: {
      const nir_load_const_instr *ca = nir_instr_as_load_const(ia);
      const nir_load_const_instr *cb = nir_instr_as_load_const(ib);
      if (ca->def.num_components != cb->def.num_components ||
          ca->def.bit_size != cb->def.bit_size)
         return false;
      for (unsigned i = 0; i < ca->def.num_components; i++) {
         if (ca->value[i].u64 != cb->value[i].u64)
            return false;
      }
      return true;
   }

   case nir_instr_type_undef: {
      const nir_undef_instr *ua = nir_instr_as_undef(ia);
      const nir_undef_instr *ub = nir_instr_as_undef(ib);
      return ua->def.num_components == ub->def.num_components &&
             ua->def.bit_size == ub->def.bit_size;
   }

   case nir_instr_type_alu: {
      const nir_alu_instr *aa = nir_instr_as_alu(ia);
      const nir_alu_instr *ab = nir_instr_as_alu(ib);
      if (aa->op != ab->op ||
          aa->def.num_components != ab->def.num_components ||
          aa->def.bit_size != ab->def.bit_size ||
          aa->fp_math_ctrl != ab->fp_math_ctrl)
         return false;
      unsigned nsrc = nir_op_infos[aa->op].num_inputs;
      for (unsigned i = 0; i < nsrc; i++) {
         if (!ssa_matches(aa->src[i].src.ssa, ab->src[i].src.ssa, local_map))
            return false;
         unsigned comps = nir_ssa_alu_instr_src_components(aa, i);
         for (unsigned c = 0; c < comps; c++) {
            if (aa->src[i].swizzle[c] != ab->src[i].swizzle[c])
               return false;
         }
      }
      return true;
   }

   case nir_instr_type_intrinsic: {
      const nir_intrinsic_instr *na = nir_instr_as_intrinsic(ia);
      const nir_intrinsic_instr *nb = nir_instr_as_intrinsic(ib);
      if (na->intrinsic != nb->intrinsic ||
          na->num_components != nb->num_components)
         return false;
      const nir_intrinsic_info *info = &nir_intrinsic_infos[na->intrinsic];
      if (info->has_dest &&
          (na->def.num_components != nb->def.num_components ||
           na->def.bit_size != nb->def.bit_size))
         return false;
      if (memcmp(na->const_index, nb->const_index, sizeof(na->const_index)) != 0)
         return false;
      for (unsigned i = 0; i < info->num_srcs; i++) {
         if (!ssa_matches(na->src[i].ssa, nb->src[i].ssa, local_map))
            return false;
      }
      return true;
   }

   default:
      return false;
   }
}

static bool
blocks_structurally_identical(nir_block *blk_a, nir_block *blk_b)
{
   struct exec_node *na = exec_list_get_head_raw(&blk_a->instr_list);
   struct exec_node *nb = exec_list_get_head_raw(&blk_b->instr_list);

   /* Require at least one instruction (empty arms are handled by nir_opt_if). */
   if (exec_node_is_tail_sentinel(na) || exec_node_is_tail_sentinel(nb))
      return false;

   void *mem_ctx = ralloc_context(NULL);
   struct hash_table *local_map = _mesa_pointer_hash_table_create(mem_ctx);
   unsigned count = 0;
   bool match = true;

   while (!exec_node_is_tail_sentinel(na) && !exec_node_is_tail_sentinel(nb)) {
      nir_instr *ia = exec_node_data(nir_instr, na, node);
      nir_instr *ib = exec_node_data(nir_instr, nb, node);

      if (++count > MAX_EXIT_ARM_INSTRS ||
          !instr_allowed_in_exit_arm(ia) ||
          !instr_allowed_in_exit_arm(ib) ||
          !instrs_equivalent(ia, ib, local_map)) {
         match = false;
         break;
      }

      nir_def *da = nir_instr_def(ia);
      nir_def *db = nir_instr_def(ib);
      if (da) {
         if (!db || def_used_outside_block(da, blk_a) ||
             def_used_outside_block(db, blk_b)) {
            match = false;
            break;
         }
         _mesa_hash_table_insert(local_map, da, db);
      }

      na = na->next;
      nb = nb->next;
   }

   if (!exec_node_is_tail_sentinel(na) || !exec_node_is_tail_sentinel(nb))
      match = false;

   ralloc_free(mem_ctx);
   return match;
}

static bool
cf_list_is_single_block(struct exec_list *list, nir_block **out_blk)
{
   if (exec_list_is_empty(list))
      return false;
   nir_cf_node *first = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   nir_cf_node *last = exec_node_data(nir_cf_node, exec_list_get_tail(list), node);
   if (first != last || first->type != nir_cf_node_block)
      return false;
   *out_blk = nir_cf_node_as_block(first);
   return true;
}

static bool
cf_list_is_pre_if_post(struct exec_list *list, nir_block **out_pre,
                       nir_if **out_if, nir_block **out_post)
{
   struct exec_node *n0 = exec_list_get_head_raw(list);
   if (exec_node_is_tail_sentinel(n0))
      return false;
   struct exec_node *n1 = n0->next;
   if (exec_node_is_tail_sentinel(n1))
      return false;
   struct exec_node *n2 = n1->next;
   if (exec_node_is_tail_sentinel(n2))
      return false;
   if (!exec_node_is_tail_sentinel(n2->next))
      return false;

   nir_cf_node *c0 = exec_node_data(nir_cf_node, n0, node);
   nir_cf_node *c1 = exec_node_data(nir_cf_node, n1, node);
   nir_cf_node *c2 = exec_node_data(nir_cf_node, n2, node);

   if (c0->type != nir_cf_node_block || c1->type != nir_cf_node_if ||
       c2->type != nir_cf_node_block)
      return false;

   *out_pre = nir_cf_node_as_block(c0);
   *out_if = nir_cf_node_as_if(c1);
   *out_post = nir_cf_node_as_block(c2);
   return true;
}

static bool
block_is_pure_prelude(nir_block *blk)
{
   unsigned count = 0;
   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_phi)
         return false;
      if (++count > MAX_PRELUDE_INSTRS || !instr_allowed_in_prelude(instr))
         return false;
   }
   return true;
}

static bool
block_has_only_phis(nir_block *blk)
{
   nir_foreach_instr(instr, blk) {
      if (instr->type != nir_instr_type_phi)
         return false;
   }
   return true;
}

struct phi_rewire {
   nir_phi_instr *outer_phi;
   nir_def *new_other_val;
};

static bool
try_merge_outer_if(nir_if *outer_if, bool arm_a_is_then)
{
   struct exec_list *arm_a_list =
      arm_a_is_then ? &outer_if->then_list : &outer_if->else_list;
   struct exec_list *other_list =
      arm_a_is_then ? &outer_if->else_list : &outer_if->then_list;

   nir_block *arm_a_blk = NULL;
   if (!cf_list_is_single_block(arm_a_list, &arm_a_blk))
      return false;

   nir_block *pre_blk = NULL;
   nir_if *inner_if = NULL;
   nir_block *post_blk = NULL;
   if (!cf_list_is_pre_if_post(other_list, &pre_blk, &inner_if, &post_blk))
      return false;

   if (!block_is_pure_prelude(pre_blk) || !block_has_only_phis(post_blk))
      return false;

   /* Find which arm of inner_if matches arm_a_blk. */
   bool arm_b_is_then = false;
   nir_block *arm_b_blk = NULL;
   for (unsigned pol = 0; pol < 2; pol++) {
      bool try_then = (pol == 0);
      struct exec_list *cand_list =
         try_then ? &inner_if->then_list : &inner_if->else_list;
      nir_block *cand_blk = NULL;
      if (cf_list_is_single_block(cand_list, &cand_blk) &&
          blocks_structurally_identical(arm_a_blk, cand_blk)) {
         arm_b_is_then = try_then;
         arm_b_blk = cand_blk;
         break;
      }
   }
   if (!arm_b_blk)
      return false;

   nir_block *outer_merge_blk =
      nir_cf_node_as_block(nir_cf_node_next(&outer_if->cf_node));

   /* Every phi in post_blk may only feed phis in outer_merge_blk. */
   nir_foreach_phi(inner_phi, post_blk) {
      nir_foreach_use_including_if(src, &inner_phi->def) {
         if (nir_src_is_if(src))
            return false;
         nir_instr *use = nir_src_use_instr(src);
         if (use->type != nir_instr_type_phi || use->block != outer_merge_blk)
            return false;
      }
   }

   /* Verify that every phi in outer_merge_blk agrees on arm_a and arm_b. */
   struct util_dynarray rewires;
   util_dynarray_init(&rewires, NULL);

   nir_block *inner_body_end = arm_b_is_then
                                  ? nir_if_last_else_block(inner_if)
                                  : nir_if_last_then_block(inner_if);

   bool phis_ok = true;
   nir_foreach_phi(outer_phi, outer_merge_blk) {
      nir_phi_src *src_a = nir_phi_get_src_from_block(outer_phi, arm_a_blk);
      nir_phi_src *src_post = nir_phi_get_src_from_block(outer_phi, post_blk);
      if (!src_a || !src_post) {
         phis_ok = false;
         break;
      }

      nir_def *val_a = src_a->src.ssa;
      nir_def *val_post = src_post->src.ssa;
      nir_def *new_other_val = NULL;

      if (val_a == val_post) {
         new_other_val = val_a;
      } else if (nir_def_instr(val_post)->type == nir_instr_type_phi &&
                 nir_def_block(val_post) == post_blk) {
         nir_phi_instr *inner_phi = nir_instr_as_phi(nir_def_instr(val_post));
         nir_phi_src *isrc_b = nir_phi_get_src_from_block(inner_phi, arm_b_blk);
         nir_phi_src *isrc_body =
            nir_phi_get_src_from_block(inner_phi, inner_body_end);
         if (!isrc_b || !isrc_body || isrc_b->src.ssa != val_a) {
            phis_ok = false;
            break;
         }
         new_other_val = isrc_body->src.ssa;
      } else {
         phis_ok = false;
         break;
      }

      struct phi_rewire rw = {
         .outer_phi = outer_phi,
         .new_other_val = new_other_val,
      };
      util_dynarray_append(&rewires, rw);
   }

   if (!phis_ok) {
      util_dynarray_fini(&rewires);
      return false;
   }

   /* 1. Move pure prelude instructions before outer_if. */
   nir_cursor before_outer = nir_before_cf_node(&outer_if->cf_node);
   nir_foreach_instr_safe(instr, pre_blk) {
      nir_instr_move(before_outer, instr);
   }

   /* 2. Build merged condition before outer_if. */
   nir_builder b = nir_builder_at(nir_before_cf_node(&outer_if->cf_node));
   nir_def *outer_cond = outer_if->condition.ssa;
   nir_def *inner_cond = inner_if->condition.ssa;
   nir_def *exit_a = arm_a_is_then ? outer_cond : nir_inot(&b, outer_cond);
   nir_def *exit_b = arm_b_is_then ? inner_cond : nir_inot(&b, inner_cond);
   nir_def *merged_exit = nir_ior(&b, exit_a, exit_b);
   nir_def *new_outer_cond =
      arm_a_is_then ? merged_exit : nir_inot(&b, merged_exit);
   nir_src_rewrite(&outer_if->condition, new_outer_cond);

   /* 3. Rewire outer_merge_blk phis off post_blk before modifying CF. */
   util_dynarray_foreach(&rewires, struct phi_rewire, rw) {
      nir_phi_src *src_post =
         nir_phi_get_src_from_block(rw->outer_phi, post_blk);
      nir_src_rewrite(&src_post->src, rw->new_other_val);
   }
   util_dynarray_fini(&rewires);

   /* 4. Remove inner_if's dead phis in post_blk, extract inner_body, and
    *    replace inner_if with inner_body inside other_list. */
   nir_foreach_phi_safe(inner_phi, post_blk) {
      nir_instr_remove(&inner_phi->instr);
   }

   struct exec_list *inner_body_list =
      arm_b_is_then ? &inner_if->else_list : &inner_if->then_list;
   nir_cf_list body_cf;
   nir_cf_extract(&body_cf, nir_before_cf_list(inner_body_list),
                  nir_after_cf_list(inner_body_list));

   nir_cf_node_remove(&inner_if->cf_node);
   nir_cf_reinsert(&body_cf, nir_after_cf_list(other_list));

   /* Clean up any redundant outer phis whose arms are now identical. */
   nir_foreach_phi_safe(outer_phi, outer_merge_blk) {
      nir_def *first = NULL;
      bool all_same = true;
      nir_foreach_phi_src(ps, outer_phi) {
         if (!first)
            first = ps->src.ssa;
         else if (ps->src.ssa != first)
            all_same = false;
      }
      if (all_same && first) {
         nir_def_replace(&outer_phi->def, first);
      }
   }

   return true;
}

static bool
merge_nested_early_exits_in_list(struct exec_list *cf_list)
{
   bool progress = false;

   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= merge_nested_early_exits_in_list(&nif->then_list);
         progress |= merge_nested_early_exits_in_list(&nif->else_list);

         bool merged;
         do {
            merged = try_merge_outer_if(nif, true) ||
                     try_merge_outer_if(nif, false);
            progress |= merged;
         } while (merged);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= merge_nested_early_exits_in_list(&loop->body);
         break;
      }
      default:
         break;
      }
   }

   return progress;
}

/*
 * Flatten a 1-bit boolean disjunction (pushing `inot` through `iand` via
 * De Morgan), fold redundant range comparisons on the same scalar operand:
 *
 *   (v < a)  || (v < b)  -> v < max(a, b)
 *   (a < v)  || (b < v)  -> min(a, b) < v
 *   (v >= a) || (v >= b) -> v >= min(a, b)
 *   (a >= v) || (b >= v) -> max(a, b) >= v
 *   (v <u a) || (v <u b) -> v <u umax(a, b)
 *   (a <u v) || (b <u v) -> umin(a, b) <u v
 *   (v >=u a)|| (v >=u b)-> v >=u umin(a, b)
 *   (a >=u v)|| (b >=u v)-> umax(a, b) >=u v
 *
 * Also normalize exact non-wrapping constant inclusive bounds `!(#c < v)` ->
 * `v < #c + 1` (when `#c < INT_MAX`) so pairs like `!(2 < x) || (x < border)`
 * fold into `x < max(3, border)`.
 *
 * Finally, hoist pure loop-invariant leaves directly to the loop preheader as
 * a single 1-bit value (costing at most 1 half-register) and put each
 * remaining loop-variant comparison on `src[0]` of `nir_ior(cmp, acc)` so
 * AGX select fusion emits one `icmpsel` per comparison.
 */

enum cmp_kind {
   CMP_NONE = 0,
   CMP_ILT, /* var < bound  -> combine bounds with imax */
   CMP_IGT, /* bound < var  -> combine bounds with imin */
   CMP_IGE, /* var >= bound -> combine bounds with imin */
   CMP_ILE, /* bound >= var -> combine bounds with imax */
   CMP_ULT, /* var <u bound -> combine bounds with umax */
   CMP_UGT, /* bound <u var -> combine bounds with umin */
   CMP_UGE, /* var >=u bound-> combine bounds with umin */
   CMP_ULE, /* bound >=u var-> combine bounds with umax */
};

struct or_leaf {
   nir_def *def;
   bool negated;
   enum cmp_kind kind;
   nir_scalar var;
   nir_scalar bound;
   bool bound_is_const;
   int64_t bound_const;
   bool is_loop_inv;
};

static bool
collect_or_leaves(nir_def *def, bool negated, struct util_dynarray *leaves)
{
   if (util_dynarray_num_elements(leaves, struct or_leaf) >= MAX_OR_LEAVES)
      return false;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_inot && alu->def.num_components == 1)
         return collect_or_leaves(alu->src[0].src.ssa, !negated, leaves);

      if (!negated && alu->op == nir_op_ior && alu->def.num_components == 1) {
         return collect_or_leaves(alu->src[0].src.ssa, false, leaves) &&
                collect_or_leaves(alu->src[1].src.ssa, false, leaves);
      }

      if (negated && alu->op == nir_op_iand && alu->def.num_components == 1) {
         return collect_or_leaves(alu->src[0].src.ssa, true, leaves) &&
                collect_or_leaves(alu->src[1].src.ssa, true, leaves);
      }

      if (alu->op == nir_op_bcsel && alu->def.num_components == 1 &&
          alu->def.bit_size == 1) {
         nir_scalar s1 =
            nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
         nir_scalar s2 =
            nir_scalar_resolved(alu->src[2].src.ssa, alu->src[2].swizzle[0]);
         if (!negated && nir_scalar_is_const(s1) && nir_scalar_as_uint(s1) == 1) {
            return collect_or_leaves(alu->src[0].src.ssa, false, leaves) &&
                   collect_or_leaves(alu->src[2].src.ssa, false, leaves);
         }
         if (!negated && nir_scalar_is_const(s2) && nir_scalar_as_uint(s2) == 1) {
            return collect_or_leaves(alu->src[0].src.ssa, true, leaves) &&
                   collect_or_leaves(alu->src[1].src.ssa, false, leaves);
         }
         if (negated && nir_scalar_is_const(s2) && nir_scalar_as_uint(s2) == 0) {
            return collect_or_leaves(alu->src[0].src.ssa, true, leaves) &&
                   collect_or_leaves(alu->src[1].src.ssa, true, leaves);
         }
         if (negated && nir_scalar_is_const(s1) && nir_scalar_as_uint(s1) == 0) {
            return collect_or_leaves(alu->src[0].src.ssa, false, leaves) &&
                   collect_or_leaves(alu->src[2].src.ssa, true, leaves);
         }
      }
   }

   struct or_leaf leaf = {
      .def = def,
      .negated = negated,
      .kind = CMP_NONE,
   };
   util_dynarray_append(leaves, leaf);
   return true;
}

static bool
def_dominates_loop(nir_def *def, nir_loop *loop)
{
   if (!loop)
      return false;
   nir_block *pre = nir_loop_predecessor_block(loop);
   return nir_block_dominates(nir_def_block(def), pre);
}

static bool
scalar_dominates_loop(nir_scalar s, nir_loop *loop)
{
   if (!loop)
      return false;
   if (nir_scalar_is_const(s))
      return true;
   return def_dominates_loop(s.def, loop);
}

static bool
leaf_can_hoist_to_preheader(const struct or_leaf *leaf, nir_loop *loop)
{
   if (!loop)
      return false;
   if (leaf->kind != CMP_NONE) {
      if (!scalar_dominates_loop(leaf->var, loop))
         return false;
      if (!leaf->bound_is_const && !scalar_dominates_loop(leaf->bound, loop))
         return false;
      return true;
   }
   return def_dominates_loop(leaf->def, loop);
}

static bool
prefer_as_bound(nir_scalar a, nir_scalar b, nir_loop *loop)
{
   bool a_const = nir_scalar_is_const(a);
   bool b_const = nir_scalar_is_const(b);
   if (a_const != b_const)
      return a_const;

   /* Prefer convergent/uniform operands (e.g. UBO/preamble limits) over
    * divergent coordinates even when both dominate the inner loop (e.g. an
    * outer-loop coordinate `y` compared against a uniform image limit). */
   if (a.def->divergent != b.def->divergent)
      return !a.def->divergent;

   bool a_inv = scalar_dominates_loop(a, loop);
   bool b_inv = scalar_dominates_loop(b, loop);
   if (a_inv != b_inv)
      return a_inv;

   return false;
}

static void
classify_leaf(struct or_leaf *leaf, nir_loop *loop)
{
   nir_instr *instr = nir_def_instr(leaf->def);
   if (instr->type != nir_instr_type_alu)
      return;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.num_components != 1)
      return;

   nir_op op = alu->op;
   if (op != nir_op_ilt && op != nir_op_ige &&
       op != nir_op_ult && op != nir_op_uge)
      return;

   nir_scalar s0 =
      nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
   nir_scalar s1 =
      nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
   if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
      return;

   /* Apply negation: !(a < b) == (a >= b), !(a >= b) == (a < b). */
   if (leaf->negated) {
      switch (op) {
      case nir_op_ilt: op = nir_op_ige; break;
      case nir_op_ige: op = nir_op_ilt; break;
      case nir_op_ult: op = nir_op_uge; break;
      case nir_op_uge: op = nir_op_ult; break;
      default: return;
      }
   }

   bool s0_is_bound = prefer_as_bound(s0, s1, loop);
   nir_scalar var = s0_is_bound ? s1 : s0;
   nir_scalar bound = s0_is_bound ? s0 : s1;

   enum cmp_kind kind = CMP_NONE;
   if (!s0_is_bound) {
      switch (op) {
      case nir_op_ilt: kind = CMP_ILT; break;
      case nir_op_ige: kind = CMP_IGE; break;
      case nir_op_ult: kind = CMP_ULT; break;
      case nir_op_uge: kind = CMP_UGE; break;
      default: return;
      }
   } else {
      switch (op) {
      case nir_op_ilt: kind = CMP_IGT; break;
      case nir_op_ige: kind = CMP_ILE; break;
      case nir_op_ult: kind = CMP_UGT; break;
      case nir_op_uge: kind = CMP_ULE; break;
      default: return;
      }
   }

   /* Normalize exact non-wrapping constant inclusive bounds into strict bounds. */
   if (nir_scalar_is_const(bound)) {
      int32_t sv = nir_scalar_as_int(bound);
      uint32_t uv = nir_scalar_as_uint(bound);
      if (kind == CMP_ILE && sv < INT32_MAX) {
         leaf->kind = CMP_ILT;
         leaf->var = var;
         leaf->bound_is_const = true;
         leaf->bound_const = (int64_t)sv + 1;
         return;
      }
      if (kind == CMP_IGE && sv > INT32_MIN) {
         leaf->kind = CMP_IGT;
         leaf->var = var;
         leaf->bound_is_const = true;
         leaf->bound_const = (int64_t)sv - 1;
         return;
      }
      if (kind == CMP_ULE && uv < UINT32_MAX) {
         leaf->kind = CMP_ULT;
         leaf->var = var;
         leaf->bound_is_const = true;
         leaf->bound_const = (int64_t)((uint64_t)uv + 1);
         return;
      }
      if (kind == CMP_UGE && uv > 0) {
         leaf->kind = CMP_UGT;
         leaf->var = var;
         leaf->bound_is_const = true;
         leaf->bound_const = (int64_t)((uint64_t)uv - 1);
         return;
      }
   }

   leaf->kind = kind;
   leaf->var = var;
   leaf->bound = bound;
   leaf->bound_is_const = false;
}

static nir_def *
materialize_scalar(nir_builder *b, nir_scalar s)
{
   if (nir_scalar_is_const(s))
      return nir_imm_intN_t(b, nir_scalar_as_uint(s), s.def->bit_size);
   return nir_channel(b, s.def, s.comp);
}

static nir_def *
materialize_bound(nir_builder *b, const struct or_leaf *leaf)
{
   if (leaf->bound_is_const)
      return nir_imm_int(b, (int32_t)leaf->bound_const);
   return materialize_scalar(b, leaf->bound);
}

static nir_def *
emit_leaf(nir_builder *b, const struct or_leaf *leaf)
{
   if (leaf->kind == CMP_NONE)
      return leaf->negated ? nir_inot(b, leaf->def) : leaf->def;

   nir_def *var = materialize_scalar(b, leaf->var);
   nir_def *bound = materialize_bound(b, leaf);
   switch (leaf->kind) {
   case CMP_ILT: return nir_ilt(b, var, bound);
   case CMP_IGT: return nir_ilt(b, bound, var);
   case CMP_IGE: return nir_ige(b, var, bound);
   case CMP_ILE: return nir_ige(b, bound, var);
   case CMP_ULT: return nir_ult(b, var, bound);
   case CMP_UGT: return nir_ult(b, bound, var);
   case CMP_UGE: return nir_uge(b, var, bound);
   case CMP_ULE: return nir_uge(b, bound, var);
   default: UNREACHABLE("invalid cmp_kind");
   }
}

static nir_def *
combine_bounds(nir_builder *b, enum cmp_kind kind, nir_def *a, nir_def *c)
{
   switch (kind) {
   case CMP_ILT:
   case CMP_ILE:
      return nir_imax(b, a, c);
   case CMP_IGT:
   case CMP_IGE:
      return nir_imin(b, a, c);
   case CMP_ULT:
   case CMP_ULE:
      return nir_umax(b, a, c);
   case CMP_UGT:
   case CMP_UGE:
      return nir_umin(b, a, c);
   default:
      UNREACHABLE("invalid cmp_kind");
   }
}

static nir_loop *
enclosing_loop(nir_cf_node *node)
{
   for (nir_cf_node *p = node->parent; p; p = p->parent) {
      if (p->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(p);
   }
   return NULL;
}

static bool
fold_if_condition_disjunction(nir_if *nif)
{
   struct util_dynarray leaves;
   util_dynarray_init(&leaves, NULL);

   if (!collect_or_leaves(nif->condition.ssa, false, &leaves)) {
      util_dynarray_fini(&leaves);
      return false;
   }

   unsigned n = util_dynarray_num_elements(&leaves, struct or_leaf);
   if (n < 2) {
      util_dynarray_fini(&leaves);
      return false;
   }

   nir_loop *loop = enclosing_loop(&nif->cf_node);
   struct or_leaf *arr = leaves.data;
   for (unsigned i = 0; i < n; i++) {
      classify_leaf(&arr[i], loop);
      arr[i].is_loop_inv = leaf_can_hoist_to_preheader(&arr[i], loop);
   }

   bool has_fold = false;
   unsigned inv_count = 0, var_count = 0;
   for (unsigned i = 0; i < n; i++) {
      if (arr[i].is_loop_inv)
         inv_count++;
      else
         var_count++;
      if (arr[i].kind == CMP_NONE)
         continue;
      for (unsigned j = i + 1; j < n; j++) {
         if (arr[j].kind == arr[i].kind &&
             nir_scalar_equal(arr[j].var, arr[i].var)) {
            has_fold = true;
            break;
         }
      }
   }

   if (!has_fold && !(loop && inv_count >= 1 && var_count >= 1)) {
      util_dynarray_fini(&leaves);
      return false;
   }

   nir_builder b_if = nir_builder_at(nir_before_cf_node(&nif->cf_node));
   nir_builder b_pre;
   bool can_hoist_pre = loop != NULL;
   if (can_hoist_pre)
      b_pre = nir_builder_at(nir_before_cf_node(&loop->cf_node));

   /* Combine matching (kind, var) leaves. Build combined bounds in the loop
    * preheader whenever all participating bounds dominate the loop, so
    * constant/uniform bounds are available before the loop as well. */
   bool *consumed = calloc(n, sizeof(bool));
   struct util_dynarray folded;
   util_dynarray_init(&folded, NULL);

   for (unsigned i = 0; i < n; i++) {
      if (consumed[i])
         continue;
      struct or_leaf cur = arr[i];
      if (cur.kind != CMP_NONE) {
         bool merged_any = false;
         bool all_bounds_dom_loop =
            can_hoist_pre &&
            (cur.bound_is_const || scalar_dominates_loop(cur.bound, loop));
         for (unsigned j = i + 1; j < n; j++) {
            if (!consumed[j] && arr[j].kind == cur.kind &&
                nir_scalar_equal(arr[j].var, cur.var)) {
               merged_any = true;
               if (all_bounds_dom_loop && !arr[j].bound_is_const &&
                   !scalar_dominates_loop(arr[j].bound, loop))
                  all_bounds_dom_loop = false;
            }
         }
         if (merged_any) {
            nir_builder *bb = all_bounds_dom_loop ? &b_pre : &b_if;
            nir_def *bound = materialize_bound(bb, &cur);
            for (unsigned j = i + 1; j < n; j++) {
               if (!consumed[j] && arr[j].kind == cur.kind &&
                   nir_scalar_equal(arr[j].var, cur.var)) {
                  bound = combine_bounds(bb, cur.kind, bound,
                                         materialize_bound(bb, &arr[j]));
                  consumed[j] = true;
               }
            }
            cur.bound = nir_get_scalar(bound, 0);
            cur.bound_is_const = false;
            cur.def = NULL;
            cur.is_loop_inv = leaf_can_hoist_to_preheader(&cur, loop);
         }
      }
      util_dynarray_append(&folded, cur);
   }
   free(consumed);

   /* Emit all loop-invariant leaves directly into the loop preheader as a
    * single 1-bit disjunction, placing each comparison on src[0] of nir_ior
    * so AGX select fusion lowers each comparison to a single icmpsel. */
   nir_def *inv_or = NULL;
   if (can_hoist_pre) {
      util_dynarray_foreach(&folded, struct or_leaf, leaf) {
         if (!leaf->is_loop_inv)
            continue;
         nir_def *v = emit_leaf(&b_pre, leaf);
         inv_or = inv_or ? nir_ior(&b_pre, v, inv_or) : v;
      }
   }

   nir_def *result = inv_or;
   util_dynarray_foreach(&folded, struct or_leaf, leaf) {
      if (can_hoist_pre && leaf->is_loop_inv)
         continue;
      nir_def *v = emit_leaf(&b_if, leaf);
      result = result ? nir_ior(&b_if, v, result) : v;
   }

   nir_src_rewrite(&nif->condition, result);

   util_dynarray_fini(&folded);
   util_dynarray_fini(&leaves);
   return true;
}

static bool
fold_disjunctions_in_list(struct exec_list *cf_list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= fold_if_condition_disjunction(nif);
         progress |= fold_disjunctions_in_list(&nif->then_list);
         progress |= fold_disjunctions_in_list(&nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= fold_disjunctions_in_list(&loop->body);
         break;
      }
      default:
         break;
      }
   }
   return progress;
}

bool
agx_nir_opt_merge_nested_early_exits(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      bool merged = merge_nested_early_exits_in_list(&impl->body);
      if (merged) {
         nir_progress(true, impl, nir_metadata_none);
         nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance);
         nir_divergence_analysis_impl(
            impl, shader->options ? shader->options->divergence_analysis_options
                                  : 0);
         fold_disjunctions_in_list(&impl->body);
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      } else {
         nir_no_progress(impl);
      }
   }

   return progress;
}

/*
 * Post-sink / post-LICM micro-hoist for a single 1-bit loop-invariant boolean
 * guard chain (`bcsel`/`ior`/`iand` over `ilt`/`ige`/`ult`/`uge`/`ieq`/`ine`
 * + `load_preamble`/`load_const`). Cost is at most 1 half-register live
 * across the loop, so it stays safe even when full LICM is skipped for high
 * register pressure, and it runs identically on AGX3B and AGX3C.
 */
static bool
guard_instr_hoistable(nir_instr *instr, nir_loop *loop,
                      struct hash_table *visited, unsigned *cmp_count)
{
   if (_mesa_hash_table_search(visited, instr))
      return true;

   nir_block *pre = nir_loop_predecessor_block(loop);
   if (nir_block_dominates(instr->block, pre))
      return true;

   if (instr->block == nir_loop_first_block(loop) &&
       instr->type == nir_instr_type_phi)
      return false;

   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      break;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_preamble)
         return false;
      break;
   }
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->def.num_components != 1)
         return false;
      switch (alu->op) {
      case nir_op_ilt:
      case nir_op_ige:
      case nir_op_ult:
      case nir_op_uge:
      case nir_op_ieq:
      case nir_op_ine:
         (*cmp_count)++;
         break;
      case nir_op_bcsel:
      case nir_op_ior:
      case nir_op_iand:
      case nir_op_inot:
         if (alu->def.bit_size != 1)
            return false;
         break;
      default:
         return false;
      }
      unsigned nsrc = nir_op_infos[alu->op].num_inputs;
      for (unsigned i = 0; i < nsrc; i++) {
         if (!guard_instr_hoistable(nir_def_instr(alu->src[i].src.ssa), loop,
                                    visited, cmp_count))
            return false;
      }
      break;
   }
   default:
      return false;
   }

   _mesa_hash_table_insert(visited, instr, instr);
   return true;
}

static void
collect_hoist_order(nir_instr *instr, nir_loop *loop,
                    struct hash_table *emitted, struct util_dynarray *order)
{
   nir_block *pre = nir_loop_predecessor_block(loop);
   if (nir_block_dominates(instr->block, pre) ||
       _mesa_hash_table_search(emitted, instr))
      return;

   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      unsigned nsrc = nir_op_infos[alu->op].num_inputs;
      for (unsigned i = 0; i < nsrc; i++)
         collect_hoist_order(nir_def_instr(alu->src[i].src.ssa), loop, emitted,
                             order);
   }

   _mesa_hash_table_insert(emitted, instr, instr);
   util_dynarray_append(order, instr);
}

static bool
hoist_guard_bit_in_loop(nir_loop *loop)
{
   if (!nir_loop_has_back_edge(loop))
      return false;

   nir_block *pre = nir_loop_predecessor_block(loop);
   bool progress = false;

   /* Walk top-level ifs directly inside the loop body (exit-dominating
    * guards on the main loop path) and hoist any pure 1-bit loop-invariant
    * sub-expression with >= 2 comparisons into `pre`. */
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node);
      if (nir_block_ends_in_break(nir_if_last_then_block(nif)) ||
          nir_block_ends_in_break(nir_if_last_else_block(nif)))
         continue;

      /* Walk down the 1-bit bcsel/ior/iand chain from nif->condition to find
       * the maximal loop-invariant 1-bit root sub-expression. */
      struct util_dynarray worklist;
      util_dynarray_init(&worklist, NULL);
      util_dynarray_append(&worklist, nif->condition.ssa);

      void *mem_ctx = ralloc_context(NULL);
      struct hash_table *seen_roots = _mesa_pointer_hash_table_create(mem_ctx);
      unsigned hoisted_roots = 0;

      while (util_dynarray_num_elements(&worklist, nir_def *) > 0 &&
             hoisted_roots < 1) {
         nir_def *cur = util_dynarray_pop(&worklist, nir_def *);
         if (_mesa_hash_table_search(seen_roots, cur))
            continue;
         _mesa_hash_table_insert(seen_roots, cur, cur);

         nir_instr *instr = nir_def_instr(cur);
         if (nir_block_dominates(instr->block, pre))
            continue;

         struct hash_table *visited = _mesa_pointer_hash_table_create(mem_ctx);
         unsigned cmp_count = 0;
         if (cur->bit_size == 1 && cur->num_components == 1 &&
             guard_instr_hoistable(instr, loop, visited, &cmp_count) &&
             cmp_count >= 2) {
            struct hash_table *emitted =
               _mesa_pointer_hash_table_create(mem_ctx);
            struct util_dynarray order;
            util_dynarray_init(&order, mem_ctx);
            collect_hoist_order(instr, loop, emitted, &order);

            /* Only hoist if every hoisted def (other than cur) is only used
             * within the hoisted set or outside the loop, avoiding widening
             * live-ins beyond the single 1-bit result `cur`. */
            bool clean = true;
            util_dynarray_foreach(&order, nir_instr *, ip) {
               nir_def *d = nir_instr_def(*ip);
               if (!d || d == cur)
                  continue;
               if ((*ip)->type == nir_instr_type_load_const ||
                   ((*ip)->type == nir_instr_type_intrinsic &&
                    nir_instr_as_intrinsic(*ip)->intrinsic ==
                       nir_intrinsic_load_preamble))
                  continue;
               nir_foreach_use_including_if(u, d) {
                  if (nir_src_is_if(u) ||
                      !_mesa_hash_table_search(emitted, nir_src_use_instr(u))) {
                     clean = false;
                     break;
                  }
               }
               if (!clean)
                  break;
            }

            if (clean) {
               nir_cursor dest = nir_before_cf_node(&loop->cf_node);
               util_dynarray_foreach(&order, nir_instr *, ip) {
                  if ((*ip)->type == nir_instr_type_load_const ||
                      ((*ip)->type == nir_instr_type_intrinsic &&
                       nir_instr_as_intrinsic(*ip)->intrinsic ==
                          nir_intrinsic_load_preamble)) {
                     /* Clone lightweight uniform/const producers so any
                      * in-loop uses stay local without extra live-ins. */
                     nir_instr *clone = nir_instr_clone(
                        pre->cf_node.parent->type == nir_cf_node_function
                           ? nir_cf_node_as_function(pre->cf_node.parent)->function->shader
                           : nir_cf_node_get_function(&loop->cf_node)->function->shader,
                        *ip);
                     nir_instr_insert(dest, clone);
                     nir_def *orig_d = nir_instr_def(*ip);
                     nir_def *clone_d = nir_instr_def(clone);
                     nir_foreach_use_safe(u, orig_d) {
                        if (_mesa_hash_table_search(emitted,
                                                    nir_src_use_instr(u)))
                           nir_src_rewrite(u, clone_d);
                     }
                  } else {
                     nir_instr_move(dest, *ip);
                  }
               }
               hoisted_roots++;
               progress = true;
               continue;
            }
         }

         if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_bcsel || alu->op == nir_op_ior ||
                alu->op == nir_op_iand || alu->op == nir_op_inot) {
               unsigned nsrc = nir_op_infos[alu->op].num_inputs;
               for (unsigned i = 0; i < nsrc; i++) {
                  if (alu->src[i].src.ssa->bit_size == 1)
                     util_dynarray_append(&worklist, alu->src[i].src.ssa);
               }
            }
         }
      }

      util_dynarray_fini(&worklist);
      ralloc_free(mem_ctx);
   }

   return progress;
}

static bool
hoist_guard_bits_in_list(struct exec_list *cf_list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= hoist_guard_bits_in_list(&nif->then_list);
         progress |= hoist_guard_bits_in_list(&nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= hoist_guard_bits_in_list(&loop->body);
         progress |= hoist_guard_bit_in_loop(loop);
         break;
      }
      default:
         break;
      }
   }
   return progress;
}

bool
agx_nir_hoist_loop_invariant_guard_bits(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_dominance);
      if (hoist_guard_bits_in_list(&impl->body)) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      } else {
         nir_no_progress(impl);
      }
   }

   return progress;
}
