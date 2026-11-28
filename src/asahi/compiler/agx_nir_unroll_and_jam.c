/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "nir.h"
#include "nir_builder.h"
#include "util/bitset.h"
#include "util/format/u_formats.h"
#include "util/hash_table.h"
#include "util/set.h"
#include "util/u_dynarray.h"

/*
 * General outer-loop unroll-and-jam for pure read-only counted outer loops
 * wrapping an outer-invariant 1- or 2-deep inner reduction loop nest.
 *
 * When an inner loop reads at least one outer-invariant address and at least
 * one outer-variant address into a small set of scalar accumulators, jamming
 * J consecutive outer iterations into a single traversal of the inner loop
 * nest eliminates (J - 1)/J of the outer-invariant memory loads and inner
 * loop overhead, and exposes cross-iteration spatial locality across the
 * outer-variant loads for subsequent unrolling, CSE, and vector merging.
 */

enum cont_rel {
   REL_LT, /* iv < bound  */
   REL_LE, /* iv <= bound */
   REL_GT, /* iv > bound  */
   REL_GE, /* iv >= bound */
};

struct counted_loop {
   nir_loop *loop;
   nir_if *nif;
   bool break_in_then;
   bool bottom;
   nir_phi_instr *iv;
   int64_t offset;
   nir_scalar bound;
   int64_t step;
   enum cont_rel rel;
   bool is_unsigned;
};

struct live_out {
   nir_def *def;
   nir_def *reg;
   bool invariant;
};

static bool
block_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node == &loop->cf_node)
         return true;
   }
   return false;
}

static bool
def_outside_loop(nir_def *def, nir_loop *loop)
{
   return !block_in_loop(nir_def_block(def), loop);
}

static nir_block *
use_block(nir_src *src)
{
   if (nir_src_is_if(src))
      return nir_cf_node_as_block(nir_cf_node_prev(&nir_src_use_if(src)->cf_node));
   return nir_src_use_instr(src)->block;
}

static bool
block_is_empty(nir_block *block)
{
   return exec_list_is_empty(&block->instr_list);
}

static bool
list_is_empty_block(struct exec_list *list)
{
   nir_cf_node *first = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   nir_cf_node *last = exec_node_data(nir_cf_node, exec_list_get_tail(list), node);
   return first == last && first->type == nir_cf_node_block &&
          block_is_empty(nir_cf_node_as_block(first));
}

static bool
list_is_break(struct exec_list *list)
{
   nir_cf_node *first = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   nir_cf_node *last = exec_node_data(nir_cf_node, exec_list_get_tail(list), node);
   if (first != last || first->type != nir_cf_node_block)
      return false;
   nir_block *block = nir_cf_node_as_block(first);
   nir_instr *head = nir_block_first_instr(block);
   if (!head || head != nir_block_last_instr(block) ||
       head->type != nir_instr_type_jump)
      return false;
   return nir_instr_as_jump(head)->type == nir_jump_break;
}

static bool
is_break_if(nir_if *nif, bool *break_in_then)
{
   if (list_is_break(&nif->then_list) && list_is_empty_block(&nif->else_list)) {
      *break_in_then = true;
      return true;
   }
   if (list_is_break(&nif->else_list) && list_is_empty_block(&nif->then_list)) {
      *break_in_then = false;
      return true;
   }
   return false;
}

static bool
header_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   nir_instr *instr = nir_def_instr(def);
   if (instr->block != nir_loop_first_block(loop) || depth == 0)
      return false;
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!header_invariant(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!nir_intrinsic_can_reorder(intr) ||
          !nir_intrinsic_infos[intr->intrinsic].has_dest)
         return false;
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         if (!header_invariant(intr->src[i].ssa, loop, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

static bool
loop_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (depth == 0)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!loop_invariant(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!nir_intrinsic_can_reorder(intr) ||
          !nir_intrinsic_infos[intr->intrinsic].has_dest)
         return false;
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         if (!loop_invariant(intr->src[i].ssa, loop, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

static nir_def *
rematerialize(nir_builder *b, nir_def *def, nir_loop *loop,
              struct hash_table *remap)
{
   if (def_outside_loop(def, loop))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         rematerialize(b, alu->src[i].src.ssa, loop, remap);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         rematerialize(b, intr->src[i].ssa, loop, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *result = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, result);
   return result;
}

static bool
instr_is_pure_alu(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!nir_intrinsic_infos[intr->intrinsic].has_dest)
         return false;
      if (intr->intrinsic == nir_intrinsic_load_preamble ||
          intr->intrinsic == nir_intrinsic_load_push_constant)
         return true;
      if (intr->intrinsic == nir_intrinsic_load_global_constant &&
          nir_intrinsic_can_reorder(intr) &&
          !(nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_COHERENT)))
         return true;
      return false;
   }
   default:
      return false;
   }
}

static bool
instr_is_readonly_load(nir_instr *instr)
{
   if (instr->type == nir_instr_type_tex)
      return !nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr));
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_agx:
      if (nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_COHERENT)))
         return false;
      return true;
   default:
      return false;
   }
}

static bool
instr_is_safe_readonly(nir_instr *instr)
{
   if (instr->type == nir_instr_type_phi || instr->type == nir_instr_type_jump)
      return true;
   if (instr_is_pure_alu(instr) || instr_is_readonly_load(instr))
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      if ((info->flags & (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP)) ||
          !(info->flags & NIR_INTRINSIC_CAN_ELIMINATE))
         return false;
      if (nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_COHERENT)))
         return false;
      switch (intr->intrinsic) {
      case nir_intrinsic_load_active_subgroup_invocation_agx:
      case nir_intrinsic_load_active_subgroup_count_agx:
      case nir_intrinsic_quad_ballot_agx:
      case nir_intrinsic_is_helper_invocation:
      case nir_intrinsic_load_helper_invocation:
      case nir_intrinsic_demote_samples:
      case nir_intrinsic_intersect_ray_agx:
      case nir_intrinsic_continue_ray_agx:
      case nir_intrinsic_load_scs_uniform_agx:
      case nir_intrinsic_store_scs_uniform_agx:
      case nir_intrinsic_mesh_payload_entry_barrier_agx:
      case nir_intrinsic_object_payload_exit_barrier_agx:
      case nir_intrinsic_fence_mesh_uvb_agx:
      case nir_intrinsic_barrier:
      case nir_intrinsic_terminate:
      case nir_intrinsic_terminate_if:
      case nir_intrinsic_demote:
      case nir_intrinsic_demote_if:
      case nir_intrinsic_ddx:
      case nir_intrinsic_ddy:
      case nir_intrinsic_ddx_fine:
      case nir_intrinsic_ddy_fine:
      case nir_intrinsic_ddx_coarse:
      case nir_intrinsic_ddy_coarse:
         return false;
      default:
         return info->has_dest && nir_intrinsic_can_reorder(intr);
      }
   }
   return false;
}

static bool
analyze_outer_counted_loop(nir_loop *loop, struct counted_loop *out)
{
   if (nir_loop_has_continue_construct(loop) || loop->partially_unrolled)
      return false;

   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   nir_cf_node *after_hdr = nir_cf_node_next(&header->cf_node);
   nir_cf_node *before_latch = nir_cf_node_prev(&latch->cf_node);

   nir_if *nif = NULL;
   bool break_in_then = false;
   bool bottom = false;
   if (after_hdr && after_hdr->type == nir_cf_node_if &&
       is_break_if(nir_cf_node_as_if(after_hdr), &break_in_then)) {
      nif = nir_cf_node_as_if(after_hdr);
      bottom = false;
   } else if (before_latch && before_latch->type == nir_cf_node_if &&
              block_is_empty(latch) &&
              is_break_if(nir_cf_node_as_if(before_latch), &break_in_then)) {
      nif = nir_cf_node_as_if(before_latch);
      bottom = true;
   } else {
      return false;
   }

   nir_scalar cond = nir_scalar_resolved(nif->condition.ssa, 0);
   bool negate = false;
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      negate = !negate;
      cond = nir_scalar_chase_alu_src(cond, 0);
   }
   if (!nir_scalar_is_alu(cond))
      return false;

   nir_op op = nir_scalar_alu_op(cond);
   if (op != nir_op_ilt && op != nir_op_ige && op != nir_op_ult &&
       op != nir_op_uge)
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   if (a.def->bit_size != 32 || a.def->num_components != 1 ||
       b.def->num_components != 1)
      return false;

   bool iv_left = false;
   nir_scalar ivs = {0}, bound = {0};
   int64_t offset = 0;
   for (unsigned side = 0; side < 2 && !ivs.def; ++side) {
      nir_scalar x = side == 0 ? a : b, y = side == 0 ? b : a;
      if (!nir_scalar_is_const(y) && !header_invariant(y.def, loop, 8))
         continue;
      int64_t c = 0;
      if (nir_scalar_is_alu(x) && nir_scalar_alu_op(x) == nir_op_iadd) {
         for (unsigned k = 0; k < 2; ++k) {
            nir_scalar kk = nir_scalar_chase_alu_src(x, k);
            nir_scalar other = nir_scalar_chase_alu_src(x, 1 - k);
            if (nir_scalar_is_const(kk) &&
                nir_def_instr(other.def)->type == nir_instr_type_phi) {
               c = nir_scalar_as_int(kk);
               x = other;
               break;
            }
         }
      }
      if (nir_def_instr(x.def)->type == nir_instr_type_phi &&
          nir_def_instr(x.def)->block == header && x.def->bit_size == 32 &&
          c >= -0x10000 && c <= 0x10000) {
         ivs = x;
         bound = y;
         offset = c;
         iv_left = side == 0;
      }
   }
   if (!ivs.def)
      return false;

   nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(ivs.def));
   nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
   if (!back)
      return false;

   nir_scalar update = nir_scalar_resolved(back->src.ssa, 0);
   if (!nir_scalar_is_alu(update) || nir_scalar_alu_op(update) != nir_op_iadd)
      return false;

   int64_t step = 0;
   bool found = false;
   for (unsigned s = 0; s < 2; ++s) {
      nir_scalar src = nir_scalar_chase_alu_src(update, s);
      nir_scalar other = nir_scalar_chase_alu_src(update, 1 - s);
      if (other.def == &phi->def && nir_scalar_is_const(src)) {
         step = nir_scalar_as_int(src);
         found = true;
      }
   }
   if (!found || (step != 1 && step != -1))
      return false;

   bool break_on_true = break_in_then != negate;
   bool less = op == nir_op_ilt || op == nir_op_ult;
   bool cont_less = break_on_true ? !less : less;
   enum cont_rel rel;
   if (iv_left)
      rel = cont_less ? REL_LT : REL_GE;
   else
      rel = cont_less ? REL_GT : REL_LE;

   if ((rel == REL_LT || rel == REL_LE) != (step > 0))
      return false;

   *out = (struct counted_loop){
      .loop = loop,
      .nif = nif,
      .break_in_then = break_in_then,
      .bottom = bottom,
      .iv = phi,
      .offset = offset,
      .bound = bound,
      .step = step,
      .rel = rel,
      .is_unsigned = op == nir_op_ult || op == nir_op_uge,
   };
   return true;
}

/* Guard value G such that `iv < G` (REL_LT/REL_LE) or `iv > G` (REL_GT/REL_GE)
 * proves the next `factor` iterations all execute and `iv + factor * step`
 * does not overflow. */
static nir_def *
guard_bound(nir_builder *b, const struct counted_loop *cl, nir_def *bound,
            unsigned factor)
{
   const int64_t astep = cl->step > 0 ? cl->step : -cl->step;
   const int64_t span = (int64_t)(factor - 1) * astep;
   const bool u = cl->is_unsigned;
   const int64_t min = u ? 0 : INT32_MIN;
   const int64_t max = u ? UINT32_MAX : INT32_MAX;

   nir_def *valid, *adj, *never;
   switch (cl->rel) {
   case REL_LT:
   case REL_LE: {
      nir_def *lowest = nir_imm_int(b, (uint32_t)(min + span));
      valid = u ? nir_uge(b, bound, lowest) : nir_ige(b, bound, lowest);
      if (cl->rel == REL_LE) {
         /* Ensure iv + factor * step <= bound + step does not wrap past max. */
         nir_def *cap = nir_imm_int(b, (uint32_t)(max - astep));
         nir_def *not_top = u ? nir_uge(b, cap, bound) : nir_ige(b, cap, bound);
         valid = nir_iand(b, valid, not_top);
      }
      int64_t sub = cl->rel == REL_LT ? span : span - 1;
      adj = nir_iadd_imm(b, bound, (uint32_t)(-sub));
      never = nir_imm_int(b, (uint32_t)min);
      break;
   }
   default: {
      nir_def *highest = nir_imm_int(b, (uint32_t)(max - span));
      valid = u ? nir_uge(b, highest, bound) : nir_ige(b, highest, bound);
      if (cl->rel == REL_GE) {
         nir_def *cap = nir_imm_int(b, (uint32_t)(min + astep));
         nir_def *not_bot = u ? nir_uge(b, bound, cap) : nir_ige(b, bound, cap);
         valid = nir_iand(b, valid, not_bot);
      }
      int64_t add = cl->rel == REL_GT ? span : span - 1;
      adj = nir_iadd_imm(b, bound, (uint32_t)add);
      never = nir_imm_int(b, (uint32_t)max);
      break;
   }
   }
   return nir_bcsel(b, valid, adj, never);
}

static bool
back_edge_live_outs(nir_loop *loop, struct live_out *outs, unsigned *count)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   if (!block_is_empty(after) &&
       nir_block_first_instr(after)->type == nir_instr_type_phi)
      return false;
   *count = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         bool outside = false;
         nir_foreach_use_including_if(src, def)
            outside |= !block_in_loop(use_block(src), loop);
         if (!outside)
            continue;
         if (instr->type == nir_instr_type_phi && block == header)
            return false;
         bool carried = false;
         nir_foreach_phi(phi, header) {
            nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
            carried |= back && back->src.ssa == def;
         }
         const bool invariant = !carried && loop_invariant(def, loop, 8);
         if ((!carried && !invariant) || *count == 16)
            return false;
         outs[(*count)++] = (struct live_out){def, NULL, invariant};
      }
   }
   return true;
}

struct uaj_match {
   struct counted_loop cl;
   nir_if *outer_guard; /* optional outer if around inner_top + post_inner */
   bool guard_in_then;
   nir_block *pre_inner_block;
   nir_loop *inner_top;
   nir_loop *inner_leaf; /* == inner_top for 2-deep, child loop for 3-deep */
   nir_block *post_inner_block;
   nir_block *merge_block; /* block after outer_guard, or post_inner_block */

   unsigned n_outer_phis;
   nir_phi_instr *outer_phis[8];
   unsigned iv_idx;
   nir_def *red_updated_defs[8]; /* value produced in post_inner_block for each non-IV phi */

   BITSET_WORD *variant; /* SSA defs inside inner_top that depend on outer phis */
};

struct bitset_src_cb {
   const BITSET_WORD *bits;
   bool any;
};

static bool
check_src_in_bitset(nir_src *src, void *data)
{
   struct bitset_src_cb *cb = data;
   if (BITSET_TEST(cb->bits, src->ssa->index))
      cb->any = true;
   return true;
}

static bool
instr_uses_bitset(nir_instr *instr, const BITSET_WORD *bits)
{
   struct bitset_src_cb cb = {bits, false};
   nir_foreach_src(instr, check_src_in_bitset, &cb);
   return cb.any;
}

/* Validate a single inner loop (no continue construct, single terminator if,
 * no other ifs, and at most one child loop if allow_child is true). */
static bool
validate_simple_loop(nir_loop *loop, bool allow_child, nir_loop **child_out)
{
   if (nir_loop_has_continue_construct(loop) || loop->partially_unrolled)
      return false;

   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   nir_cf_node *after_hdr = nir_cf_node_next(&header->cf_node);
   nir_cf_node *before_latch = nir_cf_node_prev(&latch->cf_node);

   nir_if *term = NULL;
   bool dummy;
   if (after_hdr && after_hdr->type == nir_cf_node_if &&
       is_break_if(nir_cf_node_as_if(after_hdr), &dummy)) {
      term = nir_cf_node_as_if(after_hdr);
   } else if (before_latch && before_latch->type == nir_cf_node_if &&
              block_is_empty(latch) &&
              is_break_if(nir_cf_node_as_if(before_latch), &dummy)) {
      term = nir_cf_node_as_if(before_latch);
   } else {
      return false;
   }

   nir_loop *child = NULL;
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (node->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (!instr_is_safe_readonly(instr))
               return false;
            if (instr->type == nir_instr_type_jump)
               return false;
            if (instr->type == nir_instr_type_phi && blk != header)
               return false;
         }
      } else if (node->type == nir_cf_node_if) {
         if (nir_cf_node_as_if(node) != term)
            return false;
      } else if (node->type == nir_cf_node_loop) {
         if (!allow_child || child)
            return false;
         child = nir_cf_node_as_loop(node);
      } else {
         return false;
      }
   }

   if (child && !validate_simple_loop(child, false, NULL))
      return false;

   if (child_out)
      *child_out = child;
   return true;
}

/* Match an invariant expression as `base + const_off` (with const_off == 0 when
 * there is no constant addend). */
static void
split_const_addend(nir_scalar s, nir_def **base_out, int64_t *off_out)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(s1)) {
         *base_out = s0.def;
         *off_out = nir_scalar_as_int(s1);
         return;
      }
      if (nir_scalar_is_const(s0)) {
         *base_out = s1.def;
         *off_out = nir_scalar_as_int(s0);
         return;
      }
   }
   *base_out = s.def;
   *off_out = 0;
}

/* Match `iv` or `ineg(iv)` (possibly with an invariant constant offset). */
static bool
match_iv_term(nir_scalar s, nir_def *iv, int *sign_out)
{
   s = nir_scalar_chase_movs(s);
   if (s.def == iv) {
      *sign_out = 1;
      return true;
   }
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_ineg) {
      nir_scalar inner = nir_scalar_chase_alu_src(s, 0);
      if (inner.def == iv) {
         *sign_out = -1;
         return true;
      }
   }
   return false;
}

/* Trace a variant load's address expression back through single-variant-input
 * address/clamp ops to a sliding-window affine form `(base + off) +/- iv`. */
static bool
extract_iv_sliding_key(nir_def *def, nir_def *iv, const BITSET_WORD *variant,
                       unsigned depth, nir_def **base_out, int64_t *off_out,
                       int *sign_out)
{
   if (!def || depth == 0 || !BITSET_TEST(variant, def->index))
      return false;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_scalar s = nir_get_scalar(def, 0);

   if (alu->op == nir_op_iadd || alu->op == nir_op_isub) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      int sign = 0;
      if (!BITSET_TEST(variant, a.def->index) && match_iv_term(b, iv, &sign)) {
         split_const_addend(a, base_out, off_out);
         *sign_out = alu->op == nir_op_isub ? -sign : sign;
         return true;
      }
      if (alu->op == nir_op_iadd && !BITSET_TEST(variant, b.def->index) &&
          match_iv_term(a, iv, &sign)) {
         split_const_addend(b, base_out, off_out);
         *sign_out = sign;
         return true;
      }
   }

   switch (alu->op) {
   case nir_op_mov:
   case nir_op_u2u64:
   case nir_op_i2i64:
   case nir_op_u2u32:
   case nir_op_i2i32:
   case nir_op_iadd:
   case nir_op_isub:
   case nir_op_amul:
   case nir_op_imul:
   case nir_op_ishl:
   case nir_op_ushr:
   case nir_op_ishr:
   case nir_op_iand:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax: {
      nir_def *var_src = NULL;
      unsigned n_inputs = nir_op_infos[alu->op].num_inputs;
      for (unsigned i = 0; i < n_inputs; ++i) {
         nir_def *src = alu->src[i].src.ssa;
         if (BITSET_TEST(variant, src->index)) {
            if (var_src && var_src != src)
               return false;
            var_src = src;
         }
      }
      if (!var_src)
         return false;
      return extract_iv_sliding_key(var_src, iv, variant, depth - 1, base_out,
                                    off_out, sign_out);
   }
   default:
      return false;
   }
}

/* Check whether an already-unrolled 1D inner loop's variant loads form a
 * sliding window along the outer IV so jamming by J reuses loads across
 * adjacent outer iterations via diagonal CSE instead of multiplying load
 * traffic by J. */
static bool
has_sliding_window_var_loads(nir_loop *inner_leaf, nir_def *iv, int64_t step,
                             const BITSET_WORD *variant, unsigned var_loads)
{
   struct tap_key {
      nir_def *base;
      int64_t off;
      int sign;
   };
   struct tap_key keys[16];
   unsigned n_keys = 0;

   if (var_loads < 2 || var_loads > ARRAY_SIZE(keys))
      return false;

   nir_foreach_block_in_cf_node(blk, &inner_leaf->cf_node) {
      nir_foreach_instr(instr, blk) {
         if (!instr_is_readonly_load(instr) || !instr_uses_bitset(instr, variant))
            continue;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         nir_def *addr = NULL;
         for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs;
              ++i) {
            if (BITSET_TEST(variant, intr->src[i].ssa->index)) {
               addr = intr->src[i].ssa;
               break;
            }
         }
         if (!addr || n_keys >= ARRAY_SIZE(keys))
            continue;
         nir_def *base = NULL;
         int64_t off = 0;
         int sign = 0;
         if (extract_iv_sliding_key(addr, iv, variant, 12, &base, &off, &sign))
            keys[n_keys++] = (struct tap_key){base, off, sign};
      }
   }

   if (n_keys < 2)
      return false;

   unsigned overlap = 0;
   for (unsigned i = 0; i < n_keys; ++i) {
      bool adj = false;
      for (unsigned j = 0; j < n_keys; ++j) {
         if (i == j)
            continue;
         if (keys[i].base == keys[j].base && keys[i].sign == keys[j].sign &&
             (keys[j].off == keys[i].off + step ||
              keys[j].off == keys[i].off - step)) {
            adj = true;
            break;
         }
      }
      if (adj)
         overlap++;
   }

   return overlap * 2 >= var_loads;
}

static bool
match_unroll_and_jam(nir_function_impl *impl, nir_loop *outer,
                     struct uaj_match *m)
{
   memset(m, 0, sizeof(*m));
   if (!analyze_outer_counted_loop(outer, &m->cl))
      return false;

   /* Verify entire outer loop is side-effect free and read-only. */
   nir_foreach_block_in_cf_node(blk, &outer->cf_node) {
      nir_foreach_instr(instr, blk) {
         if (!instr_is_safe_readonly(instr))
            return false;
      }
   }

   nir_block *header = nir_loop_first_block(outer);
   nir_block *latch = nir_loop_last_block(outer);
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));

   /* Collect outer header phis (limit 8). */
   m->iv_idx = UINT32_MAX;
   nir_foreach_phi(phi, header) {
      if (m->n_outer_phis >= 8)
         return false;
      if (phi == m->cl.iv)
         m->iv_idx = m->n_outer_phis;
      if (!nir_phi_get_src_from_block(phi, pre) ||
          !nir_phi_get_src_from_block(phi, latch))
         return false;
      m->outer_phis[m->n_outer_phis++] = phi;
   }
   if (m->iv_idx == UINT32_MAX)
      return false;

   /* Check outer body CF nodes (excluding terminator if and empty bottom latch). */
   nir_cf_node *body_nodes[8];
   unsigned n_body = 0;
   foreach_list_typed(nir_cf_node, node, node, &outer->body) {
      if (node == &m->cl.nif->cf_node)
         continue;
      if (m->cl.bottom && node == &latch->cf_node)
         continue;
      if (n_body >= 8)
         return false;
      body_nodes[n_body++] = node;
   }

   /* Pattern A: header_block, inner_top, post_block
    * Pattern B: header_block, outer_guard(pre_inner, inner_top, post_inner), merge_block
    * (with top-tested loops having an extra block between header and terminator). */
   nir_if *guard = NULL;
   nir_loop *inner_top = NULL;
   nir_block *pre_inner = NULL;
   nir_block *post_inner = NULL;
   nir_block *merge_blk = NULL;

   if (!m->cl.bottom) {
      if (n_body != 4 || body_nodes[0]->type != nir_cf_node_block ||
          body_nodes[1]->type != nir_cf_node_block ||
          body_nodes[3]->type != nir_cf_node_block)
         return false;
      if (body_nodes[2]->type == nir_cf_node_loop) {
         pre_inner = nir_cf_node_as_block(body_nodes[1]);
         inner_top = nir_cf_node_as_loop(body_nodes[2]);
         post_inner = nir_cf_node_as_block(body_nodes[3]);
         merge_blk = post_inner;
      } else if (body_nodes[2]->type == nir_cf_node_if) {
         guard = nir_cf_node_as_if(body_nodes[2]);
         merge_blk = nir_cf_node_as_block(body_nodes[3]);
      } else {
         return false;
      }
   } else {
      if (n_body != 3 || body_nodes[0]->type != nir_cf_node_block ||
          body_nodes[2]->type != nir_cf_node_block)
         return false;
      if (body_nodes[1]->type == nir_cf_node_loop) {
         pre_inner = header;
         inner_top = nir_cf_node_as_loop(body_nodes[1]);
         post_inner = nir_cf_node_as_block(body_nodes[2]);
         merge_blk = post_inner;
      } else if (body_nodes[1]->type == nir_cf_node_if) {
         guard = nir_cf_node_as_if(body_nodes[1]);
         merge_blk = nir_cf_node_as_block(body_nodes[2]);
      } else {
         return false;
      }
   }

   if (guard) {
      struct exec_list *active = NULL;
      nir_block *empty_blk = NULL;
      if (list_is_empty_block(&guard->else_list)) {
         m->guard_in_then = true;
         active = &guard->then_list;
         empty_blk = nir_if_first_else_block(guard);
      } else if (list_is_empty_block(&guard->then_list)) {
         m->guard_in_then = false;
         active = &guard->else_list;
         empty_blk = nir_if_first_then_block(guard);
      } else {
         return false;
      }

      nir_cf_node *gnodes[4];
      unsigned ng = 0;
      foreach_list_typed(nir_cf_node, n, node, active) {
         if (ng >= 4)
            return false;
         gnodes[ng++] = n;
      }
      if (ng != 3 || gnodes[0]->type != nir_cf_node_block ||
          gnodes[1]->type != nir_cf_node_loop ||
          gnodes[2]->type != nir_cf_node_block)
         return false;
      pre_inner = nir_cf_node_as_block(gnodes[0]);
      inner_top = nir_cf_node_as_loop(gnodes[1]);
      post_inner = nir_cf_node_as_block(gnodes[2]);

      /* Header instructions (after phis) must be pure ALU depending only on
       * iv and outer invariants (not on outer reduction phis). */
      BITSET_WORD *non_iv_phis =
         calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
      for (unsigned i = 0; i < m->n_outer_phis; ++i) {
         if (i != m->iv_idx)
            BITSET_SET(non_iv_phis, m->outer_phis[i]->def.index);
      }
      bool hdr_ok = true;
      nir_foreach_instr(instr, header) {
         if (instr->type == nir_instr_type_phi)
            continue;
         if (!instr_is_pure_alu(instr) || instr_uses_bitset(instr, non_iv_phis)) {
            hdr_ok = false;
            break;
         }
      }
      if (!m->cl.bottom && hdr_ok) {
         nir_block *b1 = nir_cf_node_as_block(body_nodes[1]);
         nir_foreach_instr(instr, b1) {
            if (!instr_is_pure_alu(instr) || instr_uses_bitset(instr, non_iv_phis)) {
               hdr_ok = false;
               break;
            }
         }
      }
      free(non_iv_phis);
      if (!hdr_ok)
         return false;

      /* Verify merge_blk phis pass through outer phis unchanged from empty_blk
       * and take updated values from post_inner. */
      for (unsigned i = 0; i < m->n_outer_phis; ++i) {
         if (i == m->iv_idx)
            continue;
         nir_phi_src *back = nir_phi_get_src_from_block(m->outer_phis[i], latch);
         nir_def *bdef = back->src.ssa;
         if (nir_def_instr(bdef)->type != nir_instr_type_phi ||
             nir_def_block(bdef) != merge_blk)
            return false;
         nir_phi_instr *mphi = nir_instr_as_phi(nir_def_instr(bdef));
         nir_phi_src *esrc = nir_phi_get_src_from_block(mphi, empty_blk);
         nir_phi_src *asrc = nir_phi_get_src_from_block(mphi, post_inner);
         if (!esrc || !asrc || esrc->src.ssa != &m->outer_phis[i]->def)
            return false;
         m->red_updated_defs[i] = asrc->src.ssa;
      }
      unsigned n_mphis = 0;
      nir_foreach_instr(instr, merge_blk) {
         if (instr->type == nir_instr_type_phi) {
            n_mphis++;
            continue;
         }
         if (!instr_is_pure_alu(instr))
            return false;
      }
      if (n_mphis != m->n_outer_phis - 1)
         return false;
   } else {
      nir_foreach_instr(instr, header) {
         if (instr->type != nir_instr_type_phi && !instr_is_pure_alu(instr))
            return false;
      }
      for (unsigned i = 0; i < m->n_outer_phis; ++i) {
         if (i == m->iv_idx)
            continue;
         nir_phi_src *back = nir_phi_get_src_from_block(m->outer_phis[i], latch);
         m->red_updated_defs[i] = back->src.ssa;
      }
   }

   /* pre_inner and post_inner must be pure ALU (no loads, no phis). */
   if (pre_inner != header) {
      nir_foreach_instr(instr, pre_inner) {
         if (!instr_is_pure_alu(instr))
            return false;
      }
   }
   nir_foreach_instr(instr, post_inner) {
      if (instr->type == nir_instr_type_phi || !instr_is_pure_alu(instr))
         return false;
   }

   /* Validate inner loop nest (1- or 2-deep). */
   nir_loop *child = NULL;
   if (!validate_simple_loop(inner_top, true, &child))
      return false;
   nir_loop *inner_leaf = child ? child : inner_top;

   /* Compute lane variance from the outer IV phi alone. Outer reduction phis
    * are updated sequentially in post_inner after the jammed inner nest, so
    * reject the loop if any non-IV outer phi reaches header, pre_inner, the
    * outer guard, or anywhere inside the inner loop nest. */
   const unsigned bs_words = BITSET_WORDS(impl->ssa_alloc);
   BITSET_WORD *variant = calloc(bs_words, sizeof(BITSET_WORD));
   BITSET_WORD *red_dep = calloc(bs_words, sizeof(BITSET_WORD));
   BITSET_SET(variant, m->outer_phis[m->iv_idx]->def.index);
   for (unsigned i = 0; i < m->n_outer_phis; ++i) {
      if (i != m->iv_idx)
         BITSET_SET(red_dep, m->outer_phis[i]->def.index);
   }

   bool changed = true;
   while (changed) {
      changed = false;
      nir_foreach_block_in_cf_node(blk, &outer->cf_node) {
         nir_foreach_instr(instr, blk) {
            nir_def *def = nir_instr_def(instr);
            if (!def)
               continue;
            if (!BITSET_TEST(variant, def->index) &&
                instr_uses_bitset(instr, variant)) {
               BITSET_SET(variant, def->index);
               changed = true;
            }
            if (!BITSET_TEST(red_dep, def->index) &&
                instr_uses_bitset(instr, red_dep)) {
               BITSET_SET(red_dep, def->index);
               changed = true;
            }
         }
      }
   }

   bool red_used_early = false;
   nir_foreach_instr(instr, header) {
      if (instr->type != nir_instr_type_phi && instr_uses_bitset(instr, red_dep))
         red_used_early = true;
   }
   if (guard) {
      if (BITSET_TEST(red_dep, guard->condition.ssa->index))
         red_used_early = true;
      if (!m->cl.bottom) {
         nir_block *b1 = nir_cf_node_as_block(nir_cf_node_next(&m->cl.nif->cf_node));
         nir_foreach_instr(instr, b1) {
            if (instr_uses_bitset(instr, red_dep))
               red_used_early = true;
         }
      }
   }
   if (pre_inner != header) {
      nir_foreach_instr(instr, pre_inner) {
         if (instr_uses_bitset(instr, red_dep))
            red_used_early = true;
      }
   }
   nir_foreach_block_in_cf_node(blk, &inner_top->cf_node) {
      nir_foreach_instr(instr, blk) {
         if (instr_uses_bitset(instr, red_dep))
            red_used_early = true;
      }
      nir_cf_node *next = nir_cf_node_next(&blk->cf_node);
      if (next && next->type == nir_cf_node_if &&
          BITSET_TEST(red_dep, nir_cf_node_as_if(next)->condition.ssa->index))
         red_used_early = true;
   }
   free(red_dep);
   if (red_used_early) {
      free(variant);
      return false;
   }

   /* Check inner_top and inner_leaf control flow is strictly outer-invariant. */
   nir_loop *inners[2] = {inner_top, inner_leaf};
   unsigned n_inners = inner_leaf == inner_top ? 1 : 2;
   for (unsigned k = 0; k < n_inners; ++k) {
      nir_loop *lp = inners[k];
      nir_block *ihdr = nir_loop_first_block(lp);
      nir_block *ipre = nir_cf_node_as_block(nir_cf_node_prev(&lp->cf_node));
      foreach_list_typed(nir_cf_node, n, node, &lp->body) {
         if (n->type == nir_cf_node_if) {
            nir_if *term = nir_cf_node_as_if(n);
            if (BITSET_TEST(variant, term->condition.ssa->index)) {
               free(variant);
               return false;
            }
         }
      }
      unsigned n_acc = 0;
      nir_foreach_phi(phi, ihdr) {
         if (!BITSET_TEST(variant, phi->def.index))
            continue;
         n_acc++;
         nir_phi_src *esrc = nir_phi_get_src_from_block(phi, ipre);
         if (!esrc) {
            free(variant);
            return false;
         }
         if (BITSET_TEST(variant, esrc->src.ssa->index)) {
            if (k == 0 ||
                nir_def_instr(esrc->src.ssa)->type != nir_instr_type_phi ||
                nir_def_block(esrc->src.ssa) != nir_loop_first_block(inner_top)) {
               free(variant);
               return false;
            }
         }
      }
      if (n_acc == 0 || n_acc > 2) {
         free(variant);
         return false;
      }
   }

   /* Every variant value defined inside inner_top used outside inner_top must
    * be a header phi of inner_top, or defined in inner_top's header block when
    * inner_top is a bottom-tested single-body loop (so its header block
    * dominates the exit). */
   nir_block *itop_hdr = nir_loop_first_block(inner_top);
   nir_cf_node *itop_after_hdr = nir_cf_node_next(&itop_hdr->cf_node);
   bool itop_bottom_body =
      itop_after_hdr && itop_after_hdr->type == nir_cf_node_if &&
      nir_cf_node_next(itop_after_hdr) == &nir_loop_last_block(inner_top)->cf_node &&
      block_is_empty(nir_loop_last_block(inner_top));

   nir_foreach_block_in_cf_node(blk, &inner_top->cf_node) {
      nir_foreach_instr(instr, blk) {
         nir_def *def = nir_instr_def(instr);
         if (!def || !BITSET_TEST(variant, def->index))
            continue;
         nir_foreach_use_including_if(src, def) {
            if (!block_in_loop(use_block(src), inner_top)) {
               bool ok_phi = (instr->type == nir_instr_type_phi && blk == itop_hdr);
               bool ok_body = (itop_bottom_body && blk == itop_hdr);
               if (!ok_phi && !ok_body) {
                  free(variant);
                  return false;
               }
            }
         }
      }
   }

   /* Fixed heuristics on inner_leaf (<= 32 non-phi instructions, <= 4 total
    * loads, <= 2 inner accumulators above, paired with J=4 for 2D / J=2 for 1D):
    * empirically sized so a jammed leaf fits within the 64-GPR / 1024-thread
    * occupancy tier on AGX3; the retry loop in agx_compile_shader_nir falls
    * back to half-factor if register allocation exceeds the tier.
    *
    * Also admit an already-unrolled 1D sliding-window stencil inner loop
    * (inner_top == inner_leaf) when variant loads overlap across adjacent
    * outer-IV steps so diagonal CSE collapses J * W loads to W + J - 1. */
   unsigned inv_loads = 0, var_loads = 0, leaf_instrs = 0;
   nir_foreach_block_in_cf_node(blk, &inner_leaf->cf_node) {
      nir_foreach_instr(instr, blk) {
         if (instr->type != nir_instr_type_phi)
            leaf_instrs++;
         if (instr_is_readonly_load(instr)) {
            if (instr_uses_bitset(instr, variant))
               var_loads++;
            else
               inv_loads++;
         }
      }
   }
   bool small_leaf = (inv_loads + var_loads <= 4 && leaf_instrs <= 32);
   bool sliding_leaf =
      !small_leaf && inner_top == inner_leaf && inv_loads <= 24 &&
      var_loads >= 2 && var_loads <= 16 && leaf_instrs <= 400 &&
      has_sliding_window_var_loads(inner_leaf, &m->outer_phis[m->iv_idx]->def,
                                   m->cl.step, variant, var_loads);
   if (inv_loads == 0 || var_loads == 0 || (!small_leaf && !sliding_leaf)) {
      free(variant);
      return false;
   }

   if (m->cl.bottom) {
      struct live_out outs[16];
      unsigned nouts = 0;
      if (!back_edge_live_outs(outer, outs, &nouts)) {
         free(variant);
         return false;
      }
   }

   m->outer_guard = guard;
   m->pre_inner_block = pre_inner;
   m->inner_top = inner_top;
   m->inner_leaf = inner_leaf;
   m->post_inner_block = post_inner;
   m->merge_block = merge_blk;
   m->variant = variant;
   return true;
}

static nir_def *
remap_lookup(struct hash_table *remap, nir_def *def)
{
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   return he ? he->data : def;
}

struct jammed_phi_rec {
   nir_phi_instr *orig;
   nir_phi_instr *clone;
   int lane; /* -1 for shared (non-variant), 0..J-1 for replicated */
};

static void
emit_jammed_cf_list(nir_builder *b, struct exec_list *list, unsigned J,
                    struct hash_table **remaps, const BITSET_WORD *variant,
                    bool mark_unrolled);

static void
emit_jammed_loop(nir_builder *b, nir_loop *orig_loop, unsigned J,
                 struct hash_table **remaps, const BITSET_WORD *variant,
                 bool mark_unrolled)
{
   nir_block *orig_hdr = nir_loop_first_block(orig_loop);
   nir_block *orig_pre = nir_cf_node_as_block(nir_cf_node_prev(&orig_loop->cf_node));
   nir_block *orig_latch = nir_loop_last_block(orig_loop);

   nir_loop *new_loop = nir_push_loop(b);
   if (mark_unrolled)
      new_loop->partially_unrolled = true;
   nir_block *new_hdr = nir_loop_first_block(new_loop);

   struct util_dynarray phis;
   util_dynarray_init(&phis, NULL);

   /* Create header phis (1 for non-variant, J for variant). Defer insertion
    * until after nir_phi_instr_add_src so all phi srcs are registered on their
    * defs' use lists when inserted. */
   nir_foreach_phi(phi, orig_hdr) {
      if (!BITSET_TEST(variant, phi->def.index)) {
         nir_phi_instr *np = nir_phi_instr_create(b->shader);
         nir_def_init(&np->instr, &np->def, phi->def.num_components,
                      phi->def.bit_size);
         for (unsigned j = 0; j < J; ++j)
            _mesa_hash_table_insert(remaps[j], &phi->def, &np->def);
         util_dynarray_append(&phis, ((struct jammed_phi_rec){phi, np, -1}));
      } else {
         for (unsigned j = 0; j < J; ++j) {
            nir_phi_instr *np = nir_phi_instr_create(b->shader);
            nir_def_init(&np->instr, &np->def, phi->def.num_components,
                         phi->def.bit_size);
            _mesa_hash_table_insert(remaps[j], &phi->def, &np->def);
            util_dynarray_append(&phis, ((struct jammed_phi_rec){phi, np, (int)j}));
         }
      }
   }

   b->cursor = nir_after_block(new_hdr);
   emit_jammed_cf_list(b, &orig_loop->body, J, remaps, variant, mark_unrolled);

   nir_pop_loop(b, new_loop);

   nir_block *new_pre = nir_cf_node_as_block(nir_cf_node_prev(&new_loop->cf_node));
   nir_block *new_latch = nir_loop_last_block(new_loop);

   util_dynarray_foreach(&phis, struct jammed_phi_rec, rec) {
      unsigned lane = rec->lane >= 0 ? (unsigned)rec->lane : 0;
      nir_phi_src *esrc = nir_phi_get_src_from_block(rec->orig, orig_pre);
      nir_phi_src *lsrc = nir_phi_get_src_from_block(rec->orig, orig_latch);
      assert(esrc && lsrc);
      nir_phi_instr_add_src(rec->clone, new_pre,
                            remap_lookup(remaps[lane], esrc->src.ssa));
      nir_phi_instr_add_src(rec->clone, new_latch,
                            remap_lookup(remaps[lane], lsrc->src.ssa));
      nir_instr_insert(nir_after_phis(new_hdr), &rec->clone->instr);
   }
   util_dynarray_fini(&phis);
}

static void
emit_jammed_cf_list(nir_builder *b, struct exec_list *list, unsigned J,
                    struct hash_table **remaps, const BITSET_WORD *variant,
                    bool mark_unrolled)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_phi)
               continue;
            if (instr->type == nir_instr_type_jump) {
               nir_jump(b, nir_instr_as_jump(instr)->type);
               continue;
            }
            nir_def *def = nir_instr_def(instr);
            if (def && !BITSET_TEST(variant, def->index)) {
               nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remaps[0]);
               nir_builder_instr_insert(b, clone);
               nir_def *cdef = nir_instr_def(clone);
               for (unsigned j = 0; j < J; ++j)
                  _mesa_hash_table_insert(remaps[j], def, cdef);
            } else {
               for (unsigned j = 0; j < J; ++j) {
                  nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remaps[j]);
                  nir_builder_instr_insert(b, clone);
                  if (def)
                     _mesa_hash_table_insert(remaps[j], def, nir_instr_def(clone));
               }
            }
         }
      } else if (node->type == nir_cf_node_if) {
         nir_if *orig_if = nir_cf_node_as_if(node);
         nir_def *cond = remap_lookup(remaps[0], orig_if->condition.ssa);
         nir_if *nif = nir_push_if(b, cond);
         emit_jammed_cf_list(b, &orig_if->then_list, J, remaps, variant, mark_unrolled);
         nir_push_else(b, nif);
         emit_jammed_cf_list(b, &orig_if->else_list, J, remaps, variant, mark_unrolled);
         nir_pop_if(b, nif);
      } else if (node->type == nir_cf_node_loop) {
         emit_jammed_loop(b, nir_cf_node_as_loop(node), J, remaps, variant, mark_unrolled);
      }
   }
}

static void
clone_pure_block(nir_builder *b, nir_block *blk, bool skip_load_regs,
                 struct hash_table *remap)
{
   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_phi)
         continue;
      if (skip_load_regs && instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_reg)
         continue;
      if (!instr_is_pure_alu(instr))
         continue;
      nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
      nir_builder_instr_insert(b, clone);
      nir_def *def = nir_instr_def(instr);
      if (def)
         _mesa_hash_table_insert(remap, def, nir_instr_def(clone));
   }
}

static void
unroll_and_jam_loop(nir_function_impl *impl, struct uaj_match *m, unsigned J)
{
   struct counted_loop *cl = &m->cl;
   nir_loop *outer = cl->loop;
   nir_block *header = nir_loop_first_block(outer);
   nir_block *latch = nir_loop_last_block(outer);
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));

   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_after_block_before_jump(pre);

   /* Compute invariant bound and guard bound G in preheader. */
   struct hash_table *bremap = _mesa_pointer_hash_table_create(NULL);
   nir_def *bound =
      nir_scalar_is_const(cl->bound)
         ? nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(cl->bound))
         : nir_channel(&b, rematerialize(&b, cl->bound.def, outer, bremap),
                       cl->bound.comp);
   _mesa_hash_table_destroy(bremap, NULL);

   nir_def *g = guard_bound(&b, cl, bound, J);

   /* Collect live-outs and check exact_tail before lowering phis. */
   struct live_out outs[16];
   unsigned nouts = 0;
   if (cl->bottom) {
      bool have_outs = back_edge_live_outs(outer, outs, &nouts);
      assert(have_outs);
   }
   nir_phi_src *iv_init_src = nir_phi_get_src_from_block(cl->iv, pre);
   nir_phi_src *iv_back_src = nir_phi_get_src_from_block(cl->iv, latch);
   const bool exact_tail = cl->bottom && cl->offset == cl->step && iv_init_src &&
                           iv_back_src;
   nir_def *iv0 = exact_tail ? iv_init_src->src.ssa : NULL;

   /* Lower outer header phis to registers manually so we keep exact reg handles. */
   nir_def *outer_regs[8];
   nir_def *outer_phi_loads[8];
   nir_def *outer_back_defs[8];
   nir_builder b_decl = nir_builder_at(nir_before_impl(impl));

   for (unsigned i = 0; i < m->n_outer_phis; ++i) {
      nir_phi_instr *phi = m->outer_phis[i];
      outer_back_defs[i] = nir_phi_get_src_from_block(phi, latch)->src.ssa;
      outer_regs[i] = nir_decl_reg(&b_decl, phi->def.num_components,
                                   phi->def.bit_size, 0);
      b.cursor = nir_after_instr(&phi->instr);
      nir_def *ld = nir_load_reg(&b, outer_regs[i]);
      nir_def_rewrite_uses(&phi->def, ld);
      outer_phi_loads[i] = ld;
      for (unsigned k = 0; k < m->n_outer_phis; ++k) {
         if (m->red_updated_defs[k] == &phi->def)
            m->red_updated_defs[k] = ld;
      }
      nir_foreach_phi_src(src, phi) {
         b.cursor = nir_after_block_before_jump(src->pred);
         nir_store_reg(&b, src->src.ssa, outer_regs[i]);
      }
      nir_instr_remove(&phi->instr);
   }

   for (unsigned i = 0; i < nouts; ++i) {
      if (outs[i].invariant)
         continue;
      for (unsigned k = 0; k < m->n_outer_phis; ++k) {
         if (outer_back_defs[k] == outs[i].def) {
            outs[i].reg = outer_regs[k];
            break;
         }
      }
      assert(outs[i].reg);
   }

   /* Recompute variant bitset with the new load_reg SSA indices. */
   nir_index_ssa_defs(impl);
   BITSET_WORD *var2 =
      calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
   for (unsigned i = 0; i < m->n_outer_phis; ++i)
      BITSET_SET(var2, outer_phi_loads[i]->index);
   bool changed = true;
   while (changed) {
      changed = false;
      nir_foreach_block_in_cf_node(blk, &outer->cf_node) {
         nir_foreach_instr(instr, blk) {
            nir_def *def = nir_instr_def(instr);
            if (!def || BITSET_TEST(var2, def->index))
               continue;
            if (instr_uses_bitset(instr, var2)) {
               BITSET_SET(var2, def->index);
               changed = true;
            }
         }
      }
   }
   free(m->variant);
   m->variant = var2;

   /* Build main_loop directly in front of outer (after preheader store_regs). */
   b.cursor = nir_before_cf_node(&outer->cf_node);
   nir_loop *main_loop = nir_push_loop(&b);
   main_loop->partially_unrolled = true;
   outer->partially_unrolled = true;

   nir_def *d0 = nir_load_reg(&b, outer_regs[m->iv_idx]);
   nir_def *iv_test = d0;
   if (cl->offset && !exact_tail)
      iv_test = nir_iadd_imm(&b, iv_test, (uint32_t)cl->offset);
   nir_def *cont;
   if (cl->rel == REL_LT || cl->rel == REL_LE)
      cont = cl->is_unsigned ? nir_ult(&b, iv_test, g) : nir_ilt(&b, iv_test, g);
   else
      cont = cl->is_unsigned ? nir_ult(&b, g, iv_test) : nir_ilt(&b, g, iv_test);

   nir_if *exit_if = nir_push_if(&b, nir_inot(&b, cont));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, exit_if);

   nir_def *d_vals[8];
   d_vals[0] = d0;
   for (unsigned j = 1; j < J; ++j)
      d_vals[j] = nir_iadd_imm(&b, d0, (uint32_t)(j * cl->step));

   struct hash_table *remaps[8];
   for (unsigned j = 0; j < J; ++j) {
      remaps[j] = _mesa_pointer_hash_table_create(NULL);
      _mesa_hash_table_insert(remaps[j], outer_phi_loads[m->iv_idx], d_vals[j]);
   }

   /* Evaluate outer_guard condition for all J lanes if present. */
   nir_if *all_ok_if = NULL;
   nir_def *any_ok = NULL;
   if (m->outer_guard) {
      nir_def *all_ok = NULL;
      for (unsigned j = 0; j < J; ++j) {
         clone_pure_block(&b, header, true, remaps[j]);
         if (!m->cl.bottom) {
            nir_block *b1 = nir_cf_node_as_block(
               nir_cf_node_next(&m->cl.nif->cf_node));
            clone_pure_block(&b, b1, false, remaps[j]);
         }
         nir_def *c = remap_lookup(remaps[j], m->outer_guard->condition.ssa);
         if (!m->guard_in_then)
            c = nir_inot(&b, c);
         all_ok = all_ok ? nir_iand(&b, all_ok, c) : c;
         any_ok = any_ok ? nir_ior(&b, any_ok, c) : c;
      }
      all_ok_if = nir_push_if(&b, all_ok);
   } else {
      for (unsigned j = 0; j < J; ++j) {
         clone_pure_block(&b, header, true, remaps[j]);
         if (m->pre_inner_block != header)
            clone_pure_block(&b, m->pre_inner_block, false, remaps[j]);
      }
   }

   /* Fast path: emit ONE jammed inner loop nest + J sequential outer updates. */
   if (m->outer_guard && m->pre_inner_block != header) {
      for (unsigned j = 0; j < J; ++j)
         clone_pure_block(&b, m->pre_inner_block, false, remaps[j]);
   }

   emit_jammed_loop(&b, m->inner_top, J, remaps, m->variant, false);

   /* Sequential outer reduction updates j = 0 .. J - 1. */
   nir_def *cur_red[8] = {0};
   for (unsigned k = 0; k < m->n_outer_phis; ++k) {
      if (k != m->iv_idx)
         cur_red[k] = nir_load_reg(&b, outer_regs[k]);
   }
   for (unsigned j = 0; j < J; ++j) {
      for (unsigned k = 0; k < m->n_outer_phis; ++k) {
         if (k != m->iv_idx)
            _mesa_hash_table_insert(remaps[j], outer_phi_loads[k], cur_red[k]);
      }
      clone_pure_block(&b, m->post_inner_block, false, remaps[j]);
      for (unsigned k = 0; k < m->n_outer_phis; ++k) {
         if (k != m->iv_idx)
            cur_red[k] = remap_lookup(remaps[j], m->red_updated_defs[k]);
      }
   }
   for (unsigned k = 0; k < m->n_outer_phis; ++k) {
      if (k != m->iv_idx)
         nir_store_reg(&b, cur_red[k], outer_regs[k]);
   }

   if (all_ok_if) {
      /* Slow path (partial boundary group): if any lane passes, run a compact
       * J-iteration scalar fallback loop with a single copy of inner_top. */
      nir_push_else(&b, all_ok_if);
      nir_if *any_if = nir_push_if(&b, any_ok);
      {
         nir_def *k_reg = nir_decl_reg(&b_decl, 1, 32, 0);
         nir_store_reg(&b, nir_imm_int(&b, 0), k_reg);
         nir_loop *fb_loop = nir_push_loop(&b);
         fb_loop->partially_unrolled = true;
         nir_def *k = nir_load_reg(&b, k_reg);
         nir_if *k_exit = nir_push_if(&b, nir_ige_imm(&b, k, J));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, k_exit);

         nir_def *d_k = cl->step == 1 ? nir_iadd(&b, d0, k) : nir_isub(&b, d0, k);
         struct hash_table *fmap = _mesa_pointer_hash_table_create(NULL);
         _mesa_hash_table_insert(fmap, outer_phi_loads[m->iv_idx], d_k);
         for (unsigned r = 0; r < m->n_outer_phis; ++r) {
            if (r != m->iv_idx)
               _mesa_hash_table_insert(fmap, outer_phi_loads[r],
                                       nir_load_reg(&b, outer_regs[r]));
         }

         clone_pure_block(&b, header, true, fmap);
         if (!m->cl.bottom) {
            nir_block *b1 = nir_cf_node_as_block(
               nir_cf_node_next(&m->cl.nif->cf_node));
            clone_pure_block(&b, b1, false, fmap);
         }
         nir_def *ck = remap_lookup(fmap, m->outer_guard->condition.ssa);
         if (!m->guard_in_then)
            ck = nir_inot(&b, ck);
         nir_if *ck_if = nir_push_if(&b, ck);
         {
            if (m->pre_inner_block != header)
               clone_pure_block(&b, m->pre_inner_block, false, fmap);

            /* Clone inner_top once, marking it partially_unrolled so the rare
             * boundary fallback does not bloat code size. */
            emit_jammed_loop(&b, m->inner_top, 1, &fmap, m->variant, true);

            clone_pure_block(&b, m->post_inner_block, false, fmap);
            for (unsigned r = 0; r < m->n_outer_phis; ++r) {
               if (r != m->iv_idx)
                  nir_store_reg(&b, remap_lookup(fmap, m->red_updated_defs[r]),
                                outer_regs[r]);
            }
         }
         nir_pop_if(&b, ck_if);
         _mesa_hash_table_destroy(fmap, NULL);

         nir_store_reg(&b, nir_iadd_imm(&b, k, 1), k_reg);
         nir_pop_loop(&b, fb_loop);
      }
      nir_pop_if(&b, any_if);
      nir_pop_if(&b, all_ok_if);
   }

   for (unsigned j = 0; j < J; ++j)
      _mesa_hash_table_destroy(remaps[j], NULL);

   nir_store_reg(&b, nir_iadd_imm(&b, d0, (uint32_t)(J * cl->step)),
                 outer_regs[m->iv_idx]);
   nir_pop_loop(&b, main_loop);

   /* Keep tail inner loop compact (at most J - 1 outer iterations). */
   m->inner_top->partially_unrolled = true;
   m->inner_leaf->partially_unrolled = true;

   /* Guard bottom-tested tail loop when exact_tail applies. */
   if (exact_tail) {
      b.cursor = nir_before_cf_node(&outer->cf_node);
      nir_def *v = nir_load_reg(&b, outer_regs[m->iv_idx]);
      const bool u = cl->is_unsigned;
      nir_def *more;
      switch (cl->rel) {
      case REL_LT: more = u ? nir_ult(&b, v, bound) : nir_ilt(&b, v, bound); break;
      case REL_LE: more = u ? nir_uge(&b, bound, v) : nir_ige(&b, bound, v); break;
      case REL_GT: more = u ? nir_ult(&b, bound, v) : nir_ilt(&b, bound, v); break;
      default:     more = u ? nir_uge(&b, v, bound) : nir_ige(&b, v, bound); break;
      }
      nir_def *run = nir_ior(&b, nir_ieq(&b, v, iv0), more);
      nir_if *tail = nir_push_if(&b, run);
      nir_pop_if(&b, tail);
      nir_cf_list moved;
      nir_cf_extract(&moved, nir_before_cf_node(&outer->cf_node),
                     nir_after_cf_node(&outer->cf_node));
      nir_cf_reinsert(&moved, nir_before_cf_list(&tail->then_list));

      b.cursor = nir_after_cf_node(&tail->cf_node);
      struct hash_table *again = _mesa_pointer_hash_table_create(NULL);
      for (unsigned i = 0; i < nouts; ++i) {
         nir_def *value = outs[i].invariant
                             ? rematerialize(&b, outs[i].def, outer, again)
                             : nir_load_reg(&b, outs[i].reg);
         nir_foreach_use_including_if_safe(src, outs[i].def) {
            if (!block_in_loop(use_block(src), outer))
               nir_src_rewrite(src, value);
         }
      }
      _mesa_hash_table_destroy(again, NULL);

      b.cursor = nir_after_cf_node(&outer->cf_node);
      for (unsigned i = 0; i < nouts; ++i) {
         if (!outs[i].invariant)
            nir_store_reg(&b, outs[i].def, outs[i].reg);
      }
   }

   free(m->variant);
   m->variant = NULL;
}

static void
collect_loops_outer_first(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops_outer_first(&nif->then_list, loops);
         collect_loops_outer_first(&nif->else_list, loops);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(node);
         util_dynarray_append(loops, lp);
         collect_loops_outer_first(&lp->body, loops);
      }
   }
}

bool
agx_nir_unroll_and_jam(nir_shader *shader, unsigned factor, unsigned *widest)
{
   const unsigned J = factor >= 8 ? 4 : 2;
   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = false;
      nir_metadata_require(impl, nir_metadata_block_index);
      nir_index_ssa_defs(impl);

      struct util_dynarray loops;
      util_dynarray_init(&loops, NULL);
      collect_loops_outer_first(&impl->body, &loops);

      util_dynarray_foreach(&loops, nir_loop *, lp) {
         struct uaj_match m;
         if (!match_unroll_and_jam(impl, *lp, &m))
            continue;
         unroll_and_jam_loop(impl, &m, J);
         if (widest && J >= 4 && *widest < 8)
            *widest = 8;
         impl_progress = true;
         any = true;
         /* One outer loop per pass iteration; keep CF indices fresh. */
         break;
      }

      util_dynarray_fini(&loops);

      if (impl_progress) {
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
      } else {
         nir_no_progress(impl);
      }
   }

   return any;
}

/* ============================================================================
 * Bit-exact FP32-blocked u8 absolute-difference accumulation
 *
 * On AGX, FP32 FADD supports native .abs and .neg modifiers on both sources,
 * and AGX_CONVERT_U8_TO_F converts unsigned bytes directly to f32 (dropping
 * low-byte masks). When a basic block accumulates chains of K >= 4 u8
 * absolute differences |u2u32(a) - u2u32(b)| into u32 accumulators AND the
 * underlying u8 operands are reused across terms (average reuse >= 1.5x, as
 * after outer-loop unroll-and-jam or multi-kernel correlation), converting
 * each unique u8 operand once via u2f32, forming fsub + balanced fadd(fabs,
 * fabs) trees, and flushing once via f2u32 + iadd is 100% bit-exact for
 * K <= 64 (max sum 16320 << 2^24) and cuts inner-loop ALU nearly in half.
 * ============================================================================
 */

static bool
match_u2u32_u8(nir_scalar s, nir_def **u8_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_u2u32)
      return false;
   nir_scalar src = nir_scalar_chase_alu_src(s, 0);
   if (src.def->num_components != 1)
      return false;
   if (src.def->bit_size == 8) {
      *u8_out = src.def;
      return true;
   }
   if (src.def->bit_size == 16 &&
       nir_def_instr(src.def)->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr =
         nir_instr_as_intrinsic(nir_def_instr(src.def));
      if ((intr->intrinsic == nir_intrinsic_load_agx ||
           intr->intrinsic == nir_intrinsic_load_constant_agx) &&
          nir_intrinsic_format(intr) == PIPE_FORMAT_R8_UINT &&
          !nir_intrinsic_sign_extend(intr)) {
         *u8_out = src.def;
         return true;
      }
   }
   return false;
}

static bool
match_u8_diff(nir_scalar s, nir_def **a_out, nir_def **b_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || s.def->bit_size != 32 ||
       s.def->num_components != 1)
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_isub) {
      return match_u2u32_u8(nir_scalar_chase_alu_src(s, 0), a_out) &&
             match_u2u32_u8(nir_scalar_chase_alu_src(s, 1), b_out);
   }
   if (op == nir_op_iadd) {
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_alu(s1) && nir_scalar_alu_op(s1) == nir_op_ineg) {
         return match_u2u32_u8(s0, a_out) &&
                match_u2u32_u8(nir_scalar_chase_alu_src(s1, 0), b_out);
      }
      if (nir_scalar_is_alu(s0) && nir_scalar_alu_op(s0) == nir_op_ineg) {
         return match_u2u32_u8(s1, a_out) &&
                match_u2u32_u8(nir_scalar_chase_alu_src(s0, 0), b_out);
      }
   }
   return false;
}

static bool
match_u8_absdiff(nir_def *def, nir_block *block, nir_def **a_out, nir_def **b_out)
{
   if (def->bit_size != 32 || def->num_components != 1 ||
       nir_def_block(def) != block || !list_is_singular(&def->uses))
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op == nir_op_iabs) {
      nir_scalar src = nir_scalar_chase_movs(nir_get_scalar(alu->src[0].src.ssa,
                                                            alu->src[0].swizzle[0]));
      return list_is_singular(&src.def->uses) &&
             match_u8_diff(src, a_out, b_out);
   }
   if (alu->op == nir_op_imax) {
      nir_scalar x = nir_scalar_chase_movs(nir_get_scalar(alu->src[0].src.ssa,
                                                          alu->src[0].swizzle[0]));
      nir_scalar y = nir_scalar_chase_movs(nir_get_scalar(alu->src[1].src.ssa,
                                                          alu->src[1].swizzle[0]));
      nir_scalar neg = (nir_scalar_is_alu(x) && nir_scalar_alu_op(x) == nir_op_ineg) ? x : y;
      nir_scalar delta = (neg.def == x.def) ? y : x;
      if (!nir_scalar_is_alu(neg) || nir_scalar_alu_op(neg) != nir_op_ineg ||
          neg.def == delta.def)
         return false;
      nir_scalar nsrc = nir_scalar_chase_alu_src(neg, 0);
      if (nsrc.def != delta.def || nsrc.comp != delta.comp ||
          !list_is_singular(&neg.def->uses) ||
          list_length(&delta.def->uses) != 2)
         return false;
      return match_u8_diff(delta, a_out, b_out);
   }
   return false;
}

struct sad_chain {
   nir_alu_instr *end;
   nir_def *base;
   unsigned len;
   nir_def *a[64];
   nir_def *b[64];
};

static bool
is_absdiff_add(nir_alu_instr *alu, nir_block *block, nir_def **prev_out,
               nir_def **a_out, nir_def **b_out)
{
   if (alu->op != nir_op_iadd || alu->def.bit_size != 32 ||
       alu->def.num_components != 1)
      return false;
   nir_def *s0 = alu->src[0].src.ssa;
   nir_def *s1 = alu->src[1].src.ssa;
   if (match_u8_absdiff(s1, block, a_out, b_out)) {
      *prev_out = s0;
      return true;
   }
   if (match_u8_absdiff(s0, block, a_out, b_out)) {
      *prev_out = s1;
      return true;
   }
   return false;
}

static nir_def *
get_or_emit_u2f32(nir_function_impl *impl, nir_block *block,
                  struct hash_table *u2f_map, nir_def *u8)
{
   struct hash_entry *he = _mesa_hash_table_search(u2f_map, u8);
   if (he)
      return he->data;

   nir_builder b = nir_builder_create(impl);
   if (nir_def_block(u8) == block)
      b.cursor = nir_after_instr(nir_def_instr(u8));
   else
      b.cursor = nir_before_block_after_phis(block);
   nir_def *f = nir_u2f32(&b, u8);
   _mesa_hash_table_insert(u2f_map, u8, f);
   return f;
}

static nir_def *
emit_balanced_fadd_tree(nir_builder *b, nir_def **vals, unsigned n)
{
   assert(n >= 1);
   if (n == 1)
      return vals[0];
   unsigned mid = n / 2;
   nir_def *l = emit_balanced_fadd_tree(b, vals, mid);
   nir_def *r = emit_balanced_fadd_tree(b, vals + mid, n - mid);
   return nir_fadd(b, l, r);
}

static bool
opt_u8_absdiff_fp_block(nir_function_impl *impl, nir_block *block)
{
   struct util_dynarray chains;
   util_dynarray_init(&chains, NULL);
   struct set *visited = _mesa_pointer_set_create(NULL);

   nir_foreach_instr_reverse(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (_mesa_set_search(visited, alu))
         continue;

      nir_def *prev, *a, *b;
      if (!is_absdiff_add(alu, block, &prev, &a, &b))
         continue;

      struct sad_chain ch = {.end = alu};
      nir_alu_instr *cur = alu;
      while (ch.len < 64) {
         if (!is_absdiff_add(cur, block, &prev, &a, &b))
            break;
         _mesa_set_add(visited, cur);
         ch.a[ch.len] = a;
         ch.b[ch.len] = b;
         ch.len++;
         ch.base = prev;

         if (nir_def_block(prev) != block || !list_is_singular(&prev->uses) ||
             nir_def_instr(prev)->type != nir_instr_type_alu)
            break;
         nir_alu_instr *palu = nir_instr_as_alu(nir_def_instr(prev));
         if (_mesa_set_search(visited, palu))
            break;
         cur = palu;
      }

      if (ch.len >= 4)
         util_dynarray_append(&chains, ch);
   }
   _mesa_set_destroy(visited, NULL);

   if (util_dynarray_num_elements(&chains, struct sad_chain) == 0) {
      util_dynarray_fini(&chains);
      return false;
   }

   /* Profitability check: require average u8 operand reuse >= 1.5x so the
    * one-time u2f32 conversions amortize across multiple absdiffs. */
   struct set *unique_u8 = _mesa_pointer_set_create(NULL);
   unsigned total_refs = 0;
   util_dynarray_foreach(&chains, struct sad_chain, ch) {
      for (unsigned i = 0; i < ch->len; ++i) {
         _mesa_set_add(unique_u8, ch->a[i]);
         _mesa_set_add(unique_u8, ch->b[i]);
         total_refs += 2;
      }
   }
   unsigned n_unique = unique_u8->entries;
   _mesa_set_destroy(unique_u8, NULL);

   if (total_refs * 2 < n_unique * 3) {
      util_dynarray_fini(&chains);
      return false;
   }

   struct hash_table *u2f_map = _mesa_pointer_hash_table_create(NULL);
   nir_builder b = nir_builder_create(impl);

   util_dynarray_foreach(&chains, struct sad_chain, ch) {
      b.cursor = nir_before_instr(&ch->end->instr);
      nir_def *abs_vals[64];
      /* Emit in forward order (ch->a[len-1] .. ch->a[0]). */
      for (unsigned i = 0; i < ch->len; ++i) {
         unsigned idx = ch->len - 1 - i;
         nir_def *fa = get_or_emit_u2f32(impl, block, u2f_map, ch->a[idx]);
         nir_def *fb = get_or_emit_u2f32(impl, block, u2f_map, ch->b[idx]);
         abs_vals[i] = nir_fabs(&b, nir_fsub(&b, fa, fb));
      }
      nir_def *sum_f = emit_balanced_fadd_tree(&b, abs_vals, ch->len);
      nir_def *sum_u = nir_f2u32(&b, sum_f);
      nir_def *res = nir_iadd(&b, ch->base, sum_u);
      nir_def_replace(&ch->end->def, res);
   }

   _mesa_hash_table_destroy(u2f_map, NULL);
   util_dynarray_fini(&chains);
   return true;
}

bool
agx_nir_opt_u8_absdiff_fp(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = false;
      nir_foreach_block(block, impl) {
         impl_progress |= opt_u8_absdiff_fp_block(impl, block);
      }
      nir_progress(impl_progress, impl, nir_metadata_control_flow);
      progress |= impl_progress;
   }
   return progress;
}
