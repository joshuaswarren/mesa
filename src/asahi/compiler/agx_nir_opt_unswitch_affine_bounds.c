/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Unswitch counted 1D/2D loops on overflow-safe affine range predicates.
 *
 * Convolution, stencil, and image-filter kernels commonly loop over a runtime
 * window (e.g. kh in [0, kh_bound), kw in [0, kw_bound)) and guard every tap
 * with an affine bounds check such as:
 *
 *    iy = oy * stride_y + kh * dil_y - pad_y;
 *    if (iy >= 0 && iy < in_h) {
 *       for (kw = 0; kw < kernel_w; kw++) {
 *          ix = ox * stride_w + kw * dil_w - pad_w;
 *          if (ix >= 0 && ix < in_w) { ... }
 *       }
 *    }
 *
 * Keeping the bounds check inside the inner loop forces per-tap control flow
 * (barriers to load/ALU overlap) or costly OOB address redirection. For the
 * interior of the domain (typically >90% of invocations), all iterations stay
 * in bounds. This pass computes an exact, overflow-free uniform-assisted guard
 * before the loop:
 *
 *    if (all_in_bounds) {
 *       // Cloned loop with matched bounds checks folded to constants;
 *       // eligible for straight-line runtime unrolling below.
 *    } else {
 *       // Original border loop (marked dont_unroll to keep code compact).
 *    }
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

#define MAX_NEST 2
#define MAX_CHECKS 8
#define MAX_LOOP_INSTRS 160

struct simple_loop {
   nir_loop *loop;
   nir_if *terminator;
   nir_phi_instr *iv;
   nir_scalar bound;
   bool bottom_exit;
};

struct bounds_check {
   nir_def *def;
   nir_scalar val;
   nir_scalar limit;
   unsigned loop_idx;
   bool replacement;
};

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
   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          block_is_lone_break(nir_cf_node_as_block(node));
}

static bool
branch_is_empty(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(node)->instr_list);
}

static bool
node_inside_cf(const nir_cf_node *node, const nir_cf_node *ancestor)
{
   for (const nir_cf_node *cur = node; cur; cur = cur->parent) {
      if (cur == ancestor)
         return true;
   }
   return false;
}

static bool
def_outside_loop(nir_def *def, nir_loop *loop)
{
   return !node_inside_cf(&nir_def_block(def)->cf_node, &loop->cf_node);
}

static nir_loop *
enclosing_loop(nir_block *block)
{
   for (nir_cf_node *cur = block->cf_node.parent; cur; cur = cur->parent) {
      if (cur->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(cur);
   }
   return NULL;
}

/* Safe to recompute before outer without introducing any fault or side
 * effect: definitions outside outer, constants, preamble/push-constant
 * loads, speculatable reorderable reads, pure ALU trees, and reorderable
 * loads in the outer header block. */
static bool
safe_invariant_def_cached(nir_def *def, nir_loop *outer, unsigned depth,
                          struct hash_table *cache)
{
   if (def_outside_loop(def, outer))
      return true;
   if (cache) {
      struct hash_entry *he = _mesa_hash_table_search(cache, def);
      if (he)
         return (uintptr_t)he->data == 2;
   }
   if (depth == 0)
      return false;
   if (cache)
      _mesa_hash_table_insert(cache, def, (void *)(uintptr_t)1);

   bool ok = false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      ok = true;
      break;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      ok = true;
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
         if (!safe_invariant_def_cached(alu->src[i].src.ssa, outer, depth - 1, cache)) {
            ok = false;
            break;
         }
      }
      break;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      if (!info->has_dest)
         break;
      const bool speculatable_read =
         nir_intrinsic_can_reorder(intr) && nir_intrinsic_has_access(intr) &&
         (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE);
      const bool always_safe =
         intr->intrinsic == nir_intrinsic_load_preamble ||
         intr->intrinsic == nir_intrinsic_load_push_constant ||
         (info->num_srcs == 0 && nir_intrinsic_can_reorder(intr)) ||
         speculatable_read;
      const bool header_safe =
         instr->block == nir_loop_first_block(outer) &&
         nir_intrinsic_can_reorder(intr);
      if (!always_safe && !header_safe)
         break;
      ok = true;
      for (unsigned i = 0; i < info->num_srcs; ++i) {
         if (!safe_invariant_def_cached(intr->src[i].ssa, outer, depth - 1, cache)) {
            ok = false;
            break;
         }
      }
      break;
   }
   default:
      ok = false;
      break;
   }

   if (cache)
      _mesa_hash_table_insert(cache, def, (void *)(uintptr_t)(ok ? 2 : 1));
   return ok;
}

static bool
safe_invariant_def(nir_def *def, nir_loop *outer, unsigned depth)
{
   struct hash_table *cache = _mesa_pointer_hash_table_create(NULL);
   bool ok = safe_invariant_def_cached(def, outer, depth, cache);
   _mesa_hash_table_destroy(cache, NULL);
   return ok;
}

static bool
safe_invariant(nir_scalar s, nir_loop *outer, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   return safe_invariant_def(s.def, outer, depth);
}

static nir_def *
rematerialize_def(nir_builder *b, nir_def *def, nir_loop *outer,
                  struct hash_table *remap)
{
   if (def_outside_loop(def, outer))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         rematerialize_def(b, alu->src[i].src.ssa, outer, remap);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         rematerialize_def(b, intr->src[i].ssa, outer, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *res = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, res);
   return res;
}

static nir_def *
rematerialize_scalar(nir_builder *b, nir_scalar s, nir_loop *outer,
                     struct hash_table *remap)
{
   s = nir_scalar_chase_movs(s);
   nir_def *d = rematerialize_def(b, s.def, outer, remap);
   return d->num_components == 1 ? d : nir_channel(b, d, s.comp);
}

static bool
lane_dependent_or_barrier(const nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_infos[intr->intrinsic].flags &
       (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP))
      return true;
   switch (intr->intrinsic) {
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
      return true;
   default:
      return false;
   }
}

static bool
is_const_scalar(nir_scalar s, int64_t val)
{
   s = nir_scalar_chase_movs(s);
   return nir_scalar_is_const(s) && nir_scalar_as_int(s) == val;
}

/* Match iv = phi(preheader: 0, latch: iv + 1) and single top or bottom exit
 * breaking when test_val >= bound (with uniform 32-bit bound). */
static bool
match_simple_loop(nir_loop *loop, nir_loop *outer, struct simple_loop *out)
{
   if (nir_loop_has_continue_construct(loop) || loop->partially_unrolled ||
       loop->control == nir_loop_control_dont_unroll)
      return false;

   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);

   nir_if *term = NULL;
   bool break_in_then = false;
   bool bottom_exit = false;

   nir_cf_node *after_header = nir_cf_node_next(&header->cf_node);
   if (after_header && after_header->type == nir_cf_node_if) {
      nir_if *nif = nir_cf_node_as_if(after_header);
      if (branch_is_lone_break(&nif->then_list) && branch_is_empty(&nif->else_list)) {
         term = nif;
         break_in_then = true;
         bottom_exit = false;
      } else if (branch_is_lone_break(&nif->else_list) && branch_is_empty(&nif->then_list)) {
         term = nif;
         break_in_then = false;
         bottom_exit = false;
      }
   }

   if (!term) {
      nir_cf_node *before_latch = nir_cf_node_prev(&latch->cf_node);
      if (before_latch && before_latch->type == nir_cf_node_if &&
          exec_list_is_empty(&latch->instr_list)) {
         nir_if *nif = nir_cf_node_as_if(before_latch);
         if (branch_is_lone_break(&nif->then_list) && branch_is_empty(&nif->else_list)) {
            term = nif;
            break_in_then = true;
            bottom_exit = true;
         } else if (branch_is_lone_break(&nif->else_list) && branch_is_empty(&nif->then_list)) {
            term = nif;
            break_in_then = false;
            bottom_exit = true;
         }
      }
   }

   if (!term)
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (enclosing_loop(block) != loop)
         continue;
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump) {
            if (block->cf_node.parent != &term->cf_node)
               return false;
         }
      }
   }

   nir_scalar cond = nir_scalar_resolved(term->condition.ssa, 0);
   bool want_true = break_in_then;
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      cond = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
      want_true = !want_true;
   }
   if (!nir_scalar_is_alu(cond))
      return false;

   nir_op eff = nir_scalar_alu_op(cond);
   if (!want_true) {
      switch (eff) {
      case nir_op_ilt: eff = nir_op_ige; break;
      case nir_op_ige: eff = nir_op_ilt; break;
      default: return false;
      }
   }
   if (eff != nir_op_ige)
      return false;

   nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
      return false;

   if (s1.def->divergent || !safe_invariant(s1, outer, 10))
      return false;

   nir_phi_instr *iv = NULL;
   nir_foreach_phi(phi, header) {
      if (phi->def.num_components != 1 || phi->def.bit_size != 32 ||
          phi->def.divergent)
         continue;
      if (exec_list_length(&phi->srcs) != 2)
         continue;
      nir_def *init_val = NULL, *latch_val = NULL;
      nir_foreach_phi_src(psrc, phi) {
         if (psrc->pred == latch)
            latch_val = psrc->src.ssa;
         else if (!node_inside_cf(&psrc->pred->cf_node, &loop->cf_node))
            init_val = psrc->src.ssa;
      }
      if (!init_val || !latch_val || !is_const_scalar(nir_get_scalar(init_val, 0), 0))
         continue;
      nir_scalar ls = nir_scalar_resolved(latch_val, 0);
      if (!nir_scalar_is_alu(ls) || nir_scalar_alu_op(ls) != nir_op_iadd)
         continue;
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(ls, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(ls, 1));
      nir_scalar iv_s = nir_get_scalar(&phi->def, 0);
      bool matches_inc =
         (nir_scalar_equal(a, iv_s) && is_const_scalar(b, 1)) ||
         (nir_scalar_equal(b, iv_s) && is_const_scalar(a, 1));
      if (!matches_inc)
         continue;

      if ((!bottom_exit && nir_scalar_equal(s0, iv_s)) ||
          (bottom_exit && nir_scalar_equal(s0, ls))) {
         iv = phi;
         break;
      }
   }

   if (!iv)
      return false;

   *out = (struct simple_loop){
      .loop = loop,
      .terminator = term,
      .iv = iv,
      .bound = s1,
      .bottom_exit = bottom_exit,
   };
   return true;
}

static bool
has_outside_use(nir_def *def, nir_loop *loop)
{
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src)) {
         if (!node_inside_cf(&nir_src_use_if(src)->cf_node, &loop->cf_node))
            return true;
      } else {
         if (!node_inside_cf(&nir_src_use_instr(src)->block->cf_node, &loop->cf_node))
            return true;
      }
   }
   return false;
}

static bool
loop_tree_safe(nir_loop *outer)
{
   unsigned count = 0;
   nir_foreach_block_in_cf_node(block, &outer->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            if (lane_dependent_or_barrier(nir_instr_as_intrinsic(instr)))
               return false;
         } else if (instr->type == nir_instr_type_tex) {
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
         } else if (instr->type == nir_instr_type_call ||
                    instr->type == nir_instr_type_deref) {
            return false;
         }

         /* Keep live-outs limited to ALU/const/undef/phi (handles/barycentrics
          * must not pass through new phis). */
         nir_def *def = nir_instr_def(instr);
         if (def && instr->type != nir_instr_type_alu &&
             instr->type != nir_instr_type_load_const &&
             instr->type != nir_instr_type_undef &&
             instr->type != nir_instr_type_phi &&
             has_outside_use(def, outer))
            return false;

         if (instr->type != nir_instr_type_phi) {
            if (++count > MAX_LOOP_INSTRS)
               return false;
         }
      }
   }
   return true;
}

static int
analyze_affine_cached(nir_scalar s, const struct simple_loop *loops, unsigned nloops,
                      nir_loop *outer, unsigned depth,
                      struct hash_table *aff_cache, struct hash_table *inv_cache)
{
   s = nir_scalar_chase_movs(s);
   if (s.def->bit_size != 32 || depth == 0)
      return -2;

   if (s.comp == 0) {
      struct hash_entry *he = _mesa_hash_table_search(aff_cache, s.def);
      if (he)
         return (int)((intptr_t)he->data) - 4;
   }

   for (unsigned i = 0; i < nloops; ++i) {
      if (s.def == &loops[i].iv->def && s.comp == 0) {
         _mesa_hash_table_insert(aff_cache, s.def, (void *)(intptr_t)((int)i + 4));
         return (int)i;
      }
   }

   if (safe_invariant_def_cached(s.def, outer, 10, inv_cache)) {
      if (s.comp == 0)
         _mesa_hash_table_insert(aff_cache, s.def, (void *)(intptr_t)(-1 + 4));
      return -1;
   }

   if (!nir_scalar_is_alu(s)) {
      if (s.comp == 0)
         _mesa_hash_table_insert(aff_cache, s.def, (void *)(intptr_t)(-2 + 4));
      return -2;
   }

   if (s.comp == 0)
      _mesa_hash_table_insert(aff_cache, s.def, (void *)(intptr_t)(-2 + 4));

   int res = -2;
   switch (nir_scalar_alu_op(s)) {
   case nir_op_iadd:
   case nir_op_isub: {
      int l0 = analyze_affine_cached(nir_scalar_chase_alu_src(s, 0), loops, nloops,
                                     outer, depth - 1, aff_cache, inv_cache);
      int l1 = analyze_affine_cached(nir_scalar_chase_alu_src(s, 1), loops, nloops,
                                     outer, depth - 1, aff_cache, inv_cache);
      if (l0 == -2 || l1 == -2)
         res = -2;
      else if (l0 == -1)
         res = l1;
      else if (l1 == -1 || l0 == l1)
         res = l0;
      break;
   }
   case nir_op_ineg:
      res = analyze_affine_cached(nir_scalar_chase_alu_src(s, 0), loops, nloops,
                                  outer, depth - 1, aff_cache, inv_cache);
      break;
   case nir_op_imul: {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      int l0 = analyze_affine_cached(a, loops, nloops, outer, depth - 1,
                                     aff_cache, inv_cache);
      int l1 = analyze_affine_cached(b, loops, nloops, outer, depth - 1,
                                     aff_cache, inv_cache);
      if (l0 == -2 || l1 == -2)
         res = -2;
      else if (l0 == -1 && l1 == -1)
         res = -1;
      else if (l0 == -1 && !a.def->divergent)
         res = l1;
      else if (l1 == -1 && !b.def->divergent)
         res = l0;
      break;
   }
   case nir_op_ishl: {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (!nir_scalar_is_const(sh) || nir_scalar_as_uint(sh) >= 31)
         res = -2;
      else
         res = analyze_affine_cached(a, loops, nloops, outer, depth - 1,
                                     aff_cache, inv_cache);
      break;
   }
   default:
      res = -2;
      break;
   }

   if (s.comp == 0)
      _mesa_hash_table_insert(aff_cache, s.def, (void *)(intptr_t)(res + 4));
   return res;
}

static int
analyze_affine(nir_scalar s, const struct simple_loop *loops, unsigned nloops,
               nir_loop *outer, unsigned depth)
{
   struct hash_table *aff_cache = _mesa_pointer_hash_table_create(NULL);
   struct hash_table *inv_cache = _mesa_pointer_hash_table_create(NULL);
   int res = analyze_affine_cached(s, loops, nloops, outer, depth, aff_cache, inv_cache);
   _mesa_hash_table_destroy(aff_cache, NULL);
   _mesa_hash_table_destroy(inv_cache, NULL);
   return res;
}

static void
emit_affine(nir_builder *b, nir_scalar s, const struct simple_loop *loops,
            unsigned loop_idx, nir_loop *outer, struct hash_table *remap,
            nir_def **out_base, nir_def **out_step)
{
   s = nir_scalar_chase_movs(s);
   if (s.def == &loops[loop_idx].iv->def && s.comp == 0) {
      *out_base = nir_imm_int(b, 0);
      *out_step = nir_imm_int(b, 1);
      return;
   }
   if (safe_invariant(s, outer, 10)) {
      *out_base = rematerialize_scalar(b, s, outer, remap);
      *out_step = nir_imm_int(b, 0);
      return;
   }
   switch (nir_scalar_alu_op(s)) {
   case nir_op_iadd: {
      nir_def *b0, *s0, *b1, *s1;
      emit_affine(b, nir_scalar_chase_alu_src(s, 0), loops, loop_idx, outer, remap, &b0, &s0);
      emit_affine(b, nir_scalar_chase_alu_src(s, 1), loops, loop_idx, outer, remap, &b1, &s1);
      *out_base = nir_iadd(b, b0, b1);
      *out_step = nir_iadd(b, s0, s1);
      return;
   }
   case nir_op_isub: {
      nir_def *b0, *s0, *b1, *s1;
      emit_affine(b, nir_scalar_chase_alu_src(s, 0), loops, loop_idx, outer, remap, &b0, &s0);
      emit_affine(b, nir_scalar_chase_alu_src(s, 1), loops, loop_idx, outer, remap, &b1, &s1);
      *out_base = nir_isub(b, b0, b1);
      *out_step = nir_isub(b, s0, s1);
      return;
   }
   case nir_op_ineg: {
      nir_def *b0, *s0;
      emit_affine(b, nir_scalar_chase_alu_src(s, 0), loops, loop_idx, outer, remap, &b0, &s0);
      *out_base = nir_ineg(b, b0);
      *out_step = nir_ineg(b, s0);
      return;
   }
   case nir_op_imul: {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (safe_invariant(a, outer, 10) && !a.def->divergent) {
         nir_def *k = rematerialize_scalar(b, a, outer, remap);
         nir_def *b1, *s1;
         emit_affine(b, c, loops, loop_idx, outer, remap, &b1, &s1);
         *out_base = nir_imul(b, k, b1);
         *out_step = nir_imul(b, k, s1);
      } else {
         nir_def *k = rematerialize_scalar(b, c, outer, remap);
         nir_def *b0, *s0;
         emit_affine(b, a, loops, loop_idx, outer, remap, &b0, &s0);
         *out_base = nir_imul(b, b0, k);
         *out_step = nir_imul(b, s0, k);
      }
      return;
   }
   case nir_op_ishl: {
      uint32_t sh = nir_scalar_as_uint(nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1)));
      nir_def *b0, *s0;
      emit_affine(b, nir_scalar_chase_alu_src(s, 0), loops, loop_idx, outer, remap, &b0, &s0);
      *out_base = nir_ishl_imm(b, b0, sh);
      *out_step = nir_ishl_imm(b, s0, sh);
      return;
   }
   default:
      UNREACHABLE("unexpected affine op");
   }
}

enum cmp_kind {
   CMP_NONE,
   CMP_LT0,
   CMP_GE0,
   CMP_LT_LIM,
   CMP_GE_LIM,
};

struct parsed_cmp {
   enum cmp_kind kind;
   nir_scalar val;
   nir_scalar limit;
};

static struct parsed_cmp
parse_cmp(nir_scalar d, nir_loop *outer)
{
   d = nir_scalar_chase_movs(d);
   bool neg = false;
   while (nir_scalar_is_alu(d) && nir_scalar_alu_op(d) == nir_op_inot) {
      d = nir_scalar_chase_movs(nir_scalar_chase_alu_src(d, 0));
      neg = !neg;
   }
   if (!nir_scalar_is_alu(d))
      return (struct parsed_cmp){ .kind = CMP_NONE };

   nir_op op = nir_scalar_alu_op(d);
   if (neg) {
      if (op == nir_op_ilt) op = nir_op_ige;
      else if (op == nir_op_ige) op = nir_op_ilt;
      else return (struct parsed_cmp){ .kind = CMP_NONE };
   }
   if (op != nir_op_ilt && op != nir_op_ige)
      return (struct parsed_cmp){ .kind = CMP_NONE };

   nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(d, 0));
   nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(d, 1));
   if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
      return (struct parsed_cmp){ .kind = CMP_NONE };

   if (op == nir_op_ilt && is_const_scalar(s1, 0))
      return (struct parsed_cmp){ .kind = CMP_LT0, .val = s0 };
   if (op == nir_op_ige && is_const_scalar(s1, 0))
      return (struct parsed_cmp){ .kind = CMP_GE0, .val = s0 };
   if (op == nir_op_ilt && is_const_scalar(s0, -1))
      return (struct parsed_cmp){ .kind = CMP_GE0, .val = s1 };
   if (op == nir_op_ige && is_const_scalar(s0, -1))
      return (struct parsed_cmp){ .kind = CMP_LT0, .val = s1 };

   if (op == nir_op_ilt && !s1.def->divergent && safe_invariant(s1, outer, 10))
      return (struct parsed_cmp){ .kind = CMP_LT_LIM, .val = s0, .limit = s1 };
   if (op == nir_op_ige && !s1.def->divergent && safe_invariant(s1, outer, 10))
      return (struct parsed_cmp){ .kind = CMP_GE_LIM, .val = s0, .limit = s1 };

   return (struct parsed_cmp){ .kind = CMP_NONE };
}

static bool
feeds_if_cond(nir_def *def, unsigned depth)
{
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src))
         return true;
      if (depth == 0)
         continue;
      nir_instr *parent = nir_src_use_instr(src);
      if (parent->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->def.bit_size != 1)
         continue;
      switch (alu->op) {
      case nir_op_inot:
      case nir_op_iand:
      case nir_op_ior:
      case nir_op_bcsel:
         if (feeds_if_cond(&alu->def, depth - 1))
            return true;
         break;
      default:
         break;
      }
   }
   return false;
}

static bool
match_bcsel_range(nir_scalar s0, nir_scalar s1, nir_scalar s2, nir_loop *outer,
                  nir_scalar *val_out, nir_scalar *limit_out,
                  bool *replacement_out)
{
   s0 = nir_scalar_chase_movs(s0);
   s1 = nir_scalar_chase_movs(s1);
   s2 = nir_scalar_chase_movs(s2);
   bool s1_true = nir_scalar_is_const(s1) && nir_scalar_as_bool(s1);
   bool s1_false = nir_scalar_is_const(s1) && !nir_scalar_as_bool(s1);
   bool s2_true = nir_scalar_is_const(s2) && nir_scalar_as_bool(s2);
   bool s2_false = nir_scalar_is_const(s2) && !nir_scalar_as_bool(s2);

   struct parsed_cmp c0 = parse_cmp(s0, outer);
   if (s1_true) {
      struct parsed_cmp c2 = parse_cmp(s2, outer);
      if (c0.kind == CMP_LT0 && c2.kind == CMP_GE_LIM && nir_scalar_equal(c0.val, c2.val)) {
         *val_out = c0.val; *limit_out = c2.limit; *replacement_out = false; return true;
      } else if (c2.kind == CMP_LT0 && c0.kind == CMP_GE_LIM && nir_scalar_equal(c0.val, c2.val)) {
         *val_out = c2.val; *limit_out = c0.limit; *replacement_out = false; return true;
      }
   } else if (s2_true) {
      struct parsed_cmp c1 = parse_cmp(s1, outer);
      if (c0.kind == CMP_GE0 && c1.kind == CMP_GE_LIM && nir_scalar_equal(c0.val, c1.val)) {
         *val_out = c0.val; *limit_out = c1.limit; *replacement_out = false; return true;
      } else if (c0.kind == CMP_LT_LIM && c1.kind == CMP_LT0 && nir_scalar_equal(c0.val, c1.val)) {
         *val_out = c1.val; *limit_out = c0.limit; *replacement_out = false; return true;
      }
   } else if (s2_false) {
      struct parsed_cmp c1 = parse_cmp(s1, outer);
      if (c0.kind == CMP_GE0 && c1.kind == CMP_LT_LIM && nir_scalar_equal(c0.val, c1.val)) {
         *val_out = c0.val; *limit_out = c1.limit; *replacement_out = true; return true;
      } else if (c1.kind == CMP_GE0 && c0.kind == CMP_LT_LIM && nir_scalar_equal(c0.val, c1.val)) {
         *val_out = c1.val; *limit_out = c0.limit; *replacement_out = true; return true;
      }
   } else if (s1_false) {
      struct parsed_cmp c2 = parse_cmp(s2, outer);
      if (c0.kind == CMP_LT0 && c2.kind == CMP_LT_LIM && nir_scalar_equal(c0.val, c2.val)) {
         *val_out = c0.val; *limit_out = c2.limit; *replacement_out = true; return true;
      } else if (c0.kind == CMP_GE_LIM && c2.kind == CMP_GE0 && nir_scalar_equal(c0.val, c2.val)) {
         *val_out = c2.val; *limit_out = c0.limit; *replacement_out = true; return true;
      }
   }
   return false;
}

static bool
in_top_exit_header(nir_block *block, const struct simple_loop *loops,
                   unsigned nloops)
{
   for (unsigned i = 0; i < nloops; ++i) {
      if (!loops[i].bottom_exit &&
          block == nir_loop_first_block(loops[i].loop))
         return true;
   }
   return false;
}

static bool
match_range_check_alu(nir_alu_instr *alu, const struct simple_loop *loops,
                      unsigned nloops, nir_loop *outer,
                      struct bounds_check *out)
{
   if (alu->def.num_components != 1 || alu->def.bit_size != 1 || !alu->def.divergent)
      return false;
   if (!feeds_if_cond(&alu->def, 3))
      return false;

   nir_scalar val = {0};
   nir_scalar limit = {0};
   bool found = false;
   bool replacement = false;
   nir_scalar self = nir_get_scalar(&alu->def, 0);

   if (alu->op == nir_op_ult || alu->op == nir_op_uge) {
      nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(self, 0));
      nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(self, 1));
      if (s0.def->bit_size == 32 && s1.def->bit_size == 32 &&
          !s1.def->divergent && safe_invariant(s1, outer, 10)) {
         val = s0;
         limit = s1;
         replacement = (alu->op == nir_op_ult);
         found = true;
      }
   } else if (alu->op == nir_op_ior || alu->op == nir_op_iand) {
      struct parsed_cmp c0 = parse_cmp(nir_scalar_chase_alu_src(self, 0), outer);
      struct parsed_cmp c1 = parse_cmp(nir_scalar_chase_alu_src(self, 1), outer);
      if (alu->op == nir_op_ior) {
         if (c0.kind == CMP_LT0 && c1.kind == CMP_GE_LIM && nir_scalar_equal(c0.val, c1.val)) {
            val = c0.val; limit = c1.limit; replacement = false; found = true;
         } else if (c1.kind == CMP_LT0 && c0.kind == CMP_GE_LIM && nir_scalar_equal(c0.val, c1.val)) {
            val = c1.val; limit = c0.limit; replacement = false; found = true;
         }
      } else {
         if (c0.kind == CMP_GE0 && c1.kind == CMP_LT_LIM && nir_scalar_equal(c0.val, c1.val)) {
            val = c0.val; limit = c1.limit; replacement = true; found = true;
         } else if (c1.kind == CMP_GE0 && c0.kind == CMP_LT_LIM && nir_scalar_equal(c0.val, c1.val)) {
            val = c1.val; limit = c0.limit; replacement = true; found = true;
         }
      }
   } else if (alu->op == nir_op_bcsel) {
      found = match_bcsel_range(nir_scalar_chase_alu_src(self, 0),
                                nir_scalar_chase_alu_src(self, 1),
                                nir_scalar_chase_alu_src(self, 2),
                                outer, &val, &limit, &replacement);
   }

   if (!found)
      return false;

   int lidx = analyze_affine(val, loops, nloops, outer, 12);
   if (lidx < 0)
      return false;

   if (!node_inside_cf(&alu->instr.block->cf_node, &loops[lidx].loop->cf_node) ||
       in_top_exit_header(alu->instr.block, loops, nloops))
      return false;

   *out = (struct bounds_check){
      .def = &alu->def,
      .val = val,
      .limit = limit,
      .loop_idx = (unsigned)lidx,
      .replacement = replacement,
   };
   return true;
}

static bool
match_range_check_phi(nir_phi_instr *phi, const struct simple_loop *loops,
                      unsigned nloops, nir_loop *outer,
                      struct bounds_check *out)
{
   if (phi->def.num_components != 1 || phi->def.bit_size != 1 || !phi->def.divergent)
      return false;
   if (exec_list_length(&phi->srcs) != 2)
      return false;
   if (!feeds_if_cond(&phi->def, 3))
      return false;

   nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
   if (!prev || prev->type != nir_cf_node_if)
      return false;
   nir_if *nif = nir_cf_node_as_if(prev);
   if (!exec_list_is_singular(&nif->then_list) ||
       !exec_list_is_singular(&nif->else_list))
      return false;

   nir_block *then_blk = nir_if_first_then_block(nif);
   nir_block *else_blk = nir_if_first_else_block(nif);
   nir_def *then_val = NULL, *else_val = NULL;
   nir_foreach_phi_src(psrc, phi) {
      if (psrc->pred == then_blk)
         then_val = psrc->src.ssa;
      else if (psrc->pred == else_blk)
         else_val = psrc->src.ssa;
   }
   if (!then_val || !else_val)
      return false;

   nir_scalar val = {0};
   nir_scalar limit = {0};
   bool replacement = false;
   if (!match_bcsel_range(nir_scalar_resolved(nif->condition.ssa, 0),
                          nir_scalar_resolved(then_val, 0),
                          nir_scalar_resolved(else_val, 0),
                          outer, &val, &limit, &replacement))
      return false;

   int lidx = analyze_affine(val, loops, nloops, outer, 12);
   if (lidx < 0)
      return false;

   if (!node_inside_cf(&phi->instr.block->cf_node, &loops[lidx].loop->cf_node) ||
       in_top_exit_header(phi->instr.block, loops, nloops))
      return false;

   *out = (struct bounds_check){
      .def = &phi->def,
      .val = val,
      .limit = limit,
      .loop_idx = (unsigned)lidx,
      .replacement = replacement,
   };
   return true;
}

static void
mark_loops_dont_unroll(nir_cf_node *node)
{
   switch (node->type) {
   case nir_cf_node_loop: {
      nir_loop *l = nir_cf_node_as_loop(node);
      l->control = nir_loop_control_dont_unroll;
      foreach_list_typed(nir_cf_node, child, node, &l->body)
         mark_loops_dont_unroll(child);
      break;
   }
   case nir_cf_node_if: {
      nir_if *nif = nir_cf_node_as_if(node);
      foreach_list_typed(nir_cf_node, child, node, &nif->then_list)
         mark_loops_dont_unroll(child);
      foreach_list_typed(nir_cf_node, child, node, &nif->else_list)
         mark_loops_dont_unroll(child);
      break;
   }
   default:
      break;
   }
}

static nir_if *
try_unswitch_loop(nir_function_impl *impl, nir_loop *outer_loop)
{
   struct simple_loop loops[MAX_NEST];
   unsigned nloops = 0;

   if (!match_simple_loop(outer_loop, outer_loop, &loops[0]))
      return NULL;
   nloops = 1;

   unsigned inner_count = 0;
   nir_loop *inner_loop = NULL;
   nir_foreach_block_in_cf_node(block, &outer_loop->cf_node) {
      nir_cf_node *next = nir_cf_node_next(&block->cf_node);
      if (next && next->type == nir_cf_node_loop) {
         inner_count++;
         inner_loop = nir_cf_node_as_loop(next);
      }
   }
   if (inner_count == 1 && inner_loop &&
       match_simple_loop(inner_loop, outer_loop, &loops[1])) {
      nloops = 2;
   }

   if (!loop_tree_safe(outer_loop))
      return NULL;

   nir_block *orig_after =
      nir_cf_node_as_block(nir_cf_node_next(&outer_loop->cf_node));
   if (!exec_list_is_empty(&orig_after->instr_list) &&
       nir_block_first_instr(orig_after)->type == nir_instr_type_phi)
      return NULL;

   struct bounds_check checks[MAX_CHECKS];
   unsigned nchecks = 0;
   nir_foreach_block_in_cf_node(block, &outer_loop->cf_node) {
      nir_foreach_instr(instr, block) {
         struct bounds_check bc;
         bool matched = false;
         if (instr->type == nir_instr_type_alu) {
            matched = match_range_check_alu(nir_instr_as_alu(instr), loops,
                                            nloops, outer_loop, &bc);
         } else if (instr->type == nir_instr_type_phi) {
            matched = match_range_check_phi(nir_instr_as_phi(instr), loops,
                                            nloops, outer_loop, &bc);
         }
         if (matched && nchecks < MAX_CHECKS)
            checks[nchecks++] = bc;
      }
   }

   if (nchecks == 0)
      return NULL;

   struct util_dynarray live_outs = UTIL_DYNARRAY_INIT;
   nir_foreach_block_in_cf_node(block, &outer_loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (def && has_outside_use(def, outer_loop))
            util_dynarray_append(&live_outs, def);
      }
   }

   nir_builder b = nir_builder_at(nir_before_cf_node(&outer_loop->cf_node));
   struct hash_table *inv_remap = _mesa_pointer_hash_table_create(NULL);
   nir_def *all_in_bounds = nir_imm_true(&b);

   for (unsigned i = 0; i < nchecks; ++i) {
      bool dup = false;
      for (unsigned j = 0; j < i; ++j) {
         if (checks[j].loop_idx == checks[i].loop_idx &&
             nir_scalar_equal(checks[j].val, checks[i].val) &&
             nir_scalar_equal(checks[j].limit, checks[i].limit)) {
            dup = true;
            break;
         }
      }
      if (dup)
         continue;

      nir_def *bound =
         rematerialize_scalar(&b, loops[checks[i].loop_idx].bound, outer_loop, inv_remap);
      nir_def *limit =
         rematerialize_scalar(&b, checks[i].limit, outer_loop, inv_remap);
      nir_def *base = NULL, *step = NULL;
      emit_affine(&b, checks[i].val, loops, checks[i].loop_idx, outer_loop,
                  inv_remap, &base, &step);

      nir_def *last = nir_iadd_imm(&b, bound, -1);
      nir_def *span = nir_imul(&b, last, step);
      nir_def *span_hi = nir_imul_high(&b, last, step);
      nir_def *span_sign = nir_ishr_imm(&b, span, 31);
      nir_def *no_mul_ov = nir_ieq(&b, span_hi, span_sign);
      nir_def *valid = nir_iand(
         &b,
         nir_iand(&b, nir_igt_imm(&b, bound, 0), nir_igt_imm(&b, limit, 0)),
         nir_iand(&b, no_mul_ov, nir_ine_imm(&b, span, (int32_t)0x80000000)));
      nir_def *step_ge0 = nir_ige_imm(&b, step, 0);
      nir_def *lo_req = nir_bcsel(&b, step_ge0, nir_imm_int(&b, 0), nir_ineg(&b, span));
      nir_def *hi_exc = nir_bcsel(&b, step_ge0, nir_isub(&b, limit, span), limit);
      nir_def *has_range = nir_iand(&b, valid, nir_ilt(&b, lo_req, hi_exc));
      nir_def *width = nir_bcsel(&b, has_range, nir_isub(&b, hi_exc, lo_req),
                                 nir_imm_int(&b, 0));
      nir_def *ok = nir_ult(&b, nir_isub(&b, base, lo_req), width);
      all_in_bounds = nir_iand(&b, all_in_bounds, ok);
   }
   _mesa_hash_table_destroy(inv_remap, NULL);

   nir_if *unswitch = nir_push_if(&b, all_in_bounds);
   nir_pop_if(&b, unswitch);

   nir_cf_list extracted;
   nir_cf_extract(&extracted, nir_before_cf_node(&outer_loop->cf_node),
                  nir_after_cf_node(&outer_loop->cf_node));

   struct hash_table *clone_remap = _mesa_pointer_hash_table_create(NULL);
   nir_cf_list_clone_and_reinsert(&extracted, &unswitch->cf_node,
                                  nir_before_block(nir_if_first_then_block(unswitch)),
                                  clone_remap);
   nir_cf_reinsert(&extracted, nir_before_block(nir_if_first_else_block(unswitch)));

   mark_loops_dont_unroll(&outer_loop->cf_node);

   nir_block *after_if = nir_cf_node_as_block(nir_cf_node_next(&unswitch->cf_node));
   nir_block *then_end = nir_if_last_then_block(unswitch);
   nir_block *else_end = nir_if_last_else_block(unswitch);

   /* Build live-out phis BEFORE folding checks so that if a folded check is
    * itself live-out, nir_def_replace() below cleanly rewrites the phi's
    * then_end source to the replacement constant instead of a deleted def. */
   util_dynarray_foreach(&live_outs, nir_def *, orig_ptr) {
      nir_def *orig_def = *orig_ptr;
      struct hash_entry *he = _mesa_hash_table_search(clone_remap, orig_def);
      assert(he);
      nir_def *fast_def = he->data;

      nir_phi_instr *phi = nir_phi_instr_create(impl->function->shader);
      nir_def_init(&phi->instr, &phi->def, orig_def->num_components, orig_def->bit_size);
      nir_phi_instr_add_src(phi, then_end, fast_def);
      nir_phi_instr_add_src(phi, else_end, orig_def);
      nir_instr_insert(nir_before_block(after_if), &phi->instr);

      nir_foreach_use_including_if_safe(src, orig_def) {
         if (nir_src_is_if(src)) {
            if (node_inside_cf(&nir_src_use_if(src)->cf_node, &unswitch->cf_node))
               continue;
         } else {
            nir_instr *parent = nir_src_use_instr(src);
            if (parent == &phi->instr ||
                node_inside_cf(&parent->block->cf_node, &unswitch->cf_node))
               continue;
         }
         nir_src_rewrite(src, &phi->def);
      }
   }

   /* Place replacement constants at the entry of the then branch so they
    * dominate both in-loop uses and any live-out merge phis. */
   nir_builder fb = nir_builder_at(nir_before_block(nir_if_first_then_block(unswitch)));
   for (unsigned i = 0; i < nchecks; ++i) {
      struct hash_entry *he = _mesa_hash_table_search(clone_remap, checks[i].def);
      assert(he);
      nir_def *fast_def = he->data;
      nir_def_replace(fast_def, nir_imm_bool(&fb, checks[i].replacement));
   }

   util_dynarray_fini(&live_outs);
   _mesa_hash_table_destroy(clone_remap, NULL);
   return unswitch;
}

static bool
process_cf_list(nir_function_impl *impl, struct exec_list *cf_list)
{
   bool progress = false;
   struct exec_node *cur = exec_list_get_head(cf_list);
   while (!exec_node_is_tail_sentinel(cur)) {
      nir_cf_node *node = exec_node_data(nir_cf_node, cur, node);
      struct exec_node *next = cur->next;
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= process_cf_list(impl, &nif->then_list);
         progress |= process_cf_list(impl, &nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         nir_if *unswitch = try_unswitch_loop(impl, loop);
         if (unswitch) {
            progress = true;
            /* nir_cf_extract stitches the block after loop into the block
             * after unswitch; continue from after unswitch so later sibling
             * loops in cf_list are still visited. */
            next = unswitch->cf_node.node.next;
         } else {
            progress |= process_cf_list(impl, &loop->body);
         }
         break;
      }
      default:
         break;
      }
      cur = next;
   }
   return progress;
}

bool
agx_nir_opt_unswitch_affine_bounds(nir_shader *shader)
{
   nir_divergence_analysis(shader);

   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      if (process_cf_list(impl, &impl->body)) {
         nir_progress(true, impl, nir_metadata_none);
         progress = true;
      } else {
         nir_progress(false, impl, nir_metadata_all);
      }
   }
   return progress;
}
