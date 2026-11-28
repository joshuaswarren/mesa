/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Guarded unrolling of counted loops whose trip count is only known at run
 * time, such as a filter window clipped at the image border:
 *
 *    for (i = lo; i <= hi; i++)
 *       body(i);
 *
 * becomes
 *
 *    for (; i <= hi - (F - 1); i += F)    // "main" loop, no exits inside
 *       body(i); body(i + 1); ... body(i + F - 1);
 *    for (; i <= hi; i++)                 // the original loop
 *       body(i);
 *
 * The main loop runs an iteration only when the original loop would run the
 * next F iterations, so every instruction it executes is one the original
 * executes, in the same order. Its guard is computed so that the iteration
 * variable cannot wrap inside those F iterations; when hi is so close to the
 * bottom of the range that hi - (F - 1) would wrap, the guard is never true
 * and the original loop does all the work.
 *
 * On a GPU that issues one instruction per lane per clock, the loop control
 * (compare, mask update, branch and induction increment) is a large share of
 * a short loop body, and the body's loads only overlap within one iteration.
 * Unrolling removes F - 1 of every F loop-control sequences, lets the
 * scheduler issue F iterations' loads together, and lets the memory
 * vectorizer merge their contiguous loads.
 *
 * Only innermost loops whose single exit is a test of a basic induction
 * variable against a loop-invariant bound at the top of the body qualify.
 * Loops with subgroup, quad or barrier operations are left alone: unrolling
 * changes which iterations of different invocations execute together.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"
#include "util/set.h"

enum cont_rel {
   REL_LT,
   REL_LE,
   REL_GT,
   REL_GE,
};

struct counted_loop {
   nir_loop *loop;
   nir_if *nif;
   bool break_in_then;
   /* The exit test ends the body (a rotated loop) instead of starting it. */
   bool bottom;
   nir_phi_instr *iv;
   /* The test reads iv + offset (or iv + step_def when offset_is_step). */
   int64_t offset;
   bool offset_is_step;
   nir_scalar bound;
   int64_t step;
   /* When step == 0, the loop steps by a loop-invariant 32-bit runtime scalar. */
   nir_scalar step_def;
   enum cont_rel rel;
   bool is_unsigned;
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

/* By the control-flow tree, not block indices: those go stale once a loop
 * of the shader has been unrolled. */
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

/* A value every iteration computes alike from values defined before the
 * loop, without side effects, in the first block: such as a bound read from a
 * uniform block. The first block runs before the loop can exit, so computing
 * the value once in the preheader reads nothing the loop would not read. */
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

/* Recompute a header_invariant() value at the builder's cursor. */
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

/* Operations whose result depends on which invocations run them together:
 * subgroup and quad operations, and the AGX forms the subgroup lowering has
 * already produced before this pass (elect, the active-lane count, quad
 * ballots), plus the AGX operations that act for the whole SIMD-group or
 * workgroup (ray traversal and its shared state, mesh payload barriers,
 * sample-mask demotes). A main loop and a remainder, or two copies of an
 * outer loop, run by different invocations at once would change them. */
static bool
lane_dependent(const nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_infos[intr->intrinsic].flags &
       (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP))
      return true;
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
      return true;
   default:
      return false;
   }
}

/* Everything in the loop that is safe to execute F times in a row per
 * main-loop iteration. Returns the number of instructions, or -1. When
 * `top_level_out` is non-NULL, stores the number of instructions directly in
 * the loop's unconditional top-level blocks (excluding nested `if` bodies). */
static int
count_unrollable(nir_loop *loop, nir_if *terminator, int *top_level_out)
{
   int count = 0;
   int top_level = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      const bool is_top = block->cf_node.parent == &loop->cf_node;
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            /* Only the terminator's break. */
            if (block->cf_node.parent != &terminator->cf_node)
               return -1;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent(intr))
               return -1;
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
               return -1;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return -1;
            break;
         case nir_instr_type_call:
            return -1;
         default:
            break;
         }
         if (instr->type != nir_instr_type_phi &&
             instr->type != nir_instr_type_load_const &&
             instr->type != nir_instr_type_undef) {
            count++;
            if (is_top)
               top_level++;
         }
      }
   }
   if (top_level_out)
      *top_level_out = top_level;
   return count;
}

struct varying_state {
   BITSET_WORD *varying;
   bool any;
};

static bool
src_varies(nir_src *src, void *data)
{
   struct varying_state *st = data;
   st->any |= BITSET_TEST(st->varying, src->ssa->index);
   return true;
}

static unsigned
loaded_dwords_in_loop(nir_function_impl *impl, nir_loop *loop,
                      bool top_level_only)
{
   BITSET_WORD *varying =
      calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
   nir_block *header = nir_loop_first_block(loop);
   unsigned dwords = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      const bool count_block =
         !top_level_only || block->cf_node.parent == &loop->cf_node;
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         struct varying_state st = {varying, false};
         if (instr->type == nir_instr_type_phi && block == header)
            st.any = true;
         else
            nir_foreach_src(instr, src_varies, &st);
         if (!st.any)
            continue;
         BITSET_SET(varying, def->index);

         if (!count_block)
            continue;
         bool load = instr->type == nir_instr_type_tex;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
            load = info->has_dest && nir_intrinsic_has_align_mul(intr) &&
                   !strncmp(info->name, "load_", 5);
         }
         if (load)
            dwords += def->num_components * DIV_ROUND_UP(def->bit_size, 32);
      }
   }
   free(varying);
   return dwords;
}

/* 32-bit registers one iteration's memory and texture reads fill, counting
 * the reads whose address depends on a value the loop carries (reads of
 * addresses that are the same every iteration, such as descriptors and the
 * bound's parameters, are shared by the unrolled copies). */
static unsigned
loaded_dwords(nir_function_impl *impl, nir_loop *loop)
{
   return loaded_dwords_in_loop(impl, loop, false);
}

static bool loop_is_readonly(nir_loop *loop);
static bool
loop_is_readonly_with_loads(nir_function_impl *impl, nir_loop *loop)
{
   if (loaded_dwords(impl, loop) == 0 || !loop_is_readonly(loop))
      return false;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (nir_intrinsic_has_access(intr) &&
             (nir_intrinsic_access(intr) & ACCESS_VOLATILE))
            return false;
      }
   }
   return true;
}

static bool
analyze_loop_opts(nir_loop *loop, struct counted_loop *out,
                  bool allow_runtime_step, bool allow_nested)
{
   if (nir_loop_has_continue_construct(loop) || loop->partially_unrolled ||
       loop->control == nir_loop_control_dont_unroll)
      return false;

   /* The single exit is a lone break in one branch of an if, either right
    * after the header block or as the last node before the final block. */
   bool bottom = false;
   nir_cf_node *first = nir_cf_node_next(&nir_loop_first_block(loop)->cf_node);
   nir_cf_node *last_if = nir_cf_node_prev(&nir_loop_last_block(loop)->cf_node);
   nir_if *nif = NULL;
   bool break_in_then = false;
   for (unsigned pass = 0; pass < 2 && !nif; ++pass) {
      /* The bottom form first: an exit after the whole body needs no
       * side-effect-free header. */
      nir_cf_node *node = pass == 0 ? last_if : first;
      if (!node || node->type != nir_cf_node_if)
         continue;
      if (pass == 0 &&
          !exec_list_is_empty(&nir_loop_last_block(loop)->instr_list))
         continue;
      nir_if *candidate = nir_cf_node_as_if(node);
      if (branch_is_lone_break(&candidate->then_list) &&
          branch_is_empty(&candidate->else_list))
         break_in_then = true;
      else if (branch_is_lone_break(&candidate->else_list) &&
               branch_is_empty(&candidate->then_list))
         break_in_then = false;
      else
         continue;
      nif = candidate;
      bottom = pass == 0;
   }
   if (!nif)
      return false;

   /* Innermost: nothing but blocks and ifs after the terminator. */
   if (!allow_nested) {
      nir_foreach_block_in_cf_node(block, &loop->cf_node) {
         nir_cf_node *parent = block->cf_node.parent;
         while (parent != &loop->cf_node) {
            if (parent->type == nir_cf_node_loop)
               return false;
            parent = parent->parent;
         }
      }
   }

   /* A top-tested main loop runs its header once more before it exits into
    * the original loop, which runs it again for the same iteration: the
    * header must have no side effects. A rotated loop's main loop exits
    * before any of its body. */
   nir_block *header = nir_loop_first_block(loop);
   nir_foreach_instr(instr, header) {
      if (bottom)
         break;
      if (instr->type == nir_instr_type_intrinsic &&
          (!(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].flags &
             NIR_INTRINSIC_CAN_ELIMINATE) ||
           (nir_intrinsic_has_access(nir_instr_as_intrinsic(instr)) &&
            (nir_intrinsic_access(nir_instr_as_intrinsic(instr)) & ACCESS_VOLATILE))))
         return false;
      if (instr->type == nir_instr_type_call)
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

   /* One side is a phi of the header, or that phi plus a constant (or plus
    * a header-invariant runtime step in a bottom-tested loop), the other is
    * invariant. */
   bool iv_left = false;
   bool offset_is_step = false;
   nir_scalar ivs = {0}, bound = {0}, cond_step = {0};
   int64_t offset = 0;
   for (unsigned side = 0; side < 2 && !ivs.def; ++side) {
      nir_scalar x = side == 0 ? a : b, y = side == 0 ? b : a;
      if (!nir_scalar_is_const(y) && !header_invariant(y.def, loop, 8))
         continue;
      int64_t c = 0;
      bool dyn_off = false;
      nir_scalar dyn_step = {0};
      if (nir_scalar_is_alu(x) && nir_scalar_alu_op(x) == nir_op_iadd) {
         for (unsigned k = 0; k < 2; ++k) {
            nir_scalar kk = nir_scalar_chase_alu_src(x, k);
            nir_scalar other = nir_scalar_chase_alu_src(x, 1 - k);
            if (nir_def_instr(other.def)->type != nir_instr_type_phi)
               continue;
            if (nir_scalar_is_const(kk)) {
               c = nir_scalar_as_int(kk);
               x = other;
               break;
            }
            if (allow_runtime_step && bottom && kk.def->bit_size == 32 &&
                header_invariant(kk.def, loop, 8)) {
               dyn_off = true;
               dyn_step = kk;
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
         offset_is_step = dyn_off;
         cond_step = dyn_step;
         iv_left = side == 0;
      }
   }
   if (!ivs.def)
      return false;

   nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(ivs.def));
   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
   if (!back)
      return false;

   nir_scalar update = nir_scalar_resolved(back->src.ssa, 0);
   if (!nir_scalar_is_alu(update) || nir_scalar_alu_op(update) != nir_op_iadd)
      return false;

   int64_t step = 0;
   nir_scalar step_def = {0};
   bool found = false;
   for (unsigned s = 0; s < 2; ++s) {
      nir_scalar src = nir_scalar_chase_alu_src(update, s);
      nir_scalar other = nir_scalar_chase_alu_src(update, 1 - s);
      if (other.def != &phi->def)
         continue;
      if (!offset_is_step && nir_scalar_is_const(src)) {
         step = nir_scalar_as_int(src);
         found = true;
         break;
      }
      if (allow_runtime_step && !nir_scalar_is_const(src) &&
          src.def->bit_size == 32 && header_invariant(src.def, loop, 8)) {
         if ((!offset_is_step && offset == 0) ||
             (offset_is_step && nir_scalar_equal(src, cond_step))) {
            step = 0;
            step_def = src;
            found = true;
            break;
         }
      }
   }
   if (!found)
      return false;
   if (!step_def.def && (step == 0 || step > 0x10000 || step < -0x10000))
      return false;

   /* Normalize to the condition that keeps the loop running, written as
    * "iv REL bound". The break condition is cond (or !cond when negated)
    * when the break is in the then-branch, the opposite otherwise. */
   bool break_on_true = break_in_then != negate;
   /* cond is (a < b) for ilt/ult and (a >= b) for ige/uge. */
   bool less = op == nir_op_ilt || op == nir_op_ult;
   /* Continue condition on (a, b): */
   bool cont_less = break_on_true ? !less : less;
   /* cont_less: a < b; otherwise a >= b. */
   enum cont_rel rel;
   if (iv_left)
      rel = cont_less ? REL_LT : REL_GE;
   else
      rel = cont_less ? REL_GT : REL_LE; /* bound < iv / bound >= iv */

   if (step_def.def) {
      if (rel != REL_LT && rel != REL_LE)
         return false;
   } else if ((rel == REL_LT || rel == REL_LE) != (step > 0)) {
      return false;
   }

   *out = (struct counted_loop){
      .loop = loop,
      .nif = nif,
      .break_in_then = break_in_then,
      .bottom = bottom,
      .iv = phi,
      .offset = offset,
      .offset_is_step = offset_is_step,
      .bound = bound,
      .step = step,
      .step_def = step_def,
      .rel = rel,
      .is_unsigned = op == nir_op_ult || op == nir_op_uge,
   };
   return true;
}

static bool
analyze_loop(nir_loop *loop, struct counted_loop *out)
{
   return analyze_loop_opts(loop, out, false, false);
}

/* The loop-invariant value G such that "iv < G" (REL_LT, REL_LE) or
 * "iv > G" (REL_GT, REL_GE) proves the next `factor` iterations run without
 * the induction variable wrapping. */
static nir_def *
guard_bound(nir_builder *b, const struct counted_loop *cl, nir_def *bound,
            nir_def *step_val, unsigned factor)
{
   const bool u = cl->is_unsigned;
   const int64_t min = u ? 0 : INT32_MIN;
   const int64_t max = u ? UINT32_MAX : INT32_MAX;

   if (cl->step == 0) {
      assert(step_val && (cl->rel == REL_LT || cl->rel == REL_LE));
      /* Require step >= 1, factor * step <= max (no mul overflow), and
       * bound - factor * step >= min (no sub underflow). Using span = factor *
       * step guarantees that for any iv < bound - span, all `factor`
       * iterations and iv + factor * step stay strictly below `bound <= max`
       * without wrapping. */
      nir_def *span = nir_imul_imm(b, step_val, factor);
      nir_def *valid;
      if (u) {
         nir_def *pos = nir_ine_imm(b, step_val, 0);
         nir_def *no_mul_ovf =
            nir_uge(b, nir_imm_int(b, UINT32_MAX / factor), step_val);
         nir_def *no_sub_udf = nir_uge(b, bound, span);
         valid = nir_iand(b, nir_iand(b, pos, no_mul_ovf), no_sub_udf);
      } else {
         nir_def *pos = nir_ilt(b, nir_imm_int(b, 0), step_val);
         nir_def *no_mul_ovf =
            nir_ige(b, nir_imm_int(b, INT32_MAX / factor), step_val);
         nir_def *min_plus_span = nir_iadd_imm(b, span, (uint32_t)INT32_MIN);
         nir_def *no_sub_udf = nir_ige(b, bound, min_plus_span);
         valid = nir_iand(b, nir_iand(b, pos, no_mul_ovf), no_sub_udf);
      }
      nir_def *adj = nir_isub(b, bound, span);
      nir_def *never = nir_imm_int(b, (uint32_t)min);
      return nir_bcsel(b, valid, adj, never);
   }

   const int64_t span = (int64_t)(factor - 1) * (cl->step > 0 ? cl->step : -cl->step);

   nir_def *valid, *adj, *never;
   switch (cl->rel) {
   case REL_LT:
   case REL_LE: {
      /* bound - span must not wrap below the minimum. */
      nir_def *lowest = nir_imm_int(b, (uint32_t)(min + span));
      valid = u ? nir_uge(b, bound, lowest) : nir_ige(b, bound, lowest);
      int64_t sub = cl->rel == REL_LT ? span : span - 1;
      adj = nir_iadd_imm(b, bound, -sub);
      never = nir_imm_int(b, (uint32_t)min);
      break;
   }
   default: {
      nir_def *highest = nir_imm_int(b, (uint32_t)(max - span));
      valid = u ? nir_uge(b, highest, bound) : nir_ige(b, highest, bound);
      int64_t add = cl->rel == REL_GT ? span : span - 1;
      adj = nir_iadd_imm(b, bound, add);
      never = nir_imm_int(b, (uint32_t)max);
      break;
   }
   }
   return nir_bcsel(b, valid, adj, never);
}

static bool
block_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node == &loop->cf_node)
         return true;
   }
   return false;
}

static nir_block *
use_block(nir_src *src)
{
   if (nir_src_is_if(src))
      return nir_cf_node_as_block(nir_cf_node_prev(&nir_src_use_if(src)->cf_node));
   return nir_src_use_instr(src)->block;
}

struct live_out {
   nir_def *def;
   /* The register of the header phi it feeds, or NULL for a loop-invariant
    * value. */
   nir_def *reg;
   bool invariant;
};

/* Computed from values defined before the loop only, by instructions that
 * may be repeated (as header_invariant(), in any block of the loop). */
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

/* The loop's values used after it, when each is the back-edge value of a
 * header phi (a rotated loop's accumulators and induction variables) or
 * loop-invariant (recomputed after the loop). */
static bool
back_edge_live_outs(nir_loop *loop, struct live_out *outs, unsigned *count)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   /* Phis after the loop would need sources from the new if: keep the
    * plain form. */
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   if (nir_block_first_instr(after) &&
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
         /* A header phi used after the loop (its value at the start of the
          * last iteration) has no register holding it after the main loop;
          * nor does a header phi that is another phi's back-edge value. */
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

/* The register a header phi became, from its write at the end of the
 * latch. */
static nir_def *
latch_register(nir_loop *loop, nir_def *value)
{
   nir_foreach_instr_reverse(instr, nir_loop_last_block(loop)) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
      if (store->intrinsic == nir_intrinsic_store_reg && store->src[0].ssa == value)
         return store->src[1].ssa;
   }
   return NULL;
}

static void
unroll_loop(nir_function_impl *impl, struct counted_loop *cl, nir_def *bound,
            nir_def *step_val, unsigned factor)
{
   nir_loop *loop = cl->loop;
   nir_builder b = nir_builder_create(impl);

   /* The guard bound, in the preheader. */
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   b.cursor = nir_after_block_before_jump(pre);
   nir_def *g = guard_bound(&b, cl, bound, step_val, factor);

   /* A rotated loop runs its body before its test, so the original loop,
    * reached after the main loop, runs at least one more iteration. When
    * every value the loop leaves is a header phi's back-edge value and the
    * test reads the stepped induction variable, skip the original loop
    * instead once the test fails: the main loop then needs only `factor`
    * iterations left, not factor + 1 (a trip count of exactly `factor`,
    * such as 8 descriptor words, runs entirely in the main loop). */
   struct live_out outs[16];
   unsigned nouts = 0;
   nir_phi_src *init = nir_phi_get_src_from_block(cl->iv, pre);
   nir_phi_src *iv_back = nir_phi_get_src_from_block(cl->iv, nir_loop_last_block(loop));
   const bool exact_tail = cl->step != 0 && cl->bottom && cl->offset == cl->step && init &&
                           iv_back && back_edge_live_outs(loop, outs, &nouts);
   nir_def *iv0 = exact_tail ? init->src.ssa : NULL;
   nir_def *iv_next = exact_tail ? iv_back->src.ssa : NULL;

   /* The guard test, in the header after its phis. A top-tested loop runs
    * iteration k when its test passes on v + k * step, k = 0 .. factor - 1,
    * with v the tested value at the top. A rotated loop runs the iteration
    * after k when its test passes on v + k * step, k = 0 .. factor - 1: the
    * main loop's factor iterations, plus the first one of the original loop,
    * which runs its body before its test. The same guard covers both. */
   b.cursor = nir_after_phis(nir_loop_first_block(loop));
   nir_def *iv = &cl->iv->def;
   if (cl->offset && !exact_tail)
      iv = nir_iadd_imm(&b, iv, cl->offset);
   nir_def *cont;
   if (cl->rel == REL_LT || cl->rel == REL_LE)
      cont = cl->is_unsigned ? nir_ult(&b, iv, g) : nir_ilt(&b, iv, g);
   else
      cont = cl->is_unsigned ? nir_ult(&b, g, iv) : nir_ilt(&b, g, iv);
   nir_def *guard = cl->break_in_then ? nir_inot(&b, cont) : cont;
   /* Which source of the compare reads the induction variable: after the
    * header phis become registers, it is the load of the iv's register. */
   nir_alu_instr *cont_alu = nir_def_as_alu(cont);
   const unsigned iv_src = (cl->rel == REL_LT || cl->rel == REL_LE) ? 0 : 1;

   /* Leave SSA for the header's phis: both loops then read and write the
    * same registers and need no phis of their own. Write the registers at
    * the end of the latch, which every copy of the body includes: placed
    * higher, a value defined in the header would be written in the exit
    * test's continue branch, which the extra copies do not repeat. */
   nir_lower_phis_to_regs_block(nir_loop_first_block(loop), true);

   /* Build the main loop in front of the original. */
   nir_loop *main_loop = nir_loop_create(impl);
   nir_cf_node_insert(nir_before_cf_node(&loop->cf_node), &main_loop->cf_node);
   main_loop->partially_unrolled = true;
   loop->partially_unrolled = true;

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_cf_list(&loop->body),
                  nir_after_cf_list(&loop->body));
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_cf_list_clone_and_reinsert(&body, &main_loop->cf_node,
                                  nir_after_cf_list(&main_loop->body), remap);
   nir_cf_reinsert(&body, nir_after_cf_list(&loop->body));

   struct hash_entry *he = _mesa_hash_table_search(remap, guard);
   assert(he);
   nir_def *main_guard = he->data;
   _mesa_hash_table_destroy(remap, NULL);

   nir_if *main_if;
   if (!cl->bottom) {
      main_if = nir_cf_node_as_if(
         nir_cf_node_next(&nir_loop_first_block(main_loop)->cf_node));
      nir_src_rewrite(&main_if->condition, main_guard);
   } else {
      /* Test at the top instead, and drop the bottom test. Its continue
       * branch holds the register writes that replaced the header phis'
       * back-edge sources: keep them, after the dropped test. */
      nir_if *bottom_if = nir_cf_node_as_if(
         nir_cf_node_prev(&nir_loop_last_block(main_loop)->cf_node));
      struct exec_list *cont_list = cl->break_in_then ? &bottom_if->else_list
                                                      : &bottom_if->then_list;
      nir_cf_list cont;
      nir_cf_extract(&cont, nir_before_cf_list(cont_list),
                     nir_after_cf_list(cont_list));
      nir_cf_reinsert(&cont, nir_after_cf_node(&bottom_if->cf_node));
      nir_cf_node_remove(&bottom_if->cf_node);
      b.cursor = nir_after_instr(nir_def_instr(main_guard));
      main_if = nir_push_if(&b, main_guard);
      if (cl->break_in_then)
         nir_jump(&b, nir_jump_break);
      nir_push_else(&b, main_if);
      if (!cl->break_in_then)
         nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, main_if);
   }
   /* Repeat the iteration factor - 1 more times inside the main loop. */
   nir_cf_list head, rest;
   nir_cf_extract(&head, nir_before_cf_list(&main_loop->body),
                  nir_before_cf_node(&main_if->cf_node));
   nir_cf_extract(&rest, nir_after_cf_node(&main_if->cf_node),
                  nir_after_cf_list(&main_loop->body));

   nir_cf_list copies[2 * 16];
   assert(factor <= 16);
   for (unsigned i = 1; i < factor; ++i) {
      remap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list_clone(&copies[2 * i], &head, &main_loop->cf_node, remap);
      nir_cf_list_clone(&copies[2 * i + 1], &rest, &main_loop->cf_node, remap);
      _mesa_hash_table_destroy(remap, NULL);
   }

   nir_cf_reinsert(&head, nir_before_cf_node(&main_if->cf_node));
   nir_cf_reinsert(&rest, nir_after_cf_list(&main_loop->body));
   for (unsigned i = 1; i < factor; ++i) {
      nir_cf_reinsert(&copies[2 * i], nir_after_cf_list(&main_loop->body));
      nir_cf_reinsert(&copies[2 * i + 1], nir_after_cf_list(&main_loop->body));
   }

   if (!exact_tail)
      return;

   /* Run the original loop only for its first iteration (no main
    * iteration ran: the induction variable is still its initial value) or
    * when the test the dropped iteration would have made passes. */
   nir_intrinsic_instr *iv_load =
      nir_def_as_intrinsic(cont_alu->src[iv_src].src.ssa);
   assert(iv_load->intrinsic == nir_intrinsic_load_reg);
   nir_def *iv_reg = iv_load->src[0].ssa;
   (void)iv_next;
   for (unsigned i = 0; i < nouts; ++i) {
      if (outs[i].invariant)
         continue;
      outs[i].reg = latch_register(loop, outs[i].def);
      assert(outs[i].reg);
   }
   b.cursor = nir_before_cf_node(&loop->cf_node);
   nir_def *v = nir_load_reg(&b, iv_reg);
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
   nir_cf_extract(&moved, nir_before_cf_node(&loop->cf_node),
                  nir_after_cf_node(&loop->cf_node));
   nir_cf_reinsert(&moved, nir_before_cf_list(&tail->then_list));

   /* The values the loop leaves: after the if, from their registers. The
    * original loop exits before its latch writes them, so write them once
    * more after it. */
   b.cursor = nir_after_cf_node(&tail->cf_node);
   struct hash_table *again = _mesa_pointer_hash_table_create(NULL);
   for (unsigned i = 0; i < nouts; ++i) {
      nir_def *value = outs[i].invariant
                          ? rematerialize(&b, outs[i].def, loop, again)
                          : nir_load_reg(&b, outs[i].reg);
      nir_foreach_use_including_if_safe(src, outs[i].def) {
         if (!block_in_loop(use_block(src), loop))
            nir_src_rewrite(src, value);
      }
   }
   _mesa_hash_table_destroy(again, NULL);
   b.cursor = nir_after_cf_node(&loop->cf_node);
   for (unsigned i = 0; i < nouts; ++i) {
      if (!outs[i].invariant)
         nir_store_reg(&b, outs[i].def, outs[i].reg);
   }
}

static void collect_loops(struct exec_list *list, struct util_dynarray *loops);

static nir_loop *
enclosing_loop(nir_cf_node *node)
{
   for (nir_cf_node *p = node->parent; p; p = p->parent) {
      if (p->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(p);
   }
   return NULL;
}

/* A nested workgroup-striped loop issues adjacent elements together; later
 * iterations access other stripes. When a tile gives each lane only a few
 * elements, the replicated main loop is skipped on every outer iteration. */
static bool
local_id_stride(const nir_shader *shader, nir_scalar id, int64_t step)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic_or_null(id.def);
   if (!intr || intr->intrinsic != nir_intrinsic_load_local_invocation_id)
      return false;

   unsigned width = shader->info.workgroup_size[id.comp];
   return width > 1 && (step == width || step == -(int64_t)width);
}

static bool
nested_workgroup_stripe(const nir_shader *shader, const struct counted_loop *cl)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable || !enclosing_loop(&cl->loop->cf_node))
      return false;

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&cl->loop->cf_node));
   nir_phi_src *init = nir_phi_get_src_from_block(cl->iv, pre);
   nir_scalar start = nir_scalar_resolved(init->src.ssa, 0);
   if (!nir_scalar_is_alu(start))
      return local_id_stride(shader, start, cl->step);

   /* A backwards stripe starts at end - id, also emitted as end + (-id). */
   if (nir_scalar_alu_op(start) == nir_op_isub)
      return local_id_stride(shader, nir_scalar_chase_alu_src(start, 1), cl->step);
   if (nir_scalar_alu_op(start) == nir_op_iadd) {
      for (unsigned i = 0; i < 2; i++) {
         nir_scalar term = nir_scalar_chase_alu_src(start, i);
         if (nir_scalar_is_alu(term) && nir_scalar_alu_op(term) == nir_op_ineg &&
             local_id_stride(shader, nir_scalar_chase_alu_src(term, 0), cl->step))
            return true;
      }
   }
   return false;
}

/* Return the linear byte scale of `iv` in `addr` (e.g. 4 for (iv + k) << 2),
 * or 0 if `addr` is not an affine expression of `iv` with a positive scale. */
static uint64_t
shared_offset_iv_scale(nir_scalar s, nir_def *iv, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (s.def == iv)
      return 1;
   if (!depth || !nir_scalar_is_alu(s))
      return 0;
   const nir_op op = nir_scalar_alu_op(s);
   switch (op) {
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_u2u64:
   case nir_op_i2i32:
   case nir_op_i2i64:
      return shared_offset_iv_scale(nir_scalar_chase_alu_src(s, 0), iv, depth - 1);
   case nir_op_ishl: {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      if (!nir_scalar_is_const(sh))
         return 0;
      const uint64_t k = nir_scalar_as_uint(sh);
      if (k >= 16)
         return 0;
      const uint64_t base =
         shared_offset_iv_scale(nir_scalar_chase_alu_src(s, 0), iv, depth - 1);
      if (base > (UINT64_MAX >> k))
         return 0;
      return base << k;
   }
   case nir_op_imul:
   case nir_op_amul: {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar c = nir_scalar_chase_alu_src(s, i);
         if (!nir_scalar_is_const(c))
            continue;
         const int64_t v = nir_scalar_as_int(c);
         if (v > 0 && v <= 0x10000) {
            const uint64_t base =
               shared_offset_iv_scale(nir_scalar_chase_alu_src(s, 1 - i), iv,
                                      depth - 1);
            uint64_t prod = 0;
            if (__builtin_mul_overflow(base, (uint64_t)v, &prod))
               return 0;
            return prod;
         }
      }
      return 0;
   }
   case nir_op_iadd: {
      uint64_t a = shared_offset_iv_scale(nir_scalar_chase_alu_src(s, 0), iv, depth - 1);
      uint64_t b = shared_offset_iv_scale(nir_scalar_chase_alu_src(s, 1), iv, depth - 1);
      if (a && !b)
         return a;
      if (b && !a)
         return b;
      return 0;
   }
   default:
      return 0;
   }
}

/* A loop whose shared-memory index advances by at least the entire workgroup
 * shared allocation in a single iteration cannot execute two in-bounds
 * iterations, so unrolling it only adds dead code and cold guard overhead. */
static bool
loop_step_exceeds_shared_size(const nir_shader *shader, nir_loop *loop,
                              const struct counted_loop *cl)
{
   if (cl->step == 0 || shader->info.shared_size == 0)
      return false;
   const uint64_t step_abs = cl->step > 0 ? (uint64_t)cl->step : (uint64_t)-cl->step;
   if (step_abs < 8)
      return false;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         unsigned addr_src;
         if (intr->intrinsic == nir_intrinsic_load_shared)
            addr_src = 0;
         else if (intr->intrinsic == nir_intrinsic_store_shared)
            addr_src = 1;
         else
            continue;
         const uint64_t scale =
            shared_offset_iv_scale(nir_get_scalar(intr->src[addr_src].ssa, 0),
                                   &cl->iv->def, 8);
         uint64_t stride = 0;
         if (scale && (__builtin_mul_overflow(step_abs, scale, &stride) ||
                       stride >= shader->info.shared_size))
            return true;
      }
   }
   return false;
}

/* An upper bound on the iterations of a counting-up loop with a constant
 * bound whose induction variable starts at a value that cannot be negative,
 * such as a local invocation index, or UINT64_MAX when there is none. A start
 * of 0 runs longest: the test that ends that run ends every larger start's
 * run no later, as long as the largest start does not wrap by then. */
static uint64_t
max_trips(nir_shader *shader, const struct counted_loop *cl)
{
   if ((cl->rel != REL_LT && cl->rel != REL_LE) || cl->step <= 0 ||
       cl->offset_is_step || !nir_scalar_is_const(cl->bound))
      return UINT64_MAX;

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&cl->loop->cf_node));
   nir_phi_src *init = nir_phi_get_src_from_block(cl->iv, pre);
   if (!init)
      return UINT64_MAX;

   struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
   const int64_t start_max =
      nir_unsigned_upper_bound(shader, ranges, nir_get_scalar(init->src.ssa, 0));
   _mesa_hash_table_destroy(ranges, NULL);

   const int64_t max = cl->is_unsigned ? UINT32_MAX : INT32_MAX;
   if (start_max > max)
      return UINT64_MAX;

   /* From a start of 0, test k keeps the loop running while
    * k * step + offset < bound (<= bound for REL_LE). */
   const int64_t bound = cl->is_unsigned ? (int64_t)nir_scalar_as_uint(cl->bound)
                                         : nir_scalar_as_int(cl->bound);
   const int64_t room = bound - cl->offset + (cl->rel == REL_LE ? 1 : 0);
   const int64_t trips = room > 0 ? DIV_ROUND_UP(room, cl->step) : 0;
   if (start_max + trips * cl->step + cl->offset > max)
      return UINT64_MAX;

   /* A rotated loop runs its body once before the first test. */
   return trips + (cl->bottom ? 1 : 0);
}

/* The iterations per test for a counted loop, or 0 when it is not unrolled. */
static unsigned
choose_factor(nir_function_impl *impl, nir_loop *loop, struct counted_loop *cl,
              unsigned factor, unsigned max_instrs)
{
   if (nested_workgroup_stripe(impl->function->shader, cl))
      return 0;
   if (loop_step_exceeds_shared_size(impl->function->shader, loop, cl))
      return 0;

   int top_size = 0;
   int size = count_unrollable(loop, cl->nif, &top_size);
   if (size < 0)
      return 0;
   /* A guarded loop with a compact unconditional prefix and a larger nested
    * conditional body executes only the prefix unconditionally per iteration.
    * When 4x already fits the standard budget, allow 8x up to 2x total code
    * size as long as the unconditional hot path stays within half the budget
    * and unconditional in-flight loads stay within 16 dwords; backend
    * occupancy / register-file checks fall back to 4x if pressure grows. */
   const bool compact_guarded_prefix =
      top_size < size && (unsigned)size * 4 <= max_instrs &&
      (unsigned)top_size * factor <= max_instrs / 2 &&
      (unsigned)size * factor <= max_instrs * 2;
   const unsigned budget = compact_guarded_prefix ? max_instrs * 2 : max_instrs;

   /* A body too large for the full factor still gains from two
    * iterations per test when it fits the budget at half the factor. */
   unsigned f = factor;
   while (f > 2 && (unsigned)size * f > budget)
      f /= 2;
   /* Beyond four iterations, keep the unconditional reads the unrolled body
    * has in flight within 16 registers: more only raises register pressure
    * (the scheduler issues them together) and can cost occupancy. Loads inside
    * nested `if` bodies are guarded and do not issue together across
    * iterations. */
   if (f > 4) {
      const unsigned dwords = loaded_dwords_in_loop(impl, loop, true);
      while (f > 4 && dwords * f > 16)
         f /= 2;
   }
   /* The main loop runs only while f iterations remain. One that can run
    * at most once saves at most f - 1 loop tests, and its guard and its f
    * copies of the body cost the same whether it runs or not: keep the
    * loop as it is. */
   const uint64_t trips = max_trips(impl->function->shader, cl);
   while (f >= 2 && 2 * f > trips)
      f /= 2;
   if (f < 2)
      return 0;
   return (unsigned)size * f > budget ? 0 : f;
}

/* Computable before `outer` from values defined before it, without memory
 * reads that could fault there (push constants, preamble values, system
 * values, and reads that may run speculatively from memory the shader does
 * not write, such as uniform buffers): a value every iteration of `outer`
 * sees alike. */
static bool
outer_invariant(nir_def *def, nir_loop *outer, unsigned depth)
{
   if (def_outside_loop(def, outer))
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
         if (!outer_invariant(alu->src[i].src.ssa, outer, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      const bool speculatable_read =
         nir_intrinsic_can_reorder(intr) && nir_intrinsic_has_access(intr) &&
         (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE);
      const bool safe = intr->intrinsic == nir_intrinsic_load_push_constant ||
                        intr->intrinsic == nir_intrinsic_load_preamble ||
                        (info->num_srcs == 0 && nir_intrinsic_can_reorder(intr)) ||
                        speculatable_read;
      if (!safe || !info->has_dest)
         return false;
      for (unsigned i = 0; i < info->num_srcs; ++i)
         if (!outer_invariant(intr->src[i].ssa, outer, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

/* As rematerialize(), relative to `outer`. */
static nir_def *
remat_outer(nir_builder *b, nir_def *def, nir_loop *outer, struct hash_table *remap)
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
         remat_outer(b, alu->src[i].src.ssa, outer, remap);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         remat_outer(b, intr->src[i].ssa, outer, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *result = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, result);
   return result;
}

static unsigned
count_instrs(nir_loop *loop)
{
   unsigned n = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block)
         ++n;
   }
   return n;
}

/* Values the outer loop leaves to the code after it become phis of the two
 * copies' values. Fine for ALU results and constants; the backend reads some
 * intrinsic results from their defining instruction (bindless handles,
 * barycentrics), so any intrinsic or texture result used after the loop
 * keeps the outer loop as it is. */
static bool
live_outs_are_alu(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def || instr->type == nir_instr_type_alu ||
             instr->type == nir_instr_type_load_const ||
             instr->type == nir_instr_type_undef ||
             instr->type == nir_instr_type_phi)
            continue;
         nir_foreach_use_including_if(src, def) {
            if (!block_in_loop(use_block(src), loop))
               return false;
         }
      }
   }
   return true;
}

/* The outer loop may run under an if whose condition differs between
 * invocations: nothing in it may need uniform control flow or the other
 * invocations (barriers, subgroup and quad operations, derivatives), and only
 * its own breaks and continues may leave it. Derefs would need
 * rematerializing before the conversion to registers: refused. */
static bool
splittable_outer(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            if (nir_instr_as_jump(instr)->type != nir_jump_break &&
                nir_instr_as_jump(instr)->type != nir_jump_continue)
               return false;
            break;
         case nir_instr_type_call:
         case nir_instr_type_deref:
            return false;
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent(intr))
               return false;
            /* A lane may wait on another lane of its subgroup through
             * memory: the two copies would run one after the other. Atomic
             * loads and stores of shared memory carry only ACCESS_ATOMIC. */
            if (strstr(nir_intrinsic_infos[intr->intrinsic].name, "atomic") ||
                (nir_intrinsic_has_access(intr) &&
                 (nir_intrinsic_access(intr) &
                  (ACCESS_COHERENT | ACCESS_VOLATILE | ACCESS_ATOMIC))))
               return false;
            switch (intr->intrinsic) {
            case nir_intrinsic_barrier:
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy_coarse:
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
            case nir_intrinsic_demote:
            case nir_intrinsic_demote_if:
               return false;
            default:
               break;
            }
            break;
         }
         default:
            break;
         }
      }
   }
   return true;
}

/*
 * A counted loop nested in another loop whose trip count the outer loop does
 * not change (a box filter's or a SAD window's x - r .. x + r): decide once,
 * before the outer loop, whether the unrolled form can run (at least
 * `factor` iterations), and otherwise run a copy of the outer loop with the
 * inner loop as it was. A 3-tap window then pays nothing for the unroll's
 * guard and structure on every outer iteration; an 11-tap one keeps it.
 * Both copies compute the same values, so the test only picks the faster.
 */
static bool
loop_has_nested_loop(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (loop_has_nested_loop(&nif->then_list) ||
             loop_has_nested_loop(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
outer_peelable(nir_loop *outer, nir_if *terminator, nir_loop *inner)
{
   nir_foreach_block_in_cf_node(block, &outer->cf_node) {
      bool in_inner = false;
      for (nir_cf_node *p = block->cf_node.parent; p != &outer->cf_node;
           p = p->parent) {
         if (p == &inner->cf_node) {
            in_inner = true;
            break;
         }
      }
      if (in_inner)
         continue;

      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            if (block->cf_node.parent != &terminator->cf_node)
               return false;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent(intr))
               return false;
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
               return false;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            break;
         case nir_instr_type_call:
            return false;
         default:
            break;
         }
      }
   }
   return true;
}

static unsigned
count_nested_loops(struct exec_list *list)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop) {
         n += 1 + count_nested_loops(&nir_cf_node_as_loop(node)->body);
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         n += count_nested_loops(&nif->then_list) +
              count_nested_loops(&nif->else_list);
      }
   }
   return n;
}

static nir_loop *
nth_loop_in_list(struct exec_list *list, unsigned *index)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         nir_loop *hit = nth_loop_in_list(&nif->then_list, index);
         if (hit)
            return hit;
         hit = nth_loop_in_list(&nif->else_list, index);
         if (hit)
            return hit;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(node);
         nir_loop *hit = nth_loop_in_list(&lp->body, index);
         if (hit)
            return hit;
         if (*index == 0)
            return lp;
         (*index)--;
      }
   }
   return NULL;
}

static bool
loop_ordinal_in_list(struct exec_list *list, nir_loop *target, unsigned *index)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (loop_ordinal_in_list(&nif->then_list, target, index) ||
             loop_ordinal_in_list(&nif->else_list, target, index))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(node);
         if (loop_ordinal_in_list(&lp->body, target, index))
            return true;
         if (lp == target)
            return true;
         (*index)++;
      }
   }
   return false;
}

/* Overflow-free exact trip-count equality predicate (trip == exact_n) for a
 * unit-step top-tested counted loop with offset == 0. */
static nir_def *
exact_trip_cond(nir_builder *b, const struct counted_loop *cl, nir_def *iv0,
                nir_def *bound, uint32_t exact_n)
{
   const bool u = cl->is_unsigned;
   const uint32_t max_u = u ? UINT32_MAX : (uint32_t)INT32_MAX;
   const uint32_t min_u = u ? 0u : (uint32_t)INT32_MIN;
   nir_def *ord, *diff, *no_wrap;

   switch (cl->rel) {
   case REL_LT:
      ord = u ? nir_ult(b, iv0, bound) : nir_ilt(b, iv0, bound);
      diff = nir_ieq_imm(b, nir_isub(b, bound, iv0), exact_n);
      return nir_iand(b, ord, diff);
   case REL_LE:
      ord = u ? nir_uge(b, bound, iv0) : nir_ige(b, bound, iv0);
      diff = nir_ieq_imm(b, nir_isub(b, bound, iv0), exact_n - 1);
      no_wrap = nir_ine_imm(b, bound, max_u);
      return nir_iand(b, nir_iand(b, ord, diff), no_wrap);
   case REL_GT:
      ord = u ? nir_ult(b, bound, iv0) : nir_ilt(b, bound, iv0);
      diff = nir_ieq_imm(b, nir_isub(b, iv0, bound), exact_n);
      return nir_iand(b, ord, diff);
   default: /* REL_GE */
      ord = u ? nir_uge(b, iv0, bound) : nir_ige(b, iv0, bound);
      diff = nir_ieq_imm(b, nir_isub(b, iv0, bound), exact_n - 1);
      no_wrap = nir_ine_imm(b, bound, min_u);
      return nir_iand(b, nir_iand(b, ord, diff), no_wrap);
   }
}

/* Peel `count` guaranteed straight-line iterations of a simple top-tested
 * counted loop. When `keep_tail` is false (exact trip == count), remove the
 * loop completely; when true (trip >= count), keep the original loop as the
 * compact remainder loop after the straight-line prefix. */
static bool
peel_straight_prefix(nir_function_impl *impl, nir_loop *loop, unsigned count,
                     bool keep_tail)
{
   struct counted_loop cl;
   if (!analyze_loop(loop, &cl) || cl.bottom)
      return false;
   if (count_unrollable(loop, cl.nif, NULL) < 0)
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node)
      nir_lower_phis_to_regs_block(block, true);
   nir_foreach_block_in_cf_node(block, &loop->cf_node)
      nir_lower_ssa_defs_to_regs_block(block);

   nir_cf_list head, rest;
   nir_cf_extract(&head, nir_before_cf_list(&loop->body),
                  nir_before_cf_node(&cl.nif->cf_node));
   nir_cf_extract(&rest, nir_after_cf_node(&cl.nif->cf_node),
                  nir_after_cf_list(&loop->body));

   nir_cf_list copies[2 * 4];
   assert(count >= 1 && count <= 4);
   for (unsigned i = 0; i < count; ++i) {
      struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list_clone(&copies[2 * i], &head, loop->cf_node.parent, remap);
      nir_cf_list_clone(&copies[2 * i + 1], &rest, loop->cf_node.parent, remap);
      _mesa_hash_table_destroy(remap, NULL);
   }

   for (unsigned i = 0; i < count; ++i) {
      nir_cf_reinsert(&copies[2 * i], nir_before_cf_node(&loop->cf_node));
      nir_cf_reinsert(&copies[2 * i + 1], nir_before_cf_node(&loop->cf_node));
   }

   if (keep_tail) {
      nir_cf_reinsert(&head, nir_before_cf_node(&cl.nif->cf_node));
      nir_cf_reinsert(&rest, nir_after_cf_list(&loop->body));
      loop->partially_unrolled = true;
   } else {
      /* On the exiting iteration (trip == count), the original top-tested loop
       * executes `head` once more before breaking, writing the final header-phi
       * values into their SSA-def registers. */
      nir_cf_reinsert(&head, nir_before_cf_node(&loop->cf_node));
      nir_cf_delete(&rest);
      nir_cf_node_remove(&loop->cf_node);
   }
   return true;
}

struct small_peel_rec {
   nir_loop *loop;
   unsigned count;
   bool keep_tail;
};

static bool
loop_has_8bit_load(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_global:
         case nir_intrinsic_load_global_constant:
         case nir_intrinsic_load_agx:
         case nir_intrinsic_load_constant_agx:
         case nir_intrinsic_load_ssbo:
            if (intr->def.bit_size == 8 && intr->def.num_components == 1)
               return true;
            break;
         default:
            break;
         }
      }
   }
   return false;
}

static bool
unswitch_nested(nir_function_impl *impl, unsigned factor, unsigned max_instrs,
                struct set *skip_runtime_unroll,
                struct util_dynarray *small_peels,
                struct util_dynarray *outer3_peels)
{
   /* With maximal reconvergence the outer loop's invocations must stay
    * together; the split could separate them. */
   if (impl->function->shader->info.maximally_reconverges)
      return false;

   struct util_dynarray loops;
   util_dynarray_init(&loops, NULL);
   collect_loops(&impl->body, &loops);
   struct set *done = _mesa_pointer_set_create(NULL);
   bool progress = false;

   util_dynarray_foreach(&loops, nir_loop *, lp) {
      nir_loop *inner = *lp;
      nir_loop *outer = enclosing_loop(&inner->cf_node);
      struct counted_loop cl;
      if (!outer || _mesa_set_search(done, outer) || enclosing_loop(&outer->cf_node) ||
          outer->partially_unrolled || nir_loop_has_continue_construct(outer) ||
          _mesa_set_search(skip_runtime_unroll, inner) ||
          !analyze_loop(inner, &cl))
         continue;
      const unsigned f = choose_factor(impl, inner, &cl, factor, max_instrs);
      nir_block *pre_inner = nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
      nir_phi_src *init = nir_phi_get_src_from_block(cl.iv, pre_inner);
      nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&outer->cf_node));
      if (f < 4 || !init ||
          (!nir_scalar_is_const(cl.bound) && !outer_invariant(cl.bound.def, outer, 8)))
         continue;
      if (!outer_invariant(init->src.ssa, outer, 8) || count_instrs(outer) > 256 ||
          !splittable_outer(outer) || !live_outs_are_alu(outer) ||
          (nir_block_first_instr(after) &&
           nir_block_first_instr(after)->type == nir_instr_type_phi))
         continue;

      /* The test, before the outer loop: would the main loop's first
       * iteration run? (For a rotated loop, the exact-tail form.) */
      nir_builder b = nir_builder_at(nir_before_cf_node(&outer->cf_node));
      struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
      nir_def *bound = nir_scalar_is_const(cl.bound)
                          ? nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(cl.bound))
                          : nir_channel(&b, remat_outer(&b, cl.bound.def, outer, remap),
                                        cl.bound.comp);
      nir_def *iv0 = remat_outer(&b, init->src.ssa, outer, remap);
      _mesa_hash_table_destroy(remap, NULL);
      if (cl.offset && !cl.bottom)
         iv0 = nir_iadd_imm(&b, iv0, cl.offset);
      nir_def *g = guard_bound(&b, &cl, bound, NULL, f);
      nir_def *cond;
      if (cl.rel == REL_LT || cl.rel == REL_LE)
         cond = cl.is_unsigned ? nir_ult(&b, iv0, g) : nir_ilt(&b, iv0, g);
      else
         cond = cl.is_unsigned ? nir_ult(&b, g, iv0) : nir_ilt(&b, g, iv0);

      /* For compact 2D stencils / box filters with f >= 8, also specialize
       * the small-window else arm (4 <= trip < 8 -> straight-line 4x + tail,
       * and trip == 3 -> straight-line 3x with zero inner loop) so 3x3..7x7
       * windows vectorize into u8x3/u8x4 loads instead of scalar byte loops. */
      const bool split_small =
         f >= 8 && !cl.bottom && cl.offset == 0 &&
         (cl.step == 1 || cl.step == -1) &&
         count_instrs(outer) <= 96 && !loop_has_nested_loop(&inner->body) &&
         loop_has_8bit_load(inner);
      nir_def *cond4 = NULL, *cond3 = NULL, *cond3_outer = NULL;
      unsigned inner_ord = 0;
      if (split_small) {
         nir_def *g4 = guard_bound(&b, &cl, bound, NULL, 4);
         if (cl.rel == REL_LT || cl.rel == REL_LE)
            cond4 = cl.is_unsigned ? nir_ult(&b, iv0, g4) : nir_ilt(&b, iv0, g4);
         else
            cond4 = cl.is_unsigned ? nir_ult(&b, g4, iv0) : nir_ilt(&b, g4, iv0);
         cond3 = exact_trip_cond(&b, &cl, iv0, bound, 3);
         bool found_ord = loop_ordinal_in_list(&outer->body, inner, &inner_ord);
         assert(found_ord);

         struct counted_loop ocl;
         nir_block *pre_outer = nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));
         if (analyze_loop_opts(outer, &ocl, false, true) && !ocl.bottom &&
             ocl.offset == 0 && (ocl.step == 1 || ocl.step == -1) &&
             count_nested_loops(&outer->body) == 1 &&
             count_instrs(outer) <= 64 &&
             outer_peelable(outer, ocl.nif, inner)) {
            nir_phi_src *oinit = nir_phi_get_src_from_block(ocl.iv, pre_outer);
            if (oinit && (nir_scalar_is_const(ocl.bound) ||
                          outer_invariant(ocl.bound.def, outer, 8))) {
               struct hash_table *oremap = _mesa_pointer_hash_table_create(NULL);
               nir_def *obound =
                  nir_scalar_is_const(ocl.bound)
                     ? nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(ocl.bound))
                     : nir_channel(&b, remat_outer(&b, ocl.bound.def, outer, oremap),
                                   ocl.bound.comp);
               _mesa_hash_table_destroy(oremap, NULL);
               cond3_outer = exact_trip_cond(&b, &ocl, oinit->src.ssa, obound, 3);
            }
         }
      }

      /* Values of the outer loop through registers, so both copies write
       * the same ones and the code after reads them whichever ran. Phi
       * sources are written in their own predecessor: hoisted into the arms
       * of an if, an inner latch value would come back as a phi of the two
       * arms, which the inner loop's analysis does not take. */
      nir_foreach_block_in_cf_node(block, &outer->cf_node)
         nir_lower_phis_to_regs_block(block, true);
      nir_foreach_block_in_cf_node(block, &outer->cf_node)
         nir_lower_ssa_defs_to_regs_block(block);

      /* The lowering writes the outer phis' initial values at the end of the
       * preheader, after cond: the if goes after those, right before the
       * outer loop, so both copies start from them. */
      b.cursor = nir_before_cf_node(&outer->cf_node);
      nir_if *nif = nir_push_if(&b, cond);
      nir_pop_if(&b, nif);
      nir_cf_list list, copy;
      nir_cf_extract(&list, nir_before_cf_node(&outer->cf_node),
                     nir_after_cf_node(&outer->cf_node));
      if (split_small) {
         b.cursor = nir_before_cf_list(&nif->else_list);
         nir_if *nif4 = nir_push_if(&b, cond4);
         nir_push_else(&b, nif4);
         nir_if *nif3 = nir_push_if(&b, cond3);
         nir_pop_if(&b, nif3);
         nir_pop_if(&b, nif4);

         nir_cf_list copy4, copy3;
         struct hash_table *cmap4 = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy4, &list, &nif4->cf_node, cmap4);
         _mesa_hash_table_destroy(cmap4, NULL);

         struct hash_table *cmap3 = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy3, &list, &nif3->cf_node, cmap3);
         _mesa_hash_table_destroy(cmap3, NULL);

         inner->partially_unrolled = true;
         struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy, &list, &nif3->cf_node, cmap);
         _mesa_hash_table_destroy(cmap, NULL);
         inner->partially_unrolled = false;

         nir_cf_reinsert(&list, nir_before_cf_list(&nif->then_list));
         nir_cf_reinsert(&copy4, nir_before_cf_list(&nif4->then_list));
         nir_cf_reinsert(&copy, nir_before_cf_list(&nif3->else_list));

         unsigned idx4 = inner_ord;
         nir_loop *inner4 = nth_loop_in_list(&nif4->then_list, &idx4);
         assert(inner4);
         struct small_peel_rec r4 = {inner4, 4, true};
         util_dynarray_append(small_peels, r4);

         if (cond3_outer) {
            b.cursor = nir_before_cf_list(&nif3->then_list);
            nir_if *nif33 = nir_push_if(&b, cond3_outer);
            nir_pop_if(&b, nif33);

            nir_cf_list copy33;
            struct hash_table *cmap33 = _mesa_pointer_hash_table_create(NULL);
            nir_cf_list_clone(&copy33, &copy3, &nif33->cf_node, cmap33);
            _mesa_hash_table_destroy(cmap33, NULL);

            nir_cf_reinsert(&copy33, nir_before_cf_list(&nif33->then_list));
            nir_cf_reinsert(&copy3, nir_before_cf_list(&nif33->else_list));

            unsigned idx33 = inner_ord;
            nir_loop *inner33 = nth_loop_in_list(&nif33->then_list, &idx33);
            assert(inner33);
            struct small_peel_rec r33 = {inner33, 3, false};
            util_dynarray_append(small_peels, r33);

            nir_loop *outer33 = enclosing_loop(&inner33->cf_node);
            assert(outer33);
            struct small_peel_rec ro33 = {outer33, 3, false};
            util_dynarray_append(outer3_peels, ro33);

            unsigned idx3 = inner_ord;
            nir_loop *inner3 = nth_loop_in_list(&nif33->else_list, &idx3);
            assert(inner3);
            struct small_peel_rec r3 = {inner3, 3, false};
            util_dynarray_append(small_peels, r3);
         } else {
            nir_cf_reinsert(&copy3, nir_before_cf_list(&nif3->then_list));

            unsigned idx3 = inner_ord;
            nir_loop *inner3 = nth_loop_in_list(&nif3->then_list, &idx3);
            assert(inner3);
            struct small_peel_rec r3 = {inner3, 3, false};
            util_dynarray_append(small_peels, r3);
         }
      } else {
         /* The copy keeps this inner loop as it is (clones copy the flag); its
          * other loops stay candidates. */
         inner->partially_unrolled = true;
         struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy, &list, &nif->cf_node, cmap);
         _mesa_hash_table_destroy(cmap, NULL);
         inner->partially_unrolled = false;
         nir_cf_reinsert(&list, nir_before_cf_list(&nif->then_list));
         nir_cf_reinsert(&copy, nir_before_cf_list(&nif->else_list));
      }
      _mesa_set_add(done, outer);
      progress = true;
   }

   _mesa_set_destroy(done, NULL);
   util_dynarray_fini(&loops);
   return progress;
}

static void
collect_loops(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops(&nif->then_list, loops);
         collect_loops(&nif->else_list, loops);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_loops(&loop->body, loops);
         util_dynarray_append(loops, loop);
         break;
      }
      default:
         break;
      }
   }
}

/* True if `loop` performs no memory writes, atomics, barriers, calls, or
 * early exits (return/halt/terminate/demote). */
static bool
loop_is_readonly(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (nir_block_ends_in_return_or_halt(block))
         return false;
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (!(nir_intrinsic_infos[intr->intrinsic].flags &
               NIR_INTRINSIC_CAN_ELIMINATE))
            return false;
         switch (intr->intrinsic) {
         case nir_intrinsic_terminate:
         case nir_intrinsic_terminate_if:
         case nir_intrinsic_demote:
         case nir_intrinsic_demote_if:
            return false;
         default:
            break;
         }
      }
   }
   return true;
}

/* True if `inner` contains a non-volatile, non-coherent global load whose
 * address does not depend on any header phi of `outer` (so flattening `inner`
 * exposes an outer-loop-invariant load). */
static bool
inner_has_outer_invariant_load(nir_function_impl *impl, nir_loop *outer,
                               nir_loop *inner)
{
   BITSET_WORD *outer_varying =
      calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
   nir_block *outer_hdr = nir_loop_first_block(outer);

   nir_foreach_block_in_cf_node(block, &outer->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         if (instr->type == nir_instr_type_phi && block == outer_hdr) {
            BITSET_SET(outer_varying, def->index);
            continue;
         }
         struct varying_state st = {outer_varying, false};
         nir_foreach_src(instr, src_varies, &st);
         if (st.any)
            BITSET_SET(outer_varying, def->index);
      }
   }

   bool found = false;
   nir_foreach_block_in_cf_node(block, &inner->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if ((intr->intrinsic == nir_intrinsic_load_global ||
              intr->intrinsic == nir_intrinsic_load_global_constant) &&
             !(nir_intrinsic_access(intr) &
               (ACCESS_VOLATILE | ACCESS_COHERENT)) &&
             !BITSET_TEST(outer_varying, intr->src[0].ssa->index)) {
            found = true;
            break;
         }
      }
      if (found)
         break;
   }

   free(outer_varying);
   return found;
}

static bool
block_in_cf_list(nir_block *block, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node == &block->cf_node)
         return true;
      for (nir_cf_node *p = block->cf_node.parent; p; p = p->parent) {
         if (p == node)
            return true;
      }
   }
   return false;
}

/*
 * When a read-only outer loop encloses a counted inner loop (0 .. bound - 1,
 * step 1) whose runtime bound is outer-invariant and whose body reads an
 * outer-invariant global address (e.g., descriptor matching where `desc1`
 * depends only on the outer thread index and `bound == factor`), unswitch the
 * outer loop on `bound == factor` and substitute `bound = factor` inside the
 * fast arm. Standard nir_opt_loop_unroll then flattens the inner loop to
 * straight-line code and agx_nir_hoist_readonly_loop_loads hoists the
 * outer-invariant loads out of the outer loop.
 */
static bool
specialize_exact_nested(nir_function_impl *impl, unsigned factor,
                        unsigned max_instrs, struct set *skip_runtime_unroll,
                        unsigned *widest)
{
   if (impl->function->shader->info.maximally_reconverges)
      return false;

   struct util_dynarray loops;
   util_dynarray_init(&loops, NULL);
   collect_loops(&impl->body, &loops);
   struct set *done = _mesa_pointer_set_create(NULL);
   bool progress = false;

   util_dynarray_foreach(&loops, nir_loop *, lp) {
      nir_loop *inner = *lp;
      nir_loop *outer = enclosing_loop(&inner->cf_node);
      struct counted_loop cl;
      if (!outer || _mesa_set_search(done, outer) ||
          outer->partially_unrolled || nir_loop_has_continue_construct(outer) ||
          !analyze_loop(inner, &cl))
         continue;

      const unsigned f = choose_factor(impl, inner, &cl, factor, max_instrs);
      nir_block *pre_inner =
         nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
      nir_phi_src *init = nir_phi_get_src_from_block(cl.iv, pre_inner);
      nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&outer->cf_node));
      if (f < 4 || !init || cl.step != 1 || cl.rel != REL_LT ||
          nir_scalar_is_const(cl.bound) || cl.bound.def->num_components != 1 ||
          !outer_invariant(cl.bound.def, outer, 8) ||
          !nir_src_is_const(init->src) || nir_src_as_uint(init->src) != 0)
         continue;
      if (count_instrs(outer) > 256 || !splittable_outer(outer) ||
          !loop_is_readonly(outer) || !live_outs_are_alu(outer) ||
          (nir_block_first_instr(after) &&
           nir_block_first_instr(after)->type == nir_instr_type_phi) ||
          !inner_has_outer_invariant_load(impl, outer, inner))
         continue;

      nir_builder b = nir_builder_at(nir_before_cf_node(&outer->cf_node));
      nir_def *bound_def = cl.bound.def;
      if (!def_outside_loop(bound_def, outer)) {
         struct hash_table *bmap = _mesa_pointer_hash_table_create(NULL);
         nir_def *hoisted = remat_outer(&b, bound_def, outer, bmap);
         _mesa_hash_table_destroy(bmap, NULL);
         nir_foreach_use_including_if_safe(src, bound_def) {
            if (block_in_loop(use_block(src), outer))
               nir_src_rewrite(src, hoisted);
         }
         bound_def = hoisted;
      }

      nir_def *cond = nir_ieq_imm(&b, bound_def, f);

      nir_foreach_block_in_cf_node(block, &outer->cf_node)
         nir_lower_phis_to_regs_block(block, true);
      nir_foreach_block_in_cf_node(block, &outer->cf_node)
         nir_lower_ssa_defs_to_regs_block(block);

      b.cursor = nir_before_cf_node(&outer->cf_node);
      nir_if *nif = nir_push_if(&b, cond);
      nir_pop_if(&b, nif);

      nir_cf_list list, copy;
      nir_cf_extract(&list, nir_before_cf_node(&outer->cf_node),
                     nir_after_cf_node(&outer->cf_node));
      /* Keep the fallback copy's inner loop un-unrolled (clones copy the flag)
       * so the cold fallback path does not inflate whole-shader GPR pressure
       * and reduce occupancy for the specialized fast arm. */
      inner->partially_unrolled = true;
      struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list_clone(&copy, &list, &nif->cf_node, cmap);
      _mesa_hash_table_destroy(cmap, NULL);
      inner->partially_unrolled = false;
      nir_cf_reinsert(&list, nir_before_cf_list(&nif->then_list));
      nir_cf_reinsert(&copy, nir_before_cf_list(&nif->else_list));

      /* Inside the exact-trip-count arm, substitute bound_def -> f so the
       * inner loop becomes compile-time constant and nir_opt_loop_unroll
       * flattens it completely without a remainder loop. Also clone any pure
       * constant-folded ALU expression of `bound_def` defined outside the arm
       * (such as `0 < bound_def` guarding the inner loop) so constant folding
       * eliminates redundant runtime guards and exposes outer-invariant loads
       * directly in the outer loop body. */
      b.cursor = nir_before_cf_list(&nif->then_list);
      nir_def *imm_f = nir_imm_intN_t(&b, f, bound_def->bit_size);
      nir_foreach_use_including_if_safe(src, bound_def) {
         if (block_in_cf_list(use_block(src), &nif->then_list))
            nir_src_rewrite(src, imm_f);
      }

      struct hash_table *const_map = _mesa_pointer_hash_table_create(NULL);
      _mesa_hash_table_insert(const_map, bound_def, imm_f);
      nir_def *work[16];
      unsigned nw = 0;
      work[nw++] = bound_def;
      for (unsigned wi = 0; wi < nw; ++wi) {
         nir_def *cur = work[wi];
         nir_foreach_use(u_src, cur) {
            nir_instr *ui = nir_src_use_instr(u_src);
            if (ui->type != nir_instr_type_alu ||
                block_in_cf_list(ui->block, &nif->then_list) ||
                block_in_cf_list(ui->block, &nif->else_list))
               continue;
            nir_alu_instr *ualu = nir_instr_as_alu(ui);
            if (_mesa_hash_table_search(const_map, &ualu->def) ||
                ualu->def.num_components != 1)
               continue;
            switch (ualu->op) {
            case nir_op_idiv:
            case nir_op_udiv:
            case nir_op_imod:
            case nir_op_umod:
            case nir_op_irem:
               continue;
            default:
               break;
            }
            bool all_const = true;
            for (unsigned s = 0; s < nir_op_infos[ualu->op].num_inputs; ++s) {
               nir_def *sd = ualu->src[s].src.ssa;
               if (!_mesa_hash_table_search(const_map, sd) &&
                   !nir_def_is_const(sd)) {
                  all_const = false;
                  break;
               }
            }
            if (!all_const)
               continue;
            bool used_in_then = false;
            nir_foreach_use_including_if(t_src, &ualu->def) {
               if (block_in_cf_list(use_block(t_src), &nif->then_list)) {
                  used_in_then = true;
                  break;
               }
            }
            if (!used_in_then)
               continue;
            nir_instr *clone = nir_instr_clone_deep(b.shader, &ualu->instr,
                                                    const_map);
            nir_builder_instr_insert(&b, clone);
            nir_def *cdef = nir_instr_def(clone);
            nir_foreach_use_including_if_safe(t_src, &ualu->def) {
               if (block_in_cf_list(use_block(t_src), &nif->then_list))
                  nir_src_rewrite(t_src, cdef);
            }
            _mesa_hash_table_insert(const_map, &ualu->def, cdef);
            if (nw < ARRAY_SIZE(work))
               work[nw++] = &ualu->def;
         }
      }
      _mesa_hash_table_destroy(const_map, NULL);
      if (widest)
         *widest = MAX2(*widest, f);
      _mesa_set_add(skip_runtime_unroll, inner);
      _mesa_set_add(done, outer);
      progress = true;
   }

   _mesa_set_destroy(done, NULL);
   util_dynarray_fini(&loops);
   return progress;
}

static bool
is_loop_invariant_pure(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (!depth)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_idiv:
      case nir_op_udiv:
      case nir_op_imod:
      case nir_op_umod:
      case nir_op_irem:
         return false;
      default:
         break;
      }
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!is_loop_invariant_pure(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      if (!info->has_dest || !nir_intrinsic_can_reorder(intr) ||
          lane_dependent(intr))
         return false;
      const bool safe =
         intr->intrinsic == nir_intrinsic_load_push_constant ||
         intr->intrinsic == nir_intrinsic_load_preamble ||
         info->num_srcs == 0 ||
         (nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE));
      if (!safe)
         return false;
      for (unsigned i = 0; i < info->num_srcs; ++i)
         if (!is_loop_invariant_pure(intr->src[i].ssa, loop, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

static bool
header_exit_computable_at_entry(nir_def *def, nir_loop *loop, nir_block *header,
                                unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (!depth)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->block != header)
      return false;
   if (instr->type == nir_instr_type_phi)
      return nir_phi_get_src_from_block(nir_instr_as_phi(instr),
                                        nir_loop_predecessor_block(loop)) != NULL;
   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_undef)
      return true;
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!header_exit_computable_at_entry(alu->src[i].src.ssa, loop, header,
                                              depth - 1))
            return false;
      return true;
   }
   return false;
}

static nir_def *
emit_header_exit_at_entry(nir_builder *b, nir_def *def, nir_loop *loop,
                          nir_block *header, nir_block *pre,
                          struct hash_table *remap)
{
   if (def_outside_loop(def, loop))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_phi) {
      nir_phi_src *ps = nir_phi_get_src_from_block(nir_instr_as_phi(instr), pre);
      _mesa_hash_table_insert(remap, def, ps->src.ssa);
      return ps->src.ssa;
   }
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         emit_header_exit_at_entry(b, alu->src[i].src.ssa, loop, header, pre,
                                   remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *res = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, res);
   return res;
}

static bool
cf_node_has_jump(nir_cf_node *node)
{
   nir_foreach_block_in_cf_node(block, node) {
      if (nir_block_ends_in_jump(block))
         return true;
   }
   return false;
}

/* Hoist loop-invariant non-volatile/non-coherent global loads out of read-only
 * loops (no stores/atomics/barriers/calls). Loads in unconditional blocks
 * before the first exit always execute on iteration 0 and hoist directly to
 * the preheader (capped at 16 dwords); unconditional body loads after a single
 * top exit hoist into a non-speculative `if (!init_exit)` guard before it. */
bool
agx_nir_hoist_readonly_loop_loads(nir_shader *shader)
{
   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      bool progress = false;
      struct util_dynarray loops;
      util_dynarray_init(&loops, NULL);
      collect_loops(&impl->body, &loops);

      util_dynarray_foreach(&loops, nir_loop *, lp) {
         nir_loop *loop = *lp;
         if (!nir_loop_has_back_edge(loop) ||
             nir_loop_has_continue_construct(loop) || !loop_is_readonly(loop))
            continue;

         nir_block *pre = nir_loop_predecessor_block(loop);
         nir_block *header = nir_loop_first_block(loop);

         /* 1. Hoist invariant global loads in unconditional blocks before the
          * first jump/exit directly to the preheader (always executed at least
          * once), bounded to 16 dwords to preserve register occupancy. */
         unsigned hoisted_dwords = 0;
         struct hash_table *hremap = _mesa_pointer_hash_table_create(NULL);
         foreach_list_typed(nir_cf_node, cf, node, &loop->body) {
            if (cf->type == nir_cf_node_block) {
               nir_block *blk = nir_cf_node_as_block(cf);
               nir_foreach_instr_safe(instr, blk) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if ((intr->intrinsic != nir_intrinsic_load_global &&
                       intr->intrinsic != nir_intrinsic_load_global_constant) ||
                      (nir_intrinsic_access(intr) &
                       (ACCESS_VOLATILE | ACCESS_COHERENT |
                        ACCESS_BUFFER_BINDING_AGX)))
                     continue;
                  if (!is_loop_invariant_pure(intr->src[0].ssa, loop, 10))
                     continue;
                  const unsigned dwords =
                     intr->def.num_components *
                     DIV_ROUND_UP(intr->def.bit_size, 32);
                  if (hoisted_dwords + dwords > 16)
                     continue;

                  nir_builder b = nir_builder_at(nir_after_block_before_jump(pre));
                  nir_def *addr = remat_outer(&b, intr->src[0].ssa, loop, hremap);
                  nir_src_rewrite(&intr->src[0], addr);
                  nir_instr_remove(&intr->instr);
                  nir_instr_insert(nir_after_block_before_jump(pre), &intr->instr);
                  hoisted_dwords += dwords;
                  progress = true;
               }
               if (nir_block_ends_in_jump(blk))
                  break;
            } else if (cf_node_has_jump(cf)) {
               break;
            }
         }
         _mesa_hash_table_destroy(hremap, NULL);

         /* 2. For a single-top-exit loop, hoist unconditional body loads
          * guarded by `if (!init_exit)` around the loop (zero speculation). */
         nir_cf_node *next_cf = nir_cf_node_next(&header->cf_node);
         if (!next_cf || next_cf->type != nir_cf_node_if)
            continue;
         nir_if *top_exit = nir_cf_node_as_if(next_cf);
         const bool break_in_then =
            branch_is_lone_break(&top_exit->then_list) &&
            branch_is_empty(&top_exit->else_list);
         const bool break_in_else =
            branch_is_lone_break(&top_exit->else_list) &&
            branch_is_empty(&top_exit->then_list);
         if (!break_in_then && !break_in_else)
            continue;

         /* Only one break in the whole loop, and only header phis live out. */
         unsigned breaks = 0;
         bool ok = true;
         nir_foreach_block_in_cf_node(block, &loop->cf_node) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_jump) {
                  if (nir_instr_as_jump(instr)->type == nir_jump_break)
                     breaks++;
                  else
                     ok = false;
               }
               nir_def *def = nir_instr_def(instr);
               if (def && !(instr->type == nir_instr_type_phi && block == header)) {
                  nir_foreach_use_including_if(use, def) {
                     if (!block_in_loop(use_block(use), loop)) {
                        ok = false;
                        break;
                     }
                  }
               }
            }
         }
         nir_block *after =
            nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
         if (!ok || breaks != 1 || !splittable_outer(loop) ||
             (nir_block_first_instr(after) &&
              nir_block_first_instr(after)->type == nir_instr_type_phi) ||
             !header_exit_computable_at_entry(top_exit->condition.ssa, loop,
                                               header, 8))
            continue;

         struct util_dynarray body_loads;
         util_dynarray_init(&body_loads, NULL);
         unsigned total_dwords = hoisted_dwords;
         for (nir_cf_node *cf = nir_cf_node_next(&top_exit->cf_node); cf;
              cf = nir_cf_node_next(cf)) {
            if (cf->type != nir_cf_node_block)
               continue;
            nir_block *blk = nir_cf_node_as_block(cf);
            nir_foreach_instr(instr, blk) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if ((intr->intrinsic != nir_intrinsic_load_global &&
                    intr->intrinsic != nir_intrinsic_load_global_constant) ||
                   (nir_intrinsic_access(intr) &
                    (ACCESS_VOLATILE | ACCESS_COHERENT |
                     ACCESS_BUFFER_BINDING_AGX)))
                  continue;
               if (!is_loop_invariant_pure(intr->src[0].ssa, loop, 10))
                  continue;
               const unsigned dwords =
                  intr->def.num_components * DIV_ROUND_UP(intr->def.bit_size, 32);
               if (total_dwords + dwords > 16)
                  continue;
               total_dwords += dwords;
               util_dynarray_append(&body_loads, intr);
            }
         }

         if (util_dynarray_num_elements(&body_loads, nir_intrinsic_instr *) == 0) {
            util_dynarray_fini(&body_loads);
            continue;
         }

         /* Verify every SSA def inside `loop` used outside `loop` is a header
          * phi with a preheader source, saving its preheader init value before
          * CFG rewrite so merge phis after `guard` preserve exact SSA values
          * on both taken and skipped paths. */
         struct live_phi {
            nir_phi_instr *phi;
            nir_def *init_val;
         };
         bool live_out_ok = true;
         struct util_dynarray live_phis;
         util_dynarray_init(&live_phis, NULL);
         nir_foreach_block_in_cf_node(block, &loop->cf_node) {
            nir_foreach_instr(instr, block) {
               nir_def *def = nir_instr_def(instr);
               if (!def)
                  continue;
               bool used_outside = false;
               nir_foreach_use_including_if(src, def) {
                  if (!block_in_loop(use_block(src), loop)) {
                     used_outside = true;
                     break;
                  }
               }
               if (!used_outside)
                  continue;
               if (block != header || instr->type != nir_instr_type_phi) {
                  live_out_ok = false;
                  break;
               }
               nir_phi_instr *phi = nir_instr_as_phi(instr);
               nir_phi_src *pre_src = nir_phi_get_src_from_block(phi, pre);
               if (!pre_src) {
                  live_out_ok = false;
                  break;
               }
               struct live_phi entry = { .phi = phi, .init_val = pre_src->src.ssa };
               util_dynarray_append(&live_phis, entry);
            }
            if (!live_out_ok)
               break;
         }
         if (!live_out_ok) {
            util_dynarray_fini(&live_phis);
            util_dynarray_fini(&body_loads);
            continue;
         }

         /* Rematerialize the pure invariant addresses in `pre`, so each load
          * address stays a preheader SSA def dominating guard_pre. */
         nir_builder b = nir_builder_at(nir_after_block_before_jump(pre));
         struct hash_table *lremap = _mesa_pointer_hash_table_create(NULL);
         util_dynarray_foreach(&body_loads, nir_intrinsic_instr *, ldp) {
            nir_intrinsic_instr *intr = *ldp;
            nir_def *addr = remat_outer(&b, intr->src[0].ssa, loop, lremap);
            nir_src_rewrite(&intr->src[0], addr);
         }
         _mesa_hash_table_destroy(lremap, NULL);

         struct hash_table *eremap = _mesa_pointer_hash_table_create(NULL);
         nir_def *init_cond = emit_header_exit_at_entry(
            &b, top_exit->condition.ssa, loop, header, pre, eremap);
         _mesa_hash_table_destroy(eremap, NULL);
         nir_def *enter = break_in_then ? nir_inot(&b, init_cond) : init_cond;

         b.cursor = nir_before_cf_node(&loop->cf_node);
         nir_if *guard = nir_push_if(&b, enter);
         nir_pop_if(&b, guard);

         nir_cf_list moved;
         nir_cf_extract(&moved, nir_before_cf_node(&loop->cf_node),
                        nir_after_cf_node(&loop->cf_node));
         nir_cf_reinsert(&moved, nir_before_cf_list(&guard->then_list));

         /* Stitch merge phis after `guard` for every loop header phi used
          * outside the loop: then-arm carries the loop exit value, else-arm
          * carries the preheader initial value. */
         nir_block *then_end = nir_if_last_then_block(guard);
         nir_block *else_end = nir_if_last_else_block(guard);
         nir_block *after_guard =
            nir_cf_node_as_block(nir_cf_node_next(&guard->cf_node));
         util_dynarray_foreach(&live_phis, struct live_phi, lp_phi) {
            nir_phi_instr *phi = lp_phi->phi;
            nir_phi_instr *mphi = nir_phi_instr_create(shader);
            nir_def_init(&mphi->instr, &mphi->def, phi->def.num_components,
                         phi->def.bit_size);
            nir_phi_instr_add_src(mphi, then_end, &phi->def);
            nir_phi_instr_add_src(mphi, else_end, lp_phi->init_val);
            nir_instr_insert(nir_before_block(after_guard), &mphi->instr);

            nir_foreach_use_including_if_safe(src, &phi->def) {
               if (!nir_src_is_if(src) && nir_src_use_instr(src) == &mphi->instr)
                  continue;
               if (!block_in_loop(use_block(src), loop))
                  nir_src_rewrite(src, &mphi->def);
            }
         }
         util_dynarray_fini(&live_phis);

         /* Move the invariant body loads right before the loop inside
          * guard->then_list (guaranteed to execute at least once). */
         nir_block *guard_pre = nir_if_first_then_block(guard);
         util_dynarray_foreach(&body_loads, nir_intrinsic_instr *, ldp) {
            nir_intrinsic_instr *intr = *ldp;
            nir_instr_remove(&intr->instr);
            nir_instr_insert(nir_after_block_before_jump(guard_pre),
                             &intr->instr);
         }
         util_dynarray_fini(&body_loads);
         progress = true;
         /* Control flow changed; finish this impl cleanly. */
         break;
      }

      util_dynarray_fini(&loops);
      if (progress) {
         nir_progress(true, impl, nir_metadata_none);
         any = true;
      } else {
         nir_no_progress(impl);
      }
   }

   return any;
}

static unsigned
count_cf_list_instrs(struct exec_list *list)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_phi &&
                instr->type != nir_instr_type_jump)
               ++n;
         }
      }
   }
   return n;
}

static bool
loop_or_ancestor_in_set(nir_loop *loop, struct set *s)
{
   for (nir_cf_node *n = &loop->cf_node; n; n = n->parent) {
      if (n->type == nir_cf_node_loop && _mesa_set_search(s, n))
         return true;
   }
   return false;
}

static bool
loop_can_unswitch_uniform(nir_loop *loop)
{
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   return !loop->partially_unrolled &&
          !nir_loop_has_continue_construct(loop) &&
          count_instrs(loop) <= 256 &&
          splittable_outer(loop) &&
          live_outs_are_alu(loop) &&
          !(nir_block_first_instr(after) &&
            nir_block_first_instr(after)->type == nir_instr_type_phi);
}

/*
 * Unswitch a loop-invariant uniform `if (cond)` whose both arms perform
 * substantial work out of its enclosing loop(s) (for example a mode flag
 * selecting between bilinear resampling and direct copy inside a 2D image
 * loop). Substituting `cond = true` / `cond = false` in the two copies lets
 * dead-CF elimination merge the surviving arm into the loop body, exposing
 * loop-invariant address/coordinate arithmetic to LICM and store-to-load
 * forwarding.
 */
static bool
unswitch_uniform_ifs(nir_function_impl *impl)
{
   if (impl->function->shader->info.maximally_reconverges)
      return false;

   struct util_dynarray loops;
   util_dynarray_init(&loops, NULL);
   collect_loops(&impl->body, &loops);
   if (!util_dynarray_num_elements(&loops, nir_loop *)) {
      util_dynarray_fini(&loops);
      return false;
   }

   nir_divergence_analysis(impl->function->shader);

   struct set *done = _mesa_pointer_set_create(NULL);
   unsigned unswitches = 0;
   bool progress = false;

   util_dynarray_foreach(&loops, nir_loop *, lp) {
      if (unswitches >= 2)
         break;
      nir_loop *inner = *lp;
      if (loop_or_ancestor_in_set(inner, done))
         continue;

      nir_def *cond = NULL;
      foreach_list_typed(nir_cf_node, node, node, &inner->body) {
         if (node->type != nir_cf_node_if)
            continue;
         nir_if *cand = nir_cf_node_as_if(node);
         if (cf_node_has_jump(&cand->cf_node))
            continue;
         if (count_cf_list_instrs(&cand->then_list) < 4 ||
             count_cf_list_instrs(&cand->else_list) < 4)
            continue;
         nir_def *c = cand->condition.ssa;
         if (c->divergent || nir_def_is_const(c) ||
             !is_loop_invariant_pure(c, inner, 8))
            continue;
         cond = c;
         break;
      }
      if (!cond)
         continue;

      /* Hoist out of the outermost enclosing loop where `cond` is still
       * invariant and splitting stays within the instruction budget. */
      nir_loop *target = inner;
      while (true) {
         nir_loop *parent = enclosing_loop(&target->cf_node);
         if (!parent || !is_loop_invariant_pure(cond, parent, 8) ||
             !loop_can_unswitch_uniform(parent))
            break;
         target = parent;
      }
      if (loop_or_ancestor_in_set(target, done) ||
          !loop_can_unswitch_uniform(target))
         continue;

      nir_builder b = nir_builder_at(nir_before_cf_node(&target->cf_node));
      if (!def_outside_loop(cond, target)) {
         struct hash_table *hmap = _mesa_pointer_hash_table_create(NULL);
         nir_def *hoisted = remat_outer(&b, cond, target, hmap);
         _mesa_hash_table_destroy(hmap, NULL);
         nir_foreach_use_including_if_safe(src, cond) {
            if (block_in_loop(use_block(src), target))
               nir_src_rewrite(src, hoisted);
         }
         cond = hoisted;
      }

      nir_foreach_block_in_cf_node(block, &target->cf_node)
         nir_lower_phis_to_regs_block(block, true);
      nir_foreach_block_in_cf_node(block, &target->cf_node)
         nir_lower_ssa_defs_to_regs_block(block);

      b.cursor = nir_before_cf_node(&target->cf_node);
      nir_if *nif = nir_push_if(&b, cond);
      nir_pop_if(&b, nif);

      nir_cf_list list, copy;
      nir_cf_extract(&list, nir_before_cf_node(&target->cf_node),
                     nir_after_cf_node(&target->cf_node));
      struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list_clone(&copy, &list, &nif->cf_node, cmap);
      _mesa_hash_table_destroy(cmap, NULL);
      nir_cf_reinsert(&list, nir_before_cf_list(&nif->then_list));
      nir_cf_reinsert(&copy, nir_before_cf_list(&nif->else_list));

      b.cursor = nir_before_cf_list(&nif->then_list);
      nir_def *imm_t = nir_imm_true(&b);
      nir_foreach_use_including_if_safe(src, cond) {
         if (block_in_cf_list(use_block(src), &nif->then_list))
            nir_src_rewrite(src, imm_t);
      }

      b.cursor = nir_before_cf_list(&nif->else_list);
      nir_def *imm_f = nir_imm_false(&b);
      nir_foreach_use_including_if_safe(src, cond) {
         if (block_in_cf_list(use_block(src), &nif->else_list))
            nir_src_rewrite(src, imm_f);
      }

      _mesa_set_add(done, target);
      unswitches++;
      progress = true;
   }

   _mesa_set_destroy(done, NULL);
   util_dynarray_fini(&loops);
   return progress;
}

bool
agx_nir_unroll_runtime_loops(nir_shader *shader, unsigned factor,
                             unsigned max_instrs, unsigned *widest)
{
   bool any = false;
   assert(factor >= 2 && factor <= 16);

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index);
      nir_index_ssa_defs(impl);

      struct set *skip_runtime_unroll = _mesa_pointer_set_create(NULL);
      bool unswitched = false;
      if (unswitch_uniform_ifs(impl)) {
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
         unswitched = true;
         any = true;
      }
      if (specialize_exact_nested(impl, factor, max_instrs,
                                  skip_runtime_unroll, widest)) {
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
         unswitched = true;
         any = true;
      }
      struct util_dynarray small_peels, outer3_peels;
      util_dynarray_init(&small_peels, NULL);
      util_dynarray_init(&outer3_peels, NULL);
      if (unswitch_nested(impl, factor, max_instrs, skip_runtime_unroll,
                          &small_peels, &outer3_peels)) {
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
         bool peeled = false;
         util_dynarray_foreach(&small_peels, struct small_peel_rec, rec) {
            if (peel_straight_prefix(impl, rec->loop, rec->count,
                                     rec->keep_tail))
               peeled = true;
            else
               rec->loop->partially_unrolled = true;
         }
         if (peeled) {
            nir_progress(true, impl, nir_metadata_none);
            nir_lower_reg_intrinsics_to_ssa_impl(impl);
            nir_metadata_require(impl, nir_metadata_block_index);
            nir_index_ssa_defs(impl);
            bool outer_peeled = false;
            util_dynarray_foreach(&outer3_peels, struct small_peel_rec, rec) {
               if (peel_straight_prefix(impl, rec->loop, rec->count,
                                        rec->keep_tail))
                  outer_peeled = true;
            }
            if (outer_peeled) {
               nir_progress(true, impl, nir_metadata_none);
               nir_lower_reg_intrinsics_to_ssa_impl(impl);
               nir_metadata_require(impl, nir_metadata_block_index);
               nir_index_ssa_defs(impl);
            }
         }
         unswitched = true;
         any = true;
      }
      util_dynarray_fini(&small_peels);
      util_dynarray_fini(&outer3_peels);

      /* One loop at a time, each analyzed on the shader the unrolls before it
       * left, back in SSA form: an unroll turns its loop's header phis into
       * registers, and a later loop's bound can be one of those phis (such as
       * a count the loop before leaves). */
      unsigned n = 0;
      while (n < 64) {
         struct util_dynarray loops;
         util_dynarray_init(&loops, NULL);
         collect_loops(&impl->body, &loops);

         struct counted_loop cl;
         unsigned f = 0;
         util_dynarray_foreach(&loops, nir_loop *, lp) {
            if (_mesa_set_search(skip_runtime_unroll, *lp) ||
                !analyze_loop_opts(*lp, &cl, true, false))
               continue;
            if (cl.step == 0 && !loop_is_readonly_with_loads(impl, *lp))
               continue;
            if ((f = choose_factor(impl, *lp, &cl, factor, max_instrs)))
               break;
         }
         util_dynarray_fini(&loops);
         if (!f)
            break;

         /* The bound (and runtime step, if any) as seen from the preheader. */
         nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&cl.loop->cf_node));
         nir_builder b = nir_builder_at(nir_after_block_before_jump(pre));
         nir_def *bound;
         if (nir_scalar_is_const(cl.bound)) {
            bound = nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(cl.bound));
         } else {
            /* Defined outside the loop and read in it, so it dominates the
             * preheader's end, or computed alike by every iteration: then
             * computed once there. */
            struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
            nir_def *def = rematerialize(&b, cl.bound.def, cl.loop, remap);
            _mesa_hash_table_destroy(remap, NULL);
            bound = nir_channel(&b, def, cl.bound.comp);
         }
         nir_def *step_val = NULL;
         if (cl.step == 0) {
            struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
            nir_def *def = rematerialize(&b, cl.step_def.def, cl.loop, remap);
            _mesa_hash_table_destroy(remap, NULL);
            step_val = nir_channel(&b, def, cl.step_def.comp);
         }
         if (widest)
            *widest = MAX2(*widest, f);
         unroll_loop(impl, &cl, bound, step_val, f);

         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
         n++;
      }

      _mesa_set_destroy(skip_runtime_unroll, NULL);

      if (n) {
         any = true;
      } else if (unswitched) {
         nir_progress(true, impl, nir_metadata_none);
      } else {
         nir_no_progress(impl);
      }
   }
   return any;
}
