/*
 * Copyright © 2014 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

/* Dead Code Elimination
 *
 * It eliminates those instructions whose results are unused and those that
 * have no effect on instructions with side effects (like stores).
 * To be precise, instructions are not dead if their result is needed by
 * non-removable instructions.
 *
 * The algorithm walks the shader backwards and:
 * - marks sources of non-removable instructions as live
 * - if an instruction is live, mark all its sources as live as well
 * - if a loop header phi source is marked as live and the source is not in
 *   the block preceding the loop, the loop is walked backwards again
 *   (beucase the phi source is likely after the phi)
 *
 * Why looking only at the number of uses of each instruction doesn't work:
 * There could be a loop in the def-use graph where every instructions has
 * non-zero uses, yet all of them are dead. Example:
 *
 * loop {
 *    %1 = phi %0, %2
 *
 *    loop {
 *       %2 = phi %1, %0
 *    }
 * }
 *
 * In this case, both phis have non-zero uses because they use each other,
 * but they have no effect on the shader and can be removed.
 */

#include "nir.h"

static inline bool
is_def_live(const nir_def *def, const BITSET_WORD *defs_live)
{
   return BITSET_TEST(defs_live, def->index);
}

static inline void
mark_src_live_fast(const nir_src *src, BITSET_WORD *defs_live)
{
   BITSET_SET(defs_live, src->ssa->index);
}

static inline bool
mark_src_live(const nir_src *src, BITSET_WORD *defs_live)
{
   unsigned idx = src->ssa->index;
   if (!BITSET_TEST(defs_live, idx)) {
      BITSET_SET(defs_live, idx);
      return true;
   }
   return false;
}

struct loop_state {
   bool header_phis_changed;
   nir_block *preheader;
};

static ALWAYS_INLINE bool
is_live_and_mark_sources(BITSET_WORD *defs_live, nir_instr *instr,
                         struct loop_state *loop, bool *phis_changed)
{
   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (!is_def_live(&alu->def, defs_live))
         return false;
      const unsigned num_inputs = nir_op_infos[alu->op].num_inputs;
      for (unsigned i = 0; i < num_inputs; ++i)
         mark_src_live_fast(&alu->src[i].src, defs_live);
      return true;
   }
   case nir_instr_type_load_const: {
      nir_load_const_instr *lc = nir_instr_as_load_const(instr);
      return is_def_live(&lc->def, defs_live);
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intrin->intrinsic];
      bool live = !(info->flags & NIR_INTRINSIC_CAN_ELIMINATE) ||
                  (info->has_dest && is_def_live(&intrin->def, defs_live));
      if (live) {
         const unsigned num_srcs = info->num_srcs;
         for (unsigned i = 0; i < num_srcs; ++i)
            mark_src_live_fast(&intrin->src[i], defs_live);
      }
      return live;
   }
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (!is_def_live(&tex->def, defs_live))
         return false;
      for (unsigned i = 0; i < tex->num_srcs; ++i)
         mark_src_live_fast(&tex->src[i].src, defs_live);
      return true;
   }
   case nir_instr_type_deref: {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      if (!is_def_live(&deref->def, defs_live))
         return false;
      if (deref->deref_type != nir_deref_type_var)
         mark_src_live_fast(&deref->parent, defs_live);
      if (deref->deref_type == nir_deref_type_array ||
          deref->deref_type == nir_deref_type_ptr_as_array)
         mark_src_live_fast(&deref->arr.index, defs_live);
      return true;
   }
   case nir_instr_type_phi: {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      if (!is_def_live(&phi->def, defs_live))
         return false;
      if (loop) {
         nir_foreach_phi_src(src, phi) {
            *phis_changed |= mark_src_live(&src->src, defs_live) &&
                             src->pred != loop->preheader;
         }
      } else {
         nir_foreach_phi_src(src, phi)
            mark_src_live_fast(&src->src, defs_live);
      }
      return true;
   }
   case nir_instr_type_undef: {
      nir_undef_instr *undef = nir_instr_as_undef(instr);
      return is_def_live(&undef->def, defs_live);
   }
   case nir_instr_type_jump: {
      nir_jump_instr *jump = nir_instr_as_jump(instr);
      if (jump->type == nir_jump_goto_if)
         mark_src_live_fast(&jump->condition, defs_live);
      return true;
   }
   case nir_instr_type_call: {
      nir_call_instr *call = nir_instr_as_call(instr);
      if (call->indirect_callee.ssa)
         mark_src_live_fast(&call->indirect_callee, defs_live);
      for (unsigned i = 0; i < call->num_params; ++i)
         mark_src_live_fast(&call->params[i], defs_live);
      return true;
   }
   case nir_instr_type_cmat_call: {
      nir_cmat_call_instr *call = nir_instr_as_cmat_call(instr);
      for (unsigned i = 0; i < call->num_params; ++i)
         mark_src_live_fast(&call->params[i], defs_live);
      return true;
   }
   default:
      UNREACHABLE("unexpected instr type");
   }
}

static void
remove_instr(nir_instr *instr, struct exec_list *dead_instrs)
{
#ifndef NDEBUG
   /* Fail an assertion if an input load is dead. This is a debug option. */
   if (instr->type == nir_instr_type_intrinsic) {
      nir_shader *nir =
         instr->block->impl->function->shader;

      if (nir->info.assert_inputs_not_dead)
         assert(!nir_is_input_load(nir_instr_as_intrinsic(instr)));
   }
#endif

   nir_instr_remove(instr);
   exec_list_push_tail(dead_instrs, &instr->node);
}

static bool
dce_block_no_loop(nir_block *block, BITSET_WORD *defs_live,
                  struct exec_list *dead_instrs)
{
   bool progress = false;
   nir_foreach_instr_reverse_safe(instr, block) {
      if (!is_live_and_mark_sources(defs_live, instr, NULL, NULL)) {
         remove_instr(instr, dead_instrs);
         progress = true;
      }
   }
   return progress;
}

static bool
dce_block(nir_block *block, BITSET_WORD *defs_live, struct loop_state *loop,
          struct exec_list *dead_instrs)
{
   if (!loop->preheader)
      return dce_block_no_loop(block, defs_live, dead_instrs);

   bool phis_changed = false;
   nir_foreach_instr_reverse_safe(instr, block) {
      instr->pass_flags = is_live_and_mark_sources(defs_live, instr, loop,
                                                   &phis_changed);
   }

   /* Because blocks are visited in reverse and this stomps header_phis_changed,
    * we don't have to check whether the current block is a loop header before
    * setting header_phis_changed.
    */
   loop->header_phis_changed = phis_changed;

   return false;
}

static bool
dce_cf_list(struct exec_list *cf_list, BITSET_WORD *defs_live,
            struct loop_state *parent_loop, struct exec_list *dead_instrs)
{
   bool progress = false;
   foreach_list_typed_reverse(nir_cf_node, cf_node, node, cf_list) {
      switch (cf_node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(cf_node);
         progress |= dce_block(block, defs_live, parent_loop, dead_instrs);
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(cf_node);
         progress |= dce_cf_list(&nif->else_list, defs_live, parent_loop, dead_instrs);
         progress |= dce_cf_list(&nif->then_list, defs_live, parent_loop, dead_instrs);
         mark_src_live_fast(&nif->condition, defs_live);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(cf_node);
         assert(!nir_loop_has_continue_construct(loop));

         struct loop_state inner_state;
         inner_state.preheader = nir_cf_node_as_block(nir_cf_node_prev(cf_node));
         inner_state.header_phis_changed = false;

         /* Fast path if the loop has no continues: we can remove instructions
          * as we mark the others live.
          */
         if (!nir_loop_has_back_edge(loop)) {
            progress |= dce_cf_list(&loop->body, defs_live, parent_loop, dead_instrs);
            break;
         }

         /* Mark instructions as live until there is no more progress. */
         do {
            /* dce_cf_list() resets inner_state.header_phis_changed itself, so
             * it doesn't have to be done here.
             */
            dce_cf_list(&loop->body, defs_live, &inner_state, dead_instrs);
         } while (inner_state.header_phis_changed);

         /* We don't know how many times mark_cf_list() will repeat, so
          * remove instructions separately.
          *
          * By checking parent_loop->preheader, we ensure that we only do this
          * walk for the outer-most loops so it only happens once.
          */
         if (!parent_loop->preheader) {
            nir_foreach_block_in_cf_node(block, cf_node) {
               nir_foreach_instr_safe(instr, block) {
                  if (!instr->pass_flags) {
                     remove_instr(instr, dead_instrs);
                     progress = true;
                  }
               }
            }
         }
         break;
      }
      case nir_cf_node_function:
         UNREACHABLE("Invalid cf type");
      }
   }

   return progress;
}

bool
nir_opt_dce_impl(nir_function_impl *impl)
{
   assert(impl->structured);

   BITSET_DECLARE(stack_live, 4096);
   BITSET_WORD *defs_live;
   if (impl->ssa_alloc <= 4096) {
      memset(stack_live, 0, BITSET_BYTES(impl->ssa_alloc));
      defs_live = stack_live;
   } else {
      defs_live = BITSET_RZALLOC(NULL, impl->ssa_alloc);
   }

   struct exec_list dead_instrs;
   exec_list_make_empty(&dead_instrs);

   struct loop_state loop;
   loop.preheader = NULL;
   bool progress = dce_cf_list(&impl->body, defs_live, &loop, &dead_instrs);

   if (defs_live != stack_live)
      ralloc_free(defs_live);

   nir_instr_free_list(&dead_instrs);

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
nir_opt_dce(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      if (nir_opt_dce_impl(impl))
         progress = true;
   }

   return progress;
}
