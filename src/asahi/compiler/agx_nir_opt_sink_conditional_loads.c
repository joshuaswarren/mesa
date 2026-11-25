/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Sink non-volatile, non-atomic, non-coherent buffer loads into a strictly
 * dominated conditional branch when no intervening instruction or control-flow
 * node may write memory or synchronize.
 *
 * nir_opt_sink only moves buffer loads marked ACCESS_CAN_REORDER. Shaders that
 * read input SSBOs without readonly/restrict qualifiers and write output SSBOs
 * later (such as outside a loop) leave those loads without ACCESS_CAN_REORDER
 * even when no store, atomic, barrier, or call exists between the load and the
 * branch that exclusively consumes it. Sinking such loads into the consuming
 * branch avoids issuing the load on skipped SIMD groups and keeps
 * nir_opt_peephole_select from flattening the guarded body.
 */

#include "agx_nir.h"
#include "compiler/nir/nir.h"

#define MAX_CF_DEPTH 32

static bool
is_sinkable_buffer_load(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_deref: {
      nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
      if (!deref ||
          !nir_deref_mode_is_one_of(deref, nir_var_mem_ssbo |
                                            nir_var_mem_global))
         return false;
      break;
   }
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global_bounded:
   case nir_intrinsic_load_agx:
      break;
   default:
      return false;
   }

   const enum gl_access_qualifier access = nir_intrinsic_access(intr);
   if (access & (ACCESS_VOLATILE | ACCESS_ATOMIC | ACCESS_COHERENT))
      return false;

   return true;
}

static bool
instr_may_interfere(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_call ||
       instr->type == nir_instr_type_cmat_call)
      return true;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (!(nir_intrinsic_infos[intr->intrinsic].flags &
         NIR_INTRINSIC_CAN_ELIMINATE))
      return true;

   if (nir_intrinsic_has_access(intr) &&
       (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC)))
      return true;

   return false;
}

static bool
cf_node_may_interfere(nir_cf_node *node)
{
   nir_foreach_block_in_cf_node(block, node) {
      nir_foreach_instr(instr, block) {
         if (instr_may_interfere(instr))
            return true;
      }
   }
   return false;
}

static nir_loop *
innermost_loop(nir_cf_node *node)
{
   for (; node != NULL; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(node);
   }
   return NULL;
}

static bool
path_is_safe(nir_instr *load, nir_block *def_block, nir_block *target,
             bool can_reorder)
{
   nir_cf_node *parent = def_block->cf_node.parent;
   nir_cf_node *chain[MAX_CF_DEPTH];
   unsigned depth = 0;

   for (nir_cf_node *node = &target->cf_node; node != parent;
        node = node->parent) {
      if (!node || depth == MAX_CF_DEPTH)
         return false;
      /* Require strictly nested ifs (never cross a loop boundary). */
      if (depth > 0 && node->type != nir_cf_node_if)
         return false;
      chain[depth++] = node;
   }

   /* Must descend into at least one nir_if strictly below def_block's parent. */
   if (depth < 2)
      return false;

   if (can_reorder)
      return true;

   /* 1. Remaining instructions in def_block after load. */
   for (struct exec_node *n = load->node.next; !exec_node_is_tail_sentinel(n);
        n = n->next) {
      if (instr_may_interfere(exec_node_data(nir_instr, n, node)))
         return false;
   }

   /* 2. Sibling CF nodes in parent strictly between def_block and the
    * outermost enclosing if. */
   nir_cf_node *outer_if = chain[depth - 1];
   bool reached = false;
   for (nir_cf_node *s = nir_cf_node_next(&def_block->cf_node); s != NULL;
        s = nir_cf_node_next(s)) {
      if (s == outer_if) {
         reached = true;
         break;
      }
      if (cf_node_may_interfere(s))
         return false;
   }
   if (!reached)
      return false;

   /* 3. Within each enclosing if arm, all CF nodes preceding the child
    * containing target. */
   for (unsigned d = depth - 1; d-- > 0;) {
      for (nir_cf_node *s = nir_cf_node_prev(chain[d]); s != NULL;
           s = nir_cf_node_prev(s)) {
         if (cf_node_may_interfere(s))
            return false;
      }
   }

   return true;
}

static nir_block *
find_sink_target(nir_instr *load)
{
   nir_def *def = nir_instr_def(load);
   if (!def)
      return NULL;

   nir_block *lca = NULL;
   nir_foreach_use_including_if(use, def) {
      lca = nir_dominance_lca(lca, nir_src_get_block(use));
   }

   nir_block *def_block = load->block;
   if (!lca || lca == def_block)
      return NULL;

   /* Never sink into or out of a loop. */
   nir_loop *def_loop = innermost_loop(&def_block->cf_node);
   while (lca && lca != def_block && innermost_loop(&lca->cf_node) != def_loop)
      lca = lca->imm_dom;

   if (!lca || lca == def_block)
      return NULL;

   const bool can_reorder =
      nir_intrinsic_can_reorder(nir_instr_as_intrinsic(load));

   for (nir_block *target = lca; target && target != def_block;
        target = target->imm_dom) {
      if (target->cf_node.parent == def_block->cf_node.parent)
         break;
      if (path_is_safe(load, def_block, target, can_reorder))
         return target;
   }

   return NULL;
}

bool
agx_nir_opt_sink_conditional_loads(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index |
                                 nir_metadata_dominance |
                                 nir_metadata_dominance_lca);

      bool impl_progress = false;
      nir_foreach_block_reverse(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block) {
            if (!is_sinkable_buffer_load(instr))
               continue;

            nir_block *target = find_sink_target(instr);
            if (!target)
               continue;

            nir_instr_remove(instr);
            nir_instr_insert(nir_after_phis(target), instr);
            impl_progress = true;
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}
