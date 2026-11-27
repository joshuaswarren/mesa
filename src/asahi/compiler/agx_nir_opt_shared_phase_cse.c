/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_instr_set.h"
#include "agx_nir.h"

/* The boundary is a workgroup-wide rendezvous and shared-memory fence.
 * Divergence is an additional refusal; correctness uses the barrier's
 * workgroup execution scope on a valid shader. */
static bool
shared_phase_boundary(nir_block *block, bool entry)
{
   if (!block || block->divergent)
      return false;

   nir_instr *instr = entry ? nir_block_last_instr(block)
                            : nir_block_first_instr(block);
   if (!instr || instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *barrier = nir_instr_as_intrinsic(instr);
   return barrier->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(barrier) == SCOPE_WORKGROUP &&
          nir_intrinsic_memory_scope(barrier) == SCOPE_DEVICE &&
          nir_intrinsic_memory_modes(barrier) == nir_var_mem_shared &&
          nir_intrinsic_memory_semantics(barrier) == NIR_MEMORY_ACQ_REL;
}

static bool
shared_phase_load(const nir_intrinsic_instr *intrin, UNUSED const void *data)
{
   return intrin->intrinsic == nir_intrinsic_load_shared &&
          nir_intrinsic_access(intrin) == 0 && intrin->def.bit_size == 32 &&
          intrin->def.num_components == 1;
}

/* Check the entire structured phase for writes, not just the candidate
 * block. Two workgroup rendezvous exclude stores from peers outside it. */
static bool
shared_phase_readonly(nir_loop *loop)
{
   if (!shared_phase_boundary(nir_loop_predecessor_block(loop), true) ||
       !shared_phase_boundary(nir_loop_successor_block(loop), false))
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_phi:
         case nir_instr_type_load_const:
         case nir_instr_type_alu:
            break;
         case nir_instr_type_intrinsic:
            if (!shared_phase_load(nir_instr_as_intrinsic(instr), NULL))
               return false;
            break;
         case nir_instr_type_jump: {
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            if (jump->type != nir_jump_break && jump->type != nir_jump_continue)
               return false;
            break;
         }
         default:
            return false;
         }
      }
   }
   return true;
}

/* Within a loop that runs between two uniform workgroup acquire-release
 * barriers on shared memory and contains nothing but ALU, constants, phis,
 * break/continue and plain scalar 32-bit shared loads, identical shared
 * loads of one block read the same value: no invocation of the workgroup
 * can store to shared memory between the two rendezvous. Keep the first of
 * each in place and use it for the others. Generic NIR on every chip. */
bool
agx_nir_opt_shared_phase_cse(nir_shader *nir)
{
   bool any = false;

   nir_divergence_analysis(nir);
   nir_foreach_function_impl(impl, nir) {
      if (impl->function->is_preamble) {
         nir_no_progress(impl);
         continue;
      }

      bool progress = false;
      nir_foreach_block(pre, impl) {
         nir_loop *loop = nir_block_get_following_loop(pre);
         if (!loop || !shared_phase_readonly(loop))
            continue;

         nir_foreach_block_in_cf_node(block, &loop->cf_node) {
            struct set loads = {0};
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               if (!loads.table)
                  nir_instr_set_init(&loads, NULL);

               /* Keep the first read in place. Do not mark it globally
                * reorderable or reuse it across a dynamic iteration. */
               if (nir_instr_set_add_or_rewrite(&loads, instr,
                                                shared_phase_load,
                                                NULL, NULL)) {
                  nir_instr_remove(instr);
                  progress = true;
               }
            }
            if (loads.table)
               nir_instr_set_fini(&loads);
         }
      }
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }

   return any;
}
