/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"

extern "C" {
#include "agx_nir.h"
}

static nir_builder
compute_shader(const char *name, unsigned wx, unsigned wy = 1, unsigned wz = 1)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = wx;
   b.shader->info.workgroup_size[1] = wy;
   b.shader->info.workgroup_size[2] = wz;
   return b;
}

static nir_intrinsic_instr *
emit_barrier(nir_builder *b, nir_variable_mode modes, mesa_scope exec,
             mesa_scope mem)
{
   nir_barrier(b, .execution_scope = exec, .memory_scope = mem,
               .memory_semantics = mem == SCOPE_NONE ? (nir_memory_semantics)0
                                                     : NIR_MEMORY_ACQ_REL,
               .memory_modes = modes);
   return nir_instr_as_intrinsic(
      nir_block_last_instr(nir_cursor_current_block(b->cursor)));
}

static void
collect_barrier_exec_scopes(nir_shader *s, mesa_scope *out, unsigned n)
{
   unsigned i = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_barrier &&
                i < n)
               out[i++] =
                  nir_intrinsic_execution_scope(nir_instr_as_intrinsic(instr));
         }
      }
   }
   ASSERT_EQ(i, n);
}

TEST(NarrowBarriers, SingleSubgroupWorkgroupAndPreservesGlobal)
{
   for (unsigned width : {32u, 8u, 64u}) {
      nir_builder b = compute_shader("single-subgroup", width);
      nir_intrinsic_instr *shared =
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      nir_intrinsic_instr *exec =
         emit_barrier(&b, (nir_variable_mode)0, SCOPE_WORKGROUP, SCOPE_NONE);
      nir_intrinsic_instr *mem_only_shared =
         emit_barrier(&b, nir_var_mem_shared, SCOPE_NONE, SCOPE_WORKGROUP);
      nir_intrinsic_instr *device = emit_barrier(
         &b, (nir_variable_mode)(nir_var_mem_shared | nir_var_mem_global),
         SCOPE_WORKGROUP, SCOPE_WORKGROUP);

      const bool narrows = width <= 32;
      EXPECT_EQ(agx_nir_narrow_single_subgroup_barriers(b.shader), narrows);
      nir_validate_shader(b.shader, "single-subgroup");

      const mesa_scope expect = narrows ? SCOPE_SUBGROUP : SCOPE_WORKGROUP;
      EXPECT_EQ(nir_intrinsic_execution_scope(shared), expect);
      EXPECT_EQ(nir_intrinsic_memory_scope(shared), expect);
      EXPECT_EQ(nir_intrinsic_execution_scope(exec), expect);
      EXPECT_EQ(nir_intrinsic_memory_scope(exec), SCOPE_NONE);
      EXPECT_EQ(nir_intrinsic_execution_scope(mem_only_shared), SCOPE_NONE);
      EXPECT_EQ(nir_intrinsic_memory_scope(mem_only_shared), expect);
      EXPECT_EQ(nir_intrinsic_execution_scope(device), SCOPE_WORKGROUP);
      EXPECT_EQ(nir_intrinsic_memory_scope(device), SCOPE_WORKGROUP);
      ralloc_free(b.shader);
   }

   /* Variable workgroup sizes must not be narrowed. */
   nir_builder var = compute_shader("variable-wg", 16);
   var.shader->info.workgroup_size_variable = true;
   nir_intrinsic_instr *bar =
      emit_barrier(&var, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   EXPECT_FALSE(agx_nir_narrow_single_subgroup_barriers(var.shader));
   EXPECT_EQ(nir_intrinsic_execution_scope(bar), SCOPE_WORKGROUP);
   EXPECT_EQ(nir_intrinsic_memory_scope(bar), SCOPE_WORKGROUP);
   ralloc_free(var.shader);
}

static nir_shader *
build_reduction_shader(unsigned first_off)
{
   nir_builder b = compute_shader("reduction", 64);
   b.shader->info.shared_size = 256;
   nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));
   nir_store_shared(&b, t, addr, .align_mul = 4);
   emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   for (unsigned off : {first_off, 16u}) {
      nir_push_if(&b, nir_ult_imm(&b, t, off));
      nir_def *other = nir_load_shared(
         &b, 1, 32, nir_u2u16(&b, nir_ishl_imm(&b, nir_iadd_imm(&b, t, off), 2)),
         .align_mul = 4);
      nir_def *mine = nir_load_shared(&b, 1, 32, addr, .align_mul = 4);
      nir_store_shared(&b, nir_iadd(&b, mine, other), addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
      emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   }
   nir_push_if(&b, nir_ieq_imm(&b, t, 0));
   nir_def *sum =
      nir_load_shared(&b, 1, 32, nir_imm_intN_t(&b, 0, 16), .align_mul = 4);
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_pop_if(&b, NULL);
   return b.shader;
}

TEST(NarrowBarriers, Subgroup0ReductionTail)
{
   nir_shader *s = build_reduction_shader(32);
   EXPECT_TRUE(agx_nir_narrow_subgroup0_barriers(s));
   nir_validate_shader(s, "reduction-32");
   mesa_scope scopes[3];
   collect_barrier_exec_scopes(s, scopes, 3);
   EXPECT_EQ(scopes[0], SCOPE_WORKGROUP);
   EXPECT_EQ(scopes[1], SCOPE_SUBGROUP);
   EXPECT_EQ(scopes[2], SCOPE_SUBGROUP);
   ralloc_free(s);

   /* With first_off == 48, lanes 32..47 access shared memory in gap 1, so the
    * second barrier must stay SCOPE_WORKGROUP while the final barrier (between
    * t < 16 and t == 0) narrows safely. */
   s = build_reduction_shader(48);
   EXPECT_TRUE(agx_nir_narrow_subgroup0_barriers(s));
   nir_validate_shader(s, "reduction-48");
   collect_barrier_exec_scopes(s, scopes, 3);
   EXPECT_EQ(scopes[0], SCOPE_WORKGROUP);
   EXPECT_EQ(scopes[1], SCOPE_WORKGROUP);
   EXPECT_EQ(scopes[2], SCOPE_SUBGROUP);
   ralloc_free(s);
}

TEST(NarrowBarriers, Subgroup0MergedPhasePreservesCrossSubgroupHazard)
{
   /* Gap 0: all 64 lanes write shared memory.
    * Barrier 0: followed by empty gap 1 (no shared ops) -> narrows & merges
    *            gap 0's `outside = true` into the active phase.
    * Barrier 1: followed by gap 2 where all 64 lanes read shared memory ->
    *            MUST stay SCOPE_WORKGROUP because merged phase has outside! */
   nir_builder b = compute_shader("merged-phase-hazard", 64);
   b.shader->info.shared_size = 256;
   nir_def *t = nir_load_local_invocation_index(&b);
   nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));
   nir_store_shared(&b, t, addr, .align_mul = 4);
   emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   /* Empty gap 1 */
   emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   nir_def *other = nir_load_shared(
      &b, 1, 32, nir_u2u16(&b, nir_ishl_imm(&b, nir_isub(&b, nir_imm_int(&b, 63), t), 2)),
      .align_mul = 4);
   nir_store_global(&b, other, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_narrow_subgroup0_barriers(b.shader));
   nir_validate_shader(b.shader, "merged-phase-hazard");
   mesa_scope scopes[2];
   collect_barrier_exec_scopes(b.shader, scopes, 2);
   EXPECT_EQ(scopes[0], SCOPE_SUBGROUP);
   EXPECT_EQ(scopes[1], SCOPE_WORKGROUP);
   ralloc_free(b.shader);
}

TEST(NarrowBarriers, Subgroup0OtherMemoryOrdered)
{
   for (bool readonly : {false, true}) {
      nir_builder b = compute_shader("other-mem-order", 64);
      b.shader->info.shared_size = 4;
      nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      const gl_access_qualifier ro =
         (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_def *flag = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0),
                                      .access = readonly ? ro : (gl_access_qualifier)0,
                                      .align_mul = 4);
      emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      nir_push_if(&b, nir_ieq_imm(&b, t, 63));
      nir_store_global(&b, nir_imm_int(&b, 1), nir_imm_int64(&b, 0),
                       .align_mul = 4);
      nir_pop_if(&b, NULL);
      nir_store_global(&b, flag, nir_imm_int64(&b, 4), .align_mul = 4);

      agx_nir_narrow_subgroup0_barriers(b.shader);
      nir_validate_shader(b.shader, "other-mem-order");
      mesa_scope scope;
      collect_barrier_exec_scopes(b.shader, &scope, 1);
      EXPECT_EQ(scope, readonly ? SCOPE_SUBGROUP : SCOPE_WORKGROUP)
         << "readonly=" << readonly;
      ralloc_free(b.shader);
   }
}

TEST(NarrowBarriers, Subgroup0RefusesUnsafeShapes)
{
   /* 0: memory-only global fence next to shared barrier
    * 1: memory-only shared fence next to shared barrier
    * 2: subgroup barrier next to workgroup barrier
    * 3: barrier inside a loop
    * 4: barrier inside a divergent if
    * 5: 2D workgroup using local_invocation_id.x (not 1D local index!) */
   for (unsigned shape = 0; shape < 6; ++shape) {
      nir_builder b = compute_shader("refusals", shape == 5 ? 32 : 64,
                                     shape == 5 ? 2 : 1);
      b.shader->info.shared_size = 256;
      nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));

      if (shape == 0) {
         nir_store_shared(&b, t, addr, .align_mul = 4);
         emit_barrier(&b, nir_var_mem_ssbo, SCOPE_NONE, SCOPE_DEVICE);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      } else if (shape == 1) {
         nir_store_shared(&b, t, addr, .align_mul = 4);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_NONE, SCOPE_WORKGROUP);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      } else if (shape == 2) {
         nir_store_shared(&b, t, addr, .align_mul = 4);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_SUBGROUP, SCOPE_SUBGROUP);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      } else if (shape == 3) {
         nir_loop *loop = nir_push_loop(&b);
         nir_push_if(&b, nir_ieq_imm(&b, t, 0));
         nir_store_shared(&b, t, addr, .align_mul = 4);
         nir_pop_if(&b, NULL);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
         nir_jump(&b, nir_jump_break);
         nir_pop_loop(&b, loop);
      } else if (shape == 4) {
         nir_push_if(&b, nir_ult_imm(&b, t, 32));
         nir_store_shared(&b, t, addr, .align_mul = 4);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
         nir_pop_if(&b, NULL);
      } else {
         /* 32x2 workgroup: local_invocation_id.x < 32 holds for BOTH
          * subgroup 0 (y=0) and subgroup 1 (y=1), so must NOT narrow! */
         nir_push_if(&b, nir_ult_imm(&b, t, 32));
         nir_store_shared(&b, t, addr, .align_mul = 4);
         nir_pop_if(&b, NULL);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
         nir_push_if(&b, nir_ult_imm(&b, t, 32));
         nir_def *v = nir_load_shared(&b, 1, 32, addr, .align_mul = 4);
         nir_store_global(&b, v, nir_imm_int64(&b, 0), .align_mul = 4);
         nir_pop_if(&b, NULL);
      }

      EXPECT_FALSE(agx_nir_narrow_subgroup0_barriers(b.shader))
         << "shape=" << shape;
      nir_validate_shader(b.shader, "refusals");
      ralloc_free(b.shader);
   }
}
