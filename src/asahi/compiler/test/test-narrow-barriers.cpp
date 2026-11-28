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
   nir_intrinsic_instr *barrier = nir_barrier(b);
   nir_intrinsic_set_execution_scope(barrier, exec);
   nir_intrinsic_set_memory_scope(barrier, mem);
   nir_intrinsic_set_memory_semantics(barrier, mem == SCOPE_NONE
      ? (nir_memory_semantics)0 : NIR_MEMORY_ACQ_REL);
   nir_intrinsic_set_memory_modes(barrier, modes);
   return barrier;
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

static nir_def *
load_shared_word(nir_builder *b, nir_def *addr)
{
   nir_def *value = nir_load_shared(b, 1, 32, addr);
   nir_intrinsic_set_align(nir_def_as_intrinsic(value), 4, 0);
   return value;
}

static void
store_shared_word(nir_builder *b, nir_def *value, nir_def *addr)
{
   nir_intrinsic_set_align(nir_store_shared(b, value, addr), 4, 0);
}

static void
store_global_word(nir_builder *b, nir_def *value, nir_def *addr)
{
   nir_intrinsic_set_align(nir_store_global(b, value, addr), 4, 0);
}

static nir_shader *
build_reduction_shader(unsigned first_off)
{
   nir_builder b = compute_shader("reduction", 64);
   b.shader->info.shared_size = 256;
   nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));
   store_shared_word(&b, t, addr);
   emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   for (unsigned off : {first_off, 16u}) {
      nir_push_if(&b, nir_ult_imm(&b, t, off));
      nir_def *other = load_shared_word(&b, nir_u2u16(&b, nir_ishl_imm(&b, nir_iadd_imm(&b, t, off), 2)));
      nir_def *mine = load_shared_word(&b, addr);
      store_shared_word(&b, nir_iadd(&b, mine, other), addr);
      nir_pop_if(&b, NULL);
      emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   }
   nir_push_if(&b, nir_ieq_imm(&b, t, 0));
   nir_def *sum =
      load_shared_word(&b, nir_imm_intN_t(&b, 0, 16));
   store_global_word(&b, sum, nir_imm_int64(&b, 0));
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
   store_shared_word(&b, t, addr);
   emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   /* Empty gap 1 */
   emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
   nir_def *other = load_shared_word(&b, nir_u2u16(&b, nir_ishl_imm(&b, nir_isub(&b, nir_imm_int(&b, 63), t), 2)));
   store_global_word(&b, other, nir_imm_int64(&b, 0));

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
      nir_def *flag = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0));
      nir_intrinsic_set_align(nir_def_as_intrinsic(flag), 4, 0);
      if (readonly)
         nir_intrinsic_set_access(nir_def_as_intrinsic(flag), ro);
      emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      nir_push_if(&b, nir_ieq_imm(&b, t, 63));
      store_global_word(&b, nir_imm_int(&b, 1), nir_imm_int64(&b, 0));
      nir_pop_if(&b, NULL);
      store_global_word(&b, flag, nir_imm_int64(&b, 4));

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
         store_shared_word(&b, t, addr);
         emit_barrier(&b, nir_var_mem_ssbo, SCOPE_NONE, SCOPE_DEVICE);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      } else if (shape == 1) {
         store_shared_word(&b, t, addr);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_NONE, SCOPE_WORKGROUP);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      } else if (shape == 2) {
         store_shared_word(&b, t, addr);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_SUBGROUP, SCOPE_SUBGROUP);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      } else if (shape == 3) {
         nir_loop *loop = nir_push_loop(&b);
         nir_push_if(&b, nir_ieq_imm(&b, t, 0));
         store_shared_word(&b, t, addr);
         nir_pop_if(&b, NULL);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
         nir_jump(&b, nir_jump_break);
         nir_pop_loop(&b, loop);
      } else if (shape == 4) {
         nir_push_if(&b, nir_ult_imm(&b, t, 32));
         store_shared_word(&b, t, addr);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
         nir_pop_if(&b, NULL);
      } else {
         /* 32x2 workgroup: local_invocation_id.x < 32 holds for BOTH
          * subgroup 0 (y=0) and subgroup 1 (y=1), so must NOT narrow! */
         nir_push_if(&b, nir_ult_imm(&b, t, 32));
         store_shared_word(&b, t, addr);
         nir_pop_if(&b, NULL);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
         nir_push_if(&b, nir_ult_imm(&b, t, 32));
         nir_def *v = load_shared_word(&b, addr);
         store_global_word(&b, v, nir_imm_int64(&b, 0));
         nir_pop_if(&b, NULL);
      }

      EXPECT_FALSE(agx_nir_narrow_subgroup0_barriers(b.shader))
         << "shape=" << shape;
      nir_validate_shader(b.shader, "refusals");
      ralloc_free(b.shader);
   }
}

TEST(NarrowBarriers, Subgroup0Linearized2DAnd3DLocalId)
{
   /* 2D 16x16x1 reduction using imadshl_agx(lid.x, 1, lid.y, 4) (lowered form
    * of gl_LocalInvocationIndex / HLSL SV_GroupIndex). Barriers after tid < 32
    * and tid < 16 must narrow while the initial full-workgroup barrier stays
    * SCOPE_WORKGROUP. */
   for (unsigned mode = 0; mode < 5; ++mode) {
      const bool is_3d = mode == 2;
      nir_builder b = is_3d ? compute_shader("linear-3d", 8, 8, 4)
                            : compute_shader("linear-2d", 16, 16, 1);
      b.shader->info.shared_size = 1024;
      nir_def *lid = nir_load_local_invocation_id(&b);
      nir_def *lx = nir_channel(&b, lid, 0);
      nir_def *ly = nir_channel(&b, lid, 1);
      nir_def *lz = nir_channel(&b, lid, 2);
      nir_def *t = NULL;
      if (mode == 0) {
         t = nir_imadshl_agx(&b, lx, nir_imm_int(&b, 1), ly, nir_imm_int(&b, 4));
      } else if (mode == 1) {
         t = nir_ior(&b, lx, nir_ishl_imm(&b, ly, 4));
      } else if (mode == 2) {
         nir_def *yz = nir_imadshl_agx(&b, ly, nir_imm_int(&b, 1), lz,
                                       nir_imm_int(&b, 3));
         t = nir_imadshl_agx(&b, lx, nir_imm_int(&b, 1), yz, nir_imm_int(&b, 3));
      } else if (mode == 3) {
         /* Wrong stride (8 instead of 16): tid < 32 reaches y=2..3 (subgroup 1)! */
         t = nir_imadshl_agx(&b, lx, nir_imm_int(&b, 1), ly, nir_imm_int(&b, 3));
      } else {
         /* Transposed (ly + lx * 16): tid < 32 covers x=0..1 across all y=0..15! */
         t = nir_imadshl_agx(&b, ly, nir_imm_int(&b, 1), lx, nir_imm_int(&b, 4));
      }
      nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));
      store_shared_word(&b, t, addr);
      emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      for (unsigned off : {32u, 16u}) {
         nir_push_if(&b, nir_ult_imm(&b, t, off));
         nir_def *other = load_shared_word(
            &b, nir_u2u16(&b, nir_ishl_imm(&b, nir_iadd_imm(&b, t, off), 2)));
         nir_def *mine = load_shared_word(&b, addr);
         store_shared_word(&b, nir_iadd(&b, mine, other), addr);
         nir_pop_if(&b, NULL);
         emit_barrier(&b, nir_var_mem_shared, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
      }
      nir_push_if(&b, nir_ieq_imm(&b, t, 0));
      nir_def *sum = load_shared_word(&b, nir_imm_intN_t(&b, 0, 16));
      store_global_word(&b, sum, nir_imm_int64(&b, 0));
      nir_pop_if(&b, NULL);

      const bool expect_narrow = mode < 3;
      EXPECT_EQ(agx_nir_narrow_subgroup0_barriers(b.shader), expect_narrow)
         << "mode=" << mode;
      nir_validate_shader(b.shader, "linear-local-id");
      mesa_scope scopes[3];
      collect_barrier_exec_scopes(b.shader, scopes, 3);
      EXPECT_EQ(scopes[0], SCOPE_WORKGROUP) << "mode=" << mode;
      EXPECT_EQ(scopes[1], expect_narrow ? SCOPE_SUBGROUP : SCOPE_WORKGROUP)
         << "mode=" << mode;
      EXPECT_EQ(scopes[2], expect_narrow ? SCOPE_SUBGROUP : SCOPE_WORKGROUP)
         << "mode=" << mode;
      ralloc_free(b.shader);
   }
}
