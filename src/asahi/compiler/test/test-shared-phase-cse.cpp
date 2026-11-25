/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* agx_nir_opt_shared_phase_cse() merges identical scalar 32-bit shared
 * loads of one block of a loop that runs between two uniform workgroup
 * acquire-release barriers on shared memory, and only when nothing in the
 * loop can write memory or leave it another way. Every refusal below is
 * one condition of that rule; the shader is otherwise the accepted one.
 */

namespace {

struct phase {
   /* Barrier before and after the loop. */
   bool pre_barrier = true, post_barrier = true;
   mesa_scope exec = SCOPE_WORKGROUP, mem = SCOPE_DEVICE;
   nir_variable_mode modes = nir_var_mem_shared;
   nir_memory_semantics sem = NIR_MEMORY_ACQ_REL;
   /* The phase runs only for some invocations. */
   bool divergent_entry = false;
   /* Something else in the loop. */
   bool store = false, atomic = false, global_load = false;
   /* The second load: other offset, in the break's block. */
   bool other_offset = false, other_block = false;
   /* The form of both loads. */
   unsigned comps = 1, bit_size = 32;
   enum gl_access_qualifier access = (enum gl_access_qualifier)0;
};

class SharedPhaseCSE : public testing::Test {
 protected:
   SharedPhaseCSE()
   {
      b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                         "shared phase");
      b.shader->info.shared_size = 256;
   }

   ~SharedPhaseCSE()
   {
      ralloc_free(b.shader);
   }

   void barrier(const phase &p)
   {
      nir_barrier(&b, .execution_scope = p.exec, .memory_scope = p.mem,
                  .memory_semantics = p.sem, .memory_modes = p.modes);
   }

   nir_def *load(unsigned comps, unsigned bit_size, nir_def *offset,
                 enum gl_access_qualifier access)
   {
      return nir_load_shared(&b, comps, bit_size, offset, .base = 0,
                             .access = access,
                             .align_mul = comps * bit_size / 8);
   }

   /* Two loads of shared[16] inside the phase, both used. */
   void build(const phase &p)
   {
      nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_if *nif = NULL;
      if (p.divergent_entry) {
         nir_def *lane = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
         nif = nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      }

      /* Keep the barrier the last instruction of the block before the loop.
       * The loads share their offset def, as after nir_opt_cse. */
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *off = nir_imm_int(&b, 16);
      nir_def *other = p.other_offset ? nir_imm_int(&b, 20) : off;
      if (p.pre_barrier)
         barrier(p);

      nir_loop *loop = nir_push_loop(&b);
      {
         first = load(p.comps, p.bit_size, off, p.access);
         nir_def *x = nir_u2u32(&b, nir_channel(&b, first, 0));
         if (p.store)
            nir_store_shared(&b, zero, nir_imm_int(&b, 64), .align_mul = 4);
         if (p.atomic)
            nir_shared_atomic(&b, 32, nir_imm_int(&b, 64), zero,
                              .atomic_op = nir_atomic_op_iadd);
         if (p.global_load)
            x = nir_iadd(&b, x, nir_load_global(&b, 1, 32, addr));

         /* Only ALU uses: anything else in the loop refuses the phase. */
         if (!p.other_block) {
            second = load(p.comps, p.bit_size, other, p.access);
            sum = nir_iadd(&b, x, nir_u2u32(&b, nir_channel(&b, second, 0)));
         }

         nir_if *brk = nir_push_if(&b, nir_ieq_imm(&b, sum ? sum : x, 7));
         if (p.other_block) {
            second = load(p.comps, p.bit_size, other, p.access);
            sum = nir_iadd(&b, x, nir_u2u32(&b, nir_channel(&b, second, 0)));
         }
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, brk);
         nir_jump(&b, nir_jump_break);
      }
      nir_pop_loop(&b, loop);

      if (p.post_barrier)
         barrier(p);
      nir_store_global(&b, zero, addr);

      if (nif)
         nir_pop_if(&b, nif);

      nir_validate_shader(b.shader, "built");
   }

   unsigned shared_loads()
   {
      unsigned n = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_intrinsic &&
                 nir_instr_as_intrinsic(instr)->intrinsic ==
                    nir_intrinsic_load_shared;
         }
      }
      return n;
   }

   /* Run the pass; it reports progress exactly when it removed a load. */
   bool run()
   {
      unsigned before = shared_loads();
      bool progress = agx_nir_opt_shared_phase_cse(b.shader);
      nir_validate_shader(b.shader, "after the pass");
      EXPECT_EQ(progress, shared_loads() != before);
      return progress;
   }

   void expect_kept(const phase &p)
   {
      build(p);
      EXPECT_FALSE(run());
      EXPECT_EQ(shared_loads(), 2u);
   }

   nir_builder b;
   nir_def *first = NULL, *second = NULL, *sum = NULL;
};

} // namespace

TEST_F(SharedPhaseCSE, MergesIdenticalLoadsOfOneBlock)
{
   build(phase());
   EXPECT_TRUE(run());
   EXPECT_EQ(shared_loads(), 1u);

   /* The first load stays where it was and takes the second's uses. */
   nir_intrinsic_instr *kept = nir_def_as_intrinsic(first);
   EXPECT_EQ(kept->intrinsic, nir_intrinsic_load_shared);
   EXPECT_EQ(kept->instr.block->cf_node.parent->type, nir_cf_node_loop);
   EXPECT_EQ(nir_def_as_alu(sum)->src[1].src.ssa, first);

   /* A second run finds nothing more. */
   EXPECT_FALSE(agx_nir_opt_shared_phase_cse(b.shader));
}

TEST_F(SharedPhaseCSE, KeepsOtherOffsets)
{
   phase p;
   p.other_offset = true;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsLoadsOfOtherBlocks)
{
   phase p;
   p.other_block = true;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsVectorLoads)
{
   phase p;
   p.comps = 2;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsNarrowLoads)
{
   phase p;
   p.bit_size = 16;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsVolatileLoads)
{
   phase p;
   p.access = ACCESS_VOLATILE;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsLoadsWhenTheLoopStores)
{
   phase p;
   p.store = true;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsLoadsWhenTheLoopHasAnAtomic)
{
   phase p;
   p.atomic = true;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, KeepsLoadsWhenTheLoopReadsOtherMemory)
{
   phase p;
   p.global_load = true;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsABarrierBefore)
{
   phase p;
   p.pre_barrier = false;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsABarrierAfter)
{
   phase p;
   p.post_barrier = false;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsWorkgroupExecutionScope)
{
   phase p;
   p.exec = SCOPE_SUBGROUP;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsDeviceMemoryScope)
{
   phase p;
   p.mem = SCOPE_WORKGROUP;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsSharedMemoryModesOnly)
{
   phase p;
   p.modes = (nir_variable_mode)(nir_var_mem_shared | nir_var_mem_global);
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsAcquireRelease)
{
   phase p;
   p.sem = NIR_MEMORY_ACQUIRE;
   expect_kept(p);
}

TEST_F(SharedPhaseCSE, NeedsUniformEntry)
{
   phase p;
   p.divergent_entry = true;
   expect_kept(p);
}
