/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

static void
check_offset(unsigned bits, unsigned add, int base, uint32_t mask,
             bool store, bool expected)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared-byte-offset");
   nir_def *v = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   v = nir_iand_imm(&b, v, mask);
   nir_def *sum = nir_iadd_imm(&b, v, add);
   nir_def *index = nir_u2u16(&b, nir_ushr_imm(&b, sum, util_logbase2(bits / 8)));
   nir_intrinsic_instr *intr;
   if (store) {
      intr = nir_store_shared(&b, nir_imm_intN_t(&b, 0, bits), index,
                              .base = base, .align_mul = bits / 8);
   } else {
      nir_def *value = nir_load_shared(&b, 1, bits, index, .base = base,
                                        .align_mul = bits / 8);
      intr = nir_def_as_intrinsic(value);
   }
   EXPECT_EQ(agx_nir_opt_shared_byte_offsets(b.shader), expected);
   EXPECT_EQ(nir_intrinsic_base(intr), base + (expected ? add : 0));
   EXPECT_EQ(nir_get_io_offset_src(intr)->ssa == index, !expected);
   nir_validate_shader(b.shader, "shared-byte-offset result");
   ralloc_free(b.shader);
}

TEST(SharedByteOffsets, Load32) { check_offset(32, 12, 0, 4095, false, true); }
TEST(SharedByteOffsets, Store32) { check_offset(32, 56, 0, 4095, true, true); }
TEST(SharedByteOffsets, Load16) { check_offset(16, 14, 0, 4095, false, true); }
TEST(SharedByteOffsets, ExistingBase) { check_offset(32, 12, 16, 4095, false, true); }
TEST(SharedByteOffsets, ByteOffsetTooLarge) { check_offset(32, 256, 0, 4095, false, false); }
TEST(SharedByteOffsets, BaseOverflow) { check_offset(32, 16, 240, 4095, false, false); }
TEST(SharedByteOffsets, Misaligned) { check_offset(32, 3, 0, 4095, false, false); }
TEST(SharedByteOffsets, AdditionWrap) { check_offset(32, 12, 0, UINT32_MAX, false, false); }
TEST(SharedByteOffsets, NarrowingWrap) { check_offset(32, 12, 0, 262143, false, false); }
TEST(SharedByteOffsets, ExactBoundary) { check_offset(32, 12, 0, 262131, false, true); }

static nir_builder
workgroup_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static unsigned
count_alu(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_alu &&
                 nir_instr_as_alu(instr)->op == op;
         }
      }
   }
   return n;
}

/* A byte offset that scales thread indices by multiples of the element size,
 * including through address multiplies, becomes an element index without a
 * shift per access. */
TEST(SharedByteOffsets, ExactElementIndex)
{
   nir_builder b = workgroup_shader("shared-exact-index");
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *x = nir_ishl_imm(&b, nir_channel(&b, id, 0), 1);
   nir_def *y = nir_ishl_imm(&b, nir_channel(&b, id, 1), 1);
   nir_def *offset = nir_iadd(&b, nir_amul(&b, x, nir_imm_int(&b, 72)),
                              nir_amul(&b, y, nir_imm_int(&b, 4)));
   nir_store_shared(&b, nir_imm_int(&b, 0), offset, .align_mul = 4);
   EXPECT_TRUE(agx_nir_lower_shared_bitsize(b.shader, true));
   EXPECT_EQ(count_alu(b.shader, nir_op_ushr), 0u);
   nir_validate_shader(b.shader, "shared-exact-index result");
   ralloc_free(b.shader);
}

/* An odd scale still needs the shift. */
TEST(SharedByteOffsets, InexactElementIndex)
{
   nir_builder b = workgroup_shader("shared-inexact-index");
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *offset = nir_amul(&b, nir_channel(&b, id, 0), nir_imm_int(&b, 6));
   nir_store_shared(&b, nir_imm_int(&b, 0), offset, .align_mul = 4);
   EXPECT_TRUE(agx_nir_lower_shared_bitsize(b.shader, true));
   EXPECT_EQ(count_alu(b.shader, nir_op_ushr), 1u);
   ralloc_free(b.shader);
}

static void
check_element_addend(bool bounded, unsigned addend, unsigned expected_base)
{
   nir_builder b = workgroup_shader("shared-element-addend");
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *x = nir_channel(&b, id, 0);
   if (!bounded)
      x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *row = nir_imul_imm(&b, x, 36);
   nir_def *col = nir_imadshl_agx(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 1),
                                  nir_channel(&b, id, 1), nir_imm_int(&b, 1));
   nir_def *index =
      nir_u2u16(&b, nir_iadd(&b, nir_iadd_imm(&b, row, addend), col));
   nir_def *value = nir_load_shared(&b, 1, 32, index, .align_mul = 4);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(value);
   EXPECT_EQ(agx_nir_opt_shared_byte_offsets(b.shader), expected_base != 0);
   EXPECT_EQ(nir_intrinsic_base(intr), expected_base);
   if (expected_base) {
      /* The constant left the index: u2u16(row + col). */
      nir_scalar narrow = nir_scalar_resolved(intr->src[0].ssa, 0);
      ASSERT_TRUE(nir_scalar_is_alu(narrow));
      ASSERT_EQ(nir_scalar_alu_op(narrow), nir_op_u2u16);
      nir_scalar sum = nir_scalar_chase_alu_src(narrow, 0);
      ASSERT_TRUE(nir_scalar_is_alu(sum));
      ASSERT_EQ(nir_scalar_alu_op(sum), nir_op_iadd);
      for (unsigned s = 0; s < 2; ++s)
         EXPECT_FALSE(nir_scalar_is_const(nir_scalar_chase_alu_src(sum, s)));
   }
   nir_validate_shader(b.shader, "shared-element-addend result");
   ralloc_free(b.shader);
}

TEST(SharedByteOffsets, NestedElementAddend) { check_element_addend(true, 18, 72); }
TEST(SharedByteOffsets, NestedElementAddendTooLarge) { check_element_addend(true, 64, 0); }
TEST(SharedByteOffsets, NestedElementAddendUnbounded) { check_element_addend(false, 18, 0); }

enum class PairBoundary {
   None,
   WorkgroupScope,
   ImmediateBase,
   AliasingWrite,
   Volatile,
   Rendezvous,
   ShortAllocation,
   NarrowWrap,
};

static void
shared_pair_case(PairBoundary boundary)
{
   nir_builder b = workgroup_shader("shared-vec2");
   b.shader->info.shared_element_index_agx = true;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.shared_size = boundary == PairBoundary::ShortAllocation ? 32 : 128;
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *x = nir_ishl_imm(&b, nir_channel(&b, id, 0), 1);
   nir_def *low = nir_u2u16(&b, x);
   const int high_base = boundary == PairBoundary::ImmediateBase ? 8 : 4;
   nir_def *high = boundary == PairBoundary::ImmediateBase ? low :
      nir_u2u16(&b, nir_iadd_imm(&b, x,
         boundary == PairBoundary::NarrowWrap ? 65537 : 1));
   auto fence = [&]() {
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                        .memory_scope = boundary == PairBoundary::WorkgroupScope ?
                           SCOPE_WORKGROUP : SCOPE_DEVICE,
                        .memory_semantics = NIR_MEMORY_ACQ_REL,
                        .memory_modes = nir_var_mem_shared);
   };

   fence();
   nir_def *a = nir_load_shared(&b, 1, 32, low, .base = 4, .align_mul = 4);
   if (boundary == PairBoundary::AliasingWrite)
      nir_store_shared(&b, nir_imm_int(&b, 41), high, .base = 4, .align_mul = 4);
   if (boundary == PairBoundary::Rendezvous)
      fence();
   nir_def *c = nir_load_shared(&b, 1, 32, high, .base = high_base, .align_mul = 4);
   if (boundary == PairBoundary::Volatile)
      nir_intrinsic_set_access(nir_def_as_intrinsic(c), ACCESS_VOLATILE);
   fence();
   nir_def *sum = nir_iadd(&b, a, c);
   nir_store_shared(&b, sum, low, .align_mul = 4);

   const bool expect = boundary == PairBoundary::None ||
                       boundary == PairBoundary::WorkgroupScope ||
                       boundary == PairBoundary::ImmediateBase;
   nir_divergence_analysis(b.shader);
   ASSERT_EQ(agx_nir_opt_shared_vec2(b.shader), expect);
   nir_validate_shader(b.shader, "shared-vec2 result");

   unsigned scalar = 0, vector = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_shared)
               continue;
            scalar += intr->def.num_components == 1;
            vector += intr->def.num_components == 2;
            if (intr->def.num_components == 2) {
               EXPECT_EQ(intr->src[0].ssa, low);
               EXPECT_EQ(nir_intrinsic_base(intr), 4u);
            }
         }
      }
   }

   EXPECT_EQ(scalar, expect ? 0u : 2u);
   EXPECT_EQ(vector, expect ? 1u : 0u);
   if (expect) {
      nir_scalar a = nir_scalar_chase_alu_src(nir_get_scalar(sum, 0), 0);
      nir_scalar c = nir_scalar_chase_alu_src(nir_get_scalar(sum, 0), 1);
      EXPECT_EQ(a.def, c.def);
      EXPECT_EQ(a.comp, 0u);
      EXPECT_EQ(c.comp, 1u);
   }
   ralloc_free(b.shader);
}

TEST(SharedVec2, OddWordAddress) { shared_pair_case(PairBoundary::None); }
TEST(SharedVec2, WorkgroupBarrier) { shared_pair_case(PairBoundary::WorkgroupScope); }
TEST(SharedVec2, AliasingWrite) { shared_pair_case(PairBoundary::AliasingWrite); }
TEST(SharedVec2, Rendezvous) { shared_pair_case(PairBoundary::Rendezvous); }
TEST(SharedVec2, AllocationBound) { shared_pair_case(PairBoundary::ShortAllocation); }
TEST(SharedVec2, NarrowWrap) { shared_pair_case(PairBoundary::NarrowWrap); }
TEST(SharedVec2, MixedAddressUnits) { shared_pair_case(PairBoundary::ImmediateBase); }
TEST(SharedVec2, VolatileRead) { shared_pair_case(PairBoundary::Volatile); }

