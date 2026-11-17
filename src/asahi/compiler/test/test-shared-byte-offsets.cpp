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

/* Lowering bit size first (as agx_preprocess_nir() and deferred AGX3 compute
 * lowering do) and then running agx_nir_lower_shared_element_index() must
 * chase through u2u16 in 32 bits so constant byte addends still fold into
 * .base without leaving a 16-bit ushr. */
TEST(SharedByteOffsets, NarrowedByteOffsetElementIndexFolds)
{
   nir_builder b = workgroup_shader("shared-narrowed-element-index");
   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_intrinsic_instr *store = nir_store_shared(
      &b, nir_imm_int(&b, 7), nir_iadd_imm(&b, nir_imul_imm(&b, tid, 8), 16),
      .align_mul = 8);

   nir_def *v = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   v = nir_iand_imm(&b, v, 4095);
   nir_def *loaded = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, v, 12),
                                     .align_mul = 4);
   nir_intrinsic_instr *load = nir_def_as_intrinsic(loaded);
   nir_store_global(&b, loaded, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_lower_shared_bitsize(b.shader, false));
   EXPECT_TRUE(agx_nir_lower_shared_element_index(b.shader));
   EXPECT_TRUE(b.shader->info.shared_element_index_agx);
   EXPECT_TRUE(agx_nir_opt_shared_byte_offsets(b.shader));
   nir_opt_dce(b.shader);

   EXPECT_EQ(nir_intrinsic_base(store), 16);
   EXPECT_EQ(nir_intrinsic_base(load), 12);
   EXPECT_EQ(count_alu(b.shader, nir_op_ushr), 1u);

   nir_scalar narrow = nir_scalar_resolved(load->src[0].ssa, 0);
   ASSERT_TRUE(nir_scalar_is_alu(narrow));
   ASSERT_EQ(nir_scalar_alu_op(narrow), nir_op_u2u16);
   nir_scalar sh = nir_scalar_chase_alu_src(narrow, 0);
   ASSERT_TRUE(nir_scalar_is_alu(sh));
   EXPECT_EQ(sh.def->bit_size, 32u);
   EXPECT_EQ(nir_scalar_alu_op(sh), nir_op_ushr);

   nir_validate_shader(b.shader, "shared-narrowed-element-index result");
   ralloc_free(b.shader);
}
