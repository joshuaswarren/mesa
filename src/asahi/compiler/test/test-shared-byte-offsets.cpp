/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include <vector>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
#include "agx_nir_match_gemm.h"
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

enum class GemmMutation {
   None,
   NoContraction,
   OverlappingTiles,
   UninitializedTileSlot,
   UninitializedAccumulator,
   DivergentBreak,
   SharedAccessAfterLoop,
};

static nir_phi_instr *
create_scalar_phi(nir_shader *shader)
{
   nir_phi_instr *phi = nir_phi_instr_create(shader);
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   return phi;
}

static nir_builder
build_gemm_shader(GemmMutation mutation)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "gemm-match");
   b.shader->info.io_lowered = true;
   b.shader->info.shared_element_index_agx = true;
   b.shader->info.workgroup_size[0] = AGX_GEMM_WIDTH;
   b.shader->info.workgroup_size[1] = AGX_GEMM_SUBGROUPS;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.min_subgroup_size = AGX_GEMM_WIDTH;
   b.shader->info.max_subgroup_size = AGX_GEMM_WIDTH;
   b.shader->info.shared_size = 8 * AGX_GEMM_TILE_WORDS;

   const int a_bytes = 0;
   const int b_bytes = mutation == GemmMutation::OverlappingTiles ?
      2 * AGX_GEMM_TILE_WORDS : 4 * AGX_GEMM_TILE_WORDS;

   nir_def *n_tiles = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *cond = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
   nir_def *local = nir_load_local_invocation_id(&b);
   nir_def *x = nir_channel(&b, local, 0);
   nir_def *y8 = nir_ishl_imm(&b, nir_channel(&b, local, 1), 3);
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *k_zero = nir_imm_int(&b, 0);

   auto fence = [&]() {
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                        .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL,
                        .memory_modes = nir_var_mem_shared);
   };

   nir_if *guard = nir_push_if(&b, nir_ine_imm(&b, n_tiles, 0));
   nir_loop *loop = nir_push_loop(&b);
   nir_block *preheader = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);

   nir_phi_instr *k_phi = create_scalar_phi(b.shader);
   nir_phi_instr_add_src(k_phi, preheader, k_zero);

   nir_phi_instr *initial[AGX_GEMM_ROWS_PER_LANE];
   nir_def *acc[AGX_GEMM_ROWS_PER_LANE];
   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      initial[row] = create_scalar_phi(b.shader);
      nir_phi_instr_add_src(initial[row], preheader, zero);
      acc[row] = &initial[row]->def;
   }

   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_def *row_idx = nir_iadd_imm(&b, y8, row);
      nir_def *word = nir_u2u16(&b, nir_iadd(&b, nir_ishl_imm(&b, row_idx, 5), x));
      nir_store_shared(&b, one, word, .base = a_bytes, .align_mul = 4);
      if (mutation == GemmMutation::UninitializedTileSlot &&
          row == AGX_GEMM_ROWS_PER_LANE - 1) {
         nir_push_if(&b, nir_ine_imm(&b, cond, 0));
         nir_store_shared(&b, one, word, .base = b_bytes, .align_mul = 4);
         nir_pop_if(&b, NULL);
      } else {
         nir_store_shared(&b, one, word, .base = b_bytes, .align_mul = 4);
      }
   }

   fence();

   for (unsigned k = 0; k < AGX_GEMM_WIDTH; ++k) {
      nir_def *b_idx = nir_u2u16(&b, nir_iadd_imm(&b, x, k * AGX_GEMM_WIDTH));
      nir_def *b_val = nir_load_shared(&b, 1, 32, b_idx, .base = b_bytes, .align_mul = 4);
      for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
         nir_def *row_idx = nir_iadd_imm(&b, y8, row);
         nir_def *a_idx =
            nir_u2u16(&b, nir_iadd_imm(&b, nir_ishl_imm(&b, row_idx, 5), k));
         nir_def *a_val =
            nir_load_shared(&b, 1, 32, a_idx, .base = a_bytes, .align_mul = 4);
         acc[row] = nir_ffma(&b, a_val, b_val, acc[row]);
         if (mutation == GemmMutation::NoContraction && k == 15 && row == 3) {
            nir_def_as_alu(acc[row])->fp_math_ctrl =
               (nir_fp_math_control)(nir_fp_no_contract | nir_fp_no_transform);
         }
      }
   }

   fence();

   nir_def *k_next = nir_iadd_imm(&b, &k_phi->def, 1);
   nir_def *done = mutation == GemmMutation::DivergentBreak ?
      nir_ine_imm(&b, x, 0) : nir_uge(&b, k_next, n_tiles);
   nir_push_if(&b, done);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_instr_add_src(k_phi, latch, k_next);
   nir_instr_insert(nir_before_block(header), &k_phi->instr);
   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_phi_instr_add_src(initial[row], latch, acc[row]);
      nir_instr_insert(nir_before_block(header), &initial[row]->instr);
   }

   nir_pop_loop(&b, loop);

   nir_push_else(&b, guard);
   nir_def *skip_val = mutation == GemmMutation::UninitializedAccumulator ?
      nir_undef(&b, 1, 32) : zero;
   nir_pop_if(&b, guard);

   nir_block *after_loop = nir_if_last_then_block(guard);
   nir_block *else_block = nir_if_last_else_block(guard);

   nir_phi_instr *exit_phi[AGX_GEMM_ROWS_PER_LANE];
   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      exit_phi[row] = create_scalar_phi(b.shader);
      nir_phi_instr_add_src(exit_phi[row], after_loop, acc[row]);
      nir_phi_instr_add_src(exit_phi[row], else_block, row == 0 ? skip_val : zero);
      nir_builder_instr_insert(&b, &exit_phi[row]->instr);
   }

   nir_def *total = &exit_phi[0]->def;
   for (unsigned row = 1; row < AGX_GEMM_ROWS_PER_LANE; ++row)
      total = nir_fadd(&b, total, &exit_phi[row]->def);

   if (mutation == GemmMutation::SharedAccessAfterLoop) {
      nir_def *tail =
         nir_load_shared(&b, 1, 32, nir_imm_intN_t(&b, 0, 16), .base = 0, .align_mul = 4);
      total = nir_fadd(&b, total, tail);
   }

   nir_store_global(&b, total, nir_imm_int64(&b, 0x1000), .align_mul = 4);
   nir_validate_shader(b.shader, "gemm-match setup");
   return b;
}

static void
check_gemm_refusal(GemmMutation mutation)
{
   nir_builder b = build_gemm_shader(mutation);
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_divergence |
                              nir_metadata_dominance);
   char *before = nir_shader_as_str(b.shader, b.shader);

   struct agx_gemm_match match = {};
   EXPECT_FALSE(agx_nir_match_gemm(b.shader, &match));
   nir_validate_shader(b.shader, "gemm-match refusal");

   char *after = nir_shader_as_str(b.shader, b.shader);
   EXPECT_STREQ(before, after);
   ralloc_free(b.shader);
}

TEST(GemmMatch, AcceptAndLower)
{
   nir_builder b = build_gemm_shader(GemmMutation::None);
   struct agx_gemm_match match = {};
   ASSERT_TRUE(agx_nir_match_gemm(b.shader, &match));
   EXPECT_EQ(match.a_bytes, 0u);
   EXPECT_EQ(match.b_bytes, 4u * AGX_GEMM_TILE_WORDS);

   ASSERT_TRUE(agx_nir_lower_gemm(b.shader, &match));
   nir_validate_shader(b.shader, "gemm-match lowered");

   unsigned cmat_ops = 0;
   unsigned shared_barriers = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (agx_nir_is_shared_phase_barrier(instr))
               ++shared_barriers;
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_cmat_muladd_agx)
               ++cmat_ops;
         }
      }
   }

   EXPECT_EQ(cmat_ops, 16u);
   EXPECT_EQ(count_alu(b.shader, nir_op_ffma), 0u);
   EXPECT_EQ(shared_barriers, 3u);
   ralloc_free(b.shader);
}

TEST(GemmMatch, NoContraction) { check_gemm_refusal(GemmMutation::NoContraction); }
TEST(GemmMatch, OverlappingTiles) { check_gemm_refusal(GemmMutation::OverlappingTiles); }
TEST(GemmMatch, UninitializedTileSlot) { check_gemm_refusal(GemmMutation::UninitializedTileSlot); }
TEST(GemmMatch, UninitializedAccumulator) { check_gemm_refusal(GemmMutation::UninitializedAccumulator); }
TEST(GemmMatch, DivergentBreak) { check_gemm_refusal(GemmMutation::DivergentBreak); }
TEST(GemmMatch, SharedAccessAfterLoop) { check_gemm_refusal(GemmMutation::SharedAccessAfterLoop); }
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

/* A strided cluster spanning multiple 256-byte windows splits each element
 * offset into a shared chunk base (V + 64*k) plus immediate byte offset
 * (0/64/128/192), letting CSE collapse 8 taps into 2 adds instead of 15. */
TEST(SharedByteOffsets, MultiChunkClusterSharesChunkBases)
{
   nir_builder b = workgroup_shader("shared-multichunk-cluster");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 16;
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *lx = nir_channel(&b, id, 0);
   nir_def *ly = nir_channel(&b, id, 1);
   nir_def *row = nir_ishl_imm(&b, ly, 4);

   nir_intrinsic_instr *loads[8];
   nir_def *sum = NULL;
   for (unsigned t = 0; t < 8; ++t) {
      nir_def *row_t = t ? nir_iadd_imm(&b, row, t * 16) : row;
      nir_def *idx = nir_u2u16(&b, nir_iadd(&b, row_t, lx));
      nir_def *val = nir_load_shared(&b, 1, 32, idx, .align_mul = 4);
      loads[t] = nir_def_as_intrinsic(val);
      sum = sum ? nir_iadd(&b, sum, val) : val;
   }
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_shared_byte_offsets(b.shader));
   nir_opt_cse(b.shader);
   nir_opt_dce(b.shader);

   for (unsigned t = 0; t < 8; ++t)
      EXPECT_EQ(nir_intrinsic_base(loads[t]), (t % 4) * 64);

   /* First 4 taps share u2u16(V); next 4 share u2u16(V + 64). */
   for (unsigned t = 1; t < 4; ++t)
      EXPECT_EQ(loads[t]->src[0].ssa, loads[0]->src[0].ssa);
   for (unsigned t = 5; t < 8; ++t)
      EXPECT_EQ(loads[t]->src[0].ssa, loads[4]->src[0].ssa);
   EXPECT_NE(loads[0]->src[0].ssa, loads[4]->src[0].ssa);

   /* Only 2 address adds remain (V = lx + row, V + 64), plus 7 value adds. */
   EXPECT_EQ(count_alu(b.shader, nir_op_iadd), 9u);

   nir_validate_shader(b.shader, "shared-multichunk-cluster result");
   ralloc_free(b.shader);
}

/* Unbounded multi-chunk clusters must not split across a possible u16 wrap. */
TEST(SharedByteOffsets, MultiChunkClusterUnboundedDoesNotSplit)
{
   nir_builder b = workgroup_shader("shared-multichunk-unbounded");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *y = nir_channel(&b, nir_load_local_invocation_id(&b), 1);

   nir_intrinsic_instr *loads[2];
   for (unsigned t = 0; t < 2; ++t) {
      nir_def *idx = nir_u2u16(&b, nir_iadd(&b, nir_iadd_imm(&b, x, 64 + t * 16), y));
      loads[t] = nir_def_as_intrinsic(
         nir_load_shared(&b, 1, 32, idx, .align_mul = 4));
   }

   EXPECT_FALSE(agx_nir_opt_shared_byte_offsets(b.shader));
   EXPECT_EQ(nir_intrinsic_base(loads[0]), 0);
   EXPECT_EQ(nir_intrinsic_base(loads[1]), 0);
   nir_validate_shader(b.shader, "shared-multichunk-unbounded result");
   ralloc_free(b.shader);
}

static std::vector<uint32_t>
eval_shared_byte_addrs(const nir_shader *orig, uint32_t lx, uint32_t ly,
                       uint32_t pc)
{
   nir_shader *s = nir_shader_clone(NULL, orig);
   nir_foreach_function_impl(impl, s) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            b.cursor = nir_before_instr(instr);
            if (intr->intrinsic == nir_intrinsic_load_local_invocation_id) {
               nir_def_replace(&intr->def, nir_imm_ivec3(&b, lx, ly, 0));
            } else if (intr->intrinsic == nir_intrinsic_load_push_constant) {
               nir_def_replace(&intr->def, nir_imm_int(&b, pc));
            }
         }
      }
   }

   bool progress;
   do {
      progress = false;
      progress |= nir_opt_constant_folding(s);
      progress |= nir_opt_copy_prop(s);
      progress |= nir_opt_dce(s);
   } while (progress);

   std::vector<uint32_t> addrs;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_shared &&
                intr->intrinsic != nir_intrinsic_store_shared)
               continue;
            unsigned bits = intr->intrinsic == nir_intrinsic_load_shared
                               ? intr->def.bit_size
                               : nir_src_bit_size(intr->src[0]);
            unsigned shift = util_logbase2(bits / 8);
            nir_src *off = nir_get_io_offset_src(intr);
            EXPECT_TRUE(nir_src_is_const(*off));
            EXPECT_LE(nir_intrinsic_base(intr), 255u);
            uint32_t elem = nir_src_as_uint(*off) & 0xffffu;
            addrs.push_back((elem << shift) + nir_intrinsic_base(intr));
         }
      }
   }

   ralloc_free(s);
   return addrs;
}

/* Nonzero .base whose sum crosses a 256-byte chunk boundary: taps whose
 * base + (e_lo << shift) would exceed 255 stay untouched while in-range taps
 * fold cleanly into chunks, preserving every effective byte address. */
TEST(SharedByteOffsets, ValueEquivalenceNonzeroBaseCrossesChunk)
{
   nir_builder b = workgroup_shader("shared-nonzero-base-chunk");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 16;
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *lx = nir_channel(&b, id, 0);
   nir_def *ly = nir_channel(&b, id, 1);
   nir_def *row = nir_ishl_imm(&b, ly, 4);

   const unsigned elems[] = {0, 16, 48, 60, 64, 80, 120, 124};
   nir_intrinsic_instr *loads[8];
   nir_def *sum = NULL;
   for (unsigned t = 0; t < 8; ++t) {
      nir_def *row_t = elems[t] ? nir_iadd_imm(&b, row, elems[t]) : row;
      nir_def *idx = nir_u2u16(&b, nir_iadd(&b, row_t, lx));
      nir_def *val = nir_load_shared(&b, 1, 32, idx, .base = 16, .align_mul = 4);
      loads[t] = nir_def_as_intrinsic(val);
      sum = sum ? nir_iadd(&b, sum, val) : val;
   }
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   nir_shader *orig = nir_shader_clone(NULL, b.shader);
   EXPECT_TRUE(agx_nir_opt_shared_byte_offsets(b.shader));
   nir_opt_cse(b.shader);
   nir_opt_dce(b.shader);

   const unsigned expected_base[] = {16, 80, 208, 16, 16, 80, 240, 16};
   for (unsigned t = 0; t < 8; ++t)
      EXPECT_EQ(nir_intrinsic_base(loads[t]), expected_base[t]);
   EXPECT_EQ(loads[4]->src[0].ssa, loads[5]->src[0].ssa);
   EXPECT_EQ(loads[4]->src[0].ssa, loads[6]->src[0].ssa);

   const uint32_t coords[][2] = {{0, 0}, {15, 0}, {0, 15}, {15, 15}, {7, 9}};
   for (const auto &c : coords) {
      EXPECT_EQ(eval_shared_byte_addrs(b.shader, c[0], c[1], 0),
                eval_shared_byte_addrs(orig, c[0], c[1], 0));
   }

   nir_validate_shader(b.shader, "shared-nonzero-base-chunk result");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

/* Multi-chunk split is accepted when variable upper bound + elements == 0xffff
 * and rejected when it reaches 0x10000. */
TEST(SharedByteOffsets, ValueEquivalenceExact0xffffBound)
{
   for (unsigned over = 0; over < 2; ++over) {
      nir_builder b = workgroup_shader("shared-exact-0xffff");
      nir_def *pc = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                           .range = 4);
      uint32_t mask = (0xffffu - 80u) + over;
      nir_def *v = nir_iand_imm(&b, pc, mask);
      nir_def *z = nir_imm_int(&b, 0);

      const unsigned elems[] = {64, 80};
      nir_def *sum = NULL;
      for (unsigned t = 0; t < 2; ++t) {
         nir_def *idx = nir_u2u16(&b, nir_iadd(&b, nir_iadd_imm(&b, v, elems[t]), z));
         nir_def *val = nir_load_shared(&b, 1, 32, idx, .align_mul = 4);
         sum = sum ? nir_iadd(&b, sum, val) : val;
      }
      nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

      nir_shader *orig = nir_shader_clone(NULL, b.shader);
      EXPECT_EQ(agx_nir_opt_shared_byte_offsets(b.shader), over == 0);
      nir_opt_cse(b.shader);
      nir_opt_dce(b.shader);

      const uint32_t pcs[] = {0u, 1234u, mask, 0xffffffffu};
      for (uint32_t p : pcs) {
         EXPECT_EQ(eval_shared_byte_addrs(b.shader, 0, 0, p),
                   eval_shared_byte_addrs(orig, 0, 0, p));
      }

      nir_validate_shader(b.shader, "shared-exact-0xffff result");
      ralloc_free(orig);
      ralloc_free(b.shader);
   }
}

/* 16-bit elements (shift = 1, 128 elements per 256-byte chunk) and
 * store_shared multi-chunk clustering preserve exact effective byte addresses. */
TEST(SharedByteOffsets, ValueEquivalence16BitElementsAndStoreShared)
{
   for (unsigned is_store = 0; is_store < 2; ++is_store) {
      nir_builder b = workgroup_shader("shared-16bit-multichunk");
      b.shader->info.workgroup_size[0] = 16;
      b.shader->info.workgroup_size[1] = 16;
      nir_def *id = nir_load_local_invocation_id(&b);
      nir_def *lx = nir_channel(&b, id, 0);
      nir_def *ly = nir_channel(&b, id, 1);
      nir_def *row = nir_ishl_imm(&b, ly, 4);

      nir_intrinsic_instr *intrs[8];
      nir_def *sum = NULL;
      for (unsigned t = 0; t < 8; ++t) {
         nir_def *row_t = t ? nir_iadd_imm(&b, row, t * 32) : row;
         nir_def *idx = nir_u2u16(&b, nir_iadd(&b, row_t, lx));
         if (is_store) {
            intrs[t] = nir_store_shared(&b, nir_imm_intN_t(&b, t, 16), idx,
                                        .base = 8, .align_mul = 2);
         } else {
            nir_def *val = nir_load_shared(&b, 1, 16, idx, .base = 8,
                                           .align_mul = 2);
            intrs[t] = nir_def_as_intrinsic(val);
            sum = sum ? nir_iadd(&b, sum, val) : val;
         }
      }
      if (!is_store)
         nir_store_global(&b, nir_u2u32(&b, sum), nir_imm_int64(&b, 0),
                          .align_mul = 4);

      nir_shader *orig = nir_shader_clone(NULL, b.shader);
      EXPECT_TRUE(agx_nir_opt_shared_byte_offsets(b.shader));
      nir_opt_cse(b.shader);
      nir_opt_dce(b.shader);

      for (unsigned t = 0; t < 8; ++t)
         EXPECT_EQ(nir_intrinsic_base(intrs[t]), 8u + (t % 4) * 64u);
      for (unsigned t = 1; t < 4; ++t)
         EXPECT_EQ(nir_get_io_offset_src(intrs[t])->ssa,
                   nir_get_io_offset_src(intrs[0])->ssa);
      for (unsigned t = 5; t < 8; ++t)
         EXPECT_EQ(nir_get_io_offset_src(intrs[t])->ssa,
                   nir_get_io_offset_src(intrs[4])->ssa);
      EXPECT_NE(nir_get_io_offset_src(intrs[0])->ssa,
                nir_get_io_offset_src(intrs[4])->ssa);

      const uint32_t coords[][2] = {{0, 0}, {15, 0}, {0, 15}, {15, 15}};
      for (const auto &c : coords) {
         EXPECT_EQ(eval_shared_byte_addrs(b.shader, c[0], c[1], 0),
                   eval_shared_byte_addrs(orig, c[0], c[1], 0));
      }

      nir_validate_shader(b.shader, "shared-16bit-multichunk result");
      ralloc_free(orig);
      ralloc_free(b.shader);
   }
}

/* Same-block shared accesses in a <= 32 KiB allocation share greedy 256-byte
 * windows even when their variable terms have unknown 32-bit upper bounds
 * (such as preamble uniforms or subgroup IDs). */
TEST(SharedByteOffsets, SmallSharedGreedyAnchorUnbounded)
{
   nir_builder b = workgroup_shader("shared-small-greedy");
   b.shader->info.shared_size = 14336;
   nir_def *u = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *row = nir_imul_imm(&b, nir_iand_imm(&b, lane, 12), 40);
   nir_def *base_var = nir_iadd(&b, u, row);

   /* 2576 + {0, 40, 80, 120} crosses the 2688 multiple-of-128 boundary at
    * 2696, but fits in one 256-byte window (byte offsets 0, 80, 160, 240)
    * when anchored at 2576. */
   nir_intrinsic_instr *intrs[4];
   nir_def *sum = NULL;
   for (unsigned t = 0; t < 4; ++t) {
      nir_def *idx = nir_u2u16(&b, nir_iadd_imm(&b, base_var, 2576u + t * 40u));
      nir_def *val = nir_load_shared(&b, 2, 16, idx, .align_mul = 4);
      intrs[t] = nir_def_as_intrinsic(val);
      nir_def *x = nir_channel(&b, val, 0);
      sum = sum ? nir_iadd(&b, sum, x) : x;
   }
   nir_store_global(&b, nir_u2u32(&b, sum), nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_shared_byte_offsets(b.shader));
   nir_opt_cse(b.shader);
   nir_opt_dce(b.shader);

   for (unsigned t = 0; t < 4; ++t) {
      EXPECT_EQ(nir_intrinsic_base(intrs[t]), (int)(t * 80u));
      EXPECT_EQ(nir_get_io_offset_src(intrs[t])->ssa,
                nir_get_io_offset_src(intrs[0])->ssa);
   }

   nir_validate_shader(b.shader, "shared-small-greedy result");
   ralloc_free(b.shader);
}

/* Allocations > 32 KiB must not use the 32 KiB no-wrap assumption. */
TEST(SharedByteOffsets, LargeSharedRejectsUnbounded)
{
   nir_builder b = workgroup_shader("shared-large-unbounded");
   b.shader->info.shared_size = 36864;
   nir_def *u = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);

   nir_def *a = nir_load_shared(&b, 2, 16, nir_u2u16(&b, u), .align_mul = 4);
   nir_def *c = nir_load_shared(&b, 2, 16, nir_u2u16(&b, nir_iadd_imm(&b, u, 40)),
                                .align_mul = 4);
   nir_store_global(&b, nir_u2u32(&b, nir_iadd(&b, nir_channel(&b, a, 0),
                                               nir_channel(&b, c, 0))),
                    nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_FALSE(agx_nir_opt_shared_byte_offsets(b.shader));
   nir_validate_shader(b.shader, "shared-large-unbounded result");
   ralloc_free(b.shader);
}
