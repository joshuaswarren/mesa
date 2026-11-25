/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

#define BUFFER_READ \
   ((gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER))

struct speculative_result {
   unsigned main_loads, preamble_loads;
   /* Reads through a selected base, and the size each compares with */
   unsigned guarded, end;
   bool main_speculatable;
};

/* A constant read of a uniform buffer under a non-uniform if. */
static speculative_result
run_speculative(gl_access_qualifier desc_access, unsigned index,
                unsigned used_halves)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "speculative buffer read");

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *desc = nir_load_constant_agx(
      &b, 4, 32, agx_test_load_preamble(&b, 1, 64, 0), zero,
      .access = desc_access, .base = 0, .format = PIPE_FORMAT_R32_UINT);
   nir_def *addr = nir_pack_64_2x32_split(&b, nir_channel(&b, desc, 0),
                                          nir_channel(&b, desc, 1));

   nir_def *elsewhere = nir_imm_vec4(&b, 0, 0, 0, 0);
   nir_push_if(&b, nir_ine_imm(&b, nir_load_sample_id(&b), 0));
   nir_def *read = nir_load_constant_agx(
      &b, 4, 32, addr, nir_imm_int(&b, index), .access = BUFFER_READ,
      .base = 2, .format = PIPE_FORMAT_R32_UINT);
   nir_pop_if(&b, NULL);
   agx_test_store_output(&b, nir_if_phi(&b, read, elsewhere), zero,
                         nir_type_float32, FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "before preamble");
   unsigned sizes[] = {used_halves, 0, 0};
   agx_nir_opt_preamble(b.shader, sizes, true, false, NULL);
   nir_validate_shader(b.shader, "after preamble");

   speculative_result r = {};
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_uge)
               r.end = nir_alu_src_as_uint(nir_instr_as_alu(instr)->src[1]);

            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (I->intrinsic != nir_intrinsic_load_constant_agx ||
                !(nir_intrinsic_access(I) & ACCESS_CAN_REORDER))
               continue;

            if (impl->function->is_preamble) {
               nir_alu_instr *base = nir_src_as_alu(I->src[0]);
               r.preamble_loads++;
               r.guarded += base && base->op == nir_op_bcsel &&
                            nir_alu_src_as_uint(base->src[2]) ==
                               AGX_ZERO_PAGE_ADDRESS;
            } else {
               r.main_loads++;
               r.main_speculatable |=
                  nir_intrinsic_access(I) & ACCESS_CAN_SPECULATE;
            }
         }
      }
   }
   ralloc_free(b.shader);
   return r;
}

#define VALID_DESCRIPTOR \
   ((gl_access_qualifier)(ACCESS_CAN_SPECULATE | ACCESS_BUFFER_DESCRIPTOR_AGX))

TEST(PreambleSpeculativeBuffer, MovesGuardedByTheBoundSize)
{
   speculative_result r = run_speculative(VALID_DESCRIPTOR, 3, 4);
   EXPECT_EQ(r.main_loads, 0u);
   EXPECT_EQ(r.preamble_loads, 1u);
   EXPECT_EQ(r.guarded, 1u);
   /* Four vec4s: the read ends 64 bytes in. */
   EXPECT_EQ(r.end, 64u);
}

TEST(PreambleSpeculativeBuffer, UnknownDescriptorStays)
{
   speculative_result r = run_speculative(ACCESS_CAN_SPECULATE, 3, 4);
   EXPECT_EQ(r.main_loads, 1u);
   EXPECT_EQ(r.guarded, 0u);
   EXPECT_FALSE(r.main_speculatable);
}

TEST(PreambleSpeculativeBuffer, StaysWhenStorageIsTaken)
{
   speculative_result r = run_speculative(VALID_DESCRIPTOR, 3, 478);
   EXPECT_EQ(r.main_loads, 1u);
   EXPECT_EQ(r.guarded, 0u);
   EXPECT_FALSE(r.main_speculatable);
}

TEST(PreambleSpeculativeBuffer, StaysPastTheZeroPage)
{
   speculative_result r =
      run_speculative(VALID_DESCRIPTOR, AGX_ZERO_PAGE_SIZE / 16, 4);
   EXPECT_EQ(r.main_loads, 1u);
   EXPECT_EQ(r.guarded, 0u);
   EXPECT_FALSE(r.main_speculatable);
}

TEST(PreambleSpeculativeBuffer, MovesInComputeAndAgx2)
{
   for (bool g17p : {false, true}) {
      SCOPED_TRACE(g17p);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "speculative compute read");

      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *desc = nir_load_constant_agx(
         &b, 4, 32, agx_test_load_preamble(&b, 1, 64, 0), zero,
         .access = VALID_DESCRIPTOR, .base = 0, .format = PIPE_FORMAT_R32_UINT);
      nir_def *addr = nir_pack_64_2x32_split(&b, nir_channel(&b, desc, 0),
                                             nir_channel(&b, desc, 1));

      nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *elsewhere = nir_imm_vec4(&b, 0, 0, 0, 0);
      nir_push_if(&b, nir_ine_imm(&b, lid, 0));
      nir_def *read = nir_load_constant_agx(
         &b, 4, 32, addr, nir_imm_int(&b, 2), .access = BUFFER_READ,
         .base = 2, .format = PIPE_FORMAT_R32_UINT);
      nir_pop_if(&b, NULL);
      nir_def *val = nir_if_phi(&b, read, elsewhere);
      nir_store_global(&b, val, nir_imm_int64(&b, UINT64_C(0x2000000000)),
                       .align_mul = 16);

      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      unsigned sizes[] = {4, 0, 0};
      agx_nir_opt_preamble(b.shader, sizes, g17p, false, NULL);
      nir_validate_shader(b.shader, "after compute preamble");

      unsigned main_loads = 0, preamble_loads = 0, guarded = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
               if (I->intrinsic != nir_intrinsic_load_constant_agx ||
                   !(nir_intrinsic_access(I) & ACCESS_CAN_REORDER))
                  continue;
               if (impl->function->is_preamble) {
                  nir_alu_instr *base = nir_src_as_alu(I->src[0]);
                  preamble_loads++;
                  guarded += base && base->op == nir_op_bcsel &&
                             nir_alu_src_as_uint(base->src[2]) ==
                                AGX_ZERO_PAGE_ADDRESS;
               } else {
                  main_loads++;
               }
            }
         }
      }
      EXPECT_EQ(main_loads, 0u);
      EXPECT_EQ(preamble_loads, 1u);
      EXPECT_EQ(guarded, 1u);
      ralloc_free(b.shader);
   }
}

/* When loads already carry ACCESS_CAN_SPECULATE (e.g. soft_fault on AGX2),
 * agx_nir_opt_preamble() must neither wrap hoisted loads in spurious zero-page
 * bcsel guards nor strip ACCESS_CAN_SPECULATE from main-shader loads. */
TEST(PreambleSpeculativeBuffer, SoftFaultAlreadySpeculatableUntouched)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "soft-fault speculative read");

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *desc = nir_load_constant_agx(
      &b, 4, 32, agx_test_load_preamble(&b, 1, 64, 0), zero,
      .access = VALID_DESCRIPTOR, .base = 0, .format = PIPE_FORMAT_R32_UINT);
   nir_def *addr = nir_pack_64_2x32_split(&b, nir_channel(&b, desc, 0),
                                          nir_channel(&b, desc, 1));

   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *elsewhere = nir_imm_vec4(&b, 0, 0, 0, 0);
   nir_push_if(&b, nir_ine_imm(&b, lid, 0));
   nir_def *hoisted = nir_load_constant_agx(
      &b, 4, 32, addr, nir_imm_int(&b, 2),
      .access = (gl_access_qualifier)(BUFFER_READ | ACCESS_CAN_SPECULATE),
      .base = 2, .format = PIPE_FORMAT_R32_UINT);
   nir_def *dynamic = nir_load_constant_agx(
      &b, 4, 32, addr, lid,
      .access = (gl_access_qualifier)(BUFFER_READ | ACCESS_CAN_SPECULATE),
      .base = 2, .format = PIPE_FORMAT_R32_UINT);
   nir_def *sum = nir_iadd(&b, hoisted, dynamic);
   nir_pop_if(&b, NULL);
   nir_def *val = nir_if_phi(&b, sum, elsewhere);
   nir_store_global(&b, val, nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 16);

   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   unsigned sizes[] = {4, 0, 0};
   agx_nir_opt_preamble(b.shader, sizes, false, false, NULL);
   nir_validate_shader(b.shader, "after soft-fault preamble");

   unsigned main_loads = 0, preamble_loads = 0, guarded = 0;
   bool main_speculatable = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (I->intrinsic != nir_intrinsic_load_constant_agx ||
                !(nir_intrinsic_access(I) & ACCESS_CAN_REORDER))
               continue;
            if (impl->function->is_preamble) {
               nir_alu_instr *base = nir_src_as_alu(I->src[0]);
               preamble_loads++;
               guarded += base && base->op == nir_op_bcsel &&
                          nir_alu_src_as_uint(base->src[2]) ==
                             AGX_ZERO_PAGE_ADDRESS;
            } else {
               main_loads++;
               main_speculatable |=
                  (nir_intrinsic_access(I) & ACCESS_CAN_SPECULATE) != 0;
            }
         }
      }
   }
   EXPECT_EQ(preamble_loads, 1u);
   EXPECT_EQ(guarded, 0u);
   EXPECT_EQ(main_loads, 1u);
   EXPECT_TRUE(main_speculatable);
   ralloc_free(b.shader);
}

/* Promoting a non-speculatable conditional buffer load must not strip
 * ACCESS_CAN_SPECULATE from unrelated main-shader loads that already had it. */
TEST(PreambleSpeculativeBuffer, PreservesPreexistingMainSpeculateFlag)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "preserve main speculate");

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *desc = nir_load_constant_agx(
      &b, 4, 32, agx_test_load_preamble(&b, 1, 64, 0), zero,
      .access = VALID_DESCRIPTOR, .base = 0, .format = PIPE_FORMAT_R32_UINT);
   nir_def *addr = nir_pack_64_2x32_split(&b, nir_channel(&b, desc, 0),
                                          nir_channel(&b, desc, 1));

   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *elsewhere = nir_imm_vec4(&b, 0, 0, 0, 0);
   nir_push_if(&b, nir_ine_imm(&b, lid, 0));
   nir_def *promoted = nir_load_constant_agx(
      &b, 4, 32, addr, nir_imm_int(&b, 2), .access = BUFFER_READ,
      .base = 2, .format = PIPE_FORMAT_R32_UINT);
   nir_def *preexisting = nir_load_constant_agx(
      &b, 4, 32, addr, lid,
      .access = (gl_access_qualifier)(BUFFER_READ | ACCESS_CAN_SPECULATE),
      .base = 2, .format = PIPE_FORMAT_R32_UINT);
   nir_def *sum = nir_iadd(&b, promoted, preexisting);
   nir_pop_if(&b, NULL);
   nir_def *val = nir_if_phi(&b, sum, elsewhere);
   nir_store_global(&b, val, nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 16);

   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   unsigned sizes[] = {4, 0, 0};
   agx_nir_opt_preamble(b.shader, sizes, true, false, NULL);
   nir_validate_shader(b.shader, "after mixed speculate preamble");

   unsigned main_loads = 0, preamble_loads = 0, guarded = 0;
   bool main_speculatable = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (I->intrinsic != nir_intrinsic_load_constant_agx ||
                !(nir_intrinsic_access(I) & ACCESS_CAN_REORDER))
               continue;
            if (impl->function->is_preamble) {
               nir_alu_instr *base = nir_src_as_alu(I->src[0]);
               preamble_loads++;
               guarded += base && base->op == nir_op_bcsel &&
                          nir_alu_src_as_uint(base->src[2]) ==
                             AGX_ZERO_PAGE_ADDRESS;
            } else {
               main_loads++;
               main_speculatable |=
                  (nir_intrinsic_access(I) & ACCESS_CAN_SPECULATE) != 0;
            }
         }
      }
   }
   EXPECT_EQ(preamble_loads, 1u);
   EXPECT_EQ(guarded, 1u);
   EXPECT_EQ(main_loads, 1u);
   EXPECT_TRUE(main_speculatable);
   ralloc_free(b.shader);
}
