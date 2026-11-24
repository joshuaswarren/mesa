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

static unsigned
count_alu_op(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

TEST(FactorConstMul, SymmetricGaussian32)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "factor-gaussian-32");

   nir_def *vals[5];
   for (unsigned i = 0; i < 5; ++i)
      vals[i] = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, i * 4),
                                       .range = 4);

   /* Symmetric 5-tap weights: 16, 86, 342, 86, 16 plus rounding bias 32768. */
   nir_def *acc = nir_imm_int(&b, 32768);
   acc = nir_iadd(&b, acc, nir_ishl_imm(&b, vals[0], 4));
   acc = nir_iadd(&b, acc, nir_imul_imm(&b, vals[1], 86));
   acc = nir_iadd(&b, acc, nir_imul_imm(&b, vals[2], 342));
   acc = nir_iadd(&b, acc, nir_imul_imm(&b, vals[3], 86));
   acc = nir_iadd(&b, acc, nir_ishl_imm(&b, vals[4], 4));
   nir_store_global(&b, acc, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 3u);
   EXPECT_TRUE(agx_nir_opt_factor_const_mul(b.shader));
   nir_opt_dce(b.shader);

   /* 86*v1 + 86*v3 factors to (v1 + v3)*86 -> 2 multiplies instead of 3. */
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 2u);
   /* 16*v0 + 16*v4 also factors into a single shift-by-4. */
   EXPECT_EQ(count_alu_op(b.shader, nir_op_ishl), 1u);

   nir_validate_shader(b.shader, "factor-gaussian-32");
   ralloc_free(b.shader);
}

TEST(FactorConstMul, Symmetric16Bit)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "factor-16bit");

   nir_def *a = nir_u2u16(&b, nir_load_push_constant(&b, 1, 32,
                                                     nir_imm_int(&b, 0),
                                                     .range = 4));
   nir_def *c = nir_u2u16(&b, nir_load_push_constant(&b, 1, 32,
                                                     nir_imm_int(&b, 4),
                                                     .range = 4));
   nir_def *sum = nir_iadd(&b, nir_imul(&b, a, nir_imm_intN_t(&b, 1052, 16)),
                           nir_imul(&b, c, nir_imm_intN_t(&b, 1052, 16)));
   nir_store_global(&b, nir_u2u32(&b, sum), nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_factor_const_mul(b.shader));
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 1u);

   nir_validate_shader(b.shader, "factor-16bit");
   ralloc_free(b.shader);
}

TEST(FactorConstMul, SkipsSmallPow2Only)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "skip-small-pow2");

   nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
   nir_def *sum = nir_iadd(&b, nir_ishl_imm(&b, a, 2), nir_ishl_imm(&b, c, 2));
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   /* Small power-of-two shifts (1..4) already fold into single-cycle shifted iadd. */
   EXPECT_FALSE(agx_nir_opt_factor_const_mul(b.shader));

   nir_validate_shader(b.shader, "skip-small-pow2");
   ralloc_free(b.shader);
}

TEST(FactorConstMul, SkipsMultiUseProduct)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "skip-multi-use");

   nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
   nir_def *pa = nir_imul_imm(&b, a, 342);
   nir_def *pc = nir_imul_imm(&b, c, 342);
   nir_def *sum = nir_iadd(&b, pa, pc);
   nir_store_global(&b, pa, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_store_global(&b, sum, nir_imm_int64(&b, 4), .align_mul = 4);

   /* pa has another consumer so factoring would not eliminate its multiply. */
   EXPECT_FALSE(agx_nir_opt_factor_const_mul(b.shader));

   nir_validate_shader(b.shader, "skip-multi-use");
   ralloc_free(b.shader);
}

TEST(FactorConstMul, LargeShiftFactors)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "factor-large-shift");

   nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
   nir_def *sum = nir_iadd(&b, nir_ishl_imm(&b, a, 8), nir_ishl_imm(&b, c, 8));
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_factor_const_mul(b.shader));
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_ishl), 1u);

   nir_validate_shader(b.shader, "factor-large-shift");
   ralloc_free(b.shader);
}

static uint32_t
eval_push_const_u32(const nir_shader *orig, const uint32_t *inputs,
                    unsigned num_inputs)
{
   nir_shader *s = nir_shader_clone(NULL, orig);
   nir_foreach_function_impl(impl, s) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_push_constant)
               continue;
            unsigned idx = nir_src_as_uint(intr->src[0]) / 4;
            EXPECT_LT(idx, num_inputs);
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def, nir_imm_int(&b, inputs[idx]));
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

   uint32_t out = 0;
   bool found = false;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_global)
               continue;
            EXPECT_TRUE(nir_src_is_const(intr->src[0]));
            out = nir_src_as_uint(intr->src[0]);
            found = true;
         }
      }
   }
   EXPECT_TRUE(found);
   ralloc_free(s);
   return out;
}

TEST(FactorConstMul, SkipsMultiUseMovAndVecChains)
{
   /* 1. Leaf imul reached through a multi-use mov must not be duplicated. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "skip-multi-use-mov-leaf");
      nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
      nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
      nir_def *pa = nir_imul_imm(&b, a, 342);
      nir_def *mv = nir_mov(&b, pa);
      nir_def *pc = nir_imul_imm(&b, c, 342);
      nir_def *sum = nir_iadd(&b, mv, pc);
      nir_store_global(&b, mv, nir_imm_int64(&b, 0), .align_mul = 4);
      nir_store_global(&b, sum, nir_imm_int64(&b, 4), .align_mul = 4);

      EXPECT_FALSE(agx_nir_opt_factor_const_mul(b.shader));
      EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 2u);
      nir_validate_shader(b.shader, "skip-multi-use-mov-leaf");
      ralloc_free(b.shader);
   }

   /* 2. Leaf imul reached through a vec2 must not be duplicated. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "skip-vec-leaf");
      nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
      nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
      nir_def *pa = nir_imul_imm(&b, a, 342);
      nir_def *vec = nir_vec2(&b, pa, a);
      nir_def *pc = nir_imul_imm(&b, c, 342);
      nir_def *sum = nir_iadd(&b, nir_channel(&b, vec, 0), pc);
      nir_store_global(&b, vec, nir_imm_int64(&b, 0), .align_mul = 8);
      nir_store_global(&b, sum, nir_imm_int64(&b, 8), .align_mul = 4);

      EXPECT_FALSE(agx_nir_opt_factor_const_mul(b.shader));
      EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 2u);
      nir_validate_shader(b.shader, "skip-vec-leaf");
      ralloc_free(b.shader);
   }

   /* 3. Interior iadd reached through a multi-use mov must not be flattened. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "skip-multi-use-mov-iadd");
      nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
      nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 4);
      nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 4);
      nir_def *inner = nir_iadd(&b, nir_imul_imm(&b, a, 342),
                                nir_imul_imm(&b, x, 86));
      nir_def *mv = nir_mov(&b, inner);
      nir_def *outer = nir_iadd(&b, mv, nir_imul_imm(&b, c, 342));
      nir_store_global(&b, mv, nir_imm_int64(&b, 0), .align_mul = 4);
      nir_store_global(&b, outer, nir_imm_int64(&b, 4), .align_mul = 4);

      EXPECT_FALSE(agx_nir_opt_factor_const_mul(b.shader));
      EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 3u);
      nir_validate_shader(b.shader, "skip-multi-use-mov-iadd");
      ralloc_free(b.shader);
   }
}

TEST(FactorConstMul, ValueEquivalence16BitWrap)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "factor-16bit-wrap-val");

   nir_def *a = nir_u2u16(&b, nir_load_push_constant(&b, 1, 32,
                                                     nir_imm_int(&b, 0),
                                                     .range = 4));
   nir_def *c = nir_u2u16(&b, nir_load_push_constant(&b, 1, 32,
                                                     nir_imm_int(&b, 4),
                                                     .range = 4));
   nir_def *sum = nir_iadd(&b, nir_imul(&b, a, nir_imm_intN_t(&b, 0xffff, 16)),
                           nir_imul(&b, c, nir_imm_intN_t(&b, 0xffff, 16)));
   nir_store_global(&b, nir_u2u32(&b, sum), nir_imm_int64(&b, 0), .align_mul = 4);

   nir_shader *orig = nir_shader_clone(NULL, b.shader);
   EXPECT_TRUE(agx_nir_opt_factor_const_mul(b.shader));
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 1u);

   const uint32_t cases[][2] = {
      {0x8000u, 0x8000u},
      {0xffffu, 0xffffu},
      {0x7fffu, 0x0001u},
      {0x1234u, 0xabcdu},
      {0x0000u, 0x8000u},
   };
   for (const auto &v : cases) {
      uint32_t expected = eval_push_const_u32(orig, v, 2);
      uint32_t actual = eval_push_const_u32(b.shader, v, 2);
      uint16_t ref = (uint16_t)((uint16_t)(v[0] * 0xffffu) +
                                (uint16_t)(v[1] * 0xffffu));
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(actual, (uint32_t)ref);
   }

   nir_validate_shader(b.shader, "factor-16bit-wrap-val");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

TEST(FactorConstMul, ValueEquivalenceIshl33MergesWithImul2)
{
   /* ishl(x, 33) masks the shift count to 33 & 31 = 1 (factor 2) and must
    * merge with imul(y, 2) in the same tree when another group makes the tree
    * profitable. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "factor-ishl33-imul2");

   nir_def *vals[4];
   for (unsigned i = 0; i < 4; ++i)
      vals[i] = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, i * 4),
                                       .range = 4);

   nir_def *t0 = nir_ishl(&b, vals[0], nir_imm_int(&b, 33));
   nir_def *t1 = nir_imul_imm(&b, vals[1], 2);
   nir_def *t2 = nir_imul_imm(&b, vals[2], 86);
   nir_def *t3 = nir_imul_imm(&b, vals[3], 86);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, t0, t1), nir_iadd(&b, t2, t3));
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   nir_shader *orig = nir_shader_clone(NULL, b.shader);
   EXPECT_TRUE(agx_nir_opt_factor_const_mul(b.shader));
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_ishl), 1u);

   const uint32_t cases[][4] = {
      {1u, 2u, 3u, 4u},
      {0x80000001u, 0x7fffffffu, 0xffffffffu, 0x12345678u},
      {0xffffffffu, 0xffffffffu, 0x80000000u, 0x80000000u},
   };
   for (const auto &v : cases) {
      uint32_t expected = eval_push_const_u32(orig, v, 4);
      uint32_t actual = eval_push_const_u32(b.shader, v, 4);
      EXPECT_EQ(actual, expected);
   }

   nir_validate_shader(b.shader, "factor-ishl33-imul2");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

TEST(FactorConstMul, ValueEquivalenceSymmetricGaussian32)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "factor-gaussian-32-val");

   nir_def *vals[5];
   for (unsigned i = 0; i < 5; ++i)
      vals[i] = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, i * 4),
                                       .range = 4);

   nir_def *acc = nir_imm_int(&b, 32768);
   acc = nir_iadd(&b, acc, nir_ishl_imm(&b, vals[0], 4));
   acc = nir_iadd(&b, acc, nir_imul_imm(&b, vals[1], 86));
   acc = nir_iadd(&b, acc, nir_imul_imm(&b, vals[2], 342));
   acc = nir_iadd(&b, acc, nir_imul_imm(&b, vals[3], 86));
   acc = nir_iadd(&b, acc, nir_ishl_imm(&b, vals[4], 4));
   nir_store_global(&b, acc, nir_imm_int64(&b, 0), .align_mul = 4);

   nir_shader *orig = nir_shader_clone(NULL, b.shader);
   EXPECT_TRUE(agx_nir_opt_factor_const_mul(b.shader));
   nir_opt_dce(b.shader);

   const uint32_t cases[][5] = {
      {0u, 0u, 0u, 0u, 0u},
      {255u, 255u, 255u, 255u, 255u},
      {0x80000000u, 0xffffffffu, 0x12345678u, 0x7fffffffu, 0xfedcba98u},
   };
   for (const auto &v : cases) {
      EXPECT_EQ(eval_push_const_u32(b.shader, v, 5),
                eval_push_const_u32(orig, v, 5));
   }

   nir_validate_shader(b.shader, "factor-gaussian-32-val");
   ralloc_free(orig);
   ralloc_free(b.shader);
}
