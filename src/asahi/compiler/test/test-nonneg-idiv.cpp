/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_range_analysis.h"
#include "util/hash_table.h"

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

/* A divisor agx_nir_opt_uniform_idiv cannot move to the preamble. */
static nir_def *
divergent_divisor(nir_builder *b)
{
   return nir_load_global(b, 1, 32, nir_imm_int64(b, 0x2000), .align_mul = 4);
}

TEST(NonnegIdiv, BothOperandsNonnegRewriteToUnsigned)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nonneg-both");
   nir_def *raw_a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *raw_b = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
   nir_def *a = nir_iand_imm(&b, raw_a, 0x7fffffff);
   nir_def *d = nir_imax(&b, raw_b, nir_imm_int(&b, 1));

   nir_def *q = nir_idiv(&b, a, d);
   nir_def *m = nir_imod(&b, a, d);
   nir_def *r = nir_irem(&b, a, d);
   nir_store_global(&b, nir_iadd(&b, nir_iadd(&b, q, m), r),
                    nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_nonneg_idiv(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "nonneg-both");

   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_irem), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 2u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_iabs), 0u);
   ralloc_free(b.shader);
}

TEST(NonnegIdiv, BoundedGlobalIdPlusBase)
{
   nir_shader_compiler_options opts = agx_nir_options;
   opts.max_workgroup_count[0] = 0x10000000u; /* * 32 > INT32_MAX: unbounded */
   opts.max_workgroup_count[1] = 65535u;      /* * 4 <= INT32_MAX: bounded */
   opts.max_workgroup_count[2] = 65535u;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &opts, "nonneg-gid-base");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 4;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *base = nir_load_base_global_invocation_id(&b, 32);
   nir_def *d = divergent_divisor(&b);
   nir_def *x = nir_iadd(&b, nir_channel(&b, gid, 0), nir_channel(&b, base, 0));
   nir_def *y = nir_iadd(&b, nir_channel(&b, gid, 1), nir_channel(&b, base, 1));

   nir_def *qx = nir_idiv(&b, x, d);
   nir_def *qy = nir_idiv(&b, y, d);
   nir_store_global(&b, nir_iadd(&b, qx, qy), nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_nonneg_idiv(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "nonneg-gid-base");

   /* Y is provably nonnegative so qy lowers to udiv; X exceeds INT32_MAX so qx
    * stays signed idiv. */
   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 1u);
   ralloc_free(b.shader);
}

TEST(NonnegIdiv, OddWorkgroupSizeGlobalIdOverflowNotStrengthReduced)
{
   /* With agx_nir_options.max_workgroup_count[0] == 0x7fffffff and odd
    * workgroup_size[0] == 3, an unclamped 32-bit multiply would wrap
    * 3u * 0x7fffffffu - 1u to 0x7ffffffcu <= INT32_MAX and miscompile signed
    * div/mod on gid.x. Verify gid.x saturates to UINT32_MAX and is NOT
    * strength-reduced, while gid.y and gid.z (3 * 65535 - 1 <= INT32_MAX) ARE. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nonneg-gid-odd-wg");
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = 3;
   b.shader->info.workgroup_size[2] = 3;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);
   nir_def *gz = nir_channel(&b, gid, 2);

   struct hash_table *range_ht = _mesa_pointer_hash_table_create(NULL);
   EXPECT_EQ(nir_unsigned_upper_bound(b.shader, range_ht, nir_get_scalar(gid, 0)),
             UINT32_MAX);
   EXPECT_EQ(nir_unsigned_upper_bound(b.shader, range_ht, nir_get_scalar(gid, 1)),
             3u * 65535u - 1u);
   EXPECT_EQ(nir_unsigned_upper_bound(b.shader, range_ht, nir_get_scalar(gid, 2)),
             3u * 65535u - 1u);
   _mesa_hash_table_destroy(range_ht, NULL);

   nir_def *d = divergent_divisor(&b);
   nir_def *c7 = nir_imm_int(&b, 7);

   nir_def *acc = nir_iadd(&b, nir_idiv(&b, gx, d), nir_imod(&b, gx, d));
   acc = nir_iadd(&b, acc, nir_irem(&b, gx, d));
   acc = nir_iadd(&b, acc, nir_idiv(&b, gx, c7));

   acc = nir_iadd(&b, acc, nir_idiv(&b, gy, d));
   acc = nir_iadd(&b, acc, nir_imod(&b, gy, d));
   acc = nir_iadd(&b, acc, nir_irem(&b, gy, d));
   acc = nir_iadd(&b, acc, nir_idiv(&b, gy, c7));

   acc = nir_iadd(&b, acc, nir_idiv(&b, gz, d));
   acc = nir_iadd(&b, acc, nir_imod(&b, gz, d));
   acc = nir_iadd(&b, acc, nir_irem(&b, gz, d));
   acc = nir_iadd(&b, acc, nir_idiv(&b, gz, c7));

   nir_store_global(&b, acc, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_nonneg_idiv(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "nonneg-gid-odd-wg");

   /* gid.x ops stay signed (idiv x2, imod x1, irem x1); gid.y & gid.z ops
    * strength-reduce (udiv x4, umod x4). */
   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 2u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_irem), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 4u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 4u);
   ralloc_free(b.shader);
}

TEST(NonnegIdiv, DominatingBranchGuardEnablesRewrite)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nonneg-branch");
   nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *d = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);

   nir_def *guard = nir_iand(&b, nir_ige_imm(&b, a, 0),
                             nir_ult_imm(&b, d, 0x80000000u));
   nir_if *nif = nir_push_if(&b, guard);
   {
      nir_def *q = nir_idiv(&b, a, d);
      nir_def *r = nir_irem(&b, a, d);
      nir_store_global(&b, nir_iadd(&b, q, r), nir_imm_int64(&b, 0), .align_mul = 4);
   }
   nir_pop_if(&b, nif);

   EXPECT_TRUE(agx_nir_opt_nonneg_idiv(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "nonneg-branch");

   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_irem), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_iabs), 0u);
   ralloc_free(b.shader);
}

TEST(NonnegIdiv, UnknownDividendConstantDivisorKeptForIdivConst)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nonneg-const-div");
   nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *q = nir_idiv(&b, a, nir_imm_int(&b, 7));
   nir_store_global(&b, q, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_FALSE(agx_nir_opt_nonneg_idiv(b.shader));
   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 1u);
   nir_validate_shader(b.shader, "nonneg-const-div");
   ralloc_free(b.shader);
}

TEST(NonnegIdiv, SingleNonnegOperandDropRedundantIabs)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nonneg-single");
   nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *d = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
   nir_def *nonneg_a = nir_iand_imm(&b, a, 0x7fffffff);
   nir_def *nonneg_d = nir_imax(&b, d, nir_imm_int(&b, 1));

   nir_def *q1 = nir_idiv(&b, nonneg_a, d);
   nir_def *m1 = nir_imod(&b, nonneg_a, d);
   nir_def *r1 = nir_irem(&b, nonneg_a, d);
   nir_def *q2 = nir_idiv(&b, a, nonneg_d);
   nir_def *m2 = nir_imod(&b, a, nonneg_d);
   nir_def *r2 = nir_irem(&b, a, nonneg_d);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, nir_iadd(&b, q1, m1), r1),
                           nir_iadd(&b, nir_iadd(&b, q2, m2), r2));
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_nonneg_idiv(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "nonneg-single");

   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_irem), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 2u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 4u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imul), 0u);

   /* Verify bit-exactness of the lowered identities across edge inputs
    * including INT32_MIN (where two's-complement |INT32_MIN| == 0x80000000u). */
   static const int32_t vals[] = {INT32_MIN, -1000003, -7, -1, 0, 1, 7, 1000003, INT32_MAX};
   for (int32_t va : vals) {
      for (int32_t vd : vals) {
         if (vd == 0)
            continue;
         if (va >= 0) {
            uint32_t ua = (uint32_t)va;
            uint32_t ad = vd < 0 ? (uint32_t)(-(int64_t)vd) : (uint32_t)vd;
            int32_t q = (int32_t)(ua / ad);
            int32_t got_q = vd < 0 ? -q : q;
            ASSERT_EQ(got_q, va / vd);
            int32_t r = (int32_t)(ua % ad);
            ASSERT_EQ(r, va % vd);
            int32_t ref_mod = (va % vd == 0 || (va >= 0 && vd > 0)) ? (va % vd) : (va % vd) + vd;
            int32_t got_mod = r == 0 ? 0 : (int32_t)((uint32_t)r + (uint32_t)(vd < 0 ? vd : 0));
            ASSERT_EQ(got_mod, ref_mod);
         }
         if (vd > 0) {
            uint32_t aa = va < 0 ? (uint32_t)(-(int64_t)va) : (uint32_t)va;
            uint32_t ud = (uint32_t)vd;
            uint32_t q = aa / ud;
            uint32_t r = aa % ud;
            int32_t got_q = va < 0 ? (int32_t)(0u - q) : (int32_t)q;
            int32_t got_r = va < 0 ? (int32_t)(0u - r) : (int32_t)r;
            if (!(va == INT32_MIN && vd == -1)) {
               ASSERT_EQ(got_q, va / vd);
               ASSERT_EQ(got_r, va % vd);
            }
            int32_t ref_mod = (int32_t)(((int64_t)va % vd + vd) % vd);
            int32_t got_mod = va < 0 ? (r == 0 ? 0 : (int32_t)(ud - r)) : (int32_t)r;
            ASSERT_EQ(got_mod, ref_mod);
         }
      }
   }
   ralloc_free(b.shader);
}

/* agx_nir_opt_uniform_idiv lowers a signed division by a uniform divisor as an
 * unsigned division of the magnitudes plus sign fixups, so the one-sided
 * rewrites apply to uniform divisors as to divergent ones. */
TEST(NonnegIdiv, UniformDivisorRewrites)
{
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nonneg-uniform-div");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *pc = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *pc_nonneg = nir_iand_imm(&b, pc, 0x7fffffff);
   nir_variable *var =
      nir_variable_create(b.shader, nir_var_mem_push_const, glsl_uint_type(), "pc");
   nir_def *pc_deref = nir_load_deref(&b, nir_build_deref_var(&b, var));
   nir_def *d = divergent_divisor(&b);

   nir_def *sum = nir_idiv(&b, lid, pc_nonneg); /* udiv */
   sum = nir_iadd(&b, sum, nir_idiv(&b, lid, pc));        /* udiv + fixup */
   sum = nir_iadd(&b, sum, nir_idiv(&b, lid, pc_deref));  /* udiv + fixup */
   sum = nir_iadd(&b, sum, nir_imod(&b, lid, pc));        /* umod + fixups */
   sum = nir_iadd(&b, sum, nir_irem(&b, lid, pc));        /* umod */
   sum = nir_iadd(&b, sum, nir_idiv(&b, lid, d));         /* udiv + fixup */
   nir_store_global(&b, sum, nir_imm_int64(&b, 0), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_nonneg_idiv(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "nonneg-uniform-div");

   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_irem), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 4u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 2u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_bcsel), 5u);
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
}
