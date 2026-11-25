/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
#include "agx_compiler.h"
}

static inline nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static inline nir_def *
make_buffer_addr(nir_builder *b, nir_def *base_u64, nir_def *byte_offset)
{
   return nir_iadd(b, base_u64, nir_u2u64(b, byte_offset));
}

static inline nir_def *
make_desc_base(nir_builder *b, int preamble_base)
{
   nir_def *set = nir_load_preamble(b, 1, 64, .base = preamble_base);
   nir_def *desc = nir_load_global_constant(
      b, 4, 32, set,
      .access = (gl_access_qualifier)ACCESS_CAN_SPECULATE,
      .align_mul = 16);
   return nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                 nir_channel(b, desc, 1));
}

static inline unsigned
count_intrinsics(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      }
   }
   return n;
}

static inline unsigned
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

static inline unsigned
count_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(block->cf_node.parent);
            if (block == nir_if_first_then_block(nif))
               n++;
         }
      }
   }
   return n;
}

TEST(FlrpContract, FlrpContractionRespectsExact)
{
   /* Inexact fast-math ffma(1.0 - t, a, t * b) contracts to ffma(t, b - a, a),
    * while exact or SignedZeroInfNanPreserve ops are preserved untouched. */
   const unsigned ctrls[4] = {
      nir_fp_fast_math,
      nir_fp_exact,
      nir_fp_preserve_signed_zero,
      nir_fp_preserve_nan | nir_fp_preserve_inf,
   };
   const char *names[4] = {
      "flrp-fast-math",
      "flrp-exact",
      "flrp-preserve-sz",
      "flrp-preserve-nan-inf",
   };
   for (unsigned mode = 0; mode < 4; ++mode) {
      nir_builder b = make_compute_shader(names[mode]);
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *a = nir_load_global(&b, 1, 32, base, .align_mul = 4);
      nir_def *bv = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, base, 4), .align_mul = 4);
      nir_def *t = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, base, 8), .align_mul = 4);

      b.fp_math_ctrl = ctrls[mode];
      nir_def *one_minus_t = nir_fadd(&b, nir_imm_float(&b, 1.0f), nir_fneg(&b, t));
      nir_def *tb = nir_fmul(&b, t, bv);
      nir_def *lerp = nir_ffma(&b, one_minus_t, a, tb);
      b.fp_math_ctrl = nir_fp_fast_math;

      nir_store_global(&b, lerp, out, .align_mul = 4);
      nir_validate_shader(b.shader, "flrp setup");

      bool changed = agx_nir_fuse_algebraic_late(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "flrp lowered");
      if (mode == 0) {
         EXPECT_TRUE(changed);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_fmul), 0u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_ffma), 1u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_fmul), 1u);
      }
      ralloc_free(b.shader);
   }
}
