/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
}

enum rhs_modifier { RHS_ABS, RHS_NEG };

/* ballot(x < |y|) or ballot(x < -y) of per-lane floats in a compute shader.
 * The fabs or fneg folds into the comparison as a source modifier, which
 * the M5 ballot compare cannot encode. */
static bool
m5_ballot_of_modified_comparison_compiles(enum rhs_modifier modifier)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "M5 ballot of a modified compare");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *params = agx_test_load_preamble(&b, 2, 32, 0);
   nir_def *out = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *lane_f = nir_u2f32(&b, lane);
   nir_def *x = nir_fmul(&b, lane_f, nir_channel(&b, params, 0));
   nir_def *y = nir_fsub(&b, lane_f, nir_channel(&b, params, 1));
   nir_def *rhs = modifier == RHS_ABS ? nir_fabs(&b, y) : nir_fneg(&b, y);
   nir_def *mask = nir_ballot(&b, 1, 32, nir_flt(&b, x, rhs));
   nir_intrinsic_instr *store = nir_store_global(
      &b, mask, nir_iadd(&b, out, nir_u2u64(&b, nir_imul_imm(&b, lane, 4))));
   nir_intrinsic_set_align(store, 4, 0);

   agx_preprocess_nir_m5(b.shader);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.reserved_preamble = 8;

   agx_shader_part part = {};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   if (!ok)
      ADD_FAILURE() << agx_last_compile_refusal();
   else
      EXPECT_NE(part.binary, nullptr);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

TEST(AGX3BallotModifiers, M5BallotOfAbsoluteComparisonPacks)
{
   EXPECT_TRUE(m5_ballot_of_modified_comparison_compiles(RHS_ABS));
}

TEST(AGX3BallotModifiers, M5BallotOfNegatedComparisonPacks)
{
   EXPECT_TRUE(m5_ballot_of_modified_comparison_compiles(RHS_NEG));
}
