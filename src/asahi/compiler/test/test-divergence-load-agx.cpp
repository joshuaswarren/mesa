/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

namespace {

nir_def *
load(nir_builder *b, nir_def *base, nir_def *index)
{
   return nir_load_agx(b, 1, 32, base, index,
                       .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                                             ACCESS_CAN_REORDER),
                       .base = 0, .format = PIPE_FORMAT_R32_UINT);
}

unsigned
count(nir_shader *s, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         n += instr->type == nir_instr_type_intrinsic &&
              nir_instr_as_intrinsic(instr)->intrinsic == op;
      }
   }
   return n;
}

} // namespace

/* load_agx, the lowered form of load_global, is divergent exactly when one of
 * its address sources is. */
TEST(DivergenceLoadAgx, FollowsTheAddress)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "load_agx divergence");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *lane = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *uniform = load(&b, base, nir_imm_int(&b, 3));
   nir_def *varying = load(&b, base, lane);
   nir_def *varying_base =
      load(&b, nir_iadd(&b, base, nir_u2u64(&b, lane)), nir_imm_int(&b, 3));
   nir_def *out = nir_imm_int64(&b, 0x100000000000ull);
   nir_store_global(&b, nir_iadd(&b, uniform, nir_iadd(&b, varying, varying_base)),
                    out);

   nir_divergence_analysis(b.shader);
   EXPECT_FALSE(uniform->divergent);
   EXPECT_TRUE(varying->divergent);
   EXPECT_TRUE(varying_base->divergent);
   ralloc_free(b.shader);
}

/* A translated readfirstlane of a value loaded from a uniform address: the
 * first active lane's copy is the value itself. */
TEST(DivergenceLoadAgx, ReadFirstLaneOfAUniformLoadFolds)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "readfirstlane of load_agx");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *x = load(&b, base, nir_imm_int(&b, 5));
   nir_def *ballot = nir_ballot(&b, 1, 32, nir_imm_true(&b));
   nir_def *first = nir_ufind_msb(&b, nir_iand(&b, ballot, nir_ineg(&b, ballot)));
   nir_def *read = nir_read_invocation(&b, x, first);
   nir_store_global(&b, read, nir_imm_int64(&b, 0x100000000000ull));

   static const nir_lower_subgroups_options options = {
      .ballot_bit_size = 32,
      .ballot_components = 1,
   };
   ASSERT_TRUE(nir_opt_uniform_subgroup(b.shader, &options));
   EXPECT_EQ(count(b.shader, nir_intrinsic_read_invocation), 0u);
   ralloc_free(b.shader);
}
