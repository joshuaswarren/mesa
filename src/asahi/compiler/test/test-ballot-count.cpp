/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

TEST(BallotCount, PreserveMembershipAcrossDemoteOnM5)
{
   for (bool m5 : {false, true}) {
      SCOPED_TRACE(m5);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "ballot count across demote");
      nir_def *before = nir_bit_count(&b, nir_ballot(&b, 1, 32, nir_imm_true(&b)));
      nir_demote_if(&b, nir_ieq_imm(&b, nir_load_subgroup_invocation(&b), 0));
      nir_def *after = nir_bit_count(&b, nir_ballot(&b, 1, 32, nir_imm_true(&b)));
      nir_store_global(&b, nir_isub(&b, before, after), nir_imm_int64(&b, 0x1000));

      if (m5)
         agx_nir_lower_subgroups_m5(b.shader, true, true);
      else
         agx_nir_lower_subgroups(b.shader);
      nir_opt_dce(b.shader);

      unsigned ballots = 0, counts = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            ballots += op == nir_intrinsic_ballot;
            counts += op == nir_intrinsic_load_active_subgroup_count_agx;
         }
      }
      EXPECT_EQ(ballots, m5 ? 2u : 0u);
      EXPECT_EQ(counts, m5 ? 0u : 2u);
      nir_validate_shader(b.shader, "ballot count across demote");
      ralloc_free(b.shader);
   }
}
