/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "nir_test.h"

class nir_opt_sink_test : public nir_test {
protected:
   nir_opt_sink_test()
      : nir_test::nir_test("nir_opt_sink_test", MESA_SHADER_COMPUTE)
   {
   }
};

TEST_F(nir_opt_sink_test, single_block_no_progress)
{
   nir_def *addr = nir_undef(b, 1, 64);
   nir_def *a = nir_load_global(b, 1, 32, addr);
   nir_def *c = nir_imm_int(b, 3);
   nir_store_global(b, nir_iadd(b, a, c), addr);

   nir_metadata_require(b->impl, nir_metadata_instr_index);
   ASSERT_FALSE(nir_opt_sink(
      b->shader, (nir_move_options)(nir_move_const_undef | nir_move_alu)));
   ASSERT_TRUE(b->impl->valid_metadata & nir_metadata_instr_index);
   nir_validate_shader(b->shader, NULL);
}
