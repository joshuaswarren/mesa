/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

static void
consume(nir_builder *b, nir_def *v)
{
   nir_store_global(b, v, nir_imm_int64(b, 0x1000));
}

class PreambleHalfExports : public testing::Test {
 protected:
   PreambleHalfExports()
   {
      glsl_type_singleton_init_or_ref();
   }
   ~PreambleHalfExports()
   {
      glsl_type_singleton_decref();
   }
};

TEST_F(PreambleHalfExports, TakeAWholeWord)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "half");
   for (unsigned i = 0; i < 3; ++i)
      consume(&b, nir_load_ubo(&b, 1, 16, nir_imm_int(&b, 0),
                               nir_imm_int(&b, 2 * i), .align_mul = 2,
                               .range = 1 << 16));

   unsigned sizes[] = {1, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, false, NULL));

   unsigned stores = 0;
   nir_foreach_block(block, nir_shader_get_preamble(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble)
            continue;
         stores++;
         EXPECT_EQ(intr->src[0].ssa->bit_size, 32u);
         EXPECT_EQ(nir_intrinsic_base(intr) % 2, 0u);
      }
   }
   EXPECT_EQ(stores, 3u);
   EXPECT_EQ(sizes[0], 8u);
   ralloc_free(b.shader);
}
