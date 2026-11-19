/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* Uniform slots go to what every fragment runs before what an if may skip. */
static nir_def *
ubo_load(nir_builder *b, unsigned comps, unsigned offset)
{
   return nir_load_ubo(b, comps, 32, nir_imm_int(b, 0), nir_imm_int(b, offset),
                       .align_mul = 4, .range = 1 << 16);
}

static void
consume(nir_builder *b, nir_def *v)
{
   nir_store_global(b, v, nir_imm_int64(b, 0x1000));
}

static unsigned
main_loads(nir_shader *s, unsigned comps)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         n += intr->intrinsic == nir_intrinsic_load_ubo &&
              intr->def.num_components == comps;
      }
   }
   return n;
}

class PreamblePriority : public testing::Test {
 protected:
   PreamblePriority()
   {
      glsl_type_singleton_init_or_ref();
   }
   ~PreamblePriority()
   {
      glsl_type_singleton_decref();
   }
};

static nir_shader *
cold_branch_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "cold");
   nir_push_if(&b, nir_ine_imm(&b, ubo_load(&b, 1, 0), 0));
   /* Scalars alone overflow the 480 halves, and cost less per half. */
   for (unsigned i = 0; i < 300; ++i)
      consume(&b, ubo_load(&b, 1, 4096 + 4 * i));
   nir_pop_if(&b, NULL);
   for (unsigned i = 0; i < 8; ++i)
      consume(&b, ubo_load(&b, 4, 16 * (i + 1)));
   return b.shader;
}

/* agx_optimize_nir's nir_opt_peephole_select after the constant program. */
static const nir_opt_peephole_select_options flatten = {
   .limit = 64,
   .expensive_alu_ok = true,
};

TEST_F(PreamblePriority, EveryFragmentBeatsAColdBranch)
{
   nir_shader *s = cold_branch_shader();
   unsigned sizes[] = {0, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(s, sizes, true, false, &flatten));
   EXPECT_EQ(main_loads(s, 4), 0u);
   EXPECT_GT(main_loads(s, 1), 0u);
   ralloc_free(s);
}

/* Only AGX3B weights: AGX3C and the older encodings keep the plain order. */
TEST_F(PreamblePriority, UnweightedOffAGX3B)
{
   const bool agx3[] = {false, true};
   const bool agx3c[] = {false, true};
   for (unsigned i = 0; i < 2; ++i) {
      nir_shader *s = cold_branch_shader();
      unsigned sizes[] = {0, 1000, 1000};
      ASSERT_TRUE(agx_nir_opt_preamble(s, sizes, agx3[i], agx3c[i], &flatten));
      EXPECT_GT(main_loads(s, 4), 0u) << "agx3c " << agx3c[i];
      ralloc_free(s);
   }
}

/* An if whose arms become selects once their loads move runs for every
 * fragment, so its loads keep their full value. */
static nir_shader *
select_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "select");
   nir_def *zero = nir_imm_int(&b, 0), *loads[100];
   nir_push_if(&b, nir_ine_imm(&b, ubo_load(&b, 1, 0), 0));
   for (unsigned i = 0; i < 100; ++i)
      loads[i] = ubo_load(&b, 1, 4096 + 4 * i);
   nir_pop_if(&b, NULL);
   for (unsigned i = 0; i < 100; ++i)
      loads[i] = nir_if_phi(&b, loads[i], zero);
   for (unsigned i = 0; i < 100; ++i)
      consume(&b, loads[i]);
   /* With the scalars, overflows the 480 halves. */
   for (unsigned i = 0; i < 40; ++i)
      consume(&b, ubo_load(&b, 4, 16 * (i + 1)));
   return b.shader;
}

TEST_F(PreamblePriority, AnIfThatBecomesSelectsIsNotCold)
{
   nir_shader *s = select_shader();
   unsigned sizes[] = {0, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(s, sizes, true, false, &flatten));
   EXPECT_EQ(main_loads(s, 1), 0u);
   EXPECT_GT(main_loads(s, 4), 0u);
   ralloc_free(s);

   /* Without the select pass the if stays, and its loads give way. */
   s = select_shader();
   unsigned sizes_branch[] = {0, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(s, sizes_branch, true, false, NULL));
   EXPECT_GT(main_loads(s, 1), 0u);
   ralloc_free(s);
}

TEST_F(PreamblePriority, FreeSlotsStillTakeDeepCode)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "deep");
   nir_def *c = nir_ine_imm(&b, ubo_load(&b, 1, 0), 0);
   nir_push_if(&b, c);
   nir_push_if(&b, c);
   nir_push_if(&b, c);
   /* The store needs a move out of the uniform, the weight must not hide it. */
   consume(&b, ubo_load(&b, 4, 16));
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);

   unsigned sizes[] = {0, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, false, NULL));
   EXPECT_EQ(main_loads(b.shader, 4), 0u);
   ralloc_free(b.shader);
}

TEST_F(PreamblePriority, FreeSlotsStillTakeAUniformSelect)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "phi");
   nir_push_if(&b, nir_ine_imm(&b, ubo_load(&b, 1, 0), 0));
   nir_def *x = ubo_load(&b, 4, 16);
   nir_push_else(&b, NULL);
   nir_def *y = ubo_load(&b, 4, 32);
   nir_pop_if(&b, NULL);
   consume(&b, nir_if_phi(&b, x, y));

   unsigned sizes[] = {0, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, false, NULL));
   EXPECT_EQ(main_loads(b.shader, 4), 0u);
   ralloc_free(b.shader);
}
