/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

TEST(PreambleTexturePolicy, M5KeepsTextureAndConsumersInMain)
{
   for (bool m5 : {false, true}) {
      for (bool dependent : {false, true}) {
         SCOPED_TRACE(m5);
         SCOPED_TRACE(dependent);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "texture preamble policy");
         b.shader->info.num_textures = 1;
         nir_def *address = agx_test_load_preamble(&b, 1, 64, 0);
         nir_def *other = agx_test_load_global_constant(
            &b, 1, 32, nir_iadd_imm(&b, address, 16), 4,
            (gl_access_qualifier)0);

         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
         tex->op = nir_texop_txf;
         tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 2;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                          nir_imm_ivec2(&b, 0, 0));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_def *value = dependent ? nir_fmul_imm(&b, &tex->def, 2.0)
                                    : &tex->def;
         nir_intrinsic_instr *store = nir_store_global(&b, value, address);
         nir_intrinsic_set_align(store, 4, 0);
         store = nir_store_global(&b, other, nir_iadd_imm(&b, address, 64));
         nir_intrinsic_set_align(store, 4, 0);
         nir_validate_shader(b.shader, "before texture preamble policy");

         unsigned sizes[] = {4, 1000, 1000};
         EXPECT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, m5, NULL));
         nir_opt_dce(b.shader);
         nir_validate_shader(b.shader, "after texture preamble policy");

         unsigned main_tex = 0, preamble_tex = 0, preamble_loads = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_tex) {
                     if (impl->function->is_preamble)
                        preamble_tex++;
                     else
                        main_tex++;
                  }
                  if (impl->function->is_preamble &&
                      instr->type == nir_instr_type_intrinsic &&
                      nir_instr_as_intrinsic(instr)->intrinsic ==
                         nir_intrinsic_load_global_constant)
                     preamble_loads++;
               }
            }
         }
         EXPECT_EQ(main_tex, m5 ? 1u : 0u);
         EXPECT_EQ(preamble_tex, m5 ? 0u : 1u);
         EXPECT_EQ(preamble_loads, 1u);
         ralloc_free(b.shader);
      }
   }
}

/* A device load reads its base from the same low uniforms a sample reads its
 * handles from, so on G17P a base exported above them moves down. */
TEST(PreambleHandlePlacement, LoadBasesMoveBelowTheHandleLimit)
{
   for (bool load_bases : {false, true}) {
      SCOPED_TRACE(load_bases);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "load base placement");
      nir_def *table = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *base = agx_test_load_global_constant(&b, 1, 64, table, 8,
                                                    (gl_access_qualifier)0);
      /* A per-thread index keeps the load itself in the main program. */
      nir_def *value = nir_load_constant_agx(
         &b, 1, 32, base, nir_load_local_invocation_index(&b),
         .access = ACCESS_CAN_REORDER, .base = 2,
         .format = PIPE_FORMAT_R32_UINT);
      nir_intrinsic_instr *store = nir_store_global(&b, value, table);
      nir_intrinsic_set_align(store, 4, 0);

      unsigned sizes[] = {300, 1000, 1000};
      ASSERT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, false, NULL));
      nir_opt_dce(b.shader);

      unsigned allocated = sizes[0];
      EXPECT_EQ(agx_nir_place_preamble_handles_low(b.shader, 8, allocated,
                                                   &allocated, load_bases),
                load_bases);
      nir_validate_shader(b.shader, "after load base placement");

      int read = -1;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_load_constant_agx)
               continue;
            nir_intrinsic_instr *base_load = nir_src_as_intrinsic(
               nir_instr_as_intrinsic(instr)->src[0]);
            ASSERT_NE(base_load, nullptr);
            ASSERT_EQ(base_load->intrinsic, nir_intrinsic_load_preamble);
            read = nir_intrinsic_base(base_load);
         }
      }
      if (load_bases) {
         EXPECT_EQ(read, 8);
      } else {
         EXPECT_EQ(read, 300);
      }
      ralloc_free(b.shader);
   }
}
