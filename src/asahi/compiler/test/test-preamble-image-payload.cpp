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
check_payload(unsigned bits, unsigned heap = 24, bool g17p = true)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "preamble image payload width");
   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 24);
   nir_def *payload = agx_test_load_global_constant(&b, 1, bits, addr, bits / 8,
                                                    (gl_access_qualifier)0);
   nir_def *handle = agx_test_bindless_image_agx(&b, payload, heap);
   nir_def *v = nir_load_vertex_id(&b);
   nir_def *coords = nir_vec4(&b, v, nir_imm_int(&b, 0), nir_undef(&b, 1, 32),
                              nir_undef(&b, 1, 32));
   nir_def *value = agx_test_bindless_image_load(
      &b, 4, 32, handle, coords, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      GLSL_SAMPLER_DIM_2D, PIPE_FORMAT_R32_UINT,
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
      nir_type_uint32);
   nir_store_uvs_agx(&b, nir_channel(&b, value, 0), nir_imm_int(&b, 0));
   /* Another independent expensive uniform expression makes a neighbouring
    * export live. Its slots must not overlap the descriptor payload. */
   nir_def *extra_addr = nir_iadd_imm(&b, addr, 16);
   nir_def *extra = agx_test_load_global_constant(&b, 1, 32, extra_addr, 4,
                                                  (gl_access_qualifier)0);
   nir_store_uvs_agx(&b, nir_iadd(&b, extra, v), nir_imm_int(&b, 1));
   nir_def *sampler_load =
      agx_test_load_preamble(&b, 1, 32, 28, nir_preamble_class_sampler);
   nir_def *image_load =
      agx_test_load_preamble(&b, 1, 32, 28, nir_preamble_class_image);
   nir_validate_shader(b.shader, "before preamble");
   unsigned sizes[] = {28, 1000, 1000};
   ASSERT_TRUE(agx_nir_opt_preamble(b.shader, sizes, g17p, false, NULL));
   bool occupied[512] = {};
   unsigned payload_stores = 0, payload_loads = 0, stores = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (I->intrinsic == nir_intrinsic_store_preamble) {
               unsigned base = nir_intrinsic_base(I);
               unsigned width = I->src[0].ssa->bit_size / 16;
               EXPECT_EQ(base % width, 0u);
               for (unsigned h = 0; h < width; ++h) {
                  EXPECT_FALSE(occupied[base + h]);
                  occupied[base + h] = true;
               }
               payload_stores += I->src[0].ssa->bit_size == bits;
               stores++;
            }
            if (!impl->function->is_preamble &&
                I->intrinsic == nir_intrinsic_bindless_image_agx) {
               EXPECT_EQ(I->src[0].ssa->bit_size, bits);
               EXPECT_EQ(I->def.bit_size, 32u);
               if (bits == 64) {
                  EXPECT_EQ(nir_intrinsic_base(nir_src_as_intrinsic(I->src[0])),
                            28u);
               }
               EXPECT_EQ(nir_intrinsic_desc_set(I), heap);
               payload_loads++;
            }
         }
      }
   }
   EXPECT_GT(payload_stores, 0u);
   EXPECT_GT(stores, 1u);
   EXPECT_EQ(payload_loads, 1u);
   EXPECT_EQ(sampler_load->bit_size, 32u);
   EXPECT_EQ(image_load->bit_size, 32u);
   nir_validate_shader(b.shader, "after preamble");
   ralloc_free(b.shader);
}
TEST(AGXPreamblePayload, Native64BitPayload)
{
   check_payload(64);
}
TEST(AGXPreamblePayload, Existing32BitPayload)
{
   check_payload(32);
}
TEST(AGXPreamblePayload, GeneralHeapAboveImageLimit)
{
   check_payload(64, 36);
}
TEST(AGXPreamblePayload, Legacy32BitCallback)
{
   check_payload(32, 24, false);
}

TEST(AGXPreamblePayload, LegacyImageClassSlots)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "legacy image class slots");
   nir_def *base = agx_test_load_preamble(&b, 1, 64, 24);
   nir_def *v = nir_load_vertex_id(&b);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coords = nir_vec4(&b, v, zero, zero, zero);
   for (unsigned i = 0; i < 2; i++) {
      nir_def *payload = agx_test_load_global_constant(
         &b, 1, 32, nir_iadd_imm(&b, base, 16 * i), 4, (gl_access_qualifier)0);
      nir_def *handle = agx_test_bindless_image_agx(&b, payload, 24);
      nir_def *value = agx_test_bindless_image_load(
         &b, 4, 32, handle, coords, zero, zero, GLSL_SAMPLER_DIM_2D,
         PIPE_FORMAT_R32_UINT,
         (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         nir_type_uint32);
      nir_store_uvs_agx(&b, nir_channel(&b, value, 0), nir_imm_int(&b, i));
   }
   unsigned sizes[] = {28, 0, 0};
   ASSERT_TRUE(agx_nir_opt_preamble(b.shader, sizes, false, false, NULL));
   EXPECT_EQ(sizes[1], 4u);
   unsigned seen = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (I->intrinsic == nir_intrinsic_store_preamble &&
                nir_intrinsic_preamble_class(I) == nir_preamble_class_image) {
               EXPECT_EQ(I->src[0].ssa->bit_size, 32u);
               EXPECT_LT(nir_intrinsic_base(I), 4u);
               EXPECT_EQ(nir_intrinsic_base(I) % 2, 0u);
               seen |= 1u << (nir_intrinsic_base(I) / 2);
            }
         }
      }
   }
   EXPECT_EQ(seen, 3u);
   nir_validate_shader(b.shader, "legacy image class slots");
   ralloc_free(b.shader);
}
