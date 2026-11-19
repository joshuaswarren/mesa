/* SPDX-License-Identifier: MIT */
#include <map>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
#include "asahi/lib/agx_abi.h"
}

TEST(ImageAtomicMetadata, AddressLoadsFollowDescriptorABI)
{
   for (unsigned generation : {9u, 10u, 10u | AGX_NIR_TEXTURE_IMAGE_SIDEBAND}) {
      for (bool metadata : {false, true}) {
         /* The sideband flag is a variant of the plain lowering only. */
         if (metadata && (generation & AGX_NIR_TEXTURE_IMAGE_SIDEBAND))
            continue;
         SCOPED_TRACE(generation);
         SCOPED_TRACE(metadata);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "image atomic metadata");
         nir_def *coord = nir_pad_vector(
            &b, nir_load_global_invocation_id(&b, 32), 4);
         nir_bindless_image_atomic(
            &b, 32, nir_imm_int(&b, 1), coord, nir_imm_int(&b, 0),
            nir_imm_int(&b, 1), .image_dim = GLSL_SAMPLER_DIM_2D,
            .image_array = true, .format = PIPE_FORMAT_R32_UINT,
            .atomic_op = nir_atomic_op_iadd);

         if (metadata)
            agx_nir_lower_texture_with_buffer_metadata(b.shader, generation,
                                                       false, 0, false);
         else
            agx_nir_lower_texture_for_generation(b.shader, generation);

         /* Give the descriptor a known base so we can check the actual memory
          * reads produced by the address helper, including its high word.
          */
         const uint64_t base = 0x100000;
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
                  b.cursor = nir_before_instr(instr);
                  nir_def_rewrite_uses(&intr->def, nir_imm_int64(&b, base));
               }
            }
         }
         while (nir_opt_constant_folding(b.shader)) {}

         std::map<uint64_t, unsigned> loads;
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                   nir_src_is_const(intr->src[0]))
                  loads[nir_src_as_uint(intr->src[0])] =
                     intr->def.bit_size * intr->def.num_components;
            }
         }
         /* A caller that declares the storage sideband (hk on G18G) reads
          * the word after the texture/PBE pair, as the metadata ABI does. */
         const unsigned offset = generation == 10 && !metadata
            ? AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET : AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE;
         EXPECT_EQ(loads.count(base + offset), 1u);
         EXPECT_EQ(loads[base + offset], 64u);
         const unsigned other = offset == 24 ? 40 : 24;
         EXPECT_EQ(loads.count(base + other), 0u);
         nir_validate_shader(b.shader, "image atomic descriptor addressing");
         ralloc_free(b.shader);
      }
   }
}

/* The sideband flag only moves non-buffer image atomics: a texel-buffer
 * atomic under it still takes the plain Gen10 form, which reads the word
 * from the PBE's last eight bytes. */
TEST(ImageAtomicMetadata, SidebandLeavesBufferAtomics)
{
   for (unsigned generation : {10u, 10u | AGX_NIR_TEXTURE_IMAGE_SIDEBAND}) {
      SCOPED_TRACE(generation);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "buffer atomic metadata");
      nir_def *coord = nir_pad_vector(
         &b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0), 4);
      nir_bindless_image_atomic(
         &b, 32, nir_imm_int(&b, 1), coord, nir_imm_int(&b, 0),
         nir_imm_int(&b, 1), .image_dim = GLSL_SAMPLER_DIM_BUF,
         .format = PIPE_FORMAT_R32_UINT, .atomic_op = nir_atomic_op_iadd);

      agx_nir_lower_texture_for_generation(b.shader, generation);

      const uint64_t base = 0x100000;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
               b.cursor = nir_before_instr(instr);
               nir_def_rewrite_uses(&intr->def, nir_imm_int64(&b, base));
            }
         }
      }
      while (nir_opt_constant_folding(b.shader)) {}

      std::map<uint64_t, unsigned> loads;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                nir_src_is_const(intr->src[0]))
               loads[nir_src_as_uint(intr->src[0])] =
                  intr->def.bit_size * intr->def.num_components;
         }
      }
      EXPECT_EQ(loads.count(base + AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET), 1u);
      EXPECT_EQ(loads[base + AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET], 64u);
      EXPECT_EQ(loads.count(base + AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE), 0u);
      nir_validate_shader(b.shader, "buffer atomic descriptor addressing");
      ralloc_free(b.shader);
   }
}
