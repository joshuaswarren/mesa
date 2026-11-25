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
store_result(nir_builder *b, nir_def *v)
{
   agx_test_store_local_pixel_agx(
      b, nir_vec4(b, v, v, v, v), nir_imm_intN_t(b, 0xff, 16),
      nir_undef(b, 2, 16), 0, 0xf, PIPE_FORMAT_R32_UINT);
   b->shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
}

TEST(QuadControl, ExecutionModesSurviveInfoRegather)
{
   /* The M4 Pro shares the G17P compiler profile, but not this capability. */
   for (unsigned profile = 0; profile < 3; ++profile) {
      const bool g17p = profile != 0, quad_control = profile == 1;
      for (unsigned mode = 0; mode < 4; ++mode) {
         SCOPED_TRACE(::testing::Message() << profile << ":" << mode);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "quad execution modes");
         b.shader->info.fs.require_full_quads = mode & 1;
         b.shader->info.fs.quad_derivatives = mode & 2;
         store_result(&b, agx_test_load_preamble(&b, 1, 32, 4));
         if (g17p)
            agx_preprocess_nir_g17p(b.shader);
         else
            agx_preprocess_nir(b.shader);

         /* No quad operation survives: only the source execution modes
          * require helpers. Linking/specialization may regather info. */
         nir_shader_gather_info(b.shader, b.impl);
         ASSERT_FALSE(b.shader->info.fs.needs_coarse_quad_helper_invocations);
         ASSERT_FALSE(b.shader->info.fs.needs_full_quad_helper_invocations);
         EXPECT_EQ(b.shader->info.fs.require_full_quads, bool(mode & 1));
         EXPECT_EQ(b.shader->info.fs.quad_derivatives, bool(mode & 2));
         agx_shader_key key = {};
         key.dev.chip = g17p ? AGX_CHIP_G17P : AGX_CHIP_G14G;
         key.dev.shader_quad_control = quad_control;
         key.dev.encoding =
            g17p ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX2;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.fs.nr_samples = 1;
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         EXPECT_GT(part.info.main_size, 0u);
         /* RequireFullQuads keeps triangle merging (a merged quad is full);
          * QuadDerivatives disables it. */
         EXPECT_EQ(part.info.needs_quad_derivatives, quad_control && (mode & 2));
         EXPECT_EQ(part.info.disable_tri_merging, quad_control && mode != 0);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(QuadControl, G17PDivergentQuadInstructions)
{
   for (unsigned op = 0; op < 16; ++op) {
      for (unsigned bits : {16u, 32u, 64u}) {
         if (op >= 6 && bits != 32)
            continue;
         SCOPED_TRACE(::testing::Message() << op << ":" << bits);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "divergent quad control");
         b.shader->info.fs.require_full_quads = true;
         b.shader->info.fs.quad_derivatives = true;
         nir_def *lane = nir_load_subgroup_invocation(&b);
         nir_def *input =
            bits == 64
               ? nir_pack_64_2x32_split(
                    &b, lane, nir_ixor(&b, lane, nir_imm_int(&b, 0x5a5a)))
               : nir_u2uN(&b, lane, bits);
         nir_def *value;
         nir_def *fallback = nir_imm_intN_t(&b, 0, bits);
         /* Adjacent quads take different branches; each quad is uniform. */
         nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, lane, 4), 0));
         switch (op) {
         case 0:
            value = nir_quad_broadcast(&b, input, nir_imm_int(&b, 3));
            break;
         case 1:
            value = nir_quad_broadcast(
               &b, input, nir_iand_imm(&b, nir_ushr_imm(&b, lane, 3), 3));
            break;
         case 2:
            value = nir_quad_swap_horizontal(&b, input);
            break;
         case 3:
            value = nir_quad_swap_vertical(&b, input);
            break;
         case 4:
            value = nir_quad_swap_diagonal(&b, input);
            break;
         case 5:
            value = nir_quad_broadcast(
               &b, input, nir_iand_imm(&b, nir_ushr_imm(&b, lane, 2), 3));
            break;
         case 6:
            value = nir_b2i32(
               &b, nir_quad_vote_any(&b, 1, nir_ine_imm(&b, lane, 0)));
            break;
         case 7:
            value = nir_b2i32(
               &b, nir_quad_vote_all(&b, 1, nir_ine_imm(&b, lane, 0)));
            break;
         case 8:
            value = nir_ddx_fine(&b, nir_u2f32(&b, lane));
            break;
         case 9:
            value = nir_ddy_coarse(&b, nir_u2f32(&b, lane));
            break;
         case 10:
            value = nir_b2i32(
               &b, nir_quad_vote_any(&b, 1, nir_ilt_imm(&b, lane, 5)));
            break;
         case 11:
            value = nir_b2i32(
               &b, nir_quad_vote_all(&b, 1, nir_ine_imm(&b, lane, 7)));
            break;
         case 12:
            value = nir_b2i32(
               &b, nir_quad_vote_any(&b, 1,
                                     nir_flt(&b, nir_u2f32(&b, lane),
                                             nir_imm_float(&b, 8.0))));
            break;
         case 13:
            value = nir_b2i32(
               &b,
               nir_quad_vote_all(
                  &b, 1, nir_ilt(&b, lane, agx_test_load_preamble(&b, 1, 32, 4))));
            break;
         case 14:
            value = nir_b2i32(&b, nir_quad_vote_any(&b, 1, nir_imm_true(&b)));
            break;
         default:
            value = nir_b2i32(&b, nir_quad_vote_all(&b, 1, nir_imm_false(&b)));
            break;
         }
         nir_pop_if(&b, NULL);
         value = nir_if_phi(&b, value, fallback);
         value = bits == 64
                    ? nir_ixor(&b, nir_unpack_64_2x32_split_x(&b, value),
                               nir_unpack_64_2x32_split_y(&b, value))
                    : nir_u2u32(&b, value);
         store_result(&b, value);
         agx_preprocess_nir_g17p(b.shader);
         agx_shader_key key = {};
         key.dev.chip = AGX_CHIP_G17P;
         key.dev.shader_quad_control = true;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.fs.nr_samples = 1;
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         EXPECT_GT(part.info.main_size, 0u);
         EXPECT_TRUE(part.info.needs_quad_derivatives);
         EXPECT_TRUE(part.info.disable_tri_merging);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(QuadControl, OnlyDerivativesDisableG17PTriangleMerging)
{
   /* A quad broadcast (which spirv_to_nir marks require_full_quads) keeps
    * merging, so a clipped primitive's pieces stay in whole quads; ddx does
    * not, because a derivative across a merged seam is meaningless. */
   for (bool derivative : {false, true}) {
      SCOPED_TRACE(derivative);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "quad merge");
      b.shader->info.fs.require_full_quads = true;
      nir_def *v = nir_u2f32(&b, nir_load_subgroup_invocation(&b));
      v = derivative ? nir_ddx(&b, v)
                     : nir_quad_broadcast(&b, v, nir_imm_int(&b, 1));
      store_result(&b, nir_f2u32(&b, v));
      agx_preprocess_nir_g17p(b.shader);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17P;
      key.dev.shader_quad_control = true;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.no_preamble = true;
      key.reserved_preamble = 8;
      key.fs.nr_samples = 1;
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_EQ(part.info.needs_quad_derivatives, derivative);
      EXPECT_TRUE(part.info.needs_full_quads);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(QuadControl, BallotFusionStaysWithinEncodingDomain)
{
   /* 0: AGX2. 1: AGX3B with quad control. 2: AGX3B without it. 3: AGX3C.
    * Every AGX3 encoder's half-width quad ballot only compares a register
    * against zero, whatever the device exposes, so only that predicate fuses
    * there. */
   for (unsigned profile = 0; profile < 4; ++profile) {
      const bool agx3 = profile != 0, quad_control = profile == 1;
      const enum agx_chip chip = profile == 0   ? AGX_CHIP_G14G
                                 : profile == 3 ? AGX_CHIP_G17S
                                                : AGX_CHIP_G17P;
      const enum agx_isa_encoding encoding = profile == 0 ? AGX_ISA_ENCODING_AGX2
                                             : profile == 3
                                                ? AGX_ISA_ENCODING_AGX3C
                                                : AGX_ISA_ENCODING_AGX3B;
      for (unsigned predicate = 0; predicate < 4; ++predicate) {
         SCOPED_TRACE(::testing::Message() << profile << ":" << predicate);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.chip = chip;
         key.dev.shader_quad_control = quad_control;
         key.dev.encoding = encoding;
         b->shader->key = &key;
         agx_index x = agx_preload(b, agx_register(4, AGX_SIZE_32));
         agx_index y = agx_preload(b, agx_register(6, AGX_SIZE_32));
         agx_index cmp = agx_temp(b->shader, AGX_SIZE_16);
         if (predicate == 0)
            agx_fcmp_to(b, cmp, x, y, AGX_FCOND_GT, false);
         else if (predicate == 1)
            agx_icmp_to(b, cmp, x, y, AGX_ICOND_ULT, false);
         else
            agx_icmp_to(b, cmp, x, agx_immediate(predicate == 2 ? 7 : 0),
                        AGX_ICOND_UEQ, true);
         agx_index out = agx_quad_ballot(b, cmp);
         agx_instr *use = agx_unit_test(b, out);
         agx_optimizer_backward(b->shader);
         agx_optimizer_forward(b->shader);
         agx_dce(b->shader, true);
         const bool keep_predicate = agx3 && predicate != 3;
         unsigned unfused = 0, fused = 0;
         agx_foreach_instr_global(b->shader, I) {
            unfused += I->op == AGX_OPCODE_QUAD_BALLOT;
            fused += I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
                     I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
         }
         EXPECT_EQ(unfused, keep_predicate ? 1u : 0u);
         EXPECT_EQ(fused, keep_predicate ? 0u : 1u);
         EXPECT_TRUE(agx_is_equiv(use->src[0], out));
         ralloc_free(mem);
      }
   }
}

TEST(QuadControl, CapabilityUnificationAcrossAGX3BAndAGX3C)
{
   EXPECT_TRUE(agx_chip_has_shader_quad_control(AGX_CHIP_G13G));
   EXPECT_FALSE(agx_chip_agx3_shader_quad_control(AGX_CHIP_G13G));

   const struct {
      enum agx_chip chip;
      enum agx_isa_encoding enc;
      bool m5;
   } agx3_chips[] = {
      {AGX_CHIP_G15G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17G, AGX_ISA_ENCODING_AGX3C, true},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C, true},
      {AGX_CHIP_G17C, AGX_ISA_ENCODING_AGX3C, true},
   };
   for (const auto &c : agx3_chips) {
      EXPECT_TRUE(agx_chip_agx3_shader_quad_control(c.chip));
      EXPECT_TRUE(agx_chip_has_shader_quad_control(c.chip));
   }
   EXPECT_FALSE(agx_chip_agx3_shader_quad_control(AGX_CHIP_G18G));
   EXPECT_FALSE(agx_chip_has_shader_quad_control(AGX_CHIP_G18G));

   /* Explicit quad-control operations (including general icmp/fcmp quad votes
    * that require the half-width ballot fusion guard) must compile and pack on
    * both AGX3B (G15G/G16G/G17P) and AGX3C (M5). */
   const struct {
      enum agx_chip chip;
      enum agx_isa_encoding enc;
      bool m5;
   } compile_targets[] = {
      {AGX_CHIP_G15G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C, true},
   };
   for (const auto &t : compile_targets) {
      for (unsigned op = 0; op < 6; ++op) {
         SCOPED_TRACE(::testing::Message() << agx_chip_name(t.chip) << ":" << op);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "agx3 quad control");
         b.shader->info.fs.require_full_quads = true;
         b.shader->info.fs.quad_derivatives = true;
         nir_def *lane = nir_load_subgroup_invocation(&b);
         nir_def *value;
         switch (op) {
         case 0:
            value = nir_b2i32(&b, nir_quad_vote_any(&b, 1, nir_ilt_imm(&b, lane, 5)));
            break;
         case 1:
            value = nir_b2i32(&b, nir_quad_vote_all(&b, 1, nir_ine_imm(&b, lane, 7)));
            break;
         case 2:
            value = nir_b2i32(
               &b, nir_quad_vote_any(
                      &b, 1, nir_flt(&b, nir_u2f32(&b, lane), nir_imm_float(&b, 8.0f))));
            break;
         case 3:
            value = nir_b2i32(
               &b, nir_quad_vote_all(
                      &b, 1, nir_ilt(&b, lane, agx_test_load_preamble(&b, 1, 32, 4))));
            break;
         case 4:
            value = nir_b2i32(&b, nir_quad_vote_any(&b, 1, nir_imm_true(&b)));
            break;
         default:
            value = nir_b2i32(&b, nir_quad_vote_all(&b, 1, nir_imm_false(&b)));
            break;
         }
         store_result(&b, value);
         if (t.m5)
            agx_preprocess_nir_m5(b.shader);
         else
            agx_preprocess_nir_native(b.shader, t.chip);
         nir_shader_gather_info(b.shader, b.impl);
         agx_shader_key key = {};
         key.dev.chip = t.chip;
         key.dev.encoding = t.enc;
         key.dev.shader_quad_control = agx_chip_agx3_shader_quad_control(t.chip);
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.fs.nr_samples = 1;
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         EXPECT_GT(part.info.main_size, 0u);
         EXPECT_TRUE(part.info.needs_quad_derivatives);
         EXPECT_TRUE(part.info.disable_tri_merging);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}
