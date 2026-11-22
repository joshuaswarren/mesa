/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx3_pixel_core.h"
#include "agx_nir.h"
}
#include <vector>
#include <algorithm>

static std::vector<uint8_t>
compile_mrt(bool secondary, unsigned targets, bool *ok, bool no_stop = false)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "standalone MRT compilation");
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.secondary = secondary;
   key.no_stop = no_stop;
   key.no_preamble = true;
   key.reserved_preamble = 8;
   key.fs.nr_samples = 1;
   for (unsigned i = 0; i < targets; ++i) {
      nir_def *value = agx_test_load_preamble(&b, 4, 16, 4);
      agx_test_store_local_pixel_agx(&b, value, nir_imm_intN_t(&b, 0xff, 16),
                                     nir_undef(&b, 2, 16), int(8 * i), 0xf,
                                     PIPE_FORMAT_R16_FLOAT);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + i);
      BITSET_SET(key.g17p_tile_partitions, 8 * i);
   }
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   *ok = agx_compile_shader_nir(b.shader, &key, &part);
   std::vector<uint8_t> bytes;
   if (*ok)
      bytes.assign(
         (uint8_t *)part.binary + part.info.main_offset,
         (uint8_t *)part.binary + part.info.main_offset + part.info.main_size);
   free(part.binary);
   return bytes;
}

TEST(InlineMRT, StandaloneMatchesExistingSecondaryProtocol)
{
   for (unsigned targets : {2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
      SCOPED_TRACE(targets);
      bool secondary_ok = false, primary_ok = false;
      auto secondary = compile_mrt(true, targets, &secondary_ok);
      auto primary = compile_mrt(false, targets, &primary_ok);
      ASSERT_TRUE(secondary_ok);
      ASSERT_TRUE(primary_ok);
      EXPECT_EQ(primary, secondary);
   }
}

TEST(InlineMRT, IncompleteMainPartStillRejected)
{
   bool ok = true;
   auto bytes = compile_mrt(false, 2, &ok, true);
   EXPECT_FALSE(ok);
}

TEST(Background, M5MultisampleTileTransaction)
{
   for (unsigned samples : {2u, 4u, 8u}) {
      SCOPED_TRACE(samples);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 background transaction");
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      key.no_preamble = true;
      key.reserved_preamble = 8;
      key.fs.background = true;
      key.fs.nr_samples = samples;
      nir_def *value = nir_load_preamble(&b, 4, 32, .base = 4);
      nir_store_local_pixel_agx(
         &b, value, nir_u2u16(&b, nir_load_sample_mask(&b)),
         nir_undef(&b, 2, 16), .write_mask = 0xf,
         .format = PIPE_FORMAT_R8_UNORM);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      const uint8_t acquire[] = {0x67, 0x40, 0, 5, 0, 0};
      const uint8_t release[] = {0x67, 0x10, 0, 4, 0, 0};
      const uint8_t *begin = (uint8_t *)part.binary + part.info.main_offset;
      const uint8_t *end = begin + part.info.main_size;
      const uint8_t *wait = std::search(begin, end, std::begin(acquire),
                                       std::end(acquire));
      const uint8_t *signal = std::search(begin, end, std::begin(release),
                                         std::end(release));
      EXPECT_NE(wait, end);
      EXPECT_NE(signal, end);
      EXPECT_LT(wait, signal);
      free(part.binary);
   }
}

TEST(LocalRead, DeferredMultiBlockTileProtocol)
{
   struct chip_case {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      const char *name;
   };
   const chip_case chips[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, "G17P"},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, "G17S"},
   };

   for (const auto &c : chips) {
      for (unsigned samples : {1u, 4u}) {
         SCOPED_TRACE(::testing::Message()
                      << c.name << " samples=" << samples);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "local_read deferred");
         agx_shader_key key = {};
         key.dev.encoding = c.enc;
         key.dev.chip = c.chip;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.agx3_local_read = true;
         key.fs.nr_samples = samples;

         /* 5 color targets: RT0..RT3 read-only G-buffer (partitions 0..3),
          * RT4 written output (partition 4). */
         const unsigned offsets[5] = {0, 8, 16, 24, 32};
         for (unsigned i = 0; i < 5; ++i)
            BITSET_SET(key.g17p_tile_partitions, offsets[i]);

         nir_def *cov =
            samples == 1 ? nir_imm_intN_t(&b, 0xff, 16)
                         : nir_u2u16(&b, nir_load_sample_mask(&b));
         nir_def *n_lights = nir_load_preamble(&b, 1, 32, .base = 4);

         for (unsigned s = 0; s < samples; ++s) {
            nir_def *smask =
               samples == 1
                  ? cov
                  : nir_iand(&b, cov, nir_imm_intN_t(&b, 1u << s, 16));
            nir_def *acc = nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f);
            for (unsigned rt = 0; rt < 4; ++rt) {
               nir_def *v = nir_load_local_pixel_agx(
                  &b, 4, 32, smask, .base = static_cast<int>(offsets[rt]),
                  .format = PIPE_FORMAT_R8_UNORM, .target = rt);
               acc = nir_fadd(&b, acc, v);
            }

            /* Dynamic lighting loop between tile loads and tile store to
             * exercise true multi-block CFG ordering. */
            nir_variable *res_var = nir_local_variable_create(
               b.impl, glsl_vec4_type(), "res");
            nir_variable *idx_var = nir_local_variable_create(
               b.impl, glsl_uint_type(), "idx");
            nir_store_var(&b, res_var, acc, 0xf);
            nir_store_var(&b, idx_var, nir_imm_int(&b, 0), 0x1);
            nir_push_loop(&b);
            {
               nir_def *idx = nir_load_var(&b, idx_var);
               nir_push_if(&b, nir_uge(&b, idx, n_lights));
               {
                  nir_jump(&b, nir_jump_break);
               }
               nir_pop_if(&b, NULL);
               nir_def *cur = nir_load_var(&b, res_var);
               nir_def *step = nir_fmul_imm(&b, cur, 0.9f);
               nir_store_var(&b, res_var, nir_fadd(&b, step, acc), 0xf);
               nir_store_var(&b, idx_var, nir_iadd_imm(&b, idx, 1), 0x1);
            }
            nir_pop_loop(&b, NULL);
            nir_def *out_val = nir_load_var(&b, res_var);

            nir_store_local_pixel_agx(&b, out_val, smask, nir_undef(&b, 2, 16),
                                      .base = static_cast<int>(offsets[4]),
                                      .write_mask = 0xf,
                                      .format = PIPE_FORMAT_R8_UNORM,
                                      .target = 4);
         }

         b.shader->info.outputs_read = BITFIELD64_RANGE(FRAG_RESULT_DATA0, 4);
         b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA4);
         b.shader->info.fs.uses_fbfetch_output = true;

         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << agx3_last_pack_refusal();
         EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
         /* All 5 partitions touched (4 read-only + 1 written): class pairs 0..2 */
         EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 0x7u);
         EXPECT_GT(part.info.main_size, 10u);
         free(part.binary);
      }
   }
}

TEST(LocalRead, DeferredMixedFormatGBuffer)
{
   struct chip_case {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      const char *name;
   };
   const chip_case chips[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, "G17P"},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, "G17S"},
   };

   for (const auto &c : chips) {
      for (unsigned samples : {1u, 4u}) {
         SCOPED_TRACE(::testing::Message()
                      << c.name << " samples=" << samples);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "local_read mixed gbuffer");
         agx_shader_key key = {};
         key.dev.encoding = c.enc;
         key.dev.chip = c.chip;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.agx3_local_read = true;
         key.fs.nr_samples = samples;

         /* Exact 5-RT mixed-format layout matching deferred G-buffer:
          * RT0: RGBA8_UNORM (4B @ 0)
          * RT1: RGBA16_SFLOAT (8B @ 8, physical R16_UINT + f2f32)
          * RT2: RGBA8_UNORM (4B @ 16)
          * RT3: R32_SFLOAT (4B @ 20, 1-comp physical R32_UINT)
          * RT4: RGBA8_UNORM (4B @ 24, output) */
         const unsigned offsets[5] = {0, 8, 16, 20, 24};
         for (unsigned i = 0; i < 5; ++i)
            BITSET_SET(key.g17p_tile_partitions, offsets[i]);

         nir_def *cov =
            samples == 1 ? nir_imm_intN_t(&b, 0xff, 16)
                         : nir_u2u16(&b, nir_load_sample_mask(&b));

         for (unsigned s = 0; s < samples; ++s) {
            nir_def *smask =
               samples == 1
                  ? cov
                  : nir_iand(&b, cov, nir_imm_intN_t(&b, 1u << s, 16));
            nir_def *g0 = nir_load_local_pixel_agx(
               &b, 4, 32, smask, .base = static_cast<int>(offsets[0]),
               .format = PIPE_FORMAT_R8_UNORM, .target = 0);
            nir_def *g1_16 = nir_load_local_pixel_agx(
               &b, 4, 16, smask, .base = static_cast<int>(offsets[1]),
               .format = PIPE_FORMAT_R16_UINT, .target = 1);
            nir_def *g1 = nir_f2f32(&b, g1_16);
            nir_def *g2 = nir_load_local_pixel_agx(
               &b, 4, 32, smask, .base = static_cast<int>(offsets[2]),
               .format = PIPE_FORMAT_R8_UNORM, .target = 2);
            nir_def *g3 = nir_load_local_pixel_agx(
               &b, 1, 32, smask, .base = static_cast<int>(offsets[3]),
               .format = PIPE_FORMAT_R32_UINT, .target = 3);

            nir_def *atten = nir_frcp(
               &b, nir_fadd_imm(&b, nir_fmul(&b, g3, g3), 1.0f));
            nir_def *out_val =
               nir_fsat(&b, nir_fmul(&b, nir_fadd(&b, nir_fadd(&b, g0, g1), g2),
                                     atten));
            nir_store_local_pixel_agx(&b, out_val, smask, nir_undef(&b, 2, 16),
                                      .base = static_cast<int>(offsets[4]),
                                      .write_mask = 0xf,
                                      .format = PIPE_FORMAT_R8_UNORM,
                                      .target = 4);
         }

         b.shader->info.outputs_read = BITFIELD64_RANGE(FRAG_RESULT_DATA0, 4);
         b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA4);
         b.shader->info.fs.uses_fbfetch_output = true;

         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << agx3_last_pack_refusal();
         EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
         EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 0x7u);
         EXPECT_GT(part.info.main_size, 10u);
         free(part.binary);
      }
   }
}

TEST(LocalRead, PhaseSplitDisjointAndLoadOnly)
{
   struct chip_case {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      const char *name;
   };
   const chip_case chips[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, "G17P"},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, "G17S"},
   };

   for (const auto &c : chips) {
      /* Case 1: Disjoint load-only (RT1+RT2, pairs 1,2) and store-only (RT0, pair 0) */
      {
         SCOPED_TRACE(::testing::Message() << c.name << " disjoint");
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "local_read disjoint");
         agx_shader_key key = {};
         key.dev.encoding = c.enc;
         key.dev.chip = c.chip;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.agx3_local_read = true;
         key.fs.nr_samples = 1;

         const unsigned offsets[3] = {0, 4, 8};
         for (unsigned i = 0; i < 3; ++i)
            BITSET_SET(key.g17p_tile_partitions, offsets[i]);

         nir_def *smask = nir_imm_intN_t(&b, 0xff, 16);
         nir_def *g1 = nir_load_local_pixel_agx(
            &b, 4, 32, smask, .base = static_cast<int>(offsets[1]),
            .format = PIPE_FORMAT_R8_UNORM, .target = 1);
         nir_def *g2 = nir_load_local_pixel_agx(
            &b, 4, 32, smask, .base = static_cast<int>(offsets[2]),
            .format = PIPE_FORMAT_R8_UNORM, .target = 2);
         nir_def *out_val = nir_fsat(&b, nir_fadd(&b, g1, g2));
         nir_store_local_pixel_agx(&b, out_val, smask, nir_undef(&b, 2, 16),
                                   .base = static_cast<int>(offsets[0]),
                                   .write_mask = 0xf,
                                   .format = PIPE_FORMAT_R8_UNORM,
                                   .target = 0);

         b.shader->info.outputs_read =
            BITFIELD64_BIT(FRAG_RESULT_DATA1) | BITFIELD64_BIT(FRAG_RESULT_DATA2);
         b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
         b.shader->info.fs.uses_fbfetch_output = true;

         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << agx3_last_pack_refusal();
         EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
         EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 0x7u);
         free(part.binary);
      }

      /* Case 2: Load-only (no tile stores at all, e.g. side-effect or discard) */
      {
         SCOPED_TRACE(::testing::Message() << c.name << " load_only");
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "local_read load_only");
         agx_shader_key key = {};
         key.dev.encoding = c.enc;
         key.dev.chip = c.chip;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.agx3_local_read = true;
         key.fs.nr_samples = 1;

         const unsigned offsets[2] = {0, 4};
         for (unsigned i = 0; i < 2; ++i)
            BITSET_SET(key.g17p_tile_partitions, offsets[i]);

         nir_def *smask = nir_imm_intN_t(&b, 0xff, 16);
         nir_def *g0 = nir_load_local_pixel_agx(
            &b, 4, 32, smask, .base = static_cast<int>(offsets[0]),
            .format = PIPE_FORMAT_R8_UNORM, .target = 0);
         nir_def *g1 = nir_load_local_pixel_agx(
            &b, 4, 32, smask, .base = static_cast<int>(offsets[1]),
            .format = PIPE_FORMAT_R8_UNORM, .target = 1);
         nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
         nir_store_global(&b, nir_fadd(&b, g0, g1), addr, .write_mask = 0xf);

         b.shader->info.outputs_read =
            BITFIELD64_BIT(FRAG_RESULT_DATA0) | BITFIELD64_BIT(FRAG_RESULT_DATA1);
         b.shader->info.writes_memory = true;
         b.shader->info.fs.early_fragment_tests = true;
         b.shader->info.fs.uses_fbfetch_output = true;

         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << agx3_last_pack_refusal();
         EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
         EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 0x3u);
         free(part.binary);
      }

      /* Case 3: 5-RT deferred (RT0..RT3 read, RT4 write -> pairs 1,2 load-only
       * and pair 0 shared R/W). Early PIXREL(0xf0) must retire the in-flight
       * load-only scoreboard slot(s) even when AGX3B reissues slot 6 for a
       * trailing shared-RW LD_TILE. */
      {
         SCOPED_TRACE(::testing::Message() << c.name << " mixed_phase_split");
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "local_read mixed_split");
         agx_shader_key key = {};
         key.dev.encoding = c.enc;
         key.dev.chip = c.chip;
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.agx3_local_read = true;
         key.fs.nr_samples = 1;

         const unsigned offsets[5] = {0, 4, 12, 16, 20};
         for (unsigned i = 0; i < 5; ++i)
            BITSET_SET(key.g17p_tile_partitions, offsets[i]);

         nir_def *smask = nir_imm_intN_t(&b, 0xff, 16);
         nir_def *g0 = nir_load_local_pixel_agx(
            &b, 4, 32, smask, .base = static_cast<int>(offsets[0]),
            .format = PIPE_FORMAT_R8_UNORM, .target = 0);
         nir_def *g1_16 = nir_load_local_pixel_agx(
            &b, 4, 16, smask, .base = static_cast<int>(offsets[1]),
            .format = PIPE_FORMAT_R16_UINT, .target = 1);
         nir_def *g1 = nir_f2f32(&b, g1_16);
         nir_def *g2 = nir_load_local_pixel_agx(
            &b, 4, 32, smask, .base = static_cast<int>(offsets[2]),
            .format = PIPE_FORMAT_R8_UNORM, .target = 2);
         nir_def *g3 = nir_load_local_pixel_agx(
            &b, 1, 32, smask, .base = static_cast<int>(offsets[3]),
            .format = PIPE_FORMAT_R32_UINT, .target = 3);
         nir_def *out_val = nir_fsat(
            &b, nir_fmul(&b, nir_fadd(&b, nir_fadd(&b, g0, g1), g2),
                         nir_frcp(&b, nir_fadd_imm(&b, nir_fmul(&b, g3, g3), 1.0f))));
         nir_store_local_pixel_agx(&b, out_val, smask, nir_undef(&b, 2, 16),
                                   .base = static_cast<int>(offsets[4]),
                                   .write_mask = 0xf,
                                   .format = PIPE_FORMAT_R8_UNORM,
                                   .target = 4);

         b.shader->info.outputs_read = BITFIELD64_RANGE(FRAG_RESULT_DATA0, 4);
         b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA4);
         b.shader->info.fs.uses_fbfetch_output = true;

         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << agx3_last_pack_refusal();
         EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
         EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 0x7u);

         const uint8_t *begin = (uint8_t *)part.binary + part.info.main_offset;
         const uint8_t *end = begin + part.info.main_size;
         uint8_t rel_nowait[AGX3_PIXEL_MAX_LENGTH], rel_nowait_len = 0;
         ASSERT_TRUE(agx3_pixel_release(c.enc, 0xf0, false, 0, rel_nowait,
                                        &rel_nowait_len));
         EXPECT_EQ(std::search(begin, end, rel_nowait,
                               rel_nowait + rel_nowait_len),
                   end);

         if (c.enc == AGX_ISA_ENCODING_AGX3B) {
            uint8_t rel_slot6[AGX3_PIXEL_MAX_LENGTH], rel_slot6_len = 0;
            ASSERT_TRUE(agx3_pixel_release(c.enc, 0xf0, false, 1u << 5,
                                           rel_slot6, &rel_slot6_len));
            EXPECT_NE(std::search(begin, end, rel_slot6,
                                  rel_slot6 + rel_slot6_len),
                      end);
         }
         free(part.binary);
      }
   }
}
