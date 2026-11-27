/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * M5 fragment epilog depth/stencil export forms, against native
 * Native G17G compiler fragments:
 *  - depth or stencil alone, whole pixel: one DEPTHF in a loop over the
 *    covered samples, with class 1 taken once ahead of it;
 *  - depth and stencil together, whole pixel: one DEPTHF word;
 *  - alpha-to-coverage: one DEPTHF per sample with the alpha coverage folded
 *    into its word, and no separate MOVMSK;
 *  - blended alpha-to-coverage without a depth write: one MOVMSK per pixel,
 *    ahead of the per-sample blend;
 *  - blended with a depth write: all of the depth words above, ahead of the
 *    per-sample blend.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <vector>
#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "agx_tilebuffer.h"
#include "agx_linker.h"
extern "C" {
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_debug.h"
#include "asahi/isa/disasm.h"
}

namespace {

class M5DepthExport : public ::testing::Test {
 protected:
   M5DepthExport() { glsl_type_singleton_init_or_ref(); }
   ~M5DepthExport() { glsl_type_singleton_decref(); }
};

struct epilog_shape {
   enum agx_chip chip;
   unsigned samples;
   bool z, s, a2c;
   bool blend = false, sample_shading = false;
   enum pipe_format format = PIPE_FORMAT_R8G8B8A8_UNORM;
};

static nir_shader *
build_epilog(const epilog_shape &shape)
{
   agx_fs_epilog_key fs = {};
   fs.nr_samples = shape.samples;
   fs.max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE;
   fs.max_bytes_per_tile = 32768;
   fs.max_samples_per_tile = 2048;
   fs.link.loc_written = 1;
   fs.link.size_32 = 1;
   fs.link.colour_comps = BITFIELD_MASK(4);
   fs.link.write_z = shape.z;
   fs.link.write_s = shape.s;
   fs.link.sample_shading = shape.sample_shading;
   fs.blend.alpha_to_coverage = shape.a2c;
   agx_fs_epilog_apply_device_policy(shape.chip, &fs);
   fs.rt_formats[0] = shape.format;
   fs.remap[0] = 0;
   for (unsigned rt = 1; rt < ARRAY_SIZE(fs.remap); ++rt)
      fs.remap[rt] = -1;
   fs.blend.rt[0].colormask = 0xf;
   fs.blend.rt[0].mode =
      shape.blend
         ? agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_SRC_ALPHA,
                                   PIPE_BLENDFACTOR_INV_SRC_ALPHA,
                                   PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                                   PIPE_BLENDFACTOR_INV_SRC_ALPHA)
         : agx_pack_blend_standard(PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
                                   PIPE_BLENDFACTOR_ZERO, PIPE_BLEND_ADD,
                                   PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 depth export epilog");
   /* As hk selects the builder for M5. */
   if (shape.sample_shading)
      agx_nir_fs_epilog_g17p(&b, &fs);
   else
      agx_nir_fs_epilog(&b, &fs);
   return b.shader;
}

struct counts {
   unsigned store_zs = 0, covered_sample = 0, transactions = 0, demotes = 0;
   unsigned coverage_updates = 0;
   unsigned stores_in_loop = 0;
};

static counts
count_nir(nir_shader *s)
{
   counts c;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      bool in_loop = false;
      for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent)
         in_loop |= n->type == nir_cf_node_loop;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
            c.store_zs++;
            c.covered_sample +=
               !!(nir_intrinsic_base(intr) & AGX_ZS_COVERED_SAMPLE);
            c.stores_in_loop += in_loop;
         }
         c.transactions += intr->intrinsic == nir_intrinsic_zs_transaction_agx;
         c.demotes += intr->intrinsic == nir_intrinsic_demote_samples;
         c.coverage_updates += intr->intrinsic == nir_intrinsic_sample_mask_agx;
      }
   }
   return c;
}

/* Mnemonics of an AGX3c binary, in order. */
static std::vector<std::string>
mnemonics(const agx_shader_part &part)
{
   std::vector<std::string> out;
   const uint8_t *code = (const uint8_t *)part.binary + part.info.main_offset;
   size_t len = part.info.main_size, i = 0;
   while (i < len) {
      char *text = NULL;
      size_t text_len = 0;
      FILE *fp = open_memstream(&text, &text_len);
      uint8_t buf[32] = {0};
      memcpy(buf, code + i, MIN2(sizeof(buf), len - i));
      signed ret = agx3c_disassemble_instr((BITSET_WORD *)buf, fp, i,
                                           len - i, false);
      fclose(fp);
      std::string line = text ? text : "";
      free(text);
      unsigned length = abs(ret);
      if (length == 0)
         break;
      /* Keep the mnemonic: the first word after the decoder's
       * "# XXX: Unknown bit set N" notes. */
      size_t at = 0;
      while (at < line.size()) {
         size_t start = line.find_first_not_of(" \t\n", at);
         if (start == std::string::npos)
            break;
         size_t end = line.find_first_of(" \t\n", start);
         std::string word = line.substr(start, end - start);
         at = end;
         if (word == "#" || word == "XXX:" || word == "Unknown" ||
             word == "bit" || word == "set" ||
             word.find_first_not_of("0123456789") == std::string::npos)
            continue;
         out.push_back(word);
         break;
      }
      i += length;
   }
   return out;
}

static unsigned
count_of(const std::vector<std::string> &m, const char *name)
{
   unsigned n = 0;
   for (const auto &s : m)
      n += s == name;
   return n;
}

static int
index_of(const std::vector<std::string> &m, const char *name)
{
   for (unsigned i = 0; i < m.size(); ++i) {
      if (m[i] == name)
         return i;
   }
   return -1;
}

static bool
compile_m5(nir_shader *s, unsigned samples, agx_shader_part *part,
           bool preprocessed = false)
{
   if (!preprocessed)
      agx_preprocess_nir(s);
   if (samples > 1)
      agx_nir_lower_g17p_tile_samples(s, samples);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.secondary = true;
   key.fs.nr_samples = samples;
   BITSET_SET(key.g17p_tile_partitions, 0);
   return agx_compile_shader_nir(s, &key, part);
}

} /* namespace */

/* Native zs_depth_{2,4,8}x, zs_cdepth_4x and zs_stencil_{2,4}x. */
TEST_F(M5DepthExport, SingleAttachmentLoopsOverCoveredSamples)
{
   for (unsigned samples : {2u, 4u, 8u}) {
      for (bool stencil : {false, true}) {
         SCOPED_TRACE(::testing::Message() << samples << " samples, "
                      << (stencil ? "stencil" : "depth"));
         nir_shader *s = build_epilog({AGX_CHIP_G17S, samples, !stencil,
                                       stencil, false});
         counts c = count_nir(s);
         EXPECT_EQ(c.store_zs, 1u);
         EXPECT_EQ(c.covered_sample, 1u);
         EXPECT_EQ(c.stores_in_loop, 1u);
         EXPECT_EQ(c.transactions, 2u);

         agx_shader_part part = {};
         ASSERT_TRUE(compile_m5(s, samples, &part)) << agx_last_compile_refusal();
         std::vector<std::string> m = mnemonics(part);
         EXPECT_EQ(count_of(m, "depthf"), 1u);
         EXPECT_EQ(count_of(m, "cndlt"), 1u);
         /* One class-1 acquire, ahead of the loop and its DEPTHF. */
         int depthf = index_of(m, "depthf"), loop = index_of(m, "cndlt");
         int last_wait = -1;
         unsigned waits_before_depthf = 0;
         for (int i = 0; i < depthf; ++i) {
            if (m[i] == "pixwait") {
               last_wait = i;
               waits_before_depthf++;
            }
         }
         EXPECT_EQ(waits_before_depthf, 2u); /* entry gate, then class 1 */
         EXPECT_LT(last_wait, depthf);
         EXPECT_LT(depthf, loop);
         bool cf_between = false;
         for (int i = last_wait + 1; i < depthf; ++i)
            cf_between |= m[i] == "cndst";
         EXPECT_TRUE(cf_between) << "class-1 wait must precede the loop";
         free(part.binary);
         ralloc_free(s);
      }
   }
}

/* Native zs_depth_stencil_{2,4}x. */
TEST_F(M5DepthExport, DepthAndStencilIsOneWord)
{
   for (unsigned samples : {2u, 4u}) {
      SCOPED_TRACE(::testing::Message() << samples << " samples");
      nir_shader *s = build_epilog({AGX_CHIP_G17S, samples, true, true, false});
      counts c = count_nir(s);
      EXPECT_EQ(c.store_zs, 1u);
      EXPECT_EQ(c.covered_sample, 0u);
      EXPECT_EQ(c.transactions, 0u);
      agx_shader_part part = {};
      ASSERT_TRUE(compile_m5(s, samples, &part)) << agx_last_compile_refusal();
      std::vector<std::string> m = mnemonics(part);
      EXPECT_EQ(count_of(m, "depthf"), 1u);
      EXPECT_EQ(count_of(m, "cndlt"), 0u);
      EXPECT_EQ(count_of(m, "movmsk"), 0u);
      free(part.binary);
      ralloc_free(s);
   }
}

/* Native zs_depth_{2,4}x_a2c, zs_cdepth_4x_a2c, zs_stencil_4x_a2c and
 * zs_depth_stencil_4x_a2c. */
TEST_F(M5DepthExport, AlphaToCoverageFoldsIntoSampleWords)
{
   for (unsigned samples : {2u, 4u}) {
      for (unsigned zs : {1u, 2u, 3u}) {
         SCOPED_TRACE(::testing::Message() << samples << " samples, zs=" << zs);
         nir_shader *s = build_epilog({AGX_CHIP_G17S, samples, !!(zs & 1),
                                       !!(zs & 2), true});
         counts c = count_nir(s);
         EXPECT_EQ(c.demotes, 0u);
         EXPECT_EQ(c.coverage_updates, 0u);
         EXPECT_EQ(c.covered_sample, 0u);
         EXPECT_FALSE(s->info.fs.uses_discard);
         agx_shader_part part = {};
         ASSERT_TRUE(compile_m5(s, samples, &part)) << agx_last_compile_refusal();
         std::vector<std::string> m = mnemonics(part);
         EXPECT_EQ(count_of(m, "depthf"), samples);
         EXPECT_EQ(count_of(m, "movmsk"), 0u);
         EXPECT_EQ(count_of(m, "cndlt"), 0u);
         free(part.binary);
         ralloc_free(s);
      }
   }
}

/* Without a depth or stencil write, alpha-to-coverage keeps its kill, and a
 * single sample keeps its one word. */
TEST_F(M5DepthExport, OtherShapesUnchanged)
{
   nir_shader *s = build_epilog({AGX_CHIP_G17S, 4, false, false, true});
   counts c = count_nir(s);
   EXPECT_EQ(c.store_zs, 0u);
   EXPECT_EQ(c.transactions, 0u);
   EXPECT_TRUE(s->info.fs.uses_discard);
   ralloc_free(s);

   s = build_epilog({AGX_CHIP_G17S, 1, true, false, false});
   c = count_nir(s);
   EXPECT_EQ(c.store_zs, 1u);
   EXPECT_EQ(c.covered_sample, 0u);
   EXPECT_EQ(c.transactions, 0u);
   ralloc_free(s);

   /* Not M5: no covered-sample loop, and alpha coverage stays a separate
    * coverage update. */
   s = build_epilog({AGX_CHIP_G17P, 4, true, false, true});
   c = count_nir(s);
   EXPECT_EQ(c.covered_sample, 0u);
   EXPECT_EQ(c.transactions, 0u);
   EXPECT_GT(c.demotes + c.coverage_updates, 0u);
   ralloc_free(s);
}

TEST_F(M5DepthExport, SampleRateCoverageTestsBeforeColour)
{
   for (bool sample_shading : {false, true}) {
      for (unsigned samples : {2u, 4u}) {
         SCOPED_TRACE(::testing::Message() << samples << " samples"
                      << (sample_shading ? ", sample shaded" : ""));
         epilog_shape shape = {AGX_CHIP_G17S, samples, false, false, true};
         shape.sample_shading = sample_shading;
         nir_shader *s = build_epilog(shape);
         agx_preprocess_nir(s);
         counts c = count_nir(s);
         EXPECT_EQ(c.coverage_updates, 1u);

         agx_shader_part part = {};
         ASSERT_TRUE(compile_m5(s, samples, &part, true))
            << agx_last_compile_refusal();
         std::vector<std::string> m = mnemonics(part);
         EXPECT_EQ(count_of(m, "movmsk"), 1u);
         EXPECT_LT(index_of(m, "movmsk"), index_of(m, "lstcol"));
         free(part.binary);
         ralloc_free(s);
      }
   }
}

TEST_F(M5DepthExport, BlendedAlphaCoverageTestsOncePerPixel)
{
   for (bool sample_shading : {false, true}) {
      for (unsigned samples : {2u, 4u, 8u}) {
         if (sample_shading && samples == 8)
            continue;
         SCOPED_TRACE(::testing::Message() << samples << " samples"
                      << (sample_shading ? ", sample shaded" : ""));
         epilog_shape shape = {AGX_CHIP_G17S, samples, false, false, true};
         shape.blend = true;
         shape.sample_shading = sample_shading;
         nir_shader *s = build_epilog(shape);
         agx_preprocess_nir(s);
         counts c = count_nir(s);
         EXPECT_EQ(c.coverage_updates, 1u);

         agx_shader_part part = {};
         ASSERT_TRUE(compile_m5(s, samples, &part, true))
            << agx_last_compile_refusal();
         std::vector<std::string> m = mnemonics(part);
         EXPECT_EQ(count_of(m, "lldcol"), samples);
         EXPECT_EQ(count_of(m, "movmsk"), 1u);
         EXPECT_LT(index_of(m, "movmsk"), index_of(m, "lldcol"));
         free(part.binary);
         ralloc_free(s);
      }
   }
}

/* Native blend_depth_{2,4,8}x and a2cb_depth_{2,4,8}x: a blended fragment
 * that writes depth finishes its depth words before blending any sample. */
TEST_F(M5DepthExport, BlendedDepthWritesPrecedeTheBlend)
{
   for (bool a2c : {false, true}) {
      for (unsigned samples : {2u, 4u, 8u}) {
         SCOPED_TRACE(::testing::Message() << samples << " samples"
                      << (a2c ? ", alpha to coverage" : ""));
         epilog_shape shape = {AGX_CHIP_G17S, samples, true, false, a2c};
         shape.blend = true;
         nir_shader *s = build_epilog(shape);
         agx_preprocess_nir(s);
         counts c = count_nir(s);
         EXPECT_EQ(c.store_zs, 1u);
         EXPECT_EQ(c.covered_sample, a2c ? 0u : 1u);
         EXPECT_EQ(c.coverage_updates, 0u);

         agx_shader_part part = {};
         ASSERT_TRUE(compile_m5(s, samples, &part, true))
            << agx_last_compile_refusal();
         std::vector<std::string> m = mnemonics(part);
         EXPECT_EQ(count_of(m, "lldcol"), samples);
         EXPECT_EQ(count_of(m, "depthf"), a2c ? samples : 1u);
         EXPECT_EQ(count_of(m, "movmsk"), 0u);
         int last_depthf = -1;
         for (unsigned i = 0; i < m.size(); ++i) {
            if (m[i] == "depthf")
               last_depthf = i;
         }
         EXPECT_LT(last_depthf, index_of(m, "lldcol"));
         free(part.binary);
         ralloc_free(s);
      }
   }
}

TEST_F(M5DepthExport, UniformBlendPublishesCompletePixelProtocol)
{
   epilog_shape shape = {AGX_CHIP_G17S, 1, false, false, false};
   shape.blend = true;
   nir_shader *s = build_epilog(shape);
   agx_preprocess_nir(s);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.m5_uniform_blend = true;
   key.fs.nr_samples = 1;
   BITSET_SET(key.g17p_tile_partitions, 0);
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(s, &key, &part))
      << agx_last_compile_refusal();
   EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
   EXPECT_TRUE(part.info.agx3_colour_link.rt_query_wait_unsupported);
   EXPECT_FALSE(part.info.agx3_colour_link.rt_query_write_acquired);
   EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 1u);
   const uint8_t entry[] = {0x07, 0x30, 0x00, 0x05, 0x00, 0x00};
   const uint8_t release[] = {0x67, 0x10, 0x00, 0x04, 0x00, 0x00};
   const uint8_t *begin = (uint8_t *)part.binary + part.info.main_offset;
   const uint8_t *end = begin + part.info.main_size;
   auto first = std::search(begin, end, std::begin(entry), std::end(entry));
   auto last = std::search(begin, end, std::begin(release), std::end(release));
   EXPECT_NE(first, end);
   EXPECT_NE(last, end);
   EXPECT_LT(first, last);
   auto ops = mnemonics(part);
   EXPECT_EQ(count_of(ops, "lldcol"), 1u);
   EXPECT_EQ(count_of(ops, "lstcol"), 1u);
   free(part.binary);
   ralloc_free(s);
}

TEST_F(M5DepthExport, UniformBlendRejectsOtherTransactions)
{
   for (unsigned test = 0; test < 4; ++test) {
      SCOPED_TRACE(test);
      epilog_shape shape = {AGX_CHIP_G17S, test == 0 ? 2u : 1u,
                            test == 1, false, false};
      shape.blend = true;
      nir_shader *s = build_epilog(shape);
      agx_preprocess_nir(s);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      key.no_preamble = true;
      key.m5_uniform_blend = true;
      key.fs.nr_samples = shape.samples;
      key.secondary = test == 2;
      key.no_stop = test == 3;
      BITSET_SET(key.g17p_tile_partitions, 0);
      agx_shader_part part = {};
      EXPECT_FALSE(agx_compile_shader_nir(s, &key, &part));
      free(part.binary);
      ralloc_free(s);
   }
}

TEST_F(M5DepthExport, PackedRgba8BackgroundClearSkipsFltnorm)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 packed RGBA8 clear");
   nir_def *packed = nir_load_preamble(&b, 1, 32, 4);
   nir_store_local_pixel_agx(&b, packed, nir_imm_intN_t(&b, 0xFF, 16),
                             nir_undef(&b, 2, 16), .base = 0,
                             .write_mask = 0x1, .format = PIPE_FORMAT_R32_UINT,
                             .target = 0);
   agx_preprocess_nir(b.shader);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.fs.ignore_tib_dependencies = true;
   key.fs.background = true;
   key.fs.nr_samples = 1;
   key.reserved_preamble = 12;
   key.no_preamble = true;

   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
      << agx_last_compile_refusal();
   EXPECT_TRUE(part.info.agx3_staging_known);
   EXPECT_GT(part.info.nr_gprs, 0u);
   EXPECT_EQ(part.info.push_count, 12u);
   EXPECT_FALSE(part.info.uses_txf);
   EXPECT_FALSE(part.info.has_preamble);
   EXPECT_EQ(part.info.scratch_size, 0u);
   auto ops = mnemonics(part);
   EXPECT_EQ(count_of(ops, "fltnorm"), 0u);
   EXPECT_EQ(count_of(ops, "lstcol"), 1u);
   free(part.binary);
   ralloc_free(b.shader);
}

TEST_F(M5DepthExport, InlineBlendPublishesCompletePixelProtocol)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R10G10B10A2_UNORM,
   };
   for (enum pipe_format fmt : formats) {
      SCOPED_TRACE(util_format_short_name(fmt));
      epilog_shape shape = {AGX_CHIP_G17S, 1, false, false, false};
      shape.blend = true;
      shape.format = fmt;
      nir_shader *s = build_epilog(shape);
      agx_preprocess_nir(s);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      key.no_preamble = true;
      key.fs.nr_samples = 1;
      BITSET_SET(key.g17p_tile_partitions, 0);
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(s, &key, &part))
         << agx_last_compile_refusal();
      EXPECT_TRUE(part.info.agx3_colour_link.entry_emitted);
      EXPECT_FALSE(part.info.agx3_colour_link.rt_query_write_acquired);
      EXPECT_EQ(part.info.agx3_colour_link.colour_classes, 1u);
      const uint8_t entry[] = {0x07, 0x30, 0x00, 0x05, 0x00, 0x00};
      const uint8_t release[] = {0x67, 0x10, 0x00, 0x04, 0x00, 0x00};
      const uint8_t *code = (uint8_t *)part.binary + part.info.main_offset;
      const size_t n = part.info.main_size;
      ASSERT_GE(n, 16u);
      ASSERT_LE(part.info.agx3_colour_link.entry_offset + sizeof(entry), n);
      EXPECT_EQ(memcmp(code + part.info.agx3_colour_link.entry_offset, entry,
                       sizeof(entry)), 0);
      EXPECT_EQ(memcmp(code + n - 10, release, sizeof(release)), 0);
      EXPECT_EQ(code[n - 4], 0x0e);
      auto ops = mnemonics(part);
      EXPECT_EQ(count_of(ops, "lldcol"), 1u);
      EXPECT_EQ(count_of(ops, "lstcol"), 1u);
      free(part.binary);
      ralloc_free(s);
   }
}

TEST_F(M5DepthExport, InlineBlendKeepsVaryingInterpBeforeTileLoad)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 inline blend interp order");
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   b.shader->info.outputs_read = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   b.shader->info.fs.uses_fbfetch_output = true;
   b.shader->info.io_lowered = true;
   nir_def *bary = nir_load_barycentric_pixel(&b, 32,
                                              .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *src = nir_load_interpolated_input(&b, 4, 32, bary, nir_imm_int(&b, 0),
                                              .base = 0, .component = 0,
                                              .io_semantics = {
                                                 .location = VARYING_SLOT_VAR0,
                                                 .num_slots = 1,
                                              });
   nir_def *mask = nir_imm_intN_t(&b, 0xff, 16);
   nir_def *dst = nir_load_local_pixel_agx(&b, 4, 32, mask,
                                           .base = 0,
                                           .format = PIPE_FORMAT_R8_UNORM,
                                           .target = 0);
   nir_def *a = nir_fsat(&b, nir_channel(&b, src, 3));
   nir_def *inv_a = nir_fsub(&b, nir_imm_float(&b, 1.0f), a);
   nir_def *blended = nir_ffma(&b, nir_fsat(&b, src), a, nir_fmul(&b, dst, inv_a));
   nir_store_local_pixel_agx(&b, blended, mask, nir_undef(&b, 2, 16),
                             .base = 0, .write_mask = 0xf,
                             .format = PIPE_FORMAT_R8_UNORM, .target = 0);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.fs.nr_samples = 1;
   BITSET_SET(key.g17p_tile_partitions, 0);
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
      << agx_last_compile_refusal();
   auto ops = mnemonics(part);
   int last_interp = -1, first_lldcol = -1;
   for (size_t i = 0; i < ops.size(); ++i) {
      if (ops[i].find("interp") != std::string::npos)
         last_interp = (int)i;
      if (first_lldcol < 0 && ops[i] == "lldcol")
         first_lldcol = (int)i;
   }
   ASSERT_GE(last_interp, 0);
   ASSERT_GE(first_lldcol, 0);
   EXPECT_LT(last_interp, first_lldcol);
   free(part.binary);
   ralloc_free(b.shader);
}
