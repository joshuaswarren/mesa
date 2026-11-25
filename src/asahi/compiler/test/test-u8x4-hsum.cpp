/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <vector>
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

namespace {

static nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.internal = false;
   return b;
}

TEST(U8x4Hsum, SwarByteSumPositiveAndNegativeCases)
{
   /* mode 0: complete dual 8x4 packed R8_UINT horizontal sum -> rewrites and
    *         compiles zero-copy on AGX3C with zero scratch.
    * mode 1: partial 3-channel sum -> must reject.
    * mode 2: extra external use of a loaded channel -> must reject. */
   for (int mode = 0; mode < 3; ++mode) {
      nir_builder b = make_compute_shader("u8x4_hsum");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *acc = nir_load_preamble(&b, 1, 32, .base = 10);

      nir_def *v0 =
         nir_load_agx(&b, 4, 8, base, idx, .format = PIPE_FORMAT_R8_UINT);
      nir_def *v1 = nir_load_agx(&b, 4, 8, base, nir_iadd_imm(&b, idx, 4),
                                 .format = PIPE_FORMAT_R8_UINT);
      acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v0, 0)));
      acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v0, 1)));
      acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v0, 2)));
      if (mode != 1)
         acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v0, 3)));
      acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v1, 0)));
      acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v1, 1)));
      acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v1, 2)));
      if (mode != 1)
         acc = nir_iadd(&b, acc, nir_u2u32(&b, nir_channel(&b, v1, 3)));
      nir_store_agx(&b, acc, out, idx, .format = PIPE_FORMAT_R32_UINT);
      if (mode == 2) {
         nir_store_agx(&b, nir_channel(&b, v0, 0), out,
                       nir_iadd_imm(&b, idx, 16),
                       .format = PIPE_FORMAT_R8_UINT);
         nir_store_agx(&b, nir_channel(&b, v1, 0), out,
                       nir_iadd_imm(&b, idx, 20),
                       .format = PIPE_FORMAT_R8_UINT);
      }

      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      bool changed = agx_nir_opt_u8x4_hsum(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum");

      if (mode == 0) {
         EXPECT_TRUE(changed);
         unsigned packs = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_alu &&
                      nir_instr_as_alu(instr)->op == nir_op_pack_32_4x8)
                     packs++;
               }
            }
         }
         EXPECT_EQ(packs, 2u);

         struct agx_shader_key key = {};
         key.dev.chip = AGX_CHIP_G17S;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.no_preamble = true;
         agx_preprocess_nir(b.shader);
         struct agx_shader_part part = {};
         EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         EXPECT_EQ(part.info.scratch_size, 0u);
         free(part.binary);
      } else {
         EXPECT_FALSE(changed);
      }
      ralloc_free(b.shader);
   }
}

TEST(U8x4Hsum, Pack324x8SwizzledFallbackCompiles)
{
   nir_builder b = make_compute_shader("pack_32_4x8_swizzled");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);

   nir_def *v =
      nir_load_agx(&b, 4, 8, base, idx, .format = PIPE_FORMAT_R8_UINT);
   unsigned rev[4] = {3, 2, 1, 0};
   nir_def *swz = nir_swizzle(&b, v, rev, 4);
   nir_def *packed = nir_pack_32_4x8(&b, swz);
   nir_store_agx(&b, packed, out, idx, .format = PIPE_FORMAT_R32_UINT);

   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.no_preamble = true;
   agx_preprocess_nir(b.shader);
   struct agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_EQ(part.info.scratch_size, 0u);
   free(part.binary);
   ralloc_free(b.shader);
}

TEST(U8x4Hsum, ExactValuesCarryChunkFlushAndMixedTree)
{
   auto build_balanced_sum = [](nir_builder *b,
                                std::vector<nir_def *> terms) -> nir_def * {
      while (terms.size() > 1) {
         std::vector<nir_def *> next;
         next.reserve((terms.size() + 1) / 2);
         for (size_t i = 0; i < terms.size(); i += 2) {
            if (i + 1 < terms.size())
               next.push_back(nir_iadd(b, terms[i], terms[i + 1]));
            else
               next.push_back(terms[i]);
         }
         terms = std::move(next);
      }
      return terms[0];
   };

   auto make_imm_u8x4 = [](nir_builder *b, uint8_t c0, uint8_t c1, uint8_t c2,
                           uint8_t c3) -> nir_def * {
      nir_const_value v[4];
      v[0] = nir_const_value_for_uint(c0, 8);
      v[1] = nir_const_value_for_uint(c1, 8);
      v[2] = nir_const_value_for_uint(c2, 8);
      v[3] = nir_const_value_for_uint(c3, 8);
      return nir_build_imm(b, 4, 8, v);
   };

   /* 1. Max-byte (0xff) inputs: verify exact SWAR sum with zero 16-bit lane
    *    carry/truncation against CPU reference. */
   {
      nir_builder b = make_compute_shader("u8x4_hsum_0xff");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);

      nir_def *v0 =
         nir_load_agx(&b, 4, 8, base, idx, .format = PIPE_FORMAT_R8_UINT);
      nir_def *v1 = nir_load_agx(&b, 4, 8, base, nir_iadd_imm(&b, idx, 4),
                                 .format = PIPE_FORMAT_R8_UINT);
      std::vector<nir_def *> terms;
      for (unsigned c = 0; c < 4; ++c)
         terms.push_back(nir_u2u32(&b, nir_channel(&b, v0, c)));
      for (unsigned c = 0; c < 4; ++c)
         terms.push_back(nir_u2u32(&b, nir_channel(&b, v1, c)));
      nir_def *sum = build_balanced_sum(&b, std::move(terms));
      nir_intrinsic_instr *store =
         nir_store_agx(&b, sum, out, idx, .format = PIPE_FORMAT_R32_UINT);

      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_opt_u8x4_hsum(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum 0xff");

      b.cursor = nir_before_impl(b.impl);
      nir_def_rewrite_uses(v0, make_imm_u8x4(&b, 0xff, 0xff, 0xff, 0xff));
      nir_def_rewrite_uses(v1, make_imm_u8x4(&b, 0xff, 0xfe, 0x01, 0x80));
      while (nir_opt_constant_folding(b.shader))
         ;
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum 0xff folded");
      ASSERT_TRUE(nir_src_is_const(store->src[0]));
      const uint32_t expected =
         (0xffu + 0xffu + 0xffu + 0xffu) + (0xffu + 0xfeu + 0x01u + 0x80u);
      EXPECT_EQ(nir_src_as_uint(store->src[0]), expected);
      ralloc_free(b.shader);
   }

   /* 2. 128-word chunk flush (> 128 words = 512 bytes): 130 all-0xff words
    *    would overflow a 16-bit lane (130 * 510 = 66300 > 65535) without the
    *    128-word flush. Verify chunk flush fires and folded sum is exact. */
   {
      constexpr unsigned n_words = 130;
      nir_builder b = make_compute_shader("u8x4_hsum_chunk_flush");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);

      std::vector<nir_def *> loads;
      std::vector<nir_def *> terms;
      loads.reserve(n_words);
      terms.reserve(n_words * 4);
      for (unsigned w = 0; w < n_words; ++w) {
         nir_def *v = nir_load_agx(&b, 4, 8, base,
                                   nir_iadd_imm(&b, idx, w * 4),
                                   .format = PIPE_FORMAT_R8_UINT);
         loads.push_back(v);
         for (unsigned c = 0; c < 4; ++c)
            terms.push_back(nir_u2u32(&b, nir_channel(&b, v, c)));
      }
      nir_def *sum = build_balanced_sum(&b, std::move(terms));
      nir_intrinsic_instr *store =
         nir_store_agx(&b, sum, out, idx, .format = PIPE_FORMAT_R32_UINT);

      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_opt_u8x4_hsum(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum chunk flush");

      unsigned packs = 0, unpack_x = 0, unpack_y = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_op op = nir_instr_as_alu(instr)->op;
               if (op == nir_op_pack_32_4x8)
                  packs++;
               else if (op == nir_op_unpack_32_2x16_split_x)
                  unpack_x++;
               else if (op == nir_op_unpack_32_2x16_split_y)
                  unpack_y++;
            }
         }
      }
      EXPECT_EQ(packs, n_words);
      EXPECT_EQ(unpack_x, 2u);
      EXPECT_EQ(unpack_y, 2u);

      b.cursor = nir_before_impl(b.impl);
      nir_def *all_ff = make_imm_u8x4(&b, 0xff, 0xff, 0xff, 0xff);
      for (nir_def *ld : loads)
         nir_def_rewrite_uses(ld, all_ff);
      while (nir_opt_constant_folding(b.shader))
         ;
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum chunk flush folded");
      ASSERT_TRUE(nir_src_is_const(store->src[0]));
      EXPECT_EQ(nir_src_as_uint(store->src[0]), n_words * 4u * 255u);
      ralloc_free(b.shader);
   }

   /* 3. Mixed tree: rewritten u8x4 words combined with leftover scalar u2u32
    *    leaves (partial 2-channel load) and a 32-bit scalar accumulator. */
   {
      nir_builder b = make_compute_shader("u8x4_hsum_mixed");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *init = nir_load_preamble(&b, 1, 32, .base = 10);

      nir_def *v0 =
         nir_load_agx(&b, 4, 8, base, idx, .format = PIPE_FORMAT_R8_UINT);
      nir_def *v1 = nir_load_agx(&b, 4, 8, base, nir_iadd_imm(&b, idx, 4),
                                 .format = PIPE_FORMAT_R8_UINT);
      nir_def *v2 = nir_load_agx(&b, 4, 8, base, nir_iadd_imm(&b, idx, 8),
                                 .format = PIPE_FORMAT_R8_UINT);
      nir_def *v_part = nir_load_agx(&b, 4, 8, base, nir_iadd_imm(&b, idx, 12),
                                     .format = PIPE_FORMAT_R8_UINT);

      std::vector<nir_def *> terms;
      terms.push_back(init);
      for (unsigned c = 0; c < 4; ++c)
         terms.push_back(nir_u2u32(&b, nir_channel(&b, v0, c)));
      terms.push_back(nir_u2u32(&b, nir_channel(&b, v_part, 0)));
      for (unsigned c = 0; c < 4; ++c)
         terms.push_back(nir_u2u32(&b, nir_channel(&b, v1, c)));
      terms.push_back(nir_u2u32(&b, nir_channel(&b, v_part, 2)));
      for (unsigned c = 0; c < 4; ++c)
         terms.push_back(nir_u2u32(&b, nir_channel(&b, v2, c)));

      nir_def *sum = build_balanced_sum(&b, std::move(terms));
      nir_intrinsic_instr *store =
         nir_store_agx(&b, sum, out, idx, .format = PIPE_FORMAT_R32_UINT);

      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_opt_u8x4_hsum(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum mixed");

      unsigned packs = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   nir_instr_as_alu(instr)->op == nir_op_pack_32_4x8)
                  packs++;
            }
         }
      }
      EXPECT_EQ(packs, 3u);

      b.cursor = nir_before_impl(b.impl);
      nir_def_rewrite_uses(init, nir_imm_int(&b, 0x12345u));
      nir_def_rewrite_uses(v0, make_imm_u8x4(&b, 0xff, 0x80, 0x7f, 0x01));
      nir_def_rewrite_uses(v1, make_imm_u8x4(&b, 0xfe, 0x00, 0xaa, 0x55));
      nir_def_rewrite_uses(v2, make_imm_u8x4(&b, 0x12, 0x34, 0x56, 0x78));
      nir_def_rewrite_uses(v_part, make_imm_u8x4(&b, 0xee, 0xdd, 0xcc, 0xbb));
      while (nir_opt_constant_folding(b.shader))
         ;
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8x4 hsum mixed folded");
      ASSERT_TRUE(nir_src_is_const(store->src[0]));
      const uint32_t expected = 0x12345u +
                                (0xffu + 0x80u + 0x7fu + 0x01u) +
                                (0xfeu + 0x00u + 0xaau + 0x55u) +
                                (0x12u + 0x34u + 0x56u + 0x78u) +
                                (0xeeu + 0xccu);
      EXPECT_EQ(nir_src_as_uint(store->src[0]), expected);
      ralloc_free(b.shader);
   }
}

static unsigned
count_alu(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

/* A sum in the merge block of an if, outside any loop: the mask constant must
 * go after the block's phis, which stay first in the block. */
TEST(U8x4Hsum, MergeBlockWithPhiKeepsPhisFirst)
{
   nir_builder b = make_compute_shader("u8x4_hsum_merge_phi");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);
   nir_def *a = nir_load_preamble(&b, 1, 32, .base = 10);
   nir_def *c = nir_load_preamble(&b, 1, 32, .base = 12);

   nir_push_if(&b, nir_ieq_imm(&b, idx, 0));
   nir_def *then_val = nir_iadd_imm(&b, a, 1);
   nir_push_else(&b, NULL);
   nir_def *else_val = nir_iadd_imm(&b, c, 2);
   nir_pop_if(&b, NULL);
   nir_def *phi = nir_if_phi(&b, then_val, else_val);
   nir_block *merge = nir_def_instr(phi)->block;

   nir_def *v0 =
      nir_load_agx(&b, 4, 8, base, idx, .format = PIPE_FORMAT_R8_UINT);
   nir_def *v1 = nir_load_agx(&b, 4, 8, base, nir_iadd_imm(&b, idx, 4),
                              .format = PIPE_FORMAT_R8_UINT);
   nir_def *sum = phi;
   for (unsigned i = 0; i < 4; ++i)
      sum = nir_iadd(&b, sum, nir_u2u32(&b, nir_channel(&b, v0, i)));
   for (unsigned i = 0; i < 4; ++i)
      sum = nir_iadd(&b, sum, nir_u2u32(&b, nir_channel(&b, v1, i)));
   nir_store_agx(&b, sum, out, idx, .format = PIPE_FORMAT_R32_UINT);

   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_opt_u8x4_hsum(b.shader));
   nir_validate_shader(b.shader, "u8x4 hsum merge phi");
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "u8x4 hsum merge phi dce");

   EXPECT_EQ(count_alu(b.shader, nir_op_pack_32_4x8), 2u);
   EXPECT_EQ(nir_block_first_instr(merge)->type, nir_instr_type_phi);
   EXPECT_EQ(nir_first_phi_in_block(merge), nir_instr_as_phi(nir_def_instr(phi)));
   ralloc_free(b.shader);
}

/* A leftover leaf that reads a non-zero component of a multi-component 32-bit
 * def keeps that component when the sum is rebuilt. */
TEST(U8x4Hsum, MixedTreeKeepsSwizzledVectorComponent)
{
   nir_builder b = make_compute_shader("u8x4_hsum_vector_leaf");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);
   nir_def *vec = nir_load_preamble(&b, 3, 32, .base = 10);

   nir_def *v0 =
      nir_load_agx(&b, 4, 8, base, idx, .format = PIPE_FORMAT_R8_UINT);
   nir_def *sum = nir_channel(&b, vec, 1);
   for (unsigned i = 0; i < 4; ++i)
      sum = nir_iadd(&b, sum, nir_u2u32(&b, nir_channel(&b, v0, i)));
   nir_intrinsic_instr *store =
      nir_store_agx(&b, sum, out, idx, .format = PIPE_FORMAT_R32_UINT);

   /* Copy propagation turns the channel move into a .y swizzle on the iadd
    * source, the shape the pass must keep. */
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_opt_u8x4_hsum(b.shader));
   nir_validate_shader(b.shader, "u8x4 hsum vector leaf");
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "u8x4 hsum vector leaf dce");
   EXPECT_EQ(count_alu(b.shader, nir_op_pack_32_4x8), 1u);
   ASSERT_EQ(store->src[0].ssa->num_components, 1u);

   b.cursor = nir_before_impl(b.impl);
   nir_def_rewrite_uses(vec, nir_imm_ivec3(&b, 0x1000, 0x2000, 0x3000));
   nir_const_value bytes[4] = {
      nir_const_value_for_uint(0x01, 8), nir_const_value_for_uint(0x80, 8),
      nir_const_value_for_uint(0xff, 8), nir_const_value_for_uint(0x7e, 8)};
   nir_def_rewrite_uses(v0, nir_build_imm(&b, 4, 8, bytes));
   while (nir_opt_constant_folding(b.shader))
      ;
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "u8x4 hsum vector leaf folded");
   ASSERT_TRUE(nir_src_is_const(store->src[0]));
   EXPECT_EQ(nir_src_as_uint(store->src[0]),
             0x2000u + 0x01u + 0x80u + 0xffu + 0x7eu);
   ralloc_free(b.shader);
}

} // namespace
