/* SPDX-License-Identifier: MIT */
#include <array>
#include <cstring>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "agx3_binder.h"
#include "agx3_alu_core.h"
#include "agx3_cf_core.h"
#include "agx3_uniform_alu.h"
#include "agx3_stage_core.h"
#include "agx3_isel_core.h"
#include "agx3_tex_core.h"
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

TEST(AGX3UniformALU, PackedUniformLoadPointerBoundaries)
{
   struct test_case {
      enum agx_isa_encoding encoding;
      unsigned low, high;
      bool modified, admitted;
   };
   const test_case cases[] = {
      {AGX_ISA_ENCODING_AGX3C, 252, 254, false, true},
      {AGX_ISA_ENCODING_AGX3C, 256, 258, false, true},
      {AGX_ISA_ENCODING_AGX3C, 448, 450, false, true},
      {AGX_ISA_ENCODING_AGX3C, 476, 478, false, true},
      {AGX_ISA_ENCODING_AGX3C, 508, 510, false, true},
      {AGX_ISA_ENCODING_AGX3C, 512, 514, false, false},
      {AGX_ISA_ENCODING_AGX3C, 250, 252, false, false},
      {AGX_ISA_ENCODING_AGX3C, 448, 452, false, false},
      {AGX_ISA_ENCODING_AGX3C, 450, 448, false, false},
      {AGX_ISA_ENCODING_AGX3C, 448, 450, true, false},
      {AGX_ISA_ENCODING_AGX3B, 448, 450, false, false},
   };
   for (const auto &t : cases) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index pointer = agx_temp(b->shader, AGX_SIZE_64);
      agx_instr *collect = agx_collect_to(b, pointer, 2);
      /* Include out-of-domain indices to test rejection by the optimizer.
       * The checked production constructor correctly rejects those inputs. */
      collect->src[0] = agx_uniform(0, AGX_SIZE_32);
      collect->src[0].value = t.low;
      collect->src[0].neg = t.modified;
      collect->src[1] = agx_uniform(0, AGX_SIZE_32);
      collect->src[1].value = t.high;
      agx_instr *load = agx_device_load_to(
         b, agx_temp(b->shader, AGX_SIZE_32), pointer, agx_immediate(0),
         AGX_FORMAT_I32, 1, 0, false);
      agx_optimizer_forward(b->shader);
      if (t.admitted) {
         EXPECT_EQ(load->src[0].type, AGX_INDEX_UNIFORM);
         EXPECT_EQ(load->src[0].value, t.low);
         EXPECT_EQ(load->src[0].size, AGX_SIZE_64);
      } else {
         EXPECT_EQ(load->src[0].type, AGX_INDEX_NORMAL);
         EXPECT_EQ(load->src[0].value, pointer.value);
      }
      ralloc_free(mem);
   }
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3C,
      AGX_OPCODE_DEVICE_STORE, 1, 448, AGX_SIZE_64));
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3C,
      AGX_OPCODE_ATOMIC, 1, 448, AGX_SIZE_64));
}

static uint32_t
uniform_difference_zero_bits(bool relax)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_temp(b->shader, AGX_SIZE_32);
   agx_index multiplier = agx_temp(b->shader, AGX_SIZE_32);
   agx_index difference = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *sub = agx_fadd_to(b, difference, agx_uniform(12, AGX_SIZE_32),
                               agx_neg(input));
   sub->no_signed_zero = relax;
   agx_index product = agx_fmul(b, difference, multiplier);
   agx_lower_uniform_sources(b->shader);

   std::vector<float> values(b->shader->alloc, 0.0f);
   values[input.value] = values[multiplier.value] = 1.0f;
   auto read = [&](agx_index x) {
      float f = x.type == AGX_INDEX_UNIFORM ? 1.0f : values[x.value];
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      if (x.abs) bits &= 0x7fffffff;
      if (x.neg) bits ^= 0x80000000;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
   };
   agx_foreach_instr_global(b->shader, I) {
      volatile float a = read(I->src[0]);
      volatile float result;
      if (I->op == AGX_OPCODE_MOV) {
         result = a;
      } else {
         volatile float c = read(I->src[1]);
         assert(I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL);
         result = I->op == AGX_OPCODE_FADD ? a + c : a * c;
      }
      values[I->dest[0].value] = result;
   }
   uint32_t bits;
   std::memcpy(&bits, &values[product.value], sizeof(bits));
   ralloc_free(mem);
   return bits;
}

/* u0 - u1 on M5: copying u0 leaves the native R + -U addition, so u1 stays a
 * uniform operand instead of being copied as well. */
TEST(AGX3UniformALU, M5UniformDifferenceCopiesOneUniform)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_instr *sub = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                agx_uniform(12, AGX_SIZE_32),
                                agx_neg(agx_uniform(14, AGX_SIZE_32)));
   agx_unit_test(b, sub->dest[0]);
   agx_lower_uniform_sources(b->shader);

   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM) {
         EXPECT_EQ(I->src[0].value, 12u);
         ++copies;
      }
   }
   EXPECT_EQ(copies, 1u);
   EXPECT_EQ(sub->src[0].type, AGX_INDEX_NORMAL);
   EXPECT_FALSE(sub->src[0].neg);
   EXPECT_EQ(sub->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(sub->src[1].value, 14u);
   EXPECT_TRUE(sub->src[1].neg);
   ralloc_free(mem);
}

TEST(AGX3UniformALU, UniformDifferencePreservesPositiveZero)
{
   EXPECT_EQ(uniform_difference_zero_bits(false), 0u);
}

TEST(AGX3UniformALU, UniformDifferenceMayNegateRelaxedZero)
{
   EXPECT_EQ(uniform_difference_zero_bits(true), 0x80000000u);
}

TEST(AGX3UniformALU, CSEKeepsDifferentSignedZeroContracts)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index input = agx_temp(b->shader, AGX_SIZE_32);
   agx_index strict = agx_temp(b->shader, AGX_SIZE_32);
   agx_index relaxed = agx_temp(b->shader, AGX_SIZE_32);
   agx_fadd_to(b, strict, agx_uniform(12, AGX_SIZE_32), agx_neg(input));
   agx_fadd_to(b, relaxed, agx_uniform(12, AGX_SIZE_32),
               agx_neg(input))->no_signed_zero = true;
   agx_instr *use = agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32),
                               strict, relaxed);
   agx_opt_cse(b->shader);
   EXPECT_NE(use->src[0].value, use->src[1].value);
   ralloc_free(mem);
}

/* Back-to-back M5 reads of four channels each; the second writes r0..r3.
 * Returns how often the packed program carries an identity wait on r0 and on
 * the last read's coordinates, the carrier a retirement of a recycled operand
 * bank must use: the r0 carrier waits only for the first read, while the
 * second still writes r0. That was the stencil resolve of
 * dEQP-VK.renderpasses.renderpass2.depth_stencil_resolve.*.stencil_min.
 */
static void
retirement_carriers(bool multisample, unsigned reads, unsigned *r0,
                    unsigned *coords)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 125;
   const unsigned destinations[] = {8, 0, 16, 24, 32, 40, 48, 56, 64};
   const unsigned stride = multisample ? 6 : 4;
   ASSERT_LE(reads, ARRAY_SIZE(destinations));
   for (unsigned i = 0; i < reads; ++i) {
      agx_index coord = agx_register(72 + stride * i, AGX_SIZE_32);
      coord.channels_m1 = multisample ? 2 : 1;
      agx_index dest = agx_register(destinations[i], AGX_SIZE_32);
      dest.channels_m1 = 3;
      if (multisample) {
         agx_texture_load_to(b, dest, coord, agx_immediate(0),
                             agx_immediate(0), agx_immediate(i),
                             agx_immediate(0), agx_null(), AGX_DIM_2D_MS,
                             AGX_LOD_MODE_AUTO_LOD, 15, false, false);
      } else {
         agx_image_load_to(b, dest, coord, agx_immediate(0),
                           agx_immediate(0), agx_immediate(i),
                           agx_immediate(0), agx_null(), AGX_DIM_2D,
                           AGX_LOD_MODE_AUTO_LOD, 15, false, false, false);
      }
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   const uint8_t *bytes = static_cast<const uint8_t *>(binary.data);
   const unsigned carriers[2] = {0, 72 + stride * (reads - 1)};
   unsigned *counts[2] = {r0, coords};
   for (unsigned c = 0; c < 2; ++c) {
      uint8_t expected[AGX3_ALU_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3C,
                                            carriers[c], 1, expected, &length));
      *counts[c] = 0;
      for (unsigned i = 0; i + length <= binary.size; ++i)
         *counts[c] += std::memcmp(bytes + i, expected, length) == 0;
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(AGX3FetchBanks, MultisampleRetirementPreservesPendingR0)
{
   /* Non-array multisample reads take eight operand banks, one per
    * scoreboard slot: the ninth read takes the first read's slot and bank
    * once that read retires, and nothing waits on r0. */
   unsigned r0 = ~0u, coords = ~0u;
   retirement_carriers(true, 9, &r0, &coords);
   EXPECT_EQ(r0, 0u);
}

TEST(AGX3FetchBanks, ImageRetirementPreservesPendingR0)
{
   /* Image reads stage from half 16 and have six operand banks: the seventh
    * read recycles the first bank while the second read's r0 is in flight,
    * and carries the retirement on its own coordinates. */
   unsigned r0 = ~0u, coords = ~0u;
   retirement_carriers(false, 7, &r0, &coords);
   EXPECT_EQ(r0, 0u);
   EXPECT_EQ(coords, 1u);
}

static agx_instr *
split_test_fetch(agx_builder *b, agx_index dest, agx_index coords,
                 enum agx_dim dim, agx_index lod = agx_immediate(0))
{
   const bool ms = dim == AGX_DIM_2D_MS;
   return agx_texture_load_to(
      b, dest, coords, lod, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), dim,
      ms ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN, 15, false, false);
}

/* Rebuild a fetch with split coordinates, as agx3_split_texture_coordinates()
 * does, from physical registers. */
static void
split_test_registers(agx_builder *b, agx_instr *I, unsigned x, unsigned y,
                     unsigned sample)
{
   agx_instr *fetch = agx_alloc_instr(b, I->op, 1, 8);
   memcpy(&fetch->op, &I->op, sizeof(*I) - offsetof(agx_instr, op));
   fetch->nr_srcs = 8;
   fetch->dest[0] = I->dest[0];
   memcpy(fetch->src, I->src, 6 * sizeof(agx_index));
   fetch->src[0] = agx_register(x, AGX_SIZE_32);
   fetch->src[6] = agx_register(y, AGX_SIZE_32);
   fetch->src[7] = I->dim != AGX_DIM_2D ? agx_register(sample, AGX_SIZE_32)
                                        : agx_null();
   fetch->agx3_split_coords = true;
   agx_cursor cursor = agx_before_instr(I);
   agx_builder_insert(&cursor, fetch);
   agx_remove_instruction(I);
   b->cursor = agx_after_instr(fetch);
}

static std::vector<uint8_t>
split_test_pack(enum agx_dim dim, int split_x, unsigned y, unsigned sample,
                bool register_lod = false)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 95;
   agx_index coords = agx_register(20, AGX_SIZE_32);
   coords.channels_m1 = dim == AGX_DIM_2D ? 1 : 2;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_instr *I = split_test_fetch(
      b, dest, coords, dim,
      register_lod ? agx_register(100, AGX_SIZE_32) : agx_immediate(0));
   if (split_x >= 0)
      split_test_registers(b, I, split_x, y, sample);
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   std::vector<uint8_t> out;
   if (agx3_pack_binary(b->shader, &binary)) {
      const uint8_t *data = static_cast<const uint8_t *>(binary.data);
      out.assign(data, data + binary.size);
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
   return out;
}

/* Split coordinates in the registers a contiguous vector would occupy stage
 * exactly like the vector, and scattered ones still encode. */
TEST(AGX3FetchBanks, SplitCoordinatesStageLikeVector)
{
   for (enum agx_dim dim : {AGX_DIM_2D, AGX_DIM_2D_MS, AGX_DIM_2D_ARRAY}) {
      SCOPED_TRACE(dim);
      const std::vector<uint8_t> vector = split_test_pack(dim, -1, 0, 0);
      ASSERT_FALSE(vector.empty());
      EXPECT_EQ(split_test_pack(dim, 20, 22, 24), vector);
      const std::vector<uint8_t> scattered = split_test_pack(dim, 40, 20, 60);
      ASSERT_FALSE(scattered.empty());
      EXPECT_EQ(scattered.size(), vector.size());
      EXPECT_NE(scattered, vector);
      /* The same register for two components still encodes. */
      EXPECT_FALSE(split_test_pack(dim, 40, 40, 40).empty());
      /* x and y may live above the seven-bit COPY range; the sample index
       * or layer may not. */
      EXPECT_FALSE(split_test_pack(dim, 200, 150, 60).empty());
      if (dim != AGX_DIM_2D) {
         EXPECT_TRUE(split_test_pack(dim, 20, 22, 200).empty());
      }
   }
   /* A register level stages beside the layer (2D array) or after the
    * coordinates (2D). */
   for (enum agx_dim dim : {AGX_DIM_2D, AGX_DIM_2D_ARRAY}) {
      SCOPED_TRACE(dim);
      const std::vector<uint8_t> vector = split_test_pack(dim, -1, 0, 0, true);
      ASSERT_FALSE(vector.empty());
      EXPECT_EQ(split_test_pack(dim, 20, 22, 24, true), vector);
      EXPECT_FALSE(split_test_pack(dim, 40, 20, 60, true).empty());
      EXPECT_FALSE(split_test_pack(dim, 60, 20, 60, true).empty());
   }
}

TEST(AGX3FetchBanks, SplitCoordinatesReplaceCollects)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index x = agx_mov_imm(b, 32, 5), y = agx_mov_imm(b, 32, 7);
   agx_index samples[2] = {agx_mov_imm(b, 32, 0), agx_mov_imm(b, 32, 1)};
   for (unsigned i = 0; i < 2; ++i) {
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, coords, 3);
      collect->src[0] = x;
      collect->src[1] = y;
      collect->src[2] = samples[i];
      split_test_fetch(b, agx_vec_temp(b->shader, AGX_SIZE_32, 4), coords,
                       AGX_DIM_2D_MS);
   }
   /* A uniform component cannot be staged in place: keep the vector. */
   agx_index uniform_coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, uniform_coords, 2);
   collect->src[0] = x;
   collect->src[1] = agx_uniform(8, AGX_SIZE_32);
   split_test_fetch(b, agx_vec_temp(b->shader, AGX_SIZE_32, 4), uniform_coords,
                    AGX_DIM_2D);

   EXPECT_TRUE(agx3_split_texture_coordinates(b->shader));
   unsigned split = 0, vector = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_TEXTURE_LOAD)
         continue;
      if (!I->agx3_split_coords) {
         EXPECT_EQ(I->nr_srcs, 6);
         EXPECT_EQ(I->src[0].value, uniform_coords.value);
         ++vector;
         continue;
      }
      ASSERT_EQ(I->nr_srcs, 8);
      EXPECT_EQ(I->src[0].value, x.value);
      EXPECT_EQ(I->src[6].value, y.value);
      EXPECT_EQ(I->src[7].value, samples[split].value);
      EXPECT_EQ(I->src[3].type, AGX_INDEX_IMMEDIATE);
      ++split;
   }
   EXPECT_EQ(split, 2u);
   EXPECT_EQ(vector, 1u);

   /* Other encodings keep their coordinate vectors. */
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   EXPECT_FALSE(agx3_split_texture_coordinates(b->shader));
   ralloc_free(mem);
}

static agx_instr *
split_test_sample(agx_builder *b, agx_index dest, agx_index coords,
                  agx_index reference)
{
   const bool shadow = !agx_is_null(reference);
   return agx_texture_sample_to(
      b, dest, coords, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), reference, AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
      shadow ? 1 : 15, false, shadow, false, AGX_GATHER_NONE, false);
}

/* A compute 2D sample at level zero, with vector coordinates in r10..r11 or
 * split ones in r<x>/r<y> (word numbers), and a comparison reference in
 * r<reference> when that is not negative. */
static std::vector<uint8_t>
split_test_pack_sample(int split_x, unsigned y, int reference,
                       bool discard = false)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 127;
   agx_index coords = agx_register(20, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_instr *I = split_test_sample(
      b, dest, coords,
      reference >= 0 ? agx_register(2 * reference, AGX_SIZE_32) : agx_null());
   if (split_x >= 0) {
      split_test_registers(b, I, 2 * split_x, 2 * y, 0);
      agx_foreach_instr_global(b->shader, split) {
         if (split->agx3_split_coords)
            split->src[0].discard = split->src[6].discard = discard;
      }
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   std::vector<uint8_t> out;
   if (agx3_pack_binary(b->shader, &binary)) {
      const uint8_t *data = static_cast<const uint8_t *>(binary.data);
      out.assign(data, data + binary.size);
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
   return out;
}

/* Split 2D sample coordinates stage byte for byte like the vector in the same
 * registers, with or without a comparison reference; scattered and high
 * coordinates still encode. */
TEST(AGX3FetchBanks, SplitSampleCoordinatesStageLikeVector)
{
   for (int reference : {-1, 30}) {
      SCOPED_TRACE(reference);
      const std::vector<uint8_t> vector =
         split_test_pack_sample(-1, 0, reference);
      ASSERT_FALSE(vector.empty());
      EXPECT_EQ(split_test_pack_sample(10, 11, reference), vector);
      const std::vector<uint8_t> scattered =
         split_test_pack_sample(21, 9, reference);
      ASSERT_FALSE(scattered.empty());
      EXPECT_EQ(scattered.size(), vector.size());
      EXPECT_NE(scattered, vector);
      EXPECT_FALSE(split_test_pack_sample(21, 21, reference).empty());
      /* x and y may live above the seven-bit COPY range. */
      EXPECT_FALSE(split_test_pack_sample(100, 75, reference).empty());
   }
   /* A coordinate in the reference's register is refused, not staged after
    * the reference may have discarded it. */
   EXPECT_TRUE(split_test_pack_sample(30, 11, 30).empty());
}

/* x and y in one register: only the second COPY may discard it. */
TEST(AGX3FetchBanks, SplitSampleSharedRegisterDiscardsOnce)
{
   const std::vector<uint8_t> binary = split_test_pack_sample(21, 21, -1, true);
   ASSERT_FALSE(binary.empty());
   bool found = false;
   for (unsigned slot = 1; slot <= 8 && !found; ++slot) {
      for (unsigned bank = 0; bank < 64 && !found; bank += 4) {
         uint8_t x[AGX3_TEX_COPY_MAX_LENGTH], y[AGX3_TEX_COPY_MAX_LENGTH];
         uint8_t x_len = 0, y_len = 0;
         ASSERT_TRUE(agx3_tex_stage_copy_scalar_m5(bank, 42, slot, 0, true,
                                                   false, x, &x_len));
         ASSERT_TRUE(agx3_tex_stage_copy_scalar_m5(bank + 2, 42, slot, 0,
                                                   false, true, y, &y_len));
         for (unsigned i = 0; i + x_len + y_len <= binary.size(); ++i) {
            found |= std::memcmp(binary.data() + i, x, x_len) == 0 &&
                     std::memcmp(binary.data() + i + x_len, y, y_len) == 0;
         }
      }
   }
   EXPECT_TRUE(found);
}

TEST(AGX3FetchBanks, SplitSampleCoordinatesReplaceCollects)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index x = agx_mov_imm(b, 32, 5), y = agx_mov_imm(b, 32, 7);
   agx_index reference = agx_mov_imm(b, 32, 9);
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coords, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   /* One coordinate pair, three filtered taps: a plain, a comparison and a
    * comparison whose reference is also its x coordinate. */
   split_test_sample(b, agx_vec_temp(b->shader, AGX_SIZE_32, 4), coords,
                     agx_null());
   split_test_sample(b, agx_vec_temp(b->shader, AGX_SIZE_32, 4), coords,
                     reference);
   split_test_sample(b, agx_vec_temp(b->shader, AGX_SIZE_32, 4), coords, x);

   EXPECT_TRUE(agx3_split_texture_coordinates(b->shader));
   unsigned split = 0, vector = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE)
         continue;
      if (!I->agx3_split_coords) {
         EXPECT_EQ(I->nr_srcs, 6);
         EXPECT_EQ(I->src[0].value, coords.value);
         EXPECT_EQ(I->src[5].value, x.value);
         ++vector;
         continue;
      }
      ASSERT_EQ(I->nr_srcs, 8);
      EXPECT_EQ(I->src[0].value, x.value);
      EXPECT_EQ(I->src[6].value, y.value);
      EXPECT_TRUE(agx_is_null(I->src[7]));
      ++split;
   }
   EXPECT_EQ(split, 2u);
   EXPECT_EQ(vector, 1u);
   ralloc_free(mem);
}

/* The M5 3D publication stages its coordinates with the wide scalar COPY, so
 * it reads them where they live instead of from fresh copies. */
TEST(AGX3FetchBanks, ScalarPublicationReadsCoordinatesInPlace)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index c[3] = {agx_mov_imm(b, 32, 5), agx_mov_imm(b, 32, 7),
                     agx_mov_imm(b, 32, 9)};
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
   agx_instr *collect = agx_collect_to(b, coords, 3);
   for (unsigned i = 0; i < 3; ++i)
      collect->src[i] = c[i];
   agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_texture_sample_to(b, result, coords, agx_immediate(0), agx_immediate(0),
                         agx_uniform(24, AGX_SIZE_64),
                         agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_3D,
                         AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
                         AGX_GATHER_NONE, false);
   agx_unit_test(b, result);

   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   unsigned publications = 0, moves = 0;
   agx_foreach_instr_global(b->shader, I) {
      moves += I->op == AGX_OPCODE_MOV;
      if (I->op != AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3)
         continue;
      ++publications;
      for (unsigned i = 0; i < 3; ++i)
         EXPECT_EQ(I->src[i].value, c[i].value);
   }
   EXPECT_EQ(publications, 1u);
   EXPECT_EQ(moves, 0u);
   ralloc_free(mem);
}

TEST(AGX3UniformALU, VertexTextureUniformQualification)
{
   /* Both operands name the same aligned uniform, a witnessed
    * uniform/uniform pair, so every G17P stage keeps them in the uniform
    * file; M5 keeps a 64-bit pair in fragment main programs (the witnessed
    * argument-buffer sample). */
   for (auto encoding : {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (auto stage : {MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT}) {
         for (bool preamble : {false, true}) {
            for (unsigned value : {0u, 12u, 252u, 2u, 254u, 256u}) {
               for (auto size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
                  void *mem = ralloc_context(NULL);
                  agx_shader_key key = {};
                  key.dev.encoding = encoding;
                  agx_builder *b = agx_test_builder(mem);
                  b->shader->key = &key;
                  b->shader->stage = stage;
                  b->shader->is_preamble = preamble;
                  agx_index texture = agx_mov(b, agx_uniform(value, size));
                  agx_index sampler = agx_mov(b, agx_uniform(value, size));
                  agx_instr *I = agx_texture_sample_to(
                     b, agx_temp(b->shader, AGX_SIZE_32),
                     agx_register(0, AGX_SIZE_32),
                     agx_uniform(8, AGX_SIZE_32), agx_uniform(0, AGX_SIZE_64),
                     texture, sampler,
                     agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
                     false, false, false, AGX_GATHER_NONE, false);
                  agx_optimizer_forward(b->shader);
                  agx_lower_uniform_sources(b->shader);
                  const bool keep =
                     value <= 252 && !(value & 3) &&
                     (encoding == AGX_ISA_ENCODING_AGX3B
                         ? size != AGX_SIZE_16
                         : size == AGX_SIZE_64 &&
                              stage == MESA_SHADER_FRAGMENT && !preamble);
                  EXPECT_EQ(I->src[3].type == AGX_INDEX_UNIFORM, keep);
                  EXPECT_EQ(I->src[4].type == AGX_INDEX_UNIFORM, keep);
                  EXPECT_NE(I->src[1].type, AGX_INDEX_UNIFORM);
                  ralloc_free(mem);
               }
            }
         }
      }
   }
}

TEST(AGX3UniformALU, TextureUniformOperandPairs)
{
   /* Only the descriptor-kind pairs native G17P code uses stay in the
    * uniform file: uniform/uniform, and (outside cube sampling) a uniform
    * texture with an immediate sampler. A uniform beside a register, or a
    * cube uniform beside an immediate, is materialized so the sample keeps
    * its register form. */
   enum other { OTHER_UNIFORM, OTHER_IMMEDIATE, OTHER_REGISTER };
   for (auto dim : {AGX_DIM_2D, AGX_DIM_CUBE, AGX_DIM_CUBE_ARRAY}) {
      for (auto other : {OTHER_UNIFORM, OTHER_IMMEDIATE, OTHER_REGISTER}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_index texture = agx_mov(b, agx_uniform(12, AGX_SIZE_64));
         agx_index sampler =
            other == OTHER_UNIFORM ? agx_mov(b, agx_uniform(16, AGX_SIZE_64))
            : other == OTHER_IMMEDIATE ? agx_immediate(3)
                                       : agx_register(4, AGX_SIZE_64);
         agx_instr *I = agx_texture_sample_to(
            b, agx_temp(b->shader, AGX_SIZE_32), agx_register(0, AGX_SIZE_32),
            agx_uniform(8, AGX_SIZE_32), agx_uniform(0, AGX_SIZE_64),
            texture, sampler, agx_null(), dim, AGX_LOD_MODE_LOD_MIN, 0,
            false, false, false, AGX_GATHER_NONE, false);
         agx_optimizer_forward(b->shader);
         agx_lower_uniform_sources(b->shader);
         const bool cube = dim != AGX_DIM_2D;
         const bool keep = other == OTHER_UNIFORM ||
                           (other == OTHER_IMMEDIATE && !cube);
         EXPECT_EQ(I->src[3].type == AGX_INDEX_UNIFORM, keep)
            << "dim " << dim << " other " << other;
         EXPECT_EQ(I->src[4].type == AGX_INDEX_UNIFORM,
                   other == OTHER_UNIFORM)
            << "dim " << dim << " other " << other;
         ralloc_free(mem);
      }
   }
}

/* ICMPSEL keeps an aligned 32-bit uniform only in the native ISEL shapes
 * (agx3_isel_core.c witnesses): as a select, or as one comparand beside a
 * 32-bit register; never beside an immediate, never in a constant program. */
TEST(AGX3UniformALU, IselUniformOperands)
{
   enum shape { SEL, CMP0_REG, CMP1_REG, CMP0_IMM, CMP_SEL, PREAMBLE, ODD };
   for (unsigned shape = SEL; shape <= ODD; ++shape) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->is_preamble = shape == PREAMBLE;
      agx_index r0 = agx_temp(b->shader, AGX_SIZE_32);
      agx_index r1 = agx_temp(b->shader, AGX_SIZE_32);
      agx_index u = agx_uniform(shape == ODD ? 9 : 8, AGX_SIZE_32);
      agx_index src[4] = {r0, r1, r0, r1};
      unsigned kept = ~0u;
      switch (shape) {
      case SEL: case PREAMBLE: case ODD: src[2] = u; kept = 2; break;
      case CMP0_REG: src[0] = u; kept = 0; break;
      case CMP1_REG: src[1] = u; kept = 1; break;
      case CMP0_IMM: src[0] = u; src[1] = agx_immediate(0); break;
      case CMP_SEL: src[0] = u; src[3] = u; kept = 3; break;
      }
      if (shape == PREAMBLE || shape == ODD)
         kept = ~0u;
      agx_instr *I = agx_icmpsel_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                    src[0], src[1], src[2], src[3],
                                    AGX_ICOND_ULT);
      agx_lower_uniform_sources(b->shader);
      for (unsigned s = 0; s < 4; ++s) {
         EXPECT_EQ(I->src[s].type == AGX_INDEX_UNIFORM, s == kept)
            << "shape " << shape << " src " << s;
      }
      ralloc_free(mem);
   }
}

TEST(AGX3UniformALU, AppleArithmeticWitnesses)
{
   /* Fresh applegpu_g17p output, also byte-identical on applegpu_g16g. */
   struct {
      agx3_uniform_float_op op;
      std::array<uint8_t, 8> bytes;
      unsigned length;
   } cases[] = {
      {AGX3_UNIFORM_FADD, {0x09, 0x01, 0x0c, 0x0d, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMUL, {0x09, 0x01, 0x0d, 0x0d, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMA, {0x09, 0x01, 0x0e, 0x0d, 0x21, 0x0e, 0x02, 0xc2}, 8},
   };
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[0].discard = true;
   src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
   src[1].value = 12;
   src[2].value = 14;
   for (const auto &c : cases) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B, c.op, 0,
                                            src, false, false, 32, bytes,
                                            &length));
      ASSERT_EQ(length, c.length);
      EXPECT_EQ(memcmp(bytes, c.bytes.data(), length), 0);
   }
}

TEST(AGX3UniformALU, AppleImmediateUniformFma)
{
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[0].discard = true;
   src[1].kind = AGX3_FLOAT_IMMEDIATE;
   src[1].value = 0x20;
   src[2].kind = AGX3_FLOAT_UNIFORM;
   src[2].value = 12;
   const uint8_t expected[] = {0x09, 0x01, 0x0e, 0xa1, 0x21, 0x0c, 0x02, 0xc2};
   uint8_t bytes[16], length = 0;
   ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B,
                                         AGX3_UNIFORM_FMA, 0, src, false, false,
                                         32, bytes, &length));
   ASSERT_EQ(length, sizeof(expected));
   EXPECT_EQ(memcmp(bytes, expected, length), 0);
}

TEST(AGX3UniformALU, AppleHalfArithmeticWitnesses)
{
   struct {
      agx3_uniform_float_op op;
      std::array<uint8_t, 8> bytes;
      unsigned length;
   } cases[] = {
      {AGX3_UNIFORM_FADD, {0x00, 0x00, 0x0c, 0x0c, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMUL, {0x00, 0x00, 0x0d, 0x0c, 0x00, 0xc2}, 6},
      {AGX3_UNIFORM_FMA, {0x00, 0x00, 0x0e, 0x0c, 0x21, 0x0e, 0x00, 0xc2}, 8},
   };
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[0].discard = true;
   src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
   src[1].value = 12;
   src[2].value = 14;
   for (const auto &c : cases) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_uniform_half_encode(AGX_ISA_ENCODING_AGX3B, c.op, 0, src,
                                           false, false, 32, bytes, &length));
      ASSERT_EQ(length, c.length);
      EXPECT_EQ(memcmp(bytes, c.bytes.data(), length), 0);
   }
}

TEST(AGX3UniformALU, HalfParityAndDistinctDependencySets)
{
   for (unsigned value : {0u, 1u, 12u, 13u, 255u, 256u, 511u}) {
      for (unsigned reg : {0u, 1u, 126u, 127u, 254u, 255u}) {
         for (auto op :
              {AGX3_UNIFORM_FADD, AGX3_UNIFORM_FMUL, AGX3_UNIFORM_FMA}) {
            std::set<std::string> encodings;
            for (unsigned wait = 0; wait < 64; ++wait) {
               agx3_uniform_float_src src[3] = {};
               src[0].kind = AGX3_FLOAT_REGISTER;
               src[0].value = reg;
               src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
               src[1].value = src[2].value = value;
               uint8_t bytes[16], length;
               ASSERT_TRUE(agx3_uniform_half_encode(AGX_ISA_ENCODING_AGX3B, op,
                                                    reg ^ 1, src, false, false,
                                                    wait, bytes, &length))
                  << "op=" << op << " uniform=" << value << " reg=" << reg
                  << " wait=" << wait;
               EXPECT_TRUE(
                  encodings.insert(std::string((char *)bytes, length)).second)
                  << "dependency mask aliased at " << wait;
            }
         }
      }
   }
}

TEST(AGX3UniformALU, AppleUniformClampWitnesses)
{
   struct {
      bool half, maximum, dest_cache;
      uint16_t dest, reg, uniform;
      uint8_t wait;
      std::array<uint8_t, 6> bytes;
   } cases[] = {
      {false, true, true, 0, 0, 12, 32, {0x02, 0x01, 0x2e, 0x0d, 0x00, 0xc2}},
      {false, false, false, 0, 0, 14, 0, {0x02, 0x01, 0x0e, 0x0f, 0x01, 0x02}},
      {true, true, true, 0, 0, 12, 32, {0x02, 0x00, 0x2c, 0x0c, 0x00, 0xc2}},
      {true, true, true, 1, 1, 12, 0, {0x0a, 0x00, 0x2c, 0x0c, 0x08, 0x02}},
      {true, false, false, 0, 0, 13, 0, {0x02, 0x00, 0x0c, 0x0c, 0x11, 0x02}},
   };
   for (const auto &c : cases) {
      uint8_t bytes[16], length;
      ASSERT_TRUE(agx3_uniform_minmax_encode(
         AGX_ISA_ENCODING_AGX3B, c.dest, c.reg, c.uniform, c.half, c.maximum,
         false, true, c.dest_cache, c.wait, bytes, &length));
      ASSERT_EQ(length, c.bytes.size());
      EXPECT_EQ(memcmp(bytes, c.bytes.data(), length), 0);
   }
}

TEST(AGX3UniformALU, MinmaxBoundariesAndNoWaitAliasing)
{
   for (bool half : {false, true}) {
      for (unsigned uniform : {0u, 12u, 254u, 256u, 510u}) {
         for (unsigned reg : {0u, 126u, 254u}) {
            for (bool maximum : {false, true}) {
               std::set<std::string> encodings;
               for (unsigned wait = 0; wait < 64; ++wait) {
                  uint8_t bytes[16], length;
                  bool ok = agx3_uniform_minmax_encode(
                     AGX_ISA_ENCODING_AGX3B, reg, reg, uniform, half, maximum,
                     false, false, false, wait, bytes, &length);
                  ASSERT_EQ(ok, (wait & (wait - 1)) == 0)
                     << "half=" << half << " uniform=" << uniform
                     << " reg=" << reg;
                  if (ok) {
                     EXPECT_TRUE(
                        encodings.insert(std::string((char *)bytes, length))
                           .second);
                  }
               }
            }
         }
      }
   }
}

TEST(AGX3UniformALU, UniformBoundariesAndDependencySets)
{
   for (unsigned value : {0u, 12u, 254u, 256u, 510u}) {
      for (unsigned wait = 0; wait < 64; ++wait) {
         agx3_uniform_float_src src[3] = {};
         src[0].kind = AGX3_FLOAT_REGISTER;
         src[0].value = 2;
         src[1].kind = src[2].kind = AGX3_FLOAT_UNIFORM;
         src[1].value = src[2].value = value;
         for (auto op :
              {AGX3_UNIFORM_FADD, AGX3_UNIFORM_FMUL, AGX3_UNIFORM_FMA}) {
            uint8_t bytes[16], length;
            ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B, op, 0,
                                                  src, false, true, wait, bytes,
                                                  &length))
               << "op=" << op << " uniform=" << value << " wait=" << wait;
            EXPECT_LE(length, 16);
         }
      }
   }
}

TEST(AGX3UniformALU, RefusesInvalidInputsWithoutWriting)
{
   agx3_uniform_float_src src[3] = {};
   src[0].kind = AGX3_FLOAT_REGISTER;
   src[1].kind = AGX3_FLOAT_UNIFORM;
   for (unsigned value : {1u, 255u, 511u, 512u, 65534u}) {
      src[1].value = value;
      uint8_t bytes[16];
      memset(bytes, 0xa5, sizeof(bytes));
      uint8_t length = 99;
      EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B,
                                             AGX3_UNIFORM_FADD, 0, src, false,
                                             false, 0, bytes, &length));
      EXPECT_EQ(length, 99);
      for (auto byte : bytes)
         EXPECT_EQ(byte, 0xa5);
   }
   src[1].value = 12;
   src[1].discard = true;
   uint8_t bytes[16], length;
   EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3B,
                                          AGX3_UNIFORM_FADD, 0, src, false,
                                          false, 0, bytes, &length));
   src[1].discard = false;
   /* M5 ADD takes one uniform and one register; only the uniform may be
    * negated, and the result is never saturated. */
   src[0].neg = true;
   EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C,
                                          AGX3_UNIFORM_FADD, 0, src, false,
                                          false, 0, bytes, &length));
   src[0].neg = false;
   EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C,
                                          AGX3_UNIFORM_FADD, 0, src, true,
                                          false, 0, bytes, &length));
   src[0].kind = AGX3_FLOAT_UNIFORM;
   EXPECT_FALSE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C,
                                          AGX3_UNIFORM_FADD, 0, src, false,
                                          false, 0, bytes, &length));
}

TEST(AGX3UniformALU, LoweringKeepsF32AndCopiesMixedWidths)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   agx_instr *same =
      agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_uniform(14, AGX_SIZE_32));
   agx_instr *mixed = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_16),
                                  agx_uniform(12, AGX_SIZE_32), agx_negzero());
   agx_lower_uniform_sources(b->shader);
   EXPECT_EQ(same->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(same->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_NE(mixed->src[0].type, AGX_INDEX_UNIFORM);
   ralloc_free(mem);
}

TEST(AGX3ArrayStaging, NativeVaryingLayerBanks)
{
   const struct {
      uint16_t bank, source;
      uint8_t slot;
      bool discard;
      std::array<uint8_t, 12> expected;
   } vectors[] = {
      {16, 50, 2, false, {0x17,0x05,0x54,0x16,0x00,0x64,0x00,0x00,0x70,0xc4,0x04,0x00}},
      {0, 1, 1, true, {0x17,0x05,0x54,0x06,0x00,0x02,0x00,0x00,0xf0,0xc0,0x04,0x00}},
      {8, 0, 3, true, {0x17,0x05,0x54,0x0e,0x00,0x00,0x00,0x00,0xf0,0xd0,0x04,0x00}},
      {24, 54, 5, false, {0x17,0x05,0x54,0x1e,0x00,0x6c,0x00,0x00,0x70,0xe0,0x04,0x00}},
   };
   for (const auto &v : vectors) {
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_stage_array_layer_at(
         AGX_ISA_ENCODING_AGX3B, v.bank, v.source, v.slot, 0,
         false, v.discard, bytes, &len));
      ASSERT_EQ(len, v.expected.size());
      EXPECT_EQ(std::memcmp(bytes, v.expected.data(), len), 0);
   }
}

TEST(AGX3ArrayStaging, OperandBounds)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   for (uint16_t bank : {1, 2, 3, 249, 252, 256, 65535}) {
      EXPECT_FALSE(agx3_tex_stage_array_layer_at(
         AGX_ISA_ENCODING_AGX3B, bank, 0, 1, 0, false, false, bytes, &len));
   }
   EXPECT_TRUE(agx3_tex_stage_array_layer_at(
      AGX_ISA_ENCODING_AGX3B, 248, 0, 1, 0, false, false, bytes, &len));
}

TEST(AGX3ArrayStaging, ReusedCompletionSlotRetainsCoordinateConsumer)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.g17p_uniform_dma = true;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->max_reg = 63;
   for (unsigned i = 0; i < 6; ++i) {
      agx_index value = agx_register(i * 8, AGX_SIZE_32);
      value.channels_m1 = 2;
      agx_device_load_to(b, value, agx_register(96, AGX_SIZE_64),
                        agx_immediate(i * 16), AGX_FORMAT_I32, 7, 0, false);
   }
   for (auto pair : {std::pair<unsigned, unsigned>{0, 48}, {8, 56}, {16, 64}}) {
      agx_index coords = agx_register(pair.first, AGX_SIZE_32);
      coords.channels_m1 = 2;
      agx_index result = agx_register(pair.second, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, coords, agx_immediate(0),
                           agx_immediate(0), agx_immediate(0),
                           agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
                           AGX_LOD_MODE_LOD_MIN, 15, false, false, false,
                           AGX_GATHER_NONE, false);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   agx3_tex_sample_desc d = {};
   d.dest = 64;
   d.coord = 16;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.array = true;
   d.vertex_stage = true;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.final = true;
   const uint8_t *bytes = static_cast<const uint8_t *>(binary.data);
   bool found = false;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      d.slot = slot;
      d.wait_mask = 1u << (slot - 1);
      uint8_t sample[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, sample, &len));
      for (unsigned i = 0; i + len <= binary.size; ++i)
         found |= std::memcmp(bytes + i, sample, len) == 0;
   }
   EXPECT_TRUE(found);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

static void
check_fetch_group(unsigned mode)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 40;
   for (unsigned i = 0; i < 3; ++i) {
      if (mode == 3 && i == 1)
         agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0);
      agx_index coords = agx_register(mode == 1 && i ? 4 : 0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_index dest = agx_register(mode == 2 && !i ? 0 : 32 + i * 16,
                                    AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_texture_load_to(b, dest, coords, agx_immediate(0), agx_immediate(0),
                           agx_immediate(i), agx_immediate(0), agx_null(),
                           AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 15, false, false);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   const uint8_t *bytes = static_cast<const uint8_t *>(binary.data);
   std::array<agx3_tex_fetch_desc, 3> fetches = {};
   unsigned found = 0;
   for (unsigned i = 0; i + 8 <= binary.size; ++i) {
      agx3_tex_fetch_desc d;
      if (agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3C, bytes + i, 8, &d)) {
         ASSERT_LT(found, fetches.size());
         fetches[found++] = d;
         i += 7;
      }
   }
   ASSERT_EQ(found, 3u);
   EXPECT_EQ(fetches[0].coord_kill, mode != 0);
   EXPECT_FALSE(fetches[1].coord_kill);
   EXPECT_TRUE(fetches[2].coord_kill);
   EXPECT_EQ(fetches[1].coord, fetches[2].coord);
   EXPECT_EQ(fetches[2].wait_mask, 0u);
   if (mode == 0) {
      EXPECT_EQ(fetches[0].coord, fetches[1].coord);
      EXPECT_EQ(fetches[1].wait_mask, 0u);
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(AGX3FetchGroups, SameCoordinates) { check_fetch_group(0); }
TEST(AGX3FetchGroups, ChangedCoordinates) { check_fetch_group(1); }
TEST(AGX3FetchGroups, OverwrittenCoordinates) { check_fetch_group(2); }
TEST(AGX3FetchGroups, InterveningInstruction) { check_fetch_group(3); }

TEST(AGX3FetchBanks, EightIndependentCoordinates)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 63;
   for (unsigned i = 0; i < 8; ++i) {
      agx_index coords = agx_register(i * 4, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_index dest = agx_register(64 + i * 8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_texture_load_to(b, dest, coords, agx_immediate(0), agx_immediate(0),
                           agx_immediate(i), agx_immediate(0), agx_null(),
                           AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 15, false, false);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   const uint8_t *bytes = static_cast<const uint8_t *>(binary.data);
   for (unsigned i = 0; i < 8; ++i) {
      agx3_tex_fetch_desc d = {};
      d.dest = 64 + i * 8;
      d.coord = i * 8;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.texture = i;
      d.slot = i + 1;
      d.wait_mask = 1u << i;
      d.coord_kill = true;
      d.final = i == 7;
      uint8_t expected[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, expected, &len));
      bool found = false;
      for (unsigned j = 0; j + len <= binary.size; ++j)
         found |= std::memcmp(bytes + j, expected, len) == 0;
      EXPECT_TRUE(found) << "bank " << i;
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(AGX3PixelTests, EmptyExecutionSkipsShortTestingBlocks)
{
   for (auto encoding : {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (unsigned mode : {0u, 2u}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = encoding;
         key.dev.chip = encoding == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P : AGX_CHIP_G17S;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_block *entry = list_first_entry(&b->shader->blocks, agx_block, link);
         agx_block *body = agx_test_block(b->shader);
         agx_block *end = agx_test_block(b->shader);
         agx_if_icmp(b, agx_register(0, AGX_SIZE_16), agx_immediate(0),
                     1, AGX_ICOND_UEQ, true, end);
         agx_builder middle = agx_init_builder(b->shader, agx_after_block(body));
         agx_sample_mask_agx3(&middle, agx_register(0, AGX_SIZE_16),
                              agx_register(2, AGX_SIZE_32), mode);
         agx_builder last = agx_init_builder(b->shader, agx_after_block(end));
         agx_stop(&last);
         agx_opt_jmp_none(b->shader);
         agx_instr *jump = agx_last_instr(entry);
         /* Every AGX3 encoding: M5 (AGX3C) stops its 3D stage on an
          * empty-EXEC testing update just as G17P does. */
         bool required = mode == 2;
         EXPECT_EQ(jump->op == AGX_OPCODE_JMP_EXEC_NONE, required);
         if (required) {
            EXPECT_EQ(jump->target, end);
         }
         ralloc_free(mem);
      }
   }
}

TEST(AGX3PixelTests, EmptyExecutionSkipsShortTestingElse)
{
   for (auto encoding : {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = encoding;
      key.dev.chip = encoding == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P : AGX_CHIP_G17S;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *body = list_first_entry(&b->shader->blocks, agx_block, link);
      agx_instr *else_ = agx_else_fcmp(b, agx_zero(), agx_zero(), 1,
                                     AGX_FCOND_EQ, false, body);
      agx_sample_mask_agx3(b, agx_register(0, AGX_SIZE_16),
                           agx_register(2, AGX_SIZE_32), 2);
      agx_pop_exec(b, 1);
      agx_opt_jmp_none(b->shader);
      agx_instr *jump = list_entry(else_->link.next, agx_instr, link);
      EXPECT_EQ(jump->op, AGX_OPCODE_JMP_EXEC_NONE_AFTER);
      EXPECT_EQ(jump->target, body);
      ralloc_free(mem);
   }
}

TEST(AGX3PixelTests, EmptyExecutionSkipsShortAluArmOnAgx3)
{
   const struct {
      enum agx_isa_encoding encoding;
      enum agx_chip chip;
      bool emit_agx3d;
      mesa_shader_stage stage;
      bool uniform_cond;
      bool expect_jump;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false, MESA_SHADER_FRAGMENT, false, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false, MESA_SHADER_COMPUTE, true, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G15G, false, MESA_SHADER_FRAGMENT, false, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, false, MESA_SHADER_FRAGMENT, false, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, false, MESA_SHADER_COMPUTE, false, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, MESA_SHADER_FRAGMENT, false, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, MESA_SHADER_COMPUTE, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false, MESA_SHADER_FRAGMENT, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false, MESA_SHADER_COMPUTE, false, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false, MESA_SHADER_COMPUTE, true, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, true, MESA_SHADER_FRAGMENT, false, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, true, MESA_SHADER_COMPUTE, true, false},
   };

   for (const auto &c : cases) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = c.encoding;
      key.dev.chip = c.chip;
      key.dev.emit_agx3d = c.emit_agx3d;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = c.stage;
      agx_block *entry = list_first_entry(&b->shader->blocks, agx_block, link);
      agx_block *body = agx_test_block(b->shader);
      agx_block *end = agx_test_block(b->shader);
      agx_instr *if_ = agx_if_icmp(b, agx_register(0, AGX_SIZE_16),
                                   agx_immediate(0), 1, AGX_ICOND_UEQ, true,
                                   end);
      if_->uniform_cond = c.uniform_cond;
      agx_builder middle = agx_init_builder(b->shader, agx_after_block(body));
      for (unsigned i = 0; i < 12; ++i)
         agx_iadd_to(&middle, agx_register(2, AGX_SIZE_32),
                     agx_register(2, AGX_SIZE_32), agx_immediate(1), 0);
      agx_builder last = agx_init_builder(b->shader, agx_after_block(end));
      agx_stop(&last);
      agx_opt_jmp_none(b->shader);
      agx_instr *jump = agx_last_instr(entry);
      EXPECT_EQ(jump->op == AGX_OPCODE_JMP_EXEC_NONE, c.expect_jump);
      ralloc_free(mem);
   }
}

TEST(AGX3UniformALU, M5NativeMultiplyAndFmaUniformWitnesses)
{
   const struct {
      agx3_uniform_float_op op; uint16_t dest; bool sat, cache; uint8_t wait;
      agx3_uniform_float_src src[3]; uint8_t length; uint8_t expected[16];
   } cases[] = {
      {AGX3_UNIFORM_FMA, 14, false, true, 2, {{AGX3_FLOAT_REGISTER, 20, false, false, true, false},{AGX3_FLOAT_UNIFORM, 22, false, true, false, false},{AGX3_FLOAT_REGISTER, 12, false, false, false, true}}, 8, {0x79,0x95,0x26,0x17,0x81,0x0c,0x02,0x4a}},
      {AGX3_UNIFORM_FMA, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false},{AGX3_FLOAT_UNIFORM, 20, false, true, false, false},{AGX3_FLOAT_REGISTER, 10, false, false, false, true}}, 8, {0x59,0x93,0x26,0x15,0x81,0x0a,0x02,0x0a}},
      {AGX3_UNIFORM_FMUL, 18, true, true, 0, {{AGX3_FLOAT_UNIFORM, 28, false, false, false, false},{AGX3_FLOAT_REGISTER, 18, false, false, false, true},{}}, 8, {0x99,0x1d,0x35,0x13,0x81,0x00,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true},{AGX3_FLOAT_UNIFORM, 36, false, false, false, false},{}}, 6, {0x59,0x01,0x2d,0x25,0x00,0x02}},
      {AGX3_UNIFORM_FMA, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true},{AGX3_FLOAT_UNIFORM, 38, false, false, false, false},{AGX3_FLOAT_REGISTER, 10, false, false, false, true}}, 6, {0x19,0x03,0x2e,0x27,0xa0,0x0a}},
      {AGX3_UNIFORM_FMA, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 4, false, false, false, true},{AGX3_FLOAT_UNIFORM, 34, false, false, false, false},{AGX3_FLOAT_REGISTER, 2, false, false, false, true}}, 6, {0x29,0x05,0x2e,0x23,0xa0,0x02}},
      {AGX3_UNIFORM_FMUL, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 10, false, false, false, true},{AGX3_FLOAT_UNIFORM, 22, false, false, false, false},{}}, 6, {0x59,0x0b,0x2d,0x17,0x00,0x02}},
      {AGX3_UNIFORM_FMA, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 4, false, false, false, true},{AGX3_FLOAT_UNIFORM, 22, false, false, false, false},{AGX3_FLOAT_REGISTER, 6, false, false, false, true}}, 6, {0x29,0x05,0x2e,0x17,0xa0,0x06}},
      {AGX3_UNIFORM_FMUL, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true},{AGX3_FLOAT_UNIFORM, 150, false, false, false, false},{}}, 6, {0x09,0x01,0x3d,0x17,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true},{AGX3_FLOAT_UNIFORM, 158, false, false, false, false},{}}, 6, {0x19,0x03,0x3d,0x1f,0x00,0x02}},
   };
   for (const auto &c : cases) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C, c.op,
                  c.dest, c.src, c.sat, c.cache, c.wait, bytes, &length));
      ASSERT_EQ(length, c.length);
      EXPECT_EQ(std::memcmp(bytes, c.expected, length), 0);
   }
}

/* Native G17G Minecraft 26.3 terrain, entity, cloud and
 * sky pipelines: every distinct f32 ADD/MUL/FMA with a uniform operand or a
 * uniform-file result in their mains and constant programs whose fields the
 * encoder names, re-encoded from the decoded operands; plus the R*R+U FMAs of
 * the agx3-diff-corpus native G17G r_heavy_loop fragment. */
TEST(AGX3UniformALU, M5NativeUniformOperandAndResultShapes)
{
   const struct {
      agx3_uniform_float_op op; bool uniform_dest; uint16_t dest;
      bool sat, cache; uint8_t wait; agx3_uniform_float_src src[3];
      uint8_t length; uint8_t expected[16];
   } cases[] = {
      {AGX3_UNIFORM_FMA, false, 8, false, true, 0, {{AGX3_FLOAT_REGISTER, 6, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 4, false, true, false, false, false}, {AGX3_FLOAT_IMMEDIATE, 48, false, false, false, false, false}}, 8, {0x49,0x87,0x26,0x05,0x21,0xb0,0x02,0x0a}},
      {AGX3_UNIFORM_FMUL, true, 32, false, false, 0, {{AGX3_FLOAT_IMMEDIATE, 8, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 8, false, false, true, false, false}, {}}, 6, {0x09,0x89,0x45,0x89,0xc0,0x00}},
      {AGX3_UNIFORM_FMUL, true, 36, false, false, 0, {{AGX3_FLOAT_IMMEDIATE, 24, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 8, false, false, false, true, false}, {}}, 6, {0x29,0x99,0x55,0x09,0xc0,0x00}},
      {AGX3_UNIFORM_FMUL, true, 34, false, false, 0, {{AGX3_FLOAT_IMMEDIATE, 24, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 10, false, false, true, false, false}, {}}, 6, {0x19,0x99,0x45,0x8b,0xc0,0x00}},
      {AGX3_UNIFORM_FMUL, true, 38, false, false, 0, {{AGX3_FLOAT_IMMEDIATE, 8, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 10, false, false, false, true, false}, {}}, 6, {0x39,0x89,0x55,0x0b,0xc0,0x00}},
      {AGX3_UNIFORM_FMUL, false, 12, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 30, false, false, false, false, false}, {}}, 6, {0x69,0x0d,0x2d,0x1f,0x00,0x02}},
      {AGX3_UNIFORM_FADD, false, 18, false, true, 0, {{AGX3_FLOAT_UNIFORM, 32, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {}}, 6, {0x99,0x21,0x24,0x83,0x80,0x00}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_UNIFORM, 34, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {}}, 6, {0x79,0x23,0x24,0x81,0x80,0x00}},
      {AGX3_UNIFORM_FADD, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 32, false, true, false, false, false}, {}}, 6, {0x99,0x83,0x24,0x21,0x00,0x0a}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 34, false, true, false, false, false}, {}}, 6, {0x79,0x81,0x24,0x23,0x00,0x0a}},
      {AGX3_UNIFORM_FADD, false, 18, false, true, 0, {{AGX3_FLOAT_UNIFORM, 36, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {}}, 6, {0x99,0x25,0x24,0x83,0x80,0x00}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 38, false, true, false, false, false}, {}}, 6, {0x79,0x81,0x24,0x27,0x00,0x0a}},
      {AGX3_UNIFORM_FADD, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 36, false, true, false, false, false}, {}}, 6, {0x99,0x83,0x24,0x25,0x00,0x0a}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_UNIFORM, 38, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {}}, 6, {0x79,0x27,0x24,0x81,0x80,0x00}},
      {AGX3_UNIFORM_FMUL, false, 90, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 20, false, false, false, false, false}, {}}, 6, {0xd9,0x03,0xad,0x15,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 88, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 22, false, false, false, false, false}, {}}, 6, {0xc9,0x01,0xad,0x17,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 88, false, true, 0, {{AGX3_FLOAT_REGISTER, 88, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}, {}}, 6, {0xc9,0x59,0xad,0x1b,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 90, false, true, 0, {{AGX3_FLOAT_REGISTER, 90, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}, {}}, 6, {0xd9,0x5b,0xad,0x19,0x00,0x02}},
      {AGX3_UNIFORM_FADD, false, 52, false, true, 0, {{AGX3_FLOAT_REGISTER, 62, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 28, false, true, false, false, false}, {}}, 6, {0xa9,0x3f,0x6c,0x1d,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 56, true, true, 0, {{AGX3_FLOAT_REGISTER, 52, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 30, false, false, false, false, false}, {}}, 8, {0xc9,0x35,0x6d,0x1f,0x01,0x02,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 20, false, false, false, false, false}, {}}, 6, {0x99,0x83,0x25,0x15,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 22, false, false, false, false, false}, {}}, 6, {0x89,0x81,0x25,0x17,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}, {}}, 6, {0x99,0x13,0x2d,0x19,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}, {}}, 6, {0x89,0x11,0x2d,0x1b,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 16, false, true, 0, {{AGX3_FLOAT_UNIFORM, 62, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {}}, 6, {0x89,0x3f,0x25,0x8f,0x80,0x00}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 40, false, true, false, false, false}, {}}, 6, {0x79,0x91,0x24,0x29,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 20, false, true, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 48, false, false, false, false, false}, {}}, 6, {0xa9,0x0f,0x2d,0x31,0x00,0x02}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 44, false, true, false, false, false}, {}}, 6, {0x79,0x93,0x24,0x2d,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 22, false, true, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 50, false, false, false, false, false}, {}}, 6, {0xb9,0x0f,0x2d,0x33,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 26, false, true, 0, {{AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 12, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}}, 8, {0xd9,0x39,0x26,0x8d,0x21,0x38,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 20, false, true, 0, {{AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 12, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}}, 8, {0xa9,0x3b,0x26,0x8d,0x21,0x3a,0x82,0x08}},
      {AGX3_UNIFORM_FMUL, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 62, false, false, false, false, false}, {}}, 6, {0x19,0x03,0x2d,0x3f,0x00,0x02}},
      /* Native G17G compile of fma(x, u, -y) in a fragment: R*U-R. */
      {AGX3_UNIFORM_FMA, false, 0, false, false, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 32, false, true, true, false, false}}, 8, {0x09,0x93,0x06,0x05,0x11,0xa0,0x02,0x02}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}}, 6, {0x09,0x83,0x26,0x39,0xa0,0x00}},
      {AGX3_UNIFORM_FMA, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}}, 6, {0x89,0x83,0x26,0x3b,0xa0,0x10}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 12, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}}, 8, {0x79,0x3d,0x26,0x8d,0x21,0x3c,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}}, 6, {0x19,0x03,0x2e,0x3d,0xa0,0x04}},
      {AGX3_UNIFORM_FADD, false, 8, false, true, 0, {{AGX3_FLOAT_REGISTER, 10, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 16, false, true, false, false, false}, {}}, 6, {0x49,0x8b,0x24,0x11,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 8, false, true, 0, {{AGX3_FLOAT_REGISTER, 8, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}, {}}, 6, {0x49,0x09,0x2d,0x19,0x00,0x02}},
      {AGX3_UNIFORM_FADD, false, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 8, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 20, false, true, false, false, false}, {}}, 6, {0x59,0x89,0x24,0x15,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 10, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}, {}}, 6, {0x59,0x0b,0x2d,0x1b,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 0, false, true, 0, {{AGX3_FLOAT_UNIFORM, 36, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {}}, 6, {0x09,0x25,0x35,0x01,0x80,0x00}},
      {AGX3_UNIFORM_FMUL, false, 2, false, true, 0, {{AGX3_FLOAT_UNIFORM, 38, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {}}, 6, {0x19,0x27,0x35,0x03,0x80,0x00}},
      {AGX3_UNIFORM_FMUL, false, 8, false, false, 0, {{AGX3_FLOAT_REGISTER, 8, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 34, false, false, false, false, false}, {}}, 6, {0x49,0x09,0x0d,0x23,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 8, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 28, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}}, 6, {0x09,0x89,0x26,0x1d,0xa0,0x00}},
      {AGX3_UNIFORM_FMA, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 8, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 30, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}}, 6, {0x19,0x89,0x26,0x1f,0xa0,0x0c}},
      {AGX3_UNIFORM_FMUL, false, 2, false, true, 0, {{AGX3_FLOAT_UNIFORM, 40, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}, {}}, 6, {0x19,0x29,0x35,0x05,0x80,0x00}},
      {AGX3_UNIFORM_FMUL, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 42, false, false, false, false, false}, {}}, 6, {0x19,0x03,0x2d,0x2b,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 8, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 32, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}}, 6, {0x29,0x09,0x2e,0x21,0xa0,0x04}},
      {AGX3_UNIFORM_FADD, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 40, false, true, false, false, false}, {}}, 6, {0x79,0x93,0x24,0x29,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 48, false, false, false, false, false}, {}}, 6, {0x79,0x0f,0x2d,0x31,0x00,0x02}},
      {AGX3_UNIFORM_FADD, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 44, false, true, false, false, false}, {}}, 6, {0x89,0x8d,0x24,0x2d,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 50, false, false, false, false, false}, {}}, 6, {0x89,0x11,0x2d,0x33,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 12, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 62, false, false, false, false, false}, {}}, 6, {0x69,0x0d,0x2d,0x3f,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 24, false, true, 0, {{AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}}, 8, {0xc9,0x39,0x26,0x83,0x21,0x38,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 22, false, true, 0, {{AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}}, 8, {0xb9,0x3b,0x26,0x83,0x21,0x3a,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}}, 6, {0x09,0x8d,0x26,0x39,0xa0,0x10}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}}, 6, {0x79,0x8d,0x26,0x3b,0xa0,0x0e}},
      {AGX3_UNIFORM_FMA, false, 18, false, true, 0, {{AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}}, 8, {0x99,0x3d,0x26,0x83,0x21,0x3c,0x82,0x08}},
      {AGX3_UNIFORM_FMUL, false, 10, false, true, 0, {{AGX3_FLOAT_UNIFORM, 62, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 6, false, false, true, false, false}, {}}, 6, {0x59,0x3f,0x25,0x87,0x80,0x00}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}}, 6, {0x29,0x0d,0x2e,0x3d,0xa0,0x04}},
      {AGX3_UNIFORM_FADD, false, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 20, false, true, false, false, false}, {}}, 6, {0x29,0x81,0x24,0x15,0x00,0x0a}},
      {AGX3_UNIFORM_FMUL, false, 6, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 8, false, false, false, false, false}, {}}, 6, {0x39,0x83,0x25,0x09,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 18, false, false, false, false, false}, {}}, 6, {0x19,0x01,0x2d,0x13,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}}, 8, {0x09,0x19,0x26,0x83,0x21,0x18,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}}, 8, {0x29,0x1b,0x26,0x83,0x21,0x1a,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 12, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}}, 6, {0x09,0x83,0x26,0x0d,0xa0,0x00}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 14, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}}, 6, {0x29,0x83,0x26,0x0f,0xa0,0x04}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_UNIFORM, 28, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 2, false, true, true, false, false}, {AGX3_FLOAT_UNIFORM, 28, false, false, false, false, false}}, 8, {0x29,0x1d,0x26,0x83,0x21,0x1c,0x82,0x08}},
      {AGX3_UNIFORM_FMA, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 16, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}}, 6, {0x19,0x03,0x2e,0x11,0xa0,0x04}},
      {AGX3_UNIFORM_FMA, true, 24, false, false, 8, {{AGX3_FLOAT_REGISTER, 64, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 68, false, false, false, true, false}}, 8, {0xc9,0xc1,0x06,0x81,0x81,0x44,0x42,0x80}},
      {AGX3_UNIFORM_FMA, true, 26, false, false, 0, {{AGX3_FLOAT_REGISTER, 64, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 66, false, false, false, true, false}}, 8, {0xd9,0xc1,0x06,0x83,0x81,0x42,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 28, false, false, 0, {{AGX3_FLOAT_REGISTER, 64, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 48, false, false, false, true, false}}, 8, {0xe9,0xc1,0x06,0x05,0x81,0x30,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 30, false, false, 0, {{AGX3_FLOAT_REGISTER, 64, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 6, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 62, false, false, false, true, false}}, 8, {0xf9,0x41,0x0e,0x07,0x81,0x3e,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 32, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 64, false, false, false, true, false}}, 8, {0x09,0xb9,0x46,0x81,0x81,0x40,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 34, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 58, false, false, false, true, false}}, 8, {0x19,0xb9,0x46,0x83,0x81,0x3a,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 36, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 48, false, false, false, true, false}}, 8, {0x29,0xb9,0x46,0x05,0x81,0x30,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 38, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 6, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 52, false, false, false, true, false}}, 8, {0x39,0x39,0x4e,0x07,0x81,0x34,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 40, false, false, 0, {{AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 50, false, false, false, true, false}}, 8, {0x49,0xaf,0x46,0x81,0x81,0x32,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 42, false, false, 0, {{AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 48, false, false, false, true, false}}, 8, {0x59,0xaf,0x46,0x83,0x81,0x30,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 44, false, false, 0, {{AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 42, false, false, false, true, false}}, 8, {0x69,0xaf,0x46,0x05,0x81,0x2a,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 46, false, false, 0, {{AGX3_FLOAT_REGISTER, 46, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 6, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 44, false, false, false, true, false}}, 8, {0x79,0x2f,0x4e,0x07,0x81,0x2c,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 48, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 40, false, false, false, true, false}}, 8, {0x89,0x8f,0x56,0x01,0x81,0x28,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 50, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}}, 8, {0x99,0x8f,0x56,0x03,0x81,0x10,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 52, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}}, 8, {0xa9,0x8f,0x56,0x05,0x81,0x00,0x42,0x00}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 70, false, false, false, false, true}, {AGX3_FLOAT_IMMEDIATE, 2, false, false, false, false, false}}, 8, {0x29,0x05,0x2e,0x46,0x21,0x82,0x02,0x02}},
      {AGX3_UNIFORM_FMA, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 70, false, false, false, false, true}, {AGX3_FLOAT_IMMEDIATE, 2, false, false, false, false, false}}, 8, {0x19,0x03,0x2e,0x46,0x21,0x82,0x02,0x02}},
      {AGX3_UNIFORM_FMA, false, 40, false, true, 0, {{AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 48, false, false, false, false, false}}, 8, {0x49,0x19,0x66,0x91,0x21,0x30,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 22, false, true, 0, {{AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 50, false, false, false, false, false}}, 8, {0xb9,0x1b,0x26,0x91,0x21,0x32,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 40, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 32, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 40, false, false, false, true, false}}, 6, {0x49,0x93,0x66,0x21,0xa0,0x28}},
      {AGX3_UNIFORM_FMA, false, 22, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 34, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 22, false, false, false, true, false}}, 6, {0xb9,0x93,0x26,0x23,0xa0,0x16}},
      {AGX3_UNIFORM_FMA, false, 40, false, true, 0, {{AGX3_FLOAT_UNIFORM, 28, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 52, false, false, false, false, false}}, 8, {0x49,0x1d,0x66,0x91,0x21,0x34,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 46, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 36, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 40, false, false, false, true, false}}, 6, {0x79,0x93,0x66,0x25,0xa0,0x28}},
      {AGX3_UNIFORM_FMA, false, 16, false, true, 0, {{AGX3_FLOAT_UNIFORM, 30, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 54, false, false, false, false, false}}, 8, {0x89,0x1f,0x36,0x11,0x21,0x36,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 38, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}}, 6, {0x89,0x93,0x26,0x27,0xa0,0x10}},
      {AGX3_UNIFORM_FMUL, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 36, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 64, false, false, false, false, false}, {}}, 6, {0x99,0xa5,0x25,0x41,0x00,0x02}},
      {AGX3_UNIFORM_FMUL, false, 22, false, true, 0, {{AGX3_FLOAT_REGISTER, 36, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 56, false, false, false, false, false}, {}}, 6, {0xb9,0x25,0x2d,0x39,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 38, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 66, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}}, 6, {0x99,0xa7,0x26,0x43,0xa0,0x12}},
      {AGX3_UNIFORM_FMA, false, 22, false, true, 0, {{AGX3_FLOAT_REGISTER, 38, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 58, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 22, false, false, false, true, false}}, 6, {0xb9,0x27,0x2e,0x3b,0xa0,0x16}},
      {AGX3_UNIFORM_FMA, false, 22, false, true, 0, {{AGX3_FLOAT_REGISTER, 20, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 60, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 22, false, false, false, true, false}}, 6, {0xb9,0x95,0x26,0x3d,0xa0,0x16}},
      {AGX3_UNIFORM_FMA, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 20, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 68, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}}, 6, {0x99,0x15,0x2e,0x45,0xa0,0x12}},
      {AGX3_UNIFORM_FMA, false, 20, false, true, 0, {{AGX3_FLOAT_REGISTER, 20, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 74, false, false, false, false, false}, {AGX3_FLOAT_UNIFORM, 72, false, false, false, false, false}}, 8, {0xa9,0x15,0x2e,0x4b,0x21,0x48,0x02,0x02}},
      {AGX3_UNIFORM_FMA, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 74, false, false, false, false, false}, {AGX3_FLOAT_UNIFORM, 72, false, false, false, false, false}}, 8, {0x99,0x13,0x2e,0x4b,0x21,0x48,0x02,0x02}},
      {AGX3_UNIFORM_FMA, true, 20, false, false, 8, {{AGX3_FLOAT_REGISTER, 64, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 68, false, false, false, true, false}}, 8, {0xa9,0xc1,0x06,0x01,0x81,0x44,0x42,0x80}},
      {AGX3_UNIFORM_FMA, true, 22, false, false, 0, {{AGX3_FLOAT_REGISTER, 64, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 66, false, false, false, true, false}}, 8, {0xb9,0xc1,0x06,0x83,0x81,0x42,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 24, false, false, 0, {{AGX3_FLOAT_REGISTER, 64, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 48, false, false, false, true, false}}, 8, {0xc9,0xc1,0x06,0x05,0x81,0x30,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 26, false, false, 0, {{AGX3_FLOAT_REGISTER, 64, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 6, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 62, false, false, false, true, false}}, 8, {0xd9,0x41,0x0e,0x07,0x81,0x3e,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 28, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 64, false, false, false, true, false}}, 8, {0xe9,0xb9,0x06,0x01,0x81,0x40,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 30, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 58, false, false, false, true, false}}, 8, {0xf9,0xb9,0x06,0x83,0x81,0x3a,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 32, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 48, false, false, false, true, false}}, 8, {0x09,0xb9,0x46,0x05,0x81,0x30,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 34, false, false, 0, {{AGX3_FLOAT_REGISTER, 56, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 6, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 52, false, false, false, true, false}}, 8, {0x19,0x39,0x4e,0x07,0x81,0x34,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 36, false, false, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 50, false, false, false, true, false}}, 8, {0x29,0x01,0x46,0xaf,0x81,0x32,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 38, false, false, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 48, false, false, false, true, false}}, 8, {0x39,0x83,0x46,0xaf,0x81,0x30,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 40, false, false, 0, {{AGX3_FLOAT_REGISTER, 4, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 42, false, false, false, true, false}}, 8, {0x49,0x05,0x46,0xaf,0x81,0x2a,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 42, false, false, 0, {{AGX3_FLOAT_REGISTER, 6, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 46, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 44, false, false, false, true, false}}, 8, {0x59,0x07,0x56,0x2f,0x81,0x2c,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 44, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 34, false, false, false, true, false}}, 8, {0x69,0x8f,0x56,0x01,0x81,0x22,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 46, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}}, 8, {0x79,0x8f,0x56,0x03,0x81,0x10,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 48, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}}, 8, {0x89,0x8f,0x56,0x05,0x81,0x12,0x42,0x00}},
      {AGX3_UNIFORM_FMA, false, 16, false, true, 0, {{AGX3_FLOAT_UNIFORM, 74, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 66, false, false, false, false, false}}, 8, {0x89,0x4b,0x36,0x13,0x21,0x42,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 12, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 72, false, false, false, false, false}, {AGX3_FLOAT_UNIFORM, 64, false, false, false, false, false}}, 8, {0x69,0x0d,0x2e,0x49,0x21,0x40,0x02,0x02}},
      {AGX3_UNIFORM_FMA, false, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 10, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 76, false, false, false, false, false}, {AGX3_FLOAT_UNIFORM, 68, false, false, false, false, false}}, 8, {0x59,0x0b,0x2e,0x4d,0x21,0x44,0x02,0x02}},
      {AGX3_UNIFORM_FMA, false, 20, false, true, 0, {{AGX3_FLOAT_UNIFORM, 28, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 44, false, false, false, false, false}}, 8, {0xa9,0x1d,0x26,0x91,0x21,0x2c,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_UNIFORM, 30, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 46, false, false, false, false, false}}, 8, {0x79,0x1f,0x26,0x91,0x21,0x2e,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 36, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 20, false, false, false, true, false}}, 6, {0x99,0x8d,0x26,0x25,0xa0,0x14}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 38, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}}, 6, {0x79,0x8d,0x26,0x27,0xa0,0x0e}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_UNIFORM, 32, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 48, false, false, false, false, false}}, 8, {0x79,0x21,0x26,0x91,0x21,0x30,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 16, false, true, 0, {{AGX3_FLOAT_UNIFORM, 34, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 50, false, false, false, false, false}}, 8, {0x89,0x23,0x36,0x11,0x21,0x32,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 40, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}}, 6, {0x79,0x8d,0x26,0x29,0xa0,0x0e}},
      {AGX3_UNIFORM_FMA, false, 12, false, true, 0, {{AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 42, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}}, 6, {0x69,0x0d,0x2e,0x2b,0xa0,0x10}},
      {AGX3_UNIFORM_FMA, true, 40, false, false, 0, {{AGX3_FLOAT_REGISTER, 46, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 50, false, false, false, true, false}}, 8, {0x49,0xaf,0x46,0x01,0x81,0x32,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 48, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 34, false, false, false, true, false}}, 8, {0x89,0x8f,0x56,0x01,0x81,0x22,0x42,0x00}},
      {AGX3_UNIFORM_FMA, true, 52, false, false, 0, {{AGX3_FLOAT_REGISTER, 14, false, false, true, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}}, 8, {0xa9,0x8f,0x56,0x05,0x81,0x12,0x42,0x00}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 70, false, false, false, false, true}, {AGX3_FLOAT_IMMEDIATE, 2, false, false, false, false, false}}, 8, {0x09,0x01,0x2e,0x46,0x21,0x82,0x02,0x02}},
      {AGX3_UNIFORM_FADD, false, 18, false, true, 0, {{AGX3_FLOAT_REGISTER, 16, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 64, false, false, false, false, false}, {}}, 6, {0x99,0x11,0x2c,0x41,0x00,0x02}},
      {AGX3_UNIFORM_FADD, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 66, false, false, false, false, false}, {}}, 6, {0x09,0x01,0x2c,0x43,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 20, false, true, 0, {{AGX3_FLOAT_UNIFORM, 24, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 48, false, false, false, false, false}}, 8, {0xa9,0x19,0x26,0x93,0x21,0x30,0x82,0x00}},
      {AGX3_UNIFORM_FADD, false, 16, false, true, 0, {{AGX3_FLOAT_REGISTER, 22, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 68, false, false, false, false, false}, {}}, 6, {0x89,0x17,0x2c,0x45,0x00,0x02}},
      {AGX3_UNIFORM_FMA, false, 32, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 32, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 20, false, false, false, true, false}}, 6, {0x09,0x81,0x66,0x21,0xa0,0x14}},
      {AGX3_UNIFORM_FMA, false, 20, false, true, 0, {{AGX3_FLOAT_UNIFORM, 28, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 52, false, false, false, false, false}}, 8, {0xa9,0x1d,0x26,0x93,0x21,0x34,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 20, false, false, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 36, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 20, false, false, false, true, false}}, 6, {0xa9,0x81,0x06,0x25,0xa0,0x14}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_UNIFORM, 26, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 50, false, false, false, false, false}}, 8, {0x29,0x1b,0x26,0x93,0x21,0x32,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 12, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 34, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}}, 6, {0x69,0x81,0x26,0x23,0xa0,0x04}},
      {AGX3_UNIFORM_FMA, false, 14, false, true, 0, {{AGX3_FLOAT_UNIFORM, 30, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 18, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 54, false, false, false, false, false}}, 8, {0x79,0x1f,0x36,0x13,0x21,0x36,0x82,0x00}},
      {AGX3_UNIFORM_FMA, false, 12, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, false, false}, {AGX3_FLOAT_UNIFORM, 38, false, false, false, false, false}, {AGX3_FLOAT_REGISTER, 14, false, false, false, true, false}}, 6, {0x69,0x01,0x26,0x27,0xa0,0x0e}},

      /* agx3-diff-corpus native G17G r_heavy_loop fragment: R*R+U. */
      {AGX3_UNIFORM_FMA, false, 2, false, true, 0, {{AGX3_FLOAT_REGISTER, 2, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 4, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 6, false, false, false, false, false}}, 8, {0x19,0x03,0x2e,0x85,0x21,0x06,0x02,0x00}},
      {AGX3_UNIFORM_FMA, false, 4, false, true, 0, {{AGX3_FLOAT_REGISTER, 4, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 0, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 6, false, false, false, false, false}}, 8, {0x29,0x05,0x2e,0x81,0x21,0x06,0x02,0x00}},
      {AGX3_UNIFORM_FMA, false, 0, false, true, 0, {{AGX3_FLOAT_REGISTER, 0, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 10, false, false, true, false, false}, {AGX3_FLOAT_UNIFORM, 6, false, false, false, false, false}}, 8, {0x09,0x01,0x2e,0x8b,0x21,0x06,0x02,0x00}},
      {AGX3_UNIFORM_FMA, false, 10, false, true, 0, {{AGX3_FLOAT_REGISTER, 10, false, false, false, true, false}, {AGX3_FLOAT_REGISTER, 12, false, false, false, true, false}, {AGX3_FLOAT_UNIFORM, 6, false, false, false, false, false}}, 8, {0x59,0x0b,0x3e,0x0d,0x21,0x06,0x02,0x00}},
   };
   unsigned failed = 0;
   for (const auto &c : cases) {
      uint8_t bytes[16] = {0}, length = 0;
      const bool ok = c.uniform_dest
         ? agx3_uniform_float_store_encode(AGX_ISA_ENCODING_AGX3C, c.op,
                                           c.dest, c.src, c.wait, bytes,
                                           &length)
         : agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C, c.op, c.dest,
                                     c.src, c.sat, c.cache, c.wait, bytes,
                                     &length);
      if (!ok || length != c.length || std::memcmp(bytes, c.expected, length)) {
         ++failed;
         fprintf(stderr, "mismatch op=%u ud=%u dest=%u ok=%u len=%u/%u got=",
                 (unsigned)c.op, c.uniform_dest, c.dest, ok, length, c.length);
         for (unsigned i = 0; i < length; ++i)
            fprintf(stderr, "%02x", bytes[i]);
         fprintf(stderr, " want=");
         for (unsigned i = 0; i < c.length; ++i)
            fprintf(stderr, "%02x", c.expected[i]);
         fprintf(stderr, "\n");
      }
   }
   EXPECT_EQ(failed, 0u);
}

/* The same native M5 vertex mains write their outputs' final FMUL/FMA
 * straight into the staging file (f9=5) that UVSW reads. Every distinct
 * staged R*R MUL, U*R+R FMA and R*R+R FMA whose fields the encoder names,
 * with sources in agx3_stage_vertex_alu order (src[0] is the second
 * hardware operand, the uniform of U*R+R). */
TEST(AGX3UniformALU, M5NativeStagedVertexOutputALU)
{
   const struct {
      bool fma; uint16_t dest; uint8_t slot; uint16_t src[3]; bool uniform0;
      bool live[3], discard[3]; uint8_t wait; uint8_t length;
      uint8_t expected[16];
   } cases[] = {
      {true, 8, 1, {40, 20, 40}, true, {false, true, false}, {false, false, true}, 0, 8, {0x49,0x95,0x26,0x29,0x81,0x28,0x42,0x02}},
      {true, 10, 2, {42, 20, 22}, true, {false, true, false}, {false, false, true}, 0, 10, {0x59,0x95,0x26,0x2b,0x82,0x16,0x42,0x02,0x00,0x04}},
      {true, 12, 2, {44, 20, 46}, true, {false, true, false}, {false, false, true}, 0, 10, {0x69,0x95,0x26,0x2d,0x82,0x2e,0x42,0x02,0x00,0x04}},
      {true, 14, 2, {46, 20, 16}, true, {false, true, false}, {false, false, true}, 0, 10, {0x79,0x95,0x26,0x2f,0x82,0x10,0x42,0x02,0x00,0x04}},
      {false, 16, 8, {40, 38, 0}, false, {false, false, false}, {true, true, false}, 0, 14, {0xa1,0x02,0x05,0x30,0x22,0x70,0xa0,0x02,0x1c,0x0a,0x20,0x00,0x02,0x10}},
      {false, 18, 8, {32, 36, 0}, false, {true, true, false}, {false, false, false}, 0, 14, {0xb1,0x80,0x05,0x30,0x22,0x70,0xa0,0x02,0x18,0x08,0x10,0x00,0x02,0x10}},
      {false, 20, 8, {34, 36, 0}, false, {true, true, false}, {false, false, false}, 0, 14, {0xa1,0x80,0x05,0x30,0x22,0x70,0xa0,0x0a,0x98,0x08,0x10,0x00,0x02,0x10}},
      {false, 22, 6, {32, 18, 0}, false, {false, true, false}, {true, false, false}, 0, 8, {0xb9,0x93,0x35,0x21,0x41,0x00,0x00,0x14}},
      {false, 24, 6, {34, 18, 0}, false, {false, true, false}, {true, false, false}, 0, 8, {0xc9,0x93,0x35,0x23,0x41,0x00,0x00,0x14}},
      {false, 26, 6, {28, 36, 0}, false, {true, false, false}, {false, true, false}, 0, 8, {0xd9,0x25,0x2d,0x9d,0x41,0x00,0x00,0x14}},
      {false, 28, 3, {28, 18, 0}, false, {false, false, false}, {true, true, false}, 0, 8, {0xe9,0x13,0x3d,0x1d,0x41,0x00,0x00,0x08}},
      {false, 0, 2, {14, 18, 0}, false, {false, false, false}, {true, true, false}, 0, 8, {0x09,0x13,0x3d,0x0f,0x41,0x00,0x00,0x04}},
      {true, 2, 3, {20, 10, 18}, true, {false, true, false}, {false, false, true}, 0, 10, {0x19,0x8b,0x26,0x15,0x82,0x12,0x42,0x02,0x00,0x08}},
      {true, 4, 4, {22, 10, 14}, true, {false, true, false}, {false, false, true}, 0, 10, {0x29,0x8b,0x26,0x17,0x82,0x0e,0x42,0x02,0x00,0x0c}},
      {true, 6, 5, {24, 10, 14}, true, {false, true, false}, {false, false, true}, 0, 10, {0x39,0x8b,0x26,0x19,0x82,0x0e,0x42,0x02,0x00,0x10}},
      {true, 8, 6, {26, 10, 12}, true, {false, false, false}, {false, true, true}, 0, 10, {0x49,0x0b,0x2e,0x1b,0x82,0x0c,0x42,0x02,0x00,0x14}},
      {true, 4, 2, {40, 16, 32}, true, {false, true, false}, {false, false, true}, 0, 10, {0x29,0x91,0x26,0x29,0x82,0x20,0x42,0x02,0x00,0x04}},
      {false, 0, 3, {28, 8, 0}, false, {false, false, false}, {true, true, false}, 1, 8, {0x09,0x09,0x3d,0x1d,0x41,0x20,0x00,0x08}},
      {false, 2, 3, {30, 10, 0}, false, {false, false, false}, {true, true, false}, 0, 8, {0x19,0x0b,0x3d,0x1f,0x41,0x00,0x00,0x08}},
      {false, 6, 3, {2, 12, 0}, false, {false, false, false}, {true, true, false}, 0, 8, {0x39,0x0d,0x3d,0x03,0x41,0x00,0x00,0x08}},
      {false, 8, 3, {4, 14, 0}, false, {false, false, false}, {true, true, false}, 0, 8, {0x49,0x0f,0x3d,0x05,0x41,0x00,0x00,0x08}},
      {true, 10, 7, {42, 16, 12}, true, {false, true, false}, {false, false, true}, 0, 10, {0x59,0x91,0x26,0x2b,0x82,0x0c,0x42,0x02,0x00,0x18}},
      {true, 12, 8, {44, 16, 20}, true, {false, true, false}, {false, false, true}, 0, 16, {0x21,0x80,0x07,0x8a,0x20,0x07,0xa3,0x1a,0x18,0x0b,0x88,0x00,0x14,0x08,0x00,0x80}},
      {true, 14, 1, {46, 16, 12}, true, {false, false, false}, {false, true, true}, 0, 8, {0x79,0x11,0x2e,0x2f,0x81,0x0c,0x42,0x02}},
      {true, 16, 4, {4, 6, 8}, false, {false, false, false}, {true, true, true}, 0, 10, {0x89,0x07,0x3e,0x05,0x82,0x08,0x42,0x00,0x00,0x0c}},

   };
   unsigned failed = 0;
   for (const auto &c : cases) {
      uint8_t bytes[16] = {0}, length = 0;
      const bool ok = agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3C, c.fma,
                                            c.dest, c.slot, c.src, c.uniform0,
                                            c.live, c.discard, c.wait, bytes,
                                            &length);
      if (!ok || length != c.length || std::memcmp(bytes, c.expected, length)) {
         ++failed;
         fprintf(stderr, "staged mismatch fma=%u dest=%u slot=%u ok=%u len=%u/%u got=",
                 c.fma, c.dest, c.slot, ok, length, c.length);
         for (unsigned i = 0; i < length; ++i)
            fprintf(stderr, "%02x", bytes[i]);
         fprintf(stderr, " want=");
         for (unsigned i = 0; i < c.length; ++i)
            fprintf(stderr, "%02x", c.expected[i]);
         fprintf(stderr, "\n");
      }
   }
   EXPECT_EQ(failed, 0u);
}

TEST(AGX3UniformALU, M5LoweringRetainsWitnessedFloatUniforms)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   agx_index value = agx_mov(b, agx_register(0, AGX_SIZE_32));
   agx_instr *mul = agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32),
                              value, agx_uniform(158, AGX_SIZE_32));
   agx_instr *add = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                              value, agx_uniform(158, AGX_SIZE_32));
   /* U*R+U and R*U+imm (native terrain/sky witnesses) keep both uniforms. */
   agx_instr *fma = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                agx_uniform(56, AGX_SIZE_32), value,
                                agx_uniform(56, AGX_SIZE_32));
   agx_instr *fma_imm = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                    value, agx_uniform(70, AGX_SIZE_16),
                                    agx_immediate_f(1.0));
   /* An absolute uniform is not a witnessed operand. */
   agx_instr *add_abs = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                    value,
                                    agx_abs(agx_uniform(158, AGX_SIZE_32)));
   agx_lower_uniform_sources(b->shader);
   EXPECT_EQ(mul->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(add->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(fma->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(fma->src[2].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(fma_imm->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_NE(add_abs->src[1].type, AGX_INDEX_UNIFORM);
   ralloc_free(mem);
}

TEST(AGX3UniformALU, M5SelectKeepsWitnessedUniformComparand)
{
   for (unsigned cond : {(unsigned)AGX_FCOND_GE, (unsigned)AGX_FCOND_LTN,
                         (unsigned)AGX_FCOND_EQ}) {
      for (bool zero : {true, false}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_index x = agx_mov(b, agx_register(0, AGX_SIZE_32));
         agx_index y = agx_mov(b, agx_register(2, AGX_SIZE_32));
         /* LTN normalizes to GE with the selected operands exchanged, so
          * its zero belongs in the other slot. */
         const bool exchanged = cond == AGX_FCOND_LTN;
         agx_index imm = agx_immediate(zero ? 0 : 1);
         agx_instr *I = agx_fcmpsel_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                       agx_uniform(40, AGX_SIZE_32), x,
                                       exchanged ? y : imm,
                                       exchanged ? imm : y,
                                       (enum agx_fcond)cond);
         agx_lower_uniform_sources(b->shader);
         EXPECT_EQ(I->src[0].type == AGX_INDEX_UNIFORM,
                   zero && cond != AGX_FCOND_EQ);
         ralloc_free(mem);
      }
   }
}

TEST(AGX3UniformALU, M5FloatUniformEightCompletionSlots)
{
   const agx3_uniform_float_src src[3] = {
      {AGX3_FLOAT_REGISTER, 26}, {AGX3_FLOAT_UNIFORM, 294},
      {AGX3_FLOAT_REGISTER, 40},
   };
   /* ADD has endpoints 0..3 only; a multi-producer wait takes its 12-byte
    * ep3, as Minecraft's terrain vertex shaders need. */
   for (auto op : {AGX3_UNIFORM_FADD, AGX3_UNIFORM_FMUL, AGX3_UNIFORM_FMA}) {
      std::set<std::string> unique;
      for (unsigned wait = 0; wait < 256; ++wait) {
         uint8_t bytes[16], length = 0;
         ASSERT_TRUE(agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C, op,
                      26, src, false, true, wait, bytes, &length));
         unique.emplace(reinterpret_cast<char *>(bytes), length);
      }
      EXPECT_EQ(unique.size(), 256u);
   }
}

TEST(AGX3UniformALU, M5NativeIntegerUniformSelectWitnesses)
{
   {
      agx3_isel_desc d = {};
      d.dest = 6;
      d.dest_cache = true;
      d.cmp0 = 28;
      d.cmp1 = 30;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)1;
      d.explicit_hints = true;
      d.sel_true = 28;
      d.sel_false = 30;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x32,0x1d,0x2e,0x1f,0x05,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 0;
      d.dest_cache = true;
      d.cmp0 = 26;
      d.cmp1 = 30;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)1;
      d.explicit_hints = true;
      d.sel_true = 26;
      d.sel_false = 30;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x02,0x1b,0x2e,0x1f,0x05,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 12;
      d.dest_cache = true;
      d.cmp0 = 4;
      d.cmp1 = 16;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)5;
      d.explicit_hints = true;
      d.sel_true = 4;
      d.sel_false = 16;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x62,0x05,0x2e,0x11,0x07,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 10;
      d.dest_cache = true;
      d.cmp0 = 6;
      d.cmp1 = 16;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)5;
      d.explicit_hints = true;
      d.sel_true = 6;
      d.sel_false = 16;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x52,0x07,0x2e,0x11,0x07,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 6;
      d.dest_cache = true;
      d.cmp0 = 4;
      d.cmp1 = 16;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)5;
      d.explicit_hints = true;
      d.sel_true = 4;
      d.sel_false = 16;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x32,0x05,0x2e,0x11,0x07,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 4;
      d.dest_cache = true;
      d.cmp0 = 4;
      d.cmp1 = 18;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)5;
      d.explicit_hints = true;
      d.sel_true = 4;
      d.sel_false = 18;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x22,0x05,0x2e,0x13,0x07,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 16;
      d.dest_cache = true;
      d.cmp0 = 14;
      d.cmp1 = 18;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)5;
      d.explicit_hints = true;
      d.sel_true = 14;
      d.sel_false = 18;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x82,0x0f,0x2e,0x13,0x07,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 14;
      d.dest_cache = true;
      d.cmp0 = 14;
      d.cmp1 = 18;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)5;
      d.explicit_hints = true;
      d.sel_true = 14;
      d.sel_false = 18;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_UNIFORM;
      d.wait_mask = 0;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = true;
      d.discard[3] = false;
      const uint8_t expected[] = {0x72,0x0f,0x2e,0x13,0x07,0x02};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 10;
      d.dest_cache = true;
      d.cmp0 = 2;
      d.cmp1 = 154;
      d.cmp1_immediate = false;
      d.cmp1_uniform = true;
      d.cond = (agx3_isel_icond)1;
      d.explicit_hints = true;
      d.sel_true = 156;
      d.sel_false = 0;
      d.true_kind = AGX3_ISEL_UNIFORM;
      d.false_kind = AGX3_ISEL_REGISTER;
      d.wait_mask = 1;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = false;
      d.discard[3] = true;
      const uint8_t expected[] = {0x52,0x03,0x3f,0x1b,0xa2,0x1c,0x05,0x22,0x80,0x00};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 4;
      d.dest_cache = false;
      d.cmp0 = 40;
      d.cmp1 = 0;
      d.cmp1_immediate = true;
      d.cmp1_uniform = false;
      d.cond = (agx3_isel_icond)4;
      d.explicit_hints = true;
      d.sel_true = 144;
      d.sel_false = 36;
      d.true_kind = AGX3_ISEL_UNIFORM;
      d.false_kind = AGX3_ISEL_REGISTER;
      d.wait_mask = 0;
      d.cache[0] = true;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = false;
      d.discard[1] = false;
      d.discard[2] = false;
      d.discard[3] = true;
      const uint8_t expected[] = {0x22,0xa9,0x07,0x80,0xa6,0x10,0x07,0x02,0x80,0x24};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
   {
      agx3_isel_desc d = {};
      d.dest = 6;
      d.dest_cache = false;
      d.cmp0 = 40;
      d.cmp1 = 0;
      d.cmp1_immediate = true;
      d.cmp1_uniform = false;
      d.cond = (agx3_isel_icond)4;
      d.explicit_hints = true;
      d.sel_true = 144;
      d.sel_false = 14;
      d.true_kind = AGX3_ISEL_UNIFORM;
      d.false_kind = AGX3_ISEL_REGISTER;
      d.wait_mask = 1;
      d.cache[0] = false;
      d.cache[1] = false;
      d.cache[2] = false;
      d.cache[3] = false;
      d.discard[0] = true;
      d.discard[1] = false;
      d.discard[2] = false;
      d.discard[3] = true;
      const uint8_t expected[] = {0x32,0x29,0x0f,0x80,0xa6,0x10,0x07,0x22,0x80,0x0e};
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, sizeof(expected));
      EXPECT_EQ(std::memcmp(bytes, expected, length), 0);
   }
}

TEST(AGX3UniformALU, M5IntegerUniformSelectLowering)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->is_preamble = mode == 1;
      agx_index x = agx_temp(b->shader, AGX_SIZE_32);
      agx_index bound = agx_uniform(mode == 3 ? 379 : 378, AGX_SIZE_32);
      agx_instr *I = agx_icmpsel_to(b,
         agx_temp(b->shader, mode == 2 ? AGX_SIZE_16 : AGX_SIZE_32),
         x, bound, agx_uniform(374, AGX_SIZE_32), agx_immediate(0), AGX_ICOND_ULT);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(I->src[1].type == AGX_INDEX_UNIFORM, mode == 0);
      EXPECT_EQ(I->src[2].type == AGX_INDEX_UNIFORM, mode == 0);
      EXPECT_EQ(I->icond, AGX_ICOND_ULT);
      ralloc_free(mem);
   }
}

TEST(AGX3UniformALU, M5IntegerUniformSelectEightCompletionSlots)
{
   agx3_isel_desc d = {};
   d.dest = 40; d.cmp0 = 36; d.cmp1 = 378; d.cmp1_uniform = true;
   d.cond = AGX3_ISEL_ICOND_ULT; d.explicit_hints = true;
   d.true_kind = AGX3_ISEL_UNIFORM; d.sel_true = 374;
   d.false_kind = AGX3_ISEL_IMMEDIATE; d.sel_false = 0;
   for (unsigned mode = 0; mode < 3; ++mode) {
      d.cmp1 = mode == 0 ? 378 : mode == 1 ? 38 : 7;
      d.cmp1_uniform = mode == 0;
      d.cmp1_immediate = mode == 2;
      d.false_kind = mode == 0 ? AGX3_ISEL_IMMEDIATE :
                     mode == 1 ? AGX3_ISEL_REGISTER : AGX3_ISEL_UNIFORM;
      d.sel_false = mode == 0 ? 0 : mode == 1 ? 42 : 376;
      std::set<std::string> unique;
      for (unsigned wait = 0; wait < 256; ++wait) {
         d.wait_mask = wait;
         uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length = 0;
         ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
         unique.emplace(reinterpret_cast<char *>(bytes), length);
      }
      EXPECT_EQ(unique.size(), 256u);
   }
}

TEST(AGX3UniformALU, M5IntegerUniformSelectInvalidFormsAreTransactional)
{
   agx3_isel_desc good = {};
   good.dest = 40; good.cmp0 = 36; good.cmp1 = 378;
   good.cmp1_uniform = true; good.cond = AGX3_ISEL_ICOND_ULT;
   good.explicit_hints = true;
   good.true_kind = AGX3_ISEL_UNIFORM; good.sel_true = 374;
   good.false_kind = AGX3_ISEL_IMMEDIATE; good.sel_false = 0;
   for (unsigned mode = 0; mode < 12; ++mode) {
      agx3_isel_desc d = good;
      switch (mode) {
      case 0: d.cmp0_uniform = true; break;
      case 1: d.dest_16 = true; break;
      case 2: d.cmp_16 = true; break;
      case 3: d.cmp_mixed_width = true; break;
      case 4: d.cmp1 = 379; break;
      case 5: d.cmp1 = 512; break;
      case 6: d.cmp1_immediate = true; break;
      case 7: d.cache[1] = true; break;
      case 8: d.discard[1] = true; break;
      case 9: d.sel_true = 375; break;
      case 10: d.sel_true = 512; break;
      case 11: d.discard[2] = true; break;
      }
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], expected[AGX3_ISEL_MAX_LENGTH];
      std::memset(bytes, 0xa5, sizeof(bytes));
      std::memset(expected, 0xa5, sizeof(expected));
      uint8_t length = 0xa5;
      EXPECT_FALSE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      EXPECT_EQ(std::memcmp(bytes, expected, sizeof(bytes)), 0);
      EXPECT_EQ(length, 0xa5);
   }
}

TEST(AGX3UniformALU, M5NativeUniformBranchWitnesses)
{
   uint8_t out[AGX3_CF_MAX_LENGTH], length = 0;
   const uint8_t reg_uniform[] = {0x0a,0x05,0x23,0x09,0x06,0x00,0x07,0x02,0x00,0x00};
   ASSERT_TRUE(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3C,
      AGX3_CF_ICOND_EQ, 0, 4, 8, {false,false}, out, &length));
   ASSERT_EQ(length, sizeof(reg_uniform));
   EXPECT_EQ(std::memcmp(out, reg_uniform, length), 0);
   const uint8_t uniform_zero[] = {0x0a,0x09,0x23,0x80,0x06,0x00,0x87,0x02,0x00,0x00};
   ASSERT_TRUE(agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3C,
      AGX3_CF_ICOND_EQ, 0, 8, 0, out, &length));
   ASSERT_EQ(length, sizeof(uniform_zero));
   EXPECT_EQ(std::memcmp(out, uniform_zero, length), 0);
   EXPECT_FALSE(agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3C,
      AGX3_CF_ICOND_EQ, 0, 9, 0, out, &length));
   EXPECT_FALSE(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3C,
      AGX3_CF_ICOND_EQ, 0, 4, 256, {false,false}, out, &length));
}

TEST(AGX3UniformALU, M5IntegerSelectPromotesConstantAndCanonicalizesUniformCmp)
{
   /* 1. Promote non-8-bit constant (#0xffffffff) into AGX3C 32-bit ICMPSEL. */
   for (unsigned g18 = 0; g18 < 2; ++g18) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.emit_agx3d = g18 != 0;
      key.promote_constants = true;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, struct agx_shader_info);
      b->shader->out->push_count = 16;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index lhs = agx_temp(b->shader, AGX_SIZE_32);
      agx_index rhs = agx_temp(b->shader, AGX_SIZE_32);
      agx_index alt = agx_temp(b->shader, AGX_SIZE_32);
      agx_index neg1 = agx_mov_imm(b, 32, 0xffffffffu);
      agx_instr *sel = agx_icmpsel_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                      lhs, rhs, neg1, alt, AGX_ICOND_SLT);
      agx_opt_promote_constants(b->shader);
      if (!g18) {
         ASSERT_EQ(sel->src[2].type, AGX_INDEX_UNIFORM);
         uint32_t value = 0;
         memcpy(&value,
                &b->shader->rodata[sel->src[2].value -
                                   b->shader->out->rodata.base_uniform],
                sizeof(value));
         EXPECT_EQ(value, 0xffffffffu);
      } else {
         EXPECT_EQ(sel->src[2].type, AGX_INDEX_NORMAL);
      }
      ralloc_free(mem);
   }

   /* 2. Canonicalize `u op r` to `r op' u` across all six agx_icond values,
    *    then pack through agx3_pack_binary and verify byte-for-byte against
    *    agx3_isel_encode on AGX3C. */
   const struct {
      enum agx_icond in;
      enum agx_icond canonical;
      enum agx3_isel_icond hw;
   } iconds[] = {
      {AGX_ICOND_UEQ, AGX_ICOND_UEQ, AGX3_ISEL_ICOND_EQ},
      {AGX_ICOND_SEQ, AGX_ICOND_SEQ, AGX3_ISEL_ICOND_EQ},
      {AGX_ICOND_ULT, AGX_ICOND_UGT, AGX3_ISEL_ICOND_UGT},
      {AGX_ICOND_UGT, AGX_ICOND_ULT, AGX3_ISEL_ICOND_ULT},
      {AGX_ICOND_SLT, AGX_ICOND_SGT, AGX3_ISEL_ICOND_SGT},
      {AGX_ICOND_SGT, AGX_ICOND_SLT, AGX3_ISEL_ICOND_SLT},
   };
   for (const auto &c : iconds) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 32;
      agx_index u = agx_uniform(16, AGX_SIZE_32);
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_index t = agx_temp(b->shader, AGX_SIZE_32);
      agx_index f = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *sel = agx_icmpsel_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                      u, r, t, f, c.in);
      agx_lower_uniform_sources(b->shader);
      unsigned copies = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_MOV)
            copies++;
      }
      EXPECT_EQ(copies, 0u);
      EXPECT_EQ(sel->src[0].type, AGX_INDEX_NORMAL);
      EXPECT_EQ(sel->src[1].type, AGX_INDEX_UNIFORM);
      EXPECT_EQ(sel->src[1].value, 16u);
      EXPECT_EQ(sel->icond, c.canonical);

      sel->dest[0] = agx_register(0, AGX_SIZE_32);
      sel->src[0] = agx_register(2, AGX_SIZE_32);
      sel->src[2] = agx_register(4, AGX_SIZE_32);
      sel->src[3] = agx_register(6, AGX_SIZE_32);
      agx_stop(b);

      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

      struct agx3_isel_desc d = {};
      d.dest = 0;
      d.cmp0 = 2;
      d.cmp1 = 16;
      d.cmp1_uniform = true;
      d.cond = c.hw;
      d.explicit_hints = true;
      d.true_kind = AGX3_ISEL_REGISTER;
      d.false_kind = AGX3_ISEL_REGISTER;
      d.sel_true = 4;
      d.sel_false = 6;
      uint8_t expected[AGX3_ISEL_MAX_LENGTH], expected_len = 0;
      ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, expected,
                                   &expected_len));
      ASSERT_GE(binary.size, expected_len);
      EXPECT_EQ(memcmp(binary.data, expected, expected_len), 0);

      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(AGX3UniformALU, M5HalfFcmpselPreservesSmallIntegerImmediate)
{
   /* 1. Verify AGX3C keeps inline 8-bit select immediates (including high
    *    bytes 0x80 and 0xff, both select slots, and reversed imm-first
    *    comparand) and packs byte-for-byte against agx3_alu_fcmp_select16_gpu1,
    *    while G18G materializes them. */
   const struct {
      bool reverse_cmp;
      bool imm_true;
      uint16_t sel_byte;
   } cases[] = {
      {false, false, 0x00u},
      {false, true,  0x7fu},
      {false, true,  0x80u},
      {false, false, 0xffu},
      {true,  true,  0x80u},
      {true,  false, 0xffu},
   };
   for (const auto &tc : cases) {
      for (unsigned g18 = 0; g18 < 2; ++g18) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.dev.chip = g18 ? AGX_CHIP_G18G : AGX_CHIP_G17S;
         key.dev.emit_agx3d = g18 != 0;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = MESA_SHADER_COMPUTE;
         b->shader->max_reg = 32;
         agx_index c0 = tc.reverse_cmp ? agx_immediate(0)
                                       : agx_temp(b->shader, AGX_SIZE_32);
         agx_index c1 = agx_temp(b->shader, AGX_SIZE_32);
         agx_index reg_sel = agx_temp(b->shader, AGX_SIZE_16);
         agx_index imm_sel = agx_immediate(tc.sel_byte);
         agx_instr *sel = agx_fcmpsel_to(
            b, agx_temp(b->shader, AGX_SIZE_16), c0, c1,
            tc.imm_true ? imm_sel : reg_sel,
            tc.imm_true ? reg_sel : imm_sel, AGX_FCOND_LT);
         agx_materialize_agx3_alu_constants(b->shader);
         const unsigned imm_slot = tc.imm_true ? 2u : 3u;
         if (!g18) {
            ASSERT_EQ(sel->src[imm_slot].type, AGX_INDEX_IMMEDIATE);
            EXPECT_EQ(sel->src[imm_slot].value, tc.sel_byte);

            sel->dest[0] = agx_register(0, AGX_SIZE_16);
            if (tc.reverse_cmp) {
               sel->src[1] = agx_register(2, AGX_SIZE_32);
            } else {
               sel->src[0] = agx_register(2, AGX_SIZE_32);
               sel->src[1] = agx_register(4, AGX_SIZE_32);
            }
            sel->src[tc.imm_true ? 3u : 2u] = agx_register(6, AGX_SIZE_16);
            agx_stop(b);

            util_dynarray binary;
            util_dynarray_init(&binary, mem);
            ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

            struct agx3_fsel32_gpu1_desc d = {};
            d.dest = 0;
            d.cmp0 = 2;
            d.cmp1 = tc.reverse_cmp ? 0 : 4;
            d.cmp1_immediate = tc.reverse_cmp;
            d.sel_immediate[0] = tc.imm_true;
            d.sel_immediate[1] = !tc.imm_true;
            d.sel_integer = true;
            d.sel[0] = tc.imm_true ? tc.sel_byte : 6;
            d.sel[1] = tc.imm_true ? 6 : tc.sel_byte;
            d.condition = tc.reverse_cmp ? AGX_FCOND_GT : AGX_FCOND_LT;
            uint8_t expected[AGX3_ALU_MAX_LENGTH], expected_len = 0;
            ASSERT_TRUE(agx3_alu_fcmp_select16_gpu1(&d, expected,
                                                    &expected_len));
            ASSERT_GE(binary.size, expected_len);
            EXPECT_EQ(memcmp(binary.data, expected, expected_len), 0);
            util_dynarray_fini(&binary);
         } else {
            EXPECT_EQ(sel->src[imm_slot].type, AGX_INDEX_NORMAL);
         }
         ralloc_free(mem);
      }
   }

   /* 2. Register-guard parity with packer: a uniform comparand must not be
    *    treated as encodable by agx3c_fsel16_imm_encodable(). */
   {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index u = agx_uniform(16, AGX_SIZE_32);
      agx_index c1 = agx_temp(b->shader, AGX_SIZE_32);
      agx_index reg_sel = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *sel = agx_fcmpsel_to(b, agx_temp(b->shader, AGX_SIZE_16),
                                      u, c1, agx_immediate(0x80), reg_sel,
                                      AGX_FCOND_LT);
      agx_materialize_agx3_alu_constants(b->shader);
      EXPECT_EQ(sel->src[2].type, AGX_INDEX_NORMAL);
      ralloc_free(mem);
   }
}

TEST(AGX3UniformALU, OpenCodedPack4x8LowersToPack2x16)
{
   auto build_tree = [](nir_builder *b, unsigned assoc, nir_def *b0,
                        nir_def *b1, nir_def *b2, nir_def *b3,
                        nir_def **inner_out) -> nir_def * {
      nir_def *s0 = nir_u2u32(b, b0);
      nir_def *s1 = nir_ishl_imm(b, nir_u2u32(b, b1), 8);
      nir_def *s2 = nir_ishl_imm(b, nir_u2u32(b, b2), 16);
      nir_def *s3 = nir_ishl_imm(b, nir_u2u32(b, b3), 24);
      nir_def *inner = NULL;
      nir_def *packed = NULL;
      switch (assoc) {
      case 0: /* Left-associative: (((b0 | b1<<8) | b2<<16) | b3<<24) */
         inner = nir_ior(b, s0, s1);
         packed = nir_ior(b, nir_ior(b, inner, s2), s3);
         break;
      case 1: /* Balanced: (b0 | b1<<8) | (b2<<16 | b3<<24) */
         inner = nir_ior(b, s0, s1);
         packed = nir_ior(b, inner, nir_ior(b, s2, s3));
         break;
      case 2: /* Right-associative (low-inner): b3<<24 | (b2<<16 | (b1<<8 | b0)) */
         inner = nir_ior(b, s1, s0);
         packed = nir_ior(b, s3, nir_ior(b, s2, inner));
         break;
      case 3: /* Right-associative (high-inner): b0 | (b1<<8 | (b2<<16 | b3<<24)) */
         inner = nir_ior(b, s2, s3);
         packed = nir_ior(b, s0, nir_ior(b, s1, inner));
         break;
      default:
         ADD_FAILURE() << "invalid association " << assoc;
         break;
      }
      if (inner_out)
         *inner_out = inner;
      return packed;
   };

   for (unsigned assoc = 0; assoc < 4; ++assoc) {
      /* 1 & 2. Dynamic 8-bit loads lower cleanly to pack_32_2x16_split, and
       * substituting concrete bytes 0x11..0x44 into the lowered IR and
       * constant-folding produces exact 0x44332211 (catching swapped lanes
       * or missing shifts). */
      {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "pack4x8_dyn_and_fold");
         nir_def *addr = nir_imm_int64(&b, 0x1000);
         nir_def *b0 = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
         nir_def *b1 = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 1), .align_mul = 1);
         nir_def *b2 = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 2), .align_mul = 1);
         nir_def *b3 = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 3), .align_mul = 1);
         nir_def *packed = build_tree(&b, assoc, b0, b1, b2, b3, NULL);
         nir_intrinsic_instr *store =
            nir_store_global(&b, packed, nir_iadd_imm(&b, addr, 16), .align_mul = 4);

         EXPECT_TRUE(agx_nir_lower_algebraic_late(b.shader));
         nir_opt_dce(b.shader);
         nir_validate_shader(b.shader, "pack4x8 lowered");

         unsigned pack2x16 = 0, u2u32_cnt = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_alu)
                     continue;
                  nir_alu_instr *alu = nir_instr_as_alu(instr);
                  if (alu->op == nir_op_pack_32_2x16_split)
                     pack2x16++;
                  if (alu->op == nir_op_u2u32)
                     u2u32_cnt++;
               }
            }
         }
         EXPECT_EQ(pack2x16, 1u);
         EXPECT_EQ(u2u32_cnt, 0u);

         b.cursor = nir_before_impl(b.impl);
         nir_def_rewrite_uses(b0, nir_imm_intN_t(&b, 0x11, 8));
         nir_def_rewrite_uses(b1, nir_imm_intN_t(&b, 0x22, 8));
         nir_def_rewrite_uses(b2, nir_imm_intN_t(&b, 0x33, 8));
         nir_def_rewrite_uses(b3, nir_imm_intN_t(&b, 0x44, 8));
         EXPECT_TRUE(nir_opt_constant_folding(b.shader));
         nir_opt_dce(b.shader);
         nir_validate_shader(b.shader, "pack4x8 folded");
         ASSERT_TRUE(nir_src_is_const(store->src[0]));
         EXPECT_EQ(nir_src_as_uint(store->src[0]), 0x44332211u);
         ralloc_free(b.shader);
      }

      /* 3. Negative control: multi-use inner ior stays untouched so the
       *    partial 32-bit word is not duplicated alongside a 16-bit chain. */
      {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "pack4x8_multiuse");
         nir_def *addr = nir_imm_int64(&b, 0x3000);
         nir_def *b0 = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
         nir_def *b1 = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 1), .align_mul = 1);
         nir_def *b2 = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 2), .align_mul = 1);
         nir_def *b3 = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 3), .align_mul = 1);
         nir_def *inner = NULL;
         nir_def *packed = build_tree(&b, assoc, b0, b1, b2, b3, &inner);
         nir_store_global(&b, packed, nir_iadd_imm(&b, addr, 16), .align_mul = 4);
         nir_store_global(&b, inner, nir_iadd_imm(&b, addr, 20), .align_mul = 4);

         agx_nir_lower_algebraic_late(b.shader);
         nir_opt_dce(b.shader);
         nir_validate_shader(b.shader, "pack4x8 multiuse");
         unsigned pack2x16 = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_alu &&
                      nir_instr_as_alu(instr)->op == nir_op_pack_32_2x16_split)
                     pack2x16++;
               }
            }
         }
         EXPECT_EQ(pack2x16, 0u);
         ralloc_free(b.shader);
      }
   }
}
