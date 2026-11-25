/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"

TEST(UniformRemat, ImmutableWordCopiesOnly)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   b->shader->key = &key;
   struct target {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
      bool remat;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, true},
      /* G18G: AGX3C re-encoded as agx3d */
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, true, false},
   };
   for (const target &t : targets) {
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      for (bool preamble : {false, true}) {
         b->shader->is_preamble = preamble;
         for (auto size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
            for (unsigned value : {0u, 1u, 126u, 254u, 510u, 512u}) {
               agx_index uniform = agx_uniform(0, size);
               uniform.value = value;
               agx_instr *I = agx_mov_to(b, agx_temp(b->shader, size), uniform);
               EXPECT_EQ(agx_can_rematerialize_uniform(b->shader, I),
                         t.remat && !preamble && size == AGX_SIZE_32 &&
                            value < 512 && !(value & 1))
                  << "chip " << t.chip << " agx3d " << t.emit_agx3d;
            }
         }
      }
   }
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.emit_agx3d = false;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->is_preamble = false;
   for (bool absolute : {false, true}) {
      agx_index src = agx_uniform(8, AGX_SIZE_32);
      src.abs = absolute;
      src.neg = !absolute;
      agx_instr *I = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32), src);
      EXPECT_FALSE(agx_can_rematerialize_uniform(b->shader, I));
   }
   ralloc_free(mem);
}

static void
spill_sixteen_uniform_copies(agx_isa_encoding encoding, agx_chip chip)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = encoding;
   key.dev.chip = chip;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index values[16];
   for (unsigned i = 0; i < 16; ++i)
      values[i] = agx_mov(b, agx_uniform(2 * i, AGX_SIZE_32));
   for (agx_index value : values)
      agx_unit_test(b, value);
   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 24, false);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_NE(I->op, AGX_OPCODE_STACK_LOAD);
      EXPECT_NE(I->op, AGX_OPCODE_STACK_STORE);
      agx_foreach_src(I, s)
         EXPECT_FALSE(I->src[s].memory);
      agx_foreach_dest(I, d)
         EXPECT_FALSE(I->dest[d].memory);
      copies += I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM;
   }
   EXPECT_GE(copies, 16u) << "chip " << chip;
   ralloc_free(mem);
}

TEST(UniformRemat, SpillerRecreatesUniformsWithoutMemoryTraffic)
{
   spill_sixteen_uniform_copies(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P);
}

TEST(UniformRemat, SpillerRecreatesUniformsWithoutMemoryTrafficOnAGX3C)
{
   for (auto chip : {AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C})
      spill_sixteen_uniform_copies(AGX_ISA_ENCODING_AGX3C, chip);
}

/* G18G keeps the spilled copies in memory (see agx_can_rematerialize_uniform) */
TEST(UniformRemat, G18GSpillsUniformCopies)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17G;
   key.dev.emit_agx3d = true;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index values[16];
   for (unsigned i = 0; i < 16; ++i)
      values[i] = agx_mov(b, agx_uniform(2 * i, AGX_SIZE_32));
   for (agx_index value : values)
      agx_unit_test(b, value);
   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 24, false);
   unsigned memory = 0;
   agx_foreach_instr_global(b->shader, I) {
      agx_foreach_src(I, s)
         memory += I->src[s].memory;
      agx_foreach_dest(I, d)
         memory += I->dest[d].memory;
   }
   EXPECT_GT(memory, 0u);
   ralloc_free(mem);
}

TEST(UniformRemat, UniformWritesKeepCapturedValuesInMemory)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index values[16];
   for (unsigned i = 0; i < 16; ++i)
      values[i] = agx_mov(b, agx_uniform(2 * i, AGX_SIZE_32));
   agx_uniform_store(b, agx_register(0, AGX_SIZE_32), agx_immediate(8), 1);
   for (agx_index value : values)
      agx_unit_test(b, value);
   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 24, false);
   EXPECT_TRUE(b->shader->uniforms_written);
   unsigned memory = 0;
   agx_foreach_instr_global(b->shader, I) {
      agx_foreach_src(I, s)
         memory += I->src[s].memory;
      agx_foreach_dest(I, d)
         memory += I->dest[d].memory;
   }
   EXPECT_GT(memory, 0u);
   ralloc_free(mem);
}
