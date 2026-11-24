/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx3_uniform_integer_instr.h"
#include "agx_builder.h"
#include "agx_test.h"

TEST(AGX3UniformInteger, LoweringRetainsNativeOperands)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (auto size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (unsigned op = 0; op < 10; ++op) {
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_index r = agx_temp(b->shader, size);
         agx_index u = agx_uniform(504, size);
         agx_index dest = agx_temp(b->shader, size);
         agx_instr *I;
         if (op < 3) {
            agx_index rhs = r;
            rhs.neg = op == 1;
            I = agx_iadd_to(b, dest, u, rhs, op == 2 ? 3 : 0);
         } else if (op == 3) {
            I = agx_imad_to(b, dest, u, r, agx_uniform(508, size), 0);
         } else {
            const unsigned tables[] = {0x8, 0xe, 0x6, 0x7, 0x1, 0x2};
            I = agx_bitop_to(b, dest, u, r, tables[op - 4]);
         }
         agx_lower_uniform_sources(b->shader);
         ASSERT_EQ(I->src[0].type, AGX_INDEX_UNIFORM);
         EXPECT_EQ(I->src[0].value, 504u);
         if (op == 3) {
            EXPECT_EQ(I->src[2].type, AGX_INDEX_UNIFORM);
         }

         /* The preserved operation must remain encodable after allocation,
          * including register cache hints and a full dependency mask. */
         I->dest[0] = agx_register(2, size);
         I->src[1].type = AGX_INDEX_REGISTER;
         I->src[1].value = 0;
         I->src[1].cache = true;
         agx3_uniform_integer_desc d;
         ASSERT_TRUE(agx3_uniform_integer_instruction(I, true, &d));
         d.wait_mask = 63;
         uint8_t bytes[16], length;
         EXPECT_TRUE(
            agx3_uniform_integer_encode(key.dev.encoding, &d, bytes, &length));
      }
   }
   ralloc_free(mem);
}

TEST(AGX3UniformInteger, LoweringCopiesUnsupportedDomains)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (unsigned variant = 0; variant < 7; ++variant) {
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_index u = agx_uniform(12, AGX_SIZE_32);
      if (variant == 0)
         u.size = AGX_SIZE_16;
      if (variant == 1)
         u.value = 511;
      if (variant == 2)
         r = agx_uniform(14, AGX_SIZE_32);
      if (variant == 3)
         u.neg = true;
      if (variant == 4)
         u.abs = true;
      agx_instr *I = agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32), u, r,
                                 variant == 5 ? 2 : 0);
      if (variant == 6)
         I->saturate = true;
      agx3_uniform_integer_desc d;
      EXPECT_FALSE(agx3_uniform_integer_instruction(I, false, &d));
      agx_lower_uniform_sources(b->shader);
      EXPECT_NE(I->src[0].type, AGX_INDEX_UNIFORM);
   }
   ralloc_free(mem);
}


TEST(AGX3UniformInteger, M5MadNativeBytesAndDomains)
{
   struct Witness {
      const char *hex;
      unsigned dest, a, u, c, wait;
      bool c_uniform, a_cache, a_discard, c_discard;
   } cases[] = {
      {"3782041a2100a112988442000500", 10, 2, 18, 20, 0,
       true, true, false, false},
      {"2f00041a2100a102a88442000f00", 0, 0, 18, 14, 128,
       true, false, true, false},
      {"2708047a2500a1020880e0000400", 0, 0, 12, 4, 2,
       false, false, true, true},
   };
   for (const auto &w : cases) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_index u = agx_uniform(w.u, AGX_SIZE_32);
      agx_index c = w.c_uniform ? agx_uniform(w.c, AGX_SIZE_32)
                               : agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *I = agx_imad_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                 w.c_uniform ? r : u,
                                 w.c_uniform ? u : r, c, 0);
      agx3_uniform_integer_desc d;
      ASSERT_TRUE(agx3_uniform_integer_for_encoding(key.dev.encoding, I,
                                                    false, &d));
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(I->src[w.c_uniform ? 1 : 0].type, AGX_INDEX_UNIFORM);
      I->dest[0] = agx_register(w.dest, AGX_SIZE_32);
      unsigned slot = w.c_uniform ? 0 : 1;
      I->src[slot] = agx_register(w.a, AGX_SIZE_32);
      I->src[slot].cache = w.a_cache;
      I->src[slot].discard = w.a_discard;
      if (!w.c_uniform) {
         I->src[2] = agx_register(w.c, AGX_SIZE_32);
         I->src[2].discard = w.c_discard;
      }
      ASSERT_TRUE(agx3_uniform_integer_for_encoding(key.dev.encoding, I,
                                                    true, &d));
      d.wait_mask = w.wait;
      uint8_t bytes[16], length;
      ASSERT_TRUE(agx3_uniform_integer_encode(key.dev.encoding, &d, bytes,
                                              &length));
      ASSERT_EQ(length, 14);
      for (unsigned j = 0; j < length; j++) {
         unsigned expected;
         ASSERT_EQ(sscanf(w.hex + 2*j, "%2x", &expected), 1);
         EXPECT_EQ(bytes[j], expected) << "byte " << j << " witness " << w.hex;
      }
      auto invalid = d;
      invalid.half = true;
      EXPECT_FALSE(agx3_uniform_integer_encode(key.dev.encoding, &invalid,
                                               bytes, &length));
      invalid = d;
      invalid.src[2] = {AGX3_UI_ZERO, 0};
      EXPECT_FALSE(agx3_uniform_integer_encode(key.dev.encoding, &invalid,
                                               bytes, &length));
      invalid = d;
      invalid.shift = 3;
      EXPECT_FALSE(agx3_uniform_integer_encode(key.dev.encoding, &invalid,
                                               bytes, &length));
      ralloc_free(mem);
   }
}

TEST(AGX3UniformInteger, M5MadPromotesBothConstantsTogether)
{
   for (bool constant_first : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, struct agx_shader_info);
      b->shader->out->push_count = 16;
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_index m = agx_mov_imm(b, 32, 1664525);
      agx_index c = agx_mov_imm(b, 32, 1013904223);
      agx_instr *I = agx_imad_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                 constant_first ? m : r,
                                 constant_first ? r : m, c, 0);
      agx_opt_promote_constants(b->shader);
      unsigned ms = constant_first ? 0 : 1;
      ASSERT_EQ(I->src[ms].type, AGX_INDEX_UNIFORM);
      ASSERT_EQ(I->src[2].type, AGX_INDEX_UNIFORM);
      uint32_t value;
      memcpy(&value, &b->shader->rodata[I->src[ms].value -
                  b->shader->out->rodata.base_uniform], sizeof(value));
      EXPECT_EQ(value, 1664525u);
      memcpy(&value, &b->shader->rodata[I->src[2].value -
                  b->shader->out->rodata.base_uniform], sizeof(value));
      EXPECT_EQ(value, 1013904223u);
      agx3_uniform_integer_desc d;
      EXPECT_TRUE(agx3_uniform_integer_for_encoding(key.dev.encoding, I,
                                                   false, &d));
      ralloc_free(mem);
   }
}


TEST(AGX3UniformInteger, M5MadPromotionRefusesUnwitnessedDomains)
{
   for (unsigned variant = 0; variant < 6; variant++) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, struct agx_shader_info);
      b->shader->out->push_count = variant == 5 ? 510 : 16;
      auto size = variant == 0 ? AGX_SIZE_16 : AGX_SIZE_32;
      agx_index r = agx_temp(b->shader, size);
      agx_index m = agx_mov_imm(b, size == AGX_SIZE_16 ? 16 : 32, 1664525);
      agx_index c = agx_mov_imm(b, size == AGX_SIZE_16 ? 16 : 32, 1013904223);
      if (variant == 1)
         m.neg = true;
      if (variant == 2)
         c.abs = true;
      agx_instr *I = agx_imad_to(b, agx_temp(b->shader, size), r, m, c,
                                 variant == 3 ? 3 : 0);
      I->saturate = variant == 4;
      agx_opt_promote_constants(b->shader);
      EXPECT_NE(I->src[2].type, AGX_INDEX_UNIFORM) << variant;
      if (variant == 5) {
         EXPECT_EQ(I->src[1].type, AGX_INDEX_UNIFORM);
         agx3_uniform_integer_desc d;
         EXPECT_TRUE(agx3_uniform_integer_for_encoding(key.dev.encoding, I,
                                                      false, &d));
      } else {
         EXPECT_NE(I->src[1].type, AGX_INDEX_UNIFORM) << variant;
      }
      ralloc_free(mem);
   }
}

TEST(AGX3IntegerLowering, Agx3bLowersIabsToCompareSelect)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned bits : {16u, 32u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "agx3b lower iabs");
      nir_def *in = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *out = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *id = nir_load_subgroup_invocation(&b);
      nir_def *addr = nir_iadd(&b, in, nir_u2u64(&b, nir_ishl_imm(&b, id, 3)));
      nir_def *pair = nir_load_global(&b, 2, bits, addr, .align_mul = bits / 8);
      nir_def *diff =
         nir_isub(&b, nir_channel(&b, pair, 0), nir_channel(&b, pair, 1));
      nir_def *sad = nir_iabs(&b, diff);
      nir_def *single = nir_iabs(&b, nir_channel(&b, pair, 0));
      nir_store_global(&b, nir_iadd(&b, sad, single), out,
                       .align_mul = bits / 8);

      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      agx_preprocess_nir(b.shader);
      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << bits;
      EXPECT_GT(part.info.stats.instrs, 0u) << bits;
      unsigned iabs_count = 0, ishr_count = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_op op = nir_instr_as_alu(instr)->op;
            if (op == nir_op_iabs)
               iabs_count++;
            else if (op == nir_op_ishr)
               ishr_count++;
         }
      }
      EXPECT_EQ(iabs_count, 0u) << bits;
      EXPECT_EQ(ishr_count, 0u) << bits;
      free(part.binary);
      ralloc_free(b.shader);
   }
   glsl_type_singleton_decref();
}
