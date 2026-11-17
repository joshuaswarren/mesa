/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* agx_nir_lower_algebraic_late rewrites extract_u16/extract_i16 into a
 * widening of an unpacked half; a half narrowed again, u2u16 of the extract,
 * must come out as the half itself in the same pass, with the value of every
 * bit pattern kept.
 */

namespace {

static void
check(bool signed_extract, unsigned half, bool vector)
{
   const uint32_t words[] = {0x00000000u, 0x00007fffu, 0x00008000u,
                             0x0000ffffu, 0x7fff0000u, 0x80000000u,
                             0xffff0000u, 0x8000ffffu, 0x12348765u};
   for (uint32_t word : words) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "narrow");
      nir_def *in = agx_test_load_preamble(&b, 2, 32, 0);
      /* The vector case reads the word through a swizzle (component 1). */
      nir_def *w = nir_channel(&b, in, vector ? 1 : 0);
      nir_def *e = signed_extract
                      ? nir_extract_i16(&b, w, nir_imm_int(&b, half))
                      : nir_extract_u16(&b, w, nir_imm_int(&b, half));
      agx_test_store_preamble(&b, nir_u2u16(&b, e), 4);

      bool progress = false;
      NIR_PASS(progress, b.shader, agx_nir_lower_algebraic_late);
      EXPECT_TRUE(progress);
      nir_opt_dce(b.shader);

      /* No widening survives: the stored value is the unpacked half. */
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            const nir_op op = nir_instr_as_alu(instr)->op;
            EXPECT_NE(op, nir_op_u2u32);
            EXPECT_NE(op, nir_op_i2i32);
            EXPECT_NE(op, nir_op_u2u16);
         }
      }

      b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
      nir_def_rewrite_uses(in, nir_imm_ivec2(&b, vector ? 0x5a5a5a5a : word,
                                             vector ? word : 0x5a5a5a5a));
      nir_opt_constant_folding(b.shader);
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_store_preamble)
               continue;
            nir_src *src = &nir_instr_as_intrinsic(instr)->src[0];
            ASSERT_TRUE(nir_src_is_const(*src));
            EXPECT_EQ(nir_src_as_uint(*src), (word >> (16 * half)) & 0xffff)
               << std::hex << "word 0x" << word << " half " << half
               << " signed " << signed_extract << " vector " << vector;
         }
      }
      ralloc_free(b.shader);
   }
}

TEST(FoldNarrowedHalf, UnsignedExtractNarrowedIsTheHalf)
{
   for (unsigned half : {0u, 1u}) {
      check(false, half, false);
      check(false, half, true);
   }
}

TEST(FoldNarrowedHalf, SignedExtractNarrowedIsTheHalf)
{
   for (unsigned half : {0u, 1u}) {
      check(true, half, false);
      check(true, half, true);
   }
}

} // namespace
