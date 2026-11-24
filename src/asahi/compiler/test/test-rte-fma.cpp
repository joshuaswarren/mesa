/* SPDX-License-Identifier: MIT */
#include <cstdlib>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/layout/layout.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

static std::vector<uint8_t>
compile_native_fma(unsigned mode, unsigned components,
                   enum agx_chip chip = AGX_CHIP_G17P, bool weak = false)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "native fused FP32 operation");
   b.shader->info.float_controls_execution_mode = mode;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *x = agx_test_load_preamble(&b, components, 32, 4);
   nir_def *y = agx_test_load_preamble(&b, components, 32, 12);
   nir_def *z = agx_test_load_preamble(&b, components, 32, 20);
   nir_def *result = weak ? nir_ffma_weak(&b, x, y, z) : nir_ffma(&b, x, y, z);
   agx_test_store_local_pixel_agx(
      &b, nir_pad_vector(&b, result, 4), nir_imm_intN_t(&b, 0xff, 16),
      nir_undef(&b, 2, 16), 0, BITFIELD_MASK(components),
      (enum pipe_format)AIL_ISA_FORMAT_I32);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = (chip == AGX_CHIP_G13G || chip == AGX_CHIP_G14G)
                         ? AGX_ISA_ENCODING_AGX2
                         : AGX_ISA_ENCODING_AGX3B;
   if (chip == AGX_CHIP_G17C)
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = chip;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   key.fs.nr_samples = 1;
   /* As the drivers do, where the shader enters them. */
   agx_nir_lower_fma_rte(b.shader, chip);
   if (chip == AGX_CHIP_G17C)
      agx_preprocess_nir_m5(b.shader);
   else
      agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   EXPECT_TRUE(compiled);
   std::vector<uint8_t> binary;
   if (compiled) {
      const uint8_t *main =
         (const uint8_t *)part.binary + part.info.main_offset;
      binary.assign(main, main + part.info.main_size);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return binary;
}

TEST(NativeFMA, ExplicitRTERepairsTininess)
{
   for (enum agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G17C}) {
      SCOPED_TRACE(chip);
      for (unsigned components = 1; components <= 4; ++components) {
         SCOPED_TRACE(components);
         auto plain = compile_native_fma(0, components, chip);
         auto rte = compile_native_fma(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32,
                                       components, chip);
         EXPECT_FALSE(plain.empty());
         EXPECT_NE(plain, rte);
      }
   }
}

TEST(NativeFMA, OnlyRTERequiresTheTininessRepair)
{
   for (enum agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G17C}) {
      SCOPED_TRACE(chip);
      for (unsigned components = 1; components <= 4; ++components) {
         SCOPED_TRACE(components);
         auto plain = compile_native_fma(0, components, chip);
         auto ftz = compile_native_fma(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                                       components, chip);
         auto rte_ftz =
            compile_native_fma(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32 |
                                  FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                               components, chip);
         EXPECT_FALSE(plain.empty());
         EXPECT_EQ(plain, ftz);
         EXPECT_NE(plain, rte_ftz);
      }
   }
}

/* GLSL.std.450 Fma compiles as it would without an explicit rounding mode. */
TEST(NativeFMA, WeakFMAIsNotRepaired)
{
   for (enum agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G17C}) {
      SCOPED_TRACE(chip);
      for (unsigned components = 1; components <= 4; ++components) {
         SCOPED_TRACE(components);
         auto plain = compile_native_fma(0, components, chip, true);
         auto rte_ftz =
            compile_native_fma(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32 |
                                  FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                               components, chip, true);
         EXPECT_FALSE(plain.empty());
         EXPECT_EQ(plain, rte_ftz);
      }
   }
}

TEST(NativeFMA, RepairTakesOnlyStrictFMA)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G14G, AGX_CHIP_G15G,
                              AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17C}) {
      SCOPED_TRACE(chip);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "strict and weak FMA");
      b.shader->info.float_controls_execution_mode =
         FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32 |
         FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
      /* Not scalarized yet: a vector FMA is repaired per component. */
      nir_def *x = agx_test_load_preamble(&b, 2, 32, 4);
      nir_def *y = agx_test_load_preamble(&b, 2, 32, 8);
      nir_def *z = agx_test_load_preamble(&b, 2, 32, 12);
      nir_store_global(&b, nir_ffma(&b, x, y, z), nir_imm_int64(&b, 0),
                       .align_mul = 4);
      nir_store_global(&b, nir_ffma_weak(&b, x, y, z), nir_imm_int64(&b, 8),
                       .align_mul = 4);

      const bool repaired = true;
      EXPECT_EQ(agx_nir_lower_fma_rte(b.shader, chip), repaired);
      nir_validate_shader(b.shader, "strict and weak FMA");

      unsigned ifs = 0, strict = 0, weak = 0;
      nir_foreach_block(block, b.impl) {
         nir_if *nif = nir_block_get_following_if(block);
         ifs += nif != NULL;
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            strict += nir_instr_as_alu(instr)->op == nir_op_ffma;
            weak += nir_instr_as_alu(instr)->op == nir_op_ffma_weak;
         }
      }
      EXPECT_EQ(ifs, repaired ? 2u : 0u);
      EXPECT_EQ(strict, 1u);
      EXPECT_EQ(weak, 1u);
      ralloc_free(b.shader);
   }
}

struct fma_case {
   uint32_t a, c, z, native, expected;
};

static void
check_fma_case(fma_case t)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "exact FMA boundary");
   nir_def *fixed =
      agx_nir_fma_rte_fixup(&b, nir_imm_int(&b, t.a), nir_imm_int(&b, t.c),
                            nir_imm_int(&b, t.z), nir_imm_int(&b, t.native));
   nir_store_global(&b, fixed, nir_imm_int64(&b, 0), .align_mul = 4);
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_remove_phis(b.shader);
   } while (progress);
   nir_intrinsic_instr *store =
      nir_instr_as_intrinsic(nir_block_last_instr(nir_start_block(b.impl)));
   ASSERT_TRUE(nir_src_is_const(store->src[0]));
   EXPECT_EQ(nir_src_as_uint(store->src[0]), t.expected)
      << std::hex << t.a << " " << t.c << " " << t.z;
   ralloc_free(b.shader);
}

TEST(NativeFMA, NormalAddendAndCancellation)
{
   const fma_case cases[] = {
      /* The hardware failure: a tiny product subtracted from FLT_MIN. */
      {0x80800000, 0x00800000, 0x00800000, 0, 0x00800000},
      {0x00800000, 0x00800000, 0x80800000, 0x80000000, 0x80800000},
      /* Half a subnormal ULP: the even tie rounds to FLT_MIN. */
      {0x9a000000, 0x1a000000, 0x00800000, 0, 0x00800000},
      /* Just beyond the tie remains flushed. */
      {0x9a000001, 0x1a000000, 0x00800000, 0, 0},
      /* Product boundary with a flushed subnormal addend. */
      {0x3f7fffff, 0x80800000, 0x800000fe, 0x80000000, 0x80800000},
      /* Nonzero native results are never replaced. */
      {0x80800000, 0x00800000, 0x00800000, 0x12345678, 0x12345678},
   };
   for (auto t : cases)
      check_fma_case(t);
}

TEST(NativeFMA, NonzeroResultSkipsWideRepair)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "FMA normal-result fast path");
   nir_def *a = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *c = agx_test_load_preamble(&b, 1, 32, 6);
   nir_def *z = agx_test_load_preamble(&b, 1, 32, 8);
   nir_def *native = agx_test_load_preamble(&b, 1, 32, 10);
   nir_def *fixed = agx_nir_fma_rte_fixup(&b, a, c, z, native);
   nir_store_global(&b, fixed, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_validate_shader(b.shader, "guarded FMA tininess repair");

   nir_block *entry = nir_start_block(b.impl);
   nir_cf_node *next = nir_cf_node_next(&entry->cf_node);
   ASSERT_NE(next, nullptr);
   ASSERT_EQ(next->type, nir_cf_node_if);
   nir_if *guard = nir_cf_node_as_if(next);
   nir_alu_instr *eq = nir_def_as_alu(guard->condition.ssa);
   ASSERT_EQ(eq->op, nir_op_ieq);
   nir_alu_instr *abs_bits = nir_src_as_alu(eq->src[0].src);
   ASSERT_EQ(abs_bits->op, nir_op_iand);
   EXPECT_EQ(abs_bits->src[0].src.ssa, native);
   EXPECT_EQ(nir_alu_src_as_uint(abs_bits->src[1]), 0x7fffffffu);

   unsigned wide_products = 0;
   nir_foreach_instr(instr, nir_if_first_then_block(guard)) {
      if (instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == nir_op_imul &&
          nir_instr_as_alu(instr)->def.bit_size == 64)
         wide_products++;
   }
   EXPECT_EQ(wide_products, 1u);
   EXPECT_TRUE(exec_list_is_empty(&nir_if_first_else_block(guard)->instr_list));
   ralloc_free(b.shader);
}

TEST(NativeFMA, ExactDyadicOracle)
{
   /* Optional extended oracle generated with exact rationals, independent of
    * the shader's fixed-point algorithm. The normal test above has no files.
    * python3 fma-rte-oracle.py /tmp/cases.txt
    * AGX_FMA_RTE_CASES=/tmp/cases.txt agx_tests --gtest_filter=NativeFMA.*
    */
   const char *path = std::getenv("AGX_FMA_RTE_CASES");
   if (!path)
      GTEST_SKIP();
   std::ifstream input(path);
   ASSERT_TRUE(input.good());
   fma_case t;
   unsigned count = 0;
   while (input >> std::hex >> t.a >> t.c >> t.z >> t.native >> t.expected) {
      check_fma_case(t);
      count++;
   }
   EXPECT_TRUE(input.eof());
   EXPECT_GT(count, 20000u);
}

TEST(NativeFMA, PreserveModeIsUnchanged)
{
   auto preserve = compile_native_fma(FLOAT_CONTROLS_DENORM_PRESERVE_FP32, 1);
   auto rte_preserve =
      compile_native_fma(FLOAT_CONTROLS_DENORM_PRESERVE_FP32 |
                            FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32,
                         1);
   EXPECT_FALSE(preserve.empty());
   EXPECT_EQ(preserve, rte_preserve);
}

TEST(NativeFMA, AllGenerationsRepairExplicitRTE)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G14G, AGX_CHIP_G15S, AGX_CHIP_G16G}) {
      auto plain = compile_native_fma(0, 1, chip);
      auto rte =
         compile_native_fma(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32, 1, chip);
      EXPECT_FALSE(plain.empty());
      EXPECT_NE(plain, rte);
   }
}
