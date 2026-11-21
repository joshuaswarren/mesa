/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* Which float min/max subgroup operations survive the AGX subgroup lowering
 * as a reduce (for the native instruction) and which become shuffle trees:
 * only 32-bit full-subgroup reductions of M5 compute shaders survive.
 */

namespace {

enum lowering { M5, M5_FTZ32, M5_FTZ16, M5_NO_MINMAX, G17P, AGX2 };

struct op_case {
   nir_intrinsic_op intrinsic;
   nir_op op;
   unsigned bit_size, components, cluster;
};

/* Builds one subgroup operation, lowers it, and returns whether a
 * reduce/scan intrinsic of the operation is left. */
static bool
kept(struct op_case c, mesa_shader_stage stage, enum lowering l)
{
   nir_builder b =
      nir_builder_init_simple_shader(stage, &agx_nir_options, "minmax");
   nir_def *x = agx_test_load_preamble(&b, c.components, c.bit_size, 0);
   nir_def *r;
   if (c.intrinsic == nir_intrinsic_reduce)
      r = nir_reduce(&b, x, .reduction_op = c.op, .cluster_size = c.cluster);
   else if (c.intrinsic == nir_intrinsic_inclusive_scan)
      r = nir_inclusive_scan(&b, x, .reduction_op = c.op);
   else
      r = nir_exclusive_scan(&b, x, .reduction_op = c.op);
   agx_test_store_preamble(&b, r, 16);

   switch (l) {
   case M5:
      agx_nir_lower_subgroups_m5(b.shader, true, true);
      break;
   case M5_FTZ32:
   case M5_FTZ16:
      b.shader->info.float_controls_execution_mode =
         l == M5_FTZ32 ? FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32
                       : FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16;
      agx_nir_lower_subgroups_m5(b.shader, true, true);
      break;
   case M5_NO_MINMAX:
      agx_nir_lower_subgroups_m5(b.shader, true, false);
      break;
   case G17P:
      agx_nir_lower_subgroups_g17p(b.shader);
      break;
   case AGX2:
      agx_nir_lower_subgroups(b.shader);
      break;
   }

   bool found = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if ((intr->intrinsic == nir_intrinsic_reduce ||
              intr->intrinsic == nir_intrinsic_inclusive_scan ||
              intr->intrinsic == nir_intrinsic_exclusive_scan) &&
             nir_intrinsic_reduction_op(intr) == c.op)
            found = true;
      }
   }
   ralloc_free(b.shader);
   return found;
}

TEST(M5FloatMinMaxReduce, FullSubgroupReductionsSurviveOnM5Compute)
{
   for (nir_op op : {nir_op_fmin, nir_op_fmax}) {
      for (unsigned cluster : {0u, 32u, 64u}) {
         /* Vectors are scalarized first; each component survives. */
         for (unsigned comps : {1u, 3u}) {
            struct op_case c = {nir_intrinsic_reduce, op, 32, comps, cluster};
            EXPECT_TRUE(kept(c, MESA_SHADER_COMPUTE, M5))
               << "op " << op << " cluster " << cluster;
            /* Flushing another width does not matter. */
            EXPECT_TRUE(kept(c, MESA_SHADER_COMPUTE, M5_FTZ16))
               << "op " << op << " cluster " << cluster;
         }
      }
   }
}

TEST(M5FloatMinMaxReduce, EverythingElseIsLowered)
{
   for (nir_op op : {nir_op_fmin, nir_op_fmax}) {
      /* Clusters smaller than the subgroup. */
      for (unsigned cluster : {1u, 2u, 4u, 8u, 16u}) {
         struct op_case c = {nir_intrinsic_reduce, op, 32, 1, cluster};
         EXPECT_FALSE(kept(c, MESA_SHADER_COMPUTE, M5))
            << "op " << op << " cluster " << cluster;
      }
      /* Other widths. */
      for (unsigned bits : {16u, 64u}) {
         struct op_case c = {nir_intrinsic_reduce, op, bits, 1, 0};
         EXPECT_FALSE(kept(c, MESA_SHADER_COMPUTE, M5)) << "bits " << bits;
      }
      /* Scans. */
      for (nir_intrinsic_op scan :
           {nir_intrinsic_inclusive_scan, nir_intrinsic_exclusive_scan}) {
         struct op_case c = {scan, op, 32, 1, 0};
         EXPECT_FALSE(kept(c, MESA_SHADER_COMPUTE, M5));
      }
      struct op_case full = {nir_intrinsic_reduce, op, 32, 1, 0};
      /* 32-bit denormals flushed, other stages, the option off (G18), other
       * targets. */
      EXPECT_FALSE(kept(full, MESA_SHADER_COMPUTE, M5_FTZ32));
      EXPECT_FALSE(kept(full, MESA_SHADER_FRAGMENT, M5));
      EXPECT_FALSE(kept(full, MESA_SHADER_VERTEX, M5));
      EXPECT_FALSE(kept(full, MESA_SHADER_COMPUTE, M5_NO_MINMAX));
      EXPECT_FALSE(kept(full, MESA_SHADER_COMPUTE, G17P));
      EXPECT_FALSE(kept(full, MESA_SHADER_COMPUTE, AGX2));
   }
}

/* With 32-bit denormals flushed, the full-subgroup reduction M5 would keep
 * is lowered instead, and the value it reduces is flushed (fadd x, -0.0)
 * before the tree, so a lone active lane's denormal comes back flushed.
 * Without that mode the reduce reads the value itself. */
static bool
reduces_flushed(nir_op op, unsigned mode, bool *kept_reduce)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "ftz");
   b.shader->info.float_controls_execution_mode = mode;
   nir_def *x = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *r = nir_reduce(&b, x, .reduction_op = op);
   agx_test_store_preamble(&b, r, 16);
   agx_nir_lower_subgroups_m5(b.shader, true, true);
   nir_opt_dce(b.shader);

   *kept_reduce = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_reduce)
            *kept_reduce = true;
      }
   }

   bool any = false, all_flushed = true;
   nir_foreach_use_including_if(src, x) {
      any = true;
      nir_instr *use = nir_src_is_if(src) ? NULL : nir_src_use_instr(src);
      if (!use || use->type != nir_instr_type_alu ||
          nir_instr_as_alu(use)->op != nir_op_fadd ||
          !nir_src_is_const(nir_instr_as_alu(use)->src[1].src) ||
          nir_src_as_uint(nir_instr_as_alu(use)->src[1].src) != 0x80000000u)
         all_flushed = false;
   }
   ralloc_free(b.shader);
   return any && all_flushed;
}

TEST(M5FloatMinMaxReduce, FlushToZeroKeepsTheFlushedTree)
{
   for (nir_op op : {nir_op_fmin, nir_op_fmax}) {
      bool kept_reduce;
      EXPECT_TRUE(reduces_flushed(op, FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                                  &kept_reduce))
         << "op " << op;
      EXPECT_FALSE(kept_reduce) << "op " << op;

      EXPECT_FALSE(reduces_flushed(op, 0, &kept_reduce)) << "op " << op;
      EXPECT_TRUE(kept_reduce) << "op " << op;

      EXPECT_FALSE(reduces_flushed(
         op, FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16, &kept_reduce))
         << "op " << op;
      EXPECT_TRUE(kept_reduce) << "op " << op;
   }
}

} // namespace
