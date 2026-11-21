/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

/* With 32-bit denormals flushed, a 32-bit float min/max reduction or scan
 * reduces a flushed value (fadd x, -0.0), since a reduction over one active
 * lane or in clusters of one returns that lane's own value without any
 * min/max. After lowering, every use of the loaded value must be the flush.
 * Nothing changes otherwise.
 */

namespace {

enum target { M5, G17P, AGX2 };

static bool
is_flush(nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fcanonicalize ||
          (alu->op == nir_op_fadd && nir_src_is_const(alu->src[1].src) &&
           nir_src_as_uint(alu->src[1].src) == 0x80000000u);
}

/* Whether every use of x is a flush (and there is at least one). */
static bool
only_flushed(nir_def *x)
{
   bool any = false;
   nir_foreach_use_including_if(src, x) {
      if (nir_src_is_if(src) || !is_flush(nir_src_use_instr(src)))
         return false;
      any = true;
   }
   return any;
}

/* Builds a reduction or scan of a loaded value, lowers it for `t`, removes
 * dead code and returns whether the value reaches it only through a flush. */
static bool
flushes(nir_intrinsic_op intrinsic, nir_op op, unsigned bit_size, unsigned mode,
        enum target t, unsigned cluster = 0)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "ftz");
   b.shader->info.float_controls_execution_mode = mode;
   nir_def *x = agx_test_load_preamble(&b, 1, bit_size, 0);
   nir_def *r =
      intrinsic == nir_intrinsic_reduce
         ? nir_reduce(&b, x, .reduction_op = op, .cluster_size = cluster)
      : intrinsic == nir_intrinsic_inclusive_scan
         ? nir_inclusive_scan(&b, x, .reduction_op = op)
         : nir_exclusive_scan(&b, x, .reduction_op = op);
   agx_test_store_preamble(&b, r, 16);
   switch (t) {
   case M5:
      agx_nir_lower_subgroups_m5(b.shader, true, true);
      break;
   case G17P:
      agx_nir_lower_subgroups_g17p(b.shader);
      break;
   case AGX2:
      agx_nir_lower_subgroups(b.shader);
      break;
   }
   nir_opt_dce(b.shader);
   const bool flushed = only_flushed(x);
   ralloc_free(b.shader);
   return flushed;
}

TEST(FtzSubgroupMinMax, FlushedWhereAsked)
{
   const unsigned ftz32 = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   for (enum target t : {M5, G17P, AGX2}) {
      for (nir_op op : {nir_op_fmin, nir_op_fmax}) {
         for (nir_intrinsic_op intr :
              {nir_intrinsic_reduce, nir_intrinsic_inclusive_scan,
               nir_intrinsic_exclusive_scan}) {
            EXPECT_TRUE(flushes(intr, op, 32, ftz32, t))
               << "target " << t << " op " << op << " intrinsic " << intr;
            EXPECT_FALSE(flushes(intr, op, 32, 0, t));
            EXPECT_FALSE(flushes(intr, op, 32,
                                 FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16, t));
         }
         /* Clusters of one return the value itself. */
         EXPECT_TRUE(flushes(nir_intrinsic_reduce, op, 32, ftz32, t, 1))
            << "target " << t << " op " << op << " cluster 1";
         /* Other widths are left to their own controls. */
         EXPECT_FALSE(flushes(nir_intrinsic_reduce, op, 16,
                              ftz32 | FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16,
                              t));
         EXPECT_FALSE(flushes(nir_intrinsic_reduce, op, 64,
                              ftz32 | FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP64,
                              t));
      }
      /* Other operations need no flush. */
      EXPECT_FALSE(flushes(nir_intrinsic_reduce, nir_op_fadd, 32, ftz32, t));
      EXPECT_FALSE(flushes(nir_intrinsic_reduce, nir_op_imax, 32, ftz32, t));
   }
}

/* G18 lowers clustered reductions before the AGX subgroup lowering: a
 * reduction in clusters of one, whose result is the loaded value itself,
 * must still store it flushed. */
static bool
g18_stores_flushed(nir_op op, unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "g18");
   b.shader->info.float_controls_execution_mode = mode;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *p = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *x = nir_load_global(&b, 1, 32, p, .align_mul = 4);
   nir_def *r = nir_reduce(&b, x, .reduction_op = op, .cluster_size = 1);
   nir_store_global(&b, r, nir_iadd_imm(&b, p, 64), .align_mul = 4);
   agx_preprocess_nir_g18g(b.shader);

   bool flushed = false, found = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_global)
            continue;
         nir_scalar v =
            nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, 0));
         flushed = is_flush(nir_def_instr(v.def));
         found = true;
      }
   }
   EXPECT_TRUE(found);
   ralloc_free(b.shader);
   return flushed;
}

TEST(FtzSubgroupMinMax, G18ClustersOfOne)
{
   for (nir_op op : {nir_op_fmin, nir_op_fmax}) {
      EXPECT_TRUE(
         g18_stores_flushed(op, FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32))
         << "op " << op;
      EXPECT_FALSE(g18_stores_flushed(op, 0)) << "op " << op;
   }
}

} // namespace
