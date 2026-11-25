/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx_nir_uniform_unit_global_id: unit-size compute workgroup axes replace
 * load_global_invocation_id[c] with load_workgroup_id[c] while preserving
 * non-unit axes and skipping variable-size workgroups.
 */
#include "agx_compile.h"
#include "agx_nir.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned failures;

#define CHECK(cond)                                                          \
   do {                                                                      \
      if (!(cond)) {                                                         \
         fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,   \
                 #cond);                                                     \
         failures++;                                                         \
      }                                                                      \
   } while (0)

static unsigned
count_intrinsic(nir_shader *s, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      }
   }
   return n;
}

static void
test_partial_unit_axes_become_uniform(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                  "partial_unit");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_store_global(&b, nir_iadd(&b, x, y), nir_imm_int64(&b, 0x1000),
                    .write_mask = 1, .align_mul = 4);
   nir_validate_shader(b.shader, "before");

   CHECK(agx_nir_uniform_unit_global_id(b.shader));
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_divergence_analysis(b.shader);
   nir_validate_shader(b.shader, "after");

   CHECK(count_intrinsic(b.shader, nir_intrinsic_load_workgroup_id) == 1);
   CHECK(count_intrinsic(b.shader, nir_intrinsic_load_global_invocation_id) == 1);

   /* Find the iadd: one operand (x) is divergent, the other (y) is uniform. */
   bool saw_add = false;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_iadd)
            continue;
         saw_add = true;
         CHECK(nir_src_is_divergent(&alu->src[0].src));
         CHECK(!nir_src_is_divergent(&alu->src[1].src));
      }
   }
   CHECK(saw_add);
   ralloc_free(b.shader);
}

static void
test_single_thread_and_internal_skipped(void)
{
   /* 1x1x1 workgroup: no intra-workgroup uniformity to expose, must skip. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "unit_1x1x1");
      b.shader->info.internal = false;
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_store_global(&b, gid, nir_imm_int64(&b, 0x1000), .write_mask = 7,
                       .align_mul = 4);
      CHECK(!agx_nir_uniform_unit_global_id(b.shader));
      CHECK(count_intrinsic(b.shader, nir_intrinsic_load_workgroup_id) == 0);
      CHECK(count_intrinsic(b.shader, nir_intrinsic_load_global_invocation_id) == 1);
      ralloc_free(b.shader);
   }

   /* Internal compute shader: must skip. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "internal_cs");
      b.shader->info.internal = true;
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_store_global(&b, gid, nir_imm_int64(&b, 0x1000), .write_mask = 7,
                       .align_mul = 4);
      CHECK(!agx_nir_uniform_unit_global_id(b.shader));
      CHECK(count_intrinsic(b.shader, nir_intrinsic_load_workgroup_id) == 0);
      ralloc_free(b.shader);
   }
}

static void
test_variable_workgroup_skipped(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                  "var_wg");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size_variable = true;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_store_global(&b, gid, nir_imm_int64(&b, 0x1000), .write_mask = 7,
                    .align_mul = 4);

   CHECK(!agx_nir_uniform_unit_global_id(b.shader));
   CHECK(count_intrinsic(b.shader, nir_intrinsic_load_workgroup_id) == 0);
   CHECK(count_intrinsic(b.shader, nir_intrinsic_load_global_invocation_id) == 1);
   ralloc_free(b.shader);
}

static void
test_no_unit_axes_skipped(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                  "no_unit");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 2;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_store_global(&b, gid, nir_imm_int64(&b, 0x1000), .write_mask = 7,
                    .align_mul = 4);

   CHECK(!agx_nir_uniform_unit_global_id(b.shader));
   CHECK(count_intrinsic(b.shader, nir_intrinsic_load_workgroup_id) == 0);
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_partial_unit_axes_become_uniform();
   test_single_thread_and_internal_skipped();
   test_variable_workgroup_skipped();
   test_no_unit_axes_skipped();
   glsl_type_singleton_decref();
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
