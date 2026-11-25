/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx_nir_opt_uniform_stride_chains: serial 32-bit uniform-stride addition
 * chains (k >= 3) flatten to base + (j * stride) only when stride is
 * preamble-hoistable so the uniform multiples move into the preamble and
 * break serial address dependencies without adding main-shader multiplies.
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

static nir_def *
uniform_u32(nir_builder *b, unsigned offset)
{
   return nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, offset),
                       .align_mul = 4, .align_offset = 0, .range_base = offset,
                       .range = 4);
}

static void
test_flattens_hoistable_uniform_stride_chain(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                  "stride_chain");
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *base = nir_channel(&b, gid, 0);
   nir_def *stride = uniform_u32(&b, 0);

   nir_def *v1 = nir_iadd(&b, base, stride);
   nir_def *v2 = nir_iadd(&b, v1, stride);
   nir_def *v3 = nir_iadd(&b, v2, stride);
   nir_def *v4 = nir_iadd(&b, v3, stride);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, v1, v2), nir_iadd(&b, v3, v4));
   nir_store_global(&b, sum, nir_imm_int64(&b, 0x1000), .write_mask = 1,
                    .align_mul = 4);
   nir_validate_shader(b.shader, "before");

   CHECK(agx_nir_opt_uniform_stride_chains(b.shader, false));
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after");

   /* Expect uniform stride multiples (2*s via ishl, 3*s via imul, 4*s via ishl)
    * and every rewritten add reading gid.x directly rather than v_{j-1}. */
   unsigned muls = 0, shls = 0, base_adds = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op == nir_op_imul)
            muls++;
         if (alu->op == nir_op_ishl)
            shls++;
         if (alu->op == nir_op_iadd &&
             (alu->src[0].src.ssa == gid || alu->src[1].src.ssa == gid))
            base_adds++;
      }
   }
   CHECK(muls == 1);
   CHECK(shls == 2);
   CHECK(base_adds == 4);
   ralloc_free(b.shader);
}

static void
test_skips_short_const_and_divergent_chains(void)
{
   /* Short chain (k == 2): must not transform. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "short_chain");
      nir_def *base = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *stride = uniform_u32(&b, 0);
      nir_def *v1 = nir_iadd(&b, base, stride);
      nir_def *v2 = nir_iadd(&b, v1, stride);
      nir_store_global(&b, nir_iadd(&b, v1, v2), nir_imm_int64(&b, 0x1000),
                       .write_mask = 1, .align_mul = 4);
      CHECK(!agx_nir_opt_uniform_stride_chains(b.shader, false));
      ralloc_free(b.shader);
   }

   /* Constant stride: already folds cleanly, must not transform. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "const_stride");
      nir_def *base = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *c = nir_imm_int(&b, 16);
      nir_def *v1 = nir_iadd(&b, base, c);
      nir_def *v2 = nir_iadd(&b, v1, c);
      nir_def *v3 = nir_iadd(&b, v2, c);
      nir_store_global(&b, nir_iadd(&b, v1, nir_iadd(&b, v2, v3)),
                       nir_imm_int64(&b, 0x1000), .write_mask = 1,
                       .align_mul = 4);
      CHECK(!agx_nir_opt_uniform_stride_chains(b.shader, false));
      ralloc_free(b.shader);
   }

   /* Divergent stride: must not transform. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "div_stride");
      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_def *base = nir_channel(&b, gid, 0);
      nir_def *stride = nir_channel(&b, gid, 1);
      nir_def *v1 = nir_iadd(&b, base, stride);
      nir_def *v2 = nir_iadd(&b, v1, stride);
      nir_def *v3 = nir_iadd(&b, v2, stride);
      nir_store_global(&b, nir_iadd(&b, v1, nir_iadd(&b, v2, v3)),
                       nir_imm_int64(&b, 0x1000), .write_mask = 1,
                       .align_mul = 4);
      CHECK(!agx_nir_opt_uniform_stride_chains(b.shader, false));
      ralloc_free(b.shader);
   }
}

static void
test_skips_non_hoistable_uniform_strides(void)
{
   /* Workgroup-uniform stride (load_workgroup_id) is uniform within a
    * workgroup but cannot hoist into a dispatch preamble: must not add
    * main-shader multiplies. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "wgid_stride");
      nir_def *base = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *stride = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *v1 = nir_iadd(&b, base, stride);
      nir_def *v2 = nir_iadd(&b, v1, stride);
      nir_def *v3 = nir_iadd(&b, v2, stride);
      nir_store_global(&b, nir_iadd(&b, v1, nir_iadd(&b, v2, v3)),
                       nir_imm_int64(&b, 0x1000), .write_mask = 1,
                       .align_mul = 4);
      nir_validate_shader(b.shader, "wgid_before");
      CHECK(!agx_nir_opt_uniform_stride_chains(b.shader, false));
      ralloc_free(b.shader);
   }

   /* Non-reorderable SSBO load is uniform but cannot hoist into the preamble:
    * must not transform. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "ssbo_stride");
      nir_def *base = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *stride = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0),
                                      nir_imm_int(&b, 0), .align_mul = 4,
                                      .align_offset = 0);
      nir_def *v1 = nir_iadd(&b, base, stride);
      nir_def *v2 = nir_iadd(&b, v1, stride);
      nir_def *v3 = nir_iadd(&b, v2, stride);
      nir_store_global(&b, nir_iadd(&b, v1, nir_iadd(&b, v2, v3)),
                       nir_imm_int64(&b, 0x1000), .write_mask = 1,
                       .align_mul = 4);
      nir_validate_shader(b.shader, "ssbo_before");
      CHECK(!agx_nir_opt_uniform_stride_chains(b.shader, false));
      ralloc_free(b.shader);
   }

   /* Loop-varying uniform stride (derived from a loop header PHI): uniform to
    * divergence analysis, but not preamble-hoistable -> must not transform. */
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                                     "loop_stride");
      nir_def *base = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *init = uniform_u32(&b, 0);
      nir_def *limit = uniform_u32(&b, 4);
      nir_variable *acc =
         nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *step =
         nir_local_variable_create(b.impl, glsl_uint_type(), "step");
      nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, step, init, 1);

      nir_push_loop(&b);
      {
         nir_def *s = nir_load_var(&b, step);
         nir_push_if(&b, nir_uge(&b, s, limit));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *v1 = nir_iadd(&b, base, s);
         nir_def *v2 = nir_iadd(&b, v1, s);
         nir_def *v3 = nir_iadd(&b, v2, s);
         nir_store_var(&b, acc,
                       nir_iadd(&b, nir_load_var(&b, acc),
                                nir_iadd(&b, v1, nir_iadd(&b, v2, v3))),
                       1);
         nir_store_var(&b, step, nir_iadd_imm(&b, s, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_global(&b, nir_load_var(&b, acc), nir_imm_int64(&b, 0x1000),
                       .write_mask = 1, .align_mul = 4);
      nir_lower_vars_to_ssa(b.shader);
      nir_validate_shader(b.shader, "loop_before");
      CHECK(!agx_nir_opt_uniform_stride_chains(b.shader, false));
      ralloc_free(b.shader);
   }

   /* Texture-derived uniform stride respects no_textures (M5 policy). */
   {
      for (unsigned i = 0; i < 2; ++i) {
         const bool no_textures = i != 0;
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "tex_stride");
         nir_def *base =
            nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
         tex->op = nir_texop_txf;
         tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
         tex->dest_type = nir_type_uint32;
         tex->coord_components = 2;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                           nir_imm_ivec2(&b, 0, 0));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_def *stride = nir_channel(&b, &tex->def, 0);
         nir_def *v1 = nir_iadd(&b, base, stride);
         nir_def *v2 = nir_iadd(&b, v1, stride);
         nir_def *v3 = nir_iadd(&b, v2, stride);
         nir_store_global(&b, nir_iadd(&b, v1, nir_iadd(&b, v2, v3)),
                          nir_imm_int64(&b, 0x1000), .write_mask = 1,
                          .align_mul = 4);
         nir_validate_shader(b.shader, "tex_before");
         CHECK(agx_nir_opt_uniform_stride_chains(b.shader, no_textures) ==
               !no_textures);
         ralloc_free(b.shader);
      }
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_flattens_hoistable_uniform_stride_chain();
   test_skips_short_const_and_divergent_chains();
   test_skips_non_hoistable_uniform_strides();
   glsl_type_singleton_decref();
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
