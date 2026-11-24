/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
#include "agx_compiler.h"
}

static inline nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static inline nir_def *
make_desc_base(nir_builder *b, int preamble_base)
{
   nir_def *set = nir_load_preamble(b, 1, 64, .base = preamble_base);
   nir_def *desc = nir_load_global_constant(
      b, 4, 32, set,
      .access = (gl_access_qualifier)ACCESS_CAN_SPECULATE,
      .align_mul = 16);
   return nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                 nir_channel(b, desc, 1));
}

static inline unsigned
count_alu_op(nir_shader *shader, nir_op op)
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

TEST(IvDivMod, IvDivModStrengthReduction)
{
   /* Unit-stepped 0-based loop with runtime loop-invariant divisor `D` lowers
    * idiv/imod into a running (q, r) recurrence (zero sign fixup when D > 0 is
    * guarded, exact fixup otherwise), while step != 1 stays untouched. */
   for (int mode = 0; mode < 3; ++mode) {
      /* mode 0: guarded D > 0, step 1; mode 1: unknown-sign D, step 1;
       * mode 2: step 2 (must not match). */
      nir_builder b = make_compute_shader(mode == 0 ? "iv-divmod-pos" :
                                          mode == 1 ? "iv-divmod-any" :
                                                      "iv-divmod-step2");
      nir_def *out = make_desc_base(&b, 8);
      nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *d = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);

      nir_if *guard = NULL;
      if (mode == 0)
         guard = nir_push_if(&b, nir_ilt(&b, nir_imm_int(&b, 0), d));

      nir_def *iv_init = nir_imm_int(&b, 0);
      nir_def *acc_init = nir_imm_int(&b, 0);

      nir_loop *loop = nir_push_loop(&b);
      nir_block *pre = nir_loop_predecessor_block(loop);
      nir_block *header = nir_loop_first_block(loop);

      nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
      nir_phi_instr_add_src(iv_phi, pre, iv_init);

      nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
      nir_phi_instr_add_src(acc_phi, pre, acc_init);

      nir_push_if(&b, nir_ige(&b, &iv_phi->def, n));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *q = nir_idiv(&b, &iv_phi->def, d);
      nir_def *r = nir_imod(&b, &iv_phi->def, d);
      nir_def *sum = nir_iadd(&b, &acc_phi->def, nir_iadd(&b, q, r));
      nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, mode == 2 ? 2 : 1);

      nir_block *latch = nir_loop_last_block(loop);
      nir_phi_instr_add_src(iv_phi, latch, iv_next);
      nir_phi_instr_add_src(acc_phi, latch, sum);
      nir_instr_insert(nir_before_block(header), &iv_phi->instr);
      nir_instr_insert(nir_before_block(header), &acc_phi->instr);
      nir_pop_loop(&b, loop);

      nir_store_global(&b, &acc_phi->def, out, .align_mul = 4);
      if (guard)
         nir_pop_if(&b, guard);
      nir_validate_shader(b.shader, "iv-divmod setup");

      bool changed = agx_nir_opt_iv_divmod(b.shader);
      nir_validate_shader(b.shader, "iv-divmod lowered");
      if (mode < 2) {
         EXPECT_TRUE(changed);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u);
         if (mode == 0)
            EXPECT_EQ(count_alu_op(b.shader, nir_op_ineg), 0u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 1u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 1u);
      }
      ralloc_free(b.shader);
   }
}
