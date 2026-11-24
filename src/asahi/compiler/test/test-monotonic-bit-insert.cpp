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
make_buffer_addr(nir_builder *b, nir_def *base_u64, nir_def *byte_offset)
{
   return nir_iadd(b, base_u64, nir_u2u64(b, byte_offset));
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
count_intrinsics(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
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

static inline unsigned
count_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(block->cf_node.parent);
            if (block == nir_if_first_then_block(nif))
               n++;
         }
      }
   }
   return n;
}

TEST(MonotonicBitInsert, MonotonicBitInsertBranchFold)
{
   /* Inside a 0..31 unit-stepped loop, `(acc | (1u << (i & 31))) != acc` is
    * always true in the loop body when `acc` starts at 0 and only ever sets
    * bit `i`, but must not fold when `acc` starts nonzero OR when tested in
    * the loop header before the `i >= 32` exit break (where i == 32 wraps
    * `1u << (32 & 31)` back to bit 0). */
   const char *names[3] = {
      "bit-init-zero",
      "bit-init-nonzero",
      "bit-header-before-break",
   };
   for (int mode = 0; mode < 3; ++mode) {
      nir_builder b = make_compute_shader(names[mode]);
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);

      nir_def *iv_init = nir_imm_int(&b, 0);
      nir_def *acc_init = nir_imm_int(&b, mode == 1 ? 1 : 0);

      nir_loop *loop = nir_push_loop(&b);
      nir_block *pre = nir_loop_predecessor_block(loop);
      nir_block *header = nir_loop_first_block(loop);

      nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
      nir_phi_instr_add_src(iv_phi, pre, iv_init);

      nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
      nir_phi_instr_add_src(acc_phi, pre, acc_init);

      nir_def *hdr_cond = NULL;
      nir_def *hdr_with_bit = NULL;
      if (mode == 2) {
         nir_def *hbit = nir_ishl(&b, nir_imm_int(&b, 1),
                                  nir_iand_imm(&b, &iv_phi->def, 31));
         hdr_with_bit = nir_ior(&b, &acc_phi->def, hbit);
         hdr_cond = nir_ine(&b, hdr_with_bit, &acc_phi->def);
      }

      nir_push_if(&b, nir_ige_imm(&b, &iv_phi->def, 32));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *with_bit = hdr_with_bit;
      nir_def *cond = hdr_cond;
      if (mode != 2) {
         nir_def *bit = nir_ishl(&b, nir_imm_int(&b, 1),
                                 nir_iand_imm(&b, &iv_phi->def, 31));
         with_bit = nir_ior(&b, &acc_phi->def, bit);
         cond = nir_ine(&b, with_bit, &acc_phi->def);
      }

      nir_if *nif = nir_push_if(&b, cond);
      nir_def *sample = nir_load_global(&b, 1, 32,
                                        make_buffer_addr(&b, base,
                                                         nir_ishl_imm(&b, &iv_phi->def, 2)),
                                        .align_mul = 4);
      nir_def *sel = nir_bcsel(&b, nir_ine_imm(&b, sample, 0), with_bit, &acc_phi->def);
      nir_pop_if(&b, nif);

      nir_def *acc_next = nir_if_phi(&b, sel, &acc_phi->def);
      nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 1);

      nir_block *latch = nir_loop_last_block(loop);
      nir_phi_instr_add_src(iv_phi, latch, iv_next);
      nir_phi_instr_add_src(acc_phi, latch, acc_next);
      nir_instr_insert(nir_before_block(header), &iv_phi->instr);
      nir_instr_insert(nir_before_block(header), &acc_phi->instr);
      nir_pop_loop(&b, loop);

      nir_def *res = &acc_phi->def;
      if (mode == 2)
         res = nir_bcsel(&b, hdr_cond, res, nir_imm_int(&b, 0));
      nir_store_global(&b, res, out, .align_mul = 4);
      nir_validate_shader(b.shader, "bit-insert setup");

      ASSERT_EQ(count_ifs(b.shader), 2u);
      bool changed = agx_nir_opt_monotonic_bit_insert(b.shader);
      nir_validate_shader(b.shader, "bit-insert lowered");
      if (mode == 0) {
         EXPECT_TRUE(changed);
         EXPECT_EQ(count_ifs(b.shader), 1u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(count_ifs(b.shader), 2u);
      }
      ralloc_free(b.shader);
   }
}
