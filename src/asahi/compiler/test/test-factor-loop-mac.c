/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Tests for agx_nir_opt_factor_loop_mac:
 *  - loop-invariant additive address splitting across nested loops
 *  - loop-invariant integer MAC reduction factoring (`acc += inv * val`)
 *  - linear-IV MAC strength reduction (`m += iv * val` -> prefix-sum `w += s`)
 *  - multi-impl (`@preamble` + `main`) metadata hygiene
 *  - negative guards: non-unit step IV, in-loop accumulator use, nested if
 */

#include "agx_nir.h"
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

static nir_phi_instr *
make_phi(nir_builder *b, nir_block *pre, nir_def *init, unsigned bit_size)
{
   nir_phi_instr *phi = nir_phi_instr_create(b->shader);
   nir_def_init(&phi->instr, &phi->def, 1, bit_size);
   nir_phi_instr_add_src(phi, pre, init);
   nir_builder_instr_insert(b, &phi->instr);
   return phi;
}

static void
finish_phi(nir_phi_instr *phi, nir_block *latch, nir_def *next)
{
   nir_phi_src *ps = nir_phi_instr_add_src(phi, latch, next);
   list_addtail(&ps->src.use_link, &next->uses);
}

static unsigned
count_alu_op_in_block(nir_block *blk, nir_op op)
{
   unsigned n = 0;
   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == op)
         n++;
   }
   return n;
}

/* 1. Nested 2D loop:
 *    - address: (y * stride + border + x) + v * stride + u
 *    - m01 += v * px (loop-invariant MAC -> sum += px, post-loop v * sum)
 *    - m10 += u * px (linear-IV MAC -> w += sum at top of body, post-loop
 *                     (u_exit - 1) * sum - w)
 */
static void
test_nested_2d_address_and_macs(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "nested_2d");

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *stride = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *border = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *limit = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *base_ptr = nir_imm_int64(&b, 0x10000);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *v_init = nir_imm_int(&b, -15);

   nir_block *outer_pre = nir_cursor_current_block(b.cursor);
   nir_loop *outer = nir_push_loop(&b);

   nir_phi_instr *v_phi = make_phi(&b, outer_pre, v_init, 32);
   nir_phi_instr *m01_outer = make_phi(&b, outer_pre, zero, 32);
   nir_phi_instr *m10_outer = make_phi(&b, outer_pre, zero, 32);

   nir_push_if(&b, nir_ilt(&b, limit, &v_phi->def));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *u_init = nir_imm_int(&b, -15);
   nir_block *inner_pre = nir_cursor_current_block(b.cursor);
   nir_loop *inner = nir_push_loop(&b);

   nir_phi_instr *u_phi = make_phi(&b, inner_pre, u_init, 32);
   nir_phi_instr *m01_inner = make_phi(&b, inner_pre, &m01_outer->def, 32);
   nir_phi_instr *m10_inner = make_phi(&b, inner_pre, &m10_outer->def, 32);

   nir_push_if(&b, nir_ilt(&b, limit, &u_phi->def));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *inner_body = nir_cursor_current_block(b.cursor);

   /* Address with mixed outer-invariant, row-invariant, and inner-variant terms:
    *   ((y + v) * stride) -> expanded via imadshl_agx(y, stride, border, 0) + x
    *   + imadshl_agx(v, stride, u, 0) */
   nir_def *a0 = nir_imadshl_agx(&b, y, stride, border, nir_imm_int(&b, 0));
   nir_def *a1 = nir_iadd(&b, a0, x);
   nir_def *a2 = nir_imadshl_agx(&b, &v_phi->def, stride, a1, nir_imm_int(&b, 0));
   nir_def *idx = nir_iadd(&b, a2, &u_phi->def);

   nir_def *addr = nir_iadd(&b, base_ptr, nir_u2u64(&b, idx));
   nir_def *px = nir_u2u32(&b, nir_load_global(&b, 1, 8, addr, .align_mul = 1));

   nir_def *m01_next =
      nir_imadshl_agx(&b, &v_phi->def, px, &m01_inner->def, nir_imm_int(&b, 0));
   nir_def *m10_next =
      nir_imadshl_agx(&b, &u_phi->def, px, &m10_inner->def, nir_imm_int(&b, 0));
   nir_def *u_next = nir_iadd_imm(&b, &u_phi->def, 1);

   finish_phi(u_phi, inner_body, u_next);
   finish_phi(m01_inner, inner_body, m01_next);
   finish_phi(m10_inner, inner_body, m10_next);
   nir_pop_loop(&b, inner);

   nir_def *v_next = nir_iadd_imm(&b, &v_phi->def, 1);
   nir_block *outer_latch = nir_cursor_current_block(b.cursor);
   finish_phi(v_phi, outer_latch, v_next);
   finish_phi(m01_outer, outer_latch, &m01_inner->def);
   finish_phi(m10_outer, outer_latch, &m10_inner->def);
   nir_pop_loop(&b, outer);

   nir_store_global(&b, nir_vec2(&b, &m01_outer->def, &m10_outer->def),
                    nir_imm_int64(&b, 0x20000), .write_mask = 3, .align_mul = 4);

   nir_validate_shader(b.shader, "before nested_2d");
   CHECK(agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after nested_2d");
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after nested_2d dce");

   /* Inner loop body must have ZERO imul and ZERO imadshl_agx remaining. */
   CHECK(count_alu_op_in_block(inner_body, nir_op_imul) == 0);
   CHECK(count_alu_op_in_block(inner_body, nir_op_imadshl_agx) == 0);

   /* Inner preheader must compute the row product (v * stride) while outer
    * preheader holds the outer-invariant (y * stride + border + x) product. */
   CHECK(count_alu_op_in_block(outer_pre, nir_op_imul) >= 1);
   CHECK(count_alu_op_in_block(inner_pre, nir_op_imul) >= 1);

   /* Inner exit (outer_latch) must have 2 imuls: v * sum and (u_exit-1) * sum. */
   CHECK(count_alu_op_in_block(outer_latch, nir_op_imul) == 2);

   ralloc_free(b.shader);
}

/* 2. Standalone 16-bit and 32-bit `acc += inv * val` using `iadd + imul`. */
static void
test_standalone_invariant_mac(unsigned bit_size)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "inv_mac");

   nir_def *inv = nir_load_preamble(&b, 1, bit_size, .base = 0);
   nir_def *n = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *acc0 = nir_load_preamble(&b, 1, bit_size, .base = 4);
   nir_def *base = nir_imm_int64(&b, 0x30000);
   nir_def *zero32 = nir_imm_int(&b, 0);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);

   nir_phi_instr *i_phi = make_phi(&b, pre, zero32, 32);
   nir_phi_instr *acc_phi = make_phi(&b, pre, acc0, bit_size);

   nir_push_if(&b, nir_ige(&b, &i_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *body = nir_cursor_current_block(b.cursor);
   nir_def *val = nir_load_global(&b, 1, bit_size,
                                  nir_iadd(&b, base, nir_u2u64(&b, &i_phi->def)),
                                  .align_mul = bit_size / 8);
   nir_def *acc_next = nir_iadd(&b, &acc_phi->def, nir_imul(&b, inv, val));
   nir_def *i_next = nir_iadd_imm(&b, &i_phi->def, 1);

   finish_phi(i_phi, body, i_next);
   finish_phi(acc_phi, body, acc_next);
   nir_pop_loop(&b, loop);

   nir_block *exit_blk = nir_cursor_current_block(b.cursor);
   nir_store_global(&b, &acc_phi->def, nir_imm_int64(&b, 0x40000),
                    .write_mask = 1, .align_mul = bit_size / 8);

   nir_validate_shader(b.shader, "before inv_mac");
   CHECK(agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after inv_mac");
   nir_opt_dce(b.shader);

   CHECK(count_alu_op_in_block(body, nir_op_imul) == 0);
   CHECK(count_alu_op_in_block(exit_blk, nir_op_imul) == 1);

   ralloc_free(b.shader);
}

/* 3. Pre-existing `sum += val` + `m += iv * val` (no inv*val MAC present). */
static void
test_preexisting_sum_with_linear_iv_mac(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "sum_lin");

   nir_def *n = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *m0 = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *base = nir_imm_int64(&b, 0x50000);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *iv_init = nir_imm_int(&b, -7);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);

   nir_phi_instr *iv_phi = make_phi(&b, pre, iv_init, 32);
   nir_phi_instr *sum_phi = make_phi(&b, pre, zero, 32);
   nir_phi_instr *m_phi = make_phi(&b, pre, m0, 32);

   nir_push_if(&b, nir_ige(&b, &iv_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *body = nir_cursor_current_block(b.cursor);
   nir_def *val = nir_load_global(&b, 1, 32,
                                  nir_iadd(&b, base, nir_u2u64(&b, &iv_phi->def)),
                                  .align_mul = 4);
   nir_def *sum_next = nir_iadd(&b, &sum_phi->def, val);
   nir_def *m_next = nir_iadd(&b, &m_phi->def, nir_imul(&b, &iv_phi->def, val));
   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 1);

   finish_phi(iv_phi, body, iv_next);
   finish_phi(sum_phi, body, sum_next);
   finish_phi(m_phi, body, m_next);
   nir_pop_loop(&b, loop);

   nir_block *exit_blk = nir_cursor_current_block(b.cursor);
   nir_store_global(&b, nir_vec2(&b, &sum_phi->def, &m_phi->def),
                    nir_imm_int64(&b, 0x60000), .write_mask = 3, .align_mul = 4);

   nir_validate_shader(b.shader, "before sum_lin");
   CHECK(agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after sum_lin");
   nir_opt_dce(b.shader);

   CHECK(count_alu_op_in_block(body, nir_op_imul) == 0);
   CHECK(count_alu_op_in_block(exit_blk, nir_op_imul) == 1);

   ralloc_free(b.shader);
}

/* 4. Multi-impl (`@preamble` + `main`) metadata hygiene. */
static void
test_multi_impl_with_preamble(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "multi_impl");

   nir_function *pre_fn = nir_function_create(b.shader, "preamble");
   pre_fn->is_preamble = true;
   nir_function_impl *pre_impl = nir_function_impl_create(pre_fn);
   nir_builder bp = nir_builder_at(nir_before_impl(pre_impl));
   nir_store_preamble(&bp, nir_imm_int(&bp, 42), .base = 0);

   nir_def *inv = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *n = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *base = nir_imm_int64(&b, 0x70000);
   nir_def *zero = nir_imm_int(&b, 0);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);
   nir_phi_instr *i_phi = make_phi(&b, pre, zero, 32);
   nir_phi_instr *acc_phi = make_phi(&b, pre, zero, 32);

   nir_push_if(&b, nir_ige(&b, &i_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *body = nir_cursor_current_block(b.cursor);
   nir_def *val = nir_load_global(&b, 1, 32,
                                  nir_iadd(&b, base, nir_u2u64(&b, &i_phi->def)),
                                  .align_mul = 4);
   nir_def *acc_next = nir_iadd(&b, &acc_phi->def, nir_imul(&b, inv, val));
   nir_def *i_next = nir_iadd_imm(&b, &i_phi->def, 1);

   finish_phi(i_phi, body, i_next);
   finish_phi(acc_phi, body, acc_next);
   nir_pop_loop(&b, loop);

   nir_store_global(&b, &acc_phi->def, nir_imm_int64(&b, 0x80000),
                    .write_mask = 1, .align_mul = 4);

   nir_validate_shader(b.shader, "before multi_impl");
   CHECK(agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after multi_impl");

   ralloc_free(b.shader);
}

/* 5. Negative: non-unit step IV (`iv += 2`) must NOT strength-reduce `m += iv * val`. */
static void
test_negative_non_unit_step_iv(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "step2");

   nir_def *n = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *base = nir_imm_int64(&b, 0x90000);
   nir_def *zero = nir_imm_int(&b, 0);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);
   nir_phi_instr *iv_phi = make_phi(&b, pre, zero, 32);
   nir_phi_instr *sum_phi = make_phi(&b, pre, zero, 32);
   nir_phi_instr *m_phi = make_phi(&b, pre, zero, 32);

   nir_push_if(&b, nir_ige(&b, &iv_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *body = nir_cursor_current_block(b.cursor);
   nir_def *val = nir_load_global(&b, 1, 32,
                                  nir_iadd(&b, base, nir_u2u64(&b, &iv_phi->def)),
                                  .align_mul = 4);
   nir_def *sum_next = nir_iadd(&b, &sum_phi->def, val);
   nir_def *m_next = nir_iadd(&b, &m_phi->def, nir_imul(&b, &iv_phi->def, val));
   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 2);

   finish_phi(iv_phi, body, iv_next);
   finish_phi(sum_phi, body, sum_next);
   finish_phi(m_phi, body, m_next);
   nir_pop_loop(&b, loop);

   nir_store_global(&b, nir_vec2(&b, &sum_phi->def, &m_phi->def),
                    nir_imm_int64(&b, 0xa0000), .write_mask = 3, .align_mul = 4);

   nir_validate_shader(b.shader, "before step2");
   CHECK(!agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after step2");
   CHECK(count_alu_op_in_block(body, nir_op_imul) == 1);

   ralloc_free(b.shader);
}

/* 6. Negative: accumulator used inside loop body outside its own update. */
static void
test_negative_in_loop_acc_use(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "acc_use");

   nir_def *inv = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *n = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *base = nir_imm_int64(&b, 0xb0000);
   nir_def *zero = nir_imm_int(&b, 0);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);
   nir_phi_instr *i_phi = make_phi(&b, pre, zero, 32);
   nir_phi_instr *acc_phi = make_phi(&b, pre, zero, 32);

   nir_push_if(&b, nir_ige(&b, &i_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_block *body = nir_cursor_current_block(b.cursor);
   nir_def *val = nir_load_global(&b, 1, 32,
                                  nir_iadd(&b, base, nir_u2u64(&b, &i_phi->def)),
                                  .align_mul = 4);
   nir_store_global(&b, &acc_phi->def, nir_imm_int64(&b, 0xc0000),
                    .write_mask = 1, .align_mul = 4);
   nir_def *acc_next = nir_iadd(&b, &acc_phi->def, nir_imul(&b, inv, val));
   nir_def *i_next = nir_iadd_imm(&b, &i_phi->def, 1);

   finish_phi(i_phi, body, i_next);
   finish_phi(acc_phi, body, acc_next);
   nir_pop_loop(&b, loop);

   nir_store_global(&b, &acc_phi->def, nir_imm_int64(&b, 0xd0000),
                    .write_mask = 1, .align_mul = 4);

   nir_validate_shader(b.shader, "before acc_use");
   CHECK(!agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after acc_use");

   ralloc_free(b.shader);
}

/* 7. Negative: loop body with nested `if` must NOT factor MAC reductions. */
static void
test_negative_nested_if_in_body(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "nested_if");

   nir_def *inv = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *n = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *base = nir_imm_int64(&b, 0xe0000);
   nir_def *zero = nir_imm_int(&b, 0);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);
   nir_phi_instr *i_phi = make_phi(&b, pre, zero, 32);
   nir_phi_instr *acc_phi = make_phi(&b, pre, zero, 32);

   nir_push_if(&b, nir_ige(&b, &i_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *val = nir_load_global(&b, 1, 32,
                                  nir_iadd(&b, base, nir_u2u64(&b, &i_phi->def)),
                                  .align_mul = 4);
   nir_push_if(&b, nir_ine_imm(&b, val, 0));
   nir_def *acc_then = nir_iadd(&b, &acc_phi->def, nir_imul(&b, inv, val));
   nir_push_else(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_def *acc_next = nir_if_phi(&b, acc_then, &acc_phi->def);
   nir_def *i_next = nir_iadd_imm(&b, &i_phi->def, 1);

   nir_block *latch = nir_cursor_current_block(b.cursor);
   finish_phi(i_phi, latch, i_next);
   finish_phi(acc_phi, latch, acc_next);
   nir_pop_loop(&b, loop);

   nir_store_global(&b, &acc_phi->def, nir_imm_int64(&b, 0xf0000),
                    .write_mask = 1, .align_mul = 4);

   nir_validate_shader(b.shader, "before nested_if");
   CHECK(!agx_nir_opt_factor_loop_mac(b.shader));
   nir_validate_shader(b.shader, "after nested_if");

   ralloc_free(b.shader);
}

int
main(void)
{
   test_nested_2d_address_and_macs();
   test_standalone_invariant_mac(32);
   test_standalone_invariant_mac(16);
   test_preexisting_sum_with_linear_iv_mac();
   test_multi_impl_with_preamble();
   test_negative_non_unit_step_iv();
   test_negative_in_loop_acc_use();
   test_negative_nested_if_in_body();

   if (failures) {
      fprintf(stderr, "%u test failure(s)\n", failures);
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}
