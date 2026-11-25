/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Unit tests for agx_nir_opt_merge_nested_early_exits,
 * agx_nir_hoist_loop_invariant_guard_bits, and comparison-first boolean
 * select fusion in agx_nir_fuse_selects.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include <limits.h>
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
count_ifs_in_list(struct exec_list *list)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         n += 1 + count_ifs_in_list(&nif->then_list) +
              count_ifs_in_list(&nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         n += count_ifs_in_list(&nir_cf_node_as_loop(node)->body);
      }
   }
   return n;
}

static unsigned
count_op_in_block(nir_block *blk, nir_op op)
{
   unsigned n = 0;
   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == op)
         n++;
   }
   return n;
}

static unsigned
count_op_in_impl(nir_function_impl *impl, nir_op op)
{
   unsigned n = 0;
   nir_foreach_block(blk, impl)
      n += count_op_in_block(blk, op);
   return n;
}

static unsigned
count_intr_in_impl(nir_function_impl *impl, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == op)
            n++;
      }
   }
   return n;
}

/*
 * Positive test 1: nested identical store_agx early exits inside a 2D loop
 * merge into one branch, fold `!(2 < x) || (x < border)` into
 * `x < imax(3, border)` (same for `y`), and hoist the loop-invariant `y`
 * guard disjunction out of the inner `x` loop.
 */
static void
test_merge_nested_bounds_and_hoist(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "nested_bounds");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 2;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *x0 = nir_channel(&b, gid, 0);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 20);
   nir_def *w = nir_load_preamble(&b, 1, 32, .base = 8);
   nir_def *h_m3 = nir_load_preamble(&b, 1, 32, .base = 10);
   nir_def *w_m3 = nir_load_preamble(&b, 1, 32, .base = 12);
   nir_def *border = nir_load_preamble(&b, 1, 32, .base = 14);
   nir_def *h_mb = nir_load_preamble(&b, 1, 32, .base = 16);
   nir_def *w_mb = nir_load_preamble(&b, 1, 32, .base = 18);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *preheader = nir_loop_predecessor_block(loop);

   nir_phi_instr *x_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&x_phi->instr, &x_phi->def, 1, 32);
   nir_phi_instr_add_src(x_phi, preheader, x0);
   nir_builder_instr_insert(&b, &x_phi->instr);
   nir_def *x = &x_phi->def;

   nir_if *brk = nir_push_if(&b, nir_ige(&b, x, w));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, brk);

   nir_def *idx = nir_iadd(&b, nir_imul(&b, y, w), x);

   /* Outer guard: !(2 < y && y < H-3 && 2 < x && x < W-3) */
   nir_def *in_outer =
      nir_iand(&b,
               nir_iand(&b, nir_ilt(&b, nir_imm_int(&b, 2), y),
                            nir_ilt(&b, y, h_m3)),
               nir_iand(&b, nir_ilt(&b, nir_imm_int(&b, 2), x),
                            nir_ilt(&b, x, w_m3)));
   nir_if *if1 = nir_push_if(&b, nir_inot(&b, in_outer));
   {
      nir_def *p = nir_load_preamble(&b, 1, 64, .base = 20);
      nir_store_agx(&b, nir_imm_int(&b, 0), p, idx,
                    .format = PIPE_FORMAT_R32_UINT);
   }
   nir_push_else(&b, if1);
   {
      /* Inner pure prelude + guard: (y < border || H-border < y || x < border || W-border < x) */
      nir_def *out_inner =
         nir_ior(&b,
                 nir_ior(&b, nir_ilt(&b, y, border), nir_ilt(&b, h_mb, y)),
                 nir_ior(&b, nir_ilt(&b, x, border), nir_ilt(&b, w_mb, x)));
      nir_if *if2 = nir_push_if(&b, out_inner);
      {
         nir_def *p = nir_load_preamble(&b, 1, 64, .base = 20);
         nir_store_agx(&b, nir_imm_int(&b, 0), p, idx,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_push_else(&b, if2);
      {
         nir_store_agx(&b, nir_imm_int(&b, 42), out_ptr, idx,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_pop_if(&b, if2);
   }
   nir_pop_if(&b, if1);

   nir_def *x_next = nir_iadd_imm(&b, x, 32);
   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_src *latch_src = nir_phi_instr_add_src(x_phi, latch, x_next);
   list_addtail(&latch_src->src.use_link, &x_next->uses);
   nir_pop_loop(&b, loop);

   nir_validate_shader(b.shader, "before merge");

   /* Before: 3 ifs (1 break + 2 nested early exits), 3 store_agx. */
   CHECK(count_ifs_in_list(&b.impl->body) == 3);
   CHECK(count_intr_in_impl(b.impl, nir_intrinsic_store_agx) == 3);

   bool progress = agx_nir_opt_merge_nested_early_exits(b.shader);
   CHECK(progress);
   nir_validate_shader(b.shader, "after merge");

   nir_opt_cse(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after dce");

   /* After merge: only 2 ifs (break + single merged early exit), 2 store_agx,
    * imax(3, border) folded for both y and x, and y's comparisons hoisted to
    * the preheader. */
   CHECK(count_ifs_in_list(&b.impl->body) == 2);
   CHECK(count_intr_in_impl(b.impl, nir_intrinsic_store_agx) == 2);
   CHECK(count_op_in_impl(b.impl, nir_op_imax) >= 1);

   nir_block *pre = nir_loop_predecessor_block(loop);
   unsigned pre_cmps = count_op_in_block(pre, nir_op_ilt) +
                       count_op_in_block(pre, nir_op_ige);
   CHECK(pre_cmps >= 2);

   ralloc_free(b.shader);
}

/*
 * Positive test 2: agx_nir_hoist_loop_invariant_guard_bits hoists a sunk
 * loop-invariant 1-bit comparison chain (with >=2 comparisons) out of a loop
 * body back into the loop preheader.
 */
static void
test_hoist_loop_invariant_guard_bits(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "hoist_guard");
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *x0 = nir_channel(&b, gid, 0);
   nir_def *w = nir_load_preamble(&b, 1, 32, .base = 8);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 20);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *preheader = nir_loop_predecessor_block(loop);

   nir_phi_instr *x_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&x_phi->instr, &x_phi->def, 1, 32);
   nir_phi_instr_add_src(x_phi, preheader, x0);
   nir_builder_instr_insert(&b, &x_phi->instr);
   nir_def *x = &x_phi->def;

   nir_if *brk = nir_push_if(&b, nir_ige(&b, x, w));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, brk);

   /* Sunk loop-invariant y comparisons combined with x comparison inside loop. */
   nir_def *lim_lo = nir_load_preamble(&b, 1, 32, .base = 10);
   nir_def *lim_hi = nir_load_preamble(&b, 1, 32, .base = 12);
   nir_def *y_lo = nir_ilt(&b, y, lim_lo);
   nir_def *y_hi = nir_ige(&b, y, lim_hi);
   nir_def *y_oob = nir_bcsel(&b, y_lo, nir_imm_true(&b), y_hi);
   nir_def *x_oob = nir_ilt(&b, x, lim_lo);
   nir_def *cond = nir_bcsel(&b, x_oob, nir_imm_true(&b), y_oob);

   nir_if *guard = nir_push_if(&b, cond);
   {
      nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                    .format = PIPE_FORMAT_R32_UINT);
   }
   nir_pop_if(&b, guard);

   nir_def *x_next = nir_iadd_imm(&b, x, 32);
   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_src *latch_src = nir_phi_instr_add_src(x_phi, latch, x_next);
   list_addtail(&latch_src->src.use_link, &x_next->uses);
   nir_pop_loop(&b, loop);

   nir_validate_shader(b.shader, "before guard hoist");
   CHECK(count_op_in_block(preheader, nir_op_ilt) == 0);
   CHECK(agx_nir_hoist_loop_invariant_guard_bits(b.shader));
   nir_validate_shader(b.shader, "after guard hoist");

   unsigned pre_cmps = count_op_in_block(preheader, nir_op_ilt) +
                       count_op_in_block(preheader, nir_op_ige);
   CHECK(pre_cmps == 2);

   ralloc_free(b.shader);
}

/*
 * Positive test 2: agx_nir_fuse_selects prefers putting a fusable comparison
 * on bcsel.src[0] even when the accumulator is src[0] of nir_ior.
 */
static void
test_fuse_selects_prefers_comparison(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "fuse_cmp");
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *a = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *c = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *d = nir_load_preamble(&b, 1, 32, .base = 4);

   nir_def *cmp0 = nir_ilt(&b, x, a);
   nir_def *cmp1 = nir_ilt(&b, x, c);
   nir_def *cmp2 = nir_ige(&b, x, d);

   /* Accumulator on src[0], fresh comparison on src[1]. */
   nir_def *or0 = nir_ior(&b, cmp0, cmp1);
   nir_def *or1 = nir_ior(&b, or0, cmp2);

   nir_def *res = nir_bcsel(&b, or1, nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_store_agx(&b, res, nir_load_preamble(&b, 1, 64, .base = 8),
                 nir_imm_int(&b, 0), .format = PIPE_FORMAT_R32_UINT);

   CHECK(agx_nir_fuse_selects(b.shader));
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after fuse_selects");

   /* Every bcsel replacing an ior must have a comparison (ilt/ige) as src[0],
    * never another bcsel. */
   bool found_outer = false;
   nir_foreach_block(blk, b.impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op == nir_op_bcsel && alu->def.bit_size == 1) {
            nir_alu_instr *cond = nir_src_as_alu(alu->src[0].src);
            CHECK(cond && (cond->op == nir_op_ilt || cond->op == nir_op_ige));
            found_outer = true;
         }
      }
   }
   CHECK(found_outer);

   ralloc_free(b.shader);
}

/*
 * Negative test 1: inner prelude has a non-speculatable memory load -> MUST
 * NOT merge (would speculate the load before the outer guard).
 */
static void
test_negative_impure_prelude(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "impure_pre");
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *in_ptr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_if *if1 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 3)));
   {
      nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                    .format = PIPE_FORMAT_R32_UINT);
   }
   nir_push_else(&b, if1);
   {
      /* Non-speculatable global read guarded by if1! */
      nir_def *v = nir_load_agx(&b, 1, 32, in_ptr, x,
                                .format = PIPE_FORMAT_R32_UINT);
      nir_if *if2 = nir_push_if(&b, nir_ieq_imm(&b, v, 0));
      {
         nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_push_else(&b, if2);
      {
         nir_store_agx(&b, v, out_ptr, x, .format = PIPE_FORMAT_R32_UINT);
      }
      nir_pop_if(&b, if2);
   }
   nir_pop_if(&b, if1);

   CHECK(!agx_nir_opt_merge_nested_early_exits(b.shader));
   CHECK(count_ifs_in_list(&b.impl->body) == 2);
   nir_validate_shader(b.shader, "negative impure prelude");
   ralloc_free(b.shader);
}

/*
 * Negative test 2: outer and inner early-exit stores write different values
 * (0 vs 1) -> MUST NOT merge.
 */
static void
test_negative_mismatched_exit_arms(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "mismatch");
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_if *if1 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 3)));
   {
      nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                    .format = PIPE_FORMAT_R32_UINT);
   }
   nir_push_else(&b, if1);
   {
      nir_if *if2 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 5)));
      {
         nir_store_agx(&b, nir_imm_int(&b, 1), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_push_else(&b, if2);
      {
         nir_store_agx(&b, nir_imm_int(&b, 2), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_pop_if(&b, if2);
   }
   nir_pop_if(&b, if1);

   CHECK(!agx_nir_opt_merge_nested_early_exits(b.shader));
   CHECK(count_ifs_in_list(&b.impl->body) == 2);
   nir_validate_shader(b.shader, "negative mismatched");
   ralloc_free(b.shader);
}

/*
 * Negative test 3: INT32_MAX inclusive bound `!(INT32_MAX < x)` must NOT wrap
 * to `x < INT32_MIN`.
 */
static void
test_negative_int32_max_no_wrap(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "intmax");
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *lim = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 4);

   /* !(INT32_MAX < x) is always true; must NOT turn into (x < INT32_MIN) */
   nir_if *if1 =
      nir_push_if(&b, nir_inot(&b, nir_ilt(&b, nir_imm_int(&b, INT32_MAX), x)));
   {
      nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                    .format = PIPE_FORMAT_R32_UINT);
   }
   nir_push_else(&b, if1);
   {
      nir_if *if2 = nir_push_if(&b, nir_ilt(&b, x, lim));
      {
         nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_push_else(&b, if2);
      {
         nir_store_agx(&b, nir_imm_int(&b, 7), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_pop_if(&b, if2);
   }
   nir_pop_if(&b, if1);

   CHECK(agx_nir_opt_merge_nested_early_exits(b.shader));
   nir_validate_shader(b.shader, "intmax no wrap");

   /* Must NOT have produced imax with INT32_MIN (wrapped INT32_MAX + 1). */
   CHECK(count_op_in_impl(b.impl, nir_op_imax) == 0);
   ralloc_free(b.shader);
}

/*
 * Multi-impl metadata hygiene test: verify a shader with an untouched
 * @preamble function validates cleanly when main transforms.
 */
static void
test_multi_impl_metadata_hygiene(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "multi_impl");
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 0);

   nir_if *if1 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 3)));
   {
      nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                    .format = PIPE_FORMAT_R32_UINT);
   }
   nir_push_else(&b, if1);
   {
      nir_if *if2 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 5)));
      {
         nir_store_agx(&b, nir_imm_int(&b, 0), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_push_else(&b, if2);
      {
         nir_store_agx(&b, nir_imm_int(&b, 9), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_pop_if(&b, if2);
   }
   nir_pop_if(&b, if1);

   /* Attach a dummy @preamble function that will not be modified. */
   nir_function *pre_fn = nir_function_create(b.shader, "@preamble");
   pre_fn->is_preamble = true;
   nir_function_impl *pre_impl = nir_function_impl_create(pre_fn);
   nir_builder pb = nir_builder_at(nir_before_impl(pre_impl));
   nir_store_preamble(&pb, nir_imm_int(&pb, 1), .base = 0);
   b.impl->preamble = pre_fn;

   nir_validate_shader(b.shader, "multi_impl before");
   CHECK(agx_nir_opt_merge_nested_early_exits(b.shader));
   nir_validate_shader(b.shader, "multi_impl after merge");
   agx_nir_hoist_loop_invariant_guard_bits(b.shader);
   nir_validate_shader(b.shader, "multi_impl after hoist");

   ralloc_free(b.shader);
}

/*
 * Empty-arm safety test: nested if with empty exit blocks must not dereference
 * NULL when checking structural equivalence (exec_list_get_head returns NULL
 * on empty lists, unlike exec_list_get_head_raw).
 */
static void
test_empty_arm_no_crash(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "empty_arm");
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *out_ptr = nir_load_preamble(&b, 1, 64, .base = 0);

   nir_if *if1 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 3)));
   {
      /* Intentionally empty outer then-arm. */
   }
   nir_push_else(&b, if1);
   {
      nir_if *if2 = nir_push_if(&b, nir_ilt(&b, x, nir_imm_int(&b, 5)));
      {
         /* Intentionally empty inner then-arm. */
      }
      nir_push_else(&b, if2);
      {
         nir_store_agx(&b, nir_imm_int(&b, 9), out_ptr, x,
                       .format = PIPE_FORMAT_R32_UINT);
      }
      nir_pop_if(&b, if2);
   }
   nir_pop_if(&b, if1);

   nir_validate_shader(b.shader, "empty_arm before");
   CHECK(!agx_nir_opt_merge_nested_early_exits(b.shader));
   nir_validate_shader(b.shader, "empty_arm after");
   ralloc_free(b.shader);
}

int
main(void)
{
   test_merge_nested_bounds_and_hoist();
   test_hoist_loop_invariant_guard_bits();
   test_fuse_selects_prefers_comparison();
   test_negative_impure_prelude();
   test_negative_mismatched_exit_arms();
   test_negative_int32_max_no_wrap();
   test_multi_impl_metadata_hygiene();
   test_empty_arm_no_crash();

   if (failures) {
      fprintf(stderr, "FAILED: %u checks\n", failures);
      return 1;
   }
   printf("PASS: merge-nested-early-exits\n");
   return 0;
}
