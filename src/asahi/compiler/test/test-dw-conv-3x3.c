/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "agx_compile.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"

static unsigned failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,      \
                 #cond);                                                       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static void
add_latch_phi_src(nir_phi_instr *phi, nir_block *latch, nir_def *val)
{
   nir_phi_src *ps = nir_phi_instr_add_src(phi, latch, val);
   list_addtail(&ps->src.use_link, &val->uses);
}

struct dw_test_opts {
   bool sliding_window_2px;
   bool const_bounds;
   bool store_in_loop;
   bool non_readonly_load;
   bool divergent_bound;
   bool vector_swizzled_bounds;
   bool add_preamble_func;
   bool early_inner_break;
   bool inner_header_live_out;
   enum {
      PREAMBLE_BOUNDS,
      DOMINATING_MEMORY_BOUNDS,
      CONDITIONAL_MEMORY_BOUNDS,
   } bounds;
};

static nir_shader *
build_dw_conv_shader(const struct dw_test_opts *opts)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "dw_conv_vec4");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 2;

   if (opts->add_preamble_func) {
      nir_function *pfn = nir_function_create(b.shader, "preamble");
      pfn->is_preamble = true;
      nir_function_impl *pimpl = nir_function_impl_create(pfn);
      nir_builder pb = nir_builder_at(nir_before_impl(pimpl));
      nir_store_preamble(&pb, nir_imm_int(&pb, 1), .base = 4);
   }

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gid_x = nir_channel(&b, gid, 0);
   nir_def *src_ptr = nir_load_preamble(&b, 1, 64, .base = 8);
   nir_def *w_ptr = nir_load_preamble(&b, 1, 64, .base = 12);
   nir_def *dst_ptr = nir_load_preamble(&b, 1, 64, .base = 16);
   nir_def *kvec = opts->vector_swizzled_bounds
      ? nir_load_preamble(&b, 4, 32, .base = 20)
      : NULL;
   nir_def *kh_bound = opts->const_bounds
      ? nir_imm_int(&b, 3)
      : (opts->divergent_bound
            ? gid_x
            : (opts->vector_swizzled_bounds
                  ? nir_channel(&b, kvec, 1)
                  : nir_load_preamble(&b, 1, 32, .base = 20)));
   nir_def *kw_bound = opts->const_bounds
      ? nir_imm_int(&b, 3)
      : (opts->vector_swizzled_bounds
            ? nir_channel(&b, kvec, 0)
            : nir_load_preamble(&b, 1, 32, .base = 22));
   if (opts->bounds != PREAMBLE_BOUNDS) {
      nir_def *params = nir_load_preamble(&b, 1, 64, .base = 24);
      kh_bound = nir_load_global_constant(&b, 1, 32, params, .align_mul = 4);
      kw_bound = nir_load_constant_agx(&b, 1, 32, params, nir_imm_int(&b, 1),
                                       .format = PIPE_FORMAT_R32_UINT);
   }
   nir_def *step_x = nir_load_preamble(&b, 1, 32, .base = 28);
   nir_def *step_y = nir_load_preamble(&b, 1, 32, .base = 30);
   nir_def *w_step_y = nir_load_preamble(&b, 1, 32, .base = 32);
   nir_def *kw_gt_0 = opts->sliding_window_2px
      ? nir_ilt(&b, nir_imm_int(&b, 0), kw_bound)
      : NULL;

   unsigned ld_access = opts->non_readonly_load ? 0 : ACCESS_NON_WRITEABLE;
   nir_def *zero_i = nir_imm_int(&b, 0);
   nir_def *zero_f = nir_imm_float(&b, 0.0f);

   unsigned n_acc = opts->sliding_window_2px ? 8 : 4;
   nir_loop *outer = nir_push_loop(&b);
   nir_block *outer_pre = nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));
   outer->control = nir_loop_control_dont_unroll;

   nir_phi_instr *kh_phi = nir_phi_instr_create(b.shader);
   nir_phi_instr *in_y_phi = nir_phi_instr_create(b.shader);
   nir_phi_instr *w_y_phi = nir_phi_instr_create(b.shader);
   nir_phi_instr *acc_y[8];
   nir_def_init(&kh_phi->instr, &kh_phi->def, 1, 32);
   nir_def_init(&in_y_phi->instr, &in_y_phi->def, 1, 32);
   nir_def_init(&w_y_phi->instr, &w_y_phi->def, 1, 32);
   nir_phi_instr_add_src(kh_phi, outer_pre, zero_i);
   nir_phi_instr_add_src(in_y_phi, outer_pre, gid_x);
   nir_phi_instr_add_src(w_y_phi, outer_pre, gid_x);
   nir_builder_instr_insert(&b, &kh_phi->instr);
   nir_builder_instr_insert(&b, &in_y_phi->instr);
   nir_builder_instr_insert(&b, &w_y_phi->instr);

   for (unsigned i = 0; i < n_acc; ++i) {
      acc_y[i] = nir_phi_instr_create(b.shader);
      nir_def_init(&acc_y[i]->instr, &acc_y[i]->def, 1, 32);
      nir_phi_instr_add_src(acc_y[i], outer_pre, zero_f);
      nir_builder_instr_insert(&b, &acc_y[i]->instr);
   }

   nir_if *outer_exit = nir_push_if(&b, nir_ige(&b, &kh_phi->def, kh_bound));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, outer_exit);
   if (opts->bounds == CONDITIONAL_MEMORY_BOUNDS) {
      nir_instr_move(b.cursor, nir_def_instr(kw_bound));
      b.cursor = nir_after_instr(nir_def_instr(kw_bound));
   }

   nir_if *guard_if = NULL;
   nir_def *acc_after_inner[8];

   if (opts->sliding_window_2px) {
      /* Exact 2-pixel sliding-window depthwise conv shape with guard_if and
       * bottom-tested inner loop marked dont_unroll. */
      guard_if = nir_push_if(&b, kw_gt_0);
      nir_def *prev_in = nir_load_agx(&b, 4, 32, src_ptr, &in_y_phi->def,
                                      .access = ld_access, .base = 2,
                                      .format = PIPE_FORMAT_R32_UINT,
                                      .sign_extend = false);
      nir_def *prev_ch[4];
      for (unsigned c = 0; c < 4; ++c)
         prev_ch[c] = nir_channel(&b, prev_in, c);
      nir_loop *inner = nir_push_loop(&b);
      nir_block *in_pre = nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
      inner->control = nir_loop_control_dont_unroll;

      nir_phi_instr *kw_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *in_x_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *w_x_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *p0_phi[4];
      nir_phi_instr *acc_x[8];

      nir_def_init(&kw_phi->instr, &kw_phi->def, 1, 32);
      nir_def_init(&in_x_phi->instr, &in_x_phi->def, 1, 32);
      nir_def_init(&w_x_phi->instr, &w_x_phi->def, 1, 32);
      nir_phi_instr_add_src(kw_phi, in_pre, zero_i);
      nir_phi_instr_add_src(in_x_phi, in_pre, &in_y_phi->def);
      nir_phi_instr_add_src(w_x_phi, in_pre, &w_y_phi->def);
      nir_builder_instr_insert(&b, &kw_phi->instr);
      nir_builder_instr_insert(&b, &in_x_phi->instr);
      nir_builder_instr_insert(&b, &w_x_phi->instr);

      for (unsigned c = 0; c < 4; ++c) {
         p0_phi[c] = nir_phi_instr_create(b.shader);
         nir_def_init(&p0_phi[c]->instr, &p0_phi[c]->def, 1, 32);
         nir_phi_instr_add_src(p0_phi[c], in_pre, prev_ch[c]);
         nir_builder_instr_insert(&b, &p0_phi[c]->instr);
      }
      for (unsigned i = 0; i < 8; ++i) {
         acc_x[i] = nir_phi_instr_create(b.shader);
         nir_def_init(&acc_x[i]->instr, &acc_x[i]->def, 1, 32);
         nir_phi_instr_add_src(acc_x[i], in_pre, &acc_y[i]->def);
         nir_builder_instr_insert(&b, &acc_x[i]->instr);
      }

      nir_def *wv = nir_load_agx(&b, 4, 32, w_ptr, &w_x_phi->def,
                                 .access = ld_access, .base = 2,
                                 .format = PIPE_FORMAT_R32_UINT,
                                 .sign_extend = false);
      nir_def *acc_next[8];
      for (unsigned c = 0; c < 4; ++c)
         acc_next[c] = nir_ffma(&b, &p0_phi[c]->def, nir_channel(&b, wv, c),
                                &acc_x[c]->def);

      nir_def *in_x_next = nir_iadd(&b, &in_x_phi->def, step_x);
      nir_def *p1 = nir_load_agx(&b, 4, 32, src_ptr, in_x_next,
                                 .access = ld_access, .base = 2,
                                 .format = PIPE_FORMAT_R32_UINT,
                                 .sign_extend = false);
      for (unsigned c = 0; c < 4; ++c)
         acc_next[4 + c] = nir_ffma(&b, nir_channel(&b, p1, c),
                                    nir_channel(&b, wv, c), &acc_x[4 + c]->def);

      if (opts->store_in_loop)
         nir_store_agx(&b, p1, dst_ptr, &in_x_phi->def,
                       .access = ACCESS_NON_READABLE, .base = 2,
                       .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      nir_def *kw_next = nir_iadd_imm(&b, &kw_phi->def, 1);
      nir_if *in_exit = nir_push_if(&b, nir_ige(&b, kw_next, kw_bound));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, in_exit);

      nir_def *w_x_next = nir_iadd(&b, &w_x_phi->def, step_x);
      nir_def *p1_c[4];
      for (unsigned c = 0; c < 4; ++c)
         p1_c[c] = nir_mov(&b, nir_channel(&b, p1, c));
      nir_block *in_latch = nir_loop_last_block(inner);

      add_latch_phi_src(kw_phi, in_latch, kw_next);
      add_latch_phi_src(in_x_phi, in_latch, in_x_next);
      add_latch_phi_src(w_x_phi, in_latch, w_x_next);
      for (unsigned c = 0; c < 4; ++c)
         add_latch_phi_src(p0_phi[c], in_latch, p1_c[c]);
      for (unsigned i = 0; i < 8; ++i)
         add_latch_phi_src(acc_x[i], in_latch, acc_next[i]);
      nir_pop_loop(&b, inner);
      nir_pop_if(&b, guard_if);

      nir_block *then_end = nir_if_last_then_block(guard_if);
      nir_block *else_end = nir_if_last_else_block(guard_if);
      for (unsigned i = 0; i < 8; ++i) {
         nir_phi_instr *mp = nir_phi_instr_create(b.shader);
         nir_def_init(&mp->instr, &mp->def, 1, 32);
         nir_phi_instr_add_src(mp, then_end, acc_next[i]);
         nir_phi_instr_add_src(mp, else_end, &acc_y[i]->def);
         nir_builder_instr_insert(&b, &mp->instr);
         acc_after_inner[i] = &mp->def;
      }
   } else {
      /* Direct nested top-tested 2D depthwise conv (4 accumulators). */
      nir_loop *inner = nir_push_loop(&b);
      nir_block *in_pre = nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
      nir_phi_instr *kw_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *in_x_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *w_x_phi = nir_phi_instr_create(b.shader);
      nir_phi_instr *acc_x[4];

      nir_def_init(&kw_phi->instr, &kw_phi->def, 1, 32);
      nir_def_init(&in_x_phi->instr, &in_x_phi->def, 1, 32);
      nir_def_init(&w_x_phi->instr, &w_x_phi->def, 1, 32);
      nir_phi_instr_add_src(kw_phi, in_pre, zero_i);
      nir_phi_instr_add_src(in_x_phi, in_pre, &in_y_phi->def);
      nir_phi_instr_add_src(w_x_phi, in_pre, &w_y_phi->def);
      nir_builder_instr_insert(&b, &kw_phi->instr);
      nir_builder_instr_insert(&b, &in_x_phi->instr);
      nir_builder_instr_insert(&b, &w_x_phi->instr);
      for (unsigned c = 0; c < 4; ++c) {
         acc_x[c] = nir_phi_instr_create(b.shader);
         nir_def_init(&acc_x[c]->instr, &acc_x[c]->def, 1, 32);
         nir_phi_instr_add_src(acc_x[c], in_pre, &acc_y[c]->def);
         nir_builder_instr_insert(&b, &acc_x[c]->instr);
      }

      nir_def *row_step = opts->inner_header_live_out
         ? nir_iadd(&b, &kw_phi->def, step_y)
         : step_y;
      nir_if *in_exit = nir_push_if(&b, nir_ige(&b, &kw_phi->def, kw_bound));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, in_exit);

      nir_def *pv = nir_load_agx(&b, 4, 32, src_ptr, &in_x_phi->def,
                                 .access = ld_access, .base = 2,
                                 .format = PIPE_FORMAT_R32_UINT,
                                 .sign_extend = false);
      nir_def *wv = nir_load_agx(&b, 4, 32, w_ptr, &w_x_phi->def,
                                 .access = ld_access, .base = 2,
                                 .format = PIPE_FORMAT_R32_UINT,
                                 .sign_extend = false);
      nir_def *acc_next[4];
      for (unsigned c = 0; c < 4; ++c)
         acc_next[c] = nir_ffma(&b, nir_channel(&b, pv, c),
                                nir_channel(&b, wv, c), &acc_x[c]->def);

      if (opts->store_in_loop)
         nir_store_agx(&b, pv, dst_ptr, &in_x_phi->def,
                       .access = ACCESS_NON_READABLE, .base = 2,
                       .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      if (opts->early_inner_break) {
         nir_if *tap = nir_push_if(&b, nir_ilt(&b, &kw_phi->def, step_x));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, tap);
      }

      nir_def *kw_next = nir_iadd_imm(&b, &kw_phi->def, 1);
      nir_def *in_x_next = nir_iadd(&b, &in_x_phi->def, step_x);
      nir_def *w_x_next = nir_iadd(&b, &w_x_phi->def, step_x);
      nir_block *in_latch = nir_loop_last_block(inner);

      add_latch_phi_src(kw_phi, in_latch, kw_next);
      add_latch_phi_src(in_x_phi, in_latch, in_x_next);
      add_latch_phi_src(w_x_phi, in_latch, w_x_next);
      for (unsigned c = 0; c < 4; ++c)
         add_latch_phi_src(acc_x[c], in_latch, acc_next[c]);
      nir_pop_loop(&b, inner);
      step_y = row_step;

      for (unsigned c = 0; c < 4; ++c)
         acc_after_inner[c] = &acc_x[c]->def;
   }

   nir_def *kh_next = nir_iadd_imm(&b, &kh_phi->def, 1);
   nir_def *in_y_next = nir_iadd(&b, &in_y_phi->def, step_y);
   nir_def *w_y_next = nir_iadd(&b, &w_y_phi->def, w_step_y);
   nir_block *outer_latch = nir_loop_last_block(outer);

   add_latch_phi_src(kh_phi, outer_latch, kh_next);
   add_latch_phi_src(in_y_phi, outer_latch, in_y_next);
   add_latch_phi_src(w_y_phi, outer_latch, w_y_next);
   for (unsigned i = 0; i < n_acc; ++i)
      add_latch_phi_src(acc_y[i], outer_latch, acc_after_inner[i]);
   nir_pop_loop(&b, outer);

   nir_def *out0 = nir_vec4(&b, &acc_y[0]->def, &acc_y[1]->def,
                            &acc_y[2]->def, &acc_y[3]->def);
   nir_store_agx(&b, out0, dst_ptr, gid_x, .access = ACCESS_NON_READABLE,
                 .base = 2, .format = PIPE_FORMAT_R32_UINT,
                 .sign_extend = false);
   if (opts->sliding_window_2px) {
      nir_def *out1 = nir_vec4(&b, &acc_y[4]->def, &acc_y[5]->def,
                               &acc_y[6]->def, &acc_y[7]->def);
      nir_store_agx(&b, out1, dst_ptr, nir_iadd_imm(&b, gid_x, 1),
                    .access = ACCESS_NON_READABLE, .base = 2,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
   }

   nir_validate_shader(b.shader, "initial");
   return b.shader;
}

static void
inspect_specialized_if(nir_shader *s, unsigned *then_loops,
                       unsigned *then_fmas, unsigned *else_loops)
{
   *then_loops = 0;
   *then_fmas = 0;
   *else_loops = 0;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node);
      foreach_list_typed(nir_cf_node, tn, node, &nif->then_list) {
         if (tn->type == nir_cf_node_loop)
            (*then_loops)++;
      }
      nir_foreach_block_in_cf_node(blk, &nif->cf_node) {
         if (blk == nir_if_first_then_block(nif)) {
            nir_foreach_instr(instr, blk) {
               if (instr->type == nir_instr_type_alu &&
                   nir_instr_as_alu(instr)->op == nir_op_ffma)
                  (*then_fmas)++;
            }
         }
      }
      foreach_list_typed(nir_cf_node, en, node, &nif->else_list) {
         if (en->type == nir_cf_node_loop)
            (*else_loops)++;
      }
   }
}

static void
check_compiles_both_agx3(nir_shader *proto)
{
   const struct {
      enum agx_chip chip;
      enum agx_isa_encoding encoding;
   } targets[2] = {
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C},
   };

   for (unsigned t = 0; t < 2; ++t) {
      nir_shader *clone = nir_shader_clone(NULL, proto);
      clone->info.io_lowered = true;
      struct agx_shader_key key = {
         .dev.chip = targets[t].chip,
         .dev.encoding = targets[t].encoding,
         .dev.soft_fault = true,
         .reserved_preamble = 40,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      bool ok = agx_compile_shader_nir(clone, &key, &part);
      CHECK(ok);
      CHECK(part.info.binary_size > 0);
      CHECK(part.info.scratch_size == 0);
      CHECK(part.info.nr_gprs <= 104);
      free(part.binary);
      ralloc_free(clone);
   }
}

int
main(void)
{
   /* 1. Positive: 2-pixel sliding-window depthwise conv (dont_unroll). */
   {
      struct dw_test_opts opts = {.sliding_window_2px = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      nir_shader *raw = nir_shader_clone(NULL, s);
      CHECK(agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after sliding 3x3");
      unsigned then_loops = 0, then_fmas = 0, else_loops = 0;
      inspect_specialized_if(s, &then_loops, &then_fmas, &else_loops);
      CHECK(then_loops == 0);
      CHECK(then_fmas == 72);
      CHECK(else_loops == 1);
      check_compiles_both_agx3(raw);
      ralloc_free(raw);
      ralloc_free(s);
   }

   /* 2. Positive: direct top-tested nested 2D depthwise conv (4 accs). */
   {
      struct dw_test_opts opts = {0};
      nir_shader *s = build_dw_conv_shader(&opts);
      nir_shader *raw = nir_shader_clone(NULL, s);
      CHECK(agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after direct 3x3");
      unsigned then_loops = 0, then_fmas = 0, else_loops = 0;
      inspect_specialized_if(s, &then_loops, &then_fmas, &else_loops);
      CHECK(then_loops == 0);
      CHECK(then_fmas == 36);
      CHECK(else_loops == 1);
      check_compiles_both_agx3(raw);
      ralloc_free(raw);
      ralloc_free(s);
   }

   /* 3. Negative: store inside inner loop must be rejected. */
   {
      struct dw_test_opts opts = {.sliding_window_2px = true,
                                  .store_in_loop = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      ralloc_free(s);
   }

   /* 4. Negative: compile-time constant bounds left to generic unroller. */
   {
      struct dw_test_opts opts = {.const_bounds = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      ralloc_free(s);
   }

   /* 5. Negative: non-readonly load rejected. */
   {
      struct dw_test_opts opts = {.non_readonly_load = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      ralloc_free(s);
   }

   /* 6. Negative: divergent loop bound rejected. */
   {
      struct dw_test_opts opts = {.divergent_bound = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      ralloc_free(s);
   }

   /* 7. Positive: vector-swizzled con 32x4 bounds (.y/.x) + preamble func. */
   {
      struct dw_test_opts opts = {
         .sliding_window_2px = true,
         .vector_swizzled_bounds = true,
         .add_preamble_func = true,
      };
      nir_shader *s = build_dw_conv_shader(&opts);
      nir_shader *raw = nir_shader_clone(NULL, s);
      CHECK(agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after vector-swizzled sliding 3x3");
      unsigned then_loops = 0, then_fmas = 0, else_loops = 0;
      inspect_specialized_if(s, &then_loops, &then_fmas, &else_loops);
      CHECK(then_loops == 0);
      CHECK(then_fmas == 72);
      CHECK(else_loops == 1);
      check_compiles_both_agx3(raw);
      ralloc_free(raw);
      ralloc_free(s);
   }

   /* 8. Positive: bottom-tested outer + inner with divergent row/tap guards. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "dw_conv_padded");
      b.shader->info.workgroup_size[0] = 8;
      b.shader->info.workgroup_size[1] = 8;
      b.shader->info.workgroup_size[2] = 2;

      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_def *gx = nir_channel(&b, gid, 0);
      nir_def *gy = nir_channel(&b, gid, 1);
      nir_def *src_ptr = nir_load_preamble(&b, 1, 64, .base = 8);
      nir_def *w_ptr = nir_load_preamble(&b, 1, 64, .base = 12);
      nir_def *dst_ptr = nir_load_preamble(&b, 1, 64, .base = 16);
      nir_def *kh_bound = nir_load_preamble(&b, 1, 32, .base = 20);
      nir_def *kw_bound = nir_load_preamble(&b, 1, 32, .base = 22);
      nir_def *img_h = nir_load_preamble(&b, 1, 32, .base = 24);
      nir_def *img_w = nir_load_preamble(&b, 1, 32, .base = 26);
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *zero_f = nir_imm_float(&b, 0.0f);

      nir_loop *outer = nir_push_loop(&b);
      nir_block *opre = nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));
      nir_phi_instr *kh = nir_phi_instr_create(b.shader);
      nir_phi_instr *acc_o = nir_phi_instr_create(b.shader);
      nir_def_init(&kh->instr, &kh->def, 1, 32);
      nir_def_init(&acc_o->instr, &acc_o->def, 1, 32);
      nir_phi_instr_add_src(kh, opre, zero);
      nir_phi_instr_add_src(acc_o, opre, zero_f);
      nir_builder_instr_insert(&b, &kh->instr);
      nir_builder_instr_insert(&b, &acc_o->instr);

      nir_def *iy = nir_iadd(&b, gy, &kh->def);
      nir_def *y_ok = nir_ilt(&b, iy, img_h);
      nir_if *gif = nir_push_if(&b, y_ok);
      nir_loop *inner = nir_push_loop(&b);
      nir_block *ipre = nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
      nir_phi_instr *kw = nir_phi_instr_create(b.shader);
      nir_phi_instr *acc_i = nir_phi_instr_create(b.shader);
      nir_def_init(&kw->instr, &kw->def, 1, 32);
      nir_def_init(&acc_i->instr, &acc_i->def, 1, 32);
      nir_phi_instr_add_src(kw, ipre, zero);
      nir_phi_instr_add_src(acc_i, ipre, &acc_o->def);
      nir_builder_instr_insert(&b, &kw->instr);
      nir_builder_instr_insert(&b, &acc_i->instr);

      nir_def *ix = nir_iadd(&b, gx, &kw->def);
      nir_def *x_ok = nir_ilt(&b, ix, img_w);
      nir_if *tif = nir_push_if(&b, x_ok);
      nir_def *iv = nir_load_agx(&b, 1, 32, src_ptr,
                                 nir_iadd(&b, nir_imul(&b, iy, img_w), ix),
                                 .access = ACCESS_NON_WRITEABLE, .base = 0,
                                 .format = PIPE_FORMAT_R32_UINT);
      nir_def *wv = nir_load_agx(&b, 1, 32, w_ptr,
                                 nir_iadd(&b, nir_imul(&b, &kh->def, kw_bound), &kw->def),
                                 .access = ACCESS_NON_WRITEABLE, .base = 0,
                                 .format = PIPE_FORMAT_R32_UINT);
      nir_def *fma = nir_ffma(&b, iv, wv, &acc_i->def);
      nir_pop_if(&b, tif);

      nir_phi_instr *tm = nir_phi_instr_create(b.shader);
      nir_def_init(&tm->instr, &tm->def, 1, 32);
      nir_phi_instr_add_src(tm, nir_if_last_then_block(tif), fma);
      nir_phi_instr_add_src(tm, nir_if_last_else_block(tif), &acc_i->def);
      nir_builder_instr_insert(&b, &tm->instr);

      nir_def *kw_n = nir_iadd_imm(&b, &kw->def, 1);
      nir_if *iex = nir_push_if(&b, nir_ige(&b, kw_n, kw_bound));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, iex);
      nir_block *il = nir_loop_last_block(inner);
      add_latch_phi_src(kw, il, kw_n);
      add_latch_phi_src(acc_i, il, &tm->def);
      nir_pop_loop(&b, inner);
      nir_pop_if(&b, gif);

      nir_phi_instr *gm = nir_phi_instr_create(b.shader);
      nir_def_init(&gm->instr, &gm->def, 1, 32);
      nir_phi_instr_add_src(gm, nir_if_last_then_block(gif), &tm->def);
      nir_phi_instr_add_src(gm, nir_if_last_else_block(gif), &acc_o->def);
      nir_builder_instr_insert(&b, &gm->instr);

      nir_def *kh_n = nir_iadd_imm(&b, &kh->def, 1);
      nir_if *oex = nir_push_if(&b, nir_ige(&b, kh_n, kh_bound));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, oex);
      nir_block *ol = nir_loop_last_block(outer);
      add_latch_phi_src(kh, ol, kh_n);
      add_latch_phi_src(acc_o, ol, &gm->def);
      nir_pop_loop(&b, outer);

      nir_store_agx(&b, &gm->def, dst_ptr, gx, .access = ACCESS_NON_READABLE,
                    .base = 0, .format = PIPE_FORMAT_R32_UINT);
      nir_validate_shader(b.shader, "initial padded");
      nir_shader *raw = nir_shader_clone(NULL, b.shader);
      CHECK(agx_nir_opt_dw_conv_3x3(b.shader));
      nir_validate_shader(b.shader, "after padded 3x3");
      unsigned then_loops = 0, then_fmas = 0, else_loops = 0;
      inspect_specialized_if(b.shader, &then_loops, &then_fmas, &else_loops);
      CHECK(then_loops == 0);
      CHECK(else_loops == 1);
      check_compiles_both_agx3(raw);
      ralloc_free(raw);
      ralloc_free(b.shader);
   }

   /* An early break in the tap branch cannot be discarded. */
   {
      struct dw_test_opts opts = {.early_inner_break = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after rejecting early inner break");
      ralloc_free(s);
   }

   /* KW may only be loaded once the outer loop enters its body. */
   {
      struct dw_test_opts opts = {.bounds = CONDITIONAL_MEMORY_BOUNDS};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after rejecting conditional memory bound");
      ralloc_free(s);

      opts.bounds = DOMINATING_MEMORY_BOUNDS;
      s = build_dw_conv_shader(&opts);
      CHECK(agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after specializing dominating memory bounds");
      ralloc_free(s);
   }

   /* The next row uses a value evaluated on the final inner header visit. */
   {
      struct dw_test_opts opts = {.inner_header_live_out = true};
      nir_shader *s = build_dw_conv_shader(&opts);
      CHECK(!agx_nir_opt_dw_conv_3x3(s));
      nir_validate_shader(s, "after rejecting top-tested inner live-out");
      ralloc_free(s);
   }

   return failures ? 1 : 0;
}
