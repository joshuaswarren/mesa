/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned
count_loops_in_list(struct exec_list *cf_list)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_loop)
         n++;
      else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         n += count_loops_in_list(&nif->then_list);
         n += count_loops_in_list(&nif->else_list);
      }
   }
   return n;
}

static unsigned
count_op_in_cf(nir_cf_node *root, nir_op op)
{
   unsigned n = 0;
   nir_foreach_block_in_cf_node(block, root) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == op)
            n++;
      }
   }
   return n;
}

static nir_if *
find_first_top_if(nir_function_impl *impl)
{
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if)
         return nir_cf_node_as_if(node);
   }
   return NULL;
}

static nir_loop *
find_first_loop_in_list(struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(node);
   }
   return NULL;
}

/* Positive test: fast copy drops ix compares, border copy keeps them. */
static void
test_basic_unswitch(void)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "unswitch_basic");
   nir_def *ox = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *kw_bound = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *in_w = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));
   nir_def *pad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8));
   nir_def *base_ptr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 16));

   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
   nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);
   nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);

   nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_push_if(&b, nir_ige(&b, k, kw_bound));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *ix = nir_isub(&b, nir_iadd(&b, ox, k), pad);
      nir_def *in_bounds = nir_iand(&b, nir_ige(&b, ix, nir_imm_int(&b, 0)),
                                    nir_ilt(&b, ix, in_w));
      nir_push_if(&b, in_bounds);
      {
         nir_def *addr = nir_iadd(&b, base_ptr, nir_u2u64(&b, nir_imul_imm(&b, ix, 4)));
         nir_def *val = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
         nir_store_var(&b, acc_var, nir_fadd(&b, nir_load_var(&b, acc_var), val), 1);
      }
      nir_pop_if(&b, NULL);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_store_global(&b, nir_load_var(&b, acc_var), base_ptr);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "before basic unswitch");

   bool progress = agx_nir_opt_unswitch_affine_bounds(b.shader);
   nir_opt_dead_cf(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after basic unswitch");

   if (!progress || count_loops_in_list(&b.impl->body) != 2) {
      fprintf(stderr, "FAIL basic: expected 2 loops (progress=%d loops=%u)\n",
              progress, count_loops_in_list(&b.impl->body));
      exit(1);
   }

   nir_if *unswitch = find_first_top_if(b.impl);
   nir_loop *fast_loop = unswitch ? find_first_loop_in_list(&unswitch->then_list) : NULL;
   nir_loop *border_loop = unswitch ? find_first_loop_in_list(&unswitch->else_list) : NULL;
   if (!fast_loop || !border_loop) {
      fprintf(stderr, "FAIL basic: missing fast/border loop\n");
      exit(1);
   }
   if (count_op_in_cf(&fast_loop->cf_node, nir_op_ilt) != 0 ||
       count_op_in_cf(&border_loop->cf_node, nir_op_ilt) == 0) {
      fprintf(stderr, "FAIL basic: fast ilt=%u (want 0), border ilt=%u (want >0)\n",
              count_op_in_cf(&fast_loop->cf_node, nir_op_ilt),
              count_op_in_cf(&border_loop->cf_node, nir_op_ilt));
      exit(1);
   }
   ralloc_free(b.shader);
}

/* Folded check that is also live-out must produce valid IR and fold the phi src. */
static void
test_live_out_folded_check(void)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "unswitch_live_out_check");
   nir_def *ox = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *w = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));
   nir_def *base_ptr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 8));

   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_variable *ok_var = nir_local_variable_create(b.impl, glsl_bool_type(), "ok");
   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
   nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);
   nir_store_var(&b, ok_var, nir_imm_false(&b), 1);
   nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);

   nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_def *ix = nir_iadd(&b, ox, k);
      nir_def *ok = nir_iand(&b, nir_ige(&b, ix, nir_imm_int(&b, 0)),
                             nir_ilt(&b, ix, w));
      nir_store_var(&b, ok_var, ok, 1);
      nir_push_if(&b, ok);
      {
         nir_def *addr = nir_iadd(&b, base_ptr, nir_u2u64(&b, nir_imul_imm(&b, ix, 4)));
         nir_def *val = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
         nir_store_var(&b, acc_var, nir_fadd(&b, nir_load_var(&b, acc_var), val), 1);
      }
      nir_pop_if(&b, NULL);
      nir_def *k_next = nir_iadd_imm(&b, k, 1);
      nir_store_var(&b, k_var, k_next, 1);
      nir_push_if(&b, nir_ige(&b, k_next, n));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
   }
   nir_pop_loop(&b, NULL);

   nir_def *out = nir_bcsel(&b, nir_load_var(&b, ok_var),
                            nir_load_var(&b, acc_var), nir_imm_float(&b, 0.0f));
   nir_store_global(&b, out, base_ptr);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "before live_out check");

   bool progress = agx_nir_opt_unswitch_affine_bounds(b.shader);
   nir_validate_shader(b.shader, "after live_out check");
   if (!progress || count_loops_in_list(&b.impl->body) != 2) {
      fprintf(stderr, "FAIL live_out_check: expected valid unswitch (progress=%d loops=%u)\n",
              progress, count_loops_in_list(&b.impl->body));
      exit(1);
   }
   ralloc_free(b.shader);
}

/* Two sibling loops in the same CF list must both be unswitched. */
static void
test_sibling_loops(void)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "unswitch_siblings");
   nir_def *ox = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *w = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));
   nir_def *base_ptr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 8));

   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);

   for (int s = 0; s < 2; ++s) {
      nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
      nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         nir_def *k = nir_load_var(&b, k_var);
         nir_push_if(&b, nir_ige(&b, k, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
         nir_def *ix = nir_iadd(&b, ox, k);
         nir_push_if(&b, nir_iand(&b, nir_ige(&b, ix, nir_imm_int(&b, 0)), nir_ilt(&b, ix, w)));
         {
            nir_def *addr = nir_iadd(&b, base_ptr, nir_u2u64(&b, nir_imul_imm(&b, ix, 4)));
            nir_store_var(&b, acc_var, nir_fadd(&b, nir_load_var(&b, acc_var),
                          nir_load_global(&b, 1, 32, addr, .align_mul = 4)), 1);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 1), 1);
      }
      nir_pop_loop(&b, NULL);
   }

   nir_store_global(&b, nir_load_var(&b, acc_var), base_ptr);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);

   bool progress = agx_nir_opt_unswitch_affine_bounds(b.shader);
   nir_validate_shader(b.shader, "after sibling loops");
   if (!progress || count_loops_in_list(&b.impl->body) != 4) {
      fprintf(stderr, "FAIL siblings: expected 4 loops (progress=%d loops=%u)\n",
              progress, count_loops_in_list(&b.impl->body));
      exit(1);
   }
   ralloc_free(b.shader);
}

/* Negative cases that must NOT transform: init != 0, step != 1, top-exit header check. */
static void
test_negative_shape(const char *name, int init_val, int step_val, bool check_in_header)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts, "%s", name);
   nir_def *ox = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *w = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));
   nir_def *base_ptr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 8));

   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
   nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);
   nir_store_var(&b, k_var, nir_imm_int(&b, init_val), 1);

   nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_def *hdr_ok = NULL;
      nir_def *hdr_ix = NULL;
      if (check_in_header) {
         hdr_ix = nir_iadd(&b, ox, k);
         hdr_ok = nir_iand(&b, nir_ige(&b, hdr_ix, nir_imm_int(&b, 0)),
                           nir_ilt(&b, hdr_ix, w));
         nir_store_global(&b, nir_bcsel(&b, hdr_ok, nir_imm_float(&b, 1.0f),
                                        nir_imm_float(&b, 0.0f)), base_ptr);
      }

      nir_push_if(&b, nir_ige(&b, k, n));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *ix = check_in_header ? hdr_ix : nir_iadd(&b, ox, k);
      nir_def *ok = check_in_header
         ? hdr_ok
         : nir_iand(&b, nir_ige(&b, ix, nir_imm_int(&b, 0)), nir_ilt(&b, ix, w));
      nir_push_if(&b, ok);
      {
         nir_def *addr = nir_iadd(&b, base_ptr, nir_u2u64(&b, nir_imul_imm(&b, ix, 4)));
         nir_store_var(&b, acc_var, nir_fadd(&b, nir_load_var(&b, acc_var),
                       nir_load_global(&b, 1, 32, addr, .align_mul = 4)), 1);
      }
      nir_pop_if(&b, NULL);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, step_val), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_store_global(&b, nir_load_var(&b, acc_var), base_ptr);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);

   bool progress = agx_nir_opt_unswitch_affine_bounds(b.shader);
   nir_validate_shader(b.shader, name);
   if (progress || count_loops_in_list(&b.impl->body) != 1) {
      fprintf(stderr, "FAIL %s: must not transform (progress=%d loops=%u)\n",
              name, progress, count_loops_in_list(&b.impl->body));
      exit(1);
   }
   ralloc_free(b.shader);
}

/* Verify emitted guard rejects (n-1)*step overflow and unsigned limit >= 2^31. */
static void
test_guard_overflow_and_unsigned_limit(void)
{
   const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "unswitch_guard_edge");
   nir_def *ox = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_def *lim = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));
   nir_def *base_ptr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 8));

   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
   nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);
   nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);

   nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_push_if(&b, nir_ige(&b, k, n));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_def *ix = nir_iadd(&b, ox, nir_ishl_imm(&b, k, 2));
      nir_push_if(&b, nir_ult(&b, ix, lim));
      {
         nir_def *addr = nir_iadd(&b, base_ptr, nir_u2u64(&b, nir_imul_imm(&b, ix, 4)));
         nir_store_var(&b, acc_var, nir_fadd(&b, nir_load_var(&b, acc_var),
                       nir_load_global(&b, 1, 32, addr, .align_mul = 4)), 1);
      }
      nir_pop_if(&b, NULL);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_store_global(&b, nir_load_var(&b, acc_var), base_ptr);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);

   bool progress = agx_nir_opt_unswitch_affine_bounds(b.shader);
   nir_validate_shader(b.shader, "after guard edge unswitch");
   if (!progress) {
      fprintf(stderr, "FAIL guard_edge: expected transform\n");
      exit(1);
   }

   const struct { uint32_t ox_v, n_v, lim_v; bool expect; const char *desc; } cases[] = {
      { 0u, 4u, 100u, true, "safe_interior" },
      { 0u, 0x20000001u, 0x7fffffffu, false, "span_overflow" },
      { 0u, 4u, 0x80000000u, false, "unsigned_lim_ge_2p31" },
   };
   for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
      nir_shader *cl = nir_shader_clone(NULL, b.shader);
      nir_function_impl *impl = nir_shader_get_entrypoint(cl);
      nir_builder cb = nir_builder_create(impl);
      nir_foreach_block(blk, impl) {
         nir_foreach_instr_safe(ins, blk) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
            if (intr->intrinsic == nir_intrinsic_load_global_invocation_id) {
               cb.cursor = nir_before_instr(ins);
               nir_def_replace(&intr->def, nir_imm_ivec3(&cb, (int)cases[c].ox_v, 0, 0));
            } else if (intr->intrinsic == nir_intrinsic_load_push_constant &&
                       intr->def.bit_size == 32) {
               uint32_t off = nir_src_as_uint(intr->src[0]);
               cb.cursor = nir_before_instr(ins);
               uint32_t val = off == 0 ? cases[c].n_v : cases[c].lim_v;
               nir_def_replace(&intr->def, nir_imm_int(&cb, (int)val));
            }
         }
      }
      bool p2;
      do {
         p2 = false;
         p2 |= nir_opt_constant_folding(cl);
         p2 |= nir_opt_copy_prop(cl);
         p2 |= nir_opt_dce(cl);
      } while (p2);
      nir_if *unsw = find_first_top_if(impl);
      if (!unsw || !nir_src_is_const(unsw->condition) ||
          nir_src_as_bool(unsw->condition) != cases[c].expect) {
         fprintf(stderr, "FAIL guard_edge %s: cond did not fold to %d\n",
                 cases[c].desc, cases[c].expect);
         exit(1);
      }
      ralloc_free(cl);
   }
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_basic_unswitch();
   test_live_out_folded_check();
   test_sibling_loops();
   test_negative_shape("neg_init_nonzero", 1, 1, false);
   test_negative_shape("neg_step_ne1", 0, 2, false);
   test_negative_shape("neg_header_check", 0, 1, true);
   test_guard_overflow_and_unsigned_limit();
   glsl_type_singleton_decref();
   return 0;
}
