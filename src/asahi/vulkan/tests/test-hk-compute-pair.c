/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "hk_nir_eval_test.h"
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_nir_fast_udiv.h"
#include "hk_nir_wgid.h"
#include "hk_nir_loop_bound.h"
#include "hk_nir_tensor_ops.h"
#include "util/blob.h"

static nir_shader *
new_compute_shader(const char *name, uint16_t lx, uint16_t ly, uint16_t lz)
{
   nir_shader *s = nir_shader_create(NULL, MESA_SHADER_COMPUTE,
                                     &agx_nir_options);
   s->info.name = ralloc_strdup(s, name);
   s->info.workgroup_size[0] = lx;
   s->info.workgroup_size[1] = ly;
   s->info.workgroup_size[2] = lz;
   nir_function *fn = nir_function_create(s, "main");
   fn->is_entrypoint = true;
   nir_function_impl *impl = nir_function_impl_create(fn);
   (void)impl;
   return s;
}

static const struct glsl_type *
ssbo_f32_array_type(void)
{
   const struct glsl_type *arr =
      glsl_array_type(glsl_float_type(), 0, 4);
   struct glsl_struct_field field = {
      .type = arr,
      .name = "data",
      .offset = 0,
   };
   return glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430,
                              false, "Buf");
}

static nir_def *
ssbo_elem_deref(nir_builder *b, unsigned set, unsigned binding,
                nir_def *idx_elems)
{
   const struct glsl_type *iface = ssbo_f32_array_type();
   nir_def *res = nir_vulkan_resource_index(
      b, 4, 32, nir_imm_int(b, 0),
      .desc_set = set, .binding = binding,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, 4, 32, res, .desc_type = nir_descriptor_type_storage_buffer);
   nir_deref_instr *blk =
      nir_build_deref_cast(b, desc, nir_var_mem_ssbo, iface, 0);
   nir_deref_instr *arr = nir_build_deref_struct(b, blk, 0);
   return &nir_build_deref_array(b, arr, idx_elems)->def;
}

static nir_def *
load_push_i32(nir_builder *b, uint32_t offset_B)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0),
                                 .base = offset_B, .range = 4);
}

/* Build a general FP32 pointwise clamp compute shader. */
static nir_shader *
build_clamp_nir(unsigned set, unsigned in_bind, unsigned out_bind,
                uint32_t count_off, const struct hk_nir_clamp_epilogue *ep,
                uint16_t lx)
{
   nir_shader *s = new_compute_shader("clamp", lx, 1, 1);
   s->info.num_ssbos = 2;
   s->info.float_controls_execution_mode = ep->float_controls_execution_mode;
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *count = load_push_i32(&b, count_off);
   nir_push_if(&b, nir_ilt(&b, gx, count));
   {
      nir_def *in_d = ssbo_elem_deref(&b, set, in_bind, gx);
      nir_def *v = nir_load_deref(&b, nir_def_as_deref(in_d));
      if (ep->has_lower && ep->has_upper) {
         if (ep->upper_first) {
            nir_def *t = nir_fmin(&b, v, nir_imm_int(&b, ep->upper_bits));
            nir_def_as_alu(t)->fp_math_ctrl = ep->min_fp_math_ctrl;
            v = nir_fmax(&b, t, nir_imm_int(&b, ep->lower_bits));
            nir_def_as_alu(v)->fp_math_ctrl = ep->max_fp_math_ctrl;
         } else {
            nir_def *t = nir_fmax(&b, v, nir_imm_int(&b, ep->lower_bits));
            nir_def_as_alu(t)->fp_math_ctrl = ep->max_fp_math_ctrl;
            v = nir_fmin(&b, t, nir_imm_int(&b, ep->upper_bits));
            nir_def_as_alu(v)->fp_math_ctrl = ep->min_fp_math_ctrl;
         }
      } else if (ep->has_lower) {
         v = nir_fmax(&b, v, nir_imm_int(&b, ep->lower_bits));
         nir_def_as_alu(v)->fp_math_ctrl = ep->max_fp_math_ctrl;
      } else if (ep->has_upper) {
         v = nir_fmin(&b, v, nir_imm_int(&b, ep->upper_bits));
         nir_def_as_alu(v)->fp_math_ctrl = ep->min_fp_math_ctrl;
      }
      nir_def *out_d = ssbo_elem_deref(&b, set, out_bind, gx);
      nir_store_deref(&b, nir_def_as_deref(out_d), v, 0x1);
   }
   nir_pop_if(&b, NULL);
   nir_validate_shader(s, "build_clamp_nir");
   return s;
}

/* Permuted push layout for depthwise conv testing (verifies zero hardcoded ABI). */
struct dw_push_layout {
   int32_t channels;
   int32_t out_w;
   int32_t out_h;
   int32_t in_h;
   int32_t in_w;
   int32_t ker_h;
   int32_t ker_w;
   int32_t pad_y;
   int32_t pad_x;
   int32_t str_y;
   int32_t str_x;
   int32_t dil_y;
   int32_t dil_x;
   int32_t has_bias;
};

static nir_shader *
build_depthwise_nir(bool with_bias_buf, bool with_bias_flag,
                    unsigned set, unsigned in_b, unsigned wt_b,
                    unsigned bs_b, unsigned out_b)
{
   nir_shader *s = new_compute_shader("depthwise", 16, 16, 1);
   s->info.num_ssbos = with_bias_buf ? 4 : 3;
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   nir_def *ch = load_push_i32(&b, offsetof(struct dw_push_layout, channels));
   nir_def *ow = load_push_i32(&b, offsetof(struct dw_push_layout, out_w));
   nir_def *oh = load_push_i32(&b, offsetof(struct dw_push_layout, out_h));
   nir_def *ih = load_push_i32(&b, offsetof(struct dw_push_layout, in_h));
   nir_def *iw = load_push_i32(&b, offsetof(struct dw_push_layout, in_w));
   nir_def *kh = load_push_i32(&b, offsetof(struct dw_push_layout, ker_h));
   nir_def *kw = load_push_i32(&b, offsetof(struct dw_push_layout, ker_w));
   nir_def *py = load_push_i32(&b, offsetof(struct dw_push_layout, pad_y));
   nir_def *px = load_push_i32(&b, offsetof(struct dw_push_layout, pad_x));
   nir_def *sy = load_push_i32(&b, offsetof(struct dw_push_layout, str_y));
   nir_def *sx = load_push_i32(&b, offsetof(struct dw_push_layout, str_x));
   nir_def *dy = load_push_i32(&b, offsetof(struct dw_push_layout, dil_y));
   nir_def *dx = load_push_i32(&b, offsetof(struct dw_push_layout, dil_x));

   nir_def *ox = nir_idiv(&b, gx, ch);
   nir_def *c = nir_irem(&b, gx, ch);

   nir_def *in_bounds = nir_iand(&b, nir_ilt(&b, ox, ow), nir_ilt(&b, gy, oh));
   nir_push_if(&b, in_bounds);
   {
      nir_variable *acc_var =
         nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);

      nir_variable *ky_var =
         nir_local_variable_create(b.impl, glsl_int_type(), "ky");
      nir_store_var(&b, ky_var, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         nir_def *ky = nir_load_var(&b, ky_var);
         nir_push_if(&b, nir_ige(&b, ky, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *iy = nir_iadd(&b, nir_iadd(&b, nir_imul(&b, gy, sy),
                                             nir_imul(&b, ky, dy)), py);
         nir_def *iy_ok =
            nir_iand(&b, nir_ige(&b, iy, nir_imm_int(&b, 0)), nir_ilt(&b, iy, ih));
         nir_push_if(&b, iy_ok);
         {
            nir_variable *kx_var =
               nir_local_variable_create(b.impl, glsl_int_type(), "kx");
            nir_store_var(&b, kx_var, nir_imm_int(&b, 0), 1);
            nir_push_loop(&b);
            {
               nir_def *kx = nir_load_var(&b, kx_var);
               nir_push_if(&b, nir_ige(&b, kx, kw));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_def *ix = nir_iadd(&b, nir_iadd(&b, nir_imul(&b, ox, sx),
                                                   nir_imul(&b, kx, dx)), px);
               nir_def *ix_ok =
                  nir_iand(&b, nir_ige(&b, ix, nir_imm_int(&b, 0)), nir_ilt(&b, ix, iw));
               nir_push_if(&b, ix_ok);
               {
                  nir_def *in_idx =
                     nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, iy, iw), ix), ch), c);
                  nir_def *wt_idx =
                     nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, ky, kw), kx), ch), c);
                  nir_def *in_v =
                     nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, set, in_b, in_idx)));
                  nir_def *wt_v =
                     nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, set, wt_b, wt_idx)));
                  nir_def *prod = nir_fmul(&b, in_v, wt_v);
                  nir_def *sum = nir_fadd(&b, nir_load_var(&b, acc_var), prod);
                  nir_store_var(&b, acc_var, sum, 1);
               }
               nir_pop_if(&b, NULL);
               nir_store_var(&b, kx_var, nir_iadd_imm(&b, kx, 1), 1);
            }
            nir_pop_loop(&b, NULL);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, ky_var, nir_iadd_imm(&b, ky, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_def *val = nir_load_var(&b, acc_var);
      if (with_bias_buf) {
         if (with_bias_flag) {
            nir_def *hb =
               load_push_i32(&b, offsetof(struct dw_push_layout, has_bias));
            nir_push_if(&b, nir_ine_imm(&b, hb, 0));
            {
               nir_def *bv =
                  nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, set, bs_b, c)));
               nir_store_var(&b, acc_var, nir_fadd(&b, val, bv), 1);
            }
            nir_pop_if(&b, NULL);
            val = nir_load_var(&b, acc_var);
         } else {
            nir_def *bv =
               nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, set, bs_b, c)));
            val = nir_fadd(&b, val, bv);
         }
      }
      nir_def *out_idx =
         nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, gy, ow), ox), ch), c);
      nir_store_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, set, out_b, out_idx)),
                      val, 0x1);
   }
   nir_pop_if(&b, NULL);

   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "build_depthwise_nir");
   return s;
}

/* Permuted push layout for generalized GEMM testing. */
struct gemm_push_layout {
   int32_t k;
   int32_t bias_mode;
   int32_t n;
   int32_t m;
};

static nir_shader *
build_gemm_nir(uint32_t rows_per_thread, bool guarded, bool has_bias_buf,
               bool has_bias_mode)
{
   nir_shader *s = new_compute_shader("gemm", 8, 4, 1);
   s->info.num_ssbos = has_bias_buf ? 4 : 3;
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *col = nir_channel(&b, gid, 0);
   nir_def *row_base = nir_imul_imm(&b, nir_channel(&b, gid, 1), rows_per_thread);

   nir_def *K = load_push_i32(&b, offsetof(struct gemm_push_layout, k));
   nir_def *N = load_push_i32(&b, offsetof(struct gemm_push_layout, n));
   nir_def *M = load_push_i32(&b, offsetof(struct gemm_push_layout, m));
   nir_def *bmode = has_bias_mode
      ? load_push_i32(&b, offsetof(struct gemm_push_layout, bias_mode))
      : NULL;

   nir_def *col_ok = guarded ? nir_ilt(&b, col, N) : nir_imm_true(&b);
   for (uint32_t r = 0; r < rows_per_thread; ++r) {
      nir_def *row = nir_iadd_imm(&b, row_base, r);
      nir_def *cond = guarded ? nir_iand(&b, nir_ilt(&b, row, M), col_ok) : NULL;
      if (guarded)
         nir_push_if(&b, cond);

      nir_variable *acc =
         nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

      nir_variable *kv =
         nir_local_variable_create(b.impl, glsl_int_type(), "k");
      nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         nir_def *k = nir_load_var(&b, kv);
         nir_push_if(&b, nir_ige(&b, k, K));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *a_idx = nir_iadd(&b, nir_imul(&b, row, K), k);
         nir_def *b_idx = nir_iadd(&b, nir_imul(&b, k, N), col);
         nir_def *av =
            nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, 0, 0, a_idx)));
         nir_def *bv =
            nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, 0, 1, b_idx)));
         nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), nir_fmul(&b, av, bv)), 1);
         nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_def *val = nir_load_var(&b, acc);
      if (has_bias_buf) {
         if (has_bias_mode) {
            nir_push_if(&b, nir_ieq_imm(&b, bmode, 0));
            {
               nir_def *bv =
                  nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, 0, 2, row)));
               nir_store_var(&b, acc, nir_fadd(&b, val, bv), 1);
            }
            nir_push_else(&b, NULL);
            {
               nir_push_if(&b, nir_ieq_imm(&b, bmode, 1));
               {
                  nir_def *bv =
                     nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, 0, 2, col)));
                  nir_store_var(&b, acc, nir_fadd(&b, val, bv), 1);
               }
               nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
            val = nir_load_var(&b, acc);
         } else {
            nir_def *bv =
               nir_load_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, 0, 2, col)));
            val = nir_fadd(&b, val, bv);
         }
      }

      nir_def *o_idx = nir_iadd(&b, nir_imul(&b, row, N), col);
      nir_store_deref(&b, nir_def_as_deref(ssbo_elem_deref(&b, 0, 3, o_idx)),
                      val, 0x1);
      if (guarded)
         nir_pop_if(&b, NULL);
   }

   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "build_gemm_nir");
   return s;
}

static void
test_depthwise_clamp_footprint(void)
{
   struct hk_nir_depthwise_output dw = {
      .output = { .set = 0, .binding = 3, .offset_B = 0 },
      .reads = {
         { .set = 0, .binding = 0, .offset_B = 0 },
         { .set = 0, .binding = 1, .offset_B = 0 },
         { .set = 0, .binding = 2, .offset_B = 0 },
      },
      .width_offset_B = 0,
      .height_offset_B = 4,
      .channels_offset_B = 8,
      .input_width_offset_B = 12,
      .input_height_offset_B = 16,
      .kernel_width_offset_B = 20,
      .kernel_height_offset_B = 24,
      .stride_x_offset_B = 28,
      .stride_y_offset_B = 32,
      .dil_x_offset_B = 36,
      .dil_y_offset_B = 40,
      .pad_x_offset_B = 44,
      .pad_y_offset_B = 48,
      .has_bias_offset_B = 52,
      .push_bytes = 56,
      .local_size = { 16, 16, 1 },
      .has_bias_buffer = true,
      .vec4_valid = true,
   };
   struct hk_nir_pointwise_clamp clamp = {
      .input = { .set = 0, .binding = 0, .offset_B = 0 },
      .output = { .set = 0, .binding = 0, .offset_B = 0 },
      .count_offset_B = 0,
      .epilogue = {
         .has_lower = true,
         .has_upper = true,
         .lower_bits = 0x00000000u,
         .upper_bits = 0x40c00000u,
         .max_fp_math_ctrl = nir_fp_preserve_inf,
         .min_fp_math_ctrl = nir_fp_preserve_inf,
      },
      .local_size_x = 64,
   };

   int32_t dw_push[14] = {
      64, 64, 32, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1
   };
   int32_t clamp_count = 64 * 64 * 32;
   struct hk_nir_direct_shape prod = {
      .push = dw_push,
      .push_size_B = sizeof(dw_push),
      .groups = { (64 * 32) / 16, 64 / 16, 1 },
      .base_group = { 0, 0, 0 },
   };
   struct hk_nir_direct_shape cons = {
      .push = &clamp_count,
      .push_size_B = sizeof(clamp_count),
      .groups = { (64 * 64 * 32) / 64, 1, 1 },
      .base_group = { 0, 0, 0 },
   };
   uint64_t p_ends[4] = {0}, c_ends[2] = {0};
   HK_EVAL_CHECK(hk_nir_depthwise_clamp_footprint(&dw, &clamp, &prod, &cons, p_ends, c_ends));
   HK_EVAL_CHECK(p_ends[2] == 32u * 4u);
   HK_EVAL_CHECK(p_ends[3] == (uint64_t)64 * 64 * 32 * 4);
   HK_EVAL_CHECK(c_ends[0] == p_ends[3] && c_ends[1] == p_ends[3]);

   /* Biasless depthwise produces p_ends[2] == 0. */
   struct hk_nir_depthwise_output dw_nobias = dw;
   dw_nobias.has_bias_buffer = false;
   dw_nobias.has_bias_offset_B = UINT32_MAX;
   HK_EVAL_CHECK(hk_nir_depthwise_clamp_footprint(&dw_nobias, &clamp, &prod, &cons, p_ends, c_ends));
   HK_EVAL_CHECK(p_ends[2] == 0);

   int32_t bad_count = clamp_count - 1;
   cons.push = &bad_count;
   HK_EVAL_CHECK(!hk_nir_depthwise_clamp_footprint(&dw, &clamp, &prod, &cons, p_ends, c_ends));

   cons.push = &clamp_count;
   prod.groups[0] -= 1;
   HK_EVAL_CHECK(!hk_nir_depthwise_clamp_footprint(&dw, &clamp, &prod, &cons, p_ends, c_ends));

   /* Nonzero base_group on either dispatch must refuse fusion and fall back
    * to the original shaders (required for bare unit-axis load_workgroup_id). */
   prod.groups[0] += 1;
   for (unsigned axis = 0; axis < 3; ++axis) {
      prod.base_group[axis] = 1;
      HK_EVAL_CHECK(!hk_nir_depthwise_clamp_footprint(&dw, &clamp, &prod, &cons, p_ends, c_ends));
      prod.base_group[axis] = 0;

      cons.base_group[axis] = 1;
      HK_EVAL_CHECK(!hk_nir_depthwise_clamp_footprint(&dw, &clamp, &prod, &cons, p_ends, c_ends));
      cons.base_group[axis] = 0;
   }
}

static void
test_gemm_clamp_footprint(void)
{
   struct hk_nir_gemm_output gemm = {
      .output = { .set = 0, .binding = 3, .offset_B = 0 },
      .reads = {
         { .set = 0, .binding = 0, .offset_B = 0 },
         { .set = 0, .binding = 1, .offset_B = 0 },
         { .set = 0, .binding = 2, .offset_B = 0 },
      },
      .m_offset_B = 0,
      .n_offset_B = 4,
      .k_offset_B = 8,
      .bias_mode_offset_B = 12,
      .local_size = { 32, 4, 1 },
      .rows_per_thread = 8,
      .bias_mode_mask = 0x7u,
      .has_bias_mode = true,
      .has_bias_buffer = true,
      .is_guarded = true,
   };
   struct hk_nir_pointwise_clamp clamp = {
      .input = { .set = 0, .binding = 0, .offset_B = 0 },
      .output = { .set = 0, .binding = 0, .offset_B = 0 },
      .count_offset_B = 0,
      .epilogue = {
         .has_lower = true,
         .has_upper = true,
         .lower_bits = 0x00000000u,
         .upper_bits = 0x40c00000u,
         .max_fp_math_ctrl = nir_fp_preserve_inf,
         .min_fp_math_ctrl = nir_fp_preserve_inf,
      },
      .local_size_x = 64,
   };

   int32_t g_push[4] = { 128, 64, 32, 2 };
   int32_t clamp_count = 128 * 64;
   struct hk_nir_direct_shape prod = {
      .push = g_push,
      .push_size_B = sizeof(g_push),
      .groups = { 64 / 32, 128 / (4 * 8), 1 },
      .base_group = { 0, 0, 0 },
   };
   struct hk_nir_direct_shape cons = {
      .push = &clamp_count,
      .push_size_B = sizeof(clamp_count),
      .groups = { (128 * 64) / 64, 1, 1 },
      .base_group = { 0, 0, 0 },
   };
   uint64_t p_ends[4] = {0}, c_ends[2] = {0};
   HK_EVAL_CHECK(hk_nir_gemm_clamp_footprint(&gemm, &clamp, &prod, &cons, p_ends, c_ends));
   HK_EVAL_CHECK(p_ends[2] == 0);
   HK_EVAL_CHECK(p_ends[3] == (uint64_t)128 * 64 * 4);
   HK_EVAL_CHECK(c_ends[0] == p_ends[3] && c_ends[1] == p_ends[3]);

   g_push[3] = 1;
   HK_EVAL_CHECK(hk_nir_gemm_clamp_footprint(&gemm, &clamp, &prod, &cons, p_ends, c_ends));
   HK_EVAL_CHECK(p_ends[2] == (uint64_t)64 * 4);

   g_push[3] = 0;
   HK_EVAL_CHECK(hk_nir_gemm_clamp_footprint(&gemm, &clamp, &prod, &cons, p_ends, c_ends));
   HK_EVAL_CHECK(p_ends[2] == (uint64_t)128 * 4);

   g_push[3] = 3;
   HK_EVAL_CHECK(!hk_nir_gemm_clamp_footprint(&gemm, &clamp, &prod, &cons, p_ends, c_ends));

   /* 4 rows per thread, biasless GEMM. */
   struct hk_nir_gemm_output gemm4 = gemm;
   gemm4.rows_per_thread = 4;
   gemm4.has_bias_mode = false;
   gemm4.has_bias_buffer = false;
   gemm4.bias_mode_mask = 0;
   prod.groups[1] = 128 / (4 * 4);
   HK_EVAL_CHECK(hk_nir_gemm_clamp_footprint(&gemm4, &clamp, &prod, &cons, p_ends, c_ends));
   HK_EVAL_CHECK(p_ends[2] == 0);

   /* Nonzero base_group on either dispatch must refuse GEMM+clamp fusion. */
   for (unsigned axis = 0; axis < 3; ++axis) {
      prod.base_group[axis] = 1;
      HK_EVAL_CHECK(!hk_nir_gemm_clamp_footprint(&gemm4, &clamp, &prod, &cons, p_ends, c_ends));
      prod.base_group[axis] = 0;

      cons.base_group[axis] = 1;
      HK_EVAL_CHECK(!hk_nir_gemm_clamp_footprint(&gemm4, &clamp, &prod, &cons, p_ends, c_ends));
      cons.base_group[axis] = 0;
   }
}

static void
run_hk_preprocess_for_test(nir_shader *nir, bool m5)
{
   nir_lower_system_values(nir);
   if (m5)
      agx_preprocess_nir_m5_defer_idiv(nir);
   else
      agx_preprocess_nir_native_defer_idiv(nir, AGX_CHIP_G17P);
   nir_lower_load_const_to_scalar(nir);
   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = true,
   };
   nir_lower_compute_system_values(nir, &csv_options);
}

static void
rewrite_ssbo_desc_comps(nir_shader *nir, unsigned comps)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
            if ((op->intrinsic == nir_intrinsic_vulkan_resource_index ||
                 op->intrinsic == nir_intrinsic_load_vulkan_descriptor) &&
                nir_intrinsic_desc_type(op) == nir_descriptor_type_storage_buffer) {
               op->num_components = comps;
               op->def.num_components = comps;
            }
         }
      }
   }
}

static void
test_clamp_match_and_splice(void)
{
   const struct hk_nir_clamp_epilogue cases[] = {
      /* Lower-only ReLU */
      { .has_lower = true, .has_upper = false, .lower_bits = 0x00000000u,
        .max_fp_math_ctrl = nir_fp_preserve_inf },
      /* Upper-only min(x, 6) */
      { .has_lower = false, .has_upper = true, .upper_bits = 0x40c00000u,
        .min_fp_math_ctrl = nir_fp_preserve_inf },
      /* ReLU6 [0, 6] */
      { .has_lower = true, .has_upper = true, .upper_first = false,
        .lower_bits = 0x00000000u, .upper_bits = 0x40c00000u,
        .max_fp_math_ctrl = nir_fp_preserve_inf,
        .min_fp_math_ctrl = nir_fp_preserve_inf },
      /* Arbitrary [-1.5, 3.25], lower first */
      { .has_lower = true, .has_upper = true, .upper_first = false,
        .lower_bits = 0xbfc00000u, .upper_bits = 0x40500000u,
        .max_fp_math_ctrl = nir_fp_preserve_nan,
        .min_fp_math_ctrl = nir_fp_preserve_inf },
      /* Arbitrary [-1.5, 3.25], upper first */
      { .has_lower = true, .has_upper = true, .upper_first = true,
        .lower_bits = 0xbfc00000u, .upper_bits = 0x40500000u,
        .max_fp_math_ctrl = nir_fp_preserve_inf,
        .min_fp_math_ctrl = nir_fp_preserve_nan },
   };

   /* Verify NaN / Inf bounds and invalid fp_math_ctrl bits are rejected. */
   struct hk_nir_clamp_epilogue bad_ctrl = cases[0];
   bad_ctrl.max_fp_math_ctrl = nir_fp_exact;
   HK_EVAL_CHECK(!hk_nir_clamp_epilogue_valid(&bad_ctrl));
   struct hk_nir_clamp_epilogue bad_nan = cases[0];
   bad_nan.lower_bits = 0x7fc00000u;
   HK_EVAL_CHECK(!hk_nir_clamp_epilogue_valid(&bad_nan));
   nir_shader *nan_nir = build_clamp_nir(0, 0, 0, 4, &bad_nan, 32);
   struct hk_nir_pointwise_clamp m_bad = {0};
   HK_EVAL_CHECK(!hk_nir_match_pointwise_clamp(nan_nir, &m_bad));
   ralloc_free(nan_nir);

   /* Special FP32 test vector: negative, -0.0, +0.0, mid, high, +Inf, -Inf, NaN */
   const uint32_t sample_bits[16] = {
      0xc1200000u, /* -10.0 */
      0xbfc00000u, /* -1.5 */
      0xbf000000u, /* -0.5 */
      0x80000000u, /* -0.0 */
      0x00000000u, /* +0.0 */
      0x3f000000u, /* 0.5 */
      0x40500000u, /* 3.25 */
      0x40c00000u, /* 6.0 */
      0x41200000u, /* 10.0 */
      0x7f800000u, /* +Inf */
      0xff800000u, /* -Inf */
      0x7fc00000u, /* qNaN */
      0xc0000000u, /* -2.0 */
      0x3fc00000u, /* 1.5 */
      0x40a00000u, /* 5.0 */
      0x42c80000u, /* 100.0 */
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      nir_shader *clamp_nir = build_clamp_nir(1, 2, 2, 8, &cases[ci], 16);
      struct hk_nir_pointwise_clamp matched = {0};
      HK_EVAL_CHECK(hk_nir_match_pointwise_clamp(clamp_nir, &matched));
      HK_EVAL_CHECK(matched.input.set == 1 && matched.input.binding == 2);
      HK_EVAL_CHECK(matched.output.set == 1 && matched.output.binding == 2);
      HK_EVAL_CHECK(matched.count_offset_B == 8);
      HK_EVAL_CHECK(matched.local_size_x == 16);
      HK_EVAL_CHECK(hk_nir_clamp_epilogue_equal(&matched.epilogue, &cases[ci]));

      /* Also verify post-preprocess matching on both G17P and M5 paths. */
      nir_shader *g17p = nir_shader_clone(NULL, clamp_nir);
      run_hk_preprocess_for_test(g17p, false);
      struct hk_nir_pointwise_clamp m_g17p = {0};
      HK_EVAL_CHECK(hk_nir_match_pointwise_clamp(g17p, &m_g17p));
      HK_EVAL_CHECK(hk_nir_clamp_epilogue_equal(&m_g17p.epilogue, &cases[ci]));
      ralloc_free(g17p);

      nir_shader *m5 = nir_shader_clone(NULL, clamp_nir);
      rewrite_ssbo_desc_comps(m5, 5);
      run_hk_preprocess_for_test(m5, true);
      struct hk_nir_pointwise_clamp m_m5 = {0};
      HK_EVAL_CHECK(hk_nir_match_pointwise_clamp(m5, &m_m5));
      HK_EVAL_CHECK(hk_nir_clamp_epilogue_equal(&m_m5.epilogue, &cases[ci]));
      ralloc_free(m5);

      /* Build a simple pass-through producer and splice epilogue onto it. */
      nir_shader *prod = new_compute_shader("copy_prod", 16, 1, 1);
      prod->info.num_ssbos = 2;
      nir_builder pb = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(prod)));
      nir_def *gx = nir_channel(&pb, nir_load_global_invocation_id(&pb, 32), 0);
      nir_def *v = nir_load_deref(&pb, nir_def_as_deref(ssbo_elem_deref(&pb, 1, 0, gx)));
      nir_store_deref(&pb, nir_def_as_deref(ssbo_elem_deref(&pb, 1, 2, gx)), v, 0x1);
      nir_validate_shader(prod, "copy_prod");

      nir_shader *spliced = hk_nir_splice_epilogue(
         prod, (struct hk_nir_clamp_buffer){ .set = 1, .binding = 2 }, &cases[ci]);
      HK_EVAL_CHECK(spliced);
      nir_validate_shader(spliced, "spliced_copy_prod");

      uint32_t in_buf[16], ref_buf[16], fused_buf[16];
      memcpy(in_buf, sample_bits, sizeof(in_buf));
      memset(ref_buf, 0, sizeof(ref_buf));
      memset(fused_buf, 0, sizeof(fused_buf));

      uint32_t push[3] = { 0, 0, 16 };
      struct hk_eval_ssbo ssbos_ref[2] = {
         { .set = 1, .binding = 0, .data = in_buf, .size = sizeof(in_buf) },
         { .set = 1, .binding = 2, .data = ref_buf, .size = sizeof(ref_buf) },
      };
      struct hk_eval_ssbo ssbos_fused[2] = {
         { .set = 1, .binding = 0, .data = in_buf, .size = sizeof(in_buf) },
         { .set = 1, .binding = 2, .data = fused_buf, .size = sizeof(fused_buf) },
      };
      const uint32_t grid[3] = { 1, 1, 1 };
      hk_eval_dispatch(prod, grid, ssbos_ref, 2, push, sizeof(push));
      hk_eval_dispatch(clamp_nir, grid, ssbos_ref, 2, push, sizeof(push));
      hk_eval_dispatch(spliced, grid, ssbos_fused, 2, push, sizeof(push));

      HK_EVAL_CHECK(memcmp(ref_buf, fused_buf, sizeof(ref_buf)) == 0);

      ralloc_free(spliced);
      ralloc_free(prod);
      ralloc_free(clamp_nir);
   }
}

static void
run_depthwise_case(bool with_bias_buf, bool with_bias_flag,
                   const struct dw_push_layout *cfg,
                   const struct hk_nir_clamp_epilogue *ep)
{
   nir_shader *orig = build_depthwise_nir(with_bias_buf, with_bias_flag,
                                          0, 0, 1, 2, 3);
   struct hk_nir_depthwise_output info = {0};
   HK_EVAL_CHECK(hk_nir_match_depthwise_output(orig, nir_address_format_64bit_global_32bit_offset, &info));
   HK_EVAL_CHECK(info.vec4_valid);
   HK_EVAL_CHECK(info.has_bias_buffer == with_bias_buf);
   HK_EVAL_CHECK(info.width_offset_B == offsetof(struct dw_push_layout, out_w));
   HK_EVAL_CHECK(info.channels_offset_B == offsetof(struct dw_push_layout, channels));
   HK_EVAL_CHECK(info.stride_x_offset_B == offsetof(struct dw_push_layout, str_x));

   /* Verify post-preprocess matching on both G17P and M5. */
   nir_shader *g17p = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(g17p, false);
   struct hk_nir_depthwise_output m_g17p = {0};
   HK_EVAL_CHECK(hk_nir_match_depthwise_output(g17p, nir_address_format_64bit_global_32bit_offset, &m_g17p));
   HK_EVAL_CHECK(m_g17p.vec4_valid);
   ralloc_free(g17p);

   nir_shader *m5 = nir_shader_clone(NULL, orig);
   rewrite_ssbo_desc_comps(m5, 5);
      run_hk_preprocess_for_test(m5, true);
   struct hk_nir_depthwise_output m_m5 = {0};
   HK_EVAL_CHECK(hk_nir_match_depthwise_output(m5, nir_address_format_64bit_global_32bit_offset_map, &m_m5));
   HK_EVAL_CHECK(m_m5.vec4_valid);
   ralloc_free(m5);

   nir_shader *vec4 = hk_nir_build_depthwise_vec4(orig, &info, ep, HK_COMPUTE_PAIR_DUAL_NONE);
   HK_EVAL_CHECK(vec4);
   nir_validate_shader(vec4, "depthwise_vec4");

   nir_shader *spliced = ep ? hk_nir_splice_epilogue(orig, info.output, ep) : NULL;
   if (spliced)
      nir_validate_shader(spliced, "depthwise_spliced");

   const size_t in_n = (size_t)cfg->in_w * cfg->in_h * cfg->channels;
   const size_t wt_n = (size_t)cfg->ker_w * cfg->ker_h * cfg->channels;
   const size_t bs_n = (size_t)cfg->channels;
   const size_t out_n = (size_t)cfg->out_w * cfg->out_h * cfg->channels;

   float *in_buf = calloc(in_n, sizeof(float));
   float *wt_buf = calloc(wt_n, sizeof(float));
   float *bs_buf = calloc(bs_n, sizeof(float));
   float *ref_out = calloc(out_n, sizeof(float));
   float *v4_out = calloc(out_n, sizeof(float));
   float *sp_out = calloc(out_n, sizeof(float));
   HK_EVAL_CHECK(in_buf && wt_buf && bs_buf && ref_out && v4_out && sp_out);

   /* Deterministic non-trivial values hitting both clamp bounds & negatives. */
   for (size_t i = 0; i < in_n; ++i)
      in_buf[i] = ((int)((i * 7u + 3u) % 13u) - 6) * 0.25f;
   for (size_t i = 0; i < wt_n; ++i)
      wt_buf[i] = ((int)((i * 5u + 1u) % 9u) - 4) * 0.5f;
   for (size_t i = 0; i < bs_n; ++i)
      bs_buf[i] = ((int)(i % 7u) - 3) * 0.75f;

   struct hk_eval_ssbo ssbos[4] = {
      { .set = 0, .binding = 0, .data = in_buf, .size = in_n * sizeof(float) },
      { .set = 0, .binding = 1, .data = wt_buf, .size = wt_n * sizeof(float) },
      { .set = 0, .binding = 2, .data = bs_buf, .size = bs_n * sizeof(float) },
      { .set = 0, .binding = 3, .data = ref_out, .size = out_n * sizeof(float) },
   };

   const uint32_t orig_grid[3] = {
      DIV_ROUND_UP((uint32_t)cfg->out_w * cfg->channels, orig->info.workgroup_size[0]),
      DIV_ROUND_UP((uint32_t)cfg->out_h, orig->info.workgroup_size[1]),
      1,
   };
   hk_eval_dispatch(orig, orig_grid, ssbos, 4, cfg, sizeof(*cfg));
   if (ep) {
      nir_shader *clamp = build_clamp_nir(0, 3, 3, 0, ep, 64);
      int32_t count = (int32_t)out_n;
      const uint32_t c_grid[3] = { DIV_ROUND_UP((uint32_t)out_n, 64u), 1, 1 };
      hk_eval_dispatch(clamp, c_grid, ssbos, 4, &count, sizeof(count));
      ralloc_free(clamp);
   }

   ssbos[3].data = v4_out;
   const uint32_t out_x_pairs = DIV_ROUND_UP((uint32_t)cfg->out_w, 2u);
   const uint32_t out_y_units = (uint32_t)cfg->out_h;
   const uint32_t v4_grid[3] = {
      DIV_ROUND_UP((uint32_t)cfg->channels / 4u, vec4->info.workgroup_size[0]),
      DIV_ROUND_UP(out_x_pairs, vec4->info.workgroup_size[1]),
      DIV_ROUND_UP(out_y_units, vec4->info.workgroup_size[2]),
   };
   hk_eval_dispatch(vec4, v4_grid, ssbos, 4, cfg, sizeof(*cfg));
   HK_EVAL_CHECK(memcmp(ref_out, v4_out, out_n * sizeof(float)) == 0);

   if (spliced) {
      ssbos[3].data = sp_out;
      hk_eval_dispatch(spliced, orig_grid, ssbos, 4, cfg, sizeof(*cfg));
      HK_EVAL_CHECK(memcmp(ref_out, sp_out, out_n * sizeof(float)) == 0);
      ralloc_free(spliced);
   }

   free(in_buf);
   free(wt_buf);
   free(bs_buf);
   free(ref_out);
   free(v4_out);
   free(sp_out);
   ralloc_free(vec4);
   ralloc_free(orig);
}

static void
test_depthwise_vec4_and_fusion(void)
{
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = 0x00000000u, .upper_bits = 0x40c00000u,
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };
   const struct hk_nir_clamp_epilogue relu = {
      .has_lower = true, .has_upper = false,
      .lower_bits = 0x00000000u,
      .max_fp_math_ctrl = nir_fp_preserve_inf,
   };
   const struct hk_nir_clamp_epilogue arb_upper_first = {
      .has_lower = true, .has_upper = true, .upper_first = true,
      .lower_bits = 0xbfa00000u, /* -1.25 */
      .upper_bits = 0x40900000u, /*  4.50 */
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };

   /* 3x3 stride 1, odd ow=5 (tests has_p1 tail), padding borders, has_bias=1. */
   struct dw_push_layout cfg1 = {
      .channels = 8, .out_w = 5, .out_h = 4, .in_w = 5, .in_h = 4,
      .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
      .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg1, NULL);
   run_depthwise_case(true, true, &cfg1, &relu6);
   run_depthwise_case(true, true, &cfg1, &relu);
   run_depthwise_case(true, true, &cfg1, &arb_upper_first);

   /* Same with has_bias=0. */
   cfg1.has_bias = 0;
   run_depthwise_case(true, true, &cfg1, &relu6);

   /* 5x5 stride 2, even ow=6, dilation 1, padding 2, biasless 3-SSBO kernel. */
   struct dw_push_layout cfg2 = {
      .channels = 4, .out_w = 6, .out_h = 5, .in_w = 11, .in_h = 9,
      .ker_w = 5, .ker_h = 5, .pad_x = -2, .pad_y = -2,
      .str_x = 2, .str_y = 2, .dil_x = 1, .dil_y = 1, .has_bias = 0,
   };
   run_depthwise_case(false, false, &cfg2, NULL);
   run_depthwise_case(false, false, &cfg2, &arb_upper_first);

   /* Unconditional bias buffer (no has_bias push flag). */
   run_depthwise_case(true, false, &cfg2, &relu);

   /* 3x3 stride 1, even interior + borders (exercises 1x2 unrolled s=1 fast path). */
   struct dw_push_layout cfg3 = {
      .channels = 16, .out_w = 8, .out_h = 8, .in_w = 8, .in_h = 8,
      .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
      .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg3, NULL);
   run_depthwise_case(true, true, &cfg3, &relu6);
   cfg3.has_bias = 0;
   run_depthwise_case(true, true, &cfg3, &relu6);
   run_depthwise_case(false, false, &cfg3, NULL);

   /* 3x3 stride 1, odd ow=7, oh=7 (both interior 1x2 pairs and odd X/Y tails). */
   struct dw_push_layout cfg4 = {
      .channels = 8, .out_w = 7, .out_h = 7, .in_w = 7, .in_h = 7,
      .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
      .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg4, &relu6);

   /* 3x3 stride 1, single output row oh=1 (verifies oy_ok guard in 2-row fallback). */
   struct dw_push_layout cfg4b = {
      .channels = 8, .out_w = 6, .out_h = 1, .in_w = 6, .in_h = 1,
      .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
      .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg4b, &relu6);

   /* 3x3 stride 2, even interior + borders (exercises 1x2 unrolled s=2 fast path). */
   struct dw_push_layout cfg5 = {
      .channels = 12, .out_w = 6, .out_h = 6, .in_w = 12, .in_h = 12,
      .ker_w = 3, .ker_h = 3, .pad_x = 0, .pad_y = 0,
      .str_x = 2, .str_y = 2, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg5, NULL);
   run_depthwise_case(true, true, &cfg5, &relu6);

   /* 3x3 stride 2, odd ow=5, oh=5 with -1 padding. */
   struct dw_push_layout cfg6 = {
      .channels = 8, .out_w = 5, .out_h = 5, .in_w = 10, .in_h = 10,
      .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
      .str_x = 2, .str_y = 2, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg6, &relu6);

   /* 3x3 dilated (dx=2, dy=2), interior + borders (generic 1x2 interior loop). */
   struct dw_push_layout cfg7 = {
      .channels = 8, .out_w = 8, .out_h = 8, .in_w = 8, .in_h = 8,
      .ker_w = 3, .ker_h = 3, .pad_x = -2, .pad_y = -2,
      .str_x = 1, .str_y = 1, .dil_x = 2, .dil_y = 2, .has_bias = 1,
   };
   run_depthwise_case(true, true, &cfg7, &relu6);
}

static void
rewrite_gid_axis_to_wgid(nir_shader *s, unsigned target_axis, unsigned wg_axis,
                         bool add_base, unsigned base_axis, uint32_t base_scale)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *wg = nir_channel(&b, nir_load_workgroup_id(&b), wg_axis);
   if (add_base) {
      nir_def *bwg =
         nir_channel(&b, nir_load_base_workgroup_id(&b, 32), base_axis);
      if (base_scale != 1)
         bwg = nir_imul_imm(&b, bwg, base_scale);
      wg = nir_iadd(&b, wg, bwg);
   }
   nir_def *repl = nir_vec3(&b, target_axis == 0 ? wg : nir_channel(&b, gid, 0),
                            target_axis == 1 ? wg : nir_channel(&b, gid, 1),
                            target_axis == 2 ? wg : nir_channel(&b, gid, 2));
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic == nir_intrinsic_load_global_invocation_id &&
             &intrin->def != gid)
            nir_def_rewrite_uses(&intrin->def, repl);
      }
   }
   nir_opt_dce(s);
   nir_validate_shader(s, "rewrite_gid_axis_to_wgid");
}

static void
test_unit_axis_wgid_compute_pair(void)
{
   /* Shaders whose source reads SPIR-V WorkgroupId (gl_WorkGroupID) directly
    * on a unit axis (local_size[axis] == 1) lower to load_workgroup_id[axis]
    * (plus load_base_workgroup_id[axis] when base workgroup offsets are
    * lowered). Verify bit-exact orig-vs-alt outputs for depthwise,
    * pointwise_clamp, and gemm_output, plus negative guards and dispatch-side
    * nonzero-base refusal. */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = 0x00000000u, .upper_bits = 0x40c00000u,
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };

   /* 1. Depthwise 3x3 with wg=32x1x1 reading wg.y (+ base.y): verify bit-exact
    *    orig+clamp vs both vec4 alternate and spliced epilogue. */
   const struct dw_push_layout cfg = {
      .channels = 8, .out_w = 5, .out_h = 4, .in_w = 5, .in_h = 4,
      .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
      .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1,
   };
   for (unsigned add_base = 0; add_base < 2; ++add_base) {
      nir_shader *dw32 = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
      dw32->info.workgroup_size[0] = 32;
      dw32->info.workgroup_size[1] = 1;
      dw32->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(dw32, 1, 1, add_base != 0, 1, 1);

      struct hk_nir_depthwise_output info = {0};
      HK_EVAL_CHECK(hk_nir_match_depthwise_output(
         dw32, nir_address_format_64bit_global_32bit_offset, &info));
      HK_EVAL_CHECK(info.vec4_valid && info.local_size[0] == 32 &&
                    info.local_size[1] == 1);
      nir_shader *vec4 = hk_nir_build_depthwise_vec4(
         dw32, &info, &relu6, HK_COMPUTE_PAIR_DUAL_NONE);
      HK_EVAL_CHECK(vec4 != NULL);
      nir_validate_shader(vec4, "dw_vec4_wg32x1_relu6");
      nir_shader *spliced = hk_nir_splice_epilogue(dw32, info.output, &relu6);
      HK_EVAL_CHECK(spliced != NULL);
      nir_validate_shader(spliced, "dw_spliced_wg32x1_relu6");

      const size_t in_n = (size_t)cfg.in_h * cfg.in_w * cfg.channels;
      const size_t wt_n = (size_t)cfg.ker_h * cfg.ker_w * cfg.channels;
      const size_t bs_n = (size_t)cfg.channels;
      const size_t out_n = (size_t)cfg.out_h * cfg.out_w * cfg.channels;
      float *in_buf = calloc(in_n, sizeof(float));
      float *wt_buf = calloc(wt_n, sizeof(float));
      float *bs_buf = calloc(bs_n, sizeof(float));
      float *ref_out = calloc(out_n, sizeof(float));
      float *v4_out = calloc(out_n, sizeof(float));
      float *sp_out = calloc(out_n, sizeof(float));
      HK_EVAL_CHECK(in_buf && wt_buf && bs_buf && ref_out && v4_out && sp_out);
      for (size_t i = 0; i < in_n; ++i)
         in_buf[i] = ((int)((i * 7u + 3u + add_base) % 13u) - 6) * 0.25f;
      for (size_t i = 0; i < wt_n; ++i)
         wt_buf[i] = ((int)((i * 5u + 1u) % 9u) - 4) * 0.5f;
      for (size_t i = 0; i < bs_n; ++i)
         bs_buf[i] = ((int)(i % 7u) - 3) * 0.75f;

      struct hk_eval_ssbo ssbos[4] = {
         { .set = 0, .binding = 0, .data = in_buf, .size = in_n * sizeof(float) },
         { .set = 0, .binding = 1, .data = wt_buf, .size = wt_n * sizeof(float) },
         { .set = 0, .binding = 2, .data = bs_buf, .size = bs_n * sizeof(float) },
         { .set = 0, .binding = 3, .data = ref_out, .size = out_n * sizeof(float) },
      };
      const uint32_t orig_grid[3] = {
         DIV_ROUND_UP((uint32_t)cfg.out_w * cfg.channels, 32u),
         (uint32_t)cfg.out_h,
         1,
      };
      hk_eval_dispatch(dw32, orig_grid, ssbos, 4, &cfg, sizeof(cfg));
      nir_shader *clamp = build_clamp_nir(0, 3, 3, 0, &relu6, 64);
      int32_t count = (int32_t)out_n;
      const uint32_t c_grid[3] = { DIV_ROUND_UP((uint32_t)out_n, 64u), 1, 1 };
      hk_eval_dispatch(clamp, c_grid, ssbos, 4, &count, sizeof(count));
      ralloc_free(clamp);

      ssbos[3].data = v4_out;
      const uint32_t out_pairs = DIV_ROUND_UP((uint32_t)cfg.out_w, 2u);
      const uint32_t v4_grid[3] = {
         DIV_ROUND_UP((uint32_t)cfg.channels / 4u, vec4->info.workgroup_size[0]),
         DIV_ROUND_UP(out_pairs, vec4->info.workgroup_size[1]),
         DIV_ROUND_UP((uint32_t)cfg.out_h, vec4->info.workgroup_size[2]),
      };
      hk_eval_dispatch(vec4, v4_grid, ssbos, 4, &cfg, sizeof(cfg));
      HK_EVAL_CHECK(memcmp(ref_out, v4_out, out_n * sizeof(float)) == 0);

      ssbos[3].data = sp_out;
      hk_eval_dispatch(spliced, orig_grid, ssbos, 4, &cfg, sizeof(cfg));
      HK_EVAL_CHECK(memcmp(ref_out, sp_out, out_n * sizeof(float)) == 0);

      free(in_buf);
      free(wt_buf);
      free(bs_buf);
      free(ref_out);
      free(v4_out);
      free(sp_out);
      ralloc_free(spliced);
      ralloc_free(vec4);

      /* Negative 1: same load_workgroup_id.y with local_size[1] == 2 rejects. */
      dw32->info.workgroup_size[1] = 2;
      struct hk_nir_depthwise_output bad = {0};
      HK_EVAL_CHECK(!hk_nir_match_depthwise_output(
         dw32, nir_address_format_64bit_global_32bit_offset, &bad));
      ralloc_free(dw32);
   }

   /* Negative 2: wrong-axis base (wg.y + base.x) rejects. */
   {
      nir_shader *dw = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
      dw->info.workgroup_size[0] = 32;
      dw->info.workgroup_size[1] = 1;
      dw->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(dw, 1, 1, true, 0, 1);
      struct hk_nir_depthwise_output bad = {0};
      HK_EVAL_CHECK(!hk_nir_match_depthwise_output(
         dw, nir_address_format_64bit_global_32bit_offset, &bad));
      ralloc_free(dw);
   }

   /* Negative 3: base scaled by wrong constant (wg.y + base.y * 2) rejects. */
   {
      nir_shader *dw = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
      dw->info.workgroup_size[0] = 32;
      dw->info.workgroup_size[1] = 1;
      dw->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(dw, 1, 1, true, 1, 2);
      struct hk_nir_depthwise_output bad = {0};
      HK_EVAL_CHECK(!hk_nir_match_depthwise_output(
         dw, nir_address_format_64bit_global_32bit_offset, &bad));
      ralloc_free(dw);
   }

   /* Negative 4: wg.x used as row index with local_size[0] == 32 rejects. */
   {
      nir_shader *dw = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
      dw->info.workgroup_size[0] = 32;
      dw->info.workgroup_size[1] = 1;
      dw->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(dw, 1, 0, false, 0, 1);
      struct hk_nir_depthwise_output bad = {0};
      HK_EVAL_CHECK(!hk_nir_match_depthwise_output(
         dw, nir_address_format_64bit_global_32bit_offset, &bad));
      ralloc_free(dw);
   }

   /* 2. hk_nir_match_pointwise_clamp with local_size = 1x1x1 reading wg.x
    *    (+ base.x): match, splice, and verify bit-exact output. */
   for (unsigned add_base = 0; add_base < 2; ++add_base) {
      nir_shader *clamp = build_clamp_nir(1, 2, 2, 8, &relu6, 1);
      rewrite_gid_axis_to_wgid(clamp, 0, 0, add_base != 0, 0, 1);

      struct hk_nir_pointwise_clamp matched = {0};
      HK_EVAL_CHECK(hk_nir_match_pointwise_clamp(clamp, &matched));
      HK_EVAL_CHECK(matched.local_size_x == 1 && matched.count_offset_B == 8);
      HK_EVAL_CHECK(hk_nir_clamp_epilogue_equal(&matched.epilogue, &relu6));

      nir_shader *prod = new_compute_shader("copy_prod_wg1", 16, 1, 1);
      prod->info.num_ssbos = 2;
      nir_builder pb = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(prod)));
      nir_def *gx = nir_channel(&pb, nir_load_global_invocation_id(&pb, 32), 0);
      nir_def *v = nir_load_deref(&pb, nir_def_as_deref(ssbo_elem_deref(&pb, 1, 0, gx)));
      nir_store_deref(&pb, nir_def_as_deref(ssbo_elem_deref(&pb, 1, 2, gx)), v, 0x1);
      nir_shader *spliced = hk_nir_splice_epilogue(
         prod, (struct hk_nir_clamp_buffer){ .set = 1, .binding = 2 }, &matched.epilogue);
      HK_EVAL_CHECK(spliced != NULL);

      float in_buf[16], ref_buf[16] = {0}, fused_buf[16] = {0};
      for (unsigned i = 0; i < 16; ++i)
         in_buf[i] = ((int)i - 4) * 1.0f;
      uint32_t push[3] = { 0, 0, 16 };
      struct hk_eval_ssbo sr[2] = {
         { .set = 1, .binding = 0, .data = in_buf, .size = sizeof(in_buf) },
         { .set = 1, .binding = 2, .data = ref_buf, .size = sizeof(ref_buf) },
      };
      struct hk_eval_ssbo sf[2] = {
         { .set = 1, .binding = 0, .data = in_buf, .size = sizeof(in_buf) },
         { .set = 1, .binding = 2, .data = fused_buf, .size = sizeof(fused_buf) },
      };
      const uint32_t pgrid[3] = { 1, 1, 1 };
      const uint32_t cgrid[3] = { 16, 1, 1 };
      hk_eval_dispatch(prod, pgrid, sr, 2, push, sizeof(push));
      hk_eval_dispatch(clamp, cgrid, sr, 2, push, sizeof(push));
      hk_eval_dispatch(spliced, pgrid, sf, 2, push, sizeof(push));
      HK_EVAL_CHECK(memcmp(ref_buf, fused_buf, sizeof(ref_buf)) == 0);
      ralloc_free(spliced);
      ralloc_free(prod);

      clamp->info.workgroup_size[0] = 4;
      struct hk_nir_pointwise_clamp bad = {0};
      HK_EVAL_CHECK(!hk_nir_match_pointwise_clamp(clamp, &bad));
      ralloc_free(clamp);
   }

   /* 3. hk_nir_match_gemm_output with wg=32x1x1 reading wg.y (+ base.y):
    *    match, splice, and verify bit-exact orig+clamp vs spliced. */
   for (unsigned add_base = 0; add_base < 2; ++add_base) {
      const uint32_t rows_per_thread = 4;
      const int32_t M = 13, N = 11, K = 7;
      nir_shader *orig = build_gemm_nir(rows_per_thread, true, true, true);
      orig->info.workgroup_size[0] = 32;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 1, 1, add_base != 0, 1, 1);

      struct hk_nir_gemm_output info = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         orig, nir_address_format_64bit_global_32bit_offset, &info));
      HK_EVAL_CHECK(info.rows_per_thread == rows_per_thread &&
                    info.local_size[0] == 32 && info.local_size[1] == 1);

      nir_shader *spliced = hk_nir_splice_epilogue(orig, info.output, &relu6);
      HK_EVAL_CHECK(spliced != NULL);
      nir_validate_shader(spliced, "gemm_spliced_wg32x1");

      const size_t a_n = (size_t)M * K, b_n = (size_t)K * N,
                   bias_n = (size_t)MAX2(M, N), o_n = (size_t)M * N;
      float *a_buf = calloc(a_n, sizeof(float));
      float *b_buf = calloc(b_n, sizeof(float));
      float *bias_buf = calloc(bias_n, sizeof(float));
      float *ref_out = calloc(o_n, sizeof(float));
      float *sp_out = calloc(o_n, sizeof(float));
      HK_EVAL_CHECK(a_buf && b_buf && bias_buf && ref_out && sp_out);
      for (size_t i = 0; i < a_n; ++i)
         a_buf[i] = ((int)((i * 3u + 1u + add_base) % 11u) - 5) * 0.5f;
      for (size_t i = 0; i < b_n; ++i)
         b_buf[i] = ((int)((i * 7u + 2u) % 9u) - 4) * 0.25f;
      for (size_t i = 0; i < bias_n; ++i)
         bias_buf[i] = ((int)(i % 5u) - 2) * 0.5f;

      struct gemm_push_layout push = { .k = K, .bias_mode = 1, .n = N, .m = M };
      struct hk_eval_ssbo ssbos[4] = {
         { .set = 0, .binding = 0, .data = a_buf, .size = a_n * sizeof(float) },
         { .set = 0, .binding = 1, .data = b_buf, .size = b_n * sizeof(float) },
         { .set = 0, .binding = 2, .data = bias_buf, .size = bias_n * sizeof(float) },
         { .set = 0, .binding = 3, .data = ref_out, .size = o_n * sizeof(float) },
      };
      const uint32_t grid[3] = {
         DIV_ROUND_UP((uint32_t)N, 32u),
         DIV_ROUND_UP((uint32_t)M, rows_per_thread),
         1,
      };
      hk_eval_dispatch(orig, grid, ssbos, 4, &push, sizeof(push));
      nir_shader *clamp = build_clamp_nir(0, 3, 3, 0, &relu6, 64);
      int32_t count = (int32_t)o_n;
      const uint32_t cgrid[3] = { DIV_ROUND_UP((uint32_t)o_n, 64u), 1, 1 };
      hk_eval_dispatch(clamp, cgrid, ssbos, 4, &count, sizeof(count));
      ralloc_free(clamp);

      ssbos[3].data = sp_out;
      hk_eval_dispatch(spliced, grid, ssbos, 4, &push, sizeof(push));
      HK_EVAL_CHECK(memcmp(ref_out, sp_out, o_n * sizeof(float)) == 0);

      free(a_buf);
      free(b_buf);
      free(bias_buf);
      free(ref_out);
      free(sp_out);
      ralloc_free(spliced);

      orig->info.workgroup_size[1] = 2;
      struct hk_nir_gemm_output bad = {0};
      HK_EVAL_CHECK(!hk_nir_match_gemm_output(
         orig, nir_address_format_64bit_global_32bit_offset, &bad));
      ralloc_free(orig);
   }

   /* 4. Dispatch-side nonzero base_group refusal gate. */
   const uint32_t zero_base[3] = { 0, 0, 0 };
   const uint32_t nz_y[3] = { 0, 1, 0 };
   HK_EVAL_CHECK(hk_nir_dispatch_zero_base(zero_base));
   HK_EVAL_CHECK(!hk_nir_dispatch_zero_base(nz_y));
}

static void
run_gemm_case(uint32_t rows_per_thread, bool guarded, bool has_bias_buf,
              bool has_bias_mode, int32_t M, int32_t N, int32_t K,
              int32_t bias_mode)
{
   nir_shader *orig =
      build_gemm_nir(rows_per_thread, guarded, has_bias_buf, has_bias_mode);
   struct hk_nir_gemm_output info = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(orig, nir_address_format_64bit_global_32bit_offset, &info));
   HK_EVAL_CHECK(info.rows_per_thread == rows_per_thread);
   HK_EVAL_CHECK(info.is_guarded == guarded);
   HK_EVAL_CHECK(info.has_bias_buffer == has_bias_buf);
   HK_EVAL_CHECK(info.has_bias_mode == has_bias_mode);
   HK_EVAL_CHECK(info.m_offset_B == (guarded ? offsetof(struct gemm_push_layout, m) : UINT32_MAX));
   HK_EVAL_CHECK(info.n_offset_B == offsetof(struct gemm_push_layout, n));
   HK_EVAL_CHECK(info.k_offset_B == offsetof(struct gemm_push_layout, k));

   /* Verify post-preprocess matching on both G17P and M5. */
   nir_shader *g17p = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(g17p, false);
   struct hk_nir_gemm_output m_g17p = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(g17p, nir_address_format_64bit_global_32bit_offset, &m_g17p));
   HK_EVAL_CHECK(m_g17p.rows_per_thread == rows_per_thread);
   ralloc_free(g17p);

   nir_shader *m5 = nir_shader_clone(NULL, orig);
   rewrite_ssbo_desc_comps(m5, 5);
      run_hk_preprocess_for_test(m5, true);
   struct hk_nir_gemm_output m_m5 = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(m5, nir_address_format_64bit_global_32bit_offset_map, &m_m5));
   HK_EVAL_CHECK(m_m5.rows_per_thread == rows_per_thread);
   ralloc_free(m5);

   const struct hk_nir_clamp_epilogue ep = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = 0xc0000000u, /* -2.0 */
      .upper_bits = 0x40b00000u, /*  5.5 */
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };
   nir_shader *spliced = hk_nir_splice_epilogue(orig, info.output, &ep);
   HK_EVAL_CHECK(spliced);
   nir_validate_shader(spliced, "gemm_spliced");

   const size_t a_n = (size_t)M * K;
   const size_t b_n = (size_t)K * N;
   const size_t bias_n = (size_t)MAX2(M, N);
   const size_t o_n = (size_t)M * N;
   float *a_buf = calloc(a_n, sizeof(float));
   float *b_buf = calloc(b_n, sizeof(float));
   float *bias_buf = calloc(bias_n, sizeof(float));
   float *ref_out = calloc(o_n, sizeof(float));
   float *sp_out = calloc(o_n, sizeof(float));
   HK_EVAL_CHECK(a_buf && b_buf && bias_buf && ref_out && sp_out);

   for (size_t i = 0; i < a_n; ++i)
      a_buf[i] = ((int)((i * 3u + 1u) % 11u) - 5) * 0.5f;
   for (size_t i = 0; i < b_n; ++i)
      b_buf[i] = ((int)((i * 7u + 2u) % 9u) - 4) * 0.25f;
   for (size_t i = 0; i < bias_n; ++i)
      bias_buf[i] = ((int)(i % 5u) - 2) * 0.5f;

   struct gemm_push_layout push = {
      .k = K, .bias_mode = bias_mode, .n = N, .m = M,
   };
   struct hk_eval_ssbo ssbos[4] = {
      { .set = 0, .binding = 0, .data = a_buf, .size = a_n * sizeof(float) },
      { .set = 0, .binding = 1, .data = b_buf, .size = b_n * sizeof(float) },
      { .set = 0, .binding = 2, .data = bias_buf, .size = bias_n * sizeof(float) },
      { .set = 0, .binding = 3, .data = ref_out, .size = o_n * sizeof(float) },
   };
   const uint32_t grid[3] = {
      DIV_ROUND_UP((uint32_t)N, orig->info.workgroup_size[0]),
      DIV_ROUND_UP((uint32_t)M, orig->info.workgroup_size[1] * rows_per_thread),
      1,
   };
   hk_eval_dispatch(orig, grid, ssbos, 4, &push, sizeof(push));

   nir_shader *clamp = build_clamp_nir(0, 3, 3, 0, &ep, 64);
   int32_t count = (int32_t)o_n;
   const uint32_t c_grid[3] = { DIV_ROUND_UP((uint32_t)o_n, 64u), 1, 1 };
   hk_eval_dispatch(clamp, c_grid, ssbos, 4, &count, sizeof(count));
   ralloc_free(clamp);

   ssbos[3].data = sp_out;
   hk_eval_dispatch(spliced, grid, ssbos, 4, &push, sizeof(push));
   HK_EVAL_CHECK(memcmp(ref_out, sp_out, o_n * sizeof(float)) == 0);

   free(a_buf);
   free(b_buf);
   free(bias_buf);
   free(ref_out);
   free(sp_out);
   ralloc_free(spliced);
   ralloc_free(orig);
}

static void
test_gemm_match_and_splice(void)
{
   /* 4 rows/thread, guarded, non-multiple dims, all 3 bias modes. */
   run_gemm_case(4, true, true, true, 13, 11, 7, 0);
   run_gemm_case(4, true, true, true, 13, 11, 7, 1);
   run_gemm_case(4, true, true, true, 13, 11, 7, 2);

   /* 8 rows/thread, unguarded, exact multiple dims, biasless 3-SSBO. */
   run_gemm_case(8, false, false, false, 32, 16, 6, 0);

   /* 8 rows/thread, guarded, fixed column bias (no bias_mode push). */
   run_gemm_case(8, true, true, false, 20, 13, 5, 0);
}

static nir_def *
ssbo_elem_deref_off(nir_builder *b, unsigned set, unsigned binding,
                    uint32_t base_off_B, nir_def *elem_idx)
{
   const struct glsl_type *arr = glsl_array_type(glsl_float_type(), 0, 4);
   struct glsl_struct_field field = {
      .type = arr,
      .name = "data",
      .offset = (int)base_off_B,
   };
   const struct glsl_type *iface =
      glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false, "Buf");
   nir_def *res = nir_vulkan_resource_index(
      b, 4, 32, nir_imm_int(b, 0), .desc_set = set, .binding = binding,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, 4, 32, res, .desc_type = nir_descriptor_type_storage_buffer);
   nir_deref_instr *blk =
      nir_build_deref_cast(b, desc, nir_var_mem_ssbo, iface, 0);
   nir_deref_instr *arr_d = nir_build_deref_struct(b, blk, 0);
   return &nir_build_deref_array(b, arr_d, elem_idx)->def;
}

struct gemm_test_opts {
   uint32_t a_extra_offset_B;
   bool guarded;
   bool has_bias_buf;
   bool has_bias_mode;
   uint32_t offset_B;
   unsigned fp_math_ctrl;
   uint32_t inner_bound;
   bool transpose_b;
   bool extra_store;
   bool preserve_nan;
   bool use_ffma;
   bool split_bias_stores;
   bool early_k_break;
   bool early_dot_break;
   bool swap_bias_arms;
   bool bias_in_mode2;
   bool drop_mode0_bias;
   bool fixed_row_bias;
   /* Guarded: store s_a directly in the guard arms instead of via a phi. */
   bool a_store_in_arms;
   /* With a_store_in_arms: extra `if (a_row != 0)` around the then-arm store. */
   bool a_store_inner_if;
   /* Wrap the whole C store section in an extra `if (col != 0)`. */
   bool c_store_outer_if;
   /* Split stores as `if (mode == 0) ... else if (mode == 1) ... else if (mode == 2)`. */
   bool nested_bias_chain;
   /* 32-step unrolled dot product with constant tile columns, no dot loop. */
   bool unroll_dot;
   /* With unroll_dot: read the shared tile before barrier_1. */
   bool unroll_early_loads;
};

/* barrier_1 followed by a 32-step unrolled dot product over the shared tile:
 * acc[r] += s_a[ly * 8 + r][kk] * s_b[kk][lx] for kk = 0..31. With
 * unroll_early_loads, the tile reads are emitted before barrier_1 instead. */
static void
emit_unrolled_dot(nir_builder *b, const struct gemm_test_opts *o,
                  nir_variable *s_a, nir_variable *s_b, nir_def *lx,
                  nir_def *ly, nir_variable *const acc[8])
{
   nir_def *a_ld[32][8], *b_ld[32];
   const bool early = o->unroll_early_loads;

   for (unsigned pass = 0; pass < 2; ++pass) {
      const bool load_pass = early ? pass == 0 : pass == 1;
      if (pass == 1)
         nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_shared);
      for (unsigned kk = 0; kk < 32; ++kk) {
         if (load_pass) {
            nir_def *kc = nir_imm_int(b, kk);
            b_ld[kk] = nir_load_deref(b, nir_build_deref_array(
               b, nir_build_deref_array(b, nir_build_deref_var(b, s_b), kc), lx));
            for (unsigned r = 0; r < 8; ++r) {
               nir_def *s_row = nir_iadd_imm(b, nir_ishl_imm(b, ly, 3), r);
               a_ld[kk][r] = nir_load_deref(b, nir_build_deref_array(
                  b, nir_build_deref_array(b, nir_build_deref_var(b, s_a), s_row), kc));
            }
         }
         if (pass == 0)
            continue;
         for (unsigned r = 0; r < 8; ++r) {
            b->fp_math_ctrl = o->fp_math_ctrl;
            nir_def *sum = o->use_ffma
               ? nir_ffma(b, a_ld[kk][r], b_ld[kk], nir_load_var(b, acc[r]))
               : nir_fadd(b, nir_load_var(b, acc[r]),
                          nir_fmul(b, a_ld[kk][r], b_ld[kk]));
            b->fp_math_ctrl = nir_fp_fast_math;
            nir_store_var(b, acc[r], sum, 1);
         }
      }
   }
}

static nir_shader *
build_tiled_gemm_nir_opts(const struct gemm_test_opts *o)
{
   const bool guarded = o->guarded;
   const bool has_bias_buf = o->has_bias_buf;
   const bool has_bias_mode = o->has_bias_mode;
   const uint32_t offset_B = o->offset_B;
   const uint32_t a_offset_B = offset_B + o->a_extra_offset_B;
   const unsigned fp_math_ctrl = o->fp_math_ctrl;
   const uint32_t inner_bound = o->inner_bound ? o->inner_bound : 32;
   const bool transpose_b = o->transpose_b;
   const bool extra_store = o->extra_store;
   const bool preserve_nan = o->preserve_nan;

   nir_shader *s = new_compute_shader("tiled_gemm", 32, 4, 1);
   s->info.num_ssbos = has_bias_buf ? 4 : 3;
   if (preserve_nan)
      s->info.float_controls_execution_mode |=
         FLOAT_CONTROLS_DENORM_PRESERVE_FP32;

   nir_variable *s_a = nir_variable_create(
      s, nir_var_mem_shared,
      glsl_array_type(glsl_array_type(glsl_float_type(), 32, 4), 32, 128), "s_a");
   nir_variable *s_b = nir_variable_create(
      s, nir_var_mem_shared,
      glsl_array_type(glsl_array_type(glsl_float_type(), 32, 4), 32, 128), "s_b");

   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *lid = nir_load_local_invocation_id(&b);
   nir_def *col = nir_channel(&b, gid, 0);
   nir_def *gid_y = nir_channel(&b, gid, 1);
   nir_def *lx = nir_channel(&b, lid, 0);
   nir_def *ly = nir_channel(&b, lid, 1);

   nir_def *K = load_push_i32(&b, offsetof(struct gemm_push_layout, k));
   nir_def *N = load_push_i32(&b, offsetof(struct gemm_push_layout, n));
   nir_def *M = load_push_i32(&b, offsetof(struct gemm_push_layout, m));
   nir_def *bmode = has_bias_mode
      ? load_push_i32(&b, offsetof(struct gemm_push_layout, bias_mode))
      : NULL;

   nir_variable *acc[8];
   for (unsigned r = 0; r < 8; ++r) {
      acc[r] = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_store_var(&b, acc[r], nir_imm_float(&b, 0.0f), 1);
   }
   nir_variable *tile_v = nir_local_variable_create(b.impl, glsl_int_type(), "t");
   nir_store_var(&b, tile_v, nir_imm_int(&b, 0), 1);
   nir_def *num_tiles = nir_idiv(&b, nir_iadd_imm(&b, K, 31), nir_imm_int(&b, 32));

   nir_push_loop(&b);
   {
      nir_def *t = nir_load_var(&b, tile_v);
      nir_push_if(&b, nir_ige(&b, t, num_tiles));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      if (o->early_k_break) {
         nir_push_if(&b, nir_ige_imm(&b, t, 1));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
      }

      nir_def *k_base = nir_ishl_imm(&b, t, 5);
      for (unsigned j = 0; j < 8; ++j) {
         nir_def *s_row = nir_iadd_imm(&b, nir_ishl_imm(&b, ly, 3), j);
         nir_def *a_row = nir_iadd_imm(&b, nir_ishl_imm(&b, gid_y, 3), j);
         nir_def *a_col = nir_iadd(&b, k_base, lx);
         nir_def *a_val = NULL;
         if (guarded && o->a_store_in_arms) {
            nir_deref_instr *da =
               nir_build_deref_array(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, s_a), s_row), lx);
            nir_push_if(&b, nir_iand(&b, nir_ilt(&b, a_row, M), nir_ilt(&b, a_col, K)));
            nir_def *a_idx = nir_iadd(&b, nir_imul(&b, a_row, K), a_col);
            nir_def *ld = nir_load_deref(&b, nir_def_as_deref(
               ssbo_elem_deref_off(&b, 0, 0, a_offset_B, a_idx)));
            if (o->a_store_inner_if)
               nir_push_if(&b, nir_ine_imm(&b, a_row, 0));
            nir_store_deref(&b, da, ld, 1);
            if (o->a_store_inner_if)
               nir_pop_if(&b, NULL);
            nir_push_else(&b, NULL);
            nir_store_deref(&b, da, nir_imm_float(&b, 0.0f), 1);
            nir_pop_if(&b, NULL);
         } else if (guarded) {
            nir_variable *av = nir_local_variable_create(b.impl, glsl_float_type(), "av");
            nir_push_if(&b, nir_iand(&b, nir_ilt(&b, a_row, M), nir_ilt(&b, a_col, K)));
            nir_def *a_idx = nir_iadd(&b, nir_imul(&b, a_row, K), a_col);
            nir_store_var(&b, av,
                          nir_load_deref(&b, nir_def_as_deref(
                             ssbo_elem_deref_off(&b, 0, 0, a_offset_B, a_idx))), 1);
            nir_push_else(&b, NULL);
            nir_store_var(&b, av, nir_imm_float(&b, 0.0f), 1);
            nir_pop_if(&b, NULL);
            a_val = nir_load_var(&b, av);
         } else {
            nir_def *a_idx = nir_iadd(&b, nir_imul(&b, a_row, K), a_col);
            a_val = nir_load_deref(&b, nir_def_as_deref(
               ssbo_elem_deref_off(&b, 0, 0, a_offset_B, a_idx)));
         }
         if (a_val) {
            nir_deref_instr *da =
               nir_build_deref_array(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, s_a), s_row), lx);
            nir_store_deref(&b, da, a_val, 1);
         }

         nir_def *b_row = nir_iadd(&b, k_base, s_row);
         nir_def *b_val;
         nir_def *b_idx = transpose_b
            ? nir_iadd(&b, nir_imul(&b, col, K), b_row)
            : nir_iadd(&b, nir_imul(&b, b_row, N), col);
         if (guarded) {
            nir_variable *bv = nir_local_variable_create(b.impl, glsl_float_type(), "bv");
            nir_push_if(&b, nir_iand(&b, nir_ilt(&b, b_row, K), nir_ilt(&b, col, N)));
            nir_store_var(&b, bv,
                          nir_load_deref(&b, nir_def_as_deref(
                             ssbo_elem_deref_off(&b, 0, 1, offset_B, b_idx))), 1);
            nir_push_else(&b, NULL);
            nir_store_var(&b, bv, nir_imm_float(&b, 0.0f), 1);
            nir_pop_if(&b, NULL);
            b_val = nir_load_var(&b, bv);
         } else {
            b_val = nir_load_deref(&b, nir_def_as_deref(
               ssbo_elem_deref_off(&b, 0, 1, offset_B, b_idx)));
         }
         nir_deref_instr *db =
            nir_build_deref_array(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, s_b), s_row), lx);
         nir_store_deref(&b, db, b_val, 1);
      }

      if (extra_store) {
         nir_store_deref(&b, nir_def_as_deref(
            ssbo_elem_deref_off(&b, 0, 3, offset_B, nir_imm_int(&b, 0))),
            nir_imm_float(&b, 0.0f), 1);
      }

      if (o->unroll_dot) {
         emit_unrolled_dot(&b, o, s_a, s_b, lx, ly, acc);
      } else {
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_shared);

         nir_variable *lv = nir_local_variable_create(b.impl, glsl_int_type(), "l");
         nir_store_var(&b, lv, nir_imm_int(&b, 0), 1);
         nir_push_loop(&b);
         {
            nir_def *l = nir_load_var(&b, lv);
            nir_push_if(&b, nir_ige_imm(&b, l, inner_bound));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);
            if (o->early_dot_break) {
               nir_push_if(&b, nir_ige_imm(&b, l, 16));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);
            }

            nir_deref_instr *db =
               nir_build_deref_array(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, s_b), l), lx);
            nir_def *bv = nir_load_deref(&b, db);
            for (unsigned r = 0; r < 8; ++r) {
               nir_def *s_row = nir_iadd_imm(&b, nir_ishl_imm(&b, ly, 3), r);
               nir_deref_instr *da =
                  nir_build_deref_array(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, s_a), s_row), l);
               nir_def *av = nir_load_deref(&b, da);
               b.fp_math_ctrl = fp_math_ctrl;
               nir_def *sum;
               if (o->use_ffma) {
                  sum = nir_ffma(&b, av, bv, nir_load_var(&b, acc[r]));
               } else {
                  nir_def *prod = nir_fmul(&b, av, bv);
                  sum = nir_fadd(&b, nir_load_var(&b, acc[r]), prod);
               }
               b.fp_math_ctrl = nir_fp_fast_math;
               nir_store_var(&b, acc[r], sum, 1);
            }
            nir_store_var(&b, lv, nir_iadd_imm(&b, l, 1), 1);
         }
         nir_pop_loop(&b, NULL);
      }

      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);

      nir_store_var(&b, tile_v, nir_iadd_imm(&b, t, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_def *row_base = nir_ishl_imm(&b, nir_channel(&b, gid, 1), 3);
   nir_def *col_ok = guarded ? nir_ilt(&b, col, N) : nir_imm_true(&b);
   if (o->c_store_outer_if)
      nir_push_if(&b, nir_ine_imm(&b, col, 0));
   if (has_bias_buf && has_bias_mode && o->split_bias_stores) {
      for (unsigned arm = 0; arm < 3; ++arm) {
         nir_if *arm_if = nir_push_if(&b, nir_ieq_imm(&b, bmode, arm));
         for (unsigned r = 0; r < 8; ++r) {
            nir_def *row = nir_iadd_imm(&b, row_base, r);
            if (guarded)
               nir_push_if(&b, nir_iand(&b, nir_ilt(&b, row, M), col_ok));
            nir_def *val = nir_load_var(&b, acc[r]);
            nir_def *b_idx = NULL;
            if (arm == 0 && !o->drop_mode0_bias)
               b_idx = o->swap_bias_arms ? col : row;
            else if (arm == 1)
               b_idx = o->swap_bias_arms ? row : col;
            else if (arm == 2 && o->bias_in_mode2)
               b_idx = row;
            if (b_idx) {
               nir_def *bv = nir_load_deref(&b, nir_def_as_deref(
                  ssbo_elem_deref_off(&b, 0, 2, offset_B, b_idx)));
               val = nir_fadd(&b, val, bv);
            }
            nir_def *o_idx = nir_iadd(&b, nir_imul(&b, row, N), col);
            nir_store_deref(&b, nir_def_as_deref(
               ssbo_elem_deref_off(&b, 0, 3, offset_B, o_idx)), val, 1);
            if (guarded)
               nir_pop_if(&b, NULL);
         }
         if (o->nested_bias_chain && arm < 2)
            nir_push_else(&b, arm_if);
         else
            nir_pop_if(&b, NULL);
      }
      if (o->nested_bias_chain) {
         nir_pop_if(&b, NULL);
         nir_pop_if(&b, NULL);
      }
   } else {
      for (unsigned r = 0; r < 8; ++r) {
         nir_def *row = nir_iadd_imm(&b, row_base, r);
         if (guarded)
            nir_push_if(&b, nir_iand(&b, nir_ilt(&b, row, M), col_ok));

         nir_def *val = nir_load_var(&b, acc[r]);
         if (has_bias_buf) {
            if (has_bias_mode) {
               nir_push_if(&b, nir_ieq_imm(&b, bmode, 0));
               {
                  if (!o->drop_mode0_bias) {
                     nir_def *bv = nir_load_deref(&b, nir_def_as_deref(
                        ssbo_elem_deref_off(&b, 0, 2, offset_B,
                                            o->swap_bias_arms ? col : row)));
                     nir_store_var(&b, acc[r], nir_fadd(&b, val, bv), 1);
                  }
               }
               nir_push_else(&b, NULL);
               {
                  nir_push_if(&b, nir_ieq_imm(&b, bmode, 1));
                  {
                     nir_def *bv = nir_load_deref(&b, nir_def_as_deref(
                        ssbo_elem_deref_off(&b, 0, 2, offset_B,
                                            o->swap_bias_arms ? row : col)));
                     nir_store_var(&b, acc[r], nir_fadd(&b, val, bv), 1);
                  }
                  nir_push_else(&b, NULL);
                  {
                     if (o->bias_in_mode2) {
                        nir_def *bv = nir_load_deref(&b, nir_def_as_deref(
                           ssbo_elem_deref_off(&b, 0, 2, offset_B, row)));
                        nir_store_var(&b, acc[r], nir_fadd(&b, val, bv), 1);
                     }
                  }
                  nir_pop_if(&b, NULL);
               }
               nir_pop_if(&b, NULL);
               val = nir_load_var(&b, acc[r]);
            } else {
               nir_def *bv = nir_load_deref(&b, nir_def_as_deref(
                  ssbo_elem_deref_off(&b, 0, 2, offset_B,
                                      o->fixed_row_bias ? row : col)));
               val = nir_fadd(&b, val, bv);
            }
         }

         nir_def *o_idx = nir_iadd(&b, nir_imul(&b, row, N), col);
         nir_store_deref(&b, nir_def_as_deref(
            ssbo_elem_deref_off(&b, 0, 3, offset_B, o_idx)), val, 1);

         if (guarded)
            nir_pop_if(&b, NULL);
      }
   }
   if (o->c_store_outer_if)
      nir_pop_if(&b, NULL);

   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "build_tiled_gemm_nir");
   return s;
}

static bool
shader_has_cmat_muladd(const nir_shader *s)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_cmat_muladd_agx)
               return true;
         }
      }
   }
   return false;
}

/* CPU reference for a shader fmul followed by fadd, each rounded on its own.
 * On targets with FMA (aarch64, x86-64-v3) the compiler may contract
 * `a * b + acc` into one fused multiply-add (clang's default
 * -ffp-contract=on within an expression, GCC's -ffp-contract=fast even
 * across statements), which rounds once and no longer matches the evaluated
 * shader bit for bit. Storing the product through a volatile forces it to be
 * rounded to float first under every compiler and contraction mode. */
static float
ref_mul_add_unfused(float a, float b, float acc)
{
   volatile float prod = a * b;
   return prod + acc;
}

static uint32_t
gemm_test_lcg(uint32_t *state)
{
   *state = *state * 1664525u + 1013904223u;
   return *state;
}

static float
gemm_test_rand_f32(uint32_t *state)
{
   uint32_t r = gemm_test_lcg(state);
   if ((r & 0x3fu) == 0)
      return 0.0f;
   if ((r & 0x7fu) == 1)
      return -0.0f;
   if ((r & 0xffu) == 2)
      return 1e-40f; /* subnormal */
   int32_t s = (int32_t)(r >> 12) - (1 << 19);
   return (float)s * (1.0f / 131072.0f);
}

/* Runs the original and the cmat alternate of `opts` through the CPU
 * workgroup evaluator and returns how many C elements differ bitwise.
 * seed == 0 feeds all-1.0f inputs and pre-fills C with -99.0f. With
 * `force_uncertified`, the shader must match as a GEMM but fail certification
 * (cmat_valid == false); the alternate is then built anyway to show what
 * certifying it would have produced. */
static unsigned
gemm_eval_mismatches(const struct gemm_test_opts *opts, uint32_t M,
                     uint32_t N, uint32_t K, int32_t bias_mode,
                     size_t runtime_a_off, size_t runtime_b_off,
                     uint32_t tile_n,
                     const struct hk_nir_clamp_epilogue *epilogue,
                     uint32_t seed, bool force_uncertified)
{
   nir_shader *orig = build_tiled_gemm_nir_opts(opts);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);

   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   HK_EVAL_CHECK(m.cmat_valid != force_uncertified);
   m.cmat_valid = true;

   nir_shader *alt = hk_nir_build_gemm_cmat(pre, &m, tile_n, epilogue, HK_COMPUTE_PAIR_DUAL_NONE);
   HK_EVAL_CHECK(alt);
   nir_validate_shader(alt, "gemm_cmat_eval_alt");
   HK_EVAL_CHECK(shader_has_cmat_muladd(alt));

   const uint32_t off_elems = opts->offset_B / 4u;
   const size_t a_elems = (size_t)M * K + off_elems;
   const size_t b_elems = (size_t)K * N + off_elems;
   const size_t bias_elems = (size_t)(M > N ? M : N) + off_elems + 1;
   const size_t c_elems = (size_t)M * N + off_elems;

   float *a_buf = calloc(a_elems, sizeof(float));
   float *b_buf = calloc(b_elems, sizeof(float));
   float *bias_buf = calloc(bias_elems, sizeof(float));
   float *c_orig = calloc(c_elems, sizeof(float));
   float *c_alt = calloc(c_elems, sizeof(float));
   HK_EVAL_CHECK(a_buf && b_buf && bias_buf && c_orig && c_alt);

   uint32_t rng = seed;
   for (size_t i = off_elems; i < a_elems; ++i)
      a_buf[i] = seed ? gemm_test_rand_f32(&rng) : 1.0f;
   for (size_t i = off_elems; i < b_elems; ++i)
      b_buf[i] = seed ? gemm_test_rand_f32(&rng) : 1.0f;
   for (size_t i = off_elems; i < bias_elems; ++i)
      bias_buf[i] = seed ? gemm_test_rand_f32(&rng) : 1.0f;
   for (size_t i = 0; i < c_elems; ++i) {
      if (i < off_elems || !seed) {
         c_orig[i] = seed ? -123.5f : -99.0f;
         c_alt[i] = c_orig[i];
      }
   }

   struct gemm_push_layout push = {
      .k = (int32_t)K,
      .n = (int32_t)N,
      .m = (int32_t)M,
      .bias_mode = bias_mode,
   };

   const uint32_t orig_grid[3] = {
      (N + 31u) / 32u,
      (M + 31u) / 32u,
      1u,
   };
   HK_EVAL_CHECK(hk_gemm_cmat_dispatch_dims_ok(opts->guarded, M, N, K, orig_grid));

   struct hk_eval_ssbo ssbos_orig[4] = {
      { 0, 0, a_buf, a_elems * sizeof(float), runtime_a_off },
      { 0, 1, b_buf, b_elems * sizeof(float), runtime_b_off },
      { 0, 2, bias_buf, bias_elems * sizeof(float), runtime_b_off },
      { 0, 3, c_orig, c_elems * sizeof(float), runtime_b_off },
   };
   hk_eval_dispatch_wg(orig, orig_grid, ssbos_orig, 4, &push, sizeof(push));

   if (epilogue) {
      float lo = hk_eval_u2f(epilogue->lower_bits);
      float hi = hk_eval_u2f(epilogue->upper_bits);
      for (size_t i = off_elems; i < c_elems; ++i) {
         float v = c_orig[i];
         if (epilogue->upper_first) {
            if (epilogue->has_upper) v = fminf(v, hi);
            if (epilogue->has_lower) v = fmaxf(v, lo);
         } else {
            if (epilogue->has_lower) v = fmaxf(v, lo);
            if (epilogue->has_upper) v = fminf(v, hi);
         }
         c_orig[i] = v;
      }
   }

   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const uint32_t alt_grid[3] = {
      (N + tile_n - 1u) / tile_n,
      (M + tile_m - 1u) / tile_m,
      1u,
   };
   struct hk_eval_ssbo ssbos_alt[4] = {
      { 0, 0, a_buf, a_elems * sizeof(float), runtime_a_off },
      { 0, 1, b_buf, b_elems * sizeof(float), runtime_b_off },
      { 0, 2, bias_buf, bias_elems * sizeof(float), runtime_b_off },
      { 0, 3, c_alt, c_elems * sizeof(float), runtime_b_off },
   };
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, 4, &push, sizeof(push));

   unsigned mismatches = 0;
   for (size_t i = 0; i < c_elems; ++i) {
      uint32_t uo = hk_eval_f2u(c_orig[i]);
      uint32_t ua = hk_eval_f2u(c_alt[i]);
      if (uo != ua) {
         if (!mismatches)
            fprintf(stderr,
                    "GEMM eval %s (M=%u N=%u K=%u bmode=%d tile_n=%u off=%u): "
                    "first idx=%zu orig=0x%08x (%g) alt=0x%08x (%g)\n",
                    force_uncertified ? "mismatch (uncertified, expected)" : "mismatch",
                    M, N, K, bias_mode, tile_n, opts->offset_B, i, uo, c_orig[i],
                    ua, c_alt[i]);
         mismatches++;
      }
   }

   free(a_buf);
   free(b_buf);
   free(bias_buf);
   free(c_orig);
   free(c_alt);
   ralloc_free(alt);
   ralloc_free(pre);
   ralloc_free(orig);
   return mismatches;
}

static void
eval_compare_gemm_case(const struct gemm_test_opts *opts, uint32_t M,
                       uint32_t N, uint32_t K, int32_t bias_mode,
                       size_t runtime_a_off, size_t runtime_b_off,
                       uint32_t tile_n,
                       const struct hk_nir_clamp_epilogue *epilogue,
                       uint32_t seed)
{
   HK_EVAL_CHECK(gemm_eval_mismatches(opts, M, N, K, bias_mode, runtime_a_off,
                                      runtime_b_off, tile_n, epilogue, seed,
                                      false) == 0);
}

static void
test_gemm_cmat_certify_and_build(void)
{
   const struct hk_nir_clamp_epilogue ep = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = 0x00000000u, .upper_bits = 0x40c00000u,
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };

   const struct gemm_test_opts pos_cases[] = {
      { .guarded = false, .has_bias_buf = true,  .has_bias_mode = true,  .offset_B = 0 },
      { .guarded = true,  .has_bias_buf = true,  .has_bias_mode = true,  .offset_B = 0 },
      { .guarded = true,  .has_bias_buf = true,  .has_bias_mode = true,  .offset_B = 4, .split_bias_stores = true },
      { .guarded = false, .has_bias_buf = true,  .has_bias_mode = true,  .offset_B = 16, .split_bias_stores = true },
      { .guarded = false, .has_bias_buf = false, .has_bias_mode = false, .offset_B = 0 },
      { .guarded = true,  .has_bias_buf = true,  .has_bias_mode = false, .offset_B = 16 },
      { .guarded = true,  .has_bias_buf = true,  .has_bias_mode = false, .offset_B = 4, .fixed_row_bias = true },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(pos_cases); ++i) {
      nir_shader *orig = build_tiled_gemm_nir_opts(&pos_cases[i]);

      nir_shader *g17p = nir_shader_clone(NULL, orig);
      run_hk_preprocess_for_test(g17p, false);
      struct hk_nir_gemm_output m_g17p = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         g17p, nir_address_format_64bit_global_32bit_offset, &m_g17p));
      HK_EVAL_CHECK(m_g17p.cmat_valid);
      HK_EVAL_CHECK(m_g17p.output.offset_B == pos_cases[i].offset_B);
      HK_EVAL_CHECK(m_g17p.reads[0].offset_B == pos_cases[i].offset_B);
      HK_EVAL_CHECK(m_g17p.reads[1].offset_B == pos_cases[i].offset_B);
      HK_EVAL_CHECK(m_g17p.desc_comps == 4);

      const uint32_t tile_ns[3] = { 16, 32, 64 };
      for (unsigned t = 0; t < ARRAY_SIZE(tile_ns); ++t) {
         const uint32_t tile_n = tile_ns[t];
         nir_shader *cmat = hk_nir_build_gemm_cmat(g17p, &m_g17p, tile_n, NULL, HK_COMPUTE_PAIR_DUAL_NONE);
         HK_EVAL_CHECK(cmat);
         nir_validate_shader(cmat, "gemm_cmat");
         HK_EVAL_CHECK(shader_has_cmat_muladd(cmat));
         ralloc_free(cmat);

         nir_shader *cmat_ep = hk_nir_build_gemm_cmat(g17p, &m_g17p, tile_n, &ep, HK_COMPUTE_PAIR_DUAL_NONE);
         HK_EVAL_CHECK(cmat_ep);
         nir_validate_shader(cmat_ep, "gemm_cmat_ep");
         HK_EVAL_CHECK(shader_has_cmat_muladd(cmat_ep));
         ralloc_free(cmat_ep);
      }
      ralloc_free(g17p);

      nir_shader *m5 = nir_shader_clone(NULL, orig);
      rewrite_ssbo_desc_comps(m5, 5);
      run_hk_preprocess_for_test(m5, true);
      struct hk_nir_gemm_output m_m5 = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         m5, nir_address_format_64bit_global_32bit_offset_map, &m_m5));
      HK_EVAL_CHECK(m_m5.cmat_valid);
      HK_EVAL_CHECK(m_m5.desc_comps == 5);
      nir_shader *cmat_m5 = hk_nir_build_gemm_cmat(m5, &m_m5, 64, &ep, HK_COMPUTE_PAIR_DUAL_NONE);
      HK_EVAL_CHECK(cmat_m5);
      nir_validate_shader(cmat_m5, "gemm_cmat_m5");
      HK_EVAL_CHECK(shader_has_cmat_muladd(cmat_m5));
      ralloc_free(cmat_m5);
      ralloc_free(m5);

      ralloc_free(orig);
   }

   /* Negative certifier cases: cmat_valid MUST be false (or unmatched). */
   const struct gemm_test_opts neg_cases[] = {
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .fp_math_ctrl = nir_fp_preserve_nan },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .preserve_nan = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .inner_bound = 31 },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .transpose_b = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .extra_store = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .early_k_break = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .early_dot_break = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .split_bias_stores = true, .swap_bias_arms = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .split_bias_stores = true, .bias_in_mode2 = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .split_bias_stores = true, .drop_mode0_bias = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .swap_bias_arms = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .bias_in_mode2 = true },
      { .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .drop_mode0_bias = true },
      { .guarded = false, .has_bias_buf = false, .has_bias_mode = false, .offset_B = 4 },
   };
   for (unsigned i = 0; i < ARRAY_SIZE(neg_cases); ++i) {
      nir_shader *bad = build_tiled_gemm_nir_opts(&neg_cases[i]);
      run_hk_preprocess_for_test(bad, false);
      struct hk_nir_gemm_output m_bad = {0};
      bool matched = hk_nir_match_gemm_output(
         bad, nir_address_format_64bit_global_32bit_offset, &m_bad);
      HK_EVAL_CHECK(!matched || !m_bad.cmat_valid);
      ralloc_free(bad);
   }

   /* Float controls mismatch between producer (mode 0) and clamp epilogue
    * MUST be refused for every cooperative-matrix tile width. */
   {
      const struct gemm_test_opts o_fc = {
         .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
      };
      nir_shader *fc_prod = build_tiled_gemm_nir_opts(&o_fc);
      run_hk_preprocess_for_test(fc_prod, false);
      struct hk_nir_gemm_output m_fc = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         fc_prod, nir_address_format_64bit_global_32bit_offset, &m_fc));
      HK_EVAL_CHECK(m_fc.cmat_valid);
      const uint32_t bad_modes[] = {
         FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32,
         FLOAT_CONTROLS_DENORM_PRESERVE_FP32,
      };
      const uint32_t tiles[] = { 16, 32, 64 };
      for (unsigned mi = 0; mi < ARRAY_SIZE(bad_modes); ++mi) {
         struct hk_nir_clamp_epilogue bad_ep = ep;
         bad_ep.float_controls_execution_mode = bad_modes[mi];
         for (unsigned ti = 0; ti < ARRAY_SIZE(tiles); ++ti)
            HK_EVAL_CHECK(hk_nir_build_gemm_cmat(fc_prod, &m_fc, tiles[ti], &bad_ep, HK_COMPUTE_PAIR_DUAL_NONE) == NULL);
      }
      ralloc_free(fc_prod);
   }

   /* Dispatch dimension gate: unguarded K = 16 mod 32 MUST be refused. */
   const uint32_t g64x64[3] = { 2, 2, 1 };
   const uint32_t g32x64[3] = { 1, 2, 1 };
   HK_EVAL_CHECK(!hk_gemm_cmat_dispatch_dims_ok(false, 64, 64, 48, g64x64));
   HK_EVAL_CHECK(!hk_gemm_cmat_dispatch_dims_ok(false, 64, 64, 16, g64x64));
   HK_EVAL_CHECK( hk_gemm_cmat_dispatch_dims_ok(false, 64, 64, 32, g64x64));
   HK_EVAL_CHECK( hk_gemm_cmat_dispatch_dims_ok(false, 64, 32, 64, g32x64));
   HK_EVAL_CHECK( hk_gemm_cmat_dispatch_dims_ok(true,  64, 64, 48, g64x64));
   HK_EVAL_CHECK( hk_gemm_cmat_dispatch_dims_ok(true,  37, 35, 19, g64x64));

   /* Original vs alternate bit-exact NIR evaluation on random FP32 inputs. */
   const struct gemm_test_opts o_guarded_dyn = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct gemm_test_opts o_guarded_split_off4 = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true,
      .offset_B = 4, .use_ffma = true, .split_bias_stores = true,
   };
   const struct gemm_test_opts o_unguarded_dyn = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true,
      .offset_B = 16, .use_ffma = true,
   };
   const struct gemm_test_opts o_unguarded_split = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true,
      .use_ffma = true, .split_bias_stores = true,
   };
   const struct gemm_test_opts o_guarded_fixed_row = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = false,
      .use_ffma = true, .fixed_row_bias = true,
   };
   const struct gemm_test_opts o_guarded_fixed_col = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = false,
      .use_ffma = true,
   };
   const struct gemm_test_opts o_guarded_nobias = {
      .guarded = true, .has_bias_buf = false, .has_bias_mode = false,
      .use_ffma = true,
   };

   for (int bm = 0; bm <= 2; ++bm) {
      eval_compare_gemm_case(&o_guarded_dyn, 37, 35, 19, bm, 4, 4, 32, NULL, 100u + bm);
      eval_compare_gemm_case(&o_guarded_dyn, 67, 69, 48, bm, 4, 4, 64, &ep,  200u + bm);
      eval_compare_gemm_case(&o_guarded_split_off4, 33, 41, 20, bm, 4, 4, 64, NULL, 300u + bm);
   }
   eval_compare_gemm_case(&o_guarded_split_off4, 71, 29, 48, 0, 4, 4, 32, &ep, 350u);

   for (int bm = 0; bm <= 2; ++bm) {
      eval_compare_gemm_case(&o_unguarded_dyn, 64, 64, 32, bm, 0, 4, 64, NULL, 400u + bm);
      eval_compare_gemm_case(&o_unguarded_split, 64, 32, 64, bm, 0, 4, 32, &ep, 500u + bm);
   }

   eval_compare_gemm_case(&o_guarded_fixed_row, 23, 39, 33, 0, 4, 4, 64, &ep,  600u);
   eval_compare_gemm_case(&o_guarded_fixed_col, 23, 39, 33, 1, 4, 4, 32, NULL, 601u);
   eval_compare_gemm_case(&o_guarded_nobias,    19, 25, 17, 2, 4, 4, 32, &ep,  602u);

   /* cmat16 (64x16 K_step=16) bit-exact coverage: narrow N<=16, non-mult4 K,
    * split bias, unguarded, fixed bias, and fused clamp epilogue. */
   for (int bm = 0; bm <= 2; ++bm) {
      eval_compare_gemm_case(&o_guarded_dyn,        37, 16, 27, bm, 4, 4, 16, NULL, 700u + bm);
      eval_compare_gemm_case(&o_guarded_dyn,        65, 13, 48, bm, 4, 4, 16, &ep,  710u + bm);
      eval_compare_gemm_case(&o_guarded_split_off4, 33, 16, 16, bm, 4, 4, 16, &ep,  720u + bm);
   }
   eval_compare_gemm_case(&o_unguarded_dyn,   64, 32, 64, 1, 0, 4, 16, &ep,  730u);
   eval_compare_gemm_case(&o_guarded_fixed_row, 23, 11, 33, 0, 4, 4, 16, &ep, 731u);
   eval_compare_gemm_case(&o_guarded_nobias,    19,  8, 17, 2, 4, 4, 16, &ep, 732u);
   /* cmat32 (64x32 K_step=16) on N=96 and cmat64 (32x64 K_step=32) on N=144. */
   eval_compare_gemm_case(&o_guarded_dyn, 65,  96, 48, 1, 4, 4, 32, &ep,  740u);
   eval_compare_gemm_case(&o_guarded_dyn, 35, 144, 27, 1, 4, 4, 64, NULL, 741u);

   /* Shapes the certifier must keep accepting: shared stores directly in the
    * guard arms, split stores under an if/else-if bias_mode chain, and the
    * unrolled dot product. */
   const struct gemm_test_opts o_a_store_in_arms = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true,
      .use_ffma = true, .a_store_in_arms = true,
   };
   const struct gemm_test_opts o_nested_chain = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true,
      .use_ffma = true, .split_bias_stores = true, .nested_bias_chain = true,
   };
   const struct gemm_test_opts o_unroll_dot = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true,
      .use_ffma = true, .unroll_dot = true,
   };
   for (int bm = 0; bm <= 2; ++bm) {
      eval_compare_gemm_case(&o_a_store_in_arms, 37, 35, 19, bm, 4, 4, 32, NULL, 700u + bm);
      eval_compare_gemm_case(&o_nested_chain,    33, 41, 20, bm, 4, 4, 64, &ep,  710u + bm);
      eval_compare_gemm_case(&o_unroll_dot,      37, 35, 48, bm, 4, 4, 32, NULL, 720u + bm);
   }

   /* Control-flow negative controls. Each shape matches as a GEMM but adds
    * a predicate (or a read order) the cmat alternate does not reproduce, so
    * it must not certify; built anyway, its alternate differs from the
    * original on M=64 N=32 K=32 all-1.0f inputs with C pre-filled to -99.0f.
    * Each is paired with the same shape without the defect, which certifies
    * and matches bit-exactly on the same inputs. */
   const struct {
      const char *name;
      struct gemm_test_opts bad;
      int32_t bias_mode;
      unsigned expect; /* forced-alternate mismatches; 0 = any nonzero */
   } cf_cases[] = {
      /* s_a store inside `if (a_row != 0)` in the guard's then-arm: matrix
       * row 0 stays 0.0 in the original, 32.0 in the alternate. */
      { "shared store in inner if",
        { .guarded = true, .use_ffma = true, .a_store_in_arms = true,
          .a_store_inner_if = true }, 2, 32 },
      /* C stores under `if (col != 0)`: the original keeps column 0 at
       * -99.0, the alternate writes all 64 rows of it. */
      { "C stores in outer if, guarded",
        { .guarded = true, .use_ffma = true, .c_store_outer_if = true }, 2, 64 },
      { "C stores in outer if, unguarded",
        { .guarded = false, .use_ffma = true, .c_store_outer_if = true }, 2, 64 },
      { "split C stores in outer if",
        { .guarded = true, .has_bias_buf = true, .has_bias_mode = true,
          .use_ffma = true, .split_bias_stores = true,
          .c_store_outer_if = true }, 0, 64 },
      /* Unrolled dot product reading the tile before barrier_1. */
      { "tile read before barrier",
        { .guarded = true, .use_ffma = true, .unroll_dot = true,
          .unroll_early_loads = true }, 2, 0 },
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cf_cases); ++i) {
      struct gemm_test_opts good = cf_cases[i].bad;
      good.a_store_inner_if = false;
      good.c_store_outer_if = false;
      good.unroll_early_loads = false;
      HK_EVAL_CHECK(gemm_eval_mismatches(&good, 64, 32, 32,
                                         cf_cases[i].bias_mode, 0, 0, 32,
                                         NULL, 0, false) == 0);
      const unsigned diff =
         gemm_eval_mismatches(&cf_cases[i].bad, 64, 32, 32,
                              cf_cases[i].bias_mode, 0, 0, 32, NULL, 0, true);
      printf("gemm cmat negative control '%s': not certified; forced "
             "alternate differs in %u outputs\n", cf_cases[i].name, diff);
      HK_EVAL_CHECK(cf_cases[i].expect ? diff == cf_cases[i].expect : diff > 0);
   }
}

static void
eval_compare_dw_dual_copy(const struct dw_push_layout *cfg,
                          const struct hk_nir_clamp_epilogue *ep)
{
   nir_shader *orig = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
   struct hk_nir_depthwise_output info = {0};
   HK_EVAL_CHECK(hk_nir_match_depthwise_output(orig, nir_address_format_64bit_global_32bit_offset, &info));
   HK_EVAL_CHECK(info.vec4_valid);

   /* Negative check: DUAL_PAD4 is not supported on depthwise_vec4. */
   HK_EVAL_CHECK(hk_nir_build_depthwise_vec4(orig, &info, ep, HK_COMPUTE_PAIR_DUAL_PAD4) == NULL);

   nir_shader *dcopy = hk_nir_build_depthwise_vec4(orig, &info, ep, HK_COMPUTE_PAIR_DUAL_COPY);
   HK_EVAL_CHECK(dcopy);
   nir_validate_shader(dcopy, "depthwise_vec4_dcopy");

   const size_t in_n = (size_t)cfg->in_w * cfg->in_h * cfg->channels;
   const size_t wt_n = (size_t)cfg->ker_w * cfg->ker_h * cfg->channels;
   const size_t bs_n = (size_t)cfg->channels;
   const size_t out_n = (size_t)cfg->out_w * cfg->out_h * cfg->channels;

   float *in_buf = calloc(in_n, sizeof(float));
   float *wt_buf = calloc(wt_n, sizeof(float));
   float *bs_buf = calloc(bs_n, sizeof(float));
   float *ref_out = calloc(out_n, sizeof(float));
   float *c_out = calloc(out_n, sizeof(float));
   float *dst2_out = calloc(out_n + 4, sizeof(float));
   HK_EVAL_CHECK(in_buf && wt_buf && bs_buf && ref_out && c_out && dst2_out);

   for (size_t i = 0; i < in_n; ++i)
      in_buf[i] = ((int)((i * 7u + 3u) % 13u) - 6) * 0.25f;
   for (size_t i = 0; i < wt_n; ++i)
      wt_buf[i] = ((int)((i * 5u + 1u) % 9u) - 4) * 0.5f;
   for (size_t i = 0; i < bs_n; ++i)
      bs_buf[i] = ((int)(i % 7u) - 3) * 0.75f;
   for (size_t i = 0; i < out_n + 4; ++i)
      dst2_out[i] = -77.5f;

   struct hk_eval_ssbo ssbos_ref[4] = {
      { 0, 0, in_buf, in_n * sizeof(float) },
      { 0, 1, wt_buf, wt_n * sizeof(float) },
      { 0, 2, bs_buf, bs_n * sizeof(float) },
      { 0, 3, ref_out, out_n * sizeof(float) },
   };
   const uint32_t orig_grid[3] = {
      DIV_ROUND_UP((uint32_t)cfg->out_w * cfg->channels, orig->info.workgroup_size[0]),
      DIV_ROUND_UP((uint32_t)cfg->out_h, orig->info.workgroup_size[1]),
      1,
   };
   hk_eval_dispatch(orig, orig_grid, ssbos_ref, 4, cfg, sizeof(*cfg));
   if (ep) {
      nir_shader *clamp = build_clamp_nir(0, 3, 3, 0, ep, 64);
      int32_t count = (int32_t)out_n;
      const uint32_t c_grid[3] = { DIV_ROUND_UP((uint32_t)out_n, 64u), 1, 1 };
      hk_eval_dispatch(clamp, c_grid, ssbos_ref, 4, &count, sizeof(count));
      ralloc_free(clamp);
   }

   uint8_t push_buf[256] = {0};
   memcpy(push_buf, cfg, sizeof(*cfg));
   const uint32_t dual_off = hk_dual_write_push_offset(info.push_bytes);
   const uint64_t dst2_va = hk_eval_make_vaddr(4, 16); /* +16B non-zero offset */
   struct hk_dual_copy_push dp = {
      .dst2_va_lo = (uint32_t)dst2_va,
      .dst2_va_hi = (uint32_t)(dst2_va >> 32),
   };
   memcpy(push_buf + dual_off, &dp, sizeof(dp));

   struct hk_eval_ssbo ssbos_alt[5] = {
      { 0, 0, in_buf, in_n * sizeof(float) },
      { 0, 1, wt_buf, wt_n * sizeof(float) },
      { 0, 2, bs_buf, bs_n * sizeof(float) },
      { 0, 3, c_out, out_n * sizeof(float) },
      { 0, 9, dst2_out, (out_n + 4) * sizeof(float) },
   };
   const uint32_t out_y_units = (uint32_t)cfg->out_h;
   const uint32_t v4_grid[3] = {
      DIV_ROUND_UP((uint32_t)cfg->channels / 4u, dcopy->info.workgroup_size[0]),
      DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)cfg->out_w, 2u), dcopy->info.workgroup_size[1]),
      DIV_ROUND_UP(out_y_units, dcopy->info.workgroup_size[2]),
   };
   hk_eval_dispatch(dcopy, v4_grid, ssbos_alt, 5, push_buf,
                    dual_off + sizeof(dp));

   for (size_t i = 0; i < 4; ++i)
      HK_EVAL_CHECK(hk_eval_f2u(dst2_out[i]) == hk_eval_f2u(-77.5f));
   for (size_t i = 0; i < out_n; ++i) {
      HK_EVAL_CHECK(hk_eval_f2u(c_out[i]) == hk_eval_f2u(ref_out[i]));
      HK_EVAL_CHECK(hk_eval_f2u(dst2_out[i + 4]) == hk_eval_f2u(ref_out[i]));
   }

   free(in_buf); free(wt_buf); free(bs_buf);
   free(ref_out); free(c_out); free(dst2_out);
   ralloc_free(dcopy);
   ralloc_free(orig);
}

static void
eval_compare_gemm_dual_copy(const struct gemm_test_opts *opts,
                            uint32_t M, uint32_t N, uint32_t K,
                            int32_t bias_mode, uint32_t tile_n,
                            const struct hk_nir_clamp_epilogue *epilogue,
                            uint32_t seed)
{
   nir_shader *orig = build_tiled_gemm_nir_opts(opts);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);

   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   HK_EVAL_CHECK(m.cmat_valid);

   nir_shader *alt = hk_nir_build_gemm_cmat(pre, &m, tile_n, epilogue,
                                            HK_COMPUTE_PAIR_DUAL_COPY);
   HK_EVAL_CHECK(alt);
   nir_validate_shader(alt, "gemm_cmat_dcopy");

   const uint32_t off_elems = opts->offset_B / 4u;
   const size_t a_elems = (size_t)M * K + off_elems;
   const size_t b_elems = (size_t)K * N + off_elems;
   const size_t bias_elems = (size_t)(M > N ? M : N) + off_elems + 1;
   const size_t c_elems = (size_t)M * N + off_elems;
   const size_t d_elems = (size_t)M * N + 4;

   float *a_buf = calloc(a_elems, sizeof(float));
   float *b_buf = calloc(b_elems, sizeof(float));
   float *bias_buf = calloc(bias_elems, sizeof(float));
   float *c_orig = calloc(c_elems, sizeof(float));
   float *c_alt = calloc(c_elems, sizeof(float));
   float *dst2 = calloc(d_elems, sizeof(float));
   HK_EVAL_CHECK(a_buf && b_buf && bias_buf && c_orig && c_alt && dst2);

   uint32_t rng = seed;
   for (size_t i = off_elems; i < a_elems; ++i) a_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = off_elems; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = off_elems; i < bias_elems; ++i) bias_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = 0; i < d_elems; ++i) dst2[i] = -55.25f;

   struct gemm_push_layout push = {
      .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = bias_mode,
   };
   const uint32_t orig_grid[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
   struct hk_eval_ssbo ssbos_orig[4] = {
      { 0, 0, a_buf, a_elems * sizeof(float), 0 },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_orig, c_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(orig, orig_grid, ssbos_orig, 4, &push, sizeof(push));
   if (epilogue) {
      float lo = hk_eval_u2f(epilogue->lower_bits);
      float hi = hk_eval_u2f(epilogue->upper_bits);
      for (size_t i = off_elems; i < c_elems; ++i) {
         float v = c_orig[i];
         if (epilogue->has_lower) v = fmaxf(v, lo);
         if (epilogue->has_upper) v = fminf(v, hi);
         c_orig[i] = v;
      }
   }

   uint8_t push_buf[256] = {0};
   memcpy(push_buf, &push, sizeof(push));
   const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
   const uint64_t dst2_va = hk_eval_make_vaddr(4, 16);
   struct hk_dual_copy_push dp = {
      .dst2_va_lo = (uint32_t)dst2_va,
      .dst2_va_hi = (uint32_t)(dst2_va >> 32),
   };
   memcpy(push_buf + dual_off, &dp, sizeof(dp));

   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const uint32_t alt_grid[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, tile_m), 1u };
   struct hk_eval_ssbo ssbos_alt[5] = {
      { 0, 0, a_buf, a_elems * sizeof(float), 0 },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_alt, c_elems * sizeof(float), 0 },
      { 0, 9, dst2, d_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, 5, push_buf, dual_off + sizeof(dp));

   for (size_t i = 0; i < 4; ++i)
      HK_EVAL_CHECK(hk_eval_f2u(dst2[i]) == hk_eval_f2u(-55.25f));
   for (size_t i = 0; i < (size_t)M * N; ++i) {
      HK_EVAL_CHECK(hk_eval_f2u(c_alt[off_elems + i]) == hk_eval_f2u(c_orig[off_elems + i]));
      HK_EVAL_CHECK(hk_eval_f2u(dst2[4 + i]) == hk_eval_f2u(c_orig[off_elems + i]));
   }

   free(a_buf); free(b_buf); free(bias_buf);
   free(c_orig); free(c_alt); free(dst2);
   ralloc_free(alt); ralloc_free(pre); ralloc_free(orig);
}

static void
eval_compare_gemm_dual_pad4(const struct gemm_test_opts *opts,
                            uint32_t in_h, uint32_t in_w, uint32_t out_h, uint32_t out_w,
                            uint32_t ch, uint32_t K, int32_t bias_mode,
                            uint32_t tile_n, const int32_t pads[4],
                            const struct hk_nir_clamp_epilogue *epilogue,
                            uint32_t seed)
{
   const uint32_t M = in_h * in_w;
   const uint32_t N = ch;
   nir_shader *orig = build_tiled_gemm_nir_opts(opts);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);

   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   HK_EVAL_CHECK(m.cmat_valid);

   nir_shader *alt = hk_nir_build_gemm_cmat(pre, &m, tile_n, epilogue,
                                            HK_COMPUTE_PAIR_DUAL_PAD4);
   HK_EVAL_CHECK(alt);
   nir_validate_shader(alt, "gemm_cmat_dpad4");

   const uint32_t off_elems = opts->offset_B / 4u;
   const size_t a_elems = (size_t)M * K + off_elems;
   const size_t b_elems = (size_t)K * N + off_elems;
   const size_t bias_elems = (size_t)(M > N ? M : N) + off_elems + 1;
   const size_t c_elems = (size_t)M * N + off_elems;
   const size_t d_elems = (size_t)out_h * out_w * N;

   float *a_buf = calloc(a_elems, sizeof(float));
   float *b_buf = calloc(b_elems, sizeof(float));
   float *bias_buf = calloc(bias_elems, sizeof(float));
   float *c_orig = calloc(c_elems, sizeof(float));
   float *c_alt = calloc(c_elems, sizeof(float));
   float *d_ref = calloc(d_elems, sizeof(float));
   float *d_alt = calloc(d_elems, sizeof(float));
   int32_t pad_buf[8] = { pads[0], 0, pads[1], 0, pads[2], 0, pads[3], 0 };
   HK_EVAL_CHECK(a_buf && b_buf && bias_buf && c_orig && c_alt && d_ref && d_alt);

   uint32_t rng = seed;
   for (size_t i = off_elems; i < a_elems; ++i) a_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = off_elems; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = off_elems; i < bias_elems; ++i) bias_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = 0; i < d_elems; ++i) d_alt[i] = -99.5f; /* sentinel: must be overwritten */

   struct gemm_push_layout push = {
      .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = bias_mode,
   };
   const uint32_t orig_grid[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
   struct hk_eval_ssbo ssbos_orig[4] = {
      { 0, 0, a_buf, a_elems * sizeof(float), 0 },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_orig, c_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(orig, orig_grid, ssbos_orig, 4, &push, sizeof(push));
   if (epilogue) {
      float lo = hk_eval_u2f(epilogue->lower_bits);
      float hi = hk_eval_u2f(epilogue->upper_bits);
      for (size_t i = off_elems; i < c_elems; ++i) {
         float v = c_orig[i];
         if (epilogue->has_lower) v = fmaxf(v, lo);
         if (epilogue->has_upper) v = fminf(v, hi);
         c_orig[i] = v;
      }
   }

   /* Exact 64-bit reference pad4d. */
   for (uint32_t oy = 0; oy < out_h; ++oy) {
      for (uint32_t ox = 0; ox < out_w; ++ox) {
         for (uint32_t oc = 0; oc < N; ++oc) {
            int64_t in_n = -(int64_t)pads[0];
            int64_t iy = (int64_t)oy - (int64_t)pads[1];
            int64_t ix = (int64_t)ox - (int64_t)pads[2];
            int64_t ic = (int64_t)oc - (int64_t)pads[3];
            size_t di = ((size_t)oy * out_w + ox) * N + oc;
            if (in_n == 0 && iy >= 0 && iy < (int64_t)in_h &&
                ix >= 0 && ix < (int64_t)in_w && ic >= 0 && ic < (int64_t)N) {
               size_t si = ((size_t)iy * in_w + (size_t)ix) * N + (size_t)ic;
               d_ref[di] = c_orig[off_elems + si];
            } else {
               d_ref[di] = 0.0f;
            }
         }
      }
   }

   uint8_t push_buf[256] = {0};
   memcpy(push_buf, &push, sizeof(push));
   const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
   const uint64_t dst2_va = hk_eval_make_vaddr(4, 0);
   const uint64_t pad_va = hk_eval_make_vaddr(5, 0);
   struct hk_dual_pad4_push dp = {
      .dst2_va_lo = (uint32_t)dst2_va,
      .dst2_va_hi = (uint32_t)(dst2_va >> 32),
      .pad_va_lo = (uint32_t)pad_va,
      .pad_va_hi = (uint32_t)(pad_va >> 32),
      .in_h = in_h,
      .in_w = in_w,
      .out_h = out_h,
      .out_w = out_w,
      .pad_idx = { 0, 2, 4, 6 },
   };
   memcpy(push_buf + dual_off, &dp, sizeof(dp));

   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const uint32_t alt_grid[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, tile_m), 1u };
   struct hk_eval_ssbo ssbos_alt[6] = {
      { 0, 0, a_buf, a_elems * sizeof(float), 0 },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_alt, c_elems * sizeof(float), 0 },
      { 0, 9, d_alt, d_elems * sizeof(float), 0 },
      { 0, 10, pad_buf, sizeof(pad_buf), 0 },
   };
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, 6, push_buf, dual_off + sizeof(dp));

   for (size_t i = 0; i < (size_t)M * N; ++i)
      HK_EVAL_CHECK(hk_eval_f2u(c_alt[off_elems + i]) == hk_eval_f2u(c_orig[off_elems + i]));
   for (size_t i = 0; i < d_elems; ++i)
      HK_EVAL_CHECK(hk_eval_f2u(d_alt[i]) == hk_eval_f2u(d_ref[i]));

   /* Verify HK_DUAL_WRITE_FLAG_SKIP_C skips C when simple_pad holds, while
    * preserving C writeback (and exact d_alt) on the non-simple fallback. */
   const bool simple_pad =
      pads[0] == 0 && pads[3] == 0 && pads[1] >= 0 && pads[2] >= 0 &&
      out_h >= in_h && (out_h - in_h) >= (uint32_t)pads[1] &&
      out_w >= in_w && (out_w - in_w) >= (uint32_t)pads[2];
   for (size_t i = 0; i < (size_t)M * N; ++i)
      c_alt[off_elems + i] = -88.5f;
   for (size_t i = 0; i < d_elems; ++i)
      d_alt[i] = -99.5f;
   dp.flags = HK_DUAL_WRITE_FLAG_SKIP_C;
   memcpy(push_buf + dual_off, &dp, sizeof(dp));
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, 6, push_buf, dual_off + sizeof(dp));
   for (size_t i = 0; i < (size_t)M * N; ++i) {
      if (simple_pad)
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[off_elems + i]) == hk_eval_f2u(-88.5f));
      else
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[off_elems + i]) == hk_eval_f2u(c_orig[off_elems + i]));
   }
   for (size_t i = 0; i < d_elems; ++i)
      HK_EVAL_CHECK(hk_eval_f2u(d_alt[i]) == hk_eval_f2u(d_ref[i]));

   free(a_buf); free(b_buf); free(bias_buf);
   free(c_orig); free(c_alt); free(d_ref); free(d_alt);
   ralloc_free(alt); ralloc_free(pre); ralloc_free(orig);
}

static void
eval_compare_gemm_cpu_and_guarded_specialize(uint32_t M, uint32_t N, uint32_t K,
                                             int32_t bias_mode, uint32_t tile_n,
                                             bool expect_unguarded, uint32_t a_offset_B,
                                             const struct hk_nir_clamp_epilogue *epilogue,
                                             uint32_t seed)
{
   const struct gemm_test_opts opts = {
      .a_extra_offset_B = a_offset_B,
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   nir_shader *orig = build_tiled_gemm_nir_opts(&opts);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);

   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   HK_EVAL_CHECK(m.cmat_valid && m.is_guarded);

   /* The evaluator gets a pointer to the descriptor range and uses
    * base_offset separately to check the alignment of memory accesses. */
   const uint32_t descriptor_offset_B = a_offset_B ? 32 : 0;
   HK_EVAL_CHECK(m.reads[0].offset_B == a_offset_B);
   const bool unguarded = hk_gemm_cmat_can_specialize_unguarded(
      m.is_guarded, M, N, K, tile_n, m.reads[0].offset_B,
      descriptor_offset_B, descriptor_offset_B);
   HK_EVAL_CHECK(unguarded == expect_unguarded);
   struct hk_nir_gemm_output build_info = m;
   build_info.is_guarded = !unguarded;

   nir_shader *alt = hk_nir_build_gemm_cmat(
      pre, &build_info, tile_n, epilogue, HK_COMPUTE_PAIR_DUAL_NONE);
   HK_EVAL_CHECK(alt != NULL);
   nir_validate_shader(alt, "gemm_cmat_specialize_or_sweep");

   const size_t a_elems = (size_t)M * K;
   const size_t a_range = a_offset_B + a_elems * sizeof(float);
   const size_t b_elems = (size_t)K * N;
   const size_t bias_elems = (size_t)(M > N ? M : N);
   const size_t c_elems = (size_t)M * N;
   float *a_buf = calloc(1, descriptor_offset_B + a_range);
   float *b_buf = calloc(b_elems, sizeof(float));
   float *bias_buf = calloc(bias_elems, sizeof(float));
   float *c_orig = calloc(c_elems, sizeof(float));
   float *c_alt = calloc(c_elems, sizeof(float));
   float *c_cpu = calloc(c_elems, sizeof(float));
   HK_EVAL_CHECK(a_buf && b_buf && bias_buf && c_orig && c_alt && c_cpu);

   float *a_descriptor = a_buf + descriptor_offset_B / sizeof(float);
   float *a_matrix = a_descriptor + a_offset_B / sizeof(float);
   uint32_t rng = seed;
   for (size_t i = 0; i < a_elems; ++i)
      a_matrix[i] = gemm_test_rand_f32(&rng);
   for (size_t i = 0; i < b_elems; ++i)
      b_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = 0; i < bias_elems; ++i)
      bias_buf[i] = gemm_test_rand_f32(&rng);

   /* Independent CPU FP32 reference matching exact K-order FMA + bias + clamp. */
   for (uint32_t r = 0; r < M; ++r) {
      for (uint32_t c = 0; c < N; ++c) {
         float acc = 0.0f;
         for (uint32_t k = 0; k < K; ++k)
            acc = fmaf(a_matrix[(size_t)r * K + k], b_buf[(size_t)k * N + c], acc);
         if (bias_mode == 0)
            acc += bias_buf[r];
         else if (bias_mode == 1)
            acc += bias_buf[c];
         if (epilogue) {
            float lo = hk_eval_u2f(epilogue->lower_bits);
            float hi = hk_eval_u2f(epilogue->upper_bits);
            if (epilogue->has_lower) acc = fmaxf(acc, lo);
            if (epilogue->has_upper) acc = fminf(acc, hi);
         }
         c_cpu[(size_t)r * N + c] = acc;
      }
   }

   struct gemm_push_layout push = {
      .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = bias_mode,
   };
   const uint32_t orig_grid[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
   struct hk_eval_ssbo ssbos_orig[4] = {
      { 0, 0, a_descriptor, a_range, descriptor_offset_B },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_orig, c_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(orig, orig_grid, ssbos_orig, 4, &push, sizeof(push));
   if (epilogue) {
      float lo = hk_eval_u2f(epilogue->lower_bits);
      float hi = hk_eval_u2f(epilogue->upper_bits);
      for (size_t i = 0; i < c_elems; ++i) {
         float v = c_orig[i];
         if (epilogue->has_lower) v = fmaxf(v, lo);
         if (epilogue->has_upper) v = fminf(v, hi);
         c_orig[i] = v;
      }
   }

   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const uint32_t alt_grid[3] = {
      (N + tile_n - 1u) / tile_n, (M + tile_m - 1u) / tile_m, 1u,
   };
   struct hk_eval_ssbo ssbos_alt[4] = {
      { 0, 0, a_descriptor, a_range, descriptor_offset_B },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_alt, c_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, 4, &push, sizeof(push));

   for (size_t i = 0; i < c_elems; ++i) {
      HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_orig[i]));
      HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_cpu[i]));
   }

   free(a_buf); free(b_buf); free(bias_buf);
   free(c_orig); free(c_alt); free(c_cpu);
   ralloc_free(alt); ralloc_free(pre); ralloc_free(orig);
}

static void
test_gemm_cmat_guarded_specialize_and_shape_sweep(void)
{
   const struct hk_nir_clamp_epilogue ep = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = 0x00000000u, .upper_bits = 0x40c00000u,
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };

   /* Tile selection unit coverage. */
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 1,   1,   true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 16,  64,  true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 17,  64,  true, true, true) == 32);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 32,  64,  true, true, true) == 32);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 33,  64,  true, true, true) == 64);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 64,  31,  true, true, true) == 32);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 64,  32,  true, true, true) == 64);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 96,  64,  true, true, true) == 32);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 128, 32,  true, true, true) == 64);
   /* Small-grid high-K selection favors cmat16 for higher workgroup occupancy. */
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(400,  32,  576, true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(400,  64,  576, true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(800,  64,  128, true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(800,  64,  96,  true, true, true) == 64);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(864,  64,  128, true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(928,  64,  576, true, true, true) == 64);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(1600, 32,  576, true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(1600, 64,  576, true, true, true) == 64);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(400,  32,  32,  true, true, true) == 32);
   /* Exact N multiples pick the widest exact tile; others fall back to
    * even-ceil(N/32) cmat64 or cmat32. */
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 24,  64,  true, true, true) == 32);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 48,  32,  true, true, true) == 16);
   HK_EVAL_CHECK(hk_gemm_cmat_select_tile_n(4096, 144, 24,  true, true, true) == 32);

   /* Guarded producer -> *_UNGUARDED specialization predicate and NIR-eval at
    * exact multiples (including K%8==0 tails) and multiple-minus-1 across all tiles. */
   const uint32_t tiles[3] = { 16, 32, 64 };
   for (unsigned ti = 0; ti < ARRAY_SIZE(tiles); ++ti) {
      const uint32_t tn = tiles[ti];
      const uint32_t tm = hk_gemm_cmat_tile_m(tn);
      const uint32_t ks = hk_gemm_cmat_k_step(tn);

      /* Exact multiples (including K=8, 16, 24) specialize; misaligned A and non-guarded refuse. */
      HK_EVAL_CHECK( hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     8,      tn, 0, 0, 0));
      HK_EVAL_CHECK( hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     16,     tn, 0, 0, 0));
      HK_EVAL_CHECK( hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     ks,     tn, 0, 0, 0));
      HK_EVAL_CHECK( hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     24,     tn, 0, 0, 0));
      HK_EVAL_CHECK( hk_gemm_cmat_can_specialize_unguarded(true,  2 * tm, 2 * tn, 2 * ks, tn, 16, 64, 32));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(false, tm,     tn,     ks,     tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     ks,     tn, 4, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     ks,     tn, 0, 4, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true,  tm,     tn,     ks,     tn, 0, 0, 4));

      /* Multiple-minus-1 on M, N, or K MUST refuse unguarded specialization. */
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, tm - 1, tn,     ks,     tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, tm,     tn - 1, ks,     tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, tm,     tn,     ks - 1, tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, tm,     tn,     7,      tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, tm,     tn,     12,     tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, 2 * tm - 1, 2 * tn,     2 * ks,     tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, 2 * tm,     2 * tn - 1, 2 * ks,     tn, 0, 0, 0));
      HK_EVAL_CHECK(!hk_gemm_cmat_can_specialize_unguarded(true, 2 * tm,     2 * tn,     2 * ks - 1, tn, 0, 0, 0));

      /* Guarded producer specialized to *_UNGUARDED at exact multiples (incl. K=8, 16, 24 tails). */
      eval_compare_gemm_cpu_and_guarded_specialize(tm, tn, 8,  1, tn, true, 0, NULL, 1090u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(tm, tn, 16, 0, tn, true, 0, NULL, 1092u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(tm, tn, 24, 1, tn, true, 0, &ep,  1091u + ti * 10u);
      /* Guarded producer specialized to *_UNGUARDED at exact multiples. */
      eval_compare_gemm_cpu_and_guarded_specialize(tm, tn, ks, 1, tn, true, 0, NULL, 1100u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(tm, tn, ks, 0, tn, true, 0, &ep,  1101u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(2 * tm, 2 * tn, 2 * ks, 2, tn, true, 0, &ep, 1102u + ti * 10u);

      /* Guarded fallback at multiple-minus-1 for M, N, and K. */
      eval_compare_gemm_cpu_and_guarded_specialize(tm - 1, tn,     ks,     1, tn, false, 0, &ep,  1103u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(tm,     tn - 1, ks,     0, tn, false, 0, NULL, 1104u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(tm,     tn,     ks - 1, 1, tn, false, 0, &ep,  1105u + ti * 10u);
      /* Nonzero A descriptor and member offsets on either side of a K tile. */
      eval_compare_gemm_cpu_and_guarded_specialize(
         tm, tn, ks, 1, tn, true, 16, NULL, 1106u + ti * 10u);
      eval_compare_gemm_cpu_and_guarded_specialize(
         tm, tn, ks - 1, 1, tn, false, 16, NULL, 1107u + ti * 10u);
   }

   /* Broad shape sweep against CPU FP32 reference: odd N, K%4!=0, K<32, edge M. */
   const struct {
      uint32_t m, n, k;
      int32_t bias_mode;
      bool clamp;
   } sweep[] = {
      {  1,  1,  1, 1, false },
      {  1, 15,  3, 0, true  },
      {  7,  7,  5, 2, false },
      { 15, 17, 13, 1, true  },
      { 31, 31, 15, 0, false },
      { 32, 33, 16, 1, true  },
      { 33, 49, 27, 1, false },
      { 63, 63, 31, 0, true  },
      { 64, 65, 19, 2, false },
      { 65, 95, 35, 1, true  },
      { 32, 64, 24, 1, false },
      { 64, 96, 32, 0, true  },
      /* Per-8-col guarded fast paths: N=24/20/48/144/8/4, K=24/27, partial M. */
      { 70, 24, 32, 1, true  },
      { 70, 20, 24, 1, false },
      { 37, 48, 32, 1, true  },
      { 64, 48, 27, 0, false },
      { 70, 144, 24, 1, true },
      { 37,  8, 24, 1, true  },
      { 37,  4, 27, 0, false },
      { 65, 32, 27, 1, true  },
      { 64, 16, 48, 1, true  },
   };
   for (unsigned i = 0; i < ARRAY_SIZE(sweep); ++i) {
      const uint32_t tn = hk_gemm_cmat_select_tile_n(sweep[i].m, sweep[i].n,
                                                     sweep[i].k, true, true, true);
      const bool ug = hk_gemm_cmat_can_specialize_unguarded(
         true, sweep[i].m, sweep[i].n, sweep[i].k, tn, 0, 0, 0);
      eval_compare_gemm_cpu_and_guarded_specialize(
         sweep[i].m, sweep[i].n, sweep[i].k, sweep[i].bias_mode, tn, ug, 0,
         sweep[i].clamp ? &ep : NULL, 1200u + i);
   }
}

static void
test_dual_write_fusion(void)
{
   struct hk_nir_clamp_epilogue ep = {
      .has_lower = true, .has_upper = true,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };

   const struct dw_push_layout dw_cfg = {
      .in_w = 7, .in_h = 5, .out_w = 7, .out_h = 5,
      .ker_w = 3, .ker_h = 3, .str_x = 1, .str_y = 1,
      .dil_x = 1, .dil_y = 1, .pad_x = -1, .pad_y = -1,
      .channels = 8, .has_bias = 1,
   };
   eval_compare_dw_dual_copy(&dw_cfg, NULL);
   eval_compare_dw_dual_copy(&dw_cfg, &ep);

   const struct gemm_test_opts o_guarded = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct gemm_test_opts o_guarded_off4 = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true,
      .offset_B = 4, .use_ffma = true,
   };
   const struct gemm_test_opts o_unguarded = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };

   eval_compare_gemm_dual_copy(&o_guarded, 35, 16, 20, 1, 16, NULL, 901u);
   eval_compare_gemm_dual_copy(&o_guarded, 67, 32, 24, 1, 32, &ep,  902u);
   eval_compare_gemm_dual_copy(&o_guarded, 70, 24, 32, 1, 32, &ep,  905u);
   eval_compare_gemm_dual_copy(&o_guarded, 37, 48, 27, 0, 64, &ep,  906u);
   eval_compare_gemm_dual_copy(&o_guarded_off4, 37, 64, 19, 0, 64, &ep, 903u);
   eval_compare_gemm_dual_copy(&o_unguarded, 64, 64, 32, 1, 64, NULL, 904u);

   /* Simple asymmetric pad (bottom-right +1, e.g. stride-2 conv input). */
   const int32_t p_br[4] = { 0, 0, 0, 0 };
   eval_compare_gemm_dual_pad4(&o_guarded, 8, 8, 9, 9, 32, 16, 1, 32, p_br, &ep, 910u);
   eval_compare_gemm_dual_pad4(&o_guarded, 6, 7, 7, 8, 64, 24, 1, 64, p_br, &ep, 911u);
   eval_compare_gemm_dual_pad4(&o_guarded, 8, 9, 9, 10, 48, 24, 1, 64, p_br, &ep, 915u);
   eval_compare_gemm_dual_pad4(&o_guarded, 8, 9, 9, 10, 24, 24, 1, 32, p_br, &ep, 916u);
   /* Simple symmetric pad on guarded & unguarded. */
   const int32_t p_sym[4] = { 0, 1, 2, 0 };
   eval_compare_gemm_dual_pad4(&o_guarded, 5, 7, 8, 11, 16, 19, 1, 16, p_sym, NULL, 912u);
   eval_compare_gemm_dual_pad4(&o_unguarded, 8, 8, 10, 11, 64, 32, 1, 64, p_sym, &ep, 913u);
   /* Degenerate 1x1 spatial input (ix==0==in_w-1 && iy==0==in_h-1). */
   eval_compare_gemm_dual_pad4(&o_guarded, 1, 1, 4, 5, 16, 12, 1, 16, p_sym, &ep, 914u);
   /* Wide spatial inputs (in_w >= 64) exercising WG-uniform row-wrap & strict-interior skip. */
   eval_compare_gemm_dual_pad4(&o_unguarded, 4, 80, 6, 82, 32, 32, 1, 32, p_sym, &ep, 917u);
   eval_compare_gemm_dual_pad4(&o_unguarded, 4, 144, 5, 145, 64, 32, 1, 64, p_br, &ep, 918u);
   eval_compare_gemm_dual_pad4(&o_guarded, 3, 70, 5, 72, 16, 24, 1, 16, p_sym, &ep, 919u);

   /* Adversarial !simple_pad cases (disjoint scatter + grid-stride zero fallback). */
   const int32_t p_bad_n[4]    = { 1, 0, 0, 0 };
   const int32_t p_bad_c[4]    = { 0, 0, 0, 3 };
   const int32_t p_neg_crop[4] = { 0, -2, -1, 0 };
   const int32_t p_ovf[4]      = { 0, INT32_MIN, INT32_MAX, 0 };
   const int32_t p_rcrop[4]    = { 0, 2, 3, 0 };
   eval_compare_gemm_dual_pad4(&o_guarded, 5, 6, 6, 7, 32, 16, 1, 32, p_bad_n,    &ep, 920u);
   eval_compare_gemm_dual_pad4(&o_guarded, 5, 6, 6, 7, 32, 16, 1, 32, p_bad_c,    &ep, 921u);
   eval_compare_gemm_dual_pad4(&o_guarded, 6, 7, 5, 6, 16, 16, 1, 16, p_neg_crop, &ep, 922u);
   eval_compare_gemm_dual_pad4(&o_guarded, 4, 4, 5, 5, 16, 16, 1, 16, p_ovf,      NULL, 923u);
   eval_compare_gemm_dual_pad4(&o_guarded, 5, 6, 6, 7, 64, 24, 1, 64, p_rcrop,    &ep, 924u);

   /* Negative certifier checks: push overflow & float_controls mismatch. */
   nir_shader *orig = build_tiled_gemm_nir_opts(&o_guarded);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);
   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   struct hk_nir_gemm_output m_big = m;
   m_big.push_bytes = 252;
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat(pre, &m_big, 32, NULL, HK_COMPUTE_PAIR_DUAL_COPY) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat(pre, &m_big, 32, NULL, HK_COMPUTE_PAIR_DUAL_PAD4) == NULL);
   struct hk_nir_clamp_epilogue bad_ep = ep;
   bad_ep.float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat(pre, &m, 32, &bad_ep, HK_COMPUTE_PAIR_DUAL_COPY) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat(pre, &m, 64, &bad_ep, HK_COMPUTE_PAIR_DUAL_PAD4) == NULL);
   ralloc_free(pre);
   ralloc_free(orig);
}

static void
test_nir_derived_pack_loop_bounds(void)
{
   /* 1. Depthwise vec4: verify NIR-derived bound == kh*kw + kh + 1 and
    *    dominates actual per-thread dynamic back-edges + 1. */
   {
      const struct dw_push_layout cfg = {
         .in_w = 7, .in_h = 6, .out_w = 7, .out_h = 6,
         .ker_w = 3, .ker_h = 3, .str_x = 1, .str_y = 1,
         .dil_x = 1, .dil_y = 1, .pad_x = -1, .pad_y = -1,
         .channels = 8, .has_bias = 1,
      };
      nir_shader *orig = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
      nir_shader *pre = nir_shader_clone(NULL, orig);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_depthwise_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_depthwise_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));

      struct hk_pack_loop_bound lb = {0};
      nir_shader *dw4 = hk_nir_build_depthwise_vec4(
         pre, &m, NULL, HK_COMPUTE_PAIR_DUAL_NONE);
      HK_EVAL_CHECK(dw4 != NULL && hk_nir_bind_depthwise_loop_bounds(dw4, &m, &lb) && lb.valid);

      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_DW_KH] = 3;
      params[HK_PACK_LOOP_PARAM_DW_KW] = 3;
      uint64_t bound = 0, eff = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb, params, &bound));
      /* ky (3) * (1 + kx (3)) + 1 = 13 back-edges */
      HK_EVAL_CHECK(bound == 3ull * 3ull + 3ull + 1ull);
      HK_EVAL_CHECK(lb.work_valid && lb.static_nir_ops > 0);
      HK_EVAL_CHECK(hk_pack_eval_effective_instrs(&lb, 469, params, &eff));
      HK_EVAL_CHECK(eff >= 469 && eff < 469 * bound);

      const size_t in_n = (size_t)cfg.in_h * cfg.in_w * cfg.channels;
      const size_t wt_n = (size_t)cfg.ker_h * cfg.ker_w * cfg.channels;
      const size_t out_n = (size_t)cfg.out_h * cfg.out_w * cfg.channels;
      float *in_b = calloc(in_n, sizeof(float));
      float *wt_b = calloc(wt_n, sizeof(float));
      float *bi_b = calloc(cfg.channels, sizeof(float));
      float *out_b = calloc(out_n, sizeof(float));
      struct hk_eval_ssbo ssbos[4] = {
         { 0, 0, in_b, in_n * 4, 0 },
         { 0, 1, wt_b, wt_n * 4, 0 },
         { 0, 2, bi_b, cfg.channels * 4, 0 },
         { 0, 3, out_b, out_n * 4, 0 },
      };
      const uint32_t g[3] = {
         DIV_ROUND_UP(cfg.channels / 4, dw4->info.workgroup_size[0]),
         DIV_ROUND_UP((cfg.out_w + 1) / 2, dw4->info.workgroup_size[1]),
         DIV_ROUND_UP(cfg.out_h, dw4->info.workgroup_size[2]),
      };
      hk_eval_dispatch(dw4, g, ssbos, 4, &cfg, sizeof(cfg));
      HK_EVAL_CHECK(bound >= hk_eval_last_max_loop_backedges + 1);
      free(in_b); free(wt_b); free(bi_b); free(out_b);
      ralloc_free(dw4); ralloc_free(pre); ralloc_free(orig);
   }

   /* 2. GEMM cmat (guarded, DUAL_PAD4 simple & adversarial):
    *    verify NIR-derived bound >= actual max per-thread dynamic back-edges + 1. */
   {
      const struct gemm_test_opts o_g = {
         .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
      };
      nir_shader *orig = build_tiled_gemm_nir_opts(&o_g);
      nir_shader *pre = nir_shader_clone(NULL, orig);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));

      const uint32_t M = 35, N = 32, K = 19, tile_n = 32;
      const uint32_t k_step = 16;
      struct hk_pack_loop_bound lb_g = {0};
      nir_shader *cmat_g = hk_nir_build_gemm_cmat_with_bound(
         pre, &m, tile_n, NULL, HK_COMPUTE_PAIR_DUAL_NONE, NULL, false, NULL, NULL, &lb_g);
      HK_EVAL_CHECK(cmat_g != NULL && lb_g.valid);

      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, k_step);
      uint64_t bound_g = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb_g, params, &bound_g));

      float *a = calloc((size_t)M * K, 4);
      float *b = calloc((size_t)K * N, 4);
      float *bi = calloc(N, 4);
      float *c = calloc((size_t)M * N, 4);
      struct gemm_push_layout push = { .k = K, .n = N, .m = M, .bias_mode = 1 };
      const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
      const uint32_t g[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, tile_m), 1 };
      struct hk_eval_ssbo ssbos[4] = {
         { 0, 0, a, (size_t)M * K * 4, 0 },
         { 0, 1, b, (size_t)K * N * 4, 0 },
         { 0, 2, bi, (size_t)N * 4, 0 },
         { 0, 3, c, (size_t)M * N * 4, 0 },
      };
      hk_eval_dispatch_wg(cmat_g, g, ssbos, 4, &push, sizeof(push));
      HK_EVAL_CHECK(bound_g >= hk_eval_last_max_loop_backedges + 1);

      /* DUAL_PAD4 (both simple_pad and adversarial !simple_pad zero-fill zlp). */
      struct hk_pack_loop_bound lb_p = {0};
      nir_shader *cmat_p = hk_nir_build_gemm_cmat_with_bound(
         pre, &m, tile_n, NULL, HK_COMPUTE_PAIR_DUAL_PAD4, NULL, false, NULL, NULL, &lb_p);
      HK_EVAL_CHECK(cmat_p != NULL && lb_p.valid && lb_p.work_valid);

      const uint32_t in_h = 5, in_w = 7, out_h = 7, out_w = 9;
      const uint32_t total_threads = g[0] * g[1] * 128u;
      const uint64_t out_total = (uint64_t)out_h * out_w * N;
      uint64_t p_params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      p_params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, k_step);
      p_params[HK_PACK_LOOP_PARAM_PAD_H] = out_h - in_h;
      p_params[HK_PACK_LOOP_PARAM_PAD_W] = out_w - in_w;
      p_params[HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS] = DIV_ROUND_UP(out_total, total_threads);
      uint64_t bound_p = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb_p, p_params, &bound_p));
      HK_EVAL_CHECK(bound_p > bound_g);

      float *d = calloc(out_total, 4);
      int32_t pad_simple[4] = { 0, 1, 1, 0 };
      int32_t pad_adv[4] = { 1, 0, 0, 0 };
      const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
      uint8_t push_buf[128] = {0};
      memcpy(push_buf, &push, sizeof(push));
      const uint64_t dst2_va = hk_eval_make_vaddr(4, 0);
      const uint64_t pad_va = hk_eval_make_vaddr(5, 0);
      struct hk_dual_pad4_push dp = {
         .dst2_va_lo = (uint32_t)dst2_va, .dst2_va_hi = (uint32_t)(dst2_va >> 32),
         .pad_va_lo = (uint32_t)pad_va, .pad_va_hi = (uint32_t)(pad_va >> 32),
         .in_h = in_h, .in_w = in_w, .out_h = out_h, .out_w = out_w,
      };
      memcpy(push_buf + dual_off, &dp, sizeof(dp));
      struct hk_eval_ssbo ssbos_p[6] = {
         { 0, 0, a, (size_t)M * K * 4, 0 },
         { 0, 1, b, (size_t)K * N * 4, 0 },
         { 0, 2, bi, (size_t)N * 4, 0 },
         { 0, 3, c, (size_t)M * N * 4, 0 },
         { 0, 9, d, out_total * 4, 0 },
         { 0, 10, pad_simple, sizeof(pad_simple), 0 },
      };
      hk_eval_dispatch_wg(cmat_p, g, ssbos_p, 6, push_buf, dual_off + sizeof(dp));
      HK_EVAL_CHECK(bound_p >= hk_eval_last_max_loop_backedges + 1);

      ssbos_p[5].data = pad_adv;
      hk_eval_dispatch_wg(cmat_p, g, ssbos_p, 6, push_buf, dual_off + sizeof(dp));
      HK_EVAL_CHECK(bound_p >= hk_eval_last_max_loop_backedges + 1);

      free(a); free(b); free(bi); free(c); free(d);
      ralloc_free(cmat_p); ralloc_free(cmat_g);

      /* 2b. cmat64 + clamp + DUAL_PAD4 (large multi-branch store epilogue):
       * verify affine constant folding keeps work_ops well within budget so
       * work_valid stays true and CF-weighted effective work stays far below
       * naive (agx_instrs * loop_bound) on large 320x320x96 -> 321x321x96. */
      {
         const struct hk_nir_clamp_epilogue relu6 = {
            .has_lower = true, .has_upper = true, .upper_first = false,
            .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
            .float_controls_execution_mode = pre->info.float_controls_execution_mode,
         };
         struct hk_pack_loop_bound lb64 = {0};
         nir_shader *cmat64_p = hk_nir_build_gemm_cmat_with_bound(
            pre, &m, 64, &relu6, HK_COMPUTE_PAIR_DUAL_PAD4, NULL, false, NULL, NULL, &lb64);
         HK_EVAL_CHECK(cmat64_p != NULL && lb64.valid && lb64.work_valid);
         HK_EVAL_CHECK(lb64.work_count <= HK_PACK_LOOP_WORK_MAX_OPS);

         uint64_t fd_params[HK_PACK_LOOP_PARAM_COUNT] = {0};
         const uint64_t fd_threads = 2ull * 3200ull * 128ull; /* 819,200 threads */
         const uint64_t fd_out_total = 321ull * 321ull * 96ull;
         fd_params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = 1;
         fd_params[HK_PACK_LOOP_PARAM_PAD_H] = 1;
         fd_params[HK_PACK_LOOP_PARAM_PAD_W] = 1;
         fd_params[HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS] =
            DIV_ROUND_UP(fd_out_total, fd_threads);

         uint64_t naive_lb = 0, eff_instrs = 0;
         HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb64, fd_params, &naive_lb));
         HK_EVAL_CHECK(hk_pack_eval_effective_instrs(&lb64, 1200, fd_params,
                                                     &eff_instrs));
         HK_EVAL_CHECK(eff_instrs < 1200ull * naive_lb / 4ull);
         HK_EVAL_CHECK(eff_instrs * fd_threads < (1ull << 35));
         ralloc_free(cmat64_p);
      }

      ralloc_free(pre); ralloc_free(orig);
   }

   /* 3. Negative fail-closed check: unrecognized loop MUST return valid == false. */
   {
      const nir_shader_compiler_options opts = {0};
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts, "unbounded");
      nir_loop *lp = nir_push_loop(&b);
      nir_if *nif = nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, nif);
      nir_pop_loop(&b, lp);
      struct hk_pack_loop_bound bad = { .valid = true };
      HK_EVAL_CHECK(!hk_nir_compute_pack_loop_bound(b.shader, NULL, 0, &bad));
      HK_EVAL_CHECK(!bad.valid);
      ralloc_free(b.shader);
   }
}

static void
test_fast_udivmod_general(void)
{
   /* Build a compute shader evaluating hk_nir_fast_udiv_prepare +
    * hk_nir_fast_udivmod across arbitrary 32-bit unsigned (num, den) pairs. */
   nir_shader *s = new_compute_shader("fast_udivmod_test", 1, 1, 1);
   s->info.num_ssbos = 2;
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *den = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                         .base = 0, .range = 4);
   nir_def *num = nir_load_deref(&b, nir_instr_as_deref(nir_def_instr(ssbo_elem_deref(&b, 0, 0, gid))));
   struct hk_nir_fast_udiv fd = hk_nir_fast_udiv_prepare(&b, den);
   nir_def *q = NULL, *r = NULL;
   hk_nir_fast_udivmod(&b, &fd, num, &q, &r);
   nir_store_deref(&b, nir_instr_as_deref(nir_def_instr(ssbo_elem_deref(&b, 0, 1, nir_ishl_imm(&b, gid, 1)))), q, 1);
   nir_store_deref(&b, nir_instr_as_deref(nir_def_instr(ssbo_elem_deref(&b, 0, 1, nir_iadd_imm(&b, nir_ishl_imm(&b, gid, 1), 1)))), r, 1);
   nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
   nir_validate_shader(s, "fast_udivmod_test");

   static const uint32_t divisors[] = {
      1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 13u, 15u, 16u, 17u,
      24u, 31u, 32u, 33u, 48u, 63u, 64u, 80u, 96u, 127u, 128u, 144u,
      255u, 256u, 1000u, 65535u, 65536u,
      0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu,
   };

   uint32_t nums[64];
   uint32_t out_qr[128];
   for (unsigned di = 0; di < ARRAY_SIZE(divisors); ++di) {
      const uint32_t d = divisors[di];
      unsigned ncnt = 0;
      nums[ncnt++] = 0u;
      nums[ncnt++] = 1u;
      nums[ncnt++] = 2u;
      nums[ncnt++] = d - 1u;
      nums[ncnt++] = d;
      nums[ncnt++] = d + 1u;
      nums[ncnt++] = d * 2u - 1u;
      nums[ncnt++] = d * 2u;
      nums[ncnt++] = d * 2u + 1u;
      nums[ncnt++] = 0x7fffffffu;
      nums[ncnt++] = 0x80000000u;
      nums[ncnt++] = 0xfffffffeu;
      nums[ncnt++] = 0xffffffffu;
      uint32_t rng = 0x9e3779b9u ^ (d * 2654435761u);
      while (ncnt < ARRAY_SIZE(nums)) {
         rng ^= rng << 13;
         rng ^= rng >> 17;
         rng ^= rng << 5;
         nums[ncnt++] = rng;
      }

      struct hk_eval_ssbo ssbos[2] = {
         { 0, 0, nums, sizeof(nums), 0 },
         { 0, 1, out_qr, sizeof(out_qr), 0 },
      };
      const uint32_t g[3] = { ncnt, 1, 1 };
      hk_eval_dispatch(s, g, ssbos, 2, &d, sizeof(d));
      for (unsigned i = 0; i < ncnt; ++i) {
         const uint32_t exp_q = nums[i] / d;
         const uint32_t exp_r = nums[i] % d;
         HK_EVAL_CHECK(out_qr[2 * i] == exp_q);
         HK_EVAL_CHECK(out_qr[2 * i + 1] == exp_r);
      }
   }
   ralloc_free(s);
}

static void
test_depthwise_shape_sweep_cpu_ref(void)
{
   /* Broad shape sweep verifying depthwise_vec4 (including rolled-ky 3x3 s1/s2
    * fast paths, odd out_w/out_h, asymmetric padding, 1x1/5x5/dilated kernels,
    * varied channels, bias/clamp/DUAL_COPY) against an independent scalar CPU
    * FP32 reference. */
   static const struct {
      struct dw_push_layout cfg;
      bool has_bias_buf;
      bool with_clamp;
      bool with_dual_copy;
   } cases[] = {
      /* 1x1 spatial (pure single-pixel edge) */
      { { .channels = 4, .out_w = 1, .out_h = 1, .in_w = 1, .in_h = 1,
          .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
          .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, true, false },
      /* Odd 3x5, 3x3 s1 */
      { { .channels = 12, .out_w = 3, .out_h = 5, .in_w = 3, .in_h = 5,
          .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
          .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, false, true },
      /* Odd 9x7, 3x3 s1, asymmetric pad, 24 channels */
      { { .channels = 24, .out_w = 9, .out_h = 7, .in_w = 10, .in_h = 8,
          .ker_w = 3, .ker_h = 3, .pad_x = 0, .pad_y = -1,
          .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, true, true },
      /* Even 10x6, 3x3 s2, asymmetric pad */
      { { .channels = 16, .out_w = 10, .out_h = 6, .in_w = 21, .in_h = 13,
          .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = 0,
          .str_x = 2, .str_y = 2, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, true, false },
      /* Odd 7x5, 3x3 s2, no bias buffer */
      { { .channels = 8, .out_w = 7, .out_h = 5, .in_w = 14, .in_h = 10,
          .ker_w = 3, .ker_h = 3, .pad_x = -1, .pad_y = -1,
          .str_x = 2, .str_y = 2, .dil_x = 1, .dil_y = 1, .has_bias = 0 },
        false, false, false },
      /* 5x5 s1, odd 7x5 */
      { { .channels = 8, .out_w = 7, .out_h = 5, .in_w = 7, .in_h = 5,
          .ker_w = 5, .ker_h = 5, .pad_x = -2, .pad_y = -2,
          .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, true, false },
      /* 5x5 s2, even 4x4 */
      { { .channels = 32, .out_w = 4, .out_h = 4, .in_w = 9, .in_h = 9,
          .ker_w = 5, .ker_h = 5, .pad_x = -1, .pad_y = -1,
          .str_x = 2, .str_y = 2, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, false, true },
      /* 1x1 s1, odd 5x3 */
      { { .channels = 16, .out_w = 5, .out_h = 3, .in_w = 5, .in_h = 3,
          .ker_w = 1, .ker_h = 1, .pad_x = 0, .pad_y = 0,
          .str_x = 1, .str_y = 1, .dil_x = 1, .dil_y = 1, .has_bias = 1 },
        true, true, false },
      /* 3x3 dil=2, odd 7x7 */
      { { .channels = 12, .out_w = 7, .out_h = 7, .in_w = 7, .in_h = 7,
          .ker_w = 3, .ker_h = 3, .pad_x = -2, .pad_y = -2,
          .str_x = 1, .str_y = 1, .dil_x = 2, .dil_y = 2, .has_bias = 1 },
        true, true, false },
   };

   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = 0x00000000u, .upper_bits = 0x40c00000u,
      .max_fp_math_ctrl = nir_fp_preserve_inf,
      .min_fp_math_ctrl = nir_fp_preserve_inf,
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const struct dw_push_layout *cfg = &cases[ci].cfg;
      const struct hk_nir_clamp_epilogue *ep = cases[ci].with_clamp ? &relu6 : NULL;
      const enum hk_compute_pair_dual_write dw_mode =
         cases[ci].with_dual_copy ? HK_COMPUTE_PAIR_DUAL_COPY : HK_COMPUTE_PAIR_DUAL_NONE;

      nir_shader *orig = build_depthwise_nir(cases[ci].has_bias_buf, true, 0, 0, 1, 2, 3);
      struct hk_nir_depthwise_output info = {0};
      HK_EVAL_CHECK(hk_nir_match_depthwise_output(
         orig, nir_address_format_64bit_global_32bit_offset, &info));
      nir_shader *alt = hk_nir_build_depthwise_vec4(orig, &info, ep, dw_mode);
      HK_EVAL_CHECK(alt != NULL);
      nir_validate_shader(alt, "dw_shape_sweep");

      const size_t in_n = (size_t)cfg->in_h * cfg->in_w * cfg->channels;
      const size_t wt_n = (size_t)cfg->ker_h * cfg->ker_w * cfg->channels;
      const size_t out_n = (size_t)cfg->out_h * cfg->out_w * cfg->channels;
      float *in_b = calloc(in_n, sizeof(float));
      float *wt_b = calloc(wt_n, sizeof(float));
      float *bi_b = calloc(cfg->channels, sizeof(float));
      float *out_alt = calloc(out_n, sizeof(float));
      float *out_dup = calloc(out_n, sizeof(float));
      float *out_ref = calloc(out_n, sizeof(float));

      for (size_t i = 0; i < in_n; ++i)
         in_b[i] = ((float)((int)((i * 37 + ci * 13) % 29) - 14)) * 0.25f;
      for (size_t i = 0; i < wt_n; ++i)
         wt_b[i] = ((float)((int)((i * 19 + ci * 7) % 17) - 8)) * 0.125f;
      for (int i = 0; i < cfg->channels; ++i)
         bi_b[i] = ((float)((i % 11) - 5)) * 0.5f;

      /* Independent scalar CPU FP32 reference. */
      for (int y = 0; y < cfg->out_h; ++y) {
         for (int x = 0; x < cfg->out_w; ++x) {
            for (int c = 0; c < cfg->channels; ++c) {
               float acc = 0.0f;
               for (int ky = 0; ky < cfg->ker_h; ++ky) {
                  int iy = y * cfg->str_y + cfg->pad_y + ky * cfg->dil_y;
                  if (iy < 0 || iy >= cfg->in_h)
                     continue;
                  for (int kx = 0; kx < cfg->ker_w; ++kx) {
                     int ix = x * cfg->str_x + cfg->pad_x + kx * cfg->dil_x;
                     if (ix < 0 || ix >= cfg->in_w)
                        continue;
                     float iv = in_b[((size_t)iy * cfg->in_w + ix) * cfg->channels + c];
                     float wv = wt_b[((size_t)ky * cfg->ker_w + kx) * cfg->channels + c];
                     acc = ref_mul_add_unfused(iv, wv, acc);
                  }
               }
               if (cases[ci].has_bias_buf && cfg->has_bias)
                  acc += bi_b[c];
               if (ep) {
                  if (acc < 0.0f) acc = 0.0f;
                  if (acc > 6.0f) acc = 6.0f;
               }
               out_ref[((size_t)y * cfg->out_w + x) * cfg->channels + c] = acc;
            }
         }
      }

      uint8_t push_buf[128] = {0};
      memcpy(push_buf, cfg, sizeof(*cfg));
      size_t push_sz = sizeof(*cfg);
      struct hk_eval_ssbo ssbos[5];
      unsigned n_ssbos = 0;
      ssbos[n_ssbos++] = (struct hk_eval_ssbo){ 0, 0, in_b, in_n * 4, 0 };
      ssbos[n_ssbos++] = (struct hk_eval_ssbo){ 0, 1, wt_b, wt_n * 4, 0 };
      if (cases[ci].has_bias_buf)
         ssbos[n_ssbos++] = (struct hk_eval_ssbo){ 0, 2, bi_b, (size_t)cfg->channels * 4, 0 };
      ssbos[n_ssbos++] = (struct hk_eval_ssbo){ 0, 3, out_alt, out_n * 4, 0 };
      if (cases[ci].with_dual_copy) {
         const uint32_t dual_off = hk_dual_write_push_offset(info.push_bytes);
         const uint64_t dst2_va = hk_eval_make_vaddr(n_ssbos, 0);
         struct hk_dual_copy_push dp = {
            .dst2_va_lo = (uint32_t)dst2_va,
            .dst2_va_hi = (uint32_t)(dst2_va >> 32),
         };
         memcpy(push_buf + dual_off, &dp, sizeof(dp));
         push_sz = dual_off + sizeof(dp);
         ssbos[n_ssbos++] = (struct hk_eval_ssbo){ 0, 9, out_dup, out_n * 4, 0 };
      }

      const uint32_t g[3] = {
         DIV_ROUND_UP(cfg->channels / 4, alt->info.workgroup_size[0]),
         DIV_ROUND_UP((cfg->out_w + 1) / 2, alt->info.workgroup_size[1]),
         DIV_ROUND_UP(cfg->out_h, alt->info.workgroup_size[2]),
      };
      hk_eval_dispatch(alt, g, ssbos, n_ssbos, push_buf, push_sz);

      for (size_t i = 0; i < out_n; ++i) {
         HK_EVAL_CHECK(memcmp(&out_alt[i], &out_ref[i], sizeof(float)) == 0);
         if (cases[ci].with_dual_copy)
            HK_EVAL_CHECK(memcmp(&out_dup[i], &out_ref[i], sizeof(float)) == 0);
      }

      free(in_b); free(wt_b); free(bi_b);
      free(out_alt); free(out_dup); free(out_ref);
      ralloc_free(alt); ralloc_free(orig);
   }
}

static void
eval_compare_gemm_dual_fadd(const struct gemm_test_opts *opts,
                            uint32_t M, uint32_t N, uint32_t K,
                            int32_t bias_mode, uint32_t tile_n,
                            const struct hk_nir_clamp_epilogue *epilogue,
                            bool prod_is_rhs, bool also_copy, uint32_t seed)
{
   nir_shader *orig = build_tiled_gemm_nir_opts(opts);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);

   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   HK_EVAL_CHECK(m.cmat_valid);

   const struct hk_nir_fadd_epilogue fadd_ep = {
      .fp_math_ctrl = 0,
      .float_controls_execution_mode = pre->info.float_controls_execution_mode,
      .prod_is_rhs = prod_is_rhs,
   };
   const enum hk_compute_pair_dual_write dw_mode =
      also_copy ? HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY : HK_COMPUTE_PAIR_DUAL_FADD_EP;
   struct hk_pack_loop_bound lb = {0};
   nir_shader *alt = hk_nir_build_gemm_cmat_with_bound(
      pre, &m, tile_n, epilogue, dw_mode, &fadd_ep, false, NULL, NULL, &lb);
   HK_EVAL_CHECK(alt && lb.valid && lb.work_valid);
   nir_validate_shader(alt, also_copy ? "gemm_cmat_dfaddcopy" : "gemm_cmat_dfadd");

   const uint32_t off_elems = opts->offset_B / 4u;
   const size_t mn = (size_t)M * N;
   const size_t a_elems = (size_t)M * K + off_elems;
   const size_t b_elems = (size_t)K * N + off_elems;
   const size_t bias_elems = (size_t)(M > N ? M : N) + off_elems + 1;
   const size_t c_elems = mn + off_elems;
   const size_t add_elems = mn + 4;
   const size_t fd_elems = mn + 4;
   const size_t d2_elems = mn + 4;

   float *a_buf = calloc(a_elems, sizeof(float));
   float *b_buf = calloc(b_elems, sizeof(float));
   float *bias_buf = calloc(bias_elems, sizeof(float));
   float *c_orig = calloc(c_elems, sizeof(float));
   float *c_alt = calloc(c_elems, sizeof(float));
   float *add_src = calloc(add_elems, sizeof(float));
   float *fadd_dst = calloc(fd_elems, sizeof(float));
   float *dst2 = calloc(d2_elems, sizeof(float));
   HK_EVAL_CHECK(a_buf && b_buf && bias_buf && c_orig && c_alt &&
                 add_src && fadd_dst && dst2);

   uint32_t rng = seed;
   for (size_t i = off_elems; i < a_elems; ++i) a_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = off_elems; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = off_elems; i < bias_elems; ++i) bias_buf[i] = gemm_test_rand_f32(&rng);
   for (size_t i = 0; i < add_elems; ++i) add_src[i] = gemm_test_rand_f32(&rng);
   for (size_t i = 0; i < fd_elems; ++i) fadd_dst[i] = -33.5f;
   for (size_t i = 0; i < d2_elems; ++i) dst2[i] = -77.25f;

   struct gemm_push_layout push = {
      .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = bias_mode,
   };
   const uint32_t orig_grid[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
   struct hk_eval_ssbo ssbos_orig[4] = {
      { 0, 0, a_buf, a_elems * sizeof(float), 0 },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_orig, c_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(orig, orig_grid, ssbos_orig, 4, &push, sizeof(push));
   if (epilogue) {
      float lo = hk_eval_u2f(epilogue->lower_bits);
      float hi = hk_eval_u2f(epilogue->upper_bits);
      for (size_t i = off_elems; i < c_elems; ++i) {
         float v = c_orig[i];
         if (epilogue->has_lower) v = fmaxf(v, lo);
         if (epilogue->has_upper) v = fminf(v, hi);
         c_orig[i] = v;
      }
   }

   uint8_t push_buf[256] = {0};
   memcpy(push_buf, &push, sizeof(push));
   const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
   const uint64_t add_va = hk_eval_make_vaddr(4, 16);
   const uint64_t fd_va = hk_eval_make_vaddr(5, 16);
   const uint64_t d2_va = hk_eval_make_vaddr(6, 16);
   size_t push_sz = 0;
   if (also_copy) {
      struct hk_dual_fadd_ep_copy_push dp = {
         .add_src_va_lo = (uint32_t)add_va, .add_src_va_hi = (uint32_t)(add_va >> 32),
         .fadd_dst_va_lo = (uint32_t)fd_va, .fadd_dst_va_hi = (uint32_t)(fd_va >> 32),
         .dst2_va_lo = (uint32_t)d2_va, .dst2_va_hi = (uint32_t)(d2_va >> 32),
      };
      memcpy(push_buf + dual_off, &dp, sizeof(dp));
      push_sz = dual_off + sizeof(dp);
   } else {
      struct hk_dual_fadd_ep_push dp = {
         .add_src_va_lo = (uint32_t)add_va, .add_src_va_hi = (uint32_t)(add_va >> 32),
         .fadd_dst_va_lo = (uint32_t)fd_va, .fadd_dst_va_hi = (uint32_t)(fd_va >> 32),
      };
      memcpy(push_buf + dual_off, &dp, sizeof(dp));
      push_sz = dual_off + sizeof(dp);
   }

   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const uint32_t alt_grid[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, tile_m), 1u };
   struct hk_eval_ssbo ssbos_alt[7] = {
      { 0, 0, a_buf, a_elems * sizeof(float), 0 },
      { 0, 1, b_buf, b_elems * sizeof(float), 0 },
      { 0, 2, bias_buf, bias_elems * sizeof(float), 0 },
      { 0, 3, c_alt, c_elems * sizeof(float), 0 },
      { 0, 9, add_src, add_elems * sizeof(float), 0 },
      { 0, 10, fadd_dst, fd_elems * sizeof(float), 0 },
      { 0, 11, dst2, d2_elems * sizeof(float), 0 },
   };
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, also_copy ? 7 : 6, push_buf, push_sz);

   for (size_t i = 0; i < 4; ++i) {
      HK_EVAL_CHECK(hk_eval_f2u(fadd_dst[i]) == hk_eval_f2u(-33.5f));
      if (also_copy)
         HK_EVAL_CHECK(hk_eval_f2u(dst2[i]) == hk_eval_f2u(-77.25f));
   }
   for (size_t i = 0; i < mn; ++i) {
      const float prod_v = c_orig[off_elems + i];
      const float add_v = add_src[4 + i];
      const float ref_sum = prod_is_rhs ? (add_v + prod_v) : (prod_v + add_v);
      HK_EVAL_CHECK(hk_eval_f2u(c_alt[off_elems + i]) == hk_eval_f2u(prod_v));
      HK_EVAL_CHECK(hk_eval_f2u(fadd_dst[4 + i]) == hk_eval_f2u(ref_sum));
      if (also_copy)
         HK_EVAL_CHECK(hk_eval_f2u(dst2[4 + i]) == hk_eval_f2u(ref_sum));
   }

   /* Verify SKIP_C (and SKIP_FADD_DST when also_copy) leave dead outputs
    * untouched while keeping live output bit-exact. */
   for (size_t i = 0; i < mn; ++i)
      c_alt[off_elems + i] = -88.5f;
   for (size_t i = 0; i < fd_elems; ++i)
      fadd_dst[i] = -33.5f;
   for (size_t i = 0; i < d2_elems; ++i)
      dst2[i] = -77.25f;
   if (also_copy) {
      struct hk_dual_fadd_ep_copy_push dp = {
         .add_src_va_lo = (uint32_t)add_va, .add_src_va_hi = (uint32_t)(add_va >> 32),
         .fadd_dst_va_lo = (uint32_t)fd_va, .fadd_dst_va_hi = (uint32_t)(fd_va >> 32),
         .dst2_va_lo = (uint32_t)d2_va, .dst2_va_hi = (uint32_t)(d2_va >> 32),
         .flags = HK_DUAL_WRITE_FLAG_SKIP_C | HK_DUAL_WRITE_FLAG_SKIP_FADD_DST,
      };
      memcpy(push_buf + dual_off, &dp, sizeof(dp));
   } else {
      struct hk_dual_fadd_ep_push dp = {
         .add_src_va_lo = (uint32_t)add_va, .add_src_va_hi = (uint32_t)(add_va >> 32),
         .fadd_dst_va_lo = (uint32_t)fd_va, .fadd_dst_va_hi = (uint32_t)(fd_va >> 32),
         .flags = HK_DUAL_WRITE_FLAG_SKIP_C,
      };
      memcpy(push_buf + dual_off, &dp, sizeof(dp));
   }
   hk_eval_dispatch_wg(alt, alt_grid, ssbos_alt, also_copy ? 7 : 6, push_buf, push_sz);
   for (size_t i = 0; i < mn; ++i) {
      const float prod_v = c_orig[off_elems + i];
      const float add_v = add_src[4 + i];
      const float ref_sum = prod_is_rhs ? (add_v + prod_v) : (prod_v + add_v);
      HK_EVAL_CHECK(hk_eval_f2u(c_alt[off_elems + i]) == hk_eval_f2u(-88.5f));
      if (also_copy) {
         HK_EVAL_CHECK(hk_eval_f2u(fadd_dst[4 + i]) == hk_eval_f2u(-33.5f));
         HK_EVAL_CHECK(hk_eval_f2u(dst2[4 + i]) == hk_eval_f2u(ref_sum));
      } else {
         HK_EVAL_CHECK(hk_eval_f2u(fadd_dst[4 + i]) == hk_eval_f2u(ref_sum));
      }
   }

   free(a_buf); free(b_buf); free(bias_buf);
   free(c_orig); free(c_alt); free(add_src); free(fadd_dst); free(dst2);
   ralloc_free(alt); ralloc_free(pre); ralloc_free(orig);
}

static void
test_gemm_dual_fadd_fusion(void)
{
   const struct gemm_test_opts o_guard = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct gemm_test_opts o_ug = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };

   /* Guarded tail shapes across all tile sizes, both operand orders, with/without clamp & copy. */
   eval_compare_gemm_dual_fadd(&o_guard, 35, 13, 19, 1, 16, NULL, false, false, 0x5101u);
   eval_compare_gemm_dual_fadd(&o_guard, 35, 13, 19, 1, 16, &relu6, true, true, 0x5102u);
   eval_compare_gemm_dual_fadd(&o_guard, 67, 29, 23, 1, 32, NULL, true, false, 0x5103u);
   eval_compare_gemm_dual_fadd(&o_guard, 67, 29, 23, 0, 32, &relu6, false, true, 0x5104u);
   eval_compare_gemm_dual_fadd(&o_guard, 37, 61, 35, 1, 64, NULL, false, true, 0x5105u);
   eval_compare_gemm_dual_fadd(&o_guard, 37, 61, 35, 2, 64, &relu6, true, false, 0x5106u);
   eval_compare_gemm_dual_fadd(&o_guard, 70, 24, 32, 1, 32, &relu6, false, true, 0x5107u);
   eval_compare_gemm_dual_fadd(&o_guard, 37, 48, 24, 0, 64, NULL, true, false, 0x5108u);
   eval_compare_gemm_dual_fadd(&o_guard, 37, 96, 32, 1, 64, &relu6, true, true, 0x5109u);

   /* Unguarded fast path (row-0 then row-1) across tile_n = 16/32/64. */
   eval_compare_gemm_dual_fadd(&o_ug, 64, 32, 32, 1, 16, NULL, false, true, 0x5201u);
   eval_compare_gemm_dual_fadd(&o_ug, 64, 32, 32, 1, 32, &relu6, true, false, 0x5202u);
   eval_compare_gemm_dual_fadd(&o_ug, 64, 64, 32, 1, 64, NULL, true, true, 0x5203u);

   /* Negative checks: missing fadd_ep, float_controls mismatch, push overflow. */
   nir_shader *orig = build_tiled_gemm_nir_opts(&o_guard);
   nir_shader *pre = nir_shader_clone(NULL, orig);
   run_hk_preprocess_for_test(pre, false);
   struct hk_nir_gemm_output m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      pre, nir_address_format_64bit_global_32bit_offset, &m));
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_with_bound(
      pre, &m, 32, NULL, HK_COMPUTE_PAIR_DUAL_FADD_EP, NULL, false, NULL, NULL, NULL) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_with_bound(
      pre, &m, 32, NULL, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, NULL, false, NULL, NULL, NULL) == NULL);
   const struct hk_nir_fadd_epilogue bad_fc = {
      .float_controls_execution_mode = FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
   };
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_with_bound(
      pre, &m, 32, NULL, HK_COMPUTE_PAIR_DUAL_FADD_EP, &bad_fc, false, NULL, NULL, NULL) == NULL);
   struct hk_nir_gemm_output m_big = m;
   m_big.push_bytes = 240;
   const struct hk_nir_fadd_epilogue ok_fc = {
      .float_controls_execution_mode = pre->info.float_controls_execution_mode,
   };
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_with_bound(
      pre, &m_big, 32, NULL, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, &ok_fc, false, NULL, NULL, NULL) == NULL);
   ralloc_free(pre);
   ralloc_free(orig);
}

static void
test_dual_copy2_fusion(void)
{
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };

   /* 2. GEMM cmat DUAL_COPY2 (guarded + unguarded) bit-exact vs original. */
   const struct { bool guarded; uint32_t M, N, K, tile_n; bool clamp; } gemm_cases[] = {
      { true,  35, 29, 19, 32, true },
      { false, 64, 64, 32, 64, false },
   };
   for (unsigned ci = 0; ci < ARRAY_SIZE(gemm_cases); ++ci) {
      const struct gemm_test_opts opts = {
         .guarded = gemm_cases[ci].guarded, .has_bias_buf = true,
         .has_bias_mode = true, .use_ffma = true,
      };
      const uint32_t M = gemm_cases[ci].M, N = gemm_cases[ci].N, K = gemm_cases[ci].K;
      const uint32_t tile_n = gemm_cases[ci].tile_n;
      const struct hk_nir_clamp_epilogue *ep = gemm_cases[ci].clamp ? &relu6 : NULL;

      nir_shader *orig = build_tiled_gemm_nir_opts(&opts);
      nir_shader *pre = nir_shader_clone(NULL, orig);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));
      nir_shader *alt = hk_nir_build_gemm_cmat(pre, &m, tile_n, ep,
                                               HK_COMPUTE_PAIR_DUAL_COPY2);
      HK_EVAL_CHECK(alt != NULL);
      nir_validate_shader(alt, "gemm_cmat_dcopy2");

      const size_t mn = (size_t)M * N;
      float *a = calloc((size_t)M * K, 4), *b = calloc((size_t)K * N, 4);
      float *bi = calloc(M > N ? M : N, 4);
      float *c_orig = calloc(mn, 4), *c_alt = calloc(mn, 4);
      float *dst2 = calloc(mn + 4, 4), *dst3 = calloc(mn + 4, 4);
      uint32_t rng = 0x6200u + ci;
      for (size_t i = 0; i < (size_t)M * K; ++i) a[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < (size_t)K * N; ++i) b[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < (M > N ? M : N); ++i) bi[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < mn + 4; ++i) { dst2[i] = -41.0f; dst3[i] = -82.0f; }

      struct gemm_push_layout push = { .k = K, .n = N, .m = M, .bias_mode = 1 };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      struct hk_eval_ssbo so[4] = {
         { 0, 0, a, (size_t)M * K * 4, 0 }, { 0, 1, b, (size_t)K * N * 4, 0 },
         { 0, 2, bi, (size_t)(M > N ? M : N) * 4, 0 }, { 0, 3, c_orig, mn * 4, 0 },
      };
      hk_eval_dispatch_wg(orig, og, so, 4, &push, sizeof(push));
      if (ep) {
         for (size_t i = 0; i < mn; ++i) {
            float v = c_orig[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 6.0f) v = 6.0f;
            c_orig[i] = v;
         }
      }

      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));
      const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
      const uint64_t d2_va = hk_eval_make_vaddr(4, 16);
      const uint64_t d3_va = hk_eval_make_vaddr(5, 16);
      struct hk_dual_copy2_push dp = {
         .dst2_va_lo = (uint32_t)d2_va, .dst2_va_hi = (uint32_t)(d2_va >> 32),
         .dst3_va_lo = (uint32_t)d3_va, .dst3_va_hi = (uint32_t)(d3_va >> 32),
      };
      memcpy(push_buf + dual_off, &dp, sizeof(dp));
      const uint32_t ag[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, hk_gemm_cmat_tile_m(tile_n)), 1u };
      struct hk_eval_ssbo sa[6] = {
         { 0, 0, a, (size_t)M * K * 4, 0 }, { 0, 1, b, (size_t)K * N * 4, 0 },
         { 0, 2, bi, (size_t)(M > N ? M : N) * 4, 0 }, { 0, 3, c_alt, mn * 4, 0 },
         { 0, 9, dst2, (mn + 4) * 4, 0 }, { 0, 10, dst3, (mn + 4) * 4, 0 },
      };
      hk_eval_dispatch_wg(alt, ag, sa, 6, push_buf, dual_off + sizeof(dp));

      for (size_t i = 0; i < 4; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(dst2[i]) == hk_eval_f2u(-41.0f));
         HK_EVAL_CHECK(hk_eval_f2u(dst3[i]) == hk_eval_f2u(-82.0f));
      }
      for (size_t i = 0; i < mn; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_orig[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dst2[4 + i]) == hk_eval_f2u(c_orig[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dst3[4 + i]) == hk_eval_f2u(c_orig[i]));
      }
      free(a); free(b); free(bi); free(c_orig); free(c_alt); free(dst2); free(dst3);
      ralloc_free(alt); ralloc_free(pre); ralloc_free(orig);
   }

   /* 3. Pointwise clamp footprint helper (used for intervening-dispatch disjointness). */
   {
      struct hk_nir_pointwise_clamp c = {
         .input = { .set = 0, .binding = 0, .offset_B = 16 },
         .output = { .set = 0, .binding = 1, .offset_B = 32 },
         .count_offset_B = 0,
         .local_size_x = 64,
      };
      uint32_t count = 100;
      struct hk_nir_direct_shape cs = { .push = &count, .push_size_B = 4, .groups = { 2, 1, 1 } };
      uint64_t ends[2] = {0};
      HK_EVAL_CHECK(hk_nir_pointwise_clamp_footprint(&c, &cs, ends));
      HK_EVAL_CHECK(ends[0] == 16 + 400 && ends[1] == 32 + 400);
      cs.groups[0] = 1; /* insufficient coverage -> reject */
      HK_EVAL_CHECK(!hk_nir_pointwise_clamp_footprint(&c, &cs, ends));
   }
}

static void
test_g17p_shared_var_info(const struct glsl_type *type, unsigned *size,
                          unsigned *align)
{
   glsl_get_natural_size_align_bytes(type, size, align);
}

static bool
lower_hk_compute_intr_for_g17p(nir_builder *b, nir_intrinsic_instr *intr,
                               void *state)
{
   uint32_t *max_half = state;
   b->cursor = nir_before_instr(&intr->instr);
   if (intr->intrinsic == nir_intrinsic_load_push_constant) {
      uint32_t off = nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[0]);
      uint32_t base = 24u + (off / 2u);
      uint32_t end = base + intr->num_components * (intr->def.bit_size / 16u);
      if (end > *max_half)
         *max_half = end;
      nir_def *v = nir_load_preamble(b, intr->num_components, intr->def.bit_size,
                                     .base = base);
      nir_def_rewrite_uses(&intr->def, v);
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_deref_buffer_address) {
      nir_deref_instr *elem = nir_src_as_deref(intr->src[0]);
      assert(elem && elem->deref_type == nir_deref_type_array);
      nir_deref_instr *cast = nir_deref_instr_parent(elem);
      assert(cast && cast->deref_type == nir_deref_type_cast);
      nir_intrinsic_instr *desc = nir_def_as_intrinsic(cast->parent.ssa);
      assert(desc && desc->intrinsic == nir_intrinsic_load_vulkan_descriptor);
      nir_intrinsic_instr *res = nir_def_as_intrinsic(desc->src[0].ssa);
      assert(res && res->intrinsic == nir_intrinsic_vulkan_resource_index);
      uint32_t bind = nir_intrinsic_binding(res);
      uint32_t base = 8u + (bind & 3u) * 4u;
      if (base + 4u > *max_half)
         *max_half = base + 4u;
      uint32_t stride = cast->cast.ptr_stride ? cast->cast.ptr_stride
                        : glsl_get_explicit_stride(cast->type);
      nir_def *va = nir_load_preamble(b, 1, 64, .base = base);
      nir_def *off_b = nir_imul_imm(b, elem->arr.index.ssa, stride);
      nir_def *addr = nir_iadd(b, va, nir_u2u64(b, off_b));
      nir_def_rewrite_uses(&intr->def, addr);
      nir_instr_remove(&intr->instr);
      return true;
   }
   return false;
}

static bool
compile_and_check_g17p_variant(const char *label, nir_shader *nir,
                               unsigned max_gprs)
{
   uint32_t max_half = 24u;
   nir_shader_intrinsics_pass(nir, lower_hk_compute_intr_for_g17p,
                              nir_metadata_control_flow, &max_half);
   nir_opt_dce(nir);
   nir_lower_vars_to_explicit_types(nir, nir_var_mem_shared,
                                    test_g17p_shared_var_info);
   nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_offset);
   nir->info.io_lowered = true;
   agx_preprocess_nir_native(nir, AGX_CHIP_G17P);

   struct agx_shader_key key = {
      .reserved_preamble = (max_half + 3u) & ~3u,
      .dev = {
         .chip = AGX_CHIP_G17P,
         .encoding = AGX_ISA_ENCODING_AGX3B,
         .soft_fault = true,
      },
      .has_scratch = true,
      .promote_constants = true,
      .promote_textures = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   HK_EVAL_CHECK(ok && part.binary != NULL && part.info.binary_size > 0);
   struct agx_occupancy occ = agx_occupancy_for_register_count(part.info.nr_gprs);
   const unsigned min_threads =
      agx_occupancy_for_register_count(max_gprs).max_threads;
   bool pass = (part.info.scratch_size == 0 && part.info.nr_gprs <= max_gprs &&
                occ.max_threads >= min_threads);
   if (!pass) {
      fprintf(stderr,
              "G17P occupancy regression: %s gprs=%u threads=%u scratch=%u uniforms=%u\n",
              label, part.info.nr_gprs, occ.max_threads, part.info.scratch_size,
              part.info.push_count);
   }
   free(part.binary);
   return pass;
}

static void
test_g17p_compile_occupancy(void)
{
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .lower_bits = 0x00000000u,
      .has_upper = true, .upper_bits = 0x40c00000u,
   };

   /* 1. Depthwise vec4 (plain + clamp6_dcopy). */
   {
      nir_shader *orig = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
      struct hk_nir_depthwise_output info = {0};
      HK_EVAL_CHECK(hk_nir_match_depthwise_output(
         orig, nir_address_format_64bit_global_32bit_offset, &info));
      nir_shader *s0 = hk_nir_build_depthwise_vec4(orig, &info, NULL,
                                                   HK_COMPUTE_PAIR_DUAL_NONE);
      nir_shader *s1 = hk_nir_build_depthwise_vec4(orig, &info, &relu6,
                                                   HK_COMPUTE_PAIR_DUAL_COPY);
      HK_EVAL_CHECK(s0 && s1);
      HK_EVAL_CHECK(compile_and_check_g17p_variant("dw_vec4", s0, 104u));
      HK_EVAL_CHECK(compile_and_check_g17p_variant("dw_vec4_clamp6_dcopy", s1, 104u));
      ralloc_free(s0);
      ralloc_free(s1);
      ralloc_free(orig);
   }

   /* 2. GEMM cmat unguarded + guarded (16/24/32/64 + dpad4 + im2col_a + dw_a). */
   const struct hk_nir_dw_a_cfg dwa_relu6 = {
      .has_epilogue = true,
      .epilogue = relu6,
   };
   const struct hk_nir_dw_a_cfg im2expdwa_relu6 = {
      .has_epilogue = true,
      .epilogue = relu6,
      .has_expand = true,
      .has_exp_im2col = true,
      .has_exp_epilogue = true,
      .exp_epilogue = relu6,
   };
   const struct hk_nir_dw_a_cfg exp1x1dwa_relu6 = {
      .has_epilogue = true,
      .epilogue = relu6,
      .has_expand = true,
      .has_exp_im2col = false,
      .has_exp_epilogue = true,
      .exp_epilogue = relu6,
   };
   const struct hk_nir_fadd_epilogue fadd_ok = {0};
   const struct {
      const char *name;
      bool guarded;
      uint32_t tile_m;
      uint32_t tile_n;
      bool clamp;
      enum hk_compute_pair_dual_write dual;
      bool im2col_a;
      bool dw_a;
      bool im2_exp_dw_a;
      uint8_t gemm_dw_proj_n;
      bool exp1x1_dw_a;
   } cases[] = {
      { "cmat16_ug",               false, 64, 16, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat32_ug",               false, 64, 32, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat32_ug_dpad4",         false, 64, 32, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat48_ug",               false, 64, 48, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat48_ug_dpad4",         false, 64, 48, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat64x64_ug",            false, 64, 64, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat64x64_ug_dpad4",      false, 64, 64, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat64_ug_dpad4",         false, 32, 64, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat16_grd",              true,  64, 16, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat32_grd",              true,  64, 32, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat32_grd_dcopy",        true,  64, 32, true,  HK_COMPUTE_PAIR_DUAL_COPY,      false, false },
      { "cmat32_grd_dpad4",        true,  64, 32, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat64_grd",              true,  32, 64, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat64_grd_dpad4",        true,  32, 64, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat16_ug_im2a_dpad4",    false, 64, 16, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat24_ug",               false, 64, 24, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat24_ug_dcopy",         false, 64, 24, true,  HK_COMPUTE_PAIR_DUAL_COPY,      false, false },
      { "cmat24_ug_dpad4",         false, 64, 24, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      false, false },
      { "cmat24_ug_im2a_dpad4",    false, 64, 24, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat32_ug_im2a_dcopy",    false, 64, 32, true,  HK_COMPUTE_PAIR_DUAL_COPY,      true,  false },
      { "cmat32_ug_im2a_dpad4",    false, 64, 32, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat48_ug_im2a_dpad4",    false, 64, 48, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat64x64_ug_im2a_dpad4", false, 64, 64, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat64_ug_im2a_dpad4",    false, 32, 64, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat16_grd_im2a_dpad4",   true,  64, 16, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat32_grd_im2a_dcopy",   true,  64, 32, true,  HK_COMPUTE_PAIR_DUAL_COPY,      true,  false },
      { "cmat32_grd_im2a_dpad4",   true,  64, 32, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat64_grd_im2a_dpad4",   true,  32, 64, true,  HK_COMPUTE_PAIR_DUAL_PAD4,      true,  false },
      { "cmat16_ug_dwa_dcopy",     false, 64, 16, false, HK_COMPUTE_PAIR_DUAL_COPY,      false, true  },
      { "cmat24_ug_dwa_dfaddcopy", false, 64, 24, false, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, false, true  },
      { "cmat32_ug_dwa_dfaddcopy", false, 64, 32, false, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, false, true  },
      { "cmat48_ug_dwa_dfaddcopy", false, 64, 48, false, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, false, true  },
      { "cmat64x64_ug_dwa_dfaddcopy", false, 64, 64, false, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, false, true },
      { "cmat64_ug_dwa_dfaddcopy", false, 32, 64, false, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, false, true  },
      { "cmat16_ug_im2expdwa_dcopy", false, 64, 16, false, HK_COMPUTE_PAIR_DUAL_COPY,      false, false, true },
      { "cmat24_ug_exp1x1dwa_dcopy", false, 64, 24, false, HK_COMPUTE_PAIR_DUAL_COPY,      false, false, false, 0, true },
      { "cmat32x32_ug",            false, 32, 32, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat32x32_ug_im2a_dcopy", false, 32, 32, true,  HK_COMPUTE_PAIR_DUAL_COPY,      true,  false },
      { "cmat32x32_ug_dwa_dcopy",  false, 32, 32, false, HK_COMPUTE_PAIR_DUAL_COPY,      false, true  },
      { "cmat32x16_ug",            false, 32, 16, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat32x16_ug_im2a_dcopy", false, 32, 16, true,  HK_COMPUTE_PAIR_DUAL_COPY,      true,  false },
      { "cmat32x16_ug_dwa_dcopy",  false, 32, 16, false, HK_COMPUTE_PAIR_DUAL_COPY,      false, true  },
      { "cmat16x32_ug",            false, 16, 32, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat16x32_ug_im2a_dcopy", false, 16, 32, true,  HK_COMPUTE_PAIR_DUAL_COPY,      true,  false },
      { "cmat16x16_ug",            false, 16, 16, true,  HK_COMPUTE_PAIR_DUAL_NONE,      false, false },
      { "cmat16x16_ug_im2a_dcopy", false, 16, 16, true,  HK_COMPUTE_PAIR_DUAL_COPY,      true,  false },
      { "cmat16_ug_gemm_dw",       false, 64, 16, true,  HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false },
      { "cmat24_ug_gemm_dw",       false, 64, 24, true,  HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false },
      { "cmat32_ug_gemm_dw",       false, 64, 32, true,  HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false },
      { "cmat48_ug_gemm_dw",       false, 64, 48, true,  HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false },
      { "cmat48_ug_gemm_dw_proj24", false, 64, 48, true, HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false, false, 24 },
      { "cmat24_ug_gemm_dw_proj32", false, 64, 24, true, HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false, false, 32 },
      { "cmat32_ug_gemm_dw_proj64", false, 64, 32, true, HK_COMPUTE_PAIR_DUAL_GEMM_DW,   false, false, false, 64 },
   };
   bool all_ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const struct gemm_test_opts opts = {
         .guarded = cases[i].guarded, .has_bias_buf = true,
         .has_bias_mode = true, .use_ffma = true,
      };
      nir_shader *orig = build_tiled_gemm_nir_opts(&opts);
      run_hk_preprocess_for_test(orig, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         orig, nir_address_format_64bit_global_32bit_offset, &m));
      struct hk_pack_loop_bound lb = {0};
      const struct hk_nir_fadd_epilogue *fe =
         hk_compute_pair_dual_is_fadd_ep(cases[i].dual) ? &fadd_ok : NULL;
      struct hk_nir_dw_a_cfg gdw_cfg = dwa_relu6;
      if (cases[i].gemm_dw_proj_n) {
         gdw_cfg.has_proj = true;
         gdw_cfg.proj_tile_n = cases[i].gemm_dw_proj_n;
      }
      const struct hk_nir_dw_a_cfg *gdw =
         cases[i].dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW ? &gdw_cfg : NULL;
      const struct hk_nir_dw_a_cfg *dwa =
         cases[i].im2_exp_dw_a ? &im2expdwa_relu6 :
         (cases[i].exp1x1_dw_a ? &exp1x1dwa_relu6 :
          (cases[i].dw_a ? &dwa_relu6 : NULL));
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         orig, &m, cases[i].tile_m, cases[i].tile_n, cases[i].clamp ? &relu6 : NULL,
         cases[i].dual, fe, cases[i].im2col_a, dwa, gdw, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      /* With Dynamic Caching ILP headroom on AGX3B, 0-scratch heavy variants
       * trade small GPR increases for higher ILP and tile reuse:
       *  - core small GEMM: <= 104 GPRs (1024t)
       *  - im2col_a / dw_a / gemm_dw / cmat48: <= 128 GPRs (832t)
       *  - 64x64 (16 accs/SG): <= 136 GPRs (768t = 6 WGs/core; 144 for dw_a) */
      const unsigned max_gprs =
         cases[i].gemm_dw_proj_n
            ? (cases[i].tile_n == 48 ? 160u : 152u) :
         cases[i].exp1x1_dw_a ? 160u :
         cases[i].im2_exp_dw_a ? 136u :
         (cases[i].tile_m == 64 && cases[i].tile_n == 64)
            ? (cases[i].dw_a ? 144u : 136u) :
         cases[i].dw_a
            ? (cases[i].tile_n >= 24 ? 152u : 136u) :
         (cases[i].im2col_a ||
          cases[i].dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW ||
          (cases[i].tile_m == 64 && cases[i].tile_n == 48))
            ? 128u : 104u;
      if (!compile_and_check_g17p_variant(cases[i].name, alt, max_gprs))
         all_ok = false;
      ralloc_free(alt);
      ralloc_free(orig);
   }
   HK_EVAL_CHECK(all_ok);
}

static void
test_gemm_im2col_a_fusion(void)
{
   /* Bit-exact NIR-eval verification of fused im2col-into-A across:
    *   - unguarded vec4 unit-stride border (K%16==0 and K%8 tail)
    *   - unguarded vec4 stride-2 borderless (multi-gx to test wg_col==0 dst write)
    *   - guarded scalar ch=3 kw=9 kh=3 (K=27, non-mult-4) + clamp + DUAL_COPY
    *   - guarded vec4 border with dst_va == 0 (dead write elision)
    *   - unguarded scalar (ch=2 kw=6 kh=4 -> K=24, K%8 tail) */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct {
      bool guarded;
      uint32_t tile_m;
      uint32_t tile_n;
      uint32_t in_h, in_w_px, ch, kh, kw_px, sh, sw, pad_y, pad_x_px;
      uint32_t N;
      bool with_clamp;
      bool with_dcopy;
      bool write_im2_dst;
   } cases[] = {
      /* 0: unguarded cmat32, vec4, borderless, M=64 (8x8), K=16, N=64 */
      { false, 64, 32, 9, 9, 4, 2, 2, 1, 1, 0, 0, 64, true, false, true },
      /* 1: unguarded cmat64, vec4, unit border, M=64 (8x8), K=24 (K%8 tail!), multi-gx (N=128) */
      { false, 32, 64, 8, 8, 8, 1, 3, 1, 1, 0, 1, 128, false, true, true },
      /* 2: unguarded cmat16, vec4, stride-2, M=64 (8x8), K=24 (K%8 tail!), multi-gx (N=32) */
      { false, 64, 16, 17, 15, 8, 3, 1, 2, 2, 0, 0, 32, true, true, false },
      /* 3: guarded cmat32, scalar ch=3 kw=9 kh=3 K=27 (Face Detection CMD 1->2 shape!), multi-gx (N=48) */
      { true,  64, 32, 9, 9, 3, 3, 3, 1, 1, 1, 1, 48, true, true, true },
      /* 4: guarded cmat16, vec4 border, odd M=35 (5x7), N=21, K=36, dst_va == 0 */
      { true,  64, 16, 5, 7, 4, 3, 3, 1, 1, 1, 1, 21, false, false, false },
      /* 5: unguarded cmat32, scalar ch=2 kw=6 kh=4 M=64 (8x8), K=24 (non-vec4 on unguarded!) */
      { false, 64, 32, 9, 8, 2, 4, 3, 1, 1, 1, 1, 64, false, false, true },
      /* 6: unguarded cmat24, vec4 ch=4 kw=12 kh=2 M=64 (8x8), K=24 (K%8 tail!), N=48 (multi-gx) */
      { false, 64, 24, 9, 10, 4, 2, 3, 1, 1, 0, 0, 48, true, true, true },
      /* 7: unguarded cmat48, vec4, M=64 (8x8), K=24 (K%8 tail!), N=96 (multi-gx) */
      { false, 64, 48, 8, 8, 8, 1, 3, 1, 1, 0, 1, 96, true, true, true },
      /* 8: unguarded cmat64x64, vec4, M=64 (8x8), K=32, N=64 */
      { false, 64, 64, 9, 9, 4, 2, 4, 1, 1, 0, 1, 64, true, true, true },
      /* 9: unguarded cmat48, COPY1X1 fast path, M=64 (64x1), K=32, N=96 */
      { false, 64, 48, 64, 1, 32, 1, 1, 1, 1, 0, 0, 96, true, true, true },
      /* 10: unguarded cmat32x32, vec4, M=64 (8x8), K=48 (K%32==16 tail), N=32 */
      { false, 32, 32, 9, 8, 8, 2, 3, 1, 1, 0, 1, 32, true, true, true },
      /* 11: unguarded cmat32x16, vec4, M=32 (4x8), K=48 (K%32==16 tail), N=16 */
      { false, 32, 16, 5, 8, 8, 2, 3, 1, 1, 0, 1, 16, true, false, true },
      /* 12: unguarded cmat16x32, vec4, M=16 (4x4), K=40 (K%32==8 tail), N=64 (multi-gx) */
      { false, 16, 32, 4, 4, 8, 5, 1, 1, 1, 2, 0, 64, true, true, true },
      /* 13: unguarded cmat16x16, vec4, M=48 (6x8), K=32, N=16 */
      { false, 16, 16, 7, 9, 4, 2, 4, 1, 1, 0, 1, 16, false, true, true },
      /* 14: unguarded cmat16x16, scalar non-vec4, M=16 (4x4), K=24 (hoisted yb/xb + K-tail) */
      { false, 16, 16, 5, 4, 2, 4, 3, 1, 1, 1, 1, 16, false, false, true },
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const uint32_t iw = cases[ci].in_w_px * cases[ci].ch;
      const uint32_t ih = cases[ci].in_h;
      const uint32_t kw = cases[ci].kw_px * cases[ci].ch;
      const uint32_t kh = cases[ci].kh;
      const uint32_t pad_x = cases[ci].pad_x_px * cases[ci].ch;
      const uint32_t pad_y = cases[ci].pad_y;
      const uint32_t step_x = cases[ci].sw * cases[ci].ch;
      const uint32_t sh = cases[ci].sh;
      const uint32_t bordered_w_px =
         cases[ci].in_w_px + (cases[ci].sw == 1 && cases[ci].sh == 1 ? 2u * cases[ci].pad_x_px : 0u);
      const uint32_t bordered_h =
         cases[ci].in_h + (cases[ci].sw == 1 && cases[ci].sh == 1 ? 2u * cases[ci].pad_y : 0u);
      const uint32_t cols = (bordered_w_px - cases[ci].kw_px) / cases[ci].sw + 1u;
      const uint32_t rows_y = (bordered_h - kh) / sh + 1u;
      const uint32_t M = rows_y * cols;
      const uint32_t K = kh * kw;
      const uint32_t N = cases[ci].N;
      const uint32_t tile_n = cases[ci].tile_n;
      const uint32_t tile_m = cases[ci].tile_m;

      const struct gemm_test_opts opts = {
         .guarded = cases[ci].guarded, .has_bias_buf = true,
         .has_bias_mode = true, .use_ffma = true,
      };
      const struct gemm_test_opts ref_opts = {
         .guarded = true, .has_bias_buf = true,
         .has_bias_mode = true, .use_ffma = true,
      };
      nir_shader *ref_orig = build_tiled_gemm_nir_opts(&ref_opts);
      nir_shader *pre = build_tiled_gemm_nir_opts(&opts);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));

      const enum hk_compute_pair_dual_write dw =
         cases[ci].with_dcopy ? HK_COMPUTE_PAIR_DUAL_COPY : HK_COMPUTE_PAIR_DUAL_NONE;
      const struct hk_nir_clamp_epilogue *ep = cases[ci].with_clamp ? &relu6 : NULL;
      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         pre, &m, tile_m, tile_n, ep, dw, NULL, true, NULL, NULL, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_im2a");

      const size_t src_elems = (size_t)ih * iw;
      const size_t a_elems = (size_t)M * K;
      const size_t b_elems = (size_t)K * N;
      const size_t c_elems = (size_t)M * N;
      float *src_buf = calloc(src_elems, 4);
      float *a_ref = calloc(a_elems, 4);
      float *a_out = calloc(a_elems, 4);
      float *a_dummy = calloc(a_elems, 4);
      float *b_buf = calloc(b_elems, 4);
      float *bias_buf = calloc(N, 4);
      float *c_ref = calloc(c_elems, 4);
      float *c_alt = calloc(c_elems, 4);
      float *dst2 = calloc(c_elems, 4);

      uint32_t rng = 0x7100u + ci * 17u;
      for (size_t i = 0; i < src_elems; ++i) src_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) bias_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < a_elems; ++i) { a_out[i] = -99.5f; a_dummy[i] = -55.25f; }

      /* Reference im2col on CPU. */
      for (uint32_t gr = 0; gr < M; ++gr) {
         const uint32_t oy = gr / cols;
         const uint32_t ox = gr % cols;
         for (uint32_t gk = 0; gk < K; ++gk) {
            const uint32_t ky = gk / kw;
            const uint32_t kx = gk % kw;
            const uint32_t y_eff = oy * sh + ky - pad_y;
            const uint32_t x_eff = ox * step_x + kx - pad_x;
            float v = 0.0f;
            if (y_eff < ih && x_eff < iw)
               v = src_buf[(size_t)y_eff * iw + x_eff];
            a_ref[(size_t)gr * K + gk] = v;
         }
      }

      /* Reference GEMM using original NIR on a_ref. */
      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
      };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      struct hk_eval_ssbo so[4] = {
         { 0, 0, a_ref, a_elems * 4, 0 }, { 0, 1, b_buf, b_elems * 4, 0 },
         { 0, 2, bias_buf, (size_t)N * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
      };
      hk_eval_dispatch_wg(ref_orig, og, so, 4, &push, sizeof(push));
      if (ep) {
         for (size_t i = 0; i < c_elems; ++i) {
            float v = c_ref[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 6.0f) v = 6.0f;
            c_ref[i] = v;
         }
      }

      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));
      struct hk_eval_ssbo sa[7];
      unsigned n_sa = 0;
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 0, a_dummy, a_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 1, b_buf, b_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 2, bias_buf, (size_t)N * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 3, c_alt, c_elems * 4, 0 };
      if (cases[ci].with_dcopy) {
         const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
         const uint64_t d2_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 9, dst2, c_elems * 4, 0 };
         struct hk_dual_copy_push dp = {
            .dst2_va_lo = (uint32_t)d2_va, .dst2_va_hi = (uint32_t)(d2_va >> 32),
         };
         memcpy(push_buf + dual_off, &dp, sizeof(dp));
      }
      const uint64_t src_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 10, src_buf, src_elems * 4, 0 };
      uint64_t dst_va = 0;
      if (cases[ci].write_im2_dst) {
         dst_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 11, a_out, a_elems * 4, 0 };
      }
      const uint32_t im2_off = hk_im2col_a_push_offset(m.push_bytes, dw);
      const bool vec4_ok = ((iw | kw | step_x | pad_x) & 3u) == 0;
      const bool is_copy1x1 =
         vec4_ok && cases[ci].kh == 1 && cases[ci].kw_px == 1 &&
         cases[ci].sh == 1 && cases[ci].sw == 1 &&
         cases[ci].pad_x_px == 0 && cases[ci].pad_y == 0 &&
         iw == kw && cols == 1u;
      struct hk_im2col_a_push ip = {
         .src_va_lo = (uint32_t)src_va, .src_va_hi = (uint32_t)(src_va >> 32),
         .dst_va_lo = (uint32_t)dst_va, .dst_va_hi = (uint32_t)(dst_va >> 32),
         .iw = iw, .ih = ih, .kw = kw, .cols = cols,
         .step_x = step_x, .sh = sh, .pad_x = pad_x, .pad_y = pad_y,
         .flags = (vec4_ok ? HK_IM2COL_A_FLAG_VEC4 : 0u) |
                  (is_copy1x1 ? HK_IM2COL_A_FLAG_COPY1X1 : 0u),
      };
      hk_im2col_a_push_init_udiv(&ip);
      memcpy(push_buf + im2_off, &ip, sizeof(ip));

      const uint32_t ag[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, tile_m), 1u };
      hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, im2_off + sizeof(ip));

      for (size_t i = 0; i < c_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));
         if (cases[ci].with_dcopy)
            HK_EVAL_CHECK(hk_eval_f2u(dst2[i]) == hk_eval_f2u(c_ref[i]));
      }
      for (size_t i = 0; i < a_elems; ++i) {
         if (cases[ci].write_im2_dst)
            HK_EVAL_CHECK(hk_eval_f2u(a_out[i]) == hk_eval_f2u(a_ref[i]));
         else
            HK_EVAL_CHECK(hk_eval_f2u(a_out[i]) == hk_eval_f2u(-99.5f));
      }

      if (cases[ci].with_dcopy) {
         /* Combined SKIP_C + elided im2col dst_va == 0. */
         for (size_t i = 0; i < c_elems; ++i) {
            c_alt[i] = -88.5f;
            dst2[i] = -77.5f;
         }
         for (size_t i = 0; i < a_elems; ++i)
            a_out[i] = -99.5f;
         const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
         const uint64_t d2_va = hk_eval_make_vaddr(4, 0);
         struct hk_dual_copy_push dp = {
            .dst2_va_lo = (uint32_t)d2_va, .dst2_va_hi = (uint32_t)(d2_va >> 32),
            .flags = HK_DUAL_WRITE_FLAG_SKIP_C,
         };
         memcpy(push_buf + dual_off, &dp, sizeof(dp));
         ip.dst_va_lo = 0;
         ip.dst_va_hi = 0;
         memcpy(push_buf + im2_off, &ip, sizeof(ip));
         hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, im2_off + sizeof(ip));
         for (size_t i = 0; i < c_elems; ++i) {
            HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(-88.5f));
            HK_EVAL_CHECK(hk_eval_f2u(dst2[i]) == hk_eval_f2u(c_ref[i]));
         }
         for (size_t i = 0; i < a_elems; ++i)
            HK_EVAL_CHECK(hk_eval_f2u(a_out[i]) == hk_eval_f2u(-99.5f));
      }

      free(src_buf); free(a_ref); free(a_out); free(a_dummy);
      free(b_buf); free(bias_buf); free(c_ref); free(c_alt); free(dst2);
      ralloc_free(alt); ralloc_free(pre); ralloc_free(ref_orig);
   }
}

static void
test_gemm_dw_a_fusion(void)
{
   /* Bit-exact NIR-eval verification of fused 3x3 DW-into-A across:
    *   - s=1 px=-1,py=-1 + bias + relu6 + dst1 + dst2 + cmat16_ug + dcopy
    *   - s=2 px=0,py=0   + bias + relu6 + dst1 + dst2 + cmat24_ug
    *   - s=1 px=-1,py=-1 + no-bias + no-clamp + dst1 only + cmat32_ug
    *   - s=1 px=-1,py=-1 + bias + relu6 + dst1 + dst2 + cmat64_ug (K=32) */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct {
      uint32_t in_h, in_w, out_h, out_w, K, N, tile_m, tile_n;
      int32_t sx, px, py;
      bool has_bias, dw_clamp, has_dst2, gemm_dcopy;
   } cases[] = {
      {  8,  8,  8,  8, 16, 16, 64, 16, 1, -1, -1, true,  true,  true,  true  },
      { 17, 17,  8,  8, 16, 24, 64, 24, 2,  0,  0, true,  true,  true,  false },
      {  8,  8,  8,  8, 16, 32, 64, 32, 1, -1, -1, false, false, false, false },
      { 10, 10,  8,  8, 16, 48, 64, 48, 1,  0,  0, true,  true,  true,  true  },
      {  8,  8,  8,  8, 32, 64, 64, 64, 1, -1, -1, true,  true,  true,  true  },
      {  8,  8,  8,  8, 32, 64, 32, 64, 1, -1, -1, true,  true,  true,  true  },
      /* K_step=32 small tiles + odd widths exercising both_int_s1 pair reuse and row-wrap fallback. */
      {  8,  8,  8,  8, 32, 16, 32, 16, 1, -1, -1, true,  true,  true,  true  },
      {  8,  8,  8,  8, 32, 32, 32, 32, 1, -1, -1, false, true,  false, true  },
      {  6,  8,  6,  8, 32, 16, 16, 16, 1, -1, -1, true,  true,  true,  false },
      {  6,  8,  6,  8, 32, 32, 16, 32, 1, -1, -1, true,  false, false, true  },
      /* Odd out_w=5 (M=35? no, M=80 = 16x5): exercises odd-width row-wrap + interior pairs. */
      { 16,  5, 16,  5, 32, 16, 16, 16, 1, -1, -1, true,  true,  true,  true  },
      /* Narrow in_w=3 (M=48 = 16x3): exercises in_w < 4 fallback to rr_lp. */
      { 16,  3, 16,  3, 32, 16, 16, 16, 1, -1, -1, true,  true,  false, false },
      /* Stride-2 on K_step=32 tile: exercises !is_s1 fallback. */
      { 17, 17,  8,  8, 32, 32, 32, 32, 2,  0,  0, true,  true,  true,  false },
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const uint32_t in_h = cases[ci].in_h, in_w = cases[ci].in_w;
      const uint32_t out_h = cases[ci].out_h, out_w = cases[ci].out_w;
      const uint32_t M = out_h * out_w, K = cases[ci].K, N = cases[ci].N;
      const uint32_t tile_n = cases[ci].tile_n;
      const uint32_t tile_m = cases[ci].tile_m;

      const struct gemm_test_opts ref_opts = {
         .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
      };
      const struct gemm_test_opts opts = {
         .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
      };
      nir_shader *ref_orig = build_tiled_gemm_nir_opts(&ref_opts);
      nir_shader *pre = build_tiled_gemm_nir_opts(&opts);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));

      struct hk_nir_dw_a_cfg dwa_cfg = {
         .has_epilogue = cases[ci].dw_clamp,
         .epilogue = relu6,
      };
      const enum hk_compute_pair_dual_write dw =
         cases[ci].gemm_dcopy ? HK_COMPUTE_PAIR_DUAL_COPY : HK_COMPUTE_PAIR_DUAL_NONE;
      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         pre, &m, tile_m, tile_n, NULL, dw, NULL, false, &dwa_cfg, NULL, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_dwa");

      const size_t in_elems = (size_t)in_h * in_w * K;
      const size_t wt_elems = 9u * K;
      const size_t a_elems = (size_t)M * K;
      const size_t b_elems = (size_t)K * N;
      const size_t c_elems = (size_t)M * N;
      float *in_buf = calloc(in_elems, 4);
      float *wt_buf = calloc(wt_elems, 4);
      float *dw_bias = calloc(K, 4);
      float *a_ref = calloc(a_elems, 4);
      float *dw_d1 = calloc(a_elems, 4);
      float *dw_d2 = calloc(a_elems, 4);
      float *a_dummy = calloc(a_elems, 4);
      float *b_buf = calloc(b_elems, 4);
      float *g_bias = calloc(N, 4);
      float *c_ref = calloc(c_elems, 4);
      float *c_alt = calloc(c_elems, 4);
      float *g_dst2 = calloc(c_elems, 4);

      uint32_t rng = 0x8200u + ci * 31u;
      for (size_t i = 0; i < in_elems; ++i) in_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < wt_elems; ++i) wt_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < K; ++i) dw_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) g_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < a_elems; ++i) {
         dw_d1[i] = -99.5f; dw_d2[i] = -77.5f; a_dummy[i] = -55.25f;
      }

      /* CPU reference DW 3x3 (+bias +clamp) in exact (ky=0..2, kx=0..2) order. */
      for (uint32_t oy = 0; oy < out_h; ++oy) {
         for (uint32_t ox = 0; ox < out_w; ++ox) {
            const int32_t ix0 = (int32_t)ox * cases[ci].sx + cases[ci].px;
            const int32_t iy0 = (int32_t)oy * cases[ci].sx + cases[ci].py;
            for (uint32_t c = 0; c < K; ++c) {
               float acc = 0.0f;
               for (int32_t ky = 0; ky < 3; ++ky) {
                  const int32_t iy = iy0 + ky;
                  if (iy < 0 || iy >= (int32_t)in_h) continue;
                  for (int32_t kx = 0; kx < 3; ++kx) {
                     const int32_t ix = ix0 + kx;
                     if (ix < 0 || ix >= (int32_t)in_w) continue;
                     const float iv = in_buf[((size_t)iy * in_w + ix) * K + c];
                     const float wv = wt_buf[(size_t)(ky * 3 + kx) * K + c];
                     acc = ref_mul_add_unfused(iv, wv, acc);
                  }
               }
               if (cases[ci].has_bias) acc += dw_bias[c];
               if (cases[ci].dw_clamp) {
                  acc = fmaxf(acc, 0.0f);
                  acc = fminf(acc, 6.0f);
               }
               a_ref[(size_t)(oy * out_w + ox) * K + c] = acc;
            }
         }
      }

      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
      };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      struct hk_eval_ssbo so[4] = {
         { 0, 0, a_ref, a_elems * 4, 0 }, { 0, 1, b_buf, b_elems * 4, 0 },
         { 0, 2, g_bias, (size_t)N * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
      };
      hk_eval_dispatch_wg(ref_orig, og, so, 4, &push, sizeof(push));

      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));
      struct hk_eval_ssbo sa[10];
      unsigned n_sa = 0;
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 0, a_dummy, a_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 1, b_buf, b_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 2, g_bias, (size_t)N * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 3, c_alt, c_elems * 4, 0 };
      if (cases[ci].gemm_dcopy) {
         const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
         const uint64_t gd2_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 9, g_dst2, c_elems * 4, 0 };
         struct hk_dual_copy_push dp = {
            .dst2_va_lo = (uint32_t)gd2_va, .dst2_va_hi = (uint32_t)(gd2_va >> 32),
         };
         memcpy(push_buf + dual_off, &dp, sizeof(dp));
      }
      const uint64_t in_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 10, in_buf, in_elems * 4, 0 };
      const uint64_t wt_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 11, wt_buf, wt_elems * 4, 0 };
      uint64_t bias_va = 0;
      if (cases[ci].has_bias) {
         bias_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 12, dw_bias, (size_t)K * 4, 0 };
      }
      const uint64_t dst1_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 13, dw_d1, a_elems * 4, 0 };
      uint64_t dst2_va = 0;
      if (cases[ci].has_dst2) {
         dst2_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 14, dw_d2, a_elems * 4, 0 };
      }

      const uint32_t dw_off = hk_dw_a_push_offset(m.push_bytes, dw);
      const uint32_t flags =
         (cases[ci].has_bias ? HK_DW_A_FLAG_HAS_BIAS : 0u) |
         HK_DW_A_FLAG_HAS_DST1 |
         (cases[ci].has_dst2 ? HK_DW_A_FLAG_HAS_DST2 : 0u) |
         (cases[ci].sx == 2 ? HK_DW_A_FLAG_STRIDE2 : 0u);
      struct hk_dw_a_push dp = {
         .in_va_lo = (uint32_t)in_va, .in_va_hi = (uint32_t)(in_va >> 32),
         .wt_va_lo = (uint32_t)wt_va, .wt_va_hi = (uint32_t)(wt_va >> 32),
         .bias_va_lo = (uint32_t)bias_va, .bias_va_hi = (uint32_t)(bias_va >> 32),
         .dst1_va_lo = (uint32_t)dst1_va, .dst1_va_hi = (uint32_t)(dst1_va >> 32),
         .dst2_va_lo = (uint32_t)dst2_va, .dst2_va_hi = (uint32_t)(dst2_va >> 32),
         .in_wh = in_w | (in_h << 16),
         .out_wh = out_w | (out_h << 16),
         .pad_flags = ((uint32_t)(uint8_t)(int8_t)cases[ci].px) |
                      (((uint32_t)(uint8_t)(int8_t)cases[ci].py) << 8) |
                      (flags << 16),
      };
      hk_dw_a_push_init_udiv(&dp, out_w);
      memcpy(push_buf + dw_off, &dp, sizeof(dp));

      const uint32_t ag[3] = { DIV_ROUND_UP(N, tile_n), DIV_ROUND_UP(M, tile_m), 1u };
      hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, dw_off + sizeof(dp));

      const uint32_t k_step = (tile_m <= 32) ? 32u : 16u;
      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, k_step);
      uint64_t bound = 0, eff = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb, params, &bound));
      HK_EVAL_CHECK(bound >= hk_eval_last_max_loop_backedges + 1);
      HK_EVAL_CHECK(hk_pack_eval_effective_instrs(&lb, 200u, params, &eff));
      HK_EVAL_CHECK(eff > 0 && eff < 200000u);

      for (size_t i = 0; i < c_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));
         if (cases[ci].gemm_dcopy)
            HK_EVAL_CHECK(hk_eval_f2u(g_dst2[i]) == hk_eval_f2u(c_ref[i]));
      }
      for (size_t i = 0; i < a_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(dw_d1[i]) == hk_eval_f2u(a_ref[i]));
         if (cases[ci].has_dst2)
            HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(a_ref[i]));
         else
            HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(-77.5f));
      }

      free(in_buf); free(wt_buf); free(dw_bias); free(a_ref);
      free(dw_d1); free(dw_d2); free(a_dummy); free(b_buf);
      free(g_bias); free(c_ref); free(c_alt); free(g_dst2);
      ralloc_free(alt); ralloc_free(pre); ralloc_free(ref_orig);
   }
}

static void
test_gemm_dw_fusion(void)
{
   /* Bit-exact NIR-eval verification of fused GEMM (+clamp) -> PAD4 -> DW 3x3
    * s=2 (+bias +clamp +copy4) across:
    *   - cmat16_ug, cmat24_ug (K=24, K%16==8 tail!), cmat32_ug (K=16 & K=32)
    *   - 16x16 (2x2 spatial tiles -> interior halo + right/bottom/corner zero halo)
    *   - GEMM bias modes (col bias, row bias) + GEMM clamp + DW bias/no-bias + DW clamp + dst2
    *   - pad_b == {0,0,0,0} fast path AND non-zero / negative pad_b fallback paths
    *   - SKIP_C / SKIP_PAD_DST / SKIP_DW_DST1 / SKIP_DW_DST2 flags
    *   - loop bound & work_valid check on 320x320x96 -> 160x160x96 FD shape. */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct {
      uint32_t in_h, in_w, K, N, tile_n;
      int32_t gemm_bias_mode;
      bool gemm_clamp, dw_bias, dw_clamp, dw_dst2;
      int32_t pad_b[4];
      uint32_t skip_flags;
   } cases[] = {
      /* 0: cmat32_ug 16x16 (2x2 spatial tiles) K=16 N=64 fast path, full chain */
      { 16, 16, 16, 64, 32, 1, true,  true,  true,  true,  { 0,  0, 0, 0 }, 0 },
      /* 1: cmat24_ug 16x16 K=24 (K%16==8 tail!) N=48 fast path, row bias */
      { 16, 16, 24, 48, 24, 0, true,  true,  true,  true,  { 0,  0, 0, 0 }, 0 },
      /* 2: cmat16_ug 8x16 K=16 N=32 fast path, no DW bias/clamp/dst2 */
      {  8, 16, 16, 32, 16, 1, false, false, false, false, { 0,  0, 0, 0 }, 0 },
      /* 3: cmat32_ug 8x8 K=32 N=32 fast path with SKIP_C + SKIP_PAD_DST + SKIP_DW_DST1 */
      {  8,  8, 32, 32, 32, 1, true,  true,  true,  true,  { 0,  0, 0, 0 },
        HK_GEMM_DW_FLAG_SKIP_C | HK_GEMM_DW_FLAG_SKIP_PAD_DST | HK_GEMM_DW_FLAG_SKIP_DW_DST1 },
      /* 4: cmat32_ug 16x16 K=16 N=32 non-zero pad_b={0,1,1,0} fallback */
      { 16, 16, 16, 32, 32, 1, true,  true,  true,  true,  { 0,  1, 1, 0 }, 0 },
      /* 5: cmat24_ug 8x8 K=24 N=24 negative pad_b={0,-1,0,0} fallback + SKIP_C + SKIP_PAD_DST */
      {  8,  8, 24, 24, 24, 1, true,  true,  false, true,  { 0, -1, 0, 0 },
        HK_GEMM_DW_FLAG_SKIP_C | HK_GEMM_DW_FLAG_SKIP_PAD_DST },
      /* 6: cmat48_ug 16x16 K=16 N=96 fast path (FD ord=2 shape!) */
      { 16, 16, 16, 96, 48, 1, true,  true,  true,  true,  { 0,  0, 0, 0 }, 0 },
      /* 7: cmat48_ug 16x16 K=24 (K%16==8 tail!) N=144 fast path, row bias (FD ord=6 shape!) */
      { 16, 16, 24, 144, 48, 0, true, true,  true,  true,  { 0,  0, 0, 0 }, 0 },
      /* 8: cmat48_ug 8x8 K=32 N=96 non-zero pad_b={0,1,1,0} fallback */
      {  8,  8, 32, 96, 48, 1, true,  true,  true,  true,  { 0,  1, 1, 0 }, 0 },
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const uint32_t in_h = cases[ci].in_h, in_w = cases[ci].in_w;
      const uint32_t out_h = in_h + 1u, out_w = in_w + 1u;
      const uint32_t dw_oh = in_h / 2u, dw_ow = in_w / 2u;
      const uint32_t M = in_h * in_w, K = cases[ci].K, N = cases[ci].N;
      const uint32_t tile_n = cases[ci].tile_n;

      const struct gemm_test_opts ref_opts = {
         .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
      };
      const struct gemm_test_opts opts = {
         .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
      };
      nir_shader *ref_orig = build_tiled_gemm_nir_opts(&ref_opts);
      nir_shader *pre = build_tiled_gemm_nir_opts(&opts);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));

      struct hk_nir_dw_a_cfg gdw_cfg = {
         .has_epilogue = cases[ci].dw_clamp,
         .epilogue = relu6,
      };
      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_with_bound(
         pre, &m, tile_n, cases[ci].gemm_clamp ? &relu6 : NULL,
         HK_COMPUTE_PAIR_DUAL_GEMM_DW, NULL, false, NULL, &gdw_cfg, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_gemmdw");

      const size_t a_elems = (size_t)M * K;
      const size_t b_elems = (size_t)K * N;
      const size_t c_elems = (size_t)M * N;
      const size_t pad_elems = (size_t)out_h * out_w * N;
      const size_t wt_elems = 9u * N;
      const size_t dw_elems = (size_t)dw_oh * dw_ow * N;
      const size_t bias_elems = MAX2(M, N);

      float *a_buf = calloc(a_elems, 4);
      float *b_buf = calloc(b_elems, 4);
      float *g_bias = calloc(bias_elems, 4);
      float *c_ref = calloc(c_elems, 4);
      float *c_alt = calloc(c_elems, 4);
      float *pad_ref = calloc(pad_elems, 4);
      float *pad_alt = calloc(pad_elems, 4);
      float *dw_wt = calloc(wt_elems, 4);
      float *dw_bias = calloc(N, 4);
      float *dw_ref = calloc(dw_elems, 4);
      float *dw_d1 = calloc(dw_elems, 4);
      float *dw_d2 = calloc(dw_elems, 4);

      uint32_t rng = 0x9300u + ci * 29u;
      for (size_t i = 0; i < a_elems; ++i) a_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < bias_elems; ++i) g_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < wt_elems; ++i) dw_wt[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) dw_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < c_elems; ++i) c_alt[i] = -88.5f;
      for (size_t i = 0; i < pad_elems; ++i) pad_alt[i] = -77.5f;
      for (size_t i = 0; i < dw_elems; ++i) { dw_d1[i] = -66.5f; dw_d2[i] = -55.5f; }

      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M,
         .bias_mode = cases[ci].gemm_bias_mode,
      };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      struct hk_eval_ssbo so[4] = {
         { 0, 0, a_buf, a_elems * 4, 0 }, { 0, 1, b_buf, b_elems * 4, 0 },
         { 0, 2, g_bias, bias_elems * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
      };
      hk_eval_dispatch_wg(ref_orig, og, so, 4, &push, sizeof(push));
      if (cases[ci].gemm_clamp) {
         for (size_t i = 0; i < c_elems; ++i) {
            float v = c_ref[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 6.0f) v = 6.0f;
            c_ref[i] = v;
         }
      }

      /* CPU reference PAD4 with runtime pad_b. */
      for (uint32_t oy = 0; oy < out_h; ++oy) {
         for (uint32_t ox = 0; ox < out_w; ++ox) {
            for (uint32_t oc = 0; oc < N; ++oc) {
               const int64_t iy = (int64_t)oy - cases[ci].pad_b[1];
               const int64_t ix = (int64_t)ox - cases[ci].pad_b[2];
               const int64_t ic = (int64_t)oc - cases[ci].pad_b[3];
               float v = 0.0f;
               if (cases[ci].pad_b[0] == 0 &&
                   iy >= 0 && iy < (int64_t)in_h &&
                   ix >= 0 && ix < (int64_t)in_w &&
                   ic >= 0 && ic < (int64_t)N) {
                  v = c_ref[((size_t)iy * in_w + (size_t)ix) * N + (size_t)ic];
               }
               pad_ref[((size_t)oy * out_w + ox) * N + oc] = v;
            }
         }
      }

      /* CPU reference DW 3x3 s=2 (+bias +clamp) reading pad_ref. */
      for (uint32_t oy = 0; oy < dw_oh; ++oy) {
         for (uint32_t ox = 0; ox < dw_ow; ++ox) {
            for (uint32_t c = 0; c < N; ++c) {
               float acc = 0.0f;
               for (uint32_t ky = 0; ky < 3; ++ky) {
                  for (uint32_t kx = 0; kx < 3; ++kx) {
                     const float iv = pad_ref[((size_t)(oy * 2u + ky) * out_w + (ox * 2u + kx)) * N + c];
                     const float wv = dw_wt[(size_t)(ky * 3u + kx) * N + c];
                     acc = ref_mul_add_unfused(iv, wv, acc);
                  }
               }
               if (cases[ci].dw_bias)
                  acc += dw_bias[c];
               if (cases[ci].dw_clamp) {
                  acc = fmaxf(acc, 0.0f);
                  acc = fminf(acc, 6.0f);
               }
               dw_ref[((size_t)oy * dw_ow + ox) * N + c] = acc;
            }
         }
      }

      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));
      int32_t pad_b_buf[4] = {
         cases[ci].pad_b[0], cases[ci].pad_b[1],
         cases[ci].pad_b[2], cases[ci].pad_b[3],
      };
      struct hk_eval_ssbo sa[10];
      unsigned n_sa = 0;
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 0, a_buf, a_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 1, b_buf, b_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 2, g_bias, bias_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 3, c_alt, c_elems * 4, 0 };
      const uint64_t pad_dst_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 9, pad_alt, pad_elems * 4, 0 };
      const uint64_t pad_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 10, pad_b_buf, sizeof(pad_b_buf), 0 };
      const uint64_t dw_wt_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 11, dw_wt, wt_elems * 4, 0 };
      uint64_t dw_bias_va = 0;
      if (cases[ci].dw_bias) {
         dw_bias_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 12, dw_bias, (size_t)N * 4, 0 };
      }
      const uint64_t dw_d1_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 13, dw_d1, dw_elems * 4, 0 };
      uint64_t dw_d2_va = 0;
      if (cases[ci].dw_dst2) {
         dw_d2_va = hk_eval_make_vaddr(n_sa, 0);
         sa[n_sa++] = (struct hk_eval_ssbo){ 0, 14, dw_d2, dw_elems * 4, 0 };
      }

      const uint32_t dual_off = hk_dual_write_push_offset(m.push_bytes);
      struct hk_dual_gemm_dw_push dp = {
         .pad_dst_va_lo = (uint32_t)pad_dst_va, .pad_dst_va_hi = (uint32_t)(pad_dst_va >> 32),
         .pad_va_lo = (uint32_t)pad_va, .pad_va_hi = (uint32_t)(pad_va >> 32),
         .dw_wt_va_lo = (uint32_t)dw_wt_va, .dw_wt_va_hi = (uint32_t)(dw_wt_va >> 32),
         .dw_bias_va_lo = (uint32_t)dw_bias_va, .dw_bias_va_hi = (uint32_t)(dw_bias_va >> 32),
         .dw_dst1_va_lo = (uint32_t)dw_d1_va, .dw_dst1_va_hi = (uint32_t)(dw_d1_va >> 32),
         .dw_dst2_va_lo = (uint32_t)dw_d2_va, .dw_dst2_va_hi = (uint32_t)(dw_d2_va >> 32),
         .in_wh = in_w | (in_h << 16),
         .out_wh = out_w | (out_h << 16),
         .pad_idx = { 0, 1, 2, 3 },
         .flags = cases[ci].skip_flags |
                  (cases[ci].dw_bias ? HK_GEMM_DW_FLAG_HAS_BIAS : 0u) |
                  (cases[ci].dw_dst2 ? HK_GEMM_DW_FLAG_HAS_DW_DST2 : 0u),
      };
      hk_dual_gemm_dw_push_init_udiv(&dp, in_w >> 3);
      memcpy(push_buf + dual_off, &dp, sizeof(dp));

      const uint32_t ag[3] = { N / tile_n, M / 64u, 1u };
      hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, dual_off + sizeof(dp));

      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, 16u);
      params[HK_PACK_LOOP_PARAM_GEMM_K] = DIV_ROUND_UP(K, 4u);
      params[HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS] =
         DIV_ROUND_UP((uint64_t)pad_elems, (uint64_t)ag[0] * ag[1] * 128u);
      uint64_t bound = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb, params, &bound));
      HK_EVAL_CHECK(bound >= hk_eval_last_max_loop_backedges + 1);

      for (size_t i = 0; i < c_elems; ++i) {
         if (cases[ci].skip_flags & HK_GEMM_DW_FLAG_SKIP_C)
            HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(-88.5f));
         else
            HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));
      }
      for (size_t i = 0; i < pad_elems; ++i) {
         if (cases[ci].skip_flags & HK_GEMM_DW_FLAG_SKIP_PAD_DST)
            HK_EVAL_CHECK(hk_eval_f2u(pad_alt[i]) == hk_eval_f2u(-77.5f));
         else
            HK_EVAL_CHECK(hk_eval_f2u(pad_alt[i]) == hk_eval_f2u(pad_ref[i]));
      }
      for (size_t i = 0; i < dw_elems; ++i) {
         if (cases[ci].skip_flags & HK_GEMM_DW_FLAG_SKIP_DW_DST1)
            HK_EVAL_CHECK(hk_eval_f2u(dw_d1[i]) == hk_eval_f2u(-66.5f));
         else
            HK_EVAL_CHECK(hk_eval_f2u(dw_d1[i]) == hk_eval_f2u(dw_ref[i]));
         if (cases[ci].dw_dst2 && !(cases[ci].skip_flags & HK_GEMM_DW_FLAG_SKIP_DW_DST2))
            HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(dw_ref[i]));
         else
            HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(-55.5f));
      }

      free(a_buf); free(b_buf); free(g_bias); free(c_ref); free(c_alt);
      free(pad_ref); free(pad_alt); free(dw_wt); free(dw_bias);
      free(dw_ref); free(dw_d1); free(dw_d2);
      ralloc_free(alt); ralloc_free(pre); ralloc_free(ref_orig);
   }
}

/* A depthwise or GEMM shader that comes from the disk cache, a pipeline cache
 * or a shader binary fuses at submission from the NIR its producer carries:
 * hk_compute_pair_producer_serialize() writes it with nir_serialize() and
 * hk_compute_pair_producer_deserialize() reads it back with nir_deserialize().
 * That NIR must match as the compiled one did and build every fused variant
 * the compiled one builds, byte for byte and with the same loop bound, or a
 * warm cache fuses differently from a cold one. */
static nir_shader *
warm_cache_round_trip(const nir_shader *cold)
{
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, cold, false);
   HK_EVAL_CHECK(!blob.out_of_memory);
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   nir_shader *warm = nir_deserialize(NULL, &agx_nir_options, &reader);
   HK_EVAL_CHECK(warm && reader.current == reader.end && !reader.overrun);
   blob_finish(&blob);
   return warm;
}

static void
check_same_loop_bound(const struct hk_pack_loop_bound *a,
                      const struct hk_pack_loop_bound *b)
{
   HK_EVAL_CHECK(a->valid == b->valid && a->count == b->count);
   HK_EVAL_CHECK(a->work_valid == b->work_valid &&
                 a->work_count == b->work_count &&
                 a->static_nir_ops == b->static_nir_ops);
   for (unsigned i = 0; i < a->count; ++i)
      HK_EVAL_CHECK(a->ops[i].kind == b->ops[i].kind &&
                    a->ops[i].lhs == b->ops[i].lhs &&
                    a->ops[i].rhs == b->ops[i].rhs &&
                    a->ops[i].imm == b->ops[i].imm);
   for (unsigned i = 0; i < a->work_count; ++i)
      HK_EVAL_CHECK(a->work_ops[i].kind == b->work_ops[i].kind &&
                    a->work_ops[i].lhs == b->work_ops[i].lhs &&
                    a->work_ops[i].rhs == b->work_ops[i].rhs &&
                    a->work_ops[i].imm == b->work_ops[i].imm);
}

/* Takes both shaders. */
static void
check_same_fused(const char *what, nir_shader *cold, nir_shader *warm,
                 const struct hk_pack_loop_bound *lb_cold,
                 const struct hk_pack_loop_bound *lb_warm)
{
   if (!cold || !warm) {
      fprintf(stderr, "%s: cold %p warm %p\n", what, (void *)cold, (void *)warm);
      HK_EVAL_CHECK(cold && warm);
   }
   nir_validate_shader(warm, what);
   struct blob a, b;
   blob_init(&a);
   blob_init(&b);
   nir_serialize(&a, cold, false);
   nir_serialize(&b, warm, false);
   if (a.size != b.size || memcmp(a.data, b.data, a.size)) {
      fprintf(stderr, "%s: warm-cache fused NIR differs (%zu vs %zu bytes)\n",
              what, a.size, b.size);
      HK_EVAL_CHECK(false);
   }
   blob_finish(&a);
   blob_finish(&b);
   if (lb_cold)
      check_same_loop_bound(lb_cold, lb_warm);
   ralloc_free(cold);
   ralloc_free(warm);
}

static void
check_gemm_warm_fusion(const struct gemm_test_opts *opts)
{
   nir_shader *cold = build_tiled_gemm_nir_opts(opts);
   run_hk_preprocess_for_test(cold, false);
   struct hk_nir_gemm_output mc = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      cold, nir_address_format_64bit_global_32bit_offset, &mc));
   HK_EVAL_CHECK(mc.cmat_valid && mc.is_guarded == opts->guarded);

   nir_shader *warm = warm_cache_round_trip(cold);
   struct hk_nir_gemm_output mw = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      warm, nir_address_format_64bit_global_32bit_offset, &mw));
   HK_EVAL_CHECK(!memcmp(&mc, &mw, offsetof(struct hk_nir_gemm_output, cmat_valid) +
                                      sizeof(mc.cmat_valid)));

   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct hk_nir_fadd_epilogue fadd = {.prod_is_rhs = true};
   const struct hk_nir_dw_a_cfg dw_cfg = {
      .has_epilogue = true,
      .epilogue = relu6,
   };

   /* What hk_compute_pair_get_or_compile_fused() asks the builders for. */
   const struct {
      uint32_t tile_n;
      bool clamp;
      enum hk_compute_pair_dual_write dual;
      bool fadd;
      bool im2col_a;
      bool dw_a;
      bool gemm_dw;
   } guarded_cases[] = {
      {16, true, HK_COMPUTE_PAIR_DUAL_NONE},
      {32, true, HK_COMPUTE_PAIR_DUAL_COPY},
      {64, true, HK_COMPUTE_PAIR_DUAL_PAD4},
      {32, false, HK_COMPUTE_PAIR_DUAL_FADD_EP, true},
      {64, true, HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY, true},
      {32, true, HK_COMPUTE_PAIR_DUAL_COPY, false, true},
   }, unguarded_cases[] = {
      {16, false, HK_COMPUTE_PAIR_DUAL_NONE},
      {32, true, HK_COMPUTE_PAIR_DUAL_NONE, false, true},
      {16, false, HK_COMPUTE_PAIR_DUAL_COPY, false, false, true},
      {24, false, HK_COMPUTE_PAIR_DUAL_NONE, false, false, true},
      {24, true, HK_COMPUTE_PAIR_DUAL_GEMM_DW, false, false, false, true},
      {32, false, HK_COMPUTE_PAIR_DUAL_GEMM_DW, false, false, false, true},
   };
   const unsigned count = opts->guarded ? ARRAY_SIZE(guarded_cases)
                                        : ARRAY_SIZE(unguarded_cases);
   for (unsigned i = 0; i < count; ++i) {
      const __typeof__(guarded_cases[0]) *c =
         opts->guarded ? &guarded_cases[i] : &unguarded_cases[i];
      struct hk_pack_loop_bound lb_cold = {0}, lb_warm = {0};
      nir_shader *fc = hk_nir_build_gemm_cmat_with_bound(
         cold, &mc, c->tile_n, c->clamp ? &relu6 : NULL, c->dual,
         c->fadd ? &fadd : NULL, c->im2col_a, c->dw_a ? &dw_cfg : NULL,
         c->gemm_dw ? &dw_cfg : NULL, &lb_cold);
      nir_shader *fw = hk_nir_build_gemm_cmat_with_bound(
         warm, &mw, c->tile_n, c->clamp ? &relu6 : NULL, c->dual,
         c->fadd ? &fadd : NULL, c->im2col_a, c->dw_a ? &dw_cfg : NULL,
         c->gemm_dw ? &dw_cfg : NULL, &lb_warm);
      HK_EVAL_CHECK(lb_cold.valid);
      check_same_fused("warm_gemm_cmat", fc, fw, &lb_cold, &lb_warm);
   }

   /* A fused clamp without an alternate splices into the producer itself. */
   check_same_fused("warm_gemm_splice",
                    hk_nir_splice_epilogue(cold, mc.output, &relu6),
                    hk_nir_splice_epilogue(warm, mw.output, &relu6), NULL, NULL);

   ralloc_free(warm);
   ralloc_free(cold);
}

/* Whether `def` is computed from the descriptor of storage buffer
 * (set, binding). Walks ALU, intrinsic and deref sources, phis included. */
static bool
def_reads_ssbo_descriptor(nir_def *def, unsigned set, unsigned binding,
                          BITSET_WORD *seen)
{
   if (BITSET_TEST(seen, def->index))
      return false;
   BITSET_SET(seen, def->index);

   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
         if (def_reads_ssbo_descriptor(alu->src[i].src.ssa, set, binding, seen))
            return true;
      }
      return false;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
      if (in->intrinsic == nir_intrinsic_vulkan_resource_index &&
          nir_intrinsic_desc_set(in) == set && nir_intrinsic_binding(in) == binding)
         return true;
      for (unsigned i = 0; i < nir_intrinsic_infos[in->intrinsic].num_srcs; ++i) {
         if (def_reads_ssbo_descriptor(in->src[i].ssa, set, binding, seen))
            return true;
      }
      return false;
   }
   case nir_instr_type_deref: {
      nir_deref_instr *d = nir_instr_as_deref(instr);
      if (d->deref_type != nir_deref_type_var &&
          def_reads_ssbo_descriptor(d->parent.ssa, set, binding, seen))
         return true;
      return (d->deref_type == nir_deref_type_array ||
              d->deref_type == nir_deref_type_ptr_as_array) &&
             def_reads_ssbo_descriptor(d->arr.index.ssa, set, binding, seen);
   }
   case nir_instr_type_phi: {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_foreach_phi_src(src, phi) {
         if (def_reads_ssbo_descriptor(src->src.ssa, set, binding, seen))
            return true;
      }
      return false;
   }
   default:
      return false;
   }
}

/* The global accesses the residual-FADD post-pass adds: its residual loads
 * and destination stores, which, unlike the GEMM's own accesses, are not
 * deref_buffer_address of a descriptor. Each must take its address from push
 * constants, never from the output descriptor: an address formed as the
 * output's address plus a push-constant difference is only as right as the
 * output base the compiled shader forms, and on G16G, which has no soft
 * faults, a wrong one faulted Background Blur (write unmapped at wrapped
 * addresses 0x7fffffca3c0.., i.e. about -0x35c40). */
static unsigned
check_gemm_fadd_epilogue_addresses(nir_shader *nir,
                                   const struct hk_nir_gemm_output *m)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   unsigned epilogue = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         nir_def *addr;
         if (in->intrinsic == nir_intrinsic_load_global)
            addr = in->src[0].ssa;
         else if (in->intrinsic == nir_intrinsic_store_global)
            addr = in->src[1].ssa;
         else
            continue;
         nir_instr *ai = nir_def_instr(addr);
         if (ai->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(ai)->intrinsic ==
                nir_intrinsic_deref_buffer_address)
            continue;
         BITSET_WORD *seen = calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
         HK_EVAL_CHECK(seen);
         HK_EVAL_CHECK(!def_reads_ssbo_descriptor(addr, m->output.set,
                                                  m->output.binding, seen));
         free(seen);
         epilogue++;
      }
   }
   return epilogue;
}

static bool
compile_gemm_fadd_native(nir_shader *nir, enum agx_chip chip)
{
   uint32_t max_half = 24u;
   nir_shader_intrinsics_pass(nir, lower_hk_compute_intr_for_g17p,
                              nir_metadata_control_flow, &max_half);
   nir_opt_dce(nir);
   nir_lower_vars_to_explicit_types(nir, nir_var_mem_shared,
                                    test_g17p_shared_var_info);
   nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_offset);
   nir->info.io_lowered = true;
   agx_preprocess_nir_native(nir, chip);

   /* The compute constant-program contract native AGX3 launches use
    * (hk_compile_nir()), with each chip's soft-fault state: G16 has none. */
   struct agx_shader_key key = {
      .reserved_preamble = (max_half + 3u) & ~3u,
      .dev = {
         .chip = chip,
         .encoding = agx_chip_isa_encoding(chip),
         .soft_fault = chip != AGX_CHIP_G16G,
      },
      .has_scratch = true,
      .promote_constants = true,
      .promote_textures = true,
      .agx3_native_cprog = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(nir, &key, &part) &&
                   part.binary != NULL && part.info.binary_size > 0 &&
                   part.info.scratch_size == 0;
   free(part.binary);
   return ok;
}

/* GEMM + residual FADD post-pass (hk_nir_apply_gemm_fadd()) over the real
 * cooperative-matrix GEMM builder: every epilogue address comes from push
 * constants, the epilogue lands bit-exact when the residual and destinations
 * sit below the output (the J773g Background Blur layout) and when the output
 * buffer is at an address no push value was computed from, and each variant
 * compiles natively for G16G, G17P and G17S. */
static void
test_gemm_fadd_postpass_g16g_addresses(void)
{
   const struct {
      bool guarded;
      uint32_t M, N, K, tile_n, offset_B;
   } shapes[] = {
      { true, 35, 13, 19, 16, 0 },
      { true, 67, 29, 23, 32, 16 },
      { true, 37, 61, 35, 64, 0 },
      { false, 64, 32, 32, 16, 16 },
      { false, 64, 32, 32, 32, 0 },
      { false, 64, 64, 32, 64, 16 },
   };
   const enum hk_compute_pair_dual_write modes[] = {
      HK_COMPUTE_PAIR_DUAL_FADD_INPLACE, HK_COMPUTE_PAIR_DUAL_FADD_OUT,
      HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE, HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT,
   };
   const enum agx_chip chips[] = { AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17S };

   for (unsigned sh = 0; sh < ARRAY_SIZE(shapes); ++sh) {
      const uint32_t M = shapes[sh].M, N = shapes[sh].N, K = shapes[sh].K;
      const uint32_t tile_n = shapes[sh].tile_n;
      const struct gemm_test_opts opts = {
         .guarded = shapes[sh].guarded, .has_bias_buf = true,
         .has_bias_mode = true, .use_ffma = true, .offset_B = shapes[sh].offset_B,
      };
      nir_shader *orig = build_tiled_gemm_nir_opts(&opts);
      nir_shader *pre = nir_shader_clone(NULL, orig);
      run_hk_preprocess_for_test(pre, false);
      struct hk_nir_gemm_output m = {0};
      HK_EVAL_CHECK(hk_nir_match_gemm_output(
         pre, nir_address_format_64bit_global_32bit_offset, &m));
      HK_EVAL_CHECK(m.cmat_valid && m.output.offset_B == shapes[sh].offset_B);

      const uint32_t off_elems = m.output.offset_B / 4u;
      const size_t mn = (size_t)M * N;
      const size_t a_elems = (size_t)M * K + off_elems;
      const size_t b_elems = (size_t)K * N + off_elems;
      const size_t bias_elems = (size_t)MAX2(M, N) + off_elems + 1;
      const size_t c_elems = mn + off_elems;
      float *a = calloc(a_elems, 4), *bb = calloc(b_elems, 4);
      float *bias = calloc(bias_elems, 4), *c_ref = calloc(c_elems, 4);
      float *c = calloc(c_elems, 4), *res = calloc(mn, 4);
      float *d1 = calloc(mn, 4), *d2 = calloc(mn, 4);
      HK_EVAL_CHECK(a && bb && bias && c_ref && c && res && d1 && d2);
      uint32_t rng = 0x4c16u + sh;
      for (size_t i = off_elems; i < a_elems; ++i) a[i] = gemm_test_rand_f32(&rng);
      for (size_t i = off_elems; i < b_elems; ++i) bb[i] = gemm_test_rand_f32(&rng);
      for (size_t i = off_elems; i < bias_elems; ++i) bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < mn; ++i) res[i] = gemm_test_rand_f32(&rng);

      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
      };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      const struct hk_eval_ssbo so[4] = {
         { 0, 0, a, a_elems * 4, 0 }, { 0, 1, bb, b_elems * 4, 0 },
         { 0, 2, bias, bias_elems * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
      };
      hk_eval_dispatch_wg(orig, og, so, 4, &push, sizeof(push));

      for (unsigned mo = 0; mo < ARRAY_SIZE(modes); ++mo) {
         const enum hk_compute_pair_dual_write mode = modes[mo];
         const bool inplace = mode == HK_COMPUTE_PAIR_DUAL_FADD_INPLACE ||
                              mode == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE;
         const bool copy = mode == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE ||
                           mode == HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT;
         nir_shader *alt = hk_nir_build_gemm_cmat(pre, &m, tile_n, NULL,
                                                  HK_COMPUTE_PAIR_DUAL_NONE);
         HK_EVAL_CHECK(alt && hk_nir_apply_gemm_fadd(alt, &m, mode));
         nir_validate_shader(alt, "gemm_fadd_postpass");
         HK_EVAL_CHECK(check_gemm_fadd_epilogue_addresses(alt, &m) > 0);

         /* The residual and destinations take the lowest addresses, below
          * the output, as on J773g. The push words name them by address only,
          * so the output may then move to an address none of them was
          * computed from (slot 6 instead of 3) and the epilogue must still
          * land exactly in them. */
         const uint64_t res_va = hk_eval_make_vaddr(0, 0);
         const uint64_t d1_va = hk_eval_make_vaddr(1, 0);
         const uint64_t d2_va = hk_eval_make_vaddr(2, 0);
         uint8_t push_buf[256] = {0};
         memcpy(push_buf, &push, sizeof(push));
         const struct hk_dual_fadd_push dp = {
            .res_va_lo = (uint32_t)res_va, .res_va_hi = (uint32_t)(res_va >> 32),
            .dst1_va_lo = (uint32_t)d1_va, .dst1_va_hi = (uint32_t)(d1_va >> 32),
            .dst2_va_lo = (uint32_t)d2_va, .dst2_va_hi = (uint32_t)(d2_va >> 32),
         };
         memcpy(push_buf + hk_dual_write_push_offset(m.push_bytes), &dp, sizeof(dp));
         const uint32_t ag[3] = { DIV_ROUND_UP(N, tile_n),
                                  DIV_ROUND_UP(M, hk_gemm_cmat_tile_m(tile_n)), 1u };
         for (unsigned moved = 0; moved < 2; ++moved) {
            memset(c, 0, c_elems * 4);
            for (size_t i = 0; i < mn; ++i) { d1[i] = -41.0f; d2[i] = -82.0f; }
            const struct hk_eval_ssbo out = { 0, 3, c, c_elems * 4, 0 };
            const struct hk_eval_ssbo fixed[6] = {
               { 1, 0, res, mn * 4, 0 }, { 1, 1, d1, mn * 4, 0 },
               { 1, 2, d2, mn * 4, 0 }, { 0, 0, a, a_elems * 4, 0 },
               { 0, 1, bb, b_elems * 4, 0 }, { 0, 2, bias, bias_elems * 4, 0 },
            };
            struct hk_eval_ssbo sa[7];
            memcpy(sa, fixed, 3 * sizeof(sa[0]));
            if (moved) {
               memcpy(sa + 3, fixed + 3, 3 * sizeof(sa[0]));
               sa[6] = out;
            } else {
               sa[3] = out;
               memcpy(sa + 4, fixed + 3, 3 * sizeof(sa[0]));
            }
            hk_eval_dispatch_wg(alt, ag, sa, 7, push_buf, sizeof(push_buf));
            for (size_t i = 0; i < mn; ++i) {
               const float prod = c_ref[off_elems + i];
               const float sum = prod + res[i];
               HK_EVAL_CHECK(hk_eval_f2u(c[off_elems + i]) ==
                             hk_eval_f2u(inplace ? sum : prod));
               HK_EVAL_CHECK(hk_eval_f2u(d1[i]) ==
                             hk_eval_f2u(inplace ? -41.0f : sum));
               HK_EVAL_CHECK(hk_eval_f2u(d2[i]) ==
                             hk_eval_f2u(copy ? sum : -82.0f));
            }
         }

         for (unsigned ci = 0; ci < ARRAY_SIZE(chips); ++ci) {
            nir_shader *native = nir_shader_clone(NULL, alt);
            const bool ok = compile_gemm_fadd_native(native, chips[ci]);
            if (!ok) {
               fprintf(stderr, "gemm fadd post-pass: %s%u mode %u fails on chip %u\n",
                       shapes[sh].guarded ? "grd" : "ug", tile_n, mode, chips[ci]);
            }
            HK_EVAL_CHECK(ok);
            ralloc_free(native);
         }
         ralloc_free(alt);
      }
      free(a); free(bb); free(bias); free(c_ref); free(c);
      free(res); free(d1); free(d2);
      ralloc_free(pre);
      ralloc_free(orig);
   }
}

static void
test_warm_cache_producer_fusion(void)
{
   const struct gemm_test_opts guarded = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct gemm_test_opts unguarded = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   check_gemm_warm_fusion(&guarded);
   check_gemm_warm_fusion(&unguarded);

   nir_shader *cold = build_depthwise_nir(true, true, 0, 0, 1, 2, 3);
   run_hk_preprocess_for_test(cold, false);
   struct hk_nir_depthwise_output mc = {0};
   HK_EVAL_CHECK(hk_nir_match_depthwise_output(
      cold, nir_address_format_64bit_global_32bit_offset, &mc));
   HK_EVAL_CHECK(mc.vec4_valid);

   nir_shader *warm = warm_cache_round_trip(cold);
   struct hk_nir_depthwise_output mw = {0};
   HK_EVAL_CHECK(hk_nir_match_depthwise_output(
      warm, nir_address_format_64bit_global_32bit_offset, &mw));
   HK_EVAL_CHECK(!memcmp(&mc, &mw, offsetof(struct hk_nir_depthwise_output, vec4_valid) +
                                      sizeof(mc.vec4_valid)));

   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const enum hk_compute_pair_dual_write duals[] = {
      HK_COMPUTE_PAIR_DUAL_NONE, HK_COMPUTE_PAIR_DUAL_COPY,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(duals); ++i) {
      struct hk_pack_loop_bound lb_cold = {0}, lb_warm = {0};
      nir_shader *fc = hk_nir_build_depthwise_vec4(
         cold, &mc, &relu6, duals[i]);
      nir_shader *fw = hk_nir_build_depthwise_vec4(
         warm, &mw, &relu6, duals[i]);
      HK_EVAL_CHECK(fc && fw &&
                    hk_nir_bind_depthwise_loop_bounds(fc, &mc, &lb_cold) &&
                    hk_nir_bind_depthwise_loop_bounds(fw, &mw, &lb_warm));
      HK_EVAL_CHECK(lb_cold.valid);
      check_same_fused("warm_depthwise_vec4", fc, fw, &lb_cold, &lb_warm);
   }
   check_same_fused("warm_depthwise_splice",
                    hk_nir_splice_epilogue(cold, mc.output, &relu6),
                    hk_nir_splice_epilogue(warm, mw.output, &relu6), NULL, NULL);

   ralloc_free(warm);
   ralloc_free(cold);
}

static void
test_gemm_cmat_k32_small_tiles(void)
{
   /* Bit-exact verification of K_step=32 unguarded tiles (32x32, 32x16, 16x32,
    * 16x16) across all K%32 tail remainders (8, 16, 24) and exact multiples,
    * plus guarded and gemm_dw refusal checks and G16G/G17P/G17S compile. */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct gemm_test_opts ug_opts = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct gemm_test_opts grd_opts = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   nir_shader *ref_orig = build_tiled_gemm_nir_opts(&grd_opts);
   nir_shader *ug_pre = build_tiled_gemm_nir_opts(&ug_opts);
   nir_shader *grd_pre = build_tiled_gemm_nir_opts(&grd_opts);
   run_hk_preprocess_for_test(ug_pre, false);
   run_hk_preprocess_for_test(grd_pre, false);
   struct hk_nir_gemm_output ug_m = {0}, grd_m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      ug_pre, nir_address_format_64bit_global_32bit_offset, &ug_m));
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      grd_pre, nir_address_format_64bit_global_32bit_offset, &grd_m));

   const struct { uint32_t tm, tn; } new_tiles[] = {
      { 32, 32 }, { 32, 16 }, { 16, 32 }, { 16, 16 },
   };
   const struct hk_nir_dw_a_cfg dummy_gdw = {0};
   const enum agx_chip chips[] = { AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17S };

   for (unsigned ti = 0; ti < ARRAY_SIZE(new_tiles); ++ti) {
      const uint32_t tm = new_tiles[ti].tm;
      const uint32_t tn = new_tiles[ti].tn;

      /* Guarded and gemm_dw must refuse these K_step=32 small tiles. */
      HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
         grd_pre, &grd_m, tm, tn, NULL, HK_COMPUTE_PAIR_DUAL_NONE,
         NULL, false, NULL, NULL, NULL) == NULL);
      HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
         ug_pre, &ug_m, tm, tn, NULL, HK_COMPUTE_PAIR_DUAL_GEMM_DW,
         NULL, false, NULL, &dummy_gdw, NULL) == NULL);

      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         ug_pre, &ug_m, tm, tn, &relu6, HK_COMPUTE_PAIR_DUAL_NONE,
         NULL, false, NULL, NULL, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_k32_small");

      const uint32_t k_cases[] = { 8, 16, 24, 32, 40, 48, 56, 64 };
      for (unsigned ki = 0; ki < ARRAY_SIZE(k_cases); ++ki) {
         const uint32_t M = tm * 2u;
         const uint32_t N = tn * 2u;
         const uint32_t K = k_cases[ki];
         const size_t a_elems = (size_t)M * K;
         const size_t b_elems = (size_t)K * N;
         const size_t c_elems = (size_t)M * N;
         float *a = calloc(a_elems, 4), *bb = calloc(b_elems, 4);
         float *bias = calloc(N, 4), *c_ref = calloc(c_elems, 4), *c_alt = calloc(c_elems, 4);
         uint32_t rng = 0x5100u + ti * 97u + ki * 13u;
         for (size_t i = 0; i < a_elems; ++i) a[i] = gemm_test_rand_f32(&rng);
         for (size_t i = 0; i < b_elems; ++i) bb[i] = gemm_test_rand_f32(&rng);
         for (size_t i = 0; i < N; ++i) bias[i] = gemm_test_rand_f32(&rng);

         struct gemm_push_layout push = {
            .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
         };
         const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
         struct hk_eval_ssbo so[4] = {
            { 0, 0, a, a_elems * 4, 0 }, { 0, 1, bb, b_elems * 4, 0 },
            { 0, 2, bias, (size_t)N * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
         };
         hk_eval_dispatch_wg(ref_orig, og, so, 4, &push, sizeof(push));
         for (size_t i = 0; i < c_elems; ++i)
            c_ref[i] = fminf(fmaxf(c_ref[i], 0.0f), 6.0f);

         const uint32_t ag[3] = { N / tn, M / tm, 1u };
         struct hk_eval_ssbo sa[4] = {
            { 0, 0, a, a_elems * 4, 0 }, { 0, 1, bb, b_elems * 4, 0 },
            { 0, 2, bias, (size_t)N * 4, 0 }, { 0, 3, c_alt, c_elems * 4, 0 },
         };
         hk_eval_dispatch_wg(alt, ag, sa, 4, &push, sizeof(push));
         for (size_t i = 0; i < c_elems; ++i)
            HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));

         free(a); free(bb); free(bias); free(c_ref); free(c_alt);
      }

      for (unsigned ci = 0; ci < ARRAY_SIZE(chips); ++ci) {
         nir_shader *native = nir_shader_clone(NULL, alt);
         HK_EVAL_CHECK(compile_gemm_fadd_native(native, chips[ci]));
         ralloc_free(native);
      }
      ralloc_free(alt);
   }

   ralloc_free(grd_pre);
   ralloc_free(ug_pre);
   ralloc_free(ref_orig);
}

static void
test_gemm_exp_dw_a_fusion(void)
{
   /* Bit-exact NIR-eval verification of fused im2col + Expand -> 3x3 s=1 DW ->
    * 1x1 Project (im2_exp_dw_a), plus refusal checks for non-im2col expand,
    * guarded, tile_m != 64, and DUAL_PAD4. */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct gemm_test_opts ref_opts = {
      .guarded = true, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   const struct gemm_test_opts ug_opts = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   nir_shader *ref_orig = build_tiled_gemm_nir_opts(&ref_opts);
   nir_shader *grd_pre = build_tiled_gemm_nir_opts(&ref_opts);
   nir_shader *ug_pre = build_tiled_gemm_nir_opts(&ug_opts);
   run_hk_preprocess_for_test(grd_pre, false);
   run_hk_preprocess_for_test(ug_pre, false);
   struct hk_nir_gemm_output grd_m = {0}, ug_m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      grd_pre, nir_address_format_64bit_global_32bit_offset, &grd_m));
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      ug_pre, nir_address_format_64bit_global_32bit_offset, &ug_m));

   /* Refusal checks: guarded, tile_m != 64, DUAL_PAD4 (both im2col and 1x1 expand). */
   const struct hk_nir_dw_a_cfg no_im2_cfg = { .has_expand = true };
   const struct hk_nir_dw_a_cfg bad_cfg = { .has_expand = true, .has_exp_im2col = true };
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
      grd_pre, &grd_m, 64, 24, NULL, HK_COMPUTE_PAIR_DUAL_NONE,
      NULL, false, &no_im2_cfg, NULL, NULL) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
      ug_pre, &ug_m, 32, 24, NULL, HK_COMPUTE_PAIR_DUAL_NONE,
      NULL, false, &no_im2_cfg, NULL, NULL) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
      ug_pre, &ug_m, 64, 24, NULL, HK_COMPUTE_PAIR_DUAL_PAD4,
      NULL, false, &no_im2_cfg, NULL, NULL) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
      grd_pre, &grd_m, 64, 32, NULL, HK_COMPUTE_PAIR_DUAL_NONE,
      NULL, false, &bad_cfg, NULL, NULL) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
      ug_pre, &ug_m, 32, 32, NULL, HK_COMPUTE_PAIR_DUAL_NONE,
      NULL, false, &bad_cfg, NULL, NULL) == NULL);
   HK_EVAL_CHECK(hk_nir_build_gemm_cmat_tiled_with_bound(
      ug_pre, &ug_m, 64, 32, NULL, HK_COMPUTE_PAIR_DUAL_PAD4,
      NULL, false, &bad_cfg, NULL, NULL) == NULL);

   /* Also verify im2_exp_dw_a (3x3 s=2 RGB im2col + 27->32 GEMM + clamp -> 3x3 s=1 DW
    * + clamp + copy4 -> 32->16 Project + copy4) on a 16x16 (2x2 spatial tiles, 32x32x3
    * input) problem, covering interior + right/bottom im2col border + skip_im2_dst=0/1. */
   for (unsigned skip_im2 = 0; skip_im2 < 2; ++skip_im2) {
      const uint32_t in_h = 16, in_w = 16, M = in_h * in_w;
      const uint32_t src_h = 32, src_w_px = 32, ch = 3, iw = src_w_px * ch;
      const uint32_t kw = 3 * ch, kh = 3, EK = kw * kh; /* 27 */
      const uint32_t K = 32, N = 16, tile_n = 16;

      struct hk_nir_dw_a_cfg cfg = {
         .has_epilogue = true,
         .epilogue = relu6,
         .has_expand = true,
         .has_exp_im2col = true,
         .has_exp_epilogue = true,
         .exp_epilogue = relu6,
      };
      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         ug_pre, &ug_m, 64, tile_n, NULL, HK_COMPUTE_PAIR_DUAL_COPY,
         NULL, false, &cfg, NULL, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_im2expdwa");

      const size_t src_elems = (size_t)src_h * iw;
      const size_t im2_elems = (size_t)M * EK;
      const size_t eb_elems = (size_t)EK * K;
      const size_t mid_elems = (size_t)M * K;
      const size_t wt_elems = 9u * K;
      const size_t pb_elems = (size_t)K * N;
      const size_t c_elems = (size_t)M * N;
      float *src_buf = calloc(src_elems, 4);
      float *im2_ref = calloc(im2_elems, 4);
      float *im2_alt = calloc(im2_elems, 4);
      float *exp_b = calloc(eb_elems, 4);
      float *exp_bias = calloc(K, 4);
      float *exp_c_ref = calloc(mid_elems, 4);
      float *exp_c_alt = calloc(mid_elems, 4);
      float *wt_buf = calloc(wt_elems, 4);
      float *dw_bias = calloc(K, 4);
      float *dw_ref = calloc(mid_elems, 4);
      float *dw_d1 = calloc(mid_elems, 4);
      float *dw_d2 = calloc(mid_elems, 4);
      float *a_dummy = calloc(mid_elems, 4);
      float *proj_b = calloc(pb_elems, 4);
      float *proj_bias = calloc(N, 4);
      float *c_ref = calloc(c_elems, 4);
      float *c_alt = calloc(c_elems, 4);
      float *g_dst2 = calloc(c_elems, 4);

      uint32_t rng = 0xa700u + skip_im2 * 53u;
      for (size_t i = 0; i < src_elems; ++i) src_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < eb_elems; ++i) exp_b[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < K; ++i) exp_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < wt_elems; ++i) wt_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < K; ++i) dw_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < pb_elems; ++i) proj_b[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) proj_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < im2_elems; ++i) im2_alt[i] = -44.5f;
      for (size_t i = 0; i < mid_elems; ++i) {
         exp_c_alt[i] = -66.5f; dw_d1[i] = -99.5f; dw_d2[i] = -77.5f; a_dummy[i] = -55.25f;
      }

      /* CPU reference im2col (sh=2, step_x=6, pad_x=0, pad_y=0, kw=9, kh=3) + Expand + DW. */
      for (uint32_t oy = 0; oy < in_h; ++oy) {
         for (uint32_t ox = 0; ox < in_w; ++ox) {
            const uint32_t p = oy * in_w + ox;
            for (uint32_t ky = 0; ky < kh; ++ky) {
               const uint32_t sy = oy * 2u + ky;
               for (uint32_t kx = 0; kx < kw; ++kx) {
                  const uint32_t sx = ox * 6u + kx;
                  im2_ref[(size_t)p * EK + ky * kw + kx] =
                     (sy < src_h && sx < iw) ? src_buf[(size_t)sy * iw + sx] : 0.0f;
               }
            }
            for (uint32_t c = 0; c < K; ++c) {
               float acc = 0.0f;
               for (uint32_t kk = 0; kk < EK; ++kk)
                  acc = fmaf(im2_ref[(size_t)p * EK + kk], exp_b[(size_t)kk * K + c], acc);
               acc += exp_bias[c];
               acc = fminf(fmaxf(acc, 0.0f), 6.0f);
               exp_c_ref[(size_t)p * K + c] = acc;
            }
         }
      }
      for (uint32_t oy = 0; oy < in_h; ++oy) {
         for (uint32_t ox = 0; ox < in_w; ++ox) {
            for (uint32_t c = 0; c < K; ++c) {
               float acc = 0.0f;
               for (int32_t ky = 0; ky < 3; ++ky) {
                  const int32_t iy = (int32_t)oy - 1 + ky;
                  for (int32_t kx = 0; kx < 3; ++kx) {
                     const int32_t ix = (int32_t)ox - 1 + kx;
                     const float iv = (iy >= 0 && iy < (int32_t)in_h &&
                                       ix >= 0 && ix < (int32_t)in_w)
                                         ? exp_c_ref[((size_t)iy * in_w + ix) * K + c]
                                         : 0.0f;
                     acc = ref_mul_add_unfused(iv, wt_buf[(size_t)(ky * 3 + kx) * K + c], acc);
                  }
               }
               acc += dw_bias[c];
               dw_ref[(size_t)(oy * in_w + ox) * K + c] = fminf(fmaxf(acc, 0.0f), 6.0f);
            }
         }
      }

      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
      };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      struct hk_eval_ssbo so[4] = {
         { 0, 0, dw_ref, mid_elems * 4, 0 }, { 0, 1, proj_b, pb_elems * 4, 0 },
         { 0, 2, proj_bias, (size_t)N * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
      };
      hk_eval_dispatch_wg(ref_orig, og, so, 4, &push, sizeof(push));

      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));
      struct hk_eval_ssbo sa[15];
      unsigned n_sa = 0;
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 0, a_dummy, mid_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 1, proj_b, pb_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 2, proj_bias, (size_t)N * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 3, c_alt, c_elems * 4, 0 };
      const uint32_t dual_off = hk_dual_write_push_offset(ug_m.push_bytes);
      const uint64_t gd2_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 9, g_dst2, c_elems * 4, 0 };
      struct hk_dual_copy_push dcp = {
         .dst2_va_lo = (uint32_t)gd2_va, .dst2_va_hi = (uint32_t)(gd2_va >> 32),
      };
      memcpy(push_buf + dual_off, &dcp, sizeof(dcp));

      const uint64_t exp_c_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 10, exp_c_alt, mid_elems * 4, 0 };
      const uint64_t wt_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 11, wt_buf, wt_elems * 4, 0 };
      const uint64_t dw_bias_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 12, dw_bias, (size_t)K * 4, 0 };
      const uint64_t dst1_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 13, dw_d1, mid_elems * 4, 0 };
      const uint64_t dst2_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 14, dw_d2, mid_elems * 4, 0 };
      const uint64_t src_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 15, src_buf, src_elems * 4, 0 };
      const uint64_t exp_b_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 16, exp_b, eb_elems * 4, 0 };
      const uint64_t exp_bias_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 17, exp_bias, (size_t)K * 4, 0 };
      const uint64_t im2_dst_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 18, im2_alt, im2_elems * 4, 0 };

      const uint32_t dw_off = hk_dw_a_push_offset(ug_m.push_bytes, HK_COMPUTE_PAIR_DUAL_COPY);
      const uint32_t exp_off = hk_exp_dw_a_push_offset(ug_m.push_bytes, HK_COMPUTE_PAIR_DUAL_COPY);
      const uint32_t dw_flags =
         HK_DW_A_FLAG_HAS_BIAS | HK_DW_A_FLAG_HAS_DST1 | HK_DW_A_FLAG_HAS_DST2;
      struct hk_dw_a_push dp = {
         .in_va_lo = (uint32_t)exp_c_va, .in_va_hi = (uint32_t)(exp_c_va >> 32),
         .wt_va_lo = (uint32_t)wt_va, .wt_va_hi = (uint32_t)(wt_va >> 32),
         .bias_va_lo = (uint32_t)dw_bias_va, .bias_va_hi = (uint32_t)(dw_bias_va >> 32),
         .dst1_va_lo = (uint32_t)dst1_va, .dst1_va_hi = (uint32_t)(dst1_va >> 32),
         .dst2_va_lo = (uint32_t)dst2_va, .dst2_va_hi = (uint32_t)(dst2_va >> 32),
         .in_wh = in_w | (in_h << 16),
         .out_wh = in_w | (in_h << 16),
         .pad_flags = ((uint32_t)(uint8_t)(int8_t)-1) |
                      (((uint32_t)(uint8_t)(int8_t)-1) << 8) |
                      (dw_flags << 16),
      };
      hk_dw_a_push_init_udiv(&dp, in_w);
      memcpy(push_buf + dw_off, &dp, sizeof(dp));

      struct hk_exp_dw_a_push ep = {
         .exp_a_va_lo = (uint32_t)src_va, .exp_a_va_hi = (uint32_t)(src_va >> 32),
         .exp_b_va_lo = (uint32_t)exp_b_va, .exp_b_va_hi = (uint32_t)(exp_b_va >> 32),
         .exp_bias_va_lo = (uint32_t)exp_bias_va, .exp_bias_va_hi = (uint32_t)(exp_bias_va >> 32),
         .im2_dst_va_lo = (uint32_t)im2_dst_va, .im2_dst_va_hi = (uint32_t)(im2_dst_va >> 32),
         .exp_k = EK,
         .flags = HK_EXP_DW_A_FLAG_HAS_BIAS |
                  (skip_im2 ? HK_EXP_DW_A_FLAG_SKIP_IM2_DST : 0u),
         .im2_iw_ih = iw | (src_h << 16),
         .im2_kw_kh = kw | (kh << 16),
         .im2_step_sh = 6u | (2u << 16),
         .im2_pad_xy = 0u,
      };
      hk_exp_dw_a_push_init_udiv(&ep, in_w >> 3);
      memcpy(push_buf + exp_off, &ep, sizeof(ep));

      const uint32_t ag[3] = { N / tile_n, (in_h >> 3) * (in_w >> 3), 1u };
      hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, exp_off + sizeof(ep));

      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, 16u);
      params[HK_PACK_LOOP_PARAM_GEMM_K] = DIV_ROUND_UP(EK, 8u);
      uint64_t bound = 0, eff = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb, params, &bound));
      HK_EVAL_CHECK(bound >= hk_eval_last_max_loop_backedges + 1);
      HK_EVAL_CHECK(hk_pack_eval_effective_instrs(&lb, 300u, params, &eff));
      HK_EVAL_CHECK(eff > 0 && eff < 500000u);

      for (size_t i = 0; i < im2_elems; ++i) {
         if (skip_im2)
            HK_EVAL_CHECK(hk_eval_f2u(im2_alt[i]) == hk_eval_f2u(-44.5f));
         else
            HK_EVAL_CHECK(hk_eval_f2u(im2_alt[i]) == hk_eval_f2u(im2_ref[i]));
      }
      for (size_t i = 0; i < mid_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(exp_c_alt[i]) == hk_eval_f2u(exp_c_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dw_d1[i]) == hk_eval_f2u(dw_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(dw_ref[i]));
      }
      for (size_t i = 0; i < c_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(g_dst2[i]) == hk_eval_f2u(c_ref[i]));
      }

      free(src_buf); free(im2_ref); free(im2_alt); free(exp_b); free(exp_bias);
      free(exp_c_ref); free(exp_c_alt); free(wt_buf); free(dw_bias); free(dw_ref);
      free(dw_d1); free(dw_d2); free(a_dummy); free(proj_b); free(proj_bias);
      free(c_ref); free(c_alt); free(g_dst2);
      ralloc_free(alt);
   }

   /* Bit-exact verification of 1x1 exp_dw_a (1x1 Expand + clamp -> 3x3 s=1 DW +
    * clamp + copy4 -> 1x1 Project + copy4) on a 16x16 (2x2 spatial tiles, EK=24,
    * K=32, N=24, tile_n=24) problem, covering border halo zeroing and
    * skip_exp_c=0/1. */
   for (unsigned skip_exp_c = 0; skip_exp_c < 2; ++skip_exp_c) {
      const uint32_t in_h = 16, in_w = 16, M = in_h * in_w;
      const uint32_t EK = 24, K = 32, N = 24, tile_n = 24;

      struct hk_nir_dw_a_cfg cfg = {
         .has_epilogue = true,
         .epilogue = relu6,
         .has_expand = true,
         .has_exp_im2col = false,
         .has_exp_epilogue = true,
         .exp_epilogue = relu6,
      };
      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         ug_pre, &ug_m, 64, tile_n, NULL, HK_COMPUTE_PAIR_DUAL_COPY,
         NULL, false, &cfg, NULL, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_exp1x1dwa");

      const size_t exp_a_elems = (size_t)M * EK;
      const size_t eb_elems = (size_t)EK * K;
      const size_t mid_elems = (size_t)M * K;
      const size_t wt_elems = 9u * K;
      const size_t pb_elems = (size_t)K * N;
      const size_t c_elems = (size_t)M * N;
      float *exp_a = calloc(exp_a_elems, 4);
      float *exp_b = calloc(eb_elems, 4);
      float *exp_bias = calloc(K, 4);
      float *exp_c_ref = calloc(mid_elems, 4);
      float *exp_c_alt = calloc(mid_elems, 4);
      float *wt_buf = calloc(wt_elems, 4);
      float *dw_bias = calloc(K, 4);
      float *dw_ref = calloc(mid_elems, 4);
      float *dw_d1 = calloc(mid_elems, 4);
      float *dw_d2 = calloc(mid_elems, 4);
      float *a_dummy = calloc(mid_elems, 4);
      float *proj_b = calloc(pb_elems, 4);
      float *proj_bias = calloc(N, 4);
      float *c_ref = calloc(c_elems, 4);
      float *c_alt = calloc(c_elems, 4);
      float *g_dst2 = calloc(c_elems, 4);

      uint32_t rng = 0xb900u + skip_exp_c * 61u;
      for (size_t i = 0; i < exp_a_elems; ++i) exp_a[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < eb_elems; ++i) exp_b[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < K; ++i) exp_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < wt_elems; ++i) wt_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < K; ++i) dw_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < pb_elems; ++i) proj_b[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) proj_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < mid_elems; ++i) {
         exp_c_alt[i] = -66.5f; dw_d1[i] = -99.5f; dw_d2[i] = -77.5f; a_dummy[i] = -55.25f;
      }

      for (uint32_t p = 0; p < M; ++p) {
         for (uint32_t c = 0; c < K; ++c) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < EK; ++kk)
               acc = fmaf(exp_a[(size_t)p * EK + kk], exp_b[(size_t)kk * K + c], acc);
            acc += exp_bias[c];
            exp_c_ref[(size_t)p * K + c] = fminf(fmaxf(acc, 0.0f), 6.0f);
         }
      }
      for (uint32_t oy = 0; oy < in_h; ++oy) {
         for (uint32_t ox = 0; ox < in_w; ++ox) {
            for (uint32_t c = 0; c < K; ++c) {
               float acc = 0.0f;
               for (int32_t ky = 0; ky < 3; ++ky) {
                  const int32_t iy = (int32_t)oy - 1 + ky;
                  for (int32_t kx = 0; kx < 3; ++kx) {
                     const int32_t ix = (int32_t)ox - 1 + kx;
                     const float iv = (iy >= 0 && iy < (int32_t)in_h &&
                                       ix >= 0 && ix < (int32_t)in_w)
                                         ? exp_c_ref[((size_t)iy * in_w + ix) * K + c]
                                         : 0.0f;
                     acc = ref_mul_add_unfused(iv, wt_buf[(size_t)(ky * 3 + kx) * K + c], acc);
                  }
               }
               acc += dw_bias[c];
               dw_ref[(size_t)(oy * in_w + ox) * K + c] = fminf(fmaxf(acc, 0.0f), 6.0f);
            }
         }
      }

      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
      };
      const uint32_t og[3] = { (N + 31u) / 32u, (M + 31u) / 32u, 1u };
      struct hk_eval_ssbo so[4] = {
         { 0, 0, dw_ref, mid_elems * 4, 0 }, { 0, 1, proj_b, pb_elems * 4, 0 },
         { 0, 2, proj_bias, (size_t)N * 4, 0 }, { 0, 3, c_ref, c_elems * 4, 0 },
      };
      hk_eval_dispatch_wg(ref_orig, og, so, 4, &push, sizeof(push));

      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));
      struct hk_eval_ssbo sa[14];
      unsigned n_sa = 0;
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 0, a_dummy, mid_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 1, proj_b, pb_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 2, proj_bias, (size_t)N * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 3, c_alt, c_elems * 4, 0 };
      const uint32_t dual_off = hk_dual_write_push_offset(ug_m.push_bytes);
      const uint64_t gd2_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 9, g_dst2, c_elems * 4, 0 };
      struct hk_dual_copy_push dcp = {
         .dst2_va_lo = (uint32_t)gd2_va, .dst2_va_hi = (uint32_t)(gd2_va >> 32),
      };
      memcpy(push_buf + dual_off, &dcp, sizeof(dcp));

      const uint64_t exp_c_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 10, exp_c_alt, mid_elems * 4, 0 };
      const uint64_t wt_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 11, wt_buf, wt_elems * 4, 0 };
      const uint64_t dw_bias_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 12, dw_bias, (size_t)K * 4, 0 };
      const uint64_t dst1_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 13, dw_d1, mid_elems * 4, 0 };
      const uint64_t dst2_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 14, dw_d2, mid_elems * 4, 0 };
      const uint64_t exp_a_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 15, exp_a, exp_a_elems * 4, 0 };
      const uint64_t exp_b_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 16, exp_b, eb_elems * 4, 0 };
      const uint64_t exp_bias_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 17, exp_bias, (size_t)K * 4, 0 };

      const uint32_t dw_off = hk_dw_a_push_offset(ug_m.push_bytes, HK_COMPUTE_PAIR_DUAL_COPY);
      const uint32_t exp_off = hk_exp_dw_a_push_offset(ug_m.push_bytes, HK_COMPUTE_PAIR_DUAL_COPY);
      const uint32_t dw_flags =
         HK_DW_A_FLAG_HAS_BIAS | HK_DW_A_FLAG_HAS_DST1 | HK_DW_A_FLAG_HAS_DST2;
      struct hk_dw_a_push dp = {
         .in_va_lo = (uint32_t)exp_c_va, .in_va_hi = (uint32_t)(exp_c_va >> 32),
         .wt_va_lo = (uint32_t)wt_va, .wt_va_hi = (uint32_t)(wt_va >> 32),
         .bias_va_lo = (uint32_t)dw_bias_va, .bias_va_hi = (uint32_t)(dw_bias_va >> 32),
         .dst1_va_lo = (uint32_t)dst1_va, .dst1_va_hi = (uint32_t)(dst1_va >> 32),
         .dst2_va_lo = (uint32_t)dst2_va, .dst2_va_hi = (uint32_t)(dst2_va >> 32),
         .in_wh = in_w | (in_h << 16),
         .out_wh = in_w | (in_h << 16),
         .pad_flags = ((uint32_t)(uint8_t)(int8_t)-1) |
                      (((uint32_t)(uint8_t)(int8_t)-1) << 8) |
                      (dw_flags << 16),
      };
      hk_dw_a_push_init_udiv(&dp, in_w);
      memcpy(push_buf + dw_off, &dp, sizeof(dp));

      struct hk_exp_dw_a_push ep = {
         .exp_a_va_lo = (uint32_t)exp_a_va, .exp_a_va_hi = (uint32_t)(exp_a_va >> 32),
         .exp_b_va_lo = (uint32_t)exp_b_va, .exp_b_va_hi = (uint32_t)(exp_b_va >> 32),
         .exp_bias_va_lo = (uint32_t)exp_bias_va, .exp_bias_va_hi = (uint32_t)(exp_bias_va >> 32),
         .exp_k = EK,
         .flags = HK_EXP_DW_A_FLAG_HAS_BIAS |
                  (skip_exp_c ? HK_EXP_DW_A_FLAG_SKIP_EXP_C : 0u),
      };
      hk_exp_dw_a_push_init_udiv(&ep, in_w >> 3);
      memcpy(push_buf + exp_off, &ep, sizeof(ep));

      const uint32_t ag[3] = { N / tile_n, (in_h >> 3) * (in_w >> 3), 1u };
      hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, exp_off + sizeof(ep));

      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, 16u);
      params[HK_PACK_LOOP_PARAM_GEMM_K] = DIV_ROUND_UP(EK, 8u);
      uint64_t bound = 0, eff = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb, params, &bound));
      HK_EVAL_CHECK(bound >= hk_eval_last_max_loop_backedges + 1);
      HK_EVAL_CHECK(hk_pack_eval_effective_instrs(&lb, 300u, params, &eff));
      HK_EVAL_CHECK(eff > 0 && eff < 500000u);

      for (size_t i = 0; i < mid_elems; ++i) {
         if (skip_exp_c)
            HK_EVAL_CHECK(hk_eval_f2u(exp_c_alt[i]) == hk_eval_f2u(-66.5f));
         else
            HK_EVAL_CHECK(hk_eval_f2u(exp_c_alt[i]) == hk_eval_f2u(exp_c_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dw_d1[i]) == hk_eval_f2u(dw_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(dw_ref[i]));
      }
      for (size_t i = 0; i < c_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(g_dst2[i]) == hk_eval_f2u(c_ref[i]));
      }

      free(exp_a); free(exp_b); free(exp_bias);
      free(exp_c_ref); free(exp_c_alt); free(wt_buf); free(dw_bias); free(dw_ref);
      free(dw_d1); free(dw_d2); free(a_dummy); free(proj_b); free(proj_bias);
      free(c_ref); free(c_alt); free(g_dst2);
      ralloc_free(alt);
   }

   ralloc_free(ug_pre);
   ralloc_free(grd_pre);
   ralloc_free(ref_orig);
}

static void
test_gemm_dw_proj_fusion(void)
{
   /* Bit-exact NIR-eval verification of DUAL_GEMM_DW with fused 1x1 Project
    * (Expand GEMM + clamp + PAD4 + 3x3 s=2 DW + clamp + copy4 + 1x1 Project +
    * copy4), covering:
    *   - tile_n=48 -> proj_n=24 (ord 2->3 shape, pad_b == 0)
    *   - tile_n=24 -> proj_n=32 (ord 6->7 shape, pad_b == 0, K=24 tail)
    *   - tile_n=32 -> proj_n=64 (ord 13->14 shape, pad_b == {0,1,0,0} fallback) */
   const struct hk_nir_clamp_epilogue relu6 = {
      .has_lower = true, .has_upper = true, .upper_first = false,
      .lower_bits = hk_eval_f2u(0.0f), .upper_bits = hk_eval_f2u(6.0f),
   };
   const struct gemm_test_opts opts_ug = {
      .guarded = false, .has_bias_buf = true, .has_bias_mode = true, .use_ffma = true,
   };
   nir_shader *ug_pre = build_tiled_gemm_nir_opts(&opts_ug);
   run_hk_preprocess_for_test(ug_pre, false);
   struct hk_nir_gemm_output ug_m = {0};
   HK_EVAL_CHECK(hk_nir_match_gemm_output(
      ug_pre, nir_address_format_64bit_global_32bit_offset, &ug_m));

   const struct {
      uint32_t tile_n, in_h, in_w, K, N, proj_n;
      bool pad_nonzero;
      bool skip_proj_c;
   } cases[] = {
      { 48, 16, 16, 16,  96, 24, false, false },
      { 24,  8, 16, 24,  72, 32, false, false },
      { 32,  8,  8, 32,  64, 64, true,  true  },
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const uint32_t tile_n = cases[ci].tile_n;
      const uint32_t in_h = cases[ci].in_h, in_w = cases[ci].in_w;
      const uint32_t M = in_h * in_w, K = cases[ci].K, N = cases[ci].N;
      const uint32_t out_h = in_h + 1, out_w = in_w + 1;
      const uint32_t dw_oh = in_h >> 1, dw_ow = in_w >> 1;
      const uint32_t M_pj = dw_oh * dw_ow, proj_n = cases[ci].proj_n;

      struct hk_nir_dw_a_cfg cfg = {
         .has_epilogue = true,
         .epilogue = relu6,
         .has_proj = true,
         .proj_tile_n = (uint8_t)proj_n,
      };
      struct hk_pack_loop_bound lb = {0};
      nir_shader *alt = hk_nir_build_gemm_cmat_tiled_with_bound(
         ug_pre, &ug_m, 64, tile_n, &relu6, HK_COMPUTE_PAIR_DUAL_GEMM_DW,
         NULL, false, NULL, &cfg, &lb);
      HK_EVAL_CHECK(alt != NULL && lb.valid && lb.work_valid);
      nir_validate_shader(alt, "gemm_cmat_gemmdwproj");

      const size_t a_elems = (size_t)M * K;
      const size_t b_elems = (size_t)K * N;
      const size_t c_elems = (size_t)M * N;
      const size_t pad_elems = (size_t)out_h * out_w * N;
      const size_t wt_elems = 9u * N;
      const size_t dw_elems = (size_t)M_pj * N;
      const size_t pb_elems = (size_t)N * proj_n;
      const size_t pc_elems = (size_t)M_pj * proj_n;

      float *a_buf = calloc(a_elems, 4);
      float *b_buf = calloc(b_elems, 4);
      float *bias_buf = calloc(N, 4);
      float *c_ref = calloc(c_elems, 4);
      float *c_alt = calloc(c_elems, 4);
      int32_t pad_b[4] = { 0, cases[ci].pad_nonzero ? 1 : 0, 0, 0 };
      float *pad_ref = calloc(pad_elems, 4);
      float *pad_alt = calloc(pad_elems, 4);
      float *wt_buf = calloc(wt_elems, 4);
      float *dw_bias = calloc(N, 4);
      float *dw_ref = calloc(dw_elems, 4);
      float *dw_d1 = calloc(dw_elems, 4);
      float *dw_d2 = calloc(dw_elems, 4);
      float *proj_b = calloc(pb_elems, 4);
      float *proj_bias = calloc(proj_n, 4);
      float *proj_c_ref = calloc(pc_elems, 4);
      float *proj_c_alt = calloc(pc_elems, 4);
      float *proj_cp_alt = calloc(pc_elems, 4);

      uint32_t rng = 0xc300u + ci * 71u;
      for (size_t i = 0; i < a_elems; ++i) a_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < b_elems; ++i) b_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) bias_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < wt_elems; ++i) wt_buf[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < N; ++i) dw_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < pb_elems; ++i) proj_b[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < proj_n; ++i) proj_bias[i] = gemm_test_rand_f32(&rng);
      for (size_t i = 0; i < pc_elems; ++i) {
         proj_c_alt[i] = -33.5f;
         proj_cp_alt[i] = -44.5f;
      }

      /* CPU reference: Expand + relu6 -> PAD4 -> DW 3x3 s=2 + relu6 -> 1x1 Project. */
      for (uint32_t p = 0; p < M; ++p) {
         for (uint32_t c = 0; c < N; ++c) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < K; ++kk)
               acc = fmaf(a_buf[(size_t)p * K + kk], b_buf[(size_t)kk * N + c], acc);
            acc += bias_buf[c];
            c_ref[(size_t)p * N + c] = fminf(fmaxf(acc, 0.0f), 6.0f);
         }
      }
      for (uint32_t oy = 0; oy < out_h; ++oy) {
         for (uint32_t ox = 0; ox < out_w; ++ox) {
            for (uint32_t c = 0; c < N; ++c) {
               const int32_t iy = (int32_t)oy - pad_b[1];
               const int32_t ix = (int32_t)ox - pad_b[2];
               pad_ref[((size_t)oy * out_w + ox) * N + c] =
                  (iy >= 0 && iy < (int32_t)in_h && ix >= 0 && ix < (int32_t)in_w)
                     ? c_ref[((size_t)iy * in_w + ix) * N + c]
                     : 0.0f;
            }
         }
      }
      for (uint32_t oy = 0; oy < dw_oh; ++oy) {
         for (uint32_t ox = 0; ox < dw_ow; ++ox) {
            for (uint32_t c = 0; c < N; ++c) {
               float acc = 0.0f;
               for (uint32_t ky = 0; ky < 3; ++ky) {
                  for (uint32_t kx = 0; kx < 3; ++kx) {
                     const float iv = pad_ref[((size_t)(oy * 2 + ky) * out_w + (ox * 2 + kx)) * N + c];
                     acc = ref_mul_add_unfused(iv, wt_buf[(size_t)(ky * 3 + kx) * N + c], acc);
                  }
               }
               acc += dw_bias[c];
               dw_ref[((size_t)oy * dw_ow + ox) * N + c] = fminf(fmaxf(acc, 0.0f), 6.0f);
            }
         }
      }
      for (uint32_t p = 0; p < M_pj; ++p) {
         for (uint32_t c = 0; c < proj_n; ++c) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < N; ++kk)
               acc = fmaf(dw_ref[(size_t)p * N + kk], proj_b[(size_t)kk * proj_n + c], acc);
            proj_c_ref[(size_t)p * proj_n + c] = acc + proj_bias[c];
         }
      }

      struct gemm_push_layout push = {
         .k = (int32_t)K, .n = (int32_t)N, .m = (int32_t)M, .bias_mode = 1,
      };
      uint8_t push_buf[256] = {0};
      memcpy(push_buf, &push, sizeof(push));

      struct hk_eval_ssbo sa[16];
      unsigned n_sa = 0;
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 0, a_buf, a_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 1, b_buf, b_elems * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 2, bias_buf, (size_t)N * 4, 0 };
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 3, c_alt, c_elems * 4, 0 };
      const uint64_t pad_dst_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 10, pad_alt, pad_elems * 4, 0 };
      const uint64_t pad_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 11, pad_b, sizeof(pad_b), 0 };
      const uint64_t wt_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 12, wt_buf, wt_elems * 4, 0 };
      const uint64_t dw_bias_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 13, dw_bias, (size_t)N * 4, 0 };
      const uint64_t dw_d1_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 14, dw_d1, dw_elems * 4, 0 };
      const uint64_t dw_d2_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 15, dw_d2, dw_elems * 4, 0 };
      const uint64_t proj_b_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 16, proj_b, pb_elems * 4, 0 };
      const uint64_t proj_bias_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 17, proj_bias, (size_t)proj_n * 4, 0 };
      const uint64_t proj_c_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 18, proj_c_alt, pc_elems * 4, 0 };
      const uint64_t proj_cp_va = hk_eval_make_vaddr(n_sa, 0);
      sa[n_sa++] = (struct hk_eval_ssbo){ 0, 19, proj_cp_alt, pc_elems * 4, 0 };

      const uint32_t dual_off = hk_dual_write_push_offset(ug_m.push_bytes);
      const uint32_t proj_off = hk_dual_gemm_dw_proj_push_offset(ug_m.push_bytes);
      struct hk_dual_gemm_dw_push dp = {
         .pad_dst_va_lo = (uint32_t)pad_dst_va, .pad_dst_va_hi = (uint32_t)(pad_dst_va >> 32),
         .pad_va_lo = (uint32_t)pad_va, .pad_va_hi = (uint32_t)(pad_va >> 32),
         .dw_wt_va_lo = (uint32_t)wt_va, .dw_wt_va_hi = (uint32_t)(wt_va >> 32),
         .dw_bias_va_lo = (uint32_t)dw_bias_va, .dw_bias_va_hi = (uint32_t)(dw_bias_va >> 32),
         .dw_dst1_va_lo = (uint32_t)dw_d1_va, .dw_dst1_va_hi = (uint32_t)(dw_d1_va >> 32),
         .dw_dst2_va_lo = (uint32_t)dw_d2_va, .dw_dst2_va_hi = (uint32_t)(dw_d2_va >> 32),
         .in_wh = in_w | (in_h << 16),
         .out_wh = out_w | (out_h << 16),
         .pad_idx = { 0, 1, 2, 3 },
         .flags = HK_GEMM_DW_FLAG_HAS_BIAS | HK_GEMM_DW_FLAG_HAS_DW_DST2,
      };
      hk_dual_gemm_dw_push_init_udiv(&dp, in_w >> 3);
      memcpy(push_buf + dual_off, &dp, sizeof(dp));

      struct hk_dual_gemm_dw_proj_push pp = {
         .proj_b_va_lo = (uint32_t)proj_b_va, .proj_b_va_hi = (uint32_t)(proj_b_va >> 32),
         .proj_bias_va_lo = (uint32_t)proj_bias_va, .proj_bias_va_hi = (uint32_t)(proj_bias_va >> 32),
         .proj_c_va_lo = (uint32_t)proj_c_va, .proj_c_va_hi = (uint32_t)(proj_c_va >> 32),
         .proj_copy_va_lo = (uint32_t)proj_cp_va, .proj_copy_va_hi = (uint32_t)(proj_cp_va >> 32),
         .proj_n = proj_n,
         .proj_flags = HK_GEMM_DW_PROJ_FLAG_HAS_BIAS | HK_GEMM_DW_PROJ_FLAG_HAS_COPY |
                       (cases[ci].skip_proj_c ? HK_GEMM_DW_PROJ_FLAG_SKIP_C : 0u),
      };
      memcpy(push_buf + proj_off, &pp, sizeof(pp));

      const uint32_t ag[3] = { 1u, (in_h >> 3) * (in_w >> 3), 1u };
      hk_eval_dispatch_wg(alt, ag, sa, n_sa, push_buf, proj_off + sizeof(pp));

      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = DIV_ROUND_UP(K, 16u);
      params[HK_PACK_LOOP_PARAM_GEMM_K] = DIV_ROUND_UP(K, 4u);
      params[HK_PACK_LOOP_PARAM_DW_KH] = N / tile_n;
      params[HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS] = DIV_ROUND_UP(pad_elems, (uint64_t)ag[1] * 128u);
      uint64_t bound = 0, eff = 0;
      HK_EVAL_CHECK(hk_pack_eval_loop_bound(&lb, params, &bound));
      HK_EVAL_CHECK(bound >= hk_eval_last_max_loop_backedges + 1);
      HK_EVAL_CHECK(hk_pack_eval_effective_instrs(&lb, 300u, params, &eff));
      HK_EVAL_CHECK(eff > 0 && eff < 5000000u);

      for (size_t i = 0; i < c_elems; ++i)
         HK_EVAL_CHECK(hk_eval_f2u(c_alt[i]) == hk_eval_f2u(c_ref[i]));
      for (size_t i = 0; i < pad_elems; ++i)
         HK_EVAL_CHECK(hk_eval_f2u(pad_alt[i]) == hk_eval_f2u(pad_ref[i]));
      for (size_t i = 0; i < dw_elems; ++i) {
         HK_EVAL_CHECK(hk_eval_f2u(dw_d1[i]) == hk_eval_f2u(dw_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(dw_d2[i]) == hk_eval_f2u(dw_ref[i]));
      }
      for (size_t i = 0; i < pc_elems; ++i) {
         if (cases[ci].skip_proj_c)
            HK_EVAL_CHECK(hk_eval_f2u(proj_c_alt[i]) == hk_eval_f2u(-33.5f));
         else
            HK_EVAL_CHECK(hk_eval_f2u(proj_c_alt[i]) == hk_eval_f2u(proj_c_ref[i]));
         HK_EVAL_CHECK(hk_eval_f2u(proj_cp_alt[i]) == hk_eval_f2u(proj_c_ref[i]));
      }

      free(a_buf); free(b_buf); free(bias_buf); free(c_ref); free(c_alt);
      free(pad_ref); free(pad_alt); free(wt_buf); free(dw_bias); free(dw_ref);
      free(dw_d1); free(dw_d2); free(proj_b); free(proj_bias);
      free(proj_c_ref); free(proj_c_alt); free(proj_cp_alt);
      ralloc_free(alt);
   }
   ralloc_free(ug_pre);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_depthwise_clamp_footprint();
   test_gemm_clamp_footprint();
   test_clamp_match_and_splice();
   test_depthwise_vec4_and_fusion();
   test_gemm_match_and_splice();
   test_unit_axis_wgid_compute_pair();
   test_gemm_cmat_certify_and_build();
   test_gemm_cmat_guarded_specialize_and_shape_sweep();
   test_gemm_cmat_k32_small_tiles();
   test_dual_write_fusion();
   test_gemm_dual_fadd_fusion();
   test_dual_copy2_fusion();
   test_gemm_im2col_a_fusion();
   test_gemm_dw_a_fusion();
   test_gemm_exp_dw_a_fusion();
   test_gemm_dw_fusion();
   test_gemm_dw_proj_fusion();
   test_nir_derived_pack_loop_bounds();
   test_fast_udivmod_general();
   test_depthwise_shape_sweep_cpu_ref();
   test_g17p_compile_occupancy();
   test_gemm_fadd_postpass_g16g_addresses();
   test_warm_cache_producer_fusion();
   glsl_type_singleton_decref();
   printf("test-hk-compute-pair: PASS\n");
   return 0;
}
