/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "hk_nir_eval_test.h"
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_nir_wgid.h"

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

   nir_shader *vec4 = hk_nir_build_depthwise_vec4(orig, &info, ep);
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
   const uint32_t out_pairs = DIV_ROUND_UP((uint32_t)cfg->out_w, 2u);
   const uint32_t v4_grid[3] = {
      DIV_ROUND_UP(out_pairs * ((uint32_t)cfg->channels / 4u),
                   vec4->info.workgroup_size[0]),
      DIV_ROUND_UP((uint32_t)cfg->out_h, vec4->info.workgroup_size[1]),
      1,
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
      nir_shader *vec4 = hk_nir_build_depthwise_vec4(dw32, &info, &relu6);
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
         DIV_ROUND_UP(out_pairs * ((uint32_t)cfg.channels / 4u),
                      vec4->info.workgroup_size[0]),
         DIV_ROUND_UP((uint32_t)cfg.out_h, vec4->info.workgroup_size[1]),
         1,
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
   glsl_type_singleton_decref();
   printf("test-hk-compute-pair: PASS\n");
   return 0;
}
