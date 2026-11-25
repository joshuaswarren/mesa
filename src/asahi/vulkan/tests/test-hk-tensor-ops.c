/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "hk_nir_tensor_ops.h"
#include "hk_nir_eval_test.h"

static nir_deref_instr *
make_ssbo_elem_deref(nir_builder *b, unsigned set, unsigned binding,
                     const struct glsl_type *elem_type, bool read_only,
                     nir_def *index)
{
   const struct glsl_type *arr_type = glsl_array_type(elem_type, 0, 4);
   struct glsl_struct_field field = {
      .type = arr_type,
      .name = "data",
      .offset = 0,
   };
   const struct glsl_type *block_type =
      glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false,
                          "Buf");

   nir_def *idx = nir_vulkan_resource_index(
      b, 3, 32, nir_imm_int(b, 0),
      .desc_set = set, .binding = binding,
      .desc_type = nir_descriptor_type_storage_buffer,
      .resource_type = read_only ? nir_resource_type_read_only_storage_buffer
                                 : nir_resource_type_read_write_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, 3, 32, idx, .desc_type = nir_descriptor_type_storage_buffer);
   nir_deref_instr *root =
      nir_build_deref_cast(b, desc, nir_var_mem_ssbo, block_type, 0);
   root->cast.align_mul = 4;
   root->cast.align_offset = 0;
   nir_deref_instr *arr = nir_build_deref_struct(b, root, 0);
   nir_deref_instr *elem = nir_build_deref_array(b, arr, index);
   elem->arr.in_bounds = true;
   return elem;
}

static nir_def *
load_push_u32(nir_builder *b, uint16_t off, uint16_t range)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, off),
                                 .base = 0, .range = range, .align_mul = 4);
}

/* Matches SPIR-V's `if (gx >= xlim || gy >= ylim) return;` lowered via
 * bcsel OR pre-peephole short-circuit phi (where inot(ige) folded to ilt). */
static nir_if *
push_2d_guard(nir_builder *b, nir_def *gx, nir_def *gy,
              nir_def *xlim, nir_def *ylim, int mode)
{
   if (mode == 2) {
      /* Pre-lowering short-circuit phi with algebraic-folded ilt condition */
      nir_def *t = nir_imm_true(b);
      nir_def *x_in = nir_ilt(b, gx, xlim);
      nir_if *sc = nir_push_if(b, x_in);
      nir_def *y_oob = nir_ige(b, gy, ylim);
      nir_push_else(b, sc);
      nir_pop_if(b, sc);
      nir_def *oob = nir_if_phi(b, y_oob, t);
      nir_if *outer = nir_push_if(b, oob);
      nir_push_else(b, outer);
      return outer;
   }
   nir_def *cond;
   if (mode == 1) {
      nir_def *x_in = nir_ilt(b, gx, xlim);
      nir_def *y_oob = nir_ige(b, gy, ylim);
      cond = nir_inot(b, nir_bcsel(b, x_in, y_oob, nir_imm_true(b)));
   } else {
      cond = nir_iand(b, nir_ilt(b, gx, xlim), nir_ilt(b, gy, ylim));
   }
   return nir_push_if(b, cond);
}

static nir_shader *
build_concat_x_orig(int guard_mode, bool has_dim_n, bool has_dst_off, bool bad_stride)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "concat_x_orig");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->num_uniforms = 28;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   if (guard_mode == 2) {
      nir_def *ls = nir_imm_ivec3(&b, 64, 1, 1);
      gid = nir_iadd(&b, gid, nir_imul(&b, nir_load_base_workgroup_id(&b, 32), ls));
   }
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   /* Push layout: [0]=dim_n, [4]=dst_sy, [8]=dst_ox, [12]=dst_sq,
    *              [16]=dim_q, [20]=dim_h, [24]=dim_c */
   nir_def *dim_n = load_push_u32(&b, 0, 28);
   nir_def *dst_sy = load_push_u32(&b, 4, 28);
   nir_def *dst_ox = has_dst_off ? load_push_u32(&b, 8, 28) : nir_imm_int(&b, 0);
   nir_def *dst_sq = load_push_u32(&b, 12, 28);
   nir_def *dim_q = load_push_u32(&b, 16, 28);
   nir_def *dim_h = load_push_u32(&b, 20, 28);
   nir_def *dim_c = load_push_u32(&b, 24, 28);

   nir_def *xlim = nir_imul(&b, dim_q, dim_c);
   nir_def *ylim = has_dim_n ? nir_imul(&b, dim_n, dim_h) : dim_h;
   nir_if *nif = push_2d_guard(&b, gx, gy, xlim, ylim, guard_mode);
   {
      nir_def *q = nir_idiv(&b, gx, dim_c);
      nir_def *r = nir_imod(&b, gx, dim_c);
      nir_def *row_off = guard_mode == 2
                            ? nir_imul(&b, nir_imul(&b, gy, dim_q), dim_c)
                            : nir_imul(&b, gy, xlim);
      nir_def *src_idx = nir_iadd(&b, row_off, gx);
      nir_def *dst_idx = nir_iadd(
         &b,
         nir_iadd(&b, nir_imul(&b, q, dst_sq), nir_imul(&b, gy, dst_sy)),
         nir_iadd(&b, dst_ox, r));
      if (bad_stride)
         dst_idx = nir_iadd_imm(&b, dst_idx, 1);

      nir_deref_instr *src =
         make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, src_idx);
      nir_def *v = nir_load_deref_with_access(
         &b, src, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_deref_instr *dst =
         make_ssbo_elem_deref(&b, 0, 1, glsl_float_type(), false, dst_idx);
      nir_store_deref_with_access(&b, dst, v, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, nif);

   nir_validate_shader(b.shader, "concat_x_orig");
   return b.shader;
}

static nir_shader *
build_concat_y_orig(int guard_mode, bool has_dim_w, bool has_dst_yo)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "concat_y_orig");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 16;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->num_uniforms = 24;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   if (guard_mode == 2) {
      nir_def *ls = nir_imm_ivec3(&b, 32, 16, 1);
      gid = nir_iadd(&b, gid, nir_imul(&b, nir_load_base_workgroup_id(&b, 32), ls));
   }
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   /* Push layout: [0]=dim_n, [4]=dst_yo, [8]=dst_sn, [12]=dim_w, [16]=dim_h, [20]=dim_c */
   nir_def *dim_n = load_push_u32(&b, 0, 24);
   nir_def *dst_yo = has_dst_yo ? load_push_u32(&b, 4, 24) : nir_imm_int(&b, 0);
   nir_def *dst_sn = load_push_u32(&b, 8, 24);
   nir_def *dim_w = load_push_u32(&b, 12, 24);
   nir_def *dim_h = load_push_u32(&b, 16, 24);
   nir_def *dim_c = load_push_u32(&b, 20, 24);

   nir_def *xlim = has_dim_w ? nir_imul(&b, dim_w, dim_c) : dim_c;
   nir_def *ylim = nir_imul(&b, dim_n, dim_h);
   nir_if *nif = push_2d_guard(&b, gx, gy, xlim, ylim, guard_mode);
   {
      nir_def *n = nir_idiv(&b, gy, dim_h);
      nir_def *h = nir_imod(&b, gy, dim_h);
      nir_def *src_row = (guard_mode == 2 && has_dim_w)
                            ? nir_imul(&b, nir_imul(&b, gy, dim_w), dim_c)
                            : nir_imul(&b, gy, xlim);
      nir_def *src_idx = nir_iadd(&b, src_row, gx);
      nir_def *dst_row = nir_iadd(&b, nir_imul(&b, n, dst_sn), nir_iadd(&b, dst_yo, h));
      nir_def *dst_off = (guard_mode == 2 && has_dim_w)
                            ? nir_imul(&b, nir_imul(&b, dst_row, dim_w), dim_c)
                            : nir_imul(&b, dst_row, xlim);
      nir_def *dst_idx = nir_iadd(&b, dst_off, gx);

      nir_deref_instr *src =
         make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, src_idx);
      nir_def *v = nir_load_deref_with_access(
         &b, src, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_deref_instr *dst =
         make_ssbo_elem_deref(&b, 0, 1, glsl_float_type(), false, dst_idx);
      nir_store_deref_with_access(&b, dst, v, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, nif);

   nir_validate_shader(b.shader, "concat_y_orig");
   return b.shader;
}

static nir_shader *
build_resample_orig(int guard_mode, bool bad_floor)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "resample_orig");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 2;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->num_uniforms = 28;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   /* Push layout: [0]=scale_x, [4]=scale_y, [8]=in_w, [16]=out_w, [20]=out_h, [24]=ch */
   nir_def *scale_x = load_push_u32(&b, 0, 28);
   nir_def *scale_y = load_push_u32(&b, 4, 28);
   nir_def *in_w = load_push_u32(&b, 8, 28);
   nir_def *out_w = load_push_u32(&b, 16, 28);
   nir_def *out_h = load_push_u32(&b, 20, 28);
   nir_def *ch = load_push_u32(&b, 24, 28);

   nir_def *xlim = nir_imul(&b, out_w, ch);
   nir_if *nif = push_2d_guard(&b, gx, gy, xlim, out_h, guard_mode);
   {
      nir_def *ox = nir_idiv(&b, gx, ch);
      nir_def *c = nir_imod(&b, gx, ch);
      nir_def *fx = nir_fmul(&b, nir_i2f32(&b, ox), scale_x);
      nir_def *fy = nir_fmul(&b, nir_i2f32(&b, gy), scale_y);
      nir_def *sx = nir_f2i32(&b, bad_floor ? nir_fceil(&b, fx) : nir_ffloor(&b, fx));
      nir_def *sy = nir_f2i32(&b, nir_ffloor(&b, fy));
      nir_def *src_idx =
         guard_mode == 2
            ? nir_iadd(&b, nir_iadd(&b, nir_imul(&b, nir_imul(&b, sy, in_w), ch),
                                    nir_imul(&b, sx, ch)), c)
            : nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, sy, in_w), sx), ch), c);
      nir_def *dst_idx =
         guard_mode == 2
            ? nir_iadd(&b, nir_iadd(&b, nir_imul(&b, nir_imul(&b, gy, out_w), ch),
                                    nir_imul(&b, ox, ch)), c)
            : nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, gy, out_w), ox), ch), c);

      nir_deref_instr *src =
         make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, src_idx);
      nir_def *v = nir_load_deref_with_access(
         &b, src, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_deref_instr *dst =
         make_ssbo_elem_deref(&b, 0, 1, glsl_float_type(), false, dst_idx);
      nir_store_deref_with_access(&b, dst, v, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, nif);

   nir_validate_shader(b.shader, "resample_orig");
   return b.shader;
}

static void
check_concat_x_case(const nir_shader *orig, const nir_shader *alt,
                    bool use_3d, unsigned vec,
                    int32_t dim_n, int32_t dim_h, int32_t dim_q, int32_t dim_c,
                    int32_t dst_sy, int32_t dst_ox, int32_t dst_sq)
{
   const int32_t push[7] = { dim_n, dst_sy, dst_ox, dst_sq, dim_q, dim_h, dim_c };
   const uint32_t x_lim = (uint32_t)dim_q * (uint32_t)dim_c;
   const uint32_t y_lim = (uint32_t)dim_n * (uint32_t)dim_h;
   const size_t src_n = (size_t)x_lim * y_lim;
   const size_t dst_n = (size_t)(dim_q - 1) * dst_sq + (size_t)(y_lim - 1) * dst_sy +
                        (size_t)dst_ox + (size_t)dim_c + 16;

   uint32_t *src = calloc(src_n, sizeof(uint32_t));
   uint32_t *ref = calloc(dst_n, sizeof(uint32_t));
   uint32_t *got = calloc(dst_n, sizeof(uint32_t));
   HK_EVAL_CHECK(src && ref && got);
   for (size_t i = 0; i < src_n; ++i)
      src[i] = 0x3f800000u ^ (uint32_t)(i * 2654435761u);
   memset(ref, 0x5a, dst_n * sizeof(uint32_t));
   memset(got, 0x5a, dst_n * sizeof(uint32_t));

   const struct hk_eval_ssbo ref_bufs[2] = {
      { .set = 0, .binding = 0, .data = src, .size = src_n * 4 },
      { .set = 0, .binding = 1, .data = ref, .size = dst_n * 4 },
   };
   const uint32_t orig_grid[3] = {
      DIV_ROUND_UP(x_lim, orig->info.workgroup_size[0]),
      DIV_ROUND_UP(y_lim, orig->info.workgroup_size[1]),
      1,
   };
   hk_eval_dispatch((nir_shader *)orig, orig_grid, ref_bufs, 2, push, sizeof(push));

   const struct hk_eval_ssbo got_bufs[2] = {
      { .set = 0, .binding = 0, .data = src, .size = src_n * 4 },
      { .set = 0, .binding = 1, .data = got, .size = dst_n * 4 },
   };
   uint32_t alt_grid[3];
   if (!use_3d) {
      const bool contig = (dst_sq == dim_c);
      const uint32_t x_cnt = contig ? (x_lim / vec) : (uint32_t)dim_q;
      alt_grid[0] = DIV_ROUND_UP(x_cnt, alt->info.workgroup_size[0]);
      alt_grid[1] = DIV_ROUND_UP(y_lim, alt->info.workgroup_size[1]);
      alt_grid[2] = 1;
   } else {
      alt_grid[0] = DIV_ROUND_UP((uint32_t)dim_c / vec, alt->info.workgroup_size[0]);
      alt_grid[1] = DIV_ROUND_UP((uint32_t)dim_q, alt->info.workgroup_size[1]);
      alt_grid[2] = DIV_ROUND_UP(y_lim, alt->info.workgroup_size[2]);
   }
   hk_eval_dispatch((nir_shader *)alt, alt_grid, got_bufs, 2, push, sizeof(push));

   HK_EVAL_CHECK(memcmp(ref, got, dst_n * sizeof(uint32_t)) == 0);
   free(src);
   free(ref);
   free(got);
}

static void
test_concat_x(void)
{
   for (int mode = 1; mode <= 2; ++mode) {
      nir_shader *orig = build_concat_x_orig(mode, true, true, false);
      struct hk_tensor_op_info info = {0};
      nir_shader *p = NULL, *s = NULL;
      HK_EVAL_CHECK(hk_nir_match_and_build_tensor_op(orig, &info, &p, &s));
      HK_EVAL_CHECK(info.kind == HK_TENSOR_OP_CONCAT_X);
      HK_EVAL_CHECK(p && s);
      nir_validate_shader(p, "concat_x_primary");
      nir_validate_shader(s, "concat_x_secondary");

      /* Primary 2D contiguous vec4 / vec2 / scalar */
      check_concat_x_case(orig, p, false, 4, 2, 3, 5, 8, 40, 8, 8);
      check_concat_x_case(orig, p, false, 2, 1, 5, 3, 6, 20, 2, 6);
      check_concat_x_case(orig, p, false, 1, 1, 4, 3, 5, 17, 1, 5);
      /* Primary 2D strided dim_c in {4, 2, 1} */
      check_concat_x_case(orig, p, false, 4, 1, 4, 6, 4, 72, 4, 12);
      check_concat_x_case(orig, p, false, 2, 2, 3, 5, 2, 30, 2, 6);
      check_concat_x_case(orig, p, false, 1, 1, 3, 7, 1, 21, 1, 3);
      /* Secondary 3D general vec4 / vec2 / scalar */
      check_concat_x_case(orig, s, true, 4, 2, 5, 7, 16, 224, 16, 32);
      check_concat_x_case(orig, s, true, 2, 1, 6, 5, 6, 70, 4, 14);
      check_concat_x_case(orig, s, true, 1, 1, 4, 5, 3, 45, 2, 9);

      ralloc_free(p);
      ralloc_free(s);
      ralloc_free(orig);
   }

   /* Negative test: extra constant offset in dst_idx must be rejected. */
   struct hk_tensor_op_info info = {0};
   nir_shader *p = NULL, *s = NULL;
   nir_shader *bad = build_concat_x_orig(0, true, true, true);
   HK_EVAL_CHECK(!hk_nir_match_and_build_tensor_op(bad, &info, &p, &s));
   HK_EVAL_CHECK(!p && !s);
   ralloc_free(bad);
}

static void
check_concat_y_case(const nir_shader *orig, const nir_shader *alt,
                    bool use_1d, unsigned vec,
                    int32_t dim_n, int32_t dim_h, int32_t dim_w, int32_t dim_c,
                    int32_t dst_sn, int32_t dst_yo)
{
   const int32_t push[6] = { dim_n, dst_yo, dst_sn, dim_w, dim_h, dim_c };
   const uint32_t x_lim = (uint32_t)dim_w * (uint32_t)dim_c;
   const uint32_t y_lim = (uint32_t)dim_n * (uint32_t)dim_h;
   const size_t src_n = (size_t)x_lim * y_lim;
   const size_t dst_rows = (size_t)(dim_n - 1) * dst_sn + dst_yo + dim_h + 4;
   const size_t dst_n = dst_rows * x_lim;

   uint32_t *src = calloc(src_n, sizeof(uint32_t));
   uint32_t *ref = calloc(dst_n, sizeof(uint32_t));
   uint32_t *got = calloc(dst_n, sizeof(uint32_t));
   HK_EVAL_CHECK(src && ref && got);
   for (size_t i = 0; i < src_n; ++i)
      src[i] = 0x40000000u ^ (uint32_t)(i * 1140071481u);
   memset(ref, 0xa5, dst_n * sizeof(uint32_t));
   memset(got, 0xa5, dst_n * sizeof(uint32_t));

   const struct hk_eval_ssbo ref_bufs[2] = {
      { .set = 0, .binding = 0, .data = src, .size = src_n * 4 },
      { .set = 0, .binding = 1, .data = ref, .size = dst_n * 4 },
   };
   const uint32_t orig_grid[3] = {
      DIV_ROUND_UP(x_lim, orig->info.workgroup_size[0]),
      DIV_ROUND_UP(y_lim, orig->info.workgroup_size[1]),
      1,
   };
   hk_eval_dispatch((nir_shader *)orig, orig_grid, ref_bufs, 2, push, sizeof(push));

   const struct hk_eval_ssbo got_bufs[2] = {
      { .set = 0, .binding = 0, .data = src, .size = src_n * 4 },
      { .set = 0, .binding = 1, .data = got, .size = dst_n * 4 },
   };
   uint32_t alt_grid[3];
   if (use_1d) {
      const uint32_t total4 = (uint32_t)(src_n / 4);
      const uint32_t wg_chunk = alt->info.workgroup_size[0] * 4u;
      alt_grid[0] = DIV_ROUND_UP(total4, wg_chunk);
      alt_grid[1] = 1;
      alt_grid[2] = 1;
   } else {
      alt_grid[0] = DIV_ROUND_UP(x_lim / vec, alt->info.workgroup_size[0]);
      alt_grid[1] = DIV_ROUND_UP((uint32_t)dim_h, alt->info.workgroup_size[1]);
      alt_grid[2] = DIV_ROUND_UP((uint32_t)dim_n, alt->info.workgroup_size[2]);
   }
   hk_eval_dispatch((nir_shader *)alt, alt_grid, got_bufs, 2, push, sizeof(push));

   HK_EVAL_CHECK(memcmp(ref, got, dst_n * sizeof(uint32_t)) == 0);
   free(src);
   free(ref);
   free(got);
}

static void
test_concat_y(void)
{
   for (int mode = 1; mode <= 2; ++mode) {
      nir_shader *orig = build_concat_y_orig(mode, true, true);
      struct hk_tensor_op_info info = {0};
      nir_shader *p = NULL, *s = NULL;
      HK_EVAL_CHECK(hk_nir_match_and_build_tensor_op(orig, &info, &p, &s));
      HK_EVAL_CHECK(info.kind == HK_TENSOR_OP_CONCAT_Y);
      HK_EVAL_CHECK(p && s);
      nir_validate_shader(p, "concat_y_primary");
      nir_validate_shader(s, "concat_y_secondary");

      /* Primary 1D 4x-vec4 (both full-WG and tail-WG sizes) */
      check_concat_y_case(orig, p, true, 4, 1, 7, 16, 4, 12, 3);
      check_concat_y_case(orig, p, true, 4, 1, 64, 16, 8, 80, 11);
      /* Secondary 3D (dim_n > 1, vec4 / vec2 / scalar) */
      check_concat_y_case(orig, s, false, 4, 3, 5, 4, 8, 9, 2);
      check_concat_y_case(orig, s, false, 2, 2, 4, 3, 2, 7, 1);
      check_concat_y_case(orig, s, false, 1, 2, 3, 5, 3, 6, 2);

      ralloc_free(p);
      ralloc_free(s);
      ralloc_free(orig);
   }
}

static void
check_resample_case(const nir_shader *orig, const nir_shader *alt, unsigned vec,
                    int32_t in_w, int32_t in_h, int32_t out_w, int32_t out_h,
                    int32_t ch, float scale_x, float scale_y)
{
   int32_t push[7] = {0};
   memcpy(&push[0], &scale_x, 4);
   memcpy(&push[1], &scale_y, 4);
   push[2] = in_w;
   push[4] = out_w;
   push[5] = out_h;
   push[6] = ch;

   const size_t src_n = (size_t)in_w * in_h * ch;
   const size_t dst_n = (size_t)out_w * out_h * ch;
   uint32_t *src = calloc(src_n, sizeof(uint32_t));
   uint32_t *ref = calloc(dst_n, sizeof(uint32_t));
   uint32_t *got = calloc(dst_n, sizeof(uint32_t));
   HK_EVAL_CHECK(src && ref && got);
   for (size_t i = 0; i < src_n; ++i)
      src[i] = 0x41200000u ^ (uint32_t)(i * 2246822519u);

   const struct hk_eval_ssbo ref_bufs[2] = {
      { .set = 0, .binding = 0, .data = src, .size = src_n * 4 },
      { .set = 0, .binding = 1, .data = ref, .size = dst_n * 4 },
   };
   const uint32_t orig_grid[3] = {
      DIV_ROUND_UP((uint32_t)(out_w * ch), orig->info.workgroup_size[0]),
      DIV_ROUND_UP((uint32_t)out_h, orig->info.workgroup_size[1]),
      1,
   };
   hk_eval_dispatch((nir_shader *)orig, orig_grid, ref_bufs, 2, push, sizeof(push));

   const struct hk_eval_ssbo got_bufs[2] = {
      { .set = 0, .binding = 0, .data = src, .size = src_n * 4 },
      { .set = 0, .binding = 1, .data = got, .size = dst_n * 4 },
   };
   const uint32_t alt_grid[3] = {
      DIV_ROUND_UP((uint32_t)ch / vec, alt->info.workgroup_size[0]),
      DIV_ROUND_UP((uint32_t)out_w, alt->info.workgroup_size[1]),
      DIV_ROUND_UP((uint32_t)out_h, alt->info.workgroup_size[2]),
   };
   hk_eval_dispatch((nir_shader *)alt, alt_grid, got_bufs, 2, push, sizeof(push));

   HK_EVAL_CHECK(memcmp(ref, got, dst_n * sizeof(uint32_t)) == 0);
   free(src);
   free(ref);
   free(got);
}

static void
test_resample_nearest(void)
{
   for (int mode = 1; mode <= 2; ++mode) {
      nir_shader *orig = build_resample_orig(mode, false);
      struct hk_tensor_op_info info = {0};
      nir_shader *p = NULL, *s = NULL;
      HK_EVAL_CHECK(hk_nir_match_and_build_tensor_op(orig, &info, &p, &s));
      HK_EVAL_CHECK(info.kind == HK_TENSOR_OP_RESAMPLE_NEAREST);
      HK_EVAL_CHECK(p && !s);
      nir_validate_shader(p, "resample_primary");

      /* Exact 2x upsample (vec4), fractional scale (vec2), odd channels (scalar) */
      check_resample_case(orig, p, 4, 8, 6, 16, 12, 16, 0.5f, 0.5f);
      check_resample_case(orig, p, 2, 7, 5, 11, 9, 6, 7.0f / 11.0f, 5.0f / 9.0f);
      check_resample_case(orig, p, 1, 5, 4, 9, 7, 3, 5.0f / 9.0f, 4.0f / 7.0f);

      ralloc_free(p);
      ralloc_free(orig);
   }

   /* Negative test: fceil instead of ffloor must be rejected. */
   struct hk_tensor_op_info info = {0};
   nir_shader *p = NULL, *s = NULL;
   nir_shader *bad = build_resample_orig(1, true);
   HK_EVAL_CHECK(!hk_nir_match_and_build_tensor_op(bad, &info, &p, &s));
   HK_EVAL_CHECK(!p && !s);
   ralloc_free(bad);
}

static nir_shader *
build_mock_gemm_cmat(uint16_t push_bytes)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "mock_gemm_cmat");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 4;
   b.shader->num_uniforms = push_bytes;

   nir_def *gx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *n = load_push_u32(&b, 0, push_bytes);
   nir_def *idx0 = nir_ishl_imm(&b, gx, 1);

   const struct glsl_type *arr_t = glsl_array_type(glsl_float_type(), 0, 4);
   nir_def *r_in = nir_vulkan_resource_index(
      &b, 3, 32, nir_imm_int(&b, 0),
      .desc_set = 0, .binding = 0,
      .desc_type = nir_descriptor_type_storage_buffer,
      .resource_type = nir_resource_type_read_only_storage_buffer);
   nir_def *d_in = nir_load_vulkan_descriptor(
      &b, 3, 32, r_in, .desc_type = nir_descriptor_type_storage_buffer);
   nir_deref_instr *a_in =
      nir_build_deref_cast_with_alignment(&b, d_in, nir_var_mem_ssbo, arr_t, 0, 4, 0);

   nir_def *r_out = nir_vulkan_resource_index(
      &b, 3, 32, nir_imm_int(&b, 0),
      .desc_set = 0, .binding = 3,
      .desc_type = nir_descriptor_type_storage_buffer,
      .resource_type = nir_resource_type_read_write_storage_buffer);
   nir_def *d_out = nir_load_vulkan_descriptor(
      &b, 3, 32, r_out, .desc_type = nir_descriptor_type_storage_buffer);
   nir_deref_instr *a_out =
      nir_build_deref_cast_with_alignment(&b, d_out, nir_var_mem_ssbo, arr_t, 0, 4, 0);

   /* Fast vec2 store path when idx0 + 2 <= n, scalar tail otherwise. */
   nir_if *fast = nir_push_if(&b, nir_uge(&b, n, nir_iadd_imm(&b, idx0, 2)));
   {
      nir_deref_instr *ei = nir_build_deref_array(&b, a_in, idx0);
      nir_def *v = nir_load_global(
         &b, 2, 32,
         nir_deref_buffer_address(&b, 1, 64, &ei->def,
                                  .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_deref_instr *eo = nir_build_deref_array(&b, a_out, idx0);
      nir_store_global(
         &b, v,
         nir_deref_buffer_address(&b, 1, 64, &eo->def, .access = ACCESS_NON_READABLE),
         .align_mul = 4, .write_mask = 3, .access = ACCESS_NON_READABLE);
   }
   nir_push_else(&b, fast);
   {
      nir_if *ok = nir_push_if(&b, nir_ult(&b, idx0, n));
      {
         nir_deref_instr *ei = nir_build_deref_array(&b, a_in, idx0);
         nir_def *v = nir_load_global(
            &b, 1, 32,
            nir_deref_buffer_address(&b, 1, 64, &ei->def,
                                     .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
            .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
         nir_deref_instr *eo = nir_build_deref_array(&b, a_out, idx0);
         nir_store_global(
            &b, v,
            nir_deref_buffer_address(&b, 1, 64, &eo->def, .access = ACCESS_NON_READABLE),
            .align_mul = 4, .write_mask = 1, .access = ACCESS_NON_READABLE);
      }
      nir_pop_if(&b, ok);
   }
   nir_pop_if(&b, fast);

   nir_validate_shader(b.shader, "mock_gemm_cmat");
   return b.shader;
}

static void
test_gemm_fadd_postpass(void)
{
   const enum hk_compute_pair_dual_write modes[4] = {
      HK_COMPUTE_PAIR_DUAL_FADD_INPLACE,
      HK_COMPUTE_PAIR_DUAL_FADD_OUT,
      HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE,
      HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT,
   };
   const uint32_t n = 19; /* exercises both vec2 fast path and scalar tail */
   const uint16_t base_pb = 16;
   struct hk_nir_gemm_output info = {
      .output = { .set = 0, .binding = 3, .offset_B = 0 },
      .push_bytes = base_pb,
   };

   float in_a[24], res[24], out1[24], dst1[24], dst2[24];
   for (unsigned i = 0; i < 24; ++i) {
      in_a[i] = (float)(i + 1) * 0.25f;
      res[i] = (float)(24 - i) * -0.5f;
   }

   for (unsigned m = 0; m < 4; ++m) {
      const enum hk_compute_pair_dual_write mode = modes[m];
      const bool inplace = mode == HK_COMPUTE_PAIR_DUAL_FADD_INPLACE ||
                           mode == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE;
      const bool has_copy = mode == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE ||
                            mode == HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT;
      nir_shader *s = build_mock_gemm_cmat(base_pb);
      HK_EVAL_CHECK(hk_nir_apply_gemm_fadd(s, &info, mode));
      nir_validate_shader(s, "gemm_fadd");

      memset(out1, 0, sizeof(out1));
      memset(dst1, 0, sizeof(dst1));
      memset(dst2, 0, sizeof(dst2));
      const struct hk_eval_ssbo bufs[5] = {
         { .set = 0, .binding = 0, .data = in_a, .size = sizeof(in_a) },
         { .set = 0, .binding = 3, .data = out1, .size = sizeof(out1) },
         { .set = 1, .binding = 0, .data = res, .size = sizeof(res) },
         { .set = 1, .binding = 1, .data = dst1, .size = sizeof(dst1) },
         { .set = 1, .binding = 2, .data = dst2, .size = sizeof(dst2) },
      };
      const uint64_t out_va = hk_eval_make_vaddr(1, 0);
      const uint64_t res_delta = hk_eval_make_vaddr(2, 0) - out_va;
      const uint64_t dst1_delta = hk_eval_make_vaddr(3, 0) - out_va;
      const uint64_t dst2_delta = hk_eval_make_vaddr(4, 0) - out_va;

      uint8_t push[64] = {0};
      memcpy(push, &n, 4);
      struct hk_dual_fadd_push dp = {
         .res_delta_lo = (uint32_t)res_delta,
         .res_delta_hi = (uint32_t)(res_delta >> 32),
         .dst1_delta_lo = (uint32_t)dst1_delta,
         .dst1_delta_hi = (uint32_t)(dst1_delta >> 32),
         .dst2_delta_lo = (uint32_t)dst2_delta,
         .dst2_delta_hi = (uint32_t)(dst2_delta >> 32),
      };
      memcpy(push + hk_dual_write_push_offset(base_pb), &dp, sizeof(dp));

      const uint32_t grid[3] = { DIV_ROUND_UP(DIV_ROUND_UP(n, 2u), 8u), 1, 1 };
      hk_eval_dispatch(s, grid, bufs, 5, push, sizeof(push));

      for (unsigned i = 0; i < n; ++i) {
         const float sum = in_a[i] + res[i];
         if (inplace) {
            HK_EVAL_CHECK(hk_eval_f2u(out1[i]) == hk_eval_f2u(sum));
         } else {
            HK_EVAL_CHECK(hk_eval_f2u(out1[i]) == hk_eval_f2u(in_a[i]));
            HK_EVAL_CHECK(hk_eval_f2u(dst1[i]) == hk_eval_f2u(sum));
         }
         if (has_copy)
            HK_EVAL_CHECK(hk_eval_f2u(dst2[i]) == hk_eval_f2u(sum));
      }
      ralloc_free(s);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_concat_x();
   test_concat_y();
   test_resample_nearest();
   test_gemm_fadd_postpass();
   glsl_type_singleton_decref();
   printf("PASS: hk_tensor_ops\n");
   return 0;
}
