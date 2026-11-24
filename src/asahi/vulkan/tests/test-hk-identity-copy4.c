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
#include "hk_identity_copy4.h"
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

static nir_def *
build_short_or(nir_builder *b, nir_def *lhs, nir_def *rhs)
{
   nir_def *not_lhs = nir_inot(b, lhs);
   nir_if *nif = nir_push_if(b, not_lhs);
   nir_push_else(b, nif);
   nir_pop_if(b, nif);
   return nir_if_phi(b, rhs, lhs);
}

static nir_def *
build_short_and(nir_builder *b, nir_def *lhs, nir_def *rhs)
{
   nir_if *nif = nir_push_if(b, lhs);
   nir_push_else(b, nif);
   nir_pop_if(b, nif);
   return nir_if_phi(b, rhs, lhs);
}

static nir_shader *
build_linear_shader(enum hk_identity_copy4_kind kind,
                    const struct glsl_type *elem_type,
                    enum gl_access_qualifier load_access,
                    bool extra_store, bool wrong_cond,
                    uint16_t count_off, uint16_t src_off, uint16_t dst_off,
                    uint16_t push_range)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "linear_test");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = (kind == HK_IDENTITY_COPY4_LINEAR_FADD) ? 3 : 2;
   b.shader->num_uniforms = push_range;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *count = load_push_u32(&b, count_off, push_range);
   nir_def *cond = wrong_cond ? nir_ult(&b, gid, count) : nir_ilt(&b, gid, count);
   nir_if *nif = nir_push_if(&b, cond);
   {
      nir_def *s_idx = gid;
      nir_def *d_idx = gid;
      if (kind == HK_IDENTITY_COPY4_LINEAR_OFFSET) {
         nir_def *so = load_push_u32(&b, src_off, push_range);
         nir_def *doff = load_push_u32(&b, dst_off, push_range);
         s_idx = nir_iadd(&b, gid, so);
         d_idx = nir_iadd(&b, gid, doff);
      }
      nir_deref_instr *src0 =
         make_ssbo_elem_deref(&b, 0, 0, elem_type, true, s_idx);
      nir_def *v = nir_load_deref_with_access(&b, src0, load_access);
      if (kind == HK_IDENTITY_COPY4_LINEAR_FADD) {
         nir_deref_instr *src1 =
            make_ssbo_elem_deref(&b, 0, 1, elem_type, true, s_idx);
         nir_def *v1 = nir_load_deref_with_access(&b, src1, load_access);
         v = nir_fadd(&b, v, v1);
      }
      unsigned dst_binding = (kind == HK_IDENTITY_COPY4_LINEAR_FADD) ? 2 : 1;
      nir_deref_instr *dst =
         make_ssbo_elem_deref(&b, 0, dst_binding, elem_type, false, d_idx);
      nir_store_deref_with_access(&b, dst, v, 1, ACCESS_NON_READABLE);
      if (extra_store)
         nir_store_deref_with_access(&b, dst, v, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, nif);

   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "linear_test");
   return b.shader;
}

static nir_shader *
build_pad4d_shader(const uint16_t in_off[4], const uint16_t out_off[4],
                   const uint32_t pad_idx[4], uint16_t push_range,
                   bool run_defer_preprocess)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "pad4d_test");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 16;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 3;
   b.shader->num_uniforms = push_range;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   nir_def *in_s[4], *out_s[4];
   for (unsigned d = 0; d < 4; ++d) {
      in_s[d] = load_push_u32(&b, in_off[d], push_range);
      out_s[d] = load_push_u32(&b, out_off[d], push_range);
   }

   nir_def *x_oob = nir_ige(&b, gx, nir_imul(&b, out_s[2], out_s[3]));
   nir_def *y_oob = nir_ige(&b, gy, nir_imul(&b, out_s[0], out_s[1]));
   nir_def *oob = build_short_or(&b, x_oob, y_oob);
   nir_if *outer = nir_push_if(&b, oob);
   nir_push_else(&b, outer);
   {
      nir_def *c[4];
      c[0] = nir_idiv(&b, gy, out_s[1]);
      c[1] = nir_irem(&b, gy, out_s[1]);
      c[2] = nir_idiv(&b, gx, out_s[3]);
      c[3] = nir_irem(&b, gx, out_s[3]);

      nir_def *p[4];
      for (unsigned d = 0; d < 4; ++d) {
         nir_deref_instr *pd = make_ssbo_elem_deref(
            &b, 0, 1, glsl_int_type(), true, nir_imm_int(&b, pad_idx[d]));
         p[d] = nir_load_deref_with_access(
            &b, pd, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      }

      nir_def *pad_cond = nir_ilt(&b, c[0], p[0]);
      for (unsigned d = 1; d < 4; ++d)
         pad_cond = build_short_or(&b, pad_cond, nir_ilt(&b, c[d], p[d]));
      for (unsigned d = 0; d < 4; ++d)
         pad_cond = build_short_or(&b, pad_cond,
                                   nir_ige(&b, nir_isub(&b, c[d], p[d]), in_s[d]));

      nir_def *out_idx = nir_iadd(
         &b,
         nir_iadd(&b,
                  nir_iadd(&b,
                           nir_imul(&b, nir_imul(&b, nir_imul(&b, c[0], out_s[1]), out_s[2]), out_s[3]),
                           nir_imul(&b, nir_imul(&b, c[1], out_s[2]), out_s[3])),
                  nir_imul(&b, c[2], out_s[3])),
         c[3]);

      nir_if *pif = nir_push_if(&b, pad_cond);
      {
         nir_deref_instr *dst =
            make_ssbo_elem_deref(&b, 0, 2, glsl_float_type(), false, out_idx);
         nir_store_deref_with_access(&b, dst, nir_imm_float(&b, 0.0f), 1,
                                     ACCESS_NON_READABLE);
      }
      nir_push_else(&b, pif);
      {
         nir_def *ic[4];
         for (unsigned d = 0; d < 4; ++d)
            ic[d] = nir_isub(&b, c[d], p[d]);
         nir_def *in_idx = nir_iadd(
            &b,
            nir_iadd(&b,
                     nir_iadd(&b,
                              nir_imul(&b, nir_imul(&b, nir_imul(&b, ic[0], in_s[1]), in_s[2]), in_s[3]),
                              nir_imul(&b, nir_imul(&b, ic[1], in_s[2]), in_s[3])),
                     nir_imul(&b, ic[2], in_s[3])),
            ic[3]);
         nir_deref_instr *src =
            make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, in_idx);
         nir_def *val = nir_load_deref_with_access(
            &b, src, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
         nir_deref_instr *dst =
            make_ssbo_elem_deref(&b, 0, 2, glsl_float_type(), false, out_idx);
         nir_store_deref_with_access(&b, dst, val, 1, ACCESS_NON_READABLE);
      }
      nir_pop_if(&b, pif);
   }
   nir_pop_if(&b, outer);

   if (run_defer_preprocess)
      agx_preprocess_nir_native_defer_idiv(b.shader, AGX_CHIP_G17P);
   else {
      nir_opt_copy_prop(b.shader);
      nir_opt_dce(b.shader);
   }
   nir_validate_shader(b.shader, "pad4d_test");
   return b.shader;
}

struct im2col_layout {
   uint16_t bw, bh, iw, ih, kw, pw, ph, sw, sh, ch;
   uint16_t push_range;
};

static nir_shader *
build_im2col_shader(const struct im2col_layout *l, bool run_defer_preprocess)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "im2col_test");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 16;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->num_uniforms = l->push_range;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *tap = nir_channel(&b, gid, 0);
   nir_def *row = nir_channel(&b, gid, 1);

   nir_def *bw = load_push_u32(&b, l->bw, l->push_range);
   nir_def *bh = load_push_u32(&b, l->bh, l->push_range);
   nir_def *iw = load_push_u32(&b, l->iw, l->push_range);
   nir_def *ih = load_push_u32(&b, l->ih, l->push_range);
   nir_def *kw = load_push_u32(&b, l->kw, l->push_range);
   nir_def *pw = load_push_u32(&b, l->pw, l->push_range);
   nir_def *ph = load_push_u32(&b, l->ph, l->push_range);
   nir_def *sw = load_push_u32(&b, l->sw, l->push_range);
   nir_def *sh = load_push_u32(&b, l->sh, l->push_range);
   nir_def *ch = load_push_u32(&b, l->ch, l->push_range);

   nir_def *oob = build_short_or(&b, nir_ige(&b, tap, pw), nir_ige(&b, row, ph));
   nir_if *outer = nir_push_if(&b, oob);
   nir_push_else(&b, outer);
   {
      nir_def *is_unit = build_short_and(&b, nir_ieq_imm(&b, sh, 1),
                                         nir_ieq_imm(&b, sw, 1));
      nir_def *pad_x = nir_imul(&b, bw, ch);
      nir_def *one_pad = nir_iadd(&b, iw, pad_x);
      nir_def *both_pads = nir_iadd(&b, one_pad, pad_x);
      nir_if *iw_if = nir_push_if(&b, is_unit);
      nir_push_else(&b, iw_if);
      nir_pop_if(&b, iw_if);
      nir_def *iw_pad = nir_if_phi(&b, both_pads, one_pad);

      nir_def *diff = nir_isub(&b, nir_idiv(&b, iw_pad, ch), nir_idiv(&b, kw, ch));
      nir_def *cols = nir_iadd_imm(&b, nir_idiv(&b, diff, sw), 1);

      nir_def *x = nir_iadd(&b, nir_imul(&b, nir_imul(&b, nir_irem(&b, row, cols), sw), ch),
                            nir_irem(&b, tap, kw));
      nir_def *y = nir_iadd(&b, nir_imul(&b, nir_idiv(&b, row, cols), sh),
                            nir_idiv(&b, tap, kw));

      nir_def *zero_f = nir_imm_float(&b, 0.0f);
      nir_def *unit_val = NULL, *nonunit_val = NULL;
      nir_if *uif = nir_push_if(&b, is_unit);
      {
         nir_def *v = NULL;
         nir_if *c0 = nir_push_if(&b, nir_ilt(&b, x, pad_x));
         nir_push_else(&b, c0);
         {
            nir_if *c1 = nir_push_if(&b, nir_ilt(&b, y, bh));
            nir_push_else(&b, c1);
            {
               nir_if *c2 = nir_push_if(&b, nir_ige(&b, x, one_pad));
               nir_push_else(&b, c2);
               {
                  nir_if *c3 = nir_push_if(&b, nir_ige(&b, y, nir_iadd(&b, ih, bh)));
                  nir_push_else(&b, c3);
                  {
                     nir_def *s_idx = nir_iadd(&b, nir_imul(&b, nir_isub(&b, y, bh), iw),
                                               nir_isub(&b, x, pad_x));
                     nir_deref_instr *src =
                        make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, s_idx);
                     v = nir_load_deref_with_access(
                        &b, src, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
                  }
                  nir_pop_if(&b, c3);
                  v = nir_if_phi(&b, zero_f, v);
               }
               nir_pop_if(&b, c2);
               v = nir_if_phi(&b, zero_f, v);
            }
            nir_pop_if(&b, c1);
            v = nir_if_phi(&b, zero_f, v);
         }
         nir_pop_if(&b, c0);
         unit_val = nir_if_phi(&b, zero_f, v);
      }
      nir_push_else(&b, uif);
      {
         nir_def *oob_nu = build_short_or(&b, nir_ige(&b, x, iw), nir_ige(&b, y, ih));
         nir_def *v = NULL;
         nir_if *nu = nir_push_if(&b, oob_nu);
         nir_push_else(&b, nu);
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, y, iw), x);
            nir_deref_instr *src =
               make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, s_idx);
            v = nir_load_deref_with_access(
               &b, src, ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
         }
         nir_pop_if(&b, nu);
         nonunit_val = nir_if_phi(&b, zero_f, v);
      }
      nir_pop_if(&b, uif);

      nir_def *payload = nir_if_phi(&b, unit_val, nonunit_val);
      nir_def *dst_idx = nir_iadd(&b, nir_imul(&b, row, pw), tap);
      nir_deref_instr *dst =
         make_ssbo_elem_deref(&b, 0, 1, glsl_float_type(), false, dst_idx);
      nir_store_deref_with_access(&b, dst, payload, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, outer);

   if (run_defer_preprocess)
      agx_preprocess_nir_native_defer_idiv(b.shader, AGX_CHIP_G17P);
   else {
      nir_opt_copy_prop(b.shader);
      nir_opt_dce(b.shader);
   }
   nir_validate_shader(b.shader, "im2col_test");
   return b.shader;
}

static unsigned
count_vec4_stores(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_global &&
                intr->src[0].ssa->num_components == 4)
               n++;
         }
      }
   }
   return n;
}

static void
fill_pattern_u32(uint32_t *buf, unsigned n, uint32_t seed)
{
   /* Include special IEEE-754 bit patterns alongside pseudo-random floats. */
   static const uint32_t specials[] = {
      0x00000000u, /* +0.0 */
      0x80000000u, /* -0.0 */
      0x3f800000u, /* 1.0 */
      0xbf800000u, /* -1.0 */
      0x7f800000u, /* +Inf */
      0xff800000u, /* -Inf */
      0x7fc00001u, /* qNaN */
      0x00000001u, /* min subnormal */
   };
   for (unsigned i = 0; i < n; ++i) {
      if (i < ARRAY_SIZE(specials)) {
         buf[i] = specials[(i + seed) % ARRAY_SIZE(specials)];
      } else {
         float f = (float)((int)(i * 17 + seed * 31) % 257 - 128) * 0.125f;
         memcpy(&buf[i], &f, sizeof(f));
      }
   }
}

static void
test_linear_bit_exact(void)
{
   /* Test both workgroup-aligned (512 vec4 = 2048 elems) and non-aligned tail (540 elems). */
   const uint32_t counts[] = { 16, 540, 2048, 2100 };

   /* 1. Linear copy (bit-exact including -0.0, NaN, Inf, subnormals). */
   {
      nir_shader *orig = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_COPY, glsl_float_type(),
         ACCESS_NON_WRITEABLE, false, false, 0, 0, 0, 4);
      struct hk_identity_copy4 info = {0};
      nir_shader *alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(alt != NULL);
      nir_validate_shader(alt, "linear_copy alt");
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_LINEAR_COPY);
      HK_EVAL_CHECK(info.src_count == 1 && info.src_binding == 0 && info.dst_binding == 1);
      HK_EVAL_CHECK(info.count_offset == 0);
      HK_EVAL_CHECK(alt->info.workgroup_size[0] == 128 && alt->info.workgroup_size[1] == 1);
      HK_EVAL_CHECK(count_vec4_stores(alt) == 8);

      for (unsigned ci = 0; ci < ARRAY_SIZE(counts); ++ci) {
         const uint32_t count = counts[ci];
         uint32_t *src = calloc(count, sizeof(uint32_t));
         uint32_t *dst_orig = calloc(count, sizeof(uint32_t));
         uint32_t *dst_alt = calloc(count, sizeof(uint32_t));
         fill_pattern_u32(src, count, ci);
         memset(dst_orig, 0xa5, count * sizeof(uint32_t));
         memset(dst_alt, 0xa5, count * sizeof(uint32_t));

         uint32_t push[1] = { count };
         struct hk_eval_ssbo s_orig[2] = {
            {0, 0, src, count * 4},
            {0, 1, dst_orig, count * 4},
         };
         struct hk_eval_ssbo s_alt[2] = {
            {0, 0, src, count * 4},
            {0, 1, dst_alt, count * 4},
         };
         uint32_t g_orig[3] = { DIV_ROUND_UP(count, 64), 1, 1 };
         uint32_t g_alt[3] = { DIV_ROUND_UP(count / 4, 128 * 4), 1, 1 };
         hk_eval_dispatch(orig, g_orig, s_orig, 2, push, sizeof(push));
         hk_eval_dispatch(alt, g_alt, s_alt, 2, push, sizeof(push));
         HK_EVAL_CHECK(memcmp(dst_orig, dst_alt, count * sizeof(uint32_t)) == 0);
         free(src);
         free(dst_orig);
         free(dst_alt);
      }
      ralloc_free(alt);
      ralloc_free(orig);
   }

   /* 2. Linear offset copy with permuted push offsets (push_range = 48). */
   {
      nir_shader *orig = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_OFFSET, glsl_uint_type(),
         ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER, false, false, 36, 20, 8, 48);
      struct hk_identity_copy4 info = {0};
      nir_shader *alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(alt != NULL);
      nir_validate_shader(alt, "linear_offset alt");
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_LINEAR_OFFSET);
      HK_EVAL_CHECK(info.count_offset == 36 && info.src_offset_offset == 20 &&
             info.dst_offset_offset == 8);

      const uint32_t count = 540, so = 16, doff = 24, total = 640;
      uint32_t *src = calloc(total, 4);
      uint32_t *dst_orig = calloc(total, 4);
      uint32_t *dst_alt = calloc(total, 4);
      fill_pattern_u32(src, total, 7);
      memset(dst_orig, 0x5a, total * 4);
      memset(dst_alt, 0x5a, total * 4);

      uint32_t push[12] = {0};
      push[36 / 4] = count;
      push[20 / 4] = so;
      push[8 / 4] = doff;
      struct hk_eval_ssbo s_orig[2] = { {0, 0, src, total * 4}, {0, 1, dst_orig, total * 4} };
      struct hk_eval_ssbo s_alt[2] = { {0, 0, src, total * 4}, {0, 1, dst_alt, total * 4} };
      uint32_t g_orig[3] = { DIV_ROUND_UP(count, 64), 1, 1 };
      uint32_t g_alt[3] = { DIV_ROUND_UP(count / 4, 128 * 4), 1, 1 };
      hk_eval_dispatch(orig, g_orig, s_orig, 2, push, sizeof(push));
      hk_eval_dispatch(alt, g_alt, s_alt, 2, push, sizeof(push));
      HK_EVAL_CHECK(memcmp(dst_orig, dst_alt, total * 4) == 0);
      free(src);
      free(dst_orig);
      free(dst_alt);
      ralloc_free(alt);
      ralloc_free(orig);
   }

   /* 3. Linear FP32 fadd bit-exactness. */
   {
      nir_shader *orig = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_FADD, glsl_float_type(),
         ACCESS_NON_WRITEABLE, false, false, 12, 0, 0, 16);
      struct hk_identity_copy4 info = {0};
      nir_shader *alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(alt != NULL);
      nir_validate_shader(alt, "linear_fadd alt");
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_LINEAR_FADD);
      HK_EVAL_CHECK(info.count_offset == 12);

      const uint32_t count = 540;
      uint32_t *a = calloc(count, 4), *b = calloc(count, 4);
      uint32_t *dst_orig = calloc(count, 4), *dst_alt = calloc(count, 4);
      fill_pattern_u32(a, count, 1);
      fill_pattern_u32(b, count, 3);
      uint32_t push[4] = {0, 0, 0, count};
      struct hk_eval_ssbo s_orig[3] = { {0, 0, a, count * 4}, {0, 1, b, count * 4}, {0, 2, dst_orig, count * 4} };
      struct hk_eval_ssbo s_alt[3] = { {0, 0, a, count * 4}, {0, 1, b, count * 4}, {0, 2, dst_alt, count * 4} };
      uint32_t g_orig[3] = { DIV_ROUND_UP(count, 64), 1, 1 };
      uint32_t g_alt[3] = { DIV_ROUND_UP(count / 4, 128 * 4), 1, 1 };
      hk_eval_dispatch(orig, g_orig, s_orig, 3, push, sizeof(push));
      hk_eval_dispatch(alt, g_alt, s_alt, 3, push, sizeof(push));
      HK_EVAL_CHECK(memcmp(dst_orig, dst_alt, count * 4) == 0);
      free(a);
      free(b);
      free(dst_orig);
      free(dst_alt);
      ralloc_free(alt);
      ralloc_free(orig);
   }
}

static void
test_pad4d_bit_exact(void)
{
   /* Permuted push layout (48 bytes) AND preprocessed with agx_preprocess_nir_native_defer_idiv. */
   const uint16_t in_off[4] = { 40, 28, 12, 4 };
   const uint16_t out_off[4] = { 32, 20, 8, 44 };
   const uint32_t pad_idx[4] = { 0, 2, 4, 6 };

   nir_shader *orig = build_pad4d_shader(in_off, out_off, pad_idx, 48, true);
   struct hk_identity_copy4 info = {0};
   nir_shader *alt = hk_nir_copy_identity4(orig, &info);
   HK_EVAL_CHECK(alt != NULL);
   nir_validate_shader(alt, "pad4d alt");
   HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_PAD4);
   for (unsigned d = 0; d < 4; ++d) {
      HK_EVAL_CHECK(info.in_shape_off[d] == in_off[d]);
      HK_EVAL_CHECK(info.out_shape_off[d] == out_off[d]);
      HK_EVAL_CHECK(info.pad_idx[d] == pad_idx[d]);
   }

   struct {
      int32_t in_s[4];
      int32_t out_s[4];
      int32_t pad[4];
   } cases[] = {
      /* out_n == 1, p3 == 0 (fast vec4 path), border & interior */
      { {1, 5, 7, 8}, {1, 7, 9, 8}, {0, 1, 1, 0} },
      /* out_n == 1, p3 == 0, channel zero-pad (out_c > in_c) */
      { {1, 3, 5, 4}, {1, 5, 7, 8}, {0, 1, 1, 0} },
      /* out_n == 2, p3 == 0, non-zero batch pad */
      { {1, 3, 5, 8}, {2, 5, 7, 8}, {1, 1, 2, 0} },
      /* out_n == 2, p3 == 2 (unaligned channel pad slow path!), workgroup edge */
      { {2, 4, 5, 8}, {3, 6, 8, 12}, {1, 1, 1, 2} },
      /* out_n == 1, p3 == 4 (aligned non-zero channel pad) */
      { {1, 3, 4, 8}, {1, 4, 6, 16}, {0, 1, 1, 4} },
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      uint32_t in_n = (uint32_t)(cases[ci].in_s[0] * cases[ci].in_s[1] *
                                 cases[ci].in_s[2] * cases[ci].in_s[3]);
      uint32_t out_n = (uint32_t)(cases[ci].out_s[0] * cases[ci].out_s[1] *
                                  cases[ci].out_s[2] * cases[ci].out_s[3]);
      uint32_t *in_buf = calloc(in_n, 4);
      uint32_t *dst_orig = calloc(out_n, 4);
      uint32_t *dst_alt = calloc(out_n, 4);
      fill_pattern_u32(in_buf, in_n, 10 + ci);
      memset(dst_orig, 0x33, out_n * 4);
      memset(dst_alt, 0x33, out_n * 4);

      int32_t pad_buf[8] = {0};
      for (unsigned d = 0; d < 4; ++d)
         pad_buf[pad_idx[d]] = cases[ci].pad[d];

      int32_t push[12] = {0};
      for (unsigned d = 0; d < 4; ++d) {
         push[in_off[d] / 4] = cases[ci].in_s[d];
         push[out_off[d] / 4] = cases[ci].out_s[d];
      }

      struct hk_eval_ssbo s_orig[3] = {
         {0, 0, in_buf, in_n * 4},
         {0, 1, pad_buf, sizeof(pad_buf)},
         {0, 2, dst_orig, out_n * 4},
      };
      struct hk_eval_ssbo s_alt[3] = {
         {0, 0, in_buf, in_n * 4},
         {0, 1, pad_buf, sizeof(pad_buf)},
         {0, 2, dst_alt, out_n * 4},
      };

      uint32_t orig_x = (uint32_t)(cases[ci].out_s[2] * cases[ci].out_s[3]);
      uint32_t orig_y = (uint32_t)(cases[ci].out_s[0] * cases[ci].out_s[1]);
      uint32_t g_orig[3] = { DIV_ROUND_UP(orig_x, 16), DIV_ROUND_UP(orig_y, 16), 1 };
      uint32_t g_alt[3] = {
         DIV_ROUND_UP(orig_x / 4, alt->info.workgroup_size[0]),
         DIV_ROUND_UP(orig_y, alt->info.workgroup_size[1]),
         1,
      };

      hk_eval_dispatch(orig, g_orig, s_orig, 3, push, sizeof(push));
      hk_eval_dispatch(alt, g_alt, s_alt, 3, push, sizeof(push));
      HK_EVAL_CHECK(memcmp(dst_orig, dst_alt, out_n * 4) == 0);

      free(in_buf);
      free(dst_orig);
      free(dst_alt);
   }

   ralloc_free(alt);
   ralloc_free(orig);
}

static void
test_im2col_bit_exact(void)
{
   /* Test BOTH standard 44B layout AND permuted 56B layout with gaps! */
   const struct im2col_layout layouts[] = {
      { .bw = 0, .bh = 4, .iw = 8, .ih = 12, .kw = 16, .pw = 24, .ph = 28, .sw = 32, .sh = 36, .ch = 40, .push_range = 44 },
      { .bw = 52, .bh = 44, .iw = 36, .ih = 28, .kw = 24, .pw = 16, .ph = 12, .sw = 8, .sh = 4, .ch = 0, .push_range = 56 },
   };

   for (unsigned li = 0; li < ARRAY_SIZE(layouts); ++li) {
      const struct im2col_layout *l = &layouts[li];
      nir_shader *orig = build_im2col_shader(l, true);
      struct hk_identity_copy4 info = {0};
      nir_shader *id_alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(id_alt != NULL);
      nir_validate_shader(id_alt, "im2col identity alt");
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_IM2COL_1X1);
      HK_EVAL_CHECK(info.im2col_vec4_ok);
      HK_EVAL_CHECK(info.border_width_offset == l->bw);
      HK_EVAL_CHECK(info.border_height_offset == l->bh);
      HK_EVAL_CHECK(info.input_width_offset == l->iw);
      HK_EVAL_CHECK(info.input_height_offset == l->ih);
      HK_EVAL_CHECK(info.kernel_width_offset == l->kw);
      HK_EVAL_CHECK(info.patches_width_offset == l->pw);
      HK_EVAL_CHECK(info.patches_height_offset == l->ph);
      HK_EVAL_CHECK(info.stride_width_offset == l->sw);
      HK_EVAL_CHECK(info.stride_height_offset == l->sh);
      HK_EVAL_CHECK(info.channels_offset == l->ch);

      nir_shader *vec4_alt = hk_nir_im2col_vec4(orig);
      HK_EVAL_CHECK(vec4_alt != NULL);
      nir_validate_shader(vec4_alt, "im2col vec4 alt");

      /* Case A: 1x1 identity copy -> compare orig vs id_alt AND vs vec4_alt. */
      {
         const int32_t ch = 8, w = 13, h = 5;
         const int32_t iw = w * ch, ih = h, kw = ch, pw = ch, ph = w * h;
         uint32_t push[16] = {0};
         push[l->bw / 4] = 0;
         push[l->bh / 4] = 0;
         push[l->iw / 4] = iw;
         push[l->ih / 4] = ih;
         push[l->kw / 4] = kw;
         push[l->pw / 4] = pw;
         push[l->ph / 4] = ph;
         push[l->sw / 4] = 1;
         push[l->sh / 4] = 1;
         push[l->ch / 4] = ch;

         const uint32_t elems = (uint32_t)(pw * ph);
         uint32_t *src = calloc(elems, 4);
         uint32_t *dst_orig = calloc(elems, 4);
         uint32_t *dst_id = calloc(elems, 4);
         uint32_t *dst_v4 = calloc(elems, 4);
         fill_pattern_u32(src, elems, 20 + li);

         struct hk_eval_ssbo so[2] = { {0, 0, src, elems * 4}, {0, 1, dst_orig, elems * 4} };
         struct hk_eval_ssbo si[2] = { {0, 0, src, elems * 4}, {0, 1, dst_id, elems * 4} };
         struct hk_eval_ssbo sv[2] = { {0, 0, src, elems * 4}, {0, 1, dst_v4, elems * 4} };
         uint32_t go[3] = { DIV_ROUND_UP((uint32_t)pw, 16), DIV_ROUND_UP((uint32_t)ph, 16), 1 };
         uint32_t gi[3] = { DIV_ROUND_UP(elems / 4, 128 * 4), 1, 1 };
         uint32_t gv[3] = {
            DIV_ROUND_UP((uint32_t)pw / 4, vec4_alt->info.workgroup_size[0]),
            DIV_ROUND_UP((uint32_t)ph, vec4_alt->info.workgroup_size[1]),
            1,
         };
         hk_eval_dispatch(orig, go, so, 2, push, l->push_range);
         hk_eval_dispatch(id_alt, gi, si, 2, push, l->push_range);
         hk_eval_dispatch(vec4_alt, gv, sv, 2, push, l->push_range);
         HK_EVAL_CHECK(memcmp(dst_orig, dst_id, elems * 4) == 0);
         HK_EVAL_CHECK(memcmp(dst_orig, dst_v4, elems * 4) == 0);
         free(src);
         free(dst_orig);
         free(dst_id);
         free(dst_v4);
      }

      /* Case B: 3x3 unit-stride padded im2col (border & interior, workgroup edge). */
      /* Case C: 3x3 stride-2 non-unit im2col (border & interior, workgroup edge). */
      struct {
         int32_t bw, bh, w, h, kdim, sw, sh, ch;
      } icases[] = {
         { 1, 1, 7, 5, 3, 1, 1, 8 },
         { 1, 1, 9, 7, 3, 2, 2, 8 },
         { 2, 1, 6, 5, 5, 1, 1, 4 },
      };
      for (unsigned ci = 0; ci < ARRAY_SIZE(icases); ++ci) {
         const int32_t bw = icases[ci].bw, bh = icases[ci].bh;
         const int32_t ch = icases[ci].ch;
         const int32_t iw = icases[ci].w * ch, ih = icases[ci].h;
         const int32_t kw = icases[ci].kdim * ch;
         const int32_t pw = kw * icases[ci].kdim;
         const int32_t sw = icases[ci].sw, sh = icases[ci].sh;
         const bool unit = (sw == 1 && sh == 1);
         const int32_t cols = ((iw + (unit ? 2 : 1) * bw * ch) / ch - kw / ch) / sw + 1;
         const int32_t rows = ((ih + (unit ? 2 : 1) * bh) - icases[ci].kdim) / sh + 1;
         const int32_t ph = cols * rows;

         uint32_t push[16] = {0};
         push[l->bw / 4] = bw;
         push[l->bh / 4] = bh;
         push[l->iw / 4] = iw;
         push[l->ih / 4] = ih;
         push[l->kw / 4] = kw;
         push[l->pw / 4] = pw;
         push[l->ph / 4] = ph;
         push[l->sw / 4] = sw;
         push[l->sh / 4] = sh;
         push[l->ch / 4] = ch;

         const uint32_t src_elems = (uint32_t)(iw * ih);
         const uint32_t dst_elems = (uint32_t)(pw * ph);
         uint32_t *src = calloc(src_elems, 4);
         uint32_t *dst_orig = calloc(dst_elems, 4);
         uint32_t *dst_v4 = calloc(dst_elems, 4);
         fill_pattern_u32(src, src_elems, 40 + li * 10 + ci);
         memset(dst_orig, 0x77, dst_elems * 4);
         memset(dst_v4, 0x77, dst_elems * 4);

         struct hk_eval_ssbo so[2] = { {0, 0, src, src_elems * 4}, {0, 1, dst_orig, dst_elems * 4} };
         struct hk_eval_ssbo sv[2] = { {0, 0, src, src_elems * 4}, {0, 1, dst_v4, dst_elems * 4} };
         uint32_t go[3] = { DIV_ROUND_UP((uint32_t)pw, 16), DIV_ROUND_UP((uint32_t)ph, 16), 1 };
         uint32_t gv[3] = {
            DIV_ROUND_UP((uint32_t)pw / 4, vec4_alt->info.workgroup_size[0]),
            DIV_ROUND_UP((uint32_t)ph, vec4_alt->info.workgroup_size[1]),
            1,
         };
         hk_eval_dispatch(orig, go, so, 2, push, l->push_range);
         hk_eval_dispatch(vec4_alt, gv, sv, 2, push, l->push_range);
         HK_EVAL_CHECK(memcmp(dst_orig, dst_v4, dst_elems * 4) == 0);
         free(src);
         free(dst_orig);
         free(dst_v4);
      }

      ralloc_free(vec4_alt);
      ralloc_free(id_alt);
      ralloc_free(orig);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   test_linear_bit_exact();
   test_pad4d_bit_exact();
   test_im2col_bit_exact();

   /* Negative: writable source load, extra store, or unsigned guard must reject. */
   {
      struct hk_identity_copy4 info = {0};
      nir_shader *s1 = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_COPY, glsl_float_type(),
         (enum gl_access_qualifier)0, false, false, 0, 0, 0, 12);
      HK_EVAL_CHECK(hk_nir_copy_identity4(s1, &info) == NULL);
      ralloc_free(s1);

      nir_shader *s2 = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_COPY, glsl_float_type(),
         ACCESS_NON_WRITEABLE, true, false, 0, 0, 0, 12);
      HK_EVAL_CHECK(hk_nir_copy_identity4(s2, &info) == NULL);
      ralloc_free(s2);

      nir_shader *s3 = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_COPY, glsl_float_type(),
         ACCESS_NON_WRITEABLE, false, true, 0, 0, 0, 12);
      HK_EVAL_CHECK(hk_nir_copy_identity4(s3, &info) == NULL);
      ralloc_free(s3);
   }

   glsl_type_singleton_decref();
   printf("test-hk-identity-copy4: PASS\n");
   return 0;
}
