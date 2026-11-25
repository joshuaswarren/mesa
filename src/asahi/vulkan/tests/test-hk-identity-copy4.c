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
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_pack_loop_bound.h"
#include "hk_nir_eval_test.h"
#include "hk_nir_wgid.h"

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
      HK_EVAL_CHECK(info.fadd_meta_valid);
      HK_EVAL_CHECK(info.fadd_float_controls_execution_mode ==
                    orig->info.float_controls_execution_mode);

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
      /* out_n == 1, p3 == 0 (fast spatial contiguous path), border & interior */
      { {1, 5, 7, 8}, {1, 7, 9, 8}, {0, 1, 1, 0} },
      /* Wide row (> 256 vec4s) exercising full 4x64 interior chunk + border tail */
      { {1, 4, 38, 32}, {1, 6, 40, 32}, {0, 1, 1, 0} },
      /* Exact 256-vec4 row with zero pad (full interior fast path) */
      { {1, 3, 32, 32}, {1, 3, 32, 32}, {0, 0, 0, 0} },
      /* Negative spatial crop (falls back to general udivmod path) */
      { {1, 6, 8, 8}, {1, 4, 5, 8}, {0, -1, -2, 0} },
      /* Large p[2] > out_w (all-zero output) */
      { {1, 3, 4, 8}, {1, 3, 4, 8}, {0, 0, 10, 0} },
      /* out_n == 1, p3 == 0, channel zero-pad (out_c > in_c) */
      { {1, 3, 5, 4}, {1, 5, 7, 8}, {0, 1, 1, 0} },
      /* out_n == 2, p3 == 0, non-zero batch pad (fast spatial path, out_n > 1) */
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

      struct hk_pack_loop_bound im2_lb = {0};
      nir_shader *vec4_alt = hk_nir_im2col_vec4_with_bound(orig, &im2_lb);
      HK_EVAL_CHECK(vec4_alt != NULL && im2_lb.valid);
      {
         uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
         params[HK_PACK_LOOP_PARAM_IM2COL_KH] = 3;
         uint64_t bval = 0;
         HK_EVAL_CHECK(hk_pack_eval_loop_bound(&im2_lb, params, &bval) && bval == 17);
      }
      HK_EVAL_CHECK(vec4_alt != NULL);
      nir_validate_shader(vec4_alt, "im2col vec4 alt");

      /* Case A: 1x1 identity copy -> compare orig vs id_alt AND vs vec4_alt. */
      {
         const int32_t ch = 8, w = 13, h = 5;
         const int32_t iw = w * ch, ih = h, kw = ch, pw = ch, ph = w * h;
         uint32_t push[20] = {0};
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
         const uint32_t im2_off = hk_im2col_push_offset(l->push_range);
         const struct hk_im2col_push ip = {
            .cols = (uint32_t)(iw / ch),
            .rows_y = (uint32_t)ih,
            .kh = 1,
         };
         memcpy((uint8_t *)push + im2_off, &ip, sizeof(ip));

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
            DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)kw, 4u), vec4_alt->info.workgroup_size[0]),
            DIV_ROUND_UP((uint32_t)(iw / ch), vec4_alt->info.workgroup_size[1]),
            DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)ih, 4u), vec4_alt->info.workgroup_size[2]),
         };
         hk_eval_dispatch(orig, go, so, 2, push, l->push_range);
         hk_eval_dispatch(id_alt, gi, si, 2, push, l->push_range);
         hk_eval_dispatch(vec4_alt, gv, sv, 2, push, im2_off + sizeof(ip));
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
         { 1, 1, 8, 10, 3, 1, 1, 8 },
         { 1, 1, 9, 13, 3, 2, 2, 8 },
         { 1, 1, 11, 15, 3, 2, 2, 8 },
         { 0, 0, 5, 8, 3, 1, 1, 8 },
         { 2, 1, 6, 5, 5, 1, 1, 4 },
         /* Scalar 3-channel RGB (kw=9, rem=1) stride-2, stride-1, 4y interior & tails. */
         { 1, 1, 10, 12, 3, 2, 2, 3 },
         { 1, 1, 11, 15, 3, 2, 2, 3 },
         { 1, 1, 7, 9, 3, 1, 1, 3 },
         { 1, 1, 8, 10, 3, 1, 1, 3 },
         { 0, 0, 5, 7, 3, 1, 1, 3 },
         { 1, 1, 11, 9, 3, 2, 3, 3 },
         { 1, 2, 9, 12, 3, 2, 2, 5 },
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

         uint32_t push[20] = {0};
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
         const uint32_t im2_off = hk_im2col_push_offset(l->push_range);
         const struct hk_im2col_push ip = {
            .cols = (uint32_t)cols,
            .rows_y = (uint32_t)rows,
            .kh = (uint32_t)icases[ci].kdim,
         };
         memcpy((uint8_t *)push + im2_off, &ip, sizeof(ip));

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
            DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)kw, 4u), vec4_alt->info.workgroup_size[0]),
            DIV_ROUND_UP((uint32_t)cols, vec4_alt->info.workgroup_size[1]),
            DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)rows, 4u), vec4_alt->info.workgroup_size[2]),
         };
         hk_eval_dispatch(orig, go, so, 2, push, l->push_range);
         hk_eval_dispatch(vec4_alt, gv, sv, 2, push, im2_off + sizeof(ip));
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

static nir_shader *
build_borderless_im2col_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options, "borderless_im2col");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 16;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->num_uniforms = 36;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);

   nir_def *iw = load_push_u32(&b, 0, 36);
   nir_def *kw = load_push_u32(&b, 8, 36);
   nir_def *pw = load_push_u32(&b, 16, 36);
   nir_def *ph = load_push_u32(&b, 20, 36);
   nir_def *sw = load_push_u32(&b, 24, 36);
   nir_def *sh = load_push_u32(&b, 28, 36);
   nir_def *ch = load_push_u32(&b, 32, 36);

   nir_def *oob = build_short_or(&b, nir_ige(&b, x, pw), nir_ige(&b, y, ph));
   nir_if *guard = nir_push_if(&b, oob);
   nir_push_else(&b, guard);
   {
      nir_def *cols = nir_iadd_imm(&b, nir_idiv(&b, nir_isub(&b, nir_idiv(&b, iw, ch), nir_idiv(&b, kw, ch)), sw), 1);
      nir_def *bx = nir_imul(&b, nir_imod(&b, y, cols), sw);
      nir_def *by = nir_imul(&b, nir_idiv(&b, y, cols), sh);
      nir_def *ix = nir_iadd(&b, nir_imul(&b, bx, ch), nir_imod(&b, x, kw));
      nir_def *iy = nir_iadd(&b, by, nir_idiv(&b, x, kw));
      nir_def *s_idx = nir_iadd(&b, nir_imul(&b, iy, iw), ix);
      nir_def *d_idx = nir_iadd(&b, nir_imul(&b, y, pw), x);
      nir_deref_instr *sd = make_ssbo_elem_deref(&b, 0, 0, glsl_float_type(), true, s_idx);
      nir_deref_instr *dd = make_ssbo_elem_deref(&b, 0, 1, glsl_float_type(), false, d_idx);
      nir_def *val = nir_load_deref_with_access(&b, sd, ACCESS_NON_WRITEABLE);
      nir_store_deref_with_access(&b, dd, val, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, guard);

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
test_borderless_im2col_bit_exact(void)
{
   nir_shader *orig = build_borderless_im2col_shader();
   struct hk_identity_copy4 info = {0};
   nir_shader *id_alt = hk_nir_copy_identity4(orig, &info);
   HK_EVAL_CHECK(id_alt != NULL);
   nir_validate_shader(id_alt, "borderless im2col identity alt");
   HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_IM2COL_1X1);
   HK_EVAL_CHECK(info.im2col_vec4_ok);
   HK_EVAL_CHECK(info.border_width_offset == UINT16_MAX);
   HK_EVAL_CHECK(info.border_height_offset == UINT16_MAX);
   HK_EVAL_CHECK(info.input_height_offset == UINT16_MAX);
   HK_EVAL_CHECK(info.input_width_offset == 0);
   HK_EVAL_CHECK(info.kernel_width_offset == 8);
   HK_EVAL_CHECK(info.patches_width_offset == 16);
   HK_EVAL_CHECK(info.patches_height_offset == 20);
   HK_EVAL_CHECK(info.stride_width_offset == 24);
   HK_EVAL_CHECK(info.stride_height_offset == 28);
   HK_EVAL_CHECK(info.channels_offset == 32);

   nir_shader *vec4_alt = hk_nir_im2col_vec4(orig);
   HK_EVAL_CHECK(vec4_alt != NULL);
   nir_validate_shader(vec4_alt, "borderless im2col vec4 alt");

   /* 1x1 identity_copy4 */
   {
      const int32_t w = 7, h = 5, ch = 8;
      const int32_t iw = w * ch, kw = ch, pw = ch, ph = w * h;
      uint32_t push[9] = { (uint32_t)iw, (uint32_t)h, (uint32_t)kw, 1, (uint32_t)pw, (uint32_t)ph, 1, 1, (uint32_t)ch };
      const uint32_t elems = (uint32_t)(iw * h);
      uint32_t *src = calloc(elems, 4), *d0 = calloc(elems, 4), *d1 = calloc(elems, 4);
      fill_pattern_u32(src, elems, 101);
      struct hk_eval_ssbo s0[2] = { {0, 0, src, elems * 4}, {0, 1, d0, elems * 4} };
      struct hk_eval_ssbo s1[2] = { {0, 0, src, elems * 4}, {0, 1, d1, elems * 4} };
      uint32_t g0[3] = { DIV_ROUND_UP((uint32_t)pw, 16), DIV_ROUND_UP((uint32_t)ph, 16), 1 };
      uint32_t g1[3] = { DIV_ROUND_UP(elems / 4, id_alt->info.workgroup_size[0] * 4), 1, 1 };
      hk_eval_dispatch(orig, g0, s0, 2, push, 36);
      hk_eval_dispatch(id_alt, g1, s1, 2, push, 36);
      HK_EVAL_CHECK(memcmp(d0, d1, elems * 4) == 0);
      free(src); free(d0); free(d1);
   }

   /* 3x3 vec4 and scalar (ch=8 s=1/2, ch=3 s=1/2, single-row and odd/even rows) */
   struct { int32_t w, h, kdim, sw, sh, ch; } cases[] = {
      { 9, 7, 3, 1, 1, 8 },
      { 9, 8, 3, 1, 1, 8 },
      { 9, 7, 3, 2, 2, 8 },
      { 9, 9, 3, 2, 2, 8 },
      { 7, 3, 3, 1, 1, 8 },
      { 9, 7, 3, 1, 1, 3 },
      { 9, 7, 3, 2, 2, 3 },
      { 9, 9, 3, 2, 2, 3 },
      { 7, 3, 3, 1, 1, 3 },
   };
   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const int32_t ch = cases[ci].ch, iw = cases[ci].w * ch, ih = cases[ci].h;
      const int32_t kw = cases[ci].kdim * ch, pw = kw * cases[ci].kdim;
      const int32_t sw = cases[ci].sw, sh = cases[ci].sh;
      const int32_t cols = (iw / ch - kw / ch) / sw + 1;
      const int32_t rows = (ih - cases[ci].kdim) / sh + 1;
      const int32_t ph = cols * rows;
      uint32_t push[12] = { (uint32_t)iw, (uint32_t)ih, (uint32_t)kw, (uint32_t)cases[ci].kdim,
                           (uint32_t)pw, (uint32_t)ph, (uint32_t)sw, (uint32_t)sh, (uint32_t)ch,
                           (uint32_t)cols, (uint32_t)rows, (uint32_t)cases[ci].kdim };
      const uint32_t se = (uint32_t)(iw * ih), de = (uint32_t)(pw * ph);
      uint32_t *src = calloc(se, 4), *d0 = calloc(de, 4), *d1 = calloc(de, 4);
      fill_pattern_u32(src, se, 200 + ci);
      memset(d0, 0x55, de * 4); memset(d1, 0x55, de * 4);
      struct hk_eval_ssbo s0[2] = { {0, 0, src, se * 4}, {0, 1, d0, de * 4} };
      struct hk_eval_ssbo s1[2] = { {0, 0, src, se * 4}, {0, 1, d1, de * 4} };
      uint32_t g0[3] = { DIV_ROUND_UP((uint32_t)pw, 16), DIV_ROUND_UP((uint32_t)ph, 16), 1 };
      uint32_t g1[3] = {
         DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)kw, 4u), vec4_alt->info.workgroup_size[0]),
         DIV_ROUND_UP((uint32_t)cols, vec4_alt->info.workgroup_size[1]),
         DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)rows, 4u), vec4_alt->info.workgroup_size[2]),
      };
      hk_eval_dispatch(orig, g0, s0, 2, push, 36);
      hk_eval_dispatch(vec4_alt, g1, s1, 2, push, 48);
      HK_EVAL_CHECK(memcmp(d0, d1, de * 4) == 0);
      free(src); free(d0); free(d1);
   }

   ralloc_free(vec4_alt);
   ralloc_free(id_alt);
   ralloc_free(orig);

   /* Negative: borderless im2col with writable source load must be rejected. */
   {
      nir_shader *bad = build_borderless_im2col_shader();
      nir_foreach_function_impl(impl, bad) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic) {
                  nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
                  if (op->intrinsic == nir_intrinsic_load_deref)
                     nir_intrinsic_set_access(op, (enum gl_access_qualifier)0);
               }
            }
         }
      }
      struct hk_identity_copy4 bad_info = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(bad, &bad_info) == NULL);
      HK_EVAL_CHECK(hk_nir_im2col_vec4(bad) == NULL);
      ralloc_free(bad);
   }
}

/* Structurally distinct permuted-layout borderless HWC im2col formulation:
 * non-zero descriptor set/bindings (s1b2 -> s1b3), 8x8 workgroup, permuted
 * 48-byte push-constant layout with padding gaps, and commuted index math. */
static nir_shader *
build_permuted_layout_im2col_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options, "permuted_layout_im2col");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->num_uniforms = 48;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *tap = nir_channel(&b, gid, 0);
   nir_def *row = nir_channel(&b, gid, 1);

   /* Permuted push layout: ch=0, sw=4, sh=8, (gap 12), kw=16, (gap 20),
    * iw=24, (gap 28), pw=32, ph=36, (tail 40..47). */
   nir_def *ch = load_push_u32(&b, 0, 48);
   nir_def *sw = load_push_u32(&b, 4, 48);
   nir_def *sh = load_push_u32(&b, 8, 48);
   nir_def *kw = load_push_u32(&b, 16, 48);
   nir_def *iw = load_push_u32(&b, 24, 48);
   nir_def *pw = load_push_u32(&b, 32, 48);
   nir_def *ph = load_push_u32(&b, 36, 48);

   nir_def *oob = build_short_or(&b, nir_ige(&b, row, ph), nir_ige(&b, tap, pw));
   nir_if *guard = nir_push_if(&b, oob);
   nir_push_else(&b, guard);
   {
      nir_def *in_cols = nir_idiv(&b, iw, ch);
      nir_def *k_cols = nir_idiv(&b, kw, ch);
      nir_def *cols = nir_iadd_imm(&b, nir_idiv(&b, nir_isub(&b, in_cols, k_cols), sw), 1);
      nir_def *ox = nir_imod(&b, row, cols);
      nir_def *oy = nir_idiv(&b, row, cols);
      nir_def *kx = nir_imod(&b, tap, kw);
      nir_def *ky = nir_idiv(&b, tap, kw);
      /* Commuted operand order relative to build_borderless_im2col_shader(). */
      nir_def *ix = nir_iadd(&b, kx, nir_imul(&b, ch, nir_imul(&b, sw, ox)));
      nir_def *iy = nir_iadd(&b, ky, nir_imul(&b, sh, oy));
      nir_def *s_idx = nir_iadd(&b, ix, nir_imul(&b, iw, iy));
      nir_def *d_idx = nir_iadd(&b, tap, nir_imul(&b, pw, row));
      nir_deref_instr *sd = make_ssbo_elem_deref(&b, 1, 2, glsl_float_type(), true, s_idx);
      nir_deref_instr *dd = make_ssbo_elem_deref(&b, 1, 3, glsl_float_type(), false, d_idx);
      nir_def *val = nir_load_deref_with_access(&b, sd, ACCESS_NON_WRITEABLE);
      nir_store_deref_with_access(&b, dd, val, 1, ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, guard);

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
test_permuted_layout_im2col_bit_exact(void)
{
   nir_shader *orig = build_permuted_layout_im2col_shader();
   struct hk_identity_copy4 info = {0};
   nir_shader *id_alt = hk_nir_copy_identity4(orig, &info);
   HK_EVAL_CHECK(id_alt != NULL);
   nir_validate_shader(id_alt, "permuted layout im2col identity alt");
   HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_IM2COL_1X1);
   HK_EVAL_CHECK(info.im2col_vec4_ok);
   HK_EVAL_CHECK(info.src_set == 1 && info.src_binding == 2);
   HK_EVAL_CHECK(info.dst_set == 1 && info.dst_binding == 3);
   HK_EVAL_CHECK(info.border_width_offset == UINT16_MAX);
   HK_EVAL_CHECK(info.border_height_offset == UINT16_MAX);
   HK_EVAL_CHECK(info.input_height_offset == UINT16_MAX);
   HK_EVAL_CHECK(info.channels_offset == 0);
   HK_EVAL_CHECK(info.stride_width_offset == 4);
   HK_EVAL_CHECK(info.stride_height_offset == 8);
   HK_EVAL_CHECK(info.kernel_width_offset == 16);
   HK_EVAL_CHECK(info.input_width_offset == 24);
   HK_EVAL_CHECK(info.patches_width_offset == 32);
   HK_EVAL_CHECK(info.patches_height_offset == 36);

   nir_shader *vec4_alt = hk_nir_im2col_vec4(orig);
   HK_EVAL_CHECK(vec4_alt != NULL);
   nir_validate_shader(vec4_alt, "permuted layout im2col vec4 alt");

   struct { int32_t w, h, kdim, sw, sh, ch; } cases[] = {
      { 11, 7, 3, 1, 1, 8 },
      { 11, 9, 3, 2, 2, 8 },
      { 10, 9, 3, 2, 2, 3 },
      { 9, 7, 3, 2, 3, 5 },
   };
   for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ++ci) {
      const int32_t ch = cases[ci].ch, iw = cases[ci].w * ch, ih = cases[ci].h;
      const int32_t kw = cases[ci].kdim * ch, pw = kw * cases[ci].kdim;
      const int32_t sw = cases[ci].sw, sh = cases[ci].sh;
      const int32_t cols = (iw / ch - kw / ch) / sw + 1;
      const int32_t rows = (ih - cases[ci].kdim) / sh + 1;
      const int32_t ph = cols * rows;
      uint32_t push[16] = {0};
      push[0] = (uint32_t)ch;
      push[1] = (uint32_t)sw;
      push[2] = (uint32_t)sh;
      push[4] = (uint32_t)kw;
      push[6] = (uint32_t)iw;
      push[8] = (uint32_t)pw;
      push[9] = (uint32_t)ph;
      const uint32_t im2_off = hk_im2col_push_offset(info.push_bytes);
      const struct hk_im2col_push ip = {
         .cols = (uint32_t)cols,
         .rows_y = (uint32_t)rows,
         .kh = (uint32_t)cases[ci].kdim,
      };
      memcpy((uint8_t *)push + im2_off, &ip, sizeof(ip));
      const uint32_t se = (uint32_t)(iw * ih), de = (uint32_t)(pw * ph);
      uint32_t *src = calloc(se, 4), *d0 = calloc(de, 4), *d1 = calloc(de, 4);
      fill_pattern_u32(src, se, 300 + ci);
      memset(d0, 0x33, de * 4); memset(d1, 0x33, de * 4);
      struct hk_eval_ssbo s0[2] = { {1, 2, src, se * 4}, {1, 3, d0, de * 4} };
      struct hk_eval_ssbo s1[2] = { {1, 2, src, se * 4}, {1, 3, d1, de * 4} };
      uint32_t g0[3] = { DIV_ROUND_UP((uint32_t)pw, 8), DIV_ROUND_UP((uint32_t)ph, 8), 1 };
      uint32_t g1[3] = {
         DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)kw, 4u), vec4_alt->info.workgroup_size[0]),
         DIV_ROUND_UP((uint32_t)cols, vec4_alt->info.workgroup_size[1]),
         DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)rows, 4u), vec4_alt->info.workgroup_size[2]),
      };
      hk_eval_dispatch(orig, g0, s0, 2, push, 48);
      hk_eval_dispatch(vec4_alt, g1, s1, 2, push, im2_off + sizeof(ip));
      HK_EVAL_CHECK(memcmp(d0, d1, de * 4) == 0);
      free(src); free(d0); free(d1);
   }

   ralloc_free(vec4_alt);
   ralloc_free(id_alt);
   ralloc_free(orig);
}

static void
test_fadd_dual_bit_exact(void)
{
   struct hk_identity_copy4 info = {0};
   nir_shader *copy_orig = build_linear_shader(
      HK_IDENTITY_COPY4_LINEAR_COPY, glsl_float_type(),
      ACCESS_NON_WRITEABLE, false, false, 0, 0, 0, 4);
   HK_EVAL_CHECK(hk_nir_copy_identity4_fadd_dual(copy_orig, &info) == NULL);
   ralloc_free(copy_orig);

   nir_shader *orig = build_linear_shader(
      HK_IDENTITY_COPY4_LINEAR_FADD, glsl_float_type(),
      ACCESS_NON_WRITEABLE, false, false, 0, 0, 0, 4);
   nir_shader *dual = hk_nir_copy_identity4_fadd_dual(orig, &info);
   HK_EVAL_CHECK(dual != NULL);
   HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_LINEAR_FADD);
   HK_EVAL_CHECK(info.src_count == 2);

   const uint32_t dual_off = hk_dual_write_push_offset(info.push_bytes);
   const uint32_t counts[] = {16, 2048, 2064, 4112};
   for (unsigned ci = 0; ci < ARRAY_SIZE(counts); ++ci) {
      const uint32_t n = counts[ci];
      float *a = malloc(n * sizeof(float));
      float *b = malloc(n * sizeof(float));
      float *ref = calloc(n, sizeof(float));
      float *dst1 = calloc(n, sizeof(float));
      float *dst2 = calloc(n, sizeof(float));
      for (uint32_t i = 0; i < n; ++i) {
         a[i] = (float)((int32_t)(i * 17u % 97u) - 48) * 0.25f;
         b[i] = (float)((int32_t)(i * 31u % 89u) - 44) * 0.125f;
      }

      uint8_t push[32] = {0};
      memcpy(push, &n, sizeof(n));
      uint64_t dst2_va = hk_eval_make_vaddr(3, 0);
      struct hk_dual_copy_push dp = {
         .dst2_va_lo = (uint32_t)dst2_va,
         .dst2_va_hi = (uint32_t)(dst2_va >> 32),
      };
      memcpy(push + dual_off, &dp, sizeof(dp));

      struct hk_eval_ssbo s0[3] = {
         {0, 0, a, n * 4}, {0, 1, b, n * 4}, {0, 2, ref, n * 4},
      };
      struct hk_eval_ssbo s1[4] = {
         {0, 0, a, n * 4}, {0, 1, b, n * 4}, {0, 2, dst1, n * 4},
         {99, 0, dst2, n * 4},
      };
      uint32_t g0[3] = {DIV_ROUND_UP(n, 64), 1, 1};
      uint32_t g1[3] = {
         DIV_ROUND_UP(n / 4, dual->info.workgroup_size[0] * 4), 1, 1,
      };
      hk_eval_dispatch(orig, g0, s0, 3, push, 4);
      hk_eval_dispatch(dual, g1, s1, 4, push, dual_off + sizeof(dp));
      HK_EVAL_CHECK(memcmp(ref, dst1, n * 4) == 0);
      HK_EVAL_CHECK(memcmp(ref, dst2, n * 4) == 0);

      /* Also test exact in-place src0 == dst1 with disjoint dst2. */
      float *inplace = malloc(n * sizeof(float));
      memcpy(inplace, a, n * 4);
      memset(dst2, 0, n * 4);
      struct hk_eval_ssbo s2[4] = {
         {0, 0, inplace, n * 4}, {0, 1, b, n * 4}, {0, 2, inplace, n * 4},
         {99, 0, dst2, n * 4},
      };
      hk_eval_dispatch(dual, g1, s2, 4, push, dual_off + sizeof(dp));
      HK_EVAL_CHECK(memcmp(ref, inplace, n * 4) == 0);
      HK_EVAL_CHECK(memcmp(ref, dst2, n * 4) == 0);

      /* Also verify HK_DUAL_WRITE_FLAG_SKIP_C leaves dst1 (and in-place src0)
       * untouched while dst2 remains bit-exact. */
      dp.flags = HK_DUAL_WRITE_FLAG_SKIP_C;
      memcpy(push + dual_off, &dp, sizeof(dp));
      for (uint32_t i = 0; i < n; ++i)
         dst1[i] = -88.5f;
      memset(dst2, 0, n * 4);
      hk_eval_dispatch(dual, g1, s1, 4, push, dual_off + sizeof(dp));
      for (uint32_t i = 0; i < n; ++i)
         HK_EVAL_CHECK(dst1[i] == -88.5f);
      HK_EVAL_CHECK(memcmp(ref, dst2, n * 4) == 0);

      memcpy(inplace, a, n * 4);
      memset(dst2, 0, n * 4);
      hk_eval_dispatch(dual, g1, s2, 4, push, dual_off + sizeof(dp));
      HK_EVAL_CHECK(memcmp(a, inplace, n * 4) == 0);
      HK_EVAL_CHECK(memcmp(ref, dst2, n * 4) == 0);

      free(inplace);
      free(a); free(b); free(ref); free(dst1); free(dst2);
   }

   ralloc_free(dual);
   ralloc_free(orig);
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
test_unit_axis_wgid_matchers(void)
{
   /* Shaders that read SPIR-V WorkgroupId (gl_WorkGroupID) directly on a unit
    * workgroup axis (local_size[axis] == 1) lower to load_workgroup_id[axis]
    * (plus load_base_workgroup_id[axis] when base workgroup offsets are
    * lowered). Verify bit-exact orig-vs-alt outputs for im2col, linear_1d, and
    * pad4d, plus negative guards and zero-base dispatch gating. */

   /* 1. im2col (both 1x1 identity copy and 3x3 padded im2col), wg=32x1x1. */
   const struct im2col_layout im_layout = {
      .bw = 0, .bh = 4, .iw = 8, .ih = 12, .kw = 16,
      .pw = 20, .ph = 24, .sw = 28, .sh = 32, .ch = 40,
      .push_range = 44,
   };
   for (unsigned add_base = 0; add_base < 2; ++add_base) {
      nir_shader *orig = build_im2col_shader(&im_layout, false);
      orig->info.workgroup_size[0] = 32;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 1, 1, add_base != 0, 1, 1);

      struct hk_identity_copy4 info = {0};
      nir_shader *id_alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(id_alt != NULL);
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_IM2COL_1X1 &&
                    info.im2col_vec4_ok && info.local[0] == 32 &&
                    info.local[1] == 1);
      nir_shader *vec4_alt = hk_nir_im2col_vec4(orig);
      HK_EVAL_CHECK(vec4_alt != NULL);

      /* 1a. 1x1 identity copy bit-exactness (orig vs id_alt & vec4_alt). */
      {
         const uint32_t pw = 48, ph = 12, elems = pw * ph;
         uint32_t push[20] = {0};
         push[im_layout.iw / 4] = pw;
         push[im_layout.ih / 4] = ph;
         push[im_layout.kw / 4] = pw;
         push[im_layout.pw / 4] = pw;
         push[im_layout.ph / 4] = ph;
         push[im_layout.sw / 4] = 1;
         push[im_layout.sh / 4] = 1;
         push[im_layout.ch / 4] = pw;
         const uint32_t im2_off = hk_im2col_push_offset(im_layout.push_range);
         const struct hk_im2col_push ip = {
            .cols = 1,
            .rows_y = ph,
            .kh = 1,
         };
         memcpy((uint8_t *)push + im2_off, &ip, sizeof(ip));
         uint32_t *src = calloc(elems, 4);
         uint32_t *dst_orig = calloc(elems, 4);
         uint32_t *dst_id = calloc(elems, 4);
         uint32_t *dst_v4 = calloc(elems, 4);
         fill_pattern_u32(src, elems, 100 + add_base);
         struct hk_eval_ssbo so[2] = { {0, 0, src, elems * 4}, {0, 1, dst_orig, elems * 4} };
         struct hk_eval_ssbo si[2] = { {0, 0, src, elems * 4}, {0, 1, dst_id, elems * 4} };
         struct hk_eval_ssbo sv[2] = { {0, 0, src, elems * 4}, {0, 1, dst_v4, elems * 4} };
         uint32_t go[3] = { DIV_ROUND_UP(pw, 32), ph, 1 };
         uint32_t gi[3] = { DIV_ROUND_UP(elems / 4, 128 * 4), 1, 1 };
         uint32_t gv[3] = {
            DIV_ROUND_UP(DIV_ROUND_UP(pw, 4u), vec4_alt->info.workgroup_size[0]),
            DIV_ROUND_UP(1u, vec4_alt->info.workgroup_size[1]),
            DIV_ROUND_UP(DIV_ROUND_UP(ph, 4u), vec4_alt->info.workgroup_size[2]),
         };
         hk_eval_dispatch(orig, go, so, 2, push, im_layout.push_range);
         hk_eval_dispatch(id_alt, gi, si, 2, push, im_layout.push_range);
         hk_eval_dispatch(vec4_alt, gv, sv, 2, push, im2_off + sizeof(ip));
         HK_EVAL_CHECK(memcmp(dst_orig, dst_id, elems * 4) == 0);
         HK_EVAL_CHECK(memcmp(dst_orig, dst_v4, elems * 4) == 0);
         free(src);
         free(dst_orig);
         free(dst_id);
         free(dst_v4);
      }

      /* 1b. 3x3 padded im2col bit-exactness (orig vs vec4_alt). */
      {
         const int32_t bw = 1, bh = 1, ch = 8, iw = 7 * ch, ih = 5;
         const int32_t kw = 3 * ch, pw = kw * 3, ph = 7 * 5;
         uint32_t push[20] = {0};
         push[im_layout.bw / 4] = bw;
         push[im_layout.bh / 4] = bh;
         push[im_layout.iw / 4] = iw;
         push[im_layout.ih / 4] = ih;
         push[im_layout.kw / 4] = kw;
         push[im_layout.pw / 4] = pw;
         push[im_layout.ph / 4] = ph;
         push[im_layout.sw / 4] = 1;
         push[im_layout.sh / 4] = 1;
         push[im_layout.ch / 4] = ch;
         const uint32_t im2_off = hk_im2col_push_offset(im_layout.push_range);
         const struct hk_im2col_push ip = {
            .cols = 7,
            .rows_y = 5,
            .kh = 3,
         };
         memcpy((uint8_t *)push + im2_off, &ip, sizeof(ip));
         const uint32_t src_elems = (uint32_t)(iw * ih);
         const uint32_t dst_elems = (uint32_t)(pw * ph);
         uint32_t *src = calloc(src_elems, 4);
         uint32_t *dst_orig = calloc(dst_elems, 4);
         uint32_t *dst_v4 = calloc(dst_elems, 4);
         fill_pattern_u32(src, src_elems, 120 + add_base);
         struct hk_eval_ssbo so[2] = { {0, 0, src, src_elems * 4}, {0, 1, dst_orig, dst_elems * 4} };
         struct hk_eval_ssbo sv[2] = { {0, 0, src, src_elems * 4}, {0, 1, dst_v4, dst_elems * 4} };
         uint32_t go[3] = { DIV_ROUND_UP((uint32_t)pw, 32), (uint32_t)ph, 1 };
         uint32_t gv[3] = {
            DIV_ROUND_UP(DIV_ROUND_UP((uint32_t)kw, 4u), vec4_alt->info.workgroup_size[0]),
            DIV_ROUND_UP(ip.cols, vec4_alt->info.workgroup_size[1]),
            DIV_ROUND_UP(DIV_ROUND_UP(ip.rows_y, 4u), vec4_alt->info.workgroup_size[2]),
         };
         hk_eval_dispatch(orig, go, so, 2, push, im_layout.push_range);
         hk_eval_dispatch(vec4_alt, gv, sv, 2, push, im2_off + sizeof(ip));
         HK_EVAL_CHECK(memcmp(dst_orig, dst_v4, dst_elems * 4) == 0);
         free(src);
         free(dst_orig);
         free(dst_v4);
      }

      ralloc_free(vec4_alt);
      ralloc_free(id_alt);

      /* Negative 1: same load_workgroup_id.y with local_size[1] == 2 rejects. */
      orig->info.workgroup_size[1] = 2;
      struct hk_identity_copy4 bad = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(orig, &bad) == NULL);
      ralloc_free(orig);
   }

   /* Negative 2: wrong-axis base (wg.y + base.x) rejects. */
   {
      nir_shader *orig = build_im2col_shader(&im_layout, false);
      orig->info.workgroup_size[0] = 32;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 1, 1, true, 0, 1);
      struct hk_identity_copy4 bad = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(orig, &bad) == NULL);
      ralloc_free(orig);
   }

   /* Negative 3: base scaled by wrong constant (wg.y + base.y * 2) rejects. */
   {
      nir_shader *orig = build_im2col_shader(&im_layout, false);
      orig->info.workgroup_size[0] = 32;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 1, 1, true, 1, 2);
      struct hk_identity_copy4 bad = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(orig, &bad) == NULL);
      ralloc_free(orig);
   }

   /* Negative 4: wg.x used as row index with local_size[0] == 32 rejects. */
   {
      nir_shader *orig = build_im2col_shader(&im_layout, false);
      orig->info.workgroup_size[0] = 32;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 1, 0, false, 0, 1);
      struct hk_identity_copy4 bad = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(orig, &bad) == NULL);
      ralloc_free(orig);
   }

   /* 2. match_linear_1d with unit local_size = 1x1x1 reading wg.x (+ base.x). */
   for (unsigned add_base = 0; add_base < 2; ++add_base) {
      nir_shader *orig = build_linear_shader(
         HK_IDENTITY_COPY4_LINEAR_COPY, glsl_float_type(),
         ACCESS_NON_WRITEABLE, false, false, 0, 0, 0, 12);
      orig->info.workgroup_size[0] = 1;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 0, 0, add_base != 0, 0, 1);

      struct hk_identity_copy4 info = {0};
      nir_shader *alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(alt != NULL);
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_LINEAR_COPY &&
                    info.local[0] == 1);

      const uint32_t count = 524;
      uint32_t *src = calloc(count, 4);
      uint32_t *dst_orig = calloc(count, 4);
      uint32_t *dst_alt = calloc(count, 4);
      fill_pattern_u32(src, count, 140 + add_base);
      uint32_t push[3] = { count, 0, 0 };
      struct hk_eval_ssbo so[2] = { {0, 0, src, count * 4}, {0, 1, dst_orig, count * 4} };
      struct hk_eval_ssbo sa[2] = { {0, 0, src, count * 4}, {0, 1, dst_alt, count * 4} };
      uint32_t go[3] = { count, 1, 1 };
      uint32_t ga[3] = { DIV_ROUND_UP(count / 4, 128 * 4), 1, 1 };
      hk_eval_dispatch(orig, go, so, 2, push, sizeof(push));
      hk_eval_dispatch(alt, ga, sa, 2, push, sizeof(push));
      HK_EVAL_CHECK(memcmp(dst_orig, dst_alt, count * 4) == 0);
      free(src);
      free(dst_orig);
      free(dst_alt);
      ralloc_free(alt);

      orig->info.workgroup_size[0] = 4;
      struct hk_identity_copy4 bad = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(orig, &bad) == NULL);
      ralloc_free(orig);
   }

   /* 3. match_pad4d with wg=32x1x1 reading wg.y (+ base.y). */
   const uint16_t in_off[4] = { 40, 28, 12, 4 };
   const uint16_t out_off[4] = { 32, 20, 8, 44 };
   const uint32_t pad_idx[4] = { 0, 2, 4, 6 };
   for (unsigned add_base = 0; add_base < 2; ++add_base) {
      nir_shader *orig = build_pad4d_shader(in_off, out_off, pad_idx, 48, true);
      orig->info.workgroup_size[0] = 32;
      orig->info.workgroup_size[1] = 1;
      orig->info.workgroup_size[2] = 1;
      rewrite_gid_axis_to_wgid(orig, 1, 1, add_base != 0, 1, 1);

      struct hk_identity_copy4 info = {0};
      nir_shader *alt = hk_nir_copy_identity4(orig, &info);
      HK_EVAL_CHECK(alt != NULL);
      HK_EVAL_CHECK(info.kind == HK_IDENTITY_COPY4_PAD4 &&
                    info.local[0] == 32 && info.local[1] == 1);

      const int32_t in_s[4] = { 1, 5, 7, 8 };
      const int32_t out_s[4] = { 1, 7, 9, 8 };
      const int32_t pad[4] = { 0, 1, 1, 0 };
      const uint32_t in_n = (uint32_t)(in_s[0] * in_s[1] * in_s[2] * in_s[3]);
      const uint32_t out_n = (uint32_t)(out_s[0] * out_s[1] * out_s[2] * out_s[3]);
      uint32_t *in_buf = calloc(in_n, 4);
      uint32_t *dst_orig = calloc(out_n, 4);
      uint32_t *dst_alt = calloc(out_n, 4);
      fill_pattern_u32(in_buf, in_n, 160 + add_base);
      int32_t pad_buf[8] = {0};
      int32_t push[12] = {0};
      for (unsigned d = 0; d < 4; ++d) {
         pad_buf[pad_idx[d]] = pad[d];
         push[in_off[d] / 4] = in_s[d];
         push[out_off[d] / 4] = out_s[d];
      }
      struct hk_eval_ssbo so[3] = {
         {0, 0, in_buf, in_n * 4}, {0, 1, pad_buf, sizeof(pad_buf)}, {0, 2, dst_orig, out_n * 4},
      };
      struct hk_eval_ssbo sa[3] = {
         {0, 0, in_buf, in_n * 4}, {0, 1, pad_buf, sizeof(pad_buf)}, {0, 2, dst_alt, out_n * 4},
      };
      const uint32_t orig_x = (uint32_t)(out_s[2] * out_s[3]);
      const uint32_t orig_y = (uint32_t)(out_s[0] * out_s[1]);
      uint32_t go[3] = { DIV_ROUND_UP(orig_x, 32), orig_y, 1 };
      uint32_t ga[3] = {
         DIV_ROUND_UP(orig_x / 4, alt->info.workgroup_size[0]),
         DIV_ROUND_UP(orig_y, alt->info.workgroup_size[1]),
         1,
      };
      hk_eval_dispatch(orig, go, so, 3, push, sizeof(push));
      hk_eval_dispatch(alt, ga, sa, 3, push, sizeof(push));
      HK_EVAL_CHECK(memcmp(dst_orig, dst_alt, out_n * 4) == 0);
      free(in_buf);
      free(dst_orig);
      free(dst_alt);
      ralloc_free(alt);

      orig->info.workgroup_size[1] = 2;
      struct hk_identity_copy4 bad = {0};
      HK_EVAL_CHECK(hk_nir_copy_identity4(orig, &bad) == NULL);
      ralloc_free(orig);
   }

   /* 4. Dispatch-side nonzero base_group refusal gate (required for bare
    *    load_workgroup_id[axis] soundness). */
   const uint32_t zero_base[3] = { 0, 0, 0 };
   const uint32_t nz_x[3] = { 1, 0, 0 };
   const uint32_t nz_y[3] = { 0, 1, 0 };
   const uint32_t nz_z[3] = { 0, 0, 1 };
   HK_EVAL_CHECK(hk_nir_dispatch_zero_base(zero_base));
   HK_EVAL_CHECK(!hk_nir_dispatch_zero_base(nz_x));
   HK_EVAL_CHECK(!hk_nir_dispatch_zero_base(nz_y));
   HK_EVAL_CHECK(!hk_nir_dispatch_zero_base(nz_z));
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   test_linear_bit_exact();
   test_fadd_dual_bit_exact();
   test_pad4d_bit_exact();
   test_im2col_bit_exact();
   test_borderless_im2col_bit_exact();
   test_permuted_layout_im2col_bit_exact();
   test_unit_axis_wgid_matchers();

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
