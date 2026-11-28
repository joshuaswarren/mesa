/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_compute_opt.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"

#include "agx_compile.h"
#include "compiler/glsl_types.h"
#include "nir_builder.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define test_nir_options agx_nir_options
#define TEST_MAX_WG_X 0x7fffffffu
#define TEST_MAX_WG_Y 65535u

static struct hk_descriptor_set_layout *
make_test_layout(unsigned count, bool uab)
{
   size_t sz = sizeof(struct hk_descriptor_set_layout) +
               count * sizeof(struct hk_descriptor_set_binding_layout);
   struct hk_descriptor_set_layout *layout = calloc(1, sz);
   assert(layout);
   layout->binding_count = count;
   for (unsigned i = 0; i < count; i++) {
      layout->binding[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layout->binding[i].array_size = 1;
      layout->binding[i].offset = i * 16;
      layout->binding[i].stride = 16;
      layout->binding[i].flags =
         uab ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0;
   }
   memset(layout->vk.blake3, 0x55, sizeof(layout->vk.blake3));
   return layout;
}

static nir_deref_instr *
make_ssbo_base(nir_builder *b, uint32_t binding, const struct glsl_type *elem,
               unsigned stride)
{
   nir_def *ridx = nir_vulkan_resource_index(
      b, 2, 32, nir_imm_int(b, 0), .desc_set = 0, .binding = binding,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, 2, 32, ridx, .desc_type = nir_descriptor_type_storage_buffer);
   return nir_build_deref_cast(b, desc, nir_var_mem_ssbo, elem, stride);
}

static nir_def *
make_pc_i32(nir_builder *b, unsigned offset_B)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = offset_B,
                                 .range = 4);
}

static void
setup_desc_state(struct hk_descriptor_state *desc,
                 struct hk_descriptor_set *ds,
                 struct hk_descriptor_set_layout *layout,
                 struct hk_buffer_address *addrs, unsigned nbufs)
{
   memset(desc, 0, sizeof(*desc));
   memset(ds, 0, sizeof(*ds));
   ds->layout = layout;
   ds->size = nbufs * 16;
   ds->mapped_ptr = addrs;
   desc->sets[0] = ds;
}

static void
set_push_i32(struct hk_descriptor_state *desc, unsigned offset_B, int32_t val)
{
   memcpy(desc->root.push + offset_B, &val, sizeof(val));
   desc->push_written_dwords |= (UINT64_C(1) << (offset_B >> 2));
}

/* ---------------------------------------------------------------------------
 * 1. RGBA8 -> Grayscale u8 tests
 * --------------------------------------------------------------------------- */

static nir_shader *
build_test_rgba8_to_gray(bool bad_extra_store)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_rgba8_gray");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 1024;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint_type(), 4);
   nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *w = make_pc_i32(&b, 0);
   nir_def *h = make_pc_i32(&b, 4);
   nir_def *ss = make_pc_i32(&b, 8);
   nir_def *ds = make_pc_i32(&b, 12);

   nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
   {
      nir_def *px = nir_load_deref(
         &b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y, ss), x)));
      nir_def *r = nir_iand_imm(&b, px, 0xff);
      nir_def *g = nir_ubitfield_extract_imm(&b, px, 8, 8);
      nir_def *bl = nir_iand_imm(&b, nir_ushr_imm(&b, px, 16), 0xff);
      nir_def *sum = nir_iadd(
         &b, nir_iadd(&b, nir_imul_imm(&b, r, 9798), nir_imul_imm(&b, g, 19235)),
         nir_imul_imm(&b, bl, 3736));
      nir_def *out = nir_u2u8(&b, nir_ushr_imm(&b, sum, 15));
      nir_store_deref(
         &b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, nir_imul(&b, y, ds), x)),
         out, 0x1);
      if (bad_extra_store)
         nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_imm_int(&b, 0)),
                         out, 0x1);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
test_rgba8_to_gray(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(2, false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_shader *s = build_test_rgba8_to_gray(false);
   struct hk_compute_opt_plan plan = {0};
   assert(hk_nir_match_compute_opt(s, 1, layouts, &plan));
   assert(plan.kind == HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8);
   assert(plan.params[0] == 9798 && plan.params[1] == 19235 &&
          plan.params[2] == 3736 && plan.params[3] == 15);

   nir_shader *opt =
      hk_nir_build_compute_opt(NULL, s, &plan, nir_address_format_64bit_bounded_global);
   assert(opt != NULL);
   ralloc_free(opt);

   /* Dispatch validation & aliasing / fallback guards. */
   struct hk_buffer_address addrs[2] = {
      {.base_addr = 0x100000, .size = 1024 * 64 * 4},
      {.base_addr = 0x200000, .size = 1024 * 64},
   };
   struct hk_descriptor_state desc;
   struct hk_descriptor_set ds;
   setup_desc_state(&desc, &ds, layout, addrs, 2);
   set_push_i32(&desc, 0, 513);
   set_push_i32(&desc, 4, 64);
   set_push_i32(&desc, 8, 516);
   set_push_i32(&desc, 12, 516);

   uint32_t gx = 0, gy = 0;
   assert(hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 64, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 3 && gy == 16);
   /* Reject alias_free == false, non-zero base, and overlapping VA ranges. */
   assert(!hk_compute_opt_validate_dispatch_core(
      false, &desc, &plan, 0, 0, 0, 1, 64, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(!hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 1, 0, 0, 1, 64, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   addrs[1].base_addr = 0x100100;
   assert(!hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 64, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   ralloc_free(s);

   nir_shader *bad = build_test_rgba8_to_gray(true);
   assert(!hk_nir_match_compute_opt(bad, 1, layouts, &plan));
   ralloc_free(bad);

   /* Verify glslang-style short-circuit `if (x >= W || y >= H) return;` with
    * push-constant load inside the else arm + base_workgroup_id + imadshl_agx. */
   nir_builder sc = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_rgba8_shortcircuit");
   sc.shader->info.internal = false;
   sc.shader->info.workgroup_size[0] = 1024;
   sc.shader->info.workgroup_size[1] = 1;
   sc.shader->info.workgroup_size[2] = 1;
   sc.shader->info.num_ssbos = 2;
   nir_deref_instr *sc_src = make_ssbo_base(&sc, 0, glsl_uint_type(), 4);
   nir_deref_instr *sc_dst = make_ssbo_base(&sc, 1, glsl_uint8_t_type(), 1);
   nir_def *sc_gid = nir_load_global_invocation_id(&sc, 32);
   nir_def *sc_bwg = nir_load_base_workgroup_id(&sc, 32);
   nir_def *sc_x = nir_iadd(&sc, nir_channel(&sc, sc_gid, 0),
                            nir_ishl_imm(&sc, nir_channel(&sc, sc_bwg, 0), 10));
   nir_def *sc_y = nir_iadd(&sc, nir_channel(&sc, sc_gid, 1),
                            nir_channel(&sc, sc_bwg, 1));
   nir_def *sc_w = make_pc_i32(&sc, 0);
   nir_variable *oob = nir_local_variable_create(sc.impl, glsl_bool_type(), "oob");
   nir_store_var(&sc, oob, nir_imm_true(&sc), 0x1);
   nir_push_if(&sc, nir_ilt(&sc, sc_x, sc_w));
   {
      nir_def *sc_h = make_pc_i32(&sc, 4);
      nir_store_var(&sc, oob, nir_ige(&sc, sc_y, sc_h), 0x1);
   }
   nir_pop_if(&sc, NULL);
   nir_push_if(&sc, nir_inot(&sc, nir_load_var(&sc, oob)));
   {
      nir_def *sc_ss = make_pc_i32(&sc, 8);
      nir_def *sc_ds = make_pc_i32(&sc, 12);
      nir_def *sidx = nir_imadshl_agx(&sc, sc_y, sc_ss, sc_x, nir_imm_int(&sc, 0));
      nir_def *didx = nir_imadshl_agx(&sc, sc_y, sc_ds, sc_x, nir_imm_int(&sc, 0));
      nir_def *px = nir_load_deref(&sc, nir_build_deref_ptr_as_array(&sc, sc_src, sidx));
      nir_def *r = nir_iand_imm(&sc, px, 0xff);
      nir_def *g = nir_ubitfield_extract_imm(&sc, px, 8, 8);
      nir_def *bl = nir_iand_imm(&sc, nir_ushr_imm(&sc, px, 16), 0xff);
      nir_def *sum = nir_iadd(
         &sc, nir_iadd(&sc, nir_imul_imm(&sc, r, 9798), nir_imul_imm(&sc, g, 19235)),
         nir_imul_imm(&sc, bl, 3736));
      nir_store_deref(&sc, nir_build_deref_ptr_as_array(&sc, sc_dst, didx),
                      nir_u2u8(&sc, nir_ushr_imm(&sc, sum, 15)), 0x1);
   }
   nir_pop_if(&sc, NULL);
   struct hk_compute_opt_plan sc_plan = {0};
   assert(hk_nir_match_compute_opt(sc.shader, 1, layouts, &sc_plan));
   assert(sc_plan.kind == HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8);
   ralloc_free(sc.shader);

   /* Also verify u8vec4 (8x4) SSBO load form of RGBA8 -> Grayscale. */
   nir_builder v4 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_rgba8_u8vec4");
   v4.shader->info.internal = false;
   v4.shader->info.workgroup_size[0] = 1024;
   v4.shader->info.workgroup_size[1] = 1;
   v4.shader->info.workgroup_size[2] = 1;
   v4.shader->info.num_ssbos = 2;
   nir_deref_instr *v4_src =
      make_ssbo_base(&v4, 0, glsl_vector_type(GLSL_TYPE_UINT8, 4), 4);
   nir_deref_instr *v4_dst = make_ssbo_base(&v4, 1, glsl_uint8_t_type(), 1);
   nir_def *v4_gid = nir_load_global_invocation_id(&v4, 32);
   nir_def *v4_x = nir_channel(&v4, v4_gid, 0);
   nir_def *v4_y = nir_channel(&v4, v4_gid, 1);
   nir_def *v4_w = make_pc_i32(&v4, 0);
   nir_def *v4_h = make_pc_i32(&v4, 4);
   nir_def *v4_ss = make_pc_i32(&v4, 8);
   nir_def *v4_ds = make_pc_i32(&v4, 12);
   nir_push_if(&v4, nir_iand(&v4, nir_ilt(&v4, v4_x, v4_w),
                             nir_ilt(&v4, v4_y, v4_h)));
   {
      nir_def *px4 = nir_load_deref(
         &v4, nir_build_deref_ptr_as_array(
                 &v4, v4_src, nir_iadd(&v4, nir_imul(&v4, v4_y, v4_ss), v4_x)));
      nir_def *r = nir_u2u32(&v4, nir_channel(&v4, px4, 0));
      nir_def *g = nir_u2u32(&v4, nir_channel(&v4, px4, 1));
      nir_def *bl = nir_u2u32(&v4, nir_channel(&v4, px4, 2));
      nir_def *sum = nir_iadd(
         &v4, nir_iadd(&v4, nir_imul_imm(&v4, r, 9798), nir_imul_imm(&v4, g, 19235)),
         nir_imul_imm(&v4, bl, 3736));
      nir_store_deref(
         &v4, nir_build_deref_ptr_as_array(
                 &v4, v4_dst, nir_iadd(&v4, nir_imul(&v4, v4_y, v4_ds), v4_x)),
         nir_u2u8(&v4, nir_ushr_imm(&v4, sum, 15)), 0x1);
   }
   nir_pop_if(&v4, NULL);
   struct hk_compute_opt_plan v4_plan = {0};
   assert(hk_nir_match_compute_opt(v4.shader, 1, layouts, &v4_plan));
   assert(v4_plan.kind == HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8);
   ralloc_free(v4.shader);
   free(layout);
}

/* ---------------------------------------------------------------------------
 * 2. Canny Hysteresis u8 tests
 * --------------------------------------------------------------------------- */

static nir_shader *
build_test_canny_hysteresis(bool missing_neighbor)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_hyst");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 128;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *w = make_pc_i32(&b, 0);
   nir_def *h = make_pc_i32(&b, 4);
   nir_def *idx = nir_iadd(&b, nir_imul(&b, y, w), x);

   nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
   {
      nir_def *v = nir_u2u32(
         &b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, idx)));
      nir_variable *out = nir_local_variable_create(b.impl, glsl_int_type(), "o");
      nir_store_var(&b, out, nir_imm_int(&b, 3), 0x1);
      nir_push_if(&b, nir_ine_imm(&b, v, 2));
      {
         nir_store_var(&b, out, v, 0x1);
         nir_push_if(&b, nir_ieq_imm(&b, v, 1));
         {
            nir_def *wm1 = nir_iadd_imm(&b, w, -1);
            nir_def *hm1 = nir_iadd_imm(&b, h, -1);
            nir_def *xgt0 = nir_ilt(&b, nir_imm_int(&b, 0), x);
            nir_def *ygt0 = nir_ilt(&b, nir_imm_int(&b, 0), y);
            nir_def *xlt = nir_ilt(&b, x, wm1);
            nir_def *ylt = nir_ilt(&b, y, hm1);
            nir_def *interior =
               nir_iand(&b, nir_iand(&b, xgt0, ygt0), nir_iand(&b, xlt, ylt));
            static const int dxs[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
            static const int dys[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
            nir_push_if(&b, interior);
            {
               nir_def *mx = NULL;
               unsigned ncnt = missing_neighbor ? 7 : 8;
               for (unsigned k = 0; k < ncnt; k++) {
                  nir_def *ni = nir_iadd(&b, nir_iadd_imm(&b, idx, dxs[k]),
                                         nir_imul_imm(&b, w, dys[k]));
                  nir_def *nv = nir_u2u32(
                     &b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, ni)));
                  mx = mx ? nir_imax(&b, mx, nv) : nv;
               }
               nir_store_var(
                  &b, out,
                  nir_bcsel(&b, nir_ige_imm(&b, mx, 2), nir_imm_int(&b, 3),
                            nir_imm_int(&b, 1)),
                  0x1);
            }
            nir_push_else(&b, NULL);
            {
               nir_store_var(&b, out, nir_imm_int(&b, 1), 0x1);
               for (unsigned k = 0; k < 8; k++) {
                  nir_def *c = NULL;
                  if (dxs[k] == -1)
                     c = c ? nir_iand(&b, c, xgt0) : xgt0;
                  if (dxs[k] == 1)
                     c = c ? nir_iand(&b, c, xlt) : xlt;
                  if (dys[k] == -1)
                     c = c ? nir_iand(&b, c, ygt0) : ygt0;
                  if (dys[k] == 1)
                     c = c ? nir_iand(&b, c, ylt) : ylt;
                  nir_push_if(&b, c);
                  {
                     nir_def *ni = nir_iadd(&b, nir_iadd_imm(&b, idx, dxs[k]),
                                            nir_imul_imm(&b, w, dys[k]));
                     nir_def *nv = nir_u2u32(
                        &b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, ni)));
                     nir_store_var(
                        &b, out,
                        nir_bcsel(&b, nir_ige_imm(&b, nv, 2), nir_imm_int(&b, 3),
                                  nir_load_var(&b, out)),
                        0x1);
                  }
                  nir_pop_if(&b, NULL);
               }
            }
            nir_pop_if(&b, NULL);
         }
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);

      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, idx),
                      nir_u2u8(&b, nir_load_var(&b, out)), 0x1);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
test_canny_hysteresis(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(2, false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_shader *s = build_test_canny_hysteresis(false);
   struct hk_compute_opt_plan plan = {0};
   assert(hk_nir_match_compute_opt(s, 1, layouts, &plan));
   assert(plan.kind == HK_COMPUTE_OPT_CANNY_HYSTERESIS_U8);
   assert(plan.tile_w == 256 && plan.tile_h == 4);

   nir_shader *opt =
      hk_nir_build_compute_opt(NULL, s, &plan, nir_address_format_64bit_bounded_global);
   assert(opt != NULL);
   ralloc_free(opt);

   struct hk_buffer_address addrs[2] = {
      {.base_addr = 0x300000, .size = 257 * 17},
      {.base_addr = 0x400000, .size = 257 * 17},
   };
   struct hk_descriptor_state desc;
   struct hk_descriptor_set ds;
   setup_desc_state(&desc, &ds, layout, addrs, 2);
   set_push_i32(&desc, 0, 257);
   set_push_i32(&desc, 4, 17);

   uint32_t gx = 0, gy = 0;
   assert(hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 3, 17, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 2 && gy == 5);
   /* Reject in-place aliasing (src == dst). */
   addrs[1].base_addr = addrs[0].base_addr;
   assert(!hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 3, 17, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   ralloc_free(s);

   nir_shader *bad = build_test_canny_hysteresis(true);
   assert(!hk_nir_match_compute_opt(bad, 1, layouts, &plan));
   ralloc_free(bad);
   free(layout);
}

/* ---------------------------------------------------------------------------
 * 3. Canny NMS tests
 * --------------------------------------------------------------------------- */

static nir_shader *
build_test_canny_nms(bool bad_lut)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_nms");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 1024;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 3;

   nir_deref_instr *dir = make_ssbo_base(&b, 0, glsl_int_type(), 4);
   nir_deref_instr *mag = make_ssbo_base(&b, 1, glsl_float_type(), 4);
   nir_deref_instr *dst = make_ssbo_base(&b, 2, glsl_uint8_t_type(), 1);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *w = make_pc_i32(&b, 0);
   nir_def *h = make_pc_i32(&b, 4);

   nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
   {
      nir_def *idx = nir_iadd(&b, nir_imul(&b, y, w), x);
      nir_def *interior = nir_iand(
         &b, nir_iand(&b, nir_ige_imm(&b, x, 1), nir_ilt(&b, x, nir_iadd_imm(&b, w, -1))),
         nir_iand(&b, nir_ige_imm(&b, y, 1), nir_ilt(&b, y, nir_iadd_imm(&b, h, -1))));
      nir_variable *out = nir_local_variable_create(b.impl, glsl_int_type(), "o");
      nir_store_var(&b, out, nir_imm_int(&b, 0), 0x1);
      nir_push_if(&b, interior);
      {
         nir_def *m = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, mag, idx));
         nir_def *d = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, dir, idx));
         nir_def *sh = nir_iand_imm(&b, nir_ishl_imm(&b, d, 1), 31);
         uint32_t dx_lut = bad_lut ? 0xffffu : 0x901au;
         nir_def *dx = nir_iadd_imm(
            &b, nir_ubitfield_extract(&b, nir_imm_int(&b, (int32_t)dx_lut), sh,
                                      nir_imm_int(&b, 2)),
            -1);
         nir_def *dy = nir_iadd_imm(
            &b, nir_ubitfield_extract(&b, nir_imm_int(&b, 0xa901), sh,
                                      nir_imm_int(&b, 2)),
            -1);
         nir_def *i1 = nir_iadd(&b, nir_imul(&b, nir_iadd(&b, y, dy), w),
                                nir_iadd(&b, x, dx));
         nir_def *i2 = nir_iadd(&b, nir_imul(&b, nir_isub(&b, y, dy), w),
                                nir_isub(&b, x, dx));
         nir_def *m1 = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, mag, i1));
         nir_def *m2 = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, mag, i2));
         nir_def *t1 = nir_bcsel(&b, nir_flt(&b, nir_imm_float(&b, 11.0f), m),
                                 nir_imm_int(&b, 1), nir_imm_int(&b, 0));
         nir_def *t2 = nir_bcsel(&b, nir_flt(&b, nir_imm_float(&b, 33.0f), m),
                                 nir_imm_int(&b, 2), t1);
         nir_def *r1 = nir_bcsel(&b, nir_flt(&b, m2, m), t2, nir_imm_int(&b, 0));
         nir_def *r2 = nir_bcsel(&b, nir_flt(&b, m1, m), r1, nir_imm_int(&b, 0));
         nir_store_var(&b, out, r2, 0x1);
      }
      nir_pop_if(&b, NULL);
      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, idx),
                      nir_u2u8(&b, nir_load_var(&b, out)), 0x1);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
test_canny_nms(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(3, false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_shader *s = build_test_canny_nms(false);
   struct hk_compute_opt_plan plan = {0};
   assert(hk_nir_match_compute_opt(s, 1, layouts, &plan));
   assert(plan.kind == HK_COMPUTE_OPT_CANNY_NMS_U8);
   assert(plan.params[0] == 0x901au && plan.params[1] == 0xa901u);

   nir_shader *opt =
      hk_nir_build_compute_opt(NULL, s, &plan, nir_address_format_64bit_bounded_global);
   assert(opt != NULL);
   ralloc_free(opt);

   struct hk_buffer_address addrs[3] = {
      {.base_addr = 0x100000, .size = 640 * 480 * 4},
      {.base_addr = 0x400000, .size = 640 * 480 * 4},
      {.base_addr = 0x800000, .size = 640 * 480},
   };
   struct hk_descriptor_state desc;
   struct hk_descriptor_set ds;
   setup_desc_state(&desc, &ds, layout, addrs, 3);
   set_push_i32(&desc, 0, 640);
   set_push_i32(&desc, 4, 480);

   uint32_t gx = 0, gy = 0;
   assert(hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 480, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 3 && gy == 120);
   ralloc_free(s);

   /* Reject invalid LUT entry (3 -> +2 offset could read out of 1px border). */
   nir_shader *bad = build_test_canny_nms(true);
   assert(!hk_nir_match_compute_opt(bad, 1, layouts, &plan));
   ralloc_free(bad);
   free(layout);
}

/* ---------------------------------------------------------------------------
 * 4. Canny Finalize tests
 * --------------------------------------------------------------------------- */

static nir_shader *
build_test_canny_finalize(bool wrong_clamp)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_finalize");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 1024;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 3;

   nir_deref_instr *st = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *mag = make_ssbo_base(&b, 1, glsl_float_type(), 4);
   nir_deref_instr *dst = make_ssbo_base(&b, 2, glsl_uint8_t_type(), 1);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *w = make_pc_i32(&b, 0);
   nir_def *h = make_pc_i32(&b, 4);
   nir_def *ds = make_pc_i32(&b, 8);

   nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
   {
      nir_def *sidx = nir_iadd(&b, nir_imul(&b, y, w), x);
      nir_def *sv = nir_u2u32(
         &b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, st, sidx)));
      nir_variable *out = nir_local_variable_create(b.impl, glsl_int_type(), "o");
      nir_store_var(&b, out, nir_imm_int(&b, 0), 0x1);
      nir_push_if(&b, nir_ige_imm(&b, sv, 3));
      {
         nir_def *m = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, mag, sidx));
         nir_def *hi = nir_imm_float(&b, wrong_clamp ? 128.0f : 255.0f);
         nir_def *c = nir_fmin(&b, nir_fmax(&b, m, nir_imm_float(&b, 0.0f)), hi);
         nir_store_var(&b, out, nir_f2i32(&b, c), 0x1);
      }
      nir_pop_if(&b, NULL);
      nir_def *didx = nir_iadd(&b, nir_imul(&b, y, ds), x);
      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, didx),
                      nir_u2u8(&b, nir_load_var(&b, out)), 0x1);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
test_canny_finalize(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(3, false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_shader *s = build_test_canny_finalize(false);
   struct hk_compute_opt_plan plan = {0};
   assert(hk_nir_match_compute_opt(s, 1, layouts, &plan));
   assert(plan.kind == HK_COMPUTE_OPT_CANNY_FINALIZE_U8);
   assert(plan.params[0] == 3);

   nir_shader *opt =
      hk_nir_build_compute_opt(NULL, s, &plan, nir_address_format_64bit_bounded_global);
   assert(opt != NULL);
   ralloc_free(opt);

   struct hk_buffer_address addrs[3] = {
      {.base_addr = 0x100000, .size = 300 * 200},
      {.base_addr = 0x200000, .size = 300 * 200 * 4},
      {.base_addr = 0x500000, .size = 320 * 200},
   };
   struct hk_descriptor_state desc;
   struct hk_descriptor_set ds;
   setup_desc_state(&desc, &ds, layout, addrs, 3);
   set_push_i32(&desc, 0, 300);
   set_push_i32(&desc, 4, 200);
   set_push_i32(&desc, 8, 320);

   uint32_t gx = 0, gy = 0;
   assert(hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 200, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 2 && gy == 50);
   ralloc_free(s);

   nir_shader *bad = build_test_canny_finalize(true);
   assert(!hk_nir_match_compute_opt(bad, 1, layouts, &plan));
   ralloc_free(bad);
   free(layout);
}

static void
emit_test_mirror_border(nir_builder *b, nir_deref_instr *dst, nir_def *x,
                        nir_def *y, nir_def *val, nir_def *dw, nir_def *dh,
                        nir_def *ds, nir_def *border)
{
   nir_def *y_row = nir_imul(b, nir_iadd(b, y, border), ds);
   nir_def *px = nir_iadd(b, x, border);
   nir_store_deref(b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, y_row, px)),
                   val, 0x1);

   nir_def *x_lo = nir_isub(b, nir_iadd_imm(b, dw, -2), border);
   nir_def *x_hi = nir_iadd_imm(b, dw, -1);
   nir_def *y_lo = nir_isub(b, nir_iadd_imm(b, dh, -2), border);
   nir_def *y_hi = nir_iadd_imm(b, dh, -1);
   nir_def *xl = nir_uge(b, border, x);
   nir_def *xr = nir_iand(b, nir_ult(b, x_lo, x), nir_ult(b, x, x_hi));
   nir_def *yt = nir_uge(b, border, y);
   nir_def *yb = nir_iand(b, nir_ult(b, y_lo, y), nir_ult(b, y, y_hi));
   nir_def *ya = nir_ior(b, yt, yb);

   nir_push_if(b, nir_ior(b, nir_ior(b, xl, xr), ya));
   {
      nir_def *xl_nz = nir_iand(b, nir_ine_imm(b, x, 0), xl);
      nir_def *yt_nz = nir_iand(b, nir_ine_imm(b, y, 0), yt);
      nir_def *ry_t = nir_imul(b, nir_isub(b, border, y), ds);
      nir_def *ry_b = nir_imul(
         b,
         nir_isub(b, nir_iadd(b, nir_ishl_imm(b, dh, 1),
                              nir_iadd_imm(b, border, -2)),
                  y),
         ds);
      nir_push_if(b, xl_nz);
      {
         nir_def *rx_l = nir_isub(b, border, x);
         nir_store_deref(
            b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, y_row, rx_l)),
            val, 0x1);
         nir_push_if(b, ya);
         {
            nir_push_if(b, yt_nz);
            nir_store_deref(
               b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, ry_t, rx_l)),
               val, 0x1);
            nir_pop_if(b, NULL);
            nir_push_if(b, yb);
            nir_store_deref(
               b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, ry_b, rx_l)),
               val, 0x1);
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
      nir_push_if(b, yt_nz);
      nir_store_deref(
         b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, ry_t, px)), val, 0x1);
      nir_pop_if(b, NULL);
      nir_push_if(b, xr);
      {
         nir_def *rx_r = nir_isub(
            b, nir_iadd(b, nir_ishl_imm(b, dw, 1), nir_iadd_imm(b, border, -2)),
            x);
         nir_store_deref(
            b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, y_row, rx_r)),
            val, 0x1);
         nir_push_if(b, ya);
         {
            nir_push_if(b, yt_nz);
            nir_store_deref(
               b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, ry_t, rx_r)),
               val, 0x1);
            nir_pop_if(b, NULL);
            nir_push_if(b, yb);
            nir_store_deref(
               b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, ry_b, rx_r)),
               val, 0x1);
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
      nir_push_if(b, yb);
      nir_store_deref(
         b, nir_build_deref_ptr_as_array(b, dst, nir_iadd(b, ry_b, px)), val, 0x1);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static nir_shader *
build_test_bilinear_resize_border(bool wrong_clamp)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "resize_border_test");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);

   nir_def *sw_sh = nir_load_push_constant(&b, 2, 32, nir_imm_int(&b, 0),
                                           .base = 0, .range = 8);
   nir_def *sw = nir_channel(&b, sw_sh, 0);
   nir_def *sh = nir_channel(&b, sw_sh, 1);
   nir_def *ss = make_pc_i32(&b, 8);
   nir_def *dst_p = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0),
                                           .base = 12, .range = 16);
   nir_def *dw = nir_channel(&b, dst_p, 0);
   nir_def *dh = nir_channel(&b, dst_p, 1);
   nir_def *ds = nir_channel(&b, dst_p, 2);
   nir_def *border = nir_channel(&b, dst_p, 3);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx0 = nir_channel(&b, gid, 0);
   nir_def *gy0 = nir_channel(&b, gid, 1);
   nir_def *nwg = nir_load_num_workgroups(&b);
   nir_def *step_x = nir_ishl_imm(&b, nir_channel(&b, nwg, 0), 5);
   nir_def *step_y = nir_ishl_imm(&b, nir_channel(&b, nwg, 1), 3);

   nir_push_if(&b, nir_ior(&b, nir_ine(&b, sw, dw), nir_ine(&b, sh, dh)));
   {
      nir_def *sx_sc = nir_fmul(&b, nir_i2f32(&b, sw), nir_frcp(&b, nir_i2f32(&b, dw)));
      nir_def *sy_sc = nir_fmul(&b, nir_i2f32(&b, sh), nir_frcp(&b, nir_i2f32(&b, dh)));
      nir_def *swm1 = nir_iadd_imm(&b, sw, -1);
      nir_def *shm1 = nir_iadd_imm(&b, sh, -1);

      nir_variable *yv = nir_local_variable_create(b.impl, glsl_int_type(), "ry");
      nir_store_var(&b, yv, gy0, 0x1);
      nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, yv);
         nir_push_if(&b, nir_uge(&b, y, dh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *sy = nir_fmul(&b, nir_u2f32(&b, y), sy_sc);
         nir_def *fy = nir_fsub(&b, sy, nir_ftrunc(&b, sy));
         nir_def *iy = nir_f2i32(&b, sy);
         nir_def *y0 = nir_imin(&b, nir_imax(&b, iy, nir_imm_int(&b, 0)), shm1);
         nir_def *y1 = nir_imin(&b, nir_imax(&b, nir_iadd_imm(&b, iy, 1), nir_imm_int(&b, 0)), shm1);
         nir_def *r0 = nir_imul(&b, y0, ss);
         nir_def *r1 = nir_imul(&b, y1, ss);

         nir_variable *xv = nir_local_variable_create(b.impl, glsl_int_type(), "rx");
         nir_store_var(&b, xv, gx0, 0x1);
         nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, xv);
            nir_push_if(&b, nir_uge(&b, x, dw));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);

            nir_def *sx = nir_fmul(&b, nir_u2f32(&b, x), sx_sc);
            nir_def *fx = nir_fsub(&b, sx, nir_ftrunc(&b, sx));
            nir_def *ix = nir_f2i32(&b, sx);
            nir_def *x0 = nir_imin(&b, nir_imax(&b, ix, nir_imm_int(&b, 0)), swm1);
            nir_def *x1 = nir_imin(&b, nir_imax(&b, nir_iadd_imm(&b, ix, 1), nir_imm_int(&b, 0)), swm1);
            nir_def *p00 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, r0, x0))));
            nir_def *p01 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, r0, x1))));
            nir_def *p10 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, r1, x0))));
            nir_def *p11 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, r1, x1))));
            nir_def *top = nir_ffma(&b, fx, nir_fsub(&b, p01, p00), p00);
            nir_def *bot = nir_ffma(&b, fx, nir_fsub(&b, p11, p10), p10);
            nir_def *val = nir_ffma(&b, fy, nir_fsub(&b, bot, top), top);
            nir_def *cl = nir_fmin(&b, nir_fmax(&b, val, nir_imm_float(&b, 0.0f)),
                                   nir_imm_float(&b, wrong_clamp ? 128.0f : 255.0f));
            nir_def *out = nir_u2u8(&b, nir_f2i32(&b, cl));
            emit_test_mirror_border(&b, dst, x, y, out, dw, dh, ds, border);
            nir_store_var(&b, xv, nir_iadd(&b, x, step_x), 0x1);
         }
         nir_pop_loop(&b, NULL);
         nir_store_var(&b, yv, nir_iadd(&b, y, step_y), 0x1);
      }
      nir_pop_loop(&b, NULL);
   }
   nir_push_else(&b, NULL);
   {
      nir_variable *yv = nir_local_variable_create(b.impl, glsl_int_type(), "cy");
      nir_store_var(&b, yv, gy0, 0x1);
      nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, yv);
         nir_push_if(&b, nir_uge(&b, y, dh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_variable *xv = nir_local_variable_create(b.impl, glsl_int_type(), "cx");
         nir_store_var(&b, xv, gx0, 0x1);
         nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, xv);
            nir_push_if(&b, nir_uge(&b, x, dw));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);

            nir_def *v = nir_load_deref(
               &b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y, ss), x)));
            emit_test_mirror_border(&b, dst, x, y, v, dw, dh, ds, border);
            nir_store_var(&b, xv, nir_iadd(&b, x, step_x), 0x1);
         }
         nir_pop_loop(&b, NULL);
         nir_store_var(&b, yv, nir_iadd(&b, y, step_y), 0x1);
      }
      nir_pop_loop(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "build_test_bilinear_resize_border");
   return b.shader;
}

static void
test_bilinear_resize_border(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(2, false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_shader *s = build_test_bilinear_resize_border(false);
   struct hk_compute_opt_plan plan = {0};
   assert(hk_nir_match_compute_opt(s, 1, layouts, &plan));
   assert(plan.kind == HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8);
   assert(plan.flags == HK_RESIZE_LERP_DIFF_FFMA);
   assert(plan.tile_w == 128 && plan.tile_h == 8);
   assert(plan.pc_off[0] == 12 && plan.pc_off[1] == 16 &&
          plan.pc_off[2] == 20 && plan.pc_off[3] == 24 &&
          plan.pc_off[4] == 0 && plan.pc_off[5] == 4 && plan.pc_off[6] == 8);

   nir_shader *opt = hk_nir_build_compute_opt(
      NULL, s, &plan, nir_address_format_64bit_bounded_global);
   assert(opt != NULL);
   ralloc_free(opt);

   struct hk_buffer_address addrs[2] = {
      {.base_addr = 0x100000, .size = 1024 * 768},
      {.base_addr = 0x500000, .size = 1056 * 800},
   };
   struct hk_descriptor_state desc;
   struct hk_descriptor_set ds;
   setup_desc_state(&desc, &ds, layout, addrs, 2);
   set_push_i32(&desc, 0, 1024);
   set_push_i32(&desc, 4, 768);
   set_push_i32(&desc, 8, 1024);
   set_push_i32(&desc, 12, 800);
   set_push_i32(&desc, 16, 600);
   set_push_i32(&desc, 20, 832);
   set_push_i32(&desc, 24, 16);

   uint32_t gx = 0, gy = 0;
   assert(hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 4, 4, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 7 && gy == 75);

   /* Enforce device workgroup count limits on rewritten grid dimensions:
    * tall narrow buffer with ceil(height / 8) == 65535 fits; 65536 must be
    * rejected so the original grid-stride dispatch is retained. */
   struct hk_buffer_address tall_addrs[2] = {
      {.base_addr = 0x1000000, .size = 16u * 1024u * 1024u},
      {.base_addr = 0x3000000, .size = 16u * 1024u * 1024u},
   };
   setup_desc_state(&desc, &ds, layout, tall_addrs, 2);
   set_push_i32(&desc, 0, 16);
   set_push_i32(&desc, 4, 65535 * 8);
   set_push_i32(&desc, 8, 16);
   set_push_i32(&desc, 12, 16);
   set_push_i32(&desc, 16, 65535 * 8);
   set_push_i32(&desc, 20, 16);
   set_push_i32(&desc, 24, 0);
   assert(hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 64, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 1 && gy == 65535);

   set_push_i32(&desc, 4, 65535 * 8 + 1);
   set_push_i32(&desc, 16, 65535 * 8 + 1);
   gx = 0xdeadbeefu;
   gy = 0xbeefdeadu;
   assert(!hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 64, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));
   assert(gx == 0xdeadbeefu && gy == 0xbeefdeadu);

   /* Also reject rewritten X group count exceeding max_groups_x. */
   set_push_i32(&desc, 0, 256);
   set_push_i32(&desc, 4, 64);
   set_push_i32(&desc, 8, 256);
   set_push_i32(&desc, 12, 256);
   set_push_i32(&desc, 16, 64);
   set_push_i32(&desc, 20, 256);
   assert(!hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 8, 1, 1, TEST_MAX_WG_Y, &gx, &gy));

   /* Reject border=0 with width < 2 (guards against dst_w - 2 - border underflow). */
   setup_desc_state(&desc, &ds, layout, addrs, 2);
   set_push_i32(&desc, 0, 1024);
   set_push_i32(&desc, 4, 768);
   set_push_i32(&desc, 8, 1024);
   set_push_i32(&desc, 12, 1);
   set_push_i32(&desc, 16, 600);
   set_push_i32(&desc, 20, 832);
   set_push_i32(&desc, 24, 0);
   assert(!hk_compute_opt_validate_dispatch_core(
      true, &desc, &plan, 0, 0, 0, 1, 1, 1, TEST_MAX_WG_X, TEST_MAX_WG_Y, &gx,
      &gy));

   /* Non-default float execution controls must be preserved on replacement NIR. */
   const uint32_t fc_modes[] = {
      FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32 |
         FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32 |
         FLOAT_CONTROLS_DENORM_PRESERVE_FP32,
   };
   for (unsigned fi = 0; fi < ARRAY_SIZE(fc_modes); fi++) {
      s->info.float_controls_execution_mode = fc_modes[fi];
      struct hk_compute_opt_plan fc_plan = {0};
      assert(hk_nir_match_compute_opt(s, 1, layouts, &fc_plan));
      nir_shader *fc_opt = hk_nir_build_compute_opt(
         NULL, s, &fc_plan, nir_address_format_64bit_bounded_global);
      assert(fc_opt != NULL);
      assert(fc_opt->info.float_controls_execution_mode == fc_modes[fi]);
      ralloc_free(fc_opt);
   }
   s->info.float_controls_execution_mode = 0;

   /* Verify unlowered relaxed nir_op_fdiv matches (as produced after deferred
    * fdiv lowering), while exact nir_op_fdiv is rejected. */
   {
      nir_shader *fdiv_s = build_test_bilinear_resize_border(false);
      nir_foreach_function_impl(impl, fdiv_s) {
         nir_builder rb = nir_builder_create(impl);
         nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op != nir_op_fmul)
                  continue;
               nir_instr *s1 = nir_def_instr(alu->src[1].src.ssa);
               if (s1->type != nir_instr_type_alu ||
                   nir_instr_as_alu(s1)->op != nir_op_frcp)
                  continue;
               rb.cursor = nir_before_instr(instr);
               nir_def *q = nir_fdiv(&rb, alu->src[0].src.ssa,
                                     nir_instr_as_alu(s1)->src[0].src.ssa);
               nir_def_rewrite_uses(&alu->def, q);
               nir_instr_remove(instr);
               nir_instr_remove(s1);
            }
         }
      }
      struct hk_compute_opt_plan fdiv_plan = {0};
      assert(hk_nir_match_compute_opt(fdiv_s, 1, layouts, &fdiv_plan));
      assert(fdiv_plan.kind == HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8);

      nir_foreach_function_impl(impl, fdiv_s) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   nir_instr_as_alu(instr)->op == nir_op_fdiv)
                  nir_instr_as_alu(instr)->fp_math_ctrl = nir_fp_exact;
            }
         }
      }
      assert(!hk_nir_match_compute_opt(fdiv_s, 1, layouts, &fdiv_plan));
      ralloc_free(fdiv_s);
   }

   /* SPIR-V FClamp lowers fmax/fmin with nir_fp_preserve_inf; matcher must accept
    * and propagate it onto the emitted clamp ops. */
   {
      nir_shader *fp_inf = build_test_bilinear_resize_border(false);
      unsigned marked_inf = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(fp_inf)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(ins);
               if (alu->op == nir_op_fmax || alu->op == nir_op_fmin) {
                  alu->fp_math_ctrl = nir_fp_preserve_inf;
                  marked_inf++;
               }
            }
         }
      }
      assert(marked_inf == 2);
      struct hk_compute_opt_plan inf_plan = {0};
      assert(hk_nir_match_compute_opt(fp_inf, 1, layouts, &inf_plan));
      assert(inf_plan.params[0] == nir_fp_preserve_inf &&
             inf_plan.params[1] == nir_fp_preserve_inf);
      nir_shader *inf_opt = hk_nir_build_compute_opt(
         NULL, fp_inf, &inf_plan, nir_address_format_64bit_bounded_global);
      assert(inf_opt != NULL);
      unsigned saw_fmax_inf = 0, saw_fmin_inf = 0;
      nir_foreach_block(blk, nir_shader_get_entrypoint(inf_opt)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(ins);
               if (alu->op == nir_op_fmax) {
                  assert(alu->fp_math_ctrl == nir_fp_preserve_inf);
                  saw_fmax_inf++;
               } else if (alu->op == nir_op_fmin) {
                  assert(alu->fp_math_ctrl == nir_fp_preserve_inf);
                  saw_fmin_inf++;
               }
            }
         }
      }
      assert(saw_fmax_inf > 0 && saw_fmin_inf > 0);
      ralloc_free(inf_opt);
      ralloc_free(fp_inf);
   }

   /* Per-ALU float math control flags that the builder does not emit must be refused. */
   {
      nir_shader *fp_mod = build_test_bilinear_resize_border(false);
      bool marked = false;
      nir_foreach_block(blk, nir_shader_get_entrypoint(fp_mod)) {
         nir_foreach_instr(ins, blk) {
            if (ins->type == nir_instr_type_alu &&
                nir_instr_as_alu(ins)->op == nir_op_fmin) {
               nir_instr_as_alu(ins)->fp_math_ctrl = nir_fp_preserve_nan;
               marked = true;
               break;
            }
         }
         if (marked)
            break;
      }
      assert(marked);
      struct hk_compute_opt_plan fp_plan = {0};
      assert(!hk_nir_match_compute_opt(fp_mod, 1, layouts, &fp_plan));
      ralloc_free(fp_mod);
   }
   ralloc_free(s);

   nir_shader *bad = build_test_bilinear_resize_border(true);
   assert(!hk_nir_match_compute_opt(bad, 1, layouts, &plan));
   ralloc_free(bad);

   /* Verify pre-lowered GB6 form: in-arm primary stores, dst reload at primary
    * index, (1 - t) * a + t * b lerp, and 4 direct sibling mirror border ifs.
    * variant 0 = valid GB6 form `(1-t)*a + t*b`; 1 = bad reload; 2 = mismatched
    * scale grouping `(x * sw) * frcp(dw)`; 3 = unlowered flrp (must reject);
    * 4 = reversed-operand `t*b + (1-t)*a` (must match as ONE_MINUS_ADD_B and
    * preserve fadd operand order so algebraic late-FMA fuses identically). */
   for (unsigned variant = 0; variant < 5; variant++) {
      const bool bad_reload = (variant == 1);
      const bool bad_scale = (variant == 2);
      const bool use_flrp = (variant == 3);
      const bool rev_add = (variant == 4);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &test_nir_options, "resize_prelower");
      b.shader->info.internal = false;
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      b.shader->info.num_ssbos = 2;

      nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
      nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);
      nir_def *sw = make_pc_i32(&b, 0);
      nir_def *sh = make_pc_i32(&b, 4);
      nir_def *ss = make_pc_i32(&b, 8);
      nir_def *dw = make_pc_i32(&b, 12);
      nir_def *dh = make_pc_i32(&b, 16);
      nir_def *ds = make_pc_i32(&b, 20);
      nir_def *border = make_pc_i32(&b, 24);
      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_def *bwg = nir_load_base_workgroup_id(&b, 32);
      nir_def *nwg = nir_load_num_workgroups(&b);
      nir_def *gx0 = nir_iadd(&b, nir_channel(&b, gid, 0),
                              nir_ishl_imm(&b, nir_channel(&b, bwg, 0), 5));
      nir_def *gy0 = nir_iadd(&b, nir_channel(&b, gid, 1),
                              nir_channel(&b, bwg, 1));
      nir_def *step_x = nir_ishl_imm(&b, nir_channel(&b, nwg, 0), 5);
      nir_def *step_y = nir_channel(&b, nwg, 1);
      nir_def *sx_scale =
         nir_fmul(&b, nir_i2f32(&b, sw), nir_frcp(&b, nir_i2f32(&b, dw)));
      nir_def *sy_scale =
         nir_fmul(&b, nir_i2f32(&b, sh), nir_frcp(&b, nir_i2f32(&b, dh)));

      nir_variable *yv = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_store_var(&b, yv, gy0, 0x1);
      nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, yv);
         nir_push_if(&b, nir_uge(&b, y, dh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
         nir_variable *xv = nir_local_variable_create(b.impl, glsl_int_type(), "x");
         nir_store_var(&b, xv, gx0, 0x1);
         nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, xv);
            nir_push_if(&b, nir_uge(&b, x, dw));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);
            nir_def *do_res = nir_ior(&b, nir_ine(&b, sw, dw), nir_ine(&b, sh, dh));
            nir_def *y_row = nir_imul(&b, nir_iadd(&b, y, border), ds);
            nir_def *px = nir_iadd(&b, x, border);
            nir_def *prim_idx = nir_iadd(&b, y_row, px);
            nir_push_if(&b, do_res);
            {
               nir_def *sx =
                  bad_scale
                     ? nir_fmul(&b,
                                nir_fmul(&b, nir_u2f32(&b, x), nir_i2f32(&b, sw)),
                                nir_frcp(&b, nir_i2f32(&b, dw)))
                     : nir_fmul(&b, nir_u2f32(&b, x), sx_scale);
               nir_def *sy = nir_fmul(&b, nir_u2f32(&b, y), sy_scale);
               nir_def *fx = nir_fsub(&b, sx, nir_ftrunc(&b, sx));
               nir_def *fy = nir_fsub(&b, sy, nir_ftrunc(&b, sy));
               nir_def *ix = nir_f2i32(&b, sx);
               nir_def *iy = nir_f2i32(&b, sy);
               nir_def *x0 = nir_imin(&b, nir_imax(&b, ix, nir_imm_int(&b, 0)), nir_iadd_imm(&b, sw, -1));
               nir_def *x1 = nir_imin(&b, nir_imax(&b, nir_iadd_imm(&b, ix, 1), nir_imm_int(&b, 0)), nir_iadd_imm(&b, sw, -1));
               nir_def *y0 = nir_imin(&b, nir_imax(&b, iy, nir_imm_int(&b, 0)), nir_iadd_imm(&b, sh, -1));
               nir_def *y1 = nir_imin(&b, nir_imax(&b, nir_iadd_imm(&b, iy, 1), nir_imm_int(&b, 0)), nir_iadd_imm(&b, sh, -1));
               nir_def *p00 = nir_i2f32(&b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y0, ss), x0)))));
               nir_def *p01 = nir_i2f32(&b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y0, ss), x1)))));
               nir_def *p10 = nir_i2f32(&b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y1, ss), x0)))));
               nir_def *p11 = nir_i2f32(&b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y1, ss), x1)))));
               nir_def *v;
               if (use_flrp) {
                  nir_def *t = nir_flrp(&b, p00, p01, fx);
                  nir_def *bt = nir_flrp(&b, p10, p11, fx);
                  v = nir_flrp(&b, t, bt, fy);
               } else if (rev_add) {
                  nir_def *omfx = nir_fsub(&b, nir_imm_float(&b, 1.0f), fx);
                  nir_def *omfy = nir_fsub(&b, nir_imm_float(&b, 1.0f), fy);
                  nir_def *t = nir_fadd(&b, nir_fmul(&b, fx, p01), nir_fmul(&b, omfx, p00));
                  nir_def *bt = nir_fadd(&b, nir_fmul(&b, fx, p11), nir_fmul(&b, omfx, p10));
                  v = nir_fadd(&b, nir_fmul(&b, fy, bt), nir_fmul(&b, omfy, t));
               } else {
                  nir_def *omfx = nir_fsub(&b, nir_imm_float(&b, 1.0f), fx);
                  nir_def *omfy = nir_fsub(&b, nir_imm_float(&b, 1.0f), fy);
                  nir_def *t = nir_fadd(&b, nir_fmul(&b, omfx, p00), nir_fmul(&b, fx, p01));
                  nir_def *bt = nir_fadd(&b, nir_fmul(&b, omfx, p10), nir_fmul(&b, fx, p11));
                  v = nir_fadd(&b, nir_fmul(&b, omfy, t), nir_fmul(&b, fy, bt));
               }
               nir_def *v8 = nir_u2u8(&b, nir_f2i32(&b, nir_fmin(&b, nir_fmax(&b, v, nir_imm_float(&b, 0.0f)), nir_imm_float(&b, 255.0f))));
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, prim_idx), v8, 0x1);
            }
            nir_push_else(&b, NULL);
            {
               nir_def *v8 = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y, ss), x)));
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, prim_idx), v8, 0x1);
            }
            nir_pop_if(&b, NULL);

            nir_def *ridx = bad_reload ? nir_iadd_imm(&b, prim_idx, 1) : prim_idx;
            nir_def *val = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, dst, ridx));
            nir_def *x_lo = nir_iadd(&b, nir_iadd_imm(&b, dw, -2), nir_ineg(&b, border));
            nir_def *x_hi = nir_iadd_imm(&b, dw, -1);
            nir_def *y_lo = nir_iadd(&b, nir_iadd_imm(&b, dh, -2), nir_ineg(&b, border));
            nir_def *y_hi = nir_iadd_imm(&b, dh, -1);
            nir_def *xl_nz = nir_iand(&b, nir_ine_imm(&b, x, 0), nir_uge(&b, border, x));
            nir_def *yt_nz = nir_iand(&b, nir_ine_imm(&b, y, 0), nir_uge(&b, border, y));
            nir_def *xr = nir_iand(&b, nir_ult(&b, x, x_hi), nir_ult(&b, x_lo, x));
            nir_def *yb = nir_iand(&b, nir_ult(&b, y, y_hi), nir_ult(&b, y_lo, y));
            nir_def *ry_t = nir_imul(&b, nir_iadd(&b, border, nir_ineg(&b, y)), ds);
            nir_def *ry_b = nir_imul(&b, nir_iadd(&b, nir_iadd(&b, nir_iadd_imm(&b, nir_ishl_imm(&b, dh, 1), -2), border), nir_ineg(&b, y)), ds);

            nir_push_if(&b, xl_nz);
            {
               nir_def *rx_l = nir_iadd(&b, border, nir_ineg(&b, x));
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, y_row, rx_l)), val, 0x1);
               nir_push_if(&b, yt_nz);
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, ry_t, rx_l)), val, 0x1);
               nir_pop_if(&b, NULL);
               nir_push_if(&b, yb);
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, ry_b, rx_l)), val, 0x1);
               nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
            nir_push_if(&b, yt_nz);
            nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, ry_t, px)), val, 0x1);
            nir_pop_if(&b, NULL);
            nir_push_if(&b, xr);
            {
               nir_def *rx_r = nir_iadd(&b, nir_iadd(&b, nir_iadd_imm(&b, nir_ishl_imm(&b, dw, 1), -2), border), nir_ineg(&b, x));
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, y_row, rx_r)), val, 0x1);
               nir_push_if(&b, yt_nz);
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, ry_t, rx_r)), val, 0x1);
               nir_pop_if(&b, NULL);
               nir_push_if(&b, yb);
               nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, ry_b, rx_r)), val, 0x1);
               nir_pop_if(&b, NULL);
            }
            nir_pop_if(&b, NULL);
            nir_push_if(&b, yb);
            nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, nir_iadd(&b, ry_b, px)), val, 0x1);
            nir_pop_if(&b, NULL);

            nir_store_var(&b, xv, nir_iadd(&b, x, step_x), 0x1);
         }
         nir_pop_loop(&b, NULL);
         nir_store_var(&b, yv, nir_iadd(&b, y, step_y), 0x1);
      }
      nir_pop_loop(&b, NULL);

      nir_lower_vars_to_ssa(b.shader);
      nir_validate_shader(b.shader, "resize_prelower");
      struct hk_compute_opt_plan p3 = {0};
      bool matched = hk_nir_match_compute_opt(b.shader, 1, layouts, &p3);
      assert(matched == (variant == 0 || variant == 4));
      if (matched) {
         assert(p3.kind == HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8);
         assert(p3.flags == (rev_add ? HK_RESIZE_LERP_ONE_MINUS_ADD_B
                                     : HK_RESIZE_LERP_ONE_MINUS_ADD_A));
         nir_shader *opt3 = hk_nir_build_compute_opt(
            NULL, b.shader, &p3, nir_address_format_64bit_bounded_global);
         assert(opt3 != NULL);
         bool saw_fadd = false, saw_ffma = false, saw_fmin = false, saw_fmax = false;
         bool saw_a_first = false, saw_b_first = false;
         nir_foreach_block(blk, nir_shader_get_entrypoint(opt3)) {
            nir_foreach_instr(ins, blk) {
               if (ins->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(ins);
               switch (alu->op) {
               case nir_op_fadd: {
                  saw_fadd = true;
                  nir_instr *s0 = nir_def_instr(alu->src[0].src.ssa);
                  nir_instr *s1 = nir_def_instr(alu->src[1].src.ssa);
                  if (s0->type == nir_instr_type_alu &&
                      s1->type == nir_instr_type_alu &&
                      nir_instr_as_alu(s0)->op == nir_op_fmul &&
                      nir_instr_as_alu(s1)->op == nir_op_fmul) {
                     nir_alu_instr *m0 = nir_instr_as_alu(s0);
                     bool s0_has_omt = false;
                     for (unsigned k = 0; k < 2; k++) {
                        nir_instr *mi = nir_def_instr(m0->src[k].src.ssa);
                        if (mi->type != nir_instr_type_alu)
                           continue;
                        nir_alu_instr *ma = nir_instr_as_alu(mi);
                        if ((ma->op == nir_op_fsub || ma->op == nir_op_fadd) &&
                            ((nir_src_is_const(ma->src[0].src) &&
                              nir_src_as_float(ma->src[0].src) == 1.0f) ||
                             (ma->op == nir_op_fadd &&
                              nir_src_is_const(ma->src[1].src) &&
                              nir_src_as_float(ma->src[1].src) == 1.0f)))
                           s0_has_omt = true;
                     }
                     if (s0_has_omt)
                        saw_a_first = true;
                     else
                        saw_b_first = true;
                  }
                  break;
               }
               case nir_op_ffma: saw_ffma = true; break;
               case nir_op_fmin: saw_fmin = true; break;
               case nir_op_fmax: saw_fmax = true; break;
               default: break;
               }
            }
         }
         assert(saw_fadd && !saw_ffma && saw_fmin && saw_fmax);
         assert(saw_a_first == !rev_add && saw_b_first == rev_add);
         ralloc_free(opt3);
      }
      ralloc_free(b.shader);
   }

   /* Also verify un-unswitched in-loop `if (do_resize)` form. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &test_nir_options, "resize_inloop");
      b.shader->info.internal = false;
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 8;
      b.shader->info.workgroup_size[2] = 1;
      b.shader->info.num_ssbos = 2;

      nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
      nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);
      nir_def *sw = make_pc_i32(&b, 0);
      nir_def *sh = make_pc_i32(&b, 4);
      nir_def *ss = make_pc_i32(&b, 8);
      nir_def *dw = make_pc_i32(&b, 12);
      nir_def *dh = make_pc_i32(&b, 16);
      nir_def *ds = make_pc_i32(&b, 20);
      nir_def *border = make_pc_i32(&b, 24);
      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_def *bwg = nir_load_base_workgroup_id(&b, 32);
      nir_def *nwg = nir_load_num_workgroups(&b);
      nir_def *gx0 = nir_iadd(&b, nir_channel(&b, gid, 0),
                              nir_ishl_imm(&b, nir_channel(&b, bwg, 0), 5));
      nir_def *gy0 = nir_iadd(&b, nir_channel(&b, gid, 1),
                              nir_ishl_imm(&b, nir_channel(&b, bwg, 1), 3));
      nir_def *step_x = nir_ishl_imm(&b, nir_channel(&b, nwg, 0), 5);
      nir_def *step_y = nir_ishl_imm(&b, nir_channel(&b, nwg, 1), 3);
      nir_def *sx_scale = nir_fmul(&b, nir_i2f32(&b, sw), nir_frcp(&b, nir_i2f32(&b, dw)));
      nir_def *sy_scale = nir_fmul(&b, nir_i2f32(&b, sh), nir_frcp(&b, nir_i2f32(&b, dh)));
      nir_def *do_res = nir_ior(&b, nir_ine(&b, sw, dw), nir_ine(&b, sh, dh));

      nir_variable *yv = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_store_var(&b, yv, gy0, 0x1);
      nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, yv);
         nir_push_if(&b, nir_uge(&b, y, dh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
         nir_variable *xv = nir_local_variable_create(b.impl, glsl_int_type(), "x");
         nir_store_var(&b, xv, gx0, 0x1);
         nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, xv);
            nir_push_if(&b, nir_uge(&b, x, dw));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);
            nir_variable *vv = nir_local_variable_create(b.impl, glsl_uint8_t_type(), "v");
            nir_push_if(&b, do_res);
            {
               nir_def *sy = nir_fmul(&b, nir_u2f32(&b, y), sy_scale);
               nir_def *fy = nir_fsub(&b, sy, nir_ftrunc(&b, sy));
               nir_def *iy = nir_f2i32(&b, sy);
               nir_def *y0 = nir_imin(&b, nir_imax(&b, iy, nir_imm_int(&b, 0)), nir_iadd_imm(&b, sh, -1));
               nir_def *y1 = nir_imin(&b, nir_imax(&b, nir_iadd_imm(&b, iy, 1), nir_imm_int(&b, 0)), nir_iadd_imm(&b, sh, -1));
               nir_def *sx = nir_fmul(&b, nir_u2f32(&b, x), sx_scale);
               nir_def *fx = nir_fsub(&b, sx, nir_ftrunc(&b, sx));
               nir_def *ix = nir_f2i32(&b, sx);
               nir_def *x0 = nir_imin(&b, nir_imax(&b, ix, nir_imm_int(&b, 0)), nir_iadd_imm(&b, sw, -1));
               nir_def *x1 = nir_imin(&b, nir_imax(&b, nir_iadd_imm(&b, ix, 1), nir_imm_int(&b, 0)), nir_iadd_imm(&b, sw, -1));
               nir_def *p00 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y0, ss), x0))));
               nir_def *p01 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y0, ss), x1))));
               nir_def *p10 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y1, ss), x0))));
               nir_def *p11 = nir_u2f32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y1, ss), x1))));
               nir_def *t = nir_ffma(&b, fx, nir_fsub(&b, p01, p00), p00);
               nir_def *bt = nir_ffma(&b, fx, nir_fsub(&b, p11, p10), p10);
               nir_def *v = nir_ffma(&b, fy, nir_fsub(&b, bt, t), t);
               nir_def *v8 = nir_u2u8(&b, nir_f2i32(&b, nir_fmin(&b, nir_fmax(&b, v, nir_imm_float(&b, 0.0f)), nir_imm_float(&b, 255.0f))));
               nir_store_var(&b, vv, v8, 0x1);
            }
            nir_push_else(&b, NULL);
            {
               nir_def *v8 = nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, nir_iadd(&b, nir_imul(&b, y, ss), x)));
               nir_store_var(&b, vv, v8, 0x1);
            }
            nir_pop_if(&b, NULL);
            emit_test_mirror_border(&b, dst, x, y, nir_load_var(&b, vv), dw, dh, ds, border);
            nir_store_var(&b, xv, nir_iadd(&b, x, step_x), 0x1);
         }
         nir_pop_loop(&b, NULL);
         nir_store_var(&b, yv, nir_iadd(&b, y, step_y), 0x1);
      }
      nir_pop_loop(&b, NULL);

      nir_lower_vars_to_ssa(b.shader);
      nir_validate_shader(b.shader, "resize_inloop");
      struct hk_compute_opt_plan p2 = {0};
      assert(hk_nir_match_compute_opt(b.shader, 1, layouts, &p2));
      assert(p2.kind == HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8);
      ralloc_free(b.shader);
   }

   /* Exact pixel-compare between scalar reference (modelling the exact matched
    * lerp algebra + [0, 255] clamp) and vectorized 128x8 quad_int + fast_u8x2 +
    * mirrored border semantics across copy, downscale (including 640->480),
    * upscale, non-multiple-of-4 widths, borders (0, 1, 3, 8, 19), and both
    * ONE_MINUS_ADD (GB6) and DIFF_FFMA lerp modes. */
   {
      /* Verify the host model distinguishes ONE_MINUS_ADD vs DIFF_FFMA on the
       * 640->480 x=1 taps (5, 2) counterexample (3 vs 4). */
      volatile float c_rcp = 1.0f / (float)480;
      volatile float c_sc = (float)640 * c_rcp;
      volatile float c_sx = 1.0f * c_sc;
      volatile float c_fx = c_sx - truncf(c_sx);
      volatile float c_om = 1.0f - c_fx;
      volatile float c_m0 = c_om * 5.0f;
      volatile float c_m1 = c_fx * 2.0f;
      volatile float c_add = c_m0 + c_m1;
      volatile float c_fma = fmaf(c_fx, 2.0f - 5.0f, 5.0f);
      assert((uint8_t)(int32_t)fminf(fmaxf(c_add, 0.0f), 255.0f) == 3u);
      assert((uint8_t)(int32_t)fminf(fmaxf(c_fma, 0.0f), 255.0f) == 4u);

      struct {
         uint32_t sw, sh, ss, dw, dh, ds, border;
      } cases[] = {
         {64, 48, 64, 64, 48, 70, 3},
         {137, 93, 144, 137, 93, 180, 19},
         {137, 93, 144, 114, 77, 156, 19},
         {640, 480, 640, 480, 360, 496, 8},
         {79, 53, 80, 66, 44, 66, 0},
         {31, 29, 32, 47, 41, 51, 1},
         {100, 60, 100, 29, 17, 37, 3},
      };
      for (unsigned lm = 0; lm < 2; lm++) {
         const bool one_minus_add = (lm == 0);
         for (unsigned ci = 0; ci < ARRAY_SIZE(cases); ci++) {
            const uint32_t sw = cases[ci].sw, sh = cases[ci].sh, ss = cases[ci].ss;
            const uint32_t dw = cases[ci].dw, dh = cases[ci].dh, ds = cases[ci].ds;
            const uint32_t bd = cases[ci].border;
            const uint32_t dh_pad = dh + 2u * bd;
            uint8_t *src = calloc(sh * ss, 1);
            uint8_t *ref = calloc(dh_pad * ds, 1);
            uint8_t *vec = calloc(dh_pad * ds, 1);
            assert(src && ref && vec);
            memset(ref, 0xA5, dh_pad * ds);
            memset(vec, 0xA5, dh_pad * ds);
            for (uint32_t y = 0; y < sh; y++)
               for (uint32_t x = 0; x < sw; x++)
                  src[y * ss + x] = (uint8_t)((x * 37u + y * 131u + ci * 17u) & 0xffu);

            const bool do_res = (sw != dw) || (sh != dh);
            const float sx_sc = (float)(int32_t)sw * (1.0f / (float)(int32_t)dw);
            const float sy_sc = (float)(int32_t)sh * (1.0f / (float)(int32_t)dh);

#define EVAL_MATCHED_BILERP(out_var, P00, P01, P10, P11, FX, FY)               \
   do {                                                                        \
      float _v;                                                                \
      if (one_minus_add) {                                                     \
         volatile float _omx = 1.0f - (FX);                                    \
         volatile float _omy = 1.0f - (FY);                                    \
         volatile float _t0 = _omx * (P00), _t1 = (FX) * (P01);                \
         volatile float _t = _t0 + _t1;                                        \
         volatile float _b0 = _omx * (P10), _b1 = (FX) * (P11);                \
         volatile float _bt = _b0 + _b1;                                       \
         volatile float _v0 = _omy * _t, _v1 = (FY) * _bt;                     \
         _v = _v0 + _v1;                                                       \
      } else {                                                                 \
         float _t = fmaf((FX), (P01) - (P00), (P00));                          \
         float _bt = fmaf((FX), (P11) - (P10), (P10));                         \
         _v = fmaf((FY), _bt - _t, _t);                                        \
      }                                                                        \
      _v = fminf(fmaxf(_v, 0.0f), 255.0f);                                     \
      (out_var) = (uint8_t)(int32_t)_v;                                        \
   } while (0)

            /* Scalar reference with full mirrored border writes. */
            for (uint32_t y = 0; y < dh; y++) {
               float sy = (float)y * sy_sc;
               float fy = sy - truncf(sy);
               int32_t iy = (int32_t)sy;
               int32_t y0 = iy < 0 ? 0 : (iy > (int32_t)sh - 1 ? (int32_t)sh - 1 : iy);
               int32_t y1 = iy + 1 < 0 ? 0 : (iy + 1 > (int32_t)sh - 1 ? (int32_t)sh - 1 : iy + 1);
               for (uint32_t x = 0; x < dw; x++) {
                  uint8_t v8;
                  if (do_res) {
                     float sx = (float)x * sx_sc;
                     float fx = sx - truncf(sx);
                     int32_t ix = (int32_t)sx;
                     int32_t x0 = ix < 0 ? 0 : (ix > (int32_t)sw - 1 ? (int32_t)sw - 1 : ix);
                     int32_t x1 = ix + 1 < 0 ? 0 : (ix + 1 > (int32_t)sw - 1 ? (int32_t)sw - 1 : ix + 1);
                     float p00 = (float)src[(uint32_t)y0 * ss + (uint32_t)x0];
                     float p01 = (float)src[(uint32_t)y0 * ss + (uint32_t)x1];
                     float p10 = (float)src[(uint32_t)y1 * ss + (uint32_t)x0];
                     float p11 = (float)src[(uint32_t)y1 * ss + (uint32_t)x1];
                     EVAL_MATCHED_BILERP(v8, p00, p01, p10, p11, fx, fy);
                  } else {
                     v8 = src[y * ss + x];
                  }
                  uint32_t yrow = (y + bd) * ds, px = x + bd;
                  ref[yrow + px] = v8;
                  bool xl = (x != 0 && x <= bd);
                  bool xr = (x + bd + 2u > dw && x + 1u < dw);
                  bool yt = (y != 0 && y <= bd);
                  bool yb = (y + bd + 2u > dh && y + 1u < dh);
                  uint32_t ryl = (bd - x), ryr = (2u * dw + bd - 2u - x);
                  uint32_t ryt = (bd - y) * ds, ryb = (2u * dh + bd - 2u - y) * ds;
                  if (xl) {
                     ref[yrow + ryl] = v8;
                     if (yt) ref[ryt + ryl] = v8;
                     if (yb) ref[ryb + ryl] = v8;
                  }
                  if (yt) ref[ryt + px] = v8;
                  if (xr) {
                     ref[yrow + ryr] = v8;
                     if (yt) ref[ryt + ryr] = v8;
                     if (yb) ref[ryb + ryr] = v8;
                  }
                  if (yb) ref[ryb + px] = v8;
               }
            }

            /* Vectorized 4-pixel-per-lane simulation matching emitted NIR. */
            uint32_t x_lim_lo = dw - 2u - bd;
            uint32_t y_lim_lo = dh - 2u - bd;
            for (uint32_t y = 0; y < dh; y++) {
               float sy = (float)y * sy_sc;
               float fy = sy - truncf(sy);
               int32_t iy = (int32_t)sy;
               int32_t y0 = iy < 0 ? 0 : (iy > (int32_t)sh - 1 ? (int32_t)sh - 1 : iy);
               int32_t y1 = iy + 1 < 0 ? 0 : (iy + 1 > (int32_t)sh - 1 ? (int32_t)sh - 1 : iy + 1);
               for (uint32_t x0 = 0; x0 < dw; x0 += 4u) {
                  bool qint = (bd < y && y <= y_lim_lo && bd < x0 && x0 + 3u <= x_lim_lo);
                  if (qint) {
                     uint8_t out[4];
                     if (do_res) {
                        uint32_t ix3 = (uint32_t)(int32_t)((float)(x0 + 3u) * sx_sc);
                        bool fast = ix3 < sw - 1u;
                        for (uint32_t k = 0; k < 4u; k++) {
                           float sx = (float)(x0 + k) * sx_sc;
                           float fx = sx - truncf(sx);
                           int32_t ix = (int32_t)sx;
                           uint32_t c0 = fast ? (uint32_t)ix : (uint32_t)(ix < 0 ? 0 : (ix > (int32_t)sw - 1 ? (int32_t)sw - 1 : ix));
                           uint32_t c1 = fast ? (uint32_t)(ix + 1) : (uint32_t)(ix + 1 < 0 ? 0 : (ix + 1 > (int32_t)sw - 1 ? (int32_t)sw - 1 : ix + 1));
                           float p00 = (float)src[(uint32_t)y0 * ss + c0];
                           float p01 = (float)src[(uint32_t)y0 * ss + c1];
                           float p10 = (float)src[(uint32_t)y1 * ss + c0];
                           float p11 = (float)src[(uint32_t)y1 * ss + c1];
                           EVAL_MATCHED_BILERP(out[k], p00, p01, p10, p11, fx, fy);
                        }
                     } else {
                        memcpy(out, &src[y * ss + x0], 4);
                     }
                     memcpy(&vec[(y + bd) * ds + (x0 + bd)], out, 4);
                  } else {
                     for (uint32_t k = 0; k < 4u; k++) {
                        uint32_t x = x0 + k;
                        if (x >= dw)
                           continue;
                        uint8_t v8;
                        if (do_res) {
                           float sx = (float)x * sx_sc;
                           float fx = sx - truncf(sx);
                           int32_t ix = (int32_t)sx;
                           int32_t c0 = ix < 0 ? 0 : (ix > (int32_t)sw - 1 ? (int32_t)sw - 1 : ix);
                           int32_t c1 = ix + 1 < 0 ? 0 : (ix + 1 > (int32_t)sw - 1 ? (int32_t)sw - 1 : ix + 1);
                           float p00 = (float)src[(uint32_t)y0 * ss + (uint32_t)c0];
                           float p01 = (float)src[(uint32_t)y0 * ss + (uint32_t)c1];
                           float p10 = (float)src[(uint32_t)y1 * ss + (uint32_t)c0];
                           float p11 = (float)src[(uint32_t)y1 * ss + (uint32_t)c1];
                           EVAL_MATCHED_BILERP(v8, p00, p01, p10, p11, fx, fy);
                        } else {
                           v8 = src[y * ss + x];
                        }
                        uint32_t yrow = (y + bd) * ds, px = x + bd;
                        vec[yrow + px] = v8;
                        bool xl = (x != 0 && x <= bd);
                        bool xr = (x + bd + 2u > dw && x + 1u < dw);
                        bool yt = (y != 0 && y <= bd);
                        bool yb = (y + bd + 2u > dh && y + 1u < dh);
                        uint32_t ryl = (bd - x), ryr = (2u * dw + bd - 2u - x);
                        uint32_t ryt = (bd - y) * ds, ryb = (2u * dh + bd - 2u - y) * ds;
                        if (xl) {
                           vec[yrow + ryl] = v8;
                           if (yt) vec[ryt + ryl] = v8;
                           if (yb) vec[ryb + ryl] = v8;
                        }
                        if (yt) vec[ryt + px] = v8;
                        if (xr) {
                           vec[yrow + ryr] = v8;
                           if (yt) vec[ryt + ryr] = v8;
                           if (yb) vec[ryb + ryr] = v8;
                        }
                        if (yb) vec[ryb + px] = v8;
                     }
                  }
               }
            }
#undef EVAL_MATCHED_BILERP
            assert(memcmp(ref, vec, dh_pad * ds) == 0);
            free(src);
            free(ref);
            free(vec);
         }
      }
   }
   free(layout);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_rgba8_to_gray();
   test_canny_hysteresis();
   test_canny_nms();
   test_canny_finalize();
   test_bilinear_resize_border();
   glsl_type_singleton_decref();
   printf("hk_compute_opt: all unit tests passed\n");
   return 0;
}
