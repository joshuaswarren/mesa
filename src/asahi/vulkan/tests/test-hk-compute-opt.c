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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define test_nir_options agx_nir_options

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
   assert(hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                1, 64, 1, &gx, &gy));
   assert(gx == 3 && gy == 16);
   /* Reject alias_free == false, non-zero base, and overlapping VA ranges. */
   assert(!hk_compute_opt_validate_dispatch_core(false, &desc, &plan, 0, 0, 0,
                                                 1, 64, 1, &gx, &gy));
   assert(!hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 1, 0, 0,
                                                 1, 64, 1, &gx, &gy));
   addrs[1].base_addr = 0x100100;
   assert(!hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                 1, 64, 1, &gx, &gy));
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
   assert(hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                3, 17, 1, &gx, &gy));
   assert(gx == 2 && gy == 5);
   /* Reject in-place aliasing (src == dst). */
   addrs[1].base_addr = addrs[0].base_addr;
   assert(!hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                 3, 17, 1, &gx, &gy));
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
   assert(hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                1, 480, 1, &gx, &gy));
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
   assert(hk_compute_opt_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                1, 200, 1, &gx, &gy));
   assert(gx == 2 && gy == 50);
   ralloc_free(s);

   nir_shader *bad = build_test_canny_finalize(true);
   assert(!hk_nir_match_compute_opt(bad, 1, layouts, &plan));
   ralloc_free(bad);
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
   glsl_type_singleton_decref();
   printf("hk_compute_opt: all unit tests passed\n");
   return 0;
}
