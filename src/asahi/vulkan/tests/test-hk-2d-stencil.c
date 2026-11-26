/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_2d_stencil.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"

#include "asahi/compiler/agx_compile.h"
#include "compiler/glsl_types.h"
#include "nir_builder.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const nir_shader_compiler_options test_nir_options = {0};

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
   memset(layout->vk.blake3, 0x57, sizeof(layout->vk.blake3));
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

enum box_variant {
   BV_NONE,
   BV_SWAP_LOOPS,      /* outer X, inner Y */
   BV_EARLY_RETURN,    /* if (x >= w || y >= h) return; */
   BV_UDIV,            /* udiv instead of idiv */
   BV_WITH_BASE_WG,    /* global_id + base_workgroup_id * wg */
   BV_EXTRA_STORE,     /* negative: extra store to dst[0] */
   BV_WRONG_DIVISOR,   /* negative: divides by 2*r+1 instead of (2*r+1)^2 */
   BV_NO_BORDER_COPY,  /* negative: border writes 0 instead of src[idx] */
   BV_SAME_BINDING,    /* negative: src and dst share binding 0 */
   BV_VOLATILE_LOAD,   /* negative: volatile load */
   BV_COUNT,
};

static nir_shader *
build_box_blur_shader(enum box_variant v, uint32_t wg_x, uint32_t wg_y)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_box_blur");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg_x;
   b.shader->info.workgroup_size[1] = wg_y;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst = make_ssbo_base(
      &b, v == BV_SAME_BINDING ? 0 : 1, glsl_uint8_t_type(), 1);

   nir_def *w =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 4);
   nir_def *h =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 4, .range = 4);
   nir_def *stride =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 8, .range = 4);
   nir_def *r =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 12, .range = 4);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   if (v == BV_WITH_BASE_WG) {
      nir_def *bwg = nir_load_base_workgroup_id(&b, 32);
      x = nir_iadd(&b, x, nir_imul_imm(&b, nir_channel(&b, bwg, 0), wg_x));
      y = nir_iadd(&b, y, nir_imul_imm(&b, nir_channel(&b, bwg, 1), wg_y));
   }

   if (v == BV_EARLY_RETURN) {
      nir_push_if(&b, nir_ior(&b, nir_ige(&b, x, w), nir_ige(&b, y, h)));
      nir_jump(&b, nir_jump_return);
      nir_pop_if(&b, NULL);
   } else {
      nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
   }

   nir_def *idx = nir_iadd(&b, nir_imul(&b, y, stride), x);
   nir_def *is_int = nir_iand(
      &b, nir_iand(&b, nir_ige(&b, x, r), nir_ilt(&b, x, nir_isub(&b, w, r))),
      nir_iand(&b, nir_ige(&b, y, r), nir_ilt(&b, y, nir_isub(&b, h, r))));

   nir_push_if(&b, is_int);
   {
      nir_variable *sum_var =
         nir_local_variable_create(b.impl, glsl_uint_type(), "sum");
      nir_variable *o_var =
         nir_local_variable_create(b.impl, glsl_int_type(), "o_iv");
      nir_variable *i_var =
         nir_local_variable_create(b.impl, glsl_int_type(), "i_iv");
      nir_store_var(&b, sum_var, nir_imm_int(&b, 0), 0x1);

      nir_def *o_c = (v == BV_SWAP_LOOPS) ? x : y;
      nir_def *i_c = (v == BV_SWAP_LOOPS) ? y : x;
      nir_store_var(&b, o_var, nir_isub(&b, o_c, r), 0x1);
      nir_push_loop(&b);
      {
         nir_def *ov = nir_load_var(&b, o_var);
         nir_push_if(&b, nir_ilt(&b, nir_iadd(&b, o_c, r), ov));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_store_var(&b, i_var, nir_isub(&b, i_c, r), 0x1);
         nir_push_loop(&b);
         {
            nir_def *iv = nir_load_var(&b, i_var);
            nir_push_if(&b, nir_ilt(&b, nir_iadd(&b, i_c, r), iv));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);

            nir_def *yy = (v == BV_SWAP_LOOPS) ? iv : ov;
            nir_def *xx = (v == BV_SWAP_LOOPS) ? ov : iv;
            nir_def *off = nir_iadd(&b, nir_imul(&b, yy, stride), xx);
            nir_deref_instr *e = nir_build_deref_ptr_as_array(&b, src, off);
            enum gl_access_qualifier acc =
               (v == BV_VOLATILE_LOAD) ? ACCESS_VOLATILE : 0;
            nir_def *px = nir_load_deref_with_access(&b, e, acc);
            nir_store_var(
               &b, sum_var,
               nir_iadd(&b, nir_load_var(&b, sum_var), nir_u2u32(&b, px)), 0x1);
            nir_store_var(&b, i_var, nir_iadd_imm(&b, iv, 1), 0x1);
         }
         nir_pop_loop(&b, NULL);

         nir_store_var(&b, o_var, nir_iadd_imm(&b, ov, 1), 0x1);
      }
      nir_pop_loop(&b, NULL);

      nir_def *diam = nir_iadd_imm(&b, nir_ishl_imm(&b, r, 1), 1);
      nir_def *denom =
         (v == BV_WRONG_DIVISOR) ? diam : nir_imul(&b, diam, diam);
      nir_def *sum = nir_load_var(&b, sum_var);
      nir_def *q = (v == BV_UDIV) ? nir_udiv(&b, sum, denom)
                                  : nir_idiv(&b, sum, denom);
      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, idx),
                      nir_u2u8(&b, q), 0x1);
      if (v == BV_EXTRA_STORE) {
         nir_store_deref(&b,
                         nir_build_deref_ptr_as_array(&b, dst, nir_imm_int(&b, 0)),
                         nir_imm_intN_t(&b, 7, 8), 0x1);
      }
   }
   nir_push_else(&b, NULL);
   {
      nir_def *bval =
         (v == BV_NO_BORDER_COPY)
            ? nir_imm_intN_t(&b, 0, 8)
            : nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, idx));
      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, idx), bval, 0x1);
   }
   nir_pop_if(&b, NULL);

   if (v != BV_EARLY_RETURN)
      nir_pop_if(&b, NULL);

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "build_box_blur_shader");
   return b.shader;
}

static void
test_box_blur_matcher_and_builder(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(2, false);
   struct hk_descriptor_set_layout *uab_layout = make_test_layout(2, true);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};
   struct vk_descriptor_set_layout *uab_layouts[1] = {&uab_layout->vk};

   for (unsigned v = 0; v < BV_COUNT; v++) {
      nir_shader *s = build_box_blur_shader((enum box_variant)v, 32, 8);
      struct hk_2d_stencil_plan plan = {0};
      bool expect = (v <= BV_WITH_BASE_WG);
      bool got = hk_nir_match_2d_stencil(s, 1, layouts, &plan);
      if (got != expect)
         fprintf(stderr, "box variant %u failed: got=%d expect=%d\n", v, got,
                 expect);
      assert(got == expect);
      if (expect) {
         assert(plan.kind == HK_2D_STENCIL_BOX_BLUR_U8);
         assert(plan.tile_w == HK_BOX_BLUR_STRIP_W &&
                plan.tile_h == HK_BOX_BLUR_TILE_H);
         assert(plan.width_pc_offset_B == 0);
         assert(plan.height_pc_offset_B == 4);
         assert(plan.stride_pc_offset_B == 8);
         assert(plan.radius_pc_offset_B == 12);
         assert(plan.bufs[0].binding == 0 && plan.bufs[1].binding == 1);

         nir_shader *tiled = hk_nir_build_tiled_2d_stencil(
            NULL, s, &plan, nir_address_format_64bit_bounded_global);
         assert(tiled != NULL);
         assert(tiled->info.workgroup_size[0] == 256);
         assert(tiled->info.workgroup_size[1] == 1);
         ralloc_free(tiled);
      }
      ralloc_free(s);
   }

   /* Test AGX-preprocessed NIR (`agx_preprocess_nir_native_defer_idiv`). */
   {
      nir_shader *s = build_box_blur_shader(BV_WITH_BASE_WG, 32, 8);
      agx_preprocess_nir_native_defer_idiv(s, AGX_CHIP_G17P);
      struct hk_2d_stencil_plan plan = {0};
      assert(hk_nir_match_2d_stencil(s, 1, layouts, &plan));
      assert(plan.kind == HK_2D_STENCIL_BOX_BLUR_U8);
      nir_shader *tiled = hk_nir_build_tiled_2d_stencil(
         NULL, s, &plan, nir_address_format_64bit_bounded_global);
      assert(tiled != NULL);
      ralloc_free(tiled);
      ralloc_free(s);
   }

   /* Reject UPDATE_AFTER_BIND descriptor sets. */
   {
      nir_shader *s = build_box_blur_shader(BV_NONE, 32, 8);
      struct hk_2d_stencil_plan plan = {0};
      assert(!hk_nir_match_2d_stencil(s, 1, uab_layouts, &plan));
      ralloc_free(s);
   }

   /* Dispatch validator checks. */
   {
      nir_shader *s = build_box_blur_shader(BV_NONE, 32, 8);
      struct hk_2d_stencil_plan plan = {0};
      assert(hk_nir_match_2d_stencil(s, 1, layouts, &plan));

      struct hk_buffer_address bufs[2] = {
         {.base_addr = 0x100000, .size = 4096 * 2048},
         {.base_addr = 0x2000000, .size = 4096 * 2048},
      };
      struct hk_descriptor_set ds = {
         .layout = layout,
         .mapped_ptr = bufs,
         .size = sizeof(bufs),
      };
      struct hk_descriptor_state desc = {0};
      desc.sets[0] = &ds;
      int32_t pc[4] = {1920, 1080, 1920, 3};
      memcpy(desc.root.push, pc, sizeof(pc));
      desc.push_written_dwords = 0xf;

      uint32_t gx = 0, gy = 0;
      assert(hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                  60, 135, 1, &gx, &gy));
      /* r = 3: strips start 4 columns left of their outputs and yield 120
       * columns; 128-row tiles. */
      assert(hk_box_blur_halo(3) == 4 && hk_box_blur_out_w(3) == 120);
      assert(gx == 16 && gy == 9);

      /* Every radius: the strip starts a multiple of 4 columns and at least
       * r + 1 columns left of its first output, the last output's window
       * ends inside the 128-column strip, and one more output lane would
       * not fit. */
      for (uint32_t r = 0; r <= HK_2D_STENCIL_MAX_RADIUS; r++) {
         const uint32_t halo = hk_box_blur_halo(r);
         const uint32_t ow = hk_box_blur_out_w(r);
         assert(halo % 4 == 0 && halo >= r + 1 && halo < r + 5);
         assert(ow % 4 == 0 && ow >= 4);
         assert(ow - 1 + halo + r <= HK_BOX_BLUR_STRIP_W - 1);
         assert(ow + 3 + halo + r > HK_BOX_BLUR_STRIP_W - 1);
      }

      /* Narrower than one u8x4 read -> reject (the original shader runs). */
      pc[0] = 3;
      pc[2] = 3;
      memcpy(desc.root.push, pc, sizeof(pc));
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   60, 135, 1, &gx, &gy));
      pc[0] = 1920;
      pc[2] = 1920;
      memcpy(desc.root.push, pc, sizeof(pc));

      /* Aliasing active -> reject. */
      assert(!hk_2d_stencil_validate_dispatch_core(false, &desc, &plan, 0, 0, 0,
                                                   60, 135, 1, &gx, &gy));
      /* Overlapping VA ranges -> reject. */
      bufs[1].base_addr = 0x100100;
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   60, 135, 1, &gx, &gy));
      bufs[1].base_addr = 0x2000000;
      /* Radius > MAX -> reject. */
      pc[3] = HK_2D_STENCIL_MAX_RADIUS + 1;
      memcpy(desc.root.push, pc, sizeof(pc));
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   60, 135, 1, &gx, &gy));
      /* Radius < 0 -> reject. */
      pc[3] = -1;
      memcpy(desc.root.push, pc, sizeof(pc));
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   60, 135, 1, &gx, &gy));
      /* Grid too small -> reject. */
      pc[3] = 3;
      memcpy(desc.root.push, pc, sizeof(pc));
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   59, 135, 1, &gx, &gy));
      ralloc_free(s);
   }

   free(layout);
   free(uab_layout);
   printf("hk_2d_stencil: box blur matcher/builder/validator tests passed\n");
}

enum stereo_variant {
   SV_NONE,
   SV_WITH_BASE_WG,
   SV_SWAP_LOOPS,
   SV_EXTRA_STORE,     /* negative */
   SV_WRONG_SHIFT,     /* negative: uses +d instead of -d */
   SV_NO_ABS,          /* negative: omits abs() */
   SV_SAME_OUT_IN,     /* negative: dst_disp aliases left */
   SV_COUNT,
};

static nir_shader *
build_stereo_sad_shader(enum stereo_variant v, uint32_t wg_x, uint32_t wg_y)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_stereo_sad");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg_x;
   b.shader->info.workgroup_size[1] = wg_y;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 6;

   nir_deref_instr *left = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *right = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);
   nir_deref_instr *pdisp = make_ssbo_base(&b, 2, glsl_uint8_t_type(), 1);
   nir_deref_instr *pcost = make_ssbo_base(&b, 3, glsl_uint_type(), 4);
   nir_deref_instr *ddisp =
      make_ssbo_base(&b, v == SV_SAME_OUT_IN ? 0 : 4, glsl_uint8_t_type(), 1);
   nir_deref_instr *dcost = make_ssbo_base(&b, 5, glsl_uint_type(), 4);

   nir_def *w =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 4);
   nir_def *h =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 4, .range = 4);
   nir_def *stride =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 8, .range = 4);
   nir_def *d_min =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 12, .range = 4);
   nir_def *d_max =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 16, .range = 4);
   nir_def *ws =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 20, .range = 4);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   if (v == SV_WITH_BASE_WG) {
      nir_def *bwg = nir_load_base_workgroup_id(&b, 32);
      x = nir_iadd(&b, x, nir_imul_imm(&b, nir_channel(&b, bwg, 0), wg_x));
      y = nir_iadd(&b, y, nir_imul_imm(&b, nir_channel(&b, bwg, 1), wg_y));
   }

   nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
   {
      nir_def *cidx = nir_iadd(&b, nir_imul(&b, y, w), x);
      nir_def *didx = nir_iadd(&b, nir_imul(&b, y, stride), x);
      nir_def *init_cost =
         nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, pcost, cidx));
      nir_def *init_disp = nir_u2u32(
         &b, nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, pdisp, didx)));

      nir_variable *bc_v =
         nir_local_variable_create(b.impl, glsl_int_type(), "bc");
      nir_variable *bd_v =
         nir_local_variable_create(b.impl, glsl_uint_type(), "bd");
      nir_variable *d_v =
         nir_local_variable_create(b.impl, glsl_int_type(), "d");
      nir_store_var(&b, bc_v, init_cost, 0x1);
      nir_store_var(&b, bd_v, init_disp, 0x1);
      nir_store_var(&b, d_v, d_min, 0x1);

      nir_def *r = nir_idiv(&b, ws, nir_imm_int(&b, 2));
      nir_def *wm1 = nir_iadd_imm(&b, w, -1);
      nir_def *hm1 = nir_iadd_imm(&b, h, -1);
      nir_def *x0 =
         nir_imin(&b, nir_imax(&b, nir_isub(&b, x, r), nir_imm_int(&b, 0)), wm1);
      nir_def *x1 =
         nir_imin(&b, nir_imax(&b, nir_iadd(&b, x, r), nir_imm_int(&b, 0)), wm1);
      nir_def *y0 =
         nir_imin(&b, nir_imax(&b, nir_isub(&b, y, r), nir_imm_int(&b, 0)), hm1);
      nir_def *y1 =
         nir_imin(&b, nir_imax(&b, nir_iadd(&b, y, r), nir_imm_int(&b, 0)), hm1);

      nir_push_loop(&b);
      {
         nir_def *d = nir_load_var(&b, d_v);
         nir_push_if(&b, nir_ilt(&b, d_max, d));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *d_ok = nir_iand(&b, nir_ige_imm(&b, nir_isub(&b, x0, d), 0),
                                  nir_ilt(&b, nir_isub(&b, x1, d), w));
         nir_push_if(&b, d_ok);
         {
            nir_variable *sad_v =
               nir_local_variable_create(b.impl, glsl_int_type(), "sad");
            nir_variable *o_v =
               nir_local_variable_create(b.impl, glsl_int_type(), "wo");
            nir_variable *i_v =
               nir_local_variable_create(b.impl, glsl_int_type(), "wi");
            nir_store_var(&b, sad_v, nir_imm_int(&b, 0), 0x1);
            nir_store_var(&b, o_v, (v == SV_SWAP_LOOPS) ? x0 : y0, 0x1);
            nir_push_loop(&b);
            {
               nir_def *ov = nir_load_var(&b, o_v);
               nir_push_if(&b,
                           nir_ilt(&b, (v == SV_SWAP_LOOPS) ? x1 : y1, ov));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_store_var(&b, i_v, (v == SV_SWAP_LOOPS) ? y0 : x0, 0x1);
               nir_push_loop(&b);
               {
                  nir_def *iv = nir_load_var(&b, i_v);
                  nir_push_if(&b,
                              nir_ilt(&b, (v == SV_SWAP_LOOPS) ? y1 : x1, iv));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, NULL);

                  nir_def *yy = (v == SV_SWAP_LOOPS) ? iv : ov;
                  nir_def *xx = (v == SV_SWAP_LOOPS) ? ov : iv;
                  nir_def *loff = nir_iadd(&b, nir_imul(&b, yy, stride), xx);
                  nir_def *roff = (v == SV_WRONG_SHIFT) ? nir_iadd(&b, loff, d)
                                                        : nir_isub(&b, loff, d);
                  nir_def *lv = nir_u2u32(
                     &b, nir_load_deref(
                            &b, nir_build_deref_ptr_as_array(&b, left, loff)));
                  nir_def *rv = nir_u2u32(
                     &b, nir_load_deref(
                            &b, nir_build_deref_ptr_as_array(&b, right, roff)));
                  nir_def *diff = nir_isub(&b, lv, rv);
                  nir_def *ad = (v == SV_NO_ABS) ? diff : nir_iabs(&b, diff);
                  nir_store_var(&b, sad_v,
                                nir_iadd(&b, nir_load_var(&b, sad_v), ad), 0x1);
                  nir_store_var(&b, i_v, nir_iadd_imm(&b, iv, 1), 0x1);
               }
               nir_pop_loop(&b, NULL);
               nir_store_var(&b, o_v, nir_iadd_imm(&b, ov, 1), 0x1);
            }
            nir_pop_loop(&b, NULL);

            nir_def *sad = nir_load_var(&b, sad_v);
            nir_def *cur_bc = nir_load_var(&b, bc_v);
            nir_def *cur_bd = nir_load_var(&b, bd_v);
            nir_def *lt = nir_ilt(&b, sad, cur_bc);
            nir_store_var(&b, bd_v, nir_bcsel(&b, lt, d, cur_bd), 0x1);
            nir_store_var(&b, bc_v, nir_imin(&b, cur_bc, sad), 0x1);
         }
         nir_pop_if(&b, NULL);

         nir_store_var(&b, d_v, nir_iadd_imm(&b, d, 1), 0x1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, ddisp, didx),
                      nir_u2u8(&b, nir_load_var(&b, bd_v)), 0x1);
      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dcost, cidx),
                      nir_load_var(&b, bc_v), 0x1);
      if (v == SV_EXTRA_STORE) {
         nir_store_deref(&b,
                         nir_build_deref_ptr_as_array(&b, dcost, nir_imm_int(&b, 0)),
                         nir_imm_int(&b, 42), 0x1);
      }
   }
   nir_pop_if(&b, NULL);

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "build_stereo_sad_shader");
   return b.shader;
}

static void
test_stereo_sad_matcher_and_builder(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(6, false);
   struct hk_descriptor_set_layout *uab_layout = make_test_layout(6, true);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};
   struct vk_descriptor_set_layout *uab_layouts[1] = {&uab_layout->vk};

   for (unsigned v = 0; v < SV_COUNT; v++) {
      nir_shader *s = build_stereo_sad_shader((enum stereo_variant)v, 256, 1);
      struct hk_2d_stencil_plan plan = {0};
      bool expect = (v <= SV_SWAP_LOOPS);
      bool got = hk_nir_match_2d_stencil(s, 1, layouts, &plan);
      if (got != expect)
         fprintf(stderr, "stereo variant %u failed: got=%d expect=%d\n", v, got,
                 expect);
      assert(got == expect);
      if (expect) {
         assert(plan.kind == HK_2D_STENCIL_STEREO_SAD_U8);
         assert(plan.tile_w == 64 && plan.tile_h == 16);
         assert(plan.width_pc_offset_B == 0);
         assert(plan.height_pc_offset_B == 4);
         assert(plan.stride_pc_offset_B == 8);
         assert(plan.d_min_pc_offset_B == 12);
         assert(plan.d_max_pc_offset_B == 16);
         assert(plan.radius_pc_offset_B == 20);
         assert(plan.cost_stride_pc_offset_B == 0);

         nir_shader *tiled = hk_nir_build_tiled_2d_stencil(
            NULL, s, &plan, nir_address_format_64bit_bounded_global);
         assert(tiled != NULL);
         assert(tiled->info.workgroup_size[0] == 256);
         ralloc_free(tiled);
      }
      ralloc_free(s);
   }

   /* Test AGX-preprocessed NIR (which rotates the disparity loop!). */
   {
      nir_shader *s = build_stereo_sad_shader(SV_WITH_BASE_WG, 256, 1);
      agx_preprocess_nir_native_defer_idiv(s, AGX_CHIP_G17P);
      struct hk_2d_stencil_plan plan = {0};
      assert(hk_nir_match_2d_stencil(s, 1, layouts, &plan));
      assert(plan.kind == HK_2D_STENCIL_STEREO_SAD_U8);
      nir_shader *tiled = hk_nir_build_tiled_2d_stencil(
         NULL, s, &plan, nir_address_format_64bit_bounded_global);
      assert(tiled != NULL);
      ralloc_free(tiled);
      ralloc_free(s);
   }

   /* Reject UPDATE_AFTER_BIND. */
   {
      nir_shader *s = build_stereo_sad_shader(SV_NONE, 256, 1);
      struct hk_2d_stencil_plan plan = {0};
      assert(!hk_nir_match_2d_stencil(s, 1, uab_layouts, &plan));
      ralloc_free(s);
   }

   /* Dispatch validation: disjoint, in-place prev==dst, and invalid overlap. */
   {
      nir_shader *s = build_stereo_sad_shader(SV_NONE, 256, 1);
      struct hk_2d_stencil_plan plan = {0};
      assert(hk_nir_match_2d_stencil(s, 1, layouts, &plan));

      struct hk_buffer_address bufs[6] = {
         {.base_addr = 0x1000000, .size = 4096 * 2048},     /* left */
         {.base_addr = 0x2000000, .size = 4096 * 2048},     /* right */
         {.base_addr = 0x3000000, .size = 4096 * 2048},     /* prev_disp */
         {.base_addr = 0x4000000, .size = 4096 * 2048 * 4}, /* prev_cost */
         {.base_addr = 0x8000000, .size = 4096 * 2048},     /* dst_disp */
         {.base_addr = 0x9000000, .size = 4096 * 2048 * 4}, /* dst_cost */
      };
      struct hk_descriptor_set ds = {
         .layout = layout,
         .mapped_ptr = bufs,
         .size = sizeof(bufs),
      };
      struct hk_descriptor_state desc = {0};
      desc.sets[0] = &ds;
      int32_t pc[6] = {1280, 720, 1280, 0, 63, 11};
      memcpy(desc.root.push, pc, sizeof(pc));
      desc.push_written_dwords = 0x3f;

      uint32_t gx = 0, gy = 0;
      assert(hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                  5, 720, 1, &gx, &gy));
      assert(gx == 20 && gy == 45);

      /* In-place prev_disp == dst_disp and prev_cost == dst_cost -> valid! */
      bufs[2] = bufs[4];
      bufs[3] = bufs[5];
      assert(hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                  5, 720, 1, &gx, &gy));

      /* Partial overlap prev_disp != dst_disp -> reject! */
      bufs[2].base_addr = bufs[4].base_addr + 4;
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   5, 720, 1, &gx, &gy));
      bufs[2].base_addr = 0x3000000;

      /* dst_disp overlaps left -> reject! */
      bufs[4].base_addr = 0x1000100;
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   5, 720, 1, &gx, &gy));
      bufs[4].base_addr = 0x8000000;

      /* window_size too large -> reject! */
      pc[5] = (int32_t)(2 * HK_2D_STENCIL_MAX_RADIUS + 2);
      memcpy(desc.root.push, pc, sizeof(pc));
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   5, 720, 1, &gx, &gy));
      ralloc_free(s);
   }

   free(layout);
   free(uab_layout);
   printf("hk_2d_stencil: stereo SAD matcher/builder/validator tests passed\n");
}

enum sobel_variant {
   SOB_NONE,
   SOB_WITH_BASE_WG,
   SOB_DUPLICATE_TAP,  /* negative: duplicate tap instead of 8 distinct */
   SOB_WRONG_SIGN,     /* negative: flipped Gx sign */
   SOB_BORDER_NONZERO, /* negative: border writes 1.0f instead of 0.0f */
   SOB_EXACT_FADD,     /* negative: NoContraction on gradient */
   SOB_COUNT,
};

static nir_shader *
build_sobel_3x3_shader(enum sobel_variant v, uint32_t wg_x, uint32_t wg_y)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options, "test_sobel_3x3");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg_x;
   b.shader->info.workgroup_size[1] = wg_y;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 3;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *dir = make_ssbo_base(&b, 1, glsl_int_type(), 4);
   nir_deref_instr *mag = make_ssbo_base(&b, 2, glsl_float_type(), 4);

   nir_def *w =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 4);
   nir_def *h =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 4, .range = 4);
   nir_def *stride =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 8, .range = 4);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   if (v == SOB_WITH_BASE_WG) {
      nir_def *bwg = nir_load_base_workgroup_id(&b, 32);
      x = nir_iadd(&b, x, nir_imul_imm(&b, nir_channel(&b, bwg, 0), wg_x));
      y = nir_iadd(&b, y, nir_imul_imm(&b, nir_channel(&b, bwg, 1), wg_y));
   }

   nir_def *out_idx = nir_iadd(&b, nir_imul(&b, y, w), x);
   nir_def *is_int = nir_iand(
      &b,
      nir_iand(&b, nir_ige_imm(&b, x, 1),
               nir_ilt(&b, x, nir_iadd_imm(&b, w, -1))),
      nir_iand(&b, nir_ige_imm(&b, y, 1),
               nir_ilt(&b, y, nir_iadd_imm(&b, h, -1))));

   nir_push_if(&b, is_int);
   {
      nir_def *xm1 = nir_iadd_imm(&b, x, -1);
      nir_def *xp1 = nir_iadd_imm(&b, x, 1);
      nir_def *ym1 = nir_iadd_imm(&b, y, -1);
      nir_def *yp1 = nir_iadd_imm(&b, y, 1);
      nir_def *r0 = nir_imul(&b, ym1, stride);
      nir_def *r1 = nir_imul(&b, y, stride);
      nir_def *r2 = nir_imul(&b, yp1, stride);

      nir_def *p00 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, xm1, r0)))));
      nir_def *p10 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, xm1, r1)))));
      nir_def *p20 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, xm1, r2)))));
      nir_def *p01 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, x, r0)))));
      nir_def *p21 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, x, r2)))));
      nir_def *p02 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, xp1, r0)))));
      nir_def *p12 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(
                              &b, nir_build_deref_ptr_as_array(
                                     &b, src,
                                     nir_iadd(&b, v == SOB_DUPLICATE_TAP ? xm1 : xp1,
                                              r1)))));
      nir_def *p22 = nir_i2f32(
         &b, nir_u2u32(&b, nir_load_deref(&b, nir_build_deref_ptr_as_array(
                                                 &b, src, nir_iadd(&b, xp1, r2)))));

      nir_def *two = nir_imm_float(&b, 2.0f);
      if (v == SOB_EXACT_FADD)
         b.fp_math_ctrl = nir_fp_exact;
      nir_def *gy = nir_fadd(
         &b,
         nir_fadd(&b, nir_fsub(&b, p00, p20),
                  nir_fsub(&b, nir_fmul(&b, two, p01), nir_fmul(&b, two, p21))),
         nir_fsub(&b, p02, p22));
      b.fp_math_ctrl = 0;

      nir_def *gx = nir_fsub(
         &b, nir_fadd(&b, nir_fadd(&b, p00, nir_fmul(&b, two, p10)), p20),
         nir_fadd(&b, nir_fadd(&b, p02, nir_fmul(&b, two, p12)), p22));
      if (v == SOB_WRONG_SIGN)
         gx = nir_fneg(&b, gx);

      nir_def *m = nir_fadd(&b, nir_fabs(&b, gy), nir_fabs(&b, gx));
      nir_def *ratio = nir_fabs(&b, nir_fmul(&b, gx, nir_frcp(&b, gy)));
      nir_def *fzero = nir_imm_float(&b, 0.0f);
      nir_def *gx_n = nir_flt(&b, gx, fzero);
      nir_def *gy_n = nir_flt(&b, gy, fzero);
      nir_def *diag = nir_bcsel(
         &b, gy_n, nir_bcsel(&b, gx_n, nir_imm_int(&b, 5), nir_imm_int(&b, 3)),
         nir_bcsel(&b, gx_n, nir_imm_int(&b, 7), nir_imm_int(&b, 1)));
      nir_def *q1 = nir_bcsel(
         &b, nir_flt(&b, ratio, nir_imm_int(&b, 0x3ed413cdu)),
         nir_bcsel(&b, gy_n, nir_imm_int(&b, 4), nir_imm_int(&b, 0)), diag);
      nir_def *q2 = nir_bcsel(
         &b, nir_flt(&b, nir_imm_int(&b, 0x401a827au), ratio),
         nir_bcsel(&b, gx_n, nir_imm_int(&b, 6), nir_imm_int(&b, 2)), q1);
      nir_def *d = nir_iand_imm(&b, nir_iadd_imm(&b, q2, 6), 7);

      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dir, out_idx), d, 0x1);
      nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, mag, out_idx), m, 0x1);
   }
   nir_push_else(&b, NULL);
   {
      nir_push_if(&b, nir_iand(&b, nir_ilt(&b, x, w), nir_ilt(&b, y, h)));
      {
         nir_def *bv = (v == SOB_BORDER_NONZERO) ? nir_imm_float(&b, 1.0f)
                                                 : nir_imm_float(&b, 0.0f);
         nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, mag, out_idx), bv,
                         0x1);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "build_sobel_3x3_shader");
   return b.shader;
}

static void
test_sobel_3x3_matcher_and_builder(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(3, false);
   struct hk_descriptor_set_layout *uab_layout = make_test_layout(3, true);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};
   struct vk_descriptor_set_layout *uab_layouts[1] = {&uab_layout->vk};

   for (unsigned v = 0; v < SOB_COUNT; v++) {
      nir_shader *s = build_sobel_3x3_shader((enum sobel_variant)v, 1024, 1);
      struct hk_2d_stencil_plan plan = {0};
      bool expect = (v <= SOB_WITH_BASE_WG);
      bool got = hk_nir_match_2d_stencil(s, 1, layouts, &plan);
      if (got != expect)
         fprintf(stderr, "sobel variant %u failed: got=%d expect=%d\n", v, got,
                 expect);
      assert(got == expect);
      if (expect) {
         assert(plan.kind == HK_2D_STENCIL_SOBEL_3X3_U8);
         assert(plan.tile_w == 128 && plan.tile_h == 8);
         assert(plan.width_pc_offset_B == 0);
         assert(plan.height_pc_offset_B == 4);
         assert(plan.stride_pc_offset_B == 8);

         nir_shader *tiled = hk_nir_build_tiled_2d_stencil(
            NULL, s, &plan, nir_address_format_64bit_bounded_global);
         assert(tiled != NULL);
         assert(tiled->info.workgroup_size[0] == 256);
         ralloc_free(tiled);
      }
      ralloc_free(s);
   }

   /* Test AGX-preprocessed NIR. */
   {
      nir_shader *s = build_sobel_3x3_shader(SOB_WITH_BASE_WG, 1024, 1);
      agx_preprocess_nir_native_defer_idiv(s, AGX_CHIP_G17P);
      struct hk_2d_stencil_plan plan = {0};
      assert(hk_nir_match_2d_stencil(s, 1, layouts, &plan));
      assert(plan.kind == HK_2D_STENCIL_SOBEL_3X3_U8);
      nir_shader *tiled = hk_nir_build_tiled_2d_stencil(
         NULL, s, &plan, nir_address_format_64bit_bounded_global);
      assert(tiled != NULL);
      ralloc_free(tiled);
      ralloc_free(s);
   }

   /* Reject UPDATE_AFTER_BIND. */
   {
      nir_shader *s = build_sobel_3x3_shader(SOB_NONE, 1024, 1);
      struct hk_2d_stencil_plan plan = {0};
      assert(!hk_nir_match_2d_stencil(s, 1, uab_layouts, &plan));
      ralloc_free(s);
   }

   /* Dispatch validation. */
   {
      nir_shader *s = build_sobel_3x3_shader(SOB_NONE, 1024, 1);
      struct hk_2d_stencil_plan plan = {0};
      assert(hk_nir_match_2d_stencil(s, 1, layouts, &plan));

      struct hk_buffer_address bufs[3] = {
         {.base_addr = 0x1000000, .size = 4096 * 2048},
         {.base_addr = 0x2000000, .size = 4096 * 2048 * 4},
         {.base_addr = 0x6000000, .size = 4096 * 2048 * 4},
      };
      struct hk_descriptor_set ds = {
         .layout = layout,
         .mapped_ptr = bufs,
         .size = sizeof(bufs),
      };
      struct hk_descriptor_state desc = {0};
      desc.sets[0] = &ds;
      int32_t pc[3] = {2048, 1024, 2048};
      memcpy(desc.root.push, pc, sizeof(pc));
      desc.push_written_dwords = 0x7;

      uint32_t gx = 0, gy = 0;
      assert(hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                  2, 1024, 1, &gx, &gy));
      assert(gx == 16 && gy == 128);

      /* dir overlaps mag -> reject. */
      bufs[2].base_addr = bufs[1].base_addr + 16;
      assert(!hk_2d_stencil_validate_dispatch_core(true, &desc, &plan, 0, 0, 0,
                                                   2, 1024, 1, &gx, &gy));
      ralloc_free(s);
   }

   free(layout);
   free(uab_layout);
   printf("hk_2d_stencil: sobel 3x3 matcher/builder/validator tests passed\n");
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_box_blur_matcher_and_builder();
   test_stereo_sad_matcher_and_builder();
   test_sobel_3x3_matcher_and_builder();
   glsl_type_singleton_decref();
   return 0;
}
