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
         assert(plan.tile_w == 64 && plan.tile_h == 16);
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
      assert(gx == 30 && gy == 68);

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

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_box_blur_matcher_and_builder();
   glsl_type_singleton_decref();
   return 0;
}
