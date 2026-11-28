/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_byte_stencil.h"
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

static const nir_shader_compiler_options test_nir_options = {0};

static struct hk_descriptor_set_layout *
make_test_layout(bool uab)
{
   size_t sz = sizeof(struct hk_descriptor_set_layout) +
               3 * sizeof(struct hk_descriptor_set_binding_layout);
   struct hk_descriptor_set_layout *layout = calloc(1, sz);
   assert(layout);
   layout->binding_count = 3;
   for (unsigned i = 0; i < 3; i++) {
      layout->binding[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layout->binding[i].array_size = 1;
      layout->binding[i].offset = i * 16;
      layout->binding[i].stride = 16;
      layout->binding[i].flags =
         uab ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0;
   }
   memset(layout->vk.blake3, 0x42, sizeof(layout->vk.blake3));
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

/* Shader variants. V_NONE and V_U2F32 are valid byte stencils; every other
 * variant changes the shader's semantics in a way the tiled kernels cannot
 * reproduce and must be rejected by the matcher. */
enum variant {
   V_NONE,
   V_U2F32,              /* u2f32(byte) instead of i2f32(u2u32(byte)) */
   V_EXTRA_STORE,        /* extra store to dst[0] */
   V_SIGNED_I8,          /* i2f32(byte): byte read as int8_t */
   V_EXACT_FADD,         /* NoContraction on the accumulate */
   V_EXACT_FMUL,         /* NoContraction on the product */
   V_IMPL_IF_RETURN,     /* if (pc == 0) return; before the outer loop */
   V_OUTER_IF_CONTINUE,  /* if (y == pc) continue; in the outer loop */
   V_OUTER_IF_BREAK,     /* if (y == pc) break; in the outer loop */
   V_INNER_IF_CONTINUE,  /* if (x == pc) continue; in the inner loop */
   V_INNER_IF_BREAK,     /* if (x == pc) break; after the border if */
   V_BORDER_BREAK,       /* break after the border copy */
   V_BORDER_RETURN,      /* return after the border copy */
   V_STENCIL_IF_BREAK,   /* if (w == 0) break; in the stencil loop */
   V_STENCIL_IF_RETURN,  /* if (w == 0) return; in the stencil loop */
   V_STENCIL_IF_CONTINUE, /* if (w == 0) { i++; continue; } (skips the tap) */
   V_AFTER_STORE_BREAK,  /* break after the interior store */
   V_AFTER_STORE_RETURN, /* return after the interior store */
   V_COUNT,
};

static const char *const variant_names[V_COUNT] = {
   [V_NONE] = "none",
   [V_U2F32] = "u2f32",
   [V_EXTRA_STORE] = "extra_store",
   [V_SIGNED_I8] = "signed_i8",
   [V_EXACT_FADD] = "exact_fadd",
   [V_EXACT_FMUL] = "exact_fmul",
   [V_IMPL_IF_RETURN] = "impl_if_return",
   [V_OUTER_IF_CONTINUE] = "outer_if_continue",
   [V_OUTER_IF_BREAK] = "outer_if_break",
   [V_INNER_IF_CONTINUE] = "inner_if_continue",
   [V_INNER_IF_BREAK] = "inner_if_break",
   [V_BORDER_BREAK] = "border_break",
   [V_BORDER_RETURN] = "border_return",
   [V_STENCIL_IF_BREAK] = "stencil_if_break",
   [V_STENCIL_IF_RETURN] = "stencil_if_return",
   [V_STENCIL_IF_CONTINUE] = "stencil_if_continue",
   [V_AFTER_STORE_BREAK] = "after_store_break",
   [V_AFTER_STORE_RETURN] = "after_store_return",
};

static nir_shader *
build_stencil_shader_variant(bool vertical, enum variant v, uint32_t wg_x,
                             uint32_t wg_y, bool lower_returns)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &test_nir_options,
      vertical ? "test_conv_v" : "test_conv_h");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg_x;
   b.shader->info.workgroup_size[1] = wg_y;
   b.shader->info.workgroup_size[2] = 1;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint8_t_type(), 1);
   nir_deref_instr *wgt = make_ssbo_base(&b, 2, glsl_float_type(), 4);

   nir_def *width =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 4);
   nir_def *height =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 4, .range = 4);
   nir_def *stride =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 8, .range = 4);
   nir_def *diameter =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 12, .range = 4);
   nir_def *extra_pc =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 16, .range = 4);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *wgid = nir_load_workgroup_id(&b);
   nir_def *bwg = nir_load_base_workgroup_id(&b, 32);
   nir_def *nwg = nir_load_num_workgroups(&b);
   nir_def *gx_init =
      (wg_x == 1) ? nir_iadd(&b, nir_channel(&b, wgid, 0), nir_channel(&b, bwg, 0))
                  : nir_channel(&b, gid, 0);
   nir_def *gy_init =
      (wg_y == 1) ? nir_iadd(&b, nir_channel(&b, wgid, 1), nir_channel(&b, bwg, 1))
                  : nir_channel(&b, gid, 1);
   nir_def *gs_x =
      (wg_x == 1) ? nir_channel(&b, nwg, 0)
                  : nir_imul_imm(&b, nir_channel(&b, nwg, 0), wg_x);
   nir_def *gs_y =
      (wg_y == 1) ? nir_channel(&b, nwg, 1)
                  : nir_imul_imm(&b, nir_channel(&b, nwg, 1), wg_y);

   nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
   nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
   nir_store_var(&b, y_var, gy_init, 0x1);

   if (v == V_IMPL_IF_RETURN) {
      nir_push_if(&b, nir_ieq_imm(&b, extra_pc, 0));
      nir_jump(&b, nir_jump_return);
      nir_pop_if(&b, NULL);
   }

   /* `continue` needs a loop continue construct, as SPIR-V produces; the
    * construct is lowered away below, as spirv_to_nir does. */
   nir_loop *outer = nir_push_loop(&b);
   if (v == V_OUTER_IF_CONTINUE)
      nir_loop_add_continue_construct(outer);
   {
      nir_def *y = nir_load_var(&b, y_var);
      nir_push_if(&b, nir_ige(&b, y, height));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      if (v == V_OUTER_IF_CONTINUE || v == V_OUTER_IF_BREAK) {
         nir_push_if(&b, nir_ieq(&b, y, extra_pc));
         nir_jump(&b, v == V_OUTER_IF_BREAK ? nir_jump_break
                                            : nir_jump_continue);
         nir_pop_if(&b, NULL);
      }

      nir_store_var(&b, x_var, gx_init, 0x1);
      nir_loop *inner = nir_push_loop(&b);
      if (v == V_INNER_IF_CONTINUE)
         nir_loop_add_continue_construct(inner);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_push_if(&b, nir_ige(&b, x, width));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         if (v == V_INNER_IF_CONTINUE) {
            nir_push_if(&b, nir_ieq(&b, x, extra_pc));
            nir_jump(&b, nir_jump_continue);
            nir_pop_if(&b, NULL);
         }

         nir_def *radius = nir_idiv(&b, diameter, nir_imm_int(&b, 2));
         nir_def *c = vertical ? y : x;
         nir_def *dim = vertical ? height : width;
         nir_def *in_border =
            nir_ior(&b, nir_ilt(&b, c, radius),
                    nir_ige(&b, c, nir_isub(&b, dim, radius)));
         nir_def *xy_off = nir_iadd(&b, x, nir_imul(&b, y, stride));

         nir_push_if(&b, in_border);
         {
            nir_def *bv = nir_load_deref(
               &b, nir_build_deref_ptr_as_array(&b, src, xy_off));
            nir_store_deref(
               &b, nir_build_deref_ptr_as_array(&b, dst, xy_off), bv, 0x1);
            if (v == V_BORDER_BREAK)
               nir_jump(&b, nir_jump_break);
            else if (v == V_BORDER_RETURN)
               nir_jump(&b, nir_jump_return);
         }
         nir_push_else(&b, NULL);
         {
            nir_variable *i_var =
               nir_local_variable_create(b.impl, glsl_int_type(), "i");
            nir_variable *off_var =
               nir_local_variable_create(b.impl, glsl_int_type(), "off");
            nir_variable *sum_var =
               nir_local_variable_create(b.impl, glsl_float_type(), "sum");

            nir_store_var(&b, i_var, nir_imm_int(&b, 0), 0x1);
            nir_store_var(&b, sum_var, nir_imm_float(&b, 0.0f), 0x1);
            nir_def *init_off =
               vertical
                  ? nir_iadd(&b, x, nir_imul(&b, nir_isub(&b, y, radius), stride))
                  : nir_iadd(&b, nir_isub(&b, x, radius), nir_imul(&b, y, stride));
            nir_store_var(&b, off_var, init_off, 0x1);

            nir_loop *sl = nir_push_loop(&b);
            if (v == V_STENCIL_IF_CONTINUE)
               nir_loop_add_continue_construct(sl);
            {
               nir_def *i = nir_load_var(&b, i_var);
               nir_push_if(&b, nir_ige(&b, i, diameter));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_def *off = nir_load_var(&b, off_var);
               nir_def *off_next = nir_iadd(
                  &b, off, vertical ? stride : nir_imm_int(&b, 1));
               nir_def *i_next = nir_iadd_imm(&b, i, 1);
               nir_def *sb = nir_load_deref(
                  &b, nir_build_deref_ptr_as_array(&b, src, off));
               nir_def *pf;
               if (v == V_SIGNED_I8)
                  pf = nir_i2f32(&b, sb);
               else if (v == V_U2F32)
                  pf = nir_u2f32(&b, sb);
               else
                  pf = nir_i2f32(&b, nir_u2u32(&b, sb));
               nir_def *wf = nir_load_deref(
                  &b, nir_build_deref_ptr_as_array(&b, wgt, i));

               if (v == V_STENCIL_IF_BREAK || v == V_STENCIL_IF_RETURN ||
                   v == V_STENCIL_IF_CONTINUE) {
                  nir_push_if(&b, nir_feq(&b, wf, nir_imm_float(&b, 0.0f)));
                  if (v == V_STENCIL_IF_CONTINUE) {
                     nir_jump(&b, nir_jump_continue);
                  } else {
                     nir_jump(&b, v == V_STENCIL_IF_BREAK ? nir_jump_break
                                                          : nir_jump_return);
                  }
                  nir_pop_if(&b, NULL);
               }

               b.fp_math_ctrl = v == V_EXACT_FMUL ? nir_fp_exact : 0;
               nir_def *prod = nir_fmul(&b, pf, wf);
               b.fp_math_ctrl = v == V_EXACT_FADD ? nir_fp_exact : 0;
               nir_def *s = nir_fadd(&b, nir_load_var(&b, sum_var), prod);
               b.fp_math_ctrl = 0;
               nir_store_var(&b, sum_var, s, 0x1);
               if (v == V_STENCIL_IF_CONTINUE)
                  nir_push_continue(&b, sl);
               nir_store_var(&b, off_var, off_next, 0x1);
               nir_store_var(&b, i_var, i_next, 0x1);
            }
            nir_pop_loop(&b, sl);

            nir_def *res = nir_u2u8(&b, nir_f2i32(&b, nir_load_var(&b, sum_var)));
            nir_store_deref(
               &b, nir_build_deref_ptr_as_array(&b, dst, xy_off), res, 0x1);
            if (v == V_EXTRA_STORE) {
               nir_store_deref(
                  &b, nir_build_deref_ptr_as_array(&b, dst, nir_imm_int(&b, 0)),
                  res, 0x1);
            } else if (v == V_AFTER_STORE_BREAK) {
               nir_jump(&b, nir_jump_break);
            } else if (v == V_AFTER_STORE_RETURN) {
               nir_jump(&b, nir_jump_return);
            }
         }
         nir_pop_if(&b, NULL);

         if (v == V_INNER_IF_BREAK) {
            nir_push_if(&b, nir_ieq(&b, x, extra_pc));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);
         }

         if (v == V_INNER_IF_CONTINUE)
            nir_push_continue(&b, inner);
         nir_store_var(&b, x_var, nir_iadd(&b, x, gs_x), 0x1);
      }
      nir_pop_loop(&b, inner);

      if (v == V_OUTER_IF_CONTINUE)
         nir_push_continue(&b, outer);
      nir_store_var(&b, y_var, nir_iadd(&b, y, gs_y), 0x1);
   }
   nir_pop_loop(&b, outer);

   nir_validate_shader(b.shader, "test_stencil_shader");
   nir_lower_continue_constructs(b.shader);
   if (lower_returns)
      nir_lower_returns(b.shader);
   nir_validate_shader(b.shader, "test_stencil_shader lowered");
   return b.shader;
}

static nir_shader *
build_stencil_shader_wg(bool vertical, bool extra_store, uint32_t wg_x,
                        uint32_t wg_y)
{
   return build_stencil_shader_variant(
      vertical, extra_store ? V_EXTRA_STORE : V_NONE, wg_x, wg_y, false);
}

static nir_shader *
build_stencil_shader(bool vertical, bool extra_store)
{
   return build_stencil_shader_wg(vertical, extra_store, 16, 16);
}

/* Every unsound variant must be rejected on both axes, with returns both
 * left in place and lowered (as the Vulkan runtime does before the driver
 * sees the shader); the valid variants must still match. */
static unsigned
test_matcher_variants(struct vk_descriptor_set_layout *const *layouts,
                      unsigned *cases_out)
{
   unsigned failures = 0, cases = 0;
   for (unsigned vert = 0; vert < 2; vert++) {
      for (unsigned v = 0; v < V_COUNT; v++) {
         for (unsigned lower = 0; lower < 2; lower++) {
            const bool expect = v == V_NONE || v == V_U2F32;
            nir_shader *nir =
               build_stencil_shader_variant(vert, v, 16, 16, lower);
            struct hk_byte_stencil_plan plan = {0};
            const bool got = hk_nir_match_byte_stencil(nir, 1, layouts, &plan);
            cases++;
            if (got != expect) {
               fprintf(stderr, "FAIL: %s %s%s: matched=%d, expected %d\n",
                       vert ? "vertical" : "horizontal", variant_names[v],
                       lower ? " (returns lowered)" : "", got, expect);
               failures++;
            } else if (got) {
               assert(plan.axis ==
                      (vert ? HK_BYTE_STENCIL_VERT : HK_BYTE_STENCIL_HORIZ));
            }
            ralloc_free(nir);
         }
      }
   }
   *cases_out = cases;
   return failures;
}

/* Near-INT32_MAX image dimensions must be rejected before any tiled-kernel
 * coordinate (`x_start + total_k`, `y0 + 7`, ...) can wrap. */
static void
test_dispatch_dimension_bounds(struct hk_descriptor_set_layout *layout,
                               const struct hk_byte_stencil_plan *h_plan,
                               const struct hk_byte_stencil_plan *v_plan)
{
   struct hk_descriptor_state desc = {0};
   struct hk_buffer_address bufs[3] = {
      {.base_addr = 0x100000000ULL, .size = 0x80000000ULL},
      {.base_addr = 0x200000000ULL, .size = 0x80000000ULL},
      {.base_addr = 0x300000000ULL, .size = 256},
   };
   struct hk_descriptor_set ds = {
      .layout = layout,
      .mapped_ptr = bufs,
      .size = sizeof(bufs),
   };
   desc.sets[0] = &ds;
   desc.push_written_dwords = 0xFu;

   const struct hk_byte_stencil_plan *plans[2] = {h_plan, v_plan};
   for (unsigned p = 0; p < 2; p++) {
      const struct hk_byte_stencil_plan *plan = plans[p];
      const int32_t max_w =
         INT32_MAX - (int32_t)(plan->tile_w + HK_BYTE_STENCIL_MAX_DIAMETER);
      const int32_t max_h =
         INT32_MAX - (int32_t)(plan->tile_h + HK_BYTE_STENCIL_MAX_DIAMETER);
      /* {width, height, stride, accept} */
      const int32_t cases[][4] = {
         {max_w, 1, max_w, 1},
         {max_w + 1, 1, max_w + 1, 0},
         {INT32_MAX, 1, INT32_MAX, 0},
         {1, max_h, 1, 1},
         {1, max_h + 1, 1, 0},
         {1, INT32_MAX, 1, 0},
      };
      for (unsigned c = 0; c < ARRAY_SIZE(cases); c++) {
         int32_t pc[4] = {cases[c][0], cases[c][1], cases[c][2], 25};
         memcpy(desc.root.push, pc, sizeof(pc));
         uint32_t gx = 0, gy = 0;
         bool ok = hk_byte_stencil_validate_dispatch_core(
            true, &desc, plan, 0, 0, 0, 16, 16, 1, &gx, &gy);
         assert(ok == (bool)cases[c][3]);
         if (ok) {
            /* Last tile origin plus one tile and one diameter still fits. */
            assert((uint64_t)(gx - 1) * plan->tile_w + plan->tile_w +
                      HK_BYTE_STENCIL_MAX_DIAMETER <=
                   (uint64_t)INT32_MAX);
            assert((uint64_t)(gy - 1) * plan->tile_h + plan->tile_h +
                      HK_BYTE_STENCIL_MAX_DIAMETER <=
                   (uint64_t)INT32_MAX);
         }
      }
   }
}

static void
test_matcher_and_builder(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   /* Positive horizontal match + tiled NIR build & validation. */
   nir_shader *h_nir = build_stencil_shader(false, false);
   struct hk_byte_stencil_plan h_plan = {0};
   assert(hk_nir_match_byte_stencil(h_nir, 1, layouts, &h_plan));
   assert(h_plan.axis == HK_BYTE_STENCIL_HORIZ);
   assert(h_plan.tile_w == 128 && h_plan.tile_h == 4);
   assert(h_plan.width_pc_offset_B == 0);
   assert(h_plan.height_pc_offset_B == 4);
   assert(h_plan.stride_pc_offset_B == 8);
   assert(h_plan.diameter_pc_offset_B == 12);
   assert(h_plan.src.binding == 0 && h_plan.dst.binding == 1 &&
          h_plan.weights.binding == 2);

   nir_shader *h_tiled = hk_nir_build_tiled_byte_stencil(
      NULL, h_nir, &h_plan, nir_address_format_64bit_bounded_global);
   assert(h_tiled != NULL);
   assert(h_tiled->info.shared_size == 0);
   assert(h_tiled->info.workgroup_size[0] == 32 &&
          h_tiled->info.workgroup_size[1] == 4);
   ralloc_free(h_tiled);
   ralloc_free(h_nir);

   /* Positive horizontal match with 1D workgroup_size = (32, 1, 1). */
   nir_shader *h1d_nir = build_stencil_shader_wg(false, false, 32, 1);
   struct hk_byte_stencil_plan h1d_plan = {0};
   assert(hk_nir_match_byte_stencil(h1d_nir, 1, layouts, &h1d_plan));
   assert(h1d_plan.axis == HK_BYTE_STENCIL_HORIZ);
   assert(h1d_plan.tile_w == 128 && h1d_plan.tile_h == 4);
   ralloc_free(h1d_nir);

   /* Positive vertical match + tiled NIR build & validation. */
   nir_shader *v_nir = build_stencil_shader(true, false);
   struct hk_byte_stencil_plan v_plan = {0};
   assert(hk_nir_match_byte_stencil(v_nir, 1, layouts, &v_plan));
   assert(v_plan.axis == HK_BYTE_STENCIL_VERT);
   assert(v_plan.tile_w == 32 && v_plan.tile_h == 64);

   nir_shader *v_tiled = hk_nir_build_tiled_byte_stencil(
      NULL, v_nir, &v_plan, nir_address_format_64bit_bounded_global);
   assert(v_tiled != NULL);
   assert(v_tiled->info.shared_size == 0);
   assert(v_tiled->info.workgroup_size[0] == 32 &&
          v_tiled->info.workgroup_size[1] == 8);
   ralloc_free(v_tiled);
   ralloc_free(v_nir);

   /* Negative: extra side-effect store must reject. */
   nir_shader *bad_nir = build_stencil_shader(false, true);
   struct hk_byte_stencil_plan bad_plan = {0};
   assert(!hk_nir_match_byte_stencil(bad_nir, 1, layouts, &bad_plan));
   ralloc_free(bad_nir);

   /* Negative: soundness variants (signed bytes, NoContraction, extra
    * control flow at every level) must reject; valid variants still match. */
   unsigned variant_cases = 0;
   assert(test_matcher_variants(layouts, &variant_cases) == 0);
   printf("hk_byte_stencil: %u matcher variant cases passed\n", variant_cases);

   /* Negative: UPDATE_AFTER_BIND layout must reject. */
   struct hk_descriptor_set_layout *uab_layout = make_test_layout(true);
   struct vk_descriptor_set_layout *uab_layouts[1] = {&uab_layout->vk};
   nir_shader *h2 = build_stencil_shader(false, false);
   assert(!hk_nir_match_byte_stencil(h2, 1, uab_layouts, &bad_plan));
   ralloc_free(h2);
   free(uab_layout);

   /* Dispatch-time validator tests. */
   struct hk_descriptor_state desc = {0};
   struct hk_buffer_address bufs[3] = {
      {.base_addr = 0x10000000ULL, .size = 1024 * 1024, .sparse_map_page = 0},
      {.base_addr = 0x20000000ULL, .size = 1024 * 1024, .sparse_map_page = 0},
      {.base_addr = 0x30000000ULL, .size = 256, .sparse_map_page = 0},
   };
   struct hk_descriptor_set ds = {
      .layout = layout,
      .mapped_ptr = bufs,
      .size = sizeof(bufs),
   };
   desc.sets[0] = &ds;

   int32_t pc[4] = {1000, 500, 1024, 25};
   memcpy(desc.root.push, pc, sizeof(pc));
   desc.push_written_dwords = 0xFu;

   uint32_t gx = 0, gy = 0;
   assert(hk_byte_stencil_validate_dispatch_core(true, &desc, &v_plan, 0, 0, 0,
                                                 16, 16, 1, &gx, &gy));
   assert(gx == DIV_ROUND_UP(1000u, 32u) && gy == DIV_ROUND_UP(500u, 64u));

   /* Reject when physical alias_free == false. */
   assert(!hk_byte_stencil_validate_dispatch_core(false, &desc, &v_plan, 0, 0, 0,
                                                  16, 16, 1, &gx, &gy));

   /* Reject when dst overlaps src. */
   bufs[1].base_addr = 0x10000100ULL;
   assert(!hk_byte_stencil_validate_dispatch_core(true, &desc, &v_plan, 0, 0, 0,
                                                  16, 16, 1, &gx, &gy));
   bufs[1].base_addr = 0x20000000ULL;

   /* Reject when diameter > MAX_DIAMETER. */
   pc[3] = 59;
   memcpy(desc.root.push, pc, sizeof(pc));
   assert(!hk_byte_stencil_validate_dispatch_core(true, &desc, &v_plan, 0, 0, 0,
                                                  16, 16, 1, &gx, &gy));
   pc[3] = 25;
   memcpy(desc.root.push, pc, sizeof(pc));

   /* Reject when a push constant dword was never written. */
   desc.push_written_dwords = 0x7u;
   assert(!hk_byte_stencil_validate_dispatch_core(true, &desc, &v_plan, 0, 0, 0,
                                                  16, 16, 1, &gx, &gy));
   desc.push_written_dwords = 0xFu;

   /* Reject sparse buffer. */
   bufs[0].sparse_map_page = 1;
   assert(!hk_byte_stencil_validate_dispatch_core(true, &desc, &v_plan, 0, 0, 0,
                                                  16, 16, 1, &gx, &gy));

   test_dispatch_dimension_bounds(layout, &h_plan, &v_plan);

   free(layout);
}

enum rgba8_variant {
   RV_NONE,
   RV_SHORT_CIRCUIT_BCSEL,
   RV_BRANCHED_GLSLANG_CFG,
   RV_EXTRA_STORE,
   RV_EXACT_FADD,
   RV_EXACT_FMUL,
   RV_STENCIL_IF_BREAK,
   RV_UNCLAMPED_MIN,
   RV_UNCLAMPED_MAX,
   RV_MISSING_ALPHA,
   RV_COUNT,
};

static const char *const rgba8_variant_names[RV_COUNT] = {
   [RV_NONE] = "none",
   [RV_SHORT_CIRCUIT_BCSEL] = "short_circuit_bcsel",
   [RV_BRANCHED_GLSLANG_CFG] = "branched_glslang_cfg",
   [RV_EXTRA_STORE] = "extra_store",
   [RV_EXACT_FADD] = "exact_fadd",
   [RV_EXACT_FMUL] = "exact_fmul",
   [RV_STENCIL_IF_BREAK] = "stencil_if_break",
   [RV_UNCLAMPED_MIN] = "unclamped_min",
   [RV_UNCLAMPED_MAX] = "unclamped_max",
   [RV_MISSING_ALPHA] = "missing_alpha",
};

static nir_shader *
build_rgba8_stencil_variant_opts(const nir_shader_compiler_options *opts,
                                 bool vertical, enum rgba8_variant v,
                                 uint32_t wg_x, uint32_t wg_y,
                                 bool lower_returns)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, opts, vertical ? "test_rgba8_v" : "test_rgba8_h");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg_x;
   b.shader->info.workgroup_size[1] = wg_y;
   b.shader->info.workgroup_size[2] = 1;

   nir_deref_instr *src = make_ssbo_base(&b, 0, glsl_uint_type(), 4);
   nir_deref_instr *dst = make_ssbo_base(&b, 1, glsl_uint_type(), 4);
   nir_deref_instr *wgt = make_ssbo_base(&b, 2, glsl_float_type(), 4);

   nir_def *width =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 4);
   nir_def *height =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 4, .range = 4);
   nir_def *stride =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 8, .range = 4);
   nir_def *diameter =
      nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base = 12, .range = 4);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *radius = nir_idiv(&b, diameter, nir_imm_int(&b, 2));

   nir_def *oob;
   if (v == RV_BRANCHED_GLSLANG_CFG) {
      /* Exact glslang short-circuit CFG with push-constant load inside arm. */
      nir_variable *oob_var =
         nir_local_variable_create(b.impl, glsl_bool_type(), "oob");
      nir_def *x_ge = nir_ige(&b, x, width);
      nir_store_var(&b, oob_var, x_ge, 0x1);
      nir_push_if(&b, nir_inot(&b, x_ge));
      {
         nir_def *h_in_arm = nir_load_push_constant(
            &b, 1, 32, nir_imm_int(&b, 0), .base = 4, .range = 4);
         nir_store_var(&b, oob_var, nir_ige(&b, y, h_in_arm), 0x1);
      }
      nir_pop_if(&b, NULL);
      oob = nir_load_var(&b, oob_var);
   } else if (v == RV_SHORT_CIRCUIT_BCSEL) {
      nir_def *x_ge = nir_ige(&b, x, width);
      nir_def *y_ge = nir_ige(&b, y, height);
      oob = nir_bcsel(&b, nir_inot(&b, x_ge), y_ge, x_ge);
   } else {
      oob = nir_ior(&b, nir_ige(&b, x, width), nir_ige(&b, y, height));
   }

   nir_push_if(&b, oob);
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   nir_def *coord = vertical ? y : x;
   nir_def *dim = vertical ? height : width;
   nir_def *c_sub = nir_isub(&b, coord, radius);
   nir_def *c_min_init;
   nir_def *c_max;
   if (v == RV_BRANCHED_GLSLANG_CFG) {
      nir_variable *min_v =
         nir_local_variable_create(b.impl, glsl_int_type(), "cmin");
      nir_push_if(&b, nir_ilt(&b, c_sub, nir_imm_int(&b, 0)));
      nir_store_var(&b, min_v, nir_imm_int(&b, 0), 0x1);
      nir_push_else(&b, NULL);
      nir_store_var(&b, min_v, nir_isub(&b, coord, radius), 0x1);
      nir_pop_if(&b, NULL);
      c_min_init = nir_load_var(&b, min_v);

      nir_variable *max_v =
         nir_local_variable_create(b.impl, glsl_int_type(), "cmax");
      nir_def *dim_in = nir_load_push_constant(
         &b, 1, 32, nir_imm_int(&b, 0), .base = vertical ? 4 : 0, .range = 4);
      nir_push_if(
         &b, nir_ilt(&b, nir_isub(&b, dim_in, nir_imm_int(&b, 1)),
                     nir_iadd(&b, coord, radius)));
      {
         nir_def *dim_arm = nir_load_push_constant(
            &b, 1, 32, nir_imm_int(&b, 0), .base = vertical ? 4 : 0, .range = 4);
         nir_store_var(&b, max_v, nir_isub(&b, dim_arm, nir_imm_int(&b, 1)),
                       0x1);
      }
      nir_push_else(&b, NULL);
      nir_store_var(&b, max_v, nir_iadd(&b, coord, radius), 0x1);
      nir_pop_if(&b, NULL);
      c_max = nir_load_var(&b, max_v);
   } else {
      c_min_init =
         (v == RV_UNCLAMPED_MIN) ? c_sub : nir_imax(&b, nir_imm_int(&b, 0), c_sub);
      nir_def *c_add = nir_iadd(&b, coord, radius);
      c_max = (v == RV_UNCLAMPED_MAX)
                 ? c_add
                 : nir_imin(&b, nir_isub(&b, dim, nir_imm_int(&b, 1)), c_add);
   }

   nir_variable *c_var = nir_local_variable_create(b.impl, glsl_int_type(), "c");
   nir_variable *i_var = nir_local_variable_create(b.impl, glsl_int_type(), "i");
   nir_variable *off_var =
      nir_local_variable_create(b.impl, glsl_int_type(), "off");
   nir_variable *s_var[4];
   for (unsigned ch = 0; ch < 4; ch++) {
      s_var[ch] = nir_local_variable_create(b.impl, glsl_float_type(), "sum");
      nir_store_var(&b, s_var[ch], nir_imm_float(&b, 0.0f), 0x1);
   }

   nir_store_var(&b, c_var, c_min_init, 0x1);
   nir_store_var(&b, i_var, nir_imm_int(&b, 0), 0x1);
   nir_def *init_off =
      vertical ? nir_iadd(&b, x, nir_imul(&b, c_min_init, stride))
               : nir_iadd(&b, c_min_init, nir_imul(&b, y, stride));
   nir_store_var(&b, off_var, init_off, 0x1);

   nir_loop *sl = nir_push_loop(&b);
   {
      nir_def *c_cur = nir_load_var(&b, c_var);
      if (v == RV_BRANCHED_GLSLANG_CFG) {
         /* spirv_to_nir form: `if (c <= c_max) { body } else { break; }` */
         nir_push_if(&b, nir_ige(&b, c_max, c_cur));
      } else {
         nir_push_if(&b, nir_ilt(&b, c_max, c_cur));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
      }

      nir_def *i = nir_load_var(&b, i_var);
      nir_def *off = nir_load_var(&b, off_var);
      nir_def *px =
         nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, src, off));

      for (unsigned ch = 0; ch < 4; ch++) {
         nir_def *byte =
            ch == 0 ? nir_iand_imm(&b, px, 0xffu)
                    : nir_iand_imm(&b, nir_ushr_imm(&b, px, ch * 8), 0xffu);
         nir_def *pf = nir_u2f32(&b, byte);
         nir_def *wf =
            nir_load_deref(&b, nir_build_deref_ptr_as_array(&b, wgt, i));
         if (v == RV_STENCIL_IF_BREAK && ch == 0) {
            nir_push_if(&b, nir_feq(&b, wf, nir_imm_float(&b, 0.0f)));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);
         }
         b.fp_math_ctrl = (v == RV_EXACT_FMUL && ch == 0) ? nir_fp_exact : 0;
         nir_def *prod = nir_fmul(&b, pf, wf);
         b.fp_math_ctrl = (v == RV_EXACT_FADD && ch == 0) ? nir_fp_exact : 0;
         nir_def *acc = nir_fadd(&b, nir_load_var(&b, s_var[ch]), prod);
         b.fp_math_ctrl = 0;
         nir_store_var(&b, s_var[ch], acc, 0x1);
      }

      if (v == RV_BRANCHED_GLSLANG_CFG) {
         nir_push_else(&b, NULL);
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
      }

      nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 0x1);
      nir_store_var(&b, off_var,
                    nir_iadd(&b, off, vertical ? stride : nir_imm_int(&b, 1)),
                    0x1);
      nir_store_var(&b, c_var, nir_iadd_imm(&b, c_cur, 1), 0x1);
   }
   nir_pop_loop(&b, sl);

   nir_def *out_px = nir_imm_int(&b, 0);
   for (unsigned ch = 0; ch < 4; ch++) {
      unsigned src_ch = (v == RV_MISSING_ALPHA && ch == 3) ? 2 : ch;
      nir_def *u = nir_iand_imm(&b, nir_f2u32(&b, nir_load_var(&b, s_var[src_ch])),
                                0xffu);
      nir_def *term = ch ? nir_ishl_imm(&b, u, ch * 8) : u;
      out_px = nir_iadd(&b, out_px, term);
   }

   nir_def *dst_off = nir_iadd(&b, x, nir_imul(&b, y, stride));
   nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, dst, dst_off), out_px,
                   0x1);
   if (v == RV_EXTRA_STORE) {
      nir_store_deref(
         &b, nir_build_deref_ptr_as_array(&b, dst, nir_imm_int(&b, 0)), out_px,
         0x1);
   }

   nir_validate_shader(b.shader, "test_rgba8_stencil");
   if (lower_returns)
      nir_lower_returns(b.shader);
   nir_validate_shader(b.shader, "test_rgba8_stencil lowered");
   return b.shader;
}

static nir_shader *
build_rgba8_stencil_variant(bool vertical, enum rgba8_variant v, uint32_t wg_x,
                            uint32_t wg_y, bool lower_returns)
{
   return build_rgba8_stencil_variant_opts(&test_nir_options, vertical, v, wg_x,
                                           wg_y, lower_returns);
}

static void
test_rgba8_matcher_builder_and_dispatch(void)
{
   struct hk_descriptor_set_layout *layout = make_test_layout(false);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   unsigned cases = 0;
   for (unsigned vert = 0; vert < 2; vert++) {
      for (unsigned v = 0; v < RV_COUNT; v++) {
         for (unsigned lower = 0; lower < 2; lower++) {
            const bool expect = (v == RV_NONE || v == RV_SHORT_CIRCUIT_BCSEL ||
                                 v == RV_BRANCHED_GLSLANG_CFG);
            nir_shader *nir =
               build_rgba8_stencil_variant(vert, v, 16, 16, lower);
            struct hk_byte_stencil_plan plan = {0};
            const bool got = hk_nir_match_byte_stencil(nir, 1, layouts, &plan);
            cases++;
            if (got != expect) {
               fprintf(stderr, "FAIL rgba8 %s %s%s: matched=%d expected=%d\n",
                       vert ? "vertical" : "horizontal", rgba8_variant_names[v],
                       lower ? " (returns lowered)" : "", got, expect);
               assert(got == expect);
            }
            if (got) {
               assert(plan.elem_kind == HK_BYTE_STENCIL_ELEM_RGBA8_CLAMPED);
               assert(plan.is_grid_stride == 0);
               assert(plan.store_conv == HK_BYTE_STENCIL_CONV_F2U32);
               assert(plan.axis ==
                      (vert ? HK_BYTE_STENCIL_VERT : HK_BYTE_STENCIL_HORIZ));
               assert(plan.tile_w == (vert ? 16 : 128));
               assert(plan.tile_h == (vert ? 64 : 4));
               if (v == RV_NONE && !lower) {
                  nir_shader *tiled = hk_nir_build_tiled_byte_stencil(
                     NULL, nir, &plan, nir_address_format_64bit_bounded_global);
                  assert(tiled != NULL);
                  assert(tiled->info.shared_size == 0);
                  assert(tiled->info.workgroup_size[0] == (vert ? 16 : 32));
                  assert(tiled->info.workgroup_size[1] == (vert ? 16 : 4));
                  /* Horizontal streaming loop loads 4 decoded pixels per 4-tap
                   * block (not 7) and stays within 8 KiB shared staging. */
                  if (!vert) {
                     unsigned shared_loads_in_loop = 0;
                     nir_function_impl *impl = nir_shader_get_entrypoint(tiled);
                     nir_foreach_block(blk, impl) {
                        bool inside_loop = false;
                        for (nir_cf_node *n = blk->cf_node.parent; n; n = n->parent) {
                           if (n->type == nir_cf_node_loop) {
                              inside_loop = true;
                              break;
                           }
                        }
                        if (!inside_loop)
                           continue;
                        unsigned blk_loads = 0;
                        nir_foreach_instr(ins, blk) {
                           if (ins->type != nir_instr_type_intrinsic)
                              continue;
                           nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
                           if (intr->intrinsic == nir_intrinsic_load_deref &&
                               nir_deref_mode_is(nir_src_as_deref(intr->src[0]),
                                                 nir_var_mem_shared))
                              blk_loads++;
                        }
                        if (blk_loads > shared_loads_in_loop)
                           shared_loads_in_loop = blk_loads;
                     }
                     assert(shared_loads_in_loop == 4);
                  }
                  ralloc_free(tiled);
               }
            }
            ralloc_free(nir);
         }
      }
   }
   printf("hk_byte_stencil: %u rgba8 matcher/builder cases passed\n", cases);

   /* Verify full AGX preprocess pipeline (agx_nir_options +
    * agx_preprocess_nir_native_defer_idiv + has_base_workgroup_id) with
    * glslang's branched CFG and (32, 2, 1) workgroup size (emits imadshl_agx
    * and lowered iabs/isign/extract_u16 before hk_maybe_build_byte_stencil). */
   for (unsigned vert = 0; vert < 2; vert++) {
      nir_shader *agx_nir = build_rgba8_stencil_variant_opts(
         &agx_nir_options, vert, RV_BRANCHED_GLSLANG_CFG, 32, 2, true);
      agx_preprocess_nir_native_defer_idiv(agx_nir, AGX_CHIP_G17P);
      const nir_lower_compute_system_values_options csv_opts = {
         .has_base_workgroup_id = true,
      };
      nir_lower_compute_system_values(agx_nir, &csv_opts);
      struct hk_byte_stencil_plan agx_plan = {0};
      assert(hk_nir_match_byte_stencil(agx_nir, 1, layouts, &agx_plan));
      assert(agx_plan.elem_kind == HK_BYTE_STENCIL_ELEM_RGBA8_CLAMPED);
      assert(agx_plan.axis ==
             (vert ? HK_BYTE_STENCIL_VERT : HK_BYTE_STENCIL_HORIZ));
      assert(agx_plan.orig_wg_size[0] == 32 && agx_plan.orig_wg_size[1] == 2);
      nir_shader *tiled = hk_nir_build_tiled_byte_stencil(
         NULL, agx_nir, &agx_plan, nir_address_format_64bit_bounded_global);
      assert(tiled != NULL);
      ralloc_free(tiled);
      ralloc_free(agx_nir);
   }

   /* Dispatch-time validator tests for RGBA8 clamped stencil. */
   nir_shader *h_nir = build_rgba8_stencil_variant(false, RV_NONE, 16, 16, true);
   nir_shader *v_nir = build_rgba8_stencil_variant(true, RV_NONE, 16, 16, true);
   struct hk_byte_stencil_plan h_plan = {0}, v_plan = {0};
   assert(hk_nir_match_byte_stencil(h_nir, 1, layouts, &h_plan));
   assert(hk_nir_match_byte_stencil(v_nir, 1, layouts, &v_plan));
   ralloc_free(h_nir);
   ralloc_free(v_nir);

   struct hk_descriptor_state desc = {0};
   const uint64_t img_bytes = (uint64_t)3840 * 2160 * 4u;
   struct hk_buffer_address bufs[3] = {
      {.base_addr = 0x10000000ULL, .size = img_bytes},
      {.base_addr = 0x20000000ULL, .size = img_bytes},
      {.base_addr = 0x30000000ULL, .size = 129 * 4u},
   };
   struct hk_descriptor_set ds = {
      .layout = layout,
      .mapped_ptr = bufs,
      .size = sizeof(bufs),
   };
   desc.sets[0] = &ds;
   desc.push_written_dwords = 0xFu;

   int32_t pc[4] = {3840, 2160, 3840, 113};
   memcpy(desc.root.push, pc, sizeof(pc));
   uint32_t gx = 0, gy = 0;

   assert(hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   assert(gx == DIV_ROUND_UP(3840u, 128u) && gy == DIV_ROUND_UP(2160u, 4u));

   assert(hk_byte_stencil_validate_dispatch_core(
      true, &desc, &v_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   assert(gx == DIV_ROUND_UP(3840u, 16u) && gy == DIV_ROUND_UP(2160u, 64u));

   /* Reject grid too small for 1-invocation-per-pixel shader. */
   assert(!hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 239, 135, 1, &gx, &gy));

   /* Max diameter 129 accepted, 130 rejected. */
   pc[3] = 129;
   memcpy(desc.root.push, pc, sizeof(pc));
   assert(hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   pc[3] = 130;
   memcpy(desc.root.push, pc, sizeof(pc));
   assert(!hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));

   /* Even diameter 128 accesses 2*(128/2)+1 = 129 weights -> reject 128*4 buf. */
   pc[3] = 128;
   memcpy(desc.root.push, pc, sizeof(pc));
   bufs[2].size = 128 * 4u;
   assert(!hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   bufs[2].size = 129 * 4u;
   assert(hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));

   /* Reject unaligned src/dst for uint32_t RGBA8 pixels. */
   bufs[0].base_addr = 0x10000002ULL;
   assert(!hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   bufs[0].base_addr = 0x10000000ULL;

   /* Reject undersized src buffer (only img_elems bytes instead of *4). */
   bufs[0].size = (uint64_t)3840 * 2160;
   assert(!hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   bufs[0].size = img_bytes;

   /* Reject VA overlap within the 4-byte-per-pixel range. */
   bufs[1].base_addr = 0x10000000ULL + img_bytes - 4u;
   assert(!hk_byte_stencil_validate_dispatch_core(
      true, &desc, &h_plan, 0, 0, 0, 240, 135, 1, &gx, &gy));
   bufs[1].base_addr = 0x20000000ULL;

   free(layout);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_matcher_and_builder();
   test_rgba8_matcher_builder_and_dispatch();
   glsl_type_singleton_decref();
   return 0;
}
