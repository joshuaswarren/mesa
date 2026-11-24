/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_byte_stencil.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"

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

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_matcher_and_builder();
   glsl_type_singleton_decref();
   return 0;
}
