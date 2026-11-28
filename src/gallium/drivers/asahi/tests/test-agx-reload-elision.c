/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Verify fullscreen-triangle NIR coverage analysis, unconditional fragment
 * colour output analysis, and first-draw colour tilebuffer reload elision
 * (both positive elision and every negative guard). */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr)                                                            \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static nir_shader *
build_fs_tri_vs(bool cw, float z_val, bool partial)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "fullscreen-tri-vs");

   nir_def *vid = nir_load_vertex_id(&b);
   nir_def *px = nir_u2f32(&b, nir_iand_imm(&b, nir_ishl_imm(&b, vid, 1), 2));
   nir_def *py = nir_u2f32(&b, nir_iand_imm(&b, vid, 2));
   if (cw) {
      nir_def *tmp = px;
      px = py;
      py = tmp;
   }
   nir_def *scale = nir_imm_float(&b, partial ? 1.0f : 2.0f);
   nir_def *x = nir_fadd_imm(&b, nir_fmul(&b, px, scale), -1.0f);
   nir_def *y = nir_fadd_imm(&b, nir_fmul(&b, py, scale), -1.0f);
   nir_def *pos = nir_vec4(&b, x, y, nir_imm_float(&b, z_val),
                           nir_imm_float(&b, 1.0f));
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   b.shader->info.outputs_written = VARYING_BIT_POS;
   return b.shader;
}

static nir_shader *
build_color_fs_masked(const uint8_t rt_wmask[PIPE_MAX_COLOR_BUFS],
                      bool discard, bool conditional)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "color-fs");

   nir_def *fc = nir_load_frag_coord(&b);
   if (discard)
      nir_demote_if(&b, nir_flt_imm(&b, nir_channel(&b, fc, 0), 1.0f));

   nir_if *nif = NULL;
   if (conditional)
      nif = nir_push_if(&b, nir_flt_imm(&b, nir_channel(&b, fc, 0), 1.0f));

   nir_def *c = nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f);
   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (!rt_wmask[rt])
         continue;
      nir_store_output(&b, c, nir_imm_int(&b, 0), .base = rt,
                       .write_mask = rt_wmask[rt],
                       .io_semantics.location = FRAG_RESULT_DATA0 + rt);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   if (conditional)
      nir_pop_if(&b, nif);

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
build_color_fs(uint8_t rt_mask, bool discard, bool conditional)
{
   uint8_t rt_wmask[PIPE_MAX_COLOR_BUFS] = {0};
   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (rt_mask & BITFIELD_BIT(rt))
         rt_wmask[rt] = 0xf;
   }
   return build_color_fs_masked(rt_wmask, discard, conditional);
}

static void
check_nir_analyzers(void)
{
   bool z01 = false, zm11 = false;

   nir_shader *vs_ccw = build_fs_tri_vs(false, 0.0f, false);
   CHECK(agx_nir_analyze_fullscreen_tri(vs_ccw, &z01, &zm11) == 1);
   CHECK(z01 && zm11);
   ralloc_free(vs_ccw);

   nir_shader *vs_cw = build_fs_tri_vs(true, -0.5f, false);
   CHECK(agx_nir_analyze_fullscreen_tri(vs_cw, &z01, &zm11) == 2);
   CHECK(!z01 && zm11);
   ralloc_free(vs_cw);

   nir_shader *vs_partial = build_fs_tri_vs(false, 0.0f, true);
   CHECK(agx_nir_analyze_fullscreen_tri(vs_partial, &z01, &zm11) == 0);
   ralloc_free(vs_partial);

   bool uses_discard = false;
   nir_shader *fs_mrt = build_color_fs(0x5, false, false);
   CHECK(agx_nir_analyze_fs_color_outputs(fs_mrt, &uses_discard) == 0x5);
   CHECK(!uses_discard);
   ralloc_free(fs_mrt);

   nir_shader *fs_disc = build_color_fs(0x1, true, false);
   CHECK(agx_nir_analyze_fs_color_outputs(fs_disc, &uses_discard) == 0);
   CHECK(uses_discard);
   ralloc_free(fs_disc);

   nir_shader *fs_cond = build_color_fs(0x1, false, true);
   CHECK(agx_nir_analyze_fs_color_outputs(fs_cond, &uses_discard) == 0);
   CHECK(!uses_discard);
   ralloc_free(fs_cond);

   const uint8_t partial_wmask[PIPE_MAX_COLOR_BUFS] = {0xf, 0x3, 0x7, 0x1};
   nir_shader *fs_partial = build_color_fs_masked(partial_wmask, false, false);
   uint8_t masks[PIPE_MAX_COLOR_BUFS] = {0};
   agx_nir_analyze_fs_color_output_masks(fs_partial, masks, &uses_discard);
   CHECK(!uses_discard);
   CHECK(masks[0] == 0xf && masks[1] == 0x3 && masks[2] == 0x7 &&
         masks[3] == 0x1);
   CHECK(agx_nir_analyze_fs_color_outputs(fs_partial, &uses_discard) == 0x1);
   ralloc_free(fs_partial);
}

static struct pipe_resource *
create_rt_format(struct pipe_screen *screen, enum pipe_format format,
                 unsigned w, unsigned h, unsigned samples)
{
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = format,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .nr_samples = samples,
      .nr_storage_samples = samples,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *rsrc = screen->resource_create(screen, &templ);
   CHECK(rsrc);
   /* Mark level 0 valid so agx_batch_init_state() schedules a colour reload. */
   agx_resource_set_valid(agx_resource(rsrc), 0);
   return rsrc;
}

static struct pipe_resource *
create_rt(struct pipe_screen *screen, unsigned w, unsigned h, unsigned samples)
{
   return create_rt_format(screen, PIPE_FORMAT_R8G8B8A8_UNORM, w, h, samples);
}

static uint32_t
run_first_draw_case(struct pipe_context *pctx, struct pipe_resource *rt0,
                    struct pipe_resource *rt1, void *vs, void *fs,
                    const struct pipe_rasterizer_state *rs_desc,
                    const struct pipe_blend_state *blend_desc,
                    const struct pipe_depth_stencil_alpha_state *zsa_desc,
                    const struct pipe_viewport_state *vp,
                    const struct pipe_scissor_state *ss,
                    unsigned sample_mask, unsigned draw_count)
{
   struct agx_context *ctx = agx_context(pctx);

   struct pipe_framebuffer_state fb = {
      .width = rt0->width0,
      .height = rt0->height0,
      .nr_cbufs = rt1 ? 2 : 1,
   };
   fb.cbufs[0].texture = rt0;
   fb.cbufs[0].format = rt0->format;
   if (rt1) {
      fb.cbufs[1].texture = rt1;
      fb.cbufs[1].format = rt1->format;
   }
   pctx->set_framebuffer_state(pctx, &fb);

   void *rs = pctx->create_rasterizer_state(pctx, rs_desc);
   void *bl = pctx->create_blend_state(pctx, blend_desc);
   void *zs = pctx->create_depth_stencil_alpha_state(pctx, zsa_desc);
   pctx->bind_rasterizer_state(pctx, rs);
   pctx->bind_blend_state(pctx, bl);
   pctx->bind_depth_stencil_alpha_state(pctx, zs);
   pctx->bind_vs_state(pctx, vs);
   pctx->bind_fs_state(pctx, fs);
   pctx->set_viewport_states(pctx, 0, 1, vp);
   if (ss)
      pctx->set_scissor_states(pctx, 0, 1, ss);
   pctx->set_sample_mask(pctx, sample_mask);

   struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES,
      .instance_count = 1,
   };
   struct pipe_draw_start_count_bias draw = {
      .start = 0,
      .count = draw_count,
   };
   pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);

   struct agx_batch *batch = agx_get_batch(ctx);
   const uint32_t load_mask = batch->load;

   pctx->flush(pctx, NULL, 0);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->delete_rasterizer_state(pctx, rs);
   pctx->delete_blend_state(pctx, bl);
   pctx->delete_depth_stencil_alpha_state(pctx, zs);
   return load_mask;
}

static void
check_draw_elision(void)
{
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);

   static const driOptionDescription options[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, cache;
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {.options = &cache, .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);

   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);

   struct pipe_shader_state vs_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_fs_tri_vs(false, 0.0f, false),
   };
   void *vs = pctx->create_vs_state(pctx, &vs_cso);
   struct pipe_shader_state fs0_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_color_fs(0x1, false, false),
   };
   void *fs0 = pctx->create_fs_state(pctx, &fs0_cso);
   struct pipe_shader_state fs01_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_color_fs(0x3, false, false),
   };
   void *fs01 = pctx->create_fs_state(pctx, &fs01_cso);
   struct pipe_shader_state fs_disc_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_color_fs(0x1, true, false),
   };
   void *fs_disc = pctx->create_fs_state(pctx, &fs_disc_cso);

   /* The fullscreen triangle has no vertex inputs, but the driver links the
    * VS prolog from the bound vertex elements state on every draw, as any
    * Gallium frontend guarantees one is bound. */
   void *ve = pctx->create_vertex_elements_state(pctx, 0, NULL);
   CHECK(ve);
   pctx->bind_vertex_elements_state(pctx, ve);

   struct pipe_resource *rt0 = create_rt(screen, 128, 64, 1);
   struct pipe_resource *rt1 = create_rt(screen, 128, 64, 1);
   struct pipe_resource *rt_ms = create_rt(screen, 128, 64, 4);

   const struct pipe_rasterizer_state base_rs = {
      .fill_front = PIPE_POLYGON_MODE_FILL,
      .fill_back = PIPE_POLYGON_MODE_FILL,
      .front_ccw = false,
      .cull_face = PIPE_FACE_BACK,
      .depth_clip_near = true,
      .depth_clip_far = true,
   };
   const struct pipe_blend_state base_bl = {
      .rt[0] = {.colormask = PIPE_MASK_RGBA},
   };
   const struct pipe_depth_stencil_alpha_state base_zs = {0};
   const struct pipe_viewport_state base_vp = {
      .scale = {64.0f, 32.0f, 0.5f},
      .translate = {64.0f, 32.0f, 0.5f},
   };
   const struct pipe_scissor_state full_ss = {0, 0, 128, 64};

   /* Positive: single-RT opaque fullscreen triangle elides COLOR0 reload. */
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs0, &base_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) == 0);

   /* Positive: 4x MSAA full sample mask elides COLOR0 reload. */
   CHECK((run_first_draw_case(pctx, rt_ms, NULL, vs, fs0, &base_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) == 0);

   /* Partial MRT: FS writes only RT0 -> elide COLOR0, keep COLOR1 reload. */
   uint32_t mrt_load = run_first_draw_case(pctx, rt0, rt1, vs, fs0, &base_rs,
                                           &base_bl, &base_zs, &base_vp,
                                           &full_ss, 0xf, 3);
   CHECK((mrt_load & PIPE_CLEAR_COLOR0) == 0);
   CHECK((mrt_load & PIPE_CLEAR_COLOR1) != 0);

   /* Full MRT: FS writes RT0 + RT1 -> elide both reloads. */
   CHECK((run_first_draw_case(pctx, rt0, rt1, vs, fs01, &base_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          (PIPE_CLEAR_COLOR0 | PIPE_CLEAR_COLOR1)) == 0);

   /* Negative: scissor smaller than FB keeps reload. */
   struct pipe_rasterizer_state sc_rs = base_rs;
   sc_rs.scissor = true;
   struct pipe_scissor_state small_ss = {0, 0, 127, 64};
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs0, &sc_rs, &base_bl,
                              &base_zs, &base_vp, &small_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: culled front face keeps reload. */
   struct pipe_rasterizer_state cull_rs = base_rs;
   cull_rs.cull_face = PIPE_FACE_FRONT;
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs0, &cull_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: blending enabled keeps reload. */
   struct pipe_blend_state blend_on = base_bl;
   blend_on.rt[0].blend_enable = true;
   blend_on.rt[0].rgb_func = PIPE_BLEND_ADD;
   blend_on.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_SRC_ALPHA;
   blend_on.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   blend_on.rt[0].alpha_func = PIPE_BLEND_ADD;
   blend_on.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_SRC_ALPHA;
   blend_on.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs0, &base_rs, &blend_on,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: partial colormask keeps reload. */
   struct pipe_blend_state mask_rgb = base_bl;
   mask_rgb.rt[0].colormask = PIPE_MASK_RGB;
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs0, &base_rs, &mask_rgb,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: fragment discard keeps reload. */
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs_disc, &base_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: depth test keeps reload. */
   struct pipe_depth_stencil_alpha_state depth_on = base_zs;
   depth_on.depth_enabled = true;
   depth_on.depth_func = PIPE_FUNC_LESS;
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs0, &base_rs, &base_bl,
                              &depth_on, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: partial MSAA sample mask keeps reload. */
   CHECK((run_first_draw_case(pctx, rt_ms, NULL, vs, fs0, &base_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0x3, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Format-aware reload elision: R32_FLOAT (wmask 0x1), R16G16_FLOAT (0x3),
    * and R8G8B8X8_UNORM (0x7) elide reloads when all live channels are written,
    * while missing a live channel preserves the reload. */
   const uint8_t wmask_r_rg[PIPE_MAX_COLOR_BUFS] = {0x1, 0x3};
   struct pipe_shader_state fs_r_rg_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_color_fs_masked(wmask_r_rg, false, false),
   };
   void *fs_r_rg = pctx->create_fs_state(pctx, &fs_r_rg_cso);

   const uint8_t wmask_rgb[PIPE_MAX_COLOR_BUFS] = {0x7};
   struct pipe_shader_state fs_rgb_cso = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_color_fs_masked(wmask_rgb, false, false),
   };
   void *fs_rgb = pctx->create_fs_state(pctx, &fs_rgb_cso);

   struct pipe_resource *rt_r32f =
      create_rt_format(screen, PIPE_FORMAT_R32_FLOAT, 128, 64, 1);
   struct pipe_resource *rt_rg16f =
      create_rt_format(screen, PIPE_FORMAT_R16G16_FLOAT, 128, 64, 1);
   struct pipe_resource *rt_rgbx =
      create_rt_format(screen, PIPE_FORMAT_R8G8B8X8_UNORM, 128, 64, 1);

   /* Positive: R32_FLOAT + R16G16_FLOAT with matching FS masks and colormasks. */
   struct pipe_blend_state bl_r_rg = {
      .independent_blend_enable = true,
      .rt[0] = {.colormask = PIPE_MASK_R},
      .rt[1] = {.colormask = PIPE_MASK_R | PIPE_MASK_G},
   };
   CHECK((run_first_draw_case(pctx, rt_r32f, rt_rg16f, vs, fs_r_rg, &base_rs,
                              &bl_r_rg, &base_zs, &base_vp, &full_ss, 0xf, 3) &
          (PIPE_CLEAR_COLOR0 | PIPE_CLEAR_COLOR1)) == 0);

   /* Positive: R8G8B8X8_UNORM with RGB-only FS output and RGB colormask. */
   CHECK((run_first_draw_case(pctx, rt_rgbx, NULL, vs, fs_rgb, &base_rs,
                              &mask_rgb, &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) == 0);

   /* Negative: R16G16_FLOAT with R-only FS output keeps reload. */
   CHECK((run_first_draw_case(pctx, rt_rg16f, NULL, vs, fs_r_rg, &base_rs,
                              &base_bl, &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: R16G16_FLOAT with R-only blend colormask keeps reload. */
   struct pipe_blend_state bl_r_only = {
      .rt[0] = {.colormask = PIPE_MASK_R},
   };
   CHECK((run_first_draw_case(pctx, rt_rg16f, NULL, vs, fs0, &base_rs,
                              &bl_r_only, &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   /* Negative: RGBA8 with RGB-only FS output keeps reload. */
   CHECK((run_first_draw_case(pctx, rt0, NULL, vs, fs_rgb, &base_rs, &base_bl,
                              &base_zs, &base_vp, &full_ss, 0xf, 3) &
          PIPE_CLEAR_COLOR0) != 0);

   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->delete_vertex_elements_state(pctx, ve);
   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->delete_vs_state(pctx, vs);
   pctx->delete_fs_state(pctx, fs0);
   pctx->delete_fs_state(pctx, fs01);
   pctx->delete_fs_state(pctx, fs_disc);
   pctx->delete_fs_state(pctx, fs_r_rg);
   pctx->delete_fs_state(pctx, fs_rgb);
   pipe_resource_reference(&rt0, NULL);
   pipe_resource_reference(&rt1, NULL);
   pipe_resource_reference(&rt_ms, NULL);
   pipe_resource_reference(&rt_r32f, NULL);
   pipe_resource_reference(&rt_rg16f, NULL);
   pipe_resource_reference(&rt_rgbx, NULL);
   pctx->destroy(pctx);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
}

int
main(int argc, char **argv)
{
   check_nir_analyzers();
   if (argc > 1 && strcmp(argv[1], "--nir-only") == 0)
      return 0;
   check_draw_elision();
   return 0;
}
