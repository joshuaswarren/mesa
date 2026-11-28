/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Driver-level regression tests under the no-op DRM shim. Fence completion is
 * controlled here so that polling, submission and blocking remain distinct. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_vdm_hal200_direct_draw_core.h"
#include "asahi/lib/agx_vdm_index_list_core.h"
#include "asahi/lib/agx_vdm_state_update_core.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_device.h"
#include "agx_public.h"
#include "agx_state.h"

static bool busy;
static unsigned polls, waits, ready_polls;

/* Interpose the libdrm entry point for this executable only. */
int
drmSyncobjWait(int fd, uint32_t *handles, unsigned count, int64_t timeout,
               unsigned flags, uint32_t *first_signaled)
{
   if (timeout == 0) {
      polls++;
      if (busy && !ready_polls)
         return -ETIME;
      if (ready_polls)
         ready_polls--;
   } else {
      waits++;
      busy = false;
   }
   if (first_signaled)
      *first_signaled = 0;
   return 0;
}

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x);               \
         return false;                                                         \
      }                                                                        \
   } while (0)

struct draw_state {
   void *vs, *fs, *rast, *blend, *depth, *ve;
};

static void *
create_vs(struct pipe_context *pctx, bool vertex_id, bool draw_id,
          bool vertex_attr)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "VS dirty tracking");
   nir_def *pos = nir_imm_vec4(&b, 0, 0, 0, 1);
   if (vertex_id || draw_id || vertex_attr) {
      nir_def *acc = nir_imm_int(&b, 0);
      if (vertex_id)
         acc = nir_iadd(&b, nir_load_vertex_id(&b), nir_load_base_instance(&b));
      if (draw_id)
         acc = nir_iadd(&b, acc, nir_load_draw_id(&b));
      nir_def *id = nir_u2f32(&b, acc);
      if (vertex_attr) {
         nir_def *attr = nir_load_input(
            &b, 2, 32, nir_imm_int(&b, 0), .base = 0,
            .io_semantics.location = VERT_ATTRIB_GENERIC0,
            .io_semantics.num_slots = 1);
         id = nir_fadd(&b, id, nir_channel(&b, attr, 0));
      }
      pos = nir_vec4(&b, id, id, nir_imm_float(&b, 0), nir_imm_float(&b, 1));
   }
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   return pctx->create_vs_state(pctx, &state);
}

static bool
setup_draw(struct pipe_context *pctx, struct draw_state *s)
{
   s->vs = create_vs(pctx, false, false, false);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "VS dirty tracking fragment");
   /* Exercise the application shader link, including inline blend epilogs. */
   b.shader->info.internal = false;
   nir_store_output(&b, nir_imm_vec4(&b, 1, 0, 0, 1), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state fs = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   s->fs = pctx->create_fs_state(pctx, &fs);
   struct pipe_rasterizer_state rast = {
      .depth_clip_near = true,
      .depth_clip_far = true,
      .point_size = 1,
      .line_width = 1,
   };
   struct pipe_blend_state blend = {.rt[0].colormask = 0xf};
   struct pipe_depth_stencil_alpha_state depth = {0};
   s->rast = pctx->create_rasterizer_state(pctx, &rast);
   s->blend = pctx->create_blend_state(pctx, &blend);
   s->depth = pctx->create_depth_stencil_alpha_state(pctx, &depth);
   s->ve = pctx->create_vertex_elements_state(pctx, 0, NULL);
   CHECK(s->vs && s->fs && s->rast && s->blend && s->depth && s->ve);
   pctx->bind_vs_state(pctx, s->vs);
   pctx->bind_fs_state(pctx, s->fs);
   pctx->bind_rasterizer_state(pctx, s->rast);
   pctx->bind_blend_state(pctx, s->blend);
   pctx->bind_depth_stencil_alpha_state(pctx, s->depth);
   pctx->bind_vertex_elements_state(pctx, s->ve);
   struct pipe_framebuffer_state fb = {.width = 16, .height = 16, .layers = 1};
   struct pipe_viewport_state vp = {.scale = {8, 8, 0.5},
                                    .translate = {8, 8, 0.5}};
   pctx->set_framebuffer_state(pctx, &fb);
   pctx->set_viewport_states(pctx, 0, 1, &vp);
   pctx->set_sample_mask(pctx, ~0u);
   return true;
}

static void
finish_draw(struct pipe_context *pctx, struct draw_state *s)
{
   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->delete_vs_state(pctx, s->vs);
   pctx->delete_fs_state(pctx, s->fs);
   pctx->delete_rasterizer_state(pctx, s->rast);
   pctx->delete_blend_state(pctx, s->blend);
   pctx->delete_depth_stencil_alpha_state(pctx, s->depth);
   pctx->delete_vertex_elements_state(pctx, s->ve);
}

static void
draw(struct pipe_context *pctx)
{
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES,
                                 .instance_count = 1};
   struct pipe_draw_start_count_bias d = {.count = 3};
   pctx->draw_vbo(pctx, &info, 0, NULL, &d, 1);
}

static bool
check_last_indexed_indirect_draw(struct agx_context *ctx,
                                 uint64_t expected_indirect_va,
                                 bool check_draw_params,
                                 unsigned expected_draw_id)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   const size_t body_len =
      AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH -
      AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH;
   struct agx_vdm_index_list_indirect_indexed_info unpacked;

   CHECK(ctx->batch && ctx->batch->vdm.bo);
   CHECK((size_t)(ctx->batch->vdm.current - (uint8_t *)agx_bo_map(ctx->batch->vdm.bo)) >=
         body_len);
   CHECK(agx_vdm_index_list_indirect_indexed_unpack(
      dev->chip, ctx->batch->vdm.current - body_len, body_len, &unpacked));
   CHECK(unpacked.indirect_buffer == expected_indirect_va);
   if (check_draw_params) {
      CHECK(ctx->batch->uniforms.draw_id == expected_draw_id);
      CHECK(ctx->batch->uniforms.tables[AGX_SYSVAL_TABLE_PARAMS] ==
            expected_indirect_va + 12);
   }
   return true;
}

static bool
check_mdi_subdraw_offset_bounds(void)
{
   struct pipe_resource fake_buf = {0};
   struct pipe_draw_indirect_info info = {.buffer = &fake_buf};
   uint32_t out = 0;

   /* Exact fit on the last draw for both indexed (20 B) and non-indexed (16 B). */
   fake_buf.width0 = 256u + 1599u * 4096u + 20u;
   info.offset = 256u;
   info.stride = 4096u;
   CHECK(agx_mdi_subdraw_offset(&info, 1599u, 20u, &out) &&
         out == 256u + 1599u * 4096u);
   CHECK(!agx_mdi_subdraw_offset(&info, 1600u, 20u, &out));

   fake_buf.width0 = 256u + 1599u * 4096u + 16u;
   CHECK(!agx_mdi_subdraw_offset(&info, 1599u, 20u, &out));
   CHECK(agx_mdi_subdraw_offset(&info, 1599u, 16u, &out) &&
         out == 256u + 1599u * 4096u);

   /* 32-bit multiply wraparound (2 * 0x80000000u == 0 in uint32_t) must fail
    * before multiplying, even against a full 4 GiB logical buffer. */
   fake_buf.width0 = UINT32_MAX;
   info.offset = 0u;
   info.stride = 0x80000000u;
   CHECK(agx_mdi_subdraw_offset(&info, 0u, 20u, &out) && out == 0u);
   CHECK(agx_mdi_subdraw_offset(&info, 1u, 20u, &out) && out == 0x80000000u);
   CHECK(!agx_mdi_subdraw_offset(&info, 2u, 20u, &out));

   fake_buf.width0 = 64u;
   CHECK(agx_mdi_subdraw_offset(&info, 0u, 20u, &out) && out == 0u);
   CHECK(!agx_mdi_subdraw_offset(&info, 1u, 20u, &out));
   CHECK(!agx_mdi_subdraw_offset(&info, 2u, 20u, &out));

   /* 32-bit addition wraparound (offset + i * stride > UINT32_MAX). */
   fake_buf.width0 = UINT32_MAX;
   info.offset = 0xfffffff0u;
   info.stride = 20u;
   CHECK(!agx_mdi_subdraw_offset(&info, 0u, 20u, &out));
   CHECK(!agx_mdi_subdraw_offset(&info, 1u, 20u, &out));

   info.offset = 0xffffffc0u;
   CHECK(agx_mdi_subdraw_offset(&info, 0u, 20u, &out) && out == 0xffffffc0u);
   CHECK(agx_mdi_subdraw_offset(&info, 2u, 20u, &out) && out == 0xffffffe8u);
   CHECK(!agx_mdi_subdraw_offset(&info, 3u, 20u, &out));

   /* Zero stride repeats base offset when the single record fits. */
   fake_buf.width0 = 36u;
   info.offset = 16u;
   info.stride = 0u;
   CHECK(agx_mdi_subdraw_offset(&info, 0u, 20u, &out) && out == 16u);
   CHECK(agx_mdi_subdraw_offset(&info, 10000u, 20u, &out) && out == 16u);
   fake_buf.width0 = 32u;
   CHECK(!agx_mdi_subdraw_offset(&info, 0u, 20u, &out));

   /* Unaligned base offset or stride must be rejected. */
   fake_buf.width0 = 256u;
   info.offset = 2u;
   info.stride = 20u;
   CHECK(!agx_mdi_subdraw_offset(&info, 0u, 20u, &out));
   info.offset = 0u;
   info.stride = 18u;
   CHECK(!agx_mdi_subdraw_offset(&info, 0u, 20u, &out));

   return true;
}

static bool
check_vs(struct pipe_context *pctx)
{
   struct agx_context *ctx = agx_context(pctx);
   struct draw_state s;
   CHECK(setup_draw(pctx, &s));
   draw(pctx);
   CHECK(ctx->skipped_no_program == 0 && ctx->batch && ctx->batch->draws == 1);
   CHECK(ctx->perf.vs_emits == 1);
   for (unsigned i = 0; i < 8; ++i)
      draw(pctx);
   CHECK(ctx->perf.vs_emits == 1 && ctx->perf.draws == 9);

   /* A PPP-only update must still emit the native bind tail, but not VS. */
   uint8_t *before = ctx->batch->vdm.current;
   struct pipe_stencil_ref ref = {.ref_value = {1, 2}};
   pctx->set_stencil_ref(pctx, ref);
   draw(pctx);
   CHECK(ctx->perf.vs_emits == 1 && ctx->batch->vdm.current > before);
   if (agx_chip_native_agx3(agx_device(pctx->screen)->chip)) {
      struct agx_vdm_state_update_info update;
      CHECK(!agx_vdm_state_update_unpack(AGX_CHIP_G17P, before,
                                         AGX_VDM_STATE_UPDATE_LENGTH, &update));
   }
   struct pipe_viewport_state vp = {.scale = {4, 4, 0.5},
                                    .translate = {4, 4, 0.5}};
   pctx->set_viewport_states(pctx, 0, 1, &vp);
   draw(pctx);
   CHECK(ctx->perf.vs_emits == 1);

   /* Fragment-only root relocation leaves the immutable VS snapshot valid. */
   uint64_t root = ctx->batch->uniforms.tables[AGX_SYSVAL_TABLE_ROOT];
   uint64_t emits = ctx->perf.vs_emits;
   struct pipe_blend_color color = {.color = {0.25, 0.5, 0.75, 1}};
   pctx->set_blend_color(pctx, &color);
   draw(pctx);
   CHECK(ctx->batch->uniforms.tables[AGX_SYSVAL_TABLE_ROOT] != root);
   CHECK(ctx->perf.vs_emits == emits);

   /* Occlusion queries change PPP state, not this VS's uniform inputs. */
   struct pipe_query *occlusion =
      pctx->create_query(pctx, PIPE_QUERY_OCCLUSION_COUNTER, 0);
   CHECK(occlusion && pctx->begin_query(pctx, occlusion));
   draw(pctx);
   CHECK(ctx->perf.vs_emits == emits && pctx->end_query(pctx, occlusion));
   draw(pctx);
   CHECK(ctx->perf.vs_emits == emits);

   /* A different shader and per-draw base parameters must refresh VS.
    * On HAL200, repeating the same VS with only VS uniform/param changes
    * elides the bind tail and selector until a VDM STREAM_LINK starts a new
    * chunk. */
   const bool native_agx3 =
      agx_chip_native_agx3(agx_device(pctx->screen)->chip);
   void *id_vs = create_vs(pctx, true, false, false);
   CHECK(id_vs);
   pctx->bind_vs_state(pctx, id_vs);
   emits = ctx->perf.vs_emits;
   draw(pctx);
   CHECK(ctx->perf.vs_emits == emits + 1 && ctx->skipped_no_program == 0);
   CHECK(ctx->linked.vs->uses_base_param);
   size_t scissor_before = ctx->batch->scissor.size;
   size_t zbias_before = ctx->batch->depth_bias.size;
   before = ctx->batch->vdm.current;
   draw(pctx);
   CHECK(ctx->perf.vs_emits == emits + 2);
   if (native_agx3) {
      CHECK(ctx->batch->scissor.size == scissor_before);
      CHECK(ctx->batch->depth_bias.size == zbias_before);
      CHECK((size_t)(ctx->batch->vdm.current - before) ==
            AGX_VDM_STATE_UPDATE_LENGTH +
               AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH);
   }

   /* Cross a 512 KiB VDM chunk boundary with VS-only updates to exercise
    * STREAM_LINK reset and bind-tail re-latching under the stream validator. */
   if (native_agx3) {
      struct agx_bo *start_vdm_bo = ctx->batch->vdm.bo;
      uint8_t *start_vdm_end = ctx->batch->vdm.end;
      bool linked = false;
      for (unsigned i = 0; i < 20000; ++i) {
         draw(pctx);
         if (ctx->batch->vdm.bo != start_vdm_bo ||
             ctx->batch->vdm.end != start_vdm_end) {
            linked = true;
            break;
         }
      }
      CHECK(linked);
      CHECK(ctx->batch->g17p_bind_tail_latched);
      CHECK(ctx->batch->g17p_have_latched_vs);
      /* The draw right after re-latching in the new chunk elides again. */
      scissor_before = ctx->batch->scissor.size;
      zbias_before = ctx->batch->depth_bias.size;
      before = ctx->batch->vdm.current;
      draw(pctx);
      CHECK(ctx->batch->scissor.size == scissor_before);
      CHECK(ctx->batch->depth_bias.size == zbias_before);
      CHECK((size_t)(ctx->batch->vdm.current - before) ==
            AGX_VDM_STATE_UPDATE_LENGTH +
               AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH);
   }
   /* With a colour attachment, the launched FS can include its blend
    * epilog. Its bind tail remains valid across vertex-only updates. */
   if (native_agx3 && agx_chip_native_preamble(agx_device(pctx->screen)->chip)) {
      const struct pipe_resource templ = {
         .target = PIPE_TEXTURE_2D, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 16, .height0 = 16, .depth0 = 1, .array_size = 1,
         .bind = PIPE_BIND_RENDER_TARGET,
      };
      struct pipe_resource *rt = pctx->screen->resource_create(pctx->screen,
                                                                &templ);
      CHECK(rt);
      struct pipe_framebuffer_state fb = {
         .width = 16, .height = 16, .layers = 1, .nr_cbufs = 1,
         .cbufs = {{.texture = rt, .format = templ.format}},
      };
      pctx->set_framebuffer_state(pctx, &fb);
      draw(pctx);
      CHECK(ctx->linked.fs && ctx->linked.fs->inline_blend_fs);
      CHECK(ctx->batch->g17p_fragment_shader == ctx->linked.fs->inline_blend_fs);
      before = ctx->batch->vdm.current;
      draw(pctx);
      CHECK((size_t)(ctx->batch->vdm.current - before) ==
            AGX_VDM_STATE_UPDATE_LENGTH + AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH);
      fb.nr_cbufs = 0;
      fb.cbufs[0] = (struct pipe_surface){0};
      pctx->set_framebuffer_state(pctx, &fb);
      pipe_resource_reference(&rt, NULL);
   }
   pctx->bind_vs_state(pctx, s.vs);
   pctx->delete_vs_state(pctx, id_vs);

   draw(pctx);
   emits = ctx->perf.vs_emits;
   uint32_t args[] = {3, 1, 0, 0};
   struct pipe_resource *indirect_buffer =
      pipe_buffer_create(pctx->screen, PIPE_BIND_COMMAND_ARGS_BUFFER,
                         PIPE_USAGE_DEFAULT, sizeof(args));
   CHECK(indirect_buffer);
   pipe_buffer_write(pctx, indirect_buffer, 0, sizeof(args), args);
   struct pipe_draw_info indirect_info = {
      .mode = MESA_PRIM_TRIANGLES,
      .instance_count = 1,
   };
   struct pipe_draw_indirect_info indirect = {
      .buffer = indirect_buffer,
      .draw_count = 1,
      .stride = sizeof(args),
   };
   struct pipe_draw_start_count_bias indirect_draw = {0};
   for (unsigned i = 0; i < 2; ++i) {
      scissor_before = ctx->batch->scissor.size;
      zbias_before = ctx->batch->depth_bias.size;
      before = ctx->batch->vdm.current;
      pctx->draw_vbo(pctx, &indirect_info, 0, &indirect, &indirect_draw, 1);
      CHECK(ctx->perf.vs_emits == emits);
      if (native_agx3) {
         CHECK(ctx->batch->scissor.size == scissor_before);
         CHECK(ctx->batch->depth_bias.size == zbias_before);
         CHECK((size_t)(ctx->batch->vdm.current - before) ==
               AGX_VDM_HAL200_INDIRECT_NONINDEXED_DRAW_LENGTH -
                  AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH);
      }
   }
   pipe_resource_reference(&indirect_buffer, NULL);

   /* Indirect multi-draw: width/range guards, state-clean (body-only),
    * per-draw gl_DrawID + base-param across a 64 KiB VDM STREAM_LINK boundary,
    * large-stride last-draw addressing, and forced fallback after STREAM_LINK. */
   CHECK(check_mdi_subdraw_offset_bounds());
   {
      const uint16_t indices[3] = {0, 1, 2};
      struct pipe_resource *ib =
         pipe_buffer_create(pctx->screen, PIPE_BIND_INDEX_BUFFER,
                            PIPE_USAGE_DEFAULT, sizeof(indices));
      CHECK(ib);
      pipe_buffer_write(pctx, ib, 0, sizeof(indices), indices);

      const unsigned mdi_count = 1600;
      const unsigned mdi_clean_link_count = 2200;
      uint32_t (*mdi_args)[5] = calloc(mdi_clean_link_count, sizeof(*mdi_args));
      CHECK(mdi_args);
      for (unsigned i = 0; i < mdi_clean_link_count; ++i) {
         mdi_args[i][0] = 3;
         mdi_args[i][1] = 1;
         mdi_args[i][2] = 0;
         mdi_args[i][3] = 0;
         mdi_args[i][4] = i;
      }
      struct pipe_resource *mdi_buf =
         pipe_buffer_create(pctx->screen, PIPE_BIND_COMMAND_ARGS_BUFFER,
                            PIPE_USAGE_DEFAULT,
                            mdi_clean_link_count * sizeof(*mdi_args));
      CHECK(mdi_buf);
      pipe_buffer_write(pctx, mdi_buf, 0,
                        mdi_clean_link_count * sizeof(*mdi_args), mdi_args);
      free(mdi_args);

      struct pipe_draw_info mdi_info = {
         .mode = MESA_PRIM_TRIANGLES,
         .index_size = 2,
         .index.resource = ib,
         .instance_count = 1,
      };
      struct pipe_draw_indirect_info mdi_clean = {
         .buffer = mdi_buf,
         .draw_count = 8,
         .stride = 20,
      };
      emits = ctx->perf.vs_emits;
      uint64_t draws_before = ctx->perf.draws;
      scissor_before = ctx->batch->scissor.size;
      pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_clean, &indirect_draw, 1);
      CHECK(ctx->perf.draws == draws_before + 8);
      if (native_agx3) {
         CHECK(ctx->perf.vs_emits <= emits + 1);
         CHECK(ctx->batch->scissor.size <= scissor_before + AGX_SCISSOR_LENGTH);
         emits = ctx->perf.vs_emits;
         scissor_before = ctx->batch->scissor.size;
         pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_clean, &indirect_draw, 1);
         CHECK(ctx->perf.vs_emits == emits);
         CHECK(ctx->batch->scissor.size == scissor_before);
         CHECK(check_last_indexed_indirect_draw(
            ctx, agx_map_gpu(agx_resource(mdi_buf)) + 7u * 20u, false, 0));
      }

      /* Include a vertex attribute so draw 0 uploads AGX_SYSVAL_TABLE_VS into
       * batch->pool, verifying that per-draw draw_id slices stay isolated from
       * static pool tables and reuse the Ioto template cache across 1,600
       * sub-draws instead of thrashing it. */
      const float verts[6] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
      struct pipe_resource *vb =
         pipe_buffer_create(pctx->screen, PIPE_BIND_VERTEX_BUFFER,
                            PIPE_USAGE_DEFAULT, sizeof(verts));
      CHECK(vb);
      pipe_buffer_write(pctx, vb, 0, sizeof(verts), verts);
      struct pipe_vertex_buffer vbuf = {
         .buffer.resource = vb,
         .buffer_offset = 0,
      };
      struct pipe_vertex_element velem = {
         .src_offset = 0,
         .src_format = PIPE_FORMAT_R32G32_FLOAT,
         .src_stride = 2 * sizeof(float),
         .vertex_buffer_index = 0,
      };
      void *mdi_ve = pctx->create_vertex_elements_state(pctx, 1, &velem);
      CHECK(mdi_ve);
      pctx->set_vertex_buffers(pctx, 1, &vbuf);
      pctx->bind_vertex_elements_state(pctx, mdi_ve);

      void *mdi_vs = create_vs(pctx, true, true, true);
      CHECK(mdi_vs);
      pctx->bind_vs_state(pctx, mdi_vs);
      struct pipe_draw_indirect_info mdi_drawid = {
         .buffer = mdi_buf,
         .draw_count = mdi_count,
         .stride = 20,
      };
      emits = ctx->perf.vs_emits;
      draws_before = ctx->perf.draws;
      scissor_before = ctx->batch->scissor.size;
      struct agx_bo *start_vdm_bo = ctx->batch->vdm.bo;
      uint8_t *start_vdm_end = ctx->batch->vdm.end;
      pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_drawid, &indirect_draw, 1);
      CHECK(ctx->perf.draws == draws_before + mdi_count);
      CHECK(ctx->perf.vs_emits == emits + mdi_count);
      CHECK(ctx->vs->b.info.uses_draw_id && ctx->linked.vs->uses_base_param);
      if (native_agx3) {
         CHECK(ctx->batch->vdm.bo != start_vdm_bo ||
               ctx->batch->vdm.end != start_vdm_end);
         CHECK(ctx->batch->g17p_bind_tail_latched &&
               ctx->batch->g17p_have_latched_vs);
         /* Only draw 0 + the VDM STREAM_LINK re-latches emit bind tails. */
         CHECK((ctx->batch->scissor.size - scissor_before) <=
               4 * AGX_SCISSOR_LENGTH);
         /* Draw 0 + isolated draw_id shape must reuse templates across all
          * 1,600 sub-draws without thrashing. */
         CHECK(ctx->linked.vs->g17p_graphics_templates_generated >= 1 &&
               ctx->linked.vs->g17p_graphics_templates_generated <= 4);
         CHECK(check_last_indexed_indirect_draw(
            ctx,
            agx_map_gpu(agx_resource(mdi_buf)) + (uint64_t)(mdi_count - 1) * 20,
            true, mdi_count - 1));

         /* Large stride + non-zero base offset: verify exact last-draw
          * addressing on the fast path, a forced per-draw fallback starting in
          * an unlatched post-STREAM_LINK chunk and crossing another
          * STREAM_LINK, a mid-MDI fast -> STREAM_LINK -> flush fallback ->
          * fast recovery, and runtime overflow/range clipping. */
         const uint32_t large_offset = 256u;
         const uint32_t large_stride = 4096u;
         const unsigned large_count = 1600u;
         const uint32_t large_size =
            large_offset + (large_count - 1u) * large_stride + 20u;
         struct pipe_resource *large_buf =
            pipe_buffer_create(pctx->screen, PIPE_BIND_COMMAND_ARGS_BUFFER,
                               PIPE_USAGE_DEFAULT, large_size);
         CHECK(large_buf);
         struct pipe_transfer *large_xfer = NULL;
         uint8_t *large_map = pipe_buffer_map(
            pctx, large_buf, PIPE_MAP_WRITE | PIPE_MAP_DISCARD_WHOLE_RESOURCE,
            &large_xfer);
         CHECK(large_map && large_xfer);
         for (unsigned i = 0; i < large_count; ++i) {
            const uint32_t cmd[5] = {3, 1, 0, 0, i};
            memcpy(large_map + large_offset + (size_t)i * large_stride, cmd,
                   sizeof(cmd));
         }
         pipe_buffer_unmap(pctx, large_xfer);
         const uint64_t large_base_va = agx_map_gpu(agx_resource(large_buf));
         const uint64_t large_last_va =
            large_base_va + large_offset +
            (uint64_t)(large_count - 1u) * large_stride;
         struct pipe_draw_indirect_info mdi_large = {
            .buffer = large_buf,
            .offset = large_offset,
            .draw_count = large_count,
            .stride = large_stride,
         };

         /* 1) Cross a 64 KiB VDM STREAM_LINK with state-clean body-only draws
          * so the new VDM chunk has g17p_bind_tail_latched == false, then
          * immediately run a forced per-draw fallback (AGX_DBG_DIRTY) in the
          * same batch across another STREAM_LINK. */
         pctx->bind_vs_state(pctx, s.vs);
         struct pipe_draw_indirect_info mdi_clean_link = {
            .buffer = mdi_buf,
            .draw_count = mdi_clean_link_count,
            .stride = 20,
         };
         start_vdm_bo = ctx->batch->vdm.bo;
         start_vdm_end = ctx->batch->vdm.end;
         draws_before = ctx->perf.draws;
         pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_clean_link, &indirect_draw, 1);
         CHECK(ctx->perf.draws == draws_before + mdi_clean_link_count);
         CHECK(ctx->batch->vdm.bo != start_vdm_bo ||
               ctx->batch->vdm.end != start_vdm_end);
         CHECK(!ctx->batch->g17p_bind_tail_latched);
         CHECK(check_last_indexed_indirect_draw(
            ctx,
            agx_map_gpu(agx_resource(mdi_buf)) +
               (uint64_t)(mdi_clean_link_count - 1u) * 20u,
            false, 0));

         pctx->bind_vs_state(pctx, mdi_vs);
         struct agx_device *dev = agx_device(pctx->screen);
         const unsigned saved_debug = dev->debug;
         struct agx_batch *same_batch = ctx->batch;
         start_vdm_bo = ctx->batch->vdm.bo;
         start_vdm_end = ctx->batch->vdm.end;
         emits = ctx->perf.vs_emits;
         draws_before = ctx->perf.draws;
         scissor_before = ctx->batch->scissor.size;
         dev->debug |= AGX_DBG_DIRTY;
         pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_large, &indirect_draw, 1);
         dev->debug = saved_debug;
         CHECK(ctx->batch == same_batch);
         CHECK(ctx->perf.draws == draws_before + large_count);
         CHECK(ctx->perf.vs_emits == emits + large_count);
         CHECK(ctx->batch->vdm.bo != start_vdm_bo ||
               ctx->batch->vdm.end != start_vdm_end);
         CHECK(ctx->batch->g17p_bind_tail_latched &&
               ctx->batch->g17p_have_latched_vs);
#ifndef NDEBUG
         CHECK(ctx->batch->scissor.size - scissor_before ==
               large_count * AGX_SCISSOR_LENGTH);
#else
         CHECK(ctx->batch->scissor.size - scissor_before >=
               2 * AGX_SCISSOR_LENGTH);
#endif
         CHECK(check_last_indexed_indirect_draw(ctx, large_last_va, true,
                                                large_count - 1u));

         /* 2) Large stride + non-zero offset on the fast path across a
          * STREAM_LINK, verifying last-draw parity with the fallback path. */
         start_vdm_bo = ctx->batch->vdm.bo;
         start_vdm_end = ctx->batch->vdm.end;
         emits = ctx->perf.vs_emits;
         draws_before = ctx->perf.draws;
         scissor_before = ctx->batch->scissor.size;
         pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_large, &indirect_draw, 1);
         CHECK(ctx->perf.draws == draws_before + large_count);
         CHECK(ctx->perf.vs_emits == emits + large_count);
         CHECK(ctx->batch->vdm.bo != start_vdm_bo ||
               ctx->batch->vdm.end != start_vdm_end);
         CHECK(ctx->batch->g17p_bind_tail_latched &&
               ctx->batch->g17p_have_latched_vs);
         CHECK((ctx->batch->scissor.size - scissor_before) <=
               4 * AGX_SCISSOR_LENGTH);
         CHECK(check_last_indexed_indirect_draw(ctx, large_last_va, true,
                                                large_count - 1u));

         /* 3) Mid-MDI fast path -> VDM STREAM_LINK -> forced fallback on batch
          * flush (> 100,000 draws) -> fast path recovery through the last draw. */
         CHECK(ctx->batch->draws < 98700u);
         /* The flushed batch's slot can be reused, so compare its sequence
          * number rather than its pointer. */
         const uint64_t pre_flush_seqnum = ctx->batch->seqnum;
         ctx->batch->draws = 98700u;
         emits = ctx->perf.vs_emits;
         draws_before = ctx->perf.draws;
         pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_large, &indirect_draw, 1);
         CHECK(!ctx->any_faults);
         CHECK(ctx->batch && ctx->batch->seqnum != pre_flush_seqnum);
         CHECK(ctx->batch->draws > 0 && ctx->batch->draws < large_count);
         CHECK(ctx->perf.draws == draws_before + large_count);
         CHECK(ctx->perf.vs_emits == emits + large_count);
         CHECK(ctx->batch->g17p_bind_tail_latched &&
               ctx->batch->g17p_have_latched_vs);
         CHECK(check_last_indexed_indirect_draw(ctx, large_last_va, true,
                                                large_count - 1u));

         /* 4) Runtime overflow and buffer-range clipping on both fast path and
          * forced fallback: 32-bit multiply wraparound (2 * 0x80000000u) and
          * trailing out-of-range draws must stop cleanly without emitting
          * wrapped or unowned addresses. */
         uint32_t small_args[3][5] = {
            {3, 1, 0, 0, 10},
            {3, 1, 0, 0, 11},
            {3, 1, 0, 0, 12},
         };
         struct pipe_resource *small_buf =
            pipe_buffer_create(pctx->screen, PIPE_BIND_COMMAND_ARGS_BUFFER,
                               PIPE_USAGE_DEFAULT, 64u);
         CHECK(small_buf);
         pipe_buffer_write(pctx, small_buf, 0, sizeof(small_args), small_args);
         const uint64_t small_va = agx_map_gpu(agx_resource(small_buf));

         struct pipe_draw_indirect_info mdi_wrap = {
            .buffer = small_buf,
            .offset = 0,
            .draw_count = 4,
            .stride = 0x80000000u,
         };
         struct pipe_draw_indirect_info mdi_oob = {
            .buffer = small_buf,
            .offset = 0,
            .draw_count = 5,
            .stride = 20u,
         };
         for (unsigned mode = 0; mode < 2; ++mode) {
            if (mode == 1)
               dev->debug |= AGX_DBG_DIRTY;
            draws_before = ctx->perf.draws;
            pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_wrap, &indirect_draw, 1);
            CHECK(ctx->perf.draws == draws_before + 1);
            CHECK(check_last_indexed_indirect_draw(ctx, small_va, true, 0));

            draws_before = ctx->perf.draws;
            pctx->draw_vbo(pctx, &mdi_info, 0, &mdi_oob, &indirect_draw, 1);
            CHECK(ctx->perf.draws == draws_before + 3);
            CHECK(check_last_indexed_indirect_draw(ctx, small_va + 40u, true,
                                                   2));
            dev->debug = saved_debug;
         }

         pipe_resource_reference(&small_buf, NULL);
         pipe_resource_reference(&large_buf, NULL);
      }
      pctx->bind_vs_state(pctx, s.vs);
      pctx->delete_vs_state(pctx, mdi_vs);
      pctx->bind_vertex_elements_state(pctx, s.ve);
      pctx->delete_vertex_elements_state(pctx, mdi_ve);
      pctx->set_vertex_buffers(pctx, 0, NULL);
      pipe_resource_reference(&vb, NULL);
      pipe_resource_reference(&mdi_buf, NULL);
      pipe_resource_reference(&ib, NULL);
   }

   /* Returning to an existing batch dirties state, including its VS. */
   struct agx_batch *first = ctx->batch;
   struct pipe_framebuffer_state fb = {.width = 32, .height = 16, .layers = 1};
   pctx->set_framebuffer_state(pctx, &fb);
   emits = ctx->perf.vs_emits;
   draw(pctx);
   CHECK(ctx->batch != first && ctx->perf.vs_emits == emits + 1);
   fb.width = 16;
   pctx->set_framebuffer_state(pctx, &fb);
   draw(pctx);
   CHECK(ctx->batch == first && ctx->perf.vs_emits == emits + 2);

   pctx->flush(pctx, NULL, PIPE_FLUSH_END_OF_FRAME);
   CHECK(!ctx->any_faults);
   CHECK(ctx->perf.frame == 1 && ctx->perf.draws == 0 &&
         ctx->perf.vs_emits == 0);
   draw(pctx);
   CHECK(ctx->perf.vs_emits == 1 && ctx->perf.draws == 1);
   pctx->destroy_query(pctx, occlusion);

   if (native_agx3) {
      /* Verify HAL200 compute Ioto loader template caching for both
       * precompiled clear_buffer dispatches and API compute dispatches. */
      struct agx_device *dev = agx_device(pctx->screen);
      const unsigned saved_debug = dev->debug;
      struct pipe_resource *cbuf =
         pipe_buffer_create(pctx->screen, PIPE_BIND_SHADER_BUFFER,
                            PIPE_USAGE_DEFAULT, 4096u);
      CHECK(cbuf);
      const uint32_t fill_val[4] = {0x11223344u, 0x55667788u, 0x99aabbcdu,
                                    0xdeadbeefu};
      const unsigned precomp_before = ctx->g17p_precomp_templates_generated;
      dev->debug |= AGX_DBG_VERIFY;
      for (unsigned k = 0; k < 16; ++k) {
         pctx->clear_buffer(pctx, cbuf, 0, 4096u, fill_val, sizeof(fill_val));
         pctx->flush(pctx, NULL, 0);
      }
      CHECK(ctx->g17p_precomp_templates_generated > precomp_before &&
            ctx->g17p_precomp_templates_generated <= precomp_before + 3);

      nir_builder cb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "cs-tpl");
      cb.shader->info.workgroup_size[0] = 32;
      cb.shader->info.workgroup_size[1] = 1;
      cb.shader->info.workgroup_size[2] = 1;
      cb.shader->info.num_ssbos = 1;
      nir_def *gid = nir_channel(&cb, nir_load_global_invocation_id(&cb, 32), 0);
      nir_store_ssbo(&cb, gid, nir_imm_int(&cb, 0),
                     nir_ishl_imm(&cb, gid, 2), .write_mask = 0x1,
                     .align_mul = 4);
      struct pipe_compute_state cs_state = {
         .ir_type = PIPE_SHADER_IR_NIR,
         .prog = cb.shader,
      };
      void *cs_cso = pctx->create_compute_state(pctx, &cs_state);
      CHECK(cs_cso);
      pctx->bind_compute_state(pctx, cs_cso);
      struct pipe_shader_buffer sb = {
         .buffer = cbuf,
         .buffer_offset = 0,
         .buffer_size = 4096u,
      };
      pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 1, &sb, 0x1);
      struct pipe_grid_info grid = {
         .block = {32, 1, 1},
         .grid = {4, 1, 1},
      };
      for (unsigned k = 0; k < 16; ++k) {
         pctx->launch_grid(pctx, &grid);
         pctx->flush(pctx, NULL, 0);
      }
      struct agx_uncompiled_shader *cs_so = cs_cso;
      CHECK(cs_so->cached_cs[0] != NULL);
      CHECK(cs_so->cached_cs[0]->g17p_compute_templates_generated >= 1 &&
            cs_so->cached_cs[0]->g17p_compute_templates_generated <= 3);
      dev->debug = saved_debug;
      pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 1, NULL, 0);
      pctx->bind_compute_state(pctx, NULL);
      pctx->delete_compute_state(pctx, cs_cso);
      pipe_resource_reference(&cbuf, NULL);
   }

   finish_draw(pctx, &s);
   return true;
}

static bool
check_queries(struct pipe_context *pctx)
{
   struct agx_context *ctx = agx_context(pctx);
   struct draw_state s;
   CHECK(setup_draw(pctx, &s));
   struct pipe_query *pq =
      pctx->create_query(pctx, PIPE_QUERY_OCCLUSION_COUNTER, 0);
   CHECK(pq);
   struct agx_query *q = (void *)pq;
   union pipe_query_result result;
   CHECK(pctx->begin_query(pctx, pq) && pctx->end_query(pctx, pq));
   CHECK(pctx->get_query_result(pctx, pq, false, &result) && result.u64 == 0);

   CHECK(pctx->begin_query(pctx, pq));
   draw(pctx);
   CHECK(pctx->end_query(pctx, pq));
   struct agx_batch *batch = ctx->batch;
   CHECK(batch && agx_batch_is_active(batch));
   unsigned slot = agx_batch_idx(batch);
   uint64_t generation = ctx->batches.generation[slot];
   unsigned old_polls = polls, old_waits = waits;
   result.u64 = 0xdeadbeef;
   pctx->render_condition(pctx, pq, false, PIPE_RENDER_COND_NO_WAIT);
   CHECK(agx_render_condition_check_inner(ctx));
   pctx->render_condition(pctx, NULL, false, PIPE_RENDER_COND_WAIT);
   CHECK(agx_batch_is_active(batch) && waits == old_waits &&
         polls == old_polls);

   /* Availability must make progress even when the app never calls Flush.
    * Only the unsubmitted writer is submitted; the result stays untouched. */
   busy = true;
   CHECK(!pctx->get_query_result(pctx, pq, false, &result));
   CHECK(agx_batch_is_submitted(batch) && result.u64 == 0xdeadbeef);
   CHECK(waits == old_waits && ctx->batches.generation[slot] == generation);
   uint64_t seqid = ctx->flush_last_seqid;
   old_polls = polls;
   CHECK(!pctx->get_query_result(pctx, pq, false, &result));
   CHECK(result.u64 == 0xdeadbeef && polls == old_polls + 1 &&
         waits == old_waits && ctx->flush_last_seqid == seqid);
   CHECK(ctx->batches.generation[slot] == generation &&
         ctx->perf.query_waits == 0);

   /* The next poll retires the batch without a blocking wait. */
   *(uint64_t *)q->ptr.cpu = 7;
   busy = false;
   CHECK(pctx->get_query_result(pctx, pq, false, &result) && result.u64 == 7);
   CHECK(ctx->batches.generation[slot] != generation && waits == old_waits);
   CHECK(!agx_batch_is_submitted(batch) && ctx->perf.query_waits == 0);
   CHECK(pctx->get_query_result(pctx, pq, true, &result) && waits == old_waits);

   /* A reused slot must not make an already retired query busy again. */
   draw(pctx);
   CHECK(pctx->get_query_result(pctx, pq, false, &result) && result.u64 == 7);
   CHECK(ctx->batch && agx_batch_is_active(ctx->batch));

   /* A blocking read submits the active writer exactly when necessary. */
   CHECK(pctx->begin_query(pctx, pq));
   draw(pctx);
   CHECK(pctx->end_query(pctx, pq));
   busy = true;
   CHECK(pctx->get_query_result(pctx, pq, true, &result));
   CHECK(waits == old_waits + 1 && ctx->perf.query_waits == 1);
   CHECK(pctx->get_query_result(pctx, pq, true, &result));
   CHECK(waits == old_waits + 1 && ctx->perf.query_waits == 1);
   pctx->destroy_query(pctx, pq);

   /* Timer queries can have several writer batches. Retiring one must not
    * publish a partial CPU result or miss the remaining writer. */
   pq = pctx->create_query(pctx, PIPE_QUERY_TIME_ELAPSED, 0);
   CHECK(pq && pctx->begin_query(pctx, pq));
   draw(pctx);
   struct agx_batch *first = ctx->batch;
   unsigned first_slot = agx_batch_idx(first);
   uint64_t first_generation = ctx->batches.generation[first_slot];
   struct pipe_framebuffer_state fb = {.width = 32, .height = 16, .layers = 1};
   pctx->set_framebuffer_state(pctx, &fb);
   draw(pctx);
   struct agx_batch *second = ctx->batch;
   CHECK(second != first && pctx->end_query(pctx, pq));
   unsigned second_slot = agx_batch_idx(second);
   uint64_t second_generation = ctx->batches.generation[second_slot];
   busy = true;
   pctx->flush(pctx, NULL, 0);
   CHECK(agx_batch_is_submitted(first) && agx_batch_is_submitted(second));
   struct agx_timestamps *timestamps = agx_bo_map(ctx->timestamps);
   timestamps[first_slot].vtx_start = 100;
   timestamps[first_slot].frag_end = 200;
   timestamps[second_slot].vtx_start = 150;
   timestamps[second_slot].frag_end = 500;
   old_waits = waits;
   ready_polls = 1;
   result.u64 = 0xdeadbeef;
   CHECK(!pctx->get_query_result(pctx, pq, false, &result));
   CHECK(result.u64 == 0xdeadbeef && waits == old_waits);
   CHECK((ctx->batches.generation[first_slot] != first_generation) !=
         (ctx->batches.generation[second_slot] != second_generation));
   busy = false;
   CHECK(pctx->get_query_result(pctx, pq, false, &result));
   CHECK(ctx->batches.generation[first_slot] != first_generation &&
         ctx->batches.generation[second_slot] != second_generation);
   CHECK(waits == old_waits);
   CHECK(result.u64 == agx_gpu_timestamp_to_ns(agx_device(pctx->screen), 400));

   /* The CPU timer-query buffer fallback must also honor PIPE_QUERY_WAIT. */
   struct pipe_resource *buffer =
      pipe_buffer_create(pctx->screen, PIPE_BIND_QUERY_BUFFER,
                         PIPE_USAGE_DEFAULT, sizeof(uint64_t));
   CHECK(buffer && pctx->begin_query(pctx, pq));
   draw(pctx);
   batch = ctx->batch;
   CHECK(pctx->end_query(pctx, pq));
   uint64_t marker = 0xdeadbeef, copied;
   pipe_buffer_write(pctx, buffer, 0, sizeof(marker), &marker);
   busy = true;
   pctx->get_query_result_resource(pctx, pq, 0, PIPE_QUERY_TYPE_U64, 0, buffer,
                                   0);
   pipe_buffer_read(pctx, buffer, 0, sizeof(copied), &copied);
   CHECK(copied == marker && agx_batch_is_submitted(batch) &&
         waits == old_waits);
   seqid = ctx->flush_last_seqid;
   pctx->get_query_result_resource(pctx, pq, 0, PIPE_QUERY_TYPE_U64, -1, buffer,
                                   0);
   pipe_buffer_read(pctx, buffer, 0, sizeof(copied), &copied);
   CHECK(copied == 0 && ctx->flush_last_seqid == seqid && waits == old_waits);
   timestamps[agx_batch_idx(batch)].vtx_start = 10;
   timestamps[agx_batch_idx(batch)].frag_end = 30;
   busy = false;
   pctx->get_query_result_resource(pctx, pq, 0, PIPE_QUERY_TYPE_U64, -1, buffer,
                                   0);
   pipe_buffer_read(pctx, buffer, 0, sizeof(copied), &copied);
   CHECK(copied == 1 && waits == old_waits);
   pctx->get_query_result_resource(pctx, pq, 0, PIPE_QUERY_TYPE_U64, 0, buffer,
                                   0);
   pipe_buffer_read(pctx, buffer, 0, sizeof(copied), &copied);
   CHECK(copied == agx_gpu_timestamp_to_ns(agx_device(pctx->screen), 20));
   CHECK(waits == old_waits);
   pipe_resource_reference(&buffer, NULL);
   pctx->destroy_query(pctx, pq);
   finish_draw(pctx, &s);
   return true;
}

int
main(int argc, char **argv)
{
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return 77;
   static const driOptionDescription options[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, cache;
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {.options = &cache,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   if (!screen)
      return 77;
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   if (!pctx)
      return 1;
   glsl_type_singleton_init_or_ref();
   bool ok = argc == 2 && (strcmp(argv[1], "vs") == 0 ? check_vs(pctx)
                                                      : check_queries(pctx));
   busy = false;
   pctx->flush(pctx, NULL, 0);
   ok &= !agx_context(pctx)->any_faults;
   pctx->destroy(pctx);
   screen->destroy(screen);
   glsl_type_singleton_decref();
   close(fd);
   return ok ? 0 : 1;
}
