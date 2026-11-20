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
create_vs(struct pipe_context *pctx, bool vertex_id)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "VS dirty tracking");
   nir_def *pos = nir_imm_vec4(&b, 0, 0, 0, 1);
   if (vertex_id) {
      nir_def *id = nir_u2f32(
         &b, nir_iadd(&b, nir_load_vertex_id(&b), nir_load_base_instance(&b)));
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
   s->vs = create_vs(pctx, false);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "VS dirty tracking fragment");
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

   /* A different shader and per-draw base parameters must refresh VS. */
   void *id_vs = create_vs(pctx, true);
   CHECK(id_vs);
   pctx->bind_vs_state(pctx, id_vs);
   emits = ctx->perf.vs_emits;
   draw(pctx);
   CHECK(ctx->perf.vs_emits == emits + 1 && ctx->skipped_no_program == 0);
   CHECK(ctx->linked.vs->uses_base_param);
   draw(pctx);
   CHECK(ctx->perf.vs_emits == emits + 2);
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
      pctx->draw_vbo(pctx, &indirect_info, 0, &indirect, &indirect_draw, 1);
      CHECK(ctx->perf.vs_emits == emits);
   }
   pipe_resource_reference(&indirect_buffer, NULL);

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
