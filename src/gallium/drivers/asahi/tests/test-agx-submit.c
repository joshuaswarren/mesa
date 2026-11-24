/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Exercise the real Gallium batch/BO lifetime and submit builder, replacing
 * only the kernel submit operation. The shim supplies device setup, not GPU
 * execution. Inspect barriers and fence placement at the ioctl boundary. */
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include "util/driconf.h"
#include "util/xmlconfig.h"
#include "agx_state.h"
#include "agx_fence.h"
#include "agx_public.h"

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static unsigned calls, computes, renders, expected_inputs;
static bool fail_submit;
static uint32_t first_output;

static int
inspect_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
               struct agx_submit_virt *virt)
{
   calls++;
   CHECK(submit->in_sync_count == expected_inputs);
   CHECK(submit->out_sync_count == 1 || submit->out_sync_count == 3);
   const struct drm_asahi_sync *syncs = (void *)(uintptr_t)submit->syncs;
   if (submit->out_sync_count == 1) {
      first_output = syncs[submit->in_sync_count].handle;
      CHECK(first_output);
   } else {
      CHECK(syncs[submit->in_sync_count].sync_type == DRM_ASAHI_SYNC_SYNCOBJ);
      CHECK(syncs[submit->in_sync_count + 1].sync_type ==
            DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ);
      CHECK(syncs[submit->in_sync_count + 1].timeline_value != 0);
      CHECK(syncs[submit->in_sync_count + 2].sync_type == DRM_ASAHI_SYNC_SYNCOBJ);
      if (first_output)
         CHECK(syncs[submit->in_sync_count - 1].handle == first_output);
   }
   unsigned prefix[2] = {0};
   const uint8_t *data = (void *)(uintptr_t)submit->cmdbuf;
   size_t cursor = 0;
   while (cursor < submit->cmdbuf_size) {
      CHECK(submit->cmdbuf_size - cursor >= sizeof(struct drm_asahi_cmd_header));
      const struct drm_asahi_cmd_header *h = (void *)(data + cursor);
      CHECK(h->size <= submit->cmdbuf_size - cursor - sizeof(*h));
      if (h->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         CHECK(h->vdm_barrier == 0 && h->cdm_barrier == 0);
         prefix[1]++;
         computes++;
      } else if (h->cmd_type == DRM_ASAHI_CMD_RENDER) {
         CHECK(h->vdm_barrier == (prefix[1] ? DRM_ASAHI_BARRIER_NONE : 0));
         CHECK(h->cdm_barrier == (prefix[1] ? 1 : 0));
         CHECK(h->size == agx_render_command_size(dev));
         prefix[0]++;
         renders++;
      } else {
         CHECK(h->cmd_type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS);
         CHECK(h->vdm_barrier == DRM_ASAHI_BARRIER_NONE &&
               h->cdm_barrier == DRM_ASAHI_BARRIER_NONE);
      }
      cursor += sizeof(*h) + h->size;
   }
   CHECK(prefix[0] <= 1 && prefix[1] <= 1);
   if (submit->out_sync_count == 1)
      expected_inputs++;
   return fail_submit ? -EINVAL : 0;
}

static void
run_batch(struct pipe_context *pctx, bool compute, bool render, bool split,
          bool fail, unsigned bo_count)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   struct agx_screen *screen = agx_screen(pctx->screen);
   dev->params.max_commands_per_submission = split ? 1 : 64;
   struct agx_batch *batch = agx_get_batch(ctx);

   /* Sparse, large handles used to allocate gigabytes or overflow growth.
    * These BOs retain a caller reference, so retirement never frees them. */
   struct agx_va va = {.addr = 0x4000};
   struct agx_bo bos[129] = {0};
   CHECK(bo_count <= ARRAY_SIZE(bos));
   for (unsigned i = 0; i < bo_count; i++) {
      bos[i].handle = i ? UINT32_MAX - i : 7;
      bos[i].refcnt = 1;
      bos[i].va = &va;
      bos[i].writer = agx_bo_writer(ctx->queue_id + 1, ctx->dummy_syncobj);
      agx_batch_add_bo(batch, &bos[i]);
      agx_batch_add_bo(batch, &bos[i]);
      CHECK(bos[i].refcnt == 2);
   }
   CHECK(batch->bo_list.set->entries == bo_count);
   CHECK(util_dynarray_num_elements(&batch->bo_list.list, struct agx_bo *) == bo_count);
   CHECK(batch->bo_list.list.capacity <= MAX2(64, bo_count * 2 * sizeof(void *)));
   CHECK(batch->bo_list.set->size < MAX2(32, bo_count * 4));

   /* BO inputs, an explicit acquire, and a cross-context timeline input
    * fill the capacity including both outputs and the split fallback fence. */
   ctx->in_sync_fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
   CHECK(ctx->in_sync_fd >= 0);
   p_atomic_set(&screen->flush_wait_seqid, p_atomic_inc_return(&screen->flush_cur_seqid));
   calls = computes = renders = first_output = 0;
   expected_inputs = bo_count + 2;
   fail_submit = fail;
   memset(&ctx->submit_perf, 0, sizeof(ctx->submit_perf));
   struct drm_asahi_cmd_compute c = {0};
   struct drm_asahi_cmd_render r = {0};
   uint64_t last_seqid = ctx->flush_last_seqid;
   agx_batch_submit(ctx, batch, compute ? &c : NULL, render ? &r : NULL);
   /* A HAL200 kernel limited to one command splits a mixed batch in two
    * (split_compute in agx_batch_submit()). */
   const bool split_compute =
      split && compute && render && agx_chip_native_hal200(dev->chip);
   CHECK(calls == (split_compute ? 2 : 1));
   CHECK(computes == compute && renders == render);
   CHECK(ctx->submit_perf.batches == 1);
   CHECK(ctx->submit_perf.mixed_batches == (compute && render));
   CHECK(ctx->submit_perf.submits == calls);
   CHECK(ctx->submit_perf.failures == fail);
   CHECK(ctx->any_faults == fail);
   CHECK(fail ? ctx->flush_last_seqid == last_seqid :
                ctx->flush_last_seqid > last_seqid);
   agx_sync_all(ctx, "submit test retirement");
   for (unsigned i = 0; i < bo_count; i++)
      CHECK(bos[i].refcnt == 1);
   pctx->flush(pctx, NULL, 0);
   CHECK(ctx->submit_perf.submits == 0 && ctx->submit_perf.batches == 0);
}

struct coalesced_record {
   unsigned in_sync_count;
   unsigned out_sync_count;
   unsigned renders;
   unsigned computes;
   unsigned att_sets;
   unsigned att_clears;
   uint16_t render_vdm_barriers[16];
   uint16_t render_cdm_barriers[16];
   uint64_t render_flags[16];
   uint16_t compute_vdm_barriers[16];
   uint16_t compute_cdm_barriers[16];
};

static struct coalesced_record coalesced_log[8];
static unsigned coalesced_calls;
static bool coalesced_fail;

/* The no-op drm-shim stubs DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD; provide a dummy
 * file descriptor so agx_fence_create() and fence_server_sync() can run. */
int
drmSyncobjExportSyncFile(int fd, uint32_t handle, int *sync_file_fd)
{
   CHECK(handle);
   int dup_fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
   CHECK(dup_fd >= 0);
   *sync_file_fd = dup_fd;
   return 0;
}

static int
inspect_coalesced_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
                         struct agx_submit_virt *virt)
{
   CHECK(coalesced_calls < ARRAY_SIZE(coalesced_log));
   struct coalesced_record *rec = &coalesced_log[coalesced_calls++];
   memset(rec, 0, sizeof(*rec));
   rec->in_sync_count = submit->in_sync_count;
   rec->out_sync_count = submit->out_sync_count;

   const uint8_t *data = (void *)(uintptr_t)submit->cmdbuf;
   size_t cursor = 0;
   while (cursor < submit->cmdbuf_size) {
      CHECK(submit->cmdbuf_size - cursor >= sizeof(struct drm_asahi_cmd_header));
      const struct drm_asahi_cmd_header *h = (void *)(data + cursor);
      CHECK(h->size <= submit->cmdbuf_size - cursor - sizeof(*h));
      if (h->cmd_type == DRM_ASAHI_CMD_RENDER) {
         if (rec->renders < ARRAY_SIZE(rec->render_vdm_barriers)) {
            const struct drm_asahi_cmd_render *rc = (const void *)(h + 1);
            rec->render_vdm_barriers[rec->renders] = h->vdm_barrier;
            rec->render_cdm_barriers[rec->renders] = h->cdm_barrier;
            rec->render_flags[rec->renders] = rc->flags;
         }
         rec->renders++;
      } else if (h->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         if (rec->computes < ARRAY_SIZE(rec->compute_vdm_barriers)) {
            rec->compute_vdm_barriers[rec->computes] = h->vdm_barrier;
            rec->compute_cdm_barriers[rec->computes] = h->cdm_barrier;
         }
         rec->computes++;
      } else if (h->cmd_type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS) {
         CHECK(h->vdm_barrier == DRM_ASAHI_BARRIER_NONE &&
               h->cdm_barrier == DRM_ASAHI_BARRIER_NONE);
         if (h->size > 0)
            rec->att_sets++;
         else
            rec->att_clears++;
      }
      cursor += sizeof(*h) + h->size;
   }
   return coalesced_fail ? -EINVAL : 0;
}

static void
run_coalesced_tests(struct pipe_screen *screen)
{
   struct agx_device *dev = agx_device(screen);
   dev->ops.submit = inspect_coalesced_submit;
   dev->params.max_commands_per_submission = 64;

   /* 1. Multi-batch render + compute coalescing + SET_FRAGMENT_ATTACHMENTS clear */
   {
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      struct agx_context *ctx = agx_context(pctx);
      struct pipe_resource templ = {
         .target = PIPE_TEXTURE_2D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 16,
         .height0 = 16,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_resource *tex = screen->resource_create(screen, &templ);
      CHECK(tex);

      struct pipe_framebuffer_state fb_att = {
         .width = 16,
         .height = 16,
         .nr_cbufs = 1,
         .cbufs[0] = {.texture = tex, .format = templ.format},
      };
      struct pipe_framebuffer_state fb_none = {.width = 16, .height = 16};

      coalesced_calls = 0;
      coalesced_fail = false;
      struct drm_asahi_cmd_render r = {0};
      struct drm_asahi_cmd_compute c = {0};

      /* Batch 0: render with attachment */
      pctx->set_framebuffer_state(pctx, &fb_att);
      struct agx_batch *b0 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b0, NULL, &r);
      CHECK(coalesced_calls == 0 && agx_batch_is_queued(b0));

      /* Batch 1: attachmentless render (must clear fragment attachments) */
      pctx->set_framebuffer_state(pctx, &fb_none);
      struct agx_batch *b1 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b1, NULL, &r);
      CHECK(coalesced_calls == 0 && agx_batch_is_queued(b1));

      /* Batch 2: compute-only batch */
      struct agx_batch *b2 = agx_get_compute_batch(ctx);
      agx_batch_queue_or_submit(ctx, b2, &c, NULL);
      CHECK(coalesced_calls == 0 && agx_batch_is_queued(b2));

      /* Batch 3: mixed compute + render batch */
      pctx->set_framebuffer_state(pctx, &fb_att);
      struct agx_batch *b3 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b3, &c, &r);
      CHECK(coalesced_calls == 0 && agx_batch_is_queued(b3));

      /* Flush drains all 4 batches in 1 ioctl. */
      pctx->flush(pctx, NULL, 0);
      CHECK(coalesced_calls == 1);
      CHECK(coalesced_log[0].renders == 3 && coalesced_log[0].computes == 2);
      CHECK(coalesced_log[0].out_sync_count == 4 + 2);
      CHECK(coalesced_log[0].att_sets == 2 && coalesced_log[0].att_clears == 1);
      CHECK(coalesced_log[0].render_vdm_barriers[0] == 0 &&
            coalesced_log[0].render_cdm_barriers[0] == 0);
      CHECK(coalesced_log[0].render_vdm_barriers[1] == 1 &&
            coalesced_log[0].render_cdm_barriers[1] == 0);
      CHECK(coalesced_log[0].compute_vdm_barriers[0] == 2 &&
            coalesced_log[0].compute_cdm_barriers[0] == 0);
      CHECK(coalesced_log[0].compute_vdm_barriers[1] == 2 &&
            coalesced_log[0].compute_cdm_barriers[1] == 1);
      CHECK(coalesced_log[0].render_vdm_barriers[2] == DRM_ASAHI_BARRIER_NONE &&
            coalesced_log[0].render_cdm_barriers[2] == 2);

      agx_sync_all(ctx, "coalesced test 1");
      pipe_resource_reference(&tex, NULL);
      pctx->destroy(pctx);
   }

   /* 2. Command-limit overflow splits cleanly across submissions. */
   {
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      struct agx_context *ctx = agx_context(pctx);
      dev->params.max_commands_per_submission = 3;
      struct pipe_framebuffer_state fb = {.width = 16, .height = 16};
      pctx->set_framebuffer_state(pctx, &fb);
      struct drm_asahi_cmd_render r = {0};

      coalesced_calls = 0;
      coalesced_fail = false;
      for (unsigned i = 0; i < 5; ++i) {
         struct agx_batch *b = agx_get_batch(ctx);
         agx_batch_queue_or_submit(ctx, b, NULL, &r);
      }
      /* 4th batch overflowed limit=3 and flushed first 3. */
      CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 3);
      pctx->flush(pctx, NULL, 0);
      CHECK(coalesced_calls == 2 && coalesced_log[1].renders == 2);
      dev->params.max_commands_per_submission = 64;
      agx_sync_all(ctx, "coalesced overflow");
      pctx->destroy(pctx);
   }

   /* 3. fence_server_sync drains prior queued batches before attaching in_sync. */
   {
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      struct agx_context *ctx = agx_context(pctx);
      struct pipe_framebuffer_state fb = {.width = 16, .height = 16};
      pctx->set_framebuffer_state(pctx, &fb);
      struct drm_asahi_cmd_render r = {0};

      coalesced_calls = 0;
      coalesced_fail = false;
      struct agx_batch *b0 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b0, NULL, &r);
      CHECK(coalesced_calls == 0);

      struct pipe_fence_handle *f = agx_fence_create(ctx);
      /* agx_fence_create drained b0 already. */
      CHECK(f && coalesced_calls == 1 && coalesced_log[0].renders == 1);

      struct agx_batch *b1 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b1, NULL, &r);
      CHECK(coalesced_calls == 1);
      pctx->fence_server_sync(pctx, f, 0);
      /* fence_server_sync must have drained b1 before recording in_sync_fd,
       * while a subsequent batch b2 receives the explicit acquire fence. */
      CHECK(coalesced_calls == 2 && coalesced_log[1].renders == 1);
      CHECK(coalesced_log[1].in_sync_count == coalesced_log[0].in_sync_count);

      struct agx_batch *b2 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b2, NULL, &r);
      pctx->flush(pctx, NULL, 0);
      CHECK(coalesced_calls == 3 && coalesced_log[2].renders == 1);
      CHECK(coalesced_log[2].in_sync_count ==
            coalesced_log[1].in_sync_count + 1);
      screen->fence_reference(screen, &f, NULL);
      agx_sync_all(ctx, "coalesced fence");
      pctx->destroy(pctx);
   }

   /* 4. Failed coalesced submission retires all queued batches cleanly. */
   {
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      struct agx_context *ctx = agx_context(pctx);
      struct pipe_framebuffer_state fb = {.width = 16, .height = 16};
      pctx->set_framebuffer_state(pctx, &fb);
      struct drm_asahi_cmd_render r = {0};

      coalesced_calls = 0;
      coalesced_fail = true;
      struct agx_batch *b0 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b0, NULL, &r);
      struct agx_batch *b1 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b1, NULL, &r);
      pctx->flush(pctx, NULL, 0);
      CHECK(coalesced_calls == 1);
      CHECK(ctx->any_faults);
      CHECK(!agx_batch_is_queued(b0) && !agx_batch_is_queued(b1));
      CHECK(!agx_batch_is_submitted(b0) && !agx_batch_is_submitted(b1));
      coalesced_fail = false;
      pctx->destroy(pctx);
   }

   /* 5. DRM_ASAHI_RENDER_FRAGMENT_BARRIERS and DRM_ASAHI_RENDER_FEW_PRIMITIVES */
   {
      const uint64_t saved_features = dev->params.features;
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      struct agx_context *ctx = agx_context(pctx);
      struct pipe_resource templ = {
         .target = PIPE_TEXTURE_2D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 16,
         .height0 = 16,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_resource *tex = screen->resource_create(screen, &templ);
      CHECK(tex);
      struct agx_resource *rsrc = agx_resource(tex);
      CHECK(!rsrc->render_written);

      struct pipe_framebuffer_state fb = {
         .width = 16,
         .height = 16,
         .nr_cbufs = 1,
         .cbufs[0] = {.texture = tex, .format = templ.format},
      };
      pctx->set_framebuffer_state(pctx, &fb);
      struct drm_asahi_cmd_render r = {0};
      struct drm_asahi_cmd_compute c = {0};

      dev->params.features =
         saved_features | DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS;
      coalesced_calls = 0;
      coalesced_fail = false;

      /* Render 0 writes tex; no VS render-written dependency -> flag set. */
      struct agx_batch *b0 = agx_get_batch(ctx);
      b0->initialized = true;
      agx_batch_writes(b0, rsrc, 0);
      CHECK(rsrc->render_written);
      agx_batch_queue_or_submit(ctx, b0, NULL, &r);

      /* Render 1 reads render-written tex in VS -> flag suppressed. */
      struct agx_batch *b1 = agx_get_batch(ctx);
      agx_batch_note_vs_read(b1, rsrc);
      CHECK(b1->vs_needs_render_barrier);
      agx_batch_queue_or_submit(ctx, b1, NULL, &r);

      /* Render 2 is mixed compute+render (vdm_barrier == NONE) -> flag clear. */
      struct agx_batch *b2 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b2, &c, &r);

      pctx->flush(pctx, NULL, 0);
      CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 3);
      CHECK(coalesced_log[0].render_flags[0] &
            DRM_ASAHI_RENDER_FRAGMENT_BARRIERS);
      CHECK(!(coalesced_log[0].render_flags[1] &
              DRM_ASAHI_RENDER_FRAGMENT_BARRIERS));
      CHECK(!(coalesced_log[0].render_flags[2] &
              DRM_ASAHI_RENDER_FRAGMENT_BARRIERS));

      /* Without kernel feature bit, flag stays clear. */
      dev->params.features =
         saved_features & ~DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS;
      coalesced_calls = 0;
      struct agx_batch *b3 = agx_get_batch(ctx);
      agx_batch_queue_or_submit(ctx, b3, NULL, &r);
      pctx->flush(pctx, NULL, 0);
      CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 1);
      CHECK(!(coalesced_log[0].render_flags[0] &
              DRM_ASAHI_RENDER_FRAGMENT_BARRIERS));

      /* FEW_PRIMITIVES via real agx_cmdbuf() path on native/M5 chips. */
      if (agx_chip_native_or_m5(dev->chip)) {
         union pipe_color_union color = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
         dev->params.features =
            saved_features | DRM_ASAHI_FEATURE_FEW_PRIMITIVES;

         coalesced_calls = 0;
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &color, 0.0, 0);
         ctx->batch->known_prims = 2;
         ctx->batch->prims_unknown = false;
         pctx->flush(pctx, NULL, 0);
         CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 1);
         CHECK(coalesced_log[0].render_flags[0] &
               DRM_ASAHI_RENDER_FEW_PRIMITIVES);

         coalesced_calls = 0;
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &color, 0.0, 0);
         ctx->batch->known_prims = 5;
         ctx->batch->prims_unknown = false;
         pctx->flush(pctx, NULL, 0);
         CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 1);
         CHECK(!(coalesced_log[0].render_flags[0] &
                 DRM_ASAHI_RENDER_FEW_PRIMITIVES));

         coalesced_calls = 0;
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &color, 0.0, 0);
         ctx->batch->known_prims = 1;
         ctx->batch->prims_unknown = true;
         pctx->flush(pctx, NULL, 0);
         CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 1);
         CHECK(!(coalesced_log[0].render_flags[0] &
                 DRM_ASAHI_RENDER_FEW_PRIMITIVES));

         dev->params.features =
            saved_features & ~DRM_ASAHI_FEATURE_FEW_PRIMITIVES;
         coalesced_calls = 0;
         pctx->clear(pctx, PIPE_CLEAR_COLOR0, 0xf, 0, NULL, &color, 0.0, 0);
         ctx->batch->known_prims = 1;
         ctx->batch->prims_unknown = false;
         pctx->flush(pctx, NULL, 0);
         CHECK(coalesced_calls == 1 && coalesced_log[0].renders == 1);
         CHECK(!(coalesced_log[0].render_flags[0] &
                 DRM_ASAHI_RENDER_FEW_PRIMITIVES));
      }

      dev->params.features = saved_features;
      agx_sync_all(ctx, "coalesced flags");
      pipe_resource_reference(&tex, NULL);
      pctx->destroy(pctx);
   }
}

int
main(void)
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
   struct agx_device *dev = agx_device(screen);
   dev->debug |= AGX_DBG_PERF;
   dev->ops.submit = inspect_submit;
   /* Each scenario gets a fresh context, including after injected failure. */
   for (unsigned scenario = 0; scenario < 7; scenario++) {
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      run_batch(pctx, scenario != 0, scenario != 1,
                scenario == 3, scenario == 4,
                scenario == 5 ? 129 : scenario == 6 ? 0 : 2);
      if (scenario == 2)
         run_batch(pctx, true, true, false, false, 2);
      pctx->destroy(pctx);
   }
   run_coalesced_tests(screen);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   return 0;
}
