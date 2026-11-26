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
#include "agx_public.h"

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static unsigned calls, computes, renders, expected_inputs;
static struct agx_context *current_ctx;
static bool fail_submit;
static bool expect_fragment_barriers;
static uint32_t first_output;

static int
inspect_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
               struct agx_submit_virt *virt)
{
   calls++;
   struct agx_context *ctx = current_ctx;
   CHECK(submit->in_sync_count == expected_inputs);
   const struct drm_asahi_sync *syncs = (void *)(uintptr_t)submit->syncs;
   /* Once a context has sibling work, the final submission of a batch also
    * signals the context's cumulative anchor, at a fresh point. */
   if (ctx->sib.anchored && submit->out_sync_count == 4) {
      const struct drm_asahi_sync *anchor =
         &syncs[submit->in_sync_count + 3];
      CHECK(anchor->sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE);
      CHECK(anchor->handle == ctx->sib.anchor);
      CHECK(anchor->timeline_value == ctx->sib.anchor_point);
   } else {
      CHECK(submit->out_sync_count == 1 || submit->out_sync_count == 3);
   }
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
         const struct drm_asahi_cmd_render *r =
            (const void *)(data + cursor + sizeof(*h));
         CHECK(!!(r->flags & DRM_ASAHI_RENDER_FRAGMENT_BARRIERS) ==
               expect_fragment_barriers);
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
   current_ctx = ctx;
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
   expect_fragment_barriers = false;
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

static void
run_fragment_barrier_checks(struct pipe_screen *screen)
{
   struct agx_device *dev = agx_device(screen);
   const uint64_t saved_features = dev->params.features;
   const bool non_classic =
      agx_chip_submission_model(dev->chip) != AGX_SUBMISSION_MODEL_CLASSIC;
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);
   current_ctx = ctx;
   struct agx_screen *ascreen = agx_screen(screen);
   p_atomic_set(&ascreen->flush_wait_seqid, 0);
   dev->params.max_commands_per_submission = 64;

   for (unsigned mode = 0; mode < 4; mode++) {
      dev->params.features = mode == 3
         ? (saved_features & ~DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS)
         : (saved_features | DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS);

      struct agx_batch *batch = agx_get_batch(ctx);
      struct agx_va va = {.addr = 0x4000};
      struct agx_bo clean_bo = {.handle = 11, .refcnt = 1, .va = &va};
      struct agx_resource clean_rsrc = {.bo = &clean_bo};
      agx_batch_note_vtx_read(batch, &clean_rsrc);
      CHECK(!batch->vtx_needs_render_barrier);

      if (mode == 1) {
         struct agx_bo shared_bo = {
            .handle = 12,
            .refcnt = 1,
            .va = &va,
            .flags = AGX_BO_SHARED,
         };
         struct agx_resource shared_rsrc = {.bo = &shared_bo};
         agx_batch_note_vtx_read(batch, &shared_rsrc);
         CHECK(batch->vtx_needs_render_barrier);
      }

      const bool compute = (mode == 2);
      calls = computes = renders = first_output = 0;
      expected_inputs = 0;
      fail_submit = false;
      expect_fragment_barriers = non_classic && mode == 0;

      struct drm_asahi_cmd_compute c = {0};
      struct drm_asahi_cmd_render r = {0};
      agx_batch_submit(ctx, batch, compute ? &c : NULL, &r);
      CHECK(calls == 1 && computes == compute && renders == 1);
      agx_sync_all(ctx, "fragment barrier test retirement");
   }

   dev->params.features = saved_features;
   pctx->destroy(pctx);
}

/* Sibling kernel queues: what one compute or render batch submission named. */
static struct {
   unsigned calls;
   uint32_t queue_id;
   uint64_t anchor_wait;
   uint64_t anchor_signal;
} sib;

static int
sibling_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
               struct agx_submit_virt *virt)
{
   const struct drm_asahi_sync *syncs = (void *)(uintptr_t)submit->syncs;
   struct agx_context *ctx = current_ctx;

   sib.calls++;
   sib.queue_id = submit->queue_id;
   sib.anchor_wait = 0;
   for (unsigned i = 0; i < submit->in_sync_count; ++i) {
      if (syncs[i].handle == ctx->sib.anchor) {
         CHECK(syncs[i].sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ);
         sib.anchor_wait = syncs[i].timeline_value;
      }
   }

   /* The anchor appears exactly when the context has started it. */
   CHECK(submit->out_sync_count == (ctx->sib.anchored ? 4 : 3));
   if (ctx->sib.anchored) {
      const struct drm_asahi_sync *anchor =
         &syncs[submit->in_sync_count + submit->out_sync_count - 1];
      CHECK(anchor->sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE);
      CHECK(anchor->handle == ctx->sib.anchor);
      CHECK(anchor->timeline_value > sib.anchor_signal);
      sib.anchor_signal = anchor->timeline_value;
   }

   /* Compute commands of a sibling submission wait only on that queue. */
   const struct drm_asahi_cmd_header *h = (void *)(uintptr_t)submit->cmdbuf;
   if (h->cmd_type == DRM_ASAHI_CMD_COMPUTE)
      CHECK(h->vdm_barrier == 0 && h->cdm_barrier == 0);
   return 0;
}

/* Submit one compute-only batch that dispatched `threads` threads through
 * shader storage, touching `bo`. */
static void
submit_storage_batch(struct agx_context *ctx, struct agx_bo *bo,
                     uint64_t threads, bool coherent)
{
   struct agx_batch *batch = agx_get_compute_batch(ctx);
   agx_batch_add_bo(batch, bo);
   batch->storage_dispatches = 1;
   batch->storage_min_threads = threads;
   batch->coherent_writes = coherent;
   if (ctx->batch == batch)
      ctx->batch = NULL;

   struct drm_asahi_cmd_compute c = {0};
   agx_batch_submit(ctx, batch, &c, NULL);
}

static void
run_sibling_checks(struct pipe_screen *screen)
{
   struct agx_device *dev = agx_device(screen);
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);
   current_ctx = ctx;
   p_atomic_set(&agx_screen(screen)->flush_wait_seqid, 0);

   /* Only kernels that can keep completion ordered get siblings. */
   CHECK((ctx->sib.count > 1) == agx_sibling_compute_supported(dev));
   if (ctx->sib.count == 1) {
      CHECK(ctx->sib.id[0] == ctx->queue_id);
      pctx->destroy(pctx);
      return;
   }

   int (*saved)(struct agx_device *, struct drm_asahi_submit *,
                struct agx_submit_virt *) = dev->ops.submit;
   dev->ops.submit = sibling_submit;
   memset(&sib, 0, sizeof(sib));

   struct agx_va va = {.addr = 0x4000};
   struct agx_bo bos[8] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(bos); i++)
      bos[i] = (struct agx_bo){.handle = 100 + i, .refcnt = 1, .va = &va};

   /* Small dispatches stay on the context's queue however deep, and do not
    * start the anchor. */
   for (unsigned i = 0; i < 4; ++i) {
      submit_storage_batch(ctx, &bos[i], AGX_SIBLING_COMPUTE_MIN_THREADS - 1,
                           false);
      CHECK(sib.queue_id == ctx->queue_id && !sib.anchor_wait);
   }
   CHECK(!ctx->sib.anchored);
   agx_sync_all(ctx, "sibling test");

   /* Independent storage-only batches go round-robin once at least two
    * earlier submissions are in flight. */
   bool sibling_used = false;
   for (unsigned i = 0; i < 6; ++i) {
      submit_storage_batch(ctx, &bos[i], AGX_SIBLING_COMPUTE_MIN_THREADS,
                           false);
      CHECK(ctx->sib.anchored);
      CHECK(!sib.anchor_wait);
      /* The first candidate starts the anchor on the context's queue. */
      if (i == 0)
         CHECK(sib.queue_id == ctx->queue_id);
      sibling_used |= sib.queue_id != ctx->queue_id;
      if (sib.queue_id != ctx->queue_id)
         CHECK(ctx->sib.pending_point == sib.anchor_signal);
   }
   CHECK(sibling_used);
   CHECK(ctx->sib.pending_point);
   const uint64_t pending = ctx->sib.pending_point;

   /* A batch with a coherent write waits for the sibling work, on the
    * context's own queue, and settles it for the work after. */
   submit_storage_batch(ctx, &bos[6], AGX_SIBLING_COMPUTE_MIN_THREADS, true);
   CHECK(sib.queue_id == ctx->queue_id && sib.anchor_wait == pending);
   CHECK(!ctx->sib.pending_point);

   /* A memory barrier starts a new epoch: batches recorded before it are no
    * longer exempt from the BO overlap check. The shim retires every
    * submission at once, so nothing is left in flight to overlap here and the
    * next storage-only batch stays independent. */
   const uint64_t epoch = ctx->memory_barrier_epoch;
   pctx->memory_barrier(pctx, PIPE_BARRIER_SHADER_BUFFER);
   CHECK(ctx->memory_barrier_epoch == epoch + 1);
   submit_storage_batch(ctx, &bos[7], AGX_SIBLING_COMPUTE_MIN_THREADS, false);
   CHECK(!sib.anchor_wait);

   agx_sync_all(ctx, "sibling test retirement");
   dev->ops.submit = saved;
   pctx->destroy(pctx);
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
   run_fragment_barrier_checks(screen);
   run_sibling_checks(screen);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   return 0;
}
