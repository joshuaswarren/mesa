/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Exercise the real Gallium batch/BO lifetime and submit builder, replacing
 * only the kernel submit operation. The shim supplies device setup, not GPU
 * execution. Inspect barriers and fence placement at the ioctl boundary. */
#include <dlfcn.h>
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
static bool fail_submit, hold_submitted, check_custom_barriers;
static uint16_t expected_vdm_barrier, expected_cdm_barrier;
static uint32_t first_output;

int
drmSyncobjWait(int fd, uint32_t *handles, unsigned count, int64_t timeout,
               unsigned flags, uint32_t *first_signaled)
{
   if (hold_submitted && timeout == 0)
      return -ETIME;
   int (*real_wait)(int, uint32_t *, unsigned, int64_t, unsigned, uint32_t *) =
      dlsym(RTLD_NEXT, "drmSyncobjWait");
   CHECK(real_wait);
   return real_wait(fd, handles, count, timeout, flags, first_signaled);
}

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
         CHECK(h->vdm_barrier == (check_custom_barriers ? expected_vdm_barrier : 0) &&
               h->cdm_barrier == (check_custom_barriers ? expected_cdm_barrier : 0));
         prefix[1]++;
         computes++;
      } else if (h->cmd_type == DRM_ASAHI_CMD_RENDER) {
         CHECK(h->vdm_barrier == (prefix[1] ? DRM_ASAHI_BARRIER_NONE
                                            : (check_custom_barriers ? expected_vdm_barrier : 0)));
         CHECK(h->cdm_barrier == (prefix[1] ? 1
                                            : (check_custom_barriers ? expected_cdm_barrier : 0)));
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

static void
submit_barrier_case(struct agx_context *ctx, struct agx_batch *batch,
                    bool is_compute, uint16_t exp_vdm, uint16_t exp_cdm)
{
   struct drm_asahi_cmd_compute c = {0};
   struct drm_asahi_cmd_render r = {0};
   calls = computes = renders = first_output = 0;
   expected_inputs = 0;
   fail_submit = false;
   check_custom_barriers = true;
   expected_vdm_barrier = exp_vdm;
   expected_cdm_barrier = exp_cdm;
   agx_batch_submit(ctx, batch, is_compute ? &c : NULL, is_compute ? NULL : &r);
   CHECK(calls == 1);
   CHECK(computes == is_compute && renders == !is_compute);
   check_custom_barriers = false;
}

static void
test_disjoint_cdm_barriers(struct pipe_screen *screen)
{
   struct agx_device *dev = agx_device(screen);
   dev->ops.submit = inspect_submit;
   const bool agx3 = agx_chip_native_agx3(dev->chip);
   const uint16_t none = agx3 ? DRM_ASAHI_BARRIER_NONE : 0;

   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);
   p_atomic_set(&agx_screen(screen)->flush_wait_seqid, 0);
   ctx->flush_my_seqid = 0;
   ctx->flush_other_seqid = 0;
   dev->params.max_commands_per_submission = 64;

   /* Keep prior submissions in flight so inter-batch overlap is exercised. */
   hold_submitted = true;

   struct agx_va va = {.addr = 0x8000};
   struct agx_bo bos[3] = {0};
   struct agx_resource res[3] = {0};
   for (unsigned i = 0; i < 3; i++) {
      bos[i].handle = 10 + i;
      bos[i].refcnt = 1;
      bos[i].va = &va;
      res[i].base.b.screen = screen;
      res[i].base.b.target = PIPE_BUFFER;
      res[i].bo = &bos[i];
   }

   /* R1 writes res[0] with no prior in-flight batches: no barriers needed. */
   struct agx_batch *b = agx_get_batch(ctx);
   b->initialized = true;
   agx_batch_writes(b, &res[0], 0);
   submit_barrier_case(ctx, b, false, none, none);

   /* C1 writes disjoint res[1] while R1 is in flight: 2-pipe CDM overlap. */
   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   agx_batch_writes(b, &res[1], 0);
   submit_barrier_case(ctx, b, true, none, none);

   /* C2 reads res[0] (RAW against R1): waits on VDM only. */
   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   agx_batch_reads(b, &res[0]);
   submit_barrier_case(ctx, b, true, 0, none);

   /* C3 writes res[1] (WAW against C1): waits on CDM only. */
   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   agx_batch_writes(b, &res[1], 0);
   submit_barrier_case(ctx, b, true, none, 0);

   /* C4 writes res[0] (WAW against R1 + WAR against C2): waits on both. */
   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   agx_batch_writes(b, &res[0], 0);
   submit_barrier_case(ctx, b, true, 0, 0);

   /* R2 writes disjoint res[2] while R1 and C1..C4 are in flight: no barrier. */
   b = agx_get_batch(ctx);
   b->initialized = true;
   agx_batch_writes(b, &res[2], 0);
   submit_barrier_case(ctx, b, false, none, none);

   /* Explicit memory barrier forces serialization until prior batches retire. */
   agx_memory_barrier(pctx, PIPE_BARRIER_ALL);
   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   submit_barrier_case(ctx, b, true, 0, 0);

   /* Once pre-barrier batches retire, disjoint CDM elision resumes. */
   hold_submitted = false;
   agx_sync_all(ctx, "retire barrier test batches");
   hold_submitted = true;

   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   submit_barrier_case(ctx, b, true, none, none);

   /* A texture barrier serializes against the in-flight compute: CDM only. */
   pctx->texture_barrier(pctx, PIPE_TEXTURE_BARRIER_SAMPLER);
   b = agx_get_batch(ctx);
   b->initialized = true;
   submit_barrier_case(ctx, b, false, none, 0);

   hold_submitted = false;
   agx_sync_all(ctx, "retire texture barrier batches");
   hold_submitted = true;

   /* A directly submitted batch with an untracked side effect takes full
    * barriers itself and keeps later disjoint batches waiting on it. */
   b = agx_get_compute_batch(ctx);
   b->initialized = true;
   b->query_access = true;
   submit_barrier_case(ctx, b, true, 0, 0);
   CHECK(b->barrier_untracked && b->submitted_has_cdm && !b->submitted_has_vdm);

   b = agx_get_batch(ctx);
   b->initialized = true;
   agx_batch_writes(b, &res[2], 0);
   submit_barrier_case(ctx, b, false, none, 0);
   CHECK(!b->barrier_untracked && b->submitted_has_vdm && !b->submitted_has_cdm);

   hold_submitted = false;
   agx_sync_all(ctx, "cleanup barrier test batches");
   for (unsigned i = 0; i < 3; i++)
      CHECK(bos[i].refcnt == 1);
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
   test_disjoint_cdm_barriers(screen);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   return 0;
}
