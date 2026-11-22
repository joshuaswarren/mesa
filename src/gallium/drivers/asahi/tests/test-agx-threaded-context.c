/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Exercise Gallium Asahi u_threaded_context creation, buffer storage
 * replacement, unsynchronized/thread-safe transfers, texture transfers,
 * occlusion queries, and async fence flushes under the no-op DRM shim. */
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_fence.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static void
test_threaded_context(struct pipe_screen *screen)
{
   /* Compute-only contexts stay unwrapped. */
   struct pipe_context *cctx = screen->context_create(
      screen, NULL, PIPE_CONTEXT_PREFER_THREADED | PIPE_CONTEXT_COMPUTE_ONLY);
   CHECK(cctx);
   CHECK(agx_context(cctx)->tc == NULL);
   cctx->destroy(cctx);

   /* Graphics contexts requesting PREFER_THREADED wrap in u_threaded_context. */
   struct pipe_context *pctx =
      screen->context_create(screen, NULL, PIPE_CONTEXT_PREFER_THREADED);
   CHECK(pctx);

   struct pipe_context *raw = threaded_context_unwrap_sync(pctx);
   CHECK(raw && raw != pctx);
   CHECK(agx_context(raw)->tc == threaded_context(pctx));
   CHECK(pctx->get_device_reset_status(pctx) == PIPE_NO_RESET);

   /* Buffer create + subdata + DISCARD_WHOLE_RESOURCE (replace_buffer_storage)
    * + UNSYNCHRONIZED + THREAD_SAFE maps. */
   struct pipe_resource btempl = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      .width0 = 4096,
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_CONSTANT_BUFFER,
      .usage = PIPE_USAGE_STREAM,
   };
   struct pipe_resource *buf = screen->resource_create(screen, &btempl);
   CHECK(buf);

   uint32_t data[64];
   for (unsigned i = 0; i < ARRAY_SIZE(data); ++i)
      data[i] = 0x11220000u + i;
   pctx->buffer_subdata(pctx, buf, PIPE_MAP_WRITE, 0, sizeof(data), data);

   struct pipe_transfer *xfer = NULL;
   struct pipe_box box;
   u_box_1d(0, sizeof(data), &box);
   void *ptr = pctx->buffer_map(
      pctx, buf, 0, PIPE_MAP_WRITE | PIPE_MAP_DISCARD_WHOLE_RESOURCE, &box,
      &xfer);
   CHECK(ptr && xfer);
   for (unsigned i = 0; i < ARRAY_SIZE(data); ++i)
      ((uint32_t *)ptr)[i] = 0xaabb0000u + i;
   pctx->buffer_unmap(pctx, xfer);

   /* Unsynchronized + thread-safe map/unmap path updates valid_buffer_range
    * under lock on the active backing resource (threaded_resource->latest). */
   xfer = NULL;
   u_box_1d(256, 64, &box);
   ptr = pctx->buffer_map(
      pctx, buf, 0,
      PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED | PIPE_MAP_THREAD_SAFE, &box,
      &xfer);
   CHECK(ptr && xfer);
   memset(ptr, 0x5a, 64);
   pctx->buffer_unmap(pctx, xfer);
   CHECK(agx_valid_buffer_range_intersects(
      agx_resource(threaded_resource(buf)->latest), 256, 320));

   /* Under TC, driver-side PIPE_BUFFER shadowing must be disabled so app-thread
    * unsynchronized maps cannot race a driver-thread BO swap. */
   {
      struct pipe_resource *nbuf = screen->resource_create(screen, &btempl);
      CHECK(nbuf);
      struct agx_resource *nrsrc = agx_resource(nbuf);
      struct agx_bo *orig_bo = nrsrc->bo;

      struct pipe_context *synced = threaded_context_unwrap_sync(pctx);
      struct agx_context *actx = agx_context(synced);
      struct agx_batch *batch = agx_get_batch(actx);
      agx_batch_reads(batch, nrsrc);
      BITSET_SET(nrsrc->data_valid, 0);
      agx_valid_buffer_range_add(nrsrc, 0, sizeof(data));

      xfer = NULL;
      u_box_1d(0, sizeof(data), &box);
      ptr = synced->buffer_map(synced, nbuf, 0,
                               PIPE_MAP_WRITE | PIPE_MAP_DISCARD_WHOLE_RESOURCE,
                               &box, &xfer);
      CHECK(ptr && xfer);
      CHECK(nrsrc->bo == orig_bo);
      synced->buffer_unmap(synced, xfer);
      CHECK(nrsrc->bo == orig_bo);

      agx_flush_all(actx, "test no-shadow cleanup");
      pipe_resource_reference(&nbuf, NULL);
   }

   /* Read back to force sync and verify replaced storage contents. */
   xfer = NULL;
   u_box_1d(0, sizeof(data), &box);
   ptr = pctx->buffer_map(pctx, buf, 0, PIPE_MAP_READ, &box, &xfer);
   CHECK(ptr && xfer);
   CHECK(threaded_resource(buf)->latest != buf);
   CHECK(agx_resource(buf)->bo == agx_resource(threaded_resource(buf)->latest)->bo);
   for (unsigned i = 0; i < ARRAY_SIZE(data); ++i)
      CHECK(((uint32_t *)ptr)[i] == 0xaabb0000u + i);
   pctx->buffer_unmap(pctx, xfer);

   /* Texture transfer map/unmap + texture_subdata. */
   struct pipe_resource ttempl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 32,
      .height0 = 32,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
      .usage = PIPE_USAGE_DEFAULT,
   };
   struct pipe_resource *tex = screen->resource_create(screen, &ttempl);
   CHECK(tex);
   uint32_t pixels[32 * 32];
   for (unsigned i = 0; i < ARRAY_SIZE(pixels); ++i)
      pixels[i] = 0xff000000u | i;
   u_box_2d(0, 0, 32, 32, &box);
   pctx->texture_subdata(pctx, tex, 0, PIPE_MAP_WRITE, &box, pixels, 32 * 4,
                         sizeof(pixels));

   xfer = NULL;
   ptr = pctx->texture_map(pctx, tex, 0, PIPE_MAP_READ, &box, &xfer);
   CHECK(ptr && xfer);
   for (unsigned y = 0; y < 32; ++y) {
      const uint32_t *row = (const uint32_t *)((const uint8_t *)ptr + y * xfer->stride);
      for (unsigned x = 0; x < 32; ++x)
         CHECK(row[x] == pixels[y * 32 + x]);
   }
   pctx->texture_unmap(pctx, xfer);

   /* Occlusion query lifecycle through threaded context. */
   struct pipe_query *q = pctx->create_query(pctx, PIPE_QUERY_OCCLUSION_COUNTER, 0);
   CHECK(q);
   CHECK(pctx->begin_query(pctx, q));
   CHECK(pctx->end_query(pctx, q));
   /* Flush so TC marks the query flushed, then enqueue fresh TC work and check
    * that get_query_result does NOT drain the TC queue. */
   pctx->flush(pctx, NULL, 0);
   CHECK(p_atomic_read(&threaded_query(q)->flushed));
   pctx->buffer_subdata(pctx, buf, PIPE_MAP_WRITE, 0, sizeof(data), data);
   struct threaded_context *tc = threaded_context(pctx);
   unsigned queued_before = tc->batch_slots[tc->next].num_total_slots;
   CHECK(queued_before > 0);
   union pipe_query_result qres = {0};
   CHECK(pctx->get_query_result(pctx, q, false, &qres));
   CHECK(tc->batch_slots[tc->next].num_total_slots == queued_before);
   CHECK(pctx->get_query_result(pctx, q, true, &qres));
   CHECK(tc->batch_slots[tc->next].num_total_slots == queued_before);
   pctx->destroy_query(pctx, q);

   /* Timestamp query through flushed TC path as well, when supported. */
   if (screen->caps.query_timestamp) {
      struct pipe_query *tq = pctx->create_query(pctx, PIPE_QUERY_TIMESTAMP, 0);
      CHECK(tq);
      CHECK(pctx->end_query(pctx, tq));
      pctx->flush(pctx, NULL, 0);
      CHECK(p_atomic_read(&threaded_query(tq)->flushed));
      pctx->buffer_subdata(pctx, buf, PIPE_MAP_WRITE, 0, sizeof(data), data);
      queued_before = tc->batch_slots[tc->next].num_total_slots;
      CHECK(queued_before > 0);
      CHECK(pctx->get_query_result(pctx, tq, true, &qres));
      CHECK(tc->batch_slots[tc->next].num_total_slots == queued_before);
      pctx->destroy_query(pctx, tq);
   }

   /* Async fence flush + fence_finish + fence_reference. */
   struct pipe_fence_handle *fence = NULL;
   pctx->flush(pctx, &fence, PIPE_FLUSH_ASYNC);
   CHECK(fence);
   CHECK(screen->fence_finish(screen, pctx, fence, OS_TIMEOUT_INFINITE));
   screen->fence_reference(screen, &fence, NULL);
   CHECK(fence == NULL);

   /* Deferred fence flush (glFenceSync) followed by synchronous flush
    * (glFinish) and zero-timeout fence_finish
    * (dEQP-GLES3.functional.fence_sync.client_wait_sync_finish). */
   pctx->flush(pctx, &fence, PIPE_FLUSH_DEFERRED);
   CHECK(fence);
   pctx->flush(pctx, NULL, 0);
   CHECK(screen->fence_finish(screen, pctx, fence, 0));
   screen->fence_reference(screen, &fence, NULL);
   CHECK(fence == NULL);

   /* Deferred fence flush waited on directly with infinite timeout (must kick
    * the unflushed TC batch and signal ready without deadlocking). */
   pctx->flush(pctx, &fence, PIPE_FLUSH_DEFERRED);
   CHECK(fence);
   CHECK(screen->fence_finish(screen, pctx, fence, OS_TIMEOUT_INFINITE));
   screen->fence_reference(screen, &fence, NULL);
   CHECK(fence == NULL);

   /* Deferred fence flush polled with timeout=0 (async kick) then waited on. */
   pctx->flush(pctx, &fence, PIPE_FLUSH_DEFERRED);
   CHECK(fence);
   (void)screen->fence_finish(screen, pctx, fence, 0);
   CHECK(screen->fence_finish(screen, pctx, fence, OS_TIMEOUT_INFINITE));
   screen->fence_reference(screen, &fence, NULL);
   CHECK(fence == NULL);

   /* Synchronous fence flush without context argument to fence_finish. */
   pctx->flush(pctx, &fence, 0);
   CHECK(fence);
   CHECK(screen->fence_finish(screen, NULL, fence, OS_TIMEOUT_INFINITE));
   screen->fence_reference(screen, &fence, NULL);

   /* Faulted pre-allocated TC fence preserves pre-TC NULL fence semantics:
    * fence_finish returns true (completes glClientWaitSync), fence_get_fd
    * returns -1, and fence_server_sync on another context is a no-op that does
    * not fault the waiting context. */
   {
      struct pipe_context *fault_ctx =
         screen->context_create(screen, NULL, PIPE_CONTEXT_PREFER_THREADED);
      struct pipe_context *wait_ctx =
         screen->context_create(screen, NULL, PIPE_CONTEXT_PREFER_THREADED);
      CHECK(fault_ctx && wait_ctx);

      struct pipe_context *raw_fault = threaded_context_unwrap_sync(fault_ctx);
      p_atomic_set(&agx_context(raw_fault)->any_faults, true);

      struct pipe_fence_handle *fault_fence = NULL;
      fault_ctx->flush(fault_ctx, &fault_fence, PIPE_FLUSH_DEFERRED);
      CHECK(fault_fence);
      fault_ctx->flush(fault_ctx, NULL, 0);

      CHECK(screen->fence_finish(screen, fault_ctx, fault_fence, 0));
      CHECK(screen->fence_finish(screen, fault_ctx, fault_fence,
                                 OS_TIMEOUT_INFINITE));
      CHECK(fault_fence->faulted);
      CHECK(screen->fence_get_fd(screen, fault_fence) == -1);

      wait_ctx->fence_server_sync(wait_ctx, fault_fence, 0);
      wait_ctx->flush(wait_ctx, NULL, 0);
      CHECK(wait_ctx->get_device_reset_status(wait_ctx) == PIPE_NO_RESET);
      CHECK(fault_ctx->get_device_reset_status(fault_ctx) ==
            PIPE_GUILTY_CONTEXT_RESET);

      screen->fence_reference(screen, &fault_fence, NULL);
      wait_ctx->destroy(wait_ctx);
      fault_ctx->destroy(fault_ctx);
   }

   pipe_resource_reference(&tex, NULL);
   pipe_resource_reference(&buf, NULL);
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

   test_threaded_context(screen);

   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   return 0;
}
