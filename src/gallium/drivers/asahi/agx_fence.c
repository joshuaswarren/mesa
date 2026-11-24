/*
 * Based on panfrost/pan_fence.c:
 *
 * Copyright 2022 Amazon.com, Inc. or its affiliates.
 * Copyright 2008 VMware, Inc.
 * Copyright 2014 Broadcom
 * Copyright 2018 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * Copyright 2012 Rob Clark
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <xf86drm.h>
#include <string.h>

#include "agx_fence.h"
#include "agx_state.h"

#include "util/libsync.h"
#include "util/os_file.h"
#include "util/os_time.h"
#include "util/u_inlines.h"

void
agx_fence_reference(struct pipe_screen *pscreen, struct pipe_fence_handle **ptr,
                    struct pipe_fence_handle *fence)
{
   struct agx_device *dev = agx_device(pscreen);
   struct pipe_fence_handle *old = *ptr;

   if (pipe_reference(old ? &old->reference : NULL,
                      fence ? &fence->reference : NULL)) {
      tc_unflushed_batch_token_reference(&old->tc_token, NULL);
      util_queue_fence_destroy(&old->ready);
      if (old->syncobj)
         drmSyncobjDestroy(dev->fd, old->syncobj);
      free(old);
   }

   *ptr = fence;
}

static bool
agx_fence_wait_ready(struct pipe_context *ctx, struct pipe_fence_handle *fence,
                     uint64_t timeout, int64_t abs_timeout)
{
   if (util_queue_fence_is_signalled(&fence->ready))
      return true;

   if (fence->tc_token && ctx)
      threaded_context_flush(ctx, fence->tc_token, timeout == 0);

   if (!timeout)
      return util_queue_fence_is_signalled(&fence->ready);

   if (timeout == OS_TIMEOUT_INFINITE) {
      util_queue_fence_wait(&fence->ready);
      return true;
   }

   return util_queue_fence_wait_timeout(&fence->ready, abs_timeout);
}

bool
agx_fence_finish(struct pipe_screen *pscreen, struct pipe_context *ctx,
                 struct pipe_fence_handle *fence, uint64_t timeout)
{
   struct agx_device *dev = agx_device(pscreen);
   int ret;

   int64_t abs_timeout = os_time_get_absolute_timeout(timeout);
   if (!agx_fence_wait_ready(ctx, fence, timeout, abs_timeout))
      return false;

   /* Preserve pre-TC NULL fence semantics for faulted fences so glClientWaitSync
    * completes instead of timing out forever; the reset is reported on the
    * faulting context itself.
    */
   if (fence->faulted || !fence->syncobj)
      return true;

   if (p_atomic_read(&fence->signaled))
      return true;

   uint64_t drm_timeout =
      (uint64_t)abs_timeout == OS_TIMEOUT_INFINITE ? INT64_MAX : abs_timeout;

   ret = drmSyncobjWait(dev->fd, &fence->syncobj, 1, drm_timeout,
                        DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);

   if (ret < 0 && ret != -ETIME) {
      agx_msg("fence wait failed: %s\n", strerror(-ret));
      return false;
   }

   bool signaled = (ret >= 0);
   if (signaled)
      p_atomic_set(&fence->signaled, true);
   return signaled;
}

int
agx_fence_get_fd(struct pipe_screen *screen, struct pipe_fence_handle *f)
{
   struct agx_device *dev = agx_device(screen);
   int fd = -1;

   if (!util_queue_fence_is_signalled(&f->ready))
      util_queue_fence_wait(&f->ready);

   if (f->faulted || !f->syncobj)
      return -1;

   int ret = drmSyncobjExportSyncFile(dev->fd, f->syncobj, &fd);
   if (ret || fd < 0) {
      agx_msg("fence export failed\n");
      return -1;
   }

   return fd;
}

struct pipe_fence_handle *
agx_fence_from_fd(struct agx_context *ctx, int fd, enum pipe_fd_type type)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   int ret;

   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f)
      return NULL;

   if (type == PIPE_FD_TYPE_NATIVE_SYNC) {
      ret = drmSyncobjCreate(dev->fd, 0, &f->syncobj);
      if (ret) {
         agx_msg("create syncobj failed\n");
         goto err_free_fence;
      }

      ret = drmSyncobjImportSyncFile(dev->fd, f->syncobj, fd);
      if (ret) {
         agx_msg("import syncfile failed\n");
         goto err_destroy_syncobj;
      }
   } else {
      assert(type == PIPE_FD_TYPE_SYNCOBJ);
      ret = drmSyncobjFDToHandle(dev->fd, fd, &f->syncobj);
      if (ret) {
         agx_msg("import syncobj FD failed\n");
         goto err_free_fence;
      }
   }

   pipe_reference_init(&f->reference, 1);
   util_queue_fence_init(&f->ready);

   return f;

err_destroy_syncobj:
   drmSyncobjDestroy(dev->fd, f->syncobj);
err_free_fence:
   free(f);
   return NULL;
}

static bool
agx_fence_snapshot_syncobj(struct agx_context *ctx, uint32_t *out_syncobj)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   int fd = -1, ret;

   /* A failed submission must not export an earlier rendering success. */
   if (ctx->any_faults)
      return false;

   /* Ensure ctx->syncobj reflects any queued work before snapshotting it. */
   agx_submit_pending(ctx);
   if (ctx->any_faults)
      return false;

   /* Common path with no deferred server-wait fence: snapshot ctx->syncobj
    * directly into a fresh syncobj via DRM_IOCTL_SYNCOBJ_TRANSFER, avoiding
    * sync_file allocation and fd install/close on every glFenceSync().
    * agx_fence_get_fd() exports a sync_file lazily only when requested.
    */
   if (ctx->in_sync_fd < 0) {
      uint32_t syncobj = 0;

      ret = drmSyncobjCreate(dev->fd, 0, &syncobj);
      if (ret) {
         agx_msg("create syncobj failed\n");
         return false;
      }

      ret = drmSyncobjTransfer(dev->fd, syncobj, 0, ctx->syncobj, 0, 0);
      if (!ret) {
         *out_syncobj = syncobj;
         return true;
      }

      int err = ret == -1 ? errno : -ret;
      drmSyncobjDestroy(dev->fd, syncobj);

      if (err != EOPNOTSUPP && err != ENOSYS) {
         agx_msg("syncobj transfer failed: %s\n", strerror(err));
         return false;
      }
   }

   /* Fallback path (or pending deferred server-wait): export a sync_file. */
   ret = drmSyncobjExportSyncFile(dev->fd, ctx->syncobj, &fd);
   if (ret || fd == -1) {
      agx_msg("export failed\n");
      return false;
   }

   /* fence_server_sync() defers an imported native fence until the next GPU
    * submission.  A fence-only flush has no batch to consume that dependency,
    * but the newly exported fence is still ordered after the server wait.  In
    * particular, Wayland explicit-sync clients use exactly this sequence when
    * forwarding an acquire fence to a release fence without drawing.
    *
    * Snapshot both the last rendering fence and the pending server waits.  A
    * completed rendering fence can be discarded, avoiding a sync-file merge
    * for the common no-work handoff.  Leave ctx->in_sync_fd intact: a later
    * real submission must wait on it too.
    */
   if (ctx->in_sync_fd >= 0) {
      ret = drmSyncobjWait(dev->fd, &ctx->syncobj, 1, 0,
                           DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);

      if (ret == 0) {
         int pending = os_dupfd_cloexec(ctx->in_sync_fd);
         if (pending >= 0) {
            close(fd);
            fd = pending;
         } else if (agx_fallback_wait(ctx->in_sync_fd, "fence dup failed")) {
            close(fd);
            fd = -1;
         }
      } else if (ret == -ETIME) {
         int merged = sync_merge("asahi-fence", fd, ctx->in_sync_fd);
         if (merged >= 0) {
            close(fd);
            fd = merged;
         } else if (agx_fallback_wait(fd, "fence merge failed") ||
                    agx_fallback_wait(ctx->in_sync_fd, "fence merge failed")) {
            close(fd);
            fd = -1;
         }
      } else {
         close(fd);
         fd = -1;
      }

      if (fd < 0) {
         agx_msg("failed to snapshot pending server wait\n");
         return false;
      }
   }

   uint32_t syncobj = 0;
   ret = drmSyncobjCreate(dev->fd, 0, &syncobj);
   if (ret) {
      agx_msg("create syncobj failed\n");
      close(fd);
      return false;
   }

   ret = drmSyncobjImportSyncFile(dev->fd, syncobj, fd);
   close(fd);
   if (ret) {
      agx_msg("import syncfile failed\n");
      drmSyncobjDestroy(dev->fd, syncobj);
      return false;
   }

   *out_syncobj = syncobj;
   return true;
}

struct pipe_fence_handle *
agx_fence_create(struct agx_context *ctx)
{
   uint32_t syncobj = 0;
   if (!agx_fence_snapshot_syncobj(ctx, &syncobj))
      return NULL;

   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f) {
      drmSyncobjDestroy(agx_device(ctx->base.screen)->fd, syncobj);
      return NULL;
   }

   pipe_reference_init(&f->reference, 1);
   util_queue_fence_init(&f->ready);
   f->syncobj = syncobj;
   return f;
}

struct pipe_fence_handle *
agx_fence_create_unflushed(struct pipe_context *pctx,
                           struct tc_unflushed_batch_token *token)
{
   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f)
      return NULL;

   pipe_reference_init(&f->reference, 1);
   util_queue_fence_init(&f->ready);
   util_queue_fence_reset(&f->ready);
   tc_unflushed_batch_token_reference(&f->tc_token, token);
   return f;
}

void
agx_fence_populate(struct agx_context *ctx, struct pipe_fence_handle *f)
{
   uint32_t syncobj = 0;
   if (agx_fence_snapshot_syncobj(ctx, &syncobj))
      f->syncobj = syncobj;
   else
      f->faulted = true;

   util_queue_fence_signal(&f->ready);
}

void
agx_create_fence_fd(struct pipe_context *pctx,
                    struct pipe_fence_handle **pfence, int fd,
                    enum pipe_fd_type type)
{
   *pfence = agx_fence_from_fd(agx_context(pctx), fd, type);
}

void
agx_fence_server_sync(struct pipe_context *pctx, struct pipe_fence_handle *f,
                      uint64_t value)
{
   struct agx_device *dev = agx_device(pctx->screen);
   struct agx_context *ctx = agx_context(pctx);
   int fd = -1, ret;
   assert(!value);

   if (!util_queue_fence_is_signalled(&f->ready))
      util_queue_fence_wait(&f->ready);

   /* Preserve pre-TC NULL fence semantics: server-waiting on a faulted fence
    * is a no-op and must not mark an innocent waiting context as faulted.
    */
   if (f->faulted || !f->syncobj)
      return;

   /* Batches queued before this server wait must not depend on the incoming
    * acquire fence (which may even depend on those earlier batches). */
   agx_submit_pending(ctx);

   ret = drmSyncobjExportSyncFile(dev->fd, f->syncobj, &fd);
   if (ret || fd < 0) {
      agx_msg("server-wait fence export failed\n");
      p_atomic_set(&ctx->any_faults, true);
      return;
   }

   ret = sync_accumulate("asahi", &ctx->in_sync_fd, fd);
   if (ret || ctx->in_sync_fd < 0) {
      /* Keep the previous accumulated fence and consume this dependency on
       * the CPU if the kernel cannot allocate a merged sync file. Dropping it
       * would let the next GPU submission run before the acquire fence. */
      if (agx_fallback_wait(fd, "fence accumulation failed")) {
         agx_msg("server-wait fence accumulation and fallback wait failed\n");
         const bool first_fault = !ctx->any_faults;
         p_atomic_set(&ctx->any_faults, true);
         if (first_fault && ctx->reset.reset)
            ctx->reset.reset(ctx->reset.data, PIPE_GUILTY_CONTEXT_RESET);
      }
   }
   close(fd);
}
