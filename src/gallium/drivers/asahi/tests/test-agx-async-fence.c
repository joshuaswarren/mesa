/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* On a queue created with DRM_ASAHI_QUEUE_CREATE_ASYNC_SUBMIT the submit
 * ioctl returns before the work retires, so a batch that fails afterwards
 * only signals its output fence with the error. Retirement must read that
 * status from the exported sync_file and report the context as reset.
 *
 * The real Gallium batch submit and retirement run against the no-op shim;
 * only the kernel submit, the syncobj export and SYNC_IOC_FILE_INFO on the
 * exported file are replaced. No command executes on a GPU.
 */
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <linux/sync_file.h>
#include <sys/ioctl.h>

#include "util/driconf.h"
#include "util/xmlconfig.h"
#include "agx_state.h"

/* after agx_state.h, as in the other gallium tests */
#include "agx_public.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

enum export_mode { EXPORT_OK, EXPORT_FAILS, EXPORT_NO_FD };

static enum export_mode export_mode;
static bool polls_pending;
static int info_status, info_errno;
static unsigned exports, infos, resets;
static enum pipe_reset_status last_reset;
static int sync_file = -1;

/* Whatever was exported has been closed again. */
static void
check_closed(void)
{
   if (sync_file >= 0) {
      CHECK(fcntl(sync_file, F_GETFD) == -1 && errno == EBADF);
      sync_file = -1;
   }
}

/* The shim stubs DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD; hand out a private file
 * whose SYNC_IOC_FILE_INFO the ioctl() below answers. */
int
drmSyncobjExportSyncFile(int fd, uint32_t handle, int *sync_file_fd)
{
   exports++;
   CHECK(handle);
   if (export_mode == EXPORT_FAILS)
      return -EINVAL;
   if (export_mode == EXPORT_NO_FD) {
      *sync_file_fd = -1;
      return 0;
   }
   check_closed();
   sync_file = open("/dev/null", O_RDONLY | O_CLOEXEC);
   CHECK(sync_file >= 0);
   *sync_file_fd = sync_file;
   return 0;
}

int
ioctl(int fd, unsigned long request, ...)
{
   va_list args;
   va_start(args, request);
   void *arg = va_arg(args, void *);
   va_end(args);

   if (fd >= 0 && fd == sync_file && request == SYNC_IOC_FILE_INFO) {
      infos++;
      if (info_errno) {
         errno = info_errno;
         return -1;
      }
      struct sync_file_info *info = arg;
      info->status = info_status;
      return 0;
   }

   /* Everything else belongs to the drm-shim preload. */
   static int (*next)(int, unsigned long, ...);
   if (!next)
      next = dlsym(RTLD_NEXT, "ioctl");
   CHECK(next);
   return next(fd, request, arg);
}

/* With polls_pending, a zero-timeout poll sees every batch still running,
 * so batches retire only through the blocking wait in agx_sync_all().
 * Otherwise the shim reports them done and agx_batch_submit() retires them
 * through agx_cleanup_batches(). */
int
drmSyncobjWait(int fd, uint32_t *handles, unsigned count, int64_t timeout,
               unsigned flags, uint32_t *first_signaled)
{
   if (polls_pending && !timeout)
      return -ETIME;
   static int (*next)(int, uint32_t *, unsigned, int64_t, unsigned, uint32_t *);
   if (!next)
      next = dlsym(RTLD_NEXT, "drmSyncobjWait");
   CHECK(next);
   return next(fd, handles, count, timeout, flags, first_signaled);
}

static int
submit_ok(struct agx_device *dev, struct drm_asahi_submit *submit,
          struct agx_submit_virt *virt)
{
   return 0;
}

static void
on_reset(void *data, enum pipe_reset_status status)
{
   resets++;
   last_reset = status;
}

/* Submit compute batches, then retire them as a wait for results would. */
static void
run(struct pipe_context *pctx, unsigned count)
{
   struct agx_context *ctx = agx_context(pctx);
   for (unsigned i = 0; i < count; i++) {
      struct agx_batch *batch = agx_get_batch(ctx);
      struct drm_asahi_cmd_compute c = {0};
      agx_batch_submit(ctx, batch, &c, NULL);
      CHECK(agx_batch_is_submitted(batch) == polls_pending);
      check_closed();
   }
   agx_sync_all(ctx, "async fence test");
   check_closed();
   CHECK(BITSET_IS_EMPTY(ctx->batches.submitted));
}

struct scenario {
   const char *name;
   bool async, callback, polls_pending;
   enum export_mode export_mode;
   int status, info_errno;
   unsigned batches;
   /* expected */
   unsigned exports, infos, resets;
   bool fault;
};

static const struct scenario scenarios[] = {
   /* Synchronous queues report failures from the submit ioctl itself. */
   {"sync queue", false, true, true, EXPORT_OK, -EIO, 0, 1, 0, 0, 0, false},
   {"signalled", true, true, true, EXPORT_OK, 1, 0, 2, 2, 2, 0, false},
   {"still active", true, true, true, EXPORT_OK, 0, 0, 1, 1, 1, 0, false},
   {"export fails", true, true, true, EXPORT_FAILS, -EIO, 0, 1, 1, 0, 0, false},
   {"export without fd", true, true, true, EXPORT_NO_FD, -EIO, 0, 1, 1, 0, 0,
    false},
   {"info fails", true, true, true, EXPORT_OK, -EIO, EINVAL, 1, 1, 1, 0, false},
   /* Both batches retire after the first failed: only it calls the reset
    * callback. */
   {"failed, blocking wait", true, true, true, EXPORT_OK, -EIO, 0, 2, 2, 2, 1,
    true},
   /* Retired by the poll in agx_batch_submit(). */
   {"failed, poll", true, true, false, EXPORT_OK, -EIO, 0, 1, 1, 1, 1, true},
   {"failed, no callback", true, false, true, EXPORT_OK, -ENODEV, 0, 1, 1, 1, 0,
    true},
};

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
   struct pipe_screen_config config = {.options = &cache,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);
   struct agx_device *dev = agx_device(screen);
   dev->ops.submit = submit_ok;

   for (unsigned i = 0; i < ARRAY_SIZE(scenarios); i++) {
      const struct scenario *s = &scenarios[i];
      struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
      CHECK(pctx);
      struct agx_context *ctx = agx_context(pctx);
      /* The shim's queue creation never grants async submission. */
      ctx->async_submit = s->async;
      if (s->callback) {
         struct pipe_device_reset_callback cb = {.reset = on_reset};
         pctx->set_device_reset_callback(pctx, &cb);
      }

      polls_pending = s->polls_pending;
      export_mode = s->export_mode;
      info_status = s->status;
      info_errno = s->info_errno;
      exports = infos = resets = 0;
      last_reset = PIPE_NO_RESET;
      run(pctx, s->batches);

      fprintf(stderr, "%s: exports %u infos %u resets %u fault %d\n", s->name,
              exports, infos, resets, ctx->any_faults);
      CHECK(exports == s->exports);
      CHECK(infos == s->infos);
      CHECK(resets == s->resets);
      CHECK(ctx->any_faults == s->fault);
      if (s->resets)
         CHECK(last_reset == PIPE_GUILTY_CONTEXT_RESET);
      CHECK(pctx->get_device_reset_status(pctx) ==
            (s->fault ? PIPE_GUILTY_CONTEXT_RESET : PIPE_NO_RESET));

      /* Context teardown retires nothing more through the export. */
      export_mode = EXPORT_FAILS;
      polls_pending = false;
      pctx->destroy(pctx);
   }

   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
   fprintf(stderr, "async fence: ok\n");
   return 0;
}
