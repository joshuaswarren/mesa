/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "agx_device.h"

/* agx_create_async_command_queue(): Gallium context queues ask the kernel for
 * DRM_ASAHI_QUEUE_CREATE_ASYNC_SUBMIT only where it reports
 * DRM_ASAHI_FEATURE_ASYNC_SUBMIT, and report whether they got it. With
 * AGX_DBG_1QUEUE the shared queue never serves a caller a flag it did not ask
 * for. QUEUE_CREATE is mocked; nothing reaches a kernel. */
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static unsigned creates;
static uint32_t sent_flags, next_id = 7;
static int fail_errno;

static int
queue_ioctl(struct agx_device *dev, unsigned cmd, void *req)
{
   if (_IOC_NR(cmd) != _IOC_NR(DRM_IOCTL_ASAHI_QUEUE_CREATE))
      return 0;
   struct drm_asahi_queue_create *create = req;
   ++creates;
   sent_flags = create->flags;
   if (fail_errno) {
      errno = fail_errno;
      return -1;
   }
   create->queue_id = next_id++;
   return 0;
}

static void
reset(struct agx_device *dev, uint64_t features, uint64_t debug)
{
   memset(dev, 0, sizeof(*dev));
   dev->ops.simple_ioctl = queue_ioctl;
   dev->params.features = features;
   dev->debug = debug;
   simple_mtx_init(&dev->vma_lock, mtx_plain);
   creates = sent_flags = 0;
   fail_errno = 0;
}

int
main(void)
{
   struct agx_device dev;
   bool async;
   uint32_t id;

   /* A kernel without the feature gets no flag and a synchronous queue. */
   reset(&dev, 0, 0);
   async = true;
   id = agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM, &async);
   CHECK(id != UINT32_MAX && creates == 1);
   CHECK(sent_flags == 0 && !async);

   /* A kernel with it gets the flag and an async queue. */
   reset(&dev, DRM_ASAHI_FEATURE_ASYNC_SUBMIT, 0);
   async = false;
   id = agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM, &async);
   CHECK(id != UINT32_MAX && creates == 1);
   CHECK(sent_flags == DRM_ASAHI_QUEUE_CREATE_ASYNC_SUBMIT && async);

   /* Other callers never ask for it, feature or not. */
   id = agx_create_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM);
   CHECK(id != UINT32_MAX && creates == 2 && sent_flags == 0);

   /* A refused create (HIGH without CAP_SYS_NICE) is not async, and the
    * caller sees the kernel's errno for its MEDIUM fallback. */
   fail_errno = EPERM;
   async = true;
   id = agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_HIGH, &async);
   CHECK(id == UINT32_MAX && !async && errno == EPERM);
   simple_mtx_destroy(&dev.vma_lock);

   /* AGX_DBG_1QUEUE, sync queue first (Vulkan): a later Gallium context
    * shares it and is told it is synchronous. */
   reset(&dev, DRM_ASAHI_FEATURE_ASYNC_SUBMIT, AGX_DBG_1QUEUE);
   const uint32_t sync_id =
      agx_create_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM);
   CHECK(sync_id != UINT32_MAX && creates == 1 && dev.queue_flags == 0);
   async = true;
   id = agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM, &async);
   CHECK(id == sync_id && creates == 1 && !async);
   simple_mtx_destroy(&dev.vma_lock);

   /* AGX_DBG_1QUEUE, async queue first (Gallium): another async caller shares
    * it; a caller that did not ask for async is refused, not served it. */
   reset(&dev, DRM_ASAHI_FEATURE_ASYNC_SUBMIT, AGX_DBG_1QUEUE);
   async = false;
   const uint32_t async_id =
      agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM, &async);
   CHECK(async_id != UINT32_MAX && async && creates == 1);
   CHECK(dev.queue_flags == DRM_ASAHI_QUEUE_CREATE_ASYNC_SUBMIT);
   async = false;
   id = agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM, &async);
   CHECK(id == async_id && async && creates == 1);
   errno = 0;
   id = agx_create_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM);
   CHECK(id == UINT32_MAX && errno == EINVAL && creates == 1);
   simple_mtx_destroy(&dev.vma_lock);

   /* AGX_DBG_1QUEUE without the feature: the shared queue is synchronous for
    * everyone. */
   reset(&dev, 0, AGX_DBG_1QUEUE);
   async = true;
   id = agx_create_async_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM, &async);
   CHECK(id != UINT32_MAX && !async && sent_flags == 0);
   CHECK(agx_create_command_queue(&dev, DRM_ASAHI_PRIORITY_MEDIUM) == id);
   simple_mtx_destroy(&dev.vma_lock);

   printf("async_queue: OK\n");
   return 0;
}
