/* SPDX-License-Identifier: MIT
 * Queue admission against an in-memory backend. No DRM device or GPU is used.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_alloc.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_instance.h"
#include "vk_semaphore.h"
#include "vk_sync.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)
struct test_sync {
   struct vk_sync base;
   bool signalled;
};
static unsigned creates, submits, resets, moves, signals;
static int create_error;
static uint64_t created_layouts[2];

static VkResult
sync_init(struct vk_device *d, struct vk_sync *s, uint64_t value)
{
   ((struct test_sync *)s)->signalled = value != 0;
   return VK_SUCCESS;
}
static void
sync_finish(struct vk_device *d, struct vk_sync *s)
{
}
static VkResult
sync_reset(struct vk_device *d, struct vk_sync *s)
{
   resets++;
   ((struct test_sync *)s)->signalled = false;
   return VK_SUCCESS;
}
static VkResult
sync_signal(struct vk_device *d, struct vk_sync *s, uint64_t value)
{
   signals++;
   ((struct test_sync *)s)->signalled = true;
   return VK_SUCCESS;
}
static VkResult
sync_move(struct vk_device *d, struct vk_sync *dst, struct vk_sync *src)
{
   moves++;
   ((struct test_sync *)dst)->signalled = ((struct test_sync *)src)->signalled;
   ((struct test_sync *)src)->signalled = false;
   return VK_SUCCESS;
}
static VkResult
sync_wait(struct vk_device *d, struct vk_sync *s, uint64_t value,
          enum vk_sync_wait_flags flags, uint64_t timeout)
{
   return ((struct test_sync *)s)->signalled ? VK_SUCCESS : VK_TIMEOUT;
}
static const struct vk_sync_type sync_type = {
   .size = sizeof(struct test_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_GPU_MULTI_WAIT | VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = sync_init,
   .finish = sync_finish,
   .reset = sync_reset,
   .signal = sync_signal,
   .wait = sync_wait,
   .move = sync_move,
};
static int
queue_ioctl(struct agx_device *dev, unsigned request, void *arg)
{
   CHECK(_IOC_NR(request) == DRM_COMMAND_BASE + DRM_ASAHI_QUEUE_CREATE);
   creates++;
   if (create_error) {
      errno = create_error;
      return -1;
   }
   const struct drm_asahi_queue_create *qc = arg;
   created_layouts[0] = qc->render_scs_layout[0];
   created_layouts[1] = qc->render_scs_layout[1];
   ((struct drm_asahi_queue_create *)arg)->queue_id = 42;
   return 0;
}
static VkResult
driver_submit(struct vk_queue *q, struct vk_queue_submit *s)
{
   submits++;
   for (unsigned i = 0; i < s->wait_count; i++)
      CHECK(((struct test_sync *)s->waits[i].sync)->signalled);
   for (unsigned i = 0; i < s->signal_count; i++)
      CHECK(vk_sync_signal(q->base.device, s->signals[i].sync, 0) ==
            VK_SUCCESS);
   return VK_SUCCESS;
}
static struct vk_semaphore *
new_sem(struct vk_device *dev, bool value)
{
   struct vk_semaphore *s = calloc(1, sizeof(*s) + sync_type.size);
   CHECK(s);
   vk_object_base_init(dev, &s->base, VK_OBJECT_TYPE_SEMAPHORE);
   s->type = VK_SEMAPHORE_TYPE_BINARY;
   CHECK(vk_sync_init(dev, &s->permanent, &sync_type, 0, value) == VK_SUCCESS);
   return s;
}

/* mode: 0 = Submit2, 1 = legacy Submit, 2 = opaque sparse bind,
 * 3 = empty recording, 4 = tiled sparse bind. */
static void
run(enum agx_chip chip, unsigned mode, bool temporary, bool threaded,
    int error)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && pdev && cmd);
   struct vk_instance instance = {0};
   vk_object_base_init(NULL, &instance.base, VK_OBJECT_TYPE_INSTANCE);
   instance.base.client_visible = true;
   list_inithead(&instance.debug_utils.callbacks);
   list_inithead(&instance.debug_report.callbacks);
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   pdev->vk.instance = &instance;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk._lost.reported = true;
   dev->vk.submit_mode = threaded ? VK_QUEUE_SUBMIT_MODE_THREADED
                                  : VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NONE;
   dev->vk.dispatch_table.QueueSubmit2 = hk_QueueSubmit2;
   list_inithead(&dev->vk.queues);
   dev->dev.chip = chip;
   dev->dev.ops.simple_ioctl = queue_ioctl;

   struct hk_queue queue = {0};
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1};
   CHECK(vk_queue_init(&queue.vk, &dev->vk, &qi, 0) == VK_SUCCESS);
   queue.vk.driver_submit = driver_submit;
   queue.drm.id = UINT32_MAX;
   queue.drm.virt_ring_idx = 1;
   struct vk_command_pool pool = {.queue_family_index = 0};
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->vk.pool = &pool;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   list_inithead(&cmd->control_streams);
   struct hk_cs cs = {0};
   if (mode != 3)
      list_addtail(&cs.node, &cmd->control_streams);
   VkCommandBuffer cb = hk_cmd_buffer_to_handle(cmd);
   struct vk_semaphore *sem[3] = {new_sem(&dev->vk, true),
                                  new_sem(&dev->vk, false),
                                  new_sem(&dev->vk, false)};
   if (temporary)
      CHECK(vk_sync_create(&dev->vk, &sync_type, 0, 1, &sem[0]->temporary) ==
            VK_SUCCESS);
   struct vk_sync *payload = sem[0]->temporary;
   VkSemaphore handles[3];
   for (unsigned i = 0; i < 3; i++)
      handles[i] = vk_semaphore_to_handle(sem[i]);
   VkSemaphoreSubmitInfo wait = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = handles[0],
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   VkSemaphoreSubmitInfo signal[2] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = handles[1]},
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .semaphore = handles[2]}};
   VkCommandBufferSubmitInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cb};
   VkSubmitInfo2 si[2] = {
      {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
       .signalSemaphoreInfoCount = 1,
       .pSignalSemaphoreInfos = &signal[0]},
      {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
       .commandBufferInfoCount = 1,
       .pCommandBufferInfos = &ci,
       .waitSemaphoreInfoCount = 1,
       .pWaitSemaphoreInfos = &wait,
       .signalSemaphoreInfoCount = 1,
       .pSignalSemaphoreInfos = &signal[1]},
   };
   VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VkSubmitInfo old[2] = {
      {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
       .signalSemaphoreCount = 1,
       .pSignalSemaphores = &handles[1]},
      {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
       .commandBufferCount = 1,
       .pCommandBuffers = &cb,
       .waitSemaphoreCount = 1,
       .pWaitSemaphores = &handles[0],
       .pWaitDstStageMask = &stage,
       .signalSemaphoreCount = 1,
       .pSignalSemaphores = &handles[2]},
   };
   VkSparseMemoryBind binding = {.size = 4096};
   VkSparseImageOpaqueMemoryBindInfo image = {.bindCount = 1,
                                              .pBinds = &binding};
   VkSparseImageMemoryBind tiled_binding = {.extent = {1, 1, 1}};
   VkSparseImageMemoryBindInfo tiled_image = {.bindCount = 1,
                                              .pBinds = &tiled_binding};
   VkBindSparseInfo bi[2] = {
      {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
       .signalSemaphoreCount = 1,
       .pSignalSemaphores = &handles[1]},
      {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
       .imageOpaqueBindCount = 1,
       .pImageOpaqueBinds = &image,
       .waitSemaphoreCount = 1,
       .pWaitSemaphores = &handles[0],
       .signalSemaphoreCount = 1,
       .pSignalSemaphores = &handles[2]},
   };
   if (mode == 4) {
      bi[1].imageOpaqueBindCount = 0;
      bi[1].imageBindCount = 1;
      bi[1].pImageBinds = &tiled_image;
   }
   creates = submits = resets = moves = signals = 0;
   create_error = error;
   VkQueue q = vk_queue_to_handle(&queue.vk);
   const bool terminal =
      mode != 3 && (error == EIO || error == EPERM || error == EACCES);
   for (unsigned attempt = 0; attempt < (mode == 3 || terminal ? 1 : 2);
        attempt++) {
      VkResult r = mode == 1 ? vk_common_QueueSubmit(q, 2, old, VK_NULL_HANDLE)
                   : (mode == 2 || mode == 4)
                      ? hk_QueueBindSparse(q, 2, bi, VK_NULL_HANDLE)
                      : hk_QueueSubmit2(q, 2, si, VK_NULL_HANDLE);
      if (mode != 3 && attempt == 0) {
         CHECK(r == (terminal          ? VK_ERROR_DEVICE_LOST
                     : error == ENOMEM ? VK_ERROR_OUT_OF_HOST_MEMORY
                                       : VK_ERROR_OUT_OF_DEVICE_MEMORY));
         CHECK(creates == 1 && submits == 0 && resets == 0 && moves == 0 &&
               signals == 0);
         CHECK(sem[0]->temporary == payload);
         CHECK(((struct test_sync *)&sem[0]->permanent)->signalled);
         CHECK(vk_device_is_lost_no_report(&dev->vk) == terminal);
         CHECK(cmd->vk.state == MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE);
         create_error = 0;
      } else
         CHECK(r == VK_SUCCESS);
   }
   vk_queue_finish(
      &queue.vk); /* Joins worker and all host-side sync cleanup. */
   CHECK(submits == (terminal ? 0 : 2) && signals == (terminal ? 0 : 2));
   CHECK(creates == (mode == 3 ? 0 : terminal ? 1 : 2));
   CHECK(sem[0]->temporary == (terminal ? payload : NULL));
   CHECK(vk_device_is_lost_no_report(&dev->vk) == terminal);
   if (sem[0]->temporary)
      vk_sync_destroy(&dev->vk, sem[0]->temporary);
   for (unsigned i = 0; i < 3; i++) {
      vk_sync_finish(&dev->vk, &sem[i]->permanent);
      vk_object_base_finish(&sem[i]->base);
      free(sem[i]);
   }
   vk_object_base_finish(&cmd->vk.base);
   free(cmd);
   vk_object_base_finish(&dev->vk.base);
   free(dev);
   vk_object_base_finish(&pdev->vk.base);
   free(pdev);
   vk_object_base_finish(&instance.base);
   printf("chip=%u mode=%u temporary=%u threaded=%u errno=%d: PASS\n", chip,
          mode, temporary, threaded, error);
}

/* G17P: admission creates a missing kernel queue with the render ray-query
 * scratch a call needs, and never replaces an existing queue (the driver
 * callback does that, ordered after the old queue's work).
 * rq_first: the first call already needs the scratch. */
static void
run_rq(bool rq_first)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && pdev && cmd);
   struct vk_instance instance = {0};
   vk_object_base_init(NULL, &instance.base, VK_OBJECT_TYPE_INSTANCE);
   instance.base.client_visible = true;
   list_inithead(&instance.debug_utils.callbacks);
   list_inithead(&instance.debug_report.callbacks);
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   pdev->vk.instance = &instance;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk._lost.reported = true;
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NONE;
   dev->vk.dispatch_table.QueueSubmit2 = hk_QueueSubmit2;
   list_inithead(&dev->vk.queues);
   dev->dev.chip = AGX_CHIP_G17P;
   dev->dev.ops.simple_ioctl = queue_ioctl;
   dev->gen1_native_fragment_rq = true;

   struct hk_queue queue = {0};
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1};
   CHECK(vk_queue_init(&queue.vk, &dev->vk, &qi, 0) == VK_SUCCESS);
   queue.vk.driver_submit = driver_submit;
   queue.drm.id = UINT32_MAX;
   queue.drm.virt_ring_idx = 1;
   struct vk_command_pool pool = {.queue_family_index = 0};
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->vk.pool = &pool;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   list_inithead(&cmd->control_streams);
   struct hk_cs cs = {.type = HK_CS_VDM};
   list_addtail(&cs.node, &cmd->control_streams);
   VkCommandBufferSubmitInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = hk_cmd_buffer_to_handle(cmd)};
   VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                       .commandBufferInfoCount = 1,
                       .pCommandBufferInfos = &ci};
   VkQueue q = vk_queue_to_handle(&queue.vk);
   creates = submits = 0;
   create_error = 0;
   created_layouts[0] = created_layouts[1] = 0;

   /* A refused first call leaves the queue missing and retryable. */
   cs.gen1_native_fragment_rq = rq_first;
   create_error = ENOSPC;
   CHECK(hk_QueueSubmit2(q, 1, &si, VK_NULL_HANDLE) ==
         VK_ERROR_OUT_OF_DEVICE_MEMORY);
   CHECK(creates == 1 && submits == 0 && queue.drm.id == UINT32_MAX);
   CHECK(!vk_device_is_lost_no_report(&dev->vk));
   create_error = 0;

   CHECK(hk_QueueSubmit2(q, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
   CHECK(creates == 2 && submits == 1 && queue.drm.id == 42);
   CHECK(queue.drm.render_rq == (rq_first ? HK_RENDER_RQ_FRAGMENT : 0));
   CHECK(created_layouts[0] == 0 && (created_layouts[1] != 0) == rq_first);

   /* A later call that needs the scratch never replaces at admission, and
    * steady-state submissions bypass admission without re-entering queue
    * creation or holding the queue lock. */
   cs.gen1_native_fragment_rq = true;
   create_error = ENOSPC;
   CHECK(hk_QueueSubmit2(q, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
   CHECK(creates == 2 && submits == 2 && queue.drm.id == 42);
   CHECK(queue.drm.render_rq == (rq_first ? HK_RENDER_RQ_FRAGMENT : 0));

   /* Emulated queues sharing an admitted backing queue take the same
    * lockless fast path. */
   struct hk_queue alias = {0};
   CHECK(vk_queue_init(&alias.vk, &dev->vk, &qi, 0) == VK_SUCCESS);
   alias.vk.driver_submit = NULL;
   alias.drm.id = UINT32_MAX;
   vk_queue_set_emulated(&alias.vk, &queue.vk);
   CHECK(hk_QueueSubmit2(vk_queue_to_handle(&alias.vk), 1, &si,
                         VK_NULL_HANDLE) == VK_SUCCESS);
   CHECK(creates == 2 && submits == 3 && queue.drm.id == 42 &&
         alias.drm.id == UINT32_MAX);
   vk_queue_finish(&alias.vk);
   create_error = 0;

   vk_queue_finish(&queue.vk);
   CHECK(!vk_device_is_lost_no_report(&dev->vk));
   vk_object_base_finish(&cmd->vk.base);
   free(cmd);
   vk_object_base_finish(&dev->vk.base);
   free(dev);
   vk_object_base_finish(&pdev->vk.base);
   free(pdev);
   vk_object_base_finish(&instance.base);
   printf("g17p render ray-query rq_first=%u: PASS\n", rq_first);
}
int
main(void)
{
   const int errors[] = {EBUSY, ENOSPC, ENOMEM, EIO, EPERM, EACCES};
   const enum agx_chip chips[] = {AGX_CHIP_G17C, AGX_CHIP_G17P};
   for (unsigned c = 0; c < ARRAY_SIZE(chips); c++)
      for (unsigned mode = 0; mode < 5; mode++)
         for (unsigned temp = 0; temp < 2; temp++)
            for (unsigned thread = 0; thread < 2; thread++)
               for (unsigned e = 0; e < ARRAY_SIZE(errors); e++)
                  run(chips[c], mode, temp, thread, errors[e]);
   run_rq(false);
   run_rq(true);
   return 0;
}
