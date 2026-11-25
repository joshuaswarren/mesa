/* SPDX-License-Identifier: MIT
 * Exercise real object teardown against a paused host submission. No GPU used.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <xf86drm.h>
#include "c11/threads.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_fence.h"
#include "vk_semaphore.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)
/* Every flag is set under `lock` and broadcast on `cond`, so each step waits
 * for the event it depends on rather than for a time. */
static mtx_t lock;
static cnd_t cond;
static int entered, released, quiescing, returned, cleanup_done, calls;
static bool pending_wait;
struct test_sync {
   struct vk_sync base;
   unsigned marker;
};
static void
set(int *flag)
{
   mtx_lock(&lock);
   p_atomic_set(flag, 1);
   cnd_broadcast(&cond);
   mtx_unlock(&lock);
}
static void
await_any(int *a, int *b)
{
   mtx_lock(&lock);
   while (!p_atomic_read(a) && !p_atomic_read(b))
      CHECK(cnd_wait(&cond, &lock) == thrd_success);
   mtx_unlock(&lock);
}
static void
await(int *flag)
{
   await_any(flag, flag);
}
/* Object teardown on a lost device joins the queue's host submissions here.
 * Report entry so the test releases the paused submission only once the
 * cleanup thread is committed to waiting for it. */
void __real_vk_queue_wait_lost_submissions(struct vk_queue *queue);
void __wrap_vk_queue_wait_lost_submissions(struct vk_queue *queue);
void
__wrap_vk_queue_wait_lost_submissions(struct vk_queue *queue)
{
   set(&quiescing);
   __real_vk_queue_wait_lost_submissions(queue);
}
static VkResult
sync_init(struct vk_device *dev, struct vk_sync *sync, uint64_t value)
{
   ((struct test_sync *)sync)->marker = 1234;
   return VK_SUCCESS;
}
static void
sync_finish(struct vk_device *dev, struct vk_sync *sync)
{
}
static VkResult
sync_signal(struct vk_device *dev, struct vk_sync *sync, uint64_t value)
{
   CHECK(((struct test_sync *)sync)->marker == 1234);
   return VK_SUCCESS;
}
static VkResult
sync_get_value(struct vk_device *dev, struct vk_sync *sync, uint64_t *value)
{
   CHECK(((struct test_sync *)sync)->marker == 1234);
   *value = 0;
   return VK_SUCCESS;
}
static VkResult
sync_wait(struct vk_device *dev, struct vk_sync *sync, uint64_t value,
          enum vk_sync_wait_flags flags, uint64_t timeout)
{
   CHECK(flags & VK_SYNC_WAIT_PENDING);
   CHECK(pending_wait);
   set(&entered);
   await(&released);
   CHECK(((struct test_sync *)sync)->marker == 1234);
   set(&returned);
   return VK_ERROR_DEVICE_LOST;
}
static const struct vk_sync_type sync_type = {
   .size = sizeof(struct test_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE |
               VK_SYNC_FEATURE_GPU_WAIT | VK_SYNC_FEATURE_GPU_MULTI_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = sync_init,
   .finish = sync_finish,
   .signal = sync_signal,
   .get_value = sync_get_value,
   .wait = sync_wait,
};
static VkResult
driver_submit(struct vk_queue *q, struct vk_queue_submit *s)
{
   p_atomic_inc(&calls);
   set(&entered);
   await(&released);
   CHECK(s->buffer_bind_count == 1 && s->image_opaque_bind_count == 1);
   VK_FROM_HANDLE(hk_buffer, buffer, s->buffer_binds[0].buffer);
   VK_FROM_HANDLE(hk_image, image, s->image_opaque_binds[0].image);
   VK_FROM_HANDLE(hk_device_memory, mem, s->buffer_binds[0].pBinds[0].memory);
   CHECK(buffer->vk.size == 4096 && image->vk.extent.width == 64);
   CHECK(mem->bo->size == 4096 && mem->bo->refcnt == 2);
   CHECK(s->signal_count == 2);
   for (unsigned i = 0; i < s->signal_count; i++)
      CHECK(vk_sync_signal(q->base.device, s->signals[i].sync,
                           s->signals[i].signal_value) == VK_SUCCESS);
   set(&returned);
   return VK_SUCCESS;
}
struct fixture {
   struct hk_device *dev;
   struct hk_buffer *buffer;
   struct hk_image *image;
   struct hk_device_memory *memory;
   struct vk_semaphore *semaphore;
   struct vk_fence *fence;
   unsigned kind;
};
static void
destroy(struct fixture *f, unsigned kind)
{
   VkDevice dev = hk_device_to_handle(f->dev);
   switch (kind) {
   case 0:
      hk_DestroyBuffer(dev, hk_buffer_to_handle(f->buffer), NULL);
      f->buffer = NULL;
      break;
   case 1:
      hk_DestroyImage(dev, hk_image_to_handle(f->image), NULL);
      f->image = NULL;
      break;
   case 2:
      hk_FreeMemory(dev, hk_device_memory_to_handle(f->memory), NULL);
      f->memory = NULL;
      break;
   case 3:
      hk_DestroySemaphore(dev, vk_semaphore_to_handle(f->semaphore), NULL);
      f->semaphore = NULL;
      break;
   case 4:
      hk_DestroyFence(dev, vk_fence_to_handle(f->fence), NULL);
      f->fence = NULL;
      break;
   default:
      abort();
   }
}
static int
cleanup(void *data)
{
   struct fixture *f = data;
   destroy(f, f->kind);
   /* The object may only go once the submission that uses it has returned. */
   CHECK(p_atomic_read(&returned));
   set(&cleanup_done);
   return 0;
}
static void
run(enum agx_chip chip, unsigned kind, bool wait)
{
   struct fixture f = {.kind = kind};
   f.dev = calloc(1, sizeof(*f.dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(f.dev && pdev);
   struct vk_device *dev = &f.dev->vk;
   vk_object_base_init(dev, &dev->base, VK_OBJECT_TYPE_DEVICE);
   dev->alloc = *vk_default_allocator();
   dev->physical = &pdev->vk;
   dev->_lost.reported = true;
   dev->submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED;
   dev->timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->queues);
   f.dev->dev.chip = chip;
   uint64_t used = 4096;
   pdev->mem_heaps[0].used = &used;
   struct agx_bo bo = {.size = 4096, .refcnt = 2};
   f.buffer = calloc(1, sizeof(*f.buffer));
   f.image = calloc(1, sizeof(*f.image));
   f.memory = calloc(1, sizeof(*f.memory));
   f.semaphore =
      calloc(1, offsetof(struct vk_semaphore, permanent) + sync_type.size);
   f.fence = calloc(1, offsetof(struct vk_fence, permanent) + sync_type.size);
   CHECK(f.buffer && f.image && f.memory && f.semaphore && f.fence);
   vk_object_base_init(dev, &f.buffer->vk.base, VK_OBJECT_TYPE_BUFFER);
   vk_object_base_init(dev, &f.image->vk.base, VK_OBJECT_TYPE_IMAGE);
   vk_object_base_init(dev, &f.memory->vk.base, VK_OBJECT_TYPE_DEVICE_MEMORY);
   vk_object_base_init(dev, &f.semaphore->base, VK_OBJECT_TYPE_SEMAPHORE);
   vk_object_base_init(dev, &f.fence->base, VK_OBJECT_TYPE_FENCE);
   f.buffer->vk.size = 4096;
   f.image->vk.extent.width = 64;
   f.memory->bo = &bo;
   f.semaphore->type = VK_SEMAPHORE_TYPE_TIMELINE;
   CHECK(vk_sync_init(dev, &f.semaphore->permanent, &sync_type,
                      VK_SYNC_IS_TIMELINE, 0) == VK_SUCCESS);
   CHECK(vk_sync_init(dev, &f.fence->permanent, &sync_type, 0, 0) ==
         VK_SUCCESS);
   struct hk_queue q = {0};
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1};
   CHECK(vk_queue_init(&q.vk, dev, &qi, 0) == VK_SUCCESS);
   q.vk.driver_submit = driver_submit;
   VkSparseMemoryBind binding = {
      .size = 4096,
      .memory = hk_device_memory_to_handle(f.memory)};
   VkSparseBufferMemoryBindInfo buffer = {
      .buffer = hk_buffer_to_handle(f.buffer),
      .bindCount = 1,
      .pBinds = &binding};
   VkSparseImageOpaqueMemoryBindInfo image = {
      .image = hk_image_to_handle(f.image),
      .bindCount = 1,
      .pBinds = &binding};
   VkSemaphore sem = vk_semaphore_to_handle(f.semaphore);
   uint64_t value = 2, wait_value = 1;
   VkTimelineSemaphoreSubmitInfo timeline = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &value,
      .waitSemaphoreValueCount = wait,
      .pWaitSemaphoreValues = &wait_value};
   VkBindSparseInfo bi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
                          .pNext = &timeline,
                          .bufferBindCount = 1,
                          .pBufferBinds = &buffer,
                          .imageOpaqueBindCount = 1,
                          .pImageOpaqueBinds = &image,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &sem,
                          .waitSemaphoreCount = wait,
                          .pWaitSemaphores = &sem};
   entered = released = quiescing = returned = cleanup_done = calls = 0;
   pending_wait = wait;
   CHECK(vk_common_QueueBindSparse(vk_queue_to_handle(&q.vk), 1, &bi,
                                   vk_fence_to_handle(f.fence)) == VK_SUCCESS);
   await(&entered);
   p_atomic_set(&dev->_lost.lost, 1);
   thrd_t thread;
   CHECK(thrd_create(&thread, cleanup, &f) == thrd_success);
   /* Teardown must join the paused submission, not finish around it. */
   await_any(&quiescing, &cleanup_done);
   CHECK(p_atomic_read(&quiescing) && !p_atomic_read(&cleanup_done));
   set(&released);
   await(&cleanup_done);
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(p_atomic_read(&calls) == !wait);
   vk_queue_finish(&q.vk);
   for (unsigned other = 0; other < 5; other++)
      if (other != kind)
         destroy(&f, other);
   CHECK(bo.refcnt == 1 && used == 0);
   vk_object_base_finish(&dev->base);
   free(f.dev);
   free(pdev);
   printf("chip=%u object=%u pending_wait=%u: PASS\n", chip, kind, wait);
}

static unsigned mock_query_calls, mock_wait_calls;
static unsigned mock_eintr_remaining;
static int64_t mock_last_timeout;
static uint32_t mock_waited_handles[4];
static uint64_t mock_waited_points[4];
static unsigned mock_waited_count;

int
drmSyncobjQuery2(int fd, uint32_t *handles, uint64_t *points,
                 uint32_t handle_count, uint32_t flags)
{
   (void)fd;
   CHECK(handle_count == 1);
   CHECK(flags == DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED);
   ++mock_query_calls;
   switch (handles[0]) {
   case 10:
      return -EOPNOTSUPP;
   case 20:
      points[0] = 0;
      return 0;
   case 30:
      points[0] = 7;
      return 0;
   case 40:
      points[0] = 9;
      return 0;
   default:
      abort();
   }
}

int
drmSyncobjTimelineWait(int fd, uint32_t *handles, uint64_t *points,
                       unsigned num_handles, int64_t timeout_nsec,
                       unsigned flags, uint32_t *first_signaled)
{
   (void)fd;
   (void)first_signaled;
   CHECK(num_handles == 1);
   CHECK(flags == 0);
   CHECK(timeout_nsec > 0);
   if (mock_wait_calls == 0)
      mock_last_timeout = timeout_nsec;
   else
      CHECK(timeout_nsec == mock_last_timeout);
   ++mock_wait_calls;
   if (handles[0] == 30) {
      if (mock_eintr_remaining > 0) {
         --mock_eintr_remaining;
         return -EINTR;
      }
      CHECK(mock_waited_count < ARRAY_SIZE(mock_waited_handles));
      mock_waited_handles[mock_waited_count] = handles[0];
      mock_waited_points[mock_waited_count++] = points[0];
      return 0;
   }
   if (handles[0] == 40) {
      CHECK(mock_waited_count < ARRAY_SIZE(mock_waited_handles));
      mock_waited_handles[mock_waited_count] = handles[0];
      mock_waited_points[mock_waited_count++] = points[0];
      return -ETIME;
   }
   abort();
}

static void
test_wait_submitted(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->dev.fd = -1;
   list_inithead(&dev->vk.queues);

   struct hk_queue queues[5] = {0};
   const uint32_t syncobjs[5] = {0, 10, 20, 30, 40};
   for (unsigned i = 0; i < ARRAY_SIZE(queues); ++i) {
      queues[i].drm.syncobj = syncobjs[i];
      list_addtail(&queues[i].vk.link, &dev->vk.queues);
   }

   mock_query_calls = 0;
   mock_wait_calls = 0;
   mock_eintr_remaining = 2;
   mock_last_timeout = 0;
   mock_waited_count = 0;

   hk_device_wait_submitted(dev, 5ull * 1000000000ull);

   /* syncobj=0 skipped before query; 10/20/30/40 queried; 30 retried twice on
    * -EINTR then succeeded; 40 timed out with -ETIME without hanging. */
   CHECK(mock_query_calls == 4);
   CHECK(mock_wait_calls == 4);
   CHECK(mock_eintr_remaining == 0);
   CHECK(mock_waited_count == 2);
   CHECK(mock_waited_handles[0] == 30 && mock_waited_points[0] == 7);
   CHECK(mock_waited_handles[1] == 40 && mock_waited_points[1] == 9);

   free(dev);
   printf("wait_submitted: PASS\n");
}

int
main(void)
{
   CHECK(mtx_init(&lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&cond) == thrd_success);
   const enum agx_chip chips[] = {AGX_CHIP_G13G, AGX_CHIP_G17P, AGX_CHIP_G17C};
   for (unsigned c = 0; c < ARRAY_SIZE(chips); c++) {
      for (unsigned kind = 0; kind < 5; kind++)
         run(chips[c], kind, false);
      run(chips[c], 3, true);
   }
   test_wait_submitted();
   cnd_destroy(&cond);
   mtx_destroy(&lock);
   return 0;
}
