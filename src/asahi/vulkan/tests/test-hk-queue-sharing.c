/* SPDX-License-Identifier: MIT
 * Queue selection and callback serialization without a DRM device.
 */
#include <stdio.h>
#include <stdlib.h>
#include "c11/threads.h"
#include "util/simple_mtx.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

#if !UTIL_FUTEX_SUPPORTED
#error "observing contention on the backing queue lock needs futex simple_mtx"
#endif

static struct hk_queue queues[6];
static int entered, released, finished, independent, active;

/* Flags change under `lock` and are broadcast on `cond`. */
static mtx_t lock;
static cnd_t cond;

static void
bump(int *flag)
{
   mtx_lock(&lock);
   p_atomic_inc(flag);
   cnd_broadcast(&cond);
   mtx_unlock(&lock);
}

static void
await(int *flag)
{
   mtx_lock(&lock);
   while (!p_atomic_read(flag))
      CHECK(cnd_wait(&cond, &lock) == thrd_success);
   mtx_unlock(&lock);
}

static VkResult
driver_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   if (queue == &queues[2].vk) {
      bump(&independent);
      return VK_SUCCESS;
   }
   CHECK(queue == &queues[0].vk);
   CHECK(p_atomic_inc_return(&active) == 1);
   bump(&entered);
   await(&released);
   p_atomic_inc(&finished);
   p_atomic_dec(&active);
   return VK_SUCCESS;
}

static void
submit(unsigned index)
{
   VkSubmitInfo2 info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
   CHECK(hk_QueueSubmit2(vk_queue_to_handle(&queues[index].vk), 1, &info,
                         VK_NULL_HANDLE) == VK_SUCCESS);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   CHECK(mtx_init(&lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&cond) == thrd_success);
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->vk.queues);
   dev->dev.chip = AGX_CHIP_G17C;

   for (unsigned i = 0; i < ARRAY_SIZE(queues); i++) {
      VkDeviceQueueCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .flags = i == 3 ? VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT : 0,
         .queueFamilyIndex = i % 3,
         .queueCount = 1};
      CHECK(vk_queue_init(&queues[i].vk, &dev->vk, &info, 0) == VK_SUCCESS);
      queues[i].drm.virt_ring_idx = i == 2 ? 1 : 2;
      queues[i].drm.id = 42 + i;
      queues[i].vk.driver_submit = driver_submit;
      if (i == 5)
         dev->dev.chip = AGX_CHIP_G17P;
      hk_queue_share(dev, &queues[i]);
   }
   dev->dev.chip = AGX_CHIP_G17C;
   CHECK(queues[0].vk.real_queue == &queues[0].vk);
   CHECK(queues[1].vk.real_queue == &queues[0].vk);
   CHECK(!queues[2].vk.real_queue);
   CHECK(!queues[3].vk.real_queue);
   CHECK(queues[4].vk.real_queue == &queues[0].vk);
   CHECK(!queues[5].vk.real_queue);
   CHECK(!queues[1].vk.driver_submit && !queues[4].vk.driver_submit);

   submit(0);
   await(&entered);
   submit(1);
   submit(2);
   await(&independent);
   /* The alias submission is serialized behind the paused one on the
    * backing queue's lock. A futex simple_mtx holds 2 exactly while another
    * thread waits for it, and only the alias' submit thread can contend
    * here, so wait for that state; if serialization were broken the alias
    * would enter driver_submit instead, which the loop also stops on. */
   while (p_atomic_read(&queues[0].vk.lock.val) != 2 &&
          p_atomic_read(&entered) == 1)
      thrd_yield();
   CHECK(p_atomic_read(&entered) == 1);
   bump(&released);

   /* Match device teardown: aliases must stop before the backing queue. */
   while (!list_is_empty(&dev->vk.queues)) {
      struct vk_queue *queue =
         list_last_entry(&dev->vk.queues, struct vk_queue, link);
      vk_queue_finish(queue);
   }
   CHECK(p_atomic_read(&entered) == 2);
   CHECK(p_atomic_read(&finished) == 2);
   vk_object_base_finish(&dev->vk.base);
   cnd_destroy(&cond);
   mtx_destroy(&lock);
   free(pdev);
   free(dev);
   puts(
      "queue matching, serialization, independent priority, reverse finish: PASS");
   return 0;
}
