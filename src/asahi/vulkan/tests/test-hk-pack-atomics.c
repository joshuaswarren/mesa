/* SPDX-License-Identifier: MIT
 * Packed-atomic gating across lazy and aliased queues without a DRM device.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_alloc.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

static void
init_queue(struct hk_device *dev, struct hk_queue *queue, uint32_t family,
           uint32_t drm_id)
{
   const VkDeviceQueueCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
   };
   memset(queue, 0, sizeof(*queue));
   CHECK(vk_queue_init(&queue->vk, &dev->vk, &info, 0) == VK_SUCCESS);
   queue->drm.id = drm_id;
   queue->drm.virt_ring_idx = DRM_ASAHI_PRIORITY_MEDIUM + 1;
   queue->sib.count = 1;
}

int
main(void)
{
   unsetenv("HK_PACK_ATOMICS");
   unsetenv("ASAHI_M5_PACK_ATOMICS");

   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_IMMEDIATE;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->vk.queues);
   dev->dev.chip = AGX_CHIP_G17P;
   dev->dev.fd = -1;

   struct hk_queue q0, q1, host_as;

   /* Single non-emulated queue on G17P: packed atomics allowed. */
   init_queue(dev, &q0, 0, 10);
   CHECK(hk_pack_atomics_allowed(dev, &q0));

   /* A second non-emulated queue that has not yet lazily created its kernel
    * queue (drm.id == UINT32_MAX) can still admit and submit concurrently with
    * an in-flight packed atomic chain: must disable packed atomics. */
   init_queue(dev, &q1, 1, UINT32_MAX);
   CHECK(!hk_pack_atomics_allowed(dev, &q0));
   CHECK(!hk_pack_atomics_allowed(dev, &q1));

   /* Once q1 is aliased/emulated onto q0, only one kernel queue can exist. */
   vk_queue_set_emulated(&q1.vk, &q0.vk);
   CHECK(hk_pack_atomics_allowed(dev, &q0));
   vk_queue_finish(&q1.vk);

   /* A non-emulated host-AS queue with lazy drm.id == UINT32_MAX must also
    * disable packed atomics; an emulated one sharing q0 does not. */
   init_queue(dev, &host_as, 1, UINT32_MAX);
   dev->host_as.queue = &host_as;
   CHECK(!hk_pack_atomics_allowed(dev, &q0));
   vk_queue_set_emulated(&host_as.vk, &q0.vk);
   CHECK(hk_pack_atomics_allowed(dev, &q0));
   dev->host_as.queue = NULL;
   vk_queue_finish(&host_as.vk);

   /* A watched compute-kick queue runs an extra kernel queue outside
    * dev->vk.queues and must disable packed atomics. */
   q0.kick.watched = true;
   CHECK(!hk_pack_atomics_allowed(dev, &q0));
   q0.kick.watched = false;
   CHECK(hk_pack_atomics_allowed(dev, &q0));

   /* G16G serializes all compute commands of a VM on one firmware queue, so a
    * second VkQueue does not block packed atomics there. */
   dev->dev.chip = AGX_CHIP_G16G;
   init_queue(dev, &q1, 1, UINT32_MAX);
   CHECK(hk_pack_atomics_allowed(dev, &q0));
   vk_queue_finish(&q1.vk);

   vk_queue_finish(&q0.vk);
   vk_object_base_finish(&dev->vk.base);
   free(pdev);
   free(dev);
   puts("packed atomics lazy queue gating: PASS");
   return 0;
}
