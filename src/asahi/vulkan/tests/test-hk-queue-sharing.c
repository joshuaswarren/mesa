/* SPDX-License-Identifier: MIT
 * Kernel queue budget, queue selection and callback serialization without a
 * DRM device.
 */
#include <stdio.h>
#include <stdlib.h>
#include "c11/threads.h"
#include "util/simple_mtx.h"
#include "hk_compute_kick.h"
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

#define IQ DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES

/* drm.virt_ring_idx of the MEDIUM, LOW and HIGH priority classes. */
#define MEDIUM (DRM_ASAHI_PRIORITY_MEDIUM + 1)
#define LOW    (DRM_ASAHI_PRIORITY_LOW + 1)
#define HIGH   (DRM_ASAHI_PRIORITY_HIGH + 1)

#define PROTECTED VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT

static struct hk_device *dev;
static struct hk_physical_device *pdev;

/* The budget and the queue choice read nothing of the chip. */
static const enum agx_chip chips[] = {
   AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15S, AGX_CHIP_G16G,
   AGX_CHIP_G17P, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static void
set_kernel(uint64_t features, bool virtio, uint32_t max_queues,
           uint32_t max_in_flight_per_queue)
{
   dev->dev.params.features = features;
   dev->dev.is_virtio = virtio;
   dev->dev.debug = 0;
   pdev->queue_limits = (struct drm_asahi_queue_limits){
      .max_queues = max_queues,
      .max_in_flight_per_queue = max_in_flight_per_queue,
   };
}

static void
test_budget(void)
{
   /* One kernel queue per class without independent kernel queues, under
    * virtio where the limits cannot be queried, or under AGX_DBG_1QUEUE,
    * whatever the limits hold. */
   set_kernel(0, false, 58, 2);
   CHECK(hk_queue_independent_budget(dev) == 1);
   set_kernel(0, false, 0, 0);
   CHECK(hk_queue_independent_budget(dev) == 1);
   set_kernel(IQ, true, 58, 2);
   CHECK(hk_queue_independent_budget(dev) == 1);
   set_kernel(IQ, false, 58, 2);
   dev->dev.debug = AGX_DBG_1QUEUE;
   CHECK(hk_queue_independent_budget(dev) == 1);
   dev->dev.debug = 0;

   /* A pool of 58 or 42 kernel queues allows 11 or 8 Vulkan queues, one per
    * 1 + HK_QUEUE_SIBLINGS_MAX kernel queues, capped at the advertised count.
    * Render credits per kernel queue do not matter. */
   for (uint32_t credits = 1; credits <= 2; credits++) {
      set_kernel(IQ, false, 58, credits);
      CHECK(hk_queue_independent_budget(dev) == HK_ADVERTISED_QUEUES);
      set_kernel(IQ, false, 42, credits);
      CHECK(hk_queue_independent_budget(dev) == HK_ADVERTISED_QUEUES);
   }

   /* A scarce pool bounds the budget, but never below one. */
   set_kernel(IQ, false, 15, 2);
   CHECK(hk_queue_independent_budget(dev) == 3);
   set_kernel(IQ, false, 2 * (1 + HK_QUEUE_SIBLINGS_MAX) - 1, 2);
   CHECK(hk_queue_independent_budget(dev) == 1);
   set_kernel(IQ, false, 8, 2);
   CHECK(hk_queue_independent_budget(dev) == 1);
   set_kernel(IQ, false, 1, 1);
   CHECK(hk_queue_independent_budget(dev) == 1);

   /* A large pool: never more than the advertised count. */
   set_kernel(IQ, false, 4096, 2);
   CHECK(hk_queue_independent_budget(dev) == HK_ADVERTISED_QUEUES);
}

static struct hk_queue pool[3 * HK_ADVERTISED_QUEUES];
static unsigned pool_count;

typedef VkResult (*submit_fn)(struct vk_queue *queue,
                              struct vk_queue_submit *submit);

static VkResult
idle_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   return VK_SUCCESS;
}

/* Like hk_queue_create(): initialize, pick the class, then share. */
static void
create(struct hk_queue *queue, uint32_t family, uint8_t ring,
       VkDeviceQueueCreateFlags flags, submit_fn driver_submit)
{
   const VkDeviceQueueCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .flags = flags,
      .queueFamilyIndex = family,
      .queueCount = 1,
   };
   CHECK(vk_queue_init(&queue->vk, &dev->vk, &info, 0) == VK_SUCCESS);
   queue->drm.virt_ring_idx = ring;
   queue->vk.driver_submit = driver_submit;
   hk_queue_share(dev, queue);
}

static unsigned
add(uint8_t ring, VkDeviceQueueCreateFlags flags)
{
   CHECK(pool_count < ARRAY_SIZE(pool));
   create(&pool[pool_count], pool_count % 3, ring, flags, idle_submit);
   return pool_count++;
}

/* Index of the queue whose kernel queue pool[i] runs on. */
static unsigned
backing(unsigned i)
{
   const struct hk_queue *queue = &pool[i];
   if (!vk_queue_is_emulated(&queue->vk)) {
      CHECK(queue->vk.driver_submit == idle_submit);
      return i;
   }

   CHECK(!queue->vk.driver_submit);
   const struct hk_queue *real =
      container_of(queue->vk.real_queue, struct hk_queue, vk);
   CHECK(real >= pool && real < pool + pool_count);
   CHECK(!vk_queue_is_emulated(&real->vk));
   /* Sharing never crosses a priority class or creation flags, except under
    * AGX_DBG_1QUEUE where every queue maps to the single shared kernel queue. */
   if (!(dev->dev.debug & AGX_DBG_1QUEUE)) {
      CHECK(real->drm.virt_ring_idx == queue->drm.virt_ring_idx);
      CHECK(real->vk.flags == queue->vk.flags);
   }
   return real - pool;
}

static unsigned
independent(void)
{
   unsigned count = 0;
   for (unsigned i = 0; i < pool_count; i++)
      count += backing(i) == i;
   return count;
}

/* Match device teardown: aliases stop before their backing queues. */
static void
finish_all(void)
{
   while (!list_is_empty(&dev->vk.queues)) {
      struct vk_queue *queue =
         list_last_entry(&dev->vk.queues, struct vk_queue, link);
      vk_queue_finish(queue);
   }
   pool_count = 0;
}

/* Queues up to the budget keep their own kernel queue; later ones of a class
 * spread over its kernel queues, the least shared first. */
static void
test_spread(void)
{
   set_kernel(IQ, false, 15, 2);
   for (unsigned i = 0; i < 9; i++)
      add(MEDIUM, 0);
   const unsigned expect[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
   for (unsigned i = 0; i < ARRAY_SIZE(expect); i++)
      CHECK(backing(i) == expect[i]);
   CHECK(independent() == 3);
   finish_all();
}

/* Past the budget, a queue only shares a kernel queue of its own priority
 * class and creation flags, and gets one of its own if its class has none. */
static void
test_isolation(void)
{
   set_kernel(IQ, false, 10, 1);
   CHECK(hk_queue_independent_budget(dev) == 2);
   const unsigned m0 = add(MEDIUM, 0);
   const unsigned m1 = add(MEDIUM, 0);
   const unsigned l0 = add(LOW, 0);
   const unsigned p0 = add(MEDIUM, PROTECTED);
   const unsigned l1 = add(LOW, 0);
   const unsigned p1 = add(MEDIUM, PROTECTED);
   const unsigned m2 = add(MEDIUM, 0);
   const unsigned h0 = add(HIGH, 0);
   const unsigned m3 = add(MEDIUM, 0);
   const unsigned h1 = add(HIGH, 0);

   CHECK(backing(m0) == m0 && backing(m1) == m1);
   CHECK(backing(l0) == l0);
   CHECK(backing(p0) == p0);
   CHECK(backing(l1) == l0);
   CHECK(backing(p1) == p0);
   CHECK(backing(m2) == m0 && backing(m3) == m1);
   CHECK(backing(h0) == h0);
   CHECK(backing(h1) == h0);
   CHECK(independent() == 5);
   finish_all();
}

/* Kernels with independent queues report pools of 42 or 58 kernel queues.
 * Every queue of one universal family then keeps a kernel queue of its own,
 * whatever the render credits, as do the first HK_ADVERTISED_QUEUES queues of
 * three families; the rest spread evenly over those. */
static void
test_advertised(void)
{
   set_kernel(IQ, false, 58, 1);
   for (unsigned i = 0; i < HK_ADVERTISED_QUEUES; i++)
      add(MEDIUM, 0);
   CHECK(independent() == HK_ADVERTISED_QUEUES);
   for (unsigned i = 0; i < pool_count; i++)
      CHECK(!pool[i].vk.real_queue);
   finish_all();

   set_kernel(IQ, false, 58, 2);
   for (unsigned i = 0; i < 3 * HK_ADVERTISED_QUEUES; i++)
      add(MEDIUM, 0);
   CHECK(independent() == HK_ADVERTISED_QUEUES);
   unsigned sharers[HK_ADVERTISED_QUEUES] = {0};
   for (unsigned i = HK_ADVERTISED_QUEUES; i < pool_count; i++) {
      CHECK(backing(i) < HK_ADVERTISED_QUEUES);
      sharers[backing(i)]++;
   }
   for (unsigned i = 0; i < HK_ADVERTISED_QUEUES; i++)
      CHECK(sharers[i] == 2);
   finish_all();
}

/* Without independent kernel queues (or under AGX_DBG_1QUEUE), every queue of
 * a class shares one. */
static void
test_single(void)
{
   set_kernel(0, false, 0, 0);
   const unsigned m0 = add(MEDIUM, 0);
   const unsigned m1 = add(MEDIUM, 0);
   const unsigned l0 = add(LOW, 0);
   const unsigned m2 = add(MEDIUM, 0);
   CHECK(backing(m0) == m0);
   CHECK(backing(m1) == m0 && backing(m2) == m0);
   CHECK(backing(l0) == l0);
   finish_all();

   set_kernel(IQ, false, 58, 2);
   dev->dev.debug = AGX_DBG_1QUEUE;
   const unsigned d0 = add(MEDIUM, 0);
   const unsigned d1 = add(MEDIUM, 0);
   const unsigned d2 = add(LOW, 0);
   const unsigned d3 = add(HIGH, PROTECTED);
   CHECK(backing(d0) == d0 && backing(d1) == d0 &&
         backing(d2) == d0 && backing(d3) == d0);
   CHECK(independent() == 1);
   hk_compute_kick_init(dev, 10);
   CHECK(dev->compute_kick == NULL);
   CHECK(!pool[d0].kick.watched);
   finish_all();
}

static struct hk_queue queues[5];
static int entered, released, finished, other, active;

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
      bump(&other);
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

/* Submissions of queues that share a kernel queue run one at a time on the
 * backing queue's callback; another class's queue is not held up. */
static void
test_serialization(void)
{
   CHECK(mtx_init(&lock, mtx_plain) == thrd_success);
   CHECK(cnd_init(&cond) == thrd_success);
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED;
   set_kernel(IQ, false, 1 + HK_QUEUE_SIBLINGS_MAX, 1);

   for (unsigned i = 0; i < ARRAY_SIZE(queues); i++) {
      create(&queues[i], i % 3, i == 2 ? LOW : MEDIUM, i == 3 ? PROTECTED : 0,
             driver_submit);
   }
   CHECK(queues[0].vk.real_queue == &queues[0].vk);
   CHECK(queues[1].vk.real_queue == &queues[0].vk);
   CHECK(!queues[2].vk.real_queue);
   CHECK(!queues[3].vk.real_queue);
   CHECK(queues[4].vk.real_queue == &queues[0].vk);
   CHECK(!queues[1].vk.driver_submit && !queues[4].vk.driver_submit);

   submit(0);
   await(&entered);
   submit(1);
   submit(2);
   await(&other);
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

   finish_all();
   CHECK(p_atomic_read(&entered) == 2);
   CHECK(p_atomic_read(&finished) == 2);
   cnd_destroy(&cond);
   mtx_destroy(&lock);
}

int
main(void)
{
   dev = calloc(1, sizeof(*dev));
   pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_IMMEDIATE;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->vk.queues);

   for (unsigned i = 0; i < ARRAY_SIZE(chips); i++) {
      dev->dev.chip = chips[i];
      test_budget();
      test_spread();
      test_isolation();
      test_advertised();
      test_single();
   }
   test_serialization();

   vk_object_base_finish(&dev->vk.base);
   free(pdev);
   free(dev);
   puts("queue budget, spreading, class isolation, serialization: PASS");
   return 0;
}
