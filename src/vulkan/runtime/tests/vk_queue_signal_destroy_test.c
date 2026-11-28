/* SPDX-License-Identifier: MIT */

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_fence.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_semaphore.h"
#include "vk_sync.h"
#include "util/os_time.h"
#include "util/u_atomic.h"

#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

struct mock_sync {
   struct vk_sync base;
   uint64_t pending_point;
};

struct guard_header {
   void *base;
   size_t total;
   size_t req_size;
};

static int driver_gate_enabled;
static int driver_in_gate;
static int driver_gate_open;

static int signal_gate_enabled;
static int signal_in_gate;
static int signal_gate_open;

static int fail_next_submit_lost;

static void
require(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "%s\n", message);
      exit(EXIT_FAILURE);
   }
}

static size_t
align_up_size(size_t val, size_t align)
{
   return (val + align - 1) & ~(align - 1);
}

static void *VKAPI_CALL
guard_alloc(void *user_data, size_t size, size_t alignment,
            VkSystemAllocationScope scope)
{
#ifndef _WIN32
   size_t page = (size_t)sysconf(_SC_PAGESIZE);
   size_t align = alignment < alignof(struct guard_header) ?
                  alignof(struct guard_header) : alignment;
   size_t offset = align_up_size(sizeof(struct guard_header), align);
   size_t total = align_up_size(offset + (size ? size : 1), page);

   void *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (base == MAP_FAILED)
      return NULL;

   void *ptr = (char *)base + offset;
   struct guard_header *hdr = (struct guard_header *)ptr - 1;
   hdr->base = base;
   hdr->total = total;
   hdr->req_size = size;
   return ptr;
#else
   return vk_default_allocator()->pfnAllocation(user_data, size, alignment, scope);
#endif
}

static void VKAPI_CALL
guard_free(void *user_data, void *mem)
{
   if (mem == NULL)
      return;
#ifndef _WIN32
   struct guard_header *hdr = (struct guard_header *)mem - 1;
   void *base = hdr->base;
   size_t total = hdr->total;
   require(munmap(base, total) == 0, "munmap failed");
#else
   vk_default_allocator()->pfnFree(user_data, mem);
#endif
}

static void *VKAPI_CALL
guard_realloc(void *user_data, void *orig, size_t size, size_t alignment,
              VkSystemAllocationScope scope)
{
#ifndef _WIN32
   if (orig == NULL)
      return guard_alloc(user_data, size, alignment, scope);
   if (size == 0) {
      guard_free(user_data, orig);
      return NULL;
   }
   struct guard_header *hdr = (struct guard_header *)orig - 1;
   size_t old_size = hdr->req_size;
   void *next = guard_alloc(user_data, size, alignment, scope);
   if (next == NULL)
      return NULL;
   memcpy(next, orig, old_size < size ? old_size : size);
   guard_free(user_data, orig);
   return next;
#else
   return vk_default_allocator()->pfnReallocation(user_data, orig, size,
                                                  alignment, scope);
#endif
}

static void
wait_flag(int *flag, const char *msg)
{
   const int64_t deadline = os_time_get_nano() + 5 * ONE_SECOND_IN_NS;
   while (!p_atomic_read(flag)) {
      require(os_time_get_nano() < deadline, msg);
      const struct timespec delay = { .tv_nsec = 1000000 };
      thrd_sleep(&delay, NULL);
   }
}

static void
sleep_ms(unsigned ms)
{
   const struct timespec delay = {
      .tv_sec = ms / 1000,
      .tv_nsec = (long)(ms % 1000) * 1000000L,
   };
   thrd_sleep(&delay, NULL);
}

static VkResult
mock_init(struct vk_device *device, struct vk_sync *sync, uint64_t initial_value)
{
   struct mock_sync *ms = (struct mock_sync *)sync;
   if (sync->flags & VK_SYNC_IS_TIMELINE)
      p_atomic_set(&ms->pending_point, initial_value);
   else
      p_atomic_set(&ms->pending_point, initial_value ? 1 : 0);
   return VK_SUCCESS;
}

static void
mock_finish(struct vk_device *device, struct vk_sync *sync)
{
}

static VkResult
mock_signal(struct vk_device *device, struct vk_sync *sync, uint64_t value)
{
   struct mock_sync *ms = (struct mock_sync *)sync;
   if (sync->flags & VK_SYNC_IS_TIMELINE) {
      uint64_t cur = p_atomic_read(&ms->pending_point);
      while (value > cur) {
         uint64_t prev = p_atomic_cmpxchg(&ms->pending_point, cur, value);
         if (prev == cur)
            break;
         cur = prev;
      }
   } else {
      p_atomic_set(&ms->pending_point, 1);
   }

   if (p_atomic_read(&signal_gate_enabled)) {
      p_atomic_set(&signal_in_gate, 1);
      wait_flag(&signal_gate_open, "signal gate was never opened");
      p_atomic_set(&signal_in_gate, 0);
   }

   return VK_SUCCESS;
}

static VkResult
mock_get_value(struct vk_device *device, struct vk_sync *sync, uint64_t *value)
{
   struct mock_sync *ms = (struct mock_sync *)sync;
   *value = p_atomic_read(&ms->pending_point);
   return VK_SUCCESS;
}

static VkResult
mock_reset(struct vk_device *device, struct vk_sync *sync)
{
   struct mock_sync *ms = (struct mock_sync *)sync;
   p_atomic_set(&ms->pending_point, 0);
   return VK_SUCCESS;
}

static VkResult
mock_move(struct vk_device *device, struct vk_sync *dst, struct vk_sync *src)
{
   struct mock_sync *mdst = (struct mock_sync *)dst;
   struct mock_sync *msrc = (struct mock_sync *)src;
   uint64_t val = p_atomic_read(&msrc->pending_point);
   p_atomic_set(&mdst->pending_point, val);
   p_atomic_set(&msrc->pending_point, 0);
   return VK_SUCCESS;
}

static bool
mock_wait_satisfied(const struct vk_sync_wait *w)
{
   struct mock_sync *ms = (struct mock_sync *)w->sync;
   uint64_t cur = p_atomic_read(&ms->pending_point);
   if (w->sync->flags & VK_SYNC_IS_TIMELINE)
      return w->wait_value == 0 || cur >= w->wait_value;
   return cur != 0;
}

static VkResult
mock_wait_many(struct vk_device *device, uint32_t wait_count,
               const struct vk_sync_wait *waits,
               enum vk_sync_wait_flags wait_flags,
               uint64_t abs_timeout_ns)
{
   for (;;) {
      bool all_ready = true;
      for (uint32_t i = 0; i < wait_count; i++) {
         if (!mock_wait_satisfied(&waits[i])) {
            all_ready = false;
            break;
         }
      }
      if (all_ready)
         return VK_SUCCESS;

      if (abs_timeout_ns == 0 || os_time_get_nano() >= abs_timeout_ns)
         return VK_TIMEOUT;

      sleep_ms(1);
   }
}

static const struct vk_sync_type mock_sync_type = {
   .size = sizeof(struct mock_sync),
   .features = VK_SYNC_FEATURE_BINARY |
               VK_SYNC_FEATURE_TIMELINE |
               VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_GPU_MULTI_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_ANY |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = mock_init,
   .finish = mock_finish,
   .signal = mock_signal,
   .get_value = mock_get_value,
   .reset = mock_reset,
   .move = mock_move,
   .wait_many = mock_wait_many,
};

static VkResult
mock_driver_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   if (p_atomic_read(&fail_next_submit_lost)) {
      p_atomic_set(&fail_next_submit_lost, 0);
      return VK_ERROR_DEVICE_LOST;
   }

   for (uint32_t i = 0; i < submit->signal_count; i++) {
      mock_signal(queue->base.device, submit->signals[i].sync,
                  submit->signals[i].signal_value);
   }

   if (p_atomic_read(&driver_gate_enabled) && submit->signal_count > 0) {
      p_atomic_set(&driver_in_gate, 1);
      wait_flag(&driver_gate_open, "driver gate was never opened");
      p_atomic_set(&driver_in_gate, 0);
   }

   return VK_SUCCESS;
}

static void
driver_gate_arm(void)
{
   p_atomic_set(&driver_in_gate, 0);
   p_atomic_set(&driver_gate_open, 0);
   p_atomic_set(&driver_gate_enabled, 1);
}

static void
driver_gate_release_and_disarm(void)
{
   p_atomic_set(&driver_gate_enabled, 0);
   p_atomic_set(&driver_gate_open, 1);
}

static void
signal_gate_arm(void)
{
   p_atomic_set(&signal_in_gate, 0);
   p_atomic_set(&signal_gate_open, 0);
   p_atomic_set(&signal_gate_enabled, 1);
}

static void
signal_gate_release_and_disarm(void)
{
   p_atomic_set(&signal_gate_enabled, 0);
   p_atomic_set(&signal_gate_open, 1);
}

struct wait_idle_args {
   struct vk_queue *queue;
   VkResult result;
   int done;
};

static int
wait_idle_thread(void *data)
{
   struct wait_idle_args *args = data;
   args->result = vk_common_QueueWaitIdle(vk_queue_to_handle(args->queue));
   p_atomic_set(&args->done, 1);
   return 0;
}

struct fence_destroy_args {
   struct vk_device *device;
   VkFence fence;
   VkResult wait_result;
   int done;
};

static int
fence_destroy_thread(void *data)
{
   struct fence_destroy_args *args = data;
   args->wait_result = vk_common_WaitForFences(
      vk_device_to_handle(args->device), 1, &args->fence, VK_TRUE, UINT64_MAX);
   vk_common_DestroyFence(vk_device_to_handle(args->device), args->fence, NULL);
   p_atomic_set(&args->done, 1);
   return 0;
}

struct tl_destroy_args {
   struct vk_device *device;
   VkSemaphore sem;
   uint64_t value;
   VkResult wait_result;
   int done;
};

static int
tl_destroy_thread(void *data)
{
   struct tl_destroy_args *args = data;
   const VkSemaphoreWaitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &args->sem,
      .pValues = &args->value,
   };
   args->wait_result = vk_common_WaitSemaphores(
      vk_device_to_handle(args->device), &info, UINT64_MAX);
   vk_common_DestroySemaphore(vk_device_to_handle(args->device), args->sem, NULL);
   p_atomic_set(&args->done, 1);
   return 0;
}

struct bin_destroy_args {
   struct vk_device *device;
   VkSemaphore sem;
   VkResult wait_result;
   int done;
};

static int
bin_destroy_thread(void *data)
{
   struct bin_destroy_args *args = data;
   VK_FROM_HANDLE(vk_semaphore, sem, args->sem);
   args->wait_result = vk_sync_wait(args->device, &sem->permanent, 0,
                                    VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
   vk_common_DestroySemaphore(vk_device_to_handle(args->device), args->sem, NULL);
   p_atomic_set(&args->done, 1);
   return 0;
}

static void
init_device_and_threaded_queue(struct vk_physical_device *pdev,
                               struct vk_device *device,
                               struct vk_queue *queue)
{
   static const struct vk_sync_type *const sync_types[] = {
      &mock_sync_type,
      NULL,
   };
   *pdev = (struct vk_physical_device) {
      .supported_sync_types = sync_types,
   };
   *device = (struct vk_device) {
      .physical = pdev,
      .alloc = {
         .pfnAllocation = guard_alloc,
         .pfnReallocation = guard_realloc,
         .pfnFree = guard_free,
      },
      .submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED,
      .timeline_mode = VK_DEVICE_TIMELINE_MODE_ASSISTED,
   };
   vk_object_base_init(NULL, &device->base, VK_OBJECT_TYPE_DEVICE);
   device->_lost.reported = true;
   list_inithead(&device->queues);

   const VkDeviceQueueCreateInfo qinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
   };
   require(vk_queue_init(queue, device, &qinfo, 0) == VK_SUCCESS,
           "vk_queue_init failed");
   queue->driver_submit = mock_driver_submit;
}

static void
test_queue_wait_idle_destroy_race(struct vk_queue *q)
{
   driver_gate_arm();

   struct wait_idle_args args = { .queue = q };
   thrd_t thr;
   require(thrd_create(&thr, wait_idle_thread, &args) == thrd_success,
           "thrd_create failed");

   wait_flag(&driver_in_gate, "QueueWaitIdle submit did not enter gate");
   /* Give the waiter time to wake from vk_sync_wait and reach vk_sync_finish.
    * Without synchronization, it frees & unmaps the internal sync while the
    * submit thread is still inside driver_submit, causing a deterministic
    * SIGSEGV in vk_sync_note_pending() once the gate opens.
    */
   sleep_ms(40);
   const bool freed_early = p_atomic_read(&args.done) != 0;

   driver_gate_release_and_disarm();
   require(thrd_join(thr, NULL) == thrd_success, "thrd_join failed");
   require(!freed_early,
           "QueueWaitIdle freed its sync before vk_queue_submit_final finished");
   require(args.result == VK_SUCCESS, "QueueWaitIdle failed");
   printf("QueueWaitIdle destroy race: PASS\n");
}

static void
test_fence_destroy_race(struct vk_device *device, struct vk_queue *q)
{
   const VkFenceCreateInfo finfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence = VK_NULL_HANDLE;
   require(vk_common_CreateFence(vk_device_to_handle(device), &finfo, NULL,
                                 &fence) == VK_SUCCESS,
           "CreateFence failed");

   driver_gate_arm();

   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
   };
   require(vk_common_QueueSubmit2(vk_queue_to_handle(q), 1, &submit, fence) ==
              VK_SUCCESS,
           "QueueSubmit2(fence) failed");

   struct fence_destroy_args args = {
      .device = device,
      .fence = fence,
   };
   thrd_t thr;
   require(thrd_create(&thr, fence_destroy_thread, &args) == thrd_success,
           "thrd_create failed");

   wait_flag(&driver_in_gate, "fence submit did not enter gate");
   sleep_ms(40);
   const bool freed_early = p_atomic_read(&args.done) != 0;

   driver_gate_release_and_disarm();
   require(thrd_join(thr, NULL) == thrd_success, "thrd_join failed");
   require(!freed_early,
           "DestroyFence freed fence before vk_queue_submit_final finished");
   require(args.wait_result == VK_SUCCESS, "WaitForFences failed");
   printf("VkFence destroy race: PASS\n");
}

static void
test_timeline_semaphore_destroy_race(struct vk_device *device,
                                     struct vk_queue *q)
{
   const VkSemaphoreTypeCreateInfo tinfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
   };
   const VkSemaphoreCreateInfo sinfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &tinfo,
   };
   VkSemaphore sem = VK_NULL_HANDLE;
   require(vk_common_CreateSemaphore(vk_device_to_handle(device), &sinfo, NULL,
                                     &sem) == VK_SUCCESS,
           "CreateSemaphore(timeline) failed");

   driver_gate_arm();

   const VkSemaphoreSubmitInfo sig = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = sem,
      .value = 1,
   };
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig,
   };
   require(vk_common_QueueSubmit2(vk_queue_to_handle(q), 1, &submit,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "QueueSubmit2(timeline) failed");

   struct tl_destroy_args args = {
      .device = device,
      .sem = sem,
      .value = 1,
   };
   thrd_t thr;
   require(thrd_create(&thr, tl_destroy_thread, &args) == thrd_success,
           "thrd_create failed");

   wait_flag(&driver_in_gate, "timeline submit did not enter gate");
   sleep_ms(40);
   const bool freed_early = p_atomic_read(&args.done) != 0;

   driver_gate_release_and_disarm();
   require(thrd_join(thr, NULL) == thrd_success, "thrd_join failed");
   require(!freed_early,
           "DestroySemaphore(timeline) freed semaphore before submit finished");
   require(args.wait_result == VK_SUCCESS, "WaitSemaphores failed");
   printf("Timeline VkSemaphore destroy race: PASS\n");
}

static void
test_binary_semaphore_destroy_race(struct vk_device *device, struct vk_queue *q)
{
   const VkSemaphoreCreateInfo sinfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
   };
   VkSemaphore sem = VK_NULL_HANDLE;
   require(vk_common_CreateSemaphore(vk_device_to_handle(device), &sinfo, NULL,
                                     &sem) == VK_SUCCESS,
           "CreateSemaphore(binary) failed");

   driver_gate_arm();

   const VkSemaphoreSubmitInfo sig = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = sem,
   };
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig,
   };
   require(vk_common_QueueSubmit2(vk_queue_to_handle(q), 1, &submit,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "QueueSubmit2(binary) failed");

   struct bin_destroy_args args = {
      .device = device,
      .sem = sem,
   };
   thrd_t thr;
   require(thrd_create(&thr, bin_destroy_thread, &args) == thrd_success,
           "thrd_create failed");

   wait_flag(&driver_in_gate, "binary submit did not enter gate");
   sleep_ms(40);
   const bool freed_early = p_atomic_read(&args.done) != 0;

   driver_gate_release_and_disarm();
   require(thrd_join(thr, NULL) == thrd_success, "thrd_join failed");
   require(!freed_early,
           "DestroySemaphore(binary) freed semaphore before submit finished");
   require(args.wait_result == VK_SUCCESS, "binary sync wait failed");
   printf("Binary VkSemaphore destroy race: PASS\n");
}

static void
test_dropped_signal_destroy_race(void)
{
   struct vk_physical_device pdev;
   struct vk_device device;
   struct vk_queue q;
   init_device_and_threaded_queue(&pdev, &device, &q);

   p_atomic_set(&fail_next_submit_lost, 1);
   signal_gate_arm();

   struct wait_idle_args args = { .queue = &q };
   thrd_t thr;
   require(thrd_create(&thr, wait_idle_thread, &args) == thrd_success,
           "thrd_create failed");

   wait_flag(&signal_in_gate, "dropped signal did not enter signal gate");
   sleep_ms(40);
   const bool freed_early = p_atomic_read(&args.done) != 0;

   signal_gate_release_and_disarm();
   require(thrd_join(thr, NULL) == thrd_success, "thrd_join failed");
   require(!freed_early,
           "QueueWaitIdle freed sync while dropped vk_sync_signal was active");
   require(args.result == VK_ERROR_DEVICE_LOST,
           "QueueWaitIdle did not observe device loss");

   vk_queue_finish(&q);
   vk_object_base_finish(&device.base);
   printf("Dropped signal destroy race: PASS\n");
}

int
main(void)
{
   struct vk_physical_device pdev;
   struct vk_device device;
   struct vk_queue q;
   init_device_and_threaded_queue(&pdev, &device, &q);

   test_queue_wait_idle_destroy_race(&q);
   test_fence_destroy_race(&device, &q);
   test_timeline_semaphore_destroy_race(&device, &q);
   test_binary_semaphore_destroy_race(&device, &q);

   /* Verify queue still drains cleanly afterwards. */
   require(vk_common_QueueWaitIdle(vk_queue_to_handle(&q)) == VK_SUCCESS,
           "final QueueWaitIdle failed");

   vk_queue_finish(&q);
   vk_object_base_finish(&device.base);

   test_dropped_signal_destroy_race();
   return EXIT_SUCCESS;
}
