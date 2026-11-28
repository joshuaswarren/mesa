/* SPDX-License-Identifier: MIT */

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_semaphore.h"
#include "vk_sync.h"
#include "util/os_time.h"
#include "util/u_atomic.h"

#include <stdio.h>
#include <stdlib.h>

struct mock_sync {
   struct vk_sync base;
   uint64_t pending_point;
};

static int pending_wait_calls;
static int pending_wait_items;
static int reset_calls;
static int driver_submit_calls;
static int alloc_calls;
static bool fail_next_submit_oom;
static int track_allocs;

/* Gate for holding a submit inside driver_submit, the way a real submit
 * ioctl returns some time after the kernel made its fence visible to other
 * waiters.
 */
enum gate_mode {
   GATE_NONE,
   GATE_AFTER_SIGNAL,
   GATE_BEFORE_SIGNAL,
};
static int gate_mode;
static int in_gate;
static int gate_open;

static void
require(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "%s\n", message);
      exit(EXIT_FAILURE);
   }
}

static void *VKAPI_CALL
counting_alloc(void *user_data, size_t size, size_t alignment,
               VkSystemAllocationScope scope)
{
   if (p_atomic_read(&track_allocs))
      p_atomic_inc(&alloc_calls);
   return vk_default_allocator()->pfnAllocation(user_data, size, alignment, scope);
}

static void *VKAPI_CALL
counting_realloc(void *user_data, void *orig, size_t size, size_t alignment,
                 VkSystemAllocationScope scope)
{
   return vk_default_allocator()->pfnReallocation(user_data, orig, size,
                                                  alignment, scope);
}

static void VKAPI_CALL
counting_free(void *user_data, void *mem)
{
   vk_default_allocator()->pfnFree(user_data, mem);
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
   p_atomic_inc(&reset_calls);
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
   if (wait_flags & VK_SYNC_WAIT_PENDING) {
      p_atomic_inc(&pending_wait_calls);
      p_atomic_add(&pending_wait_items, (int)wait_count);
   }

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

      const struct timespec delay = { .tv_nsec = 1000000 };
      thrd_sleep(&delay, NULL);
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

static void
gate_hold(void)
{
   p_atomic_set(&in_gate, 1);
   while (!p_atomic_read(&gate_open)) {
      const struct timespec delay = { .tv_nsec = 1000000 };
      thrd_sleep(&delay, NULL);
   }
   p_atomic_set(&in_gate, 0);
}

static void
gate_close(enum gate_mode mode)
{
   p_atomic_set(&gate_open, 0);
   p_atomic_set(&gate_mode, mode);
}

static void
gate_release(void)
{
   p_atomic_set(&gate_open, 1);
}

static void
wait_in_gate(void)
{
   const int64_t deadline = os_time_get_nano() + 2 * ONE_SECOND_IN_NS;
   while (!p_atomic_read(&in_gate)) {
      require(os_time_get_nano() < deadline, "submit never reached the gate");
      const struct timespec delay = { .tv_nsec = 1000000 };
      thrd_sleep(&delay, NULL);
   }
}

static int
release_gate_later(void *data)
{
   const struct timespec delay = { .tv_nsec = 100000000 };
   thrd_sleep(&delay, NULL);
   gate_release();
   return 0;
}

static VkResult
mock_driver_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   if (fail_next_submit_oom) {
      fail_next_submit_oom = false;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   const int mode = p_atomic_read(&gate_mode);
   if (mode == GATE_BEFORE_SIGNAL && submit->signal_count)
      gate_hold();

   for (uint32_t i = 0; i < submit->signal_count; i++)
      mock_signal(queue->base.device, submit->signals[i].sync,
                  submit->signals[i].signal_value);

   if (mode == GATE_AFTER_SIGNAL && submit->signal_count)
      gate_hold();

   p_atomic_inc(&driver_submit_calls);
   return VK_SUCCESS;
}

static struct vk_semaphore *
create_sem(struct vk_device *device, VkSemaphoreType type,
           enum vk_sync_flags extra_flags, uint64_t initial_value)
{
   size_t size = offsetof(struct vk_semaphore, permanent) + mock_sync_type.size;
   struct vk_semaphore *sem = calloc(1, size);
   require(sem != NULL, "semaphore alloc failed");
   vk_object_base_init(device, &sem->base, VK_OBJECT_TYPE_SEMAPHORE);
   sem->type = type;

   enum vk_sync_flags flags = extra_flags;
   if (type == VK_SEMAPHORE_TYPE_TIMELINE)
      flags |= VK_SYNC_IS_TIMELINE;

   require(vk_sync_init(device, &sem->permanent, &mock_sync_type, flags,
                        initial_value) == VK_SUCCESS,
           "vk_sync_init failed");
   return sem;
}

static void
destroy_sem(struct vk_device *device, struct vk_semaphore *sem)
{
   vk_sync_finish(device, &sem->permanent);
   vk_object_base_finish(&sem->base);
   free(sem);
}

static void
wait_for_submits(int target, const char *msg)
{
   const int64_t deadline = os_time_get_nano() + 2 * ONE_SECOND_IN_NS;
   while (p_atomic_read(&driver_submit_calls) < target) {
      require(os_time_get_nano() < deadline, msg);
      const struct timespec delay = { .tv_nsec = 1000000 };
      thrd_sleep(&delay, NULL);
   }
}

static VkResult
submit_binary(struct vk_queue *q, struct vk_semaphore *wait_sem,
              struct vk_semaphore *signal_sem)
{
   const VkSemaphoreSubmitInfo w = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = wait_sem ? vk_semaphore_to_handle(wait_sem) : VK_NULL_HANDLE,
   };
   const VkSemaphoreSubmitInfo s = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = signal_sem ? vk_semaphore_to_handle(signal_sem) : VK_NULL_HANDLE,
   };
   const VkSubmitInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = wait_sem ? 1 : 0,
      .pWaitSemaphoreInfos = &w,
      .signalSemaphoreInfoCount = signal_sem ? 1 : 0,
      .pSignalSemaphoreInfos = &s,
   };
   return vk_common_QueueSubmit2(vk_queue_to_handle(q), 1, &info,
                                 VK_NULL_HANDLE);
}

struct immediate_signal {
   struct vk_queue *queue;
   struct vk_semaphore *sem;
   VkResult result;
};

static int
immediate_signal_thread(void *data)
{
   struct immediate_signal *args = data;
   args->result = submit_binary(args->queue, NULL, args->sem);
   return 0;
}

static void
queue_wait_idle(struct vk_queue *q)
{
   require(vk_common_QueueWaitIdle(vk_queue_to_handle(q)) == VK_SUCCESS,
           "vkQueueWaitIdle failed");
}

/* A sync must never be known pending while its payload is empty. */
static void
require_pending_has_payload(struct vk_semaphore *sem, const char *msg)
{
   struct mock_sync *ms = (struct mock_sync *)&sem->permanent;
   require(!vk_sync_wait_is_known_pending(&sem->permanent, 0) ||
           p_atomic_read(&ms->pending_point) != 0, msg);
}

int
main(void)
{
   static const struct vk_sync_type *const sync_types[] = {
      &mock_sync_type,
      NULL,
   };
   struct vk_physical_device pdev = {
      .supported_sync_types = sync_types,
   };
   struct vk_device device = {
      .physical = &pdev,
      .alloc = {
         .pfnAllocation = counting_alloc,
         .pfnReallocation = counting_realloc,
         .pfnFree = counting_free,
      },
      .submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND,
      .timeline_mode = VK_DEVICE_TIMELINE_MODE_ASSISTED,
   };
   vk_object_base_init(NULL, &device.base, VK_OBJECT_TYPE_DEVICE);
   device._lost.reported = true;
   list_inithead(&device.queues);

   const VkDeviceQueueCreateInfo qinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 2,
   };
   struct vk_queue q0, q1;
   require(vk_queue_init(&q0, &device, &qinfo, 0) == VK_SUCCESS, "q0 init");
   require(vk_queue_init(&q1, &device, &qinfo, 1) == VK_SUCCESS, "q1 init");
   q0.driver_submit = mock_driver_submit;
   q1.driver_submit = mock_driver_submit;

   /* 1. Private timeline ping-pong across queues skips WAIT_PENDING and
    * stays on the zero-allocation stack fast path. */
   struct vk_semaphore *tl =
      create_sem(&device, VK_SEMAPHORE_TYPE_TIMELINE, 0, 0);
   VkSemaphoreSubmitInfo sig1 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 1,
   };
   VkSubmitInfo2 s0 = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig1,
   };
   p_atomic_set(&pending_wait_calls, 0);
   p_atomic_set(&alloc_calls, 0);
   p_atomic_set(&track_allocs, true);
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q0), 1, &s0,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "q0 timeline signal failed");

   VkSemaphoreSubmitInfo wait1 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 1,
   };
   VkSemaphoreSubmitInfo sig2 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 2,
   };
   VkSubmitInfo2 s1 = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &wait1,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig2,
   };
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q1), 1, &s1,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "q1 timeline wait+signal failed");
   p_atomic_set(&track_allocs, false);

   require(p_atomic_read(&pending_wait_calls) == 0,
           "known-pending private timeline invoked WAIT_PENDING");
   require(p_atomic_read(&alloc_calls) == 0,
           "known-pending private timeline allocated on heap");
   require(q1.submit.mode == VK_QUEUE_SUBMIT_MODE_IMMEDIATE,
           "known-pending timeline started submit thread");
   printf("private timeline fast path: PASS\n");

   /* 2. Timeline wait_value == 0 on stack fast path (exercises compaction). */
   VkSemaphoreSubmitInfo wait0 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 0,
   };
   VkSemaphoreSubmitInfo waits_zero[2] = { wait0, wait1 };
   VkSubmitInfo2 s_zero = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 2,
      .pWaitSemaphoreInfos = waits_zero,
   };
   p_atomic_set(&alloc_calls, 0);
   p_atomic_set(&track_allocs, true);
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q1), 1, &s_zero,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "zero-value timeline wait failed");
   p_atomic_set(&track_allocs, false);
   require(p_atomic_read(&pending_wait_calls) == 0,
           "zero-value timeline wait invoked WAIT_PENDING");
   require(p_atomic_read(&alloc_calls) == 0,
           "zero-value timeline wait allocated on heap");
   printf("zero-value timeline compaction: PASS\n");

   /* 3. Non-zero initial_value and host vkSignalSemaphore advance pending. */
   struct vk_semaphore *tl_init =
      create_sem(&device, VK_SEMAPHORE_TYPE_TIMELINE, 0, 5);
   require(vk_sync_wait_is_known_pending(&tl_init->permanent, 5),
           "initial_value was not noted as pending");
   require(!vk_sync_wait_is_known_pending(&tl_init->permanent, 6),
           "value past initial_value was falsely known pending");
   VkSemaphoreSignalInfo host_sig = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = vk_semaphore_to_handle(tl_init),
      .value = 10,
   };
   require(vk_common_SignalSemaphore(vk_device_to_handle(&device), &host_sig) ==
              VK_SUCCESS,
           "host SignalSemaphore failed");
   require(vk_sync_wait_is_known_pending(&tl_init->permanent, 10),
           "host signal was not noted as pending");
   printf("initial value and host signal: PASS\n");

   /* 4. >4 waits (heap path) skips WAIT_PENDING when all are known pending,
    * and only passes unknown waits when mixed. */
   VkSemaphoreSubmitInfo five_waits[5];
   for (unsigned i = 0; i < 5; i++) {
      five_waits[i] = (VkSemaphoreSubmitInfo) {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
         .semaphore = vk_semaphore_to_handle(tl_init),
         .value = 10,
      };
   }
   VkSubmitInfo2 s_five = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 5,
      .pWaitSemaphoreInfos = five_waits,
   };
   p_atomic_set(&pending_wait_calls, 0);
   p_atomic_set(&pending_wait_items, 0);
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q0), 1, &s_five,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "5 known waits failed");
   require(p_atomic_read(&pending_wait_calls) == 0,
           "5 known waits invoked WAIT_PENDING");

   /* Mixed: 4 known + 1 shareable (unknown) -> only 1 wait passed to backend. */
   struct vk_semaphore *tl_share =
      create_sem(&device, VK_SEMAPHORE_TYPE_TIMELINE, VK_SYNC_IS_SHAREABLE, 1);
   five_waits[3].semaphore = vk_semaphore_to_handle(tl_share);
   five_waits[3].value = 1;
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q0), 1, &s_five,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "mixed 5 waits failed");
   require(p_atomic_read(&pending_wait_calls) == 1 &&
           p_atomic_read(&pending_wait_items) == 1,
           "mixed waits did not filter known-pending entries");
   printf("filtered multi-wait and shareable bypass: PASS\n");

   /* 5. Failed OOM submit must NOT advance pending_value. */
   VkSemaphoreSubmitInfo sig30 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 30,
   };
   VkSubmitInfo2 s_oom = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig30,
   };
   fail_next_submit_oom = true;
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q0), 1, &s_oom,
                                  VK_NULL_HANDLE) == VK_ERROR_OUT_OF_HOST_MEMORY,
           "injected OOM was not returned");
   require(!vk_sync_wait_is_known_pending(&tl->permanent, 30),
           "failed OOM submit advanced pending_value");
   printf("OOM submit leaves pending untouched: PASS\n");

   /* 6. Private binary semaphore: signal -> wait skips WAIT_PENDING, resets
    * after submit, and a wait+signal on the same semaphore keeps it pending. */
   struct vk_semaphore *bin =
      create_sem(&device, VK_SEMAPHORE_TYPE_BINARY, 0, 0);
   VkSemaphoreSubmitInfo bin_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(bin),
   };
   VkSubmitInfo2 s_bin_sig = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &bin_info,
   };
   VkSubmitInfo2 s_bin_wait_sig = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &bin_info,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &bin_info,
   };
   VkSubmitInfo2 s_bin_wait = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &bin_info,
   };
   p_atomic_set(&pending_wait_calls, 0);
   p_atomic_set(&reset_calls, 0);
   p_atomic_set(&alloc_calls, 0);
   p_atomic_set(&track_allocs, true);
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q0), 1, &s_bin_sig,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "binary signal failed");
   require(vk_sync_wait_is_known_pending(&bin->permanent, 0),
           "binary signal did not mark pending");
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q1), 1, &s_bin_wait_sig,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "binary wait+signal failed");
   require(p_atomic_read(&reset_calls) == 0,
           "binary wait+signal unexpectedly reset semaphore");
   require(vk_sync_wait_is_known_pending(&bin->permanent, 0),
           "binary wait+signal lost pending state");
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q1), 1, &s_bin_wait,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "binary wait failed");
   p_atomic_set(&track_allocs, false);

   require(p_atomic_read(&pending_wait_calls) == 0,
           "known-pending private binary invoked WAIT_PENDING");
   require(p_atomic_read(&alloc_calls) == 0,
           "known-pending private binary allocated on heap");
   require(p_atomic_read(&reset_calls) == 1,
           "binary wait did not reset permanent semaphore");
   require(!vk_sync_wait_is_known_pending(&bin->permanent, 0),
           "binary wait left semaphore marked pending after reset");
   printf("private binary fast path and reset: PASS\n");

   /* 7. Wait-before-signal on private timeline still invokes WAIT_PENDING,
    * starts the submit thread, and does not advance queued signals until
    * driver_submit actually runs. */
   struct vk_semaphore *tl_out =
      create_sem(&device, VK_SEMAPHORE_TYPE_TIMELINE, 0, 0);
   struct vk_semaphore *bin_out =
      create_sem(&device, VK_SEMAPHORE_TYPE_BINARY, 0, 0);
   VkSemaphoreSubmitInfo wait20 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 20,
   };
   VkSemaphoreSubmitInfo queued_sigs[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
         .semaphore = vk_semaphore_to_handle(tl_out),
         .value = 1,
      },
      {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
         .semaphore = vk_semaphore_to_handle(bin_out),
      },
   };
   VkSubmitInfo2 s_wbs = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &wait20,
      .signalSemaphoreInfoCount = 2,
      .pSignalSemaphoreInfos = queued_sigs,
   };
   const int submits_before = p_atomic_read(&driver_submit_calls);
   p_atomic_set(&pending_wait_calls, 0);
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q1), 1, &s_wbs,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "wait-before-signal submit failed");
   require(p_atomic_read(&pending_wait_calls) >= 1,
           "wait-before-signal did not invoke WAIT_PENDING");
   require(q1.submit.mode == VK_QUEUE_SUBMIT_MODE_THREADED,
           "wait-before-signal did not enable submit thread");
   require(p_atomic_read(&driver_submit_calls) == submits_before,
           "wait-before-signal ran driver_submit prematurely");
   require(!vk_sync_wait_is_known_pending(&tl_out->permanent, 1),
           "queued timeline signal marked pending before driver_submit");
   require(!vk_sync_wait_is_known_pending(&bin_out->permanent, 0),
           "queued binary signal marked pending before driver_submit");

   /* Now signal tl = 20 on q0 and confirm q1's submit thread completes. */
   VkSemaphoreSubmitInfo sig20 = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = vk_semaphore_to_handle(tl),
      .value = 20,
   };
   VkSubmitInfo2 s_unblock = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig20,
   };
   require(vk_common_QueueSubmit2(vk_queue_to_handle(&q0), 1, &s_unblock,
                                  VK_NULL_HANDLE) == VK_SUCCESS,
           "unblock signal failed");
   wait_for_submits(submits_before + 2,
                    "threaded submit did not finish after signal");
   /* Wait for vk_queue_submit_final to finish noting pending values. */
   queue_wait_idle(&q1);
   require(vk_sync_wait_is_known_pending(&tl_out->permanent, 1),
           "threaded timeline signal did not note pending after submit");
   require(vk_sync_wait_is_known_pending(&bin_out->permanent, 0),
           "threaded binary signal did not note pending after submit");
   printf("wait-before-signal and threaded deferral: PASS\n");

   /* 8. A binary signal consumed while its submit is still inside
    * driver_submit must not be marked pending afterwards.
    *
    * q1 is threaded now.  Its submit thread installs X's payload and is held
    * inside driver_submit; the wait on X, submitted from this thread, sees
    * the payload through WAIT_PENDING and moves it to a temporary.  Once
    * driver_submit returns, X is empty and must not be known pending, or
    * the next wait on X skips WAIT_PENDING and moves an empty payload,
    * which on a threaded queue waits forever.
    */
   struct vk_semaphore *x = create_sem(&device, VK_SEMAPHORE_TYPE_BINARY, 0, 0);
   require(q1.submit.mode == VK_QUEUE_SUBMIT_MODE_THREADED,
           "q1 is not threaded");

   gate_close(GATE_AFTER_SIGNAL);
   require(submit_binary(&q1, NULL, x) == VK_SUCCESS,
           "threaded binary signal failed");
   wait_in_gate();
   require(submit_binary(&q1, x, NULL) == VK_SUCCESS,
           "threaded binary wait failed");
   gate_release();
   queue_wait_idle(&q1);
   gate_close(GATE_NONE);
   require_pending_has_payload(x, "consumed threaded binary signal "
                                  "left an empty payload marked pending");

   /* The next signal of X is still queued on q1's thread when the next wait
    * on X comes in; the wait must reach driver_submit.
    */
   int before = p_atomic_read(&driver_submit_calls);
   gate_close(GATE_BEFORE_SIGNAL);
   require(submit_binary(&q1, NULL, x) == VK_SUCCESS,
           "second threaded binary signal failed");
   wait_in_gate();
   thrd_t releaser;
   require(thrd_create(&releaser, release_gate_later, NULL) == thrd_success,
           "thrd_create failed");
   require(submit_binary(&q1, x, NULL) == VK_SUCCESS,
           "second threaded binary wait failed");
   wait_for_submits(before + 2,
                    "wait on a reused binary semaphore never reached "
                    "driver_submit");
   thrd_join(releaser, NULL);
   queue_wait_idle(&q1);
   gate_close(GATE_NONE);
   require_pending_has_payload(x, "reused threaded binary semaphore left "
                                  "an empty payload marked pending");

   /* Same race across queues: q1's thread installs X and is held inside
    * driver_submit while an immediate submit on q0 waits on X and resets
    * it.  X must not be known pending afterwards, and the next immediate
    * wait on X must probe it instead of taking the fast path.
    */
   require(q0.submit.mode == VK_QUEUE_SUBMIT_MODE_IMMEDIATE,
           "q0 is not immediate");
   gate_close(GATE_AFTER_SIGNAL);
   require(submit_binary(&q1, NULL, x) == VK_SUCCESS,
           "cross-queue binary signal failed");
   wait_in_gate();
   require(submit_binary(&q0, x, NULL) == VK_SUCCESS,
           "cross-queue immediate binary wait failed");
   require(q0.submit.mode == VK_QUEUE_SUBMIT_MODE_IMMEDIATE,
           "cross-queue wait unexpectedly started q0's submit thread");
   gate_release();
   queue_wait_idle(&q1);
   gate_close(GATE_NONE);
   require(!vk_sync_wait_is_known_pending(&x->permanent, 0),
           "cross-queue consumed binary signal left X marked pending");
   require_pending_has_payload(x, "cross-queue consumed binary signal left "
                                  "an empty payload marked pending");

   /* And with the signal on the immediate queue: q0's driver_submit runs
    * on another application thread and is held after installing X while a
    * wait on threaded q1 moves X's payload away.
    */
   gate_close(GATE_AFTER_SIGNAL);
   thrd_t signaler;
   struct immediate_signal sig_args = { .queue = &q0, .sem = x };
   require(thrd_create(&signaler, immediate_signal_thread,
                       &sig_args) == thrd_success,
           "thrd_create failed");
   wait_in_gate();
   require(submit_binary(&q1, x, NULL) == VK_SUCCESS,
           "threaded wait on immediate binary signal failed");
   gate_release();
   thrd_join(signaler, NULL);
   require(sig_args.result == VK_SUCCESS, "immediate binary signal failed");
   queue_wait_idle(&q1);
   gate_close(GATE_NONE);
   require(!vk_sync_wait_is_known_pending(&x->permanent, 0),
           "immediate binary signal consumed inside driver_submit left X "
           "marked pending");

   /* A clear that overlaps a host signal also wins. */
   const uint64_t token = vk_sync_pending_token(&x->permanent);
   vk_sync_clear_pending(&x->permanent);
   vk_sync_note_pending(&x->permanent, 0, token);
   require(!vk_sync_wait_is_known_pending(&x->permanent, 0),
           "stale token marked a cleared binary sync pending");
   printf("binary signal consumed inside driver_submit: PASS\n");
   destroy_sem(&device, x);

   vk_queue_finish(&q1);
   vk_queue_finish(&q0);
   destroy_sem(&device, bin_out);
   destroy_sem(&device, tl_out);
   destroy_sem(&device, bin);
   destroy_sem(&device, tl_share);
   destroy_sem(&device, tl_init);
   destroy_sem(&device, tl);
   vk_object_base_finish(&device.base);
   return EXIT_SUCCESS;
}
