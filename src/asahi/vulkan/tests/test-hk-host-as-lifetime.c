/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Host-operation lifetime test with dispatch fixtures. The empty build batch
 * isolates the internal pool/queue lifecycle; it is not Vulkan conformance.
 * No command recording backend, DRM device, or GPU is used.
 */
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_queue.h"
#undef NDEBUG
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static struct hk_device dev;
static struct hk_queue queue;
static atomic_uint active, peak, fences;
static unsigned fail_stage;
static _Thread_local unsigned stage;
static pthread_barrier_t start;

static VkResult step(void)
{
   return ++stage == fail_stage ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

static VkResult VKAPI_CALL
create_fence(VkDevice d, const VkFenceCreateInfo *ci,
             const VkAllocationCallbacks *a, VkFence *out)
{
   VkResult result = step();
   if (result != VK_SUCCESS) return result;
   *out = (VkFence)(uintptr_t)malloc(1);
   assert(*out);
   atomic_fetch_add(&fences, 1);
   return VK_SUCCESS;
}

static void VKAPI_CALL
destroy_fence(VkDevice d, VkFence fence, const VkAllocationCallbacks *a)
{
   free((void *)(uintptr_t)fence);
   assert(atomic_fetch_sub(&fences, 1));
}

static VkResult VKAPI_CALL
allocate_cmd(VkDevice d, const VkCommandBufferAllocateInfo *ci, VkCommandBuffer *out)
{
   VkResult result = step();
   if (result != VK_SUCCESS) return result;
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   assert(cmd);
   vk_object_base_init(&dev.vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   *out = hk_cmd_buffer_to_handle(cmd);
   unsigned n = atomic_fetch_add(&active, 1) + 1;
   unsigned old = atomic_load(&peak);
   while (n > old && !atomic_compare_exchange_weak(&peak, &old, n)) {}
   return VK_SUCCESS;
}

static void VKAPI_CALL
free_cmd(VkDevice d, VkCommandPool pool, uint32_t count, const VkCommandBuffer *cmds)
{
   assert(count == 1);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cmds[0]);
   vk_object_base_finish(&cmd->vk.base);
   free(cmd);
   assert(atomic_fetch_sub(&active, 1));
}

static VkResult VKAPI_CALL
begin_cmd(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *ci)
{
   /* Pool access includes recording, not just allocation/free. The
    * single-threaded runs in main() reach this with no other thread able to
    * hold the lock, so it is held here by the operation being recorded.
    * simple_mtx_assert_locked() vanishes when the headers above were
    * compiled with NDEBUG, so read the futex word directly. */
#if UTIL_FUTEX_SUPPORTED
   assert(p_atomic_read(&dev.host_as.lock.val) != 0);
#else
   simple_mtx_assert_locked(&dev.host_as.lock);
#endif
   return step();
}
static VkResult VKAPI_CALL end_cmd(VkCommandBuffer cmd) { return step(); }
static VkResult VKAPI_CALL
submit(VkQueue q, uint32_t count, const VkSubmitInfo2 *s, VkFence fence)
{ return step(); }
static VkResult VKAPI_CALL
wait_fences(VkDevice d, uint32_t count, const VkFence *f, VkBool32 all, uint64_t ns)
{ return step(); }

void __wrap_hk_meta_begin(struct hk_cmd_buffer *, struct hk_meta_save *, VkPipelineBindPoint);
void __wrap_hk_meta_end(struct hk_cmd_buffer *, struct hk_meta_save *, VkPipelineBindPoint);
void __wrap_hk_meta_begin(struct hk_cmd_buffer *c, struct hk_meta_save *s, VkPipelineBindPoint p) {}
void __wrap_hk_meta_end(struct hk_cmd_buffer *c, struct hk_meta_save *s, VkPipelineBindPoint p) {}

static VkResult run(VkDeferredOperationKHR deferred)
{
   stage = 0;
   return hk_BuildAccelerationStructuresKHR(hk_device_to_handle(&dev), deferred, 0, NULL, NULL);
}

static void *worker(void *unused)
{
   pthread_barrier_wait(&start);
   for (unsigned i = 0; i < 64; ++i)
      assert(run(VK_NULL_HANDLE) == VK_SUCCESS);
   return NULL;
}

int main(void)
{
   vk_object_base_init(&dev.vk, &dev.vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(&dev.vk, &queue.vk.base, VK_OBJECT_TYPE_QUEUE);
   dev.g17p_native_as = true;
   dev.host_as.queue = &queue;
   dev.host_as.pool = (VkCommandPool)(uintptr_t)1;
   simple_mtx_init(&dev.host_as.lock, mtx_plain);
   dev.vk.dispatch_table.CreateFence = create_fence;
   dev.vk.dispatch_table.DestroyFence = destroy_fence;
   dev.vk.dispatch_table.AllocateCommandBuffers = allocate_cmd;
   dev.vk.dispatch_table.FreeCommandBuffers = free_cmd;
   dev.vk.dispatch_table.BeginCommandBuffer = begin_cmd;
   dev.vk.dispatch_table.EndCommandBuffer = end_cmd;
   dev.vk.dispatch_table.QueueSubmit2 = submit;
   dev.vk.dispatch_table.WaitForFences = wait_fences;

   for (unsigned i = 1; i <= 6; ++i) {
      fail_stage = i;
      assert(run(VK_NULL_HANDLE) == VK_ERROR_OUT_OF_HOST_MEMORY);
      assert(atomic_load(&active) == 0 && atomic_load(&fences) == 0);
      fail_stage = 0;
      assert(run(VK_NULL_HANDLE) == VK_SUCCESS);
   }
   assert(run((VkDeferredOperationKHR)(uintptr_t)1) == VK_OPERATION_NOT_DEFERRED_KHR);
   pthread_barrier_init(&start, NULL, 2);
   pthread_t threads[2];
   for (unsigned i = 0; i < 2; ++i) assert(!pthread_create(&threads[i], NULL, worker, NULL));
   for (unsigned i = 0; i < 2; ++i) assert(!pthread_join(threads[i], NULL));
   pthread_barrier_destroy(&start);
   assert(atomic_load(&peak) == 1);
   assert(atomic_load(&active) == 0 && atomic_load(&fences) == 0);
   simple_mtx_destroy(&dev.host_as.lock);
   vk_object_base_finish(&queue.vk.base);
   vk_object_base_finish(&dev.vk.base);
   puts("PASS: 128 concurrent host operations serialized; six dispatch failures release pool, fence and lock");
   return 0;
}
