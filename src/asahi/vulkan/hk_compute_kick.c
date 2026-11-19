/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 *
 * Compute kick for the M5 Max (G17C).
 *
 * While a compute command of one client runs, the G17C starts no larger
 * compute launch of other clients, and none of their work that depends on
 * one (a compositor's texture uploads and the frames that use them), until
 * that command ends. A command that runs for seconds stalls the other
 * clients for seconds, and long enough makes them lose their device. A small
 * compute job that completes on a second kernel queue of the client that
 * runs the long command, in the same priority class, lets the other
 * clients' work start. One such job helps for a while; one every few
 * milliseconds keeps all of it moving.
 *
 * So each priority class of the device's queues gets a kick queue: a queue
 * the application cannot get, with a kernel queue of its own on the
 * device's DRM file. A helper thread watches the device's queues. Once a
 * queue with unfinished compute work has completed nothing for
 * HK_KICK_AFTER_MS, the thread submits a kick to the kick queue of the
 * queue's class, and then one every hk_compute_kick_ms until the queue
 * completes something, with at most one kick in flight per kick queue. A
 * kick is a 16-byte vkCmdFillBuffer (one workgroup) into a buffer the kicker
 * owns, recorded once. Progress is tracked per submission: an application
 * whose submissions each complete within HK_KICK_AFTER_MS sends no kicks
 * (an inference engine's submissions of 20 to 30 ms, for example, would
 * otherwise get kicks they have no use for, and those cost it a few
 * percent), but one submission of many short commands that together run
 * longer gets kicks like one long command.
 *
 * The thread starts at the device's first compute submission and sleeps,
 * in a condition variable, while no watched queue has unfinished compute
 * work; otherwise in a wait for that work, until the next kick is due. The
 * kick queues' kernel queues and the kick's objects are created at the first
 * kick.
 *
 * The lock only guards the watched queues' kick state and the thread's
 * start and stop. The thread sets up and submits kicks without it: those
 * can wait for the application's submit threads after device loss, and
 * those threads take the lock in hk_compute_kick_note. The kick queues are
 * not in the device's queue list, so the application's queue walks
 * (vkDeviceWaitIdle and the like) never reach them; the kicker waits for
 * and finishes them itself.
 */

#include "hk_compute_kick.h"

#include <errno.h>
#include <strings.h>
#include <xf86drm.h>

#include "util/bitscan.h"
#include "util/cnd_monotonic.h"
#include "util/os_time.h"
#include "util/timespec.h"
#include "util/u_thread.h"
#include "vk_alloc.h"
#include "vk_device.h"

#include "hk_device.h"
#include "hk_queue.h"

/* Classes a queue can have: LOW, MEDIUM and HIGH. */
#define HK_KICK_CLASSES (DRM_ASAHI_PRIORITY_HIGH + 1)

/* How long a queue with unfinished compute work completes nothing before the
 * first kick. Longer than the commands of interactive and inference work. */
#define HK_KICK_AFTER_MS 50

/* Each class fills its own 16 bytes of the buffer. */
#define HK_KICK_BYTES 16

struct hk_kick_class {
   /* NULL if no watched queue has this class. Not in the device's queue
    * list; finished by hk_compute_kick_stop. */
   struct hk_queue *queue;

   /* The rest belongs to the thread, and to hk_compute_kick_stop once the
    * thread has exited. Created at the class's first kick. */
   bool ready, failed;
   VkCommandPool pool;
   VkCommandBuffer cmd;

   /* Timeline semaphore that the nth kick signals to n. */
   VkSemaphore done;
   uint64_t sent;
};

struct hk_compute_kick {
   struct hk_device *dev;
   int64_t period_ns;

   mtx_t lock;
   struct u_cnd_monotonic wake;
   thrd_t thread;
   bool started, idle, stop, failed;

   /* The device's queues that have a kernel queue of their own, and the
    * thread's arguments for waiting on them (thread only). */
   unsigned watch_count;
   struct hk_queue **watch;
   uint32_t *handles;
   uint64_t *points;

   /* Thread only. Created at the first kick. */
   bool buffer_failed;
   VkBuffer buffer;
   VkDeviceMemory memory;

   struct hk_kick_class class[HK_KICK_CLASSES];
};

static VkResult
hk_compute_kick_setup_buffer(struct hk_compute_kick *k)
{
   const struct vk_device_dispatch_table *t = &k->dev->vk.dispatch_table;
   VkDevice d = hk_device_to_handle(k->dev);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = HK_KICK_BYTES * HK_KICK_CLASSES,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkResult result = t->CreateBuffer(d, &buffer_info, NULL, &k->buffer);
   if (result != VK_SUCCESS)
      return result;

   VkMemoryRequirements req;
   t->GetBufferMemoryRequirements(d, k->buffer, &req);
   const VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = ffs(req.memoryTypeBits) - 1,
   };
   result = t->AllocateMemory(d, &alloc_info, NULL, &k->memory);
   if (result != VK_SUCCESS)
      return result;

   return t->BindBufferMemory(d, k->buffer, k->memory, 0);
}

static VkResult
hk_compute_kick_setup_class(struct hk_compute_kick *k, unsigned class)
{
   struct hk_kick_class *c = &k->class[class];
   const struct vk_device_dispatch_table *t = &k->dev->vk.dispatch_table;
   VkDevice d = hk_device_to_handle(k->dev);

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = c->queue->vk.queue_family_index,
   };
   VkResult result = t->CreateCommandPool(d, &pool_info, NULL, &c->pool);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = c->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   result = t->AllocateCommandBuffers(d, &cmd_info, &c->cmd);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   result = t->BeginCommandBuffer(c->cmd, &begin);
   if (result != VK_SUCCESS)
      return result;
   t->CmdFillBuffer(c->cmd, k->buffer, class * HK_KICK_BYTES, HK_KICK_BYTES,
                    0);
   result = t->EndCommandBuffer(c->cmd);
   if (result != VK_SUCCESS)
      return result;

   const VkSemaphoreTypeCreateInfo type_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
   };
   const VkSemaphoreCreateInfo sem_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &type_info,
   };
   return t->CreateSemaphore(d, &sem_info, NULL, &c->done);
}

/* Called by the thread without the lock: setup can wait for the
 * application's submit threads after device loss. Sends nothing while the
 * class's last kick is in flight. A class whose setup or submission fails
 * sends no more kicks. */
static void
hk_compute_kick_send(struct hk_compute_kick *k, unsigned class)
{
   struct hk_kick_class *c = &k->class[class];
   const struct vk_device_dispatch_table *t = &k->dev->vk.dispatch_table;
   VkDevice d = hk_device_to_handle(k->dev);

   if (!c->queue || c->failed || k->buffer_failed)
      return;

   if (!k->buffer && hk_compute_kick_setup_buffer(k) != VK_SUCCESS) {
      k->buffer_failed = true;
      return;
   }

   if (!c->ready) {
      if (hk_compute_kick_setup_class(k, class) != VK_SUCCESS) {
         c->failed = true;
         return;
      }
      c->ready = true;
   }

   if (c->sent) {
      uint64_t done;
      if (t->GetSemaphoreCounterValue(d, c->done, &done) != VK_SUCCESS) {
         c->failed = true;
         return;
      }
      if (done < c->sent)
         return;
   }

   const VkCommandBufferSubmitInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = c->cmd,
   };
   const VkSemaphoreSubmitInfo signal = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = c->done,
      .value = c->sent + 1,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmd_info,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signal,
   };
   if (t->QueueSubmit2(vk_queue_to_handle(&c->queue->vk), 1, &submit,
                       VK_NULL_HANDLE) != VK_SUCCESS) {
      c->failed = true;
      return;
   }
   c->sent++;
}

static int
hk_compute_kick_thread(void *data)
{
   struct hk_compute_kick *k = data;
   const int fd = k->dev->dev.fd;

   u_thread_setname("hk-compute-kick");

   mtx_lock(&k->lock);
   while (!k->stop) {
      /* A lost device completes nothing more and takes no kicks: wait for
       * hk_compute_kick_stop. */
      if (vk_device_is_lost_no_report(&k->dev->vk)) {
         u_cnd_monotonic_wait(&k->wake, &k->lock);
         continue;
      }

      const int64_t now = os_time_get_nano();
      int64_t deadline = INT64_MAX;
      unsigned count = 0;
      unsigned due = 0;

      for (unsigned i = 0; i < k->watch_count; i++) {
         struct hk_queue *queue = k->watch[i];
         if (!queue->kick.pending)
            continue;

         uint64_t value = 0;
         if (drmSyncobjQuery(fd, &queue->drm.syncobj, &value, 1) ||
             value >= queue->kick.pending) {
            queue->kick.pending = 0;
            queue->kick.due_ns = 0;
            continue;
         }

         /* New work, or the queue completed something: the first kick is
          * due once it has completed nothing for HK_KICK_AFTER_MS. */
         if (!queue->kick.due_ns || value != queue->kick.seen) {
            queue->kick.seen = value;
            queue->kick.due_ns = now + HK_KICK_AFTER_MS * 1000000ll;
         } else if (now >= queue->kick.due_ns) {
            due |= BITFIELD_BIT(queue->drm.virt_ring_idx - 1);
            queue->kick.due_ns = now + k->period_ns;
         }

         deadline = MIN2(deadline, queue->kick.due_ns);
         k->handles[count] = queue->drm.syncobj;
         k->points[count] = queue->kick.pending;
         count++;
      }

      if (!count) {
         k->idle = true;
         u_cnd_monotonic_wait(&k->wake, &k->lock);
         k->idle = false;
         continue;
      }

      mtx_unlock(&k->lock);

      u_foreach_bit(class, due)
         hk_compute_kick_send(k, class);

      /* Until one of the queues completes its compute work or a kick is
       * due. Work submitted meanwhile is seen at the latest then. */
      int ret = drmSyncobjTimelineWait(fd, k->handles, k->points, count,
                                       deadline,
                                       DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                       NULL);
      mtx_lock(&k->lock);

      /* Never spin on an unexpected error. */
      if (ret && ret != -ETIME && !k->stop) {
         const struct timespec until = {
            .tv_sec = deadline / NSEC_PER_SEC,
            .tv_nsec = deadline % NSEC_PER_SEC,
         };
         u_cnd_monotonic_timedwait(&k->wake, &k->lock, &until);
      }
   }
   mtx_unlock(&k->lock);

   return 0;
}

void
hk_compute_kick_note(struct hk_device *dev, struct hk_queue *queue)
{
   /* Valid until hk_compute_kick_finish, which runs once the application's
    * queues and their submit threads are gone. */
   struct hk_compute_kick *k = dev->compute_kick;

   mtx_lock(&k->lock);
   if (k->stop) {
      mtx_unlock(&k->lock);
      return;
   }
   queue->kick.pending = queue->drm.timeline_value;
   if (!k->started && !k->failed) {
      if (u_thread_create(&k->thread, hk_compute_kick_thread, k) ==
          thrd_success)
         k->started = true;
      else
         k->failed = true;
   } else if (k->idle) {
      u_cnd_monotonic_signal(&k->wake);
   }
   mtx_unlock(&k->lock);
}

static void
hk_compute_kick_free_mem(struct hk_device *dev, struct hk_compute_kick *k)
{
   vk_free(&dev->vk.alloc, k->watch);
   vk_free(&dev->vk.alloc, k->handles);
   vk_free(&dev->vk.alloc, k->points);
   vk_free(&dev->vk.alloc, k);
}

void
hk_compute_kick_init(struct hk_device *dev, unsigned period_ms)
{
   /* Only G17C: the kick is a property of its firmware's scheduling, not a
    * general cure for one client's long compute command holding up the
    * others. Measured with one client keeping GPU-filling compute commands
    * in flight and another submitting a 512-workgroup job every 100 ms:
    *
    *    M1 Pro (G13S), M2 (G14G): the job waits for the running command
    *       (500 ms commands: 500 ms; 1.5 s commands: 1.5 s). On the M1 Pro
    *       it still waits with the kicker running and with the kernel's
    *       compute preemption turned on.
    *    M4 Pro (G17P): the job does not wait (10-12 ms median, 69 ms worst
    *       beside 1.5 s commands), so there is nothing to kick.
    */
   if (dev->dev.chip != AGX_CHIP_G17C || !period_ms)
      return;

   unsigned count = 0;
   vk_foreach_queue(iter, &dev->vk) {
      if (!vk_queue_is_emulated(iter))
         count++;
   }
   if (!count)
      return;

   struct hk_compute_kick *k =
      vk_zalloc(&dev->vk.alloc, sizeof(*k), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!k)
      return;
   k->watch = vk_zalloc(&dev->vk.alloc, count * sizeof(*k->watch), 8,
                        VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   k->handles = vk_zalloc(&dev->vk.alloc, count * sizeof(*k->handles), 8,
                          VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   k->points = vk_zalloc(&dev->vk.alloc, count * sizeof(*k->points), 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!k->watch || !k->handles || !k->points) {
      hk_compute_kick_free_mem(dev, k);
      return;
   }
   if (mtx_init(&k->lock, mtx_plain) != thrd_success) {
      hk_compute_kick_free_mem(dev, k);
      return;
   }
   if (u_cnd_monotonic_init(&k->wake) != thrd_success) {
      mtx_destroy(&k->lock);
      hk_compute_kick_free_mem(dev, k);
      return;
   }

   k->dev = dev;
   k->period_ns = (int64_t)period_ms * 1000000;
   vk_foreach_queue(iter, &dev->vk) {
      if (!vk_queue_is_emulated(iter))
         k->watch[k->watch_count++] = container_of(iter, struct hk_queue, vk);
   }

   for (unsigned i = 0; i < k->watch_count; i++) {
      struct hk_queue *queue = k->watch[i];
      const unsigned class = queue->drm.virt_ring_idx - 1;
      if (class >= HK_KICK_CLASSES)
         continue;

      struct hk_kick_class *c = &k->class[class];
      if (!c->queue && !c->failed &&
          hk_queue_init_kick(dev, class, &c->queue) != VK_SUCCESS)
         c->failed = true;
      queue->kick.watched = c->queue != NULL;
   }

   dev->compute_kick = k;
}

void
hk_compute_kick_stop(struct hk_device *dev)
{
   struct hk_compute_kick *k = dev->compute_kick;
   if (!k)
      return;

   /* From here on notes change nothing, and no thread starts. */
   mtx_lock(&k->lock);
   k->stop = true;
   u_cnd_monotonic_signal(&k->wake);
   mtx_unlock(&k->lock);
   if (k->started)
      thrd_join(k->thread, NULL);

   /* The kick objects are the thread's, now exited. Nothing below holds
    * the lock: freeing them can wait for the application's submit threads
    * after device loss, and those may still call hk_compute_kick_note. */
   const struct vk_device_dispatch_table *t = &dev->vk.dispatch_table;
   VkDevice d = hk_device_to_handle(dev);

   for (unsigned class = 0; class < HK_KICK_CLASSES; class++) {
      struct hk_kick_class *c = &k->class[class];

      /* The kicks write the kicker's buffer and run the device's fill
       * kernel: let them complete before either goes away. */
      if (c->sent) {
         const VkSemaphoreWaitInfo wait = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &c->done,
            .pValues = &c->sent,
         };
         t->WaitSemaphores(d, &wait, UINT64_MAX);
      }

      /* Before the objects its submissions refer to. */
      if (c->queue)
         hk_queue_finish(dev, c->queue);
      c->queue = NULL;

      if (c->cmd)
         t->FreeCommandBuffers(d, c->pool, 1, &c->cmd);
      if (c->pool)
         t->DestroyCommandPool(d, c->pool, NULL);
      if (c->done)
         t->DestroySemaphore(d, c->done, NULL);
   }

   if (k->buffer)
      t->DestroyBuffer(d, k->buffer, NULL);
   if (k->memory)
      t->FreeMemory(d, k->memory, NULL);
}

void
hk_compute_kick_finish(struct hk_device *dev)
{
   struct hk_compute_kick *k = dev->compute_kick;
   if (!k)
      return;

   dev->compute_kick = NULL;
   u_cnd_monotonic_destroy(&k->wake);
   mtx_destroy(&k->lock);
   hk_compute_kick_free_mem(dev, k);
}
