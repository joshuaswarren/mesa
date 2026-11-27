/* SPDX-License-Identifier: MIT */

#include "vk_device.h"
#include "vk_drm_syncobj.h"
#include "util/os_time.h"
#include "util/u_sync_provider.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xf86drm.h>

/* Model a kernel wait whose payload may never arrive, independently of the
 * device-loss flag. This exercises the real runtime sync type without a GPU.
 */
struct test_provider {
   struct util_sync_provider base;
   struct vk_device *device;
   uint64_t payload_time;
   uint64_t loss_time;
   uint64_t probe_delay_ns;
   unsigned waits;
};

static void
require(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "%s\n", message);
      exit(EXIT_FAILURE);
   }
}

static void
sleep_until(uint64_t deadline)
{
   struct timespec ts = {
      .tv_sec = deadline / 1000000000,
      .tv_nsec = deadline % 1000000000,
   };
   while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR)
      ;
}

static int
create(struct util_sync_provider *p, uint32_t flags, uint32_t *handle)
{
   *handle = 1;
   return 0;
}

static int
destroy(struct util_sync_provider *p, uint32_t handle)
{
   return 0;
}

static int
probe_wait(struct util_sync_provider *p, uint32_t *handles, unsigned count,
           int64_t timeout, unsigned flags, uint32_t *first)
{
   return 0;
}

static int
pending_wait(struct util_sync_provider *base, uint32_t *handles,
             uint64_t *points, unsigned count, int64_t timeout,
             unsigned flags, uint32_t *first)
{
   struct test_provider *p = (struct test_provider *)base;
   require(count == 1 && handles[0] == 1 && points[0] == 0,
           "unexpected pending wait payload");
   require(flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE,
           "pending wait must wait for payload availability");
   p->waits++;

   const uint64_t now = os_time_get_nano();

   /* A loss whose wall time has passed is visible to every call, including
    * zero-timeout poll probes. */
   if (p->loss_time && now >= p->loss_time) {
      p_atomic_set(&p->device->_lost.lost, 1);
      p->loss_time = 0;
   } else if (p->loss_time && p->loss_time <= (uint64_t)timeout) {
      /* Losing the device does not wake the kernel's fence-availability
       * wait. */
      sleep_until(p->loss_time);
      p_atomic_set(&p->device->_lost.lost, 1);
      p->loss_time = 0;
   }

   /* A payload that already materialized is visible to any probe. */
   if (p->payload_time && now >= p->payload_time)
      return 0;

   if (p->payload_time && p->payload_time <= (uint64_t)timeout) {
      sleep_until(p->payload_time);
      return 0;
   }

   if ((uint64_t)timeout <= now) {
      /* Zero-timeout poll probes may straddle the caller's poll deadline;
       * a probe that takes real time makes the miss recording deterministic.
       */
      if (timeout == 0 && p->probe_delay_ns) {
         struct timespec ts = {0, (long)p->probe_delay_ns};
         while (nanosleep(&ts, &ts) == EINTR)
            ;
      }
      errno = ETIME;
      return -1;
   }

   sleep_until(timeout);
   errno = ETIME;
   return -1;
}

static void
run(const char *name, uint64_t payload_delay, uint64_t loss_delay,
    uint64_t timeout_delay, bool already_lost, VkResult expected)
{
   struct vk_device device = {0};
   /* The test supplies the loss flag directly; no driver report is needed. */
   device._lost.reported = true;
   device._lost.lost = already_lost;
   const uint64_t start = os_time_get_nano();
   struct test_provider provider = {
      .base = {
         .create = create,
         .destroy = destroy,
         .wait = probe_wait,
         .timeline_wait = pending_wait,
      },
      .device = &device,
      .payload_time = payload_delay ? start + payload_delay : 0,
      .loss_time = loss_delay ? start + loss_delay : 0,
   };
   device.sync = &provider.base;
   struct vk_sync_type type =
      vk_drm_syncobj_get_type_from_provider(&provider.base);
   struct vk_drm_syncobj sync = {
      .base = {.type = &type},
      .syncobj = 1,
   };
   const struct vk_sync_wait wait = {.sync = &sync.base, .wait_value = 0};
   const uint64_t deadline = timeout_delay ? start + timeout_delay : INT64_MAX;
   VkResult result = type.wait_many(&device, 1, &wait,
                                    VK_SYNC_WAIT_PENDING, deadline);
   require(result == expected, name);
   if (already_lost)
      require(provider.waits == 0, "lost device reached the kernel wait");
   if (expected == VK_TIMEOUT)
      require(os_time_get_nano() >= deadline, "healthy wait timed out early");
   printf("%s: PASS\n", name);
}

/* Same driver loop with device->sync_wait_poll_us set: the bounded
 * zero-timeout poll phase must still honour the caller's deadline, keep its
 * backoff state after a blocking fallback hit, and notice device loss.
 */
static void
run_poll(const char *name, uint64_t payload_delay, uint64_t loss_delay,
         uint64_t timeout_delay, uint32_t poll_us, uint64_t probe_delay_ns,
         VkResult expected, uint64_t max_elapsed, uint32_t min_misses)
{
   struct vk_device device = {0};
   device._lost.reported = true;
   device.sync_wait_poll_us = poll_us;
   const uint64_t start = os_time_get_nano();
   struct test_provider provider = {
      .base = {
         .create = create,
         .destroy = destroy,
         .wait = probe_wait,
         .timeline_wait = pending_wait,
      },
      .device = &device,
      .payload_time = payload_delay ? start + payload_delay : 0,
      .loss_time = loss_delay ? start + loss_delay : 0,
      .probe_delay_ns = probe_delay_ns,
   };
   device.sync = &provider.base;
   struct vk_sync_type type =
      vk_drm_syncobj_get_type_from_provider(&provider.base);
   struct vk_drm_syncobj sync = {
      .base = {.type = &type},
      .syncobj = 1,
   };
   const struct vk_sync_wait wait = {.sync = &sync.base, .wait_value = 0};
   const uint64_t deadline = timeout_delay ? start + timeout_delay : INT64_MAX;
   VkResult result = type.wait_many(&device, 1, &wait,
                                    VK_SYNC_WAIT_PENDING, deadline);
   const uint64_t elapsed = os_time_get_nano() - start;
   require(result == expected, name);
   if (max_elapsed)
      require(elapsed < max_elapsed, "wait ran past its bound");
   if (expected == VK_TIMEOUT)
      require(elapsed + 5000000 >= deadline - start,
              "wait timed out before its deadline");
   if (min_misses)
      require((uint32_t)p_atomic_read(&device.sync_poll_misses) >= min_misses,
              "blocking fallback cleared the poll backoff");
   printf("%s: PASS\n", name);
}

int
main(void)
{
   run("already lost", 0, 0, 0, true, VK_ERROR_DEVICE_LOST);
   run("loss while waiting forever", 0, 50000000, 0, false,
       VK_ERROR_DEVICE_LOST);
   run("payload after multiple wait intervals", 250000000, 0, 1000000000,
       false, VK_SUCCESS);
   run("finite deadline without a payload", 0, 0, 250000000, false, VK_TIMEOUT);
   /* A deadline that already passed must not spin the poll budget. */
   run_poll("poll: elapsed deadline returns at once", 0, 0, 1,
            200000 /* us */, 0, VK_TIMEOUT, 100000000, 0);
   /* A deadline shorter than the poll budget is still honoured. */
   run_poll("poll: short deadline is honoured", 0, 0, 20000000,
            200000, 0, VK_TIMEOUT, 150000000, 0);
   /* A hit on the blocking fallback must keep the recorded poll miss.
    * 8 ms probes guarantee one straddles the 20 ms poll deadline. */
   run_poll("poll: backoff survives blocking fallback", 60000000, 0,
            500000000, 20000, 8000000, VK_SUCCESS, 0, 1);
   /* Device loss is reported during the poll, not after the budget. */
   run_poll("poll: device loss during polling", 0, 50000000, 0,
            500000, 0, VK_ERROR_DEVICE_LOST, 300000000, 0);
   return EXIT_SUCCESS;
}
