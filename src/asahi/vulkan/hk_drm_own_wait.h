/* SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "drm-uapi/asahi_drm.h"
#include "util/macros.h"

/* A submission waits on the queue's own timeline for several reasons: a
 * replaced kernel queue's first submission waits for every earlier point,
 * an earlier empty submission's waits order later work, and work that
 * depends on an earlier submission waits for the sibling kernel queues.
 * The kernel refuses two points of one timeline in a submission, so all of
 * them share one entry, the latest point.
 *
 * *own_wait is that entry's index in syncs, -1 while there is none. A new
 * entry goes at syncs[*wait_count], after the waits; the `tail` entries
 * already placed there (the signals) move up by one, and the caller's
 * array has room for one more.
 */
static inline void
hk_drm_wait_own_timeline(struct drm_asahi_sync *syncs, unsigned *wait_count,
                         unsigned tail, int *own_wait, uint32_t syncobj,
                         uint64_t point)
{
   if (*own_wait >= 0) {
      syncs[*own_wait].timeline_value =
         MAX2(syncs[*own_wait].timeline_value, point);
      return;
   }

   memmove(&syncs[*wait_count + 1], &syncs[*wait_count], tail * sizeof(*syncs));
   *own_wait = *wait_count;
   syncs[(*wait_count)++] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
      .handle = syncobj,
      .timeline_value = point,
   };
}

/* The point that took an empty submission's waits still orders later work
 * unless the queue timeline is known to have reached it. A failed query
 * (queried false) proves nothing. */
static inline bool
hk_drm_empty_wait_pending(uint64_t empty_wait_point, bool queried,
                          uint64_t done)
{
   return empty_wait_point && !(queried && done >= empty_wait_point);
}
