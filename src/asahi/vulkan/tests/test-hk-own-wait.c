/* SPDX-License-Identifier: MIT
 * The sync array of a kernel submission waits on the queue's own timeline
 * at most once (the kernel refuses two points of one timeline), at the
 * latest point any reason asks for: a replaced kernel queue, an earlier
 * empty submission's waits, or the sibling queues. The signals placed
 * before the sibling wait is added keep their order. No GPU used.
 */
#include <stdio.h>
#include <stdlib.h>
#include "hk_drm_own_wait.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

#define QUEUE_SYNCOBJ 7
#define MAX_WAITS     3
#define MAX_SIGNALS   2

/* The sync array queue_submit_recordings() builds, in its order. 0 for a
 * point means that reason is absent. */
struct case_in {
   uint64_t replaced, empty, sib;
   unsigned waits, signals;
};

struct built {
   struct drm_asahi_sync syncs[MAX_WAITS + MAX_SIGNALS + 3];
   unsigned wait_count, signal_count;
};

static void
build(const struct case_in *c, struct built *out)
{
   struct drm_asahi_sync *syncs = out->syncs;
   unsigned wait_count = 0;
   int own_wait = -1;

   memset(out, 0xa5, sizeof(*out));

   if (c->replaced)
      hk_drm_wait_own_timeline(syncs, &wait_count, 0, &own_wait, QUEUE_SYNCOBJ,
                               c->replaced);
   if (c->empty)
      hk_drm_wait_own_timeline(syncs, &wait_count, 0, &own_wait, QUEUE_SYNCOBJ,
                               c->empty);

   /* Application waits: other syncobjs. */
   for (unsigned i = 0; i < c->waits; ++i)
      syncs[wait_count++] = (struct drm_asahi_sync){
         .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
         .handle = 100 + i,
         .timeline_value = 1000 + i,
      };

   for (unsigned i = 0; i < c->signals; ++i)
      syncs[wait_count + i] = (struct drm_asahi_sync){
         .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
         .handle = 200 + i,
      };
   syncs[wait_count + c->signals] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
      .handle = QUEUE_SYNCOBJ,
      .timeline_value = 5000,
   };

   /* Only without a replaced kernel queue, as in queue_submit_recordings(). */
   if (c->sib && !c->replaced)
      hk_drm_wait_own_timeline(syncs, &wait_count, c->signals + 1, &own_wait,
                               QUEUE_SYNCOBJ, c->sib);

   out->wait_count = wait_count;
   out->signal_count = c->signals + 1;
}

static void
check(const struct case_in *c)
{
   struct built b;
   build(c, &b);

   uint64_t expect = MAX2(c->replaced, c->empty);
   if (!c->replaced)
      expect = MAX2(expect, c->sib);

   /* One own-timeline wait at the latest point, or none. */
   unsigned own = 0;
   uint64_t own_point = 0;
   for (unsigned i = 0; i < b.wait_count; ++i) {
      if (b.syncs[i].handle == QUEUE_SYNCOBJ) {
         ++own;
         own_point = b.syncs[i].timeline_value;
         CHECK(b.syncs[i].sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ);
      }
   }
   CHECK(own == (expect ? 1 : 0));
   CHECK(own_point == expect);

   /* Every application wait is there once, in order. */
   CHECK(b.wait_count == c->waits + own);
   unsigned next = 0;
   for (unsigned i = 0; i < b.wait_count; ++i) {
      if (b.syncs[i].handle == QUEUE_SYNCOBJ)
         continue;
      CHECK(b.syncs[i].handle == 100 + next);
      CHECK(b.syncs[i].timeline_value == 1000 + next);
      ++next;
   }
   CHECK(next == c->waits);

   /* The signals follow the waits unchanged; the queue's own signal ends
    * the array. */
   for (unsigned i = 0; i < c->signals; ++i) {
      CHECK(b.syncs[b.wait_count + i].sync_type == DRM_ASAHI_SYNC_SYNCOBJ);
      CHECK(b.syncs[b.wait_count + i].handle == 200 + i);
   }
   const struct drm_asahi_sync *last = &b.syncs[b.wait_count + c->signals];
   CHECK(last->handle == QUEUE_SYNCOBJ && last->timeline_value == 5000);

   /* queue_submit_recordings() sizes the array for wait_count + signals +
    * 3; the own-timeline entry takes one of those at most. */
   CHECK(b.wait_count + b.signal_count <= c->waits + c->signals + 3);
}

int
main(void)
{
   /* Every combination of reasons, in both point orders, with and without
    * application waits and signals. */
   static const uint64_t points[] = {0, 3, 9};
   unsigned cases = 0;
   for (unsigned r = 0; r < 3; ++r)
      for (unsigned e = 0; e < 3; ++e)
         for (unsigned s = 0; s < 3; ++s)
            for (unsigned w = 0; w <= MAX_WAITS; ++w)
               for (unsigned g = 0; g <= MAX_SIGNALS; ++g) {
                  const struct case_in c = {
                     .replaced = points[r],
                     .empty = points[e],
                     .sib = points[s],
                     .waits = w,
                     .signals = g,
                  };
                  check(&c);
                  ++cases;
               }

   /* The sibling wait after an empty submission's raises its entry and
    * adds none, so the signals stay where they were placed. */
   {
      struct drm_asahi_sync syncs[4] = {0};
      unsigned wait_count = 0;
      int own_wait = -1;
      hk_drm_wait_own_timeline(syncs, &wait_count, 0, &own_wait, QUEUE_SYNCOBJ,
                               4);
      syncs[1] = (struct drm_asahi_sync){.handle = 200};
      syncs[2] = (struct drm_asahi_sync){.handle = QUEUE_SYNCOBJ,
                                         .timeline_value = 5000};
      hk_drm_wait_own_timeline(syncs, &wait_count, 2, &own_wait, QUEUE_SYNCOBJ,
                               6);
      CHECK(own_wait == 0 && wait_count == 1);
      CHECK(syncs[0].timeline_value == 6);
      CHECK(syncs[1].handle == 200);
      CHECK(syncs[2].timeline_value == 5000);
      CHECK(syncs[3].handle == 0);

      /* A lower point never lowers the wait. */
      hk_drm_wait_own_timeline(syncs, &wait_count, 2, &own_wait, QUEUE_SYNCOBJ,
                               2);
      CHECK(syncs[0].timeline_value == 6 && wait_count == 1);
   }

   /* The empty submission's point is dropped only once the timeline is
    * known to have reached it. */
   CHECK(!hk_drm_empty_wait_pending(0, true, 0));
   CHECK(!hk_drm_empty_wait_pending(0, false, 0));
   CHECK(hk_drm_empty_wait_pending(5, true, 4));
   CHECK(!hk_drm_empty_wait_pending(5, true, 5));
   CHECK(!hk_drm_empty_wait_pending(5, true, 6));
   /* A failed query says nothing, whatever it left in `done`. */
   CHECK(hk_drm_empty_wait_pending(5, false, 6));
   CHECK(hk_drm_empty_wait_pending(UINT64_MAX, true, UINT64_MAX - 1));
   CHECK(!hk_drm_empty_wait_pending(UINT64_MAX, true, UINT64_MAX));

   printf("hk_own_wait: OK (%u cases)\n", cases);
   return 0;
}
