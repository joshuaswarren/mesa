/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_va_runtime_hole.h"

/* A user window like the one agx_open_device builds: 1 TiB of user VA. */
#define USER_START 0x2000000000ull
#define USER_SIZE  0x10000000000ull
#define PAGE       16384ull

static unsigned failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,      \
                 #cond);                                                       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static struct agx_device *
open_va(void)
{
   struct agx_device *dev = calloc(1, sizeof(*dev));
   simple_mtx_init(&dev->vma_lock, mtx_plain);
   dev->guard_size = PAGE;
   util_vma_heap_init(&dev->main_heap, USER_START, USER_SIZE);
   dev->replay_heap_initialized = agx_capture_replay_heap_init(
      &dev->main_heap, &dev->replay_heap, USER_SIZE, &dev->replay_start,
      &dev->replay_size);
   return dev;
}

static void
close_va(struct agx_device *dev)
{
   util_vma_heap_finish(&dev->replay_heap);
   util_vma_heap_finish(&dev->main_heap);
   simple_mtx_destroy(&dev->vma_lock);
   free(dev);
}

static bool
inside(const struct agx_device *dev, const struct agx_va *va)
{
   return va->addr < dev->replay_start + dev->replay_size &&
          dev->replay_start < va->addr + va->size_B;
}

int
main(void)
{
   struct agx_device *dev = open_va();
   CHECK(dev->replay_heap_initialized);
   const uint64_t start = dev->replay_start, size = dev->replay_size;
   const uint64_t end = start + size, sz = 4 * PAGE;

   /* The range is carved out of the user window and is aligned. */
   CHECK(size == 64ull << 30);
   CHECK(start >= USER_START && end <= USER_START + USER_SIZE);
   CHECK(start % (1ull << 30) == 0);

   /* Valid and invalid replay addresses. The guard counts as part of the
    * allocation, and nothing may wrap. */
   CHECK(agx_va_capture_valid(dev, start, sz, PAGE));
   CHECK(agx_va_capture_valid(dev, end - sz - dev->guard_size, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, end - sz, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, end, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, start - PAGE, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, 0, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, UINT64_MAX - PAGE + 1, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, start + 4096, sz, PAGE));
   CHECK(!agx_va_capture_valid(dev, start, UINT64_MAX - 100, PAGE));
   CHECK(!agx_va_capture_valid(dev, start, size, PAGE));
   CHECK(!agx_va_capture_valid(dev, start, sz, 0));

   /* A fixed request inside the range works once, then the address is in use.
    * One outside the range is refused without asserting. */
   struct agx_va *a = agx_va_alloc(dev, sz, PAGE,
                                   AGX_VA_CAPTURE_REPLAY | AGX_VA_FIXED,
                                   start + 8 * PAGE);
   CHECK(a && a->addr == start + 8 * PAGE);
   CHECK(!agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE_REPLAY | AGX_VA_FIXED,
                       start + 8 * PAGE));
   CHECK(!agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE_REPLAY | AGX_VA_FIXED,
                       end - PAGE));
   CHECK(!agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE_REPLAY | AGX_VA_FIXED,
                       UINT64_MAX - PAGE + 1));
   agx_va_free(dev, a, false);
   a = agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE_REPLAY | AGX_VA_FIXED,
                    start + 8 * PAGE);
   CHECK(a && a->addr == start + 8 * PAGE);
   agx_va_free(dev, a, false);

   /* Ordinary allocations never land in the range, even when the ordinary
    * heap runs dry. */
   for (unsigned i = 0; i < 64; ++i) {
      struct agx_va *va = agx_va_alloc(dev, 4ull << 30, PAGE, 0, 0);
      if (!va)
         break;
      CHECK(!inside(dev, va));
   }
   struct agx_va *big = agx_va_alloc(dev, 4 * USER_SIZE, PAGE, 0, 0);
   CHECK(!big);

   /* Replay buffers without an address come from the range. */
   struct agx_va *r = agx_va_alloc(dev, sz, PAGE, AGX_VA_CAPTURE_REPLAY, 0);
   CHECK(r && inside(dev, r));
   agx_va_free(dev, r, false);

   close_va(dev);

   if (failures) {
      fprintf(stderr, "%u checks failed\n", failures);
      return 1;
   }
   puts("PASS: capture/replay address range");
   return 0;
}
