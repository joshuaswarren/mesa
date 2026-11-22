/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx_bo.h"
#include "agx_device.h"
#include "util/os_time.h"

#define MIB (1024ull * 1024ull)

struct mock_state {
   uint32_t next_handle;
   uint32_t freed_handles[64];
   unsigned freed_count;
   bool expect_bo_map_locked;
   bool madvise_retain;
   unsigned dontneed_calls;
   unsigned willneed_calls;
   uint32_t last_dontneed_handle;
   uint32_t last_willneed_handle;
};

static struct mock_state mock;

static struct agx_bo *
mock_bo_alloc(struct agx_device *dev, size_t size, size_t align,
              enum agx_bo_flags flags)
{
   const uint32_t handle = ++mock.next_handle;

   simple_mtx_lock(&dev->bo_map_lock);
   struct agx_bo *bo = agx_lookup_bo(dev, handle);
   agx_include_bo_handle(dev, handle);
   simple_mtx_unlock(&dev->bo_map_lock);

   assert(!memcmp(bo, &((struct agx_bo){}), sizeof(*bo)));
   bo->dev = dev;
   bo->size = size;
   bo->align = align;
   bo->flags = flags;
   bo->handle = bo->uapi_handle = handle;
   bo->prime_fd = -1;
   return bo;
}

static void
assert_bo_cache_unlocked(struct agx_device *dev)
{
   /* Acquire and release bo_cache.lock without touching simple_mtx_t
    * internals; if the caller erroneously held the non-recursive lock, this
    * deadlocks cleanly and fails the test timeout. */
   simple_mtx_lock(&dev->bo_cache.lock);
   simple_mtx_unlock(&dev->bo_cache.lock);
   (void)agx_bo_cache_has(dev, AIL_PAGESIZE, AIL_PAGESIZE, 0);
}

static void
mock_bo_free(struct agx_device *dev, uint32_t handle)
{
   /* Deferred free invariant: bo_cache.lock must never be held across
    * agx_bo_free(), while eviction from agx_bo_unreference() preserves
    * bo_map_lock. */
   assert_bo_cache_unlocked(dev);

   if (mock.expect_bo_map_locked)
      simple_mtx_assert_locked(&dev->bo_map_lock);

   assert(mock.freed_count < ARRAY_SIZE(mock.freed_handles));
   mock.freed_handles[mock.freed_count++] = handle;
}

static int
mock_bo_madvise(struct agx_device *dev, uint32_t handle, bool dontneed,
                bool *retained)
{
   if (dontneed) {
      mock.dontneed_calls++;
      mock.last_dontneed_handle = handle;
      *retained = true;
   } else {
      mock.willneed_calls++;
      mock.last_willneed_handle = handle;
      *retained = mock.madvise_retain;
   }
   return 0;
}

static void
init_mock_device(struct agx_device *dev)
{
   memset(dev, 0, sizeof(*dev));
   memset(&mock, 0, sizeof(mock));
   mock.madvise_retain = true;

   util_sparse_array_init(&dev->bo_map, sizeof(struct agx_bo), 64);
   simple_mtx_init(&dev->bo_map_lock, mtx_plain);
   simple_mtx_init(&dev->bo_mmap_lock, mtx_plain);
   simple_mtx_init(&dev->bo_cache.lock, mtx_plain);
   simple_mtx_init(&dev->vma_lock, mtx_plain);

   list_inithead(&dev->bo_cache.lru);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i)
      list_inithead(&dev->bo_cache.buckets[i]);

   dev->ops.bo_alloc = mock_bo_alloc;
   dev->ops.bo_free = mock_bo_free;
   dev->ops.bo_madvise = mock_bo_madvise;
}

static void
finish_mock_device(struct agx_device *dev)
{
   mock.expect_bo_map_locked = false;
   agx_bo_cache_evict_all(dev);
   assert(dev->bo_cache.size == 0);
   assert(list_is_empty(&dev->bo_cache.lru));

   util_sparse_array_finish(&dev->bo_map);
   simple_mtx_destroy(&dev->bo_cache.lock);
   simple_mtx_destroy(&dev->bo_map_lock);
   simple_mtx_destroy(&dev->bo_mmap_lock);
   simple_mtx_destroy(&dev->vma_lock);
}

static void
test_lifo_bucket_reuse(void)
{
   struct agx_device dev;
   init_mock_device(&dev);

   const size_t sz = 64 * 1024;
   struct agx_bo *bo1 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo1");
   struct agx_bo *bo2 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo2");
   struct agx_bo *bo3 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo3");
   assert(bo1 && bo2 && bo3);

   const uint32_t h1 = bo1->handle;
   const uint32_t h2 = bo2->handle;
   const uint32_t h3 = bo3->handle;

   /* Put in order bo1, bo2, bo3. */
   agx_bo_unreference(&dev, bo1);
   agx_bo_unreference(&dev, bo2);
   agx_bo_unreference(&dev, bo3);
   assert(dev.bo_cache.size == 3 * sz);

   /* LIFO: most-recently-put BO must come back first. */
   struct agx_bo *f1 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "f1");
   struct agx_bo *f2 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "f2");
   struct agx_bo *f3 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "f3");
   assert(f1->handle == h3);
   assert(f2->handle == h2);
   assert(f3->handle == h1);
   assert(dev.bo_cache.size == 0);

   agx_bo_unreference(&dev, f1);
   agx_bo_unreference(&dev, f2);
   agx_bo_unreference(&dev, f3);
   finish_mock_device(&dev);
}

static void
test_lifo_avoids_purgeable_willneed(void)
{
   struct agx_device dev;
   init_mock_device(&dev);

   const size_t sz = 64 * 1024;
   struct agx_bo *old_bo = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "old");
   struct agx_bo *fresh_bo = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "fresh");
   const uint32_t old_h = old_bo->handle;
   const uint32_t fresh_h = fresh_bo->handle;

   agx_bo_unreference(&dev, old_bo);

   /* Age old_bo past AGX_BO_CACHE_ADVISE_NS (250 ms) while keeping it fresh
    * enough not to be evicted as stale (> 2 s). */
   simple_mtx_lock(&dev.bo_cache.lock);
   old_bo->cached_ns -= INT64_C(500) * 1000 * 1000;
   simple_mtx_unlock(&dev.bo_cache.lock);

   /* Putting fresh_bo triggers agx_bo_cache_advise_aged() on old_bo. */
   agx_bo_unreference(&dev, fresh_bo);
   assert(mock.dontneed_calls == 1);
   assert(mock.last_dontneed_handle == old_h);
   assert(old_bo->purgeable);
   assert(!fresh_bo->purgeable);

   /* LIFO fetch returns fresh_bo first with zero WILLNEED ioctls. */
   struct agx_bo *got1 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "got1");
   assert(got1->handle == fresh_h);
   assert(mock.willneed_calls == 0);

   /* Next fetch falls back to the older purgeable BO and reclaims it. */
   struct agx_bo *got2 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "got2");
   assert(got2->handle == old_h);
   assert(mock.willneed_calls == 1);
   assert(mock.last_willneed_handle == old_h);
   assert(!got2->purgeable);

   /* Also verify purged old_bo is skipped when a fresh entry exists, and freed
    * cleanly when eventually reached. */
   mock.madvise_retain = false;
   mock.willneed_calls = 0;
   mock.freed_count = 0;
   mock.expect_bo_map_locked = true;

   agx_bo_unreference(&dev, got2);
   simple_mtx_lock(&dev.bo_cache.lock);
   got2->cached_ns -= INT64_C(500) * 1000 * 1000;
   simple_mtx_unlock(&dev.bo_cache.lock);
   agx_bo_unreference(&dev, got1);
   assert(got2->purgeable && !got1->purgeable);

   struct agx_bo *got3 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "got3");
   assert(got3->handle == fresh_h);
   assert(mock.willneed_calls == 0);
   assert(mock.freed_count == 0);

   /* Fetching again hits purged old_bo, frees it outside bo_cache.lock (under
    * bo_map_lock), and allocates a new BO. */
   struct agx_bo *got4 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "got4");
   assert(got4 != NULL);
   assert(got4->handle != old_h);
   assert(mock.willneed_calls == 1);
   assert(mock.freed_count == 1);
   assert(mock.freed_handles[0] == old_h);

   agx_bo_unreference(&dev, got3);
   agx_bo_unreference(&dev, got4);
   finish_mock_device(&dev);
}

static void
test_lru_stale_eviction_drops_oldest(void)
{
   struct agx_device dev;
   init_mock_device(&dev);
   mock.expect_bo_map_locked = true;

   const size_t sz = 64 * 1024;
   struct agx_bo *bo1 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo1");
   struct agx_bo *bo2 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo2");
   struct agx_bo *bo3 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo3");
   struct agx_bo *bo4 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo4");
   const uint32_t h1 = bo1->handle;
   const uint32_t h2 = bo2->handle;
   const uint32_t h3 = bo3->handle;
   const uint32_t h4 = bo4->handle;

   agx_bo_unreference(&dev, bo1);
   agx_bo_unreference(&dev, bo2);
   agx_bo_unreference(&dev, bo3);

   /* Backdate only the oldest entry past the 2-second stale threshold. */
   simple_mtx_lock(&dev.bo_cache.lock);
   bo1->last_used -= 5;
   simple_mtx_unlock(&dev.bo_cache.lock);

   /* Putting bo4 triggers stale eviction: bo1 (oldest on LRU) must be evicted
    * and freed outside bo_cache.lock while bo2..bo4 remain cached. */
   agx_bo_unreference(&dev, bo4);
   assert(mock.freed_count == 1);
   assert(mock.freed_handles[0] == h1);
   assert(dev.bo_cache.size == 3 * sz);

   /* Remaining entries still pop in LIFO order: bo4, bo3, bo2. */
   struct agx_bo *f1 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "f1");
   struct agx_bo *f2 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "f2");
   struct agx_bo *f3 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "f3");
   assert(f1->handle == h4);
   assert(f2->handle == h3);
   assert(f3->handle == h2);

   agx_bo_unreference(&dev, f1);
   agx_bo_unreference(&dev, f2);
   agx_bo_unreference(&dev, f3);
   finish_mock_device(&dev);
}

static void
test_lru_capacity_eviction_drops_oldest(void)
{
   struct agx_device dev;
   init_mock_device(&dev);
   mock.expect_bo_map_locked = true;

   /* Cache max size is 256 MiB. Put two 128 MiB BOs, then a 64 MiB BO. */
   struct agx_bo *bo1 = agx_bo_create(&dev, 128 * MIB, AIL_PAGESIZE, 0, "a");
   struct agx_bo *bo2 = agx_bo_create(&dev, 128 * MIB, AIL_PAGESIZE, 0, "b");
   struct agx_bo *bo3 = agx_bo_create(&dev, 64 * MIB, AIL_PAGESIZE, 0, "c");
   const uint32_t h1 = bo1->handle;
   const uint32_t h2 = bo2->handle;
   const uint32_t h3 = bo3->handle;

   agx_bo_unreference(&dev, bo1);
   agx_bo_unreference(&dev, bo2);
   assert(dev.bo_cache.size == 256 * MIB);
   assert(mock.freed_count == 0);

   /* Putting bo3 must evict bo1 (oldest on LRU) outside bo_cache.lock. */
   agx_bo_unreference(&dev, bo3);
   assert(mock.freed_count == 1);
   assert(mock.freed_handles[0] == h1);
   assert(dev.bo_cache.size == 192 * MIB);

   struct agx_bo *f3 = agx_bo_create(&dev, 64 * MIB, AIL_PAGESIZE, 0, "fc");
   struct agx_bo *f2 = agx_bo_create(&dev, 128 * MIB, AIL_PAGESIZE, 0, "fb");
   assert(f3->handle == h3);
   assert(f2->handle == h2);

   agx_bo_unreference(&dev, f3);
   agx_bo_unreference(&dev, f2);
   finish_mock_device(&dev);
}

static void
test_idle_pending_wake_protocol(void)
{
   struct agx_device dev;
   init_mock_device(&dev);

   /* Set up idle_lock/idle_cond without spawning the background thread so the
    * test stays deterministic and sleep-free. */
   assert(mtx_init(&dev.bo_cache.idle_lock, mtx_plain) == thrd_success);
   assert(u_cnd_monotonic_init(&dev.bo_cache.idle_cond) == thrd_success);
   dev.bo_cache.idle_started = true;

   const size_t sz = 64 * 1024;
   struct agx_bo *bo1 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo1");
   struct agx_bo *bo2 = agx_bo_create(&dev, sz, AIL_PAGESIZE, 0, "bo2");

   /* Slow path: idle_pending == false must acquire idle_lock and set it. */
   p_atomic_set(&dev.bo_cache.idle_pending, false);
   agx_bo_unreference(&dev, bo1);
   assert(p_atomic_read(&dev.bo_cache.idle_pending) == true);

   /* Fast path: when idle_pending is already true, put must not touch
    * idle_lock (holding idle_lock here would deadlock if it tried). */
   mtx_lock(&dev.bo_cache.idle_lock);
   agx_bo_unreference(&dev, bo2);
   assert(p_atomic_read(&dev.bo_cache.idle_pending) == true);
   mtx_unlock(&dev.bo_cache.idle_lock);

   dev.bo_cache.idle_started = false;
   u_cnd_monotonic_destroy(&dev.bo_cache.idle_cond);
   mtx_destroy(&dev.bo_cache.idle_lock);
   finish_mock_device(&dev);
}

int
main(void)
{
   test_lifo_bucket_reuse();
   test_lifo_avoids_purgeable_willneed();
   test_lru_stale_eviction_drops_oldest();
   test_lru_capacity_eviction_drops_oldest();
   test_idle_pending_wake_protocol();

   printf("agx bo cache: ok\n");
   return 0;
}
