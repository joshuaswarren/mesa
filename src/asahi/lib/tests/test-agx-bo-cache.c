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
   unsigned alloc_calls;
   unsigned bind_calls;
   struct drm_asahi_gem_bind_op last_bind;
   bool bind_fail;
   bool expect_bo_map_locked;
   bool madvise_retain;
   unsigned dontneed_calls;
   unsigned willneed_calls;
   uint32_t last_dontneed_handle;
   uint32_t last_willneed_handle;
   unsigned mmap_offset_queries;
   unsigned mmap_calls;
   unsigned munmap_calls;
   bool alloc_va;
};

static struct mock_state mock;

static struct agx_bo *
mock_bo_alloc(struct agx_device *dev, size_t size, size_t align,
              enum agx_bo_flags flags)
{
   mock.alloc_calls++;
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
   if (mock.alloc_va) {
      bo->va = agx_va_alloc(dev, size, align, 0, 0);
      assert(bo->va != NULL);
   }
   return bo;
}

static int
mock_bo_bind(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
             uint32_t count)
{
   assert(count >= 1);
   mock.bind_calls += count;
   mock.last_bind = ops[count - 1];
   return mock.bind_fail ? -1 : 0;
}

static void
mock_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
             size_t length)
{
   assert(bo->_map == NULL);
   mock.mmap_calls++;
   if (!bo->has_mmap_offset) {
      mock.mmap_offset_queries++;
      bo->mmap_offset = UINT64_C(0x100000000) + (uint64_t)bo->handle * 0x10000ull;
      bo->has_mmap_offset = true;
   }
   bo->_map_size = length ?: bo->size;
   bo->_map = fixed_addr ? fixed_addr : (void *)(uintptr_t)(0x20000000ull + bo->handle * 0x10000ull);
}

static void
mock_bo_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   mock.munmap_calls++;
   bo->_map_size = 0;
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

   dev->guard_size = AIL_PAGESIZE;
   dev->sparse_ro_offset = UINT64_C(1) << 36;
   util_vma_heap_init(&dev->main_heap, UINT64_C(0x2000000000),
                      UINT64_C(1) << 36);

   dev->ops.bo_alloc = mock_bo_alloc;
   dev->ops.bo_bind = mock_bo_bind;
   dev->ops.bo_mmap = mock_bo_mmap;
   dev->ops.bo_munmap = mock_bo_munmap;
   dev->ops.bo_mmap_independent = true;
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

   util_vma_heap_finish(&dev->main_heap);
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

static bool
deny_alloc(size_t size, void *data)
{
   unsigned *calls = data;
   (*calls)++;
   return false;
}

static void
test_ro_shadow_cache_reuse(void)
{
   struct agx_device dev;
   init_mock_device(&dev);
   mock.alloc_va = true;

   const size_t sz = 64 * 1024;
   struct agx_bo *bo = agx_bo_create(&dev, sz, 64 * 1024, 0, "app");
   assert(bo && !bo->ro_shadow_bound);

   /* First bind maps the read-only shadow once and records the state. */
   assert(agx_bo_ensure_ro_shadow(&dev, bo) == 0);
   assert(bo->ro_shadow_bound);
   assert(mock.bind_calls == 1);
   assert(mock.last_bind.addr == agx_rw_addr_to_ro(&dev, bo->va->addr));
   assert(mock.last_bind.range == bo->size);
   assert(mock.last_bind.offset == 0);
   assert(mock.last_bind.handle == bo->uapi_handle);
   assert(mock.last_bind.flags == DRM_ASAHI_BIND_READ);

   /* Repeated ensure on the same BO skips VM_BIND. */
   assert(agx_bo_ensure_ro_shadow(&dev, bo) == 0);
   assert(mock.bind_calls == 1);

   /* Returning to cache and reusing preserves the RO shadow with zero binds. */
   const uint32_t cached_handle = bo->handle;
   agx_bo_unreference(&dev, bo);
   struct agx_bo *reused = agx_bo_create(&dev, sz, 64 * 1024, 0, "app2");
   assert(reused && reused->handle == cached_handle);
   assert(reused->ro_shadow_bound);
   assert(agx_bo_ensure_ro_shadow(&dev, reused) == 0);
   assert(mock.bind_calls == 1);

   /* Evicting and freeing clears the slot so a fresh BO binds again. */
   agx_bo_unreference(&dev, reused);
   agx_bo_cache_evict_all(&dev);
   assert(mock.freed_count == 1 && mock.freed_handles[0] == cached_handle);

   /* A failed shadow bind must leave ro_shadow_bound false so a retry binds. */
   struct agx_bo *fresh = agx_bo_create(&dev, sz, 64 * 1024, 0, "fresh");
   assert(fresh && !fresh->ro_shadow_bound);
   mock.bind_fail = true;
   assert(agx_bo_ensure_ro_shadow(&dev, fresh) != 0);
   assert(!fresh->ro_shadow_bound);
   assert(mock.bind_calls == 2);
   mock.bind_fail = false;
   assert(agx_bo_ensure_ro_shadow(&dev, fresh) == 0);
   assert(fresh->ro_shadow_bound);
   assert(mock.bind_calls == 3);
   agx_bo_unreference(&dev, fresh);

   /* Shareable/shared BOs take bo_map_lock cleanly and still bind once. */
   struct agx_bo *shared =
      agx_bo_create(&dev, sz, 64 * 1024, AGX_BO_SHAREABLE, "shared");
   assert(shared && !shared->ro_shadow_bound);
   assert(agx_bo_ensure_ro_shadow(&dev, shared) == 0);
   assert(shared->ro_shadow_bound);
   assert(mock.bind_calls == 4);
   assert(agx_bo_ensure_ro_shadow(&dev, shared) == 0);
   assert(mock.bind_calls == 4);
   agx_bo_unreference(&dev, shared);

   finish_mock_device(&dev);
}

static void
test_create_checked_admission_and_purged_miss(void)
{
   struct agx_device dev;
   init_mock_device(&dev);

   struct agx_bo *bo = agx_bo_create(&dev, 64 * MIB, 64 * 1024, 0, "64m");
   assert(bo);
   const uint32_t h = bo->handle;
   agx_bo_unreference(&dev, bo);

   /* Cache hit must succeed without invoking the admission callback. */
   unsigned check_calls = 0;
   unsigned allocs_before = mock.alloc_calls;
   struct agx_bo *hit = agx_bo_create_checked(&dev, 64 * MIB, 64 * 1024, 0,
                                              "hit", deny_alloc, &check_calls);
   assert(hit && hit->handle == h);
   assert(check_calls == 0);
   assert(mock.alloc_calls == allocs_before);

   /* Cache miss must invoke admission check and stop before bo_alloc(). */
   struct agx_bo *miss = agx_bo_create_checked(&dev, 16 * MIB, 64 * 1024, 0,
                                               "miss", deny_alloc, &check_calls);
   assert(miss == NULL);
   assert(check_calls == 1);
   assert(mock.alloc_calls == allocs_before);

   /* A kernel-purged cached BO must be freed and still consult admission. */
   agx_bo_unreference(&dev, hit);
   simple_mtx_lock(&dev.bo_cache.lock);
   hit->cached_ns -= INT64_C(500) * 1000 * 1000;
   simple_mtx_unlock(&dev.bo_cache.lock);
   struct agx_bo *poke = agx_bo_create(&dev, AIL_PAGESIZE, AIL_PAGESIZE, 0, "poke");
   agx_bo_unreference(&dev, poke);
   assert(hit->purgeable);

   mock.madvise_retain = false;
   check_calls = 0;
   allocs_before = mock.alloc_calls;
   struct agx_bo *purged_miss =
      agx_bo_create_checked(&dev, 64 * MIB, 64 * 1024, 0, "purged",
                            deny_alloc, &check_calls);
   assert(purged_miss == NULL);
   assert(check_calls == 1);
   assert(mock.alloc_calls == allocs_before);
   assert(mock.freed_count == 1 && mock.freed_handles[0] == h);

   finish_mock_device(&dev);
}

static void
test_large_buckets_flag_isolation_and_cross_pot_reuse(void)
{
   struct agx_device dev;
   init_mock_device(&dev);

   /* Cache 16 MiB, 64 MiB, 128 MiB (flags=0) and 16 MiB (AGX_BO_WRITEBACK).
    * Total = 224 MiB <= 256 MiB cap -> zero evictions. */
   struct agx_bo *b16 = agx_bo_create(&dev, 16 * MIB, 64 * 1024, 0, "16m");
   struct agx_bo *b64 = agx_bo_create(&dev, 64 * MIB, 64 * 1024, 0, "64m");
   struct agx_bo *b128 = agx_bo_create(&dev, 128 * MIB, 64 * 1024, 0, "128m");
   struct agx_bo *b16_wb =
      agx_bo_create(&dev, 16 * MIB, 64 * 1024, AGX_BO_WRITEBACK, "16m-wb");
   const uint32_t h16 = b16->handle, h64 = b64->handle, h128 = b128->handle;
   const uint32_t h16_wb = b16_wb->handle;

   agx_bo_unreference(&dev, b16);
   agx_bo_unreference(&dev, b64);
   agx_bo_unreference(&dev, b128);
   agx_bo_unreference(&dev, b16_wb);
   assert(dev.bo_cache.size == 224 * MIB);
   assert(mock.freed_count == 0);

   /* Flag isolation + dedicated large buckets hit exact BOs. */
   struct agx_bo *g16_wb =
      agx_bo_create(&dev, 16 * MIB, 64 * 1024, AGX_BO_WRITEBACK, "g16-wb");
   struct agx_bo *g16 = agx_bo_create(&dev, 16 * MIB, 64 * 1024, 0, "g16");
   struct agx_bo *g64 = agx_bo_create(&dev, 64 * MIB, 64 * 1024, 0, "g64");
   struct agx_bo *g128 = agx_bo_create(&dev, 128 * MIB, 64 * 1024, 0, "g128");
   assert(g16_wb->handle == h16_wb);
   assert(g16->handle == h16);
   assert(g64->handle == h64);
   assert(g128->handle == h128);
   assert(dev.bo_cache.size == 0);

   agx_bo_unreference(&dev, g16_wb);
   agx_bo_unreference(&dev, g16);
   agx_bo_unreference(&dev, g64);
   agx_bo_unreference(&dev, g128);
   agx_bo_cache_evict_all(&dev);
   assert(dev.bo_cache.size == 0);

   /* Cross-POT <= 2x reuse: 6 MiB reuses cached 8 MiB, while 3 MiB misses. */
   struct agx_bo *b8 = agx_bo_create(&dev, 8 * MIB, 64 * 1024, 0, "8m");
   const uint32_t h8 = b8->handle;
   agx_bo_unreference(&dev, b8);
   assert(!agx_bo_cache_has(&dev, 3 * MIB, 64 * 1024, 0));
   assert(agx_bo_cache_has(&dev, 6 * MIB, 64 * 1024, 0));
   struct agx_bo *g6 = agx_bo_create(&dev, 6 * MIB, 64 * 1024, 0, "6m");
   assert(g6->handle == h8);
   agx_bo_unreference(&dev, g6);
   agx_bo_cache_evict_all(&dev);

   /* Uncacheable flags and > 256 MiB allocations bypass cache completely. */
   const unsigned freed_before = mock.freed_count;
   struct agx_bo *nc =
      agx_bo_create(&dev, 64 * 1024, 64 * 1024, AGX_BO_NO_CACHE, "nocache");
   struct agx_bo *sh =
      agx_bo_create(&dev, 64 * 1024, 64 * 1024, AGX_BO_SHARED, "shared");
   struct agx_bo *huge =
      agx_bo_create(&dev, 300 * MIB, 64 * 1024, 0, "huge");
   agx_bo_unreference(&dev, nc);
   agx_bo_unreference(&dev, sh);
   agx_bo_unreference(&dev, huge);
   assert(dev.bo_cache.size == 0);
   assert(mock.freed_count == freed_before + 3);
   assert(!agx_bo_cache_has(&dev, 64 * 1024, 64 * 1024, AGX_BO_NO_CACHE));
   assert(!agx_bo_cache_has(&dev, 300 * MIB, 64 * 1024, 0));

   finish_mock_device(&dev);
}

static void
test_mmap_offset_cache_propagation(void)
{
   struct agx_device dev;
   init_mock_device(&dev);

   struct agx_bo *bo = agx_bo_create(&dev, 64 * 1024, 64 * 1024, 0, "map");
   assert(bo && !bo->has_mmap_offset);

   /* First agx_bo_map() queries the mmap offset once and caches it on bo. */
   void *p1 = agx_bo_map(bo);
   assert(p1 != NULL);
   assert(bo->has_mmap_offset && bo->mmap_offset != 0);
   assert(mock.mmap_offset_queries == 1);
   assert(mock.mmap_calls == 1);

   /* Simulate vkUnmapMemory + cache round trip: remap skips offset query. */
   bo->_map = NULL;
   bo->_map_size = 0;
   const uint32_t h = bo->handle;
   agx_bo_unreference(&dev, bo);
   struct agx_bo *reused = agx_bo_create(&dev, 64 * 1024, 64 * 1024, 0, "map2");
   assert(reused && reused->handle == h && reused->has_mmap_offset);

   void *p2 = agx_bo_map(reused);
   assert(p2 != NULL);
   assert(mock.mmap_offset_queries == 1);
   assert(mock.mmap_calls == 2);

   /* Owned mapping also reuses the cached offset without another query. */
   size_t owned_size = 0;
   void *p3 = agx_bo_map_owned(reused, NULL, 0, &owned_size);
   assert(p3 != NULL && owned_size == reused->size);
   assert(mock.mmap_offset_queries == 1);
   assert(mock.mmap_calls == 3);
   agx_bo_unmap_owned(reused, p3, owned_size);
   assert(mock.munmap_calls == 1);

   agx_bo_unreference(&dev, reused);
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
   test_ro_shadow_cache_reuse();
   test_create_checked_admission_and_purged_miss();
   test_large_buckets_flag_isolation_and_cross_pot_reuse();
   test_mmap_offset_cache_propagation();

   printf("agx bo cache: ok\n");
   return 0;
}
