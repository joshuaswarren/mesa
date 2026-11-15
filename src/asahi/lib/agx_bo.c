/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "agx_bo.h"
#include <inttypes.h>
#include <stdlib.h>
#include "util/hash_table.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "agx_device.h"
#include "decode.h"
#include "layout.h"

/* A time-only eviction policy can retain a complete multi-gigabyte resource
 * phase while its replacement is allocated.  Apart from defeating memory
 * budgets, that lets the kernel's cgroup OOM kill arrive before bo_alloc()
 * can fail and trigger agx_bo_cache_evict_all().  Keep enough for ordinary
 * command/resource reuse while making the cache a bounded optimization. */
#define AGX_BO_CACHE_MAX_SIZE (256ull * 1024 * 1024)

/* Helper to calculate the bucket index of a BO */
static unsigned
agx_bucket_index(size_t size)
{
   /* Round down to POT to compute a bucket index */
   unsigned bucket_index = util_logbase2(size);

   /* Clamp to supported buckets. Huge allocations use the largest bucket */
   bucket_index = CLAMP(bucket_index, MIN_BO_CACHE_BUCKET, MAX_BO_CACHE_BUCKET);

   /* Reindex from 0 */
   return (bucket_index - MIN_BO_CACHE_BUCKET);
}

static struct list_head *
agx_bucket(struct agx_device *dev, size_t size, uint32_t flags)
{
   /* Shader, command and resource allocations of the same size often have
    * incompatible flags. Keep their searches apart instead of repeatedly
    * walking every unrelated BO while holding the cache lock. Hash all flag
    * bits; collisions remain safe because fetch checks exact equality.
    */
   static_assert(NR_BO_CACHE_FLAG_BUCKETS == 16, "flag hash width");
   const unsigned flag_bucket = (flags * UINT32_C(0x9e3779b9)) >> 28;
   return &dev->bo_cache.buckets[agx_bucket_index(size) *
                                NR_BO_CACHE_FLAG_BUCKETS + flag_bucket];
}

static void
agx_bo_cache_remove_locked(struct agx_device *dev, struct agx_bo *bo)
{
   simple_mtx_assert_locked(&dev->bo_cache.lock);
   if (dev->bo_cache.next_unadvised == bo->handle) {
      struct list_head *next = bo->lru_link.next;
      dev->bo_cache.next_unadvised = next == &dev->bo_cache.lru ? 0 :
         list_entry(next, struct agx_bo, lru_link)->handle;
   }
   list_del(&bo->bucket_link);
   list_del(&bo->lru_link);
   dev->bo_cache.size -= bo->size;
   if (dev->bo_cache.size_counter)
      p_atomic_add(&dev->bo_cache.size_counter->value, -(int64_t)bo->size);
}

/* Tries to fetch a BO of sufficient size with the appropriate flags from the
 * BO cache. If it succeeds, it returns that BO and removes the BO from the
 * cache. If it fails, it returns NULL signaling the caller to allocate a new
 * BO. */

/* Tell the kernel a cached BO is not needed. Only DRM backends with the
 * ioctl do anything; the first ENOTTY/EINVAL disables it for the device. */
static void
agx_bo_cache_advise_dontneed(struct agx_device *dev, struct agx_bo *bo)
{
   /* Only VM-private objects accept advice; shareable ones are exportable. */
   if (dev->bo_cache.madvise < 0 || !dev->ops.bo_madvise ||
       (bo->flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE)))
      return;

   bool retained;
   int ret = dev->ops.bo_madvise(dev, bo->handle, true, &retained);
   if (ret == -ENOTTY || ret == -EINVAL) {
      dev->bo_cache.madvise = -1;
      return;
   }
   if (ret)
      return;
   dev->bo_cache.madvise = 1;
   bo->purgeable = true;
}

/* Advice costs the kernel work on both ends, roughly proportional to the
 * BO's pages (on A18 Pro a 4 MiB BO's DONTNEED + WILLNEED added 7.5 us to an
 * allocate/free round trip), so advise only BOs that have stayed in the
 * cache for a while; a BO reused at once is never advised. The LRU is in
 * insertion order, so stop at the first BO younger than the threshold. */
#define AGX_BO_CACHE_ADVISE_NS (250ll * 1000 * 1000)

static void
agx_bo_cache_advise_aged(struct agx_device *dev, int64_t now_ns)
{
   if (dev->bo_cache.madvise < 0 || !dev->ops.bo_madvise)
      return;

   const uint32_t handle = dev->bo_cache.next_unadvised;
   if (!handle)
      return;

   struct agx_bo *entry = util_sparse_array_get(&dev->bo_map, handle);
   assert(entry->handle == handle);
   bool advance = true;
   for (struct list_head *link = &entry->lru_link;
        link != &dev->bo_cache.lru; link = link->next) {
      entry = list_entry(link, struct agx_bo, lru_link);
      if (now_ns - entry->cached_ns < AGX_BO_CACHE_ADVISE_NS)
         break;
      if (!entry->purgeable)
         agx_bo_cache_advise_dontneed(dev, entry);

      /* Keep failed advice eligible for retry, but do not rescan a prefix
       * whose objects are already purgeable or never accept advice. */
      const bool complete = entry->purgeable ||
         (entry->flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE)) ||
         dev->bo_cache.madvise < 0;
      if (advance && complete) {
         struct list_head *next = link->next;
         dev->bo_cache.next_unadvised = next == &dev->bo_cache.lru ? 0 :
            list_entry(next, struct agx_bo, lru_link)->handle;
      } else {
         advance = false;
      }
   }
}

/* Reclaim a cached BO; returns false if the kernel purged its contents, in
 * which case the caller must free the BO instead of handing it out. */
static bool
agx_bo_cache_advise_willneed(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo->purgeable)
      return true;

   bo->purgeable = false;
   bool retained = false;
   /* If the kernel did not confirm the object is needed again, it may still
    * be purgeable: never hand it out. */
   if (dev->ops.bo_madvise(dev, bo->handle, false, &retained))
      return false;
   return retained;
}

/* Whether a cached BO can serve this request. Never hand out more than 2x
 * oversized BOs. */
static bool
agx_bo_cache_entry_fits(const struct agx_bo *entry, size_t size, size_t align,
                        uint32_t flags)
{
   return entry->size >= size && entry->flags == flags &&
          entry->size <= 2 * size && align <= entry->align;
}

struct agx_bo *
agx_bo_cache_fetch(struct agx_device *dev, size_t size, size_t align,
                   uint32_t flags, const bool dontwait)
{
   struct list_head purged;
   list_inithead(&purged);

   simple_mtx_lock(&dev->bo_cache.lock);
   struct list_head *bucket = agx_bucket(dev, size, flags);
   struct agx_bo *bo = NULL;

   /* Iterate the bucket looking for something suitable */
   list_for_each_entry_safe(struct agx_bo, entry, bucket, bucket_link) {
      if (!agx_bo_cache_entry_fits(entry, size, align, flags))
         continue;

      agx_bo_cache_remove_locked(dev, entry);

      /* The kernel may have discarded a purgeable BO's contents under
       * memory pressure. Such a BO is dead: free it below and keep looking. */
      if (!agx_bo_cache_advise_willneed(dev, entry)) {
         list_addtail(&entry->bucket_link, &purged);
         continue;
      }

      /* This one works, use it */
      bo = entry;
      break;
   }
   simple_mtx_unlock(&dev->bo_cache.lock);

   if (!list_is_empty(&purged)) {
      /* Match the unreference lock order: bo_map_lock, then the cache. */
      simple_mtx_lock(&dev->bo_map_lock);
      list_for_each_entry_safe(struct agx_bo, entry, &purged, bucket_link) {
         list_del(&entry->bucket_link);
         agx_bo_free(dev, entry);
      }
      simple_mtx_unlock(&dev->bo_map_lock);
   }

   return bo;
}

/* Whether agx_bo_create() would be served from the cache, without taking the
 * BO. Advisory: another thread can take it first. Mirrors the size and
 * alignment rounding of agx_bo_create(). */
bool
agx_bo_cache_has(struct agx_device *dev, size_t size, size_t align,
                 uint32_t flags)
{
   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);

   simple_mtx_lock(&dev->bo_cache.lock);
   bool found = false;
   list_for_each_entry(struct agx_bo, entry, agx_bucket(dev, size, flags),
                       bucket_link) {
      if (agx_bo_cache_entry_fits(entry, size, align, flags)) {
         found = true;
         break;
      }
   }
   simple_mtx_unlock(&dev->bo_cache.lock);
   return found;
}

static void
agx_bo_cache_evict_stale_bos(struct agx_device *dev, int64_t now_sec)
{
   list_for_each_entry_safe(struct agx_bo, entry, &dev->bo_cache.lru,
                            lru_link) {
      /* We want all entries that have been used more than 1 sec ago to be
       * dropped, others can be kept.  Note the <= 2 check and not <= 1. It's
       * here to account for the fact that we're only testing ->tv_sec, not
       * ->tv_nsec.  That means we might keep entries that are between 1 and 2
       * seconds old, but we don't really care, as long as unused BOs are
       * dropped at some point.
       */
      if (now_sec - entry->last_used <= 2)
         break;

      agx_bo_cache_remove_locked(dev, entry);
      agx_bo_free(dev, entry);
   }
}

static int
agx_bo_cache_idle_trim(void *data)
{
   struct agx_device *dev = data;
   u_thread_setname("agx-bo-cache");

   mtx_lock(&dev->bo_cache.idle_lock);
   while (!dev->bo_cache.idle_stop) {
      while (!dev->bo_cache.idle_pending && !dev->bo_cache.idle_stop)
         u_cnd_monotonic_wait(&dev->bo_cache.idle_cond,
                              &dev->bo_cache.idle_lock);
      if (dev->bo_cache.idle_stop)
         break;

      struct timespec deadline;
      timespec_from_nsec(&deadline, os_time_get_nano() + INT64_C(1000000000));
      /* Insertion wakes an empty cache, but must not postpone eviction of
       * older entries or turn a busy cache into a polling loop. */
      while (!dev->bo_cache.idle_stop &&
             u_cnd_monotonic_timedwait(&dev->bo_cache.idle_cond,
                                       &dev->bo_cache.idle_lock,
                                       &deadline) == thrd_success) {}
      if (dev->bo_cache.idle_stop)
         break;

      dev->bo_cache.idle_pending = false;
      mtx_unlock(&dev->bo_cache.idle_lock);

      /* Match unreference/import lock order. Freeing also updates bo_map and
       * VMA heaps; do not hold idle_lock while acquiring either BO lock. */
      simple_mtx_lock(&dev->bo_map_lock);
      simple_mtx_lock(&dev->bo_cache.lock);
      agx_bo_cache_evict_stale_bos(dev,
                                  os_time_get_nano() / INT64_C(1000000000));
      bool pending = !list_is_empty(&dev->bo_cache.lru);
      simple_mtx_unlock(&dev->bo_cache.lock);
      simple_mtx_unlock(&dev->bo_map_lock);

      mtx_lock(&dev->bo_cache.idle_lock);
      /* Preserve any insertion which occurred after we cleared pending. */
      dev->bo_cache.idle_pending |= pending;
   }
   mtx_unlock(&dev->bo_cache.idle_lock);
   return 0;
}

void
agx_bo_cache_start_idle_trim(struct agx_device *dev)
{
#ifdef HAVE_LIBDRM
   if (dev->ops.bo_free != agx_drm_bo_close && !dev->is_virtio)
      return;
   if (mtx_init(&dev->bo_cache.idle_lock, mtx_plain) != thrd_success)
      return;
   if (u_cnd_monotonic_init(&dev->bo_cache.idle_cond) != thrd_success) {
      mtx_destroy(&dev->bo_cache.idle_lock);
      return;
   }
   dev->bo_cache.idle_pending = true;
   if (u_thread_create(&dev->bo_cache.idle_thread, agx_bo_cache_idle_trim,
                      dev) != thrd_success) {
      u_cnd_monotonic_destroy(&dev->bo_cache.idle_cond);
      mtx_destroy(&dev->bo_cache.idle_lock);
      return;
   }
   dev->bo_cache.idle_started = true;
#endif
}

void
agx_bo_cache_stop_idle_trim(struct agx_device *dev)
{
   if (!dev->bo_cache.idle_started)
      return;
   mtx_lock(&dev->bo_cache.idle_lock);
   dev->bo_cache.idle_stop = true;
   u_cnd_monotonic_signal(&dev->bo_cache.idle_cond);
   mtx_unlock(&dev->bo_cache.idle_lock);
   thrd_join(dev->bo_cache.idle_thread, NULL);
   dev->bo_cache.idle_started = false;
   u_cnd_monotonic_destroy(&dev->bo_cache.idle_cond);
   mtx_destroy(&dev->bo_cache.idle_lock);
}

static void
agx_bo_cache_put_locked(struct agx_device *dev, struct agx_bo *bo)
{
   struct list_head *bucket = agx_bucket(dev, bo->size, bo->flags);
   /* Add us to the bucket */
   list_addtail(&bo->bucket_link, bucket);

   /* An empty LRU initializes the cursor, including on its first insertion. */
   if (list_is_empty(&dev->bo_cache.lru) || !dev->bo_cache.next_unadvised)
      dev->bo_cache.next_unadvised = bo->handle;

   /* Add us to the LRU list and update the last_used field. */
   list_addtail(&bo->lru_link, &dev->bo_cache.lru);
   bo->last_used = os_time_get_nano() / INT64_C(1000000000);

   /* Update statistics */
   dev->bo_cache.size += bo->size;
   if (dev->bo_cache.size_counter)
      p_atomic_add(&dev->bo_cache.size_counter->value, bo->size);

   if (0) {
      printf("BO cache: %zu KiB (+%zu KiB from %s, hit/miss %" PRIu64
             "/%" PRIu64 ")\n",
             DIV_ROUND_UP(dev->bo_cache.size, 1024),
             DIV_ROUND_UP(bo->size, 1024), bo->label,
             p_atomic_read(&dev->bo_cache.hits),
             p_atomic_read(&dev->bo_cache.misses));
   }


   /* Update label for debug */
   bo->label = "Unused (BO cache)";

   /* Let the kernel discard BOs that stay cached under memory pressure;
    * nothing references them until agx_bo_cache_fetch() hands them out. */
   bo->cached_ns = os_time_get_nano();
   agx_bo_cache_advise_aged(dev, bo->cached_ns);

   /* Let's do some cleanup in the BO cache while we hold the lock. */
   agx_bo_cache_evict_stale_bos(dev, bo->last_used);

   if (dev->bo_cache.idle_started) {
      mtx_lock(&dev->bo_cache.idle_lock);
      if (!dev->bo_cache.idle_pending) {
         dev->bo_cache.idle_pending = true;
         u_cnd_monotonic_signal(&dev->bo_cache.idle_cond);
      }
      mtx_unlock(&dev->bo_cache.idle_lock);
   }
}

/* Tries to add a BO to the cache. Returns if it was successful */
static bool
agx_bo_cache_put(struct agx_device *dev, struct agx_bo *bo)
{
   if (bo->flags & (AGX_BO_SHARED | AGX_BO_NO_CACHE)) {
      return false;
   } else {
      simple_mtx_lock(&dev->bo_cache.lock);

      if (bo->size > AGX_BO_CACHE_MAX_SIZE) {
         simple_mtx_unlock(&dev->bo_cache.lock);
         return false;
      }

      /* Prefer the newly freed BO to older cache entries. Refusing it when
       * the cache is full makes a repeated large allocation miss forever:
       * even one small cached BO prevents a 256 MiB BO from being retained.
       * Evict only zero-reference BOs, under the same locks as stale eviction,
       * and keep the existing total-byte limit. */
      while (dev->bo_cache.size > AGX_BO_CACHE_MAX_SIZE - bo->size) {
         struct agx_bo *entry =
            list_first_entry(&dev->bo_cache.lru, struct agx_bo, lru_link);
         agx_bo_cache_remove_locked(dev, entry);
         agx_bo_free(dev, entry);
      }

      agx_bo_cache_put_locked(dev, bo);
      simple_mtx_unlock(&dev->bo_cache.lock);

      return true;
   }
}

void
agx_bo_cache_evict_all(struct agx_device *dev)
{
   simple_mtx_lock(&dev->bo_cache.lock);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i) {
      struct list_head *bucket = &dev->bo_cache.buckets[i];

      list_for_each_entry_safe(struct agx_bo, entry, bucket, bucket_link) {
         agx_bo_cache_remove_locked(dev, entry);
         agx_bo_free(dev, entry);
      }
   }
   simple_mtx_unlock(&dev->bo_cache.lock);
}

void
agx_bo_reference(struct agx_bo *bo)
{
   if (bo) {
      ASSERTED int count = p_atomic_inc_return(&bo->refcnt);
      assert(count != 1);
   }
}

struct label_stat {
   const char *label;
   uint32_t count;
   size_t alloc_B;
   size_t mapped_B;
};

static void
account_bo(struct label_stat *stat, struct agx_bo *bo)
{
   stat->count++;
   stat->alloc_B += bo->size;

   if (bo->_map != NULL)
      stat->mapped_B += bo->size;
}

static void
print_size(FILE *fp, size_t size_B)
{
   if (size_B >= (1024 * 1024 * 1024)) {
      fprintf(fp, "%.1f GiB", (double)size_B / (double)(1024 * 1024 * 1024));
   } else if (size_B >= (1024 * 1024)) {
      fprintf(fp, "%.1f MiB", (double)size_B / (double)(1024 * 1024));
   } else if (size_B >= 1024) {
      fprintf(fp, "%zu KiB", DIV_ROUND_UP(size_B, 1024));
   } else {
      fprintf(fp, "%zu B", size_B);
   }
}

static void
print_stat(FILE *fp, struct label_stat *stat)
{
   const char *BOLD = "\033[1m";
   const char *RESET = "\033[0m";

   fprintf(fp, "%s%s%s: ", BOLD, stat->label, RESET);
   print_size(fp, stat->alloc_B);

   if (stat->mapped_B) {
      fprintf(fp, ", mapped ");
      print_size(fp, stat->mapped_B);
   }

   fprintf(fp, ", %u BOs\n", stat->count);
}

static int
compare_size(const void *a_, const void *b_)
{
   const struct label_stat *const *label_a = a_;
   const struct label_stat *const *label_b = b_;

   size_t a = (*label_a)->alloc_B;
   size_t b = (*label_b)->alloc_B;

   return (a > b) ? 1 : (a < b) ? -1 : 0;
}

static void
agx_bo_dump_all(struct agx_device *dev)
{
   struct label_stat accum = {.label = "Total"};
   struct hash_table *totals = _mesa_string_hash_table_create(NULL);
   bool verbose = dev->debug & AGX_DBG_BODUMPVERBOSE;

   if (verbose)
      fprintf(stderr, "---\n");

   for (uint32_t handle = 0; handle < dev->max_handle; handle++) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->size)
         continue;

      if (verbose) {
         fprintf(stderr,
                 "%u: %s %zu KiB va=%#llx span=%#llx flags=%#x heap=%s\n",
                 handle, bo->label, bo->size / 1024,
                 (unsigned long long)(bo->va ? bo->va->addr : 0),
                 (unsigned long long)(bo->va ? bo->va->size_B : 0), bo->flags,
                 bo->va && (bo->va->flags & AGX_VA_USC) ? "usc" : "main");
      }

      account_bo(&accum, bo);

      assert(bo->label != NULL && "Everything must be labeled");

      struct hash_entry *ent = _mesa_hash_table_search(totals, bo->label);
      struct label_stat *ls;
      if (ent != NULL) {
         ls = ent->data;
      } else {
         ls = rzalloc(totals, struct label_stat);
         ls->label = bo->label;
         _mesa_hash_table_insert(totals, bo->label, ls);
      }

      account_bo(ls, bo);
   }

   if (verbose) {
      fprintf(stderr, "\n");
   }

   unsigned nr_labels = _mesa_hash_table_num_entries(totals);
   struct label_stat **stats =
      rzalloc_array(totals, struct label_stat *, nr_labels);
   unsigned i = 0;

   hash_table_foreach(totals, ent) {
      assert(i < nr_labels);
      stats[i++] = ent->data;
   }

   assert(i == nr_labels);

   /* Sort labels in ascending order of size */
   qsort(stats, nr_labels, sizeof(struct label_stat *), compare_size);

   for (i = 0; i < nr_labels; ++i) {
      print_stat(stderr, stats[i]);
   }

   print_stat(stderr, &accum);

   if (verbose)
      fprintf(stderr, "---\n\n");
   else
      fprintf(stderr, "\n");

   ralloc_free(totals);
}

static void
agx_bo_dump_all_periodic(struct agx_device *dev)
{
   if (likely(!(dev->debug & (AGX_DBG_BODUMP | AGX_DBG_BODUMPVERBOSE))))
      return;

   static time_t agx_last_dumped_time = 0;

   time_t now = time(NULL);
   if (now == agx_last_dumped_time)
      return;

   agx_bo_dump_all(dev);
   agx_last_dumped_time = now;
}

void
agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;

   int refcnt = p_atomic_dec_return(&bo->refcnt);
   assert(refcnt >= 0);

   /* Don't return to cache if there are still references */
   if (refcnt)
      return;



   simple_mtx_lock(&dev->bo_map_lock);

   /* Someone might have imported this BO while we were waiting for the
    * lock, let's make sure it's still not referenced before freeing it.
    */
   if (p_atomic_read(&bo->refcnt) == 0) {
      assert(!p_atomic_read_relaxed(&bo->writer));

      if (dev->debug &
          (AGX_DBG_TRACE | AGX_DBG_ASYNCTRACE))
         agxdecode_track_free(dev->agxdecode, bo);

      if (!agx_bo_cache_put(dev, bo))
         agx_bo_free(dev, bo);
   }

   agx_bo_dump_all_periodic(dev);

   simple_mtx_unlock(&dev->bo_map_lock);
}

struct agx_bo *
agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
              enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo;
   bool reused = false;
   assert(size > 0);

   /* BOs are allocated in pages */
   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);

   /* See if we have a BO already in the cache */
   bo = agx_bo_cache_fetch(dev, size, align, flags, true);
   reused = bo != NULL;

   /* Update stats based on the first attempt to fetch */
   if (bo != NULL)
      p_atomic_inc(&dev->bo_cache.hits);
   else
      p_atomic_inc(&dev->bo_cache.misses);

   /* Otherwise, allocate a fresh BO. If allocation fails, we can try waiting
    * for something in the cache. But if there's no nothing suitable, we should
    * flush the cache to make space for the new allocation.
    */
   if (!bo)
      bo = dev->ops.bo_alloc(dev, size, align, flags);
   if (!bo) {
      bo = agx_bo_cache_fetch(dev, size, align, flags, false);
      reused = bo != NULL;
   }
   if (!bo) {
      agx_bo_cache_evict_all(dev);
      bo = dev->ops.bo_alloc(dev, size, align, flags);
   }

   if (!bo) {
      /* Three unrelated failures reach here -- GEM_CREATE, the userspace VA
       * heap, and a kernel VM_BIND rejection -- and every caller turns all
       * three into one VK_ERROR_OUT_OF_DEVICE_MEMORY. Print what is left in
       * each heap so a hardware log says which one it was without having to
       * reproduce it again. */
      fprintf(stderr, "BO creation failed: size=%#zx align=%#x flags=%#x "
                      "label=%s", size, align, flags,
              label ? label : "(none)");

      /* A winsys with its own VA allocator does not use these heaps, and
       * printing their untouched sizes would name the wrong resource. */
      {
         simple_mtx_lock(&dev->vma_lock);
         const uint64_t usc_free = dev->usc_heap.free_size;
         const uint64_t main_free = dev->main_heap.free_size;
         simple_mtx_unlock(&dev->vma_lock);

         fprintf(stderr,
                 "; free USC window %#" PRIx64 " of %#" PRIx64
                 ", free user heap %#" PRIx64 " of %#" PRIx64,
                 usc_free, AGX_USC_WINDOW_SIZE, main_free,
                 dev->main_heap_size);
      }

      fputc('\n', stderr);
      return NULL;
   }

   /* A cache fetch owns this unreferenced, unlinked BO exclusively. Clear it
    * before publishing the new reference; a failed map must neither expose
    * stale contents nor put an unmappable object back into the cache. */
   if (reused && (flags & AGX_BO_ZERO_ON_REUSE)) {
      void *map = agx_bo_map(bo);
      if (!map) {
         simple_mtx_lock(&dev->bo_map_lock);
         agx_bo_free(dev, bo);
         simple_mtx_unlock(&dev->bo_map_lock);
         return NULL;
      }
      memset(map, 0, bo->size);
   }

   bo->label = label;
   p_atomic_set(&bo->refcnt, 1);

   if (dev->debug &
       (AGX_DBG_TRACE | AGX_DBG_ASYNCTRACE)) {
      agx_bo_map(bo);
      agxdecode_track_alloc(dev->agxdecode, bo);
   }

   agx_bo_dump_all_periodic(dev);
   return bo;
}
