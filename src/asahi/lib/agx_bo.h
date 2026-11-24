/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "util/list.h"

struct agx_device;

enum agx_bo_flags {
   /* BO is shared across processes (imported or exported) and therefore cannot
    * be cached locally
    */
   AGX_BO_SHARED = 1 << 0,

   /* BO must be allocated in the low 32-bits of VA space */
   AGX_BO_LOW_VA = 1 << 1,

   /* BO is executable */
   AGX_BO_EXEC = 1 << 2,

   /* BO should be mapped write-back on the CPU (else, write combine) */
   AGX_BO_WRITEBACK = 1 << 3,

   /* BO could potentially be shared (imported or exported) and therefore cannot
    * be allocated as private
    */
   AGX_BO_SHAREABLE = 1 << 4,

   /* BO is read-only from the GPU side
    */
   AGX_BO_READONLY = 1 << 5,

   /* Fresh kernel allocations are already zeroed. If a BO is recycled from
    * the userspace cache, clear it before returning it to the caller. */
   AGX_BO_ZERO_ON_REUSE = 1 << 6,

   /* Short-lived backing whose release must immediately close the GEM handle,
    * including objects which fail before their first successful export. */
   AGX_BO_NO_CACHE = 1 << 7,

   /* BO over memory the application owns (DRM_IOCTL_ASAHI_GEM_USERPTR): _map
    * is the application's pointer, which is never unmapped or moved, and the
    * BO is never cached. Implies AGX_BO_NO_CACHE. */
   AGX_BO_HOST_PTR = 1 << 8,
};

enum agx_va_flags {
   /* VA must be inside the USC region, otherwise unrestricted. */
   AGX_VA_USC = (1 << 0),

   /* VA must be fixed, otherwise allocated by the driver. */
   AGX_VA_FIXED = (1 << 1),

   /* Ordinary allocations never occupy the capture/replay address region. */
   AGX_VA_CAPTURE_REPLAY = (1 << 2),

   /* Take the lowest free address instead of the heap's default end. */
   AGX_VA_LOW = (1 << 3),
};

struct agx_va {
   enum agx_va_flags flags;
   uint64_t addr;
   uint64_t size_B;
};

struct agx_ptr {
   /* If CPU mapped, CPU address. NULL if not mapped */
   void *cpu;

   /* Mapped GPU address */
   uint64_t gpu;
};

struct agx_bo {
   /* Must be first for casting */
   struct list_head bucket_link;

   /* Used to link the BO to the BO cache LRU list. */
   struct list_head lru_link;

   /* Convenience */
   struct agx_device *dev;

   /* The time this BO was used last, so we can evict stale BOs. */
   time_t last_used;

   /* When the BO entered the cache, and whether the kernel was since told
    * it is not needed (GEM_MADVISE). */
   int64_t cached_ns;
   bool purgeable;

   /* True once the read-only sparse-resident shadow mapping at
    * agx_rw_addr_to_ro(dev, va->addr) has been bound in the GPU VM. The
    * mapping lives with the GEM handle and its VA across BO-cache reuse;
    * cleared when agx_bo_free() zeroes the slot. */
   bool ro_shadow_bound;

   /* True once mmap_offset holds the kernel's immutable GEM mmap offset. */
   bool has_mmap_offset;

   /* Creation attributes */
   enum agx_bo_flags flags;
   size_t size;
   size_t align;

   /* Mapping */
   struct agx_va *va;

   /* Cached DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET result for this GEM handle. */
   uint64_t mmap_offset;

   /* Gallium's compact G17P command address. Protected by vma_lock and
    * retained through BO-cache reuse; released with the final GEM handle. */
   uint64_t g17p_vdm_alias;

   /* Suffixed to force agx_bo_map access */
   void *_map;

   /* Bytes actually mapped at `_map`, which is NOT always `size`.  A placed
    * map (VK_EXT_map_memory_placed) covers only the VkDeviceMemory's
    * allocationSize, while the BO behind it comes from the cache and is
    * routinely larger; unmapping `size` would reach into address space the
    * application owns.  Zero means "the whole BO", the ordinary non-placed
    * case. */
   size_t _map_size;

   /* Process-local index */
   uint32_t handle;

   /* Handle to refer to this BO in uAPI calls. This is either the GEM handle
    * on native Linux, or the virtio resource ID with virtgpu.
    */
   uint32_t uapi_handle;

   /* DMA-BUF fd clone for adding fences to imports/exports */
   int prime_fd;

   /* Current writer, if any (queue in upper 32 bits, syncobj in lower 32 bits) */
   uint64_t writer;

   /* Update atomically */
   int32_t refcnt;

   /* For debugging */
   const char *label;
};

static inline uint32_t
agx_bo_writer_syncobj(uint64_t writer)
{
   return writer;
}

static inline uint32_t
agx_bo_writer_queue(uint64_t writer)
{
   return writer >> 32;
}

static inline uint64_t
agx_bo_writer(uint32_t queue, uint32_t syncobj)
{
   return (((uint64_t)queue) << 32) | syncobj;
}

typedef bool (*agx_bo_alloc_check_fn)(size_t size, void *data);

struct agx_bo *agx_bo_create_checked(struct agx_device *dev, size_t size,
                                     unsigned align, enum agx_bo_flags flags,
                                     const char *label,
                                     agx_bo_alloc_check_fn can_alloc,
                                     void *can_alloc_data);

struct agx_bo *agx_bo_create(struct agx_device *dev, size_t size,
                             unsigned align, enum agx_bo_flags flags,
                             const char *label);

int agx_bo_ensure_ro_shadow(struct agx_device *dev, struct agx_bo *bo);

void agx_bo_reference(struct agx_bo *bo);
void agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo);

struct agx_bo *agx_bo_import(struct agx_device *dev, int fd);
/* Which application memory agx_bo_import_host_ptr() takes. */
enum agx_host_ptr_access {
   /* Memory the kernel can pin for writing, bound read-write. */
   AGX_HOST_PTR_WRITABLE,
   /* That, or else memory it can pin for reading, bound read-only. */
   AGX_HOST_PTR_ANY,
   /* Only memory the kernel cannot pin for writing, bound read-only. */
   AGX_HOST_PTR_READ_ONLY,
};

struct agx_bo *agx_bo_import_host_ptr(struct agx_device *dev, void *ptr,
                                      size_t size,
                                      enum agx_host_ptr_access access);
int agx_bo_set_resident(struct agx_device *dev, struct agx_bo *bo, bool keep);
int agx_bo_export(struct agx_device *dev, struct agx_bo *bo);
bool agx_bo_make_shared(struct agx_device *dev, struct agx_bo *bo);

void agx_bo_free(struct agx_device *dev, struct agx_bo *bo);
uint64_t agx_bo_get_g17p_vdm_alias(struct agx_device *dev, struct agx_bo *bo);
struct agx_bo *agx_bo_cache_fetch(struct agx_device *dev, size_t size,
                                  size_t align, uint32_t flags,
                                  const bool dontwait);
bool agx_bo_cache_has(struct agx_device *dev, size_t size, size_t align,
                      uint32_t flags);
void agx_bo_cache_evict_all(struct agx_device *dev);
void agx_bo_cache_start_idle_trim(struct agx_device *dev);
void agx_bo_cache_stop_idle_trim(struct agx_device *dev);
