/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_pool.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "drm-uapi/asahi_drm.h"
#include "util/log.h"
#include "util/u_debug.h"

struct hk_vdm_slab {
   struct list_head link;
   struct agx_bo *bo;
   void *map;
   uint64_t alias;
   enum hk_cmd_slab_class class;
   uint32_t spans;
   struct util_vma_heap heap;
};

#include "hk_cmd_buffer.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "util/cache_ops.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include "util/libsync.h"
#include "util/u_atomic.h"
#include "vk_queue.h"

static void
hk_g17p_vdm_alias_free(struct agx_device *dev, uint64_t alias, uint64_t size);

static uint64_t
hk_g17p_vdm_alias_alloc(struct agx_device *dev, uint64_t size)
{
   simple_mtx_lock(&dev->vma_lock);
   const uint64_t alias = util_vma_heap_alloc(
      &dev->g17p_vdm_alias_heap, size, AIL_PAGESIZE);
   simple_mtx_unlock(&dev->vma_lock);

   if (agx_g17p_vdm_alias_range_valid(alias, size))
      return alias;

   if (alias)
      hk_g17p_vdm_alias_free(dev, alias, size);
   return 0;
}

static void
hk_g17p_vdm_alias_free(struct agx_device *dev, uint64_t alias, uint64_t size)
{
   simple_mtx_lock(&dev->vma_lock);
   util_vma_heap_free(&dev->g17p_vdm_alias_heap, alias, size);
   simple_mtx_unlock(&dev->vma_lock);
}

VkResult
hk_cmd_bo_ensure_vdm_alias(struct hk_cmd_pool *pool, struct hk_cmd_bo *bo)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   /* Command pools are externally synchronized. A mapping is established
    * only when this BO first supplies a VDM pointer, then remains immutable
    * through submission and cached BO reuse. Ordinary data/compute uploads
    * consume no space in the finite VDM aperture. */
   if (bo->g17p_vdm_alias || !hk_g17p_vdm_low_window_enabled(&dev->dev))
      return VK_SUCCESS;

   const uint64_t alias = hk_g17p_vdm_alias_alloc(&dev->dev, bo->bo->size);
   const uint32_t flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;
   if (!alias || agx_bo_bind(&dev->dev, bo->bo, alias, bo->bo->size, 0, flags)) {
      if (alias)
         hk_g17p_vdm_alias_free(&dev->dev, alias, bo->bo->size);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   bo->g17p_vdm_alias = alias;
   return VK_SUCCESS;
}

static VkResult
hk_cmd_bo_create(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                 struct hk_cmd_bo **bo_out)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_bo *bo;
   /* Internal backing metadata can outlive the API command pool. */
   const VkAllocationCallbacks *alloc = &dev->vk.alloc;

   bo = vk_zalloc(alloc, sizeof(*bo), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);


   static const char *labels[HK_CMD_BO_TYPE_COUNT] = {
      [HK_CMD_BO_MAIN] = "Main command pool",
      [HK_CMD_BO_USC] = "USC command pool",
      [HK_CMD_BO_ROOT] = "Root descriptor pool",
   };
   const enum agx_bo_flags flags =
      (type == HK_CMD_BO_USC    ? AGX_BO_LOW_VA
       : type == HK_CMD_BO_ROOT ? AGX_BO_WRITEBACK
                                : 0) |
      (hk_g17p_cached_cmd_memory(&dev->dev) ? AGX_BO_WRITEBACK : 0);

   bo->bo = agx_bo_create(&dev->dev, HK_CMD_BO_SIZE, 0, flags, labels[type]);
   if (bo->bo == NULL) {
      vk_free(alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   bo->map = agx_bo_map(bo->bo);
   if (!bo->map) {
      agx_bo_unreference(&dev->dev, bo->bo);
      vk_free(alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   bo->addr = bo->bo->va->addr;
   bo->size = bo->bo->size;

   /* G17P and M5 command recording must not take first-write faults midway
    * through a draw. These BOs are recycled by the command pool, so populate
    * and initialize their shared mappings once at creation; subsequent command
    * buffers reuse already-resident pages without adding per-draw work. */
   if (agx_chip_native_or_m5(dev->dev.chip)) {
      memset(bo->map, 0, bo->bo->size);
      if (flags & AGX_BO_WRITEBACK)
         util_flush_range(bo->map, bo->bo->size);
   }

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
hk_cmd_bo_destroy(struct hk_device *dev, UNUSED const VkAllocationCallbacks *alloc,
                   struct hk_cmd_bo *bo)
{
   if (bo->span) {
      hk_device_free_span(dev, bo->span);
      vk_free(&dev->vk.alloc, bo);
      return;
   }

   if (bo->g17p_vdm_alias) {
      /* A failed unbind may leave the old mapping installed. Keep its
       * address reserved until VM teardown rather than alias a later BO. */
      if (!agx_bo_bind(&dev->dev, NULL, bo->g17p_vdm_alias, bo->bo->size, 0,
                       DRM_ASAHI_BIND_UNBIND))
         hk_g17p_vdm_alias_free(&dev->dev, bo->g17p_vdm_alias, bo->bo->size);
   }
   agx_bo_unreference(&dev->dev, bo->bo);
   vk_free(&dev->vk.alloc, bo);
}

static bool
hk_cmd_slab_is_gpu_only(enum hk_cmd_slab_class class)
{
   return class == HK_CMD_SLAB_VDM_GPU || class == HK_CMD_SLAB_IOTO_GPU ||
          class == HK_CMD_SLAB_DATA_GPU;
}

static struct hk_vdm_slab *
hk_device_create_vdm_slab(struct hk_device *dev, enum hk_cmd_slab_class class,
                          uint64_t min_size)
{
   static const char *const labels[HK_CMD_SLAB_COUNT] = {
      [HK_CMD_SLAB_VDM] = "VDM command slab",
      [HK_CMD_SLAB_MAIN] = "Main command slab",
      [HK_CMD_SLAB_USC] = "USC command slab",
      [HK_CMD_SLAB_IOTO] = "G17P Ioto program slab",
      [HK_CMD_SLAB_VDM_GPU] = "GPU indirect VDM slab",
      [HK_CMD_SLAB_IOTO_GPU] = "GPU indirect Ioto slab",
      [HK_CMD_SLAB_DATA_GPU] = "GPU indirect data slab",
   };
   const bool gpu_only = hk_cmd_slab_is_gpu_only(class);
   const enum agx_bo_flags flags =
      (class == HK_CMD_SLAB_USC ? AGX_BO_LOW_VA
       : (class == HK_CMD_SLAB_IOTO || class == HK_CMD_SLAB_IOTO_GPU)
          ? AGX_BO_EXEC | AGX_BO_LOW_VA
          : 0) |
      (!gpu_only && hk_g17p_cached_cmd_memory(&dev->dev) ? AGX_BO_WRITEBACK
                                                         : 0);

   if (min_size > HK_CMD_SLAB_MAX_SIZE)
      return NULL;

   struct hk_vdm_slab *slab =
      vk_zalloc(&dev->vk.alloc, sizeof(*slab), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!slab)
      return NULL;

   uint32_t size = dev->vdm.next_size[class] ?: HK_CMD_BO_SIZE;
   while (size < min_size && size < HK_CMD_SLAB_MAX_SIZE)
      size = MIN2(size * 2, HK_CMD_SLAB_MAX_SIZE);
   dev->vdm.next_size[class] = MIN2(size * 2, HK_CMD_SLAB_MAX_SIZE);
   slab->bo = agx_bo_create(&dev->dev, size, 0, flags, labels[class]);
   if (!slab->bo) {
      vk_free(&dev->vk.alloc, slab);
      return NULL;
   }

   /* Map before creating the GPU alias so a failed CPU mapping cannot
    * publish a slab with an invalid recording pointer. GPU-only slabs are
    * written by pre-graphics compute and stay unmapped on the CPU. */
   if (!gpu_only) {
      slab->map = agx_bo_map(slab->bo);
      if (!slab->map) {
         agx_bo_unreference(&dev->dev, slab->bo);
         vk_free(&dev->vk.alloc, slab);
         return NULL;
      }
   }

   /* Uploader spans name their primary mapping, so they spend none of the
    * finite alias aperture. GPU indirect VDM slabs also use their primary
    * mapping whenever the whole BO already lies inside the lower-root compact
    * VDM window, reserving the finite alias aperture for addresses outside it. */
   bool need_vdm_alias = class == HK_CMD_SLAB_VDM;
   if (class == HK_CMD_SLAB_VDM_GPU &&
       hk_g17p_vdm_low_window_enabled(&dev->dev)) {
      uint64_t start_compact = 0, end_compact = 0;
      need_vdm_alias =
         !agx_g17p_vdm_payload_compact(slab->bo->va->addr, &start_compact) ||
         !agx_g17p_vdm_payload_compact(slab->bo->va->addr + slab->bo->size - 4,
                                       &end_compact) ||
         start_compact >= (UINT64_C(1) << 40) ||
         end_compact >= (UINT64_C(1) << 40);
   }
   if (need_vdm_alias) {
      slab->alias = hk_g17p_vdm_alias_alloc(&dev->dev, slab->bo->size);
      const uint32_t bind = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;
      if (!slab->alias || agx_bo_bind(&dev->dev, slab->bo, slab->alias,
                                      slab->bo->size, 0, bind)) {
         if (slab->alias)
            hk_g17p_vdm_alias_free(&dev->dev, slab->alias, slab->bo->size);
         agx_bo_unreference(&dev->dev, slab->bo);
         vk_free(&dev->vk.alloc, slab);
         return NULL;
      }
   }

   slab->class = class;
   util_vma_heap_init(&slab->heap, slab->bo->va->addr, slab->bo->size);
   slab->heap.alloc_high = false;
   list_addtail(&slab->link, &dev->vdm.slabs[class]);

   return slab;
}

static void
hk_vdm_slab_destroy(struct hk_device *dev, struct hk_vdm_slab *slab)
{
   util_vma_heap_finish(&slab->heap);
   if (slab->alias &&
       !agx_bo_bind(&dev->dev, NULL, slab->alias, slab->bo->size, 0,
                    DRM_ASAHI_BIND_UNBIND))
      hk_g17p_vdm_alias_free(&dev->dev, slab->alias, slab->bo->size);
   agx_bo_unreference(&dev->dev, slab->bo);
   vk_free(&dev->vk.alloc, slab);
}

VkResult
hk_cmd_pool_alloc_span(struct hk_cmd_pool *pool, enum hk_cmd_slab_class class,
                       uint64_t size, uint64_t alignment, struct agx_ptr *ptr,
                       struct hk_vdm_allocation **allocation)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   const bool gpu_only = hk_cmd_slab_is_gpu_only(class);
   const uint64_t max_size = gpu_only ? HK_CMD_SLAB_MAX_SIZE : HK_CMD_BO_SIZE;
   if (!size || size > max_size || !alignment || !ptr || !allocation)
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   simple_mtx_lock(&dev->vdm.lock);
   uint64_t addr = 0;
   struct hk_vdm_slab *owner = NULL;
   list_for_each_entry_safe(struct hk_vdm_slab, slab, &dev->vdm.slabs[class],
                            link) {
      addr = util_vma_heap_alloc(&slab->heap, size, alignment);
      if (addr) {
         owner = slab;
         break;
      }
      if (!gpu_only || size <= HK_CMD_BO_SIZE)
         list_move_to(&slab->link, &dev->vdm.full[class]);
   }

   if (!owner) {
      owner = hk_device_create_vdm_slab(dev, class, size);
      if (!owner) {
         simple_mtx_unlock(&dev->vdm.lock);
         return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      addr = util_vma_heap_alloc(&owner->heap, size, alignment);
      if (!addr) {
         simple_mtx_unlock(&dev->vdm.lock);
         return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   struct hk_vdm_allocation *lease = calloc(1, sizeof(*lease));
   if (!lease) {
      util_vma_heap_free(&owner->heap, addr, size);
      simple_mtx_unlock(&dev->vdm.lock);
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   owner->spans++;
   *lease = (struct hk_vdm_allocation){
      .slab = owner,
      .addr = addr,
      .alias = owner->alias ? owner->alias + (addr - owner->bo->va->addr) : 0,
      .size = size,
   };
   *ptr = (struct agx_ptr){
      .gpu = addr,
      .cpu = owner->map ? (uint8_t *)owner->map + (addr - owner->bo->va->addr)
                        : NULL,
   };
   *allocation = lease;
   simple_mtx_unlock(&dev->vdm.lock);
   return VK_SUCCESS;
}

VkResult
hk_cmd_pool_alloc_bo_span(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                          uint32_t size, struct hk_cmd_bo **bo_out)
{
   const enum hk_cmd_span_bucket bucket = hk_cmd_span_bucket_for_size(size);
   list_for_each_entry(struct hk_cmd_bo, bo, &pool->free_spans[type][bucket],
                       link) {
      if (bo->size == size) {
         list_del(&bo->link);
         pool->num_free_spans[type][bucket]--;
         *bo_out = bo;
         return VK_SUCCESS;
      }
   }

   struct hk_device *dev = hk_cmd_pool_device(pool);
   const VkAllocationCallbacks *alloc = &dev->vk.alloc;
   struct hk_cmd_bo *bo =
      vk_zalloc(alloc, sizeof(*bo), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct agx_ptr ptr;
   VkResult result = hk_cmd_pool_alloc_span(
      pool, type == HK_CMD_BO_USC ? HK_CMD_SLAB_USC : HK_CMD_SLAB_MAIN, size,
      HK_CMD_SPAN_SIZE, &ptr, &bo->span);
   if (result != VK_SUCCESS) {
      vk_free(alloc, bo);
      return result;
   }

   bo->bo = bo->span->slab->bo;
   bo->map = ptr.cpu;
   bo->addr = ptr.gpu;
   bo->size = size;
   *bo_out = bo;
   return VK_SUCCESS;
}

VkResult
hk_cmd_pool_alloc_ioto_span(struct hk_cmd_pool *pool, struct agx_ptr *ptr,
                            struct hk_vdm_allocation **allocation)
{
   if (!list_is_empty(&pool->free_ioto_spans)) {
      struct hk_vdm_allocation *span = list_first_entry(
         &pool->free_ioto_spans, struct hk_vdm_allocation, link);
      list_del(&span->link);
      pool->num_free_ioto_spans--;
      *ptr = (struct agx_ptr){
         .gpu = span->addr,
         .cpu = (uint8_t *)span->slab->map +
                (span->addr - span->slab->bo->va->addr),
      };
      *allocation = span;
      return VK_SUCCESS;
   }

   return hk_cmd_pool_alloc_span(pool, HK_CMD_SLAB_IOTO, HK_CMD_SPAN_SIZE,
                                 HK_CMD_SPAN_SIZE, ptr, allocation);
}

void
hk_cmd_pool_free_ioto_span(struct hk_cmd_pool *pool,
                           struct hk_vdm_allocation *allocation)
{
   if (!allocation)
      return;

   struct hk_device *dev = hk_cmd_pool_device(pool);
   const unsigned max_free = hk_cmd_pool_ioto_span_cache_max(&dev->dev);
   if (pool->num_free_ioto_spans < max_free) {
      list_add(&allocation->link, &pool->free_ioto_spans);
      pool->num_free_ioto_spans++;
      return;
   }

   hk_device_free_span(dev, allocation);
}

VkResult
hk_cmd_pool_alloc_vdm_span(struct hk_cmd_pool *pool, uint64_t size,
                           uint64_t alignment, struct agx_ptr *ptr,
                           struct hk_vdm_allocation **allocation)
{
   if (size == HK_G17P_VDM_UPLOAD_CHUNK_SIZE &&
       alignment <= HK_G17P_VDM_UPLOAD_CHUNK_SIZE &&
       !list_is_empty(&pool->free_vdm_spans)) {
      struct hk_vdm_allocation *span = list_first_entry(
         &pool->free_vdm_spans, struct hk_vdm_allocation, link);
      list_del(&span->link);
      pool->num_free_vdm_spans--;
      *ptr = (struct agx_ptr){
         .gpu = span->addr,
         .cpu = span->slab->map
                   ? (uint8_t *)span->slab->map +
                        (span->addr - span->slab->bo->va->addr)
                   : NULL,
      };
      *allocation = span;
      return VK_SUCCESS;
   }

   return hk_cmd_pool_alloc_span(pool, HK_CMD_SLAB_VDM, size, alignment, ptr,
                                 allocation);
}

void
hk_cmd_pool_free_vdm_span(struct hk_cmd_pool *pool,
                          struct hk_vdm_allocation *allocation)
{
   if (!allocation)
      return;

   struct hk_device *dev = hk_cmd_pool_device(pool);
   const unsigned max_free = hk_cmd_pool_vdm_span_cache_max(&dev->dev);
   if (allocation->slab->class == HK_CMD_SLAB_VDM &&
       allocation->size == HK_G17P_VDM_UPLOAD_CHUNK_SIZE &&
       pool->num_free_vdm_spans < max_free) {
      list_add(&allocation->link, &pool->free_vdm_spans);
      pool->num_free_vdm_spans++;
      return;
   }

   hk_device_free_span(dev, allocation);
}

void
hk_device_free_span(struct hk_device *dev, struct hk_vdm_allocation *allocation)
{
   if (!allocation)
      return;

   struct hk_vdm_slab *slab = allocation->slab;
   simple_mtx_lock(&dev->vdm.lock);
   util_vma_heap_free(&slab->heap, allocation->addr, allocation->size);
   list_move_to(&slab->link, &dev->vdm.slabs[slab->class]);
   /* An idle slab is kept only if it is the sole one with room, so that a
    * recording that frees its last span and starts again does not rebuild it,
    * and a burst of one-shot recordings gives its memory back. */
   if (!--slab->spans && !list_is_singular(&dev->vdm.slabs[slab->class])) {
      list_del(&slab->link);
      hk_vdm_slab_destroy(dev, slab);
   }
   simple_mtx_unlock(&dev->vdm.lock);
   free(allocation);
}

void
hk_device_finish_vdm_slabs(struct hk_device *dev)
{
   for (unsigned class = 0; class < HK_CMD_SLAB_COUNT; ++class) {
      list_splicetail(&dev->vdm.full[class], &dev->vdm.slabs[class]);
      list_inithead(&dev->vdm.full[class]);
      list_for_each_entry_safe(struct hk_vdm_slab, slab,
                               &dev->vdm.slabs[class], link) {
         list_del(&slab->link);
         hk_vdm_slab_destroy(dev, slab);
      }
      list_inithead(&dev->vdm.slabs[class]);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateCommandPool(VkDevice _device,
                     const VkCommandPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkCommandPool *pCmdPool)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   struct hk_cmd_pool *pool;

   pool = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pool), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pool == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_pool_init(&device->vk, &pool->vk, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, pool);
      return result;
   }

   for (enum hk_cmd_bo_type type = 0; type < HK_CMD_BO_TYPE_COUNT; ++type) {
      list_inithead(&pool->free_bos[type]);
      pool->num_free_bos[type] = 0;
      for (unsigned bucket = 0; bucket < HK_CMD_SPAN_BUCKET_COUNT; ++bucket) {
         list_inithead(&pool->free_spans[type][bucket]);
         pool->num_free_spans[type][bucket] = 0;
      }
   }
   list_inithead(&pool->free_ioto_spans);
   pool->num_free_ioto_spans = 0;
   list_inithead(&pool->free_vdm_spans);
   pool->num_free_vdm_spans = 0;
   list_inithead(&pool->free_cs);
   pool->num_free_cs = 0;

   *pCmdPool = hk_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

static void
hk_cmd_pool_destroy_bos(struct hk_cmd_pool *pool)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   for (enum hk_cmd_bo_type type = 0; type < HK_CMD_BO_TYPE_COUNT; ++type) {
      list_for_each_entry_safe(struct hk_cmd_bo, bo, &pool->free_bos[type],
                               link)
         hk_cmd_bo_destroy(dev, &pool->vk.alloc, bo);
      list_inithead(&pool->free_bos[type]);
      pool->num_free_bos[type] = 0;

      for (unsigned bucket = 0; bucket < HK_CMD_SPAN_BUCKET_COUNT; ++bucket) {
         list_for_each_entry_safe(struct hk_cmd_bo, bo,
                                  &pool->free_spans[type][bucket], link)
            hk_cmd_bo_destroy(dev, &pool->vk.alloc, bo);
         list_inithead(&pool->free_spans[type][bucket]);
         pool->num_free_spans[type][bucket] = 0;
      }
   }

   list_for_each_entry_safe(struct hk_vdm_allocation, span,
                            &pool->free_ioto_spans, link) {
      list_del(&span->link);
      hk_device_free_span(dev, span);
   }
   list_inithead(&pool->free_ioto_spans);
   pool->num_free_ioto_spans = 0;

   list_for_each_entry_safe(struct hk_vdm_allocation, span,
                            &pool->free_vdm_spans, link) {
      list_del(&span->link);
      hk_device_free_span(dev, span);
   }
   list_inithead(&pool->free_vdm_spans);
   pool->num_free_vdm_spans = 0;

   list_for_each_entry_safe(struct hk_cs, cs, &pool->free_cs, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   list_inithead(&pool->free_cs);
   pool->num_free_cs = 0;
}

VkResult
hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                     struct hk_cmd_bo **bo_out)
{
   struct hk_cmd_bo *bo = NULL;
   if (!list_is_empty(&pool->free_bos[type])) {
      bo = list_first_entry(&pool->free_bos[type], struct hk_cmd_bo, link);
      pool->num_free_bos[type]--;
   }

   if (bo) {
      list_del(&bo->link);
      *bo_out = bo;
      return VK_SUCCESS;
   }

   return hk_cmd_bo_create(pool, type, bo_out);
}

void
hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                         struct list_head *bos)
{
   const struct hk_device *dev = hk_cmd_pool_device(pool);
   /* Under virtio every BO is an external object, which the host kernel locks
    * on every submit: a pool hoarding hundreds of idle BOs slows all submits. */
   const unsigned max_free =
      agx_chip_native_or_m5(dev->dev.chip) &&
         type == HK_CMD_BO_MAIN && !dev->dev.is_virtio
         ? HK_G17P_CMD_POOL_MAIN_BO_MAX
         : HK_CMD_POOL_BO_MAX;
   const unsigned max_free_spans =
      hk_cmd_pool_span_cache_max(&dev->dev, type);

   list_for_each_entry_safe(struct hk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (bo->span) {
         const enum hk_cmd_span_bucket bucket =
            hk_cmd_span_bucket_for_size(bo->size);
         if (pool->num_free_spans[type][bucket] < max_free_spans) {
            list_add(&bo->link, &pool->free_spans[type][bucket]);
            pool->num_free_spans[type][bucket]++;
         } else {
            hk_cmd_bo_destroy(hk_cmd_pool_device(pool), &pool->vk.alloc, bo);
         }
      } else if (pool->num_free_bos[type] >= max_free) {
         hk_cmd_bo_destroy(hk_cmd_pool_device(pool),
                           &pool->vk.alloc, bo);
      } else {
         list_addtail(&bo->link, &pool->free_bos[type]);
         pool->num_free_bos[type]++;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyCommandPool(VkDevice _device, VkCommandPool commandPool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   if (!pool)
      return;

   vk_command_pool_finish(&pool->vk);
   hk_cmd_pool_destroy_bos(pool);
   vk_free2(&device->vk.alloc, pAllocator, pool);
}

VKAPI_ATTR void VKAPI_CALL
hk_TrimCommandPool(VkDevice device, VkCommandPool commandPool,
                   VkCommandPoolTrimFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   vk_command_pool_trim(&pool->vk, flags);
   hk_cmd_pool_destroy_bos(pool);
}

void
hk_cmd_pool_clean_vdm(const struct hk_vdm_allocation *allocation)
{
   const struct hk_vdm_slab *slab = allocation->slab;
   if (!slab->map)
      return;
   util_flush_range_no_fence(
      (uint8_t *)slab->map + (allocation->addr - slab->bo->va->addr),
      allocation->size);
}

/* Driver-owned buffers are pinned independently by each submission. */
struct hk_cmd_pending {
   struct list_head link;

   /* One private binary syncobj per submit ioctl, whether its fence is
    * already known to have signalled without error, and scratch space for
    * polling the others. Polling holds cmd_pending_lock. A syncobj may be
    * reused from an earlier submission: a successful ioctl replaces its
    * fence, so only submissions whose ioctls all succeeded are polled.
    */
   uint32_t *signals;
   uint32_t *unknown;
   bool *succeeded;
   unsigned chunks, attempted;

   /* The queue timeline point its last ioctl signals. */
   uint64_t point;

   unsigned count;
   struct agx_bo *bos[];
};

void
hk_cmd_pending_init(struct hk_device *dev)
{
   simple_mtx_init(&dev->cmd_pending_lock, mtx_plain);
   list_inithead(&dev->cmd_pending_queues);
   list_inithead(&dev->cmd_pending_failed);
   list_inithead(&dev->cmd_lost_bos);
   list_inithead(&dev->cmd_lost_spans);
   list_inithead(&dev->cmd_lost_large_bos);
   dev->cmd_pending_ready = true;
}

static void
hk_cmd_pending_release(struct hk_device *dev, struct hk_cmd_pending *p,
                       bool teardown)
{
   for (unsigned i = 0; i < p->count; ++i) {
      struct agx_bo *bo = p->bos[i];

      /* Only successful completion allows reuse. Kernel mapping pins cover
       * outstanding GPU accesses once userspace closes the handle. */
      if (teardown)
         bo->flags |= AGX_BO_NO_CACHE;

      agx_bo_unreference(&dev->dev, bo);
   }

   for (unsigned i = 0; i < p->chunks; ++i)
      drmSyncobjDestroy(dev->dev.fd, p->signals[i]);

   vk_free(&dev->vk.alloc, p->signals);
   vk_free(&dev->vk.alloc, p);
}

/* Keeps the syncobjs of a submission that completed without error, or that
 * no ioctl used, for later submissions; release destroys the rest. Called
 * with cmd_pending_lock held.
 */
static void
hk_cmd_pending_recycle(struct hk_device *dev, struct hk_cmd_pending *p)
{
   while (p->chunks && dev->cmd_syncobj_count < HK_CMD_SYNCOBJ_POOL)
      dev->cmd_syncobjs[dev->cmd_syncobj_count++] = p->signals[--p->chunks];
}

enum hk_cmd_pending_state {
   HK_PENDING_BUSY,
   HK_PENDING_SUCCEEDED,
   HK_PENDING_FAILED,
};

/* A signalled ERROR is deliberately not completion for ownership purposes.
 * A private binary syncobj keeps the original fence, including its error;
 * exporting a pruned queue timeline point could instead yield a success
 * stub. Unless the queue timeline has already shown that the submission's
 * fences signalled, one zero-timeout wait rejects a submission that is still
 * running. The status of each signalled fence is then read once and
 * remembered.
 */
static enum hk_cmd_pending_state
hk_cmd_pending_poll(struct hk_device *dev, struct hk_cmd_pending *p, bool wait)
{
   unsigned unknown_count = 0;

   for (unsigned i = 0; wait && i < p->attempted; ++i) {
      if (!p->succeeded[i])
         p->unknown[unknown_count++] = p->signals[i];
   }

   if (unknown_count &&
       drmSyncobjWait(dev->dev.fd, p->unknown, unknown_count, 0,
                      DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL)) {
      /* A syncobj without any fence never gets one: keep the memory, but
       * stop polling it.
       */
      return errno == EINVAL ? HK_PENDING_FAILED : HK_PENDING_BUSY;
   }

   for (unsigned i = 0; i < p->attempted; ++i) {
      if (p->succeeded[i])
         continue;

      int fd = -1;
      if (drmSyncobjExportSyncFile(dev->dev.fd, p->signals[i], &fd))
         return errno == EINVAL ? HK_PENDING_FAILED : HK_PENDING_BUSY;

      struct sync_file_info info = {0};
      int ret = ioctl(fd, SYNC_IOC_FILE_INFO, &info);
      close(fd);
      if (ret || info.status == 0)
         return HK_PENDING_BUSY;
      if (info.status < 0)
         return HK_PENDING_FAILED;

      p->succeeded[i] = true;
   }

   return HK_PENDING_SUCCEEDED;
}

/* Reaps after which a queue's oldest submission still blocks its timeline
 * before later ones are polled, and how many still-running ones a reap
 * polls then.
 */
#define HK_CMD_STALL_REAPS 16
#define HK_CMD_STALL_POLLS 4

/* A queue's submissions are in timeline order, and a reached timeline point
 * means every timeline signal up to it has happened. Check the queue's
 * submissions oldest first and stop at the first one past the reached point.
 * When the kernel's VM status mirror is mapped and clean (0), a single-chunk
 * submission whose point is reached completed without error and can be reaped
 * without per-syncobj ExportSyncFile + SYNC_IOC_FILE_INFO + close syscalls.
 * Multi-chunk submissions, lost devices, and non-zero mirror words fall back
 * to per-syncobj polling. Fences on the timeline can signal out of order, so
 * an unfinished submission can hold back finished ones behind it; once it has
 * done so for a while, a few of them are also polled on each reap. Finished
 * work behind several long-running submissions stays pinned until those
 * finish. Called with cmd_pending_lock held.
 */
static void
hk_cmd_pending_reap_queue(struct hk_device *dev, struct hk_queue *queue,
                          struct list_head *ready,
                          const struct hk_queue *cached_queue,
                          bool *cached_valid, uint64_t *cached_done)
{
   struct hk_cmd_pending *oldest =
      list_first_entry(&queue->cmd_pending, struct hk_cmd_pending, link);
   struct hk_cmd_pending *newest =
      list_last_entry(&queue->cmd_pending, struct hk_cmd_pending, link);

   uint64_t reached = p_atomic_read(&queue->drm.signalled_value);
   if (cached_queue == queue && cached_valid && *cached_valid) {
      reached = MAX2(reached, *cached_done);
   } else if (reached < newest->point) {
      uint64_t queried = 0;
      if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &queried, 1) == 0) {
         hk_queue_note_signalled(queue, queried);
         reached = MAX2(reached, queried);
         if (cached_queue == queue && cached_valid && cached_done) {
            *cached_done = reached;
            *cached_valid = true;
         }
      }
   }

   /* The kernel stores any sticky VM error into vm_status_mirror before
    * signalling that job's fence (hk_init_vm_status_mirror()). An acquire load
    * after observing the reached timeline point guarantees every single-chunk
    * submission up to `reached` completed without error, avoiding three
    * syscalls (ExportSyncFile + SYNC_IOC_FILE_INFO + close) per retired
    * submission. Any non-zero error word, lost device, or multi-chunk
    * submission falls back to per-syncobj polling. */
   const bool vm_clean =
      dev->vm_status_mirror.word != NULL &&
      !vk_device_is_lost_no_report(&dev->vk) &&
      __atomic_load_n(dev->vm_status_mirror.word, __ATOMIC_ACQUIRE) == 0;

   if (oldest->point <= reached)
      queue->cmd_stalled = 0;
   else if (queue->cmd_stalled < HK_CMD_STALL_REAPS)
      queue->cmd_stalled++;

   unsigned polls =
      queue->cmd_stalled >= HK_CMD_STALL_REAPS ? HK_CMD_STALL_POLLS : 0;

   list_for_each_entry_safe(struct hk_cmd_pending, p, &queue->cmd_pending,
                            link) {
      const bool signalled = p->point <= reached;
      if (!signalled && !polls)
         break;

      const enum hk_cmd_pending_state state =
         (signalled && vm_clean && p->chunks == 1 && p->attempted == 1)
            ? HK_PENDING_SUCCEEDED
            : hk_cmd_pending_poll(dev, p, !signalled);

      switch (state) {
      case HK_PENDING_BUSY:
         if (!signalled)
            polls--;
         break;
      case HK_PENDING_SUCCEEDED:
         list_del(&p->link);
         hk_cmd_pending_recycle(dev, p);
         list_addtail(&p->link, ready);
         break;
      case HK_PENDING_FAILED:
         /* Its memory stays out of reuse until device destruction. */
         list_del(&p->link);
         list_addtail(&p->link, &dev->cmd_pending_failed);
         break;
      }
   }

   if (list_is_empty(&queue->cmd_pending)) {
      list_del(&queue->cmd_link);
      p_atomic_dec(&dev->cmd_pending_active);
      queue->cmd_stalled = 0;
   }
}

void
hk_cmd_pending_reap_cached(struct hk_device *dev,
                           const struct hk_queue *cached_queue,
                           bool *cached_valid, uint64_t *cached_done)
{
   /* Meta command pools can be destroyed during device-init failure. Skip the
    * device lock when no driver-owned pending submissions are active. */
   if (!dev->cmd_pending_ready || !p_atomic_read(&dev->cmd_pending_active))
      return;

   struct list_head ready;
   list_inithead(&ready);

   simple_mtx_lock(&dev->cmd_pending_lock);
   list_for_each_entry_safe(struct hk_queue, queue, &dev->cmd_pending_queues,
                            cmd_link)
      hk_cmd_pending_reap_queue(dev, queue, &ready, cached_queue, cached_valid,
                                cached_done);
   simple_mtx_unlock(&dev->cmd_pending_lock);

   /* Releasing a span takes vdm.lock; never do it under the pending lock. */
   list_for_each_entry_safe(struct hk_cmd_pending, p, &ready, link) {
      list_del(&p->link);
      hk_cmd_pending_release(dev, p, false);
   }
}

void
hk_cmd_pending_reap(struct hk_device *dev)
{
   hk_cmd_pending_reap_cached(dev, NULL, NULL, NULL);
}

VkResult
hk_cmd_pending_begin(struct hk_device *dev, unsigned chunks,
                     struct agx_bo *const *extra_bos, unsigned extra_bo_count,
                     struct hk_queue *reap_queue, bool *reap_done_valid,
                     uint64_t *reap_done, bool already_reaped,
                     struct hk_cmd_pending **out)
{
   *out = NULL;
   if (!already_reaped)
      hk_cmd_pending_reap_cached(dev, reap_queue, reap_done_valid, reap_done);

   size_t count = extra_bo_count;
   if (count > (SIZE_MAX - sizeof(struct hk_cmd_pending)) /
                  sizeof(struct agx_bo *))
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct hk_cmd_pending *p =
      vk_zalloc(&dev->vk.alloc, sizeof(*p) + count * sizeof(p->bos[0]), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!p)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* The polling scratch and success flags share the handle allocation. */
   p->signals = vk_zalloc(&dev->vk.alloc,
                          (size_t)chunks * (2 * sizeof(uint32_t) + sizeof(bool)),
                          8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!p->signals) {
      vk_free(&dev->vk.alloc, p);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   p->unknown = p->signals + chunks;
   p->succeeded = (bool *)(p->unknown + chunks);

   simple_mtx_lock(&dev->cmd_pending_lock);
   while (p->chunks < chunks && dev->cmd_syncobj_count)
      p->signals[p->chunks++] = dev->cmd_syncobjs[--dev->cmd_syncobj_count];
   simple_mtx_unlock(&dev->cmd_pending_lock);

   for (; p->chunks < chunks; ++p->chunks) {
      if (drmSyncobjCreate(dev->dev.fd, 0, &p->signals[p->chunks])) {
         simple_mtx_lock(&dev->cmd_pending_lock);
         hk_cmd_pending_recycle(dev, p);
         simple_mtx_unlock(&dev->cmd_pending_lock);
         hk_cmd_pending_release(dev, p, false);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   for (unsigned i = 0; i < extra_bo_count; ++i) {
      if (extra_bos[i]) {
         agx_bo_reference(extra_bos[i]);
         p->bos[p->count++] = extra_bos[i];
      }
   }

   *out = p;
   return VK_SUCCESS;
}

uint32_t
hk_cmd_pending_signal(struct hk_cmd_pending *p)
{
   assert(p->attempted < p->chunks);
   return p->signals[p->attempted++];
}

/* Called by the queue's serialized submission, after its last ioctl. */
void
hk_cmd_pending_end(struct hk_device *dev, struct hk_cmd_pending *p,
                   struct hk_queue *queue, VkResult result)
{
   if (!p)
      return;

   /* Once p is on a queue's list and the lock drops, a reap on another
    * thread may release it: decide before then whether this thread does.
    */
   const bool release = !p->attempted;

   simple_mtx_lock(&dev->cmd_pending_lock);
   if (release) {
      /* No ioctl used its syncobjs or memory. */
      hk_cmd_pending_recycle(dev, p);
   } else if (result != VK_SUCCESS) {
      /* A failed ioctl may leave a reused syncobj with its earlier fence.
       * Never judge such a submission by its syncobjs: keep its memory.
       */
      list_addtail(&p->link, &dev->cmd_pending_failed);
   } else {
      /* The submission's last ioctl signals this point (earlier ioctls do too
       * with chunk timelines; otherwise only their own syncobjs, which the
       * reap still reads). Later submissions to the queue signal higher
       * points.
       */
      p->point = queue->drm.timeline_value;
      if (list_is_empty(&queue->cmd_pending)) {
         list_addtail(&queue->cmd_link, &dev->cmd_pending_queues);
         p_atomic_inc(&dev->cmd_pending_active);
      } else {
         assert(list_last_entry(&queue->cmd_pending, struct hk_cmd_pending,
                                link)->point < p->point);
      }
      list_addtail(&p->link, &queue->cmd_pending);
   }
   simple_mtx_unlock(&dev->cmd_pending_lock);

   if (release)
      hk_cmd_pending_release(dev, p, false);
}

static bool
hk_device_lost_or_vm_error(struct hk_device *dev)
{
   if (unlikely(vk_device_is_lost_no_report(&dev->vk)))
      return true;

   return unlikely(dev->vm_status_mirror.word &&
                   __atomic_load_n(dev->vm_status_mirror.word,
                                   __ATOMIC_ACQUIRE) != 0);
}

/* Transfer recording ownership to the device if Vulkan or the kernel reports
 * loss. Jobs may still read these bytes while the application resets or frees
 * its recordings. Use the existing list links so loss cleanup cannot fail
 * while the system is out of memory. */
void
hk_cmd_pending_retain_lost(struct hk_device *dev, struct hk_cmd_buffer *cmd)
{
   if (!dev->cmd_pending_ready || !hk_device_lost_or_vm_error(dev))
      return;

   struct list_head *lists[] = {
      &cmd->uploader.main.bos,
      &cmd->uploader.usc.bos,
      &cmd->uploader.root.bos,
   };

   simple_mtx_lock(&dev->cmd_pending_lock);
   for (unsigned i = 0; i < ARRAY_SIZE(lists); ++i) {
      list_splicetail(lists[i], &dev->cmd_lost_bos);
      list_inithead(lists[i]);
   }

   list_splicetail(&cmd->vdm_allocations, &dev->cmd_lost_spans);
   list_inithead(&cmd->vdm_allocations);
   if (cmd->ioto.span) {
      list_addtail(&cmd->ioto.span->link, &dev->cmd_lost_spans);
      cmd->ioto.span = NULL;
   }

   util_dynarray_foreach(&cmd->large_bos, struct agx_bo *, entry) {
      struct agx_bo *bo = *entry;

      /* An active BO cannot be on the cache LRU. Keep a device reference
       * until teardown, once per BO regardless of how many recordings use
       * it. The recording drops its own reference during normal cleanup. */
      if (!list_is_linked(&bo->lru_link)) {
         agx_bo_reference(bo);
         list_addtail(&bo->lru_link, &dev->cmd_lost_large_bos);
      }
   }
   simple_mtx_unlock(&dev->cmd_pending_lock);
}

/* The queue's timeline goes away with it. As at device destruction, its
 * pending submissions keep their memory until the device is destroyed.
 */
void
hk_cmd_pending_finish_queue(struct hk_device *dev, struct hk_queue *queue)
{
   /* Queues are finished during device-init failure, before any submission. */
   if (!dev->cmd_pending_ready)
      return;

   simple_mtx_lock(&dev->cmd_pending_lock);
   if (!list_is_empty(&queue->cmd_pending)) {
      list_del(&queue->cmd_link);
      p_atomic_dec(&dev->cmd_pending_active);
      list_splicetail(&queue->cmd_pending, &dev->cmd_pending_failed);
      list_inithead(&queue->cmd_pending);
   }
   simple_mtx_unlock(&dev->cmd_pending_lock);
}

void
hk_cmd_pending_finish(struct hk_device *dev)
{
   /* Called only at device teardown, after all common submit workers joined
    * and kernel queues were destroyed. No application command recording is
    * allowed concurrently with DestroyDevice. This does not claim that queue
    * destruction drained the GPU: outstanding kernel mapping pins must
    * survive the unbind/close operations below. Finishing each queue moved
    * its pending submissions to the failed list.
    */
   assert(list_is_empty(&dev->cmd_pending_queues));
   list_for_each_entry_safe(struct hk_cmd_pending, p,
                            &dev->cmd_pending_failed, link) {
      list_del(&p->link);
      hk_cmd_pending_release(dev, p, true);
   }

   list_for_each_entry_safe(struct hk_cmd_bo, bo, &dev->cmd_lost_bos, link) {
      list_del(&bo->link);
      bo->bo->flags |= AGX_BO_NO_CACHE;
      hk_cmd_bo_destroy(dev, &dev->vk.alloc, bo);
   }

   list_for_each_entry_safe(struct hk_vdm_allocation, span,
                            &dev->cmd_lost_spans, link) {
      list_del(&span->link);
      span->slab->bo->flags |= AGX_BO_NO_CACHE;
      hk_device_free_span(dev, span);
   }

   list_for_each_entry_safe(struct agx_bo, bo, &dev->cmd_lost_large_bos,
                            lru_link) {
      list_del(&bo->lru_link);
      bo->flags |= AGX_BO_NO_CACHE;
      agx_bo_unreference(&dev->dev, bo);
   }

   for (unsigned i = 0; i < dev->cmd_syncobj_count; ++i)
      drmSyncobjDestroy(dev->dev.fd, dev->cmd_syncobjs[i]);
   dev->cmd_syncobj_count = 0;
}
