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

   bo->refcount = 1;

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
      if (hk_g17p_cached_cmd_memory(&dev->dev))
         util_flush_range(bo->map, bo->bo->size);
   }

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
hk_cmd_bo_destroy(struct hk_device *dev, UNUSED const VkAllocationCallbacks *alloc,
                   struct hk_cmd_bo *bo)
{
   if (!p_atomic_dec_zero(&bo->refcount))
      return;
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

static struct hk_vdm_slab *
hk_device_create_vdm_slab(struct hk_device *dev, enum hk_cmd_slab_class class)
{
   static const char *const labels[HK_CMD_SLAB_COUNT] = {
      [HK_CMD_SLAB_VDM] = "VDM command slab",
      [HK_CMD_SLAB_MAIN] = "Main command slab",
      [HK_CMD_SLAB_USC] = "USC command slab",
      [HK_CMD_SLAB_IOTO] = "G17P Ioto program slab",
   };
   const enum agx_bo_flags flags =
      (class == HK_CMD_SLAB_USC    ? AGX_BO_LOW_VA
       : class == HK_CMD_SLAB_IOTO ? AGX_BO_EXEC | AGX_BO_LOW_VA
                                   : 0) |
      (hk_g17p_cached_cmd_memory(&dev->dev) ? AGX_BO_WRITEBACK : 0);

   struct hk_vdm_slab *slab =
      vk_zalloc(&dev->vk.alloc, sizeof(*slab), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!slab)
      return NULL;

   const uint32_t size = dev->vdm.next_size[class] ?: HK_CMD_BO_SIZE;
   dev->vdm.next_size[class] = MIN2(size * 2, HK_CMD_SLAB_MAX_SIZE);
   slab->bo = agx_bo_create(&dev->dev, size, 0, flags, labels[class]);
   if (!slab->bo) {
      vk_free(&dev->vk.alloc, slab);
      return NULL;
   }

   /* Map before creating the GPU alias so a failed CPU mapping cannot
    * publish a slab with an invalid recording pointer. */
   slab->map = agx_bo_map(slab->bo);
   if (!slab->map) {
      agx_bo_unreference(&dev->dev, slab->bo);
      vk_free(&dev->vk.alloc, slab);
      return NULL;
   }

   /* Uploader spans name their primary mapping, so they spend none of the
    * finite alias aperture. */
   if (class == HK_CMD_SLAB_VDM) {
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
   if (!size || size > HK_CMD_BO_SIZE || !alignment || !ptr || !allocation)
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
      list_move_to(&slab->link, &dev->vdm.full[class]);
   }

   if (!owner) {
      owner = hk_device_create_vdm_slab(dev, class);
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
      .refcount = 1,
      .slab = owner,
      .addr = addr,
      .alias = owner->alias ? owner->alias + (addr - owner->bo->va->addr) : 0,
      .size = size,
   };
   *ptr = (struct agx_ptr){
      .gpu = addr,
      .cpu = (uint8_t *)owner->map + (addr - owner->bo->va->addr),
   };
   *allocation = lease;
   simple_mtx_unlock(&dev->vdm.lock);
   return VK_SUCCESS;
}

VkResult
hk_cmd_pool_alloc_bo_span(struct hk_cmd_pool *pool, enum hk_cmd_bo_type type,
                          uint32_t size, struct hk_cmd_bo **bo_out)
{
   list_for_each_entry(struct hk_cmd_bo, bo, &pool->free_spans[type], link) {
      if (bo->size == size) {
         list_del(&bo->link);
         pool->num_free_spans[type]--;
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

   bo->refcount = 1;
   bo->bo = bo->span->slab->bo;
   bo->map = ptr.cpu;
   bo->addr = ptr.gpu;
   bo->size = size;
   *bo_out = bo;
   return VK_SUCCESS;
}

void
hk_device_free_span(struct hk_device *dev, struct hk_vdm_allocation *allocation)
{
   if (!allocation || !p_atomic_dec_zero(&allocation->refcount))
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
      list_inithead(&pool->free_spans[type]);
      pool->num_free_spans[type] = 0;
   }

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
      list_for_each_entry_safe(struct hk_cmd_bo, bo, &pool->free_spans[type],
                               link)
         hk_cmd_bo_destroy(dev, &pool->vk.alloc, bo);

      list_inithead(&pool->free_bos[type]);
      pool->num_free_bos[type] = 0;
      list_inithead(&pool->free_spans[type]);
      pool->num_free_spans[type] = 0;
   }
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

   list_for_each_entry_safe(struct hk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (p_atomic_read(&bo->refcount) != 1) {
         /* Drop recording ownership, but never put pending bytes in either
          * pool reuse list. Submission pins also retain aliases and spans. */
         hk_cmd_bo_destroy(hk_cmd_pool_device(pool), &pool->vk.alloc, bo);
      } else if (bo->span && pool->num_free_spans[type] < HK_CMD_POOL_BO_MAX) {
         list_add(&bo->link, &pool->free_spans[type]);
         pool->num_free_spans[type]++;
      } else if (bo->span || pool->num_free_bos[type] >= max_free) {
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
   util_flush_range_no_fence(
      (uint8_t *)slab->map + (allocation->addr - slab->bo->va->addr),
      allocation->size);
}

/* One immutable resource snapshot per driver submission. An occurrence on
 * another queue takes independent pins; no scalar "last reader" is used.
 */
enum hk_cmd_pin_kind {
   HK_PIN_BO,
   HK_PIN_SPAN,
   HK_PIN_LARGE,
};

struct hk_cmd_pin {
   enum hk_cmd_pin_kind kind;
   void *object;
};

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
   struct hk_cmd_pin pins[];
};

static void
hk_cmd_add_pin(struct hk_cmd_pin *pins, size_t *count,
               enum hk_cmd_pin_kind kind, void *object)
{
   if (pins)
      pins[*count] = (struct hk_cmd_pin){kind, object};
   ++*count;
}

/* Count first, then fill a preallocated array. No fallible allocation may
 * happen after acquiring only a subset of the references. Repeated command
 * buffers are intentional: each occurrence gets balanced references.
 */
static size_t
hk_cmd_collect_pins(const struct vk_queue_submit *submit,
                    struct hk_cmd_pin *pins)
{
   size_t count = 0;

   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmd = (void *)submit->command_buffers[i];
      struct list_head *lists[] = {
         &cmd->uploader.main.bos,
         &cmd->uploader.usc.bos,
         &cmd->uploader.root.bos,
      };

      /* uploader.streams uses main.bos; uploader.vdm owns the leases below. */
      for (unsigned l = 0; l < ARRAY_SIZE(lists); ++l) {
         list_for_each_entry(struct hk_cmd_bo, bo, lists[l], link)
            hk_cmd_add_pin(pins, &count, HK_PIN_BO, bo);
      }

      list_for_each_entry(struct hk_vdm_allocation, span,
                          &cmd->vdm_allocations, link)
         hk_cmd_add_pin(pins, &count, HK_PIN_SPAN, span);

      if (cmd->ioto.span)
         hk_cmd_add_pin(pins, &count, HK_PIN_SPAN, cmd->ioto.span);

      util_dynarray_foreach(&cmd->large_bos, struct agx_bo *, bo)
         hk_cmd_add_pin(pins, &count, HK_PIN_LARGE, *bo);
   }

   return count;
}

static void
hk_cmd_pin_ref(struct hk_cmd_pin pin)
{
   switch (pin.kind) {
   case HK_PIN_BO:
      p_atomic_inc(&((struct hk_cmd_bo *)pin.object)->refcount);
      break;
   case HK_PIN_SPAN:
      p_atomic_inc(&((struct hk_vdm_allocation *)pin.object)->refcount);
      break;
   case HK_PIN_LARGE:
      agx_bo_reference(pin.object);
      break;
   }
}

static struct agx_bo *
hk_cmd_pin_backing(struct hk_cmd_pin pin)
{
   switch (pin.kind) {
   case HK_PIN_BO:
      return ((struct hk_cmd_bo *)pin.object)->bo;
   case HK_PIN_SPAN:
      return ((struct hk_vdm_allocation *)pin.object)->slab->bo;
   case HK_PIN_LARGE:
      return pin.object;
   }
   UNREACHABLE("command pin kind");
}

static void
hk_cmd_pending_release(struct hk_device *dev, struct hk_cmd_pending *p,
                       bool teardown)
{
   for (unsigned i = 0; i < p->count; ++i) {
      struct hk_cmd_pin pin = p->pins[i];

      /* Teardown is not a success witness. Do not make uncertain backing
       * available to the BO cache, even briefly while other objects die.
       * Kernel-owned mapping pins, not an error fence, protect GPU accesses
       * after the final userspace reference is dropped.
       */
      if (teardown)
         hk_cmd_pin_backing(pin)->flags |= AGX_BO_NO_CACHE;

      switch (pin.kind) {
      case HK_PIN_BO:
         hk_cmd_bo_destroy(dev, &dev->vk.alloc, pin.object);
         break;
      case HK_PIN_SPAN:
         hk_device_free_span(dev, pin.object);
         break;
      case HK_PIN_LARGE:
         agx_bo_unreference(&dev->dev, pin.object);
         break;
      }
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
 * The point only skips the wait: each submission's own syncobjs are still
 * read before its memory is released. Fences on the timeline can signal out
 * of order, so an unfinished submission can hold back finished ones behind
 * it; once it has done so for a while, a few of them are also polled on each
 * reap. Finished work behind several long-running submissions stays pinned
 * until those finish. Called with cmd_pending_lock held.
 */
static void
hk_cmd_pending_reap_queue(struct hk_device *dev, struct hk_queue *queue,
                          struct list_head *ready)
{
   uint64_t reached = 0;
   if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &reached, 1))
      reached = 0;

   struct hk_cmd_pending *oldest =
      list_first_entry(&queue->cmd_pending, struct hk_cmd_pending, link);
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

      switch (hk_cmd_pending_poll(dev, p, !signalled)) {
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
      queue->cmd_stalled = 0;
   }
}

void
hk_cmd_pending_reap(struct hk_device *dev)
{
   /* Meta command pools can be destroyed during device-init failure. */
   if (!dev->cmd_pending_ready)
      return;

   struct list_head ready;
   list_inithead(&ready);

   simple_mtx_lock(&dev->cmd_pending_lock);
   list_for_each_entry_safe(struct hk_queue, queue, &dev->cmd_pending_queues,
                            cmd_link)
      hk_cmd_pending_reap_queue(dev, queue, &ready);
   simple_mtx_unlock(&dev->cmd_pending_lock);

   /* Releasing a span takes vdm.lock; never do it under the pending lock. */
   list_for_each_entry_safe(struct hk_cmd_pending, p, &ready, link) {
      list_del(&p->link);
      hk_cmd_pending_release(dev, p, false);
   }
}

VkResult
hk_cmd_pending_begin(struct hk_device *dev,
                     const struct vk_queue_submit *submit, unsigned chunks,
                     struct agx_bo *extra_bo, struct hk_cmd_pending **out)
{
   *out = NULL;
   hk_cmd_pending_reap(dev);

   size_t count = hk_cmd_collect_pins(submit, NULL) + (extra_bo != NULL);
   if (count > UINT32_MAX ||
       count > (SIZE_MAX - sizeof(struct hk_cmd_pending)) /
                  sizeof(struct hk_cmd_pin))
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct hk_cmd_pending *p =
      vk_zalloc(&dev->vk.alloc, sizeof(*p) + count * sizeof(p->pins[0]), 8,
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

   size_t used = hk_cmd_collect_pins(submit, p->pins);
   if (extra_bo)
      hk_cmd_add_pin(p->pins, &used, HK_PIN_LARGE, extra_bo);
   assert(used == count);
   p->count = used;
   for (unsigned i = 0; i < p->count; ++i)
      hk_cmd_pin_ref(p->pins[i]);

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

   simple_mtx_lock(&dev->cmd_pending_lock);
   if (!p->attempted) {
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
      } else {
         assert(list_last_entry(&queue->cmd_pending, struct hk_cmd_pending,
                                link)->point < p->point);
      }
      list_addtail(&p->link, &queue->cmd_pending);
   }
   simple_mtx_unlock(&dev->cmd_pending_lock);

   if (!p->attempted)
      hk_cmd_pending_release(dev, p, false);
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

   for (unsigned i = 0; i < dev->cmd_syncobj_count; ++i)
      drmSyncobjDestroy(dev->dev.fd, dev->cmd_syncobjs[i]);
   dev->cmd_syncobj_count = 0;
}
