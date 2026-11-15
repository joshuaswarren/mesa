/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_device_memory.h"

#include "hk_device.h"
#include "hk_queue.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_instance.h"
#include "hk_physical_device.h"

#include "asahi/lib/agx_bo.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "vk_common_entrypoints.h"

#include "asahi/lib/agx_device.h"
#include "util/log.h"
#include "util/os_misc.h"

#include <errno.h>
#include <inttypes.h>
#include <string.h>
#include <sys/mman.h>

/* GPU memory is system memory that the kernel cannot evict or swap, and
 * heapBudget is only advisory, so the heap limit alone lets a process (or
 * several) take every free page and invoke the OOM killer on the desktop.
 * Refuse an allocation of 1 MiB or more that would leave less than
 * max(1 GiB, 2% of RAM) available. Applications that allocate per tensor
 * (tens of MiB per call) are covered. Smaller allocations always pass:
 * below the reserve every vkAllocateMemory would fail, including ones the
 * compositor needs.
 */
#define HK_MEMORY_RESERVE_MIN   (1ull << 30)
#define HK_MEMORY_CHECK_MIN     (1ull << 20)

static bool
hk_system_memory_fits(uint64_t size)
{
   uint64_t available, total;

   if (size < HK_MEMORY_CHECK_MIN)
      return true;

   /* If the kernel does not tell us, only the heap limit applies. */
   if (!os_get_available_system_memory(&available))
      return true;

   uint64_t reserve = HK_MEMORY_RESERVE_MIN;
   if (os_get_total_physical_memory(&total))
      reserve = MAX2(reserve, total / 50);

   return available >= size + reserve;
}

/* Supports opaque fd only */
const VkExternalMemoryProperties hk_opaque_fd_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
};

/* Supports opaque fd and dma_buf. */
const VkExternalMemoryProperties hk_dma_buf_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
};

/* Host allocations can be imported, nothing can be exported from them. */
const VkExternalMemoryProperties hk_host_ptr_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes = 0,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
};

static enum agx_bo_flags
hk_memory_type_flags(const VkMemoryType *type,
                     VkExternalMemoryHandleTypeFlagBits handle_types)
{
   unsigned flags = 0;

   if (handle_types)
      flags |= AGX_BO_SHARED | AGX_BO_SHAREABLE;

   return flags;
}

static void
hk_add_ext_bo_locked(struct hk_device *dev, struct agx_bo *bo)
{
   uint32_t id = bo->uapi_handle;

   unsigned count = util_dynarray_num_elements(&dev->external_bos.list,
                                               struct asahi_ccmd_submit_res);

   for (unsigned i = 0; i < count; i++) {
      struct asahi_ccmd_submit_res *p = util_dynarray_element(
         &dev->external_bos.list, struct asahi_ccmd_submit_res, i);

      if (p->res_id == id) {
         ++*util_dynarray_element(&dev->external_bos.counts, unsigned, i);
         return;
      }
   }

   struct asahi_ccmd_submit_res res = {
      .res_id = id,
      .flags = ASAHI_EXTRES_READ | ASAHI_EXTRES_WRITE,
   };
   util_dynarray_append(&dev->external_bos.list, res);
   util_dynarray_append_typed(&dev->external_bos.counts, unsigned, 1);
}

static void
hk_add_ext_bo(struct hk_device *dev, struct agx_bo *bo)
{
   if (dev->dev.is_virtio) {
      u_rwlock_wrlock(&dev->external_bos.lock);
      hk_add_ext_bo_locked(dev, bo);
      u_rwlock_wrunlock(&dev->external_bos.lock);
   }
}

static void
hk_remove_ext_bo_locked(struct hk_device *dev, struct agx_bo *bo)
{
   uint32_t id = bo->uapi_handle;
   unsigned count = util_dynarray_num_elements(&dev->external_bos.list,
                                               struct asahi_ccmd_submit_res);

   for (unsigned i = 0; i < count; i++) {
      struct asahi_ccmd_submit_res *p = util_dynarray_element(
         &dev->external_bos.list, struct asahi_ccmd_submit_res, i);

      if (p->res_id == id) {
         unsigned *ctr =
            util_dynarray_element(&dev->external_bos.counts, unsigned, i);
         if (!--*ctr) {
            *ctr = util_dynarray_pop(&dev->external_bos.counts, unsigned);
            *p = util_dynarray_pop(&dev->external_bos.list,
                                   struct asahi_ccmd_submit_res);
         }
         return;
      }
   }

   UNREACHABLE("BO not found");
}

static void
hk_remove_ext_bo(struct hk_device *dev, struct agx_bo *bo)
{
   if (dev->dev.is_virtio) {
      u_rwlock_wrlock(&dev->external_bos.lock);
      hk_remove_ext_bo_locked(dev, bo);
      u_rwlock_wrunlock(&dev->external_bos.lock);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetMemoryFdPropertiesKHR(VkDevice device,
                            VkExternalMemoryHandleTypeFlagBits handleType,
                            int fd,
                            VkMemoryFdPropertiesKHR *pMemoryFdProperties)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct agx_bo *bo;

   switch (handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      bo = agx_bo_import(&dev->dev, fd);
      if (bo == NULL)
         return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      break;
   default:
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   uint32_t type_bits = 0;
   for (unsigned t = 0; t < ARRAY_SIZE(pdev->mem_types); t++) {
      const unsigned flags =
         hk_memory_type_flags(&pdev->mem_types[t], handleType);
      if (!(flags & ~bo->flags))
         type_bits |= (1 << t);
   }

   pMemoryFdProperties->memoryTypeBits = type_bits;

   agx_bo_unreference(&dev->dev, bo);

   return VK_SUCCESS;
}

/* Memory priority 1.0 asks the kernel to release the memory after the VM's
 * other BOs when it reclaims idle memory; any other priority leaves it to the
 * kernel's default policy. */
static void
hk_set_memory_priority(struct hk_device *dev, struct hk_device_memory *mem,
                       float priority)
{
   const bool keep = priority >= 1.0f;
   if (keep == mem->keep_resident)
      return;

   int ret = agx_bo_set_resident(&dev->dev, mem->bo, keep);
   if (ret) {
      mesa_logw("memory priority %.2f: GEM_RESIDENCY failed: %s", priority,
                strerror(-ret));
      return;
   }

   mem->keep_resident = keep;
}

VKAPI_ATTR void VKAPI_CALL
hk_SetDeviceMemoryPriorityEXT(VkDevice device, VkDeviceMemory memory,
                              float priority)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, mem, memory);

   hk_set_memory_priority(dev, mem, priority);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetMemoryHostPointerPropertiesEXT(
   VkDevice device, VkExternalMemoryHandleTypeFlagBits handleType,
   const void *pHostPointer,
   VkMemoryHostPointerPropertiesEXT *pMemoryHostPointerProperties)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   if (handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT ||
       !agx_has_userptr(&dev->dev))
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   const uint64_t alignment =
      pdev->vk.properties.minImportedHostPointerAlignment;
   if ((uintptr_t)pHostPointer % alignment)
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   /* The application's own mapping is cacheable, and the GPU coherent with
    * it: offer the cached types. */
   uint32_t type_bits = 0;
   for (unsigned t = 0; t < pdev->mem_type_count; t++) {
      if (pdev->mem_types[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
         type_bits |= 1u << t;
   }

   pMemoryHostPointerProperties->memoryTypeBits = type_bits;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_device_memory *mem;
   VkResult result = VK_SUCCESS;

   const VkImportMemoryFdInfoKHR *fd_info =
      vk_find_struct_const(pAllocateInfo->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   const VkExportMemoryAllocateInfo *export_info =
      vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   const VkMemoryOpaqueCaptureAddressAllocateInfo *capture_info =
      vk_find_struct_const(pAllocateInfo->pNext,
                           MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO);
   const VkImportMemoryHostPointerInfoEXT *host_info = vk_find_struct_const(
      pAllocateInfo->pNext, IMPORT_MEMORY_HOST_POINTER_INFO_EXT);
   const VkMemoryPriorityAllocateInfoEXT *priority_info =
      vk_find_struct_const(pAllocateInfo->pNext,
                           MEMORY_PRIORITY_ALLOCATE_INFO_EXT);
   const VkMemoryType *type = &pdev->mem_types[pAllocateInfo->memoryTypeIndex];

   VkExternalMemoryHandleTypeFlagBits handle_types = 0;
   if (export_info != NULL)
      handle_types |= export_info->handleTypes;
   if (fd_info != NULL)
      handle_types |= fd_info->handleType;

   const unsigned flags = hk_memory_type_flags(type, handle_types);

   uint32_t alignment = 16384; /* Apple page size */

   struct hk_memory_heap *heap = &pdev->mem_heaps[type->heapIndex];
   if (pAllocateInfo->allocationSize > UINT64_MAX - (alignment - 1))
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   /* No size beyond the heap can be satisfied. Refuse it before creating a
    * BO: a request whose page rounding below wraps to 0 would otherwise
    * reach agx_bo_create() as a 0-byte BO and fail in the kernel. */
   if (!(fd_info && fd_info->handleType) &&
       pAllocateInfo->allocationSize > heap->size) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "allocationSize %" PRIu64 " exceeds heap %u "
                       "(%" PRIu64 " bytes)",
                       (uint64_t)pAllocateInfo->allocationSize,
                       type->heapIndex, (uint64_t)heap->size);
   }

   /* The spec requires allocationSize > 0 when nothing is imported
    * (VUID-VkMemoryAllocateInfo-allocationSize-07897), but CTS
    * cooperative_matrix.op_constant_null creates zero-sized buffers and
    * allocates their zero-sized requirements. Back such an allocation with
    * one page, so it can still be bound and mapped, rather than fail it with
    * VK_ERROR_OUT_OF_DEVICE_MEMORY. */
   const uint64_t aligned_size =
      align64(MAX2(pAllocateInfo->allocationSize, 1), alignment);
   uint64_t used = p_atomic_read(heap->used);
   if (used > heap->size || aligned_size > heap->size - used)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   mem = vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator,
                                 sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->map = NULL;
   if (fd_info && fd_info->handleType) {
      assert(
         fd_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT ||
         fd_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);

      mem->bo = agx_bo_import(&dev->dev, fd_info->fd);
      if (mem->bo == NULL) {
         result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
         goto fail_alloc;
      }
      assert(!(flags & ~mem->bo->flags));
   } else if (host_info && host_info->handleType) {
      /* The pointer and the size are multiples of
       * minImportedHostPointerAlignment (VUID-VkImportMemoryHostPointerInfoEXT-
       * pHostPointer-01749, VUID-VkMemoryAllocateInfo-allocationSize-01745).
       * Memory the kernel cannot pin for writing (a read-only mapping of a
       * file) is refused: nothing in the API keeps a buffer bound to it from
       * being a transfer destination or a storage buffer the GPU writes. An
       * application that only reads such memory on the GPU asks for it with
       * hk_host_memory_read_only. With hk_host_memory_import_read_only only
       * such a range is taken, and a range the kernel can pin for writing is
       * refused, also with hk_host_memory_read_only: the GPU then writes
       * through no host import (hk_cdm_alias_scope()).
       */
      assert(host_info->handleType ==
             VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT);
      const struct hk_instance *instance = hk_physical_device_instance(pdev);
      const enum agx_host_ptr_access access =
         instance->drirc.misc.host_memory_import_read_only
            ? AGX_HOST_PTR_READ_ONLY
         : instance->drirc.misc.host_memory_read_only ? AGX_HOST_PTR_ANY
                                                       : AGX_HOST_PTR_WRITABLE;
      mem->bo = agx_bo_import_host_ptr(&dev->dev, host_info->pHostPointer,
                                       pAllocateInfo->allocationSize, access);
      if (mem->bo == NULL) {
         const int err = errno;
         result = vk_errorf(dev,
                            err == ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                                          : VK_ERROR_INVALID_EXTERNAL_HANDLE,
                            "importing %" PRIu64 " bytes of host memory at %p: %s",
                            (uint64_t)pAllocateInfo->allocationSize,
                            host_info->pHostPointer, strerror(err));
         goto fail_alloc;
      }
   } else {
      enum agx_bo_flags flags = 0;
      if (handle_types)
         flags |= AGX_BO_SHAREABLE;
      /* A failed WSI export must not leave pinned backing in the BO cache.
       * Successful exports already acquire AGX_BO_SHARED. */
      if (dev->dev.chip == AGX_CHIP_G17P &&
          vk_find_struct_const(pAllocateInfo->pNext, WSI_MEMORY_ALLOCATE_INFO_MESA))
         flags |= AGX_BO_NO_CACHE;
      if (type->propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
         flags |= AGX_BO_WRITEBACK;

      /* A BO from the cache allocates nothing new. If the kernel purged it
       * meanwhile, one allocation can still pass the check. */
      if (!hk_system_memory_fits(aligned_size) &&
          !agx_bo_cache_has(&dev->dev, aligned_size, 64 * 1024, flags)) {
         result = vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                            "Not enough free system memory for %" PRIu64
                            " MiB", aligned_size >> 20);
         goto fail_alloc;
      }

      /* D3D12 promises 64 KiB alignment for heap addresses, and this costs
       * only virtual address space. */
      mem->bo = agx_bo_create(&dev->dev, aligned_size, 64 * 1024, flags,
                              "App memory");
      if (!mem->bo) {
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto fail_alloc;
      }
   }

   /* VK_EXT_descriptor_buffer capture/replay: image descriptors embed the
    * address of the image's memory, so memory allocated for capture/replay
    * also maps at an address in the capture/replay region, the captured one
    * on replay when that is one. Buffers and images bound to it take that
    * address (hk_device_memory_addr()); buffers with a capture/replay address
    * of their own keep it. */
   if ((mem->vk.alloc_flags &
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT) &&
       dev->vk.enabled_features.descriptorBufferCaptureReplay &&
       dev->dev.replay_heap_initialized) {
      const uint64_t fixed_addr =
         capture_info ? capture_info->opaqueCaptureAddress : 0;
      mem->replay_va = agx_va_alloc(
         &dev->dev, mem->bo->size, 64 * 1024,
         AGX_VA_CAPTURE_REPLAY | (fixed_addr ? AGX_VA_FIXED : 0), fixed_addr);
      if (mem->replay_va &&
          (agx_bo_bind(&dev->dev, mem->bo, mem->replay_va->addr,
                       mem->bo->size, 0,
                       DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE) ||
           agx_bo_bind(&dev->dev, mem->bo,
                       agx_rw_addr_to_ro(&dev->dev, mem->replay_va->addr),
                       mem->bo->size, 0, DRM_ASAHI_BIND_READ))) {
         agx_va_free(&dev->dev, mem->replay_va, true);
         mem->replay_va = NULL;
         agx_bo_unreference(&dev->dev, mem->bo);
         mem->bo = NULL;
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto fail_alloc;
      }
      /* An address this region cannot take was not captured with the
       * feature (on a device without it, it is the BO's own address): keep
       * buffer device address replay as it was, through the buffers' own
       * capture/replay addresses. */
      /* The memory is reached through two read-write addresses now. */
      if (mem->replay_va)
         p_atomic_inc(&dev->va_alias_resources);
   }

   if (mem->vk.alloc_flags &
       VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT) {
      mem->opaque_capture_address =
         mem->replay_va ? mem->replay_va->addr
         : capture_info && capture_info->opaqueCaptureAddress
            ? capture_info->opaqueCaptureAddress
            : mem->bo->va->addr;
   }

   /* Shadow map in case this is used for a sparse resident buffer */
   int ret = agx_bo_bind(&dev->dev, mem->bo,
                         agx_rw_addr_to_ro(&dev->dev, mem->bo->va->addr),
                         mem->bo->size, 0, DRM_ASAHI_BIND_READ);
   if (ret) {
      /* The bind fails once the VM carries a sticky error. */
      agx_bo_unreference(&dev->dev, mem->bo);
      mem->bo = NULL;
      vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   if (mem->bo->flags & (AGX_BO_SHAREABLE | AGX_BO_SHARED))
      hk_add_ext_bo(dev, mem->bo);

   if (priority_info)
      hk_set_memory_priority(dev, mem, priority_info->priority);

   uint64_t heap_used = p_atomic_add_return(heap->used, mem->bo->size);
   if (heap_used > heap->size) {
      hk_FreeMemory(device, hk_device_memory_to_handle(mem), pAllocator);
      return vk_errorf(device, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Out of heap memory");
   }

   if (fd_info && fd_info->handleType) {
      /* From the Vulkan spec:
       *
       *    "Importing memory from a file descriptor transfers ownership of
       *    the file descriptor from the application to the Vulkan
       *    implementation. The application must not perform any operations on
       *    the file descriptor after a successful import."
       *
       * Only consume it after heap accounting has accepted the allocation;
       * every failure must leave the caller's file descriptor open.
       */
      close(fd_info->fd);
   }

   *pMem = hk_device_memory_to_handle(mem);

   return VK_SUCCESS;

fail_alloc:
   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
hk_FreeMemory(VkDevice device, VkDeviceMemory _mem,
              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, mem, _mem);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   if (!mem)
      return;

   hk_device_quiesce_lost_submissions(dev);

   const VkMemoryType *type = &pdev->mem_types[mem->vk.memory_type_index];
   struct hk_memory_heap *heap = &pdev->mem_heaps[type->heapIndex];
   p_atomic_add(heap->used, -((int64_t)mem->bo->size));

   if (mem->bo->flags & (AGX_BO_SHAREABLE | AGX_BO_SHARED))
      hk_remove_ext_bo(dev, mem->bo);

   /* Freeing a mapped memory object unmaps it. The shared mapping goes with
    * the BO; an owned one belongs to this object alone. */
   if (mem->map_owned)
      agx_bo_unmap_owned(mem->bo, mem->map, mem->map_size);

   /* The BO may outlive this object in the BO cache. */
   hk_set_memory_priority(dev, mem, 0.5f);

   if (mem->replay_va) {
      agx_va_free(&dev->dev, mem->replay_va, true);
      p_atomic_dec(&dev->va_alias_resources);
   }

   agx_bo_unreference(&dev->dev, mem->bo);

   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_MapMemory2KHR(VkDevice device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                 void **ppData)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, mem, pMemoryMapInfo->memory);

   if (mem == NULL) {
      *ppData = NULL;
      return VK_SUCCESS;
   }

   const VkDeviceSize offset = pMemoryMapInfo->offset;
   const VkDeviceSize size = vk_device_memory_range(
      &mem->vk, pMemoryMapInfo->offset, pMemoryMapInfo->size);

   void *fixed_addr = NULL;
   if (pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) {
      const VkMemoryMapPlacedInfoEXT *placed_info = vk_find_struct_const(
         pMemoryMapInfo->pNext, MEMORY_MAP_PLACED_INFO_EXT);
      fixed_addr = placed_info->pPlacedAddress;
   }

   /* From the Vulkan spec version 1.0.32 docs for MapMemory:
    *
    *  * If size is not equal to VK_WHOLE_SIZE, size must be greater than 0
    *    assert(size != 0);
    *  * If size is not equal to VK_WHOLE_SIZE, size must be less than or
    *    equal to the size of the memory minus offset
    */
   assert(size > 0);
   assert(offset + size <= mem->bo->size);

   if (size != (size_t)size) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "requested size 0x%" PRIx64 " does not fit in %u bits",
                       size, (unsigned)(sizeof(size_t) * 8));
   }

   /* From the Vulkan 1.2.194 spec:
    *
    *    "memory must not be currently host mapped"
    */
   if (mem->map != NULL) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "Memory object already mapped.");
   }

   /* Host memory the kernel could only pin read-only has no mapping the
    * application may write through, and Vulkan has no read-only mapping. */
   if ((mem->bo->flags & AGX_BO_HOST_PTR) && (mem->bo->flags & AGX_BO_READONLY)) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "read-only host memory cannot be mapped");
   }

   /* Bound a placed mapping by the memory object, not the BO: the BO cache
    * may have handed us one up to twice as large, and the application has
    * reserved only mem->vk.size at this address.
    *
    * An exported allocation and its imports into this device are distinct
    * memory objects over one agx_bo. Each maps, unmaps, frees and reserves
    * independently, so each gets a mapping of its own rather than the BO's
    * shared one, which unmapping any of them would otherwise tear down
    * under the others. A BO only one memory object can reach keeps the
    * cached mapping. */
   if (mem->bo->flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE)) {
      mem->map = agx_bo_map_owned(mem->bo, fixed_addr, mem->vk.size,
                                  &mem->map_size);
      mem->map_owned = mem->map != NULL;
   }
   if (!mem->map_owned)
      mem->map = agx_bo_map_placed(mem->bo, fixed_addr, mem->vk.size);
   if (mem->map == NULL) {
      *ppData = NULL;
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "failed to map memory BO handle %u",
                       mem->bo->uapi_handle);
   }

   assert(!fixed_addr || mem->map == fixed_addr);
   *ppData = mem->map + offset;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_UnmapMemory2KHR(VkDevice device,
                   const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(hk_device_memory, mem, pMemoryUnmapInfo->memory);

   if (mem == NULL)
      return VK_SUCCESS;

   if (mem->map_owned) {
      if (pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT) {
         VK_FROM_HANDLE(hk_device, dev, device);

         /* Exactly this object's mapping, as below. */
         void *err = mmap(mem->map, mem->map_size, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
         if (err == MAP_FAILED) {
            return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                             "Failed to map over original mapping");
         }
      } else {
         agx_bo_unmap_owned(mem->bo, mem->map, mem->map_size);
      }
      mem->map = NULL;
      mem->map_size = 0;
      mem->map_owned = false;
      return VK_SUCCESS;
   }

   /* Imported host memory stays mapped where the application put it
    * (VUID-VkMemoryUnmapInfo-flags-09579 excludes reserving it). */
   if (mem->bo->flags & AGX_BO_HOST_PTR) {
      mem->map = NULL;
      return VK_SUCCESS;
   }

   if (pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT) {
      VK_FROM_HANDLE(hk_device, dev, device);

      /* Reserve exactly what was mapped. A placed map covers the
       * VkDeviceMemory's allocationSize, not the cached BO's size, and
       * PROT_NONE'ing the difference lands on address space the application
       * reserved for itself -- the guard pages dEQP-VK.memory.map_placed.*
       * fills and reads back. */
      void *err = mmap(mem->bo->_map, agx_bo_map_size(mem->bo), PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (err == MAP_FAILED) {
         return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                          "Failed to map over original mapping");
      }
      mem->bo->_map = NULL;
      mem->bo->_map_size = 0;
   } else {
      if (mem->bo->_map) {
         munmap(mem->bo->_map, agx_bo_map_size(mem->bo));
         mem->bo->_map = NULL;
         mem->bo->_map_size = 0;
      }
   }

   mem->map = NULL;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_FlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                           const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_InvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory _mem,
                             VkDeviceSize *pCommittedMemoryInBytes)
{
   VK_FROM_HANDLE(hk_device_memory, mem, _mem);

   *pCommittedMemoryInBytes = mem->bo->size;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR *pGetFdInfo,
                  int *pFD)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, memory, pGetFdInfo->memory);

   switch (pGetFdInfo->handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      *pFD = agx_bo_export(&dev->dev, memory->bo);
      if (*pFD < 0 && dev->dev.chip == AGX_CHIP_G17P)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      return VK_SUCCESS;
   default:
      assert(!"unsupported handle type");
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "vkGetMemoryFdKHR: unsupported handle type 0x%x",
                       pGetFdInfo->handleType);
   }
}

VKAPI_ATTR uint64_t VKAPI_CALL
hk_GetDeviceMemoryOpaqueCaptureAddress(
   UNUSED VkDevice device,
   const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   VK_FROM_HANDLE(hk_device_memory, memory, pInfo->memory);

   return memory->opaque_capture_address;
}
