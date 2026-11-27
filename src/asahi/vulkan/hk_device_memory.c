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
#include "vk_common_entrypoints.h"

#include "asahi/lib/agx_device.h"
#include "util/log.h"
#include "util/os_misc.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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

static bool
hk_allow_bo_alloc(size_t size, void *data)
{
   bool *mem_rejected = data;

   /* agx_bo_create_checked() asks again after evicting the BO cache, so
    * record only the latest answer. */
   *mem_rejected = !hk_system_memory_fits(size);
   return !*mem_rejected;
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
       !agx_has_host_import(&dev->dev))
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

/* Track host-pointer imports per import on every chip except M5 when the
 * static per-device rule is active (ASAHI_M5_HOST_ALIAS_STATIC=1 or no
 * aliasing-capable features). */
static inline bool
hk_host_import_tracks_alias(const struct hk_device *dev)
{
   return !agx_chip_g17_m5(dev->dev.chip) || dev->host_alias.dynamic;
}

/* The class of the mappings over [pointer, pointer + size) in
 * /proc/self/maps (hk_host_maps_classify_fd()); HK_HOST_MAP_OTHER when the
 * file cannot be read. */
static enum hk_host_map_class
hk_host_maps_classify_self(const void *pointer, uint64_t size)
{
   const uintptr_t start = (uintptr_t)pointer;
   if (size == 0 || size > UINTPTR_MAX - start)
      return HK_HOST_MAP_OTHER;

   const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return HK_HOST_MAP_OTHER;
   const enum hk_host_map_class cls =
      hk_host_maps_classify_fd(fd, start, start + (uintptr_t)size);
   close(fd);
   return cls;
}

/* Enter a host import into the device's set so va_alias_resources (and
 * alias_live on M5 with host_alias.dynamic) only counts imports that may
 * alias another GPU virtual address (hk_host_import_aliases()). A
 * GPU-writable import of pages that are all private anonymous memory of this
 * process (/proc/self/maps), and that no other live import overlaps, has its
 * pages at no other GPU address. A read-only one needs only private mappings:
 * the GPU writes none of its pages through it, and writes them through
 * another address only through a writable import of a shared mapping, which
 * aliases itself, or of the same pages, which overlaps it. An import whose
 * maps cannot be read aliases. The mappings are read before the kernel pins
 * the pages (before_pin) and again now, and the worse of the two counts, so
 * an application that replaces the mapping meanwhile is not taken at its
 * word. What the maps cannot show: pages a private mapping holds that are
 * pinned already, see hk_host_alias.h on mremap.
 *
 * Costs two reads of /proc/self/maps up to the lines that cover the range.
 * The imports that began to alias count in va_alias_resources (and
 * alias_live via hk_device_alias_add() on M5) before the import is returned. */
static VkResult
hk_host_import_track(struct hk_device *dev, struct hk_device_memory *mem,
                     const void *pointer, uint64_t size,
                     enum hk_host_map_class before_pin)
{
   if (!hk_host_import_tracks_alias(dev)) {
      /* Under the per-device rule nothing is tracked; say the import
       * happened, so a trace shows whether an application imports at all. */
      if (dev->host_alias.trace)
         fprintf(stderr, "M5_HOST_ALIAS import %p+%#" PRIx64 " (per-device rule)\n",
                 pointer, size);
      return VK_SUCCESS;
   }

   const uintptr_t start = (uintptr_t)pointer;
   const uintptr_t end =
      size <= UINTPTR_MAX - start ? start + (uintptr_t)size : UINTPTR_MAX;
   const enum hk_host_map_class cls =
      MAX2(before_pin, hk_host_maps_classify_self(pointer, size));

   const bool gpu_writes = !(mem->bo->flags & AGX_BO_READONLY);
   const struct hk_host_import import = {
      .start = start,
      .end = end,
      .owner = mem,
      .gpu_writes = gpu_writes,
      .maps_alias = hk_host_import_maps_alias(gpu_writes, cls),
   };

   /* The counts change with the set, under its lock, so a free of an
    * overlapping import never subtracts ahead of this add
    * (hk_device_alias_add()). The patch runs after the lock is released. */
   simple_mtx_lock(&dev->host_alias.lock);
   const int began = hk_host_imports_add(&dev->host_alias.imports, &import);
   bool first = false;
   if (began > 0) {
      p_atomic_add(&dev->va_alias_resources, (int32_t)began);
      first = hk_device_alias_add(dev, (uint32_t)began);
   }
   simple_mtx_unlock(&dev->host_alias.lock);
   if (began < 0)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->host_alias_tracked = true;
   if (first)
      hk_device_alias_began(dev, HK_ALIAS_CAUSE_IMPORT);

   if (dev->host_alias.trace)
      fprintf(stderr, "M5_HOST_ALIAS import %p+%#" PRIx64 " writes=%u maps=%u "
              "began=%d live=%u\n", pointer, size, gpu_writes, (unsigned)cls,
              began, p_atomic_read(&dev->alias_live));
   return VK_SUCCESS;
}

static void
hk_host_import_untrack(struct hk_device *dev, struct hk_device_memory *mem)
{
   if (!mem->host_alias_tracked)
      return;

   simple_mtx_lock(&dev->host_alias.lock);
   const unsigned stopped =
      hk_host_imports_remove(&dev->host_alias.imports, mem);
   if (stopped > 0) {
      p_atomic_add(&dev->va_alias_resources, -(int32_t)stopped);
      hk_device_alias_put(dev, stopped);
   }
   simple_mtx_unlock(&dev->host_alias.lock);
   mem->host_alias_tracked = false;

   if (dev->host_alias.trace)
      fprintf(stderr, "M5_HOST_ALIAS free %p stopped=%u live=%u\n",
              (void *)mem, stopped, p_atomic_read(&dev->alias_live));
}

/* Memory mapped again in the capture/replay region (replay_va) counts from
 * its mapping to its release (hk_device_va_alias_get()). */
static void
hk_memory_release_replay_va(struct hk_device *dev,
                            struct hk_device_memory *mem)
{
   if (!mem->replay_va)
      return;
   agx_va_free(&dev->dev, mem->replay_va, true);
   mem->replay_va = NULL;
   hk_device_va_alias_put(dev);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_device_memory *mem;
   VkResult result = VK_SUCCESS;

   const VkImportMemoryFdInfoKHR *fd_info = NULL;
   const VkExportMemoryAllocateInfo *export_info = NULL;
   const VkMemoryOpaqueCaptureAddressAllocateInfo *capture_info = NULL;
   const VkImportMemoryHostPointerInfoEXT *host_info = NULL;
   const VkMemoryPriorityAllocateInfoEXT *priority_info = NULL;
   bool wsi_alloc = false;

   if (unlikely(pAllocateInfo->pNext != NULL)) {
      vk_foreach_struct_const(sType, ext, pAllocateInfo->pNext) {
         switch ((uint32_t)sType) {
         case VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR:
            if (!fd_info)
               fd_info = ext;
            break;
         case VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO:
            if (!export_info)
               export_info = ext;
            break;
         case VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO:
            if (!capture_info)
               capture_info = ext;
            break;
         case VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT:
            if (!host_info)
               host_info = ext;
            break;
         case VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT:
            if (!priority_info)
               priority_info = ext;
            break;
         case VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA:
            wsi_alloc = true;
            break;
         default:
            break;
         }
      }
   }
   const VkMemoryType *type = &pdev->mem_types[pAllocateInfo->memoryTypeIndex];
   /* The mappings over a host import before its pages are pinned
    * (hk_host_import_track()). */
   enum hk_host_map_class host_maps_before_pin = HK_HOST_MAP_OTHER;

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

      /* Every byte of allocationSize must be backed by the imported object.
       * An image bound to the memory is checked against the BO
       * (hk_image_plane_bind()), but a buffer may use the whole of
       * allocationSize, which would then run past the BO. */
      if (mem->bo->size < pAllocateInfo->allocationSize) {
         result = vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                            "allocationSize %" PRIu64 " exceeds the imported "
                            "object (%" PRIu64 " bytes)",
                            (uint64_t)pAllocateInfo->allocationSize,
                            (uint64_t)mem->bo->size);
         agx_bo_unreference(&dev->dev, mem->bo);
         mem->bo = NULL;
         goto fail_alloc;
      }
      assert(!(flags & ~mem->bo->flags));
      mem->alias_import = true;
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
       * through no host import (hk_cdm_alias_scope()). Otherwise,
       * hk_host_import_track() decides below whether this import may alias.
       */
      assert(host_info->handleType ==
             VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT);
      const struct hk_instance *instance = hk_physical_device_instance(pdev);
      const enum agx_host_ptr_access access =
         instance->drirc.misc.host_memory_import_read_only
            ? AGX_HOST_PTR_READ_ONLY
         : instance->drirc.misc.host_memory_read_only ? AGX_HOST_PTR_ANY
                                                       : AGX_HOST_PTR_WRITABLE;
      if (hk_host_import_tracks_alias(dev))
         host_maps_before_pin = hk_host_maps_classify_self(
            host_info->pHostPointer, pAllocateInfo->allocationSize);
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
      if (!hk_host_import_tracks_alias(dev))
         mem->alias_import = access != AGX_HOST_PTR_READ_ONLY;
   } else {
      enum agx_bo_flags flags = 0;
      if (handle_types)
         flags |= AGX_BO_SHAREABLE;
      /* A failed WSI export must not leave pinned backing in the BO cache.
       * Successful exports already acquire AGX_BO_SHARED. */
      if (!dev->dev.is_virtio && wsi_alloc)
         flags |= AGX_BO_NO_CACHE;
      if (type->propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
         flags |= AGX_BO_WRITEBACK;

      /* A BO from the cache allocates nothing new. Fetch from the cache once
       * and run hk_system_memory_fits() only on a real miss (including after
       * discarding purged entries), avoiding both a duplicate cache-lock walk
       * and per-allocation /proc/meminfo reads. */
      bool mem_rejected = false;
      /* D3D12 promises 64 KiB alignment for heap addresses, and this costs
       * only virtual address space. */
      mem->bo = agx_bo_create_checked(&dev->dev, aligned_size, 64 * 1024, flags,
                                      "App memory", hk_allow_bo_alloc,
                                      &mem_rejected);
      if (!mem->bo) {
         if (mem_rejected) {
            result = vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                               "Not enough free system memory for %" PRIu64
                               " MiB", aligned_size >> 20);
         } else {
            result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         }
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
      /* The range follows the requested size, not the BO's: a BO from the
       * cache can be up to twice as large, so its size differs between
       * capture and replay, and a larger replayed range would cover the next
       * captured address. Bindings only reach offsets below that size. */
      mem->replay_va = agx_va_alloc(
         &dev->dev, aligned_size, 64 * 1024,
         AGX_VA_CAPTURE_REPLAY | (fixed_addr ? AGX_VA_FIXED : 0), fixed_addr);
      if (mem->replay_va &&
          (agx_bo_bind(&dev->dev, mem->bo, mem->replay_va->addr,
                       aligned_size, 0,
                       DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE) ||
           agx_bo_bind(&dev->dev, mem->bo,
                       agx_rw_addr_to_ro(&dev->dev, mem->replay_va->addr),
                       aligned_size, 0, DRM_ASAHI_BIND_READ))) {
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
         hk_device_va_alias_get(dev, HK_ALIAS_CAUSE_CAPTURE_REPLAY);
   }

   if (mem->vk.alloc_flags &
       VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT) {
      mem->opaque_capture_address =
         mem->replay_va ? mem->replay_va->addr
         : capture_info && capture_info->opaqueCaptureAddress
            ? capture_info->opaqueCaptureAddress
            : mem->bo->va->addr;
   }

   /* Shadow map in case this is used for a sparse resident buffer. The mapping
    * persists with the BO across userspace cache reuse, so cached allocations
    * skip the redundant VM_BIND ioctl. */
   int ret = agx_bo_ensure_ro_shadow(&dev->dev, mem->bo);
   if (ret) {
      /* The bind fails once the VM carries a sticky error. */
      hk_memory_release_replay_va(dev, mem);
      agx_bo_unreference(&dev->dev, mem->bo);
      mem->bo = NULL;
      vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   if (host_info && host_info->handleType) {
      result = hk_host_import_track(dev, mem, host_info->pHostPointer,
                                    mem->bo->size, host_maps_before_pin);
      if (result != VK_SUCCESS) {
         hk_memory_release_replay_va(dev, mem);
         agx_bo_unreference(&dev->dev, mem->bo);
         mem->bo = NULL;
         vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
         return result;
      }
   }

   /* An fd or dma-buf import has an address of its own, and another import,
    * of this device or of another device or process, may map the same pages:
    * it counts for its lifetime (hk_cdm_alias_scope()). An export reaches
    * other devices and processes only across a submission boundary, after
    * the memory-scoped flush that ends every command; brought back into this
    * device it is an import, which counts here. */
   if (fd_info && fd_info->handleType) {
      mem->alias_fd = true;
      hk_device_alias_get(dev, 1, HK_ALIAS_CAUSE_FD);
   }

   if (mem->bo->flags & (AGX_BO_SHAREABLE | AGX_BO_SHARED))
      hk_add_ext_bo(dev, mem->bo);

   if (mem->alias_import)
      p_atomic_inc(&dev->va_alias_resources);

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

/* The CPU mapping that host image copies of this memory go through.
 *
 * It is this object's own mapping of the whole BO (agx_bo_map_owned()), made
 * at the first copy and kept until vkFreeMemory. The application's
 * vkMapMemory2 and vkUnmapMemory2 of the memory, which may run on other
 * threads while copies do, map, move and unmap only the BO's shared mapping
 * or their own, so they never touch the one a copy is using.
 *
 * Host image copies need no external synchronization on the memory, so two
 * threads can make the mapping at once: the first one installed is kept and
 * the other is unmapped. Host memory stays where the application put it, and
 * a backend that cannot map a BO twice has only the shared mapping; copies
 * use the BO's shared mapping there. NULL if the BO cannot be mapped.
 */
void *
hk_device_memory_copy_map(struct hk_device_memory *mem)
{
   void *map = p_atomic_read(&mem->copy_map);
   if (map)
      return map;

   if (!mem->bo->dev->ops.bo_mmap_independent ||
       (mem->bo->flags & AGX_BO_HOST_PTR)) {
      map = agx_bo_map(mem->bo);
   } else {
      size_t size;
      map = agx_bo_map_owned(mem->bo, NULL, 0, &size);
      if (map == NULL)
         return NULL;

      void *installed = p_atomic_cmpxchg(&mem->copy_map, NULL, map);
      if (installed != NULL) {
         agx_bo_unmap_owned(mem->bo, map, size);
         return installed;
      }
   }

   return map;
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
   if (mem->copy_map)
      agx_bo_unmap_owned(mem->bo, mem->copy_map, mem->bo->size);

   /* The BO may outlive this object in the BO cache. */
   hk_set_memory_priority(dev, mem, 0.5f);

   hk_memory_release_replay_va(dev, mem);

   if (mem->alias_import)
      p_atomic_dec(&dev->va_alias_resources);

   /* No work that may still run reaches this memory any more. */
   hk_host_import_untrack(dev, mem);
   if (mem->alias_fd)
      hk_device_alias_put(dev, 1);

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
      if (mem->bo->_map) {
         void *err = mmap(mem->bo->_map, agx_bo_map_size(mem->bo), PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
         if (err == MAP_FAILED) {
            return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                             "Failed to map over original mapping");
         }
         mem->bo->_map = NULL;
         mem->bo->_map_size = 0;
      }
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
      if (*pFD < 0)
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
