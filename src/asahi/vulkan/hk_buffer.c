/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_buffer.h"
#include "hk_acceleration_structure.h"

#include <errno.h>

#include "agx_bo.h"
#include "agx_device.h"
#include "hk_device.h"
#include "hk_queue.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "util/u_atomic.h"

VkResult
hk_update_physical_sparse_map(struct hk_device *dev, uint64_t address,
                               uint64_t size, bool nonresident)
{
   struct agx_bo *bo = dev->physical_sparse_map;
   if (!bo || !size)
      return VK_SUCCESS;
   const uint64_t base = dev->dev.main_heap_base;
   const uint64_t span = dev->dev.main_heap_size;
   if (address < base || address - base > span ||
       size > span - (address - base) ||
       (address | size) & (AIL_PAGESIZE - 1))
      return vk_error(dev, VK_ERROR_UNKNOWN);

   const uint64_t first = (address - base) / AIL_PAGESIZE;
   const uint64_t last = first + size / AIL_PAGESIZE;
   uint32_t *header = agx_bo_map(bo);
   uint32_t *map = header + 4;
   int32_t changed = 0;
   for (uint64_t page = first; page < last;) {
      const unsigned count = MIN2(last - page, 32 - (page & 31));
      const uint32_t mask = BITFIELD_MASK(count) << (page & 31);
      if (nonresident) {
         uint32_t old = __atomic_fetch_or(&map[page >> 5], mask,
                                           __ATOMIC_RELAXED);
         changed += util_bitcount(~old & mask);
      } else {
         uint32_t old = __atomic_fetch_and(&map[page >> 5], ~mask,
                                            __ATOMIC_RELAXED);
         changed -= util_bitcount(old & mask);
      }
      page += count;
   }
   __atomic_fetch_add(&header[0], changed, __ATOMIC_RELEASE);
   const uint64_t offset = 16 + (first >> 5) * sizeof(uint32_t);
   const uint64_t end = 16 + DIV_ROUND_UP(last, 32) * sizeof(uint32_t);
   if (agx_bo_sync_for_device(bo, offset, end - offset) ||
       agx_bo_sync_for_device(bo, 0, 16))
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   return VK_SUCCESS;
}

uint32_t
hk_buffer_sparse_texel_word(const struct hk_buffer *buffer,
                            unsigned blocksize_B)
{
   /* A non-power-of-two element size (the emulated RGB32 layout) takes the
    * raw load path, which claims residency anyway. */
   if (!buffer || !buffer->sparse_map ||
       !util_is_power_of_two_nonzero(blocksize_B))
      return 0;

   const uint64_t map_addr = hk_buffer_sparse_map_addr(buffer);
   const uint64_t map_page = map_addr / AIL_PAGESIZE;
   assert((map_addr % AIL_PAGESIZE) == 0);
   /* The packed field leaves 28 bits for the map page.  Fail loudly rather
    * than silently truncating if the VA layout ever outgrows it; a wrong map
    * address would read residency from arbitrary memory. */
   if (map_page >= (1u << (32 - AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT))) {
      assert(!"sparse residency map outside the packed VA range");
      return 0;
   }
   return (uint32_t)(map_page << AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT) |
          (uint32_t)util_logbase2(blocksize_B);
}

/* The sparse buffer that holds address. Call with hk_device::sparse_buffers.lock
 * held. */
static const struct hk_buffer *
hk_sparse_buffer_at_locked(struct hk_device *dev, uint64_t address)
{
   list_for_each_entry(struct hk_buffer, buffer, &dev->sparse_buffers.list,
                       sparse_link) {
      if (address - buffer->va->addr < buffer->vk.size)
         return buffer;
   }
   return NULL;
}

uint32_t
hk_sparse_texel_word_at(struct hk_device *dev, uint64_t address,
                        unsigned blocksize_B)
{
   /* A buffer joins the list before vkCreateBuffer() returns it and leaves
    * it in vkDestroyBuffer(), so a buffer an application can name an address
    * of is already counted. */
   if (!p_atomic_read(&dev->sparse_buffers.count))
      return 0;

   simple_mtx_lock(&dev->sparse_buffers.lock);
   const uint32_t word = hk_buffer_sparse_texel_word(
      hk_sparse_buffer_at_locked(dev, address), blocksize_B);
   simple_mtx_unlock(&dev->sparse_buffers.lock);
   return word;
}

uint32_t
hk_sparse_map_page_at(struct hk_device *dev, uint64_t address)
{
   /* Only M5 lowers storage loads through this word (get_buffer_address()).
    * Elsewhere it is the address format's byte offset and must stay zero. */
   if (!agx_chip_g17_m5(dev->dev.chip) ||
       !p_atomic_read(&dev->sparse_buffers.count))
      return 0;

   uint32_t page = 0;
   simple_mtx_lock(&dev->sparse_buffers.lock);
   const struct hk_buffer *buffer = hk_sparse_buffer_at_locked(dev, address);
   if (buffer && buffer->sparse_map) {
      const uint64_t map_addr = hk_buffer_sparse_map_addr(buffer);
      assert((map_addr % AIL_PAGESIZE) == 0);
      assert((map_addr / AIL_PAGESIZE) <= UINT32_MAX);
      page = map_addr / AIL_PAGESIZE;
   }
   simple_mtx_unlock(&dev->sparse_buffers.lock);
   return page;
}

bool
hk_sparse_address_is_sparse(struct hk_device *dev, uint64_t address)
{
   if (!p_atomic_read(&dev->sparse_buffers.count))
      return false;

   simple_mtx_lock(&dev->sparse_buffers.lock);
   const bool sparse = hk_sparse_buffer_at_locked(dev, address) != NULL;
   simple_mtx_unlock(&dev->sparse_buffers.lock);
   return sparse;
}

static bool
hk_buffer_in_sparse_list(const struct hk_buffer *buffer)
{
   return buffer->sparse_map != NULL ||
          (buffer->va != NULL &&
           (buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT));
}

static uint32_t
hk_get_buffer_alignment(const struct hk_physical_device *pdev,
                        VkBufferUsageFlags2KHR usage_flags,
                        VkBufferCreateFlags create_flags)
{
   uint32_t alignment = 16;

   if (usage_flags & VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR)
      alignment = MAX2(alignment, HK_MIN_UBO_ALIGNMENT);

   if (usage_flags & VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR)
      alignment = MAX2(alignment, HK_MIN_SSBO_ALIGNMENT);

   if (usage_flags & (VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT_KHR |
                      VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT_KHR))
      alignment = MAX2(alignment, HK_MIN_TEXEL_BUFFER_ALIGNMENT);

   /* Descriptor buffer addresses must meet descriptorBufferOffsetAlignment */
   if (usage_flags & (VK_BUFFER_USAGE_2_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                      VK_BUFFER_USAGE_2_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                      VK_BUFFER_USAGE_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT))
      alignment = MAX2(alignment, HK_MIN_UBO_ALIGNMENT);

   if (create_flags & (VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
                       VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT))
      alignment = MAX2(alignment, 16384);

   /* D3D12 heaps and tiles sit on 64 KiB boundaries, and vkd3d-proton backs
    * its heaps with sparse-capable buffers. */
   if (create_flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT)
      alignment = MAX2(alignment, 64 * 1024);

   return alignment;
}

static void
hk_get_buffer_replay_addrs(const VkBufferCreateInfo *pCreateInfo,
                           uint64_t *bda_addr_out,
                           uint64_t *sparse_map_addr_out)
{
   uint64_t addr = 0;
   uint64_t sparse_map_addr = 0;
   vk_foreach_struct_const(sType, ext, pCreateInfo->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO: {
         const VkBufferOpaqueCaptureAddressCreateInfo *bda = ext;
         if (bda->opaqueCaptureAddress != 0) {
            assert(addr == 0 || bda->opaqueCaptureAddress == addr);
            addr = bda->opaqueCaptureAddress;
         }
         break;
      }

      case VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT: {
         const VkBufferDeviceAddressCreateInfoEXT *bda = ext;
         if (bda->deviceAddress != 0) {
            assert(addr == 0 || bda->deviceAddress == addr);
            addr = bda->deviceAddress;
         }
         break;
      }

      case VK_STRUCTURE_TYPE_OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT: {
         /* VK_EXT_descriptor_buffer replay: hk_GetBufferOpaqueCapture-
          * DescriptorDataEXT() recorded the buffer's VA and, for sparse-
          * resident buffers, its residency map's VA. */
         if (!(pCreateInfo->flags &
               VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT))
            break;
         const VkOpaqueCaptureDescriptorDataCreateInfoEXT *info = ext;
         if (!info->opaqueCaptureDescriptorData)
            break;
         struct hk_buffer_capture capture;
         memcpy(&capture, info->opaqueCaptureDescriptorData, sizeof(capture));
         if (capture.va != 0) {
            assert(addr == 0 || capture.va == addr);
            addr = capture.va;
         }
         if (capture.sparse_map_va != 0) {
            assert(sparse_map_addr == 0 ||
                   capture.sparse_map_va == sparse_map_addr);
            sparse_map_addr = capture.sparse_map_va;
         }
         break;
      }

      default:
         break;
      }
   }

   *bda_addr_out = addr;
   *sparse_map_addr_out = sparse_map_addr;
}

VkResult
hk_bind_scratch(struct hk_device *dev, struct agx_va *va, unsigned offset_B,
                size_t size_B)
{
   uint64_t addr = va->addr + offset_B;
   uint32_t flags = DRM_ASAHI_BIND_READ | agx_fill_bind_flags(&dev->dev);

   /* The kernel refuses a fill it cannot back (ENOMEM; on M5 also its
    * hypervisor page-table budget): that is device memory exhaustion. */
   /* Map read-write scratch to the primary (bottom half) VA range */
   int ret = agx_bo_bind(&dev->dev, dev->sparse_fill, addr, size_B, 0,
                         flags | DRM_ASAHI_BIND_WRITE);
   if (ret)
      return errno == ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_UNKNOWN;

   /* Map read-only scratch to the secondary (top half) VA range */
   ret = agx_bo_bind(&dev->dev, dev->dev.zero_bo,
                     addr + dev->dev.sparse_ro_offset, size_B, 0, flags);
   if (ret)
      return errno == ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_UNKNOWN;

   return VK_SUCCESS;
}

VkResult
hk_bind_unbound_sparse_buffer(struct hk_device *dev, struct agx_va *va,
                              size_t size_B)
{
   if (!hk_sparse_buffer_unmapped(&dev->dev))
      return hk_bind_scratch(dev, va, 0, size_B);

   int ret = agx_bo_bind(&dev->dev, dev->dev.zero_bo,
                         va->addr + dev->dev.sparse_ro_offset, size_B, 0,
                         DRM_ASAHI_BIND_READ | agx_fill_bind_flags(&dev->dev));
   if (ret)
      return errno == ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_UNKNOWN;

   return VK_SUCCESS;
}

/* M5 publishes every scratch-filled sparse page through the hypervisor,
 * which allocates one page-table frame from its own heap per 32 MiB of VA and
 * never frees it. Keep each device within its advertised
 * sparseAddressSpaceSize; beyond it creation fails cleanly.
 */
VkResult
hk_reserve_sparse_scratch(struct hk_device *dev, uint64_t size_B)
{
   if (!agx_chip_g17_m5(dev->dev.chip))
      return VK_SUCCESS;

   uint64_t used = p_atomic_add_return(&dev->sparse_scratch_B, size_B);
   if (used > HK_M5_SPARSE_ADDR_SPACE_SIZE) {
      p_atomic_add(&dev->sparse_scratch_B, -(int64_t)size_B);
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "M5 sparse address space exhausted");
   }
   return VK_SUCCESS;
}

void
hk_release_sparse_scratch(struct hk_device *dev, uint64_t size_B)
{
   if (size_B)
      p_atomic_add(&dev->sparse_scratch_B, -(int64_t)size_B);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateBuffer(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_buffer *buffer;

   if (pCreateInfo->size > HK_MAX_BUFFER_SIZE)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   buffer =
      vk_buffer_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*buffer));
   if (!buffer)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT)
      p_atomic_set(&dev->sparse_used, 1);

   if (buffer->vk.size > 0 &&
       (buffer->vk.create_flags &
        (VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
         VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT))) {

      const uint32_t alignment = hk_get_buffer_alignment(
         hk_device_physical(dev), buffer->vk.usage, buffer->vk.create_flags);
      assert(alignment >= 16384);
      uint64_t vma_size_B = align64(buffer->vk.size, alignment);

      /* A buffer with its own VA replays it for descriptor buffers too. */
      const bool desc_capture_replay =
         buffer->vk.create_flags &
         VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT;
      const bool bda_capture_replay =
         desc_capture_replay ||
         (buffer->vk.create_flags &
          VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT);

      enum agx_va_flags flags = 0;
      uint64_t bda_fixed_addr = 0;
      uint64_t sparse_map_fixed_addr = 0;
      if (bda_capture_replay) {
         if (dev->dev.replay_heap_initialized)
            flags |= AGX_VA_CAPTURE_REPLAY;
         hk_get_buffer_replay_addrs(pCreateInfo, &bda_fixed_addr,
                                    &sparse_map_fixed_addr);
         if (bda_fixed_addr != 0)
            flags |= AGX_VA_FIXED;
      }

      buffer->va =
         agx_va_alloc(&dev->dev, vma_size_B, alignment, flags, bda_fixed_addr);

      if (!buffer->va) {
         vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
         /* A replay address that is already taken is the application's
          * error to handle, not an out-of-memory condition. */
         if (bda_fixed_addr != 0)
            return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                             "Capture/replay address %#llx is in use",
                             (unsigned long long)bda_fixed_addr);
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "Sparse VMA allocation failed");
      }
      buffer->vk.device_address = buffer->va->addr;

      /* Binding-only sparse buffers must be fully bound before GPU use.
       * On M5, reserve their VA without eagerly populating both views with
       * scratch pages through SPTM. Partially resident buffers still need
       * safe unbound accesses.
       */
      const bool binding_only =
         agx_chip_g17_m5(dev->dev.chip) &&
         (buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) &&
         !(buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT);
      if (!binding_only) {
         VkResult result = hk_reserve_sparse_scratch(dev, vma_size_B);
         if (result != VK_SUCCESS) {
            agx_va_free(&dev->dev, buffer->va, true);
            buffer->va = NULL;
            vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
            return result;
         }
         buffer->sparse_scratch_B =
            agx_chip_g17_m5(dev->dev.chip) ? vma_size_B : 0;
         result = hk_bind_unbound_sparse_buffer(dev, buffer->va, vma_size_B);
         if (result != VK_SUCCESS) {
            hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
            agx_va_free(&dev->dev, buffer->va, true);
            buffer->va = NULL;
            vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
            return result;
         }
      }

      if (agx_chip_g17p_or_m5(dev->dev.chip) &&
          (buffer->vk.create_flags &
           VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT)) {
         const uint64_t entries = DIV_ROUND_UP(vma_size_B, AIL_PAGESIZE);

         if (entries > UINT32_MAX ||
             DIV_ROUND_UP(entries, 32) >
                (SIZE_MAX - sizeof(struct hk_sparse_buffer_map_header)) /
                   sizeof(uint32_t)) {
            hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
            agx_va_free(&dev->dev, buffer->va, true);
            buffer->va = NULL;
            vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
            return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                             "Sparse buffer residency map is too large");
         }

         const uint64_t map_size_B = align64(
            sizeof(struct hk_sparse_buffer_map_header) +
               (DIV_ROUND_UP(entries, 32) * sizeof(uint32_t)),
            AIL_PAGESIZE);

         buffer->sparse_map = agx_bo_create(
            &dev->dev, map_size_B, AIL_PAGESIZE,
            AGX_BO_WRITEBACK | AGX_BO_ZERO_ON_REUSE,
            "Vulkan sparse buffer residency map");
         if (!buffer->sparse_map || !agx_bo_map(buffer->sparse_map)) {
            agx_bo_unreference(&dev->dev, buffer->sparse_map);
            buffer->sparse_map = NULL;
            hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
            agx_va_free(&dev->dev, buffer->va, true);
            buffer->va = NULL;
            vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
            return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                             "Sparse buffer residency map allocation failed");
         }

         buffer->sparse_map_entries = entries;
         struct hk_sparse_buffer_map_header *map =
            agx_bo_map(buffer->sparse_map);
         memset(map, 0, map_size_B);
         map->resource_base = buffer->va->addr;
         map->page_count = entries;

         agx_bo_assert_range(buffer->sparse_map, 0, map_size_B);

         if (desc_capture_replay) {
            if (sparse_map_fixed_addr % AIL_PAGESIZE != 0) {
               agx_bo_unreference(&dev->dev, buffer->sparse_map);
               buffer->sparse_map = NULL;
               hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
               agx_va_free(&dev->dev, buffer->va, true);
               buffer->va = NULL;
               vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
               return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                                "Misaligned sparse map capture/replay address %#llx",
                                (unsigned long long)sparse_map_fixed_addr);
            }

            enum agx_va_flags map_flags =
               sparse_map_fixed_addr ? AGX_VA_FIXED : 0;
            if (dev->dev.replay_heap_initialized)
               map_flags |= AGX_VA_CAPTURE_REPLAY;

            /* Reserve the deterministic map_size_B span rather than
             * buffer->sparse_map->size so BO-cache over-allocation cannot
             * change the VA footprint between capture and replay. */
            buffer->sparse_map_va = agx_va_alloc(
               &dev->dev, map_size_B, AIL_PAGESIZE, map_flags,
               sparse_map_fixed_addr);
            if (!buffer->sparse_map_va) {
               agx_bo_unreference(&dev->dev, buffer->sparse_map);
               buffer->sparse_map = NULL;
               hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
               agx_va_free(&dev->dev, buffer->va, true);
               buffer->va = NULL;
               vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
               if (sparse_map_fixed_addr != 0)
                  return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                                   "Sparse map capture/replay address %#llx is in use",
                                   (unsigned long long)sparse_map_fixed_addr);
               return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                                "Sparse map capture/replay VA allocation failed");
            }

            if (agx_bo_bind(&dev->dev, buffer->sparse_map,
                            buffer->sparse_map_va->addr, map_size_B, 0,
                            DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE)) {
               agx_va_free(&dev->dev, buffer->sparse_map_va, false);
               buffer->sparse_map_va = NULL;
               agx_bo_unreference(&dev->dev, buffer->sparse_map);
               buffer->sparse_map = NULL;
               hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
               agx_va_free(&dev->dev, buffer->va, true);
               buffer->va = NULL;
               vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
               return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                                "Sparse map capture/replay bind failed");
            }
         }
      }
   }

   if (hk_buffer_in_physical_sparse_map(dev, buffer)) {
      VkResult result = hk_update_physical_sparse_map(
         dev, buffer->va->addr, align64(buffer->vk.size, AIL_PAGESIZE), true);
      if (result != VK_SUCCESS) {
         hk_update_physical_sparse_map(dev, buffer->va->addr,
            align64(buffer->vk.size, AIL_PAGESIZE), false);
         if (buffer->sparse_map_va) {
            agx_va_free(&dev->dev, buffer->sparse_map_va, true);
            buffer->sparse_map_va = NULL;
         }
         agx_bo_unreference(&dev->dev, buffer->sparse_map);
         hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);
         agx_va_free(&dev->dev, buffer->va, true);
         vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
         return result;
      }
   }

   /* Bound memory is reached through this buffer's own address as well as
    * the memory's, or other resources' (hk_cdm_alias_scope()). It counts
    * before memory can be bound to it. */
   if (buffer->va)
      hk_device_va_alias_get(dev, (buffer->vk.create_flags &
                                   VK_BUFFER_CREATE_SPARSE_BINDING_BIT)
                                     ? HK_ALIAS_CAUSE_SPARSE
                                     : HK_ALIAS_CAUSE_CAPTURE_REPLAY);

   if (hk_buffer_in_sparse_list(buffer)) {
      simple_mtx_lock(&dev->sparse_buffers.lock);
      list_addtail(&buffer->sparse_link, &dev->sparse_buffers.list);
      p_atomic_inc(&dev->sparse_buffers.count);
      simple_mtx_unlock(&dev->sparse_buffers.lock);
   }

   *pBuffer = hk_buffer_to_handle(buffer);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyBuffer(VkDevice device, VkBuffer _buffer,
                 const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   if (!buffer)
      return;

   hk_device_quiesce_lost_submissions(dev);

   if (hk_buffer_in_sparse_list(buffer)) {
      simple_mtx_lock(&dev->sparse_buffers.lock);
      list_del(&buffer->sparse_link);
      p_atomic_dec(&dev->sparse_buffers.count);
      simple_mtx_unlock(&dev->sparse_buffers.lock);
   }

   if (hk_buffer_in_physical_sparse_map(dev, buffer))
      hk_update_physical_sparse_map(dev, buffer->va->addr,
         align64(buffer->vk.size, AIL_PAGESIZE), false);

   if (buffer->va) {
      agx_va_free(&dev->dev, buffer->va, true);
      hk_device_va_alias_put(dev);
   }
   hk_release_sparse_scratch(dev, buffer->sparse_scratch_B);

   if (buffer->sparse_map_va) {
      agx_va_free(&dev->dev, buffer->sparse_map_va, true);
      buffer->sparse_map_va = NULL;
   }
   agx_bo_unreference(&dev->dev, buffer->sparse_map);

   vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceBufferMemoryRequirements(
   VkDevice device, const VkDeviceBufferMemoryRequirements *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   VkBufferUsageFlags2KHR usage = pInfo->pCreateInfo->usage;
   const VkBufferUsageFlags2CreateInfoKHR *usage2 = vk_find_struct_const(
      pInfo->pCreateInfo->pNext, BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR);
   if (usage2)
      usage = usage2->usage;

   const uint32_t alignment =
      hk_get_buffer_alignment(pdev, usage, pInfo->pCreateInfo->flags);

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = align64(pInfo->pCreateInfo->size, alignment),
      .alignment = alignment,
      .memoryTypeBits = BITFIELD_MASK(pdev->mem_type_count),
   };

   vk_foreach_struct(sType, ext, pMemoryRequirements->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = ext;
         dedicated->prefersDedicatedAllocation = false;
         dedicated->requiresDedicatedAllocation = false;
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceExternalBufferProperties(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
   VkExternalBufferProperties *pExternalBufferProperties)
{
   /* The Vulkan 1.3.256 spec says:
    *
    *    VUID-VkPhysicalDeviceExternalBufferInfo-handleType-parameter
    *
    *    "handleType must be a valid VkExternalMemoryHandleTypeFlagBits value"
    *
    * This differs from VkPhysicalDeviceExternalImageFormatInfo, which
    * surprisingly permits handleType == 0.
    */
   assert(pExternalBufferInfo->handleType != 0);

   /* All of the current flags are for sparse which we don't support yet.
    * Even when we do support it, doing sparse on external memory sounds
    * sketchy.  Also, just disallowing flags is the safe option.
    */
   if (pExternalBufferInfo->flags)
      goto unsupported;

   switch (pExternalBufferInfo->handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      pExternalBufferProperties->externalMemoryProperties =
         hk_dma_buf_mem_props;
      return;
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT: {
      VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
      if (!pdev->vk.supported_extensions.EXT_external_memory_host)
         goto unsupported;
      pExternalBufferProperties->externalMemoryProperties =
         hk_host_ptr_mem_props;
      return;
   }
   default:
      goto unsupported;
   }

unsupported:
   /* From the Vulkan 1.3.256 spec:
    *
    *    compatibleHandleTypes must include at least handleType.
    */
   pExternalBufferProperties->externalMemoryProperties =
      (VkExternalMemoryProperties){
         .compatibleHandleTypes = pExternalBufferInfo->handleType,
      };
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_BindBufferMemory2(VkDevice device, uint32_t bindInfoCount,
                     const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VK_FROM_HANDLE(hk_device_memory, mem, pBindInfos[i].memory);
      VK_FROM_HANDLE(hk_buffer, buffer, pBindInfos[i].buffer);

      if (mem && !p_atomic_read(&mem->has_buffer_binding)) {
         VK_FROM_HANDLE(hk_device, dev, device);
         p_atomic_set(&mem->has_buffer_binding, true);
         __atomic_thread_fence(__ATOMIC_SEQ_CST);
         if (p_atomic_read(&mem->has_image_binding))
            p_atomic_inc(&dev->buffer_alias_mem_gen);
      }

      if (buffer->va) {
         VK_FROM_HANDLE(hk_device, dev, device);
         /* The VA allocation includes an overread guard, not backed by this
          * buffer's memory. Mapping it too can exceed the BO when a captured
          * BDA buffer occupies the last suballocation (Firefox/Zink). */
         size_t size = align64(buffer->vk.size, 16384);

         /* Lower mapping: read-write */
         int ret = agx_bo_bind(&dev->dev, mem->bo, buffer->vk.device_address,
                               size, pBindInfos[i].memoryOffset,
                               DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
         if (ret)
            return VK_ERROR_UNKNOWN;

         /* Upper mapping: read-only */
         ret =
            agx_bo_bind(&dev->dev, mem->bo,
                        buffer->vk.device_address + dev->dev.sparse_ro_offset,
                        size, pBindInfos[i].memoryOffset, DRM_ASAHI_BIND_READ);
         if (ret)
            return VK_ERROR_UNKNOWN;
      } else {
         assert(buffer->vk.device_address == 0);
         buffer->vk.device_address =
            hk_device_memory_addr(mem) + pBindInfos[i].memoryOffset;
         if (!mem->replay_va &&
             !(mem->bo->flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE | AGX_BO_HOST_PTR))) {
            buffer->storage_backing_handle = mem->bo->handle;
            buffer->storage_backing_offset = pBindInfos[i].memoryOffset;
         }
      }

      VK_FROM_HANDLE(hk_device, dev, device);
      const VkBindMemoryStatusKHR *status =
         vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (dev->g17p_native_as &&
          (buffer->vk.usage & VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) &&
          !hk_gen1_publish_buffer_as_addresses(dev, &buffer->vk)) {
         /* No address space below an acceleration structure in this buffer
          * for its traversal base. */
         const VkResult result = vk_errorf(
            dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
            "no traversal base for an acceleration structure in this buffer");
         if (status != NULL && status->pResult != NULL)
            *status->pResult = result;
         return result;
      }
      if (dev->m5_native_as)
         hk_m5_as_buffer_bound(dev, &buffer->vk);

      if (status != NULL && status->pResult != NULL)
         *status->pResult = VK_SUCCESS;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetBufferOpaqueCaptureDescriptorDataEXT(
   VkDevice device, const VkBufferCaptureDescriptorDataInfoEXT *pInfo,
   void *pData)
{
   VK_FROM_HANDLE(hk_buffer, buffer, pInfo->buffer);

   /* A buffer bound to memory takes the memory's (replayed) address; only a
    * VA of its own (and, for sparse-resident buffers, its residency map's VA)
    * needs replaying. */
   const struct hk_buffer_capture capture = {
      .va = buffer->va ? buffer->va->addr : 0,
      .sparse_map_va =
         buffer->sparse_map_va ? buffer->sparse_map_va->addr : 0,
   };
   memcpy(pData, &capture, sizeof(capture));
   return VK_SUCCESS;
}

VKAPI_ATTR uint64_t VKAPI_CALL
hk_GetBufferOpaqueCaptureAddress(UNUSED VkDevice device,
                                 const VkBufferDeviceAddressInfo *pInfo)
{
   VK_FROM_HANDLE(hk_buffer, buffer, pInfo->buffer);

   return hk_buffer_address_rw(buffer, 0);
}

uint64_t
hk_buffer_address_sparse_ro(const struct hk_buffer *buffer, uint64_t offset)
{
   struct hk_device *dev = (struct hk_device *)buffer->vk.base.device;

   /* Sparse resident buffers read through the read-only shadow mapping so
    * non-resident pages return zeroes. */
   assert(buffer->va != NULL);
   return agx_rw_addr_to_ro(&dev->dev, vk_buffer_address(&buffer->vk, offset));
}
