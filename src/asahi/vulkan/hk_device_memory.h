/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "asahi/lib/agx_bo.h"
#include "vk_device_memory.h"

#include "util/list.h"

struct hk_device;
struct hk_image_plane;

struct hk_device_memory {
   struct vk_device_memory vk;

   struct agx_bo *bo;

   /* Opaque application-visible token used for BDA capture/replay. This is
    * deliberately separate from the BO's current mapping: replay buffers own
    * the fixed VA that determines their device address. */
   uint64_t opaque_capture_address;

   /* VK_EXT_descriptor_buffer capture/replay (hk_AllocateMemory()): a second
    * mapping of the BO at an address that replays, which buffers and images
    * bound to this memory take as its address. NULL otherwise. */
   struct agx_va *replay_va;

   void *map;

   /* map is this object's own mapping of map_size bytes rather than the
    * BO's shared one: the BO may be aliased by other memory objects. */
   size_t map_size;
   bool map_owned;

   /* The mapping host image copies go through (hk_device_memory_copy_map()),
    * NULL until the first copy. It covers the whole BO and lives until
    * vkFreeMemory. */
   void *copy_map;

   /* Residency preference the kernel accepted (memory priority 1.0). */
   bool keep_resident;

   /* An untracked import the GPU may write that another import can map at a
    * second address too (fd/dma-buf, or a writable host import under the
    * static M5 rule): counted in hk_device::va_alias_resources while it lives. */
   bool alias_import;

   /* A host import tracked in hk_device::host_alias.imports; counts in
    * hk_device::va_alias_resources (and hk_device::alias_live on M5 with
    * host_alias.dynamic) only while it may alias. */
   bool host_alias_tracked;
   /* An fd or dma-buf import, counted in hk_device::alias_live for its
    * lifetime (hk_device_alias_get()). */
   bool alias_fd;

   /* Atomic flags tracking whether an image or buffer has ever been bound to
    * this memory object. Used by exact-alias dependency tracking so optimal
    * placed images in image-only heaps do not force HK_SYNC_GFX_MEM. */
   bool has_image_binding;
   bool has_buffer_binding;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

/* The device address of the start of the memory object */
static inline uint64_t
hk_device_memory_addr(const struct hk_device_memory *mem)
{
   return mem->replay_va ? mem->replay_va->addr : mem->bo->va->addr;
}

void *hk_device_memory_copy_map(struct hk_device_memory *mem);

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;
extern const VkExternalMemoryProperties hk_host_ptr_mem_props;
