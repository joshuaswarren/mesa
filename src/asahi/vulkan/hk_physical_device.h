/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_device.h"
#include <float.h>
#include <sys/types.h>
#include "hk_private.h"
#include "util/os_misc.h"
#include "vk_physical_device.h"
#include "vk_sync.h"
#include "wsi_common.h"

/*
 * User-visible driver identity, reported in VkPhysicalDeviceDriverProperties
 * by every physical device this driver opens.  It does not depend on the chip,
 * so a newly enabled GPU reports it with no further change.  The driverID is
 * VK_DRIVER_ID_AURORA_BOREALIS, registered by Khronos for Aurora Silicon.
 */
#define HK_DRIVER_NAME "Borealis"
#define HK_DRIVER_INFO_SUFFIX ", Borealis, derived from Honeykrisp"

struct hk_instance;
struct hk_physical_device;

struct hk_queue_family {
   VkQueueFlags queue_flags;
   uint32_t queue_count;
};

/* VK_EXT_memory_budget: "A heap's usage is an estimate of how much memory
 * the process is currently using in that heap." Every VkInstance enumerates
 * its own hk_physical_device, so the counters live in one record per render
 * node, shared by every physical device of that node in the process.
 */
struct hk_heap_usage {
   struct hk_heap_usage *next;
   dev_t render_dev;
   unsigned refcount;
   uint64_t used[3];
   p_atomic_uint64_t cached[3];
};

struct hk_memory_heap {
   uint64_t size;
   /* Point into the physical device's hk_heap_usage. */
   uint64_t *used;
   p_atomic_uint64_t *cached;
   VkMemoryHeapFlags flags;
};

struct hk_physical_device {
   struct vk_physical_device vk;
   dev_t render_dev;
   int master_fd;

   /* Only used for VK_EXT_memory_budget */
   struct agx_device dev;

   struct wsi_device wsi_device;

   uint8_t device_uuid[VK_UUID_SIZE];

   // TODO: add mapable VRAM heap if possible
   struct hk_heap_usage *heap_usage;
   struct hk_memory_heap mem_heaps[3];
   VkMemoryType mem_types[3];
   uint8_t mem_heap_count;
   uint8_t mem_type_count;
   uint64_t sysmem;
   float sysmem_heap_percent;

   struct hk_queue_family queue_families[3];
   uint8_t queue_family_count;

   /* Kernel queue limits with DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES, zero
    * when the kernel does not report them. */
   struct drm_asahi_queue_limits queue_limits;

   struct vk_sync_type syncobj_sync_type;
   /* syncobj_sync_type's original operations, see hk_gpu_waits.h. */
   struct vk_sync_type syncobj_base_type;
   const struct vk_sync_type *sync_types[2];

   simple_mtx_t debug_compile_lock;

   /* Whether any [SU]SCALED format has sampled-image (or input-attachment)
    * features. Only then can a valid application descriptor carry a nonzero
    * hk_sampled_image_descriptor_sw::scaled_conversion kind. */
   bool scaled_sampling;
   bool scaled_storage;

   /* Set once a device of this GPU creates a sampler whose border colour G17P
    * shaders must emulate (hk_CreateSampler). Shaders compiled before then
    * leave the emulation out and are replaced by emulating twins when bound
    * (hk_border_twin); the shader cache key carries it. Per physical device
    * because the cache key is computed from it (hk_hash_graphics_state). */
   uint32_t border_emulation;

   /* Set once a device of this GPU creates an image view with a non-zero
    * relative minLod (hk_image_view_init). Shaders compiled before then leave
    * image-view minLod emulation out and are replaced by emulating twins when
    * bound (hk_min_lod_twin); the shader cache key carries it. */
   uint32_t min_lod_emulation;
};

VK_DEFINE_HANDLE_CASTS(hk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

static inline struct hk_instance *
hk_physical_device_instance(struct hk_physical_device *pdev)
{
   return (struct hk_instance *)pdev->vk.instance;
}

VkResult hk_create_drm_physical_device(struct vk_instance *vk_instance,
                                       struct _drmDevice *drm_device,
                                       struct vk_physical_device **pdev_out);

void hk_physical_device_destroy(struct vk_physical_device *vk_device);

uint64_t hk_codegen_env_hash(void);

/* optimalTilingLayoutUUID: the device's generation contract only. */
void hk_layout_identity(const struct agx_device *dev,
                        uint8_t uuid[VK_UUID_SIZE]);

/* pipelineCacheUUID (and shaderBinaryUUID) and the disk cache's driver flags:
 * the build, the generation contract, the compiler flags, the codegen
 * environment and the kernel's feature bits. */
void hk_cache_identity(const uint8_t build_sha[32],
                       const struct agx_device *dev, uint64_t compiler_flags,
                       uint64_t env_hash, uint8_t uuid[VK_UUID_SIZE],
                       uint64_t *disk_cache_id);

void hk_border_emulation_recall(struct hk_physical_device *pdev);
void hk_border_emulation_remember(struct hk_physical_device *pdev);
void hk_min_lod_emulation_recall(struct hk_physical_device *pdev);
void hk_min_lod_emulation_remember(struct hk_physical_device *pdev);

uint64_t hk_get_sysmem_heap_size(struct hk_physical_device *pdev);

/* Size Borealis's UMA system-memory heap from the kernel's exported memacct
 * admission capacities (drm_asahi_params_global::memory_limit and
 * client_memory_limit, where 0 means no driver limit when has_memory_limits is
 * true) so VkMemoryHeap::size matches what GEM_CREATE will admit:
 *
 * - When has_memory_limits is true, kernel_limit_B is the smaller non-zero
 *   limit of client_memory_limit_B and memory_limit_B, or 0 if both driver
 *   limits are disabled.
 * - An explicit driconf heap_memory_percent (> 0) scales total_ram_B up to
 *   total_aligned_B, clamped to kernel_limit_B when non-zero so the advertised
 *   heap never exceeds kernel admission (and able to raise the heap above 50%
 *   when driver limits are disabled).
 * - With OS_GPU_HEAP_SIZE_HEURISTIC (0.0f), use kernel_limit_B when non-zero,
 *   or cap at 50% of total_ram_B when kernel_limit_B is 0 (both driver limits
 *   disabled, legacy 600-byte kernels, and virtio) to avoid swapping.
 */
static inline uint64_t
hk_resolve_uma_sysmem_heap(uint64_t total_ram_B, float driconf_pct,
                           bool has_memory_limits, uint64_t memory_limit_B,
                           uint64_t client_memory_limit_B, float *percent_out)
{
   if (total_ram_B == 0)
      return 0;

   uint64_t kernel_limit_B = 0;
   if (has_memory_limits) {
      if (client_memory_limit_B > 0 && memory_limit_B > 0)
         kernel_limit_B = MIN2(client_memory_limit_B, memory_limit_B);
      else
         kernel_limit_B = client_memory_limit_B ? client_memory_limit_B
                                                : memory_limit_B;
   }

   const uint64_t total_aligned_B = ROUND_DOWN_TO(total_ram_B, 1ull << 20);
   uint64_t heap_B;

   if (driconf_pct > OS_GPU_HEAP_SIZE_HEURISTIC) {
      double scaled = (double)total_ram_B * (double)driconf_pct;
      uint64_t requested_B = scaled >= (double)total_aligned_B
                                ? total_aligned_B
                                : ROUND_DOWN_TO((uint64_t)scaled, 1ull << 20);
      heap_B = MIN2(requested_B, total_aligned_B);
      if (kernel_limit_B > 0)
         heap_B = MIN2(heap_B, ROUND_DOWN_TO(kernel_limit_B, 1ull << 20));
   } else if (kernel_limit_B > 0) {
      heap_B = ROUND_DOWN_TO(MIN2(kernel_limit_B, total_ram_B), 1ull << 20);
   } else {
      heap_B = ROUND_DOWN_TO(total_ram_B / 2, 1ull << 20);
   }

   if (percent_out) {
      float pct = (float)((double)heap_B / (double)total_ram_B);
      pct = CLAMP(pct, 0.0f, 1.0f);
      if (heap_B > 0 && pct == 0.0f)
         pct = FLT_MIN;
      *percent_out = pct;
   }

   return heap_B;
}

#if defined(VK_USE_PLATFORM_WAYLAND_KHR) ||                                    \
   defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR) ||    \
   defined(VK_USE_PLATFORM_DISPLAY_KHR)
#define HK_USE_WSI_PLATFORM
#endif
