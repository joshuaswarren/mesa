/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>

#include "vk_log.h"
#include "util/u_debug.h"
#include "asahi/lib/agx_chip.h"
#include "asahi/lib/agx_device.h"
#include "vk_physical_device_features.h"
#include "vk_util.h"

#define HK_MAX_SETS                      16
#define HK_MAX_PUSH_SIZE                 256
#define HK_MAX_DYNAMIC_BUFFERS           64
#define HK_MAX_RTS                       8
#define HK_MIN_SSBO_ALIGNMENT            4
#define HK_MIN_TEXEL_BUFFER_ALIGNMENT    16
#define HK_MIN_UBO_ALIGNMENT             64
#define HK_MAX_VIEWPORTS                 16
/* The largest per-descriptor stride any binding can take. G17P storage images
 * occupy 80 bytes and round to 96 so every array element's PBE half remains
 * 32-byte aligned. Mutable descriptors must be large enough for that member.
 */
#define HK_MAX_DESCRIPTOR_SIZE           96
#define HK_MAX_PUSH_DESCRIPTORS          32
#define HK_MAX_DESCRIPTOR_BUFFERS        32
#define HK_MAX_DESCRIPTOR_SET_SIZE       (1u << 30)
#define HK_MAX_INLINE_UNIFORM_BLOCK_SIZE (1u << 16)
#define HK_MAX_DESCRIPTORS               (1 << 20)
#define HK_MAX_PLANES                    3
/* The largest push descriptor set. A push layout has at most
 * HK_MAX_PUSH_DESCRIPTORS elements of at most HK_MAX_DESCRIPTOR_SIZE bytes,
 * except that an element of a binding whose immutable samplers convert a
 * multi-planar format takes that per plane (hk_CreateDescriptorSetLayout()
 * multiplies the stride by the plane count), and each binding starts after
 * less than one more of alignment padding. So every push layout within the
 * limits fits.
 */
#define HK_PUSH_DESCRIPTOR_SET_SIZE                                            \
   (HK_MAX_PUSH_DESCRIPTORS * (HK_MAX_PLANES + 1) * HK_MAX_DESCRIPTOR_SIZE)
/* Storage buffer descriptors hold the exact byte range and every bounds check
 * (explicit I/O lowering, check_in_bounds) tests offset + size - 1 < range,
 * so ranges are not rounded up for robustBufferAccess2.
 */
#define HK_SSBO_BOUNDS_CHECK_ALIGNMENT 1
#define HK_MAX_MULTIVIEW_VIEW_COUNT    32

#define HK_SPARSE_ADDR_SPACE_SIZE (1ull << 39)
/* M5 fills unbound sparse VA with scratch through the hypervisor's SPTM
 * provider, which spends one page-table frame of its own heap per 32 MiB of
 * filled VA and never returns it to the heap. Bound what one device fills.
 * Interim capability limit: concurrent 4 x 64 GiB fills still reset the
 * machine (at >= 7930 live table frames) while x1n1's real headroom is
 * unknown; 16 GiB keeps one device near 1024 frames under the kernel's
 * 2048-frame budget.
 */
#define HK_M5_SPARSE_ADDR_SPACE_SIZE (1ull << 34)
#define HK_MAX_BUFFER_SIZE        (1ull << 37)
#define HK_MAX_SHARED_SIZE        (32 * 1024)

/* Device slabs of command memory, one kind per BO placement. */
enum hk_cmd_slab_class {
   HK_CMD_SLAB_VDM,
   HK_CMD_SLAB_MAIN,
   HK_CMD_SLAB_USC,
   HK_CMD_SLAB_IOTO,
   /* Unmapped, non-writeback slabs for GPU-written indirect command streams,
    * Ioto vertex programs, and draw-ID tables. Keeping GPU-written spans out of
    * CPU-cached slabs prevents shared-cacheline writebacks from clobbering
    * pre-graphics compute output. */
   HK_CMD_SLAB_VDM_GPU,
   HK_CMD_SLAB_IOTO_GPU,
   HK_CMD_SLAB_DATA_GPU,
   HK_CMD_SLAB_COUNT,
};

struct hk_addr_range {
   uint64_t addr;
   uint64_t range;
};

#define hk_cmd_buffer_device(cmd) ((struct hk_device *)(cmd)->vk.base.device)

#define perf_debug_dev(dev, fmt, ...)                                          \
   do {                                                                        \
      if ((dev)->debug & AGX_DBG_PERF)                                         \
         mesa_log(MESA_LOG_WARN, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__);        \
   } while (0)

#define perf_debug(cmd, fmt, ...)                                              \
   do {                                                                        \
      if (hk_cmd_buffer_device(cmd)->dev.debug & AGX_DBG_PERF)                 \
         mesa_log(MESA_LOG_WARN, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__);        \
   } while (0)

/* Native ray queries submit the extended compute command (prior fields and
 * intersection state after the legacy payload). A kernel that does not
 * advertise both features parses those bytes differently and rejects the
 * submission, so ray tracing stays off there instead of failing at submit. */
#define HK_G17P_RT_KERNEL_FEATURES                                             \
   (DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS | DRM_ASAHI_FEATURE_RENDER_RAY_QUERY)

static inline bool
hk_g17p_rt_enabled(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P &&
          (dev->params.features & HK_G17P_RT_KERNEL_FEATURES) ==
             HK_G17P_RT_KERNEL_FEATURES;
}

/* Fallback for kernels that advertise the bounded AS execution allowance and
 * exact-prior submission contract without RENDER_RAY_QUERY: ordinary compute
 * AS construction, copies and refits can still use the GEN1 node layout
 * without creating a traversal context. Decide from the architecture and
 * kernel capability bits rather than chip IDs. Do not infer traversal support
 * from this predicate.
 */
static inline bool
hk_g16s_as_enabled(const struct agx_device *dev)
{
   const uint64_t required = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                             DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD;
   return dev->chip == AGX_CHIP_G17P &&
          (dev->params.features & required) == required;
}

static inline bool
hk_gen1_as_enabled(const struct agx_device *dev)
{
   return hk_g17p_rt_enabled(dev) || hk_g16s_as_enabled(dev);
}

/* Whether a logical device enables any ray-tracing feature. Only such a
 * device sets up the native G17P ray-tracing state: the ray query program
 * image and its fixed code binds, the acceleration-structure tables, and the
 * render SCS reservation in the kernel context of each of its queues
 * (hk_device_init_accel_struct_state()). Without one of these features no
 * acceleration structure, ray query or ray-tracing pipeline can exist on the
 * device. Every feature here is advertised only with the ray-tracing
 * extensions. G16S also admits an AS-only device without traversal state.
 *
 * Shaders compiled for the two kinds of device differ, so this is part of
 * every shader cache key (HK_FEAT_RAY_TRACING). */
static inline bool
hk_rt_features_enabled(const struct vk_features *f)
{
   return f->accelerationStructure || f->accelerationStructureCaptureReplay ||
          f->accelerationStructureIndirectBuild ||
          f->accelerationStructureHostCommands ||
          f->descriptorBindingAccelerationStructureUpdateAfterBind ||
          f->rayQuery || f->rayTraversalPrimitiveCulling ||
          f->rayTracingPipeline ||
          f->rayTracingPipelineShaderGroupHandleCaptureReplay ||
          f->rayTracingPipelineShaderGroupHandleCaptureReplayMixed ||
          f->rayTracingPipelineTraceRaysIndirect ||
          f->pipelineLibraryGroupHandles || f->rayTracingMaintenance1 ||
          f->rayTracingPipelineTraceRaysIndirect2 ||
          f->rayTracingPositionFetch || f->rayTracingInvocationReorder;
}
