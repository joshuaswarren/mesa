/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include "asahi_g15s_stream_gate.h"
#include "asahi_m5_submit_gate.h"
#include "asahi_g17p_stream_validator.h"
#include "util/macros.h"
#include "util/simple_mtx.h"
#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"

/* Bind/unbind mutate the shared mapping array and own its BO references.
 * Submission validation retains pointers into that array while walking it. */
static simple_mtx_t shim_mapping_lock = SIMPLE_MTX_INITIALIZER;

/*
 * T8103 / M1, the AGX2 identity Honeykrisp is already conformant on.  This is
 * the control arm for G17P work: run the same client twice, changing only
 * ASAHI_SHIM_CHIP, and any difference is attributable to the AGX3 paths.
 *
 * Only the architectural identity differs from params_g17p below - generation,
 * variant, chip id, ISA, HAL, core topology, soft faults and the 39-bit user
 * VA range.  Everything else is held equal deliberately so the diff isolates
 * the generation.  A stub here (the previous zero num_clusters_total) makes
 * agx_chip_from_params return AGX_CHIP_UNKNOWN and the device fails to open,
 * which is why the fields are spelled out.
 */
static const struct drm_asahi_params_global params_g13g = {
   .features = 0,
   .gpu_generation = 13,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .chip_id = 0x8103,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 8,
   .core_masks = {0xff},
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .max_frequency_khz = 1278000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 2,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

/* Compile-only M1 Pro, M2 and M4 Pro profiles. Timing and submission
 * ceilings deliberately use the existing fixtures, not hardware measurements.
 * The product identity selects the real driver/compiler contract; T6040's
 * two ten-core clusters exercise its larger helper tables (Mesa PR #55).
 * No RT capabilities are invented for the M4 firmware runtime.
 */
static const struct drm_asahi_params_global params_g13x = {
   .features = 0,
   .gpu_generation = 13,
   .gpu_variant = 'S',
   .gpu_revision = 0,
   .chip_id = 0x6000,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 8,
   .core_masks = {0xff, 0xff},
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .max_frequency_khz = 1278000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 2,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

static const struct drm_asahi_params_global params_g14g = {
   .features = 0,
   .gpu_generation = 14,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .chip_id = 0x8112,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .max_frequency_khz = 1278000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 2,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

static const struct drm_asahi_params_global params_g16s = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS,
   .gpu_generation = 16,
   .gpu_variant = 'S',
   .gpu_revision = 0x11,
   .chip_id = 0x6040,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff, 0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/* An explicit capability-bearing fixture tests RT exposure and eight queues.
 * The plain g16s fixture remains a negative control for the kernel RT gate.
 * Neither fixture is a claim that those commands executed on hardware. */
static const struct drm_asahi_params_global params_g16s_rt = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
               DRM_ASAHI_FEATURE_RENDER_RAY_QUERY |
               DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES,
   .gpu_generation = 16,
   .gpu_variant = 'S',
   .gpu_revision = 0x11,
   .chip_id = 0x6040,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff, 0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/* The AS-only contract deliberately lacks the traversal capability. */
static const struct drm_asahi_params_global params_g16s_as = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
               DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD |
               DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES,
   .gpu_generation = 16,
   .gpu_variant = 'S',
   .gpu_revision = 0x11,
   .chip_id = 0x6040,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff, 0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/*
 * T8140 / A18 Pro, exactly as our G17P kernel adapter reports it
 * (drivers/gpu/drm/asahi/g17_drm.rs:24-27,118-140 and file.rs:295-324): a
 * 42-bit user root two pages below the top, one die, one cluster, USC
 * generation 3, HAL 200, and soft faults on because fault_control defaults to
 * 0xb.  Select with ASAHI_SHIM_CHIP=g17p.  This exists so the Honeykrisp G17P
 * shader and command-stream paths can be exercised without hardware.
 * ASAHI_SHIM_RENDER_RAY_QUERY=1 adds the render ray query feature of kernels
 * that admit them, which G17P ray tracing pipelines need (shim_features()).
 */
static const struct drm_asahi_params_global params_g17p = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS |
               DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS,
   .gpu_generation = 17,
   .gpu_variant = 'P',
   .gpu_revision = 0,
   .chip_id = 0x8140,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 5,
   .core_masks = {0x1f},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   /* file.rs:60 MAX_COMMANDS_PER_SUBMISSION. With 1 the driver would split
    * every command into its own ioctl and the per-command barrier fields
    * would all collapse to 0, hiding what the driver actually requests. */
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

#define SHIM_T6041_AURORA_PARAMS(variant)                                     \
   {                                                                          \
      .features = DRM_ASAHI_FEATURE_VM_STATUS |                               \
                  DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |                      \
                  DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |                 \
                  DRM_ASAHI_FEATURE_VM_STATUS_MIRROR |                        \
                  DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |                    \
                  DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS |                       \
                  DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD |                        \
                  DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS,                       \
      .gpu_generation = 16, .gpu_variant = (variant), .gpu_revision = 0x11,   \
      .chip_id = 0x6041, .num_dies = 1, .num_clusters_total = 4,              \
      .num_cores_per_cluster = 10,                                            \
      .core_masks = {0x3ff, 0x3ff, 0x3ff, 0x3ff},                             \
      .vm_start = 0x4000, .vm_end = (1ull << 42) - 0x8000ull,                 \
      .vm_kernel_min_size = 0x20000000,                                       \
      .max_commands_per_submission = 64, .max_attachments = 16,               \
      .max_frequency_khz = 1398000,                                           \
      .command_timestamp_frequency_hz = 1000000000, .usc_generation = 3,      \
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,                            \
   }

static const struct drm_asahi_params_global params_t6041 =
   SHIM_T6041_AURORA_PARAMS('C');

static const struct drm_asahi_queue_limits queue_limits_g17 = {
   .max_queues = 42,
   .max_in_flight_per_queue = 1,
};

static const struct drm_asahi_params_global params_g16g = {
   .features = DRM_ASAHI_FEATURE_VM_STATUS |
               DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
               DRM_ASAHI_FEATURE_VM_STATUS_MIRROR |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS,
   .gpu_generation = 16,
   .gpu_variant = 'G',
   .gpu_revision = 0x10,
   .chip_id = 0x8132,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1470000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/*
 * T6030 / M3 Pro, as the M3 kernel (drivers/gpu/drm/apple/gpu-g15-backend.c
 * g15_params) reports it: AGX3 shaders on the legacy firmware interface, one
 * die, two clusters of ten cores, VM_STATUS | INDEPENDENT_QUEUES |
 * PER_COMMAND_INPUT_SYNCS | VM_STATUS_MIRROR | EXACT_PRIOR_BARRIERS. Select
 * with ASAHI_SHIM_CHIP=g15s. The submission then runs through the kernel's
 * packet rules and stream_gate, mirrored in asahi_g15s_stream_gate.c, because
 * a refused record on hardware is at best an EINVAL and a mis-encoded one a
 * fail-stop GPU fault.
 */
static const struct drm_asahi_params_global params_g15s = {
   .features = DRM_ASAHI_FEATURE_VM_STATUS |
               DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |
               DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
               DRM_ASAHI_FEATURE_VM_STATUS_MIRROR |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS,
   .gpu_generation = 15,
   .gpu_variant = 'S',
   .gpu_revision = 0x11,
   .chip_id = 0x6030,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 10,
   .core_masks = {0x1ff, 0x1ff},
   .vm_start = 0x4000,
   .vm_end = 0x2ffffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1398000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

/*
 * T6031 / M3 Max: the same G15 generation as T6030 with twice the clusters.
 * One die, four clusters of ten cores (the 40-core part, every core present),
 * revision B2 (0x12), AGX3 shaders on the legacy firmware interface, and the
 * top GPU clock of the J516C, 1380 MHz. Only the native ABI contract bits are
 * reported by default, so
 * the driver takes its baseline submission paths; the ASAHI_SHIM_* feature
 * switches add bits on top. Submissions go through the same packet rules and
 * stream gate as G15S, and the same per-VM kernel range is refused for binds.
 * Select with ASAHI_SHIM_CHIP=g15c.
 */
static const struct drm_asahi_params_global params_g15c = {
   .features = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS,
   .gpu_generation = 15,
   .gpu_variant = 'C',
   .gpu_revision = 0x12,
   .chip_id = 0x6031,
   .num_dies = 1,
   .num_clusters_total = 4,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff, 0x3ff, 0x3ff, 0x3ff},
   .vm_start = 0x4000,
   .vm_end = 0x2ffffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1380000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

/* Synthetic G15G contract for host-only device/allocator checks. The feature
 * baseline matches Aurora m3_drm.rs; this is not a hardware execution model. */
static const struct drm_asahi_params_global params_g15g = {
   .features = DRM_ASAHI_FEATURE_VM_STATUS |
               DRM_ASAHI_FEATURE_SCHEDULED_QUEUES,
   .gpu_generation = 15,
   .gpu_variant = 'G',
   .gpu_revision = 0x20,
   .chip_id = 0x8122,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = 0x2ffffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 618000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

/* QUEUE_LIMITS (group 2) of the M3 kernel. */
static const struct drm_asahi_queue_limits queue_limits_g15s = {
   .max_queues = 58,
   .max_in_flight_per_queue = 2,
};

/* The M3 kernel keeps every per-VM object in [0x7000000000, 0x7100000000)
 * and refuses user binds there. */
#define SHIM_G15S_KERNEL_VA_START UINT64_C(0x7000000000)
#define SHIM_G15S_KERNEL_VA_END   UINT64_C(0x7100000000)
/* Explicit offline M5 Pro identity for command-generation tests. */
static const struct drm_asahi_params_global params_g17s = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS | DRM_ASAHI_FEATURE_RENDER_RESOURCES |
               DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2,
   .gpu_generation = 17, .gpu_variant = 'S', .gpu_revision = 0,
   .chip_id = 0x6050, .num_dies = 1, .num_clusters_total = 2,
   .num_cores_per_cluster = 10, .core_masks = {0x3ff, 0x3ff},
   .vm_start = 0x4000, .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000, .max_commands_per_submission = 64,
   .max_attachments = 16, .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3, .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

/* Explicit offline M5 Ultra identity (ASAHI_SHIM_CHIP=g17d) for
 * command-generation tests: the M5 Pro profile's features with the M5 Ultra's
 * admission contract (agx_params_m5_supported()): T6050, variant D, two dies,
 * eight clusters of ten cores. It exercises what only a two-die, eight-cluster
 * part selects: the scoped compute ordering record and the eight-cluster
 * helper-table offsets. */
static const struct drm_asahi_params_global params_g17d = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS | DRM_ASAHI_FEATURE_RENDER_RESOURCES |
               DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2,
   .gpu_generation = 17, .gpu_variant = 'D', .gpu_revision = 0,
   .chip_id = 0x6050, .num_dies = 2, .num_clusters_total = 8,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff, 0x3ff, 0x3ff, 0x3ff, 0x3ff, 0x3ff, 0x3ff, 0x3ff},
   .vm_start = 0x4000, .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000, .max_commands_per_submission = 64,
   .max_attachments = 16, .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3, .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

/* The M5 Pro as the G17 kernel runtime presents it: GET_PARAMS of the E2
 * kernel on a J716s (features 0x23cfe7c06, two clusters of ten cores): no
 * soft faults, the M5 command set's features (program storage lists, shader
 * write classes, staged renders, whole attachment renders, task/mesh
 * storage, compressed multisample depth and stencil), exact prior and
 * fragment barriers, cumulative signals and few-primitive renders. The VM
 * status (ASAHI_SHIM_VM_STATUS adds it) and independent queue features are
 * left out because this shim does not emulate their ioctls by default.
 * Select with ASAHI_SHIM_CHIP=g17s-kernel; submissions then pass the M5
 * submission gate (asahi_m5_submit_gate.c), which refuses what that kernel
 * refuses. */
static const struct drm_asahi_params_global params_g17s_kernel = {
   .features = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
               DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
               DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS |
               DRM_ASAHI_FEATURE_FEW_PRIMITIVES |
               DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS |
               DRM_ASAHI_FEATURE_RENDER_RESOURCES |
               DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2 |
               DRM_ASAHI_FEATURE_COMPUTE_RESOURCES |
               DRM_ASAHI_FEATURE_SPARSE_TILES |
               DRM_ASAHI_FEATURE_COMPUTE_MATRIX |
               DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES |
               DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES |
               DRM_ASAHI_FEATURE_WHOLE_ATTACHMENT_RENDERS |
               DRM_ASAHI_FEATURE_TASK_MESH_STORAGE |
               DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION |
               DRM_ASAHI_FEATURE_M5_STENCIL_MSAA_COMPRESSION,
   .gpu_generation = 17, .gpu_variant = 'S', .gpu_revision = 17,
   .chip_id = 0x6050, .num_dies = 1, .num_clusters_total = 2,
   .num_cores_per_cluster = 10, .core_masks = {0x3ff, 0x3ff},
   .vm_start = 0x4000, .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000, .max_commands_per_submission = 64,
   .max_attachments = 16, .max_frequency_khz = 1620000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3, .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

/* M5 Pro with the task (object->mesh) mesh rendering buffer bound
 * (m5_mesh_storage_chunks != 0): DRM_ASAHI_FEATURE_TASK_MESH_STORAGE. */
static const struct drm_asahi_params_global params_g17s_task = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS | DRM_ASAHI_FEATURE_RENDER_RESOURCES |
               DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2 |
               DRM_ASAHI_FEATURE_COMPUTE_RESOURCES |
               DRM_ASAHI_FEATURE_TASK_MESH_STORAGE,
   .gpu_generation = 17, .gpu_variant = 'S', .gpu_revision = 0,
   .chip_id = 0x6050, .num_dies = 1, .num_clusters_total = 2,
   .num_cores_per_cluster = 10, .core_masks = {0x3ff, 0x3ff},
   .vm_start = 0x4000, .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000, .max_commands_per_submission = 64,
   .max_attachments = 16, .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3, .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

/* Compiler diagnostics only: J813's observed G17G C0/HAL 300 topology.
 * This profile rejects every submission; it cannot validate GPU execution or
 * produce a conformance result. Keep its explicit name distinct from hardware.
 */
static const struct drm_asahi_params_global params_g17g_compile_only = {
   .gpu_generation = 17,
   .gpu_variant = 'G',
   .gpu_revision = 0x20,
   .chip_id = 0x8142,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

/*
 * T8152 / M6 (J873), selected with ASAHI_SHIM_CHIP=g18g. There is no G18G
 * kernel driver yet, so like the G17G profile above this one rejects every
 * submission: it exists to drive the G18G identity, record packers and
 * compiler gates offline, not to validate execution.
 *
 * Grounded fields: generation 18, variant 'G' and chip 0x8152 (the J873 is
 * Mac18,5 / T8152, its ADT GPU node is gpu,t8152 and its kext
 * com.apple.AGXG18G); USC generation 3 (the applegpu_g18g Mach-O
 * cpusubtype 0x1c3); a 42-bit user VA two pages below the top, as on M5 (the
 * UAT is 43 bits wide with root selection at bit 42, per the SPTM GET_INFO
 * replies). Revision and topology are measured
 * on the J873: the Linux driver's SGX ID probe decodes revision B1 and two
 * clusters of six cores (mask 0xfff), and the native ioreg reports
 * num_cores=12, num_mgpus=2, core_mask_list=(63,63). (The kext's class
 * name, AGX::PI_400::G::A0, ends in A0; the ID register reads B1.)
 *
 * gpu_hal_generation stays DRM_ASAHI_GPU_HAL_300 until the kernel settles on a
 * value: Apple's G18G kext is PI_400, but no Linux G18G driver publishes one
 * and the uAPI defines no DRM_ASAHI_GPU_HAL_400. Mesa selects the G18G record
 * layouts from agx_chip_command_stream_generation, never from this number.
 */
static const struct drm_asahi_params_global params_g18g = {
   .gpu_generation = 18,
   .gpu_variant = 'G',
   .gpu_revision = 0x11,
   .chip_id = 0x8152,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 6,
   .core_masks = {0x3f, 0x3f},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

static bool
g17g_compile_only(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   return chip && !strcmp(chip, "g17g-compile-only");
}

static bool
g18g_profile(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   return chip && !strcmp(chip, "g18g");
}

static const struct drm_asahi_params_global *
shim_params(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");

   if (chip && !strcmp(chip, "g17s-task"))
      return &params_g17s_task;
   if (chip && !strcmp(chip, "g17s-kernel"))
      return &params_g17s_kernel;

   if (chip && !strcmp(chip, "g13x"))
      return &params_g13x;
   if (chip && !strcmp(chip, "g14g"))
      return &params_g14g;
   if (chip && !strcmp(chip, "g16s-as"))
      return &params_g16s_as;
   if (chip && !strcmp(chip, "g16s-rt"))
      return &params_g16s_rt;
   if (chip && !strcmp(chip, "g16s"))
      return &params_g16s;
   if (g17g_compile_only())
      return &params_g17g_compile_only;

   if (g18g_profile())
      return &params_g18g;

   if (chip && !strcmp(chip, "g17p"))
      return &params_g17p;
   if (chip && !strcmp(chip, "g15s"))
      return &params_g15s;
   if (chip && !strcmp(chip, "g15g"))
      return &params_g15g;
   if (chip && !strcmp(chip, "g15c"))
      return &params_g15c;
   if (chip && !strcmp(chip, "g16g"))
      return &params_g16g;
   if (chip && !strcmp(chip, "g17s"))
      return &params_g17s;
   if (chip && !strcmp(chip, "g17d"))
      return &params_g17d;
   if (chip && !strcmp(chip, "t6041"))
      return &params_t6041;

   return &params_g13g;
}

static bool
shim_g15s(void)
{
   return shim_params() == &params_g15s;
}

static bool
shim_g17_runtime(void)
{
   const struct drm_asahi_params_global *params = shim_params();
   return params == &params_g17p || params == &params_t6041;
}

/* Whether the selected identity is driven by the M3 (G15) kernel, whose
 * submission gate and per-VM object range the shim mirrors. */
static bool
shim_g15_kernel(void)
{
   const struct drm_asahi_params_global *params = shim_params();
   return params == &params_g15s || params == &params_g15c;
}

/* The queue's USC window, as QUEUE_CREATE set it. One queue base per run is
 * enough here: Honeykrisp gives every queue of a device the same base. */
static uint64_t shim_usc_exec_base;

/* Identity is unconditional: flags and chip selection never turn a registered
 * shim fd into a physical node, even when fstat/version emulate that node. */
bool asahi_noop_is_registered_fd(int fd)
   __attribute__((visibility("default")));

bool
asahi_noop_is_registered_fd(int fd)
{
   return drm_shim_fd_lookup(fd) != NULL;
}

/* This symbol is available only from the actual loaded noop driver. D3D12
 * has a weak reference so normal clients retain the closed hardware gate. */
bool asahi_noop_d3d12_graphics_qualification_allowed(int fd)
   __attribute__((visibility("default")));

bool
asahi_noop_d3d12_graphics_qualification_allowed(int fd)
{
   const char *enable = getenv("AURORA_D3D12_SHIM_GRAPHICS");
   const char *validate = getenv("ASAHI_SHIM_VALIDATE_G17P_STREAM");
   return drm_shim_fd_lookup(fd) != NULL && shim_params() == &params_g17p &&
          enable && !strcmp(enable, "1") && validate && !strcmp(validate, "1");
}

struct asahi_bo {
   struct shim_bo base;
   uint32_t offset;
   uint32_t gem_flags;
};

static struct asahi_bo *
asahi_bo(struct shim_bo *bo)
{
   return (struct asahi_bo *)bo;
}

struct asahi_device {
   uint64_t next_offset;
};

static struct asahi_device asahi = {
   .next_offset = 0x1000,
};

struct asahi_vm_mapping {
   int fd;
   uint64_t address;
   uint64_t size;
   uint64_t object_offset;
   bool single_page;
   struct shim_bo *bo;
};

static struct {
   struct asahi_vm_mapping *entries;
   size_t count;
   size_t capacity;
} shim_mappings;

/*
 * The kernel's T8140 native render-context view registry, mirrored.
 *
 * drivers/gpu/drm/asahi/mmu.rs keeps a seven-slot registry of the fixed
 * process resources native AGXSecureGart presents twice: once at a canonical
 * high "source" address userspace owns, and once at a low "context" address
 * the driver materializes.  A slot is filled ONLY by a VM_BIND map whose range
 * is a superset of that view's source interval
 * (Vm::track_t8140_native_context_binding), and is cleared by any map or unmap
 * that merely overlaps it (the same function, plus
 * untrack_t8140_native_context_range).
 *
 * Every render submission then walks all seven slots
 * (Uat::install_t8140_job_context_aliases) and refuses the job with ENOENT if
 * one is empty, which is the exact hardware failure this models:
 *
 *   MMU: T8140 job context view 0 source 0x1000000000+0x10000 -> 0x0 is not
 *   bound
 *   G17P render: STAGE FAIL stage=new-user-render-storage errno=-2
 *
 * A no-op VM_BIND has no such registry, so a driver that never establishes
 * these bindings -- or that later takes one away -- looks perfectly healthy
 * under the shim and only fails on the target.
 *
 * The addresses are the kernel's, not a restatement of Mesa's: they come from
 * T8140_NATIVE_CONTEXT_VIEWS, so a driver that moves its context BO without
 * the kernel moving with it fails here rather than on Neo.
 */
struct asahi_t8140_context_view {
   uint64_t source_address;
   uint64_t size;
   uint64_t context_address;
};

static const struct asahi_t8140_context_view t8140_context_views[] = {
   {UINT64_C(0x1000000000), UINT64_C(0x10000), UINT64_C(0)},
   {UINT64_C(0x1000018000), UINT64_C(0x8000), UINT64_C(0x18000)},
   {UINT64_C(0x1000028000), UINT64_C(0x8000), UINT64_C(0x28000)},
   {UINT64_C(0x1000038000), UINT64_C(0x8000), UINT64_C(0x38000)},
   {UINT64_C(0x1000048000), UINT64_C(0x8000), UINT64_C(0x48000)},
   {UINT64_C(0x1000058000), UINT64_C(0x8000), UINT64_C(0x58000)},
   {UINT64_C(0x1000068000), UINT64_C(0xc000), UINT64_C(0x68000)},
};

/*
 * The window the kernel reserves for the client null view for a VM's whole
 * lifetime (T8140_CLIENT_NULL_VIEW_START/SIZE).  file.rs rejects any user bind
 * that overlaps a driver range with EINVAL, so Mesa must bind its low
 * presentations from 0x18000 upwards and never below.
 */
#define T8140_CLIENT_NULL_VIEW_START UINT64_C(0)
#define T8140_CLIENT_NULL_VIEW_SIZE  UINT64_C(0x10000)

struct asahi_t8140_context_slot {
   bool bound;
   uint32_t handle;
   uint64_t object_offset;
};

static struct asahi_t8140_context_slot
   t8140_context_slots[ARRAY_SIZE(t8140_context_views)];
static unsigned t8140_render_submits_checked;
static unsigned t8140_render_submits_unbound;

static bool
t8140_registry_active(void)
{
   return shim_g17_runtime();
}

static bool
range_overlaps(uint64_t a_start, uint64_t a_size, uint64_t b_start,
               uint64_t b_size)
{
   return a_size && b_size && a_start < b_start + b_size &&
          b_start < a_start + a_size;
}

static bool
range_is_superset(uint64_t outer_start, uint64_t outer_size,
                  uint64_t inner_start, uint64_t inner_size)
{
   return outer_size && inner_start >= outer_start &&
          inner_start + inner_size <= outer_start + outer_size;
}

static void
t8140_track_bind(const struct drm_asahi_gem_bind_op *op)
{
   if (!t8140_registry_active())
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
      const struct asahi_t8140_context_view *view = &t8140_context_views[i];

      /* Order matters and is the kernel's: a bind that merely overlaps the
       * view invalidates the slot, and only a bind that fully contains it
       * re-establishes one. */
      if (range_overlaps(op->addr, op->range, view->source_address,
                         view->size))
         t8140_context_slots[i].bound = false;

      if (range_is_superset(op->addr, op->range, view->source_address,
                            view->size)) {
         t8140_context_slots[i] = (struct asahi_t8140_context_slot){
            .bound = true,
            .handle = op->handle,
            .object_offset = op->offset + (view->source_address - op->addr),
         };
      }
   }
}

static void
t8140_untrack_range(uint64_t address, uint64_t size)
{
   if (!t8140_registry_active())
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
      const struct asahi_t8140_context_view *view = &t8140_context_views[i];
      if (range_overlaps(address, size, view->source_address, view->size))
         t8140_context_slots[i].bound = false;
   }
}

/*
 * The submission-time gate, reported in the kernel's own words so a shim log
 * and a Neo dmesg can be compared line for line.
 *
 * Reporting is unconditional; failing the submit is not, because a driver that
 * trips this still records a valid command stream and the other oracles should
 * keep testing it.  ASAHI_SHIM_G17P_ENFORCE_CONTEXT_VIEWS reproduces the
 * kernel's ENOENT, and ASAHI_SHIM_G17P_EXPECT_CONTEXT_VIEWS asserts at exit
 * that no render submission was ever missing a view.
 */
static bool
t8140_check_context_views(void)
{
   if (!t8140_registry_active())
      return true;

   t8140_render_submits_checked++;

   bool complete = true;
   for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
      if (t8140_context_slots[i].bound)
         continue;

      const struct asahi_t8140_context_view *view = &t8140_context_views[i];
      fprintf(stderr,
              "shim: MMU: T8140 job context view %u source %#llx+%#llx -> "
              "%#llx is not bound\n",
              i, (unsigned long long)view->source_address,
              (unsigned long long)view->size,
              (unsigned long long)view->context_address);
      complete = false;
   }

   if (!complete) {
      for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
         const struct asahi_t8140_context_slot *slot = &t8140_context_slots[i];
         fprintf(stderr,
                 "shim: T8140 context view %u: %s handle=%u object_offset=%#llx\n",
                 i, slot->bound ? "bound" : "UNBOUND", slot->handle,
                 (unsigned long long)slot->object_offset);
      }
   }

   if (!complete)
      t8140_render_submits_unbound++;

   return complete;
}

/*
 * A submit the stream oracle refused is only reported today: the ioctl returns
 * EINVAL, Mesa prints it, and the client carries on and exits 0, so `meson
 * test` stays green over a command stream the kernel would never have run.
 * This knob makes a run fail closed on that.
 */
static unsigned g17p_streams_validated;
static unsigned g17p_streams_rejected;

static void __attribute__((destructor))
check_g17p_stream_validity(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_VALID_STREAM"))
      return;

   if (!g17p_streams_validated || g17p_streams_rejected) {
      fprintf(stderr,
              "shim: G17P command streams accepted by the oracle: %u, "
              "rejected: %u\n",
              g17p_streams_validated, g17p_streams_rejected);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: all %u G17P command streams were accepted\n",
           g17p_streams_validated);
}

static void __attribute__((destructor))
check_t8140_context_views(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_CONTEXT_VIEWS"))
      return;

   if (!t8140_render_submits_checked || t8140_render_submits_unbound) {
      fprintf(stderr,
              "shim: G17P render submissions with a complete native context "
              "view set: %u of %u\n",
              t8140_render_submits_checked - t8140_render_submits_unbound,
              t8140_render_submits_checked);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: all %u G17P render submissions carried the complete native "
           "context view set\n",
           t8140_render_submits_checked);
}

/* Mirrors AGX_G17P_RENDER_AUX_ALIAS_* in agx_va_runtime_hole.h. */
#define SHIM_G17P_RENDER_AUX_ALIAS_START UINT64_C(0x10002000000)
#define SHIM_G17P_RENDER_AUX_ALIAS_END   UINT64_C(0x10002400000)

static unsigned submit_calls;

static struct asahi_g17p_stream_stats stream_totals;

static struct asahi_g17p_compute_stats compute_stats;
static unsigned synchronized_render_submits;

/* What the swap lifecycle check reads, summed over the run's streams. */
static void
merge_stream_stats(const struct asahi_g17p_stream_stats *stats)
{
   stream_totals.draws += stats->draws;
   stream_totals.state_updates += stats->state_updates;
   stream_totals.elided_bind_tails += stats->elided_bind_tails;
   stream_totals.incremental_fs_ppp_updates += stats->incremental_fs_ppp_updates;
   stream_totals.incremental_cull_ppp_updates +=
      stats->incremental_cull_ppp_updates;
   stream_totals.no_varying_fs |= stats->no_varying_fs;
}

static void __attribute__((destructor))
check_bind_tail_counts(void)
{
   const char *spec = getenv("ASAHI_SHIM_G17P_EXPECT_BIND_TAIL_COUNTS");
   if (!spec || !spec[0])
      return;

   unsigned exp_full = 0, exp_elided = 0, exp_fs_ppp = 0, exp_cull_ppp = 0;
   const int parsed = sscanf(spec, "%u,%u,%u,%u", &exp_full, &exp_elided,
                             &exp_fs_ppp, &exp_cull_ppp);
   if (parsed != 3 && parsed != 4) {
      fprintf(stderr, "shim: invalid ASAHI_SHIM_G17P_EXPECT_BIND_TAIL_COUNTS=%s\n",
              spec);
      _Exit(EXIT_FAILURE);
   }

   const unsigned full =
      stream_totals.state_updates - stream_totals.elided_bind_tails;
   if (full != exp_full || stream_totals.elided_bind_tails != exp_elided ||
       stream_totals.incremental_fs_ppp_updates != exp_fs_ppp ||
       (parsed == 4 &&
        stream_totals.incremental_cull_ppp_updates != exp_cull_ppp)) {
      if (parsed == 4) {
         fprintf(stderr,
                 "shim: G17P bind-tail counts mismatch: got full=%u elided=%u "
                 "fs_ppp=%u cull_ppp=%u, expected %u,%u,%u,%u\n",
                 full, stream_totals.elided_bind_tails,
                 stream_totals.incremental_fs_ppp_updates,
                 stream_totals.incremental_cull_ppp_updates, exp_full,
                 exp_elided, exp_fs_ppp, exp_cull_ppp);
      } else {
         fprintf(stderr,
                 "shim: G17P bind-tail counts mismatch: got full=%u elided=%u "
                 "fs_ppp=%u, expected %u,%u,%u\n",
                 full, stream_totals.elided_bind_tails,
                 stream_totals.incremental_fs_ppp_updates, exp_full,
                 exp_elided, exp_fs_ppp);
      }
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: G17P bind-tail counts matched: full=%u elided=%u fs_ppp=%u "
           "cull_ppp=%u\n",
           full, stream_totals.elided_bind_tails,
           stream_totals.incremental_fs_ppp_updates,
           stream_totals.incremental_cull_ppp_updates);
}

static void __attribute__((destructor))
check_swap_lifecycle_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_SWAP_LIFECYCLE"))
      return;

   /* Eight swaps at each of two window geometries. Each swap must retire one
    * real draw; a lost back buffer, stale context binding, or resize failure
    * makes the count short even when EGL itself reports success. */
   if (stream_totals.draws != 16 || stream_totals.state_updates != 16 ||
       synchronized_render_submits != 16 || !stream_totals.no_varying_fs) {
      fprintf(stderr,
              "shim: incomplete G17P swap lifecycle: draws=%u state=%u "
              "synchronized=%u no-varying=%u\n",
              stream_totals.draws, stream_totals.state_updates,
              synchronized_render_submits, stream_totals.no_varying_fs);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P 16-frame synchronized swap/resize lifecycle "
           "coverage\n");
}

static void __attribute__((destructor))
check_compute_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_COMPUTE"))
      return;

   if (!compute_stats.launches ||
       compute_stats.launches !=
          compute_stats.barriers + compute_stats.unbarriered_launches ||
       !compute_stats.terminated ||
       (getenv("ASAHI_SHIM_G17P_EXPECT_STRONG_COMPUTE_BARRIER") &&
        !compute_stats.strong_barriers) ||
       (getenv("ASAHI_SHIM_G17P_EXPECT_BACK_TO_BACK") &&
        !compute_stats.unbarriered_launches) ||
       (getenv("ASAHI_SHIM_G17P_EXPECT_STRONG_INDIRECT_COMPUTE_BARRIER") &&
        !compute_stats.indirect_strong_barriers)) {
      fprintf(stderr,
              "shim: incomplete G17P compute coverage: launches=%u "
              "back-to-back=%u barriers=%u strong=%u indirect=%u "
              "indirect-strong=%u terminated=%u\n",
              compute_stats.launches, compute_stats.unbarriered_launches,
              compute_stats.barriers, compute_stats.strong_barriers,
              compute_stats.indirect_launches,
              compute_stats.indirect_strong_barriers,
              compute_stats.terminated);
      _Exit(EXIT_FAILURE);
   }
}

static bool
range_contains(uint64_t base, uint64_t size, uint64_t address,
               size_t access_size)
{
   return address >= base && access_size <= size &&
          address - base <= size - access_size;
}

static const struct asahi_vm_mapping *
find_mapping(int fd, uint64_t address, size_t size)
{
   for (size_t i = shim_mappings.count; i > 0; --i) {
      const struct asahi_vm_mapping *mapping = &shim_mappings.entries[i - 1];
      if (mapping->fd == fd &&
          range_contains(mapping->address, mapping->size, address, size))
         return mapping;
   }

   return NULL;
}

static void
remove_mappings(int fd, uint64_t address, uint64_t size)
{
   const uint64_t end = address + size;
   for (size_t i = 0; i < shim_mappings.count;) {
      struct asahi_vm_mapping *mapping = &shim_mappings.entries[i];
      const uint64_t mapping_end = mapping->address + mapping->size;
      if (mapping->fd != fd || address >= mapping_end ||
          mapping->address >= end) {
         ++i;
         continue;
      }

      if (getenv("ASAHI_SHIM_TRACE_G17P_STREAM"))
         fprintf(stderr, "shim: unmap %#llx..%#llx (bound %#llx..%#llx)\n",
                 (unsigned long long)address, (unsigned long long)end,
                 (unsigned long long)mapping->address,
                 (unsigned long long)mapping_end);

      drm_shim_bo_put(mapping->bo);
      *mapping = shim_mappings.entries[--shim_mappings.count];
   }
}

static bool
add_mapping(int fd, const struct drm_asahi_gem_bind_op *op)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, op->handle);
   const bool single_page = op->flags & DRM_ASAHI_BIND_SINGLE_PAGE;
   const uint64_t object_range = single_page ? UINT64_C(0x4000) : op->range;
   if (!bo || op->offset > bo->size ||
       object_range > bo->size - op->offset) {
      if (bo)
         drm_shim_bo_put(bo);
      return false;
   }

   remove_mappings(fd, op->addr, op->range);
   if (shim_mappings.count == shim_mappings.capacity) {
      const size_t capacity = MAX2(shim_mappings.capacity * 2, 64);
      void *entries = realloc(shim_mappings.entries,
                              capacity * sizeof(*shim_mappings.entries));
      if (!entries) {
         drm_shim_bo_put(bo);
         return false;
      }
      shim_mappings.entries = entries;
      shim_mappings.capacity = capacity;
   }

   shim_mappings.entries[shim_mappings.count++] = (struct asahi_vm_mapping){
      .fd = fd,
      .address = op->addr,
      .size = op->range,
      .object_offset = op->offset,
      .single_page = single_page,
      .bo = bo,
   };
   if (getenv("ASAHI_SHIM_TRACE_G17P_STREAM"))
      fprintf(stderr, "shim: map %#llx..%#llx handle %u offset %#llx\n",
              (unsigned long long)op->addr,
              (unsigned long long)(op->addr + op->range), op->handle,
              (unsigned long long)op->offset);
   return true;
}

static bool
shim_gpu_owned(void *data, uint64_t address, size_t size)
{
   return find_mapping((int)(intptr_t)data, address, size) != NULL;
}

static bool
shim_gpu_span(void *data, uint64_t address, size_t *available)
{
   const struct asahi_vm_mapping *mapping =
      find_mapping((int)(intptr_t)data, address, 1);
   if (!mapping || !available)
      return false;

   *available = mapping->size - (address - mapping->address);
   return true;
}

static bool
shim_gpu_read(void *data, uint64_t address, void *dst, size_t size)
{
   const struct asahi_vm_mapping *mapping =
      find_mapping((int)(intptr_t)data, address, size);
   if (!mapping)
      return false;

   size_t copied = 0;
   while (copied < size) {
      const uint64_t mapping_offset = address - mapping->address + copied;
      const uint64_t page_offset = mapping_offset & UINT64_C(0x3fff);
      const size_t chunk = mapping->single_page
                              ? MIN2(size - copied,
                                     (size_t)(UINT64_C(0x4000) - page_offset))
                              : size - copied;
      const uint64_t bo_offset =
         mapping->object_offset +
         (mapping->single_page ? page_offset : mapping_offset);
      const uint64_t file_offset = mapping->bo->mem_addr + bo_offset;
      if (pread(shim_device.mem_fd, (uint8_t *)dst + copied, chunk,
                (off_t)file_offset) != (ssize_t)chunk)
         return false;
      copied += chunk;
   }

   return true;
}

/*
 * ASAHI_SHIM_DUMP_SUBMIT=<directory>: byte-exactness oracle for driver CPU
 * work (draw-cpu-20260926). At every submit, append to one file per process
 * the submit's command buffer (hex) and, for every VM mapping of the
 * submitting fd in address order, its address, size and a 64-bit hash of its
 * current contents (a single-page mapping hashes its one page). Two drivers
 * that emit identical GPU-visible bytes into identically placed allocations
 * produce identical files. The file is named after a hash of the command
 * line and the ASAHI_ / HK_ / AGX_ / MESA_ / BENCH_ environment, so the same
 * test run against two drivers lands in the same file name. Diagnostic only;
 * it changes no ioctl result.
 */
static uint64_t
dump_hash_bytes(uint64_t h, const uint8_t *p, size_t n)
{
   for (size_t i = 0; i < n; ++i) {
      h ^= p[i];
      h *= UINT64_C(0x100000001b3);
   }
   return h;
}

/* Replace a build-output directory prefix ("/.../outputs/mesa/<build>/")
 * with "BUILD/" so the same test run from two build trees hashes alike. */
static size_t
dump_normalize(const char *in, size_t len, char *out, size_t cap)
{
   size_t o = 0;
   for (size_t i = 0; i < len && o + 8 < cap;) {
      const char *marker = "/outputs/mesa/";
      const size_t mlen = strlen(marker);
      if (i + mlen <= len && !memcmp(in + i, marker, mlen)) {
         size_t j = i + mlen;
         while (j < len && in[j] != '/' && in[j] != '\0')
            ++j;
         /* Drop what precedes the marker in this string too. */
         size_t k = o;
         while (k > 0 && out[k - 1] != '\0' && out[k - 1] != '=' &&
                out[k - 1] != ' ' && out[k - 1] != ':')
            --k;
         o = k;
         memcpy(out + o, "BUILD", 5);
         o += 5;
         i = j;
         continue;
      }
      out[o++] = in[i++];
   }
   return o;
}

static FILE *
dump_submit_file(void)
{
   static FILE *file;
   static bool tried;
   if (tried)
      return file;
   tried = true;

   const char *dir = getenv("ASAHI_SHIM_DUMP_SUBMIT");
   if (!dir || !*dir)
      return NULL;

   uint64_t h = UINT64_C(0xcbf29ce484222325);
   static char raw[65536], norm[65536];
   size_t raw_len = 0, norm_len = 0;
   FILE *cmd = fopen("/proc/self/cmdline", "rb");
   if (cmd) {
      raw_len = fread(raw, 1, sizeof(raw) - 1, cmd);
      fclose(cmd);
      norm_len = dump_normalize(raw, raw_len, norm, sizeof(norm));
      h = dump_hash_bytes(h, (const uint8_t *)norm, norm_len);
   }
   extern char **environ;
   /* Order-independent over the selected variables. */
   uint64_t env_sum = 0;
   for (char **e = environ; e && *e; ++e) {
      if (strncmp(*e, "ASAHI_", 6) && strncmp(*e, "HK_", 3) &&
          strncmp(*e, "AGX_", 4) && strncmp(*e, "MESA_", 5) &&
          strncmp(*e, "BENCH_", 6))
         continue;
      if (!strncmp(*e, "ASAHI_SHIM_DUMP_SUBMIT=", 23))
         continue;
      static char env_norm[8192];
      const size_t env_len =
         dump_normalize(*e, strlen(*e), env_norm, sizeof(env_norm));
      env_sum += dump_hash_bytes(UINT64_C(0xcbf29ce484222325),
                                 (const uint8_t *)env_norm, env_len);
   }
   h = dump_hash_bytes(h, (const uint8_t *)&env_sum, sizeof(env_sum));

   const char *name = "prog";
   FILE *comm = fopen("/proc/self/comm", "r");
   static char comm_name[64];
   if (comm) {
      if (fgets(comm_name, sizeof(comm_name), comm)) {
         comm_name[strcspn(comm_name, "\n")] = 0;
         name = comm_name;
      }
      fclose(comm);
   }

   char path[4096];
   for (unsigned k = 0; k < 64; ++k) {
      snprintf(path, sizeof(path), "%s/%s-%016llx.%u.dump", dir, name,
               (unsigned long long)h, k);
      file = fopen(path, "wx");
      if (file)
         break;
   }
   if (file) {
      fputs("CMD", file);
      for (size_t i = 0; i < norm_len; ++i)
         fputc(norm[i] ? norm[i] : ' ', file);
      fputc('\n', file);
   }
   return file;
}

static int
dump_mapping_cmp(const void *a, const void *b)
{
   const struct asahi_vm_mapping *x = *(const struct asahi_vm_mapping *const *)a;
   const struct asahi_vm_mapping *y = *(const struct asahi_vm_mapping *const *)b;
   return x->address < y->address ? -1 : x->address > y->address;
}

static void
dump_submit(int fd, const struct drm_asahi_submit *submit, unsigned index)
{
   FILE *f = dump_submit_file();
   if (!f)
      return;

   const uint8_t *cmdbuf = (const void *)(uintptr_t)submit->cmdbuf;
   fprintf(f, "SUBMIT %u queue=%u in_syncs=%u out_syncs=%u cmdbuf_size=%u\nCMDBUF ",
           index, submit->queue_id, submit->in_sync_count,
           submit->out_sync_count, submit->cmdbuf_size);
   for (uint32_t i = 0; cmdbuf && i < submit->cmdbuf_size; ++i)
      fprintf(f, "%02x", cmdbuf[i]);
   fputc('\n', f);

   const struct asahi_vm_mapping **list =
      malloc(sizeof(*list) * (shim_mappings.count + 1));
   size_t n = 0;
   for (size_t i = 0; i < shim_mappings.count; ++i) {
      if (shim_mappings.entries[i].fd == fd)
         list[n++] = &shim_mappings.entries[i];
   }
   qsort(list, n, sizeof(*list), dump_mapping_cmp);

   static uint8_t *chunk;
   const size_t chunk_size = 1 << 20;
   if (!chunk)
      chunk = malloc(chunk_size);

   for (size_t i = 0; i < n; ++i) {
      const struct asahi_vm_mapping *m = list[i];
      const uint64_t bytes = m->single_page ? UINT64_C(0x4000) : m->size;
      uint64_t h = UINT64_C(0xcbf29ce484222325);
      for (uint64_t off = 0; off < bytes; off += chunk_size) {
         const size_t len = MIN2(chunk_size, bytes - off);
         const uint64_t file_offset =
            m->bo->mem_addr + m->object_offset + off;
         if (pread(shim_device.mem_fd, chunk, len, (off_t)file_offset) !=
             (ssize_t)len) {
            memset(chunk, 0xee, len);
         }
         h = dump_hash_bytes(h, chunk, len);
      }
      fprintf(f, "MAP %#llx size=%#llx single=%u hash=%016llx backing=%#llx offset=%#llx\n",
              (unsigned long long)m->address, (unsigned long long)m->size,
              m->single_page, (unsigned long long)h,
              (unsigned long long)m->bo->mem_addr,
              (unsigned long long)m->object_offset);

      /* ASAHI_SHIM_DUMP_BYTES=<hex address>: also write that mapping's raw
       * bytes, one file per submit, next to the dump. */
      const char *want = getenv("ASAHI_SHIM_DUMP_BYTES");
      if (want && (!strcmp(want, "all") || strtoull(want, NULL, 16) == m->address)) {
         char path[4096];
         snprintf(path, sizeof(path), "%s/bytes-%llx-%u.bin",
                  getenv("ASAHI_SHIM_DUMP_SUBMIT"),
                  (unsigned long long)m->address, index);
         FILE *raw = fopen(path, "wb");
         for (uint64_t off = 0; raw && off < bytes; off += chunk_size) {
            const size_t len = MIN2(chunk_size, bytes - off);
            if (pread(shim_device.mem_fd, chunk, len,
                      (off_t)(m->bo->mem_addr + m->object_offset + off)) ==
                (ssize_t)len)
               fwrite(chunk, 1, len, raw);
         }
         if (raw)
            fclose(raw);
      }
   }
   free(list);
   fflush(f);
}

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
asahi_ioctl_queue_create(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_queue_create *create = arg;
   static uint32_t next_queue_id = 1;

   /* Model authorization and a credential change after the screen probe,
    * and ("unsupported") a GPU backend that does not schedule HIGH. The
    * kernel reserves REALTIME everywhere. */
   static unsigned high_requests;
   const char *policy = getenv("ASAHI_SHIM_HIGH_PRIORITY");
   if (create->priority > DRM_ASAHI_PRIORITY_HIGH ||
       (create->priority == DRM_ASAHI_PRIORITY_HIGH && policy &&
        !strcmp(policy, "unsupported"))) {
      errno = EINVAL;
      return -1;
   }
   if (create->priority == DRM_ASAHI_PRIORITY_HIGH) {
      high_requests++;
      if (policy && (!strcmp(policy, "deny") ||
                     (!strcmp(policy, "once") && high_requests > 1))) {
         errno = EACCES;
         return -1;
      }
   }
   shim_usc_exec_base = create->usc_exec_base;
   create->queue_id = next_queue_id++;
   return 0;
}

static int
asahi_ioctl_queue_destroy(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
asahi_ioctl_gem_bind_object(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_gem_bind_object *bind = arg;

   switch (bind->op) {
   case DRM_ASAHI_BIND_OBJECT_OP_BIND:
      if (!bind->handle ||
          bind->flags != DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS) {
         errno = EINVAL;
         return -1;
      }

      /* The no-op winsys has no firmware object table.  Preserve a stable,
       * nonzero identity so timestamp-pool recording exercises the same
       * driver lifetime path without pretending an object was submitted. */
      bind->object_handle = bind->handle;
      return 0;

   case DRM_ASAHI_BIND_OBJECT_OP_UNBIND:
      if (!bind->object_handle) {
         errno = EINVAL;
         return -1;
      }
      return 0;

   default:
      errno = EINVAL;
      return -1;
   }
}

static int
asahi_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_create *create = arg;
   struct asahi_bo *bo = calloc(1, sizeof(*bo));

   if (!bo) {
      errno = ENOMEM;
      return -1;
   }

   int ret = drm_shim_bo_init(&bo->base, create->size);
   if (ret) {
      free(bo);
      errno = -ret;
      return -1;
   }

   assert(UINT64_MAX - asahi.next_offset > create->size);
   bo->offset = asahi.next_offset;
   bo->gem_flags = create->flags;
   asahi.next_offset += create->size;

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);

   if (getenv("ASAHI_SHIM_TRACE_G17P_STREAM"))
      fprintf(stderr,
              "shim: gem_create size=%#llx flags=%#x%s vm_id=%u -> handle %u\n",
              (unsigned long long)create->size, create->flags,
              (create->flags & DRM_ASAHI_GEM_VM_PRIVATE) ? " (VM_PRIVATE)"
                                                         : "",
              create->vm_id, create->handle);

   drm_shim_bo_put(&bo->base);

   return 0;
}

static int
asahi_ioctl_gem_mmap_offset(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_mmap_offset *map = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, map->handle);

   map->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

/* v48 VM-status mirror checks (test infrastructure only; off by default).
 * ASAHI_SHIM_VM_STATUS=1 advertises group 1, ASAHI_SHIM_VM_STATUS_MIRROR=1
 * advertises group 3, ASAHI_SHIM_VM_STATUS_ERROR=<errno> is the sticky VM
 * error both report. Counts are printed at exit for the harness. */
static unsigned shim_vm_status_queries;
static unsigned shim_vm_status_mirrors;

static bool
shim_env_on(const char *name)
{
   const char *value = getenv(name);
   return value && !strcmp(value, "1");
}

static int32_t
shim_vm_status_error(void)
{
   const char *value = getenv("ASAHI_SHIM_VM_STATUS_ERROR");
   return value ? (int32_t)strtol(value, NULL, 0) : 0;
}

static void
shim_vm_status_report(void)
{
   fprintf(stderr, "shim: vm-status queries=%u mirrors=%u\n",
           shim_vm_status_queries, shim_vm_status_mirrors);
}

static void
shim_vm_status_register_report(void)
{
   static bool registered;
   if (!registered && (shim_env_on("ASAHI_SHIM_VM_STATUS") ||
                       shim_env_on("ASAHI_SHIM_VM_STATUS_MIRROR"))) {
      registered = true;
      atexit(shim_vm_status_report);
   }
}

static int
asahi_ioctl_vm_status_mirror(int fd, struct drm_asahi_get_params *gp)
{
   if (!shim_env_on("ASAHI_SHIM_VM_STATUS_MIRROR") &&
       !(shim_params()->features & DRM_ASAHI_FEATURE_VM_STATUS_MIRROR)) {
      errno = EOPNOTSUPP;
      return -1;
   }
   struct drm_asahi_vm_status_mirror *m = (void *)(uintptr_t)gp->pointer;
   if (gp->size < sizeof(*m) || m->flags || (m->offset & 3)) {
      errno = EINVAL;
      return -1;
   }
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, m->handle);
   if (!bo || (uint64_t)m->offset + 4 > bo->size) {
      if (bo)
         drm_shim_bo_put(bo);
      errno = EINVAL;
      return -1;
   }
   const long page = sysconf(_SC_PAGESIZE);
   const uint64_t at = bo->mem_addr + m->offset;
   const uint64_t base = at & ~(uint64_t)(page - 1);
   char *map = mmap(NULL, page, PROT_READ | PROT_WRITE, MAP_SHARED,
                    shim_device.mem_fd, base);
   drm_shim_bo_put(bo);
   if (map == MAP_FAILED)
      return -1;
   int32_t *word = (int32_t *)(map + (at - base));
   __atomic_store_n(word, 0, __ATOMIC_SEQ_CST);
   const int32_t error = shim_vm_status_error();
   if (error)
      __atomic_store_n(word, error, __ATOMIC_SEQ_CST);
   munmap(map, page);
   shim_vm_status_mirrors++;
   return 0;
}

/* The feature word GET_PARAMS reports: the identity's own bits plus the
 * ASAHI_SHIM_* switches, so both old and capability-aware kernels can be
 * tested without GPU execution. */
static uint64_t
shim_features(void)
{
   uint64_t features = shim_params()->features;
   const char *per_command = getenv("ASAHI_SHIM_PER_COMMAND_INPUT_SYNCS");
   if (per_command && !strcmp(per_command, "1"))
      features |= DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS;
   if (shim_env_on("ASAHI_SHIM_VM_STATUS"))
      features |= DRM_ASAHI_FEATURE_VM_STATUS;
   if (shim_env_on("ASAHI_SHIM_VM_STATUS_MIRROR"))
      features |= DRM_ASAHI_FEATURE_VM_STATUS_MIRROR;
   /* A kernel that admits render ray queries, which G17P ray tracing
    * pipelines need; the shim executes nothing either way. */
   if (shim_env_on("ASAHI_SHIM_RENDER_RAY_QUERY"))
      features |= DRM_ASAHI_FEATURE_RENDER_RAY_QUERY;
   /* An M5 kernel without the compressed multisample ZLS opt-in, as the
    * M5 Max (J775c) kernel reports its features: its render validation, and
    * so the M5 submission gate, then refuses a single-layer multisample
    * render with compressed depth or stencil. */
   if (shim_env_on("ASAHI_SHIM_NO_M5_MSAA_ZLS_COMPRESSION"))
      features &= ~(uint64_t)(DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION |
                              DRM_ASAHI_FEATURE_M5_STENCIL_MSAA_COMPRESSION);
   /* The M6 (G18G) features (bits 31 and 32); only the G18G identity
    * reports them. */
   if (g18g_profile()) {
      /* Replays both the refusing and the suffix-emitting G18 graphics
       * scratch paths; the shim executes nothing either way. */
      if (shim_env_on("ASAHI_SHIM_G18_GRAPHICS_SCRATCH"))
         features |= DRM_ASAHI_FEATURE_G18_GRAPHICS_SCRATCH;
      /* Replays a kernel that admits the native 40/56/72-byte 32x32 tiles. */
      if (shim_env_on("ASAHI_SHIM_G18_NATIVE_TILE_SAMPLES"))
         features |= DRM_ASAHI_FEATURE_G18_NATIVE_TILE_SAMPLES;
   }
   return features;
}

static uint64_t
shim_vm_end(const struct drm_asahi_params_global *params)
{
   if (params == &params_g15g && shim_env_on("ASAHI_SHIM_G15G_HAL200"))
      return UINT64_C(0x3ffffff8000);
   return params->vm_end;
}

static int
asahi_ioctl_get_param(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *gp = arg;

   shim_vm_status_register_report();
   switch (gp->param_group) {
   case 0: {
      /* Mirror the kernel, which copies min(sizeof(params), data.size) and
       * reports back what it wrote (file.rs get_params).  Asserting equality
       * instead made the shim unusable against any caller whose header
       * differs by a field. */
      size_t size = MIN2((size_t)gp->size, sizeof(struct drm_asahi_params_global));
      struct drm_asahi_params_global params = *shim_params();
      params.features = shim_features();
      if (shim_params() == &params_g15g &&
          shim_env_on("ASAHI_SHIM_G15G_HAL200")) {
         params.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
         params.vm_end = shim_vm_end(shim_params());
      }
      const char *command_limit = getenv("ASAHI_SHIM_COMMAND_LIMIT");
      if (command_limit) {
         unsigned limit = strtoul(command_limit, NULL, 0);
         if (limit > 0 && limit <= params.max_commands_per_submission)
            params.max_commands_per_submission = limit;
      }
      memcpy((void *)gp->pointer, &params, size);
      gp->size = size;
      return 0;
   }

   case DRM_ASAHI_PARAM_GROUP_VM_STATUS: {
      if (!shim_env_on("ASAHI_SHIM_VM_STATUS") &&
          !(shim_params()->features & DRM_ASAHI_FEATURE_VM_STATUS)) {
         errno = EINVAL;
         return -1;
      }
      struct drm_asahi_vm_status *status = (void *)(uintptr_t)gp->pointer;
      status->error = shim_vm_status_error();
      shim_vm_status_queries++;
      return 0;
   }

   case DRM_ASAHI_PARAM_GROUP_VM_STATUS_MIRROR:
      return asahi_ioctl_vm_status_mirror(fd, gp);

   case DRM_ASAHI_PARAM_GROUP_QUEUE_LIMITS:
      if (shim_params() == &params_g16s_rt ||
          shim_params() == &params_g16s_as) {
         const struct drm_asahi_queue_limits limits = {
            .max_queues = 8, .max_in_flight_per_queue = 2,
         };
         if (gp->size < sizeof(limits)) {
            errno = EINVAL;
            return -1;
         }
         memcpy((void *)(uintptr_t)gp->pointer, &limits, sizeof(limits));
         gp->size = sizeof(limits);
         return 0;
      }
      if (shim_g17_runtime() && shim_params() != &params_g17p) {
         if (gp->size < sizeof(queue_limits_g17)) {
            errno = EINVAL;
            return -1;
         }
         memcpy((void *)(uintptr_t)gp->pointer, &queue_limits_g17,
                sizeof(queue_limits_g17));
         gp->size = sizeof(queue_limits_g17);
         return 0;
      }
      if (!shim_g15s() || gp->size < sizeof(queue_limits_g15s)) {
         errno = EINVAL;
         return -1;
      }
      memcpy((void *)(uintptr_t)gp->pointer, &queue_limits_g15s,
             sizeof(queue_limits_g15s));
      gp->size = sizeof(queue_limits_g15s);
      return 0;

   default:
      fprintf(stderr, "Unknown DRM_IOCTL_ASAHI_GET_PARAMS %d\n",
              gp->param_group);
      return -1;
   }
}

/* The kernel's VA rules, mirrored so a bad address layout fails here instead
 * of costing a boot on hardware.
 *
 * drivers/gpu/drm/asahi/file.rs rejects a bind that is not contained in the
 * device user range, or that overlaps the VM's kernel range, both with EINVAL.
 * A no-op VM_BIND accepts every address, so an allocator geometry that the
 * real driver refuses looks perfectly healthy under the shim.  These two
 * checks are the whole difference between reproducing that on a laptop and
 * discovering it on the target. */
static struct {
   uint64_t kernel_start, kernel_end;
   bool valid;
} shim_vm;

static int
asahi_ioctl_vm_create_unlocked(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_vm_create *create = arg;

   shim_vm.kernel_start = create->kernel_start;
   shim_vm.kernel_end = create->kernel_end;
   shim_vm.valid = true;
   create->vm_id = 1;
   return 0;
}

static int
asahi_ioctl_vm_bind_unlocked(int fd, unsigned long request, void *arg)
{
   const struct drm_asahi_vm_bind *bind = arg;
   const struct drm_asahi_params_global *params = shim_params();
   const uint64_t vm_end = shim_vm_end(params);

   for (uint32_t i = 0; i < bind->num_binds; ++i) {
      const struct drm_asahi_gem_bind_op *op =
         (const struct drm_asahi_gem_bind_op *)(uintptr_t)(bind->userptr +
                                                          (uint64_t)i *
                                                             bind->stride);
      uint64_t start = op->addr;
      uint64_t end = start + op->range;

      if (!op->range) {
         fprintf(stderr, "shim: gem_bind: zero-sized range at %#llx\n",
                 (unsigned long long)start);
         return -EINVAL;
      }

      if (end < start || start < params->vm_start || end > vm_end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(not contained in user range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)params->vm_start,
                 (unsigned long long)vm_end);
         return -EINVAL;
      }

      /* T8140 installs four pair/generation AXFB aliases in every
       * render-capable client root, and that range lies inside G17P's USC
       * heap. The kernel refuses a bind that lands on them; a run that gets
       * its allocator geometry wrong here produces thousands of rejected
       * binds on hardware and nothing at all under a no-op shim. */
      if (shim_g17_runtime() &&
          start < SHIM_G17P_RENDER_AUX_ALIAS_END &&
          SHIM_G17P_RENDER_AUX_ALIAS_START < end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(overlaps the kernel render-aux alias %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)SHIM_G17P_RENDER_AUX_ALIAS_START,
                 (unsigned long long)SHIM_G17P_RENDER_AUX_ALIAS_END);
         return -EINVAL;
      }

      if (params == &params_g15g &&
          (range_overlaps(start, op->range, UINT64_C(0x7000000000),
                          UINT64_C(0x1000000000)) ||
           range_overlaps(start, op->range, UINT64_C(0x10080000000),
                          UINT64_C(0x80000000)) ||
           range_overlaps(start, op->range, UINT64_C(0x1000000000),
                          UINT64_C(0x8000000)) ||
           range_overlaps(start, op->range, UINT64_C(0x1074000000),
                          UINT64_C(0x28000)))) {
         fprintf(stderr, "shim: gem_bind overlaps a T8122 runtime reservation\n");
         return -EINVAL;
      }

      if (shim_g15_kernel() &&
          range_overlaps(start, op->range, SHIM_G15S_KERNEL_VA_START,
                         SHIM_G15S_KERNEL_VA_END -
                            SHIM_G15S_KERNEL_VA_START)) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(overlaps the G15%c kernel objects %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (char)params->gpu_variant,
                 (unsigned long long)SHIM_G15S_KERNEL_VA_START,
                 (unsigned long long)SHIM_G15S_KERNEL_VA_END);
         return -EINVAL;
      }

      if (shim_vm.valid && start < shim_vm.kernel_end &&
          shim_vm.kernel_start < end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(intrudes in kernel range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)shim_vm.kernel_start,
                 (unsigned long long)shim_vm.kernel_end);
         return -EINVAL;
      }

      /* file.rs rejects any user bind overlapping a driver-reserved range,
       * and the client null view is reserved for the VM's whole lifetime. */
      if (t8140_registry_active() &&
          range_overlaps(start, op->range, T8140_CLIENT_NULL_VIEW_START,
                         T8140_CLIENT_NULL_VIEW_SIZE)) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx (overlaps "
                 "the driver-reserved T8140 client null view %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)T8140_CLIENT_NULL_VIEW_START,
                 (unsigned long long)(T8140_CLIENT_NULL_VIEW_START +
                                      T8140_CLIENT_NULL_VIEW_SIZE));
         return -EINVAL;
      }

      if (op->flags & DRM_ASAHI_BIND_UNBIND) {
         remove_mappings(fd, op->addr, op->range);
         t8140_untrack_range(op->addr, op->range);
      } else if (!add_mapping(fd, op)) {
         fprintf(stderr, "shim: gem_bind: invalid BO mapping\n");
         return -EINVAL;
      } else {
         /* A repeated single page is not the same physical layout as any
          * native restricted resource; the kernel refuses the overlap outright
          * (Vm::validate_t8140_native_context_binding). */
         if (t8140_registry_active() &&
             (op->flags & DRM_ASAHI_BIND_SINGLE_PAGE)) {
            for (unsigned v = 0; v < ARRAY_SIZE(t8140_context_views); ++v) {
               if (range_overlaps(op->addr, op->range,
                                  t8140_context_views[v].source_address,
                                  t8140_context_views[v].size)) {
                  fprintf(stderr,
                          "shim: gem_bind: single-page map %#llx..%#llx "
                          "overlaps native context view %u\n",
                          (unsigned long long)start, (unsigned long long)end,
                          v);
                  return -EINVAL;
               }
            }
         }
         t8140_track_bind(op);
      }
   }

   return 0;
}

/* A submission's ordering is carried entirely by the barrier fields in each
 * command header and by which ioctl the in/out syncobjs ride on. Both are
 * write-only from userspace: the kernel is the only thing that has ever read
 * them, so a mistake in what Honeykrisp asks for and a kernel that does not
 * honour the request look identical from a readback that comes back stale.
 * Set ASAHI_SHIM_SUBMIT_LOG to a path and every submit ioctl is appended to
 * it, one line per command, exactly as the kernel would parse it. */
static void
asahi_log_submit(const struct drm_asahi_submit *submit)
{
   const char *path = getenv("ASAHI_SHIM_SUBMIT_LOG");
   if (!path)
      return;

   FILE *log = fopen(path, "a");
   if (!log)
      return;

   static unsigned ioctl_index;
   fprintf(log, "submit %u in_syncs=%u out_syncs=%u bytes=%u\n", ioctl_index++,
           submit->in_sync_count, submit->out_sync_count, submit->cmdbuf_size);

   const uint8_t *cmdbuf = (const uint8_t *)(uintptr_t)submit->cmdbuf;
   uint32_t offset = 0;
   while (offset + sizeof(struct drm_asahi_cmd_header) <= submit->cmdbuf_size) {
      struct drm_asahi_cmd_header header;
      memcpy(&header, cmdbuf + offset, sizeof(header));

      const char *name = header.cmd_type == DRM_ASAHI_CMD_RENDER    ? "render"
                         : header.cmd_type == DRM_ASAHI_CMD_COMPUTE ? "compute"
                         : header.cmd_type == DRM_ASAHI_SET_VERTEX_ATTACHMENTS
                            ? "set-vertex-attachments"
                         : header.cmd_type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS
                            ? "set-fragment-attachments"
                         : header.cmd_type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS
                            ? "set-compute-attachments"
                            : "unknown";

      fprintf(log, "  cmd %s type=%u size=%u vdm_barrier=%u cdm_barrier=%u",
              name, header.cmd_type, header.size, header.vdm_barrier,
              header.cdm_barrier);

      /* The declared control-stream extent is the other write-only field the
       * kernel acts on. Print it: a stream whose declared end is far past the
       * records it actually contains is indistinguishable from a correct one
       * until the hardware reads the gap. */
      if (header.cmd_type == DRM_ASAHI_CMD_COMPUTE &&
          header.size >= sizeof(struct drm_asahi_cmd_compute)) {
         struct drm_asahi_cmd_compute compute;
         memcpy(&compute, cmdbuf + offset + sizeof(header), sizeof(compute));
         fprintf(log, " cdm_base=0x%llx cdm_end=0x%llx cdm_len=%llu",
                 (unsigned long long)compute.cdm_ctrl_stream_base,
                 (unsigned long long)compute.cdm_ctrl_stream_end,
                 (unsigned long long)(compute.cdm_ctrl_stream_end -
                                      compute.cdm_ctrl_stream_base));
         fprintf(log, "\n");
         /* Timestamp destinations are the other write-only request: a
          * query folded onto a command is visible only here. Own line, so
          * the "  cmd" line keeps the shape the recorded tests parse. */
         if (compute.ts.start.handle || compute.ts.end.handle)
            fprintf(log, "    ts start=%u:%u end=%u:%u\n",
                    compute.ts.start.handle, compute.ts.start.offset,
                    compute.ts.end.handle, compute.ts.end.offset);
         offset += sizeof(header) + header.size;
         continue;
      } else if (header.cmd_type == DRM_ASAHI_CMD_RENDER &&
                 header.size >= offsetof(struct drm_asahi_cmd_render,
                                         bg_eot_rsrc_spec_hi)) {
         /* Only G17P submits the resource-specifier high words. */
         struct drm_asahi_cmd_render render = {0};
         memcpy(&render, cmdbuf + offset + sizeof(header),
                MIN2(sizeof(render), header.size));
         fprintf(log, " vdm_base=0x%llx",
                 (unsigned long long)render.vdm_ctrl_stream_base);
         /* ZLS: the compression bits (2/4/6/8) and the metadata base and
          * stride are write-only too; print them for depth compression. */
         if (render.depth.base || render.stencil.base)
            fprintf(log,
                    " zls_ctrl=0x%llx depth=0x%llx/0x%llx stride=0x%x/0x%x"
                    " stencil=0x%llx/0x%llx stride=0x%x/0x%x",
                    (unsigned long long)render.zls_ctrl,
                    (unsigned long long)render.depth.base,
                    (unsigned long long)render.depth.comp_base,
                    render.depth.stride, render.depth.comp_stride,
                    (unsigned long long)render.stencil.base,
                    (unsigned long long)render.stencil.comp_base,
                    render.stencil.stride, render.stencil.comp_stride);
         fprintf(log, "\n");
         if (render.ts_vtx.start.handle || render.ts_frag.end.handle)
            fprintf(log, "    ts start=%u:%u end=%u:%u\n",
                    render.ts_vtx.start.handle, render.ts_vtx.start.offset,
                    render.ts_frag.end.handle, render.ts_frag.end.offset);
         offset += sizeof(header) + header.size;
         continue;
      }
      fprintf(log, "\n");

      offset += sizeof(header) + header.size;
   }
   if (offset != submit->cmdbuf_size)
      fprintf(log, "  ragged walk ended at %u of %u\n", offset,
              submit->cmdbuf_size);

   fclose(log);
}

/* 0 (default) silent, 1 report, 2 reject. See asahi_g17p_stream_validator.h. */
static unsigned
g17p_strict_descriptors(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_STRICT_DESCRIPTORS");

   if (value == NULL || value[0] == '\0')
      return 0;

   return (unsigned)strtoul(value, NULL, 10);
}

/*
 * Validation normally rejects the submit, which stops the application at the
 * first complaint. A long-running client -- a game walking a level -- then
 * validates only the streams it emitted before that point. Warn-only mode
 * reports every failure and lets the run continue, so one session covers the
 * whole stream population instead of its prefix.
 */
static bool
g17p_stream_warn_only(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_STREAM_WARN_ONLY");

   return value != NULL && value[0] != '\0' && strcmp(value, "0");
}

static unsigned g17p_stream_failures;

static int
g17p_stream_reject(const char *what)
{
   g17p_stream_failures++;

   if (!g17p_stream_warn_only())
      return -EINVAL;

   fprintf(stderr, "shim: continuing after %s failure %u (warn-only)\n", what,
           g17p_stream_failures);
   return 0;
}

/* G15S gate totals; ASAHI_SHIM_G15S_EXPECT_VALID makes any refusal fail the
 * run at exit, so a client that carries on after an EINVAL cannot pass. */
static struct asahi_g15s_gate_stats g15s_stats;
static unsigned g15s_rejected;

static void __attribute__((destructor))
report_g15s_gate(void)
{
   if (!shim_g15_kernel())
      return;

   fprintf(stderr,
           "shim: G15%c gate: %u renders, %u computes, %u launches, %u "
           "indirect, %u barriers (%u wide), %u links; %u rejected\n",
           (char)shim_params()->gpu_variant, g15s_stats.renders, g15s_stats.computes, g15s_stats.launches,
           g15s_stats.indirect_launches, g15s_stats.barriers,
           g15s_stats.wide_barriers, g15s_stats.links, g15s_rejected);

   if (getenv("ASAHI_SHIM_G15S_EXPECT_VALID") && g15s_rejected)
      _Exit(EXIT_FAILURE);
}

static int
g15s_submit_gate(int fd, const struct drm_asahi_submit *submit)
{
   const uint8_t *cmdbuf = (const void *)(uintptr_t)submit->cmdbuf;
   const struct asahi_g15s_gate_memory memory = {
      .data = (void *)(intptr_t)fd,
      .read = shim_gpu_read,
   };
   const uint64_t features = shim_features();
   unsigned commands = 0;
   size_t cursor = 0;

   while (cursor < submit->cmdbuf_size) {
      if (submit->cmdbuf_size - cursor < sizeof(struct drm_asahi_cmd_header))
         return -EINVAL;

      const struct drm_asahi_cmd_header *header =
         (const void *)(cmdbuf + cursor);
      const size_t command_size = sizeof(*header) + header->size;
      if (command_size > submit->cmdbuf_size - cursor)
         return -EINVAL;

      bool ok = true;
      if (header->cmd_type == DRM_ASAHI_CMD_RENDER) {
         struct drm_asahi_cmd_render render = {0};
         memcpy(&render, header + 1, MIN2(sizeof(render), header->size));
         commands++;
         g15s_stats.renders++;
         ok = asahi_g15s_render_packet_validate(&render, header + 1,
                                                header->size,
                                                features) &&
              asahi_g15s_vdm_gate(&memory, render.vdm_ctrl_stream_base);
      } else if (header->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         struct drm_asahi_cmd_compute compute = {0};
         memcpy(&compute, header + 1, MIN2(sizeof(compute), header->size));
         commands++;
         g15s_stats.computes++;
         ok = asahi_g15s_compute_packet_validate(&compute, header + 1,
                                                 header->size,
                                                 features) &&
              asahi_g15s_cdm_gate(&memory, shim_usc_exec_base,
                                  compute.cdm_ctrl_stream_base, &g15s_stats);
      }

      if (!ok) {
         g15s_rejected++;
         return -EINVAL;
      }

      cursor += command_size;
   }

   if (commands > shim_params()->max_commands_per_submission)
      return -EINVAL;

   return 0;
}

/* M5 submission gate totals. ASAHI_SHIM_M5_GATE=1 applies the gate to any
 * M5 identity, =warn reports refusals and accepts the submission, =0 turns
 * it off for g17s-kernel. ASAHI_SHIM_M5_EXPECT_VALID makes a refusal fail
 * the run at exit, so a client that carries on after an EINVAL cannot pass. */
static struct asahi_m5_gate_stats m5_gate_stats;
static unsigned m5_gate_submits, m5_gate_refused;

static bool
shim_m5_gate(void)
{
   const char *gate = getenv("ASAHI_SHIM_M5_GATE");
   if (gate)
      return strcmp(gate, "0") != 0;
   return shim_params() == &params_g17s_kernel;
}

static void __attribute__((destructor))
report_m5_gate(void)
{
   if (!m5_gate_submits)
      return;

   fprintf(stderr,
           "shim: M5 gate: %u submits, %u renders, %u computes, %u render "
           "programs, %u compute programs; %u refused\n",
           m5_gate_submits, m5_gate_stats.renders, m5_gate_stats.computes,
           m5_gate_stats.render_programs, m5_gate_stats.compute_programs,
           m5_gate_refused);

   if (getenv("ASAHI_SHIM_M5_EXPECT_VALID") && m5_gate_refused)
      _Exit(EXIT_FAILURE);
}

static int
m5_submit_gate(const struct drm_asahi_submit *submit)
{
   const char *reason;
   m5_gate_submits++;
   const int ret = asahi_m5_submit_gate((const void *)(uintptr_t)submit->cmdbuf,
                                        submit->cmdbuf_size, shim_features(),
                                        &m5_gate_stats, &reason);
   if (!ret)
      return 0;

   m5_gate_refused++;
   fprintf(stderr, "shim: M5 gate refused submit %u (%s): %s\n", submit_calls,
           strerror(-ret), reason);
   const char *gate = getenv("ASAHI_SHIM_M5_GATE");
   if (gate && !strcmp(gate, "warn"))
      return 0;
   errno = -ret;
   return -1;
}

static int
asahi_ioctl_submit_unlocked(int fd, unsigned long request, void *arg)
{
   if (g17g_compile_only()) {
      fprintf(stderr, "shim: G17G compiler-only profile rejects GPU submission\n");
      errno = EOPNOTSUPP;
      return -1;
   }

   /* Offline triage only: accept (and drop) G18G submissions so one
    * rejected submit does not poison the context and hide every later
    * refusal in the same test. */
   if (g18g_profile() && !getenv("ASAHI_SHIM_G18G_ACCEPT_SUBMIT")) {
      fprintf(stderr, "shim: G18G profile has no kernel contract and rejects "
                      "GPU submission\n");
      errno = EOPNOTSUPP;
      return -1;
   }

   const struct drm_asahi_submit *submit = arg;
   submit_calls++;
   asahi_log_submit(submit);
   dump_submit(fd, submit, submit_calls);

   if (shim_g15_kernel())
      return g15s_submit_gate(fd, submit);

   if (shim_params()->gpu_hal_generation == DRM_ASAHI_GPU_HAL_300 &&
       shim_m5_gate()) {
      const int ret = m5_submit_gate(submit);
      if (ret)
         return ret;
   }

   /* The kernel's native-context gate runs for every render submission, not
    * only when the stream oracle is enabled, so check it first. */
   if (t8140_registry_active()) {
      const uint8_t *scan = (const void *)(uintptr_t)submit->cmdbuf;
      size_t at = 0;
      bool complete = true;

      while (at + sizeof(struct drm_asahi_cmd_header) <= submit->cmdbuf_size) {
         const struct drm_asahi_cmd_header *header = (const void *)(scan + at);
         const size_t command_size = sizeof(*header) + header->size;

         if (command_size > submit->cmdbuf_size - at)
            break;

         if (header->cmd_type == DRM_ASAHI_CMD_RENDER)
            complete &= t8140_check_context_views();

         at += command_size;
      }

      if (!complete && getenv("ASAHI_SHIM_G17P_ENFORCE_CONTEXT_VIEWS")) {
         fprintf(stderr,
                 "shim: G17P render: STAGE FAIL stage=new-user-render-storage "
                 "errno=-2\n");
         errno = ENOENT;
         return -1;
      }
   }

   const char *validate = getenv("ASAHI_SHIM_VALIDATE_G17P_STREAM");
   if (!validate || !strcmp(validate, "0") ||
       (!shim_g17_runtime() && shim_params() != &params_g16g))
      return 0;

   const uint8_t *cmdbuf = (const void *)(uintptr_t)submit->cmdbuf;
   size_t cursor = 0;
   while (cursor < submit->cmdbuf_size) {
      if (submit->cmdbuf_size - cursor < sizeof(struct drm_asahi_cmd_header))
         return -EINVAL;

      const struct drm_asahi_cmd_header *header = (const void *)(cmdbuf + cursor);
      const size_t command_size = sizeof(*header) + header->size;
      if (command_size > submit->cmdbuf_size - cursor)
         return -EINVAL;

      if (header->cmd_type == DRM_ASAHI_CMD_RENDER) {
         if (header->size < offsetof(struct drm_asahi_cmd_render,
                                     vdm_ctrl_stream_base) +
                               sizeof(uint64_t))
            return -EINVAL;

         struct drm_asahi_cmd_render render = {0};
         memcpy(&render, header + 1, MIN2(sizeof(render), header->size));

         /* G17P creates a per-job render-storage view from the fixed HAL200
          * context aperture.  Unlike ordinary client-VM shader and resource
          * mappings, its source GEM must therefore be visible outside the
          * private VM.  Mirror that kernel requirement so a private context
          * allocation fails on the host with the same ENOENT class instead
          * of reaching hardware first. */
         if (shim_g17_runtime()) {
            const struct asahi_vm_mapping *context =
               find_mapping(fd, UINT64_C(0x1000000000), UINT64_C(0x10000));
            if (!context ||
                (asahi_bo(context->bo)->gem_flags & DRM_ASAHI_GEM_VM_PRIVATE)) {
               fprintf(stderr,
                       "shim: G17P render context is absent from shared job view\n");
               return -ENOENT;
            }
         }

         /* A private publication fence may follow the batch outputs. */
         if (submit->in_sync_count > 0 && submit->out_sync_count >= 2)
            synchronized_render_submits++;
         const struct asahi_g17p_stream_memory memory = {
            .data = (void *)(intptr_t)fd,
            .read = shim_gpu_read,
            .owned = shim_gpu_owned,
            .span = shim_gpu_span,
            .verbose = getenv("ASAHI_SHIM_TRACE_G17P_STREAM") != NULL,
            .strict_descriptors = g17p_strict_descriptors(),
         };
         struct asahi_g17p_stream_stats stats;
         const struct asahi_g17p_stream_config config = {
            .root = render.vdm_ctrl_stream_base,
            .scissor_base = render.isp_scissor_base,
            .depth_bias_base = render.isp_dbias_base,
         };
         if (!asahi_g17p_render_aux_validate(&memory, &render) ||
             !asahi_g17p_stream_validate(&memory, &config, &stats)) {
            g17p_streams_rejected++;
            const int rejected = g17p_stream_reject("VDM");
            if (rejected)
               return rejected;
            cursor += command_size;
            continue;
         }
         merge_stream_stats(&stats);
         g17p_streams_validated++;
         fprintf(stderr,
                 "shim: G17P VDM validated: %u draws, %u indexed, %u state "
                 "updates (%u elided, %u fs_ppp, %u cull_ppp), %u links, %u "
                 "barriers (%u inter-draw, %u unbarriered republications)\n",
                 stats.draws, stats.indexed_draws, stats.state_updates,
                 stats.elided_bind_tails, stats.incremental_fs_ppp_updates,
                 stats.incremental_cull_ppp_updates,
                 stats.links, stats.barriers, stats.inter_draw_barriers,
                 stats.unbarriered_state_republications);
      } else if (header->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         if (header->size < offsetof(struct drm_asahi_cmd_compute,
                                     cdm_ctrl_stream_end) +
                               sizeof(uint64_t))
            return -EINVAL;

         struct drm_asahi_cmd_compute compute = {0};
         memcpy(&compute, header + 1, MIN2(sizeof(compute), header->size));
         const struct asahi_g17p_stream_memory memory = {
            .data = (void *)(intptr_t)fd,
            .read = shim_gpu_read,
            .owned = shim_gpu_owned,
            .span = shim_gpu_span,
            .verbose = getenv("ASAHI_SHIM_TRACE_G17P_STREAM") != NULL,
            .strict_descriptors = g17p_strict_descriptors(),
         };
         struct asahi_g17p_compute_stats stats;
         if (!asahi_g17p_compute_validate(&memory, &compute, &stats)) {
            g17p_streams_rejected++;
            const int rejected = g17p_stream_reject("CDM");
            if (rejected)
               return rejected;
            cursor += command_size;
            continue;
         }
         compute_stats.launches += stats.launches;
         compute_stats.unbarriered_launches += stats.unbarriered_launches;
         compute_stats.two_dimensional_launches +=
            stats.two_dimensional_launches;
         compute_stats.barriers += stats.barriers;
         compute_stats.prologue_barriers += stats.prologue_barriers;
         compute_stats.strong_barriers += stats.strong_barriers;
         compute_stats.indirect_launches += stats.indirect_launches;
         compute_stats.indirect_strong_barriers +=
            stats.indirect_strong_barriers;
         compute_stats.sstt_flushes += stats.sstt_flushes;
         compute_stats.terminated |= stats.terminated;
         g17p_streams_validated++;
         fprintf(stderr,
                 "shim: G17P CDM validated: %u launches (%u 2D, %u back to "
                 "back), %u barriers, %u prologue (%u strong), %u SSTT "
                 "flushes\n",
                 stats.launches, stats.two_dimensional_launches,
                 stats.unbarriered_launches, stats.barriers,
                 stats.prologue_barriers,
                 stats.strong_barriers, stats.sstt_flushes);
      }

      cursor += command_size;
   }

   return cursor == submit->cmdbuf_size ? 0 : -EINVAL;
}

static int
asahi_ioctl_vm_bind(int fd, unsigned long request, void *arg)
{
   simple_mtx_lock(&shim_mapping_lock);
   int ret = asahi_ioctl_vm_bind_unlocked(fd, request, arg);
   simple_mtx_unlock(&shim_mapping_lock);
   return ret;
}

static int
asahi_ioctl_vm_create(int fd, unsigned long request, void *arg)
{
   simple_mtx_lock(&shim_mapping_lock);
   int ret = asahi_ioctl_vm_create_unlocked(fd, request, arg);
   simple_mtx_unlock(&shim_mapping_lock);
   return ret;
}

static int
asahi_ioctl_submit(int fd, unsigned long request, void *arg)
{
   simple_mtx_lock(&shim_mapping_lock);
   int ret = asahi_ioctl_submit_unlocked(fd, request, arg);
   simple_mtx_unlock(&shim_mapping_lock);
   return ret;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_param,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_vm_create,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_vm_bind,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_queue_create,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_queue_destroy,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_gem_bind_object,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_submit,
};

void
drm_shim_driver_init(void)
{
   shim_device.bus_type = DRM_BUS_PLATFORM;
   shim_device.driver_name = "asahi";
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   /* libdrm tells the bus from this link; without it drmGetDevices2() skips
    * the node, and Vulkan and the Gallium loader find no device. */
   drm_shim_override_link("../../../../../bus/platform",
                          "/sys/dev/char/%d:%d/device/subsystem", DRM_MAJOR,
                          render_node_minor);

   const char *chip = getenv("ASAHI_SHIM_CHIP");
   const bool g13x = chip && !strcmp(chip, "g13x");
   const bool g14g = chip && !strcmp(chip, "g14g");
   const bool g16s = chip &&
      (!strcmp(chip, "g16s") || !strcmp(chip, "g16s-rt") ||
       !strcmp(chip, "g16s-as"));
   const bool g17p = chip && !strcmp(chip, "g17p");
   const bool g15s = chip && !strcmp(chip, "g15s");
   const bool g15c = chip && !strcmp(chip, "g15c");
   const bool g16g = chip && !strcmp(chip, "g16g");
   /* ASAHI_SHIM_CHIP=g17g-compile-only and g18g. */
   const bool g17g = g17g_compile_only();
   const bool g18g = g18g_profile();

   drm_shim_override_file(g13x   ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,agx-g13s\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g14g ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,agx-g14g\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g16s ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,gpu-g16s\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g17g ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,gpu-g17g\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g18g ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,gpu-g18g\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g17p ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,gpu-g17p\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g15s ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,agx-g15s\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g15c ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,agx-g15c\n"
                                   "OF_COMPATIBLE_N=1\n"
                          : g16g ? "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,gpu-g16g\n"
                                   "OF_COMPATIBLE_N=1\n"
                                 : "DRIVER=asahi\n"
                                   "OF_FULLNAME=/soc/agx\n"
                                   "OF_COMPATIBLE_0=apple,gpu-g13g\n"
                                   "OF_COMPATIBLE_N=1\n",
                          "/sys/dev/char/%d:%d/device/uevent", DRM_MAJOR,
                          render_node_minor);
}
