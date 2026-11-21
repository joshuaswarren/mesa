/* SPDX-License-Identifier: MIT */
/*
 * Copyright (C) The Asahi Linux Contributors
 * Copyright (C) 2018-2023 Collabora Ltd.
 * Copyright (C) 2014-2018 Broadcom
 */
#ifndef _ASAHI_DRM_H_
#define _ASAHI_DRM_H_

#include "drm.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * DOC: Introduction to the Asahi UAPI
 *
 * This documentation describes the Asahi IOCTLs.
 *
 * Just a few generic rules about the data passed to the Asahi IOCTLs (cribbed
 * from Panthor):
 *
 * - Structures must be aligned on 64-bit/8-byte. If the object is not
 *   naturally aligned, a padding field must be added.
 * - Fields must be explicitly aligned to their natural type alignment with
 *   pad[0..N] fields.
 * - All padding fields will be checked by the driver to make sure they are
 *   zeroed.
 * - Flags can be added, but not removed/replaced.
 * - New fields can be added to the main structures (the structures
 *   directly passed to the ioctl). Those fields can be added at the end of
 *   the structure, or replace existing padding fields. Any new field being
 *   added must preserve the behavior that existed before those fields were
 *   added when a value of zero is passed.
 * - New fields can be added to indirect objects (objects pointed by the
 *   main structure), iff those objects are passed a size to reflect the
 *   size known by the userspace driver (see
 *   drm_asahi_cmd_header::size).
 * - If the kernel driver is too old to know some fields, those will be
 *   ignored if zero, and otherwise rejected (and so will be zero on output).
 * - If userspace is too old to know some fields, those will be zeroed
 *   (input) before the structure is parsed by the kernel driver.
 * - Each new flag/field addition must come with a driver version update so
 *   the userspace driver doesn't have to guess which flags are supported.
 * - Structures should not contain unions, as this would defeat the
 *   extensibility of such structures.
 * - IOCTLs can't be removed or replaced. New IOCTL IDs should be placed
 *   at the end of the drm_asahi_ioctl_id enum.
 */

/**
 * enum drm_asahi_ioctl_id - IOCTL IDs
 *
 * Place new ioctls at the end, don't re-order, don't replace or remove entries.
 *
 * These IDs are not meant to be used directly. Use the DRM_IOCTL_ASAHI_xxx
 * definitions instead.
 */
enum drm_asahi_ioctl_id {
	/** @DRM_ASAHI_GET_PARAMS: Query device properties. */
	DRM_ASAHI_GET_PARAMS = 0,

	/** @DRM_ASAHI_GET_TIME: Query device time. */
	DRM_ASAHI_GET_TIME,

	/** @DRM_ASAHI_VM_CREATE: Create a GPU VM address space. */
	DRM_ASAHI_VM_CREATE,

	/** @DRM_ASAHI_VM_DESTROY: Destroy a VM. */
	DRM_ASAHI_VM_DESTROY,

	/** @DRM_ASAHI_VM_BIND: Bind/unbind memory to a VM. */
	DRM_ASAHI_VM_BIND,

	/** @DRM_ASAHI_GEM_CREATE: Create a buffer object. */
	DRM_ASAHI_GEM_CREATE,

	/**
	 * @DRM_ASAHI_GEM_MMAP_OFFSET: Get offset to pass to mmap() to map a
	 * given GEM handle.
	 */
	DRM_ASAHI_GEM_MMAP_OFFSET,

	/** @DRM_ASAHI_GEM_BIND_OBJECT: Bind memory as a special object */
	DRM_ASAHI_GEM_BIND_OBJECT,

	/** @DRM_ASAHI_QUEUE_CREATE: Create a scheduling queue. */
	DRM_ASAHI_QUEUE_CREATE,

	/** @DRM_ASAHI_QUEUE_DESTROY: Destroy a scheduling queue. */
	DRM_ASAHI_QUEUE_DESTROY,

	/** @DRM_ASAHI_SUBMIT: Submit commands to a queue. */
	DRM_ASAHI_SUBMIT,

	/** @DRM_ASAHI_GEM_MADVISE: Mark a VM-private object purgeable or needed. */
	DRM_ASAHI_GEM_MADVISE,

	/**
	 * @DRM_ASAHI_GEM_USERPTR: Create a buffer object over the caller's own
	 * memory (DRM_ASAHI_FEATURE_USERPTR).
	 */
	DRM_ASAHI_GEM_USERPTR,

	/**
	 * @DRM_ASAHI_GEM_RESIDENCY: Ask the kernel to release an object's
	 * memory after that of the other objects of its VM when it reclaims
	 * the VM's idle memory, or withdraw that request
	 * (DRM_ASAHI_FEATURE_GEM_RESIDENCY). A hint, see struct
	 * drm_asahi_gem_residency.
	 */
	DRM_ASAHI_GEM_RESIDENCY,
};

#define DRM_ASAHI_MAX_CLUSTERS	64

/**
 * struct drm_asahi_params_global - Global parameters.
 *
 * This struct may be queried by drm_asahi_get_params.
 */
struct drm_asahi_params_global {
	/** @features: Feature bits from drm_asahi_feature */
	__u64 features;

	/** @gpu_generation: GPU generation, e.g. 13 for G13G */
	__u32 gpu_generation;

	/** @gpu_variant: GPU variant as a character, e.g. 'C' for G13C */
	__u32 gpu_variant;

	/**
	 * @gpu_revision: GPU revision in BCD, e.g. 0x00 for 'A0' or
	 * 0x21 for 'C1'
	 */
	__u32 gpu_revision;

	/** @chip_id: Chip ID in BCD, e.g. 0x8103 for T8103 */
	__u32 chip_id;

	/** @num_dies: Number of dies in the SoC */
	__u32 num_dies;

	/** @num_clusters_total: Number of GPU clusters (across all dies) */
	__u32 num_clusters_total;

	/**
	 * @num_cores_per_cluster: Number of logical cores per cluster
	 * (including inactive/nonexistent)
	 */
	__u32 num_cores_per_cluster;

	/** @max_frequency_khz: Maximum GPU core clock frequency */
	__u32 max_frequency_khz;

	/** @core_masks: Bitmask of present/enabled cores per cluster */
	__u64 core_masks[DRM_ASAHI_MAX_CLUSTERS];

	/**
	 * @vm_start: VM range start VMA. Together with @vm_end, this defines
	 * the window of valid GPU VAs. Userspace is expected to subdivide VAs
	 * out of this window.
	 *
	 * This window contains all virtual addresses that userspace needs to
	 * know about. There may be kernel-internal GPU VAs outside this range,
	 * but that detail is not relevant here.
	 */
	__u64 vm_start;

	/** @vm_end: VM range end VMA */
	__u64 vm_end;

	/**
	 * @vm_kernel_min_size: Minimum kernel VMA window size.
	 *
	 * When creating a VM, userspace is required to carve out a section of
	 * virtual addresses (within the range given by @vm_start and
	 * @vm_end). The kernel will allocate various internal structures
	 * within the specified VA range.
	 *
	 * Allowing userspace to choose the VA range for the kernel, rather than
	 * the kernel reserving VAs and requiring userspace to cope, can assist
	 * in implementing SVM.
	 */
	__u64 vm_kernel_min_size;

	/**
	 * @max_commands_per_submission: Maximum number of supported commands
	 * per submission. This mirrors firmware limits. Userspace must split up
	 * larger command buffers, which may require inserting additional
	 * synchronization.
	 */
	__u32 max_commands_per_submission;

	/**
	 * @max_attachments: Maximum number of drm_asahi_attachment's per
	 * command
	 */
	__u32 max_attachments;

	/**
	 * @command_timestamp_frequency_hz: Timebase frequency for timestamps
	 * written during command execution, specified via drm_asahi_timestamp
	 * structures. As this rate is controlled by the firmware, it is a
	 * queryable parameter.
	 *
	 * Userspace must divide by this frequency to convert timestamps to
	 * seconds, rather than hardcoding a particular firmware's rate.
	 */
	__u64 command_timestamp_frequency_hz;

	/**
	 * @usc_generation: Unified shader core (USC) ISA generation.
	 *
	 * This is 2 for G13/G14 and 3 for G15/G16/G17. It does not
	 * distinguish the binary encoding groups within AGX3.
	 */
	__u32 usc_generation;

	/**
	 * @gpu_hal_generation: Apple GPU HAL/command-stream generation.
	 *
	 * This is DRM_ASAHI_GPU_HAL_LEGACY for G13-G15,
	 * DRM_ASAHI_GPU_HAL_200 for G16 and G17P/G17A, and
	 * DRM_ASAHI_GPU_HAL_300 for G17G/G17S/G17C.
	 */
	__u32 gpu_hal_generation;

	/**
	 * @memory_limit: Device-wide limit on user committed plus kernel
	 * resident bytes. Zero means no driver limit. This is a capacity,
	 * not current free memory; other clients share it.
	 */
	__u64 memory_limit;

	/**
	 * @client_memory_limit: Per-DRM-file limit on user committed bytes.
	 * Zero means no driver limit. Userspace must zero-initialize the reply
	 * buffer: older kernels leave these optional tail fields untouched.
	 */
	__u64 client_memory_limit;
};

/**
 * enum drm_asahi_gpu_hal_generation - GPU HAL/command-stream generation
 */
enum drm_asahi_gpu_hal_generation {
	/** @DRM_ASAHI_GPU_HAL_LEGACY: GPU predates Apple's numbered HALs. */
	DRM_ASAHI_GPU_HAL_LEGACY = 0,

	/** @DRM_ASAHI_GPU_HAL_200: Apple HAL200 command stream. */
	DRM_ASAHI_GPU_HAL_200 = 200,

	/** @DRM_ASAHI_GPU_HAL_300: Apple HAL300 command stream. */
	DRM_ASAHI_GPU_HAL_300 = 300,
};

/**
 * enum drm_asahi_feature - Feature bits
 *
 * This covers only features that userspace cannot infer from the architecture
 * version. Most features don't need to be here.
 */
enum drm_asahi_feature {
	/**
	 * @DRM_ASAHI_FEATURE_SOFT_FAULTS: GPU has "soft fault" enabled. Shader
	 * loads of unmapped memory will return zero. Shader stores to unmapped
	 * memory will be silently discarded. Note that only shader load/store
	 * is affected. Other hardware units are not affected, notably including
	 * texture sampling.
	 *
	 * Soft fault is set when initializing the GPU and cannot be runtime
	 * toggled. Therefore, it is exposed as a feature bit and not a
	 * userspace-settable flag on the VM. When soft fault is enabled,
	 * userspace can speculate memory accesses more aggressively.
	 */
	DRM_ASAHI_FEATURE_SOFT_FAULTS = (1UL) << 0,
	/** @DRM_ASAHI_FEATURE_VM_STATUS: GET_PARAMS supports sticky VM status. */
	DRM_ASAHI_FEATURE_VM_STATUS = (1UL) << 1,
	/** @DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES: Reserved independent queues and limits. */
	DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES = (1UL) << 2,

	/**
	 * @DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS: Every hardware command in
	 * a submit waits on all of that ioctl's input syncs, including errors.
	 * Independent commands need no extra dependency on the first command.
	 * Userspace splitting a batch across ioctls must supply its input syncs
	 * to every chunk relying on this guarantee.
	 */
	/*
	 * Bits 3 to 9 belong to the M3 (G15S) and J613 runtimes:
	 * SCHEDULED_QUEUES, RENDER_TIMESTAMP_COPIES, COMPUTE_TIMESTAMP_COPIES,
	 * FRAGMENT_DEPENDENCY, FRAGMENT_DEPENDENCY_COMPUTE,
	 * COMPUTE_WIDE_VISIBILITY and CACHED_PROGRAM_PROBE.
	 */
	DRM_ASAHI_FEATURE_SCHEDULED_QUEUES = (1UL) << 3,
	DRM_ASAHI_FEATURE_RENDER_TIMESTAMP_COPIES = (1UL) << 4,
	DRM_ASAHI_FEATURE_COMPUTE_TIMESTAMP_COPIES = (1UL) << 5,
	DRM_ASAHI_FEATURE_FRAGMENT_DEPENDENCY = (1UL) << 6,
	DRM_ASAHI_FEATURE_FRAGMENT_DEPENDENCY_COMPUTE = (1UL) << 7,
	DRM_ASAHI_FEATURE_COMPUTE_WIDE_VISIBILITY = (1UL) << 8,
	DRM_ASAHI_FEATURE_CACHED_PROGRAM_PROBE = (1UL) << 9,
	DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS = (1UL) << 10,

	/**
	 * @DRM_ASAHI_FEATURE_VM_STATUS_MIRROR: GET_PARAMS group
	 * DRM_ASAHI_PARAM_GROUP_VM_STATUS_MIRROR registers a userspace word that
	 * always holds the VM's sticky status (the value group VM_STATUS would
	 * return). The kernel stores a failure there before signalling any fence
	 * of the failed job, so after a successful syncobj wait a load of the word
	 * replaces the VM_STATUS query. Registration may still be refused.
	 */
	DRM_ASAHI_FEATURE_VM_STATUS_MIRROR = (1UL) << 11,
	/** Exact queue-local prior hardware command prefixes in extended payloads. */
	DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS = (1UL) << 12,

	/**
	 * @DRM_ASAHI_FEATURE_FEW_PRIMITIVES: Render commands accept
	 * DRM_ASAHI_RENDER_FEW_PRIMITIVES.
	 */
	DRM_ASAHI_FEATURE_FEW_PRIMITIVES = (1UL) << 13,

	/**
	 * @DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS: Render commands accept
	 * DRM_ASAHI_RENDER_FRAGMENT_BARRIERS.
	 */
	DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS = (1UL) << 14,
	DRM_ASAHI_FEATURE_RENDER_RAY_QUERY = (1UL) << 15,

	/**
	 * @DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD: Compute commands accept
	 * DRM_ASAHI_COMPUTE_AS_BUILD for a bounded long execution allowance.
	 */
	DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD = (1UL) << 16,

	/*
	 * Bits 17 to 28 are reported only by the M5 (G17G/G17S) runtime. They
	 * follow the numbering above, so one userspace reads every kernel's
	 * features the same way. Bits 25 and 27 are reserved for that runtime.
	 */

	/** @DRM_ASAHI_FEATURE_RENDER_RESOURCES: Per-program render storage declarations. */
	DRM_ASAHI_FEATURE_RENDER_RESOURCES = (1UL) << 17,
	/** @DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2: Versioned SCS and mesh storage. */
	DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2 = (1UL) << 18,
	/** @DRM_ASAHI_FEATURE_SPARSE_TILES: DRM_ASAHI_BIND_SPARSE_TILE accepted. */
	DRM_ASAHI_FEATURE_SPARSE_TILES = (1UL) << 19,
	/** @DRM_ASAHI_FEATURE_COMPUTE_MATRIX: Native compute matrix instructions. */
	DRM_ASAHI_FEATURE_COMPUTE_MATRIX = (1UL) << 20,
	/*
	 * Render programs accept DRM_ASAHI_RENDER_PROGRAM_TASK; task kicks get the
	 * TA mesh rendering buffer that backs the object payload address.
	 */
	DRM_ASAHI_FEATURE_TASK_MESH_STORAGE = (1UL) << 21,
	/** @DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES: Per-render external shader writes. */
	DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES = (1UL) << 22,
	/** @DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION: D32 2x/4x compressed depth and copy admission. */
	DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION = (1UL) << 23,
	/**
	 * @DRM_ASAHI_FEATURE_COMPUTE_STAGE_BARRIERS: A render may carry a
	 * separate compute prefix for its tiling half
	 * (DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS,
	 * struct drm_asahi_cmd_render_with_stage_prior).
	 */
	DRM_ASAHI_FEATURE_COMPUTE_STAGE_BARRIERS = (1UL) << 24,

	/**
	 * @DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES: Render commands accept
	 * DRM_ASAHI_RENDER_STAGE_PREFIXES (struct
	 * drm_asahi_cmd_render_with_stage_prefixes).
	 */
	DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES = (1UL) << 25,

	/**
	 * @DRM_ASAHI_FEATURE_WHOLE_ATTACHMENT_RENDERS: Render commands accept
	 * DRM_ASAHI_RENDER_WHOLE_ATTACHMENTS.
	 */
	DRM_ASAHI_FEATURE_WHOLE_ATTACHMENT_RENDERS = (1UL) << 27,
	/** @DRM_ASAHI_FEATURE_M5_STENCIL_MSAA_COMPRESSION: single-layer 2x/4x compressed stencil in the native ZLS form. */
	DRM_ASAHI_FEATURE_M5_STENCIL_MSAA_COMPRESSION = (1UL) << 26,
	/** @DRM_ASAHI_FEATURE_COMPUTE_RESOURCES: Per-dispatch storage declarations. */
	DRM_ASAHI_FEATURE_COMPUTE_RESOURCES = (1UL) << 28,
	/**
	 * @DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS: SUBMIT accepts one
	 * DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE output. All outputs
	 * include the earlier points of that timeline without adding GPU waits.
	 */
	DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS = (1UL) << 29,
	/**
	 * @DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS: Compute commands accept
	 * DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS.
	 */
	DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS = (1UL) << 30,

	/**
	 * @DRM_ASAHI_FEATURE_G18_GRAPHICS_SCRATCH: G18 render commands accept
	 * DRM_ASAHI_RENDER_G18_RESOURCES. The kernel sizes the shared TA/3D
	 * private-memory pool and both launch pairs from that suffix.
	 */
	DRM_ASAHI_FEATURE_G18_GRAPHICS_SCRATCH = (1ULL) << 31,
};

/*
 * DRM_ASAHI_FEATURE_G18_NATIVE_TILE_SAMPLES: G18: single-sample,
 * single-layer renders with 32x32 utiles also accept 40-, 56- and 72-byte
 * samples (the hardware's native layouts), besides tiles of at most 32 KiB
 * and 48..128-byte samples in steps of 16. A feature bit like those of enum
 * drm_asahi_feature; a 64-bit enumerator would widen that enum's type.
 */
#define DRM_ASAHI_FEATURE_G18_NATIVE_TILE_SAMPLES ((1ULL) << 32)

/*
 * DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2: render commands accept
 * DRM_ASAHI_RENDER_STAGE_PREFIXES as the complete struct
 * drm_asahi_cmd_render_staged: the 448-byte render command (priors and
 * ray-query stages included), then struct drm_asahi_render_stage_prefixes.
 * The kernel's drm_asahi_cmd_render_m5 has the same layout. A feature bit
 * like those of enum drm_asahi_feature.
 */
#define DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2 ((1ULL) << 33)

/*
 * DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION: M5: compute commands accept
 * DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION. A feature bit like those of enum
 * drm_asahi_feature, defined outside it so that the enum keeps its 32-bit type.
 */
#define DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION ((1ULL) << 34)

/*
 * DRM_ASAHI_FEATURE_GEM_RESIDENCY: DRM_IOCTL_ASAHI_GEM_RESIDENCY is
 * available. A feature bit like those of enum drm_asahi_feature, defined
 * here like bit 32.
 */
#define DRM_ASAHI_FEATURE_GEM_RESIDENCY ((1ULL) << 35)

/*
 * DRM_ASAHI_FEATURE_COMPUTE_TIMEOUT: drm_asahi_queue_create::compute_timeout_ms
 * sets the watchdog allowance of the queue's compute commands. A feature bit
 * like those of enum drm_asahi_feature, defined here like bit 32.
 */
#define DRM_ASAHI_FEATURE_COMPUTE_TIMEOUT ((1ULL) << 36)

/*
 * DRM_ASAHI_FEATURE_USERPTR: DRM_IOCTL_ASAHI_GEM_USERPTR is available. A
 * feature bit like those of enum drm_asahi_feature, defined here like bit 32.
 */
#define DRM_ASAHI_FEATURE_USERPTR ((1ULL) << 37)

/*
 * DRM_ASAHI_FEATURE_RENDER_MESH_OUTPUT: render commands accept
 * DRM_ASAHI_RENDER_MESH_OUTPUT. A feature bit like those of enum
 * drm_asahi_feature, defined here like bit 32.
 */
#define DRM_ASAHI_FEATURE_RENDER_MESH_OUTPUT ((1ULL) << 38)

/*
 * DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE: the render-start VDM barrier at the
 * root of a render's VDM stream may set the USC cache invalidate (bit 3).
 * Reported on every GPU: the kernel's stream checks admit the barrier word
 * with or without it. A feature bit like those of enum drm_asahi_feature,
 * defined here like bit 32.
 */
#define DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE ((1ULL) << 40)

#define DRM_ASAHI_PARAM_GROUP_VM_STATUS 1
#define DRM_ASAHI_PARAM_GROUP_QUEUE_LIMITS 2
#define DRM_ASAHI_PARAM_GROUP_VM_STATUS_MIRROR 3

/* GET_PARAMS group VM_STATUS_MIRROR (input only; feature bit 4).
 * handle names a GEM object (at most 64 KiB) of the calling file; offset is a 4-byte-aligned
 * byte offset of a __s32 inside it; flags is zero. The kernel keeps its own
 * reference to the object for the VM status lifetime, zeroes the word, then
 * keeps it equal to the sticky VM error (0 or a negative errno). One
 * registration per VM. EOPNOTSUPP means "keep using group VM_STATUS".
 */
struct drm_asahi_vm_status_mirror {
   __u32 vm_id;
   __u32 handle;
   __u32 offset;
   __u32 flags;
};

/* GET_PARAMS group VM_STATUS is an in/out query on the calling file's VM.
 * Set pointer to this structure and size to sizeof(structure).
 * vm_id selects the VM; flags and pad must be zero. error is output only.
 * error == 0 means no observed submission error; a negative Linux errno is
 * sticky for this VM's lifetime. The first error is retained. Creating a new
 * queue or successfully waiting another fence does not clear it.
 * An execution-domain failure also marks other queried VMs lost when their
 * hardware owner cannot progress. A query never recovers/resets firmware.
 * The driver publishes an accepted job's failure before signaling its fence.
 * Thus userspace must query after a successful DRM syncobj wait/poll before
 * treating that fence as successful execution. Syncobj signal alone is not
 * an execution-success indication.
 * The feature bit is the availability contract; older kernels leave it zero.
 * Kernel errors for invalid requests: EINVAL flags/pad/size, ENOENT foreign or
 * missing vm_id, EFAULT inaccessible pointer. No output on invalid request.
 */
struct drm_asahi_vm_status {
   __u32 vm_id;
   __u32 flags;
   __s32 error;
   __u32 pad;
};

/**
 * struct drm_asahi_queue_limits - GET_PARAMS group2 output, feature bit2.
 * @max_queues: Maximum independent DRM queues on the physical device. This is
 * a hardware resource bound, shared by all files/VMs; it is not a promise that
 * all slots are currently free. QUEUE_CREATE reserves a queue's hardware state
 * and execution context, returning an allocation error if resources run out.
 * @max_in_flight_per_queue: Hardware scheduler credits per queue. Software
 * may enqueue additional jobs, which wait behind preceding work/dependencies.
 * @flags: Reserved, zero.
 * @pad: Reserved, zero.
 *
 * Independent compute execution is supported. Graphics work retains its own
 * engine scheduling constraints. Query does not allocate or reset anything.
 */
struct drm_asahi_queue_limits {
    __u32 max_queues;
    __u32 max_in_flight_per_queue;
    __u32 flags;
    __u32 pad;
};

/**
 * struct drm_asahi_get_params - Arguments passed to DRM_IOCTL_ASAHI_GET_PARAMS
 */
struct drm_asahi_get_params {
	/** @param_group: 0 global params, 1 VM status, 2 queue limits. */
	__u32 param_group;

	/** @pad: MBZ */
	__u32 pad;

	/** @pointer: User pointer to write parameter struct */
	__u64 pointer;

	/**
	 * @size: Size of the user buffer. In case of older userspace, this may
	 * be less than sizeof(struct drm_asahi_params_global). The kernel will
	 * not write past the length specified here, allowing extensibility.
	 */
	__u64 size;
};

/**
 * struct drm_asahi_vm_create - Arguments passed to DRM_IOCTL_ASAHI_VM_CREATE
 */
struct drm_asahi_vm_create {
	/**
	 * @kernel_start: Start of the kernel-reserved address range. See
	 * drm_asahi_params_global::vm_kernel_min_size.
	 *
	 * Both @kernel_start and @kernel_end must be within the range of
	 * valid VAs given by drm_asahi_params_global::vm_start and
	 * drm_asahi_params_global::vm_end. The size of the kernel range
	 * (@kernel_end - @kernel_start) must be at least
	 * drm_asahi_params_global::vm_kernel_min_size.
	 *
	 * Userspace must not bind any memory on this VM into this reserved
	 * range, it is for kernel use only.
	 */
	__u64 kernel_start;

	/**
	 * @kernel_end: End of the kernel-reserved address range. See
	 * @kernel_start.
	 */
	__u64 kernel_end;

	/** @vm_id: Returned VM ID */
	__u32 vm_id;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * struct drm_asahi_vm_destroy - Arguments passed to DRM_IOCTL_ASAHI_VM_DESTROY
 */
struct drm_asahi_vm_destroy {
	/** @vm_id: VM ID to be destroyed */
	__u32 vm_id;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * enum drm_asahi_gem_flags - Flags for GEM creation
 */
enum drm_asahi_gem_flags {
	/**
	 * @DRM_ASAHI_GEM_WRITEBACK: BO should be CPU-mapped as writeback.
	 *
	 * Map as writeback instead of write-combine. This optimizes for CPU
	 * reads.
	 */
	DRM_ASAHI_GEM_WRITEBACK = (1L << 0),

	/**
	 * @DRM_ASAHI_GEM_VM_PRIVATE: BO is private to this GPU VM (no exports).
	 */
	DRM_ASAHI_GEM_VM_PRIVATE = (1L << 1),

	/**
	 * @DRM_ASAHI_GEM_USERPTR_READ_ONLY: The GPU only reads the memory of a
	 * DRM_IOCTL_ASAHI_GEM_USERPTR object. Only that ioctl accepts it.
	 */
	DRM_ASAHI_GEM_USERPTR_READ_ONLY = (1L << 2),
};

/**
 * struct drm_asahi_gem_create - Arguments passed to DRM_IOCTL_ASAHI_GEM_CREATE
 */
struct drm_asahi_gem_create {
	/** @size: Size of the BO */
	__u64 size;

	/** @flags: Combination of drm_asahi_gem_flags flags. */
	__u32 flags;

	/**
	 * @vm_id: VM ID to assign to the BO, if DRM_ASAHI_GEM_VM_PRIVATE is set
	 */
	__u32 vm_id;

	/** @handle: Returned GEM handle for the BO */
	__u32 handle;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * struct drm_asahi_gem_userptr - Arguments passed to
 * DRM_IOCTL_ASAHI_GEM_USERPTR
 *
 * Creates a buffer object whose backing is the caller's memory at
 * [@userptr, @userptr + @size), without a copy: the CPU accesses it through
 * the caller's own mapping, and the GPU through VM_BIND mappings of the
 * object, as for any other object. Available when DRM_ASAHI_FEATURE_USERPTR
 * is set; otherwise the ioctl fails with EOPNOTSUPP or EINVAL.
 *
 * The pages of the range are pinned when the object is created and stay
 * pinned until it is freed: all of them or, on failure, none. They are the
 * pages the range maps at creation; later changes to the caller's mappings
 * (munmap, mremap, a new mapping over the range, truncating a mapped file)
 * leave the object on those pages, which stay allocated until it is freed.
 * After fork(), the child gets copies of the pinned anonymous pages; the
 * object stays the parent's memory. The CPU and GPU views are coherent
 * without cache maintenance.
 *
 * The pinned pages count against the caller's RLIMIT_MEMLOCK, together with
 * its other long-term pins, unless it has CAP_IPC_LOCK (ENOMEM otherwise),
 * and their size against the client and device limits of
 * drm_asahi_params_global::client_memory_limit and
 * drm_asahi_params_global::memory_limit (ENOMEM). They are never released by
 * memory-pressure reclaim.
 *
 * Without DRM_ASAHI_GEM_USERPTR_READ_ONLY, the whole range must be mapped
 * writable, and not be a shared mapping of a file whose filesystem tracks
 * writes. With it, any readable mapping is accepted, and the object takes
 * only GPU read-only VM_BIND mappings (a mapping with DRM_ASAHI_BIND_WRITE
 * fails with EACCES). Pages of private mappings, file mappings included, are
 * pinned as the caller's own copies, as a write would make them: a private
 * file mapping is copied out of the page cache. A read-only shared file
 * mapping pins the page-cache pages themselves, which then stay resident
 * while the object lives. Device and PFN mappings, including CPU mappings of
 * other buffer objects, are refused (EFAULT).
 *
 * The object cannot be mapped with DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, exported,
 * given advice with DRM_IOCTL_ASAHI_GEM_MADVISE, bound with
 * DRM_IOCTL_ASAHI_GEM_BIND_OBJECT or registered as a VM status mirror (all
 * EINVAL).
 */
struct drm_asahi_gem_userptr {
	/**
	 * @userptr: Start of the range, aligned to both the CPU page size and
	 * the GPU page size (16 KiB).
	 */
	__u64 userptr;

	/** @size: Size of the range, a non-zero multiple of both page sizes. */
	__u64 size;

	/**
	 * @flags: DRM_ASAHI_GEM_VM_PRIVATE and DRM_ASAHI_GEM_USERPTR_READ_ONLY.
	 */
	__u32 flags;

	/**
	 * @vm_id: VM ID to assign to the BO, if DRM_ASAHI_GEM_VM_PRIVATE is set
	 */
	__u32 vm_id;

	/** @handle: Returned GEM handle for the BO */
	__u32 handle;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * struct drm_asahi_gem_mmap_offset - Arguments passed to
 * DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET
 */
struct drm_asahi_gem_mmap_offset {
	/** @handle: Handle for the object being mapped. */
	__u32 handle;

	/** @flags: Must be zero */
	__u32 flags;

	/** @offset: The fake offset to use for subsequent mmap call */
	__u64 offset;
};

/**
 * enum drm_asahi_gem_madv - Advice for DRM_IOCTL_ASAHI_GEM_MADVISE
 */
enum drm_asahi_gem_madv {
	/** @DRM_ASAHI_GEM_MADV_WILLNEED: The object's contents are needed. */
	DRM_ASAHI_GEM_MADV_WILLNEED = 0,
	/**
	 * @DRM_ASAHI_GEM_MADV_DONTNEED: The object is not referenced by any
	 * pending or future GPU work until the next WILLNEED. The kernel may
	 * discard its contents under memory pressure.
	 */
	DRM_ASAHI_GEM_MADV_DONTNEED = 1,
};

/**
 * struct drm_asahi_gem_madvise - Arguments passed to
 * DRM_IOCTL_ASAHI_GEM_MADVISE
 *
 * Only VM-private objects accept advice. A purged object keeps its handle
 * and mappings but has no contents; userspace must not use it again.
 */
struct drm_asahi_gem_madvise {
	/** @handle: Handle of the object. */
	__u32 handle;

	/** @madv: One of enum drm_asahi_gem_madv. */
	__u32 madv;

	/** @retained: Set by the kernel: 1 if the contents are retained, 0 if purged. */
	__u32 retained;

	/** @pad: Must be zero. */
	__u32 pad;
};

/**
 * enum drm_asahi_gem_residency_flags - Flags for DRM_IOCTL_ASAHI_GEM_RESIDENCY
 */
enum drm_asahi_gem_residency_flags {
	/**
	 * @DRM_ASAHI_GEM_RESIDENCY_KEEP: When reclaim releases idle memory of
	 * the object's VM, release this object's memory after that of the
	 * VM's objects without the flag, and only if theirs was not enough.
	 * Without the flag, the object returns to the default policy.
	 */
	DRM_ASAHI_GEM_RESIDENCY_KEEP = (1L << 0),
};

/**
 * struct drm_asahi_gem_residency - Arguments passed to
 * DRM_IOCTL_ASAHI_GEM_RESIDENCY
 *
 * When the system reclaims memory, the kernel may release the memory of the
 * VM-private objects of an idle VM. It brings the memory back before the
 * VM's next job runs, which then waits for it.
 * DRM_ASAHI_GEM_RESIDENCY_KEEP is a priority among the objects that may be
 * released this way. It changes the order in which they go, not whether
 * reclaim gets to them:
 *
 * - Only reclaim on behalf of the whole system releases GPU memory. Reclaim
 *   inside a memory cgroup below the root, such as that of a cgroup at its
 *   memory limit, releases none, kept or not.
 * - Reclaim releases memory of a VM only while the VM is idle: no job,
 *   binding change or other use of the VM is in progress, the last one (or
 *   the VM's creation) is at least two seconds back, and after a job had to
 *   wait for released memory, a pause that grows with the time that took
 *   (between 1 and 30 seconds) is over. The memory of a VM that is not idle
 *   stays, kept or not.
 * - Of an idle VM's objects, a reclaim pass releases those without the flag
 *   first, and kept objects only if that left the amount the pass set out
 *   to free unmet.
 * - Contents userspace gave up with DRM_ASAHI_GEM_MADV_DONTNEED are purged
 *   as before, kept or not.
 *
 * The request is therefore a hint. Userspace must not rely on a kept object
 * being resident, and a client that stays active, or whose memory cgroup is
 * the one that runs out of memory, keeps its GPU memory whether it asked or
 * not.
 *
 * The request applies once this ioctl returns; memory already released stays
 * so until the VM's next job or binding brings it back. Other objects
 * (shared, imported or GEM_USERPTR ones) are never released and accept the
 * flag without effect on them.
 *
 * Kept objects still count against the memory limits as before; the device
 * total of their sizes is reported as user_kept in the GPU's gpu_memory
 * sysfs file. Available when DRM_ASAHI_FEATURE_GEM_RESIDENCY is set;
 * otherwise the ioctl fails with EOPNOTSUPP or EINVAL.
 */
struct drm_asahi_gem_residency {
	/** @handle: Handle of the object. */
	__u32 handle;

	/** @flags: Combination of drm_asahi_gem_residency_flags flags. */
	__u32 flags;
};

/**
 * enum drm_asahi_bind_flags - Flags for GEM binding
 */
enum drm_asahi_bind_flags {
	/**
	 * @DRM_ASAHI_BIND_UNBIND: Instead of binding a GEM object to the range,
	 * simply unbind the GPU VMA range.
	 */
	DRM_ASAHI_BIND_UNBIND = (1L << 0),

	/** @DRM_ASAHI_BIND_READ: Map BO with GPU read permission */
	DRM_ASAHI_BIND_READ = (1L << 1),

	/** @DRM_ASAHI_BIND_WRITE: Map BO with GPU write permission */
	DRM_ASAHI_BIND_WRITE = (1L << 2),

	/**
	 * @DRM_ASAHI_BIND_SINGLE_PAGE: Map a single page of the BO repeatedly
	 * across the VA range.
	 *
	 * This is useful to fill a VA range with scratch pages or zero pages.
	 * It is intended as a mechanism to accelerate sparse.
	 */
	DRM_ASAHI_BIND_SINGLE_PAGE = (1L << 3),

	/**
	 * @DRM_ASAHI_BIND_SPARSE_TILE: With DRM_ASAHI_BIND_SINGLE_PAGE, fill
	 * the range from kernel-owned, physically contiguous tiles instead of
	 * repeating the BO's first page: read-only binds read zeroes, writable
	 * binds target a sink; both tile pools are private to the VM. The BO
	 * only anchors the mapping. Requires DRM_ASAHI_FEATURE_SPARSE_TILES.
	 */
	DRM_ASAHI_BIND_SPARSE_TILE = (1L << 4),
};

/**
 * struct drm_asahi_gem_bind_op - Description of a single GEM bind operation.
 */
struct drm_asahi_gem_bind_op {
	/** @flags: Combination of drm_asahi_bind_flags flags. */
	__u32 flags;

	/** @handle: GEM object to bind (except for UNBIND) */
	__u32 handle;

	/**
	 * @offset: Offset into the object (except for UNBIND).
	 *
	 * For a regular bind, this is the beginning of the region of the GEM
	 * object to bind.
	 *
	 * For a single-page bind, this is the offset to the single page that
	 * will be repeatedly bound.
	 *
	 * Must be page-size aligned.
	 */
	__u64 offset;

	/**
	 * @range: Number of bytes to bind/unbind to @addr.
	 *
	 * Must be page-size aligned.
	 */
	__u64 range;

	/**
	 * @addr: Address to bind to.
	 *
	 * Must be page-size aligned.
	 */
	__u64 addr;
};

/**
 * struct drm_asahi_vm_bind - Arguments passed to
 * DRM_IOCTL_ASAHI_VM_BIND
 */
struct drm_asahi_vm_bind {
	/** @vm_id: The ID of the VM to bind to */
	__u32 vm_id;

	/** @num_binds: number of binds in this IOCTL. */
	__u32 num_binds;

	/**
	 * @stride: Stride in bytes between consecutive binds. This allows
	 * extensibility of drm_asahi_gem_bind_op.
	 */
	__u32 stride;

	/** @pad: MBZ */
	__u32 pad;

	/**
	 * @userptr: User pointer to an array of @num_binds structures of type
	 * @drm_asahi_gem_bind_op and size @stride bytes.
	 */
	__u64 userptr;
};

/**
 * enum drm_asahi_bind_object_op - Special object bind operation
 */
enum drm_asahi_bind_object_op {
	/** @DRM_ASAHI_BIND_OBJECT_OP_BIND: Bind a BO as a special GPU object */
	DRM_ASAHI_BIND_OBJECT_OP_BIND = 0,

	/** @DRM_ASAHI_BIND_OBJECT_OP_UNBIND: Unbind a special GPU object */
	DRM_ASAHI_BIND_OBJECT_OP_UNBIND = 1,
};

/**
 * enum drm_asahi_bind_object_flags - Special object bind flags
 */
enum drm_asahi_bind_object_flags {
	/**
	 * @DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS: Map a BO as a timestamp
	 * buffer.
	 */
	DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS = (1L << 0),
};

/**
 * struct drm_asahi_gem_bind_object - Arguments passed to
 * DRM_IOCTL_ASAHI_GEM_BIND_OBJECT
 */
struct drm_asahi_gem_bind_object {
	/** @op: Bind operation (enum drm_asahi_bind_object_op) */
	__u32 op;

	/** @flags: Combination of drm_asahi_bind_object_flags flags. */
	__u32 flags;

	/** @handle: GEM object to bind/unbind (BIND) */
	__u32 handle;

	/** @vm_id: The ID of the VM to operate on (MBZ currently) */
	__u32 vm_id;

	/** @offset: Offset into the object (BIND only) */
	__u64 offset;

	/** @range: Number of bytes to bind/unbind (BIND only) */
	__u64 range;

	/** @object_handle: Object handle (out for BIND, in for UNBIND) */
	__u32 object_handle;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * enum drm_asahi_cmd_type - Command type
 */
enum drm_asahi_cmd_type {
	/**
	 * @DRM_ASAHI_CMD_RENDER: Render command, executing on the render
	 * subqueue. Combined vertex and fragment operation.
	 *
	 * Followed by a @drm_asahi_cmd_render payload.
	 */
	DRM_ASAHI_CMD_RENDER = 0,

	/**
	 * @DRM_ASAHI_CMD_COMPUTE: Compute command on the compute subqueue.
	 *
	 * Followed by a @drm_asahi_cmd_compute payload.
	 */
	DRM_ASAHI_CMD_COMPUTE = 1,

	/**
	 * @DRM_ASAHI_SET_VERTEX_ATTACHMENTS: Software command to set
	 * attachments for subsequent vertex shaders in the same submit.
	 *
	 * Followed by (possibly multiple) @drm_asahi_attachment payloads.
	 */
	DRM_ASAHI_SET_VERTEX_ATTACHMENTS = 2,

	/**
	 * @DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS: Software command to set
	 * attachments for subsequent fragment shaders in the same submit.
	 *
	 * Followed by (possibly multiple) @drm_asahi_attachment payloads.
	 */
	DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS = 3,

	/**
	 * @DRM_ASAHI_SET_COMPUTE_ATTACHMENTS: Software command to set
	 * attachments for subsequent compute shaders in the same submit.
	 *
	 * Followed by (possibly multiple) @drm_asahi_attachment payloads.
	 */
	DRM_ASAHI_SET_COMPUTE_ATTACHMENTS = 4,

	/**
	 * @DRM_ASAHI_SET_RENDER_RESOURCES: Replace linked render program storage.
	 * Followed by zero or more drm_asahi_render_program entries. This software
	 * command has no barriers and does not advance either command index.
	 */
	DRM_ASAHI_SET_RENDER_RESOURCES = 5,

	/** @DRM_ASAHI_SET_COMPUTE_RESOURCES: Replace compute storage declarations. */
	DRM_ASAHI_SET_COMPUTE_RESOURCES = 6,
	/** Replace the render list with drm_asahi_render_program_v2 entries. */
	DRM_ASAHI_SET_RENDER_RESOURCES_V2 = 7,
};

/**
 * enum drm_asahi_priority - Scheduling queue priority.
 *
 * These priorities are forwarded to the firmware to influence firmware
 * scheduling. The exact policy is ultimately decided by firmware, but
 * these enums allow userspace to communicate the intentions.
 */
enum drm_asahi_priority {
	/** @DRM_ASAHI_PRIORITY_LOW: Low priority queue. */
	DRM_ASAHI_PRIORITY_LOW = 0,

	/** @DRM_ASAHI_PRIORITY_MEDIUM: Medium priority queue. */
	DRM_ASAHI_PRIORITY_MEDIUM = 1,

	/**
	 * @DRM_ASAHI_PRIORITY_HIGH: High priority queue.
	 *
	 * Reserved for future extension.
	 */
	DRM_ASAHI_PRIORITY_HIGH = 2,

	/**
	 * @DRM_ASAHI_PRIORITY_REALTIME: Real-time priority queue.
	 *
	 * Reserved for future extension.
	 */
	DRM_ASAHI_PRIORITY_REALTIME = 3,
};

/**
 * struct drm_asahi_queue_create - Arguments passed to
 * DRM_IOCTL_ASAHI_QUEUE_CREATE
 */
struct drm_asahi_queue_create {
	/** @flags: MBZ */
	__u32 flags;

	/** @vm_id: The ID of the VM this queue is bound to */
	__u32 vm_id;

	/** @priority: One of drm_asahi_priority */
	__u32 priority;

	/** @queue_id: The returned queue ID */
	__u32 queue_id;

	/**
	 * @usc_exec_base: GPU base address for all USC binaries (shaders) on
	 * this queue. USC addresses are 32-bit relative to this 64-bit base.
	 *
	 * This sets the following registers on all queue commands:
	 *
	 *	USC_EXEC_BASE_TA  (vertex)
	 *	USC_EXEC_BASE_ISP (fragment)
	 *	USC_EXEC_BASE_CP  (compute)
	 *
	 * While the hardware lets us configure these independently per command,
	 * we do not have a use case for this. Instead, we expect userspace to
	 * fix a 4GiB VA carveout for USC memory and pass its base address here.
	 */
	__u64 usc_exec_base;
	__u64 render_scs_layout[2];

	/**
	 * @compute_timeout_ms: Watchdog allowance of each compute command of
	 * this queue, in milliseconds. Zero keeps the driver's default.
	 * Requires DRM_ASAHI_FEATURE_COMPUTE_TIMEOUT; zero otherwise.
	 *
	 * The driver's watchdog polls a queue every two seconds. By default it
	 * takes a compute command that a poll finds still running at the head
	 * of the queue (every earlier compute command of the queue has
	 * completed) to have hung, and fails it, and the queue's VM with it.
	 * With an allowance, the watchdog lets the command run on at each poll
	 * until the allowance has passed. For a nonzero request the driver
	 * raises a shorter allowance to its default of one poll, lowers a
	 * longer one to the maximum the system allows, and returns the
	 * allowance in effect here; a request of 0 is returned unchanged.
	 *
	 * This is an allowance, not a deadline:
	 *
	 * - The watchdog acts at its polls only, so a command may run up to
	 *   one poll longer than its allowance.
	 * - The start of a command is known only approximately. The time is
	 *   counted from one poll before the first poll that finds the command
	 *   at the head of the queue. That is no earlier than the command got
	 *   there and may be later: commands submitted to the queue in the
	 *   meantime postpone that poll.
	 * - Once the allowance has passed, the deferrals the watchdog grants
	 *   any command still apply, as they do by default: a command it takes
	 *   to be waiting behind other clients' work, or one of a kind it
	 *   gives extra polls, runs on until those are used up as well. The
	 *   polls it renews any command for while the GPU keeps completing
	 *   other work (up to 30 s in all, on the GPUs where it does that)
	 *   run alongside the allowance, not after it: a command is renewed
	 *   for the longer of the two.
	 *
	 * The allowance is per command, and nothing bounds the sum: a queue
	 * that is kept full of long commands occupies the GPU for as long as
	 * it is fed. The driver does not shield the other clients from such a
	 * queue: they share the GPU with it as the firmware schedules them,
	 * and a command of another client may have to wait until a long
	 * command ends. The watchdog does not fail such a waiting command,
	 * render or compute, while the earlier command it waits behind runs
	 * inside its allowance; afterwards the usual deferrals apply. A
	 * waiting command is excused this way for at most the longest
	 * allowance the system permits in total, so a command that keeps
	 * waiting behind other clients' work longer than that and its own
	 * deferrals is still failed. Userspace should ask for an allowance
	 * only for queues whose commands need one, and keep those commands as
	 * short as it can.
	 */
	__u32 compute_timeout_ms;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * struct drm_asahi_queue_destroy - Arguments passed to
 * DRM_IOCTL_ASAHI_QUEUE_DESTROY
 */
struct drm_asahi_queue_destroy {
	/** @queue_id: The queue ID to be destroyed */
	__u32 queue_id;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * enum drm_asahi_sync_type - Sync item type
 */
enum drm_asahi_sync_type {
	/** @DRM_ASAHI_SYNC_SYNCOBJ: Binary sync object */
	DRM_ASAHI_SYNC_SYNCOBJ = 0,

	/** @DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ: Timeline sync object */
	DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ = 1,

	/**
	 * @DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE: Output-only queue
	 * timeline anchor, requiring CUMULATIVE_SIGNALS and a nonzero point.
	 * At most one is allowed. All outputs receive completion of this submit
	 * and the earlier anchor points.
	 * Other outputs must not alias the anchor. No execution dependency is
	 * added. Calls updating this private anchor must be externally serialized.
	 */
	DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE = 2,
};

/**
 * struct drm_asahi_sync - Sync item
 */
struct drm_asahi_sync {
	/** @sync_type: One of drm_asahi_sync_type */
	__u32 sync_type;

	/** @handle: The sync object handle */
	__u32 handle;

	/** @timeline_value: Timeline value for timeline sync objects */
	__u64 timeline_value;
};

/**
 * define DRM_ASAHI_BARRIER_NONE - Command index for no barrier
 *
 * This special value may be passed in to drm_asahi_command::vdm_barrier or
 * drm_asahi_command::cdm_barrier to indicate that the respective subqueue
 * should not wait on any previous work.
 */
#define DRM_ASAHI_BARRIER_NONE (0xFFFFu)

/**
 * struct drm_asahi_compute_program - Storage of one linked compute dispatch.
 *
 * SET_COMPUTE_RESOURCES replaces the list for subsequent compute commands.
 * Include each distinct dispatch profile in the command stream. The kernel
 * combines their requirements before publishing work. All sizes are bytes.
 */
struct drm_asahi_compute_program {
	/** @primary_bytes: Per-thread primary storage. */
	__u32 primary_bytes;
	/** @secondary_bytes: Per-thread secondary storage. */
	__u32 secondary_bytes;
	/** @linked_bytes: Additional primary storage after linking. */
	__u32 linked_bytes;
	/** @minimum_primary_bytes: Minimum primary storage across variants. */
	__u32 minimum_primary_bytes;
	/** @threadgroup_bytes: Storage shared by a workgroup. */
	__u32 threadgroup_bytes;
	/** @threads: Number of invocations in a workgroup. */
	__u32 threads;
	/** @flags: Reserved; must be zero. */
	__u32 flags;
};

/**
 * struct drm_asahi_render_program - Storage of one linked render program.
 *
 * SET_RENDER_RESOURCES replaces the list for subsequent renders in this submit.
 * Include every vertex and fragment program, including background and end-of-tile
 * programs. Sizes describe the final linked program, in bytes. Each entry is
 * evaluated separately before taking the maximum demand across the pass.
 */
struct drm_asahi_render_program {
	/** @stage: 0 for vertex, 1 for fragment (including background/end-of-tile). */
	__u32 stage;
	/** @primary_bytes: Primary per-thread storage after linking. */
	__u32 primary_bytes;
	/** @secondary_bytes: Secondary per-thread storage after linking. */
	__u32 secondary_bytes;
	/** @linked_bytes: Additional primary storage from linked programs. */
	__u32 linked_bytes;
	/** @minimum_primary_bytes: Minimum primary storage across linked variants. */
	__u32 minimum_primary_bytes;
	/** @auxiliary_count: Fragment auxiliary entries; zero for vertex programs. */
	__u32 auxiliary_count;
	/** @threadgroup_bytes: Fragment threadgroup storage; zero for vertex. */
	__u32 threadgroup_bytes;
	/** @imageblock_bytes: Fragment imageblock storage; zero for vertex. */
	__u32 imageblock_bytes;
	/** @multisample_bytes: Fragment multisample storage; zero for vertex. */
	__u32 multisample_bytes;
	/** @flags: MBZ. */
	__u32 flags;
};
/** Version 2 linked render storage. Version 1 remains exactly 40 bytes. */
#define DRM_ASAHI_RENDER_PROGRAM_SCS (1U << 0)
#define DRM_ASAHI_RENDER_PROGRAM_MESH (1U << 1)
#define DRM_ASAHI_RENDER_PROGRAM_OBJECT (1U << 2)
/* With OBJECT: the object program is an API task stage. The kernel binds the
 * TA mesh rendering buffer (the backing of the object payload address) to
 * such kicks only. Requires DRM_ASAHI_FEATURE_TASK_MESH_STORAGE. */
#define DRM_ASAHI_RENDER_PROGRAM_TASK (1U << 3)
struct drm_asahi_render_program_v2 {
	struct drm_asahi_render_program legacy;
	/** @threads: Mesh/object workgroup invocation count; zero otherwise. */
	__u32 threads;
	/** @usage_flags: DRM_ASAHI_RENDER_PROGRAM_* bits. */
	__u32 usage_flags;
	/** @scs_layout: Packed depth/levels/cliques/user/driver SCS layout. */
	__u64 scs_layout;
	/** @auxiliary_allocation: Stage native auxiliary allocation word. */
	__u32 auxiliary_allocation;
	__u32 reserved;
	__u64 intersection_pool;
	__u64 intersection_pool_bytes;
	__u32 intersection_first;
	__u32 intersection_count;
	__u64 intersection_default_program;
	__u64 intersection_default_control;
};

/**
 * struct drm_asahi_cmd_header - Top level command structure
 *
 * This struct is core to the command buffer definition and therefore is not
 * extensible.
 */
struct drm_asahi_cmd_header {
	/** @cmd_type: One of drm_asahi_cmd_type */
	__u16 cmd_type;

	/**
	 * @size: Size of this command, not including this header.
	 *
	 * For hardware commands, this enables extensibility of commands without
	 * requiring extra command types. Passing a command that is shorter
	 * than expected is explicitly allowed for backwards-compatibility.
	 * Truncated fields will be zeroed.
	 *
	 * For the synthetic attachment setting commands, this implicitly
	 * encodes the number of attachments. These commands take multiple
	 * fixed-size @drm_asahi_attachment structures as their payload, so size
	 * equals number of attachments * sizeof(struct drm_asahi_attachment).
	 */
	__u16 size;

	/**
	 * @vdm_barrier: VDM (render) command index to wait on.
	 *
	 * Barriers are indices relative to the beginning of a given submit. A
	 * barrier of 0 waits on commands submitted to the respective subqueue
	 * in previous submit ioctls. A barrier of N waits on N previous
	 * commands on the subqueue within the current submit ioctl. As a
	 * special case, passing @DRM_ASAHI_BARRIER_NONE avoids waiting on any
	 * commands in the subqueue.
	 *
	 * Examples:
	 *
	 *   0: This waits on all previous work.
	 *
	 *   NONE: This does not wait for anything on this subqueue.
	 *
	 *   1: This waits on the first render command in the submit.
	 *   This is valid only if there are multiple render commands in the
	 *   same submit.
	 *
	 * Barriers are valid only for hardware commands. Synthetic software
	 * commands to set attachments must pass NONE here.
	 */
	__u16 vdm_barrier;

	/**
	 * @cdm_barrier: CDM (compute) command index to wait on.
	 *
	 * See @vdm_barrier, and replace VDM/render with CDM/compute.
	 */
	__u16 cdm_barrier;
};

/**
 * struct drm_asahi_submit - Arguments passed to DRM_IOCTL_ASAHI_SUBMIT
 */
struct drm_asahi_submit {
	/**
	 * @syncs: An optional pointer to an array of drm_asahi_sync. The first
	 * @in_sync_count elements are in-syncs, then the remaining
	 * @out_sync_count elements are out-syncs. Using a single array with
	 * explicit partitioning simplifies handling.
	 */
	__u64 syncs;

	/**
	 * @cmdbuf: Pointer to the command buffer to submit.
	 *
	 * This is a flat command buffer. By design, it contains no CPU
	 * pointers, which makes it suitable for a virtgpu wire protocol without
	 * requiring any serializing/deserializing step.
	 *
	 * It consists of a series of commands. Each command begins with a
	 * fixed-size @drm_asahi_cmd_header header and is followed by a
	 * variable-length payload according to the type and size in the header.
	 *
	 * The combined count of "real" hardware commands must be nonzero and at
	 * most drm_asahi_params_global::max_commands_per_submission.
	 */
	__u64 cmdbuf;

	/** @flags: Flags for command submission (MBZ) */
	__u32 flags;

	/** @queue_id: The queue ID to be submitted to */
	__u32 queue_id;

	/**
	 * @in_sync_count: Number of sync objects to wait on before starting
	 * this job.
	 */
	__u32 in_sync_count;

	/**
	 * @out_sync_count: Number of sync objects to signal upon completion of
	 * this job.
	 */
	__u32 out_sync_count;

	/** @cmdbuf_size: Command buffer size in bytes */
	__u32 cmdbuf_size;

	/** @pad: MBZ */
	__u32 pad;
};

/**
 * struct drm_asahi_attachment - Describe an "attachment".
 *
 * Attachments are any memory written by shaders, notably including render
 * target attachments written by the end-of-tile program. This is purely a hint
 * about the accessed memory regions. It is optional to specify, which is
 * fortunate as it cannot be specified precisely with bindless access anyway.
 * But where possible, it's probably a good idea for userspace to include these
 * hints, forwarded to the firmware.
 *
 * This struct is implicitly sized and therefore is not extensible.
 */
struct drm_asahi_attachment {
	/** @pointer: Base address of the attachment */
	__u64 pointer;

	/** @size: Size of the attachment in bytes */
	__u64 size;

	/** @pad: MBZ */
	__u32 pad;

	/** @flags: MBZ */
	__u32 flags;
};

enum drm_asahi_render_flags {
	/**
	 * @DRM_ASAHI_RENDER_VERTEX_SCRATCH: A vertex stage shader uses scratch
	 * memory.
	 */
	DRM_ASAHI_RENDER_VERTEX_SCRATCH = (1U << 0),

	/**
	 * @DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES: Process even empty tiles.
	 * This must be set when clearing render targets.
	 */
	DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES = (1U << 1),

	/**
	 * @DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING: Run vertex stage on a single
	 * cluster (on multi-cluster GPUs)
	 *
	 * This harms performance but can workaround certain sync/coherency
	 * bugs, and therefore is useful for debugging.
	 */
	DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING = (1U << 2),

	/**
	 * @DRM_ASAHI_RENDER_RSRC_SPEC_HI: The appended resource-specifier high
	 * dwords are present. This flag lets old kernels report an unsupported
	 * render payload instead of silently truncating 64-bit G17 values.
	 */
	DRM_ASAHI_RENDER_RSRC_SPEC_HI = (1U << 4),

	/**
	 * @DRM_ASAHI_RENDER_DBIAS_IS_INT: Use integer depth bias formula.
	 *
	 * Graphics specifications contain two alternate formulas for depth
	 * bias, a float formula used with floating-point depth buffers and an
	 * integer formula using with unorm depth buffers. This flag specifies
	 * that the integer formula should be used. If omitted, the float
	 * formula is used instead.
	 *
	 * This corresponds to bit 18 of the relevant hardware control register,
	 * so we match that here for efficiency.
	 */
	DRM_ASAHI_RENDER_DBIAS_IS_INT = (1U << 18),

	/**
	 * @DRM_ASAHI_RENDER_FEW_PRIMITIVES: The render pass is known to rasterize
	 * at most four primitives in total.
	 *
	 * The count includes every primitive the pass generates: topology-derived
	 * primitive counts of all draws multiplied by their instance counts, plus
	 * internal work (clears, visibility, scissor/state generated work). Passes
	 * with indirect draws, mesh work or any work whose primitive count is not
	 * known at record time must not set this flag. The flag sets bit 0 of the
	 * auxiliary framebuffer control word for the fragment stage; the hardware
	 * effect is an optimization for small known passes and its omission is
	 * always safe. Requires DRM_ASAHI_FEATURE_FEW_PRIMITIVES.
	 */
	DRM_ASAHI_RENDER_FEW_PRIMITIVES = (1U << 19),

	/**
	 * @DRM_ASAHI_RENDER_FRAGMENT_BARRIERS: Only the fragment stage waits for
	 * this command's barriers.
	 *
	 * The render dependencies resolved from @vdm_barrier and @prior_vdm are
	 * attached to the fragment half of the render instead of its vertex
	 * (tiling) half, so vertex processing may start while the renders this
	 * command depends on are still executing their fragment work. The
	 * fragment half still waits for its own vertex half and for the previous
	 * fragment work of its physical queue; the compute dependencies from
	 * @cdm_barrier and @prior_cdm keep ordering the vertex half. Set this
	 * only when every render dependency into the command is consumed by
	 * fragment-stage or attachment accesses alone: nothing the vertex stage
	 * reads (vertex/index/indirect buffers, vertex, tessellation or geometry
	 * shader resources, transform feedback, conditional-rendering predicates)
	 * may depend on it. Requires DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS.
	 */
	DRM_ASAHI_RENDER_FRAGMENT_BARRIERS = (1U << 20),


	/**
	 * @DRM_ASAHI_RENDER_SHADER_WRITES: A graphics program in this command
	 * writes external memory, including buffer/image stores or atomic RMW.
	 * Ordinary raster outputs and compiler-private transport do not count.
	 * Requires DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES. Omission retains the
	 * existing encoding, allowing older userspace to submit unchanged.
	 */
	DRM_ASAHI_RENDER_SHADER_WRITES = (1U << 21),

	/**
	 * @DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS: The command is a struct
	 * drm_asahi_cmd_render_with_stage_prior. Its tiling half waits only on
	 * the compute prefix named by ta_cdm_barrier and ta_prior_cdm, which must
	 * not exceed the command's own compute prefix; its fragment half waits on
	 * the whole one. Requires DRM_ASAHI_FEATURE_COMPUTE_STAGE_BARRIERS.
	 */
	DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS = (1U << 22),

	/**
	 * @DRM_ASAHI_RENDER_STAGE_PREFIXES: The command is a struct
	 * drm_asahi_cmd_render_with_stage_prefixes. Its tiling half waits only on
	 * the render and compute prefixes of the tail, each no later than the
	 * command's own prefix of that subqueue; its fragment half waits on the
	 * whole ones. Earlier renders in the tiling prefix are waited on to
	 * completion. Not combined with DRM_ASAHI_RENDER_FRAGMENT_BARRIERS or
	 * DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS. Requires
	 * DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES.
	 */
	DRM_ASAHI_RENDER_STAGE_PREFIXES = (1U << 23),

	/**
	 * @DRM_ASAHI_RENDER_WHOLE_ATTACHMENTS: The render area covers the full
	 * extent of every attachment, so the render's tile stores write no pixel
	 * outside it. The kernel need not order such a render against another
	 * one that writes the same attachments without a dependency between
	 * them: any overlap is the application's own race. Requires
	 * DRM_ASAHI_FEATURE_WHOLE_ATTACHMENT_RENDERS.
	 */
	DRM_ASAHI_RENDER_WHOLE_ATTACHMENTS = (1U << 24),

	/**
	 * @DRM_ASAHI_RENDER_G18_RESOURCES: G18 only, requires
	 * DRM_ASAHI_FEATURE_G18_GRAPHICS_SCRATCH. The payload is a
	 * drm_asahi_cmd_render_g18: the complete render command, including the
	 * prior and ray-query space, followed by drm_asahi_g18_render_resources.
	 */
	DRM_ASAHI_RENDER_G18_RESOURCES = (1U << 25),

	/**
	 * @DRM_ASAHI_RENDER_MESH_OUTPUT: The render contains mesh shading. The
	 * kernel may size the render's tiler heap for mesh output ahead of it.
	 * A hint only: a render without the flag is still correct to submit.
	 * Requires DRM_ASAHI_FEATURE_RENDER_MESH_OUTPUT.
	 */
	DRM_ASAHI_RENDER_MESH_OUTPUT = (1U << 26),
};

/**
 * struct drm_asahi_zls_buffer - Describe a depth or stencil buffer.
 *
 * These fields correspond to hardware registers in the ZLS (Z Load/Store) unit.
 * There are three hardware registers for each field respectively for loads,
 * stores, and partial renders. In practice, it makes sense to set all to the
 * same values, except in exceptional cases not yet implemented in userspace, so
 * we do not duplicate here for simplicity/efficiency.
 *
 * This struct is embedded in other structs and therefore is not extensible.
 */
struct drm_asahi_zls_buffer {
	/** @base: Base address of the buffer */
	__u64 base;

	/**
	 * @comp_base: If the load buffer is compressed, address of the
	 * compression metadata section.
	 */
	__u64 comp_base;

	/**
	 * @stride: If layered rendering is enabled, the number of bytes
	 * between each layer of the buffer.
	 */
	__u32 stride;

	/**
	 * @comp_stride: If layered rendering is enabled, the number of bytes
	 * between each layer of the compression metadata.
	 */
	__u32 comp_stride;
};

/**
 * struct drm_asahi_timestamp - Describe a timestamp write.
 *
 * The firmware can optionally write the GPU timestamp at render pass
 * granularities, but it needs to be mapped specially via
 * DRM_IOCTL_ASAHI_GEM_BIND_OBJECT. This structure therefore describes where to
 * write as a handle-offset pair, rather than a GPU address like normal.
 *
 * This struct is embedded in other structs and therefore is not extensible.
 */
struct drm_asahi_timestamp {
	/**
	 * @handle: Handle of the timestamp buffer, or 0 to skip this
	 * timestamp. If nonzero, this must equal the value returned in
	 * drm_asahi_gem_bind_object::object_handle.
	 */
	__u32 handle;

	/** @offset: Offset to write into the timestamp buffer */
	__u32 offset;
};

/**
 * struct drm_asahi_timestamps - Describe timestamp writes.
 *
 * Each operation that can be timestamped, can be timestamped at the start and
 * end. Therefore, drm_asahi_timestamp structs always come in pairs, bundled
 * together into drm_asahi_timestamps.
 *
 * This struct is embedded in other structs and therefore is not extensible.
 */
struct drm_asahi_timestamps {
	/** @start: Timestamp recorded at the start of the operation */
	struct drm_asahi_timestamp start;

	/** @end: Timestamp recorded at the end of the operation */
	struct drm_asahi_timestamp end;
};

/**
 * struct drm_asahi_helper_program - Describe helper program configuration.
 *
 * The helper program is a compute-like kernel required for various hardware
 * functionality. Its most important role is dynamically allocating
 * scratch/stack memory for individual subgroups, by partitioning a static
 * allocation shared for the whole device. It is supplied by userspace via
 * drm_asahi_helper_program and internally dispatched by the hardware as needed.
 *
 * This struct is embedded in other structs and therefore is not extensible.
 */
struct drm_asahi_helper_program {
	/**
	 * @binary: USC address to the helper program binary. This is a tagged
	 * pointer with configuration in the bottom bits.
	 */
	__u32 binary;

	/** @cfg: Additional configuration bits for the helper program. */
	__u32 cfg;

	/**
	 * @data: Data passed to the helper program. This value is not
	 * interpreted by the kernel, firmware, or hardware in any way. It is
	 * simply a sideband for userspace, set with the submit ioctl and read
	 * via special registers inside the helper program.
	 *
	 * In practice, userspace will pass a 64-bit GPU VA here pointing to the
	 * actual arguments, which presumably don't fit in 64-bits.
	 */
	__u64 data;
};

/**
 * struct drm_asahi_bg_eot - Describe a background or end-of-tile program.
 *
 * The background and end-of-tile programs are dispatched by the hardware at the
 * beginning and end of rendering. As the hardware "tilebuffer" is simply local
 * memory, these programs are necessary to implement API-level render targets.
 * The fragment-like background program is responsible for loading either the
 * clear colour or the existing render target contents, while the compute-like
 * end-of-tile program stores the tilebuffer contents to memory.
 *
 * This struct is embedded in other structs and therefore is not extensible.
 */
struct drm_asahi_bg_eot {
	/**
	 * @usc: USC address of the hardware USC words binding resources
	 * (including images and uniforms) and the program itself. Note this is
	 * an additional layer of indirection compared to the helper program,
	 * avoiding the need for a sideband for data. This is a tagged pointer
	 * with additional configuration in the bottom bits.
	 */
	__u32 usc;

	/**
	 * @rsrc_spec: Resource specifier for the program. This is a packed
	 * hardware data structure describing the required number of registers,
	 * uniforms, bound textures, and bound samplers.
	 */
	__u32 rsrc_spec;
};

/**
 * struct drm_asahi_cmd_render - Command to submit 3D
 *
 * This command submits a single render pass. The hardware control stream may
 * include many draws and subpasses, but within the command, the framebuffer
 * dimensions and attachments are fixed.
 *
 * The hardware requires the firmware to set a large number of Control Registers
 * setting up state at render pass granularity before each command rendering 3D.
 * The firmware bundles this state into data structures. Unfortunately, we
 * cannot expose either any of that directly to userspace, because the
 * kernel-firmware ABI is not stable. Although we can guarantee the firmware
 * updates in tandem with the kernel, we cannot break old userspace when
 * upgrading the firmware and kernel. Therefore, we need to abstract well the
 * data structures to avoid tying our hands with future firmwares.
 *
 * The bulk of drm_asahi_cmd_render therefore consists of values of hardware
 * control registers, marshalled via the firmware interface.
 *
 * The framebuffer/tilebuffer dimensions are also specified here. In addition to
 * being passed to the firmware/hardware, the kernel requires these dimensions
 * to calculate various essential tiling-related data structures. It is
 * unfortunate that our submits are heavier than on vendors with saner
 * hardware-software interfaces. The upshot is all of this information is
 * readily available to userspace with all current APIs.
 *
 * It looks odd - but it's not overly burdensome and it ensures we can remain
 * compatible with old userspace.
 */
struct drm_asahi_cmd_render {
	/** @flags: Combination of drm_asahi_render_flags flags. */
	__u32 flags;

	/**
	 * @isp_zls_pixels: ISP_ZLS_PIXELS register value. This contains the
	 * depth/stencil width/height, which may differ from the framebuffer
	 * width/height.
	 */
	__u32 isp_zls_pixels;

	/**
	 * @vdm_ctrl_stream_base: VDM_CTRL_STREAM_BASE register value. GPU
	 * address to the beginning of the VDM control stream.
	 */
	__u64 vdm_ctrl_stream_base;

	/** @vertex_helper: Helper program used for the vertex shader */
	struct drm_asahi_helper_program vertex_helper;

	/** @fragment_helper: Helper program used for the fragment shader */
	struct drm_asahi_helper_program fragment_helper;

	/**
	 * @isp_scissor_base: ISP_SCISSOR_BASE register value. GPU address of an
	 * array of scissor descriptors indexed in the render pass.
	 */
	__u64 isp_scissor_base;

	/**
	 * @isp_dbias_base: ISP_DBIAS_BASE register value. GPU address of an
	 * array of depth bias values indexed in the render pass.
	 */
	__u64 isp_dbias_base;

	/**
	 * @isp_oclqry_base: ISP_OCLQRY_BASE register value. GPU address of an
	 * array of occlusion query results written by the render pass.
	 */
	__u64 isp_oclqry_base;

	/** @depth: Depth buffer */
	struct drm_asahi_zls_buffer depth;

	/** @stencil: Stencil buffer */
	struct drm_asahi_zls_buffer stencil;

	/** @zls_ctrl: ZLS_CTRL register value */
	__u64 zls_ctrl;

	/** @ppp_multisamplectl: PPP_MULTISAMPLECTL register value */
	__u64 ppp_multisamplectl;

	/**
	 * @sampler_heap: Base address of the sampler heap. This heap is used
	 * for both vertex shaders and fragment shaders. The registers are
	 * per-stage, but there is no known use case for separate heaps.
	 */
	__u64 sampler_heap;

	/** @ppp_ctrl: PPP_CTRL register value */
	__u32 ppp_ctrl;

	/** @width_px: Framebuffer width in pixels */
	__u16 width_px;

	/** @height_px: Framebuffer height in pixels */
	__u16 height_px;

	/** @layers: Number of layers in the framebuffer */
	__u16 layers;

	/** @sampler_count: Number of samplers in the sampler heap. */
	__u16 sampler_count;

	/** @utile_width_px: Width of a logical tilebuffer tile in pixels */
	__u8 utile_width_px;

	/** @utile_height_px: Height of a logical tilebuffer tile in pixels */
	__u8 utile_height_px;

	/** @samples: # of samples in the framebuffer. Must be 1, 2, or 4. */
	__u8 samples;

	/** @sample_size_B: # of bytes in the tilebuffer required per sample. */
	__u8 sample_size_B;

	/**
	 * @isp_merge_upper_x: 32-bit float used in the hardware triangle
	 * merging. Calculate as: tan(60 deg) * width.
	 *
	 * Making these values UAPI avoids requiring floating-point calculations
	 * in the kernel in the hot path.
	 */
	__u32 isp_merge_upper_x;

	/**
	 * @isp_merge_upper_y: 32-bit float. Calculate as: tan(60 deg) * height.
	 * See @isp_merge_upper_x.
	 */
	__u32 isp_merge_upper_y;

	/** @bg: Background program run for each tile at the start */
	struct drm_asahi_bg_eot bg;

	/** @eot: End-of-tile program ran for each tile at the end */
	struct drm_asahi_bg_eot eot;

	/**
	 * @partial_bg: Background program ran at the start of each tile when
	 * resuming the render pass during a partial render.
	 */
	struct drm_asahi_bg_eot partial_bg;

	/**
	 * @partial_eot: End-of-tile program ran at the end of each tile when
	 * pausing the render pass during a partial render.
	 */
	struct drm_asahi_bg_eot partial_eot;

	/**
	 * @isp_bgobjdepth: ISP_BGOBJDEPTH register value. This is the depth
	 * buffer clear value, encoded in the depth buffer's format: either a
	 * 32-bit float or a 16-bit unorm (with upper bits zeroed).
	 */
	__u32 isp_bgobjdepth;

	/**
	 * @isp_bgobjvals: ISP_BGOBJVALS register value. The bottom 8-bits
	 * contain the stencil buffer clear value.
	 */
	__u32 isp_bgobjvals;

	/** @ts_vtx: Timestamps for the vertex portion of the render */
	struct drm_asahi_timestamps ts_vtx;

	/** @ts_frag: Timestamps for the fragment portion of the render */
	struct drm_asahi_timestamps ts_frag;

	/** @bg_eot_rsrc_spec_hi: High dword of @eot.rsrc_spec */
	__u32 bg_eot_rsrc_spec_hi;

	/** @bg_eot_partial_rsrc_spec_hi: High dword of @partial_eot.rsrc_spec */
	__u32 bg_eot_partial_rsrc_spec_hi;

	/** @bg_rsrc_spec_hi: High dword of @bg.rsrc_spec */
	__u32 bg_rsrc_spec_hi;

	/** @bg_partial_rsrc_spec_hi: High dword of @partial_bg.rsrc_spec */
	__u32 bg_partial_rsrc_spec_hi;

	/**
	 * @ts_frag_end_copies: Timestamp copies used by the M3 (J613) runtime.
	 * Must be zero for G17P.
	 */
	struct drm_asahi_timestamp ts_frag_end_copies[8];
};

/* Feature-gated extensions preserve the legacy packet sizes used by other
 * clients. Ordinals are 1-based per logical queue and hardware command type,
 * counting every submitted hardware command including maintenance commands.
 * Zero means absent. Nonzero values must not exceed that engine's sequence
 * at the beginning of this ioctl; they merge with the header's relative
 * barrier prefix. NONE in the header does not disable these dependencies.
 */
struct drm_asahi_cmd_render_with_prior {
	struct drm_asahi_cmd_render legacy;
	__u64 prior_vdm;
	__u64 prior_cdm;
};

/* DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS: the tiling half's compute prefix,
 * a command index like drm_asahi_cmd_header::cdm_barrier and an exact prior
 * ordinal like prior_cdm. Its tail takes the place of the ray-query stage
 * state of struct drm_asahi_cmd_render_with_ray_query, so a command carries
 * one of the two. */
struct drm_asahi_cmd_render_with_stage_prior {
	struct drm_asahi_cmd_render_with_prior legacy;
	__u16 ta_cdm_barrier;
	__u16 reserved0;
	__u32 reserved1;
	__u64 ta_prior_cdm;
};

/* DRM_ASAHI_RENDER_STAGE_PREFIXES: the tiling half's render and compute
 * prefixes, each a command index like the header barriers plus an exact prior
 * ordinal. The reserved fields are zero. */
struct drm_asahi_cmd_render_with_stage_prefixes {
	struct drm_asahi_cmd_render_with_prior legacy;
	__u16 ta_cdm_barrier;
	__u16 reserved0;
	__u32 reserved1;
	__u64 ta_prior_cdm;
	__u16 ta_vdm_barrier;
	__u16 reserved2;
	__u32 reserved3;
	__u64 ta_prior_vdm;
};

struct drm_asahi_render_ray_query_stage {
	__u64 scs_layout;
	__u32 intersection_aux;
	__u32 intersection_pad;
	__u64 intersection_pool;
	__u64 intersection_pool_bytes;
	__u32 intersection_first;
	__u32 intersection_count;
	__u64 intersection_default;
	__u64 intersection_default_control;
};

struct drm_asahi_cmd_render_with_ray_query {
	struct drm_asahi_cmd_render_with_prior legacy;
	struct drm_asahi_render_ray_query_stage ray_query[2];
};

/* The tiling half's render and compute prefixes of a staged render
 * (DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2), each a command index like the
 * header barriers plus an exact prior ordinal. The reserved fields are zero. */
struct drm_asahi_render_stage_prefixes {
	__u16 ta_cdm_barrier;
	__u16 reserved0;
	__u32 reserved1;
	__u64 ta_prior_cdm;
	__u16 ta_vdm_barrier;
	__u16 reserved2;
	__u32 reserved3;
	__u64 ta_prior_vdm;
};

/* A render with DRM_ASAHI_RENDER_STAGE_PREFIXES under
 * DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2: 480 bytes. */
struct drm_asahi_cmd_render_staged {
	struct drm_asahi_cmd_render_with_ray_query render;
	struct drm_asahi_render_stage_prefixes stage_prefixes;
};

/* G18 per-render program demand, maximized over every draw of the render:
 * 16-bit registers, the static thread-private (TPS) frame in bytes as
 * programmed in SPECTPR (a multiple of 16), and IPR/staging halfwords. Tile
 * programs are background, end-of-tile, partial background and partial
 * end-of-tile; a zero register count marks an absent program, and tile
 * programs never carry scratch. flags must be 3 (vertex and fragment
 * present); reserved must be zero. Tile storage comes from the command. */
struct drm_asahi_g18_render_resources {
	__u32 version;
	__u32 flags;
	__u32 vertex_registers_16;
	__u32 vertex_scratch_bytes;
	__u32 vertex_staging_16;
	__u32 fragment_registers_16;
	__u32 fragment_scratch_bytes;
	__u32 fragment_staging_16;
	__u32 tile_registers_16[4];
	__u32 tile_staging_16[4];
	__u32 reserved[4];
};

/* DRM_ASAHI_RENDER_G18_RESOURCES payload: the complete render command
 * (448 bytes: drm_asahi_cmd_render, prior_vdm/prior_cdm and the ray-query
 * stages, all zero when absent) followed by the suffix at offset 448. */
struct drm_asahi_cmd_render_g18 {
	struct drm_asahi_cmd_render_with_ray_query legacy;
	struct drm_asahi_g18_render_resources resources;
};


/**
 * struct drm_asahi_cmd_compute - Command to submit compute
 *
 * This command submits a control stream consisting of compute dispatches. There
 * is essentially no limit on how many compute dispatches may be included in a
 * single compute command, although timestamps are at command granularity.
 */
struct drm_asahi_cmd_compute {
	/**
	 * @flags: Combination of drm_asahi_compute_flags.
	 *
	 * The G17P add3 proof flag is a bring-up-only contract. It treats exactly
	 * three compute attachments as input A, input B, and output respectively.
	 */
	__u32 flags;

	/** @sampler_count: Number of samplers in the sampler heap. */
	__u32 sampler_count;

	/**
	 * @cdm_ctrl_stream_base: CDM_CTRL_STREAM_BASE register value. GPU
	 * address to the beginning of the CDM control stream.
	 */
	__u64 cdm_ctrl_stream_base;

	/**
	 * @cdm_ctrl_stream_end: GPU base address to the end of the hardware
	 * control stream. Note this only considers the first contiguous segment
	 * of the control stream, as the stream might jump elsewhere.
	 */
	__u64 cdm_ctrl_stream_end;

	/** @sampler_heap: Base address of the sampler heap. */
	__u64 sampler_heap;

	/** @helper: Helper program used for this compute command */
	struct drm_asahi_helper_program helper;

	/** @ts: Timestamps for the compute command */
	struct drm_asahi_timestamps ts;

	/**
	 * @ts_end_copies: Timestamp copies used by the M3 (J613) runtime.
	 * Must be zero for G17P.
	 */
	struct drm_asahi_timestamp ts_end_copies[8];
};

/* G18 per-dispatch metadata. Counts are compiler halfwords or bytes as named;
 * the kernel converts them to native units. Both versions are exactly 64 bytes.
 * Version 1 scales local_bytes + variable_shared_bytes by four for legacy
 * shaders using byte offsets in element-index fields. Version 2 uses actual
 * byte footprints for shaders with correctly lowered shared element indices.
 * Flags: bit0 main staging known, bit1 constant program present, bit2 constant
 * staging known, bit3 local geometry unknown (local_size must then be zero).
 * All other flag bits and reserved must be zero. */
struct drm_asahi_g18_compute_resources {
	__u32 version;
	__u32 flags;
	__u32 registers_16;
	__u32 scratch_bytes;
	__u32 spectpr_scratch_bytes;
	__u32 staging_16;
	__u32 constant_registers_16;
	__u32 constant_scratch_bytes;
	__u32 constant_staging_16;
	__u32 local_bytes;
	__u32 imageblock_stride;
	__u32 variable_shared_bytes;
	__u32 local_size[3];
	__u32 reserved;
};

struct drm_asahi_cmd_compute_with_prior {
	struct drm_asahi_cmd_compute legacy;
	/* Canonical kernel compute payload already has scs_layout at byte64.
	 * Mesa's legacy compute packet ends before it. Keep this zero until the
	 * shader-context scratch contract is implemented here. */
	__u64 scs_layout;
	__u64 prior_vdm;
	__u64 prior_cdm;
	/* Native M5 compute allocation word, independent of scs_layout.
	 * Zero preserves the legacy path. Older kernels reject a nonzero tail. */
	__u32 cdm_auxiliary_allocation;
	__u32 reserved;
	/** @intersection_pool: M5 intersection-table pool, 16 KiB aligned. */
	__u64 intersection_pool;
	/** @intersection_pool_bytes: Mapped size, a multiple of 16 KiB. */
	__u64 intersection_pool_bytes;
	/** @intersection_first: First 128-byte driver table block. */
	__u32 intersection_first;
	/** @intersection_count: Number of driver table blocks (1..262143). */
	__u32 intersection_count;
	/**
	 * @intersection_default_program: Native default program descriptor.
	 * Low 48 bits are the 64-byte aligned program address; upper 16 bits
	 * contain the low resource-specification bits.
	 */
	__u64 intersection_default_program;
	/**
	 * @intersection_default_control: Native default program control word.
	 * Bits 13:0 and 46:32 are defined by the M5 program descriptor. All
	 * other bits must be zero. A nonzero default program requires bit 40.
	 * An all-zero intersection extension preserves the ordinary compute path.
	 */
	__u64 intersection_default_control;
};

/* DRM_ASAHI_COMPUTE_G18_RESOURCES payload: the complete compute command
 * (200 bytes: drm_asahi_cmd_compute, scs_layout, prior_vdm/prior_cdm,
 * cdm_auxiliary_allocation and the intersection fields, all zero on G18)
 * followed by the suffix at offset 200. */
struct drm_asahi_cmd_compute_g18 {
	struct drm_asahi_cmd_compute_with_prior base;
	struct drm_asahi_g18_compute_resources resources;
};

/**
 * enum drm_asahi_compute_flags - Compute command flags
 */
enum drm_asahi_compute_flags {
	/**
	 * @DRM_ASAHI_COMPUTE_G17P_ADD3_PROOF: Populate the retained G17P
	 * three-buffer table from this command's three compute attachments.
	 */
	DRM_ASAHI_COMPUTE_G17P_ADD3_PROOF = (1L << 0),

	/**
	 * @DRM_ASAHI_COMPUTE_AS_BUILD: Acceleration-structure build work,
	 * including compute-based transfers. Requires BOUNDED_AS_BUILD.
	 * This is an execution hint, not proof of progress or permission to
	 * access additional memory. The kernel validates the command normally
	 * and caps its execution allowance at ten seconds (four extra 2s polls).
	 * Other commands in the same submission receive the same allowance so
	 * ordered helpers cannot time out while waiting for the build.
	 */
	DRM_ASAHI_COMPUTE_AS_BUILD = (1L << 2),

	/*
	 * Bits 16 and up are accepted only by the M5 (G17G/G17S) runtime.
	 */

	/**
	 * @DRM_ASAHI_COMPUTE_USES_MATRIX: The command's dispatches use native
	 * matrix instructions. Requires DRM_ASAHI_FEATURE_COMPUTE_MATRIX.
	 */
	DRM_ASAHI_COMPUTE_USES_MATRIX = (1L << 16),

	/**
	 * @DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS: The command reads and writes no
	 * timestamp or availability word whose host publication by the kernel
	 * is still pending for an earlier command, so it need not wait for that
	 * publication. The command may copy or reset queries whose words are
	 * all already published, or were never bound for host timestamps.
	 * Requires DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS.
	 */
	DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS = (1L << 17),

	/**
	 * @DRM_ASAHI_COMPUTE_USES_TEXTURE_STATE: G18G only. The command's
	 * dispatch binds texture or image state (a texture table, a sampled
	 * texture, a storage image). The kernel selects the compute mode the
	 * native driver uses for such commands (desc+0x62a, register 0x1a458
	 * 0x14c08a60 instead of 0x10c08860). Other kernels refuse the bit.
	 */
	DRM_ASAHI_COMPUTE_USES_TEXTURE_STATE = (1L << 18),

	/**
	 * @DRM_ASAHI_COMPUTE_G18_RESOURCES: G18 only. The payload is a
	 * drm_asahi_cmd_compute_g18: the complete compute command followed by
	 * the versioned drm_asahi_g18_compute_resources suffix.
	 */
	DRM_ASAHI_COMPUTE_G18_RESOURCES = (1L << 19),

	/**
	 * @DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION: M5: the command's control
	 * stream ends with a strong, memory-scoped cache flush, so its writes
	 * are visible once it completes. A compute command of another queue may
	 * then wait for it in firmware rather than on the host. Requires
	 * DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION.
	 */
	DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION = (1L << 20),
};

/**
 * struct drm_asahi_get_time - Arguments passed to DRM_IOCTL_ASAHI_GET_TIME
 */
struct drm_asahi_get_time {
	/** @flags: MBZ. */
	__u64 flags;

	/** @gpu_timestamp: On return, the GPU timestamp in nanoseconds. */
	__u64 gpu_timestamp;
};

/**
 * DRM_IOCTL_ASAHI() - Build an Asahi IOCTL number
 * @__access: Access type. Must be R, W or RW.
 * @__id: One of the DRM_ASAHI_xxx id.
 * @__type: Suffix of the type being passed to the IOCTL.
 *
 * Don't use this macro directly, use the DRM_IOCTL_ASAHI_xxx
 * values instead.
 *
 * Return: An IOCTL number to be passed to ioctl() from userspace.
 */
#define DRM_IOCTL_ASAHI(__access, __id, __type) \
	DRM_IO ## __access(DRM_COMMAND_BASE + DRM_ASAHI_ ## __id, \
			   struct drm_asahi_ ## __type)

/* Note: this is an enum so that it can be resolved by Rust bindgen. */
enum {
	DRM_IOCTL_ASAHI_GET_PARAMS       = DRM_IOCTL_ASAHI(W, GET_PARAMS, get_params),
	DRM_IOCTL_ASAHI_GET_TIME         = DRM_IOCTL_ASAHI(WR, GET_TIME, get_time),
	DRM_IOCTL_ASAHI_VM_CREATE        = DRM_IOCTL_ASAHI(WR, VM_CREATE, vm_create),
	DRM_IOCTL_ASAHI_VM_DESTROY       = DRM_IOCTL_ASAHI(W, VM_DESTROY, vm_destroy),
	DRM_IOCTL_ASAHI_VM_BIND          = DRM_IOCTL_ASAHI(W, VM_BIND, vm_bind),
	DRM_IOCTL_ASAHI_GEM_CREATE       = DRM_IOCTL_ASAHI(WR, GEM_CREATE, gem_create),
	DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET  = DRM_IOCTL_ASAHI(WR, GEM_MMAP_OFFSET, gem_mmap_offset),
	DRM_IOCTL_ASAHI_GEM_BIND_OBJECT  = DRM_IOCTL_ASAHI(WR, GEM_BIND_OBJECT, gem_bind_object),
	DRM_IOCTL_ASAHI_QUEUE_CREATE     = DRM_IOCTL_ASAHI(WR, QUEUE_CREATE, queue_create),
	DRM_IOCTL_ASAHI_QUEUE_DESTROY    = DRM_IOCTL_ASAHI(W, QUEUE_DESTROY, queue_destroy),
	DRM_IOCTL_ASAHI_SUBMIT           = DRM_IOCTL_ASAHI(W, SUBMIT, submit),
	DRM_IOCTL_ASAHI_GEM_MADVISE      = DRM_IOCTL_ASAHI(WR, GEM_MADVISE, gem_madvise),
	DRM_IOCTL_ASAHI_GEM_USERPTR      = DRM_IOCTL_ASAHI(WR, GEM_USERPTR, gem_userptr),
	DRM_IOCTL_ASAHI_GEM_RESIDENCY    = DRM_IOCTL_ASAHI(W, GEM_RESIDENCY, gem_residency),
};

#if defined(__cplusplus)
}
#endif

#endif /* _ASAHI_DRM_H_ */
