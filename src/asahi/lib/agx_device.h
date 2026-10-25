/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>
#ifdef HAVE_LIBDRM
#include <xf86drm.h>
#endif
#include "drm-uapi/asahi_drm.h"
#include "util/ralloc.h"
#include "util/cnd_monotonic.h"
#include "util/u_thread.h"
#include "util/simple_mtx.h"
#include "util/sparse_array.h"
#include "util/timespec.h"
#include "util/u_atomic.h"
#include "util/u_printf.h"
#include "util/vma.h"

#include "agx_bo.h"
#include "agx_chip.h"
#include "agx_pack.h"
#include "decode.h"
#include "layout.h"
#include "libagx_dgc.h"

#ifdef HAVE_LIBDRM
#include "vdrm.h"
#include "asahi_proto.h"
#else
struct vdrm_device;
struct asahi_ccmd_submit_res;
#endif

enum agx_dbg {
   AGX_DBG_TRACE = BITFIELD_BIT(0),
   AGX_DBG_BODUMP = BITFIELD_BIT(1),
   AGX_DBG_NO16 = BITFIELD_BIT(2),
   AGX_DBG_DIRTY = BITFIELD_BIT(3),
   AGX_DBG_PRECOMPILE = BITFIELD_BIT(4),
   AGX_DBG_PERF = BITFIELD_BIT(5),
   AGX_DBG_NOCOMPRESS = BITFIELD_BIT(6),
   AGX_DBG_NOCLUSTER = BITFIELD_BIT(7),
   AGX_DBG_SYNC = BITFIELD_BIT(8),
   AGX_DBG_STATS = BITFIELD_BIT(9),
   AGX_DBG_RESOURCE = BITFIELD_BIT(10),
   AGX_DBG_BATCH = BITFIELD_BIT(11),
   AGX_DBG_NOWC = BITFIELD_BIT(12),
   AGX_DBG_SYNCTVB = BITFIELD_BIT(13),
   AGX_DBG_SMALLTILE = BITFIELD_BIT(14),
   AGX_DBG_NOMSAA = BITFIELD_BIT(15),
   AGX_DBG_NOSHADOW = BITFIELD_BIT(16),
   AGX_DBG_BODUMPVERBOSE = BITFIELD_BIT(17),
   AGX_DBG_SCRATCH = BITFIELD_BIT(18),
   AGX_DBG_NOSOFT = BITFIELD_BIT(19),
   AGX_DBG_FEEDBACK = BITFIELD_BIT(20),
   AGX_DBG_1QUEUE = BITFIELD_BIT(21),
   AGX_DBG_NOMERGE = BITFIELD_BIT(22),
   /* Track BOs without serializing submissions, then decode only a failed
    * submit while its command stream and referenced allocations are live. */
   /* Decode every command stream at submit time without waiting for the GPU. */
   AGX_DBG_ASYNCTRACE = BITFIELD_BIT(24),
};

/* How many power-of-two levels in the BO cache do we want? 2^14 minimum chosen
 * as it is the page size that all allocations are rounded to
 */
#define MIN_BO_CACHE_BUCKET (14) /* 2^14 = 16KB */
#define MAX_BO_CACHE_BUCKET (22) /* 2^22 = 4MB */

/* Fencepost problem, hence the off-by-one */
#define NR_BO_CACHE_FLAG_BUCKETS 16
#define NR_BO_CACHE_BUCKETS \
   ((MAX_BO_CACHE_BUCKET - MIN_BO_CACHE_BUCKET + 1) * NR_BO_CACHE_FLAG_BUCKETS)

/* Forward decl only, do not pull in all of NIR */
struct nir_shader;

#define BARRIER_RENDER  (1 << DRM_ASAHI_SUBQUEUE_RENDER)
#define BARRIER_COMPUTE (1 << DRM_ASAHI_SUBQUEUE_COMPUTE)

struct agx_submit_virt {
   uint32_t extres_count;
   struct asahi_ccmd_submit_res *extres;
   uint32_t ring_idx;
};

typedef struct {
   struct agx_bo *(*bo_alloc)(struct agx_device *dev, size_t size, size_t align,
                              enum agx_bo_flags flags);
   int (*bo_bind)(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
                  uint32_t count);
   void (*bo_mmap)(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
                   size_t length);
   /* bo_mmap/bo_munmap may map one BO any number of times: each bo_mmap of a
    * copy of the agx_bo returns a new mapping and touches nothing but the
    * copy's _map/_map_size (agx_bo_map_owned()). */
   bool bo_mmap_independent;
   ssize_t (*get_params)(struct agx_device *dev, void *buf, size_t size);
   int (*submit)(struct agx_device *dev, struct drm_asahi_submit *submit,
                 struct agx_submit_virt *virt);
   int (*bo_bind_object)(struct agx_device *dev,
                         struct drm_asahi_gem_bind_object *bind);
   int (*bo_unbind_object)(struct agx_device *dev, uint32_t object_handle);

   /*
    * Purge advice for a VM-private BO (DRM_IOCTL_ASAHI_GEM_MADVISE). On
    * success *retained says whether the contents survived (false once the
    * kernel purged them). Returns -errno; -ENOTTY/-EINVAL from a kernel
    * without the ioctl disables the BO cache's use of it. NULL when the
    * backend has no equivalent.
    */
   int (*bo_madvise)(struct agx_device *dev, uint32_t handle, bool dontneed,
                     bool *retained);

   int (*simple_ioctl)(struct agx_device *dev, unsigned cmd, void *req);
   void (*bo_munmap)(struct agx_device *dev, struct agx_bo *bo);
   void (*bo_free)(struct agx_device *dev, uint32_t handle);

} agx_device_ops_t;

int agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
                size_t size_B, uint64_t offset_B, uint32_t flags);

int agx_bind_timestamps(struct agx_device *dev, struct agx_bo *bo,
                        uint32_t *handle);

/*
 * Linux/virtio-shared op bodies backing the three ops added just above.
 * Both agx_device_drm_ops (agx_device.c) and agx_virtio_device_ops
 * (agx_device_virtio.c) install the SAME functions here: generic DRM-core
 * behavior (GEM_CLOSE, plain munmap) and asahi_simple_ioctl's own existing
 * is_virtio dispatch don't differ between the two Linux backends -- exactly
 * why this code was unconditional (not backend-specific) before this ops
 * widening existed. Declared here (like agx_bo_bind/agx_bind_timestamps
 * above) so agx_device_virtio.c can reference the agx_device.c definitions.
 */
int asahi_simple_ioctl(struct agx_device *dev, unsigned cmd, void *req);
void agx_drm_bo_munmap(struct agx_device *dev, struct agx_bo *bo);
void agx_drm_bo_close(struct agx_device *dev, uint32_t handle);

struct agx_device {
   /* Optional allocator for linked executables: returns a referenced BO with
    * the allocation at *offset. Without it, each program gets its own BO. */
   struct agx_bo *(*code_alloc)(struct agx_device *dev, size_t size,
                                uint32_t *offset);

   uint32_t debug;

   /* Precompiled libagx binary table */
   const uint32_t **libagx_programs;

   char name[64];
   struct drm_asahi_params_global params;
   bool g15_legacy_wire;
   bool is_virtio;
   /* Native DRM 0.2 added the four render resource-specifier high dwords.
    * Stock 0.0 kernels do not consume an oversized render packet's tail. */
   bool render_rsrc_spec_hi;
   agx_device_ops_t ops;

   /* vdrm device */
   struct vdrm_device *vdrm;
   uint32_t next_blob_id;

   /* Device handle */
   int fd;

   /* VM handle */
   uint32_t vm_id;

   /* Global queue handle */
   uint32_t queue_id;
   /* AGX_DBG_1QUEUE: render SCS layouts the shared queue was created with. */
   uint64_t queue_layouts[2];

   /* VMA heaps */
   simple_mtx_t vma_lock;
   uint64_t shader_base;

   /* G17P only: the resident page bound at shader_base, so USC_EXEC_BASE_CP
    * names mapped memory the way it does in every m1n1 witness. */
   struct agx_bo *usc_base_bo;
   struct agx_bo *g17p_render_context_bo;
   struct util_vma_heap main_heap;
   struct util_vma_heap usc_heap;
   struct util_vma_heap replay_heap;
   /* Heap selection must stay stable when replay_heap.free_size reaches 0. */
   bool replay_heap_initialized;
   /* The capture/replay range, for validating a replayed address. */
   uint64_t replay_start;
   uint64_t replay_size;
   /* Native command-pool BO aliases with representable 32-bit context offsets.
    * Protected by vma_lock like the other heaps. */
   struct util_vma_heap g17p_vdm_alias_heap;
   uint64_t main_heap_base;
   uint64_t main_heap_size;
   uint64_t guard_size;

   /* To emulate sparse-resident buffers, we map buffers in both the bottom half
    * and top half of the address space. sparse_ro_offset controls the
    * partitioning. This is a power-of-two that &'s zero in bottom (read-write)
    * buffers but non-zero in top (read-only) shadow mappings.
    *
    * In other words, given an address X, we can check if it is in the top half
    * if (X & sparse_ro_offset) != 0.
    *
    * Given a bottom half address X, we can get the top half address
    * equivalently as (X + sparse_ro_offset) or (X | sparse_ro_offset).
    */
   uint64_t sparse_ro_offset;

   struct agx_bo *zero_bo, *scratch_bo;

   struct renderonly *ro;

   simple_mtx_t bo_map_lock;
   struct util_sparse_array bo_map;
   /* Exclusive upper bound for every loop over bo_map. */
   uint32_t max_handle;

   struct {
      simple_mtx_t lock;

      /* List containing all cached BOs sorted in LRU (Least Recently Used)
       * order so we can quickly evict BOs that are more than 1 second old.
       */
      struct list_head lru;

      /* Power-of-two size classes, subdivided by allocation-flag hash.
       * Exact flags are still checked when fetching; collisions only affect
       * lookup cost. Each bucket is a list of free agx_bo objects.
       */
      struct list_head buckets[NR_BO_CACHE_BUCKETS];

      /* Current size of the BO cache in bytes (sum of sizes of cached BOs) */
      size_t size;

      /* Optional frontend-wide accounting, shared by logical devices. */
      p_atomic_uint64_t *size_counter;

      /* Number of hits/misses for the BO cache */
      uint64_t hits, misses;

      /* Kernel purge advice for cached BOs: 0 untested, 1 works, -1 absent.
       * With advice the kernel can discard cached BOs under memory pressure;
       * without it they pin their pages until evicted. */
      int madvise;

      /* DRM shmem pages remain pinned while a BO is cached. Wake only while
       * the cache has entries, so an idle client can release stale BOs. */
      mtx_t idle_lock;
      struct u_cnd_monotonic idle_cond;
      thrd_t idle_thread;
      bool idle_started, idle_stop, idle_pending;

      /* First cached object whose purge advice still needs checking. */
      uint32_t next_unadvised;
   } bo_cache;

   struct agxdecode_ctx *agxdecode;

   /* Prepacked USC Sampler word to bind the txf sampler, used for
    * precompiled shaders on both drivers.
    */
   struct agx_usc_sampler_packed txf_sampler;

   /* Simplified device selection */
   enum agx_chip chip;

   struct {
      uint64_t num;
      uint64_t den;
   } user_timestamp_to_ns;

   struct u_printf_ctx printf;
   uint64_t printf_buffer_address;
};

static inline void
agx_bo_assert_range(ASSERTED struct agx_bo *bo, ASSERTED size_t offset,
                    ASSERTED size_t size)
{
   assert(bo != NULL && bo->dev != NULL);
   assert(offset <= bo->size && size <= bo->size - offset);
}

/* Make CPU writes to a BO range visible to the GPU. Linux DRM mappings are
 * coherent, so this only checks the range; the M5 sparse/physical paths call
 * it where a non-coherent winsys would need a flush. Returns 0 on success. */
static inline int
agx_bo_sync_for_device(struct agx_bo *bo, size_t offset, size_t size)
{
   agx_bo_assert_range(bo, offset, size);
   return 0;
}

/*
 * Determine if an address is in the read-only section. See the documentation
 * for sparse_ro_offset.
 */
static inline bool
agx_addr_is_ro(struct agx_device *dev, uint64_t addr)
{
   return (addr & dev->sparse_ro_offset);
}

/*
 * Convert a read-write address to its read-only shadow address. See the
 * documentation for sparse_ro_offset.
 */
/* Flags for filling a range with the scratch or zero page. When the kernel
 * offers DRM_ASAHI_FEATURE_SPARSE_TILES it fills the range from its own
 * physically contiguous tiles (zero for read-only, a per-VM sink for
 * writable binds), so each provider operation maps up to 64 pages instead of
 * one; the BO then only anchors the mapping.
 */
static inline uint32_t
agx_fill_bind_flags(const struct agx_device *dev)
{
   return DRM_ASAHI_BIND_SINGLE_PAGE |
          ((dev->params.features & DRM_ASAHI_FEATURE_SPARSE_TILES)
              ? DRM_ASAHI_BIND_SPARSE_TILE
              : 0);
}

static inline uint64_t
agx_rw_addr_to_ro(struct agx_device *dev, uint64_t addr)
{
   assert(!agx_addr_is_ro(dev, addr));
   return addr + dev->sparse_ro_offset;
}

/*
 * `length` bounds a PLACED mapping.  Zero means the whole BO, which is what
 * every ordinary map wants.
 *
 * A placed map may not be given the BO's size: agx_bo_cache_fetch() hands back
 * BOs up to twice the requested size (agx_bo.c), so a VkDeviceMemory of n bytes
 * can be backed by a BO of 2n, and mapping bo->size with MAP_FIXED at the
 * address the application reserved would land on whatever the application put
 * after its allocation.  VK_EXT_map_memory_placed requires exactly the memory
 * object's range; dEQP-VK.memory.map_placed.* reserves guard pages
 * immediately after it, fills them, and checks the pattern survives.
 */
/*
 * Bytes mapped at `bo->_map`. Backends that map the whole BO may leave
 * `_map_size` zero; everything that tears a mapping down must go through this
 * rather than assuming `bo->size`. See the field comment in agx_bo.h.
 */
static inline size_t
agx_bo_map_size(const struct agx_bo *bo)
{
   return bo->_map_size ?: bo->size;
}

static inline void *
agx_bo_map_placed(struct agx_bo *bo, void *fixed_addr, size_t length)
{
   /* The BO cache deliberately preserves CPU mappings.  A Vulkan allocation
    * may therefore inherit a mapping from an earlier allocation even though
    * the new VkDeviceMemory object has never been mapped.  A placed map must
    * move that stale process mapping to the application's reserved address;
    * returning it would violate VK_EXT_map_memory_placed.
    *
    * Keep ordinary mapping reuse unchanged.  Only the explicit placed-map
    * contract is allowed to relocate an existing BO mapping. */
   if (fixed_addr && bo->_map && bo->_map != fixed_addr) {
      bo->dev->ops.bo_munmap(bo->dev, bo);
      bo->_map = NULL;
      bo->_map_size = 0;
   }

   if (!bo->_map)
      bo->dev->ops.bo_mmap(bo->dev, bo, fixed_addr, fixed_addr ? length : 0);

   return bo->_map;
}

static inline void *
agx_bo_map(struct agx_bo *bo)
{
   return agx_bo_map_placed(bo, NULL, 0);
}

/*
 * A CPU mapping of `bo` owned by the caller rather than shared through
 * bo->_map: one VkDeviceMemory object's mapping of a BO that other memory
 * objects alias (an exported allocation and its imports resolve to one
 * agx_bo), which that object alone maps, unmaps or reserves. `length` bounds
 * a placed mapping as for agx_bo_map_placed(); *size receives the mapped
 * length. NULL when the backend cannot map a BO more than once.
 */
static inline void *
agx_bo_map_owned(struct agx_bo *bo, void *fixed_addr, size_t length,
                 size_t *size)
{
   if (!bo->dev->ops.bo_mmap_independent)
      return NULL;

   struct agx_bo copy = *bo;
   copy._map = NULL;
   copy._map_size = 0;
   bo->dev->ops.bo_mmap(bo->dev, &copy, fixed_addr, fixed_addr ? length : 0);
   *size = agx_bo_map_size(&copy);
   return copy._map;
}

static inline void
agx_bo_unmap_owned(struct agx_bo *bo, void *map, size_t size)
{
   struct agx_bo copy = *bo;
   copy._map = map;
   copy._map_size = size;
   bo->dev->ops.bo_munmap(bo->dev, &copy);
}

static inline bool
agx_has_soft_fault(struct agx_device *dev)
{
   return (dev->params.features & DRM_ASAHI_FEATURE_SOFT_FAULTS) &&
          !(dev->debug & AGX_DBG_NOSOFT);
}

/* Base of the G17P USC exec window.  MEASURED RULE: bit 40 of the program VA
 * must be set.  Do not lower this.  See the note in agx_open_device(). */
#define AGX_G17P_USC_WINDOW_BASE UINT64_C(0x10000000000)

/* Metal's HAL200 direct encoder binds fixed offsets below +0x80000.  The
 * kernel-owned PM metrics mapping begins exactly at +0x80000. */
#define AGX_G17P_RENDER_CONTEXT_BASE UINT64_C(0x1000000000)
#define AGX_G17P_RENDER_CONTEXT_SIZE UINT64_C(0x00080000)

static uint32_t
agx_usc_addr(struct agx_device *dev, uint64_t addr)
{
   assert(addr >= dev->shader_base);
   assert((addr - dev->shader_base) <= UINT32_MAX);

   return addr - dev->shader_base;
}

/* The USC exec window the kernel is told about at queue-create time.
 *
 * agx_create_command_queue() passes dev->shader_base as
 * drm_asahi_queue_create.usc_exec_base, and the kernel programs it into
 * USC_EXEC_BASE_{TA,ISP,CP} for every command on that queue.  The uapi
 * contract is a fixed 4 GiB carveout at that base (asahi_drm.h: "we expect
 * userspace to fix a 4GiB VA carveout for USC memory and pass its base address
 * here"), which is exactly the size dev->usc_heap is initialised with.
 *
 * Note the window BASE PAGE is deliberately not required to be mapped:
 * util_vma_heap defaults to alloc_high, so USC allocations descend from
 * shader_base + 4 GiB and the first page of the window is never handed out
 * until the heap is nearly exhausted.  Residency of `shader_base` itself is
 * therefore not a valid health check -- containment of each individual
 * exec pointer is.
 */
#define AGX_USC_WINDOW_SIZE UINT64_C(0x100000000)

static inline bool
agx_va_in_usc_window(const struct agx_device *dev, uint64_t addr)
{
   return addr >= dev->shader_base &&
          (addr - dev->shader_base) < AGX_USC_WINDOW_SIZE;
}

/* The caller retains its fd on failure; common close owns it on success. */
bool agx_open_device(void *memctx, struct agx_device *dev);

void agx_close_device(struct agx_device *dev);

static inline struct agx_bo *
agx_lookup_bo(struct agx_device *dev, uint32_t handle)
{
   return util_sparse_array_get(&dev->bo_map, handle);
}

static inline void
agx_include_bo_handle(struct agx_device *dev, uint32_t handle)
{
   /* All backends reserve an all-ones handle as allocation failure. */
   assert(handle != UINT32_MAX);
   const uint32_t exclusive = handle + 1;
   if (exclusive > dev->max_handle)
      dev->max_handle = exclusive;
}

/* Returns UINT32_MAX on allocation failure, preserving the backend errno. */
uint32_t agx_create_command_queue(struct agx_device *dev,
                                  enum drm_asahi_priority priority);
uint32_t agx_create_command_queue_with_render_scs(struct agx_device *dev,
   enum drm_asahi_priority priority, const uint64_t layouts[2]);
int agx_destroy_command_queue(struct agx_device *dev, uint32_t queue_id);

int agx_import_sync_file(struct agx_device *dev, struct agx_bo *bo, int fd);
int agx_export_sync_file(struct agx_device *dev, struct agx_bo *bo);

void agx_debug_fault(struct agx_device *dev, uint64_t addr);

uint64_t agx_get_gpu_timestamp(struct agx_device *dev);

static inline uint64_t
agx_gpu_timestamp_to_ns(struct agx_device *dev, uint64_t gpu_timestamp)
{
   return (gpu_timestamp * dev->user_timestamp_to_ns.num) /
          dev->user_timestamp_to_ns.den;
}

void agx_get_device_uuid(const struct agx_device *dev, void *uuid);
void agx_get_driver_uuid(void *uuid);
unsigned agx_get_num_cores(const struct agx_device *dev);

struct agx_device_key agx_gather_device_key(struct agx_device *dev);

/* Native AGX3 recording: Ioto programs and HAL200-format records. */
static inline bool
agx_native_graphics(const struct agx_device *dev)
{
   return agx_chip_native_agx3(dev->chip);
}

/* T6040 uses the G17P architecture profile, while capability rollout may
 * still be product-specific. Device admission has already checked USC/HAL. */
static inline bool
agx_is_m4_pro(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P && dev->params.chip_id == 0x6040 &&
          dev->params.gpu_generation == 16 && dev->params.gpu_variant == 'S';
}

struct agx_va *agx_va_alloc(struct agx_device *dev, uint64_t size_B,
                            uint64_t align_B, enum agx_va_flags flags,
                            uint64_t fixed_va);
void agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind);

/* Whether a fixed range, with its guard, lies inside the capture/replay range. */
bool agx_va_capture_valid(const struct agx_device *dev, uint64_t addr,
                          uint64_t size_B, uint64_t align_B);

static inline struct drm_asahi_cmd_header
agx_cmd_header(bool compute, uint16_t barrier_vdm, uint16_t barrier_cdm)
{
   return (struct drm_asahi_cmd_header){
      .cmd_type = compute ? DRM_ASAHI_CMD_COMPUTE : DRM_ASAHI_CMD_RENDER,
      .size = compute ? sizeof(struct drm_asahi_cmd_compute)
                      : sizeof(struct drm_asahi_cmd_render),
      .vdm_barrier = barrier_vdm,
      .cdm_barrier = barrier_cdm,
   };
}

/* Size of the render command passed to the kernel. drm_asahi_cmd_render grew
 * the high words of the background/EOT resource specifiers for the native
 * AGX3 path, whose USC window sits at 2^40, and kernels that predate them
 * reject the longer packet with EINVAL. Every other chip keeps the original
 * layout, as Honeykrisp does in hk_drm_render_command_size().
 */
static inline uint16_t
agx_render_command_size(const struct agx_device *dev)
{
   return agx_chip_native_agx3(dev->chip)
             ? sizeof(struct drm_asahi_cmd_render)
             : offsetof(struct drm_asahi_cmd_render, bg_eot_rsrc_spec_hi);
}
