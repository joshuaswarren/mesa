/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/u_atomic.h"
#include "util/u_queue.h"

#include "asahi/lib/agx_device.h"
#include "util/rwlock.h"

#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "agx_bg_eot.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "decode.h"
#include "vk_cmd_queue.h"
#include "vk_dispatch_table.h"

#include "hk_private.h"

#include "hk_descriptor_table.h"
#include "hk_gpu_waits.h"
#include "hk_host_alias.h"
#include "hk_native_sampler.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_acceleration_structure.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct hk_shader_binary_cache_entry;
struct vk_command_buffer;
struct vk_pipeline_cache;

#define HK_SHADER_BINARY_CACHE_MAX 16
#define HK_CMD_SYNCOBJ_POOL 64

typedef void (*hk_internal_builder_t)(struct nir_builder *b, const void *key);

struct hk_internal_key {
   hk_internal_builder_t builder;
   size_t key_size;

   /* A compilation input that changes the emitted binary but is not part of
    * the caller's key bytes, so it has to be hashed alongside them: `stop`
    * decides whether the part ends the shader or falls through to a footer
    * the fast linker appends.
    *
    * internal_key_hash() hashes sizeof(struct hk_internal_key) + key_size
    * bytes, which is wider than builder + key_size + the key data once a
    * member sits between key_size and key[]. Every producer of this struct
    * therefore zeroes the whole allocation, not just the header, so that no
    * padding byte reaches the hash uninitialized.
    */
   bool stop;

   uint8_t key[];
};

struct hk_internal_shaders {
   simple_mtx_t lock;
   struct hash_table *ht;
};

struct hk_rc_sampler {
   struct agx_sampler_packed key;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;

   /* A VK_EXT_descriptor_buffer capture/replay sampler uses it: the index
    * goes back to the replay pool (hk_descriptor_table_remove_replay()). */
   bool replay;
};

struct hk_sampler_heap {
   simple_mtx_t lock;

   struct hk_descriptor_table table;

   /* Map of agx_sampler_packed to hk_rc_sampler */
   struct hash_table *ht;
};

/* Executable memory for shader binaries and launch setups, bump-allocated
 * from chunks. A chunk lives while anything allocated from it does. */
struct hk_code_heap {
   simple_mtx_t lock;
   struct agx_bo *chunk;
   uint32_t offset;
};

struct hk_gen1_sparse_table {
   struct agx_va *va;
   struct util_dynarray pages;
   uint32_t entry_bytes;
   enum agx_bo_flags bo_flags;
   uint32_t bind_flags;
   const char *label;
};

/* G17P AS IDs index this registry directly, with a free list in vacant slots.
 * base is the traversal base published in the AS table: the structure's own
 * address when that is 16 KiB-aligned, otherwise base page window_base of
 * window `window` (UINT32_MAX for none). */
struct hk_gen1_as_slot {
   struct vk_acceleration_structure *as;
   uint32_t next_free;
   uint32_t window;
   uint32_t window_base;
   uint64_t base;
};

/* 256 consecutive 16 KiB base pages; see hk_gen1_as_window_assign(). */
struct hk_gen1_as_window {
   struct agx_va *va;
   struct agx_bo *trampolines;
   uint32_t used[8];
   uint32_t live;
};
struct hk_frame_anatomy;
struct hk_compute_kick;

struct hk_device {
   struct vk_device vk;
   struct agx_device dev;
   /* Same-queue binary semaphore waits ordered on the GPU. */
   struct hk_gpu_waits gpu_waits;
   /* Scratch-filled sparse VA (M5: bounded by sparseAddressSpaceSize). */
   uint64_t sparse_scratch_B;
   /* Live resources through which memory has or may have a second GPU virtual
    * address: sparse/capture-replay buffers and image planes, capture/replay
    * memory mappings (hk_device_va_alias_get()), fd/dma-buf imports, and host
    * imports that may alias (hk_host_import_track()). Atomic; see
    * hk_cdm_alias_scope(). */
   uint32_t va_alias_resources;
   /* Whether anything this device enabled can map memory at a second GPU
    * virtual address at any time, decided at creation
    * (hk_device_may_alias_memory()). M5 uses it in place of
    * va_alias_resources, see hk_cdm_alias_scope(). */
   bool may_alias_memory;
   /* The kernel runs a render's fragment work after the previous render's
    * on a queue, partial renders included
    * (DRM_ASAHI_FEATURE_ORDERED_PARTIAL_RENDERS): a render that only loads
    * whole tiles an earlier render stores waits for it in its fragment half
    * alone (hk_sync.h, tile hazards). */
   bool load_hazard_frag;
   /* M5 with host_alias.dynamic: live resources through which memory has or
    * may have a second GPU address (hk_device_alias_get()), atomic:
    * va_alias_resources, fd and dma-buf imports, and host imports that may
    * alias. hk_cdm_alias_scope() asks for memory scope while it is not 0. */
   uint32_t alias_live;

   /* Monotonic counter incremented atomically whenever a buffer is first bound
    * to an hk_device_memory that already has an image binding. Checked at
    * submit time so command buffers recorded before a post-record buffer bind
    * conservatively upgrade their aliased attachment renders to
    * HK_SYNC_GFX_MEM. */
   uint32_t buffer_alias_mem_gen;
   struct {
      /* M5: the device enabled a feature through which memory can get a
       * second address, so the scope follows alias_live instead of
       * may_alias_memory, which was left false
       * (hk_device_may_alias_memory()). */
      bool dynamic;
      /* ASAHI_M5_HOST_ALIAS_TRACE=1. */
      bool trace;
      /* The memory-scoped strong flush, written over every patch site
       * (struct hk_alias_site). */
      uint32_t flush_word;
      /* The device's live host imports. */
      simple_mtx_t lock;
      struct hk_host_imports imports;
      /* Executable command buffers with unwritten patch sites
       * (hk_cmd_buffer::alias_link), patched when alias_live leaves 0. */
      simple_mtx_t cmd_lock;
      struct list_head cmd_buffers;
      /* Submit packing copies the streams of the command buffers it
       * submits, holding this shared (queue_submit()), and the first
       * write of a command buffer's sites that a submission could see takes
       * it exclusively (taken after cmd_lock). */
      struct u_rwlock patch_lock;
   } host_alias;
   /* One bit per main-heap page: only unbound sparse buffer pages are set. */
   struct agx_bo *physical_sparse_map;
   /* The buffers with a residency map (hk_buffer::sparse_map), for texel
    * buffer descriptors that name only an address (VK_EXT_descriptor_buffer,
    * hk_sparse_texel_word_at()). */
   struct {
      simple_mtx_t lock;
      struct list_head list;
      /* Entries in list, read without the lock */
      uint32_t count;
   } sparse_buffers;
   /* The zeroed page every unbound sparse page maps, read-write (the
    * read-only shadow maps the zero page). It is no write sink of the
    * driver's, unlike the scratch page, so it stays zero for strict
    * residency. shortcut: an application store to an unbound page lands
    * here and later reads of unbound pages see it; guard stores with the
    * residency map if that ever matters. */
   struct agx_bo *sparse_fill;
   struct agxdecode_ctx *decode_ctx;
   /* Opt-in frame anatomy profiler (hk_frame_anatomy.h); NULL unless the
    * HK_FRAME_ANATOMY environment variable is set. */
   struct hk_frame_anatomy *frame_anatomy;
   /* Compute kick (hk_compute_kick.h); NULL unless G17C with
    * hk_compute_kick_ms set. */
   struct hk_compute_kick *compute_kick;

   struct hk_descriptor_table occlusion_queries;
   struct hk_sampler_heap samplers;

   /* Native sampled-image indices are stable for an image-view lifetime.
    * G17P shaders dereference this table on the GPU, including descriptors
    * updated after command recording. Other generations keep inline state.
    */
   struct hk_descriptor_table sampled_images;

   /* VK_EXT_descriptor_buffer texel-buffer textures (hk_buffer_view.c),
    * NULL unless the device enables descriptorBuffer. */
   struct hk_texel_windows *texel_windows;

   struct vk_meta_device meta;
   /* G17P PBE buffer fills recorded so far, atomic. Their sweep direction
    * alternates, see hk_g17p_fill_buffer_pbe(). */
   uint32_t g17p_fills;

   /* Parameters for Mesa's common acceleration-structure builder. */
   struct vk_acceleration_structure_build_args accel_struct_args;
   bool g17p_native_as;
   bool gen1_native_rq;
   bool gen1_native_fragment_rq;
   bool gen1_native_graphics_rq;
   bool gen1_rq_instances, gen1_rq_candidates, gen1_as_lock_ready;
   simple_mtx_t gen1_as_lock;
   struct util_dynarray gen1_ases; /* struct hk_gen1_as_slot */
   uint32_t gen1_as_free_head;
   struct hash_table *gen1_as_ids;
   struct hk_gen1_sparse_table gen1_as_table_pages;
   struct hk_gen1_sparse_table gen1_as_metadata_pages;
   struct hk_gen1_sparse_table gen1_as_address_pages;
   struct hk_gen1_sparse_table gen1_vertex_directory_pages;
   struct agx_bo *gen1_as_metadata;
   struct agx_bo *gen1_as_addresses;
   struct agx_bo *gen1_as_window_roots;
   struct util_dynarray gen1_as_windows; /* struct hk_gen1_as_window */
   uint32_t gen1_as_window_hint;
   struct agx_bo *gen1_rq_code, *gen1_rq_ift;
   /* Immutable generated user image, shared by all query launches. */
   struct agx_bo *gen1_rq_state;
   struct agx_va *gen1_rq_code_alias;
   struct agx_bo *gen1_vertex_directory;
   struct hash_table *gen1_vertex_tables;
   uint64_t *gen1_rq_maps;
   unsigned gen1_rq_bound_maps;
   uint8_t *gen1_rq_user_image;
   uint8_t gen1_rq_uniforms[320];
   struct {
      simple_mtx_t lock;
      bool lock_ready;
      struct hk_queue *queue;
      VkCommandPool pool;
   } host_as;
   bool m5_native_as;
   simple_mtx_t m5_as_lock;
   struct util_dynarray m5_ases;
   /* Stable native AS-ID -> address table. VA is reserved once; backing
    * pages are added as IDs are allocated, so recorded/pending commands
    * also see acceleration structures created after recording. */
   struct agx_va *m5_as_table_va;
   struct util_dynarray m5_as_table_pages;
   /* Only AS objects created before their ordinary storage buffer is bound.
    * Binding publishes their final addresses without a submit-time walk. */
   struct util_dynarray m5_pending_as_ids;
   uint32_t m5_pending_as_count;
   struct agx_bg_eot_cache bg_eot;
   /* frg-codegen v4 BG/EOT device cache (hk_cmd_draw.c). */
   struct hk_g17p_bg_eot_dev_cache *g17p_bg_eot_dev_cache;

   /* VK_KHR_ray_tracing_pipeline shader group handles.  A handle is one
    * device-unique index, so a handle captured from a pipeline library keeps
    * naming the same group in every pipeline that imports the library.  It
    * starts at one: zero is the value an unwritten shader binding table
    * record has, and no group may collide with it. */
   uint32_t rt_group_handle_alloc;

   /* Own group handles of each RT pipeline library content key created on
    * this device, so a library loaded from a pipeline cache takes the handles
    * an identical library got here rather than ones from another device.
    * Created on first use (hk_ray_tracing_pipeline.c). */
   struct {
      simple_mtx_t lock;
      struct hash_table *table;
   } rt_library_handles;

   /* Device-lifetime templates for precompiled G17P meta compute programs.
    * The libagx programs and their executable BOs have the same lifetime. */
   struct {
      void *templates[LIBAGX_NUM_PROGRAMS];
   } g17p_compute_program_cache;

   struct hk_code_heap code_heap;

   /* Kernel-maintained copy of this VM's sticky status
    * (DRM_ASAHI_FEATURE_VM_STATUS_MIRROR). NULL: query per check instead. */
   struct {
      struct agx_bo *bo;
      int32_t *word;
   } vm_status_mirror;

   /* virtio: dev->dev.async_errors when the device was created. */
   uint32_t async_errors_base;

   struct {
      /* Immutable for the device's lifetime: the native texel-fetch sampler
       * table. Its address is published to the GPU as a sampler table base by
       * every G17P background reload program, so nothing that any subsystem
       * writes at draw time may share this buffer object. */
      struct agx_bo *bo;

      /* G17P texture and sampler heaps are allocated at full capacity during
       * device creation, so their VAs never move.  Keep the pair in immutable
       * device-owned storage: uploading it per draw makes its address part of
       * every indexed-texture Ioto key and defeats program reuse. */
      struct agx_bo *indexed_heap_pointers_bo;
      uint64_t indexed_heap_pointers;

      /* The geometry/tessellation heap descriptor. hk_init_heap() writes it
       * from the CPU on the first tessellated or geometry-shaded draw, and the
       * libagx bump allocator then atomically updates poly_heap::bottom from
       * the GPU on every such draw. Kept in its own buffer object so that
       * mutation cannot reach the sampler table above. */
      struct agx_bo *heap_bo;
      uint64_t heap;
   } rodata;

   uint32_t heap_next_slot;

   struct hk_internal_shaders prolog_epilog;
   struct hk_internal_shaders kernels;
   struct hk_api_shader *null_fs;

   /* A small device-owned cache of authenticated immutable shader payloads.
    * Every API create still gets a distinct vk_shader wrapper and distinct
    * mutable link caches. */
   struct {
      simple_mtx_t lock;
      struct hk_shader_binary_cache_entry
         *entries[HK_SHADER_BINARY_CACHE_MAX];
   } shader_binary_cache;

   /* Device-wide command memory. Command buffers from different pools
    * suballocate immutable spans: compact VDM backing from the same finite
    * G17P aperture, and the first span of each uploader. A slab that refused
    * an allocation waits on full until one of its spans is freed. */
   struct {
      simple_mtx_t lock;
      struct list_head slabs[HK_CMD_SLAB_COUNT];
      struct list_head full[HK_CMD_SLAB_COUNT];
      /* The next slab's size: a process that records little pins little. */
      uint32_t next_size[HK_CMD_SLAB_COUNT];
   } vdm;

   bool cmd_pending_ready;
   /* Number of queues currently linked into cmd_pending_queues. */
   int cmd_pending_active;
   simple_mtx_t cmd_pending_lock;
   /* Queues with pending submissions (hk_queue::cmd_link). */
   struct list_head cmd_pending_queues;
   /* Submissions that completed with an error: kept until destruction. */
   struct list_head cmd_pending_failed;
   /* Recording memory quarantined after device loss. */
   struct list_head cmd_lost_bos;
   struct list_head cmd_lost_spans;
   struct list_head cmd_lost_large_bos;
   /* Completion syncobjs of retired submissions, for reuse. */
   uint32_t cmd_syncobjs[HK_CMD_SYNCOBJ_POOL];
   unsigned cmd_syncobj_count;

   /* In-flight kernel host writes to bound timestamp buffers (timestamp
    * queries and stamped occlusion availability slots), used to emit
    * DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS on compute commands whose query
    * accesses are disjoint from every pending host write. */
   struct {
      bool ready;
      bool oom;
      simple_mtx_t lock;
      struct util_dynarray pending;
      /* Open-addressed index into pending[] keyed on (handle, offset_B, queue):
       * 0 is empty, value is pending index + 1. Power-of-two size when non-empty. */
      uint32_t *slot_index;
      size_t slot_cap;
      size_t prune_threshold;
   } query_stamps;

   /* Device-wide slab allocator for VkEvent status words, bound once per slab
    * as a kernel timestamp buffer on HAL200/M5 so vkCmdSetEvent2 can ride on
    * the producing stream's retirement timestamp without a CDM store. */
   struct {
      bool ready;
      simple_mtx_t lock;
      struct list_head slabs;
   } events;

   /* Native G17P indirect compute grid helper.  A compact indirect helper
    * record carries no dimensions and no resource table, so the only way a
    * runtime grid can reach the main record is for this program to publish
    * the six-u32 geometry block.  Compiled once, from NIR, through the same
    * path as any other Borealis compute shader, so it uses the driver's own
    * proven root/uniform ABI rather than a replayed one.  NULL on chips
    * without native indirect compute; on those that have it, device creation
    * fails if it does not compile. */
   struct hk_api_shader *g17p_grid_helper;

   /* The same contract for an indirect-local grid, which an earlier kernel
    * already wrote as a launch geometry: this helper copies it. */
   struct hk_api_shader *g17p_geometry_copy_helper;

   /* The same contract again, for an indirect-global grid whose arguments
    * live in a sparse residency buffer: identical except that it consults the
    * buffer's residency bitmap and publishes a zero grid for arguments in a
    * non-resident page.  Kept as its own program so the ordinary indirect
    * dispatch path records exactly the code it recorded before. */
   struct hk_api_shader *g17p_grid_helper_sparse;

   /* Indirected for common secondary emulation */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Heaps used for GPU-side memory allocation for geometry/tessellation and
    * software index robustness. Each ring slot lazily allocates a full 128 MiB
    * buffer on first use so per-pass capacity is never reduced while distinct
    * slots execute without global serialization.
    */
   struct agx_bo *heap_slots[HK_HEAP_SLOTS];
   simple_mtx_t heap_lock;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   /* Set once the device creates a sparse buffer or image; the CDM barrier
    * then keeps the full word (hk_cdm_cache_flush). */
   uint32_t sparse_used;

   uint32_t perftest;
   /* drirc hk_submit_batch_size: at most this many commands per kernel
    * submission (0: the kernel and batching limits alone). */
   unsigned submit_batch_size;
   /* The AGX2 CDM_BARRIER flag word emitted at a tracked dependency on
    * this chip (agx2_cdm_barrier_default_word()): the reduced word on the
    * dies it was measured on, the full word elsewhere. */
   uint32_t cdm_barrier_default_mask;

   struct {
      struct u_rwlock lock;
      struct util_dynarray list;
      struct util_dynarray counts;
   } external_bos;
};

void hk_g17p_compute_program_cache_finish(struct hk_device *dev);
bool hk_g17p_compute_program_cache_prewarm(struct hk_device *dev,
                                           enum libagx_program program,
                                           size_t data_size);

VK_DEFINE_HANDLE_CASTS(hk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

enum hk_perftest {
   HK_PERF_NOTESS = BITFIELD_BIT(0),
   HK_PERF_NOBORDER = BITFIELD_BIT(1),
   HK_PERF_NOBARRIER = BITFIELD_BIT(2),
   HK_PERF_NOBATCH = BITFIELD_BIT(3),
   HK_PERF_NOROBUST = BITFIELD_BIT(4),
   /* Keep only the preceding launch's weak ordering barrier at pure
    * compute/transfer pipeline barriers in the same CDM stream instead of
    * upgrading it to a strong cache flush, when memory is unaliased and no
    * host or pending graphics transfer dependency is involved. */
   HK_PERF_WEAKBARRIER = BITFIELD_BIT(5),
   /* Force the full CDM barrier word on every chip (perftest only). */
   HK_PERF_FULLCDMBARRIER = BITFIELD_BIT(11),
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

/* G17P and M5 GPU caches are tagged by virtual address, so when memory may be
 * mapped at more than one GPU VA an application compute/transfer barrier needs
 * the memory-scoped strong flush (hk_cdm_visibility_flush()).
 *
 * On M5, devices whose barriers always take memory scope
 * (agx_cdm_ordering_memory_scope()) take it on every barrier and in every
 * launch's ordering record. Other M5 devices that enable aliasing-capable
 * features (sparse, BDA, capture/replay, fd/dma-buf, writable host imports)
 * track active aliasing resources in alias_live under host_alias.dynamic (or
 * force memory scope via may_alias_memory / ASAHI_M5_HOST_ALIAS_STATIC=1).
 * Barriers recorded while alias_live == 0 record patch sites (struct
 * hk_alias_site) that are upgraded in place when alias_live leaves 0 or at
 * submission; unpatched paths (graphics dependencies, event-using command
 * buffers) and stream-end flushes keep memory scope unconditionally while
 * dynamic mode is active.
 *
 * On G17P, memory scope is selected at barrier recording time whenever
 * va_alias_resources != 0 (live sparse, capture/replay, fd, or aliasing host
 * imports). The flush is emitted into an open compute stream holding work;
 * earlier closed streams already flushed at stream end.
 *
 * Not covered on G17P: resources or imports created after command buffer
 * recording and reached via device address or UpdateAfterBind descriptors;
 * graphics-only sources writing through a sparse image read via a plain alias
 * receive only the ordinary post-graphics strong flush (untested).
 */
static inline bool
hk_cdm_alias_scope(const struct hk_device *dev)
{
   if (agx_chip_g17_m5(dev->dev.chip))
      return agx_cdm_ordering_memory_scope(&dev->dev) ||
             dev->may_alias_memory || p_atomic_read(&dev->alias_live);

   return dev->dev.chip == AGX_CHIP_G17P &&
          p_atomic_read(&dev->va_alias_resources) != 0;
}

/* Whether the CDM strong flush has a memory-scoped form, which memory
 * reached through a second address needs (hk_cdm_alias_scope()). Keyed on
 * the chip table's AGX_RECORD_CDM_BARRIER_FLUSH_MEMORY capability (G17P
 * and M5); G16 has only the plain strong flush and G17A's caches are not
 * known (hk_cdm_caches_pending()). */
static inline bool
hk_cdm_memory_scope_form(const struct hk_device *dev)
{
   return agx_chip_has_record(dev->dev.chip,
                              AGX_RECORD_CDM_BARRIER_FLUSH_MEMORY);
}

/* Whether a compute command whose stream ends with the strong memory-scoped
 * flush says so (DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION), so the kernel can
 * let another queue's kick wait for its completion in firmware. */
static inline bool
hk_cdm_visible_completion(const struct hk_device *dev)
{
   return agx_chip_g17_m5(dev->dev.chip) &&
          (dev->dev.params.features &
           DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION);
}

/* Whether it is still unknown if the GPU's caches are tagged by virtual
 * address, as on G17P and M5, so that a plain strong flush may not make
 * memory visible through a second address. G17A sits between those two and
 * its caches have not been checked: it is pending until they are. G16 is
 * known to be untagged. */
static inline bool
hk_cdm_caches_pending(const struct hk_device *dev)
{
   return dev->dev.chip == AGX_CHIP_G17A;
}

/* As hk_cdm_alias_scope(), for an application dependency recorded without a
 * patch site (struct hk_alias_site): with host_alias.dynamic always, since
 * nothing would give the record memory scope once a resource may alias. */
static inline bool
hk_cdm_alias_scope_unpatched(const struct hk_device *dev)
{
   return hk_cdm_alias_scope(dev) || dev->host_alias.dynamic;
}

/* Writes the patch sites of every executable command buffer that has
 * unwritten ones (host_alias.cmd_buffers), when the first resource that may
 * alias appears. Returns how many command buffers it wrote. */
unsigned hk_device_patch_alias_sites(struct hk_device *dev);

/* What made alias_live leave 0 (ASAHI_M5_HOST_ALIAS_TRACE). */
enum hk_alias_cause {
   HK_ALIAS_CAUSE_IMPORT,
   HK_ALIAS_CAUSE_SPARSE,
   HK_ALIAS_CAUSE_CAPTURE_REPLAY,
   HK_ALIAS_CAUSE_FD,
};

/* alias_live left 0: patch every live command buffer
 * (hk_device_patch_alias_sites()). */
void hk_device_alias_began(struct hk_device *dev, enum hk_alias_cause cause);

/* M5 with host_alias.dynamic: n more live resources through which memory
 * has or may have a second GPU address (alias_live). Returns whether this
 * call took the count from 0; the caller then runs hk_device_alias_began()
 * before the call that creates the resources returns, and without holding
 * host_alias.lock.
 *
 * The count never wraps, so exactly the add that takes it from 0 sees n:
 * every resource's hk_device_alias_put() follows its own add (a resource is
 * destroyed only after the call that created it returned), and host
 * imports add and subtract under host_alias.lock together with the set
 * that says how many of them alias (hk_host_import_track()). A subtraction
 * that ran ahead of its add could cancel another resource's add back to 0
 * and skip both its patch and the scope. A caller that adds to a count
 * another caller is still patching for is covered by the check at
 * submission (hk_queue_submit()). */
static inline bool
hk_device_alias_add(struct hk_device *dev, uint32_t n)
{
   if (!dev->host_alias.dynamic || n == 0)
      return false;
   return p_atomic_add_return(&dev->alias_live, n) == n;
}

/* hk_device_alias_add() and, when the count left 0, the patch of every live
 * command buffer (hk_device_alias_began()), before the call that creates
 * the resources returns. For callers that hold no lock. */
static inline void
hk_device_alias_get(struct hk_device *dev, uint32_t n,
                    enum hk_alias_cause cause)
{
   if (hk_device_alias_add(dev, n))
      hk_device_alias_began(dev, cause);
}

/* Undoes the add of n resources, after it (hk_device_alias_add()). */
static inline void
hk_device_alias_put(struct hk_device *dev, uint32_t n)
{
   if (dev->host_alias.dynamic && n)
      p_atomic_add(&dev->alias_live, -(int32_t)n);
}

/* A buffer, image plane or memory object that maps memory at a GPU address
 * of its own (va_alias_resources), from its creation to its destruction. */
static inline void
hk_device_va_alias_get(struct hk_device *dev, enum hk_alias_cause cause)
{
   p_atomic_inc(&dev->va_alias_resources);
   hk_device_alias_get(dev, 1, cause);
}

static inline void
hk_device_va_alias_put(struct hk_device *dev)
{
   p_atomic_dec(&dev->va_alias_resources);
   hk_device_alias_put(dev, 1);
}

/* At submission, while an import may alias: writes the patch sites of the
 * submitted command buffers, before submit packing copies their streams. */
void hk_submit_patch_alias_sites(struct hk_device *dev,
                                 struct vk_command_buffer *const *cmds,
                                 unsigned count);

static inline struct hk_physical_device *
hk_device_physical(struct hk_device *dev)
{
   return (struct hk_physical_device *)dev->vk.physical;
}

VkResult hk_device_init_meta(struct hk_device *dev);
void hk_device_finish_meta(struct hk_device *dev);

/* Force the geometry/tessellation heap to exist now rather than inside the
 * first draw that needs it.  Idempotent and safe to call from any thread. */
VkResult hk_device_ensure_geometry_heap(struct hk_device *dev);

VkResult hk_sampler_heap_add(struct hk_device *dev,
                             const struct hk_native_sampler_descriptor *desc,
                             struct hk_rc_sampler **out);

VkResult
hk_sampler_heap_add_capture(struct hk_device *dev,
                            const struct hk_native_sampler_descriptor *desc,
                            struct hk_rc_sampler **out);
VkResult hk_sampler_heap_add_at(struct hk_device *dev,
                                const struct hk_native_sampler_descriptor *desc,
                                uint32_t index, struct hk_rc_sampler **out);
void hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc);

static inline struct agx_scratch *
hk_device_scratch_locked(struct hk_device *dev, mesa_shader_stage stage)
{
   simple_mtx_assert_locked(&dev->scratch.lock);

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      return &dev->scratch.fs;
   case MESA_SHADER_VERTEX:
      return &dev->scratch.vs;
   default:
      return &dev->scratch.cs;
   }
}

static inline void
hk_device_alloc_scratch(struct hk_device *dev, mesa_shader_stage stage,
                        unsigned size)
{
   simple_mtx_lock(&dev->scratch.lock);
   agx_scratch_alloc(hk_device_scratch_locked(dev, stage), size, 0);
   simple_mtx_unlock(&dev->scratch.lock);
}

/* frg-codegen v4: device-lifetime cache of single-attachment G17P BG/EOT
 * launch programs (hk_cmd_draw.c). */
void hk_g17p_bg_eot_dev_cache_init(struct hk_device *dev);
void hk_g17p_bg_eot_dev_cache_finish(struct hk_device *dev);

/* Alignment of every hk_code_alloc() offset. */
#define HK_CODE_ALIGNMENT 256

/* Allocate size bytes of executable memory. Returns a reference to the chunk
 * BO holding it, with the allocation at *offset; release it with
 * agx_bo_unreference(). */
struct agx_bo *hk_code_alloc(struct hk_device *dev, size_t size,
                             uint32_t *offset);
