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
#include "hk_native_sampler.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_acceleration_structure.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct hk_shader_binary_cache_entry;
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
   /* Scratch-filled sparse VA (M5: bounded by sparseAddressSpaceSize). */
   uint64_t sparse_scratch_B;
   /* Live buffers and image planes with a GPU virtual address of their own
    * (sparse binding, capture/replay buffers), atomic. Their memory is also
    * mapped at another address, see hk_cdm_alias_scope(). */
   uint32_t va_alias_resources;
   /* Whether anything this device enabled can map memory at a second GPU
    * virtual address, decided at creation (hk_device_may_alias_memory()).
    * M5 uses it in place of va_alias_resources, see hk_cdm_alias_scope(). */
   bool may_alias_memory;
   /* One bit per main-heap page: only unbound sparse buffer pages are set. */
   struct agx_bo *physical_sparse_map;
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
   struct agx_bo *gen1_as_query_bank;
   struct agx_bo *gen1_as_window_roots;
   struct util_dynarray gen1_as_windows; /* struct hk_gen1_as_window */
   uint32_t gen1_as_window_hint;
   bool gen1_rq_live_as_table;
   bool gen1_as_chunk_cache_lock_ready;
   simple_mtx_t gen1_as_chunk_cache_lock;
   struct list_head gen1_as_free_chunks[2];
   uint32_t gen1_as_free_chunk_count[2];
   struct agx_bo *gen1_rq_code, *gen1_rq_ift;
   /* Immutable generated user image, shared by all query launches. */
   struct agx_bo *gen1_rq_state;
   struct agx_va *gen1_rq_code_alias;
   struct agx_bo *gen1_vertex_directory;
   struct agx_va *gen1_vertex_directory_va;
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
   simple_mtx_t cmd_pending_lock;
   /* Queues with pending submissions (hk_queue::cmd_link). */
   struct list_head cmd_pending_queues;
   /* Submissions that completed with an error: kept until destruction. */
   struct list_head cmd_pending_failed;
   /* Completion syncobjs of retired submissions, for reuse. */
   uint32_t cmd_syncobjs[HK_CMD_SYNCOBJ_POOL];
   unsigned cmd_syncobj_count;

   /* Native G17P indirect compute grid helper.  A compact indirect helper
    * record carries no dimensions and no resource table, so the only way a
    * runtime grid can reach the main record is for this program to publish
    * the six-u32 geometry block.  Compiled once, from NIR, through the same
    * path as any other Honeykrisp compute shader, so it uses the driver's own
    * proven root/uniform ABI rather than a replayed one.  NULL on every other
    * chip, and NULL on G17P if the backend refused it -- in which case
    * indirect dispatch stays refused instead of launching a program that does
    * not exist. */
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

   /* Heap used for GPU-side memory allocation for geometry/tessellation.
    *
    * Control streams accessing the heap must be serialized. This is not
    * expected to be a legitimate problem. If it is, we can rework later.
    */
   struct agx_bo *heap;
   util_once_flag heap_init_once;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   uint32_t perftest;

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
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

/* G17P and M5 GPU caches are tagged by virtual address: a strong CDM flush
 * makes a write visible only to reads through the same mapping. While memory
 * may be read through a second address, an application dependency between
 * compute or transfer work needs the memory-scope flush
 * (hk_cdm_visibility_flush()). Devices without such memory keep the ordinary
 * strong flush.
 *
 * On M5 this is decided once, when the device is created
 * (hk_device_may_alias_memory()): a device that enabled no feature or
 * extension through which memory can get a second GPU address (sparse
 * binding, capture/replay, fd imports, writable host imports) never has such
 * memory, and every other device flushes with memory scope at every
 * dependency, also in command buffers recorded before such memory existed
 * or that reach it only while they run.
 *
 * On G17P the state is read when a barrier is recorded: live sparse and
 * capture/replay resources. A resource the command buffer references exists
 * by then, and it counts from creation, so later sparse binds are covered.
 * Not covered on G17P: a resource created after the command buffer was
 * recorded and reached through a path recording did not see, that is a
 * device address the command buffer reads indirectly, or a descriptor set
 * bound at recording and updated after it (descriptorBinding*UpdateAfterBind),
 * and memory imports.
 *
 * The flush goes into the open compute stream, and only when that stream
 * already holds work (hk_cdm_visibility_flush_needed()). It writes back every
 * dirty line, so the writes it covers include those of earlier streams and
 * earlier command buffers. A barrier recorded with no open compute stream
 * holding work records no flush. This includes the first barrier of a
 * command buffer and barriers after the stream was closed, for example by a
 * timestamp or query copy; the compute and transfer writes before it were
 * flushed with memory scope at the end of their stream
 * (hk_cdm_end_stream_flush()). When the open stream holds work unrelated to
 * the source, the flush goes at its end, but a graphics destination is still
 * ordered only after the source's stream, so it can start before the flush.
 * Graphics-only sources also do not request memory scope: attachment or
 * shader writes through a sparse image followed by transfer or compute reads
 * through a plain alias receive the ordinary post-graphics strong flush. That
 * case has not been hardware-tested. */
static inline bool
hk_cdm_alias_scope(const struct hk_device *dev)
{
   if (agx_chip_g17_m5(dev->dev.chip))
      return dev->may_alias_memory;

   return dev->dev.chip == AGX_CHIP_G17P &&
          p_atomic_read(&dev->va_alias_resources) != 0;
}

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

/* Allocate size bytes of executable memory. Returns a reference to the chunk
 * BO holding it, with the allocation at *offset; release it with
 * agx_bo_unreference(). */
struct agx_bo *hk_code_alloc(struct hk_device *dev, size_t size,
                             uint32_t *offset);
