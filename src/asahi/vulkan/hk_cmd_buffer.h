/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/macros.h"

#include "util/list.h"
#include "agx_abi.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "agx_uvs.h"
#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"

#include "asahi/compiler/agx3_ioto_uniform_program.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_ppp_hal300_graphics_core.h"
#include "asahi/lib/agx_ppp_viewport_payload_core.h"
#include "asahi/lib/agx_vdm_hal200_bind_tail_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"

#include "hk_private.h"
#include "hk_barrier.h"
#include "hk_conditional_render_core.h"
#include "hk_shader.h"
#include "hk_sync.h"

#include "hk_cmd_pool.h"
#include "hk_cs_root_size.h"
#include "hk_descriptor_set.h"
#include "hk_drm_render_core.h"
#include "hk_ioto_arena_core.h"
#include "hk_pipeline_statistics_core.h"

#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "vk_command_buffer.h"

#include <stdio.h>

struct set;
struct hk_buffer;
struct hk_cmd_bo;
struct hk_cmd_pool;
struct hk_image_view;
struct hk_push_descriptor_set;
struct hk_ray_tracing_pipeline;
struct hk_api_shader;
struct hk_shader;
struct hk_linked_shader;
struct agx_usc_builder;
struct vk_shader;

/* A complete but unpublished HAL200 state update.  The referenced VDM state
 * is command-buffer-owned.  The scissor/depth records stay inline until the
 * draw chooses its live stream, so preparation cannot perturb stream-local
 * indices or make firmware-visible bytes reachable early. */
struct hk_g17p_prepared_bind_tail {
   struct agx_vdm_hal200_bind_tail tail;
   uint8_t scissors[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                   [AGX_SCISSOR_LENGTH];
   uint8_t depth_bias[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                    [AGX_DEPTH_BIAS_LENGTH];
   uint64_t dynamic_generation;
   uint64_t descriptor_generation;
   uint64_t vertex_buffer_generation;
   uint64_t root;
   /* The fragment shader variant the tail launches and the uniform-DMA set
    * base its program was built with (hk_g17p_dma_set_base()). */
   const struct hk_shader *fragment_shader;
   uint64_t fragment_dma_set_base;
   struct hk_image_view *render_color0;
   struct hk_image_view *render_depth;
   struct hk_image_view *render_stencil;
   VkRect2D render_area;
   VkFormat render_depth_format;
   uint32_t render_width;
   uint32_t render_height;
   uint32_t render_layer_count;
   uint32_t render_view_mask;
   uint8_t render_color_count;
   uint8_t render_samples;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];
   struct hk_linked_shader *linked_vs;
   struct hk_linked_shader *linked_fs;
   uint64_t varyings;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   uint32_t occlusion_index;
   uint8_t topology;
   uint8_t object_type;
   uint8_t provoking;
   uint8_t occlusion_mode;
   uint8_t viewport_count;
   bool generate_primitive_id;
   bool force_never_in_shader;
   bool in_meta;
   bool valid;
};

/** Root descriptor table. */
struct hk_root_descriptor_table {
   /* Address of this descriptor itself. Must be first for reflection. */
   uint64_t root_desc_addr;

   /* Descriptor set base addresses. Must follow root_desc_addr to match our
    * push layout.
    */
   uint64_t sets[HK_MAX_SETS];

   union {
      struct {
         uint32_t view_index;
         uint32_t ppp_multisamplectl;

         /* Vertex input state, indexed by compacted attribute slot. The
          * bases live at the end of this arm (attrib_base below): the clamp
          * and stride tables at 32 entries exactly fill the 256 bytes the
          * three 16-entry tables used to, which keeps every later field --
          * notably the pinned pipeline_stats offsets -- where it was.
          */
         uint32_t attrib_clamps[AGX_MAX_VBUFS];
         uint32_t attrib_strides[AGX_MAX_VBUFS];

         /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
         uint64_t vertex_outputs;

         /* Address of vertex param buffer if geom/tess is used, else 0 */
         uint64_t vertex_params;

         /* Address of tessellation param buffer if tessellation used, else 0 */
         uint64_t tess_params;

         /* Address of geometry param buffer if GS is used, else 0 */
         uint64_t geometry_params;

         /* Pipeline statistics queries. This is a base address with flags. */
         uint64_t pipeline_stats;
         VkQueryPipelineStatisticFlags pipeline_stats_flags;

         float blend_constant[4];
         uint16_t no_epilog_discard;
         uint16_t _pad1;
         uint16_t api_sample_mask;
         uint16_t _pad2;
         uint16_t force_never_in_shader;
         uint16_t _pad3;
         uint16_t provoking;
         uint16_t _pad4;

         /* True if there is an API geometry shader. If false, there may still
          * be a geometry shader in use (notably for transform feedback) but it
          * should not contribute to pipeline statistics.
          */
         uint16_t api_gs;
         uint16_t _pad5;

         uint16_t rasterization_stream;
         uint16_t _pad6;

         /* Alpha-to-coverage enable for fragment shaders that write the
          * sample mask; they apply it before their own coverage update.
          * Sysval fields must sit at 4-byte offsets (root table loads). */
         uint16_t alpha_to_coverage;
         uint16_t _pad7;

         /* Mapping from varying slots written by the last vertex stage to UVS
          * indices. This mapping must be compatible with the fragment shader.
          */
         uint8_t uvs_index[VARYING_SLOT_MAX];

         /* GPU addresses of the immutable NIR constant-data blobs of the
          * bound vertex and fragment shaders, or zero if they have none. */
         uint64_t vs_constant_data_addr;
         uint64_t fs_constant_data_addr;

         /* Vertex attribute base addresses, indexed by compacted slot. */
         uint64_t attrib_base[AGX_MAX_VBUFS];
      } draw;
      struct {
         /* GPU address of the immutable NIR constant-data blob for the
          * currently bound compute shader. */
         uint64_t constant_data_addr;
         uint64_t group_count_addr;
         uint32_t base_group[3];

         /* Native G17P indirect dispatch only.  The grid helper reads the
          * threadgroup counts at group_count_addr, multiplies them by
          * grid_helper_local_size, and publishes the six-u32 launch geometry
          * at grid_helper_geometry_addr for the main record to consume.  For
          * an indirect-local grid, group_count_addr instead holds a launch
          * geometry, which the geometry copy helper publishes as it is.  Both
          * are zero for every direct dispatch and for every other chip. */
         uint64_t grid_helper_geometry_addr;

         /* Native G17P indirect dispatch whose argument buffer is a sparse
          * residency buffer: the base of that buffer's residency bitmap (a
          * struct hk_sparse_buffer_map_header followed by one bit per page),
          * and zero otherwise.  When it is set, group_count_addr names the
          * PRIMARY mapping rather than the read-only shadow, and the grid
          * helper reads the bitmap to decide whether the arguments are
          * resident, publishing a zero grid when they are not.  See
          * hk_CmdDispatchIndirect() for why the shadow cannot be read
          * directly here. */
         uint64_t grid_helper_sparse_map_addr;

         uint32_t grid_helper_local_size[3];
         uint32_t _pad_grid_helper;

         /* VK_KHR_ray_tracing_pipeline: device address of the
          * VkTraceRaysIndirectCommand2KHR record the megakernel reads its
          * shader binding tables and launch size from.  Zero for every
          * ordinary compute dispatch; the ray-tracing megakernel is the only
          * shader that loads it. */
         uint64_t trace_rays_addr;
      } cs;
   };

   /* Client push constants */
   uint8_t push[HK_MAX_PUSH_SIZE];

   /* Dynamic buffer bindings */
   struct hk_buffer_address dynamic_buffers[HK_MAX_DYNAMIC_BUFFERS];

   /* Start index in dynamic_buffers where each set starts */
   uint8_t set_dynamic_buffer_start[HK_MAX_SETS];

   /* Process-local on G17P; loaded through the root instead of cached code. */
   uint64_t printf_buffer_address;
   uint32_t mesh_draw_id;
   uint64_t mesh_grid_addr;
   uint64_t mesh_prims_query;
   uint64_t sampler_heap_address;
   uint64_t physical_sparse_map;
   uint64_t physical_sparse_base;
   uint64_t physical_sparse_size;

   /* Upper four sample positions on M5. It follows every agx3b root field,
    * so the draw arm, and with it every field after the union, keeps the
    * agx3b layout. */
   uint32_t ppp_multisamplectl_hi;

   /* M5 task draws whose grid, count or predicate live in GPU memory:
    * address of the word libagx_m5_task_dispatch writes (nonzero = execute);
    * zero for every other draw. Appended after every agx3b field. */
   uint64_t mesh_task_flag;
};

/* The graphics arm is the larger of the two, so everything after the union --
 * push constants, dynamic buffers, the printf address -- keeps its offset when
 * a compute field is added. Growing the compute arm past it would move all of
 * them at once, which no single change should do silently. */
static_assert(sizeof(((struct hk_root_descriptor_table *)0)->cs) <=
                 sizeof(((struct hk_root_descriptor_table *)0)->draw),
              "compute root fields must not grow the root descriptor table");

/* helper macro for computing root descriptor byte offsets */
#define hk_root_descriptor_offset(member)                                      \
   offsetof(struct hk_root_descriptor_table, member)

/* Counts only these CPU upload paths, enabled by ASAHI_MESA_DEBUG=perf. */
struct hk_upload_stats {
   uint64_t draws, descriptor_binds, descriptor_noops;
   uint64_t root_uploads, root_bytes, root_reuses;
   uint64_t draw_param_uploads, draw_param_bytes, draw_param_reuses;
   uint64_t draw_id_uploads, draw_id_bytes, draw_id_reuses;
   uint64_t constant_uploads, constant_bytes, constant_reuses;
};

#define hk_upload_stat(cmd, field, value)                                      \
   do {                                                                       \
      if (unlikely(hk_cmd_buffer_device(cmd)->dev.debug & AGX_DBG_PERF))         \
         (cmd)->state.upload_stats.field += (value);                            \
   } while (0)

/* Root bytes the fragment stage reads only through these classes of access,
 * recorded per shader in hk_shader_info::agx3_root_reads. */
enum hk_root_reads {
   /* The root's own address: the shader can then reach any root byte. */
   HK_ROOT_READS_ADDRESS = BITFIELD_BIT(0),
   /* Vertex attribute bases, clamps and strides, and the vertex shader's
    * constant data. */
   HK_ROOT_READS_VERTEX_INPUT = BITFIELD_BIT(1),
   HK_ROOT_READS_DYNAMIC_BUFFERS = BITFIELD_BIT(2),
};

/* Every root byte a draw may republish without the fragment stage observing
 * it unless the fragment shader reads that class (hk_m5_fragment_root()), as
 * X(from, to, class) spans in ascending order. The vertex-input spans hold
 * just their fields: the clamp and stride tables are adjacent and followed by
 * vertex_outputs, and attrib_base ends the graphics arm. */
#define HK_ROOT_FRAGMENT_HIDDEN_SPANS(X)                                       \
   X(root_desc_addr, sets, HK_ROOT_READS_ADDRESS)                              \
   X(draw.attrib_clamps, draw.vertex_outputs, HK_ROOT_READS_VERTEX_INPUT)      \
   X(draw.vs_constant_data_addr, draw.fs_constant_data_addr,                   \
     HK_ROOT_READS_VERTEX_INPUT)                                               \
   X(draw.attrib_base, push, HK_ROOT_READS_VERTEX_INPUT)                       \
   X(dynamic_buffers, set_dynamic_buffer_start, HK_ROOT_READS_DYNAMIC_BUFFERS)

/* The HK_ROOT_READS_* classes a read of root bytes [start_B, end_B) falls
 * in. */
static inline uint8_t
hk_root_reads_of(uint64_t start_B, uint64_t end_B)
{
   uint8_t reads = 0;
#define HK_ROOT_CLASSIFY(from, to, class)                                      \
   if (start_B < hk_root_descriptor_offset(to) &&                              \
       end_B > hk_root_descriptor_offset(from))                                \
      reads |= (class);
   HK_ROOT_FRAGMENT_HIDDEN_SPANS(HK_ROOT_CLASSIFY)
#undef HK_ROOT_CLASSIFY
   return reads;
}

/* Whether two roots agree on every byte outside the hidden spans. */
static inline bool
hk_root_matches_for_fragment(const struct hk_root_descriptor_table *a,
                             const struct hk_root_descriptor_table *b)
{
   size_t at = 0;
#define HK_ROOT_COMPARE_BEFORE(from, to, class)                                \
   if (memcmp((const uint8_t *)a + at, (const uint8_t *)b + at,                \
              hk_root_descriptor_offset(from) - at))                           \
      return false;                                                            \
   at = hk_root_descriptor_offset(to);
   HK_ROOT_FRAGMENT_HIDDEN_SPANS(HK_ROOT_COMPARE_BEFORE)
#undef HK_ROOT_COMPARE_BEFORE
   return !memcmp((const uint8_t *)a + at, (const uint8_t *)b + at,
                  sizeof(*a) - at);
}

/* Build the same visible-byte sequence as hk_root_matches_for_fragment().
 * Keeping this derived from the shared span list includes future root fields
 * and padding without duplicating the root ABI. */
#define HK_ROOT_FRAGMENT_SUBTRACT(from, to, class)                             \
   -(hk_root_descriptor_offset(to) - hk_root_descriptor_offset(from))
enum {
   HK_ROOT_FRAGMENT_VISIBLE_SIZE = sizeof(struct hk_root_descriptor_table)
      HK_ROOT_FRAGMENT_HIDDEN_SPANS(HK_ROOT_FRAGMENT_SUBTRACT),
};
#undef HK_ROOT_FRAGMENT_SUBTRACT

struct hk_root_fragment_snapshot {
   uint8_t bytes[HK_ROOT_FRAGMENT_VISIBLE_SIZE];
};

static inline void
hk_root_fragment_snapshot_store(struct hk_root_fragment_snapshot *snapshot,
                                const struct hk_root_descriptor_table *root)
{
   size_t at = 0, out = 0;
#define HK_ROOT_FRAGMENT_COPY(from, to, class)                                 \
   do {                                                                        \
      const size_t size = hk_root_descriptor_offset(from) - at;                \
      memcpy(snapshot->bytes + out, (const uint8_t *)root + at, size);         \
      out += size;                                                             \
      at = hk_root_descriptor_offset(to);                                      \
   } while (0);
   HK_ROOT_FRAGMENT_HIDDEN_SPANS(HK_ROOT_FRAGMENT_COPY)
#undef HK_ROOT_FRAGMENT_COPY
   memcpy(snapshot->bytes + out, (const uint8_t *)root + at,
          sizeof(*root) - at);
   assert(out + sizeof(*root) - at == sizeof(snapshot->bytes));
}

static_assert(hk_root_descriptor_offset(draw.attrib_strides) ==
                    hk_root_descriptor_offset(draw.attrib_clamps) +
                       sizeof(((struct hk_root_descriptor_table *)0)
                                 ->draw.attrib_clamps) &&
                 hk_root_descriptor_offset(draw.vertex_outputs) ==
                    hk_root_descriptor_offset(draw.attrib_strides) +
                       sizeof(((struct hk_root_descriptor_table *)0)
                                 ->draw.attrib_strides),
              "only the clamp and stride tables precede vertex_outputs");
static_assert(hk_root_descriptor_offset(draw.fs_constant_data_addr) ==
                 hk_root_descriptor_offset(draw.vs_constant_data_addr) +
                    sizeof(uint64_t),
              "the fragment constant data follows the vertex constant data");
static_assert(hk_root_descriptor_offset(push) ==
                 hk_root_descriptor_offset(draw.attrib_base) +
                    sizeof(((struct hk_root_descriptor_table *)0)
                              ->draw.attrib_base),
              "attrib_base ends the graphics arm");
static_assert(hk_root_descriptor_offset(set_dynamic_buffer_start) ==
                 hk_root_descriptor_offset(dynamic_buffers) +
                    sizeof(((struct hk_root_descriptor_table *)0)
                              ->dynamic_buffers),
              "the dynamic buffer table is followed by its set starts");

struct hk_descriptor_state {
   bool root_dirty;
   uint64_t generation;
   struct hk_root_descriptor_table root;
   /* Immutable command-buffer storage; reset with descriptor state. Only the
    * uploaded head and tail are compared, excluding unused dynamic slots. */
   struct agx_ptr uploaded_root;
   uint32_t uploaded_root_size;
   uint64_t uploaded_root_generation;
   uint8_t uploaded_dynamic_buffer_end;

   /* One past the highest root.dynamic_buffers entry any bind wrote. Later
    * entries are unread by every bound layout and stay out of uploads. */
   uint8_t dynamic_buffer_end;

   uint32_t set_sizes[HK_MAX_SETS];
   struct hk_descriptor_set *sets[HK_MAX_SETS];
   uint32_t sets_dirty;

   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   uint32_t push_dirty;
   uint32_t push_active;
};

struct hk_attachment {
   VkRenderingAttachmentFlagBitsKHR flags;

   VkFormat vk_format;
   struct hk_image_view *iview;

   /* The API load and store operations and the layout the pass was begun
    * with. */
   VkAttachmentLoadOp load_op;
   VkAttachmentStoreOp store_op;
   VkImageLayout layout;

   VkResolveModeFlagBits resolve_mode;
   struct hk_image_view *resolve_iview;

   bool clear;
   uint32_t clear_colour[4];
};

struct hk_bg_eot {
   uint64_t usc;
   uint32_t primary_storage_B;
   uint32_t staging_storage_B;
   union {
      struct agx_counts_packed counts;
      uint64_t rsrc_spec;
   };
   /* Frame anatomy profiler (hk_frame_anatomy.h): per-RT op summary of the
    * program. Written only while HK_FRAME_ANATOMY is set; otherwise zero and
    * never read. Not part of any hardware or kernel structure. */
   uint8_t anatomy_ops[HK_MAX_RTS];
};

/* BG/EOT programs are self-addressed and may reference command-buffer pool
 * uploads, so reuse them only inside the command buffer which owns both. One
 * exact render-pass key owns its main/partial BG/EOT variants; hash collisions
 * remain harmless misses. */
#define HK_G17P_BG_EOT_CACHE_ENTRIES 16
#define HK_G17P_BG_EOT_CACHE_KEY_MAX 768
struct hk_g17p_bg_eot_dev_entry;

struct hk_g17p_bg_eot_cache_entry {
   uint8_t key[HK_G17P_BG_EOT_CACHE_KEY_MAX];
   struct hk_bg_eot program[4];
   /* frg-codegen v4: the device-lifetime entry this pass fills or reuses,
    * or NULL. */
   struct hk_g17p_bg_eot_dev_entry *dev_entry;
   uint64_t hash;
   uint16_t key_size;
   uint8_t valid_variants;
   uint8_t direct_color_resolve_variants;
   bool valid;
};

struct hk_render_registers {
   uint32_t width, height, layers;
   uint32_t zls_width, zls_height;
   uint32_t isp_bgobjdepth;
   uint32_t isp_bgobjvals;
   struct agx_zls_control_packed zls_control, zls_control_partial;
   uint32_t depth_dimensions;
   bool process_empty_tiles;
   enum u_tristate dbias_is_int;

   struct {
      uint32_t dimensions;
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } depth;

   struct {
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } stencil;

   struct {
      struct hk_bg_eot main;
      struct hk_bg_eot partial;
   } bg;

   struct {
      struct hk_bg_eot main;
      struct hk_bg_eot partial;
   } eot;

   /* Frame anatomy profiler only: bit i set when colour attachment i's image
    * layout is compressed. Zero unless HK_FRAME_ANATOMY is set. */
   uint8_t color_compressed_mask;
};

struct hk_rendering_state {
   VkRenderingFlagBits flags;

   /* May have written storage memory/XFB anywhere in this API pass, including
    * before a physical render split. Used only for conservative barriers. */
   bool barrier_shader_writes;

   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;

   /* The main G17P EOT writes the single-sample resolve target directly. */
   bool g17p_direct_color_resolve;

   /* Exact admission result for the current physical tilebuffer layout.
    * The layout is immutable after BeginRendering except for the one
    * attachmentless late-sample transition, which refreshes this value. */
   bool g17p_rt0_layout_supported;

   uint32_t color_att_count;
   struct hk_attachment color_att[HK_MAX_RTS];
   struct hk_attachment depth_att;
   struct hk_attachment stencil_att;

   struct agx_tilebuffer_layout tilebuffer;
   struct hk_render_registers cr;

   /* G17P precise depth store: the ZLS words for the
    * stream that ends the pass, when it is also the first stream (only) or
    * a later one (last). Depth honours storeOp there; stencil keeps the
    * forced store. zls_first_cs is the stream BeginRendering created. */
   bool zls_precise;
   struct agx_zls_control_packed zls_control_only, zls_control_last;
   const void *zls_first_cs;

   /* Track depth and stencil independently: an untouched aspect can be
    * removed from every stream even when the other aspect is cleared or
    * used. NONE/NONE aspects must retain their memory contents; eliding
    * untouched LOAD/STORE aspects also avoids redundant traffic.
    */
   VkImageAspectFlags g17p_zls_elide_mask;
   VkImageAspectFlags g17p_zls_touched;

   /* Of the touched aspects, those a draw or clear may have changed. A pass
    * that only tests an elidable aspect leaves its memory contents as they
    * were, so it keeps the load and drops the store
    * (hk_g17p_zls_elide_store()).
    */
   VkImageAspectFlags g17p_zls_written;

   /* M5: an attachment of the pass can alias another resource (aliased,
    * sparse or linear image), so its stores count as memory writes for
    * dependency tracking (HK_SYNC_GFX_MEM). */
   bool m5_attachments_reach_memory;

   /* AGX3 dead depth store: the "only" ZLS word with the depth store
    * dropped, and whether the pass processes empty tiles without that store,
    * for a later pass that makes the store dead (hk_take_pending_z_store()).
    * Valid when no_z_store_ok. */
   bool no_z_store_ok;
   bool process_empty_tiles_without_z;
   struct agx_zls_control_packed zls_control_only_no_z_store;

   /* G17P: colour attachments a draw or clear of the pass may write, and
    * whether a fragment shader reads the tilebuffer. See
    * hk_g17p_elide_untouched_color(). */
   uint8_t color_written;
   bool color_read;

   /* G17P: the pass's first draw was seen, and the colour attachments it
    * overwrote at every sample before anything could read them. See
    * hk_g17p_note_first_draw(). */
   bool first_draw_seen;
   uint8_t color_covered;
};

struct hk_index_buffer_state {
   struct hk_addr_range buffer;
   enum agx_index_size size;
};

/* Dirty tracking bits for state not tracked by vk_dynamic_graphics_state or
 * shaders_dirty.
 */
enum hk_dirty {
   HK_DIRTY_VB = BITFIELD_BIT(0),
   HK_DIRTY_OCCLUSION = BITFIELD_BIT(1),
   HK_DIRTY_PROVOKING = BITFIELD_BIT(2),
   HK_DIRTY_VARYINGS = BITFIELD_BIT(3),
};

/* Where a G17P graphics uniform range's source address comes from. */
enum hk_g17p_range_source {
   /* Fixed for the shader and device (heap pointers, promoted constants). */
   HK_G17P_RANGE_SOURCE_CONSTANT = 0,
   /* The graphics root plus a fixed offset. */
   HK_G17P_RANGE_SOURCE_ROOT = 1,
   HK_G17P_RANGE_SOURCE_DRAW_PARAMS = 2,
   HK_G17P_RANGE_SOURCE_DRAW_ID = 3,
   /* A bound descriptor set's address plus a fixed offset, or the zero page
    * when the set is not bound (uniform DMA spans). */
   HK_G17P_RANGE_SOURCE_SET = 4,
   /* A whole uniform DMA buffer span: the buffer whose descriptor the root
    * (dynamic buffers) or a static descriptor set holds at a fixed byte, plus
    * a fixed offset, while that buffer covers the span. */
   HK_G17P_RANGE_SOURCE_BUFFER = 5,
   /* Uploaded while collecting the ranges (combined-sampler pointers,
    * spilled render-target descriptors), or a buffer span the buffer does
    * not wholly cover: not re-derivable. */
   HK_G17P_RANGE_SOURCE_UPLOADED,
};

struct hk_g17p_gfx_loader;

/* How a stage's uniform ranges derive from the draw state, recorded from a
 * collection by hk_g17p_graphics_uniform_ranges(), and the loader they were
 * published with. A draw with the same shader, linked executable, imageblock,
 * range-selecting state (hk_g17p_launch_flags()) and, for a fragment shader,
 * admission inputs derives its ranges from it. Every BO the loader names was
 * retained when it was recorded. */
#define HK_G17P_LAUNCH_SLOTS 8
#define HK_G17P_LAUNCH_ROOT_SET 0xff

struct hk_g17p_launch {
   const struct hk_shader *s;
   const struct hk_linked_shader *linked;
   const struct hk_g17p_gfx_loader *loader;
   const struct hk_shader *hw_vs;
   uint16_t tile_size_px;
   uint8_t sample_size_B;
   uint8_t nr_samples;
   uint8_t topology;
   uint8_t provoking;
   uint8_t flags;
   uint8_t count;
   bool valid;
   uint8_t source[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   /* Descriptor set of a SET or BUFFER range; HK_G17P_LAUNCH_ROOT_SET for a
    * BUFFER range whose descriptor is in the root. */
   uint8_t set[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t start_halfs[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t size_halfs[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   /* Byte of a BUFFER range's descriptor in its root or set. */
   uint16_t descriptor_B[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   /* Root/set/buffer offset, or the address of a constant range. */
   uint64_t offset[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
};

struct hk_g17p_graphics_ioto_cache_entry {
   uint16_t m5_range_count;
   uint16_t m5_tile_size_px;
   uint8_t m5_sample_size_B;
   uint8_t m5_nr_samples;
   /* The loader of the last draw, the program published from it and the
    * range addresses written into it (hk_g17p_publish_gfx_loader()). */
   const struct hk_g17p_gfx_loader *loader;
   const struct hk_linked_shader *loader_linked;
   uint64_t loader_program;
   struct agx3_ioto_uniform_range
      loader_ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];

   /* Inputs of the last admitted program (hk_upload_usc_words_internal()). */
   struct {
      const struct hk_shader *s;
      const struct hk_linked_shader *linked;
      const struct hk_shader *hw_vs;
      bool spills;
      uint8_t topology;
      uint8_t provoking;
   } admit;
};

/* The inputs of a G17P VertexProgramVariant state update other than the
 * pipeline address (hk_agx3_publish_vertex_state()). */
struct hk_g17p_vdm_update_key {
   const struct hk_shader *hw_vs;
   uint32_t selector;
   uint8_t draw_path;
   uint8_t provoking;
   bool rasterization_enabled;
   bool primitive_id;
   bool ordinary;
   bool selector_valid;
};

struct hk_graphics_state {
   /* A bound graphics program writes external memory. */
   bool shaders_write_memory;

   struct hk_rendering_state render;
   struct hk_descriptor_state descriptors;

   enum hk_dirty dirty;
   /* CmdBindVertexBuffers is not vk_dynamic_graphics_state and therefore does
    * not advance its generation.  Prepared G17P roots snapshot this separate
    * identity so a post-BeginRendering bind cannot publish stale attrib VAs. */
   uint64_t vertex_buffer_generation;
   /* The vertex_buffer_generation that descriptors.root.draw.attrib_* were
    * last derived from (hk_flush_dynamic_state_slow()'s vertex-buffer
    * block).  A prepared bind tail records this, not the live generation:
    * a tail rebuilt outside a full flush (vkCmdBindIndexBuffer) must not
    * vouch for a root whose attrib VAs predate the latest vertex binding. */
   uint64_t root_vertex_buffer_generation;

   uint64_t root;
   /* CPU copy of the root uploaded at root (hk_cmd_buffer_upload_root()). */
   const struct hk_root_descriptor_table *root_cpu;
   /* The root the fragment program was last published with, and its CPU
    * copy (hk_m5_fragment_root()). */
   uint64_t fs_root;
   const struct hk_root_descriptor_table *fs_root_cpu;

   /* A successful equality check between the immutable root snapshots at
    * fs_root and this address. Cleared whenever fs_root is replaced. */
   uint64_t fs_root_checked_root;

   /* CPU-cached copies of exactly the fragment-visible published bytes. The
    * pointer keys name the immutable GPU-mapped copies, never their contents. */
   struct hk_root_fragment_snapshot root_fragment_snapshot;
   struct hk_root_fragment_snapshot fs_root_fragment_snapshot;
   const struct hk_root_descriptor_table *root_fragment_snapshot_cpu;
   const struct hk_root_descriptor_table *fs_root_fragment_snapshot_cpu;

   uint64_t draw_params;
   uint64_t draw_id_ptr;
   uint32_t g17p_draw_params_value[2];
   uint32_t g17p_draw_id_value;
   bool g17p_draw_params_host_valid;
   bool g17p_draw_id_host_valid;

   struct hk_g17p_prepared_bind_tail g17p_bind_tail;
   /* The last complete build of g17p_bind_tail, whose other fields still
    * describe its inputs. A rebuild whose only changed inputs are the root,
    * descriptor or vertex-buffer identity (a descriptor push or vertex-buffer
    * bind) reads exactly the same state for every record except the fragment
    * record's stage program, which is recomputed from fs_fields; see
    * hk_g17p_build_native_bind_tail(). */
   struct {
      struct agx_ppp_hal200_fragment_fields fs_fields;
      /* The state block of the last build or memo rebuild: CPU copy, its
       * compact address and size. */
      const uint8_t *state_cpu;
      uint32_t state_compact;
      uint32_t state_size;
      const struct hk_shader *fs;
      const struct hk_shader *hw_vs;
      bool valid;
   } g17p_bind_tail_memo;

   /* Per vertex-attribute slot: the last API format seen and the pipe format
    * and interchange block size hk_flush_dynamic_state_slow() derives from
    * it, which are pure functions of the format. */
   struct {
      VkFormat vk_format;
      enum pipe_format format;
      unsigned interchange_align;
   } vb_format_memo[AGX_MAX_VBUFS];

   /* The last VertexProgramVariant state update this command buffer encoded,
    * keyed by every encoder input except the pipeline address; see
    * hk_agx3_publish_vertex_state(). */
   struct {
      struct hk_g17p_vdm_update_key key;
      struct agx_vdm_state_update encoded;
      bool valid;
   } g17p_vdm_update_memo;

   /* The last M5 HAL300 graphics core agx_ppp_hal300_graphics_core_pack()
    * packed, a pure function of its fields (zero-initialised, so they compare
    * bytewise); see hk_flush_ppp_state(). */
   struct {
      struct agx_ppp_hal300_graphics_fields fields;
      struct agx_ppp_hal300_graphics_core state;
      bool valid;
   } m5_ppp_pack_memo;

   uint32_t shaders_dirty;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];

   /* Stages bound to a shader compiled without custom-border emulation that
    * has an emulating twin to switch to once the device needs it. */
   uint32_t border_free_stages;

   /* Vertex buffers */
   struct hk_addr_range vb[AGX_MAX_VBUFS];

   /* Transform feedback buffers */
   struct hk_addr_range xfb[4];

   /* Is transform feedback enabled? */
   bool xfb_enabled;

   /* Internal transform feedback offset vec4.
    *
    * TODO: Strictly could be global.
    */
   uint64_t xfb_offsets;

   /* Pointer to the GPU memory backing active transform feedback queries,
    * per-stream. Zero if no query is bound.
    */
   uint64_t xfb_query[4];
   /* Active VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT reports, per stream. */
   uint64_t prims_generated_query[4];

   struct hk_index_buffer_state index;
   enum agx_primitive topology;
   enum agx_object_type object_type;

   /* Provoking vertex 0, 1, or 2. Usually 0 or 2 for FIRST/LAST. 1 can only be
    * set for tri fans.
    */
   uint8_t provoking;

   struct {
      enum agx_visibility_mode mode;

      /* If enabled, index of the current occlusion query in the occlusion heap.
       * There can only be one active at a time (hardware contraint).
       */
      uint16_t index;

      /* On M5, an empty query needs only its logical availability word.
       * Assign its physical counter when the first draw needs it. */
      struct hk_query_pool *pending_pool;
      uint32_t pending_query;

      /* Recording order (hk_sync_stream::seq) of the render the active query
       * began in, 0 if none: only a query that also ends there may become
       * available with that render's retirement. */
      uint32_t begin_render_seq;
   } occlusion;

   /* Fast linked shader data structures */
   uint64_t varyings;
   struct agx_varyings_vs linked_varyings;
   /* hk_g17p_smooth_varyings_supported() of the last (linked FS, hardware
    * VS, topology, provoking) it evaluated against the linked_varyings
    * above. Cleared whenever linked_varyings is reassigned. */
   struct {
      const void *linked;
      const void *vs;
      uint32_t topology;
      uint32_t provoking;
      bool valid;
      bool result;
   } g17p_smooth_memo;

   uint32_t linked_dirty;
   struct hk_linked_shader *linked[MESA_SHADER_STAGES];

   /* Last complete fast-link selection for this command buffer. The shader
    * identity is part of the cache key: identical bytes owned by a different
    * shader may have a different lifetime. Graphics-state invalidation clears
    * these records with the rest of gfx state. */
   struct {
      struct hk_shader *shader;
      struct hk_fast_link_key_vs key;
      bool valid;
   } fast_link_vs;
   struct {
      struct hk_shader *shader;
      struct hk_fast_link_key_fs key;
      bool valid;
      /* Colour attachments the linked fragment program writes, and whether
       * it reads the tilebuffer. */
      uint8_t color_written;
      bool reads_tib;
   } fast_link_fs;

   /* Keep these four native graphics caches contiguous: AGX2 resets leave
    * their zero-initialized storage untouched.
    *
    * Exact last VS/FS loader identities. These are recording-only selectors;
    * the published programs and every referenced BO remain command-buffer-
    * owned. Submission never mutates or consults this cache. */
   struct hk_g17p_graphics_ioto_cache_entry g17p_ioto[2];
   struct hk_g17p_launch g17p_launch[2][HK_G17P_LAUNCH_SLOTS];
   /* Exact command-buffer-owned uploads for combined texture/sampler pointer
    * pairs.  Shader/material switches may revisit the same pair thousands of
    * times; a fresh upload address would otherwise force a fresh Ioto program
    * even though the GPU-visible values are identical. */
   struct {
      /* Key: descriptor-set texture word address and sampler state address. */
      uint64_t pointers[2];
      /* The uploaded pair; its texture pointer is 32-byte aligned
       * (hk_g17p_texture_state_pointer). */
      uint64_t published[2];
      uint64_t address;
      bool valid;
   } g17p_combined_sampler_pointers[32];

   /* Exact immutable eMRT descriptor-table upload used by every fragment
    * program in one spilled render configuration. */
   struct {
      uint8_t entries[HK_MAX_RTS][AGX_G17P_EMRT_DESCRIPTOR_SIZE];
      uint64_t address;
      uint64_t pointer;
      uint8_t color_att_count;
      bool valid;
   } g17p_spilled_rt_descs;

   bool generate_primitive_id;

   /* Tessellation state */
   struct {
      uint64_t out_draws;
      uint64_t grids;
      /* Direct thread count bounding the TES-before-GS kernel of this draw,
       * or 0 to use the indirect grid (hk_tes_direct_query_threads()). */
      uint32_t tes_direct_threads;
      struct hk_tess_info info;
      enum mesa_prim prim;
   } tess;

   /* Native mesh draw state (VK_EXT_mesh_shader, native TA path). */
   struct {
      struct agx_ptr params;
      uint64_t grid, object_program, object_params;
      /* Direct-draw API task grid, used for the object dispatch when an
       * task shader is bound. */
      uint32_t task_grid[3];
   } mesh;

   /* Needed by vk_command_buffer::dynamic_graphics_state */
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
};

/* The CPU source is mutable. Snapshot its visible bytes at publication,
 * before copying into the immutable GPU-mapped root. */
static inline void
hk_root_fragment_snapshot_publish(struct hk_graphics_state *gfx,
                                  const struct hk_root_descriptor_table *cpu,
                                  const struct hk_root_descriptor_table *source)
{
   hk_root_fragment_snapshot_store(&gfx->root_fragment_snapshot, source);
   gfx->root_fragment_snapshot_cpu = cpu;
}

/* Select the same immutable root as the fresh byte comparison. Hidden reads
 * always use the current root; snapshots and the positive identity memo only
 * accelerate the equality check for shaders that cannot observe those bytes. */
static inline uint64_t
hk_m5_fragment_root_select(struct hk_graphics_state *gfx, uint8_t root_reads,
                           bool use_memo, bool use_shadow)
{
   const uint8_t hidden = HK_ROOT_READS_ADDRESS | HK_ROOT_READS_VERTEX_INPUT |
                          HK_ROOT_READS_DYNAMIC_BUFFERS;

   const bool shadow = use_shadow &&
                       gfx->root_fragment_snapshot_cpu == gfx->root_cpu &&
                       gfx->fs_root_fragment_snapshot_cpu == gfx->fs_root_cpu;

   if (gfx->fs_root && gfx->fs_root != gfx->root && gfx->root_cpu &&
       !(root_reads & hidden) &&
       ((use_memo && gfx->fs_root_checked_root &&
         gfx->fs_root_checked_root == gfx->root) ||
        (shadow
            ? !memcmp(&gfx->fs_root_fragment_snapshot,
                      &gfx->root_fragment_snapshot,
                      sizeof(gfx->root_fragment_snapshot))
            : hk_root_matches_for_fragment(gfx->fs_root_cpu, gfx->root_cpu)))) {
      if (use_memo)
         gfx->fs_root_checked_root = gfx->root;
      return gfx->fs_root;
   }

   if (use_shadow && gfx->root_cpu &&
       gfx->root_fragment_snapshot_cpu == gfx->root_cpu) {
      gfx->fs_root_fragment_snapshot = gfx->root_fragment_snapshot;
      gfx->fs_root_fragment_snapshot_cpu = gfx->root_cpu;
   } else {
      gfx->fs_root_fragment_snapshot_cpu = NULL;
   }

   gfx->fs_root_checked_root = 0;
   gfx->fs_root = gfx->root;
   gfx->fs_root_cpu = gfx->root_cpu;
   return gfx->root;
}

struct hk_compute_state {
   struct hk_descriptor_state descriptors;
   struct hk_api_shader *shader;
};

/*
 * VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR state.  Descriptor sets and push
 * constants bound to the ray-tracing bind point are separate from the compute
 * ones, even though the pipeline executes as a compute shader: an application
 * may bind a compute pipeline's sets between vkCmdBindPipeline and
 * vkCmdTraceRaysKHR and neither may disturb the other.
 */
struct hk_ray_tracing_state {
   struct hk_descriptor_state descriptors;
   struct hk_ray_tracing_pipeline *pipeline;
};

#define HK_RT_SHADER_STAGES                                                    \
   (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |         \
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |       \
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR)

struct hk_cmd_push {
   void *map;
   uint64_t addr;
   uint32_t range;
   bool no_prefetch;
};

struct hk_scratch_req {
   bool main;
   bool preamble;
};

/* struct agx_timestamp_req lives in hk_sync.h. */

/*
 * hk_cs represents a single control stream, to be enqueued either to the
 * CDM or VDM for compute/3D respectively.
 */
enum hk_cs_type {
   HK_CS_CDM,
   HK_CS_VDM,
};

/* Fragment programs a control stream records for the frame anatomy
 * profiler (hk_frame_anatomy.h). */
#define HK_FA_FS_MAX 4

struct hk_cs {
   struct list_head node;

   /* Parent command buffer. Convenience. */
   struct hk_cmd_buffer *cmd;

   /* Data master */
   enum hk_cs_type type;

   /* Address of the root control stream for the job */
   uint64_t addr;

   /* Fat pointer to the start of the current chunk of the control stream */
   struct agx_ptr chunk;

   /* Start pointer of the root control stream */
   void *start;

   /* Current pointer within the control stream */
   void *current;

   /* End pointer of the current chunk of the control stream */
   void *end;

   /* AGX3 CDM: the value of current just after a strong cache flush, so a
    * flush requested with nothing recorded since it can be elided. */
   void *strong_flush_end;

   /* Likewise for the weak ordering barrier attached to a native launch. A
    * later cache-flush request upgrades this record in place instead of
    * emitting two adjacent barriers. */
   void *weak_barrier_end;

   /* AGX3 CDM: the value of current just after a strong flush that also has
    * memory scope (AGX3_CDM_BARRIER_FLUSH_MEMORY). On M5 every strong flush
    * has it; on G17P only the flushes hk_cdm_visibility_flush() asks for. */
   void *memory_flush_end;

   /* AGX3 CDM: the value of current just after a standalone barrier that is
    * the stream's *first* record. Apple's emitter only opens a stream with a
    * barrier (the strong flush that makes a preceding render visible), so a
    * stream is canonical on its own; but concatenating it after a stream that
    * already ends in a barrier leaves two adjacent barriers with no launch
    * between them, which Apple never emits. hk_cs_merge_cdm() uses this to
    * fold the leading record into the destination's trailing one. */
   void *leading_barrier_end;

   /* AGX3 CDM: weak_barrier_end when that barrier ends an application
    * dispatch. Where agx_chip_elide_app_dispatch_barrier() holds, a following
    * application dispatch overwrites it, so consecutive application dispatches
    * carry no barrier between them; any other record still follows it. */
   void *app_dispatch_end;

   /* Whether there is more than just the root chunk */
   bool stream_linked;

   /* Active-context selector consumed by AGX3 VertexProgramVariant. */
   bool agx3_active_context_selector_valid;
   uint32_t agx3_raw_active_context_selector;

   /* Whether the sampler heap is required. Although we always must maintain the
    * heap for correctness, it's often not necessary since we can push lots of
    * samplers (especially for GL/DX11-era engines).
    */
   bool uses_sampler_heap;
   bool gen1_native_rq;
   bool gen1_native_fragment_rq;
   bool gen1_native_vertex_rq;
   bool m5_native_rq;
   bool m5_matrix_ops;

   /* At least one AS-build dispatch or compute transfer is in this stream. */
   bool as_build;

   /* M5: a later render pass instance continues in this stream
    * (hk_m5_pass_merge_begin()); its draws may not share the stream's
    * committed sample locations (hk_flush_gfx_state()). */
   bool m5_continued_pass;

   /* A graphics program recorded in this stream writes external memory. */
   bool m5_shader_writes_memory;

   /* M5: the continued instance was ordered after this stream's earlier work
    * by a dependency that is not framebuffer-local (hk_m5_pass_merge_begin()).
    * While this is the open render stream, every draw recorded into it must
    * leave the merge memory-equivalent to the two separate instances, or the
    * stream ends first (hk_m5_ordered_draw_hazard()). */
   bool m5_ordered;

   /* M5: every attachment of the continued instances is in an
    * attachment-only layout, so no descriptor can access one of them and
    * descriptor sets bound for graphics need not end an ordered continuation
    * (hk_m5_attachments_exclude_descriptors()). */
   bool m5_layout_proof;

   /* Scratch requirements */
   struct {
      union {
         struct hk_scratch_req vs;
         struct hk_scratch_req cs;
      };

      struct hk_scratch_req fs;
   } scratch;

   /* Immediate writes, type libagx_imm_write. These happen at the end of the
    * control stream (parallel on AGX2, native ordered stores on G17P). This
    * accelerates queries. Implies CDM.
    */
   struct util_dynarray imm_writes;

   /* Native private-memory profiles for every dispatch in this stream. */
   struct util_dynarray compute_resources;
   /* Fields of the M5 HAL300 graphics core this stream last emitted, and the
    * records they packed to. The PPP latches each record for the following
    * draws of the stream, so an identical core is not emitted again and a
    * changed one emits only its changed records; any other PPP state emitted
    * into the stream clears m5_ppp_emitted_valid. See hk_flush_ppp_state(). */
   struct agx_ppp_hal300_graphics_fields m5_ppp_emitted;
   struct agx_ppp_hal300_graphics_core m5_ppp_emitted_state;
   bool m5_ppp_emitted_valid;

   struct util_dynarray render_resources;
   /* The render_resources entry the last draw matched or appended, see
    * hk_m5_note_render_resources(). */
   uint8_t m5_render_resources_last[3 * sizeof(struct drm_asahi_render_program_v2)];
   size_t m5_render_resources_last_size;
   struct drm_asahi_render_program tile_resources;
   /* Native AGX3: struct hk_oq_stamp for each immediate write that makes an
    * occlusion query available and could instead ride on the retirement
    * timestamp of the render that ended the query. While oq_avail_deferred,
    * the writes are not emitted yet: the end of the command buffer decides
    * (hk_resolve_deferred_oq_available()). Implies CDM. */
   struct util_dynarray oq_stamps;
   bool oq_avail_deferred;

   /* Number of successfully emitted draws whose last raster producer is
    * positively known to export only the default Layer=0, without multiview.
    * Compare with stats.cmds: untracked emitters conservatively disqualify.
    * The wider proof count does not wrap with stats.cmds.
    */
   uint64_t g17p_layer_zero_draws;
   bool g17p_layer_zero_unknown;

   /* An upper bound of the primitives every draw recorded into this G17P
    * render stream submits (hk_draw_primitive_bound()), saturating at
    * UINT64_MAX, which also stands for "not bounded when recorded". Secondary
    * command buffers replay into the primary's stream and add to it there. */
   uint64_t g17p_primitive_bound;

   /* This stream begins (vkCmdBeginRendering without RESUMING) and ends
    * (vkCmdEndRendering without SUSPENDING) its render pass: it is the whole
    * pass, not a part split off or continued. */
   bool g17p_pass_begins, g17p_pass_ends;

   /* Consecutive GPU query resets share one strong flush, at the first
    * consumer or the end of the stream. G13 never sets this. */
   bool query_reset_pending;

   /* Statistics */
   struct {
      uint32_t calls, cmds, flushes, merged;
      uint64_t query_copy_dispatches, query_copy_queries, query_copy_counters;
      uint64_t query_reset_dispatches, query_reset_queries, query_reset_flushes;
      uint32_t robust_index_draws, robust_index_dispatches,
         robust_index_batches;
   } stats;

   /* Frame anatomy profiler only: the first fragment programs this stream
    * draws with and their draw counts; later programs are only counted.
    * Zero unless HK_FRAME_ANATOMY is set. */
   struct {
      const struct hk_shader *fs;
      uint32_t draws;
      /* Compute streams: total threads and workgroups of the direct
       * dispatches, and how many were indirect. */
      uint64_t threads, groups;
      uint32_t indirect;
   } anatomy_fs[HK_FA_FS_MAX];
   uint32_t anatomy_fs_other;

   /* Likewise for the hardware vertex programs. */
   struct {
      const struct hk_shader *vs;
      uint32_t draws;
   } anatomy_vs[HK_FA_FS_MAX];
   uint32_t anatomy_vs_other;

   /* Frame anatomy profiler only: driver-internal (precompiled) kernels this
    * stream dispatches, with counts and total threads (UINT64_MAX once any
    * dispatch is indirect). */
   struct {
      uint16_t program;
      uint16_t dispatches;
      uint64_t threads;
   } anatomy_precomp[HK_FA_FS_MAX];
   uint32_t anatomy_precomp_other;

   /* Frame anatomy profiler only: the API pipeline barriers recorded between
    * the previous stream's creation and this one's (count, and the union of
    * their source and destination stage masks). */
   uint32_t anatomy_api_barriers;
   uint16_t anatomy_api_mem, anatomy_api_buf, anatomy_api_img;
   /* Frame anatomy profiler only: draws that republished render state,
    * vertex-program state updates and how many named a program other than
    * the stream's previous one, and PPP state records emitted. */
   uint32_t anatomy_republish, anatomy_vdm, anatomy_vdm_new, anatomy_ppp;
   /* Colour attachments whose load the first draw made unnecessary. */
   uint8_t anatomy_covered;
   /* Frame anatomy profiler only, M5: draws by the latched fragment pass type
    * (opaque, translucent, punch-through, translucent punch-through), and
    * draws with triangle merging disabled, with depth writes, and with a
    * depth mode other than UNCHANGED (shader depth or side effects). */
   uint32_t anatomy_pass[4], anatomy_nomerge, anatomy_zwrite, anatomy_late_z;
   /* Frame anatomy profiler only: the render area of a pass that does not
    * cover its whole framebuffer (zero extent otherwise). */
   VkRect2D anatomy_area;
   uint64_t anatomy_last_program;
   uint64_t anatomy_api_src, anatomy_api_dst;
   uint64_t anatomy_api_src_access, anatomy_api_dst_access;

   /* Timestamp writes: the start of the command (compute start, render
    * vertex start) and its end (compute end, render fragment end). The
    * kernel writes both from the firmware's native pass record.
    */
   struct {
      struct agx_timestamp_req start;
      struct agx_timestamp_req end;
   } timestamp;

   /* Render only: occlusion availability slots written by the kernel at
    * retirement, indexed vertex start, vertex end, fragment start, fragment
    * end. Slots 0 and 3 are only used when timestamp.start/end are not. */
   struct agx_timestamp_req oq_stamp[4];

   /* G17P query commands observe timestamp writes from earlier streams.
    * Firmware start timestamps reach query memory only at retirement, so a
    * producer cannot be merged across this consumer boundary.
    */
   bool g17p_timestamp_query;

   /* Recorded dependencies, see hk_sync.h. */
   struct hk_sync_stream sync;

   /* Writable destinations known while recording meta compute dispatches. */
   struct hk_drm_attachment_set compute_attachments;

   /* Remaining state is for graphics only, ignored for compute */
   struct agx_tilebuffer_layout tib;

   struct util_dynarray scissor, depth_bias;
   uint64_t uploaded_scissor, uploaded_zbias;

   /* G17P: the last uploaded draw-state selector record and the scissor /
    * depth-bias array sizes and viewport count it selects for. */
   struct {
      uint32_t offset;
      uint32_t scissor_size, depth_bias_size;
      uint8_t viewport_count;
   } g17p_draw_state;

   /* We can only set ppp_multisamplectl once per batch. has_sample_locations
    * tracks if we've committed to a set of sample locations yet. vk_meta
    * operations do not set has_sample_locations since they don't care and it
    * would interfere with the app-provided samples.
    *
    */
   bool has_sample_locations;
   uint64_t ppp_multisamplectl;

   struct hk_render_registers cr;

   /* Snapshot of the writable image ranges for this exact VDM command. Render
    * state may change before queue submission, so the synthetic fragment
    * attachment command must travel with the control stream it describes. */
   struct hk_drm_attachment_set fragment_attachments;

   /* The restart index last emitted to this stream, valid once
    * restart_index_set. VK_EXT_primitive_restart_index makes every value,
    * zero included, a legal custom index, so no value can mean "unset".
    * This is used in place of dirty tracking, because dirty tracking
    * restart indices is complicated and just checking the saved value is cheap.
    */
   uint32_t restart_index;
   bool restart_index_set;
};

static inline unsigned
hk_sync_sq(const struct hk_cs *cs)
{
   return cs->type == HK_CS_VDM ? HK_SYNC_VDM : HK_SYNC_CDM;
}

/* The later of two streams in list order; NULL is earlier than anything. */
static inline struct hk_cs *
hk_sync_later(struct hk_cs *a, struct hk_cs *b)
{
   if (!a)
      return b;
   if (!b)
      return a;
   return b->sync.seq > a->sync.seq ? b : a;
}

void *hk_cmd_buffer_emit_cs_jump(struct hk_cmd_buffer *cmd, void *current,
                                 uint64_t target, bool vdm);

/*
 * Helper to merge two compute control streams, concatenating the second control
 * stream to the first one. Must sync with hk_cs.
 */
static inline void
hk_cs_merge_cdm(struct hk_cs *a, const struct hk_cs *b)
{
   assert(a->type == HK_CS_CDM && b->type == HK_CS_CDM);
   assert(a->cmd == b->cmd);
   assert(!a->timestamp.end.handle);
   a->m5_native_rq |= b->m5_native_rq;
   a->m5_matrix_ops |= b->m5_matrix_ops;

   const size_t b_size = (uint8_t *)b->current - (uint8_t *)b->start;
   const size_t remaining = (uint8_t *)a->end - (uint8_t *)a->current;
   const bool inline_b = !a->stream_linked && !b->stream_linked &&
                         hk_cs_space_fits(remaining, b_size,
                                          AGX_CDM_STREAM_LINK_LENGTH);

   if (inline_b) {
      uint8_t *dst = a->current;

      /* Two barriers with no launch between them is not a shape Apple's CDM
       * emitter produces, and the G17P stream oracle rejects it. Within one
       * stream hk_cdm_cache_flush() already strengthens the trailing record
       * in place; do the same across a concatenation. b's leading barrier is
       * the strong flush, so copying its word over a's trailing barrier is
       * exactly that upgrade -- no encoder needed, and it removes a record
       * rather than adding one. */
      const bool b_opens_with_barrier =
         b->leading_barrier_end &&
         (uint8_t *)b->leading_barrier_end ==
            (uint8_t *)b->start + AGX3_CDM_BARRIER_SIZE;
      const bool a_ends_with_barrier =
         ((uint8_t *)a->current == (uint8_t *)a->strong_flush_end ||
          (uint8_t *)a->current == (uint8_t *)a->weak_barrier_end) &&
         (uint8_t *)a->current - (uint8_t *)a->start >= AGX3_CDM_BARRIER_SIZE;
      const size_t b_skip = (b_opens_with_barrier && a_ends_with_barrier)
                               ? AGX3_CDM_BARRIER_SIZE
                               : 0;
      const bool a_memory = (uint8_t *)a->current == (uint8_t *)a->memory_flush_end;
      if (b_skip) {
         /* b's strong flush must not drop the memory scope a's trailing
          * record carried for the work before it. */
         uint32_t word = agx3_cmdstream_load_le32(b->start);
         if (a_memory)
            word |= AGX3_CDM_BARRIER_FLUSH_MEMORY;
         agx3_cmdstream_store_le32(dst - AGX3_CDM_BARRIER_SIZE, word);
      }

      memcpy(dst, (const uint8_t *)b->start + b_skip, b_size - b_skip);
      a->current = dst + (b_size - b_skip);

      /* These markers point just past an emitted barrier. Preserve them in
       * the destination root so another adjacent merge still has an exact
       * view of the stream tail. A marker on the folded record maps onto the
       * trailing record that absorbed it. */
      a->strong_flush_end =
         b->strong_flush_end
            ? dst + ((uint8_t *)b->strong_flush_end - (uint8_t *)b->start) -
                 b_skip
            : NULL;
      a->weak_barrier_end =
         b->weak_barrier_end
            ? dst + ((uint8_t *)b->weak_barrier_end - (uint8_t *)b->start) -
                 b_skip
            : NULL;
      a->memory_flush_end =
         b->memory_flush_end
            ? dst + ((uint8_t *)b->memory_flush_end - (uint8_t *)b->start) -
                 b_skip
         : (a_memory && b_skip && a->current == dst) ? dst
                                                     : NULL;
      a->app_dispatch_end = NULL;
   } else {
      void *link_end =
         hk_cmd_buffer_emit_cs_jump(a->cmd, a->current, b->addr, false);
      const bool linked = link_end != a->current;
      a->current = b->current;
      a->end = b->end;
      a->chunk = b->chunk;
      a->stream_linked |= linked || b->stream_linked;
      a->strong_flush_end = b->strong_flush_end;
      a->weak_barrier_end = b->weak_barrier_end;
      a->memory_flush_end = b->memory_flush_end;
      a->app_dispatch_end = NULL;
   }

   a->uses_sampler_heap |= b->uses_sampler_heap;
   a->gen1_native_rq |= b->gen1_native_rq;
   a->as_build |= b->as_build;
   a->scratch.cs.main |= b->scratch.cs.main;
   a->scratch.cs.preamble |= b->scratch.cs.preamble;

   /* Submission sees only the merged root. Keep the destinations recorded
    * by both streams without changing their inline/link batching policy. */
   for (unsigned i = 0; i < b->compute_attachments.count; ++i) {
      const struct drm_asahi_attachment *attachment =
         &b->compute_attachments.list[i];
      hk_drm_attachment_set_add(&a->compute_attachments,
                                attachment->pointer, attachment->size);
   }
   a->compute_attachments.complete &= b->compute_attachments.complete;

   /* The merged command ends where b ended. It starts where a started; a
    * start stamp b carried was a top-of-pipe write, which any earlier value
    * also satisfies. The caller refuses the merge when both have one. */
   a->timestamp.end = b->timestamp.end;
   if (!a->timestamp.start.handle)
      a->timestamp.start = b->timestamp.start;

   util_dynarray_append_array(&a->compute_resources, uint8_t,
                             b->compute_resources.data, b->compute_resources.size);

   a->g17p_timestamp_query |= b->g17p_timestamp_query;

   a->stats.calls += b->stats.calls;
   a->stats.cmds += b->stats.cmds;
   a->stats.flushes += b->stats.flushes;
   a->stats.robust_index_draws += b->stats.robust_index_draws;
   a->stats.robust_index_dispatches += b->stats.robust_index_dispatches;
   a->stats.robust_index_batches += b->stats.robust_index_batches;
   a->stats.merged++;
   a->stats.query_copy_dispatches += b->stats.query_copy_dispatches;
   a->stats.query_copy_queries += b->stats.query_copy_queries;
   a->stats.query_copy_counters += b->stats.query_copy_counters;
   a->stats.query_reset_dispatches += b->stats.query_reset_dispatches;
   a->stats.query_reset_queries += b->stats.query_reset_queries;
   a->stats.query_reset_flushes += b->stats.query_reset_flushes;
   assert(!a->query_reset_pending && !b->query_reset_pending);
}

static inline uint64_t
hk_cs_current_addr(struct hk_cs *cs)
{
   return cs->chunk.gpu + ((uint8_t *)cs->current - (uint8_t *)cs->chunk.cpu);
}

struct hk_uploader {
   /** List of hk_cmd_bo */
   struct list_head bos;

   /* Current addresses */
   uint8_t *map;
   uint64_t base;
   uint32_t offset;
   uint32_t capacity;

   /* VDM uploader only: low alias of base (its hk_vdm_allocation's alias). */
   uint64_t vdm_alias;
};

/*
 * Bump arena for the G17P Ioto programs one command buffer records. Semantic
 * program reuse happens before reservation; resource programs embed their
 * reserved address, so post-composition byte hashing cannot deduplicate them.
 * See hk_ioto_arena_core.h for why programs cannot each have a BO.
 */
struct hk_ioto_arena {
   /* Chunk currently being filled. Owned by hk_cmd_buffer::large_bos. */
   struct agx_bo *bo;
   uint8_t *map;
   uint64_t base;
   uint32_t cursor;
   uint32_t capacity;

   /* Recording statistics, reported when the arena cannot grow. */
   uint32_t programs;
   uint64_t program_bytes;
   uint64_t chunk_bytes;
};

struct hk_gen1_as_table_chunk {
   struct list_head link;
   struct agx_bo *tail;
   struct agx_bo *bank;
   struct agx_va *va;
   uint32_t next, count;
};

#define HK_MAX_DEFERRED_CLEARS 8

struct hk_cmd_buffer {
   struct vk_command_buffer vk;

   /* Frame anatomy profiler only: API pipeline barriers recorded since the
    * last stream was created (see hk_cs::anatomy_api_barriers). */
   struct {
      uint32_t count;
      uint16_t mem, buf, img;
      uint64_t src, dst, src_access, dst_access;
   } anatomy_bar;

   /* frg-codegen v4: the per-recording BG/EOT cache (13.5 KiB) lives outside
    * `state` so the full-state memset of every reset no longer clears it;
    * hk_reset_cmd_buffer() invalidates the entries instead. Entries are only
    * read when valid, so this is the same cache contract. */
   struct hk_g17p_bg_eot_cache_entry
      g17p_bg_eot_cache[HK_G17P_BG_EOT_CACHE_ENTRIES];
   uint8_t g17p_bg_eot_cache_next;

   /* Set by hk_reset_cmd_buffer, cleared by vkBeginCommandBuffer. */
   bool g17p_pristine;

   struct {
      struct hk_upload_stats upload_stats;
      struct {
         uint64_t id, addr;
      } constant_data[MESA_SHADER_STAGES];
      struct hk_graphics_state gfx;
      struct hk_compute_state cs;
      struct hk_ray_tracing_state rt;
      struct hk_conditional_render_state conditional;

      /* Whole-image clears not yet recorded, see
       * hk_flush_deferred_clears(). */
      struct hk_deferred_clear {
         struct hk_image *image;
         VkImageAspectFlags aspects;
         VkClearValue value;
      } deferred_clear[HK_MAX_DEFERRED_CLEARS];
      uint8_t deferred_clear_count;
      bool deferred_clear_flushing;
      /* Recording a buffer-only command that does not touch their memory. */
      bool deferred_clear_hold;

      /* Suppress consumption of pending resets while recording another reset. */
      bool query_reset_recording;

      /* Scoped around direct/indirect AS builds, including their meta blits. */
      bool as_build_recording;

      /* AGX3: the last render pass instance stored a whole depth-only image
       * that nothing recorded since may have read. A later instance that
       * clears the whole image, or does not care about its contents, makes
       * that store dead (hk_take_pending_z_store()). Kept across exactly the
       * commands deferred clears survive (deferred_clear_hold). */
      struct hk_pending_z_store {
         struct hk_cs *cs;
         const struct hk_image *image;
         struct agx_zls_control_packed zls_control;
         bool process_empty_tiles;
      } pending_z_store;
      /* M5: the render pass instance that ended last, while the next one may
       * continue in its render stream instead of storing and reloading the
       * attachments (hk_m5_pass_merge_begin()). */
      struct hk_pass_merge_candidate {
         bool valid;
         /* The instance's single render stream, where its draws ended and
          * where its terminator left the stream and chunk. */
         struct hk_cs *cs;
         void *resume_current;
         void *after_current;
         void *end;
         /* The tracked state when the instance ended; the next instance may
          * continue only if none of it moved. */
         uint32_t next_seq;
         uint32_t sync_epoch;
         struct hk_cs *open_cs;
         void *open_cs_current;
         /* A recorded dependency orders the next instance after this one
          * beyond attachment operations; continuing then needs the stronger
          * conditions of hk_m5_ordered_refusal(). */
         bool ordered;
         /* The render state the instance ended with. */
         struct hk_rendering_state render;
      } pass_merge;

      /* M5: the open render stream is an ordered continuation (hk_cs::
       * m5_ordered) and a command recorded since may read the instance's
       * attachments through memory; the next draw ends the stream first. */
      bool m5_ordered_split;

      /* Per recording; reported at submission so reused command buffers are
       * counted each time. Reasons overlap, total counts physical splits. */
      struct {
         uint64_t kept, splits;
         uint64_t reason[HK_BARRIER_SPLIT_COUNT];
      } barrier_stats;
   } state;

   struct {
      struct hk_uploader main, usc, root, vdm, streams;
   } uploader;

   /* Command-pool slab spans leased by this command buffer's VDM uploader. */
   struct list_head vdm_allocations;
   /* Most recently addressed span. Recording often references the current
    * 4 KiB VDM chunk many times before moving to the next one. */
   struct hk_vdm_allocation *g17p_vdm_last_allocation;
   struct list_head gen1_as_table_chunks;

   /* List of all recorded control streams */
   struct list_head control_streams;

   /* Current recorded control stream */
   struct {
      /* VDM stream for 3D */
      struct hk_cs *gfx;

      /* CDM stream for compute */
      struct hk_cs *cs;

      /* CDM stream that executes immediately before the current graphics
       * control stream. Used for geometry shading, tessellation, etc.
       */
      struct hk_cs *pre_gfx;

      /* CDM stream that will execute after the current graphics control stream
       * finishes. Used for queries.
       */
      struct hk_cs *post_gfx;
   } current_cs;

   /* Recorded dependencies between control streams, see hk_sync.h. */
   struct hk_sync_state sync;

   /* Tile hazards: struct hk_sync_pass_record for every recorded pass
    * instance footprint, in recording order. Cleared by hk_sync_reset. */
   struct util_dynarray sync_pass_records;

   /* struct hk_oq_end for every occlusion query this command buffer ends. */
   struct util_dynarray oq_ends;

   /* Are we currently inside a vk_meta operation? This alters sample location
    * behaviour.
    */
   bool in_meta;

   /* CmdClearAttachments is the sole vk_meta operation affected by
    * conditional rendering. */
   bool conditional_meta;

   /* M5 native ray traversal: the per-raygen dispatches of one
    * vkCmdTraceRays* share a single initialized SCS state block, since only
    * the selected kernel has threadgroups. Valid only while `share` is set. */
   struct {
      bool share;
      uint64_t packed;
      struct agx_ptr state;
      /* Traversal uniforms shared by the same dispatches. */
      struct agx_ptr uniforms;
      uint64_t uniforms_packed, uniforms_state;
      /* Opt-in ASAHI_M5_RT_STATE_DIAG: initial images of the SCS state
       * blocks this recording allocated, compared after execution. */
      struct util_dynarray diag;
   } m5_rt_state;

   /* Single-kick vkCmdTraceRays* (ASAHI_M5_RT_SINGLE_KICK, default on): while set,
    * hk_dispatch_m5_native writes the kernel's launch record (40-byte direct
    * launch + barrier) here instead of into the control stream. */
   uint8_t *m5_rt_capture;
   bool m5_rt_captured;

   /* XXX: move me?
    *
    * Indirect draw generated by the indirect GS translator.
    */
   uint64_t geom_index_buffer;
   uint32_t geom_index_count;
   uint32_t geom_instance_count;

   /* Does the command buffer use the geometry heap? */
   bool uses_heap;

   /* Owned BOs and retained references needed by recorded commands. */
   struct util_dynarray large_bos;
   /* Direct-mapped temporal cache avoids rescanning large_bos for the linked
    * vertex/fragment shader, setup and rodata BOs repeated by every draw. */
   struct agx_bo *recent_retained_bos[16];
   /* Open-addressed pointer set of the BOs hk_cmd_buffer_retain_bo() has
    * appended to large_bos (capacity a power of two, or zero). A recording
    * retains one BO per distinct shader/setup/program it publishes, so a
    * linear scan of large_bos on every temporal-cache miss grew quadratically
    * with the number of pipelines a command buffer uses. */
   struct agx_bo **retained_set;
   uint32_t retained_set_capacity;
   uint32_t retained_set_count;

   /* M5: every attachment-capable image (hk_image) a graphics push descriptor
    * has referenced since recording began. Push descriptors stay valid until
    * overwritten, so an image recorded here may still be readable by the
    * bound shaders (hk_m5_ordered_refusal()). Created on first use. */
   struct set *m5_pushed_images;

   /* Executable USC-window arena the G17P Ioto programs recorded by this
    * command buffer are bump-allocated from. Its chunks are owned by
    * large_bos, so it dies with the recording like every other program
    * reference. */
   struct hk_ioto_arena ioto;
};

/* Transfer entry points bracket their recording so the streams they create
 * belong to the transfer stage class (VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT and
 * its sub-stages). Nesting is fine: meta helpers re-enter through the
 * dispatch table. */
static inline void
hk_sync_xfer_begin(struct hk_cmd_buffer *cmd)
{
   /* An outermost transfer without a known destination may write anything
    * (hk_sync_note_xfer_write()). */
   if (cmd->sync.xfer_depth++ == 0)
      cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){0};
}

/* As hk_sync_xfer_begin(), for a transfer whose writes stay inside the GPU
 * VA range [lo, hi). device_address says shaders may also reach the range
 * through a buffer device address. A nested transfer widens the range of the
 * outermost one. */
static inline void
hk_sync_xfer_begin_dst(struct hk_cmd_buffer *cmd, uint64_t lo, uint64_t hi,
                       bool device_address)
{
   struct hk_sync_xfer_dst *dst = &cmd->sync.xfer_dst;
   if (cmd->sync.xfer_depth++ == 0) {
      *dst = (struct hk_sync_xfer_dst){
         .lo = lo,
         .hi = hi,
         .valid = lo < hi,
         .device_address = device_address,
      };
   } else if (dst->valid) {
      dst->lo = MIN2(dst->lo, lo);
      dst->hi = MAX2(dst->hi, hi);
      dst->device_address |= device_address;
   }
}

static inline void
hk_sync_xfer_end(struct hk_cmd_buffer *cmd)
{
   assert(cmd->sync.xfer_depth > 0);
   if (--cmd->sync.xfer_depth == 0)
      cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){0};
}

/*
 * One reservation inside the Ioto arena: where the program is to be composed,
 * and the address the composer must be given for it.
 */
struct hk_ioto_reservation {
   void *map;
   uint64_t va;
};

/*
 * Reserve `size` bytes of executable USC-window space for one Ioto program.
 * On failure the command buffer error is already set.
 */
bool hk_cmd_reserve_ioto_program_slow(struct hk_cmd_buffer *cmd, size_t size,
                                      const char *label,
                                      struct hk_ioto_reservation *reservation);

/* AGX_IOTO_PROGRAM_ALIGNMENT, checked in hk_ioto_arena.c. */
#define HK_IOTO_PROGRAM_ALIGNMENT 64u

static inline bool
hk_cmd_reserve_ioto_program(struct hk_cmd_buffer *cmd, size_t size,
                            const char *label,
                            struct hk_ioto_reservation *reservation)
{
   /* The placement hk_ioto_arena_place() computes, in the current chunk. */
   struct hk_ioto_arena *arena = &cmd->ioto;
   const uint32_t mask = HK_IOTO_PROGRAM_ALIGNMENT - 1;
   if (likely(reservation && size != 0 && size <= UINT32_MAX && arena->map &&
              arena->cursor <= arena->capacity &&
              arena->cursor <= UINT32_MAX - mask)) {
      const uint32_t offset = (arena->cursor + mask) & ~mask;
      if (likely(offset <= arena->capacity &&
                 (uint32_t)size <= arena->capacity - offset)) {
         *reservation = (struct hk_ioto_reservation){
            .map = arena->map + offset,
            .va = arena->base + offset,
         };
         arena->cursor = offset + (uint32_t)size;
         return true;
      }
   }

   return hk_cmd_reserve_ioto_program_slow(cmd, size, label, reservation);
}

/*
 * Publish the `length` bytes composed at reservation->map and return the
 * reserved address for the launch record.
 */
uint64_t hk_cmd_publish_ioto_program(struct hk_cmd_buffer *cmd,
                                     const struct hk_ioto_reservation *
                                        reservation,
                                     size_t length);

/* Clean the current chunk's programs from the CPU cache (cached command
 * memory); hk_cmd_buffer_publish_cached_uploads() fences it. */
void hk_ioto_arena_clean(struct hk_cmd_buffer *cmd);

/* Release the arena's bookkeeping. The chunks themselves are in large_bos. */
void hk_ioto_arena_finish(struct hk_cmd_buffer *cmd);

VK_DEFINE_HANDLE_CASTS(hk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops hk_cmd_buffer_ops;

static inline struct hk_cmd_pool *
hk_cmd_buffer_pool(struct hk_cmd_buffer *cmd)
{
   return (struct hk_cmd_pool *)cmd->vk.pool;
}

/* Whether the current blend state reads the blend constants. Derived from
 * the dynamic graphics state at each use rather than cached: the fragment
 * stage program (its blend-constant uniform range) is built by bind-time and
 * BeginRendering preparation as well as by the draw, and that preparation
 * runs before the draw-time dirty-state flush. A cached copy refreshed only
 * by that flush was stale there, so the first draw after a blend-equation
 * change published a fragment program without the blend-constant range and
 * the epilog read unwritten uniforms. Prepared programs are keyed by
 * dynamic_graphics_state.generation, which covers every input here. */
static inline bool
hk_uses_blend_constant(const struct vk_color_blend_state *cb)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                    VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                 "constant factors are contiguous");

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      const unsigned factors[] = {
         cb->attachments[i].src_color_blend_factor,
         cb->attachments[i].src_alpha_blend_factor,
         cb->attachments[i].dst_color_blend_factor,
         cb->attachments[i].dst_alpha_blend_factor,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(factors); ++j) {
         if (factors[j] >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
             factors[j] <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
            return true;
      }
   }

   return false;
}

static inline bool
hk_cmd_uses_blend_constant(const struct hk_cmd_buffer *cmd)
{
   return hk_uses_blend_constant(&cmd->vk.dynamic_graphics_state.cb);
}

/*
 * The hardware vertex shader is supplied by the last geometry stage. The
 * geometry pipeline is vertex->tess->geometry so we search backwards.
 */
static inline struct hk_shader *
hk_bound_hw_vs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];

   /* A mesh pipeline has no vertex, tessellation or geometry stage; its
    * native mesh variant is the hardware vertex program. */
   if (mesh)
      return &mesh->variants[HK_MESH_VARIANT_MAIN];
   else if (gs)
      return &gs->variants[HK_GS_VARIANT_RAST];
   else if (tes)
      return &tes->variants[HK_VS_VARIANT_HW];
   else
      return &vs->variants[HK_VS_VARIANT_HW];
}

static inline struct hk_shader *
hk_bound_sw_vs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);

   /* Mesh pipelines: the raster variant is both the hardware and the only
    * vertex program (no attributes, no prolog remap). */
   if (!vs)
      return hw_vs;

   if (hw_vs == &vs->variants[HK_VS_VARIANT_HW])
      return hw_vs;
   else
      return &vs->variants[HK_VS_VARIANT_SW];
}

static inline struct hk_shader *
hk_bound_sw_vs_before_gs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_api_shader *api = tes ?: vs;

   /* Mesh pipelines have no software vertex stage. */
   if (!api)
      return NULL;

   return &api->variants[HK_VS_VARIANT_SW];
}

struct agx_ptr hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd,
                                       struct hk_uploader *uploader,
                                       uint32_t size, uint32_t alignment,
                                       enum hk_cmd_bo_type type);

/* Bump allocation from the uploader's current BO; a new BO or a large
 * allocation takes hk_pool_alloc_from_slow(). */
static inline struct agx_ptr
hk_pool_alloc_from(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                   uint32_t size, uint32_t alignment,
                   enum hk_cmd_bo_type type)
{
   assert(alignment > 0);
   const uint32_t offset = align(uploader->offset, alignment);
   if (likely(uploader->map != NULL && size <= HK_CMD_BO_SIZE &&
              offset <= HK_CMD_BO_SIZE && size <= HK_CMD_BO_SIZE - offset)) {
      uploader->offset = offset + size;
      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   return hk_pool_alloc_from_slow(cmd, uploader, size, alignment, type);
}

struct agx_ptr hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                                uint32_t alignment);
struct agx_ptr hk_pool_vdm_primary_alloc(struct hk_cmd_buffer *cmd,
                                         uint32_t size, uint32_t alignment);

uint64_t hk_g17p_vdm_compact(struct hk_cmd_buffer *cmd, uint64_t primary);
uint64_t hk_g17p_vdm_alias(struct hk_cmd_buffer *cmd, uint64_t primary);

uint64_t hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data,
                        uint32_t size, uint32_t alignment);

/*
 * G17P does not honour the low five bits of the texture-table pointer that
 * the combined-sampler ABIs publish in uniform halves 0..3: the 32-byte
 * texture state is read as if the pointer were 32-byte aligned.  Sampled-image
 * descriptors only have 8-byte alignment and a 72-byte stride in a descriptor
 * set, so set address + binding offset is generally not a valid table
 * pointer.  Measured on G17P with vkCmdBlitImage of D32_SFLOAT_S8_UINT, whose
 * pushed set holds the depth and stencil texture words at offsets 72 and 144:
 * every depth and stencil texel read back wrong, and when the qword before
 * the stencil texture word (the depth descriptor's last qword) was nonzero the
 * texture unit's read faulted (VM error at 0x40000000000, VK_ERROR_DEVICE_LOST
 * in Pokemon Scarlet).  The same descriptor bytes published at a 32-byte
 * aligned address -- including 32 mod 64 -- read back exactly and never
 * faulted, while 8 and 16 mod 32 still read back wrong.
 *
 * Returns texture_va when it is already aligned, otherwise the address of a
 * 32-byte-aligned command-buffer copy of the descriptor record (texture word
 * plus the software fields after it, which descriptor crawls through the
 * same pointer may read), or 0 on allocation failure.  The ABIs using this
 * that need a copy are not selected when sampled-image update-after-bind is
 * enabled.  The direct txf ABI may still publish an already aligned live set
 * pointer in that case.  The combined-sampler ABIs also snapshot the
 * descriptor's sampler index at record time, so their copies hold what the set
 * holds when the commands execute.
 */
#define HK_G17P_TEXTURE_STATE_ALIGN 32

uint64_t hk_g17p_texture_state_pointer(struct hk_cmd_buffer *cmd,
                                       uint64_t texture_va,
                                       const void *descriptor,
                                       uint32_t available_B);
uint64_t hk_pool_usc_upload(struct hk_cmd_buffer *cmd, const void *data,
                            uint32_t size, uint32_t alignment);

/* Keep externally owned storage referenced by recorded GPU commands alive for
 * the command-buffer lifetime. Repeated uses of the same BO are deduplicated.
 */
void hk_cmd_buffer_retain_bo(struct hk_cmd_buffer *cmd, struct agx_bo *bo);
void hk_cmd_buffer_release_bos(struct hk_cmd_buffer *cmd,
                               struct agx_device *dev);

static inline struct agx_ptr
hk_pool_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_from(cmd, &cmd->uploader.main, size, alignment,
                             HK_CMD_BO_MAIN);
}

static inline struct agx_ptr
hk_pool_usc_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_from(cmd, &cmd->uploader.usc, size, alignment,
                             HK_CMD_BO_USC);
}

/*
 * Allocate a payload that a VDM record references through its 40-bit pointer.
 * HAL200 draw-state fields are narrower than the VDM control-stream pointer,
 * so those payloads retain a device-global low alias. Control-stream roots
 * themselves use the constrained main heap directly.
 */
static inline struct agx_ptr
hk_pool_vdm_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Bump-allocate from the current VDM chunk and derive its compact pointer
    * from the chunk's alias.  Allocations never overlap, so this is the
    * address hk_g17p_vdm_compact() resolves through the chunk's
    * hk_vdm_allocation; anything else takes the general path below. */
   struct hk_uploader *vdm = &cmd->uploader.vdm;
   const uint32_t vdm_offset = align(vdm->offset, alignment);
   uint32_t vdm_compact;
   if (likely(vdm->map && vdm_offset <= vdm->capacity &&
              size <= vdm->capacity - vdm_offset &&
              hk_g17p_vdm_low_window_enabled(&dev->dev) &&
              agx_g17p_vdm_alias_compact_inline(vdm->vdm_alias + vdm_offset,
                                                &vdm_compact) &&
              vdm_compact != 0)) {
      vdm->offset = vdm_offset + size;
      return (struct agx_ptr){
         .gpu = vdm_compact,
         .cpu = vdm->map + vdm_offset,
      };
   }

   struct agx_ptr T = hk_pool_vdm_primary_alloc(cmd, size, alignment);

   if (T.gpu && hk_g17p_vdm_low_window_enabled(&dev->dev)) {
      T.gpu = hk_g17p_vdm_compact(cmd, T.gpu);
      if (!T.gpu) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         T.cpu = NULL;
      }
   }

   return T;
}

bool hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
/* A depth/stencil clear touched the pass's attachment (ZLS elision). */
void hk_g17p_note_depth_clear(struct hk_cmd_buffer *cmd,
                              VkImageAspectFlags aspects);

uint64_t hk_default_sample_positions(unsigned nr_samples);

void hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why);
#define hk_flush_deferred_clears(cmd) hk_flush_deferred_clears_why(cmd, __func__)

/* The depth image of the pending store is overwritten as a whole before
 * anything read it: the store is dead (see hk_take_pending_z_store()). */
static inline void
hk_kill_pending_z_store(struct hk_cmd_buffer *cmd)
{
   struct hk_pending_z_store *p = &cmd->state.pending_z_store;
   p->cs->cr.zls_control = p->zls_control;
   p->cs->cr.process_empty_tiles = p->process_empty_tiles;
   p->cs = NULL;
}
void hk_deferred_clears_hold_buffer(struct hk_cmd_buffer *cmd, uint64_t addr,
                                    uint64_t size);
bool hk_defer_clear(struct hk_cmd_buffer *cmd, struct hk_image *image,
                    VkImageAspectFlags aspects, const VkClearValue *value,
                    uint32_t range_count, const VkImageSubresourceRange *ranges);
const VkRenderingInfo *
hk_take_deferred_clears(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                        VkRenderingInfo *copy,
                        VkRenderingAttachmentInfo *color,
                        VkRenderingAttachmentInfo *depth,
                        VkRenderingAttachmentInfo *stencil);

void hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

static inline struct hk_cs *
hk_cmd_buffer_get_cs_general(struct hk_cmd_buffer *cmd, struct hk_cs **ptr,
                             bool compute)
{
   /* Any recorded work may read an image with a deferred clear, or the
    * depth image of a pending store. */
   if (unlikely(cmd->state.deferred_clear_count) &&
       !cmd->state.deferred_clear_flushing && !cmd->state.deferred_clear_hold)
      hk_flush_deferred_clears(cmd);
   if (unlikely(cmd->state.pending_z_store.cs) &&
       !cmd->state.deferred_clear_hold)
      cmd->state.pending_z_store.cs = NULL;

   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

   /* A recorded dependency may have ordered the open general compute stream
    * after work that follows it in the list. It is then ended, and the work
    * goes into a stream positioned after that requirement. Work of a new
    * stage class must also update the reused stream's kind and dependencies,
    * so later barriers include transfers and meta graphics emitted there. */
   if (*ptr && ptr == &cmd->current_cs.cs &&
       (cmd->sync.cs_check_pending ||
        (hk_sync_general_cs_kind(cmd) & ~(*ptr)->sync.kind)))
      hk_sync_check_cs(cmd);

   if ((*ptr) == NULL) {
      /* Allocate root control stream */
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      size_t initial_size = hk_cs_root_size(dev->dev.chip, compute);
      struct agx_ptr root = hk_pool_cs_alloc(cmd, initial_size, 1024);
      if (!root.cpu)
         return NULL;

      /* Allocate hk_cs for the new stream */
      struct hk_cs *cs = malloc(sizeof(*cs));
      *cs = (struct hk_cs){
         .cmd = cmd,
         .type = compute ? HK_CS_CDM : HK_CS_VDM,
         .addr = root.gpu,
         .start = root.cpu,
         .chunk = root,
         .current = root.cpu,
         .end = root.cpu + initial_size,
      };

      util_dynarray_init(&cs->compute_resources, NULL);
      util_dynarray_init(&cs->render_resources, NULL);
      list_inithead(&cs->node);
      hk_drm_attachment_set_init(&cs->compute_attachments);
      /* Only known meta destinations are collected so far. */
      cs->compute_attachments.complete = false;

      bool before_gfx = (ptr == &cmd->current_cs.pre_gfx);

      if (!compute) {
         const bool initialized = hk_cs_init_graphics(cmd, cs);
         if (!hk_graphics_cs_publishable(
                dev->dev.chip, initialized,
                cs->agx3_active_context_selector_valid)) {
            free(cs);
            return NULL;
         }
      }

      /* Insert into the command buffer. We usually append to the end of the
       * command buffer, except for pre-graphics streams which go right before
       * the graphics workload. (This implies a level of out-of-order processing
       * that's allowed by Vulkan and required for efficient
       * geometry/tessellation shaders.)
       */
      if (before_gfx && cmd->current_cs.gfx) {
         list_addtail(&cs->node, &cmd->current_cs.gfx->node);
      } else {
         list_addtail(&cs->node, &cmd->control_streams);
      }

      *ptr = cs;

      hk_sync_stream_created(cmd, cs, ptr, compute);
   }

   assert(*ptr != NULL);
   if (compute && unlikely((*ptr)->query_reset_pending) &&
       !cmd->state.query_reset_recording)
      hk_flush_query_resets(cmd, *ptr);
   if (compute && cmd->state.as_build_recording)
      (*ptr)->as_build = true;
   /* Transfer entry points also reuse an open CDM stream, so note each use,
    * not just stream creation (hk_sync_note_xfer_write()). */
   if (compute && unlikely(cmd->sync.xfer_depth) && !cmd->sync.in_pass &&
       hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G17P)
      hk_sync_note_xfer_write(cmd, *ptr);
   return *ptr;
}

static inline struct hk_cs *
hk_cmd_buffer_get_cs(struct hk_cmd_buffer *cmd, bool compute)
{
   struct hk_cs **ptr = compute ? &cmd->current_cs.cs : &cmd->current_cs.gfx;
   return hk_cmd_buffer_get_cs_general(cmd, ptr, compute);
}

void hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            size_t space);

static inline uint64_t
hk_cs_alloc_for_indirect(struct hk_cs *cs, size_t size_B)
{
   hk_ensure_cs_has_space(cs->cmd, cs, size_B);

   uint64_t addr = hk_cs_current_addr(cs);
   cs->current += size_B;
   return addr;
}

static void
hk_cmd_buffer_dirty_all(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   vk_dynamic_graphics_state_dirty_all(dyn);
   gfx->dirty = ~0;
   gfx->shaders_dirty = ~0;
   gfx->linked_dirty = ~0;
   gfx->descriptors.root_dirty = true;
}

static inline void
hk_cs_destroy(struct hk_cs *cs)
{
   util_dynarray_fini(&cs->compute_resources);
   util_dynarray_fini(&cs->render_resources);
   if (cs->type == HK_CS_VDM) {
      util_dynarray_fini(&cs->scissor);
      util_dynarray_fini(&cs->depth_bias);
   } else {
      util_dynarray_fini(&cs->imm_writes);
      util_dynarray_fini(&cs->oq_stamps);
   }

   free(cs);
}

void hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
bool hk_defer_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_resolve_deferred_oq_available(struct hk_cmd_buffer *cmd);

bool hk_g17p_smooth_varyings_supported(
   struct hk_cmd_buffer *cmd, const struct hk_linked_shader *linked);

void *hk_cmd_buffer_emit_vdm_terminate(struct hk_cmd_buffer *cmd,
                                       void *current);

static void
hk_cmd_buffer_end_compute_internal(struct hk_cmd_buffer *cmd,
                                   struct hk_cs **ptr)
{
   if (*ptr) {
      struct hk_cs *cs = *ptr;
      if (unlikely(cs->query_reset_pending))
         hk_flush_query_resets(cmd, cs);

      /* This control stream may write immediates as it ends. Queue the writes
       * now that we're done emitting everything else. Occlusion availability
       * alone may instead wait for the end of the command buffer, which
       * keeps the (still empty) stream in the list.
       */
      if (cs->imm_writes.size) {
         if (hk_defer_oq_available(cmd, cs)) {
            *ptr = NULL;
            return;
         }
         hk_dispatch_imm_writes(cmd, cs);
      }

      /* A transformed draw may reserve a pre-graphics stream and then prove
       * that it has no vertices to process. Do not leave an empty CDM job in
       * the submission list. A stream that only carries a timestamp stays:
       * that is the one job the stamp can ride on. */
      if (cs->stats.cmds == 0 && cs->stats.flushes == 0 &&
          !cs->timestamp.end.handle && !cs->timestamp.start.handle) {
         hk_sync_forget_stream(cmd, cs);
         list_del(&cs->node);
         hk_cs_destroy(cs);
      }
   }

   *ptr = NULL;
}

static void
hk_cmd_buffer_end_compute(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.cs);
}

/* Return true when the VDM was replaced/removed, false when it needs normal
 * graphics finalization (for example, a native background clear). */
bool hk_optimize_empty_vdm(struct hk_cmd_buffer *cmd);

/* Return whether a physical render was finalized (empty renders can vanish). */
static bool
hk_cmd_buffer_end_graphics(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;

   if (cs && cs->stats.cmds == 0 && hk_optimize_empty_vdm(cmd))
      cs = NULL;

   if (cs) {
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      const bool full_usc_state = agx_native_graphics(&dev->dev);
      /* Scissor and depth bias arrays are staged to dynamic arrays on the
       * CPU. When we end the control stream, they're done growing and are
       * ready for upload. G17P's ISP consumes these as full bit-40-set GART
       * addresses, matching native's USC-window allocations; a main-heap
       * address has bit 39 set and incorrectly selects the upper root.
       */
      cs->uploaded_scissor =
         full_usc_state
            ? hk_pool_usc_upload(cmd, cs->scissor.data, cs->scissor.size, 64)
            : hk_pool_upload(cmd, cs->scissor.data, cs->scissor.size, 64);

      cs->uploaded_zbias =
         full_usc_state
            ? hk_pool_usc_upload(cmd, cs->depth_bias.data,
                                 cs->depth_bias.size, 64)
            : hk_pool_upload(cmd, cs->depth_bias.data,
                             cs->depth_bias.size, 64);

      /* TODO: maybe free scissor/depth_bias now? */
      cs->current = hk_cmd_buffer_emit_vdm_terminate(cmd, cs->current);
   }

   cmd->current_cs.gfx = NULL;

   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.pre_gfx);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   assert(cmd->current_cs.pre_gfx == NULL);
   assert(cmd->current_cs.post_gfx == NULL);

   /* We just flushed out the heap use. If we want to use it again, we'll need
    * to queue a free for it again.
    */
   cmd->uses_heap = false;
   return cs != NULL;
}

static inline uint64_t
hk_pipeline_stat_addr(struct hk_cmd_buffer *cmd,
                      VkQueryPipelineStatisticFlagBits stat)
{
   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   VkQueryPipelineStatisticFlags flags = root->draw.pipeline_stats_flags;

   if (flags & stat) {
      assert(!cmd->in_meta && "queries paused for meta");
      assert(util_bitcount(stat) == 1 && "by construction");

      uint32_t index;
      const bool present =
         hk_pipeline_statistics_report_index(flags, stat, &index);
      assert(present && "enabled statistic has a compact report index");

      return root->draw.pipeline_stats + (sizeof(uint64_t) * index);
   } else {
      /* Query disabled */
      return AGX_SCRATCH_PAGE_ADDRESS;
   }
}

static inline bool
hk_stat_enabled(uint64_t addr)
{
   return addr != AGX_SCRATCH_PAGE_ADDRESS;
}

/* AGX3 render pass continuation (hk_cmd_draw.c): a pipeline barrier recorded
 * while an ended instance is still continuable decides whether the next one
 * may continue in its stream. */
void hk_m5_pass_merge_barrier(struct hk_cmd_buffer *cmd,
                              const VkDependencyInfo *info);

/* An ordered continuation (hk_cs::m5_ordered) ends at a command that may
 * break its memory equivalence (reason names the command). */
void hk_m5_pass_merge_end_ordered(struct hk_cmd_buffer *cmd,
                                  const char *reason);

/* Graphics descriptor updates while an ordered continuation is open, and
 * bookkeeping of the images graphics push descriptors reference. */
void hk_m5_pass_merge_note_push(struct hk_cmd_buffer *cmd,
                                struct hk_image_view *iview);
void hk_m5_pass_merge_note_set_bind(struct hk_cmd_buffer *cmd);

/* The vertex stage program of record draw_id > 0 of a multi-draw
 * (hk_cmd_buffer.c): only the draw parameters and draw index may differ from
 * the program the previous record of the same multi-draw published. */
uint64_t hk_m5_multi_draw_vertex_program(struct hk_cmd_buffer *cmd,
                                         struct hk_shader *s,
                                         struct hk_linked_shader *linked);

void hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                                  const VkCommandBufferBeginInfo *pBeginInfo);
void hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                                 const VkCommandBufferBeginInfo *pBeginInfo);

void hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd);
void hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd);

void hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                         const mesa_shader_stage *stages,
                         struct vk_shader **const shaders);

void hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                                 const mesa_shader_stage stage,
                                 struct hk_api_shader *shader);

void hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                                struct hk_api_shader *shader);

void hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                               struct hk_addr_range addr_range);

static inline struct hk_descriptor_state *
hk_get_descriptors_state(struct hk_cmd_buffer *cmd,
                         VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmd->state.gfx.descriptors;
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
      return &cmd->state.rt.descriptors;
   default:
      UNREACHABLE("Unhandled bind point");
   }
};

void hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                          struct hk_descriptor_state *desc);

void hk_meta_resolve_rendering(struct hk_cmd_buffer *cmd,
                               const VkRenderingInfo *pRenderingInfo);

void hk_cmd_upload_draw_params(struct hk_cmd_buffer *cmd, uint32_t first,
                               uint32_t first_instance);
void hk_cmd_upload_draw_id(struct hk_cmd_buffer *cmd, uint32_t draw_id);
uint64_t hk_cmd_upload_constants(struct hk_cmd_buffer *cmd,
                                 const struct hk_shader *shader,
                                 mesa_shader_stage stage);
void hk_cmd_print_upload_stats(struct hk_cmd_buffer *cmd);

uint64_t hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                                   VkPipelineBindPoint bind_point);

void hk_reserve_scratch_slow(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                             struct hk_shader *s);

static inline void
hk_reserve_scratch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct hk_shader *s)
{
   /* Not scratch but this is the most convenient place for this... */
   cs->uses_sampler_heap |= s->b.info.uses_sampler_heap;

   /* G17P and M5 publish a shader's spill frame as thread-private storage in their
    * SPECTPR. There is no helper-program stack, and the kernel rejects a
    * command that names one. */
   if ((s->b.info.scratch_size == 0 && s->b.info.preamble_scratch_size == 0) ||
       agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
       agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip))
      return;

   hk_reserve_scratch_slow(cmd, cs, s);
}

/* Whether image barriers of the command buffer order only the vertex work
 * that reads images (hk_sync.h, HK_SYNC_GFX_IMG). */
static inline bool
hk_sync_image_class(const struct hk_cmd_buffer *cmd)
{
   const enum agx_chip chip = hk_cmd_buffer_device(cmd)->dev.chip;
   return chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip);
}

/* AGX2 returns its shader-base-relative USC offset.  G17P returns the absolute
 * 48-bit Ioto program VA consumed by HAL200 VDM state. */
struct agx3_ioto_uniform_range;
void hk_m5_rt_state_diag_record(struct hk_cmd_buffer *cmd, struct agx_ptr state,
                                uint64_t size);

bool hk_m5_compute_uniform_dma_ranges(
   struct hk_cmd_buffer *cmd, const struct hk_shader *s, uint64_t root_ptr,
   struct agx3_ioto_uniform_range *ranges, size_t capacity, size_t *count);

uint64_t hk_g17p_dma_set_base(const struct hk_cmd_buffer *cmd,
                              const struct hk_shader *s);

struct agx3_ioto_uniform_range;
bool hk_g17p_dma_ranges(const struct hk_descriptor_state *desc,
                        const struct hk_shader *s, uint64_t root_ptr,
                        struct agx3_ioto_uniform_range *ranges,
                        size_t max_ranges, size_t *count_io,
                        uint8_t *sources);

uint64_t hk_upload_native_mesh_program(struct hk_cmd_buffer *cmd,
                                      struct hk_shader *s,
                                      struct hk_linked_shader *linked, bool object,
                                      bool task);
uint64_t hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked);

void hk_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                                    struct hk_cmd_buffer *cmd);

void hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs);

/* As hk_cdm_cache_flush(), for a flush an application dependency asks for.
 * With `memory`, the flush also has memory scope on G17P (M5 flushes always
 * do): needed for host availability, and for device dependencies once the
 * same memory may be reached through more than one virtual address
 * (hk_cdm_alias_scope()). A trailing strong flush without memory scope is
 * upgraded in place. The caller ensures space for one barrier record. */
void hk_cdm_visibility_flush(struct hk_device *dev, struct hk_cs *cs,
                             bool memory);

/* Whether hk_cdm_visibility_flush(dev, cs, memory) would record anything. */
bool hk_cdm_visibility_flush_needed(const struct hk_device *dev,
                                    const struct hk_cs *cs, bool memory);

void hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                                 struct agx_cdm_launch_word_0_packed launch,
                                 uint32_t usc, struct agx_grid grid,
                                 struct agx_workgroup local_size);

void hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                          struct agx_shader_info *info, uint32_t usc,
                          struct agx_grid grid,
                          struct agx_workgroup local_size);

void hk_dispatch_m5_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            struct hk_shader *s, struct agx_grid grid,
                            struct agx_workgroup local_size);

void hk_dispatch_g17p_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                             struct hk_shader *s, struct agx_grid grid,
                             struct agx_workgroup local_size);

bool hk_gen1_upload_fragment_query_state(struct hk_cmd_buffer *cmd, bool vertex_stage,
   struct agx3_ioto_uniform_range *range);

void hk_g17p_prewarm_linked_compute_shader(struct hk_device *dev,
                                            struct hk_shader *s,
                                            struct hk_linked_shader *linked);
void hk_g17p_linked_compute_cache_finish(struct hk_linked_shader *linked);
void hk_g17p_linked_graphics_cache_finish(struct hk_device *dev,
                                          struct hk_linked_shader *linked);

/*
 * Saved application state around a driver-internal meta operation.  The
 * acceleration-structure builder needs this as much as a blit does: the
 * common build shaders bind their own compute pipeline and push constants,
 * so a build recorded between two application dispatches would otherwise
 * silently clobber them.
 */
struct hk_meta_save {
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
   struct vk_dynamic_graphics_state dynamic;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];
   struct hk_addr_range vb0;
   struct hk_descriptor_set *desc0;
   /* Nested meta must retain the outer query/conditional-rendering scope. */
   bool in_meta;
   bool has_push_desc0;
   bool push_active0;
   enum agx_visibility_mode occlusion;
   struct hk_push_descriptor_set push_desc0;
   VkQueryPipelineStatisticFlags pipeline_stats_flags;
   uint8_t push[HK_MAX_PUSH_SIZE];
};

void hk_meta_begin(struct hk_cmd_buffer *cmd, struct hk_meta_save *save,
                   VkPipelineBindPoint bind_point);
void hk_meta_end(struct hk_cmd_buffer *cmd, struct hk_meta_save *save,
                 VkPipelineBindPoint bind_point);

void hk_cmd_dispatch_rays(struct hk_cmd_buffer *cmd,
                          struct hk_api_shader *shader, uint64_t record_addr,
                          uint64_t group_count_addr, struct agx_grid grid);

void hk_cmd_dispatch_unaligned(VkCommandBuffer commandBuffer,
                               uint32_t invocations_x, uint32_t invocations_y,
                               uint32_t invocations_z);

/* Driver-owned, resident indirect arguments addressed through their primary
 * GPU VA. Sparse application buffers must use the public entrypoint. */
void hk_cmd_dispatch_indirect_addr(VkCommandBuffer command_buffer,
                                   VkDeviceAddress address);

static inline void
hk_dispatch_with_local_size(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            struct hk_shader *s, struct agx_grid grid,
                            struct agx_workgroup local_size)
{
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (agx_chip_g17_m5(dev->dev.chip)) {
      hk_dispatch_m5_native(cmd, cs, s, grid, local_size);
      return;
   }
   if (agx_native_graphics(&dev->dev)) {
      hk_reserve_scratch(cmd, cs, s);
      hk_dispatch_g17p_native(cmd, cs, s, grid, local_size);
      /* Software graphics stages communicate through memory: VS outputs,
       * geometry counts and XFB bases are consumed by the next dispatch.
       * The application cannot supply barriers between these internal
       * stages. Preserve the AGX2 flush-after-launch contract, including
       * indirect launches, by strengthening the native ordering barrier. */
      if (s->info.stage != MESA_SHADER_COMPUTE) {
         hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
         hk_cdm_cache_flush(dev, cs);
      }
      return;
   }

   uint64_t uploaded = hk_upload_usc_words(cmd, s, s->only_linked);
   if (uploaded > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t usc = uploaded;

   hk_reserve_scratch(cmd, cs, s);
   hk_dispatch_with_usc(dev, cs, &s->b.info, usc, grid, local_size);
}

void hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         void *data, size_t data_size);
void hk_dispatch_precomp_m5(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                            enum agx_barrier barrier, enum libagx_program idx,
                            const void *data, size_t data_size);
void hk_dispatch_precomp_g17p(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                              enum agx_barrier barrier,
                              enum libagx_program idx, const void *data,
                              size_t data_size);

/* Turn a three/four/five-word indirect argument record into an execution-time
 * conditional record. `barrier` selects the ordered compute stream which must
 * precede its consumer. */
uint64_t hk_apply_conditional_rendering(struct hk_cmd_buffer *cmd,
                                        uint64_t source, uint32_t words,
                                        enum agx_barrier barrier);

#define MESA_DISPATCH_PRECOMP hk_dispatch_precomp

void hk_queue_write(struct hk_cmd_buffer *cmd, uint64_t address, uint32_t value,
                    bool after_gfx);
bool hk_ensure_occlusion_query_for_draw(struct hk_cmd_buffer *cmd);

/* Dispatch the native G17P raw-address fill meta shader into an explicitly
 * selected compute stream.  The target may be the ordinary or post-graphics
 * stream; the latter preserves query-availability ordering. */
void hk_g17p_meta_fill_addr(struct hk_cmd_buffer *cmd, struct hk_cs **target,
                            uint64_t address, uint64_t size, uint32_t value);

/* Poll a naturally aligned 32-bit device address from a native G17P compute
 * stream until it reaches value. */
void hk_g17p_meta_wait_addr(struct hk_cmd_buffer *cmd, uint64_t address,
                            uint32_t value);

void agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                         uint64_t attribs_read, struct agx_velem_key *keys);

struct agx_robustness hk_prolog_robustness(struct hk_device *dev);
