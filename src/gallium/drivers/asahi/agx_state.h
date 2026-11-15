/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019-2021 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <xf86drm.h>
#include "asahi/compiler/agx_compile.h"
#include "asahi/genxml/agx_pack.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "asahi/lib/agx_scratch.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/lib/agx_uvs.h"
#include "asahi/lib/pool.h"
#include "compiler/shader_enums.h"
#include "gallium/auxiliary/util/u_blitter.h"
#include "gallium/include/pipe/p_context.h"
#include "gallium/include/pipe/p_screen.h"
#include "gallium/include/pipe/p_state.h"
#include "pipe/p_defines.h"
#include "poly/geometry.h"
#include "poly/nir/poly_nir.h"
#include "util/bitset.h"
#include "util/disk_cache.h"
#include "util/hash_table.h"
#include "util/rwlock.h"
#include "util/set.h"
#include "util/u_atomic.h"
#include "util/u_range.h"
#include "agx_bg_eot.h"
#include "agx_helpers.h"
#include "agx_native_descriptor.h"
#include "agx_nir_texture.h"

/* Diagnostic environment switches. Some are tested on every draw or upload,
 * so each is read from the environment once and cached; getenv() scans the
 * whole environment on every call.
 */
enum agx_env_flag {
   AGX_ENV_NEO_DRAW_DIAGNOSTICS,
   AGX_ENV_NEO_UNIFORM_DIAGNOSTICS,
   AGX_ENV_NEO_BINARY_DIAGNOSTICS,
   AGX_ENV_G18_RESOURCE_TRACE,
   AGX_ENV_NO_STAGED_BUFFER_WRITES,
   AGX_ENV_NO_M5_PROGRAM_CACHE,
   AGX_ENV_M5_VERIFY_PROGRAM_CACHE,
   AGX_ENV_FLAG_COUNT,
};

/* -1 until first read, then 0 or 1. */
extern int agx_env_flags[AGX_ENV_FLAG_COUNT];

bool agx_env_flag_init(enum agx_env_flag flag);

static inline bool
agx_env_flag(enum agx_env_flag flag)
{
   int v = p_atomic_read_relaxed(&agx_env_flags[flag]);

   return likely(v >= 0) ? v : agx_env_flag_init(flag);
}

#ifdef __GLIBC__
#include <errno.h>
#define agx_msg(fmt, ...)                                                      \
   fprintf(stderr, "[%s] " fmt, program_invocation_short_name, ##__VA_ARGS__)
#else
#define agx_msg(...) fprintf(stderr, __VA_ARGS__)
#endif

/* The CPU-wait fallbacks for failed fence imports and merges block without a
 * timeout: log each one so a stalled process shows where it is waiting.
 */
#define agx_fallback_wait(fd, why)                                                (agx_msg("CPU wait on a fence: %s\n", why), sync_wait(fd, -1))

#define AGX_NUM_TEXTURE_STATE_REGS 16

struct agx_streamout_target {
   struct pipe_stream_output_target base;
   struct pipe_resource *offset;

   /* Current stride (bytes per vertex) */
   uint32_t stride;
};

static inline struct agx_streamout_target *
agx_so_target(struct pipe_stream_output_target *target)
{
   return (struct agx_streamout_target *)target;
}

struct agx_streamout {
   struct pipe_stream_output_target *targets[PIPE_MAX_SO_BUFFERS];
   unsigned num_targets;
};

/* Shaders can access fixed-function state through system values.
 * It is convenient to stash all of this information into a single "root"
 * descriptor, then push individual parts as needed.
 *
 * In the future, we could optimize this to reduce CPU overhead, e.g. splitting
 * into multiple descriptors for finer dirty tracking. This is not ABI with the
 * compiler. The layout is up to us and handled by our code lowering system
 * values to uniforms.
 */
enum agx_sysval_table {
   AGX_SYSVAL_TABLE_ROOT,
   AGX_SYSVAL_TABLE_PARAMS,
   AGX_SYSVAL_TABLE_GRID,
   AGX_SYSVAL_TABLE_VS,
   AGX_SYSVAL_TABLE_TCS,
   AGX_SYSVAL_TABLE_TES,
   AGX_SYSVAL_TABLE_GS,
   AGX_SYSVAL_TABLE_FS,
   AGX_SYSVAL_TABLE_CS,
   AGX_NUM_SYSVAL_TABLES
};

#define AGX_SYSVAL_STAGE(stage) (AGX_SYSVAL_TABLE_VS + (stage))

static_assert(AGX_SYSVAL_STAGE(MESA_SHADER_VERTEX) == AGX_SYSVAL_TABLE_VS,
              "fixed enum orderings");
static_assert(AGX_SYSVAL_STAGE(MESA_SHADER_TESS_CTRL) == AGX_SYSVAL_TABLE_TCS,
              "fixed enum orderings");
static_assert(AGX_SYSVAL_STAGE(MESA_SHADER_TESS_EVAL) == AGX_SYSVAL_TABLE_TES,
              "fixed enum orderings");
static_assert(AGX_SYSVAL_STAGE(MESA_SHADER_GEOMETRY) == AGX_SYSVAL_TABLE_GS,
              "fixed enum orderings");
static_assert(AGX_SYSVAL_STAGE(MESA_SHADER_FRAGMENT) == AGX_SYSVAL_TABLE_FS,
              "fixed enum orderings");
static_assert(AGX_SYSVAL_STAGE(MESA_SHADER_COMPUTE) == AGX_SYSVAL_TABLE_CS,
              "fixed enum orderings");

/* Root system value table */
struct PACKED agx_draw_uniforms {
   /* Pointers to the system value tables themselves (for indirection) */
   uint64_t tables[AGX_NUM_SYSVAL_TABLES];

   /* Vertex buffer object bases, if present. If vertex robustness is disabled,
    * attrib_base maps VBOs directly and attrib_max_index is undefined. If
    * vertex robustness is enabled, attrib_base maps attributes and
    * attrib_clamp is an inclusive clamp on vertex/divided instance indices.
    */
   uint64_t attrib_base[PIPE_MAX_ATTRIBS];
   uint32_t attrib_clamp[PIPE_MAX_ATTRIBS];

   /* Addresses for the results of pipeline statistics queries */
   uint64_t pipeline_statistics[PIPE_STAT_QUERY_MS_INVOCATIONS];

   /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
   uint64_t vertex_outputs;

   /* Address of input assembly buffer if geom/tess is used, else 0 */
   uint64_t vertex_params;

   /* Address of tessellation param buffer if tessellation is used, else 0 */
   uint64_t tess_params;

   /* Address of geometry param buffer if geometry shaders are used, else 0 */
   uint64_t geometry_params;

   /* Address of polygon stipple mask if used */
   uint64_t polygon_stipple;

   /* Blend constant if any */
   float blend_constant[4];

   /* glPointSize value */
   float fixed_point_size;

   /* Value of the multisample control register, containing sample positions in
    * each byte (x in low nibble, y in high nibble).
    */
   uint32_t ppp_multisamplectl;

   /* gl_DrawID for a direct multidraw */
   uint32_t draw_id;

   /* Sprite coord replacement mask */
   uint16_t sprite_mask;

   /* glSampleMask */
   uint16_t sample_mask;

   /* Nonzero for indexed draws, zero otherwise */
   uint16_t is_indexed_draw;

   /* Zero for [0, 1] clipping, 0.5 for [-1, 1] clipping. */
   uint16_t clip_z_coeff;

   /* ~0/0 boolean whether the epilog lacks any discard instrction */
   uint16_t no_epilog_discard;

   /* Provoking vertex: 0, 1, 2 */
   uint16_t provoking_vertex;

   /* G17G: the layer a split layered draw is bound to (agx_g17g_bind_layer),
    * which a fragment gl_Layer read returns. */
   uint16_t layer_id;

   /* Mapping from varying slots written by the last vertex stage to UVS
    * indices. This mapping must be compatible with the fragment shader.
    */
   uint16_t uvs_index[VARYING_SLOT_MAX];
};

struct PACKED agx_stage_uniforms {
   /* Pointer to binding table for texture descriptor, or 0 if none. This must
    * be first so that u0_u1 is always available for lowering binding
    * tables to bindless access.
    */
   uint64_t texture_base;
   uint64_t buffer_texture_base;

   /* Uniform buffer objects */
   uint64_t ubo_base[PIPE_MAX_CONSTANT_BUFFERS];
   uint32_t ubo_size[PIPE_MAX_CONSTANT_BUFFERS];

   /* Shader storage buffer objects */
   uint64_t ssbo_base[PIPE_MAX_SHADER_BUFFERS];
   uint32_t ssbo_size[PIPE_MAX_SHADER_BUFFERS];

   /* Gen9 has no AGX2 buffer-image size sideband in its native descriptor. */
   uint32_t image_buffer_size[PIPE_MAX_SHADER_IMAGES];

   /* If lowered to bindless, sampler index in the heap */
   uint16_t sampler_handle[PIPE_MAX_SAMPLERS];

   /* HAL200 register-indirect texture operands name local table slots. */
   uint16_t texture_handle[PIPE_MAX_SAMPLERS];

   /* G17P custom-border emulation uses a clamp-to-zero companion sampler.
    * Negative means that the bound sampler has a native standard border. */
   int16_t custom_border_sampler_handle[PIPE_MAX_SAMPLERS];

   /* Logical API colour returned after texture swizzling. */
   uint32_t custom_border[PIPE_MAX_SAMPLERS][4];

   /* LOD bias as float16 */
   uint16_t lod_bias[PIPE_MAX_SAMPLERS];

   /* G16G/G17P shadow-gather border comparison (see
    * agx_shadow_gather_custom_border), in pipe_compare_func order. Kept
    * after the existing fields so classic shader bindings do not move. Each
    * halfword has padding for Ioto's dword upload, including sampler 31. */
   uint32_t custom_border_compare_func[PIPE_MAX_SAMPLERS];

   /* G18G-only fields follow, after every other chip's fields so that their
    * uniform offsets do not move. */

   /* G18G custom-border shadow comparison: PIPE_FUNC_* in bits 0..2; bit 3
    * clamps a fixed-point depth reference. */
   uint16_t custom_border_compare[PIPE_MAX_SAMPLERS];

   /* Native texture descriptors do not carry the legacy software buffer
    * size. Keep the bound view's texel count separate from hardware state. */
   uint32_t texture_buffer_size[PIPE_MAX_SHADER_SAMPLER_VIEWS];
   uint32_t texture_buffer_rgb32[PIPE_MAX_SHADER_SAMPLER_VIEWS];
   uint64_t texture_buffer_address[PIPE_MAX_SHADER_SAMPLER_VIEWS];

   /* G18 bound raw image addresses: base lo/hi, linear row pitch B, tiled
    * stride el, tile width/height el, layer stride B lo/hi, sample count,
    * actual bound-view texel size B. */
   uint32_t image_address_metadata[PIPE_MAX_SHADER_IMAGES][10];
};

#define AGX_BORDER_COMPARE_CLAMP_REF BITFIELD_BIT(8)

/* In the architecture, there are 512 uniform registers, each 16-bits. In a
 * theoretical worst case, we could push to all of them. We use a worst-case
 * maximum because the expression for a tight upper bound is too messy and easy
 * to go out of sync with the code.
 */
#define AGX_MAX_PUSH_RANGES (512)

struct agx_push_range {
   /* Base 16-bit uniform to push to */
   uint16_t uniform;

   /* Offset into the table to push in bytes */
   uint16_t offset;

   /* Which table to push from */
   uint8_t table;

   /* Number of consecutive 16-bit uniforms to push */
   uint8_t length;
};

struct agx_compiled_shader {
   /* Base struct */
   struct agx_shader_part b;

   /* Uncompiled shader that we belong to */
   const struct agx_uncompiled_shader *so;

   /* Mapped executable memory */
   struct agx_bo *bo;

   /* The backend refused to encode this variant. agx_compile_shader_nir() is
    * fail-closed: it returns false and leaves a zeroed agx_shader_part, so
    * binary_size is 0 and no executable BO is ever allocated. A main shader
    * in that state must never reach a draw or a dispatch -- consuming it
    * dereferences a null bo and takes the process down, which for a
    * compositor means the whole desktop rather than one degraded draw.
    *
    * This is NOT the same as `bo == NULL`. A fragment main whose work all
    * lives in the epilog legitimately has an empty binary, and prologs,
    * epilogs and other secondary parts never own a BO at all.
    */
   bool failed;

   /* Set once the draw/dispatch path has reported that this program is
    * unusable, so a refused shader logs once instead of once per draw.
    */
   bool refusal_reported;

   /* G17P graphics and compute programs reserve uniform halves 0..7 for the
    * native texture and sampler table pointers when they access resources. */
   bool g17p_texture_resources;

   /* Exact G17P compute write-only image protocol: PBE pointer and native
    * write-control record occupy uniform halves 0..15. */
   bool g17p_storage_write;

   /* The variant contains Honeykrisp-style custom-border emulation. */
   bool g17p_custom_border_emulation;

   /* G17G: the fragment runs per sample by repeating its body once per
    * sample, each with its own colour exports, and its epilog repeats the
    * same way; the link adds no external sample loop. */
   bool g17g_samples_unrolled;

   /* Uniforms the driver must push */
   unsigned push_range_count;
   struct agx_push_range push[AGX_MAX_PUSH_RANGES];

   /* UVS layout for the last vertex stage */
   struct agx_unlinked_uvs_layout uvs;

   /* For a vertex shader, the mask of vertex attributes read. Used to key the
    * prolog so the prolog doesn't write components not actually read.
    */
   BITSET_DECLARE(attrib_components_read, AGX_MAX_ATTRIBS * 4);

   struct agx_fs_epilog_link_info epilog_key;

   /* Auxiliary programs, or NULL if not used */
   struct agx_compiled_shader *gs_count, *pre_gs;
   struct agx_compiled_shader *gs_copy;

   struct poly_gs_info gs;

   /* Logical shader stage used for descriptor access. This may differ from the
    * physical shader stage of the compiled shader, for example when executing a
    * tessellation eval shader as a vertex shader.
    */
   mesa_shader_stage stage;
};

struct agx_fast_link_key {
   union {
      struct agx_vs_prolog_key vs;
      struct agx_fs_prolog_key fs;
   } prolog;

   struct agx_compiled_shader *main;

   union {
      struct agx_fs_epilog_key fs;
   } epilog;

   unsigned nr_samples_shaded;
};

struct agx_uncompiled_shader {
   struct pipe_shader_state base;
   mesa_shader_stage type;
   /* An application shader: API NIR that is not marked internal. Only these
    * report compiler statistics through the GL debug callback (see
    * agx_compile_variant). */
   bool report_shader_info;
   struct blob early_serialized_nir;
   struct blob serialized_nir;
   uint8_t nir_blake3[BLAKE3_KEY_LEN];

   struct {
      uint64_t inputs_flat_shaded;
      uint64_t inputs_linear_shaded;
      /* Every varying slot a fragment shader reads. */
      uint64_t inputs_read;
      uint8_t cull_distance_size;
      bool has_edgeflags;
      bool uses_fbfetch;
      bool uses_sample_shading;
      /* The fragment shader reads the sample count (samples_log2), which
       * G17G/G18G specialize at compile time as Apple's pipelines do. */
      bool needs_sample_count;
      bool uses_buffer_textures;
      bool uses_query_lod;
      bool writes_memory;
      /* API output, excluding synthetic early depth/stencil mask updates. */
      bool writes_api_sample_mask;

      /* Some image intrinsic other than a read or a size/sample query:
       * stores, atomics, texel addresses and anything not known to only
       * read. G18G gives an image PBE only to such shaders. */
      bool writes_images;

      /* Some image intrinsic other than a read, a query or a store: the
       * atomics and texel addresses whose libagx lowering reads the PBE's
       * AGX2 software sideband. A G18G shader without one writes its
       * images with native stores alone. */
      bool reads_image_sideband;

      /* Bound images whose global atomics/stores use software view metadata. */
      uint64_t image_address_mask;

      /* Bound image descriptors actually consumed as PBEs after lowering
       * global stores/atomics. Dynamic consumers conservatively mark all. */
      uint64_t image_pbe_mask;

      /* Number of bindful textures, images used */
      unsigned nr_bindful_textures, nr_bindful_images;

      /* Texture-table records per image: a texture and a PBE descriptor,
       * plus on G17P the software tail Honeykrisp's storage descriptor
       * carries (see AGX_G17P_IMAGE_ATOMIC_SW_OFFSET).
       */
      uint8_t image_records;

   } info;

   struct hash_table *variants;
   struct agx_uncompiled_shader *passthrough_progs[MESA_PRIM_COUNT][3][2];
   struct agx_uncompiled_shader *passthrough_tcs[32];

   /* agx_fast_link_key -> agx_linked_shader */
   struct hash_table *linked_shaders;

   uint32_t xfb_strides[4];
   bool has_xfb_info;
   bool is_xfb_passthrough;

   /* G17G: this vertex shader for the draw module, which captures transform
    * feedback on the CPU (agx_xfb_cpu.c). */
   struct draw_vertex_shader *xfb_cpu_vs;

   /* G17G: this shader for the draw module running the stages before
    * rasterization on the CPU (agx_swtnl.c), created on first use. */
   void *swtnl_shader;
   bool swtnl_failed;

   /* G17G fragment shaders: the varyings each split MRT pass's program still
    * reads once its other colour outputs are gone (agx_g17g_split_mrt);
    * bit rt of g17g_mrt_inputs_valid marks g17g_mrt_inputs[rt] computed. */
   uint64_t g17g_mrt_inputs[PIPE_MAX_COLOR_BUFS];
   uint8_t g17g_mrt_inputs_valid;

   enum mesa_prim gs_mode;

   /* Whether the shader accesses indexed samplers via the bindless heap */
   bool uses_bindless_samplers;

   /* Set on VS, passed to FS for linkage */
   unsigned base_varying;

   /* Tessellation info */
   struct {
      uint64_t per_vertex_outputs;
      uint32_t output_stride;
      enum gl_tess_spacing spacing;
      enum tess_primitive_mode primitive;
      uint8_t output_patch_size;
      uint8_t nr_patch_outputs;
      bool ccw;
      bool point_mode;
   } tess;
};

enum agx_stage_dirty {
   AGX_STAGE_DIRTY_CONST = BITFIELD_BIT(0),
   AGX_STAGE_DIRTY_SSBO = BITFIELD_BIT(1),
   AGX_STAGE_DIRTY_IMAGE = BITFIELD_BIT(2),
   AGX_STAGE_DIRTY_SAMPLER = BITFIELD_BIT(3),
};

struct agx_stage {
   struct agx_uncompiled_shader *shader;
   uint32_t dirty;

   struct pipe_constant_buffer cb[PIPE_MAX_CONSTANT_BUFFERS];
   uint32_t cb_mask;

   struct pipe_shader_buffer ssbo[PIPE_MAX_SHADER_BUFFERS];
   uint32_t ssbo_writable_mask;
   uint32_t ssbo_mask;

   struct pipe_image_view images[PIPE_MAX_SHADER_IMAGES];
   uint64_t image_mask;

   /* Need full CSOs for u_blitter */
   struct agx_sampler_state *samplers[PIPE_MAX_SAMPLERS];
   struct agx_sampler_view *textures[PIPE_MAX_SHADER_SAMPLER_VIEWS];

   /* Does any bound sampler require custom border colours? */
   bool custom_borders;

   unsigned sampler_count, texture_count;
   uint32_t valid_samplers;
};

struct agx_timestamps {
   uint64_t vtx_start;
   uint64_t vtx_end;
   uint64_t frag_start;
   uint64_t frag_end;
   uint64_t comp_start;
   uint64_t comp_end;
};

/* This is a firmware limit. It should be possible to raise to 2048 in the
 * future... still not good enough for VK though :-(
 */
#define AGX_SAMPLER_HEAP_SIZE (1024)

struct agx_sampler_heap {
   struct agx_bo *bo;
   uint16_t count;
};

uint16_t agx_sampler_heap_add(struct agx_device *dev,
                              struct agx_sampler_heap *heap,
                              struct agx_sampler_packed *sampler);

struct agx_cdm_dispatch {
   uint32_t start, end;
   /* G18G: the command carries DRM_ASAHI_COMPUTE_USES_TEXTURE_STATE. */
   bool texture_state;
   struct drm_asahi_g18_compute_resources resources;
   /* CPU metadata for tracing and resolving empty indirect work after its
    * dependencies retire; never serialized into a GPU command. */
   uint32_t trace_mode, trace_global[3], trace_local[3], trace_program;
   uint64_t trace_indirect, trace_shader;
   uint64_t trace_tess_params, trace_vertex_params, trace_geometry_params;
};

struct agx_encoder {
   struct agx_bo *bo;
   uint64_t g17p_vdm_alias;
   uint8_t *current;
   uint8_t *end;
};

/* Primitives per G17G render; see agx_draw_vbo(). 128K tessellated
 * primitives in one render timed out the submission (GLES31
 * primitive_bounding_box.depth.user_defined_depth.per_primitive_bbox_equal,
 * two 64K-primitive draws); 64K passed. Keep a margin below that. */
#define AGX_G17G_RENDER_PRIMS (32 * 1024)

/* Primitives per G17G render of draws without a tessellation stage. 512K
 * small depth-tested triangles in one render draw the image the split
 * renders draw, at 1024x1024 and at 3456x2160, on an M5 Pro; more has not
 * been tried. */
#define AGX_G17G_RENDER_PRIMS_PLAIN (512 * 1024)

struct agx_batch {
   struct agx_context *ctx;
   struct pipe_framebuffer_state key;
   uint64_t seqnum;
   uint32_t syncobj;
   /* Failed waits retain these fences and all batch resources until exit. */
   uint32_t failed_syncobjs[2];
   bool submit_quarantined;
   uint32_t draws;

   /* Executable storage is owned by this batch's pipeline pool. */
   uint64_t g17p_fragment_program;
   uint64_t g17p_emrt_heap;
   struct agx_compiled_shader *g17p_fragment_shader;
   struct agx_linked_shader *g17p_fragment_linked;

   /* G17G: primitives drawn into this render (see AGX_G17G_RENDER_PRIMS) */
   uint64_t g17g_prims;
   /* G17G: this render holds the output of a tessellation stage and keeps
    * to AGX_G17G_RENDER_PRIMS. */
   bool g17g_small_budget;

   struct agx_tilebuffer_layout tilebuffer_layout;

   /* PIPE_CLEAR_* bitmask */
   uint32_t clear, draw, load, resolve, feedback;
   bool initialized;

   /* Only a loader-provided resolve may seed the multisample tile reload. */
   bool g17p_resolve_reload;
   bool g17p_resolve_preserve;

   /* Set when a descriptor or state in this batch has no proven native
    * encoding on the current chip. Nothing from the batch may reach the
    * hardware; agx_batch_submit drops it without a context reset.
    */
   bool native_rejected;

   /* M5 left this batch's compute part out of the submission (it had no
    * dispatch), so the GPU wrote no compute timestamps. */
   bool compute_dropped;

   /* Draws the G17G profile refused. A batch whose render carries none of its
    * draws has nothing valid to rasterize: its VDM stream never received
    * them, and submitting it anyway fails UAPI translation with a silent
    * EINVAL. A batch with no refusals and no draws is an ordinary clear and
    * still needs its render. */
   unsigned draws_refused;

   /* M5 on the G17 kernel runtime (agx_m5_render_resources): the storage
    * the programs of this batch need, sent ahead of its render and compute
    * commands. m5_render_programs holds one drm_asahi_render_program_v2 per
    * distinct linked vertex or fragment program demand, m5_tile_program the
    * background and end-of-tile programs' record, m5_compute_programs one
    * drm_asahi_compute_program per distinct dispatch demand. The kernel takes
    * the maximum over each list, so they are sets. */
   struct util_dynarray m5_render_programs;
   struct util_dynarray m5_compute_programs;
   struct drm_asahi_render_program m5_tile_program;

   /* A graphics program of this render writes memory (its kernel write
    * class, DRM_ASAHI_RENDER_SHADER_WRITES). */
   bool m5_shader_writes;

   uint64_t uploaded_clear_color[PIPE_MAX_COLOR_BUFS];
   union pipe_color_union clear_color[PIPE_MAX_COLOR_BUFS];
   double clear_depth;
   unsigned clear_stencil;

   /* Whether we're drawing points, lines, or triangles */
   enum mesa_prim reduced_prim;

   /* Whether the bound FS needs a primitive ID that is not supplied by the
    * bound hardware VS (software GS)
    */
   bool generate_primitive_id;

   /* Current varyings linkage structures */
   uint64_t varyings;
   struct agx_varyings_vs linked_varyings;

   struct agx_draw_uniforms uniforms;

   /* CPU snapshot: never read back the write-combined GPU root upload. */
   struct agx_draw_uniforms vs_root;
   bool vs_root_valid;
   struct agx_stage_uniforms stage_uniforms[MESA_SHADER_STAGES];

   /* Heap descriptor if dynamic allocation is required */
   uint64_t heap;

   /* Uploaded descriptors */
   uint32_t texture_count[MESA_SHADER_STAGES];

   uint64_t samplers[MESA_SHADER_STAGES];
   uint32_t sampler_count[MESA_SHADER_STAGES];

   struct agx_sampler_heap sampler_heap;

   /* Pointer membership for amortized O(1) insertion, plus a dense BO list
    * for submission and retirement. Neither allocation depends on GEM handles.
    */
   struct {
      struct set *set;
      struct util_dynarray list; /* struct agx_bo * */
   } bo_list;

   /* If true, this batch contains a shader with a potentially incoherent write
    * (e.g. image_write), needing a barrier later to access.
    */
   bool incoherent_writes;

   struct agx_pool pool, pipeline_pool;
   struct util_dynarray g17p_vdm_aliases;

   /* With AGX_DBG_FAULTTRACE, retain the small DRM command wrapper while this
    * batch is in flight. Its referenced BOs and encoders already remain live
    * until batch cleanup, allowing an asynchronous queue fault to be decoded
    * when a later submit observes it. */
   struct util_dynarray faulttrace_cmdbuf;
   uint64_t faulttrace_seqid;

   /* We may enqueue both CDM and VDM work, possibly to the same batch for
    * geometry/tessellation.
    */
   struct agx_encoder vdm;
   struct agx_encoder cdm;

   /* The native M5 compute profile covers one dispatch per command: its
    * captured control stream is a single launch, its grid, a barrier and the
    * terminator, and the descriptor carries one dispatch identity, one
    * submission index and one pair of dispatch stamps. A geometry-shader or
    * transform-feedback prepass issues several launches, so on AGX3c each one
    * is terminated where it ends and submitted as its own command, in order.
    * Byte offsets into cdm.bo; empty on every other part, where the whole
    * stream is submitted once. A batch can contain more launches than one
    * geometry prepass, so retain a record for every emitted dispatch. */
   struct util_dynarray cdm_dispatch; /* struct agx_cdm_dispatch */

   /* Scissor and depth-bias descriptors, uploaded at GPU time */
   struct util_dynarray scissor, depth_bias;

   /* Arrays of GPU pointers that should be written with the batch timestamps */
   struct util_dynarray timestamps;

   /* Actual pointer in a uniform */
   struct agx_bo *geom_params_bo, *geom_index_bo;
   uint64_t geom_index;

   /* Whether each stage uses scratch */
   bool vs_scratch;
   bool fs_scratch;
   bool cs_scratch;

   /* G18: program demand of every vertex/fragment/tile program this render
    * generated, reported in drm_asahi_g18_render_resources when any API
    * program carries a static TPS frame. */
   struct {
      struct drm_asahi_g18_render_resources resources;
      bool scratch;
   } g18;

   /* Whether each stage has preambles using scratch, and if so which bucket.
    * This just needs to be zero/nonzero for correctness, the magnitude in
    * buckets is for statistics.
    */
   unsigned vs_preamble_scratch;
   unsigned fs_preamble_scratch;
   unsigned cs_preamble_scratch;
};

struct agx_zsa {
   struct pipe_depth_stencil_alpha_state base;
   struct agx_fragment_face_packed depth;
   struct agx_fragment_stencil_packed front_stencil, back_stencil;

   /* PIPE_CLEAR_* bitmask corresponding to this depth/stencil state */
   uint32_t load, store;
};

struct agx_blend {
   struct agx_blend_key key;

   /* PIPE_CLEAR_* bitmask corresponding to this blend state */
   uint32_t store;
};

struct asahi_vs_shader_key {
   /* If true, this is running as a hardware vertex shader. If false, this is a
    * compute job used to feed a TCS or GS.
    */
   bool hw;
   bool custom_borders;
   /* G18's native line state has no rasterizer point-size output. */
   bool omit_point_size;
   /* Export Layer as a flat input only when the fragment shader reads it. */
   bool layer_as_varying;

   /* G17G has no UVS store with a register index, so its vertex shaders
    * take the UVS layout linked against the bound fragment shader's
    * interpolation as constants rather than as uniforms (separable
    * programs do not know it at compile time). */
   bool g17g_link_uvs;
   uint64_t flat_varyings;
   uint64_t linear_varyings;

   /* Nonzero: export only these generic varyings (VARYING_SLOT_VAR0 and
    * up), the ones a split MRT pass's fragment program reads. */
   uint64_t g17g_fs_inputs;
};

struct agx_vertex_elements {
   unsigned num_attribs;
   struct agx_velem_key key[PIPE_MAX_ATTRIBS];

   /* These parts do not affect the generated code so are not in the key */
   uint16_t src_offsets[PIPE_MAX_ATTRIBS];
   uint16_t buffers[PIPE_MAX_ATTRIBS];

   /* The elements as bound, for the draw module (agx_xfb_cpu.c). */
   struct pipe_vertex_element pipe[PIPE_MAX_ATTRIBS];
};

struct asahi_fs_shader_key {
   enum pipe_format rt_formats[PIPE_MAX_COLOR_BUFS];
   uint8_t nr_samples;
   bool custom_borders;
   /* G17G: keep colour output g17g_mrt_output - 1 alone, at location 0
    * (agx_g17g_split_mrt). Zero keeps every output. */
   uint8_t g17g_mrt_output;
   bool padding[5];
};
static_assert(sizeof(struct asahi_fs_shader_key) == 40, "no holes");

struct asahi_cs_shader_key {
   bool custom_borders;
};

/* The geometry copy program is a hardware vertex shader compiled before the
 * fragment shader is known, so it normally reads its UVS indices from
 * uniforms. AGX3's UVSW takes only an immediate index (every native G17P and
 * G17G vertex program), so on those chips the copy program is compiled per
 * fragment interpolation layout: the same flat/linear masks the draw-time
 * link uses, so both sides derive one UVS layout. */
struct asahi_gs_shader_key {
   uint64_t flat_varyings;
   uint64_t linear_varyings;
   uint64_t link_uvs;
   /* When trim_rast_varyings is set, the copy program rasterizes only the
    * user varyings in rast_varyings: the ones the bound fragment shader
    * reads. Transform feedback is written by the copy program's own stores
    * and is not affected. */
   uint64_t rast_varyings;
   uint64_t trim_rast_varyings;
};
static_assert(sizeof(struct asahi_gs_shader_key) == 40, "no holes");

union asahi_shader_key {
   struct asahi_vs_shader_key vs;
   struct asahi_fs_shader_key fs;
   struct asahi_cs_shader_key cs;
   struct asahi_gs_shader_key gs;
};

enum agx_dirty {
   AGX_DIRTY_VERTEX = BITFIELD_BIT(0),
   AGX_DIRTY_VIEWPORT = BITFIELD_BIT(1),
   AGX_DIRTY_SCISSOR_ZBIAS = BITFIELD_BIT(2),
   AGX_DIRTY_ZS = BITFIELD_BIT(3),
   AGX_DIRTY_STENCIL_REF = BITFIELD_BIT(4),
   AGX_DIRTY_RS = BITFIELD_BIT(5),
   AGX_DIRTY_SPRITE_COORD_MODE = BITFIELD_BIT(6),
   AGX_DIRTY_PRIM = BITFIELD_BIT(7),

   /* Vertex/fragment pipelines, including uniforms and textures */
   AGX_DIRTY_VS = BITFIELD_BIT(8),
   AGX_DIRTY_FS = BITFIELD_BIT(9),

   /* Just the progs themselves */
   AGX_DIRTY_VS_PROG = BITFIELD_BIT(10),
   AGX_DIRTY_FS_PROG = BITFIELD_BIT(11),

   AGX_DIRTY_BLEND = BITFIELD_BIT(12),
   AGX_DIRTY_QUERY = BITFIELD_BIT(13),
   AGX_DIRTY_XFB = BITFIELD_BIT(14),
   AGX_DIRTY_SAMPLE_MASK = BITFIELD_BIT(15),
   AGX_DIRTY_BLEND_COLOR = BITFIELD_BIT(16),
   AGX_DIRTY_POLY_STIPPLE = BITFIELD_BIT(17),
};

/* Maximum number of in-progress + under-construction GPU batches.
 * Must be large enough for silly workloads that do things like
 * glGenerateMipmap on every frame, otherwise we end up losing performance.
 */
#define AGX_MAX_BATCHES (128)

static_assert(PIPE_TEX_FILTER_NEAREST < 2, "known order");
static_assert(PIPE_TEX_FILTER_LINEAR < 2, "known order");

enum asahi_blit_clamp {
   ASAHI_BLIT_CLAMP_NONE,
   ASAHI_BLIT_CLAMP_UINT_TO_SINT,
   ASAHI_BLIT_CLAMP_SINT_TO_UINT,

   /* keep last */
   ASAHI_BLIT_CLAMP_COUNT,
};

struct asahi_blit_key {
   enum pipe_format src_format, dst_format;
   bool array;
   bool aligned;
   bool pad[2];
};
static_assert(sizeof(struct asahi_blit_key) == 12, "packed");

DERIVE_HASH_TABLE(asahi_blit_key);

struct asahi_blitter {
   bool active;
   struct hash_table *blit_cs;

   /* [filter] */
   void *sampler[2];

   struct pipe_constant_buffer saved_cb;

   bool has_saved_image;
   struct pipe_image_view saved_image;

   unsigned saved_num_sampler_states;
   void *saved_sampler_states[PIPE_MAX_SAMPLERS];

   struct pipe_sampler_view *saved_sampler_view;

   void *saved_cs;
};

struct agx_oq_heap;

struct agx_context {
   struct pipe_context base;
   struct agx_compiled_shader *vs, *fs, *gs, *tcs;
   struct {
      struct agx_linked_shader *vs, *fs;

      /* G17P: the geometry copy program linked as the hardware vertex
       * shader of a geometry-assisted draw. NULL on other chips, which run
       * the copy program's main part directly. */
      struct agx_linked_shader *gs_copy;
   } linked;
   uint32_t dirty;

   /* ASAHI_MESA_DEBUG=perf, accumulated until PIPE_FLUSH_END_OF_FRAME. */
   struct {
      uint64_t frame;
      uint64_t draws;
      uint64_t vs_emits;
      uint64_t query_waits;
   } perf;

   /* Heap for dynamic memory allocation for geometry/tessellation shaders */
   struct pipe_resource *heap;

   /* Occlusion query heap */
   struct agx_oq_heap *oq;

   /* Acts as a context-level shader key */
   bool support_lod_bias;
   bool robust;

   /* Draws and dispatches dropped because a bound program had no encoding.
    * Exposed so an offline oracle can tell "the draw was skipped" apart from
    * "the draw silently rendered nothing".
    */
   unsigned skipped_no_program;

   /* Set of batches. When full, the LRU entry (the batch with the smallest
    * seqnum) is flushed to free a slot.
    */
   struct {
      uint64_t seqnum;
      struct agx_batch slots[AGX_MAX_BATCHES];

      /** Set of active batches for faster traversal */
      BITSET_DECLARE(active, AGX_MAX_BATCHES);

      /** Set of submitted batches for faster traversal */
      BITSET_DECLARE(submitted, AGX_MAX_BATCHES);

      /* Monotonic counter for each batch incremented when resetting a batch to
       * invalidate all associated queries. Compared to
       * agx_query::writer_generation.
       */
      uint64_t generation[AGX_MAX_BATCHES];
   } batches;

   /* Queue handle */
   uint32_t queue_id, virt_ring_idx;

   struct agx_batch *batch;
   struct agx_bo *timestamps;
   uint32_t timestamp_handle;

   struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
   uint32_t vb_mask;

   /* G17G: transform feedback on the CPU (agx_xfb_cpu.c). */
   struct draw_context *xfb_draw;
   struct vbuf_render *xfb_render;

   /* G17G renders one colour attachment at a time: while an application draw
    * is split, the bound framebuffer holds attachment g17g_mrt_rt alone at
    * slot 0, and g17g_mrt_pass counts the passes of the draw. */
   bool g17g_mrt_split;
   uint8_t g17g_mrt_rt, g17g_mrt_pass;

   /* G17G: an application draw runs once over every colour attachment,
    * attachment 0 in the tilebuffer and the others spilled
    * (agx_g17g_mrt_one_pass). */
   bool g17g_mrt_spill;

   /* Whether the bound framebuffer's spilled attachments can be described
    * (agx_g17g_mrt_spills_encodable), once it has been worked out. */
   bool g17g_mrt_spills_known, g17g_mrt_spills_ok;

   /* G17G renders one layer at a time (agx_g17g_bind_layer): while a draw
    * or clear of a layered framebuffer is split, the bound framebuffer holds
    * one layer's views. g17g_layered_fb is the application's framebuffer
    * while agx_swtnl.c sorts a CPU geometry stage's primitives by layer. */
   bool g17g_layer_split;
   uint16_t g17g_layer;
   const struct pipe_framebuffer_state *g17g_layered_fb;

   /* G17G: the stages before rasterization on the CPU (agx_swtnl.c). */
   struct draw_context *swtnl_draw;
   struct vbuf_render *swtnl_render;
   struct hash_table_u64 *swtnl_vs;
   bool swtnl_inner;

   unsigned patch_vertices;
   float default_outer_level[4];
   float default_inner_level[2];

   struct agx_stage stage[MESA_SHADER_STAGES];
   struct agx_vertex_elements *attributes;
   struct agx_rasterizer *rast;
   struct agx_zsa *zs;
   struct agx_blend *blend;
   struct pipe_blend_color blend_color;
   struct pipe_viewport_state viewport[AGX_MAX_VIEWPORTS];
   struct pipe_scissor_state scissor[AGX_MAX_VIEWPORTS];
   struct pipe_stencil_ref stencil_ref;
   struct agx_streamout streamout;
   uint16_t sample_mask;
   struct pipe_framebuffer_state framebuffer;

   uint32_t poly_stipple[32];

   struct pipe_query *cond_query;
   bool cond_cond;
   enum pipe_render_cond_flag cond_mode;

   struct agx_query *occlusion_query;
   struct agx_query *prims_generated[4];
   struct agx_query *tf_prims_generated[4];
   struct agx_query *tf_overflow[4];
   struct agx_query *tf_any_overflow;
   struct agx_query *pipeline_statistics[PIPE_STAT_QUERY_TS_INVOCATIONS];
   struct agx_query *time_elapsed;
   bool active_queries;
   bool active_draw_without_restart;

   struct util_debug_callback debug;
   bool is_noop;

   bool in_tess;

   /* M5: the part of a patch draw being drawn, when one render cannot hold
    * all of it (agx_draw_patches_split). Zero otherwise.
    */
   struct {
      unsigned first_patch;
      unsigned first_instance;
   } tess_split;

   struct blitter_context *blitter;
   struct asahi_blitter compute_blitter;

   /* Map of GEM handle to (batch index + 1) that (conservatively) writes that
    * BO, or 0 if no writer.
    */
   struct util_dynarray writer;

   /* Bound CL global buffers */
   struct util_dynarray global_buffers;

   struct hash_table *generic_meta;
   struct agx_bg_eot_cache bg_eot;

   bool any_faults;
   bool submit_quarantined;

   /* Work since the previous Gallium flush, including hazard-driven submits. */
   struct {
      uint64_t batches, mixed_batches, submits, failures;
   } submit_perf;
   struct pipe_device_reset_callback reset;

   uint32_t syncobj;
   uint32_t dummy_syncobj;
   int in_sync_fd;
   uint32_t in_sync_obj;
   uint64_t flush_last_seqid;
   uint64_t flush_my_seqid;
   uint64_t flush_other_seqid;

   struct agx_scratch scratch_vs;
   struct agx_scratch scratch_fs;
   struct agx_scratch scratch_cs;
};

static inline unsigned
agx_batch_idx(struct agx_batch *batch)
{
   return batch - batch->ctx->batches.slots;
}

static void
agx_writer_add(struct agx_context *ctx, uint8_t batch_index, unsigned handle)
{
   assert(batch_index < AGX_MAX_BATCHES && "invariant");
   static_assert(AGX_MAX_BATCHES < 0xFF, "no overflow on addition");

   /* If we need to grow, double the capacity so insertion is amortized O(1). */
   if (unlikely(handle >= ctx->writer.size)) {
      unsigned new_size =
         MAX2(ctx->writer.capacity * 2, util_next_power_of_two(handle + 1));
      unsigned grow = new_size - ctx->writer.size;

      memset(util_dynarray_grow(&ctx->writer, uint8_t, grow), 0,
             grow * sizeof(uint8_t));
   }

   /* There is now room */
   uint8_t *value = util_dynarray_element(&ctx->writer, uint8_t, handle);
   assert((*value) == 0 && "there should be no existing writer");
   *value = batch_index + 1;
}

static struct agx_batch *
agx_writer_get(struct agx_context *ctx, unsigned handle)
{
   if (handle >= ctx->writer.size)
      return NULL;

   uint8_t value = *util_dynarray_element(&ctx->writer, uint8_t, handle);

   if (value > 0)
      return &ctx->batches.slots[value - 1];
   else
      return NULL;
}

static void
agx_writer_remove(struct agx_context *ctx, unsigned handle)
{
   if (handle >= ctx->writer.size)
      return;

   uint8_t *value = util_dynarray_element(&ctx->writer, uint8_t, handle);
   *value = 0;
}

static inline struct agx_context *
agx_context(struct pipe_context *pctx)
{
   return (struct agx_context *)pctx;
}

struct agx_linked_shader;

typedef void (*meta_shader_builder_t)(struct nir_builder *b, const void *key);

void agx_init_meta_shaders(struct agx_context *ctx);

void agx_destroy_meta_shaders(struct agx_context *ctx);

struct agx_compiled_shader *agx_build_meta_shader(struct agx_context *ctx,
                                                  meta_shader_builder_t builder,
                                                  void *data, size_t data_size);

void agx_launch(struct agx_batch *batch, struct agx_grid grid,
                struct agx_workgroup wg, struct agx_compiled_shader *cs,
                struct agx_linked_shader *linked, mesa_shader_stage stage,
                unsigned variable_shared_mem, uint64_t indirect_num_workgroups);

void agx_launch_precomp(struct agx_batch *batch, struct agx_grid grid,
                        enum agx_barrier barrier, enum libagx_program program,
                        void *args, size_t arg_size);

#define MESA_DISPATCH_PRECOMP agx_launch_precomp

void agx_init_query_functions(struct pipe_context *ctx);

void
agx_primitives_update_direct(struct agx_context *ctx,
                             const struct pipe_draw_info *info,
                             const struct pipe_draw_start_count_bias *draw);

void agx_draw_vbo_from_xfb(struct pipe_context *pctx,
                           const struct pipe_draw_info *info,
                           unsigned drawid_offset,
                           const struct pipe_draw_indirect_info *indirect);

uint64_t agx_batch_get_so_address(struct agx_batch *batch, unsigned buffer,
                                  uint32_t *size);

void agx_init_streamout_functions(struct pipe_context *ctx);

void agx_xfb_cpu_create_vs(struct agx_context *ctx,
                           struct agx_uncompiled_shader *so,
                           const struct pipe_shader_state *cso,
                           const nir_shader *nir);
void agx_xfb_cpu_delete_vs(struct agx_context *ctx,
                           struct agx_uncompiled_shader *so);
bool agx_xfb_cpu_applies(struct agx_context *ctx,
                         const struct pipe_draw_indirect_info *indirect);
void agx_xfb_cpu_draw(struct agx_context *ctx,
                      const struct pipe_draw_info *info,
                      const struct pipe_draw_start_count_bias *draw);
void agx_xfb_cpu_destroy(struct agx_context *ctx);

bool agx_swtnl_lower_for_draw(nir_shader *nir);
bool agx_m5_gpu_tess(struct agx_context *ctx);
bool agx_swtnl_applies(struct agx_context *ctx);
void agx_swtnl_draw(struct agx_context *ctx, const struct pipe_draw_info *info,
                    const struct pipe_draw_start_count_bias *draw);
void agx_swtnl_delete_shader(struct agx_context *ctx,
                             struct agx_uncompiled_shader *so);
void agx_swtnl_destroy(struct agx_context *ctx);

void agx_g17g_bind_layer(struct agx_context *ctx,
                         const struct pipe_framebuffer_state *layered,
                         unsigned layer);
void agx_g17g_layered_clear(struct agx_context *ctx, unsigned buffers,
                            uint32_t color_clear_mask,
                            uint8_t stencil_clear_mask,
                            const union pipe_color_union *color,
                            double depth, unsigned stencil);
bool agx_g17g_level_renderable(struct agx_context *ctx,
                               struct pipe_resource *prsc,
                               enum pipe_format format, unsigned level,
                               unsigned layer);
void agx_g17g_split_mrt_clear(struct agx_context *ctx, unsigned buffers,
                              uint32_t color_clear_mask,
                              uint8_t stencil_clear_mask,
                              const union pipe_color_union *color,
                              double depth, unsigned stencil);

static inline void
agx_dirty_all(struct agx_context *ctx)
{
   ctx->dirty = ~0;

   for (unsigned i = 0; i < ARRAY_SIZE(ctx->stage); ++i)
      ctx->stage[i].dirty = ~0;
}

static inline void
agx_dirty_reset_graphics(struct agx_context *ctx)
{
   ctx->dirty = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(ctx->stage); ++i) {
      if (i != MESA_SHADER_COMPUTE)
         ctx->stage[i].dirty = 0;
   }
}

struct agx_rasterizer {
   struct pipe_rasterizer_state base;
   uint8_t cull[AGX_CULL_LENGTH];
   uint8_t line_width;
   uint8_t polygon_mode;
   bool depth_bias;
};

struct agx_query {
   unsigned type;
   unsigned index;

   uint64_t writer_generation[AGX_MAX_BATCHES];
   struct agx_bo *bo;
   struct agx_ptr ptr;
};

struct agx_sampler_state {
   struct pipe_sampler_state base;

   /* Prepared descriptor */
   struct agx_sampler_packed desc, desc_without_custom_border;

   /* Whether a custom border colour is required */
   bool uses_custom_border;

   /* Native compact border selector. CUSTOM is used only with the packed
    * custom-border payload below. */
   enum agx_border_colour border_colour;

   /* Packed custom border colour, or zero if none is required */
   struct agx_border_packed border;

   /* Logical colour consumed by G17P's shader-backed border emulation. */
   uint32_t custom_border[4];

   /* LOD bias packed as fp16, the form we'll pass to the shader */
   uint16_t lod_bias_as_fp16;
};

struct agx_sampler_view {
   struct pipe_sampler_view base;

   /* Resource/format, may differ from base in case of separate stencil */
   struct agx_resource *rsrc;
   enum pipe_format format;

   /* Prepared descriptor */
   struct agx_gallium_texture_descriptor desc;
};

struct agx_screen {
   struct pipe_screen pscreen;
   struct agx_device dev;
   struct disk_cache *disk_cache;

   /* Atomic: a context with unknown completion keeps the device alive. */
   bool submit_quarantined;
   struct agx_bo *rodata;

   /* Shared timeline syncobj and value to serialize flushes across contexts */
   uint32_t flush_syncobj;
   uint64_t flush_cur_seqid;
   uint64_t flush_wait_seqid;
   /* Lock to protect flush_wait_seqid updates (reads are just atomic) */
   simple_mtx_t flush_seqid_lock;

   /* Lock to protect syncobj usage vs. destruction in context destroy */
   struct u_rwlock destroy_lock;

   float heap_memory_percent;
};

static inline struct agx_screen *
agx_screen(struct pipe_screen *p)
{
   return (struct agx_screen *)p;
}

static inline struct agx_device *
agx_device(struct pipe_screen *p)
{
   return &(agx_screen(p)->dev);
}

#define perf_debug(dev, ...)                                                   \
   do {                                                                        \
      if (unlikely((dev)->debug & AGX_DBG_PERF))                               \
         mesa_logw(__VA_ARGS__);                                               \
   } while (0)

#define perf_debug_ctx(ctx, ...)                                               \
   perf_debug(agx_device((ctx)->base.screen), __VA_ARGS__)

struct agx_resource {
   struct pipe_resource base;
   uint64_t modifier;

   /* Should probably be part of the modifier. Affects the tiling algorithm, or
    * something like that.
    */
   bool mipmapped;

   /* Hardware backing */
   struct agx_bo *bo;

   struct renderonly_scanout *scanout;

   BITSET_DECLARE(data_valid, PIPE_MAX_TEXTURE_LEVELS);

   struct ail_layout layout;

   /* Metal does not support packed depth/stencil formats; presumably AGX does
    * not either. Instead, we create separate depth and stencil resources,
    * managed by u_transfer_helper.  We provide the illusion of packed
    * resources.
    */
   struct agx_resource *separate_stencil;

   /* Valid buffer range tracking, to optimize buffer appends */
   struct util_range valid_buffer_range;

   /* Cumulative shadowed byte count for this resource, that is, the number of
    * times multiplied by the resource size.
    */
   size_t shadowed_bytes;
};

static inline struct agx_resource *
agx_resource(struct pipe_resource *pctx)
{
   return (struct agx_resource *)pctx;
}

static inline bool
agx_resource_valid(struct agx_resource *rsrc, int level)
{
   /* Shared BOs can always be potentially valid */
   if (rsrc->bo && rsrc->bo->flags & AGX_BO_SHARED) {
      assert(level == 0);
      return true;
   }

   return BITSET_TEST(rsrc->data_valid, level);
}

static inline void *
agx_map_texture_cpu(struct agx_resource *rsrc, unsigned level, unsigned z)
{
   return ((uint8_t *)agx_bo_map(rsrc->bo)) +
          ail_get_layer_level_B(&rsrc->layout, z, level);
}

static inline uint64_t
agx_map_gpu(struct agx_resource *rsrc)
{
   return rsrc->bo->va->addr + rsrc->layout.level_offsets_B[0];
}

static inline uint64_t
agx_map_texture_gpu(struct agx_resource *rsrc, unsigned z)
{
   return agx_map_gpu(rsrc) +
          (uint64_t)ail_get_layer_offset_B(&rsrc->layout, z);
}

void agx_decompress(struct agx_context *ctx, struct agx_resource *rsrc,
                    const char *reason);

void agx_legalize_compression(struct agx_context *ctx,
                              struct agx_resource *rsrc,
                              enum pipe_format format);

struct agx_transfer {
   struct pipe_transfer base;
   void *map;
   struct {
      struct pipe_resource *rsrc;
      struct pipe_box box;
   } staging;
};

static inline struct agx_transfer *
agx_transfer(struct pipe_transfer *p)
{
   return (struct agx_transfer *)p;
}

void agx_upload_vbos(struct agx_batch *batch);
void agx_upload_uniforms(struct agx_batch *batch,
                         struct agx_compiled_shader *hw_vs);

void agx_set_sampler_uniforms(struct agx_batch *batch, mesa_shader_stage stage);

void agx_set_cbuf_uniforms(struct agx_batch *batch, mesa_shader_stage stage);

void agx_set_ssbo_uniforms(struct agx_batch *batch, mesa_shader_stage stage);

bool agx_nir_lower_point_size(nir_shader *nir, bool insert_write);

bool agx_nir_lower_sysvals(nir_shader *shader, mesa_shader_stage desc_stage,
                           bool lower_draw_params,
                           unsigned texture_descriptor_size);

bool agx_nir_layout_uniforms(nir_shader *shader,
                             struct agx_compiled_shader *compiled,
                             unsigned *push_size, unsigned uniform_base,
                             bool dword_ranges);

struct agx_lower_bindings_options {
   bool *uses_bindless_samplers;
   bool g17p_plain_txf_crawl;
   bool g17p_native_sampler_handles;
   bool g17p_shadow_gather;
   bool native_buffer_metadata;
   uint64_t *image_address_mask;
   uint64_t *image_pbe_mask;
   /* G18G: no G17P query-LOD bank; LOD queries keep the ordinary handle. */
   bool g18;
   bool g17p_bindless_storage;
   unsigned image_records;
   /* Keep a crawled texture bindful when its index fits the state table:
    * the handle serves the descriptor read, the operation names the entry. */
   bool table_index_crawls;
   unsigned bindful_sampler_count;
};

bool agx_nir_lower_bindings(nir_shader *shader,
                       struct agx_lower_bindings_options *options);
bool agx_nir_lower_custom_border_g17p(nir_shader *shader, bool shadow_gather);
/* As agx_nir_lower_custom_border_g17p; g18 selects G18G's shadow border
 * emulation and reports shadow operations it cannot emulate. */
bool agx_nir_lower_custom_border_native(nir_shader *shader, bool shadow_gather,
                                        bool g18, bool *fully_supported);
bool agx_nir_lower_custom_border_g17g(nir_shader *shader);
bool agx_nir_rebind_g17p_texture_crawls(nir_shader *shader);

/* Chips whose shadow gathers rebuild the API border depth comparison: the
 * native custom-border sampler returns one, so border taps would otherwise
 * compare against 1.0. This selects the probe sampler bank, the comparison
 * uniforms and the shadow-gather lowering together, and with them the
 * recovery of native sampler indices (handle i+1) in custom-border
 * lowering. */
static inline bool
agx_shadow_gather_custom_border(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P || chip == AGX_CHIP_G16G;
}

/* Bytes of struct agx_stage_uniforms uploaded as the stage sysval table. Other
 * chips stop before the G18G-only fields that follow
 * custom_border_compare_func; they never read them. */
static inline size_t
agx_stage_uniforms_size(enum agx_chip chip)
{
   if (chip == AGX_CHIP_G18G)
      return sizeof(struct agx_stage_uniforms);

   /* One word past lod_bias gives a lone lod_bias[31] the padding half that
    * Ioto's dword upload reads. */
   return agx_shadow_gather_custom_border(chip)
             ? offsetof(struct agx_stage_uniforms, custom_border_compare)
             : offsetof(struct agx_stage_uniforms, custom_border_compare_func) +
                  sizeof(uint32_t);
}

bool agx_nir_lower_g17p_storage_records(nir_shader *shader,
                                        unsigned record_size_B);
bool agx_nir_rebind_g17g_image_loads(nir_shader *shader);

bool agx_batch_is_active(struct agx_batch *batch);
bool agx_batch_is_submitted(struct agx_batch *batch);

/* BO pointers remain stable while the batch holds its reference. */
static bool
agx_batch_uses_bo(struct agx_batch *batch, struct agx_bo *bo)
{
   return _mesa_set_search(batch->bo_list.set, bo) != NULL;
}

static inline void
agx_batch_add_bo_internal(struct agx_batch *batch, struct agx_bo *bo)
{
   bool found;
   _mesa_set_search_or_add(batch->bo_list.set, bo, &found);
   if (found)
      return;

   /* One reference per distinct BO, released when the batch retires. */
   util_dynarray_append(&batch->bo_list.list, bo);
   agx_bo_reference(bo);
}

static inline void
agx_batch_add_bo(struct agx_batch *batch, struct agx_bo *bo)
{
   agx_batch_add_bo_internal(batch, bo);
   assert(agx_batch_uses_bo(batch, bo));
}

#define AGX_BATCH_FOREACH_BO(batch, entry) \
   util_dynarray_foreach(&(batch)->bo_list.list, struct agx_bo *, entry)

struct drm_asahi_cmd_compute;
struct drm_asahi_cmd_render;

void agx_batch_submit(struct agx_context *ctx, struct agx_batch *batch,
                      struct drm_asahi_cmd_compute *compute,
                      struct drm_asahi_cmd_render *render);

void agx_flush_batch(struct agx_context *ctx, struct agx_batch *batch);
void agx_flush_batch_for_reason(struct agx_context *ctx,
                                struct agx_batch *batch, const char *reason);
void agx_flush_all(struct agx_context *ctx, const char *reason);
void agx_flush_readers(struct agx_context *ctx, struct agx_resource *rsrc,
                       const char *reason);
void agx_flush_writer(struct agx_context *ctx, struct agx_resource *rsrc,
                      const char *reason);

void agx_sync_writer(struct agx_context *ctx, struct agx_resource *rsrc,
                     const char *reason);
void agx_sync_readers(struct agx_context *ctx, struct agx_resource *rsrc,
                      const char *reason);
void agx_sync_batch(struct agx_context *ctx, struct agx_batch *batch);
bool agx_poll_batch(struct agx_context *ctx, struct agx_batch *batch);
void agx_sync_all(struct agx_context *ctx, const char *reason);
void agx_sync_batch_for_reason(struct agx_context *ctx, struct agx_batch *batch,
                               const char *reason);
void agx_memory_barrier(struct pipe_context *pctx, unsigned flags);

/* Use these instead of batch_add_bo for proper resource tracking */
void agx_batch_reads(struct agx_batch *batch, struct agx_resource *rsrc);
void agx_batch_writes(struct agx_batch *batch, struct agx_resource *rsrc,
                      unsigned level);
void agx_batch_writes_range(struct agx_batch *batch, struct agx_resource *rsrc,
                            unsigned offset, unsigned size);
void agx_batch_track_image(struct agx_batch *batch,
                           struct pipe_image_view *image);

bool agx_any_batch_uses_resource(struct agx_context *ctx,
                                 struct agx_resource *rsrc);

/* 16384 is the maximum framebuffer dimension, so we use a larger width (the
 * maximum uint16_t) as a sentinel to identify the compute batch. This ensures
 * compute batches don't mix with graphics. This is a bit of a hack but it
 * works.
 */
#define AGX_COMPUTE_BATCH_WIDTH 0xFFFF

static inline bool
agx_batch_is_compute(struct agx_batch *batch)
{
   return batch->key.width == AGX_COMPUTE_BATCH_WIDTH;
}

struct agx_batch *agx_get_batch(struct agx_context *ctx);
struct agx_batch *agx_get_compute_batch(struct agx_context *ctx);
void agx_batch_reset(struct agx_context *ctx, struct agx_batch *batch);
void agx_batch_retire_faulted(struct agx_context *ctx,
                              struct agx_batch *batch);
int agx_cleanup_batches(struct agx_context *ctx);

void agx_batch_add_timestamp_query(struct agx_batch *batch,
                                   struct agx_query *q);
void agx_add_timestamp_end_query(struct agx_context *ctx, struct agx_query *q);

void agx_query_increment_cpu(struct agx_context *ctx, struct agx_query *query,
                             uint64_t increment);

enum asahi_blitter_op /* bitmask */
{
   ASAHI_SAVE_TEXTURES = 1,
   ASAHI_SAVE_FRAMEBUFFER = 2,
   ASAHI_SAVE_FRAGMENT_STATE = 4,
   ASAHI_SAVE_FRAGMENT_CONSTANT = 8,
   ASAHI_DISABLE_RENDER_COND = 16,
};

enum {
   ASAHI_CLEAR = ASAHI_SAVE_FRAGMENT_STATE | ASAHI_SAVE_FRAGMENT_CONSTANT,

   ASAHI_BLIT =
      ASAHI_SAVE_FRAMEBUFFER | ASAHI_SAVE_TEXTURES | ASAHI_SAVE_FRAGMENT_STATE,
};

void agx_blitter_save(struct agx_context *ctx, struct blitter_context *blitter,
                      enum asahi_blitter_op op);

void agx_blit(struct pipe_context *pipe, const struct pipe_blit_info *info);

bool agx_g18_msaa4_storage(const struct agx_device *dev,
                           const struct agx_resource *rsrc,
                           enum pipe_format format);

bool agx_g17p_direct_resolve_blit_shape(const struct pipe_blit_info *info,
                                        bool source_multisampled);

bool agx_g17p_staged_resolve_blit_shape(const struct pipe_blit_info *info);

bool agx_g17p_stencil_plane_blit_shape(const struct pipe_blit_info *info);

bool agx_generate_mipmap_g17g(struct pipe_context *pctx,
                              struct pipe_resource *prsc,
                              enum pipe_format format, unsigned base_level,
                              unsigned last_level, unsigned first_layer,
                              unsigned last_layer);

void agx_resource_copy_region(struct pipe_context *pctx,
                              struct pipe_resource *dst, unsigned dst_level,
                              unsigned dstx, unsigned dsty, unsigned dstz,
                              struct pipe_resource *src, unsigned src_level,
                              const struct pipe_box *src_box);

void agx_copy_buffer_range(struct pipe_context *pctx, struct pipe_resource *dst,
                           unsigned dst_offset, struct pipe_resource *src,
                           unsigned src_offset, unsigned size);

/* Batch logic */

struct agx_encoder agx_encoder_allocate(struct agx_batch *batch,
                                        struct agx_device *dev);

uint64_t agx_batch_get_g17p_vdm_alias(struct agx_batch *batch,
                                      struct agx_bo *bo, uint64_t address);

void agx_batch_init_state(struct agx_batch *batch);

struct asahi_bg_eot {
   uint64_t usc;
   uint64_t rsrc_spec;

   /* M5: the program's primary (thread-private) and staging storage, for
    * the render's tile record (agx_m5_render_resources). */
   uint32_t primary_storage_B;
   uint32_t staging_storage_B;
};

/* Per-field maxima. Page-count descriptor words are exact under this merge;
 * the per-warp KiB words are an upper bound when different programs supply
 * the register, frame and staging maxima (graphics-scratch-design-1077). */
static inline void
agx_g18_note_program(struct agx_batch *batch, bool fragment,
                     unsigned registers_16, unsigned scratch_bytes,
                     unsigned staging_16)
{
   struct drm_asahi_g18_render_resources *r = &batch->g18.resources;
   if (fragment) {
      r->fragment_registers_16 = MAX2(r->fragment_registers_16, registers_16);
      r->fragment_scratch_bytes = MAX2(r->fragment_scratch_bytes, scratch_bytes);
      r->fragment_staging_16 = MAX2(r->fragment_staging_16, staging_16);
   } else {
      r->vertex_registers_16 = MAX2(r->vertex_registers_16, registers_16);
      r->vertex_scratch_bytes = MAX2(r->vertex_scratch_bytes, scratch_bytes);
      r->vertex_staging_16 = MAX2(r->vertex_staging_16, staging_16);
   }
   batch->g18.scratch |= scratch_bytes != 0;
}

/* Tile programs: slot 0 background (including partial), slot 1 EOT. */
static inline void
agx_g18_note_tile(struct agx_batch *batch, bool store, unsigned registers_16,
                  unsigned staging_16)
{
   struct drm_asahi_g18_render_resources *r = &batch->g18.resources;
   const unsigned slot = store ? 1 : 0;
   r->tile_registers_16[slot] = MAX2(r->tile_registers_16[slot], registers_16);
   r->tile_staging_16[slot] = MAX2(r->tile_staging_16[slot], staging_16);
}

void agx_batch_note_m5_render_program(
   struct agx_batch *batch, const struct drm_asahi_render_program_v2 *program);
void agx_batch_note_m5_compute_program(
   struct agx_batch *batch, const struct drm_asahi_compute_program *program);
void agx_batch_set_m5_tile_program(struct agx_batch *batch,
                                   const struct asahi_bg_eot *const *programs,
                                   unsigned count);

struct asahi_bg_eot agx_build_bg_eot(struct agx_batch *batch, bool store,
                                     bool partial_render);

/* Query management */
uint16_t agx_get_oq_index(struct agx_batch *batch, struct agx_query *query);
uint64_t agx_get_query_address(struct agx_batch *batch,
                               struct agx_query *query);
uint64_t agx_get_occlusion_heap(struct agx_batch *batch);

void agx_finish_batch_queries(struct agx_batch *batch, uint64_t begin_ts,
                              uint64_t end_ts);

bool agx_render_condition_check_inner(struct agx_context *ctx);

static inline bool
agx_render_condition_check(struct agx_context *ctx)
{
   if (likely(!ctx->cond_query))
      return true;
   else
      return agx_render_condition_check_inner(ctx);
}

static inline uint32_t
agx_texture_buffer_size_el(enum pipe_format format, uint32_t size)
{
   unsigned blocksize = util_format_get_blocksize(format);

   return MIN2(AGX_TEXTURE_BUFFER_MAX_SIZE, size / blocksize);
}

void agx_decompress_inplace(struct agx_batch *batch, struct pipe_surface *surf,
                            const char *reason);
