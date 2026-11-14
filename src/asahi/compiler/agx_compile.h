/*
 * Copyright 2018-2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/nir/nir.h"
#include "asahi/lib/agx_chip.h"
#include "asahi/lib/agx_ioto_load_core.h"
#include "agx3_fltnorm_core.h"
#include "agx3_cprog_input.h"
#include "util/bitset.h"
#include "util/shader_stats.h"
#include "util/u_dynarray.h"
#include "util/u_tristate.h"
#include "shader_enums.h"

/* Cache-line align shader programs. This matches the prop compiler. */
#define AGX_CODE_ALIGN 0x80

/* G17P's register file: halves 0..191. */
#define AGX3B_NUM_REGS (192)

struct agx_cf_binding {
   /* Base coefficient register */
   uint8_t cf_base;

   /* Slot being bound */
   gl_varying_slot slot : 8;

   /* First component bound.
    *
    * Must be 2 (Z) or 3 (W) if slot == VARYING_SLOT_POS.
    */
   unsigned offset : 2;

   /* Number of components bound */
   unsigned count : 3;

   /* Is smooth shading enabled? If false, flat shading is used */
   bool smooth : 1;

   /* Perspective correct interpolation */
   bool perspective : 1;

   uint8_t pad;
};

/* Conservative bound, * 4 due to offsets (TODO: maybe worth eliminating
 * coefficient register aliasing?)
 */
#define AGX_MAX_CF_BINDINGS (VARYING_SLOT_MAX * 4)

struct agx_varyings_fs {
   /* Number of coefficient registers used */
   unsigned nr_cf;

   /* Number of coefficient register bindings */
   unsigned nr_bindings;

   /* Whether gl_FragCoord.z is read */
   bool reads_z;

   /* Coefficient register bindings */
   struct agx_cf_binding bindings[AGX_MAX_CF_BINDINGS];
};

union agx_varyings {
   struct agx_varyings_fs fs;
};

struct agx_interp_info {
   /* Bit masks indexed by I/O location of flat and linear varyings */
   uint64_t flat;
   uint64_t linear;
};
static_assert(sizeof(struct agx_interp_info) == 16, "packed");

struct agx_rodata {
   /* Offset in the binary */
   uint32_t offset;

   /* Base uniform to map constants */
   uint16_t base_uniform;

   /* Number of 16-bit constants to map contiguously there */
   uint16_t size_16;
};

#define AGX3_UNIFORM_DMA_MAX_SPANS 6

enum agx3_uniform_dma_source {
   /* Memory at the 64-bit pointer held in uniform halves [pointer,
    * pointer + 4): the root table or one descriptor set. */
   AGX3_UNIFORM_DMA_POINTER,

   /* The buffer whose descriptor -- a 64-bit address and a 32-bit size -- is
    * stored at byte descriptor_B through that pointer: a dynamic buffer in the
    * root table or a buffer in a descriptor set the driver knows when it
    * records the launch (agx_shader_key::g17p_dma_buffers_B and
    * g17p_dma_static_set_mask). The driver copies the dwords below the
    * buffer's size rounded up to 4 bytes from the buffer and every later dword
    * from the zero page, so the compiler drops the per-component bounds checks
    * of the loads the span replaces: they differ only within that rounding,
    * which the robust buffer access size alignments (at least 4) allow.
    */
   AGX3_UNIFORM_DMA_BUFFER,
};

struct agx3_uniform_dma_span {
   uint8_t source; /* enum agx3_uniform_dma_source */
   uint16_t pointer;
   uint16_t descriptor_B; /* AGX3_UNIFORM_DMA_BUFFER only */
   uint16_t dest;         /* first uniform half written */
   uint16_t halfs;        /* even */
   uint32_t offset_B;     /* dword-aligned, within the pointed-to memory */
};

static inline bool
agx3_uniform_dma_span_equal(const struct agx3_uniform_dma_span *a,
                            const struct agx3_uniform_dma_span *b)
{
   return a->source == b->source && a->pointer == b->pointer &&
          a->descriptor_B == b->descriptor_B && a->dest == b->dest &&
          a->halfs == b->halfs && a->offset_B == b->offset_B;
}

enum agx_g18_sample_mask_form {
   AGX_G18_SAMPLE_MASK_KILL = (1 << 0),
   AGX_G18_SAMPLE_MASK_FULL = (1 << 1),
   AGX_G18_SAMPLE_MASK_OTHER = (1 << 2),
   /* Native1121 psdiscard: per-sample kill and update words (run 1128). */
   AGX_G18_SAMPLE_MASK_SAMPLE = (1 << 3),
};

struct agx_shader_info {
   mesa_shader_stage stage;
   uint32_t binary_size;

   union agx_varyings varyings;

   /* Number of uniforms */
   uint16_t push_count;

   /* G17P load_preamble declarations after NIR optimization, in native
    * half-register units. This is a publication contract, not a CProg TPS
    * import layout. Consumers must also publish compiler rodata separately. */
   uint64_t agx3_uniform_halfs[AGX3_CPROG_INPUT_WORDS];
   bool agx3_uniform_halfs_valid;

   /* G17P uniform DMA: constant-index loads of data whose address the driver
    * knows when it records the launch were replaced by uniform reads. The
    * driver publishes each span as one Ioto uniform range, copied in by the
    * launch program as the loads would have read it. */
   uint8_t agx3_uniform_dma_count;
   struct agx3_uniform_dma_span agx3_uniform_dma[AGX3_UNIFORM_DMA_MAX_SPANS];

   /* Local memory allocation in bytes */
   uint16_t local_size;

   /* Local imageblock allocation in bytes per thread */
   uint16_t imageblock_stride;

   /* Scratch memory allocation in bytes for main/preamble respectively */
   unsigned scratch_size, preamble_scratch_size;

   /* G17P compiler footprint programmed in SPECTPR. This is separate from
    * explicit shader scratch and does not use the legacy helper program. */
   unsigned agx3_spectpr_scratch_size;

   /* Exact emitted code extents, excluding inter-function alignment and
    * rodata. A missing preamble has size zero. Native constant-program
    * launchers need its extent independently of the main function's offset. */
   unsigned main_size, preamble_size;

   /* Does the shader have a preamble? If so, it is preamble_size bytes long
    * at offset preamble_offset.
    * The main shader is at offset main_offset. The preamble is executed first.
    */
   bool has_preamble;
   /* Native constant-program live-ins. External G17P launch needs the
    * uniform-input prelude when a uniform half is read before definition. */
   bool agx3_preamble_uniform_inputs;
   /* The machine program consumes the frozen layout through native TPS RLDs.
    * Older uniform-read programs must not silently select this loader ABI. */
   bool agx3_preamble_tps_inputs;
   /* Versioned compiler proof: imported inputs followed only by RA-created
    * static spill slots. Set only after final IR validation and successful pack.
    * This occupies existing alignment padding, but all ABI layouts are checked. */
   uint8_t agx3_preamble_static_spill_version;
   struct agx3_cprog_inputs agx3_preamble_inputs;
   unsigned preamble_offset, main_offset;

   /* Does the shader read the tilebuffer? */
   bool reads_tib;

   /* Does the shader require early fragment tests? */
   bool early_fragment_tests;

   /* Does the shader potentially draw to a nonzero viewport? */
   bool nonzero_viewport;

   /* Does the shader write layer and/or viewport index? Written together */
   bool writes_layer_viewport;

   /* Does the shader control the sample mask? */
   bool writes_sample_mask;

   /* Union of emitted G18 2x/4x sample-mask forms. A full late transaction
    * does not establish proof for an unrelated partial or early update. */
   uint8_t g18_sample_mask_forms;

   /* Depth layout, never equal to NONE */
   enum gl_frag_depth_layout depth_layout;

   /* Based only the compiled shader, should tag writes be disabled? This is set
    * based on what is outputted. Note if rasterizer discard is used, that needs
    * to disable tag writes regardless of this flag.
    */
   bool tag_write_disable;

   /* Shader is incompatible with triangle merging */
   bool disable_tri_merging;

   /* Fragment shader takes quad derivatives: an implicit-LOD texture op or
    * ddx/ddy, or it declares QuadDerivativesKHR. Triangle merging
    * packs fragments of two primitives that share an edge into a single 2x2
    * quad, so the lanes of a quad straddling that edge interpolate from two
    * different plane equations and any derivative taken across it is
    * meaningless. A quad intrinsic alone does not count: it reads another
    * lane's own value, and merging is what keeps a clipped primitive's pieces
    * in whole quads (Metal leaves it on for quad_broadcast on the M4 Pro).
    *
    * This is deliberately narrower than disable_tri_merging, which also covers
    * memory writes and wide subgroup intrinsics. Apple's own G17P shader
    * metadata leaves the native merge-disable bit clear for those, and neither
    * needs quad coherence, so G17P consumes only this term.
    */
   bool needs_quad_derivatives;

   /* Fragment shader needs whole 2x2 quads: a quad operation
    * (require_full_quads) or QuadDerivativesKHR. */
   bool needs_full_quads;

   /* Reads draw ID system value */
   bool uses_draw_id;

   /* Reads base vertex/instance */
   bool uses_base_param;

   /* Uses txf and hence needs a txf sampler mapped */
   bool uses_txf;

   /* G17P fragment programs containing SMP use Apple's sampled-fragment
    * constant-program protocol.  This is independent of descriptor source,
    * LOD mode, and sampler-heap use. */
   bool g17p_sampled_fragment;

   /* Potentially uses the sampler heap (conservative) */
   bool uses_sampler_heap;

   /* Number of texture/sampler state registers pushed by the preamble. */
   uint8_t texture_state_count, sampler_state_count;

   /* Number of 16-bit registers used by the main shader and preamble
    * respectively.
    */
   uint16_t nr_gprs, nr_preamble_gprs;

   /* AGX3 IPR footprints in halfwords. The launch path doubles these values
    * to obtain bytes: ShaderInfo fields 38/42 for main, and field 37
    * (thread_invariant_largest_ipr_bytes) for the preamble. These are not
    * temporary GPR counts. */
   uint16_t agx3_staging_halfs;
   bool agx3_staging_known;
   /* Separate thread-invariant IPR footprint for native CProg. It must not
    * inherit the main program's texture/staging domain or its GPR count. */
   uint16_t agx3_preamble_staging_halfs;
   bool agx3_preamble_staging_known;

   /* Output mask set during driver lowering */
   uint64_t outputs;

   /* Workgroup size */
   uint16_t workgroup_size[3];

   /* Exact AGX3 direct-launch ShaderInfo producers. Resource-info queries
    * come from NIR. Every backend lowering which emits a ray traversal/query
    * opcode must set uses_ray_traversal as well; ray_queries alone is not a
    * future-proof proxy for full RT pipelines. ShaderInfo ID32 is independent
    * from traversal use and must have its own committed-path producer. */
   bool uses_matrix16;
   bool uses_resource_info_query;
   bool uses_ray_traversal;
   bool ray_traversal_depth_nonzero;

   /* There may be constants in the binary. The driver must map these to uniform
    * registers as specified hre.
    */
   struct agx_rodata rodata;

   struct agx2_stats stats;

   /* Main of this part begins with G17P FLTNORM carrying f68. The linker
    * clears it if another part precedes this one in the executable. */
   bool agx3_fltnorm_start;
   struct agx3_colour_link_info agx3_colour_link;

   /* Pixel classes this fragment part waited for outside control flow, so
    * holds on every path into the next part. A G17G epilog must not wait for
    * them again (framebuffer fetch: the main waits for class 8). */
   uint8_t agx3_pixel_held;
};

struct agx_precompiled_kernel_info {
   uint32_t preamble_offset, main_offset;
   uint32_t main_size, preamble_size, binary_size;
   struct agx_rodata rodata;
   uint16_t nr_gprs, nr_preamble_gprs;
   uint16_t push_count;
   uint16_t workgroup_size[3];
   uint16_t local_size;
   uint16_t imageblock_stride;
   uint16_t scratch_size;
   uint16_t agx3_spectpr_scratch_size;
   bool uses_txf;
   bool agx3_staging_known;
   uint16_t agx3_staging_halfs;
   bool agx3_preamble_staging_known;
   uint16_t agx3_preamble_staging_halfs;

   /* AGX3 kernels read the printf buffer address from a 64-bit uniform at this
    * half, the first multiple of 4 after their arguments, or 0 if they do not
    * use it. AGX2 kernels use the fixed LIBAGX_PRINTF_BUFFER_ADDRESS.
    */
   uint16_t printf_uniform;

   /* AGX3 thread-private storage frame in bytes: spills and returned atomic
    * words. AGX2 kernels do not spill.
    */
   uint32_t agx3_tps_size;
};

static inline struct agx_precompiled_kernel_info
agx_compact_kernel_info(struct agx_shader_info *info)
{
   assert(info->has_preamble == (info->nr_preamble_gprs > 0));
   assert(info->texture_state_count <= 8 && "static maximum, no need to plumb");

   return (struct agx_precompiled_kernel_info){
      .preamble_offset = info->preamble_offset,
      .main_offset = info->main_offset,
      .main_size = info->main_size,
      .preamble_size = info->preamble_size,
      .binary_size = info->binary_size,
      .rodata = info->rodata,
      .nr_gprs = info->nr_gprs,
      .nr_preamble_gprs = info->nr_preamble_gprs,
      .push_count = info->push_count,
      .workgroup_size = {info->workgroup_size[0], info->workgroup_size[1],
                         info->workgroup_size[2]},
      .local_size = info->local_size,
      .imageblock_stride = info->imageblock_stride,
      .scratch_size = (uint16_t)info->scratch_size,
      .agx3_spectpr_scratch_size = (uint16_t)info->agx3_spectpr_scratch_size,
      .uses_txf = info->uses_txf,
      .agx3_staging_known = info->agx3_staging_known,
      .agx3_staging_halfs = info->agx3_staging_halfs,
      .agx3_preamble_staging_known = info->agx3_preamble_staging_known,
      .agx3_preamble_staging_halfs = info->agx3_preamble_staging_halfs,
      .agx3_tps_size =
         MAX2(info->scratch_size, info->agx3_spectpr_scratch_size),
   };
}

struct agx_shader_part {
   struct agx_shader_info info;
   void *binary;
};

static inline bool
agx_is_shader_empty(struct agx_shader_part *s)
{
   /* Last instruction is a stop, so if there's one instruction, there is
    * nothing but a stop. The shader is thus empty.
    */
   return (s->info.stats.instrs == 1);
}

#define AGX_MAX_RTS (8)

enum agx_format {
   AGX_FORMAT_I8 = 0,
   AGX_FORMAT_I16 = 1,
   AGX_FORMAT_I32 = 2,
   AGX_FORMAT_F16 = 3,
   AGX_FORMAT_U8NORM = 4,
   AGX_FORMAT_S8NORM = 5,
   AGX_FORMAT_U16NORM = 6,
   AGX_FORMAT_S16NORM = 7,
   AGX_FORMAT_RGB10A2 = 8,
   AGX_FORMAT_SRGBA8 = 10,
   AGX_FORMAT_RG11B10F = 12,
   AGX_FORMAT_RGB9E5 = 13,

   /* Keep last */
   AGX_NUM_FORMATS,
};

struct agx_fs_shader_key {
   /* Normally, access to the tilebuffer must be guarded by appropriate fencing
    * instructions to ensure correct results in the presence of out-of-order
    * hardware optimizations. However, specially dispatched clear shaders are
    * not subject to these conditions and can omit the wait instructions.
    *
    * Must (only) be set for special clear shaders.
    *
    * Must not be used with sample mask writes (including discards) or
    * tilebuffer loads (including blending).
    */
   bool ignore_tib_dependencies;

   /* Driver background programs use the tile layer as their layer input. */
   bool background;

   /* When dynamic sample shading is used, the fragment shader is wrapped in a
    * loop external to the API shader. This bit indicates that we are compiling
    * inside the sample loop, meaning the execution nesting counter is already
    * zero and must be preserved.
    */
   bool inside_sample_loop;

   /* Base coefficient register. 0 for API shaders but nonzero for FS prolog */
   uint8_t cf_base;

   /* G17P rasterization sample count that coverage updates (sample mask
    * writes and discards) are compiled for. Apple's fragments build the
    * coverage word from different system values, and with a different
    * all-samples constant, at one and at several samples. 0 is one sample.
    */
   uint8_t nr_samples;
};

struct agx_device_key {
   /* Some native AGX3 contracts differ between G16G and G17P despite sharing
    * the instruction encoding, notably formatted PBE store inputs. */
   enum agx_chip chip;

   /*
    * Binary encoding of the instruction set to emit.
    *
    * Every part this compiler supports is AGX2, so the packer accepts only
    * that and asserts on anything else. Carrying it explicitly is what makes
    * a future AGX3 table a dispatch rather than a rewrite, and it means a
    * caller that reaches the compiler with an unsupported part is caught in
    * the compiler instead of producing an instruction stream for the wrong
    * hardware. See agx_chip.h for why the encoding, rather than the chip, is
    * the thing that decides.
    */
   enum agx_isa_encoding encoding;

   /* Does the target GPU need explicit cluster coherency for atomics?
    * Only used on G13X.
    */
   enum u_tristate needs_g13x_coherency;

   /* Is soft fault enabled? This is technically system-wide policy set by the
    * kernel, but that's functionally a hardware feature.
    */
   bool soft_fault;

   /* Physical A18 Pro capability, not the G17P profile shared with T6040.
    * Offline/shared library keys leave this disabled. */
   bool shader_quad_control;

   /*
    * Re-encode the finished code as agx3d (G18G). The rest of the key then
    * describes G17G: G18G executes the G17G instruction set, so the compiler
    * runs exactly as for G17G and agx3_transcode_agx3d() re-encodes each
    * function's code after packing. See agx3_transcode.c for the evidence.
    */
   bool emit_agx3d;
};

/*
 * Is this a compile for the M6 GPU (G18G)? G18G compiles with the G17G key
 * (AGX_ISA_ENCODING_AGX3C, the G17G-family encoding shared by G17G, G17S and
 * G18P) and re-encodes as agx3d, so a gate on the encoding alone also fires
 * on every G17G-family GPU. Gate M6-only workarounds
 * and facts grounded only by G18G witnesses on this; keep the encoding gate
 * for G17G instruction-set behaviour that G17G and G18G share.
 */
static inline bool
agx_is_g18g(const struct agx_device_key *dev)
{
   return dev->encoding == AGX_ISA_ENCODING_AGX3C && dev->emit_agx3d;
}

struct agx_shader_key {
   /* Device info */
   struct agx_device_key dev;

   /* Number of reserved preamble slots at the start */
   unsigned reserved_preamble;

   /* Uniform half containing the indexed texture heap; sampler follows at+4. */
   uint16_t g17p_texture_uniform_base;

   /* G17P state-0 storage image read and written by the shader: the state-0
    * pointer names its texture/PBE descriptor pair, so the store's PBE is
    * record 1 of it (a write-only image's pointer names the PBE itself). */
   bool g17p_storage_image_pair;

   /* Whether scratch memory is available in the given shader stage */
   bool has_scratch;

   /* Whether we're compiling the helper program used for scratch allocation.
    * This has special register allocation requirements.
    */
   bool is_helper;

   /* Whether the driver supports uploading constants for this shader. If
    * false, constants will not be promoted to uniforms.
    */
   bool promote_constants;

   /* Similarly whether the driver supports promoting bindless
    * textures/samplers.  Currently this works only if non-bindless
    * textures/samplers are not used, but none of our drivers mix bindless /
    * non-bindless usage.
    */
   bool promote_textures;

   /* Keep the main shader monolithic even when profitable instructions could
    * otherwise be moved to a generated preamble.
    */
   bool no_preamble;

   /* A complete, single-sample M5 uniform-colour blend variant. The packer
    * still proves its one read/write pixel transaction before admitting it. */
   bool m5_uniform_blend;

   /* AGX3B: build the preamble as a native constant program (CProg inputs
    * imported through its TPS frame, arithmetic results selected straight
    * into uniform destinations, half exports legalized) for this shader.
    * G17P vertex programs always use it.
    */
   bool agx3_native_cprog;
   /* The native constant-program lowering for THIS compile. Decided per
    * compile by the driver (G17P fragment and vertex constant programs) and
    * carried in the key, never in process-global state that concurrently
    * compiling threads read. */
   bool g17p_native_uniforms;

   /* G17P uniform DMA: copy constant-index loads into driver-published
    * uniform ranges (agx_shader_info::agx3_uniform_dma) when they read
    * through the 64-bit pointer uniform at g17p_dma_root (the root table) or
    * at g17p_dma_root + 4 * (1 + s) for each set s in g17p_dma_set_mask, or
    * through a buffer descriptor the driver can read when it records the
    * launch: one of the g17p_dma_buffer_count 16-byte descriptors stored in
    * the root table from byte g17p_dma_buffers_B on, or any in a set in
    * g17p_dma_static_set_mask. !g17p_uniform_dma leaves code unchanged. */
   bool g17p_uniform_dma;
   uint16_t g17p_dma_root;
   uint8_t g17p_dma_set_mask;
   uint8_t g17p_dma_static_set_mask;
   uint8_t g17p_dma_buffer_count;
   uint16_t g17p_dma_buffers_B;

   /* Set if this is a non-monolithic shader that must be linked with additional
    * shader parts before the program can be used. This suppresses omission of
    * `stop` instructions, which the linker must insert instead.
    */
   bool no_stop;

   /* Exact number of 32-bit imageblock words in the G17P RT0 EOT store.
    * U16NORM is shared by RG16 and RGBA16, while I32 is shared by scalar R32
    * and R32G32 after tilebuffer lowering, so the backend cannot recover the
    * logical layout from the final AGX IR alone. */
   uint8_t g17p_rt0_eot_words;

   /* Select the M5 tile-store ABI for a compiler-owned end-of-tile program. */
   bool m5_rt0_eot;
   bool m5_rt0_eot_store_and_resolve;

   /* Store the tile without resolving it. Apple's own builtin end-of-tile
    * factory emits a second program per selector whose only differences are
    * the PACK_IMGWIR wire width and the IMGWBLK dimension class (2D -> 2D
    * multisample). The two forms are byte-distinct, so the backend must be
    * told which one the caller means rather than inferring it from the
    * tilebuffer sample count: a multisample tile is stored resolved through a
    * single-sample target on every path except VK_ATTACHMENT_STORE_OP_STORE. */
   bool g17p_rt0_eot_multisample;

   /* Authorize the exact I32 G17P EOT builtin. Set only for logical formats
    * whose I32 store shape and PBE descriptor are both grounded. */
   bool g17p_rt0_eot_i32;

   /* Authorize the exact four-word I32 builtin. The native RGBA32 UINT and
    * SFLOAT keys use the scalar 0x35 selector, unlike the two-word unsigned
    * pair. */
   bool g17p_rt0_eot_i32_four_words;

   /* Authorize the signed four-word I32 builtin, which stays single-sample. */
   bool g17p_rt0_eot_i32_signed_four_words;

   /* Authorize the G17G native single-sample, single-layer RT0 builtin. The
    * program itself is picked by the store-shape fields above, which describe
    * Mesa's lowered imageblock store rather than a GPU generation: Apple's
    * builtin factory maps an attachment key to the same format class on both
    * G17P and G17G. Its PBE table and tile byte offset occupy uniform
    * halves 0..5. */
   bool g17g_rt0_eot;

   /* Authorize the exact two-partition packed R11G11B10 EOT. Its post-RA
    * graph is otherwise indistinguishable from unsupported dual-I32 layouts. */
   bool g17p_r11g11b10_mrt_eot;

   /* Background program for a G17P AGX_G17P_MRT_MIXED layout. Apple's
    * builtin for such layouts stores each target without pixel transactions,
    * at f13=0 even from high registers, and in no required partition order. */
   bool g17p_mrt_bg;
   /* Generated native eMRT epilog. Memory writes may split its sample groups
    * into blocks, but tile operations retain the per-sample partition order. */
   bool g17p_emrt;

   /* A fragment epilog, background or end-of-tile program of a G18
    * single-sample mixed-format tile (agx_g18g_native_mixed_mask, run 1139).
    * Selects the native1130 tile forms: partial masks of every witnessed
    * format, blend reads, lstcol's lifetime bit by last use, the long lldcol
    * form, and the 8-bit-integer and R11G11B10 end-of-tile classes. */
   bool g18_native_mixed;

   /* A Honeykrisp hardware vertex binary adjacent to software geometry or
    * transform feedback. The API VS hardware variant may expose only values
    * consumed by GS/XFB, while the generated post-GS raster VS may expose only
    * values GS produced. Both may therefore have a sparse terminal UVS
    * suffix. */
   bool g17p_sparse_vertex_outputs;

   /* G17P multi-target epilog: bit n is set when a colour target of the
    * framebuffer starts at tile byte n. Apple's fragments select a written
    * target's pixel classes from its position among all targets, so an epilog
    * writing only some targets needs the others' offsets too. An empty set
    * derives the partitions from the tile operations. */
   BITSET_DECLARE(g17p_tile_partitions, AGX_G17P_MAX_BYTES_PER_SAMPLE);

   /* Number of targets, two to eight, stored by a G17P multi-target EOT for
    * arbitrary target formats, and the IMGWBLK selector of each target in
    * store order. Zero selects the format-specific EOT recognizers. */
   uint8_t g17p_mrt_eot_count;
   /* Override selected multisample stores with single-sample resolves.
    * A source-preserving resolve binds separate MS and SS PBE outputs. */
   uint8_t g17p_mrt_eot_resolve_mask;
   uint8_t g17p_mrt_eot_selectors[AGX_MAX_RTS];

   /* Set if this is a secondary shader part (prolog or epilog). This prevents
    * the compiler from allocating uniform registers. For example, this turns
    * off preambles.
    */
   bool secondary;

   /* G17P inline blending compiles a fragment program again with its colour
    * epilog inlined. The draw keeps loading the split main program's uniform
    * window, so promoted constants must sit where that program's rodata has
    * them. When ordinary promotion does not produce a prefix of this table,
    * place each constant at its slot here (or leave it unpromoted). NULL
    * outside that recompile. */
   const uint16_t *g17p_rodata_hint;
   uint16_t g17p_rodata_hint_base;
   uint16_t g17p_rodata_hint_size_16;

   union {
      struct agx_fs_shader_key fs;
   };
};

#ifdef __cplusplus
extern "C" {
#endif

uint64_t agx_gather_texcoords(nir_shader *nir);

void agx_preprocess_nir(nir_shader *nir);
void agx_preprocess_nir_g17p(nir_shader *nir);
/* For any native AGX3 chip (agx_native_graphics); passes validated on G17P
 * alone check chip. */
void agx_preprocess_nir_native(nir_shader *nir, enum agx_chip chip);
void agx_preprocess_nir_g18g(nir_shader *nir);

/* Access private (scratch) memory in whole 32-bit words, the only width the
 * AGX3 stack protocol encodes. agx_preprocess_nir_g17p() runs it. */
bool agx3_nir_lower_scratch_words(nir_shader *nir);
void agx_preprocess_nir_m5(nir_shader *nir);
bool agx_nir_lower_discard_zs_emit(nir_shader *s);
bool agx_nir_lower_sample_mask(nir_shader *s);
/* TARGET/LIVE retain their normal semantics. This flag records the proof that
 * the update is either whole-sample testing or a disjoint non-testing kill. */
#define AGX_SAMPLE_MASK_FULL_OR_KILL (1u << 2)
/* Proof about the unmasked source; not permission to emit a full update. */
#define AGX_SAMPLE_MASK_SUBSET_FULL_OR_KILL (1u << 3)
/* sample_mask_agx3 IMM bit: stage the word at staging 2 with scoreboard slot
 * 2 and no class transaction (native1121 psdiscard's update word). */
#define AGX_SAMPLE_MASK_AGX3_SECOND_STAGING (1u << 2)
bool agx_nir_mark_g18_sample_mask_tests(nir_shader *s);
bool agx_nir_coalesce_sample_mask_updates(nir_shader *s, unsigned nr_samples);
bool agx_nir_lower_sample_mask_m5(nir_shader *s);

/* store_zs_agx base bit: the sample mask names exactly one covered sample,
 * and the store sits in a covered-sample loop that zs_transaction_agx
 * brackets (M5 only). The backend stages one DEPTHF word for it. */
#define AGX_ZS_COVERED_SAMPLE (4)
bool agx_nir_lower_sample_mask_deferred(nir_shader *s);
bool agx_nir_lower_sample_mask_one_update(nir_shader *s);
bool agx_nir_lower_sample_mask_one_update_m5(nir_shader *s);
bool agx_nir_lower_sample_mask_deferred_g17p(nir_shader *s, bool *a2c_in_main);
bool agx_nir_lower_sample_mask_deferred_m5(nir_shader *s);
bool agx_nir_lower_interpolation(nir_shader *s, bool layer_as_varying);
bool agx_nir_fix_g17g_constant_interpolation(nir_shader *s);

bool agx_nir_lower_clip_m1_1(struct nir_shader *s);
bool agx_nir_clip_m1_1_complete(struct nir_shader *s);
bool agx_nir_lower_cull_distance_vs(struct nir_shader *s);
bool agx_nir_lower_cull_distance_fs(struct nir_shader *s,
                                    unsigned nr_distances,
                                    bool native_constant_planes);
bool agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                          unsigned bit_size, unsigned num_components,
                          int64_t hole_size, nir_intrinsic_instr *low,
                          nir_intrinsic_instr *high, void *data);

/* Returns false when the selected backend cannot encode the shader.  On
 * failure, out is cleared and owns no allocation. */
/* Private extension range ends at the audited 360-byte CProg frame, including
 * its last spill word. Hardware qualification is required before publication. */
#define AGX3_CPROG_STATIC_SPILL_MAX_B 360u
#define AGX3_CPROG_STATIC_SPILL_VERSION 1u
/* Fragment TI resources have a separate hardware qualification scope. Keep
 * the published vertex version/range unchanged and never cross-admit tags. */
#define AGX3_CPROG_FS_STATIC_SPILL_MAX_B 192u
#define AGX3_CPROG_FS_STATIC_SPILL_VERSION 2u
bool agx3_cprog_static_spill_info_valid(const struct agx_shader_info *info);

bool agx_compile_shader_nir(nir_shader *nir, struct agx_shader_key *key,
                            struct agx_shader_part *out);

struct agx_occupancy {
   unsigned max_registers;
   unsigned max_threads;
};

struct agx_occupancy agx_occupancy_for_register_count(unsigned halfregs);
unsigned agx_max_registers_for_occupancy(unsigned occupancy);

/* Vertex attributes a vertex prolog can hand to the main shader on this
 * device through the register ABI (AGX_ABI_VIN_ATTRIB): component i of the
 * compacted attribute slots occupies half-registers 2 * (8 + i) and the one
 * above it, and the allocator's export accounting needs one more half above
 * the last one, so a register file of N halves holds (N - 17) / 8 vec4
 * attributes -- 29 on the 256-half AGX2 file, 21 on G17P's 192 general
 * halves. Never above AGX_MAX_ATTRIBS. This is the device's
 * maxVertexInputAttributes.
 */
unsigned agx_max_vertex_attribs(const struct agx_device_key *dev);

#ifdef __cplusplus
}
#endif

static inline unsigned
agx_round_registers(unsigned halfregs)
{
   return agx_occupancy_for_register_count(halfregs).max_registers;
}

static const nir_shader_compiler_options agx_nir_options = {
   .lower_fdiv = true,
   .float_mul_add16 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .float_mul_add32 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_fpow = true,
   .lower_fmod = true,
   .lower_bitfield_extract8 = true,
   .lower_bitfield_extract16 = true,
   .lower_bitfield_insert = true,
   .lower_ifind_msb = true,
   .lower_find_lsb = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_fisnormal = true,
   .lower_scmp = true,
   .lower_isign = true,
   .lower_fsign = true,
   .lower_iabs = true,
   .lower_fminmax_signed_zero = true,
   .lower_fdph = true,
   .lower_ffract = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_64_2x32 = true,
   /* A 64-bit value is assembled from two 32-bit register words. Keep the
    * pack_32_2x16_split pair instead of letting nir_opt_algebraic fuse it into
    * a four-way 16-bit pack, which has no register form here. */
   .lower_pack_64_4x16 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_extract_byte = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .has_cs_global_id = true,
   .lower_device_index_to_zero = true,
   .lower_hadd = true,
   .has_amul = true,
   .has_isub = true,
   .has_load_global_bounded = true,
   .support_16bit_alu = true,
   .max_unroll_iterations = 32,
   .lower_uniforms_to_ubo = true,
   .late_lower_int64 = true,
   .lower_int64_options =
      (nir_lower_int64_options) ~(nir_lower_iadd64 | nir_lower_imul_2x32_64),
   .lower_doubles_options = (nir_lower_doubles_options)(~0),
   .support_indirect_inputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
                              BITFIELD_BIT(MESA_SHADER_TESS_EVAL) |
                              BITFIELD_BIT(MESA_SHADER_FRAGMENT),
   .support_indirect_outputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_STAGES),
   /* Compiler range bound covers M5's native eight-sample path. Device
    * properties retain each chip's supported sample counts. */
   .max_samples = 8,
   .lower_fquantize2f16 = true,
   .compact_arrays = true,
   .discard_is_demote = true,
   .scalarize_ddx = true,
   .io_options = nir_io_always_interpolate_convergent_fs_inputs,
};
