/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_bo.h"
#include "agx_ioto_program_core.h"
#include "agx_compile.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "nir_lower_blend.h"
#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx_g17p_graphics_template;
struct agx_m5_graphics_template;

struct agx_linked_shader {
   /* Mapped executable memory */
   struct agx_bo *bo;

   /* Gallium-owned, bounded Ioto template cache. Entries are ralloc children
    * of this linked shader and contain no live batch allocations. */
   struct agx_g17p_graphics_template *g17p_graphics_templates;

   /* The same for the M5 family's native programs (agx_state.c). */
   struct agx_m5_graphics_template *m5_graphics_templates;

   /* Exact executable byte count. BO allocations are page-rounded, so the
    * BO size is not a safe disassembler bound. */
   size_t size;

   /* Byte offset of the entry point inside `bo`.
    *
    * Zero everywhere except G17P, where the executable is wrapped in Apple's
    * AGX3 code container and the entry sits at +0x3C0.  The shader loader
    * reads the container's block-size dword at `entry - 0x80`, so the entry
    * address handed to LDSHDR is `bo->va->addr + entry_offset`, never the BO
    * base -- a shader at offset 0 has nothing mapped in front of it and the
    * loader's first read takes a GMMU page fault. */
   uint32_t entry_offset;

   /* Aggregated main-program resources after fast linking. */
   uint32_t scratch_size;
   uint32_t agx3_spectpr_scratch_size;
   uint16_t nr_gprs;
   uint16_t g17p_emrt_heap_base;
   uint16_t g17p_blend_constant_base;

   /* Apple's compiler ProgramVariant field 38 describes staging bytes,
    * independently of GPR count. Valid only for accounted-for staging paths.
    */
   bool g17p_compiler_d_valid;
   uint32_t g17p_compiler_d;

   /* Compiler-proven G17P colour pixel classes consumed by the linked parts,
    * bit n for the tile partitions n, n + 3, ... */
   uint8_t g17p_colour_classes;

   /* Set if the linked SW vertex shader reads base vertex/instance. The VS
    * prolog can read base instance even when the API VS does not, which is why
    * this needs to be aggregated in the linker.
    */
   bool uses_base_param;

   /* Set if the linked shader uses txf. The epilog may even if the main shader
    * does not, in the case of spilled render targets.
    */
   bool uses_txf;

   /* Whether the program is a no-op and can be skipped. */
   bool no_op;

   /* Coefficient register bindings */
   struct agx_varyings_fs cf;

   /* Data structures packed for the linked program */
   struct agx_usc_shader_packed shader;
   struct agx_usc_registers_packed regs;
   struct agx_usc_fragment_properties_packed fragment_props;
   struct agx_output_select_packed osel;
   struct agx_fragment_control_packed fragment_control;
};

/* Write the AGX3 code container's fixed leading block. Its size lives in
 * agx_ioto_program_core.h so decoders can share it. */
void agx3_build_container_lead(uint8_t *image, enum agx_chip chip,
                               unsigned cores_per_cluster);
void agx3_build_usc_base_page(uint8_t *image, size_t size, enum agx_chip chip,
                              unsigned cores_per_cluster,
                              unsigned num_clusters);

/* `uvs_map` (agx_uvs_relocation_map()) places the varyings of a vertex
 * program with a relocatable UVS layout, which cannot link without it. */
bool agx_fast_link(struct agx_linked_shader *linked, struct agx_device *dev,
                   bool fragment, struct agx_shader_part *main,
                   struct agx_shader_part *prolog,
                   struct agx_shader_part *epilog, unsigned nr_samples_shaded,
                   const uint8_t *uvs_map);

/* These parts of the vertex element affect the generated code */
struct agx_velem_key {
   uint32_t divisor;
   uint16_t stride;
   uint8_t format;
   bool instanced;
};

struct agx_vs_prolog_key {
   struct agx_velem_key attribs[AGX_MAX_VBUFS];

   /* Bit mask of attribute components to load */
   BITSET_DECLARE(component_mask, AGX_MAX_ATTRIBS * 4);

   /* Whether running as a hardware vertex shader (versus compute) */
   bool hw;

   /* First uniform half of the ordinary vertex ABI. G17P reserves the
    * preceding halves for texture/sampler table pointers when the VS samples
    * textures. */
   uint8_t uniform_base;

   /* Whether the shader uses static vertex input and the prolog just fixes up
    * the vertex ID.
    */
   bool static_vi;

   /* If !hw and the draw call is indexed, the index size */
   uint8_t sw_index_size_B;

   /* If !hw, fetch by the instance ID plus poly_vertex_params::instance_offset
    * while still exporting the dispatch's instance ID.
    */
   bool sw_instance_offset;

   /* Adjacency primitive to emulate */
   enum mesa_prim adjacency;

   /* Robustness settings for the vertex fetch */
   struct agx_robustness robustness;
};

struct agx_fs_prolog_key {
   /* glSampleMask() mask */
   uint8_t api_sample_mask;

   /* Number of cull planes requiring lowering */
   uint8_t cull_distance_size;

   /* Need to count FRAGMENT_SHADER_INVOCATIONS */
   bool statistics;

   /* Need to lower desktop OpenGL polygon stipple */
   bool polygon_stipple;

   /* If we discard, whether we need to run Z/S tests */
   bool run_zs_tests;

   /* G17P rasterization samples its coverage updates are compiled for */
   uint8_t nr_samples;

   /* If we emulate cull distance, the base offset for our allocated coefficient
    * registers so we don't interfere with the main shader.
    */
   unsigned cf_base;
};

struct agx_blend_rt_key {
   unsigned colormask      : 4;
   unsigned advanced_blend : 1;
   unsigned pad            : 1;
   unsigned mode           : 26;
};
static_assert(sizeof(struct agx_blend_rt_key) == 4, "packed");

struct agx_blend_standard {
   enum pipe_blend_func rgb_func          : 3;
   enum pipe_blendfactor rgb_src_factor   : 5;
   enum pipe_blendfactor rgb_dst_factor   : 5;
   enum pipe_blend_func alpha_func        : 3;
   enum pipe_blendfactor alpha_src_factor : 5;
   enum pipe_blendfactor alpha_dst_factor : 5;
   unsigned pad                           : 6;
};
static_assert(sizeof(struct agx_blend_standard) == 4, "packed");

static inline unsigned
agx_pack_blend_standard(enum pipe_blend_func rgb_func,
                        enum pipe_blendfactor rgb_src_factor,
                        enum pipe_blendfactor rgb_dst_factor,
                        enum pipe_blend_func alpha_func,
                        enum pipe_blendfactor alpha_src_factor,
                        enum pipe_blendfactor alpha_dst_factor)
{
   struct agx_blend_standard blend = {
      .rgb_func = rgb_func,
      .rgb_src_factor = rgb_src_factor,
      .rgb_dst_factor = rgb_dst_factor,
      .alpha_func = alpha_func,
      .alpha_src_factor = alpha_src_factor,
      .alpha_dst_factor = alpha_dst_factor,
   };
   unsigned val;
   typed_memcpy(&val, &blend, 1);
   return val;
}

static inline struct agx_blend_standard
agx_unpack_blend_standard(unsigned mode)
{
   struct agx_blend_standard blend;
   typed_memcpy(&blend, &mode, 1);
   return blend;
}

struct agx_blend_advanced {
   enum pipe_advanced_blend_mode op     : 8;
   enum pipe_blend_overlap_mode overlap : 2;
   unsigned src_premultiplied           : 1;
   unsigned dst_premultiplied           : 1;
   unsigned clamp_results               : 1;
   unsigned pad                         : 19;
};
static_assert(sizeof(struct agx_blend_advanced) == 4, "packed");

static inline unsigned
agx_pack_blend_advanced(enum pipe_advanced_blend_mode op,
                        enum pipe_blend_overlap_mode overlap,
                        bool src_premultiplied, bool dst_premultiplied,
                        bool clamp_results)
{
   struct agx_blend_advanced blend = {
      .op = op,
      .overlap = overlap,
      .src_premultiplied = src_premultiplied,
      .dst_premultiplied = dst_premultiplied,
      .clamp_results = clamp_results,
   };
   unsigned val;
   typed_memcpy(&val, &blend, 1);
   return val;
}

static inline struct agx_blend_advanced
agx_unpack_blend_advanced(unsigned mode)
{
   struct agx_blend_advanced blend;
   typed_memcpy(&blend, &mode, 1);
   return blend;
}

struct agx_blend_key {
   struct agx_blend_rt_key rt[8];
   uint8_t logicop_func;
   bool alpha_to_coverage, alpha_to_one;
   bool logicop_enable;
};
static_assert(sizeof(struct agx_blend_key) == 36, "packed");

struct agx_fs_epilog_link_info {
   /* Base index of spilled render targets in the binding table */
   uint8_t rt_spill_base;

   /* Bit mask of the bit size written to each render target. Bit i set if RT i
    * uses 32-bit registers, else 16-bit registers.
    */
   uint8_t size_32;

   /* Mask of locations written by the main shader */
   uint8_t loc_written;

   /* If set, the API fragment shader uses sample shading. This means the epilog
    * will be invoked per-sample as well.
    */
   unsigned sample_shading : 1;

   /* If set, broadcast location #0 value to all render targets. This
    * implements gl_FragColor semantics. This tells the driver to set remap
    * appropriately.
    */
   unsigned broadcast_rt0 : 1;

   /* If set, force location 0's W channel to 1.0. This optimizes blending
    * calculations in some applications.
    */
   unsigned loc0_w_1 : 1;

   /* If set, the API fragment shader wants to write depth/stencil respectively.
    * This happens in the epilog for correctness when the epilog discards.
    */
   unsigned write_z : 1;
   unsigned write_s : 1;

   /* Whether the fragment prolog or main fragment shader already ran tests due
    * to early_fragment_tests. In this case, the epilog must not run tests.
    */
   unsigned already_ran_zs : 1;

   /* Whether the main fragment shader ran tests before discards due to
    * early_fragment_tests. In this case, the epilog must mask the stores in
    * software instead.
    */
   bool sample_mask_after_force_early : 1;

   /* If set, the colour exports are packed by the components each location
    * actually writes rather than laid out on the fragment ABI's
    * four-component slots (struct agx_g17p_colour_layout). A G17P fragment
    * unrolled per sample repeats this layout once per sample, so the packing
    * is what decides whether its exports fit the register file.
    */
   unsigned packed_colour : 1;

   /* Components each colour location exports, four bits per location. Only
    * consulted for the packing above.
    */
   uint32_t colour_comps;
};
static_assert(sizeof(struct agx_fs_epilog_link_info) == 8, "packed");

struct agx_fs_epilog_key {
   struct agx_fs_epilog_link_info link;

   /* Blend state. Blending happens in the epilog. */
   struct agx_blend_key blend;

   /* AGX has no native smooth-line rasterizer. Drivers may request the
    * conservative two-pixel fallback and attenuate the fragment alpha in
    * this epilog before ordinary API blending. */
   bool smooth_line_alpha;

   /* The main fragment shader applied alpha-to-coverage in the coverage update
    * that tests, so this epilog neither kills nor tests for it. */
   bool a2c_in_main;

   /* Colour attachment remapping for Vulkan. Negative values indicate that an
    * attachment is discarded. Positive values indicate the output location we
    * want to store at the indexed colour attachment.
    */
   int8_t remap[8];

   /* Tilebuffer configuration */
   enum pipe_format rt_formats[8];
   /* Optional API format when the tile storage has additional channels. */
   enum pipe_format blend_formats[8];
   /* Optional scaled API format stored through an integer tilebuffer. */
   enum pipe_format scaled_formats[8];
   uint8_t nr_samples;
   /* agx_tilebuffer_layout::max_sample_size_B of the render pass. */
   uint8_t max_sample_size_B;
   /* agx_tilebuffer_layout::resident_mask of the render pass. */
   uint8_t resident_mask;
   bool native_imageblock;
   uint32_t max_bytes_per_tile;
   uint16_t max_samples_per_tile;
   bool force_small_tile;
   bool preserve_zero_mask;
   /* Execute resident MRT preservation separately for each MSAA sample. */
   bool sample_loop_mrt;
   bool lower_srgb;
   bool lower_r11g11b10;
   /* Multisampled tile accesses move whole samples: U8NORM, I32 and F16, no
    * I16 and no channel mask (G18G). */
   bool ms_whole_sample;
   /* The layout keeps attachment 0 alone in the tilebuffer and spills the
    * others (agx_build_tilebuffer_layout_rt0_resident). */
   bool rt0_resident;
   /* Pixel classes the main part already holds (agx_shader_info::
    * agx3_pixel_held); the epilog does not wait for them again. */
   uint8_t main_pixel_held;
   /* M5: the epilog's coverage update is the single native testing form
    * (agx_nir_lower_sample_mask_m5). */
   bool m5_coverage;
   /* G17P tile storage: sRGB as U8NORM bytes converted by the native packs,
    * R11G11B10/RGB9E5 as one raw 32-bit word. */
   bool g17p_tile_formats;
   /* G18G builds the native epilog without G17P's tile formats. */
   bool native_epilog;
   bool native_spills;
   /* Zero retains the Gen9 private-table consumer. Gen10 uses the same
    * software tail with its own native descriptors and texture queries. */
   uint8_t native_spill_generation;
   /* Absolute uniform-half address of the private native spill table. */
   uint16_t emrt_heap_base;

   /* G17P's pre-main MOVMSK also runs depth/stencil tests. For late-tests
    * shaders, defer the API sample mask to the epilog so it can be combined
    * with shader and alpha-to-coverage masks before any output is committed.
    */
   bool defer_api_sample_mask;
   uint8_t api_sample_mask;

   /* Leave pixel-shaded MSAA lowering to a caller that will unroll the
    * per-sample epilog itself. Used by the G17P sample-shaded path.
    */
   bool defer_monolithic_msaa;

   /* Absolute uniform-half base for dynamic blend constants. Zero selects
    * the standard fragment ABI at AGX_ABI_FUNI_BLEND_R. HAL200 resource
    * pointers occupy that standard range, so a linked G17P epilog may place
    * the constants after the main shader's uniform footprint instead. */
   uint16_t blend_constant_base;

   /* Render targets whose tile value must be stored back unchanged (colour
    * write disabled while raster coverage still makes the tile nonempty).
    * Independent of blending, so it also holds for pure-integer formats. */
   uint8_t preserve_rt;

   /* Every colour the main shader exports is
    * provably in [0, 1] (agx_nir_fs_colour_outputs_unit_range), so the blend
    * source clamp nir_lower_blend inserts for UNORM targets is an identity
    * and is dropped. Occupies the key's former tail padding byte. */
   uint8_t colour_unit_range;

   /* The render pass uses a G18 single-sample mixed-format tile
    * (agx_build_tilebuffer_layout_g18_mixed, run 1139): build the epilog's
    * layout the same way and store exactly what Apple's fragments store -- a
    * target whose write mask selects none of its channels is neither loaded
    * nor stored. Set by the driver; zero elsewhere. */
   uint8_t g18_native_mixed;
};

/* Apply framebuffer rules which are properties of the hardware generation,
 * not API state. Keep Gallium and Honeykrisp on the same epilog contract. */
void agx_fs_epilog_apply_device_policy(enum agx_chip chip,
                                       struct agx_fs_epilog_key *key);

/* Depth/stencil exports run tests in the epilog just like alpha-to-coverage
 * can discard there. Forced-early main shaders have already run those tests. */
bool agx_fs_epilog_may_discard(const struct agx_fs_epilog_link_info *link,
                               bool alpha_to_coverage,
                               bool nontrivial_force_early);

void agx_nir_vs_prolog(struct nir_builder *b, const void *key_);
void agx_nir_vs_prolog_raw(struct nir_builder *b, const void *key_);
void agx_nir_fs_epilog(struct nir_builder *b, const void *key_);
void agx_nir_fs_epilog_agx2(struct nir_builder *b, const void *key_);
void agx_nir_fs_epilog_g17p(struct nir_builder *b, const void *key);
bool agx_nir_lower_g17p_sample_shading(
   nir_shader *nir, struct agx_fs_epilog_link_info *link, unsigned samples);
bool agx_nir_lower_g17p_sample_shading_file(
   nir_shader *nir, struct agx_fs_epilog_link_info *link, unsigned samples,
   unsigned registers, bool eight_samples);
void agx_nir_fs_prolog(struct nir_builder *b, const void *key_);

bool agx_nir_gather_vs_inputs(nir_shader *s,
                              BITSET_WORD *attrib_components_read);

bool agx_nir_lower_vs_input_to_prolog(nir_shader *s);

bool agx_nir_lower_fs_output_to_epilog(nir_shader *s,
                                       struct agx_fs_epilog_link_info *out);

/* True if every colour output component of this (not yet lowered) fragment
 * shader is a float provably in [0, 1]. Conservative: unknown is false. */
bool agx_nir_fs_colour_outputs_unit_range(nir_shader *s);
bool agx_nir_lower_fs_output_to_epilog_g17p(
   nir_shader *s, struct agx_fs_epilog_link_info *out);
bool agx_nir_lower_fs_output_to_epilog_for_device(
   nir_shader *s, enum agx_chip chip, struct agx_fs_epilog_link_info *out);

/* Pack the colour exports of a fragment already split for its epilog, so a
 * G17P per-sample unroll costs only the components each location writes. */
bool agx_nir_pack_g17p_colour_exports(nir_shader *s,
                                      struct agx_fs_epilog_link_info *link);

/* The colour layout the split above would record, read off an unlowered
 * shader and rounded up wherever it cannot be read exactly. */
void agx_nir_gather_fs_colour_exports(const nir_shader *s,
                                      uint8_t *loc_written, uint8_t *size_32,
                                      uint32_t *colour_comps);

bool agx_nir_lower_g17p_tile_samples(nir_shader *s, unsigned nr_samples);
bool agx_nir_lower_g17p_1x_tile_samples(nir_shader *s);

/* msaa163: the four-sample G18G form of the lowering above.
 * The live coverage is the 16-bit load_active_samples_agx, which the G18G
 * backend reads as W16 SR39 -- Apple's G18 4x fragment passes that register
 * straight to lstcol. Only four samples are admitted; callers gate it. */
bool agx_nir_lower_g18_tile_samples(nir_shader *s, unsigned nr_samples);

/* G17P currently exposes only single-sample rasterization. Collapse API
 * sample shading before split FS lowering so fast-linking never needs the
 * legacy AGX2 external sample-loop byte sequences. */
bool agx_nir_lower_g17p_1x_sample_shading(nir_shader *s);

bool agx_nir_lower_fs_active_samples_to_register(nir_shader *s);

bool agx_nir_lower_non_monolithic_uniforms(nir_shader *nir, unsigned nr);
bool agx_nir_lower_non_monolithic_uniforms_at(nir_shader *nir, unsigned nr,
                                              unsigned uniform_base);

#ifdef __cplusplus
} /* extern C */
#endif
