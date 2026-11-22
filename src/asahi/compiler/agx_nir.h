/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

struct nir_shader;
struct nir_instr;
struct nir_builder;
struct nir_def;

/* BASE bit3 selects an explicit byte-scaled AGX3 load/store index. */
#define AGX_NIR_MEMORY_INDEX_BYTES 8u
/* BASE bits4..6 carry the native packed element count (1..4). */
#define AGX_NIR_MEMORY_PACKED_COUNT_SHIFT 4u
#define AGX_NIR_MEMORY_PACKED_COUNT_MASK (7u << AGX_NIR_MEMORY_PACKED_COUNT_SHIFT)
/* BASE bit7 marks a late AGX3B W8 load whose unsigned lanes are unpacked to
 * words. It does not change the memory transaction or address. */
#define AGX_NIR_MEMORY_UNPACK_W8_WORD (1u << 7)
/* BASE bits16..31 carry a signed byte offset that AGX3C adds to the address
 * after scaling the index (the lddev/stdev immediate offset). */
#define AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT 16u
#define AGX_NIR_MEMORY_BYTE_OFFSET_MASK (0xffffu << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT)
/* FLAGS bits of load_scs_indexed_agx and load_scs_base_agx (G17P). TOUCH makes
 * the load destination-less: the same address is accessed, but nothing is
 * written to the intrinsic's value, which must have no use. The indexed form
 * reads the low 16 bits of its index and scales them by COMPONENT, 2 or 4.
 * The fence bits add a traversal barrier before the touch (the scope used
 * ahead of a launch) or after it (the scope used at the end of a traversal
 * program). */
#define AGX_NIR_SCS_TOUCH 16u
#define AGX_NIR_SCS_TOUCH_FENCE_BEFORE 32u
#define AGX_NIR_SCS_TOUCH_FENCE_AFTER 64u
#define AGX_NIR_SCS_TOUCH_FLAGS                                                \
   (AGX_NIR_SCS_TOUCH | AGX_NIR_SCS_TOUCH_FENCE_BEFORE |                       \
    AGX_NIR_SCS_TOUCH_FENCE_AFTER)
/* byte_offsets: fold constant index addends into that byte offset.
 * checked_offsets: see agx_shader_key::checked_offsets. */
bool agx_nir_lower_address(struct nir_shader *shader, bool byte_indices,
                           bool byte_offsets, bool checked_offsets);
bool agx_nir_preserve_position_order(struct nir_shader *shader);
bool agx_nir_scalarize_int64(struct nir_shader *shader);
bool agx_nir_lower_divmod64(struct nir_shader *shader, bool native_add64);
bool agx_nir_lower_algebraic_late(struct nir_shader *shader);
bool agx_nir_avoid_square_overflow(struct nir_shader *shader);
bool agx_nir_lower_f2bf_rtne(struct nir_shader *shader);
bool agx_nir_lower_g17p_tile_samples(struct nir_shader *s, unsigned nr_samples);
bool agx_nir_lower_g18_tile_samples(struct nir_shader *s, unsigned nr_samples);
bool agx_nir_lower_g17p_1x_tile_samples(struct nir_shader *s);
bool agx_nir_fold_d3d_float_to_int(struct nir_shader *shader);
bool agx_nir_opt_u8_absdiff(struct nir_shader *shader);
bool agx_nir_opt_m5_narrow_to_float(struct nir_shader *shader);
bool agx_nir_opt_popcount_idiom(struct nir_shader *shader);
bool agx_nir_fold_pixel_coord(struct nir_shader *shader);
bool agx_nir_fold_extract16_bfe(struct nir_shader *shader);
bool agx_nir_cleanup_amul(struct nir_shader *shader);
bool agx_nir_fuse_lea(struct nir_shader *shader);
bool agx_nir_lower_lea(struct nir_shader *shader);
bool agx_nir_fuse_selects(struct nir_shader *shader);
bool agx_nir_m5_split_or_selects(struct nir_shader *shader);
bool agx_nir_opt_select_load(struct nir_shader *shader);
bool agx_nir_opt_flatten_loads(struct nir_shader *shader, unsigned limit,
                               unsigned max_loads);
bool agx_nir_opt_sink_loop_sample_uses(struct nir_shader *shader);
bool agx_nir_opt_hoist_loop_samples(struct nir_shader *shader);
bool agx_nir_fuse_algebraic_late(struct nir_shader *shader);
bool agx_nir_fuse_doubled_add(struct nir_shader *shader);
bool agx_nir_fold_byte_to_float(struct nir_shader *shader);
bool agx_nir_fold_ubyte_to_float(struct nir_shader *shader);
bool agx_nir_fence_images(struct nir_shader *shader);
bool agx_nir_lower_layer(struct nir_shader *s);
bool agx_nir_lower_clip_distance(struct nir_shader *s);
bool agx_nir_lower_subgroups(struct nir_shader *s);
bool agx_nir_lower_subgroups_g17p(struct nir_shader *s);
bool agx_nir_lower_subgroups_m5(struct nir_shader *s, bool native_shuffle_xor,
                                bool native_float_minmax);
bool agx_nir_flush_subgroup_minmax(struct nir_shader *s);
bool agx_nir_lower_dynamic_shuffle_xor(struct nir_shader *s);
bool agx_nir_lower_quad_derivative_ids(struct nir_shader *s);
bool agx_nir_lower_compute_derivatives(struct nir_shader *s);
bool agx_workgroup_split_size(const struct nir_shader *nir, uint16_t size[3]);
bool agx_nir_lower_workgroup_split(struct nir_shader *nir,
                                   const uint16_t size[3]);
struct nir_def *agx_nir_quad_ddx(struct nir_builder *b, struct nir_def *v);
struct nir_def *agx_nir_quad_ddy(struct nir_builder *b, struct nir_def *v);

/* AGX3 ldtg/sttg index threadgroup memory in scalar elements, scaled by the
 * access width; NIR shared offsets are bytes.  Idempotent, so a frontend may
 * run it as early as the optimizer can use it and agx_compile_shader_nir()
 * still guarantees it for any AGX3 target. */
bool agx_nir_lower_shared_element_index(struct nir_shader *s);
bool agx_nir_opt_shared_byte_offsets(struct nir_shader *s);
bool agx_nir_opt_licm(struct nir_shader *shader, unsigned budget);
bool agx_nir_lower_fminmax(struct nir_shader *s, bool ftz_only);
bool agx_nir_lower_fminmax_signed_zero(struct nir_shader *s, bool ftz_only);
bool agx_nir_lower_word_bounds(struct nir_shader *s);
bool agx_nir_share_bounds(struct nir_shader *s);
bool agx_nir_lower_atomics(struct nir_shader *s, bool native_fadd);
bool agx_nir_opt_canonical_minmax(struct nir_shader *s);

bool agx_nir_lower_texture_early(struct nir_shader *s, bool support_lod_bias);
bool agx_nir_lower_bound_image_atomics(struct nir_shader *s);
bool agx_nir_lower_bound_ms_image_stores(struct nir_shader *s);
bool agx_nir_lower_g17p_helper_samples(struct nir_shader *s);
bool agx_nir_lower_g17p_texture_early(struct nir_shader *s,
                                     bool support_lod_bias,
                                     bool sampler_has_lod_bias);
bool agx_nir_lower_texture(struct nir_shader *s);
/* G18G (M6) shares the G17G family's Gen10 descriptors, but keeps the texture
 * lowering forms the J873 GL 4.6 campaign qualified. A G18G caller ORs this
 * into the generation it passes to agx_nir_lower_texture_for_generation(),
 * agx_nir_lower_texture_with_buffer_metadata[_cb]() and
 * agx_nir_lower_multisampled_image_store_for_generation(); the pass lowers
 * for the remaining descriptor generation (10). */
#define AGX_NIR_TEXTURE_GENERATION_G18G (1u << 8)
/* A caller whose Gen10 storage descriptors keep the image-atomic addressing
 * word after the texture/PBE pair (AGX_G17P_IMAGE_ATOMIC_SW_OFFSET, as
 * Honeykrisp's 96-byte storage record does) without the buffer-metadata
 * record ORs this into the generation it passes to
 * agx_nir_lower_texture_for_generation(). Without it, Gen10 image atomics
 * read the word from the PBE's last eight bytes, where Gallium keeps it.
 * Texel-buffer atomics are not affected. */
#define AGX_NIR_TEXTURE_IMAGE_SIDEBAND (1u << 9)
bool agx_nir_lower_texture_for_generation(struct nir_shader *s, unsigned generation);
/* robust_formatless_images: formatless image loads have not been lowered for
 * view-bounds robustness and require it. */
/* sparse_buffer_residency: ordinary texel-buffer reads must consult the
 * software sparse residency map, because the descriptor names the primary
 * mapping rather than the read-only shadow (hk_buffer_view.c). */
bool agx_nir_lower_texture_with_buffer_metadata(struct nir_shader *s,
                                               unsigned generation,
                                               bool robust_formatless_images,
                                               uint64_t sparse_ro_offset,
                                               bool sparse_buffer_residency);
struct nir_builder;
struct nir_def;
struct nir_tex_instr;
typedef struct nir_def *(*agx_buffer_texture_metadata_cb)(
   struct nir_builder *b, struct nir_tex_instr *tex);
bool agx_nir_lower_texture_with_buffer_metadata_cb(
   struct nir_shader *s, unsigned generation,
   agx_buffer_texture_metadata_cb buffer_metadata, bool robust_texel_fetch);
bool agx_nir_lower_multisampled_image_store(struct nir_shader *s);
bool agx_nir_lower_multisampled_image_store_for_generation(struct nir_shader *s,
                                                         unsigned generation);
bool agx_nir_needs_texture_crawl(struct nir_instr *instr);

/* Local arrays larger than this many bytes are lowered to scratch memory by
 * agx_preprocess_nir(); smaller ones become csel ladders. */
#define AGX_SCRATCH_ARRAY_THRESHOLD_B 256

/* Existing reserved software tail, relative to a Gen9 sampled texture. */
#define AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET 68
/* G17P appends one native texture-heap index after the 64-byte texture/PBE
 * storage descriptor pair. Other generations retain their existing ABI. */
#define AGX_G17P_STORAGE_HEAP_INDEX_OFFSET 64

/* G17P storage descriptors reserve the following eight bytes for the two
 * texture-heap indices.  Image atomics keep their software address metadata
 * in a separate packed word after those indices, 40 bytes past the native PBE
 * handle seen by agx_nir_lower_texture_for_generation(). */
#define AGX_G17P_IMAGE_ATOMIC_SW_OFFSET 72
#define AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE 40

/* Pack that word: level offset [0:26] and layer stride [35:61] in 128-byte
 * units, log2 tile width [27:30] and height [31:34] in pixels, log2 sample
 * count [62:63]. libagx_image_texel_address_gen9() decodes it. Returns false
 * when the addressing does not fit. Honeykrisp packs the same word with
 * hk_pack_g17p_image_atomic_sw(), which stays in its dependency-free header. */
static inline bool
agx_g17p_pack_image_atomic_sw(uint64_t level_offset_B, unsigned tile_width_px,
                              unsigned tile_height_px, uint64_t layer_stride_B,
                              unsigned samples, uint64_t *out)
{
   if (!out || (level_offset_B & 0x7f) || (layer_stride_B & 0x7f) ||
       tile_width_px == 0 || (tile_width_px & (tile_width_px - 1)) ||
       tile_height_px == 0 || (tile_height_px & (tile_height_px - 1)) ||
       (samples != 1 && samples != 2 && samples != 4 && samples != 8))
      return false;

   unsigned tile_width_log2 = 0, tile_height_log2 = 0, samples_log2 = 0;
   for (unsigned x = tile_width_px; x > 1; x >>= 1)
      ++tile_width_log2;
   for (unsigned x = tile_height_px; x > 1; x >>= 1)
      ++tile_height_log2;
   for (unsigned x = samples; x > 1; x >>= 1)
      ++samples_log2;

   const uint64_t level_offset = level_offset_B >> 7;
   const uint64_t layer_stride = layer_stride_B >> 7;
   if (level_offset >= (UINT64_C(1) << 27) ||
       layer_stride >= (UINT64_C(1) << 27) || tile_width_log2 >= 16 ||
       tile_height_log2 >= 16 || samples_log2 >= 4)
      return false;

   *out = level_offset | ((uint64_t)tile_width_log2 << 27) |
          ((uint64_t)tile_height_log2 << 31) | (layer_stride << 35) |
          ((uint64_t)samples_log2 << 62);
   return true;
}
/* Native texture/PBE pair, heap indices, address sideband, then padding. */
#define AGX_G17P_EMRT_DESCRIPTOR_SIZE 96
#define AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET 80
/* Declines without changing the shader, so callers need no copy. g17p
 * admits the shapes only G17P's backend encodes (agx_optimizer_inline_imm). */
bool agx_nir_lower_g17p_indexed_samplers(struct nir_shader *s, bool g17p);
bool agx_nir_lower_g17g_indexed_reads(struct nir_shader *s);

struct agx_g17p_combined_sampler_binding {
   uint32_t texture_offset_B;
   uint32_t sampler_index_offset_B;
};

/* Native compute texture/sampler table pointers occupy halves0..7. */
#define AGX_G17P_TXF_ROOT_HALFS 8u
/* Native read_write textures consume four four-half binding slots: the read
 * texture, the write PBE, their access-control descriptor, and the coordinate
 * clamp vector. */
#define AGX_G17P_STORAGE_ROOT_HALFS 16u
/* G18G's write-only state-0 root is two pointers, halves 0..7: the resolved
 * PBE and a sampler-shaped state record (native run112 kick 19 launch 1
 * loads exactly these 16 bytes into u0..u3). */
#define AGX_G18G_STORAGE_ROOT_HALFS 8u
/* Initializer of that 32-byte state record, shared by Gallium and hk. It is
 * also entry 0 of the sampler table on the G18G image-write routes. Moved
 * unchanged from Gallium's agx_g18g_native_write_state1 (agx_state.c), whose
 * comment records where the value comes from and how it decodes. */
#define AGX_G18G_WRITE_STATE1_INIT                                             \
   {                                                                           \
      0x00, 0x00, 0x0f, 0x08, 0x00, 0x07, 0x00, 0x00,                          \
   }
struct agx_g17p_txf_binding {
   uint32_t texture_offset_B;
};

/* Native compute image read/write uses one texture-state and one PBE-state
 * pointer in the same four-quad prefix as the existing TXF path. */
struct agx_g17p_storage_image_binding {
   uint32_t texture_offset_B;
   uint32_t pbe_offset_B;
   bool write_only;
   /* G18G multi-image table: bytes from texture_offset_B the state table
    * spans (0 = one image: 32 write-only, else 64). */
   uint32_t table_size_B;
   /* G18G: descriptor set holding the table (0 for the older routes). */
   uint8_t set;
};

/* Apple's sampled-texture-to-write-image kernel ABI places a sampled texture,
 * its native sampler, the destination PBE state, and its write control in
 * uniform slots 0..3.  Keep the two descriptor sets explicit: Vulkan does not
 * require sampled and storage images to share a set. */
struct agx_g17p_sampled_storage_binding {
   uint32_t texture_offset_B;
   uint32_t sampler_index_offset_B;
   uint32_t pbe_offset_B;
   uint8_t texture_set;
   uint8_t pbe_set;
};

/* Indexed texture/sampler heaps occupy native slots 0/0; a formatted storage
 * destination shares that ABI by occupying PBE state 1. */
struct agx_g17p_indexed_storage_binding {
   uint32_t pbe_offset_B;
   uint8_t pbe_set;
};

bool agx_nir_lower_g17p_indexed_storage(
   struct nir_shader *s,
   struct agx_g17p_indexed_storage_binding *binding);

/* Turn one statically indexed set-0 Gen9 combined image/sampler used by the
 * initial G17P graphics or compute path into texture-state 0 / sampler-state
 * 0. The caller must place the corresponding native states in those CProg
 * regions.
 */
/* Whether agx_nir_lower_g17p_combined_sampler() may apply: a false result
 * is final, a true one can still be refused while lowering. */
bool agx_nir_g17p_combined_sampler_candidate(const struct nir_shader *s,
                                             bool g17p);

bool agx_nir_lower_g17p_combined_sampler(
   struct nir_shader *s, struct agx_g17p_combined_sampler_binding *binding,
   bool g17p);

/* Gen10 shares the 32-byte texture/eight-byte sampler descriptor layout.
 * Initially admit only a single 2D float sample at explicit level zero. */
bool agx_nir_lower_g17g_combined_sampler(
   struct nir_shader *s, struct agx_g17p_combined_sampler_binding *binding);
bool agx_nir_lower_g17g_indexed_reads(struct nir_shader *s);

/* Turn the one statically indexed set-0 Gen9 texture used by the initial
 * G17P compute texel-fetch path into texture-state 0.  Sampler-state 0 remains
 * the device TXF sampler.  The caller must publish both native states through
 * the G17P resource program; no executable shader preamble is generated.
 */
bool agx_nir_lower_g17p_txf(
   struct nir_shader *s, struct agx_g17p_txf_binding *binding);

/* Turn one set-0, single-sample 2D RGBA8UI read/write image into native
 * texture-state 0 / PBE-state 0. Every other image shape stays fail-closed. */
bool agx_nir_lower_g17p_storage_image(
   struct nir_shader *s, struct agx_g17p_storage_image_binding *binding);

/* G18G: the same state-0 route for one storage image of any colour format,
 * write-only or read-write (a {texture, PBE} table of two entries). */
bool agx_nir_lower_g18g_storage_image(
   struct nir_shader *s, struct agx_g17p_storage_image_binding *binding);

/* G18G: several storage images in one descriptor set through one state
 * table spanning their descriptors (binding->table_size_B). */
bool agx_nir_lower_g18g_storage_images(
   struct nir_shader *shader, struct agx_g17p_storage_image_binding *binding);

/* G18G per-dispatch native tables (agx_nir_lower_g18g_combined_tables):
 * sampled textures and storage images in one compute shader, in any sets. */
#define AGX_G18G_TABLE_MAX_TEXTURES 16
#define AGX_G18G_TABLE_MAX_SAMPLERS 8
struct agx_g18g_table_entry {
   uint8_t set;
   uint32_t offset_B;
};
struct agx_g18g_table_binding {
   /* Texture-state entries: 32-byte copies from set memory. */
   uint8_t texture_count;
   /* Sampler entries 1..: heap records named by the published 16-bit index
    * at (set, offset_B); entry 0 is the G18G write-state record. */
   uint8_t sampler_count;
   struct agx_g18g_table_entry textures[AGX_G18G_TABLE_MAX_TEXTURES];
   struct agx_g18g_table_entry samplers[AGX_G18G_TABLE_MAX_SAMPLERS - 1];
};
bool agx_nir_lower_g18g_combined_tables(struct nir_shader *shader,
                                        struct agx_g18g_table_binding *binding);

/* Lower Gallium's exact write-only scalar-integer image state to native
 * G17P PBE state 0. The caller publishes the resolved PBE pointer and native
 * write-control record in the first four 64-bit resource slots. */
bool agx_nir_lower_g17p_gallium_storage_write(struct nir_shader *s,
                                               unsigned pbe_state);

/* Lower the exact compute textureLod(..., 0) -> rgba8 imageStore form emitted
 * by vkQuake.  The caller publishes the four-slot native resource tuple. */
bool agx_nir_lower_g17p_sampled_storage(
   struct nir_shader *s, struct agx_g17p_sampled_storage_binding *binding);

bool agx_nir_lower_to_per_sample_agx2(struct nir_shader *shader);
bool agx_nir_lower_monolithic_msaa_agx2(struct nir_shader *shader, uint8_t nr_samples);

bool agx_nir_lower_denorm_preserve(struct nir_shader *shader);
bool agx_nir_fma_rte_applies(const struct nir_shader *shader);
bool agx_nir_lower_fma_rte(struct nir_shader *shader);
struct nir_def *agx_nir_fma_rte_fixup(struct nir_builder *b, struct nir_def *a,
                                    struct nir_def *c, struct nir_def *z,
                                    struct nir_def *native);
bool agx_nir_lower_cmat(struct nir_shader *s);
bool agx_nir_lower_cmat_calls(struct nir_shader *s);
bool agx_nir_cmat_supported(struct nir_shader *s);
bool agx_nir_lower_cmat_tensor(struct nir_shader *s, bool pairing);
bool agx_nir_lower_cmat_flexible(struct nir_shader *s, bool *pairing);

bool agx_nir_lower_texture_early_with_2d_robustness(struct nir_shader *s,
                                                    bool support_lod_bias,
                                                    bool lower_2d_robustness);

bool agx_nir_merge_global_words(struct nir_shader *shader);
bool agx_nir_narrow_single_subgroup_barriers(struct nir_shader *shader);
bool agx_nir_narrow_subgroup0_barriers(struct nir_shader *shader);
bool agx_nir_opt_uniform_stride(struct nir_shader *shader);
bool agx_nir_fold_m5_offsets(struct nir_shader *shader);
bool agx_nir_unroll_runtime_loops(struct nir_shader *shader, unsigned factor,
                                  unsigned max_instrs, unsigned *widest);
