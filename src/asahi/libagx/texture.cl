/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_abi.h"
#include "compiler/libcl/libcl.h"
#include "libagx_intrinsics.h"
#include <agx_pack.h>

uint3
libagx_txs(constant struct agx_texture_packed *ptr, uint16_t lod,
           unsigned nr_comps, bool is_buffer, bool is_1d, bool is_2d,
           bool is_cube, bool is_array)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   /* From the Vulkan spec:
    *
    *    OpImageQuery*...  return 0 if the bound descriptor is a null descriptor
    */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* Buffer textures are lowered to 2D so the original size is irrecoverable.
    * Instead, we stash it in the software-defined section.
    */
   if (is_buffer)
      return d.buffer_size_sw;

   /* Load standard dimensions */
   uint3 size = (uint3)(d.width, d.height, d.depth);
   lod += d.first_level;

   /* Linear 2D arrays are special.
    *
    * TODO: Optimize this, since linear 2D arrays aren't needed for APIs and
    * this just gets used internally for blits.
    */
   if (is_2d && is_array && d.layout == AGX_LAYOUT_LINEAR)
      size.z = d.depth_linear;

   /* 1D Arrays have their second component as the layer count */
   if (is_1d && is_array)
      size.y = size.z;

   /* Adjust for LOD, do not adjust array size */
   size.x = max(size.x >> lod, 1u);

   if (nr_comps - (uint)is_array >= 2)
      size.y = max(size.y >> lod, 1u);

   if (nr_comps - (uint)is_array >= 3)
      size.z = max(size.z >> lod, 1u);

   /* Cube maps have equal width and height, we save some instructions by only
    * reading one. Dead code elimination will remove the redundant instructions.
    */
   if (is_cube)
      size.y = size.x;

   return size;
}

uint
libagx_texture_samples(constant struct agx_texture_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   /* As above */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* We may assume the input is multisampled, so just check the samples */
   return (d.samples == AGX_SAMPLE_COUNT_2) ? 2 : 4;
}

uint
libagx_texture_levels(constant struct agx_texture_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;
   else
      return (d.last_level - d.first_level) + 1;
}

/*
 * Fix robustness behaviour of txf with out-of-bounds LOD. The hardware
 * returns the correct out-of-bounds colour for out-of-bounds coordinates,
 * just not LODs. So translate out-of-bounds LOD into an out-of-bounds
 * coordinate to get correct behaviour in 1 instruction.
 *
 * Returns the fixed X-coordinate.
 *
 * TODO: This looks like it might be an erratum workaround on G13 (Apple does
 * it), maybe check if G15 is affected.
 */
uint
libagx_lower_txf_robustness(constant struct agx_texture_packed *ptr,
                            bool check_lod, ushort lod, bool check_min_lod,
                            ushort min_lod, bool check_layer, uint layer,
                            uint x)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   bool valid = true;

   if (check_lod)
      valid &= lod <= (d.last_level - d.first_level);

   if (check_min_lod)
      valid &= lod >= min_lod;

   if (check_layer) {
      bool linear = (d.layout == AGX_LAYOUT_LINEAR);
      valid &= layer < (linear ? d.depth_linear : d.depth);
   }

   /* The maximum tail offset is 0xF so by returning 0xFFF0 for out-of-bounds we
    * stay under 0xFFFF and keep robustness after offsetting.
    */
   return valid ? x : 0xFFF0;
}

uint32_t
libagx_twiddle_coordinates(ushort2 coord, uint16_t tile_w_px,
                           uint16_t tile_h_px, uint32_t aligned_width_px)
{
   /* Modulo by the tile width/height to get the offsets within the tile */
   ushort2 tile_mask_vec = (ushort2)(tile_w_px - 1, tile_h_px - 1);
   uint32_t tile_mask = upsample(tile_mask_vec.y, tile_mask_vec.x);
   uint32_t coord_xy = upsample(coord.y, coord.x);
   ushort2 offs_px = as_ushort2(coord_xy & tile_mask);
   uint32_t offset_within_tile_px = nir_interleave(offs_px.x, offs_px.y);

   /* Get the coordinates of the corner of the tile */
   ushort2 tile_px = as_ushort2(coord_xy & ~tile_mask);

   /* tile row start (px) =
    *   (y // tile height) * (# of tiles/row) * (# of pix/tile) =
    *   align_down(y, tile height) / tile height * width_tl *tile width *
    *        tile height =
    *   align_down(y, tile height) * width_tl * tile width
    */
   uint32_t tile_row_start_px = tile_px.y * aligned_width_px;

   /* tile column start (px) =
    *   (x // tile width) * (# of pix/tile) =
    *   align_down(x, tile width) / tile width * tile width * tile height =
    *   align_down(x, tile width) * tile height
    */
   uint32_t tile_col_start_px = tile_px.x * tile_h_px;

   /* Get the total offset */
   return tile_row_start_px + tile_col_start_px + offset_within_tile_px;
}

static uint32_t
libagx_spread_morton_16_gen9(uint16_t input)
{
   uint32_t x = input;
   x = (x | (x << 8)) & 0x00ff00ffu;
   x = (x | (x << 4)) & 0x0f0f0f0fu;
   x = (x | (x << 2)) & 0x33333333u;
   x = (x | (x << 1)) & 0x55555555u;
   return x;
}

/* AGX2 has a dedicated INTL instruction.  No INTL exists in the complete
 * G17P compiler corpus; Apple's texture-atomic shaders instead build their
 * Morton index from integer bit operations.  Use the canonical four-stage
 * SWAR expansion on Gen9, which keeps this architectural lowering out of the
 * AGX2 path and avoids inventing an AGX3 encoding. */
static uint32_t
libagx_twiddle_coordinates_gen9(ushort2 coord, uint16_t tile_w_px,
                                uint16_t tile_h_px,
                                uint32_t aligned_width_px)
{
   const ushort2 tile_mask_vec =
      (ushort2)(tile_w_px - 1, tile_h_px - 1);
   const uint32_t tile_mask = upsample(tile_mask_vec.y, tile_mask_vec.x);
   const uint32_t coord_xy = upsample(coord.y, coord.x);
   const ushort2 offs_px = as_ushort2(coord_xy & tile_mask);
   const uint32_t offset_within_tile_px =
      libagx_spread_morton_16_gen9(offs_px.x) |
      (libagx_spread_morton_16_gen9(offs_px.y) << 1);
   const ushort2 tile_px = as_ushort2(coord_xy & ~tile_mask);

   return tile_px.y * aligned_width_px +
          tile_px.x * tile_h_px + offset_within_tile_px;
}

uint64_t
libagx_image_texel_address(constant const struct agx_pbe_packed *ptr,
                           uint4 coord, uint sample_idx,
                           uint bytes_per_sample_B, bool is_1d, bool is_msaa,
                           bool is_layered, bool return_index)
{
   agx_unpack(NULL, ptr, PBE, d);

   /* We do not allow atomics on linear 2D or linear 2D arrays, as there are no
    * known use cases. So we're twiddled in this path, unless we're handling a
    * 1D image which will be always linear, even if it uses a twiddled layout
    * degrading to linear-equivalent 1x1 tiles. (1D uses this path, not the
    * buffer path, for 1D arrays.)
    */
   uint total_px;
   if (is_1d) {
      total_px = coord.x;
   } else {
      uint aligned_width_px;
      if (is_msaa) {
         aligned_width_px = d.aligned_width_msaa_sw;
      } else {
         uint width_px = max(d.width >> d.level, 1u);
         aligned_width_px = align(width_px, d.tile_width_sw);
      }

      total_px =
         libagx_twiddle_coordinates(convert_ushort2(coord.xy), d.tile_width_sw,
                                    d.tile_height_sw, aligned_width_px);
   }

   uint samples_log2 = is_msaa ? d.sample_count_log2_sw : 0;

   if (is_layered) {
      total_px += coord[is_1d ? 1 : 2] *
                  ((d.layer_stride_sw / bytes_per_sample_B) >> samples_log2);
   }

   uint total_sa = (total_px << samples_log2) + sample_idx;

   if (return_index)
      return total_sa;
   else
      return (d.buffer + (is_msaa ? 0 : d.level_offset_sw)) +
             (uint64_t)total_sa * bytes_per_sample_B;
}

/* Software view metadata, independent of the native texture/PBE packet:
 * base: address lo/hi, linear row pitch B (zero if tiled), tiled stride el;
 * layout: tile width/height el, layer stride B lo/hi. Tile coordinates count
 * pixels; MS samples are interleaved within each pixel. Runtime validation
 * restricts MS to tiled raw32-lane storage. The caller applied image robustness. */
uint64_t
libagx_bound_image_texel_address(uint4 base, uint4 layout, uint4 coord,
                                uint samples, uint sample_idx,
                                uint bytes_per_texel, bool is_1d,
                                bool is_layered)
{
   const uint64_t address = (uint64_t)base.x | ((uint64_t)base.y << 32);
   const uint64_t layer_stride =
      (uint64_t)layout.z | ((uint64_t)layout.w << 32);
   uint64_t offset;

   if (is_1d) {
      offset = (uint64_t)coord.x * bytes_per_texel;
   } else if (base.z) {
      offset = (uint64_t)coord.y * base.z +
               (uint64_t)coord.x * bytes_per_texel;
   } else {
      const uint64_t pixel = libagx_twiddle_coordinates_gen9(
         convert_ushort2(coord.xy), layout.x, layout.y, base.w);
      offset = (pixel * samples + sample_idx) * bytes_per_texel;
   }

   if (is_layered)
      offset += (uint64_t)coord[is_1d ? 1 : 2] * layer_stride;

   return address + offset;
}

uint64_t
libagx_image_texel_address_gen9(
   constant const struct agx_pbe_gen9_packed *ptr,
   constant const uint64_t *sw_ptr, uint4 coord, uint sample_idx,
   uint bytes_per_sample_B, bool is_1d, bool is_msaa, bool is_layered,
   bool return_index, bool gen10)
{
   /* Decode only the native fields used for address arithmetic.  Calling the
    * generated unpacker here would also inline its diagnostic mask checks;
    * those retain a printf-buffer intrinsic in release NIR even with a NULL
    * FILE and are intentionally rejected by the AGX3 backend. */
   const uint64_t native01 =
      (uint64_t)ptr->opaque[0] | ((uint64_t)ptr->opaque[1] << 32);
   const uint64_t native23 =
      (uint64_t)ptr->opaque[2] | ((uint64_t)ptr->opaque[3] << 32);
   const uint width_mask = gen10 ? 0x7fffu : 0x3fffu;
   const uint width_px_lod0 = (uint)((native01 >> 24) & width_mask) + 1;
   /* A linear PBE shortens the address to 40 bits and stores its row pitch
    * minus 16 bytes at bit 40, see agx_pbe_agx3_2d_pack(). */
   const bool linear = ((native01 >> 4) & 0x3u) == AGX_LAYOUT_LINEAR;
   const uint64_t buffer =
      (native23 & (linear ? 0xffffffffffUL : 0xfffffffffffUL)) << 4;
   const uint linear_stride_B = (uint)((native23 >> 40) & 0x1fffffu) + 16;
   const uint level = (ptr->opaque[5] >> 12) & 0xfu;
   const uint64_t sw = *sw_ptr;
   const uint level_offset_B = (uint)(sw & 0x7ffffffu) << 7;
   const uint tile_width_px = 1u << ((sw >> 27) & 0xfu);
   const uint tile_height_px = 1u << ((sw >> 31) & 0xfu);
   const uint layer_stride_B = (uint)((sw >> 35) & 0x7ffffffu) << 7;

   uint total_px;
   if (is_1d) {
      total_px = coord.x;
   } else if (linear) {
      total_px = coord.y * (linear_stride_B / bytes_per_sample_B) + coord.x;
   } else {
      const uint width_px = max(width_px_lod0 >> level, 1u);
      /* libcl's generic align() asserts dynamically that its alignment is a
       * power of two.  This field is written as a four-bit logarithm, so that
       * property is already structural; spell out the integer operation to
       * avoid retaining the assertion's printf path in backend NIR. */
      const uint aligned_width_px =
         (width_px + tile_width_px - 1u) & ~(tile_width_px - 1u);
      total_px = libagx_twiddle_coordinates_gen9(
         convert_ushort2(coord.xy), tile_width_px, tile_height_px,
         aligned_width_px);
   }

   /* Native multisample atomics need a separately recovered sample-count
    * sideband.  Keep the path explicit until that field is grounded. */
   const uint samples_log2 = is_msaa ? (uint)(sw >> 62) : 0;

   if (is_layered) {
      total_px += coord[is_1d ? 1 : 2] *
                  ((layer_stride_B / bytes_per_sample_B) >> samples_log2);
   }

   const uint total_sa = (total_px << samples_log2) + sample_idx;
   if (return_index)
      return total_sa;

   return (buffer + (is_msaa ? 0 : level_offset_B)) +
          (uint64_t)total_sa * bytes_per_sample_B;
}

/* eMRT writes MS images through a linear PBE view. Its native descriptor
 * therefore cannot describe the source image's tile geometry. The private
 * eMRT sideband carries the original geometry and aligned pixel width. */
uint
libagx_emrt_ms_index_gen9(constant const uint64_t *sw_ptr,
                         constant const uint *width_ptr, uint4 coord,
                         uint sample, uint bytes_per_sample)
{
   const uint64_t sw = *sw_ptr;
   const uint tile_width = 1u << ((sw >> 27) & 0xfu);
   const uint tile_height = 1u << ((sw >> 31) & 0xfu);
   const uint layer_stride = (uint)((sw >> 35) & 0x7ffffffu) << 7;
   const uint samples_log2 = (uint)(sw >> 62);
   const uint pixel = libagx_twiddle_coordinates_gen9(
      convert_ushort2(coord.xy), tile_width, tile_height, *width_ptr);
   return (pixel << samples_log2) + sample +
          coord.z * (layer_stride / bytes_per_sample);
}

uint64_t
libagx_buffer_texel_address(constant const struct agx_pbe_packed *ptr,
                            uint4 coord, uint bytes_per_pixel_B)
{
   agx_unpack(NULL, ptr, PBE, d);

   uint32_t x_el = d.buffer_offset_sw + coord.x;
   return d.buffer + (uint64_t)(x_el * bytes_per_pixel_B);
}

/* Buffer texture lowerings */
bool
libagx_texture_is_rgb32(constant struct agx_texture_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE, d);
   return d.channels == AGX_CHANNELS_R32G32B32_EMULATED;
}

uint4
libagx_texture_load_rgb32(constant struct agx_texture_packed *ptr, uint coord,
                          bool is_float)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   /* This is carefully written to let us do the * 3 + offset with a 32-bit
    * operation while still using the free 64-bit add-extend-shift for the rest.
    * For emulated RGB32 descriptors, buffer_offset_sw stores the sub-16-byte
    * base-address tail in 4-byte components (0..3).
    */
   uint64_t addr =
      d.address + ((uint64_t)(coord * 3 + d.buffer_offset_sw)) * 4;
   typedef uint3 __attribute__((aligned(4))) uint3_align4;
   constant uint3_align4 *data = (constant uint3_align4 *)addr;

   return (uint4)(*data, is_float ? as_uint(1.0f) : 1);
}

uint
libagx_buffer_texture_offset(constant struct agx_texture_packed *ptr, uint x)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   return x + d.buffer_offset_sw;
}

/*
 * Whether texel x of a texel buffer lies in an unbound sparse page, from the
 * device-wide map of non-resident pages: a 16-byte header, then one bit per
 * 16 KiB page of [heap_base, heap_base + heap_size), set while the page is
 * not resident.  The descriptor is the plain linear 2D texture hk builds over
 * the buffer's read-only alias, AGX_TEXTURE_BUFFER_WIDTH texels wide, so the
 * texel size is its row stride over its width.  Out-of-bounds texels, RGB32
 * views and buffers outside the heap count as resident.
 */
uint
libagx_buffer_texture_nonresident(constant struct agx_texture_packed *ptr,
                                  uint x, constant uint *map,
                                  uint64_t heap_base, uint64_t heap_size,
                                  uint64_t ro_offset)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   if (map == NULL || x >= d.buffer_size_sw ||
       d.channels == AGX_CHANNELS_R32G32B32_EMULATED)
      return 0;

   uint texel_B = (d.stride + 16) / d.width;
   uint64_t addr = (d.address & ~ro_offset) +
                   (uint64_t)(x + d.buffer_offset_sw) * texel_B;
   uint64_t offset = addr - heap_base;
   if (offset >= heap_size)
      return 0;

   uint64_t page = offset >> 14;
   uint word = map[4 + (page >> 5)];

   /* A variable mask rather than (word >> page) & 1, which would fold into a
    * bitfield extract whose offset is not masked to 0..31. */
   return (word & (1u << (uint)(page & 31))) != 0;
}

uint
libagx_buffer_image_offset(constant struct agx_pbe_packed *ptr, uint x)
{
   agx_unpack(NULL, ptr, PBE, d);

   return x + d.buffer_offset_sw;
}

/* Gen9 queries consume the same descriptor bytes as the native texture
 * hardware. Address, sample-count and mip-level fields differ from AGX2;
 * arrays retain their native tiled/linear depth fields. */
uint3
libagx_txs_gen9(constant struct agx_texture_gen9_packed *ptr, uint16_t lod,
           unsigned nr_comps, bool is_1d, bool is_2d,
           bool is_cube, bool is_array)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN9, d);

   /* From the Vulkan spec:
    *
    *    OpImageQuery*...  return 0 if the bound descriptor is a null descriptor
    */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* Load standard dimensions */
   uint3 size = (uint3)(d.width, d.height, d.depth);
   lod += d.first_level;

   /* Linear 2D arrays are special.
    *
    * TODO: Optimize this, since linear 2D arrays aren't needed for APIs and
    * this just gets used internally for blits.
    */
   if (is_2d && is_array && d.layout == AGX_LAYOUT_LINEAR)
      size.z = d.depth_linear;

   /* 1D Arrays have their second component as the layer count */
   if (is_1d && is_array)
      size.y = size.z;

   /* Adjust for LOD, do not adjust array size */
   size.x = max(size.x >> lod, 1u);

   if (nr_comps - (uint)is_array >= 2)
      size.y = max(size.y >> lod, 1u);

   if (nr_comps - (uint)is_array >= 3)
      size.z = max(size.z >> lod, 1u);

   /* Cube maps have equal width and height, we save some instructions by only
    * reading one. Dead code elimination will remove the redundant instructions.
    */
   if (is_cube)
      size.y = size.x;

   return size;
}

uint
libagx_texture_samples_gen9(constant struct agx_texture_gen9_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN9, d);

   /* As above */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* We may assume the input is multisampled, so just check the samples */
   return 2u << d.samples;
}

/* Gen10 widens the dimensions and moves the sample field. Decode the
 * matching descriptor while retaining the shared view/LOD query semantics. */
uint3
libagx_txs_gen10(constant struct agx_texture_gen10_packed *ptr, uint16_t lod,
           unsigned nr_comps, bool is_1d, bool is_2d,
           bool is_cube, bool is_array)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN10, d);

   /* From the Vulkan spec:
    *
    *    OpImageQuery*...  return 0 if the bound descriptor is a null descriptor
    */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* Load standard dimensions */
   uint3 size = (uint3)(d.width, d.height, d.depth);
   lod += d.first_level;

   /* Linear 2D arrays are special.
    *
    * TODO: Optimize this, since linear 2D arrays aren't needed for APIs and
    * this just gets used internally for blits.
    */
   if (is_2d && is_array && d.layout == AGX_LAYOUT_LINEAR)
      size.z = d.depth_linear;

   /* 1D Arrays have their second component as the layer count */
   if (is_1d && is_array)
      size.y = size.z;

   /* Adjust for LOD, do not adjust array size */
   size.x = max(size.x >> lod, 1u);

   if (nr_comps - (uint)is_array >= 2)
      size.y = max(size.y >> lod, 1u);

   if (nr_comps - (uint)is_array >= 3)
      size.z = max(size.z >> lod, 1u);

   /* Cube maps have equal width and height, we save some instructions by only
    * reading one. Dead code elimination will remove the redundant instructions.
    */
   if (is_cube)
      size.y = size.x;

   return size;
}

uint
libagx_texture_samples_gen10(constant struct agx_texture_gen10_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN10, d);

   /* As above */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* Single-sampled views use an ordinary 2D dimension; the sample field
    * encodes 2/4/8 only for multisampled dimensions. */
   const bool multisampled =
      d.dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      d.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   return multisampled ? (2u << d.samples) : 1u;
}

uint
libagx_texture_levels_gen10(constant struct agx_texture_gen10_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN10, d);

   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;
   else
      return (d.last_level - d.first_level) + 1;
}

/* robustImageAccess2 reads outside a view return (0, 0, 0, 1) for a format
 * without alpha and zero otherwise. A view's descriptor records a missing alpha
 * as a constant one alpha swizzle; the null texture swizzles every channel to
 * zero. */
bool
libagx_image_oob_alpha_one_gen9(constant struct agx_texture_gen9_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN9, d);
   return d.swizzle_a == AGX_CHANNEL_1;
}

uint
libagx_texture_levels_gen9(constant struct agx_texture_gen9_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN9, d);

   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;
   else
      return (d.last_level - d.first_level) + 1;
}

/*
 * The same workaround against a Gen10 descriptor. Without it a G17G texel
 * fetch unpacked its descriptor with the AGX2 field positions, read a
 * nonsense level range, and translated every fetch into the out-of-bounds
 * coordinate -- which is how the depth blitter, whose fragment shader is one
 * txf, wrote nothing at all.
 */
uint
libagx_lower_txf_robustness_gen10(constant struct agx_texture_gen10_packed *ptr,
                            bool check_lod, ushort lod, bool check_min_lod,
                            ushort min_lod, bool check_layer, uint layer,
                            uint x)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN10, d);

   bool valid = true;

   if (check_lod)
      valid &= lod <= (d.last_level - d.first_level);

   if (check_min_lod)
      valid &= lod >= min_lod;

   if (check_layer) {
      bool linear = (d.layout == AGX_LAYOUT_LINEAR);
      valid &= layer < (linear ? d.depth_linear : d.depth);
   }

   /* The maximum tail offset is 0xF so by returning 0xFFF0 for out-of-bounds we
    * stay under 0xFFFF and keep robustness after offsetting.
    */
   return valid ? x : 0xFFF0;
}

uint
libagx_lower_txf_robustness_gen9(constant struct agx_texture_gen9_packed *ptr,
                            bool check_lod, ushort lod, bool check_min_lod,
                            ushort min_lod, bool check_layer, uint layer,
                            uint x)
{
   agx_unpack(NULL, ptr, TEXTURE_GEN9, d);

   bool valid = true;

   if (check_lod)
      valid &= lod <= (d.last_level - d.first_level);

   if (check_min_lod)
      valid &= lod >= min_lod;

   if (check_layer) {
      bool linear = (d.layout == AGX_LAYOUT_LINEAR);
      valid &= layer < (linear ? d.depth_linear : d.depth);
   }

   /* The maximum tail offset is 0xF so by returning 0xFFF0 for out-of-bounds we
    * stay under 0xFFFF and keep robustness after offsetting.
    */
   return valid ? x : 0xFFF0;
}
