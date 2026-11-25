/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "layout.h"
#include "util/format/u_format.h"
#include "util/u_math.h"

static void
ail_initialize_linear(struct ail_layout *layout)
{
   /* Select the optimal stride if none is forced */
   if (layout->linear_stride_B == 0) {
      uint32_t minimum_stride_B =
         util_format_get_stride(layout->format, layout->width_px);

      layout->linear_stride_B = ALIGN_POT(minimum_stride_B, AIL_CACHELINE);
   }

   assert((layout->linear_stride_B % 16) == 0 && "Strides must be aligned");

   /* Layer stride must be cache line aligned to pack linear 2D arrays */
   layout->layer_stride_B = align64(
      (uint64_t)layout->linear_stride_B * layout->height_px, AIL_CACHELINE);

   layout->size_B =
      layout->level_offsets_B[0] + (layout->layer_stride_B * layout->depth_px);
}

/*
 * Calculate the number of bytes in a block. This must take both block
 * dimensions and multisampling into account.
 */
static uint32_t
ail_get_block_size_B(struct ail_layout *layout)
{
   ASSERTED const struct util_format_description *desc =
      util_format_description(layout->format);

   assert(((layout->sample_count_sa == 1) ||
           (desc->block.width == 1 && desc->block.height == 1)) &&
          "multisampling and block-compression are mutually-exclusive");

   return util_format_get_blocksize(layout->format) * layout->sample_count_sa;
}

static void
ail_initialize_gpu_tiled(struct ail_layout *layout)
{
   unsigned offset_B = 0;
   unsigned blocksize_B = ail_get_block_size_B(layout);
   unsigned w_el = util_format_get_nblocksx(layout->format, layout->width_px);
   unsigned h_el = util_format_get_nblocksy(layout->format, layout->height_px);
   unsigned bw_px = util_format_get_blockwidth(layout->format);
   unsigned bh_px = util_format_get_blockheight(layout->format);
   bool compressed = util_format_is_compressed(layout->format);

   /* Calculate the tile size used for the large miptree, and the dimensions of
    * level 0 given that tile size.
    */
   struct ail_tile tilesize_el = ail_get_max_tile_size(blocksize_B);
   unsigned stx_tiles = DIV_ROUND_UP(w_el, tilesize_el.width_el);
   unsigned sty_tiles = DIV_ROUND_UP(h_el, tilesize_el.height_el);
   unsigned sarea_tiles = stx_tiles * sty_tiles;

   /* Calculate which level the small power-of-two miptree begins at. The
    * power-of-two miptree is used when either the width or the height is
    * smaller than a single large tile.
    */
   unsigned pot_level = 0;
   unsigned pot_w_px = bw_px * w_el;
   unsigned pot_h_px = bh_px * h_el;
   do {
      unsigned pot_w_el = util_format_get_nblocksx(layout->format, pot_w_px);
      unsigned pot_h_el = util_format_get_nblocksy(layout->format, pot_h_px);
      if (pot_w_el < tilesize_el.width_el || pot_h_el < tilesize_el.height_el)
         break;
      pot_w_px = u_minify(pot_w_px, 1);
      pot_h_px = u_minify(pot_h_px, 1);
      pot_level++;
   } while (1);

   /* First allocate the large miptree. All tiles in the large miptree are of
    * size tilesize_el and have their dimensions given by stx/sty/sarea.
    */
   for (unsigned l = 0; l < MIN2(pot_level, layout->levels); ++l) {
      unsigned tiles = (sarea_tiles >> (2 * l));

      bool pad_left = (stx_tiles & BITFIELD_MASK(l));
      bool pad_bottom = (sty_tiles & BITFIELD_MASK(l));
      bool pad_corner = pad_left && pad_bottom;

      if (pad_left)
         tiles += (sty_tiles >> l);

      if (pad_bottom)
         tiles += (stx_tiles >> l);

      if (pad_corner)
         tiles += 1;

      unsigned size_el = tiles * tilesize_el.width_el * tilesize_el.height_el;
      layout->level_offsets_B[l] = offset_B;
      offset_B = ALIGN_POT(offset_B + (blocksize_B * size_el), AIL_CACHELINE);

      layout->stride_el[l] = util_format_get_nblocksx(
         layout->format, u_minify(layout->width_px, l));

      /* Compressed textures pad the stride in this case */
      if (compressed && pad_left)
         layout->stride_el[l]++;

      layout->tilesize_el[l] = tilesize_el;
   }

   /* Then begin the POT miptree. Note that we round up to a power-of-two
    * outside the loop. That ensures correct handling of cases like 33x33
    * images, where the round-down error of right-shifting could cause incorrect
    * tile size calculations.
    */
   unsigned potw_el, poth_el;
   if (compressed) {
      /* Compressed formats round then minify instead of minifying then rounding
       */
      potw_el = u_minify(util_next_power_of_two(w_el), pot_level);
      poth_el = u_minify(util_next_power_of_two(h_el), pot_level);
   } else {
      potw_el = util_next_power_of_two(u_minify(w_el, pot_level));
      poth_el = util_next_power_of_two(u_minify(h_el, pot_level));
   }

   /* Finally we allocate the POT miptree, starting at level pot_level. Each
    * level uses the largest power-of-two tile that fits the level.
    */
   for (unsigned l = pot_level; l < layout->levels; ++l) {
      unsigned size_el = potw_el * poth_el;
      layout->level_offsets_B[l] = offset_B;
      offset_B = ALIGN_POT(offset_B + (blocksize_B * size_el), AIL_CACHELINE);

      /* On M5 (AIL_LAYOUT_POLICY_M5), hardware places compressed mip levels
       * on a 2 KiB boundary (observed after a 32x32 1-byte level, which is
       * 1 KiB of texels; 2-16 byte compressed levels are already >= 2 KiB
       * through 32x32 and levels below 16x16 are uncompressed). */
      if (layout->policy == AIL_LAYOUT_POLICY_M5 && !layout->mipmapped_z &&
          layout->compressed && l + 1 < layout->levels &&
          ail_is_level_logically_compressed(layout, l + 1))
         offset_B = ALIGN_POT(offset_B, AIL_M5_COMPRESSED_LEVEL_ALIGN_B);

      /* The tilesize is based on the true mipmap level size, not the POT
       * rounded size, except for compressed textures */
      unsigned tilesize_el;
      if (compressed)
         tilesize_el = util_next_power_of_two(MIN2(potw_el, poth_el));
      else
         tilesize_el = util_next_power_of_two(u_minify(MIN2(w_el, h_el), l));
      layout->tilesize_el[l] = (struct ail_tile){tilesize_el, tilesize_el};
      layout->stride_el[l] = util_format_get_nblocksx(
         layout->format, u_minify(layout->width_px, l));

      potw_el = u_minify(potw_el, 1);
      poth_el = u_minify(poth_el, 1);
   }

   /* Add the end offset so we can easily recover the size of a level */
   assert(layout->levels < ARRAY_SIZE(layout->level_offsets_B));
   layout->level_offsets_B[layout->levels] = offset_B;

   /* Determine the start of the miptail. From that level on, we can no longer
    * precisely bind at page granularity.
    */
   layout->mip_tail_first_lod = MIN2(pot_level, layout->levels);

   /* Determine the stride of the miptail. Sparse arrayed images inherently
    * require page-aligned layers to be able to bind individual layers.
    */
   unsigned tail_offset_B = layout->level_offsets_B[layout->mip_tail_first_lod];
   layout->mip_tail_stride = align(offset_B - tail_offset_B, AIL_PAGESIZE);

   if (layout->policy == AIL_LAYOUT_POLICY_M5 && !layout->mipmapped_z) {
      /* Live M5 allocation corpus: single layers are never padded; small
       * arrays use a 2 KiB granule only when compression metadata exists;
       * arrays larger than one GPU page keep the 16 KiB granule. Shader-write
       * usage by itself does not force page alignment on this family, except
       * for mipmapped arrays (below). */
      layout->layer_stride_B =
         ail_m5_layer_stride_B(offset_B, layout->depth_px, layout->compressed);
      layout->page_aligned_layers =
         layout->depth_px > 1 && offset_B >= AIL_PAGESIZE;
      /* ZLS layer strides use page units even when color/texture arrays can
       * use compact strides. A sub-page depth/stencil layer cannot be
       * represented by the render submission's ZLS descriptor.
       *
       * G18G keeps compact depth/stencil layers: its ZLS array stride is
       * encoded in 128-byte cachelines below a page (agx_zls_layer_stride),
       * except for the writeable mipmapped arrays below, whose page-sized
       * layers it can encode too. */
      if (!layout->g18g && layout->depth_px > 1 && layout->renderable &&
          util_format_is_depth_or_stencil(layout->format)) {
         layout->page_aligned_layers = true;
         layout->layer_stride_B = ALIGN_POT(offset_B, AIL_PAGESIZE);
      }
      /* The exception to the above: mipmapped arrays the GPU can write. A
       * store to a layer of an array is described as a 3D store on the chips
       * that encode layered stores in z (G17G and G17S in Honeykrisp and
       * Gallium, G18G in Gallium; hk_image_view.c, agx_state.c), with the
       * depth scaled by the level. The 3D slice stride follows a mip chain as
       * deep as the described depth. That chain can have more levels than the
       * array's, and a compact stride cannot hold the difference; a
       * page-aligned one can.
       *
       * Both drivers mark every uncompressed image writeable, and
       * Honeykrisp's buffer-to-image copies store through such a layered
       * view, so this pads every uncompressed mipmapped array under this
       * policy, including on G17C, which keeps the array form. Do not narrow
       * it to shader-storage usage: uploads to small mipmapped arrays and
       * cube maps would land in the wrong layer again. */
      if (layout->writeable_image && layout->depth_px > 1 &&
          layout->levels > 1) {
         layout->page_aligned_layers = true;
         layout->layer_stride_B = ALIGN_POT(offset_B, AIL_PAGESIZE);
      }
   } else {
      /* Align layer size if we have mipmaps and one miptree is larger than one
       * page */
      layout->page_aligned_layers =
         layout->levels != 1 && offset_B > AIL_PAGESIZE;

      /* Single-layer images are not padded unless they are Z/S */
      bool zs = util_format_is_depth_or_stencil(layout->format);
      if (layout->depth_px == 1 && !zs)
         layout->page_aligned_layers = false;

      /* For writable images, we require page-aligned layers. This appears to
       * be required for PBE stores, including block stores for colour
       * rendering. Likewise, ZLS layer strides are specified in pages. */
      layout->page_aligned_layers |= layout->writeable_image;
      layout->page_aligned_layers |= layout->renderable && layout->depth_px > 1;

      if (layout->page_aligned_layers)
         layout->layer_stride_B = ALIGN_POT(offset_B, AIL_PAGESIZE);
      else
         layout->layer_stride_B = offset_B;
   }

   /* Each sparse layer has its own page-map range. Include the partial
    * final page so mip-tail entries cannot be mistaken for the next layer. */
   if (layout->sparse_residency) {
      layout->page_aligned_layers = true;
      layout->layer_stride_B = ALIGN_POT(layout->layer_stride_B, AIL_PAGESIZE);
   }

   layout->size_B = (uint64_t)layout->layer_stride_B * layout->depth_px;
}

static void
ail_initialize_twiddled(struct ail_layout *layout)
{
   unsigned offset_B = 0;
   unsigned blocksize_B = ail_get_block_size_B(layout);
   unsigned w_el = util_format_get_nblocksx(layout->format, layout->width_px);
   unsigned h_el = util_format_get_nblocksy(layout->format, layout->height_px);
   bool compressed = util_format_is_compressed(layout->format);

   for (unsigned l = 0; l < layout->levels; ++l) {
      unsigned alloc_w_el = u_minify(util_next_power_of_two(w_el), l);
      unsigned alloc_h_el = u_minify(util_next_power_of_two(h_el), l);
      unsigned logical_w_el = util_next_power_of_two(u_minify(w_el, l));
      unsigned logical_h_el = util_next_power_of_two(u_minify(h_el, l));

      /* AGX: Even the exceptions have exceptions */
      if (compressed) {
         logical_w_el = alloc_w_el;
         logical_h_el = alloc_h_el;
      }

      unsigned size_el = alloc_w_el * alloc_h_el;
      layout->level_offsets_B[l] = offset_B;
      offset_B = ALIGN_POT(offset_B + (blocksize_B * size_el), AIL_CACHELINE);

      unsigned minor_el = MIN2(logical_w_el, logical_h_el);
      layout->stride_el[l] = logical_w_el;
      layout->tilesize_el[l] = (struct ail_tile){minor_el, minor_el};
   }

   /* Add the end offset so we can easily recover the size of a level */
   assert(layout->levels < ARRAY_SIZE(layout->level_offsets_B));
   layout->level_offsets_B[layout->levels] = offset_B;

   /* Determine the start of the miptail. From that level on, we can no longer
    * precisely bind at page granularity.
    */
   /* XXX TODO */
   layout->mip_tail_first_lod = 0;

   /* Determine the stride of the miptail. Sparse arrayed images inherently
    * require page-aligned layers to be able to bind individual layers.
    */
   unsigned tail_offset_B = layout->level_offsets_B[layout->mip_tail_first_lod];
   layout->mip_tail_stride = align(offset_B - tail_offset_B, AIL_PAGESIZE);

   layout->page_aligned_layers = true;
   layout->layer_stride_B = ALIGN_POT(offset_B, AIL_PAGESIZE);
   layout->size_B = (uint64_t)layout->layer_stride_B * layout->depth_px;
}

/*
 * Interchange: macroblock (tx, ty) occupies 256 elements at
 * (ty * tiles_x + tx) * 256 * blocksize.  Reporting a 16x16 tile and a stride
 * of tiles_x * 16 elements makes every consumer that addresses tiles as
 * "row-major tiles, twiddled inside" (libagx_fast_clear's
 * libagx_twiddle_coordinates) land on the right macroblock.
 */
static void
ail_initialize_interchange(struct ail_layout *layout)
{
   assert(layout->levels == 1 && layout->depth_px == 1 &&
          layout->sample_count_sa == 1 && !layout->mipmapped_z &&
          "Interchange layouts are single-level, single-layer, single-sample");
   assert(util_format_get_blockwidth(layout->format) == 1 &&
          util_format_get_blockheight(layout->format) == 1 &&
          util_format_get_blocksize(layout->format) == 4 &&
          "Interchange layouts are 32-bit uncompressed formats");
   assert(layout->compressed && "Interchange layouts are always compressed");

   const uint32_t tiles_x = DIV_ROUND_UP(layout->width_px, 16);
   const uint32_t tiles_y = DIV_ROUND_UP(layout->height_px, 16);

   layout->level_offsets_B[0] = 0;
   layout->level_offsets_B[1] = (uint64_t)tiles_y * ail_get_interchange_stride_B(layout);
   layout->stride_el[0] = tiles_x * 16;
   layout->tilesize_el[0] = (struct ail_tile){16, 16};
   layout->mip_tail_first_lod = 1;
   layout->mip_tail_stride = 0;
   layout->page_aligned_layers = false;
   layout->layer_stride_B = layout->level_offsets_B[1];
   layout->size_B = layout->layer_stride_B;
}

static void
ail_initialize_compression(struct ail_layout *layout)
{
   assert(!util_format_is_compressed(layout->format) &&
          "Compressed pixel formats not supported");
   assert(util_format_get_blockwidth(layout->format) == 1);
   assert(util_format_get_blockheight(layout->format) == 1);

   unsigned width_sa =
      ail_effective_width_sa(layout->width_px, layout->sample_count_sa);
   unsigned height_sa =
      ail_effective_height_sa(layout->height_px, layout->sample_count_sa);

   assert(width_sa >= 16 && "Small textures are never compressed");
   assert(height_sa >= 16 && "Small textures are never compressed");

   layout->metadata_offset_B = layout->size_B;

   const struct ail_tile metadata_tile = ail_metadata_tile(layout);

   width_sa = ALIGN_POT(width_sa, metadata_tile.width_el);
   height_sa = ALIGN_POT(height_sa, metadata_tile.height_el);

   unsigned compbuf_B = 0;

   for (unsigned l = 0;
        l < layout->levels && ail_is_level_allocated_compressed(layout, l);
        ++l) {

      layout->level_offsets_compressed_B[l] = compbuf_B;

      /* The metadata buffer contains 8 bytes per compression tile. M5 uses
       * 16x8 for 16 B/px and 16x16 otherwise; legacy layouts use 16x16.
       * Addressing is fully twiddled, so both axes are padded to powers-of-two.
       */
      unsigned w_tl = DIV_ROUND_UP(util_next_power_of_two(width_sa),
                                   metadata_tile.width_el);
      unsigned h_tl = DIV_ROUND_UP(util_next_power_of_two(height_sa),
                                   metadata_tile.height_el);
      unsigned B_per_tl_2 = 8;

      compbuf_B += ALIGN_POT(w_tl * h_tl * B_per_tl_2, AIL_CACHELINE);

      width_sa = DIV_ROUND_UP(width_sa, 2);
      height_sa = DIV_ROUND_UP(height_sa, 2);
   }

   layout->compression_layer_stride_B = compbuf_B;
   layout->size_B +=
      (uint64_t)(layout->compression_layer_stride_B * layout->depth_px);
}

static void
ail_initialize_sparse_table(struct ail_layout *layout)
{
   layout->sparse_folios_per_layer =
      DIV_ROUND_UP(layout->layer_stride_B, AIL_IMAGE_SIZE_PER_FOLIO_B);

   unsigned folios = layout->sparse_folios_per_layer * layout->depth_px;
   layout->sparse_table_size_B = folios * AIL_FOLIO_SIZE_B;
}

void
ail_make_miptree(struct ail_layout *layout)
{
   assert(layout->width_px >= 1 && "Invalid dimensions");
   assert(layout->height_px >= 1 && "Invalid dimensions");
   assert(layout->depth_px >= 1 && "Invalid dimensions");

   if (layout->tiling == AIL_TILING_LINEAR) {
      assert(layout->levels == 1 && "Invalid linear layout");
      assert(layout->sample_count_sa == 1 &&
             "Multisampled linear layouts not supported");
      /* Horizontal subsampling retains ordinary rows. get_stride already
       * measures the row in packed words, and height/layer strides are unchanged. */
      assert((util_format_get_blockwidth(layout->format) == 1 ||
              util_format_description(layout->format)->layout == UTIL_FORMAT_LAYOUT_SUBSAMPLED) &&
             "Strided linear block formats unsupported");
      assert(util_format_get_blockheight(layout->format) == 1 &&
             "Strided linear block formats unsupported");
   } else {
      assert(layout->linear_stride_B == 0 && "Invalid nonlinear layout");
      assert(layout->levels >= 1 && "Invalid dimensions");
      assert(layout->sample_count_sa >= 1 && "Invalid sample count");
      assert(layout->level_offsets_B[0] == 0 && "Invalid offset");
   }

   assert(!(layout->writeable_image && layout->compressed) &&
          "Writeable images must not be compressed");

   /* Hardware strides are based on the maximum number of levels, so always
    * allocate them all.
    */
   if (layout->levels > 1) {
      unsigned major_axis_px = MAX2(layout->width_px, layout->height_px);

      if (layout->mipmapped_z)
         major_axis_px = MAX2(major_axis_px, layout->depth_px);

      layout->levels = util_logbase2(major_axis_px) + 1;
   }

   assert(util_format_get_blockdepth(layout->format) == 1 &&
          "Deep formats unsupported");

   switch (layout->tiling) {
   case AIL_TILING_LINEAR:
      ail_initialize_linear(layout);
      break;
   case AIL_TILING_GPU:
      ail_initialize_gpu_tiled(layout);
      break;
   case AIL_TILING_TWIDDLED:
      ail_initialize_twiddled(layout);
      break;
   case AIL_TILING_INTERCHANGE:
      ail_initialize_interchange(layout);
      break;
   default:
      UNREACHABLE("Unsupported tiling");
   }

   if (layout->compressed) {
      ail_initialize_compression(layout);
   }

   ail_initialize_sparse_table(layout);

   layout->size_B = ALIGN_POT(layout->size_B, AIL_CACHELINE);
   assert(layout->size_B > 0 && "Invalid dimensions");
}
