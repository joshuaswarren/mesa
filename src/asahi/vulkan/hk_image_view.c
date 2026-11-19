/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_image_view.h"
#include "util/u_debug.h"
#include "util/half_float.h"
#include "util/format/u_format.h"
#include "vulkan/vulkan_core.h"

#include "agx_helpers.h"
#include "agx_bg_eot_key.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_frame_anatomy.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "layout.h"
#include "vk_format.h"
#include "vk_meta.h"

enum hk_desc_usage {
   HK_DESC_USAGE_SAMPLED,
   HK_DESC_USAGE_SAMPLED_FILTERABLE,
   HK_DESC_USAGE_STORAGE,
   HK_DESC_USAGE_INPUT,
   HK_DESC_USAGE_BG_EOT,
   HK_DESC_USAGE_LAYERED_BG_EOT,
   HK_DESC_USAGE_EMRT,
};

static bool
hk_image_view_type_is_array(VkImageViewType view_type)
{
   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
   case VK_IMAGE_VIEW_TYPE_3D:
   case VK_IMAGE_VIEW_TYPE_CUBE:
      return false;

   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return true;

   default:
      UNREACHABLE("Invalid image view type");
   }
}

static enum agx_texture_dimension
translate_image_view_type(VkImageViewType view_type, bool msaa, bool layered,
                          enum hk_desc_usage usage, bool cubes_to_2d)
{
   if (usage == HK_DESC_USAGE_EMRT || usage == HK_DESC_USAGE_INPUT ||
       (usage == HK_DESC_USAGE_LAYERED_BG_EOT && layered)) {
      return msaa ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D_ARRAY;
   }

   /* For background/EOT, we ignore the application-provided view type */
   if (usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT) {
      return msaa ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D;
   }

   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
      return msaa ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D;

   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      return msaa ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D_ARRAY;

   case VK_IMAGE_VIEW_TYPE_3D:
      assert(!msaa);
      return AGX_TEXTURE_DIMENSION_3D;

   case VK_IMAGE_VIEW_TYPE_CUBE:
      assert(!msaa);
      return cubes_to_2d ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                         : AGX_TEXTURE_DIMENSION_CUBE;

   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      assert(!msaa);
      return cubes_to_2d ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                         : AGX_TEXTURE_DIMENSION_CUBE_ARRAY;

   default:
      UNREACHABLE("Invalid image view type");
   }
}

static enum pipe_swizzle
vk_swizzle_to_pipe(VkComponentSwizzle swizzle)
{
   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_R:
      return PIPE_SWIZZLE_X;
   case VK_COMPONENT_SWIZZLE_G:
      return PIPE_SWIZZLE_Y;
   case VK_COMPONENT_SWIZZLE_B:
      return PIPE_SWIZZLE_Z;
   case VK_COMPONENT_SWIZZLE_A:
      return PIPE_SWIZZLE_W;
   case VK_COMPONENT_SWIZZLE_ONE:
      return PIPE_SWIZZLE_1;
   case VK_COMPONENT_SWIZZLE_ZERO:
      return PIPE_SWIZZLE_0;
   default:
      UNREACHABLE("Invalid component swizzle");
   }
}

static enum pipe_format
get_stencil_format(enum pipe_format format)
{
   /* Note: R8 is not a stencil format, but our copy code will internally cast
    * between R8 and S8. It's simplest to handle here.
    */
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_S8_UINT:
      return PIPE_FORMAT_S8_UINT;
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      return PIPE_FORMAT_X24S8_UINT;
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      return PIPE_FORMAT_S8X24_UINT;
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return PIPE_FORMAT_X32_S8X24_UINT;
   default:
      UNREACHABLE("Unsupported depth/stencil format");
   }
}

struct hk_3d {
   unsigned x, y, z;
};

/* A texel block wider or taller than one pixel means the API extent and the
 * layout's element grid differ, so a view that does not share the block has to
 * divide the extent down.  Block-compressed formats are the usual case, but
 * the single-plane 4:2:2 formats are blocked (2x1) without being compressed --
 * util_format_is_compressed() is false for them.  Testing the block itself
 * covers both; it is what the extent arithmetic below actually needs.
 */
static bool
format_is_blocked(enum pipe_format format)
{
   return util_format_get_blockwidth(format) > 1 ||
          util_format_get_blockheight(format) > 1 ||
          util_format_get_blockdepth(format) > 1;
}

static struct hk_3d
view_denominator(struct hk_image_view *view)
{
   enum pipe_format view_format = hk_format_to_pipe_format(view->vk.format);
   enum pipe_format img_format =
      hk_format_to_pipe_format(view->vk.image->format);

   if (format_is_blocked(view_format)) {
      /*
       * We can do an unblocked view of a blocked image but not the other
       * way around.
       */
      assert(format_is_blocked(img_format));
      assert(util_format_get_blockwidth(img_format) ==
             util_format_get_blockwidth(view_format));
      assert(util_format_get_blockheight(img_format) ==
             util_format_get_blockheight(view_format));
      assert(util_format_get_blockdepth(img_format) ==
             util_format_get_blockdepth(view_format));

      return (struct hk_3d){1, 1, 1};
   }

   if (!format_is_blocked(img_format)) {
      /* Both formats have 1x1 blocks */
      return (struct hk_3d){1, 1, 1};
   }

   /* Else, img is blocked but view is not */
   return (struct hk_3d){
      util_format_get_blockwidth(img_format),
      util_format_get_blockheight(img_format),
      util_format_get_blockdepth(img_format),
   };
}

static enum pipe_format
format_for_plane(struct hk_image_view *view, unsigned view_plane)
{
   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(view->vk.format);

   assert(ycbcr_info || view_plane == 0);
   VkFormat plane_format =
      ycbcr_info ? ycbcr_info->planes[view_plane].format : view->vk.format;

   if (view->vk.format == VK_FORMAT_D16_UNORM_S8_UINT)
      plane_format = hk_depth_stencil_plane_format(view->vk.format,
         view->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT);
   enum pipe_format p_format = hk_format_to_pipe_format(plane_format);
   if (view->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      p_format = get_stencil_format(p_format);

   return p_format;
}

static bool
pack_texture(struct hk_device *dev, struct hk_image_view *view,
             unsigned view_plane, enum hk_desc_usage usage,
             struct hk_native_texture_descriptor *out)
{
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct hk_image_plane *plane = &image->planes[image_plane];
   struct ail_layout *layout = &plane->layout;
   uint64_t base_addr = hk_image_base_address(image, image_plane);

   const bool sampled_like = usage == HK_DESC_USAGE_SAMPLED ||
                             usage == HK_DESC_USAGE_SAMPLED_FILTERABLE;
   bool cubes_to_2d = !sampled_like;

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = format_for_plane(view, view_plane);
   const unsigned packed_ycbcr = agx_native_graphics(&dev->dev)
      ? hk_packed_ycbcr_bits(view->vk.format) : 0;
   if (packed_ycbcr)
      p_format = packed_ycbcr == 8 ? PIPE_FORMAT_R8G8B8A8_UNORM :
                                    PIPE_FORMAT_R16G16B16A16_UNORM;
   const bool padded_rgb =
      (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
                          hk_padded_image_format(p_format) != p_format;
   if (padded_rgb)
      p_format = hk_padded_image_format(p_format);
   if (agx_native_graphics(&dev->dev)) {
      enum pipe_format integer = hk_scaled_texture_format(p_format);
      enum pipe_format normalized = hk_uscaled_normalized_format(p_format);
      if (usage == HK_DESC_USAGE_SAMPLED_FILTERABLE &&
          normalized != PIPE_FORMAT_NONE) {
         p_format = normalized;
         integer = PIPE_FORMAT_NONE;
      }
      /* Scaled formats sample through an integer descriptor and a shader
       * conversion (see hk_descriptor_set.c). Background tile reloads and
       * storage views need the integer physical values too, or a later
       * partial render saturates untouched USCALED pixels. */
      if (integer != PIPE_FORMAT_NONE)
         p_format = integer;
   }
   const struct util_format_description *desc =
      util_format_description(p_format);

   struct hk_3d denom = view_denominator(view);

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   /* A three-component image is stored in a four-component physical format,
    * so the padded channel is whatever the allocation happened to contain.
    * The API format has no alpha, and Vulkan requires a sampled texel with no
    * alpha channel to read one.  Taking the swizzle from the physical format
    * returns the padding instead, which is why every sampled RGB/BGR format
    * mismatched.  Force it back.
    */
   if (padded_rgb && sampled_like)
      format_swizzle[3] = PIPE_SWIZZLE_1;

   /* Different APIs have different depth/stencil swizzle rules. Vulkan expects
    * R001 behaviour, override here because Mesa's format table is not that.
    */
   if (util_format_is_depth_or_stencil(p_format)) {
      format_swizzle[0] = PIPE_SWIZZLE_X;
      format_swizzle[1] = PIPE_SWIZZLE_0;
      format_swizzle[2] = PIPE_SWIZZLE_0;
      format_swizzle[3] = PIPE_SWIZZLE_1;
   }

   /* We only have a single swizzle for the user swizzle and the format
    * fixup, so compose them now.
    */
   uint8_t out_swizzle[4];
   uint8_t view_swizzle[4] = {
      vk_swizzle_to_pipe(view->vk.swizzle.r),
      vk_swizzle_to_pipe(view->vk.swizzle.g),
      vk_swizzle_to_pipe(view->vk.swizzle.b),
      vk_swizzle_to_pipe(view->vk.swizzle.a),
   };

   unsigned layers = view->vk.layer_count;
   if (view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D) {
      layers = DIV_ROUND_UP(layout->depth_px, denom.z);
   } else if (!cubes_to_2d &&
              (view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
               view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {

      layers /= 6;
   }

   util_format_compose_swizzles(format_swizzle, view_swizzle, out_swizzle);

   struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
   {
      cfg.dimension = translate_image_view_type(
         view->vk.view_type, view->vk.image->samples > 1, layers > 1, usage,
         cubes_to_2d);
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.srgb = util_format_is_srgb(p_format);

      cfg.swizzle_r = agx_channel_from_pipe(out_swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(out_swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(out_swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(out_swizzle[3]);

      if (denom.x > 1) {
         assert(view->vk.level_count == 1);
         assert(view->vk.layer_count == 1);
         /* The level is described as a standalone image, so the hardware
          * derives that image's layer stride, not the allocation's: only the
          * first layer or slice is addressed correctly. Applications cannot
          * create such views (blockTexelViewCompatibleMultipleLayers is
          * false) and native meta copies view one layer
          * (copied_layers_src_view).
          */
         if (unlikely(layers > 1) && agx_native_graphics(&dev->dev))
            mesa_loge_once("hk: uncompressed view of %u layers of a "
                           "block-compressed image addresses only the first",
                           layers);

         cfg.address = base_addr + ail_get_layer_level_B(layout, layer, level);
         cfg.width = DIV_ROUND_UP(u_minify(layout->width_px, level), denom.x);
         cfg.height = DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
         cfg.first_level = 0;
         cfg.last_level = agx_native_graphics(&dev->dev) ? 0 : 1;
      } else {
         cfg.address = base_addr + ail_get_layer_offset_B(layout, layer);
         cfg.width = layout->width_px;
         cfg.height = layout->height_px;
         cfg.first_level = level;
         cfg.last_level = level + view->vk.level_count - 1;
      }

      if (packed_ycbcr)
         cfg.width = DIV_ROUND_UP(cfg.width, 2);

      /* To implement sparse resident textures, the hardware texture descriptor
       * can instead point to a secondary page table controlled in userspace.
       * This allows remapping pages and - crucially - disabling unmapped pages
       * to read zero and report non-resident with shader residency queries.
       * When we have a sparse map, we need to point to it here.
       *
       * However, there's a wrinkle: when handling uncompressed views of
       * compressed images in the above code, we need to offset the image
       * address to point to the specific mip level rather than use the hardware
       * "first level" field. This ensures the layouts are consistent despite us
       * munging the image dimensions. In that case, we need to also offset the
       * sparse page table accordingly. Of course, the sparse page table is in
       * terms of pages, so this trick only works when the mip level is
       * page-aligned.
       *
       * However, if the mip level is NOT page-aligned, it is in the mip tail by
       * definition. As the mip tail is always resident, there is no need for a
       * sparse page table. So either:
       *
       * 1. We are in the mip tail and don't need a sparse map, or
       * 2. We are not but the level is page-aligned in the sparse map.
       *
       * Either way we're okay.
       */
      if (plane->sparse_map && level < layout->mip_tail_first_lod) {
         unsigned page = 0;
         if (denom.x > 1) {
            page = ail_bytes_to_pages(layout->level_offsets_B[level]);
         }

         cfg.mode = AGX_IMAGE_MODE_SPARSE;
         cfg.address = plane->sparse_map_addr +
                       ail_page_to_sparse_index_el(layout, layer, page) *
                          AIL_SPARSE_ELSIZE_B;
      }

      cfg.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      cfg.unk_mipmapped = layout->levels > 1;
      cfg.srgb_2_channel = cfg.srgb && util_format_colormask(desc) == 0x3;

      if (layout->compressed) {
         cfg.compressed = true;
         cfg.extended = true;
      }

      if (layout->compressed) {
         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }

      if (layout->tiling == AIL_TILING_LINEAR &&
          (hk_image_view_type_is_array(view->vk.view_type))) {

         cfg.depth_linear = layers;
         cfg.layer_stride_linear = layout->layer_stride_B - 0x80;
         cfg.extended = true;
      } else {
         assert((layout->tiling != AIL_TILING_LINEAR) || (layers == 1));
         cfg.depth = layers;
      }

      if (view->vk.image->samples > 1) {
         cfg.samples = agx_translate_sample_count(view->vk.image->samples);
      }

      if (layout->tiling == AIL_TILING_LINEAR) {
         cfg.stride = ail_get_linear_stride_B(layout, 0) - 16;
      } else if (layout->tiling == AIL_TILING_INTERCHANGE) {
         /* On HAL200 Interchange uses the Linear row-stride field of the
          * texture descriptor (word1[61:46]), in macroblocks per row minus
          * one; genxml stores stride >> 4. */
         cfg.stride = (layout->stride_el[0] / 16 - 1) << 4;
      } else {
         cfg.page_aligned_layers = layout->page_aligned_layers;
      }

   }

   if (!hk_pack_native_texture(agx_chip_descriptor_generation(dev->dev.chip), out, &cfg))
      return false;

   /* Only sampled and storage descriptors answer an application's size
    * query. Input attachments are fetched as arrays, and the attachment,
    * eMRT and background descriptors are read by driver code that does not
    * take the level-zero lowering (AGX_NIR_TEXTURE_LEVEL_ZERO_EXTENT). */
   if (agx_chip_texture_descriptor_spare_word(dev->dev.chip) &&
       (sampled_like || usage == HK_DESC_USAGE_STORAGE)) {
      hk_native_texture_set_level_zero_extent(
         out, u_minify(cfg.width, cfg.first_level),
         u_minify(cfg.height, cfg.first_level));
   }

   return true;
}

static bool
pack_pbe(struct hk_device *dev, struct hk_image_view *view, unsigned view_plane,
         enum hk_desc_usage usage, struct hk_native_pbe_descriptor *out,
         uint64_t *atomic_sw_out, uint32_t *store_window_out)
{
   if (store_window_out)
      memset(store_window_out, 0, HK_M5_STORE_WINDOW_SIZE);

   const unsigned generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   if (!hk_native_pbe_hardware_supported(generation)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct ail_layout *layout = &image->planes[image_plane].layout;
   uint64_t base_addr = hk_image_base_address(image, image_plane);

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = format_for_plane(view, view_plane);
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      p_format = hk_padded_image_format(p_format);
      enum pipe_format integer = hk_scaled_texture_format(p_format);
      if (integer != PIPE_FORMAT_NONE)
         p_format = integer;
   }

   bool eot =
      usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT;

   /* The tilebuffer is already in sRGB space if needed. Do not convert for
    * end-of-tile descriptors.
    */
   if (eot)
      p_format = util_format_linear(p_format);

   if (eot && (agx_native_graphics(&dev->dev) ||
                agx_chip_g17_m5(dev->dev.chip)))
      p_format = hk_eot_pbe_format(&dev->dev, p_format);

   const struct util_format_description *desc =
      util_format_description(p_format);

   bool msaa = view->vk.image->samples > 1;
   struct hk_3d denom = view_denominator(view);

   unsigned layers = view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D
                        ? image->vk.extent.depth
                        : view->vk.layer_count;
   /* A non-layered auxiliary program addresses only layer zero, even when
    * the attachment view spans more layers. Keep the linear extension and
    * native layer count consistent with its 2D descriptor dimension. */
   if (usage == HK_DESC_USAGE_BG_EOT)
      layers = 1;

   struct AGX_PBE cfg = {AGX_PBE_header};
   if (atomic_sw_out)
      *atomic_sw_out = 0;
   {
      cfg.dimension =
         translate_image_view_type(view->vk.view_type, msaa, layers > 1,
                                   usage, usage != HK_DESC_USAGE_SAMPLED);

      /* The M5 and M5 Pro (G17G, G17S) write a layered image as 3D, with the
       * layer in the z coordinate (agx3_pack_binary.c); only the M5 Max
       * (G17C) takes a layer operand. Describe those stores as 3D, as
       * Gallium does: a slice of a 3D image is a layer of the 2D array.
       * End-of-tile stores use block access and keep their own dimension.
       */
      const bool layers_as_3d =
         generation == 10 && agx_chip_g17_m5(dev->dev.chip) &&
         dev->dev.chip != AGX_CHIP_G17C && !eot &&
         layout->tiling != AIL_TILING_LINEAR &&
         cfg.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY;

      /* The packer admits no compressed 3D PBE. A one-layer storage view of a
       * compressed image instead gets the image's 2D PBE, and its store
       * record makes layered stores store to it in 2D (hk_native_descriptor.h).
       */
      const bool store_2d =
         layers_as_3d && store_window_out &&
         usage == HK_DESC_USAGE_STORAGE && layout->compressed &&
         layers == 1 && !msaa && denom.x == 1 && denom.y == 1 &&
         hk_pack_m5_store_2d(u_minify(layout->width_px, level),
                             u_minify(layout->height_px, level),
                             store_window_out);
      const bool store_3d = layers_as_3d && !store_2d;
      if (store_2d)
         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
      else if (store_3d)
         cfg.dimension = AGX_TEXTURE_DIMENSION_3D;

      /* That 3D store addresses at most HK_M5_STORE_MAX_SLICES slices, and
       * the view is described below as layers << level slices deep. When
       * that is more, describe a linear window over the level instead, which
       * hk_nir_lower_descriptors() stores through at the texel's tiled
       * element index (hk_native_descriptor.h), with the tiled arithmetic of
       * image atomics. Arrays of a 3D image and compressed (one-layer views:
       * the 2D description above), block-compressed or interchange layouts
       * keep their description.
       */
      const unsigned window_element_B = util_format_get_blocksize(p_format);
      uint32_t window_rows = 0;
      if (store_3d && store_window_out &&
          usage == HK_DESC_USAGE_STORAGE && !msaa && !layout->mipmapped_z &&
          (layout->tiling == AIL_TILING_GPU ||
           layout->tiling == AIL_TILING_TWIDDLED) &&
          !layout->compressed && denom.x == 1 && denom.y == 1 &&
          ((uint64_t)layers << level) > HK_M5_STORE_MAX_SLICES &&
          window_element_B == util_format_get_blocksize(layout->format)) {
         window_rows = hk_pack_m5_store_window(
            layout->layer_stride_B,
            layout->level_offsets_B[level + 1] - layout->level_offsets_B[level],
            window_element_B, u_minify(layout->width_px, level),
            u_minify(layout->height_px, level),
            layout->tilesize_el[level].width_el,
            layout->tilesize_el[level].height_el, layers, store_window_out);
      }

      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.srgb = util_format_is_srgb(p_format);

      assert(desc->nr_channels >= 1 && desc->nr_channels <= 4);

      for (unsigned i = 0; i < desc->nr_channels; ++i) {
         if (desc->swizzle[i] == 0)
            cfg.swizzle_r = i;
         else if (desc->swizzle[i] == 1)
            cfg.swizzle_g = i;
         else if (desc->swizzle[i] == 2)
            cfg.swizzle_b = i;
         else if (desc->swizzle[i] == 3)
            cfg.swizzle_a = i;
      }

      cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
      cfg.unk_mipmapped = layout->levels > 1;

      if (msaa & !eot) {
         /* Multisampled images are bound like buffer textures, with
          * addressing arithmetic to determine the texel to write.
          *
          * Note that the end-of-tile program uses real multisample images
          * with image_write_block instructions.
          */
         unsigned blocksize_B = util_format_get_blocksize(p_format);
         unsigned size_px =
            (layout->size_B - layout->layer_stride_B * layer) / blocksize_B;

         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(size_px, cfg.width);
         cfg.stride = (cfg.width * blocksize_B) - 4;
         cfg.layers = 1;
         cfg.levels = 1;

         cfg.buffer += layout->level_offsets_B[level];
         cfg.level = 0;
      } else if (window_rows) {
         /* The store window: the level's texels of every layer of the view,
          * as a linear 2D buffer, as texel buffers are written. */
         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.buffer = base_addr + ail_get_layer_level_B(layout, layer, level);
         cfg.unk_mipmapped = false;
         cfg.width = HK_M5_STORE_WINDOW_WIDTH;
         cfg.height = window_rows;
         cfg.stride = (cfg.width * window_element_B) - 4;
         cfg.layers = 1;
         cfg.levels = 1;
         cfg.level = 0;
      } else {
         if (denom.x > 1) {
            assert(denom.z == 1 && "todo how to handle?");
            assert(view->vk.level_count == 1);
            assert(view->vk.layer_count == 1);

            cfg.buffer =
               base_addr + ail_get_layer_level_B(layout, layer, level);
            cfg.width =
               DIV_ROUND_UP(u_minify(layout->width_px, level), denom.x);
            cfg.height =
               DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
            cfg.level = 0;
         } else {
            cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
            cfg.width = layout->width_px;
            cfg.height = layout->height_px;
            cfg.level = level;
         }

         /* A single-layer BG/EOT descriptor is 2D even when the API view
          * is an array. It does not need the linear array-stride extension. */
         if (layout->tiling == AIL_TILING_LINEAR &&
             hk_image_view_type_is_array(view->vk.view_type) &&
             (layers > 1 || cfg.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY)) {

            cfg.depth_linear = layers;
            cfg.layer_stride_linear = (layout->layer_stride_B - 0x80);
            cfg.extended = true;
         } else {
            assert((layout->tiling != AIL_TILING_LINEAR) || (layers == 1));
            cfg.layers = layers;

            /* A 3D image's depth shrinks with the selected level, the layer
             * count of an array does not: scale the depth so that, at this
             * level, it still bounds every layer of the view. */
            if (store_3d && cfg.level)
               cfg.layers = MIN2((uint64_t)layers << cfg.level, 1u << 16);
         }

         cfg.levels = image->vk.mip_levels;

         if (layout->tiling == AIL_TILING_LINEAR) {
            cfg.stride = ail_get_linear_stride_B(layout, level) - 4;
            assert(cfg.levels == 1);
         } else if (layout->tiling == AIL_TILING_INTERCHANGE) {
            /* Gen9 PBE: Interchange takes the Linear stride field, macroblocks
             * per row minus one at word1[59:44]; the codec encodes
             * stride_B / 16 - 1 there (stride_B = stride + 4).
             * Hardware-verified: v6 ix-pattern@mb. */
            cfg.stride = (layout->stride_el[0] / 16) * 16 - 4;
            assert(cfg.levels == 1);
         } else {
            cfg.page_aligned_layers = layout->page_aligned_layers;
         }

         if (image->vk.samples > 1)
            cfg.samples = agx_translate_sample_count(image->vk.samples);
      }

      if (layout->compressed && usage != HK_DESC_USAGE_EMRT) {
         cfg.compressed = true;
         cfg.extended = true;

         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }

      /* When the descriptor isn't extended architecturally, we use
       * the last 8 bytes as a sideband to accelerate image atomics.
       */
      if (hk_native_pbe_supported(generation) && !cfg.extended &&
          (layout->writeable_image || usage == HK_DESC_USAGE_EMRT)) {

         if (msaa) {
            assert(denom.x == 1 && "no MSAA of block-compressed");

            cfg.aligned_width_msaa_sw =
               align(u_minify(layout->width_px, level),
                     layout->tilesize_el[level].width_el);

            cfg.sample_count_log2_sw = util_logbase2(image->vk.samples);
         } else {
            cfg.level_offset_sw = ail_get_level_offset_B(layout, cfg.level);
         }

         if (layout->tiling != AIL_TILING_LINEAR) {
            struct ail_tile tile_size = layout->tilesize_el[level];
            cfg.tile_width_sw = tile_size.width_el;
            cfg.tile_height_sw = tile_size.height_el;

            cfg.layer_stride_sw = layout->layer_stride_B;
         }
      }

      /* Gen9's architectural PBE consumes all 32 bytes, so it cannot reuse
       * AGX2 words 4/5 as software metadata.  Preserve the same compact ABI in
       * the descriptor's dedicated tail: level offset [0:26], log2 tile width
       * [27:30], log2 tile height [31:34], layer stride [35:61], and log2
       * sample count [62:63].  The
       * shader-side Gen9 decoder consumes this word independently of the
       * native PBE. */
      if (atomic_sw_out && (generation == 9 || generation == 10) && !cfg.extended &&
          (layout->writeable_image || usage == HK_DESC_USAGE_EMRT)) {
         /* A linear image addresses texels through the native PBE's row
          * stride and is never layered here, so its sideband carries no tile
          * geometry or layer stride. */
         const bool linear = layout->tiling == AIL_TILING_LINEAR;
         if (!hk_pack_g17p_image_atomic_sw(
                ail_get_level_offset_B(layout, cfg.level),
                linear ? 1 : layout->tilesize_el[level].width_el,
                linear ? 1 : layout->tilesize_el[level].height_el,
                linear ? 0 : layout->layer_stride_B,
                msaa ? image->vk.samples : 1, atomic_sw_out)) {
            if (store_window_out)
               memset(store_window_out, 0, HK_M5_STORE_WINDOW_SIZE);
            return false;
         }
      }
   }

   /* eMRT uses the ordinary native array-store path, not the non-array
    * end-of-tile builtin. Multisample stores instead address the linear
    * buffer descriptor constructed above. */

   /* A window record must never describe a PBE that did not pack. */
   const bool packed = hk_pack_native_pbe(generation, out, &cfg);
   if (!packed && store_window_out)
      memset(store_window_out, 0, HK_M5_STORE_WINDOW_SIZE);
   return packed;
}

/* Put a view texture in the device heap: at the captured index when the view
 * replays (VK_EXT_descriptor_buffer), else anywhere. *index is 0 on failure,
 * so the destroy path never frees an index the view does not own. */
static VkResult
hk_image_view_heap_add(struct hk_device *dev, const struct hk_image_view *view,
                       const struct hk_image_view_capture *replay, unsigned p,
                       enum hk_view_heap which, const void *desc,
                       uint32_t *index)
{
   VkResult result;
   if (replay && replay->index[p][which]) {
      result = hk_descriptor_table_add_at(dev, &dev->sampled_images, desc, 32,
                                          replay->index[p][which]);
      *index = result == VK_SUCCESS ? replay->index[p][which] : 0;
   } else if (view->vk.create_flags &
              VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT) {
      result = hk_descriptor_table_add_replay(dev, &dev->sampled_images, desc,
                                              32, index);
      if (result != VK_SUCCESS)
         *index = 0;
   } else {
      result = hk_descriptor_table_add(dev, &dev->sampled_images, desc, 32,
                                       index);
      if (result != VK_SUCCESS)
         *index = 0;
   }
   return result;
}

/* Capture/replay views free their textures to the table's replay pool. */
static void
hk_image_view_heap_remove(struct hk_device *dev,
                          const struct hk_image_view *view, uint32_t index)
{
   if (view->vk.create_flags &
       VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT)
      hk_descriptor_table_remove_replay(dev, &dev->sampled_images, index);
   else
      hk_descriptor_table_remove(dev, &dev->sampled_images, index);
}

static VkResult
hk_image_view_init(struct hk_device *dev, struct hk_image_view *view,
                   const VkImageViewCreateInfo *pCreateInfo)
{
   VK_FROM_HANDLE(hk_image, image, pCreateInfo->image);
   memset(view, 0, sizeof(*view));

   vk_image_view_init(&dev->vk, &view->vk, pCreateInfo);

   const VkOpaqueCaptureDescriptorDataCreateInfoEXT *replay_info =
      (pCreateInfo->flags &
       VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT)
         ? vk_find_struct_const(pCreateInfo->pNext,
                                OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT)
         : NULL;
   struct hk_image_view_capture replay_data;
   if (replay_info)
      memcpy(&replay_data, replay_info->opaqueCaptureDescriptorData,
             sizeof(replay_data));
   const struct hk_image_view_capture *replay =
      replay_info ? &replay_data : NULL;

   const float min_lod = MAX2(view->vk.min_lod - view->vk.base_mip_level, 0.0);
   view->sampled_min_lod_fp16 = _mesa_float_to_half(min_lod);
   view->sampled_min_lod_uint16 = min_lod;

   {
      const enum pipe_format logical = hk_format_to_pipe_format(view->vk.format);
      const enum pipe_format integer = hk_scaled_texture_format(logical);
      view->g17p_scaled_conversion =
         integer == PIPE_FORMAT_NONE ? 0
         : util_format_is_pure_sint(integer) ? 1 : 2;
      view->g17p_uscaled_normalized =
         hk_uscaled_normalized_format(logical) != PIPE_FORMAT_NONE;
   }

   /* First, figure out which image planes we need. For depth/stencil, we only
    * have one aspect viewed at a time.
    */
   if (image->vk.aspects &
       (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {

      view->plane_count = 1;
      view->planes[0].image_plane =
         hk_image_aspects_to_plane(image, view->vk.aspects);
   } else {
      /* For other formats, retrieve the plane count from the aspect mask
       * and then walk through the aspect mask to map each image plane
       * to its corresponding view plane
       */
      assert(util_bitcount(view->vk.aspects) ==
             vk_format_get_plane_count(view->vk.format));
      view->plane_count = 0;
      u_foreach_bit(aspect_bit, view->vk.aspects) {
         view->planes[view->plane_count++].image_plane =
            hk_image_aspects_to_plane(image, 1u << aspect_bit);
      }
   }

   /* A driver-internal view made only to be sampled, as meta operations
    * make for their sources, is only ever written to sampled-image
    * descriptors: skip the storage, attachment and end-of-tile packets.
    */
   const bool sampled_only =
      (pCreateInfo->flags & VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA) &&
      view->vk.usage == VK_IMAGE_USAGE_SAMPLED_BIT;

   /* Fill in each view plane separately */
   for (unsigned p = 0; p < view->plane_count; p++) {
      const bool textures_valid =
         pack_texture(dev, view, p, HK_DESC_USAGE_SAMPLED,
                      &view->planes[p].sampled) &&
         (sampled_only ||
          (pack_texture(dev, view, p, HK_DESC_USAGE_STORAGE,
                        &view->planes[p].ro_storage) &&
         pack_texture(dev, view, p, HK_DESC_USAGE_INPUT,
                      &view->planes[p].ia) &&
         pack_texture(dev, view, p, HK_DESC_USAGE_EMRT,
                      &view->planes[p].emrt_texture) &&
         pack_texture(dev, view, p, HK_DESC_USAGE_BG_EOT,
                      &view->planes[p].background) &&
         pack_texture(dev, view, p, HK_DESC_USAGE_LAYERED_BG_EOT,
                      &view->planes[p].layered_background)));
      if (!textures_valid)
         return VK_ERROR_FORMAT_NOT_SUPPORTED;

      if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ||
          dev->dev.chip == AGX_CHIP_G18G) {
         VkResult result = hk_image_view_heap_add(
            dev, view, replay, p, HK_VIEW_HEAP_SAMPLED, &view->planes[p].sampled,
            &view->planes[p].sampled_heap_index);
         if (result != VK_SUCCESS)
            return result;
         view->planes[p].filterable_valid =
            hk_uscaled_normalized_format(hk_padded_image_format(
               hk_format_to_pipe_format(view->vk.format))) != PIPE_FORMAT_NONE &&
            pack_texture(dev, view, p, HK_DESC_USAGE_SAMPLED_FILTERABLE,
                         &view->planes[p].sampled_filterable);
         if (view->planes[p].filterable_valid) {
            result = hk_image_view_heap_add(
               dev, view, replay, p, HK_VIEW_HEAP_FILTERABLE,
               &view->planes[p].sampled_filterable,
               &view->planes[p].filterable_heap_index);
            if (result != VK_SUCCESS) {
               /* Nothing was reserved, so the destroy path must not try to
                * release an index we do not own. */
               view->planes[p].filterable_valid = false;
               return result;
            }
         }
         if (view->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT) {
            result = hk_image_view_heap_add(
               dev, view, replay, p, HK_VIEW_HEAP_STORAGE,
               &view->planes[p].ro_storage,
               &view->planes[p].storage_heap_index);
            if (result != VK_SUCCESS)
               return result;
         }
         /* Input attachments use an array-shaped descriptor independently
          * of the application sampled view. Keep the exact native bytes in
          * the indexed heap rather than sampling through the wrong shape. */
         if (view->vk.usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) {
            result = hk_image_view_heap_add(
               dev, view, replay, p, HK_VIEW_HEAP_INPUT, &view->planes[p].ia,
               &view->planes[p].input_heap_index);
            if (result != VK_SUCCESS)
               return result;
         }
      }

      if (!sampled_only) {
         view->planes[p].storage_pbe_valid = pack_pbe(
            dev, view, p, HK_DESC_USAGE_STORAGE, &view->planes[p].storage,
            &view->planes[p].storage_atomic_sw,
            view->planes[p].storage_window);
         view->planes[p].eot_pbe_valid =
            pack_pbe(dev, view, p, HK_DESC_USAGE_BG_EOT, &view->planes[p].eot,
                     NULL, NULL);
         view->planes[p].emrt_pbe_valid = pack_pbe(
            dev, view, p, HK_DESC_USAGE_EMRT, &view->planes[p].emrt_pbe,
            &view->planes[p].emrt_atomic_sw, NULL);
         if ((agx_native_graphics(&dev->dev) ||
              agx_chip_g17_m5(dev->dev.chip)) &&
             (view->vk.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
             view->planes[p].emrt_pbe_valid) {
            const struct ail_layout *layout =
               &image->planes[view->planes[p].image_plane].layout;
            const unsigned level = view->vk.base_mip_level;
            view->planes[p].emrt_aligned_width =
               align(u_minify(layout->width_px, level),
                     layout->tilesize_el[level].width_el);
            VkResult result = hk_image_view_heap_add(
               dev, view, replay, p, HK_VIEW_HEAP_EMRT_READ,
               &view->planes[p].emrt_texture,
               &view->planes[p].emrt_read_heap_index);
            if (result != VK_SUCCESS)
               return result;
         }
         view->planes[p].layered_eot_pbe_valid =
            pack_pbe(dev, view, p, HK_DESC_USAGE_LAYERED_BG_EOT,
                     &view->planes[p].layered_eot, NULL, NULL);
      }
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyImageView(VkDevice _device, VkImageView imageView,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VK_FROM_HANDLE(hk_image_view, view, imageView);

   if (!view)
      return;

   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ||
       dev->dev.chip == AGX_CHIP_G18G) {
      for (unsigned p = 0; p < view->plane_count; ++p) {
         if (view->planes[p].emrt_read_heap_index)
            hk_image_view_heap_remove(dev, view,
                                      view->planes[p].emrt_read_heap_index);
         if (view->planes[p].storage_heap_index)
            hk_image_view_heap_remove(dev, view,
                                      view->planes[p].storage_heap_index);
         if (view->planes[p].input_heap_index)
            hk_image_view_heap_remove(dev, view,
                                      view->planes[p].input_heap_index);
         if (view->planes[p].sampled_heap_index)
            hk_image_view_heap_remove(dev, view,
                                      view->planes[p].sampled_heap_index);
         /* The filterable descriptor was reserved in hk_image_view_init but
          * never released here, leaking one sampled_images entry per view
          * whose format has a uscaled-normalized equivalent. The table only
          * grows (doubling to max_alloc, then VK_ERROR_OUT_OF_HOST_MEMORY),
          * so a long run creating many views eventually cannot create any
          * more. Keyed on filterable_valid rather than a nonzero index
          * because index 0 is a valid descriptor index. */
         if (view->planes[p].filterable_valid)
            hk_image_view_heap_remove(dev, view,
                                      view->planes[p].filterable_heap_index);
      }
   }

   vk_image_view_finish(&view->vk);
   vk_free2(&dev->vk.alloc, pAllocator, view);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_image_view *view;
   VkResult result;

   view = vk_alloc2(&dev->vk.alloc, pAllocator, sizeof(*view), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = hk_image_view_init(dev, view, pCreateInfo);
   if (result != VK_SUCCESS) {
      hk_DestroyImageView(_device, hk_image_view_to_handle(view), pAllocator);
      return result;
   }

   /* Compressed data only decodes through its own channel layout. Only a
    * mutable image created without a format list under the
    * hk_mutable_format_srgb_pair application option can reach this. */
   const struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   if (view->plane_count == 1 && view->vk.aspects == VK_IMAGE_ASPECT_COLOR_BIT &&
       !ail_is_view_compatible(
          &image->planes[view->planes[0].image_plane].layout,
          hk_format_to_pipe_format(view->vk.view_format))) {
      mesa_loge_once("compressed %s image viewed as %s: the "
                     "hk_mutable_format_srgb_pair assumption does not hold",
                     util_format_short_name(
                        hk_format_to_pipe_format(image->vk.format)),
                     util_format_short_name(
                        hk_format_to_pipe_format(view->vk.view_format)));
   }

   /* Frame anatomy profiler: the formats that reinterpret a mutable image,
    * which its compression decision could not see without a format list. */
   const struct vk_image *vk_image = view->vk.image;
   if (unlikely(dev->frame_anatomy) &&
       (vk_image->create_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
       view->vk.view_format != vk_image->format) {
      hk_frame_anatomy_note(
         dev, "view %ux%u fmt=%s samples=%u -> %s usage=0x%x",
         vk_image->extent.width, vk_image->extent.height,
         util_format_short_name(hk_format_to_pipe_format(vk_image->format)),
         vk_image->samples,
         util_format_short_name(hk_format_to_pipe_format(view->vk.view_format)),
         (unsigned)view->vk.usage);
   }

   *pView = hk_image_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetImageViewOpaqueCaptureDescriptorDataEXT(
   VkDevice device, const VkImageViewCaptureDescriptorDataInfoEXT *pInfo,
   void *pData)
{
   VK_FROM_HANDLE(hk_image_view, view, pInfo->imageView);

   struct hk_image_view_capture capture = {0};
   for (unsigned p = 0; p < view->plane_count; ++p) {
      capture.index[p][HK_VIEW_HEAP_SAMPLED] = view->planes[p].sampled_heap_index;
      capture.index[p][HK_VIEW_HEAP_FILTERABLE] =
         view->planes[p].filterable_valid
            ? view->planes[p].filterable_heap_index
            : 0;
      capture.index[p][HK_VIEW_HEAP_STORAGE] = view->planes[p].storage_heap_index;
      capture.index[p][HK_VIEW_HEAP_INPUT] = view->planes[p].input_heap_index;
      capture.index[p][HK_VIEW_HEAP_EMRT_READ] =
         view->planes[p].emrt_read_heap_index;
   }
   memcpy(pData, &capture, sizeof(capture));
   return VK_SUCCESS;
}
