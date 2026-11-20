/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_buffer_view.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "asahi/compiler/agx_nir.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"

#include "agx_helpers.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "vk_acceleration_structure.h"

/* Build shaders read AS positions through load_vertices(), which also decodes
 * these beyond the common set. Reported where they are tested: G17P. */
static bool
hk_as_vtx_format_supported(const struct hk_physical_device *pdev,
                           VkFormat vk_format)
{
   switch (vk_format) {
   case VK_FORMAT_R8G8B8_SNORM:
   case VK_FORMAT_R16G16B16_SNORM:
   case VK_FORMAT_R64G64_SFLOAT:
   case VK_FORMAT_R64G64B64_SFLOAT:
   case VK_FORMAT_R64G64B64A64_SFLOAT:
      return pdev->dev.chip == AGX_CHIP_G17P;
   default:
      return vk_acceleration_struct_vtx_format_supported(vk_format);
   }
}

VkFormatFeatureFlags2
hk_get_buffer_format_features(struct hk_physical_device *pdev,
                              VkFormat vk_format)
{
   VkFormatFeatureFlags2 features = 0;
   enum pipe_format p_format = hk_format_to_pipe_format(vk_format);

   if (pdev->vk.supported_extensions.KHR_acceleration_structure &&
       hk_as_vtx_format_supported(pdev, vk_format))
      features |= VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;

   if (p_format == PIPE_FORMAT_NONE)
      return features;

   if (agx_vbo_supports_format(p_format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;

   if (ail_pixel_format[p_format].texturable &&
       !util_format_is_depth_or_stencil(p_format)) {

      features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

      /* RGB32 specially supported for uniform texel buffers only. */
      if (util_is_power_of_two_nonzero(util_format_get_blocksize(p_format))) {
         /* Reads obtain their format from the buffer texture descriptor. */
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
                     VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                     VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
      }

      if (p_format == PIPE_FORMAT_R32_UINT || p_format == PIPE_FORMAT_R32_SINT ||
          (p_format == PIPE_FORMAT_R32_FLOAT &&
           agx_chip_g17p_or_m5(pdev->dev.chip)))
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }

   return features;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkBufferView *pBufferView)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_buffer, buffer, pCreateInfo->buffer);
   struct hk_buffer_view *view;

   view = vk_buffer_view_create(&device->vk, pCreateInfo, pAllocator,
                                sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   enum pipe_format format = hk_format_to_pipe_format(view->vk.format);
   const struct util_format_description *desc = util_format_description(format);
   const bool g17p = agx_chip_native_or_m5(device->dev.chip);
   const bool rgb32 =
      ail_pixel_format[format].channels == AGX_CHANNELS_R32G32B32_EMULATED;

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   if (util_format_is_depth_or_stencil(format)) {
      assert(!util_format_is_depth_and_stencil(format) &&
             "separate stencil always used");

      /* Broadcast depth and stencil */
      format_swizzle[0] = 0;
      format_swizzle[1] = 0;
      format_swizzle[2] = 0;
      format_swizzle[3] = 0;
   }

   /* Decompose the offset into a multiple of 16-bytes (which we can include in
    * the address) and an extra texel-aligned tail offset of up to 15 bytes.
    *
    * This lets us offset partially in the shader instead, getting
    * around alignment restrictions on the base address pointer.
    */
   uint64_t head_offset_B = view->vk.offset & ~0xf;
   uint64_t rw_base = hk_buffer_address_rw(buffer, 0) + head_offset_B;
   uint64_t ro_base = hk_buffer_address_ro(buffer, 0) + head_offset_B;

   /* Which mapping the sampled texel-buffer descriptor addresses on G17P.
    *
    * The read-only shadow at +sparse_ro_offset is the mechanism every other
    * read-only descriptor uses to get sparseResidencyNonResidentStrict for
    * free: an unbound page reads as zero there.  For a texel buffer G17P
    * cannot rely on it, so the descriptor names the primary mapping and the
    * shader masks a non-resident texel to zero from the residency bitmap
    * instead -- one address and one residency source for both the sparse
    * fetch and the ordinary read.  See buffer_texel_resident() in
    * agx_nir_lower_texture.c.
    *
    * For a non-sparse buffer hk_buffer_address() returns the same address
    * either way (the shadow only exists for buffer->va != NULL), so this
    * cannot move an ordinary texel buffer.
    */
   const uint64_t sampled_base = g17p ? rw_base : ro_base;

   uint32_t tail_offset_B = view->vk.offset & 0xf;
   /* The G17P metadata names the view's exact address
    * (AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET). RGB32 permits component-aligned
    * offsets, which cannot in general be expressed as a whole-texel tail, and
    * it reads only that address; its unused hardware texture stays aligned.
    */
   const uint64_t metadata_base = sampled_base + tail_offset_B;
   if (g17p && rgb32)
      tail_offset_B = 0;
   uint32_t tail_offset_el = tail_offset_B / util_format_get_blocksize(format);
   assert(tail_offset_el * util_format_get_blocksize(format) == tail_offset_B &&
          "must be texel aligned");

   const unsigned generation =
      agx_chip_descriptor_generation(device->dev.chip);

   struct AGX_TEXTURE texture = {AGX_TEXTURE_header};
   {
      texture.dimension = AGX_TEXTURE_DIMENSION_2D;
      texture.layout = AGX_LAYOUT_LINEAR;
      texture.channels = ail_pixel_format[format].channels;
      texture.type = ail_pixel_format[format].type;
      texture.swizzle_r = agx_channel_from_pipe(format_swizzle[0]);
      texture.swizzle_g = agx_channel_from_pipe(format_swizzle[1]);
      texture.swizzle_b = agx_channel_from_pipe(format_swizzle[2]);
      texture.swizzle_a = agx_channel_from_pipe(format_swizzle[3]);

      texture.width = AGX_TEXTURE_BUFFER_WIDTH;
      texture.height = DIV_ROUND_UP(view->vk.elements, texture.width);
      texture.first_level = texture.last_level = 0;

      texture.address = sampled_base;
      texture.buffer_size_sw = view->vk.elements;
      texture.buffer_offset_sw = tail_offset_el;

      texture.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      texture.srgb_2_channel =
         texture.srgb && util_format_colormask(desc) == 0x3;

      texture.depth = 1;
      texture.stride =
         (texture.width * util_format_get_blocksize(format)) - 16;
   }
   if (g17p && rgb32) {
      texture.channels = AGX_CHANNELS_R32;
      texture.swizzle_g = AGX_CHANNEL_0;
      texture.swizzle_b = AGX_CHANNEL_0;
      texture.stride = texture.width * sizeof(uint32_t) - 16;
   }
   if (g17p) {
      /* A sparse-resident buffer carries a per-page residency bitmap, and the
       * buffer texture descriptor has no hardware sparse mode to report
       * residency with, so a sparse texel fetch reads that bitmap instead.
       * Pack the map and the element size the shader needs to find the page;
       * see AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET.  A non-power-of-two
       * element size (the emulated RGB32 layout) takes the raw load path,
       * which claims residency anyway, so leave the word zero for it. */
      const uint32_t sparse = hk_buffer_sparse_texel_word(
         buffer, util_format_get_blocksize(format));

      view->buffer_metadata = (struct hk_g17p_buffer_texture_metadata){
         .elements = view->vk.elements,
         .offset_el = tail_offset_el,
         .address = metadata_base,
         .rgb32 = rgb32,
         .sparse = sparse,
      };
      /* Gen9 has no AGX2 software tail fields. The shader consumes the
       * explicit metadata, while hardware sees the aligned base texture.
       */
      texture.buffer_offset_sw = 0;
      texture.height = DIV_ROUND_UP(view->vk.elements + tail_offset_el,
                                    texture.width);
   }
   if (!hk_pack_native_texture(generation, &view->tex, &texture)) {
      vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
      return vk_error(device, VK_ERROR_FORMAT_NOT_SUPPORTED);
   }

   struct AGX_PBE pbe = {AGX_PBE_header};
   {
      pbe.dimension = AGX_TEXTURE_DIMENSION_2D;
      pbe.layout = AGX_LAYOUT_LINEAR;
      pbe.channels = ail_pixel_format[format].channels;
      pbe.type = ail_pixel_format[format].type;
      pbe.srgb = util_format_is_srgb(format);

      assert(desc->nr_channels >= 1 && desc->nr_channels <= 4);

      for (unsigned i = 0; i < desc->nr_channels; ++i) {
         if (desc->swizzle[i] == 0)
            pbe.swizzle_r = i;
         else if (desc->swizzle[i] == 1)
            pbe.swizzle_g = i;
         else if (desc->swizzle[i] == 2)
            pbe.swizzle_b = i;
         else if (desc->swizzle[i] == 3)
            pbe.swizzle_a = i;
      }

      pbe.buffer = rw_base;
      pbe.buffer_offset_sw = tail_offset_el;

      pbe.width = AGX_TEXTURE_BUFFER_WIDTH;
      pbe.height = DIV_ROUND_UP(view->vk.elements, pbe.width);
      pbe.level = 0;
      pbe.stride = (pbe.width * util_format_get_blocksize(format)) - 4;
      pbe.layers = 1;
      pbe.levels = 1;
   }

   if (hk_native_pbe_supported(generation)) {
      hk_pack_native_pbe(generation, &view->pbe, &pbe);
   } else if (agx_chip_native_or_m5(device->dev.chip)) {
      /* AGX3's linear PBE is architectural and has no software tail field,
       * exactly like the texture above: the store lowering adds the tail from
       * the buffer metadata. A format the native packer cannot represent (the
       * emulated RGB32 layout) has no storage usage, so its view keeps the
       * null PBE.
       */
      pbe.buffer_offset_sw = 0;
      pbe.height = DIV_ROUND_UP(view->vk.elements + tail_offset_el, pbe.width);
      if (!hk_pack_native_pbe(generation, &view->pbe, &pbe))
         hk_set_null_native_pbe(generation, &view->pbe);
   } else {
      memset(&view->pbe, 0, sizeof(view->pbe));
   }

   /* G17P reaches a sampled texture through the device heap, so the view's
    * texture descriptor has to live there and the descriptor carries only its
    * index.  Without this the index stays 0 and every texel-buffer fetch
    * resolves to the null texture.  Mirrors hk_image_view.c's per-plane
    * hk_descriptor_table_add(). */
   if (g17p) {
      VkResult result = hk_descriptor_table_add(
         device, &device->sampled_images, &view->tex,
         hk_texture_descriptor_size(generation), &view->sampled_heap_index);
      if (result != VK_SUCCESS) {
         vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
         return result;
      }
   }

   *pBufferView = hk_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_buffer_view, view, bufferView);

   if (!view)
      return;

   if (view->sampled_heap_index)
      hk_descriptor_table_remove(device, &device->sampled_images,
                                 view->sampled_heap_index);

   vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
}

/*
 * VK_EXT_descriptor_buffer texel buffers on G17P.
 *
 * vkGetDescriptorEXT() names a texel buffer by address, range and format, so
 * there is no view object to own the device heap texture that G17P samples
 * through (hk_CreateBufferView()). Instead each format gets a fixed set of
 * maximum-sized buffer textures, "windows", over the application VA: 2^28
 * texels each, one starting every 2^27 texels, so any view of at most 2^27
 * texels lies inside one of them. The descriptor's software metadata selects
 * the view's texels in the window: the element offset and count that the
 * texel fetch, image load/store and atomic lowerings already apply
 * (agx_nir_lower_texture.c). The heap indices are reserved when the device is
 * created, in a fixed order, and a window is written when first used, so a
 * descriptor does not depend on the order views were asked for
 * (capture/replay). A larger view, or one outside the VA, gets a texture of
 * its own, kept until the device is destroyed.
 *
 * The emulated RGB32 formats read memory directly at the metadata address;
 * their texture is only read out of bounds, so one window serves each.
 */
#define HK_TEXEL_WINDOW_EL      (AGX_TEXTURE_BUFFER_WIDTH * AGX_TEXTURE_BUFFER_MAX_HEIGHT)
#define HK_TEXEL_WINDOW_STEP_EL (HK_TEXEL_WINDOW_EL / 2)

struct hk_texel_window_key {
   uint64_t base;
   uint32_t height;
   uint32_t format;
};
DERIVE_HASH_TABLE(hk_texel_window_key);

struct hk_texel_windows {
   simple_mtx_t lock;
   uint64_t heap_base, heap_size;
   uint32_t first_index;
   uint32_t count;
   /* Slot of a format's first window, or UINT32_MAX */
   uint32_t format_first[PIPE_FORMAT_COUNT];
   uint32_t format_windows[PIPE_FORMAT_COUNT];
   BITSET_WORD *written;
   /* hk_texel_window_key -> heap index */
   struct hash_table *dedicated;
};

static bool
hk_texel_rgb32(enum pipe_format format)
{
   return ail_pixel_format[format].channels == AGX_CHANNELS_R32G32B32_EMULATED;
}

/* The buffer texture hk_CreateBufferView() builds for G17P, over `height`
 * rows of AGX_TEXTURE_BUFFER_WIDTH texels at `base`. */
static bool
hk_pack_texel_window(const struct hk_device *dev, enum pipe_format format,
                     uint64_t base, uint32_t height,
                     struct hk_native_texture_descriptor *out)
{
   const struct util_format_description *desc = util_format_description(format);
   struct AGX_TEXTURE texture = {AGX_TEXTURE_header};

   texture.dimension = AGX_TEXTURE_DIMENSION_2D;
   texture.layout = AGX_LAYOUT_LINEAR;
   texture.channels = ail_pixel_format[format].channels;
   texture.type = ail_pixel_format[format].type;
   texture.swizzle_r = agx_channel_from_pipe(desc->swizzle[0]);
   texture.swizzle_g = agx_channel_from_pipe(desc->swizzle[1]);
   texture.swizzle_b = agx_channel_from_pipe(desc->swizzle[2]);
   texture.swizzle_a = agx_channel_from_pipe(desc->swizzle[3]);
   texture.width = AGX_TEXTURE_BUFFER_WIDTH;
   texture.height = height;
   texture.first_level = texture.last_level = 0;
   texture.address = base;
   texture.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
   texture.srgb_2_channel = texture.srgb && util_format_colormask(desc) == 0x3;
   texture.depth = 1;
   texture.stride =
      (texture.width * util_format_get_blocksize(format)) - 16;
   if (hk_texel_rgb32(format)) {
      texture.channels = AGX_CHANNELS_R32;
      texture.swizzle_g = AGX_CHANNEL_0;
      texture.swizzle_b = AGX_CHANNEL_0;
      texture.stride = texture.width * sizeof(uint32_t) - 16;
   }
   texture.buffer_offset_sw = 0;

   return hk_pack_native_texture(
      agx_chip_descriptor_generation(dev->dev.chip), out, &texture);
}

/* The matching PBE of hk_CreateBufferView(), for storage texel buffers. */
static void
hk_pack_texel_window_pbe(const struct hk_device *dev, enum pipe_format format,
                         uint64_t base, uint32_t height,
                         struct hk_native_pbe_descriptor *out)
{
   const unsigned generation = agx_chip_descriptor_generation(dev->dev.chip);
   const struct util_format_description *desc = util_format_description(format);
   struct AGX_PBE pbe = {AGX_PBE_header};

   pbe.dimension = AGX_TEXTURE_DIMENSION_2D;
   pbe.layout = AGX_LAYOUT_LINEAR;
   pbe.channels = ail_pixel_format[format].channels;
   pbe.type = ail_pixel_format[format].type;
   pbe.srgb = util_format_is_srgb(format);

   for (unsigned i = 0; i < desc->nr_channels; ++i) {
      if (desc->swizzle[i] == 0)
         pbe.swizzle_r = i;
      else if (desc->swizzle[i] == 1)
         pbe.swizzle_g = i;
      else if (desc->swizzle[i] == 2)
         pbe.swizzle_b = i;
      else if (desc->swizzle[i] == 3)
         pbe.swizzle_a = i;
   }

   pbe.buffer = base;
   pbe.buffer_offset_sw = 0;
   pbe.width = AGX_TEXTURE_BUFFER_WIDTH;
   pbe.height = height;
   pbe.level = 0;
   pbe.stride = (pbe.width * util_format_get_blocksize(format)) - 4;
   pbe.layers = 1;
   pbe.levels = 1;

   if (!hk_pack_native_pbe(generation, out, &pbe))
      hk_set_null_native_pbe(generation, out);
}

static uint64_t
hk_texel_window_step_B(enum pipe_format format)
{
   return (uint64_t)HK_TEXEL_WINDOW_STEP_EL * util_format_get_blocksize(format);
}

VkResult
hk_texel_windows_init(struct hk_device *dev)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_texel_windows *w = vk_zalloc(&dev->vk.alloc, sizeof(*w), 8,
                                          VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!w)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   w->heap_base = dev->dev.main_heap_base;
   w->heap_size = dev->dev.main_heap_size;
   for (unsigned f = 0; f < PIPE_FORMAT_COUNT; ++f)
      w->format_first[f] = UINT32_MAX;

   /* A fixed walk over the formats with texel-buffer support. */
   static const VkFormat extra[] = {
      VK_FORMAT_A4R4G4B4_UNORM_PACK16,
      VK_FORMAT_A4B4G4R4_UNORM_PACK16,
      VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR,
      VK_FORMAT_A8_UNORM_KHR,
   };
   const unsigned core = VK_FORMAT_ASTC_12x12_SRGB_BLOCK + 1;
   for (unsigned i = 1; i < core + ARRAY_SIZE(extra); ++i) {
      const VkFormat vk_format = i < core ? (VkFormat)i : extra[i - core];
      if (!(hk_get_buffer_format_features(pdev, vk_format) &
            (VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT |
             VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT)))
         continue;

      const enum pipe_format format = hk_format_to_pipe_format(vk_format);
      if (format == PIPE_FORMAT_NONE || w->format_first[format] != UINT32_MAX)
         continue;

      w->format_first[format] = w->count;
      w->format_windows[format] =
         hk_texel_rgb32(format)
            ? 1
            : DIV_ROUND_UP(w->heap_size, hk_texel_window_step_B(format));
      w->count += w->format_windows[format];
   }

   w->written = vk_zalloc(&dev->vk.alloc, BITSET_WORDS(w->count) *
                             sizeof(BITSET_WORD),
                          8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   w->dedicated = hk_texel_window_key_table_create(NULL);
   VkResult result = (w->written && w->dedicated)
                        ? hk_descriptor_table_reserve_range(
                             dev, &dev->sampled_images, w->count,
                             &w->first_index)
                        : VK_ERROR_OUT_OF_HOST_MEMORY;
   if (result != VK_SUCCESS) {
      ralloc_free(w->dedicated);
      vk_free(&dev->vk.alloc, w->written);
      vk_free(&dev->vk.alloc, w);
      return vk_error(dev, result);
   }

   simple_mtx_init(&w->lock, mtx_plain);
   dev->texel_windows = w;
   return VK_SUCCESS;
}

void
hk_texel_windows_finish(struct hk_device *dev)
{
   struct hk_texel_windows *w = dev->texel_windows;
   if (!w)
      return;

   /* The reserved and dedicated heap entries go with the heap. */
   simple_mtx_destroy(&w->lock);
   ralloc_free(w->dedicated);
   vk_free(&dev->vk.alloc, w->written);
   vk_free(&dev->vk.alloc, w);
   dev->texel_windows = NULL;
}

/* The heap texture for a view at `address` of `elements` texels, and the
 * texel offset of the view in it. */
static bool
hk_texel_window_get(struct hk_device *dev, enum pipe_format format,
                    uint64_t address, uint64_t elements, uint32_t *index,
                    uint64_t *base, uint32_t *offset_el,
                    struct hk_native_texture_descriptor *tex)
{
   struct hk_texel_windows *w = dev->texel_windows;
   const unsigned blocksize_B = util_format_get_blocksize(format);
   const bool rgb32 = hk_texel_rgb32(format);
   const uint32_t first = w->format_first[format];
   bool ok = true;

   simple_mtx_lock(&w->lock);

   if (first != UINT32_MAX && rgb32) {
      /* Read in bounds at the exact address, see above. */
      *base = address;
      *offset_el = 0;
      *index = w->first_index + first;
      if (!BITSET_TEST(w->written, first)) {
         ok = hk_pack_texel_window(dev, format, w->heap_base,
                                   AGX_TEXTURE_BUFFER_MAX_HEIGHT, tex);
         if (ok) {
            memcpy(hk_descriptor_table_entry(&dev->sampled_images, *index), tex,
                   sizeof(*tex));
            BITSET_SET(w->written, first);
         }
      }
      if (ok)
         memcpy(tex, hk_descriptor_table_entry(&dev->sampled_images, *index),
                sizeof(*tex));
      simple_mtx_unlock(&w->lock);
      return ok;
   }

   if (first != UINT32_MAX && address >= w->heap_base &&
       elements <= HK_TEXEL_WINDOW_STEP_EL) {
      const uint64_t step_B = hk_texel_window_step_B(format);
      const uint64_t window = (address - w->heap_base) / step_B;
      const uint64_t window_base = w->heap_base + window * step_B;
      const uint64_t offset_B = address - window_base;

      if (window < w->format_windows[format] && offset_B % blocksize_B == 0) {
         const uint32_t slot = first + window;
         *index = w->first_index + slot;
         *base = window_base;
         *offset_el = offset_B / blocksize_B;
         assert(*offset_el + elements <= HK_TEXEL_WINDOW_EL);

         if (!BITSET_TEST(w->written, slot)) {
            ok = hk_pack_texel_window(dev, format, window_base,
                                      AGX_TEXTURE_BUFFER_MAX_HEIGHT, tex);
            if (ok) {
               memcpy(hk_descriptor_table_entry(&dev->sampled_images, *index),
                      tex, sizeof(*tex));
               BITSET_SET(w->written, slot);
            }
         }
         if (ok)
            memcpy(tex,
                   hk_descriptor_table_entry(&dev->sampled_images, *index),
                   sizeof(*tex));
         simple_mtx_unlock(&w->lock);
         return ok;
      }
   }

   /* The raw RGB32 path needs its window (above). */
   if (rgb32) {
      simple_mtx_unlock(&w->lock);
      return false;
   }

   /* A texture of its own, as hk_CreateBufferView() would build it. */
   *base = address & ~UINT64_C(0xf);
   *offset_el = (address & 0xf) / blocksize_B;
   const uint64_t height =
      DIV_ROUND_UP(elements + *offset_el, AGX_TEXTURE_BUFFER_WIDTH);
   if ((address & 0xf) % blocksize_B || height > AGX_TEXTURE_BUFFER_MAX_HEIGHT) {
      simple_mtx_unlock(&w->lock);
      return false;
   }

   const struct hk_texel_window_key key = {
      .base = *base,
      .height = MAX2(height, 1),
      .format = format,
   };
   ok = hk_pack_texel_window(dev, format, key.base, key.height, tex);
   struct hash_entry *ent = ok ? _mesa_hash_table_search(w->dedicated, &key)
                               : NULL;
   if (ent) {
      *index = (uintptr_t)ent->data;
   } else if (ok) {
      struct hk_texel_window_key *k =
         ralloc(w->dedicated, struct hk_texel_window_key);
      ok = k && hk_descriptor_table_add(dev, &dev->sampled_images, tex,
                                        sizeof(*tex), index) == VK_SUCCESS;
      if (ok) {
         *k = key;
         _mesa_hash_table_insert(w->dedicated, k, (void *)(uintptr_t)*index);
      }
   }

   simple_mtx_unlock(&w->lock);
   return ok;
}

void
hk_texel_buffer_descriptor(struct hk_device *dev,
                           const VkDescriptorAddressInfoEXT *info,
                           bool storage, void *out)
{
   const unsigned generation = agx_chip_descriptor_generation(dev->dev.chip);
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(generation);
   struct hk_native_texture_descriptor tex = {0};
   struct hk_native_pbe_descriptor pbe = {0};
   /* Zeroed by hk_GetDescriptorEXT() */
   uint8_t *packed = out;

   const enum pipe_format format =
      info && info->address ? hk_format_to_pipe_format(info->format)
                            : PIPE_FORMAT_NONE;
   uint32_t index = 0, offset_el = 0;
   uint64_t base = 0, elements = 0;
   if (format != PIPE_FORMAT_NONE && dev->texel_windows) {
      elements = info->range / util_format_get_blocksize(format);
      if (!hk_texel_window_get(dev, format, info->address, elements, &index,
                               &base, &offset_el, &tex))
         index = 0;
   }

   if (!index) {
      /* The null texel buffer of write_buffer_view_desc(): zero, which the
       * heap resolves to the null texture, with a null PBE. */
      memset(&tex, 0, sizeof(tex));
      hk_set_null_native_pbe(generation, &pbe);
      memcpy(packed + native.texture, &tex, sizeof(tex));
      memcpy(packed + native.pbe, &pbe, sizeof(pbe));
      return;
   }

   if (storage) {
      hk_pack_texel_window_pbe(
         dev, format, base,
         DIV_ROUND_UP(elements + offset_el, AGX_TEXTURE_BUFFER_WIDTH), &pbe);
   }

   memcpy(packed + native.texture, &tex, sizeof(tex));
   memcpy(packed + native.pbe, &pbe, sizeof(pbe));
   memcpy(packed + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET, &index, sizeof(index));
   memcpy(packed + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET, &index, sizeof(index));

   const struct hk_g17p_buffer_texture_metadata metadata = {
      .elements = elements,
      .offset_el = offset_el,
      .address = info->address,
      .rgb32 = hk_texel_rgb32(format),
      /* The buffer behind the address, as hk_CreateBufferView() knows it */
      .sparse = hk_sparse_texel_word_at(dev, info->address,
                                        util_format_get_blocksize(format)),
   };
   static_assert(sizeof(metadata) == 24, "texel-buffer software descriptor ABI");
   memcpy(packed + AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET, &metadata,
          sizeof(metadata));
}
