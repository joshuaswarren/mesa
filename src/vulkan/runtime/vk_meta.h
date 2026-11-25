/*
 * Copyright © 2022 Collabora Ltd
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifndef VK_META_H
#define VK_META_H

#include "vk_internal_exts.h"
#include "vk_limits.h"
#include "vk_object.h"
#include "vk_util.h"

#include "util/simple_mtx.h"

#include "compiler/shader_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

struct hash_table;
struct vk_command_buffer;
struct vk_buffer;
struct vk_device;
struct vk_image;

struct vk_meta_rect {
   uint32_t x0, y0, x1, y1;
   float z;
   uint32_t layer;
};

struct vk_meta_copy_image_properties {
   union {
      struct {
         /* Format to use for the image view of a color aspect.
          * Format must not be compressed and be in the RGB/sRGB colorspace.
          */
         VkFormat view_format;
      } color;

      struct {
         /* Format to use for a specific image plane. */
         VkFormat view_format;
      } plane[3];

      struct {
         struct {
            /* Format to use for the image view of a depth aspect.
             * Format must not be compressed and be in the RGB/sRGB colorspace.
             */
            VkFormat view_format;

            /* Describe the depth/stencil componant layout. Bits in the mask
             * must be consecutive and match the original depth bit size.
             */
            uint8_t component_mask;
         } depth;

         struct {
            /* Format to use for the image view of a stencil aspect.
             * Format must not be compressed and be in the RGB/sRGB colorspace.
             */
            VkFormat view_format;

            /* Describe the depth/stencil componant layout. Bits in the mask
             * must be consecutive and match the original depth bit size.
             */
            uint8_t component_mask;
         } stencil;
      };
   };

   /* Keep depth/stencil aspects on internal color-format views when the
    * driver uses the aspect to select separate backing planes. */
   bool depth_stencil_aspect_views;

   /* When this image is the source of a graphics-path image copy, view only
    * the copied array layers (3D slices as a 2D array), so the copy shader
    * addresses them from layer zero, rather than every layer of the image.
    * For drivers whose views of the image cannot address all of its layers,
    * such as uncompressed views of block-compressed images.
    */
   bool copied_layers_src_view;

   /* Size of the image tile. Used to select the optimal workgroup size. */
   VkExtent3D tile_size;
};

enum vk_meta_buffer_chunk_size_id {
   VK_META_BUFFER_1_BYTE_CHUNK = 0,
   VK_META_BUFFER_2_BYTE_CHUNK,
   VK_META_BUFFER_4_BYTE_CHUNK,
   VK_META_BUFFER_8_BYTE_CHUNK,
   VK_META_BUFFER_16_BYTE_CHUNK,
   VK_META_BUFFER_CHUNK_SIZE_COUNT,
};

struct vk_meta_device {
   struct hash_table *cache;
   simple_mtx_t cache_mtx;

   VkPipelineCache pipeline_cache;

   uint32_t max_bind_map_buffer_size_B;
   bool use_layered_rendering;
   bool use_gs_for_layer;
   bool use_layer_output;
   bool use_stencil_export;
   /* Bind each blit source as a combined image sampler rather than a separate
    * sampled image and sampler.  A driver whose descriptor contents depend on
    * the sampler -- for example one that must pick a filterable format for a
    * scaled image -- cannot make that choice when the two arrive separately.
    */
   bool use_combined_blit_descriptors;
   bool use_masked_rgba8_buffer_unpack;
   /* Match fixed-function RGBA8 4x average resolves that round half up. */
   bool resolve_rgba8_4x_half_up;
   /* A driver without layered rendering may split a 3D graphics copy into
    * one attachment view per destination Z slice.  The fragment Layer value
    * is then relative to that view, so it must not be rebased by the absolute
    * destination Z when addressing the source buffer. */
   bool use_view_relative_3d_layer;

   /* Specialize graphics buffer uploads whose attachment view has exactly
    * one layer. Requires non-layered rendering: fragment Layer is then zero
    * even when the attachment selects an array layer or a 3D slice.
    */
   bool specialize_single_layer_buffer_upload;

   bool use_rect_list_pipeline;

   struct {
      /* Optimal workgroup size for each possible chunk size. This should be
       * chosen to keep things cache-friendly (something big enough to maximize
       * cache hits on executing threads, but small enough to not trash the
       * cache) while keeping GPU utilization high enough to not make copies
       * fast enough.
       */
      uint32_t optimal_wg_size[VK_META_BUFFER_CHUNK_SIZE_COUNT];

      /* Maximum aligned fill chunk, in bytes. Zero retains scalar 32-bit
       * fills; drivers may opt in to 8- or 16-byte vector stores. */
      uint32_t max_fill_chunk_size;
   } buffer_access;

   VkResult (*cmd_bind_map_buffer)(struct vk_command_buffer *cmd,
                                   struct vk_meta_device *meta,
                                   VkBuffer buffer,
                                   void **map_out);

   void (*cmd_draw_rects)(struct vk_command_buffer *cmd,
                          struct vk_meta_device *meta,
                          uint32_t rect_count,
                          const struct vk_meta_rect *rects);

   void (*cmd_draw_volume)(struct vk_command_buffer *cmd,
                           struct vk_meta_device *meta,
                           const struct vk_meta_rect *rect,
                           uint32_t layer_count);
};

static inline VkOffset3D
vk_meta_buffer_to_image_gfx_source_offset(const struct vk_meta_device *meta,
                                          VkImageType image_type,
                                          VkOffset3D destination_offset)
{
   if (meta->use_view_relative_3d_layer && image_type == VK_IMAGE_TYPE_3D)
      destination_offset.z = 0;

   return destination_offset;
}

static inline uint32_t
vk_meta_buffer_access_wg_size(const struct vk_meta_device *meta,
                              uint32_t chunk_size)
{
   assert(util_is_power_of_two_nonzero(chunk_size));
   unsigned idx = ffs(chunk_size) - 1;

   assert(idx < ARRAY_SIZE(meta->buffer_access.optimal_wg_size));
   assert(meta->buffer_access.optimal_wg_size[idx] != 0);

   return meta->buffer_access.optimal_wg_size[idx];
}

VkResult vk_meta_device_init(struct vk_device *device,
                             struct vk_meta_device *meta);
void vk_meta_device_finish(struct vk_device *device,
                           struct vk_meta_device *meta);

/** Keys should start with one of these to ensure uniqueness */
enum vk_meta_object_key_type {
   VK_META_OBJECT_KEY_TYPE_INVALID = 0,

   VK_META_OBJECT_KEY_CLEAR,
   VK_META_OBJECT_KEY_BLIT,
   VK_META_OBJECT_KEY_COPY_BUFFER,
   VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER,
   VK_META_OBJECT_KEY_COPY_BUFFER_TO_IMAGE_CS,
   VK_META_OBJECT_KEY_COPY_BUFFER_TO_IMAGE_GFX,
   VK_META_OBJECT_KEY_COPY_IMAGE_CS,
   VK_META_OBJECT_KEY_COPY_IMAGE_GFX,
   VK_META_OBJECT_KEY_FILL_BUFFER,

   /* BVH build pipelines */
   VK_META_OBJECT_KEY_BVH_PIPELINE_LAYOUT,
   VK_META_OBJECT_KEY_LEAF,
   VK_META_OBJECT_KEY_LEAF_MORTON,
   VK_META_OBJECT_KEY_MORTON,
   VK_META_OBJECT_KEY_PAIR_TRIANGLES,
   VK_META_OBJECT_KEY_ID_PREFIX_SCAN,
   VK_META_OBJECT_KEY_LBVH_MAIN,
   VK_META_OBJECT_KEY_LBVH_GENERATE_IR,
   VK_META_OBJECT_KEY_PLOC,
   VK_META_OBJECT_KEY_HPLOC,
   VK_META_OBJECT_KEY_HPLOC_SHUFFLE,

   /* Should be used as an offset for driver-specific object types. */
   VK_META_OBJECT_KEY_DRIVER_OFFSET = 0x80000000,
};

uint64_t vk_meta_lookup_object(struct vk_meta_device *meta,
                                VkObjectType obj_type,
                                const void *key_data, size_t key_size);

uint64_t vk_meta_cache_object(struct vk_device *device,
                              struct vk_meta_device *meta,
                              const void *key_data, size_t key_size,
                              VkObjectType obj_type,
                              uint64_t handle);

static inline VkDescriptorSetLayout
vk_meta_lookup_descriptor_set_layout(struct vk_meta_device *meta,
                                     const void *key_data, size_t key_size)
{
   return (VkDescriptorSetLayout)
      vk_meta_lookup_object(meta, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                            key_data, key_size);
}

static inline VkPipelineLayout
vk_meta_lookup_pipeline_layout(struct vk_meta_device *meta,
                               const void *key_data, size_t key_size)
{
   return (VkPipelineLayout)
      vk_meta_lookup_object(meta, VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                            key_data, key_size);
}

static inline VkPipeline
vk_meta_lookup_pipeline(struct vk_meta_device *meta,
                        const void *key_data, size_t key_size)
{
   return (VkPipeline)vk_meta_lookup_object(meta, VK_OBJECT_TYPE_PIPELINE,
                                            key_data, key_size);
}

static inline VkSampler
vk_meta_lookup_sampler(struct vk_meta_device *meta,
                       const void *key_data, size_t key_size)
{
   return (VkSampler)vk_meta_lookup_object(meta, VK_OBJECT_TYPE_SAMPLER,
                                           key_data, key_size);
}

struct vk_meta_rendering_info {
   uint32_t view_mask;
   uint32_t samples;
   uint32_t color_attachment_count;
   VkFormat color_attachment_formats[MESA_VK_MAX_COLOR_ATTACHMENTS];
   VkColorComponentFlags color_attachment_write_masks[MESA_VK_MAX_COLOR_ATTACHMENTS];
   VkFormat depth_attachment_format;
   VkFormat stencil_attachment_format;
};

VkResult
vk_meta_create_descriptor_set_layout(struct vk_device *device,
                                     struct vk_meta_device *meta,
                                     const VkDescriptorSetLayoutCreateInfo *info,
                                     const void *key_data, size_t key_size,
                                     VkDescriptorSetLayout *layout_out);

VkResult
vk_meta_create_pipeline_layout(struct vk_device *device,
                               struct vk_meta_device *meta,
                               const VkPipelineLayoutCreateInfo *info,
                               const void *key_data, size_t key_size,
                               VkPipelineLayout *layout_out);

VkResult
vk_meta_get_pipeline_layout(struct vk_device *device,
                            struct vk_meta_device *meta,
                            const VkDescriptorSetLayoutCreateInfo *desc_info,
                            const VkPushConstantRange *push_range,
                            const void *key_data, size_t key_size,
                            VkPipelineLayout *layout_out);

VkResult
vk_meta_create_graphics_pipeline(struct vk_device *device,
                                 struct vk_meta_device *meta,
                                 const VkGraphicsPipelineCreateInfo *info,
                                 const struct vk_meta_rendering_info *render,
                                 const void *key_data, size_t key_size,
                                 VkPipeline *pipeline_out);

VkResult
vk_meta_create_compute_pipeline(struct vk_device *device,
                                struct vk_meta_device *meta,
                                const VkComputePipelineCreateInfo *info,
                                const void *key_data, size_t key_size,
                                VkPipeline *pipeline_out);

VkResult
vk_meta_create_sampler(struct vk_device *device,
                       struct vk_meta_device *meta,
                       const VkSamplerCreateInfo *info,
                       const void *key_data, size_t key_size,
                       VkSampler *sampler_out);

VkResult vk_meta_create_buffer(struct vk_command_buffer *cmd,
                               struct vk_meta_device *meta,
                               const VkBufferCreateInfo *info,
                               VkBuffer *buffer_out);

VkResult vk_meta_create_buffer_view(struct vk_command_buffer *cmd,
                                    struct vk_meta_device *meta,
                                    const VkBufferViewCreateInfo *info,
                                    VkBufferView *buffer_view_out);

VkResult vk_meta_create_image_view(struct vk_command_buffer *cmd,
                                   struct vk_meta_device *meta,
                                   const VkImageViewCreateInfo *info,
                                   VkImageView *image_view_out);

void vk_meta_draw_rects(struct vk_command_buffer *cmd,
                        struct vk_meta_device *meta,
                        uint32_t rect_count,
                        const struct vk_meta_rect *rects);

void vk_meta_draw_volume(struct vk_command_buffer *cmd,
                         struct vk_meta_device *meta,
                         const struct vk_meta_rect *rect,
                         uint32_t layer_count);

void vk_meta_clear_attachments(struct vk_command_buffer *cmd,
                               struct vk_meta_device *meta,
                               const struct vk_meta_rendering_info *render,
                               uint32_t attachment_count,
                               const VkClearAttachment *attachments,
                               uint32_t rect_count,
                               const VkClearRect *rects);

void vk_meta_clear_rendering(struct vk_meta_device *meta,
                             struct vk_command_buffer *cmd,
                             const VkRenderingInfo *pRenderingInfo);

void vk_meta_clear_color_image(struct vk_command_buffer *cmd,
                               struct vk_meta_device *meta,
                               struct vk_image *image,
                               VkImageLayout image_layout,
                               VkFormat format,
                               const VkClearColorValue *color,
                               uint32_t range_count,
                               const VkImageSubresourceRange *ranges);

void vk_meta_clear_depth_stencil_image(struct vk_command_buffer *cmd,
                                       struct vk_meta_device *meta,
                                       struct vk_image *image,
                                       VkImageLayout image_layout,
                                       const VkClearDepthStencilValue *depth_stencil,
                                       uint32_t range_count,
                                       const VkImageSubresourceRange *ranges);

void vk_meta_blit_image(struct vk_command_buffer *cmd,
                        struct vk_meta_device *meta,
                        struct vk_image *src_image,
                        VkFormat src_format,
                        VkImageLayout src_image_layout,
                        struct vk_image *dst_image,
                        VkFormat dst_format,
                        VkImageLayout dst_image_layout,
                        uint32_t region_count,
                        const VkImageBlit2 *regions,
                        VkFilter filter);

void vk_meta_blit_image2(struct vk_command_buffer *cmd,
                         struct vk_meta_device *meta,
                         const VkBlitImageInfo2 *blit);

void vk_meta_resolve_image(struct vk_command_buffer *cmd,
                           struct vk_meta_device *meta,
                           struct vk_image *src_image,
                           VkFormat src_format,
                           VkImageLayout src_image_layout,
                           struct vk_image *dst_image,
                           VkFormat dst_format,
                           VkImageLayout dst_image_layout,
                           uint32_t region_count,
                           const VkImageResolve2 *regions,
                           VkResolveModeFlagBits resolve_mode,
                           VkResolveModeFlagBits stencil_resolve_mode);

void vk_meta_resolve_image2(struct vk_command_buffer *cmd,
                            struct vk_meta_device *meta,
                            const VkResolveImageInfo2 *resolve);

void vk_meta_resolve_rendering(struct vk_command_buffer *cmd,
                               struct vk_meta_device *meta,
                               const VkRenderingInfo *pRenderingInfo);

VkDeviceAddress vk_meta_buffer_address(struct vk_device *device,
                                       VkBuffer buffer, uint64_t offset,
                                       uint64_t range);

void vk_meta_copy_memory(struct vk_command_buffer *cmd,
                         struct vk_meta_device *meta,
                         const VkCopyDeviceMemoryInfoKHR *info);

void vk_meta_copy_buffer(struct vk_command_buffer *cmd,
                         struct vk_meta_device *meta,
                         const VkCopyBufferInfo2 *info);

void vk_meta_copy_image_to_memory(
   struct vk_command_buffer *cmd, struct vk_meta_device *meta,
   const VkCopyDeviceMemoryImageInfoKHR *info,
   const struct vk_meta_copy_image_properties *img_props);

void vk_meta_copy_buffer_addresses(struct vk_command_buffer *cmd,
                                   struct vk_meta_device *meta,
                                   VkDeviceAddress src_addr,
                                   VkDeviceAddress dst_addr,
                                   VkDeviceSize size);

void vk_meta_copy_image_to_buffer(
   struct vk_command_buffer *cmd, struct vk_meta_device *meta,
   const VkCopyImageToBufferInfo2 *info,
   const struct vk_meta_copy_image_properties *img_props);

void vk_meta_copy_memory_to_image(
   struct vk_command_buffer *cmd, struct vk_meta_device *meta,
   const VkCopyDeviceMemoryImageInfoKHR *info,
   const struct vk_meta_copy_image_properties *img_props,
   VkPipelineBindPoint bind_point);

void vk_meta_copy_buffer_to_image(
   struct vk_command_buffer *cmd, struct vk_meta_device *meta,
   const VkCopyBufferToImageInfo2 *info,
   const struct vk_meta_copy_image_properties *img_props,
   VkPipelineBindPoint bind_point);

void vk_meta_copy_image(struct vk_command_buffer *cmd,
                        struct vk_meta_device *meta,
                        const VkCopyImageInfo2 *info,
                        const struct vk_meta_copy_image_properties *src_props,
                        const struct vk_meta_copy_image_properties *dst_props,
                        VkPipelineBindPoint bind_point);

void vk_meta_update_buffer(struct vk_command_buffer *cmd,
                           struct vk_meta_device *meta, VkBuffer buffer,
                           VkDeviceSize offset, VkDeviceSize size,
                           const void *data);

void vk_meta_update_memory(struct vk_command_buffer *cmd,
                           struct vk_meta_device *meta,
                           const VkDeviceAddressRangeKHR* dst_range,
                           const VkAddressCommandFlagsKHR dstFlags,
                           VkDeviceSize dataSize, const void *data);

void vk_meta_fill_memory(struct vk_command_buffer *cmd,
                         struct vk_meta_device *meta,
                         const VkDeviceAddressRangeKHR* dst_range,
                         const VkAddressCommandFlagsKHR dstFlags,
                         const uint32_t data);

/* Fill a raw, naturally aligned device-address range.  This is the internal
 * counterpart of vk_meta_fill_buffer() for driver-owned allocations which do
 * not have a public VkBuffer wrapper. */
void vk_meta_fill_buffer_addr(struct vk_command_buffer *cmd,
                              struct vk_meta_device *meta,
                              VkDeviceAddress address, VkDeviceSize size,
                              uint32_t data);

static inline enum glsl_sampler_dim
vk_image_view_type_to_sampler_dim(VkImageViewType view_type)
{
   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      return GLSL_SAMPLER_DIM_1D;

   case VK_IMAGE_VIEW_TYPE_2D:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      return GLSL_SAMPLER_DIM_2D;

   case VK_IMAGE_VIEW_TYPE_CUBE:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return GLSL_SAMPLER_DIM_CUBE;

   case VK_IMAGE_VIEW_TYPE_3D:
      return GLSL_SAMPLER_DIM_3D;

   default:
      UNREACHABLE("");
   }
}

static inline bool
vk_image_view_type_is_array(VkImageViewType view_type)
{
   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return true;

   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
   case VK_IMAGE_VIEW_TYPE_3D:
   case VK_IMAGE_VIEW_TYPE_CUBE:
      return false;

   default:
      UNREACHABLE("");
   }
}

static inline VkFormat
vk_meta_get_uint_format_for_blk_size(unsigned blk_sz)
{
   switch (blk_sz) {
   case 1:
      return VK_FORMAT_R8_UINT;
   case 2:
      return VK_FORMAT_R8G8_UINT;
   case 3:
      return VK_FORMAT_R8G8B8_UINT;
   case 4:
      return VK_FORMAT_R8G8B8A8_UINT;
   case 6:
      return VK_FORMAT_R16G16B16_UINT;
   case 8:
      return VK_FORMAT_R32G32_UINT;
   case 12:
      return VK_FORMAT_R32G32B32_UINT;
   case 16:
      return VK_FORMAT_R32G32B32A32_UINT;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

#ifdef __cplusplus
}
#endif

#endif /* VK_META_H */
