/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set.h"
#include "asahi/lib/agx_abi.h"

#include "vk_acceleration_structure.h"
#include "asahi/lib/agx_bo.h"
#include "util/half_float.h"
#include "vulkan/vulkan_core.h"

#include "agx_helpers.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_buffer_view.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_image_view.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "asahi/compiler/agx_nir.h"

#include "util/cache_ops.h"
#include "util/log.h"
#include "util/u_debug.h"

#include <inttypes.h>

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline void *
desc_ubo_data(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
              uint32_t *size_out)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   /* An element at or past the end of the set has no bytes: the callers
    * that write refuse it, in every build. */
   uint32_t offset = binding_layout->offset + elem * binding_layout->stride;

   if (size_out != NULL)
      *size_out = offset < set->size ? set->size - offset : 0;

   return (char *)set->mapped_ptr + offset;
}

static void write_desc(struct hk_descriptor_set *set, uint32_t binding,
                       uint32_t elem, const void *desc_data, size_t desc_size);

/*
 * An acceleration structure descriptor is just its BVH address; the software
 * traversal walks it with global loads.  A null handle writes zero, which the
 * traversal treats as an empty structure.
 */
static VkFilter
hk_scaled_view_filter(VkDescriptorType descriptor_type,
                      const struct hk_descriptor_set *set, uint32_t binding,
                      uint32_t elem, const VkDescriptorImageInfo *info)
{
   if (descriptor_type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
      return VK_FILTER_NEAREST;

   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];
   const struct hk_sampler *sampler =
      binding_layout->immutable_samplers
         ? binding_layout->immutable_samplers[elem]
         : (info && info->sampler ? hk_sampler_from_handle(info->sampler)
                                  : NULL);
   if (sampler == NULL)
      return VK_FILTER_NEAREST;

   return (sampler->mag_filter == VK_FILTER_LINEAR ||
           sampler->min_filter == VK_FILTER_LINEAR)
             ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

static void
write_accel_struct_desc(struct hk_descriptor_set *set,
                        VkAccelerationStructureKHR _accel_struct,
                        uint32_t binding, uint32_t elem)
{
   VK_FROM_HANDLE(vk_acceleration_structure, accel_struct, _accel_struct);
   uint64_t addr =
      accel_struct ? vk_acceleration_structure_get_va(accel_struct) : 0;
   if (accel_struct) {
      struct hk_device *dev = container_of(accel_struct->base.device,
                                             struct hk_device, vk);
      if (dev->m5_native_as)
         addr = ALIGN_POT(addr, UINT64_C(1024));
   }

   write_desc(set, binding, elem, &addr, sizeof(addr));
}

/* Make a CPU write to descriptor-set memory visible to the GPU (see
 * hk_descriptor_pool::cpu_cached); the fence is queue submission's. */
static inline void
hk_descriptor_set_publish(const struct hk_descriptor_set *set, void *data,
                          size_t size)
{
   if (set->cpu_cached)
      util_flush_range_no_fence(data, size);
}

static void
write_desc(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
           const void *desc_data, size_t desc_size)
{
   uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   if (desc_size > dst_size)
      return;

   memcpy(dst, desc_data, desc_size);
   hk_descriptor_set_publish(set, dst, desc_size);
}

/* Select the existing heap entry for zero bias, avoiding an extra allocation. */
static uint16_t
unbiased_sampler_index(const struct hk_sampler *sampler, unsigned plane)
{
   const struct hk_rc_sampler *hw = sampler->planes[plane].unbiased;
   return hw ? hw->index : sampler->planes[plane].hw->index;
}

static void
write_g17p_sampler_bias(struct hk_sampled_image_descriptor_sw *sw,
                       const struct hk_sampler *sampler, unsigned plane)
{
   sw->g17p.lod_bias = sampler->mip_lod_bias;
   sw->g17p.unbiased_sampler_index = unbiased_sampler_index(sampler, plane);
   sw->g17p.unbiased_clamp_0_sampler_index =
      sampler->has_border ? unbiased_sampler_index(sampler, 1) : 0;
}

/* The G17P descriptor of a single-plane view, optionally with a single-plane
 * sampler without a custom border: exactly the bytes the general path below
 * writes for that case, built directly instead of through its three-plane
 * staging.  Returns false for every other case. */
/* A custom border follows the view's component mapping. An image without its
 * sampler has no use for its border words, so its first one carries the mapping
 * for the shader: three bits per channel, 0-3 for a source channel, 4 for zero,
 * 5 for one, and the top bit to say it is there. */
static uint32_t
hk_border_swizzle_word(const struct hk_image_view *view)
{
   const VkComponentSwizzle swizzle[4] = {view->vk.swizzle.r, view->vk.swizzle.g,
                                          view->vk.swizzle.b, view->vk.swizzle.a};
   uint32_t word = BITFIELD_BIT(31);

   for (unsigned c = 0; c < 4; ++c) {
      const unsigned code = swizzle[c] == VK_COMPONENT_SWIZZLE_ZERO ? 4
                            : swizzle[c] == VK_COMPONENT_SWIZZLE_ONE ? 5
                            : swizzle[c] - VK_COMPONENT_SWIZZLE_R;
      word |= code << (3 * c);
   }
   return word;
}

static inline bool
write_sampled_image_view_desc_g17p_single(struct hk_descriptor_set *set,
                                          const VkDescriptorImageInfo *info,
                                          uint32_t binding, uint32_t elem,
                                          VkDescriptorType descriptor_type)
{
   static_assert(sizeof(struct hk_sampled_image_descriptor) ==
                    HK_TEXTURE_DESCRIPTOR_AGX3_SIZE +
                       sizeof(struct hk_sampled_image_descriptor_sw),
                 "G17P sampled descriptor is texture then software words");

   if (set->layout->descriptor_generation != 9 || !info)
      return false;

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER) {
      const struct hk_descriptor_set_binding_layout *binding_layout =
         &set->layout->binding[binding];
      const struct hk_sampler *sampler =
         binding_layout->immutable_samplers
            ? binding_layout->immutable_samplers[elem]
            : hk_sampler_from_handle(info->sampler);
      if (sampler->has_border || sampler->plane_count != 1)
         return false;
      assert(sampler->planes[0].hw->index < (1 << 12));

      struct hk_sampled_image_descriptor desc = {
         .sw =
            {
               .sampler = sampler->planes[0].hw->key,
               .sampler_index = hk_sampler_descriptor_index(
                  set->layout, sampler->planes[0].hw->index),
               .clamp_0_sampler_index_or_negative = -1,
               .lod_bias_fp16 = sampler->lod_bias_fp16,
            },
      };
      write_g17p_sampler_bias(&desc.sw, sampler, 0);
      if (!hk_set_null_native_texture(9, &desc.tex))
         return false;
      write_desc(set, binding, elem, &desc, sizeof(desc));
      return true;
   }

   if (info->imageView == VK_NULL_HANDLE)
      return false;

   VK_FROM_HANDLE(hk_image_view, view, info->imageView);
   if (view->plane_count != 1 || view->g17p_uscaled_normalized)
      return false;

   const struct hk_sampler *sampler = NULL;
   if (descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      const struct hk_descriptor_set_binding_layout *binding_layout =
         &set->layout->binding[binding];
      sampler = binding_layout->immutable_samplers
                   ? binding_layout->immutable_samplers[elem]
                   : hk_sampler_from_handle(info->sampler);
      if (sampler->has_border || sampler->plane_count != 1)
         return false;
      assert(sampler->planes[0].hw->index < (1 << 12));
   }

   const bool ia = descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
   struct hk_sampled_image_descriptor desc = {
      .tex = ia ? view->planes[0].ia : view->planes[0].sampled,
      .sw =
         {
            .sampler_index = sampler
                                ? hk_sampler_descriptor_index(
                                     set->layout, sampler->planes[0].hw->index)
                                : hk_sampler_descriptor_index(set->layout, 0),
            .clamp_0_sampler_index_or_negative = -1,
            .lod_bias_fp16 = sampler ? sampler->lod_bias_fp16 : 0,
            .min_lod_fp16 = view->sampled_min_lod_fp16,
            .min_lod_uint16 = view->sampled_min_lod_uint16,
            .scaled_conversion = view->g17p_scaled_conversion,
            .sampled_heap_index = ia ? view->planes[0].input_heap_index
                                     : view->planes[0].sampled_heap_index,
         },
   };
   if (sampler)
      write_g17p_sampler_bias(&desc.sw, sampler, 0);
   else
      desc.sw.border[0] = hk_border_swizzle_word(view);

   write_desc(set, binding, elem, &desc, sizeof(desc));
   return true;
}

static void
write_sampled_image_view_desc(struct hk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem,
                              VkDescriptorType descriptor_type)
{
   if (write_sampled_image_view_desc_g17p_single(set, info, binding, elem,
                                                 descriptor_type))
      return;

   struct hk_sampled_image_descriptor desc[3] = {};
   const unsigned generation = set->layout->descriptor_generation;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(generation);

   /* "Zero" initialize.  The null texture word is filled below, only for
    * the planes that are written out without an image view's word. */
   for (unsigned i = 0; i < 3; ++i) {
      /* G17P native entry zero is the reserved robust sampler. Other Gen9
       * chips retain their biased software index. A zero clamp companion
       * means custom border present; -1 is the no-border sentinel. */
      if (generation == 9) {
         desc[i].sw.sampler_index = hk_sampler_descriptor_index(set->layout, 0);
         desc[i].sw.clamp_0_sampler_index_or_negative = -1;
      }
   }

   uint8_t plane_count = 1;
   /* Planes [0, view_planes) carry the image view's texture word. */
   uint8_t view_planes = 0;
   bool ia = (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);

      plane_count = view->plane_count;
      view_planes = view->plane_count;
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         desc[plane].tex =
            ia ? view->planes[plane].ia : view->planes[plane].sampled;
         desc[plane].sw.sampled_heap_index = ia ?
            view->planes[plane].input_heap_index :
            view->planes[plane].sampled_heap_index;

         desc[plane].sw.min_lod_fp16 = view->sampled_min_lod_fp16;
         desc[plane].sw.min_lod_uint16 = view->sampled_min_lod_uint16;
         if (generation == 9 &&
             descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
            desc[plane].sw.border[0] = hk_border_swizzle_word(view);
         const enum pipe_format logical = hk_format_to_pipe_format(view->vk.format);
         const enum pipe_format integer = hk_scaled_texture_format(logical);
         if (generation == 9 && integer != PIPE_FORMAT_NONE)
            desc[plane].sw.scaled_conversion =
               util_format_is_pure_sint(integer) ? 1 : 2;
         /* Pure tests, cheapest first: almost no format is USCALED. */
         if (generation == 9 &&
             hk_uscaled_normalized_format(logical) != PIPE_FORMAT_NONE &&
             view->planes[plane].filterable_valid &&
             hk_scaled_view_filter(descriptor_type, set, binding, elem, info) ==
                VK_FILTER_LINEAR) {
            desc[plane].tex = view->planes[plane].sampled_filterable;
            desc[plane].sw.sampled_heap_index =
               view->planes[plane].filterable_heap_index;
            const struct util_format_description *fmt =
               util_format_description(logical);
            const VkComponentSwizzle sw[4] = {
               view->vk.swizzle.r, view->vk.swizzle.g,
               view->vk.swizzle.b, view->vk.swizzle.a};
            uint16_t kind = 3 | ((fmt->channel[0].size == 16) << 2) |
                                ((fmt->channel[0].size == 10) << 7);
            for (unsigned c = 0; c < 4; ++c) {
               unsigned comp = sw[c] == VK_COMPONENT_SWIZZLE_IDENTITY ? c :
                  sw[c] >= VK_COMPONENT_SWIZZLE_R ? sw[c] - VK_COMPONENT_SWIZZLE_R : 4;
               if (comp < 4 && fmt->swizzle[comp] < PIPE_SWIZZLE_0) {
                  kind |= BITFIELD_BIT(3 + c);
                  if (fmt->channel[fmt->swizzle[comp]].size == 2)
                     kind |= BITFIELD_BIT(8 + c);
               }
            }
            desc[plane].sw.scaled_conversion = kind;
         }
         /* The normalized descriptor path -- sample a UNORM view and rescale
          * per channel in the shader -- returned the wrong texel for every
          * scaled format it was measured against, so scaled sampling uses the
          * integer descriptor set above. */
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      const struct hk_descriptor_set_binding_layout *binding_layout =
         &set->layout->binding[binding];

      struct hk_sampler *sampler;
      if (binding_layout->immutable_samplers) {
         sampler = binding_layout->immutable_samplers[elem];
      } else {
         sampler = hk_sampler_from_handle(info->sampler);
      }

      if (sampler->has_border)
         assert(plane_count == 1);
      else
         plane_count = MAX2(plane_count, sampler->plane_count);

      for (uint8_t plane = 0; plane < plane_count; plane++) {
         /* We need to replicate the last sampler plane out to all image
          * planes due to sampler table entry limitations. See
          * hk_CreateSampler in hk_sampler.c for more details.
          */
         uint8_t sampler_plane = MIN2(plane, sampler->plane_count - 1);
         assert(sampler->planes[sampler_plane].hw->index < (1 << 12));

         /* Publish the device's sampler-index ABI, including every replicated
          * image plane. G17P consumes the native heap index directly. */
         desc[plane].sw.sampler_index = hk_sampler_descriptor_index(
            set->layout, sampler->planes[sampler_plane].hw->index);
         if (set->layout->descriptor_generation == 9)
            write_g17p_sampler_bias(&desc[plane].sw, sampler, sampler_plane);
         else
            desc[plane].sw.sampler = sampler->planes[sampler_plane].hw->key;
         desc[plane].sw.lod_bias_fp16 = sampler->lod_bias_fp16;
         desc[plane].sw.clamp_0_sampler_index_or_negative = -1;
      }

      if (sampler->has_border) {
         assert(sampler->plane_count == 2);
         desc[0].sw.clamp_0_sampler_index_or_negative =
            hk_sampler_descriptor_index(set->layout,
                                        sampler->planes[1].hw->index) |
            (sampler->border_is_int ? HK_BORDER_IS_INT_BIT : 0);

         assert(desc[0].sw.clamp_0_sampler_index_or_negative >= 0 &&
                "we have a border colour");

         static_assert(sizeof(desc[0].sw.border) ==
                          sizeof(sampler->custom_border),
                       "fixed format");

         memcpy(desc[0].sw.border, sampler->custom_border.uint32,
                sizeof(sampler->custom_border));

         if (generation == 9 &&
             descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
             info->imageView != VK_NULL_HANDLE) {
            VK_FROM_HANDLE(hk_image_view, view, info->imageView);
            const uint32_t word = hk_border_swizzle_word(view);

            for (unsigned c = 0; c < 4; ++c) {
               const unsigned code = (word >> (3 * c)) & 7;
               desc[0].sw.border[c] =
                  code < 4
                     ? sampler->custom_border.uint32[code]
                     : code == 4 ? 0
                     : sampler->border_is_int ? 1 : fui(1.0f);
            }
         }
      }
   }

   for (unsigned plane = view_planes; plane < plane_count; ++plane)
      hk_set_null_native_texture(generation, &desc[plane].tex);

   /* When the semantic struct already matches the native layout, publish it
    * with one bulk copy. This avoids the intermediate packed array while
    * preserving the existing whole-descriptor destination copy. In particular,
    * do not split mapped descriptor BO writes into separate texture/tail copies.
    */
   static_assert(sizeof(desc[0]) == sizeof(desc[0].tex) + sizeof(desc[0].sw),
                 "sampled descriptor has no holes");
   if (native.texture == offsetof(struct hk_sampled_image_descriptor, tex) &&
       native.sampler == offsetof(struct hk_sampled_image_descriptor, sw) &&
       native.stride == sizeof(desc[0])) {
      write_desc(set, binding, elem, desc, native.stride * plane_count);
      return;
   }

   /* Only the written planes are read, so only they need zeroing. */
   uint8_t packed[3 * HK_MAX_DESCRIPTOR_SIZE];
   assert(native.stride * plane_count <= sizeof(packed));
   memset(packed, 0, native.stride * plane_count);
   for (unsigned plane = 0; plane < plane_count; ++plane) {
      uint8_t *dst = packed + plane * native.stride;
      memcpy(dst + native.texture, &desc[plane].tex,
             hk_texture_descriptor_size(generation));
      memcpy(dst + native.sampler, &desc[plane].sw, sizeof(desc[plane].sw));
   }
   write_desc(set, binding, elem, packed, native.stride * plane_count);
}

static void
write_storage_image_view_desc(struct hk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem)
{
   struct hk_storage_image_descriptor desc = {};
   const unsigned generation = set->layout->descriptor_generation;
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(generation);
   bool supported = false;
   bool texture_supported = false;
   uint32_t store_window[4] = {0};

   if (info && info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);

      /* Gate on what the PBE packer accepts (storage_pbe_valid carries every
       * real constraint) plus the shape limits of this fallback; any
       * uncompressed channel layout packs. */
      /* Layered views carry the native array PBE that PACK_IMGWIR-staged
       * writes address (1D and 2D arrays, and cubes, which storage packs as
       * 2D arrays), and 3D views the native 3D PBE. */
      const bool native_basic =
         hk_native_pbe_hardware_supported(generation) &&
         view->plane_count == 1 &&
         view->vk.image->samples == VK_SAMPLE_COUNT_1_BIT &&
         view->planes[0].storage_pbe_valid;
      supported = hk_native_pbe_supported(generation) || native_basic;
      /* A readable native texture does not require a writable PBE. In
       * particular, linear and multisampled storage views have a valid
       * texture descriptor even when the native PBE packer cannot represent
       * their layout.
       * Keep imageLoad/imageSize bound to that view instead of replacing
       * both halves with a null descriptor because stores are unsupported.
       */
      texture_supported = supported ||
         (hk_descriptor_generation_is_agx3(generation) &&
          view->plane_count == 1);
      if (texture_supported) {
         /* Storage images are always single plane */
         assert(view->plane_count == 1);
         uint8_t plane = 0;

         desc.tex = view->planes[plane].ro_storage;
         desc.texture_heap_index =
            view->planes[plane].storage_heap_index;
      }
      if (supported) {
         const uint8_t plane = 0;
         desc.pbe = view->planes[plane].storage;
         desc.atomic_sw = view->planes[plane].storage_atomic_sw;
         memcpy(store_window, view->planes[plane].storage_window,
                sizeof(store_window));
      }
   }

   if (!texture_supported)
      hk_set_null_native_texture(generation, &desc.tex);
   if (!supported)
      hk_set_null_native_pbe(generation, &desc.pbe);

   uint8_t packed[HK_MAX_DESCRIPTOR_SIZE] = {0};
   memcpy(packed + native.texture, &desc.tex,
          hk_texture_descriptor_size(generation));
   memcpy(packed + native.pbe, &desc.pbe, hk_pbe_descriptor_size(generation));
   const uint32_t stride = set->layout->binding[binding].stride;
   if (stride >= AGX_G17P_STORAGE_HEAP_INDEX_OFFSET + sizeof(uint64_t))
      memcpy(packed + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET,
             &desc.texture_heap_index, sizeof(desc.texture_heap_index));
   if (stride >= AGX_G17P_IMAGE_ATOMIC_SW_OFFSET + sizeof(desc.atomic_sw))
      memcpy(packed + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET, &desc.atomic_sw,
             sizeof(desc.atomic_sw));
   /* The M5 store window record; zero (the native store) unless the view's
    * PBE is a store window. */
   static_assert(sizeof(store_window) == HK_M5_STORE_WINDOW_SIZE,
                 "store window record");
   if (stride >= HK_M5_STORE_WINDOW_OFFSET + HK_M5_STORE_WINDOW_SIZE)
      memcpy(packed + HK_M5_STORE_WINDOW_OFFSET, store_window,
             sizeof(store_window));
   /* A descriptor buffer stride (128) exceeds the record; the caller zeroed
    * the rest (hk_GetDescriptorEXT()). */
   write_desc(set, binding, elem, packed, MIN2(stride, sizeof(packed)));
}

static struct hk_buffer_address
get_buffer_address(const VkDescriptorBufferInfo *const info, bool read_only)
{
   VK_FROM_HANDLE(hk_buffer, buffer, info->buffer);
   const struct hk_addr_range addr_range =
      hk_buffer_addr_range(buffer, info->offset, info->range, read_only);
   assert(addr_range.range <= UINT32_MAX);

   /* Only M5 lowers storage loads through this word. Elsewhere the word is the
    * address format's byte offset and must stay zero. */
   uint32_t sparse_map_page = 0;
   if (!read_only && buffer && buffer->sparse_map &&
       agx_chip_g17_m5(
          ((const struct hk_device *)buffer->vk.base.device)->dev.chip)) {
      const uint64_t map_addr = buffer->sparse_map->va->addr;
      assert((map_addr % AIL_PAGESIZE) == 0);
      assert((map_addr / AIL_PAGESIZE) <= UINT32_MAX);
      sparse_map_page = map_addr / AIL_PAGESIZE;
   }

   return (struct hk_buffer_address){
      .base_addr = addr_range.addr,
      .size = addr_range.range,
      .sparse_map_page = sparse_map_page,
   };
}

static void
write_buffer_desc(struct hk_descriptor_set *set,
                  const VkDescriptorBufferInfo *const info, uint32_t binding,
                  uint32_t elem, bool read_only)
{
   const struct hk_buffer_address desc = get_buffer_address(info, read_only);

   /* A null descriptor in a mutable slot reads as null through any type, so it
    * must not keep the previous descriptor's heap index past the buffer
    * record. All zero is the null of every type on G17P. */
   const uint32_t stride = set->layout->binding[binding].stride;
   if (info->buffer == VK_NULL_HANDLE && stride > sizeof(desc) &&
       set->layout->descriptor_generation == 9) {
      uint8_t packed[HK_MAX_DESCRIPTOR_SIZE] = {0};
      memcpy(packed, &desc, sizeof(desc));
      write_desc(set, binding, elem, packed, MIN2(stride, sizeof(packed)));
      return;
   }

   write_desc(set, binding, elem, &desc, sizeof(desc));
}

/* Publish consecutive buffer records with one cache-clean operation. Mutable
 * bindings with larger strides keep per-record publication, so strided writes
 * do not clean unwritten gaps. All batching state stays local to this call.
 */
static void
write_buffer_desc_array(struct hk_descriptor_set *set, const void *data,
                        size_t data_stride, uint32_t binding,
                        uint32_t first_elem, uint32_t count, bool read_only)
{
   if (count <= 1 || !set->cpu_cached ||
       set->layout->binding[binding].stride != sizeof(struct hk_buffer_address)) {
      for (uint32_t i = 0; i < count; ++i) {
         const VkDescriptorBufferInfo *info =
            (const void *)((const uint8_t *)data + i * data_stride);
         write_buffer_desc(set, info, binding, first_elem + i, read_only);
      }
      return;
   }

   uint32_t available;
   uint8_t *dst = desc_ubo_data(set, binding, first_elem, &available);
   if (count > available / sizeof(struct hk_buffer_address))
      return;

   for (uint32_t i = 0; i < count; ++i) {
      const VkDescriptorBufferInfo *info =
         (const void *)((const uint8_t *)data + i * data_stride);
      const struct hk_buffer_address desc = get_buffer_address(info, read_only);
      memcpy(dst + i * sizeof(desc), &desc, sizeof(desc));
   }
   hk_descriptor_set_publish(set, dst,
                             (size_t)count * sizeof(struct hk_buffer_address));
}

static void
write_dynamic_buffer_desc(struct hk_descriptor_set *set,
                          const VkDescriptorBufferInfo *const info,
                          uint32_t binding, uint32_t elem, bool read_only)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   struct hk_buffer_address *desc =
      &set->dynamic_buffers[binding_layout->dynamic_buffer_index + elem];

   *desc = get_buffer_address(info, read_only);
}

static void
write_buffer_view_desc(struct hk_descriptor_set *set,
                       const VkBufferView bufferView, uint32_t binding,
                       uint32_t elem, bool storage)
{
   struct hk_storage_image_descriptor desc = {};
   const unsigned generation = set->layout->descriptor_generation;
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(generation);
   uint32_t heap_index = 0;

   /* Native buffer-image reads use the same texture heap and software
    * metadata as uniform texel buffers. A missing writable PBE must not
    * replace their readable texture with the null descriptor. */
   if ((!storage || hk_descriptor_generation_is_agx3(generation) ||
        hk_native_pbe_supported(generation)) &&
       bufferView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_buffer_view, view, bufferView);

      desc.tex = view->tex;
      if (storage)
         desc.pbe = view->pbe;
      heap_index = view->sampled_heap_index;
   } else {
      /* A mutable slot can be read as any type, and vkd3d-proton nulls it with
       * a null texel buffer. G17P finds a texture by heap index and index 0 is
       * the null texture, so keep the bytes zero: read as a storage or uniform
       * buffer they are then the null buffer, not a size taken from a packed
       * texture word. */
      if (generation != 9)
         hk_set_null_native_texture(generation, &desc.tex);
      hk_set_null_native_pbe(generation, &desc.pbe);
   }

   uint8_t packed[HK_MAX_DESCRIPTOR_SIZE] = {0};
   memcpy(packed + native.texture, &desc.tex,
          hk_texture_descriptor_size(generation));
   memcpy(packed + native.pbe, &desc.pbe, hk_pbe_descriptor_size(generation));
   /* G17P reaches a texture through the device heap; the descriptor carries
    * only the index, which hk_descriptor_stride_align_for_type() reserved the
    * eight extra bytes for.  Leaving it zero resolves to the pinned null
    * texture, so every texel buffer read black.
    *
    * Both offsets are written because both readers exist in
    * agx_nir_lower_g17p_combined_sampler.c and a buffer view has one texture
    * for both: a texel fetch (nir_tex, the lowered 2D txf) takes
    * AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET, while an imageLoad on an imageBuffer
    * takes AGX_G17P_STORAGE_HEAP_INDEX_OFFSET.  The two live in the reserved
    * eight bytes at 64 and 68 and do not overlap. */
   const uint32_t stride = set->layout->binding[binding].stride;
   if (heap_index &&
       stride >= AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET + sizeof(heap_index)) {
      memcpy(packed + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET, &heap_index,
             sizeof(heap_index));
      memcpy(packed + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET, &heap_index,
             sizeof(heap_index));
   }
   if (heap_index &&
       stride >= AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE) {
      VK_FROM_HANDLE(hk_buffer_view, view, bufferView);
      static_assert(sizeof(view->buffer_metadata) == 24,
                    "texel-buffer software descriptor ABI");
      memcpy(packed + AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET,
             &view->buffer_metadata, sizeof(view->buffer_metadata));
   }
   /* Write the binding's whole stride, not just the texture/PBE pair: the
    * reserved heap-index bytes live past native.stride, so a native.stride
    * write would drop them.  write_storage_image_view_desc() already does
    * this; texel buffers were the one caller that did not, which is the
    * second half of why their index was always zero. */
   write_desc(set, binding, elem, packed, MIN2(stride, sizeof(packed)));
}

static void
write_inline_uniform_data(struct hk_descriptor_set *set,
                          const VkWriteDescriptorSetInlineUniformBlock *info,
                          uint32_t binding, uint32_t offset)
{
   assert(set->layout->binding[binding].stride == 1);
   write_desc(set, binding, offset, info->pData, info->dataSize);
}

static ATTRIBUTE_NOINLINE void
hk_copy_descriptor_sets(uint32_t descriptorCopyCount,
                        const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      VK_FROM_HANDLE(hk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(hk_descriptor_set, dst, copy->dstSet);

      const struct hk_descriptor_set_binding_layout *src_binding_layout =
         &src->layout->binding[copy->srcBinding];
      const struct hk_descriptor_set_binding_layout *dst_binding_layout =
         &dst->layout->binding[copy->dstBinding];

      /* Equal strides make the existing per-record copies contiguous. Vulkan
       * requires source and destination descriptor ranges not to overlap,
       * including when they belong to the same descriptor set.
       */
      if (dst_binding_layout->stride > 0 &&
          dst_binding_layout->stride == src_binding_layout->stride) {
         uint32_t dst_max_size, src_max_size;
         void *dst_map = desc_ubo_data(dst, copy->dstBinding,
                                      copy->dstArrayElement, &dst_max_size);
         const void *src_map = desc_ubo_data(src, copy->srcBinding,
                                            copy->srcArrayElement, &src_max_size);
         const size_t size =
            (size_t)copy->descriptorCount * dst_binding_layout->stride;
         /* A range past the end of either set is not copied. */
         if (size <= dst_max_size && size <= src_max_size) {
            if (size == sizeof(struct hk_buffer_address))
               memcpy(dst_map, src_map, sizeof(struct hk_buffer_address));
            else
               memcpy(dst_map, src_map, size);
            hk_descriptor_set_publish(dst, dst_map, size);
         }
      } else if (dst_binding_layout->stride > 0 && src_binding_layout->stride > 0) {
         for (uint32_t j = 0; j < copy->descriptorCount; j++) {
            uint32_t dst_max_size, src_max_size;
            void *dst_map = desc_ubo_data(
               dst, copy->dstBinding, copy->dstArrayElement + j, &dst_max_size);
            const void *src_map = desc_ubo_data(
               src, copy->srcBinding, copy->srcArrayElement + j, &src_max_size);
            const uint32_t copy_size =
               MIN2(dst_binding_layout->stride, src_binding_layout->stride);
            if (copy_size > dst_max_size || copy_size > src_max_size)
               break;

            memcpy(dst_map, src_map, copy_size);
            hk_descriptor_set_publish(dst, dst_map, copy_size);
         }
      }

      if (vk_descriptor_type_is_dynamic(src_binding_layout->type)) {
         const uint32_t dst_dyn_start =
            dst_binding_layout->dynamic_buffer_index + copy->dstArrayElement;
         const uint32_t src_dyn_start =
            src_binding_layout->dynamic_buffer_index + copy->srcArrayElement;
         typed_memcpy(&dst->dynamic_buffers[dst_dyn_start],
                      &src->dynamic_buffers[src_dyn_start],
                      copy->descriptorCount);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                        const VkWriteDescriptorSet *pDescriptorWrites,
                        uint32_t descriptorCopyCount,
                        const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t w = 0; w < descriptorWriteCount; w++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[w];
      VK_FROM_HANDLE(hk_descriptor_set, set, write->dstSet);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(
               set, write->pTexelBufferView[j], write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (likely(write->descriptorCount == 1)) {
            write_buffer_desc(set, write->pBufferInfo, write->dstBinding,
                              write->dstArrayElement,
                              write->descriptorType ==
                                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         } else {
            write_buffer_desc_array(
               set, write->pBufferInfo, sizeof(*write->pBufferInfo),
               write->dstBinding, write->dstArrayElement, write->descriptorCount,
               write->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_dynamic_buffer_desc(
               set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType ==
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
         }
         break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
         const VkWriteDescriptorSetAccelerationStructureKHR *write_accel =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_accel_struct_desc(set,
                                    write_accel->pAccelerationStructures[j],
                                    write->dstBinding,
                                    write->dstArrayElement + j);
         }
         break;
      }

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
         const VkWriteDescriptorSetInlineUniformBlock *write_inline =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
         assert(write_inline->dataSize == write->descriptorCount);
         write_inline_uniform_data(set, write_inline, write->dstBinding,
                                   write->dstArrayElement);
         break;
      }

      default:
         break;
      }
   }

   if (descriptorCopyCount)
      hk_copy_descriptor_sets(descriptorCopyCount, pDescriptorCopies);
}

/* Points a push descriptor set at layout for an update, and returns false,
 * leaving it empty, if the layout does not fit the storage. Every push layout
 * within the limits does (HK_PUSH_DESCRIPTOR_SET_SIZE): this keeps one that
 * does not from being written, saved or uploaded past it, and says so.
 */
static bool
hk_push_descriptor_set_begin(struct hk_push_descriptor_set *push_set,
                             struct hk_descriptor_set_layout *layout,
                             struct hk_descriptor_set *set)
{
   memcpy(push_set->layout_hash, layout->vk.blake3,
          sizeof(push_set->layout_hash));
   push_set->layout_flags = layout->vk.flags;
   if (layout->non_variable_descriptor_buffer_size > sizeof(push_set->data)) {
      mesa_loge("push descriptor layout of %u bytes refused: the push set "
                "holds %zu",
                layout->non_variable_descriptor_buffer_size,
                sizeof(push_set->data));
      push_set->size = 0;
      return false;
   }

   push_set->size = layout->non_variable_descriptor_buffer_size;
   *set = (struct hk_descriptor_set){
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };
   return true;
}

bool
hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                              struct hk_descriptor_set_layout *layout,
                              uint32_t write_count,
                              const VkWriteDescriptorSet *writes)
{
   struct hk_descriptor_set set;
   if (!hk_push_descriptor_set_begin(push_set, layout, &set))
      return false;

   for (uint32_t w = 0; w < write_count; w++) {
      const VkWriteDescriptorSet *write = &writes[w];

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               &set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(&set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(
               &set, write->pTexelBufferView[j], write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(
               &set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
         const VkWriteDescriptorSetAccelerationStructureKHR *write_accel =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_accel_struct_desc(&set,
                                    write_accel->pAccelerationStructures[j],
                                    write->dstBinding,
                                    write->dstArrayElement + j);
         }
         break;
      }

      default:
         break;
      }
   }

   return true;
}

static void hk_descriptor_pool_free(struct hk_descriptor_pool *pool,
                                    uint64_t addr, uint64_t size);

static void
hk_descriptor_set_destroy(struct hk_device *dev,
                          struct hk_descriptor_pool *pool,
                          struct hk_descriptor_set *set)
{
   list_del(&set->link);
   if (set->size > 0)
      hk_descriptor_pool_free(pool, set->addr, set->size);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);

   vk_object_free(&dev->vk, NULL, set);
}

static void
hk_destroy_descriptor_pool(struct hk_device *dev,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct hk_descriptor_set, set, &pool->sets, link)
      hk_descriptor_set_destroy(dev, pool, set);

   util_vma_heap_finish(&pool->heap);
   agx_bo_unreference(&dev->dev, pool->bo);

   vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateDescriptorPool(VkDevice _device,
                        const VkDescriptorPoolCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_descriptor_pool *pool;

   pool = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   list_inithead(&pool->sets);

   /* The heap must be a valid empty heap before anything can fail: the error
    * paths below run hk_destroy_descriptor_pool(), which calls
    * util_vma_heap_finish() and walks heap->holes.  On a vk_object_zalloc'd
    * pool that list head is all zeroes, so walking it dereferences NULL and
    * the application takes a SIGSEGV instead of the VkResult it asked for.
    * Firefox hit exactly this on G17P: one failed "Descriptor pool" BO -- the
    * VM was already carrying a sticky error from an earlier submission --
    * turned an orderly VK_ERROR_OUT_OF_DEVICE_MEMORY into a browser crash
    * inside hk_destroy_descriptor_pool().
    */
   util_vma_heap_init(&pool->heap, 0, 0);

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t max_align = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_type(pdev, pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   uint64_t bo_size = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_type(pdev, pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      bo_size +=
         MAX2(stride, max_align) * pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   /* Individual descriptor sets are aligned to the min UBO alignment to
    * ensure that we don't end up with unaligned data access in any shaders.
    * This means that each descriptor buffer allocated may burn up to 16B of
    * extra space to get the right alignment.  (Technically, it's at most 28B
    * because we're always going to start at least 4B aligned but we're being
    * conservative here.)  Allocate enough extra space that we can chop it
    * into maxSets pieces and align each one of them to 32B.
    */
   bo_size += HK_MIN_UBO_ALIGNMENT * pCreateInfo->maxSets;

   if (bo_size) {
      /* Descriptor-set memory is never fully written: a binding the
       * application declares and never updates -- PARTIALLY_BOUND, an
       * unwritten array element, the deliberately over-allocated
       * variable-count tail -- is still dereferenced by the shader.  On every
       * other chip an unmapped or nonsense address read out of those bytes is
       * absorbed by hardware soft fault; G17P halts with a GMMU reason-3
       * fault and takes the device with it, so the bytes have to be
       * predictable.  A recycled cache BO carries whatever the previous
       * owner (a command slab, an upload pool, a geometry heap) left behind,
       * which is exactly the shape of a wild descriptor address.
       */
      /* G17P: write descriptors through the CPU cache and clean what is
       * written (hk_descriptor_set_publish()); write-combined stores of
       * every descriptor and of the per-allocation zeroing above stalled
       * the recording core. */
      pool->cpu_cached = hk_g17p_cached_cmd_memory(&dev->dev);
      const enum agx_bo_flags pool_flags =
         agx_native_graphics(&dev->dev)
            ? AGX_BO_ZERO_ON_REUSE |
                 (pool->cpu_cached ? AGX_BO_WRITEBACK : 0)
            : 0;
      pool->bo =
         agx_bo_create(&dev->dev, bo_size, 0, pool_flags, "Descriptor pool");
      if (!pool->bo) {
         hk_destroy_descriptor_pool(dev, pAllocator, pool);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      pool->mapped_ptr = agx_bo_map(pool->bo);

      /* The BO may be larger thanks to GPU page alignment.  We may as well
       * make that extra space available to the client.
       */
      assert(pool->bo->size >= bo_size);
      /* Re-initialising is safe: the heap above was created empty, so there
       * are no holes to leak here. */
      util_vma_heap_init(&pool->heap, pool->bo->va->addr, pool->bo->size);
   }

   *pDescriptorPool = hk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static VkResult
hk_descriptor_pool_alloc(struct hk_descriptor_pool *pool, uint64_t size,
                         uint64_t alignment, uint64_t *addr_out, void **map_out)
{
   assert(size > 0);
   uint64_t addr = util_vma_heap_alloc(&pool->heap, size, alignment);
   if (addr == 0)
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   assert(addr >= pool->bo->va->addr);
   assert(addr + size <= pool->bo->va->addr + pool->bo->size);
   uint64_t offset = addr - pool->bo->va->addr;

   *addr_out = addr;
   *map_out = pool->mapped_ptr + offset;

   return VK_SUCCESS;
}

static void
hk_descriptor_pool_free(struct hk_descriptor_pool *pool, uint64_t addr,
                        uint64_t size)
{
   assert(size > 0);
   assert(addr >= pool->bo->va->addr);
   assert(addr + size <= pool->bo->va->addr + pool->bo->size);
   util_vma_heap_free(&pool->heap, addr, size);
}

static VkResult
hk_descriptor_set_create(struct hk_device *dev, struct hk_descriptor_pool *pool,
                         struct hk_descriptor_set_layout *layout,
                         uint32_t variable_count,
                         struct hk_descriptor_set **out_set)
{
   struct hk_descriptor_set *set;
   VkResult result;

   uint32_t mem_size =
      sizeof(struct hk_descriptor_set) +
      layout->vk.dynamic_descriptor_count * sizeof(struct hk_buffer_address);

   set =
      vk_object_zalloc(&dev->vk, NULL, mem_size, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   set->size = layout->non_variable_descriptor_buffer_size;

   if (layout->binding_count > 0 &&
       (layout->binding[layout->binding_count - 1].flags &
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
      uint32_t stride = layout->binding[layout->binding_count - 1].stride;

      /* Off by one so we don't underallocate the end. Otherwise vkd3d-proton
       * descriptor-performance underallocates.
       */
      set->size += stride * (variable_count + 1);
   }

   set->size = align64(set->size, HK_MIN_UBO_ALIGNMENT);

   if (set->size > 0) {
      result = hk_descriptor_pool_alloc(pool, set->size, HK_MIN_UBO_ALIGNMENT,
                                        &set->addr, &set->mapped_ptr);
      if (result != VK_SUCCESS) {
         vk_object_free(&dev->vk, NULL, set);
         return result;
      }

      /* Zeroing the pool BO on reuse is not enough on its own: a pool
       * outlives its sets, and vkFreeDescriptorSets/vkResetDescriptorPool
       * return the range to the VMA heap with the destroyed set's bytes
       * intact.  The next set allocated over that range would otherwise
       * inherit a live-looking texture, PBE or buffer address belonging to an
       * object that has since been destroyed.  All-zero reads as the reserved
       * null texture / heap index 0, which is bound and safe.
       */
      set->cpu_cached = pool->cpu_cached;
      if (agx_native_graphics(&dev->dev)) {
         memset(set->mapped_ptr, 0, set->size);
         hk_descriptor_set_publish(set, set->mapped_ptr, set->size);
      }
   }

   vk_descriptor_set_layout_ref(&layout->vk);
   set->layout = layout;

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      if (layout->binding[b].type != VK_DESCRIPTOR_TYPE_SAMPLER &&
          layout->binding[b].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         continue;

      if (layout->binding[b].immutable_samplers == NULL)
         continue;

      uint32_t array_size = layout->binding[b].array_size;
      if (layout->binding[b].flags &
          VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         array_size = variable_count;

      for (uint32_t j = 0; j < array_size; j++) {
         write_sampled_image_view_desc(set, NULL, b, j,
                                       layout->binding[b].type);
      }
   }

   list_addtail(&set->link, &pool->sets);
   *out_set = set;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_AllocateDescriptorSets(VkDevice device,
                          const VkDescriptorSetAllocateInfo *pAllocateInfo,
                          VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   struct hk_descriptor_set *set = NULL;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_desc_count =
      vk_find_struct_const(
         pAllocateInfo->pNext,
         DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   /* allocate a set of buffers for each shader to contain descriptors */
   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(hk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      /* If descriptorSetCount is zero or this structure is not included in
       * the pNext chain, then the variable lengths are considered to be zero.
       */
      const uint32_t variable_count =
         var_desc_count && var_desc_count->descriptorSetCount > 0
            ? var_desc_count->pDescriptorCounts[i]
            : 0;

      result =
         hk_descriptor_set_create(dev, pool, layout, variable_count, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = hk_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      hk_FreeDescriptorSets(device, pAllocateInfo->descriptorPool, i,
                            pDescriptorSets);
      for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
         pDescriptorSets[i] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                      uint32_t descriptorSetCount,
                      const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, descriptorPool);

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      VK_FROM_HANDLE(hk_descriptor_set, set, pDescriptorSets[i]);

      if (set)
         hk_descriptor_set_destroy(dev, pool, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyDescriptorPool(VkDevice device, VkDescriptorPool _pool,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, _pool);

   if (!_pool)
      return;

   hk_destroy_descriptor_pool(dev, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                       VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, descriptorPool);

   list_for_each_entry_safe(struct hk_descriptor_set, set, &pool->sets, link)
      hk_descriptor_set_destroy(dev, pool, set);

   return VK_SUCCESS;
}

static void
hk_descriptor_set_write_template(
   struct hk_descriptor_set *set,
   const struct vk_descriptor_update_template *template, const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];

      switch (entry->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_sampled_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j,
                                          entry->type);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_storage_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkBufferView *bview =
               data + entry->offset + j * entry->stride;

            write_buffer_view_desc(
               set, *bview, entry->binding, entry->array_element + j,
               entry->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (likely(entry->array_count == 1)) {
            write_buffer_desc(
               set, data + entry->offset, entry->binding, entry->array_element,
               entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         } else {
            write_buffer_desc_array(
               set, data + entry->offset, entry->stride, entry->binding,
               entry->array_element, entry->array_count,
               entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_dynamic_buffer_desc(
               set, info, entry->binding, entry->array_element + j,
               entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
         }
         break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkAccelerationStructureKHR *accel_struct =
               data + entry->offset + j * entry->stride;

            write_accel_struct_desc(set, *accel_struct, entry->binding,
                                    entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         write_desc(set, entry->binding, entry->array_element,
                    data + entry->offset, entry->array_count);
         break;

      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateDescriptorSetWithTemplate(
   VkDevice device, VkDescriptorSet descriptorSet,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData)
{
   VK_FROM_HANDLE(hk_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  descriptorUpdateTemplate);

   hk_descriptor_set_write_template(set, template, pData);
}

bool
hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data)
{
   struct hk_descriptor_set tmp_set;
   if (!hk_push_descriptor_set_begin(push_set, layout, &tmp_set))
      return false;

   hk_descriptor_set_write_template(&tmp_set, template, data);
   return true;
}

/*
 * VK_EXT_descriptor_buffer. A descriptor written here is byte for byte what
 * vkUpdateDescriptorSets() writes for the same resource, so the shaders read
 * descriptor buffers exactly like pool sets: the writers above run on a
 * one-binding layout whose stride is the descriptor size of the type
 * (hk_descriptor_buffer_size()), over the caller's memory.
 */
VKAPI_ATTR void VKAPI_CALL
hk_GetDescriptorEXT(VkDevice _device,
                    const VkDescriptorGetInfoEXT *pDescriptorInfo,
                    size_t dataSize, void *pDescriptor)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   const VkDescriptorType type = pDescriptorInfo->type;

   union {
      struct hk_descriptor_set_layout layout;
      uint8_t storage[sizeof(struct hk_descriptor_set_layout) +
                      sizeof(struct hk_descriptor_set_binding_layout)];
   } get;
   memset(&get, 0, sizeof(get));
   struct hk_descriptor_set_layout *layout = &get.layout;
   layout->descriptor_generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   layout->native_sampler_indices =
      agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip);
   layout->binding_count = 1;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = type,
      .array_size = 1,
      .stride = hk_descriptor_buffer_size(&dev->dev, type),
   };
   assert(layout->binding[0].stride <= dataSize);
   /* The records are smaller than the power-of-two size reported for them,
    * so give the padding a defined value. */
   memset(pDescriptor, 0, dataSize);

   struct hk_descriptor_set set = {
      .layout = layout,
      .mapped_ptr = pDescriptor,
      .size = dataSize,
   };

   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER: {
      const VkDescriptorImageInfo info = {
         .sampler = *pDescriptorInfo->data.pSampler,
      };
      write_sampled_image_view_desc(&set, &info, 0, 0, type);
      break;
   }

   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      /* A YCbCr sampler takes one descriptor per plane:
       * combinedImageSamplerDescriptorCount of them. */
      write_sampled_image_view_desc(
         &set, pDescriptorInfo->data.pCombinedImageSampler, 0, 0, type);
      break;

   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      write_sampled_image_view_desc(&set, pDescriptorInfo->data.pSampledImage,
                                    0, 0, type);
      break;

   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      write_sampled_image_view_desc(
         &set, pDescriptorInfo->data.pInputAttachmentImage, 0, 0, type);
      break;

   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      write_storage_image_view_desc(&set, pDescriptorInfo->data.pStorageImage,
                                    0, 0);
      break;

   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      hk_texel_buffer_descriptor(dev, pDescriptorInfo->data.pUniformTexelBuffer,
                                 false, pDescriptor);
      break;

   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      hk_texel_buffer_descriptor(dev, pDescriptorInfo->data.pStorageTexelBuffer,
                                 true, pDescriptor);
      break;

   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      const VkDescriptorAddressInfoEXT *info =
         type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            ? pDescriptorInfo->data.pUniformBuffer
            : pDescriptorInfo->data.pStorageBuffer;
      /* A null buffer is based at the zero page (hk_buffer_addr_range()). */
      struct hk_buffer_address desc = {
         .base_addr = AGX_ZERO_PAGE_ADDRESS,
      };
      if (info != NULL && info->address != 0 && info->range != 0) {
         assert(info->range <= UINT32_MAX);
         desc = (struct hk_buffer_address){
            .base_addr = info->address,
            .size = MIN2(info->range, UINT32_MAX),
         };
      }
      write_desc(&set, 0, 0, &desc, sizeof(desc));
      break;
   }

   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
      uint64_t addr = pDescriptorInfo->data.accelerationStructure;
      if (addr && dev->m5_native_as)
         addr = ALIGN_POT(addr, UINT64_C(1024));
      write_desc(&set, 0, 0, &addr, sizeof(addr));
      break;
   }

   default:
      UNREACHABLE("Invalid descriptor type for vkGetDescriptorEXT");
   }
}
