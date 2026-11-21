/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set_layout.h"
#include "asahi/lib/agx_abi.h"

#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_native_descriptor.h"

#include "vk_pipeline_layout.h"

static bool
binding_has_immutable_samplers(const VkDescriptorSetLayoutBinding *binding)
{
   switch (binding->descriptorType) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return binding->pImmutableSamplers != NULL;

   default:
      return false;
   }
}

void
hk_descriptor_stride_align_for_type(
   const struct hk_physical_device *pdev, VkDescriptorType type,
   const VkMutableDescriptorTypeListEXT *type_list, uint32_t *stride,
   uint32_t *alignment)
{
   const unsigned generation = agx_chip_descriptor_generation(pdev->dev.chip);
   const bool agx2 = hk_descriptor_generation_is_agx2(generation);

   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      /* TODO: How do samplers work? */
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      *stride = hk_sampled_image_descriptor_layout(generation).stride;
      /* Preserve the upstream AGX2 descriptor-set ABI. The native AGX3
       * layout uses a smaller alignment, which must not change G13/G14
       * binding offsets or the alignment inferred by descriptor lowering.
       */
      *alignment = agx2 ? *stride : 8;
      break;

   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      *stride = hk_storage_image_descriptor_layout(
                   agx_chip_descriptor_generation(pdev->dev.chip)).stride;
      /* G18G uses the same 96-byte record: without it the image-atomic
       * sideband (+72) did not fit the bare 64-byte texture/PBE pair, the
       * shader read zero tile geometry there, and atomics addressed the
       * twiddled image row-major. */
      if (agx_chip_agx3_path(pdev->dev.chip)) {
         *stride += HK_G17P_STORAGE_SW_SIZE;
         /* imgwci addresses PBE records in a heap, and stores treat the
          * descriptor set as that heap. The PBE descriptor starts 32 bytes
          * into the combined texture/PBE record, so both the binding offset
          * and every array stride must keep it on a record boundary. Without
          * this, an SSBO followed by a storage image places the PBE at +48
          * and stores silently target the wrong block. */
         *alignment = HK_G17P_STORAGE_ALIGNMENT;
         *stride = HK_G17P_STORAGE_STRIDE;
      } else {
         *alignment = 8;
      }
      break;

   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      *stride = hk_storage_image_descriptor_layout(
                   agx_chip_descriptor_generation(pdev->dev.chip)).stride;
      /* Native texture/PBE records occupy 64 bytes, followed by the two
       * heap indices and 24 bytes of buffer size/offset/address metadata.
       * The shader must never decode AGX2 software fields from Gen9 bytes.
       */
      if (agx_chip_native_or_m5(pdev->dev.chip)) {
         *stride = AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE;
         /* Texel-buffer stores address the PBE through the set's PBE heap,
          * like storage images. */
         *alignment = HK_G17P_STORAGE_ALIGNMENT;
      } else {
         *alignment = 8;
      }
      break;

   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      *stride = *alignment = sizeof(struct hk_buffer_address);
      break;

   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      *stride = *alignment = 0; /* These don't take up buffer space */
      break;

   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      /* An acceleration structure descriptor is nothing but the address of
       * its BVH: the traversal walks it with plain global loads, so there is
       * no hardware descriptor and no bounds information to carry. */
      *stride = *alignment = sizeof(uint64_t);
      break;

   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      *stride = 1; /* Array size is bytes */
      *alignment = HK_MIN_UBO_ALIGNMENT;
      break;

   case VK_DESCRIPTOR_TYPE_MUTABLE_EXT:
      *stride = *alignment = 0;
      if (type_list == NULL || type_list->descriptorTypeCount == 0) {
         if (agx2) {
            *stride = *alignment =
               hk_sampled_image_descriptor_layout(generation).stride;
         } else {
            *stride = HK_MAX_DESCRIPTOR_SIZE;
            *alignment = agx_chip_native_or_m5(pdev->dev.chip)
                            ? HK_G17P_STORAGE_ALIGNMENT
                            : 8;
         }
      }
      for (unsigned i = 0; type_list && i < type_list->descriptorTypeCount;
           i++) {
         /* This shouldn't recurse */
         assert(type_list->pDescriptorTypes[i] !=
                VK_DESCRIPTOR_TYPE_MUTABLE_EXT);
         uint32_t desc_stride, desc_align;
         hk_descriptor_stride_align_for_type(pdev,
                                             type_list->pDescriptorTypes[i],
                                             NULL, &desc_stride, &desc_align);
         *stride = MAX2(*stride, desc_stride);
         *alignment = MAX2(*alignment, desc_align);
      }
      *stride = align(*stride, *alignment);
      break;

   default:
      UNREACHABLE("Invalid descriptor type");
   }

   assert(*stride <= HK_MAX_DESCRIPTOR_SIZE);
}

/* Applications place descriptor buffer arrays at the reported descriptor
 * sizes and copy descriptors of those sizes themselves. vkd3d-proton copies
 * only 4, 8, 16, 32, 48, 64 and 128 bytes: for the G17P 72- and 96-byte
 * records it has no copy, never maps its sampler and resource heaps, and
 * writes the first descriptor through NULL. So descriptor buffers take every
 * record at the next power of two, which the reported sizes and the strides
 * of descriptor buffer layouts both use. A push layout's memory is the
 * driver's own, so it keeps the packed stride.
 */
static uint32_t
hk_descriptor_buffer_stride(uint32_t stride)
{
   return stride > 1 ? util_next_power_of_two(stride) : stride;
}

void
hk_descriptor_stride_align_for_layout(
   const struct hk_physical_device *pdev, VkDescriptorSetLayoutCreateFlags flags,
   VkDescriptorType type, const VkMutableDescriptorTypeListEXT *type_list,
   uint32_t *stride, uint32_t *alignment)
{
   hk_descriptor_stride_align_for_type(pdev, type, type_list, stride,
                                       alignment);

   /* In a descriptor buffer, element i of a mutable binding sits at i times
    * the largest descriptor size of its types (VK_EXT_descriptor_buffer),
    * with no padding to their largest alignment. The alignment the hardware
    * needs, the 32-byte PBE record of storage images and texel buffers,
    * belongs to types of the largest size, so every element keeps it.
    */
   if ((flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT) &&
       type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT && type_list != NULL &&
       type_list->descriptorTypeCount > 0) {
      uint32_t max_stride = 0;
      for (unsigned i = 0; i < type_list->descriptorTypeCount; i++) {
         uint32_t desc_stride, desc_align;
         hk_descriptor_stride_align_for_type(pdev,
                                             type_list->pDescriptorTypes[i],
                                             NULL, &desc_stride, &desc_align);
         max_stride = MAX2(max_stride, desc_stride);
      }
      *stride = max_stride;
   }

   if ((flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT) &&
       !(flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR))
      *stride = hk_descriptor_buffer_stride(*stride);
}

uint32_t
hk_descriptor_buffer_size(const struct agx_device *dev, VkDescriptorType type)
{
   const struct hk_physical_device *pdev =
      container_of(dev, struct hk_physical_device, dev);
   uint32_t stride, alignment;
   hk_descriptor_stride_align_for_type(pdev, type, NULL, &stride, &alignment);
   return hk_descriptor_buffer_stride(stride);
}

static void
hk_descriptor_set_layout_destroy(struct vk_device *vk_dev,
                                 struct vk_descriptor_set_layout *vk_layout)
{
   struct hk_descriptor_set_layout *layout =
      vk_to_hk_descriptor_set_layout(vk_layout);

   vk_free(&vk_dev->alloc, layout->embedded_samplers);
   vk_descriptor_set_layout_destroy(vk_dev, vk_layout);
}

/* Embedded immutable samplers have no descriptor buffer: keep their
 * descriptors with the layout for the command buffer to upload at bind. */
static VkResult
hk_descriptor_set_layout_embed_samplers(struct hk_device *dev,
                                        struct hk_descriptor_set_layout *layout)
{
   const uint32_t size =
      MAX2(layout->non_variable_descriptor_buffer_size, HK_MIN_UBO_ALIGNMENT);
   layout->embedded_samplers = vk_zalloc(&dev->vk.alloc, size, 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!layout->embedded_samplers)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      const struct hk_descriptor_set_binding_layout *binding =
         &layout->binding[b];
      if (binding->type != VK_DESCRIPTOR_TYPE_SAMPLER ||
          binding->immutable_samplers == NULL)
         continue;

      for (uint32_t i = 0; i < binding->array_size; i++) {
         const VkSampler sampler =
            hk_sampler_to_handle(binding->immutable_samplers[i]);
         const VkDescriptorGetInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .data.pSampler = &sampler,
         };
         hk_GetDescriptorEXT(
            hk_device_to_handle(dev), &info, binding->stride,
            (uint8_t *)layout->embedded_samplers + binding->offset +
               i * binding->stride);
      }
   }
   return VK_SUCCESS;
}

static const VkMutableDescriptorTypeListEXT *
hk_descriptor_get_type_list(VkDescriptorType type,
                            const VkMutableDescriptorTypeCreateInfoEXT *info,
                            const uint32_t info_idx)
{
   const VkMutableDescriptorTypeListEXT *type_list = NULL;
   if (type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT) {
      assert(info != NULL);
      assert(info_idx < info->mutableDescriptorTypeListCount);
      type_list = &info->pMutableDescriptorTypeLists[info_idx];
   }
   return type_list;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateDescriptorSetLayout(VkDevice device,
                             const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator,
                             VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   uint32_t num_bindings = 0;
   uint32_t immutable_sampler_count = 0;
   for (uint32_t j = 0; j < pCreateInfo->bindingCount; j++) {
      const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[j];
      num_bindings = MAX2(num_bindings, binding->binding + 1);

      /* From the Vulkan 1.1.97 spec for VkDescriptorSetLayoutBinding:
       *
       *    "If descriptorType specifies a VK_DESCRIPTOR_TYPE_SAMPLER or
       *    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER type descriptor, then
       *    pImmutableSamplers can be used to initialize a set of immutable
       *    samplers. [...]  If descriptorType is not one of these descriptor
       *    types, then pImmutableSamplers is ignored.
       *
       * We need to be careful here and only parse pImmutableSamplers if we
       * have one of the right descriptor types.
       */
      if (binding_has_immutable_samplers(binding))
         immutable_sampler_count += binding->descriptorCount;
   }

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL(&ma, struct hk_descriptor_set_layout, layout, 1);
   VK_MULTIALLOC_DECL(&ma, struct hk_descriptor_set_binding_layout, bindings,
                      num_bindings);
   VK_MULTIALLOC_DECL(&ma, struct hk_sampler *, samplers,
                      immutable_sampler_count);

   if (!vk_descriptor_set_layout_multizalloc(&dev->vk, &ma, pCreateInfo))
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->binding_count = num_bindings;
   layout->descriptor_generation =
      agx_chip_descriptor_generation(pdev->dev.chip);
   layout->native_sampler_indices = agx_chip_native_or_m5(pdev->dev.chip);

   for (uint32_t j = 0; j < pCreateInfo->bindingCount; j++) {
      const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[j];
      uint32_t b = binding->binding;
      /* We temporarily store pCreateInfo->pBindings[] index (plus one) in the
       * immutable_samplers pointer.  This provides us with a quick-and-dirty
       * way to sort the bindings by binding number.
       */
      layout->binding[b].immutable_samplers = (void *)(uintptr_t)(j + 1);
   }

   const VkDescriptorSetLayoutBindingFlagsCreateInfo *binding_flags_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t buffer_size = 0;
   uint32_t max_variable_size = 0;
   uint8_t dynamic_buffer_count = 0;
   for (uint32_t b = 0; b < num_bindings; b++) {
      /* We stashed the pCreateInfo->pBindings[] index (plus one) in the
       * immutable_samplers pointer.  Check for NULL (empty binding) and then
       * reset it and compute the index.
       */
      if (layout->binding[b].immutable_samplers == NULL)
         continue;
      const uint32_t info_idx =
         (uintptr_t)(void *)layout->binding[b].immutable_samplers - 1;
      layout->binding[b].immutable_samplers = NULL;

      const VkDescriptorSetLayoutBinding *binding =
         &pCreateInfo->pBindings[info_idx];

      if (binding->descriptorCount == 0)
         continue;

      layout->binding[b].type = binding->descriptorType;

      if (binding_flags_info && binding_flags_info->bindingCount > 0) {
         assert(binding_flags_info->bindingCount == pCreateInfo->bindingCount);
         layout->binding[b].flags = binding_flags_info->pBindingFlags[info_idx];
      }

      layout->binding[b].array_size = binding->descriptorCount;

      if (vk_descriptor_type_is_dynamic(binding->descriptorType)) {
         layout->binding[b].dynamic_buffer_index = dynamic_buffer_count;
         dynamic_buffer_count += binding->descriptorCount;
      }

      const VkMutableDescriptorTypeListEXT *type_list =
         hk_descriptor_get_type_list(binding->descriptorType, mutable_info,
                                     info_idx);

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_layout(pdev, pCreateInfo->flags,
                                            binding->descriptorType,
                                            type_list, &stride, &alignment);

      uint8_t max_plane_count = 1;

      if (binding_has_immutable_samplers(binding)) {
         layout->binding[b].immutable_samplers = samplers;
         samplers += binding->descriptorCount;
         for (uint32_t i = 0; i < binding->descriptorCount; i++) {
            VK_FROM_HANDLE(hk_sampler, sampler, binding->pImmutableSamplers[i]);
            layout->binding[b].immutable_samplers[i] = sampler;
            const uint8_t sampler_plane_count =
               sampler->vk.ycbcr_conversion
                  ? vk_format_get_plane_count(
                       sampler->vk.ycbcr_conversion->state.format)
                  : 1;
            if (max_plane_count < sampler_plane_count)
               max_plane_count = sampler_plane_count;
         }
      }

      stride *= max_plane_count;

      if (stride > 0) {
         assert(stride <= UINT16_MAX);
         assert(util_is_power_of_two_nonzero(alignment));

         buffer_size = align64(buffer_size, alignment);
         layout->binding[b].offset = buffer_size;
         layout->binding[b].stride = stride;

         if (layout->binding[b].flags &
             VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) {
            /* From the Vulkan 1.3.256 spec:
             *
             *    VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03004
             *    "If an element of pBindingFlags includes
             *    VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT, then
             *    all other elements of
             *    VkDescriptorSetLayoutCreateInfo::pBindings must have a
             *    smaller value of binding"
             *
             * In other words, it has to be the last binding.
             */
            assert(b == num_bindings - 1);
            max_variable_size = stride * binding->descriptorCount;
         } else {
            /* the allocation size will be computed at descriptor allocation,
             * but the buffer size will be already aligned as this binding will
             * be the last
             */
            buffer_size += stride * binding->descriptorCount;
         }
      }
   }

   layout->non_variable_descriptor_buffer_size = buffer_size;
   layout->max_buffer_size = buffer_size + max_variable_size;
   layout->vk.dynamic_descriptor_count = dynamic_buffer_count;
   layout->vk.destroy = hk_descriptor_set_layout_destroy;

   hk_descriptor_set_layout_hash(layout);

   if (pCreateInfo->flags &
       VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT) {
      VkResult result = hk_descriptor_set_layout_embed_samplers(dev, layout);
      if (result != VK_SUCCESS) {
         vk_descriptor_set_layout_unref(&dev->vk, &layout->vk);
         return result;
      }
   }

   *pSetLayout = hk_descriptor_set_layout_to_handle(layout);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDescriptorSetLayoutSupport(
   VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
   VkDescriptorSetLayoutSupport *pSupport)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *binding_flags =
      vk_find_struct_const(pCreateInfo->pNext,
                           DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);

   /* Figure out the maximum alignment up-front.  Otherwise, we need to sort
    * the list of descriptors by binding number in order to get the size
    * accumulation right.
    */
   uint32_t max_align = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[i];
      const VkMutableDescriptorTypeListEXT *type_list =
         hk_descriptor_get_type_list(binding->descriptorType, mutable_info, i);

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_layout(pdev, pCreateInfo->flags,
                                            binding->descriptorType,
                                            type_list, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   uint64_t non_variable_size = 0;
   uint32_t variable_stride = 0;
   uint32_t variable_count = 0;
   bool variable_is_inline_uniform_block = false;
   uint8_t dynamic_buffer_count = 0;

   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[i];

      VkDescriptorBindingFlags flags = 0;
      if (binding_flags != NULL && binding_flags->bindingCount > 0)
         flags = binding_flags->pBindingFlags[i];

      if (vk_descriptor_type_is_dynamic(binding->descriptorType))
         dynamic_buffer_count += binding->descriptorCount;

      const VkMutableDescriptorTypeListEXT *type_list =
         hk_descriptor_get_type_list(binding->descriptorType, mutable_info, i);

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_layout(pdev, pCreateInfo->flags,
                                            binding->descriptorType,
                                            type_list, &stride, &alignment);

      if (stride > 0) {
         assert(stride <= UINT16_MAX);
         assert(util_is_power_of_two_nonzero(alignment));

         if (flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) {
            /* From the Vulkan 1.3.256 spec:
             *
             *    "For the purposes of this command, a variable-sized
             *    descriptor binding with a descriptorCount of zero is treated
             *    as if the descriptorCount is one"
             */
            variable_count = MAX2(1, binding->descriptorCount);
            variable_stride = stride;

            variable_is_inline_uniform_block =
               binding->descriptorType ==
               VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
         } else {
            /* Since we're aligning to the maximum and since this is just a
             * check for whether or not the max buffer size is big enough, we
             * keep non_variable_size aligned to max_align.
             */
            non_variable_size += stride * binding->descriptorCount;
            non_variable_size = align64(non_variable_size, max_align);
         }
      }
   }

   uint64_t buffer_size = non_variable_size;
   if (variable_stride > 0) {
      buffer_size += variable_stride * variable_count;
      buffer_size = align64(buffer_size, max_align);
   }

   uint32_t max_buffer_size;
   if (pCreateInfo->flags &
       VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR)
      max_buffer_size = HK_PUSH_DESCRIPTOR_SET_SIZE;
   else
      max_buffer_size = HK_MAX_DESCRIPTOR_SET_SIZE;

   pSupport->supported = dynamic_buffer_count <= HK_MAX_DYNAMIC_BUFFERS &&
                         buffer_size <= max_buffer_size;

   vk_foreach_struct(sType, ext, pSupport->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT: {
         VkDescriptorSetVariableDescriptorCountLayoutSupport *vs = ext;
         uint32_t max_var_count;

         if (variable_stride > 0) {
            max_var_count =
               (max_buffer_size - non_variable_size) / variable_stride;
         } else {
            max_var_count = 0;
         }

         if (variable_is_inline_uniform_block) {
            max_var_count =
               MIN2(max_var_count, HK_MAX_INLINE_UNIFORM_BLOCK_SIZE);
         }

         vs->maxVariableDescriptorCount = max_var_count;
         break;
      }

      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDescriptorSetLayoutSizeEXT(VkDevice device,
                                 VkDescriptorSetLayout _layout,
                                 VkDeviceSize *pLayoutSizeInBytes)
{
   VK_FROM_HANDLE(hk_descriptor_set_layout, layout, _layout);

   *pLayoutSizeInBytes = layout->max_buffer_size;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDescriptorSetLayoutBindingOffsetEXT(VkDevice device,
                                          VkDescriptorSetLayout _layout,
                                          uint32_t binding,
                                          VkDeviceSize *pOffset)
{
   VK_FROM_HANDLE(hk_descriptor_set_layout, layout, _layout);

   *pOffset = layout->binding[binding].offset;
}
