/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Push descriptor layouts at maxPushDescriptors whose immutable samplers
 * convert a multi-planar format, which multiplies the stride by the plane
 * count, fit the push storage, and an update writes only within it; an
 * update whose layout does not fit is refused.
 */
#undef NDEBUG
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "vk_ycbcr_conversion.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CANARY 0xa5

struct canaried_push_set {
   struct hk_push_descriptor_set set;
   uint8_t after[1024];
};

static struct hk_descriptor_set_layout *
create_layout(struct hk_device *dev,
              const VkDescriptorSetLayoutBinding *bindings,
              uint32_t binding_count)
{
   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = binding_count,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout handle = VK_NULL_HANDLE;
   assert(hk_CreateDescriptorSetLayout(hk_device_to_handle(dev), &info, NULL,
                                       &handle) == VK_SUCCESS);
   return hk_descriptor_set_layout_from_handle(handle);
}

/* Pushes every element of the combined image sampler bindings of layout,
 * without an image, and checks that the bytes past the layout's, and past the
 * storage, keep their canary. Returns whether the update wrote the last
 * element of the last binding. */
static bool
push_all(struct hk_descriptor_set_layout *layout,
         const VkDescriptorSetLayoutBinding *bindings, uint32_t binding_count)
{
   struct canaried_push_set *push = malloc(sizeof(*push));
   assert(push);
   memset(push, CANARY, sizeof(*push));

   VkDescriptorImageInfo images[128] = {0};
   VkWriteDescriptorSet writes[8];
   uint32_t write_count = 0;
   assert(binding_count <= ARRAY_SIZE(writes));
   for (uint32_t b = 0; b < binding_count; ++b) {
      if (bindings[b].descriptorType !=
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         continue;
      assert(bindings[b].descriptorCount <= ARRAY_SIZE(images));
      writes[write_count++] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = bindings[b].binding,
         .descriptorCount = bindings[b].descriptorCount,
         .descriptorType = bindings[b].descriptorType,
         .pImageInfo = images,
      };
   }
   const bool accepted =
      hk_push_descriptor_set_update(&push->set, layout, write_count, writes);
   assert(accepted == (layout->non_variable_descriptor_buffer_size <=
                       sizeof(push->set.data)));

   const uint32_t size = push->set.size;
   assert(size <= sizeof(push->set.data));
   for (uint32_t i = size; i < sizeof(push->set.data); ++i)
      assert(push->set.data[i] == CANARY);
   for (uint32_t i = 0; i < sizeof(push->after); ++i)
      assert(push->after[i] == CANARY);

   const struct hk_descriptor_set_binding_layout *last =
      &layout->binding[bindings[binding_count - 1].binding];
   const uint32_t last_offset =
      last->offset + (last->array_size - 1) * last->stride;
   bool written = false;
   for (uint32_t i = 0; size && i < last->stride; ++i)
      written |= push->set.data[last_offset + i] != CANARY;

   free(push);
   return written;
}

static void
check_chip(enum agx_chip chip, struct hk_sampler *samplers)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   assert(pdev && dev);
   pdev->dev.chip = chip;
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);

   VkSampler handles[HK_MAX_PUSH_DESCRIPTORS * 4];
   for (unsigned i = 0; i < ARRAY_SIZE(handles); ++i)
      handles[i] = hk_sampler_to_handle(&samplers[i % 2]);

   /* maxPushDescriptors combined image samplers whose immutable samplers
    * convert a three-plane format, and the same after a storage buffer and a
    * storage image, whose alignment pads the samplers' binding. */
   const VkDescriptorSetLayoutBinding full[] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, HK_MAX_PUSH_DESCRIPTORS,
       VK_SHADER_STAGE_FRAGMENT_BIT, handles},
   };
   const VkDescriptorSetLayoutBinding mixed[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, NULL},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       HK_MAX_PUSH_DESCRIPTORS - 2, VK_SHADER_STAGE_FRAGMENT_BIT, handles},
   };
   struct {
      const VkDescriptorSetLayoutBinding *bindings;
      uint32_t count;
   } cases[] = {{full, ARRAY_SIZE(full)}, {mixed, ARRAY_SIZE(mixed)}};

   for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
      struct hk_descriptor_set_layout *layout =
         create_layout(dev, cases[c].bindings, cases[c].count);
      const struct hk_descriptor_set_binding_layout *samplers_binding =
         &layout->binding[cases[c].count - 1];
      const uint32_t native =
         hk_sampled_image_descriptor_layout(layout->descriptor_generation)
            .stride;
      assert(samplers_binding->stride == HK_MAX_PLANES * native);
      assert(layout->non_variable_descriptor_buffer_size <=
             HK_PUSH_DESCRIPTOR_SET_SIZE);
      assert(push_all(layout, cases[c].bindings, cases[c].count));
      vk_descriptor_set_layout_unref(&dev->vk, &layout->vk);
   }

   /* A push layout beyond maxPushDescriptors that does not fit is left empty
    * rather than written past the storage. */
   const VkDescriptorSetLayoutBinding over[] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ARRAY_SIZE(handles),
       VK_SHADER_STAGE_FRAGMENT_BIT, handles},
   };
   struct hk_descriptor_set_layout *layout =
      create_layout(dev, over, ARRAY_SIZE(over));
   assert(layout->non_variable_descriptor_buffer_size >
          HK_PUSH_DESCRIPTOR_SET_SIZE);
   assert(!push_all(layout, over, ARRAY_SIZE(over)));
   vk_descriptor_set_layout_unref(&dev->vk, &layout->vk);

   vk_object_base_finish(&dev->vk.base);
   vk_object_base_finish(&pdev->vk.base);
   free(dev);
   free(pdev);
}

int
main(void)
{
   struct vk_ycbcr_conversion conversion = {
      .state.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
   };
   vk_object_base_init(NULL, &conversion.base,
                       VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION);
   struct hk_rc_sampler hw[2] = {{.index = 1}, {.index = 2}};

   /* A converting sampler takes up to two hardware samplers (hk_sampler.c);
    * both counts reach every plane of the binding. */
   struct hk_sampler samplers[2];
   for (unsigned i = 0; i < 2; ++i) {
      samplers[i] = (struct hk_sampler){
         .plane_count = i + 1,
         .planes = {{.hw = &hw[0]}, {.hw = &hw[1]}},
      };
      vk_object_base_init(NULL, &samplers[i].vk.base, VK_OBJECT_TYPE_SAMPLER);
      samplers[i].vk.ycbcr_conversion = &conversion;
      samplers[i].ycbcr = conversion.state;
   }

   const enum agx_chip chips[] = {AGX_CHIP_G13G, AGX_CHIP_G14G, AGX_CHIP_G15G,
                                  AGX_CHIP_G17P, AGX_CHIP_G17C};
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i)
      check_chip(chips[i], samplers);

   printf("PASS: push layouts of %u three-plane descriptors fit %u bytes on "
          "%zu chips\n",
          HK_MAX_PUSH_DESCRIPTORS, HK_PUSH_DESCRIPTOR_SET_SIZE,
          ARRAY_SIZE(chips));
   return 0;
}
