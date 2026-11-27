/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_descriptor_set_layout.h"
#include "vk_object.h"

struct agx_device;
struct hk_device;
struct hk_physical_device;
struct hk_sampler;
struct vk_pipeline_layout;

struct hk_descriptor_set_binding_layout {
   /* The type of the descriptors in this binding */
   VkDescriptorType type;

   /* Flags provided when this binding was created */
   VkDescriptorBindingFlags flags;

   /* Shader stages that may access this binding */
   VkShaderStageFlags stage_flags;

   /* Number of array elements in this binding (or size in bytes for inline
    * uniform data)
    */
   uint32_t array_size;

   /* Offset into the descriptor buffer where this descriptor lives */
   uint32_t offset;

   /* Stride between array elements in the descriptor buffer. A multi-plane
    * sampler binding in a descriptor buffer layout takes one combined image
    * sampler descriptor per plane. */
   uint16_t stride;

   /* Index into the dynamic buffer binding array */
   uint8_t dynamic_buffer_index;

   /* Immutable samplers (or NULL if no immutable samplers) */
   struct hk_sampler **immutable_samplers;
};

struct hk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;

   /* Selects both hardware descriptor size and the surrounding Honeykrisp
    * software descriptor ABI.  This is persistent pipeline-layout state, not
    * a property that may be inferred from sizeof(agx_texture_packed).
    */
   uint8_t descriptor_generation;

   /* G17P native heap indices are zero-based. Other chips retain AGX2's
    * software descriptor bias; descriptor generation alone does not select
    * this ABI because Gen9 covers several distinct chips. */
   bool native_sampler_indices;

   /* Precomputed at layout creation: true if the layout or any binding permits
    * UPDATE_AFTER_BIND or UPDATE_UNUSED_WHILE_PENDING. */
   bool has_update_after_bind;

   /* Precomputed at layout creation: true if any vertex-visible image binding
    * permits UPDATE_AFTER_BIND or UPDATE_UNUSED_WHILE_PENDING. */
   bool has_uab_vs_images;

   /* Size of the descriptor buffer for this descriptor set */
   /* Does not contain the size needed for variable count descriptors */
   uint32_t non_variable_descriptor_buffer_size;

   /* VK_EXT_descriptor_buffer: the size with the variable-count binding at
    * its layout descriptorCount (vkGetDescriptorSetLayoutSizeEXT()). */
   uint32_t max_buffer_size;

   /* VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT: the
    * set's descriptors (non_variable_descriptor_buffer_size bytes), uploaded
    * by vkCmdBindDescriptorBufferEmbeddedSamplers2EXT(). */
   void *embedded_samplers;

   /* Number of bindings in this descriptor set */
   uint32_t binding_count;

   /* Bindings in this descriptor set */
   struct hk_descriptor_set_binding_layout binding[0];
};

static inline bool
hk_descriptor_set_layout_allows_update_after_bind(
   const struct hk_descriptor_set_layout *layout)
{
   return !layout || layout->has_update_after_bind;
}

static inline uint16_t
hk_sampler_descriptor_index(const struct hk_descriptor_set_layout *layout,
                            uint16_t heap_index)
{
   return heap_index + (layout->native_sampler_indices ? 0 : 28);
}

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_set_layout, vk.base,
                               VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)

void hk_descriptor_stride_align_for_type(
   const struct hk_physical_device *pdev, VkDescriptorType type,
   const VkMutableDescriptorTypeListEXT *type_list, uint32_t *stride,
   uint32_t *alignment);

void hk_descriptor_stride_align_for_layout(
   const struct hk_physical_device *pdev, VkDescriptorSetLayoutCreateFlags flags,
   VkDescriptorType type, const VkMutableDescriptorTypeListEXT *type_list,
   uint32_t *stride, uint32_t *alignment);

/* The size of one descriptor of this type in a descriptor buffer */
uint32_t hk_descriptor_buffer_size(const struct agx_device *dev,
                                   VkDescriptorType type);

void hk_descriptor_set_layout_hash(struct hk_descriptor_set_layout *layout);

struct blob;
struct blob_reader;

/* Writes the part of `layout` that compiling a shader against it reads: the
 * fields hk_descriptor_set_layout_hash() covers. Returns false, and writes
 * nothing, for a layout with immutable or embedded samplers, which a blob
 * cannot carry. */
bool hk_descriptor_set_layout_serialize(
   struct blob *blob, const struct hk_descriptor_set_layout *layout);

/* Rebuilds a layout that hk_descriptor_set_layout_serialize() wrote, to
 * compile shaders against; no descriptor set is allocated from it. Returns
 * NULL if the blob holds no layout of this device. The caller owns the one
 * reference. */
struct hk_descriptor_set_layout *
hk_descriptor_set_layout_deserialize(struct hk_device *dev,
                                     struct blob_reader *blob);

static inline struct hk_descriptor_set_layout *
vk_to_hk_descriptor_set_layout(struct vk_descriptor_set_layout *layout)
{
   return container_of(layout, struct hk_descriptor_set_layout, vk);
}
