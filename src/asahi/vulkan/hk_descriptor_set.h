/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "hk_device.h"
#include "hk_native_descriptor_pack.h"
#include "vk_descriptor_update_template.h"
#include "vk_object.h"

#include "util/list.h"
#include "util/vma.h"

struct hk_descriptor_set_layout;

/* Software tail following a generation-sized native texture descriptor. */
struct hk_sampled_image_descriptor_sw {
   union {
      struct agx_sampler_packed sampler;
      /* G17P uses heap samplers, leaving the embedded AGX2 word available. */
      struct {
         uint16_t unbiased_sampler_index;
         uint16_t unbiased_clamp_0_sampler_index;
         float lod_bias;
      } g17p;
   };
   /* Raw native heap index on G17P; AGX2-biased index on other chips. */
   uint16_t sampler_index;

   /* Negative if there is no border colour, else the clamp=0 sampler index used
    * for custom border colour emulation. Bit 14 marks an integer-typed
    * border (VkBorderColor INT variants); indices stay below it.
    */
   int16_t clamp_0_sampler_index_or_negative;
#define HK_BORDER_IS_INT_BIT (1 << 14)

   /* We want LOD bias and min LOD together as fp16 in this order and 32-bit
    * aligned. This lets us keep everything vectorized on the shader side, since
    * this matches the AGX2 lod_bias + min_lod descriptor.
    */
   uint16_t lod_bias_fp16;
   uint16_t min_lod_fp16;

   /* This is only used for txf, it can be separate from the other stuff */
   uint16_t min_lod_uint16;
   /* Low two bits: 0 = none, 1 = SINT, 2 = UINT, 3 = normalized USCALED.
    * For kind 3, bit 2 selects 16 rather than 8 bits, and bits 3..6 mark
    * swizzled output channels to rescale (constant zero/one stay unchanged).
    * Bit 7 selects packed 10:10:10:2; bits 8..11 mark its swizzled 2-bit
    * channels. */
   uint16_t scaled_conversion;
   /* TODO: This should probably be a heap! */
   uint32_t border[4];
   /* Native G17P global texture index; the descriptor's size is unchanged. */
   uint32_t sampled_heap_index;
};
static_assert(offsetof(struct hk_sampled_image_descriptor_sw, sampled_heap_index) == 36,
              "native texture heap index keeps the reserved software tail ABI");
static_assert(sizeof(struct hk_sampled_image_descriptor_sw) ==
                 HK_SAMPLED_IMAGE_SW_SIZE,
              "hk_sampled_image_descriptor_sw has no holes");

struct hk_sampled_image_descriptor {
   struct hk_native_texture_descriptor tex;
   struct hk_sampled_image_descriptor_sw sw;
};
static_assert(sizeof(struct hk_sampled_image_descriptor) == 72,
              "maximum sampled descriptor has no holes");

struct hk_storage_image_descriptor {
   struct hk_native_texture_descriptor tex;
   struct hk_native_pbe_descriptor pbe;
   uint32_t texture_heap_index;
   uint32_t pad;
   uint64_t atomic_sw;
};
static_assert(sizeof(struct hk_storage_image_descriptor) == 80,
              "maximum storage descriptor has no holes");
/* The G17P state-0 read/write pair takes the PBE as record 1 after the
 * texture: hk_g17p_storage_image_pointers() and
 * agx_nir_lower_g17p_storage_image() derive one offset from the other. */
static_assert(offsetof(struct hk_storage_image_descriptor, pbe) -
                    offsetof(struct hk_storage_image_descriptor, tex) ==
                 32,
              "G17P storage descriptor records are 32 bytes apart");

/* This has to match nir_address_format_64bit_bounded_global */
struct hk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   /*
    * G17P sparse-buffer descriptors store the page number of the residency
    * map here. Descriptor lowering consumes it and restores the address
    * format's required zero byte offset before explicit-I/O lowering.
    */
   uint32_t sparse_map_page;
};

struct hk_descriptor_pool {
   struct vk_object_base base;

   struct list_head sets;

   struct agx_bo *bo;
   uint8_t *mapped_ptr;
   struct util_vma_heap heap;

   /* The pool BO is CPU-cached: every CPU write is cleaned to the point of
    * coherency as it is made, and queue submission fences those cleans. */
   bool cpu_cached;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

struct hk_descriptor_set {
   struct vk_object_base base;

   /* Link in hk_descriptor_pool::sets */
   struct list_head link;

   struct hk_descriptor_set_layout *layout;
   void *mapped_ptr;
   uint64_t addr;
   uint32_t size;
   /* mapped_ptr is in a CPU-cached pool BO (hk_descriptor_pool::cpu_cached). */
   bool cpu_cached;

   struct hk_buffer_address dynamic_buffers[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)

static inline uint64_t
hk_descriptor_set_addr(const struct hk_descriptor_set *set)
{
   return set->addr;
}

struct hk_push_descriptor_set {
   blake3_hash layout_hash;
   VkDescriptorSetLayoutCreateFlags layout_flags;
   /* Bytes of data the last update's layout uses */
   uint32_t size;
   uint8_t data[HK_PUSH_DESCRIPTOR_SET_SIZE];
};

bool hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                                   struct hk_descriptor_set_layout *layout,
                                   uint32_t write_count,
                                   const VkWriteDescriptorSet *writes);

bool hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data);
