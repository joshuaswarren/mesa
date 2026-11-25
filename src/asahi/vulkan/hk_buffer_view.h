/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "agx_pack.h"
#include "hk_private.h"
#include "hk_native_descriptor_pack.h"

#include "vk_buffer_view.h"

struct hk_physical_device;

VkFormatFeatureFlags2
hk_get_buffer_format_features(struct hk_physical_device *pdevice,
                              VkFormat format);

struct hk_buffer_view {
   struct vk_buffer_view vk;
   struct hk_native_texture_descriptor tex;
   struct hk_native_pbe_descriptor pbe;
   /* Sampled-image device heap slot on AGX3/M5; zero on earlier chips, which
    * read texture descriptors inline.
    */
   uint32_t sampled_heap_index;
   struct hk_g17p_buffer_texture_metadata buffer_metadata;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)

struct hk_device;

/* VK_EXT_descriptor_buffer texel buffers (G17P, see hk_buffer_view.c) */
VkResult hk_texel_windows_init(struct hk_device *dev);
void hk_texel_windows_finish(struct hk_device *dev);

/* Write the AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE bytes of a texel buffer
 * descriptor for vkGetDescriptorEXT() into zeroed memory; a null or
 * unrepresentable buffer gets the null descriptor. */
void hk_texel_buffer_descriptor(struct hk_device *dev,
                                const VkDescriptorAddressInfoEXT *info,
                                bool storage, void *out);
