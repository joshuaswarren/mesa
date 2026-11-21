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
#include "vk_image.h"

struct hk_device;

#define HK_MAX_IMAGE_DESCS (10 * HK_MAX_PLANES)

struct hk_image_view {
   struct vk_image_view vk;

   uint8_t plane_count;
   /* Descriptor encodings of the view's fixed minimum LOD (relative to its
    * base level), computed once instead of on every descriptor write. */
   uint16_t sampled_min_lod_fp16;
   uint16_t sampled_min_lod_uint16;
   /* G17P sampled-descriptor format state of the view's logical format,
    * computed once: the integer conversion kind of a scaled format (0 when
    * not scaled) and whether it is USCALED with a normalized filter form. */
   uint16_t g17p_scaled_conversion;
   bool g17p_uscaled_normalized;
   struct {
      uint8_t image_plane;
      uint32_t sampled_heap_index;
      uint32_t storage_heap_index;
      uint32_t input_heap_index;
      uint32_t emrt_read_heap_index;
      uint32_t filterable_heap_index;

      struct hk_native_texture_descriptor ia, sampled, ro_storage, background,
         layered_background, emrt_texture, sampled_filterable;
      bool filterable_valid;

      struct hk_native_pbe_descriptor storage, eot, layered_eot, emrt_pbe;
      uint64_t storage_atomic_sw;
      /* M5 store window record (hk_native_descriptor.h), zero if unused. */
      uint32_t storage_window[4];
      uint64_t emrt_atomic_sw;
      uint32_t emrt_aligned_width;
      bool storage_pbe_valid, eot_pbe_valid, layered_eot_pbe_valid,
         emrt_pbe_valid;
   } planes[3];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)

/* VK_EXT_descriptor_buffer capture/replay data of an image view: the device
 * heap index of each of its textures, 0 for none. */
enum hk_view_heap {
   HK_VIEW_HEAP_SAMPLED,
   HK_VIEW_HEAP_FILTERABLE,
   HK_VIEW_HEAP_STORAGE,
   HK_VIEW_HEAP_INPUT,
   HK_VIEW_HEAP_EMRT_READ,
   HK_VIEW_HEAP_COUNT,
};

struct hk_image_view_capture {
   uint32_t index[HK_MAX_PLANES][HK_VIEW_HEAP_COUNT];
};
