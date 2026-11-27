/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "vulkan/vulkan_core.h"
#include <stdbool.h>
#include <stdint.h>

static inline bool
hk_g17p_decompress_image_type_supported(VkImageType image_type)
{
   /* 2D layer offsets and compression_layer_stride_B address 2D array layers,
    * not 3D depth slices; refuse 3D images here (note ail_layout::depth_px
    * holds arrayLayers for 2D arrays, so check the API image type directly). */
   return image_type != VK_IMAGE_TYPE_3D;
}

static inline bool
hk_g17p_decompress_pack_mode(enum agx_channels channels, uint32_t *out)
{
   switch (channels) {
   case AGX_CHANNELS_R8:
      *out = LIBAGX_AGX3_DECOMPRESS_R8;
      return true;
   case AGX_CHANNELS_R8G8:
      *out = LIBAGX_AGX3_DECOMPRESS_R8G8;
      return true;
   case AGX_CHANNELS_R16:
      *out = LIBAGX_AGX3_DECOMPRESS_R16;
      return true;
   case AGX_CHANNELS_R5G6B5:
      *out = LIBAGX_AGX3_DECOMPRESS_R5G6B5;
      return true;
   case AGX_CHANNELS_R8G8B8A8:
      *out = LIBAGX_AGX3_DECOMPRESS_R8G8B8A8;
      return true;
   case AGX_CHANNELS_R10G10B10A2:
      *out = LIBAGX_AGX3_DECOMPRESS_R10G10B10A2;
      return true;
   case AGX_CHANNELS_R16G16:
      *out = LIBAGX_AGX3_DECOMPRESS_R16G16;
      return true;
   case AGX_CHANNELS_R32:
      *out = LIBAGX_AGX3_DECOMPRESS_R32;
      return true;
   case AGX_CHANNELS_R16G16B16A16:
      *out = LIBAGX_AGX3_DECOMPRESS_R16G16B16A16;
      return true;
   case AGX_CHANNELS_R32G32:
      *out = LIBAGX_AGX3_DECOMPRESS_R32G32;
      return true;
   case AGX_CHANNELS_R32G32B32A32:
      *out = LIBAGX_AGX3_DECOMPRESS_R32G32B32A32;
      return true;
   default:
      return false;
   }
}
