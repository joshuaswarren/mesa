/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "util/u_math.h"

/* Reduced AIL power-of-two tile geometry consumed by libagx copy and mipmap
 * compute kernels so every texel uses shifts and a seven-bit Morton encoder
 * instead of integer division and the generic fourteen-bit encoder. */
struct agx_tile_address_config {
   uint32_t tiles_per_row;
   uint32_t tile_width_log2;
   uint32_t tile_height_log2;
};

static inline bool
agx_tile_address_config_init(uint32_t stride_el, uint32_t tile_width_el,
                             uint32_t tile_height_el,
                             struct agx_tile_address_config *config)
{
   if (config)
      memset(config, 0, sizeof(*config));

   if (!config || !stride_el || !tile_width_el || !tile_height_el ||
       (tile_width_el & (tile_width_el - 1)) ||
       (tile_height_el & (tile_height_el - 1)))
      return false;

   const uint32_t width_log2 = util_logbase2(tile_width_el);
   const uint32_t height_log2 = util_logbase2(tile_height_el);
   if (width_log2 > 7 || height_log2 > 7)
      return false;

   *config = (struct agx_tile_address_config){
      .tiles_per_row =
         ((uint64_t)stride_el + tile_width_el - 1) >> width_log2,
      .tile_width_log2 = width_log2,
      .tile_height_log2 = height_log2,
   };
   return true;
}

/* Precompute packed Q8 source coordinates (lower | (weight << 16)) for one
 * axis of an odd-size UNORM8x4 compute mipmap downsample, shared between
 * Borealis and Gallium Asahi so GL and Vulkan filter weights remain identical. */
static inline bool
agx_build_mip_axis(uint32_t src_extent, uint32_t dst_extent, uint32_t *entries)
{
   if (!entries || !src_extent || !dst_extent ||
       src_extent > UINT16_MAX + 1u ||
       !(src_extent == dst_extent ||
         src_extent == (uint64_t)dst_extent * 2u ||
         src_extent == (uint64_t)dst_extent * 2u + 1u))
      return false;

   if (src_extent == dst_extent) {
      for (uint32_t i = 0; i < dst_extent; ++i)
         entries[i] = i;
      return true;
   }

   /* vk_meta computes the normalized scale in double on the CPU, casts it to
    * f32 push data, then applies it to FragCoord with an FFMA. */
   const float scale = (float)(1.0 / (double)dst_extent);
   for (uint32_t i = 0; i < dst_extent; ++i) {
      /* This work happens once while recording the dispatch, not per pixel. */
      const float uv = ((float)i + 0.5f) * scale;
      const float source = uv * (float)src_extent - 0.5f;
      uint32_t lower = (uint32_t)source;
      uint32_t weight = (uint32_t)((source - (float)lower) * 256.0f + 0.5f);

      if (weight == 256) {
         ++lower;
         weight = 0;
      }

      if (lower >= src_extent)
         lower = src_extent - 1;

      entries[i] = lower | (weight << 16);
   }

   return true;
}
