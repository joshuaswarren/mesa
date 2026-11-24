/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/format/u_formats.h"

#include <stdbool.h>

/* Colour formats whose G17P images Gallium may allocate losslessly
 * compressed.
 *
 * These are the pipe formats of Honeykrisp's G17P colour policy
 * (hk_g17p_compression_color_format()) plus the X variants GL renders to.
 * An X variant has the same AIL channel class, type and tilebuffer format as
 * its A variant, so its PBE and texture descriptors differ only in swizzle,
 * and compression is a property of the channel class. The compressed packets
 * themselves are the shared Gen9 ones (agx_pbe_agx3_core.c), pinned against
 * native bytes by test-hk-native-descriptor.c.
 *
 * Every format here has a tilebuffer footprint of at most 8 bytes per
 * sample, and none has a different store class (R11G11B10 and RGB9E5 are
 * stored as R32_UINT words). At 1x, with at most seven other render targets
 * of at most 16 bytes, a G17P tilebuffer (120 bytes per sample) never spills
 * one of these; test-tilebuffer.cpp checks both properties at 1x. At 2x/4x
 * MSAA, wide MRT framebuffers (such as 5x RGBA16F at 4x) can exceed the
 * tilebuffer byte budget and spill, so Gallium pre-decompresses spilling
 * private targets via agx_decompress() in agx_set_framebuffer_state() rather
 * than relying on AGX2-only in-place decompression.
 */
static inline bool
agx_g17p_compression_color_format(enum pipe_format format)
{
   switch (format) {
   /* 1 byte */
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   /* 2 bytes */
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   /* 4 bytes */
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_SNORM:
   case PIPE_FORMAT_R8G8B8X8_UINT:
   case PIPE_FORMAT_R8G8B8X8_SINT:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R10G10B10X2_UNORM:
   case PIPE_FORMAT_B10G10R10X2_UNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   /* 8 bytes */
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16B16X16_UINT:
   case PIPE_FORMAT_R16G16B16X16_SINT:
   case PIPE_FORMAT_R16G16B16X16_FLOAT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_FLOAT:
      return true;
   default:
      return false;
   }
}

/* 16-byte RGBA32 colour formats that G17P may allocate losslessly compressed
 * on single-sample, single-layer 2D images (including mip chains).
 *
 * Kept separate from agx_g17p_compression_color_format() because a wide MRT
 * framebuffer (e.g. eight 16-byte targets = 128 bytes > G17P's 120-byte
 * tilebuffer sample budget) spills the tail attachment; Gallium handles that
 * by pre-decompressing spilling private targets via agx_decompress() in
 * agx_set_framebuffer_state(). Multisample 16-byte targets stay uncompressed.
 */
static inline bool
agx_g17p_compression_color_format_16b(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32X32_FLOAT:
   case PIPE_FORMAT_R32G32B32X32_UINT:
   case PIPE_FORMAT_R32G32B32X32_SINT:
      return true;
   default:
      return false;
   }
}
