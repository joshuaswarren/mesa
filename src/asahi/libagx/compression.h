/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "agx_pack.h"

#pragma once

struct libagx_decompress_images {
   struct agx_texture_packed compressed;
   struct agx_pbe_packed uncompressed;
};
static_assert(sizeof(struct libagx_decompress_images) == 48);

enum libagx_agx3_decompress_pack {
   LIBAGX_AGX3_DECOMPRESS_R8 = 0,
   LIBAGX_AGX3_DECOMPRESS_R8G8 = 1,
   LIBAGX_AGX3_DECOMPRESS_R16 = 2,
   LIBAGX_AGX3_DECOMPRESS_R5G6B5 = 3,
   LIBAGX_AGX3_DECOMPRESS_R8G8B8A8 = 4,
   LIBAGX_AGX3_DECOMPRESS_R10G10B10A2 = 5,
   LIBAGX_AGX3_DECOMPRESS_R16G16 = 6,
   LIBAGX_AGX3_DECOMPRESS_R32 = 7,
   LIBAGX_AGX3_DECOMPRESS_R16G16B16A16 = 8,
   LIBAGX_AGX3_DECOMPRESS_R32G32 = 9,
   LIBAGX_AGX3_DECOMPRESS_R32G32B32A32 = 10,
};
