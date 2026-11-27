/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_NIR_LOWER_IMAGE_VIEW_MIN_LOD_H
#define HK_NIR_LOWER_IMAGE_VIEW_MIN_LOD_H

#include "compiler/nir/nir.h"

bool hk_nir_needs_image_view_min_lod(const nir_shader *nir);
bool hk_nir_lower_image_view_min_lod(nir_shader *nir);
bool hk_nir_lower_image_view_min_lod_post_desc(nir_shader *nir,
                                               unsigned descriptor_generation);

#endif
