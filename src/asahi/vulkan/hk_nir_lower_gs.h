/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_NIR_LOWER_GS_H
#define HK_NIR_LOWER_GS_H

#include "nir.h"
#include "poly/nir/poly_nir.h"

struct agx_gen1_ray_query_layout;
struct poly_gs_info;

/* Split a GS, then lower native queries for each execution stage. Returns
 * success, not pass progress. The caller owns all output NIR even on failure.
 * side_effects and remove_dead are those of
 * poly_nir_lower_gs_with_options().
 */
bool hk_nir_lower_gs(nir_shader *gs, nir_shader **count, nir_shader **rast,
                     nir_shader **pre_gs, struct poly_gs_info *info,
                     const struct agx_gen1_ray_query_layout *rq_layout,
                     enum poly_gs_side_effects side_effects,
                     poly_nir_remove_dead_fn remove_dead);

#endif
