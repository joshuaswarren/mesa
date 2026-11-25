/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_NIR_LOWER_GS_H
#define HK_NIR_LOWER_GS_H

#include "asahi/lib/agx_chip.h"
#include "nir.h"
#include "poly/nir/poly_nir.h"

struct agx_gen1_ray_query_layout;
struct poly_gs_info;

/* Select where geometry-shader side effects run for a given chip.
 * Generations whose pre-raster CDM path can launch a MESA_SHADER_GEOMETRY
 * main kernel (AGX2 via hk_dispatch_with_usc(), native AGX3 via
 * hk_dispatch_g17p_native(), and M5 via hk_dispatch_m5_native()) move side
 * effects to main whenever replay is independent of them, keeping
 * side-effect-only loops off the tiler while preserving replay when emitted
 * geometry depends on those side effects. G18G's hk_dispatch_g17g_native()
 * rejects non-compute stages and indirect grids, and
 * hk_dispatch_with_usc_launch() rejects non-AGX2 command-stream generations,
 * so those keep replay until they have a graphics-stage CDM launcher. */
static inline enum poly_gs_side_effects
hk_gs_side_effects_mode(enum agx_chip chip)
{
   const bool supports_gs_main_cdm =
      agx_chip_native_agx3(chip) || agx_chip_g17_m5(chip) ||
      agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_AGX2;
   return supports_gs_main_cdm ? POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT
                               : POLY_GS_SIDE_EFFECTS_REPLAY;
}

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
