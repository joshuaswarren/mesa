/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_lower_gs.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "poly/nir/poly_nir.h"

bool
hk_nir_lower_gs(nir_shader *gs, nir_shader **count, nir_shader **rast,
                nir_shader **pre_gs, struct poly_gs_info *info,
                const struct agx_gen1_ray_query_layout *rq_layout,
                enum poly_gs_side_effects side_effects,
                poly_nir_remove_dead_fn remove_dead)
{
   /* rq_layout is supplied only by the G17P native ray-query profile. */
   poly_nir_lower_gs_with_options(gs, count, rast, pre_gs, info,
                                  rq_layout != NULL, side_effects,
                                  remove_dead);

   if (!rq_layout)
      return true;

   /* Poly clones the GS for rasterization and changes its stage to VERTEX.
    * Native queries must be lowered after that split: the pre-raster/count
    * shaders use the CDM query context, while the raster replay uses the
    * graphics context. Lowering before Poly bakes the CDM ABI into the vertex
    * replay, including a two-component context load where graphics needs four.
    * It also lets Poly strip query-internal global stores as GS side effects.
    */
   nir_shader *variants[] = {gs, *count, *rast};
   for (unsigned i = 0; i < ARRAY_SIZE(variants); ++i) {
      nir_shader *nir = variants[i];
      if (!nir)
         continue;

      nir_opt_ray_queries(nir);
      nir_remove_dead_variables(nir, nir_var_function_temp | nir_var_shader_temp,
                                 NULL);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      if (!nir->info.ray_queries)
         continue;

      nir_opt_ray_query_ranges(nir);
      if (!agx_nir_lower_gen1_ray_query(nir, rq_layout))
         return false;
   }

   return true;
}
