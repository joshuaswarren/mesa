/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_private.h"
#include "hk_nir_lower_mesh.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
   struct agx_device dev = {0};
   dev.chip = AGX_CHIP_G17P;
   dev.params.features = HK_G17P_RT_KERNEL_FEATURES;
   assert(hk_g17p_rt_enabled(&dev));

   /* A stale environment from an older driver cannot hide an implemented
    * feature. Missing kernel support still must. */
   setenv("ASAHI_G17P_RT_DISABLE", "1", 1);
   assert(hk_g17p_rt_enabled(&dev));
   dev.params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!hk_g17p_rt_enabled(&dev));

   dev.params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                         DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD;
   static const uint32_t g17p_ids[] = {0x6040, 0x6041, 0x8140};
   for (unsigned i = 0; i < ARRAY_SIZE(g17p_ids); ++i) {
      dev.params.chip_id = g17p_ids[i];
      assert(hk_g16s_as_enabled(&dev));
      assert(!hk_g17p_rt_enabled(&dev));
   }
   dev.params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!hk_g16s_as_enabled(&dev));
   dev.chip = AGX_CHIP_G13G;
   dev.params.features = HK_G17P_RT_KERNEL_FEATURES;
   assert(!hk_g17p_rt_enabled(&dev));
   assert(!hk_g16s_as_enabled(&dev));
   unsetenv("ASAHI_G17P_RT_DISABLE");

   assert(!hk_m5_task_record(0));
   assert(hk_m5_task_record(DRM_ASAHI_FEATURE_TASK_MESH_STORAGE));

   /* Application compute dispatch barrier elision defaults on for every chip
    * sharing the native AGX3 CDM dispatch paths (G15, G16G/G17P HAL200, and
    * M5 G17G/G17S/G17C HAL300), and off on AGX2 and bring-up G18G. */
   static const enum agx_chip elide_chips[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C,
      AGX_CHIP_G16G, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(elide_chips); ++i)
      assert(agx_chip_elide_app_dispatch_barrier(elide_chips[i]));

   assert(!agx_chip_elide_app_dispatch_barrier(AGX_CHIP_G13G));
   assert(!agx_chip_elide_app_dispatch_barrier(AGX_CHIP_G14X));
   assert(!agx_chip_elide_app_dispatch_barrier(AGX_CHIP_G15P));
   assert(!agx_chip_elide_app_dispatch_barrier(AGX_CHIP_G18G));

   /* Dynamic attachment feedback loop state tracks the shared depth/stencil
    * twin and draw-time selection across native AGX3 and M5. */
   for (unsigned i = 0; i < ARRAY_SIZE(elide_chips); ++i)
      assert(agx_chip_has_feedback_loop_dynamic_state(elide_chips[i]));
   assert(!agx_chip_has_feedback_loop_dynamic_state(AGX_CHIP_G13G));
   assert(!agx_chip_has_feedback_loop_dynamic_state(AGX_CHIP_G14X));
   assert(!agx_chip_has_feedback_loop_dynamic_state(AGX_CHIP_G18G));

   puts("native feature defaults: ok");
   return 0;
}
