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

   dev.params.chip_id = 0x6040;
   dev.params.gpu_generation = 16;
   dev.params.gpu_variant = 'S';
   dev.params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                         DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD;
   assert(hk_g16s_as_enabled(&dev));
   assert(!hk_g17p_rt_enabled(&dev));
   dev.params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!hk_g16s_as_enabled(&dev));
   dev.chip = AGX_CHIP_G13G;
   dev.params.features = HK_G17P_RT_KERNEL_FEATURES;
   assert(!hk_g17p_rt_enabled(&dev));
   assert(!hk_g16s_as_enabled(&dev));
   unsetenv("ASAHI_G17P_RT_DISABLE");

   unsetenv("ASAHI_MESH");
   assert(hk_m5_task_enabled());
   assert(!hk_m5_task_record(0));
   assert(hk_m5_task_record(DRM_ASAHI_FEATURE_TASK_MESH_STORAGE));
   puts("native feature defaults: ok");
   return 0;
}
