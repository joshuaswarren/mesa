/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_device.h"
#include "hk_private.h"
#include "hk_queue.h"
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
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D,
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

   /* NIR-lowered VK_EXT_depth_clip_control stays on across AGX2, native AGX3,
    * and AGX3C/HAL300 (M5), and fails closed on AGX3D (G18G). */
   assert(agx_chip_has_depth_clip_control(AGX_CHIP_G13G));
   assert(agx_chip_has_depth_clip_control(AGX_CHIP_G14X));
   for (unsigned i = 0; i < ARRAY_SIZE(elide_chips); ++i)
      assert(agx_chip_has_depth_clip_control(elide_chips[i]));
   assert(!agx_chip_has_depth_clip_control(AGX_CHIP_G18G));

   /* Shader-reconstructed packed 4:2:2 YCbCr and 10/12-bit plane UNORM range
    * normalization apply across native AGX3 and M5, keeping AGX2 and AGX3D
    * (G18G) off. */
   for (unsigned i = 0; i < ARRAY_SIZE(elide_chips); ++i)
      assert(agx_chip_has_packed_ycbcr(elide_chips[i]));
   assert(!agx_chip_has_packed_ycbcr(AGX_CHIP_G13G));
   assert(!agx_chip_has_packed_ycbcr(AGX_CHIP_G14X));
   assert(!agx_chip_has_packed_ycbcr(AGX_CHIP_G15P));
   assert(!agx_chip_has_packed_ycbcr(AGX_CHIP_G18G));

   /* Hardware ASTC HDR decode stays on across AGX2, native AGX3, and
    * AGX3C/HAL300 (M5), and fails closed on AGX3D (G18G). */
   assert(agx_chip_has_astc_hdr(AGX_CHIP_G13G));
   assert(agx_chip_has_astc_hdr(AGX_CHIP_G14X));
   for (unsigned i = 0; i < ARRAY_SIZE(elide_chips); ++i)
      assert(agx_chip_has_astc_hdr(elide_chips[i]));
   assert(!agx_chip_has_astc_hdr(AGX_CHIP_G18G));

   /* Kernel-feature-gated cumulative signals and sibling compute queues apply
    * uniformly across G16G, G17P, and M5 (G17G/G17S/G17C/G17D) without chip
    * forks. */
   const uint64_t sib_features = DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |
                                 DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS |
                                 DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS;
   const enum agx_chip sib_chips[] = {
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S,
      AGX_CHIP_G17C, AGX_CHIP_G17D,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(sib_chips); ++i) {
      struct agx_device qdev = {
         .chip = sib_chips[i],
         .params.features = sib_features,
      };
      assert(hk_queue_uses_cumulative_signals(&qdev));
      assert(hk_queue_kernel_cumulative_supported(&qdev, true, false, 1));
      assert(!hk_queue_kernel_cumulative_supported(&qdev, true, true, 1));
      assert(!hk_queue_kernel_cumulative_supported(&qdev, true, false, 0));
      assert(hk_queue_sibling_compute_supported(&qdev, true));
      assert(!hk_queue_sibling_compute_supported(&qdev, false));

      qdev.is_virtio = true;
      assert(!hk_queue_sibling_compute_supported(&qdev, true));
      qdev.is_virtio = false;

      qdev.debug = AGX_DBG_1QUEUE;
      assert(!hk_queue_sibling_compute_supported(&qdev, true));
      qdev.debug = 0;

      /* A kernel that fans compute out itself gets no sibling queues. */
      qdev.params.features =
         sib_features | DRM_ASAHI_FEATURE_COMPUTE_MULTI_CHANNEL;
      assert(hk_queue_uses_cumulative_signals(&qdev));
      assert(!hk_queue_sibling_compute_supported(&qdev, true));
      qdev.params.features = sib_features;

      qdev.params.features = DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |
                             DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS;
      assert(!hk_queue_kernel_cumulative_supported(&qdev, true, false, 1));
      assert(!hk_queue_sibling_compute_supported(&qdev, true));

      qdev.params.features = DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |
                             DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS;
      assert(!hk_queue_sibling_compute_supported(&qdev, true));
   }

   /* Serialized AGX2 queues have exact_prior == false and never open siblings;
    * chips without INDEPENDENT_QUEUES (other than legacy G17P) stay off. */
   struct agx_device g16_no_indep = {
      .chip = AGX_CHIP_G16G,
      .params.features = DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS |
                         DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS,
   };
   assert(!hk_queue_uses_cumulative_signals(&g16_no_indep));
   assert(!hk_queue_sibling_compute_supported(&g16_no_indep, true));
   struct agx_device g13_dev = {
      .chip = AGX_CHIP_G13G,
      .params.features = sib_features,
   };
   assert(!hk_queue_sibling_compute_supported(&g13_dev, false));

   /* Memory scope on every compute and transfer dependency: an M5 device of
    * more than one die, where the scoped ordering record is grounded (the M5
    * Ultra). There it holds on a device through which no memory can alias;
    * one-die M5 devices keep the plain flush there. */
   static const struct {
      enum agx_chip chip;
      unsigned dies;
      bool scoped;
   } scope[] = {
      {AGX_CHIP_G17D, 2, true},  {AGX_CHIP_G17D, 1, false},
      {AGX_CHIP_G17G, 1, false}, {AGX_CHIP_G17S, 1, false},
      {AGX_CHIP_G17C, 1, false}, {AGX_CHIP_G17C, 2, false},
      {AGX_CHIP_G17P, 1, false}, {AGX_CHIP_G18G, 1, false},
      {AGX_CHIP_G16G, 1, false}, {AGX_CHIP_G13X, 2, false},
   };
   static struct hk_device hkdev;
   for (unsigned i = 0; i < ARRAY_SIZE(scope); ++i) {
      hkdev.dev.chip = scope[i].chip;
      hkdev.dev.params.num_dies = scope[i].dies;
      assert(agx_cdm_ordering_memory_scope(&hkdev.dev) == scope[i].scoped);
      if (!agx_chip_g17_m5(scope[i].chip))
         continue;
      hkdev.may_alias_memory = false;
      assert(hk_cdm_alias_scope(&hkdev) == scope[i].scoped);
      hkdev.may_alias_memory = true;
      assert(hk_cdm_alias_scope(&hkdev));
   }

   puts("native feature defaults: ok");
   return 0;
}
