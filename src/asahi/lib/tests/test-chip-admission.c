/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "agx_chip.h"
#include "agx_va_runtime_hole.h"
#include "util/macros.h"

static struct drm_asahi_params_global
g15_params(char variant, uint32_t chip_id, uint32_t clusters)
{
   struct drm_asahi_params_global params = {
      .gpu_generation = 15,
      .gpu_variant = variant,
      .chip_id = chip_id,
      .num_dies = 1,
      .num_clusters_total = clusters,
      .num_cores_per_cluster = 10,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
   };

   for (unsigned i = 0; i < clusters; ++i)
      params.core_masks[i] = 0x3ff;
   return params;
}

static bool
overlaps(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
   return a0 < b1 && b0 < a1;
}

/* The main heap and sparse alias agx_open_device derives for a native chip
 * below the USC window (the same split and halving). */
static void
native_user_window(enum agx_chip chip, uint64_t *start, uint64_t *end)
{
   const uint64_t limit = agx_va_user_limit(chip, true, UINT64_C(0x2f7ffff8000));
   uint64_t base = MAX2(UINT64_C(0x100000000), AGX_G17P_USER_LIMIT / 2);
   uint64_t size = limit - base;

   assert(agx_va_split_user_window(base, size, &base, &size));
   while (size > 0 && base + 2 * size > limit)
      size /= 2;
   assert(size > 0);
   *start = base;
   *end = base + 2 * size;
}

static void
check_native_g15(enum agx_chip chip)
{
   assert(agx_chip_native_agx3(chip));
   assert(agx_chip_native_g15(chip));
   assert(agx_chip_native_imageblock(chip));
   assert(agx_chip_native_indirect_compute(chip));
   assert(agx_chip_driver_supported(chip));
   assert(agx_chip_usc_helper_table_slots(chip, 0) == 10);
}

int
main(void)
{
   check_native_g15(AGX_CHIP_G15S);
   check_native_g15(AGX_CHIP_G15C);
   check_native_g15(AGX_CHIP_G15G);

   /* The other M3 parts stay off the native path. */
   assert(!agx_chip_native_agx3(AGX_CHIP_G15D));
   assert(!agx_chip_driver_supported(AGX_CHIP_G15D));

   /* T6030 and T6031 are each admitted with their own identity only. */
   struct drm_asahi_params_global g15s = g15_params('S', 0x6030, 2);
   struct drm_asahi_params_global g15c = g15_params('C', 0x6031, 4);
   assert(agx_chip_from_params(&g15s) == AGX_CHIP_G15S);
   assert(agx_chip_from_params(&g15c) == AGX_CHIP_G15C);
   assert(agx_params_driver_supported(&g15s));
   assert(agx_params_driver_supported(&g15c));

   /* A binned part reports fewer cores in its masks and is still admitted. */
   struct drm_asahi_params_global binned = g15c;
   binned.core_masks[0] = binned.core_masks[3] = 0x1ff;
   assert(agx_params_driver_supported(&binned));

   struct drm_asahi_params_global bad = g15c;
   bad.chip_id = 0x6030;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_clusters_total = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_dies = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.chip_id = 0x6031;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.num_clusters_total = 4;
   assert(!agx_params_driver_supported(&bad));

   /* The same GPU identity must not admit the incompatible legacy payloads. */
   assert(!agx_params_native_kernel_abi_supported(NULL));
   assert(!agx_params_native_kernel_abi_supported(&g15s));
   const uint64_t native = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                           DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   g15s.features = native;
   g15c.features = native;
   assert(agx_params_native_kernel_abi_supported(&g15s));
   assert(agx_params_native_kernel_abi_supported(&g15c));
   /* Extra advertised capabilities do not change this packet contract. */
   for (unsigned bit = 3; bit <= 16; ++bit) {
      bad = g15c;
      bad.features |= UINT64_C(1) << bit;
      assert(agx_params_native_kernel_abi_supported(&bad));
   }
   bad = g15c;
   bad.features = (UINT64_C(1) << 3) | (UINT64_C(1) << 5);
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = g15c;
   bad.features &= ~DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = (struct drm_asahi_params_global){
      .gpu_generation = 17, .gpu_variant = 'P', .num_clusters_total = 1,
      .chip_id = 0x8140, .num_dies = 1, .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200, .features = native,
   };
   assert(agx_params_driver_supported(&bad));
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad.features = UINT64_C(0xfc00) | native;
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad.features = (UINT64_C(1) << 3) | (UINT64_C(1) << 5);
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = (struct drm_asahi_params_global){
      .gpu_generation = 13, .gpu_variant = 'G', .num_clusters_total = 1,
   };
   assert(agx_params_native_kernel_abi_supported(&bad));
   /* The M4 Pro (T6040, G16S, AGX3, HAL200) runs as G17P with its two
    * clusters; other G16S identities stay G16S and unsupported. */
   struct drm_asahi_params_global t6040 = g15_params('S', 0x6040, 2);
   t6040.gpu_generation = 16;
   t6040.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(agx_chip_from_params(&t6040) == AGX_CHIP_G17P);
   assert(agx_params_driver_supported(&t6040));

   bad = t6040;
   bad.num_clusters_total = 3;
   assert(!agx_params_driver_supported(&bad));

   bad = t6040;
   bad.chip_id = 0x6041;
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G16S);
   assert(!agx_params_driver_supported(&bad));

   const char variants[] = {'C'};
   for (unsigned v = 0; v < sizeof(variants); ++v) {
      struct drm_asahi_params_global t6041 = g15_params(variants[v], 0x6041, 4);
      t6041.gpu_generation = 16;
      t6041.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
      assert(agx_chip_from_params(&t6041) == AGX_CHIP_G17P);
      assert(agx_params_driver_supported(&t6041));

      /* The out-of-tree T6041 X identity is not the Aurora contract. */
      bad = t6041;
      bad.gpu_variant = 'X';
      assert(agx_chip_from_params(&bad) == AGX_CHIP_UNKNOWN);
      assert(!agx_params_driver_supported(&bad));

      bad = t6041;
      bad.core_masks[1] = 0x1ff;
      assert(agx_params_driver_supported(&bad));

      bad = t6041;
      bad.num_clusters_total = 2;
      assert(!agx_params_driver_supported(&bad));

      bad = t6041;
      bad.num_dies = 2;
      assert(!agx_params_driver_supported(&bad));

      bad = t6041;
      bad.usc_generation = 0;
      bad.gpu_hal_generation = 0;
      assert(agx_chip_from_params(&bad) != AGX_CHIP_G17P);
      assert(!agx_params_driver_supported(&bad));

      bad = t6041;
      bad.chip_id = 0x6042;
      assert(!agx_params_driver_supported(&bad));
   }

   bad = t6040;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY;
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G16S);

   {
      const uint64_t g17_features =
         DRM_ASAHI_FEATURE_VM_STATUS | DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES |
         DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
         DRM_ASAHI_FEATURE_VM_STATUS_MIRROR |
         DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
         DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS |
         DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD |
         DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS;
      const char letters[] = {'C'};
      for (unsigned v = 0; v < sizeof(letters); ++v) {
         struct drm_asahi_params_global aurora =
            g15_params(letters[v], 0x6041, 4);
         aurora.gpu_generation = 16;
         aurora.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
         aurora.gpu_revision = 0x11;
         aurora.features = g17_features;
         const struct drm_asahi_params_global before = aurora;
         assert(agx_chip_from_params(&aurora) == AGX_CHIP_G17P);
         assert(agx_params_driver_supported(&aurora));
         assert(agx_chip_usc_base_page_fill_returns_for(
            AGX_CHIP_G17P, aurora.num_clusters_total));
         assert(agx_chip_usc_helper_table_slots(
                   AGX_CHIP_G17P, aurora.num_cores_per_cluster) == 10);

         struct drm_asahi_params_global other = before;
         other.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
         assert(!agx_params_driver_supported(&other));
         other = before;
         other.usc_generation = 2;
         assert(agx_chip_from_params(&other) != AGX_CHIP_G17P);
         assert(!agx_params_driver_supported(&other));
      }
   }

   /* G17P helper tables have a slot per core of a cluster, never fewer than
    * the A18 Pro's six: six there, ten on the M4 Pro. */
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 0) == 6);
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 6) == 6);
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 10) == 10);

   /* Implemented matrix paths are available without process flags. */
   const enum agx_chip matrix_chips[] = {
      AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(matrix_chips); ++i)
      assert(agx_chip_cooperative_matrix_supported(matrix_chips[i]));
   assert(!agx_chip_cooperative_matrix_supported(AGX_CHIP_G18G));

   /* The M5 Max (T6050, G17C) is admitted with its four clusters only, by
    * default and on the M5 bring-up path. */
   struct drm_asahi_params_global t6050c = g15_params('C', 0x6050, 4);
   t6050c.gpu_generation = 17;
   t6050c.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
   assert(agx_chip_from_params(&t6050c) == AGX_CHIP_G17C);
   assert(agx_chip_g17_m5(AGX_CHIP_G17C));
   assert(agx_params_m5_supported(&t6050c));
   assert(!agx_params_driver_supported(&t6050c));

   bad = t6050c;
   bad.num_clusters_total = 2;
   assert(!agx_params_m5_supported(&bad));

   bad = t6050c;
   bad.chip_id = 0x6051;
   assert(!agx_params_m5_supported(&bad));

   bad = t6050c;
   bad.num_dies = 2;
   assert(!agx_params_m5_supported(&bad));

   bad = t6050c;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_m5_supported(&bad));

   /* M5 Pro's exact shipping topology is admitted without an environment
    * override. Another topology remains unsupported. */
   struct drm_asahi_params_global t6050s = g15_params('S', 0x6050, 2);
   t6050s.gpu_generation = 17;
   t6050s.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
   assert(agx_chip_from_params(&t6050s) == AGX_CHIP_G17S);
   assert(agx_params_m5_supported(&t6050s));

   bad = t6050s;
   bad.num_dies = 2;
   assert(!agx_params_m5_supported(&bad));
   bad = t6050s;
   bad.chip_id = 0x6051;
   assert(!agx_params_m5_supported(&bad));
   bad = t6050s;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_m5_supported(&bad));
   bad = t6050s;
   bad.usc_generation = 2;
   assert(!agx_params_m5_supported(&bad));

   /* Four clusters do not describe an M5 Pro. */
   bad = t6050c;
   bad.gpu_variant = 'S';
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G17S);
   assert(!agx_params_m5_supported(&bad));

   /* The M5 Ultra (T6050, G17D) is admitted with its two dies and eight
    * clusters only. */
   struct drm_asahi_params_global t6050d = g15_params('D', 0x6050, 8);
   t6050d.gpu_generation = 17;
   t6050d.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
   t6050d.num_dies = 2;
   assert(agx_chip_from_params(&t6050d) == AGX_CHIP_G17D);
   assert(agx_chip_g17_m5(AGX_CHIP_G17D));
   assert(agx_params_m5_supported(&t6050d));
   assert(!agx_params_driver_supported(&t6050d));

   bad = t6050d;
   bad.num_dies = 1;
   assert(!agx_params_m5_supported(&bad));
   bad = t6050d;
   bad.num_clusters_total = 4;
   assert(!agx_params_m5_supported(&bad));
   bad = t6050d;
   bad.chip_id = 0x6051;
   assert(!agx_params_m5_supported(&bad));
   bad = t6050d;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_m5_supported(&bad));

   /* Two dies describe neither an M5 Pro nor an M5 Max. */
   bad = t6050d;
   bad.gpu_variant = 'C';
   bad.num_clusters_total = 4;
   assert(!agx_params_m5_supported(&bad));
   bad.num_dies = 1;
   assert(agx_params_m5_supported(&bad));

   /* GL runs through zink on the M5 Ultra and on no other chip. */
   for (unsigned c = 0; c <= AGX_CHIP_UNKNOWN; ++c)
      assert(agx_chip_gl_through_zink(c) == (c == AGX_CHIP_G17D));

   /* The M5 (T8142, G17G) is admitted by its exact kernel identity. */
   struct drm_asahi_params_global m5 = {
      .gpu_generation = 17,
      .gpu_variant = 'G',
      .chip_id = 0x8142,
      .num_dies = 1,
      .num_clusters_total = 1,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
   };
   assert(!agx_params_driver_supported(&m5));
   assert(agx_params_m5_supported(&m5));
   m5.chip_id = 0x8140;
   assert(!agx_params_m5_supported(&m5));
   m5.chip_id = 0x8142;
   m5.gpu_variant = 'P';
   assert(!agx_params_m5_supported(&m5));
   m5.gpu_variant = 'G';
   m5.num_dies = 2;
   assert(!agx_params_m5_supported(&m5));
   m5.num_dies = 1;
   m5.num_clusters_total = 2;
   assert(!agx_params_m5_supported(&m5));
   m5.num_clusters_total = 1;
   m5.usc_generation = 2;
   assert(!agx_params_m5_supported(&m5));
   m5.usc_generation = 3;
   m5.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_m5_supported(&m5));
   assert(!agx_params_m5_supported(NULL));

   struct drm_asahi_params_global m4 = t6040;
   m4.gpu_variant = 'G';
   m4.chip_id = 0x8132;
   m4.num_clusters_total = 1;
   assert(agx_chip_driver_supported(AGX_CHIP_G16G));
   assert(agx_params_driver_supported(&m4));

   /* T8122 (G15G) is admitted by default for its exact identity and
    * supported legacy firmware/kernel contract. */
   struct drm_asahi_params_global g15g = g15_params('G', 0x8122, 1);
   assert(agx_chip_from_params(&g15g) == AGX_CHIP_G15G);
   assert(agx_params_driver_supported(&g15g));
   assert(!agx_params_native_kernel_abi_supported(&g15g));
   /* m3_drm.rs Backend::feature_flags: the Air runtime does not advertise
    * PER_COMMAND_INPUT_SYNCS or EXACT_PRIOR_BARRIERS. */
   g15g.features = DRM_ASAHI_FEATURE_VM_STATUS |
                   DRM_ASAHI_FEATURE_SCHEDULED_QUEUES;
   assert(agx_params_native_kernel_abi_supported(&g15g));
   bad = g15g;
   bad.features &= ~DRM_ASAHI_FEATURE_VM_STATUS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = g15g;
   bad.features &= ~DRM_ASAHI_FEATURE_SCHEDULED_QUEUES;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad = g15g;
   bad.features |= DRM_ASAHI_FEATURE_COMPUTE_WIDE_VISIBILITY;
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad = g15g;
   bad.features = (UINT64_C(1) << 3) | (UINT64_C(1) << 5);
   assert(!agx_params_native_kernel_abi_supported(&bad));
   /* The shared g16 backend on T8122: VM status and independent queues
    * plus the native contract, no scheduled queues (J615 feature word). */
   bad = g15g;
   bad.features = UINT64_C(0x10260005c06);
   assert(agx_params_native_kernel_abi_supported(&bad));
   bad.features &= ~DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad.features = UINT64_C(0x10260005c06) &
                  ~DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS;
   assert(!agx_params_native_kernel_abi_supported(&bad));
   bad.features = UINT64_C(0x10260005c06) & ~DRM_ASAHI_FEATURE_VM_STATUS;
   assert(!agx_params_native_kernel_abi_supported(&bad));

   bad = g15g;
   bad.chip_id = 0x6030;
   assert(!agx_params_driver_supported(&bad));

   bad = g15g;
   bad.num_clusters_total = 2;
   assert(!agx_params_driver_supported(&bad));

   struct drm_asahi_params_global modern = g15g;
   modern.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   modern.gpu_revision = 0x20;
   assert(agx_params_driver_supported(&modern));
   assert(agx_params_native_kernel_abi_supported(&modern));
   enum agx_chip modern_chip = agx_chip_from_params(&modern);
   assert(modern_chip == AGX_CHIP_G15G);
   assert(agx_chip_command_stream_generation(modern_chip) ==
          AGX_CS_GENERATION_G15);
   assert(agx_chip_sampler_generation(modern_chip) ==
          AGX_SAMPLER_GENERATION_GEN4);
   assert(agx_chip_sampler_heap_element_size(modern_chip) == 32);
   assert(agx_chip_isa_encoding(modern_chip) ==
          agx_chip_isa_encoding(agx_chip_from_params(&g15g)));
   assert(agx_chip_cdm_wide_visibility(modern_chip));
   assert(!agx_chip_sampler_has_lod_bias(modern_chip));
   modern.core_masks[0] = 0x3cf;
   assert(agx_params_driver_supported(&modern));
   modern.core_masks[0] = 0x3ff;

   bad = modern;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.gpu_revision = 0x21;
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.num_cores_per_cluster = 9;
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.core_masks[0] = 0x1ff;
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.core_masks[0] = 0x7ff;
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.core_masks[1] = 1;
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.gpu_variant = 'S';
   assert(!agx_params_driver_supported(&bad));
   bad = modern;
   bad.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                  DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS;
   assert(!agx_params_native_kernel_abi_supported(&bad));

   bad = g15g;
   bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));

   /* T8122 (G15G): the heap and its sparse alias stay below 2^39, the
    * absolute USC Texture/Sampler bound, and clear of every per-VM
    * reservation of the kernel's T8122 runtime. */
   uint64_t start, end;
   native_user_window(AGX_CHIP_G15G, &start, &end);
   assert(start == UINT64_C(0x4000000000) && end == UINT64_C(0x6000000000));
   assert(end <= UINT64_C(1) << 39);
   assert(!overlaps(start, end, AGX_T8122_RUNTIME_RESERVED_START,
                    AGX_T8122_RUNTIME_RESERVED_END));
   assert(!overlaps(start, end, UINT64_C(0x1000000000),
                    UINT64_C(0x1008000000)));
   assert(!overlaps(start, end, UINT64_C(0x1074000000),
                    UINT64_C(0x1074028000)));
   assert(!overlaps(start, end, AGX_T8122_RUNTIME_WINDOW_HIGH,
                    AGX_T8122_RUNTIME_WINDOW_END));

   /* G15S keeps the G17P-derived limit, and non-native chips the kernel's. */
   assert(agx_va_user_limit(AGX_CHIP_G15S, true, 1) == AGX_G17P_USER_LIMIT);
   assert(agx_va_user_limit(AGX_CHIP_G13G, false, 1234) == 1234);

   printf("chip admission: ok\n");
   return 0;
}
