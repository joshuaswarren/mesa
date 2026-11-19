/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_chip.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(struct drm_asahi_params_global) == 616,
               "drm_asahi_params_global ABI");
_Static_assert(offsetof(struct drm_asahi_params_global, usc_generation) == 592,
               "USC generation ABI");
_Static_assert(offsetof(struct drm_asahi_params_global, gpu_hal_generation) ==
                  596,
               "GPU HAL generation ABI");
_Static_assert(offsetof(struct drm_asahi_params_global, memory_limit) == 600,
               "memory limit ABI");
_Static_assert(offsetof(struct drm_asahi_params_global,
                        client_memory_limit) == 608,
               "client memory limit ABI");
_Static_assert(DRM_ASAHI_GPU_HAL_LEGACY == 0, "legacy HAL ABI");
_Static_assert(DRM_ASAHI_GPU_HAL_200 == 200, "HAL200 ABI");
_Static_assert(DRM_ASAHI_GPU_HAL_300 == 300, "HAL300 ABI");

static unsigned failures;
static unsigned checks;

#define CHECK_EQ(actual, expected)                                             \
   do {                                                                        \
      unsigned value = (unsigned)(actual);                                     \
      checks++;                                                                \
      if (value != (unsigned)(expected)) {                                     \
         fprintf(stderr, "%s:%d: got %u, expected %u\n", __FILE__, __LINE__,   \
                 value, (unsigned)(expected));                                 \
         failures++;                                                           \
      }                                                                        \
   } while (0)

#define CHECK(condition)                                                       \
   do {                                                                        \
      checks++;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

struct chip_case {
   uint32_t generation;
   uint32_t variant;
   uint32_t clusters;
   enum agx_chip chip;
   enum agx_isa isa;
   enum agx_isa_encoding encoding;
   const char *name;
   bool ray_tracing;
   bool scheduling_stalls;
};

/*
 * Exact driver-core identities recovered from Apple's readChipInfo table.
 * P/A are not Mac performance tiers: G15P is the A16 AGX2 part, while G17P
 * (A18 Pro) and G17A stay on the AGX3b/HAL200 side of the M5 boundary.
 */
static const struct chip_case chip_cases[] = {
   {13, 'G', 1, AGX_CHIP_G13G, AGX_ISA_AGX2, AGX_ISA_ENCODING_AGX2,
    "G13G", false, false},
   {13, 'S', 2, AGX_CHIP_G13X, AGX_ISA_AGX2, AGX_ISA_ENCODING_AGX2,
    "G13X", false, false},
   {14, 'G', 1, AGX_CHIP_G14G, AGX_ISA_AGX2, AGX_ISA_ENCODING_AGX2,
    "G14G", false, false},
   {14, 'C', 4, AGX_CHIP_G14X, AGX_ISA_AGX2, AGX_ISA_ENCODING_AGX2,
    "G14X", false, false},

   {15, 'P', 1, AGX_CHIP_G15P, AGX_ISA_AGX2, AGX_ISA_ENCODING_AGX2,
    "G15P", false, false},
   {15, 'G', 1, AGX_CHIP_G15G, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G15G", true, true},
   {15, 'S', 2, AGX_CHIP_G15S, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G15S", true, false},
   {15, 'C', 4, AGX_CHIP_G15C, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G15C", true, false},
   {15, 'D', 8, AGX_CHIP_G15D, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G15D", true, false},

   {16, 'P', 1, AGX_CHIP_G16P, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G16P", true, false},
   {16, 'G', 1, AGX_CHIP_G16G, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G16G", true, false},
   {16, 'S', 2, AGX_CHIP_G16S, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G16S", true, false},
   {16, 'C', 4, AGX_CHIP_G16C, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G16C", true, false},

   {17, 'P', 1, AGX_CHIP_G17P, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G17P", true, false},
   {17, 'A', 1, AGX_CHIP_G17A, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3B,
    "G17A", true, false},
   {17, 'G', 1, AGX_CHIP_G17G, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3C,
    "G17G", true, false},
   {17, 'S', 2, AGX_CHIP_G17S, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3C,
    "G17S", true, false},
   {17, 'C', 4, AGX_CHIP_G17C, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3C,
    "G17C", true, false},

   /* M6 (J873): the third AGX3 encoding. */
   {18, 'G', 1, AGX_CHIP_G18G, AGX_ISA_AGX3, AGX_ISA_ENCODING_AGX3D,
    "G18G", true, false},
};

static enum agx_chip
chip_for(uint32_t generation, uint32_t variant, uint32_t clusters)
{
   struct drm_asahi_params_global params = {0};
   params.gpu_generation = generation;
   params.gpu_variant = variant;
   params.num_clusters_total = clusters;
   return agx_chip_from_params(&params);
}

static void
test_recovered_table(void)
{
   bool seen[AGX_CHIP_UNKNOWN + 1] = {0};

   for (unsigned i = 0; i < sizeof(chip_cases) / sizeof(chip_cases[0]); ++i) {
      const struct chip_case *c = &chip_cases[i];

      CHECK_EQ(chip_for(c->generation, c->variant, c->clusters), c->chip);
      CHECK_EQ(agx_chip_isa(c->chip), c->isa);
      CHECK_EQ(agx_chip_isa_encoding(c->chip), c->encoding);
      CHECK(strcmp(agx_chip_name(c->chip), c->name) == 0);
      CHECK(agx_chip_has_hardware_ray_tracing(c->chip) == c->ray_tracing);
      CHECK(agx_chip_needs_scheduling_stalls(c->chip) ==
            c->scheduling_stalls);
      /* Every named chip has a known encoding. Every encoding but AGX3d has
       * a packer: AGX2 through agx_pack.c, AGX3b/AGX3c through
       * agx3_pack_binary.c. */
      CHECK(c->encoding != AGX_ISA_ENCODING_UNKNOWN);
      CHECK(agx_chip_codegen_supported(c->chip) ==
            (c->encoding != AGX_ISA_ENCODING_AGX3D));
      CHECK(!seen[c->chip]);
      seen[c->chip] = true;
   }

   /* Every named enum must have exactly one row above. */
   for (int chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip)
      CHECK(seen[chip]);
}

static void
test_variant_is_load_bearing(void)
{
   /* Cluster count must never be used as a substitute for gpu_variant. */
   CHECK_EQ(chip_for(17, 'P', 4), AGX_CHIP_G17P);
   CHECK_EQ(chip_for(17, 'G', 4), AGX_CHIP_G17G);
   CHECK_EQ(chip_for(15, 'P', 2), AGX_CHIP_G15P);
   CHECK_EQ(chip_for(15, 'S', 1), AGX_CHIP_G15S);

   CHECK_EQ(agx_chip_isa(AGX_CHIP_G15P), AGX_ISA_AGX2);
   CHECK_EQ(agx_chip_isa_encoding(AGX_CHIP_G17P),
            AGX_ISA_ENCODING_AGX3B);
   CHECK_EQ(agx_chip_isa_encoding(AGX_CHIP_G17A),
            AGX_ISA_ENCODING_AGX3B);
   CHECK_EQ(agx_chip_isa_encoding(AGX_CHIP_G17G),
            AGX_ISA_ENCODING_AGX3C);
   CHECK_EQ(chip_for(18, 'G', 4), AGX_CHIP_G18G);
   CHECK_EQ(agx_chip_isa_encoding(AGX_CHIP_G18G),
            AGX_ISA_ENCODING_AGX3D);
}

static void
test_independent_generation_axes(void)
{
   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G13G),
            AGX_CS_GENERATION_AGX2);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G13G),
            AGX_DESCRIPTOR_GENERATION_7);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G14X),
            AGX_DESCRIPTOR_GENERATION_8);
   CHECK_EQ(agx_chip_sampler_generation(AGX_CHIP_G14X),
            AGX_SAMPLER_GENERATION_GEN4);
   CHECK_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G14X), 8);

   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G15G),
            AGX_CS_GENERATION_G15);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G15D),
            AGX_DESCRIPTOR_GENERATION_9);
   CHECK_EQ(agx_chip_sampler_generation(AGX_CHIP_G15D),
            AGX_SAMPLER_GENERATION_GEN4);
   CHECK_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G15D), 32);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G15D),
            AGX_SUBMISSION_MODEL_CLASSIC);

   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G16G),
            AGX_CS_GENERATION_HAL200);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G16S),
            AGX_DESCRIPTOR_GENERATION_9);
   CHECK_EQ(agx_chip_sampler_generation(AGX_CHIP_G16S),
            AGX_SAMPLER_GENERATION_GEN4_HAL200);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G16G),
            AGX_SUBMISSION_MODEL_CLASSIC);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G16S),
            AGX_SUBMISSION_MODEL_DUAL);

   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G17P),
            AGX_CS_GENERATION_HAL200);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G17P),
            AGX_DESCRIPTOR_GENERATION_9);
   CHECK_EQ(agx_chip_sampler_generation(AGX_CHIP_G17P),
            AGX_SAMPLER_GENERATION_GEN4_HAL200);
   CHECK_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G17P), 32);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G17P),
            AGX_SUBMISSION_MODEL_DUAL);

   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G17G),
            AGX_CS_GENERATION_HAL300);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G17S),
            AGX_DESCRIPTOR_GENERATION_10);
   CHECK_EQ(agx_chip_sampler_generation(AGX_CHIP_G17C),
            AGX_SAMPLER_GENERATION_GEN5);
   CHECK_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G17S), 32);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G17G),
            AGX_SUBMISSION_MODEL_SKSM);

   /* G18G writes the M5 records with a new shader encoding. */
   CHECK_EQ(agx_chip_isa(AGX_CHIP_G18G), AGX_ISA_AGX3);
   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G18G),
            AGX_CS_GENERATION_HAL300);
   CHECK_EQ(agx_chip_descriptor_generation(AGX_CHIP_G18G),
            AGX_DESCRIPTOR_GENERATION_10);
   CHECK_EQ(agx_chip_sampler_generation(AGX_CHIP_G18G),
            AGX_SAMPLER_GENERATION_GEN5);
   CHECK_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G18G), 32);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G18G),
            AGX_SUBMISSION_MODEL_SKSM);
   CHECK(agx_chip_has_hardware_ray_tracing(AGX_CHIP_G18G));
   CHECK(!agx_chip_needs_scheduling_stalls(AGX_CHIP_G18G));
   CHECK(!agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_G18G));
   CHECK(agx_chip_exact_product_name(AGX_CHIP_G18G) == NULL);

   /* The witnessed HAL300 record set is G17G and G18G only. */
   for (int chip = AGX_CHIP_G13G; chip <= AGX_CHIP_UNKNOWN; ++chip) {
      const enum agx_chip c = (enum agx_chip)chip;
      CHECK(agx_chip_hal300_records(c) ==
            (c == AGX_CHIP_G17G || c == AGX_CHIP_G18G));
      if (agx_chip_hal300_records(c))
         CHECK_EQ(agx_chip_command_stream_generation(c),
                  AGX_CS_GENERATION_HAL300);
   }

   /* These phone identities are useful for selecting the ISA, but the Mac
    * driver corpus did not contain enough evidence to select every other
    * driver axis. They remain outside the runnable device set. */
   CHECK_EQ(agx_chip_command_stream_generation(AGX_CHIP_G15P),
            AGX_CS_GENERATION_UNKNOWN);
   CHECK_EQ(agx_chip_sampler_heap_element_size(AGX_CHIP_G15P), 0);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G16P),
            AGX_SUBMISSION_MODEL_UNKNOWN);
   CHECK_EQ(agx_chip_submission_model(AGX_CHIP_G17A),
            AGX_SUBMISSION_MODEL_UNKNOWN);

   CHECK(agx_chip_driver_supported(AGX_CHIP_G13G));
   CHECK(agx_chip_driver_supported(AGX_CHIP_G14X));

   /* Codegen and device support are separate axes. G17P has the exact
    * T8140/HAL200 path; the other AGX3 identities have encoders but not a
    * complete Honeykrisp device path. */
   CHECK(agx_chip_codegen_supported(AGX_CHIP_G15P));
   CHECK(agx_chip_codegen_supported(AGX_CHIP_G15G));
   CHECK(agx_chip_codegen_supported(AGX_CHIP_G17P));
   CHECK(agx_chip_codegen_supported(AGX_CHIP_G17S));
   CHECK(!agx_chip_codegen_supported(AGX_CHIP_UNKNOWN));
   /* No agx3d table exists, so G18G has no code generator and therefore no
    * device path, whatever its command-stream axes say. */
   CHECK(!agx_chip_codegen_supported(AGX_CHIP_G18G));
   CHECK(!agx_chip_driver_supported(AGX_CHIP_G18G));

   CHECK(!agx_chip_driver_supported(AGX_CHIP_G15P));
   CHECK(agx_chip_driver_supported(AGX_CHIP_G15G));
   CHECK(agx_chip_driver_supported(AGX_CHIP_G17P));
   CHECK(!agx_chip_driver_supported(AGX_CHIP_G17S));
   CHECK(!agx_chip_driver_supported(AGX_CHIP_UNKNOWN));

   struct drm_asahi_params_global g17p = {
      .gpu_generation = 17,
      .gpu_variant = 'P',
      .chip_id = 0x8140,
      .num_dies = 1,
      .num_clusters_total = 1,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
   };
   CHECK(agx_params_driver_supported(&g17p));
   CHECK(strcmp(agx_chip_exact_product_name(AGX_CHIP_G17P), "A18 Pro") ==
         0);
   CHECK(agx_chip_exact_product_name(AGX_CHIP_G17G) == NULL);
   g17p.chip_id = 0x8141;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.chip_id = 0x8140;
   g17p.num_dies = 2;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.num_dies = 1;
   g17p.num_clusters_total = 2;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.num_clusters_total = 1;
   g17p.usc_generation = 2;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.usc_generation = 3;
   g17p.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
   CHECK(!agx_params_driver_supported(&g17p));
   CHECK(!agx_params_driver_supported(NULL));

   struct drm_asahi_params_global m5 = {
      .gpu_generation = 17,
      .gpu_variant = 'G',
      .chip_id = 0x8142,
      .num_dies = 1,
      .num_clusters_total = 1,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
   };
   CHECK(!agx_params_driver_supported(&m5));
   CHECK(agx_params_m5_supported(&m5));
   m5.chip_id = 0x8140;
   CHECK(!agx_params_m5_supported(&m5));
   m5.chip_id = 0x8142;
   m5.gpu_variant = 'P';
   CHECK(!agx_params_m5_supported(&m5));
   m5.gpu_variant = 'G';
   m5.num_dies = 2;
   CHECK(!agx_params_m5_supported(&m5));
   m5.num_dies = 1;
   m5.num_clusters_total = 2;
   CHECK(!agx_params_m5_supported(&m5));
   m5.num_clusters_total = 1;
   m5.usc_generation = 2;
   CHECK(!agx_params_m5_supported(&m5));
   m5.usc_generation = 3;
   m5.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   CHECK(!agx_params_m5_supported(&m5));
   CHECK(!agx_params_m5_supported(NULL));

   /* The M5 bring-up contract must not admit the M6. */
   struct drm_asahi_params_global m6 = m5;
   m6.gpu_generation = 18;
   m6.chip_id = 0x8152;
   CHECK_EQ(agx_chip_from_params(&m6), AGX_CHIP_G18G);
   CHECK(!agx_params_m5_supported(&m6));
   CHECK(!agx_params_driver_supported(&m6));

   CHECK(agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_G13G));
   CHECK(agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_G14X));
   CHECK(!agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_G17P));
   CHECK(!agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_UNKNOWN));

   CHECK(strcmp(agx_command_stream_generation_name(
                   AGX_CS_GENERATION_HAL200),
                "HAL200") == 0);
   CHECK(strcmp(agx_command_stream_generation_name(
                   AGX_CS_GENERATION_HAL300),
                "HAL300") == 0);
}

static void
test_invalid_identity(void)
{
   CHECK_EQ(chip_for(12, 'G', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(18, 'P', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(18, 'S', 2), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(19, 'G', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(15, 'A', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(16, 'D', 8), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(17, 'D', 8), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(13, 'P', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(13, 'G', 0), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(17, 'S', DRM_ASAHI_MAX_CLUSTERS + 1),
            AGX_CHIP_UNKNOWN);
}

static void
test_classification_invariants(void)
{
   CHECK_EQ(AGX_ISA_AGX2, 2);
   CHECK_EQ(AGX_ISA_AGX3, 3);

   for (int chip = AGX_CHIP_G13G; chip <= AGX_CHIP_UNKNOWN; ++chip) {
      const enum agx_chip c = (enum agx_chip)chip;
      const enum agx_isa isa = agx_chip_isa(c);
      const enum agx_isa_encoding encoding = agx_chip_isa_encoding(c);
      const bool known = c != AGX_CHIP_UNKNOWN;
      const bool agx3_encoding = encoding == AGX_ISA_ENCODING_AGX3B ||
                                 encoding == AGX_ISA_ENCODING_AGX3C ||
                                 encoding == AGX_ISA_ENCODING_AGX3D;

      CHECK(strcmp(agx_chip_name(c), "") != 0);
      CHECK((isa != AGX_ISA_UNKNOWN) == known);
      CHECK((encoding != AGX_ISA_ENCODING_UNKNOWN) == known);
      CHECK(agx3_encoding == (isa == AGX_ISA_AGX3));
      CHECK((encoding == AGX_ISA_ENCODING_AGX2) == (isa == AGX_ISA_AGX2));
      CHECK(agx_chip_has_hardware_ray_tracing(c) == (isa == AGX_ISA_AGX3));
   }

   CHECK(strcmp(agx_isa_name(AGX_ISA_AGX3), "AGX3") == 0);
   CHECK(strcmp(agx_chip_name(AGX_CHIP_G18G), "G18G") == 0);
   CHECK(strcmp(agx_chip_name(AGX_CHIP_UNKNOWN), "unknown") == 0);
}

/* The inline lookups answer every per-draw query; they must match the
 * exhaustive reference switches for every identity, including UNKNOWN and
 * out-of-range values. */
static void
test_inline_lookups_match_reference(void)
{
   for (int chip = AGX_CHIP_G13G; chip <= AGX_CHIP_UNKNOWN + 3; ++chip) {
      const enum agx_chip c = (enum agx_chip)chip;
      CHECK_EQ(agx_chip_command_stream_generation(c),
               agx_chip_command_stream_generation_reference(c));
      CHECK_EQ(agx_chip_descriptor_generation(c),
               agx_chip_descriptor_generation_reference(c));
      CHECK_EQ(agx_chip_sampler_generation(c),
               agx_chip_sampler_generation_reference(c));
      CHECK_EQ(agx_chip_sampler_heap_element_size(c),
               agx_chip_sampler_heap_element_size_reference(c));
   }
   CHECK_EQ(agx_chip_command_stream_generation((enum agx_chip)-1),
            AGX_CS_GENERATION_UNKNOWN);
   CHECK_EQ(agx_chip_descriptor_generation((enum agx_chip)-1),
            AGX_DESCRIPTOR_GENERATION_UNKNOWN);
}

int
main(void)
{
   test_recovered_table();
   test_variant_is_load_bearing();
   test_independent_generation_axes();
   test_invalid_identity();
   test_classification_invariants();
   test_inline_lookups_match_reference();

   if (failures) {
      fprintf(stderr, "agx_chip: %u of %u checks failed\n", failures, checks);
      return 1;
   }

   printf("agx_chip: all %u checks passed\n", checks);
   return 0;
}
