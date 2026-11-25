/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_chip.h"

#include <stddef.h>

#include "util/bitscan.h"
#include "util/macros.h"

enum agx_chip
agx_chip_from_params(const struct drm_asahi_params_global *params)
{
   if (params->num_clusters_total == 0 ||
       params->num_clusters_total > DRM_ASAHI_MAX_CLUSTERS)
      return AGX_CHIP_UNKNOWN;

   return agx_chip_from_identity(params);
}

bool
agx_chip_codegen_supported(enum agx_chip chip)
{
   /*
    * Two independent packers exist, one per encoding family:
    *
    *   AGX2         -> agx_pack.c            (AGX2.xml)
    *   AGX3b/AGX3c  -> agx3_pack_binary.c    (agx3_compose/, AGX3{b,c}.xml)
    *
    * The AGX3 side is a real encoder, not a decoder: agx3_compose
    * reproduces Apple's instructions byte-exactly, and agx3_pack_binary
    * emits compute, vertex and fragment shaders end to end.
    *
    * "Has a code generator" is all this predicate claims. Instruction
    * coverage is explicit: an AGX3 shader outside the grounded domain makes
    * agx_compile_shader_nir return false rather than emit wrong code. The
    * runnable device set is tracked separately by
    * agx_chip_driver_supported.
    *
    * UNKNOWN remains unsupported: without an encoding there is no packer.
    */
   const enum agx_isa_encoding encoding = agx_chip_isa_encoding(chip);

   return encoding == AGX_ISA_ENCODING_AGX2 ||
          agx_isa_is_agx3(encoding);
}

bool
agx_chip_driver_supported(enum agx_chip chip)
{
   /*
    * Codegen is necessary, not sufficient. An AGX3 chip is admitted only on
    * the native path, and only with the exact axes that path is wired for:
    * Gen9 descriptors in a 32-byte sampler heap, with HAL200 records and
    * samplers for G17P and G16G, or G15 records and Gen4 samplers for G15S
    * and G15C. G17P uses dual queues; the other native profiles use the
    * classic queue model. Other AGX3 identities still lack at least one
    * of those pieces.
    */
   if (!agx_chip_codegen_supported(chip))
      return false;

   if (agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_AGX2) {
      return agx_chip_descriptor_generation(chip) !=
                AGX_DESCRIPTOR_GENERATION_UNKNOWN &&
             agx_chip_sampler_generation(chip) !=
                AGX_SAMPLER_GENERATION_UNKNOWN;
   }

   if (!agx_chip_native_agx3(chip) ||
       agx_chip_descriptor_generation(chip) != AGX_DESCRIPTOR_GENERATION_9 ||
       agx_chip_sampler_heap_element_size(chip) != 32)
      return false;

   switch (chip) {
   case AGX_CHIP_G17P:
      return agx_chip_command_stream_generation(chip) ==
                AGX_CS_GENERATION_HAL200 &&
             agx_chip_sampler_generation(chip) ==
                AGX_SAMPLER_GENERATION_GEN4_HAL200 &&
             agx_chip_submission_model(chip) == AGX_SUBMISSION_MODEL_DUAL;
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
      return agx_chip_command_stream_generation(chip) ==
                AGX_CS_GENERATION_G15 &&
             agx_chip_sampler_generation(chip) ==
                AGX_SAMPLER_GENERATION_GEN4 &&
             agx_chip_submission_model(chip) == AGX_SUBMISSION_MODEL_CLASSIC;
   case AGX_CHIP_G16G:
      return agx_chip_command_stream_generation(chip) ==
                AGX_CS_GENERATION_HAL200 &&
             agx_chip_sampler_generation(chip) ==
                AGX_SAMPLER_GENERATION_GEN4_HAL200 &&
             agx_chip_submission_model(chip) == AGX_SUBMISSION_MODEL_CLASSIC;
   default:
      return false;
   }
}

/* T8122's HAL200 firmware is admitted only on the revision and core layout
 * it was measured with: one ten-core cluster with eight or ten cores enabled.
 */
static bool
agx_t8122_hal200_check(const struct drm_asahi_params_global *params)
{
   if (params->gpu_revision != 0x20 || params->num_cores_per_cluster != 10)
      return false;

   const uint64_t mask = params->core_masks[0];
   const unsigned cores = util_bitcount64(mask);
   if ((mask >> 10) || (cores != 8 && cores != 10))
      return false;
   for (unsigned i = 1; i < DRM_ASAHI_MAX_CLUSTERS; ++i) {
      if (params->core_masks[i])
         return false;
   }

   return true;
}

/* Which device-open path admits a SoC. */
enum agx_soc_contract {
   /* The native AGX3 path: agx_params_driver_supported(). */
   AGX_SOC_NATIVE,
   /* The M5 path: agx_params_m5_supported(). */
   AGX_SOC_M5,
   /* M6 bring-up: agx_params_g18g_bringup_supported(). */
   AGX_SOC_G18G_BRINGUP,
};

/*
 * The AGX3 SoCs the driver admits, one row per kernel contract. A chip name
 * such as G17P covers several SoCs, and recognising it is not enough: each
 * SoC is admitted only with the topology and firmware interface its rows
 * were measured on. Every admitted SoC executes AGX3 shaders.
 */
struct agx_soc {
   /* drm_asahi_params_global::chip_id */
   uint32_t chip_id;
   /* What agx_chip_from_params() names the reported identity. */
   enum agx_chip chip;
   uint32_t dies;
   /* 0 accepts any cluster count agx_chip_from_params() accepts. */
   uint32_t clusters;
   /* enum drm_asahi_gpu_hal_generation */
   uint32_t hal;
   enum agx_soc_contract contract;
   /* Further checks of the kernel's report, or NULL. */
   bool (*check)(const struct drm_asahi_params_global *params);
};

static const struct agx_soc agx_socs[] = {
   /* T8122 (M3). The firmware interface version does not determine the
    * record layout: both interfaces take G15 records and Gen4 samplers, and
    * the chip stays G15G so scheduling and hardware workarounds agree. */
   {0x8122, AGX_CHIP_G15G, 1, 1, DRM_ASAHI_GPU_HAL_LEGACY, AGX_SOC_NATIVE,
    NULL},
   {0x8122, AGX_CHIP_G15G, 1, 1, DRM_ASAHI_GPU_HAL_200, AGX_SOC_NATIVE,
    agx_t8122_hal200_check},

   /* T6030 (M3 Pro) and T6031 (M3 Max), AGX3 shaders on the legacy firmware
    * interface. Binned parts differ only in their core masks, which every
    * consumer reads from the parameters, so the masks are not checked. */
   {0x6030, AGX_CHIP_G15S, 1, 2, DRM_ASAHI_GPU_HAL_LEGACY, AGX_SOC_NATIVE,
    NULL},
   {0x6031, AGX_CHIP_G15C, 1, 4, DRM_ASAHI_GPU_HAL_LEGACY, AGX_SOC_NATIVE,
    NULL},

   /* T8132 (M4). */
   {0x8132, AGX_CHIP_G16G, 1, 1, DRM_ASAHI_GPU_HAL_200, AGX_SOC_NATIVE, NULL},

   /* G17P is the A18 Pro, and the M4 Pro and M4 Max on the G17 kernel
    * runtime. The shared chip name must not admit another SoC or topology;
    * a binned M4 Pro keeps two clusters. */
   {AGX_CHIP_ID_A18_PRO, AGX_CHIP_G17P, 1, 1, DRM_ASAHI_GPU_HAL_200,
    AGX_SOC_NATIVE, NULL},
   {AGX_CHIP_ID_M4_PRO, AGX_CHIP_G17P, 1, 2, DRM_ASAHI_GPU_HAL_200,
    AGX_SOC_NATIVE, NULL},
   {AGX_CHIP_ID_M4_MAX, AGX_CHIP_G17P, 1, 4, DRM_ASAHI_GPU_HAL_200,
    AGX_SOC_NATIVE, NULL},

   /* T8142 (M5), and T6050 as the M5 Pro (J714s, J716s), the M5 Max (J716c),
    * and the M5 Ultra (J775D). */
   {0x8142, AGX_CHIP_G17G, 1, 1, DRM_ASAHI_GPU_HAL_300, AGX_SOC_M5, NULL},
   {0x6050, AGX_CHIP_G17S, 1, 2, DRM_ASAHI_GPU_HAL_300, AGX_SOC_M5, NULL},
   {0x6050, AGX_CHIP_G17C, 1, 4, DRM_ASAHI_GPU_HAL_300, AGX_SOC_M5, NULL},
   {0x6050, AGX_CHIP_G17D, 2, 8, DRM_ASAHI_GPU_HAL_300, AGX_SOC_M5, NULL},

   /* T8152 (M6, J873). The shim's G18G profile and the J873 G18G kernel
    * driver both publish HAL 300 until the uAPI has a PI_400 value. */
   {0x8152, AGX_CHIP_G18G, 1, 0, DRM_ASAHI_GPU_HAL_300, AGX_SOC_G18G_BRINGUP,
    NULL},
};

/* The row admitting what the kernel reports, or NULL. */
static const struct agx_soc *
agx_soc_lookup(const struct drm_asahi_params_global *params)
{
   if (!params || params->usc_generation != AGX_ISA_AGX3)
      return NULL;

   const enum agx_chip chip = agx_chip_from_params(params);

   for (unsigned i = 0; i < ARRAY_SIZE(agx_socs); ++i) {
      const struct agx_soc *soc = &agx_socs[i];

      if (soc->chip == chip && soc->chip_id == params->chip_id &&
          soc->dies == params->num_dies &&
          (!soc->clusters || soc->clusters == params->num_clusters_total) &&
          soc->hal == params->gpu_hal_generation &&
          (!soc->check || soc->check(params)))
         return soc;
   }

   return NULL;
}

bool
agx_params_driver_supported(const struct drm_asahi_params_global *params)
{
   if (!params)
      return false;

   const enum agx_chip chip = agx_chip_from_params(params);
   if (!agx_chip_driver_supported(chip))
      return false;

   /* AGX2 parts are admitted by their chip. */
   if (agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_AGX2)
      return true;

   const struct agx_soc *soc = agx_soc_lookup(params);
   return soc && soc->contract == AGX_SOC_NATIVE;
}

/* G15G uses the separately admitted T8122 runtime: its baseline advertises
 * VM_STATUS and SCHEDULED_QUEUES, not exact-prior barriers. Other native
 * profiles use the unified high-bit prior-command contract. Refuse a known
 * mismatch before VM creation; these checks are not general ABI negotiation.
 */
bool
agx_params_native_kernel_abi_supported(
   const struct drm_asahi_params_global *params)
{
   if (!params)
      return false;

   const enum agx_chip chip = agx_chip_from_params(params);
   if (!agx_chip_native_agx3(chip))
      return true;

   /* The shared native contract (the g16 backend, and every native AGX3 part). */
   const uint64_t native = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                           DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;

   if (chip == AGX_CHIP_G15G) {
      /* T8122 runs either the older M3 runtime, which advertises VM status
       * and scheduled queues, or the shared g16 backend, which advertises
       * VM status and the native contract. Either complete contract is
       * accepted; VM status is required in both. */
      const uint64_t m3_runtime = DRM_ASAHI_FEATURE_VM_STATUS |
                                  DRM_ASAHI_FEATURE_SCHEDULED_QUEUES;
      const uint64_t g16 = DRM_ASAHI_FEATURE_VM_STATUS | native;
      return (params->features & m3_runtime) == m3_runtime ||
             (params->features & g16) == g16;
   }

   return (params->features & native) == native;
}

const char *
agx_chip_exact_product_name(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G17P:
      return "A18 Pro";
   default:
      return NULL;
   }
}

bool
agx_params_m5_supported(const struct drm_asahi_params_global *params)
{
   const struct agx_soc *soc = agx_soc_lookup(params);
   return soc && soc->contract == AGX_SOC_M5;
}

/* Like the M5 contracts, M6 bring-up is admitted by its exact hardware
 * identity; see agx_open_device. */
bool
agx_params_g18g_bringup_supported(const struct drm_asahi_params_global *params)
{
   const struct agx_soc *soc = agx_soc_lookup(params);
   return soc && soc->contract == AGX_SOC_G18G_BRINGUP;
}

bool
agx_chip_uses_legacy_precompiled_programs(enum agx_chip chip)
{
   return agx_chip_driver_supported(chip) &&
          agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_AGX2;
}

/*
 * Whether the compiler must insert explicit scheduling stalls.
 *
 * The M3 base part needs them and nothing else does: stripping its stall words
 * leaves an instruction stream 99.5% identical in count and 85.1%
 * byte-identical to M4's, with a further 7.4% differing only by branch
 * displacements shifted by exactly the removed padding.
 *
 * This is a property of one chip's scheduler, not of its encoding -- G15G
 * shares the agx3b encoder table with every other variant-0 part. It is a
 * codegen obligation, so it is only meaningful for a chip this driver can
 * generate code for.
 */
bool
agx_chip_has_hardware_ray_tracing(enum agx_chip chip)
{
   return agx_chip_isa(chip) == AGX_ISA_AGX3;
}

bool
agx_chip_needs_scheduling_stalls(enum agx_chip chip)
{
   /* The M3 base part alone. Not a property of its encoding: G15G shares the
    * agx3b table with G15S/C/D, every G16 part, and G17P/A, none of which
    * pad. */
   return agx_chip_info(chip)->quirks & AGX_QUIRK_SCHEDULING_STALLS;
}

const char *
agx_isa_name(enum agx_isa isa)
{
   switch (isa) {
   case AGX_ISA_AGX2: return "AGX2";
   case AGX_ISA_AGX3: return "AGX3";
   case AGX_ISA_UNKNOWN:
   default: return "unknown";
   }
}

const char *
agx_command_stream_generation_name(
   enum agx_command_stream_generation generation)
{
   switch (generation) {
   case AGX_CS_GENERATION_AGX2:
      return "AGX2";
   case AGX_CS_GENERATION_G15:
      return "G15";
   case AGX_CS_GENERATION_HAL200:
      return "HAL200";
   case AGX_CS_GENERATION_HAL300:
      return "HAL300";
   case AGX_CS_GENERATION_UNKNOWN:
   default:
      return "unknown";
   }
}
