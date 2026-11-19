/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_chip.h"

#include <stddef.h>

#include "util/bitscan.h"

enum agx_chip
agx_chip_from_params(const struct drm_asahi_params_global *params)
{
   if (params->num_clusters_total == 0 ||
       params->num_clusters_total > DRM_ASAHI_MAX_CLUSTERS)
      return AGX_CHIP_UNKNOWN;

   /* These are command-stream architecture identities, not an ordering.
    *
    * Generations 15 and up are recognised so diagnostics can name the part and
    * its command-stream generation. agx_chip_driver_supported decides whether
    * the driver proceeds; agx_chip_codegen_supported covers only the shader
    * packer.
    *
    * The variant character is load-bearing. Cluster count cannot distinguish
    * G17P (A18 Pro, AGX3b) from G17G (M5, AGX3c), and G15P is an AGX2 part in
    * the same numeric generation as AGX3 M3. The kernel already publishes the
    * exact P/G/S/C/D/A character, so use it directly.
    */
   switch (params->gpu_generation) {
   case 13:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G13G;
      case 'S':
      case 'C':
      case 'D': return AGX_CHIP_G13X;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 14:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G14G;
      case 'S':
      case 'C':
      case 'D': return AGX_CHIP_G14X;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 15:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G15P;
      case 'G': return AGX_CHIP_G15G;
      case 'S': return AGX_CHIP_G15S;
      case 'C': return AGX_CHIP_G15C;
      case 'D': return AGX_CHIP_G15D;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 16:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G16P;
      case 'G': return AGX_CHIP_G16G;
      case 'S':
         /* The M4 Pro (T6040) on the G17 kernel runtime takes the A18 Pro's
          * contract: its firmware has the A18 Pro firmware's interface, and
          * every G16S axis in these tables (AGX3B, HAL200, Gen9, Gen4-HAL200,
          * 32-byte sampler heap, dual submission) equals G17P's. The M4 Max
          * (T6041) below takes it for the same reason. G17P therefore names
          * the AGX3B + HAL200 profile of three SoCs, not the A18 Pro alone:
          * a performance gate on it needs rows on those parts (see enum
          * agx_chip).
          */
         if (params->chip_id == 0x6040 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200)
            return AGX_CHIP_G17P;
         return AGX_CHIP_G16S;
      case 'C':
         if (params->chip_id == 0x6041 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200)
            return AGX_CHIP_G17P;
         return AGX_CHIP_G16C;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 17:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G17P;
      case 'A': return AGX_CHIP_G17A;
      case 'G': return AGX_CHIP_G17G;
      case 'S': return AGX_CHIP_G17S;
      case 'C': return AGX_CHIP_G17C;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 18:
      /* Only the M6 base part has been observed. G18M and G19P share its
       * shader encoding but nothing else about them is known. */
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G18G;
      default: return AGX_CHIP_UNKNOWN;
      }
   default:
      return AGX_CHIP_UNKNOWN;
   }
}

enum agx_isa
agx_chip_isa(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
   case AGX_CHIP_G15P:
      return AGX_ISA_AGX2;

   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
   case AGX_CHIP_G18G:
      return AGX_ISA_AGX3;

   case AGX_CHIP_UNKNOWN:
   default:
      return AGX_ISA_UNKNOWN;
   }
}

enum agx_isa_encoding
agx_chip_isa_encoding(enum agx_chip chip)
{
   /*
    * Which encoder table a part uses. Apple's own tables decode each family
    * 100% under exactly one of these and essentially 0% under the other, so
    * this is a measurement rather than a reading of the naming scheme.
    *
    * gpu_gen and gpu_var are a string split of the kernel core-name table, so
    * they map directly. P/A are not Mac performance tiers: G15P is AGX2 and
    * G17P/G17A remain on the AGX3b/HAL200 side while the G17 Mac parts use
    * AGX3c/HAL300.
    */
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
   case AGX_CHIP_G15P:
      return AGX_ISA_ENCODING_AGX2;

   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
      return AGX_ISA_ENCODING_AGX3B;

   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return AGX_ISA_ENCODING_AGX3C;

   /* The agx3d encoding table, shared with G18M and G19P. */
   case AGX_CHIP_G18G:
      return AGX_ISA_ENCODING_AGX3D;

   case AGX_CHIP_UNKNOWN:
   default:
      return AGX_ISA_ENCODING_UNKNOWN;
   }
}

enum agx_command_stream_generation
agx_chip_command_stream_generation_reference(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
      return AGX_CS_GENERATION_AGX2;

   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
      return AGX_CS_GENERATION_G15;

   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
      return AGX_CS_GENERATION_HAL200;

   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return AGX_CS_GENERATION_HAL300;

   /* The record format, not the kernel HAL: the J873 native captures decode
    * and repack under the HAL300 codecs. See agx_chip_hal300_records. */
   case AGX_CHIP_G18G:
      return AGX_CS_GENERATION_HAL300;

   /* The A16 compiler target is known to be AGX2, but its command-stream
    * class was not present in the Mac driver corpus. Do not infer it. */
   case AGX_CHIP_G15P:
   case AGX_CHIP_UNKNOWN:
   default:
      return AGX_CS_GENERATION_UNKNOWN;
   }
}

enum agx_descriptor_generation
agx_chip_descriptor_generation_reference(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
      return AGX_DESCRIPTOR_GENERATION_7;
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
      return AGX_DESCRIPTOR_GENERATION_8;

   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
      return AGX_DESCRIPTOR_GENERATION_9;

   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return AGX_DESCRIPTOR_GENERATION_10;

   /* The J873 colour render's PBE descriptor (ffe40a02 00007f80 00005800
    * 0003f010) and its texture descriptor (f6880a02 0007f80f 00005800
    * 000fc010) unpack as Gen10 and repack byte-identically. */
   case AGX_CHIP_G18G:
      return AGX_DESCRIPTOR_GENERATION_10;

   case AGX_CHIP_G15P:
   case AGX_CHIP_UNKNOWN:
   default:
      return AGX_DESCRIPTOR_GENERATION_UNKNOWN;
   }
}

enum agx_sampler_generation
agx_chip_sampler_generation_reference(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
      return AGX_SAMPLER_GENERATION_GEN4;

   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
      return AGX_SAMPLER_GENERATION_GEN4_HAL200;

   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return AGX_SAMPLER_GENERATION_GEN5;

   /* The J873 texture-sample render's sampler (Metal's default descriptor
    * with nearest filters and clamp-to-edge) is 000f0000 00000780 at
    * element 2 of a 32-byte heap. It repacks byte-identically as Gen5 and
    * not as either Gen4 layout, whose maximum-LOD clamp of 14.0 turns the
    * 15.0 there into 000e0000. */
   case AGX_CHIP_G18G:
      return AGX_SAMPLER_GENERATION_GEN5;

   case AGX_CHIP_G15P:
   case AGX_CHIP_UNKNOWN:
   default:
      return AGX_SAMPLER_GENERATION_UNKNOWN;
   }
}

unsigned
agx_chip_sampler_heap_element_size_reference(enum agx_chip chip)
{
   switch (agx_chip_command_stream_generation_reference(chip)) {
   case AGX_CS_GENERATION_AGX2:
      return 8;
   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      return 32;
   case AGX_CS_GENERATION_UNKNOWN:
   default:
      return 0;
   }
}

enum agx_submission_model
agx_chip_submission_model(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
      return AGX_SUBMISSION_MODEL_CLASSIC;

   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
      return AGX_SUBMISSION_MODEL_DUAL;

   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return AGX_SUBMISSION_MODEL_SKSM;

   /* On macOS 26A428 the G18G GPU has only SKSM channels, for 3D, TA
    * and compute work; there is no classic queue. */
   case AGX_CHIP_G18G:
      return AGX_SUBMISSION_MODEL_SKSM;

   /* No corresponding accelerator class was present in the Mac corpus. */
   case AGX_CHIP_G15P:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G17A:
   case AGX_CHIP_UNKNOWN:
   default:
      return AGX_SUBMISSION_MODEL_UNKNOWN;
   }
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
          encoding == AGX_ISA_ENCODING_AGX3B ||
          encoding == AGX_ISA_ENCODING_AGX3C;
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

/* The firmware interface version does not determine the GPU record layout.
 * Both T8122 interfaces consume G15 records and Gen4 sampler words. Keep the
 * physical chip as G15G so compiler scheduling and hardware workarounds agree.
 */
static bool
agx_t8122_params_supported(const struct drm_asahi_params_global *params)
{
   if (params->chip_id != 0x8122 || params->num_dies != 1 ||
       params->num_clusters_total != 1 ||
       params->usc_generation != AGX_ISA_AGX3)
      return false;

   if (params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_LEGACY)
      return true;

   if (params->gpu_hal_generation != DRM_ASAHI_GPU_HAL_200 ||
       params->gpu_revision != 0x20 || params->num_cores_per_cluster != 10)
      return false;

   const uint64_t mask = params->core_masks[0];
   const unsigned cores = util_bitcount64(mask);
   if ((mask >> 10) || (cores != 8 && cores != 10))
      return false;
   for (unsigned i = 1; i < DRM_ASAHI_MAX_CLUSTERS; ++i) {
      if (params->core_masks[i])
         return false;
   }

   return agx_chip_command_stream_generation(AGX_CHIP_G15G) ==
             AGX_CS_GENERATION_G15 &&
          agx_chip_sampler_generation(AGX_CHIP_G15G) ==
             AGX_SAMPLER_GENERATION_GEN4 &&
          agx_chip_sampler_heap_element_size(AGX_CHIP_G15G) == 32;
}

bool
agx_params_driver_supported(const struct drm_asahi_params_global *params)
{
   if (!params)
      return false;

   const enum agx_chip chip = agx_chip_from_params(params);
   if (!agx_chip_driver_supported(chip))
      return false;

   switch (chip) {
   case AGX_CHIP_G17P: {
      /* Honeykrisp's first AGX3 path is the exact T8140/A18 Pro contract.
       * Do not let the shared G17P ISA name admit another SoC or topology. */
      const bool a18_pro =
         params->chip_id == 0x8140 && params->num_clusters_total == 1;
      /* M4 Pro: two clusters (a binned part keeps two). */
      const bool m4_pro =
         params->chip_id == 0x6040 && params->num_clusters_total == 2;
      const bool m4_max =
         params->chip_id == 0x6041 && params->num_clusters_total == 4;
      return (a18_pro || m4_pro || m4_max) && params->num_dies == 1 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200;
   }
   case AGX_CHIP_G15G:
      return agx_t8122_params_supported(params);
   case AGX_CHIP_G15S:
      /* T6030/M3 Pro: one die, two clusters, AGX3 shaders on the legacy
       * (pre-HAL200) firmware interface. */
      return params->chip_id == 0x6030 && params->num_dies == 1 &&
             params->num_clusters_total == 2 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_LEGACY;
   case AGX_CHIP_G15C:
      /* T6031/M3 Max: the same generation as T6030 with four clusters.
       * Binned parts differ only in their core masks, which every consumer
       * reads from the parameters, so the masks are not part of the check. */
      return params->chip_id == 0x6031 && params->num_dies == 1 &&
             params->num_clusters_total == 4 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_LEGACY;
   case AGX_CHIP_G16G:
      return params->chip_id == 0x8132 && params->num_dies == 1 &&
             params->num_clusters_total == 1 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200;
   default:
      return true;
   }
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

   if (chip == AGX_CHIP_G15G) {
      const uint64_t required = DRM_ASAHI_FEATURE_VM_STATUS |
                                DRM_ASAHI_FEATURE_SCHEDULED_QUEUES;
      return (params->features & required) == required;
   }

   const uint64_t required = DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                             DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   return (params->features & required) == required;
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
   if (!params || params->num_dies != 1 ||
       params->usc_generation != AGX_ISA_AGX3 ||
       params->gpu_hal_generation != DRM_ASAHI_GPU_HAL_300)
      return false;

   const enum agx_chip chip = agx_chip_from_params(params);
   return (chip == AGX_CHIP_G17G && params->chip_id == 0x8142 &&
           params->num_clusters_total == 1) ||
          /* T6050 M5 Pro (J714s/J716s): G17S, one die, two clusters. */
          (chip == AGX_CHIP_G17S && params->chip_id == 0x6050 &&
           params->num_clusters_total == 2) ||
          /* T6050 M5 Max (J716c): G17C, one die, four clusters. */
          (chip == AGX_CHIP_G17C && params->chip_id == 0x6050 &&
           params->num_clusters_total == 4);
}

/*
 * The M6 bring-up contract: the J873's G18G (chip 0x8152, one die, USC 3).
 * The shim's G18G profile and the J873 G18G kernel driver both publish HAL
 * 300 until the uAPI has a PI_400 value. Like the M5 contract above, it is
 * admitted by its exact hardware identity; see agx_open_device.
 */
bool
agx_params_g18g_bringup_supported(const struct drm_asahi_params_global *params)
{
   return params && agx_chip_from_params(params) == AGX_CHIP_G18G &&
          params->chip_id == 0x8152 && params->num_dies == 1 &&
          params->usc_generation == AGX_ISA_AGX3 &&
          params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_300;
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
   return chip == AGX_CHIP_G15G;
}

const char *
agx_chip_name(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G13G: return "G13G";
   case AGX_CHIP_G13X: return "G13X";
   case AGX_CHIP_G14G: return "G14G";
   case AGX_CHIP_G14X: return "G14X";
   case AGX_CHIP_G15P: return "G15P";
   case AGX_CHIP_G15G: return "G15G";
   case AGX_CHIP_G15S: return "G15S";
   case AGX_CHIP_G15C: return "G15C";
   case AGX_CHIP_G15D: return "G15D";
   case AGX_CHIP_G16P: return "G16P";
   case AGX_CHIP_G16G: return "G16G";
   case AGX_CHIP_G16S: return "G16S";
   case AGX_CHIP_G16C: return "G16C";
   case AGX_CHIP_G17P: return "G17P";
   case AGX_CHIP_G17A: return "G17A";
   case AGX_CHIP_G17G: return "G17G";
   case AGX_CHIP_G17S: return "G17S";
   case AGX_CHIP_G17C: return "G17C";
   case AGX_CHIP_G18G: return "G18G";
   case AGX_CHIP_UNKNOWN:
   default: return "unknown";
   }
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
