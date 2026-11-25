/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 direct VDM Tessellate representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_TESSELLATE_DIRECT_LENGTH 36u

struct agx_vdm_tessellate_direct_info {
   uint64_t factor_buffer;
   uint32_t patch_count;
   uint32_t instance_count;
   uint32_t base_patch;
   uint32_t base_instance;

   /* Raw tessellation draw fields; no implicit scaling. */
   uint32_t instance_stride_word;
   uint32_t factor_buffer_size_minus_one;
};

struct agx_vdm_tessellate_direct {
   uint8_t bytes[AGX_VDM_TESSELLATE_DIRECT_LENGTH];
   uint8_t length;
};

bool agx_vdm_tessellate_direct_target_supported(enum agx_chip chip);

bool agx_vdm_tessellate_direct_pack(
   enum agx_chip chip, const struct agx_vdm_tessellate_direct_info *info,
   struct agx_vdm_tessellate_direct *out);

bool agx_vdm_tessellate_direct_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_tessellate_direct_info *info);

#define AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH 28u

struct agx_vdm_tessellate_indirect_info {
   uint64_t factor_buffer;

   /* Four words: patch count, instance count, patch start, base instance. */
   uint64_t indirect_buffer;

   /* As agx_vdm_tessellate_direct_info. */
   uint32_t instance_stride_word;
   uint32_t factor_buffer_size_minus_one;
};

struct agx_vdm_tessellate_indirect {
   uint8_t bytes[AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH];
   uint8_t length;
};

bool agx_vdm_tessellate_indirect_pack(
   enum agx_chip chip, const struct agx_vdm_tessellate_indirect_info *info,
   struct agx_vdm_tessellate_indirect *out);

bool agx_vdm_tessellate_indirect_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_tessellate_indirect_info *info);

#define AGX_VDM_TESSELLATION_STATE_LENGTH 12u

enum agx_vdm_tessellation_partition {
   AGX_VDM_TESSELLATION_PARTITION_POW2 = 0,
   AGX_VDM_TESSELLATION_PARTITION_INTEGER = 1,
   AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_ODD = 2,
   AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_EVEN = 3,
};

enum agx_vdm_tessellation_step {
   AGX_VDM_TESSELLATION_STEP_CONSTANT = 0,
   AGX_VDM_TESSELLATION_STEP_PER_PATCH = 1,
   AGX_VDM_TESSELLATION_STEP_PER_INSTANCE = 2,
   AGX_VDM_TESSELLATION_STEP_PER_PATCH_AND_PER_INSTANCE = 3,
};

struct agx_vdm_tessellation_state_info {
   enum agx_vdm_tessellation_partition partition;
   bool counterclockwise;
   /* The tessellator's maximum factor, 1..64. A fractional-odd record
    * stores one less, so its largest factor is odd. */
   uint8_t max_factor;
   bool quads;
   bool float_factors;
   enum agx_vdm_tessellation_step step;
   /* Native vertex program flag, set from the post-tessellation program's
    * linker metadata. Zero in every captured record. */
   bool program_flag;
   /* The factor scale as an IEEE half; 1.0 is 0x3c00. */
   uint16_t scale_half;
};

bool agx_vdm_tessellation_state_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_tessellation_state_info *info);

#ifdef __cplusplus
}
#endif
