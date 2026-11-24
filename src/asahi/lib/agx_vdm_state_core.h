/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM-state representations.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH 8u
#define AGX_VDM_STATE_VERTEX_PIPELINE_TEMPLATE_MASK UINT32_C(0x003fffff)

struct agx_vdm_state_vertex_pipeline_pair {
   uint8_t bytes[AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH];
   uint8_t length;
};

/*
 * Two-word AGX3 VDM State vertex-pipeline pair (64-byte-aligned 48-bit VA):
 *   word 0[21:0]:  caller-preserved raw state-template bits
 *   word 0[31:22]: pipeline[47:38]
 *   word 1[31:0]:  pipeline[37:6]
 */
bool agx_vdm_state_vertex_pipeline_pair_supported(enum agx_chip chip);

bool agx_vdm_state_vertex_pipeline_pair_pack(
   enum agx_chip chip, uint64_t pipeline, uint32_t raw_template,
   struct agx_vdm_state_vertex_pipeline_pair *out);

bool agx_vdm_state_vertex_pipeline_pair_unpack(
   enum agx_chip chip, const void *bytes, size_t size, uint64_t *pipeline,
   uint32_t *raw_template);

#ifdef __cplusplus
}
#endif
