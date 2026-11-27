/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_state_core.h"
#include "agx_bytes.h"

#include <string.h>

bool
agx_vdm_state_vertex_pipeline_pair_supported(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_VDM_STATE);
}

bool
agx_vdm_state_vertex_pipeline_pair_pack(
   enum agx_chip chip, uint64_t pipeline, uint32_t raw_template,
   struct agx_vdm_state_vertex_pipeline_pair *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !agx_vdm_state_vertex_pipeline_pair_supported(chip) ||
       (pipeline & UINT64_C(0x3f)) != 0 || pipeline >= (UINT64_C(1) << 48) ||
       (raw_template & ~AGX_VDM_STATE_VERTEX_PIPELINE_TEMPLATE_MASK) != 0)
      return false;

   const uint32_t word0 =
      raw_template | (uint32_t)(((pipeline >> 38) & UINT64_C(0x3ff)) << 22);
   const uint32_t word1 = (uint32_t)(pipeline >> 6);

   agx_store_le32(&out->bytes[0], word0);
   agx_store_le32(&out->bytes[4], word1);
   out->length = AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH;
   return true;
}

bool
agx_vdm_state_vertex_pipeline_pair_unpack(
   enum agx_chip chip, const void *bytes, size_t size, uint64_t *pipeline,
   uint32_t *raw_template)
{
   if (pipeline)
      *pipeline = 0;
   if (raw_template)
      *raw_template = 0;

   if (!bytes || !pipeline || !raw_template ||
       size != AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH ||
       !agx_vdm_state_vertex_pipeline_pair_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   const uint32_t word0 = agx_load_le32(&raw[0]);
   const uint32_t word1 = agx_load_le32(&raw[4]);

   *pipeline = ((uint64_t)(word0 >> 22) << 38) | ((uint64_t)word1 << 6);
   *raw_template = word0 & AGX_VDM_STATE_VERTEX_PIPELINE_TEMPLATE_MASK;
   return true;
}
