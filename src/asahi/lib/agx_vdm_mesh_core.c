/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#include "agx_vdm_mesh_core.h"
#include <string.h>
#include "agx_bytes.h"
#include "agx_vdm_state_core.h"


static bool
supported(enum agx_chip chip)
{
   return agx_chip_g17p_or_m5(chip);
}
bool
agx_vdm_mesh_threadgroup_pack(const uint32_t size[3], uint32_t *word)
{
   if (!size || !word || !size[0] || !size[1] || !size[2] ||
       size[0] > 2047 || size[1] > 2047 || size[2] > 1023 ||
       (uint64_t)size[0]*size[1]*size[2] > 1024)
      return false;
   *word = size[0] | (size[1] << 11) | (size[2] << 22);
   return true;
}

bool
agx_vdm_m5_mesh_set_output_limits(struct agx_vdm_mesh_state_info *info,
                                  uint32_t max_vertices,
                                  uint32_t max_primitives,
                                  uint32_t uvs_words)
{
   if (!info || !max_vertices || max_vertices > 256 ||
       !max_primitives || max_primitives > 256 ||
       uvs_words < 4 || uvs_words > 255)
      return false;

   const uint32_t last_primitive = max_primitives - 1;
   const uint32_t aligned_vertices = (max_vertices + 3) & ~3u;
   /* Native mesh-cell (6V/2P/4 words), primitive-ID (6V/2P/5 words)
    * and overlap (192V/64P/4 words) packets establish these fields. */
   info->raw_prefix[0] = uvs_words | ((uvs_words + 1) << 8);
   info->raw_tail[1] = last_primitive | ((last_primitive >> 4) << 16);
   info->raw_tail[4] = 1 + aligned_vertices * uvs_words / 4;
   return true;
}

bool
agx_vdm_m5_mesh_set_task_payload(struct agx_vdm_mesh_state_info *info,
                                 uint32_t payload_size_B)
{
   if (!info || payload_size_B > AGX_VDM_M5_MESH_MAX_TASK_PAYLOAD_BYTES)
      return false;

   info->raw_tail[0] =
      payload_size_B ? (payload_size_B + 1023u) / 1024u : 1u;
   return true;
}

bool
agx_vdm_mesh_state_pack(enum agx_chip chip,
                        const struct agx_vdm_mesh_state_info *info,
                        uint8_t out[AGX_VDM_MESH_STATE_LENGTH])
{
   if (out)
      memset(out, 0, AGX_VDM_MESH_STATE_LENGTH);
   struct agx_vdm_state_vertex_pipeline_pair object, mesh;
   if (!out || !info || !supported(chip) || !info->object_pipeline ||
       (chip == AGX_CHIP_G17P && !(info->raw_prefix[1] & 0x10)) ||
       !info->mesh_pipeline ||
       !agx_vdm_state_vertex_pipeline_pair_pack(
          chip, info->object_pipeline, info->raw_object_template, &object) ||
       !agx_vdm_state_vertex_pipeline_pair_pack(chip, info->mesh_pipeline,
                                                info->raw_mesh_template, &mesh))
      return false;
   agx_store_le32(out, AGX_VDM_MESH_STATE_HEADER);
   for (unsigned i = 0; i < 4; i++)
      agx_store_le32(out + 4 + 4 * i, info->raw_prefix[i]);
   memcpy(out + 20, object.bytes, 8);
   memcpy(out + 28, mesh.bytes, 8);
   for (unsigned i = 0; i < 5; i++)
      agx_store_le32(out + 36 + 4 * i, info->raw_tail[i]);
   return true;
}
bool
agx_vdm_mesh_state_unpack(enum agx_chip chip, const void *bytes, size_t size,
                          struct agx_vdm_mesh_state_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));
   if (!bytes || !info || !supported(chip) || size != AGX_VDM_MESH_STATE_LENGTH)
      return false;
   const uint8_t *raw = bytes;
   struct agx_vdm_mesh_state_info tmp = {0};
   if (agx_load_le32(raw) != AGX_VDM_MESH_STATE_HEADER ||
       !agx_vdm_state_vertex_pipeline_pair_unpack(
          chip, raw + 20, 8, &tmp.object_pipeline, &tmp.raw_object_template) ||
       !agx_vdm_state_vertex_pipeline_pair_unpack(
          chip, raw + 28, 8, &tmp.mesh_pipeline, &tmp.raw_mesh_template) ||
       !tmp.object_pipeline || !tmp.mesh_pipeline)
      return false;
   for (unsigned i = 0; i < 4; i++)
      tmp.raw_prefix[i] = agx_load_le32(raw + 4 + 4 * i);
   for (unsigned i = 0; i < 5; i++)
      tmp.raw_tail[i] = agx_load_le32(raw + 36 + 4 * i);
   *info = tmp;
   return true;
}
static bool
valid_grid(const struct agx_vdm_mesh_dispatch_info *info)
{
   if (!info)
      return false;
   uint64_t total = 1;
   for (unsigned i = 0; i < 3; i++) {
      if (!info->threads[i] || !info->local_size[i] ||
          info->local_size[i] > 1024 || info->threads[i] % info->local_size[i])
         return false;
      total *= info->local_size[i];
   }
   return total <= 1024;
}
bool
agx_vdm_mesh_dispatch_pack(enum agx_chip chip,
                           const struct agx_vdm_mesh_dispatch_info *info,
                           uint8_t out[AGX_VDM_MESH_DISPATCH_LENGTH])
{
   if (out)
      memset(out, 0, AGX_VDM_MESH_DISPATCH_LENGTH);
   if (!out || !supported(chip) || !valid_grid(info) ||
       (info->primitive != 0 && info->primitive != 1 && info->primitive != 6))
      return false;
   agx_store_le32(out, AGX_VDM_MESH_DISPATCH_BASE | (info->primitive << 8));
   for (unsigned i = 0; i < 3; i++) {
      agx_store_le32(out + 4 + 4 * i, info->threads[i]);
      agx_store_le32(out + 16 + 4 * i, info->local_size[i]);
   }
   return true;
}
bool
agx_vdm_mesh_dispatch_unpack(enum agx_chip chip, const void *bytes, size_t size,
                             struct agx_vdm_mesh_dispatch_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));
   if (!bytes || !info || !supported(chip) ||
       size != AGX_VDM_MESH_DISPATCH_LENGTH)
      return false;
   const uint8_t *raw = bytes;
   struct agx_vdm_mesh_dispatch_info tmp = {0};
   const uint32_t header = agx_load_le32(raw);
   tmp.primitive = (header >> 8) & 0xff;
   if ((header & ~UINT32_C(0xff00)) != AGX_VDM_MESH_DISPATCH_BASE ||
       (tmp.primitive != 0 && tmp.primitive != 1 && tmp.primitive != 6))
      return false;
   for (unsigned i = 0; i < 3; i++) {
      tmp.threads[i] = agx_load_le32(raw + 4 + 4 * i);
      tmp.local_size[i] = agx_load_le32(raw + 16 + 4 * i);
   }
   if (!valid_grid(&tmp))
      return false;
   *info = tmp;
   return true;
}
