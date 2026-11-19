/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "agx_chip.h"
#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_MESH_STATE_LENGTH    56u
#define AGX_VDM_MESH_DISPATCH_LENGTH 28u
#define AGX_VDM_MESH_STATE_HEADER    UINT32_C(0x40000d2c)
/* Index List block with the mesh bit; bits 8..15 are the Index List
 * "Primitive" field (cmdbuf.xml): native triangle mesh records carry 6. */
#define AGX_VDM_MESH_DISPATCH_HEADER UINT32_C(0x70000600)
#define AGX_VDM_MESH_DISPATCH_BASE   UINT32_C(0x70000000)


struct agx_vdm_mesh_state_info {
   uint32_t raw_prefix[4]; 
   uint64_t object_pipeline, mesh_pipeline;
   uint32_t raw_object_template, raw_mesh_template;
   uint32_t raw_tail[5]; 
};
struct agx_vdm_mesh_dispatch_info {
   uint32_t threads[3], local_size[3];
   /* Index List "Primitive" of the mesh output: 0 points, 1 lines,
    * 6 triangles. */
   uint8_t primitive;
};

bool agx_vdm_mesh_threadgroup_pack(const uint32_t size[3], uint32_t *word);
/* Captured HAL300 limits: max primitive index and vertex output arena. */
bool agx_vdm_m5_mesh_set_output_limits(struct agx_vdm_mesh_state_info *info,
                                     uint32_t max_vertices,
                                     uint32_t max_primitives,
                                     uint32_t uvs_words);

bool agx_vdm_mesh_state_pack(enum agx_chip chip,
                             const struct agx_vdm_mesh_state_info *info,
                             uint8_t out[AGX_VDM_MESH_STATE_LENGTH]);
bool agx_vdm_mesh_state_unpack(enum agx_chip chip, const void *bytes,
                               size_t size,
                               struct agx_vdm_mesh_state_info *info);
bool agx_vdm_mesh_dispatch_pack(enum agx_chip chip,
                                const struct agx_vdm_mesh_dispatch_info *info,
                                uint8_t out[AGX_VDM_MESH_DISPATCH_LENGTH]);
bool agx_vdm_mesh_dispatch_unpack(enum agx_chip chip, const void *bytes,
                                  size_t size,
                                  struct agx_vdm_mesh_dispatch_info *info);

#ifdef __cplusplus
}
#endif
