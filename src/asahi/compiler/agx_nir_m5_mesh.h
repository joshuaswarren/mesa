/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include "compiler/nir/nir.h"
#include "agx_mesh_abi.h"
#ifdef __cplusplus
extern "C" {
#endif






bool agx_nir_m5_canonicalize_empty_mesh(nir_shader *shader);
bool agx_nir_lower_m5_mesh_triangle_probe(nir_shader *shader);














struct agx_m5_mesh_varyings {
   unsigned size;
   bool dynamic;
   uint8_t primitive_components;
   uint16_t primitive_stride_B;
   union { uint8_t clip_first; uint8_t clip_distance_first; };
   union { uint8_t clip_count; uint8_t clip_distance_count; };
   uint8_t lv_first;
   bool lv;
   uint8_t first[64];
   uint8_t components[64];
   uint8_t primitive_first[64];
   uint8_t primitive_comps[64];
   /* gl_CullPrimitiveEXT is written (M5: dropped before the export). */
   bool cull_primitive;
   /* G17P: remap position z from [-w, w] at the export
    * (agx_nir_lower_gen1_mesh), whatever the shape of the position stores. */
   bool clip_m1_1;
   /* G17P and M5 lines: root byte offset of the signed line row bias
    * (0: none), see agx_nir_lower_gen1_mesh and m5_mesh_line_bias. */
   uint16_t line_bias_root_B;
   /* G17P: the layer is a per-vertex VARYING_SLOT_LAYER output that every
    * vertex of a workgroup writes with the same value (hk's multiview
    * lowering writes the view index). It goes straight to each vertex's
    * layer word, so primitives keep their shared vertices and no primitive
    * record is written. lv is set; no per-primitive layer or viewport. */
   bool vertex_layer;
};



bool agx_nir_lower_m5_mesh_fragment(nir_shader *shader);
bool agx_nir_lower_m5_mesh_fragment_layout(nir_shader *shader,
                                           const struct agx_m5_mesh_varyings *mesh);

bool agx_m5_mesh_gather_varyings(const nir_shader *shader,
                                 struct agx_m5_mesh_varyings *varyings);

bool agx_nir_lower_m5_mesh(nir_shader *shader,
                           const struct agx_m5_mesh_varyings *varyings);


bool agx_nir_lower_m5_task_payload_probe(nir_shader *shader);
bool agx_nir_lower_m5_task_probe(nir_shader *shader);
bool agx_nir_lower_m5_task_payload(nir_shader *shader, unsigned max_payload_B,
                                   unsigned payload_base_B);
bool agx_nir_lower_m5_task(nir_shader *shader, unsigned max_payload_B,
                           bool grid_record);
bool agx_nir_lower_m5_mesh_task_record(nir_shader *shader);
bool agx_gen1_mesh_gather_varyings(const nir_shader *, struct agx_m5_mesh_varyings *);
bool agx_nir_lower_gen1_mesh(nir_shader *, const struct agx_m5_mesh_varyings *);
bool agx_nir_lower_gen1_mesh_triangle_probe(nir_shader *);
void agx_gen1_mesh_record_layout(struct agx_m5_mesh_varyings *);
bool agx_nir_lower_gen1_mesh_fragment_layout(nir_shader *, const struct agx_m5_mesh_varyings *);
bool agx_nir_lower_gen1_task_payload_probe(nir_shader *);
bool agx_nir_lower_gen1_task_probe(nir_shader *);

#ifdef __cplusplus
}
#endif
