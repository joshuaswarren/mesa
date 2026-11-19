/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "compiler/nir/nir.h"
#include "compiler/shader_enums.h"
#include "drm-uapi/asahi_drm.h"


/* A native G17P mesh launch runs at most this many mesh workgroups: a larger
 * child grid launches none. A draw's mesh workgroups therefore go out in
 * launches of whole rows: launch k publishes (X, rows, 1) children, and row
 * r = workgroup y + k * hk_mesh_launch_rows(X). */
#define HK_MESH_LAUNCH_WGS ((1u << 19) - 1)
/* Launches for a grid only the GPU knows: enough for the 2^22 workgroups of
 * maxMeshWorkGroupTotalCount at any row width up to 65535. */
#define HK_MESH_LAUNCHES_MAX 10

/* A launch publishes its rows as y rows by z slices: a child grid's y is at
 * most this, a larger one launches nothing. */
#define HK_MESH_LAUNCH_MAX_Y 65535u

/* Rows of x workgroups one launch takes (as hk_mesh_launch_shape() in
 * hk_shader.c): at most HK_MESH_LAUNCH_WGS workgroups, in equal slices of at
 * most HK_MESH_LAUNCH_MAX_Y rows. */
static inline uint32_t
hk_mesh_launch_rows(uint32_t x)
{
   uint32_t per = HK_MESH_LAUNCH_WGS / (x ? x : 1);
   per = per ? per : 1;
   const uint32_t slices = (per + HK_MESH_LAUNCH_MAX_Y - 1) / HK_MESH_LAUNCH_MAX_Y;
   return (per / slices) * slices;
}

/* M5 task (object) shaders on G17S. taskShader is a required feature of
 * VK_EXT_mesh_shader. The native object->mesh record
 * (child grid and payload at the SR62/SR63 payload address) needs the kernel
 * to bind the TA mesh rendering buffer (0x1ca48) to task kicks, which it
 * reports with DRM_ASAHI_FEATURE_TASK_MESH_STORAGE; taskShader is
 * advertised only then. Without it only payload-free task shaders are
 * accepted. hk_m5_task_record() reports that kernel storage; whether a task
 * also writes the child grid at +0 is decided per task/mesh pair
 * (hk_mesh_info::m5_no_grid_record). */
static inline bool hk_m5_task_record(uint64_t kernel_features) {
   return kernel_features & DRM_ASAHI_FEATURE_TASK_MESH_STORAGE;
}
static inline unsigned hk_m5_task_payload_limit(uint64_t kernel_features) {
   return hk_m5_task_record(kernel_features) ? 16384 : 0;
}

struct hk_mesh_info {
   uint32_t max_vertices;
   uint32_t max_primitives;
   uint32_t vertex_stride_B;
   uint32_t primitive_stride_B;
   uint32_t local_size[3];

   


   uint64_t vertex_outputs;
   uint64_t primitive_outputs;
   bool writes_cull_primitive;

   

   bool reads_draw_id;
   bool reads_num_workgroups;

   
   uint8_t prim;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;

   /* M5 task and mesh shaders compiled together whose mesh never reads
    * gl_NumWorkGroups (hk_compile_shaders()): the task writes no child grid
    * record and both put the payload at +0 of the payload address. Every
    * other M5 task and mesh shader on a kernel with task mesh storage keeps
    * the grid at +0 and the payload at +16. A task and a mesh shader drawn
    * together always agree. */
   bool m5_no_grid_record;

   
   uint32_t payload_size_B;
};
