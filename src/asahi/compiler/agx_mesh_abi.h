#pragma once
#include "compiler/shader_enums.h"

#define AGX_MESH_PRIMITIVE_PAGE_SLOT VARYING_SLOT_FACE

enum agx_mesh_export_fence {
   AGX_MESH_EXPORT_FENCE_INDICES,
   AGX_MESH_EXPORT_FENCE_VERTEX_BATCH,
};
