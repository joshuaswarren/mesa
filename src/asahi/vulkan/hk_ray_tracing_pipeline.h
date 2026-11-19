/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_ray_tracing_pipeline for Honeykrisp on G17P.
 *
 * The whole pipeline compiles to ONE compute shader.  Every stage the
 * application supplies is inlined into it as a NIR function, traceRayEXT
 * becomes an inlined copy of the software traversal that VK_KHR_ray_query
 * already uses, and the shader binding table selects between the inlined
 * copies with a comparison on the group index.  There is no indirect shader
 * call, no resume shader and no shader-call stack: recursion is bounded by
 * maxPipelineRayRecursionDepth at compile time and unrolled.  Pipelines deeper
 * than HK_RT_INLINE_MAX_RAY_RECURSION_DEPTH instead split their stages at
 * every call and run them as continuations from one scheduler loop, with
 * return records on the invocation's scratch stack.
 *
 * This is the shape lavapipe uses, and it is used here for the same reason:
 * the AGX3 backend has no way to branch to an address the command stream
 * supplies, so a real shader-binding-table dispatch is not expressible today.
 */

#ifndef HK_RAY_TRACING_PIPELINE_H
#define HK_RAY_TRACING_PIPELINE_H

#include "hk_private.h"
#include "hk_ray_tracing_core.h"

#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

extern const struct vk_pipeline_ops hk_ray_tracing_pipeline_ops;

struct hk_api_shader;
struct vk_pipeline_cache_object;
struct hk_device;
struct nir_shader;

/*
 * A stage's NIR, reference counted so a pipeline library's stages can be shared
 * with every pipeline that imports the library without being recompiled from
 * SPIR-V.
 */
struct hk_rt_stage_nir {
   uint32_t ref_cnt;
   struct nir_shader *nir;
};

struct hk_ray_tracing_pipeline {
   struct vk_pipeline base;

   struct vk_pipeline_layout *layout;

   uint8_t binary_key[32];
   /* This pipeline's pipeline cache entry. The device's memory cache holds
    * entries only while something references them. */
   struct vk_pipeline_cache_object *cache_object;

   uint32_t stage_count;
   uint32_t group_count;

   struct hk_rt_stage_nir **stages;
   struct hk_ray_tracing_group *groups;

   /* The compiled megakernel.  NULL on M5, which compiles kernels instead,
    * and for a pipeline library, which is never executable. */
   struct hk_api_shader *shader;

   /*
    * On M5, one compute kernel per ray-generation group.  Each kernel inlines only
    * its own raygen plus every hit/miss/intersection/callable stage reachable
    * through the shader binding table; vkCmdTraceRays selects the kernel on the
    * GPU from the raygen record's group handle (hk_cmd_trace_rays.c), because
    * the record's contents are only required to be valid at execution time.
    * Empty elsewhere and for a pipeline library, which is never executable.
    */
   uint32_t kernel_count;
   struct hk_rt_kernel {
      struct hk_api_shader *shader;
      /* The hk_ray_tracing_group_handle::index this kernel's raygen answers to. */
      uint32_t raygen_handle_index;
   } *kernels;

   /* What the kernels reserve above their own frame (maximum over kernels),
    * reported through vkGetRayTracingShaderGroupStackSizeKHR and used by
    * nothing else: the stack is static scratch inside each compiled binary,
    * so the dynamic stack-size state cannot change it. */
   struct hk_rt_stack_sizes stack_sizes;
   uint32_t stack_size;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_ray_tracing_pipeline, base.base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE);

void hk_rt_device_finish(struct hk_device *dev);

static inline struct hk_ray_tracing_pipeline *
hk_pipeline_to_ray_tracing(struct vk_pipeline *pipeline)
{
   assert(pipeline->bind_point == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
   return container_of(pipeline, struct hk_ray_tracing_pipeline, base);
}

#endif /* HK_RAY_TRACING_PIPELINE_H */
