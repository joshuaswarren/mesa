/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Pure, dependency-free parts of Honeykrisp's VK_KHR_ray_tracing_pipeline
 * implementation: the shader-group table, the shader binding table handle
 * layout, and the pipeline stack-size arithmetic.
 *
 * These are split out so they can be unit-tested on the host with no device,
 * no NIR and no command buffer.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize is
 * required by the specification to be exactly 32
 * (VUID-VkPhysicalDeviceRayTracingPipelinePropertiesKHR-shaderGroupHandleSize).
 * Our handle carries a single 32-bit group index; the remaining 28 bytes are
 * zero and MUST be written, because an application may compare or hash a whole
 * handle.
 */
#define HK_RAY_TRACING_GROUP_HANDLE_SIZE 32

/*
 * maxRayHitAttributeSize.  The hit attribute lives in the per-invocation
 * scratch stack, immediately at the current stack pointer, so this is a
 * reservation rather than a hardware limit.  32 bytes is what the software
 * traversal writes (two floats of barycentrics) rounded up to the value
 * lavapipe uses, which is the shape the CTS exercises.
 */
#define HK_RAY_HIT_ATTRIBS_SIZE 32

/*
 * The megakernel is one compute shader.  Its workgroup is one AGX subgroup
 * wide so that a ray-generation shader's divergence costs at most one
 * subgroup.  For 2D/3D launches (height > 1), each 32-lane workgroup is
 * remapped in the shader to an 8x4 2D tile for spatial ray coherence and
 * G-buffer locality; 1D launches (height <= 1) keep a flat 32x1 strip so no
 * lanes are wasted on empty rows.
 */
/*
 * maxRayRecursionDepth.  Recursion is unrolled into the megakernel at compile
 * time, so every extra level multiplies the compiled code; this is a code-size
 * decision, not a hardware limit. Keep the legacy limit unchanged.
 */
#define HK_MAX_RAY_RECURSION_DEPTH 3
/* Pipelines up to this depth keep the inlined megakernel. Deeper G17P
 * pipelines run their stages as continuations from one scheduler loop, which
 * holds one copy of every stage and of traversal at any depth. */
#define HK_RT_INLINE_MAX_RAY_RECURSION_DEPTH 3
/* The specification's maximum; KHR requires at least one level. */
#define HK_G17P_MAX_RAY_RECURSION_DEPTH 31
/* Defensive ceiling for invalid requests from older stage-library clients. */
#define HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH 31

#define HK_RT_WORKGROUP_SIZE_X      32
#define HK_RT_WORKGROUP_SIZE_X_LOG2 5
#define HK_RT_WORKGROUP_SIZE_Y 1
#define HK_RT_WORKGROUP_SIZE_Z 1

#define HK_RT_TILE_2D_X      8
#define HK_RT_TILE_2D_X_LOG2 3
#define HK_RT_TILE_2D_Y      4
#define HK_RT_TILE_2D_Y_LOG2 2

struct hk_ray_tracing_group_handle {
   uint32_t index;
};

/*
 * A shader group as the megakernel consumes it.  Every index is an index into
 * the pipeline's flattened stage array, or VK_SHADER_UNUSED_KHR.
 *
 * `recursive_index` is the group's *callable* shader: the general shader for a
 * general group (raygen, miss or callable) and the closest-hit shader for a hit
 * group.  Any-hit and intersection shaders are not reached through the SBT
 * dispatch switch, they are reached from inside traversal, so they are named
 * separately.
 */
struct hk_ray_tracing_group {
   struct hk_ray_tracing_group_handle handle;
   uint32_t recursive_index;
   uint32_t ahit_index;
   uint32_t isec_index;
};

/*
 * Derive the three stage indices of one group from its create info.  Returns
 * false for a group type this driver does not implement, in which case *out is
 * left untouched.
 */
bool hk_rt_group_from_create_info(struct hk_ray_tracing_group *out,
                                  VkRayTracingShaderGroupTypeKHR type,
                                  uint32_t general_shader,
                                  uint32_t closest_hit_shader,
                                  uint32_t any_hit_shader,
                                  uint32_t intersection_shader);

/*
 * Whether every stage index of a group is VK_SHADER_UNUSED_KHR or names one of
 * the pipeline's stage_count stages.  Creation holds the groups it builds to
 * this; a pipeline loaded from a binary or a cache holds the stored groups to
 * it too.
 */
bool hk_rt_group_indices_valid(const struct hk_ray_tracing_group *group,
                               uint32_t stage_count);

/*
 * Whether a group's indices are valid and each names a shader of a stage its
 * slot takes: a ray generation, miss, callable or closest-hit shader as the
 * recursive shader, an any-hit shader and an intersection shader in those
 * slots.  stages[i] is the stage of shader i.  Groups built from create info
 * satisfy this by the API's valid usage; groups read back from stored data
 * are checked.
 */
bool hk_rt_group_stages_valid(const struct hk_ray_tracing_group *group,
                              uint32_t stage_count,
                              const VkShaderStageFlagBits *stages);

/*
 * Rebase one group of an imported pipeline library onto the importing
 * pipeline's flattened stage array.  `stage_base` is the number of stages that
 * precede this library.  VK_SHADER_UNUSED_KHR never moves.
 */
struct hk_ray_tracing_group
hk_rt_group_rebase(struct hk_ray_tracing_group src, uint32_t stage_base);

/*
 * Serialise a group handle into the application's buffer.  Writes exactly
 * HK_RAY_TRACING_GROUP_HANDLE_SIZE bytes: the little-endian group index
 * followed by zeroes.
 */
void hk_rt_write_group_handle(void *dst,
                              struct hk_ray_tracing_group_handle handle);

/*
 * Per-stage scratch requirements of a compiled pipeline, in bytes.  Each is the
 * maximum over every stage of that kind that the megakernel inlined.
 */
struct hk_rt_stack_sizes {
   uint32_t raygen;
   uint32_t any_hit;
   uint32_t closest_hit;
   uint32_t miss;
   uint32_t intersection;
   uint32_t callable;
};

/*
 * Scratch the megakernel must reserve above its own frame, for a pipeline whose
 * maxPipelineRayRecursionDepth is `recursion_depth`.
 *
 * The megakernel does not recurse: every stage is inlined and a traceRay inside
 * a closest-hit or miss shader is another inlined copy at a higher stack
 * pointer.  So the worst case is one raygen frame, then for the first level of
 * recursion either a closest-hit frame, a miss frame, or an intersection frame
 * with an any-hit frame stacked on it, then for every further level either a
 * closest-hit or a miss frame (an intersection shader cannot itself trace).
 * Callables are bounded separately at the 31 the specification's minimum
 * maxRayRecursionDepth implies.
 *
 * This is the same formula lavapipe uses.  It is an upper bound, not a measured
 * dynamic size: VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR cannot
 * shrink it, because the stack is static scratch in the compiled binary.
 */
uint32_t hk_rt_pipeline_stack_size(const struct hk_rt_stack_sizes *sizes,
                                   uint32_t recursion_depth);

/*
 * A continuation call record: resume pc, caller stack pointer, caller payload
 * offset, padding, caller shader record pointer, caller vertex position
 * address. It sits between the caller's frame and the callee's.
 */
#define HK_RT_RETURN_RECORD_SIZE 32

/*
 * Scratch a continuation-mode pipeline reserves above its own frame, for
 * `sizes` holding the largest frame of each stage kind (a split stage's frame
 * includes its spills).  The stack starts with one return record for raygen.
 * Every call pushes the caller's frame, rounded to 16 bytes, and one record:
 * raygen once, a closest-hit or miss frame at most once per recursion level,
 * and callables at most 31 times, as in hk_rt_pipeline_stack_size().  The
 * innermost frame is at most *max_frame, which also covers the intersection
 * and any-hit frames of a traversal (reserved side by side, as
 * hk_rt_pipeline_stack_size() does), on top of `traversing_frame` when a
 * stage of that frame traverses inline (hit objects).
 *
 * A call is taken only when the callee's base plus *max_frame stays inside
 * the reservation, so no call sequence can overflow it; calls within the
 * declared depth always fit.
 */
uint32_t hk_rt_continuation_stack_size(const struct hk_rt_stack_sizes *sizes,
                                       uint32_t recursion_depth,
                                       bool has_callables,
                                       uint32_t traversing_frame,
                                       uint32_t *max_frame);

/*
 * vkGetRayTracingShaderGroupStackSizeKHR: the scratch one shader of one group
 * occupies.  Returns 0 for a shader the group does not have.
 */
uint32_t hk_rt_group_stack_size(const struct hk_rt_stack_sizes *sizes,
                               const struct hk_ray_tracing_group *group,
                               VkShaderGroupShaderKHR which,
                               uint32_t recursive_stage_size);

/*
 * Threadgroup counts for a trace-rays launch of `width` x `height` x `depth`
 * rays, for the megakernel's fixed workgroup size.  The megakernel bounds-checks
 * its own launch id, so a partial trailing workgroup is safe.
 */
void hk_rt_launch_group_count(uint32_t width, uint32_t height, uint32_t depth,
                              uint32_t out[3]);

/*
 * Byte offsets inside VkTraceRaysIndirectCommand2KHR, which is the record both
 * the direct and the indirect entry points publish and the megakernel reads.
 * Named here so the shader builder and the unit test agree by construction.
 */
#define HK_RT_CMD_RAYGEN_ADDRESS                                               \
   offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress)
#define HK_RT_CMD_MISS_ADDRESS                                                 \
   offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress)
#define HK_RT_CMD_HIT_ADDRESS                                                  \
   offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress)
#define HK_RT_CMD_CALLABLE_ADDRESS                                             \
   offsetof(VkTraceRaysIndirectCommand2KHR, callableShaderBindingTableAddress)
#define HK_RT_CMD_WIDTH offsetof(VkTraceRaysIndirectCommand2KHR, width)

/*
 * A shader binding table region's stride sits two VkDeviceSize after its
 * address.  hk_rt_sbt_stride_offset() is the one place that relationship is
 * written down; the NIR builder and the test both use it.
 */
static inline uint32_t
hk_rt_sbt_stride_offset(uint32_t address_offset)
{
   return address_offset + 2 * sizeof(VkDeviceSize);
}

#ifdef __cplusplus
}
#endif
