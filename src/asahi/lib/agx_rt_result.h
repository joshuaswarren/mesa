/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 hardware ray-query state and SCS ray-staging offsets.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* IntersectRayResult byte offsets (distinct from inline ray-query state). */
#define AGX_RT_HIT_OFF_TYPE               0u
#define AGX_RT_HIT_OFF_T                  4u
#define AGX_RT_HIT_OFF_PRIMITIVE_ID       8u
#define AGX_RT_HIT_OFF_GEOMETRY_ID        12u
#define AGX_RT_HIT_OFF_PRIMITIVE_DATA     16u
#define AGX_RT_HIT_OFF_INSTANCE_ID        24u
#define AGX_RT_HIT_OFF_USER_INSTANCE_ID   28u
#define AGX_RT_HIT_OFF_INSTANCE_COUNT     32u
#define AGX_RT_HIT_OFF_BARYCENTRICS       40u
#define AGX_RT_HIT_OFF_FRONT_FACE         48u
#define AGX_RT_HIT_OFF_CURVE_PARAM        52u
#define AGX_RT_HIT_OFF_INTERSECTION_FN_ID 56u
#define AGX_RT_HIT_OFF_OBJECT_TO_WORLD    60u
#define AGX_RT_HIT_OFF_WORLD_TO_OBJECT    108u
#define AGX_RT_HIT_RESULT_SIZE            156u
#define AGX_RT_TRANSFORM_FLOATS            12u

/* Inline ray-query state layout (168 bytes; distinct from IntersectRayResult). */
#define AGX_RT_RAY_QUERY_STATE_SIZE 168u

#define AGX_RT_QUERY_OFF_TMIN       0u
#define AGX_RT_QUERY_OFF_TMAX       4u
#define AGX_RT_QUERY_OFF_ORIGIN    12u /* 3 x f32 */
#define AGX_RT_QUERY_OFF_DIRECTION 24u /* 3 x f32 */
#define AGX_RT_QUERY_OFF_PARAMS    48u /* 10 bytes of intersection_params */
#define AGX_RT_QUERY_OFF_STATE     68u /* intersection_query::State */
#define AGX_RT_QUERY_OFF_CANDIDATE 88u /* ray_query_candidate, 20 bytes */
#define AGX_RT_QUERY_OFF_COMMITTED 112u

/* Committed-hit sub-record offsets within the 168-byte ray-query state.
 * Committed instance/user-instance IDs resolve through the candidate record
 * rather than fixed committed fields.
 */
#define AGX_RT_QUERY_OFF_COMMITTED_T              112u /* f32   */
#define AGX_RT_QUERY_OFF_COMMITTED_GEOMETRY_ID    120u /* i32   */
#define AGX_RT_QUERY_OFF_COMMITTED_PRIMITIVE_ID   124u /* i32   */
#define AGX_RT_QUERY_OFF_COMMITTED_INSTANCE_COUNT 128u /* i32   */
#define AGX_RT_QUERY_OFF_COMMITTED_TYPE           132u /* i8, hit_kind */
#define AGX_RT_QUERY_OFF_COMMITTED_FRONT_FACE     133u /* i8, bool     */
#define AGX_RT_QUERY_OFF_COMMITTED_BARYCENTRICS   136u /* 2 x f32 */
#define AGX_RT_QUERY_OFF_COMMITTED_CURVE_PARAM    144u /* f32   */
#define AGX_RT_QUERY_OFF_COMMITTED_PRIMITIVE_DATA 152u /* pointer */

/* SCS bank 0 ray staging segment (64 bytes, two ray slots). Origin and
 * direction are interleaved per axis (x0, dx, y0, dy, z0, dz), followed by
 * tmax (+24) and the 64-bit packed control word (+28). tmin is not staged here.
 * Non-bank-0 SCS offsets are per-shader layout assignments.
 */
#define AGX_RT_SCS_RAY_AXIS_STRIDE   8u
#define AGX_RT_SCS_RAY_ORIGIN(axis)    ((axis) * AGX_RT_SCS_RAY_AXIS_STRIDE)
#define AGX_RT_SCS_RAY_DIRECTION(axis)                                       \
   ((axis) * AGX_RT_SCS_RAY_AXIS_STRIDE + 4u)
#define AGX_RT_SCS_RAY_TMAX         24u
#define AGX_RT_SCS_RAY_PACKED_WORD  28u
#define AGX_RT_SCS_RAY_SEGMENT_SIZE 36u
#define AGX_RT_SCS_RAY_SLOT_B       36u
#define AGX_RT_SCS_SEGMENT0_SIZE    64u

/* TLAS instance-leaf offsets. */
#define AGX_RT_INSTANCE_LEAF_OFF_HEADER           0u
#define AGX_RT_INSTANCE_LEAF_OFF_OBJECT_TO_WORLD 12u
#define AGX_RT_INSTANCE_LEAF_OFF_AABB             60u
#define AGX_RT_INSTANCE_LEAF_OFF_IDS              84u
#define AGX_RT_INSTANCE_LEAF_OFF_INSTANCE_FLAGS  108u
#define AGX_RT_INSTANCE_LEAF_SIZE                124u

#ifdef __cplusplus
}
#endif
