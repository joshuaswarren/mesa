/*
 * Copyright © 2022 Konstantin Seurer
 * Copyright © 2024 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Push-constant blocks shared between the Honeykrisp acceleration-structure
 * encode shaders and the C code that dispatches them.  Everything the
 * encoder needs travels in push constants; there are no descriptor sets, so
 * a build never disturbs the application's bound descriptors.
 */

#ifndef HK_BVH_BUILD_INTERFACE_H
#define HK_BVH_BUILD_INTERFACE_H

#ifdef VULKAN
/* The Neo base uses vk_bvh_helpers for the same fixed IR nodes. */
#define VK_USED_BUILD_FLAGS 0
#include "vk_bvh_helpers.h"
#include "hk_bvh.h"

TYPE(hk_bvh_triangle_node, 4);
TYPE(hk_bvh_aabb_node, 4);
TYPE(hk_bvh_instance_node, 8);
TYPE(hk_bvh_box_node, 4);
TYPE(hk_bvh_header, 4);
#else
#include <stdint.h>

#include "hk_bvh.h"

#define REF(type) uint64_t
#define VOID_REF  uint64_t
#endif

#include "../../libagx/gen1_as_metadata.h"
#ifdef VULKAN
TYPE(libagx_gen1_as_metadata, 8);
#endif

/* Native refits (updates).
 *
 * A full build that allows updates keeps its encoding jobs after the image,
 * at the retained offset: an hk_agx3_retained_state, one hk_agx3_retained_job
 * per job, then the IR offsets of the children of every leaf job,
 * concatenated. An update keeps the topology: it regenerates the IR leaves
 * and re-encodes each leaf job from them, then each inner job from its
 * children's bounds: refit encoding (agx3_leaf.comp built with
 * HK_AGX3_REFIT) encodes the leaf jobs, then agx3_refit.comp the inner
 * jobs. It reads neither IR internal nodes nor a plan.
 *
 * A child job hands its bounds to its parent at its position: the parent's
 * first position plus the child's index. Positions number the children of
 * all inner jobs, so they run to job_count - 2; the root's is job_count - 1.
 *
 * Inner jobs have levels, a third of the binary depth of the job's IR node:
 * an inner job's inner children are one level deeper. After the child
 * offsets (room for every leaf) come the inner jobs of the first
 * HK_AGX3_REFIT_LEVELS levels by level: a count and a fill cursor per
 * level, then the jobs of level 0, of level 1, and so on. An update encodes
 * those a level per dispatch, deepest first, with no arrivals; refit
 * propagation encodes any deeper ones.
 *
 * Last come the slots of the leaves of a triangle structure: for each IR
 * leaf, the index of its child offset, or ~0 for a leaf in no leaf job. A
 * triangle update may skip generating IR leaves: its leaf pass leaves each
 * triangle (hk_agx3_refit_triangle) at its slot, so that each leaf job
 * finds its children together.
 *
 * The retained jobs are numbered leaf first: the leaf jobs (every job but
 * the inner ones) take ids from 0 and the inner jobs the rest, so that an
 * update's refit encoding and propagation cover only the leaf jobs.
 */
struct hk_agx3_retained_state {
   uint32_t job_count;
   uint32_t root_job;
   uint32_t end_slot;
   uint32_t depth;
   uint32_t deepest_level;
   /* The full build's active leaves: with updatable structures every
    * primitive is active (VK_BUILD_FLAG_ALWAYS_ACTIVE), so an update has as
    * many. */
   uint32_t active_leaves;
   /* The leaf and inner jobs: ids below leaf_jobs are leaf jobs. */
   uint32_t leaf_jobs;
   uint32_t inner_jobs;
};

struct hk_agx3_retained_job {
   uint32_t root_slot;
   uint32_t child_slot;
   uint32_t kind_count; /* child count | node kind << 4 | inner level << 8 */
   /* A leaf job's first child offset; an inner job's first position. */
   uint32_t first;
   uint32_t parent; /* 0xffffffff for the root */
   uint32_t position;
};

#ifdef VULKAN
TYPE(hk_agx3_retained_state, 4);
TYPE(hk_agx3_retained_job, 4);
#endif

#define HK_AGX3_RETAINED_STATE_BYTES 32u
#define HK_AGX3_RETAINED_JOB_BYTES 24u
/* Levels of inner jobs an update encodes level by level. */
#define HK_AGX3_REFIT_LEVELS 16u
#define HK_AGX3_RETAINED_LEVEL_BYTES (HK_AGX3_REFIT_LEVELS * 8u)
/* Where the retained jobs of a structure with `jobs` jobs and `leaves`
 * leaves keep their child offsets, level table (and the inner jobs by level
 * after it) and leaf slots, and their size. */
#define HK_AGX3_RETAINED_OFFSETS(jobs)                                         \
   (HK_AGX3_RETAINED_STATE_BYTES + (jobs) * HK_AGX3_RETAINED_JOB_BYTES)
#define HK_AGX3_RETAINED_LEVELS(jobs, leaves)                                  \
   (HK_AGX3_RETAINED_OFFSETS(jobs) + (leaves) * 4u)
#define HK_AGX3_RETAINED_SLOTS(jobs, leaves)                                   \
   (HK_AGX3_RETAINED_LEVELS(jobs, leaves) + HK_AGX3_RETAINED_LEVEL_BYTES + (jobs) * 4u)
#define HK_AGX3_RETAINED_BYTES(jobs, leaves)                                   \
   (HK_AGX3_RETAINED_SLOTS(jobs, leaves) + (leaves) * 4u)

/* Refit phases. A full build that allows updates saves its jobs, numbers
 * them leaf first, then counts its inner jobs by level and lists them. An
 * update prepares (after its leaves) to encode from them. After refit
 * encoding has encoded the leaf jobs, propagation encodes the inner jobs
 * of the levels past those the update encodes by level: each leaf job
 * below them arrives at its parent, and the last child of an inner job to
 * arrive encodes that job's node and arrives at its parent in turn. Then
 * the update encodes the inner jobs of one level per dispatch, deepest
 * first (inner). */
#define HK_AGX3_REFIT_SAVE 0u
#define HK_AGX3_REFIT_PREPARE 1u
#define HK_AGX3_REFIT_PROPAGATE 2u
#define HK_AGX3_REFIT_COUNT_LEVELS 3u
#define HK_AGX3_REFIT_LIST_LEVELS 4u
#define HK_AGX3_REFIT_INNER 5u
#define HK_AGX3_REFIT_NUMBER 6u

/* Written by the prepare phase in encode scratch, read by refit encoding. */
struct hk_agx3_refit_state {
   /* Refit encoding dispatch (VkDispatchIndirectCommand, workgroups of
    * HK_AGX3_NODE_JOBS_PER_GROUP jobs). */
   uint32_t dispatch[3];
   uint32_t job_count;
   /* Retained jobs this update encodes from. */
   uint64_t retained;
   /* Refit propagation dispatch, one thread per job. */
   uint32_t propagate[3];
   uint32_t end_slot;
};

#define HK_AGX3_REFIT_PROPAGATE_DISPATCH 24u /* bytes into hk_agx3_refit_state */
#define HK_AGX3_REFIT_GROUP_SIZE 64u

#ifdef VULKAN
TYPE(hk_agx3_refit_state, 8);
#endif

struct hk_agx3_refit_args {
   /* This structure's retained jobs: what the save phase writes, and where
    * the prepare phase copies the source's when updating into another
    * structure. */
   VOID_REF retained;
   /* Prepare: the source's metadata, which finds its retained jobs. */
   VOID_REF metadata;
   VOID_REF intermediate_bvh;
   REF(vk_ir_header) header;
   /* Save: the plan in encode scratch. */
   VOID_REF plan_nodes;
   VOID_REF plan_jobs;
   VOID_REF plan_state;
   /* Prepare: refit state, arrival counters, child bounds and the envelope
    * job, all in encode scratch. */
   VOID_REF refit;
   VOID_REF counters;
   VOID_REF bounds;
   VOID_REF envelope;
   VOID_REF output_bvh;
   uint32_t root_offset;
   uint32_t leaf_bound;
   uint32_t node_bound;
   uint32_t phase;
   uint32_t resource_id;
   uint32_t geometry_type;
   /* Prepare: the source is another structure. */
   uint32_t copy;
   /* Propagate and inner: as for node encoding. */
   uint32_t slot_bias;
   /* Propagate and inner: as for node encoding. Save and numbering: each
    * plan job's first child offset or position, then its retained id, in
    * the node statuses, which node encoding has finished with. */
   REF(uint32_t) status;
   /* Prepare: the level dispatches (VkDispatchIndirectCommands, workgroups
    * of HK_AGX3_REFIT_GROUP_SIZE / 8 jobs) to write. */
   VOID_REF levels;
   /* Prepare and propagate: the levels the update encodes by level; inner:
    * the level to encode. */
   uint32_t level;
   /* Prepare: the update generated no IR leaves, so no active leaf count. */
   uint32_t no_leaves;
};

/* The separate M5 backend keeps its qualified IR-topology refit protocol.
 * GEN1 uses hk_agx3_refit_args and the current retained-job protocol. */
struct hk_m5_refit_args {
   VOID_REF retained;
   VOID_REF metadata;
   VOID_REF intermediate_bvh;
   REF(vk_ir_header) header;
   VOID_REF parents;
   uint32_t root_offset;
   uint32_t leaf_bound;
   uint32_t node_bound;
   uint32_t phase;
   VOID_REF plan_nodes;
   VOID_REF plan_jobs;
   VOID_REF plan_state;
};

/* Push constants of refit encoding. */
struct hk_agx3_refit_encode_args {
   VOID_REF refit;
   VOID_REF bounds;
   VOID_REF intermediate_bvh;
   VOID_REF output_bvh;
   REF(uint32_t) status;
   uint32_t slot_bias;
   uint32_t leaf_bound;
   /* Nonzero when the update's leaf pass left its triangles at their slots
    * instead of generating IR leaves: triangle leaf jobs read their children
    * from these hk_agx3_refit_triangles at their child offsets' indices. */
   VOID_REF triangles;
};

/* A triangle as an update's leaf pass leaves it, at its leaf's slot: its
 * vertices, zero for an inactive triangle as updatable structures keep it,
 * and its geometry and primitive IDs, as an IR triangle leaf has them. */
struct hk_agx3_refit_triangle {
   float coords[3][3];
   uint32_t geometry_id_and_flags;
   uint32_t triangle_id;
   uint32_t padding;
};

#ifdef VULKAN
TYPE(hk_agx3_refit_triangle, 4);
#endif

/* Push constants of an update's leaf pass, one dispatch per geometry with
 * a thread per primitive. */
struct hk_agx3_refit_leaf_args {
   /* The refit state, which finds the retained jobs. */
   VOID_REF refit;
   REF(hk_agx3_refit_triangle) triangles;
   vk_bvh_geometry_data geometry;
   uint32_t leaf_bound;
   uint32_t primitive_count;
};


struct hk_encode_args {
   /* Base of the common builder's intermediate representation: the packed
    * leaf array, immediately followed by the internal box-node array. */
   VOID_REF intermediate_bvh;
   /* Base of the acceleration structure we are writing. */
   VOID_REF output_bvh;
   REF(vk_ir_header) header;
   /* Byte offset of the internal box-node array inside intermediate_bvh,
    * i.e. leaf_node_count * the IR leaf size. */
   uint32_t root_offset;
   /* Upper bound the dispatch was sized from; the shader still clamps
    * against the counts the builder wrote into the header. */
   uint32_t internal_node_bound;
   uint32_t leaf_node_bound;
   uint32_t geometry_type;
   /* What get_as_size() reserved, recorded into the header so a compacting
    * copy can find it without recomputing. */
   uint32_t as_size;
};

/* Native Gen3 node encoder. Jobs refer to Mesa's common triangle/box IR,
 * so the build planner can select/reorder leaves without copying vertices.
 * Each job owns disjoint root and child regions. Slots are relative to its
 * output AS, not the job or the root. The caller supplies the AS metadata.
 * This covers inner nodes, triangle leaf clusters, instance records and
 * procedural bounding-box leaf clusters. */
struct hk_agx3_leaf_job {
   VOID_REF intermediate_bvh;
   VOID_REF output_bvh;
   uint32_t root_slot;
   uint32_t child_slot;
   uint32_t child_count;
   uint32_t node_kind;
   uint32_t triangle_offsets[8];
};

#ifdef VULKAN
TYPE(hk_agx3_leaf_job, 8);
#endif

struct hk_agx3_leaf_args {
   REF(hk_agx3_leaf_job) jobs;
   REF(uint32_t) status;
   uint32_t job_count;
   /* Slots from the traversal base to output_bvh, added to every child
    * reference (see libagx/gen1_as.h). Zero when output_bvh is the base.
    * M5 uses this ABI word for its diagnostic bounding-box record stride. */
   uint32_t slot_bias;
   /* When nonzero, the plan state written by planning (job_count at 0,
    * next_slot at 12), which indirect or over-provisioned direct dispatches
    * clamp against and TLAS instance encoding uses for tail offsets. */
   VOID_REF counts;
   /* When nonzero alongside counts, TLAS instance encoding derives transform
    * and pointer table offsets from counts->next_slot and leaf_bound directly
    * instead of reading them back from the envelope header. */
   uint32_t leaf_bound;
   uint32_t padding;
};

/* Node jobs per encoding workgroup, eight lanes each. */
#define HK_AGX3_NODE_JOBS_PER_GROUP 4u

#define HK_AGX3_LEAF_SUCCESS 0
#define HK_AGX3_LEAF_INVALID_COUNT 1
#define HK_AGX3_LEAF_NONFINITE 2
#define HK_AGX3_LEAF_RANGE 4
#define HK_AGX3_LEAF_UNSUPPORTED_GEOMETRY 8
#define HK_AGX3_NODE_LEAF 0
#define HK_AGX3_NODE_INNER 1
#define HK_AGX3_NODE_INSTANCES 2
/* Procedural (AABB) leaf cluster: 48-byte native bounding-box records. */
#define HK_AGX3_NODE_AABBS 3

/* Finish a single opaque triangle geometry after the node jobs. Bounds name
 * a vk_ir_node from the common builder; metadata offsets remain AS-relative. */
struct hk_agx3_envelope_job {
   REF(vk_ir_node) bounds;
   VOID_REF output_bvh;
   uint32_t end_slot;
   uint32_t primitive_count;
   uint32_t node_count;
   uint32_t depth;
   uint32_t opaque_build_word;
   uint32_t resource_id;
   uint32_t geometry_type;
   uint32_t leaf_bound;
};

#ifdef VULKAN
TYPE(hk_agx3_envelope_job, 8);
#endif

struct hk_agx3_envelope_args {
   REF(hk_agx3_envelope_job) jobs;
   REF(uint32_t) status;
   uint32_t job_count;
   uint32_t reserved;
   VOID_REF metadata;
   REF(vk_ir_header) ir_header;
   uint64_t retained_offset;
   uint32_t build_flags;
   uint32_t node_bound;
   uint32_t keep_retained;
   /* A refit: the prepare phase cleared the instance pointers already. */
   uint32_t refit;
};

/* Scratch records for parallel binary-to-wide planning. Phases are separate
 * dispatches with storage barriers; no workgroup waits for another to run. */
struct hk_agx3_plan_node {
   uint32_t parent;
   uint32_t leaf_count; /* saturated at nine */
   uint32_t depth;
   uint32_t job;
};

struct hk_agx3_plan_state {
   /* The node encoding dispatch (VkDispatchIndirectCommand, workgroups of
    * HK_AGX3_NODE_JOBS_PER_GROUP jobs) follows this state, at
    * HK_AGX3_PLAN_ENCODE_DISPATCH; plan phase 4 writes it. */
   uint32_t job_count;
   uint32_t groups_y;
   uint32_t groups_z;
   uint32_t next_slot;
   uint32_t depth;
   uint32_t error;
   /* Leaf-job child offsets and positions handed out by the refit save
    * phase (see hk_agx3_retained_job). */
   uint32_t saved_offsets;
   uint32_t saved_positions;
};

#ifdef VULKAN
TYPE(hk_agx3_plan_node, 4);
TYPE(hk_agx3_plan_state, 4);
#endif

struct hk_agx3_plan_args {
   VOID_REF intermediate_bvh;
   REF(vk_ir_header) header;
   REF(hk_agx3_plan_node) nodes;
   REF(hk_agx3_leaf_job) jobs;
   REF(hk_agx3_envelope_job) envelope;
   REF(hk_agx3_plan_state) state;
   VOID_REF output_bvh;
   uint32_t root_offset;
   uint32_t leaf_bound;
   uint32_t node_bound;
   uint32_t slot_bound;
   uint32_t phase;
   uint32_t resource_id;
   uint32_t geometry_type;
   /* GEN1: phases 0..3 were replaced by a restored plan (an update): phase
    * 4 rebuilds the jobs from the records in nodes. Unused on M5. */
   uint32_t plan_restored;
   VOID_REF metadata;
   uint64_t retained_offset;
   uint32_t build_flags;
   uint32_t keep_retained;
   /* Nonzero to encode sibling triangle leaves that share an edge as one
    * native record (see hk_agx3_shared_edge); triangle geometry only. */
   uint32_t pairs;
   uint32_t padding;
};

/* Bytes of an M5 native bounding-box record (record class 4 in root word0
 * bits 5..7), packed at this stride in the leaf's child region: the marker
 * word (leaf type 24) at 0, the minimum at 4, the maximum at 16, the geometry
 * id with the opaque flag in bit 31 at 28, the primitive id at 32, its flags
 * word at 36 (bit 30: optional primitive data present) and the relative
 * primitive data pointer at 40. struct hk_agx3_m5_aabb_record spells it out
 * for the host. */
#define HK_AGX3_M5_AABB_RECORD_B 48u

#define HK_AGX3_PLAN_ENCODE_DISPATCH 32u /* bytes after hk_agx3_plan_state */
/* Single-workgroup fused planner limits (bvh/agx3_plan.comp). */
#define HK_AGX3_PLAN_FUSED_MAX_NODES 2048u
#define HK_AGX3_PLAN_FUSED_GROUP_SIZE 256u
/* hk_agx3_plan_node.depth flags of an internal node whose two children are
 * triangles encoded as one record, and whether that record is opaque. */
#define HK_AGX3_PLAN_PAIR 0x80000000u
#define HK_AGX3_PLAN_PAIR_OPAQUE 0x40000000u
/* Set in a leaf job's triangle offset when it names such an internal node. */
#define HK_AGX3_LEAF_PAIR_OFFSET 1u
#define HK_AGX3_PLAN_INDEX 1
#define HK_AGX3_PLAN_CYCLE 2
#define HK_AGX3_PLAN_ALLOCATION 4
#define HK_AGX3_PLAN_SHAPE 8
#define HK_AGX3_PLAN_EMPTY 16

#ifdef VULKAN
/* A native triangle record holds two triangles that share an edge with
 * opposite directions, as neighbours in a consistently wound mesh do. Edge
 * e of a triangle runs from its vertex e to vertex e + 1. Returns
 * e_a | e_b << 2 for the first such edge pair, or ~0u when there is none. */
uint
hk_agx3_shared_edge(vec3 a[3], vec3 b[3])
{
   for (uint ea = 0; ea < 3; ++ea)
      for (uint eb = 0; eb < 3; ++eb)
         if (all(equal(a[ea], b[(eb + 1u) % 3u])) &&
             all(equal(a[(ea + 1u) % 3u], b[eb])))
            return ea | (eb << 2);
   return ~0u;
}
#endif

#endif /* HK_BVH_BUILD_INTERFACE_H */
