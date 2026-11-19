#pragma once
#include "compiler/nir/nir.h"
struct agx_gen1_ray_query_layout {
   unsigned payload_uniform, bvh_uniform, status_uniform;
   uint32_t launch_header;
   unsigned context_uniform, as_table_uniform;
   unsigned dispatch_group_count_offset;
   unsigned as_table_first;
   bool instance_traversal, candidates;
   bool fragment_shader;
   bool graphics_stage;
   bool compute_kernel;
   bool position_fetch;
   bool diagnostic_features;
};
#define AGX_GEN1_VERTEX_DIRECTORY_VA UINT64_C(0x10040000000)
#define AGX_GEN1_VERTEX_DIRECTORY_BYTES 0x8000u
#define AGX_GEN1_VERTEX_TABLE_MAGIC 0x534f5056u
nir_def *agx_nir_gen1_vertex_position_address(struct nir_builder *b, nir_def *instance,
                                              nir_def *geometry, nir_def *primitive);
bool agx_nir_lower_gen1_ray_query(nir_shader *shader,
   const struct agx_gen1_ray_query_layout *layout);

/* Native G17P candidate profile: 20-byte records, ten records per lane.
 * These helpers read an existing queue; they do not proceed or confirm it. */
struct agx_gen1_ray_candidate {
   nir_def *compressed_address, *flags, *barycentrics, *distance;
};
struct nir_builder;
struct agx_gen1_ray_candidate
agx_nir_gen1_load_candidate(struct nir_builder *b, nir_def *query,
                            nir_def *lane, nir_def *index);

nir_def *agx_nir_gen1_candidate_instance_pointer(struct nir_builder *b,
                                                nir_def *query, nir_def *lane);

nir_def *agx_nir_gen1_refill_candidate_queue(struct nir_builder *b, nir_def *query,
                                        nir_def *lane, nir_def *id,
                                        nir_def *best_distance);
