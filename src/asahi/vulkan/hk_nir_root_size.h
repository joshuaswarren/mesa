/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "nir.h"
#include "util/macros.h"

/* Required prefix of the root table. Unknown/escaping accesses require full_size. */
unsigned hk_nir_root_size(const nir_shader *nir, unsigned full_size);

/* True when the shader may read any byte in [start_B, end_B) of the root
 * table. Unknown, dynamic, or escaping root uses return true (fail closed). */
bool hk_nir_root_reads_range(const nir_shader *nir, uint64_t start_B,
                             uint64_t end_B, unsigned full_size);

/* True when a compute NIR shader with API workgroup size <= 1024 (multiple of
 * 32) observes only global invocation ID, base_group, API num_workgroups, and
 * API workgroup_size (no shared memory, barriers, local/workgroup/subgroup/
 * quad/tile/ray/cmat intrinsics, calls, or implicit derivatives), so direct
 * zero-base dispatches may coalesce or split workgroups without altering
 * invocation coordinates or semantics. On direct dispatches,
 * root.cs.group_count_addr holds the unreshaped API group count and
 * load_workgroup_size folds to the compile-time API local size. */
bool hk_nir_can_coalesce_workgroup(const nir_shader *nir);

/* Coalesce a pure-global API workgroup shape (<= 256, multiple of 32) up to
 * MIN2(MIN2(target_wg_threads, 512), max_wg_threads), bounded per axis by
 * max_wg_size[], keeping at least MAX2(min_workgroups, 256) workgroups up to
 * 256 threads and at least MAX2(min_workgroups, 1024) workgroups above 256
 * threads. Exact-divisor steps run first (zero partial workgroups); if room
 * remains, bounded partial-edge steps (<= 6.25% inactive edge lanes, clipped
 * natively by AGX3 global-thread CDM launches) choose the axis minimizing
 * padded invocations. Returns true if out_local differs from api_local. */
bool hk_coalesce_workgroup_size_ex(const uint16_t api_local[3],
                                   const uint32_t grid_threads[3],
                                   const uint32_t max_wg_size[3],
                                   uint32_t max_wg_threads,
                                   uint32_t target_wg_threads,
                                   uint64_t min_workgroups,
                                   uint16_t out_local[3]);

/* Default coalescing target: up to 256 threads with at least 256 workgroups
 * and standard {1024, 1024, 64} per-axis bounds. */
bool hk_coalesce_workgroup_size(const uint16_t api_local[3],
                                const uint32_t grid_threads[3],
                                uint32_t max_wg_threads,
                                uint16_t out_local[3]);

/* Split a pure-global multi-subgroup API workgroup shape (multiple of 32,
 * >= 64 threads) along the outermost axis with extent > 1 when the dispatch
 * has fewer than min_workgroups workgroups, increasing parallelism on wide
 * multi-core GPUs while preserving exact 32-invocation subgroup membership.
 * Returns true if out_local differs from api_local. */
bool hk_split_workgroup_size_ex(const uint16_t api_local[3],
                                const uint32_t grid_threads[3],
                                uint64_t min_workgroups,
                                uint16_t out_local[3]);

static inline bool
hk_nir_dispatch_zero_base(const uint32_t base_group[3])
{
   return base_group[0] == 0 && base_group[1] == 0 && base_group[2] == 0;
}

/* The part of the root every launch of a stage reads, whatever its NIR.
 * Uniform lowering loads the root pointer and set bases separately, and the
 * root uniform range loads both (8 * (1 + set_count) bytes from the root), so
 * retain that prefix.  Hardware VS and FS prologs/epilogs access only the
 * draw arm preceding push constants, so flooring graphics stages at
 * push_offset preserves every driver-synthesized root load. */
static inline uint32_t
hk_shader_root_descriptor_floor(mesa_shader_stage sw_stage,
                                unsigned set_count, unsigned sets_offset,
                                unsigned push_offset, unsigned full_size)
{
   unsigned bytes = sets_offset + set_count * (unsigned)sizeof(uint64_t);
   if (sw_stage != MESA_SHADER_COMPUTE)
      bytes = MAX2(bytes, push_offset);
   return MIN2(ALIGN_POT(bytes, 16), full_size);
}

/* Compute the trimmed root descriptor upload size for a shader stage: its
 * floor (above) and every root byte the shader itself reads, trimming
 * unused push-constant, dynamic-buffer, and trailing tail bytes. */
static inline uint32_t
hk_shader_root_descriptor_size(const nir_shader *nir,
                               mesa_shader_stage sw_stage,
                               unsigned set_count, unsigned sets_offset,
                               unsigned push_offset, unsigned full_size)
{
   const unsigned bytes = hk_nir_root_size(nir, full_size);
   return MAX2(MIN2(ALIGN_POT(bytes, 16), full_size),
               hk_shader_root_descriptor_floor(sw_stage, set_count,
                                               sets_offset, push_offset,
                                               full_size));
}

/* Active graphics root prefix: maximum of the bound VS and optional FS
 * footprints, falling back to full_size whenever either stage lacks a proven
 * trimmed size. */
static inline size_t
hk_gfx_root_size_from_stages(uint32_t vs_size, bool has_fs, uint32_t fs_size,
                             size_t full_size)
{
   if (!vs_size)
      return full_size;

   size_t size = vs_size;
   if (has_fs) {
      if (!fs_size)
         return full_size;
      size = MAX2(size, (size_t)fs_size);
   }

   return MIN2(size, full_size);
}

/* Force a root re-upload when a newly bound shader needs more of the root
 * table than the last upload published. */
static inline bool
hk_root_upload_needs_growth(uint32_t uploaded_root_size,
                            size_t required_root_size)
{
   return uploaded_root_size < required_root_size;
}
