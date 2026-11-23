/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Construction and decoding of the directly emitted AGX3 CDM compute launch:
 * the 40-byte launch record and its template words.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_CDM_DIRECT_LAUNCH_SIZE 40u

/* Finalize supplies only these directly recovered template bit ranges. */
#define AGX3_CDM_DIRECT_TEMPLATE_WORD0_MASK UINT32_C(0x00887ff0)
#define AGX3_CDM_DIRECT_TEMPLATE_WORD1_MASK UINT32_C(0x001ffff8)

/* The two direct API paths have different grid sources but serialize the same
 * canonical 40-byte record containing global thread counts.  The source is
 * intentionally absent from the decoded record because it is not encoded. */
enum agx3_cdm_direct_grid_source {
   AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS = 0,
   AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS = 1,
};

struct agx3_cdm_direct_template_fields {
   uint16_t word0_bits_3_14;
   uint32_t word1_bits_3_20;
   bool ray_traversal_depth_nonzero;
   bool subkernel_x_target_workitems;
};

/* Inputs to the exact direct-template transforms: the constant
 * program's thread-invariant (TI) resources, the TI spill and TI IPR byte
 * counts the compiler reports, the same pair the vertex and fragment
 * programs carry in their stage state. */
struct agx3_cdm_direct_template_sources {
   /* thread_invariant_spill_buffer_bytes: the constant program's TPS frame. */
   uint32_t ti_spill_bytes;

   /* thread_invariant_largest_ipr_bytes: its staging footprint. */
   uint32_t ti_largest_ipr_bytes;

   /* Finalize consumes only whether ShaderInfo ID32 is nonzero.  Its full
    * 32-bit RT metadata subfields remain outside this launch-only contract. */
   bool ray_traversal_depth_nonzero;
};

struct agx3_cdm_direct_template {
   uint32_t word0;
   uint32_t word1;
};

struct agx3_cdm_direct_launch_info {
   enum agx_chip chip;
   struct agx3_cdm_direct_template template_words;
   uint64_t pipeline;
   uint8_t dimensions;

   /* Word 3 bit 31 is set for multi-cluster parts, and only on the
    * aggregate G15X family (G15S, G15C and G15D here). */
   bool g15x_has_multiple_agcs;

   uint32_t global_size[3];
   uint32_t local_size[3];
};

bool agx3_cdm_direct_launch_target_grounded(enum agx_chip chip);

/* Resolve either direct API's grid source into the serialized global thread
 * counts.  `global_size` is cleared on every failure. */
bool agx3_cdm_direct_grid_resolve(
   enum agx3_cdm_direct_grid_source source, const uint32_t input_size[3],
   const uint32_t local_size[3], uint32_t global_size[3]);

bool agx3_cdm_direct_template_build_from_sources(
   const struct agx3_cdm_direct_template_sources *sources,
   struct agx3_cdm_direct_template *result);

/* Derive the exact direct-compute ShaderInfo source tuple witnessed on G17X.
 * max_total_threads_per_threadgroup is an Apple schema name, not the Metal
 * pipeline limit: AGC repurposes it as the table below.  The extended texture
 * write field remains zero even with 65 live writable textures.  ShaderInfo
 * ID32/launch bit23 is an independent committed-path property and is rejected
 * if supplied without traversal use. */
bool agx3_cdm_direct_template_sources_compute(
   bool uses_resource_info_query, bool uses_ray_traversal,
   bool ray_traversal_depth_nonzero,
   struct agx3_cdm_direct_template_sources *sources);

bool agx3_cdm_direct_template_decode(
   const struct agx3_cdm_direct_template *template_words,
   struct agx3_cdm_direct_template_fields *fields);

/* Pack one canonical 40-byte record.  `packed` is cleared on every failure. */
bool agx3_cdm_direct_launch_pack(
   const struct agx3_cdm_direct_launch_info *info,
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE]);

/* Decode the first canonical record in a stream.  `available` may include
 * following records.  `info` is cleared on every failure. */
bool agx3_cdm_direct_launch_unpack(
   enum agx_chip chip, const void *packed, size_t available,
   struct agx3_cdm_direct_launch_info *info);

#ifdef __cplusplus
}
#endif
