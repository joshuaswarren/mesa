/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM State Update representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_STATE_UPDATE_LENGTH 28u
#define AGX_VDM_STATE_UPDATE_HEADER UINT32_C(0x4000002e)

struct agx_vdm_state_update_info {
   uint32_t raw_vertex_shader_word0;
   uint64_t pipeline;
   uint32_t raw_pipeline_template;
   uint32_t raw_vertex_outputs[2];
   uint32_t raw_vertex_unknown;
};

struct agx_vdm_state_update {
   uint8_t bytes[AGX_VDM_STATE_UPDATE_LENGTH];
   uint8_t length;
};

/* The live renderer selects one of six 28-byte templates.  The low selector
 * bit is boolean; the high axis is the three-valued provoking-vertex value
 * stored by setProvokingVertexMode:. */
enum agx_vdm_state_update_template_variant {
   AGX_VDM_STATE_UPDATE_TEMPLATE_00 = 0,
   AGX_VDM_STATE_UPDATE_TEMPLATE_01 = 1,
   AGX_VDM_STATE_UPDATE_TEMPLATE_10 = 2,
   AGX_VDM_STATE_UPDATE_TEMPLATE_11 = 3,
   AGX_VDM_STATE_UPDATE_TEMPLATE_20 = 4,
   AGX_VDM_STATE_UPDATE_TEMPLATE_21 = 5,
};

/* Callers must explicitly select the AGX3 VertexProgramVariant layout;
 * zero, legacy AGX2, and unknown values fail closed. */
enum agx_vdm_state_update_source_layout {
   AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED = 0,
   AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX2_LEGACY = 1,
   AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT = 2,
};

enum agx_vdm_state_update_draw_path {
   AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED = 0,
   AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED = 1,
   AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED = 2,
   AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED = 3,
};

/* Template inputs for the AGX3 28-byte VDM State Update record. */
struct agx_vdm_state_update_template_info {
   uint64_t pipeline;

   uint16_t raw_state_load_count_12;
   uint32_t raw_state_load_count_18;

   uint8_t raw_vs_output_count_0;
   uint8_t raw_vs_output_count_2;
   bool raw_vs_output_count_1_extra;
   uint8_t raw_vs_output_selector;
   bool raw_vs_output_flag_6;

   /* Pre-HAL300 (G15/G16/G17P) only.  These are respectively
    * `(flags & 0xc) != 0` and `(flags & 0x10) != 0` from VsStateConfig.
    * HAL300 does not consume them and rejects nonzero values rather than
    * silently discarding them. */
   bool pre_hal300_flag_group_nonzero;
   bool pre_hal300_flag_bit_4;

   enum agx_vdm_state_update_template_variant variant;

   /* UINT32_MAX preserves a zero final word; any other value encodes
    * `(raw_active_context_selector - 1) & 7`. */
   uint32_t raw_active_context_selector;
};

/* Shader-lifetime portion of the VertexProgramVariant state.  Zero-filled
 * shader objects remain UNSPECIFIED and are rejected.  The draw path replaces
 * `template_info.pipeline` with its freshly uploaded pipeline pointer, ignores
 * `template_info.variant` (selected dynamically), and replaces
 * `template_info.raw_active_context_selector` from render-context state. */
struct agx_vdm_state_update_program_info {
   enum agx_vdm_state_update_source_layout source_layout;
   struct agx_vdm_state_update_template_info template_info;
};

/*
 * Exact renderer-side selector inputs.  The low bit is the pipeline's
 * rasterization-enabled property, fixed when the pipeline is built. The
 * provoking selector is the exact stored renderer value, whose accepted domain
 * is 0..2.  The renderer computes
 *
 *   variant = rasterization_enabled |
 *             (raw_provoking_selector << 1)
 *
 * before indexing all six 28-byte VertexProgramVariant templates.
 */
struct agx_vdm_state_update_draw_info {
   enum agx_vdm_state_update_source_layout source_layout;
   enum agx_vdm_state_update_draw_path draw_path;
   bool rasterization_enabled;
   uint8_t raw_provoking_selector;
   struct agx_vdm_state_update_template_info template_info;
};

/* Pack/unpack the 7-word AGX3 VDM State Update (header 0x4000002e, 64B-aligned
 * 48-bit vertex pipeline pair, and raw vertex-shader/output/unknown words). */
bool agx_vdm_state_update_supported(enum agx_chip chip);

bool agx_vdm_state_update_pack(enum agx_chip chip,
                               const struct agx_vdm_state_update_info *info,
                               struct agx_vdm_state_update *out);

bool agx_vdm_state_update_unpack(enum agx_chip chip, const void *bytes,
                                 size_t size,
                                 struct agx_vdm_state_update_info *info);

bool agx_vdm_state_update_template_materialize(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update_info *raw);

bool agx_vdm_state_update_template_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out);

bool agx_vdm_state_update_vertex_ti_resources(
   enum agx_chip chip, uint32_t spill_bytes, uint32_t largest_ipr_bytes,
   struct agx_vdm_state_update_template_info *info);

/* Select and pack the 28-byte VDM State Update prefix for a draw.
 * Clears `out` on failure. */
bool agx_vdm_state_update_draw_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out);

/* Final compiler UVS and CProg resources shared by native API frontends.
 * This describes shader state only; callers still own context and draw gates. */
struct agx_g17p_vertex_compile_info {
   bool hardware_vertex;
   bool has_preamble;
   /* The caller compiled the preamble as a native constant program (inputs
    * imported through its TPS frame). Without it a preamble is refused. */
   bool constant_program_allowed;
   /* The constant program's texture staging footprint, published as its
    * thread-invariant largest-IPR bytes (as for the fragment CProg). */
   bool preamble_staging_known;
   uint16_t preamble_staging_halfs;
   uint16_t nr_preamble_gprs;
   uint32_t preamble_scratch_size;
   uint64_t uvs_written;
   uint8_t position_off;
   uint8_t varyings_off;
   uint8_t psiz_off;
   uint8_t layer_viewport_off;
   uint8_t clip_off;
   uint8_t uvs_size;
   uint8_t user_size;
};

bool agx_g17p_vertex_program_info(
   enum agx_chip chip, const struct agx_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program);

#ifdef __cplusplus
}
#endif
