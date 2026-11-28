/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Honeykrisp bridge for bounded AGX3 VDM State Update template packing and
 * strict VDM record decoding.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "asahi/lib/agx_vdm_decode_core.h"
#include "asahi/lib/agx_chip.h"

struct blob;
struct blob_reader;

#ifdef __cplusplus
extern "C" {
#endif

/* Bump whenever the Honeykrisp shader payload below changes.  Vulkan shader
 * binaries carry this version outside the driver-private blob, so stale
 * layouts are rejected before any fields are consumed. */
/* AGX3 moves to 37 when M5 binaries begin to keep the NIR of uniform-colour
 * blended fragment shaders, to 38 when agx_shader_info gains
 * needs_full_quads, to 39 when G17P payloads record the fragment
 * twins of unlinked mesh shaders, to 42 when unlinked G17P vertex
 * shaders over the canonical size carry relocatable varying writes, to 43
 * when agx_shader_info gains launch_size, to 44 when unlinked G17P vertex
 * shaders drop the canonical layout for the relocatable one, to 45 for
 * attachmentless layer-zero root lowering and fs.reads_layer, to 46 when
 * AGX3 fragment binaries carry on-chip local_read NIR payloads, to 47
 * when agx_shader_info gains AGX3 vertex preamble hoist metadata and
 * vertex binaries may carry extracted preamble NIR, to 48 when
 * hk_shader_info.vs records per-pipeline vertex input robustness, to 49
 * when compute shaders carry packed vec4 alternates, and to 50 when they
 * carry compute-pair state in a third variant byte. AGX3 moves to 51 and
 * classic to 39 when compute-pair binaries also carry the producer state
 * needed to compile fused variants, and to 52 / 40 when compute shaders
 * carry their proven submit-pack loop bounds. */
#define HK_AGX3_SHADER_BINARY_VERSION 54u
#define HK_CLASSIC_SHADER_BINARY_VERSION 41u
/* AGX2 binaries include attachmentless layer-zero root lowering, fs.reads_layer,
 * and per-pipeline vertex input robustness in hk_shader_info.vs. */
#define HK_AGX2_SHADER_BINARY_VERSION 39u

static inline uint32_t
hk_shader_binary_version(enum agx_chip chip)
{
   if (agx_chip_isa(chip) == AGX_ISA_AGX2)
      return HK_AGX2_SHADER_BINARY_VERSION;
   if (agx_chip_g17p_or_m5(chip))
      return HK_AGX3_SHADER_BINARY_VERSION;
   return HK_CLASSIC_SHADER_BINARY_VERSION;
}

/* API outputs include position. Unlinked shaders additionally reserve a
 * default Layer varying and PointSize, up to eight fixed clip-distance copies,
 * and one packed layer/viewport word. The 125 ordinary words plus RHW and
 * FragCoordZ still fit the separate 127-coefficient bound.
 */
#define HK_AGX3_G17P_MAX_UVS_WORDS 139u
#define HK_AGX3_G17P_MAX_USER_WORDS 125u
#define HK_AGX3_G17P_MAX_COEFFICIENTS 127u

bool hk_agx3_vdm_program_cache_serialize(
   struct blob *blob, const struct agx_vdm_state_update_program_info *info);

bool hk_agx3_vdm_program_cache_deserialize(
   struct blob_reader *blob, struct agx_vdm_state_update_program_info *info);

enum agx_vdm_state_update_source_layout
hk_agx3_vdm_live_source_layout(
   const struct agx_vdm_state_update_program_info *program,
   bool active_context_selector_valid, bool ordinary_draw_path);

struct hk_agx3_g17p_vertex_compile_info {
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

/* Build the exact G17P VertexProgramVariant fields for a hardware vertex
 * shader with position, ordinary user locations, and the Vulkan-bounded
 * eight-word combined clip/cull-distance tail. System outputs are a separate
 * state shape. A constant program is admitted only when
 * `constant_program_allowed` is set; its thread-invariant spill (the CProg's
 * imported-input TPS frame) and staging footprint then populate the two
 * state-load counts (agx_vdm_state_update_vertex_ti_resources). */
bool hk_agx3_g17p_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program);

/* HAL300 hardware vertex profiles: position, optional point size, and one
 * float4 user color. No preamble. Pre-HAL300 flag bits are always absent. */
bool hk_agx3_g17g_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program);

/*
 * Decode one record from a potentially longer AGX3 VDM stream through the
 * shared Mesa classifier. This is a validation surface only: it does not
 * follow links, inspect PPP payloads, emit commands, submit work, admit an
 * AGX3 device, or expose a Vulkan capability.
 */
bool hk_agx3_decode_vdm_record(enum agx_chip chip, const void *command,
                               size_t available,
                               struct agx_vdm_decoded_record *record);

/* Validation-only structural bridge. */
bool hk_agx3_encode_vdm_state_update_template(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out);

/* Exact AGX3 VertexProgramVariant selector bridge used by live draw emission. */
bool hk_agx3_encode_vdm_state_update_draw(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out);

#ifdef __cplusplus
}
#endif
