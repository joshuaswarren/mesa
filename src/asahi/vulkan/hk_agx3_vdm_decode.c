/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_vdm_decode.h"

#include "compiler/shader_enums.h"
#include "util/blob.h"
#include "util/bitscan.h"

#include <string.h>

bool
hk_agx3_vdm_program_cache_serialize(
   struct blob *blob, const struct agx_vdm_state_update_program_info *info)
{
   if (!blob || !info)
      return false;

   const struct agx_vdm_state_update_template_info *template =
      &info->template_info;
   blob_write_uint32(blob, info->source_layout);
   blob_write_uint64(blob, template->pipeline);
   blob_write_uint16(blob, template->raw_state_load_count_12);
   blob_write_uint32(blob, template->raw_state_load_count_18);
   blob_write_uint8(blob, template->raw_vs_output_count_0);
   blob_write_uint8(blob, template->raw_vs_output_count_2);
   blob_write_uint8(blob, template->raw_vs_output_count_1_extra);
   blob_write_uint8(blob, template->raw_vs_output_selector);
   blob_write_uint8(blob, template->raw_vs_output_flag_6);
   blob_write_uint8(blob, template->pre_hal300_flag_group_nonzero);
   blob_write_uint8(blob, template->pre_hal300_flag_bit_4);
   blob_write_uint32(blob, template->variant);
   blob_write_uint32(blob, template->raw_active_context_selector);
   return !blob->out_of_memory;
}

bool
hk_agx3_vdm_program_cache_deserialize(
   struct blob_reader *blob, struct agx_vdm_state_update_program_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));
   if (!blob || !info)
      return false;

   const uint32_t source_layout = blob_read_uint32(blob);
   const uint64_t pipeline = blob_read_uint64(blob);
   const uint16_t state_load_count_12 = blob_read_uint16(blob);
   const uint32_t state_load_count_18 = blob_read_uint32(blob);
   const uint8_t output_count_0 = blob_read_uint8(blob);
   const uint8_t output_count_2 = blob_read_uint8(blob);
   const uint8_t output_count_1_extra = blob_read_uint8(blob);
   const uint8_t output_selector = blob_read_uint8(blob);
   const uint8_t output_flag_6 = blob_read_uint8(blob);
   const uint8_t pre_hal300_flag_group_nonzero = blob_read_uint8(blob);
   const uint8_t pre_hal300_flag_bit_4 = blob_read_uint8(blob);
   const uint32_t variant = blob_read_uint32(blob);
   const uint32_t active_context_selector = blob_read_uint32(blob);

   if (blob->overrun ||
       source_layout >
          AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT ||
       variant > AGX_VDM_STATE_UPDATE_TEMPLATE_21 ||
       output_count_1_extra > 1 || output_flag_6 > 1 ||
       pre_hal300_flag_group_nonzero > 1 || pre_hal300_flag_bit_4 > 1)
      return false;

   *info = (struct agx_vdm_state_update_program_info){
      .source_layout =
         (enum agx_vdm_state_update_source_layout)source_layout,
      .template_info = {
         .pipeline = pipeline,
         .raw_state_load_count_12 = state_load_count_12,
         .raw_state_load_count_18 = state_load_count_18,
         .raw_vs_output_count_0 = output_count_0,
         .raw_vs_output_count_2 = output_count_2,
         .raw_vs_output_count_1_extra = output_count_1_extra,
         .raw_vs_output_selector = output_selector,
         .raw_vs_output_flag_6 = output_flag_6,
         .pre_hal300_flag_group_nonzero =
            pre_hal300_flag_group_nonzero,
         .pre_hal300_flag_bit_4 = pre_hal300_flag_bit_4,
         .variant = (enum agx_vdm_state_update_template_variant)variant,
         .raw_active_context_selector = active_context_selector,
      },
   };
   return true;
}

enum agx_vdm_state_update_source_layout
hk_agx3_vdm_live_source_layout(
   const struct agx_vdm_state_update_program_info *program,
   bool active_context_selector_valid, bool ordinary_draw_path)
{
   return program && ordinary_draw_path && active_context_selector_valid &&
                 program->source_layout ==
                    AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
             ? AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
             : AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED;
}

bool
hk_agx3_g17p_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program)
{
   if (!compiler_info)
      return agx_g17p_vertex_program_info(chip, NULL, program);
   const struct agx_g17p_vertex_compile_info shared = {
      .hardware_vertex = compiler_info->hardware_vertex,
      .has_preamble = compiler_info->has_preamble,
      .constant_program_allowed = compiler_info->constant_program_allowed,
      .preamble_staging_known = compiler_info->preamble_staging_known,
      .preamble_staging_halfs = compiler_info->preamble_staging_halfs,
      .nr_preamble_gprs = compiler_info->nr_preamble_gprs,
      .preamble_scratch_size = compiler_info->preamble_scratch_size,
      .uvs_written = compiler_info->uvs_written,
      .position_off = compiler_info->position_off,
      .varyings_off = compiler_info->varyings_off,
      .psiz_off = compiler_info->psiz_off,
      .layer_viewport_off = compiler_info->layer_viewport_off,
      .clip_off = compiler_info->clip_off,
      .uvs_size = compiler_info->uvs_size,
      .user_size = compiler_info->user_size,
   };
   return agx_g17p_vertex_program_info(chip, &shared, program);
}

bool
hk_agx3_g17g_vertex_program_info(
   enum agx_chip chip,
   const struct hk_agx3_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program)
{
   const bool position = compiler_info && compiler_info->uvs_size == 4 &&
                         compiler_info->uvs_written == 0;
   /* Dynamic-topology pipelines retain the default point-size output even
    * for triangle draws. Use the shared UVS footprint encoding for that
    * fifth word; layer/viewport output remains a separate profile.
    */
   const bool position_psiz = compiler_info &&
      compiler_info->uvs_size == 5 && compiler_info->psiz_off == 4 &&
      compiler_info->layer_viewport_off == 5 && compiler_info->clip_off == 5 &&
      compiler_info->uvs_written == 0;
   /* Native applegpu_g17g color and color+point-size vertex pipelines report
    * __GPU_LD_MD root.field1.field3 = 8/9 with field4 absent. This is the
    * same all-f32 VsStateConfig footprint consumed by the shared encoder,
    * with HAL300's pre-HAL300 flags absent. Fragment coefficient state is
    * validated independently before any draw can be submitted.
    */
   const bool color = compiler_info &&
      compiler_info->user_size == 4 &&
      compiler_info->uvs_written == BITFIELD64_BIT(VARYING_SLOT_VAR0) &&
      compiler_info->psiz_off == 8 &&
      (compiler_info->layer_viewport_off == 8 ||
       compiler_info->layer_viewport_off == 9) &&
      compiler_info->clip_off == compiler_info->layer_viewport_off;
   if (!agx_chip_hal300_records(chip) || !compiler_info ||
       (chip == AGX_CHIP_G17G &&
        (!color && ((!position && !position_psiz) ||
                    compiler_info->user_size != 0)))) {
      if (program)
         memset(program, 0, sizeof(*program));
      return false;
   }
   /* G18G: the M5 GL path builds the draw record from the shader's actual
    * UVS footprint for general layouts, so admit every layout that passes
    * the shared invariant-checked builder; the profile restriction above is
    * the G17G witness set. */
   return hk_agx3_g17p_vertex_program_info(chip, compiler_info, program);
}

bool
hk_agx3_decode_vdm_record(enum agx_chip chip, const void *command,
                          size_t available,
                          struct agx_vdm_decoded_record *record)
{
   return agx_vdm_decode_one(chip, command, available, record);
}

bool
hk_agx3_encode_vdm_state_update_template(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out)
{
   return agx_vdm_state_update_template_pack(chip, info, out);
}

bool
hk_agx3_encode_vdm_state_update_draw(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out)
{
   return agx_vdm_state_update_draw_pack(chip, info, out);
}
