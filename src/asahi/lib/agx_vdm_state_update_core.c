/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_state_update_core.h"
#include "agx_bytes.h"

#include "agx_vdm_state_core.h"

#include <string.h>

/*
 * Pack/unpack the complete seven-word AGX3 VDM State Update emitted by the
 * inspected Apple VDMEncoderGen5 and VDMEncoderGen6 paths:
 *
 *   word 0: exact 0x4000002e header
 *   word 1: raw vertex-shader word 0
 *   words 2..3: widened 48-bit vertex-pipeline pair
 *   words 4..5: raw two-word vertex-output payload
 *   word 6: raw vertex-unknown payload
 *
 * Only the pipeline field's 64-byte alignment and 48-bit address range are
 * fully named.  The remaining payloads are preserved losslessly because their
 * complete AGX3 bit semantics are not independently grounded.
 *
 * The target allowlist contains only identities directly served by the
 * inspected Apple Metal plugins.  G16P and G17A remain closed despite sharing
 * neighboring command-stream generations.  This helper does not emit a live
 * command or enable lowering, scheduling, submission, or device admission.
 */
bool
agx_vdm_state_update_supported(enum agx_chip chip)
{
   return agx_vdm_state_vertex_pipeline_pair_supported(chip);
}

bool
agx_vdm_state_update_pack(enum agx_chip chip,
                          const struct agx_vdm_state_update_info *info,
                          struct agx_vdm_state_update *out)
{
   struct agx_vdm_state_vertex_pipeline_pair pair;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info || !agx_vdm_state_vertex_pipeline_pair_pack(
                           chip, info->pipeline, info->raw_pipeline_template,
                           &pair))
      return false;

   _Static_assert(sizeof(pair.bytes) == 8,
                  "AGX3 VDM pipeline pair must occupy two words");
   agx_store_le32(&out->bytes[0],
                                   AGX_VDM_STATE_UPDATE_HEADER);
   agx_store_le32(&out->bytes[4],
                                   info->raw_vertex_shader_word0);
   memcpy(&out->bytes[8], pair.bytes, sizeof(pair.bytes));
   agx_store_le32(&out->bytes[16],
                                   info->raw_vertex_outputs[0]);
   agx_store_le32(&out->bytes[20],
                                   info->raw_vertex_outputs[1]);
   agx_store_le32(&out->bytes[24],
                                   info->raw_vertex_unknown);
   out->length = AGX_VDM_STATE_UPDATE_LENGTH;
   return true;
}

bool
agx_vdm_state_update_unpack(enum agx_chip chip, const void *bytes, size_t size,
                            struct agx_vdm_state_update_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || size != AGX_VDM_STATE_UPDATE_LENGTH ||
       !agx_vdm_state_update_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(&raw[0]) !=
       AGX_VDM_STATE_UPDATE_HEADER)
      return false;

   if (!agx_vdm_state_vertex_pipeline_pair_unpack(
          chip, &raw[8], AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH,
          &info->pipeline, &info->raw_pipeline_template)) {
      memset(info, 0, sizeof(*info));
      return false;
   }

   info->raw_vertex_shader_word0 = agx_load_le32(&raw[4]);
   info->raw_vertex_outputs[0] = agx_load_le32(&raw[16]);
   info->raw_vertex_outputs[1] = agx_load_le32(&raw[20]);
   info->raw_vertex_unknown = agx_load_le32(&raw[24]);
   return true;
}

/*
 * Materialize/pack the exact six-template path observed in:
 *
 *   G15X_M1, G16X, G17P: VertexProgramVariant + VDMEncoderGen5
 *   G17G, G17X:          VertexProgramVariant + VDMEncoderGen6
 *
 * HAL300 intentionally differs: vertex word 0 has no pre-HAL300 bit-31 rule,
 * and the six per-template output constants are 0x20/0/0x21/1/0x22/2 instead
 * of 0x30/0/0x31/1/0x32/2.  Both calls are transactional and fail closed
 * outside the directly observed target/input domains.
 */
bool
agx_vdm_state_update_template_materialize(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update_info *raw)
{
   if (raw)
      memset(raw, 0, sizeof(*raw));

   if (!raw || !info || !agx_vdm_state_update_supported(chip) ||
       info->raw_state_load_count_12 > UINT16_C(0x0fff) ||
       info->raw_state_load_count_18 > UINT32_C(0x0003ffff) ||
       info->variant < AGX_VDM_STATE_UPDATE_TEMPLATE_00 ||
       info->variant > AGX_VDM_STATE_UPDATE_TEMPLATE_21)
      return false;

   const bool hal300 = agx_chip_hal300_path(chip);
   if (hal300 && (info->pre_hal300_flag_group_nonzero ||
                  info->pre_hal300_flag_bit_4))
      return false;

   uint32_t vertex_shader_word0 =
      (uint32_t)info->raw_state_load_count_12 << 8;

   if (!hal300 && (!info->pre_hal300_flag_group_nonzero ||
                   info->pre_hal300_flag_bit_4))
      vertex_shader_word0 |= UINT32_C(1) << 31;

   const uint32_t output_count_1 =
      ((uint32_t)info->raw_vs_output_count_0 +
       (uint32_t)info->raw_vs_output_count_2 +
       (info->raw_vs_output_count_1_extra ? 1u : 0u)) &
      UINT32_C(0xff);

   const uint32_t vertex_outputs_word0 =
      (uint32_t)info->raw_vs_output_count_0 | (output_count_1 << 8) |
      ((uint32_t)info->raw_vs_output_count_2 << 16);

   static const uint8_t pre_hal300_variant_bits[6] = {
      0x30, 0x00, 0x31, 0x01, 0x32, 0x02,
   };
   static const uint8_t hal300_variant_bits[6] = {
      0x20, 0x00, 0x21, 0x01, 0x22, 0x02,
   };
   const uint8_t *variant_bits =
      hal300 ? hal300_variant_bits : pre_hal300_variant_bits;

   const uint32_t vertex_outputs_word1 =
      ((uint32_t)info->raw_vs_output_selector << 2) |
      (info->raw_vs_output_flag_6 ? UINT32_C(0x40) : 0) |
      variant_bits[info->variant];

   const struct agx_vdm_state_update_info candidate = {
      .raw_vertex_shader_word0 = vertex_shader_word0,
      .pipeline = info->pipeline,
      .raw_pipeline_template = info->raw_state_load_count_18 << 3,
      .raw_vertex_outputs = {vertex_outputs_word0, vertex_outputs_word1},
      .raw_vertex_unknown =
         info->raw_active_context_selector == UINT32_MAX
            ? 0
            : (info->raw_active_context_selector - 1) & UINT32_C(0x7),
   };
   struct agx_vdm_state_update validation;

   /* Reuse the raw packer's address and target validation. */
   if (!agx_vdm_state_update_pack(chip, &candidate, &validation))
      return false;

   *raw = candidate;
   return true;
}

bool
agx_vdm_state_update_template_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_template_info *info,
   struct agx_vdm_state_update *out)
{
   struct agx_vdm_state_update_info raw;

   if (out)
      memset(out, 0, sizeof(*out));

   return out &&
          agx_vdm_state_update_template_materialize(chip, info, &raw) &&
          agx_vdm_state_update_pack(chip, &raw, out);
}

/*
 * Publish a vertex constant program's thread-invariant (TI) resources in the
 * two state-load counts, from the compiler's TI spill-buffer bytes and TI
 * largest-IPR bytes. G17P and HAL300.
 *
 * G17P (HAL200):
 *
 *   The two counts are
 *     spill = (thread_invariant_spill_buffer_bytes + 3) >> 2
 *     ipr   = (((thread_invariant_largest_ipr_bytes + 3) >> 2) + 1) & ~1
 *   (the IPR byte count is 0 when absent).
 *
 *   Every one of the six templates carries them as
 *     vertex-shader word 0 |= ipr << 8     (12-bit field at bit 8)
 *     pipeline template     = spill << 3   (18-bit field at bit 3)
 *
 * i.e. raw_state_load_count_12 is the TI IPR quad count and
 * raw_state_load_count_18 the TI spill quad count. The fragment variant
 * publishes the identical pair in its HAL200 record
 * (agx_ppp_hal200_fragment_ti_resources). A vertex shader without a constant
 * program leaves both zero. Fails closed, leaving `info` unchanged, outside
 * the encodable range.
 *
 * HAL300 uses the same two quad counts and packs each template's vertex-shader
 * word 0 as IPR << 8 (12 bits) and its pipeline template as spill << 3 (18
 * bits); the HAL300 form only omits bit 31.
 */
bool
agx_vdm_state_update_vertex_ti_resources(
   enum agx_chip chip, uint32_t spill_bytes, uint32_t largest_ipr_bytes,
   struct agx_vdm_state_update_template_info *info)
{
   const uint64_t spill_quads = ((uint64_t)spill_bytes + 3) >> 2;
   const uint64_t ipr_quads = ((((uint64_t)largest_ipr_bytes + 3) >> 2) + 1) &
                              ~UINT64_C(1);

   /* Same bounds as the HAL200 fragment record: an 18-bit spill field and
    * the 12-bit IPR field, whose even rounding caps it at 0xffe. */
   if (!agx_chip_agx3_path(chip) ||
       !info || spill_quads > UINT32_C(0x3ffff) ||
       ipr_quads > UINT32_C(0xffe))
      return false;

   info->raw_state_load_count_12 = (uint16_t)ipr_quads;
   info->raw_state_load_count_18 = (uint32_t)spill_quads;
   return true;
}

bool
agx_vdm_state_update_draw_pack(
   enum agx_chip chip, const struct agx_vdm_state_update_draw_info *info,
   struct agx_vdm_state_update *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info ||
       info->source_layout !=
          AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT ||
       info->raw_provoking_selector > 2)
      return false;

   switch (info->draw_path) {
   case AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED:
   case AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED:
   case AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED:
   case AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED:
      break;
   default:
      return false;
   }

   struct agx_vdm_state_update_template_info selected = info->template_info;
   selected.variant =
      (enum agx_vdm_state_update_template_variant)(
         (info->rasterization_enabled ? 1u : 0u) |
         ((uint32_t)info->raw_provoking_selector << 1));

   return agx_vdm_state_update_template_pack(chip, &selected, out);
}

bool
agx_g17p_vertex_program_info(
   enum agx_chip chip,
   const struct agx_g17p_vertex_compile_info *compiler_info,
   struct agx_vdm_state_update_program_info *program)
{
   if (program)
      memset(program, 0, sizeof(*program));

   const bool writes_psiz =
      compiler_info &&
      compiler_info->layer_viewport_off == compiler_info->psiz_off + 1;
   const bool writes_layer_viewport =
      compiler_info &&
      compiler_info->clip_off == compiler_info->layer_viewport_off + 1;
   /* Derive this from the final UVS layout. Vulkan I/O lowering clears
    * outputs_written before backend compilation, so a separate semantic bit
    * can be stale even when the emitted position-only layout is complete. */
   const bool position_only =
      compiler_info && compiler_info->user_size == 0 && !writes_psiz &&
      !writes_layer_viewport;

   /* A constant program needs its register allocation, a known staging
    * footprint and a caller that built it as a native CProg; without a
    * preamble there must be no preamble resources at all. */
   const bool preamble_ok =
      compiler_info &&
      (compiler_info->has_preamble
          ? agx_chip_native_or_m5(chip) &&
               compiler_info->constant_program_allowed &&
               compiler_info->preamble_staging_known &&
               compiler_info->nr_preamble_gprs != 0 &&
               compiler_info->nr_preamble_gprs <= 256
          : compiler_info->nr_preamble_gprs == 0 &&
               compiler_info->preamble_scratch_size == 0);

   if (!agx_chip_agx3_path(chip) ||
       !compiler_info || !program ||
       !compiler_info->hardware_vertex || !preamble_ok ||
       compiler_info->position_off != 0 || compiler_info->varyings_off != 4 ||
       compiler_info->psiz_off != 4 + compiler_info->user_size ||
       (compiler_info->layer_viewport_off != compiler_info->psiz_off &&
        compiler_info->layer_viewport_off != compiler_info->psiz_off + 1) ||
       (compiler_info->clip_off != compiler_info->layer_viewport_off &&
        compiler_info->clip_off != compiler_info->layer_viewport_off + 1) ||
       compiler_info->uvs_size < compiler_info->clip_off ||
       compiler_info->uvs_size - compiler_info->clip_off > 8 ||
       (position_only && compiler_info->uvs_written != 0) ||
       compiler_info->user_size > 125u ||
       /* Besides the 128 API components, unlinked G17P shaders reserve a
        * default Layer varying and PointSize, eight fixed clip distances,
        * and one packed layer/viewport word. UVSW and the native state counts
        * both have eight-bit indices; the fragment coefficient bound is
        * checked separately when linking.
        */
       compiler_info->uvs_size > 139u ||
       __builtin_popcountll(compiler_info->uvs_written) > compiler_info->user_size)
      return false;

   struct agx_vdm_state_update_program_info staged = {
      .source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .template_info =
         {
            /* The no-preamble Ioto loader finishes with both state counts zero.
             * A constant program replaces them below with its TI resources.
             */
            .raw_state_load_count_12 = 0,
            .raw_state_load_count_18 = 0,
            /* VsStateConfig+0xc is the total f32 UVS footprint, including
             * position AND user outputs; +0x14 stays zero in this all-f32
             * path. Apple vs0/vs2/vs4 __GPU_LD_MD root.field1.field3 gives
             * 4/6/8, with field4 absent. The native position+point-size
             * vertex shader likewise gives field3=5 with field4 absent. The former 4/user split described
             * the wrong vertex layout even though its middle sum matched. */
            .raw_vs_output_count_0 = compiler_info->uvs_size,
            .raw_vs_output_count_2 = 0,
            .raw_vs_output_count_1_extra = false,
            .raw_vs_output_selector = 0,
            .raw_vs_output_flag_6 = false,
            .pre_hal300_flag_group_nonzero =
               !agx_chip_hal300_path(chip),
            .pre_hal300_flag_bit_4 = false,
            .variant = AGX_VDM_STATE_UPDATE_TEMPLATE_00,
            .raw_active_context_selector = UINT32_MAX,
         },
   };

   /* Native CProg inputs live at the start of the preamble's TPS frame, so
    * the preamble scratch size is the thread-invariant spill-buffer size
    * (Apple's terrain VS: eight imported words, 32 bytes). */
   if (compiler_info->has_preamble &&
       !agx_vdm_state_update_vertex_ti_resources(
          chip, compiler_info->preamble_scratch_size,
          (uint32_t)compiler_info->preamble_staging_halfs * 2u,
          &staged.template_info))
      return false;

   *program = staged;
   return true;
}
