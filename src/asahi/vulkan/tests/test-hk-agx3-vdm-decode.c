/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_vdm_decode.h"
#include "hk_ppp_state.h"

#include "asahi/lib/agx_vdm_barrier_core.h"
#include "asahi/lib/agx_vdm_ppp_state_core.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_vdm_stream_terminate_core.h"
#include "compiler/shader_enums.h"
#include "util/blob.h"
#include "util/macros.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(condition)                                                       \
   do {                                                                        \
      ++checks;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #condition);                                                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const enum agx_chip targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static bool
record_is_cleared(const struct agx_vdm_decoded_record *record)
{
   const uint8_t *bytes = (const uint8_t *)record;
   for (size_t i = 0; i < sizeof(*record); ++i) {
      if (bytes[i] != 0)
         return false;
   }
   return true;
}

static bool
program_info_equal(const struct agx_vdm_state_update_program_info *a,
                   const struct agx_vdm_state_update_program_info *b)
{
   const struct agx_vdm_state_update_template_info *x = &a->template_info;
   const struct agx_vdm_state_update_template_info *y = &b->template_info;
   return a->source_layout == b->source_layout && x->pipeline == y->pipeline &&
          x->raw_state_load_count_12 == y->raw_state_load_count_12 &&
          x->raw_state_load_count_18 == y->raw_state_load_count_18 &&
          x->raw_vs_output_count_0 == y->raw_vs_output_count_0 &&
          x->raw_vs_output_count_2 == y->raw_vs_output_count_2 &&
          x->raw_vs_output_count_1_extra ==
             y->raw_vs_output_count_1_extra &&
          x->raw_vs_output_selector == y->raw_vs_output_selector &&
          x->raw_vs_output_flag_6 == y->raw_vs_output_flag_6 &&
          x->pre_hal300_flag_group_nonzero ==
             y->pre_hal300_flag_group_nonzero &&
          x->pre_hal300_flag_bit_4 == y->pre_hal300_flag_bit_4 &&
          x->variant == y->variant &&
          x->raw_active_context_selector == y->raw_active_context_selector;
}

static bool
program_info_is_cleared(
   const struct agx_vdm_state_update_program_info *program)
{
   const uint8_t *bytes = (const uint8_t *)program;
   for (size_t i = 0; i < sizeof(*program); ++i) {
      if (bytes[i] != 0)
         return false;
   }
   return true;
}

static int
test_program_cache_lifetime(void)
{
   const struct agx_vdm_state_update_program_info expected = {
      .source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .template_info = {
         .pipeline = UINT64_C(0xac12345680),
         .raw_state_load_count_12 = UINT16_C(0x0abc),
         .raw_state_load_count_18 = UINT32_C(0x00023456),
         .raw_vs_output_count_0 = UINT8_C(0x12),
         .raw_vs_output_count_2 = UINT8_C(0x34),
         .raw_vs_output_count_1_extra = true,
         .raw_vs_output_selector = UINT8_C(0x03),
         .raw_vs_output_flag_6 = true,
         .pre_hal300_flag_group_nonzero = true,
         .pre_hal300_flag_bit_4 = true,
         .variant = AGX_VDM_STATE_UPDATE_TEMPLATE_21,
         .raw_active_context_selector = UINT32_C(0x12345678),
      },
   };

   CHECK(hk_agx3_vdm_live_source_layout(&expected, true, true) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
   CHECK(hk_agx3_vdm_live_source_layout(&expected, false, true) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);
   CHECK(hk_agx3_vdm_live_source_layout(&expected, true, false) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);
   CHECK(hk_agx3_vdm_live_source_layout(NULL, true, true) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);

   struct blob writer;
   blob_init(&writer);
   CHECK(hk_agx3_vdm_program_cache_serialize(&writer, &expected));
   CHECK(writer.size > 0);

   struct blob_reader reader;
   blob_reader_init(&reader, writer.data, writer.size);
   struct agx_vdm_state_update_program_info actual;
   memset(&actual, 0xa5, sizeof(actual));
   CHECK(hk_agx3_vdm_program_cache_deserialize(&reader, &actual));
   CHECK(program_info_equal(&actual, &expected));

   for (size_t size = 0; size < writer.size; ++size) {
      blob_reader_init(&reader, writer.data, size);
      memset(&actual, 0xa5, sizeof(actual));
      CHECK(!hk_agx3_vdm_program_cache_deserialize(&reader, &actual));
      CHECK(program_info_is_cleared(&actual));
   }

   CHECK(!hk_agx3_vdm_program_cache_serialize(NULL, &expected));
   CHECK(!hk_agx3_vdm_program_cache_serialize(&writer, NULL));
   CHECK(!hk_agx3_vdm_program_cache_deserialize(NULL, &actual));
   CHECK(!hk_agx3_vdm_program_cache_deserialize(&reader, NULL));
   blob_finish(&writer);
   return 0;
}

static int
test_g17p_vertex_program_info(void)
{
   const struct hk_agx3_g17p_vertex_compile_info compiler_info = {
      .hardware_vertex = true,
      .position_off = 0,
      .varyings_off = 4,
      .psiz_off = 4,
      .layer_viewport_off = 4,
      .clip_off = 4,
      .uvs_size = 4,
   };
   struct agx_vdm_state_update_program_info program;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &compiler_info,
                                          &program));
   CHECK(program.source_layout ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
   CHECK(program.template_info.raw_state_load_count_12 == 0);
   CHECK(program.template_info.raw_state_load_count_18 == 0);
   CHECK(program.template_info.raw_vs_output_count_0 == 4);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);
   CHECK(program.template_info.pre_hal300_flag_group_nonzero);
   CHECK(!program.template_info.pre_hal300_flag_bit_4);

   program.template_info.pipeline = UINT64_C(0x10000128000);
   const struct agx_vdm_state_update_draw_info draw = {
      .source_layout = program.source_layout,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled = true,
      .raw_provoking_selector = 0,
      .template_info = program.template_info,
   };
   struct agx_vdm_state_update encoded;
   CHECK(hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17P, &draw, &encoded));
   static const uint8_t m1n1_direct_triangle_prefix[] = {
      0x2e, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x00, 0x4a, 0x00, 0x00, 0x04, 0x04, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(encoded.length == sizeof(m1n1_direct_triangle_prefix));
   CHECK(memcmp(encoded.bytes, m1n1_direct_triangle_prefix,
                sizeof(m1n1_direct_triangle_prefix)) == 0);

   struct agx_vdm_state_update_program_info m5;
   CHECK(hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &compiler_info, &m5));
   CHECK(!m5.template_info.pre_hal300_flag_group_nonzero);
   CHECK(!m5.template_info.pre_hal300_flag_bit_4);
   struct agx_vdm_state_update_draw_info m5_draw = draw;
   m5_draw.template_info = m5.template_info;
   m5_draw.template_info.pipeline = UINT64_C(0x10000160000);
   CHECK(hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17G, &m5_draw, &encoded));
   /* Complete M5 native triangle VDM state at command-stream offset0x18. */
   static const uint8_t m5_native_triangle_state[] = {
      0x2e,0x00,0x00,0x40,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x01,0x00,0x58,0x00,0x00,
      0x04,0x04,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,
   };
   CHECK(encoded.length == sizeof(m5_native_triangle_state));
   CHECK(memcmp(encoded.bytes, m5_native_triangle_state, encoded.length) == 0);
   struct hk_agx3_g17p_vertex_compile_info m5_psiz = compiler_info;
   m5_psiz.layer_viewport_off = m5_psiz.clip_off = m5_psiz.uvs_size = 5;
   CHECK(hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_psiz, &m5));
   CHECK(m5.template_info.raw_vs_output_count_0 == 5);
   CHECK(!m5.template_info.pre_hal300_flag_group_nonzero);
   m5_psiz.clip_off = m5_psiz.uvs_size = 6;
   CHECK(!hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_psiz, &m5));
   CHECK(program_info_is_cleared(&m5));
   struct hk_agx3_g17p_vertex_compile_info m5_invalid = compiler_info;
   m5_invalid.has_preamble = true;
   CHECK(!hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_invalid, &m5));
   CHECK(program_info_is_cleared(&m5));
   m5_invalid = compiler_info;
   m5_invalid.user_size = 1;
   CHECK(!hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_invalid, &m5));
   CHECK(program_info_is_cleared(&m5));
   CHECK(!hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17P, &compiler_info, &m5));
   CHECK(program_info_is_cleared(&m5));

   /* Native M5 m5_color_vertex and m5_color_point_vertex metadata:
    * root.field1.field3 = 8/9, field4 absent (all outputs are f32).
    * Keep extra varyings, packed layer/viewport, and preambles rejected. */
   struct hk_agx3_g17p_vertex_compile_info m5_color = compiler_info;
   m5_color.uvs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   m5_color.user_size = 4;
   m5_color.psiz_off = m5_color.layer_viewport_off = m5_color.clip_off =
      m5_color.uvs_size = 8;
   CHECK(hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_color, &m5));
   CHECK(m5.template_info.raw_vs_output_count_0 == 8);
   CHECK(m5.template_info.raw_vs_output_count_2 == 0);
   CHECK(!m5.template_info.pre_hal300_flag_group_nonzero);
   m5_color.layer_viewport_off = m5_color.clip_off = m5_color.uvs_size = 9;
   CHECK(hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_color, &m5));
   CHECK(m5.template_info.raw_vs_output_count_0 == 9);
   CHECK(m5.template_info.raw_vs_output_count_2 == 0);
   CHECK(!m5.template_info.pre_hal300_flag_group_nonzero);
   m5_color.clip_off = m5_color.uvs_size = 10;
   CHECK(!hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_color, &m5));
   CHECK(program_info_is_cleared(&m5));
   m5_color.clip_off = m5_color.uvs_size = 9;
   m5_color.uvs_written |= BITFIELD64_BIT(VARYING_SLOT_VAR1);
   CHECK(!hk_agx3_g17g_vertex_program_info(AGX_CHIP_G17G, &m5_color, &m5));
   CHECK(program_info_is_cleared(&m5));

   /* Apple vs2 float2 varying witness: __GPU_LD_MD root.field1 gives
    * field3=6 and field4 absent, mapping to VsStateConfig+0xc/+0x14.
    * This protects the actual producer semantics, not just byte packing. */
   struct hk_agx3_g17p_vertex_compile_info vs2_info = compiler_info;
   vs2_info.uvs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   vs2_info.psiz_off = vs2_info.layer_viewport_off = vs2_info.clip_off = 6;
   vs2_info.uvs_size = 6;
   vs2_info.user_size = 2;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &vs2_info, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 6);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   /* Fresh native point-size pipeline has five f32 outputs (four position
    * plus point-size) and no secondary-width stream. */
   struct hk_agx3_g17p_vertex_compile_info point_info = compiler_info;
   point_info.layer_viewport_off = point_info.clip_off = point_info.uvs_size = 5;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &point_info, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 5);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);
   point_info.layer_viewport_off = point_info.clip_off = point_info.uvs_size = 6;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &point_info, &program));

   /* The packed layer/viewport sysval occupies one word after ordinary
    * varyings (and optional point size).  Multiview's ViewIndex-to-Layer
    * lowering produces this exact position + packed-layer shape. */
   struct hk_agx3_g17p_vertex_compile_info layer_info = compiler_info;
   layer_info.clip_off = layer_info.uvs_size = 5;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &layer_info,
                                          &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 5);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   /* The XFB clip-distance raster variant appends the combined clip/cull
    * group after ordinary varyings and optional point size. VsStateConfig's
    * field3 remains the total f32 UVS footprint. */
   struct hk_agx3_g17p_vertex_compile_info clip_info = compiler_info;
   clip_info.uvs_written = BITFIELD64_BIT(VARYING_SLOT_CLIP_DIST0) |
                           BITFIELD64_BIT(VARYING_SLOT_CLIP_DIST1);
   clip_info.user_size = 8;
   clip_info.psiz_off = 12;
   clip_info.layer_viewport_off = clip_info.clip_off = 13;
   clip_info.uvs_size = 20;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &clip_info,
                                          &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 20);
   clip_info.uvs_size = 22;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &clip_info,
                                           &program));

   const struct hk_agx3_g17p_vertex_compile_info vkcube_info = {
      .hardware_vertex = true,
      .uvs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0) |
                     BITFIELD64_BIT(VARYING_SLOT_VAR1),
      .position_off = 0,
      .varyings_off = 4,
      .psiz_off = 11,
      .layer_viewport_off = 11,
      .clip_off = 11,
      .uvs_size = 11,
      .user_size = 7,
   };
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &vkcube_info,
                                          &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 11);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);
   CHECK(!program.template_info.raw_vs_output_count_1_extra);

   program.template_info.pipeline = UINT64_C(0x10000128000);
   const struct agx_vdm_state_update_draw_info vkcube_draw = {
      .source_layout = program.source_layout,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED,
      .rasterization_enabled = true,
      .raw_provoking_selector = 0,
      .template_info = program.template_info,
   };
   CHECK(hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17P, &vkcube_draw,
                                               &encoded));
   static const uint8_t vkcube_prefix[] = {
      0x2e, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x00, 0x4a, 0x00, 0x00, 0x0b, 0x0b, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(encoded.length == sizeof(vkcube_prefix));
   CHECK(memcmp(encoded.bytes, vkcube_prefix, sizeof(vkcube_prefix)) == 0);

   /* Three scalar/vector slots still occupy the same seven f32 user words. */
   struct hk_agx3_g17p_vertex_compile_info quake_info = vkcube_info;
   quake_info.uvs_written |= BITFIELD64_BIT(VARYING_SLOT_VAR2);
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &quake_info, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 11);

   struct hk_agx3_g17p_vertex_compile_info native32 = vkcube_info;
   native32.user_size = 28;
   native32.psiz_off = native32.layer_viewport_off = native32.clip_off = 32;
   native32.uvs_size = 32;
   native32.uvs_written = BITFIELD64_MASK(7) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &native32, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 32);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   /* Fresh Apple G17P vs36/fs36 pipeline: position plus eight float4 user
    * locations.  __GPU_LD_MD reports field0=36 and the shader writes all 36
    * f32 UVS words.  GTK's wide vertex variants use 30 and 32 user words, so
    * the earlier seven-location bound rejected valid desktop pipelines. */
   struct hk_agx3_g17p_vertex_compile_info native36 = vkcube_info;
   native36.user_size = 32;
   native36.psiz_off = native36.layer_viewport_off = native36.clip_off = 36;
   native36.uvs_size = 36;
   native36.uvs_written = BITFIELD64_MASK(8) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &native36, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 36);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   /* Fresh MetalToolchain v27.1.5194.15 G17P vs52/fs52 pipeline, AIR SHA256
    * d9967d614233... / native SHA256 ebadfd1cefb6.... The VS exports position
    * plus twelve float4 user locations and its 1304-byte program contains the
    * complete dense 52-scalar UVSW protocol. Firefox WebRender uses the same
    * state shape with 47 user words (51 total outputs). */
   struct hk_agx3_g17p_vertex_compile_info native52 = vkcube_info;
   native52.user_size = 48;
   native52.psiz_off = native52.layer_viewport_off = native52.clip_off = 52;
   native52.uvs_size = 52;
   native52.uvs_written = BITFIELD64_MASK(12) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &native52, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 52);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   struct hk_agx3_g17p_vertex_compile_info firefox51 = vkcube_info;
   firefox51.user_size = 47;
   firefox51.psiz_off = firefox51.layer_viewport_off = firefox51.clip_off = 51;
   firefox51.uvs_size = 51;
   firefox51.uvs_written = BITFIELD64_MASK(12) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &firefox51, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 51);

   /* The Vulkan minimum is 64 user components. Apple's vs68/fs68 pipeline
    * (native archive SHA256 6e71a71797784d58266ffbb0cd89181a8054a9e2b0d6db34770e36689be1bcf2;
    * VS text SHA256 70980b666b28386a9288b9ac31ff6b4a4f6257e16af0911912e653e538519620)
    * exports position plus sixteen float4 user locations as a 1506-byte
    * native vertex program. Retain exactly that witnessed public boundary. */
   struct hk_agx3_g17p_vertex_compile_info native68 = vkcube_info;
   native68.user_size = 64;
   native68.psiz_off = native68.layer_viewport_off = native68.clip_off = 68;
   native68.uvs_size = 68;
   native68.uvs_written = BITFIELD64_MASK(16) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &native68, &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 68);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   /* Chromium 151's ANGLE Vulkan backend reserves one additional float4 for
    * transform-feedback capture of gl_Position: position plus seventeen
    * float4 user locations, 72 UVS words. */
   struct hk_agx3_g17p_vertex_compile_info angle72 = vkcube_info;
   angle72.user_size = 68;
   angle72.psiz_off = angle72.layer_viewport_off = angle72.clip_off = 72;
   angle72.uvs_size = 72;
   angle72.uvs_written = BITFIELD64_MASK(17) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &angle72,
                                          &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 72);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   /* Apple's widest G17P vertex program: position plus thirty-one float4
    * user locations, UVSW indices 0..127, __GPU_LD_MD root.field1.field3
    * 128. */
   struct hk_agx3_g17p_vertex_compile_info native128 = vkcube_info;
   native128.user_size = 124;
   native128.psiz_off = native128.layer_viewport_off = native128.clip_off =
      128;
   native128.uvs_size = 128;
   native128.uvs_written = BITFIELD64_MASK(31) << VARYING_SLOT_VAR0;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &native128,
                                          &program));
   CHECK(program.template_info.raw_vs_output_count_0 == 128);
   CHECK(program.template_info.raw_vs_output_count_2 == 0);

   struct hk_agx3_g17p_vertex_compile_info invalid_info = vkcube_info;
   invalid_info.has_preamble = true;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &invalid_info,
                                           &program));
   CHECK(program.source_layout ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);

   /* Vertex constant program. Apple's G17P terrain VS (MetalToolchain 27.1,
    * terrain-isa-diff-20260926/apple-metadata.json) imports eight CProg
    * uniform words: thread_invariant_spill_buffer_bytes 32, no TI IPR, 36
    * CProg temporaries. The vertex program state puts the TI spill
    * quads (8) in pipeline-template bits 3..20 and the TI IPR quads in
    * vertex-shader word 0 bits 8..19. */
   struct hk_agx3_g17p_vertex_compile_info cprog_info = vkcube_info;
   cprog_info.has_preamble = true;
   cprog_info.nr_preamble_gprs = 36;
   cprog_info.preamble_scratch_size = 32;
   cprog_info.preamble_staging_known = true;
   memset(&program, 0xa5, sizeof(program));
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &cprog_info,
                                           &program));
   CHECK(program.source_layout ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);
   cprog_info.constant_program_allowed = true;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &cprog_info,
                                          &program));
   CHECK(program.source_layout ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
   CHECK(program.template_info.raw_state_load_count_12 == 0);
   CHECK(program.template_info.raw_state_load_count_18 == 8);
   CHECK(program.template_info.raw_vs_output_count_0 == 11);

   program.template_info.pipeline = UINT64_C(0x10000128000);
   struct agx_vdm_state_update_draw_info cprog_draw = {
      .source_layout = program.source_layout,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED,
      .rasterization_enabled = true,
      .raw_provoking_selector = 0,
      .template_info = program.template_info,
   };
   CHECK(hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17P, &cprog_draw,
                                               &encoded));
   /* vkcube_prefix with 0x40 (8 << 3) in the pipeline-pair template. */
   static const uint8_t cprog_prefix[] = {
      0x2e, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
      0x00, 0x01, 0x00, 0x4a, 0x00, 0x00, 0x0b, 0x0b, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(encoded.length == sizeof(cprog_prefix));
   CHECK(memcmp(encoded.bytes, cprog_prefix, sizeof(cprog_prefix)) == 0);
   struct agx_vdm_decoded_record cprog_record;
   CHECK(hk_agx3_decode_vdm_record(AGX_CHIP_G17P, encoded.bytes,
                                   encoded.length, &cprog_record));
   CHECK(cprog_record.kind == AGX_VDM_RECORD_STATE_UPDATE);
   CHECK(cprog_record.data.state_update.pipeline == UINT64_C(0x10000128000));
   CHECK(cprog_record.data.state_update.raw_pipeline_template == 0x40);

   /* A 40-byte staging footprint: ((40 + 3) >> 2) + 1, even-rounded = 10
    * quads in vertex-shader word 0 bits 8..19. */
   cprog_info.preamble_staging_halfs = 20;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &cprog_info,
                                          &program));
   CHECK(program.template_info.raw_state_load_count_12 == 10);
   CHECK(program.template_info.raw_state_load_count_18 == 8);
   program.template_info.pipeline = UINT64_C(0x10000128000);
   cprog_draw.template_info = program.template_info;
   CHECK(hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17P, &cprog_draw,
                                               &encoded));
   CHECK(encoded.bytes[4] == 0x00 && encoded.bytes[5] == 0x0a &&
         encoded.bytes[6] == 0x00 && encoded.bytes[7] == 0x00);
   CHECK(encoded.bytes[8] == 0x40);

   /* Fail closed: unknown staging, no or too many CProg registers, and a TI
    * spill beyond the 18-bit quad field. */
   struct hk_agx3_g17p_vertex_compile_info bad_cprog = cprog_info;
   bad_cprog.preamble_staging_known = false;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                           &program));
   bad_cprog = cprog_info;
   bad_cprog.nr_preamble_gprs = 0;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                           &program));
   bad_cprog.nr_preamble_gprs = 257;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                           &program));
   bad_cprog = cprog_info;
   bad_cprog.preamble_scratch_size = UINT32_C(0x3ffff) * 4 + 1;
   memset(&program, 0xa5, sizeof(program));
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                           &program));
   CHECK(program.source_layout ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);
   bad_cprog.preamble_scratch_size = UINT32_C(0x3ffff) * 4;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                          &program));
   CHECK(program.template_info.raw_state_load_count_18 == UINT32_C(0x3ffff));
   /* Allowed but absent: exactly the no-preamble state. */
   bad_cprog = vkcube_info;
   bad_cprog.constant_program_allowed = true;
   CHECK(hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                          &program));
   CHECK(program.template_info.raw_state_load_count_12 == 0);
   CHECK(program.template_info.raw_state_load_count_18 == 0);
   bad_cprog.preamble_scratch_size = 4;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &bad_cprog,
                                           &program));
   invalid_info = vkcube_info;
   invalid_info.user_size = HK_AGX3_G17P_MAX_USER_WORDS + 1;
   invalid_info.psiz_off = invalid_info.layer_viewport_off =
      invalid_info.clip_off = invalid_info.uvs_size = invalid_info.user_size + 4;
   CHECK(!hk_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &invalid_info,
                                           &program));
   return 0;
}

static int
test_honeykrisp_bridge(void)
{
   for (unsigned t = 0; t < sizeof(targets) / sizeof(targets[0]); ++t) {
      const enum agx_chip chip = targets[t];
      struct agx_vdm_decoded_record record;

      struct agx_vdm_ppp_state_core ppp;
      CHECK(agx_vdm_ppp_state_core_pack(chip, UINT64_C(0xee12345000) + t * 4,
                                        0x20 + t, &ppp));
      CHECK(hk_agx3_decode_vdm_record(chip, ppp.bytes, ppp.length, &record));
      CHECK(record.kind == AGX_VDM_RECORD_PPP_STATE_UPDATE);
      CHECK(record.length == ppp.length);
      CHECK(record.data.ppp_state_update.pointer ==
            UINT64_C(0xee12345000) + t * 4);

      struct agx_vdm_barrier_core barrier;
      CHECK(agx_vdm_barrier_core_pack(chip, false, &barrier));
      uint8_t barrier_stream[AGX_VDM_BARRIER_CORE_LENGTH + 4];
      memcpy(barrier_stream, barrier.bytes, barrier.length);
      memset(barrier_stream + barrier.length, 0xa5, 4);
      CHECK(hk_agx3_decode_vdm_record(chip, barrier_stream,
                                      sizeof(barrier_stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_BARRIER);
      CHECK(record.length == barrier.length);
      CHECK(record.chip == chip);

      const struct agx_vdm_state_update_info state_info = {
         .raw_vertex_shader_word0 = UINT32_C(0x12345678) + t,
         .pipeline = UINT64_C(0xac12345680) + t * 64,
         .raw_pipeline_template = UINT32_C(0x00134567) ^ t,
         .raw_vertex_outputs = {UINT32_C(0x23456789) + t,
                                UINT32_C(0x3456789a) - t},
         .raw_vertex_unknown = UINT32_C(0x456789ab) ^ t,
      };
      struct agx_vdm_state_update state;
      CHECK(agx_vdm_state_update_pack(chip, &state_info, &state));
      CHECK(
         hk_agx3_decode_vdm_record(chip, state.bytes, state.length, &record));
      CHECK(record.kind == AGX_VDM_RECORD_STATE_UPDATE);
      CHECK(record.length == state.length);
      CHECK(record.data.state_update.pipeline == state_info.pipeline);

      struct agx_vdm_state_update_template_info template_info = {
         .pipeline = UINT64_C(0xac12345680) + t * 64,
         .raw_state_load_count_12 = UINT16_C(0x0abc),
         .raw_state_load_count_18 = UINT32_C(0x00023456),
         .raw_vs_output_count_0 = UINT8_C(0x12),
         .raw_vs_output_count_2 = UINT8_C(0x34),
         .raw_vs_output_count_1_extra = true,
         .raw_vs_output_selector = UINT8_C(0x03),
         .raw_vs_output_flag_6 = true,
         .pre_hal300_flag_group_nonzero = true,
         .pre_hal300_flag_bit_4 = true,
         .variant = AGX_VDM_STATE_UPDATE_TEMPLATE_10,
         .raw_active_context_selector =
            (t & 1) ? UINT32_MAX : UINT32_C(8),
      };
      const bool hal300 = chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
                          chip == AGX_CHIP_G17C;
      if (hal300) {
         template_info.pre_hal300_flag_group_nonzero = false;
         template_info.pre_hal300_flag_bit_4 = false;
      }

      struct agx_vdm_state_update template_state;
      CHECK(hk_agx3_encode_vdm_state_update_template(
         chip, &template_info, &template_state));
      CHECK(hk_agx3_decode_vdm_record(chip, template_state.bytes,
                                      template_state.length, &record));
      CHECK(record.kind == AGX_VDM_RECORD_STATE_UPDATE);
      CHECK(record.data.state_update.raw_vertex_shader_word0 ==
            (hal300 ? UINT32_C(0x000abc00) : UINT32_C(0x800abc00)));
      CHECK(record.data.state_update.raw_pipeline_template ==
            UINT32_C(0x0011a2b0));
      CHECK(record.data.state_update.raw_vertex_outputs[0] ==
            UINT32_C(0x00344712));
      CHECK(record.data.state_update.raw_vertex_outputs[1] ==
            (hal300 ? UINT32_C(0x0000006d) : UINT32_C(0x0000007d)));
      CHECK(record.data.state_update.raw_vertex_unknown == ((t & 1) ? 0 : 7));

      for (unsigned selector = 0; selector < 6; ++selector) {
         for (unsigned draw = 0; draw < 4; ++draw) {
            struct agx_vdm_state_update_draw_info draw_info = {
               .source_layout =
                  AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
               .draw_path = (enum agx_vdm_state_update_draw_path)draw,
               .rasterization_enabled = (selector & 1) != 0,
               .raw_provoking_selector = selector >> 1,
               .template_info = template_info,
            };
            struct agx_vdm_state_update hk_draw;
            struct agx_vdm_state_update shared_draw;

            /* Prove the wrapper uses the renderer selector instead of a
             * caller-supplied legacy/template enum. */
            draw_info.template_info.variant =
               (enum agx_vdm_state_update_template_variant)6;
            CHECK(hk_agx3_encode_vdm_state_update_draw(chip, &draw_info,
                                                        &hk_draw));
            CHECK(agx_vdm_state_update_draw_pack(chip, &draw_info,
                                                  &shared_draw));
            CHECK(hk_draw.length == shared_draw.length);
            CHECK(memcmp(hk_draw.bytes, shared_draw.bytes,
                         AGX_VDM_STATE_UPDATE_LENGTH) == 0);
         }
      }

      const struct agx_vdm_index_list_direct_nonindexed_info nonindexed_info = {
         .primitive = 11,
         .count = UINT32_C(0x56789abc) + t,
         .instances = 2 + t,
         .start = UINT32_C(0x6789abcd) ^ t,
      };
      struct agx_vdm_index_list_direct_nonindexed nonindexed;
      CHECK(agx_vdm_index_list_direct_nonindexed_pack(chip, &nonindexed_info,
                                                      &nonindexed));
      CHECK(hk_agx3_decode_vdm_record(chip, nonindexed.bytes, nonindexed.length,
                                      &record));
      CHECK(record.kind == AGX_VDM_RECORD_INDEX_LIST_NONINDEXED);
      CHECK(record.length == nonindexed.length);
      CHECK(record.data.index_list_nonindexed.count == nonindexed_info.count);

      const struct agx_vdm_index_list_direct_indexed_info indexed_info = {
         .primitive = 14,
         .restart_enable = true,
         .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U16,
         .index_buffer = UINT64_C(0xac1234567890) + t * 4,
         .count = UINT32_C(0x789abcde) + t,
         .instances = 3 + t,
         .start = UINT32_C(0x89abcdef) ^ t,
         .index_buffer_size_word = UINT32_C(0x9abcdef0) + t,
      };
      struct agx_vdm_index_list_direct_indexed indexed;
      CHECK(
         agx_vdm_index_list_direct_indexed_pack(chip, &indexed_info, &indexed));
      CHECK(hk_agx3_decode_vdm_record(chip, indexed.bytes, indexed.length,
                                      &record));
      CHECK(record.kind == AGX_VDM_RECORD_INDEX_LIST_INDEXED);
      CHECK(record.length == indexed.length);
      CHECK(record.data.index_list_indexed.index_buffer ==
            indexed_info.index_buffer);

      struct agx_vdm_stream_link_core link;
      CHECK(agx_vdm_stream_link_core_pack(chip, UINT64_C(0xce12345678) + t,
                                          &link));
      uint8_t stream[AGX_VDM_STREAM_LINK_CORE_LENGTH + 4];
      memcpy(stream, link.bytes, link.length);
      memset(stream + link.length, 0xa5, 4);
      CHECK(hk_agx3_decode_vdm_record(chip, stream, sizeof(stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_STREAM_LINK);
      CHECK(record.length == link.length);
      CHECK(record.chip == chip);
      CHECK(record.data.stream_link.target == UINT64_C(0xce12345678) + t);

      const struct agx_vdm_tessellate_direct_info tessellate_info = {
         .factor_buffer = UINT64_C(0xac1234567800) + t,
         .patch_count = UINT32_C(0x10203040) + t,
         .instance_count = UINT32_C(0x50607080) - t,
         .base_patch = UINT32_C(0x90a0b0c0) ^ t,
         .base_instance = UINT32_C(0xd0e0f001) + t,
         .instance_stride_word = UINT32_C(0x70000) + t,
         .factor_buffer_size_minus_one = UINT32_C(0xfedcba98) - t,
      };
      struct agx_vdm_tessellate_direct tessellate;
      CHECK(agx_vdm_tessellate_direct_pack(chip, &tessellate_info,
                                            &tessellate));
      uint8_t tessellate_stream[AGX_VDM_TESSELLATE_DIRECT_LENGTH + 4];
      memcpy(tessellate_stream, tessellate.bytes, tessellate.length);
      memset(tessellate_stream + tessellate.length, 0xa5, 4);
      CHECK(hk_agx3_decode_vdm_record(chip, tessellate_stream,
                                      sizeof(tessellate_stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_TESSELLATE_DIRECT);
      CHECK(record.length == tessellate.length);
      CHECK(record.chip == chip);
      CHECK(record.data.tessellate_direct.factor_buffer ==
            tessellate_info.factor_buffer);
      CHECK(record.data.tessellate_direct.patch_count ==
            tessellate_info.patch_count);
      CHECK(record.data.tessellate_direct.instance_count ==
            tessellate_info.instance_count);

      struct agx_vdm_stream_terminate_core terminate;
      CHECK(agx_vdm_stream_terminate_core_pack(chip, &terminate));
      uint8_t terminate_stream[AGX_VDM_STREAM_TERMINATE_CORE_LENGTH + 4];
      memcpy(terminate_stream, terminate.bytes, terminate.length);
      memset(terminate_stream + terminate.length, 0xa5, 4);
      CHECK(hk_agx3_decode_vdm_record(chip, terminate_stream,
                                      sizeof(terminate_stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_STREAM_TERMINATE);
      CHECK(record.length == terminate.length);
      CHECK(record.chip == chip);
   }

   return 0;
}

static int
test_rejection(void)
{
   struct agx_vdm_ppp_state_core ppp;
   CHECK(agx_vdm_ppp_state_core_pack(AGX_CHIP_G17S, UINT64_C(0x123456780), 8,
                                     &ppp));

   {
      struct agx_vdm_state_update_draw_info draw_info = {
         .source_layout = AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX2_LEGACY,
         .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
         .template_info = {
            .pipeline = UINT64_C(0x123456780),
         },
      };
      struct agx_vdm_state_update encoded;
      memset(&encoded, 0xa5, sizeof(encoded));
      CHECK(!hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, &draw_info,
                                                   &encoded));
      for (unsigned i = 0; i < sizeof(encoded.bytes); ++i)
         CHECK(encoded.bytes[i] == 0);
      CHECK(encoded.length == 0);
      CHECK(!hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, NULL,
                                                   &encoded));
      CHECK(!hk_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, &draw_info,
                                                   NULL));
   }

   for (size_t available = 0; available < ppp.length; ++available) {
      struct agx_vdm_decoded_record record;
      memset(&record, 0xff, sizeof(record));
      CHECK(!hk_agx3_decode_vdm_record(AGX_CHIP_G17S, ppp.bytes, available,
                                       &record));
      CHECK(record_is_cleared(&record));
   }

   for (enum agx_chip chip = AGX_CHIP_G13G; chip <= AGX_CHIP_G17C; ++chip) {
      bool supported = false;
      for (unsigned t = 0; t < sizeof(targets) / sizeof(targets[0]); ++t)
         supported |= chip == targets[t];
      if (supported)
         continue;

      struct agx_vdm_decoded_record record;
      memset(&record, 0xff, sizeof(record));
      CHECK(!hk_agx3_decode_vdm_record(chip, ppp.bytes, ppp.length, &record));
      CHECK(record_is_cleared(&record));
   }

   const uint8_t ungrounded_return[4] = {0, 0, 0, 0xe0};
   struct agx_vdm_decoded_record record;
   for (unsigned t = 0; t < sizeof(targets) / sizeof(targets[0]); ++t) {
      memset(&record, 0xff, sizeof(record));
      CHECK(!hk_agx3_decode_vdm_record(targets[t], ungrounded_return,
                                       sizeof(ungrounded_return), &record));
      CHECK(record_is_cleared(&record));
   }

   const struct agx_vdm_tessellate_direct_info tessellate_info = {
      .factor_buffer = UINT64_C(0xab1234567890),
      .patch_count = UINT32_C(0x10203040),
      .instance_count = UINT32_C(0x50607080),
      .base_patch = UINT32_C(0x90a0b0c0),
      .base_instance = UINT32_C(0xd0e0f001),
      .instance_stride_word = UINT32_C(0x71234),
      .factor_buffer_size_minus_one = UINT32_C(0xfedcba98),
   };
   struct agx_vdm_tessellate_direct tessellate;
   CHECK(agx_vdm_tessellate_direct_pack(AGX_CHIP_G15G, &tessellate_info,
                                         &tessellate));

   for (size_t available = 0; available < tessellate.length; ++available) {
      memset(&record, 0xff, sizeof(record));
      CHECK(!hk_agx3_decode_vdm_record(AGX_CHIP_G17S, tessellate.bytes,
                                       available, &record));
      CHECK(record_is_cleared(&record));
   }

   for (enum agx_chip chip = AGX_CHIP_G13G; chip <= AGX_CHIP_G17C; ++chip) {
      bool supported = false;
      for (unsigned t = 0; t < sizeof(targets) / sizeof(targets[0]); ++t)
         supported |= chip == targets[t];
      if (supported)
         continue;

      memset(&record, 0xff, sizeof(record));
      CHECK(!hk_agx3_decode_vdm_record(chip, tessellate.bytes,
                                       tessellate.length, &record));
      CHECK(record_is_cleared(&record));
   }

   /* All three pinned families use the same bytes; cross-family decoding is
    * valid but cannot classify the originating generation. */
   CHECK(hk_agx3_decode_vdm_record(AGX_CHIP_G17S, tessellate.bytes,
                                   tessellate.length, &record));
   CHECK(record.kind == AGX_VDM_RECORD_TESSELLATE_DIRECT);
   CHECK(record.data.tessellate_direct.factor_buffer ==
         tessellate_info.factor_buffer);

   for (unsigned bit = 0; bit < tessellate.length * 8u; ++bit) {
      uint8_t changed[AGX_VDM_TESSELLATE_DIRECT_LENGTH];
      struct agx_vdm_tessellate_direct_info shared;
      memcpy(changed, tessellate.bytes, tessellate.length);
      changed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
      const bool shared_accepts = agx_vdm_tessellate_direct_unpack(
         AGX_CHIP_G17S, changed, sizeof(changed), &shared);
      memset(&record, 0xff, sizeof(record));
      CHECK(hk_agx3_decode_vdm_record(AGX_CHIP_G17S, changed, sizeof(changed),
                                      &record) == shared_accepts);
      if (shared_accepts) {
         CHECK(record.kind == AGX_VDM_RECORD_TESSELLATE_DIRECT);
         CHECK(memcmp(&record.data.tessellate_direct, &shared,
                      sizeof(shared)) == 0);
      } else {
         CHECK(record_is_cleared(&record));
      }
   }

   const uint8_t aggregate_multi_agc[AGX_VDM_BARRIER_CORE_LENGTH] = {
      0x7c,
      0x09,
      0x00,
      0x21,
   };
   for (unsigned t = 0; t < sizeof(targets) / sizeof(targets[0]); ++t) {
      memset(&record, 0xff, sizeof(record));
      CHECK(!hk_agx3_decode_vdm_record(targets[t], aggregate_multi_agc,
                                       sizeof(aggregate_multi_agc), &record));
      CHECK(record_is_cleared(&record));
   }
   CHECK(
      !hk_agx3_decode_vdm_record(AGX_CHIP_G17S, ppp.bytes, ppp.length, NULL));
   return 0;
}

int
main(void)
{
   /* Native M5 first PPP: full mapped VA is 0x1000058000, while the VDM
    * record contains the task-relative 0x58000. Preserve all output on
    * rejected roots rather than wrapping into the other UAT root. */
   const uint8_t m5_ppp[] = {0,6,0,0, 0,0x80,5,0};
   uint8_t ppp[8];
   uint8_t *cursor = ppp;
   CHECK(hk_ppp_emit_state(AGX_CHIP_G17G, &cursor,
                           UINT64_C(0x1000058000), 24));
   CHECK(cursor == ppp + 8 && memcmp(ppp, m5_ppp, 8) == 0);
   cursor = ppp;
   CHECK(!hk_ppp_emit_state(AGX_CHIP_G17G, &cursor, 0x58000, 24));
   CHECK(!hk_ppp_emit_state(AGX_CHIP_G17G, &cursor,
                            UINT64_C(0x9000058000), 24));
   CHECK(cursor == ppp && memcmp(ppp, m5_ppp, 8) == 0);
   CHECK(hk_ppp_emit_state(AGX_CHIP_G17G, &cursor,
                           UINT64_C(0x5ffff30c00), 24));
   const uint8_t relocated_ppp[] = {0x4f,6,0,0, 0,0x0c,0xf3,0xff};
   CHECK(memcmp(ppp, relocated_ppp, 8) == 0);
   const struct {
      uint64_t address;
      size_t size;
   } g18_updates[] = {
      {UINT64_C(0x5ffe784ca0), 40},
      {UINT64_C(0x5ffe784c8c), 20},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(g18_updates); ++i) {
      cursor = ppp;
      uint64_t wire;
      uint32_t words;
      CHECK(hk_ppp_emit_state(AGX_CHIP_G18G, &cursor,
                              g18_updates[i].address, g18_updates[i].size));
      CHECK(agx_vdm_ppp_state_core_unpack(AGX_CHIP_G18G, ppp, sizeof(ppp),
                                          &wire, &words));
      CHECK(wire + AGX_M5_DRIVER_BASE == g18_updates[i].address);
      CHECK(words * 4 == g18_updates[i].size);
   }
   CHECK(test_program_cache_lifetime() == 0);
   CHECK(test_g17p_vertex_program_info() == 0);
   CHECK(test_honeykrisp_bridge() == 0);
   CHECK(test_rejection() == 0);

   printf("Honeykrisp AGX3 VDM decode: %u checks passed\n", checks);
   return 0;
}
