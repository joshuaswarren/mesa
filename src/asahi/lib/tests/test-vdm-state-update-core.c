/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_vdm_state_update_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const enum agx_chip supported_chips[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D,
};

static const enum agx_chip unsupported_chips[] = {
   AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
   AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   (enum agx_chip)999,
};

static const struct agx_vdm_state_update_info witness_info = {
   .raw_vertex_shader_word0 = UINT32_C(0x11223344),
   .pipeline = UINT64_C(0x123456789a80),
   .raw_pipeline_template = UINT32_C(0x2abcde),
   .raw_vertex_outputs = {UINT32_C(0x55667788), UINT32_C(0x99aabbcc)},
   .raw_vertex_unknown = UINT32_C(0xddeeff00),
};

static const uint8_t witness[AGX_VDM_STATE_UPDATE_LENGTH] = {
   0x2e, 0x00, 0x00, 0x40, 0x44, 0x33, 0x22, 0x11,
   0xde, 0xbc, 0x2a, 0x12, 0x6a, 0xe2, 0x59, 0xd1,
   0x88, 0x77, 0x66, 0x55, 0xcc, 0xbb, 0xaa, 0x99,
   0x00, 0xff, 0xee, 0xdd,
};

static const struct agx_vdm_state_update_template_info template_witness_info = {
   .pipeline = UINT64_C(0x123456789a80),
   .raw_state_load_count_12 = UINT16_C(0x0abc),
   .raw_state_load_count_18 = UINT32_C(0x00023456),
   .raw_vs_output_count_0 = UINT8_C(0x12),
   .raw_vs_output_count_2 = UINT8_C(0x34),
   .raw_vs_output_count_1_extra = true,
   .raw_vs_output_selector = UINT8_C(0x03),
   .raw_vs_output_flag_6 = true,
   .pre_hal300_flag_group_nonzero = true,
   .pre_hal300_flag_bit_4 = true,
   .variant = AGX_VDM_STATE_UPDATE_TEMPLATE_00,
   .raw_active_context_selector = 5,
};

static const uint8_t pre_hal300_template_witness
   [AGX_VDM_STATE_UPDATE_LENGTH] = {
      0x2e, 0x00, 0x00, 0x40, 0x00, 0xbc, 0x0a, 0x80,
      0xb0, 0xa2, 0x11, 0x12, 0x6a, 0xe2, 0x59, 0xd1,
      0x12, 0x47, 0x34, 0x00, 0x7c, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00,
   };

static const uint8_t hal300_template_witness[AGX_VDM_STATE_UPDATE_LENGTH] = {
   0x2e, 0x00, 0x00, 0x40, 0x00, 0xbc, 0x0a, 0x00,
   0xb0, 0xa2, 0x11, 0x12, 0x6a, 0xe2, 0x59, 0xd1,
   0x12, 0x47, 0x34, 0x00, 0x6c, 0x00, 0x00, 0x00,
   0x04, 0x00, 0x00, 0x00,
};

static int
check_info_equal(const struct agx_vdm_state_update_info *actual,
                 const struct agx_vdm_state_update_info *expected)
{
   CHECK(actual->raw_vertex_shader_word0 ==
         expected->raw_vertex_shader_word0);
   CHECK(actual->pipeline == expected->pipeline);
   CHECK(actual->raw_pipeline_template == expected->raw_pipeline_template);
   CHECK(actual->raw_vertex_outputs[0] == expected->raw_vertex_outputs[0]);
   CHECK(actual->raw_vertex_outputs[1] == expected->raw_vertex_outputs[1]);
   CHECK(actual->raw_vertex_unknown == expected->raw_vertex_unknown);
   return 0;
}

static int
check_zero_info(const struct agx_vdm_state_update_info *info)
{
   const struct agx_vdm_state_update_info zero = {0};
   return check_info_equal(info, &zero);
}

static int
check_zero_encoding(const struct agx_vdm_state_update *encoded)
{
   for (unsigned i = 0; i < sizeof(encoded->bytes); ++i)
      CHECK(encoded->bytes[i] == 0);
   CHECK(encoded->length == 0);
   return 0;
}

static int
check_exact_witnesses(void)
{
   for (unsigned i = 0; i < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++i) {
      struct agx_vdm_state_update encoded;
      struct agx_vdm_state_update_info decoded;

      CHECK(agx_vdm_state_update_supported(supported_chips[i]));
      CHECK(agx_vdm_state_update_pack(supported_chips[i], &witness_info,
                                      &encoded));
      CHECK(encoded.length == sizeof(witness));
      CHECK(memcmp(encoded.bytes, witness, sizeof(witness)) == 0);
      CHECK(agx_vdm_state_update_unpack(supported_chips[i], witness,
                                        sizeof(witness), &decoded));
      if (check_info_equal(&decoded, &witness_info))
         return 1;
   }

   return 0;
}

static int
check_template_witnesses(void)
{
   static const enum agx_chip pre_hal300_chips[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   };
   static const enum agx_chip hal300_chips[] = {
      AGX_CHIP_G17G,
      AGX_CHIP_G17S,
      AGX_CHIP_G17C,
      AGX_CHIP_G17D,
   };

   for (unsigned i = 0;
        i < sizeof(pre_hal300_chips) / sizeof(pre_hal300_chips[0]); ++i) {
      struct agx_vdm_state_update_info raw;
      struct agx_vdm_state_update encoded;
      CHECK(agx_vdm_state_update_template_materialize(
         pre_hal300_chips[i], &template_witness_info, &raw));
      CHECK(raw.raw_vertex_shader_word0 == UINT32_C(0x800abc00));
      CHECK(raw.pipeline == template_witness_info.pipeline);
      CHECK(raw.raw_pipeline_template == UINT32_C(0x0011a2b0));
      CHECK(raw.raw_vertex_outputs[0] == UINT32_C(0x00344712));
      CHECK(raw.raw_vertex_outputs[1] == UINT32_C(0x0000007c));
      CHECK(raw.raw_vertex_unknown == 4);
      CHECK(agx_vdm_state_update_template_pack(
         pre_hal300_chips[i], &template_witness_info, &encoded));
      CHECK(encoded.length == sizeof(pre_hal300_template_witness));
      CHECK(memcmp(encoded.bytes, pre_hal300_template_witness,
                   sizeof(pre_hal300_template_witness)) == 0);
   }

   struct agx_vdm_state_update_template_info hal300 = template_witness_info;
   hal300.pre_hal300_flag_group_nonzero = false;
   hal300.pre_hal300_flag_bit_4 = false;
   for (unsigned i = 0;
        i < sizeof(hal300_chips) / sizeof(hal300_chips[0]); ++i) {
      struct agx_vdm_state_update_info raw;
      struct agx_vdm_state_update encoded;
      CHECK(agx_vdm_state_update_template_materialize(hal300_chips[i],
                                                       &hal300, &raw));
      CHECK(raw.raw_vertex_shader_word0 == UINT32_C(0x000abc00));
      CHECK(raw.pipeline == hal300.pipeline);
      CHECK(raw.raw_pipeline_template == UINT32_C(0x0011a2b0));
      CHECK(raw.raw_vertex_outputs[0] == UINT32_C(0x00344712));
      CHECK(raw.raw_vertex_outputs[1] == UINT32_C(0x0000006c));
      CHECK(raw.raw_vertex_unknown == 4);
      CHECK(agx_vdm_state_update_template_pack(hal300_chips[i], &hal300,
                                                &encoded));
      CHECK(encoded.length == sizeof(hal300_template_witness));
      CHECK(memcmp(encoded.bytes, hal300_template_witness,
                   sizeof(hal300_template_witness)) == 0);
   }

   return 0;
}

static int
check_template_variants_and_selectors(void)
{
   static const uint8_t pre_hal300_variant_bits[] = {
      0x30, 0x00, 0x31, 0x01, 0x32, 0x02,
   };
   static const uint8_t hal300_variant_bits[] = {
      0x20, 0x00, 0x21, 0x01, 0x22, 0x02,
   };

   static const uint32_t raw_context_selectors[] = {
      UINT32_MAX, 0, 1, 7, 8, 9, UINT32_C(0x12345678), UINT32_MAX - 1,
   };

   for (unsigned variant = 0; variant < 6; ++variant) {
      for (unsigned selector_index = 0;
           selector_index < sizeof(raw_context_selectors) /
                               sizeof(raw_context_selectors[0]);
           ++selector_index) {
         const uint32_t raw_context_selector =
            raw_context_selectors[selector_index];
         for (unsigned output_selector = 0; output_selector <= UINT8_MAX;
              ++output_selector) {
            struct agx_vdm_state_update_template_info info =
               template_witness_info;
            struct agx_vdm_state_update_info raw;
            info.variant =
               (enum agx_vdm_state_update_template_variant)variant;
            info.raw_active_context_selector = raw_context_selector;
            info.raw_vs_output_selector = output_selector;

            CHECK(agx_vdm_state_update_template_materialize(
               AGX_CHIP_G15G, &info, &raw));
            CHECK(raw.raw_vertex_outputs[1] ==
                  (((uint32_t)output_selector << 2) | UINT32_C(0x40) |
                   pre_hal300_variant_bits[variant]));
            CHECK(raw.raw_vertex_unknown ==
                  (raw_context_selector == UINT32_MAX
                      ? 0
                      : (raw_context_selector - 1) & UINT32_C(0x7)));

            info.pre_hal300_flag_group_nonzero = false;
            info.pre_hal300_flag_bit_4 = false;
            CHECK(agx_vdm_state_update_template_materialize(
               AGX_CHIP_G17G, &info, &raw));
            CHECK(raw.raw_vertex_outputs[1] ==
                  (((uint32_t)output_selector << 2) | UINT32_C(0x40) |
                   hal300_variant_bits[variant]));
            CHECK(raw.raw_vertex_unknown ==
                  (raw_context_selector == UINT32_MAX
                      ? 0
                      : (raw_context_selector - 1) & UINT32_C(0x7)));
         }
      }
   }

   for (unsigned group_nonzero = 0; group_nonzero <= 1; ++group_nonzero) {
      for (unsigned flag_bit_4 = 0; flag_bit_4 <= 1; ++flag_bit_4) {
         struct agx_vdm_state_update_template_info info =
            template_witness_info;
         struct agx_vdm_state_update_info raw;
         info.pre_hal300_flag_group_nonzero = group_nonzero;
         info.pre_hal300_flag_bit_4 = flag_bit_4;
         CHECK(agx_vdm_state_update_template_materialize(AGX_CHIP_G17P, &info,
                                                          &raw));
         const bool bit_31 = !group_nonzero || flag_bit_4;
         CHECK(raw.raw_vertex_shader_word0 ==
               (UINT32_C(0x000abc00) |
                (bit_31 ? UINT32_C(0x80000000) : 0)));
      }
   }

   return 0;
}

static int
check_template_ranges_and_rejections(void)
{
   {
      struct agx_vdm_state_update_template_info info =
         template_witness_info;
      struct agx_vdm_state_update_info raw;
      info.raw_state_load_count_12 = UINT16_C(0x0fff);
      info.raw_state_load_count_18 = UINT32_C(0x0003ffff);
      info.raw_vs_output_count_0 = UINT8_MAX;
      info.raw_vs_output_count_2 = UINT8_MAX;
      info.raw_vs_output_count_1_extra = true;
      info.raw_active_context_selector = UINT32_MAX;
      CHECK(agx_vdm_state_update_template_materialize(AGX_CHIP_G16C, &info,
                                                       &raw));
      CHECK(raw.raw_vertex_shader_word0 == UINT32_C(0x800fff00));
      CHECK(raw.raw_pipeline_template == UINT32_C(0x001ffff8));
      CHECK(raw.raw_vertex_outputs[0] == UINT32_C(0x00ffffff));
      CHECK(raw.raw_vertex_unknown == 0);
   }

   {
      struct agx_vdm_state_update_template_info info =
         template_witness_info;
      struct agx_vdm_state_update_info raw;
      info.raw_state_load_count_12 = UINT16_C(0x1000);
      memset(&raw, 0xa5, sizeof(raw));
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G15G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;

      info = template_witness_info;
      info.raw_state_load_count_18 = UINT32_C(0x00040000);
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G15G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;

      info = template_witness_info;
      info.variant = (enum agx_vdm_state_update_template_variant)-1;
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G15G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;

      info.variant = (enum agx_vdm_state_update_template_variant)6;
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G15G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;
   }

   {
      struct agx_vdm_state_update_template_info info = template_witness_info;
      struct agx_vdm_state_update_info raw;
      info.pre_hal300_flag_group_nonzero = true;
      info.pre_hal300_flag_bit_4 = false;
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G17G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;
      info.pre_hal300_flag_group_nonzero = false;
      info.pre_hal300_flag_bit_4 = true;
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G17G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;
   }

   static const uint64_t invalid_pipelines[] = {
      1, UINT64_C(0x123456789a81), UINT64_C(1) << 48, UINT64_MAX,
   };
   for (unsigned i = 0;
        i < sizeof(invalid_pipelines) / sizeof(invalid_pipelines[0]); ++i) {
      struct agx_vdm_state_update_template_info info = template_witness_info;
      struct agx_vdm_state_update_info raw;
      info.pipeline = invalid_pipelines[i];
      CHECK(!agx_vdm_state_update_template_materialize(AGX_CHIP_G15G, &info,
                                                        &raw));
      if (check_zero_info(&raw))
         return 1;
   }

   for (unsigned i = 0;
        i < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++i) {
      struct agx_vdm_state_update_info raw;
      struct agx_vdm_state_update encoded;
      memset(&raw, 0xa5, sizeof(raw));
      memset(&encoded, 0xa5, sizeof(encoded));
      CHECK(!agx_vdm_state_update_template_materialize(
         unsupported_chips[i], &template_witness_info, &raw));
      if (check_zero_info(&raw))
         return 1;
      CHECK(!agx_vdm_state_update_template_pack(
         unsupported_chips[i], &template_witness_info, &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
   }

   {
      struct agx_vdm_state_update_info raw;
      struct agx_vdm_state_update encoded;
      CHECK(!agx_vdm_state_update_template_materialize(
         AGX_CHIP_G15G, NULL, &raw));
      if (check_zero_info(&raw))
         return 1;
      CHECK(!agx_vdm_state_update_template_materialize(
         AGX_CHIP_G15G, &template_witness_info, NULL));
      CHECK(!agx_vdm_state_update_template_pack(AGX_CHIP_G15G, NULL,
                                                 &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!agx_vdm_state_update_template_pack(
         AGX_CHIP_G15G, &template_witness_info, NULL));
   }

   return 0;
}

static int
check_native_draw_template_producer(void)
{
   for (unsigned target = 0;
        target < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++target) {
      const enum agx_chip chip = supported_chips[target];
      const bool hal300 = chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
                          chip == AGX_CHIP_G17C ||
                          chip == AGX_CHIP_G17D;
      struct agx_vdm_state_update first_path[6];

      for (unsigned selector = 0; selector < 6; ++selector) {
         for (unsigned draw = 0; draw < 4; ++draw) {
            struct agx_vdm_state_update_draw_info info = {
               .source_layout =
                  AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
               .draw_path = (enum agx_vdm_state_update_draw_path)draw,
               .rasterization_enabled = (selector & 1) != 0,
               .raw_provoking_selector = selector >> 1,
               .template_info = template_witness_info,
            };
            struct agx_vdm_state_update expected;
            struct agx_vdm_state_update actual;

            if (hal300) {
               info.template_info.pre_hal300_flag_group_nonzero = false;
               info.template_info.pre_hal300_flag_bit_4 = false;
            }

            /* The renderer selector, not a stale caller value, chooses the
             * template. */
            info.template_info.variant =
               (enum agx_vdm_state_update_template_variant)6;
            struct agx_vdm_state_update_template_info selected =
               info.template_info;
            selected.variant =
               (enum agx_vdm_state_update_template_variant)selector;

            CHECK(agx_vdm_state_update_template_pack(chip, &selected,
                                                       &expected));
            CHECK(agx_vdm_state_update_draw_pack(chip, &info, &actual));
            CHECK(actual.length == AGX_VDM_STATE_UPDATE_LENGTH);
            CHECK(memcmp(actual.bytes, expected.bytes,
                         AGX_VDM_STATE_UPDATE_LENGTH) == 0);

            if (draw == 0)
               first_path[selector] = actual;
            else
               CHECK(memcmp(actual.bytes, first_path[selector].bytes,
                            AGX_VDM_STATE_UPDATE_LENGTH) == 0);
         }
      }
   }

   {
      struct agx_vdm_state_update_draw_info info = {
         .source_layout =
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
         .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
         .template_info = template_witness_info,
      };
      struct agx_vdm_state_update encoded;

      static const enum agx_vdm_state_update_source_layout rejected_layouts[] = {
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED,
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX2_LEGACY,
         (enum agx_vdm_state_update_source_layout)-1,
         (enum agx_vdm_state_update_source_layout)3,
      };
      for (unsigned i = 0;
           i < sizeof(rejected_layouts) / sizeof(rejected_layouts[0]); ++i) {
         info.source_layout = rejected_layouts[i];
         memset(&encoded, 0xa5, sizeof(encoded));
         CHECK(!agx_vdm_state_update_draw_pack(AGX_CHIP_G15G, &info,
                                                &encoded));
         if (check_zero_encoding(&encoded))
            return 1;
      }

      info.source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
      for (int draw = -4; draw <= 7; ++draw) {
         if (draw >= AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED &&
             draw <= AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED)
            continue;
         info.draw_path = (enum agx_vdm_state_update_draw_path)draw;
         memset(&encoded, 0xa5, sizeof(encoded));
         CHECK(!agx_vdm_state_update_draw_pack(AGX_CHIP_G15G, &info,
                                                &encoded));
         if (check_zero_encoding(&encoded))
            return 1;
      }

      info.draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED;
      info.template_info.pipeline = 1;
      CHECK(!agx_vdm_state_update_draw_pack(AGX_CHIP_G15G, &info,
                                             &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      info.template_info = template_witness_info;

      info.raw_provoking_selector = 3;
      CHECK(!agx_vdm_state_update_draw_pack(AGX_CHIP_G15G, &info,
                                             &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      info.raw_provoking_selector = 0;

      for (unsigned i = 0;
           i < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++i) {
         memset(&encoded, 0xa5, sizeof(encoded));
         CHECK(!agx_vdm_state_update_draw_pack(unsupported_chips[i], &info,
                                                &encoded));
         if (check_zero_encoding(&encoded))
            return 1;
      }

      CHECK(!agx_vdm_state_update_draw_pack(AGX_CHIP_G15G, NULL, &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!agx_vdm_state_update_draw_pack(AGX_CHIP_G15G, &info, NULL));
   }

   return 0;
}

static int
check_field_domains(void)
{
   for (unsigned bit = 6; bit < 48; ++bit) {
      struct agx_vdm_state_update_info info = {0};
      struct agx_vdm_state_update_info decoded;
      struct agx_vdm_state_update encoded;
      info.pipeline = UINT64_C(1) << bit;
      CHECK(agx_vdm_state_update_pack(AGX_CHIP_G17S, &info, &encoded));
      CHECK(agx_vdm_state_update_unpack(AGX_CHIP_G17S, encoded.bytes,
                                        encoded.length, &decoded));
      if (check_info_equal(&decoded, &info))
         return 1;
   }

   for (unsigned bit = 0; bit < 22; ++bit) {
      struct agx_vdm_state_update_info info = {0};
      struct agx_vdm_state_update_info decoded;
      struct agx_vdm_state_update encoded;
      info.raw_pipeline_template = UINT32_C(1) << bit;
      CHECK(agx_vdm_state_update_pack(AGX_CHIP_G15G, &info, &encoded));
      CHECK(agx_vdm_state_update_unpack(AGX_CHIP_G15G, encoded.bytes,
                                        encoded.length, &decoded));
      if (check_info_equal(&decoded, &info))
         return 1;
   }

   /* Exercise every bit of all four completely raw payload words. */
   for (unsigned word = 0; word < 4; ++word) {
      for (unsigned bit = 0; bit < 32; ++bit) {
         struct agx_vdm_state_update_info info = {0};
         struct agx_vdm_state_update_info decoded;
         struct agx_vdm_state_update encoded;
         uint32_t *raw_word = NULL;

         switch (word) {
         case 0:
            raw_word = &info.raw_vertex_shader_word0;
            break;
         case 1:
         case 2:
            raw_word = &info.raw_vertex_outputs[word - 1];
            break;
         case 3:
            raw_word = &info.raw_vertex_unknown;
            break;
         default:
            CHECK(false);
         }

         *raw_word = UINT32_C(1) << bit;
         CHECK(agx_vdm_state_update_pack(AGX_CHIP_G16C, &info, &encoded));
         CHECK(agx_vdm_state_update_unpack(AGX_CHIP_G16C, encoded.bytes,
                                           encoded.length, &decoded));
         if (check_info_equal(&decoded, &info))
            return 1;
      }
   }

   return 0;
}

static int
check_every_bit_classification(void)
{
   for (unsigned bit = 0; bit < AGX_VDM_STATE_UPDATE_LENGTH * 8u; ++bit) {
      uint8_t changed[AGX_VDM_STATE_UPDATE_LENGTH];
      struct agx_vdm_state_update_info decoded;
      struct agx_vdm_state_update repacked;
      memcpy(changed, witness, sizeof(changed));
      changed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
      memset(&decoded, 0xa5, sizeof(decoded));

      const bool accepted = agx_vdm_state_update_unpack(
         AGX_CHIP_G17S, changed, sizeof(changed), &decoded);
      if (bit < 32) {
         CHECK(!accepted);
         if (check_zero_info(&decoded))
            return 1;
      } else {
         CHECK(accepted);
         CHECK(agx_vdm_state_update_pack(AGX_CHIP_G17S, &decoded, &repacked));
         CHECK(memcmp(changed, repacked.bytes, sizeof(changed)) == 0);
      }
   }

   return 0;
}

static int
check_rejections(void)
{
   for (unsigned i = 0;
        i < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++i) {
      struct agx_vdm_state_update encoded;
      struct agx_vdm_state_update_info decoded;
      memset(&encoded, 0xa5, sizeof(encoded));
      CHECK(!agx_vdm_state_update_supported(unsupported_chips[i]));
      CHECK(!agx_vdm_state_update_pack(unsupported_chips[i], &witness_info,
                                       &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      memset(&decoded, 0xa5, sizeof(decoded));
      CHECK(!agx_vdm_state_update_unpack(unsupported_chips[i], witness,
                                         sizeof(witness), &decoded));
      if (check_zero_info(&decoded))
         return 1;
   }

   static const uint64_t invalid_pipelines[] = {
      1, 63, UINT64_C(0x123456789a81), UINT64_C(1) << 48, UINT64_MAX,
   };
   for (unsigned i = 0;
        i < sizeof(invalid_pipelines) / sizeof(invalid_pipelines[0]); ++i) {
      struct agx_vdm_state_update_info info = witness_info;
      struct agx_vdm_state_update encoded;
      info.pipeline = invalid_pipelines[i];
      memset(&encoded, 0xa5, sizeof(encoded));
      CHECK(!agx_vdm_state_update_pack(AGX_CHIP_G17S, &info, &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
   }

   static const uint32_t invalid_templates[] = {
      UINT32_C(1) << 22, UINT32_C(0x00400001), UINT32_MAX,
   };
   for (unsigned i = 0;
        i < sizeof(invalid_templates) / sizeof(invalid_templates[0]); ++i) {
      struct agx_vdm_state_update_info info = witness_info;
      struct agx_vdm_state_update encoded;
      info.raw_pipeline_template = invalid_templates[i];
      memset(&encoded, 0xa5, sizeof(encoded));
      CHECK(!agx_vdm_state_update_pack(AGX_CHIP_G15G, &info, &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
   }

   {
      struct agx_vdm_state_update encoded;
      CHECK(!agx_vdm_state_update_pack(AGX_CHIP_G15G, NULL, &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!agx_vdm_state_update_pack(AGX_CHIP_G15G, &witness_info, NULL));
   }

   for (size_t size = 0; size < sizeof(witness); ++size) {
      struct agx_vdm_state_update_info decoded;
      CHECK(!agx_vdm_state_update_unpack(AGX_CHIP_G17S, witness, size,
                                         &decoded));
      if (check_zero_info(&decoded))
         return 1;
   }

   {
      struct agx_vdm_state_update_info decoded;
      uint8_t unaligned[AGX_VDM_STATE_UPDATE_LENGTH + 1];
      memcpy(&unaligned[1], witness, sizeof(witness));
      CHECK(agx_vdm_state_update_unpack(AGX_CHIP_G17S, &unaligned[1],
                                        sizeof(witness), &decoded));
      if (check_info_equal(&decoded, &witness_info))
         return 1;
      CHECK(!agx_vdm_state_update_unpack(AGX_CHIP_G17S, witness,
                                         sizeof(witness) + 1, &decoded));
      if (check_zero_info(&decoded))
         return 1;
      CHECK(!agx_vdm_state_update_unpack(AGX_CHIP_G17S, NULL,
                                         sizeof(witness), &decoded));
      if (check_zero_info(&decoded))
         return 1;
      CHECK(!agx_vdm_state_update_unpack(AGX_CHIP_G17S, witness,
                                         sizeof(witness), NULL));
   }

   return 0;
}

static int
check_vertex_ti_resources(void)
{
   /* VertexProgramVariantESLState: spill quads = (bytes + 3) >> 2, IPR
    * quads = (((bytes + 3) >> 2) + 1) & ~1, both published by finalize. */
   static const struct {
      uint32_t spill_bytes, ipr_bytes;
      uint32_t count_18;
      uint16_t count_12;
   } cases[] = {
      {0, 0, 0, 0},       /* no TI resources */
      {32, 0, 8, 0},      /* Apple G17P terrain VS: eight imported words */
      {48, 160, 12, 40},  /* Apple terrain FS-sized pair */
      {1, 1, 1, 2},       /* partial quads round up; IPR rounds to even */
      {4, 4, 1, 2},
      {5, 8, 2, 2},
      {UINT32_C(0x3ffff) * 4, 0xffe * 4, UINT32_C(0x3ffff), 0xffe},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      struct agx_vdm_state_update_template_info info;
      memcpy(&info, &template_witness_info, sizeof(info));
      CHECK(agx_vdm_state_update_vertex_ti_resources(
         AGX_CHIP_G17P, cases[i].spill_bytes, cases[i].ipr_bytes, &info));
      CHECK(info.raw_state_load_count_18 == cases[i].count_18);
      CHECK(info.raw_state_load_count_12 == cases[i].count_12);

      /* Every other template field is untouched, and the result packs. */
      struct agx_vdm_state_update_template_info expect;
      memcpy(&expect, &template_witness_info, sizeof(expect));
      expect.raw_state_load_count_12 = cases[i].count_12;
      expect.raw_state_load_count_18 = cases[i].count_18;
      CHECK(memcmp(&info, &expect, sizeof(info)) == 0);
      struct agx_vdm_state_update_info raw;
      CHECK(agx_vdm_state_update_template_materialize(AGX_CHIP_G17P, &info,
                                                       &raw));
      CHECK(raw.raw_pipeline_template == cases[i].count_18 << 3);
      CHECK(((raw.raw_vertex_shader_word0 >> 8) & 0xfff) == cases[i].count_12);
   }

   /* Out of range or off-target: fail closed, output unchanged. */
   struct agx_vdm_state_update_template_info info;
   memcpy(&info, &template_witness_info, sizeof(info));
   CHECK(!agx_vdm_state_update_vertex_ti_resources(
      AGX_CHIP_G17P, UINT32_C(0x3ffff) * 4 + 1, 0, &info));
   CHECK(!agx_vdm_state_update_vertex_ti_resources(
      AGX_CHIP_G17P, 0, 0xffe * 4 + 1, &info));
   CHECK(!agx_vdm_state_update_vertex_ti_resources(AGX_CHIP_G17P, UINT32_MAX,
                                                   UINT32_MAX, &info));
   CHECK(!agx_vdm_state_update_vertex_ti_resources(AGX_CHIP_G14G, 32, 0,
                                                   &info));
   CHECK(!agx_vdm_state_update_vertex_ti_resources(AGX_CHIP_G17P, 32, 0,
                                                   NULL));
   CHECK(memcmp(&info, &template_witness_info, sizeof(info)) == 0);
   return 0;
}

int
main(void)
{
   if (check_exact_witnesses() ||
       check_vertex_ti_resources() ||
       check_template_witnesses() ||
       check_template_variants_and_selectors() ||
       check_template_ranges_and_rejections() ||
       check_native_draw_template_producer() || check_field_domains() ||
       check_every_bit_classification() || check_rejections())
      return 1;

   printf("agx_vdm_state_update_core: %u checks passed\n", checks);
   return 0;
}
