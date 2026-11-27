/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_chip.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/vulkan/hk_agx3_vdm_decode.h"
#include "agx3_cdm_direct_launch_semantic_core.h"
#include "agx3_compute_program_core.h"
#include "agx_ioto_core.h"
#include "agx_ioto_load_core.h"
#include "agx_ioto_preshader_core.h"
#include "agx_ioto_profile_core.h"
#include "agx_ioto_program_core.h"
#include "agx_ioto_state_core.h"
#include "agx_ppp.h"
#include "agx_ppp_hal200_graphics_core.h"
#include "agx_ppp_hal300_depth_bias_scissor_core.h"
#include "agx_ppp_hal300_depth_stencil_core.h"
#include "agx_ppp_viewport_payload_core.h"
#include "agx_ppp_w_clamp_cull2_core.h"
#include "agx_vdm_barrier_core.h"
#include "agx_vdm_index_list_core.h"
#include "agx_vdm_ppp_state_core.h"
#include "agx_vdm_state_core.h"
#include "agx_vdm_state_update_core.h"
#include "agx_vdm_stream_link_core.h"
#include "agx_vdm_stream_terminate_core.h"
#include "agx_vdm_tessellate_core.h"
#include "decode.h"

_Static_assert(sizeof(struct drm_asahi_params_global) == 616,
               "drm_asahi_params_global ABI");
_Static_assert(offsetof(struct drm_asahi_params_global, usc_generation) == 592,
               "USC generation ABI");
_Static_assert(offsetof(struct drm_asahi_params_global, gpu_hal_generation) ==
                  596,
               "GPU HAL generation ABI");
_Static_assert(offsetof(struct drm_asahi_params_global, memory_limit) == 600,
               "memory limit ABI");
_Static_assert(offsetof(struct drm_asahi_params_global,
                        client_memory_limit) == 608,
               "client memory limit ABI");
_Static_assert(DRM_ASAHI_GPU_HAL_LEGACY == 0, "legacy HAL ABI");
_Static_assert(DRM_ASAHI_GPU_HAL_200 == 200, "HAL200 ABI");
_Static_assert(DRM_ASAHI_GPU_HAL_300 == 300, "HAL300 ABI");

static unsigned failures;
static unsigned checks;

#define CHECK_EQ(actual, expected)                                             \
   do {                                                                        \
      unsigned value = (unsigned)(actual);                                     \
      checks++;                                                                \
      if (value != (unsigned)(expected)) {                                     \
         fprintf(stderr, "%s:%d: got %u, expected %u\n", __FILE__, __LINE__,   \
                 value, (unsigned)(expected));                                 \
         failures++;                                                           \
      }                                                                        \
   } while (0)

#define CHECK(condition)                                                       \
   do {                                                                        \
      checks++;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

/*
 * The truth table: what every chip predicate, table accessor and per-record
 * allowlist answers for every enum agx_chip value. It pins behaviour, so a
 * refactor of the chip tables must leave it passing unedited; a row changes
 * only together with the hardware evidence for the new answer.
 *
 * Each row is an expression of `c` and its value for the twenty-one identities in
 * enum order, one base-36 digit each, grouped by generation:
 *
 *    G13G G13X G14G G14X | G15P G15G G15S G15C G15D | G16P G16G G16S G16C |
 *    G17P G17A G17G G17S G17C G17D | G18G | UNKNOWN
 *
 * A failing row prints the string the code produces.
 */
#define TRUTH_TABLE(ROW)                                                       \
   /* Identity and generation axes. */                                         \
   ROW(agx_chip_isa(c), "2222 23333 3333 333333 3 0")                           \
   ROW(agx_chip_isa_encoding(c), "0000 01111 1111 112222 3 4")                  \
   ROW(agx_chip_command_stream_generation(c), "0000 41111 2222 223333 3 4")     \
   ROW(agx_chip_descriptor_generation(c), "7788 09999 9999 99aaaa a 0")         \
   ROW(agx_chip_sampler_generation(c), "0000 30000 1111 112222 2 3")            \
   ROW(agx_chip_sampler_heap_element_size(c) / 8, "1111 04444 4444 444444 4 0") \
   ROW(agx_chip_submission_model(c), "0000 30000 3011 132222 2 3")              \
   ROW(agx_chip_record_family(c), "0123 4ddd8 9dbc defghi j k")                 \
   ROW(agx_encoding_is_agx3(agx_chip_isa_encoding(c)),                         \
       "0000 01111 1111 111111 1 0")                                            \
   ROW(agx_isa_is_agx3(agx_chip_isa_encoding(c)), "0000 01111 1111 111111 0 0") \
   ROW(agx_encoding_is_g17_class(agx_chip_isa_encoding(c)),                    \
       "0000 00000 0000 001111 1 0")                                            \
   /* Paths. */                                                                \
   ROW(agx_chip_info(c)->path, "1111 02220 0200 203333 3 0")                    \
   ROW(agx_chip_native_agx3(c), "0000 01110 0100 100000 0 0")                   \
   ROW(agx_chip_native_g15(c), "0000 01110 0000 000000 0 0")                    \
   ROW(agx_chip_g17_m5(c), "0000 00000 0000 001111 0 0")                        \
   ROW(agx_chip_hal300_records(c), "0000 00000 0000 001000 1 0")                \
   ROW(agx_chip_m5_paths(c), "0000 00000 0000 001000 1 0")                      \
   ROW(agx_chip_m5_layer_operand(c), "0000 00000 0000 000010 0 0")              \
   ROW(agx_chip_gl_through_zink(c), "0000 00000 0000 000001 0 0")               \
   ROW(agx_chip_codegen_supported(c), "1111 11111 1111 111111 0 0")             \
   ROW(agx_chip_driver_supported(c), "1111 01110 0100 100000 0 0")              \
   ROW(agx_chip_uses_legacy_precompiled_programs(c),                           \
       "1111 00000 0000 000000 0 0")                                            \
   ROW(agx_chip_hal300_path(c), "0000 00000 0000 001111 1 0")                   \
   ROW(agx_chip_agx3_path(c), "0000 01110 0100 101111 1 0")                     \
   ROW(agx_chip_native_or_m5(c), "0000 01110 0100 101111 0 0")                  \
   ROW(agx_chip_native_or_g18g(c), "0000 01110 0100 100000 1 0")                \
   ROW(agx_chip_g17p_or_m5(c), "0000 00000 0000 101111 0 0")                    \
   /* Capabilities and quirks. */                                              \
   ROW(agx_chip_has_native_render_context(c), "0000 00000 0000 100000 0 0")     \
   ROW(agx_chip_native_indirect_compute(c), "0000 01110 0100 100000 0 0")       \
   ROW(agx_chip_indirect_geometry_flush(c), "0000 01110 0000 000000 0 0")       \
   ROW(agx_chip_cdm_wide_visibility(c), "0000 01110 0000 000000 0 0")           \
   ROW(agx_chip_elide_app_dispatch_barrier(c), "0000 01110 0100 101111 0 0")    \
   ROW(agx_chip_usc_helper_table_slots(c, 0), "aaaa aaaaa aaaa 6aaaaa a a")     \
   ROW(agx_chip_usc_helper_table_slots(c, 6), "aaaa aaaaa aaaa 6aaaaa a a")     \
   ROW(agx_chip_usc_helper_table_slots(c, 10), "aaaa aaaaa aaaa aaaaaa a a")    \
   ROW(agx_chip_usc_base_page_fill_returns(c), "0000 00010 0000 000000 0 0")    \
   ROW(agx_chip_usc_base_page_fill_returns_for(c, 1),                          \
       "0000 00010 0000 000000 0 0")                                            \
   ROW(agx_chip_usc_base_page_fill_returns_for(c, 2),                          \
       "0000 00010 0000 000000 0 0")                                            \
   ROW(agx_chip_usc_base_page_fill_returns_for(c, 4),                          \
       "0000 01110 0100 100000 0 0")                                            \
   ROW(agx_chip_explicit_robust_image_bounds(c), "0000 01110 0100 000000 0 0")  \
   ROW(agx_chip_vdm_indirect_index_bounds(c), "0000 00000 0000 101110 0 0")     \
   ROW(agx_chip_hw_texel_fetch_bounds(c), "0000 00000 0000 001110 0 0")         \
   ROW(agx_chip_native_imageblock(c), "0000 01110 0100 100000 0 0")             \
   ROW(agx_chip_uncovered_tile_ms_defect(c), "0000 01110 0100 100000 0 0")      \
   ROW(agx_chip_uncovered_tile_ms_reload_defect(c, 0x8140, 2),                 \
       "0000 01110 0100 100000 0 0")                                            \
   ROW(agx_chip_uncovered_tile_ms_reload_defect(c, 0x8140, 4),                 \
       "0000 01110 0100 100000 0 0")                                            \
   ROW(agx_chip_uncovered_tile_ms_reload_defect(c, 0x6040, 2),                 \
       "0000 01110 0100 000000 0 0")                                            \
   ROW(agx_chip_uncovered_tile_ms_reload_defect(c, 0x6040, 4),                 \
       "0000 01110 0100 100000 0 0")                                            \
   ROW(agx_chip_uncovered_tile_ms_reload_defect(c, 0x6041, 2),                 \
       "0000 01110 0100 000000 0 0")                                            \
   ROW(agx_chip_uncovered_tile_ms_reload_defect(c, 0x6041, 4),                 \
       "0000 01110 0100 100000 0 0")                                            \
   ROW(agx_chip_tri_merge_across_side_effects(c), "0000 00000 0000 101110 0 0") \
   ROW(agx_chip_image_load_misses_own_store(c), "0000 00000 0000 100000 0 0")   \
   ROW(agx_chip_exact_fma32(c), "1111 11111 1111 111111 1 1")                   \
   ROW(agx_chip_sampler_has_lod_bias(c), "0000 00000 1111 111111 1 0")          \
   ROW(agx_chip_native_preamble(c), "0000 00000 0100 100000 0 0")               \
   ROW(agx_chip_query_ti_spill(c), "0000 00000 0100 100000 0 0")                \
   ROW(agx_chip_has_hardware_ray_tracing(c), "0000 01111 1111 111111 1 0")      \
   ROW(agx_chip_has_agx3_ray_opcodes(c), "0000 01111 1111 111111 0 0")          \
   ROW(agx_chip_needs_scheduling_stalls(c), "0000 01000 0000 000000 0 0")       \
   ROW(agx_chip_cooperative_matrix_supported(c), "1111 01111 1111 111111 0 0")  \
   ROW(agx_chip_local_read_tilebuffer(c), "0000 00000 1111 111111 0 0")         \
   /* Per-record allowlists: VDM. */                                           \
   ROW(agx_vdm_state_vertex_pipeline_pair_supported(c),                        \
       "0000 01111 0111 101111 1 0")                                            \
   ROW(agx_vdm_state_update_supported(c), "0000 01111 0111 101111 1 0")         \
   ROW(agx_vdm_ppp_state_core_supported(c), "0000 01111 0111 101111 1 0")       \
   ROW(agx_vdm_barrier_core_supported(c), "0000 01111 0111 101111 1 0")         \
   ROW(agx_vdm_barrier_core_word(c, false) == AGX_VDM_BARRIER_CORE_PLAIN,      \
       "0000 00110 0100 000000 0 0")                                            \
   ROW(agx_vdm_stream_link_core_supported(c), "0000 01111 0111 101111 1 0")     \
   ROW(agx_chip_vdm_stream_link_supported(c), "1111 01111 0111 101111 1 0")     \
   ROW(agx_vdm_stream_terminate_core_supported(c),                             \
       "0000 01111 0111 101111 1 0")                                            \
   ROW(agx_vdm_tessellate_direct_target_supported(c),                          \
       "0000 01111 0111 101111 0 0")                                            \
   ROW(agx_vdm_index_list_direct_target_supported_inline(c),                   \
       "0000 01111 0111 101111 0 0")                                            \
   ROW(agx_vdm_index_list_direct_is_hal300_inline(c),                          \
       "0000 00000 0000 001111 0 0")                                            \
   ROW(agx_vdm_index_list_record_chip(c), "0123 45678 9abc defghi f k")         \
   ROW(agx_vdm_index_list_indexed_or_indirect_supported_inline(c),             \
       "0000 01111 0111 101111 1 0")                                            \
   ROW(agx_vdm_index_list_nonindexed_has_tail(c), "0000 00000 0000 001111 1 0") \
   ROW(agx_vdm_index_list_hal300_unk2(c, 12), "7777 77777 7777 777777 5 7")     \
   /* Per-record allowlists: PPP. */                                           \
   ROW(agx_ppp_legacy_payload_supported(c), "1111 01111 0111 100000 0 0")       \
   ROW(agx_ppp_hal200_graphics_core_supported(c), "0000 01110 0100 100000 0 0") \
   ROW(agx_ppp_hal300_depth_stencil_core_supported(c),                         \
       "0000 00000 0000 001111 1 0")                                            \
   ROW(agx_ppp_hal300_depth_bias_scissor_core_supported(c),                    \
       "0000 00000 0000 001111 1 0")                                            \
   ROW(agx_ppp_viewport_payload_core_supported(c),                             \
       "0000 01111 0111 101111 1 0")                                            \
   ROW(viewport_payload_generation(c), "0000 06666 0666 607777 7 0")            \
   ROW(w_clamp_cull2_generation(c), "0000 06666 0666 607777 7 0")               \
   /* Per-record allowlists: CDM. */                                           \
   ROW(agx3_cdm_launch_target_grounded(c), "0000 01111 0111 101111 1 0")        \
   ROW(agx3_cdm_direct_launch_target_grounded(c), "0000 01111 0111 101111 1 0") \
   ROW(agx3_cdm_indirect_launch_target_grounded(c),                            \
       "0000 01111 0111 101111 1 0")                                            \
   ROW(agx3_cdm_indirect_launch_36_target_grounded(c),                         \
       "0000 01111 0111 101111 1 0")                                            \
   ROW(agx3_cdm_stream_link_target_grounded(c), "0000 01111 0111 101111 0 0")   \
   ROW(agx3_cdm_link_padding_target_grounded(c), "0000 01111 0111 101111 0 0")  \
   ROW(agx3_cdm_barrier_target_grounded(c), "0000 01111 0111 101111 1 0")       \
   ROW(agx3_cdm_barrier_flags_grounded(c, 0), "0000 01111 0111 101111 1 0")     \
   ROW(agx3_cdm_barrier_flags_grounded(c, 1), "0000 00000 0000 001111 0 0")     \
   ROW(agx3_cdm_barrier_flags_grounded(c, 5), "0000 01111 0111 101111 1 0")     \
   ROW(agx3_cdm_barrier_flags_grounded(c, 0xf), "0000 00000 0000 001111 0 0")   \
   ROW(agx3_cdm_barrier_flags_grounded(c, 0x10), "0000 00000 0000 000000 0 0")  \
   ROW(agx3_cdm_barrier_wide_visibility_grounded(c),                           \
       "0000 01110 0100 000000 0 0")                                            \
   ROW(agx3_cdm_barrier_flush_memory_grounded(c), "0000 00000 0000 101111 0 0") \
   ROW(agx3_cdm_substream_barrier_target_grounded(c),                          \
       "0000 01111 0111 101111 0 0")                                            \
   ROW(agx3_cdm_secondary_barrier_target_grounded(c),                          \
       "0000 00000 0000 001111 0 0")                                            \
   ROW(agx3_g17s_minimal_compute_dispatch_ready(c),                            \
       "0000 00000 0000 000100 0 0")                                            \
   ROW(agx3_native_indirect_compute_dispatch_size(c) / 4,                      \
       "iiii ijjji iiii iiiiii i i")                                            \
   /* Per-record allowlists: Ioto. */                                          \
   ROW(agx_chip_ioto_generation(c), "0000 01111 0111 102222 0 0")               \
   ROW(agx_ioto_target_supported(c), "0000 01111 0111 101111 0 0")              \
   ROW(agx_ioto_load_target_encoding(c), "0000 01111 0111 102222 0 0")          \
   ROW(agx_ioto_profile_control_supported(c), "0000 01111 0111 101111 0 0")     \
   ROW(agx_ioto_profile_control_size(c), "0000 04444 0444 406666 0 0")          \
   ROW(ioto_preshader_generation(c), "0000 01111 0111 102222 0 0")              \
   ROW(ioto_state_generation(c), "0000 01111 0111 102222 0 0")                  \
   ROW(ioto_program_inline_lddev(c), "0000 01111 0111 100000 0 0")              \
   /* Outside lib: the command-stream decoder and the Vulkan driver. */        \
   ROW(decoder_chip(c), "0123 45678 9abc defghi j k")                           \
   ROW(shader_binary_version_class(c), "2222 21111 1111 313333 1 1")

/* The Ioto encoding a codec packs for a chip: 1 for GPU0, 2 for GPU1, or 0
 * when it refuses the chip. The preshader branch starts with a different byte
 * in each encoding, and each encoding refuses the other's SPECTPR state. */
static unsigned
ioto_preshader_generation(enum agx_chip chip)
{
   uint8_t out[AGX_IOTO_PRESHADER_BRANCH_SIZE];

   if (!agx_ioto_preshader_pack_branch(chip, 0, false, out))
      return 0;

   return out[0] == 0x0f ? 1 : out[0] == 0x5e ? 2 : 9;
}

static unsigned
ioto_state_generation(enum agx_chip chip)
{
   const struct agx_ioto_state gpu0 = {0}, gpu1 = {.f = {3, 166}};
   uint8_t out[AGX_IOTO_SPECTPR_SIZE];

   return agx_ioto_state_pack_spectpr(chip, &gpu0, out, sizeof(out))   ? 1
          : agx_ioto_state_pack_spectpr(chip, &gpu1, out, sizeof(out)) ? 2
                                                                       : 0;
}

/* Whether the program codec takes the LDDEV that GPU0 programs carry inline,
 * which no Ioto record decoder knows. */
static unsigned
ioto_program_inline_lddev(enum agx_chip chip)
{
   const uint8_t lddev[AGX_IOTO_GPU0_INLINE_LDDEV_SIZE] = {0x67};
   size_t offset = 0;

   return agx_ioto_program_copy_suffix_offset(chip, lddev, sizeof(lddev),
                                              &offset) &&
          offset == sizeof(lddev);
}

static unsigned decoder_chip(enum agx_chip chip);

/* Which of the Vulkan driver's three shader binary versions a chip's
 * binaries carry: 1 classic, 2 AGX2, 3 AGX3. */
static unsigned
shader_binary_version_class(enum agx_chip chip)
{
   switch (hk_shader_binary_version(chip)) {
   case HK_CLASSIC_SHADER_BINARY_VERSION:
      return 1;
   case HK_AGX2_SHADER_BINARY_VERSION:
      return 2;
   case HK_AGX3_SHADER_BINARY_VERSION:
      return 3;
   default:
      return 9;
   }
}

/* The PPP encoder generation a payload codec packs for a chip: 6 or 7, or 0
 * when it refuses the chip. */
static unsigned
viewport_payload_generation(enum agx_chip chip)
{
   const struct agx_ppp_viewport_payload_fields fields = {.viewport_count = 1};
   struct agx_ppp_viewport_payload_core out;
   uint32_t header;

   if (!agx_ppp_viewport_payload_core_pack(chip, &fields, &out))
      return 0;

   memcpy(&header, out.bytes, sizeof(header));
   return header == AGX_PPP_VIEWPORT_PAYLOAD_GEN6_BASE_HEADER   ? 6
          : header == AGX_PPP_VIEWPORT_PAYLOAD_GEN7_BASE_HEADER ? 7
                                                                : 9;
}

static unsigned
w_clamp_cull2_generation(enum agx_chip chip)
{
   const struct agx_ppp_w_clamp_cull2_fields fields = {0};
   struct agx_ppp_w_clamp_cull2_core out;
   uint32_t header;

   if (!agx_ppp_w_clamp_cull2_core_pack(chip, &fields, &out))
      return 0;

   memcpy(&header, out.bytes, sizeof(header));
   return header == AGX_PPP_W_CLAMP_CULL2_GEN6_HEADER   ? 6
          : header == AGX_PPP_W_CLAMP_CULL2_GEN7_HEADER ? 7
                                                        : 9;
}

static const char *const chip_names[AGX_CHIP_UNKNOWN + 1] = {
   "G13G", "G13X", "G14G", "G14X", "G15P", "G15G",    "G15S",
   "G15C", "G15D", "G16P", "G16G", "G16S", "G16C",    "G17P",
   "G17A", "G17G", "G17S", "G17C", "G17D", "G18G", "unknown",
};

static void
check_row(const char *expr, const unsigned *got, const char *want)
{
   char got_s[AGX_CHIP_UNKNOWN + 2] = {0}, want_s[AGX_CHIP_UNKNOWN + 2] = {0};
   unsigned n = 0;

   for (; *want && n <= AGX_CHIP_UNKNOWN; ++want) {
      if (*want != ' ')
         want_s[n++] = *want;
   }
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i)
      got_s[i] =
         got[i] < 36 ? "0123456789abcdefghijklmnopqrstuvwxyz"[got[i]] : '?';

   checks++;
   if (strcmp(got_s, want_s)) {
      fprintf(stderr, "%s:\n  got      \"%.4s %.5s %.4s %.6s %.1s %.1s\"\n",
              expr, got_s, got_s + 4, got_s + 9, got_s + 13, got_s + 19,
              got_s + 20);
      fprintf(stderr, "  expected \"%s\"\n", want_s);
      failures++;
   }
}

static void
test_truth_table(void)
{
   _Static_assert(AGX_CHIP_UNKNOWN == 20, "one column per chip");

#define ROW(expr, want)                                                        \
   {                                                                           \
      unsigned got[AGX_CHIP_UNKNOWN + 1];                                      \
      for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {                       \
         const enum agx_chip c = (enum agx_chip)i;                             \
         got[i] = (unsigned)(expr);                                            \
      }                                                                        \
      check_row(#expr, got, want);                                             \
      /* A value outside the enum answers as UNKNOWN does (or, for the rows    \
       * that name a chip, as itself). */                                      \
      for (unsigned i = 0; i < 3; ++i) {                                       \
         const enum agx_chip c =                                               \
            (enum agx_chip)(i ? AGX_CHIP_UNKNOWN + i : -1);                    \
         const unsigned value = (unsigned)(expr);                              \
         CHECK(value == got[AGX_CHIP_UNKNOWN] || value == (unsigned)c);        \
      }                                                                        \
   }
   TRUTH_TABLE(ROW)
#undef ROW

   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const char *product = agx_chip_exact_product_name(c);

      CHECK(strcmp(agx_chip_name(c), chip_names[i]) == 0);
      if (c == AGX_CHIP_G17P)
         CHECK(product && strcmp(product, "A18 Pro") == 0);
      else
         CHECK(product == NULL);
   }
}

/* Each chip-set predicate is the comparison chain it replaced at the call
 * sites. */
static void
test_chip_sets(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool native = agx_chip_native_agx3(c);
      const bool m5 = agx_chip_g17_m5(c);

      CHECK(agx_chip_hal300_path(c) == (m5 || c == AGX_CHIP_G18G));
      CHECK(agx_chip_uses_shade_model_5(c) == (m5 || c == AGX_CHIP_G18G));
      CHECK(agx_chip_agx3_path(c) == (native || m5 || c == AGX_CHIP_G18G));
      CHECK(agx_chip_native_or_m5(c) == (native || m5));
      CHECK(agx_chip_native_or_g18g(c) == (native || c == AGX_CHIP_G18G));
      CHECK(agx_chip_g17p_or_m5(c) == (c == AGX_CHIP_G17P || m5));
      CHECK(m5 == (c == AGX_CHIP_G17G || c == AGX_CHIP_G17S ||
                   c == AGX_CHIP_G17C || c == AGX_CHIP_G17D));
   }
}

/* Exact driver-core identities recovered from Apple's readChipInfo table. */
static const struct {
   uint32_t generation;
   uint32_t variant;
   uint32_t clusters;
   enum agx_chip chip;
} chip_cases[] = {
   {13, 'G', 1, AGX_CHIP_G13G}, {13, 'S', 2, AGX_CHIP_G13X},
   {13, 'C', 4, AGX_CHIP_G13X}, {13, 'D', 8, AGX_CHIP_G13X},
   {14, 'G', 1, AGX_CHIP_G14G}, {14, 'S', 2, AGX_CHIP_G14X},
   {14, 'C', 4, AGX_CHIP_G14X}, {14, 'D', 8, AGX_CHIP_G14X},
   {15, 'P', 1, AGX_CHIP_G15P}, {15, 'G', 1, AGX_CHIP_G15G},
   {15, 'S', 2, AGX_CHIP_G15S}, {15, 'C', 4, AGX_CHIP_G15C},
   {15, 'D', 8, AGX_CHIP_G15D}, {16, 'P', 1, AGX_CHIP_G16P},
   {16, 'G', 1, AGX_CHIP_G16G}, {16, 'S', 2, AGX_CHIP_G16S},
   {16, 'C', 4, AGX_CHIP_G16C}, {17, 'P', 1, AGX_CHIP_G17P},
   {17, 'A', 1, AGX_CHIP_G17A}, {17, 'G', 1, AGX_CHIP_G17G},
   {17, 'S', 2, AGX_CHIP_G17S}, {17, 'C', 4, AGX_CHIP_G17C},
   {17, 'D', 8, AGX_CHIP_G17D},
   {18, 'G', 1, AGX_CHIP_G18G},
};

/* The chip the command-stream decoder takes a chip's first kernel identity
 * as. */
static unsigned
decoder_chip(enum agx_chip chip)
{
   for (unsigned i = 0; i < sizeof(chip_cases) / sizeof(chip_cases[0]); ++i) {
      if (chip_cases[i].chip == chip) {
         const struct drm_asahi_params_global params = {
            .gpu_generation = chip_cases[i].generation,
            .gpu_variant = chip_cases[i].variant,
         };
         return agxdecode_chip_from_params(&params);
      }
   }

   return agxdecode_chip_from_params(NULL);
}

/* The decoder has its own mapping from the kernel's identity. It is that of
 * agx_chip_from_params() without the cluster-count check (the counts here
 * are zero), plus variant 'X': a trace that names the M4 Max that way decodes
 * as G17P, where device open refuses the variant. */
static void
test_decoder_identity(void)
{
   for (unsigned i = 0; i < sizeof(chip_cases) / sizeof(chip_cases[0]); ++i) {
      const struct drm_asahi_params_global params = {
         .gpu_generation = chip_cases[i].generation,
         .gpu_variant = chip_cases[i].variant,
      };
      CHECK_EQ(agxdecode_chip_from_params(&params), chip_cases[i].chip);
   }
   CHECK_EQ(agxdecode_chip_from_params(NULL), AGX_CHIP_UNKNOWN);

   /* The M4 Pro and M4 Max on the G17 kernel runtime are G17P. */
   static const struct {
      uint32_t variant, chip_id, hal;
      enum agx_chip decoder, device;
   } g17_runtime[] = {
      {'S', 0x6040, DRM_ASAHI_GPU_HAL_200, AGX_CHIP_G17P, AGX_CHIP_G17P},
      {'S', 0x6041, DRM_ASAHI_GPU_HAL_200, AGX_CHIP_G16S, AGX_CHIP_G16S},
      {'S', 0x6040, DRM_ASAHI_GPU_HAL_LEGACY, AGX_CHIP_G16S, AGX_CHIP_G16S},
      {'C', 0x6041, DRM_ASAHI_GPU_HAL_200, AGX_CHIP_G17P, AGX_CHIP_G17P},
      {'C', 0x6040, DRM_ASAHI_GPU_HAL_200, AGX_CHIP_G16C, AGX_CHIP_G16C},
      {'C', 0x6041, DRM_ASAHI_GPU_HAL_LEGACY, AGX_CHIP_G16C, AGX_CHIP_G16C},
      {'X', 0x6041, DRM_ASAHI_GPU_HAL_200, AGX_CHIP_G17P, AGX_CHIP_UNKNOWN},
      {'X', 0x6040, DRM_ASAHI_GPU_HAL_200, AGX_CHIP_UNKNOWN, AGX_CHIP_UNKNOWN},
      {'X', 0x6041, DRM_ASAHI_GPU_HAL_LEGACY, AGX_CHIP_UNKNOWN,
       AGX_CHIP_UNKNOWN},
   };
   for (unsigned i = 0; i < sizeof(g17_runtime) / sizeof(g17_runtime[0]); ++i) {
      const struct drm_asahi_params_global params = {
         .gpu_generation = 16,
         .gpu_variant = g17_runtime[i].variant,
         .chip_id = g17_runtime[i].chip_id,
         .num_clusters_total = 4,
         .usc_generation = 3,
         .gpu_hal_generation = g17_runtime[i].hal,
      };
      CHECK_EQ(agxdecode_chip_from_params(&params), g17_runtime[i].decoder);
      CHECK_EQ(agx_chip_from_params(&params), g17_runtime[i].device);
   }

   /* 'X' is the M4 Max only. */
   struct drm_asahi_params_global x = {
      .gpu_generation = 17,
      .gpu_variant = 'X',
      .chip_id = 0x6041,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
   };
   CHECK_EQ(agxdecode_chip_from_params(&x), AGX_CHIP_UNKNOWN);
   x.gpu_generation = 16;
   x.usc_generation = 2;
   CHECK_EQ(agxdecode_chip_from_params(&x), AGX_CHIP_UNKNOWN);
}

static enum agx_chip
chip_for(uint32_t generation, uint32_t variant, uint32_t clusters)
{
   struct drm_asahi_params_global params = {0};
   params.gpu_generation = generation;
   params.gpu_variant = variant;
   params.num_clusters_total = clusters;
   return agx_chip_from_params(&params);
}

static void
test_recovered_table(void)
{
   bool seen[AGX_CHIP_UNKNOWN + 1] = {0};

   for (unsigned i = 0; i < sizeof(chip_cases) / sizeof(chip_cases[0]); ++i) {
      CHECK_EQ(chip_for(chip_cases[i].generation, chip_cases[i].variant,
                        chip_cases[i].clusters),
               chip_cases[i].chip);
      seen[chip_cases[i].chip] = true;
   }

   /* Every named enum is reachable from the kernel's identity. */
   for (int chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip)
      CHECK(seen[chip]);
}

static void
test_variant_is_load_bearing(void)
{
   /* Cluster count must never be used as a substitute for gpu_variant. */
   CHECK_EQ(chip_for(17, 'P', 4), AGX_CHIP_G17P);
   CHECK_EQ(chip_for(17, 'G', 4), AGX_CHIP_G17G);
   CHECK_EQ(chip_for(15, 'P', 2), AGX_CHIP_G15P);
   CHECK_EQ(chip_for(15, 'S', 1), AGX_CHIP_G15S);
   CHECK_EQ(chip_for(18, 'G', 4), AGX_CHIP_G18G);
}

/* The kernel-published identity is checked beyond the chip: each admitted
 * profile is one exact SoC contract. */
static void
test_params_contracts(void)
{
   struct drm_asahi_params_global g17p = {
      .gpu_generation = 17,
      .gpu_variant = 'P',
      .chip_id = 0x8140,
      .num_dies = 1,
      .num_clusters_total = 1,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
   };
   CHECK(agx_params_driver_supported(&g17p));
   g17p.chip_id = 0x8141;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.chip_id = 0x8140;
   g17p.num_dies = 2;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.num_dies = 1;
   g17p.num_clusters_total = 2;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.num_clusters_total = 1;
   g17p.usc_generation = 2;
   CHECK(!agx_params_driver_supported(&g17p));
   g17p.usc_generation = 3;
   g17p.gpu_hal_generation = DRM_ASAHI_GPU_HAL_300;
   CHECK(!agx_params_driver_supported(&g17p));
   CHECK(!agx_params_driver_supported(NULL));

   struct drm_asahi_params_global m5 = {
      .gpu_generation = 17,
      .gpu_variant = 'G',
      .chip_id = 0x8142,
      .num_dies = 1,
      .num_clusters_total = 1,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
   };
   CHECK(!agx_params_driver_supported(&m5));
   CHECK(agx_params_m5_supported(&m5));
   m5.chip_id = 0x8140;
   CHECK(!agx_params_m5_supported(&m5));
   m5.chip_id = 0x8142;
   m5.gpu_variant = 'P';
   CHECK(!agx_params_m5_supported(&m5));
   m5.gpu_variant = 'G';
   m5.num_dies = 2;
   CHECK(!agx_params_m5_supported(&m5));
   m5.num_dies = 1;
   m5.num_clusters_total = 2;
   CHECK(!agx_params_m5_supported(&m5));
   m5.num_clusters_total = 1;
   m5.usc_generation = 2;
   CHECK(!agx_params_m5_supported(&m5));
   m5.usc_generation = 3;
   m5.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   CHECK(!agx_params_m5_supported(&m5));
   CHECK(!agx_params_m5_supported(NULL));

   /* The M5 bring-up contract must not admit the M6. */
   struct drm_asahi_params_global m6 = m5;
   m6.gpu_generation = 18;
   m6.chip_id = 0x8152;
   CHECK_EQ(agx_chip_from_params(&m6), AGX_CHIP_G18G);
   CHECK(!agx_params_m5_supported(&m6));
   CHECK(!agx_params_driver_supported(&m6));

   CHECK(strcmp(agx_command_stream_generation_name(
                   AGX_CS_GENERATION_HAL200),
                "HAL200") == 0);
   CHECK(strcmp(agx_command_stream_generation_name(
                   AGX_CS_GENERATION_HAL300),
                "HAL300") == 0);
}

static void
test_invalid_identity(void)
{
   CHECK_EQ(chip_for(12, 'G', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(18, 'P', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(18, 'S', 2), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(19, 'G', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(15, 'A', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(16, 'D', 8), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(18, 'D', 8), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(13, 'P', 1), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(13, 'G', 0), AGX_CHIP_UNKNOWN);
   CHECK_EQ(chip_for(17, 'S', DRM_ASAHI_MAX_CLUSTERS + 1),
            AGX_CHIP_UNKNOWN);
}

static void
test_classification_invariants(void)
{
   CHECK_EQ(AGX_ISA_AGX2, 2);
   CHECK_EQ(AGX_ISA_AGX3, 3);

   for (int chip = AGX_CHIP_G13G; chip <= AGX_CHIP_UNKNOWN; ++chip) {
      const enum agx_chip c = (enum agx_chip)chip;
      const enum agx_isa isa = agx_chip_isa(c);
      const enum agx_isa_encoding encoding = agx_chip_isa_encoding(c);
      const bool known = c != AGX_CHIP_UNKNOWN;
      const bool agx3_encoding = encoding == AGX_ISA_ENCODING_AGX3B ||
                                 encoding == AGX_ISA_ENCODING_AGX3C ||
                                 encoding == AGX_ISA_ENCODING_AGX3D;

      CHECK(strcmp(agx_chip_name(c), "") != 0);
      CHECK((isa != AGX_ISA_UNKNOWN) == known);
      CHECK((encoding != AGX_ISA_ENCODING_UNKNOWN) == known);
      CHECK(agx3_encoding == (isa == AGX_ISA_AGX3));
      CHECK((encoding == AGX_ISA_ENCODING_AGX2) == (isa == AGX_ISA_AGX2));
      CHECK(agx_chip_has_hardware_ray_tracing(c) == (isa == AGX_ISA_AGX3));
   }

   CHECK(strcmp(agx_isa_name(AGX_ISA_AGX2), "AGX2") == 0);
   CHECK(strcmp(agx_isa_name(AGX_ISA_AGX3), "AGX3") == 0);
   CHECK(strcmp(agx_isa_name(AGX_ISA_UNKNOWN), "unknown") == 0);
}

/* Device tree compatibles name a chip as the kernel's generation and variant
 * do; the loader reads them to pick zink (agx_chip_gl_through_zink()). */
static void
test_compatible(void)
{
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g17d"), AGX_CHIP_G17D);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g17c"), AGX_CHIP_G17C);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g13g"), AGX_CHIP_G13G);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g14s"), AGX_CHIP_G14X);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g13x"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-t6050"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g17"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g17dd"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g017d"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-g"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible("apple,agx-gD"), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible(""), AGX_CHIP_UNKNOWN);
   CHECK_EQ(agx_chip_from_compatible(NULL), AGX_CHIP_UNKNOWN);

   /* Every chip GL reaches through zink is named by its compatible. */
   for (unsigned i = 0; i < AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      if (!agx_chip_gl_through_zink(c))
         continue;
      char compatible[32];
      snprintf(compatible, sizeof(compatible), "apple,agx-g%.2s%c",
               chip_names[i] + 1, chip_names[i][3] - 'A' + 'a');
      CHECK_EQ(agx_chip_from_compatible(compatible), c);
   }
}

int
main(void)
{
   test_truth_table();
   test_chip_sets();
   test_recovered_table();
   test_decoder_identity();
   test_variant_is_load_bearing();
   test_params_contracts();
   test_invalid_identity();
   test_classification_invariants();
   test_compatible();

   if (failures) {
      fprintf(stderr, "agx_chip: %u of %u checks failed\n", failures, checks);
      return 1;
   }

   printf("agx_chip: all %u checks passed\n", checks);
   return 0;
}
