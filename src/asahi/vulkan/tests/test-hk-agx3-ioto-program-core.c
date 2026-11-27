/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"

#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_ioto_profile_core.h"
#include "asahi/lib/agx_ioto_state_core.h"
#include "asahi/lib/agx_tilebuffer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static int
test_g17p_constant_program_binary_extent(void)
{
   uint8_t binary[512] = {0};
   /* A valid inline stream, followed by bytes outside its exact extent. */
   memcpy(binary + 32, (uint8_t[]){0x0c, 0x10, 0x0e, 0, 0, 0}, 6);
   struct agx_ioto_constant_program cp;
   CHECK(hk_agx3_g17p_constant_program_from_binary(
      binary, sizeof(binary), 32, 6, 0x90000, 4096, &cp));
   CHECK(cp.code == binary + 32 && cp.code_size == 6);
   CHECK(cp.code_va == 0x90020 && cp.absolute_branch);
   CHECK(!cp.force_external && !cp.request_prelude && !cp.suppress_prelude);

   const struct {
      size_t offset, size, allocation_size;
      uint64_t va;
   } invalid[] = {
      {32, 0, 4096, 0x90000},
      {33, 6, 4096, 0x90000},
      {32, 5, 4096, 0x90000},
      {514, 2, 4096, 0x90000},
      {510, 4, 4096, 0x90000},
      {32, SIZE_MAX - 1, 4096, 0x90000},
      {SIZE_MAX - 1, 2, 4096, 0x90000},
      {32, 6, 511, 0x90000},
      {32, 6, 4096, 0},
      {32, 6, 4096, 0x90001},
      {32, 6, 4096, UINT64_C(1) << 48},
      {32, 6, 4096, (UINT64_C(1) << 48) - 510},
      {32, 6, 4096, UINT64_MAX - 1},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(invalid); ++i) {
      memset(&cp, 0xa5, sizeof(cp));
      struct agx_ioto_constant_program before;
      memcpy(&before, &cp, sizeof(cp));
      CHECK(!hk_agx3_g17p_constant_program_from_binary(
         binary, sizeof(binary), invalid[i].offset, invalid[i].size,
         invalid[i].va, invalid[i].allocation_size, &cp));
      CHECK(memcmp(&before, &cp, sizeof(cp)) == 0);
   }
   CHECK(!hk_agx3_g17p_constant_program_from_binary(
      NULL, sizeof(binary), 32, 6, 0x90000, 4096, &cp));
   CHECK(!hk_agx3_g17p_constant_program_from_binary(
      binary, sizeof(binary), 32, 6, 0x90000, 4096, NULL));

   /* Inline and external suffixes are both independent of setup placement.
    * Exercise the exact final byte of the 48-bit executable address space. */
   for (unsigned external = 0; external < 2; ++external) {
      CHECK(hk_agx3_g17p_constant_program_from_binary(
         binary, sizeof(binary), 32, external ? 480 : 6,
         (UINT64_C(1) << 48) - sizeof(binary), sizeof(binary), &cp));
      struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = 64, .shader_va = 0x100000003c0,
         .shader_binary_size = 128, .nr_gprs = 16,
         .fragment = true,
         .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
         .constant_program = cp,
      };
      struct hk_agx3_g17p_graphics_program first, moved;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &desc, &first));
      desc.program_va = 0x800000000000;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &desc, &moved));
      CHECK(first.length == moved.length);
      CHECK(memcmp(first.bytes, moved.bytes, first.length) == 0);
   }
   return 0;
}

static int
test_g17p_graphics_constant_program(void)
{
   /* The bridge must carry the compiler's exact suffix through both
    * generation and cached relocation, not silently retain an empty END. */
   static const uint8_t code[] = {0x0c, 0x10, 0x0e, 0, 0, 0};
   struct hk_agx3_g17p_graphics_program_desc desc = {
      .program_va = 0x10000, .shader_va = 0x100000003c0,
      .shader_binary_size = 128, .nr_gprs = 16,
      .fragment = true,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
      .constant_program = {
         .code = code, .code_size = sizeof(code), .code_va = 0x90000,
      },
   };
   struct hk_agx3_g17p_graphics_program program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &desc, &program));
   const size_t offset = program.length - sizeof(code);
   CHECK(offset == AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE - 4);
   CHECK(memcmp(program.bytes + offset, code, sizeof(code)) == 0);

   desc.constant_program.force_external = true;
   desc.constant_program.request_prelude = true;
   struct agx3_g17p_graphics_program_template cached;
   CHECK(hk_agx3_g17p_graphics_program_template_generate(
      AGX_CHIP_G17P, &desc, &cached));
   struct hk_agx3_g17p_graphics_program_desc moved = desc;
   moved.program_va = 0xa0000;
   CHECK(hk_agx3_g17p_graphics_program_template_relocate(
      AGX_CHIP_G17P, &desc, &moved, &cached, program.bytes,
      sizeof(program.bytes), &program.length));
   CHECK(program.length == offset + 12);
   uint64_t displacement = 0;
   for (unsigned byte = 0; byte < 6; ++byte)
      displacement |= (uint64_t)program.bytes[offset + 2 + 3 + byte]
                      << (8 * byte);
   CHECK(((moved.program_va + offset + 2 + displacement) &
          ((UINT64_C(1) << 48) - 1)) == desc.constant_program.code_va);
   return 0;
}

static int
test_g17p_graphics_root_requirement(void)
{
   CHECK(!hk_agx3_g17p_graphics_needs_root(false, false, 0));
   CHECK(hk_agx3_g17p_graphics_needs_root(true, false, 0));
   CHECK(hk_agx3_g17p_graphics_needs_root(false, true, 0));
   CHECK(hk_agx3_g17p_graphics_needs_root(false, false, 1));

   /* Set pointers after a root pointer at half 12: set k at 16 + 4k. */
   uint64_t halfs[8] = {0};
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 12, 2) == 0);
   CHECK(!hk_agx3_g17p_graphics_needs_root(false, false, 0));
   /* Unknown reads, or no sets: the layout's count. */
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, false, 12, 2) == 2);
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 12, 0) == 0);
   /* The root pointer itself (halves 12..15) is not a set pointer. */
   halfs[0] = BITFIELD64_RANGE(12, 4);
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 12, 2) == 0);
   /* Any half of any set pointer keeps all of them. */
   for (unsigned h = 16; h < 24; ++h) {
      halfs[0] = BITFIELD64_BIT(h);
      CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 12, 2) == 2);
   }
   halfs[0] = BITFIELD64_BIT(24);
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 12, 2) == 0);
   /* Across a word boundary: root at 56, sets at 60..67. */
   halfs[0] = 0;
   halfs[1] = BITFIELD64_BIT(65 - 64);
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 56, 2) == 2);
   halfs[1] = 0;
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 8, true, 56, 2) == 0);
   /* A window past the recorded halves is not known to be dead. */
   CHECK(hk_agx3_g17p_graphics_live_set_count(halfs, 1, true, 56, 2) == 2);
   return 0;
}

static int
test_g17p_precompiled_data_layout(void)
{
   uint32_t upload = 0;
   uint16_t halfs = 0;

   CHECK(hk_agx3_g17p_precompiled_data_layout(1, &upload, &halfs));
   CHECK(upload == 4 && halfs == 2);
   CHECK(hk_agx3_g17p_precompiled_data_layout(4, &upload, &halfs));
   CHECK(upload == 4 && halfs == 2);
   CHECK(hk_agx3_g17p_precompiled_data_layout(50, &upload, &halfs));
   CHECK(upload == 52 && halfs == 26);
   CHECK(hk_agx3_g17p_precompiled_data_layout(
      (UINT16_MAX - 1u) * 2u, &upload, &halfs));
   CHECK(upload == (UINT16_MAX - 1u) * 2u && halfs == UINT16_MAX - 1u);

   upload = 17;
   halfs = 19;
   CHECK(!hk_agx3_g17p_precompiled_data_layout(0, &upload, &halfs));
   CHECK(upload == 17 && halfs == 19);
   CHECK(!hk_agx3_g17p_precompiled_data_layout(
      (UINT16_MAX - 1u) * 2u + 1u, &upload, &halfs));
   CHECK(upload == 17 && halfs == 19);
   CHECK(!hk_agx3_g17p_precompiled_data_layout(4, NULL, &halfs));
   CHECK(!hk_agx3_g17p_precompiled_data_layout(4, &upload, NULL));
   return 0;
}

static const enum agx_chip targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D,
};

static bool
is_gpu1(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C || chip == AGX_CHIP_G17D;
}

static struct hk_agx3_ioto_program_desc
make_desc(enum agx_chip chip)
{
   static const uint8_t body[] = {0x06, 0x00};
   struct hk_agx3_ioto_program_desc desc = {
      .program_va = UINT64_C(0x123456789a80),
      .body = body,
      .body_size = sizeof(body),
      .tail.type = AGX_IOTO_STATE_TAIL_TYPE0,
      .profile_payload0 = is_gpu1(chip) ? UINT16_C(0x8098)
                                       : UINT16_C(0x0555),
      .profile_payload1 = is_gpu1(chip) ? UINT16_C(0xa400) : 0,
   };

   desc.tail.ldshdr.f[0] = UINT64_C(0x123456789ac0);
   if (is_gpu1(chip)) {
      desc.tail.speclm.f[0] = 3;
      desc.tail.speclm.f[1] = 165;
      desc.tail.ldshdr.f[5] = 3;
      desc.tail.ldshdr.f[6] = 107;
      desc.tail.spectpr.f[0] = 3;
      desc.tail.spectpr.f[1] = 166;
   } else {
      desc.tail.ldshdr.f[23] = 1;
   }

   return desc;
}

static int
test_exact_target_order(void)
{
   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      const enum agx_chip chip = targets[i];
      const struct hk_agx3_ioto_program_desc desc = make_desc(chip);
      const struct agx_ioto_profile_control profile = {
         .chip = chip,
         .payload0 = desc.profile_payload0,
         .payload1 = desc.profile_payload1,
      };
      uint8_t packed_profile[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE];
      struct agx_ioto_state_tail_output packed_tail;
      CHECK(agx_ioto_profile_control_pack(&profile, packed_profile));
      CHECK(agx_ioto_state_tail_pack(chip, &desc.tail, &packed_tail));

      const size_t profile_size = agx_ioto_profile_control_size(chip);
      const size_t total = desc.body_size + profile_size + packed_tail.length +
                           AGX_IOTO_PROGRAM_TERMINATE_SIZE;
      uint8_t output[64];
      memset(output, 0xa5, sizeof(output));
      size_t written = SIZE_MAX;
      CHECK(hk_agx3_ioto_program_generate(chip, &desc, output, total,
                                          &written));
      CHECK(written == total);
      CHECK(memcmp(output, desc.body, desc.body_size) == 0);
      CHECK(memcmp(output + desc.body_size, packed_profile, profile_size) ==
            0);
      CHECK(memcmp(output + desc.body_size + profile_size, packed_tail.bytes,
                   packed_tail.length) == 0);
      static const uint8_t terminate[] = {0x0e, 0x00, 0x00, 0x00};
      CHECK(memcmp(output + total - sizeof(terminate), terminate,
                   sizeof(terminate)) == 0);
      CHECK(output[total] == 0xa5);
   }

   return 0;
}

static int
test_terminal_suffix_modes(void)
{
   static const uint8_t inline_code[] = {0x1b, 0x0c, 0x06, 0x00};

   for (size_t i = 0; i < ARRAY_SIZE(targets); ++i) {
      const enum agx_chip chip = targets[i];
      const struct hk_agx3_ioto_program_desc desc = make_desc(chip);
      struct agx_ioto_state_tail_output packed_tail;
      CHECK(agx_ioto_state_tail_pack(chip, &desc.tail, &packed_tail));
      const size_t prefix_size = desc.body_size +
                                 agx_ioto_profile_control_size(chip) +
                                 packed_tail.length;

      for (unsigned mode = 0; mode < 3; ++mode) {
         struct agx_ioto_preshader_desc preshader = {
            .external_va = AGX_IOTO_PRESHADER_NO_VA,
         };
         if (mode == 1) {
            preshader.code = inline_code;
            preshader.code_size = sizeof(inline_code);
         } else if (mode == 2) {
            preshader.force_external = true;
            preshader.request_prelude = true;
            preshader.external_va = UINT64_C(0x123456789abc);
         }

         struct agx_ioto_preshader_output suffix;
         CHECK(agx_ioto_preshader_emit(chip, &preshader, &suffix));
         const size_t total = prefix_size + suffix.length;
         uint8_t output[96];
         memset(output, 0xa5, sizeof(output));
         size_t written = SIZE_MAX;
         CHECK(hk_agx3_ioto_program_generate_no_pixrel(
            chip, &desc, &preshader, output, total, &written));
         CHECK(written == total);
         CHECK(memcmp(output + prefix_size, suffix.bytes, suffix.length) == 0);
         CHECK(output[total] == 0xa5);
         CHECK(suffix.mode == (mode == 0 ? AGX_IOTO_PRESHADER_MODE_ABSENT
                                        : mode == 1
                                             ? AGX_IOTO_PRESHADER_MODE_INLINE
                                             : AGX_IOTO_PRESHADER_MODE_EXTERNAL));
         if (mode != 0) {
            static const uint8_t absent[] = {0x0e, 0x00, 0x00, 0x00};
            CHECK(suffix.length < sizeof(absent) ||
                  memcmp(output + total - sizeof(absent), absent,
                         sizeof(absent)) != 0);
         }

         uint8_t failed[96];
         uint8_t before[sizeof(failed)];
         memset(failed, 0x5a, sizeof(failed));
         memcpy(before, failed, sizeof(failed));
         size_t failed_written = 29;
         CHECK(!hk_agx3_ioto_program_generate_no_pixrel(
            chip, &desc, &preshader, failed, total - 1, &failed_written));
         CHECK(memcmp(failed, before, sizeof(failed)) == 0);
         CHECK(failed_written == 29);
      }
   }

   return 0;
}

static int
expect_failure_unchanged(enum agx_chip chip,
                         const struct hk_agx3_ioto_program_desc *desc,
                         size_t capacity)
{
   uint8_t output[96];
   uint8_t before[sizeof(output)];
   memset(output, 0x5a, sizeof(output));
   memcpy(before, output, sizeof(output));
   size_t written = SIZE_MAX - 7;
   const size_t before_written = written;

   CHECK(!hk_agx3_ioto_program_generate(chip, desc, output, capacity,
                                        &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == before_written);
   return 0;
}

static int
test_fail_closed(void)
{
   static const enum agx_chip rejected[] = {
      AGX_CHIP_G13G, AGX_CHIP_G14X, AGX_CHIP_G15P,
      AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   };
   struct hk_agx3_ioto_program_desc desc = make_desc(AGX_CHIP_G15G);
   for (size_t i = 0; i < ARRAY_SIZE(rejected); ++i)
      CHECK(expect_failure_unchanged(rejected[i], &desc, 96) == 0);

   static const uint8_t branch[] = {
      0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   desc.body = branch;
   desc.body_size = sizeof(branch);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 96) == 0);

   desc = make_desc(AGX_CHIP_G15G);
   desc.program_va |= 1;
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 96) == 0);
   desc = make_desc(AGX_CHIP_G15G);
   desc.profile_payload0 = UINT16_C(0x0800);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 96) == 0);
   desc = make_desc(AGX_CHIP_G15G);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, &desc, 1) == 0);
   CHECK(expect_failure_unchanged(AGX_CHIP_G15G, NULL, 96) == 0);

   uint8_t output[96];
   size_t written = 9;
   CHECK(!hk_agx3_ioto_program_generate(AGX_CHIP_G15G, &desc, NULL,
                                         sizeof(output), &written));
   CHECK(written == 9);
   CHECK(!hk_agx3_ioto_program_generate(AGX_CHIP_G15G, &desc, output,
                                         sizeof(output), NULL));

   memset(output, 0x5a, sizeof(output));
   uint8_t before[sizeof(output)];
   memcpy(before, output, sizeof(output));
   written = 11;
   CHECK(!hk_agx3_ioto_program_generate_no_pixrel(
      AGX_CHIP_G15G, &desc, NULL, output, sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == 11);
   return 0;
}

static int
expect_g17s_minimal_hk_failure(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   size_t capacity)
{
   uint8_t output[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   uint8_t before[sizeof(output)];
   memset(output, 0x5a, sizeof(output));
   memcpy(before, output, sizeof(output));
   size_t written = 47;
   struct agx3_compute_program program;
   memset(&program, 0xa5, sizeof(program));
   uint8_t program_before[sizeof(program)];
   memcpy(program_before, &program, sizeof(program));

   CHECK(!hk_agx3_g17s_minimal_compute_program_prepare(
      chip, desc, output, capacity, &written, &program));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == 47);
   CHECK(memcmp(&program, program_before, sizeof(program)) == 0);
   return 0;
}

static int
test_g17s_minimal_compute_admission(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &sources));
   const struct agx3_g17s_minimal_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000040),
      .shader_va = UINT64_C(0x100000006c0),
      .root_uniform = {
         .user_buffer_binding = 1,
         .byte_offset = 0,
         .uniform_data_destination = 4,
         .uniform_data_count = 1,
         .temporary_base = 0,
      },
      .launch_sources = sources,
   };
   uint8_t direct[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   uint8_t bridged[sizeof(direct)];
   memset(direct, 0xa5, sizeof(direct));
   memset(bridged, 0x5a, sizeof(bridged));
   size_t direct_written = SIZE_MAX, bridged_written = SIZE_MAX;
   struct agx3_compute_program direct_program, bridged_program;

   CHECK(agx3_g17s_minimal_compute_program_prepare(
      &desc, direct, sizeof(direct), &direct_written, &direct_program));
   CHECK(hk_agx3_g17s_minimal_compute_program_prepare(
      AGX_CHIP_G17S, &desc, bridged, sizeof(bridged), &bridged_written,
      &bridged_program));
   CHECK(bridged_written == AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE);
   CHECK(bridged_written == direct_written);
   CHECK(memcmp(bridged, direct, bridged_written) == 0);
   CHECK(bridged[bridged_written] == 0x5a);
   CHECK(bridged_program.chip == AGX_CHIP_G17S);
   CHECK(bridged_program.program_va == desc.program_va);
   CHECK(memcmp(&bridged_program.launch_template,
                &direct_program.launch_template,
                sizeof(bridged_program.launch_template)) == 0);

   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t mapped[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   memset(mapped, 0x3c, sizeof(mapped));
   struct agx3_compute_program_upload upload = {
      .cpu = mapped,
      .capacity = sizeof(mapped),
      .gpu_va = desc.program_va,
   };
   size_t mapped_written = SIZE_MAX;
   struct agx3_compute_program mapped_program;
   CHECK(hk_agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &upload, &mapped_written, &mapped_program));
   CHECK(mapped_written == direct_written);
   CHECK(memcmp(mapped, direct, mapped_written) == 0);
   CHECK(mapped[mapped_written] == 0x3c);
   CHECK(mapped_program.chip == AGX_CHIP_G17S);
   CHECK(mapped_program.program_va == upload.gpu_va);

   const struct agx_ioto_g17s_minimal_compute_resources resources = {
      .gather_buffer_va = UINT64_C(0x100000e1ba0),
      .user_compute_argument_gpu_va_1 = UINT64_C(0x10000030100),
      .absolute_uniform_data_va =
         AGX_IOTO_G17S_MINIMAL_COMPUTE_WITNESS_ABSOLUTE_VA,
   };
   _Alignas(AGX_IOTO_PROGRAM_ALIGNMENT)
      uint8_t live_mapped[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   memset(live_mapped, 0x6d, sizeof(live_mapped));
   const struct agx3_compute_program_upload live_upload = {
      .cpu = live_mapped,
      .capacity = sizeof(live_mapped),
      .gpu_va = desc.program_va,
   };
   size_t live_written = SIZE_MAX;
   struct agx3_compute_program live_program;
   CHECK(hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
      AGX_CHIP_G17S, &desc, &resources, &live_upload, &live_written,
      &live_program));
   CHECK(live_written == direct_written);
   CHECK(memcmp(live_mapped + 32, direct + 32, live_written - 32) == 0);
   CHECK(memcmp(live_mapped, direct, 32) != 0);
   CHECK(live_mapped[live_written] == 0x6d);
   CHECK(live_program.chip == AGX_CHIP_G17S &&
         live_program.program_va == live_upload.gpu_va);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS,
      .size = {2, 3, 4},
      .local_size = {8, 4, 2},
      .dimensions = 3,
   };
   uint8_t expected_launch[AGX3_CDM_DIRECT_LAUNCH_SIZE];
   CHECK(agx3_compute_program_pack_direct_launch(&live_program, &grid,
                                                  expected_launch));
   uint8_t dispatch[AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE + 3];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink dispatch_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 2,
   };
   CHECK(hk_agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &dispatch_sink));
   CHECK(dispatch_sink.length ==
         2 + AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE);
   CHECK(dispatch[0] == 0xb6 && dispatch[1] == 0xb6 &&
         dispatch[dispatch_sink.length] == 0xb6);
   CHECK(memcmp(dispatch + 2, expected_launch,
                AGX3_CDM_DIRECT_LAUNCH_SIZE) == 0);
   static const uint8_t strong_barrier[] = {0x60, 0x15, 0x00, 0x60};
   CHECK(memcmp(dispatch + 2 + AGX3_CDM_DIRECT_LAUNCH_SIZE, strong_barrier,
                sizeof(strong_barrier)) == 0);

   uint8_t dispatch_before[sizeof(dispatch)];
   memcpy(dispatch_before, dispatch, sizeof(dispatch));
   struct agx3_compute_dispatch_sink failed_sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
      .length = 1,
   };
   CHECK(!hk_agx3_g17s_minimal_compute_dispatch_append(
      &direct_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   struct agx3_compute_program sibling_program = live_program;
   sibling_program.chip = AGX_CHIP_G17G;
   CHECK(!hk_agx3_g17s_minimal_compute_dispatch_append(
      &sibling_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);
   failed_sink.capacity = AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE;
   CHECK(!hk_agx3_g17s_minimal_compute_dispatch_append(
      &live_program, &grid, &failed_sink));
   CHECK(failed_sink.length == 1 &&
         memcmp(dispatch, dispatch_before, sizeof(dispatch)) == 0);

   uint8_t live_before[sizeof(live_mapped)];
   memcpy(live_before, live_mapped, sizeof(live_mapped));
   const struct agx3_compute_program live_program_before = live_program;

   uint8_t mapped_before[sizeof(mapped)];
   memcpy(mapped_before, mapped, sizeof(mapped));
   const struct agx3_compute_program mapped_program_before = mapped_program;

   static const enum agx_chip rejected[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
      AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
      AGX_CHIP_G17G, AGX_CHIP_G17C, AGX_CHIP_G17D, AGX_CHIP_UNKNOWN,
   };
   for (size_t i = 0; i < ARRAY_SIZE(rejected); ++i) {
      CHECK(expect_g17s_minimal_hk_failure(rejected[i], &desc,
                                            sizeof(bridged)) == 0);
      mapped_written = 47;
      CHECK(!hk_agx3_g17s_minimal_compute_program_upload_prepare(
         rejected[i], &desc, &upload, &mapped_written, &mapped_program));
      CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
            mapped_written == 47);
      CHECK(memcmp(&mapped_program, &mapped_program_before,
                   sizeof(mapped_program)) == 0);
      live_written = 47;
      CHECK(!hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
         rejected[i], &desc, &resources, &live_upload, &live_written,
         &live_program));
      CHECK(memcmp(live_mapped, live_before, sizeof(live_mapped)) == 0 &&
            live_written == 47);
      CHECK(memcmp(&live_program, &live_program_before,
                   sizeof(live_program)) == 0);
   }

   struct agx3_compute_program_upload invalid_upload = upload;
   invalid_upload.gpu_va += AGX_IOTO_PROGRAM_ALIGNMENT;
   mapped_written = 47;
   CHECK(!hk_agx3_g17s_minimal_compute_program_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_upload, &mapped_written,
      &mapped_program));
   CHECK(memcmp(mapped, mapped_before, sizeof(mapped)) == 0 &&
         mapped_written == 47);
   CHECK(memcmp(&mapped_program, &mapped_program_before,
                sizeof(mapped_program)) == 0);

   struct agx_ioto_g17s_minimal_compute_resources invalid_resources =
      resources;
   invalid_resources.absolute_uniform_data_va = 0;
   live_written = 47;
   CHECK(!hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
      AGX_CHIP_G17S, &desc, &invalid_resources, &live_upload, &live_written,
      &live_program));
   CHECK(memcmp(live_mapped, live_before, sizeof(live_mapped)) == 0 &&
         live_written == 47);
   CHECK(memcmp(&live_program, &live_program_before,
                sizeof(live_program)) == 0);

   struct agx3_g17s_minimal_compute_program_desc invalid = desc;
   invalid.root_uniform.user_buffer_binding = 0;
   CHECK(expect_g17s_minimal_hk_failure(AGX_CHIP_G17S, &invalid,
                                         sizeof(bridged)) == 0);
   CHECK(expect_g17s_minimal_hk_failure(
            AGX_CHIP_G17S, &desc,
            AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE - 1) == 0);
   CHECK(expect_g17s_minimal_hk_failure(AGX_CHIP_G17S, NULL,
                                         sizeof(bridged)) == 0);
   return 0;
}

static int
test_g17p_dynamic_compute_program(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(true, false, false,
                                                   &sources));
   const struct agx3_ioto_uniform_range ranges[] = {
      {
         .start_halfs = 0,
         .size_halfs = 12,
         .address = UINT64_C(0x10000024000),
      },
      {
         .start_halfs = 12,
         .size_halfs = 8,
         .address = UINT64_C(0x10000030000),
      },
   };
   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000b0000),
      .shader_va = UINT64_C(0x10000000100),
      .uniform_ranges = ranges,
      .uniform_range_count = ARRAY_SIZE(ranges),
      .shader_binary_size = 96,
      .nr_gprs = 8,
      .compiler_d_valid = true,
      .compiler_d = 32,
      .threadgroup_memory_size_B = 4096,
      .launch_sources = sources,
   };
   struct hk_agx3_g17p_compute_program program;
   memset(&program, 0xa5, sizeof(program));
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc,
                                               &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.length < sizeof(program.bytes));
   CHECK(program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
         0x0e);
   CHECK(program.launch.chip == AGX_CHIP_G17P);
   CHECK(program.launch.program_va == desc.program_va);
   CHECK(program.launch.shader_va == desc.shader_va);
   CHECK(program.launch.g17p_native_compute_live_bound);

   struct agx_ioto_state_tail_output expected_tail;
   /* The direct-dispatch kind. This expectation used to say GRID_HELPER,
    * because it was written against the size-derived f1 that this shader's
    * 118-byte length happened to land on -- a fourth self-pinned expectation
    * rather than a witness. LDSHDR f1 is a program-kind selector (permuted
    * through agx_ioto_e4), so a direct dispatch announces the direct kind. */
   CHECK(agx_ioto_g17p_runtime_compute_tail_build_shared(
      desc.shader_va, AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      desc.threadgroup_memory_size_B, &expected_tail));
   /* The runtime builder leaves SPECTPR zero (the hand-written add3 witness's
    * footprint). A compiler-produced shader must declare its own, exactly as
    * the graphics path does, so the expectation carries it too. */
   struct agx_ioto_state expected_spectpr;
   CHECK(agx_ioto_g17p_compiler_spectpr_build(desc.scratch_size,
                                              desc.compiler_d,
                                              &expected_spectpr));
   CHECK(agx_ioto_state_pack_spectpr(
      AGX_CHIP_G17P, &expected_spectpr,
      expected_tail.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
      AGX_IOTO_SPECTPR_SIZE));
   size_t tail_matches = 0;
   for (size_t i = 0; i + expected_tail.length <= program.length; ++i) {
      if (memcmp(program.bytes + i, expected_tail.bytes,
                 expected_tail.length) == 0)
         tail_matches++;
   }
   CHECK(tail_matches == 1);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {96, 4, 1},
      .local_size = {32, 2, 1},
      .dimensions = 2,
   };
   static const uint8_t expected_dispatch[] = {
      0x00, 0x00, 0x08, 0x00, 0x40, 0x00, 0x00, 0x01,
      0x00, 0x2c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x60, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x60, 0x01, 0x00, 0x60,
   };
   uint8_t dispatch[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + 1];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   CHECK(hk_agx3_g17p_native_compute_dispatch_append(
      &program.launch, &grid, &sink));
   CHECK(sink.length == sizeof(expected_dispatch));
   CHECK(memcmp(dispatch, expected_dispatch, sizeof(expected_dispatch)) == 0);
   CHECK(dispatch[sink.length] == 0xb6);

   uint8_t strong_dispatch[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + 1];
   memset(strong_dispatch, 0xb6, sizeof(strong_dispatch));
   struct agx3_compute_dispatch_sink strong_sink = {
      .data = strong_dispatch,
      .capacity = sizeof(strong_dispatch),
   };
   CHECK(hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
      &program.launch, &grid, 0x5, &strong_sink));
   CHECK(strong_sink.length == sizeof(expected_dispatch));
   CHECK(memcmp(strong_dispatch, expected_dispatch,
                AGX3_CDM_DIRECT_LAUNCH_SIZE) == 0);
   static const uint8_t strong_barrier[] = {0x60, 0x15, 0x00, 0x60};
   CHECK(memcmp(strong_dispatch + AGX3_CDM_DIRECT_LAUNCH_SIZE,
                strong_barrier, sizeof(strong_barrier)) == 0);
   CHECK(strong_dispatch[strong_sink.length] == 0xb6);

   const struct hk_agx3_g17p_compute_program before = program;
   struct hk_agx3_g17p_compute_program_desc scratch_desc = desc;
   scratch_desc.scratch_size = 16;
   struct hk_agx3_g17p_compute_program scratch_program;
   CHECK(hk_agx3_g17p_compute_program_generate(
      AGX_CHIP_G17P, &scratch_desc, &scratch_program));
   CHECK(scratch_program.length == before.length);

   struct agx_ioto_state scratch_spectpr;
   CHECK(agx_ioto_g17p_compiler_spectpr_build(
      scratch_desc.scratch_size, scratch_desc.compiler_d, &scratch_spectpr));
   struct agx_ioto_state_tail_output scratch_tail = expected_tail;
   CHECK(agx_ioto_state_pack_spectpr(
      AGX_CHIP_G17P, &scratch_spectpr,
      scratch_tail.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
      AGX_IOTO_SPECTPR_SIZE));
   size_t scratch_tail_matches = 0;
   for (size_t i = 0; i + scratch_tail.length <= scratch_program.length; ++i) {
      if (memcmp(scratch_program.bytes + i, scratch_tail.bytes,
                 scratch_tail.length) == 0)
         scratch_tail_matches++;
   }
   CHECK(scratch_tail_matches == 1);

   size_t scratch_differences = 0;
   for (size_t i = 0; i < before.length; ++i) {
      if (scratch_program.bytes[i] != before.bytes[i])
         scratch_differences++;
   }
   CHECK(scratch_differences > 0);
   CHECK(scratch_differences <= AGX_IOTO_SPECTPR_SIZE);

   /* A register spill frame publishes its bytes the same way. */
   struct hk_agx3_g17p_compute_program_desc spill = desc;
   spill.scratch_size = 56;
   struct hk_agx3_g17p_compute_program spill_program;
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &spill,
                                               &spill_program));
   struct agx_ioto_state spill_spectpr;
   CHECK(agx_ioto_g17p_compiler_spectpr_build(56, desc.compiler_d,
                                              &spill_spectpr));
   CHECK(spill_spectpr.f[20] == 14 && spill_spectpr.f[21] == 14);

   /* A frame must be whole quads and within the published bound. */
   struct hk_agx3_g17p_compute_program_desc invalid = desc;
   invalid.scratch_size = 18;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid.scratch_size = AGX3_G17P_MAX_COMPUTE_SPILL_B + 4;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.compiler_d_valid = false;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.uniform_ranges = NULL;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.threadgroup_memory_size_B = 257;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17S, &desc,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);

   /* The two records of a native indirect pair announce program kinds that a
    * direct dispatch never uses, so the generator has to be able to select
    * them. Each selector must land its own witnessed LoadShader tail in the
    * generated program, exactly once, and must change nothing else: the three
    * programs differ only inside that record. */
   static const enum agx_ioto_g17p_compute_ldshdr_f1 kinds[] = {
      AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1,
      AGX3_G17P_GRID_HELPER_LDSHDR_F1,
   };
   struct hk_agx3_g17p_compute_program kind_programs[ARRAY_SIZE(kinds)];
   for (unsigned k = 0; k < ARRAY_SIZE(kinds); ++k) {
      struct hk_agx3_g17p_compute_program_desc kind_desc = desc;
      kind_desc.ldshdr_f1 = kinds[k];
      memset(&kind_programs[k], 0xa5, sizeof(kind_programs[k]));
      CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &kind_desc,
                                                  &kind_programs[k]));
      CHECK(kind_programs[k].length == before.length);

      struct agx_ioto_state_tail_output kind_tail;
      CHECK(agx_ioto_g17p_runtime_compute_tail_build_shared(
         desc.shader_va, kinds[k], desc.threadgroup_memory_size_B,
         &kind_tail));
      CHECK(agx_ioto_state_pack_spectpr(
         AGX_CHIP_G17P, &expected_spectpr,
         kind_tail.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
         AGX_IOTO_SPECTPR_SIZE));

      size_t kind_matches = 0;
      for (size_t i = 0; i + kind_tail.length <= kind_programs[k].length; ++i) {
         if (memcmp(kind_programs[k].bytes + i, kind_tail.bytes,
                    kind_tail.length) == 0)
            kind_matches++;
      }
      CHECK(kind_matches == 1);
   }

   /* The default selector is the direct kind, so every existing caller keeps
    * the exact program it had before this field existed. */
   CHECK(memcmp(&kind_programs[0], &before, sizeof(before)) == 0);

   /* Different kinds are different programs, and they differ only in the
    * LoadShader record: the byte-difference count cannot exceed one record. */
   for (unsigned k = 1; k < ARRAY_SIZE(kinds); ++k) {
      size_t differing = 0;
      for (size_t i = 0; i < before.length; ++i) {
         if (kind_programs[k].bytes[i] != before.bytes[i])
            differing++;
      }
      CHECK(differing > 0);
      CHECK(differing <= AGX_IOTO_LDSHDR_SIZE);
   }

   /* An unwitnessed program kind fails the compose rather than inventing an
    * encoding for it. */
   invalid = desc;
   invalid.ldshdr_f1 = (enum agx_ioto_g17p_compute_ldshdr_f1)2;
   memset(&program, 0xa5, sizeof(program));
   const struct hk_agx3_g17p_compute_program poisoned = program;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &poisoned, sizeof(program)) == 0);
   return 0;
}

static int
test_g17p_basic_graphics_program(void)
{
   static const uint8_t expected_fragment[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01, 0x2a, 0x01,
      0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x14, 0x81, 0x11, 0x06, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x80, 0x02, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8,
      0x10, 0x05, 0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x12,
      0x54, 0x00, 0x4c, 0x00, 0x07, 0x02, 0x54, 0xf3, 0x02, 0x00, 0x0e, 0x00,
      0x00, 0x00,
   };
   CHECK(sizeof(expected_fragment) ==
         AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE);
   /* Both the valid cold G1a vary0 tail and the matched applegpu_g17p
    * constant-colour/no-varying probe leave compiler ProgramVariant field 38
    * absent. FragmentProgramVariantESLState therefore supplies D=0, making
    * the complete SPECTPR payload above zero. */
   const struct hk_agx3_g17p_graphics_program_desc fragment = {
      .program_va = UINT64_C(0x10001990240),
      .shader_va = UINT64_C(0x10000000080),
      .shader_binary_size = 54,
      .scratch_size = 0,
      .nr_gprs = 8,
      .fragment = true,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct hk_agx3_g17p_graphics_program program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &fragment,
                                                      &program));
   CHECK(program.length == sizeof(expected_fragment));
   CHECK(memcmp(program.bytes, expected_fragment, sizeof(expected_fragment)) ==
         0);

   struct hk_agx3_g17p_graphics_program_desc scratch_fragment = fragment;
   scratch_fragment.scratch_size = 16;
   struct hk_agx3_g17p_graphics_program scratch_fragment_program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &scratch_fragment, &scratch_fragment_program));
   CHECK(scratch_fragment_program.length == program.length);

   struct agx_ioto_state scratch_fragment_spectpr;
   CHECK(agx_ioto_g17p_compiler_spectpr_build(
      16, fragment.nr_gprs, &scratch_fragment_spectpr));
   uint8_t packed_scratch_fragment[AGX_IOTO_SPECTPR_SIZE];
   CHECK(agx_ioto_state_pack_spectpr(
      AGX_CHIP_G17P, &scratch_fragment_spectpr, packed_scratch_fragment,
      sizeof(packed_scratch_fragment)));
   CHECK(memcmp(scratch_fragment_program.bytes + AGX_IOTO_SPECLM_SIZE +
                   AGX_IOTO_LDSHDR_SIZE,
                packed_scratch_fragment, sizeof(packed_scratch_fragment)) ==
         0);

   size_t scratch_fragment_differences = 0;
   for (size_t i = 0; i < program.length; ++i) {
      if (scratch_fragment_program.bytes[i] != program.bytes[i])
         scratch_fragment_differences++;
   }
   CHECK(scratch_fragment_differences > 0);
   CHECK(scratch_fragment_differences <= AGX_IOTO_SPECTPR_SIZE);

   const struct hk_agx3_g17p_graphics_program_desc vertex = {
      .program_va = UINT64_C(0x10000128000),
      .shader_va = UINT64_C(0x10000000000),
      .shader_binary_size = 74,
      .scratch_size = 0,
      .nr_gprs = 14,
      .fragment = false,
   };
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &vertex,
                                                      &program));
   CHECK(program.length == AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   /* The 74-byte shader spans two 64-byte fetch lines, matching native f1=1
    * and encoded LDSHDR word 4 = 0x0000. */
   static const uint8_t vertex_tail[] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0xf7, 0x00,
      0x2a, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes, vertex_tail, sizeof(vertex_tail)) == 0);

   /* The exact Apple 88-byte float2 VS has four 32-bit staging values and
    * compiler ProgramVariant field 38 is 16. Preserve the distinct domains:
    * metadata D=16 yields SPECTPR f22=4, whereas using
    * physical GPR halfs=8 here would incorrectly yield f22=2.
    */
   struct hk_agx3_g17p_graphics_program_desc apple_vertex = vertex;
   apple_vertex.shader_binary_size = 88;
   apple_vertex.nr_gprs = 8;
   apple_vertex.compiler_d_valid = true;
   apple_vertex.compiler_d = 16;
   const struct hk_agx3_g17p_graphics_program normal_vertex = program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &apple_vertex, &program));
   CHECK(program.length == normal_vertex.length);
   CHECK(memcmp(program.bytes, normal_vertex.bytes, program.length) == 0);
   apple_vertex.nr_gprs = 4;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &apple_vertex, &program));
   CHECK(memcmp(program.bytes, normal_vertex.bytes, program.length) == 0);
   apple_vertex.nr_gprs = 8;
   apple_vertex.compiler_d_valid = false;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &apple_vertex, &program));
   static const uint8_t old_d_from_gprs[] = {
      0xf7, 0x00, 0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes + 18, old_d_from_gprs,
                sizeof(old_d_from_gprs)) == 0);

   static const struct agx3_ioto_uniform_range descriptor_root = {
      .start_halfs = 20,
      .size_halfs = 4,
      .address = 0x20,
   };
   static const uint8_t expected_resource_prefix[] = {
      0x0c, 0x20, 0x1c, 0x00, 0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00,
      0x59, 0x00, 0x00, 0x40, 0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00,
      0x06, 0x00, 0xab, 0x04, 0x09, 0x04, 0xbb, 0x06, 0x09, 0x04,
   };
   struct hk_agx3_g17p_graphics_program_desc resource = fragment;
   resource.uniform_ranges = &descriptor_root;
   resource.uniform_range_count = 1;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,
                                                      &resource, &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
   const size_t pre_copy_size = sizeof(expected_resource_prefix) - 8;
   static const uint8_t expected_resource_tail[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x01, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00,
      0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes, expected_resource_prefix, pre_copy_size) == 0);
   CHECK(memcmp(program.bytes + pre_copy_size,
                expected_resource_tail, sizeof(expected_resource_tail)) == 0);
   static const uint8_t fragment_termination[] = {
      0x07, 0x02, 0x54, 0xf3, 0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   CHECK(memcmp(program.bytes + program.length - sizeof(fragment_termination),
                fragment_termination, sizeof(fragment_termination)) == 0);

   static const struct agx3_ioto_uniform_range two_attribs_and_set[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 8, .size_halfs = 4, .address = 0x2000},
      {.start_halfs = 24, .size_halfs = 8, .address = 0x3000},
   };
   resource = vertex;
   resource.uniform_ranges = two_attribs_and_set;
   resource.uniform_range_count = ARRAY_SIZE(two_attribs_and_set);
   resource.shader_binary_size = 344;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,
                                                      &resource, &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
   CHECK(program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
         0x0e);

   const struct hk_agx3_g17p_graphics_program three_range_program = program;
   for (unsigned i = 0; i < ARRAY_SIZE(two_attribs_and_set); ++i) {
      struct agx3_ioto_uniform_range changed[ARRAY_SIZE(two_attribs_and_set)];
      memcpy(changed, two_attribs_and_set, sizeof(changed));
      changed[i].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == three_range_program.length &&
            memcmp(program.bytes, three_range_program.bytes,
                   program.length) != 0);
   }
   resource.uniform_ranges = two_attribs_and_set;

   /* Resource graphics programs validate their aligned placement VA but do
    * not encode it. This permits a command-buffer-local semantic cache to
    * reuse one published program at its original address. */
   struct hk_agx3_g17p_graphics_program_desc moved_resource = resource;
   moved_resource.program_va += 0x2000;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &moved_resource, &program));
   CHECK(program.length == three_range_program.length);
   CHECK(memcmp(program.bytes, three_range_program.bytes, program.length) == 0);

   /* Exact Omarchy shell fragment interface: native texture/sampler heap
    * pointers in halves0..7, root plus three descriptor-set pointers in
    * halves12..27, and the linked epilog's relocated blend constants in
    * halves28..35. All three independently addressed ranges must coexist in
    * one HAL200 Ioto program. */
   const struct agx3_ioto_uniform_range indexed_blend_root[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 12, .size_halfs = 16, .address = 0x2000},
      {.start_halfs = 28, .size_halfs = 8, .address = 0x3000},
   };
   resource = fragment;
   resource.uniform_ranges = indexed_blend_root;
   resource.uniform_range_count = ARRAY_SIZE(indexed_blend_root);
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &resource, &program));
   CHECK(program.length > AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE &&
         program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
   const struct hk_agx3_g17p_graphics_program indexed_blend_program = program;
   for (unsigned i = 0; i < ARRAY_SIZE(indexed_blend_root); ++i) {
      struct agx3_ioto_uniform_range changed[ARRAY_SIZE(indexed_blend_root)];
      memcpy(changed, indexed_blend_root, sizeof(changed));
      changed[i].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == indexed_blend_program.length &&
            memcmp(program.bytes, indexed_blend_program.bytes,
                   program.length) != 0);
   }

   /* Full attribute pointers + clamps, descriptor root, and constant pool.
    * Every range must affect actual native CProg bytes independently. */
   const struct agx3_ioto_uniform_range full_vertex[] = {
      {.start_halfs = 0, .size_halfs = 64, .address = 0x1000},
      {.start_halfs = 64, .size_halfs = 32, .address = 0x2000},
      {.start_halfs = 108, .size_halfs = 36, .address = 0x3000},
      {.start_halfs = 144, .size_halfs = 128, .address = 0x4000},
   };
   resource = vertex;
   resource.uniform_ranges = full_vertex;
   resource.uniform_range_count = ARRAY_SIZE(full_vertex);
   size_t bound = 0;
   CHECK(hk_agx3_g17p_graphics_program_bound(AGX_CHIP_G17P, full_vertex,
                                              ARRAY_SIZE(full_vertex), &bound));
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &resource, &program));
   CHECK(program.length <= bound);
   const struct hk_agx3_g17p_graphics_program full_program = program;
   unsigned copies = 0, copy_then_load = 0;
   bool saw_copy = false;
   const size_t loader_end = program.length - AGX_IOTO_G17P_DEFAULT_PROFILE_SUFFIX_SIZE -
                             AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   for (size_t at = 0; at < loader_end;) {
      struct agx_ioto_program_record record;
      size_t consumed = 0;
      CHECK(agx_ioto_decode_program_record(AGX_CHIP_G17P, program.bytes + at,
                                             loader_end - at, &record,
                                             &consumed));
      CHECK(consumed > 0);
      if (record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO &&
          (record.ioto.kind == AGX_IOTO_KIND_COPY_EP0 ||
           record.ioto.kind == AGX_IOTO_KIND_COPY_EP1)) {
         copies++;
         saw_copy = true;
      }
      if (record.kind == AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV && saw_copy)
         copy_then_load++;
      at += consumed;
   }
   fprintf(stderr, "full-interface native COPY count=%u, copies before later loads=%u\n", copies, copy_then_load);
   CHECK(copies == (64 + 32 + 36 + 128) / 2);
   CHECK(copy_then_load > 0);

   for (unsigned r = 0; r < ARRAY_SIZE(full_vertex); ++r) {
      struct agx3_ioto_uniform_range changed[ARRAY_SIZE(full_vertex)];
      memcpy(changed, full_vertex, sizeof(changed));
      changed[r].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == full_program.length);
      CHECK(memcmp(program.bytes, full_program.bytes, program.length) != 0);
   }
   resource.uniform_ranges = full_vertex;

   /* Exact Firefox indexed-vertex range pressure: native sampled-image heap,
    * attribute bases, clamps, direct indexed-draw parameters, a root with
    * three descriptor sets, and the two-half promoted constant at uniform 72.
    * These are six independently addressed CProg inputs; the generic loader
    * already bounds their load chunks and complete program size. */
   const struct agx3_ioto_uniform_range firefox_indexed_vertex[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 8, .size_halfs = 16, .address = 0x2000},
      {.start_halfs = 24, .size_halfs = 8, .address = 0x3000},
      {.start_halfs = 32, .size_halfs = 4, .address = 0x4000},
      {.start_halfs = 48, .size_halfs = 16, .address = 0x5000},
      {.start_halfs = 72, .size_halfs = 2, .address = 0x6000},
   };
   resource = vertex;
   resource.uniform_ranges = firefox_indexed_vertex;
   resource.uniform_range_count = ARRAY_SIZE(firefox_indexed_vertex);
   resource.shader_binary_size = 512;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &resource, &program));
   const struct hk_agx3_g17p_graphics_program firefox_program = program;
   for (unsigned r = 0; r < ARRAY_SIZE(firefox_indexed_vertex); ++r) {
      struct agx3_ioto_uniform_range changed[
         ARRAY_SIZE(firefox_indexed_vertex)];
      memcpy(changed, firefox_indexed_vertex, sizeof(changed));
      changed[r].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == firefox_program.length &&
            memcmp(program.bytes, firefox_program.bytes, program.length) != 0);
   }
   resource.uniform_ranges = firefox_indexed_vertex;

   /* The complete Honeykrisp vertex worst case: the six above plus a
    * gl_DrawID range. The Ioto loader bounds a program by load chunks and
    * body size, never by how many ranges produced them, so the driver budget
    * must admit all seven or a legal program is refused fail-closed. (The GL
    * driver's collector is data-dependent and needs more -- Firefox's
    * WebRender vertex shaders push nine -- so the budget is no longer exactly
    * seven; it only has to be at least this.) */
   const struct agx3_ioto_uniform_range firefox_drawid_vertex[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 8, .size_halfs = 24, .address = 0x2000},
      {.start_halfs = 32, .size_halfs = 12, .address = 0x3000},
      {.start_halfs = 44, .size_halfs = 4, .address = 0x4000},
      {.start_halfs = 48, .size_halfs = 2, .address = 0x4400},
      {.start_halfs = 56, .size_halfs = 16, .address = 0x5000},
      {.start_halfs = 72, .size_halfs = 2, .address = 0x6000},
   };
   CHECK(ARRAY_SIZE(firefox_drawid_vertex) <= HK_AGX3_G17P_GRAPHICS_MAX_RANGES);
   size_t drawid_bound = 0;
   CHECK(hk_agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, firefox_drawid_vertex,
      ARRAY_SIZE(firefox_drawid_vertex), &drawid_bound));
   CHECK(drawid_bound != 0);
   resource = vertex;
   resource.uniform_ranges = firefox_drawid_vertex;
   resource.uniform_range_count = ARRAY_SIZE(firefox_drawid_vertex);
   resource.shader_binary_size = 512;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &resource, &program));
   const struct hk_agx3_g17p_graphics_program drawid_program = program;
   for (unsigned r = 0; r < ARRAY_SIZE(firefox_drawid_vertex); ++r) {
      struct agx3_ioto_uniform_range changed[
         ARRAY_SIZE(firefox_drawid_vertex)];
      memcpy(changed, firefox_drawid_vertex, sizeof(changed));
      changed[r].address += 0x40;
      resource.uniform_ranges = changed;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &resource, &program));
      CHECK(program.length == drawid_program.length &&
            memcmp(program.bytes, drawid_program.bytes, program.length) != 0);
   }

   /* A uniform range is loaded in 32-bit dwords, so an odd size_halfs is not
    * representable and is rejected outright. The gl_DrawID range used to be
    * spelled with size_halfs = 1, which is why every gl_DrawID vertex program
    * was refused even with range budget to spare. */
   struct agx3_ioto_uniform_range odd_draw_id[
      ARRAY_SIZE(firefox_drawid_vertex)];
   memcpy(odd_draw_id, firefox_drawid_vertex, sizeof(odd_draw_id));
   odd_draw_id[4].size_halfs = 1;
   size_t odd_bound = 456;
   CHECK(!hk_agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, odd_draw_id, ARRAY_SIZE(odd_draw_id), &odd_bound));
   CHECK(odd_bound == 456);
   resource.uniform_ranges = odd_draw_id;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P,
                                                       &resource, &program));

   /* An eighth range is legal: the GL driver's WebRender vertex programs
    * push eight or nine, and the loader never counts ranges. */
   struct agx3_ioto_uniform_range eight_ranges[
      ARRAY_SIZE(firefox_drawid_vertex) + 1];
   memcpy(eight_ranges, firefox_drawid_vertex, sizeof(firefox_drawid_vertex));
   eight_ranges[ARRAY_SIZE(firefox_drawid_vertex)] =
      (struct agx3_ioto_uniform_range){
         .start_halfs = 80, .size_halfs = 2, .address = 0x7000};
   size_t eight_bound = 0;
   CHECK(hk_agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, eight_ranges, ARRAY_SIZE(eight_ranges), &eight_bound));
   CHECK(eight_bound > drawid_bound);

   /* Honeykrisp's collector and the shared composer's workspace have
    * different hard capacities. Exercise each API's actual boundary. */
   struct agx3_ioto_uniform_range over_budget[
      AGX3_G17P_GRAPHICS_MAX_RANGES + 1];
   for (unsigned i = 0; i < ARRAY_SIZE(over_budget); ++i) {
      over_budget[i] = (struct agx3_ioto_uniform_range){
         .start_halfs = 2 * i, .size_halfs = 2, .address = 0x7000 + 0x10 * i};
   }
   size_t over_bound = 789;
   CHECK(!hk_agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, over_budget, HK_AGX3_G17P_GRAPHICS_MAX_RANGES + 1,
      &over_bound));
   CHECK(over_bound == 789);

   resource = vertex;
   resource.uniform_ranges = firefox_indexed_vertex;
   resource.uniform_range_count = ARRAY_SIZE(firefox_indexed_vertex);
   resource.shader_binary_size = 512;

   size_t untouched = 123;
   CHECK(!hk_agx3_g17p_graphics_program_bound(AGX_CHIP_G17S, full_vertex,
                                               ARRAY_SIZE(full_vertex), &untouched));
   CHECK(untouched == 123);

   struct hk_agx3_g17p_graphics_program_desc boundary = vertex;
   static const struct {
      uint32_t size;
      uint8_t encoded_word4_lo;
   } proven_fetch_lines[] = {
      {1, 0x04},   {64, 0x04},  {65, 0x00},  {128, 0x00},
      {129, 0x00}, {192, 0x00}, {193, 0x02}, {256, 0x02},
      {257, 0x02}, {320, 0x02}, {321, 0x06}, {344, 0x06},
      {392, 0x06}, {512, 0x06},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(proven_fetch_lines); ++i) {
      boundary.shader_binary_size = proven_fetch_lines[i].size;
      CHECK(hk_agx3_g17p_basic_graphics_program_generate(
         AGX_CHIP_G17P, &boundary, &program));
      CHECK(program.bytes[16] == proven_fetch_lines[i].encoded_word4_lo &&
            program.bytes[17] == 0x00);
   }
   const struct hk_agx3_g17p_graphics_program before = program;
   struct hk_agx3_g17p_graphics_program_desc invalid = vertex;
   invalid.shader_binary_size = 0;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &invalid,
                                                       &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17S, &vertex,
                                                       &program));
   invalid = resource;
   invalid.uniform_ranges = over_budget;
   invalid.uniform_range_count = ARRAY_SIZE(over_budget);
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &invalid,
                                                       &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   return 0;
}

static int
test_g17p_combined_sampler_pointers(void)
{
   const struct hk_agx3_g17p_combined_sampler_heap heap = {
      .descriptor_set_va = UINT64_C(0x10000020000),
      .sampler_heap_va = UINT64_C(0x10000080000),
      .texture_offset_B = 64,
      .sampler_index = 7,
      .sampler_count = 1024,
      .sampler_element_size_B = 32,
   };
   uint64_t pointers[2] = {0};
   CHECK(hk_agx3_g17p_combined_sampler_pointers(&heap, pointers));
   CHECK(pointers[0] == UINT64_C(0x10000020040));
   CHECK(pointers[1] == UINT64_C(0x100000800e0));

   /* Explicit null descriptors use software index zero. They select the
    * reserved robust sampler at native heap entry zero, never an underflowed
    * register descriptor. Raw index 1 selects the immediately following entry. */
   struct hk_agx3_g17p_combined_sampler_heap null_heap = heap;
   null_heap.sampler_index = 0;
   CHECK(hk_agx3_g17p_combined_sampler_pointers(&null_heap, pointers));
   CHECK(pointers[0] == UINT64_C(0x10000020040));
   CHECK(pointers[1] == UINT64_C(0x10000080000));
   null_heap.sampler_index = 1;
   CHECK(hk_agx3_g17p_combined_sampler_pointers(&null_heap, pointers));
   CHECK(pointers[1] == UINT64_C(0x10000080020));

   /* Native ESL consumes these two qwords, not the inline 32-byte records
    * that they address. Both hardware table indices select entry zero.
    */
   const struct agx3_ioto_uniform_range range = {
      .start_halfs = 0,
      .size_halfs = 8,
      .address = UINT64_C(0x10000090000),
   };

   const struct hk_agx3_g17p_graphics_program_desc desc = {
      .program_va = UINT64_C(0x10001990240),
      .shader_va = UINT64_C(0x10000000080),
      .uniform_ranges = &range,
      .uniform_range_count = 1,
      .shader_binary_size = 54,
      .nr_gprs = 8,
      .fragment = true,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct hk_agx3_g17p_graphics_program program;
   CHECK(hk_agx3_g17p_basic_graphics_program_generate(AGX_CHIP_G17P, &desc,
                                                      &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE &&
         program.length <= HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE &&
         program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
            0x0e);

   const uint64_t before[2] = {pointers[0], pointers[1]};
   struct hk_agx3_g17p_combined_sampler_heap bad = heap;
   bad.sampler_index = 1024;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_element_size_B = 8;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.texture_offset_B++;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.descriptor_set_va = UINT64_MAX - 32;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_heap_va = UINT64_MAX - 32;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.texture_offset_B = 0;
   bad.descriptor_set_va = UINT64_MAX - 15;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   bad = heap;
   bad.sampler_index = 0;
   bad.sampler_heap_va = UINT64_MAX - 15;
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&bad, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(NULL, pointers));
   CHECK(memcmp(pointers, before, sizeof(before)) == 0);
   CHECK(!hk_agx3_g17p_combined_sampler_pointers(&heap, NULL));

   struct hk_agx3_g17p_combined_sampler_heap client = heap;
   client.descriptor_set_va = UINT64_C(0x87ffed0000);
   client.sampler_heap_va = UINT64_C(0x87ffef4000);
   CHECK(hk_agx3_g17p_combined_sampler_pointers(&client, pointers));
   CHECK(pointers[0] == UINT64_C(0x87ffed0040));
   CHECK(pointers[1] == UINT64_C(0x87ffef40e0));

   const struct hk_agx3_g17p_sampled_storage_heap sampled_storage = {
      .sampled = client,
      .pbe_descriptor_set_va = UINT64_C(0x87fff10000),
      .write_control_va = UINT64_C(0x87fff20000),
      .pbe_offset_B = 160,
   };
   uint64_t four[4] = {0};
   CHECK(hk_agx3_g17p_sampled_storage_pointers(&sampled_storage, four));
   CHECK(four[0] == UINT64_C(0x87ffed0040));
   CHECK(four[1] == UINT64_C(0x87ffef40e0));
   CHECK(four[2] == UINT64_C(0x87fff100a0));
   CHECK(four[3] == sampled_storage.write_control_va);

   const uint64_t four_before[4] = {four[0], four[1], four[2], four[3]};
   struct hk_agx3_g17p_sampled_storage_heap invalid = sampled_storage;
   const uint64_t bad_controls[] = {0, 0xffff, UINT64_MAX};
   for (unsigned i = 0; i < 3; ++i) {
      invalid.write_control_va = bad_controls[i];
      CHECK(!hk_agx3_g17p_sampled_storage_pointers(&invalid, four));
      CHECK(memcmp(four, four_before, sizeof(four)) == 0);
   }
   invalid = sampled_storage;
   invalid.pbe_offset_B++;
   CHECK(!hk_agx3_g17p_sampled_storage_pointers(&invalid, four));
   CHECK(memcmp(four, four_before, sizeof(four)) == 0);
   invalid = sampled_storage;
   invalid.pbe_descriptor_set_va = UINT64_MAX - 64;
   CHECK(!hk_agx3_g17p_sampled_storage_pointers(&invalid, four));
   CHECK(memcmp(four, four_before, sizeof(four)) == 0);

   const struct hk_agx3_g17p_indexed_storage_heap indexed_storage = {
      .texture_heap_va = UINT64_C(0x9000000000),
      .sampler_heap_va = UINT64_C(0x9100000000),
      .pbe_descriptor_set_va = UINT64_C(0x9200000000),
      .write_control_va = UINT64_C(0x9300000000),
      .pbe_offset_B = 176,
   };
   CHECK(hk_agx3_g17p_indexed_storage_pointers(&indexed_storage, four));
   CHECK(four[0] == indexed_storage.texture_heap_va);
   CHECK(four[1] == indexed_storage.sampler_heap_va);
   CHECK(four[2] == UINT64_C(0x92000000b0));
   CHECK(four[3] == indexed_storage.write_control_va);

   const uint64_t indexed_before[4] = {
      four[0], four[1], four[2], four[3],
   };
   struct hk_agx3_g17p_indexed_storage_heap bad_indexed = indexed_storage;
   for (unsigned i = 0; i < 3; ++i) {
      bad_indexed.write_control_va = bad_controls[i];
      CHECK(!hk_agx3_g17p_indexed_storage_pointers(&bad_indexed, four));
      CHECK(memcmp(four, indexed_before, sizeof(four)) == 0);
   }
   bad_indexed = indexed_storage;
   bad_indexed.pbe_offset_B++;
   CHECK(!hk_agx3_g17p_indexed_storage_pointers(&bad_indexed, four));
   CHECK(memcmp(four, indexed_before, sizeof(four)) == 0);
   return 0;
}

static int
test_g17p_root_range(void)
{
   struct agx3_ioto_uniform_range range = {0};
   CHECK(hk_agx3_g17p_root_range(12, 0, UINT64_C(0x10000024000), 8, &range));
   CHECK(range.start_halfs == 12 && range.size_halfs == 4 &&
         range.address == UINT64_C(0x10000024000));
   CHECK(hk_agx3_g17p_root_range(12, 1, UINT64_C(0x10000024000), 16, &range));
   CHECK(range.start_halfs == 12 && range.size_halfs == 8);
   CHECK(hk_agx3_g17p_root_range(12, 8, UINT64_C(0x10000024000), 4096,
                                 &range));
   CHECK(range.start_halfs == 12 && range.size_halfs == 36);

   const struct agx3_ioto_uniform_range before = range;
   CHECK(!hk_agx3_g17p_root_range(13, 0, UINT64_C(0x10000024000), 4096,
                                  &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);
   CHECK(!hk_agx3_g17p_root_range(12, HK_AGX3_G17P_MAX_ROOT_SETS + 1,
                                  UINT64_C(0x10000024000), 4096, &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);
   CHECK(!hk_agx3_g17p_root_range(12, 0, 0, 4096, &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);
   /* The set pointers must lie inside the uploaded root: one set needs 16
    * bytes, so an 8-byte root is refused rather than read past. */
   CHECK(!hk_agx3_g17p_root_range(12, 1, UINT64_C(0x10000024000), 8, &range));
   CHECK(!hk_agx3_g17p_root_range(12, 1, UINT64_C(0x10000024000), 15,
                                  &range));
   CHECK(memcmp(&range, &before, sizeof(range)) == 0);

   struct agx3_ioto_uniform_range rodata = {0};
   CHECK(hk_agx3_g17p_rodata_range(
      8, 1, UINT64_C(0x10000030000), 4, 128, &rodata));
   CHECK(rodata.start_halfs == 8 && rodata.size_halfs == 2 &&
         rodata.address == UINT64_C(0x10000030004));
   CHECK(hk_agx3_g17p_rodata_range(
      12, 3, UINT64_C(0x10000030000), 8, 128, &rodata));
   CHECK(rodata.start_halfs == 12 && rodata.size_halfs == 4 &&
         rodata.address == UINT64_C(0x10000030008));
   const struct agx3_ioto_uniform_range rodata_before = rodata;
   CHECK(!hk_agx3_g17p_rodata_range(
      9, 1, UINT64_C(0x10000030000), 4, 128, &rodata));
   CHECK(memcmp(&rodata, &rodata_before, sizeof(rodata)) == 0);
   CHECK(!hk_agx3_g17p_rodata_range(
      510, 3, UINT64_C(0x10000030000), 4, 128, &rodata));
   CHECK(memcmp(&rodata, &rodata_before, sizeof(rodata)) == 0);
   CHECK(!hk_agx3_g17p_rodata_range(
      8, 1, UINT64_C(0x10000030000), 126, 128, &rodata));
   CHECK(memcmp(&rodata, &rodata_before, sizeof(rodata)) == 0);
   CHECK(!hk_agx3_g17p_rodata_range(
      8, 1, UINT64_C(0x10000030002), 4, 128, &rodata));
   CHECK(memcmp(&rodata, &rodata_before, sizeof(rodata)) == 0);

   struct agx3_ioto_uniform_range blend = {0};
   CHECK(hk_agx3_g17p_fragment_blend_constants_range(
      4, UINT64_C(0x10000028040), &blend));
   CHECK(blend.start_halfs == 4 && blend.size_halfs == 8 &&
         blend.address == UINT64_C(0x10000028040));

   CHECK(hk_agx3_g17p_fragment_blend_constants_range(
      28, UINT64_C(0x10000028040), &blend));
   CHECK(blend.start_halfs == 28 && blend.size_halfs == 8 &&
         blend.address == UINT64_C(0x10000028040));
   const struct agx3_ioto_uniform_range blend_before = blend;
   CHECK(!hk_agx3_g17p_fragment_blend_constants_range(28, 0, &blend));
   CHECK(memcmp(&blend, &blend_before, sizeof(blend)) == 0);
   CHECK(!hk_agx3_g17p_fragment_blend_constants_range(
      28, UINT64_C(0x10000028040), NULL));
   CHECK(!hk_agx3_g17p_fragment_blend_constants_range(
      27, UINT64_C(0x10000028040), &blend));
   CHECK(memcmp(&blend, &blend_before, sizeof(blend)) == 0);
   CHECK(!hk_agx3_g17p_fragment_blend_constants_range(
      506, UINT64_C(0x10000028040), &blend));
   CHECK(memcmp(&blend, &blend_before, sizeof(blend)) == 0);
   return 0;
}

static int
test_g17p_single_layer_rt0_layout(void)
{
   CHECK(hk_agx3_g17p_single_layer_rendering(0, 1));
   CHECK(!hk_agx3_g17p_single_layer_rendering(0, 0));
   CHECK(!hk_agx3_g17p_single_layer_rendering(0, 2));
   CHECK(!hk_agx3_g17p_single_layer_rendering(1, 1));

   enum pipe_format empty[AGX_MAX_RENDER_TARGETS] = {PIPE_FORMAT_NONE};
   struct agx_tilebuffer_layout empty_tib =
      agx_build_tilebuffer_layout(empty, 0, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   CHECK(empty_tib.sample_size_B == 0);
   empty_tib.sample_size_B = 8;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));

   empty_tib = agx_build_tilebuffer_layout(empty, 0, 0, false);
   const struct agx_tilebuffer_layout deferred = empty_tib;
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   CHECK(memcmp(&empty_tib, &deferred, sizeof(empty_tib)) == 0);
   empty_tib.sample_size_B = 8;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   empty_tib = deferred;
   empty_tib.logical_format[1] = PIPE_FORMAT_R8G8B8A8_UNORM;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   empty_tib = deferred;
   empty_tib.spilled[0] = true;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   empty_tib = deferred;
   agx_tilebuffer_set_samples(&empty_tib, 1);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));
   empty_tib.nr_samples = 8;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&empty_tib));

   enum pipe_format rgba[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(rgba, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   tib.nr_samples = 0;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format bgra[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B8G8R8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(bgra, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format narrow[] = {
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8G8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(narrow); ++i) {
      enum pipe_format format[AGX_MAX_RENDER_TARGETS] = {narrow[i]};
      tib = agx_build_tilebuffer_layout(format, 1, 1, false);
      CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
      CHECK(tib.sample_size_B == 8);
      CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   }

   enum pipe_format rg16[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rg16, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format bgr10a2_unorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B10G10R10A2_UNORM,
   };
   tib = agx_build_tilebuffer_layout(bgr10a2_unorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) ==
         PIPE_FORMAT_R10G10B10A2_UNORM);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rg16_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_UINT,
   };
   tib = agx_build_tilebuffer_layout(rg16_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format bgra8_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B8G8R8A8_SNORM,
   };
   tib = agx_build_tilebuffer_layout(bgra8_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba8_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UINT,
   };
   tib = agx_build_tilebuffer_layout(rgba8_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba16_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_SNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba16_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba16_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(rgba16_float, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r16_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(r16_float, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rg16_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(rg16_float, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r16_unorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16_UNORM,
   };
   tib = agx_build_tilebuffer_layout(r16_unorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r16_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16_SNORM,
   };
   tib = agx_build_tilebuffer_layout(r16_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r16_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16_UINT,
   };
   tib = agx_build_tilebuffer_layout(r16_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r16_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16_SINT,
   };
   tib = agx_build_tilebuffer_layout(r16_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r8_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8_SNORM,
   };
   tib = agx_build_tilebuffer_layout(r8_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rg8_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8_SNORM,
   };
   tib = agx_build_tilebuffer_layout(rg8_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba8_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_SNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba8_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba8_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_SINT,
   };
   tib = agx_build_tilebuffer_layout(rgba8_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rg8_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8_SINT,
   };
   tib = agx_build_tilebuffer_layout(rg8_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rg8_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8_UINT,
   };
   tib = agx_build_tilebuffer_layout(rg8_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format unqualified_i8_siblings[][AGX_MAX_RENDER_TARGETS] = {
      {PIPE_FORMAT_R8G8B8_UINT},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(unqualified_i8_siblings); ++i) {
      tib = agx_build_tilebuffer_layout(unqualified_i8_siblings[i], 1, 1,
                                        false);
      CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   }

   enum pipe_format r8_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8_UINT,
   };
   tib = agx_build_tilebuffer_layout(r8_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R8_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(r8_uint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format r8_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8_SINT,
   };
   tib = agx_build_tilebuffer_layout(r8_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R8_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(r8_sint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rg16_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_SINT,
   };
   tib = agx_build_tilebuffer_layout(rg16_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba16_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_UINT,
   };
   tib = agx_build_tilebuffer_layout(rgba16_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R16_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rgba16_uint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rgba16_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_SINT,
   };
   tib = agx_build_tilebuffer_layout(rgba16_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R16_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rgba16_sint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rgba32_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   tib = agx_build_tilebuffer_layout(rgba32_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 16);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rgba32_uint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rgba32_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_SINT,
   };
   tib = agx_build_tilebuffer_layout(rgba32_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 16);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rgba32_sint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rgba32_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(rgba32_float, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 16);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rgba32_float, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rgb10a2_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R10G10B10A2_UINT,
   };
   tib = agx_build_tilebuffer_layout(rgb10a2_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R16_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rgb10a2_uint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   /* A2R10G10B10_UINT_PACK32 swaps channels in the PBE, not in the tile. */
   enum pipe_format bgr10a2_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B10G10R10A2_UINT,
   };
   tib = agx_build_tilebuffer_layout(bgr10a2_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R16_UINT);
   enum pipe_format bgr10a2_uint_mrt[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B10G10R10A2_UINT, PIPE_FORMAT_A4R4G4B4_UNORM,
      PIPE_FORMAT_B5G5R5A1_UNORM, PIPE_FORMAT_B8G8R8A8_SNORM,
      PIPE_FORMAT_A4B4G4R4_UNORM,
   };
   tib = agx_build_tilebuffer_layout(bgr10a2_uint_mrt, 5, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format r32_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32_UINT,
   };
   tib = agx_build_tilebuffer_layout(r32_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format r32_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(r32_float, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   /* The exact scalar float evidence is single-sample. */
   tib = agx_build_tilebuffer_layout(r32_float, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rg32_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(rg32_float, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   /* The exact two-word float evidence is single-sample and format-exact. */
   tib = agx_build_tilebuffer_layout(rg32_float, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   enum pipe_format r32_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32_SINT,
   };
   tib = agx_build_tilebuffer_layout(r32_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   tib = agx_build_tilebuffer_layout(r32_sint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   enum pipe_format rg32_sint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32_SINT,
   };
   tib = agx_build_tilebuffer_layout(rg32_sint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rg32_sint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format rg32_uint[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32_UINT,
   };
   tib = agx_build_tilebuffer_layout(rg32_uint, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 1 && tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_physical_format(&tib, 0) == PIPE_FORMAT_R32_UINT);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   tib = agx_build_tilebuffer_layout(rg32_uint, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   enum pipe_format rg16_snorm[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16_SNORM,
   };
   tib = agx_build_tilebuffer_layout(rg16_snorm, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   enum pipe_format rgba16[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba16, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);

   tib = agx_build_tilebuffer_layout(rg16, 1, 1, false);

   tib.layered = true;
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.layered);

   enum pipe_format mrt[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(mrt, 2, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 8);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   CHECK(agx_tilebuffer_offset_B(&tib, 1) == 4);
   tib = agx_build_tilebuffer_layout(mrt, 2, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   enum pipe_format wboit[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.sample_size_B == 16);
   CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
   CHECK(agx_tilebuffer_offset_B(&tib, 1) == 8);
   tib = agx_build_tilebuffer_layout(wboit, 2, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));

   tib = agx_build_tilebuffer_layout(rgba, 1, 2, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 2 && tib.sample_size_B == 8 &&
         tib.tile_size == 32 * 32 && !tib.spilled[0]);
   CHECK(hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, true, true, false, false));

   enum pipe_format bgra_srgb[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_B8G8R8A8_SRGB,
   };
   tib = agx_build_tilebuffer_layout(bgra_srgb, 1, 1, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.logical_format[0] == PIPE_FORMAT_B8G8R8A8_SRGB &&
         tib.sample_size_B == 8 && tib.tile_size == 32 * 32 &&
         !tib.spilled[0]);
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 2, true, true, false, false));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, false, true, false, false));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, true, false, false, false));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, true, true, true, false));
   tib = agx_build_tilebuffer_layout(rgba, 1, 4, false);
   const struct agx_usc_shared_packed msaa4_usc = tib.usc;
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(tib.nr_samples == 4 && tib.sample_size_B == 8 &&
         tib.tile_size == 32 * 16 && !tib.spilled[0]);
   CHECK(memcmp(&tib.usc, &msaa4_usc, sizeof(tib.usc)) == 0);
   struct drm_asahi_cmd_render msaa4_cmd = {0};
   agx_tilebuffer_set_drm_cmd(&msaa4_cmd, &tib);
   CHECK(msaa4_cmd.utile_width_px == 32);
   CHECK(msaa4_cmd.utile_height_px == 16);
   CHECK(agx_tilebuffer_total_size(&tib) < 32768);
   CHECK(hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, true, true, false, false));
   tib = agx_build_tilebuffer_layout(rgba, 1, 8, false);
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, true, true, false, false));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      NULL, 1, true, true, false, false));

   /* Raw 32-bit float lanes keep their samples for the resolve after the
    * pass, while the store of those samples stays admitted. */
   enum pipe_format r32f[AGX_MAX_RENDER_TARGETS] = {PIPE_FORMAT_R32_FLOAT};
   tib = agx_build_tilebuffer_layout(r32f, 1, 4, false);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
      &tib, 1, true, true, false, false));
   CHECK(hk_agx3_g17p_multisample_store_supported(&tib, 1, true, 4));

   /* So do RGB9E5 samples, each one packed word. */
   enum pipe_format rgb9e5[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R9G9B9E5_FLOAT,
   };
   for (unsigned samples = 2; samples <= 4; samples += 2) {
      tib = agx_build_tilebuffer_layout(rgb9e5, 1, samples, false);
      CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
      CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
         &tib, 1, true, true, false, false));
      CHECK(hk_agx3_g17p_multisample_store_supported(&tib, 1, true, samples));
   }

   /* R11G11B10_FLOAT direct resolve requires the own-class PBE (G17P); on
    * raw-class chips (G15x, and G16G without own-class PBE) it falls back to
    * multisample store + shader resolve. */
   enum pipe_format r11g11b10[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R11G11B10_FLOAT,
   };
   for (unsigned samples = 2; samples <= 4; samples += 2) {
      tib = agx_build_tilebuffer_layout(r11g11b10, 1, samples, false);
      CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
      CHECK(!hk_agx3_g17p_direct_color_resolve_supported(
         &tib, 1, true, true, false, false));
      CHECK(hk_agx3_g17p_direct_color_resolve_supported(
         &tib, 1, true, true, false, true));
      CHECK(hk_agx3_g17p_multisample_store_supported(&tib, 1, true, samples));
   }
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(NULL));
   const enum pipe_format spill_formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   tib = agx_build_tilebuffer_layout_bounded(
      spill_formats, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE);
   CHECK(tib.spilled[0] && !tib.spilled[1] && !tib.spilled[2]);
   CHECK(agx_tilebuffer_total_size(&tib) < 32768);
   const struct agx_tilebuffer_layout spilled = tib;
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   CHECK(memcmp(&tib, &spilled, sizeof(tib)) == 0);
   tib.logical_format[0] = PIPE_FORMAT_NONE;
   CHECK(!hk_agx3_g17p_rt0_tile_layout_supported(&tib));
   return 0;
}

static int
test_alias(void)
{
   struct hk_agx3_ioto_program_desc desc = make_desc(AGX_CHIP_G17G);
   uint8_t storage[96] = {0x06, 0x00, 0x06, 0x00};
   static const uint8_t expected[] = {0x06, 0x00, 0x06, 0x00};
   desc.body = storage;
   desc.body_size = sizeof(expected);
   size_t written;
   CHECK(hk_agx3_ioto_program_generate(AGX_CHIP_G17G, &desc, storage + 2,
                                       sizeof(storage) - 2, &written));
   CHECK(memcmp(storage + 2, expected, sizeof(expected)) == 0);

   memset(storage, 0, sizeof(storage));
   memcpy(storage, expected, sizeof(expected));
   static const uint8_t suffix_code[] = {0x1b, 0x0c, 0x06, 0x00};
   memcpy(storage + 4, suffix_code, sizeof(suffix_code));
   desc.body = storage;
   desc.body_size = sizeof(expected);
   const struct agx_ioto_preshader_desc preshader = {
      .code = storage + 4,
      .code_size = sizeof(suffix_code),
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
   };
   CHECK(hk_agx3_ioto_program_generate_no_pixrel(
      AGX_CHIP_G17G, &desc, &preshader, storage + 2, sizeof(storage) - 2,
      &written));
   CHECK(memcmp(storage + 2, expected, sizeof(expected)) == 0);
   CHECK(storage[2 + written - 2] == 0x1b &&
         storage[2 + written - 1] == 0x0c);
   return 0;
}

static int
test_coverage_initial_pixel_dependencies(void)
{
   const struct agx3_ioto_uniform_range root = {
      .start_halfs = 12,
      .size_halfs = 4,
      .address = 0x2000,
   };
   struct hk_agx3_g17p_graphics_program_desc desc = {
      .program_va = UINT64_C(0x10000040000),
      .shader_va = UINT64_C(0x10000000080),
      .shader_binary_size = 280,
      .nr_gprs = 15,
      .compiler_d_valid = true,
      .compiler_d = 4,
      .fragment = true,
      .uniform_ranges = &root,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   static const uint8_t ordinary_low[3] = {0xf3, 0xc3, 0x03};
   static const uint8_t coverage_low[3] = {0xf2, 0xc2, 0x02};
   for (unsigned resources = 0; resources <= 1; resources++) {
      desc.uniform_range_count = resources;
      for (unsigned partitions = 1; partitions <= 3; ++partitions) {
         struct hk_agx3_g17p_graphics_program opaque, coverage;
         desc.colour_classes = BITFIELD_MASK(partitions);
         desc.writes_sample_mask = false;
         CHECK(hk_agx3_g17p_basic_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &opaque));
         desc.writes_sample_mask = true;
         CHECK(hk_agx3_g17p_basic_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &coverage));
         CHECK(coverage.length == opaque.length);
         const uint8_t native_tail[] = {
            0x07, 0x02, 0x54, coverage_low[partitions - 1], 0x02,
            0x00, 0x0e, 0x00, 0x00, 0x00,
         };
         CHECK(!memcmp(coverage.bytes + coverage.length - sizeof(native_tail),
                       native_tail, sizeof(native_tail)));
         CHECK(opaque.bytes[opaque.length - 7] ==
               ordinary_low[partitions - 1]);
         opaque.bytes[opaque.length - 7] = coverage_low[partitions - 1];
         CHECK(!memcmp(opaque.bytes, coverage.bytes, coverage.length));
      }
   }

   struct hk_agx3_g17p_graphics_program invalid;
   desc.colour_classes = 8;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &desc, &invalid));
   desc.colour_classes = 0;
   desc.fragment = false;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &desc, &invalid));
   desc.writes_sample_mask = false;
   desc.colour_classes = 1;
   CHECK(!hk_agx3_g17p_basic_graphics_program_generate(
      AGX_CHIP_G17P, &desc, &invalid));
   return 0;
}

static int
test_g17p_precompiled_fill_program(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                   &sources));
   const struct agx3_ioto_uniform_range args = {
      .start_halfs = 0,
      .size_halfs = 14,
      .address = UINT64_C(0x10000024000),
   };
   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000b0000),
      .shader_va = UINT64_C(0x100000003c0),
      .uniform_ranges = &args,
      .uniform_range_count = 1,
      .shader_binary_size = 58,
      .nr_gprs = 16,
      .compiler_d_valid = true,
      .compiler_d = 0,
      .launch_sources = sources,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   struct hk_agx3_g17p_compute_program program;
   memset(&program, 0xa5, sizeof(program));
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc,
                                               &program));
   CHECK(program.length > AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(program.bytes[program.length - AGX_IOTO_PROGRAM_TERMINATE_SIZE] ==
         0x0e);
   CHECK(program.launch.program_va == desc.program_va);
   CHECK(program.launch.shader_va == desc.shader_va);
   CHECK(program.launch.g17p_native_compute_live_bound);

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {96, 4, 1},
      .local_size = {32, 1, 1},
      .dimensions = 2,
   };
   static const uint8_t expected_dispatch[] = {
      0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x2c, 0x00, 0x00, 0x01, 0x00, 0x00, 0x40,
      0x60, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x60, 0x01, 0x00, 0x60,
   };
   uint8_t dispatch[sizeof(expected_dispatch) + 1];
   memset(dispatch, 0xb6, sizeof(dispatch));
   struct agx3_compute_dispatch_sink sink = {
      .data = dispatch,
      .capacity = sizeof(dispatch),
   };
   CHECK(hk_agx3_g17p_native_compute_dispatch_append(&program.launch, &grid,
                                                      &sink));
   CHECK(sink.length == sizeof(expected_dispatch));
   CHECK(memcmp(dispatch, expected_dispatch, sizeof(expected_dispatch)) == 0);
   CHECK(dispatch[sink.length] == 0xb6);

   const struct hk_agx3_g17p_compute_program before = program;
   struct hk_agx3_g17p_compute_program_desc invalid = desc;
   invalid.compiler_d_valid = false;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   invalid = desc;
   invalid.uniform_range_count = 1;
   invalid.uniform_ranges = NULL;
   CHECK(!hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &invalid,
                                                &program));
   CHECK(memcmp(&program, &before, sizeof(program)) == 0);
   return 0;
}

static int
test_g17p_compute_program_va_independent(void)
{
   struct agx3_cdm_direct_template_sources sources;
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false, &sources));
   const struct agx3_ioto_uniform_range args = {
      .start_halfs = 0, .size_halfs = 14,
      .address = UINT64_C(0x5ffde8c000),
   };
   struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = UINT64_C(0x100000b0000),
      .shader_va = UINT64_C(0x100000003c0),
      .uniform_ranges = &args,
      .shader_binary_size = 58, .nr_gprs = 16,
      .compiler_d_valid = true, .launch_sources = sources,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };

   for (unsigned ranges = 0; ranges <= 1; ++ranges) {
      desc.uniform_range_count = ranges;
      desc.program_va = UINT64_C(0x100000b0000);
      struct hk_agx3_g17p_compute_program retained, duplicate;
      CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &retained));
      desc.program_va += 0x400;
      CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &duplicate));
      /* These really are identical arena keys at different reservations. */
      CHECK(retained.length == duplicate.length);
      CHECK(!memcmp(retained.bytes, duplicate.bytes, retained.length));
      CHECK(retained.launch.program_va != duplicate.launch.program_va);
   }
   return 0;
}

/* SuperTuxKart 1.4's advanced ("deferred") pipeline, transcribed from
 * stk-code/src/graphics/rtts.cpp.  Every framebuffer the game builds must be
 * admitted by the G17P bounded RT0 contract, because a refusal is not a
 * degraded frame: hk_CmdBeginRendering() (hk_cmd_draw.c:1064) records
 * VK_ERROR_FEATURE_NOT_PRESENT, Zink's zink_batch.c:777 turns that into
 * is_device_lost and skips vkQueueSubmit, and the whole frame is dropped.
 *
 * Formats follow rtts.cpp:86-105.  With GL_EXT_color_buffer_float present -
 * which Honeykrisp advertises and SuperTuxKart logs as "EXT Color Buffer
 * Float Present" - rgba is RGBA16F, red is R16F, red32 is R32F and
 * diffuse/specular is R11F_G11F_B10F (rtts.cpp:92).  Without it every target
 * collapses to 8-bit (rtts.cpp:97-104).
 *
 * The game builds exactly two multi-attachment framebuffers, both at
 * attachment-1 byte offset 4: FBO_SP (rtts.cpp:180-182) and
 * FBO_COMBINED_DIFFUSE_SPECULAR (rtts.cpp:193-195).  It never builds a
 * mixed-format pair, a third colour target, or a multisampled MRT, so the
 * classifier's remaining fail-closed shapes are out of SuperTuxKart's reach.
 */
struct stk_framebuffer {
   const char *name;
   enum pipe_format format[2];
   unsigned count;
   enum agx_g17p_mrt_layout mrt;
};

static int
test_stk_advanced_pipeline_framebuffers(void)
{
   /* rtts.cpp:92 - GL_EXT_color_buffer_float present (Honeykrisp's case). */
   static const struct stk_framebuffer hdr[] = {
      {"FBO_COLORS", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      {"FBO_NORMAL_AND_DEPTHS", {PIPE_FORMAT_R8G8B8A8_UNORM}, 1},
      {"FBO_SP",
       {PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM}, 2,
       AGX_G17P_MRT_RGBA8},
      {"FBO_RGBA_1", {PIPE_FORMAT_R8G8B8A8_UNORM}, 1},
      {"FBO_RGBA_2", {PIPE_FORMAT_R8G8B8A8_UNORM}, 1},
      {"FBO_COMBINED_DIFFUSE_SPECULAR",
       {PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R11G11B10_FLOAT}, 2,
       AGX_G17P_MRT_R11G11B10},
      {"FBO_TMP1_WITH_DS", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      {"FBO_HALF1_R", {PIPE_FORMAT_R16_FLOAT}, 1},
      {"FBO_HALF1", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      {"FBO_HALF2", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      {"FBO_RGBA_3", {PIPE_FORMAT_R8G8B8A8_UNORM}, 1},
      {"FBO_SSAO", {PIPE_FORMAT_R16_FLOAT}, 1},
      {"FBO_LINEAR_DEPTH", {PIPE_FORMAT_R32_FLOAT}, 1},
      {"FBO_HALF2_R", {PIPE_FORMAT_R16_FLOAT}, 1},
      {"FBO_QUARTER1", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      {"FBO_QUARTER2", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      /* rtts.cpp:253-262 - the ten bloom/lens targets share rgba. */
      {"FBO_BLOOM_1024", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
      {"FBO_LENS_128", {PIPE_FORMAT_R16G16B16A16_FLOAT}, 1},
   };

   /* rtts.cpp:97-104 - the 8-bit fallback path. */
   static const struct stk_framebuffer ldr[] = {
      {"FBO_COLORS", {PIPE_FORMAT_R8G8B8A8_UNORM}, 1},
      {"FBO_SP",
       {PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM}, 2,
       AGX_G17P_MRT_RGBA8},
      {"FBO_COMBINED_DIFFUSE_SPECULAR",
       {PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM}, 2,
       AGX_G17P_MRT_RGBA8},
      {"FBO_HALF1_R", {PIPE_FORMAT_R8_UNORM}, 1},
      {"FBO_LINEAR_DEPTH", {PIPE_FORMAT_R8_UNORM}, 1},
   };

   const struct stk_framebuffer *sets[] = {hdr, ldr};
   const unsigned counts[] = {ARRAY_SIZE(hdr), ARRAY_SIZE(ldr)};

   for (unsigned s = 0; s < ARRAY_SIZE(sets); ++s) {
      for (unsigned i = 0; i < counts[s]; ++i) {
         const struct stk_framebuffer *fb = &sets[s][i];
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
         for (unsigned rt = 0; rt < fb->count; ++rt)
            formats[rt] = fb->format[rt];

         struct agx_tilebuffer_layout tib =
            agx_build_tilebuffer_layout(formats, fb->count, 1, false);

         if (!hk_agx3_g17p_rt0_tile_layout_supported(&tib)) {
            fprintf(stderr,
                    "FAIL line %d: SuperTuxKart %s is fail-closed on G17P\n",
                    __LINE__, fb->name);
            return 1;
         }
         ++checks;

         CHECK(agx_g17p_classify_mrt(&tib) == fb->mrt);

         /* Both of SuperTuxKart's MRT framebuffers pack attachment 1 at byte
          * offset 4 in an 8-byte sample.  Nothing in the game asks for the
          * offset-8 or 16-byte shapes. */
         if (fb->count == 2) {
            CHECK(agx_tilebuffer_offset_B(&tib, 0) == 0);
            CHECK(agx_tilebuffer_offset_B(&tib, 1) == 4);
            CHECK(tib.sample_size_B == 8);
         }
      }
   }

   /* rtts.cpp:296-299 - the cascaded shadow framebuffer has no colour
    * attachment and four layers.  hk clears `layered` before the bounded RT0
    * check (hk_agx3_ioto_program_core.c:85-87), so it must be admitted. */
   enum pipe_format none[AGX_MAX_RENDER_TARGETS] = {0};
   struct agx_tilebuffer_layout shadow =
      agx_build_tilebuffer_layout(none, 0, 1, true);
   CHECK(shadow.layered);
   CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&shadow));

   return 0;
}

static int
test_g17p_mixed_direct_color_resolve(void)
{
   /* Deferred local-read 5-RT MSAA4 shape: 4 G-buffer attachments
    * (DONT_CARE) plus 1 RGBA8 color attachment at offset 20 B (resolve-only). */
   const enum pipe_format deferred_formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(deferred_formats, 5, 4, false);
   CHECK(tib.nr_samples == 4 && tib.sample_size_B == 24 &&
         !agx_tilebuffer_spills(&tib));
   CHECK(agx_tilebuffer_offset_B(&tib, 4) == 20);

   struct hk_agx3_g17p_mixed_resolve_target one = {
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .view_format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .has_resolve_target = true,
      .average_resolve = true,
      .store_source = false,
      .resolve_pbe_valid = true,
      .resolve_format_matches = true,
      .resolve_samples = 1,
   };

   /* Final EOT on full render area with 5 MSAA + 1 resolve view (6 <= 14). */
   CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, false, false, false, false, 6));

   /* SNORM resolve targets are admitted (hardware PBE averages signed lanes). */
   struct hk_agx3_g17p_mixed_resolve_target snorm = one;
   snorm.format = PIPE_FORMAT_R8G8B8A8_SNORM;
   snorm.view_format = PIPE_FORMAT_R8G8B8A8_SNORM;
   CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &snorm, 1, false, false, false, false, 6));
   snorm.format = PIPE_FORMAT_R16G16B16A16_SNORM;
   snorm.view_format = PIPE_FORMAT_R16G16B16A16_SNORM;
   CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &snorm, 1, false, false, false, false, 6));

   /* sRGB resolve targets are rejected via the unmapped view format, even if
    * the EOT key format were remapped to linear UNORM. */
   struct hk_agx3_g17p_mixed_resolve_target srgb = one;
   srgb.format = PIPE_FORMAT_B8G8R8A8_SRGB;
   srgb.view_format = PIPE_FORMAT_B8G8R8A8_SRGB;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &srgb, 1, false, false, false, false, 6));
   srgb.format = PIPE_FORMAT_B8G8R8A8_UNORM;
   srgb.view_format = PIPE_FORMAT_B8G8R8A8_SRGB;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &srgb, 1, false, false, false, false, 6));

   /* 2x MSAA is admitted; 1x and 8x are refused. */
   struct agx_tilebuffer_layout tib2 =
      agx_build_tilebuffer_layout(deferred_formats, 5, 2, false);
   CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib2, &one, 1, false, false, false, false, 6));
   struct agx_tilebuffer_layout tib1 =
      agx_build_tilebuffer_layout(deferred_formats, 5, 1, false);
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib1, &one, 1, false, false, false, false, 6));

   /* Partial render, incomplete render area (including sparse multiview
    * viewMask), suspended/resumed dynamic rendering
    * (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT), unstored
    * resolve, and exceeding the 14 color+resolve write-range budget all fall
    * back. */
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, true, false, false, false, 6));
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, false, true, false, false, 6));
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, false, false, true, false, 6));
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, false, false, false, true, 6));
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, false, false, false, false,
      HK_AGX3_G17P_MAX_COLOR_RESOLVE_WRITE_RANGES + 1));
   CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &one, 1, false, false, false, false,
      HK_AGX3_G17P_MAX_COLOR_RESOLVE_WRITE_RANGES));

   /* STORE_OP_STORE, non-AVERAGE, missing/mismatched/MS resolve view fall back. */
   struct hk_agx3_g17p_mixed_resolve_target bad = one;
   bad.store_source = true;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));
   bad = one;
   bad.average_resolve = false;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));
   bad = one;
   bad.has_resolve_target = false;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));
   bad = one;
   bad.resolve_pbe_valid = false;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));
   bad = one;
   bad.resolve_format_matches = false;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));
   bad = one;
   bad.resolve_samples = 4;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));

   /* Raw-word formats (R32F, RGB9E5, R11G11B10F) and pure integers fall back. */
   const enum pipe_format shader_resolve_formats[] = {
      PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R9G9B9E5_FLOAT,
      PIPE_FORMAT_R11G11B10_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UINT,
      PIPE_FORMAT_R16G16_SINT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(shader_resolve_formats); ++i) {
      bad = one;
      bad.format = shader_resolve_formats[i];
      bad.view_format = shader_resolve_formats[i];
      CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
         &tib, &bad, 1, false, false, false, false, 6));
   }

   /* On G17P hk_eot_key_format() remaps R11G11B10_FLOAT to R10G10B10A2_UNORM
    * in key.tib.logical_format[]. Direct resolve must still reject it via the
    * unmapped view_format. */
   bad = one;
   bad.format = PIPE_FORMAT_R10G10B10A2_UNORM;
   bad.view_format = PIPE_FORMAT_R11G11B10_FLOAT;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, &bad, 1, false, false, false, false, 6));

   /* Multi-target direct resolve succeeds when every stored target qualifies
    * and falls back if any target stores source or needs shader resolve. */
   struct hk_agx3_g17p_mixed_resolve_target pair[2] = {one, one};
   pair[1].format = PIPE_FORMAT_R16G16B16A16_FLOAT;
   pair[1].view_format = PIPE_FORMAT_R16G16B16A16_FLOAT;
   CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, pair, 2, false, false, false, false, 7));
   pair[1].store_source = true;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, pair, 2, false, false, false, false, 7));
   pair[1].store_source = false;
   pair[1].format = PIPE_FORMAT_R32_FLOAT;
   pair[1].view_format = PIPE_FORMAT_R32_FLOAT;
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &tib, pair, 2, false, false, false, false, 7));

   /* Spilled tilebuffers refuse direct resolve. */
   const enum pipe_format spill_formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   struct agx_tilebuffer_layout spilled = agx_build_tilebuffer_layout_bounded(
      spill_formats, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE);
   CHECK(agx_tilebuffer_spills(&spilled));
   CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
      &spilled, &one, 1, false, false, false, false, 4));

   /* Verify the Ioto EOT program generator accepts both the single-stored-RT
    * deferred shape (count=1 at offset 20 B in a 24 B/sample MSAA4 tile) and a
    * 2-target mixed-MRT MSAA4 resolve shape. */
   struct agx3_g17p_rt0_eot_program eot = {0};
   const struct agx3_g17p_rt0_eot_program_desc single_desc = {
      .program_va = 0x1000000000ULL,
      .shader_va = 0x1000002000ULL,
      .pbe_argument_va = 0x1000004000ULL,
      .imageblock_offset_B = 20,
      .attachment_count = 1,
      .mixed_mrt = false,
      .shader_binary_size = 68,
      .scratch_size = 0,
      .nr_gprs = 8,
      .staging_halfs = 8,
      .imageblock = {
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size_px = tib.tile_size,
      },
   };
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G16G, &single_desc, &eot));
   CHECK(eot.length > 0);

   const struct agx3_g17p_rt0_eot_program_desc dual_desc = {
      .program_va = 0x1000000000ULL,
      .shader_va = 0x1000002000ULL,
      .pbe_argument_va = 0x1000004000ULL,
      .imageblock_offset_B = 0,
      .imageblock_offset1_B = 20,
      .attachment_count = 2,
      .mixed_mrt = true,
      .shader_binary_size = 32 + 44 * 2,
      .scratch_size = 0,
      .nr_gprs = 8,
      .staging_halfs = 16,
      .imageblock = {
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size_px = tib.tile_size,
      },
   };
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G16G, &dual_desc, &eot));
   CHECK(eot.length > 0);

   return 0;
}

/* The fused end-of-tile store + resolve admits exactly the single-target
 * final pass that stores its multisample samples and AVERAGE-resolves them
 * over the whole attachment; every other shape keeps the after-pass resolve. */
static int
test_g17p_store_and_resolve(void)
{
   const enum pipe_format rgba[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(rgba, 1, 4, false);
   CHECK(tib.nr_samples == 4 && !tib.layered && !agx_tilebuffer_spills(&tib));

   const struct hk_agx3_g17p_store_and_resolve_target ok = {
      .color_attachment_count = 1,
      .final_store = true,
      .whole_attachments = true,
      .store_source = true,
      .has_resolve_target = true,
      .average_resolve = true,
      .source_samples = 4,
      .resolve_samples = 1,
      .resolve_format_matches = true,
      .source_pbe_valid = true,
      .resolve_pbe_valid = true,
   };
   CHECK(hk_agx3_g17p_store_and_resolve_supported(&tib, &ok));
   /* The stored half is the admitted multisample store, and the shape is
    * outside the resolve-only program, which needs storeOp != STORE. */
   CHECK(hk_agx3_g17p_multisample_store_supported(&tib, 1, true, 4));
   CHECK(!hk_agx3_g17p_direct_color_resolve_supported(&tib, 1, true, true,
                                                     true, false));
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(NULL, &ok));
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, NULL));

   /* Every requirement on its own. */
   struct hk_agx3_g17p_store_and_resolve_target t;
#define REFUSED_WITH(field, value)                                             \
   do {                                                                        \
      t = ok;                                                                  \
      t.field = value;                                                         \
      CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &t));              \
   } while (0)
   REFUSED_WITH(color_attachment_count, 2);
   REFUSED_WITH(color_attachment_count, 0);
   /* A partial render keeps its samples for the reload. */
   REFUSED_WITH(final_store, false);
   /* No background merge outside a partial render area. */
   REFUSED_WITH(whole_attachments, false);
   /* Not stored: the resolve-only program's shape. */
   REFUSED_WITH(store_source, false);
   REFUSED_WITH(has_resolve_target, false);
   /* SAMPLE_ZERO, MIN and MAX resolve after the pass. */
   REFUSED_WITH(average_resolve, false);
   REFUSED_WITH(source_samples, 2);
   REFUSED_WITH(source_samples, 1);
   REFUSED_WITH(resolve_samples, 4);
   REFUSED_WITH(resolve_samples, 0);
   REFUSED_WITH(resolve_format_matches, false);
   REFUSED_WITH(source_pbe_valid, false);
   REFUSED_WITH(resolve_pbe_valid, false);
#undef REFUSED_WITH

   /* 2x is admitted; 1x and 8x are not. */
   tib = agx_build_tilebuffer_layout(rgba, 1, 2, false);
   t = ok;
   t.source_samples = 2;
   CHECK(hk_agx3_g17p_store_and_resolve_supported(&tib, &t));
   tib = agx_build_tilebuffer_layout(rgba, 1, 1, false);
   t.source_samples = 1;
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &t));
   tib = agx_build_tilebuffer_layout(rgba, 1, 8, false);
   t.source_samples = 8;
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &t));

   /* No layered or spilled two-output program is built. */
   tib = agx_build_tilebuffer_layout(rgba, 1, 4, true);
   CHECK(tib.layered);
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &ok));
   tib = agx_build_tilebuffer_layout(rgba, 1, 4, false);
   tib.spilled[0] = true;
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &ok));

   /* Formats: those whose lanes the end-of-tile resolve averages. */
   const struct {
      enum pipe_format format;
      bool admitted;
   } formats[] = {
      {PIPE_FORMAT_R8G8B8A8_UNORM, true},
      {PIPE_FORMAT_B8G8R8A8_UNORM, true},
      /* The caller sets the PBE's linear-resolve transfer bit. */
      {PIPE_FORMAT_B8G8R8A8_SRGB, true},
      {PIPE_FORMAT_R8G8B8A8_SRGB, true},
      {PIPE_FORMAT_R16G16B16A16_FLOAT, true},
      {PIPE_FORMAT_R10G10B10A2_UNORM, true},
      /* Raw words: their averages are not the mean of the samples. */
      {PIPE_FORMAT_R32_FLOAT, false},
      {PIPE_FORMAT_R9G9B9E5_FLOAT, false},
      {PIPE_FORMAT_R11G11B10_FLOAT, false},
      /* Integer formats never AVERAGE. */
      {PIPE_FORMAT_R8G8B8A8_UINT, false},
      {PIPE_FORMAT_R16G16_SINT, false},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i) {
      const enum pipe_format f[AGX_MAX_RENDER_TARGETS] = {formats[i].format};
      tib = agx_build_tilebuffer_layout(f, 1, 4, false);
      if (formats[i].admitted) {
         /* Admitted formats keep a supported multisample tile. */
         CHECK(hk_agx3_g17p_rt0_tile_layout_supported(&tib));
         CHECK(hk_agx3_g17p_store_and_resolve_supported(&tib, &ok));
      } else {
         CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &ok));
      }
   }

   /* A colourless tile has nothing to resolve. */
   const enum pipe_format none[AGX_MAX_RENDER_TARGETS] = {PIPE_FORMAT_NONE};
   tib = agx_build_tilebuffer_layout(none, 1, 4, false);
   CHECK(!hk_agx3_g17p_store_and_resolve_supported(&tib, &ok));
   return 0;
}

int
main(void)
{
   if (test_g17p_constant_program_binary_extent() ||
       test_g17p_graphics_constant_program() ||
       test_g17p_precompiled_data_layout() ||
       test_g17p_graphics_root_requirement() ||
       test_g17p_compute_program_va_independent() ||
       test_coverage_initial_pixel_dependencies() ||
       test_g17p_precompiled_fill_program() || test_exact_target_order() || test_terminal_suffix_modes() ||
       test_fail_closed() || test_g17s_minimal_compute_admission() ||
       test_g17p_dynamic_compute_program() ||
       test_g17p_basic_graphics_program() ||
       test_g17p_combined_sampler_pointers() ||
       test_g17p_root_range() ||
       test_g17p_single_layer_rt0_layout() ||
       test_stk_advanced_pipeline_framebuffers() ||
       test_g17p_mixed_direct_color_resolve() || test_g17p_store_and_resolve() ||
       test_alias())
      return 1;

   printf("Honeykrisp AGX3 Ioto program core: %u checks passed\n", checks);
   return 0;
}
