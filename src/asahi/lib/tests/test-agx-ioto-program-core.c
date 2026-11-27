/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../agx_ioto_program_core.h"

#include "../agx_ioto_core.h"
#include "../agx_ioto_load_core.h"
#include "../../compiler/test/probes/g17p-sparse-mrt.h"

#include <stdio.h>
#include <string.h>

static uint64_t checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

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

static struct agx_ioto_state_tail
make_tail(enum agx_chip chip, enum agx_ioto_state_tail_type type)
{
   struct agx_ioto_state_tail tail = {.type = type};
   if (is_gpu1(chip)) {
      if (type == AGX_IOTO_STATE_TAIL_TYPE0) {
         tail.speclm.f[0] = 3;
         tail.speclm.f[1] = 165;
      }
      tail.ldshdr.f[5] = 3;
      tail.ldshdr.f[6] = 107;
      tail.spectpr.f[0] = 3;
      tail.spectpr.f[1] = 166;
   } else {
      tail.ldshdr.f[23] = 1;
   }
   tail.ldshdr.f[0] = UINT64_C(0x123456780);
   return tail;
}

static int
test_exact_order(void)
{
   static const uint8_t body[] = {0x06, 0x00};

   for (size_t t = 0; t < ARRAY_SIZE(targets); ++t) {
      const enum agx_chip chip = targets[t];
      for (unsigned type = 0; type <= 1; ++type) {
         const struct agx_ioto_state_tail tail =
            make_tail(chip, (enum agx_ioto_state_tail_type)type);
         const struct agx_ioto_profile_control profile = {
            .chip = chip,
            .payload0 = is_gpu1(chip) ? UINT16_C(0x8098)
                                     : UINT16_C(0x0555),
            .payload1 = is_gpu1(chip) ? UINT16_C(0xa400) : 0,
         };
         struct agx_ioto_state_tail_output packed_tail;
         uint8_t packed_profile[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE];
         CHECK(agx_ioto_state_tail_pack(chip, &tail, &packed_tail));
         CHECK(agx_ioto_profile_control_pack(&profile, packed_profile));
         const size_t profile_size = agx_ioto_profile_control_size(chip);
         const size_t expected = sizeof(body) + packed_tail.length +
                                 profile_size +
                                 AGX_IOTO_PROGRAM_TERMINATE_SIZE;

         uint8_t output[64];
         memset(output, 0xa5, sizeof(output));
         size_t written = SIZE_MAX;
         CHECK(agx_ioto_straight_line_program_compose(
            chip, UINT64_C(0x100000040), body, sizeof(body), &tail, &profile,
            output, expected, &written));

         CHECK(written == expected);
         CHECK(memcmp(output, body, sizeof(body)) == 0);
         CHECK(memcmp(output + sizeof(body), packed_profile, profile_size) ==
               0);
         CHECK(memcmp(output + sizeof(body) + profile_size, packed_tail.bytes,
                      packed_tail.length) == 0);
         static const uint8_t terminate[] = {0x0e, 0x00, 0x00, 0x00};
         CHECK(memcmp(output + expected - sizeof(terminate), terminate,
                      sizeof(terminate)) == 0);
         CHECK(output[expected] == 0xa5);
      }
   }

   return 0;
}

static int
test_terminal_suffix_modes(void)
{
   static const uint8_t body[] = {0x06, 0x00};
   static const uint8_t inline_code[] = {0x1b, 0x0c, 0x06, 0x00};
   static const uint8_t all_nops[] = {0x06, 0x00, 0x06, 0x00};

   for (size_t t = 0; t < ARRAY_SIZE(targets); ++t) {
      const enum agx_chip chip = targets[t];
      const struct agx_ioto_state_tail tail =
         make_tail(chip, AGX_IOTO_STATE_TAIL_TYPE1);
      const struct agx_ioto_profile_control profile = {
         .chip = chip,
         .payload0 = is_gpu1(chip) ? UINT16_C(0x8098)
                                  : UINT16_C(0x0555),
         .payload1 = is_gpu1(chip) ? UINT16_C(0xa400) : 0,
      };
      struct agx_ioto_state_tail_output packed_tail;
      uint8_t packed_profile[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE];
      CHECK(agx_ioto_state_tail_pack(chip, &tail, &packed_tail));
      CHECK(agx_ioto_profile_control_pack(&profile, packed_profile));
      const size_t profile_size = agx_ioto_profile_control_size(chip);

      for (unsigned mode = 0; mode < 3; ++mode) {
         struct agx_ioto_preshader_desc preshader = {
            .external_va = AGX_IOTO_PRESHADER_NO_VA,
         };
         if (mode == 0) {
            preshader.code = all_nops;
            preshader.code_size = sizeof(all_nops);
         } else if (mode == 1) {
            preshader.code = inline_code;
            preshader.code_size = sizeof(inline_code);
         } else {
            preshader.force_external = true;
            preshader.request_prelude = true;
            preshader.external_va = UINT64_C(0x123456789abc);
         }

         struct agx_ioto_preshader_output packed_preshader;
         CHECK(agx_ioto_preshader_emit(chip, &preshader,
                                       &packed_preshader));
         CHECK(packed_preshader.mode ==
               (mode == 0 ? AGX_IOTO_PRESHADER_MODE_ABSENT
                          : mode == 1 ? AGX_IOTO_PRESHADER_MODE_INLINE
                                      : AGX_IOTO_PRESHADER_MODE_EXTERNAL));

         const size_t total = sizeof(body) + profile_size +
                              packed_tail.length + packed_preshader.length;
         uint8_t output[96];
         memset(output, 0xa5, sizeof(output));
         size_t written = SIZE_MAX;
         CHECK(agx_ioto_program_compose_no_pixrel(
            chip, UINT64_C(0x100000040), body, sizeof(body), &tail, &profile,
            &preshader, output, total, &written));
         CHECK(written == total);
         CHECK(memcmp(output, body, sizeof(body)) == 0);
         CHECK(memcmp(output + sizeof(body), packed_profile, profile_size) ==
               0);
         CHECK(memcmp(output + sizeof(body) + profile_size, packed_tail.bytes,
                      packed_tail.length) == 0);
         const size_t suffix_offset =
            sizeof(body) + profile_size + packed_tail.length;
         CHECK(memcmp(output + suffix_offset, packed_preshader.bytes,
                      packed_preshader.length) == 0);
         CHECK(output[total] == 0xa5);
         if (mode != 0) {
            static const uint8_t absent[] = {0x0e, 0x00, 0x00, 0x00};
            CHECK(packed_preshader.length < sizeof(absent) ||
                  memcmp(output + total - sizeof(absent), absent,
                         sizeof(absent)) != 0);
         }

         uint8_t failed[96];
         uint8_t before[sizeof(failed)];
         memset(failed, 0x5a, sizeof(failed));
         memcpy(before, failed, sizeof(failed));
         size_t failed_written = 17;
         CHECK(!agx_ioto_program_compose_no_pixrel(
            chip, UINT64_C(0x100000040), body, sizeof(body), &tail, &profile,
            &preshader, failed, total - 1, &failed_written));
         CHECK(memcmp(failed, before, sizeof(failed)) == 0);
         CHECK(failed_written == 17);
      }
   }

   return 0;
}

static int
expect_failure_unchanged(enum agx_chip chip, uint64_t program_va,
                         const void *body, size_t body_size,
                         const struct agx_ioto_state_tail *tail,
                         const struct agx_ioto_profile_control *profile,
                         size_t capacity)
{
   uint8_t output[96];
   uint8_t before[sizeof(output)];
   memset(output, 0x5a, sizeof(output));
   memcpy(before, output, sizeof(output));
   size_t written = UINT64_C(0x123456789abcdef0);
   const size_t written_before = written;

   CHECK(!agx_ioto_straight_line_program_compose(
      chip, program_va, body, body_size, tail, profile, output, capacity,
      &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0);
   CHECK(written == written_before);
   return 0;
}

static int
test_fail_closed(void)
{
   const enum agx_chip chip = AGX_CHIP_G15G;
   struct agx_ioto_state_tail tail =
      make_tail(chip, AGX_IOTO_STATE_TAIL_TYPE0);
   struct agx_ioto_profile_control profile = {.chip = chip};
   static const uint8_t nop[] = {0x06, 0x00};
   static const uint8_t truncated[] = {0x06};
   static const uint8_t branch_gpu0[] = {
      0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_record decoded;
   size_t consumed;
   CHECK(agx_ioto_decode_one(chip, branch_gpu0, sizeof(branch_gpu0), &decoded,
                             &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_B_EP0 &&
         consumed == sizeof(branch_gpu0));

   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000041), nop,
                                  sizeof(nop), &tail, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x1000000000000), nop,
                                  sizeof(nop), &tail, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(AGX_CHIP_G14X, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), NULL,
                                  sizeof(nop), &tail, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), truncated,
                                  sizeof(truncated), &tail, &profile, 96) ==
         0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), branch_gpu0,
                                  sizeof(branch_gpu0), &tail, &profile, 96) ==
         0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), NULL, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, NULL, 96) == 0);

   profile.chip = AGX_CHIP_G17G;
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, &profile, 96) == 0);
   profile.chip = chip;
   tail.type = (enum agx_ioto_state_tail_type)2;
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, &profile, 96) == 0);

   tail = make_tail(chip, AGX_IOTO_STATE_TAIL_TYPE0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, &profile, 1) == 0);
   struct agx_ioto_state_tail_output packed_tail;
   CHECK(agx_ioto_state_tail_pack(chip, &tail, &packed_tail));
   const size_t exact_size = sizeof(nop) + packed_tail.length +
                             agx_ioto_profile_control_size(chip) +
                             AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, &profile,
                                  exact_size - 1) == 0);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop, SIZE_MAX,
                                  &tail, &profile, 96) == 0);

   profile.payload0 = UINT16_C(0x0800);
   CHECK(expect_failure_unchanged(chip, UINT64_C(0x100000040), nop,
                                  sizeof(nop), &tail, &profile, 96) == 0);
   profile.payload0 = 0;

   uint8_t output[96];
   size_t written = 7;
   CHECK(!agx_ioto_straight_line_program_compose(
      chip, UINT64_C(0x100000040), nop, sizeof(nop), &tail, &profile, NULL,
      sizeof(output), &written));
   CHECK(written == 7);
   CHECK(!agx_ioto_straight_line_program_compose(
      chip, UINT64_C(0x100000040), nop, sizeof(nop), &tail, &profile, output,
      sizeof(output), NULL));

   return 0;
}

static int
test_rejects_composer_owned_and_pixel_records(void)
{
   static const uint8_t state_gpu0[] = {
      0x77, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t profile_gpu0[] = {0xf7, 0x03, 0x00, 0x00};
   static const uint8_t pixrel_gpu1[] = {0x07, 0x00, 0x00,
                                         0x04, 0x00, 0x00};
   static const uint8_t pixwait_gpu1[] = {0x07, 0x00, 0x00,
                                          0x05, 0x00, 0x00};

   struct agx_ioto_record decoded;
   size_t consumed;
   CHECK(agx_ioto_decode_one(AGX_CHIP_G15G, state_gpu0,
                             sizeof(state_gpu0), &decoded, &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_LDSHDR_EP0 &&
         consumed == sizeof(state_gpu0));
   CHECK(agx_ioto_decode_one(AGX_CHIP_G15G, profile_gpu0,
                             sizeof(profile_gpu0), &decoded, &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_SETPROFILECTL_EP0 &&
         consumed == sizeof(profile_gpu0));
   CHECK(agx_ioto_decode_one(AGX_CHIP_G17G, pixrel_gpu1,
                             sizeof(pixrel_gpu1), &decoded, &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_PIXREL_EP0 &&
         consumed == sizeof(pixrel_gpu1));
   CHECK(agx_ioto_decode_one(AGX_CHIP_G17G, pixwait_gpu1,
                             sizeof(pixwait_gpu1), &decoded, &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_PIXWAIT_EP0 &&
         consumed == sizeof(pixwait_gpu1));

   struct agx_ioto_state_tail tail =
      make_tail(AGX_CHIP_G15G, AGX_IOTO_STATE_TAIL_TYPE0);
   struct agx_ioto_profile_control profile = {.chip = AGX_CHIP_G15G};
   CHECK(expect_failure_unchanged(
            AGX_CHIP_G15G, UINT64_C(0x100000040), state_gpu0,
            sizeof(state_gpu0), &tail, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(
            AGX_CHIP_G15G, UINT64_C(0x100000040), profile_gpu0,
            sizeof(profile_gpu0), &tail, &profile, 96) == 0);

   tail = make_tail(AGX_CHIP_G17G, AGX_IOTO_STATE_TAIL_TYPE0);
   profile.chip = AGX_CHIP_G17G;
   CHECK(expect_failure_unchanged(
            AGX_CHIP_G17G, UINT64_C(0x100000040), pixrel_gpu1,
            sizeof(pixrel_gpu1), &tail, &profile, 96) == 0);
   CHECK(expect_failure_unchanged(
            AGX_CHIP_G17G, UINT64_C(0x100000040), pixwait_gpu1,
            sizeof(pixwait_gpu1), &tail, &profile, 96) == 0);

   return 0;
}

static int
test_empty_body_and_alias(void)
{
   const enum agx_chip chip = AGX_CHIP_G17G;
   const struct agx_ioto_state_tail tail =
      make_tail(chip, AGX_IOTO_STATE_TAIL_TYPE1);
   const struct agx_ioto_profile_control profile = {.chip = chip};

   uint8_t output[64];
   memset(output, 0xcc, sizeof(output));
   size_t written;
   CHECK(agx_ioto_straight_line_program_compose(
      chip, UINT64_C(0x100000040), NULL, 0, &tail, &profile, output,
      sizeof(output), &written));
   CHECK(written == AGX_IOTO_STATE_TAIL_TYPE1_SIZE +
                       agx_ioto_profile_control_size(chip) +
                       AGX_IOTO_PROGRAM_TERMINATE_SIZE);

   uint8_t aliased[64] = {0x06, 0x00};
   CHECK(agx_ioto_straight_line_program_compose(
      chip, UINT64_C(0x100000040), aliased, 2, &tail, &profile, aliased,
      sizeof(aliased), &written));
   CHECK(aliased[0] == 0x06 && aliased[1] == 0x00);

   static const uint8_t two_nops[] = {0x06, 0x00, 0x06, 0x00};
   uint8_t overlap[96];
   memset(overlap, 0xcc, sizeof(overlap));
   memcpy(overlap, two_nops, sizeof(two_nops));
   CHECK(agx_ioto_straight_line_program_compose(
      chip, 0, overlap, sizeof(two_nops), &tail, &profile, overlap + 2,
      sizeof(overlap) - 2, &written));
   CHECK(memcmp(overlap + 2, two_nops, sizeof(two_nops)) == 0);

   memset(overlap, 0xcc, sizeof(overlap));
   memcpy(overlap + 2, two_nops, sizeof(two_nops));
   CHECK(agx_ioto_straight_line_program_compose(
      chip, UINT64_C(0xffffffffffc0), overlap + 2, sizeof(two_nops), &tail,
      &profile, overlap, sizeof(overlap), &written));
   CHECK(memcmp(overlap, two_nops, sizeof(two_nops)) == 0);

   memset(overlap, 0xcc, sizeof(overlap));
   memcpy(overlap, two_nops, sizeof(two_nops));
   static const uint8_t inline_suffix[] = {0x1b, 0x0c, 0x06, 0x00};
   memcpy(overlap + 4, inline_suffix, sizeof(inline_suffix));
   const struct agx_ioto_preshader_desc preshader = {
      .code = overlap + 4,
      .code_size = sizeof(inline_suffix),
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
   };
   CHECK(agx_ioto_program_compose_no_pixrel(
      chip, UINT64_C(0x100000040), overlap, sizeof(two_nops), &tail, &profile,
      &preshader, overlap + 2, sizeof(overlap) - 2, &written));
   CHECK(memcmp(overlap + 2, two_nops, sizeof(two_nops)) == 0);
   CHECK(overlap[2 + written - 2] == 0x1b &&
         overlap[2 + written - 1] == 0x0c);

   uint8_t failed[96];
   uint8_t before[sizeof(failed)];
   memset(failed, 0x5a, sizeof(failed));
   memcpy(before, failed, sizeof(failed));
   size_t failed_written = 23;
   CHECK(!agx_ioto_program_compose_no_pixrel(
      chip, UINT64_C(0x100000040), two_nops, sizeof(two_nops), &tail, &profile,
      NULL, failed, sizeof(failed), &failed_written));
   CHECK(memcmp(failed, before, sizeof(failed)) == 0);
   CHECK(failed_written == 23);
   return 0;
}

static int
test_g17p_no_resource_program(void)
{
   static const uint8_t expected[AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01, 0x2a, 0x01,
      0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00, 0x2a, 0x80, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x14, 0x81, 0x11, 0x06, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x80, 0x02, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8,
      0x10, 0x05, 0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x12,
      0x54, 0x00, 0x4c, 0x00, 0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x60, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_state_tail_output tail = {
      .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
   };
   memcpy(tail.bytes, expected, tail.length);

   uint8_t output[sizeof(expected) + 1];
   memset(output, 0xa5, sizeof(output));
   size_t written = SIZE_MAX;
   CHECK(agx_ioto_g17p_no_resource_program_compose_trusted(
      UINT64_C(0x10001990240), &tail, output, sizeof(expected), &written));
   CHECK(written == sizeof(expected));
   CHECK(memcmp(output, expected, sizeof(expected)) == 0);
   CHECK(output[sizeof(expected)] == 0xa5);

   uint8_t before[sizeof(output)];
   memset(before, 0xa5, sizeof(before));
   memcpy(output, before, sizeof(output));
   written = 23;
   CHECK(!agx_ioto_g17p_no_resource_program_compose_trusted(
      UINT64_C(0x10001990241), &tail, output, sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 23);
   CHECK(!agx_ioto_g17p_no_resource_program_compose_trusted(
      0, &tail, output, sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 23);
   CHECK(!agx_ioto_g17p_no_resource_program_compose_trusted(
      UINT64_C(0x10001990240), &tail, output, sizeof(expected) - 1, &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 23);

   tail.length--;
   CHECK(!agx_ioto_g17p_no_resource_program_compose_trusted(
      UINT64_C(0x10001990240), &tail, output, sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 23);
   tail.length++;
   tail.bytes[11] = 0;
   tail.bytes[15] = 0;
   CHECK(!agx_ioto_g17p_no_resource_program_compose_trusted(
      UINT64_C(0x10001990240), &tail, output, sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 23);
   return 0;
}

static int
test_g17p_fragment_no_resource_program(void)
{
   static const uint8_t tail_bytes[AGX_IOTO_STATE_TAIL_TYPE0_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x01, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00,
      0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t expected[] = {
         0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01,
         0x2a, 0x01, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00,
         0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0x80,
         0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x81, 0x11, 0x06,
         0x00, 0x00, 0x00, 0x00, 0x0c, 0x80, 0x02, 0x00, 0x04, 0x00,
         0x00, 0x00, 0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8,
         0x10, 0x05, 0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
         0x0f, 0x12, 0x54, 0x00, 0x4c, 0x00, 0x07, 0x02, 0x54, 0xf3,
         0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
      };
   CHECK(sizeof(expected) ==
         AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE);
   struct agx_ioto_state_tail_output tail = {
      .length = sizeof(tail_bytes),
   };
   memcpy(tail.bytes, tail_bytes, sizeof(tail_bytes));

   uint8_t output[sizeof(expected) + 1];
   memset(output, 0xa5, sizeof(output));
   size_t written = SIZE_MAX;
   CHECK(agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
      UINT64_C(0x10000130640), &tail, 1, output, sizeof(expected), &written));
   CHECK(written == sizeof(expected));
   CHECK(memcmp(output, expected, sizeof(expected)) == 0);
   CHECK(output[sizeof(expected)] == 0xa5);

   /* applegpu_g17p metadata makes the consumed colour classes observable in
    * exactly one PIXREL descriptor: p0=0x2f3, p0+p1=0x2c3, p0+p1+p2=0x203,
    * and for fragments writing only some targets p1+p2=0x20f and p2=0x23f.
    * Coverage holds class0 as well.  Compare each complete composed program
    * so the record position and terminal framing are also part of the
    * oracle. */
   static const struct {
      uint8_t classes;
      uint8_t pixrel[2][6];
   } native_pixrel[] = {
      {1, {{0x07,0x02,0x54,0xf3,0x02,0x00}, {0x07,0x02,0x54,0xf2,0x02,0x00}}},
      {3, {{0x07,0x02,0x54,0xc3,0x02,0x00}, {0x07,0x02,0x54,0xc2,0x02,0x00}}},
      {7, {{0x07,0x02,0x54,0x03,0x02,0x00}, {0x07,0x02,0x54,0x02,0x02,0x00}}},
      {6, {{0x07,0x02,0x54,0x0f,0x02,0x00}, {0x07,0x02,0x54,0x0e,0x02,0x00}}},
      {4, {{0x07,0x02,0x54,0x3f,0x02,0x00}, {0x07,0x02,0x54,0x3e,0x02,0x00}}},
   };
   const size_t pixrel_offset = sizeof(expected) - 10;
   for (unsigned i = 0; i < ARRAY_SIZE(native_pixrel); ++i) {
      const unsigned classes = native_pixrel[i].classes;
      uint8_t expected_variant[sizeof(expected)];
      memcpy(expected_variant, expected, sizeof(expected_variant));
      memcpy(expected_variant + pixrel_offset, native_pixrel[i].pixrel[0], 6);
      CHECK(agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
         UINT64_C(0x10000130640), &tail, classes, output,
         sizeof(expected), &written));
      CHECK(written == sizeof(expected));
      CHECK(memcmp(output, expected_variant, sizeof(expected_variant)) == 0);

      memcpy(expected_variant + pixrel_offset, native_pixrel[i].pixrel[1], 6);
      CHECK(agx_ioto_g17p_coverage_fragment_program_compose_trusted(
         UINT64_C(0x10000130640), NULL, 0, NULL, 0, &tail, classes,
         output, sizeof(expected), &written));
      CHECK(written == sizeof(expected));
      CHECK(memcmp(output, expected_variant, sizeof(expected_variant)) == 0);
      CHECK(output[sizeof(expected)] == 0xa5);
   }

   /* Independent native metadata for holes, including no attachment zero. */
   for (unsigned i = 0; i < sizeof(g17p_sparse_mrt_oracles) /
                            sizeof(g17p_sparse_mrt_oracles[0]); ++i) {
      unsigned release = g17p_sparse_mrt_oracles[i].loader_release;
      uint8_t expected_variant[sizeof(expected)];
      memcpy(expected_variant, expected, sizeof(expected_variant));
      expected_variant[pixrel_offset + 3] = release;
      expected_variant[pixrel_offset + 4] = release >> 8;
      CHECK(agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
         UINT64_C(0x10000130640), &tail,
         g17p_sparse_mrt_oracles[i].partition_mask, output,
         sizeof(expected), &written));
      CHECK(written == sizeof(expected));
      CHECK(!memcmp(output, expected_variant, sizeof(expected_variant)));
   }

   uint8_t before[sizeof(output)];
   memcpy(before, output, sizeof(output));
   written = 37;
   CHECK(!agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
      UINT64_C(0x10000130640), &tail, 0, output, sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 37);
   CHECK(!agx_ioto_g17p_coverage_fragment_program_compose_trusted(
      UINT64_C(0x10000130640), NULL, 0, NULL, 0, &tail, 8, output,
      sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 37);
   CHECK(!agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
      UINT64_C(0x10000130641), &tail, 1, output, sizeof(expected), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 37);
   CHECK(!agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
      UINT64_C(0x10000130640), &tail, 1, output, sizeof(expected) - 1,
      &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 37);
   return 0;
}

static int
test_g17p_resource_program(void)
{
   enum { GENERIC_RESOURCE_PROGRAM_SIZE = 144 };
   static const uint8_t tail_bytes[AGX_IOTO_STATE_TAIL_TYPE0_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00, 0x77, 0x01,
      0x2a, 0x01, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0xf7, 0x00,
      0x2a, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t pre_copy[] = {
      0x0c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x46, 0x00, 0x1c, 0x80,
      0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x67, 0x00, 0x54, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x40, 0x22, 0x00,
      0x06, 0x08, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
   };
   static const uint8_t copies[] = {
      0x2b, 0x00, 0x01, 0x04, 0x3b, 0x02, 0x01, 0x04,
      0x4b, 0x04, 0x01, 0x04, 0x5b, 0x06, 0x01, 0x04,
   };
   struct agx_ioto_state_tail_output tail = {
      .length = sizeof(tail_bytes),
   };
   memcpy(tail.bytes, tail_bytes, sizeof(tail_bytes));

   uint8_t output[GENERIC_RESOURCE_PROGRAM_SIZE + 1];
   memset(output, 0xa5, sizeof(output));
   size_t written = SIZE_MAX;
   CHECK(agx_ioto_g17p_resource_program_compose_trusted(
      UINT64_C(0x10001990240), pre_copy, sizeof(pre_copy), copies,
      sizeof(copies), &tail, output, sizeof(output) - 1, &written));
   CHECK(written == GENERIC_RESOURCE_PROGRAM_SIZE);
   CHECK(memcmp(output, pre_copy, sizeof(pre_copy)) == 0);
   CHECK(memcmp(output + sizeof(pre_copy), tail.bytes, tail.length) == 0);
   const size_t copies_offset = sizeof(pre_copy) + tail.length +
                                AGX_IOTO_G17P_DEFAULT_PROFILE_PREFIX_SIZE;
   CHECK(memcmp(output + copies_offset, copies, sizeof(copies)) == 0);
   static const uint8_t terminate[] = {0x0e, 0x00, 0x00, 0x00};
   CHECK(memcmp(output + written - sizeof(terminate), terminate,
                sizeof(terminate)) == 0);
   CHECK(output[written] == 0xa5);

   /* Native coverage/resource loaders can copy a completed load before the
    * final copy suffix. This has been accepted since the coverage loader
    * integration; only non-straight-line records are invalid here. */
   CHECK(agx_ioto_g17p_program_compose_with_preshader(
      UINT64_C(0x10001990240), copies, sizeof(copies), copies, sizeof(copies),
      &tail, AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE, 0, NULL, output,
      sizeof(output), &written));

   uint8_t before[sizeof(output)];
   memcpy(before, output, sizeof(output));
   written = 41;
   /* Copy instructions are now valid before the state tail as well as after
    * the profile prefix: preamble uniform stores use exactly that shape.
    * Truncate the final instruction instead so this remains a malformed-body
    * control and continues to verify transactional failure. */
   CHECK(!agx_ioto_g17p_program_compose_with_preshader(
      UINT64_C(0x10001990240), copies, sizeof(copies) - 1, copies,
      sizeof(copies), &tail, AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE, 0, NULL,
      output, sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 41);
   CHECK(!agx_ioto_g17p_program_compose_with_preshader(
      UINT64_C(0x10001990240), terminate, sizeof(terminate), copies, sizeof(copies),
      &tail, AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE, 0, NULL, output,
      sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 41);
   static const uint8_t nop[] = {0x06, 0x00};
   CHECK(!agx_ioto_g17p_program_compose_with_preshader(
      UINT64_C(0x10001990240), pre_copy, sizeof(pre_copy), nop, sizeof(nop),
      &tail, AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE, 0, NULL, output,
      sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 41);
   CHECK(!agx_ioto_g17p_resource_program_compose_trusted(
      UINT64_C(0x10001990240), pre_copy, sizeof(pre_copy), copies,
      sizeof(copies), &tail, output, GENERIC_RESOURCE_PROGRAM_SIZE - 1,
      &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 41);
   return 0;
}

static int
test_g17p_fragment_resource_program(void)
{
   /* Valid EXP-G1a fbuf1 fragment block, 0x10000130000 +0x640..+0x6f1. */
   static const uint8_t pre_copy[] = {
      0x2c, 0x80, 0x42, 0x00, 0x00, 0x04, 0x10, 0x00,
      0x3c, 0x80, 0x42, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x6c, 0xa0, 0x42, 0x00, 0x00, 0x00, 0x20, 0x00,
      0x7c, 0x80, 0x42, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x67, 0x00, 0x54, 0x24, 0x12, 0x00, 0x00, 0x00,
      0x57, 0x04, 0x00, 0x50, 0x26, 0x00, 0x67, 0x00,
      0x54, 0x2c, 0x16, 0x00, 0x00, 0x00, 0x59, 0x04,
      0x00, 0x40, 0x26, 0x00,
   };
   static const uint8_t tail_bytes[] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
      0x77, 0x01, 0xaa, 0x07, 0x00, 0x00, 0x00, 0x02,
      0x00, 0x00, 0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t copies[] = {
      0x0b, 0x00, 0x01, 0x04, 0x1b, 0x00, 0x01, 0x04,
      0x2b, 0x24, 0x09, 0x44, 0x3b, 0x26, 0x09, 0x04,
      0x4b, 0x28, 0x09, 0x04, 0x5b, 0x2a, 0x09, 0x04,
      0x6b, 0x2c, 0x09, 0x04, 0x7b, 0x2e, 0x09, 0x04,
   };
   static const uint8_t expected[] = {
      0x2c, 0x80, 0x42, 0x00, 0x00, 0x04, 0x10, 0x00,
      0x3c, 0x80, 0x42, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x6c, 0xa0, 0x42, 0x00, 0x00, 0x00, 0x20, 0x00,
      0x7c, 0x80, 0x42, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x67, 0x00, 0x54, 0x24, 0x12, 0x00, 0x00, 0x00,
      0x57, 0x04, 0x00, 0x50, 0x26, 0x00, 0x67, 0x00,
      0x54, 0x2c, 0x16, 0x00, 0x00, 0x00, 0x59, 0x04,
      0x00, 0x40, 0x26, 0x00,
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
      0x77, 0x01, 0xaa, 0x07, 0x00, 0x00, 0x00, 0x02,
      0x00, 0x00, 0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x14, 0x81, 0x11, 0x06, 0x00, 0x00, 0x00, 0x00,
      0x0c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8,
      0x10, 0x05, 0x1c, 0x80, 0x02, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x0f, 0x12, 0x54, 0x00, 0x4c, 0x00,
      0x0b, 0x00, 0x01, 0x04, 0x1b, 0x00, 0x01, 0x04,
      0x2b, 0x24, 0x09, 0x44, 0x3b, 0x26, 0x09, 0x04,
      0x4b, 0x28, 0x09, 0x04, 0x5b, 0x2a, 0x09, 0x04,
      0x6b, 0x2c, 0x09, 0x04, 0x7b, 0x2e, 0x09, 0x04,
      0x07, 0x02, 0x54, 0xf3, 0x02, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_state_tail_output tail = {
      .length = sizeof(tail_bytes),
   };
   memcpy(tail.bytes, tail_bytes, sizeof(tail_bytes));

   uint8_t output[sizeof(expected) + 1];
   memset(output, 0xa5, sizeof(output));
   size_t written = 0;
   CHECK(agx_ioto_g17p_fragment_resource_program_compose_trusted(
      UINT64_C(0x10000130640), pre_copy, sizeof(pre_copy), copies,
      sizeof(copies), &tail, 1, output, sizeof(expected), &written));
   CHECK(written == sizeof(expected));
   CHECK(memcmp(output, expected, sizeof(expected)) == 0);
   CHECK(output[sizeof(expected)] == 0xa5);
   /* Reusing the scheduled loads must not reuse the ordinary FS release.
    * Native BG/partial-BG use 0x200 while ordinary FS uses 0x2f3. The full
    * composed programs differ in exactly this one descriptor byte.
    */
   uint8_t bg_expected[sizeof(expected)];
   memcpy(bg_expected, expected, sizeof(expected));
   bg_expected[sizeof(expected) - 7] = 0;
   CHECK(agx_ioto_g17p_background_resource_program_compose_trusted(
      UINT64_C(0x10000130640), pre_copy, sizeof(pre_copy), copies,
      sizeof(copies), &tail, output, sizeof(expected), &written));
   CHECK(written == sizeof(expected));
   CHECK(memcmp(output, bg_expected, sizeof(expected)) == 0);
   CHECK(output[sizeof(expected)] == 0xa5);

   uint8_t before[sizeof(output)];
   memcpy(before, output, sizeof(output));
   written = 41;
#define BG_REJECT(va, pre, pre_size, cp, cp_size, state, cap)                  \
   do {                                                                       \
      CHECK(!agx_ioto_g17p_background_resource_program_compose_trusted(        \
         va, pre, pre_size, cp, cp_size, state, output, cap, &written));        \
      CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 41);     \
   } while (0)
   /* The trusted form skips record validation; malformed bodies are rejected
    * by the validating composer, which shares the same span checks. */
#define BODY_REJECT(pre, pre_size, cp, cp_size)                                \
   do {                                                                       \
      CHECK(!agx_ioto_g17p_program_compose_with_preshader(                     \
         UINT64_C(0x10000130640), pre, pre_size, cp, cp_size, &tail,           \
         AGX_IOTO_G17P_PROGRAM_FRAGMENT, 1, NULL, output, sizeof(expected),    \
         &written));                                                          \
      CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 41);     \
   } while (0)
   BG_REJECT(0, pre_copy, sizeof(pre_copy), copies, sizeof(copies), &tail,
             sizeof(expected));
   BG_REJECT(UINT64_C(0x10000130641), pre_copy, sizeof(pre_copy), copies,
             sizeof(copies), &tail, sizeof(expected));
   BG_REJECT(UINT64_C(0x10000130640), pre_copy, sizeof(pre_copy), copies,
             sizeof(copies), &tail, sizeof(expected) - 1);
   BG_REJECT(UINT64_C(0x10000130640), pre_copy, sizeof(pre_copy), copies,
             sizeof(copies), NULL, sizeof(expected));
   BODY_REJECT(copies, sizeof(copies) - 1, copies, sizeof(copies));
   static const uint8_t bad_pre[] = {0x0e, 0x00, 0x00, 0x00};
   BODY_REJECT(bad_pre, sizeof(bad_pre), copies, sizeof(copies));
   static const uint8_t bad_copy[] = {0x06, 0x00};
   BODY_REJECT(pre_copy, sizeof(pre_copy), bad_copy, sizeof(bad_copy));
   BG_REJECT(UINT64_C(0x10000130640), NULL, sizeof(pre_copy), copies,
             sizeof(copies), &tail, sizeof(expected));
   BODY_REJECT(pre_copy, sizeof(pre_copy), copies, sizeof(copies) - 1);
#undef BODY_REJECT
#undef BG_REJECT
   return 0;
}


static int
test_g17s_minimal_compute_program(void)
{
   static const uint8_t expected[] = {
      0x2c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x3c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xcc, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xdc, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xec, 0xa0, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00,
      0xfc, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xaf, 0x04, 0x03, 0x00, 0xa8, 0x0c, 0x00, 0x40,
      0xcf, 0x18, 0x03, 0x00, 0x88, 0x04, 0x00, 0xc0,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xef, 0x1c, 0x03, 0x00, 0xc8, 0x0c, 0x00, 0x40,
      0x07, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00,
      0xc7, 0x06, 0x40, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x07, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x34, 0x81, 0x01, 0x00, 0x20, 0x00, 0xa0, 0x02,
      0x0c, 0x80, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x27, 0x04, 0x04, 0x1a, 0x21, 0x00, 0xa3, 0x02,
      0x88, 0x80, 0x00, 0x00,
      0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x27, 0x04, 0x04, 0x00, 0x30, 0x00, 0xa4, 0x12,
      0x2b, 0x00, 0x01, 0x04, 0x3b, 0x00, 0x01, 0x04,
      0x0b, 0x14, 0x09, 0x44, 0x1b, 0x16, 0x09, 0x04,
      0x4b, 0x18, 0x09, 0x04, 0x5b, 0x1c, 0x09, 0x04,
      0x6b, 0x1e, 0x09, 0x04, 0x7b, 0x20, 0x09, 0x04,
      0x0e, 0x00, 0x00, 0x00,
   };
   CHECK(sizeof(expected) == AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE);

   uint8_t output[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE + 1];
   memset(output, 0xa5, sizeof(output));
   size_t written = 0;
   CHECK(agx_ioto_g17s_minimal_compute_program_compose(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), output,
      sizeof(output), &written));
   CHECK(written == sizeof(expected));
   CHECK(memcmp(output, expected, sizeof(expected)) == 0);
   CHECK(output[sizeof(expected)] == 0xa5);

   const struct agx_ioto_g17s_minimal_compute_resources resources = {
      .gather_buffer_va = UINT64_C(0x100000e1ba0),
      .user_compute_argument_gpu_va_1 = UINT64_C(0x10000030100),
      .absolute_uniform_data_va =
         AGX_IOTO_G17S_MINIMAL_COMPUTE_WITNESS_ABSOLUTE_VA,
   };
   static const uint8_t expected_live_resources[48] = {
      0x2c, 0xa0, 0x02, 0x00, 0x0e, 0x0c, 0x70, 0x00,
      0x3c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xcc, 0x80, 0x02, 0x00, 0x04, 0x00, 0x1c, 0x00,
      0xdc, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0xec, 0xa0, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00,
      0xfc, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
   };
   uint8_t live[sizeof(output)];
   memset(live, 0x3c, sizeof(live));
   size_t live_written = SIZE_MAX;
   CHECK(agx_ioto_g17s_minimal_compute_program_compose_resources(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), &resources, live,
      sizeof(live), &live_written));
   CHECK(live_written == sizeof(expected));
   CHECK(memcmp(live, expected_live_resources,
                sizeof(expected_live_resources)) == 0);
   CHECK(memcmp(live + sizeof(expected_live_resources),
                expected + sizeof(expected_live_resources),
                sizeof(expected) - sizeof(expected_live_resources)) == 0);
   CHECK(live[live_written] == 0x3c);

   struct agx_ioto_g17s_minimal_compute_resources alternate_resources =
      resources;
   alternate_resources.absolute_uniform_data_va = UINT64_C(0x10000018030);
   CHECK(agx_ioto_g17s_minimal_compute_program_compose_resources(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0),
      &alternate_resources, live, sizeof(live), &live_written));
   CHECK(live[32] == 0xec && live[33] == 0xb0);
   CHECK(memcmp(live, expected_live_resources, 32) == 0);
   CHECK(memcmp(live + 48, expected + 48, sizeof(expected) - 48) == 0);

   uint8_t live_before[sizeof(live)];
   memcpy(live_before, live, sizeof(live));
   struct agx_ioto_g17s_minimal_compute_resources invalid_resources =
      resources;
   invalid_resources.absolute_uniform_data_va = 0;
   live_written = 23;
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose_resources(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), &invalid_resources,
      live, sizeof(live), &live_written));
   CHECK(memcmp(live, live_before, sizeof(live)) == 0 && live_written == 23);
   invalid_resources = resources;
   invalid_resources.gather_buffer_va = 0;
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose_resources(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), &invalid_resources,
      live, sizeof(live), &live_written));
   CHECK(memcmp(live, live_before, sizeof(live)) == 0 && live_written == 23);
   invalid_resources = resources;
   invalid_resources.user_compute_argument_gpu_va_1 = 0;
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose_resources(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), &invalid_resources,
      live, sizeof(live), &live_written));
   CHECK(memcmp(live, live_before, sizeof(live)) == 0 && live_written == 23);
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose_resources(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), NULL, live,
      sizeof(live), &live_written));
   CHECK(memcmp(live, live_before, sizeof(live)) == 0 && live_written == 23);

   uint8_t before[sizeof(output)];
   memcpy(before, output, sizeof(output));
   written = 29;
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose(
      UINT64_C(0x100000041), UINT64_C(0x100000006c0), output,
      sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 29);
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose(
      UINT64_C(0x100000040), UINT64_C(0x100000006c1), output,
      sizeof(output), &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 29);
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), output,
      sizeof(expected) - 1, &written));
   CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 29);
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), NULL,
      sizeof(output), &written));
   CHECK(!agx_ioto_g17s_minimal_compute_program_compose(
      UINT64_C(0x100000040), UINT64_C(0x100000006c0), output,
      sizeof(output), NULL));
   return 0;
}

static int
test_g17g_graphics_no_resource_programs(void)
{
   /* M5 25G76 programs from the native triangle's complete context image.
    * These exact programs also rendered three consecutive Linux frames.
    * Bundle SHA256: 915c3c1e1bf7c679bd95f552a1a6a41d4eba86048a3a746a2d3a2ecbb1aa708c.
    * Vertex VA0x10000160000 and fragment VA0x10000170700. */
   static const uint8_t vertex[] = {
      0x07,0x00,0x00,0x0c,0x00,0x00,0x00,0x00,
      0x87,0x04,0x48,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
      0x87,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00,
      0x1c,0x80,0x02,0x00,0x00,0x00,0x00,0x00,
      0x34,0x81,0x01,0x00,0x20,0x00,0xa0,0x02,
      0x0c,0x80,0x02,0x00,0x08,0x00,0x00,0x00,
      0x27,0x04,0x04,0x1a,0x21,0x00,0xa3,0x02,0x88,0x80,0x00,0x00,
      0x1c,0x80,0x02,0x00,0x04,0x00,0x00,0x00,
      0x27,0x04,0x04,0x00,0x30,0x00,0xa4,0x12,
      0x0e,0x00,0x00,0x00,
   };
   static const uint8_t fragment[] = {
      0x17,0x00,0x00,0x2c,0x00,0x00,0x18,0x08,
      0xc7,0x03,0x40,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
      0x07,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00,
      0x1c,0x80,0x02,0x00,0x00,0x00,0x00,0x00,
      0x34,0x81,0x01,0x00,0x20,0x00,0xa0,0x02,
      0x0c,0x80,0x02,0x00,0x04,0x00,0x00,0x00,
      0x27,0x04,0x04,0x1a,0x21,0x00,0xa3,0x02,0x88,0x80,0x00,0x00,
      0x1c,0x80,0x02,0x00,0x04,0x00,0x00,0x00,
      0x27,0x04,0x04,0x00,0x30,0x00,0xa4,0x12,
      0x9f,0x17,0x00,0x04,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   CHECK(sizeof(vertex) == AGX_IOTO_G17G_VERTEX_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(sizeof(fragment) == AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE);
   /* GPU1 loads followed by one final COPY round. The native M5 EOT places
    * its 80-byte state/profile prefix between those spans. The full private
    * 362-byte EOT witness also reconstructs byte for byte with this API. */
   static const uint8_t body[] = {
      0x2c,0x80,0x02,0x00,0x00,0x00,0x80,0x00,
      0x3c,0x80,0x02,0x00,0x04,0x00,0x00,0x00,
      0x0b,0x24,0x09,0x04,
   };
   static const uint8_t background_suffix[] = {7, 0x10, 0, 4, 0, 0, 14, 0, 0, 0};
   /* Native M5 punch-through fragment Ioto, captured from the `discard` mode
    * of gpu/m5-state-matrix-reference.m: one bit of the release record
    * against the opaque fragment above, nothing else. */
   static const uint8_t coverage_suffix[] = {0x97, 0x17, 0, 4, 0, 0, 14, 0, 0, 0};
   static const enum agx_ioto_g17g_program_stage stages[] = {
      AGX_IOTO_G17G_PROGRAM_VERTEX,
      AGX_IOTO_G17G_PROGRAM_FRAGMENT,
      AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT,
      AGX_IOTO_G17G_PROGRAM_EOT,
      AGX_IOTO_G17G_PROGRAM_BACKGROUND,
   };
   for (unsigned index = 0; index < ARRAY_SIZE(stages); ++index) {
      const enum agx_ioto_g17g_program_stage stage = stages[index];
      const bool vertex_stage = stage == AGX_IOTO_G17G_PROGRAM_VERTEX;
      const bool releases_pixel =
         stage == AGX_IOTO_G17G_PROGRAM_FRAGMENT ||
         stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT ||
         stage == AGX_IOTO_G17G_PROGRAM_BACKGROUND;
      const uint8_t *base = vertex_stage ? vertex : fragment;
      const size_t suffix = releases_pixel ? 10 : 4;
      const size_t size = sizeof(body) + 80 + suffix;
      struct agx_ioto_state_tail_output tail = {
         .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
      };
      memcpy(tail.bytes, base, tail.length);
      uint8_t output[128], before[128], alias[128];
      memset(output, 0xa5, sizeof(output));
      size_t written = 23;
      CHECK(agx_ioto_g17g_resource_program_compose_classes(
         UINT64_C(0x10000160000), body, sizeof(body), &tail, stage, 1,
         output, sizeof(output), &written));
      CHECK(written == size);
      CHECK(memcmp(output, body, 16) == 0);
      CHECK(memcmp(output + 16, base, 80) == 0);
      CHECK(memcmp(output + 96, body + 16, 4) == 0);
      const uint8_t *expected_suffix =
         stage == AGX_IOTO_G17G_PROGRAM_BACKGROUND ? background_suffix
         : stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT ? coverage_suffix
         : base + (vertex_stage ? sizeof(vertex) : sizeof(fragment)) - suffix;
      CHECK(memcmp(output + 100, expected_suffix, suffix) == 0);
      for (size_t i = size; i < sizeof(output); ++i)
         CHECK(output[i] == 0xa5);

      memcpy(alias, body, sizeof(body));
      CHECK(agx_ioto_g17g_resource_program_compose_classes(
         UINT64_C(0x10000160000), alias, sizeof(body), &tail, stage, 1,
         alias, sizeof(alias), &written));
      CHECK(written == size && memcmp(alias, output, size) == 0);

      memcpy(before, output, sizeof(output));
      written = 23;
      CHECK(!agx_ioto_g17g_resource_program_compose_classes(
         UINT64_C(0x10000160000), body, sizeof(body), &tail, stage, 1,
         output, size - 1, &written));
      CHECK(!agx_ioto_g17g_resource_program_compose_classes(
         UINT64_C(0x10000160000), body, sizeof(body) - 1, &tail, stage, 1,
         output, sizeof(output), &written));
      CHECK(!agx_ioto_g17g_resource_program_compose_classes(
         UINT64_C(0x10000160000), body, sizeof(body), &tail, 99, 1,
         output, sizeof(output), &written));
      CHECK(!agx_ioto_g17g_resource_program_compose_classes(
         UINT64_C(0x10000160000), fragment + 80, 6, &tail, stage, 1,
         output, sizeof(output), &written));
      CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 23);
   }
   for (unsigned stage = 0; stage < 2; ++stage) {
      const uint8_t *expected = stage ? fragment : vertex;
      const size_t size = stage ? sizeof(fragment) : sizeof(vertex);
      struct agx_ioto_state_tail_output tail = {
         .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
      };
      memcpy(tail.bytes, expected, tail.length);
      uint8_t output[96], before[96];
      memset(output, 0xa5, sizeof(output));
      size_t written = 19;
      CHECK(agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0x10000160000), &tail, stage, output, sizeof(output), &written));
      CHECK(written == size && memcmp(output, expected, size) == 0);
      for (size_t i = size; i < sizeof(output); ++i)
         CHECK(output[i] == 0xa5);
      memcpy(before, output, sizeof(output));
      written = 19;
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0x10000160001), &tail, stage, output, sizeof(output), &written));
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         0, &tail, stage, output, sizeof(output), &written));
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0xffffffffffc0), &tail, stage, output, sizeof(output), &written));
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0x10000160000), &tail, stage, output, size - 1, &written));
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0x10000160000), NULL, stage, output, sizeof(output), &written));
      tail.bytes[8] = 7;
      tail.bytes[9] = 0;
      memset(tail.bytes + 14, 0, 4);
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0x10000160000), &tail, stage, output, sizeof(output), &written));
      CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 19);
      memcpy(tail.bytes, expected, tail.length);
      static const uint8_t neo_speclm[] = {0x77,0x00,0x2a,0x41,0,0,0,0};
      memcpy(tail.bytes, neo_speclm, sizeof(neo_speclm));
      CHECK(!agx_ioto_g17g_no_resource_program_compose(
         UINT64_C(0x10000160000), &tail, stage, output, sizeof(output), &written));
      CHECK(memcmp(output, before, sizeof(output)) == 0 && written == 19);
   }
   return 0;
}

/* The profile loader's LDIMM loads the byte offset of the helper-table group
 * its stage calls, where the USC window's lead puts it for the device's
 * cluster count: 0x100 a cluster, twice that for vertex. Eight clusters (the
 * M5 Ultra) need 0x800 and 0x1000, past what the template's raw byte can
 * carry; a count of 0 is refused. */
static int
test_m5_profile_loader_table_base(void)
{
   static const uint8_t body[] = {
      0x2c,0x80,0x02,0x00,0x00,0x00,0x80,0x00,
      0x3c,0x80,0x02,0x00,0x04,0x00,0x00,0x00,
      0x0b,0x24,0x09,0x04,
   };
   static const uint8_t tail_bytes[2][AGX_IOTO_STATE_TAIL_TYPE0_SIZE] = {
      {0x07,0x00,0x00,0x0c,0x00,0x00,0x00,0x00,
       0x87,0x04,0x48,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
       0x87,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00},
      {0x17,0x00,0x00,0x2c,0x00,0x00,0x18,0x08,
       0xc7,0x03,0x40,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
       0x07,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00},
   };
   static const struct {
      enum agx_chip chip;
      unsigned clusters;
      uint32_t fragment, vertex;
   } cases[] = {
      {AGX_CHIP_G17G, 1, 0x100, 0x200},
      {AGX_CHIP_G17S, 2, 0x200, 0x400},
      {AGX_CHIP_G17C, 4, 0x400, 0x800},
      {AGX_CHIP_G17D, 8, 0x800, 0x1000},
   };
   const uint64_t field = agx_ioto_load_ldimm_long_immediate(UINT32_MAX);
   /* The template's LDIMM destination, with the immediate cleared. */
   const uint64_t dest = UINT64_C(0x0002800c);

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      size_t copy_offset;
      CHECK(agx_ioto_program_first_copy_offset(cases[i].chip, body,
                                               sizeof(body), &copy_offset));
      for (unsigned fragment = 0; fragment < 2; ++fragment) {
         struct agx_ioto_state_tail_output tail = {
            .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
         };
         memcpy(tail.bytes, tail_bytes[fragment], tail.length);
         uint8_t output[160];
         size_t written = 0;
         const enum agx_ioto_g17g_program_stage stage =
            fragment ? AGX_IOTO_G17G_PROGRAM_FRAGMENT
                     : AGX_IOTO_G17G_PROGRAM_VERTEX;
         CHECK(!agx_ioto_m5_resource_program_compose_classes(
            cases[i].chip, 0, UINT64_C(0x10000160000), body, sizeof(body),
            &tail, stage, 1, output, sizeof(output), &written));
         CHECK(agx_ioto_m5_resource_program_compose_classes(
            cases[i].chip, cases[i].clusters, UINT64_C(0x10000160000), body,
            sizeof(body), &tail, stage, 1, output, sizeof(output),
            &written));
         const uint8_t *ldimm =
            output + copy_offset + AGX_IOTO_STATE_TAIL_TYPE0_SIZE + 16;
         uint64_t word = 0;
         for (unsigned b = 0; b < 8; ++b)
            word |= (uint64_t)ldimm[b] << (8 * b);
         CHECK((word & ~field) == dest);
         CHECK((word & field) ==
               agx_ioto_load_ldimm_long_immediate(
                  fragment ? cases[i].fragment : cases[i].vertex));
      }
   }
   return 0;
}


int
main(void)
{
   CHECK(test_exact_order() == 0);
   CHECK(test_terminal_suffix_modes() == 0);
   CHECK(test_fail_closed() == 0);
   CHECK(test_rejects_composer_owned_and_pixel_records() == 0);
   CHECK(test_empty_body_and_alias() == 0);
   CHECK(test_g17p_no_resource_program() == 0);
   CHECK(test_g17p_fragment_no_resource_program() == 0);
   CHECK(test_g17p_resource_program() == 0);
   CHECK(test_g17p_fragment_resource_program() == 0);
   CHECK(test_g17s_minimal_compute_program() == 0);
   CHECK(test_g17g_graphics_no_resource_programs() == 0);
   CHECK(test_m5_profile_loader_table_base() == 0);
   printf("agx_ioto_program_core: %llu checks passed\n",
          (unsigned long long)checks);
   return 0;
}
