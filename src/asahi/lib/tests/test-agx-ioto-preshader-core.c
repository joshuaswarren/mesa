/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "../agx_ioto_preshader_core.h"

#include "../agx_ioto_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const enum agx_chip gpu0_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
};

static const enum agx_chip gpu1_targets[] = {
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,
   AGX_CHIP_G17D,
};

static bool
all_bytes(const void *data, size_t size, uint8_t value)
{
   const uint8_t *bytes = data;
   for (size_t i = 0; i < size; ++i) {
      if (bytes[i] != value)
         return false;
   }
   return true;
}

static int
check_bound_range(void)
{
   uint8_t binary[512] = {0};
   const size_t offset = 128, size = 258;
   const uint64_t base = UINT64_C(0x123456789a3c);
   struct agx_ioto_preshader_desc desc = {
      .force_external = true,
      .request_prelude = true,
      .suppress_prelude = true,
   };
   CHECK(agx_ioto_preshader_bind_range(binary, sizeof(binary), base,
                                       sizeof(binary), offset, size, &desc));
   CHECK(desc.code == binary + offset && desc.code_size == size);
   CHECK(desc.external_va == UINT64_C(0x123456789abc));
   CHECK(desc.force_external && desc.request_prelude && desc.suppress_prelude);

   /* Independent external B fixture: includes the original binary's offset,
    * and keeps the caller's prelude suppression. These are range/packing
    * checks, not execution of the synthetic buffer contents. */
   static const uint8_t branch[] = {
      0x0f, 0x00, 0x54, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x00,
   };
   struct agx_ioto_preshader_output output;
   CHECK(agx_ioto_preshader_emit(AGX_CHIP_G17P, &desc, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_EXTERNAL);
   CHECK(output.length == sizeof(branch));
   CHECK(memcmp(output.bytes, branch, sizeof(branch)) == 0);

   struct {
      size_t cpu_size, gpu_size, offset, size;
      uint64_t base;
   } invalid[] = {
      {512, 512, 128, 0, base},
      {126, 512, 128, 4, base},
      {130, 512, 128, 4, base},
      {512, 126, 128, 4, base},
      {512, 130, 128, 4, base},
      {512, 512, 129, 4, base},
      {512, 512, 128, 3, base},
      {512, 512, 128, 4, base | 1},
      {512, 512, 128, 4, AGX_IOTO_PRESHADER_NO_VA},
      {512, 512, 128, 4, UINT64_C(1) << 48},
      {512, 512, 128, 4, (UINT64_C(1) << 48) - 128},
      {512, 512, 128, 4, (UINT64_C(1) << 48) - 130},
      {SIZE_MAX, SIZE_MAX, SIZE_MAX - 1, 4, base},
      {SIZE_MAX, SIZE_MAX, 2, SIZE_MAX - 1, base},
   };
   uint8_t saved[sizeof(desc)];
   memcpy(saved, &desc, sizeof(desc));
   for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      CHECK(!agx_ioto_preshader_bind_range(
         binary, invalid[i].cpu_size, invalid[i].base, invalid[i].gpu_size,
         invalid[i].offset, invalid[i].size, &desc));
      CHECK(memcmp(saved, &desc, sizeof(desc)) == 0);
   }
   CHECK(!agx_ioto_preshader_bind_range(NULL, 512, base, 512, 128, 4, &desc));
   CHECK(memcmp(saved, &desc, sizeof(desc)) == 0);
   CHECK(!agx_ioto_preshader_bind_range(binary, 512, base, 512, 128, 4, NULL));

   /* Both buffer ends and the top of the GPU address domain are inclusive
    * bounds for the final byte, not reasons to discard a valid range. */
   CHECK(agx_ioto_preshader_bind_range(binary, 132,
                                       (UINT64_C(1) << 48) - 132,
                                       132, 128, 4, &desc));
   CHECK(desc.external_va == (UINT64_C(1) << 48) - 4);
   CHECK(desc.code == binary + 128 && desc.code_size == 4);
   return 0;
}

static int
check_branch(enum agx_chip chip, bool gpu1)
{
   static const uint64_t addresses[] = {
      UINT64_C(0),
      UINT64_C(2),
      UINT64_C(0x123456789abc),
      UINT64_C(0xfffffffffffe),
   };

   for (size_t i = 0; i < sizeof(addresses) / sizeof(addresses[0]); ++i) {
      for (unsigned link = 0; link < 2; ++link) {
         uint8_t bytes[AGX_IOTO_PRESHADER_BRANCH_SIZE];
         memset(bytes, 0xa5, sizeof(bytes));
         CHECK(agx_ioto_preshader_pack_branch(chip, addresses[i], link, bytes));

         const enum agx_ioto_kind kind =
            gpu1 ? (link ? AGX_IOTO_KIND_BL_EP1 : AGX_IOTO_KIND_B_EP1)
                 : (link ? AGX_IOTO_KIND_BL_EP0 : AGX_IOTO_KIND_B_EP0);
         struct agx_ioto_record record;
         size_t consumed;
         CHECK(agx_ioto_validate_kind(chip, kind, bytes, sizeof(bytes), &record,
                                      &consumed));
         CHECK(consumed == sizeof(bytes));
      }
   }

   uint8_t unchanged[AGX_IOTO_PRESHADER_BRANCH_SIZE];
   memset(unchanged, 0xa5, sizeof(unchanged));
   CHECK(!agx_ioto_preshader_pack_branch(chip, 1, false, unchanged));
   CHECK(all_bytes(unchanged, sizeof(unchanged), 0xa5));
   CHECK(!agx_ioto_preshader_pack_branch(chip, UINT64_C(1) << 48, false,
                                         unchanged));
   CHECK(all_bytes(unchanged, sizeof(unchanged), 0xa5));
   return 0;
}

static int
check_branch_witnesses(void)
{
   static const uint8_t gpu0_b[] = {
      0x0f, 0x00, 0x54, 0xa8, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00,
   };
   static const uint8_t gpu1_b[] = {
      0x5e, 0x0f, 0xf3, 0x1e, 0x1f, 0x00, 0xff, 0xc7, 0xff, 0x7f,
   };

   uint8_t bytes[AGX_IOTO_PRESHADER_BRANCH_SIZE];
   CHECK(agx_ioto_preshader_pack_branch(AGX_CHIP_G15G, UINT64_C(0xffffffffffa8),
                                        false, bytes));
   CHECK(memcmp(bytes, gpu0_b, sizeof(bytes)) == 0);
   CHECK(agx_ioto_preshader_pack_branch(AGX_CHIP_G15G, UINT64_C(0xffffffffffa8),
                                        true, bytes));
   CHECK(bytes[0] == 0x8f);
   CHECK(memcmp(bytes + 1, gpu0_b + 1, sizeof(bytes) - 1) == 0);

   CHECK(agx_ioto_preshader_pack_branch(AGX_CHIP_G17G, UINT64_C(0xffffffffffa0),
                                        false, bytes));
   CHECK(memcmp(bytes, gpu1_b, sizeof(bytes)) == 0);
   CHECK(agx_ioto_preshader_pack_branch(AGX_CHIP_G17G, UINT64_C(0xffffffffffa0),
                                        true, bytes));
   CHECK(bytes[0] == 0x7e);
   CHECK(memcmp(bytes + 1, gpu1_b + 1, sizeof(bytes) - 1) == 0);
   return 0;
}

static int
check_inline(enum agx_chip chip)
{
   uint8_t code[AGX_IOTO_PRESHADER_INLINE_LIMIT] = {0x1b, 0x0c, 0x01, 0x08};
   code[250] = 0x06;
   code[251] = 0x00;
   code[252] = 0x06;
   code[253] = 0x00;
   code[254] = 0x06;
   code[255] = 0x00;

   struct agx_ioto_preshader_desc desc = {
      .code = code,
      .code_size = sizeof(code),
      .external_va = UINT64_C(0x123456789abc),
   };
   struct agx_ioto_preshader_output output;
   memset(&output, 0xa5, sizeof(output));
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_INLINE);
   CHECK(output.length == 250);
   CHECK(output.branch_offset == AGX_IOTO_PRESHADER_NO_BRANCH);
   CHECK(memcmp(output.bytes, code, output.length) == 0);

   const uint8_t all_nops[] = {0x06, 0x00, 0x06, 0x00};
   desc.code = all_nops;
   desc.code_size = sizeof(all_nops);
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_ABSENT);
   CHECK(output.length == AGX_IOTO_PRESHADER_ABSENT_SIZE);
   CHECK(memcmp(output.bytes, "\x0e\x00\x00\x00", 4) == 0);

   desc.code = NULL;
   desc.code_size = 0;
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_ABSENT);

   memset(&output, 0xa5, sizeof(output));
   desc.code = code;
   desc.code_size = 3;
   CHECK(!agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(all_bytes(&output, sizeof(output), 0xa5));
   return 0;
}

static int
check_external(enum agx_chip chip, bool gpu1)
{
   uint8_t code[AGX_IOTO_PRESHADER_INLINE_LIMIT + 2] = {0};
   struct agx_ioto_preshader_desc desc = {
      .code = code,
      .code_size = sizeof(code),
      .external_va = UINT64_C(0x123456789abc),
      .request_prelude = true,
   };
   struct agx_ioto_preshader_output output;
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_EXTERNAL);
   CHECK(output.length ==
         AGX_IOTO_PRESHADER_PRELUDE_SIZE + AGX_IOTO_PRESHADER_BRANCH_SIZE);
   CHECK(output.branch_offset == AGX_IOTO_PRESHADER_PRELUDE_SIZE);
   CHECK(output.bytes[0] == (gpu1 ? 0x04 : 0x0c));
   CHECK(output.bytes[1] == 0x10);

   struct agx_ioto_record record;
   size_t consumed;
   CHECK(agx_ioto_validate_kind(
      chip, gpu1 ? AGX_IOTO_KIND_B_EP1 : AGX_IOTO_KIND_B_EP0,
      output.bytes + output.branch_offset, output.length - output.branch_offset,
      &record, &consumed));
   CHECK(consumed == AGX_IOTO_PRESHADER_BRANCH_SIZE);

   desc.suppress_prelude = true;
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.length == AGX_IOTO_PRESHADER_BRANCH_SIZE);
   CHECK(output.branch_offset == 0);

   desc.suppress_prelude = false;
   desc.request_prelude = false;
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.length == AGX_IOTO_PRESHADER_BRANCH_SIZE);
   CHECK(output.branch_offset == 0);

   desc.code_size = 2;
   desc.force_external = true;
   desc.external_va = AGX_IOTO_PRESHADER_NO_VA;
   CHECK(agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_ABSENT);
   CHECK(output.length == AGX_IOTO_PRESHADER_ABSENT_SIZE);

   memset(&output, 0xa5, sizeof(output));
   desc.external_va = 1;
   CHECK(!agx_ioto_preshader_emit(chip, &desc, &output));
   CHECK(all_bytes(&output, sizeof(output), 0xa5));
   return 0;
}

static int
check_constant_program(enum agx_chip chip)
{
   const uint64_t limit = UINT64_C(1) << 48;
   struct agx_ioto_constant_program program = {
      .code_size = 4,
      .force_external = true,
   };
   static const struct {
      uint64_t suffix_va, code_va, displacement;
      bool prelude, suppress;
   } relocations[] = {
      {0x1000, 0x2000, 0x1000, false, false},
      {0x1000, 0x2000, 0x0ffe, true, false},
      {0x1000, 0x2000, 0x1000, true, true},
      {0x10000, 0x8000, UINT64_C(0xffffffff8000), false, false},
      {0x10000, 0x8000, UINT64_C(0xffffffff7ffe), true, false},
      {2, UINT64_C(0x800000000000), UINT64_C(0x7ffffffffffe), false, false},
      {UINT64_C(0x800000000002), 2, UINT64_C(0x800000000000), false, false},
      {UINT64_C(0xfffffffffff6), UINT64_C(0xffffffffffee),
       UINT64_C(0xfffffffffff8), false, false},
   };
   for (size_t i = 0; i < sizeof(relocations) / sizeof(relocations[0]); ++i) {
      program.code_va = relocations[i].code_va;
      program.request_prelude = relocations[i].prelude;
      program.suppress_prelude = relocations[i].suppress;
      struct agx_ioto_preshader_output output;
      CHECK(agx_ioto_constant_program_emit(
         chip, &program, relocations[i].suffix_va, &output));
      CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_EXTERNAL);
      CHECK(output.branch_offset ==
            (program.request_prelude && !program.suppress_prelude ? 2u : 0u));
      CHECK(output.length == output.branch_offset + 10);
      uint8_t expected[AGX_IOTO_PRESHADER_BRANCH_SIZE];
      CHECK(agx_ioto_preshader_pack_branch(
         chip, relocations[i].displacement, false, expected));
      CHECK(memcmp(output.bytes + output.branch_offset,
                   expected, sizeof(expected)) == 0);
   }

   /* An inline program exactly at the native threshold is copied and trimmed;
    * one halfword above it requires an external target, even with CPU bytes. */
   uint8_t code[AGX_IOTO_PRESHADER_INLINE_LIMIT + 2] = {0x0e, 0, 0, 0};
   for (size_t i = 4; i < sizeof(code); i += 2)
      code[i] = 0x06;
   program = (struct agx_ioto_constant_program){
      .code = code,
      .code_size = AGX_IOTO_PRESHADER_INLINE_LIMIT,
   };
   struct agx_ioto_preshader_output output;
   CHECK(agx_ioto_constant_program_emit(chip, &program, 0x1000, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_INLINE);
   CHECK(output.length == 4);
   CHECK(memcmp(output.bytes, code, 4) == 0);
   CHECK(agx_ioto_constant_program_emit(chip, &program, limit - 4, &output));
   program.code_size += 2;
   memset(&output, 0xa5, sizeof(output));
   CHECK(!agx_ioto_constant_program_emit(chip, &program, 0x1000, &output));
   CHECK(all_bytes(&output, sizeof(output), 0xa5));
   program.code_va = 0x2000;
   CHECK(agx_ioto_constant_program_emit(chip, &program, 0x1000, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_EXTERNAL);

   /* Zero extent is absent even if a stale address or policy flag is present. */
   program.code_size = 0;
   program.force_external = true;
   CHECK(agx_ioto_constant_program_emit(chip, &program, 0x1000, &output));
   CHECK(output.mode == AGX_IOTO_PRESHADER_MODE_ABSENT);
   CHECK(output.length == 4);
   return 0;
}

static int
check_g17p_absolute_constant_program(void)
{
   /* Native finish()/B_0 emitter witness: code VA 0x90000 with f11=1.
    * The relative control-flow form has the same target bits but f11=0. */
   static const uint8_t expected[] = {
      0x0f, 0x00, 0x54, 0x01, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00,
   };
   struct agx_ioto_constant_program program = {
      .code_size = 4, .code_va = 0x90000, .force_external = true,
      .absolute_branch = true,
   };
   struct agx_ioto_preshader_output a, b;
   CHECK(agx_ioto_constant_program_emit(AGX_CHIP_G17P, &program, 0x1000, &a));
   CHECK(a.length == sizeof(expected));
   CHECK(memcmp(a.bytes, expected, sizeof(expected)) == 0);

   for (unsigned bit = 1; bit < 48; ++bit) {
      program.code_va = UINT64_C(1) << bit;
      for (unsigned prelude = 0; prelude < 2; ++prelude) {
         program.request_prelude = prelude;
         CHECK(agx_ioto_constant_program_emit(
            AGX_CHIP_G17P, &program, 0x1000, &a));
         CHECK(agx_ioto_constant_program_emit(
            AGX_CHIP_G17P, &program, UINT64_C(0x900000000000), &b));
         CHECK(a.length == b.length && a.branch_offset == b.branch_offset);
         CHECK(memcmp(a.bytes, b.bytes, a.length) == 0);
         const uint8_t *branch = a.bytes + a.branch_offset;
         CHECK(branch[3] & 1);
         uint64_t target = 0;
         for (unsigned byte = 0; byte < 6; ++byte)
            target |= (uint64_t)branch[3 + byte] << (8 * byte);
         CHECK((target & ~UINT64_C(1)) == program.code_va);
         struct agx_ioto_record record;
         size_t consumed = 0;
         CHECK(agx_ioto_validate_kind(AGX_CHIP_G17P, AGX_IOTO_KIND_B_EP0,
                                     branch, 10, &record, &consumed));
         CHECK(consumed == 10);
      }
   }

   /* Absolute code may be farther away than a signed 48-bit displacement.
    * Bounds still apply to both the executable extent and output suffix. */
   program.code_va = UINT64_C(0x800000000100);
   CHECK(agx_ioto_constant_program_emit(AGX_CHIP_G17P, &program, 2, &a));
   program.absolute_branch = false;
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G17P, &program, 2, &b));
   program.absolute_branch = true;
   memset(&b, 0xa5, sizeof(b));
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G18G, &program, 2, &b));
   CHECK(all_bytes(&b, sizeof(b), 0xa5));
   program.code_va = (UINT64_C(1) << 48) - 2;
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G17P, &program, 2, &b));
   CHECK(all_bytes(&b, sizeof(b), 0xa5));
   return 0;
}

static int
check_constant_program_refusals(void)
{
   const uint64_t limit = UINT64_C(1) << 48;
   static const uint8_t code[] = {0x0e, 0, 0, 0};
   static const struct {
      uint64_t suffix_va, code_va;
      size_t code_size;
      bool external, prelude;
      const void *code;
   } invalid[] = {
      {0, 0x2000, 4, true, false, NULL},
      {1, 0x2000, 4, true, false, NULL},
      {UINT64_C(1) << 48, 0x2000, 4, true, false, NULL},
      {0x1000, 0, 4, true, false, NULL},
      {0x1000, 0x2001, 4, true, false, NULL},
      {0x1000, UINT64_C(1) << 48, 4, true, false, NULL},
      {0x1000, UINT64_C(0xfffffffffffe), 4, true, false, NULL},
      /* SIZE_MAX - 1 is at least 2^32 - 2, so this extent runs past the
       * 48-bit address space for every size_t width. */
      {0x1000, UINT64_C(0xffff00000004), SIZE_MAX - 1, true, false, NULL},
      {0x1000, 0x2000, 3, true, false, NULL},
      {2, UINT64_C(0x800000000002), 4, true, false, NULL},
      {UINT64_C(0x800000000004), 2, 4, true, false, NULL},
      {UINT64_C(0xfffffffffff8), UINT64_C(0xffffffffff00),
       4, true, false, NULL},
      {UINT64_C(0xfffffffffff6), UINT64_C(0xffffffffff00),
       4, true, true, NULL},
      {0x1000, 0, 4, false, false, NULL},
      {0x1000, 0, 3, false, false, code},
      {UINT64_C(0xfffffffffffe), 0, 4, false, false, code},
      {UINT64_C(0xfffffffffffe), 0, 0, false, false, NULL},
   };
   for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      struct agx_ioto_constant_program program = {
         .code = invalid[i].code,
         .code_size = invalid[i].code_size,
         .code_va = invalid[i].code_va,
         .force_external = invalid[i].external,
         .request_prelude = invalid[i].prelude,
      };
      struct agx_ioto_preshader_output output;
      memset(&output, 0xa5, sizeof(output));
      CHECK(!agx_ioto_constant_program_emit(
         AGX_CHIP_G17P, &program, invalid[i].suffix_va, &output));
      CHECK(all_bytes(&output, sizeof(output), 0xa5));
   }

   struct agx_ioto_constant_program absent = {0};
   struct agx_ioto_preshader_output output;
   memset(&output, 0xa5, sizeof(output));
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G14G, &absent, 2, &output));
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G17P, NULL, 2, &output));
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G17P, &absent, 2, NULL));
   CHECK(!agx_ioto_constant_program_emit(AGX_CHIP_G17P, &absent, limit, &output));
   CHECK(all_bytes(&output, sizeof(output), 0xa5));
   return 0;
}

int
main(void)
{
   if (check_g17p_absolute_constant_program() ||
       check_bound_range() || check_branch_witnesses() ||
       check_constant_program_refusals())
      return 1;

   for (size_t i = 0; i < sizeof(gpu0_targets) / sizeof(gpu0_targets[0]); ++i) {
      if (check_branch(gpu0_targets[i], false) ||
          check_inline(gpu0_targets[i]) ||
          check_external(gpu0_targets[i], false) ||
          check_constant_program(gpu0_targets[i]))
         return 1;
   }

   for (size_t i = 0; i < sizeof(gpu1_targets) / sizeof(gpu1_targets[0]); ++i) {
      if (check_branch(gpu1_targets[i], true) ||
          check_inline(gpu1_targets[i]) ||
          check_external(gpu1_targets[i], true) ||
          check_constant_program(gpu1_targets[i]))
         return 1;
   }

   uint8_t bytes[AGX_IOTO_PRESHADER_BRANCH_SIZE];
   memset(bytes, 0xa5, sizeof(bytes));
   CHECK(!agx_ioto_preshader_pack_branch(AGX_CHIP_G14G, 0, false, bytes));
   CHECK(all_bytes(bytes, sizeof(bytes), 0xa5));
   CHECK(!agx_ioto_preshader_pack_branch(AGX_CHIP_UNKNOWN, 0, false, NULL));

   printf("agx_ioto_preshader_core: %u checks passed\n", checks);
   return 0;
}
