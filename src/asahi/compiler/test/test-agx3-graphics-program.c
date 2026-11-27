/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_graphics_program.h"
#include "agx3_copy_core.h"
#include "agx3_ioto_load_packer.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
#include "util/macros.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,  \
                 #x);                                                          \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static int
test_g17g_programs(void)
{
   /* Native M5 triangle's complete state records, independently captured
    * from the currently compiled shader. Runtime D remains unqualified
    * until these generated programs execute with Mesa-owned resources. */
   static const uint8_t expected_tail[2][28] = {
      {0x07,0x00,0x00,0x0c,0x00,0x00,0x00,0x00,
       0x87,0x04,0x48,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
       0x87,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00},
      {0x17,0x00,0x00,0x2c,0x00,0x00,0x18,0x08,
       0xc7,0x03,0x40,0x0e,0x00,0x00,0x00,0x00,0x00,0x01,
       0x07,0x00,0x00,0x0d,0x00,0x00,0x00,0x00,0x00,0x00},
   };
   for (unsigned fragment = 0; fragment < 2; ++fragment) {
      struct agx3_g17p_graphics_program_desc desc = {
         .program_va = UINT64_C(0x10000100000),
         .shader_va = fragment ? UINT64_C(0x100000003c0)
                              : UINT64_C(0x10000000480),
         .shader_binary_size = fragment ? 60 : 128,
         .nr_gprs = 16,
         .compiler_d_valid = true,
         .compiler_d = fragment ? 0 : 32,
         .fragment = fragment,
         .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
      };
      struct agx3_g17p_graphics_program output;
      size_t empty_bound;
      CHECK(agx3_g17g_graphics_program_bound(AGX_CHIP_G17G, NULL, 0,
                                              &empty_bound));
      CHECK(empty_bound == 90);
      CHECK(agx3_g18g_graphics_program_generate(&desc, &output));
      CHECK(output.length == (fragment ? 90 : 84));
      CHECK(memcmp(output.bytes, expected_tail[fragment], 28) == 0);

      /* Eight RGBA8 attachments make the imageblock sample size 32 bytes.
       * Apple's native eight-target render publishes the same SPECLM as the
       * one-target render with byte 7 alone moved 0x08 -> 0x20, keeping the
       * 32x32 tile at exactly the 32 KiB bound. */
      if (fragment) {
         struct agx3_g17p_graphics_program mrt_output;
         const uint8_t mrt_sizes[] = {16, 32};
         for (unsigned i = 0; i < sizeof(mrt_sizes); ++i) {
            struct agx3_g17p_graphics_program_desc mrt = desc;
            mrt.imageblock.sample_size_B = mrt_sizes[i];
            CHECK(agx3_g18g_graphics_program_generate(&mrt, &mrt_output));
            uint8_t expected_mrt[28];
            memcpy(expected_mrt, expected_tail[1], sizeof(expected_mrt));
            expected_mrt[7] = mrt_sizes[i];
            CHECK(mrt_output.length == output.length);
            CHECK(memcmp(mrt_output.bytes, expected_mrt, 28) == 0);
         }

         /* A colour-free render keeps the eight-byte spelling. */
         struct agx3_g17p_graphics_program_desc bare = desc;
         bare.imageblock.sample_size_B = 0;
         struct agx3_g17p_graphics_program bare_output;
         CHECK(agx3_g18g_graphics_program_generate(&bare, &bare_output));
         CHECK(memcmp(bare_output.bytes, expected_tail[1], 28) == 0);

         /* The admitted single-sample layouts include 24-byte samples. */
         const uint8_t ungrounded[] = {4, 12};
         for (unsigned i = 0; i < sizeof(ungrounded); ++i) {
            struct agx3_g17p_graphics_program_desc bad = desc;
            bad.imageblock.sample_size_B = ungrounded[i];
            CHECK(!agx3_g18g_graphics_program_generate(&bad, &mrt_output));
         }
      }

      const uint32_t original_size = desc.shader_binary_size;
      desc.shader_binary_size = 512;
      CHECK(agx3_g18g_graphics_program_generate(&desc, &output));
      const struct agx3_g17p_graphics_program saturated = output;
      const uint32_t larger_sizes[] = {513, 790, 1024, 65536, UINT32_MAX};
      for (unsigned i = 0; i < sizeof(larger_sizes)/sizeof(larger_sizes[0]); ++i) {
         desc.shader_binary_size = larger_sizes[i];
         CHECK(agx3_g18g_graphics_program_generate(&desc, &output));
         CHECK(output.length == saturated.length &&
               !memcmp(output.bytes, saturated.bytes, output.length));
      }
      desc.shader_binary_size = 0;
      CHECK(!agx3_g18g_graphics_program_generate(&desc, &output));
      desc.shader_binary_size = original_size;

      struct agx3_ioto_uniform_range ranges[] = {
         {.start_halfs = 0, .size_halfs = 4, .address = 0x4000001000},
         {.start_halfs = 4, .size_halfs = 2, .address = 0x4000002000},
      };
      desc.uniform_ranges = ranges;
      desc.uniform_range_count = 2;
      size_t bound;
      CHECK(agx3_g17g_graphics_program_bound(AGX_CHIP_G17G, ranges, 2, &bound));
      CHECK(agx3_g18g_graphics_program_generate(&desc, &output));
      CHECK(output.length > 90 && output.length <= bound);
      struct agx3_g17p_graphics_program baseline = output;
      for (unsigned i = 0; i < 2; ++i) {
         ranges[i].address += 64;
         CHECK(agx3_g18g_graphics_program_generate(&desc, &output));
         CHECK(output.length == baseline.length &&
               memcmp(output.bytes, baseline.bytes, output.length) != 0);
         ranges[i].address -= 64;
      }

      baseline = output;
      desc.scratch_size = 4;
      CHECK(!agx3_g18g_graphics_program_generate(&desc, &output));
      desc.scratch_size = 0;
      desc.compiler_d_valid = false;
      CHECK(!agx3_g18g_graphics_program_generate(&desc, &output));
      desc.compiler_d_valid = true;
      desc.shader_va++;
      CHECK(!agx3_g18g_graphics_program_generate(&desc, &output));
      desc.shader_va--;
      CHECK(!agx3_g17g_graphics_program_generate(AGX_CHIP_G17P, &desc, &output));
      CHECK(memcmp(&output, &baseline, sizeof(output)) == 0);
      if (fragment) {
         static const uint8_t bg_end[] = {7, 0x10, 0, 4, 0, 0, 14, 0, 0, 0};
         static const uint8_t eot_end[] = {14, 0, 0, 0};
         CHECK(agx3_g18g_tile_program_generate(&desc, false, &output));
         CHECK(output.length == baseline.length);
         CHECK(memcmp(output.bytes + output.length - sizeof(bg_end),
                      bg_end, sizeof(bg_end)) == 0);
         CHECK(agx3_g18g_tile_program_generate(&desc, true, &output));
         CHECK(output.length + 6 == baseline.length);
         CHECK(memcmp(output.bytes + output.length - sizeof(eot_end),
                      eot_end, sizeof(eot_end)) == 0);
      } else {
         CHECK(!agx3_g18g_tile_program_generate(&desc, false, &output));
         /* Captured M5 compute profile: no imageblock or PIXREL, and the
          * third loader LDIMM contains4. The native shader preamble is
          * deliberately absent in this Mesa program, which ends directly. */
         desc.uniform_ranges = NULL;
         desc.uniform_range_count = 0;
         desc.shader_va = UINT64_C(0x100000003c0);
         desc.shader_binary_size = 64;
         desc.compiler_d = 0;
         CHECK(agx3_g18g_compute_program_generate(&desc, &output));
         CHECK(output.length == 84);
         CHECK(!memcmp(output.bytes, expected_tail[0], 8));
         CHECK(!memcmp(output.bytes + 8, expected_tail[1] + 8, 20));
         CHECK(output.bytes[28 + 20] == 4);
         static const uint8_t end[] = {14, 0, 0, 0};
         CHECK(!memcmp(output.bytes + 80, end, sizeof(end)));
         desc.fragment = true;
         CHECK(!agx3_g18g_compute_program_generate(&desc, &output));
      }
   }
   return 0;
}

static uint64_t
branch_target(const uint8_t *branch, uint64_t pc)
{
   uint64_t displacement = 0;
   for (unsigned i = 0; i < 6; ++i)
      displacement |= (uint64_t)branch[3 + i] << (i * 8);
   if (displacement & 1)
      return displacement & ~UINT64_C(1);
   return (pc + displacement) & ((UINT64_C(1) << 48) - 1);
}

static int
test_constant_programs(void)
{
   /* Native LDIMM prelude, END, and a trailing NOP. The suffix encoder
    * must trim only the NOP, retaining the program's own termination. */
   static const uint8_t code[] = {0x0c, 0x10, 0x0e, 0, 0, 0, 0x06, 0};
   for (unsigned terminal = 0; terminal < 3; ++terminal) {
      for (unsigned loads = 0; loads < 3; ++loads) {
         const bool rooted = loads == 2;
         struct agx3_ioto_uniform_range range = {
            .start_halfs = 8, .size_halfs = 8,
            .address = rooted ? 24 : 0x12340000,
         };
         struct agx3_g17p_graphics_program_desc desc = {
            .program_va = 0x10000, .shader_va = 0x100000003c0,
            .uniform_ranges = loads ? &range : NULL,
            .uniform_range_count = loads ? 1 : 0,
            .shader_binary_size = 128, .nr_gprs = 16,
            .fragment = terminal != 0, .writes_sample_mask = terminal == 2,
            .colour_classes = terminal ? 2 : 0, .preloaded_root = rooted,
            .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
         };
         struct agx3_g17p_graphics_program absent, inlined;
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &absent));
         const size_t offset = absent.length - 4;
         desc.constant_program = (struct agx_ioto_constant_program){
            .code = code, .code_size = sizeof(code), .code_va = 0x90000,
         };
         size_t bound;
         CHECK(agx3_g17p_graphics_program_desc_bound(
            AGX_CHIP_G17P, &desc, &bound));
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &inlined));
         CHECK(inlined.length == offset + sizeof(code) - 2);
         CHECK(inlined.length <= bound);
         CHECK(memcmp(inlined.bytes, absent.bytes, offset) == 0);
         CHECK(memcmp(inlined.bytes + offset, code, sizeof(code) - 2) == 0);

         for (unsigned mode = 0; mode < 6; ++mode) {
            desc.constant_program.code = mode ? NULL : code;
            desc.constant_program.code_size = mode == 3 ? 258 : sizeof(code);
            desc.constant_program.force_external = mode != 0 && mode != 3;
            desc.constant_program.request_prelude = mode == 2 || mode == 5;
            desc.constant_program.absolute_branch = mode >= 4;
            struct agx3_g17p_graphics_program_template cached;
            CHECK(agx3_g17p_graphics_program_template_generate(
               AGX_CHIP_G17P, &desc, &cached));
            CHECK(cached.constant_suffix_offset == offset);

            struct agx3_g17p_graphics_program_desc moved = desc;
            struct agx3_ioto_uniform_range moved_range = range;
            if (loads && !rooted)
               moved_range.address = 0x56780000;
            moved.uniform_ranges = loads ? &moved_range : NULL;
            moved.program_va = 0x40000;
            moved.constant_program.code_va = 0x8000;
            struct agx3_g17p_graphics_program fresh, relocated = {0};
            CHECK(agx3_g17p_graphics_program_generate(
               AGX_CHIP_G17P, &moved, &fresh));
            CHECK(agx3_g17p_graphics_program_template_relocate(
               AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
               sizeof(relocated.bytes), &relocated.length));
            CHECK(fresh.length == relocated.length);
            CHECK(memcmp(fresh.bytes, relocated.bytes, fresh.length) == 0);

            if (mode) {
               const size_t branch = offset +
                  (desc.constant_program.request_prelude ? 2 : 0);
               CHECK(branch_target(cached.program.bytes + branch,
                                   desc.program_va + branch) ==
                     desc.constant_program.code_va);
               CHECK(branch_target(relocated.bytes + branch,
                                   moved.program_va + branch) ==
                     moved.constant_program.code_va);
               if (mode >= 4) {
                  CHECK(cached.program.bytes[branch + 3] & 1);
                  struct agx3_g17p_graphics_program_desc address_only = desc;
                  address_only.program_va = moved.program_va;
                  struct agx3_g17p_graphics_program copied;
                  CHECK(agx3_g17p_graphics_program_generate(
                     AGX_CHIP_G17P, &address_only, &copied));
                  CHECK(copied.length == cached.program.length);
                  CHECK(memcmp(copied.bytes, cached.program.bytes,
                               copied.length) == 0);
               }
            }

            /* A cache hit cannot silently reuse different inline code, a
             * corrupt branch, or a malformed program extent/address. */
            struct agx3_g17p_graphics_program before = relocated;
            cached.program.bytes[offset] ^= 1;
            CHECK(!agx3_g17p_graphics_program_template_relocate(
               AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
               sizeof(relocated.bytes), &relocated.length));
            CHECK(memcmp(&before, &relocated, sizeof(before)) == 0);
            cached.program.bytes[offset] ^= 1;
            if (!mode) {
               uint8_t changed[sizeof(code)];
               memcpy(changed, code, sizeof(code));
               changed[1] ^= 1;
               moved.constant_program.code = changed;
               CHECK(!agx3_g17p_graphics_program_template_relocate(
                  AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
                  sizeof(relocated.bytes), &relocated.length));
               moved.constant_program.code = code;
            } else {
               moved.constant_program.code_va = (UINT64_C(1) << 48) - 2;
               CHECK(!agx3_g17p_graphics_program_generate(
                  AGX_CHIP_G17P, &moved, &relocated));
            }
            moved.constant_program.code_size |= 1;
            CHECK(!agx3_g17p_graphics_program_desc_bound(
               AGX_CHIP_G17P, &moved, &bound));
            CHECK(!agx3_g17p_graphics_program_generate(
               AGX_CHIP_G17P, &moved, &relocated));
            CHECK(memcmp(&before, &relocated, sizeof(before)) == 0);
         }

         if (rooted) {
            /* The persistent suffix is a second allocation. Resolve the
             * constant branch from its own PC, never the launcher PC. */
            desc.program_va = 0xa0000;
            desc.constant_program.force_external = true;
            desc.constant_program.request_prelude = true;
            uint8_t external[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
            size_t written;
            CHECK(agx3_g17p_graphics_external_setup_suffix_generate(
               AGX_CHIP_G17P, &desc, external, sizeof(external), &written));
            CHECK(written >= 12);
            CHECK(branch_target(external + written - 10,
                                desc.program_va + written - 10) ==
                  desc.constant_program.code_va);
         }
      }
   }
   return 0;
}

static int
test_cprog_input_composition(void)
{
   static const uint8_t end[] = {0x0e, 0, 0, 0};
   for (unsigned mode = 0; mode < 6; ++mode) {
      const bool rooted = mode % 2;
      uint64_t halves[8] = {0x32};
      struct agx3_cprog_inputs inputs, equal_inputs, changed_inputs;
      CHECK(agx3_cprog_inputs_init(halves, &inputs));
      equal_inputs = inputs;
      halves[0] = 0x62;
      CHECK(agx3_cprog_inputs_init(halves, &changed_inputs));
      struct agx3_ioto_uniform_range range = {0, 8, rooted ? 24 : 0x12340000};
      struct agx3_g17p_graphics_program_desc desc = {
         .program_va = 0x10000, .shader_va = 0x100000003c0,
         .uniform_ranges = &range, .uniform_range_count = 1,
         .shader_binary_size = 128, .nr_gprs = 16,
         .fragment = mode / 2 != 0, .writes_sample_mask = mode / 2 == 2,
         .colour_classes = mode / 2 ? 2 : 0, .preloaded_root = rooted,
         .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
         .constant_program = {.code = end, .code_size = sizeof(end),
                              .code_va = 0x90000},
      };
      struct agx3_g17p_graphics_program baseline, generated, fresh, relocated;
      CHECK(agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &desc, &baseline));
      desc.cprog_inputs = &inputs;
      size_t bound;
      CHECK(agx3_g17p_graphics_program_desc_bound(AGX_CHIP_G17P, &desc, &bound));
      CHECK(agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &desc, &generated));
      CHECK(generated.length == baseline.length + 30);
      CHECK(generated.length <= bound);
      if (mode == 0) {
         struct agx3_cprog_inputs full;
         uint64_t all[8];
         for (unsigned w = 0; w < 8; ++w)
            all[w] = UINT64_MAX;
         CHECK(agx3_cprog_inputs_init(all, &full));
         struct agx3_ioto_uniform_range full_range = {0, 512, 0x12340000};
         struct agx3_g17p_graphics_program_desc dense = desc;
         dense.uniform_ranges = &full_range;
         dense.cprog_inputs = &full;
         size_t dense_bound;
         CHECK(agx3_g17p_graphics_program_desc_bound(
            AGX_CHIP_G17P, &dense, &dense_bound));
         CHECK(agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &dense, &fresh));
         CHECK(fresh.length <= dense_bound);
         CHECK(dense_bound <= AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
      }
      struct agx3_g17p_graphics_program_template cached;
      CHECK(agx3_g17p_graphics_program_template_generate(AGX_CHIP_G17P, &desc, &cached));
      struct agx3_g17p_graphics_program_desc moved = desc;
      moved.program_va = 0x60000;
      moved.cprog_inputs = &equal_inputs;
      CHECK(agx3_g17p_graphics_program_template_relocate(
         AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
         sizeof(relocated.bytes), &relocated.length));
      CHECK(agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &moved, &fresh));
      CHECK(relocated.length == fresh.length);
      CHECK(!memcmp(relocated.bytes, fresh.bytes, fresh.length));
      moved.cprog_inputs = &changed_inputs;
      struct agx3_g17p_graphics_program before = relocated;
      CHECK(!agx3_g17p_graphics_program_template_relocate(
         AGX_CHIP_G17P, &desc, &moved, &cached, relocated.bytes,
         sizeof(relocated.bytes), &relocated.length));
      CHECK(!memcmp(&before, &relocated, sizeof(before)));
      moved.uniform_range_count = 0;
      CHECK(!agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &moved, &relocated));
      CHECK(!memcmp(&before, &relocated, sizeof(before)));
      if (rooted) {
         uint8_t external[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
         size_t with_inputs, without_inputs;
         CHECK(agx3_g17p_graphics_external_setup_suffix_generate(
            AGX_CHIP_G17P, &desc, external, sizeof(external), &with_inputs));
         desc.cprog_inputs = NULL;
         CHECK(agx3_g17p_graphics_external_setup_suffix_generate(
            AGX_CHIP_G17P, &desc, external, sizeof(external), &without_inputs));
         CHECK(with_inputs == without_inputs + 30);
      }
   }
   return 0;
}

static uint8_t *
eot_offset(struct agx3_g17p_rt0_eot_program_desc *desc, unsigned rt)
{
   uint8_t *const offsets[AGX3_G17P_MRT_MAX_TARGETS] = {
      &desc->imageblock_offset_B,  &desc->imageblock_offset1_B,
      &desc->imageblock_offset2_B, &desc->imageblock_offset3_B,
      &desc->imageblock_offset4_B, &desc->imageblock_offset5_B,
      &desc->imageblock_offset6_B, &desc->imageblock_offset7_B,
   };
   return offsets[rt];
}

static int
test_preamble_programs(void)
{
   /* Compiler-produced G17P uniform-pointer preamble, including its END. */
   static const uint8_t code[] = {
      0x67, 0x00, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x19, 0x00, 0x00, 0x40, 0x02, 0x00, 0x0b, 0x01,
      0x07, 0x01, 0x00, 0x20, 0x00, 0x80, 0x00, 0x00,
      0x4b, 0x00, 0x01, 0x04, 0x5b, 0x02, 0x01, 0x04,
      0x0e, 0x00, 0x00, 0x00,
   };
   static const uint8_t branch[] = {
      0x0f, 0x00, 0x54, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x00,
   };
   uint8_t binary[128 + sizeof(code)];
   memset(binary, 0xa5, sizeof(binary));
   memcpy(binary + 128, code, sizeof(code));
   const struct agx3_ioto_uniform_range range = {
      .start_halfs = 0, .size_halfs = 8, .address = UINT64_C(0x10000024000),
   };
   for (unsigned stage = 0; stage < 3; ++stage) {
      for (unsigned resources = 0; resources < 2; ++resources) {
         struct agx3_g17p_graphics_program_desc desc = {
            .program_va = UINT64_C(0x10000100000),
            .shader_va = UINT64_C(0x100002003c0),
            .uniform_ranges = resources ? &range : NULL,
            .uniform_range_count = resources,
            .shader_binary_size = 128,
            .nr_gprs = 16,
            .fragment = stage != 0,
            .writes_sample_mask = stage == 2,
            .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
         };
         struct agx3_g17p_graphics_program baseline, program;
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &baseline));
         struct agx_ioto_preshader_desc preamble = {0};
         CHECK(agx_ioto_preshader_bind_range(
            binary, sizeof(binary), UINT64_C(0x123456789a3c), sizeof(binary),
            128, sizeof(code), &preamble));
         desc.preshader = &preamble;
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &program));
         const size_t prefix = baseline.length - AGX_IOTO_PRESHADER_ABSENT_SIZE;
         CHECK(program.length == prefix + sizeof(code));
         CHECK(memcmp(program.bytes, baseline.bytes, prefix) == 0);
         CHECK(memcmp(program.bytes + prefix, code, sizeof(code)) == 0);
         size_t old_bound, bound;
         CHECK(agx3_g17p_graphics_program_bound(
            AGX_CHIP_G17P, desc.uniform_ranges, resources, &old_bound));
         CHECK(agx3_g17p_graphics_program_bound_with_preshader(
            AGX_CHIP_G17P, desc.uniform_ranges, resources, &preamble, &bound));
         CHECK(bound == old_bound - AGX_IOTO_PRESHADER_ABSENT_SIZE + sizeof(code));
         CHECK(program.length <= bound);

         preamble.force_external = true;
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &program));
         CHECK(program.length == prefix + sizeof(branch));
         CHECK(memcmp(program.bytes, baseline.bytes, prefix) == 0);
         CHECK(memcmp(program.bytes + prefix, branch, sizeof(branch)) == 0);

         /* The automatic external selection retains the same branch. */
         preamble.force_external = false;
         preamble.code = NULL;
         preamble.code_size = AGX_IOTO_PRESHADER_INLINE_LIMIT + 2;
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &program));
         CHECK(program.length == prefix + sizeof(branch));
         CHECK(memcmp(program.bytes + prefix, branch, sizeof(branch)) == 0);

         const struct agx3_g17p_graphics_program before = program;
         preamble.external_va |= 1;
         CHECK(!agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &desc, &program));
         CHECK(memcmp(&before, &program, sizeof(program)) == 0);
         bound = 123;
         CHECK(!agx3_g17p_graphics_program_bound_with_preshader(
            AGX_CHIP_G17P, desc.uniform_ranges, resources, &preamble, &bound));
         CHECK(bound == 123);
      }
   }
   return 0;
}

int
main(void)
{
   CHECK(test_g17g_programs() == 0);
   /* Glamor's instanced rectangle needs six independent VS ranges: VBO
    * bases, clamps, draw parameters, UBO pointer, provoking vertex and clip
    * coefficient. The old five-range limit silently left its pipeline zero. */
   struct agx3_ioto_uniform_range rectangle_ranges[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x4000001078},
      {.start_halfs = 8, .size_halfs = 4, .address = 0x4000001178},
      {.start_halfs = 12, .size_halfs = 4, .address = 0x400000000c},
      {.start_halfs = 16, .size_halfs = 4, .address = 0x4000000068},
      {.start_halfs = 20, .size_halfs = 2, .address = 0x4000001290},
      {.start_halfs = 22, .size_halfs = 2, .address = 0x40000012a0},
   };
   const size_t rectangle_count =
      sizeof(rectangle_ranges) / sizeof(rectangle_ranges[0]);
   size_t rectangle_bound = 0;
   CHECK(agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, rectangle_ranges, rectangle_count, &rectangle_bound));
   struct agx3_g17p_graphics_program_desc rectangle_desc = {
      .program_va = UINT64_C(0x10000100000),
      .shader_va = UINT64_C(0x100002003c0),
      .uniform_ranges = rectangle_ranges,
      .uniform_range_count = rectangle_count,
      .shader_binary_size = 282,
      .nr_gprs = 29,
      .compiler_d_valid = true,
      .compiler_d = 20,
   };
   struct agx3_g17p_graphics_program rectangle;
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &rectangle_desc, &rectangle));
   CHECK(rectangle.length > 0 && rectangle.length <= rectangle_bound);
   const struct agx3_g17p_graphics_program rectangle_baseline = rectangle;
   for (unsigned i = 0; i < rectangle_count; ++i) {
      rectangle_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &rectangle_desc, &rectangle));
      CHECK(rectangle.length == rectangle_baseline.length);
      CHECK(memcmp(rectangle.bytes, rectangle_baseline.bytes,
                   rectangle.length) != 0);
      rectangle_ranges[i].address -= 0x40;
   }

   /* More ranges must not bypass the existing program byte budget. */
   struct agx3_ioto_uniform_range oversized_ranges[128];
   for (unsigned i = 0; i < 128; ++i) {
      oversized_ranges[i] = (struct agx3_ioto_uniform_range){
         .start_halfs = 2 * i, .size_halfs = 2,
         .address = UINT64_C(0x4000010000) + 4 * i,
      };
   }
   const size_t previous_bound = rectangle_bound;
   CHECK(!agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, oversized_ranges, 128, &rectangle_bound));
   CHECK(rectangle_bound == previous_bound);
   rectangle_desc.uniform_ranges = oversized_ranges;
   rectangle_desc.uniform_range_count = 128;
   const struct agx3_g17p_graphics_program previous_program = rectangle;
   CHECK(!agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &rectangle_desc, &rectangle));
   CHECK(memcmp(&rectangle, &previous_program, sizeof(rectangle)) == 0);

   CHECK(test_constant_programs() == 0);
   CHECK(test_cprog_input_composition() == 0);
   /* The inline compact program begins with exactly two root-address
    * immediates. The immutable shader-owned setup template follows them. */
   const uint64_t root_va = UINT64_C(0x2ffff6c540);
   struct agx3_g17p_graphics_root_prefix prefix;
   CHECK(agx3_g17p_graphics_root_prefix_generate(
      AGX_CHIP_G17P, root_va, &prefix));
   uint8_t expected[AGX3_G17P_GRAPHICS_ROOT_PREFIX_MAX_SIZE] = {0};
   uint8_t low_length = 0, high_length = 0;
   CHECK(agx_ioto_load_pack_ldimm(
      AGX_CHIP_G17P, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF,
      (uint32_t)root_va, expected, &low_length));
   CHECK(agx_ioto_load_pack_ldimm(
      AGX_CHIP_G17P, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF + 2,
      (uint32_t)(root_va >> 32), expected + low_length,
      &high_length));
   CHECK(prefix.length == low_length + high_length);
   CHECK(prefix.length <= AGX3_G17P_GRAPHICS_ROOT_PREFIX_MAX_SIZE);
   CHECK(memcmp(prefix.bytes, expected, prefix.length) == 0);
   CHECK(!agx3_g17p_graphics_root_prefix_generate(
      AGX_CHIP_G17S, root_va, &prefix));
   CHECK(!agx3_g17p_graphics_root_prefix_generate(
      AGX_CHIP_G17P, 0, &prefix));

   /* A persistent root range is a true execution-time two-level load.  Its
    * source VA lives in ordinary root memory, while the immutable setup only
    * contains the root-field offset, destination and width.  Exercise more
    * than four dwords so the fixed payload window must be retired and reused. */
   uint8_t indirect_body[512];
   size_t indirect_size = 0;
   CHECK(agx3_ioto_load_indirect_uniform_range(
      AGX_CHIP_G17P, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF, 24, 40, 18,
      indirect_body, sizeof(indirect_body), &indirect_size));
   CHECK(indirect_size > 0 && indirect_size < sizeof(indirect_body));
   const size_t valid_indirect_size = indirect_size;
   uint8_t uniform_root_body[512];
   size_t uniform_root_size = 0;
   CHECK(agx3_ioto_load_indirect_uniform_range_from_uniform_root(
      AGX_CHIP_G17P, AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF, 24,
      40, 18, uniform_root_body, sizeof(uniform_root_body),
      &uniform_root_size));
   CHECK(uniform_root_size == valid_indirect_size);
   CHECK(memcmp(uniform_root_body, indirect_body, valid_indirect_size) != 0);
   CHECK(!agx3_ioto_load_indirect_uniform_range(
      AGX_CHIP_G17S, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF, 24, 40, 18,
      indirect_body, sizeof(indirect_body), &indirect_size));
   CHECK(indirect_size == valid_indirect_size);
   CHECK(!agx3_ioto_load_indirect_uniform_range(
      AGX_CHIP_G17P, AGX3_G17P_GRAPHICS_ROOT_BASE_HALF, 24, 40, 17,
      indirect_body, sizeof(indirect_body), &indirect_size));
   CHECK(indirect_size == valid_indirect_size);

   struct agx3_ioto_uniform_range rooted_ranges[] = {
      {.start_halfs = 8, .size_halfs = 4, .address = 24},
      {.start_halfs = 20, .size_halfs = 18, .address = 32},
   };
   const struct agx3_g17p_graphics_program_desc rooted_desc = {
      .program_va = UINT64_C(0x10000000000),
      .shader_va = UINT64_C(0x100000003c0),
      .uniform_ranges = rooted_ranges,
      .uniform_range_count = ARRAY_SIZE(rooted_ranges),
      .shader_binary_size = 128,
      .nr_gprs = 16,
      .preloaded_root = true,
   };
   struct agx3_g17p_graphics_program_template rooted_template;
   CHECK(agx3_g17p_graphics_program_template_generate(
      AGX_CHIP_G17P, &rooted_desc, &rooted_template));
   CHECK(rooted_template.program.length >
         AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE);
   CHECK(rooted_template.relocation_count == 0);

   /* The production split keeps the root address in a bounded launcher and
    * the two-level load schedule in an immutable external suffix. A different
    * draw root changes only the launcher; the suffix remains byte-identical. */
   uint8_t external_suffix[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   size_t external_suffix_size = 0;
   CHECK(agx3_g17p_graphics_external_setup_suffix_generate(
      AGX_CHIP_G17P, &rooted_desc, external_suffix,
      sizeof(external_suffix), &external_suffix_size));
   CHECK(external_suffix_size > 4 &&
         external_suffix_size < rooted_template.program.length);
   CHECK(external_suffix[external_suffix_size - 4] == 0x0e);
   CHECK(external_suffix[external_suffix_size - 3] == 0x00);
   CHECK(external_suffix[external_suffix_size - 2] == 0x00);
   CHECK(external_suffix[external_suffix_size - 1] == 0x00);

   uint8_t launcher_a[AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE];
   uint8_t launcher_b[AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE];
   size_t launcher_a_size = 0, launcher_b_size = 0;
   const uint64_t setup_va = UINT64_C(0x10000008000);
   CHECK(agx3_g17p_graphics_external_launcher_generate(
      AGX_CHIP_G17P, &rooted_desc, rooted_desc.program_va, root_va, setup_va,
      launcher_a, sizeof(launcher_a), &launcher_a_size));
   CHECK(launcher_a_size > prefix.length &&
         launcher_a_size <= sizeof(launcher_a));
   uint8_t carrier[2][AGX3_COPY_MAX_LENGTH];
   uint8_t carrier_length[2];
   CHECK(agx3_copy_encode_reg_to_uniform(
      AGX_ISA_ENCODING_AGX3B,
      AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF,
      AGX3_G17P_GRAPHICS_ROOT_BASE_HALF, false, 0, carrier[0],
      &carrier_length[0]));
   CHECK(agx3_copy_encode_reg_to_uniform(
      AGX_ISA_ENCODING_AGX3B,
      AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF + 2,
      AGX3_G17P_GRAPHICS_ROOT_BASE_HALF + 2, false, 0, carrier[1],
      &carrier_length[1]));
   CHECK(memcmp(launcher_a + prefix.length, carrier[0],
                carrier_length[0]) == 0);
   CHECK(memcmp(launcher_a + prefix.length + carrier_length[0], carrier[1],
                carrier_length[1]) == 0);
   CHECK(launcher_a_size ==
         prefix.length + carrier_length[0] + carrier_length[1] +
            AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
            AGX_IOTO_G17P_DEFAULT_PROFILE_PREFIX_SIZE +
            AGX_IOTO_PRESHADER_PRELUDE_SIZE +
            AGX_IOTO_PRESHADER_BRANCH_SIZE);
   uint8_t expected_branch[AGX_IOTO_PRESHADER_BRANCH_SIZE];
   const uint64_t branch_pc =
      rooted_desc.program_va + launcher_a_size - sizeof(expected_branch);
   const uint64_t branch_displacement =
      (setup_va - branch_pc) & ((UINT64_C(1) << 48) - 1);
   CHECK(agx_ioto_preshader_pack_branch(
      AGX_CHIP_G17P, branch_displacement, false, expected_branch));
   CHECK(memcmp(launcher_a + launcher_a_size - sizeof(expected_branch),
                expected_branch, sizeof(expected_branch)) == 0);
   CHECK(agx3_g17p_graphics_external_launcher_generate(
      AGX_CHIP_G17P, &rooted_desc, rooted_desc.program_va,
      root_va + 0x1000, setup_va, launcher_b, sizeof(launcher_b),
      &launcher_b_size));
    CHECK(launcher_a_size == launcher_b_size);
    CHECK(memcmp(launcher_a, launcher_b, launcher_a_size) != 0);

    /* The linked shader owns one fully composed launcher template. Recording a
     * draw only relocates its root prefix and terminal PC-relative branch; the
     * result must remain byte-identical to the authoritative full composer. */
    struct agx3_g17p_graphics_external_launcher_template launcher_template;
    CHECK(agx3_g17p_graphics_external_launcher_template_generate(
       AGX_CHIP_G17P, &rooted_desc, setup_va, &launcher_template));
    uint8_t relocated_a[AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE];
    uint8_t relocated_b[AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE];
    size_t relocated_a_size = 0, relocated_b_size = 0;
    CHECK(agx3_g17p_graphics_external_launcher_template_relocate(
       AGX_CHIP_G17P, &launcher_template, rooted_desc.program_va, root_va,
       setup_va, relocated_a, sizeof(relocated_a), &relocated_a_size));
    CHECK(agx3_g17p_graphics_external_launcher_template_relocate(
       AGX_CHIP_G17P, &launcher_template, rooted_desc.program_va,
       root_va + 0x1000, setup_va, relocated_b, sizeof(relocated_b),
       &relocated_b_size));
    CHECK(relocated_a_size == launcher_a_size);
    CHECK(relocated_b_size == launcher_b_size);
    CHECK(memcmp(relocated_a, launcher_a, launcher_a_size) == 0);
    CHECK(memcmp(relocated_b, launcher_b, launcher_b_size) == 0);

    /* A root outside the template's compact LDIMM length class must refuse
     * transactionally so HK can retain the qualified inline fallback. */
    memset(relocated_a, 0xa5, sizeof(relocated_a));
    const size_t refused_written = SIZE_MAX;
    relocated_a_size = refused_written;
    CHECK(!agx3_g17p_graphics_external_launcher_template_relocate(
       AGX_CHIP_G17P, &launcher_template, rooted_desc.program_va,
       UINT64_C(0x18000000080), setup_va, relocated_a, sizeof(relocated_a),
       &relocated_a_size));
    CHECK(relocated_a_size == refused_written);
    for (unsigned i = 0; i < sizeof(relocated_a); ++i)
       CHECK(relocated_a[i] == 0xa5);

    CHECK(!agx3_g17p_graphics_external_launcher_generate(
       AGX_CHIP_G17P, &rooted_desc, rooted_desc.program_va, root_va, 0,
       launcher_a, sizeof(launcher_a), &launcher_a_size));
   CHECK(!agx3_g17p_graphics_external_setup_suffix_generate(
      AGX_CHIP_G17S, &rooted_desc, external_suffix,
      sizeof(external_suffix), &external_suffix_size));

   /* The carrier is outside ordinary linked-shader uniform ownership.  A
    * shader which genuinely consumes the final four halves keeps the proven
    * inline path; external generation must fail without changing that path. */
   struct agx3_ioto_uniform_range carrier_overlap = {
      .start_halfs = AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF,
      .size_halfs = 4,
      .address = 24,
   };
   struct agx3_g17p_graphics_program_desc overlap_desc = rooted_desc;
   overlap_desc.uniform_ranges = &carrier_overlap;
   overlap_desc.uniform_range_count = 1;
   CHECK(!agx3_g17p_graphics_external_setup_suffix_generate(
      AGX_CHIP_G17P, &overlap_desc, external_suffix,
      sizeof(external_suffix), &external_suffix_size));
   CHECK(!agx3_g17p_graphics_external_launcher_generate(
      AGX_CHIP_G17P, &overlap_desc, overlap_desc.program_va, root_va,
      setup_va, launcher_a, sizeof(launcher_a), &launcher_a_size));
   CHECK(agx3_g17p_graphics_program_template_generate(
      AGX_CHIP_G17P, &overlap_desc, &rooted_template));

   CHECK(test_preamble_programs() == 0);
   uint16_t blend_base = UINT16_C(0xa5a5);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 0,
                                                 &blend_base));
   CHECK(blend_base == 8);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 16,
                                                 &blend_base));
   CHECK(blend_base == 16);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 17,
                                                 &blend_base));
   CHECK(blend_base == 18);
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 504,
                                                 &blend_base));
   CHECK(blend_base == 504);
   const uint16_t valid_blend_base = blend_base;
   CHECK(!agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 505,
                                                  &blend_base));
   CHECK(blend_base == valid_blend_base);
   /* The M5 parts (G17G, G17S, G17C, G17D) take the same placement; a pre-AGX3
    * chip is refused and leaves the output alone. */
   for (enum agx_chip m5 = AGX_CHIP_G17G; m5 <= AGX_CHIP_G17D;
        m5 = (enum agx_chip)(m5 + 1)) {
      uint16_t m5_base = UINT16_C(0xa5a5);
      CHECK(agx3_g17p_fragment_blend_constant_base(m5, 16, &m5_base));
      CHECK(m5_base == 16);
   }
   CHECK(!agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G13G, 16,
                                                  &blend_base));
   CHECK(blend_base == valid_blend_base);
   CHECK(!agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 16, NULL));

   /* Exact compositor uniform geometry: the texture/sampler pointer pair,
    * root pointer, and relocated four-float blend constant each occupy a
    * disjoint native range.  Every source must independently change the
    * generated CProg bytes. */
   CHECK(agx3_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, 16,
                                                 &blend_base));
   struct agx3_ioto_uniform_range compositor_ranges[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 12, .size_halfs = 4, .address = 0x2000},
      {.start_halfs = blend_base, .size_halfs = 8, .address = 0x3000},
   };
   const size_t compositor_range_count =
      sizeof(compositor_ranges) / sizeof(compositor_ranges[0]);
   struct agx3_g17p_graphics_program_desc compositor_desc = {
      .program_va = UINT64_C(0x10000100000),
      .shader_va = UINT64_C(0x100002003c0),
      .uniform_ranges = compositor_ranges,
      .uniform_range_count = compositor_range_count,
      .shader_binary_size = 128,
      .nr_gprs = 16,
      .fragment = true,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct agx3_g17p_graphics_program compositor_program;
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &compositor_desc, &compositor_program));
   CHECK(compositor_program.length >
         AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE);
   uint8_t direct_program[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   memset(direct_program, 0xa5, sizeof(direct_program));
   size_t direct_written = SIZE_MAX;
   CHECK(agx3_g17p_graphics_program_write_unpublished(
      AGX_CHIP_G17P, &compositor_desc, direct_program,
      sizeof(direct_program), &direct_written));
   CHECK(direct_written == compositor_program.length);
   CHECK(memcmp(direct_program, compositor_program.bytes,
                direct_written) == 0);
   CHECK(direct_program[direct_written] == 0xa5);

   const size_t successful_written = direct_written;
   CHECK(!agx3_g17p_graphics_program_write_unpublished(
      AGX_CHIP_G17P, &compositor_desc, direct_program,
      compositor_program.length - 1, &direct_written));
   CHECK(direct_written == successful_written);
   CHECK(!agx3_g17p_graphics_program_write_unpublished(
      AGX_CHIP_G17P, &compositor_desc, NULL, sizeof(direct_program),
      &direct_written));
   CHECK(direct_written == successful_written);
   const struct agx3_g17p_graphics_program compositor_baseline =
      compositor_program;
   struct agx3_g17p_graphics_program_desc mixed_desc = compositor_desc;
   mixed_desc.imageblock.sample_size_B = 16;
   struct agx3_g17p_graphics_program mixed_program;
   CHECK(agx3_g17p_graphics_program_generate(AGX_CHIP_G17P, &mixed_desc,
                                              &mixed_program));
   CHECK(mixed_program.length == compositor_baseline.length);
   unsigned imageblock_differences = 0;
   for (unsigned i = 0; i < mixed_program.length; ++i) {
      if (mixed_program.bytes[i] != compositor_baseline.bytes[i]) {
         CHECK(compositor_baseline.bytes[i] == 0x43);
         CHECK(mixed_program.bytes[i] == 0x45);
         imageblock_differences++;
      }
   }
   CHECK(imageblock_differences == 1);
   for (unsigned i = 0; i < compositor_range_count; ++i) {
      compositor_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &compositor_desc, &compositor_program));
      CHECK(compositor_program.length == compositor_baseline.length);
      CHECK(memcmp(compositor_program.bytes, compositor_baseline.bytes,
                   compositor_program.length) != 0);
      compositor_ranges[i].address -= 0x40;
   }

   /* Firefox/WebGL through Zink reached this exact six-source vertex shape:
    * one vertex buffer's base and clamp after the indexed texture prefix,
    * baseVertex/firstInstance, bindless texture/sampler heap pointers, a root
    * plus three descriptor-set pointers, and one promoted float constant at
    * half72.  Range count is only composer bookkeeping; each range becomes
    * ordinary LDDEV/COPY records, so it must not refuse the sixth source. */
   struct agx3_ioto_uniform_range webgl_vertex_ranges[] = {
      {.start_halfs = 8, .size_halfs = 4, .address = 0x1000},
      {.start_halfs = 12, .size_halfs = 2, .address = 0x2000},
      {.start_halfs = 14, .size_halfs = 4, .address = 0x3000},
      {.start_halfs = 0, .size_halfs = 8, .address = 0x4000},
      {.start_halfs = 28, .size_halfs = 16, .address = 0x5000},
      {.start_halfs = 72, .size_halfs = 2, .address = 0x6000},
   };
   const size_t webgl_vertex_range_count =
      sizeof(webgl_vertex_ranges) / sizeof(webgl_vertex_ranges[0]);
   struct agx3_g17p_graphics_program_desc webgl_vertex_desc = {
      .program_va = UINT64_C(0x10000300000),
      .shader_va = UINT64_C(0x100004003c0),
      .uniform_ranges = webgl_vertex_ranges,
      .uniform_range_count = webgl_vertex_range_count,
      .shader_binary_size = 192,
      .nr_gprs = 16,
   };
   struct agx3_g17p_graphics_program webgl_vertex_program;
   size_t webgl_vertex_bound = 0;
   CHECK(agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, webgl_vertex_ranges, webgl_vertex_range_count,
      &webgl_vertex_bound));
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &webgl_vertex_desc, &webgl_vertex_program));
   CHECK(webgl_vertex_program.length <= webgl_vertex_bound);
   const struct agx3_g17p_graphics_program webgl_vertex_baseline =
      webgl_vertex_program;
   /* Vertex state uses a fixed SPECLM tail. Framebuffer imageblock geometry
    * is a fragment-only dependency, so every such variation must remain
    * byte-identical and may share one exact command-buffer program. */
   for (unsigned samples = 1; samples <= 8; samples *= 2) {
      for (unsigned sample_size_B = 4; sample_size_B <= 16;
           sample_size_B *= 2) {
         webgl_vertex_desc.imageblock.nr_samples = samples;
         webgl_vertex_desc.imageblock.sample_size_B = sample_size_B;
         webgl_vertex_desc.imageblock.tile_size_px =
            sample_size_B * samples <= 8 ? 32 : 16;
         CHECK(agx3_g17p_graphics_program_generate(
            AGX_CHIP_G17P, &webgl_vertex_desc, &webgl_vertex_program));
         CHECK(webgl_vertex_program.length == webgl_vertex_baseline.length);
         CHECK(memcmp(webgl_vertex_program.bytes,
                      webgl_vertex_baseline.bytes,
                      webgl_vertex_program.length) == 0);
      }
   }
   for (unsigned i = 0; i < webgl_vertex_range_count; ++i) {
      webgl_vertex_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &webgl_vertex_desc, &webgl_vertex_program));
      CHECK(webgl_vertex_program.length == webgl_vertex_baseline.length);
      CHECK(memcmp(webgl_vertex_program.bytes, webgl_vertex_baseline.bytes,
                   webgl_vertex_program.length) != 0);
      webgl_vertex_ranges[i].address -= 0x40;
   }

   /* AGX3 IOTO programs are command-buffer-local, but their loader schedule
    * is not: independently allocated ranges may move while their program
    * shape remains identical.  Exercise 10,000 such address sets and require
    * relocation to reproduce the full generator byte for byte. */
   struct agx3_ioto_uniform_range template_ranges[] = {
      {.start_halfs = 8, .size_halfs = 4,
       .address = UINT64_C(0x4000012340)},
      {.start_halfs = 12, .size_halfs = 2,
       .address = UINT64_C(0x4000212340)},
      {.start_halfs = 14, .size_halfs = 4,
       .address = UINT64_C(0x4000412340)},
      {.start_halfs = 0, .size_halfs = 8,
       .address = UINT64_C(0x4000612340)},
      {.start_halfs = 28, .size_halfs = 16,
       .address = UINT64_C(0x4000812340)},
      {.start_halfs = 72, .size_halfs = 2,
       .address = UINT64_C(0x4000a12340)},
   };
   struct agx3_g17p_graphics_program_desc template_desc = webgl_vertex_desc;
   template_desc.uniform_ranges = template_ranges;
   struct agx3_g17p_graphics_program_template program_template;
   CHECK(agx3_g17p_graphics_program_template_generate(
      AGX_CHIP_G17P, &template_desc, &program_template));
   CHECK(program_template.relocation_count ==
         sizeof(template_ranges) / sizeof(template_ranges[0]));

   uint32_t random = UINT32_C(0x81418a2d);
   for (unsigned iteration = 0; iteration < 10000; ++iteration) {
      struct agx3_ioto_uniform_range moved[ARRAY_SIZE(template_ranges)];
      memcpy(moved, template_ranges, sizeof(moved));
      for (unsigned i = 0; i < ARRAY_SIZE(moved); ++i) {
         random = random * UINT32_C(1664525) + UINT32_C(1013904223);
         moved[i].address = UINT64_C(0x5000012340) +
                            (uint64_t)i * UINT64_C(0x20000000) +
                            (uint64_t)(random & 0xff) * UINT64_C(0x20000);
      }

      struct agx3_g17p_graphics_program_desc moved_desc = template_desc;
      moved_desc.uniform_ranges = moved;
      struct agx3_g17p_graphics_program reference;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &moved_desc, &reference));

      uint8_t relocated[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      uint16_t relocated_size = 0;
      CHECK(agx3_g17p_graphics_program_template_relocate(
         AGX_CHIP_G17P, &template_desc, &moved_desc, &program_template,
         relocated, sizeof(relocated), &relocated_size));
      CHECK(relocated_size == reference.length);
      CHECK(memcmp(relocated, reference.bytes, reference.length) == 0);
   }

   /* Moving two formerly independent ranges into the 16-bit base-address
    * reuse window changes the loader schedule and must not be patched. */
   struct agx3_ioto_uniform_range merged[ARRAY_SIZE(template_ranges)];
   memcpy(merged, template_ranges, sizeof(merged));
   merged[1].address = merged[0].address + 0x100;
   struct agx3_g17p_graphics_program_desc merged_desc = template_desc;
   merged_desc.uniform_ranges = merged;
   uint8_t refused[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t refused_size = UINT16_C(0xa5a5);
   CHECK(!agx3_g17p_graphics_program_template_relocate(
      AGX_CHIP_G17P, &template_desc, &merged_desc, &program_template,
      refused, sizeof(refused), &refused_size));
   CHECK(refused_size == UINT16_C(0xa5a5));

   /* Components may be interleaved in uniform order and connected only
    * transitively through the 16-bit address-cache window. Moving whole
    * components may also reverse their address order. Neither operation
    * changes the program, except for its relocated immediate addresses. */
   struct agx3_ioto_uniform_range chained_ranges[] = {
      {.start_halfs = 0, .size_halfs = 2, .address = UINT64_C(0x400002e340)},
      {.start_halfs = 2, .size_halfs = 2, .address = UINT64_C(0x4000212340)},
      {.start_halfs = 4, .size_halfs = 2, .address = UINT64_C(0x4000412340)},
      {.start_halfs = 6, .size_halfs = 2, .address = UINT64_C(0x4000028340)},
      {.start_halfs = 8, .size_halfs = 2, .address = UINT64_C(0x4000216340)},
      {.start_halfs = 10, .size_halfs = 2, .address = UINT64_C(0x4000022340)},
   };
   struct agx3_g17p_graphics_program_desc chained_desc = template_desc;
   chained_desc.uniform_ranges = chained_ranges;
   chained_desc.uniform_range_count = ARRAY_SIZE(chained_ranges);
   struct agx3_g17p_graphics_program_template chained_template;
   CHECK(agx3_g17p_graphics_program_template_generate(
      AGX_CHIP_G17P, &chained_desc, &chained_template));

   struct agx3_ioto_uniform_range chained_moved[ARRAY_SIZE(chained_ranges)];
   memcpy(chained_moved, chained_ranges, sizeof(chained_moved));
   const uint64_t shifts[] = {0x600000, 0x800000, 0, 0x600000, 0x800000,
                              0x600000};
   for (unsigned i = 0; i < ARRAY_SIZE(chained_moved); ++i)
      chained_moved[i].address += shifts[i];
   struct agx3_g17p_graphics_program_desc chained_target = chained_desc;
   chained_target.uniform_ranges = chained_moved;
   struct agx3_g17p_graphics_program chained_reference;
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &chained_target, &chained_reference));
   uint8_t chained_output[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t chained_length;
   CHECK(agx3_g17p_graphics_program_template_relocate(
      AGX_CHIP_G17P, &chained_desc, &chained_target, &chained_template,
      chained_output, sizeof(chained_output), &chained_length));
   CHECK(chained_length == chained_reference.length);
   CHECK(memcmp(chained_output, chained_reference.bytes, chained_length) == 0);

   /* Altering an offset within a component, splitting one component, or
    * merging two components must fall back to full generation. */
   chained_moved[3].address += 4;
   CHECK(!agx3_g17p_graphics_program_template_relocate(
      AGX_CHIP_G17P, &chained_desc, &chained_target, &chained_template,
      chained_output, sizeof(chained_output), &chained_length));
   chained_moved[3].address -= 4;
   chained_moved[4].address += 0x100000;
   CHECK(!agx3_g17p_graphics_program_template_relocate(
      AGX_CHIP_G17P, &chained_desc, &chained_target, &chained_template,
      chained_output, sizeof(chained_output), &chained_length));
   chained_moved[4].address -= 0x100000;
   chained_moved[2].address = chained_moved[1].address + 0x4000;
   CHECK(!agx3_g17p_graphics_program_template_relocate(
      AGX_CHIP_G17P, &chained_desc, &chained_target, &chained_template,
      chained_output, sizeof(chained_output), &chained_length));

   /* Large ranges cross allocator rounds and therefore materialize the same
    * source more than once after the address cache is cleared.  Every such
    * site must relocate, not only the first LDIMM pair for each range. */
   struct agx3_ioto_uniform_range rollover_ranges[] = {
      {.start_halfs = 0, .size_halfs = 256,
       .address = UINT64_C(0x4001012340)},
      {.start_halfs = 256, .size_halfs = 256,
       .address = UINT64_C(0x4002012340)},
   };
   struct agx3_g17p_graphics_program_desc rollover_desc = template_desc;
   rollover_desc.uniform_ranges = rollover_ranges;
   rollover_desc.uniform_range_count = ARRAY_SIZE(rollover_ranges);
   struct agx3_g17p_graphics_program_template rollover_template;
   CHECK(agx3_g17p_graphics_program_template_generate(
      AGX_CHIP_G17P, &rollover_desc, &rollover_template));
   CHECK(rollover_template.relocation_count > ARRAY_SIZE(rollover_ranges));
   for (unsigned iteration = 0; iteration < 1000; ++iteration) {
      struct agx3_ioto_uniform_range moved[ARRAY_SIZE(rollover_ranges)];
      memcpy(moved, rollover_ranges, sizeof(moved));
      for (unsigned i = 0; i < ARRAY_SIZE(moved); ++i)
         moved[i].address += (uint64_t)(iteration + 1) *
                             (i + 1) * UINT64_C(0x40000);
      struct agx3_g17p_graphics_program_desc moved_desc = rollover_desc;
      moved_desc.uniform_ranges = moved;
      struct agx3_g17p_graphics_program reference;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &moved_desc, &reference));
      uint8_t relocated[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      uint16_t relocated_size;
      CHECK(agx3_g17p_graphics_program_template_relocate(
         AGX_CHIP_G17P, &rollover_desc, &moved_desc, &rollover_template,
         relocated, sizeof(relocated), &relocated_size));
      CHECK(relocated_size == reference.length);
      CHECK(memcmp(relocated, reference.bytes, reference.length) == 0);
   }
   /* Firefox's WebRender through the GL driver (J700, 2026-09-18) reaches a
    * nine-source vertex shape that a seven-range budget refused outright:
    * the texture/sampler pointer pair, six attributes' VBO bases and clamps,
    * draw parameters, the stage table's texture and UBO bases, the UBO size,
    * two 16-bit root sysvals, and two halves of promoted constants.  The
    * loader bounds a program by chunks and body size, never by range count,
    * so the ninth source must compose like the first. */
   struct agx3_ioto_uniform_range webrender_vertex_ranges[] = {
      {.start_halfs = 0, .size_halfs = 8, .address = 0x1000},
      {.start_halfs = 8, .size_halfs = 24, .address = 0x2048},
      {.start_halfs = 32, .size_halfs = 12, .address = 0x2148},
      {.start_halfs = 44, .size_halfs = 4, .address = 0x3000},
      {.start_halfs = 48, .size_halfs = 8, .address = 0x4000},
      {.start_halfs = 56, .size_halfs = 2, .address = 0x4108},
      {.start_halfs = 58, .size_halfs = 2, .address = 0x2260},
      {.start_halfs = 60, .size_halfs = 2, .address = 0x2270},
      {.start_halfs = 62, .size_halfs = 2, .address = 0x5000},
   };
   const size_t webrender_vertex_range_count =
      sizeof(webrender_vertex_ranges) / sizeof(webrender_vertex_ranges[0]);
   CHECK(webrender_vertex_range_count > 7);
   CHECK(webrender_vertex_range_count <= AGX3_G17P_GRAPHICS_MAX_RANGES);
   struct agx3_g17p_graphics_program_desc webrender_vertex_desc = {
      .program_va = UINT64_C(0x10000500000),
      .shader_va = UINT64_C(0x100006003c0),
      .uniform_ranges = webrender_vertex_ranges,
      .uniform_range_count = webrender_vertex_range_count,
      .shader_binary_size = 640,
      .nr_gprs = 24,
   };
   struct agx3_g17p_graphics_program webrender_vertex_program;
   size_t webrender_vertex_bound = 0;
   CHECK(agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, webrender_vertex_ranges, webrender_vertex_range_count,
      &webrender_vertex_bound));
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &webrender_vertex_desc, &webrender_vertex_program));
   CHECK(webrender_vertex_program.length <= webrender_vertex_bound);
   CHECK(webrender_vertex_program.length <=
         AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);
   const struct agx3_g17p_graphics_program webrender_vertex_baseline =
      webrender_vertex_program;
   for (unsigned i = 0; i < webrender_vertex_range_count; ++i) {
      webrender_vertex_ranges[i].address += 0x40;
      CHECK(agx3_g17p_graphics_program_generate(
         AGX_CHIP_G17P, &webrender_vertex_desc, &webrender_vertex_program));
      CHECK(webrender_vertex_program.length ==
            webrender_vertex_baseline.length);
      CHECK(memcmp(webrender_vertex_program.bytes,
                   webrender_vertex_baseline.bytes,
                   webrender_vertex_program.length) != 0);
      webrender_vertex_ranges[i].address -= 0x40;
   }

   /* The budget is still a budget: one past it is refused, and the refusal
    * leaves the caller's program untouched. */
   struct agx3_ioto_uniform_range
      over_budget[AGX3_G17P_GRAPHICS_MAX_RANGES + 1];
   for (unsigned i = 0; i < AGX3_G17P_GRAPHICS_MAX_RANGES + 1; ++i) {
      over_budget[i] = (struct agx3_ioto_uniform_range){
         .start_halfs = 2 * i, .size_halfs = 2, .address = 0x8000 + 0x10 * i,
      };
   }
   struct agx3_g17p_graphics_program_desc over_budget_desc =
      webrender_vertex_desc;
   over_budget_desc.uniform_ranges = over_budget;
   over_budget_desc.uniform_range_count = AGX3_G17P_GRAPHICS_MAX_RANGES + 1;
   size_t over_budget_bound = 0;
   CHECK(!agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, over_budget, AGX3_G17P_GRAPHICS_MAX_RANGES + 1,
      &over_budget_bound));
   struct agx3_g17p_graphics_program untouched = webrender_vertex_baseline;
   CHECK(!agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &over_budget_desc, &untouched));
   CHECK(memcmp(&untouched, &webrender_vertex_baseline,
                sizeof(untouched)) == 0);
   /* J700's 32-range workload fits the byte budget. The shared descriptor
    * limit also permits larger, coalescible lists; it does not promise that
    * every uncoalesced list fits the program's byte limit. */
   over_budget_desc.uniform_range_count = 32;
   CHECK(agx3_g17p_graphics_program_bound(
      AGX_CHIP_G17P, over_budget, over_budget_desc.uniform_range_count,
      &over_budget_bound));
   CHECK(agx3_g17p_graphics_program_generate(
      AGX_CHIP_G17P, &over_budget_desc, &untouched));
   CHECK(untouched.length <= over_budget_bound);

   uint64_t clear_spec = 0, reload_spec = 0, none_spec = 0;
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_CLEAR, &clear_spec));
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_RELOAD, &reload_spec));
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_NONE, &none_spec));
   CHECK(clear_spec == reload_spec && clear_spec != none_spec);
   CHECK((clear_spec >> 47) == UINT64_C(0x1ffff));
   CHECK(!agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17S, AGX3_G17P_RT0_BG_CLEAR, &clear_spec));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_RT0_CLEAR_PUSH_HALFS, false));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS, false));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR,
      AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS, false));
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_RT0_CLEAR_PUSH_HALFS, true));
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_RELOAD, 0, true));

   struct agx3_g17p_rt0_bg_program_desc bg = {
      .program_va = UINT64_C(0x10000100000),
      .shader_va = UINT64_C(0x100002003c0),
      .clear_value_va = UINT64_C(0x4000010000),
      .shader_binary_size = 84,
      .nr_gprs = 8,
      .staging_known = true,
      .clear_push_halfs = AGX3_G17P_RT0_CLEAR_PUSH_HALFS,
      .op = AGX3_G17P_RT0_BG_CLEAR,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct agx3_g17p_rt0_bg_program bg_program;
   memset(&bg_program, 0xa5, sizeof(bg_program));
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE);
   bg.clear_push_halfs = AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS;
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE);
   bg.clear_push_halfs = AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS;
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE);
   bg.clear_push_halfs = AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS;
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, bg.clear_push_halfs, false));
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_QUAD_CLEAR_PROGRAM_MAX_SIZE);
   for (unsigned count = 5; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      bg.clear_push_halfs = 4 + 8 * count;
      CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                              &bg_program));
      CHECK(bg_program.length > 0 &&
            bg_program.length <=
               AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE);
   }
   bg.clear_push_halfs = AGX3_G17P_OCTA_CLEAR_PUSH_HALFS + 8;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));
   bg.clear_push_halfs = AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS - 1;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));
   bg.clear_value_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));

   bg.clear_value_va = 0;
   bg.texture_pointer_va = UINT64_C(0x4000020000);
   bg.sampler_pointer_va = UINT64_C(0x4000020008);
   bg.clear_push_halfs = 0;
   bg.op = AGX3_G17P_RT0_BG_RELOAD;
   CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                           &bg_program));
   CHECK(bg_program.length > 0 &&
         bg_program.length <= AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE);
   bg.sampler_pointer_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));

   /* Reloaded targets precede cleared ones. Each clear value keeps its
    * target's eight halves, which start after the texture/sampler pointers
    * only when at least one target reloads. */
   CHECK(agx3_g17p_rt0_bg_resource_spec(
      AGX_CHIP_G17P, AGX3_G17P_RT0_BG_CLEAR_RELOAD, &clear_spec));
   CHECK(clear_spec == reload_spec);
   CHECK(agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR_RELOAD, AGX3_G17P_QUAD_CLEAR_PUSH_HALFS, true));
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR_RELOAD, AGX3_G17P_QUAD_CLEAR_PUSH_HALFS, false));
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR_RELOAD, AGX3_G17P_RT0_CLEAR_PUSH_HALFS, true));
   /* Five to eight mixed-format targets: eight halves per clear value. */
   for (unsigned count = 5; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      CHECK(agx3_g17p_rt0_bg_shader_resources_match(
         AGX3_G17P_RT0_BG_CLEAR, 4 + 8 * count, false));
      CHECK(agx3_g17p_rt0_bg_shader_resources_match(
         AGX3_G17P_RT0_BG_CLEAR_RELOAD, 4 + 8 * count, true));
   }
   CHECK(!agx3_g17p_rt0_bg_shader_resources_match(
      AGX3_G17P_RT0_BG_CLEAR, AGX3_G17P_OCTA_CLEAR_PUSH_HALFS + 8, false));
   bg.sampler_pointer_va = UINT64_C(0x4000020008);
   bg.clear_value_va = UINT64_C(0x4000010000);
   bg.op = AGX3_G17P_RT0_BG_CLEAR_RELOAD;
   for (unsigned count = 2; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      bg.clear_push_halfs = 4 + 8 * count;
      for (unsigned reloads = 1; reloads < count; ++reloads) {
         bg.clear_start_halfs = 4 + 8 * reloads;
         CHECK(agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                                 &bg_program));
         CHECK(bg_program.length > 0 &&
               bg_program.length <=
                  (count > 4 ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
                             : AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE));
      }
      bg.clear_start_halfs = 4;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
      bg.clear_start_halfs = 4 + 8 * count;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
      bg.clear_start_halfs = 13;
      CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                               &bg_program));
   }
   bg.clear_start_halfs = 12;
   bg.clear_value_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));
   bg.clear_value_va = UINT64_C(0x4000010000);
   bg.texture_pointer_va = 0;
   CHECK(!agx3_g17p_rt0_bg_program_generate(AGX_CHIP_G17P, &bg,
                                            &bg_program));

   struct agx3_g17p_rt0_eot_program_desc eot = {
      .program_va = UINT64_C(0x10000300000),
      .shader_va = UINT64_C(0x100004003c0),
      .pbe_argument_va = UINT64_C(0x4000030000),
      .attachment_count = 1,
      .shader_binary_size = 84,
      .nr_gprs = 8,
      .staging_halfs = 8,
      .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
   };
   struct agx3_g17p_rt0_eot_program eot_program;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   eot.pbe_argument_va++;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.pbe_argument_va = UINT64_C(0x4000030000);
   eot.shader_binary_size = 120;
   eot.staging_halfs = 16;
   eot.attachment_count = 2;
   eot.imageblock_offset1_B = 4;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   eot.imageblock_offset1_B = 8;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.shader_binary_size = 238;
   eot.imageblock.sample_size_B = 16;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   eot.imageblock_offset1_B = 4;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));
   eot.imageblock_offset1_B = 12;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   /* Four native PBE entries share one argument table. Both witnessed tile
    * strides use the same shader and sparse offset uniforms. */
   for (unsigned stride = 4; stride <= 16; stride *= 4) {
      struct agx3_g17p_rt0_eot_program_desc quad = eot;
      quad.shader_binary_size = 208;
      quad.staging_halfs = 32;
      quad.attachment_count = 4;
      quad.imageblock_offset1_B = stride;
      quad.imageblock_offset2_B = 2 * stride;
      quad.imageblock_offset3_B = 3 * stride;
      quad.imageblock.sample_size_B = 4 * stride;
      quad.imageblock.tile_size_px = stride == 16 ? 16 * 16 : 32 * 32;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                             &eot_program));
      CHECK(eot_program.length == 144);
      quad.imageblock_offset3_B++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.imageblock_offset3_B--;
      quad.staging_halfs = 16;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.staging_halfs = 32;
      quad.attachment_count = 3;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.attachment_count = 4;
      quad.pbe_argument_va++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
      quad.pbe_argument_va = UINT64_C(0xffffffffffe0);
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &quad,
                                              &eot_program));
   }

   /* Three RGBA8 attachments occupy twelve bytes but the tilebuffer rounds
    * the sample stride to sixteen. The native EOT still publishes offsets
    * 0, 4 and 8, with no fourth offset operand. */
   struct agx3_g17p_rt0_eot_program_desc rgba8_three = eot;
   rgba8_three.shader_binary_size = 164;
   rgba8_three.staging_halfs = 24;
   rgba8_three.attachment_count = 3;
   rgba8_three.imageblock_offset1_B = 4;
   rgba8_three.imageblock_offset2_B = 8;
   rgba8_three.imageblock_offset3_B = 0;
   rgba8_three.imageblock.sample_size_B = 16;
   rgba8_three.imageblock.tile_size_px = 32 * 32;
   CHECK(agx3_g17p_rt0_eot_program_generate(
      AGX_CHIP_G17P, &rgba8_three, &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE);
   rgba8_three.imageblock.sample_size_B = 12;
   CHECK(!agx3_g17p_rt0_eot_program_generate(
      AGX_CHIP_G17P, &rgba8_three, &eot_program));

   /* Two and three RGBA32 targets retain the same sparse per-attachment
    * offset ABI, with only their actual descriptor and staging counts. */
   for (unsigned count = 2; count <= 3; ++count) {
      struct agx3_g17p_rt0_eot_program_desc raw = eot;
      raw.shader_binary_size = 32 + 44 * count;
      raw.staging_halfs = 8 * count;
      raw.attachment_count = count;
      raw.imageblock_offset1_B = 16;
      raw.imageblock_offset2_B = count > 2 ? 32 : 0;
      raw.imageblock_offset3_B = 0;
      raw.imageblock.sample_size_B = 16 * count;
      raw.imageblock.tile_size_px = count == 2 ? 32 * 16 : 16 * 16;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                             &eot_program));
      CHECK(eot_program.length > 0 && eot_program.length <= 144);
      raw.imageblock_offset3_B = 48;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
      raw.imageblock_offset3_B = 0;
      raw.staging_halfs -= 8;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
      raw.staging_halfs += 8;
      raw.attachment_count++;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
      raw.attachment_count--;
      raw.pbe_argument_va = UINT64_C(0xffffffffffe0);
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &raw,
                                              &eot_program));
   }

   /* Targets of arbitrary formats keep the multi-target program and offset
    * uniforms, but each target may sit at any offset inside the sample. */
   for (unsigned count = 2; count <= AGX3_G17P_MRT_MAX_TARGETS; ++count) {
      struct agx3_g17p_rt0_eot_program_desc mixed = eot;
      static const uint8_t offsets[AGX3_G17P_MRT_MAX_TARGETS] = {
         16, 24, 0, 8, 32, 40, 48, 56};
      mixed.shader_binary_size = 32 + 44 * count;
      mixed.staging_halfs = 8 * count;
      mixed.attachment_count = count;
      for (unsigned rt = 0; rt < AGX3_G17P_MRT_MAX_TARGETS; ++rt)
         *eot_offset(&mixed, rt) = rt < count ? offsets[rt] : 0;
      mixed.imageblock.sample_size_B = count > 4 ? 64 : 32;
      mixed.imageblock.tile_size_px = count > 4 ? 16 * 16 : 32 * 16;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &mixed,
                                              &eot_program));
      mixed.mixed_mrt = true;
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &mixed,
                                             &eot_program));
      /* Each target beyond the second adds one offset LDIMM and two COPYs. */
      CHECK(eot_program.length > 0 &&
            eot_program.length <= (count > 4
                                      ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                                      : 144));

      struct agx3_g17p_rt0_eot_program_desc bad = mixed;
      bad.imageblock_offset1_B = mixed.imageblock.sample_size_B;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
      bad = mixed;
      if (count < AGX3_G17P_MRT_MAX_TARGETS) {
         *eot_offset(&bad, count) = 4;
         CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                                 &eot_program));
      }
      bad = mixed;
      bad.staging_halfs -= 8;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
      /* The multisample program has the same shape; the imageblock carries
       * the count. */
      if (count <= 4) {
         bad = mixed;
         bad.imageblock.nr_samples = 2;
         bad.imageblock.tile_size_px = 16 * 16;
         CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                                &eot_program));
      }
      bad = mixed;
      bad.imageblock.nr_samples = 3;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
      bad = mixed;
      bad.pbe_argument_va = UINT64_C(0xffffffffffe0);
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &bad,
                                              &eot_program));
   }

   /* dEQP-VK.renderpasses.*.attachment.8.620 stores eight targets of an
    * 80-byte sample. Each offset reaches its uniform unchanged, including
    * those past 64: Apple's configurePBE writes
    * ((draw_state >> 12) & 255) % sample_stride, which is the offset itself
    * for any target inside the sample. */
   {
      static const uint8_t offsets[AGX3_G17P_MRT_MAX_TARGETS] = {
         0, 16, 48, 56, 60, 68, 32, 72};
      struct agx3_g17p_rt0_eot_program_desc wide = eot;
      wide.shader_binary_size = 32 + 44 * 8;
      wide.staging_halfs = 8 * 8;
      wide.attachment_count = 8;
      wide.mixed_mrt = true;
      wide.imageblock.sample_size_B = 80;
      wide.imageblock.tile_size_px = 16 * 16;
      uint8_t loads[2 * AGX3_G17P_MRT_MAX_TARGETS];
      for (unsigned rt = 0; rt < AGX3_G17P_MRT_MAX_TARGETS; ++rt) {
         *eot_offset(&wide, rt) = offsets[rt];
         loads[2 * rt] = 0x0c | ((8 + 2 * rt) >> 1) << 4;
         loads[2 * rt + 1] = offsets[rt];
      }
      CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &wide,
                                               &eot_program));
      bool found = false;
      for (size_t at = 0; !found && at + sizeof(loads) <= eot_program.length;
           ++at)
         found = !memcmp(eot_program.bytes + at, loads, sizeof(loads));
      CHECK(found);
      wide.imageblock_offset7_B = 80;
      CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &wide,
                                                &eot_program));
   }

   /* The native layered array EOT is an 84-byte, eight-half program. It keeps
    * the same single-attachment Ioto resource ABI and 128-byte LDSHDR fetch
    * span as the other admitted RT0 builtins. */
   eot.pbe_argument_va = UINT64_C(0x4000030000);
   eot.shader_binary_size = 84;
   eot.imageblock.sample_size_B = 8;
   eot.nr_gprs = 8;
   eot.staging_halfs = 8;
   eot.attachment_count = 1;
   eot.imageblock_offset1_B = 0;
   CHECK(agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                            &eot_program));
   CHECK(eot_program.length > 0 &&
         eot_program.length <= AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE);
   eot.shader_binary_size = 110;
   eot.nr_gprs = 16;
   CHECK(!agx3_g17p_rt0_eot_program_generate(AGX_CHIP_G17P, &eot,
                                             &eot_program));

   return 0;
}
