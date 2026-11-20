/* SPDX-License-Identifier: MIT */


#include "agx3_graphics_program.h"
#include "agx3_copy_core.h"
#include "agx3_binder.h"
#include "asahi/lib/agx_ioto_program_core.h"
#include "agx3_ioto_load_packer.h"
#include "agx3_cprog_input.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
#include "asahi/lib/agx_ppp_hal300_graphics_core.h"
#include "asahi/lib/agx3_compute_program_core.h"
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
test_g17g_programs(enum agx_chip chip)
{

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
         .imageblock = { .sample_size_B = 8, .nr_samples = 1, .tile_size_px = 1024 },
      };
      struct agx3_g17p_graphics_program output;
      size_t empty_bound;
      CHECK(agx3_g17g_graphics_program_bound(chip, NULL, 0,
                                              &empty_bound));
      CHECK(empty_bound == 90);
      CHECK(agx3_g17g_graphics_program_generate(chip, &desc,
                                                 &output));
      CHECK(output.length == (fragment ? 90 : 84));
      CHECK(memcmp(output.bytes, expected_tail[fragment], 28) == 0);
      CHECK(output.bytes[48] == (fragment ? 4 : 8) *
            ((chip == AGX_CHIP_G17S || chip == AGX_CHIP_G17C) ? 2 : 1));


      {
         const unsigned d[] = {256, 288, 320, 512};
         const uint8_t expected_spectpr[][10] = {
            {0x07,0x04,0x00,0x0d,0,0,0,0,0,0},
            {0x87,0x04,0x00,0x0d,0,0,0,0,0,0},
            {0x07,0x05,0x00,0x0d,0,0,0,0,0,0},
            {0x07,0x08,0x00,0x0d,0,0,0,0,0,0},
         };
         for (unsigned i = 0; i < ARRAY_SIZE(d); ++i) {
            struct agx3_g17p_graphics_program_desc wide = desc;
            struct agx3_g17p_graphics_program wide_output;
            wide.compiler_d = d[i];
            CHECK(agx3_g17g_graphics_program_generate(chip, &wide, &wide_output));
            CHECK(wide_output.length == output.length);
            CHECK(memcmp(wide_output.bytes + 18, expected_spectpr[i], 10) == 0);
         }
      }
      const uint32_t scratch_bytes[] = {4, 272, 400, 528, 1040, 262140};
      const uint8_t scratch_fields[][4] = {
         {1, 0, 1, 0}, {68, 0, 68, 0}, {100, 0, 100, 0},
         {132, 0, 132, 0}, {4, 1, 4, 1}, {255, 255, 255, 255},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(scratch_bytes); ++i) {
         struct agx3_g17p_graphics_program_desc spill = desc;
         struct agx3_g17p_graphics_program spill_output;
         spill.scratch_size = scratch_bytes[i];
         CHECK(agx3_g17g_graphics_program_generate(chip, &spill, &spill_output));
         CHECK(spill_output.length == output.length);
         CHECK(memcmp(spill_output.bytes + 24, scratch_fields[i], 4) == 0);
         CHECK(memcmp(spill_output.bytes, output.bytes, 24) == 0);
      }
      struct agx3_g17p_graphics_program_desc too_wide = desc;
      too_wide.compiler_d = 513;
      CHECK(!agx3_g17g_graphics_program_generate(chip, &too_wide, &output));


      if (fragment) {
         struct agx3_g17p_graphics_program_desc mrt = desc;
         mrt.imageblock.sample_size_B = 32;
         struct agx3_g17p_graphics_program mrt_output;
         CHECK(agx3_g17g_graphics_program_generate(chip, &mrt,
                                                    &mrt_output));
         uint8_t expected_mrt[28];
         memcpy(expected_mrt, expected_tail[1], sizeof(expected_mrt));
         expected_mrt[7] = 0x20;
         CHECK(mrt_output.length == output.length);
         CHECK(memcmp(mrt_output.bytes, expected_mrt, 28) == 0);


         struct agx3_g17p_graphics_program_desc bare = desc;
         bare.imageblock.sample_size_B = 0;
         struct agx3_g17p_graphics_program bare_output;
         CHECK(agx3_g17g_graphics_program_generate(chip, &bare,
                                                    &bare_output));
         uint8_t expected_bare[28];
         memcpy(expected_bare, expected_tail[1], sizeof(expected_bare));
         memcpy(expected_bare, expected_tail[0], 8);
         CHECK(memcmp(bare_output.bytes, expected_bare, sizeof(expected_bare)) == 0);

         for (unsigned stride = 8; stride <= 64; stride += 8) {
            mrt.imageblock.sample_size_B = stride;
            CHECK(agx3_g17g_graphics_program_generate(chip, &mrt, &mrt_output));
            expected_mrt[7] = stride;
            CHECK(memcmp(mrt_output.bytes, expected_mrt, sizeof(expected_mrt)) == 0);
         }

         const uint8_t ungrounded[] = {4, 12, 65};
         for (unsigned i = 0; i < sizeof(ungrounded); ++i) {
            struct agx3_g17p_graphics_program_desc bad = desc;
            bad.imageblock.sample_size_B = ungrounded[i];
            CHECK(!agx3_g17g_graphics_program_generate(chip, &bad,
                                                        &mrt_output));
         }
      }

      const uint32_t original_size = desc.shader_binary_size;
      desc.shader_binary_size = 512;
      CHECK(agx3_g17g_graphics_program_generate(chip, &desc, &output));
      const struct agx3_g17p_graphics_program saturated = output;
      const uint32_t larger_sizes[] = {513, 790, 1024, 65536, UINT32_MAX};
      for (unsigned i = 0; i < sizeof(larger_sizes)/sizeof(larger_sizes[0]); ++i) {
         desc.shader_binary_size = larger_sizes[i];
         CHECK(agx3_g17g_graphics_program_generate(chip, &desc, &output));
         CHECK(output.length == saturated.length &&
               !memcmp(output.bytes, saturated.bytes, output.length));
      }
      desc.shader_binary_size = 0;
      CHECK(!agx3_g17g_graphics_program_generate(chip, &desc, &output));
      desc.shader_binary_size = original_size;

      if (fragment) {
         const struct agx3_ioto_uniform_range pointer = {
            .start_halfs = 0, .size_halfs = 6, .address = 0x4000001000,
         };
         const uint8_t copies[] = {
            0x0b,0x04,0x09,0x24, 0x1b,0x06,0x09,0x04,
            0x2b,0x08,0x09,0x04, 0x0e,0x00,0x00,0x00,
         };
         struct agx3_g17p_graphics_program_desc pointer_desc = desc;
         pointer_desc.uniform_ranges = &pointer;
         pointer_desc.uniform_range_count = 1;
         CHECK(agx3_g17g_tile_program_generate(chip, &pointer_desc, true, &output));
         CHECK(output.length >= sizeof(copies));
         CHECK(!memcmp(output.bytes + output.length - sizeof(copies),
                       copies, sizeof(copies)));
      }

      struct agx3_ioto_uniform_range ranges[] = {
         {.start_halfs = 0, .size_halfs = 4, .address = 0x4000001000},
         {.start_halfs = 4, .size_halfs = 2, .address = 0x4000002000},
      };
      desc.uniform_ranges = ranges;
      desc.uniform_range_count = 2;
      size_t bound;
      CHECK(agx3_g17g_graphics_program_bound(chip, ranges, 2, &bound));
      CHECK(agx3_g17g_graphics_program_generate(chip, &desc,
                                                 &output));
      CHECK(output.length > 90 && output.length <= bound);
      struct agx3_g17p_graphics_program baseline = output;
      for (unsigned i = 0; i < 2; ++i) {
         ranges[i].address += 64;
         CHECK(agx3_g17g_graphics_program_generate(chip, &desc,
                                                    &output));
         CHECK(output.length == baseline.length &&
               memcmp(output.bytes, baseline.bytes, output.length) != 0);
         ranges[i].address -= 64;
      }

      baseline = output;
      desc.scratch_size = UINT16_MAX * 4u + 1;
      CHECK(!agx3_g17g_graphics_program_generate(chip, &desc, &output));
      desc.scratch_size = 0;
      desc.compiler_d_valid = false;
      CHECK(!agx3_g17g_graphics_program_generate(chip, &desc, &output));
      desc.compiler_d_valid = true;
      desc.shader_va++;
      CHECK(!agx3_g17g_graphics_program_generate(chip, &desc, &output));
      desc.shader_va--;
      CHECK(!agx3_g17g_graphics_program_generate(AGX_CHIP_G17P, &desc, &output));
      CHECK(memcmp(&output, &baseline, sizeof(output)) == 0);
      if (fragment) {
         static const uint8_t bg_end[] = {7, 0x10, 0, 4, 0, 0, 14, 0, 0, 0};
         static const uint8_t eot_end[] = {14, 0, 0, 0};
         CHECK(agx3_g17g_tile_program_generate(chip, &desc, false,
                                               &output));
         CHECK(output.length == baseline.length);
         CHECK(memcmp(output.bytes + output.length - sizeof(bg_end),
                      bg_end, sizeof(bg_end)) == 0);
         CHECK(agx3_g17g_tile_program_generate(chip, &desc, true,
                                               &output));
         CHECK(output.length + 6 == baseline.length);
         CHECK(memcmp(output.bytes + output.length - sizeof(eot_end),
                      eot_end, sizeof(eot_end)) == 0);
      } else {
         CHECK(!agx3_g17g_tile_program_generate(chip, &desc, false,
                                                &output));

         desc.uniform_ranges = NULL;
         desc.uniform_range_count = 0;
         desc.shader_va = UINT64_C(0x100000003c0);
         desc.shader_binary_size = 64;
         desc.compiler_d = 0;
         CHECK(agx3_g17g_compute_program_generate(chip, &desc, &output));
         CHECK(output.length == 84);
         CHECK(!memcmp(output.bytes, expected_tail[0], 8));
         CHECK(!memcmp(output.bytes + 8, expected_tail[1] + 8, 20));
         CHECK(output.bytes[28 + 20] ==
               ((chip == AGX_CHIP_G17S || chip == AGX_CHIP_G17C) ? 8 : 4));
         static const uint8_t end[] = {14, 0, 0, 0};
         CHECK(!memcmp(output.bytes + 80, end, sizeof(end)));

         const struct {
            unsigned bytes;
            uint8_t spectpr[10];
         } staging[] = {
            {256, {0x07,0x04,0x00,0x0d,0,0,0,0,0,0}},
            {384, {0x07,0x06,0x00,0x0d,0,0,0,0,0,0}},
            {512, {0x07,0x08,0x00,0x0d,0,0,0,0,0,0}},
         };
         for (unsigned i = 0; i < ARRAY_SIZE(staging); ++i) {
            desc.compiler_d = staging[i].bytes;
            CHECK(agx3_g17g_compute_program_generate(chip, &desc, &output));
            CHECK(output.length == 84);
            CHECK(!memcmp(output.bytes + 18, staging[i].spectpr, 10));
         }
         desc.compiler_d = 513;
         CHECK(!agx3_g17g_compute_program_generate(chip, &desc, &output));
         desc.compiler_d = 0;

         const struct {
            uint32_t size;
            uint8_t speclm[8];
         } shared[] = {
            {256,   {0x0f,0x20,0x00,0x0c,0,0,0,0}},
            {512,   {0x0f,0x40,0x00,0x0c,0,0,0,0}},
            {1024,  {0x0f,0x80,0x00,0x0c,0,0,0,0}},
            {4096,  {0x0f,0x00,0x10,0x0c,0,0,0,0}},
            {16384, {0x0f,0x00,0x40,0x0c,0,0,0,0}},
            {32768, {0x0f,0x00,0x80,0x0c,0,0,0,0}},
         };
         for (unsigned i = 0; i < ARRAY_SIZE(shared); ++i) {
            desc.threadgroup_memory_size_B = shared[i].size;
            CHECK(agx3_g17g_compute_program_generate(chip, &desc, &output));
            CHECK(output.length == 84);
            CHECK(!memcmp(output.bytes, shared[i].speclm, 8));
            CHECK(!agx3_g17g_graphics_program_generate(chip, &desc, &output));
         }
         const uint32_t invalid_shared[] = {1, 255, 257, 32769, UINT32_MAX};
         for (unsigned i = 0; i < ARRAY_SIZE(invalid_shared); ++i) {
            desc.threadgroup_memory_size_B = invalid_shared[i];
            CHECK(!agx3_g17g_compute_program_generate(chip, &desc, &output));
         }
         desc.threadgroup_memory_size_B = 0;
         desc.fragment = true;
         CHECK(!agx3_g17g_compute_program_generate(chip, &desc, &output));
      }
   }
   return 0;
}


static int
test_m5_dispatch(enum agx_chip chip)
{
   struct agx3_g17p_native_compute_program_desc desc = {
      .program_va = UINT64_C(0x10000090000),
      .shader_va = UINT64_C(0x100000003c0),
   };
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &desc.launch_sources));
   struct agx3_compute_program program;
   CHECK(agx3_m5_native_compute_program_bind(chip, &desc, &program));
   struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1024,1,1}, .local_size = {64,1,1}, .dimensions = 1,
   };
   const uint32_t expected[] = {
      0, 0x01000000, 0x2400, 0x40000001,
      1024, 1, 1, 64, 1, 1, 0x60000160,
   };
   uint8_t out[sizeof(expected)];
   struct agx3_compute_dispatch_sink sink = {.data = out, .capacity = sizeof(out)};
   CHECK(agx3_m5_native_compute_dispatch_append(&program, &grid, 0, &sink));
   CHECK(sink.length == sizeof(out));
   CHECK(!memcmp(out, expected, sizeof(out)));
   const uint32_t local[] = {64,2,1};
   const uint32_t indirect_expected[] = {
      0x08080000, 0x01000000, 0x2400, 0x40000001,
      0x23, 0x45678000, 64, 2, 1, 0x6000157c,
   };
   sink.length = 0;
   CHECK(agx3_m5_native_indirect_dispatch_append(
      &program, UINT64_C(0x2345678000), local, 5, &sink));
   CHECK(sink.length == sizeof(indirect_expected));
   CHECK(!memcmp(out, indirect_expected, sizeof(indirect_expected)));
   uint8_t saved[sizeof(out)];
   memcpy(saved, out, sizeof(out));
   CHECK(!agx3_m5_native_indirect_dispatch_append(
      &program, UINT64_C(0x2345678001), local, 5, &sink));
   CHECK(!memcmp(saved, out, sizeof(out)));
   const uint32_t indirect_local[] = {
      0x10080000, 0x01000000, 0x2400, 0x40000001,
      0x23, 0x45678000, 0x60000160,
   };
   sink.length = 0;
   CHECK(agx3_m5_native_indirect_dispatch_append(
      &program, UINT64_C(0x2345678000), NULL, 0, &sink));
   CHECK(sink.length == sizeof(indirect_local));
   CHECK(!memcmp(out, indirect_local, sizeof(indirect_local)));
   return 0;
}

/* Host model of libagx_trace_rays_select_launch (libagx/draws.cl): the words
 * it writes into the single-kick slot from a kernel's template. */
static void
select_launch_model(const uint32_t tmpl[11], bool run, uint32_t groups_x,
                    uint32_t height, uint32_t depth, unsigned log2,
                    uint32_t out[11])
{
   uint32_t word0 = tmpl[0];
   if (!run || (height < 2 && depth < 2))
      word0 &= 0xfff1ffffu;
   out[0] = word0;
   out[1] = tmpl[1];
   out[2] = tmpl[2];
   out[3] = tmpl[3];
   out[4] = run ? groups_x << log2 : 0;
   out[5] = run ? height : 0;
   out[6] = run ? depth : 0;
   out[7] = run ? tmpl[7] : 0;
   out[8] = run ? tmpl[8] : 0;
   out[9] = run ? tmpl[9] : 0;
   out[10] = tmpl[10];
}

/* The GPU-written single-kick launch must be byte-identical to the record the
 * host encodes for the same kernel and grid (hk_dispatch_m5_native). */
static int
test_m5_single_kick_launch(enum agx_chip chip)
{
   for (unsigned sources = 0; sources < 2; ++sources) {
      struct agx3_g17p_native_compute_program_desc desc = {
         .program_va = UINT64_C(0x10000090000),
         .shader_va = UINT64_C(0x100000003c0),
      };
      CHECK(agx3_cdm_direct_template_sources_compute(sources, sources, sources,
                                                     &desc.launch_sources));
      struct agx3_compute_program program;
      CHECK(agx3_m5_native_compute_program_bind(chip, &desc, &program));

      /* The template hk captures: a 3D grid of the kernel's local size. */
      const struct agx3_compute_direct_grid tmpl_grid = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {32, 2, 2}, .local_size = {32, 1, 1}, .dimensions = 3,
      };
      uint32_t tmpl[11];
      struct agx3_compute_dispatch_sink sink = {
         .data = (uint8_t *)tmpl, .capacity = sizeof(tmpl)};
      CHECK(agx3_m5_native_compute_dispatch_append(&program, &tmpl_grid, 0,
                                                    &sink));
      CHECK(sink.length == sizeof(tmpl));

      const uint32_t dims[][3] = {
         {1, 1, 1}, {31, 1, 1}, {32, 1, 1}, {33, 1, 1}, {1920, 1080, 1},
         {64, 1, 7}, {5, 3, 2}, {2976, 1872, 1},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(dims); ++i) {
         const uint32_t w = dims[i][0], h = dims[i][1], d = dims[i][2];
         const uint32_t groups_x = (w >> 5) + ((w & 31) ? 1 : 0);
         uint32_t model[11];
         select_launch_model(tmpl, true, groups_x, h, d, 5, model);

         /* hk_dispatch_m5_native's direct record for the same launch. */
         const struct agx3_compute_direct_grid grid = {
            .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
            .size = {groups_x * 32, h, d}, .local_size = {32, 1, 1},
            .dimensions = d != 1 ? 3 : h != 1 ? 2 : 1,
         };
         uint32_t host[11];
         sink = (struct agx3_compute_dispatch_sink){
            .data = (uint8_t *)host, .capacity = sizeof(host)};
         CHECK(agx3_m5_native_compute_dispatch_append(&program, &grid, 0,
                                                       &sink));
         CHECK(!memcmp(model, host, sizeof(host)));
      }

      /* No matching kernel: kernel 0's header, all six sizes zero, 1D bits. */
      uint32_t empty[11];
      select_launch_model(tmpl, false, 3, 4, 5, 5, empty);
      CHECK(empty[4] == 0 && empty[5] == 0 && empty[6] == 0 &&
            empty[7] == 0 && empty[8] == 0 && empty[9] == 0);
      CHECK(empty[10] == tmpl[10] && (empty[0] & 0x000e0000u) == 0);
   }
   return 0;
}

static int
test_color_disabled(void)
{
   struct agx_ppp_hal300_graphics_fields fields = {
      .fragment_program = 0x1000,
      .line_width = 15,
   };
   struct agx_ppp_hal300_graphics_core with_color, without_color;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S,&fields,&with_color));
   fields.no_color_writes=true;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S,&fields,&without_color));
   const unsigned offset = with_color.offsets[5] + 16;
   const uint8_t enabled[] = {0xff,0xff,0x01,0x00};
   CHECK(memcmp(with_color.bytes+offset,enabled,4)==0);
   memset(with_color.bytes+offset,0,4);
   CHECK(memcmp(with_color.bytes,without_color.bytes,sizeof(with_color.bytes))==0);
   return 0;
}


/* The native M5 driver's visibility-result-mode setter (macOS 26A428)
 * ORs the mode into bits 15:14 of the fragment control word and inserts
 * offset / 8 into bits 31:17 of the word 0x14 bytes after it: the last word
 * of the 0x020000e0 record. Disabled visibility leaves the bytes unchanged. */
static int
test_visibility(void)
{
   struct agx_ppp_hal300_graphics_fields fields = {
      .fragment_program = 0x1000,
   };
   struct agx_ppp_hal300_graphics_core none, counting, boolean;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &none));
   fields.visibility_mode = 2;
   fields.visibility_index = 5;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &counting));
   fields.visibility_mode = 3;
   fields.visibility_index = 0x7fff;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &boolean));
   const unsigned control = none.offsets[4] + 4, query = control + 0x14;
   uint32_t w;
   memcpy(&w, counting.bytes + control, 4);
   CHECK(w == (*(uint32_t *)(none.bytes + control) | 0x8000));
   memcpy(&w, counting.bytes + query, 4);
   CHECK(w == 5u << 17);
   memcpy(&w, boolean.bytes + control, 4);
   CHECK(w == (*(uint32_t *)(none.bytes + control) | 0xc000));
   memcpy(&w, boolean.bytes + query, 4);
   CHECK(w == 0x7fffu << 17);
   memcpy(counting.bytes + control, none.bytes + control, 4);
   memset(counting.bytes + query, 0, 4);
   CHECK(!memcmp(counting.bytes, none.bytes, sizeof(none.bytes)));
   fields.visibility_mode = 1;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &none));
   return 0;
}

static int
test_line_width(void)
{
   struct agx_ppp_hal300_graphics_fields fields = {
      .fragment_program = 0x1000,
   };
   struct agx_ppp_hal300_graphics_core state;
   const unsigned widths[] = {0, 15, 31, 127, 255};
   for (unsigned i = 0; i < ARRAY_SIZE(widths); ++i) {
      fields.line_width = widths[i];
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
      /* Each face carries the encoded width in bits 8..15. */
      CHECK(state.bytes[state.offsets[4] + 9] == widths[i]);
      CHECK(state.bytes[state.offsets[4] + 13] == widths[i]);
   }
   struct agx_ppp_hal300_graphics_core saved = state;
   fields.line_width = 256;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   CHECK(!memcmp(&saved, &state, sizeof(state)));
   return 0;
}


static int
test_polygon_and_primitive_msaa(void)
{
   struct agx_ppp_hal300_graphics_fields fields = {
      .fragment_program = 0x1000,
      .line_width = 15,
   };
   struct agx_ppp_hal300_graphics_core state;
   for (unsigned mode = 0; mode <= 2; ++mode) {
      fields.polygon_mode = mode;
      for (unsigned msaa = 0; msaa <= 1; ++msaa) {
         fields.primitive_msaa = msaa;
         CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
         CHECK(((state.bytes[state.offsets[4] + 10] >> 2) & 3) == mode);
         CHECK(((state.bytes[state.offsets[4] + 14] >> 2) & 3) == mode);
         CHECK(((state.bytes[state.offsets[7] + 5] >> 7) & 1) == msaa);
      }
   }
   struct agx_ppp_hal300_graphics_core saved = state;
   fields.polygon_mode = 3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   CHECK(!memcmp(&saved, &state, sizeof(state)));
   return 0;
}


static int
test_multiround_uniforms(enum agx_chip chip)
{
   struct agx3_ioto_uniform_range ranges[] = {
      { .start_halfs = 0, .size_halfs = 8, .address = 0x4000001000 },
      { .start_halfs = 8, .size_halfs = 180, .address = 0x5000010000 },
   };
   struct agx3_g17p_graphics_program_desc desc = {
      .program_va = 0x10000100000, .shader_va = 0x10000000480,
      .shader_binary_size = 128, .nr_gprs = 16,
      .compiler_d_valid = true, .uniform_ranges = ranges,
      .uniform_range_count = ARRAY_SIZE(ranges), .patchable = true,
   };
   struct agx3_g17p_graphics_program_template first, moved;
   CHECK(agx3_g17g_compute_program_template_generate(chip, &desc, &first));
   uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   struct agx_ioto_inflight_load loads[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   struct agx_ioto_inflight_load sort[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   struct agx_ioto_load_cache_entry cache[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   const struct agx3_ioto_uniform_program_desc load_desc = {
      .program_va = desc.program_va, .ranges = ranges,
      .range_count = ARRAY_SIZE(ranges), .patchable = true,
   };
   const struct agx3_ioto_uniform_program_workspace workspace = {
      .body = body, .body_capacity = sizeof(body),
      .loads = loads, .load_capacity = ARRAY_SIZE(loads),
      .sort = sort, .sort_capacity = ARRAY_SIZE(sort),
      .cache = cache, .cache_capacity = ARRAY_SIZE(cache),
   };
   size_t body_size, last_copy, first_copy;
   CHECK(agx3_ioto_uniform_program_body_generate(chip, &load_desc, workspace,
                                                 &body_size, &last_copy));
   CHECK(agx_ioto_program_first_copy_offset(chip, body, body_size, &first_copy));
   CHECK(first_copy < last_copy);

   struct agx3_g17p_graphics_program_desc empty_desc = desc;
   empty_desc.uniform_ranges = NULL;
   empty_desc.uniform_range_count = 0;
   struct agx3_g17p_graphics_program empty;
   CHECK(agx3_g17g_compute_program_generate(chip, &empty_desc, &empty));
   const size_t prefix = AGX_IOTO_STATE_TAIL_TYPE0_SIZE;
   CHECK(first.program.length == body_size + empty.length);
   CHECK(!memcmp(first.program.bytes, body, first_copy));
   CHECK(!memcmp(first.program.bytes + first_copy, empty.bytes, prefix));
   CHECK(!memcmp(first.program.bytes + first_copy + prefix,
                 body + first_copy, body_size - first_copy));
   /* Keep every upload, including later allocation rounds, before the
    * helper: a native instruction probe detects zero uniforms otherwise. */
   CHECK(!memcmp(first.program.bytes + body_size + prefix,
                 empty.bytes + prefix, empty.length - prefix));

   /* Template addresses after the early COPY round must still patch at
    * their actual positions in the composed instruction stream. */
   uint8_t patched[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   memcpy(patched, first.program.bytes, first.program.length);
   bool later_relocation = false;
   for (unsigned i = 0; i < first.relocation_count; ++i) {
      const struct agx_ioto_load_relocation *rel = &first.relocations[i];
      uint8_t low[8], high[8];
      const uint8_t ln = 8, hn = 8;
      agx_ioto_load_pack_ldimm_long(rel->dest_half, rel->address, low);
      agx_ioto_load_pack_ldimm_long(rel->dest_half + 2, rel->address >> 32, high);
      CHECK(ln == rel->low_length && hn == rel->high_length);
      CHECK(rel->offset + ln + hn <= first.program.length);
      CHECK(!memcmp(patched + rel->offset, low, ln));
      CHECK(!memcmp(patched + rel->offset + ln, high, hn));
      later_relocation |= rel->offset > first_copy;
      uint64_t address = rel->address + 0x100000;
      agx_ioto_load_pack_ldimm_long(rel->dest_half, address, low);
      agx_ioto_load_pack_ldimm_long(rel->dest_half + 2, address >> 32, high);
      memcpy(patched + rel->offset, low, ln);
      memcpy(patched + rel->offset + ln, high, hn);
   }
   CHECK(later_relocation);
   for (unsigned i = 0; i < ARRAY_SIZE(ranges); ++i)
      ranges[i].address += 0x100000;
   CHECK(agx3_g17g_compute_program_template_generate(chip, &desc, &moved));
   CHECK(first.program.length == moved.program.length);
   CHECK(!memcmp(patched, moved.program.bytes, moved.program.length));
   return 0;
}

/* Compute loaders that publish a few hundred constant-program inputs (UBO
 * contents read by a preamble, as in ubo.*_compute) must fit the M5 program
 * buffer, and the body bound must still cover what the loader writes for
 * aligned pairs, lone halves and pairs at odd compact positions. */
static int
test_cprog_input_bound(enum agx_chip chip)
{
   const struct agx3_ioto_uniform_range ranges[] = {
      { .start_halfs = 0, .size_halfs = 8, .address = 0x4000001000 },
      { .start_halfs = 8, .size_halfs = 128, .address = 0x5000010000 },
      { .start_halfs = 136, .size_halfs = 128, .address = 0x5000010240 },
      { .start_halfs = 264, .size_halfs = 128, .address = 0x5000010480 },
      { .start_halfs = 392, .size_halfs = 24, .address = 0x5000020000 },
      { .start_halfs = 416, .size_halfs = 6, .address = 0x4000001098 },
   };
   for (unsigned pattern = 0; pattern < 3; ++pattern) {
      uint64_t halves[AGX3_CPROG_INPUT_WORDS] = {0};
      for (unsigned h = 8; h < 422; ++h) {
         const bool selected = pattern == 0 ? true
                             : pattern == 1 ? (h & 1)
                             : (h == 9 || h >= 10);
         if (selected)
            halves[h / 64] |= UINT64_C(1) << (h % 64);
      }
      struct agx3_cprog_inputs inputs;
      CHECK(agx3_cprog_inputs_init(halves, &inputs));
      const struct agx3_ioto_uniform_program_desc load_desc = {
         .program_va = 0x10000100000, .ranges = ranges,
         .range_count = ARRAY_SIZE(ranges), .patchable = true,
         .cprog_inputs = &inputs,
      };
      size_t bound, chunks;
      CHECK(agx3_ioto_uniform_program_desc_body_bound(chip, &load_desc,
                                                      &bound, &chunks));
      if (pattern == 0)
         CHECK(bound <= AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE);

      uint8_t body[2 * AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load loads[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load sort[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry cache[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
      const struct agx3_ioto_uniform_program_workspace workspace = {
         .body = body, .body_capacity = sizeof(body),
         .loads = loads, .load_capacity = ARRAY_SIZE(loads),
         .sort = sort, .sort_capacity = ARRAY_SIZE(sort),
         .cache = cache, .cache_capacity = ARRAY_SIZE(cache),
      };
      size_t body_size, copy_offset;
      CHECK(agx3_ioto_uniform_program_body_generate(chip, &load_desc, workspace,
                                                    &body_size, &copy_offset));
      CHECK(body_size <= bound);
   }
   return 0;
}

static int
test_load_completion_wait(enum agx_chip chip)
{
   for (unsigned words = 1; words <= 4; ++words) {
      uint8_t body[512];
      struct agx_ioto_inflight_load loads[4], sorted[4];
      struct agx_ioto_load_cache_entry cache[4];
      struct agx_ioto_load_core core;
      CHECK(agx_ioto_load_core_init(&core, chip,
         (struct agx_ioto_load_sink){body, sizeof(body), 0},
         loads, ARRAY_SIZE(loads), cache, ARRAY_SIZE(cache), 0, false, false));
      struct agx_ioto_load_absolute load = {
         .native_slot = 8, .count_dwords = words,
         .address = 0x1234567800ull,
         .lane_mask = (1u << (2 * words)) - 1,
      };
      struct agx_ioto_load_allocation allocation;
      struct agx_ioto_load_workspace workspace = {sorted, ARRAY_SIZE(sorted)};
      CHECK(agx3_ioto_load_absolute_native(&core, &load, workspace, &allocation));
      CHECK(agx3_ioto_load_finish_round_native(&core, true, workspace));
      /* Physical load slot 1 is wait bit zero, independently of length. */
      const struct agx3_field_kv fields[] = {
         {9, 3}, {10, 2}, {23, 2}, {24, 2}, {27, 1}, {28, 1}, {29, 35},
         {3, 16}, {16, 4}, {30, 1},
      };
      uint8_t expected[AGX_IOTO_MAX_RECORD_SIZE], length;
      CHECK(agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_copy, 0,
         fields, ARRAY_SIZE(fields), NULL, 0, expected, &length) ||
            agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_copy, 1,
         fields, ARRAY_SIZE(fields), NULL, 0, expected, &length));
      CHECK(core.copy_suffix_offset + length <= core.sink.length);
      CHECK(!memcmp(body + core.copy_suffix_offset, expected, length));
   }
   return 0;
}

static int
test_unused_vertex_outputs(void)
{
   struct agx_ppp_hal300_graphics_fields fields = {
      .fragment_program = 0x1000,
      .line_width = 15,
      .vertex_varying_components = 9,
      .clip_distance_words = 8,
      .layer_output = true,
   };
   struct agx_ppp_hal300_graphics_core state;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   const uint8_t words22[] = {22, 0, 0, 0};
   CHECK(!memcmp(state.bytes + state.offsets[3] + 16, words22, 4));
   fields.vertex_varying_components = 115;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   const uint8_t words128[] = {128, 0, 0, 0};
   CHECK(!memcmp(state.bytes + state.offsets[3] + 16, words128, 4));
   fields.vertex_varying_components = 116;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   fields.vertex_varying_components = 9;
   fields.fragment_varying_components = 1;
   fields.cf_binding_count = 1;
   fields.cf_register_count = 1;
   fields.varying_table = 0x2000;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   fields.vertex_varying_components = 8;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   return 0;
}

/* Native mesh draws select their provoking vertex in the PPP cull word's
 * flat-shading field ((vertex + 1) << 7); vertex-stage draws keep 0 there
 * and select it through the VDM vertex program template. */
static int
test_flat_shading_vertex(void)
{
   struct agx_ppp_hal300_graphics_fields fields = {
      .fragment_program = 0x1000,
      .line_width = 15,
   };
   struct agx_ppp_hal300_graphics_core state;
   for (unsigned vertex = 0; vertex <= 2; ++vertex) {
      fields.flat_shading_vertex = vertex;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
      uint32_t cull;
      memcpy(&cull, state.bytes + state.offsets[7] + 4, 4);
      CHECK(((cull >> 7) & 3) == vertex + 1);
   }
   fields.flat_shading_vertex = 3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17S, &fields, &state));
   return 0;
}

int main(void) {
   int r = test_unused_vertex_outputs() || test_flat_shading_vertex() || test_load_completion_wait(AGX_CHIP_G17G) ||
           test_load_completion_wait(AGX_CHIP_G17S) ||
           test_load_completion_wait(AGX_CHIP_G17C) ||
           test_multiround_uniforms(AGX_CHIP_G17G) ||
           test_multiround_uniforms(AGX_CHIP_G17S) ||
           test_multiround_uniforms(AGX_CHIP_G17C) ||
           test_cprog_input_bound(AGX_CHIP_G17G) ||
           test_cprog_input_bound(AGX_CHIP_G17S) ||
           test_cprog_input_bound(AGX_CHIP_G17C) ||
           test_color_disabled() || test_visibility() || test_line_width() || test_polygon_and_primitive_msaa() || test_g17g_programs(AGX_CHIP_G17G) || test_g17g_programs(AGX_CHIP_G17S) ||
           test_g17g_programs(AGX_CHIP_G17C) ||
           test_m5_dispatch(AGX_CHIP_G17G) || test_m5_dispatch(AGX_CHIP_G17S) ||
           test_m5_dispatch(AGX_CHIP_G17C) ||
           test_m5_single_kick_launch(AGX_CHIP_G17G) ||
           test_m5_single_kick_launch(AGX_CHIP_G17S) ||
           test_m5_single_kick_launch(AGX_CHIP_G17C);
   if (!r) puts("M5_PROGRAM_DISPATCH_PASS");
   return r;
}
