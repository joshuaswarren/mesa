/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * G18G graphics program templates against full compositions. A template is
 * the agx3d form of a patchable program (agx3_g18g_graphics_program_template_
 * generate()) for one range shape; for other addresses the plan fits, it must
 * give exactly the bytes a fresh patchable composition re-encoded in full
 * has, or refuse when an agx3d LDIMM takes another size. The full composition
 * is at another program address than the template, as in the driver, which
 * makes templates at one address and emits them at the pool's. Vertex,
 * fragment, coverage-fragment and TPS-frame programs.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/compiler/agx3_reencode_cache.h"
#include "asahi/lib/agx_ioto_state_core.h"
#include "asahi/vulkan/hk_g17p_patch.h"
#include "util/ralloc.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #x);                                                          \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static uint64_t
rnd(void)
{
   rng ^= rng << 13;
   rng ^= rng >> 7;
   rng ^= rng << 17;
   return rng;
}

#define CLASSES 5

/* A shader's promoted constants: in the USC window (a high half of 0x100), at
 * the same address for every draw */
#define RODATA UINT64_C(0x10000000400)

/* The driver makes templates at one program address and emits them at
 * another */
#define TEMPLATE_VA UINT64_C(0x10000010000)
#define PROGRAM_VA UINT64_C(0x10000234000)

static bool
generate_full(struct agx3_g17p_graphics_program_desc *desc,
              struct agx3_g17p_graphics_program *program)
{
   const uint64_t va = desc->program_va;
   desc->program_va = PROGRAM_VA;
   const bool ok = desc->scratch_size
      ? agx3_g18g_graphics_program_generate_tps(desc, program)
      : agx3_g18g_graphics_program_generate(desc, program);
   desc->program_va = va;
   return ok;
}

/* The address of uniform range i in the k-th draw: a batch's running upload
 * buffer, 64 bytes a draw; short or zero 32-bit halves, for which agx3d
 * LDIMMs take other sizes; and random 40-bit addresses. Range 1 of three or
 * more is the shader's constants. */
static uint64_t
address(unsigned class, unsigned k, unsigned i, unsigned range_count)
{
   if (i == 1 && range_count >= 3)
      return RODATA;
   switch (class) {
   case 0:
      return UINT64_C(0x4000001000) + 0x1000 * i + 64 * k;
   case 1:
      return rnd() & 0xfff0;
   case 2:
      return (rnd() & 0xff) << 32 | (rnd() & 0xff0);
   case 3:
      return (rnd() & 0xff) << 32;
   default:
      return rnd() & 0xfffffffff0;
   }
}

int
main(void)
{
   void *mem = ralloc_context(NULL);
   unsigned made = 0, emitted[CLASSES] = {0}, refused[CLASSES] = {0},
            unfit = 0;

   /* Vertex, fragment, coverage fragment, then both again with a TPS frame */
   for (unsigned variant = 0; variant < 6; ++variant) {
      const unsigned fragment = (variant % 3) != 0;
      const bool coverage = (variant % 3) == 2;
      const unsigned scratch = variant >= 3 ? 528 : 0;
      for (unsigned range_count = 1; range_count <= 6; ++range_count) {
         struct agx3_ioto_uniform_range ranges[6];
         struct agx3_g17p_graphics_program_desc desc = {
            .program_va = TEMPLATE_VA,
            .shader_va =
               fragment ? UINT64_C(0x100000003c0) : UINT64_C(0x10000000480),
            .shader_binary_size = fragment ? 60 : 128,
            .scratch_size = scratch,
            .nr_gprs = 16,
            .compiler_d_valid = true,
            .compiler_d = fragment ? 0 : 32,
            .fragment = fragment,
            .writes_sample_mask = coverage,
            .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
            .uniform_ranges = ranges,
            .uniform_range_count = range_count,
            .patchable = true,
         };
         const struct agx3_transcode_ioto ioto = {
            .api_program = true,
            .shader_gprs = desc.nr_gprs,
         };

         /* Mixed range sizes, the last one long enough to load in chunks */
         for (unsigned i = 0; i < range_count; ++i) {
            ranges[i].start_halfs = 8 * i;
            ranges[i].size_halfs = i + 1 == range_count ? 64 : 4 + 2 * (i & 1);
            ranges[i].address = address(0, 0, i, range_count);
         }

         /* The template, from the first draw's addresses */
         struct agx3_g17p_graphics_program_template composed;
         struct hk_g17p_patch_plan plan;
         unsigned offsets[2 * HK_G17P_PATCH_MAX_RELOCS];
         const char *why = NULL;
         unsigned at = 0;
         CHECK(agx3_g18g_graphics_program_template_generate(&desc, &composed));
         CHECK(hk_g17p_patch_plan_init(&plan, ranges, range_count,
                                       composed.relocations,
                                       composed.relocation_count,
                                       composed.program.length, true));
         const unsigned count =
            hk_g17p_patch_plan_ldimms(&plan, NULL, offsets, NULL);
         struct agx3_reencode_template *t = agx3_reencode_template_create(
            mem, &composed.program, &ioto, offsets, count, &why, &at);
         CHECK(t);
         ++made;

         for (unsigned k = 0; k < 100 * CLASSES; ++k) {
            const unsigned class = k / 100;
            for (unsigned i = 0; i < range_count; ++i)
               ranges[i].address = address(class, k % 100, i, range_count);
            if (!hk_g17p_patch_plan_fits(&plan, ranges)) {
               ++unfit;
               continue;
            }

            struct agx3_g17p_graphics_program full, patched;
            uint32_t immediates[2 * HK_G17P_PATCH_MAX_RELOCS];
            CHECK(generate_full(&desc, &full));
            CHECK(agx3_reencode_program(&full, &ioto, &why, &at));
            hk_g17p_patch_plan_ldimms(&plan, ranges, NULL, immediates);
            if (!agx3_reencode_template_emit(t, immediates, &patched)) {
               /* Only for addresses whose LDIMMs take another size */
               ++refused[class];
               CHECK(class != 0);
               continue;
            }
            CHECK(patched.length == full.length);
            CHECK(!memcmp(patched.bytes, full.bytes, full.length));
            ++emitted[class];
         }
      }
   }

   /* Another high half for the constants (a shader elsewhere in the USC
    * window) patches like any other LDIMM */
   {
      struct agx3_ioto_uniform_range ranges[3] = {
         {.start_halfs = 0, .size_halfs = 4, .address = address(0, 0, 0, 3)},
         {.start_halfs = 8, .size_halfs = 4, .address = RODATA},
         {.start_halfs = 16, .size_halfs = 4, .address = address(0, 0, 2, 3)},
      };
      struct agx3_g17p_graphics_program_desc desc = {
         .program_va = TEMPLATE_VA,
         .shader_va = UINT64_C(0x10000000480),
         .shader_binary_size = 128,
         .nr_gprs = 16,
         .compiler_d_valid = true,
         .compiler_d = 32,
         .uniform_ranges = ranges,
         .uniform_range_count = 3,
         .patchable = true,
      };
      const struct agx3_transcode_ioto ioto = {.api_program = true,
                                               .shader_gprs = 16};
      struct agx3_g17p_graphics_program_template composed;
      struct agx3_g17p_graphics_program patched, full;
      struct hk_g17p_patch_plan plan;
      unsigned offsets[2 * HK_G17P_PATCH_MAX_RELOCS];
      uint32_t immediates[2 * HK_G17P_PATCH_MAX_RELOCS];
      const char *why = NULL;
      unsigned at = 0;
      CHECK(agx3_g18g_graphics_program_template_generate(&desc, &composed));
      CHECK(hk_g17p_patch_plan_init(&plan, ranges, 3, composed.relocations,
                                    composed.relocation_count,
                                    composed.program.length, true));
      const unsigned count =
         hk_g17p_patch_plan_ldimms(&plan, NULL, offsets, NULL);
      struct agx3_reencode_template *t = agx3_reencode_template_create(
         mem, &composed.program, &ioto, offsets, count, &why, &at);
      CHECK(t);
      ranges[1].address = UINT64_C(0x20000000400);
      CHECK(hk_g17p_patch_plan_fits(&plan, ranges));
      hk_g17p_patch_plan_ldimms(&plan, ranges, NULL, immediates);
      CHECK(agx3_reencode_template_emit(t, immediates, &patched));
      CHECK(generate_full(&desc, &full));
      CHECK(agx3_reencode_program(&full, &ioto, &why, &at));
      CHECK(patched.length == full.length &&
            !memcmp(patched.bytes, full.bytes, full.length));
   }

   for (unsigned c = 0; c < CLASSES; ++c)
      printf("class %u: %u emitted, %u refused\n", c, emitted[c], refused[c]);
   printf("%u templates, %u address sets the plans do not fit\n", made, unfit);
   /* Every running-buffer draw comes from its template */
   CHECK(made == 36 && emitted[0] == 100 * made);
   ralloc_free(mem);
   return 0;
}
