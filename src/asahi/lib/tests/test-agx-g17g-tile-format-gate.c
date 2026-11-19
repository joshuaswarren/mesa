/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Locks in which colour formats the G17G native end-of-tile store admits.
 *
 * The admitted set is a property of the compiler, not of this table: AGX3C has
 * no generic block-store encoder, and agx3_rt0_eot_kind() substitutes one of
 * Apple's five witnessed applegpu_g17g texts, one per imageblock format class
 * (see compiler/test/probes/README-g17g-native-rt0-eot.md). The driver gate in
 * agx_build_end_of_tile_shader() restates that admission on the block store's
 * physical format, so this test walks every pipe format through the real
 * ail_pixel_format table and checks that exactly the classes with a text pass,
 * that the hardware-verified RGBA8/BGRA8 pair is still inside, and that the
 * two packed-float classes with no witness stay refused.
 */

#include "asahi/layout/layout.h"

#include "agx_bg_eot.h"

#include <inttypes.h>
#include <stdio.h>

static uint64_t checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

/* The imageblock classes the five native texts cover. Growing this list means
 * a sixth text was witnessed; shrinking it means one was withdrawn. Either way
 * it must move together with the g17g_rt0_eot table in agx3_pack_binary.c.
 */
static const enum ail_isa_format admitted_isa[] = {
   AIL_ISA_FORMAT_U8NORM, AIL_ISA_FORMAT_SRGBA8,  AIL_ISA_FORMAT_S8NORM,
   AIL_ISA_FORMAT_RGB10A2, AIL_ISA_FORMAT_I8,     AIL_ISA_FORMAT_I16,
   AIL_ISA_FORMAT_I32,    AIL_ISA_FORMAT_F16,     AIL_ISA_FORMAT_U16NORM,
   AIL_ISA_FORMAT_S16NORM,
};

/* One logical format per admitted class, plus the formats named by the
 * conformance failures this widening is for. The per-entry ISA format is what
 * documents why each one is inside, so a layout-table change that moves a
 * format into another class is caught here rather than on hardware.
 */
struct classified_format {
   enum pipe_format logical;
   enum ail_isa_format physical;
};

static const struct classified_format admitted[] = {
   {PIPE_FORMAT_R8G8B8A8_UNORM, AIL_ISA_FORMAT_U8NORM},
   {PIPE_FORMAT_B8G8R8A8_UNORM, AIL_ISA_FORMAT_U8NORM},
   {PIPE_FORMAT_R8_UNORM, AIL_ISA_FORMAT_U8NORM},
   {PIPE_FORMAT_R8G8B8A8_SRGB, AIL_ISA_FORMAT_SRGBA8},
   {PIPE_FORMAT_R8_SNORM, AIL_ISA_FORMAT_S8NORM},
   {PIPE_FORMAT_R10G10B10A2_UNORM, AIL_ISA_FORMAT_RGB10A2},
   {PIPE_FORMAT_R8_UINT, AIL_ISA_FORMAT_I8},
   {PIPE_FORMAT_R8G8_SINT, AIL_ISA_FORMAT_I8},
   {PIPE_FORMAT_R16_UINT, AIL_ISA_FORMAT_I16},
   {PIPE_FORMAT_R32_FLOAT, AIL_ISA_FORMAT_I32},
   {PIPE_FORMAT_R32G32_UINT, AIL_ISA_FORMAT_I32},
   {PIPE_FORMAT_B5G6R5_UNORM, AIL_ISA_FORMAT_F16},
   {PIPE_FORMAT_R5G6B5_UNORM, AIL_ISA_FORMAT_F16},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, AIL_ISA_FORMAT_F16},
   {PIPE_FORMAT_R16_UNORM, AIL_ISA_FORMAT_U16NORM},
   {PIPE_FORMAT_R16G16_SNORM, AIL_ISA_FORMAT_S16NORM},
};

/* The two formats with hardware-verified passes before the gate was widened.
 * A regression here is worse than any gap the widening closes.
 */
static const enum pipe_format verified[] = {
   PIPE_FORMAT_R8G8B8A8_UNORM,
   PIPE_FORMAT_B8G8R8A8_UNORM,
   PIPE_FORMAT_R8G8B8A8_SRGB,
   PIPE_FORMAT_B8G8R8A8_SRGB,
};

/* The packed-float classes. G17P rewrites R11G11B10 to a raw 32-bit word
 * before the store; G17G keeps the packed physical format, and neither it nor
 * the shared-exponent class has a witnessed text.
 */
static const struct classified_format refused[] = {
   {PIPE_FORMAT_R11G11B10_FLOAT, AIL_ISA_FORMAT_RG11B10F},
   {PIPE_FORMAT_R9G9B9E5_FLOAT, AIL_ISA_FORMAT_RGB9E5},
};

/* Non-renderable formats have no physical format at all. The gate must not
 * read that NONE as anything, and depth/stencil never reaches this path.
 */
static const enum pipe_format non_renderable[] = {
   PIPE_FORMAT_Z16_UNORM, PIPE_FORMAT_Z32_FLOAT, PIPE_FORMAT_S8_UINT,
   PIPE_FORMAT_A8_UNORM,  PIPE_FORMAT_L8_UNORM,
};

static bool
isa_admitted(enum ail_isa_format physical)
{
   for (unsigned i = 0; i < ARRAY_SIZE(admitted_isa); ++i) {
      if (admitted_isa[i] == physical)
         return true;
   }
   return false;
}

int
main(void)
{
   /* The predicate is on the physical format. Over the whole enum it must
    * accept exactly the ISA formats with a native text, and never a
    * non-renderable NONE.
    */
   unsigned physical_admitted = 0;
   for (unsigned f = 0; f < PIPE_FORMAT_COUNT; ++f) {
      const bool admit =
         agx_g17g_rt0_eot_physical_format_supported((enum pipe_format)f);
      CHECK(admit == isa_admitted((enum ail_isa_format)f));
      physical_admitted += admit;
   }
   CHECK(physical_admitted == ARRAY_SIZE(admitted_isa));
   CHECK(!agx_g17g_rt0_eot_physical_format_supported(PIPE_FORMAT_NONE));

   /* Composed with the layout table, every listed logical format reaches the
    * class it is listed with, and that class is admitted.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(admitted); ++i) {
      const enum pipe_format physical =
         ail_pixel_format[admitted[i].logical].renderable;
      CHECK(physical == (enum pipe_format)admitted[i].physical);
      CHECK(agx_g17g_rt0_eot_physical_format_supported(physical));
   }

   for (unsigned i = 0; i < ARRAY_SIZE(verified); ++i) {
      CHECK(agx_g17g_rt0_eot_physical_format_supported(
         ail_pixel_format[verified[i]].renderable));
   }

   for (unsigned i = 0; i < ARRAY_SIZE(refused); ++i) {
      const enum pipe_format physical =
         ail_pixel_format[refused[i].logical].renderable;
      CHECK(physical == (enum pipe_format)refused[i].physical);
      CHECK(!agx_g17g_rt0_eot_physical_format_supported(physical));
   }

   for (unsigned i = 0; i < ARRAY_SIZE(non_renderable); ++i) {
      const enum pipe_format physical =
         ail_pixel_format[non_renderable[i]].renderable;
      CHECK(physical == PIPE_FORMAT_NONE);
      CHECK(!agx_g17g_rt0_eot_physical_format_supported(physical));
   }

   printf("test-agx-g17g-tile-format-gate: %" PRIu64 " checks passed\n",
          checks);
   return 0;
}
