/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* The root-only draw flush and the vertex-state bind-tail result, built with
 * NDEBUG as a release driver is: neither may depend on an assertion. */
#ifndef NDEBUG
#error "this test checks release-build behaviour and must be built with NDEBUG"
#endif

#include <stdio.h>

#include "asahi/lib/agx_vdm_state_update_core.h"
#include "hk_cmd_buffer.h"

static unsigned failures = 0;

#define CHECK(cond, fmt, ...)                                                  \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__,                    \
                 __LINE__, ##__VA_ARGS__);                                     \
         failures++;                                                           \
      }                                                                        \
   } while (0)

/* The root-only flush is chosen from the stream's latched HAL300 graphics core
 * and the vertex-state capabilities, never from the chip identity, and a
 * stream without a latched core always takes hk_flush_dynamic_state_slow(). */
static void
test_root_only_flush_selection(void)
{
   const struct hk_cs latched = {.m5_ppp_emitted_valid = true};

   for (unsigned c = 0; c <= AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = c;
      const char *name = agx_chip_info(chip)->name;
      struct hk_cs cs = {0};

      CHECK(hk_agx3_vertex_state(chip) == agx_vdm_state_update_supported(chip),
            "%s: vertex-state predicate disagrees with the encoder", name);

      cs.m5_ppp_emitted_valid = false;
      CHECK(!hk_root_only_flush_supported(chip, &cs),
            "%s: root-only flush without a latched graphics core", name);

      cs.m5_ppp_emitted_valid = true;
      CHECK(hk_root_only_flush_supported(chip, &cs) ==
               (hk_agx3_vertex_state(chip) &&
                !hk_vertex_state_has_bind_tail(chip)),
            "%s: root-only flush not selected by its capabilities", name);
   }

   /* A chip whose vertex-state publication appends a bind tail takes the slow
    * path even with a latched graphics core, so the tail is never dropped. */
   const enum agx_chip tailed[] = {AGX_CHIP_G15G, AGX_CHIP_G16G, AGX_CHIP_G17P};
   for (unsigned i = 0; i < ARRAY_SIZE(tailed); ++i) {
      CHECK(!hk_root_only_flush_supported(tailed[i], &latched),
            "%s: chip with a bind tail takes the root-only flush",
            agx_chip_info(tailed[i])->name);
   }

   /* Every chip that packs the HAL300 graphics core takes it; AGX2 cannot. */
   const enum agx_chip hal300[] = {AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
                                   AGX_CHIP_G17D, AGX_CHIP_G18G};
   for (unsigned i = 0; i < ARRAY_SIZE(hal300); ++i) {
      CHECK(hk_root_only_flush_supported(hal300[i], &latched),
            "%s: HAL300 chip misses the root-only flush",
            agx_chip_info(hal300[i])->name);
   }
   CHECK(!hk_root_only_flush_supported(AGX_CHIP_G13G, &latched),
         "G13G has no AGX3 VDM State Update");
   CHECK(!hk_root_only_flush_supported(AGX_CHIP_UNKNOWN, &latched),
         "unknown chip takes the root-only flush");
}

/* A flush reports the bind tail its vertex-state publication emitted on the
 * native AGX3 path, and none elsewhere, in a release build as in a debug
 * one. */
static void
test_vertex_state_bind_tail(void)
{
   for (unsigned c = 0; c <= AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = c;
      const char *name = agx_chip_info(chip)->name;

      CHECK(!hk_vertex_state_bind_tail(chip, false),
            "%s: reports a tail it did not emit", name);
      CHECK(hk_vertex_state_has_bind_tail(chip) == agx_chip_native_agx3(chip),
            "%s: bind tail not appended by the native AGX3 path", name);
      CHECK(hk_vertex_state_bind_tail(chip, true) ==
               hk_vertex_state_has_bind_tail(chip),
            "%s: drops or invents an emitted bind tail", name);
   }

   const enum agx_chip native[] = {AGX_CHIP_G15G, AGX_CHIP_G16G, AGX_CHIP_G17P};
   for (unsigned i = 0; i < ARRAY_SIZE(native); ++i) {
      CHECK(hk_vertex_state_bind_tail(native[i], true),
            "%s: emitted native bind tail is not reported",
            agx_chip_info(native[i])->name);
   }
   CHECK(!hk_vertex_state_bind_tail(AGX_CHIP_G17S, true),
         "G17S has no native bind tail");
}

int
main(void)
{
   test_root_only_flush_selection();
   test_vertex_state_bind_tail();

   if (failures) {
      fprintf(stderr, "%u failure(s)\n", failures);
      return 1;
   }
   printf("hk root-only flush: PASS\n");
   return 0;
}
