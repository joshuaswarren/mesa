/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_chip_caps.h"

#include <stdio.h>

static unsigned failures;
static unsigned checks;

#define CHECK(condition)                                                       \
   do {                                                                        \
      checks++;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

#define CHECK_CHIP(condition, chip)                                            \
   do {                                                                        \
      checks++;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,                \
                 agx_chip_name(chip), #condition);                             \
         failures++;                                                           \
      }                                                                        \
   } while (0)

/* Every capability, with the agx_chip.h predicate it was introduced from. */
#define CAPS(X)                                                                \
   X(AGX_CAP_NATIVE_INDIRECT_COMPUTE, agx_chip_native_indirect_compute)        \
   X(AGX_CAP_NATIVE_IMAGEBLOCK, agx_chip_native_imageblock)                    \
   X(AGX_CAP_NATIVE_PREAMBLE, agx_chip_native_preamble)                        \
   X(AGX_CAP_ELIDE_APP_DISPATCH_BARRIER, agx_chip_elide_app_dispatch_barrier)  \
   X(AGX_CAP_VDM_INDIRECT_INDEX_BOUNDS, agx_chip_vdm_indirect_index_bounds)    \
   X(AGX_CAP_HW_TXF_BOUNDS, agx_chip_hw_texel_fetch_bounds)                    \
   X(AGX_CAP_TRI_MERGE_ACROSS_SIDE_EFFECTS,                                    \
     agx_chip_tri_merge_across_side_effects)                                   \
   X(AGX_CAP_COOPERATIVE_MATRIX, agx_chip_cooperative_matrix_supported)

#define CAP_BIT(cap, predicate) | (cap)
static const unsigned all_caps =
   (0 CAPS(CAP_BIT)) | AGX_CAP_SPARSE_RESIDENCY_3D;
#undef CAP_BIT

/* Each chip of AGX_CHIPS has exactly one row, and the rows are in its
 * order. */
#define ROW_COUNT(row, caps, chip) +((chip) == AGX_CHIP_##row)
#define ROW_CHIP(row, caps, ...)   AGX_CHIP_##row,
#define ROW_CAPS(row, caps, ...)   (caps),

static void
test_table_covers_every_chip(void)
{
   static const enum agx_chip rows[] = {AGX_CHIP_CAPS(ROW_CHIP)};
   static const unsigned row_caps[] = {AGX_CHIP_CAPS(ROW_CAPS)};

   CHECK(sizeof(rows) / sizeof(rows[0]) == AGX_CHIP_UNKNOWN);

   for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
      CHECK(rows[i] == (enum agx_chip)i);
      /* No row sets a bit outside enum agx_chip_cap. */
      CHECK((row_caps[i] & ~all_caps) == 0);
   }

   for (unsigned i = 0; i < AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      CHECK_CHIP((0 AGX_CHIP_CAPS(ROW_COUNT, c)) == 1, c);
   }
}

/* Every capability answers what its predicate answers, on every chip. */
static void
test_caps_match_predicates(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;

#define CAP_MATCHES(cap, predicate)                                            \
   CHECK_CHIP(agx_chip_has_cap(c, cap) == (bool)predicate(c), c);
      CAPS(CAP_MATCHES)
#undef CAP_MATCHES
   }
}

/* Unknown chips and values outside the enum have no capability. */
static void
test_unknown_has_none(void)
{
   const int outside[] = {-1, AGX_CHIP_UNKNOWN, AGX_CHIP_UNKNOWN + 1, 255};

   for (unsigned i = 0; i < sizeof(outside) / sizeof(outside[0]); ++i) {
      for (unsigned bit = 0; bit < 32; ++bit)
         CHECK(!agx_chip_has_cap((enum agx_chip)outside[i], 1u << bit));
   }

   /* Asking for no capability is never true. */
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i)
      CHECK(!agx_chip_has_cap((enum agx_chip)i, 0));
}

int
main(void)
{
   test_table_covers_every_chip();
   test_caps_match_predicates();
   test_unknown_has_none();

   if (failures) {
      fprintf(stderr, "agx_chip_caps: %u of %u checks failed\n", failures,
              checks);
      return 1;
   }

   printf("agx_chip_caps: all %u checks passed\n", checks);
   return 0;
}
