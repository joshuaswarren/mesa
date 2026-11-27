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
   (0 CAPS(CAP_BIT)) | AGX_CAP_SPARSE_RESIDENCY_3D | AGX_CAP_DESCRIPTOR_BUFFER |
   AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS | AGX_CAP_CDM_WAR_WEAK_BARRIER |
   AGX_CAP_GEN9_2D_COMPRESSION | AGX_CAP_MOD3_REG_BANK_RA |
   AGX_CAP_GEN10_2D_COMPRESSION | AGX_CAP_SPARSE_BINDING |
   AGX_CAP_ZLS_FULL_RENDER_BOUNDS | AGX_CAP_NATIVE_MIXED_EPILOG |
   AGX_CAP_HAL200_LOSSLESS_COMPRESSION | AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION |
   AGX_CAP_SIGNED_ZERO_MINMAX_SELECT | AGX_CAP_CDM_OVERLAP_SPLIT |
   AGX_CAP_CACHED_CMD_MEMORY | AGX_CAP_COMPUTE_SKIP_PRESSURE_SCHED |
   AGX_CAP_FOLD_SHARED_OFFSETS | AGX_CAP_CDM_DEFER_FLUSH |
   AGX_CAP_PBE_QUEUED_BUFFER_COPY;
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

/* AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS is enabled on G17P and M5 (G17G,
 * G17S, G17C); G15, G16, G18G, and AGX2 stay off until per-generation
 * hardware A/B benchmarks and CTS rows qualify them. */
static void
test_coalesce_pure_global_workgroups(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G17P || c == AGX_CHIP_G17G ||
                            c == AGX_CHIP_G17S || c == AGX_CHIP_G17C;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS) ==
                    expected,
                 c);
   }
}

/* AGX_CAP_CDM_WAR_WEAK_BARRIER is enabled on G17P and M5 only. G15 and G16G
 * elide application dispatch barriers too but keep the flush at execution-only
 * and WAR barriers until hardware rows qualify them. */
static void
test_cdm_war_weak_barrier(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G17P || c == AGX_CHIP_G17G ||
                            c == AGX_CHIP_G17S || c == AGX_CHIP_G17C;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_CDM_WAR_WEAK_BARRIER) == expected,
                 c);
   }
}

/* AGX_CAP_MOD3_REG_BANK_RA is qualified on G17P and M5 (G17G/G17S/G17C);
 * G15, G16, G18G, and AGX2 stay off until per-generation hardware A/B
 * benchmarks and CTS rows qualify them. */
static void
test_mod3_reg_bank_ra(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G15G || c == AGX_CHIP_G15S ||
                            c == AGX_CHIP_G15C || c == AGX_CHIP_G16G ||
                            c == AGX_CHIP_G17P || c == AGX_CHIP_G17G ||
                            c == AGX_CHIP_G17S || c == AGX_CHIP_G17C;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_MOD3_REG_BANK_RA) == expected, c);
   }
}

/* G17D partitions AGX_CAPS_M5 into measured and excluded capabilities so any
 * new M5 capability fails this test until assigned to one set. */
static void
test_g17d_partitions_m5(void)
{
   CHECK((AGX_CAPS_G17D_MEASURED & AGX_CAPS_G17D_EXCLUDED) == 0);
   CHECK((AGX_CAPS_G17D_MEASURED | AGX_CAPS_G17D_EXCLUDED) == AGX_CAPS_M5);
   for (unsigned bit = 1; bit; bit <<= 1) {
      if (!(all_caps & bit))
         continue;
      CHECK_CHIP(agx_chip_has_cap(AGX_CHIP_G17D, bit) ==
                    ((AGX_CAPS_G17D_MEASURED & bit) != 0),
                 AGX_CHIP_G17D);
   }
}

/* AGX_CAP_GEN10_2D_COMPRESSION is enabled on the M5 parts only; G18G, the
 * other Gen10 descriptor chip, keeps it off until measured. */
static void
test_gen10_2d_compression(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G17G || c == AGX_CHIP_G17S ||
                            c == AGX_CHIP_G17C || c == AGX_CHIP_G17D;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_GEN10_2D_COMPRESSION) == expected,
                 c);
   }
}

/* AGX_CAP_SPARSE_BINDING is enabled on AGX2, G17P, and M5. G15, G16, and
 * G18G keep it off until sparse residency and SSTT flushing are qualified. */
static void
test_sparse_binding(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected =
         c == AGX_CHIP_G13G || c == AGX_CHIP_G13X || c == AGX_CHIP_G14G ||
         c == AGX_CHIP_G14X || c == AGX_CHIP_G17P || c == AGX_CHIP_G17G ||
         c == AGX_CHIP_G17S || c == AGX_CHIP_G17C || c == AGX_CHIP_G17D;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_SPARSE_BINDING) == expected, c);
   }
}

/* AGX_CAP_ZLS_FULL_RENDER_BOUNDS is carried by the G18G render-submit ABI
 * that validates packed ZLS dimensions against the render command bounds. */
static void
test_zls_full_render_bounds(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_ZLS_FULL_RENDER_BOUNDS) ==
                    (c == AGX_CHIP_G18G),
                 c);
   }
}

/* AGX_CAP_NATIVE_MIXED_EPILOG is set where the native1130 single-sample
 * fragment epilog encoding has been verified (G18G). */
static void
test_native_mixed_epilog(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_NATIVE_MIXED_EPILOG) ==
                    (c == AGX_CHIP_G18G),
                 c);
   }
}

/* AGX_CAP_HAL200_LOSSLESS_COMPRESSION is enabled across native HAL200 AGX3B
 * parts (G16G and G17P) and off on G15* (M3), M5 HAL300 (Gen10), G18G, AGX2,
 * and unadmitted parts. */
static void
test_hal200_lossless_compression(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G16G || c == AGX_CHIP_G17P;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_HAL200_LOSSLESS_COMPRESSION) ==
                    expected,
                 c);
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_HAL200_LOSSLESS_COMPRESSION) ==
                    agx_chip_native_hal200(c),
                 c);
   }
}

/* AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION is enabled across native HAL200 AGX3B
 * parts (G16G and G17P) and off on every chip without the base HAL200 cap. */
static void
test_hal200_msaa_color_compression(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G16G || c == AGX_CHIP_G17P;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION) ==
                    expected,
                 c);
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION) ==
                    agx_chip_native_hal200(c),
                 c);
      /* MSAA colour never enables without the base HAL200 cap. */
      if (agx_chip_has_cap(c, AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION))
         CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_HAL200_LOSSLESS_COMPRESSION), c);
   }
}

/* AGX_CAP_SIGNED_ZERO_MINMAX_SELECT is enabled on G17P and the M5 parts
 * only. */
static void
test_signed_zero_minmax_select(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G15G || c == AGX_CHIP_G15S ||
                            c == AGX_CHIP_G15C || c == AGX_CHIP_G16G ||
                            c == AGX_CHIP_G17P || c == AGX_CHIP_G17G ||
                            c == AGX_CHIP_G17S || c == AGX_CHIP_G17C;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_SIGNED_ZERO_MINMAX_SELECT) ==
                    expected,
                 c);
   }
}

/* AGX_CAP_COMPUTE_SKIP_PRESSURE_SCHED and AGX_CAP_FOLD_SHARED_OFFSETS are
 * enabled on the measured G13 rows (G13G, G13X) only. */
static void
test_g13_compute_codegen_caps(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G13G || c == AGX_CHIP_G13X;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_COMPUTE_SKIP_PRESSURE_SCHED) ==
                    expected,
                 c);
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_FOLD_SHARED_OFFSETS) == expected,
                 c);
   }
}

/* AGX_CAP_CDM_DEFER_FLUSH is enabled on the measured G13 rows (G13G, G13X). */
static void
test_cdm_defer_flush(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      const bool expected = c == AGX_CHIP_G13G || c == AGX_CHIP_G13X;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_CDM_DEFER_FLUSH) == expected, c);
   }
}

/* AGX_CAP_PBE_QUEUED_BUFFER_COPY is enabled only on G17C where queued >= 32 MiB
 * PBE buffer copies have been hardware-qualified; G17P serializes queued PBE
 * copies, and other M5 SKUs stay off until measured on hardware. */
static void
test_pbe_queued_buffer_copy(void)
{
   for (unsigned i = 0; i <= AGX_CHIP_UNKNOWN; ++i) {
      const enum agx_chip c = (enum agx_chip)i;
      CHECK_CHIP(agx_chip_has_cap(c, AGX_CAP_PBE_QUEUED_BUFFER_COPY) ==
                    (c == AGX_CHIP_G17C),
                 c);
   }
}

int
main(void)
{
   test_table_covers_every_chip();
   test_caps_match_predicates();
   test_unknown_has_none();
   test_coalesce_pure_global_workgroups();
   test_cdm_war_weak_barrier();
   test_mod3_reg_bank_ra();
   test_g17d_partitions_m5();
   test_gen10_2d_compression();
   test_sparse_binding();
   test_zls_full_render_bounds();
   test_native_mixed_epilog();
   test_hal200_lossless_compression();
   test_hal200_msaa_color_compression();
   test_signed_zero_minmax_select();
   test_g13_compute_codegen_caps();
   test_cdm_defer_flush();
   test_pbe_queued_buffer_copy();

   if (failures) {
      fprintf(stderr, "agx_chip_caps: %u of %u checks failed\n", failures,
              checks);
      return 1;
   }

   printf("agx_chip_caps: all %u checks passed\n", checks);
   return 0;
}
