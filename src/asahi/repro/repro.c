/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 *
 * Case table for the Honeykrisp miscompile reproductions. See README.md
 * for what each case isolates and where the observed-on-hardware numbers
 * come from.
 */

#include "vk.h"

/*
 * Case 1 inputs: 200 words, every byte nonzero so a zeroed readback can
 * only be a failed load, never valid data.
 */
static uint32_t case1_in[200];

/* Case 3/4/5 packed-bool inputs: 33 bools, all true. 4 per word. */
static uint32_t packed33[9];
static uint32_t packed64[16];

static void
fill_inputs(void)
{
   for (unsigned i = 0; i < 200; i++)
      case1_in[i] = (i + 1u) * 0x01010101u;

   for (unsigned i = 0; i < 8; i++)
      packed33[i] = 0x01010101u;
   packed33[8] = 0x00000001u; /* element 32 */

   for (unsigned i = 0; i < 16; i++)
      packed64[i] = 0x01010101u;
}

/* ---- case 1: word loads in a multi-iteration grid-stride loop ------ */

static bool
check1(const struct hk_case *c, const uint32_t *out[HK_MAX_SSBOS])
{
   printf("  observed on M1: out[w] == 0 unless (w & 3) == 0\n");
   return hk_check_words("copy", out[1], c->data[0], 200);
}

/* ---- case 2: scan fed by a dynamic shift --------------------------- */

static bool
check2(const struct hk_case *c, const uint32_t *out[HK_MAX_SSBOS])
{
   (void)c;
   uint32_t want[256];
   for (unsigned i = 0; i < 256; i++)
      want[i] = i / 2u + 1u;
   printf("  observed on M1: out[255] == 32 (window-64 scan)\n");
   return hk_check_words("scan", out[0], want, 256);
}

/* ---- case 3: four-arm selector matrix ------------------------------- */

static bool
check_arm(const struct hk_case *c, const uint32_t *out[HK_MAX_SSBOS])
{
   (void)c;
   uint32_t want[33];
   for (unsigned i = 0; i < 33; i++)
      want[i] = 1u;

   printf("  receipt evidence: combined selector shader red,\n"
          "  selector-free control green (suite-level, confounded with\n"
          "  an init fix); per-byte wrong values were never recorded\n");
   return hk_check_words("bool op", out[2], want, 33);
}

/* ---- case 4: packed bool logical AND ------------------------------- */

static bool
check4(const struct hk_case *c, const uint32_t *out[HK_MAX_SSBOS])
{
   (void)c;
   uint32_t want[33];
   for (unsigned i = 0; i < 33; i++)
      want[i] = 1u;
   printf("  observed on M1: 13 of 33 bytes wrong (LogicalAnd case),\n"
          "  15 of 33 against a scalar true\n");
   return hk_check_words("and", out[2], want, 33);
}

/* ---- case 5: reduce over load_truthy ------------------------------- */

static bool
check_reduce(const struct hk_case *c, const uint32_t *out[HK_MAX_SSBOS])
{
   uint32_t n = c->push[0];
   uint32_t words = (n + 3u) / 4u;
   bool ok = true;

   printf("  observed on M1: reduce == 0 for all-true input at n >= 5;\n"
          "  33-element map: truthy reads correct only at positions 0-3\n"
          "  and 16-19, falsy elsewhere\n");

   if (out[1][0] != 1u) {
      printf("  reduce[0]: got %u, want 1\n", out[1][0]);
      ok = false;
   } else {
      printf("  reduce[0] == 1\n");
   }

   unsigned bad = 0;
   for (unsigned i = 0; i < n; i++) {
      if (out[2][i] != 1u && bad < 8) {
         printf("  truthy[%u]: got %u, want 1\n", i, out[2][i]);
         bad++;
      } else if (out[2][i] != 1u) {
         bad++;
      }
   }
   if (bad) {
      printf("  FAIL: %u/%u truthy reads wrong\n", bad, (unsigned)n);
      ok = false;
   } else {
      printf("  PASS: truthy reads match (%u elements)\n", (unsigned)n);
   }

   ok &= hk_check_words("raw words", out[3], c->data[0], words);
   return ok;
}

/* ---- case table ----------------------------------------------------- */

const struct hk_case hk_cases[] = {
   {1, "01-word-load-divergent-loop", "01-word-load-divergent-loop.spv",
    1, {0, 0, 0, 0},
    {200, 200, 0, 0}, {case1_in, NULL, NULL, NULL}, check1},
   {2, "02-scan-dynamic-shift", "02-scan-dynamic-shift.spv",
    1, {0, 0, 0, 0},
    {256, 0, 0, 0}, {NULL, NULL, NULL, NULL}, check2},

   /* Case 3 matrix. push = {const_shift, op, n, pad}. op 0 is the
    * checked bool comparison. */
   {3, "03-A-selector-dynshift", "03-selector-bytepath.spv",
    1, {0, 0, 33, 0},
    {9, 9, 33, 0}, {packed33, packed33, NULL, NULL}, check_arm},
   {3, "03-B-noselector-dynshift", "03-noselector-bytepath.spv",
    1, {0, 0, 33, 0},
    {9, 9, 33, 0}, {packed33, packed33, NULL, NULL}, check_arm},
   {3, "03-C-selector-constshift", "03-selector-bytepath.spv",
    1, {1, 0, 33, 0},
    {9, 9, 33, 0}, {packed33, packed33, NULL, NULL}, check_arm},
   {3, "03-D-noselector-constshift", "03-noselector-bytepath.spv",
    1, {1, 0, 33, 0},
    {9, 9, 33, 0}, {packed33, packed33, NULL, NULL}, check_arm},

   {4, "04-packed-bool-and", "04-packed-bool-and.spv",
    1, {0, 0, 0, 0},
    {9, 9, 33, 0}, {packed33, packed33, NULL, NULL}, check4},
   {5, "05-reduce-load-truthy-n64", "05-reduce-load-truthy.spv",
    1, {64, 0, 0, 0},
    {16, 1, 64, 16}, {packed64, NULL, NULL, NULL}, check_reduce},
   {50, "05-reduce-load-truthy-n33", "05-reduce-load-truthy.spv",
    1, {33, 0, 0, 0},
    {9, 1, 64, 9}, {packed33, NULL, NULL, NULL}, check_reduce},
};

const int hk_case_count = sizeof(hk_cases) / sizeof(hk_cases[0]);

int
main(int argc, char **argv)
{
   fill_inputs();

   int only = -1;
   if (argc > 1)
      only = atoi(argv[1]);

   return hk_run(only);
}
