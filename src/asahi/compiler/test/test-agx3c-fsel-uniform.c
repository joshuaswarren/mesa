/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * M5 (AGX3C) 32-bit float compare-selects that read the uniform file in
 * place. Every encoding below is one the M5 Metal compiler emitted in native
 * fragment and compute programs; agx3_alu_fcmp_select32_gpu1() must rebuild
 * it byte for byte from the operands. Uniform shapes the native compiler was
 * not seen to emit stay refused, so those uniforms are still copied.
 */
#include "agx3_alu_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define C AGX3_ALU_HINT_CACHE
#define D AGX3_ALU_HINT_DISCARD
#define N AGX3_ALU_HINT_NONE

enum kind { R, I, U, U16 };

struct operand {
   enum kind kind;
   uint16_t value;
   enum agx3_alu_hint hint;
};

struct witness {
   const char *name;
   uint16_t dest;
   bool dest_cache;
   struct operand src[4]; /* cmp0, cmp1, sel0 (taken when true), sel1 */
   bool sel_integer;
   unsigned condition;
   const char *bytes;
};

static const struct witness witnesses[] = {
   /* r ? u : r, r: the uniform second comparand, 32- and 16-bit */
   {"r34 = r36 == u74 ? r38 : r34", 34, true,
    {{R, 36, C}, {U, 74, N}, {R, 38, D}, {R, 34, D}}, true, 0,
    "12a5674b85260002"},
   {"r30 = r54 == u74 ? r52 : r32", 30, true,
    {{R, 54, C}, {U, 74, N}, {R, 52, D}, {R, 32, D}}, true, 0,
    "f2b7274b863400028020"},
   {"r86 = r28 < u236 ? r86 : r24", 86, false,
    {{R, 28, C}, {U, 236, N}, {R, 86, D}, {R, 24, C}}, true, 1,
    "b29d976d00980302"},
   {"r32 = r4 >= u42h ? r32 : r4", 32, true,
    {{R, 4, C}, {U16, 42, N}, {R, 32, D}, {R, 4, C}}, true, 5,
    "0285672a04840202"},
   {"r108 = r28 < u236 ? r108 : 3", 108, false,
    {{R, 28, D}, {U, 236, N}, {R, 108, D}, {I, 3, N}}, true, 1,
    "621ddf6d20830302"},
   /* a uniform selected operand */
   {"r2 = r0 < r46 ? u34 : r26", 2, true,
    {{R, 0, C}, {R, 46, C}, {U, 34, N}, {R, 26, C}}, true, 1,
    "128127af22220300009a"},
   {"r24 = r30 < 0.0 ? u276 : r44", 24, true,
    {{R, 30, C}, {I, 0, N}, {U, 276, N}, {R, 44, D}}, true, 1,
    "c29f278162140302802c"},
   {"r52 = r40 < r54 ? u134 : 2", 52, true,
    {{R, 40, N}, {R, 54, N}, {U, 134, N}, {I, 2, N}}, true, 1,
    "a2296737a20603002082"},
   {"r2 = r2 < r48 ? 1 : u34", 2, true,
    {{R, 2, D}, {R, 48, D}, {I, 1, N}, {U, 34, N}}, true, 1,
    "12033f31228103002022"},
   {"r2 = r2 <= r4 ? r2 : u18", 2, true,
    {{R, 2, D}, {R, 4, D}, {R, 2, D}, {U, 18, N}}, true, 6,
    "12033f0524120300"},
   /* two uniforms */
   {"r80 = r72 > 0.0 ? u136 : u138", 80, true,
    {{R, 72, C}, {I, 0, N}, {U, 136, N}, {U, 138, N}}, true, 2,
    "82c9a781a2080202a00a"},
   {"r24 = r20 < u62 ? u62 : r20", 24, true,
    {{R, 20, C}, {U, 62, N}, {U, 62, N}, {R, 20, C}}, true, 1,
    "c295273f223e03020094"},
   {"r26 = r20 < u62 ? r20 : u62", 26, true,
    {{R, 20, C}, {U, 62, N}, {R, 20, C}, {U, 62, N}}, true, 1,
    "d295263f0302"},
   /* tied min/max against a uniform (conditions 3 and 7) */
   {"r4 = min(r4, u56)", 4, true,
    {{R, 4, D}, {U, 56, N}, {R, 4, D}, {U, 56, N}}, true, 3,
    "22052e390102"},
   {"r16 = max(r14, u50)", 16, true,
    {{R, 14, D}, {U, 50, N}, {R, 14, D}, {U, 50, N}}, true, 7,
    "820f2e330002"},
   /* u ? #0 : r, r: a 16-bit uniform first comparand against zero */
   {"r32 = u173h == 0.0 ? r44 : r46", 32, true,
    {{U16, 173, N}, {I, 0, N}, {R, 44, C}, {R, 46, C}}, true, 0,
    "022c6f8106ac880200ae"},
   {"r46 = u173h == 0.0 ? r46 : r44", 46, true,
    {{U16, 173, N}, {I, 0, N}, {R, 46, D}, {R, 44, D}}, true, 0,
    "722c6f81842c8802"},
   {"r0 = u48h == 0.0 ? r2 : r0", 0, true,
    {{U16, 48, N}, {I, 0, N}, {R, 2, D}, {R, 0, D}}, true, 0,
    "0230278185028002"},
   /* fself32: a selected minifloat beside a uniform */
   {"r32 = r4 <= u46 ? r32 : 0.0", 32, true,
    {{R, 4, C}, {U, 46, N}, {R, 32, D}, {I, 0, N}}, false, 6,
    "0285632f24800302"},
   {"r18 = r20 < u62 ? 0.0 : r18", 18, true,
    {{R, 20, D}, {U, 62, N}, {I, 0, N}, {R, 18, D}}, false, 1,
    "92152b3f21800302"},
   {"r20 = r4 < 2.0 ? 1.0 : u20", 20, true,
    {{R, 4, D}, {I, 32, N}, {I, 48, N}, {U, 20, N}}, false, 1,
    "a2052ba122b003022014"},
};

static bool
describe(const struct witness *w, struct agx3_fsel32_gpu1_desc *d)
{
   memset(d, 0, sizeof(*d));
   d->dest = w->dest;
   d->dest_cache = w->dest_cache;
   d->sel_integer = w->sel_integer;
   d->condition = w->condition;
   const struct operand *s = w->src;
   if (s[0].kind == I)
      return false;
   d->cmp0 = s[0].value;
   d->cmp0_hint = s[0].hint;
   d->cmp0_uniform = s[0].kind == U || s[0].kind == U16;
   d->cmp0_16 = s[0].kind == U16;
   d->cmp1 = s[1].value;
   d->cmp1_hint = s[1].hint;
   d->cmp1_immediate = s[1].kind == I;
   d->cmp1_uniform = s[1].kind == U || s[1].kind == U16;
   d->cmp1_16 = s[1].kind == U16;
   for (unsigned i = 0; i < 2; ++i) {
      d->sel[i] = s[2 + i].value;
      d->sel_hint[i] = s[2 + i].hint;
      d->sel_immediate[i] = s[2 + i].kind == I;
      d->sel_uniform[i] = s[2 + i].kind == U;
      if (s[2 + i].kind == U16)
         return false;
   }
   return true;
}

static void
hex(char *out, const uint8_t *b, unsigned n)
{
   for (unsigned i = 0; i < n; ++i)
      sprintf(out + 2 * i, "%02x", b[i]);
   out[2 * n] = 0;
}

int
main(void)
{
   unsigned failures = 0;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      const struct witness *w = &witnesses[i];
      struct agx3_fsel32_gpu1_desc d;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      char got[2 * AGX3_ALU_MAX_LENGTH + 1] = "refused";
      if (describe(w, &d) && agx3_alu_fcmp_select32_gpu1(&d, bytes, &len))
         hex(got, bytes, len);
      if (strcmp(got, w->bytes)) {
         fprintf(stderr, "%s: got %s, native %s\n", w->name, got, w->bytes);
         failures++;
      }
   }

   /* Shapes outside the witnessed ones are refused. */
   static const struct witness refused[] = {
      /* both comparands uniform */
      {"u2 < u4 ? r6 : r8", 0, false,
       {{U, 2, N}, {U, 4, N}, {R, 6, N}, {R, 8, N}}, true, 1, NULL},
      /* a uniform first comparand beside a register: callers exchange it */
      {"u2 < r4 ? r6 : r8", 0, false,
       {{U, 2, N}, {R, 4, N}, {R, 6, N}, {R, 8, N}}, true, 1, NULL},
      /* a uniform first comparand against zero, other than equality */
      {"u2 < 0.0 ? r6 : r8", 0, false,
       {{U, 2, N}, {I, 0, N}, {R, 6, N}, {R, 8, N}}, true, 1, NULL},
      /* a uniform first comparand against an immediate other than zero */
      {"u2h == 2.0 ? r6 : r8", 0, false,
       {{U16, 2, N}, {I, 32, N}, {R, 6, N}, {R, 8, N}}, true, 0, NULL},
      {"u2 == 2.0 ? r6 : r8", 0, false,
       {{U, 2, N}, {I, 32, N}, {R, 6, N}, {R, 8, N}}, true, 0, NULL},
      /* a uniform first comparand selecting an immediate (fseli32) */
      {"u2 == 0.0 ? 1 : r8", 0, false,
       {{U, 2, N}, {I, 0, N}, {I, 1, N}, {R, 8, N}}, true, 0, NULL},
      /* two uniform selects beside a register comparand */
      {"r2 < r4 ? u6 : u8", 0, false,
       {{R, 2, N}, {R, 4, N}, {U, 6, N}, {U, 8, N}}, true, 1, NULL},
      /* a uniform with a lifetime hint */
      {"r2 < u4 ? r6 : r8 (hinted)", 0, false,
       {{R, 2, N}, {U, 4, C}, {R, 6, N}, {R, 8, N}}, true, 1, NULL},
      /* an unaligned 32-bit uniform */
      {"r2 < u5 ? r6 : r8", 0, false,
       {{R, 2, N}, {U, 5, N}, {R, 6, N}, {R, 8, N}}, true, 1, NULL},
      /* a uniform selected operand in fself32 */
      {"r2 < r4 ? u6 : 0.0", 0, false,
       {{R, 2, N}, {R, 4, N}, {U, 6, N}, {I, 0, N}}, false, 1, NULL},
   };
   for (unsigned i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i) {
      struct agx3_fsel32_gpu1_desc d;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (describe(&refused[i], &d) &&
          agx3_alu_fcmp_select32_gpu1(&d, bytes, &len)) {
         fprintf(stderr, "%s: encoded, expected a refusal\n", refused[i].name);
         failures++;
      }
   }

   if (failures)
      fprintf(stderr, "%u failures\n", failures);
   return failures ? 1 : 0;
}
