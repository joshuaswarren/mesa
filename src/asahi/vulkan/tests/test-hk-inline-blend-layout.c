/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "hk_inline_blend_layout.h"

static unsigned checks, failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      ++checks;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static bool
reason_is(const char *reason, const char *expected)
{
   return reason && !strcmp(reason, expected);
}

/* A fragment part laid out as the backend lays one out: the constant
 * program first, then the main. */
struct part {
   struct agx_shader_part b;
   uint8_t code[64];
};

static void
part_init(struct part *p, bool preamble, uint8_t seed)
{
   memset(p, 0, sizeof(*p));
   for (unsigned i = 0; i < sizeof(p->code); ++i)
      p->code[i] = (uint8_t)(seed + i);
   p->b.binary = p->code;
   p->b.info.binary_size = sizeof(p->code);
   if (preamble) {
      p->b.info.has_preamble = true;
      p->b.info.preamble_offset = 0;
      p->b.info.preamble_size = 16;
      p->b.info.nr_preamble_gprs = 4;
      p->b.info.agx3_preamble_staging_known = true;
      p->b.info.agx3_preamble_staging_halfs = 8;
      p->b.info.agx3_preamble_inputs.count = 1;
      p->b.info.agx3_preamble_inputs.storage_bytes = 4;
      p->b.info.agx3_preamble_inputs.halves[0] = 0x3;
   }
   p->b.info.main_offset = preamble ? 16 : 0;
   p->b.info.main_size = sizeof(p->code) - p->b.info.main_offset;
}

/* An API shader without a constant program: a program that forms one has
 * its own layout (the push-constant colour case), one that forms none keeps
 * the API shader's. */
static void
test_api_without_cprog(void)
{
   struct part api, mono;
   part_init(&api, false, 0);

   part_init(&mono, true, 0x40);
   CHECK(hk_g17p_inline_blend_own_layout(&api.b, &mono.b));
   CHECK(hk_g17p_cprog_mismatch(&api.b, &mono.b) != NULL);

   part_init(&mono, false, 0x40);
   CHECK(!hk_g17p_inline_blend_own_layout(&api.b, &mono.b));
   CHECK(hk_g17p_cprog_mismatch(&api.b, &mono.b) == NULL);
}

/* An API shader with a constant program (a colour read from a uniform
 * buffer). The same program keeps the API layout; any other program --
 * typically one that also took the inlined blend's uniform terms -- has its
 * own. A program without a constant program has neither: the API shader's
 * would run for a main that does not read it. */
static void
test_api_with_cprog(void)
{
   struct part api, mono;
   part_init(&api, true, 0);

   /* Same code and live-ins; the mains may differ. */
   part_init(&mono, true, 0);
   for (unsigned i = 16; i < sizeof(mono.code); ++i)
      mono.code[i] ^= 0xff;
   CHECK(hk_g17p_cprog_mismatch(&api.b, &mono.b) == NULL);
   CHECK(!hk_g17p_inline_blend_own_layout(&api.b, &mono.b));

   /* The blend's source terms moved in: a longer constant program. */
   part_init(&mono, true, 0);
   mono.b.info.preamble_size = 24;
   mono.b.info.main_offset = 24;
   mono.b.info.main_size = sizeof(mono.code) - 24;
   CHECK(hk_g17p_cprog_mismatch(&api.b, &mono.b) != NULL);
   CHECK(hk_g17p_inline_blend_own_layout(&api.b, &mono.b));

   /* Same size, different code. */
   part_init(&mono, true, 0);
   mono.code[3] ^= 1;
   CHECK(reason_is(hk_g17p_cprog_mismatch(&api.b, &mono.b), "preamble code"));
   CHECK(hk_g17p_inline_blend_own_layout(&api.b, &mono.b));

   /* Same code, different frozen live-ins. */
   part_init(&mono, true, 0);
   mono.b.info.agx3_preamble_inputs.halves[0] = 0x5;
   CHECK(reason_is(hk_g17p_cprog_mismatch(&api.b, &mono.b), "preamble inputs"));
   CHECK(hk_g17p_inline_blend_own_layout(&api.b, &mono.b));

   /* Each launch parameter of the constant program on its own. */
#define DIFFERS_IN(field, value)                                               \
   do {                                                                        \
      part_init(&mono, true, 0);                                               \
      mono.b.info.field = value;                                               \
      CHECK(reason_is(hk_g17p_cprog_mismatch(&api.b, &mono.b), #field));       \
      CHECK(hk_g17p_inline_blend_own_layout(&api.b, &mono.b));                 \
   } while (0)
   DIFFERS_IN(nr_preamble_gprs, 6);
   DIFFERS_IN(agx3_preamble_uniform_inputs, true);
   DIFFERS_IN(agx3_preamble_tps_inputs, true);
   DIFFERS_IN(preamble_scratch_size, 16);
   DIFFERS_IN(agx3_preamble_staging_known, false);
   DIFFERS_IN(agx3_preamble_staging_halfs, 12);
   DIFFERS_IN(agx3_preamble_inputs.count, 2);
   DIFFERS_IN(agx3_preamble_inputs.storage_bytes, 8);
#undef DIFFERS_IN

   /* No constant program: refused by the layout check, never its own. */
   part_init(&mono, false, 0);
   CHECK(reason_is(hk_g17p_cprog_mismatch(&api.b, &mono.b), "has_preamble"));
   CHECK(!hk_g17p_inline_blend_own_layout(&api.b, &mono.b));
}

/* A constant program outside the program's own binary is never launched. */
static void
test_extent(void)
{
   struct part api, mono;
   part_init(&api, true, 0);

   part_init(&mono, true, 0);
   mono.code[0] ^= 1;
   mono.b.info.preamble_offset = sizeof(mono.code) - 8;
   CHECK(!hk_g17p_inline_blend_own_layout(&api.b, &mono.b));
   CHECK(hk_g17p_cprog_mismatch(&api.b, &mono.b) != NULL);

   part_init(&mono, true, 0);
   mono.b.info.preamble_offset = sizeof(mono.code) + 2;
   CHECK(!hk_g17p_inline_blend_own_layout(&api.b, &mono.b));

   part_init(&mono, true, 0);
   mono.b.binary = NULL;
   CHECK(!hk_g17p_inline_blend_own_layout(&api.b, &mono.b));

   struct part none;
   part_init(&none, false, 0);
   part_init(&mono, true, 0);
   mono.b.info.preamble_size = 0;
   CHECK(!hk_g17p_inline_blend_own_layout(&none.b, &mono.b));
}

int
main(void)
{
   test_api_without_cprog();
   test_api_with_cprog();
   test_extent();

   if (failures) {
      fprintf(stderr, "hk_inline_blend_layout: %u of %u checks failed\n",
              failures, checks);
      return 1;
   }
   printf("hk_inline_blend_layout: %u checks passed\n", checks);
   return 0;
}
