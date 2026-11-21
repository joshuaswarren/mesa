/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The in-tree AGX3 encoder must reproduce real Apple shader instructions. This
 * replays a pinned slice of the G16G/G17S corpus -- each instruction decoded to
 * its field vector by the Python composer -- and requires the C encoder to
 * produce identical bytes for every one. The G18G set does the same for the
 * agx3d table, over the committed test/native/g18g witness texts.
 */

#include "agx3_compose/agx3_compose.h"
#include "agx3_compose_corpus_vectors.c"
#include "agx3d_compose_corpus_vectors.c"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* `agx3_compose_corpus_test N` replays both corpora N times and prints what
 * one instruction costs in the fastest replay: fill the field vector, find
 * the form, encode. */
static int
bench(unsigned passes)
{
   uint8_t out[AGX3_MAX_WORDS * 2];
   unsigned sink = 0;
   double best = DBL_MAX, best_g18g = DBL_MAX;

   /* The vectors spell their mnemonics; an encoder has them as constants. */
   static enum agx3_mnemonic mnemonics[4096], mnemonics_g18g[4096];
   if (agx3_compose_corpus_case_count > 4096 ||
       agx3d_compose_corpus_case_count > 4096)
      return 1;
   for (size_t i = 0; i < agx3_compose_corpus_case_count; ++i)
      mnemonics[i] =
         agx3_mnemonic_from_name(agx3_compose_corpus_cases[i].mnemonic);
   for (size_t i = 0; i < agx3d_compose_corpus_case_count; ++i)
      mnemonics_g18g[i] =
         agx3_mnemonic_from_name(agx3d_compose_corpus_cases[i].mnemonic);

   for (unsigned pass = 0; pass < passes; ++pass) {
      double start = now_ns();
      for (size_t i = 0; i < agx3_compose_corpus_case_count; ++i) {
         const struct agx3_compose_corpus_case *c = &agx3_compose_corpus_cases[i];
         const agx3_table *t = c->enc == 0 ? &agx3b_table : &agx3c_table;
         uint64_t fields[AGX3_MAX_FIELDS] = {0};
         for (uint16_t k = 0; k < c->nfields; ++k)
            fields[c->field_ids[k]] = c->field_vals[k];
         int32_t form = agx3_form_index(t, mnemonics[i], c->ep);
         sink += agx3_compose(t, (uint32_t)form, fields, out) + out[0];
      }
      double took = now_ns() - start;
      if (took < best)
         best = took;

      start = now_ns();
      for (size_t i = 0; i < agx3d_compose_corpus_case_count; ++i) {
         const struct agx3d_compose_corpus_case *c = &agx3d_compose_corpus_cases[i];
         uint64_t fields[AGX3_MAX_FIELDS] = {0};
         for (uint8_t k = 0; k < c->nfields; ++k)
            fields[c->field_ids[k]] = c->field_vals[k];
         int32_t form = agx3_form_index(&agx3d_table, mnemonics_g18g[i], c->ep);
         sink += agx3_compose(&agx3d_table, (uint32_t)form, fields, out) + out[0];
      }
      took = now_ns() - start;
      if (took < best_g18g)
         best_g18g = took;
   }
   printf("agx3b/agx3c: %.1f ns per instruction\n",
          best / agx3_compose_corpus_case_count);
   printf("agx3d: %.1f ns per instruction\n",
          best_g18g / agx3d_compose_corpus_case_count);
   return sink == 0;
}

int
main(int argc, char **argv)
{
   if (argc > 1)
      return bench((unsigned)atoi(argv[1]));

   unsigned pass = 0, fail = 0;

   const agx3_table *tables[] = {&agx3b_table, &agx3c_table, &agx3d_table};
   for (unsigned table = 0; table < 3; ++table) {
      const agx3_table *t = tables[table];
      const char letter = "bcd"[table];

      for (uint32_t i = 0; i < t->num_forms; ++i) {
         const agx3_form *f = &t->forms[i];
         const char *name = agx3_mnemonic_name(f->mnemonic);
         if (agx3_mnemonic_from_name(name) != f->mnemonic ||
             agx3_form_index(t, f->mnemonic, f->ep) != (int32_t)i) {
            fprintf(stderr, "FAIL form lookup %s.ep%u (agx3%c)\n", name,
                    f->ep, letter);
            fail++;
         }

         if (f->size % 2 || f->size / 2 > AGX3_MAX_WORDS) {
            fprintf(stderr, "FAIL form size %s.ep%u = %u (agx3%c)\n",
                    name, f->ep, f->size, letter);
            fail++;
         }

         if (i > 0) {
            const agx3_form *prev = &t->forms[i - 1];
            if (prev->mnemonic > f->mnemonic ||
                (prev->mnemonic == f->mnemonic && prev->ep >= f->ep)) {
               fprintf(stderr, "FAIL unsorted forms %s.ep%u, %s.ep%u "
                               "(agx3%c)\n",
                       agx3_mnemonic_name(prev->mnemonic), prev->ep, name,
                       f->ep, letter);
               fail++;
            }
         }
      }

      if (agx3_mnemonic_from_name("") != AGX3_MN_NONE ||
          agx3_mnemonic_from_name("~unknown") != AGX3_MN_NONE ||
          strcmp(agx3_mnemonic_name(AGX3_MN_NONE), "?") ||
          strcmp(agx3_mnemonic_name(AGX3_MN_COUNT), "?") ||
          agx3_form_index(t, AGX3_MN_NONE, 0) != -1 ||
          agx3_form_index(t, AGX3_MN_COUNT, UINT8_MAX) != -1) {
         fprintf(stderr, "FAIL unknown form lookup (agx3%c)\n", letter);
         fail++;
      }
   }

   for (size_t i = 0; i < agx3_compose_corpus_case_count; ++i) {
      const struct agx3_compose_corpus_case *c = &agx3_compose_corpus_cases[i];
      const agx3_table *t = c->enc == 0 ? &agx3b_table : &agx3c_table;

      uint64_t fields[AGX3_MAX_FIELDS] = {0};
      for (uint16_t k = 0; k < c->nfields; ++k)
         fields[c->field_ids[k]] = c->field_vals[k];

      int32_t form =
         agx3_form_index(t, agx3_mnemonic_from_name(c->mnemonic), c->ep);
      uint8_t out[AGX3_MAX_WORDS * 2];

      if (form < 0 || !agx3_compose(t, (uint32_t)form, fields, out) ||
          memcmp(out, c->bytes, c->nbytes) != 0) {
         if (fail < 8)
            fprintf(stderr, "FAIL case %zu: %s.ep%d (agx3%c)\n", i, c->mnemonic,
                    c->ep, c->enc == 0 ? 'b' : 'c');
         fail++;
      } else {
         pass++;
      }
   }

   unsigned g18g_pass = 0;
   for (size_t i = 0; i < agx3d_compose_corpus_case_count; ++i) {
      const struct agx3d_compose_corpus_case *c = &agx3d_compose_corpus_cases[i];
      const agx3_table *t = &agx3d_table;

      uint64_t fields[AGX3_MAX_FIELDS] = {0};
      for (uint8_t k = 0; k < c->nfields; ++k)
         fields[c->field_ids[k]] = c->field_vals[k];

      int32_t form =
         agx3_form_index(t, agx3_mnemonic_from_name(c->mnemonic), c->ep);
      uint8_t out[AGX3_MAX_WORDS * 2];

      if (form < 0 || t->forms[form].size != c->nbytes ||
          !agx3_compose(t, (uint32_t)form, fields, out) ||
          memcmp(out, c->bytes, c->nbytes) != 0) {
         if (fail < 8)
            fprintf(stderr, "FAIL G18G case %zu: %s.ep%u (%s+0x%x)\n", i,
                    c->mnemonic, c->ep, c->text, c->offset);
         fail++;
      } else {
         g18g_pass++;
      }
   }

   if (fail) {
      fprintf(stderr, "agx3_compose corpus: %u of %zu failed\n", fail,
              agx3_compose_corpus_case_count +
                 agx3d_compose_corpus_case_count);
      return 1;
   }
   printf("agx3_compose corpus: all %u instructions reproduced\n", pass);
   printf("agx3_compose G18G corpus: all %u instructions reproduced\n",
          g18g_pass);
   return 0;
}
