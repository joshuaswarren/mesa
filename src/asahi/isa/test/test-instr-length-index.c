/*
 * Copyright 2026 Aurora Silicon Project
 * SPDX-License-Identifier: MIT
 */

/*
 * The indexed instruction length lookup (gen-disasm.py, length_index()) must
 * give exactly what testing every form in table order gives: the same length
 * and the same form (its display string). Checked for every encoding on
 *
 *  - random code at every `available` from 0 to 24 bytes, and
 *  - every byte offset of every committed G17G/G18G native witness text
 *    (src/asahi/compiler/test/native/g18g, the directory argument), under
 *    every encoding, so real instructions and their neighbours are covered.
 */

#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/isa/disasm.h"

typedef signed (*length_fn)(const uint8_t *, size_t, const char **);

static const struct {
   const char *name;
   length_fn indexed, linear;
} isas[] = {
   {"agx2", agx2_instr_length, agx2_instr_length_linear},
   {"agx3b", agx3b_instr_length, agx3b_instr_length_linear},
   {"agx3c", agx3c_instr_length, agx3c_instr_length_linear},
   {"agx3d", agx3d_instr_length, agx3d_instr_length_linear},
};

#define MAX_INSTR 24

static unsigned long checks, matches, failures;

static void
check(unsigned isa, const uint8_t *code, size_t available)
{
   /* Exactly `available` bytes, so a sanitizer sees any read past them */
   uint8_t *copy = malloc(available ? available : 1);
   memcpy(copy, code, available);

   const char *display_indexed = NULL, *display_linear = NULL;
   const signed a = isas[isa].indexed(copy, available, &display_indexed);
   const signed b = isas[isa].linear(copy, available, &display_linear);
   free(copy);

   checks++;
   matches += b > 0;
   if (a != b || (b > 0 && display_indexed != display_linear)) {
      if (failures++ < 20) {
         fprintf(stderr, "%s: available %zu:", isas[isa].name, available);
         for (size_t i = 0; i < available; ++i)
            fprintf(stderr, " %02x", code[i]);
         fprintf(stderr, ": indexed %d (%s), linear %d (%s)\n", a,
                 display_indexed ? display_indexed : "-", b,
                 display_linear ? display_linear : "-");
      }
   }
}

static uint64_t
xorshift(uint64_t *s)
{
   *s ^= *s << 13;
   *s ^= *s >> 7;
   *s ^= *s << 17;
   return *s;
}

static void
check_random(void)
{
   uint64_t state = 0x9e3779b97f4a7c15ull;
   for (unsigned n = 0; n < 200000; ++n) {
      uint8_t code[MAX_INSTR];
      for (unsigned i = 0; i < MAX_INSTR; i += 8) {
         const uint64_t r = xorshift(&state);
         memcpy(code + i, &r, 8);
      }
      for (unsigned isa = 0; isa < ARRAY_SIZE(isas); ++isa)
         check(isa, code, n % (MAX_INSTR + 1));
   }
}

static bool
check_corpus(const char *dir)
{
   DIR *d = opendir(dir);
   if (!d) {
      fprintf(stderr, "cannot open %s\n", dir);
      return false;
   }

   unsigned texts = 0;
   struct dirent *ent;
   while ((ent = readdir(d))) {
      const size_t len = strlen(ent->d_name);
      if (len < 4 || strcmp(ent->d_name + len - 4, ".bin"))
         continue;

      char path[4096];
      snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
      FILE *fp = fopen(path, "rb");
      if (!fp)
         continue;
      uint8_t buf[1 << 16];
      const size_t size = fread(buf, 1, sizeof(buf), fp);
      fclose(fp);

      for (size_t i = 0; i < size; ++i) {
         const size_t left = size - i;
         for (unsigned isa = 0; isa < ARRAY_SIZE(isas); ++isa) {
            check(isa, buf + i, left < MAX_INSTR ? left : MAX_INSTR);
            /* and truncated */
            check(isa, buf + i, (left < MAX_INSTR ? left : MAX_INSTR) / 2);
         }
      }
      texts++;
   }
   closedir(d);

   if (!texts) {
      fprintf(stderr, "no native texts in %s\n", dir);
      return false;
   }
   printf("%u native texts\n", texts);
   return true;
}

int
main(int argc, char **argv)
{
   if (argc != 2) {
      fprintf(stderr, "usage: %s <native witness directory>\n", argv[0]);
      return 1;
   }

   check_random();
   const bool corpus = check_corpus(argv[1]);

   printf("%lu lookups (%lu matched a form), %lu mismatches\n", checks,
          matches, failures);
   return corpus && !failures ? 0 : 1;
}
