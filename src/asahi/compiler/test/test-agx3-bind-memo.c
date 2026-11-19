/*
 * Copyright 2026 Aurora Silicon Project
 * SPDX-License-Identifier: MIT
 */

/*
 * agx3_bind_compose() remembers recent compositions. A remembered one must be
 * exactly what the table walk composes. Every instruction of the committed
 * G17G/G18G native witness texts (src/asahi/compiler/test/native/g18g, the
 * directory argument) is decoded to its form and field vector and composed
 * through agx3_bind_compose() -- by its encoding point and in Apple's order
 * for agx3c, in Apple's order for agx3d -- in two passes over the whole
 * corpus, so the second runs on a memo full of colliding entries. Each result
 * is compared with agx3_compose()/agx3_compose_apple_order() on the same
 * vector, and an encoding-point composition with the instruction's own bytes.
 */

#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx3_binder.h"
#include "agx3_compose/agx3_decode.h"
#include "asahi/isa/disasm.h"

static unsigned long checks, failures;

static void
fail(const char *what, const char *name, size_t at, const char *mnemonic)
{
   if (failures++ < 20)
      fprintf(stderr, "%s: 0x%zx %s: %s\n", name, at, mnemonic, what);
}

/* Compose through the binder and compare with the reference bytes. */
static void
bind_check(enum agx_isa_encoding enc, const char *mnemonic, int ep,
           const struct agx3_field_kv *kv, unsigned n, const uint8_t *ref,
           unsigned ref_len, const char *name, size_t at)
{
   uint8_t out[2 * AGX3_MAX_WORDS];
   uint8_t len = 0;
   checks++;
   if (!agx3_bind_compose(enc, mnemonic, ep, kv, n, NULL, 0, out, &len))
      fail("binder refused", name, at, mnemonic);
   else if (len != ref_len || memcmp(out, ref, len))
      fail("binder differs from the table walk", name, at, mnemonic);
}

static void
check_text(const char *name, const uint8_t *code, size_t size, bool g18g)
{
   const agx3_table *t = g18g ? &agx3d_table : &agx3c_table;
   const enum agx_isa_encoding enc =
      g18g ? AGX_ISA_ENCODING_AGX3D : AGX_ISA_ENCODING_AGX3C;

   for (size_t i = 0; i < size;) {
      const signed n = g18g ? agx3d_instr_length(code + i, size - i, NULL)
                            : agx3c_instr_length(code + i, size - i, NULL);
      if (n <= 0)
         break;

      uint64_t fields[AGX3_MAX_FIELDS];
      const int32_t form = agx3_decode(t, code + i, n, NULL, fields);
      if (form >= 0) {
         const char *mnemonic = t->forms[form].mnemonic;
         struct agx3_field_kv kv[AGX3_MAX_FIELDS];
         unsigned nkv = 0;
         for (unsigned f = 0; f < AGX3_MAX_FIELDS; ++f) {
            if (fields[f])
               kv[nkv++] = (struct agx3_field_kv){.field = f, .value = fields[f]};
         }

         uint8_t ref[2 * AGX3_MAX_WORDS];
         if (!g18g) {
            /* By its own encoding point: the instruction's bytes */
            if (!agx3_compose(t, form, fields, ref) || memcmp(ref, code + i, n))
               fail("decode does not compose back", name, i, mnemonic);
            else
               bind_check(enc, mnemonic, t->forms[form].ep, kv, nkv, ref, n,
                          name, i);
         }

         const int32_t apple = agx3_compose_apple_order(t, mnemonic, fields, ref);
         if (apple >= 0)
            bind_check(enc, mnemonic, AGX3_BIND_APPLE_ORDER, kv, nkv, ref,
                       t->forms[apple].size, name, i);
      }
      i += n;
   }
}

static unsigned
check_corpus(const char *dir)
{
   DIR *d = opendir(dir);
   if (!d)
      return 0;

   unsigned texts = 0;
   struct dirent *ent;
   while ((ent = readdir(d))) {
      const char *s = ent->d_name;
      const size_t len = strlen(s);
      const bool g17g = len > 9 && !strcmp(s + len - 9, ".g17g.bin");
      const bool g18g = len > 9 && !strcmp(s + len - 9, ".g18g.bin");
      if (!g17g && !g18g)
         continue;

      char path[4096];
      snprintf(path, sizeof(path), "%s/%s", dir, s);
      FILE *fp = fopen(path, "rb");
      if (!fp)
         continue;
      static uint8_t buf[1 << 16];
      const size_t size = fread(buf, 1, sizeof(buf), fp);
      fclose(fp);

      /* The constant program at [0, 0x40), then the main program */
      check_text(s, buf, size < 0x40 ? size : 0x40, g18g);
      if (size > 0x40)
         check_text(s, buf + 0x40, size - 0x40, g18g);
      texts++;
   }
   closedir(d);
   return texts;
}

int
main(int argc, char **argv)
{
   if (argc != 2) {
      fprintf(stderr, "usage: %s <native witness directory>\n", argv[0]);
      return 1;
   }

   unsigned texts = 0;
   for (unsigned pass = 0; pass < 2; ++pass)
      texts = check_corpus(argv[1]);
   if (!texts) {
      fprintf(stderr, "no native texts in %s\n", argv[1]);
      return 1;
   }

   printf("%u native texts, %lu compositions, %lu failures\n", texts, checks,
          failures);
   return failures ? 1 : 0;
}
