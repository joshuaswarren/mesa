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
 * A last one composes a different mnemonic from a remembered field vector, so
 * a memo that did not key on the mnemonic fails.
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
fail(const char *what, const char *name, size_t at, enum agx3_mnemonic mnemonic)
{
   if (failures++ < 20)
      fprintf(stderr, "%s: 0x%zx %s: %s\n", name, at,
              agx3_mnemonic_name(mnemonic), what);
}

/* Compose through the binder and compare with the reference bytes. */
static void
bind_compare(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic, int ep,
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

/* The first agx3c instruction of the corpus that composes in Apple's order. */
static enum agx3_mnemonic sample_mnemonic;
static uint64_t sample_fields[AGX3_MAX_FIELDS];

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
      const int32_t form = agx3_decode(t, code + i, n, AGX3_MN_NONE, fields);
      if (form >= 0) {
         enum agx3_mnemonic mnemonic = t->forms[form].mnemonic;
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
               bind_compare(enc, mnemonic, t->forms[form].ep, kv, nkv, ref, n,
                          name, i);
         }

         const int32_t apple = agx3_compose_apple_order(t, mnemonic, fields, ref);
         if (apple >= 0)
            bind_compare(enc, mnemonic, AGX3_BIND_APPLE_ORDER, kv, nkv, ref,
                       t->forms[apple].size, name, i);
         if (apple >= 0 && !g18g && !sample_mnemonic) {
            sample_mnemonic = mnemonic;
            memcpy(sample_fields, fields, sizeof(fields));
         }
      }
      i += n;
   }
}

/*
 * Without the mnemonic, the memo key is the same for every mnemonic composed
 * from one field vector. Compose the sample, then another mnemonic from the
 * same vector: the binder must give what the table walk gives for that
 * mnemonic, a refusal included.
 */
static bool
check_other_mnemonic(void)
{
   const agx3_table *t = &agx3c_table;
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3C;
   if (!sample_mnemonic)
      return false;

   struct agx3_field_kv kv[AGX3_MAX_FIELDS];
   unsigned nkv = 0;
   for (unsigned f = 0; f < AGX3_MAX_FIELDS; ++f) {
      if (sample_fields[f])
         kv[nkv++] =
            (struct agx3_field_kv){.field = f, .value = sample_fields[f]};
   }

   uint8_t ref[2 * AGX3_MAX_WORDS];
   int32_t form =
      agx3_compose_apple_order(t, sample_mnemonic, sample_fields, ref);
   if (form < 0)
      return false;
   bind_compare(enc, sample_mnemonic, AGX3_BIND_APPLE_ORDER, kv, nkv, ref,
              t->forms[form].size, "other mnemonic", 0);

   enum agx3_mnemonic other = AGX3_MN_NONE;
   for (uint32_t f = 0; f < t->num_forms && !other; ++f) {
      if (t->forms[f].mnemonic != sample_mnemonic)
         other = t->forms[f].mnemonic;
   }
   if (!other)
      return false;
   uint8_t out[2 * AGX3_MAX_WORDS];
   uint8_t len = 0;
   const bool bound = agx3_bind_compose(enc, other, AGX3_BIND_APPLE_ORDER,
                                        kv, nkv, NULL, 0, out, &len);
   form = agx3_compose_apple_order(t, other, sample_fields, ref);
   checks++;
   if (bound != (form >= 0) ||
       (bound && (len != t->forms[form].size || memcmp(out, ref, len))))
      fail("binder differs from the table walk for another mnemonic",
           "other mnemonic", 0, other);

   /* Verify that an override setting a nonzero baseline field back to zero
    * clears its nonzero bit and composes identically to omitting that field. */
   for (unsigned extra = 0; extra < AGX3_MAX_FIELDS; ++extra) {
      if (sample_fields[extra])
         continue;
      kv[nkv] = (struct agx3_field_kv){.field = extra, .value = 1};
      const struct agx3_field_kv zero_override = {.field = extra, .value = 0};
      form = agx3_compose_apple_order(t, sample_mnemonic, sample_fields, ref);
      uint8_t ov_out[2 * AGX3_MAX_WORDS];
      uint8_t ov_len = 0;
      checks++;
      if (!agx3_bind_compose(enc, sample_mnemonic, AGX3_BIND_APPLE_ORDER, kv,
                             nkv + 1, &zero_override, 1, ov_out, &ov_len) ||
          ov_len != t->forms[form].size || memcmp(ov_out, ref, ov_len))
         fail("zero override did not clear baseline nonzero bit",
              "zero override", 0, sample_mnemonic);
      break;
   }

   /* Verify agx3_bind_compose_ep_range matches looping agx3_bind_compose,
    * honors skip_ep and max_length, and refuses AGX3D. */
   uint8_t range_out[2 * AGX3_MAX_WORDS];
   uint8_t range_len = 0;
   checks++;
   if (!agx3_bind_compose_ep_range(enc, sample_mnemonic, 0, 35, -1, 0, false,
                                   kv, nkv, NULL, 0, range_out, &range_len)) {
      fail("ep_range failed for valid sample", "ep_range", 0, sample_mnemonic);
   } else {
      uint8_t loop_out[2 * AGX3_MAX_WORDS];
      uint8_t loop_len = 0;
      bool loop_ok = false;
      unsigned matched_ep = 0;
      for (unsigned ep = 0; ep <= 35; ++ep) {
         if (agx3_bind_compose(enc, sample_mnemonic, (int)ep, kv, nkv, NULL, 0,
                               loop_out, &loop_len)) {
            loop_ok = true;
            matched_ep = ep;
            break;
         }
      }
      if (!loop_ok || range_len != loop_len ||
          memcmp(range_out, loop_out, loop_len))
         fail("ep_range differed from ep loop", "ep_range", 0, sample_mnemonic);

      checks++;
      uint8_t capped_out[2 * AGX3_MAX_WORDS];
      uint8_t capped_len = 0;
      if (agx3_bind_compose_ep_range(enc, sample_mnemonic, matched_ep,
                                     matched_ep, -1, loop_len, true, kv, nkv,
                                     NULL, 0, capped_out, &capped_len))
         fail("ep_range ignored max_length cap", "ep_range", 0,
              sample_mnemonic);

      checks++;
      if (agx3_bind_compose_ep_range(enc, sample_mnemonic, matched_ep,
                                     matched_ep, (int)matched_ep, 0, false, kv,
                                     nkv, NULL, 0, capped_out, &capped_len))
         fail("ep_range ignored skip_ep", "ep_range", 0, sample_mnemonic);
   }

   checks++;
   if (agx3_bind_compose_ep_range(AGX_ISA_ENCODING_AGX3D, sample_mnemonic, 0,
                                  35, -1, 0, false, kv, nkv, NULL, 0,
                                  range_out, &range_len))
      fail("ep_range accepted AGX3D", "ep_range", 0, sample_mnemonic);

   return true;
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
   if (!check_other_mnemonic()) {
      fprintf(stderr, "no agx3c sample for the reused name check\n");
      return 1;
   }

   printf("%u native texts, %lu compositions, %lu failures\n", texts, checks,
          failures);
   return failures ? 1 : 0;
}
