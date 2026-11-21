/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx3_transcode against Apple: every G17G witness instruction whose G18G
 * counterpart Apple compiled from the same source must re-encode to exactly
 * Apple's G18G bytes (agx3d_transcode_vectors.c, from the committed witness
 * texts), and every branch of a re-encoded G17G program must land where
 * Apple's G18G branch lands. The Ioto rules are checked against an
 * instruction from Apple's own G18G Ioto programs, and Ioto-only
 * instructions must be refused in shader code.
 */
#include <stdio.h>
#include <string.h>

#include "util/macros.h"
#include "util/u_dynarray.h"
#include "agx3_compose/agx3_decode.h"
#include "asahi/isa/disasm.h"
#include "agx3_transcode.h"
#include "agx3d_transcode_vectors.c"

/* Apple's two compilers allocate scoreboard slots on their own; for those
 * vectors compare the decoded G18G fields with the slot field left out. */
static bool
same_fields_except(const char *mnemonic, const uint8_t *a, unsigned alen,
                   const uint8_t *b, unsigned blen, unsigned ignore)
{
   uint64_t fa[AGX3_MAX_FIELDS], fb[AGX3_MAX_FIELDS];
   const enum agx3_mnemonic m = agx3_mnemonic_from_name(mnemonic);
   if (m == AGX3_MN_NONE || agx3_decode(&agx3d_table, a, alen, m, fa) < 0 ||
       agx3_decode(&agx3d_table, b, blen, m, fb) < 0)
      return false;
   fa[ignore] = fb[ignore] = 0;
   return !memcmp(fa, fb, sizeof(fa));
}

static bool
transcode(const uint8_t *in, unsigned len,
          const struct agx3_transcode_ioto *ioto, uint8_t *out,
          unsigned *out_len, const char **why)
{
   struct util_dynarray code;
   util_dynarray_init(&code, NULL);
   memcpy(util_dynarray_grow_bytes(&code, 1, len), in, len);
   unsigned at;
   const bool ok = ioto ? agx3_transcode_agx3d_ioto(&code, 0, ioto, why, &at)
                        : agx3_transcode_agx3d(&code, 0, why, &at);
   if (ok) {
      *out_len = code.size;
      memcpy(out, code.data, code.size);
   }
   util_dynarray_fini(&code);
   return ok;
}

int
main(void)
{
   unsigned failures = 0;
   const unsigned count =
      sizeof(agx3d_transcode_vectors) / sizeof(agx3d_transcode_vectors[0]);

   for (unsigned i = 0; i < count; ++i) {
      const struct agx3d_transcode_vector *v = &agx3d_transcode_vectors[i];
      uint8_t out[64];
      unsigned len = 0;
      const char *why = NULL;
      const bool ok =
         transcode(v->g17g, v->g17g_len, NULL, out, &len, &why) &&
         (v->ignore_field == 0xff
             ? len == v->g18g_len && !memcmp(out, v->g18g, len)
             : same_fields_except(v->mnemonic, out, len, v->g18g, v->g18g_len,
                                  v->ignore_field));
      if (!ok) {
         fprintf(stderr, "MISMATCH %s (%s): %s\n", v->mnemonic, v->text,
                 why ? why : "different bytes");
         failures++;
      }
   }

   /* The M5 generator's Ioto helper call, against the G18G helper call in
    * every Ioto program of the J873 native captures (sessions 20260925). */
   static const uint8_t g17g_cndcall[] = {0x27, 0x04, 0x04, 0x00,
                                          0x30, 0x00, 0xa4, 0x12};
   static const uint8_t apple_g18g_cndcall[] = {0x10, 0x52, 0x0d, 0x00};
   const struct agx3_transcode_ioto ioto = {.api_program = true,
                                            .shader_gprs = 8};
   uint8_t out[64];
   unsigned len = 0;
   const char *why = NULL;
   if (!transcode(g17g_cndcall, sizeof(g17g_cndcall), &ioto, out, &len,
                  &why) ||
       len != sizeof(apple_g18g_cndcall) ||
       memcmp(out, apple_g18g_cndcall, len)) {
      fprintf(stderr, "MISMATCH Ioto cndcall: %s\n",
              why ? why : "different bytes");
      failures++;
   }

   /* Branches: re-encode Apple's whole G17G program, then every b/bn must be
    * Apple's ten-byte G18G form and land on the instruction Apple's G18G
    * counterpart branch lands on. */
   const unsigned programs = sizeof(agx3d_transcode_branch_programs) /
                             sizeof(agx3d_transcode_branch_programs[0]);
   unsigned branches = 0;
   for (unsigned p = 0; p < programs; ++p) {
      const struct agx3d_transcode_branch_program *prog =
         &agx3d_transcode_branch_programs[p];
      struct util_dynarray code;
      util_dynarray_init(&code, NULL);
      memcpy(util_dynarray_grow_bytes(&code, 1, prog->g17g_len), prog->g17g,
             prog->g17g_len);
      unsigned at = 0;
      why = NULL;
      if (!agx3_transcode_agx3d(&code, 0, &why, &at)) {
         fprintf(stderr, "MISMATCH %s: refused at +0x%x: %s\n", prog->text, at,
                 why);
         failures++;
         util_dynarray_fini(&code);
         continue;
      }

      /* Instruction offsets of the re-encoded program, plus its end. */
      unsigned offsets[512], count = 0;
      const uint8_t *bytes = code.data;
      bool segmented = true;
      for (unsigned off = 0; off < code.size && segmented;) {
         const signed n = agx3d_instr_length(bytes + off, code.size - off, NULL);
         segmented = n > 0 && count + 1 < ARRAY_SIZE(offsets);
         if (segmented) {
            offsets[count++] = off;
            off += n;
         }
      }
      offsets[count] = code.size;

      for (unsigned c = 0; c < prog->count && segmented; ++c) {
         const unsigned b = prog->checks[c].branch, t = prog->checks[c].target;
         uint64_t fields[AGX3_MAX_FIELDS];
         const bool ok =
            b < count && t <= count && offsets[b + 1] - offsets[b] == 10 &&
            (agx3_decode(&agx3d_table, bytes + offsets[b], 10, AGX3_MN_b, fields) >= 0 ||
             agx3_decode(&agx3d_table, bytes + offsets[b], 10, AGX3_MN_bn, fields) >= 0) &&
            (int64_t)offsets[b] + ((int64_t)(fields[0] << 16) >> 16) ==
               (int64_t)offsets[t];
         if (!ok) {
            fprintf(stderr, "MISMATCH %s: branch %u does not land on %u\n",
                    prog->text, b, t);
            failures++;
         }
         branches++;
      }
      if (!segmented) {
         fprintf(stderr, "MISMATCH %s: output does not segment\n", prog->text);
         failures++;
      }
      util_dynarray_fini(&code);
   }

   /* Negative control: the same helper call is Ioto-only and must not be
    * re-encoded as shader code. */
   why = NULL;
   if (transcode(g17g_cndcall, sizeof(g17g_cndcall), NULL, out, &len, &why) ||
       !why || !strstr(why, "Ioto-only")) {
      fprintf(stderr, "shader-code cndcall was not refused as Ioto-only\n");
      failures++;
   }

   printf("agx3_transcode: %u Apple vectors, %u branches in %u programs, "
          "%u failures\n",
          count, branches, programs, failures);
   return failures ? 1 : 0;
}
