/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Byte-exact replay of the AGX3 ray-traversal barrier at both witnessed
 * scopes, and refusal of every other scope and of AGX2.
 */

#include "agx3_rt_core.h"

#include <stdio.h>
#include <string.h>

static unsigned failures;
static unsigned checks;

static void
expect_result(const char *what, bool ok, const uint8_t *bytes, uint8_t len,
       const char *expected_hex)
{
   char got[2 * AGX3_RT_CORE_MAX_LENGTH + 1] = {0};

   checks++;

   if (!ok) {
      fprintf(stderr, "%s: refused\n", what);
      failures++;
      return;
   }

   for (unsigned i = 0; i < len; ++i)
      snprintf(got + 2 * i, 3, "%02x", bytes[i]);

   if (strcmp(got, expected_hex) != 0) {
      fprintf(stderr, "%s: got %s, expected %s\n", what, got, expected_hex);
      failures++;
   }
}

static void
expect_refused_result(const char *what, bool ok, const uint8_t *bytes, uint8_t len)
{
   checks++;
   if (ok) {
      fprintf(stderr, "%s: accepted, expected refusal\n", what);
      failures++;
      return;
   }
   /* Refusal must zero the transaction. */
   for (unsigned i = 0; i < AGX3_RT_CORE_MAX_LENGTH; ++i) {
      if (bytes[i] != 0) {
         fprintf(stderr, "%s: refusal left byte %u nonzero\n", what, i);
         failures++;
         return;
      }
   }
   if (len != 0) {
      fprintf(stderr, "%s: refusal left length %u\n", what, len);
      failures++;
   }
}


/* Each encoder writes len.  Finish that call before reading len for the
 * comparison; C does not order function argument evaluation.
 */
#define expect(what, encode, bytes, len, expected_hex)                          \
   do {                                                                       \
      const bool encoded = (encode);                                          \
      expect_result((what), encoded, (bytes), (len), (expected_hex));           \
   } while (0)

#define expect_refused(what, encode, bytes, len)                               \
   do {                                                                       \
      const bool encoded = (encode);                                          \
      expect_refused_result((what), encoded, (bytes), (len));                  \
   } while (0)

int
main(void)
{
   uint8_t b[AGX3_RT_CORE_MAX_LENGTH];
   uint8_t len;

   /* Full traversal native witnesses: entry versus in-loop is positional,
    * not selected by whether the next operation is intersect or continue. */
   static const char *barrier_hex[2][2] = {
      {"070454080900", "070454480900"},
      {"170900060000", "171100060000"},
   };
   for (unsigned family = 0; family < 2; ++family) {
      const enum agx_isa_encoding enc = family ? AGX_ISA_ENCODING_AGX3C :
                                                AGX_ISA_ENCODING_AGX3B;
      for (unsigned scope = 0; scope <= 255; ++scope) {
         memset(b, 0xa5, sizeof(b));
         len = 0xa5;
         const bool ok = agx3_rt_traversal_barrier(
            enc, (enum agx3_rt_barrier_scope)scope, b, &len);
         if (scope == 1 || scope == 2)
            expect("native traversal barrier", ok, b, len,
                   barrier_hex[family][scope - 1]);
         else
            expect_refused("unwitnessed traversal barrier scope", ok, b, len);
      }
   }
   memset(b, 0xa5, sizeof(b));
   len = 0xa5;
   const bool agx2_barrier = agx3_rt_traversal_barrier(
      AGX_ISA_ENCODING_AGX2, AGX3_RT_BARRIER_ENTRY, b, &len);
   expect_refused("AGX2 traversal barrier", agx2_barrier, b, len);

   if (failures) {
      fprintf(stderr, "agx3_rt_core: %u/%u checks failed\n", failures, checks);
      return 1;
   }

   printf("agx3_rt_core: all %u checks passed\n", checks);
   return 0;
}
