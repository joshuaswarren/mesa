/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Disk-cache records of shader parts (hk_part_from_blob()). A record in the
 * stored layout -- agx_shader_info followed by the binary -- loads back
 * unchanged. One whose coefficient binding count or code extents do not fit
 * its binding array or binary is a miss, so the part is rebuilt rather than
 * linked.
 */
#include "hk_cmd_buffer.h"
#include "util/ralloc.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

#define BINARY_SIZE 256

static uint8_t binary[BINARY_SIZE];

/* The record hk_part_disk_put() stores for a part. */
static uint8_t *
record(const struct agx_shader_info *info, size_t *size)
{
   *size = sizeof(*info) + info->binary_size;
   uint8_t *data = malloc(*size);
   CHECK(data);
   memcpy(data, info, sizeof(*info));
   memcpy(data + sizeof(*info), binary, info->binary_size);
   return data;
}

static bool
loads(const struct agx_shader_info *info)
{
   size_t size;
   uint8_t *data = record(info, &size);
   struct agx_shader_part *part = hk_part_from_blob(data, size, NULL);
   free(data);
   if (part)
      free(part->binary);
   ralloc_free(part);
   return part != NULL;
}

int
main(void)
{
   for (unsigned i = 0; i < BINARY_SIZE; ++i)
      binary[i] = (uint8_t)(i * 37 + 11);

   /* A fragment part as the compiler builds one: a preamble, then the main,
    * filling the binary, and two coefficient bindings. */
   struct agx_shader_info valid;
   memset(&valid, 0, sizeof(valid));
   valid.stage = MESA_SHADER_FRAGMENT;
   valid.binary_size = BINARY_SIZE;
   valid.has_preamble = true;
   valid.preamble_offset = 0;
   valid.preamble_size = 64;
   valid.main_offset = 64;
   valid.main_size = BINARY_SIZE - 64;
   valid.varyings.fs.nr_cf = 8;
   valid.varyings.fs.nr_bindings = 2;
   valid.varyings.fs.bindings[0] = (struct agx_cf_binding){
      .cf_base = 0, .slot = VARYING_SLOT_VAR0, .count = 4, .smooth = true,
      .perspective = true,
   };
   valid.varyings.fs.bindings[1] = (struct agx_cf_binding){
      .cf_base = 4, .slot = VARYING_SLOT_VAR1, .count = 4, .smooth = true,
   };

   /* Round trip: the stored record loads back field for field. */
   size_t size;
   uint8_t *data = record(&valid, &size);
   void *ctx = ralloc_context(NULL);
   struct agx_shader_part *part = hk_part_from_blob(data, size, ctx);
   CHECK(part);
   CHECK(memcmp(&part->info, &valid, sizeof(valid)) == 0);
   CHECK(memcmp(part->binary, binary, BINARY_SIZE) == 0);
   free(part->binary);

   /* A record of the wrong length is a miss, as before. */
   CHECK(!hk_part_from_blob(data, size - 1, ctx));
   CHECK(!hk_part_from_blob(data, sizeof(valid) - 1, ctx));
   free(data);

   /* A stored length that does not match the record is a miss, including
    * lengths whose sum with the info header wraps a 32-bit size_t. */
   const uint32_t bad_lengths[] = {
      0, BINARY_SIZE - 1, BINARY_SIZE + 1, UINT32_MAX,
      (uint32_t)(UINT32_MAX - sizeof(valid) + 1 + BINARY_SIZE),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(bad_lengths); ++i) {
      struct agx_shader_info t = valid;
      t.binary_size = bad_lengths[i];
      uint8_t *rec = malloc(sizeof(t) + BINARY_SIZE);
      CHECK(rec);
      memcpy(rec, &t, sizeof(t));
      memcpy(rec + sizeof(t), binary, BINARY_SIZE);
      CHECK(!hk_part_from_blob(rec, sizeof(t) + BINARY_SIZE, ctx));
      free(rec);
   }
   ralloc_free(ctx);

   struct agx_shader_info t;

   /* Binding counts: up to the array's capacity loads. */
   t = valid;
   t.varyings.fs.nr_bindings = AGX_MAX_CF_BINDINGS;
   CHECK(loads(&t));
   const unsigned bad_counts[] = {AGX_MAX_CF_BINDINGS + 1, UINT_MAX};
   for (unsigned i = 0; i < ARRAY_SIZE(bad_counts); ++i) {
      t = valid;
      t.varyings.fs.nr_bindings = bad_counts[i];
      CHECK(!loads(&t));
   }

   /* The main must lie inside the binary. */
   t = valid;
   t.main_offset = BINARY_SIZE;
   t.main_size = 0;
   CHECK(loads(&t));
   t = valid;
   t.main_size = BINARY_SIZE - 64 + 1;
   CHECK(!loads(&t));
   t = valid;
   t.main_offset = BINARY_SIZE + 4;
   t.main_size = 0;
   CHECK(!loads(&t));
   t = valid;
   t.main_size = UINT_MAX - 32; /* main_offset + main_size wraps */
   CHECK(!loads(&t));

   /* So must a preamble, when the part has one. */
   t = valid;
   t.preamble_offset = 200;
   CHECK(!loads(&t));
   t = valid;
   t.preamble_size = UINT_MAX;
   CHECK(!loads(&t));
   t.has_preamble = false;
   CHECK(loads(&t));

   printf("PASS\n");
   return 0;
}
