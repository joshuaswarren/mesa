/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * The G18G re-encoding cache against full re-encodings. The generators'
 * programs for one shader differ only in their uniform table addresses; for
 * each of many addresses, the cache must give exactly the bytes the
 * transcoder writes, whatever size the agx3d LDIMMs take for them, and it
 * must both hit and re-encode on a size it was not made with.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/compiler/agx3_reencode_cache.h"
#include "asahi/lib/agx_ioto_state_core.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #x);                                                          \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static uint64_t
rnd(void)
{
   rng ^= rng << 13;
   rng ^= rng >> 7;
   rng ^= rng << 17;
   return rng;
}

#define CLASSES 6

/* The address of uniform range i in the k-th program of a class: one
 * running through a buffer 64 bytes a program, as a batch's uploads go;
 * then short or zero 32-bit halves and random 40-bit addresses, for which the
 * agx3d LDIMMs take other sizes; then a class picked for each address. */
static uint64_t
address(unsigned class, unsigned k, unsigned i)
{
   if (class == CLASSES - 1)
      class = 1 + rnd() % (CLASSES - 2);

   switch (class) {
   case 0:
      return UINT64_C(0x4000001000) + 0x1000 * i + 64 * k;
   case 1:
      return rnd() & 0xfff0; /* a short low half, a zero high half */
   case 2:
      return (rnd() & 0xff) << 32 | (rnd() & 0xff0); /* both halves short */
   case 3:
      return (rnd() & 0xff) << 32; /* a zero low half */
   default:
      return rnd() & 0xfffffffff0;
   }
}

struct counts {
   unsigned cases, hits, misses, refused;
};

/* One program through the cache and through the transcoder */
static int
check_program(struct agx3_reencode_cache *cache,
              const struct agx3_g17p_graphics_program *program,
              bool api_program, unsigned shader_gprs, struct counts *n)
{
   const struct agx3_transcode_ioto ioto = {
      .api_program = api_program,
      .shader_gprs = shader_gprs,
   };
   struct agx3_g17p_graphics_program full = *program, cached = *program;
   const char *why = NULL;
   unsigned at = 0;
   const bool ok = agx3_reencode_program(&full, &ioto, &why, &at);

   why = NULL;
   enum agx3_reencode_result r =
      agx3_reencode_cached(cache, &cached, &ioto, &why, &at);

   n->cases++;
   CHECK(r != AGX3_REENCODE_UNCACHEABLE);
   if (!ok) {
      CHECK(r == AGX3_REENCODE_REFUSED);
      CHECK(!memcmp(&cached, program, sizeof(cached)));
      n->refused++;
      return 0;
   }
   CHECK(r == AGX3_REENCODE_HIT || r == AGX3_REENCODE_MISS);
   CHECK(cached.length == full.length);
   CHECK(!memcmp(cached.bytes, full.bytes, full.length));
   if (r == AGX3_REENCODE_HIT)
      n->hits++;
   else
      n->misses++;
   return 0;
}

int
main(void)
{
   struct agx3_reencode_cache cache = AGX3_REENCODE_CACHE_INITIALIZER;
   struct counts n[CLASSES] = {0}, all = {0};

   for (unsigned fragment = 0; fragment < 2; ++fragment) {
      for (unsigned range_count = 1; range_count <= 4; ++range_count) {
         struct agx3_ioto_uniform_range ranges[4];
         struct agx3_g17p_graphics_program_desc desc = {
            .program_va = UINT64_C(0x10000100000),
            .shader_va =
               fragment ? UINT64_C(0x100000003c0) : UINT64_C(0x10000000480),
            .shader_binary_size = fragment ? 60 : 128,
            .nr_gprs = 16,
            .compiler_d_valid = true,
            .compiler_d = fragment ? 0 : 32,
            .fragment = fragment,
            .imageblock = AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32,
            .uniform_ranges = ranges,
            .uniform_range_count = range_count,
         };

         for (unsigned i = 0; i < range_count; ++i) {
            ranges[i].start_halfs = 4 * i;
            ranges[i].size_halfs = 4;
         }

         /* 100 programs of each class in turn */
         for (unsigned k = 0; k < 100 * CLASSES; ++k) {
            struct agx3_g17p_graphics_program program;
            struct counts *c = &n[k / 100];

            for (unsigned i = 0; i < range_count; ++i)
               ranges[i].address = address(k / 100, k % 100, i);

            /* A vertex or fragment program, as for a draw */
            CHECK(agx3_g18g_graphics_program_generate(&desc, &program));
            if (check_program(&cache, &program, true, desc.nr_gprs, c))
               return 1;

            /* A batch's background and end-of-tile programs */
            for (unsigned store = 0; fragment && store < 2; ++store) {
               CHECK(agx3_g18g_tile_program_generate(&desc, store, &program));
               if (check_program(&cache, &program, false, desc.nr_gprs, c))
                  return 1;
            }
         }
      }
   }

   for (unsigned c = 0; c < CLASSES; ++c) {
      printf("class %u: %u programs, %u hits, %u re-encoded in full, "
             "%u refused\n",
             c, n[c].cases, n[c].hits, n[c].misses, n[c].refused);
      all.cases += n[c].cases;
      all.hits += n[c].hits;
      all.misses += n[c].misses;
      all.refused += n[c].refused;
   }
   printf("all: %u programs, %u hits, %u re-encoded in full, %u refused\n",
          all.cases, all.hits, all.misses, all.refused);
   /* A running buffer hits; the other classes change sizes, so the cache
    * re-encodes in full far more often than for the first program of each
    * stage and range count (16) */
   CHECK(n[0].hits * 10 >= n[0].cases * 9);
   CHECK(all.hits > 0 && all.misses > 16 * CLASSES);
   agx3_reencode_cache_clear(&cache);
   return 0;
}
