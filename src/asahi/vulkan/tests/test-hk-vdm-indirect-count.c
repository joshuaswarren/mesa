/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/libagx/libagx_dgc.h"

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
         return 1;                                                             \
      }                                                                        \
      checks++;                                                                \
   } while (0)

static unsigned checks = 0;

static int
test_span_coalescing_and_lookup(void)
{
   /* Simulate 16,384 draws where draw 0 includes state flush (420 B), draws
    * 1..N-1 are 28 B each, and chunks wrap every 2,000 draws with both higher
    * and lower chunk base addresses. */
   enum { MAX_DRAWS = 16384 };
   static const uint64_t chunk_bases[9] = {
      0x100000000ULL, 0x100020000ULL, 0x0f0000000ULL,
      0x200000000ULL, 0x180000000ULL, 0x200040000ULL,
      0x200080000ULL, 0x0c0000000ULL, 0x300000000ULL,
   };

   uint64_t *expected = malloc(sizeof(uint64_t) * MAX_DRAWS);
   CHECK(expected != NULL);

   struct libagx_vdm_draw_span spans[32];
   uint32_t num_spans = 0;

   uint64_t cur = chunk_bases[0];
   for (uint32_t i = 0; i < MAX_DRAWS; ++i) {
      if (i > 0 && (i % 2000) == 0)
         cur = chunk_bases[i / 2000];

      expected[i] = cur;
      CHECK(libagx_vdm_span_append(spans, &num_spans, ARRAY_SIZE(spans), i,
                                   cur));
      cur += (i == 0) ? 420u : 28u;
   }

   /* Draw 0 + draw 1 coalesce into span 0, then one span per chunk. */
   CHECK(num_spans <= 12);

   for (uint32_t i = 0; i < MAX_DRAWS; ++i) {
      CHECK(libagx_vdm_span_find_addr(spans, num_spans, i) == expected[i]);
   }
   CHECK(libagx_vdm_span_find_addr(spans, num_spans, MAX_DRAWS) == 0);
   CHECK(libagx_vdm_span_find_addr(spans, num_spans, UINT32_MAX) == 0);

   /* Capacity limit returns false without corrupting num_spans. */
   uint32_t small_n = 0;
   struct libagx_vdm_draw_span one_span[1];
   CHECK(libagx_vdm_span_append(one_span, &small_n, 1, 0, 0x2000));
   CHECK(libagx_vdm_span_append(one_span, &small_n, 1, 1, 0x2020));
   CHECK(!libagx_vdm_span_append(one_span, &small_n, 1, 2, 0x1000));
   CHECK(small_n == 1);

   free(expected);
   return 0;
}

static int
test_reusable_patch_state_machine(void)
{
   /* Simulate 3 disjoint chunks holding 12 draws (each 16 B = 4 uint32_t) and
    * verify every boundary count: 0, 1, mid, chunk boundary, max-1, max, >max,
    * and back to 0/1 across repeated submissions of the same command buffer. */
   enum { MAX_DRAWS = 12, WORDS_PER_DRAW = 4 };
   uint32_t chunk0[4 * WORDS_PER_DRAW];
   uint32_t chunk1[4 * WORDS_PER_DRAW];
   uint32_t chunk2[4 * WORDS_PER_DRAW];
   uint32_t orig[MAX_DRAWS][WORDS_PER_DRAW];
   uint32_t *draw_ptrs[MAX_DRAWS];

   struct libagx_vdm_draw_span spans[8];
   uint32_t num_spans = 0;

   for (uint32_t i = 0; i < MAX_DRAWS; ++i) {
      uint32_t *base = (i < 4) ? chunk0 : (i < 8) ? chunk1 : chunk2;
      uint32_t *p = base + (i % 4) * WORDS_PER_DRAW;
      draw_ptrs[i] = p;
      for (uint32_t w = 0; w < WORDS_PER_DRAW; ++w) {
         uint32_t val = 0x40000000u | (i << 8) | w;
         p[w] = val;
         orig[i][w] = val;
      }
      CHECK(libagx_vdm_span_append(spans, &num_spans, ARRAY_SIZE(spans), i,
                                   (uintptr_t)p));
   }

   struct agx_vdm_stream_link_core link;
   const uint64_t end_target = UINT64_C(0x123456780);
   CHECK(agx_vdm_stream_link_core_pack(AGX_CHIP_G17P, end_target, &link));
   uint32_t link_w[2];
   memcpy(link_w, link.bytes, sizeof(link_w));

   struct libagx_vdm_truncate_state state = {0};

   /* Sequence exercising every boundary and repeated restore/repatch cycles. */
   static const uint32_t counts[] = {
      0, 1, 3, 4, 7, 8, 11, 12, 500, 0, 12, 1, 9, 1000000, 0,
   };

   for (unsigned step = 0; step < ARRAY_SIZE(counts); ++step) {
      uint32_t cnt = counts[step];
      libagx_vdm_patch_indirect_count_impl(&state, spans, &cnt, num_spans,
                                           MAX_DRAWS, link_w[0], link_w[1]);

      if (cnt >= MAX_DRAWS) {
         CHECK(state.patched_addr == 0);
      } else {
         CHECK(state.patched_addr == (uintptr_t)draw_ptrs[cnt]);
         CHECK(state.orig_w0 == orig[cnt][0]);
         CHECK(state.orig_w1 == orig[cnt][1]);
      }

      /* Verify every draw slot: patched slot holds the exact VDM_STREAM_LINK
       * (with trailing words intact); every other slot holds its original
       * words bit-for-bit. */
      for (uint32_t i = 0; i < MAX_DRAWS; ++i) {
         if (i == cnt && cnt < MAX_DRAWS) {
            CHECK(draw_ptrs[i][0] == link_w[0]);
            CHECK(draw_ptrs[i][1] == link_w[1]);
            CHECK(draw_ptrs[i][2] == orig[i][2]);
            CHECK(draw_ptrs[i][3] == orig[i][3]);

            uint64_t decoded = 0;
            CHECK(agx_vdm_stream_link_core_unpack(
               AGX_CHIP_G17P, draw_ptrs[i], AGX_VDM_STREAM_LINK_CORE_LENGTH,
               &decoded));
            CHECK(decoded == end_target);
         } else {
            for (uint32_t w = 0; w < WORDS_PER_DRAW; ++w)
               CHECK(draw_ptrs[i][w] == orig[i][w]);
         }
      }
   }

   return 0;
}

static int
test_cross_chip_link_roundtrip(void)
{
   static const enum agx_chip supported_chips[] = {
      AGX_CHIP_G15G,
      AGX_CHIP_G16G,
      AGX_CHIP_G17P,
      AGX_CHIP_G17G,
   };
   const uint64_t target = UINT64_C(0x3a4b5c6d8);

   for (unsigned i = 0; i < ARRAY_SIZE(supported_chips); ++i) {
      struct agx_vdm_stream_link_core link;
      CHECK(agx_vdm_stream_link_core_supported(supported_chips[i]));
      CHECK(agx_vdm_stream_link_core_pack(supported_chips[i], target, &link));
      CHECK(link.length == AGX_VDM_STREAM_LINK_CORE_LENGTH);

      uint64_t unpacked = 0;
      CHECK(agx_vdm_stream_link_core_unpack(supported_chips[i], link.bytes,
                                            link.length, &unpacked));
      CHECK(unpacked == target);
   }

   /* AGX2 witnessed chips use the record-table capability and emit an 8-byte
    * non-returning VDM_STREAM_LINK; unverified chips must fail closed. */
   static const enum agx_chip agx2_chips[] = {
      AGX_CHIP_G13G,
      AGX_CHIP_G13X,
      AGX_CHIP_G14G,
      AGX_CHIP_G14X,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(agx2_chips); ++i) {
      CHECK(agx_chip_vdm_stream_link_supported(agx2_chips[i]));
      CHECK(!agx_vdm_stream_link_core_supported(agx2_chips[i]));
   }
   uint32_t agx2_link[2] = {0, 0};
   uint32_t *agx2_end = agx_cs_jump(agx2_link, target, true);
   CHECK(agx2_end == agx2_link + 2);

   static const enum agx_chip closed_chips[] = {
      AGX_CHIP_G15P,
      AGX_CHIP_G16P,
      AGX_CHIP_G17A,
      AGX_CHIP_G18G,
      AGX_CHIP_UNKNOWN,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(closed_chips); ++i) {
      CHECK(!agx_chip_vdm_stream_link_supported(closed_chips[i]));
      CHECK(!agx_vdm_stream_link_core_supported(closed_chips[i]));
   }
   return 0;
}

static int
test_stream_split_runtime_guard(void)
{
   uint8_t cs_a_storage = 0;
   uint8_t cs_b_storage = 0;
   uint8_t buf[64];

   /* Same stream, >= 8 bytes written in same chunk: valid. */
   CHECK(libagx_vdm_draw_site_valid(&cs_a_storage, &cs_a_storage, 0x1000,
                                    0x1000, &buf[0], &buf[8]));
   CHECK(libagx_vdm_draw_site_valid(&cs_a_storage, &cs_a_storage, 0x1000,
                                    0x1000, &buf[0], &buf[28]));

   /* Same stream, draw wrapped onto a new chunk: valid. */
   CHECK(libagx_vdm_draw_site_valid(&cs_a_storage, &cs_a_storage, 0x1000,
                                    0x2000, &buf[32], &buf[4]));

   /* Same stream, < 8 bytes written in same chunk (or reversed): rejected. */
   CHECK(!libagx_vdm_draw_site_valid(&cs_a_storage, &cs_a_storage, 0x1000,
                                     0x1000, &buf[0], &buf[0]));
   CHECK(!libagx_vdm_draw_site_valid(&cs_a_storage, &cs_a_storage, 0x1000,
                                     0x1000, &buf[0], &buf[4]));
   CHECK(!libagx_vdm_draw_site_valid(&cs_a_storage, &cs_a_storage, 0x1000,
                                     0x1000, &buf[16], &buf[8]));

   /* Simulate a mid-loop graphics stream split (hk_flush_gfx_state ended cs_a
    * and opened cs_b on draw 3 of 8): the runtime guard must trip immediately
    * and prevent emitting a cross-stream VDM_STREAM_LINK. */
   struct libagx_vdm_draw_span spans[8];
   uint32_t num_spans = 0;
   bool aborted = false;
   uint32_t recorded = 0;
   for (uint32_t draw_id = 0; draw_id < 8; ++draw_id) {
      const void *cs_after = (draw_id == 3) ? &cs_b_storage : &cs_a_storage;
      const uint64_t chunk_after = (draw_id == 3) ? 0x2000 : 0x1000;
      const uint8_t *cur_before = &buf[draw_id * 8];
      const uint8_t *cur_after = &buf[(draw_id + 1) * 8];
      if (!libagx_vdm_draw_site_valid(&cs_a_storage, cs_after, 0x1000,
                                      chunk_after, cur_before, cur_after)) {
         aborted = true;
         break;
      }
      CHECK(libagx_vdm_span_append(spans, &num_spans, ARRAY_SIZE(spans),
                                   draw_id, 0x1000 + draw_id * 8));
      recorded++;
   }
   CHECK(aborted);
   CHECK(recorded == 3);
   CHECK(libagx_vdm_span_find_addr(spans, num_spans, 3) == 0);
   return 0;
}

static int
test_template_word_patch_equivalence(void)
{
   static const enum agx_chip chips[] = {
      AGX_CHIP_G15G,
      AGX_CHIP_G16G,
      AGX_CHIP_G17P,
      AGX_CHIP_G17G,
   };
   static const uint64_t addrs[] = {
      UINT64_C(0x100000004),
      UINT64_C(0x1fffffffc),
      UINT64_C(0x200000000),
      UINT64_C(0x7fff12345678),
      UINT64_C(0xffffffffffec),
   };
   const uint64_t ib = UINT64_C(0x5a0012340080);

   for (unsigned ci = 0; ci < ARRAY_SIZE(chips); ++ci) {
      for (unsigned indexed = 0; indexed < 2; ++indexed) {
         struct agx_vdm_index_list_indirect_words tmpl;
         CHECK(agx_vdm_index_list_indirect_pack_words(
            chips[ci], (uint8_t)AGX_PRIMITIVE_TRIANGLES, indexed != 0,
            indexed != 0, (uint8_t)AGX_INDEX_SIZE_U32, indexed ? ib : 0,
            addrs[0], indexed ? 0xffffffffu : 0u, indexed ? 0x12345u : 0u,
            &tmpl));
         const unsigned hi_idx = indexed ? 4u : 1u;
         const unsigned lo_idx = indexed ? 5u : 2u;

         for (unsigned ai = 0; ai < ARRAY_SIZE(addrs); ++ai) {
            uint32_t patched[ARRAY_SIZE(tmpl.words)];
            memcpy(patched, tmpl.words, tmpl.count * sizeof(uint32_t));
            patched[hi_idx] = (uint32_t)((addrs[ai] >> 32) & 0xffffu);
            patched[lo_idx] = (uint32_t)addrs[ai];

            struct agx_vdm_index_list_indirect_words ref;
            CHECK(agx_vdm_index_list_indirect_pack_words(
               chips[ci], (uint8_t)AGX_PRIMITIVE_TRIANGLES, indexed != 0,
               indexed != 0, (uint8_t)AGX_INDEX_SIZE_U32, indexed ? ib : 0,
               addrs[ai], indexed ? 0xffffffffu : 0u, indexed ? 0x12345u : 0u,
               &ref));
            CHECK(ref.count == tmpl.count);
            CHECK(memcmp(patched, ref.words, ref.count * sizeof(uint32_t)) ==
                  0);
         }
      }
   }
   return 0;
}

int
main(void)
{
   if (test_span_coalescing_and_lookup() ||
       test_reusable_patch_state_machine() ||
       test_cross_chip_link_roundtrip() ||
       test_stream_split_runtime_guard() ||
       test_template_word_patch_equivalence())
      return 1;

   printf("hk_vdm_indirect_count: %u checks passed\n", checks);
   return 0;
}
