/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Byte-exact replay of the witnessed AGX3 ray-traversal instructions.
 *
 * Every expected byte string below is a pinned witness: the intersect_ray /
 * continue_ray strings are the decoded G16G/G17S corpus vectors in
 * agx3_compose_corpus_vectors.c and the five-kernel corpus ladder bytes that
 * agx3_pack_{intersect,continue}_ray reproduce; the incStk / barrier /
 * update_node / update_bvh strings are the exact fixed witnesses pinned in
 * agx3_pack.c.  If this core ever stops reproducing them it is emitting
 * something Apple never emitted.
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

   /* intersect_ray: the corpus differential pairs, varying only f5. */
   static const struct {
      uint16_t reg;
      const char *hex;
   } b_isect[] = {
      {0, "5f10540000412640c7504c00"},
      {4, "5f10540400412640c7504c00"},
   }, c_isect[] = {
      {0, "27042901a2c1ae02019827000020"},
      {8, "27042901a2c1ae02019827000022"},
   };

   for (unsigned i = 0; i < 2; ++i) {
      expect("AGX3b intersect_ray",
             agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3B, b_isect[i].reg, b,
                                   &len),
             b, len, b_isect[i].hex);
      expect("AGX3c intersect_ray",
             agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3C, c_isect[i].reg, b,
                                   &len),
             b, len, c_isect[i].hex);
   }

   /* Every witnessed intersect_ray site, including the anonymous
    * scheduling-context variants (corpus vectors 263, 1006, 1453). */
   static const char *b_sites[] = {
      "5f10540000412640c7504c00", /* 561 */
      "5f10540400412640c7504c00", /* 470/651 */
      "5f1054000041264047504c00", /* 263: f92=130 */
   };
   static const char *c_sites[] = {
      "27042901a2c1ae02019827000020", /* 1351 */
      "27042901a2c1ae02019827000022", /* 1252 */
      "27042901a2c1ae020198020000a0", /* 1006: f17=4, f28=1 */
      "27042901a2c1ae020198270000a2", /* 1453: f28=1 */
   };

   if (agx3_rt_intersect_ray_num_witnessed_sites(AGX_ISA_ENCODING_AGX3B) != 3 ||
       agx3_rt_intersect_ray_num_witnessed_sites(AGX_ISA_ENCODING_AGX3C) != 4 ||
       agx3_rt_intersect_ray_num_witnessed_sites(AGX_ISA_ENCODING_AGX2) != 0) {
      fprintf(stderr, "witnessed site counts wrong\n");
      failures++;
   }
   checks++;

   for (unsigned i = 0; i < 3; ++i) {
      expect("AGX3b intersect_ray site",
             agx3_rt_intersect_ray_witnessed_site(AGX_ISA_ENCODING_AGX3B, i, b,
                                                  &len),
             b, len, b_sites[i]);
   }
   for (unsigned i = 0; i < 4; ++i) {
      expect("AGX3c intersect_ray site",
             agx3_rt_intersect_ray_witnessed_site(AGX_ISA_ENCODING_AGX3C, i, b,
                                                  &len),
             b, len, c_sites[i]);
   }

   /* continue_ray: the full accepted ladders, byte-identical to the
    * agx3_pack_continue_ray deposit scheme on every value. */
   static const struct {
      uint16_t reg;
      const char *hex;
   } b_cont[] = {
      {0, "5f1154000000"},  {2, "5f1154020000"}, {4, "5f1154040000"},
      {6, "5f1154060000"},  {8, "5f1154080000"}, {10, "5f11540a0000"},
   }, c_cont[] = {
      {0, "2704090022c0af02"},  {2, "2704090022c0af42"},
      {8, "2704090032c0af02"},  {10, "2704090032c0af42"},
      {12, "270409003ac0af02"}, {16, "27040900a2c0af02"},
      {18, "27040900a2c0af42"},
   };

   for (unsigned i = 0; i < 6; ++i) {
      expect("AGX3b continue_ray",
             agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3B, b_cont[i].reg, b,
                                  &len),
             b, len, b_cont[i].hex);
   }
   for (unsigned i = 0; i < 7; ++i) {
      expect("AGX3c continue_ray",
             agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3C, c_cont[i].reg, b,
                                  &len),
             b, len, c_cont[i].hex);
   }

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

   /* The fixed forms: exact agx3_pack.c witnesses. */
   expect("AGX3b incStk", agx3_rt_inc_stk(AGX_ISA_ENCODING_AGX3B, b, &len), b,
          len, "0700541280000000");
   expect("AGX3c incStk", agx3_rt_inc_stk(AGX_ISA_ENCODING_AGX3C, b, &len), b,
          len, "9f04000000000000");
   expect("AGX3b update barrier",
          agx3_rt_update_barrier(AGX_ISA_ENCODING_AGX3B, b, &len), b, len,
          "070454080900");
   expect("AGX3c update barrier",
          agx3_rt_update_barrier(AGX_ISA_ENCODING_AGX3C, b, &len), b, len,
          "170900060000");
   expect("AGX3b update_node",
          agx3_rt_update_node(AGX_ISA_ENCODING_AGX3B, b, &len), b, len,
          "df055400");
   expect("AGX3c update_node",
          agx3_rt_update_node(AGX_ISA_ENCODING_AGX3C, b, &len), b, len,
          "270001002200ad02");
   expect("AGX3c update_bvh",
          agx3_rt_update_bvh(AGX_ISA_ENCODING_AGX3C, b, &len), b, len,
          "270021002280ac42");

   /* The complete expansions, byte-identical to agx3_pack_update_node and
    * agx3_pack_update_bvh_fixed_witness. */
   expect("AGX3b update_node expansion",
          agx3_rt_update_node_expansion(AGX_ISA_ENCODING_AGX3B, b, &len), b,
          len, "070454080900df055400");
   expect("AGX3c update_node expansion",
          agx3_rt_update_node_expansion(AGX_ISA_ENCODING_AGX3C, b, &len), b,
          len, "170900060000270001002200ad02");
   expect("AGX3c update_bvh expansion",
          agx3_rt_update_bvh_expansion(AGX_ISA_ENCODING_AGX3C, b, &len), b,
          len, "170900060000270021002280ac42");

   /* Fail-closed: unwitnessed state registers, cross-encoding ladders, the
    * unwitnessed AGX3b update_bvh, out-of-range sites, and AGX2. */
   expect_refused("intersect_ray b reg 2",
                  agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3B, 2, b, &len), b,
                  len);
   expect_refused("intersect_ray b reg 8",
                  agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3B, 8, b, &len), b,
                  len);
   expect_refused("intersect_ray c reg 4",
                  agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3C, 4, b, &len), b,
                  len);
   expect_refused("intersect_ray on AGX2",
                  agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX2, 0, b, &len), b,
                  len);
   expect_refused("continue_ray b reg 1",
                  agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3B, 1, b, &len), b,
                  len);
   expect_refused("continue_ray b reg 12",
                  agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3B, 12, b, &len), b,
                  len);
   expect_refused("continue_ray c reg 4",
                  agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3C, 4, b, &len), b,
                  len);
   expect_refused("continue_ray c reg 6",
                  agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3C, 6, b, &len), b,
                  len);
   expect_refused("continue_ray on UNKNOWN",
                  agx3_rt_continue_ray(AGX_ISA_ENCODING_UNKNOWN, 0, b, &len),
                  b, len);
   expect_refused("update_bvh on AGX3b",
                  agx3_rt_update_bvh(AGX_ISA_ENCODING_AGX3B, b, &len), b, len);
   expect_refused("update_bvh expansion on AGX3b",
                  agx3_rt_update_bvh_expansion(AGX_ISA_ENCODING_AGX3B, b,
                                               &len),
                  b, len);
   expect_refused("intersect_ray b site 3",
                  agx3_rt_intersect_ray_witnessed_site(AGX_ISA_ENCODING_AGX3B,
                                                       3, b, &len),
                  b, len);
   expect_refused("intersect_ray c site 4",
                  agx3_rt_intersect_ray_witnessed_site(AGX_ISA_ENCODING_AGX3C,
                                                       4, b, &len),
                  b, len);
   expect_refused("intersect_ray site on AGX2",
                  agx3_rt_intersect_ray_witnessed_site(AGX_ISA_ENCODING_AGX2,
                                                       0, b, &len),
                  b, len);
   expect_refused("inc_stk on AGX2",
                  agx3_rt_inc_stk(AGX_ISA_ENCODING_AGX2, b, &len), b, len);
   expect_refused("update barrier on AGX2",
                  agx3_rt_update_barrier(AGX_ISA_ENCODING_AGX2, b, &len), b,
                  len);
   expect_refused("update_node on UNKNOWN",
                  agx3_rt_update_node(AGX_ISA_ENCODING_UNKNOWN, b, &len), b,
                  len);

   /*
    * The incStk operand differential, measured on 2026-09-10 from Apple's own
    * compiler.  Reproduce with the harness in README-metal-witness.md:
    *
    *   metal -mmacosx-version-min=26.0 -c <src>.metal -o <src>.air
    *   metallib <src>.air -o <src>.metallib
    *   metal-nt -arch applegpu_g17p -N <src>.mtlp-json \
    *            -platform_version macos 26.0 26.0 <src>.metallib -o <src>.nat
    *
    * over two sources: five ray-query kernels and a 1/2/3/4-concurrent-query
    * family.  Linear-scanning both __text images yields exactly these 24
    * (f2, f24) pairs.  Each comment names one site the bytes were read at.
    *
    * f2 and f24 are independent: 46 appears with both f24=5 and f24=8, and 62
    * with both 352 and 520.  No hardware was involved; these are compiler
    * outputs.
    */
   static const struct {
      uint16_t f2, f24;
      const char *hex;
   } g17p_inc_stk[] = {
      {  0,    4, "0700540080000000"}, /* rq_bary+0x430 */
      {  0,  184, "0700540000170000"}, /* nq2+0xe0     */
      {  4,   16, "0700540200020000"}, /* nq2+0xc8     */
      { 10,    8, "0700540500010000"}, /* nq2+0xbc     */
      { 12,   16, "0700540600020000"}, /* nq3+0xc8     */
      { 18,    4, "0700540980000000"}, /* nq4+0xd4     */
      { 20,  352, "0700540a002c0000"}, /* nq3+0xf8     */
      { 22,  184, "0700540b00170000"}, /* nq4+0x134    */
      { 24,  184, "0700540c00170000"}, /* nq3+0xe0     */
      { 26,   16, "0700540d00020000"}, /* nq4+0xcc     */
      { 28,  520, "0700540e00410000"}, /* nq4+0x1e4    */
      { 32,  352, "07005410002c0000"}, /* nq4+0x1dc    */
      { 40,    4, "0700541480000000"}, /* rq_prim+0x1d0 */
      { 42,    4, "0700541580000000"}, /* rq_inst+0x1f0 */
      { 46,    5, "07005417a0000000"}, /* nq4+0x13c    */
      { 46,    8, "0700541700010000"}, /* nq3+0xbc     */
      { 60,  688, "0700541e00560000"}, /* nq4+0x704    */
      { 62,  352, "0700541f002c0000"}, /* nq2+0x384    */
      { 62,  520, "0700541f00410000"}, /* nq3+0x55a    */
      { 66,    7, "07005421e0000000"}, /* nq4+0x206    */
      { 74,    6, "07005425c0000000"}, /* nq4+0x1ec    */
      { 92,    4, "0700542e80000000"}, /* nq2+0xd0     */
      { 96,    5, "07005430a0000000"}, /* nq2+0xe8     */
      {100,    6, "07005432c0000000"}, /* nq3+0x100    */
   };

   for (unsigned i = 0; i < sizeof(g17p_inc_stk) / sizeof(g17p_inc_stk[0]);
        ++i) {
      char what[64];
      snprintf(what, sizeof(what), "G17P incStk f2=%u f24=%u",
               g17p_inc_stk[i].f2, g17p_inc_stk[i].f24);
      expect(what,
             agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B,
                                      g17p_inc_stk[i].f2,
                                      g17p_inc_stk[i].f24, b, &len),
             b, len, g17p_inc_stk[i].hex);
   }

   /* The fixed witness must remain reachable through the operand codec. */
   expect("incStk operands reproduce the fixed AGX3b witness",
          agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B, 36, 4, b, &len), b,
          len, "0700541280000000");

   /* Fail-closed boundaries: no AGX3c operand witness, no odd f2, no
    * out-of-range f24. */
   expect_refused("incStk operands on AGX3c",
                  agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3C, 40, 4, b,
                                           &len),
                  b, len);
   expect_refused("incStk operands on AGX2",
                  agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX2, 40, 4, b,
                                           &len),
                  b, len);
   expect_refused("incStk odd f2",
                  agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B, 41, 4, b,
                                           &len),
                  b, len);
   expect_refused("incStk f2 out of range",
                  agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B, 512, 4, b,
                                           &len),
                  b, len);
   expect_refused("incStk f24 out of range",
                  agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B, 40, 8192, b,
                                           &len),
                  b, len);

   /*
    * The ray-query intersect_ray differential, measured 2026-09-10 from
    * Apple's compiler.  Two and three simultaneously live named queries,
    * built for applegpu_g17p with metal-nt exactly as README-metal-witness.md
    * describes, then linear-disassembled.  f77 takes one distinct value per
    * query while the SCS ray staging offsets stay 0 and 16 throughout; f5
    * stays within {0, 4} regardless of how many queries are live.
    */
   static const struct {
      uint16_t site, per_query, context;
      const char *hex;
      const char *note;
   } g17p_query_sites[] = {
      {0,  88, 124, "5f10540000412680854fcc00", "two queries, q1"},
      {4,  88, 124, "5f10540400412680854fcc00", "two queries, q1"},
      {0,  14, 124, "5f105400004126e0804fcc00", "two queries, q2"},
      {4,  14, 124, "5f105404004126e0804fcc00", "two queries, q2"},
      {0, 106, 122, "5f105400004126a0464fcc00", "three queries, q1"},
      {4, 106, 122, "5f105404004126a0464fcc00", "three queries, q1"},
      {0,  32, 122, "5f10540000412600424fcc00", "three queries, q2"},
      {4,  32, 122, "5f10540400412600424fcc00", "three queries, q2"},
      {0,   2, 122, "5f10540000412620404fcc00", "three queries, q3"},
      {4,   2, 122, "5f10540400412620404fcc00", "three queries, q3"},
   };

   for (unsigned i = 0;
        i < sizeof(g17p_query_sites) / sizeof(g17p_query_sites[0]); ++i) {
      char what[96];
      snprintf(what, sizeof(what), "G17P intersect_ray %s f5=%u f77=%u",
               g17p_query_sites[i].note, g17p_query_sites[i].site,
               g17p_query_sites[i].per_query);
      expect(what,
             agx3_rt_intersect_ray_query(
                AGX_ISA_ENCODING_AGX3B, g17p_query_sites[i].site,
                g17p_query_sites[i].per_query, g17p_query_sites[i].context, b,
                &len),
             b, len, g17p_query_sites[i].hex);
   }

   /* Measured bounds: f5 is 13 bits, f77 is refused at 192. */
   expect_refused("intersect_ray query f5 above 13 bits",
                  agx3_rt_intersect_ray_query(AGX_ISA_ENCODING_AGX3B, 8192, 2,
                                              122, b, &len),
                  b, len);
   expect_refused("intersect_ray query f77 = 192",
                  agx3_rt_intersect_ray_query(AGX_ISA_ENCODING_AGX3B, 0, 192,
                                              122, b, &len),
                  b, len);
   expect_refused("intersect_ray query on AGX3c",
                  agx3_rt_intersect_ray_query(AGX_ISA_ENCODING_AGX3C, 0, 2,
                                              122, b, &len),
                  b, len);
   checks++;
   if (agx3_rt_intersect_ray_query(AGX_ISA_ENCODING_AGX3B, 0, 2, 122, NULL,
                                   &len) ||
       agx3_rt_intersect_ray_query(AGX_ISA_ENCODING_AGX3B, 0, 2, 122, b,
                                   NULL)) {
      fprintf(stderr, "intersect_ray query accepted a NULL argument\n");
      failures++;
   }

   /* NULL-argument hardening. */
   checks++;
   if (agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B, 40, 4, NULL, &len) ||
       agx3_rt_inc_stk_operands(AGX_ISA_ENCODING_AGX3B, 40, 4, b, NULL)) {
      fprintf(stderr, "incStk operands accepted a NULL argument\n");
      failures++;
   }

   checks++;
   if (agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3B, 0, NULL, &len) ||
       agx3_rt_intersect_ray(AGX_ISA_ENCODING_AGX3B, 0, b, NULL) ||
       agx3_rt_continue_ray(AGX_ISA_ENCODING_AGX3B, 0, NULL, &len) ||
       agx3_rt_update_node_expansion(AGX_ISA_ENCODING_AGX3B, b, NULL)) {
      fprintf(stderr, "NULL argument accepted\n");
      failures++;
   }

   if (failures) {
      fprintf(stderr, "agx3_rt_core: %u/%u checks failed\n", failures, checks);
      return 1;
   }

   printf("agx3_rt_core: all %u checks passed\n", checks);
   return 0;
}
