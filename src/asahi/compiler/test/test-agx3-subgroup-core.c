/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx3_subgroup_core.h"
#include "util/macros.h"

#include <stdio.h>
#include <string.h>

#define CHECK(COND)                                                            \
   do {                                                                        \
      if (!(COND)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #COND);       \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static bool
bytes_equal(const uint8_t *actual, uint8_t length, const uint8_t *expected,
            size_t expected_length)
{
   if (length == expected_length &&
       memcmp(actual, expected, expected_length) == 0)
      return true;

   fprintf(stderr, "length %u expected %zu; actual ", length, expected_length);
   for (unsigned i = 0; i < length; ++i)
      fprintf(stderr, "%02x", actual[i]);
   fprintf(stderr, " expected ");
   for (unsigned i = 0; i < expected_length; ++i)
      fprintf(stderr, "%02x", expected[i]);
   fprintf(stderr, "\n");
   return false;
}

static int
test_shuffle_vectors(void)
{
   static const uint8_t reg_b[] = {
      0x47, 0x04, 0x54, 0x04, 0x03, 0x00, 0x08, 0x2c, 0x1a, 0x00,
   };
   static const uint8_t reg_c[] = {
      0x27, 0x80, 0x01, 0x1a, 0x28, 0x00, 0xa8, 0x0a, 0x08, 0x81,
   };
   static const uint8_t uniform_b[] = {
      0x47, 0x04, 0x54, 0x00, 0x02, 0x00, 0x10, 0x2c, 0x0c, 0x00,
   };
   static const uint8_t uniform_c[] = {
      0x27, 0x00, 0x21, 0x1a, 0x20, 0x00, 0xa8, 0x02, 0x18, 0x02,
   };
   static const uint8_t quad_broadcast_wait6[4][10] = {
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x00, 0x2c, 0x02, 0x00},
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00},
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00},
      {0x47, 0x00, 0x56, 0x00, 0x02, 0x0c, 0x06, 0x2c, 0x02, 0x00},
   };
   static const uint8_t quad_xor_lane1_live[] = {
      0xc7, 0x00, 0x54, 0x02, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00,
   };
   static const uint8_t quad_xor_lane1_cached[] = {
      0xc7, 0x00, 0x54, 0x02, 0x03, 0x0c, 0x02, 0x2c, 0x02, 0x00,
   };
   static const uint8_t quad_xor_lane3_final[] = {
      0xc7, 0x00, 0x54, 0x06, 0x02, 0x0c, 0x06, 0x2c, 0x04, 0x00,
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;

   /* G15G and G16G independently produce this AGX3b word, whose destination
    * and source stay cached while the lane is read for the last time. */
   struct agx3_subgroup_shuffle_desc d = {
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
      .dest = 4,
      .src = 0,
      .lane = 4,
      .dest_hint = true,
      .src_cache = true,
      .lane_discard = true,
   };
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, reg_b, sizeof(reg_b)));
   /* G15G and G16G independently produce this same AGX3b word. */
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, reg_b, sizeof(reg_b)));
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, reg_c, sizeof(reg_c)));

   d.dest = d.src = 0;
   d.lane = 8;
   d.lane_kind = AGX3_SUBGROUP_LANE_UNIFORM;
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uniform_b, sizeof(uniform_b)));
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, uniform_c, sizeof(uniform_c)));

   d = (struct agx3_subgroup_shuffle_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .src = 6,
      .lane = 2,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   for (unsigned lane = 0; lane < 4; ++lane) {
      d.lane = lane;
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                         &length));
      CHECK(bytes_equal(out, length, quad_broadcast_wait6[lane],
                        sizeof(quad_broadcast_wait6[lane])));
   }

   d.op = AGX3_SUBGROUP_SHUFFLE_XOR;
   d.dest = 2;
   d.lane = 1;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   CHECK(bytes_equal(out, length, quad_xor_lane1_live,
                     sizeof(quad_xor_lane1_live)));

   /* Fresh Apple G17P quad output retains f11 when its result remains live. */
   d.dest_hint = true;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, quad_xor_lane1_cached,
                     sizeof(quad_xor_lane1_cached)));
   d.dest_hint = false;

   d.dest = 6;
   d.lane = 3;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   CHECK(bytes_equal(out, length, quad_xor_lane3_final,
                     sizeof(quad_xor_lane3_final)));

   d.lane = 8;
   for (unsigned quad = 0; quad < 2; ++quad) {
      d.quad = quad;
      for (unsigned op = 0; op < AGX3_SUBGROUP_SHUFFLE_NUM; ++op) {
         d.op = op;
         for (unsigned kind = AGX3_SUBGROUP_LANE_REGISTER;
              kind <= AGX3_SUBGROUP_LANE_UNIFORM; ++kind) {
            d.lane_kind = kind;
            CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                               &length));
            CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C, &d, out,
                                               &length));
         }
      }
   }

   /*
    * Apple's own applegpu_g17p words for the immediate-lane family, one
    * kernel per lane.
    * The lane is f29 and nothing else moves with it, on all six mnemonics --
    * which is why `subgroupBroadcast(v, 31)`, the read that ends every
    * subgroup scan, needs no materialising mov any more.
    */
   static const struct {
      bool quad;
      enum agx3_subgroup_shuffle_op op;
      uint16_t dest, src, lane;
      bool src_live, dest_hint;
      uint8_t length;
      uint8_t bytes[10];
   } immediate[] = {
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 0, false, false, 10,
       {0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x00, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 31, false, false, 10,
       {0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x3e, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 63, false, false, 10,
       {0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x7e, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_XOR, 0, 0, 63, false, false, 10,
       {0xc7, 0x04, 0x56, 0x00, 0x02, 0x00, 0x7e, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_UP, 0, 0, 0, false, false, 10,
       {0x47, 0x05, 0x56, 0x00, 0x02, 0x00, 0x00, 0x2c, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_UP, 4, 0, 31, true, true, 10,
       {0x47, 0x05, 0x56, 0x04, 0x03, 0x00, 0x3e, 0x2c, 0x02, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_DOWN, 4, 0, 31, true, true, 10,
       {0xc7, 0x05, 0x56, 0x04, 0x03, 0x00, 0x3e, 0x2c, 0x02, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 0, 0, 3, false, false, 10,
       {0x47, 0x00, 0x56, 0x00, 0x02, 0x00, 0x06, 0x2c, 0x04, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_XOR, 0, 0, 3, false, false, 10,
       {0xc7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x06, 0x2c, 0x04, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_UP, 4, 0, 3, true, true, 10,
       {0x47, 0x01, 0x56, 0x04, 0x03, 0x00, 0x06, 0x2c, 0x02, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_DOWN, 4, 0, 3, true, true, 10,
       {0xc7, 0x01, 0x56, 0x04, 0x03, 0x00, 0x06, 0x2c, 0x02, 0x00}},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(immediate); ++i) {
      const struct agx3_subgroup_shuffle_desc w = {
         .quad = immediate[i].quad,
         .op = immediate[i].op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = immediate[i].dest,
         .src = immediate[i].src,
         .lane = immediate[i].lane,
         .wait_mask = 1u << 5,
         .dest_hint = immediate[i].dest_hint,
         .src_live = immediate[i].src_live,
      };
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &w, out,
                                         &length));
      CHECK(bytes_equal(out, length, immediate[i].bytes,
                        immediate[i].length));
   }
   return 0;
}

static int
test_register_lane_hint_vectors(void)
{
   /* metal-nt -arch applegpu_g17p, every SIMD shuffle from a register lane,
    * with the data and lane read again or not; sources on slot 1. */
   static const struct {
      enum agx3_subgroup_shuffle_op op;
      bool half;
      uint16_t dest, src, lane, wait_mask;
      bool dest_hint, src_cache, src_discard, lane_cache, lane_discard;
      uint8_t bytes[10];
   } vec[] = {
      {AGX3_SUBGROUP_SHUFFLE_BROADCAST, false, 0, 0, 4, 0x1,
       false, false, true, false, true,
       {0x47, 0x14, 0x54, 0x00, 0x02, 0x00, 0x08, 0x2c, 0x1c, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_BROADCAST, false, 8, 0, 3, 0x1,
       false, false, false, false, true,
       {0x47, 0x14, 0x54, 0x08, 0x02, 0x00, 0x06, 0x2c, 0x18, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_BROADCAST, false, 4, 2, 10, 0x1,
       false, false, true, true, false,
       {0x47, 0x14, 0x54, 0x04, 0x02, 0x04, 0x14, 0x2e, 0x14, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_BROADCAST, false, 0, 0, 10, 0x2,
       false, false, true, false, true,
       {0x47, 0x24, 0x54, 0x00, 0x02, 0x00, 0x14, 0x2c, 0x1c, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_BROADCAST, false, 6, 0, 9, 0x0,
       true, true, false, true, false,
       {0x47, 0x04, 0x54, 0x06, 0x03, 0x00, 0x12, 0x2e, 0x12, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_BROADCAST, false, 0, 0, 9, 0x0,
       true, false, true, true, false,
       {0x47, 0x04, 0x54, 0x00, 0x03, 0x00, 0x12, 0x2e, 0x14, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_XOR, false, 0, 0, 4, 0x1,
       false, false, true, false, true,
       {0xc7, 0x14, 0x54, 0x00, 0x02, 0x00, 0x08, 0x2c, 0x1c, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_XOR, false, 4, 0, 2, 0x1,
       true, false, true, true, false,
       {0xc7, 0x14, 0x54, 0x04, 0x03, 0x00, 0x04, 0x2e, 0x14, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_UP, false, 4, 0, 8, 0x1,
       true, true, false, true, false,
       {0x47, 0x15, 0x54, 0x04, 0x03, 0x00, 0x10, 0x2e, 0x12, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_DOWN, false, 2, 0, 8, 0x1,
       true, true, false, true, false,
       {0xc7, 0x15, 0x54, 0x02, 0x03, 0x00, 0x10, 0x2e, 0x12, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_XOR, true, 2, 3, 2, 0x1,
       false, false, true, false, true,
       {0xc7, 0x14, 0x54, 0x02, 0x02, 0x06, 0x04, 0x20, 0x1c, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_XOR, true, 5, 1, 0, 0x1,
       true, false, true, true, false,
       {0xc7, 0x14, 0x54, 0x05, 0x03, 0x02, 0x00, 0x22, 0x14, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_UP, true, 4, 3, 2, 0x1,
       true, true, false, true, false,
       {0x47, 0x15, 0x54, 0x04, 0x03, 0x06, 0x04, 0x22, 0x12, 0x00}},
      {AGX3_SUBGROUP_SHUFFLE_DOWN, true, 4, 1, 0, 0x1,
       true, true, false, true, false,
       {0xc7, 0x15, 0x54, 0x04, 0x03, 0x02, 0x00, 0x22, 0x12, 0x00}},
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(vec); ++i) {
      const struct agx3_subgroup_shuffle_desc d = {
         .op = vec[i].op,
         .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
         .dest = vec[i].dest,
         .src = vec[i].src,
         .lane = vec[i].lane,
         .wait_mask = vec[i].wait_mask,
         .dest_hint = vec[i].dest_hint,
         .half = vec[i].half,
         .src_cache = vec[i].src_cache,
         .src_discard = vec[i].src_discard,
         .lane_cache = vec[i].lane_cache,
         .lane_discard = vec[i].lane_discard,
      };
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                         &length));
      CHECK(bytes_equal(out, length, vec[i].bytes, sizeof(vec[i].bytes)));
   }
   return 0;
}

static int
test_half_broadcast_vectors(void)
{
   /* metal-nt -arch applegpu_g17p, simd_broadcast() of 16-bit data. */
   static const struct {
      uint16_t dest, src, lane, wait_mask;
      bool dest_hint, src_cache, src_discard, lane_cache, lane_discard;
      uint8_t bytes[10];
   } reg[] = {
      {2, 3, 2, 0x1, false, false, true, false, true,
       {0x47, 0x14, 0x54, 0x02, 0x02, 0x06, 0x04, 0x20, 0x1c, 0x00}},
      {7, 8, 6, 0x1, false, false, false, false, false,
       {0x47, 0x14, 0x54, 0x07, 0x02, 0x10, 0x0c, 0x20, 0x10, 0x00}},
      {5, 0, 4, 0x1, false, false, true, false, false,
       {0x47, 0x14, 0x54, 0x05, 0x02, 0x00, 0x08, 0x20, 0x14, 0x00}},
      {5, 4, 0, 0x1, false, false, false, false, true,
       {0x47, 0x14, 0x54, 0x05, 0x02, 0x08, 0x00, 0x20, 0x18, 0x00}},
      {6, 3, 5, 0x1, true, true, false, false, true,
       {0x47, 0x14, 0x54, 0x06, 0x03, 0x06, 0x0a, 0x20, 0x1a, 0x00}},
      {7, 3, 4, 0x0, true, true, false, true, false,
       {0x47, 0x04, 0x54, 0x07, 0x03, 0x06, 0x08, 0x22, 0x12, 0x00}},
   };
   static const uint8_t immediate_lane5[] = {0x47, 0x04, 0x56, 0x02, 0x02,
                                             0x04, 0x0a, 0x20, 0x04, 0x00};
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(reg); ++i) {
      const struct agx3_subgroup_shuffle_desc d = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
         .dest = reg[i].dest,
         .src = reg[i].src,
         .lane = reg[i].lane,
         .wait_mask = reg[i].wait_mask,
         .dest_hint = reg[i].dest_hint,
         .half = true,
         .src_cache = reg[i].src_cache,
         .src_discard = reg[i].src_discard,
         .lane_cache = reg[i].lane_cache,
         .lane_discard = reg[i].lane_discard,
      };
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                         &length));
      CHECK(bytes_equal(out, length, reg[i].bytes, sizeof(reg[i].bytes)));
   }

   struct agx3_subgroup_shuffle_desc d = {
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest = 2,
      .src = 2,
      .lane = 5,
      .wait_mask = 1u << 5,
      .half = true,
   };
   CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, immediate_lane5, sizeof(immediate_lane5)));

   /* The quad selector composes with the narrowed width pair.  metal-nt
    * -arch applegpu_g17p on `ushort`, immediate lane, identical registers:
    * only byte 1 moves between a quad word and its SIMD control. */
   static const struct {
      bool quad;
      enum agx3_subgroup_shuffle_op op;
      uint16_t lane;
      uint8_t bytes[10];
   } half_imm[] = {
      {true, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 3,
       {0x47, 0x00, 0x56, 0x02, 0x02, 0x04, 0x06, 0x20, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_BROADCAST, 3,
       {0x47, 0x04, 0x56, 0x02, 0x02, 0x04, 0x06, 0x20, 0x04, 0x00}},
      {true, AGX3_SUBGROUP_SHUFFLE_XOR, 1,
       {0xc7, 0x00, 0x56, 0x02, 0x02, 0x04, 0x02, 0x20, 0x04, 0x00}},
      {false, AGX3_SUBGROUP_SHUFFLE_XOR, 1,
       {0xc7, 0x04, 0x56, 0x02, 0x02, 0x04, 0x02, 0x20, 0x04, 0x00}},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(half_imm); ++i) {
      const struct agx3_subgroup_shuffle_desc q = {
         .quad = half_imm[i].quad,
         .op = half_imm[i].op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = 2,
         .src = 2,
         .lane = half_imm[i].lane,
         .wait_mask = 1u << 5,
         .half = true,
      };
      CHECK(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &q, out,
                                         &length));
      CHECK(bytes_equal(out, length, half_imm[i].bytes,
                        sizeof(half_imm[i].bytes)));
   }

   /* The shifts are not this composition: Apple's 16-bit simd_shuffle_up is
    * 47055603030402200200, whose f9/f10 pair is 3/3 rather than the 2/1 the
    * broadcast/butterfly point deposits.  Keep them refused until mapped. */
   d.op = AGX3_SUBGROUP_SHUFFLE_UP;
   CHECK(!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   d.op = AGX3_SUBGROUP_SHUFFLE_DOWN;
   d.quad = true;
   CHECK(!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   /* A 16-bit quad shuffle is witnessed only from an immediate lane. */
   d.op = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
   d.quad = true;
   d.lane_kind = AGX3_SUBGROUP_LANE_REGISTER;
   d.lane = 4;
   CHECK(!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   d.quad = false;
   d.lane_kind = AGX3_SUBGROUP_LANE_UNIFORM;
   d.lane = 8;
   CHECK(!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d, out,
                                       &length));
   return 0;
}

static int
test_ballot_vectors(void)
{
   static const uint8_t int_b[] = {
      0x17, 0x07, 0x54, 0x00, 0x02, 0x08, 0x00, 0xf8, 0xa2, 0x02,
   };
   static const uint8_t int_c[] = {
      0x27, 0x00, 0x00, 0x3a, 0x25, 0x00, 0xae, 0x02, 0x08, 0x80, 0x20, 0x04,
   };
   static const uint8_t float_b[] = {
      0x97, 0x07, 0x54, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x89, 0x00, 0x00, 0x00,
   };
   static const uint8_t float_c[] = {
      0x27, 0x00, 0x00, 0x1a, 0x25, 0x00, 0xaf, 0x02, 0x30, 0x80, 0x02, 0x00,
   };
   static const uint8_t int_lt_b[] = {
      0x17, 0x07, 0x54, 0x00, 0x02, 0x00, 0x00, 0x58, 0x22, 0x06,
   };
   static const uint8_t int_lt_c[] = {
      0x27, 0x00, 0x40, 0x1a, 0x25, 0x00, 0xae, 0x02, 0x38, 0x00, 0x00, 0x04,
   };
   static const uint8_t quad_int_ne_b[] = {
      0x17, 0x02, 0x54, 0x00, 0x03, 0x08, 0x00, 0xf8, 0xa2, 0x12,
   };
   static const uint8_t quad_int_ne_c[] = {
      0x27, 0x00, 0x00, 0x3a, 0x2d, 0x00, 0xa6, 0x02, 0x08, 0x80, 0x28, 0x04,
   };
   static const uint8_t float_lt_b[] = {
      0x97, 0x07, 0x54, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x89, 0x80, 0x00, 0x00,
   };
   static const uint8_t float_lt_c[] = {
      0x27, 0x00, 0x00, 0x1a, 0x25, 0x00, 0xaf, 0x02, 0x30, 0x80, 0x02, 0x40,
   };
   /* applegpu-nt 32023.917, apple-g17p, simd_ballot(tid < 16/5). */
   static const uint8_t uint_lt16_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x40, 0x18, 0x22, 0x0c,
   };
   static const uint8_t uint_lt16_cached_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x03, 0x04, 0x40, 0x18, 0x22, 0x0c,
   };
   static const uint8_t uint_lt5_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x14, 0x18, 0x22, 0x0c,
   };
   static const uint8_t uint_lt16_dead_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x00, 0x40, 0x58, 0x22, 0x0c,
   };
   /* Apple Metal/applegpu-nt 32023.917, apple-g17p.  The compiler
    * canonicalizes tid >= 16 to the native unsigned tid > 15 form. */
   static const uint8_t uint_gt15_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x3c, 0x18, 0x22, 0x0a,
   };
   static const uint8_t uint_eq0_live_b[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x00, 0x18, 0x22, 0x02,
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   struct agx3_subgroup_ballot_desc d = {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 4,
      .condition = 4,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, int_b, sizeof(int_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, int_c, sizeof(int_c)));

   /* G17P retains a live ballot input for the following subgroup broadcast.
    * f27 is the source discard bit in the native live/dead witnesses above
    * the encoder; the register-zero comparison fields remain unchanged.
    */
   static const uint8_t int_live_b[] = {
      0x17, 0x07, 0x54, 0x00, 0x02, 0x08, 0x00, 0xb8, 0xa2, 0x02,
   };
   d.src_live = true;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, int_live_b, sizeof(int_live_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, int_c, sizeof(int_c)));
   d.src_live = false;

   d.dest = 4;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   d.dest = 0;

   d.src = 0;
   d.condition = 5;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, int_lt_b, sizeof(int_lt_b)));
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, int_lt_c, sizeof(int_lt_c)));

   d.quad = true;
   d.src = 4;
   d.condition = 4;
   d.invert = true;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, quad_int_ne_b, sizeof(quad_int_ne_b)));
   CHECK(agx3_subgroup_encode_ballot_g18g(AGX_ISA_ENCODING_AGX3C, &d, out,
                                          &length));
   CHECK(bytes_equal(out, length, quad_int_ne_c, sizeof(quad_int_ne_c)));

   d.type = AGX3_SUBGROUP_BALLOT_FLOAT;
   d.quad = false;
   d.src = 0;
   d.condition = 0;
   d.invert = false;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, float_b, sizeof(float_b)));
   CHECK(agx3_subgroup_encode_ballot_g18g(AGX_ISA_ENCODING_AGX3C, &d, out,
                                          &length));
   CHECK(bytes_equal(out, length, float_c, sizeof(float_c)));

   d.condition = 1;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, float_lt_b, sizeof(float_lt_b)));
   CHECK(agx3_subgroup_encode_ballot_g18g(AGX_ISA_ENCODING_AGX3C, &d, out,
                                          &length));
   CHECK(bytes_equal(out, length, float_lt_c, sizeof(float_lt_c)));

   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 2,
      .condition = 1,
      .compare_immediate = true,
      .compare_value = 16,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt16_live_b,
                     sizeof(uint_lt16_live_b)));
   d.dest_cache = true;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt16_cached_b,
                     sizeof(uint_lt16_cached_b)));
   d.dest_cache = false;
   d.compare_value = 5;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt5_live_b,
                     sizeof(uint_lt5_live_b)));
   d.src = 0;
   d.compare_value = 16;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, uint_lt16_dead_b,
                     sizeof(uint_lt16_dead_b)));

   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 2,
      .condition = 2,
      .compare_immediate = true,
      .compare_value = 15,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, uint_gt15_live_b,
                     sizeof(uint_gt15_live_b)));

   d.condition = 4;
   d.compare_value = 0;
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, uint_eq0_live_b,
                     sizeof(uint_eq0_live_b)));

   /* metal-nt applegpu_g17p simd_broadcast_first(): the 16-bit active thread
    * index in h4 compared equal to zero, discarded, into a cached r2
    * (half 4). */
   static const uint8_t half_eq0_b[] = {0x17, 0x17, 0x54, 0x04, 0x03,
                                        0x08, 0x00, 0x48, 0x22, 0x02};
   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .dest = 4,
      .src = 4,
      .condition = 4,
      .compare_immediate = true,
      .src_half = true,
      .dest_cache = true,
   };
   CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, half_eq0_b, sizeof(half_eq0_b)));
   /* The 16-bit quad ballot against zero of a ushort or uint, sources
    * arriving on slot 6. */
   static const struct {
      uint16_t dest, src;
      bool src_half, live, invert;
      uint8_t bytes[10];
   } quad_half[] = {
      {0, 0, true, false, true,
       {0x17, 0x02, 0x56, 0x00, 0x03, 0x00, 0x00, 0x40, 0x22, 0x12}},
      {1, 1, true, false, false,
       {0x17, 0x02, 0x56, 0x01, 0x03, 0x02, 0x00, 0x40, 0x22, 0x02}},
      {0, 1, true, true, true,
       {0x17, 0x02, 0x56, 0x00, 0x03, 0x02, 0x02, 0x00, 0x22, 0x12}},
      {0, 0, false, false, false,
       {0x17, 0x02, 0x56, 0x00, 0x03, 0x00, 0x00, 0x50, 0x22, 0x02}},
      {0, 0, false, false, true,
       {0x17, 0x02, 0x56, 0x00, 0x03, 0x00, 0x00, 0x50, 0x22, 0x12}},
      {2, 0, false, true, true,
       {0x17, 0x02, 0x56, 0x02, 0x03, 0x00, 0x02, 0x10, 0x22, 0x12}},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(quad_half); ++i) {
      const struct agx3_subgroup_ballot_desc q = {
         .quad = true,
         .type = AGX3_SUBGROUP_BALLOT_INTEGER,
         .dest = quad_half[i].dest,
         .src = quad_half[i].src,
         .src_half = quad_half[i].src_half,
         .dest_half = true,
         .dest_cache = true,
         .src_live = quad_half[i].live,
         .condition = 4,
         .invert = quad_half[i].invert,
         .wait_mask = 1u << 5,
      };
      CHECK(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &q, out,
                                        &length));
      CHECK(bytes_equal(out, length, quad_half[i].bytes,
                        sizeof(quad_half[i].bytes)));
   }

   d.compare_value = 1;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.compare_value = 0;
   d.condition = 1;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.condition = 4;
   d.compare_immediate = false;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));

   d = (struct agx3_subgroup_ballot_desc) {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .src = 2,
      .condition = 2,
      .compare_immediate = true,
      .compare_value = 15,
      .src_live = true,
   };

   /* No immediate signed, inverted, or quad form was witnessed. */
   d.condition = 5;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.condition = 2;
   d.invert = true;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.invert = false;
   d.quad = true;
   CHECK(!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   return 0;
}

static int
test_reduce_vectors(void)
{
   static const uint8_t fadd_b[] = {
      0x3f, 0x06, 0x54, 0x00, 0x02, 0x00, 0x14, 0x12,
   };
   static const uint8_t fadd_c[] = {
      0x27, 0x00, 0x04, 0x08, 0x27, 0x80, 0xac, 0x02, 0x08, 0x02,
   };
   static const uint8_t qfadd_b[] = {
      0x37, 0x06, 0x54, 0x00, 0x02, 0x00, 0x14, 0x12,
   };
   static const uint8_t qfadd_c[] = {
      0x27, 0x00, 0x00, 0x08, 0x27, 0x80, 0xac, 0x02, 0x08, 0x02,
   };
   static const uint8_t prefix_iadd_b[] = {
      0xbf, 0x01, 0x54, 0x00, 0x02, 0x00, 0x14, 0x0b,
   };
   static const uint8_t prefix_iadd_c[] = {
      0x27, 0x00, 0x04, 0x0a, 0x27, 0x80, 0xa3, 0x02, 0x00, 0x00,
   };
   static const uint8_t prefix_fadd_b[] = {
      0x3f, 0x06, 0x54, 0x00, 0x02, 0x00, 0x14, 0x32,
   };
   static const uint8_t prefix_fadd_c[] = {
      0x27, 0x00, 0x04, 0x08, 0x27, 0x80, 0xac, 0x02, 0x08, 0x00,
   };
   static const uint8_t and_b[] = {
      0x3f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x14, 0x03,
   };
   static const uint8_t and_c[] = {
      0x27, 0x00, 0x04, 0x0a, 0x27, 0x80, 0xa0, 0x02, 0x00, 0x02,
   };
   static const uint8_t smin_b[] = {
      0x3f, 0x02, 0x54, 0x00, 0x02, 0x00, 0x14, 0x07,
   };
   static const uint8_t smin_c[] = {
      0x27, 0x00, 0x04, 0x0a, 0x27, 0x80, 0xa4, 0x02, 0x40, 0x02,
   };
   static const uint8_t smin_wait6_b[] = {
      0x3f, 0x02, 0x56, 0x00, 0x02, 0x00, 0x14, 0x07,
   };
   static const uint8_t smax_wait6_b[] = {
      0xbf, 0x02, 0x56, 0x00, 0x02, 0x00, 0x14, 0x07,
   };
   static const uint8_t qiadd_live_wait6_b[] = {
      0xb7, 0x01, 0x56, 0x00, 0x02, 0x0c, 0x16, 0x01,
   };
   static const uint8_t qumin_live_b[] = {
      0x37, 0x03, 0x54, 0x02, 0x02, 0x0c, 0x16, 0x01,
   };
   static const uint8_t qumax_live_b[] = {
      0xb7, 0x03, 0x54, 0x04, 0x02, 0x0c, 0x16, 0x01,
   };
   static const uint8_t qsmin_live_wait6_b[] = {
      0x37, 0x02, 0x56, 0x00, 0x02, 0x04, 0x16, 0x05,
   };
   static const uint8_t qsmax_final_b[] = {
      0xb7, 0x02, 0x54, 0x02, 0x02, 0x04, 0x14, 0x07,
   };
   static const uint8_t qand_live_wait6_b[] = {
      0x37, 0x00, 0x56, 0x00, 0x02, 0x08, 0x16, 0x01,
   };
   static const uint8_t qor_live_b[] = {
      0xb7, 0x00, 0x54, 0x02, 0x02, 0x08, 0x16, 0x01,
   };
   static const uint8_t qxor_final_b[] = {
      0x37, 0x01, 0x54, 0x04, 0x02, 0x08, 0x14, 0x03,
   };
   static const uint8_t qfadd_live_wait6_b[] = {
      0x37, 0x06, 0x56, 0x00, 0x02, 0x08, 0x16, 0x02,
   };
   static const uint8_t qfmin_live_b[] = {
      0xb7, 0x05, 0x54, 0x02, 0x02, 0x08, 0x0a, 0x02,
   };
   static const uint8_t qfmax_final_b[] = {
      0xb7, 0x07, 0x54, 0x04, 0x02, 0x08, 0x08, 0x12,
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   struct agx3_subgroup_reduce_desc d = {
      .op = AGX3_SUBGROUP_REDUCE_FADD,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, fadd_b, sizeof(fadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, fadd_c, sizeof(fadd_c)));

   d.prefix = true;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_fadd_b, sizeof(prefix_fadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_fadd_c, sizeof(prefix_fadd_c)));
   d.prefix = false;

   d.quad = true;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfadd_b, sizeof(qfadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfadd_c, sizeof(qfadd_c)));
   d.quad = false;

   for (unsigned op = 0; op < AGX3_SUBGROUP_REDUCE_NUM; ++op) {
      d.op = op;
      CHECK(
         agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
      CHECK(
         agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   }

   d.op = AGX3_SUBGROUP_REDUCE_AND;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, and_b, sizeof(and_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, and_c, sizeof(and_c)));

   d.op = AGX3_SUBGROUP_REDUCE_SMIN;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, smin_b, sizeof(smin_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, smin_c, sizeof(smin_c)));

   d.wait_mask = 1u << 5;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, smin_wait6_b, sizeof(smin_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_SMAX;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, smax_wait6_b, sizeof(smax_wait6_b)));
   d.wait_mask = 0;

   d.op = AGX3_SUBGROUP_REDUCE_IADD;
   d.prefix = true;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_iadd_b, sizeof(prefix_iadd_b)));
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   CHECK(bytes_equal(out, length, prefix_iadd_c, sizeof(prefix_iadd_c)));

   d = (struct agx3_subgroup_reduce_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_IADD,
      .src = 6,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qiadd_live_wait6_b,
                     sizeof(qiadd_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_UMIN;
   d.dest = 2;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qumin_live_b, sizeof(qumin_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_UMAX;
   d.dest = 4;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qumax_live_b, sizeof(qumax_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_SMIN;
   d.dest = 0;
   d.src = 2;
   d.wait_mask = 1u << 5;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qsmin_live_wait6_b,
                     sizeof(qsmin_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_SMAX;
   d.dest = 2;
   d.wait_mask = 0;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qsmax_final_b, sizeof(qsmax_final_b)));

   d = (struct agx3_subgroup_reduce_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_AND,
      .src = 4,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qand_live_wait6_b,
                     sizeof(qand_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_OR;
   d.dest = 2;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qor_live_b, sizeof(qor_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_XOR;
   d.dest = 4;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qxor_final_b, sizeof(qxor_final_b)));

   d = (struct agx3_subgroup_reduce_desc){
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_FADD,
      .src = 4,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfadd_live_wait6_b,
                     sizeof(qfadd_live_wait6_b)));

   d.op = AGX3_SUBGROUP_REDUCE_FMIN;
   d.dest = 2;
   d.wait_mask = 0;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfmin_live_b, sizeof(qfmin_live_b)));

   d.op = AGX3_SUBGROUP_REDUCE_FMAX;
   d.dest = 4;
   d.src_live = false;
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   CHECK(bytes_equal(out, length, qfmax_final_b, sizeof(qfmax_final_b)));
   return 0;
}

static int
test_rejection(void)
{
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   struct agx3_subgroup_shuffle_desc s = {
      .op = AGX3_SUBGROUP_SHUFFLE_XOR,
      .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
      .dest = 0,
      .src = 2,
      .lane = 4,
   };
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX2, &s, out, &length));
   /* The G17P register-lane forms carry waits for the six witnessed slots. */
   s.wait_mask = 1;
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 1u << 6;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 0;
   /* An immediate lane is now the whole seven-bit field on every one of the
    * six mnemonics, so the shapes this used to refuse all encode. What still
    * fails closed is a lane the field cannot hold and a wait outside the six
    * witnessed slots. */
   s.lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE;
   s.lane = 0x80;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.lane = 0x7f;
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 1u << 6;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.wait_mask = 0;
   s.quad = false;
   s.lane_kind = AGX3_SUBGROUP_LANE_REGISTER;
   /* G17P reads the lane as 16 bits, from any half. */
   s.lane = 3;
   CHECK(
      agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));
   s.lane = 4;
   s.op = (enum agx3_subgroup_shuffle_op)-1;
   CHECK(
      !agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &s, out, &length));

   struct agx3_subgroup_ballot_desc b = {
      .type = AGX3_SUBGROUP_BALLOT_INTEGER,
      .condition = 0,
   };
   CHECK(
      !agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &b, out, &length));
   b.condition = 4;
   b.wait_mask = 1;
   CHECK(
      !agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &b, out, &length));
   b.wait_mask = 0;
   b.type = (enum agx3_subgroup_ballot_type)-1;
   CHECK(
      !agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &b, out, &length));

   struct agx3_subgroup_reduce_desc r = {
      .quad = true,
      .op = AGX3_SUBGROUP_REDUCE_AND,
   };
   r.prefix = true;
   CHECK(
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &r, out, &length));
   r.quad = false;
   r.prefix = false;
   r.op = (enum agx3_subgroup_reduce_op)-1;
   CHECK(
      !agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &r, out, &length));
   return 0;
}

static int
test_half_reduce_vectors(void)
{
   /* metal-nt -arch applegpu_g17p. Every source arrives from a device load on
    * scoreboard slot 6. The unsigned quad words come from o = quad_op(x16),
    * whose result escapes the quad and so also sets f62 (byte 1, bit 3);
    * Apple leaves that bit clear when the result feeds another quad
    * reduction, and so does this encoder, as the signed pair shows:
    * quad_min(short) is 370a560003001007 escaping and 3702560003001007
    * chained. */
   static const struct {
      bool quad, prefix, dest_hint, src_live;
      enum agx3_subgroup_reduce_op op;
      uint16_t dest, src;
      uint8_t bytes[8];
   } vec[] = {
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_IADD, 2, 2,
       {0xbf, 0x01, 0x56, 0x02, 0x02, 0x04, 0x10, 0x03}},
      {false, true, true, true, AGX3_SUBGROUP_REDUCE_IADD, 2, 3,
       {0xbf, 0x01, 0x56, 0x02, 0x03, 0x06, 0x12, 0x09}},
      {false, true, false, false, AGX3_SUBGROUP_REDUCE_IADD, 2, 2,
       {0xbf, 0x01, 0x56, 0x02, 0x02, 0x04, 0x10, 0x0b}},
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_UMIN, 2, 2,
       {0x3f, 0x03, 0x56, 0x02, 0x02, 0x04, 0x10, 0x03}},
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_SMIN, 2, 2,
       {0x3f, 0x02, 0x56, 0x02, 0x02, 0x04, 0x10, 0x07}},
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_SMAX, 2, 2,
       {0xbf, 0x02, 0x56, 0x02, 0x02, 0x04, 0x10, 0x07}},
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_AND, 2, 2,
       {0x3f, 0x00, 0x56, 0x02, 0x02, 0x04, 0x10, 0x03}},
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_OR, 2, 2,
       {0xbf, 0x00, 0x56, 0x02, 0x02, 0x04, 0x10, 0x03}},
      {false, false, false, false, AGX3_SUBGROUP_REDUCE_XOR, 2, 2,
       {0x3f, 0x01, 0x56, 0x02, 0x02, 0x04, 0x10, 0x03}},
      {false, true, true, true, AGX3_SUBGROUP_REDUCE_FMUL, 3, 2,
       {0xbf, 0x04, 0x56, 0x03, 0x03, 0x04, 0x8a, 0x10}},
      {false, true, false, false, AGX3_SUBGROUP_REDUCE_FMUL, 2, 2,
       {0xbf, 0x04, 0x56, 0x02, 0x02, 0x04, 0x88, 0x18}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_SMAX, 0, 0,
       {0xb7, 0x02, 0x56, 0x00, 0x03, 0x00, 0x10, 0x07}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_SMIN, 0, 0,
       {0x37, 0x02, 0x56, 0x00, 0x03, 0x00, 0x10, 0x07}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_FADD, 0, 0,
       {0x37, 0x04, 0x56, 0x00, 0x03, 0x00, 0x88, 0x08}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_FMIN, 0, 0,
       {0xb7, 0x05, 0x56, 0x00, 0x03, 0x00, 0x08, 0x13}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_FMAX, 0, 0,
       {0xb7, 0x07, 0x56, 0x00, 0x03, 0x00, 0x08, 0x13}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_IADD, 0, 0,
       {0xb7, 0x01, 0x56, 0x00, 0x03, 0x00, 0x10, 0x03}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_AND, 0, 0,
       {0x37, 0x00, 0x56, 0x00, 0x03, 0x00, 0x10, 0x03}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_UMIN, 0, 0,
       {0x37, 0x03, 0x56, 0x00, 0x03, 0x00, 0x10, 0x03}},
      {true, false, true, false, AGX3_SUBGROUP_REDUCE_UMAX, 0, 0,
       {0xb7, 0x03, 0x56, 0x00, 0x03, 0x00, 0x10, 0x03}},
   };
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], length = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(vec); ++i) {
      const struct agx3_subgroup_reduce_desc d = {
         .quad = vec[i].quad,
         .prefix = vec[i].prefix,
         .op = vec[i].op,
         .dest = vec[i].dest,
         .src = vec[i].src,
         .wait_mask = 1u << 5,
         .dest_hint = vec[i].dest_hint,
         .src_live = vec[i].src_live,
         .half = true,
      };
      CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                        &length));
      CHECK(bytes_equal(out, length, vec[i].bytes, sizeof(vec[i].bytes)));
   }

   /* simd_prefix_inclusive_sum(uint) keeps its source live: f25, not f27. */
   static const uint8_t psum32[] = {0xbf, 0x01, 0x56, 0x04,
                                    0x03, 0x00, 0x16, 0x09};
   struct agx3_subgroup_reduce_desc d = {
      .prefix = true,
      .op = AGX3_SUBGROUP_REDUCE_IADD,
      .dest = 4,
      .src = 0,
      .wait_mask = 1u << 5,
      .dest_hint = true,
      .src_live = true,
   };
   CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                     &length));
   CHECK(bytes_equal(out, length, psum32, sizeof(psum32)));

   /* The remaining integer scans set their table prefix operand: word 3 bit
    * 11 (f53) on signed min/max, bit 10 (f52) on the bitwise and unsigned
    * forms, and nothing else changes. */
   static const struct {
      enum agx3_subgroup_reduce_op op;
      uint8_t bit;
   } scan[] = {
      {AGX3_SUBGROUP_REDUCE_SMIN, 0x08}, {AGX3_SUBGROUP_REDUCE_SMAX, 0x08},
      {AGX3_SUBGROUP_REDUCE_AND, 0x04},  {AGX3_SUBGROUP_REDUCE_OR, 0x04},
      {AGX3_SUBGROUP_REDUCE_XOR, 0x04},  {AGX3_SUBGROUP_REDUCE_UMIN, 0x04},
      {AGX3_SUBGROUP_REDUCE_UMAX, 0x04},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(scan); ++i) {
      for (unsigned half = 0; half < 2; ++half) {
         uint8_t reduce[AGX3_SUBGROUP_MAX_LENGTH], reduce_length = 0;
         d = (struct agx3_subgroup_reduce_desc){
            .op = scan[i].op,
            .dest = 2,
            .src = 4,
            .half = half,
         };
         CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, reduce,
                                           &reduce_length));
         d.prefix = true;
         CHECK(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                           &length));
         CHECK(length == reduce_length && length == 8);
         reduce[7] |= scan[i].bit;
         CHECK(bytes_equal(out, length, reduce, reduce_length));
      }
   }

   /* No float min/max scan and no quad scan exist. */
   d = (struct agx3_subgroup_reduce_desc){
      .prefix = true,
      .op = AGX3_SUBGROUP_REDUCE_FMIN,
   };
   CHECK(!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   d.op = AGX3_SUBGROUP_REDUCE_IADD;
   d.quad = true;
   CHECK(!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d, out,
                                      &length));
   return 0;
}

int
main(void)
{
   CHECK(test_shuffle_vectors() == 0);
   CHECK(test_half_broadcast_vectors() == 0);
   CHECK(test_register_lane_hint_vectors() == 0);
   CHECK(test_ballot_vectors() == 0);
   CHECK(test_reduce_vectors() == 0);
   CHECK(test_half_reduce_vectors() == 0);
   CHECK(test_rejection() == 0);
   return 0;
}
