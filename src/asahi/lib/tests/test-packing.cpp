/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_pack.h"
#include "agx_uvs.h"
#include "asahi/compiler/agx_compile.h"
#include "agx_vdm_mesh_core.h"
#include <cstring>
#include <array>

#include <gtest/gtest.h>

const struct {
   float f;
   uint16_t encoded;
   bool inexact;
} lod_cases[] = {
   /* Lower bound clamp */
   {-INFINITY, 0x00, true},
   {-0.1, 0x00, true},
   {-0.0, 0x00, true},

   /* Exact bounds */
   {0.0, 0x00},
   {14.0, 0x380},

   /* A positive clamp below one step still minifies */
   {0.01, 0x01, true},
   {1.0 / 64.0, 0x01},

   /* Upper bound clamp */
   {14.1, 0x380, true},
   {18.1, 0x380, true},
   {INFINITY, 0x380, true},
};

const struct {
   uint32_t group_size;
   uint32_t length;
   uint32_t value;
   uint32_t encoded;
} group_cases[] = {
   /* Groups of 16 in a 4-bit word */
   {16, 4, 0, 0x1},   {16, 4, 1, 0x1},   {16, 4, 16, 0x1},  {16, 4, 17, 0x2},
   {16, 4, 31, 0x2},  {16, 4, 32, 0x2},  {16, 4, 33, 0x3},  {16, 4, 239, 0xF},
   {16, 4, 240, 0xF}, {16, 4, 241, 0x0}, {16, 4, 255, 0x0}, {16, 4, 256, 0x0},
};

TEST(LODClamp, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i)
      ASSERT_EQ(__gen_pack_lod(lod_cases[i].f, 0, 9), lod_cases[i].encoded);
}

TEST(LODClamp, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i) {
      if (lod_cases[i].inexact)
         continue;

      uint32_t cl;
      memcpy(&cl, &lod_cases[i].encoded, sizeof(lod_cases[i].encoded));

      ASSERT_EQ(__gen_unpack_lod(&cl, 0, 10), lod_cases[i].f);
   }
}

TEST(Groups, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      ASSERT_EQ(__gen_to_groups(group_cases[i].value, group_cases[i].group_size,
                                group_cases[i].length),
                group_cases[i].encoded);
   }
}

TEST(Groups, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      unsigned expected =
         ALIGN_POT(group_cases[i].value, group_cases[i].group_size);

      /* Clamp to minimum encodable */
      if (group_cases[i].value == 0)
         expected = group_cases[i].group_size;

      ASSERT_EQ(
         __gen_from_groups(group_cases[i].encoded, group_cases[i].group_size,
                           group_cases[i].length),
         expected);
   }
}


TEST(M5MeshLimits, MatchesNativePackets)
{
   struct { uint32_t vertices, primitives, words, prefix, prim_limit, vert_limit; } cases[] = {
      {6, 2, 4, 0x504, 0x1, 0x9},
      {6, 2, 5, 0x605, 0x1, 0xb},
      {192, 64, 4, 0x504, 0x3003f, 0xc1},
   };
   for (const auto &c : cases) {
      agx_vdm_mesh_state_info info = {};
      ASSERT_TRUE(agx_vdm_m5_mesh_set_output_limits(&info, c.vertices, c.primitives, c.words));
      EXPECT_EQ(info.raw_prefix[0], c.prefix);
      EXPECT_EQ(info.raw_tail[1], c.prim_limit);
      EXPECT_EQ(info.raw_tail[4], c.vert_limit);
   }
}

/* The mesh dispatch record is an Index List block whose Primitive field
 * (bits 8..15) names the mesh output topology; native triangle meshes
 * carry 6 (0x70000600). */
TEST(M5MeshDispatch, PrimitiveFollowsTopology)
{
   for (const auto &c : {std::array<uint32_t, 2>{0, 0x70000000}, {1, 0x70000100},
                        {6, 0x70000600}}) {
      agx_vdm_mesh_dispatch_info info = {};
      for (unsigned i = 0; i < 3; ++i)
         info.threads[i] = info.local_size[i] = i ? 1 : 3;
      info.primitive = c[0];
      uint8_t bytes[AGX_VDM_MESH_DISPATCH_LENGTH];
      ASSERT_TRUE(agx_vdm_mesh_dispatch_pack(AGX_CHIP_G17S, &info, bytes));
      uint32_t header;
      memcpy(&header, bytes, 4);
      EXPECT_EQ(header, c[1]);
      agx_vdm_mesh_dispatch_info back;
      ASSERT_TRUE(agx_vdm_mesh_dispatch_unpack(AGX_CHIP_G17S, bytes, sizeof(bytes), &back));
      EXPECT_EQ(back.primitive, c[0]);
   }
   agx_vdm_mesh_dispatch_info info = {};
   for (unsigned i = 0; i < 3; ++i)
      info.threads[i] = info.local_size[i] = 1;
   info.primitive = 3; /* line strip: no mesh topology */
   uint8_t bytes[AGX_VDM_MESH_DISPATCH_LENGTH];
   EXPECT_FALSE(agx_vdm_mesh_dispatch_pack(AGX_CHIP_G17S, &info, bytes));
}

TEST(M5MeshLimits, RejectsInvalidLimitsWithoutMutation)
{
   for (const auto &c : {std::array<uint32_t, 3>{0, 1, 4}, {1, 0, 4},
                        {257, 1, 4}, {1, 257, 4}, {1, 1, 3}, {1, 1, 256}}) {
      agx_vdm_mesh_state_info info;
      memset(&info, 0x5a, sizeof(info));
      const auto before = info;
      EXPECT_FALSE(agx_vdm_m5_mesh_set_output_limits(&info, c[0], c[1], c[2]));
      EXPECT_EQ(memcmp(&info, &before, sizeof(info)), 0);
   }
}

TEST(M5MeshTaskPayload, SizesSlotStrideAndRoundTrips)
{
   const struct {
      uint32_t payload_size_B;
      uint32_t expected_blocks;
   } cases[] = {
      {0, 1},
      {16, 1},
      {1024, 1},
      {1025, 2},
      {1040, 2},
      {2048, 2},
      {16400, 17},
   };

   for (const auto &c : cases) {
      agx_vdm_mesh_state_info info = {};
      info.object_pipeline = UINT64_C(0x100000);
      info.mesh_pipeline = UINT64_C(0x200000);
      info.raw_mesh_template = 0x40;
      ASSERT_TRUE(agx_vdm_m5_mesh_set_output_limits(&info, 192, 64, 8));
      ASSERT_TRUE(agx_vdm_m5_mesh_set_task_payload(&info, c.payload_size_B));
      EXPECT_EQ(info.raw_tail[0], c.expected_blocks);

      for (enum agx_chip chip :
           {AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D}) {
         uint8_t bytes[AGX_VDM_MESH_STATE_LENGTH];
         ASSERT_TRUE(agx_vdm_mesh_state_pack(chip, &info, bytes));

         uint32_t packed_tail0 = 0;
         memcpy(&packed_tail0, bytes + 36, sizeof(packed_tail0));
         EXPECT_EQ(packed_tail0, c.expected_blocks);

         agx_vdm_mesh_state_info unpacked = {};
         ASSERT_TRUE(
            agx_vdm_mesh_state_unpack(chip, bytes, sizeof(bytes), &unpacked));
         EXPECT_EQ(unpacked.raw_tail[0], c.expected_blocks);
         EXPECT_EQ(unpacked.raw_tail[1], info.raw_tail[1]);
         EXPECT_EQ(unpacked.raw_tail[4], info.raw_tail[4]);
      }
   }
}

TEST(M5MeshTaskPayload, RejectsInvalidPayloadWithoutMutation)
{
   EXPECT_FALSE(agx_vdm_m5_mesh_set_task_payload(nullptr, 1024));

   for (uint32_t bad : {AGX_VDM_M5_MESH_MAX_TASK_PAYLOAD_BYTES + 1u,
                        UINT32_C(32768), UINT32_MAX}) {
      agx_vdm_mesh_state_info info;
      memset(&info, 0x5a, sizeof(info));
      const auto before = info;
      EXPECT_FALSE(agx_vdm_m5_mesh_set_task_payload(&info, bad));
      EXPECT_EQ(memcmp(&info, &before, sizeof(info)), 0);
   }
}

TEST(Packing, CfBindingsSelectPerspective)
{
   struct agx_varyings_fs fs = {};
   fs.nr_bindings = 3;
   fs.bindings[0] = { .cf_base = 0, .count = 4, .smooth = true, .perspective = true };
   fs.bindings[1] = { .cf_base = 4, .count = 4, .smooth = true, .perspective = false };
   fs.bindings[2] = { .cf_base = 8, .count = 4, .smooth = false, .perspective = false };

   uint8_t buf[AGX_CF_BINDING_HEADER_LENGTH + 3 * AGX_CF_BINDING_LENGTH] = {};
   struct agx_cf_binding_packed *packed =
      (struct agx_cf_binding_packed *)(buf + AGX_CF_BINDING_HEADER_LENGTH);

   packed[0].opaque[0] = (AGX_SHADE_MODEL_PERSPECTIVE << 2) | (0x123u << 5);
   packed[1].opaque[0] = (AGX_SHADE_MODEL_LINEAR << 2) | (0x456u << 5);
   packed[2].opaque[0] = (AGX_SHADE_MODEL_FLAT_VERTEX_0 << 2) | (0x789u << 5);

   agx_cf_bindings_select_perspective(buf, &fs);

   EXPECT_EQ((packed[0].opaque[0] >> 2) & 0x7u, 5u);
   EXPECT_EQ(packed[0].opaque[0] & ~BITFIELD_RANGE(2, 3), 0x123u << 5);
   EXPECT_EQ((packed[1].opaque[0] >> 2) & 0x7u, (unsigned)AGX_SHADE_MODEL_LINEAR);
   EXPECT_EQ(packed[1].opaque[0] & ~BITFIELD_RANGE(2, 3), 0x456u << 5);
   EXPECT_EQ((packed[2].opaque[0] >> 2) & 0x7u, (unsigned)AGX_SHADE_MODEL_FLAT_VERTEX_0);
   EXPECT_EQ(packed[2].opaque[0] & ~BITFIELD_RANGE(2, 3), 0x789u << 5);
}
