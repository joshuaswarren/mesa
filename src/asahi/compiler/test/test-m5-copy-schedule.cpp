/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_test.h"

#include <gtest/gtest.h>
#include <vector>

/* M5 copy loops: reorderable image reads issue ahead of the writes that
 * consume them, as Apple's compiler does for a vertex copy loop (four SMP,
 * then four IMGWCI). */

namespace {

struct copy_shader {
   void *mem;
   agx_shader_key key;
   agx_builder *b;
   agx_block *body;
};

static void
copy_shader_init(copy_shader *s, mesa_shader_stage stage, bool loop)
{
   s->mem = ralloc_context(NULL);
   s->key = {};
   s->key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   s->key.dev.chip = AGX_CHIP_G17S;
   s->b = agx_test_builder(s->mem);
   s->b->shader->key = &s->key;
   s->b->shader->stage = stage;
   s->b->shader->nir = rzalloc(s->mem, nir_shader);
   s->b->shader->nir->info.stage = stage;
   s->body = agx_start_block(s->b->shader);
   if (loop) {
      agx_block *entry = s->body;
      s->body = agx_test_block(s->b->shader);
      agx_block *exit = agx_test_block(s->b->shader);
      s->body->loop_header = true;
      agx_block_add_successor(entry, s->body);
      agx_block_add_successor(s->body, s->body);
      agx_block_add_successor(s->body, exit);
   }
}

/* Copy n texels at constant x coordinates defined in the entry block. */
static void
emit_copies(copy_shader *s, unsigned n, bool reorderable)
{
   agx_builder *b = s->b;
   agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index z = agx_preload(b, agx_register(4, AGX_SIZE_32));
   agx_index x[16];
   assert(n <= 16);
   for (unsigned i = 0; i < n; ++i)
      x[i] = agx_mov_imm(b, 32, i);

   b->cursor = agx_after_block(s->body);
   for (unsigned i = 0; i < n; ++i) {
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, coords, 3);
      collect->src[0] = x[i];
      collect->src[1] = y;
      collect->src[2] = z;

      agx_index texel = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *load = agx_image_load_to(
         b, texel, coords, agx_zero(), agx_zero(), agx_immediate(1),
         agx_immediate(0), agx_null(), AGX_DIM_3D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false);
      load->mask = 1;
      load->memory_can_reorder = reorderable;
      agx_image_write(b, texel, coords, agx_zero(),
                      agx_uniform(8, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_3D, false);
   }
}

/* Largest number of reads issued ahead of the first unissued write. */
static unsigned
max_reads_in_flight(agx_block *block, unsigned *first_write_at)
{
   unsigned loads = 0, writes = 0, max = 0, pos = 0;
   *first_write_at = ~0u;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IMAGE_LOAD) {
         ++loads;
         max = MAX2(max, loads - writes);
      } else if (I->op == AGX_OPCODE_IMAGE_WRITE) {
         if (writes == 0)
            *first_write_at = pos;
         ++writes;
      }
      ++pos;
   }
   return max;
}

static unsigned
loads_before_first_write(agx_block *block)
{
   unsigned loads = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IMAGE_WRITE)
         break;
      loads += I->op == AGX_OPCODE_IMAGE_LOAD;
   }
   return loads;
}

/* n device loads, each added into a running sum right after it. */
static void
emit_load_uses(copy_shader *s, unsigned n)
{
   agx_builder *b = s->b;
   b->cursor = agx_after_block(s->body);
   agx_index address = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index sum = agx_preload(b, agx_register(6, AGX_SIZE_32));
   for (unsigned i = 0; i < n; ++i) {
      agx_index value = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, value, address, agx_immediate(i * 4),
                         AGX_FORMAT_I32, 1, 0, false);
      sum = agx_iadd(b, sum, value, 0);
   }
   agx_unit_test(b, sum);
}

static unsigned
loads_before_first_add(agx_block *block)
{
   unsigned loads = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IADD)
         break;
      loads += I->op == AGX_OPCODE_DEVICE_LOAD;
   }
   return loads;
}

} /* namespace */

TEST(M5CopySchedule, VertexReadsIssueAheadOfWrites)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, false);
   emit_copies(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 4u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 4u);
   ralloc_free(s.mem);
}

TEST(M5CopySchedule, OrderedReadsStayBehindWrites)
{
   /* A read that may observe an earlier write keeps memory order. */
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, false);
   emit_copies(&s, 8, false);
   agx_ilp_schedule(s.b->shader);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 1u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 1u);
   ralloc_free(s.mem);
}

TEST(M5CopySchedule, LoopConstantsRematerializedInBody)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, true);
   emit_copies(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   /* Every coordinate constant is now defined in the loop body. */
   unsigned body_constants = 0;
   agx_foreach_instr_in_block(s.body, I)
      body_constants += I->op == AGX_OPCODE_MOV_IMM;
   EXPECT_EQ(body_constants, 8u);
   agx_foreach_instr_in_block(agx_start_block(s.b->shader), I)
      EXPECT_NE(I->op, AGX_OPCODE_MOV_IMM);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 4u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 4u);
   ralloc_free(s.mem);
}

TEST(M5CopySchedule, OtherGenerationsUnchanged)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_VERTEX, false);
   s.key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   s.key.dev.chip = AGX_CHIP_G17P;
   emit_copies(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   unsigned first_write;
   EXPECT_EQ(loads_before_first_write(s.body), 1u);
   EXPECT_EQ(max_reads_in_flight(s.body, &first_write), 1u);
   ralloc_free(s.mem);
}

/* M5 compute device loads issue ahead of their uses, as samples do. */
TEST(M5CopySchedule, ComputeDeviceLoadsIssueAheadOfUses)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
   emit_load_uses(&s, 6);
   agx_ilp_schedule(s.b->shader);
   EXPECT_GT(loads_before_first_add(s.body), 1u);
   ralloc_free(s.mem);
}

/* G18G compiles as AGX3C with emit_agx3d and keeps the pressure order. */
TEST(M5CopySchedule, G18GComputeDeviceLoadsKeepTheirOrder)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
   s.key.dev.chip = AGX_CHIP_G17G;
   s.key.dev.emit_agx3d = true;
   emit_load_uses(&s, 6);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(loads_before_first_add(s.body), 1u);
   ralloc_free(s.mem);
}

namespace {

/* n device loads of `channels` words, each consumed right after it, with
 * `live` 32-bit values defined before them and read after the last. */
static void
emit_wide_load_uses(copy_shader *s, unsigned n, unsigned channels,
                    unsigned live)
{
   agx_builder *b = s->b;
   b->cursor = agx_after_block(s->body);
   agx_index address = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index held[64];
   assert(live <= 64);
   for (unsigned i = 0; i < live; ++i)
      held[i] = agx_mov_imm(b, 32, 1000 + i);
   agx_index sum = agx_mov_imm(b, 32, 0);
   for (unsigned i = 0; i < n; ++i) {
      agx_index value = agx_vec_temp(b->shader, AGX_SIZE_32, channels);
      agx_device_load_to(b, value, address, agx_immediate(i * 4 * channels),
                         AGX_FORMAT_I32, BITFIELD_MASK(channels), 0, false);
      agx_index first = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *split = agx_split(b, channels, value);
      split->dest[0] = first;
      for (unsigned c = 1; c < channels; ++c)
         split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
      sum = agx_iadd(b, sum, first, 0);
      for (unsigned c = 1; c < channels; ++c)
         sum = agx_iadd(b, sum, split->dest[c], 0);
   }
   for (unsigned i = 0; i < live; ++i)
      sum = agx_iadd(b, sum, held[i], 0);
   agx_unit_test(b, sum);
}

/* Largest number of device loads issued whose results nothing has read yet */
static unsigned
max_device_loads_in_flight(agx_block *block)
{
   unsigned issued = 0, read = 0, max = 0;
   std::vector<unsigned> pending;
   agx_foreach_instr_in_block(block, I) {
      agx_foreach_ssa_src(I, s) {
         for (unsigned p = 0; p < pending.size(); ++p) {
            if (pending[p] == I->src[s].value) {
               pending.erase(pending.begin() + p);
               ++read;
               break;
            }
         }
      }
      if (I->op == AGX_OPCODE_DEVICE_LOAD) {
         pending.push_back(I->dest[0].value);
         ++issued;
         max = MAX2(max, issued - read);
      }
   }
   return max;
}

} /* namespace */

/* Compute device loads issue ahead of their uses a few at a time (four in
 * flight), not all of a block's loads before the first use. */
TEST(M5CopySchedule, ComputeDeviceLoadsStayBoundedInFlight)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
   emit_wide_load_uses(&s, 16, 1, 0);
   agx_ilp_schedule(s.b->shader);
   const unsigned in_flight = max_device_loads_in_flight(s.body);
   EXPECT_GE(in_flight, 2u);
   EXPECT_LE(in_flight, 4u);
   ralloc_free(s.mem);
}

/* Near the top of an occupancy step there is no room for a vec4 load's
 * result ahead of its use: the loads keep their order. */
TEST(M5CopySchedule, ComputeDeviceLoadsStayInsideTheOccupancyStep)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
   emit_wide_load_uses(&s, 8, 4, 46);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(max_device_loads_in_flight(s.body), 1u);
   ralloc_free(s.mem);
}

/* The headroom rule at its edges: at least 16 halves free below the top of
 * the step enables the windows, the step lookup includes its top, and the
 * last step (256 halves) keeps the issue-order schedule. */
TEST(M5CopySchedule, DeviceLoadCapBoundaries)
{
   EXPECT_EQ(agx_ilp_m5_device_load_cap(0), 96);
   EXPECT_EQ(agx_ilp_m5_device_load_cap(88), 96); /* 16 free below 104 */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(89), 0);  /* 15 free */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(104), 0); /* the top of its step */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(112), 0); /* 112 is in the 112 step */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(113), 0); /* 15 free below 128 */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(216), 224); /* 16 free below 232 */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(217), 0);
   EXPECT_EQ(agx_ilp_m5_device_load_cap(233), 0); /* the last step */
   EXPECT_EQ(agx_ilp_m5_device_load_cap(240), 0);
   EXPECT_EQ(agx_ilp_m5_device_load_cap(256), 0);
}

/* With more values held across vec4 loads, four in flight stop fitting and
 * the schedule takes two, then one: the count only falls, and every window
 * is used. */
TEST(M5CopySchedule, ComputeDeviceLoadWindowsFallBack)
{
   unsigned last = ~0u;
   bool seen[5] = {false};
   for (unsigned live = 0; live <= 50; ++live) {
      SCOPED_TRACE(live);
      copy_shader s;
      copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
      emit_wide_load_uses(&s, 8, 4, live);
      agx_ilp_schedule(s.b->shader);
      const unsigned in_flight = max_device_loads_in_flight(s.body);
      EXPECT_TRUE(in_flight == 4 || in_flight == 2 || in_flight == 1);
      EXPECT_LE(in_flight, last);
      if (in_flight <= 4)
         seen[in_flight] = true;
      last = in_flight;
      ralloc_free(s.mem);
   }
   EXPECT_TRUE(seen[4]);
   EXPECT_TRUE(seen[2]);
   EXPECT_TRUE(seen[1]);
}

/* The program's peak decides, whichever block sets it: a light load block
 * after a block holding many values keeps the previous schedule, which hoists
 * all of its loads ahead of their uses, while the same load block alone takes
 * the in-flight windows. */
TEST(M5CopySchedule, ComputeDeviceLoadsFollowTheProgramPeak)
{
   for (bool heavy : {false, true}) {
      SCOPED_TRACE(heavy);
      copy_shader s;
      copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
      agx_builder *b = s.b;
      if (heavy) {
         agx_index held[46];
         for (unsigned i = 0; i < 46; ++i)
            held[i] = agx_mov_imm(b, 32, 1000 + i);
         agx_index sum = agx_mov_imm(b, 32, 0);
         for (unsigned i = 0; i < 46; ++i)
            sum = agx_iadd(b, sum, held[i], 0);
         agx_unit_test(b, sum);
      }
      agx_block *loads = agx_test_block(b->shader);
      agx_block_add_successor(s.body, loads);
      s.body = loads;
      emit_wide_load_uses(&s, 16, 1, 0);
      agx_ilp_schedule(b->shader);
      const unsigned in_flight = max_device_loads_in_flight(loads);
      if (heavy)
         EXPECT_EQ(in_flight, 16u);
      else
         EXPECT_LE(in_flight, 4u);
      ralloc_free(s.mem);
   }
}
