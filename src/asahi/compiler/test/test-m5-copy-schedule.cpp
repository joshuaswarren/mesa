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

/* Global device memory and threadgroup local memory are disjoint address
 * spaces: a local_store must not serialize independent device_loads behind it,
 * while a barrier still orders both spaces. Applies across AGX2 and AGX3. */
TEST(M5CopySchedule, DeviceLoadsHoistPastLocalStoresAcrossGenerations)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B,
         AGX_ISA_ENCODING_AGX3C}) {
      SCOPED_TRACE(enc);
      copy_shader s;
      copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
      s.key.dev.encoding = enc;
      agx_builder *b = s.b;
      agx_index base = agx_uniform(0, AGX_SIZE_64);
      agx_index off = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index lbase = agx_preload(b, agx_register(4, AGX_SIZE_16));

      for (unsigned i = 0; i < 6; ++i) {
         agx_index idx = agx_iadd(b, off, agx_immediate(i * 4), 0);
         agx_index v = agx_temp(b->shader, AGX_SIZE_32);
         agx_device_load_to(b, v, base, idx, AGX_FORMAT_I32, 0x1, 0, 0);
         agx_index loff = agx_iadd(b, lbase, agx_immediate(i * 4), 0);
         agx_local_store(b, v, loff, agx_zero(), AGX_FORMAT_I32, 0x1);
      }

      agx_ilp_schedule(b->shader);
      EXPECT_GE(max_device_loads_in_flight(s.body), 4u);
      ralloc_free(s.mem);
   }
}

/* Independent threadgroup local_loads batch ahead of their ALU consumers
 * within the shader's occupancy step on both AGX2 and AGX3. */
TEST(M5CopySchedule, ComputeLocalLoadsHoistWithinOccupancyStep)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B,
         AGX_ISA_ENCODING_AGX3C}) {
      SCOPED_TRACE(enc);
      copy_shader s;
      copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
      s.key.dev.encoding = enc;
      agx_builder *b = s.b;
      agx_index lbase = agx_preload(b, agx_register(2, AGX_SIZE_16));
      agx_index bv = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index acc = agx_mov_imm(b, 32, 0);

      for (unsigned i = 0; i < 8; ++i) {
         agx_index loff = agx_iadd(b, lbase, agx_immediate(i * 4), 0);
         agx_index v = agx_temp(b->shader, AGX_SIZE_32);
         agx_local_load_to(b, v, loff, agx_zero(), AGX_FORMAT_I32, 0x1);
         acc = agx_ffma(b, v, bv, acc);
      }
      agx_unit_test(b, acc);

      agx_ilp_schedule(b->shader);
      unsigned leading_local_loads = 0;
      agx_foreach_instr_in_block(s.body, I) {
         if (I->op == AGX_OPCODE_LOCAL_LOAD)
            ++leading_local_loads;
         else if (I->op == AGX_OPCODE_FFMA)
            break;
      }
      EXPECT_GE(leading_local_loads, 4u);
      ralloc_free(s.mem);
   }
}

/* Uniform stores in a preamble must preserve RAW/WAR ordering against uniform
 * reads even when the shader has no AGX3 native uniform ALU instructions. */
TEST(M5CopySchedule, PreambleUniformStoreDependenciesPreserved)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B}) {
      SCOPED_TRACE(enc);
      copy_shader s;
      copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
      s.key.dev.encoding = enc;
      s.b->shader->is_preamble = true;
      agx_builder *b = s.b;

      agx_index in_ptr = agx_uniform(0, AGX_SIZE_64);
      agx_index loaded = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_device_load_to(b, loaded, in_ptr, agx_zero(), AGX_FORMAT_I32, 0x3, 0, 0);
      agx_uniform_store(b, loaded, agx_immediate(16), 0x3);

      agx_index dep_ptr = agx_uniform(16, AGX_SIZE_64);
      agx_index loaded2 = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, loaded2, dep_ptr, agx_zero(), AGX_FORMAT_I32, 0x1, 0, 0);
      agx_uniform_store(b, loaded2, agx_immediate(24), 0x1);

      agx_ilp_schedule(b->shader);

      int store_pos = -1, dep_load_pos = -1, idx = 0;
      agx_foreach_instr_in_block(s.body, I) {
         if (I->op == AGX_OPCODE_UNIFORM_STORE && I->src[1].value == 16)
            store_pos = idx;
         if (I->op == AGX_OPCODE_DEVICE_LOAD &&
             I->src[0].type == AGX_INDEX_UNIFORM && I->src[0].value == 16)
            dep_load_pos = idx;
         ++idx;
      }
      ASSERT_GE(store_pos, 0);
      ASSERT_GE(dep_load_pos, 0);
      EXPECT_LT(store_pos, dep_load_pos);
      ralloc_free(s.mem);
   }
}

namespace {

enum frag_sample_kind {
   FRAG_SAMPLE_TEX,
   FRAG_SAMPLE_IMAGE_LOAD,
   FRAG_SAMPLE_TEX_LOAD,
};

enum frag_desc_kind {
   FRAG_DESC_IMM,
   FRAG_DESC_UNIFORM,
   FRAG_DESC_REG,
};

static unsigned
emit_frag_samples_and_count(enum agx_isa_encoding enc, enum agx_chip chip,
                            enum agx_dim dim, enum frag_sample_kind kind,
                            enum frag_desc_kind desc_kind,
                            enum agx_lod_mode lod_mode, bool reg_lod,
                            bool with_cube_array, unsigned prefix_live_32)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_FRAGMENT, false);
   s.key.dev.encoding = enc;
   s.key.dev.chip = chip;

   agx_builder *b = s.b;
   if (prefix_live_32 > 0) {
      /* Heavy entry block setting program_peak, followed by a light sample
       * block so the program's occupancy/budget ceiling directly controls
       * whether the light block's 5 vec4 samples may hoist. */
      agx_index held[96];
      assert(prefix_live_32 <= 96);
      for (unsigned i = 0; i < prefix_live_32; ++i)
         held[i] = agx_mov_imm(b, 32, 1000 + i);
      agx_index psum = agx_mov_imm(b, 32, 0);
      for (unsigned i = 0; i < prefix_live_32; ++i)
         psum = agx_iadd(b, psum, held[i], 0);
      agx_unit_test(b, psum);

      agx_block *samp_blk = agx_test_block(b->shader);
      agx_block_add_successor(s.body, samp_blk);
      s.body = samp_blk;
   }

   b->cursor = agx_after_block(s.body);
   agx_index uv = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index reg_desc = agx_preload(b, agx_register(6, AGX_SIZE_64));
   agx_index reg_smp = agx_preload(b, agx_register(10, AGX_SIZE_64));
   agx_index reg_lod_val = agx_preload(b, agx_register(14, AGX_SIZE_32));
   agx_index acc = agx_mov_imm(b, 32, 0);

   if (with_cube_array) {
      /* Place a single cube-array sample in a predecessor block so the shader
       * carries the 112-half cube-array register budget while the sample block
       * holds 5 relocatable 2D samples. */
      agx_index c1 = agx_mov_imm(b, 32, 1);
      agx_index c2 = agx_mov_imm(b, 32, 2);
      agx_index c3 = agx_mov_imm(b, 32, 3);
      agx_index ccoords = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *cc = agx_collect_to(b, ccoords, 4);
      cc->src[0] = uv;
      cc->src[1] = c1;
      cc->src[2] = c2;
      cc->src[3] = c3;
      agx_index ctex = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_sample_to(
         b, ctex, ccoords, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_CUBE_ARRAY,
         AGX_LOD_MODE_AUTO_LOD, 0xf, false, false, false, AGX_GATHER_NONE,
         false);
      agx_instr *csp = agx_split(b, 4, ctex);
      for (unsigned k = 0; k < 4; ++k)
         csp->dest[k] = agx_temp(b->shader, AGX_SIZE_32);
      agx_unit_test(b, csp->dest[0]);

      agx_block *samp_blk = agx_test_block(b->shader);
      agx_block_add_successor(s.body, samp_blk);
      s.body = samp_blk;
      b->cursor = agx_after_block(s.body);
   }

   const unsigned coord_comps =
      (dim == AGX_DIM_CUBE_ARRAY) ? 4
      : (dim == AGX_DIM_2D_ARRAY || dim == AGX_DIM_3D || dim == AGX_DIM_CUBE ||
         dim == AGX_DIM_2D_MS)
         ? 3
         : 2;
   for (unsigned i = 0; i < 5; ++i) {
      agx_index csrc[4];
      csrc[0] = uv;
      for (unsigned k = 1; k < coord_comps; ++k)
         csrc[k] = agx_mov_imm(b, 32, i + k);
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, coord_comps);
      agx_instr *c = agx_collect_to(b, coords, coord_comps);
      for (unsigned k = 0; k < coord_comps; ++k)
         c->src[k] = csrc[k];

      agx_index lod = agx_immediate(0);
      if (lod_mode == AGX_LOD_MODE_LOD_GRAD) {
         agx_index gsrc[4];
         for (unsigned k = 0; k < 4; ++k)
            gsrc[k] = agx_mov_imm(b, 32, k + 1);
         lod = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
         agx_instr *gc = agx_collect_to(b, lod, 4);
         for (unsigned k = 0; k < 4; ++k)
            gc->src[k] = gsrc[k];
      } else if (reg_lod) {
         lod = reg_lod_val;
      }

      agx_index tex = (desc_kind == FRAG_DESC_UNIFORM)
                         ? agx_uniform(4, AGX_SIZE_64)
                         : (desc_kind == FRAG_DESC_REG) ? reg_desc
                                                        : agx_immediate(i);
      agx_index smp = (desc_kind == FRAG_DESC_UNIFORM)
                         ? agx_uniform(8, AGX_SIZE_64)
                         : (desc_kind == FRAG_DESC_REG) ? reg_smp
                                                        : agx_immediate(0);

      agx_index texel = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      if (kind == FRAG_SAMPLE_IMAGE_LOAD) {
         agx_instr *ld = agx_image_load_to(
            b, texel, coords, lod, agx_zero(), tex, agx_immediate(0),
            agx_null(), dim, lod_mode, 0, false, false, false);
         ld->mask = 0xf;
      } else if (kind == FRAG_SAMPLE_TEX_LOAD) {
         agx_texture_load_to(
            b, texel, coords, lod, agx_zero(), tex, agx_immediate(0),
            agx_null(), dim, lod_mode, 0xf, false, false);
      } else {
         agx_texture_sample_to(
            b, texel, coords, lod, agx_immediate(0), tex, smp, agx_null(), dim,
            lod_mode, 0xf, false, false, false, AGX_GATHER_NONE, false);
      }

      agx_instr *sp = agx_split(b, 4, texel);
      for (unsigned k = 0; k < 4; ++k) {
         agx_index ch = agx_temp(b->shader, AGX_SIZE_32);
         sp->dest[k] = ch;
         acc = agx_fadd(b, acc, ch);
      }
   }
   agx_unit_test(b, acc);
   agx_validate(b->shader, "agx3b fragment schedule test");

   agx_ilp_schedule(b->shader);

   unsigned samples_before_fadd = 0;
   agx_foreach_instr_in_block(s.body, I) {
      if (I->op == AGX_OPCODE_FADD)
         break;
      samples_before_fadd +=
         (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
          I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD);
   }
   ralloc_free(s.mem);
   return samples_before_fadd;
}

} /* namespace */

/* AGX3B fragment shaders with >=4 multi-channel relocatable samples hoist all
 * samples ahead of their consumers when the growth fits within the 1024-thread
 * occupancy bucket (104 halves), even when it exceeds the fixed 16-half slack. */
TEST(M5CopySchedule, AGX3BFragmentSamplesUseOccupancySlack)
{
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                false, 0),
             5u);
   /* Uniform-descriptor 2D_ARRAY samples allocate disjoint staging banks on
    * AGX3B (relocatable_2d_array) and hoist inside the occupancy bucket. */
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D_ARRAY,
                FRAG_SAMPLE_TEX, FRAG_DESC_UNIFORM, AGX_LOD_MODE_AUTO_LOD,
                false, false, 0),
             5u);
}

/* AGX3B fragment shaders recognize relocatable AGX_OPCODE_IMAGE_LOAD (2D and
 * immediate-LOD 2D_ARRAY) as native SMP samples and hoist them within the
 * occupancy step. */
TEST(M5CopySchedule, AGX3BFragmentImageLoadsHoist)
{
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_IMAGE_LOAD, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD,
                false, false, 0),
             5u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, AGX_DIM_2D_ARRAY,
                FRAG_SAMPLE_IMAGE_LOAD, FRAG_DESC_REG, AGX_LOD_MODE_AUTO_LOD,
                false, false, 0),
             5u);
}

/* Non-relocatable AGX3B fragment samples stay pinned to staging bank 0 in
 * agx3_pack_binary.c, so hoisting them would only inflate GPRs and emit
 * serializing wait barriers. Keep them in pressure order. */
TEST(M5CopySchedule, AGX3BFragmentNonRelocatableSamplesStayInOrder)
{
   /* Register-descriptor 2D_ARRAY samples (e.g. G16G without preamble or
    * bindless 2D_ARRAY) are non-relocatable on AGX3B. */
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, AGX_DIM_2D_ARRAY,
                FRAG_SAMPLE_TEX, FRAG_DESC_REG, AGX_LOD_MODE_AUTO_LOD, false,
                false, 0),
             1u);
   /* 3D, cube-array, gradient-LOD samples, multisample image loads, and
    * register-LOD texture loads also keep staging bank 0 on AGX3B. */
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_3D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                false, 0),
             1u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_CUBE_ARRAY,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                false, 0),
             1u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_LOD_GRAD, false,
                false, 0),
             1u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D_MS,
                FRAG_SAMPLE_IMAGE_LOAD, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD,
                false, false, 0),
             1u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX_LOAD, FRAG_DESC_IMM, AGX_LOD_MODE_LOD_MIN, true,
                false, 0),
             1u);
}

/* AGX3B fragment occupancy cap stops at 184 halves (highest step in the
 * 192-half file): a program at ~160 halves may still hoist a light block
 * inside the 184-half step, while one past 176 halves falls back to fixed
 * 16-half slack and rejects the 5-sample hoist. */
TEST(M5CopySchedule, AGX3BFragmentRespects184HalfCeiling)
{
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                false, 80),
             5u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                false, 90),
             1u);
}

/* AGX3B shaders with a cube-array sample carry a 112-half register budget:
 * relocatable 2D samples in the same shader may hoist below 104 halves, but
 * are refused when the program already exceeds the 112-half ceiling. */
TEST(M5CopySchedule, AGX3BFragmentCubeBudgetRespected)
{
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                true, 0),
             5u);
   EXPECT_EQ(emit_frag_samples_and_count(
                AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, AGX_DIM_2D,
                FRAG_SAMPLE_TEX, FRAG_DESC_IMM, AGX_LOD_MODE_AUTO_LOD, false,
                true, 80),
             1u);
}

/* Helper-kernel protocol ops (doorbell, stack_map, stack_unmap) must carry
 * AGX_SCHEDULE_CLASS_BARRIER and never be reordered past get_sr_barrier inside
 * AGX2 loop bodies (libagx_helper deadlock regression). */
TEST(M5CopySchedule, AGX2HelperDoorbellOrderPreserved)
{
   EXPECT_EQ(agx_opcodes_info[AGX_OPCODE_DOORBELL].schedule_class,
             AGX_SCHEDULE_CLASS_BARRIER);
   EXPECT_EQ(agx_opcodes_info[AGX_OPCODE_STACK_MAP].schedule_class,
             AGX_SCHEDULE_CLASS_BARRIER);
   EXPECT_EQ(agx_opcodes_info[AGX_OPCODE_STACK_UNMAP].schedule_class,
             AGX_SCHEDULE_CLASS_BARRIER);

   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_COMPUTE, true);
   s.key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   s.key.dev.chip = AGX_CHIP_G13G;

   agx_builder *b = s.b;
   b->cursor = agx_after_block(s.body);
   agx_doorbell(b, 0x20);
   agx_index op = agx_get_sr_barrier(b, 32, (enum agx_sr)0x90);
   agx_index arg = agx_get_sr_barrier(b, 32, (enum agx_sr)0x92);
   agx_index sum = agx_iadd(b, op, arg, 0);
   agx_if_icmp(b, sum, agx_zero(), 1, AGX_ICOND_UEQ, false, s.body);

   agx_pressure_schedule(b->shader);
   agx_ilp_schedule(b->shader);

   int doorbell_pos = -1, first_sr_pos = -1, pos = 0;
   agx_foreach_instr_in_block(s.body, I) {
      if (I->op == AGX_OPCODE_DOORBELL)
         doorbell_pos = pos;
      else if (I->op == AGX_OPCODE_GET_SR_BARRIER && first_sr_pos < 0)
         first_sr_pos = pos;
      ++pos;
   }
   EXPECT_GE(doorbell_pos, 0);
   EXPECT_GT(first_sr_pos, doorbell_pos);
   ralloc_free(s.mem);
}

/* stack_adjust must live in SCHED_MEM_STACK so stack_load/stack_store stay
 * strictly ordered between frame allocation and deallocation. */
TEST(M5CopySchedule, StackAdjustSerializesStackAccesses)
{
   copy_shader s;
   copy_shader_init(&s, MESA_SHADER_COMPUTE, false);
   s.key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   s.key.dev.chip = AGX_CHIP_G13G;

   agx_builder *b = s.b;
   b->cursor = agx_after_block(s.body);
   agx_index val = agx_mov_imm(b, 32, 0x1234);
   agx_index off = agx_zero();
   agx_stack_adjust(b, 16);
   agx_stack_store(b, val, off, AGX_FORMAT_I32, 1);
   agx_index loaded = agx_stack_load(b, off, AGX_FORMAT_I32, 1);
   agx_stack_adjust(b, -16);
   agx_device_store(b, loaded, agx_uniform(0, AGX_SIZE_64), agx_zero(),
                    AGX_FORMAT_I32, 1, 0, false);

   agx_pressure_schedule(b->shader);
   agx_ilp_schedule(b->shader);

   int adj_alloc = -1, st_pos = -1, ld_pos = -1, adj_free = -1, pos = 0;
   agx_foreach_instr_in_block(s.body, I) {
      if (I->op == AGX_OPCODE_STACK_ADJUST) {
         if (adj_alloc < 0)
            adj_alloc = pos;
         else
            adj_free = pos;
      } else if (I->op == AGX_OPCODE_STACK_STORE) {
         st_pos = pos;
      } else if (I->op == AGX_OPCODE_STACK_LOAD) {
         ld_pos = pos;
      }
      ++pos;
   }
   EXPECT_GE(adj_alloc, 0);
   EXPECT_GT(st_pos, adj_alloc);
   EXPECT_GT(ld_pos, st_pos);
   EXPECT_GT(adj_free, ld_pos);
   ralloc_free(s.mem);
}
