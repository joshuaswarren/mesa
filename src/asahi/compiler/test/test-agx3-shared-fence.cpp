/* SPDX-License-Identifier: MIT */
/*
 * A shared-memory fence without workgroup execution scope must not become the
 * AGX3 workgroup rendezvous: memoryBarrierShared() in divergent control flow
 * otherwise counts arrivals the rest of the workgroup never makes. It takes
 * the threadgroup-memory fence instead. The cases compile compute shaders
 * through selection, scheduling and encoding and count barrier encodings in
 * the binary; the fence's fields are checked against the witnessed
 * barrier/fence pairs by the table decoder.
 */
#include <cstring>
#include <map>
#include <gtest/gtest.h>
#include "agx3_pack.h"
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decode.h"
}

namespace {

struct counts {
   int threadgroup, execution, device_fence, threadgroup_fence;
};

enum shape {
   /* if (i < 7) { shared[i] = i; fence; shared[64 + i] = 1; } */
   DIVERGENT_FENCE,
   /* shared[i] = i; barrier(); shared[64 + i] = shared[i ^ 1]; */
   WORKGROUP_BARRIER,
};

int
occurrences(const agx_shader_part &part, enum agx_isa_encoding encoding,
            enum agx3_barrier_scope scope)
{
   struct agx3_encoded_scoped_barrier bar;
   if (!agx3_pack_scoped_barrier(encoding, scope, &bar) || !bar.length)
      return -1;
   const uint8_t *code = static_cast<const uint8_t *>(part.binary);
   int n = 0;
   for (unsigned i = part.info.main_offset;
        i + bar.length <= part.info.main_offset + part.info.main_size; ++i)
      n += !memcmp(code + i, bar.bytes, bar.length);
   return n;
}

counts
compile(enum agx_chip chip, enum agx_isa_encoding encoding, enum shape shape,
        mesa_scope execution, mesa_scope memory, nir_variable_mode modes)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared fence");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 512;

   nir_def *i = nir_load_local_invocation_index(&b);
   nir_def *offset = nir_ishl_imm(&b, i, 2);
   nir_if *nif = NULL;
   if (shape == DIVERGENT_FENCE)
      nif = nir_push_if(&b, nir_ult_imm(&b, i, 7));

   nir_store_shared(&b, i, offset, .align_mul = 4);
   nir_barrier(&b, .execution_scope = execution, .memory_scope = memory,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = modes);
   nir_def *value =
      shape == DIVERGENT_FENCE
         ? nir_imm_int(&b, 1)
         : nir_load_shared(&b, 1, 32,
                           nir_ishl_imm(&b, nir_ixor(&b, i, nir_imm_int(&b, 1)),
                                        2),
                           .align_mul = 4);
   nir_store_shared(&b, value, nir_iadd_imm(&b, offset, 256), .align_mul = 4);

   if (nif)
      nir_pop_if(&b, nif);

   agx_preprocess_nir_native(b.shader, chip);
   agx_shader_key key = {};
   key.dev.chip = chip;
   key.dev.encoding = encoding;
   key.reserved_preamble = 8;
   key.no_preamble = true;
   agx_shader_part part = {};
   counts c = {-1, -1, -1, -1};
   if (agx_compile_shader_nir(b.shader, &key, &part) && part.binary) {
      c.threadgroup =
         occurrences(part, encoding, AGX3_BARRIER_SCOPE_THREADGROUP);
      c.execution = occurrences(part, encoding, AGX3_BARRIER_SCOPE_EXECUTION);
      c.device_fence =
         occurrences(part, encoding, AGX3_BARRIER_SCOPE_DEVICE_FENCE);
      c.threadgroup_fence =
         occurrences(part, encoding, AGX3_BARRIER_SCOPE_THREADGROUP_FENCE);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return c;
}

const struct {
   enum agx_chip chip;
   enum agx_isa_encoding encoding;
} targets[] = {
   {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B},
   {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B},
   {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C},
};

/* The nonzero fields of one 6-byte barrier word. */
std::map<unsigned, uint64_t>
fields(enum agx_isa_encoding encoding, const uint8_t *bytes)
{
   const agx3_table *t =
      encoding == AGX_ISA_ENCODING_AGX3B ? &agx3b_table : &agx3c_table;
   uint64_t f[AGX3_MAX_FIELDS] = {0};
   std::map<unsigned, uint64_t> m;
   if (agx3_decode(t, bytes, 6, AGX3_MN_barrier, f) < 0)
      return m;
   for (unsigned i = 0; i < AGX3_MAX_FIELDS; ++i) {
      if (f[i])
         m[i] = f[i];
   }
   return m;
}

std::map<unsigned, uint64_t>
scope_fields(enum agx_isa_encoding encoding, enum agx3_barrier_scope scope,
             unsigned word = 0)
{
   struct agx3_encoded_scoped_barrier bar;
   if (!agx3_pack_scoped_barrier(encoding, scope, &bar) ||
       bar.length < 6 * (word + 1))
      return {};
   return fields(encoding, bar.bytes + 6 * word);
}

/* A barrier's memory-only fence: f29 (rendezvous) cleared, f40 set. */
std::map<unsigned, uint64_t>
as_fence(std::map<unsigned, uint64_t> m)
{
   m.erase(29);
   m[40] = 1;
   return m;
}

} /* namespace */

/* memoryBarrierShared(): no execution scope. At workgroup memory scope and
 * wider it is the threadgroup-memory fence, never a rendezvous; within a
 * subgroup it needs no instruction. */
TEST(AGX3SharedFence, MemoryOnlyFenceIsNoRendezvous)
{
   for (const auto &t : targets) {
      for (mesa_scope memory : {SCOPE_SUBGROUP, SCOPE_WORKGROUP,
                                SCOPE_QUEUE_FAMILY, SCOPE_DEVICE}) {
         SCOPED_TRACE(::testing::Message()
                      << "chip=" << t.chip << " memory=" << memory);
         const counts c = compile(t.chip, t.encoding, DIVERGENT_FENCE,
                                  SCOPE_NONE, memory, nir_var_mem_shared);
         EXPECT_EQ(c.threadgroup, 0);
         EXPECT_EQ(c.execution, 0);
         EXPECT_EQ(c.threadgroup_fence, memory == SCOPE_SUBGROUP ? 0 : 1);
      }
   }
}

/* memoryBarrier(): every mode, no execution scope. Each memory class takes
 * its memory-only fence and nothing takes the rendezvous. */
TEST(AGX3SharedFence, MemoryBarrierAllModesIsNoRendezvous)
{
   const nir_variable_mode all = (nir_variable_mode)(
      nir_var_mem_shared | nir_var_mem_global | nir_var_image);
   for (const auto &t : targets) {
      SCOPED_TRACE(::testing::Message() << "chip=" << t.chip);
      const counts c = compile(t.chip, t.encoding, DIVERGENT_FENCE,
                               SCOPE_NONE, SCOPE_DEVICE, all);
      EXPECT_EQ(c.threadgroup, 0);
      EXPECT_EQ(c.execution, 0);
      EXPECT_GE(c.device_fence, 1);
      EXPECT_EQ(c.threadgroup_fence, 1);
   }
}

/* barrier() keeps the workgroup rendezvous with its shared-memory class. */
TEST(AGX3SharedFence, WorkgroupBarrierKeepsRendezvous)
{
   for (const auto &t : targets) {
      SCOPED_TRACE(::testing::Message() << "chip=" << t.chip);
      const counts c = compile(t.chip, t.encoding, WORKGROUP_BARRIER,
                               SCOPE_WORKGROUP, SCOPE_WORKGROUP,
                               nir_var_mem_shared);
      EXPECT_EQ(c.threadgroup, 1);
      EXPECT_EQ(c.execution, 0);
      EXPECT_EQ(c.threadgroup_fence, 0);
   }
}

/* The witnessed fences are their barriers with f29 cleared and f40 set; the
 * threadgroup-memory fence is the threadgroup barrier under the same rule. */
TEST(AGX3SharedFence, FenceIsTheBarrierWithoutRendezvous)
{
   for (auto enc : {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      SCOPED_TRACE(::testing::Message() << "encoding=" << enc);
      const auto device = scope_fields(enc, AGX3_BARRIER_SCOPE_DEVICE);
      const auto threadgroup =
         scope_fields(enc, AGX3_BARRIER_SCOPE_THREADGROUP);
      ASSERT_FALSE(device.empty());
      ASSERT_FALSE(threadgroup.empty());
      EXPECT_EQ(device.at(29), 1u);
      EXPECT_EQ(threadgroup.at(29), 1u);
      EXPECT_EQ(threadgroup.at(26), 1u);
      EXPECT_EQ(threadgroup.at(39), 1u);
      EXPECT_EQ(as_fence(device),
                scope_fields(enc, AGX3_BARRIER_SCOPE_DEVICE_FENCE));
      EXPECT_EQ(as_fence(threadgroup),
                scope_fields(enc, AGX3_BARRIER_SCOPE_THREADGROUP_FENCE));
   }

   /* AGX3C's threadgroup-scope texture fence, both words. */
   for (unsigned w = 0; w < 2; ++w) {
      EXPECT_EQ(as_fence(scope_fields(AGX_ISA_ENCODING_AGX3C,
                                      AGX3_BARRIER_SCOPE_TEXTURE, w)),
                scope_fields(AGX_ISA_ENCODING_AGX3C,
                             AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP, w));
   }

   const struct {
      enum agx_isa_encoding encoding;
      uint8_t bytes[6];
   } rows[] = {
      {AGX_ISA_ENCODING_AGX3B, {0x07, 0x04, 0x54, 0x60, 0x0b, 0x00}},
      {AGX_ISA_ENCODING_AGX3C, {0x27, 0x13, 0x00, 0x06, 0x00, 0x00}},
   };
   for (const auto &r : rows) {
      struct agx3_encoded_scoped_barrier bar;
      ASSERT_TRUE(agx3_pack_scoped_barrier(
         r.encoding, AGX3_BARRIER_SCOPE_THREADGROUP_FENCE, &bar));
      ASSERT_EQ(bar.length, 6u);
      EXPECT_EQ(memcmp(bar.bytes, r.bytes, 6), 0) << r.encoding;
   }
}
