/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

/* AGX3 multi-component 8-bit loads (device, shared and tile memory) return
 * their bytes packed two to a half register, and agx_unpack_integer_8_agx3
 * gives each byte its own half. AGX3B and AGX3C do that in one 16-bit
 * operation per byte: a mask for an even byte, a right shift by 8 for an odd
 * one. AGX3B's half shift takes a register count, shared by every odd byte.
 *
 * Growing the payload of a compute shader that loads and uses N bytes by
 * one byte therefore costs AGX3B no more than AGX3C, for each payload width
 * and both kinds of consumer. The 32-bit form (a widen, a 32-bit shift, a
 * re-narrow and a mask per byte) is still what G18G uses, and it must still
 * compile.
 */

namespace {

enum mem {
   MEM_GLOBAL,
   MEM_SHARED,
};

enum use {
   USE_FLOAT, /* u2f32 of each byte, as a packed RGBA8 pixel is read */
   USE_INT,   /* the bytes zero-extended and summed as integers */
};

struct target {
   agx_isa_encoding encoding;
   agx_chip chip;
   bool emit_agx3d;
   const char *name;
};

static const target agx3b_targets[] = {
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G15G, false, "G15G"},
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, false, "G16G"},
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, "G17P"},
};

static const target agx3c = {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false,
                             "G17C"};
static const target g18 = {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true,
                           "G18G"};

static void
set_align(nir_intrinsic_instr *intr, unsigned align_mul)
{
   nir_intrinsic_set_align(intr, align_mul, 0);
}

static void
workgroup_barrier(nir_builder *b)
{
   nir_intrinsic_instr *bar =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_barrier);
   nir_intrinsic_set_execution_scope(bar, SCOPE_WORKGROUP);
   nir_intrinsic_set_memory_scope(bar, SCOPE_WORKGROUP);
   nir_intrinsic_set_memory_semantics(bar, NIR_MEMORY_ACQ_REL);
   nir_intrinsic_set_memory_modes(bar, nir_var_mem_shared);
   nir_builder_instr_insert(b, &bar->instr);
}

/* Loads `nr` bytes per invocation (from global memory, or from shared
 * memory a 32-bit store filled), uses each byte, and stores the results. */
static unsigned
compile_bytes(const target &t, enum mem mem, enum use use, unsigned nr,
              bool *ok)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "bytes");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *id = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *offs = nir_imul_imm(&b, id, 4);
   nir_def *bytes;
   if (mem == MEM_GLOBAL) {
      nir_def *src = nir_iadd(&b, addr, nir_u2u64(&b, offs));
      bytes = nir_load_global(&b, nr, 8, src);
      set_align(nir_def_as_intrinsic(bytes), 4);
      nir_intrinsic_set_access(nir_def_as_intrinsic(bytes),
                               ACCESS_NON_WRITEABLE);
   } else {
      b.shader->info.shared_size = 32 * 4;
      nir_def *word =
         nir_load_global(&b, 1, 32, nir_iadd(&b, addr, nir_u2u64(&b, offs)));
      set_align(nir_def_as_intrinsic(word), 4);
      set_align(nir_store_shared(&b, word, offs), 4);
      workgroup_barrier(&b);
      nir_def *other = nir_iand_imm(&b, nir_iadd_imm(&b, offs, 4), 127);
      bytes = nir_load_shared(&b, nr, 8, other);
      set_align(nir_def_as_intrinsic(bytes), 4);
   }

   nir_def *out[4];
   for (unsigned i = 0; i < nr; ++i) {
      nir_def *ch = nir_channel(&b, bytes, i);
      out[i] = use == USE_FLOAT ? nir_u2f32(&b, ch) : nir_u2u32(&b, ch);
   }
   nir_def *dst = nir_iadd_imm(&b, nir_iadd(&b, addr, nir_u2u64(&b,
                               nir_imul_imm(&b, id, 16))), 4096);
   if (use == USE_FLOAT) {
      set_align(nir_store_global(&b, nir_vec(&b, out, nr), dst), 16);
   } else {
      nir_def *sum = out[0];
      for (unsigned i = 1; i < nr; ++i)
         sum = nir_iadd(&b, nir_imul_imm(&b, sum, 3), out[i]);
      set_align(nir_store_global(&b, sum, dst), 4);
   }

   agx_shader_key key = {};
   key.dev.encoding = t.encoding;
   key.dev.chip = t.chip;
   key.dev.emit_agx3d = t.emit_agx3d;
   key.no_preamble = true;
   key.reserved_preamble = 16;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   *ok = agx_compile_shader_nir(b.shader, &key, &part);
   const unsigned instrs = part.info.stats.instrs;
   free(part.binary);
   ralloc_free(b.shader);
   return instrs;
}

static const char *
mem_name(enum mem m)
{
   return m == MEM_GLOBAL ? "global" : "shared";
}

static const char *
use_name(enum use u)
{
   return u == USE_FLOAT ? "u2f32" : "u2u32";
}

} /* namespace */

TEST(AGX3BByteUnpack, CompilesOnEveryTarget)
{
   for (enum mem m : {MEM_GLOBAL, MEM_SHARED}) {
      for (enum use u : {USE_FLOAT, USE_INT}) {
         for (unsigned nr : {1u, 2u, 3u, 4u}) {
            bool ok = false;
            for (const target &t : agx3b_targets) {
               compile_bytes(t, m, u, nr, &ok);
               EXPECT_TRUE(ok) << t.name << " " << mem_name(m) << " "
                               << use_name(u) << " u8x" << nr;
            }
            compile_bytes(agx3c, m, u, nr, &ok);
            EXPECT_TRUE(ok) << "G17C " << mem_name(m) << " u8x" << nr;
            compile_bytes(g18, m, u, nr, &ok);
            EXPECT_TRUE(ok) << "G18G " << mem_name(m) << " u8x" << nr;
         }
      }
   }
}

/* Every AGX3B part shares the unpack, so they agree with one another. Past
 * the second byte, each byte a payload grows by costs AGX3B no more than
 * AGX3C, whatever else its consumer costs: one half operation, the shift of
 * an odd byte reusing the count. (The second byte is not compared: it pays
 * for that count, a mov_imm and the add carrying it, and AGX3B keeps the
 * even-byte mask that AGX3C's native byte conversion absorbs. The 32-bit
 * form cost the same there.) The 32-bit form paid a widen and a re-narrow
 * for every further byte, which the u8x3 -> u8x4 step catches. */
TEST(AGX3BByteUnpack, OneHalfOperationPerByte)
{
   for (enum mem m : {MEM_GLOBAL, MEM_SHARED}) {
      for (enum use u : {USE_FLOAT, USE_INT}) {
         bool ok = false;
         unsigned b[5] = {0}, c[5] = {0};
         for (unsigned nr = 1; nr <= 4; ++nr) {
            for (unsigned i = 0; i < ARRAY_SIZE(agx3b_targets); ++i) {
               const unsigned n = compile_bytes(agx3b_targets[i], m, u, nr, &ok);
               ASSERT_TRUE(ok);
               if (i == 0)
                  b[nr] = n;
               EXPECT_EQ(n, b[nr]) << agx3b_targets[i].name << " u8x" << nr;
            }
            c[nr] = compile_bytes(agx3c, m, u, nr, &ok);
            ASSERT_TRUE(ok);
         }

         printf("%s %s: AGX3B u8x1..4 %u %u %u %u, AGX3C %u %u %u %u\n",
                mem_name(m), use_name(u), b[1], b[2], b[3], b[4], c[1], c[2],
                c[3], c[4]);
         const char *what = use_name(u);
         EXPECT_LE(b[3] - b[2], c[3] - c[2])
            << mem_name(m) << " " << what << ": even byte";
         EXPECT_LE(b[4] - b[3], c[4] - c[3])
            << mem_name(m) << " " << what << ": odd byte, shared count";
      }
   }
}
