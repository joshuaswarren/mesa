/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "util/bitset.h"

extern "C" signed agx2_disassemble_instr(BITSET_WORD *code, FILE *fp,
                                         unsigned offset, size_t available,
                                         bool verbose);

/* Apple's Metal toolchain output for the scoreprobe multiply kernel on
 * applegpu_g14g (M2 Max class), the AGX2 encoding. The checked-in fixture is
 * _agc.main (30 bytes); native/g14g/manifest.json also documents the full
 * 94-byte __TEXT,__text layout -- a 64-byte _agc.main.constant_program ahead
 * of _agc.main at offset 0x40 -- and the test decodes from the right offset
 * for whichever form is on disk. The expected decode comes from Apple's own
 * bytes, recorded in the manifest, not from Mesa's tables.
 */

static const char *kWitnessDir = "src/asahi/compiler/test/native/g14g";
static const char *kWitnessName =
   "probe.scoreprobe-reference.scoreprobe.g14g.bin";

static constexpr unsigned kConstantProgramSize = 0x40;

struct expected_instr {
   unsigned size;
   const char *asm_text;
};

static const expected_instr kExpected[] = {
   {4, "mov      r1, thread_in_grid_x"},
   {8, "load     r0, du0, r1, i32, x, a"},
   {2, "wait     a"},
   {6, "fmul     r0, ^r0, 2.0"},
   {8, "store    r0, du2, r1, i32, x, a"},
   {2, "stop"},
};

static constexpr unsigned kMainSize = 30;

static std::vector<uint8_t>
read_fixture()
{
   std::string path = std::string(kWitnessDir) + "/" + kWitnessName;
   const char *root = getenv("MESON_SOURCE_ROOT");
   std::string full = root ? (std::string(root) + "/" + path) : path;
   FILE *f = fopen(full.c_str(), "rb");
   if (!f) {
      fprintf(stderr, "g14g witness fixture missing: %s\n", full.c_str());
      return {};
   }
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);
   std::vector<uint8_t> buf(size > 0 ? size : 0);
   size_t got = fread(buf.data(), 1, buf.size(), f);
   fclose(f);
   buf.resize(got);
   return buf;
}

TEST(AGX2G14GWitness, ScoreprobeDecode)
{
   std::vector<uint8_t> text = read_fixture();
   ASSERT_GE(text.size(), (size_t)kMainSize)
      << "fixture must hold at least _agc.main; see "
         "native/g14g/manifest.json";

   /* The full section carries the constant program ahead of _agc.main; the
    * checked-in fixture may hold _agc.main alone. Decode from the symbol
    * offset for whichever form is on disk.
    */
   unsigned entry = text.size() - kMainSize;
   if (entry != 0)
      EXPECT_EQ(entry, kConstantProgramSize)
         << "unexpected fixture layout: " << text.size() << " bytes";

   unsigned offset = entry;
   for (const expected_instr &exp : kExpected) {
      char *out = NULL;
      size_t out_size = 0;
      FILE *fp = open_memstream(&out, &out_size);
      signed ret = agx2_disassemble_instr(
         (BITSET_WORD *)(text.data() + offset), fp, offset,
         text.size() - offset, false);
      fclose(fp);
      std::string asm_text = out ? out : "";
      free(out);
      while (!asm_text.empty() &&
             (asm_text.back() == '\n' || asm_text.back() == ' '))
         asm_text.pop_back();

      EXPECT_EQ(ret, (signed)exp.size)
         << "instruction length at 0x" << std::hex << offset;
      EXPECT_EQ(asm_text, exp.asm_text)
         << "at 0x" << std::hex << offset;
      offset += exp.size;
   }
   EXPECT_EQ(offset, text.size()) << "_agc.main must end the section";
}
