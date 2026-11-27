/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
/* Field-by-field comparison of our compiler's output against an Apple
 * Metal witness (src/asahi/isa/README-metal-witness.md). The fixture is
 * Apple's 94-byte `__TEXT,__text` for a multiply kernel on applegpu_g14g:
 * the 0x40-byte constant program, then the 30-byte entry program. We build
 * the equivalent NIR, compile it through the AGX2 path and report, per
 * instruction and per program field, what is established and what is not.
 * Only exact round trips are facts, so the report names bytes and decoded
 * text and marks everything else unestablished rather than guessing.
 */
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "asahi/isa/disasm.h"
}

namespace {

/* Same convention as test-agx2-g14g-witness.cpp: MESON_SOURCE_ROOT is set
 * for `meson test` runs. */
std::string
fixture_path(const char *name)
{
   const char *root = getenv("MESON_SOURCE_ROOT");
   return (root ? std::string(root) : ".") +
          "/src/asahi/compiler/test/native/g14g/" + name;
}

std::vector<uint8_t>
read_file(const std::string &path)
{
   FILE *fp = fopen(path.c_str(), "rb");
   EXPECT_NE(fp, nullptr) << "cannot open fixture " << path;
   if (!fp)
      return {};
   std::vector<uint8_t> out;
   uint8_t chunk[4096];
   size_t n;
   while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0)
      out.insert(out.end(), chunk, chunk + n);
   fclose(fp);
   return out;
}

struct Decoded {
   unsigned offset;
   unsigned size;
   std::string text;
   bool ok;
};

/* Split a code range into instructions through the AGX2 decoder, which is
 * the only authority on instruction length and form we trust. */
std::vector<Decoded>
disasm_range(const uint8_t *code, size_t size, unsigned base = 0)
{
   std::vector<Decoded> out;
   for (unsigned i = 0; i < size;) {
      char *text = nullptr;
      size_t text_len = 0;
      FILE *fp = open_memstream(&text, &text_len);
      signed ret = agx2_disassemble_instr((BITSET_WORD *)(code + i), fp,
                                          base + i, size - i, false);
      fclose(fp);
      std::string s = text ? text : "";
      free(text);
      while (!s.empty() && (s.back() == '\n' || s.back() == ' '))
         s.pop_back();
      Decoded d = {base + i, 0, s, ret > 0};
      if (ret <= 0) {
         d.size = 0;
         out.push_back(d);
         break;
      }
      d.size = ret;
      out.push_back(d);
      i += ret;
   }
   return out;
}

/* Mnemonic = the opcode word; operands = the rest of the decoded line. */
std::string
form_of(const Decoded &d)
{
   size_t sp = d.text.find_first_of(" \t");
   return sp == std::string::npos ? d.text : d.text.substr(0, sp);
}

} // namespace

TEST(Agx2WitnessCompare, MultiplyKernelFieldByField)
{
   /* The witness is Apple's scoreprobe multiply kernel on applegpu_g14g:
    * the checked-in fixture holds _agc.main alone (30 bytes); the manifest
    * documents the full 94-byte __TEXT,__text layout and the test accepts
    * either form, decoding from the entry-point offset. */
   const std::vector<uint8_t> witness =
      read_file(fixture_path("probe.scoreprobe-reference.scoreprobe.g14g.bin"));
   ASSERT_GE(witness.size(), 30u) << "fixture must hold at least _agc.main";
   const unsigned entry = witness.size() - 30;
   if (entry != 0) {
      EXPECT_EQ(entry, 0x40u) << "unexpected fixture layout";
   }
   const uint8_t *wmain = witness.data() + entry;
   std::vector<Decoded> winstr = disasm_range(wmain, 30, entry);
   ASSERT_EQ(winstr.size(), 6u) << "witness entry must decode as 6 instrs";
   ASSERT_TRUE(winstr.back().ok);
   EXPECT_EQ(form_of(winstr.back()), "stop");

   /* The equivalent NIR: i = thread_position_in_grid.x; two buffer base
    * addresses through the preamble uniform path; the same
    * y[i] = x[i] * 2.0 the MSL expresses. Preamble bases 0 and 4 encode as
    * du0 and du2 (established: the du index is half the preamble base). */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "witness multiply");
   nir_def *i = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *base_x = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *base_y = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *off = nir_u2u64(&b, nir_imul_imm(&b, i, 4));
   nir_def *addr_x = nir_iadd(&b, base_x, off);
   nir_def *addr_y = nir_iadd(&b, base_y, off);
   nir_def *v = nir_load_global(&b, 1, 32, addr_x, .align_mul = 4);
   nir_def *r = nir_fmul_imm(&b, v, 2.0);
   nir_store_global(&b, r, addr_y, .align_mul = 4);

   /* The witness constant program is a fixed header, not a real preamble;
    * compile monolithic so the instruction streams compare directly. */
   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G14G;
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   key.no_preamble = true;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));

   const uint8_t *ours =
      static_cast<const uint8_t *>(part.binary) + part.info.main_offset;
   std::vector<Decoded> oinstr = disasm_range(ours, part.info.main_size);
   for (const Decoded &d : oinstr)
      EXPECT_TRUE(d.ok) << "our code failed to decode at " << d.offset;

   /* Program ABI, established facts only. The witness section layout is
    * documented by the manifest; the checked-in fixture holds _agc.main. */
   printf("== program ABI ==\n");
   printf("witness: manifest documents 0x40 B constant program, entry 0x40, "
          "entry program 0x1e B, total 0x5e B\n");
   printf("ours   : binary 0x%x B, preamble %s (0x%x B at 0x%x), "
          "main 0x%x B at 0x%x\n",
          part.info.binary_size, part.info.has_preamble ? "yes" : "no",
          part.info.preamble_size, part.info.preamble_offset,
          part.info.main_size, part.info.main_offset);
   printf("established: layouts differ (witness section wraps a 0x40 B "
          "constant program; ours compiled monolithic with no_preamble)\n");

   /* Full stream of ours first: the witness is the reference, so seeing
    * every instruction we emit keeps the pair list honest. */
   printf("== our instruction stream ==\n");
   for (const Decoded &d : oinstr)
      printf(" @0x%02x %-6s %s\n", d.offset, form_of(d).c_str(),
             d.text.c_str());

   /* Align the streams by form in program order: each witness instruction
    * is matched to the next instruction of the same form in ours, so an
    * instruction we emit extra never shifts the pairs it follows. What the
    * matcher cannot pair is reported, not guessed. */
   printf("== instruction comparison (form-aligned) ==\n");
   unsigned n_identical = 0, n_paired = 0;
   std::vector<bool> ours_used(oinstr.size(), false);
   unsigned scan = 0;
   for (const Decoded &w : winstr) {
      signed match = -1;
      while (scan < oinstr.size()) {
         if (!ours_used[scan] && form_of(oinstr[scan]) == form_of(w)) {
            match = scan++;
            break;
         }
         ++scan;
      }
      if (match < 0) {
         printf(" witness @0x%02x %-6s %s | NO instruction of this form "
                "in ours\n",
                w.offset, form_of(w).c_str(), w.text.c_str());
         continue;
      }
      const Decoded &o = oinstr[match];
      ours_used[match] = true;
      ++n_paired;
      bool bytes_same = w.size == o.size &&
                        !memcmp(wmain + w.offset - entry,
                                ours + o.offset, w.size);
      bool text_same = w.text == o.text;
      const char *verdict =
         bytes_same ? "established: byte-identical"
         : text_same  ? "established: same decode, different encoding"
                      : "established: same form, operands differ";
      n_identical += bytes_same;
      printf(" witness @0x%02x %-6s %s | ours @0x%02x %-6s %s | %s\n",
             w.offset, form_of(w).c_str(), w.text.c_str(), o.offset,
             form_of(o).c_str(), o.text.c_str(), verdict);
   }
   printf(" ours-only instructions (inserted vs witness):\n");
   for (unsigned k = 0; k < oinstr.size(); ++k)
      if (!ours_used[k])
         printf("   @0x%02x %-6s %s\n", oinstr[k].offset,
                form_of(oinstr[k]).c_str(), oinstr[k].text.c_str());
   printf(" %u/%zu witness instructions paired, %u byte-identical; "
          "staging order and wait placement unestablished beyond the "
          "reported stream\n",
          n_paired, winstr.size(), n_identical);

   /* Established so far: both programs decode fully. Termination form is
    * reported, not assumed. */
   printf("termination: witness '%s', ours '%s'%s\n",
          form_of(winstr.back()).c_str(), form_of(oinstr.back()).c_str(),
          form_of(winstr.back()) == form_of(oinstr.back()) ? " (match)"
                                                          : " (established difference)");

   free(part.binary);
   ralloc_free(b.shader);
}
