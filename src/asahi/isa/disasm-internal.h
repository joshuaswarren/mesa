/*
 * Copyright 2025 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "asahi/isa/agx_minifloat.h"
#include "util/bitset.h"
#include "util/u_math.h"
#include "disasm.h"

struct disasm_ctx {
   FILE *fp;
   bool any_operands;
   bool error;
};

static inline uint64_t
bits(BITSET_WORD *word, unsigned start, unsigned size, unsigned shift)
{
   /* A single field range can be wider than a BITSET_WORD -- AGX3 carries
    * 48-bit addresses and immediates in one contiguous range -- so extract in
    * <=32-bit chunks. __bitset_extract asserts on count > BITSET_WORDBITS, so
    * a 42- or 47-bit AGX3 field would otherwise abort the disassembler on real
    * shader code. */
   uint64_t v = 0;
   for (unsigned got = 0; got < size;) {
      unsigned chunk = MIN2(size - got, 32u);
      v |= ((uint64_t)__bitset_extract(word, start + got, chunk)) << got;
      got += chunk;
   }
   return v << shift;
}

typedef void (*disasm_instr)(struct disasm_ctx *ctx, BITSET_WORD *code);

/* Words of instruction a decoder sees. The longest encodings are 24 bytes --
 * AGX3d's smp, msg, nop and brk forms, which common G18G code uses -- i.e.
 * 192 bits = 6 words (AGX3c tops out at 22 bytes, AGX3b at 20). Every array
 * holding an instruction or a per-form mask must be this wide: at 5 words
 * BITSET_FOREACH_SET over n * 8 bits and the print callbacks for fields above
 * bit 159 read one word past the end, and bytes 20-23 were never matched.
 * gen-disasm.py emits mask/exact/known rows with this many words; brace-
 * initialised rows zero-fill any word it trims. */
#define AGX_DISASM_WORDS 6

struct disasm_spec {
   const char *display;
   disasm_instr disassemble;
   unsigned length_bit;
   unsigned length_short;
   unsigned length_long;
   BITSET_WORD mask[AGX_DISASM_WORDS];
   BITSET_WORD exact[AGX_DISASM_WORDS];
   BITSET_WORD known[AGX_DISASM_WORDS];
};

static void
introduce_operand(struct disasm_ctx *ctx)
{
   if (ctx->any_operands) {
      fprintf(ctx->fp, ", ");
   } else {
      fprintf(ctx->fp, " ");
   }

   ctx->any_operands = true;
}

enum operand_kind {
   KIND_NONE = 0,
   KIND_REG,
   KIND_UNIFORM,
   KIND_CF,
   KIND_TS,
   KIND_SS,
   KIND_IMM,
   KIND_FIMM,
};

struct operand_desc {
   enum operand_kind kind;
   int32_t value;
   unsigned hint, count;
   bool optional, size16, size32, size64, abs, neg, sx, cache, lu;
};

static void
_print_operand(struct disasm_ctx *ctx, struct operand_desc d)
{
   if (d.kind == KIND_NONE) {
      if (!d.optional) {
         introduce_operand(ctx);
         fprintf(ctx->fp, "_");
      }

      return;
   }

   unsigned size = d.size64 ? 64 : d.size32 ? 32 : 16;
   bool cache = d.cache || d.hint == 2;
   bool lu = d.lu || d.hint == 3;
   unsigned count = MAX2(d.count, 1);

   introduce_operand(ctx);

   if (lu && cache) {
      fprintf(ctx->fp, "XXX invalid cache+lu set\n");
      ctx->error = true;
   }

   if (lu)
      fprintf(ctx->fp, "^");
   if (cache)
      fprintf(ctx->fp, "$");

   const char *prefixes[][2] = {
      [KIND_REG] = {"r", "dr"}, [KIND_UNIFORM] = {"u", "du"},
      [KIND_CF] = {"cf"},       [KIND_TS] = {"ts"},
      [KIND_SS] = {"ss"},
   };

   if (d.kind == KIND_IMM) {
      fprintf(ctx->fp, "%d", d.value);
   } else if (d.kind == KIND_FIMM) {
      float f = agx_minifloat_decode(d.value);

      /* Match python's float printing */
      if (f == (int)f)
         fprintf(ctx->fp, "%g.0", agx_minifloat_decode(d.value));
      else
         fprintf(ctx->fp, "%g", agx_minifloat_decode(d.value));
   } else if (d.kind == KIND_CF || d.kind == KIND_TS || d.kind == KIND_SS) {
      fprintf(ctx->fp, "%s%u", prefixes[d.kind][0], d.value);
   } else {
      for (unsigned i = 0; i < count; ++i) {
         if (i)
            fprintf(ctx->fp, "_");

         unsigned reg = d.value + (i * (size / 16));
         unsigned whole = reg >> 1;
         unsigned part = reg & 1;
         const char *prefix = prefixes[d.kind][size == 64];

         if (size == 16) {
            fprintf(ctx->fp, "%s%u%c", prefix, whole, "lh"[part]);
         } else {
            if (part != 0) {
               fprintf(ctx->fp, "# 32-bit must be expected, but got raw %u\n",
                       reg);
               ctx->error = true;
            }

            fprintf(ctx->fp, "%s%u", prefix, whole);
         }
      }
   }

   if (d.abs)
      fprintf(ctx->fp, ".abs");
   if (d.neg)
      fprintf(ctx->fp, ".neg");
   if (d.sx)
      fprintf(ctx->fp, ".sx");
}

static void
print_immediate(struct disasm_ctx *ctx, bool is_signed, uint32_t value)
{
   introduce_operand(ctx);
   fprintf(ctx->fp, is_signed ? "%d" : "%u", value);
}

static void
_print_enum(struct disasm_ctx *ctx, const char **arr, unsigned n,
            unsigned value)
{
   if (value >= n || arr[value] == NULL) {
      introduce_operand(ctx);
      fprintf(ctx->fp, "XXX: Unknown enum value %u", value);
      ctx->error = true;
   } else if (arr[value][0] != 0) {
      introduce_operand(ctx);
      fprintf(ctx->fp, "%s", arr[value]);
   }
}

#define print_enum(ctx, arr, value)                                            \
   _print_enum(ctx, arr, ARRAY_SIZE(arr), value)

static void
print_modifier(struct disasm_ctx *ctx, const char *display, unsigned value)
{
   if (value) {
      introduce_operand(ctx);
      fprintf(ctx->fp, "%s", display);
   }
}

/* The forms of a table indexed by some of an instruction's first 64 bits
 * (gen-disasm.py, length_index()): the forms of bucket `key` are
 * forms[start[key]] up to forms[start[key + 1]], in table order. */
struct disasm_length_index {
   unsigned nr_bits;
   const uint8_t *bits;
   const uint16_t *start;
   const uint16_t *forms;
};

/* The length of spec's instruction if the code (at most `available` bytes,
 * zero-padded in tmp) is one, else 0: the matching rule _disassemble_instr
 * uses. */
static inline unsigned
_spec_length(const BITSET_WORD *tmp, const struct disasm_spec *spec,
             size_t available)
{
   const unsigned n = BITSET_TEST(tmp, spec->length_bit) ? spec->length_long
                                                         : spec->length_short;
   if (n > available)
      return 0;

   for (unsigned w = 0; w < AGX_DISASM_WORDS; ++w) {
      BITSET_WORD masked = tmp[w];
      const unsigned base = w * 4;
      if (base >= n)
         masked = 0;
      else if (n - base < 4)
         masked &= (UINT32_C(1) << ((n - base) * 8)) - 1;
      if ((masked & spec->mask[w]) != spec->exact[w])
         return 0;
   }
   return n;
}

/* The length and display name of the instruction at code, by exactly the
 * matching rule _disassemble_instr uses, without printing: the first form in
 * table order that matches. -1 if no form matches within `available` bytes.
 * The display name may carry trailing padding spaces. Only the forms of the
 * code's index bucket can match, and they keep their table order. */
static signed
_instr_length(const uint8_t *code, const struct disasm_spec *specs,
              const struct disasm_length_index *index, size_t available,
              const char **display)
{
   BITSET_WORD tmp[AGX_DISASM_WORDS] = {0};
   memcpy(tmp, code, MIN2(available, sizeof(tmp)));

   unsigned key = 0;
   for (unsigned k = 0; k < index->nr_bits; ++k)
      key |= (unsigned)BITSET_TEST(tmp, index->bits[k]) << k;

   for (unsigned i = index->start[key]; i < index->start[key + 1]; ++i) {
      const struct disasm_spec *spec = &specs[index->forms[i]];
      const unsigned n = _spec_length(tmp, spec, available);
      if (n) {
         if (display)
            *display = spec->display;
         return n;
      }
   }
   return -1;
}

/* _instr_length over the whole table, the reference for its index. */
static signed
_instr_length_linear(const uint8_t *code, const struct disasm_spec *specs,
                     unsigned nr_specs, size_t available, const char **display)
{
   BITSET_WORD tmp[AGX_DISASM_WORDS] = {0};
   memcpy(tmp, code, MIN2(available, sizeof(tmp)));

   for (unsigned i = 0; i < nr_specs; ++i) {
      const unsigned n = _spec_length(tmp, &specs[i], available);
      if (n) {
         if (display)
            *display = specs[i].display;
         return n;
      }
   }
   return -1;
}

static signed
_disassemble_instr(BITSET_WORD *code, FILE *fp, struct disasm_spec *specs,
                   unsigned nr_specs, size_t available, unsigned offset,
                   bool verbose)
{
   /* AGX_DISASM_WORDS covers the longest (24-byte) instruction. */
   BITSET_WORD tmp[AGX_DISASM_WORDS] = {0};
   size_t copy_size = MIN2(available, sizeof(tmp));

   memcpy(tmp, code, copy_size);
   struct disasm_spec *spec = NULL;
   unsigned n = 0;
   bool match = false;
   BITSET_WORD masked[AGX_DISASM_WORDS] = {0};

   for (unsigned i = 0; i < nr_specs; ++i) {
      spec = &specs[i];

      n = spec->length_short;
      if (BITSET_TEST(tmp, spec->length_bit)) {
         n = spec->length_long;
      }

      /* Zero-padded lookahead is decoder-private. Never accept an encoding
       * whose declared length extends beyond the caller's real byte range. */
      if (n > available)
         continue;

      match = true;
      for (unsigned i = 0; i < ARRAY_SIZE(tmp); ++i) {
         masked[i] = tmp[i];

         /* `n - i*4` is unsigned: once i*4 passes n it wrapped to a huge
          * value, the mask was skipped, and bytes belonging to the NEXT
          * instruction stayed in masked[] to be reported as unknown bits. */
         const unsigned base = i * 4;
         if (base >= n) {
            masked[i] = 0;
         } else {
            const unsigned bytes_left = n - base;
            if (bytes_left < 4)
               masked[i] &= ((UINT32_C(1) << (bytes_left * 8)) - 1);
         }

         if ((masked[i] & spec->mask[i]) != spec->exact[i]) {
            match = false;
            break;
         }
      }

      if (match)
         break;
   }

   struct disasm_ctx ctx = {.fp = fp};

   if (match) {
      BITSET_WORD unknown[AGX_DISASM_WORDS];
      __bitset_andnot(unknown, masked, spec->known, ARRAY_SIZE(masked));
      int i;
      BITSET_FOREACH_SET(i, unknown, n * 8) {
         fprintf(fp, "# XXX: Unknown bit set %u\n", i);
         ctx.error = true;
      }
   } else {
      n = 2;
   }

   if (verbose) {
      fprintf(fp, "%4x: ", offset);
      for (unsigned i = 0; i < n; ++i) {
         fprintf(fp, "%02x", __bitset_extract(masked, i * 8, 8));
      }
      for (unsigned i = n; i < 11; ++i) {
         fprintf(fp, "  ");
      }
      fprintf(fp, " ");
   }

   if (!match) {
      fprintf(fp, "<unknown instr>\n");
      return -n;
   }

   fprintf(fp, "%s", spec->display);

   if (spec->disassemble)
      spec->disassemble(&ctx, masked);

   fprintf(fp, "\n");

   return ctx.error ? -n : n;
}
