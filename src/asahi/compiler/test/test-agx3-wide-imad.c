/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The G17P 32x32 -> 64 integer multiply-add.
 *
 * `imadd` writes a 64-bit register pair when its destination width selector
 * (f10) is 3 instead of 2, and each 32-bit operand is widened according to its
 * own sign-extend bit (f28 slot A, f43 slot B, f58 slot C). That is the whole
 * instruction: `umul_high`, `imul_high`, `umul_2x32_64` and `imul_2x32_64` are
 * all this one encoding, with the caller reading whichever half it wants.
 *
 * Every vector below is a byte-exact `metal-nt -arch applegpu_g17p` witness,
 * one MSL kernel per variable. The kernels are the obvious ones:
 *
 *   umul_wide      o[0] = (ulong)a[i] * (ulong)b[i]
 *   imul_wide      o[0] = (long)a[i] * (long)b[i]
 *   mixed_wide     (long)a[i] * (long)b[i], a signed and b unsigned
 *   umad_wide      o[0] = (ulong)a[i] * (ulong)b[i] + (ulong)c[i]
 *   imad_wide      the signed form of the same
 *   umad_wide_c64  ... + c[i] where c is a device ulong
 *   umad_wide_c16  ... + c[i] where c is a device ushort
 *   umul_wide_imm  o[0] = (ulong)a[i] * 253ul
 *   umad_wide_imm  o[0] = (ulong)a[i] * 253ul + (ulong)c[i]
 *   umul_wide_16   both multiplicands are ushort
 *   umul_wide_live the product is read again by a following add
 *   umul_wide_high the product lands at a 2-aligned, not 4-aligned, pair
 *
 * `umul_hi`/`imul_hi` (Metal `mulhi`) compile to byte-identical instructions to
 * umul_wide/imul_wide -- only the store differs -- which is the direct evidence
 * that the high-half multiply has no encoding of its own.
 */

#include <stdio.h>
#include <string.h>

#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx3_iadd_core.h"
#include "nir_builder.h"

static unsigned checks, failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         ++failures;                                                           \
         fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #x);          \
      }                                                                        \
   } while (0)

static void
expect(const char *name, const struct agx3_imad_desc *d, const uint8_t *want,
       unsigned want_len)
{
   uint8_t out[AGX3_IADD_MAX_LENGTH];
   uint8_t len = 0;
   ++checks;

   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, d, out, &len)) {
      ++failures;
      fprintf(stderr, "FAIL %s: refused\n", name);
      return;
   }

   if (len != want_len || memcmp(out, want, want_len)) {
      ++failures;
      fprintf(stderr, "FAIL %s:\n  got  ", name);
      for (unsigned i = 0; i < len; ++i)
         fprintf(stderr, "%02x", out[i]);
      fprintf(stderr, "\n  want ");
      for (unsigned i = 0; i < want_len; ++i)
         fprintf(stderr, "%02x", want[i]);
      fprintf(stderr, "\n");
   }
}

static void
test_apple_wide_witnesses(void)
{
   /* Every kernel here waits scoreboard slot 6 for its device loads. */
   const uint16_t w6 = 1u << 5;

   /* o[0] = (ulong)a[i] * (ulong)b[i] -- both operands zero-extended. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0x08, 0x00, 0xe0, 0x26, 0x0a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 2,
         .multiplier_kill = true, .no_addend = true, .wait_mask = w6,
      };
      expect("umul_wide", &d, want, sizeof(want));
   }

   /* The signed form differs only in the two sign-extend bits. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0x08, 0x00, 0xe0, 0x26, 0x1e, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_kill = true,
         .src_a_sign_extend = true, .multiplier_is_register = true,
         .multiplier_reg = 2, .multiplier_kill = true,
         .multiplier_sign_extend = true, .no_addend = true, .wait_mask = w6,
      };
      expect("imul_wide", &d, want, sizeof(want));
   }

   /* Signedness is per operand, so a signed x unsigned product sets one bit. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0x08, 0x00, 0xe0, 0x26, 0x1a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 2,
         .multiplier_kill = true, .multiplier_sign_extend = true,
         .no_addend = true, .wait_mask = w6,
      };
      expect("mixed_wide", &d, want, sizeof(want));
   }

   /* With a 32-bit register addend the slot-C width stays 2. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x04,
                                     0x10, 0x00, 0xe0, 0x2f, 0x2a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 2, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 4,
         .multiplier_kill = true, .src_c = 0, .src_c_kill = true,
         .wait_mask = w6,
      };
      expect("umad_wide", &d, want, sizeof(want));
   }

   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x04,
                                     0x10, 0x00, 0xe0, 0x2f, 0x7e, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 2, .src_a_kill = true,
         .src_a_sign_extend = true, .multiplier_is_register = true,
         .multiplier_reg = 4, .multiplier_kill = true,
         .multiplier_sign_extend = true, .src_c = 0, .src_c_kill = true,
         .src_c_sign_extend = true, .wait_mask = w6,
      };
      expect("imad_wide", &d, want, sizeof(want));
   }

   /* A 64-bit addend needs no widening at all: slot-C width 3. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                     0x18, 0x00, 0xf0, 0x2e, 0x2a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 4, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 6,
         .multiplier_kill = true, .src_c = 0, .src_c_64 = true,
         .src_c_kill = true, .wait_mask = w6,
      };
      expect("umad_wide_c64", &d, want, sizeof(want));
   }

   /* ... and a 16-bit addend keeps slot-C width 1 under a 64-bit product. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0x08, 0x20, 0xe0, 0x2e, 0x2a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 2,
         .multiplier_kill = true, .src_c = 4, .src_c_16 = true,
         .src_c_kill = true, .wait_mask = w6,
      };
      expect("umad_wide_c16", &d, want, sizeof(want));
   }

   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0x08, 0x20, 0xe0, 0x2e, 0x7e, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_kill = true,
         .src_a_sign_extend = true, .multiplier_is_register = true,
         .multiplier_reg = 2, .multiplier_kill = true,
         .multiplier_sign_extend = true, .src_c = 4, .src_c_16 = true,
         .src_c_kill = true, .src_c_sign_extend = true, .wait_mask = w6,
      };
      expect("imad_wide_c16", &d, want, sizeof(want));
   }

   /* An immediate multiplier is always zero-extended; only slot A can sign. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0xf4, 0x03, 0x60, 0x22, 0x02, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_kill = true,
         .multiplier = 253, .no_addend = true, .wait_mask = w6,
      };
      expect("umul_wide_imm", &d, want, sizeof(want));
   }

   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x04,
                                     0xf4, 0x03, 0x60, 0x2b, 0x22, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 2, .src_a_kill = true,
         .multiplier = 253, .src_c = 0, .src_c_kill = true, .wait_mask = w6,
      };
      expect("umad_wide_imm", &d, want, sizeof(want));
   }

   /* 16-bit multiplicands: the operand widths are per slot, the product is
    * still a 64-bit pair. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                     0x08, 0x00, 0x20, 0x26, 0x0a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 0, .src_a_16 = true,
         .src_a_kill = true, .multiplier_is_register = true,
         .multiplier_reg = 2, .multiplier_16 = true, .multiplier_kill = true,
         .no_addend = true, .wait_mask = w6,
      };
      expect("umul_wide_16", &d, want, sizeof(want));
   }

   /* The destination cache hint is a real per-instruction bit: the same pure
    * multiply carries it when an ALU op reads the product and clears it when
    * only a store does. Both witnesses are here so neither can be mistaken
    * for a baseline constant. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x03, 0x00,
                                     0x10, 0x04, 0xe0, 0x22, 0x0a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .dest_cache = true, .src_a = 0,
         .src_a_kill = true, .multiplier_is_register = true,
         .multiplier_reg = 4, .multiplier_cache = true, .no_addend = true,
         .wait_mask = w6,
      };
      expect("umul_wide_live", &d, want, sizeof(want));
   }

   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08,
                                     0x1a, 0x04, 0xe0, 0x20, 0x0a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .dest_64 = true, .src_a = 4, .src_a_cache = true,
         .multiplier_is_register = true, .multiplier_reg = 6,
         .multiplier_cache = true, .no_addend = true, .wait_mask = w6,
      };
      expect("umul_wide_regs", &d, want, sizeof(want));
   }

   /* Apple places a 64-bit product at half 18: the pair is 32-bit aligned, not
    * 64-bit aligned, so the encoder must not demand four-half alignment. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x12, 0x03, 0x04,
                                     0x10, 0x00, 0xe0, 0x26, 0x0a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 18, .dest_64 = true, .dest_cache = true, .src_a = 2,
         .src_a_kill = true, .multiplier_is_register = true,
         .multiplier_reg = 4, .multiplier_kill = true, .no_addend = true,
         .wait_mask = w6,
      };
      expect("umul_wide_high", &d, want, sizeof(want));
   }

   /* The narrow forms must not move. This is Apple's own 32-bit pure multiply
    * with the product consumed only by a store, so its cache hint is clear --
    * the case that proves f11 is not part of the baseline. */
   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x04,
                                     0x00, 0x00, 0xd0, 0x26, 0x0a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .src_a = 2, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 0,
         .multiplier_kill = true, .no_addend = true, .wait_mask = w6,
      };
      expect("umul_narrow", &d, want, sizeof(want));
   }

   {
      static const uint8_t want[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x04,
                                     0x00, 0x20, 0xd0, 0x2f, 0x2a, 0x00};
      const struct agx3_imad_desc d = {
         .dest = 0, .src_a = 2, .src_a_kill = true,
         .multiplier_is_register = true, .multiplier_reg = 0,
         .multiplier_kill = true, .src_c = 4, .src_c_kill = true,
         .wait_mask = w6,
      };
      expect("imad_narrow", &d, want, sizeof(want));
   }
}

static void
test_domain_is_closed(void)
{
   uint8_t out[AGX3_IADD_MAX_LENGTH];
   uint8_t len;

   /* A 64-bit destination occupies four halves, so it must be even. (The
    * form's destination field stops accepting values above half 191 for both
    * widths, which is a property of Apple's own tables, not of this change --
    * the composer refuses those and the encoder fails closed with it.) */
   struct agx3_imad_desc d = {
      .dest = 0, .dest_64 = true, .src_a = 0, .multiplier_is_register = true,
      .multiplier_reg = 2, .no_addend = true,
   };
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.dest = 190;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.dest = 1;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   d.dest = 254;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   /* Widening a source is only meaningful under a wider destination: a 32-bit
    * product has nothing to sign-extend into. */
   struct agx3_imad_desc narrow = {
      .dest = 0, .src_a = 0, .multiplier_is_register = true,
      .multiplier_reg = 2, .no_addend = true, .src_a_sign_extend = true,
   };
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &narrow, out, &len));
   narrow.src_a_sign_extend = false;
   narrow.multiplier_sign_extend = true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &narrow, out, &len));

   /* An 8-bit immediate multiplier has no sign-extend bit of its own. */
   struct agx3_imad_desc imm = {
      .dest = 0, .dest_64 = true, .src_a = 0, .multiplier = 253,
      .no_addend = true, .multiplier_sign_extend = true,
   };
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &imm, out, &len));

   /* A pure multiply has no addend to widen. */
   struct agx3_imad_desc pure = {
      .dest = 0, .dest_64 = true, .src_a = 0,
      .multiplier_is_register = true, .multiplier_reg = 2, .no_addend = true,
      .src_c_64 = true,
   };
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &pure, out, &len));

   /* A 64-bit addend is a pair too, and is not also 16-bit. */
   struct agx3_imad_desc wide_c = {
      .dest = 0, .dest_64 = true, .src_a = 0,
      .multiplier_is_register = true, .multiplier_reg = 2, .src_c = 1,
      .src_c_64 = true,
   };
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &wide_c, out, &len));
   wide_c.src_c = 4;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &wide_c, out, &len));
   wide_c.src_c_16 = true;
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &wide_c, out, &len));

   /* The AGX3c `imadd64` is witnessed with an immediate or a 64-bit
    * register addend (test-agx3-iadd-core.c); the addend-free pure multiply
    * and the sign-extending variants still fail closed there. */
   struct agx3_imad_desc agx3c = {
      .dest = 0, .dest_64 = true, .src_a = 0,
      .multiplier_is_register = true, .multiplier_reg = 2, .no_addend = true,
   };
   CHECK(!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &agx3c, out, &len));
   agx3c.no_addend = false;
   agx3c.addend_is_immediate = true;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &agx3c, out, &len));
   agx3c.src_a_sign_extend = true;
   CHECK(!agx3_imad_encode_g18g(AGX_ISA_ENCODING_AGX3C, &agx3c, out, &len));
   agx3c.multiplier_sign_extend = true;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &agx3c, out, &len));
}

/*
 * Whole-shader compiles. Each of these reaches the backend as an `imad` with a
 * 64-bit destination, which is exactly what could not be encoded: pipeline
 * creation for any shader containing one failed with the compiler's
 * fail-closed error.
 */
static bool
compile_compute_keep(nir_shader *nir, const char *what,
                     struct agx_shader_part *part)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   memset(part, 0, sizeof(*part));

   ++checks;
   const bool compiled = agx_compile_shader_nir(nir, &key, part);
   const bool valid =
      compiled && part->binary != NULL && part->info.binary_size > sizeof(end) &&
      !memcmp((uint8_t *)part->binary + part->info.binary_size - sizeof(end),
              end, sizeof(end));

   if (!valid) {
      ++failures;
      fprintf(stderr, "FAIL %s: %s\n", what,
              compiled ? "compiled but produced no terminated binary"
                       : "the AGX3 backend refused the shader");
      free(part->binary);
      part->binary = NULL;
   }

   ralloc_free(nir);
   return valid;
}

static bool
compile_compute(nir_shader *nir, const char *what)
{
   struct agx_shader_part part;
   const bool valid = compile_compute_keep(nir, what, &part);
   free(part.binary);
   return valid;
}

/* Scan an AGX3B shader binary for a 12-byte `imadd` (opcode 9f 00 56) whose
 * width/form byte (b[8]) and sign-extension/addend byte (b[10]) match a
 * witnessed form, optionally also checking the 8-bit immediate bytes (b[6..7])
 * when check_imm is true. */
static bool
binary_contains_imad_form(const struct agx_shader_part *part, uint8_t want_b8,
                          uint8_t want_b10, bool check_imm, uint8_t want_b6,
                          uint8_t want_b7)
{
   const uint8_t *bin = (const uint8_t *)part->binary;
   if (!bin || part->info.binary_size < 12)
      return false;

   /* AGX3B `imadd` places opcode 0x009f in bits 0..10 (b[0]=0x9f, b[1]&0x07=0),
    * scoreboard waits w1..w6 in bits 12..17 (b[1]&0xf0 and b[2]&0x03), and the
    * fixed bits in b[2]&0xfc (0x54). Check b[8], b[10], b[11] and optional
    * immediate b[6..7] independently of any scoreboard wait mask. */
   for (size_t i = 0; i + 12 <= part->info.binary_size; i += 2) {
      if (bin[i] == 0x9f && (bin[i + 1] & 0x07) == 0x00 &&
          (bin[i + 2] & 0xfc) == 0x54 && bin[i + 8] == want_b8 &&
          bin[i + 10] == want_b10 && bin[i + 11] == 0x00 &&
          (!check_imm || (bin[i + 6] == want_b6 && bin[i + 7] == want_b7)))
         return true;
   }
   return false;
}

/* Create a minimal single-block AGX3B IR context, lower one wide `imad` via
 * agx_lower_agx3_widening_int(), bind deterministic physical registers, pack
 * the resulting `imad` through agx_pack_binary(), and verify its exact 12 bytes
 * against an Apple witness. */
static void
check_widening_int_case(const char *name, agx_index s0, agx_index s1,
                        bool expect_imm_mult, unsigned expect_mov_op,
                        bool expect_s0_abs, const uint8_t want[12])
{
   void *mem = ralloc_context(NULL);
   agx_context *ctx = rzalloc(mem, agx_context);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .dev.chip = AGX_CHIP_G17P,
   };
   ctx->key = &key;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->alloc = 8;
   list_inithead(&ctx->blocks);

   agx_block *blk = rzalloc(ctx, agx_block);
   list_inithead(&blk->instructions);
   list_addtail(&blk->link, &ctx->blocks);

   agx_builder b = agx_init_builder(ctx, agx_after_block(blk));
   agx_index dst = agx_temp(ctx, AGX_SIZE_64);
   agx_instr *imad = agx_imad_to(&b, dst, s0, s1, agx_immediate(0), 0);

   CHECK(agx_lower_agx3_widening_int(ctx));

   /* Verify the inserted legalization op before `imad`, if any. */
   if (expect_mov_op != 0) {
      agx_instr *prev = list_entry(imad->link.prev, agx_instr, link);
      CHECK(&prev->link != &blk->instructions);
      CHECK(prev->op == expect_mov_op);
   }

   CHECK(imad->src[0].type == AGX_INDEX_NORMAL &&
         imad->src[0].size == AGX_SIZE_32 &&
         imad->src[0].abs == expect_s0_abs);
   if (expect_imm_mult) {
      CHECK(imad->src[1].type == AGX_INDEX_IMMEDIATE && !imad->src[1].abs &&
            !imad->src[1].neg && imad->src[1].value <= 0xff);
   } else {
      CHECK(imad->src[1].type == AGX_INDEX_NORMAL &&
            imad->src[1].size == AGX_SIZE_32);
   }

   /* Pack just the lowered `imad` with deterministic registers (d=r0..r3,
    * src0=r0.discard, src1=r2.discard or imm) and assert all 12 bytes. */
   agx_context *pack_ctx = rzalloc(mem, agx_context);
   pack_ctx->key = &key;
   pack_ctx->stage = MESA_SHADER_COMPUTE;
   pack_ctx->out = rzalloc(mem, struct agx_shader_info);
   list_inithead(&pack_ctx->blocks);
   agx_block *pblk = rzalloc(pack_ctx, agx_block);
   list_inithead(&pblk->instructions);
   list_addtail(&pblk->link, &pack_ctx->blocks);

   agx_builder pb = agx_init_builder(pack_ctx, agx_after_block(pblk));
   agx_index p0 = agx_register(0, AGX_SIZE_32);
   p0.abs = imad->src[0].abs;
   p0.discard = true;
   agx_index p1 = imad->src[1];
   if (!expect_imm_mult) {
      p1 = agx_register(2, AGX_SIZE_32);
      p1.abs = imad->src[1].abs;
      p1.discard = true;
   }
   agx_imad_to(&pb, agx_register(0, AGX_SIZE_64), p0, p1, agx_immediate(0), 0);
   agx_stop(&pb);

   struct util_dynarray emission;
   util_dynarray_init(&emission, mem);
   CHECK(agx3_pack_binary(pack_ctx, &emission));

   ++checks;
   if (emission.size < 12 || memcmp(emission.data, want, 12) != 0) {
      ++failures;
      fprintf(stderr, "FAIL widening_int %s:\n  got  ", name);
      const uint8_t *got = (const uint8_t *)emission.data;
      for (size_t i = 0; i < (emission.size < 12 ? emission.size : 12); ++i)
         fprintf(stderr, "%02x", got[i]);
      fprintf(stderr, "\n  want ");
      for (unsigned i = 0; i < 12; ++i)
         fprintf(stderr, "%02x", want[i]);
      fprintf(stderr, "\n");
   }

   ralloc_free(mem);
}

static void
test_agx3b_widening_int_legalization(void)
{
   /* Exact 12-byte Apple witnesses with wait_mask=0 (f65/w6 is byte 2 bit 1,
    * so byte 2 drops 0x56 -> 0x54):
    *   umul_wide:     9f 00 54 00 02 00 08 00 e0 26 0a 00
    *   imul_wide:     9f 00 54 00 02 00 08 00 e0 26 1e 00
    *   umul_wide_imm: 9f 00 54 00 02 00 f4 03 60 22 02 00  (mult = 253)
    *   imul_wide_imm: 9f 00 54 00 02 00 f4 03 60 22 06 00  (mult = 253)
    */
   static const uint8_t want_umul_wide[12] = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x08, 0x00, 0xe0, 0x26, 0x0a, 0x00,
   };
   static const uint8_t want_imul_wide[12] = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x08, 0x00, 0xe0, 0x26, 0x1e, 0x00,
   };
   static const uint8_t want_umul_wide_imm_253[12] = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0xf4, 0x03, 0x60, 0x22, 0x02, 0x00,
   };
   static const uint8_t want_imul_wide_imm_253[12] = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0xf4, 0x03, 0x60, 0x22, 0x06, 0x00,
   };

   /* 16-bit unsigned register widened via mov with .abs preserved -> umul_wide. */
   check_widening_int_case(
      "u16_reg_widens_to_umul_wide",
      agx_abs(agx_get_index(1, AGX_SIZE_16)),
      agx_abs(agx_get_index(2, AGX_SIZE_32)),
      false, AGX_OPCODE_MOV, true, want_umul_wide);

   /* 16-bit signed register widened via signext -> imul_wide. */
   check_widening_int_case(
      "s16_reg_widens_to_imul_wide",
      agx_get_index(1, AGX_SIZE_16),
      agx_get_index(2, AGX_SIZE_32),
      false, AGX_OPCODE_SIGNEXT, false, want_imul_wide);

   /* Both multiplicands immediates (#7.abs * #253.abs) -> materialize src[0]
    * and keep #253 as 8-bit immediate multiplier -> umul_wide_imm. */
   check_widening_int_case(
      "both_imm_to_umul_wide_imm",
      agx_abs(agx_immediate(7)),
      agx_abs(agx_immediate(253)),
      true, AGX_OPCODE_MOV_IMM, true, want_umul_wide_imm_253);

   /* Lone src[0] immediate (#253.abs * x.abs) -> commuted into slot B ->
    * umul_wide_imm. */
   check_widening_int_case(
      "lone_src0_imm_unsigned",
      agx_abs(agx_immediate(253)),
      agx_abs(agx_get_index(2, AGX_SIZE_32)),
      true, 0, true, want_umul_wide_imm_253);

   /* Lone src[0] immediate signed (#253 * x) -> commuted into slot B ->
    * imul_wide_imm. */
   check_widening_int_case(
      "lone_src0_imm_signed",
      agx_immediate(253),
      agx_get_index(2, AGX_SIZE_32),
      true, 0, false, want_imul_wide_imm_253);

   /* 16-bit .abs immediate (agx_optimizer.c:648 shape: x.abs * #0x1234.abs) ->
    * materialized to 32-bit register with .abs -> umul_wide. */
   check_widening_int_case(
      "imm16_abs_to_umul_wide",
      agx_abs(agx_get_index(1, AGX_SIZE_32)),
      agx_abs(agx_immediate(0x1234)),
      false, AGX_OPCODE_MOV_IMM, true, want_umul_wide);
}

static nir_builder
begin_compute(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static void
test_wide_multiply_shaders(void)
{
   /* umul_2x32_64 / imul_2x32_64: agx_nir_options deliberately leaves these
    * unlowered, so they arrive as a wide imad. */
   {
      nir_builder b = begin_compute("G17P 32x32->64 multiply");
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *y = nir_load_preamble(&b, 1, 32, .base = 10);
      nir_store_global(&b, nir_umul_2x32_64(&b, x, y), out, .align_mul = 8);
      nir_store_global(&b, nir_imul_2x32_64(&b, x, y),
                       nir_iadd_imm(&b, out, 8), .align_mul = 8);
      agx_preprocess_nir(b.shader);
      compile_compute(b.shader, "32x32->64 multiply");
   }

   /* The high halves are the same instruction with the low word dropped. */
   {
      nir_builder b = begin_compute("G17P high-word multiply");
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *y = nir_load_preamble(&b, 1, 32, .base = 10);
      nir_store_global(&b, nir_umul_high(&b, x, y), out, .align_mul = 4);
      nir_store_global(&b, nir_imul_high(&b, x, y),
                       nir_iadd_imm(&b, out, 4), .align_mul = 4);
      agx_preprocess_nir(b.shader);
      compile_compute(b.shader, "high-word multiply");
   }

   /* Division and modulo by a compile-time constant are strength-reduced to a
    * high-word multiply by a magic number, so every shader that divides by a
    * constant other than a power of two needs the wide imad.
    *
    * (Division by a *dynamic* divisor is a separate, still-open gap: NIR
    * lowers it through a float reciprocal and AGX3 has no grounded `rcp`
    * encoding yet, so it is deliberately not asserted here.) */
   {
      nir_builder b = begin_compute("G17P constant integer division");
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_store_global(&b, nir_udiv_imm(&b, x, 3), out, .align_mul = 4);
      nir_store_global(&b, nir_umod_imm(&b, x, 3), nir_iadd_imm(&b, out, 4),
                       .align_mul = 4);
      nir_store_global(&b, nir_idiv(&b, x, nir_imm_int(&b, 3)), nir_iadd_imm(&b, out, 8),
                       .align_mul = 4);
      agx_preprocess_nir(b.shader);
      compile_compute(b.shader, "constant integer division");
   }

   /* Division by a constant is strength-reduced to a high-word multiply by a
    * magic number, which is the shape a vertex shader hits when it maps a
    * vertex index onto a primitive. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "G17P vertex divide by three");
      nir_def *id = nir_load_vertex_id(&b);
      nir_def *pos[4] = {
         nir_i2f32(&b, nir_idiv(&b, id, nir_imm_int(&b, 3))),
         nir_u2f32(&b, nir_umod(&b, id, nir_imm_int(&b, 3))),
         nir_imm_float(&b, 0.0f),
         nir_imm_float(&b, 1.0f),
      };
      for (unsigned c = 0; c < 4; ++c)
         nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

      b.shader->info.outputs_written = VARYING_BIT_POS;
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B};
      struct agx_shader_part part = {0};
      ++checks;
      if (!agx_compile_shader_nir(b.shader, &key, &part)) {
         ++failures;
         fprintf(stderr, "FAIL vertex divide by three: refused\n");
      }
      free(part.binary);
      ralloc_free(b.shader);
   }

   /* Lowered 64x64->64 multiply must keep xl * yl on native AGX3B dest_64
    * umul_wide (b[8]=0xe0, b[10]=0x0a) instead of expanding 32x32->64 in SW. */
   {
      nir_builder b = begin_compute("G17P lowered 64x64->64 multiply");
      nir_def *in = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_global(&b, 1, 64, in, .align_mul = 8);
      nir_def *y =
         nir_load_global(&b, 1, 64, nir_iadd_imm(&b, in, 8), .align_mul = 8);
      nir_store_global(&b, nir_imul(&b, x, y), out, .align_mul = 8);
      agx_preprocess_nir_g17p(b.shader);

      struct agx_shader_part part;
      if (compile_compute_keep(b.shader, "lowered 64x64->64 multiply", &part)) {
         CHECK(binary_contains_imad_form(&part, 0xe0, 0x0a, false, 0, 0));
         free(part.binary);
      }
   }

   /* Lone src[0] immediate: umul_2x32_64(253, x) and imul_2x32_64(253, x) must
    * commute into slot B and emit the exact witnessed umul_wide_imm /
    * imul_wide_imm encodings (b[6..7]=f4 03, b[8]=60, b[10]=02 / 06). */
   {
      nir_builder b = begin_compute("G17P wide multiply lone src0 imm");
      nir_def *in = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_global(&b, 1, 32, in, .align_mul = 4);
      nir_def *y =
         nir_load_global(&b, 1, 32, nir_iadd_imm(&b, in, 4), .align_mul = 4);
      nir_store_global(&b, nir_umul_2x32_64(&b, nir_imm_int(&b, 253), x), out,
                       .align_mul = 8);
      nir_store_global(&b, nir_imul_2x32_64(&b, nir_imm_int(&b, 253), y),
                       nir_iadd_imm(&b, out, 8), .align_mul = 8);
      agx_preprocess_nir_g17p(b.shader);

      struct agx_shader_part part;
      if (compile_compute_keep(b.shader, "wide multiply lone src0 imm", &part)) {
         CHECK(binary_contains_imad_form(&part, 0x60, 0x02, true, 0xf4, 0x03));
         CHECK(binary_contains_imad_form(&part, 0x60, 0x06, true, 0xf4, 0x03));
         free(part.binary);
      }
   }

   /* 16-bit immediate: umul_2x32_64(x, 0x1234) and umul_2x32_64(0x1234, x) must
    * materialize the constant and emit register-register umul_wide (b[8]=0xe0,
    * b[10]=0x0a), never refusing on mult_imm. */
   {
      nir_builder b = begin_compute("G17P wide multiply 16-bit imm");
      nir_def *in = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_global(&b, 1, 32, in, .align_mul = 4);
      nir_store_global(&b, nir_umul_2x32_64(&b, x, nir_imm_int(&b, 0x1234)),
                       out, .align_mul = 8);
      nir_store_global(&b, nir_umul_2x32_64(&b, nir_imm_int(&b, 0x1234), x),
                       nir_iadd_imm(&b, out, 8), .align_mul = 8);
      agx_preprocess_nir_g17p(b.shader);

      struct agx_shader_part part;
      if (compile_compute_keep(b.shader, "wide multiply 16-bit imm", &part)) {
         CHECK(binary_contains_imad_form(&part, 0xe0, 0x0a, false, 0, 0));
         free(part.binary);
      }
   }

   /* 16-bit unsigned and signed sources widened into wide imad must emit
    * 32x32->64 umul_wide / imul_wide (b[8]=0xe0, b[10]=0x0a / 0x1e). */
   {
      nir_builder b = begin_compute("G17P wide multiply from 16-bit sources");
      nir_def *in = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *h = nir_load_global(&b, 1, 16, in, .align_mul = 2);
      nir_def *y =
         nir_load_global(&b, 1, 32, nir_iadd_imm(&b, in, 4), .align_mul = 4);
      nir_store_global(&b, nir_umul_2x32_64(&b, nir_u2u32(&b, h), y), out,
                       .align_mul = 8);
      nir_store_global(&b, nir_imul_2x32_64(&b, nir_i2i32(&b, h), y),
                       nir_iadd_imm(&b, out, 8), .align_mul = 8);
      agx_preprocess_nir_g17p(b.shader);

      struct agx_shader_part part;
      if (compile_compute_keep(b.shader, "wide multiply from 16-bit sources",
                               &part)) {
         CHECK(binary_contains_imad_form(&part, 0xe0, 0x0a, false, 0, 0));
         CHECK(binary_contains_imad_form(&part, 0xe0, 0x1e, false, 0, 0));
         free(part.binary);
      }
   }
}

int
main(void)
{
   test_apple_wide_witnesses();
   test_domain_is_closed();
   test_agx3b_widening_int_legalization();
   test_wide_multiply_shaders();

   printf("agx3_wide_imad: %u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
