/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ioto_core.h"

#include <string.h>

enum agx_ioto_encoding {
   AGX_IOTO_ENCODING_GPU0,
   AGX_IOTO_ENCODING_GPU1,
   AGX_IOTO_ENCODING_UNSUPPORTED,
};

struct agx_ioto_fixed_word {
   uint8_t word;
   uint16_t value;
   uint16_t mask;
};

struct agx_ioto_form {
   enum agx_ioto_kind kind;
   enum agx_ioto_encoding encoding;
   uint8_t length;
   uint8_t fixed_word_count;
   struct agx_ioto_fixed_word fixed[5];
};

#define FW(index, val, msk)                                                    \
   { .word = (index), .value = UINT16_C(val), .mask = UINT16_C(msk) }
#define FORM1(k, enc, len, a)                                                  \
   {                                                                           \
      .kind = AGX_IOTO_KIND_##k, .encoding = AGX_IOTO_ENCODING_##enc,          \
      .length = (len), .fixed_word_count = 1, .fixed = {a},                    \
   }
#define FORM2(k, enc, len, a, b)                                               \
   {                                                                           \
      .kind = AGX_IOTO_KIND_##k, .encoding = AGX_IOTO_ENCODING_##enc,          \
      .length = (len), .fixed_word_count = 2, .fixed = {a, b},                 \
   }
#define FORM3(k, enc, len, a, b, c)                                            \
   {                                                                           \
      .kind = AGX_IOTO_KIND_##k, .encoding = AGX_IOTO_ENCODING_##enc,          \
      .length = (len), .fixed_word_count = 3, .fixed = {a, b, c},              \
   }
#define FORM4(k, enc, len, a, b, c, d)                                         \
   {                                                                           \
      .kind = AGX_IOTO_KIND_##k, .encoding = AGX_IOTO_ENCODING_##enc,          \
      .length = (len), .fixed_word_count = 4, .fixed = {a, b, c, d},           \
   }
#define FORM5(k, enc, len, a, b, c, d, e)                                      \
   {                                                                           \
      .kind = AGX_IOTO_KIND_##k, .encoding = AGX_IOTO_ENCODING_##enc,          \
      .length = (len), .fixed_word_count = 5, .fixed = {a, b, c, d, e},        \
   }

/*
 * Fixed bits and lengths are kept as a literal transcription of the recovered
 * gpu0/gpu1 encoder graph.  Unlisted bits are opaque and are never rewritten.
 */
static const struct agx_ioto_form agx_ioto_gpu0_forms[] = {
   FORM1(NOP_EP0, GPU0, 2, FW(0, 0x0006, 0x001f)),
   FORM2(NOP_EP1, GPU0, 4, FW(0, 0x001e, 0x001f),
         FW(1, 0x0100, 0x0f00)),
   FORM2(NOP_EP2, GPU0, 6, FW(0, 0x001e, 0x001f),
         FW(1, 0x0200, 0x0f00)),
   FORM2(NOP_EP3, GPU0, 8, FW(0, 0x001e, 0x001f),
         FW(1, 0x0300, 0x0f00)),
   FORM2(NOP_EP4, GPU0, 10, FW(0, 0x001e, 0x001f),
         FW(1, 0x0400, 0x0f00)),
   FORM2(NOP_EP5, GPU0, 12, FW(0, 0x001e, 0x001f),
         FW(1, 0x0500, 0x0f00)),
   FORM2(NOP_EP6, GPU0, 14, FW(0, 0x001e, 0x001f),
         FW(1, 0x0600, 0x0f00)),
   FORM2(NOP_EP7, GPU0, 16, FW(0, 0x001e, 0x001f),
         FW(1, 0x0700, 0x0f00)),
   FORM2(NOP_EP8, GPU0, 18, FW(0, 0x001e, 0x001f),
         FW(1, 0x0800, 0x0f00)),
   FORM2(NOP_EP9, GPU0, 20, FW(0, 0x001e, 0x001f),
         FW(1, 0x0900, 0x0f00)),
   FORM1(LDIMM_EP0, GPU0, 2, FW(0, 0x0004, 0x8007)),
   FORM2(LDIMM_EP1, GPU0, 8, FW(0, 0x8004, 0x8007),
         FW(1, 0x0002, 0x0003)),
   FORM2(LDIMM_EP2, GPU0, 10, FW(0, 0x8004, 0x8007),
         FW(1, 0x0003, 0x0003)),
   FORM2(COPY_EP0, GPU0, 4, FW(0, 0x0003, 0x0007),
         FW(1, 0x0001, 0x0007)),
   FORM3(COPY_EP1, GPU0, 10, FW(0, 0x0003, 0x0007),
         FW(1, 0x0007, 0x0007), FW(2, 0x0002, 0x0803)),
   FORM2(RLD_EP1, GPU0, 8, FW(0, 0x8004, 0x8007),
         FW(1, 0x0001, 0x0003)),
   FORM1(SPECLM_EP0, GPU0, 8, FW(0, 0x0077, 0x07ff)),
   FORM1(LDSHDR_EP0, GPU0, 10, FW(0, 0x0177, 0x07ff)),
   FORM1(SPECTPR_EP0, GPU0, 10, FW(0, 0x00f7, 0x07ff)),
   FORM1(IADD_EP0, GPU0, 10, FW(0, 0x019f, 0x07ff)),
   FORM1(B_EP0, GPU0, 10, FW(0, 0x000f, 0x07ff)),
   FORM1(BL_EP0, GPU0, 10, FW(0, 0x008f, 0x07ff)),
   FORM1(CNDCALL_EP0, GPU0, 6, FW(0, 0x020f, 0x07ff)),
   FORM1(CNDRET_EP0, GPU0, 4, FW(0, 0x028f, 0x07ff)),
   FORM1(SETPROFILECTL_EP0, GPU0, 4, FW(0, 0x03f7, 0x07ff)),

};

static const struct agx_ioto_form agx_ioto_gpu1_forms[] = {
   FORM1(NOP_EP0, GPU1, 2, FW(0, 0x0006, 0x001f)),
   FORM2(NOP_EP1, GPU1, 4, FW(0, 0x001e, 0x001f),
         FW(1, 0x0000, 0x0f80)),
   FORM2(NOP_EP2, GPU1, 6, FW(0, 0x001e, 0x001f),
         FW(1, 0x0100, 0x0f80)),
   FORM2(NOP_EP3, GPU1, 8, FW(0, 0x001e, 0x001f),
         FW(1, 0x0200, 0x0f80)),
   FORM2(NOP_EP4, GPU1, 10, FW(0, 0x001e, 0x001f),
         FW(1, 0x0300, 0x0f80)),
   FORM2(NOP_EP5, GPU1, 12, FW(0, 0x001e, 0x001f),
         FW(1, 0x0400, 0x0f80)),
   FORM2(NOP_EP6, GPU1, 14, FW(0, 0x001e, 0x001f),
         FW(1, 0x0500, 0x0f80)),
   FORM2(NOP_EP7, GPU1, 16, FW(0, 0x001e, 0x001f),
         FW(1, 0x0600, 0x0f80)),
   FORM2(NOP_EP8, GPU1, 18, FW(0, 0x001e, 0x001f),
         FW(1, 0x0700, 0x0f80)),
   FORM2(NOP_EP9, GPU1, 20, FW(0, 0x001e, 0x001f),
         FW(1, 0x0800, 0x0f80)),
   FORM2(NOP_EP10, GPU1, 22, FW(0, 0x001e, 0x001f),
         FW(1, 0x0900, 0x0f80)),
   FORM1(LDIMM_EP0, GPU1, 2, FW(0, 0x0004, 0x800f)),
   FORM1(LDIMM_EP1, GPU1, 2, FW(0, 0x000c, 0x800f)),
   FORM2(LDIMM_EP2, GPU1, 8, FW(0, 0x8004, 0x8007),
         FW(1, 0x0002, 0x0003)),
   FORM2(LDIMM_EP3, GPU1, 10, FW(0, 0x8004, 0x8007),
         FW(1, 0x0003, 0x0003)),
   FORM2(COPY_EP0, GPU1, 4, FW(0, 0x0003, 0x0007),
         FW(1, 0x0001, 0x0007)),
   FORM3(COPY_EP1, GPU1, 8, FW(0, 0x0003, 0x0007),
         FW(1, 0x0007, 0x0007), FW(2, 0x0002, 0x0803)),
   FORM2(RLD_EP0, GPU1, 4, FW(0, 0x8004, 0x8007),
         FW(1, 0x0000, 0x0003)),
   FORM2(RLD_EP1, GPU1, 8, FW(0, 0x8004, 0x8007),
         FW(1, 0x0001, 0x0003)),
   FORM3(SPECLM_EP0, GPU1, 8, FW(0, 0x0007, 0x0007),
         FW(1, 0x0c00, 0x0f07), FW(2, 0x0000, 0x0003)),
   FORM3(LDSHDR_EP0, GPU1, 10, FW(0, 0x0007, 0x0007),
         FW(1, 0x0e00, 0x0f07), FW(2, 0x0000, 0x0003)),
   FORM3(SPECTPR_EP0, GPU1, 10, FW(0, 0x0007, 0x0007),
         FW(1, 0x0d00, 0x0f07), FW(2, 0x0000, 0x0003)),
   FORM4(IADD_EP0, GPU1, 12, FW(0, 0x0007, 0x0007),
         FW(1, 0x0004, 0x0007), FW(2, 0x0001, 0x0003),
         FW(3, 0x0003, 0x000f)),
   FORM2(B_EP1_COMPACT, GPU1, 4, FW(0, 0x005e, 0x007f),
         FW(1, 0x0e10, 0x0f11)),
   FORM2(B_EP1, GPU1, 10, FW(0, 0x005e, 0x007f),
         FW(1, 0x0e11, 0x0f11)),
   FORM2(BL_EP1_COMPACT, GPU1, 4, FW(0, 0x007e, 0x007f),
         FW(1, 0x0e10, 0x0f11)),
   FORM2(BL_EP1, GPU1, 10, FW(0, 0x007e, 0x007f),
         FW(1, 0x0e11, 0x0f11)),
   FORM4(CNDCALL_EP0, GPU1, 8, FW(0, 0x0007, 0x0007),
         FW(1, 0x0004, 0x0007), FW(2, 0x0000, 0x0003),
         FW(3, 0x0004, 0x000f)),
   FORM4(CNDRET_EP0, GPU1, 8, FW(0, 0x0007, 0x0007),
         FW(1, 0x0004, 0x0007), FW(2, 0x0000, 0x0003),
         FW(3, 0x0005, 0x000f)),
   FORM4(LDDEV_EP0, GPU1, 8, FW(0, 0x0007, 0x0007),
         FW(1, 0x0003, 0x0007), FW(2, 0x0000, 0x0003),
         FW(3, 0x0000, 0x8006)),
   FORM5(LDDEV_EP1, GPU1, 10, FW(0, 0x0007, 0x0007),
         FW(1, 0x0003, 0x0007), FW(2, 0x0000, 0x0003),
         FW(3, 0x8000, 0x8006), FW(4, 0x0000, 0x0001)),
   FORM5(LDDEV_EP2, GPU1, 14, FW(0, 0x0007, 0x0007),
         FW(1, 0x0003, 0x0007), FW(2, 0x0000, 0x0003),
         FW(3, 0x8000, 0x8006), FW(4, 0x0001, 0x0001)),
   FORM4(LDDEV_EP3, GPU1, 10, FW(0, 0x0007, 0x0007),
         FW(1, 0x0007, 0x0007), FW(3, 0x0000, 0x0003),
         FW(4, 0x0000, 0x8006)),
   FORM5(LDDEV_EP4, GPU1, 12, FW(0, 0x0007, 0x0007),
         FW(1, 0x0007, 0x0007), FW(3, 0x0000, 0x0003),
         FW(4, 0x8000, 0x8006), FW(5, 0x0000, 0x0001)),
   FORM5(LDDEV_EP5, GPU1, 16, FW(0, 0x0007, 0x0007),
         FW(1, 0x0007, 0x0007), FW(3, 0x0000, 0x0003),
         FW(4, 0x8000, 0x8006), FW(5, 0x0001, 0x0001)),
   FORM3(PIXREL_EP0, GPU1, 6, FW(0, 0x0007, 0x0007),
         FW(1, 0x0400, 0x0f07), FW(2, 0x0000, 0x0003)),
   FORM3(PIXWAIT_EP0, GPU1, 6, FW(0, 0x0007, 0x0007),
         FW(1, 0x0500, 0x0f07), FW(2, 0x0000, 0x0003)),
   FORM3(SETPROFILECTL_EP0, GPU1, 6, FW(0, 0x0007, 0x0007),
         FW(1, 0x0f00, 0x0f07), FW(2, 0x0000, 0x0003)),
};

#undef FORM5
#undef FORM4
#undef FORM3
#undef FORM2
#undef FORM1
#undef FW

static enum agx_ioto_encoding
agx_ioto_target_encoding(enum agx_chip chip)
{
   switch (agx_chip_ioto_generation(chip)) {
   case AGX_IOTO_GENERATION_GPU0:
      return AGX_IOTO_ENCODING_GPU0;
   case AGX_IOTO_GENERATION_GPU1:
      return AGX_IOTO_ENCODING_GPU1;
   default:
      return AGX_IOTO_ENCODING_UNSUPPORTED;
   }
}

bool
agx_ioto_target_supported(enum agx_chip chip)
{
   return agx_ioto_target_encoding(chip) != AGX_IOTO_ENCODING_UNSUPPORTED;
}

static const struct agx_ioto_form *
agx_ioto_encoding_forms(enum agx_ioto_encoding encoding, size_t *count)
{
   switch (encoding) {
   case AGX_IOTO_ENCODING_GPU0:
      *count = sizeof(agx_ioto_gpu0_forms) / sizeof(agx_ioto_gpu0_forms[0]);
      return agx_ioto_gpu0_forms;
   case AGX_IOTO_ENCODING_GPU1:
      *count = sizeof(agx_ioto_gpu1_forms) / sizeof(agx_ioto_gpu1_forms[0]);
      return agx_ioto_gpu1_forms;
   default:
      *count = 0;
      return NULL;
   }
}

static const struct agx_ioto_form *
agx_ioto_find_form(enum agx_ioto_encoding encoding, enum agx_ioto_kind kind)
{
   size_t count;
   const struct agx_ioto_form *forms = agx_ioto_encoding_forms(encoding, &count);
   for (size_t i = 0; i < count; ++i) {
      if (forms[i].kind == kind)
         return &forms[i];
   }

   return NULL;
}

static bool
agx_ioto_prefix_matches(const struct agx_ioto_form *form,
                        const uint8_t *bytes, size_t available)
{
   const size_t visible = available < form->length ? available : form->length;

   for (uint8_t i = 0; i < form->fixed_word_count; ++i) {
      const struct agx_ioto_fixed_word *fixed = &form->fixed[i];
      const size_t offset = (size_t)fixed->word * 2;
      uint16_t known_mask = 0;
      uint16_t value = 0;

      if (offset < visible) {
         known_mask |= fixed->mask & UINT16_C(0x00ff);
         value |= bytes[offset];
      }
      if (offset + 1 < visible) {
         known_mask |= fixed->mask & UINT16_C(0xff00);
         value |= (uint16_t)bytes[offset + 1] << 8;
      }

      if (((value ^ fixed->value) & known_mask) != 0)
         return false;
   }

   return true;
}

static void
agx_ioto_clear_outputs(struct agx_ioto_record *out, size_t *consumed)
{
   if (out)
      memset(out, 0, sizeof(*out));
   if (consumed)
      *consumed = 0;
}

static void
agx_ioto_set_output(const struct agx_ioto_form *form, const uint8_t *bytes,
                    struct agx_ioto_record *out, size_t *consumed)
{
   out->kind = form->kind;
   out->length = form->length;
   memcpy(out->bytes, bytes, form->length);
   *consumed = form->length;
}

bool
agx_ioto_validate_kind(enum agx_chip chip, enum agx_ioto_kind expected,
                       const void *stream, size_t available,
                       struct agx_ioto_record *out, size_t *consumed)
{
   agx_ioto_clear_outputs(out, consumed);

   const struct agx_ioto_form *form =
      agx_ioto_find_form(agx_ioto_target_encoding(chip), expected);
   if (!form || !stream || !out || !consumed || available < form->length)
      return false;

   const uint8_t *bytes = stream;
   if (!agx_ioto_prefix_matches(form, bytes, form->length))
      return false;

   agx_ioto_set_output(form, bytes, out, consumed);
   return true;
}

bool
agx_ioto_decode_one(enum agx_chip chip, const void *stream, size_t available,
                    struct agx_ioto_record *out, size_t *consumed)
{
   agx_ioto_clear_outputs(out, consumed);

   const enum agx_ioto_encoding encoding = agx_ioto_target_encoding(chip);
   if (encoding == AGX_IOTO_ENCODING_UNSUPPORTED || !stream || !out ||
       !consumed || available == 0)
      return false;

   const uint8_t *bytes = stream;
   const struct agx_ioto_form *match = NULL;
   bool incomplete_match = false;

   size_t count;
   const struct agx_ioto_form *forms = agx_ioto_encoding_forms(encoding, &count);
   for (size_t i = 0; i < count; ++i) {
      const struct agx_ioto_form *form = &forms[i];
      /* Every form begins with a word-zero opcode predicate. Reject using
       * the first available byte before checking longer prefixes. Keep all
       * candidates: ambiguous and truncated streams must still fail. */
      if (((bytes[0] ^ form->fixed[0].value) &
           form->fixed[0].mask & 0xff) != 0 ||
          !agx_ioto_prefix_matches(form, bytes, available))
         continue;

      if (available < form->length) {
         incomplete_match = true;
         continue;
      }

      if (match)
         return false;
      match = form;
   }

   if (!match || incomplete_match)
      return false;

   agx_ioto_set_output(match, bytes, out, consumed);
   return true;
}

static bool
agx_ioto_program_gpu0(enum agx_chip chip)
{
   return agx_ioto_target_encoding(chip) == AGX_IOTO_ENCODING_GPU0;
}

/* Low byte of a G17P fragment PIXREL descriptor: bit 1 set, each colour class
 * field either wholly released or wholly consumed, and at least one consumed.
 */
static bool
agx_ioto_g17p_fragment_release(uint8_t low)
{
   if (!(low & 0x02) || (low & 0xfc) == 0xfc)
      return false;

   for (unsigned c = 0; c < 3; ++c) {
      const unsigned field = (low >> (2 + 2 * c)) & 0x3;
      if (field != 0 && field != 0x3)
         return false;
   }
   return true;
}

static void
agx_ioto_clear_program_outputs(struct agx_ioto_program_record *out,
                               size_t *consumed)
{
   if (out)
      memset(out, 0, sizeof(*out));
   if (consumed)
      *consumed = 0;
}

bool
agx_ioto_decode_program_record(enum agx_chip chip, const void *stream,
                               size_t available,
                               struct agx_ioto_program_record *out,
                               size_t *consumed)
{
   agx_ioto_clear_program_outputs(out, consumed);

   if (!agx_ioto_target_supported(chip) || !stream || !out || !consumed ||
       available == 0)
      return false;

   const uint8_t *bytes = stream;
   static const uint8_t terminate[AGX_IOTO_PROGRAM_TERMINATE_SIZE] = {
      (uint8_t)AGX_IOTO_PROGRAM_TERMINATE_WORD,
      (uint8_t)(AGX_IOTO_PROGRAM_TERMINATE_WORD >> 8),
      (uint8_t)(AGX_IOTO_PROGRAM_TERMINATE_WORD >> 16),
      (uint8_t)(AGX_IOTO_PROGRAM_TERMINATE_WORD >> 24),
   };

   const size_t terminal_visible =
      available < sizeof(terminate) ? available : sizeof(terminate);
   if (memcmp(bytes, terminate, terminal_visible) == 0) {
      if (available < sizeof(terminate))
         return false;

      out->kind = AGX_IOTO_PROGRAM_RECORD_TERMINATE;
      out->length = sizeof(terminate);
      memcpy(out->bytes, bytes, sizeof(terminate));
      *consumed = sizeof(terminate);
      return true;
   }

   /* G17P's ordinary, coverage and background fragment releases are GPU0-only
    * six-byte encodings emitted by the bounded program composers. They are
    * not the GPU1 PIXREL form in the public record table, so the boundary is
    * carried by the exact descriptor family rather than a mask.
    *
    * Bytes three and four hold the release descriptor. 0x200 is the bounded
    * clear/reload background release. A fragment release starts from 0x2ff
    * and clears the 0x0c, 0x30 or 0xc0 field of each colour class it consumes,
    * and coverage additionally clears bit 0. applegpu_g17p fragments writing
    * each subset of two to four colour targets, with and without a sample
    * mask, carry exactly these descriptors, 0x2f3 through 0x202. Accepting
    * only the all-partition forms left subset fragments undecodable, so a
    * program walk lost its boundary there. */
   if (agx_chip_native_agx3(chip) && available >= 6 && bytes[0] == 0x07 &&
       bytes[1] == 0x02 && bytes[2] == 0x54 &&
       (bytes[3] == 0x00 || agx_ioto_g17p_fragment_release(bytes[3])) &&
       bytes[4] == 0x02 &&
       bytes[5] == 0x00) {
      out->kind = AGX_IOTO_PROGRAM_RECORD_IOTO;
      out->length = 6;
      out->ioto.kind = AGX_IOTO_KIND_PIXREL_EP0;
      out->ioto.length = 6;
      memcpy(out->ioto.bytes, bytes, 6);
      memcpy(out->bytes, bytes, 6);
      *consumed = 6;
      return true;
   }

   /* The GPU0 compiler writes this AGX3b device load inline instead of
    * constructing an AGXIotoInstruction object.  Its low eleven opcode bits
    * and fourteen-byte boundary are exact; every other bit remains opaque. */
   if (agx_ioto_program_gpu0(chip) && bytes[0] == UINT8_C(0x67)) {
      if (available < sizeof(uint16_t))
         return false;

      const uint16_t word =
         (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
      if ((word & UINT16_C(0x07ff)) == UINT16_C(0x0067)) {
         if (available < AGX_IOTO_GPU0_INLINE_LDDEV_SIZE)
            return false;

         out->kind = AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV;
         out->length = AGX_IOTO_GPU0_INLINE_LDDEV_SIZE;
         memcpy(out->bytes, bytes, out->length);
         *consumed = out->length;
         return true;
      }
   }

   struct agx_ioto_record record;
   size_t record_size = 0;
   if (!agx_ioto_decode_one(chip, stream, available, &record, &record_size))
      return false;

   out->kind = AGX_IOTO_PROGRAM_RECORD_IOTO;
   out->length = record.length;
   out->ioto = record;
   memcpy(out->bytes, record.bytes, record.length);
   *consumed = record_size;
   return true;
}
