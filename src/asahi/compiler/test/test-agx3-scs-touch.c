/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The AGX3b destination-less SCS load against Mesa's own Gen1 leaf programs.
 *
 * Every ldscs row of the four native op tables is composed from its field
 * list. A row without a destination (no f9, no f12) must come back from
 * agx3_scs_encode_touch byte for byte and must not decode as an ordinary
 * load; every other row must not decode as a touch.
 */

#include "agx3_scs_core.h"

#include "agx3_binder.h"
#include "agx3_rt_gen1_aabb_ops.h"
#include "agx3_rt_gen1_clique_ops.h"
#include "agx3_rt_gen1_instance_push_ops.h"
#include "agx3_rt_gen1_triangle_ops.h"

#include <stdio.h>
#include <string.h>

#define ENC   AGX_ISA_ENCODING_AGX3B
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static unsigned failures;
static unsigned checks;

static void
expect(bool ok, const char *what, const char *where, unsigned row)
{
   checks++;
   if (!ok) {
      fprintf(stderr, "%s row %u: %s\n", where, row, what);
      failures++;
   }
}

static uint64_t
field(const struct agx3_rt_gen1_native_op *op, unsigned id)
{
   for (unsigned i = 0; i < op->nfields; ++i) {
      if (op->fields[i].field == id)
         return op->fields[i].value;
   }
   return 0;
}

/* Returns the destination-less loads seen; counts the ordinary ones. */
static unsigned
check_table(const char *name, const struct agx3_rt_gen1_native_op *ops,
            unsigned count, unsigned *ordinary)
{
   unsigned touches = 0;
   for (unsigned i = 0; i < count; ++i) {
      const struct agx3_rt_gen1_native_op *op = &ops[i];
      if (op->name != AGX3_MN_ldscs && op->name != AGX3_MN_stscs)
         continue;

      uint8_t bytes[AGX3_SCS_MAX_LENGTH] = {0}, length = 0;
      const bool composed =
         agx3_bind_compose(ENC, op->name, op->ep, op->fields, op->nfields,
                           NULL, 0, bytes, &length) && length == op->size;
      expect(composed, "table row does not compose", name, i);
      if (!composed)
         continue;

      struct agx3_scs_touch_desc touch;
      struct agx3_scs_desc load;
      const bool is_touch = op->name == AGX3_MN_ldscs &&
                            field(op, 9) == 0 && field(op, 12) == 0;
      const bool decoded = agx3_scs_decode_touch(ENC, bytes, length, &touch);
      if (!is_touch) {
         expect(!decoded, "ordinary access decodes as a touch", name, i);
         *ordinary += op->name == AGX3_MN_ldscs;
         continue;
      }

      touches++;
      expect(decoded, "destination-less load does not decode", name, i);
      expect(field(op, 28) == 1 && field(op, 80) == 1,
             "destination-less load without its fixed controls", name, i);
      expect(!agx3_scs_decode_load(ENC, bytes, length, &load),
             "destination-less load decodes as an ordinary load", name, i);
      if (!decoded)
         continue;

      uint8_t again[AGX3_SCS_MAX_LENGTH] = {0}, again_length = 0;
      expect(agx3_scs_encode_touch(ENC, &touch, again, &again_length) &&
                again_length == length && !memcmp(again, bytes, length),
             "re-encoding differs from the table's bytes", name, i);
      expect(touch.base_register == field(op, 31) &&
                touch.has_index == (field(op, 23) == 2) &&
                touch.index == field(op, 16) &&
                touch.component_mask == field(op, 68) &&
                touch.address_selector == field(op, 70) &&
                (uint16_t)touch.byte_offset == field(op, 71) &&
                touch.base_kill == !!field(op, 42) &&
                touch.index_kill == !!field(op, 27),
             "decoded operands differ from the table's fields", name, i);
   }
   return touches;
}

static void
check_tables(void)
{
   unsigned ordinary = 0;
#define TABLE(t) check_table(#t, t, COUNT(t), &ordinary)
   expect(TABLE(agx3_rt_gen1_triangle_ops) == 1, "expected one", "triangle", 0);
   expect(TABLE(agx3_rt_gen1_aabb_ops) == 1, "expected one", "aabb", 0);
   expect(TABLE(agx3_rt_gen1_instance_push_ops) == 0, "expected none",
          "instance push", 0);
   const unsigned clique =
      TABLE(agx3_rt_gen1_clique_190_ops) + TABLE(agx3_rt_gen1_clique_102_ops) +
      TABLE(agx3_rt_gen1_clique_209_ops) + TABLE(agx3_rt_gen1_clique_196_ops) +
      TABLE(agx3_rt_gen1_clique_197_ops) + TABLE(agx3_rt_gen1_clique_195_ops) +
      TABLE(agx3_rt_gen1_clique_106_ops);
#undef TABLE
   expect(clique == 3, "expected three", "clique", clique);
   expect(ordinary >= 40, "too few ordinary loads were checked", "tables",
          ordinary);
}

/* The leaf programs' instance: base r0 at its final use, two words. */
static const struct agx3_scs_touch_desc leaf = {
   .base_register = 0, .component_mask = 3, .base_kill = true,
};

static void
check_indexed(void)
{
   /* Four words per lane at one row, and the one-word row form. */
   const struct agx3_scs_touch_desc rows[] = {
      {.base_register = 4, .has_index = true, .index = 9, .byte_offset = 512,
       .component_mask = 15, .address_selector = 4},
      {.base_register = 190, .has_index = true, .index = 191,
       .byte_offset = 9216, .component_mask = 1, .address_selector = 2},
      {.base_register = 6, .has_index = true, .index = 2, .byte_offset = -16,
       .component_mask = 3, .address_selector = 4, .index_kill = true,
       .base_kill = true},
      {.base_register = 8, .byte_offset = 288, .component_mask = 15},
   };
   for (unsigned i = 0; i < COUNT(rows); ++i) {
      const struct agx3_scs_touch_desc *d = &rows[i];
      uint8_t bytes[AGX3_SCS_MAX_LENGTH] = {0}, length = 0;
      const bool ok = agx3_scs_encode_touch(ENC, d, bytes, &length);
      expect(ok && length == 14, "refused", "indexed", i);
      if (!ok)
         continue;

      /* The descriptor is exactly this field list and nothing else. */
      const struct agx3_field_kv fields[] = {
         {10, 2}, {16, d->index}, {23, d->has_index ? 2 : 1}, {24, 1},
         {27, d->index_kill}, {28, 1}, {31, d->base_register}, {38, 2},
         {39, 2}, {42, d->base_kill}, {54, 1}, {68, d->component_mask},
         {70, d->address_selector}, {71, (uint16_t)d->byte_offset}, {80, 1},
      };
      uint8_t direct[AGX3_SCS_MAX_LENGTH] = {0}, direct_length = 0;
      expect(agx3_bind_compose(ENC, AGX3_MN_ldscs, 0, fields, COUNT(fields), NULL,
                               0, direct, &direct_length) &&
                direct_length == length && !memcmp(direct, bytes, length),
             "bytes differ from the field list", "indexed", i);

      struct agx3_scs_touch_desc back;
      struct agx3_scs_desc load;
      expect(agx3_scs_decode_touch(ENC, bytes, length, &back) &&
                back.base_register == d->base_register &&
                back.has_index == d->has_index && back.index == d->index &&
                back.byte_offset == d->byte_offset &&
                back.component_mask == d->component_mask &&
                back.address_selector == d->address_selector &&
                back.index_kill == d->index_kill &&
                back.base_kill == d->base_kill,
             "does not decode to itself", "indexed", i);
      expect(!agx3_scs_decode_load(ENC, bytes, length, &load) &&
                !agx3_scs_decode_store(ENC, bytes, length, &load),
             "decodes as an ordinary access", "indexed", i);
   }

   /* An ordinary load of the same address is a different instruction and is
    * not a touch. */
   const struct agx3_scs_desc ordinary = {
      .data = 2, .width = AGX3_MEM_W32, .byte_offset = 512,
      .address_kind = AGX3_SCS_ADDRESS_REGISTER_INDEXED, .index = 8,
      .base_register = 4, .component_mask = 15, .has_address_selector = true,
      .address_selector = 4, .slot = 1,
   };
   uint8_t bytes[AGX3_SCS_MAX_LENGTH] = {0}, touch_bytes[AGX3_SCS_MAX_LENGTH];
   uint8_t length = 0, touch_length = 0;
   struct agx3_scs_touch_desc back;
   expect(agx3_scs_encode_load(ENC, &ordinary, bytes, &length) &&
             !agx3_scs_decode_touch(ENC, bytes, length, &back),
          "ordinary indexed load decodes as a touch", "indexed", 0);
   expect(agx3_scs_encode_touch(ENC, &rows[0], touch_bytes, &touch_length) &&
             (touch_length != length || memcmp(touch_bytes, bytes, length)),
          "touch equals an ordinary load", "indexed", 0);
}

static void
check_refusals(void)
{
   const struct agx3_scs_touch_desc row = {
      .base_register = 4, .has_index = true, .index = 8, .byte_offset = 512,
      .component_mask = 15, .address_selector = 4,
   };
   uint8_t bytes[AGX3_SCS_MAX_LENGTH], length;
   expect(agx3_scs_encode_touch(ENC, &row, bytes, &length), "control",
          "refusal", 0);

   for (unsigned invalid = 0; invalid < 15; ++invalid) {
      struct agx3_scs_touch_desc bad = row;
      enum agx_isa_encoding enc = ENC;
      switch (invalid) {
      case 0: bad.byte_offset = 2; break;
      case 1: bad.component_mask = 0; break;
      case 2: bad.component_mask = 7; break;
      case 3: bad.component_mask = 16; break;
      case 4: bad.address_selector = 3; break;
      case 5: bad.address_selector = 0; break;
      case 6: bad.index = 192; break;
      case 7: bad.base_register = 5; break;
      case 8: bad.base_register = 192; break;
      case 9: bad.has_index = false; break; /* index operand left behind */
      case 10: bad.has_index = false; bad.index = 0; break; /* selector */
      case 11:
         bad = leaf;
         bad.index_kill = true;
         break;
      case 12: enc = AGX_ISA_ENCODING_AGX3C; break;
      case 13: enc = AGX_ISA_ENCODING_AGX2; break;
      case 14: bad.base_register = 191; break;
      }
      memset(bytes, 0xff, sizeof(bytes));
      length = 0xff;
      bool clean = !agx3_scs_encode_touch(enc, &bad, bytes, &length) &&
                   length == 0;
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         clean &= bytes[j] == 0;
      expect(clean, "accepted, or refusal left output behind", "refusal",
             invalid);
   }

   struct agx3_scs_touch_desc back;
   expect(agx3_scs_encode_touch(ENC, &row, bytes, &length) &&
             !agx3_scs_decode_touch(ENC, bytes, length - 2, &back) &&
             !agx3_scs_decode_touch(AGX_ISA_ENCODING_AGX3C, bytes, length,
                                    &back) &&
             !agx3_scs_decode_touch(ENC, NULL, length, &back) &&
             !agx3_scs_decode_touch(ENC, bytes, length, NULL) &&
             !agx3_scs_encode_touch(ENC, NULL, bytes, &length) &&
             !agx3_scs_encode_touch(ENC, &row, NULL, &length) &&
             !agx3_scs_encode_touch(ENC, &row, bytes, NULL),
          "bad arguments accepted", "refusal", 0);

   /* A touch with any other control bit is not a touch. */
   expect(agx3_scs_encode_touch(ENC, &row, bytes, &length), "control",
          "refusal", 1);
   unsigned accepted = 0;
   for (unsigned bit = 0; bit < 14 * 8; ++bit) {
      uint8_t flipped[AGX3_SCS_MAX_LENGTH];
      memcpy(flipped, bytes, sizeof(flipped));
      flipped[bit / 8] ^= 1u << (bit % 8);
      if (!agx3_scs_decode_touch(ENC, flipped, length, &back))
         continue;
      /* Whatever still decodes must be a touch that encodes to those bytes. */
      uint8_t again[AGX3_SCS_MAX_LENGTH], again_length;
      expect(agx3_scs_encode_touch(ENC, &back, again, &again_length) &&
                again_length == length && !memcmp(again, flipped, length),
             "decode accepted bytes it cannot reproduce", "bit", bit);
      accepted++;
   }
   expect(accepted > 0 && accepted < 14 * 8, "bit sweep", "refusal", accepted);
}

int
main(void)
{
   check_tables();
   check_indexed();
   check_refusals();

   if (failures) {
      fprintf(stderr, "agx3_scs_touch: %u of %u checks failed\n", failures,
              checks);
      return 1;
   }
   printf("agx3_scs_touch: %u checks passed\n", checks);
   return 0;
}
