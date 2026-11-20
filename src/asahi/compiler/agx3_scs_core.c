/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_scs_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static unsigned
slot_count(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3C   ? 8
          : enc == AGX_ISA_ENCODING_AGX3B ? 6
                                          : 0;
}

static bool
width_ok(enum agx3_mem_width width)
{
   return width == AGX3_MEM_W8 || width == AGX3_MEM_W16 ||
          width == AGX3_MEM_W32;
}

static bool
desc_ok(enum agx_isa_encoding enc, const struct agx3_scs_desc *d,
        bool is_load)
{
   const unsigned slots = slot_count(enc);
   if (d == NULL || slots == 0)
      return false;

   const bool uniform_indexed =
      d->address_kind == AGX3_SCS_ADDRESS_UNIFORM_INDEXED;
   const bool reg_indexed =
      d->address_kind == AGX3_SCS_ADDRESS_REGISTER_INDEXED;
   const bool reg_base = d->address_kind == AGX3_SCS_ADDRESS_REGISTER_BASE;
   const bool register_based = reg_indexed || reg_base;
   const bool has_index = uniform_indexed || reg_indexed;
   const bool indexed = uniform_indexed;

   /* Apple's M5 instanced intersector also uses two GPR address operands. */
   if (enc == AGX_ISA_ENCODING_AGX3C && d->component_mask > 1 &&
       d->width != AGX3_MEM_W32)
      return false;

   if (!width_ok(d->width) || d->data >= 256 ||
       d->wait_mask >= (1u << slots) ||
       (is_load && (d->slot < 1 || d->slot > slots)) ||
       (!is_load && d->slot != 0) || (is_load && d->value_kill) ||
       (!is_load && d->zero_extend) ||
       (!is_load && d->c_load_marker) ||
       (d->base_kill && (!register_based || enc != AGX_ISA_ENCODING_AGX3C)) ||
       (d->b_status_load &&
        (enc != AGX_ISA_ENCODING_AGX3B || !is_load ||
         d->width != AGX3_MEM_W8 || !reg_base ||
         d->component_mask != 1 || !d->has_address_selector ||
         d->address_selector != 4)) ||
       (d->c_status_load &&
        (enc != AGX_ISA_ENCODING_AGX3C || !is_load ||
         d->width != AGX3_MEM_W8 || !reg_base ||
         d->component_mask != 1 || !d->has_address_selector ||
         d->address_selector != 4 || !d->c_load_marker)) ||
       (d->index_kill && (!has_index || enc != AGX_ISA_ENCODING_AGX3C)) ||
       (d->c_store_marker && (is_load || enc != AGX_ISA_ENCODING_AGX3C)) ||
       (enc != AGX_ISA_ENCODING_AGX3C &&
        (d->zero_extend || d->c_load_marker)) ||
       (d->address_kind != AGX3_SCS_ADDRESS_DIRECT && !indexed &&
        !register_based) ||
       (!has_index && d->index != 0) ||
       (!uniform_indexed && d->uniform_base != 0) ||
       (!register_based && d->base_register != 0) ||
       (register_based && d->base_register >= 256) ||
       (has_index &&
        d->index >= (enc == AGX_ISA_ENCODING_AGX3B ? 192u : 512u)) ||
       (uniform_indexed &&
        (d->uniform_base >= 512 ||
         (d->value_kill && enc != AGX_ISA_ENCODING_AGX3C))) ||
       d->component_mask > 15 ||
       d->address_selector > 4 ||
       (!d->has_address_selector && register_based) ||
       (d->has_address_selector && d->address_kind == AGX3_SCS_ADDRESS_DIRECT) ||
       (d->zero_extend &&
        (!indexed || !is_load || d->width == AGX3_MEM_W32)))
      return false;

   const unsigned bytes = 1u << d->width;
   unsigned components = 1;
   for (unsigned mask = d->component_mask; mask > 1; mask >>= 1)
      components++;
   const unsigned halves = (d->width == AGX3_MEM_W32 ? 2 : 1) * components;
   return d->data + halves <= 256 &&
          (d->width != AGX3_MEM_W32 || (d->data & 1u) == 0) &&
          ((uint16_t)d->byte_offset & (bytes - 1u)) == 0;
}

static bool
encode(enum agx_isa_encoding enc, const struct agx3_scs_desc *d,
       bool is_load, uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_SCS_MAX_LENGTH);
   *length = 0;
   if (!desc_ok(enc, d, is_load))
      return false;

   struct agx3_field_kv base[9];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   const bool indexed =
      d->address_kind == AGX3_SCS_ADDRESS_UNIFORM_INDEXED;
   const bool reg_indexed =
      d->address_kind == AGX3_SCS_ADDRESS_REGISTER_INDEXED;
   const bool reg_base = d->address_kind == AGX3_SCS_ADDRESS_REGISTER_BASE;
   const bool register_based = reg_indexed || reg_base;
   const bool slot_a_register = indexed || reg_indexed;
   /* Slot A: 1 = immediate, 2 = register.  Slot B: 1 = immediate,
    * 2 = register, 3 = uniform.  Matches all four witnessed combinations. */
   base[nb++] = (struct agx3_field_kv)KV(23, slot_a_register ? 2 : 1);
   base[nb++] = (struct agx3_field_kv)KV(24, slot_a_register ? 2 : 1);
   base[nb++] =
      (struct agx3_field_kv)KV(38, indexed ? 3 : (register_based ? 2 : 1));
   base[nb++] =
      (struct agx3_field_kv)KV(39, (indexed || register_based) ? 2 : 1);
   base[nb++] = (struct agx3_field_kv)KV(54, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(45, 15);
      base[nb++] = (struct agx3_field_kv)KV(53, 1);
      base[nb++] = (struct agx3_field_kv)KV(59, is_load ? 106 : 188);
   }

   struct agx3_field_kv ov[32];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->data);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(
      68, d->component_mask ? d->component_mask : 1u);
   ov[n++] = (struct agx3_field_kv)KV(71, (uint16_t)d->byte_offset);
   if (slot_a_register)
      ov[n++] = (struct agx3_field_kv)KV(16, d->index);
   if (d->index_kill)
      ov[n++] = (struct agx3_field_kv)KV(27, 1);
   if (indexed)
      ov[n++] = (struct agx3_field_kv)KV(32, d->uniform_base);
   if (register_based)
      ov[n++] = (struct agx3_field_kv)KV(31, d->base_register);
   if (d->base_kill)
      ov[n++] = (struct agx3_field_kv)KV(42, 1);
   if (d->has_address_selector)
      ov[n++] = (struct agx3_field_kv)KV(70, d->address_selector);
   else if (indexed)
      ov[n++] = (struct agx3_field_kv)KV(70, d->width);
   if (is_load) {
      ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
      if (d->b_status_load) {
         ov[n++] = (struct agx3_field_kv)KV(72, 2);
         ov[n++] = (struct agx3_field_kv)KV(73, 2);
      }
      if (enc == AGX_ISA_ENCODING_AGX3C) {
         ov[n++] = (struct agx3_field_kv)KV(11, d->zero_extend);
         ov[n++] =
            (struct agx3_field_kv)KV(76, d->c_load_marker ? 2 : 0);
         if (d->c_status_load) {
            ov[n++] = (struct agx3_field_kv)KV(75, 2);
            ov[n++] = (struct agx3_field_kv)KV(77, 2);
         }
      }
   } else {
      ov[n++] = (struct agx3_field_kv)KV(13, d->value_kill);
      if (d->c_store_marker)
         ov[n++] = (struct agx3_field_kv)KV(76, 1);
   }
   for (unsigned i = 0; i < slot_count(enc); ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(60 + i, 1);
   }

   uint8_t bytes[AGX3_SCS_MAX_LENGTH] = {0};
   uint8_t len = 0;
   if (!agx3_bind_compose(enc, is_load ? AGX3_MN_ldscs : AGX3_MN_stscs, 0, base, nb,
                          ov, n, bytes, &len) ||
       len > sizeof(bytes))
      return false;

   memcpy(out, bytes, len);
   *length = len;
   return true;
}

bool
agx3_scs_encode_load(enum agx_isa_encoding enc,
                     const struct agx3_scs_desc *desc,
                     uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length)
{
   return encode(enc, desc, true, out, length);
}

bool
agx3_scs_encode_store(enum agx_isa_encoding enc,
                      const struct agx3_scs_desc *desc,
                      uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length)
{
   return encode(enc, desc, false, out, length);
}

static uint16_t
word(const uint8_t *bytes, unsigned i)
{
   return (uint16_t)bytes[2 * i] | ((uint16_t)bytes[2 * i + 1] << 8);
}

static unsigned
inverse_width(unsigned value)
{
   /* Inverse of Apple's f10 LUT {2, 0, 1, 3, 4}. */
   return value == 2 ? AGX3_MEM_W8
          : value == 0 ? AGX3_MEM_W16
          : value == 1 ? AGX3_MEM_W32
                       : 3;
}

static uint16_t
decode_c_data(const uint8_t *b)
{
   const uint16_t w0 = word(b, 0), w1 = word(b, 1), w3 = word(b, 3);
   return ((w1 >> 8) & 1u) | (((w0 >> 4) & 1u) << 1) |
          (((w3 >> 11) & 3u) << 2) | (((w0 >> 7) & 1u) << 4) |
          (((w3 >> 13) & 1u) << 5) | (((w1 >> 7) & 1u) << 6) |
          (((w1 >> 3) & 3u) << 7);
}

static uint16_t
decode_c_offset(const uint8_t *b, bool is_load)
{
   const uint16_t w1 = word(b, 1), w2 = word(b, 2), w4 = word(b, 4);
   const uint16_t w5 = word(b, 5), w7 = is_load ? word(b, 7) : 0;
   uint16_t v = ((w4 >> 12) & 0xfu) | (((w4 >> 4) & 3u) << 4) |
                (((w4 >> 10) & 1u) << 6) | (((w5 >> 12) & 1u) << 7) |
                (((w5 >> 6) & 1u) << 8) |
                (((w1 >> 13) & 3u) << 14);
   if (is_load)
      v |= ((w7 >> 9) & 0x1fu) << 9;
   else
      v |= (((w5 >> 15) & 1u) << 9) | (((w2 >> 3) & 1u) << 10) |
           (((w2 >> 8) & 7u) << 11);
   return v;
}

static uint16_t
decode_waits(enum agx_isa_encoding enc, const uint8_t *b)
{
   const uint16_t w0 = word(b, 0), w1 = word(b, 1);
   if (enc == AGX_ISA_ENCODING_AGX3B)
      return (uint16_t)(((w0 >> 12) & 0xfu) | ((w1 & 3u) << 4));

   const uint16_t w3 = word(b, 3);
   return (uint16_t)(((w0 >> 10) & 0x1fu) | (((w3 >> 15) & 1u) << 5) |
                     (((w0 >> 15) & 1u) << 6) |
                     (((w0 >> 3) & 1u) << 7));
}

bool
agx3_scs_encode_epilogue(enum agx_isa_encoding enc,
                          const struct agx3_scs_epilogue_desc *d,
                          uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_SCS_MAX_LENGTH);
   *length = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || d == NULL || d->index >= 256 ||
       d->base_register >= 256 || (d->byte_offset & 3) ||
       d->component_mask == 0 || d->component_mask > 15 ||
       (d->address_selector != 2 && d->address_selector != 4))
      return false;

   const struct agx3_field_kv base[] = {
      KV(10, 2), KV(23, 2), KV(24, 1), KV(28, 1), KV(38, 2),
      KV(39, 2), KV(45, 15), KV(53, 1), KV(54, 1), KV(59, 106), KV(80, 1),
   };
   const struct agx3_field_kv fields[] = {
      KV(16, d->index), KV(31, d->base_register),
      KV(27, d->index_kill), KV(42, d->base_kill),
      KV(68, d->component_mask), KV(70, d->address_selector),
      KV(71, (uint16_t)d->byte_offset), KV(76, d->c_load_marker ? 2 : 0),
   };
   return agx3_bind_compose(enc, AGX3_MN_ldscs, 0, base,
                           sizeof(base) / sizeof(base[0]), fields,
                           sizeof(fields) / sizeof(fields[0]), out, length);
}

bool
agx3_scs_decode_epilogue(enum agx_isa_encoding enc, const uint8_t *bytes,
                          size_t length, struct agx3_scs_epilogue_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (enc != AGX_ISA_ENCODING_AGX3C || bytes == NULL || length != 16)
      return false;

   const uint16_t w0 = word(bytes, 0), w1 = word(bytes, 1);
   const uint16_t w2 = word(bytes, 2), w3 = word(bytes, 3);
   const uint16_t w4 = word(bytes, 4), w5 = word(bytes, 5);
   struct agx3_scs_epilogue_desc d = {
      .index = ((w5 >> 4) & 3u) | (((w5 >> 7) & 0x1fu) << 2) |
               (((w5 >> 13) & 1u) << 7) | ((w4 & 1u) << 8),
      .base_register = ((w2 >> 12) & 0xfu) | (((w3 >> 14) & 1u) << 4) |
                       (((w1 >> 10) & 1u) << 5) | (((w1 >> 15) & 1u) << 6) |
                       (((w0 >> 8) & 3u) << 7),
      .byte_offset = (int16_t)decode_c_offset(bytes, true),
      .index_kill = (w4 >> 6) & 1u,
      .base_kill = (w2 >> 7) & 1u,
      .c_load_marker = ((w2 >> 5) & 3u) == 0,
   };

   /* Only accept a descriptor whose complete re-encoding matches. This also
    * rejects ordinary loads, unknown controls and extra operand bits. */
   for (unsigned mask = 1; mask <= 15; ++mask) {
      for (unsigned selector = 2; selector <= 4; selector += 2) {
         d.component_mask = mask;
         d.address_selector = selector;
         uint8_t canonical[AGX3_SCS_MAX_LENGTH], canonical_length;
         if (agx3_scs_encode_epilogue(enc, &d, canonical, &canonical_length) &&
             canonical_length == length && !memcmp(canonical, bytes, length)) {
            *out = d;
            return true;
         }
      }
   }
   return false;
}

/* G17P's register file is 192 halves. */
#define AGX3B_SCS_TOUCH_HALVES 192u

bool
agx3_scs_encode_touch(enum agx_isa_encoding enc,
                      const struct agx3_scs_touch_desc *d,
                      uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_SCS_MAX_LENGTH);
   *length = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || d == NULL ||
       (d->base_register & 1) ||
       d->base_register + 2u > AGX3B_SCS_TOUCH_HALVES ||
       (d->byte_offset & 3) ||
       (d->component_mask != 1 && d->component_mask != 3 &&
        d->component_mask != 15))
      return false;
   if (d->has_index
          ? (d->index >= AGX3B_SCS_TOUCH_HALVES ||
             (d->address_selector != 2 && d->address_selector != 4))
          : (d->index != 0 || d->index_kill || d->address_selector != 0))
      return false;

   /* The leaf programs' own instance is this baseline with f23=1, a two-word
    * mask and the base's final use; the AGX3c epilogue's is f23=2 plus that
    * encoding's f45/f53/f59, which no AGX3b SCS form carries. */
   const struct agx3_field_kv base[] = {
      KV(10, 2), KV(23, d->has_index ? 2 : 1), KV(24, 1), KV(28, 1),
      KV(38, 2), KV(39, 2), KV(54, 1), KV(80, 1),
   };
   const struct agx3_field_kv fields[] = {
      KV(16, d->index), KV(27, d->index_kill),
      KV(31, d->base_register), KV(42, d->base_kill),
      KV(68, d->component_mask), KV(70, d->address_selector),
      KV(71, (uint16_t)d->byte_offset),
   };
   uint8_t bytes[AGX3_SCS_MAX_LENGTH] = {0};
   uint8_t len = 0;
   if (!agx3_bind_compose(enc, AGX3_MN_ldscs, 0, base,
                          sizeof(base) / sizeof(base[0]), fields,
                          sizeof(fields) / sizeof(fields[0]), bytes, &len) ||
       len > sizeof(bytes))
      return false;

   memcpy(out, bytes, len);
   *length = len;
   return true;
}

bool
agx3_scs_decode_touch(enum agx_isa_encoding enc, const uint8_t *bytes,
                      size_t length, struct agx3_scs_touch_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (enc != AGX_ISA_ENCODING_AGX3B || bytes == NULL || length != 14)
      return false;

   /* The operands sit where the ordinary AGX3b load keeps them. */
   const uint16_t w2 = word(bytes, 2), w3 = word(bytes, 3);
   const uint16_t w4 = word(bytes, 4), w5 = word(bytes, 5);
   const uint16_t index = (uint16_t)(((w2 >> 9) & 0x7fu) | ((w3 & 1u) << 7));
   static const uint8_t masks[] = {1, 3, 15};
   static const uint8_t selectors[] = {2, 4};

   /* Only accept a descriptor whose complete re-encoding matches. This also
    * rejects ordinary loads, unknown controls and extra operand bits. */
   for (unsigned has_index = 0; has_index < 2; ++has_index) {
      for (unsigned m = 0; m < sizeof(masks); ++m) {
         for (unsigned s = 0; s < (has_index ? sizeof(selectors) : 1); ++s) {
            for (unsigned kills = 0; kills < (has_index ? 4u : 2u); ++kills) {
               const struct agx3_scs_touch_desc d = {
                  .base_register = w2 & 0xffu,
                  .has_index = has_index,
                  .index = has_index ? index : 0,
                  .byte_offset = (int16_t)(((w4 >> 15) & 1u) |
                                           ((w5 & 0x7fffu) << 1)),
                  .component_mask = masks[m],
                  .address_selector = has_index ? selectors[s] : 0,
                  .index_kill = kills & 2,
                  .base_kill = kills & 1,
               };
               uint8_t canonical[AGX3_SCS_MAX_LENGTH], canonical_length;
               if (agx3_scs_encode_touch(enc, &d, canonical,
                                         &canonical_length) &&
                   canonical_length == length &&
                   !memcmp(canonical, bytes, length)) {
                  *out = d;
                  return true;
               }
            }
         }
      }
   }
   return false;
}

static bool
decode_address(enum agx_isa_encoding enc, const uint8_t *b,
               struct agx3_scs_desc *d)
{
   const bool indexed =
      enc == AGX_ISA_ENCODING_AGX3B ? ((word(b, 3) >> 5) & 1u)
                                    : ((word(b, 4) >> 1) & 1u);
   if (!indexed) {
      d->address_kind = AGX3_SCS_ADDRESS_DIRECT;
      return true;
   }

   d->address_kind = AGX3_SCS_ADDRESS_UNIFORM_INDEXED;
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      const uint16_t w2 = word(b, 2), w3 = word(b, 3);
      d->uniform_base = w2 & 0x1ffu;
      d->index = (uint16_t)(((w2 >> 9) & 0x7fu) | ((w3 & 1u) << 7));
   } else {
      const uint16_t w0 = word(b, 0), w1 = word(b, 1);
      const uint16_t w2 = word(b, 2), w3 = word(b, 3);
      const uint16_t w4 = word(b, 4), w5 = word(b, 5);
      d->uniform_base =
         (uint16_t)(((w2 >> 12) & 0xfu) | (((w3 >> 14) & 1u) << 4) |
                    (((w1 >> 10) & 1u) << 5) |
                    (((w1 >> 15) & 1u) << 6) |
                    (((w0 >> 8) & 3u) << 7));
      d->index =
         (uint16_t)(((w5 >> 4) & 3u) | (((w5 >> 7) & 0x1fu) << 2) |
                    (((w5 >> 13) & 1u) << 7) | ((w4 & 1u) << 8));
   }
   return true;
}

static bool
decode(enum agx_isa_encoding enc, const uint8_t *bytes, size_t length,
       bool is_load, struct agx3_scs_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (bytes == NULL || slot_count(enc) == 0 ||
       length != (enc == AGX_ISA_ENCODING_AGX3C && is_load ? 16 : 14))
      return false;

   struct agx3_scs_desc d = {0};
   const uint16_t w1 = word(bytes, 1), w2 = word(bytes, 2);
   const uint16_t w4 = word(bytes, 4), w5 = word(bytes, 5);
   d.data = enc == AGX_ISA_ENCODING_AGX3B ? w1 >> 8 : decode_c_data(bytes);
   const unsigned raw_width =
      enc == AGX_ISA_ENCODING_AGX3B
         ? (w4 >> 6) & 3u
         : ((w1 >> 9) & 1u) | (((w1 >> 11) & 1u) << 1);
   d.width = (enum agx3_mem_width)inverse_width(raw_width);
   if (!width_ok(d.width))
      return false;

   if (!decode_address(enc, bytes, &d))
      return false;

   d.wait_mask = decode_waits(enc, bytes);
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      const uint16_t offset =
         is_load ? ((w4 >> 15) & 1u) | ((w5 & 0x7fffu) << 1)
                 : ((w4 >> 13) & 7u) | ((w5 & 0x1fffu) << 3);
      d.byte_offset = (int16_t)offset;
      d.value_kill = !is_load && ((w4 >> 9) & 1u);
   } else {
      d.byte_offset = (int16_t)decode_c_offset(bytes, is_load);
      d.value_kill = !is_load && (w5 & 1u);
      d.index_kill = (w4 >> 6) & 1u;
      d.base_kill = (w2 >> 7) & 1u;
      d.c_store_marker = !is_load && ((w2 >> 5) & 3u) == 2;
      if (is_load) {
         d.zero_extend = (w2 >> 3) & 1u;
         const unsigned raw_marker = (w2 >> 5) & 3u;
         if (raw_marker != 0 && raw_marker != 1)
            return false;
         d.c_load_marker = raw_marker == 0;
         d.c_status_load = ((word(bytes, 0) >> 5) & 3u) == 0 &&
                           ((word(bytes, 3) >> 5) & 3u) == 0;
      }
   }

   const unsigned first_slot = is_load ? 1 : 0;
   const unsigned last_slot = is_load ? slot_count(enc) : 0;
   if (is_load && enc == AGX_ISA_ENCODING_AGX3C) {
      d.slot = (uint8_t)(((w2 >> 8) & 7u) + 1);
      if (d.slot > last_slot)
         return false;
   }

   /*
    * Trial decode.  Every candidate is re-encoded and compared byte-for-byte,
    * so a wrong guess fails closed instead of reporting a wrong field.  The
    * first candidate is exactly what this decoder tried before -- the address
    * kind read from the mode bit, a single component and no explicit selector
    * -- so previously accepted encodings still take the same path and yield
    * the same descriptor.
    *
    * Later candidates cover the register-base kinds, whose base register
    * shares slot B with the uniform field, and the vector component masks.
    * AGX3c uses the same split slot-B field for uniform and register bases.
    */
   static const uint8_t mask_candidates[] = {1, 3, 15, 2, 4, 5, 6, 7,
                                             8, 9, 10, 11, 12, 13, 14};
   const enum agx3_scs_address_kind decoded_kind = d.address_kind;
   const uint16_t decoded_index = d.index;
   const uint16_t decoded_uniform = d.uniform_base;
   /* Slot B's register operand occupies the low byte of the same word the
    * uniform base is read from. */
   const uint16_t slot_b_register = enc == AGX_ISA_ENCODING_AGX3C
      ? (uint16_t)(((w2 >> 12) & 0xfu) |
                   (((word(bytes, 3) >> 14) & 1u) << 4) |
                   (((w1 >> 10) & 1u) << 5) | (((w1 >> 15) & 1u) << 6) |
                   (((word(bytes, 0) >> 8) & 3u) << 7))
      : (uint16_t)(w2 & 0xffu);

   for (unsigned kind_try = 0; kind_try < 3; ++kind_try) {
      enum agx3_scs_address_kind kind = decoded_kind;
      uint16_t index = decoded_index, uniform = decoded_uniform, base = 0;

      if (kind_try == 1) {
         kind = AGX3_SCS_ADDRESS_REGISTER_INDEXED;
         index = decoded_index;
         uniform = 0;
         base = slot_b_register;
      } else if (kind_try == 2) {
         kind = AGX3_SCS_ADDRESS_REGISTER_BASE;
         index = 0;
         uniform = 0;
         base = slot_b_register;
      }

      for (unsigned m = 0; m < sizeof(mask_candidates) / sizeof(uint8_t); ++m) {
         for (unsigned sel = 0; sel < 6; ++sel) {
            d.address_kind = kind;
            d.index = index;
            d.uniform_base = uniform;
            d.base_register = base;
            d.component_mask = mask_candidates[m];
            d.has_address_selector = sel != 0;
            d.address_selector = sel != 0 ? (uint8_t)(sel - 1) : 0u;

   for (unsigned status = 0; status <= (is_load && enc == AGX_ISA_ENCODING_AGX3B); ++status) {
   d.b_status_load = status != 0;
   for (unsigned slot = first_slot; slot <= last_slot; ++slot) {
      if (!(is_load && enc == AGX_ISA_ENCODING_AGX3C))
         d.slot = (uint8_t)slot;
      uint8_t canonical[AGX3_SCS_MAX_LENGTH];
      uint8_t canonical_length = 0;
      const bool ok = is_load
                         ? agx3_scs_encode_load(enc, &d, canonical,
                                                &canonical_length)
                         : agx3_scs_encode_store(enc, &d, canonical,
                                                 &canonical_length);
      if (ok && canonical_length == length &&
          memcmp(canonical, bytes, length) == 0) {
         *out = d;
         return true;
      }
      if (is_load && enc == AGX_ISA_ENCODING_AGX3C)
         break;
   }
   }
         }
      }
   }
   return false;
}

bool
agx3_scs_decode_load(enum agx_isa_encoding enc, const uint8_t *bytes,
                     size_t length, struct agx3_scs_desc *out)
{
   return decode(enc, bytes, length, true, out);
}

bool
agx3_scs_decode_store(enum agx_isa_encoding enc, const uint8_t *bytes,
                      size_t length, struct agx3_scs_desc *out)
{
   return decode(enc, bytes, length, false, out);
}
