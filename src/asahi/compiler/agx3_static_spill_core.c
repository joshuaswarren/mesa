/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_static_spill_core.h"

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
   return width == AGX3_MEM_W16 || width == AGX3_MEM_W32;
}

static bool
store_ok(enum agx_isa_encoding enc,
         const struct agx3_static_spill_store_desc *d)
{
   const unsigned slots = slot_count(enc);
   if (d == NULL || slots == 0 || !width_ok(d->width) || d->src >= 256 ||
       d->tps_half > AGX3_STATIC_SPILL_MAX_TPS_HALF || d->slot < 1 ||
       d->slot > slots || d->wait_mask >= (1u << slots))
      return false;

   const unsigned halves = d->width == AGX3_MEM_W32 ? 2 : 1;
   return d->src + halves <= 256 &&
          d->tps_half + halves <= AGX3_STATIC_SPILL_MAX_TPS_HALF + 1u &&
          (d->width != AGX3_MEM_W32 || (((d->src | d->tps_half) & 1u) == 0));
}

static bool
load_ok(enum agx_isa_encoding enc, const struct agx3_static_spill_load_desc *d)
{
   const unsigned slots = slot_count(enc);
   if (d == NULL || slots == 0 || !width_ok(d->width) || d->dest >= 256 ||
       d->tps_half > AGX3_STATIC_SPILL_MAX_TPS_HALF || d->slot < 1 ||
       d->slot > slots || d->slot_mirror > slots ||
       d->wait_mask >= (1u << slots))
      return false;

   const unsigned halves = d->width == AGX3_MEM_W32 ? 2 : 1;
   return d->dest + halves <= 256 &&
          d->tps_half + halves <= AGX3_STATIC_SPILL_MAX_TPS_HALF + 1u &&
          (d->width != AGX3_MEM_W32 || (((d->dest | d->tps_half) & 1u) == 0));
}

static unsigned
append_waits(const struct agx3_static_spill_store_desc *d,
             struct agx3_field_kv *ov, unsigned n)
{
   for (unsigned i = 0; i < 8; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }
   return n;
}

static bool
encode_store_form(enum agx_isa_encoding enc,
                  const struct agx3_static_spill_store_desc *d, uint8_t ep,
                  uint8_t *out, uint8_t *length)
{
   if (!store_ok(enc, d) || ep > 1)
      return false;

   struct agx3_field_kv base[4];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 4);
   base[nb++] = (struct agx3_field_kv)KV(23, 2);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 1);
      base[nb++] = (struct agx3_field_kv)KV(29, 35);
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(4, d->tps_half);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(16, d->src);
   ov[n++] = (struct agx3_field_kv)KV(24, d->width);
   ov[n++] = (struct agx3_field_kv)KV(27, d->src_kill);
   ov[n++] = (struct agx3_field_kv)KV(enc == AGX_ISA_ENCODING_AGX3C ? 42 : 39,
                                      d->structural_flag);
   n = append_waits(d, ov, n);

   return agx3_bind_compose(enc, AGX3_MN_copy, ep, base, nb, ov, n, out, length);
}

bool
agx3_static_spill_encode_store_form(
   enum agx_isa_encoding enc, const struct agx3_static_spill_store_desc *desc,
   uint8_t ep, uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH], uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_STATIC_SPILL_MAX_LENGTH);
   *length = 0;

   uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH] = {0};
   uint8_t len = 0;
   if (!encode_store_form(enc, desc, ep, bytes, &len) || len > sizeof(bytes))
      return false;
   memcpy(out, bytes, len);
   *length = len;
   return true;
}

bool
agx3_static_spill_encode_store(enum agx_isa_encoding enc,
                               const struct agx3_static_spill_store_desc *desc,
                               uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH],
                               uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_STATIC_SPILL_MAX_LENGTH);
   *length = 0;
   if (!store_ok(enc, desc))
      return false;

   /* Match Apple's witnessed static-copy form choice. AGX3b uses compact ep0
    * for 16-bit spills and ep1 for 32-bit spills; AGX3c uses ep1.  A 16-bit
    * AGX3b value that does not fit ep0 may still use the grounded ep1 form. */
   const uint8_t first =
      enc == AGX_ISA_ENCODING_AGX3B && desc->width == AGX3_MEM_W16 ? 0 : 1;
   if (agx3_static_spill_encode_store_form(enc, desc, first, out, length))
      return true;
   return first == 0 &&
          agx3_static_spill_encode_store_form(enc, desc, 1, out, length);
}

static bool
encode_load_form(enum agx_isa_encoding enc,
                 const struct agx3_static_spill_load_desc *d, uint8_t ep,
                 uint8_t *out, uint8_t *length)
{
   if (!load_ok(enc, d) || ep > 1)
      return false;

   struct agx3_field_kv base[5];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 4);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 4);
      base[nb++] = (struct agx3_field_kv)KV(29, 144);
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(11, d->narrow_extend);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(18, d->tps_half);
   ov[n++] = (struct agx3_field_kv)KV(24, d->width);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot_mirror);
   ov[n++] = (struct agx3_field_kv)KV(27, !d->tps_live);
   ov[n++] = (struct agx3_field_kv)KV(enc == AGX_ISA_ENCODING_AGX3C ? 44 : 41,
                                      d->structural_flag);
   for (unsigned i = 0; i < 8; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }

   return agx3_bind_compose(enc, AGX3_MN_rld, ep, base, nb, ov, n, out, length);
}

bool
agx3_static_spill_encode_load_form(
   enum agx_isa_encoding enc, const struct agx3_static_spill_load_desc *desc,
   uint8_t ep, uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH], uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_STATIC_SPILL_MAX_LENGTH);
   *length = 0;

   uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH] = {0};
   uint8_t len = 0;
   if (!encode_load_form(enc, desc, ep, bytes, &len) || len > sizeof(bytes))
      return false;
   memcpy(out, bytes, len);
   *length = len;
   return true;
}

bool
agx3_static_spill_encode_load(enum agx_isa_encoding enc,
                              const struct agx3_static_spill_load_desc *desc,
                              uint8_t out[AGX3_STATIC_SPILL_MAX_LENGTH],
                              uint8_t *length)
{
   if (out == NULL || length == NULL)
      return false;
   memset(out, 0, AGX3_STATIC_SPILL_MAX_LENGTH);
   *length = 0;
   if (!load_ok(enc, desc))
      return false;

   for (uint8_t ep = 0; ep <= 1; ++ep) {
      if (agx3_static_spill_encode_load_form(enc, desc, ep, out, length))
         return true;
   }
   return false;
}

static unsigned
get_bit(const uint8_t *bytes, unsigned bit)
{
   return (bytes[bit >> 3] >> (bit & 7)) & 1u;
}

static unsigned
get_bits(const uint8_t *bytes, unsigned bit, unsigned width)
{
   unsigned value = 0;
   for (unsigned i = 0; i < width; ++i)
      value |= get_bit(bytes, bit + i) << i;
   return value;
}

static uint32_t
deposit_bit(uint32_t value, unsigned dst, const uint8_t *bytes, unsigned src)
{
   return value | ((uint32_t)get_bit(bytes, src) << dst);
}

/* The extraction below is the inverse of Apple's anonymous-field deposits.
 * A canonical re-encode is always compared afterwards, so opcode constants,
 * selector LUTs, mirrored slot fields and every structural bit must agree. */
static void
decode_compact_copy_operands(const uint8_t *b, uint32_t *tps, uint16_t *src)
{
   uint32_t o = 0;
   o = deposit_bit(o, 0, b, 24);
   o |= (uint32_t)get_bits(b, 4, 4) << 1;
   o |= (uint32_t)get_bits(b, 22, 2) << 5;
   o = deposit_bit(o, 15, b, 26);
   o = deposit_bit(o, 16, b, 21);
   *tps = o;

   *src = (uint16_t)(get_bits(b, 8, 7) | (get_bit(b, 28) << 7));
}

static void
decode_b_long_copy_operands(const uint8_t *b, uint32_t *tps, uint16_t *src)
{
   uint32_t o = 0;
   o = deposit_bit(o, 0, b, 34);
   o |= (uint32_t)get_bits(b, 4, 4) << 1;
   o |= (uint32_t)get_bits(b, 22, 2) << 5;
   o = deposit_bit(o, 7, b, 44);
   o |= (uint32_t)get_bits(b, 50, 4) << 8;
   o = deposit_bit(o, 12, b, 67);
   o = deposit_bit(o, 13, b, 68);
   o = deposit_bit(o, 14, b, 76);
   o = deposit_bit(o, 15, b, 38);
   o = deposit_bit(o, 16, b, 21);
   *tps = o;

   *src = (uint16_t)(get_bit(b, 35) | (get_bits(b, 9, 6) << 1) |
                     (get_bit(b, 40) << 7));
}

static void
decode_c_long_copy_operands(const uint8_t *b, uint32_t *tps, uint16_t *src)
{
   static const uint8_t positions[17] = {
      24, 4, 59, 60, 7, 61, 23, 19, 20, 62, 35, 36, 39, 40, 41, 26, 42,
   };
   uint32_t o = 0;
   for (unsigned i = 0; i < 17; ++i)
      o = deposit_bit(o, i, b, positions[i]);
   *tps = o;

   *src = (uint16_t)(get_bits(b, 8, 2) | (get_bits(b, 27, 5) << 2) |
                     (get_bit(b, 48) << 7));
}

static uint16_t
decode_compact_wait(uint8_t code)
{
   if (code == 0)
      return 0;
   if (code <= 6)
      return (uint16_t)(1u << (code - 1));
   return 3; /* Apple's compact wait1+wait2 selector */
}

static uint16_t
decode_b_long_wait(const uint8_t *b)
{
   return (uint16_t)(get_bits(b, 45, 3) | (get_bits(b, 61, 3) << 3));
}

static uint16_t
decode_c_long_wait(const uint8_t *b)
{
   return (uint16_t)(get_bits(b, 10, 5) | (get_bit(b, 63) << 5) |
                     (get_bit(b, 22) << 6) | (get_bit(b, 3) << 7));
}

bool
agx3_static_spill_decode_store(enum agx_isa_encoding enc, const uint8_t *bytes,
                               size_t length,
                               struct agx3_static_spill_store_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (bytes == NULL || slot_count(enc) == 0)
      return false;

   uint8_t ep;
   struct agx3_static_spill_store_desc d = {0};
   if (length == 4) {
      ep = 0;
      decode_compact_copy_operands(bytes, &d.tps_half, &d.src);
      d.slot = 1;
      d.wait_mask = decode_compact_wait((uint8_t)get_bits(bytes, 29, 3));
   } else if (enc == AGX_ISA_ENCODING_AGX3B && length == 10) {
      ep = 1;
      decode_b_long_copy_operands(bytes, &d.tps_half, &d.src);
      d.slot = (uint8_t)(get_bits(bytes, 58, 3) + 1);
      d.wait_mask = decode_b_long_wait(bytes);
   } else if (enc == AGX_ISA_ENCODING_AGX3C && length == 8) {
      ep = 1;
      decode_c_long_copy_operands(bytes, &d.tps_half, &d.src);
      d.slot = (uint8_t)(get_bits(bytes, 44, 3) + 1);
      d.wait_mask = decode_c_long_wait(bytes);
   } else {
      return false;
   }

   for (unsigned width = AGX3_MEM_W16; width <= AGX3_MEM_W32; ++width) {
      d.width = (enum agx3_mem_width)width;
      for (unsigned kill = 0; kill < 2; ++kill) {
         d.src_kill = kill;
         for (unsigned structural = 0; structural < 2; ++structural) {
            d.structural_flag = structural;
            uint8_t canonical[AGX3_STATIC_SPILL_MAX_LENGTH];
            uint8_t canonical_length = 0;
            if (agx3_static_spill_encode_store_form(enc, &d, ep, canonical,
                                                    &canonical_length) &&
                canonical_length == length &&
                memcmp(canonical, bytes, length) == 0) {
               *out = d;
               return true;
            }
         }
      }
   }
   return false;
}

static void
decode_compact_rld_operands(const uint8_t *b, uint16_t *dest, uint32_t *tps)
{
   *dest = (uint16_t)(get_bit(b, 18) | (get_bits(b, 4, 4) << 1) |
                      (get_bits(b, 22, 2) << 5));
   *tps = get_bits(b, 8, 7);
}

static void
decode_b_long_rld_operands(const uint8_t *b, uint16_t *dest, uint32_t *tps)
{
   decode_compact_rld_operands(b, dest, tps);
   *dest |= (uint16_t)(get_bit(b, 60) << 7);
   *tps |= (uint32_t)get_bits(b, 33, 4) << 7;
   *tps |= (uint32_t)get_bits(b, 42, 2) << 11;
   *tps |= (uint32_t)get_bits(b, 48, 4) << 13;
}

static void
decode_c_long_rld_operands(const uint8_t *b, uint16_t *dest, uint32_t *tps)
{
   static const uint8_t dest_positions[8] = {24, 4, 59, 60, 7, 61, 23, 19};
   static const uint8_t tps_positions[17] = {
      8, 9, 27, 28, 25, 30, 31, 29, 49, 62, 36, 39, 44, 45, 46, 47, 26,
   };
   uint16_t d = 0;
   uint32_t o = 0;
   for (unsigned i = 0; i < 8; ++i)
      d |= (uint16_t)(get_bit(b, dest_positions[i]) << i);
   for (unsigned i = 0; i < 17; ++i)
      o = deposit_bit(o, i, b, tps_positions[i]);
   *dest = d;
   *tps = o;
}

bool
agx3_static_spill_decode_load(enum agx_isa_encoding enc, const uint8_t *bytes,
                              size_t length,
                              struct agx3_static_spill_load_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (bytes == NULL || slot_count(enc) == 0)
      return false;

   uint8_t ep;
   struct agx3_static_spill_load_desc d = {0};
   if (length == 4) {
      ep = 0;
      decode_compact_rld_operands(bytes, &d.dest, &d.tps_half);
      d.slot = (uint8_t)(get_bits(bytes, 29, 3) + 1);
   } else if (enc == AGX_ISA_ENCODING_AGX3B && length == 8) {
      ep = 1;
      decode_b_long_rld_operands(bytes, &d.dest, &d.tps_half);
      d.slot = (uint8_t)(get_bits(bytes, 29, 3) + 1);
      d.wait_mask = decode_b_long_wait(bytes);
   } else if (enc == AGX_ISA_ENCODING_AGX3C && length == 8) {
      ep = 1;
      decode_c_long_rld_operands(bytes, &d.dest, &d.tps_half);
      d.slot = (uint8_t)(get_bits(bytes, 40, 3) + 1);
      d.wait_mask = decode_c_long_wait(bytes);
   } else {
      return false;
   }

   for (unsigned width = AGX3_MEM_W16; width <= AGX3_MEM_W32; ++width) {
      d.width = (enum agx3_mem_width)width;
      for (unsigned mirror = 0; mirror <= slot_count(enc); ++mirror) {
         d.slot_mirror = mirror;
         for (unsigned narrow = 0; narrow < 2; ++narrow) {
            d.narrow_extend = narrow;
             for (unsigned structural = 0; structural < 2; ++structural) {
                d.structural_flag = structural;
                for (unsigned live = 0; live < 2; ++live) {
                  d.tps_live = live;
                  uint8_t canonical[AGX3_STATIC_SPILL_MAX_LENGTH];
                  uint8_t canonical_length = 0;
                  if (agx3_static_spill_encode_load_form(
                         enc, &d, ep, canonical, &canonical_length) &&
                      canonical_length == length &&
                      memcmp(canonical, bytes, length) == 0) {
                     *out = d;
                      return true;
                   }
                }
            }
         }
      }
   }
   return false;
}
