/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_local_tps_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static unsigned
slot_count(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3C ? 8 :
          enc == AGX_ISA_ENCODING_AGX3B ? 6 : 0;
}

static bool
width_ok(enum agx3_mem_width width)
{
   return width == AGX3_MEM_W8 || width == AGX3_MEM_W16 ||
          width == AGX3_MEM_W32;
}

static bool
local_ok(enum agx_isa_encoding enc, const struct agx3_local_desc *d,
         bool is_load)
{
   const unsigned slots = slot_count(enc);
   if (d == NULL || slots == 0 || d->data >= 256 || !width_ok(d->width) ||
       d->mask < 1 || d->mask > 15 || d->shift > 4 || d->index >= 256 ||
       d->base_kind < AGX3_LOCAL_BASE_IMMEDIATE ||
       d->base_kind > AGX3_LOCAL_BASE_UNIFORM ||
       d->wait_mask >= (1u << slots) ||
       (is_load && (d->slot < 1 || d->slot > slots)) ||
       (!is_load && d->slot != 0))
      return false;

   if (d->base_kind == AGX3_LOCAL_BASE_REGISTER && d->base >= 256)
      return false;
   if (d->base_kind == AGX3_LOCAL_BASE_UNIFORM && d->base >= 512)
      return false;
   if (enc != AGX_ISA_ENCODING_AGX3C && d->c_load_marker)
      return false;
   return true;
}

static unsigned
local_overrides(const struct agx3_local_desc *d, bool is_load,
                struct agx3_field_kv *ov)
{
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->data);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(11, d->narrow_extend);
   ov[n++] = (struct agx3_field_kv)KV(68, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(70, d->shift);
   ov[n++] = (struct agx3_field_kv)KV(71, d->byte_offset);
   ov[n++] = (struct agx3_field_kv)KV(24, 1);

   if (d->index_is_register) {
      ov[n++] = (struct agx3_field_kv)KV(23, 2);
      ov[n++] = (struct agx3_field_kv)KV(16, d->index);
      ov[n++] = (struct agx3_field_kv)KV(27, d->index_kill);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(23, 1);
      ov[n++] = (struct agx3_field_kv)KV(15, d->index);
   }

   ov[n++] = (struct agx3_field_kv)KV(38, d->base_kind);
   if (d->base_kind == AGX3_LOCAL_BASE_IMMEDIATE)
      ov[n++] = (struct agx3_field_kv)KV(30, d->base);
   else if (d->base_kind == AGX3_LOCAL_BASE_REGISTER) {
      ov[n++] = (struct agx3_field_kv)KV(31, d->base);
      ov[n++] = (struct agx3_field_kv)KV(42, d->base_kill);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(32, d->base);
   }

   if (is_load) {
      ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
      ov[n++] = (struct agx3_field_kv)KV(76, d->c_load_marker ? 2 : 0);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(13, d->value_kill);
   }

   for (unsigned i = 0; i < 8; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(60 + i, 1);
   }
   return n;
}

static unsigned
local_baseline(enum agx_isa_encoding enc, bool is_load,
               struct agx3_field_kv *base)
{
   unsigned n = 0;
   base[n++] = (struct agx3_field_kv)KV(9, 2);
   base[n++] = (struct agx3_field_kv)KV(39, 1);
   base[n++] = (struct agx3_field_kv)KV(54, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[n++] = (struct agx3_field_kv)KV(45, 15);
      base[n++] = (struct agx3_field_kv)KV(53, 1);
      base[n++] = (struct agx3_field_kv)KV(59, is_load ? 108 : 189);
   }
   return n;
}

static bool
local_form(enum agx_isa_encoding enc, const struct agx3_local_desc *d,
           bool is_load, uint8_t ep, uint8_t *out, uint8_t *len)
{
   if (!local_ok(enc, d, is_load) ||
       (enc == AGX_ISA_ENCODING_AGX3B ? ep != 0 : ep > 5))
      return false;

   struct agx3_field_kv base[6];
   const unsigned nbase = local_baseline(enc, is_load, base);
   struct agx3_field_kv ov[32];
   const unsigned n = local_overrides(d, is_load, ov);
   return agx3_bind_compose(enc, is_load ? AGX3_MN_ldtg : AGX3_MN_sttg, ep, base,
                            nbase, ov, n, out, len);
}

static bool
local_auto(enum agx_isa_encoding enc, const struct agx3_local_desc *d,
           bool is_load, uint8_t *out, uint8_t *len)
{
   const unsigned forms = enc == AGX_ISA_ENCODING_AGX3C ? 6 : 1;
   for (unsigned ep = 0; ep < forms; ++ep) {
      if (local_form(enc, d, is_load, ep, out, len))
         return true;
   }
   return false;
}

bool
agx3_local_encode_load(enum agx_isa_encoding enc,
                       const struct agx3_local_desc *d,
                       uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH], uint8_t *len)
{
   return local_auto(enc, d, true, out, len);
}

bool
agx3_local_encode_store(enum agx_isa_encoding enc,
                        const struct agx3_local_desc *d,
                        uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH], uint8_t *len)
{
   return local_auto(enc, d, false, out, len);
}

static bool
tps_ok(enum agx_isa_encoding enc, const struct agx3_tps_desc *d, bool is_load)
{
   const unsigned slots = slot_count(enc);
   return d != NULL && slots != 0 && d->data < 256 && width_ok(d->width) &&
          d->mask >= 1 && d->mask <= 15 && d->shift <= 3 && d->index < 256 &&
          (!d->index_immediate || (enc == AGX_ISA_ENCODING_AGX3C &&
           d->index == 0 && !d->index_kill && !d->index_signed)) &&
          d->frame_region >= 1 && d->frame_region <= 6 &&
          d->wait_mask < (1u << slots) &&
          (!is_load || (d->slot >= 1 && d->slot <= slots)) &&
          (is_load || d->slot == 0);
}

static unsigned
tps_overrides(const struct agx3_tps_desc *d, bool is_load,
              struct agx3_field_kv *ov)
{
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->data);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(11, d->narrow_extend);
   ov[n++] = (struct agx3_field_kv)KV(16, d->index);
   ov[n++] = (struct agx3_field_kv)KV(23, d->index_immediate ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(24, d->index_immediate ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(27, d->index_kill);
   ov[n++] = (struct agx3_field_kv)KV(28, d->index_signed);
   ov[n++] = (struct agx3_field_kv)KV(41, d->frame_region);
   ov[n++] = (struct agx3_field_kv)KV(53, d->structural_bit53);
   ov[n++] = (struct agx3_field_kv)KV(54, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(56, d->shift);
   ov[n++] = (struct agx3_field_kv)KV(57, d->byte_offset);
   if (is_load)
      ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   else
      ov[n++] = (struct agx3_field_kv)KV(13, d->value_kill);

   for (unsigned i = 0; i < 8; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(45 + i, 1);
   }
   return n;
}

static unsigned
tps_baseline(enum agx_isa_encoding enc, bool is_load, bool retain_tps,
             struct agx3_field_kv *base)
{
   unsigned n = 0;
   base[n++] = (struct agx3_field_kv)KV(9, 2);
   base[n++] = (struct agx3_field_kv)KV(38, 1);
   base[n++] = (struct agx3_field_kv)KV(39, 1);
   if (is_load)
      base[n++] = (struct agx3_field_kv)KV(66, !retain_tps);
   if (enc == AGX_ISA_ENCODING_AGX3C)
      base[n++] = (struct agx3_field_kv)KV(44, is_load ? 109 : 190);
   return n;
}

static bool
tps_form(enum agx_isa_encoding enc, const struct agx3_tps_desc *d,
         bool is_load, uint8_t ep, uint8_t *out, uint8_t *len)
{
   if (!tps_ok(enc, d, is_load) ||
       (enc == AGX_ISA_ENCODING_AGX3B ? ep != 0 : ep > 5))
      return false;

   struct agx3_field_kv base[5];
   const unsigned nbase = tps_baseline(enc, is_load, d->retain_tps, base);
   struct agx3_field_kv ov[32];
   const unsigned n = tps_overrides(d, is_load, ov);
   return agx3_bind_compose(enc, is_load ? AGX3_MN_ldtps : AGX3_MN_sttps, ep, base,
                            nbase, ov, n, out, len);
}

static bool
tps_auto(enum agx_isa_encoding enc, const struct agx3_tps_desc *d,
         bool is_load, uint8_t *out, uint8_t *len)
{
   const unsigned forms = enc == AGX_ISA_ENCODING_AGX3C ? 6 : 1;
   for (unsigned ep = 0; ep < forms; ++ep) {
      if (tps_form(enc, d, is_load, ep, out, len))
         return true;
   }
   return false;
}

bool
agx3_tps_encode_load(enum agx_isa_encoding enc, const struct agx3_tps_desc *d,
                     uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH], uint8_t *len)
{
   return tps_auto(enc, d, true, out, len);
}

bool
agx3_tps_encode_store(enum agx_isa_encoding enc,
                      const struct agx3_tps_desc *d,
                      uint8_t out[AGX3_LOCAL_TPS_MAX_LENGTH], uint8_t *len)
{
   return tps_auto(enc, d, false, out, len);
}
