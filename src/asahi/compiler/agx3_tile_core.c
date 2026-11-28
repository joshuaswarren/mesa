/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_tile_core.h"

#include "agx3_binder.h"
#include "agx3_sr_core.h"
#include "agx3_m5_mrt_eot_data.h"

#include <string.h>

#define AGX_MRT_ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

/* The genuine G17G EOT builtin carries the byte offset in each coordinate
 * wire and indexes the PBE array separately in IMGWBLK. Its ordinary
 * registers end at half5; preserve the qualified RT0 eight-half allocation.
 * Each complete per-target wire occupies eight staging halves. */
uint16_t
agx3_m5_mrt_eot_nr_gprs(unsigned targets)
{
   return targets >= 1 && targets <= 8 ? 8 : 0;
}

uint16_t
agx3_m5_mrt_eot_staging_halfs(unsigned targets)
{
   return targets >= 1 && targets <= 8 ? 8 * targets : 0;
}

static bool
m5_mrt_append(uint8_t *bytes, size_t *size, enum agx3_mnemonic op,
              const struct agx3_field_kv *fields, unsigned count,
              uint8_t ep, uint8_t expected_length)
{
   uint8_t instruction[16], length = 0;
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, op, ep, fields, count,
                          NULL, 0, instruction, &length) ||
       (expected_length && length != expected_length) ||
       length > AGX3_M5_MRT_EOT_MAX_LENGTH - *size)
      return false;
   memcpy(bytes + *size, instruction, length);
   *size += length;
   return true;
}

static bool
m5_mrt_wire(uint8_t *bytes, size_t *size, unsigned rt, unsigned targets,
             bool layered)
{
   struct agx3_field_kv pack[19] = {
      KV(5, 8 * rt + 4), KV(9, 5), KV(10, 2), KV(12, rt + 1),
      KV(17, 4 + 4 * rt), KV(23, 3), KV(24, 1), KV(37, 1), KV(38, 1),
      KV(51, layered ? 2 : 1), KV(52, 2), KV(56, 1), KV(57, 118), KV(74, 8),
      KV(75, 16), KV(76, 7), KV(77, 27),
   };
   unsigned count = 17;
   if (layered) {
      pack[count++] = (struct agx3_field_kv)KV(44, 4);
      pack[count++] = (struct agx3_field_kv)KV(55, rt + 1 == targets);
   }
   if (!m5_mrt_append(bytes, size, AGX3_MN_pack_imgwir, pack, count, 0, 14))
      return false;
   if (targets == 1)
      return true;
   for (unsigned xy = 0; xy < 2; ++xy) {
      const bool last = rt + 1 == targets;
      const struct agx3_field_kv copy[] = {
         KV(5, 8 * rt + 2 * xy), KV(9, 5), KV(10, 2), KV(12, rt + 1),
         KV(16, xy ? 0 : 2), KV(23, 2), KV(24, 2),
         KV(25, !last), KV(27, last), KV(28, 1), KV(29, 35),
      };
      if (!m5_mrt_append(bytes, size, AGX3_MN_copy, copy, AGX_MRT_ARRAY_SIZE(copy),
                           rt == 0 ? 0 : 1, rt == 0 ? 4 : 8))
         return false;
   }
   return true;
}

static bool
m5_mrt_store(uint8_t *bytes, size_t *size, unsigned rt, uint8_t selector,
             bool multisample, bool layered)
{
   struct agx3_field_kv img[19] = {
      KV(5, 8 * rt), KV(9, 5), KV(10, 2), KV(12, rt + 1), KV(13, 1),
      KV(14, 2), KV(15, 1), KV(16, 86), KV(17 + rt, 1),
      KV(29, selector == 0x15 ? 1 : selector == 0x35 ? 2 :
             selector == 0x55 ? 0 : selector == 0x95 ? 4 : 3),
      KV(30, multisample ? 2 : 1), KV(32, 1), KV(45, 3), KV(46, 3), KV(51, rt),
      KV(59, 1), KV(60, 1), KV(73, 1),
   };
   unsigned count = 18;
   if (layered)
      img[count++] = (struct agx3_field_kv)KV(31, 2);
   return m5_mrt_append(bytes, size, AGX3_MN_imgwblk, img, count, 0, 12);
}

/* Reuse the qualified RT0 array encoding: SR12 is the tile layer, packed
 * into ordinary half 4 before every target's XY coordinate wire consumes it.
 * Only the last target releases the shared layer operand. */
static bool
m5_mrt_layer(uint8_t *bytes, size_t *size)
{
   const struct agx3_sr_desc layer = {
      .dest = 6, .width = AGX3_SR_W16, .sr = 12, .slot = 2,
   };
   uint8_t sr[AGX3_SR_MAX_LENGTH], length;
   if (!agx3_sr_encode(AGX_ISA_ENCODING_AGX3C, &layer, sr, &length) ||
       length > AGX3_M5_MRT_EOT_MAX_LENGTH - *size)
      return false;
   memcpy(bytes + *size, sr, length);
   *size += length;
   const struct agx3_field_kv fields[] = {
      KV(2, 4), KV(9, 2), KV(10, 2), KV(11, 1), KV(16, 6),
      KV(23, 2), KV(24, 1), KV(27, 1), KV(38, 1), KV(51, 1),
      KV(52, 2), KV(56, 3), KV(57, 118), KV(59, 1), KV(74, 4),
   };
   return m5_mrt_append(bytes, size, AGX3_MN_pack_imgwir, fields,
                        AGX_MRT_ARRAY_SIZE(fields), 0, 14);
}

bool
agx3_m5_mrt_eot_generate_sample_mask(unsigned targets, const uint8_t *selectors,
                                 unsigned multisample_mask, bool layered, uint8_t *output,
                                 size_t capacity, size_t *written)
{
   if (!output || !written || !selectors || targets < 1 || targets > 8 ||
       (multisample_mask & ~((1u << targets) - 1)))
      return false;
   for (unsigned rt = 0; rt < targets; ++rt) {
      /* Reuse the RT0 store classes: F16/I16 f29=1, raw32 f29=2,
       * normalized8 f29=3, normalized16 f29=4 and integer8 f29=0. Descriptor indices and per-target byte offsets
       * are independent of the tile's scalar width. */
      if (selectors[rt] != 0x15 && selectors[rt] != 0x35 && selectors[rt] != 0x55 &&
          selectors[rt] != 0x75 && selectors[rt] != 0x95)
         return false;
   }
   uint8_t bytes[AGX3_M5_MRT_EOT_MAX_LENGTH];
   size_t size = sizeof(m5_mrt_prefix);
   memcpy(bytes, targets == 1 ? m5_rt0_prefix : m5_mrt_prefix, size);
   if (layered && !m5_mrt_layer(bytes, &size))
      return false;
   const unsigned first = targets < 2 ? targets : 2;
   for (unsigned rt = 0; rt < first; ++rt) {
      if (!m5_mrt_wire(bytes, &size, rt, targets, layered))
         return false;
   }
   for (unsigned rt = 0; rt < first; ++rt) {
      if (!m5_mrt_store(bytes, &size, rt, selectors[rt], (multisample_mask & (1u << rt)) != 0, layered))
         return false;
   }
   for (unsigned rt = first; rt < targets; ++rt) {
      if (!m5_mrt_wire(bytes, &size, rt, targets, layered))
         return false;
   }
   for (unsigned rt = first; rt < targets; ++rt) {
      if (!m5_mrt_store(bytes, &size, rt, selectors[rt], (multisample_mask & (1u << rt)) != 0, layered))
         return false;
   }
   const uint8_t stop[] = {0x0e, 0, 0, 0};
   if (sizeof(stop) > sizeof(bytes) - size)
      return false;
   memcpy(bytes + size, stop, sizeof(stop));
   size += sizeof(stop);
   if (capacity < size)
      return false;
   memcpy(output, bytes, size);
   *written = size;
   return true;
}

bool
agx3_m5_mrt_eot_generate_layered(unsigned targets, const uint8_t *selectors,
                                 bool multisample, bool layered, uint8_t *output,
                                 size_t capacity, size_t *written)
{
   if (targets < 1 || targets > 8)
      return false;
   return agx3_m5_mrt_eot_generate_sample_mask(targets, selectors,
      multisample ? (1u << targets) - 1 : 0, layered, output, capacity, written);
}

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

static bool
one_or_zero(uint16_t value)
{
   return (value & (value - 1)) == 0;
}

/*
 * The sample-mask operand. The immediate alternative is witnessed at every 1x
 * width. G17P and G17G multisample fragments read W16 SR39 and pass that
 * register to LSTCOL with f51=2, f44=the mask half, f55=last use. G17G's
 * native 2x/4x/8x plain fragments use the same register-mask operand fields.
 * On AGX3d the register form is admitted only when the caller opts in
 * (g17_register): the G18G/G17G 4x per-pixel store witness is
 * 270605082901ad02000071440000 = {f44:10, f51:2, f55:1, wait f18}
 * (native-msaa-reference-147), and nothing else is measured there.
 */
static bool
sample_ok(enum agx_isa_encoding enc, struct agx3_tile_sample s)
{
   if (s.reg)
      return (is_b(enc) && s.value < 192) ||
             (enc == AGX_ISA_ENCODING_AGX3C && s.value < 256) ||
             (s.g17_register && enc == AGX_ISA_ENCODING_AGX3D &&
              s.value < 192);

   return !s.kill && s.value <= 255;
}

static unsigned
sample_overrides(struct agx3_tile_sample s, struct agx3_field_kv *ov,
                 unsigned n)
{
   if (s.reg) {
      ov[n++] = (struct agx3_field_kv)KV(51, 2);
      ov[n++] = (struct agx3_field_kv)KV(44, s.value);
      if (s.kill)
         ov[n++] = (struct agx3_field_kv)KV(55, 1);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(51, 1);
      ov[n++] = (struct agx3_field_kv)KV(43, s.value);
   }

   return n;
}

static bool
common_ok(uint16_t reg, enum agx3_tile_width w, uint8_t mask,
          struct agx3_tile_sample sample, enum agx_isa_encoding enc,
          uint16_t wait_mask, uint16_t tile_off)
{
   const unsigned nwait = is_b(enc) ? 6 : 8;
   return (enc == AGX_ISA_ENCODING_AGX3B ||
           agx_encoding_is_g17_class(enc)) &&
          reg < (is_b(enc) ? 192 : 256) &&
          (w != AGX3_TILE_W32 || (reg & 1) == 0) &&
          (w == AGX3_TILE_W8 || w == AGX3_TILE_W16 || w == AGX3_TILE_W32) &&
          mask >= 1 && mask <= 15 && sample_ok(enc, sample) &&
          tile_off <= UINT8_MAX &&
          (wait_mask & ~((1u << nwait) - 1)) == 0 &&
          one_or_zero(wait_mask);
}

static unsigned
wait_overrides(enum agx_isa_encoding enc, uint16_t wait_mask,
               struct agx3_field_kv *ov, unsigned n)
{
   /* Wait bits are f18..f25 (slots 1..8). */
   for (unsigned i = 0; i < (is_b(enc) ? 6 : 8); ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(18 + i, 1);
   return n;
}

/*
 * PIXINFO: the sample groups of the fragment's pixel for the W16 SR39 coverage
 * word in half `src`, as one 16-bit word in half `dest`, on scoreboard slot
 * `slot`. f9=2 f10=1 select the 16-bit result and f51=2 the register source,
 * the form LLDCol's sample mask takes; f35/f37/f38/f52/f66 are those of the
 * tile accesses beside it, with f56=0x100 and f67=1. The wait field is the
 * ordinary mask (f18..f23 for slots 1..6).
 */
bool
agx3_pixinfo(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
             uint8_t slot, uint16_t wait_mask,
             uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   if (!is_b(enc) || dest >= 192 || src >= 192 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3fu))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2),  KV(10, 1), KV(35, 9),     KV(37, 7),  KV(38, 1),
      KV(51, 2), KV(52, 1), KV(56, 0x100), KV(66, 1), KV(67, 1),
   };
   struct agx3_field_kv ov[3 + 6];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(44, src);
   n = wait_overrides(enc, wait_mask, ov, n);
   if (!agx3_bind_compose(enc, AGX3_MN_pixinfo, 0, base,
                          AGX_MRT_ARRAY_SIZE(base), ov, n, out, len) ||
       *len != 14) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tile_store(enum agx_isa_encoding enc, uint16_t data,
                enum agx3_tile_width width, uint8_t chan_mask,
                uint16_t tile_off, struct agx3_tile_sample sample,
                uint16_t wait_mask, uint8_t out[AGX3_TILE_MAX_LENGTH],
                uint8_t *len)
{
   return agx3_tile_store_f13(enc, data, width, chan_mask, tile_off, sample,
                              wait_mask, !is_b(enc), out, len);
}

bool
agx3_tile_store_f13(enum agx_isa_encoding enc, uint16_t data,
                    enum agx3_tile_width width, uint8_t chan_mask,
                    uint16_t tile_off, struct agx3_tile_sample sample,
                    uint16_t wait_mask, bool f13,
                    uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   if (!common_ok(data, width, chan_mask, sample, enc, wait_mask, tile_off))
      return false;
   /* G18G (agx3d) keeps lstcol bit for bit: the 96 aligned
    * applegpu_g17g/applegpu_g18g lstcol words of the m5 reference corpus are
    * byte-identical, as is the oracle colour store 27 01 05 08 21 01 ad 02 00
    * 00 61 5c 00 00; the wait field stays a mask. */
   const struct agx3_field_kv base_b[] = {KV(9, 2),  KV(35, 9), KV(37, 7),
                                          KV(38, 1), KV(52, 1), KV(65, 1),
                                          KV(66, 1)};
   const struct agx3_field_kv base_c[] = {
      KV(9, 2),  KV(17, 111), KV(35, 9), KV(37, 7), KV(38, 1),
      KV(52, 1), KV(65, 1),   KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, data);
   if (f13)
      ov[n++] = (struct agx3_field_kv)KV(13, 1);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   n = sample_overrides(sample, ov, n);
   n = wait_overrides(enc, wait_mask, ov, n);
   if (!agx3_bind_compose(enc, AGX3_MN_lstcol, agx3_bind_ep(enc, 0),
                          is_b(enc) ? base_b : base_c, is_b(enc) ? 7 : 8, ov,
                          n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tile_store_f13_g18g(enum agx_isa_encoding enc, uint16_t data,
                         enum agx3_tile_width width, uint8_t chan_mask,
                         uint16_t tile_off, struct agx3_tile_sample sample,
                         uint16_t wait_mask, bool f13,
                         uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   /* The explicit C/D register-sample path receives the caller's data
    * lifetime bit. Preserve legacy whole-pixel and G17P behavior. */
   return agx3_tile_store_f13(enc, data, width, chan_mask, tile_off, sample,
                              wait_mask,
                              f13 || (!is_b(enc) && !sample.g17_register),
                              out, len);
}

bool
agx3_tile_store_explicit(enum agx_isa_encoding enc, uint16_t data,
                         enum agx3_tile_width width, uint8_t chan_mask,
                         uint16_t tile_off, uint16_t coords, bool last,
                         uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   /* 8-bit lanes one or four at a time, or one 32-bit word, each at an offset
    * aligned to the lane; the coordinate pair starts on a 32-bit register. */
   const bool shape =
      (width == AGX3_TILE_W8 && (chan_mask == 1 || chan_mask == 15)) ||
      (width == AGX3_TILE_W32 && chan_mask == 1 && (tile_off % 4) == 0 &&
       (data & 1) == 0);
   if (!is_b(enc) || !shape || data >= 192 || coords >= 192 ||
       (coords & 1) || tile_off > UINT8_MAX)
      return false;

   /* f30 names the coordinate halves; f41 releases the imageblock write
    * transaction after the last store. The rest is the fragment store's
    * whole-pixel form without its implicit-pixel fields. */
   const struct agx3_field_kv base[] = {KV(9, 2),  KV(13, 1), KV(37, 2),
                                        KV(38, 1), KV(43, 1), KV(51, 1),
                                        KV(52, 1), KV(65, 1), KV(66, 1)};
   const struct agx3_field_kv ov[] = {
      KV(2, data),     KV(10, width), KV(26, chan_mask),
      KV(57, tile_off), KV(30, coords), KV(41, last),
   };
   if (!agx3_bind_compose(enc, AGX3_MN_lstcol, 0, base, AGX_MRT_ARRAY_SIZE(base),
                          ov, AGX_MRT_ARRAY_SIZE(ov), out, len) ||
       *len != 12) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

int
agx3_block_store_lane_class(enum agx3_tile_width width, bool f16)
{
   if (width == AGX3_TILE_W8 && !f16)
      return 3;
   if (width == AGX3_TILE_W16 && f16)
      return 1;
   if (width == AGX3_TILE_W32 && !f16)
      return 2;
   return -1;
}

static bool
block_store_append(uint8_t *out, size_t *len, enum agx3_mnemonic op,
                   const struct agx3_field_kv *fields, unsigned count,
                   uint8_t expected)
{
   uint8_t instruction[16], length = 0;
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, op, 0, fields, count, NULL,
                          0, instruction, &length) ||
       length != expected ||
       length > AGX3_BLOCK_STORE_MAX_LENGTH - *len)
      return false;
   memcpy(out + *len, instruction, length);
   *len += length;
   return true;
}

bool
agx3_block_store_compute(enum agx_isa_encoding enc,
                         const struct agx3_block_store_desc *d,
                         uint8_t out[AGX3_BLOCK_STORE_MAX_LENGTH], size_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   *len = 0;
   if (!is_b(enc) || d == NULL || d->lane_class < 0 || d->lane_class > 3 ||
       d->x >= 192 || d->y >= 192 || (d->x & 1) || (d->y & 1) ||
       (d->base_uniform & 3) || d->base_uniform >= 512 ||
       (d->index_kind == AGX3_BLOCK_STORE_INDEX_UNIFORM && d->index >= 512) ||
       (d->index_kind == AGX3_BLOCK_STORE_INDEX_REGISTER && d->index >= 192))
      return false;

   /* Staging halves 4..5: the imageblock byte offset and the layer, both the
    * immediate zero of a single-layer tile at the imageblock's start. */
   const struct agx3_field_kv pack[] = {
      KV(5, 4),  KV(9, 5),  KV(10, 2), KV(12, 1), KV(23, 1),
      KV(24, 1), KV(37, 1), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(74, 8), KV(75, 16), KV(76, 7), KV(77, 24),
   };
   if (!block_store_append(out, len, AGX3_MN_pack_imgwir, pack,
                           AGX_MRT_ARRAY_SIZE(pack), 12))
      goto fail;

   /* Staging halves 0..1 and 2..3: the tile origin, one 32-bit word each. f27
    * is the source's last read, f25 keeps it. */
   for (unsigned xy = 0; xy < 2; ++xy) {
      const bool kill = xy ? d->y_kill : d->x_kill;
      const struct agx3_field_kv copy[] = {
         KV(5, 2 * xy), KV(9, 5),  KV(10, 2), KV(12, 1),
         KV(16, xy ? d->y : d->x), KV(23, 2), KV(24, 2),
         KV(kill ? 27 : 25, 1),
      };
      if (!block_store_append(out, len, AGX3_MN_copy, copy,
                              AGX_MRT_ARRAY_SIZE(copy), 4))
         goto fail;
   }

   /* The block store from staging half 0. f28 is the compute form; f39 the
    * uniform holding the descriptor heap base; f51/f53/f52 the descriptor
    * index as an immediate, a uniform or a register (f59/f60 its kind, f63
    * the register's last read). */
   struct agx3_field_kv img[20] = {
      KV(9, 5),  KV(12, 1), KV(13, 1), KV(28, 1),
      KV(29, d->lane_class), KV(30, 1), KV(32, 1), KV(45, 3),
      KV(46, 3), KV(39, d->base_uniform), KV(73, 1),
   };
   unsigned n = 11;
   switch (d->index_kind) {
   case AGX3_BLOCK_STORE_INDEX_IMMEDIATE:
      img[n++] = (struct agx3_field_kv)KV(59, 1);
      img[n++] = (struct agx3_field_kv)KV(60, 1);
      img[n++] = (struct agx3_field_kv)KV(51, d->index);
      img[n++] = (struct agx3_field_kv)KV(17, 1);
      break;
   case AGX3_BLOCK_STORE_INDEX_UNIFORM:
      img[n++] = (struct agx3_field_kv)KV(59, 3);
      img[n++] = (struct agx3_field_kv)KV(60, 2);
      img[n++] = (struct agx3_field_kv)KV(53, d->index);
      img[n++] = (struct agx3_field_kv)KV(17, 1);
      break;
   case AGX3_BLOCK_STORE_INDEX_REGISTER:
      img[n++] = (struct agx3_field_kv)KV(59, 2);
      img[n++] = (struct agx3_field_kv)KV(60, 2);
      img[n++] = (struct agx3_field_kv)KV(52, d->index);
      img[n++] = (struct agx3_field_kv)KV(22, 1);
      if (d->index_kill)
         img[n++] = (struct agx3_field_kv)KV(63, 1);
      break;
   default:
      goto fail;
   }
   if (!block_store_append(out, len, AGX3_MN_imgwblk, img, n, 12))
      goto fail;
   return true;

fail:
   memset(out, 0, AGX3_BLOCK_STORE_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_tile_load(enum agx_isa_encoding enc, uint16_t dest,
               enum agx3_tile_width width, uint8_t slot, uint8_t chan_mask,
               uint16_t tile_off, struct agx3_tile_sample sample,
               uint16_t wait_mask, bool first, bool last,
               uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   /* lldcol gains f79 on G18G (the one aligned applegpu_g18g tile load has
    * f79 = 2, its G17G twin none), which is not modelled: refuse agx3d. */
   if (enc == AGX_ISA_ENCODING_AGX3D)
      return false;

   /* Native454/513 ground raw32 reloads; native567 additionally grounds
    * packed BGRA8 and RGBA16Float register-sample reloads. Their full
    * tuples occupy four, eight or sixteen bytes according to the width.
    * The G18 long-form field is supplied by the transcode rule. */
   bool scalar_ms = sample.g17_register && sample.reg &&
                    enc == AGX_ISA_ENCODING_AGX3C &&
                    width == AGX3_TILE_W32 && chan_mask == 1 &&
                    !(tile_off % 4);
   if (sample.g17_register &&
       (enc != AGX_ISA_ENCODING_AGX3C || !sample.reg ||
        (width != AGX3_TILE_W8 && width != AGX3_TILE_W16 &&
         width != AGX3_TILE_W32) ||
        (!scalar_ms &&
         (chan_mask != 15 || (tile_off % (4u << width))))))
      return false;

   /* Framebuffer-fetch channel masks in f26: mask 1 (R8), mask 3 (RG8/RG16/
    * RG32 at all widths), and mask 7 (RGB565/B5G6R5 at W16 on G17P and G17G).
    * None carries an incoming wait. */
   if (!common_ok(dest, width, chan_mask, sample, enc, wait_mask, tile_off) ||
       (chan_mask != 15 && chan_mask != 1 && chan_mask != 3 &&
        !(chan_mask == 7 && width == AGX3_TILE_W16)) ||
       wait_mask != 0 || slot < 1 ||
       slot > (is_b(enc) ? 6 : 8))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2),  KV(35, 9), KV(37, 7),
                                          KV(38, 1), KV(52, 1), KV(65, 1),
                                          KV(66, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(17, 110), KV(35, 9),
                                          KV(37, 7), KV(38, 1),   KV(52, 1),
                                          KV(65, 1), KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   n = sample_overrides(sample, ov, n);
   ov[n++] = (struct agx3_field_kv)KV(78, first ? 1 : 0);
   if (!is_b(enc))
      ov[n++] = (struct agx3_field_kv)KV(74, last ? 2 : 0);
   if (!agx3_bind_compose(enc, AGX3_MN_lldcol, 0, is_b(enc) ? base_b : base_c,
                          is_b(enc) ? 7 : 8, ov, n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tile_store_lifetime(enum agx_isa_encoding enc, uint16_t data,
                         enum agx3_tile_width width, uint8_t chan_mask,
                         uint16_t tile_off, struct agx3_tile_sample sample,
                         uint16_t wait_mask, bool last_use,
                         uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   /* Native1130: 167 of the 169 mixed fragment stores carry f13 = 1 and the
    * two whose payload a later store reads carry f13 = 0; all other fields
    * are the whole-pixel form below. */
   if (enc != AGX_ISA_ENCODING_AGX3C || sample.reg || sample.kill ||
       sample.value != 1 ||
       !common_ok(data, width, chan_mask, sample, enc, wait_mask, tile_off) ||
       (tile_off % (1u << width)))
      return false;

   const struct agx3_field_kv base_c[] = {
      KV(9, 2),  KV(17, 111), KV(35, 9), KV(37, 7), KV(38, 1),
      KV(52, 1), KV(65, 1),   KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, data);
   if (last_use)
      ov[n++] = (struct agx3_field_kv)KV(13, 1);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   n = sample_overrides(sample, ov, n);
   n = wait_overrides(enc, wait_mask, ov, n);
   if (!agx3_bind_compose(enc, AGX3_MN_lstcol, agx3_bind_ep(enc, 0), base_c, 8, ov,
                          n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_tile_load_g18_mixed(enum agx_isa_encoding enc, uint16_t dest,
                         enum agx3_tile_width width, uint8_t slot,
                         uint8_t chan_mask, uint16_t tile_off,
                         struct agx3_tile_sample sample, bool first, bool last,
                         uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TILE_MAX_LENGTH);
   *len = 0;

   /* Native1130: the 26 reads of the 39 mixed fragments, at offsets 0..36,
    * are these four forms with the whole-pixel immediate 1, slot f12 counting
    * the reads from 1, f78 on the first and f74 = 2 on the last. */
   const bool form = (width == AGX3_TILE_W8 && chan_mask == 15) ||
                     (width == AGX3_TILE_W16 &&
                      (chan_mask == 7 || chan_mask == 15)) ||
                     (width == AGX3_TILE_W32 && chan_mask == 1);
   if (enc != AGX_ISA_ENCODING_AGX3C || !form || sample.reg || sample.kill ||
       sample.value != 1 || slot < 1 || slot > 8 ||
       !common_ok(dest, width, chan_mask, sample, enc, 0, tile_off) ||
       (tile_off % (1u << width)))
      return false;

   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(17, 110), KV(35, 9),
                                          KV(37, 7), KV(38, 1),   KV(52, 1),
                                          KV(65, 1), KV(66, 1)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(10, width);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(26, chan_mask);
   ov[n++] = (struct agx3_field_kv)KV(57, tile_off);
   n = sample_overrides(sample, ov, n);
   ov[n++] = (struct agx3_field_kv)KV(78, first ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(74, last ? 2 : 0);
   if (!agx3_bind_compose(enc, AGX3_MN_lldcol, 0, base_c, 8, ov, n, out, len)) {
      memset(out, 0, AGX3_TILE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}
