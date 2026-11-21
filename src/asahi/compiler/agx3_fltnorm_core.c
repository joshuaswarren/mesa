/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_fltnorm_core.h"
#include "agx3_pixel_core.h"
#include "agx3_tile_core.h"

#include <string.h>

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
enc_ok(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B || agx_encoding_is_g17_class(enc);
}

static unsigned
nslots(enum agx_isa_encoding enc)
{
   return agx_encoding_is_g17_class(enc) ? 8 : 6;
}

static uint16_t
wait_limit(enum agx_isa_encoding enc)
{
   return (uint16_t)((1u << nslots(enc)) - 1);
}

/* Per-source field triple: the class selector, the value field for that class,
 * the two liveness bits a register source picks between, and its modifiers. */
struct src_fields {
   uint8_t f_kind, f_imm, f_reg, f_uniform, f_width, f_live, f_kill, f_abs,
      f_neg;
};

static bool
append_src(struct agx3_field_kv *ov, unsigned *n,
           const struct agx3_fltnorm_src *s, const struct src_fields *f)
{
   uint8_t value_field;
   switch (s->kind) {
   case AGX3_FLTNORM_SRC_IMMEDIATE:
      /* AGX minifloats are 8 bits. */
      if (s->value > 0xff)
         return false;
      value_field = f->f_imm;
      break;
   case AGX3_FLTNORM_SRC_REGISTER:
      if ((s->value & 1) || s->value >= 256)
         return false;
      value_field = f->f_reg;
      break;
   case AGX3_FLTNORM_SRC_UNIFORM:
      if (s->value >= 512)
         return false;
      value_field = f->f_uniform;
      break;
   default:
      return false;
   }

   ov[(*n)++] = (struct agx3_field_kv)KV(f->f_kind, s->kind);
   ov[(*n)++] = (struct agx3_field_kv)KV(f->f_width, 2); /* 32-bit */
   ov[(*n)++] = (struct agx3_field_kv)KV(value_field, s->value);

   /* Only a register source carries liveness and modifiers; the composer
    * refuses the bits on the immediate and uniform alternatives. */
   if (s->kind == AGX3_FLTNORM_SRC_REGISTER) {
      ov[(*n)++] =
         (struct agx3_field_kv)KV(s->kill ? f->f_kill : f->f_live, 1);
      ov[(*n)++] = (struct agx3_field_kv)KV(f->f_abs, s->abs);
      ov[(*n)++] = (struct agx3_field_kv)KV(f->f_neg, s->neg);
   } else if (s->abs || s->neg) {
      return false;
   }
   return true;
}

/*
 * Build the shared normalized-format field, f58 on `fltnorm` and f40 on
 * `normflt`. Both pass through the same encoder LUT, so both take the same
 * value here. Returns false for a combination the hardware has no mode for.
 */
static bool
norm_mode(bool snorm, bool is_16bit, uint8_t srgb, bool two_lanes,
          uint8_t *out)
{
   switch (srgb) {
   case AGX3_FLTNORM_SRGB_NONE:
      *out = (uint8_t)((is_16bit ? 2 : 0) | (snorm ? 1 : 0));
      return true;

   case AGX3_FLTNORM_SRGB_BOTH:
      /* Witnessed with one lane (R8Unorm_sRGB) and with two (RG8Unorm_sRGB
       * and the leading pair of every RGBA8Unorm_sRGB shader). */
      if (snorm || is_16bit)
         return false;
      *out = 4;
      return true;

   case AGX3_FLTNORM_SRGB_LANE0:
      /* Only meaningful for a pair: it names lane 1 as the linear one. */
      if (snorm || is_16bit || !two_lanes)
         return false;
      *out = 6;
      return true;

   default:
      return false;
   }
}

bool
agx3_fltnorm_encode(enum agx_isa_encoding enc,
                    const struct agx3_fltnorm_desc *d,
                    uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL || d == NULL)
      return false;
   memset(out, 0, AGX3_FLTNORM_MAX_LENGTH);
   *len = 0;

   /* M5's native flat-colour program also marks its leading FLTNORM with
    * f68. Its second immediate pack leaves the bit clear. */
   if (!enc_ok(enc) || d->dest >= 256)
      return false;
   /* G18G (agx3d) keeps fltnorm's agx3c encoding: 147 of the 152 aligned
    * applegpu_g17g/applegpu_g18g fltnorm words of the m5 reference corpus are
    * byte-identical and the rest differ only in register and wait choices,
    * and the wait field stays a mask. No G18G program carries the f68 start
    * marker, so it is refused there. */
   if (enc == AGX_ISA_ENCODING_AGX3D && d->first_instruction)
      return false;
   if (enc == AGX_ISA_ENCODING_AGX3C && d->first_instruction &&
       (!d->two_lanes || d->snorm || d->is_16bit ||
        d->srgb != AGX3_FLTNORM_SRGB_NONE ||
        d->src[0].kind != AGX3_FLTNORM_SRC_IMMEDIATE ||
        d->src[1].kind != AGX3_FLTNORM_SRC_IMMEDIATE))
      return false;
   if ((d->wait_mask & ~wait_limit(enc)) != 0)
      return false;
   if (d->src[0].kind == AGX3_FLTNORM_SRC_NONE)
      return false;
   if (d->two_lanes != (d->src[1].kind != AGX3_FLTNORM_SRC_NONE))
      return false;
   /* G18G normalized16 output witnesses carry modes 2/3 in the same operand
    * map; G18G compiles to agx3c before its agx3d re-encode. */

   static const struct src_fields s0 = {
      .f_kind = 23, .f_imm = 15, .f_reg = 16, .f_uniform = 17,
      .f_width = 24, .f_live = 25, .f_kill = 27, .f_abs = 28, .f_neg = 29,
   };
   static const struct src_fields s1 = {
      .f_kind = 40, .f_imm = 32, .f_reg = 33, .f_uniform = 34,
      .f_width = 41, .f_live = 42, .f_kill = 44, .f_abs = 45, .f_neg = 46,
   };

   /* AGX3c carries the same operand map plus this form's sub-opcode pair. */
   struct agx3_field_kv base[4];
   unsigned nbase = 0;
   base[nbase++] = (struct agx3_field_kv)KV(9, 2);
   base[nbase++] = (struct agx3_field_kv)KV(10, 1);
   if (agx_encoding_is_g17_class(enc)) {
      base[nbase++] = (struct agx3_field_kv)KV(48, 3);
      base[nbase++] = (struct agx3_field_kv)KV(49, 72);
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_hint ? 1 : 0);
   uint8_t mode;
   if (!norm_mode(d->snorm, d->is_16bit, d->srgb, d->two_lanes, &mode))
      return false;
   ov[n++] = (struct agx3_field_kv)KV(58, mode);
   ov[n++] = (struct agx3_field_kv)KV(59, d->two_lanes ? 1 : 0);
   if (d->first_instruction)
      ov[n++] = (struct agx3_field_kv)KV(68, 1);

   if (!append_src(ov, &n, &d->src[0], &s0))
      return false;

   if (d->two_lanes) {
      if (!append_src(ov, &n, &d->src[1], &s1))
         return false;
   } else {
      /* The one-lane alternative: no second source, and the second operand's
       * width selector reads 1 rather than 2 (probe fn_r). */
      ov[n++] = (struct agx3_field_kv)KV(40, 0);
      ov[n++] = (struct agx3_field_kv)KV(41, 1);
   }

   for (unsigned i = 0; i < nslots(enc); ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(50 + i, 1);

   if (agx3_bind_compose(enc, AGX3_MN_fltnorm, agx3_bind_ep(enc, 0), base, nbase,
                         ov, n, out, len))
      return true;

   memset(out, 0, AGX3_FLTNORM_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_fltnorm_relocate_start(uint8_t *code, size_t size,
                            bool has_start_marker, size_t linked_offset)
{
   if (!has_start_marker)
      return true;

   /* Provenance comes from the packer. Check the promised ten-byte FLTNORM
    * and its f68 bit before changing it; f68 is physical bit 11 on G17P. */
   if (!code || size < 10)
      return false;
   const bool fltnorm = code[0] == 0x97 && (code[1] & 0xf) == 0xc;
   const bool rgb10a2 = size >= 14 && code[0] == 0xa7 &&
                       (code[1] & 0xf) == 0xe;
   if (!fltnorm && !rgb10a2)
      return false;
   if (linked_offset != 0)
      code[1] &= ~0x08;
   return true;
}

bool
agx3_fuse_colour_parts(const struct agx3_colour_link_info *main,
                       const struct agx3_colour_link_info *epilog,
                       uint8_t out[AGX3_COLOUR_LINK_SIZE])
{
   if (!main || !epilog || !out || !main->copy_only ||
       !main->copy_count || main->copy_count > 4 || !epilog->opaque_pack)
      return false;

   for (unsigned i = 0; i < main->copy_count; ++i) {
      if ((main->copies[i].reg | main->copies[i].uniform) & 1 ||
          main->copies[i].reg >= 256 || main->copies[i].uniform >= 512)
         return false;
      for (unsigned j = 0; j < i; ++j)
         if (main->copies[i].reg == main->copies[j].reg)
            return false;
   }

   /* Tile emitters clear their full14-byte destination even for12-byte G17P
    * output. Reserve the largest emitter capacity, not the returned size. */
   uint8_t code[AGX3_COLOUR_LINK_SIZE], len;
   uint8_t bytes[AGX3_TILE_MAX_LENGTH > AGX3_FLTNORM_MAX_LENGTH
                    ? AGX3_TILE_MAX_LENGTH : AGX3_FLTNORM_MAX_LENGTH];
   unsigned offset = 0, used = 0;
   for (unsigned i = 0; i < 2; ++i) {
      struct agx3_fltnorm_desc d = epilog->packs[i];
      if (!d.two_lanes || d.snorm || d.is_16bit || d.srgb || d.wait_mask ||
          d.dest != epilog->store_reg + i)
         return false;
      /* A pack must not overwrite a register whose exported value a later
       * pack could read. Keep this sufficient alias rule deliberately tight. */
      for (unsigned j = 0; j < main->copy_count; ++j)
         if (d.dest == main->copies[j].reg ||
             d.dest == main->copies[j].reg + 1)
            return false;
      for (unsigned s = 0; s < 2; ++s) {
         if (d.src[s].kind == AGX3_FLTNORM_SRC_REGISTER) {
            unsigned j;
            for (j = 0; j < main->copy_count; ++j)
               if (d.src[s].value == main->copies[j].reg)
                  break;
            if (j == main->copy_count)
               return false;
            used |= 1u << j;
            d.src[s].kind = AGX3_FLTNORM_SRC_UNIFORM;
            d.src[s].value = main->copies[j].uniform;
            d.src[s].kill = false;
         } else if (d.src[s].kind != AGX3_FLTNORM_SRC_IMMEDIATE) {
            return false;
         }
      }
      d.first_instruction = i == 0;
      if (!agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len) ||
          len != 10)
         return false;
      memcpy(code + offset, bytes, len);
      offset += len;
   }
   if (used != (1u << main->copy_count) - 1)
      return false;

   if (!agx3_pixel_colour_entry(AGX_ISA_ENCODING_AGX3B, bytes, &len) || len != 6)
      return false;
   memcpy(code + offset, bytes, len); offset += len;
   if (!agx3_pixel_wait(AGX_ISA_ENCODING_AGX3B, 12, 0, bytes, &len) || len != 6)
      return false;
   memcpy(code + offset, bytes, len); offset += len;
   if (!agx3_tile_store(AGX_ISA_ENCODING_AGX3B, epilog->store_reg, AGX3_TILE_W8,
                       15, 0, AGX3_TILE_SAMPLE_1X, 0, bytes, &len) || len != 12)
      return false;
   memcpy(code + offset, bytes, len); offset += len;
   if (!agx3_pixel_release(AGX_ISA_ENCODING_AGX3B, 12, true, 0, bytes, &len) ||
       len != 6)
      return false;
   memcpy(code + offset, bytes, len); offset += len;
   const uint8_t end[] = {14, 0, 0, 0};
   memcpy(code + offset, end, sizeof(end)); offset += sizeof(end);
   if (offset != sizeof(code))
      return false;
   memcpy(out, code, sizeof(code));
   return true;
}

bool
agx3_persp_rcp_encode(enum agx_isa_encoding enc,
                       const struct agx3_persp_rcp_desc *d,
                       uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || !d || !enc_ok(enc) || d->dest >= (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 192) ||
       d->rhw >= (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 192) || ((d->dest | d->rhw) & 1) ||
       d->coef_index > 0x7f || (d->wait_mask & ~wait_limit(enc)))
      return false;

   /* Fresh native smooth_scalar: af005400030000482001 (r0 -> r0).
    * Native exactgradient: af005402030002482001 (r0 -> r1, cache).
    * The final byte's bit0 selects the coefficient-bearing operand leaf;
    * f39 is the RHW coefficient index, not a discard hint. */
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(25, 2), KV(26, 2),
      KV(42, 6), KV(43, 2), KV(57, 1),
   };
   struct agx3_field_kv ov[15] = {
      KV(2, d->dest), KV(11, 1), KV(18, d->rhw),
      KV(27, d->rhw_cache), KV(39, d->coef_index),
   };
   unsigned n = 5;
   if (agx_encoding_is_g17_class(enc)) {
      ov[n++] = (struct agx3_field_kv)KV(47, 3);
      ov[n++] = (struct agx3_field_kv)KV(48, 64);
   }
   for (unsigned i = 0; i < nslots(enc); ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(49 + i, 1);
   return agx3_bind_compose(enc, AGX3_MN_f32rcp, agx3_bind_ep(enc, 0),
                            base, 7, ov, n, out, len);
}

bool
agx3_persp_mul_encode(enum agx_isa_encoding enc,
                      const struct agx3_persp_mul_desc *d,
                      uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL || d == NULL)
      return false;
   memset(out, 0, AGX3_FLTNORM_MAX_LENGTH);
   *len = 0;

   if (!enc_ok(enc))
      return false;
   if ((d->dest & 1) || d->dest >= 256 || (d->reciprocal & 1) ||
       d->reciprocal >= 256 || (d->raw & 1) || d->raw >= 256)
      return false;
   if ((d->wait_mask & ~wait_limit(enc)) != 0)
      return false;
   /* f70 is seven bits wide. */
   if (d->vary_index > 0x7f)
      return false;
   if (d->stage_dest ? (d->slot < 1 || d->slot > 6) : (d->slot != 0))
      return false;
   /* The G18G (agx3d) witnesses here use a general-register destination. */
   if (enc == AGX_ISA_ENCODING_AGX3D && d->stage_dest)
      return false;

   /* The form's root predicates -- the composer refuses ep4 outright without
    * them -- plus the two operand classes. */
   const struct agx3_field_kv base[] = {
      KV(10, 2), /* 32-bit destination */
      KV(25, 2), KV(26, 2), /* the interpolated operand: register, 32-bit */
      KV(42, 2), KV(43, 2), /* the reciprocal operand: register, 32-bit */
      KV(59, 6), KV(60, 2), KV(73, 6), KV(74, 2), KV(88, 1),
   };

   struct agx3_field_kv ov[20];
   unsigned n = 0;
   if (d->stage_dest) {
      ov[n++] = (struct agx3_field_kv)KV(9, 5);
      ov[n++] = (struct agx3_field_kv)KV(5, d->dest);
      ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(9, 2);
      ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
      ov[n++] = (struct agx3_field_kv)KV(11, d->dest_hint ? 1 : 0);
   }
   ov[n++] = (struct agx3_field_kv)KV(18, d->raw);
   if (d->raw_liveness != AGX3_OPERAND_UNMARKED) {
      ov[n++] = (struct agx3_field_kv)KV(
         d->raw_liveness == AGX3_OPERAND_KILL ? 29 : 27, 1);
   }
   ov[n++] = (struct agx3_field_kv)KV(35, d->reciprocal);
   if (d->reciprocal_liveness != AGX3_OPERAND_UNMARKED) {
      ov[n++] = (struct agx3_field_kv)KV(
         d->reciprocal_liveness == AGX3_OPERAND_KILL ? 46 : 44, 1);
   }
   ov[n++] = (struct agx3_field_kv)KV(70, d->vary_index);

   if (agx_encoding_is_g17_class(enc)) {
      ov[n++] = (struct agx3_field_kv)KV(78, 3);
      ov[n++] = (struct agx3_field_kv)KV(79, 62);
   }
   for (unsigned i = 0; i < nslots(enc); ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(80 + i, 1);

   /* The compact ep4 carries staging slots1..5.  Apple's register-indexed
    * fragment path puts coordinate staging on slot6 and consequently selects
    * the extended ep5 form.  Try the canonical compact form first so all
    * existing GPR and slot1 witnesses remain byte-identical. */
   for (unsigned ep = 4; ep <= 5; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32mul, agx3_bind_ep(enc, ep),
                            base, 10, ov, n, out, len))
         return true;
   }

   memset(out, 0, AGX3_FLTNORM_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_normflt_encode(enum agx_isa_encoding enc,
                    const struct agx3_normflt_desc *d,
                    uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL || d == NULL)
      return false;
   memset(out, 0, AGX3_FLTNORM_MAX_LENGTH);
   *len = 0;

   if (!enc_ok(enc) || (d->dest & 1) || d->dest >= 256 || d->src >= 256)
      return false;
   if ((d->wait_mask & ~wait_limit(enc)) != 0)
      return false;

   /* f25 = 2 register source and f29 = 1 in every witness. f26 is the source
    * width: one 16-bit register for packed 8-bit lanes and a single 16-bit
    * lane, and two for a pair of 16-bit lanes (applegpu_g17p RG16Unorm,
    * RGBA16Unorm and RGBA16Snorm blending reads `1c` where R16 and RGBA8
    * read `14`). Fresh G17G/G18G R16/RG16 UNORM/SNORM tile reads and
    * unpack_unorm2x16_to_float/unpack_snorm2x16_to_float programs establish
    * the same scalar/pair widths and normalized modes 2/3. AGX3c adds this
    * form's sub-opcode pair. */
   struct agx3_field_kv base[7];
   unsigned nbase = 0;
   base[nbase++] = (struct agx3_field_kv)KV(9, 2);
   base[nbase++] = (struct agx3_field_kv)KV(10, 2);
   base[nbase++] = (struct agx3_field_kv)KV(25, 2);
   base[nbase++] =
      (struct agx3_field_kv)KV(26, d->is_16bit && d->two_lanes ? 2 : 1);
   base[nbase++] = (struct agx3_field_kv)KV(29, 1);
   /* G18G (agx3d) keeps normflt's agx3c encoding: 8 of the 14 aligned
    * applegpu_g17g/applegpu_g18g normflt words of the m5 reference corpus are
    * byte-identical and the rest differ only in wait choice (f32..f39, still
    * a mask on agx3d). */
   if (agx_encoding_is_g17_class(enc)) {
      base[nbase++] = (struct agx3_field_kv)KV(30, 3);
      base[nbase++] = (struct agx3_field_kv)KV(31, 115);
   }

   struct agx3_field_kv ov[14];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_hint ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(18, d->src);
   ov[n++] = (struct agx3_field_kv)KV(41, d->two_lanes ? 1 : 0);
   uint8_t mode;
   if (!norm_mode(d->snorm, d->is_16bit, d->srgb, d->two_lanes, &mode))
      return false;
   ov[n++] = (struct agx3_field_kv)KV(40, mode);

   for (unsigned i = 0; i < nslots(enc); ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(32 + i, 1);

   if (agx3_bind_compose(enc, AGX3_MN_normflt, agx3_bind_ep(enc, 0), base, nbase,
                         ov, n, out, len))
      return true;

   memset(out, 0, AGX3_FLTNORM_MAX_LENGTH);
   *len = 0;
   return false;
}

/* Fresh Apple RGB10A2Unorm pipelines, with dead sources and a second float32
 * attachment that retains all sources, establish both lifetime encodings.
 * One native conversion replaces per-channel multiply/convert/shift/or work.
 */
bool
agx3_rgb10a2_encode_for_encoding(enum agx_isa_encoding enc,
                    const struct agx3_rgb10a2_desc *d,
                    uint8_t out[16], uint8_t *length)
{
   if (!enc_ok(enc) || !d || !out || !length || d->dest >= 256 || (d->dest & 1) ||
       (d->wait_mask & ~wait_limit(enc)) || d->layout > AGX3_PACKED_RG11B10)
      return false;
   /* Packed floats and part-start markers stay outside the G18G (agx3d)
    * contract until their shader-linking paths are qualified. */
   if (enc == AGX_ISA_ENCODING_AGX3D &&
       (d->layout != AGX3_PACKED_RGB10A2 || d->first_instruction))
      return false;
   const struct agx3_field_kv base[] = {KV(9, 2), KV(10, 2)};
   struct agx3_field_kv fields[48];
   unsigned n = 0;
   fields[n++] = (struct agx3_field_kv)KV(2, d->dest);
   const unsigned lanes = d->layout == AGX3_PACKED_RGB10A2 ? 4 : 3;
   if (lanes == 3) {
      /* The absent fourth lane reads width 1, as fltnorm's absent second
       * lane does. */
      if (d->kind[3] != AGX3_FLTNORM_SRC_NONE)
         return false;
      fields[n++] = (struct agx3_field_kv)KV(75, 1);
      fields[n++] = (struct agx3_field_kv)KV(92, d->layout);
      fields[n++] = (struct agx3_field_kv)KV(93, 1);
   }
   for (unsigned i = 0; i < lanes; ++i) {
      const bool reg = d->kind[i] == AGX3_FLTNORM_SRC_REGISTER;
      const bool imm = d->kind[i] == AGX3_FLTNORM_SRC_IMMEDIATE;
      if ((!reg && !imm) || d->source[i] >= 256 ||
          (reg && (d->source[i] & 1)) || (!reg && (d->abs[i] || d->neg[i])))
         return false;
      fields[n++] = (struct agx3_field_kv)KV(23 + 17 * i, d->kind[i]);
      fields[n++] = (struct agx3_field_kv)KV(24 + 17 * i, 2);
      fields[n++] = (struct agx3_field_kv)KV((reg ? 16 : 15) + 17 * i, d->source[i]);
      if (reg) {
         fields[n++] = (struct agx3_field_kv)KV(25 + 17 * i, d->cache[i]);
         fields[n++] = (struct agx3_field_kv)KV(27 + 17 * i, d->discard[i]);
         fields[n++] = (struct agx3_field_kv)KV(28 + 17 * i, d->abs[i]);
         fields[n++] = (struct agx3_field_kv)KV(29 + 17 * i, d->neg[i]);
      }
   }
   /* Fresh texture-to-RGB10 native main waits for smp slot1 at field84. */
   /* Fresh G17G/G18G render pipelines establish the same register/immediate,
    * modifier and lifetime fields as G17P. Their pack sub-opcode is 3/116. */
   if (agx_encoding_is_g17_class(enc)) {
      fields[n++] = (struct agx3_field_kv)KV(82, 3);
      fields[n++] = (struct agx3_field_kv)KV(83, 116);
   }
   for (unsigned i = 0; i < nslots(enc); ++i)
      if (d->wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv)KV(84 + i, 1);
   fields[n++] = (struct agx3_field_kv)KV(102, d->first_instruction);
   fields[n++] = (struct agx3_field_kv)KV(94, 0);
   fields[n++] = (struct agx3_field_kv)KV(95, 0);
   fields[n++] = (struct agx3_field_kv)KV(96, 0);
   return agx3_bind_compose(enc, AGX3_MN_pack, agx3_bind_ep(enc, 0),
                           base, 2, fields, n, out, length);
}

bool
agx3_rgb10a2_encode(const struct agx3_rgb10a2_desc *d,
                    uint8_t out[14], uint8_t *length)
{
   uint8_t bytes[16];
   if (!agx3_rgb10a2_encode_for_encoding(AGX_ISA_ENCODING_AGX3B,
                                        d, bytes, length))
      return false;
   memcpy(out, bytes, *length);
   return true;
}

bool
agx3_rgb10a2_encode_g18g(enum agx_isa_encoding enc,
                         const struct agx3_rgb10a2_desc *d,
                         uint8_t out[16], uint8_t *length)
{
   /* G18G compiles as agx3c: packed floats and part-start markers remain
    * G17P-only there until their shader-linking paths are qualified. */
   if (d && agx_encoding_is_g17_class(enc) &&
       (d->layout != AGX3_PACKED_RGB10A2 || d->first_instruction))
      return false;
   return agx3_rgb10a2_encode_for_encoding(enc, d, out, length);
}

/* Fresh single- and two-attachment fetch programs identify the source register
 * and slot1/slot6 wait fields independently. The result is four contiguous f32s. */
bool
agx3_rgb10a2_unpack_encode(
   enum agx_isa_encoding enc, const struct agx3_rgb10a2_unpack_desc *d,
   uint8_t out[16], uint8_t *length)
{
   if (!enc_ok(enc) || !d || !out || !length || d->dest > 248 ||
       d->source > 254 || ((d->dest | d->source) & 1) ||
       (d->wait_mask & ~wait_limit(enc)) ||
       d->layout > AGX3_PACKED_RG11B10)
      return false;
   struct agx3_field_kv base[7] = {
      KV(9, 2), KV(10, 2), KV(25, 2), KV(26, 2),
   };
   unsigned nbase = 4;
   /* G17G UInt1010102Normalized and FloatRG11B10 vertex witnesses. */
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nbase++] = (struct agx3_field_kv)KV(11, d->dest_hint);
      base[nbase++] = (struct agx3_field_kv)KV(30, 3);
      base[nbase++] = (struct agx3_field_kv)KV(31, 191);
   }
   struct agx3_field_kv fields[15] = {
      KV(2, d->dest), KV(18, d->source), KV(29, d->discard),
      KV(40, d->layout), KV(41, 0), KV(42, 0), KV(43, 0),
   };
   unsigned n = 7;
   for (unsigned i = 0; i < nslots(enc); ++i)
      if (d->wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv)KV(32 + i, 1);
   return agx3_bind_compose(enc, AGX3_MN_unpack, 0,
                           base, nbase, fields, n, out, length);
}

bool
agx3_rgb10a2_unpack_encode_g18g(
   enum agx_isa_encoding enc, const struct agx3_rgb10a2_unpack_desc *d,
   uint8_t out[16], uint8_t *length)
{
   if (enc == AGX_ISA_ENCODING_AGX3B)
      return agx3_rgb10a2_unpack_encode(enc, d, out, length);

   if (!out || !length)
      return false;
   memset(out, 0, 10);
   *length = 0;
   if (!d || !agx_encoding_is_g17_class(enc) || d->dest > 248 ||
       d->source > 254 || ((d->dest | d->source) & 1) ||
       (d->wait_mask & ~0xff) || d->layout != AGX3_PACKED_RGB10A2)
      return false;

   /* Native G17G/G18G RGB10A2 framebuffer reads and
    * unpack_unorm10a2_to_float use a ten-byte instruction, with f30/f31
    * selecting the unpack sub-opcode. A retained raw source uses f27;
    * a discarded one uses f29. Both forms are witnessed with compute
    * consumers, and framebuffer reads also establish the destination hint.
    * Packed-float layouts remain outside this G17/G18 contract. */
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(25, 2), KV(26, 2), KV(30, 3), KV(31, 191),
   };
   struct agx3_field_kv fields[13] = {
      KV(2, d->dest), KV(11, d->dest_hint), KV(18, d->source),
      KV(27, !d->discard), KV(29, d->discard),
   };
   unsigned n = 5;
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv)KV(32 + i, 1);
   return agx3_bind_compose(enc, AGX3_MN_unpack, agx3_bind_ep(enc, 0),
                           base, 6, fields, n, out, length);
}
