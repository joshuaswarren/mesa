/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "agx3_tex_core.h"

#include "agx3_alu_core.h"
#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
one_or_zero(uint16_t value)
{
   return (value & (value - 1)) == 0;
}

/* Apple's applegpu_g17p compiler also sets bits 117 and 123 in every SMP it
 * emits (metal-nt witnesses). No smp encoding point in AGX3b.xml models
 * either bit and this packer does not emit them. */

/*
 * KNOWN GAP, not specific to sparse: the 20-byte ep4 form carries three
 * things this encoder leaves zero, and every ep4 texture operation it emits
 * -- dense and sparse alike -- differs from Apple's own bytes in exactly
 * those places.  Matched `xcrun metal-nt -arch applegpu_g17p` compute kernels
 * show:
 *
 *   f19 upper byte  a copy of the destination half-register; dest 0 encodes
 *                   f19 = 0, dest 2 encodes 512, dest 10 encodes 2560, and
 *                   destination half 2 over coordinate bank 4 encodes 516
 *   f91             set for a 32-bit result, clear for a 16-bit one
 *                   (05800cb8b4...00800098 W32 against
 *                    05000cb8b4...00000098 W16, otherwise identical)
 *   bits 154..159   Apple sets 155/156/159; the generated AGX3b tables
 *                   deposit nothing into those bits, so no field value can
 *                   produce them at all
 *
 * Adding the first two would change every existing ep4 golden in the tree on
 * an inference no host test can settle, so they are recorded here rather than
 * encoded.  Whether the hardware reads any of them is unknown.
 */

static bool
fetch_ok(enum agx_isa_encoding enc, const struct agx3_tex_fetch_desc *d)
{
   if (d == NULL || (enc != AGX_ISA_ENCODING_AGX3B &&
                     enc != AGX_ISA_ENCODING_AGX3C))
      return false;

   if ((d->width != AGX3_TEX_W16 && d->width != AGX3_TEX_W32) ||
       d->dest >= (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 128) ||
       d->dest + __builtin_popcount(d->mask) * (d->width == AGX3_TEX_W32 ? 2 : 1) > 256 ||
       (d->width == AGX3_TEX_W32 && (d->dest & 1)) ||
       (d->coord & 3) ||
       d->coord > ((enc == AGX_ISA_ENCODING_AGX3C && d->multisample &&
                    !d->array) ? 120 : 60) ||
       (!d->mask || d->mask > 15) ||
       d->texture > 15)
      return false;

   /* A18 Pro witnesses both 2D-MS and 2D-MS-array reads.  The non-array
    * form uses staging halves 0/2 and carries a bounded immediate sample.
    * The array form uses halves 16/18 and PACK_SMPIR publishes the packed
    * sample/layer word separately. G17G 2D-MS fragments instead publish the
    * sample beside each of four independent eight-half coordinate banks. */
   if ((d->sample && !d->multisample) ||
       (d->staged_sample && (!d->multisample || d->sample)) ||
       (d->g18g_ms && (enc != AGX_ISA_ENCODING_AGX3C ||
                       !d->multisample || !d->staged_sample || d->coord != 4)) ||
       (d->multisample && !d->g18g_ms &&
        (d->sample > 3 ||
         (d->array && d->sample) ||
         (enc == AGX_ISA_ENCODING_AGX3C
            ? (!d->staged_sample || (d->coord & 7) ||
               d->coord > (d->array ? 24 : 120))
            : d->coord != (d->array && !d->staged_sample ? 16 : 0)))))
      return false;

   /* Own-MSL texture3d::read witnesses use f76=3 with a third integer
    * coordinate in the contiguous staging bank. The immediate and indirect
    * texture forms are both captured on G17P. A runtime integer level uses
    * the same f139=4 PACK_SMPIR form as 2D; the dimension remains selected by
    * f76=3. M5 retains these dimension and integer-level selectors. */
   if (d->three_d && (d->array || d->multisample))
      return false;

   const unsigned tk = d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   const unsigned sk = d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   /* Native G18 2D-MS and MS-array reads name sampler zero for bindful,
    * uniform-indexed and lane-indexed descriptors. The descriptor kind
    * changes only the texture operand profile. */
   if (tk > AGX3_TEX_OPERAND_UNIFORM || sk > AGX3_TEX_OPERAND_UNIFORM ||
       (tk != AGX3_TEX_OPERAND_IMMEDIATE &&
        ((d->texture_value & 1) || d->texture_value > 252)) ||
       (sk == AGX3_TEX_OPERAND_IMMEDIATE ? d->sampler_value > 15
         : ((d->sampler_value & 1) || d->sampler_value > 252)) ||
       (d->texture_kill && tk != AGX3_TEX_OPERAND_REGISTER) ||
       (d->multisample &&
        (sk != AGX3_TEX_OPERAND_IMMEDIATE ||
         (d->g18g_ms ? d->sampler_value != 0
          : enc == AGX_ISA_ENCODING_AGX3C ? d->sampler_value > 1
            : d->sampler_value != (d->staged_sample ? 1 : 0)))))
      return false;
   if (d->three_d &&
       (sk != AGX3_TEX_OPERAND_IMMEDIATE ||
        d->sampler_value > (enc == AGX_ISA_ENCODING_AGX3C ? 1 : 0)))
      return false;

   /* The level. f29 is 12 bits of 1/64 fixed point; the per-lane form has no
    * immediate at all. */
   if (d->lod_x64 > 2047 || (d->register_lod && d->lod_x64) ||
       (d->multisample && (d->lod_x64 || d->register_lod)))
      return false;

   /* Sparse reads use AGX3b ep4 or the 22-byte AGX3c ep3 form.
    * No sparse multisample form is qualified. */
   if (d->sparse && d->multisample)
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~waits) == 0 && one_or_zero(d->wait_mask);
}

bool
agx3_tex_encode_fetch(enum agx_isa_encoding enc,
                      const struct agx3_tex_fetch_desc *d,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!fetch_ok(enc, d) || (d->heap_uniform_base & 3) ||
       d->heap_uniform_base > 248)
      return false;

   struct agx3_field_kv base[20];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 5);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   base[nb++] = (struct agx3_field_kv)KV(52, 1);
   base[nb++] = (struct agx3_field_kv)KV(89, 3);
   base[nb++] = (struct agx3_field_kv)KV(90, 3);
   const enum agx3_tex_operand_kind tk =
      d->texture_kind ? d->texture_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const enum agx3_tex_operand_kind sk =
      d->sampler_kind ? d->sampler_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate_operands = tk == AGX3_TEX_OPERAND_IMMEDIATE &&
                                   sk == AGX3_TEX_OPERAND_IMMEDIATE;

   base[nb++] = (struct agx3_field_kv)KV(103, tk);
   base[nb++] = (struct agx3_field_kv)KV(104, tk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   /* The operand profile.  A sparse residency request does NOT change it: see
    * the note on agx3_tex_sample_desc::sparse below, and
    * agx3_tex_fetch_desc::sparse in the header. */
   if (d->heap_uniform_base)
      base[nb++] = (struct agx3_field_kv)KV(83, d->heap_uniform_base);
   base[nb++] = (struct agx3_field_kv)KV(112, d->heap_uniform_base + 4);
   base[nb++] = (struct agx3_field_kv)KV(118, 3);
   base[nb++] = (struct agx3_field_kv)KV(119, 3);
   base[nb++] = (struct agx3_field_kv)KV(132, sk);
   base[nb++] = (struct agx3_field_kv)KV(133, sk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(138, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(24, 2);
      base[nb++] = (struct agx3_field_kv)KV(60, 2);
      base[nb++] = (struct agx3_field_kv)KV(61, 164);
   }

   struct agx3_field_kv ov[28];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(19, d->coord);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(27, d->coord_kill ? 1 : 0);
   /* Keep the native literal-layer2 witness; staged arrays use f77=2 instead.
    * Non-MS fetches use f29 for immediate LOD and f37 to mark that operand. */
   if (d->multisample && d->array && !d->staged_sample) {
      ov[n++] = (struct agx3_field_kv)KV(29, 2);
      ov[n++] = (struct agx3_field_kv)KV(37, 1);
   } else if (!d->register_lod) {
      ov[n++] = (struct agx3_field_kv)KV(37, d->multisample ? 0 : 1);
      if (d->lod_x64)
         ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
   }
   ov[n++] = (struct agx3_field_kv)KV(75, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(
      76, d->three_d ? 3 : (d->multisample ? 2 : 1));
   if (d->array)
      ov[n++] = (struct agx3_field_kv)KV(
         77, d->multisample && !d->staged_sample ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(78, 1);  /* texel fetch */
   /* Each operand lands in the field its kind selects: 95/96/97 for the
    * texture, 124/125/126 for the sampler. */
   ov[n++] = (struct agx3_field_kv)KV(
      tk == AGX3_TEX_OPERAND_IMMEDIATE   ? 95
      : tk == AGX3_TEX_OPERAND_REGISTER  ? 96
                                         : 97,
      tk == AGX3_TEX_OPERAND_IMMEDIATE ? d->texture : d->texture_value);
   ov[n++] = (struct agx3_field_kv)KV(
      sk == AGX3_TEX_OPERAND_IMMEDIATE   ? 124
      : sk == AGX3_TEX_OPERAND_REGISTER  ? 125
                                         : 126,
      d->sampler_value);
   ov[n++] = (struct agx3_field_kv)KV(107, d->texture_kill);
   ov[n++] = (struct agx3_field_kv)KV(
      139, d->multisample ? 0 : d->register_lod ? 4 : 2);
   if (enc == AGX_ISA_ENCODING_AGX3B)
      ov[n++] = (struct agx3_field_kv)KV(147, d->final ? 2 : 0);
   else {
      ov[n++] = (struct agx3_field_kv)KV(150, d->final ? 2 : 0);
      ov[n++] = (struct agx3_field_kv)KV(155, 1);
   }
   if (d->sparse) {
      if (enc == AGX_ISA_ENCODING_AGX3B)
         ov[n++] = (struct agx3_field_kv)KV(120, 1);
      ov[n++] = (struct agx3_field_kv)KV(142, 1);
   }
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(62 + i, 1);

   /*
    * Immediate operands are ep3; register and uniform operands are ep4. Only
    * AGX3b uses these encoding points for the non-immediate kinds too.
    * AGX3c uses its own bounded ep0/1/2 forms below. f120 deposits
    * instruction bit 142, past the 112-bit ep3 form, so a sparse read is
    * always ep4.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && !immediate_operands &&
       (d->width != AGX3_TEX_W32 ||
        sk != AGX3_TEX_OPERAND_IMMEDIATE || d->sampler_value > 1 ||
        d->texture_kill))
      return false;
   const int ep = enc == AGX_ISA_ENCODING_AGX3B
                     ? (immediate_operands && !d->sparse && !d->heap_uniform_base ? 3 : 4)
                     : d->sparse ? 3 : 0;
   bool composed = agx3_bind_compose(enc, AGX3_MN_smp, ep, base, nb, ov, n, out, len);
   if (!composed && enc == AGX_ISA_ENCODING_AGX3C)
      composed = agx3_bind_compose(enc, AGX3_MN_smp, 1, base, nb, ov, n, out, len);
   if (!composed && enc == AGX_ISA_ENCODING_AGX3C)
      composed = agx3_bind_compose(enc, AGX3_MN_smp, 2, base, nb, ov, n, out, len);
   if (!composed && enc == AGX_ISA_ENCODING_AGX3C && d->heap_uniform_base)
      composed = agx3_bind_compose(enc, AGX3_MN_smp, 3, base, nb, ov, n, out, len);
   if (!composed) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }

   if (d->multisample && !d->array) {
      /* Own-MSL reads of samples 0..3 isolate this two-bit immediate: byte 5
       * is 00/02/04/06 while the remaining sample-0 instruction is fixed. */
      if (out[5] & 0x06) {
         memset(out, 0, AGX3_TEX_MAX_LENGTH);
         *len = 0;
         return false;
      }
      out[5] |= d->sample << 1;
   }
   return true;
}

static bool
mask_witnessed(uint8_t mask)
{
   /* All 15 nonempty RGBA subsets are witnessed by native fragment samples.
    * Results remain densely packed in selected-component order. */
   return mask != 0 && (mask & ~15u) == 0;
}

static bool
sample_ok(enum agx_isa_encoding enc, const struct agx3_tex_sample_desc *d)
{
   if (d == NULL || (enc != AGX_ISA_ENCODING_AGX3B &&
                     enc != AGX_ISA_ENCODING_AGX3C))
      return false;

   if (d->sample_abi > AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT ||
       (d->sample_abi == AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT &&
        enc != AGX_ISA_ENCODING_AGX3B))
      return false;

   /* G18G only (d->g18g): native G18 sample shapes the G18G layer encodes
    * beyond the admission below. Each is zero for every other GPU. */
   /* Native685r4 covers 2D gathers through register and uniform samplers,
    * including raw depth, comparisons and constant/runtime offsets. The
    * sampler descriptor selector is orthogonal to the gather publication. */
   const bool g18g_indirect_gather = d->g18g &&
      enc == AGX_ISA_ENCODING_AGX3C &&
      d->gather != AGX3_TEX_GATHER_NONE && !d->array && !d->cube &&
      !d->three_d && d->width == AGX3_TEX_W32 && d->mask == 15 &&
      !d->offset_staged && !d->query_lod && !d->min_lod && !d->sparse &&
      d->lod == AGX3_TEX_LOD_AUTO &&
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER &&
      (d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE &&
      (d->sampler_kind == AGX3_TEX_OPERAND_REGISTER ||
       d->sampler_kind == AGX3_TEX_OPERAND_UNIFORM);
   /* Native371/403/408/413 cover ordinary, runtime-offset, constant-offset
    * and cube gathers, including their distinct array/face publications. */
   const bool g18g_gather = d->g18g && enc == AGX_ISA_ENCODING_AGX3C &&
      d->gather != AGX3_TEX_GATHER_NONE &&
      d->width == AGX3_TEX_W32 && d->mask == 15 &&
      (!d->cube || (!d->offset && !d->offset_register)) && !d->three_d &&
      !d->offset_staged &&
      !d->query_lod && !d->min_lod &&
      d->lod == AGX3_TEX_LOD_AUTO &&
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER &&
      (d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE &&
      ((d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE ||
       g18g_indirect_gather);
   /* Ordinary 2D/array colour and comparison samples with immediate
    * descriptors. Vertex-stage witnesses cover explicit register levels and
    * explicit gradients. */
   const bool g18g_plain_2d = d->g18g && enc == AGX_ISA_ENCODING_AGX3C &&
      !d->cube && !d->three_d && d->gather == AGX3_TEX_GATHER_NONE &&
      !d->query_lod && !d->min_lod &&
      (!d->vertex_stage || d->lod == AGX3_TEX_LOD_REGISTER ||
       d->lod == AGX3_TEX_LOD_GRADIENT) &&
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER &&
      (d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE &&
      (d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE;
   /* Native653 independently covers all nine descriptor-class pairs for
    * ordinary 2D/array/3D samples, with masks 1/15, register level in every
    * stage and implicit/register bias in fragments. Native809 adds the same
    * descriptor pairs for full-vector cube/cube-array colour and scalar
    * 2D/array/cube/cube-array comparisons, with register levels in every
    * stage and implicit fragment LOD. Native816 adds fragment register
    * bias for those same descriptor pairs and masks. The stream packer still
    * owns staging and retirement. */
   const bool g18g_binding = d->g18g && enc == AGX_ISA_ENCODING_AGX3C &&
      !(d->three_d && (d->array || d->cube || d->shadow)) &&
      d->width == AGX3_TEX_W32 &&
      (d->shadow ? d->mask == 1 : d->cube ? d->mask == 15
                                          : (d->mask == 1 || d->mask == 15)) &&
      !d->offset && !d->offset_register && !d->offset_staged &&
      !d->query_lod && !d->min_lod && !d->sparse &&
      d->gather == AGX3_TEX_GATHER_NONE &&
      (d->lod == AGX3_TEX_LOD_REGISTER ||
       (!d->vertex_stage && !d->compute_stage &&
        (d->lod == AGX3_TEX_LOD_AUTO ||
         (d->lod == AGX3_TEX_LOD_BIAS_REGISTER)))) &&
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER;
   /* Native917 pairs half4/float4 non-array cube results. Each pair
    * differs only in f10. The stream packer retains its bank0 staging;
    * bank4 reproduces the native instruction without rewriting evidence. */
   const bool g18g_cube_vector = d->g18g && enc == AGX_ISA_ENCODING_AGX3C &&
      d->cube && !d->three_d && !d->array && !d->shadow &&
      (d->width == AGX3_TEX_W16 || d->width == AGX3_TEX_W32) &&
      d->mask == 15 && (d->coord == 0 || d->coord == 4) &&
      !d->offset && !d->offset_register && !d->offset_staged &&
      !d->query_lod && !d->min_lod && !d->sparse &&
      d->gather == AGX3_TEX_GATHER_NONE &&
      (d->lod == AGX3_TEX_LOD_REGISTER ||
       (!d->vertex_stage && !d->compute_stage &&
        d->lod == AGX3_TEX_LOD_BIAS_REGISTER)) &&
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER &&
      (d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE &&
      (d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE;
   /* Native949: scalar/pair float32 2D/3D fragment queries. G17G uses
    * f155=2, not the f146 selector of G17P, and the compact ep1 form. */
   const bool g18g_query_lod = d->g18g && enc == AGX_ISA_ENCODING_AGX3C &&
      d->query_lod && !d->array && !d->cube && !d->vertex_stage &&
      !d->compute_stage && d->coord == 0 && !d->min_lod && !d->sparse &&
      !d->offset_register && !d->offset_staged &&
      !d->sample_group_nonterminal && !d->sample_group_end &&
      (d->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE &&
      (d->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) == AGX3_TEX_OPERAND_IMMEDIATE;

   /* Sparse residency is witnessed on AGX3b for the implicit, immediate
    * level, immediate bias, 2D-array, 3D, cube and gather shapes.  A depth
    * comparison and a LOD query have no witness. AGX3c uses its 22-byte
    * ep3 form to carry f142. */
   if (d->sparse && (d->shadow || d->query_lod))
      return false;

   /* The native clamp corpus covers fragment 2D, 2D-array and 3D samples,
    * including comparison/offset and indexed descriptor operands, and
    * biased cube, cube-array and comparison cube samples. A staged 3D offset
    * is witnessed only beside a bias. */

   if (d->min_lod &&
       ((d->cube && d->lod != AGX3_TEX_LOD_BIAS_REGISTER) ||
        d->vertex_stage || d->query_lod ||
        (d->three_d && d->offset_staged &&
         d->lod != AGX3_TEX_LOD_BIAS_REGISTER) ||
        d->gather != AGX3_TEX_GATHER_NONE ||
        d->sample_abi != AGX3_TEX_SAMPLE_ABI_DRIVER ||
        (d->lod != AGX3_TEX_LOD_BIAS_REGISTER &&
         d->lod != AGX3_TEX_LOD_GRADIENT)))
      return false;

   /* Fresh native calculate_{clamped,unclamped}_lod witnesses cover the
    * ordinary derivative path only, returning packed X/Y float32 results. */
   if (d->query_lod &&
       (d->width != AGX3_TEX_W32 ||
        (d->lod != AGX3_TEX_LOD_AUTO &&
         !(g18g_query_lod && d->lod == AGX3_TEX_LOD_BIAS_REGISTER)) ||
        (d->mask & ~3u) || d->array_layer ||
        d->gather != AGX3_TEX_GATHER_NONE || d->shadow || d->offset ||
        d->sample_abi != AGX3_TEX_SAMPLE_ABI_DRIVER))
      return false;

   if ((d->sample_group_nonterminal || d->sample_group_end) &&
       (d->lod == AGX3_TEX_LOD_GRADIENT ||
        (d->sample_group_nonterminal && d->sample_group_end)))
      return false;

   if ((d->width != AGX3_TEX_W16 && d->width != AGX3_TEX_W32) ||
       d->dest >= (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 128) ||
       (d->width == AGX3_TEX_W32 && (d->dest & 1)) ||
       (d->coord & 3) ||
       d->coord > 252 ||
       !mask_witnessed(d->mask) ||
       d->texture > 15 || d->sampler > 15)
      return false;

   if (d->lod == AGX3_TEX_LOD_LEVEL) {
      /* 12-bit immediate level field; witnessed up to 464 (level 7.25). */
      if (d->lod_x64 > 0xfff)
         return false;
   } else if (d->lod == AGX3_TEX_LOD_BIAS) {
      /* The same 12-bit x64 field, but read as *signed*: Apple encodes a
       * negative bias in two's complement (bias(-1.0) is 4032, bias(-32.0)
       * is 2048) and saturates at 2047 / 2048. The caller passes the field
       * already masked to 12 bits. Apple emits the plain implicit form
       * rather than a zero bias, so refuse the degenerate encoding. */
      if (d->lod_x64 == 0 ||
          d->lod_x64 > 0xfff)
         return false;
   } else if (d->lod == AGX3_TEX_LOD_REGISTER ||
              d->lod == AGX3_TEX_LOD_BIAS_REGISTER ||
              d->lod == AGX3_TEX_LOD_GRADIENT) {
      /* Per-lane levels/biases use bank 6. Gradients use the shared
       * interleaved scalar banks 8..18. */
      if (d->lod_x64 != 0)
         return false;
   } else if (d->lod != AGX3_TEX_LOD_AUTO || d->lod_x64 != 0) {
      return false;
   }


   /* G17P names a dynamic array layer directly with f31/f37 in every
    * witnessed LOD mode. Do not accept a value the 32-bit register operand
    * cannot name, or silently ignore a layer on a non-array sample. */
   if ((d->array &&
        ((d->array_layer & 1) ||
         d->array_layer > 254)) ||
       (!d->array && d->array_layer != 0))
      return false;

   /* The 3D dimension is f76 = 3 and is orthogonal to the LOD shape exactly
    * as the 2D one is. Five texture3d fragments, differing only in the LOD
    * argument, differ only in f139 and its operand:
    *
    *   t.sample(s, uvw)                 05800cb8b0008001010010001200
    *   t.sample(s, uvw, bias(0.5f))     05800cb8b0003b00010010021200  f139 1, f29 32
    *   t.sample(s, uvw, bias(in.dx.x))  05800cb8b0009001010010001200  f139 3
    *   t.sample(s, uvw, level(in.dx.x)) 05800cb8b0008821010010001200  f139 4
    *   t.sample(s, uvw, gradient3d(..)) 05800cb8b0009c01000010001200  f139 5
    *
    * and the per-lane staging is the same bank-6 publication (f5 = 6) our 2D
    * path already emits, read straight out of the level(reg) kernel.
    *
    * gradient3d carries *six* scalars rather than four; the extra pair
    * continues the same interleave into banks 16 and 18, witnessed on two
    * literal-gradient kernels (see AGX3_TEX_GRADIENT_BANK_DDX_Z). Array,
    * gather and comparison combinations remain closed. */
   if (d->three_d &&
       (d->array || d->cube ||
        d->gather != AGX3_TEX_GATHER_NONE || d->shadow))
      return false;

   /* Native cube sampling is a distinct dimension (SMP f76=4). Apple
    * publishes its third f32 coordinate with PACK_SMPIR f74=10, then samples
    * fixed coordinate bank 0. Implicit LOD and the immediate and register
    * level and bias selectors are independently witnessed, each with the same
    * f139/f29/f37 operands as a 2D sample:
    *
    *   texturecube       bias(0.5)  05800cb8b0001101010010020100
    *   texturecube_array bias(0.5)  05800cb8b0009101010010020100
    *
    * Cube and cube-array comparisons reuse that CEM path with their own
    * packed face / layer operands. */
   if (d->cube &&
       (d->three_d ||
        (d->coord != 0 && !g18g_cube_vector) ||
        (d->lod != AGX3_TEX_LOD_AUTO && d->lod != AGX3_TEX_LOD_LEVEL &&
         d->lod != AGX3_TEX_LOD_REGISTER && d->lod != AGX3_TEX_LOD_BIAS &&
         d->lod != AGX3_TEX_LOD_BIAS_REGISTER)))
      return false;

   const unsigned tk = d->texture_kind ? d->texture_kind : 1;
   const unsigned sk = d->sampler_kind ? d->sampler_kind : 1;
   if (tk > 3 || sk > 3 ||
       (tk != 1 && ((d->texture_value & 1) || d->texture_value > 252)) ||
       (sk != 1 && ((d->sampler_value & 1) || d->sampler_value > 252)) ||
       (d->texture_kill && tk != 2) || (d->sampler_kill && sk != 2) ||
       (enc == AGX_ISA_ENCODING_AGX3C && d->sampler_kill && !d->g18g))
      return false;

   /*
    * A dimensioned comparison used to require fixed texture and sampler
    * state. The descriptor operand class is orthogonal to both the depth
    * comparison and the packed array layer, and Apple's own bytes say so:
    * an `array<depth2d_array<float>, 8>` sampled through a runtime index,
    * against the same shader with a fixed texture and sampler --
    *
    *   fixed     1d800ca090002004010000001200          ep3, 14 B
    *   indexed   1dc00ca09c002004930380011200802000000000   ep4, 20 B
    *
    * differ in exactly the four descriptor fields and the endpoint they
    * imply: the texture gains f97 = 24 with f103/f104 going 1 -> 3/2, the
    * sampler gains f126 = 28 with f132/f133 going 1 -> 3/2. **f143 (the
    * comparison) and f77 = 2 (the packed layer) are identical in both**, so
    * neither depends on where the descriptor came from.
    *
    * That distinction is what a real application runs into. A shader with a
    * single texture instruction gets immediate descriptors from
    * agx_nir_lower_g17p_combined_sampler(); one with several -- SuperTuxKart's
    * sunlightshadow.frag samples its cascade four times -- falls back to
    * agx_nir_lower_g17p_indexed_samplers() and arrives here with registers,
    * where the whole shader used to fail closed.
    *
    * The non-driver sample ABIs stay refused: those change the operand
    * profile itself rather than the descriptor's class.
    */
   if (d->shadow && (d->array || d->cube) &&
       d->sample_abi != AGX3_TEX_SAMPLE_ABI_DRIVER)
      return false;

   /* C's indexed cube-array samples also mix GPR and uniform descriptor
    * operands in either direction. B retains its matching-class pairs. */
   if (d->cube && tk != sk &&
       !(enc == AGX_ISA_ENCODING_AGX3C &&
         ((tk == 2 && sk == 3) || (tk == 3 && sk == 2))) &&
       /* G18G: Native809 adds all mixed cube descriptor pairs. */
       !g18g_binding)
      return false;

   /* The coordinate offset is one immediate nibble pair in f43 plus two
    * selectors, f51 and f141, and it is orthogonal to every other property
    * of the sample. Each of these differs from its offset-free control by
    * exactly {f43 = (y << 4) | x, f51 = 1, f141 = 1} and nothing else
    * (metal-nt 27.1, -arch applegpu_g17p, fragment render pipelines):
    *
    *   t.sample(s, uv)                          05800cb8b0000000010010001200
    *   t.sample(s, uv, int2(1,1))               05800cb8b0000102018810001200
    *   t.sample(s, uv, level(2), int2(1,1))     05800cb8b0001b02018810081200
    *   t.sample(s, uv, level(reg), int2(1,1))   05800cb8b0001102018810001200
    *   t.sample(s, uv, bias(reg), int2(1,1))    05800cb8b0008102018810001200
    *   t.sample(s, uv, grad, int2(1,1))         05800cb8b0000902008810001200
    *   t.gather(s, uv, int2(1,1))               05800ca490000102018800001200
    *   t.sample(s, uv, layer, int2(1,1))        05800cb8b0001bb7018810001200
    *   depth.sample_compare(s, uv, r, int2(1,1))1d800ca090002102018800001200
    *   heap t.sample(s, uv, int2(1,1))          05a00cb8b8000102938b90011200...
    *
    * and the value sweep is the same in every LOD mode: offset (1,1) is
    * f43 = 17, (2,0) is f43 = 2, (0,2) is f43 = 32.
    *
    * Offset (0, 0) is folded to the plain form by native, so do not
    * manufacture the enabled encoding without a corresponding witness.
    *
    * f43 is a single shared immediate slot, and an *array* sample carrying
    * an immediate level or bias already claims it (see the LOD block below).
    * Apple does not co-encode an offset there: `t.sample(s, uv, 3u,
    * bias(0.5f), int2(1,1))` is 05800cb8b000451b010010001202, which keeps
    * the bias in f43 and selects f141 = 2 -- a form that publishes the
    * offset through an extra twelve-byte staged operand. That staging has no
    * slot plan here, so the collision is refused rather than silently
    * dropping one of the two values.
    *
    * Fresh runtime-indexed G17P argument-buffer witnesses use the driver
    * profile and the same offset fields with register descriptors. The 3D
    * form extends f43 to twelve bits, one signed nibble per coordinate.
    * A staged runtime 2D offset owns operand bank 6. The driver ABI therefore
    * prepackages its array layer and any register LOD/bias with the offset and
    * publishes the complete bank-6 word once. The direct APPLE_FRAGMENT
    * immediate layer still has the f43 collision above.
    */
   const bool direct_array_immediate_lod =
      d->array && d->sample_abi == AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT &&
      (d->lod == AGX3_TEX_LOD_BIAS || d->lod == AGX3_TEX_LOD_LEVEL);
   /* A staged 3D offset is witnessed beside a bias and a minimum: Apple packs
    * the X/Y nibbles and the bias into one word, adds the Z nibble at bank-6
    * bit 0, and names the minimum's packed register in SMP. */
   const bool staged_3d_bias_min =
      d->three_d && d->offset_staged &&
      (d->min_lod_register ||
       (enc == AGX_ISA_ENCODING_AGX3C && d->min_lod &&
        d->lod == AGX3_TEX_LOD_BIAS_REGISTER));
   const bool any_staged_offset = d->offset_staged || d->offset_register;
   /* AGX3C immediate offsets (native G17G, texwit3): exactly {f43, f51 = 1,
    * f141 = 1} over the offset-free form for 2D, register-layer 2D-array, 3D
    * (twelve-bit f43), comparison (2D and array) and gather (any component,
    * array, comparison) samples, in AUTO, LEVEL, REGISTER, BIAS_REGISTER and
    * GRADIENT LOD shapes, including negative nibbles:
    *   o_arr_auto  0f071600418062290100030000801100008000000000  f77=2 f43=17
    *   o_3d_neg    0f05160045806129010003000080780f008000000000  f43=0xf78
    *   o_cmp_grad  0f071200418000210100030000801100000000000000  f139=5 f43=17
    *   o_gath_cmp  0f071600418003210118110000801100000000000000  f144=1 f43=17
    * Runtime (staged, f141 = 2) offsets read bank 6 on C as on B: d_auto,
    * d_lvlr (f139 = 4) and d_arr (f77 = 2) differ from it exactly as B's
    * register-LOD and array forms do (M5StagedOffsetSampleWitnesses). */
   if (d->offset &&
       ((enc != AGX_ISA_ENCODING_AGX3B &&
         !(enc == AGX_ISA_ENCODING_AGX3C &&
           (!any_staged_offset ||
            /* Runtime (f141 = 2) 2D offsets, texwit3/4 d_*; a 2D array
             * reads its layer from the same bank-6 word (d_arr). */
            ((!d->three_d || staged_3d_bias_min) &&
             (!d->array || d->offset_staged_array)))) &&
         !((g18g_plain_2d && !d->offset_register) || g18g_gather)) ||
        (any_staged_offset
            ? d->offset_xyz != 0 || (d->three_d && !staged_3d_bias_min)
            : d->offset_xyz == 0 ||
                 d->offset_xyz > (d->three_d ? 0xfffu : 0xffu)) ||
        d->cube || (direct_array_immediate_lod && !any_staged_offset)))
      return false;

   if (!d->offset &&
       (d->offset_xyz != 0 || d->offset_staged || d->offset_register))
      return false;
   if (d->offset_register &&
       (d->offset_staged || d->gather == AGX3_TEX_GATHER_NONE || d->three_d))
      return false;
   if (d->offset_staged_array &&
       (!d->offset || !d->offset_staged || !d->array || d->three_d))
      return false;
   if (d->array && d->offset_staged &&
       d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER &&
       !d->offset_staged_array)
      return false;

   /* Metal's gather takes no LOD argument, so every gather witness is an
    * implicit-LOD one on AGX3b. The array layer, the coordinate offset and
    * the depth comparison do compose with it, each contributing exactly the
    * fields it contributes to an ordinary sample:
    *
    *   t.gather(s, uv)                       05800ca490000000010000001200
    *   t.gather(s, uv, layer)                05800ca490000db4010000001200  f77/f31/f37
    *   t.gather(s, uv, layer, int2(1,1))     05800ca490001bb7018800001200  + f43/f51/f141
    *   depth.gather_compare(s, uv, r)        05800ca490002000010000001200  + f143
    *   depth.gather_compare(s, uv, r, off)   05800ca490002102018800001200  + f43/f51/f141
    *
    * Fresh runtime-descriptor cube and depthcube witnesses use the ordinary
    * CEM coordinates with f144=1:
    *   05800ca414001000010280010100a02800004000
    *   05800ca414003000010280010100a02800004000
    * Runtime 2D offsets set f141=2 and read the merged bank-6 operand.
    * 3D gathers remain unsupported.
    */
   if (d->gather != AGX3_TEX_GATHER_NONE &&
       (d->gather > AGX3_TEX_GATHER_A ||
        d->lod != AGX3_TEX_LOD_AUTO || d->three_d))
      return false;

   /* Apple's depth sample_compare witnesses stage their reference value into
    * fixed bank 4 and set f143. The comparison is orthogonal to the LOD
    * shape: each of these differs from the corresponding *colour* sample by
    * f143 = 1, f75 = 1 instead of 15, and the dropped f74 mask-group bit,
    * and by nothing else.
    *
    *   depth.sample_compare(s, uv, r)             1d800ca090002000010000001200
    *   depth.sample_compare(s, uv, r, level(0))   1d800ca090002900010000001200  f139=2
    *   depth.sample_compare(s, uv, r, level(2))   1d800ca090002900010000081200  f139=2 f29=128
    *   depth.sample_compare(s, uv, r, bias(0.5))  1d800ca090002700010000021200  f139=1 f29=32
    *   depth.sample_compare(s, uv, r, level(reg)) 1d800ca090002001010000001200  f139=4
    *   depth.sample_compare(s, uv, r, bias(reg))  1d800ca090002400010000001200  f139=3
    *   depth.sample_compare(s, uv, r, grad)       1d800ca090002401000000001200  f139=5
    *   depth.gather_compare(s, uv, r)             05800ca490002000010000001200  f144=1
    *
    * A comparison gather returns four texels, so it carries the full f75
    * mask the ordinary gather does; every other comparison returns one.
    *
    * The array and cube comparisons here use their own separately packed
    * layer / CEM operands (f77 = 2, f76 = 4) rather than the register-layer
    * form the LOD sweep above was taken on, so they keep the implicit LOD
    * they were witnessed with. */
   /* A level or bias staged from a register is the varying-driven shadow
    * comparison: the plain cube shares one operand between CEM's face and the
    * level, and the cube array adds the auxiliary layer word beside its
    * face+level word, so SMP carries f77 = 2 or f76 = 4 with f139 = 3/4 --
    * exactly as Apple's depthcube_array `sample_compare(..., level(v))` and
    * `bias(v)` fragments do. */
   const bool staged_register_lod = d->lod == AGX3_TEX_LOD_REGISTER ||
                                    d->lod == AGX3_TEX_LOD_BIAS_REGISTER;
   if (d->shadow &&
       (d->three_d ||
        (enc == AGX_ISA_ENCODING_AGX3C && !d->g18g && d->offset && d->array &&
         (d->offset_register ||
          (d->offset_staged && !d->offset_staged_array))) ||
        d->mask != (d->gather != AGX3_TEX_GATHER_NONE ? 15u : 1u) ||
        (((d->cube && !staged_register_lod) ||
          (d->array && !d->offset && !d->vertex_stage &&
           !staged_register_lod)) &&
         d->lod != AGX3_TEX_LOD_AUTO &&
         /* An explicit level zero is what a vertex stage produces for a GLSL
          * `texture()` comparison, having no derivatives. It carries no LOD
          * operand at all, so it composes with the cube array's witnessed
          * face+layer word exactly as it already does with a plain cube. */
         !(d->lod == AGX3_TEX_LOD_LEVEL && d->lod_x64 == 0) &&
         /* A noncube array comparison gradient, with or without a minimum.
          * The four staged scalars and f139 = 5 compose with the packed
          * layer exactly as they do without the comparison, and the same
          * descriptor from a pre-raster stage -- which `vertex_stage` above
          * already exempts -- encodes byte for byte identically, because
          * nothing in this encoder reads the stage for a gradient. */
         !(d->array && !d->cube &&
           (d->lod == AGX3_TEX_LOD_GRADIENT ||
            (d->min_lod && d->min_lod_register))) &&
         !(d->cube && !d->array &&
           (d->lod == AGX3_TEX_LOD_REGISTER ||
            (d->min_lod && d->lod == AGX3_TEX_LOD_BIAS_REGISTER))))))
      return false;

   if (d->min_lod_register &&
       (!d->min_lod || d->cube || d->lod != AGX3_TEX_LOD_BIAS_REGISTER ||
        !((d->shadow && d->array) || (d->three_d && d->offset_staged)) ||
        d->min_lod_value > 190 || (d->min_lod_value & 1)))
      return false;

   const unsigned slots = enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~waits) == 0 && one_or_zero(d->wait_mask);
}


bool
agx3_tex_encode_sample(enum agx_isa_encoding enc,
                       const struct agx3_tex_sample_desc *d,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!sample_ok(enc, d) || (d->heap_uniform_base & 3) ||
       d->heap_uniform_base > 248 ||
       (d->heap_uniform_base && d->sample_abi != AGX3_TEX_SAMPLE_ABI_DRIVER))
      return false;

   /* The fetch baseline plus f74=1: the sample witnesses carry the extra
    * mask-group bit on every mask value, where the fetch form does not.
    * A gather is the one sample shape that also clears it. */
   const unsigned tk = d->texture_kind ? d->texture_kind : 1;
   const unsigned sk = d->sampler_kind ? d->sampler_kind : 1;
   const bool immediate = tk == 1 && sk == 1;
   const bool apple_fragment_abi =
      d->sample_abi == AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   const bool shadow_array = d->shadow && d->array;
   /* A staged offset occupies the same sampler-private word as the array
    * layer. The driver path prepackages both fields; only Apple's separate
    * fragment ABI retains its witnessed direct layer operand. */
   const bool direct_offset_array =
      apple_fragment_abi && d->array && d->offset_staged &&
      !d->offset_staged_array;
   /* SMP's direct f31/f37=3 layer names a uniform half. A driver array's
    * layer is a GPR, so every driver array reads a staged layer (f77=2): a
    * cube array stages it with CEM's face, any other array with PACK_SMPIR,
    * whatever the stage and LOD form. */
   const bool staged_driver_array =
      d->array && d->sample_abi == AGX3_TEX_SAMPLE_ABI_DRIVER &&
      !direct_offset_array;
   unsigned operand_profile = apple_fragment_abi ? 3 : 0;
   struct agx3_field_kv base[20];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 5);
   base[nb++] = (struct agx3_field_kv)KV(38, 1);
   base[nb++] = (struct agx3_field_kv)KV(52, 1);
   if (d->gather == AGX3_TEX_GATHER_NONE && !d->shadow && !d->query_lod)
      base[nb++] = (struct agx3_field_kv)KV(74, 1);
   base[nb++] = (struct agx3_field_kv)KV(89, 3);
   base[nb++] = (struct agx3_field_kv)KV(90, 3);
   base[nb++] = (struct agx3_field_kv)KV(103, tk);
   base[nb++] =
      (struct agx3_field_kv)KV(104, tk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   /* The driver descriptor ABI uses f83/f112=0/4 for every dimension.
    * Apple's f83/f112=4/8 fragment profile is a complete, incompatible ABI,
    * selected only by AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT.  In particular,
    * using Apple's profile with the driver's 2D-array descriptor produces a
    * fixed near-half-gray sample on G17P, just as it does for cubes.
    */
   if ((operand_profile & 1) || d->heap_uniform_base)
      base[nb++] = (struct agx3_field_kv)KV(
         83, d->heap_uniform_base + ((operand_profile & 1) ? 4 : 0));
   /* The operand profile, which a sparse residency request does NOT change.
    *
    * f112 reads as a small integer but the generated AGX3b tables deposit it
    * as a scattered bitmask: bit 0 must be zero, bits 5..2 land in instruction
    * bits 99..96, and bits 1, 6, 7 and 8 land elsewhere again.  So the "+4"
    * that matched-pair witnesses appear to show is not an increment of a
    * counter -- it clears profile bit 2 and sets bit 3 instead.
    *
    * Dropping bit 2 breaks sparse residency on G17P.  Measured on Neo over the
    * whole 3,420-case dEQP-VK.sparse_resources.shader_intrinsics family, one
    * field at a time and nothing else changed:
    *
    *   fetch refused, sample +4 (origin/agx3)     360 Pass   1,640 Fail
    *   fetch dense,   sample +4                 1,340 Pass     660 Fail
    *   fetch dense,   sample dense              1,490 Pass     510 Fail
    *
    * and on a 30-case fetch probe (2d_sparse_fetch and 2d_sparse_read over
    * r8/r32i/rgba8i and all four sizes), isolating this field alone:
    *
    *   fetch +4      0/30 Pass        fetch dense    30/30 Pass
    *
    * The final 510 are the layered sampled groups, which refuse at
    * endCommandBuffer before any sparse instruction runs, identically before
    * and after.
    *
    * The +4 fails per shape, not per encoder, which is why it read as a fetch
    * problem when the sparse encoders landed: with it, every texel fetch and
    * every explicit-LOD sample reports an unbacked page as resident and
    * returns texels that do not match, while the automatic-LOD sample and the
    * gather are correct.  (Both symptoms are what a zeroed destination would
    * produce -- a zero residency code reads as resident -- but that reading is
    * an inference; only the CTS outcomes above are measured.)  Apple's own
    * `sparse_sample`/`sparse_read` bytes do carry bit 3 rather than bit 2; why
    * their argument-buffer descriptors tolerate it and Honeykrisp's register
    * and uniform descriptors do not is not yet understood.  Hardware decides,
    * so the profile stays where the dense form puts it. */
   base[nb++] = (struct agx3_field_kv)KV(
      112, d->heap_uniform_base + ((operand_profile & 2) ? 8 : 4));
   base[nb++] = (struct agx3_field_kv)KV(118, 3);
   base[nb++] = (struct agx3_field_kv)KV(119, 3);
   base[nb++] = (struct agx3_field_kv)KV(132, sk);
   base[nb++] =
      (struct agx3_field_kv)KV(133, sk == AGX3_TEX_OPERAND_IMMEDIATE ? 1 : 2);
   base[nb++] = (struct agx3_field_kv)KV(138, 1);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(24, 2);
      base[nb++] = (struct agx3_field_kv)KV(60, 2);
      base[nb++] = (struct agx3_field_kv)KV(61, 164);
   }

   struct agx3_field_kv ov[40];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(19, d->coord);
   ov[n++] = (struct agx3_field_kv)KV(26, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(27, d->coord_kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(75, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(76, d->cube ? 4 : d->three_d ? 3 : 1);
   ov[n++] = (struct agx3_field_kv)KV(78, 2);  /* filtered sample */
   if (d->min_lod)
      ov[n++] = (struct agx3_field_kv)KV(140, 2);

   /* f156 is a group terminator, not a per-instruction implicit-LOD bit.
    * A lone AUTO/BIAS, staged register LOD, array, or cube sample forms a
    * one-op group and keeps the historical encoding. The stream packer marks
    * nonterminal members to suppress that default and supplies
    * sample_group_end when a multi-op group ends on a shape (including an
    * immediate LEVEL) that would not infer it in isolation. Explicit-gradient
    * samples are outside these groups. */
   const bool intrinsic_group = !d->vertex_stage && !d->compute_stage &&
      (d->lod == AGX3_TEX_LOD_AUTO || d->lod == AGX3_TEX_LOD_BIAS ||
       d->lod == AGX3_TEX_LOD_REGISTER ||
       d->lod == AGX3_TEX_LOD_BIAS_REGISTER ||
       (!d->g18g &&
        ((enc == AGX_ISA_ENCODING_AGX3C && d->lod == AGX3_TEX_LOD_LEVEL) ||
         (d->lod != AGX3_TEX_LOD_GRADIENT && (d->array || d->cube)))));
   if (d->sample_group_end ||
       (intrinsic_group && !d->sample_group_nonterminal))
      ov[n++] = (struct agx3_field_kv)KV(156, 1);
   if (d->array && !(enc == AGX_ISA_ENCODING_AGX3C &&
                     d->query_lod && !d->cube)) {
      /* Honeykrisp's driver descriptor ABI pairs arrays with the
       * sampler-private layer staged by PACK_SMPIR.  Native Apple fragment
       * witnesses use the direct f31/f37 path as part of their complete 4/8
       * ABI, where f31 is a uniform half.
       */
      if (d->query_lod) {
         /* No layer coordinate participates in the LOD derivative. */
         ov[n++] = (struct agx3_field_kv)KV(29, 0);
         ov[n++] = (struct agx3_field_kv)KV(37, 1);
         ov[n++] = (struct agx3_field_kv)KV(77, 1);
      } else if ((shadow_array && !direct_offset_array) ||
                 /* G18G stages every array layer in PACK_SMPIR bank 6. */
                 staged_driver_array || d->g18g) {
         ov[n++] = (struct agx3_field_kv)KV(77, 2);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(31, d->array_layer);
         ov[n++] = (struct agx3_field_kv)KV(37, 3);
         ov[n++] = (struct agx3_field_kv)KV(77, 1);
      }
   }
   ov[n++] = (struct agx3_field_kv)KV(94 + tk, tk == 1 ? d->texture : d->texture_value);
   ov[n++] = (struct agx3_field_kv)KV(123 + sk, sk == 1 ? d->sampler : d->sampler_value);
   if (d->texture_kill)
      ov[n++] = (struct agx3_field_kv)KV(107, 1);
   if (d->sampler_kill)
      ov[n++] = (struct agx3_field_kv)KV(136, 1);
   if (d->lod == AGX3_TEX_LOD_AUTO) {
      /* AUTO has no explicit LOD operand. Its group marker is emitted above. */
   } else if (d->lod == AGX3_TEX_LOD_BIAS) {
      if (d->array && !staged_driver_array) {
         ov[n++] = (struct agx3_field_kv)KV(43, d->lod_x64);
         ov[n++] = (struct agx3_field_kv)KV(51, 1);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(37, 1);
         ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
      }
      ov[n++] = (struct agx3_field_kv)KV(139, 1); /* bias mode */
   } else if (d->lod == AGX3_TEX_LOD_REGISTER) {
      ov[n++] = (struct agx3_field_kv)KV(28, 0); /* packed level bank */
      ov[n++] = (struct agx3_field_kv)KV(139, 4);
   } else if (d->lod == AGX3_TEX_LOD_BIAS_REGISTER) {
      ov[n++] = (struct agx3_field_kv)KV(28, 0); /* packed bias bank */
      ov[n++] = (struct agx3_field_kv)KV(139, 3);
      if (d->min_lod_register) {
         ov[n++] = (struct agx3_field_kv)KV(30, d->min_lod_value);
         ov[n++] = (struct agx3_field_kv)KV(37, 2);
         ov[n++] = (struct agx3_field_kv)KV(41, d->min_lod_kill);
      }
   } else if (d->lod == AGX3_TEX_LOD_GRADIENT) {
      /* The gradient form clears the implicit-derivative marker: the four
       * staged scalars are the derivative. */
      ov[n++] = (struct agx3_field_kv)KV(139, 5);
   } else {
       /* G18G: the layer is already in PACK_SMPIR bank 6; its immediate
        * level remains in the ordinary f29/f37 operand, even for arrays. */
      /* A level-zero comparison keeps the packed f77=2 layer operand.
       * Native array_compare_zero therefore uses f29/f37 for the level,
       * just like a non-array sample, rather than the direct-layer f43. */
      if (d->array && !shadow_array && !staged_driver_array && !d->g18g) {
         ov[n++] = (struct agx3_field_kv)KV(43, d->lod_x64);
         ov[n++] = (struct agx3_field_kv)KV(51, 1);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(37, 1);
         ov[n++] = (struct agx3_field_kv)KV(29, d->lod_x64);
      }
      ov[n++] = (struct agx3_field_kv)KV(139, 2); /* explicit-level mode */
   }
   if (d->gather != AGX3_TEX_GATHER_NONE) {
      ov[n++] = (struct agx3_field_kv)KV(144, 1);
      if (d->gather != AGX3_TEX_GATHER_R)
         ov[n++] = (struct agx3_field_kv)KV(145, d->gather -
                                                 AGX3_TEX_GATHER_R);
   }
   if (d->shadow)
      ov[n++] = (struct agx3_field_kv)KV(143, 1);
   if (d->query_lod) {
      /* LOD queries use their own selector on each ISA. */
      ov[n++] = enc == AGX_ISA_ENCODING_AGX3C
         ? (struct agx3_field_kv)KV(155, 2)
         : (struct agx3_field_kv)KV(146, 1);
   }
   if (d->offset) {
      if (d->offset_staged || d->offset_register) {
         ov[n++] = (struct agx3_field_kv)KV(141, 2);
      } else {
         ov[n++] = (struct agx3_field_kv)KV(43, d->offset_xyz);
         ov[n++] = (struct agx3_field_kv)KV(51, 1);
         ov[n++] = (struct agx3_field_kv)KV(141, 1);
      }
   }
   if (enc == AGX_ISA_ENCODING_AGX3B)
      ov[n++] = (struct agx3_field_kv)KV(147, d->final ? 2 : 0);
   else
      ov[n++] = (struct agx3_field_kv)KV(150, d->final ? 2 : 0);
   if (d->sparse) {
      if (enc == AGX_ISA_ENCODING_AGX3B)
         ov[n++] = (struct agx3_field_kv)KV(120, 1);
      ov[n++] = (struct agx3_field_kv)KV(142, 1);
   }
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(62 + i, 1);

   /* AGX3b sparse sampling needs ep4 even with immediate descriptors.
    * AGX3c carries the residency selector in its 22-byte ep3 form. */
   /* G17G's long form includes register/uniform descriptors in ep3.
    * Unlike Neo it has no separate ep4 form. */
   const int ep = enc == AGX_ISA_ENCODING_AGX3C ||
                        (immediate && !d->sparse && !d->heap_uniform_base) ? 3 : 4;
   if (enc == AGX_ISA_ENCODING_AGX3C &&
       (d->query_lod || d->shadow || d->gather != AGX3_TEX_GATHER_NONE)) {
      /* Query, comparison and gather use compact forms, including indexed
       * descriptors. Escalate when the operands need a longer form. */
      const int first_ep = d->query_lod ? (d->cube && !d->array ? 2 : 1)
                           : d->gather != AGX3_TEX_GATHER_NONE ? 2 : 0;
      for (int query_ep = first_ep; query_ep <= 3; ++query_ep)
         if (agx3_bind_compose(enc, AGX3_MN_smp, query_ep, base, nb, ov, n, out, len))
            return true;
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }
   if (!agx3_bind_compose(enc, AGX3_MN_smp, ep, base, nb, ov, n, out, len)) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }

   return true;
}

static bool
agx3_tex_stage_copy_hints_impl(enum agx_isa_encoding enc, bool wide_source,
                    uint16_t dest, uint16_t src,
                    uint8_t slot, uint16_t wait_mask,
                          bool src_cache, bool src_discard,
                    uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_COPY_MAX_LENGTH);
   *len = 0;
   if ((src_cache && src_discard) ||
       (enc == AGX_ISA_ENCODING_AGX3C && src_cache == src_discard))
      return false;

   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;

   /* High source registers require the extended M5 encoding. Keep the
    * original helper's range unchanged for other publication paths.
    */
   if ((wide_source && enc != AGX_ISA_ENCODING_AGX3C) ||
       (dest & 1) || (src & 1) || dest >= 256 ||
       src >= (wide_source ? 256 : 128))
      return false;

   const unsigned slots = agx_encoding_is_g17_class(enc) ? 8 : 6;
   const uint16_t waits = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   if (slot < 1 || slot > slots || (wait_mask & ~waits) ||
       !one_or_zero(wait_mask))
      return false;

   struct agx3_field_kv base[8];
   unsigned nb = 0;
   base[nb++] = (struct agx3_field_kv)KV(9, 5);
   base[nb++] = (struct agx3_field_kv)KV(10, 2);
   base[nb++] = (struct agx3_field_kv)KV(23, 2);
   base[nb++] = (struct agx3_field_kv)KV(24, 2);
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 1);
      base[nb++] = (struct agx3_field_kv)KV(29, 35);
   }

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   /* The compact form first; Apple escalates the encoding point when a slot
    * or wait bit does not fit, and the composer's first-success order
    * reproduces that choice on every witness. */
   for (int ep = wide_source && src >= 128 ? 1 : 0; ep <= 1; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_copy, ep, base, nb, ov, n, out, len))
         return true;
   }
   memset(out, 0, AGX3_TEX_COPY_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_tex_stage_copy_hints(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                          uint8_t slot, uint16_t wait_mask,
                          bool src_cache, bool src_discard,
                          uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_copy_hints_impl(enc, false, dest, src, slot, wait_mask,
                                         src_cache, src_discard, out, len);
}

bool
agx3_tex_stage_copy_scalar_m5(uint16_t dest, uint16_t src, uint8_t slot,
                              uint16_t wait_mask, bool src_cache,
                              bool src_discard,
                              uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_copy_hints_impl(AGX_ISA_ENCODING_AGX3C, true, dest, src,
                                         slot, wait_mask, src_cache,
                                         src_discard, out, len);
}

bool
agx3_tex_stage_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint8_t slot, uint16_t wait_mask,
                    uint8_t out[AGX3_TEX_COPY_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_copy_hints(enc,dest,src,slot,wait_mask,false,true,out,len);
}

bool
agx3_tex_encode_image_write_2d(
   enum agx_isa_encoding enc, const struct agx3_image_write_desc *d,
   uint16_t uniform_base, uint8_t staging_slot,
   uint8_t out[AGX3_IMAGE_WRITE_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   memset(out, 0, AGX3_IMAGE_WRITE_LENGTH);
   *len = 0;
   /* Operand dependencies are consumed by the preceding COPYs. This packet
    * waits for their staging slot. The PBE heap base and descriptor index
    * are independent operands, including for an immediate index. */
   const unsigned kind =
      d->image_kind ? d->image_kind : AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate = kind == AGX3_TEX_OPERAND_IMMEDIATE;
   const bool reg = kind == AGX3_TEX_OPERAND_REGISTER;
   if (enc != AGX_ISA_ENCODING_AGX3B ||
       (uniform_base & 3) || uniform_base > 508 ||
       staging_slot < 1 || staging_slot > 6 ||
       (d->wait_mask & ~UINT16_C(0x3f)) != 0 ||
       (!immediate && !reg) ||
       (immediate && (d->image_kill ||
          !((d->image_value == 0 && !d->normalized) ||
            d->image_value == 1))) ||
       /* f70 is the float-vs-integer bit and the state operand's form is a
        * different field family (f91 immediate, f92 GPR, f93 uniform); they
        * are orthogonal, so an indexed state does not imply normalized data.
        * metal-nt -arch applegpu_g17p on a uniform-indexed
        * `array<texture2d<T, access::write>, 4>`, one variable per kernel:
        *
        *   texture2d<float>  d72054000040001020040000883a0901
        *   texture2d<uint>   d72054000040001020040000881a0901
        *   texture2d<int>    d72054000040001020040000881a0901
        *
        * the state operand moves against the immediate-state controls and f70
        * does not, and int and uint are byte-identical. */
       (reg && ((d->image_value & 1) || d->image_value > 126)) ||
       (d->staging_bank & 7) || d->staging_bank > 8176 ||
       d->result_slot > 6 ||
       d->dim > AGX3_IMAGE_WRITE_3D)
      return false;

   /* The native integer direct-state0 packet is the same composed form
    * with base0/index0 and a slot1 wait. Do not special-case its bytes: that
    * would silently discard an explicit nonzero PBE base. */
   /* Nonzero field values recovered from native imgwci.ep0 instructions.
    * Every omitted field is intentionally zero. The immediate descriptor index
    * is f91. A GPR index is f92 with f99=2, and f103 kills it. metal-nt -arch
    * applegpu_g17p writing array<texture2d<int, access::write>, 4> ts at a
    * computed index emits
    *
    *   ts[gid.x & 3], index dead   d71054000040000021040000881a8900  f103
    *   same index read afterwards  d71054000040000020040000881a8900
    *
    * f93 with f99=3 instead names a uniform half: ts[i] with i a constant
    * buffer scalar, or an argument-buffer entry, is d71054000040000c20040000881a0901
    * (f93 12). The composer validates the complete selected form. */
   /* f72 is the texture type and f73=2 marks a PACK_IMGWIR-staged layer.
    * metal-nt -arch applegpu_g17p, int data at a register coordinate:
    *
    *   texture2d        d71054000040000000040000881a0900
    *   texture2d_array  d71054002000000000040000881a0900  (f79=4, f73=2)
    *   texture3d        d71054000040000000080000881a0900  (f72=3)
    */
   /* f5 names the staging bank holding the coordinates and data, and f12
    * the slot the write signals. metal-nt -arch applegpu_g17p gives each of
    * six conditional writes to array<texture2d<int, access::write>, 8> its
    * own bank and slot, f5 0/8/16/24/32/40 with f12 1..6. Forty
    * unconditional writes stage the first three at f5 0/256/304 on slots
    * 1/2/3 and issue all three imgwci with f12 1, each waiting its own
    * staging slot. */
   struct agx3_field_kv fields[25] = {
      KV(9, 5),   KV(12, d->result_slot ? d->result_slot : 1),
      KV(13, 1),  KV(24, 1),  KV(38, 1),
      KV(51, 1),  KV(52, 1),  KV(58 + staging_slot, 1), KV(71, 2),
      KV(72, d->dim == AGX3_IMAGE_WRITE_3D ? 3 : 1),
      KV(74, 1),  KV(85, 3),  KV(86, 3),   KV(113, 1),
      KV(70, d->normalized),
   };
   unsigned n = 15;
   fields[n++] = (struct agx3_field_kv)KV(79, uniform_base);
   if (d->staging_bank)
      fields[n++] = (struct agx3_field_kv)KV(5, d->staging_bank);
   if (d->dim == AGX3_IMAGE_WRITE_2D_ARRAY)
      fields[n++] = (struct agx3_field_kv)KV(73, 2);
   if (immediate) {
      fields[n++] = (struct agx3_field_kv)KV(91, d->image_value);
      fields[n++] = (struct agx3_field_kv)KV(99, 1);
      fields[n++] = (struct agx3_field_kv)KV(100, 1);
   } else if (d->image_kill) {
      fields[n++] = (struct agx3_field_kv)KV(92, d->image_value);
      fields[n++] = (struct agx3_field_kv)KV(99, 2);
      fields[n++] = (struct agx3_field_kv)KV(100, 2);
      fields[n++] = (struct agx3_field_kv)KV(103, 1);
   } else {
      fields[n++] = (struct agx3_field_kv)KV(92, d->image_value);
      fields[n++] = (struct agx3_field_kv)KV(99, 2);
      fields[n++] = (struct agx3_field_kv)KV(100, 2);
   }

   if (!agx3_bind_compose(enc, AGX3_MN_imgwci, 0, fields,
                          n,
                          NULL, 0, out, len) ||
       *len != AGX3_IMAGE_WRITE_LENGTH) {
      memset(out, 0, AGX3_IMAGE_WRITE_LENGTH);
      *len = 0;
      return false;
   }

   return true;
}

/* Preserve the native witness interface used by existing callers. Hk's
 * resolved descriptors use the explicit-base interface above. */
bool
agx3_tex_encode_image_write_rgba8_2d(
   enum agx_isa_encoding enc, const struct agx3_image_write_desc *d,
   uint8_t out[AGX3_IMAGE_WRITE_LENGTH], uint8_t *len)
{
   if (!d)
      return false;

   const bool immediate = !d->image_kind ||
                          d->image_kind == AGX3_TEX_OPERAND_IMMEDIATE;
   const uint16_t base = immediate && d->image_value == 1 ? 4 : 0;
   const uint8_t slot = immediate && d->image_value == 0 ? 1 : 2;
   return agx3_tex_encode_image_write_2d(enc, d, base, slot, out, len);
}

bool
agx3_tex_encode_image_write_g18g(enum agx_isa_encoding enc,
                                 const struct agx3_image_write_g18g_desc *d,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH],
                                 uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (d == NULL)
      return false;

   const unsigned kind = d->image_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate = kind == AGX3_TEX_OPERAND_IMMEDIATE;
   const bool reg = kind == AGX3_TEX_OPERAND_REGISTER;
   const bool uniform = kind == AGX3_TEX_OPERAND_UNIFORM;

   /* The data sits just below X, Y just above it and a layer or z just above
    * Y, all inside the seven-bit f116/f118 fields agx3_transcode_agx3d()
    * derives from f5. The one-component form is the 2D integer write alone,
    * as kick 19; everything else stages all four components. */
   const bool third = d->dim == AGX3_IMAGE_WRITE_2D_ARRAY ||
                      d->dim == AGX3_IMAGE_WRITE_3D;
   const unsigned data_halves = 2u * d->components;
   if (enc != AGX_ISA_ENCODING_AGX3C ||
       (d->components != 1 && d->components != 4) ||
       (d->components == 1 &&
        (d->float_data || d->dim != AGX3_IMAGE_WRITE_2D)) ||
       d->dim > AGX3_IMAGE_WRITE_3D || (d->x_half & 1) ||
       d->x_half < data_halves || d->x_half + (third ? 4 : 2) > 126 ||
       (!immediate && !reg && !uniform) ||
       (d->uniform_base & 3) || d->uniform_base > 252 ||
       (immediate && d->image_index > AGX3_IMAGE_WRITE_G18G_MAX_INDEX) ||
       (!immediate && ((d->image_index & 1) ||
                       d->image_index > (reg ? 126 : 252))) ||
       (d->image_kill && !reg) ||
       d->slot < 1 || d->slot > AGX3_IMAGE_WRITE_G18G_MAX_SLOT)
      return false;

   /* Apple's applegpu_g17g image_write(texture2d<uint, access::write>), the
    * kernel native run112 kick 19 ran as launch 1, writes its
    * uint4(v, 0, 0, 0) with
    *
    *   250411042420a0020080008c002000000100   f5 0 (bank 0)
    *
    * f10, f56 and f58 are this form's selectors; f114 4 stages one data
    * component, the other three being zero (a uint4(v) splat is f114 5 and
    * an RGBA write leaves it 0). f12 is the result slot, f59 the wait on
    * staging slot 1, f13 the release of the staged operands. f5 here names
    * the X half of G18G's layout, not G17G's bank: see the imgwci rule in
    * agx3_transcode.c.
    *
    * The rest of the vector, from metal-nt -arch applegpu_g18g
    * (scratchpad g18ref-out/probe/m6-image-write{,2,3,4}):
    *
    *  - f114 0 stages four components: iw_cs_rgba32f, iw2_{f4,u4,i4}_t*.
    *  - f70 is the stored data type, 1 float and 0 integer, int and uint
    *    byte-identical (iw2_u4_t0 = iw2_i4_t0), as on G17P.
    *  - f72 3 is texture3d (iw_cs_rgba32ui_3d, iw4_f4_3d), whose z sits at
    *    Y + 2; f73 2 is a 2D array whose PACK_IMGWIR layer sits at Y + 2
    *    (iw_cs_r32i_array, iw4_f4_array, iw4_i4_array, iw_frag_rgba8_array).
    *  - f91 is the immediate image index in the heap f79 names: 0..3 for a
    *    kernel's four write textures (iw4_four), 1 and 2 beside reads.
    *  - f12 is the slot the write signals, 1..4 across the four writes of
    *    iw4_four, and f59..f63 the wait on its staging slot.
    *  - f13 (and G18G's f117) releases the staged operands: every write that
    *    alone reads its operands sets it, including four writes in one
    *    kernel (iw4_four) and a write after a sample (iw4_sample_then_write);
    *    Apple clears it only on a write whose staged X/Y a later write or an
    *    in-flight sample still reads (iw3_two, iw4_two_apart, iw3_read2_write),
    *    which this packer never shares.
    *
    * Native459's paired G17G/G18G typed 2D/array writes ground explicit
    * heap bases (f79) and descriptor indices: f91/f99=1/f100=1 immediate,
    * f92/f99=2/f100=2 GPR, f93/f99=3/f100=2 uniform. A dead GPR sets f103.
    * The operand form is independent of the stored data type in f70. */
   struct agx3_field_kv fields[28] = {
      KV(5, d->x_half), KV(9, 5),   KV(10, 2),  KV(12, d->slot), KV(13, 1),
      KV(24, 1),        KV(38, 1),  KV(51, 1),  KV(52, 1),       KV(56, 2),
      KV(58, 0x57),     KV(58 + d->slot, 1),    KV(71, 2),
      KV(72, d->dim == AGX3_IMAGE_WRITE_3D ? 3 : 1),             KV(74, 1),
      KV(85, 3),        KV(86, 3),  KV(99, kind),
      KV(100, immediate ? 1 : 2), KV(113, 1),
      KV(114, d->components == 1 ? 4 : 0),
   };
   unsigned n = 21;
   fields[n++] = (struct agx3_field_kv)KV(79, d->uniform_base);
   if (d->float_data)
      fields[n++] = (struct agx3_field_kv)KV(70, 1);
   if (d->dim == AGX3_IMAGE_WRITE_2D_ARRAY)
      fields[n++] = (struct agx3_field_kv)KV(73, 2);
   if (d->image_index)
      fields[n++] = (struct agx3_field_kv)KV(90 + kind, d->image_index);
   if (d->image_kill)
      fields[n++] = (struct agx3_field_kv)KV(103, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_imgwci, 0, fields, n, NULL, 0, out, len)) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

static uint16_t
word(const uint8_t *p, unsigned i)
{
   return (uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8);
}

static bool
decode_b(const uint8_t *p, struct agx3_tex_fetch_desc *d)
{
   const uint16_t w0 = word(p, 0), w1 = word(p, 1), w2 = word(p, 2),
                  w4 = word(p, 4), w5 = word(p, 5);
   d->width = (w0 & 0x8000) ? AGX3_TEX_W32 : AGX3_TEX_W16;
   d->dest = d->width == AGX3_TEX_W32
                ? (((w0 >> 3) & 0x1f) << 1) | (((w1 >> 6) & 1) << 6)
                : ((w0 >> 3) & 0x1f) | (((w1 >> 6) & 3) << 5);
   d->coord = (((w0 >> 8) & 7) << 1) | (((w1 >> 8) & 3) << 4);
   d->coord_kill = (w1 >> 13) & 1;
   d->final = (w1 >> 15) & 1;

   const unsigned mask_code = ((w1 >> 11) & 3) | (((w2 >> 5) & 1) << 2) |
                              (((w5 >> 6) & 1) << 3);
   static const uint8_t masks[16] = {
      1, 2, 4, 8, 9, 3, 7, 15, 5, 6, 10, 12, 13, 11, 14, 0,
   };
   d->mask = masks[mask_code];
   if (!d->mask)
      return false;
   d->sampler_value = ((w4 >> 8) & 1) | ((w2 & 0xf) << 1) |
                      (((w4 >> 1) & 7) << 5);

   const unsigned slot_code = (w2 >> 10) & 7;
   if (slot_code > 5)
      return false;
   d->slot = slot_code + 1;
   d->texture = ((w4 >> 7) & 1) | (((w0 >> 11) & 0xf) << 1) |
                (((w4 >> 4) & 7) << 5);

   const unsigned wait_code = (w2 >> 7) & 7;
   if (wait_code > 6)
      return false;
   d->wait_mask = wait_code == 0 ? 0 : (1u << (wait_code - 1));
   return true;
}

static bool
decode_c(const uint8_t *p, struct agx3_tex_fetch_desc *d)
{
   const uint16_t w0 = word(p, 0), w1 = word(p, 1), w2 = word(p, 2),
                  w3 = word(p, 3);
   d->width = (w0 & 8) ? AGX3_TEX_W32 : AGX3_TEX_W16;
   d->dest = ((w3 >> 2) & 1) | (((w0 >> 4) & 0xf) << 1) |
             (((w1 >> 6) & 3) << 5) | (((w3 >> 12) & 1) << 7);
   d->coord = (((w1 >> 10) & 0x3f) << 2) | (((w1 >> 9) & 1) << 8);
   d->coord_kill = (w1 >> 4) & 1;
   d->final = (w2 >> 15) & 1;

   const unsigned mask_code = ((w3 >> 5) & 3) | (((w3 >> 11) & 1) << 2);
   if (mask_code == 0)
      d->mask = 1;
   else if (mask_code == 7)
      d->mask = 15;
   else
      return false;

   d->slot = ((w2 >> 3) & 3) | (((w3 >> 4) & 1) << 2);
   d->slot++;
   d->texture = ((w3 >> 3) & 1) | (((w2 >> 12) & 1) << 1) |
                (((w2 >> 13) & 3) << 2);
   const unsigned wait_code = (w3 >> 13) & 7;
   d->wait_mask = wait_code == 0 ? 0 : (1u << (wait_code - 1));
   return true;
}

bool
agx3_tex_decode_fetch(enum agx_isa_encoding enc, const void *instruction,
                      uint32_t instruction_size,
                      struct agx3_tex_fetch_desc *out)
{
   if (out == NULL)
      return false;
   memset(out, 0, sizeof(*out));
   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;
   if (enc == AGX_ISA_ENCODING_AGX3B && instruction && instruction_size == 20) {
      const uint8_t *p = instruction;
      struct agx3_tex_fetch_desc d = {0};
      if (!decode_b(p, &d))
         return false;
      d.texture = 0;
      d.texture_value = (((word(p, 0) >> 12) & 7) << 1) |
                        (((word(p, 4) >> 4) & 7) << 4) |
                        (((p[15] >> 2) & 1) << 7);
      d.wait_mask = ((p[4] >> 7) & 1) | ((p[5] & 3) << 1) |
                    (((p[18] >> 4) & 7) << 3);
      d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
      /* Repack both native index kinds. Their context selector, upper bits,
       * and every reserved field are validated by exact byte comparison. */
      for (unsigned kind = AGX3_TEX_OPERAND_IMMEDIATE;
           kind <= AGX3_TEX_OPERAND_UNIFORM; ++kind) {
         d.texture_kind = kind;
         d.texture_kill = kind == AGX3_TEX_OPERAND_REGISTER && (p[15] & 8);
         for (unsigned array = 0; array < 2; ++array) {
            for (unsigned three_d = 0; three_d < 2; ++three_d) {
               /* Sparse is the other ep4 axis: it forces the 20-byte form
                * even for an immediate descriptor, so both readings of an
                * immediate ep4 read have to be tried. */
               for (unsigned sparse = 0; sparse < 2; ++sparse) {
                  d.array = array;
                  d.three_d = three_d;
                  d.sparse = sparse;
                  if (kind == AGX3_TEX_OPERAND_IMMEDIATE) {
                     if (!sparse)
                        continue;
                     d.texture = (((word(p, 0) >> 12) & 7) << 1) |
                                 (((word(p, 4) >> 4) & 7) << 4) |
                                 (((p[15] >> 2) & 1) << 7);
                     d.texture_value = 0;
                  } else {
                     d.texture = 0;
                     d.texture_value = (((word(p, 0) >> 12) & 7) << 1) |
                                       (((word(p, 4) >> 4) & 7) << 4) |
                                       (((p[15] >> 2) & 1) << 7);
                  }
                  uint8_t canonical[AGX3_TEX_MAX_LENGTH], length;
                  if (agx3_tex_encode_fetch(enc, &d, canonical, &length) &&
                      length == instruction_size &&
                      !memcmp(canonical, instruction, length)) {
                     *out = d;
                     return true;
                  }
               }
            }
         }
      }
      return false;
   }
   const uint32_t expected = enc == AGX_ISA_ENCODING_AGX3B ? 14 : 8;
   if (instruction == NULL || instruction_size != expected)
      return false;

   struct agx3_tex_fetch_desc d = {0};
   const uint8_t *p = instruction;
   if (!(enc == AGX_ISA_ENCODING_AGX3B ? decode_b(p, &d) : decode_c(p, &d)) ||
       !fetch_ok(enc, &d))
      return false;

   /* The array and 3D axes are selectors the hand decoders above do not
    * extract. Recover them the same way everything else here is validated,
    * by requiring one reading to repack byte-for-byte. */
   for (unsigned a = 0; a < 2; ++a) {
      for (unsigned three_d = 0; three_d < 2; ++three_d) {
         uint8_t canonical[AGX3_TEX_MAX_LENGTH];
         uint8_t length = 0;
         d.array = a != 0;
         d.three_d = three_d != 0;
         if (agx3_tex_encode_fetch(enc, &d, canonical, &length) &&
             length == instruction_size &&
             memcmp(canonical, instruction, length) == 0) {
            *out = d;
            return true;
         }
      }
   }

   return false;
}

/* Fresh native G17P LOD fragment witnesses use this 12-byte PACK_SMPIR for a
 * scalar LOD, preserving the source when it remains live. f5=6 selects
 * operand bank 6, not architectural r3. The FLOAT source is a half, while the
 * INTEGER source is the full uint required by texture2d::read. */
/* PACK_SMPIR publishes a read's sampler operands at half bank + 6, beside the
 * coordinates that read stages at `bank`. A read that shares a shader with an
 * image write stages at bank 16, clear of imgwci's halves 0..14: metal-nt
 * -arch applegpu_g17p publishes a texture2d_array read's layer followed by a
 * write as 1705541600000000f0c00400, a texture2d read's uint level as
 * 1705541600000004f0c00c00, and both for texture2d_array as
 * 170554160010020070c06500 (layer r8), each with SMP f19=16. */
static bool
sampler_bank_valid(uint16_t bank)
{
   return bank == 0 || bank == 16;
}

static bool
stage_typed_lod_half(enum agx_isa_encoding enc, uint16_t half, uint16_t src,
                     uint8_t slot, uint16_t wait_mask, bool src_cache,
                     bool src_discard, enum agx3_tex_level_kind kind,
                     bool live_integer_source,
                     uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   /* M5 uses a seven-bit source field and eight producer slots. Integer
    * levels may remain live; floating levels consume their source here.
    * That admits the G18 helper's live integer level (native539) too, so
    * `live_integer_source` selects nothing here. */
   (void)live_integer_source;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) ||
       src >= (gpu1 ? 128 : 256) || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) || (src_cache && src_discard) ||
       (kind != AGX3_TEX_LEVEL_MINIMUM && kind != AGX3_TEX_LEVEL_FLOAT &&
        kind != AGX3_TEX_LEVEL_INTEGER) ||
       (gpu1 && kind != AGX3_TEX_LEVEL_MINIMUM &&
        (src_cache || (kind == AGX3_TEX_LEVEL_FLOAT && !src_discard))))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, half), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, (unsigned)kind), KV(75, 16),
      KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                         gpu1 ? 12 : 10, ov, n, out, len))
      return false;

   /* The source is an eight-bit half-register index (f16 bits 0..6 at
    * instruction bits 41..47, bit 7 at bit 48). Apple's encoder's 191
    * bound is a compiler allocation policy, not a narrower wire field.
    * Keep the same native instruction for Mesa's full 256-half bank. */
   if (!gpu1)
      out[6] |= (src >> 7) & 1;
   return true;
}

bool
agx3_tex_stage_typed_lod_at(enum agx_isa_encoding enc, uint16_t bank,
                            uint16_t src, uint8_t slot, uint16_t wait_mask,
                            bool src_cache, bool src_discard,
                            enum agx3_tex_level_kind kind,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C ? ((bank & 7) || bank > 248)
                                    : !sampler_bank_valid(bank)) {
      if (out && len) {
         memset(out, 0, AGX3_TEX_MAX_LENGTH);
         *len = 0;
      }
      return false;
   }
   return stage_typed_lod_half(enc, bank + 6, src, slot, wait_mask,
                               src_cache, src_discard, kind, false, out, len);
}

/* A 2D-array sample with a bias and a minimum LOD needs bank 6 for its layer
 * and bias, so Apple publishes the minimum at half 4 instead. metal-nt -arch
 * applegpu_g17p compiles t.sample(s, uv, layer, bias(b), min_lod_clamp(m)) on
 * a texture2d_array to
 *
 *   1705540400001004f0c00800   minimum r0, discarded, slot 1
 *   1705540600040c00f0c02500   layer r2 and bias h3 at bank 6
 *
 * and the cube-array form publishes a minimum in r4 as
 * 1705540400081004f0c00800. */
bool
agx3_tex_stage_array_minimum_lod(enum agx_isa_encoding enc, uint16_t src,
                                 uint8_t slot, uint16_t wait_mask,
                                 bool src_cache, bool src_discard,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH],
                                 uint8_t *len)
{
   return stage_typed_lod_half(enc, 4, src, slot, wait_mask, src_cache,
                               src_discard, AGX3_TEX_LEVEL_MINIMUM, false, out, len);
}

bool
agx3_tex_stage_typed_lod_g18g(uint16_t bank, uint16_t src, uint8_t slot,
                              uint16_t wait_mask, bool src_cache,
                              bool src_discard, enum agx3_tex_level_kind kind,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (bank > 120 || (bank & 7)) {
      if (out && len) {
         memset(out, 0, AGX3_TEX_MAX_LENGTH);
         *len = 0;
      }
      return false;
   }
   return stage_typed_lod_half(AGX_ISA_ENCODING_AGX3C,
                               bank + AGX3_TEX_G18G_LEVEL_WORD, src, slot,
                               wait_mask, src_cache, src_discard, kind, true, out,
                               len);
}

bool
agx3_tex_stage_typed_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                         uint16_t wait_mask, bool src_cache, bool src_discard,
                         enum agx3_tex_level_kind kind,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_typed_lod_at(enc, 0, src, slot, wait_mask, src_cache,
                                      src_discard, kind, out, len);
}

bool
agx3_tex_stage_lod(enum agx_isa_encoding enc, uint16_t src, uint8_t slot,
                   uint16_t wait_mask, bool src_cache, bool src_discard,
                   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_typed_lod(enc, src, slot, wait_mask, src_cache,
                                   src_discard, AGX3_TEX_LEVEL_FLOAT, out,
                                   len);
}

/* G17G's PACK_SMPIR can also build the per-lane level/bias word in an
 * ordinary register instead of staging bank 6. Apple uses that form when
 * one bias feeds several in-flight samples: the three-comparison bias(reg)
 * fragment of gpu/m5-shadow-2d-reference.metal packs its half-6 bias once
 *
 *   2702003a2d00aa02000040301000   f2 = 0 (r0), f16 = 6, f27, f11 = 1, f56 = 3
 *
 * and then COPYs r0 into each block's +6 word (3b802104, 338007022281a41a,
 * b30007022682a40a). Against the bank-6 stager above, f2 replaces f5, f9
 * drops from 5 to 2, f11 = 1 selects the register destination, f56 moves
 * from 1 to 3, and the slot and wait fields are gone: the word is written
 * synchronously, so the COPY that publishes it carries the transaction. */
bool
agx3_tex_pack_lod_word(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       bool src_cache, bool src_discard,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   /* One even 32-bit destination below half 128, a seven-bit half source,
    * and exactly one of the two source hints, as for every G17G COPY. */
   if (enc != AGX_ISA_ENCODING_AGX3C || (dest & 1) || dest >= 128 ||
       src >= 128 || src_cache == src_discard)
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(11, 1), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(56, 3), KV(57, 119),
      KV(74, AGX3_TEX_LEVEL_FLOAT), KV(75, 16),
   };
   const struct agx3_field_kv ov[] = {
      KV(2, dest), KV(16, src), KV(25, src_cache), KV(27, src_discard),
   };
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov,
                            sizeof(ov) / sizeof(ov[0]), out, len);
}

/* Native bias+minimum publishes both halves in one PACK_SMPIR. f16 is the
 * minimum, f30 the bias, regardless of their order in the source register.
 * NIR packs bias low and minimum high, so the caller reverses their addresses.
 * Two scalar publications would overwrite the shared bank-6 operand. */
bool
agx3_tex_stage_bias_min(enum agx_isa_encoding enc, uint16_t bias,
                        uint16_t minimum, uint8_t slot, uint16_t wait_mask,
                        bool src_cache, bool src_discard,
                        uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || minimum >= 128 ||
       bias >= (gpu1 ? 128 : 256) ||
       slot < 1 || slot > slots || (wait_mask & ~((1u << slots) - 1)) ||
       (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 2), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(74, 1), KV(76, 2), KV(77, 16),
   };
   struct agx3_field_kv ov[18];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, minimum);
   ov[n++] = (struct agx3_field_kv)KV(30, bias);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   ov[n++] = (struct agx3_field_kv)KV(39, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, src_discard);
   if (gpu1) {
      ov[n++] = (struct agx3_field_kv)KV(56, 1);
      ov[n++] = (struct agx3_field_kv)KV(57, 119);
   }
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* Native G17P gather offsets insert signed nibbles from two 16-bit
 * registers into bits 12 and 28 of bank 6. Array gathers merge the normalized
 * layer as the third operand, never as a separate bank-6 write.
 * Runtime descriptor witnesses, slot 6, discarded inputs:
 *   x=12, y=14:          1705540600183800f0e40100
 *   x=16, y=17, layer=8: 1705540600204440f0e60300
 */
bool
agx3_tex_stage_gather_offset_at(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src,
                             bool array, uint16_t packed_layer,
                             uint8_t slot, bool cache, bool discard,
                             uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   /* AGX3C: the plain (non-array) form of the same PACK_SMPIR with C's
    * fixed f56 = 1 / f57 = 119 (texwit3 d_gath). Array gathers stage their
    * layer differently on C (a GPR-packed layer, d_arr) and stay refused.
    * G18G alone stages a runtime gather offset at half 2 (dest == 2, see
    * agx3_pack_binary.c g18g_runtime_gather), including the array form with
    * its packed layer. */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const bool g18g = m5 && dest == 2;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !(m5 && (!array || g18g))) ||
       (dest != 6 && !g18g) || (src & 1) ||
       src > 126 || slot < 1 || slot > (m5 ? 8 : 6) || (cache && discard) ||
       (array && ((packed_layer & 1) || packed_layer > (g18g ? 126 : 190))))
      return false;

   const struct agx3_field_kv fields[] = {
      KV(5, dest), KV(9, 5), KV(10, 2), KV(12, slot),
      KV(16, src), KV(23, 2), KV(24, 1), KV(25, cache), KV(27, discard),
      KV(30, src + 1), KV(37, 2), KV(38, 1), KV(39, cache), KV(41, discard),
      KV(44, array ? packed_layer : 0), KV(51, array ? 2 : 1),
      KV(52, 2), KV(55, array),
      KV(74, 5), KV(75, 12), KV(76, 5), KV(77, 28),
      KV(56, 1), KV(57, 119),
   };
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                            sizeof(fields) / sizeof(fields[0]) - (m5 ? 0 : 2),
                            NULL, 0, out, len);
}

bool
agx3_tex_stage_gather_offset(enum agx_isa_encoding enc, uint16_t src,
                           bool array, uint16_t packed_layer,
                           uint8_t slot, bool cache, bool discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_gather_offset_at(enc, 6, src, array, packed_layer,
                                        slot, cache, discard, out, len);
}

/* The dynamic array-layer witness uses PACK_SMPIR like register LOD, but its
 * operand-type selectors differ. In particular, f74=4/f75=0 produces the
 * native 1705540600080000f0c00400 for discarded r2 on slot 1; the LOD
 * baseline's f74=2/f75=16 silently leaves the sample destination unwritten.
 * Register-indexed array samples use the independently witnessed slot-6 form
 * 1705540600180000f0e40400 for discarded r6. */
bool
agx3_tex_stage_array_layer_at(enum agx_isa_encoding enc, uint16_t bank,
                              uint16_t src, uint8_t slot, uint16_t wait_mask,
                              bool src_cache, bool src_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) ||
       (gpu1 ? ((bank & 7) || bank > 248) : ((bank & 3) || bank > 248)) ||
       src >= (gpu1 ? 128 : 256) || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, bank + 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4), KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                         gpu1 ? 11 : 9, ov, n, out, len))
      return false;

   if (!gpu1)
      out[6] |= (src >> 7) & 1;
   return true;
}

bool
agx3_tex_stage_array_layer(enum agx_isa_encoding enc, uint16_t src,
                           uint8_t slot, uint16_t wait_mask, bool src_cache,
                           bool src_discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_array_layer_at(enc, 0, src, slot, wait_mask,
                                        src_cache, src_discard, out, len);
}

/* metal-nt -arch applegpu_g17g / applegpu_g18g (toolchain 27.1.266.1) on the
 * same texture2d_array fragments, the layer a per-lane value:
 *
 *   image.read(uint2(v.xy), uint(v.z), 0)       read, level zero
 *   image.sample(s, v.xy, uint(v.z))            filtered sample
 *   G17G  pack_smpir f5 6 f16 src   smp f19 0            (X 0, Y 2, layer 6)
 *   G18G  pack_smpir f5 2 f16 src   smp f19 4 f56 0      (X 4, Y 0, layer 2)
 *
 * and the corpus's m5_sample_array_level_zero pair shows the same move. The
 * two PACK_SMPIR vectors differ in f5 alone (the source half follows
 * register allocation). */
bool
agx3_tex_stage_array_layer_g18g(uint16_t src, uint8_t slot,
                                uint16_t wait_mask, bool src_cache,
                                bool src_discard,
                                uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (src >= 128 || slot < 1 || slot > 8 || (wait_mask & ~0xffu) ||
       (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, AGX3_TEX_G18G_ARRAY_LAYER), KV(9, 5), KV(10, 2), KV(23, 2),
      KV(24, 1), KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4), KV(56, 1),
      KV(57, 119),
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_pack_smpir, 0, base,
                            11, ov, n, out, len);
}

/* metal-nt applegpu_g17g / applegpu_g18g, texture2d_array
 * sample(s, v.xy, uint(v.z), level(v.x)) and bias(v.x), layer and level
 * per-lane (m6-fetch-array-varying-layer):
 *
 *   G17G  pack_smpir f5 6 f16 layer f30 lod f37 2 f76 2 f77 16   smp f19 0
 *   G18G  pack_smpir f5 2 f16 layer f30 lod f37 2 f76 2 f77 16   smp f19 4
 *
 * agx3_tex_stage_array_typed_lod()'s G17G vector with f5 = 2. */
bool
agx3_tex_stage_array_lod_g18g(uint16_t layer, uint16_t lod, uint8_t slot,
                              uint16_t wait_mask, bool layer_cache,
                              bool layer_discard, bool lod_cache,
                              bool lod_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_array_typed_lod_g18g(
      layer, lod, AGX3_TEX_LEVEL_FLOAT, slot, wait_mask, layer_cache,
      layer_discard, lod_cache, lod_discard, out, len);
}

bool
agx3_tex_stage_array_typed_lod_g18g(uint16_t layer, uint16_t lod,
                                    enum agx3_tex_level_kind kind,
                                    uint8_t slot, uint16_t wait_mask,
                                    bool layer_cache, bool layer_discard,
                                    bool lod_cache, bool lod_discard,
                                    uint8_t out[AGX3_TEX_MAX_LENGTH],
                                    uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (layer >= 128 || (layer & 1) || lod >= 128 || slot < 1 || slot > 8 ||
       (wait_mask & ~0xffu) || (layer_cache && layer_discard) ||
       (lod_cache && lod_discard) ||
       (kind != AGX3_TEX_LEVEL_FLOAT && kind != AGX3_TEX_LEVEL_INTEGER))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, AGX3_TEX_G18G_ARRAY_LAYER), KV(9, 5), KV(10, 2), KV(23, 2),
      KV(24, 1), KV(37, 2), KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4),
      KV(76, (unsigned)kind), KV(77, 16), KV(56, 1),
      KV(57, 119),
   };
   struct agx3_field_kv ov[15];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, layer);
   ov[n++] = (struct agx3_field_kv)KV(30, lod);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, layer_discard);
   ov[n++] = (struct agx3_field_kv)KV(39, lod_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, lod_discard);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_pack_smpir, 0, base,
                            14, ov, n, out, len);
}

/* metal-nt -arch applegpu_g17p (AIR-NT 32023.917) stages a texture2d_array
 * write's layer with PACK_IMGWIR, whose operand fields are PACK_SMPIR's,
 * including the eighth source bit in bit 48:
 *
 *   t.write(v, gid.xy, gid.z), discarded r2:  9705540600081000f0c00000
 *   t.write(v, gid.xy, l), discarded r0:      9705540600001000f0c00000
 *   layer at half 184 (90 live values):       9705540600701100f0c00000
 */
bool
agx3_tex_stage_image_write_layer(enum agx_isa_encoding enc, uint16_t src,
                                 uint16_t dest, uint8_t slot,
                                 uint16_t wait_mask,
                                 bool src_cache, bool src_discard,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH],
                                 uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      if ((src & 1) || src > 254 || (dest & 1) || dest > 8190 ||
          slot < 1 || slot > 8 || wait_mask || (src_cache && src_discard))
         return false;
      const struct agx3_field_kv fields[] = {
         KV(5, dest), KV(9, 5), KV(10, 2), KV(12, slot), KV(16, src),
         KV(23, 2), KV(24, 1), KV(25, src_cache), KV(27, src_discard),
         KV(38, 1), KV(51, 1), KV(52, 2), KV(56, 1), KV(57, 118), KV(74, 4),
      };
      return agx3_bind_compose(enc, AGX3_MN_pack_imgwir, 0, fields,
                               sizeof(fields) / sizeof(fields[0]),
                               NULL, 0, out, len);
   }

   if (enc != AGX_ISA_ENCODING_AGX3B || (src & 1) || src > 254 ||
       (dest & 1) || dest > 8190 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4),
   };
   struct agx3_field_kv ov[11];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_pack_imgwir, 0, base,
                          sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   out[6] |= (src >> 7) & 1;
   return true;
}

/* Runtime-indexed texture2d_array and depth2d_array sample witnesses publish
 * layer half 12 and LOD half 13 on slot 6 with the same single instruction:
 *
 *   level(reg), bias(reg): 1705540600183400f0e42500
 *   layer alone:          1705540600180000f0e40400
 *
 * Exact recomposition isolates f30/f37/f41 as the second register operand,
 * f76=2 as its float type, and f77=16 as its position in the packed word.
 * Two independent publications to bank 6 would overwrite the first operand.
 */
static bool
agx3_tex_stage_array_typed_lod(enum agx_isa_encoding enc, uint16_t bank,
                               uint16_t layer, uint16_t lod, uint8_t slot,
                               uint16_t wait_mask, bool layer_cache,
                               bool layer_discard, bool lod_cache,
                               bool lod_discard,
                               enum agx3_tex_level_kind kind,
                               uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || (gpu1 ? ((bank & 7) || bank > 248) : !sampler_bank_valid(bank)) || layer >= 128 ||
       (!gpu1 && (layer & 1)) || lod >= (gpu1 ? 128 : 256) || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) ||
       (kind != AGX3_TEX_LEVEL_FLOAT && kind != AGX3_TEX_LEVEL_MINIMUM &&
        kind != AGX3_TEX_LEVEL_INTEGER) ||
       (!gpu1 && kind == AGX3_TEX_LEVEL_INTEGER && (lod & 1)) ||
       (layer_cache && layer_discard) || (lod_cache && lod_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, bank + 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 2), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(74, 4), KV(76, (unsigned)kind), KV(77, 16), KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[15];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, layer);
   ov[n++] = (struct agx3_field_kv)KV(30, lod);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, layer_discard);
   ov[n++] = (struct agx3_field_kv)KV(39, lod_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, lod_discard);
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base, gpu1 ? 14 : 12,
                            ov, n, out, len);
}

/* A G17G cube sample with a per-lane level publishes that level *paired with
 * the cube Z value*, in the single PACK_SMPIR that follows the CEM group --
 * it does not stage the level separately beforehand. Apple's fragment
 * (gpu/m5-shadow-array-cube-reference.metal; text-16 of the extracted
 * corpus is the plain depthcube sample_compare with level(reg), whose sample
 * is the byte-exact 0f05160041810221010002000000 this packer already emits)
 * carries exactly one PACK_SMPIR between the finalizers and the sample:
 *
 *   3702000a2500aa0ac00060300900
 *
 * which is agx3_tex_stage_cube_z's f74 = 10 form with the array pair's
 * second operand attached -- f30 the level's half, f37/f41 marking it. Byte
 * for byte it is agx3_tex_stage_array_lod's encoding with **f74 = 10 in
 * place of f74 = 4**; that one field is what makes the pair a cube Z plus a
 * level rather than a layer plus a level. Recomposed against the witness by
 * sweeping f74, f76, f77 and both discard bits: exactly one combination
 * reproduces it.
 *
 * Publishing the level on its own beforehand through the plain 2D stager
 * (f74 = 2, f75 = 16 -- the "LOD baseline" spelling noted above as silently
 * leaving the sample unserved) makes the sampler read level zero for every
 * lane, whatever level was asked for.
 */
static bool
agx3_tex_stage_cube_z_lod_at(enum agx_isa_encoding enc, bool cube_array,
                          uint8_t bank,
                          uint16_t z, uint16_t lod, uint8_t slot,
                          uint16_t wait_mask, bool z_cache, bool z_discard,
                          bool lod_cache, bool lod_discard,
                          uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || (bank != 2 && bank != 6) ||
       lod >= 128 || slot < 1 || slot > 8 ||
       (wait_mask & ~0xffu) || (z_cache && z_discard) ||
       (lod_cache && lod_discard) ||
       (cube_array ? (z >= 256 || (z & 1)) : z >= 128))
      return false;

   /* The cube-array pair keeps the same f74/f76/f77 trio and adds the four
    * selectors that make its first operand the combined face-and-layer word
    * agx3_tex_stage_cube_array_shadow already publishes. Recomposed against
    * Apple's depthcube_array sample_compare with level(reg),
    * 3702004a2500aa0ac00261402900. */
   const struct agx3_field_kv base[] = {
      KV(5, bank), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 2), KV(38, 1), KV(51, cube_array ? 2 : 1), KV(52, 2),
      KV(74, 10), KV(76, 2), KV(77, 16), KV(56, 1), KV(57, 119),
      KV(44, 8), KV(55, 1), KV(75, 12),
   };
   struct agx3_field_kv ov[15];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, cube_array ? z >> 1 : z);
   ov[n++] = (struct agx3_field_kv)KV(30, lod);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, z_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, z_discard);
   ov[n++] = (struct agx3_field_kv)KV(39, lod_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, lod_discard);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base, cube_array ? 17 : 14,
                            ov, n, out, len);
}

bool
agx3_tex_stage_cube_z_lod_g17g(enum agx_isa_encoding enc, bool cube_array,
                          uint16_t z, uint16_t lod, uint8_t slot,
                          uint16_t wait_mask, bool z_cache, bool z_discard,
                          bool lod_cache, bool lod_discard,
                          uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_cube_z_lod_at(enc, cube_array, 6, z, lod, slot,
      wait_mask, z_cache, z_discard, lod_cache, lod_discard, out, len);
}

/* Native G18 cube and cube-array level/bias references differ from the
 * G17 publication only in its destination half: Y + 2, with Y at zero. */
bool
agx3_tex_stage_cube_z_lod_g18g(enum agx_isa_encoding enc, bool cube_array,
                          uint16_t z, uint16_t lod, uint8_t slot,
                          uint16_t wait_mask, bool z_cache, bool z_discard,
                          bool lod_cache, bool lod_discard,
                          uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_cube_z_lod_at(enc, cube_array, 2, z, lod, slot,
      wait_mask, z_cache, z_discard, lod_cache, lod_discard, out, len);
}

/* A G17G cube-array comparison normalizes its integer array layer into a
 * general register before the face-and-layer publication consumes it. Apple's
 * depthcube_array fragment (gpu/m5-shadow-array-cube-reference.metal,
 * text-08 of the extracted corpus) emits
 *
 *   2700009a2d00aa12000040308100
 *
 * between the CEM axes, and it is the only PACK_SMPIR in this file that
 * writes a **general register** rather than the staging file: f9 = 2 instead
 * of 5, with f56 raised from 1 to 3. `agx3_tex_pack_ms_array_layer` does the
 * same job for a multisample array. Without it the publication reads the
 * IR's raw layer half and the sampler never sees the layer CEM's face is
 * meant to join, which is why every cube-array comparison rendered as one
 * face for the whole quad.
 *
 * Recomposed by sweeping f10, f11, f24, f51, f56, f74 and both source hints:
 * exactly one combination reproduces the witness.
 */
bool
agx3_tex_normalize_cube_array_layer(enum agx_isa_encoding enc, uint16_t dest,
                                    uint16_t layer, bool layer_cache,
                                    bool layer_discard,
                                    uint8_t out[AGX3_TEX_MAX_LENGTH],
                                    uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || dest >= 128 || (dest & 1) ||
       layer >= 128 || (layer_cache && layer_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2),  KV(10, 2), KV(11, 1), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(56, 3), KV(57, 119), KV(74, 4),
   };
   struct agx3_field_kv ov[4];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, layer);
   ov[n++] = (struct agx3_field_kv)KV(25, layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, layer_discard);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base, 11, ov, n, out, len);
}

bool
agx3_tex_stage_array_lod(enum agx_isa_encoding enc, uint16_t layer,
                         uint16_t lod, uint8_t slot, uint16_t wait_mask,
                         bool layer_cache, bool layer_discard,
                         bool lod_cache, bool lod_discard,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_array_typed_lod(
      enc, 0, layer, lod, slot, wait_mask, layer_cache, layer_discard,
      lod_cache, lod_discard, AGX3_TEX_LEVEL_FLOAT, out, len);
}

/* A 2D-array gradient sample with a minimum LOD packs the minimum beside the
 * layer the same way (texture2d_array gradient2d + min_lod_clamp, layer r14
 * and minimum h15: 17055406001c3c00f0c02d00). */
bool
agx3_tex_stage_array_minimum_gradient(enum agx_isa_encoding enc,
                                      uint16_t layer, uint16_t minimum,
                                      uint8_t slot, uint16_t wait_mask,
                                      bool layer_cache, bool layer_discard,
                                      bool minimum_cache,
                                      bool minimum_discard,
                                      uint8_t out[AGX3_TEX_MAX_LENGTH],
                                      uint8_t *len)
{
   return agx3_tex_stage_array_typed_lod(
      enc, 0, layer, minimum, slot, wait_mask, layer_cache, layer_discard,
      minimum_cache, minimum_discard, AGX3_TEX_LEVEL_MINIMUM, out, len);
}

bool
agx3_tex_stage_array_integer_lod_at(
   enum agx_isa_encoding enc, uint16_t bank, uint16_t layer, uint16_t lod,
   uint8_t slot, uint16_t wait_mask, bool layer_cache, bool layer_discard,
   bool lod_cache, bool lod_discard, uint8_t out[AGX3_TEX_MAX_LENGTH],
   uint8_t *len)
{
   return agx3_tex_stage_array_typed_lod(
      enc, bank, layer, lod, slot, wait_mask, layer_cache, layer_discard,
      lod_cache, lod_discard, AGX3_TEX_LEVEL_INTEGER, out, len);
}

/* Apple's immediate-layer plus immediate-LOD offset witness cannot use SMP's
 * shared f43 field. It instead emits this 12-byte PACK_SMPIR into bank 6 and
 * selects it with SMP f141=2. Sweeping int2(1,1), int2(2,0), int2(0,2), and
 * int2(-1,-1) fixes the two signed lanes at positions 12 and 28. The recovered
 * composer grammar accepts the same form with f16/f30 register sources and
 * f23/f37 selecting them, which is the runtime pcOffset form used here. */
bool
agx3_tex_stage_offset_2d(enum agx_isa_encoding enc, uint16_t x, uint16_t y,
                         uint8_t slot, uint16_t wait_mask, bool source_cache,
                         bool source_discard,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   /* AGX3C: the same PACK_SMPIR offset form plus C's fixed f56 = 1 and
    * f57 = 119, eight slots, and seven-bit X (no high bit fixup). Native G17G
    * runtime offsets (texwit3 d_auto, d_gath): X = h2, Y = h4, slot 1, both
    * last uses
    *   3702000a2500aa0a000160300000
    * and d_cmp reads the two halves of one register (h4, h5). */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || x >= (m5 ? 128 : 256) ||
       y >= 256 || (!m5 && ((x & 1) || (y & 1))) || slot < 1 ||
       slot > (m5 ? 8 : 6) || (wait_mask & ~(m5 ? 0xff : 0x3f)) ||
       (source_cache && source_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 2), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(74, 5), KV(75, 12), KV(76, 5), KV(77, 28),
      KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[20];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, x & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(30, y);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, source_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, source_discard);
   if (m5) {
      /* C names each source's own lifetime hint. */
      ov[n++] = (struct agx3_field_kv)KV(27, source_discard);
      ov[n++] = (struct agx3_field_kv)KV(39, source_cache);
   }
   for (unsigned i = 0; i < (m5 ? 8u : 6u); ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                          sizeof(base) / sizeof(base[0]) - (m5 ? 0 : 2),
                          ov, n, out, len))
      return false;

   if (!m5)
      out[6] |= (x >> 7) & 1;
   return true;
}

/* Apple's G17P cube path first runs one CEM calculation for each axis.  The
 * three source fields are the input XYZ half-register indices; axis 2 is the
 * last consumer in the witnessed sequence and carries their discard bits.
 *
 * f_plain from the native cube corpus, with x/y/z in 4/6/20, is exactly:
 *
 *   axis 0  1701540003040d5254219200
 *   axis 1  1701540703040d5254219204
 *   axis 2  1701540203040c50502f9208
 */
/* AGX3B witnesses independent per-coordinate discard hints on the final
 * axis. On AGX3C only uniform hints are witnessed, and a C comparison
 * reference aliasing a coordinate must survive CEM until its COPY. */
static bool
cube_cem_calc_to(enum agx_isa_encoding enc, uint16_t x, uint16_t y,
                 uint16_t z, uint16_t dest, uint8_t axis,
                 const bool discard[3], uint16_t reference,
                 uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || !discard)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       x >= 128 || y >= 128 || z >= 128 || (dest & 1) ||
       dest > 190 || axis > 2 ||
       ((discard[0] || discard[1] || discard[2]) && axis != 2) ||
       (enc == AGX_ISA_ENCODING_AGX3C &&
        (discard[0] != discard[1] || discard[1] != discard[2])))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2),  KV(10, 2), KV(11, 1), KV(23, 2), KV(24, 2),
      KV(40, 2), KV(41, 2), KV(57, 2), KV(58, 2),
   };
   struct agx3_field_kv ov[13];
   unsigned n = 0;
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      /* Axis zero writes the packed magnitude/face pair. */
      ov[n++] = (struct agx3_field_kv)KV(10, axis == 0 ? 1 : 2);
      ov[n++] = (struct agx3_field_kv)KV(65, 3);
      ov[n++] = (struct agx3_field_kv)KV(66, 24);
   }
   ov[n++] = (struct agx3_field_kv)KV(16, x);
   ov[n++] = (struct agx3_field_kv)KV(33, y);
   ov[n++] = (struct agx3_field_kv)KV(50, z);
   ov[n++] = (struct agx3_field_kv)KV(75, axis);
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   const uint16_t source[3] = {x, y, z};
   static const uint16_t cache_field[3] = {25, 42, 59};
   static const uint16_t discard_field[3] = {27, 44, 61};
   for (unsigned i = 0; i < 3; ++i) {
      /* A C comparison publishes its reference after CEM. Preserve an
       * aliased coordinate until that COPY, including on the final axis. */
      const bool keep_reference = enc == AGX_ISA_ENCODING_AGX3C &&
                                  source[i] == reference;
      if (discard[i] && !keep_reference)
         ov[n++] = (struct agx3_field_kv)KV(discard_field[i], 1);
      else if (axis != 2 || keep_reference)
         ov[n++] = (struct agx3_field_kv)KV(cache_field[i], 1);
   }

   return agx3_bind_compose(enc, AGX3_MN_cem_calc, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

bool
agx3_tex_cube_cem_calc_hints_to(enum agx_isa_encoding enc, uint16_t x,
                                uint16_t y, uint16_t z, uint16_t dest,
                                uint8_t axis, const bool discard[3],
                                uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return cube_cem_calc_to(enc, x, y, z, dest, axis, discard, UINT16_MAX,
                           out, len);
}

bool
agx3_tex_cube_cem_calc_to(enum agx_isa_encoding enc, uint16_t x, uint16_t y,
                          uint16_t z, uint16_t dest, uint8_t axis, bool discard,
                          uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   const bool hints[3] = {discard, discard, discard};
   return cube_cem_calc_to(enc, x, y, z, dest, axis, hints, UINT16_MAX,
                           out, len);
}

bool
agx3_tex_cube_cem_calc_reference_to(
   enum agx_isa_encoding enc, uint16_t x, uint16_t y, uint16_t z,
   uint16_t dest, uint8_t axis, bool discard, uint16_t reference,
   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   const bool hints[3] = {discard, discard, discard};
   return enc == AGX_ISA_ENCODING_AGX3C &&
          cube_cem_calc_to(enc, x, y, z, dest, axis, hints, reference,
                           out, len);
}

bool
agx3_tex_cube_cem_calc(enum agx_isa_encoding enc, uint16_t x, uint16_t y,
                       uint16_t z, uint8_t axis, bool discard,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   const uint16_t dest[3] = {0, 14, 4};
   if (axis > 2)
      return false;
   return agx3_tex_cube_cem_calc_to(enc, x, y, z, dest[axis], axis, discard,
                                    out, len);
}

/* CEM axis 0 leaves the major-axis magnitude in its destination. Apple's cube
 * shaders apply an in-place reciprocal before calculating the two normalized
 * face coordinates. */
bool
agx3_tex_cube_cem_rcp_to(enum agx_isa_encoding enc, uint16_t value,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   return agx3_alu_f32rcp(enc, value, value, AGX3_ALU_HINT_DISCARD, true, 0,
                          out, len);
}

bool
agx3_tex_cube_cem_rcp(enum agx_isa_encoding enc,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_cube_cem_rcp_to(enc, 0, out, len);
}

/* CEM leaves the signed face coordinates in its axis destinations. Apple's
 * native cube shaders finish them with this special FMA form: f35 names the
 * reciprocal major axis, f18 names the selected axis value and f12 groups the
 * result with the face PACK_SMPIR transaction. */
static bool
agx3_tex_cube_cem_finalize_at(enum agx_isa_encoding enc, uint8_t component,
                              uint8_t bank,
                              uint16_t reciprocal, uint16_t axis_value,
                              uint8_t slot,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   /* `bank` is f5: agx3_tex_cube_cem_finalize_to() keeps component * 2 (the
    * second component's baseline), the G18G layer names its own block. */
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       component > 1 || bank > 6 || (bank & 1) ||
       (reciprocal & 1) || reciprocal > 190 ||
       (axis_value & 1) || axis_value > 190 || slot < 1 ||
       slot > (enc == AGX_ISA_ENCODING_AGX3C ? 8 : 6))
      return false;

   const struct agx3_field_kv base0[] = {
      KV(9, 5),   KV(10, 2), KV(14, 1), KV(25, 2), KV(26, 2),
      KV(29, 1),  KV(42, 2), KV(43, 2), KV(44, 1), KV(47, 1),
      KV(51, 32), KV(59, 1), KV(60, 2), KV(77, 2), KV(91, 2),
   };
   const struct agx3_field_kv base1[] = {
      KV(5, 2),  KV(9, 5),   KV(10, 2), KV(14, 1), KV(25, 2),
      KV(26, 2), KV(29, 1),  KV(42, 2), KV(43, 2), KV(46, 1),
      KV(47, 1), KV(51, 32), KV(59, 1), KV(60, 2), KV(77, 2),
      KV(91, 2),
   };
   const struct agx3_field_kv ov[] = {
      KV(5, bank), KV(12, slot), KV(18, axis_value), KV(35, reciprocal),
      KV(95, 1), KV(96, 55),
   };
   return agx3_bind_compose(
      enc, AGX3_MN_f32fma, 5, component ? base1 : base0,
      component ? sizeof(base1) / sizeof(base1[0])
                : sizeof(base0) / sizeof(base0[0]),
      ov, enc == AGX_ISA_ENCODING_AGX3C ? 6 : 4, out, len);
}

bool
agx3_tex_cube_cem_finalize_to(enum agx_isa_encoding enc, uint8_t component,
                              uint16_t reciprocal, uint16_t axis_value,
                              uint8_t slot,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_cube_cem_finalize_at(enc, component, component * 2,
                                       reciprocal, axis_value, slot, out, len);
}

bool
agx3_tex_cube_cem_finalize_g18g(enum agx_isa_encoding enc, uint8_t component,
                               bool cube_array, uint16_t reciprocal,
                               uint16_t axis_value, uint8_t slot,
                               uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3C)
      return false;
   /* Native G18: cube X/reference 6/4, cube-array X/reference 4/6;
    * both put Y at zero and face/layer/level at half 2. */
   return agx3_tex_cube_cem_finalize_at(enc, component,
      component ? 0 : cube_array ? 4 : 6,
      reciprocal, axis_value, slot, out, len);
}

bool
agx3_tex_cube_cem_finalize(enum agx_isa_encoding enc, uint8_t component,
                           bool discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (component > 1 || (component == 0 && discard))
      return false;
   const uint16_t axis_value[2] = {14, 4};
   if (!agx3_tex_cube_cem_finalize_to(enc, component, 0,
                                      axis_value[component], 1, out, len))
      return false;
   if (discard && enc == AGX_ISA_ENCODING_AGX3B)
      out[10] |= 0x60;
   return true;
}

/* CEM publishes the selected face through the FLOAT cube form of PACK_SMPIR.
 * For discarded special r1 on slot 1 Apple emits exactly
 * 1705540600040004f0c00000; only f74 differs from the array-layer form. */
static bool
agx3_tex_stage_cube_z_at(enum agx_isa_encoding enc, uint8_t bank, uint16_t src,
                      uint8_t slot, uint16_t wait_mask, bool src_cache,
                      bool src_discard,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) ||
       (bank != 6 && !(gpu1 && bank == 2)) ||
       src >= (gpu1 ? 128 : 256) || (src & 1) ||
       slot < 1 || slot > slots || (wait_mask & ~((1u << slots) - 1)) ||
       !one_or_zero(wait_mask) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, bank), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 10), KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                         gpu1 ? 11 : 9, ov, n, out, len))
      return false;

   if (!gpu1)
      out[6] |= (src >> 7) & 1;
   return true;
}

bool
agx3_tex_stage_cube_z(enum agx_isa_encoding enc, uint16_t src,
                      uint8_t slot, uint16_t wait_mask, bool src_cache,
                      bool src_discard,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_cube_z_at(enc, 6, src, slot, wait_mask,
                                  src_cache, src_discard, out, len);
}

bool
agx3_tex_stage_cube_z_g18g(enum agx_isa_encoding enc, uint16_t src,
                           uint8_t slot, uint16_t wait_mask, bool src_cache,
                           bool src_discard,
                           uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3C)
      return false;
   return agx3_tex_stage_cube_z_at(enc, 2, src, slot, wait_mask,
                                  src_cache, src_discard, out, len);
}

/* A cube-array sample with a GPR layer publishes the integer layer and CEM's
 * face in one PACK_SMPIR form: operand 1 (f16) is the 16-bit layer half with
 * type f74=4 at bit 0, operand 2 (f30) the face half with type f76=10 at bit
 * 12. Both name half-registers, as every PACK_SMPIR operand does. metal-nt
 * -arch applegpu_g17p texturecube_array::sample at a buffer-loaded layer:
 *
 *   compute, flt2int to 16-bit half11   1705540600162800f0c02900
 *   fragment, layer in r0, face half2   1705540600000800f0c02900
 *
 * applegpu-nt depthcube_array::sample_compare emits the same form for layer
 * half3, 1705540600060800f0c02900.
 */
bool
agx3_tex_stage_cube_array_layer(enum agx_isa_encoding enc, uint16_t face,
                                uint16_t layer, uint8_t slot,
                                uint16_t wait_mask, bool layer_cache,
                                bool layer_discard,
                                uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || face >= 128 || (face & 1) ||
       layer >= 128 || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) || !one_or_zero(wait_mask) ||
       (layer_cache && layer_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6),  KV(9, 5),  KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 2), KV(38, 1), KV(41, 1), KV(51, 1), KV(52, 2),
      KV(74, 4), KV(76, 10), KV(77, 12),
   };
   struct agx3_field_kv ov[18];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(30, face);
   ov[n++] = (struct agx3_field_kv)KV(16, layer);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, layer_discard);
   if (gpu1) {
      ov[n++] = (struct agx3_field_kv)KV(56, 1);
      ov[n++] = (struct agx3_field_kv)KV(57, 119);
   }
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* A cube sample with a register level or bias publishes CEM's face and the
 * 16-bit float LOD in one PACK_SMPIR: the face as operand 1 (type f74=10 at
 * bit 0) and the LOD as operand 2 (type f76=2 at bit 16). SMP f139 alone
 * distinguishes level from bias. metal-nt -arch applegpu_g17p texturecube:
 *
 *   compute  sample(s, c.xyz, level(c.w))  face half10, LOD half11
 *            1705540600142c00f0c01100      05800cb8b0000003000010000100
 *   fragment sample(s, dir, level(b[i]))   face half2, LOD half3
 *            1705540600040c00f0c01100      05800cb8b0000003010010000100
 *   fragment sample(s, dir, bias(b[i]))    same PACK_SMPIR
 *                                          05800cb8b0001400010010000100
 *
 * Publishing the face and the LOD with two PACK_SMPIRs would replace the
 * first bank word with the second.
 */
bool
agx3_tex_stage_cube_lod(enum agx_isa_encoding enc, uint16_t face,
                        uint16_t lod, uint8_t slot, uint16_t wait_mask,
                        bool lod_cache, bool lod_discard,
                        uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) ||
       face >= (gpu1 ? 128 : 256) || (face & 1) ||
       lod >= 128 || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) ||
       !one_or_zero(wait_mask) || (lod_cache && lod_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6),  KV(9, 5),  KV(10, 2), KV(23, 2), KV(24, 1),
      KV(27, 1), KV(37, 2), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(74, 10), KV(76, (unsigned)AGX3_TEX_LEVEL_FLOAT), KV(77, 16),
      KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[13];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, face & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(30, lod);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(39, lod_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, lod_discard);
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                         gpu1 ? 15 : 13, ov, n, out, len))
      return false;

   /* f16 bit 7, as for the face-only form. */
   if (!gpu1)
      out[6] |= (face >> 7) & 1;
   return true;
}

/* A cube sample with a bias and a minimum LOD fills both typed bank-6
 * operands with them, so Apple packs CEM's face alone into a GPR first and
 * names it as untyped operand 3, as the cube array names its layer.
 * metal-nt -arch applegpu_g17p, texturecube bias(b) + min_lod_clamp(m):
 *
 *   1705540003100004f0880400   face half8 -> r0
 *   1705540600202400f0c22300   minimum half16, bias half9, r0
 *
 * and the depthcube comparison packs face half10 as 1705540003140004f0880400
 * and minimum half3 with bias half2 as 1705540600060800f0c22300. SMP then
 * selects f139=3 and f140=2 on the plain cube dimension. */
bool
agx3_tex_pack_cube_face(enum agx_isa_encoding enc, uint16_t face,
                        uint16_t dest, uint8_t out[AGX3_TEX_MAX_LENGTH],
                        uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) ||
       face >= (gpu1 ? 128 : 256) || (face & 1) ||
       dest > (gpu1 ? 254 : 190) || (dest & 1))
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2, dest), KV(9, 2), KV(10, 2), KV(11, 1), KV(16, face & 0x7f),
      KV(23, 2), KV(24, 1), KV(27, 1), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(74, 10), KV(75, 12),
   };
   const struct agx3_field_kv c_fields[] = {KV(56, 3), KV(57, 119)};
   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                          sizeof(fields) / sizeof(fields[0]), c_fields,
                          gpu1 ? 2 : 0, out, len))
      return false;

   /* f16 bit 7, as for the face-only staging form. */
   if (!gpu1)
      out[6] |= (face >> 7) & 1;
   return true;
}

bool
agx3_tex_stage_cube_bias_min(enum agx_isa_encoding enc, uint16_t bias,
                             uint16_t minimum, uint16_t packed_face,
                             uint8_t slot, uint16_t wait_mask, bool src_cache,
                             bool src_discard,
                             uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || minimum >= 128 ||
       bias >= (gpu1 ? 128 : 256) ||
       packed_face > 190 || (packed_face & 1) || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) || !one_or_zero(wait_mask) ||
       (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1), KV(37, 2),
      KV(38, 1), KV(51, 2), KV(52, 2), KV(55, 1),
      KV(74, (unsigned)AGX3_TEX_LEVEL_MINIMUM),
      KV(76, (unsigned)AGX3_TEX_LEVEL_FLOAT), KV(77, 16),
   };
   struct agx3_field_kv ov[18];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, minimum);
   ov[n++] = (struct agx3_field_kv)KV(30, bias);
   ov[n++] = (struct agx3_field_kv)KV(44, packed_face);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   ov[n++] = (struct agx3_field_kv)KV(39, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, src_discard);
   if (gpu1) {
      ov[n++] = (struct agx3_field_kv)KV(56, 1);
      ov[n++] = (struct agx3_field_kv)KV(57, 119);
   }
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* A cube-array sample with a register level or bias needs three typed
 * operands, but PACK_SMPIR types only the first two. Apple first packs the
 * layer alone into a 32-bit GPR (agx3_tex_pack_ms_array_layer) and then
 * publishes the face (f74=10 at bit 12), the float LOD (f76=2 at bit 16) and
 * that prepacked word as untyped operand 3 (f44, f51=2, f52=2). metal-nt
 * -arch applegpu_g17p texturecube_array:
 *
 *   compute  sample(s, c.xyz, uint(c.w), level(c.x))
 *            17055400031c0000f0880400  layer half14 -> r0
 *            1705540600142c00f0c21700  face half10, LOD half11, r0
 *   compute  sample(s, c.xyz, l[i], level(lod[i]))
 *            1705540003000000f0880400  layer half0 -> r0
 *            1705540600102400f0c21700  face half8, LOD half9, r0
 *   fragment sample(s, dir, l[i], level(b[j]))
 *            1705540003000000f0880400  layer half0 -> r0
 *            1705540600040c00f0c21700  face half2, LOD half3, r0
 *
 * SMP selects the staged layer with f77=2 and the LOD mode with f139=4 for a
 * level (05800cb8b0000803010010000100) or 3 for a bias
 * (05800cb8b0001800010010000100), as for 2D samples.
 */
bool
agx3_tex_stage_cube_array_lod(enum agx_isa_encoding enc, uint16_t face,
                              uint16_t lod, uint16_t packed_layer,
                              uint8_t slot, uint16_t wait_mask,
                              bool lod_cache, bool lod_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || face >= 128 || (face & 1) ||
       lod >= 128 || packed_layer > 190 || (packed_layer & 1) || slot < 1 ||
       slot > slots || (wait_mask & ~((1u << slots) - 1)) || !one_or_zero(wait_mask) ||
       (lod_cache && lod_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 6),   KV(9, 5),   KV(10, 2),  KV(23, 2),  KV(24, 1),
      KV(27, 1),  KV(37, 2),  KV(38, 1),  KV(51, 2),  KV(52, 2),
      KV(55, 1),  KV(74, 10), KV(75, 12),
      KV(76, (unsigned)AGX3_TEX_LEVEL_FLOAT), KV(77, 16),
   };
   struct agx3_field_kv ov[18];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, face);
   ov[n++] = (struct agx3_field_kv)KV(30, lod);
   ov[n++] = (struct agx3_field_kv)KV(44, packed_layer);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(39, lod_cache);
   ov[n++] = (struct agx3_field_kv)KV(41, lod_discard);
   if (gpu1) {
      ov[n++] = (struct agx3_field_kv)KV(56, 1);
      ov[n++] = (struct agx3_field_kv)KV(57, 119);
   }
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* Native runtime MS-array reads normalize the layer to a GPR, then merge
 * that word with the sample at staging half4. NIR provides sample in low16
 * and layer in high16, so both half addresses must be consumed separately.
 * Fresh native runtime witnesses: layer half6 -> packed half2 is
 * 17055402030c0000f0880400; sample half0 plus packed half2 is
 * 1705540400000010d0c20a00. */
/* depth2d_array sample_compare(s, uv, layer, ref, bias(b), min_lod_clamp(m))
 * needs half 4 for the reference and bank 6 for the layer and bias, so the
 * minimum goes to a GPR, padded with an integer zero at bit 16, and SMP names
 * that register. metal-nt -arch applegpu_g17p, two such samples in one
 * fragment:
 *
 *   1705540002000000d0886000   minimum h0 -> r0
 *   1705540202040000d0886000   minimum h2 -> r2 (half 2)
 *   05800ca090004200010000080100   SMP f37=2, f30=0, f41=1
 *   0d840ca090244210010000080100   SMP f37=2, f30=2, f41=1
 */
bool
agx3_tex_pack_float_lod_m5(uint16_t src, uint16_t dest,
                          bool cache, bool discard, uint16_t wait_mask,
                          uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (src >= 128 || dest >= 256 || (dest & 1) ||
       (cache && discard) || (wait_mask & ~0xffu))
      return false;

   /* Native shared_level/shared_bias return the packed parameter in a GPR,
    * f9=2/f11=1 and f56=3, instead of publishing a staging result slot. */
   const struct agx3_field_kv fields[] = {
      KV(2, dest), KV(9, 2), KV(10, 2), KV(11, 1),
      KV(16, src), KV(23, 2), KV(24, 1),
      KV(25, cache), KV(27, discard), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(56, 3), KV(57, 119), KV(74, AGX3_TEX_LEVEL_FLOAT), KV(75, 16),
   };
   struct agx3_field_kv waits[8];
   unsigned n = 0;
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         waits[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_pack_smpir, 0,
                            fields, sizeof(fields) / sizeof(fields[0]),
                            waits, n, out, len);
}

bool
agx3_tex_pack_minimum(enum agx_isa_encoding enc, uint16_t src, uint16_t dest,
                      bool cache, bool discard,
                      uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 128 || dest > 190 ||
       (dest & 1) || (cache && discard))
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2, dest), KV(9, 2), KV(10, 2), KV(16, src), KV(23, 2), KV(24, 1),
      KV(25, cache), KV(27, discard), KV(37, 1), KV(38, 1), KV(51, 1),
      KV(52, 2), KV(74, (unsigned)AGX3_TEX_LEVEL_MINIMUM),
      KV(76, (unsigned)AGX3_TEX_LEVEL_INTEGER), KV(77, 16),
   };
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                           sizeof(fields) / sizeof(fields[0]), NULL, 0, out, len);
}

bool
agx3_tex_stage_minimum_bank8_m5(uint16_t src, uint8_t slot, uint16_t wait_mask,
                                bool cache, bool discard,
                                uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (src >= 128 || slot < 1 || slot > 8 || (wait_mask & ~0xffu) ||
       (cache && discard))
      return false;
   /* Native G17G comparison/3D minimums with a register bias (port/texwit8
    * ca_br_min_{d,k,n}, m3_br_min_d): like G17P's minimum pack (minimum
    * f74 = 1 at 0 beside an immediate at 16), but the immediate's format is
    * f76 = 2 (G17P: 3), written to bank 8 instead of a GPR the SMP names:
    *   2702002a2500aa1200004a301800  bank8 <- min(^h6), slot 1 */
   const struct agx3_field_kv base[] = {
      KV(5, 8), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 1), KV(38, 1), KV(51, 1), KV(52, 2), KV(56, 1), KV(57, 119),
      KV(74, (unsigned)AGX3_TEX_LEVEL_MINIMUM), KV(76, 2), KV(77, 16),
   };
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, cache);
   ov[n++] = (struct agx3_field_kv)KV(27, discard);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

bool
agx3_tex_pack_ms_array_layer(enum agx_isa_encoding enc, uint16_t src,
                            uint16_t dest, bool cache, bool discard,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       src >= 128 || dest > (enc == AGX_ISA_ENCODING_AGX3C ? 254 : 190) ||
       (dest & 1) || (cache && discard))
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2, dest), KV(9, 2), KV(10, 2), KV(11, 1),
      KV(16, src), KV(23, 2), KV(24, 1), KV(25, cache), KV(27, discard),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 4),
   };
   const struct agx3_field_kv c_fields[] = {KV(56, 3), KV(57, 119)};
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                           sizeof(fields) / sizeof(fields[0]), c_fields,
                           enc == AGX_ISA_ENCODING_AGX3C ? 2 : 0, out, len);
}

bool
agx3_tex_stage_ms_array_sample(enum agx_isa_encoding enc, uint16_t sample,
                              uint16_t packed_layer, bool cache, bool discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || sample >= 128 || (sample & 1) ||
       packed_layer > 190 || (packed_layer & 1) || (cache && discard))
      return false;
   const struct agx3_field_kv fields[] = {
      KV(5, 4), KV(9, 5), KV(10, 2), KV(12, 1),
      KV(16, sample), KV(23, 2), KV(24, 1), KV(25, cache), KV(27, discard),
      KV(37, 1), KV(38, 1), KV(44, packed_layer), KV(51, 2),
      KV(52, 2), KV(55, 1), KV(74, 9), KV(75, 12), KV(76, 1), KV(77, 16),
   };
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                           sizeof(fields) / sizeof(fields[0]), NULL, 0, out, len);
}

/* G17G arraydynamic/arrayindexed merge a normalized layer word with the
 * sample half before SMP.ep2. The two native publishers are
 * 2700003a2d00aa0a000040308100 and2702002a2500aa0a00004a400c00. */
bool
agx3_tex_stage_ms_array_sample_at(
   enum agx_isa_encoding enc, uint16_t bank, uint16_t sample,
   uint16_t packed_layer, uint8_t slot, bool cache, bool discard,
   uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3B && bank == 0)
      return agx3_tex_stage_ms_array_sample(enc, sample, packed_layer,
                                             cache, discard, out, len);
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || (bank & 7) || bank > 24 ||
       sample >= 128 || (sample & 1) || packed_layer > 254 ||
       (packed_layer & 1) || slot < 1 || slot > 8 || (cache && discard))
      return false;
   const struct agx3_field_kv fields[] = {
      KV(5, bank + 4), KV(9, 5), KV(10, 2), KV(12, slot),
      KV(16, sample), KV(23, 2), KV(24, 1), KV(25, cache), KV(27, discard),
      KV(37, 1), KV(38, 1), KV(44, packed_layer), KV(51, 2),
      KV(52, 2), KV(55, 1), KV(56, 1), KV(57, 119),
      KV(74, 9), KV(75, 12), KV(76, 1), KV(77, 16),
   };
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                            sizeof(fields) / sizeof(fields[0]), NULL, 0, out, len);
}

/* Native texture2d_ms reads publish the sample at fixed staging half4.
 * Field24=1 reads the low16 of the sample producer. Field16 uses the same
 * half-register address as the producer, without converting it to words.
 * The native discarded-half2 witness is 1705540400040000d0c00800. */
/* G18 native multisample reads take a 32-bit sample index and optionally
 * OR a separately normalized layer word into staging half 2. The packed
 * NIR sample/layer source must be zero-extended before this publication. */
bool
agx3_tex_stage_ms_g18g(enum agx_isa_encoding enc, uint16_t sample,
                       uint16_t packed_layer, bool array, uint8_t slot,
                       bool cache, bool discard,
                       uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || sample >= 128 || (sample & 1) ||
       (array && (packed_layer >= 128 || (packed_layer & 1))) ||
       slot < 1 || slot > 8 || (cache && discard))
      return false;
   const struct agx3_field_kv fields[] = {
      KV(5, 2), KV(9, 5), KV(10, 2), KV(12, slot),
      /* The G17 intermediate accepts a half source; G18 transcoding widens
       * this operand after the caller has zero-extended the sample. */
      KV(16, sample), KV(23, 2), KV(24, 1), KV(25, cache), KV(27, discard),
      KV(37, 1), KV(38, 1), KV(44, array ? packed_layer : 0),
      KV(51, array ? 2 : 1), KV(52, 2), KV(55, array), KV(56, 1), KV(57, 119),
      KV(74, 9), KV(75, 12), KV(76, 1), KV(77, 16),
   };
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
      sizeof(fields) / sizeof(fields[0]), NULL, 0, out, len);
}

bool
agx3_tex_stage_ms_sample(enum agx_isa_encoding enc, uint16_t src,
                         uint16_t wait_mask, bool src_cache,
                         bool src_discard,
                         uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || (src & 1) ||
       (wait_mask & ~0x3f) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 4),  KV(9, 5),  KV(10, 2), KV(12, 1), KV(23, 2), KV(24, 1), KV(37, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, 9), KV(75, 12),
      KV(76, 1), KV(77, 16),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* G17G dynamicreg publishes the sample's low half to bank4 with
 * 2700002a2500aa0a00004a300800. Its per-sample fragment uses the same
 * publisher at banks4/12/20/28 beside four independent coordinate banks. */
bool
agx3_tex_stage_ms_sample_at(enum agx_isa_encoding enc, uint16_t bank,
                            uint16_t src, uint8_t slot, uint16_t wait_mask,
                            bool src_cache, bool src_discard,
                            uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3B && bank == 0)
      return agx3_tex_stage_ms_sample(enc, src, wait_mask, src_cache,
                                       src_discard, out, len);
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || (bank & 7) || bank > 120 ||
       src >= 256 || (src & 1) || slot < 1 || slot > 8 ||
       (wait_mask & ~0xffu) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv fields[] = {
      KV(5, bank + 4), KV(9, 5), KV(10, 2), KV(12, slot),
      KV(16, src), KV(23, 2), KV(24, 1),
      KV(25, src_cache), KV(27, src_discard),
      KV(37, 1), KV(38, 1), KV(51, 1), KV(52, 2),
      KV(56, 1), KV(57, 119), KV(74, 9), KV(75, 12), KV(76, 1), KV(77, 16),
   };
   struct agx3_field_kv waits[8];
   unsigned n = 0;
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         waits[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, fields,
                            sizeof(fields) / sizeof(fields[0]), waits, n,
                            out, len);
}

/* Fresh native G17P LOD fragment witnesses use this 12-byte PACK_SMPIR for a
 * scalar LOD, preserving the source when it remains live. f5=6 selects
 * operand bank 6, not architectural r3. The FLOAT source is a half, while the
 * INTEGER source is the full uint required by texture2d::read. */
bool
agx3_tex_stage_typed_lod_bank(enum agx_isa_encoding enc, uint16_t bank,
                              uint16_t src, uint8_t slot, uint16_t wait_mask,
                              bool src_cache, bool src_discard,
                              enum agx3_tex_level_kind kind,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || src >= 256 || slot < 1 || slot > 6 ||
       (wait_mask & ~0x3f) || (src_cache && src_discard) ||
       (bank & 7) != 6 || bank > 22 ||
       (kind != AGX3_TEX_LEVEL_MINIMUM && kind != AGX3_TEX_LEVEL_FLOAT &&
        kind != AGX3_TEX_LEVEL_INTEGER))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, bank), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(38, 1), KV(51, 1), KV(52, 2), KV(74, (unsigned)kind), KV(75, 16),
   };
   struct agx3_field_kv ov[10];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, src & 0x7f);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, src_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, src_discard);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);
   if (!agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
                         sizeof(base) / sizeof(base[0]), ov, n, out, len))
      return false;

   /* The source is an eight-bit half-register index (f16 bits 0..6 at
    * instruction bits 41..47, bit 7 at bit 48). Apple's encoder's 191
    * bound is a compiler allocation policy, not a narrower wire field.
    * Keep the same native instruction for Mesa's full 256-half bank. */
   out[6] |= (src >> 7) & 1;
   return true;
}


/* depthcube_array::sample_compare combines CEM's selected face and the
 * integer array layer in one PACK_SMPIR form. applegpu-nt 32023.917 emits
 * exactly 1705540600060800f0c02900 for discarded 32-bit r3 on slot 1.
 *
 * Unlike the ordinary scalar PACK forms, f16 names a full 32-bit register,
 * not a half-register. The five additional selectors below are the exact
 * delta which selects the combined face/layer operand in Apple's table.
 */
static bool
agx3_tex_stage_cube_array_shadow_at(enum agx_isa_encoding enc, uint8_t bank,
                                 uint16_t layer,
                                 uint8_t slot, uint16_t wait_mask,
                                 bool layer_cache, bool layer_discard,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH],
                                 uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || layer >= 256 ||
       (bank != 6 && !(gpu1 && bank == 2)) ||
       (layer & 1) || slot < 1 || slot > slots ||
       (wait_mask & ~((1u << slots) - 1)) ||
       !one_or_zero(wait_mask) || (layer_cache && layer_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, bank), KV(9, 5),  KV(10, 2), KV(23, 2), KV(24, 1),
      KV(30, 2), KV(37, 2), KV(38, 1), KV(41, 1), KV(51, 1),
      KV(52, 2), KV(74, 4), KV(76, 10), KV(77, 12), KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(16, layer >> 1);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(25, layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, layer_discard);
   for (unsigned i = 0; i < slots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base, gpu1 ? 16 : 14,
                            ov, n, out, len);
}

bool
agx3_tex_stage_cube_array_shadow(enum agx_isa_encoding enc, uint16_t layer,
                                 uint8_t slot, uint16_t wait_mask,
                                 bool layer_cache, bool layer_discard,
                                 uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_cube_array_shadow_at(enc, 6, layer, slot, wait_mask,
                                             layer_cache, layer_discard, out, len);
}

/* G18 cube-array PACK_SMPIR names independent half-registers. Implicit
 * sampling packs raw layer/face; register LOD packs face/LOD/normalized
 * layer. Native vertex and fragment references use face/layer pairs 10/8
 * and 2/0 respectively. None of these register numbers is an ABI constant. */
bool
agx3_tex_stage_cube_array_g18g(enum agx_isa_encoding enc,
                              uint16_t face, uint16_t layer, uint16_t lod,
                              bool register_lod, uint8_t slot,
                              uint16_t wait_mask, bool layer_cache,
                              bool layer_discard, bool lod_cache,
                              bool lod_discard,
                              uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C || face >= 128 || (face & 1) ||
       layer >= 128 || (register_lod && ((layer & 1) || lod >= 128)) ||
       slot < 1 || slot > 8 || (wait_mask & ~0xffu) ||
       !one_or_zero(wait_mask) || (layer_cache && layer_discard) ||
       (lod_cache && lod_discard))
      return false;

   const struct agx3_field_kv base[] = {
      KV(5, 2), KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 1),
      KV(37, 2), KV(38, 1), KV(52, 2), KV(56, 1), KV(57, 119),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(16, register_lod ? face : layer);
   ov[n++] = (struct agx3_field_kv)KV(25, register_lod ? false : layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, register_lod ? true : layer_discard);
   ov[n++] = (struct agx3_field_kv)KV(30, register_lod ? lod : face);
   ov[n++] = (struct agx3_field_kv)KV(39, register_lod ? lod_cache : false);
   ov[n++] = (struct agx3_field_kv)KV(41, register_lod ? lod_discard : true);
   ov[n++] = (struct agx3_field_kv)KV(44, register_lod ? layer : 0);
   ov[n++] = (struct agx3_field_kv)KV(51, register_lod ? 2 : 1);
   ov[n++] = (struct agx3_field_kv)KV(53, register_lod && layer_cache);
   ov[n++] = (struct agx3_field_kv)KV(55, register_lod && layer_discard);
   ov[n++] = (struct agx3_field_kv)KV(74, register_lod ? 10 : 4);
   ov[n++] = (struct agx3_field_kv)KV(75, register_lod ? 12 : 0);
   ov[n++] = (struct agx3_field_kv)KV(76, register_lod ? 2 : 10);
   ov[n++] = (struct agx3_field_kv)KV(77, register_lod ? 16 : 12);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(58 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_pack_smpir, 0, base,
      sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

bool
agx3_tex_stage_array_integer_lod(
   enum agx_isa_encoding enc, uint16_t layer, uint16_t lod, uint8_t slot,
   uint16_t wait_mask, bool layer_cache, bool layer_discard, bool lod_cache,
   bool lod_discard, uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   return agx3_tex_stage_array_integer_lod_at(
      enc, 0, layer, lod, slot, wait_mask, layer_cache, layer_discard,
      lod_cache, lod_discard, out, len);
}

bool
agx3_tex_encode_image_write_agx3c(
   const struct agx3_image_write_desc *d, uint16_t uniform_base,
   uint8_t staging_slot, uint8_t out[AGX3_TEX_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_TEX_MAX_LENGTH);
   *len = 0;
   if (!d)
      return false;

   const unsigned kind = d->image_kind ?: AGX3_TEX_OPERAND_IMMEDIATE;
   const bool immediate = kind == AGX3_TEX_OPERAND_IMMEDIATE;
   const bool reg = kind == AGX3_TEX_OPERAND_REGISTER;
   const unsigned result_slot = d->result_slot ?: 1;
   if ((uniform_base & 3) || uniform_base > 252 ||
       staging_slot < 1 || staging_slot > 8 || result_slot > 8 ||
       (d->wait_mask & ~UINT16_C(0xff)) ||
       (kind != AGX3_TEX_OPERAND_IMMEDIATE && !reg &&
        kind != AGX3_TEX_OPERAND_UNIFORM) ||
       (immediate ? d->image_value > 15
                  : ((d->image_value & 1) || d->image_value > (reg ? 126 : 254))) ||
       (d->image_kill && !reg) ||
       (d->staging_bank & 15) || d->staging_bank > 240 ||
       (d->dim != AGX3_IMAGE_WRITE_2D &&
        d->dim != AGX3_IMAGE_WRITE_2D_ARRAY &&
        d->dim != AGX3_IMAGE_WRITE_3D))
      return false;

   /* Native compute-write corpus: float/uint differ only in f70; a
    * descriptor-set heap at uniform4 sets f79 independently of its index.
    * writes8/writes12 witness banks0..176 and result slots beyond four.
    * reuse_index clears f103 until the index's final consumer. */
   const struct agx3_field_kv fields[] = {
      KV(5, d->staging_bank), KV(9, 5), KV(10, 2), KV(12, result_slot),
      KV(13, 1), KV(24, 1), KV(38, 1), KV(51, 1), KV(52, 1),
      KV(56, 2), KV(58, 87), KV(58 + staging_slot, 1),
      KV(70, d->normalized), KV(71, 2),
      KV(72, d->dim == AGX3_IMAGE_WRITE_3D ? 3 : 1),
      KV(73, d->dim == AGX3_IMAGE_WRITE_2D_ARRAY ? 2 : 0), KV(74, 1),
      KV(79, uniform_base), KV(85, 3), KV(86, 3),
      KV(immediate ? 91 : reg ? 92 : 93, d->image_value),
      KV(99, immediate ? 1 : reg ? 2 : 3), KV(100, immediate ? 1 : 2),
      KV(103, d->image_kill), KV(113, 1),
   };
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_imgwci, 0,
                          fields, sizeof(fields) / sizeof(fields[0]),
                          NULL, 0, out, len) || *len != 18) {
      memset(out, 0, AGX3_TEX_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}
