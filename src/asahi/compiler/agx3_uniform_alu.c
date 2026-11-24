/* SPDX-License-Identifier: MIT */
#include "agx3_uniform_alu.h"
#include <string.h>
#include "agx3_binder.h"

static bool
uniform_float_encode(enum agx_isa_encoding encoding,
                     enum agx3_uniform_float_op op, uint16_t dest,
                     const struct agx3_uniform_float_src src[3], bool saturate,
                     bool dest_cache, uint8_t wait_mask, bool half,
                     bool uniform_dest,
                     uint8_t output[16], uint8_t *length)
{
   const bool m5 = encoding == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && encoding != AGX_ISA_ENCODING_AGX3B) || !src || !output || !length ||
       (unsigned)op > AGX3_UNIFORM_FMA || dest >= 256 ||
       (!half && (dest & 1)) || wait_mask > (m5 ? 0xff : 0x3f))
      return false;

   /* Retained native G17G shaders witness, for a GPR result:
    *   MUL R*U and U*R (also saturated);
    *   ADD R+U and U+R, the uniform optionally negated;
    *   FMA R*U+R, R*U+U and U*R+U, and R*U+imm, the uniform multiplicand
    *   optionally negated and a register multiplicand negated in U*R+U;
    *   FMA R*U-R with plain multiplicands (a native compiler probe of
    *   fma(x, u, -y): 0993060511a00202);
    *   FMA R*R+U with nothing negated (the agx3-diff-corpus fragments);
    * and for a uniform result in a constant program:
    *   FMA R*R+R and MUL imm*R.
    * Native Minecraft terrain/entity/sky pipelines compiled natively for
    * G17G (vertex constant programs: 48 U=RRR FMAs, fragment
    * constant programs: 12 U=IR MULs; mains: 39 ADDs, 29 U-addend FMAs and 5
    * R*U+imm FMAs) are asserted byte for byte in test-agx3-uniform-alu.cpp.
    * Half arithmetic stays outside this admission. */
   if (m5) {
      const unsigned count = op == AGX3_UNIFORM_FMA ? 3 : 2;
      unsigned regs = 0, uniforms = 0, immediates = 0;
      for (unsigned i = 0; i < count; ++i) {
         regs += src[i].kind == AGX3_FLOAT_REGISTER;
         uniforms += src[i].kind == AGX3_FLOAT_UNIFORM;
         immediates += src[i].kind == AGX3_FLOAT_IMMEDIATE;
         /* A 16-bit uniform multiplicand of an f32 FMA is witnessed (R*U+imm,
          * f43=1); other half operands stay outside. */
         if (src[i].abs ||
             (src[i].half && (src[i].kind != AGX3_FLOAT_UNIFORM ||
                              op != AGX3_UNIFORM_FMA || i == 2)) ||
             (src[i].kind == AGX3_FLOAT_REGISTER && src[i].value >= 252))
            return false;
      }
      if (half || dest >= 252 || (saturate && op != AGX3_UNIFORM_FMUL))
         return false;
      if (uniform_dest) {
         /* U=RRR FMA and U=IR MUL, unmodified. */
         if (op == AGX3_UNIFORM_FADD ||
             (op == AGX3_UNIFORM_FMA && regs != 3) ||
             (op == AGX3_UNIFORM_FMUL && (regs != 1 || immediates != 1)))
            return false;
         for (unsigned i = 0; i < count; ++i) {
            if (src[i].neg)
               return false;
         }
      } else {
         const bool u0 = src[0].kind == AGX3_FLOAT_UNIFORM;
         const bool u1 = src[1].kind == AGX3_FLOAT_UNIFORM;
         if (op != AGX3_UNIFORM_FMA) {
            /* One register and one uniform; only ADD negates (the
             * uniform). */
            if (regs != 1 || uniforms != 1)
               return false;
            for (unsigned i = 0; i < 2; ++i) {
               if (src[i].neg && (op != AGX3_UNIFORM_FADD ||
                                  src[i].kind != AGX3_FLOAT_UNIFORM))
                  return false;
            }
         } else if (!u0 && !u1) {
            /* R*R+U: register multiplicands and a plain uniform addend
             * (agx3-diff-corpus native G17G r_heavy_loop fragment). */
            if (src[0].kind != AGX3_FLOAT_REGISTER ||
                src[1].kind != AGX3_FLOAT_REGISTER ||
                src[2].kind != AGX3_FLOAT_UNIFORM || src[0].neg ||
                src[1].neg || src[2].neg)
               return false;
         } else {
            /* Exactly one uniform multiplicand and one register one. */
            if (u0 == u1 || (src[0].kind != AGX3_FLOAT_REGISTER &&
                             src[1].kind != AGX3_FLOAT_REGISTER))
               return false;
            switch (src[2].kind) {
            case AGX3_FLOAT_REGISTER:
               /* A negated register addend is witnessed beside a plain
                * register and a plain uniform multiplicand (R*U-R). */
               if (src[2].neg && (src[0].neg || src[1].neg))
                  return false;
               break;
            case AGX3_FLOAT_UNIFORM:
               if (src[2].neg || src[2].value >= 512)
                  return false;
               break;
            case AGX3_FLOAT_IMMEDIATE:
               break;
            default:
               return false;
            }
            for (unsigned i = 0; i < 2; ++i) {
               /* Negated register multiplicands are witnessed only with a
                * uniform addend. */
               if (src[i].neg && src[i].kind == AGX3_FLOAT_REGISTER &&
                   src[2].kind != AGX3_FLOAT_UNIFORM)
                  return false;
            }
         }
      }
   }

   /* Direct uniform destinations are witnessed in constant programs, with
    * aligned low-file f32 results and plain GPR/immediate sources. Keep the
    * independently qualified main-program source contract unchanged. */
   if (uniform_dest && (half || dest >= 128 || dest_cache || saturate))
      return false;

   const unsigned count = op == AGX3_UNIFORM_FMA ? 3 : 2;
   bool uniform = false, reg = false;
   unsigned half_uniforms = 0;
   for (unsigned i = 0; i < count; ++i) {
      const bool source_half = half || src[i].half;
      if (src[i].half && !half && src[i].kind != AGX3_FLOAT_UNIFORM)
         return false;
      if (src[i].half && !half && (++half_uniforms > 1 || saturate))
         return false;
      if (uniform_dest && (src[i].kind == AGX3_FLOAT_UNIFORM ||
                           src[i].abs || src[i].neg))
         return false;
      switch (src[i].kind) {
      case AGX3_FLOAT_UNIFORM:
         uniform = true;
         if (src[i].value >= 512 || (!source_half && (src[i].value & 1)) ||
             src[i].abs ||
             (src[i].neg &&
              (source_half ||
               (op != AGX3_UNIFORM_FMA && !(m5 && op == AGX3_UNIFORM_FADD)) ||
               (!m5 && i != 2))) ||
             src[i].cache || src[i].discard)
            return false;
         break;
      case AGX3_FLOAT_REGISTER:
         reg = true;
         if (src[i].value >= 256 || (!half && (src[i].value & 1)) ||
             (src[i].cache && src[i].discard))
            return false;
         break;
      case AGX3_FLOAT_IMMEDIATE:
         if (src[i].value > 255 || src[i].cache || src[i].discard)
            return false;
         break;
      default:
         return false;
      }
      /* The initial half-uniform corpus grounds plain operands only. */
      if (half && (src[i].abs || src[i].neg || saturate))
         return false;
   }
   /* Native main-program witnesses all retain at least one GPR operand.
    * All-uniform/immediate expressions keep one materialized input until
    * their port and constant-program contracts have their own witness. */
   if (!reg || (!uniform_dest && !uniform))
      return false;

   /* Apple G17P/G16G uniform_repeat: 09010e0d210e02c2. Domain 3 selects
    * uniform f36/f53 instead of register f35/f52. ADD and MUL use the same
    * first two operand records. */
   enum agx3_mnemonic mnemonic = half ? (op == AGX3_UNIFORM_FADD   ? AGX3_MN_f16add
                                  : op == AGX3_UNIFORM_FMUL ? AGX3_MN_f16mul
                                                            : AGX3_MN_f16fma)
                               : (op == AGX3_UNIFORM_FADD   ? AGX3_MN_f32add
                                  : op == AGX3_UNIFORM_FMUL ? AGX3_MN_f32mul
                                                            : AGX3_MN_f32fma);
   uint8_t best[16], best_length = 0;
   for (unsigned swap = 0; swap < 2; ++swap) {
      /* M5 places an FMA's uniform multiplicand in the second operand
       * unless the addend is a uniform too (U*R+U), and a uniform-result
       * MUL's immediate in the first. */
      if (m5 && op == AGX3_UNIFORM_FMA && !uniform_dest &&
          src[2].kind != AGX3_FLOAT_UNIFORM &&
          src[1 ^ swap].kind != AGX3_FLOAT_UNIFORM)
         continue;
      if (m5 && op == AGX3_UNIFORM_FMUL && uniform_dest &&
          src[swap].kind != AGX3_FLOAT_IMMEDIATE)
         continue;
      struct agx3_field_kv fields[48] = {
         {uniform_dest ? 3 : 2, dest},
         {9, uniform_dest ? 3 : 2}, {10, half ? 1 : 2},
         {11, dest_cache}, {14, saturate},
      };
      unsigned n = 5;
      for (unsigned i = 0; i < count; ++i) {
         const struct agx3_uniform_float_src *s = &src[i < 2 ? i ^ swap : i];
         const unsigned base = (half ? 16 : 17) + 17 * i;
         unsigned value = s->value;
         if (s->kind == AGX3_FLOAT_IMMEDIATE) {
            if (s->abs)
               value &= 0x7f;
            if (s->neg)
               value ^= 0x80;
         }
         fields[n++] = (struct agx3_field_kv){base + s->kind - 1, value};
         fields[n++] = (struct agx3_field_kv){base + 8, s->kind};
         fields[n++] =
            (struct agx3_field_kv){base + 9, half || s->half ? 1 : 2};
         fields[n++] = (struct agx3_field_kv){base + 10, s->cache};
         fields[n++] = (struct agx3_field_kv){base + 12, s->discard};
         fields[n++] = (struct agx3_field_kv){
            base + 13, s->kind != AGX3_FLOAT_IMMEDIATE && s->abs};
         fields[n++] = (struct agx3_field_kv){
            base + 14, s->kind != AGX3_FLOAT_IMMEDIATE && s->neg};
      }
      if (!half && op == AGX3_UNIFORM_FMUL) {
         fields[n++] = (struct agx3_field_kv){60, 2};
         fields[n++] = (struct agx3_field_kv){74, 2};
      } else if (!half && op == AGX3_UNIFORM_FMA) {
         fields[n++] = (struct agx3_field_kv){77, 2};
         fields[n++] = (struct agx3_field_kv){91, 2};
      }
      if (m5) {
         /* f95/f78 is 3 for a GPR result and 1 for a uniform one; ADD has
          * f50=3/f51=48 instead. */
         const unsigned kind = uniform_dest ? 1 : 3;
         fields[n++] = (struct agx3_field_kv){
            op == AGX3_UNIFORM_FMA ? 95 : op == AGX3_UNIFORM_FMUL ? 78 : 50,
            op == AGX3_UNIFORM_FADD ? 3 : kind};
         fields[n++] = (struct agx3_field_kv){
            op == AGX3_UNIFORM_FMA ? 96 : op == AGX3_UNIFORM_FMUL ? 79 : 51,
            op == AGX3_UNIFORM_FMA ? 55 : op == AGX3_UNIFORM_FMUL ? 62 : 48};
      }
      const unsigned wait_base = half ? (op == AGX3_UNIFORM_FMA ? 68 : 51)
                                 : op == AGX3_UNIFORM_FADD ? 52
                                 : op == AGX3_UNIFORM_FMUL ? 80
                                                           : 97;
      for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
         fields[n++] =
            (struct agx3_field_kv){wait_base + i, (wait_mask >> i) & 1};

      /* M5 adds the slot-seven/eight endpoints, and its MUL endpoints
       * are not ordered by length. Use the same complete endpoint domain
       * as the native register-only FALU encoder. */
      /* M5 f32add has endpoints 0..3 only; its 12-byte ep3 carries the
       * independent wait fields, as for the register ADD encoder. */
      const unsigned last_ep = m5 ? (op == AGX3_UNIFORM_FMA    ? 7
                                     : op == AGX3_UNIFORM_FADD ? 3
                                                               : 5)
                                  : op == AGX3_UNIFORM_FMA ? 5 : 3;
      /* Compact wait selectors express one slot. Use the independent fields
       * for multiple producers, without draining unrelated outstanding work. */
      const unsigned first_ep = (wait_mask & (wait_mask - 1))
                                   ? (m5 && op == AGX3_UNIFORM_FMUL ? 3 : last_ep)
                                   : 0;
      for (unsigned ep = first_ep; ep <= last_ep; ++ep) {
         /* Only a strictly shorter encoding can replace the first swap's,
          * and the endpoints grow in length: stop trying once they reach it. */
         if (best_length &&
             agx3_bind_form_size(encoding, mnemonic, ep) >= best_length) {
            if (m5)
               continue;
            break;
         }
         uint8_t bytes[16], size;
         if (agx3_bind_compose(encoding, mnemonic, ep, NULL, 0, fields, n,
                               bytes, &size)) {
            if (!best_length || size < best_length) {
               memcpy(best, bytes, size);
               best_length = size;
            }
            /* These arithmetic endpoints are ordered by increasing length. */
            break;
         }
      }
   }
   if (!best_length)
      return false;
   memcpy(output, best, best_length);
   *length = best_length;
   return true;
}

bool
agx3_uniform_float_encode(enum agx_isa_encoding encoding,
                          enum agx3_uniform_float_op op, uint16_t dest,
                          const struct agx3_uniform_float_src src[3],
                          bool saturate, bool dest_cache, uint8_t wait_mask,
                          uint8_t output[16], uint8_t *length)
{
   return uniform_float_encode(encoding, op, dest, src, saturate, dest_cache,
                               wait_mask, false, false, output, length);
}

bool
agx3_uniform_half_encode(enum agx_isa_encoding encoding,
                         enum agx3_uniform_float_op op, uint16_t dest,
                         const struct agx3_uniform_float_src src[3],
                         bool saturate, bool dest_cache, uint8_t wait_mask,
                         uint8_t output[16], uint8_t *length)
{
   return uniform_float_encode(encoding, op, dest, src, saturate, dest_cache,
                               wait_mask, true, false, output, length);
}

bool
agx3_uniform_float_store_encode(enum agx_isa_encoding encoding,
                                enum agx3_uniform_float_op op, uint16_t dest,
                                const struct agx3_uniform_float_src src[3],
                                uint8_t wait_mask,
                                uint8_t output[16], uint8_t *length)
{
   return uniform_float_encode(encoding, op, dest, src, false, false,
                               wait_mask, false, true, output, length);
}

bool
agx3_uniform_minmax_encode(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t reg, uint16_t uniform, bool half,
                           bool maximum, bool cache, bool discard,
                           bool dest_cache, uint8_t wait_mask,
                           uint8_t output[16], uint8_t *length)
{
   if (encoding != AGX_ISA_ENCODING_AGX3B || !output || !length ||
       dest >= 256 || reg >= 256 || uniform >= 512 ||
       (!half && ((dest | reg | uniform) & 1)) || (cache && discard) ||
       wait_mask > 0x3f || (wait_mask & (wait_mask - 1)))
      return false;

   const unsigned width = half ? 1 : 2;
   struct agx3_field_kv fields[32] = {
      {3, dest},        {10, 2},       {11, width},
      {12, dest_cache}, {35, reg},     {42, 2},
      {43, width},      {44, cache},   {46, discard},
      {53, uniform},    {59, 3},       {60, width},
      {69, reg},        {76, 2},       {77, width},
      {78, cache},      {80, discard}, {87, uniform},
      {93, 3},          {94, width},   {110, maximum ? 7 : 3},
   };
   unsigned n = 21;
   for (unsigned i = 0; i < 6; ++i)
      fields[n++] = (struct agx3_field_kv){102 + i, (wait_mask >> i) & 1};
   for (unsigned ep = half ? 5 : 7; ep < 28; ++ep) {
      uint8_t candidate[16], size;
      if (agx3_bind_compose(encoding, AGX3_MN_fsel, ep, NULL, 0, fields, n, candidate,
                            &size)) {
         memcpy(output, candidate, size);
         *length = size;
         return true;
      }
   }
   return false;
}
