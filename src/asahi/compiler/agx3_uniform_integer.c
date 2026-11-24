/* SPDX-License-Identifier: MIT */
#include "agx3_uniform_integer.h"
#include <string.h>
#include "agx3_binder.h"

static bool
source_valid(struct agx3_uniform_integer_src s, bool half, bool addend,
              unsigned gprs)
{
   if (s.kind == AGX3_UI_ZERO)
      return addend && !s.value && !s.cache && !s.discard;
   if (s.kind != AGX3_UI_REGISTER && s.kind != AGX3_UI_UNIFORM)
      return false;
   if (s.value >=
          (s.kind == AGX3_UI_REGISTER ? gprs : 512) ||
       (!half && (s.value & 1)) || (s.cache && s.discard))
      return false;
   return s.kind != AGX3_UI_UNIFORM || (!s.cache && !s.discard);
}

bool
agx3_uniform_integer_encode(enum agx_isa_encoding encoding,
                            const struct agx3_uniform_integer_desc *d,
                            uint8_t output[16], uint8_t *length)
{
   const bool m5 = encoding == AGX_ISA_ENCODING_AGX3C;
   const unsigned gprs = m5 ? AGX3_UNIFORM_INTEGER_M5_GPRS :
                             AGX3_UNIFORM_INTEGER_GPRS;
   if ((encoding != AGX_ISA_ENCODING_AGX3B && !m5) || !d || !output || !length ||
       d->op > AGX3_UI_ANDC || (int)d->op < 0 ||
       d->dest >= gprs || (!d->half && (d->dest & 1)) ||
       (!m5 && d->wait_mask >= 64) ||
       (d->shift && (d->op != AGX3_UI_ADD || d->shift != 3)))
      return false;

   /* M5 witnesses: full-width unshifted ADD in either operand order;
    * SUB has a varying first operand and uniform second operand. */
   if (m5 && (d->half || d->shift ||
              (d->op != AGX3_UI_ADD && d->op != AGX3_UI_SUB &&
               d->op != AGX3_UI_MAD) ||
              (d->op == AGX3_UI_SUB &&
               (d->src[0].kind != AGX3_UI_REGISTER ||
                d->src[1].kind != AGX3_UI_UNIFORM))))
      return false;
   const bool mad = d->op == AGX3_UI_MAD;
   const bool bitwise = d->op >= AGX3_UI_AND;
   const unsigned count = mad ? 3 : 2, width = d->half ? 1 : 2;
   struct agx3_uniform_integer_src src[3];
   memcpy(src, d->src, sizeof(src));
   for (unsigned i = 0; i < count; ++i)
      if (!source_valid(src[i], d->half, mad && i == 2, gprs))
         return false;

   /* Main-program witnesses have one varying multiplicand/binary operand.
    * Canonicalize commutative forms to Apple's uniform-first specimens. */
   if (!((src[0].kind == AGX3_UI_UNIFORM && src[1].kind == AGX3_UI_REGISTER) ||
         (src[0].kind == AGX3_UI_REGISTER && src[1].kind == AGX3_UI_UNIFORM)))
      return false;
   if (!m5 && src[1].kind == AGX3_UI_UNIFORM && d->op != AGX3_UI_SUB &&
       d->op != AGX3_UI_ANDC) {
      /* A shifted ADD is src0 + (src1 << shift); swapping would change it. */
      if (d->shift)
         return false;
      struct agx3_uniform_integer_src tmp = src[0];
      src[0] = src[1];
      src[1] = tmp;
   }
   if (m5 && mad) {
      /* Native full-width endpoints: R/U/U (cs_loop, ld_constant) and
       * U/R/R (sr_ids). Keep absent/immediate C and narrow operands closed. */
      if (src[2].kind != AGX3_UI_REGISTER &&
          src[2].kind != AGX3_UI_UNIFORM)
         return false;
      const bool uniform_first = src[2].kind == AGX3_UI_REGISTER;
      if ((src[0].kind == AGX3_UI_UNIFORM) != uniform_first) {
         struct agx3_uniform_integer_src tmp = src[0];
         src[0] = src[1];
         src[1] = tmp;
      }
   }
   if (d->op == AGX3_UI_ANDC && src[0].kind != AGX3_UI_UNIFORM)
      return false;

   enum agx3_mnemonic mnemonic = AGX3_MN_NONE;
   unsigned endpoint = 0;
   switch (d->op) {
   case AGX3_UI_ADD:
      mnemonic = m5 ? AGX3_MN_iadd32 : AGX3_MN_iadd;
      break;
   case AGX3_UI_SUB:
      mnemonic = m5 ? AGX3_MN_isub32 : AGX3_MN_isub;
      break;
   case AGX3_UI_MAD:
      mnemonic = m5 ? AGX3_MN_imadd32 : AGX3_MN_imadd;
      break;
   case AGX3_UI_AND:
      mnemonic = AGX3_MN_bit_and;
      endpoint = 1;
      break;
   case AGX3_UI_OR:
      mnemonic = AGX3_MN_bit_or;
      endpoint = 1;
      break;
   case AGX3_UI_XOR:
      mnemonic = AGX3_MN_bit_xor;
      endpoint = 1;
      break;
   case AGX3_UI_NAND:
      mnemonic = AGX3_MN_bit_nand;
      break;
   case AGX3_UI_NOR:
      mnemonic = AGX3_MN_bit_nor;
      break;
   case AGX3_UI_ANDC:
      mnemonic = AGX3_MN_bit_andc;
      break;
   }

   struct agx3_field_kv fields[32];
   unsigned n = 0;
#define FIELD(f, v) fields[n++] = (struct agx3_field_kv){f, v}
   FIELD(2, d->dest);
   FIELD(9, 2);
   FIELD(10, width);
   FIELD(11, d->dest_cache);
   for (unsigned i = 0; i < count; ++i) {
      /* Integer arithmetic sources are 15 fields apart. Bitwise sources
       * have a 14-field stride after their first operand. */
      unsigned base = 15 + i * (bitwise ? 14 : 15);
      FIELD(base + src[i].kind - 1, src[i].value);
      FIELD(base + 8, src[i].kind);
      FIELD(base + 9, src[i].kind == AGX3_UI_ZERO ? 1 : width);
      if (src[i].kind == AGX3_UI_REGISTER) {
         FIELD(base + 10, src[i].cache);
         FIELD(base + 12, src[i].discard);
      }
   }
   if (d->shift)
      FIELD(54, d->shift);
   if (m5) {
      FIELD(mad ? 59 : 44, 3);
      FIELD(mad ? 60 : 45, mad ? 84 : d->op == AGX3_UI_SUB ? 100 : 80);
   }
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      FIELD((m5 ? (mad ? 61 : 46) : mad       ? 60
             : bitwise ? 44
                       : 45) +
               slot,
            !!(d->wait_mask & (1u << slot)));
#undef FIELD
   uint8_t bytes[16], size;
   if (!agx3_bind_compose(encoding, mnemonic, endpoint, NULL, 0, fields, n,
                          bytes, &size))
      return false;
   memcpy(output, bytes, size);
   *length = size;
   return true;
}
