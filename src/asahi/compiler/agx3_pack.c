/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_pack.h"
#include "agx3_binder.h"

#include <string.h>

static bool
is_agx3_encoding(enum agx_isa_encoding encoding)
{
   return encoding == AGX_ISA_ENCODING_AGX3B ||
          encoding == AGX_ISA_ENCODING_AGX3C;
}

static void
write_le16(uint8_t *bytes, uint16_t value)
{
   bytes[0] = value;
   bytes[1] = value >> 8;
}

bool
agx3_pack_ldimm_u32(enum agx_isa_encoding encoding, uint16_t dest,
                    uint32_t imm, struct agx3_encoded_ldimm *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 256 ||
       (dest & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   /*
    * The compact encoding has a four-bit destination in units of 32-bit
    * registers and a seven-bit immediate.  Only its opcode nibble differs:
    * AGX3b uses 0xc and AGX3c uses 0x4.  These bytes match every mi_1..mi_127
    * probe on G15/G16 and G17 respectively.
    */
   if (dest < 32 && imm < 128) {
      const uint8_t opcode =
         encoding == AGX_ISA_ENCODING_AGX3B ? 0x0c : 0x04;

      out->bytes[0] = opcode | ((dest >> 1) << 4);
      out->bytes[1] = imm;
      out->length = 2;
      return true;
   }

   /*
    * The full u32 form is byte-identical in AGX3b and AGX3c.  Apple's f14
    * immediate is scattered across four 16-bit words:
    *
    *   i[0:6]   -> w0[8:14]
    *   i[7:10]  -> w2[1:4]
    *   i[11:12] -> w2[10:11]
    *   i[13:16] -> w3[0:3]
    *   i[17:24] -> w3[4:11]
    *   i[25:31] -> w1[9:15]
    *
    * f2 is the destination half-register: bits 0, 1:4, 5:6 and 7 land in
    * w1[2], w0[4:7], w1[6:7] and w1[3].  Bit 0 is zero for this u32 form.
    */
   const uint16_t words[] = {
      0x800c | (((dest >> 1) & 0xf) << 4) | ((imm & 0x7f) << 8),
      0x0002 | ((dest & 1) << 2) | (((dest >> 5) & 0x3) << 6) |
         (((dest >> 7) & 0x1) << 3) | (((imm >> 25) & 0x7f) << 9),
      (((imm >> 7) & 0xf) << 1) | (((imm >> 11) & 0x3) << 10),
      ((imm >> 13) & 0xf) | (((imm >> 17) & 0xff) << 4),
   };

   for (unsigned i = 0; i < 4; ++i)
      write_le16(&out->bytes[2 * i], words[i]);

   out->length = 8;
   return true;
}

bool
agx3_pack_iadd_u32_imm8(enum agx_isa_encoding encoding, uint16_t dest,
                        uint16_t src, uint8_t imm,
                        struct agx3_encoded_iadd_u32_imm8 *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* This encoding point has 96 addressable 32-bit registers. */
      if (dest >= 192 || src >= 192)
         return false;

      const uint8_t canonical[] = {
         0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x00, 0x88, 0x11, 0x04,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is deposited contiguously for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f31: s[1:5] -> b6[3:7], s[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src & 0x3e) << 2;
      out->bytes[7] |= src >> 6;

      /* f15: i[0:6] -> b5[1:7], i[7] -> b6[0]. */
      out->bytes[5] |= (imm & 0x7f) << 1;
      out->bytes[6] |= imm >> 7;
      out->length = sizeof(canonical);
      return true;
   }

   if (dest >= 256 || src >= 256)
      return false;

   const uint8_t canonical[] = {
      0x27, 0x04, 0x04, 0x1a, 0x25, 0x00,
      0xa3, 0x02, 0x20, 0x80, 0x01, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f31: s[1] -> b8[7], s[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src >> 1) & 0x1) << 7;
   out->bytes[9] |= (src >> 2) & 0x3f;

   /* f15: i[0:1], i[2:4], i[5], i[6:7] are scattered over four bytes. */
   out->bytes[1] |= imm & 0x3;
   out->bytes[3] |= (imm & 0x1c) << 3;
   out->bytes[5] |= (imm & 0x20) << 2;
   out->bytes[8] |= imm >> 6;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_iadd_u32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_iadd_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src0 & 1) != 0 || (src1 & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src0 >= register_limit ||
       src1 >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target Apple compiler replay. */
      const uint8_t canonical[] = {
         0x9f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x13, 0x05,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (src0 & 0x7e) << 1;
      out->bytes[6] |= src0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src1 & 0x3e) << 2;
      out->bytes[7] |= src1 >> 6;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S Apple compiler replay. */
   const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x1a, 0x21, 0x00,
      0xa3, 0x02, 0x28, 0x80, 0x01, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_fadd_f32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_fadd_f32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 256 ||
       src0 >= 256 || src1 >= 256 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   /*
    * Exact common compact selection from the G15G/G16G/G17S delayed-use
    * replay.  Apple fields f2/f35/f18 are destination/source0/source1.
    * Anonymous context fields remain fixed rather than being named here.
    */
   const uint8_t canonical[AGX3_FADD_F32_REG_REG_LENGTH] = {
      0x09, 0x01, 0x04, 0x01, 0x00, 0x20,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7], d[7] -> b5[4]. */
   out->bytes[0] |= (dest & 0x1e) << 3;
   out->bytes[2] |= (dest & 0x60) << 1;
   out->bytes[5] |= (dest & 0x80) >> 3;

   /* f35: s0[1:6] -> b3[1:6], s0[7] -> b5[2]. */
   out->bytes[3] |= src0 & 0x7e;
   out->bytes[5] |= (src0 & 0x80) >> 5;

   /* f18: s1[1:6] -> b1[1:6], s1[7] -> b5[0]. */
   out->bytes[1] |= src1 & 0x7e;
   out->bytes[5] |= (src1 & 0x80) >> 7;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_fmul_f32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_fmul_f32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src0 >= 128 || src1 >= 128 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   /*
    * Exact compact selections from the pinned multiply ladders.  Apple
    * fields f2/f35/f18 are destination/source0/source1.  The selected AGX3b
    * L/P2 witness and selected G17S AGX3c Q witness differ in their fixed
    * anonymous/context trailer, so keep those exact selections distinct.
    */
   const uint8_t canonical_b[AGX3_FMUL_F32_REG_REG_LENGTH] = {
      0x09, 0x01, 0x1d, 0x01, 0x00, 0xc0,
   };
   const uint8_t canonical_c[AGX3_FMUL_F32_REG_REG_LENGTH] = {
      0x09, 0x01, 0x1d, 0x01, 0x00, 0x20,
   };
   const uint8_t *canonical =
      encoding == AGX_ISA_ENCODING_AGX3B ? canonical_b : canonical_c;
   memcpy(out->bytes, canonical, AGX3_FMUL_F32_REG_REG_LENGTH);

   /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7]. */
   out->bytes[0] |= (dest & 0x1e) << 3;
   out->bytes[2] |= (dest & 0x60) << 1;

   /* f35: s0[1:6] -> b3[1:6]. */
   out->bytes[3] |= src0 & 0x7e;

   /* f18: s1[1:6] -> b1[1:6]. */
   out->bytes[1] |= src1 & 0x7e;

   out->length = AGX3_FMUL_F32_REG_REG_LENGTH;
   return true;
}

bool
agx3_pack_fma_f32_reg_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t mul0,
   uint16_t mul1, uint16_t addend,
   struct agx3_encoded_fma_f32_reg_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       mul0 >= 128 || mul1 >= 128 || addend >= 128 || (dest & 1) != 0 ||
       (mul0 & 1) != 0 || (mul1 & 1) != 0 || (addend & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact f32fma.ep3 P3/L anonymous/context selection. */
      const uint8_t canonical[] = {
         0x09, 0x01, 0x1e, 0x01, 0x81, 0x00, 0x02, 0xc0,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;

      /* f35/f18/f52: operand[1:6] -> the matching byte[1:6]. */
      out->bytes[3] |= mul0 & 0x7e;
      out->bytes[1] |= mul1 & 0x7e;
      out->bytes[5] |= addend & 0x7e;

      out->length = sizeof(canonical);
      return true;
   }

   /* Exact f32fma.ep7 P3/L anonymous/context selection. */
   const uint8_t canonical[] = {
      0x29, 0x00, 0x07, 0x0a, 0x20, 0x20, 0xa3, 0x02,
      0x38, 0x00, 0xa0, 0x00, 0x00, 0x08, 0x00, 0x80,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* f35: m0[1] -> b8[7], m0[2:6] -> b9[0:4]. */
   out->bytes[8] |= (mul0 & 0x02) << 6;
   out->bytes[9] |= (mul0 & 0x7c) >> 2;

   /* f18 uses the exact AGX3c table permutation for m1[2:5]. */
   out->bytes[1] |= mul1 & 0x02;
   out->bytes[3] |= (mul1 & 0x04) << 2;
   out->bytes[3] |= (mul1 & 0x08) << 2;
   out->bytes[3] |= (mul1 & 0x10) << 3;
   out->bytes[3] |= (mul1 & 0x20) << 1;
   out->bytes[8] |= (mul1 & 0x40) >> 6;

   /* f52: a[1:5] -> b12[1:5], a[6] -> b10[0]. */
   out->bytes[12] |= addend & 0x3e;
   out->bytes[10] |= (addend & 0x40) >> 6;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_fmin_f32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_fmin_f32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src0 >= 128 || src1 >= 128 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact fsel.ep7 P2/L f32-minimum anonymous/context selection. */
      const uint8_t canonical[] = {
         0x02, 0x01, 0x1e, 0x01, 0x01, 0xc0,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f3: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;

      /* The tied f35/f69 and f52/f86 pairs share these deposits. */
      out->bytes[1] |= src0 & 0x7e;
      out->bytes[3] |= src1 & 0x7e;

      out->length = sizeof(canonical);
      return true;
   }

   /* Exact fseli32.ep4 P2/L f32-minimum anonymous/context selection. */
   const uint8_t canonical[] = {
      0x22, 0x00, 0x47, 0x0a, 0x23, 0x00, 0xa0,
      0x02, 0x14, 0x00, 0xa1, 0x60, 0x00, 0x40,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* Tied comparison/select source0 fields f16/f50. */
   out->bytes[1] |= src0 & 0x02;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 & 0x04) >> 1;
   out->bytes[8] |= src0 & 0x20;
   out->bytes[8] |= (src0 & 0x40) >> 6;
   out->bytes[12] |= src0 & 0x5a;

   /* Tied comparison/select source1 fields f33/f64. */
   out->bytes[3] |= (src1 & 0x02) >> 1;
   out->bytes[8] |= (src1 & 0x02) << 6;
   out->bytes[9] |= (src1 & 0x7c) >> 2;
   out->bytes[12] |= (src1 & 0x04) << 3;
   out->bytes[12] |= (src1 & 0x08) >> 1;
   out->bytes[13] |= (src1 & 0x10) >> 4;
   out->bytes[13] |= (src1 & 0x60) >> 3;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_fmax_f32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_fmax_f32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src0 >= 128 || src1 >= 128 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact fsel.ep7 P2/L f32-maximum anonymous/context selection. */
      const uint8_t canonical[] = {
         0x02, 0x01, 0x1e, 0x01, 0x00, 0xc0,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f3: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;

      /* The tied f35/f69 and f52/f86 pairs share these deposits. */
      out->bytes[1] |= src0 & 0x7e;
      out->bytes[3] |= src1 & 0x7e;

      out->length = sizeof(canonical);
      return true;
   }

   /* Exact fseli32.ep4 P2/L f32-maximum anonymous/context selection. */
   const uint8_t canonical[] = {
      0x22, 0x00, 0x07, 0x0a, 0x23, 0x00, 0xa0,
      0x02, 0x14, 0x00, 0xa1, 0x64, 0x00, 0x40,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* Tied comparison/select source0 fields f16/f50. */
   out->bytes[1] |= src0 & 0x02;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 & 0x04) >> 1;
   out->bytes[8] |= src0 & 0x20;
   out->bytes[8] |= (src0 & 0x40) >> 6;
   out->bytes[12] |= src0 & 0x5a;

   /* Tied comparison/select source1 fields f33/f64. */
   out->bytes[3] |= (src1 & 0x02) >> 1;
   out->bytes[8] |= (src1 & 0x02) << 6;
   out->bytes[9] |= (src1 & 0x7c) >> 2;
   out->bytes[12] |= (src1 & 0x04) << 3;
   out->bytes[12] |= (src1 & 0x08) >> 1;
   out->bytes[13] |= (src1 & 0x10) >> 4;
   out->bytes[13] |= (src1 & 0x60) >> 3;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_i32_to_f32_reg(enum agx_isa_encoding encoding, uint16_t dest,
                         uint16_t src,
                         struct agx3_encoded_i32_to_f32_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src >= 128 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact signed int2flt.ep0 L/PC anonymous/context selection. */
      const uint8_t canonical[] = {
         0xa7, 0x07, 0x56, 0x00, 0x02, 0x00, 0xac, 0x60,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is the destination; f18 is the source. */
      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7e) << 1;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact signed int2flt.ep0 L/PC anonymous/context selection. */
   const uint8_t canonical[] = {
      0x2f, 0x00, 0x00, 0x02, 0x22, 0x80, 0xaf, 0x02, 0x68, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* f18: s1 -> b1[1], s2..3 -> b3[4:5], s4 -> b3[7],
    *      s5 -> b3[6], s6 -> b8[0]. */
   out->bytes[1] |= src & 0x02;
   out->bytes[3] |= (src & 0x0c) << 2;
   out->bytes[3] |= (src & 0x10) << 3;
   out->bytes[3] |= (src & 0x20) << 1;
   out->bytes[8] |= (src & 0x40) >> 6;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_u32_to_f32_reg(enum agx_isa_encoding encoding, uint16_t dest,
                         uint16_t src,
                         struct agx3_encoded_u32_to_f32_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src >= 128 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact unsigned int2flt.ep0 u2f/PC anonymous/context selection. */
      const uint8_t canonical[] = {
         0xa7, 0x07, 0x56, 0x00, 0x02, 0x00, 0xac, 0x20,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is the destination; f18 is the source. */
      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7e) << 1;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact unsigned int2flt.ep0 u2f/PC anonymous/context selection. */
   const uint8_t canonical[] = {
      0x2f, 0x00, 0x00, 0x02, 0x22, 0x80, 0xaf, 0x02, 0x28, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* f18: s1 -> b1[1], s2..3 -> b3[4:5], s4 -> b3[7],
    *      s5 -> b3[6], s6 -> b8[0]. */
   out->bytes[1] |= src & 0x02;
   out->bytes[3] |= (src & 0x0c) << 2;
   out->bytes[3] |= (src & 0x10) << 3;
   out->bytes[3] |= (src & 0x20) << 1;
   out->bytes[8] |= (src & 0x40) >> 6;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_f32_rint_reg(enum agx_isa_encoding encoding, uint16_t dest,
                       uint16_t src, struct agx3_encoded_f32_rint_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src >= 128 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      static const uint8_t canonical[] = {
         0x2f, 0x00, 0x56, 0x00, 0x03, 0x00, 0xb0, 0x40, 0x00, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* Apple f2 is contiguous at b3; f18 occupies b5[2:7]. */
      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7e) << 1;
      out->length = sizeof(canonical);
      return true;
   }

   static const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x18, 0x2a, 0x20, 0xa0, 0x02, 0x30, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* f18: s1 -> b1[1], s2..4 -> b3[5:7], s5 -> b5[7],
    *      s6 -> b8[0]. */
   out->bytes[1] |= src & 0x02;
   out->bytes[3] |= (src & 0x1c) << 3;
   out->bytes[5] |= (src & 0x20) << 2;
   out->bytes[8] |= (src & 0x40) >> 6;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_f32_rcp_reg(enum agx_isa_encoding encoding, uint16_t dest,
                      uint16_t src, struct agx3_encoded_f32_rcp_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src >= 128 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      static const uint8_t canonical[] = {
         0xaf, 0x00, 0x56, 0x00, 0x02, 0x00, 0x10, 0x48, 0x20, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7e) << 1;
      out->length = sizeof(canonical);
      return true;
   }

   static const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x18, 0x22, 0x20, 0xa1, 0x02, 0x30, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   out->bytes[1] |= src & 0x02;
   out->bytes[3] |= (src & 0x1c) << 3;
   out->bytes[5] |= (src & 0x20) << 2;
   out->bytes[8] |= (src & 0x40) >> 6;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_f32_rsqrt_reg(enum agx_isa_encoding encoding, uint16_t dest,
                        uint16_t src,
                        struct agx3_encoded_f32_rsqrt_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 128 ||
       src >= 128 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      static const uint8_t canonical[] = {
         0xaf, 0x01, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7e) << 1;
      out->length = sizeof(canonical);
      return true;
   }

   static const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x18, 0x22, 0x20, 0xa3, 0x02, 0x30, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   out->bytes[1] |= src & 0x02;
   out->bytes[3] |= (src & 0x1c) << 3;
   out->bytes[5] |= (src & 0x20) << 2;
   out->bytes[8] |= (src & 0x40) >> 6;

   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_f32_to_int_rtz_reg(
   enum agx_isa_encoding encoding, enum agx3_f32_to_int_rtz_mode mode,
   uint16_t dest, uint16_t src,
   struct agx3_encoded_f32_to_int_rtz_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) ||
       (mode != AGX3_F32_TO_I32_RTZ && mode != AGX3_F32_TO_U32_RTZ) ||
       dest >= 128 || src >= 128 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      static const uint8_t canonical_i32[] = {
         0x27, 0x07, 0x56, 0x00, 0x02, 0x00, 0xb4, 0x48, 0x03, 0x00,
      };
      static const uint8_t canonical_u32[] = {
         0x27, 0x07, 0x56, 0x00, 0x02, 0x00, 0xb4, 0x08, 0x02, 0x00,
      };
      const uint8_t *canonical =
         mode == AGX3_F32_TO_I32_RTZ ? canonical_i32 : canonical_u32;
      memcpy(out->bytes, canonical, AGX3_F32_TO_INT_RTZ_REG_LENGTH);

      /* f2 is the integer destination; f16 is the float source. */
      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7e) << 1;
      out->length = AGX3_F32_TO_INT_RTZ_REG_LENGTH;
      return true;
   }

   static const uint8_t canonical_i32[] = {
      0x2f, 0x00, 0x00, 0x1a, 0x22, 0x00, 0xae, 0x02, 0x70, 0x01,
   };
   static const uint8_t canonical_u32[] = {
      0x2f, 0x00, 0x00, 0x1a, 0x22, 0x00, 0xae, 0x02, 0xb0, 0x03,
   };
   const uint8_t *canonical =
      mode == AGX3_F32_TO_I32_RTZ ? canonical_i32 : canonical_u32;
   memcpy(out->bytes, canonical, AGX3_F32_TO_INT_RTZ_REG_LENGTH);

   /* f2: d1 -> b0[4], d2..3 -> b7[3:4], d4 -> b0[7],
    *     d5 -> b7[5], d6 -> b2[7]. */
   out->bytes[0] |= (dest & 0x02) << 3;
   out->bytes[7] |= (dest & 0x0c) << 1;
   out->bytes[0] |= (dest & 0x10) << 3;
   out->bytes[7] |= dest & 0x20;
   out->bytes[2] |= (dest & 0x40) << 1;

   /* f16: s1 -> b1[1], s2..4 -> b3[5:7], s5 -> b5[7],
    *      s6 -> b8[0]. */
   out->bytes[1] |= src & 0x02;
   out->bytes[3] |= (src & 0x1c) << 3;
   out->bytes[5] |= (src & 0x20) << 2;
   out->bytes[8] |= (src & 0x40) >> 6;

   out->length = AGX3_F32_TO_INT_RTZ_REG_LENGTH;
   return true;
}

bool
agx3_pack_bit_and_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_and_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 256 ||
       src0 >= 256 || src1 >= 256 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target Apple compiler replay. */
      const uint8_t canonical[AGX3_BIT_AND_U32_REG_REG_LENGTH] = {
         0x0b, 0x01, 0x0f, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7], d[7] -> b5[4]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;
      out->bytes[5] |= (dest & 0x80) >> 3;

      /* f16: s0[1:6] -> b1[1:6], s0[7] -> b5[0]. */
      out->bytes[1] |= src0 & 0x7e;
      out->bytes[5] |= src0 >> 7;

      /* f30: s1[1:6] -> b3[1:6], s1[7] -> b5[2]. */
      out->bytes[3] |= src1 & 0x7e;
      out->bytes[5] |= (src1 & 0x80) >> 5;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S Apple compiler replay. */
   const uint8_t canonical[AGX3_BIT_AND_U32_REG_REG_LENGTH] = {
      0x2b, 0x00, 0x07, 0x80, 0x30, 0x30, 0xa0, 0x02, 0x04, 0x80,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2 uses the bounded eight-bit AGX3c destination scatter. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16 has generation-C-specific fragments in b1, b3, and b5. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[5] |= (src0 & 0x0c) << 4;
   out->bytes[3] |= (src0 & 0x10) >> 3;
   out->bytes[3] |= (src0 & 0xe0) >> 2;

   /* f30: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_bit_or_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_or_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 256 ||
       src0 >= 256 || src1 >= 256 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target Apple compiler replay. */
      const uint8_t canonical[AGX3_BIT_OR_U32_REG_REG_LENGTH] = {
         0x0b, 0x01, 0x0f, 0x01, 0x02, 0x08, 0x00, 0x80, 0x00, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7], d[7] -> b5[4]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;
      out->bytes[5] |= (dest & 0x80) >> 3;

      /* f16: s0[1:6] -> b1[1:6], s0[7] -> b5[0]. */
      out->bytes[1] |= src0 & 0x7e;
      out->bytes[5] |= src0 >> 7;

      /* f30: s1[1:6] -> b3[1:6], s1[7] -> b5[2]. */
      out->bytes[3] |= src1 & 0x7e;
      out->bytes[5] |= (src1 & 0x80) >> 5;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S Apple compiler replay. */
   const uint8_t canonical[AGX3_BIT_OR_U32_REG_REG_LENGTH] = {
      0x2b, 0x00, 0x07, 0x80, 0x32, 0x38, 0xa0, 0x02, 0x04, 0x80,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2 uses the bounded eight-bit AGX3c destination scatter. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16 has generation-C-specific fragments in b1, b3, and b5. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[5] |= (src0 & 0x0c) << 4;
   out->bytes[3] |= (src0 & 0x10) >> 3;
   out->bytes[3] |= (src0 & 0xe0) >> 2;

   /* f30: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_bit_xor_u32_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src0,
   uint16_t src1, struct agx3_encoded_bit_xor_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 256 ||
       src0 >= 256 || src1 >= 256 || (dest & 1) != 0 || (src0 & 1) != 0 ||
       (src1 & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target Apple compiler replay. */
      const uint8_t canonical[AGX3_BIT_XOR_U32_REG_REG_LENGTH] = {
         0x0b, 0x01, 0x0e, 0x01, 0x02, 0x08, 0x00, 0x80, 0x00, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7], d[7] -> b5[4]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;
      out->bytes[5] |= (dest & 0x80) >> 3;

      /* f16: s0[1:6] -> b1[1:6], s0[7] -> b5[0]. */
      out->bytes[1] |= src0 & 0x7e;
      out->bytes[5] |= src0 >> 7;

      /* f30: s1[1:6] -> b3[1:6], s1[7] -> b5[2]. */
      out->bytes[3] |= src1 & 0x7e;
      out->bytes[5] |= (src1 & 0x80) >> 5;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S Apple compiler replay. */
   const uint8_t canonical[AGX3_BIT_XOR_U32_REG_REG_LENGTH] = {
      0x2b, 0x00, 0x06, 0x80, 0x32, 0x38, 0xa0, 0x02, 0x04, 0x80,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2 uses the bounded eight-bit AGX3c destination scatter. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16 has generation-C-specific fragments in b1, b3, and b5. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[5] |= (src0 & 0x0c) << 4;
   out->bytes[3] |= (src0 & 0x10) >> 3;
   out->bytes[3] |= (src0 & 0xe0) >> 2;

   /* f30: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_bit_not_u32_reg(enum agx_isa_encoding encoding, uint16_t dest,
                          uint16_t src,
                          struct agx3_encoded_bit_not_u32_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || dest >= 256 ||
       src >= 256 || (dest & 1) != 0 || (src & 1) != 0)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target forced-live replay. */
      const uint8_t canonical[AGX3_BIT_NOT_U32_REG_MAX_LENGTH] = {
         0x0b, 0x01, 0x26, 0x00, 0x01, 0x08, 0x00, 0x80, 0x00, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2: d[1:4] -> b0[4:7], d[5:6] -> b2[6:7], d[7] -> b5[4]. */
      out->bytes[0] |= (dest & 0x1e) << 3;
      out->bytes[2] |= (dest & 0x60) << 1;
      out->bytes[5] |= (dest & 0x80) >> 3;

      /* f16: s[1:6] -> b1[1:6], s[7] -> b5[0]. */
      out->bytes[1] |= src & 0x7e;
      out->bytes[5] |= src >> 7;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S forced-live replay. */
   const uint8_t canonical[] = {
      0x2b, 0x00, 0x06, 0x1a, 0x29, 0x08, 0xa8, 0x02,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2 uses the bounded eight-bit AGX3c destination scatter. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s[1] in b1, s[2:4] in b3, s[5] in b5, s[6:7] in b6. */
   out->bytes[1] |= src & 0x2;
   out->bytes[3] |= (src & 0x1c) << 3;
   out->bytes[5] |= (src & 0x20) << 2;
   out->bytes[6] |= (src & 0xc0) >> 6;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_isub_u32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_isub_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src0 & 1) != 0 || (src1 & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src0 >= register_limit ||
       src1 >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target forced-live replay. */
      const uint8_t canonical[] = {
         0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x11, 0x05,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (src0 & 0x7e) << 1;
      out->bytes[6] |= src0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src1 & 0x3e) << 2;
      out->bytes[7] |= src1 >> 6;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S forced-live replay. */
   const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x1a, 0x21, 0x00,
      0xa2, 0x02, 0x08, 0x80, 0x01, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_sll_u32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          struct agx3_encoded_sll_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src0 & 1) != 0 || (src1 & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src0 >= register_limit ||
       src1 >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target forced-live replay. */
      const uint8_t canonical[] = {
         0x27, 0x10, 0x54, 0x00, 0x02, 0x00,
         0x00, 0x00, 0xf0, 0x10, 0x05, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (src0 & 0x7e) << 1;
      out->bytes[6] |= src0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src1 & 0x3e) << 2;
      out->bytes[7] |= src1 >> 6;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S forced-live replay. */
   const uint8_t canonical[] = {
      0x27, 0x04, 0x00, 0x1a, 0x22, 0x00, 0xa0,
      0x02, 0x08, 0x80, 0x00, 0x34, 0x00, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_asr_i32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          struct agx3_encoded_asr_i32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src0 & 1) != 0 || (src1 & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src0 >= register_limit ||
       src1 >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target forced-live replay. */
      const uint8_t canonical[] = {
         0xa7, 0x11, 0x54, 0x00, 0x02, 0x00, 0x00, 0x18, 0xe2, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (src0 & 0x7e) << 1;
      out->bytes[6] |= src0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src1 & 0x3e) << 2;
      out->bytes[7] |= src1 >> 6;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S forced-live replay. */
   const uint8_t canonical[] = {
      0x27, 0x04, 0x00, 0x1a, 0x22, 0x00,
      0xa3, 0x02, 0x08, 0x00, 0x40, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_lsr_u32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          struct agx3_encoded_lsr_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src0 & 1) != 0 || (src1 & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src0 >= register_limit ||
       src1 >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact canonical fields from the seven-target forced-live replay. */
      const uint8_t canonical[] = {
         0xa7, 0x10, 0x54, 0x00, 0x02, 0x00,
         0x00, 0x00, 0xf0, 0x10, 0x05, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (src0 & 0x7e) << 1;
      out->bytes[6] |= src0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src1 & 0x3e) << 2;
      out->bytes[7] |= src1 >> 6;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact canonical fields from the G17G/G17S forced-live replay. */
   const uint8_t canonical[] = {
      0x27, 0x04, 0x00, 0x1a, 0x22, 0x00, 0xa1,
      0x02, 0x08, 0x80, 0x00, 0x34, 0x00, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_imul_u32_reg_reg(enum agx_isa_encoding encoding, uint16_t dest,
                           uint16_t src0, uint16_t src1,
                           struct agx3_encoded_imul_u32_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src0 & 1) != 0 || (src1 & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src0 >= register_limit ||
       src1 >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact implicit-zero addend fields from seven forced-live targets. */
      const uint8_t canonical[] = {
         0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
         0x00, 0x00, 0xd0, 0x20, 0x0a, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (src0 & 0x7e) << 1;
      out->bytes[6] |= src0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (src1 & 0x3e) << 2;
      out->bytes[7] |= src1 >> 6;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact implicit-zero addend fields from the G17G/G17S replay. */
   const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x1a, 0x21, 0x00, 0xa1,
      0x02, 0x08, 0x80, 0x00, 0x00, 0x01, 0x02,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= src0 & 0x2;
   out->bytes[3] |= (src0 & 0x1c) << 3;
   out->bytes[5] |= (src0 & 0x20) << 2;
   out->bytes[8] |= (src0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((src1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (src1 >> 2) & 0x3f;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_imadd_u32_reg_reg_reg(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t mul0,
   uint16_t mul1, uint16_t addend,
   struct agx3_encoded_imadd_u32_reg_reg_reg *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (mul0 & 1) != 0 || (mul1 & 1) != 0 || (addend & 1) != 0)
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || mul0 >= register_limit ||
       mul1 >= register_limit || addend >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Exact real-register addend fields from seven forced-live targets. */
      const uint8_t canonical[] = {
         0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
         0x00, 0x00, 0xd0, 0x21, 0x2a, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 is contiguous at b3 for this AGX3b encoding point. */
      out->bytes[3] = dest;

      /* f16: s0[1:6] -> b5[2:7], s0[7] -> b6[0]. */
      out->bytes[5] |= (mul0 & 0x7e) << 1;
      out->bytes[6] |= mul0 >> 7;

      /* f31: s1[1:5] -> b6[3:7], s1[6:7] -> b7[0:1]. */
      out->bytes[6] |= (mul1 & 0x3e) << 2;
      out->bytes[7] |= mul1 >> 6;

      /* f46: add[1:4] -> b7[4:7], add[5:7] -> b8[0:2]. */
      out->bytes[7] |= (addend & 0x1e) << 3;
      out->bytes[8] |= addend >> 5;
      out->length = sizeof(canonical);
      return true;
   }

   /* Exact real-register addend fields from the G17G/G17S replay. */
   const uint8_t canonical[] = {
      0x2f, 0x00, 0x04, 0x1a, 0x21, 0x00, 0xa1,
      0x02, 0x08, 0x80, 0x40, 0x00, 0x00, 0x00,
   };
   memcpy(out->bytes, canonical, sizeof(canonical));

   /* f2: d[1,4] -> b0[4,7], d[2:3,5] -> b7[3:5], d[6:7] -> b2[7,3]. */
   out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
   out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
   out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
   out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
   out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
   out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

   /* f16: s0[1], s0[2:4], s0[5], s0[6:7] use four byte fragments. */
   out->bytes[1] |= mul0 & 0x2;
   out->bytes[3] |= (mul0 & 0x1c) << 3;
   out->bytes[5] |= (mul0 & 0x20) << 2;
   out->bytes[8] |= (mul0 >> 6) & 0x3;

   /* f31: s1[1] -> b8[7], s1[2:7] -> b9[0:5]. */
   out->bytes[8] |= ((mul1 >> 1) & 0x1) << 7;
   out->bytes[9] |= (mul1 >> 2) & 0x3f;

   /* f46: add[1:3,5,7] -> b12; add[4] -> b8; add[6] -> b10. */
   out->bytes[8] |= addend & 0x10;
   out->bytes[10] |= (addend >> 6) & 0x1;
   out->bytes[12] |= addend & 0xae;
   out->length = sizeof(canonical);
   return true;
}

bool
agx3_pack_scoped_barrier(enum agx_isa_encoding encoding,
                         enum agx3_barrier_scope scope,
                         struct agx3_encoded_scoped_barrier *out)
{
   if (scope == AGX3_BARRIER_SCOPE_SUBGROUP_SHARED) {
      if (!is_agx3_encoding(encoding) || out == NULL)
         return false;
      memset(out, 0, sizeof(*out));
      return true;
   }
   if (scope == AGX3_BARRIER_SCOPE_MESH_PAYLOAD_ENTRY ||
       scope == AGX3_BARRIER_SCOPE_OBJECT_PAYLOAD_EXIT) {
      if (encoding != AGX_ISA_ENCODING_AGX3C || out == NULL) return false;
      /* mesh_amplified +0xa, before reading SR62/63 and LDDEV payload.
       * Retain the capture's anonymous selector fields exactly. */
      const struct agx3_field_kv mesh_fields[] = {
         {14,3},{15,9},{24,1},{30,3},{40,1},
      };
      const struct agx3_field_kv object_fields[] = {
         {14,3},{15,9},{24,1},{30,4},{39,1},{40,1},
      };
      bool object=scope==AGX3_BARRIER_SCOPE_OBJECT_PAYLOAD_EXIT;
      const struct agx3_field_kv *fields=object ? object_fields : mesh_fields;
      unsigned count=object ? sizeof(object_fields)/sizeof(object_fields[0])
                            : sizeof(mesh_fields)/sizeof(mesh_fields[0]);
      memset(out,0,sizeof(*out));
      return agx3_bind_compose(encoding,AGX3_MN_barrier,0,fields,
         count,NULL,0,out->bytes,&out->length);
   }

   static const uint8_t canonical_b[][AGX3_SCOPED_BARRIER_MAX_LENGTH] = {
      [AGX3_BARRIER_SCOPE_EXECUTION] =
         {0x07, 0x04, 0x54, 0x41, 0x09, 0x00},
      [AGX3_BARRIER_SCOPE_THREADGROUP] =
         {0x07, 0x04, 0x54, 0x61, 0x09, 0x00},
      [AGX3_BARRIER_SCOPE_DEVICE] =
         {0x07, 0x04, 0x54, 0x85, 0x08, 0x00},
      [AGX3_BARRIER_SCOPE_TEXTURE] = {
         0x07, 0x04, 0x54, 0x51, 0x0e, 0x00,
         0x07, 0x04, 0x54, 0xd1, 0x0e, 0x00,
      },
      [AGX3_BARRIER_SCOPE_DEVICE_FENCE] =
         {0x07, 0x04, 0x54, 0x84, 0x0a, 0x00},
      [AGX3_BARRIER_SCOPE_TEXTURE_FENCE] = {
         0x07, 0x04, 0x54, 0x50, 0x06, 0x00,
         0x07, 0x04, 0x54, 0xd0, 0x06, 0x00,
      },
      [AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT] = {
         0x07, 0x14, 0x54, 0x50, 0x06, 0x00,
         0x07, 0x04, 0x54, 0xd0, 0x06, 0x00,
      },
   };
   static const uint8_t canonical_c[][AGX3_SCOPED_BARRIER_MAX_LENGTH] = {
      [AGX3_BARRIER_SCOPE_EXECUTION] =
         {0x07, 0x51, 0x00, 0x06, 0x00, 0x00},
      [AGX3_BARRIER_SCOPE_THREADGROUP] =
         {0x27, 0x51, 0x00, 0x06, 0x00, 0x00},
      /* G17G atomic_thread_fence(mem_device, seq_cst, thread_scope_device).
       * Unlike the workgroup barrier, this is valid in one active lane. */
      [AGX3_BARRIER_SCOPE_DEVICE_FENCE] =
         {0x0f, 0x2b, 0x00, 0x06, 0x00, 0x00},
      [AGX3_BARRIER_SCOPE_DEVICE] =
         {0x0f, 0x69, 0x00, 0x06, 0x00, 0x00},
      [AGX3_BARRIER_SCOPE_TEXTURE] = {
         0x87, 0x67, 0x00, 0x06, 0x00, 0x00,
         0x87, 0x5f, 0x00, 0x06, 0x00, 0x00,
      },
      /* G17G texture.fence() and atomic_thread_fence(mem_texture) at device
       * or simdgroup scope (port/fencewit k_fence, k_atf_dev, k_atf_simd). */
      [AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT] = {
         0x87, 0x26, 0x00, 0x06, 0x00, 0x00,
         0x87, 0x1e, 0x00, 0x06, 0x00, 0x00,
      },
      /* atomic_thread_fence(mem_texture, thread_scope_threadgroup). */
      [AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP] = {
         0x87, 0x27, 0x00, 0x06, 0x00, 0x00,
         0x87, 0x1f, 0x00, 0x06, 0x00, 0x00,
      },
   };
   const uint8_t(*canonical)[AGX3_SCOPED_BARRIER_MAX_LENGTH] =
      encoding == AGX_ISA_ENCODING_AGX3B ? canonical_b :
      encoding == AGX_ISA_ENCODING_AGX3C ? canonical_c : NULL;
   const uint8_t length =
      scope == AGX3_BARRIER_SCOPE_TEXTURE ||
            scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE ||
            scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT ||
            scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP
         ? 12
         : 6;

   if (canonical == NULL || out == NULL ||
       scope < AGX3_BARRIER_SCOPE_EXECUTION ||
       scope > AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP ||
       (encoding == AGX_ISA_ENCODING_AGX3B &&
        scope > AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT) ||
       (encoding == AGX_ISA_ENCODING_AGX3C &&
        scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE))
      return false;

   memset(out, 0, sizeof(*out));
   memcpy(out->bytes, canonical[scope], length);
   out->length = length;
   return true;
}

bool
agx3_pack_scoped_barrier_wait(enum agx_isa_encoding encoding,
                              enum agx3_barrier_scope scope,
                              uint16_t wait_mask,
                              struct agx3_encoded_scoped_barrier *out)
{
   if (!agx3_pack_scoped_barrier(encoding, scope, out))
      return false;
   if (!wait_mask)
      return true;

   /* Field 29 is the threadgroup rendezvous and field 39 the threadgroup
    * memory scope; the first word is scope 4, the second scope 3. */
   const bool rendezvous = scope == AGX3_BARRIER_SCOPE_TEXTURE;
   const bool threadgroup =
      rendezvous || scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP;
   if (encoding != AGX_ISA_ENCODING_AGX3C || (wait_mask & ~UINT16_C(0xff)) ||
       (scope != AGX3_BARRIER_SCOPE_TEXTURE &&
        scope != AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT &&
        scope != AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   const struct agx3_field_kv base[] = {
      {14, 3}, {15, 9}, {28, 1}, {29, rendezvous}, {30, 4},
      {39, threadgroup}, {40, 1}, {41, 1},
   };
   struct agx3_field_kv wait[8];
   unsigned n = 0;
   for (unsigned i = 0; i < 8; ++i) {
      if (wait_mask & (1u << i))
         wait[n++] = (struct agx3_field_kv){16 + i, 1};
   }

   uint8_t first[AGX3_SCOPED_BARRIER_MAX_LENGTH], len = 0;
   if (!agx3_bind_compose(encoding, AGX3_MN_barrier, 0, base,
                          sizeof(base) / sizeof(base[0]), wait, n, first,
                          &len) ||
       len != 6) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   memcpy(out->bytes, first, 6);
   return true;
}

bool
agx3_pack_threadgroup_barrier(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_threadgroup_barrier *out)
{
   if (out == NULL)
      return false;

   struct agx3_encoded_scoped_barrier scoped;
   if (!agx3_pack_scoped_barrier(encoding, AGX3_BARRIER_SCOPE_THREADGROUP,
                                 &scoped))
      return false;

   memcpy(out->bytes, scoped.bytes, AGX3_THREADGROUP_BARRIER_LENGTH);
   out->length = AGX3_THREADGROUP_BARRIER_LENGTH;
   return true;
}

bool
agx3_pack_simd_broadcast_f32_imm(
   enum agx_isa_encoding encoding, uint16_t dest, uint16_t src, uint8_t lane,
   struct agx3_encoded_simd_broadcast_f32_imm *out)
{
   if (out == NULL || !is_agx3_encoding(encoding) || (dest & 1) != 0 ||
       (src & 1) != 0 || (lane != 0 && lane != 3))
      return false;

   const uint16_t register_limit =
      encoding == AGX_ISA_ENCODING_AGX3B ? 192 : 256;
   if (dest >= register_limit || src >= register_limit)
      return false;

   memset(out, 0, sizeof(*out));
   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* simdbcast.ep0, f9=2 f10=2 f23=2 f24=2 f27=1 f37=1 f38=1
       * f49=1.  The register/lane placements are the grade-A f2/f16/f29
       * roles separated by SD_dst2 and SB_bcast0/SB_bcast3. */
      static const uint8_t canonical[AGX3_SIMD_BROADCAST_F32_IMM_LENGTH] = {
         0x47, 0x04, 0x56, 0x00, 0x02, 0x00, 0x00, 0x2c, 0x04, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));
      out->bytes[3] |= dest;
      out->bytes[5] |= (src & 0x7f) << 1;
      out->bytes[6] |= src >> 7;
      out->bytes[6] |= lane << 1;
   } else {
      /* simdbcast.ep0, f9=2 f10=2 f23=2 f24=2 f27=1 f37=1 f38=1
       * f42=3 f43=154 f51=1. */
      static const uint8_t canonical[AGX3_SIMD_BROADCAST_F32_IMM_LENGTH] = {
         0x2f, 0x00, 0x21, 0x1a, 0x20, 0x00, 0xa8, 0x02, 0x38, 0x00,
      };
      memcpy(out->bytes, canonical, sizeof(canonical));

      /* f2 (destination); the unobserved ninth bit is intentionally absent. */
      out->bytes[0] |= ((dest >> 1) & 0x1) << 4;
      out->bytes[0] |= ((dest >> 4) & 0x1) << 7;
      out->bytes[7] |= ((dest >> 2) & 0x3) << 3;
      out->bytes[7] |= ((dest >> 5) & 0x1) << 5;
      out->bytes[2] |= ((dest >> 6) & 0x1) << 7;
      out->bytes[2] |= ((dest >> 7) & 0x1) << 3;

      /* f16 (source); likewise bounded to the witnessed eight-bit domain. */
      out->bytes[1] |= src & 0x2;
      out->bytes[3] |= (src & 0x1c) << 3;
      out->bytes[5] |= (src & 0x20) << 2;
      out->bytes[8] |= (src >> 6) & 0x3;

      /* f29 immediate lane. */
      out->bytes[8] |= (lane & 0x3) << 6;
   }

   out->length = AGX3_SIMD_BROADCAST_F32_IMM_LENGTH;
   return true;
}

bool
agx3_pack_end(enum agx_isa_encoding encoding, struct agx3_encoded_end *out)
{
   if (!is_agx3_encoding(encoding) || out == NULL)
      return false;

   memset(out, 0, sizeof(*out));
   out->bytes[0] = 0x0e;
   out->length = AGX3_END_LENGTH;
   return true;
}

/* Direct differential pairs from Apple's instructions:
 *
 * AGX3b cases 561/470 differ only in f5=0/4.
 * AGX3c cases 1351/1252 differ only in f5=0/8.
 *
 * Keep the anonymous, non-f5 fields byte-exact instead of synthesising other
 * table-valid forms whose scheduling and traversal roles are not grounded.
 */
static const uint8_t intersect_ray_canonical_agx3b[] = {
   0x5f, 0x10, 0x54, 0x00, 0x00, 0x41,
   0x26, 0x40, 0xc7, 0x50, 0x4c, 0x00,
};

static const uint8_t intersect_ray_canonical_agx3c[] = {
   0x27, 0x04, 0x29, 0x01, 0xa2, 0xc1, 0xae,
   0x02, 0x01, 0x98, 0x27, 0x00, 0x00, 0x20,
};

static const uint8_t *
intersect_ray_canonical_encoding(enum agx_isa_encoding encoding,
                                 size_t *length)
{
   switch (encoding) {
   case AGX_ISA_ENCODING_AGX3B:
      *length = sizeof(intersect_ray_canonical_agx3b);
      return intersect_ray_canonical_agx3b;
   case AGX_ISA_ENCODING_AGX3C:
      *length = sizeof(intersect_ray_canonical_agx3c);
      return intersect_ray_canonical_agx3c;
   default:
      *length = 0;
      return NULL;
   }
}

static bool
intersect_ray_state_register_supported(enum agx_isa_encoding encoding,
                                       uint16_t state_register)
{
   return (encoding == AGX_ISA_ENCODING_AGX3B &&
           (state_register == 0 || state_register == 4)) ||
          (encoding == AGX_ISA_ENCODING_AGX3C &&
           (state_register == 0 || state_register == 8));
}

bool
agx3_pack_intersect_ray(enum agx_isa_encoding encoding,
                        uint16_t state_register,
                        struct agx3_encoded_intersect_ray *out)
{
   size_t length = 0;
   const uint8_t *canonical =
      intersect_ray_canonical_encoding(encoding, &length);

   if (canonical == NULL || out == NULL ||
       !intersect_ray_state_register_supported(encoding, state_register))
      return false;

   memset(out, 0, sizeof(*out));
   memcpy(out->bytes, canonical, length);

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* f5[0:7] -> word 1 bits 8:15; f5[8:12] -> word 2 bits 0:4. */
      out->bytes[3] |= state_register & 0xff;
      out->bytes[4] |= (state_register >> 8) & 0x1f;
   } else {
      /* AGX3c f5[1:12], from Apple's intersect_ray encoder table. */
      out->bytes[12] |= (state_register & 0x0002) >> 1;
      out->bytes[12] |= (state_register & 0x0004) << 1;
      out->bytes[13] |= (state_register & 0x0008) >> 2;
      out->bytes[13] |= state_register & 0x0010;
      out->bytes[12] |= (state_register & 0x0020) >> 1;
      out->bytes[13] |= state_register & 0x0040;
      out->bytes[12] |= (state_register & 0x0080) >> 1;
      out->bytes[12] |= (state_register & 0x0100) >> 3;
      out->bytes[13] |= (state_register & 0x0200) >> 9;
      out->bytes[13] |= (state_register & 0x0400) >> 8;
      out->bytes[2] |= (state_register & 0x0800) >> 5;
      out->bytes[0] |= (state_register & 0x1000) >> 9;
   }

   out->length = length;
   return true;
}

enum agx_isa_encoding
agx3_intersect_ray_witnessed_chip_encoding(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G17P:
      return AGX_ISA_ENCODING_AGX3B;
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
      return AGX_ISA_ENCODING_AGX3C;
   default:
      return AGX_ISA_ENCODING_UNKNOWN;
   }
}

static const uint8_t continue_ray_canonical_agx3b[] = {
   0x5f, 0x11, 0x54, 0x00, 0x00, 0x00,
};

static const uint8_t continue_ray_canonical_agx3c[] = {
   0x27, 0x04, 0x09, 0x00, 0x22, 0xc0, 0xaf, 0x02,
};

static const uint8_t *
continue_ray_canonical_encoding(enum agx_isa_encoding encoding,
                                size_t *length)
{
   switch (encoding) {
   case AGX_ISA_ENCODING_AGX3B:
      *length = sizeof(continue_ray_canonical_agx3b);
      return continue_ray_canonical_agx3b;
   case AGX_ISA_ENCODING_AGX3C:
      *length = sizeof(continue_ray_canonical_agx3c);
      return continue_ray_canonical_agx3c;
   default:
      *length = 0;
      return NULL;
   }
}

static bool
continue_ray_state_register_supported(enum agx_isa_encoding encoding,
                                      uint16_t state_register)
{
   static const uint16_t agx3b_registers[] = {0, 2, 4, 6, 8, 10};
   static const uint16_t agx3c_registers[] = {0, 2, 8, 10, 12, 16, 18};
   const uint16_t *registers;
   size_t count;

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      registers = agx3b_registers;
      count = sizeof(agx3b_registers) / sizeof(agx3b_registers[0]);
   } else if (encoding == AGX_ISA_ENCODING_AGX3C) {
      registers = agx3c_registers;
      count = sizeof(agx3c_registers) / sizeof(agx3c_registers[0]);
   } else {
      return false;
   }

   for (size_t i = 0; i < count; ++i) {
      if (registers[i] == state_register)
         return true;
   }

   return false;
}

bool
agx3_pack_continue_ray(enum agx_isa_encoding encoding,
                       uint16_t state_register,
                       struct agx3_encoded_continue_ray *out)
{
   size_t length = 0;
   const uint8_t *canonical =
      continue_ray_canonical_encoding(encoding, &length);

   if (canonical == NULL || out == NULL ||
       !continue_ray_state_register_supported(encoding, state_register))
      return false;

   memset(out, 0, sizeof(*out));
   memcpy(out->bytes, canonical, length);

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      /* f5[0:7] -> word 1 bits 8:15; f5[8:12] -> word 2 bits 0:4. */
      out->bytes[3] |= state_register & 0xff;
      out->bytes[4] |= (state_register >> 8) & 0x1f;
   } else {
      /* AGX3c scatters f5[1:12] across four instruction words. */
      out->bytes[7] |= (state_register & 0x0002) << 5;
      out->bytes[4] |= (state_register & 0x000c) << 1;
      out->bytes[4] |= (state_register & 0x0010) << 3;
      out->bytes[5] |= (state_register & 0x0060) >> 4;
      out->bytes[5] |= (state_register & 0x0180) >> 3;
      out->bytes[2] |= (state_register & 0x0200) >> 2;
      out->bytes[3] |= (state_register & 0x0400) >> 8;
      out->bytes[0] |= (state_register & 0x0800) >> 7;
      out->bytes[0] |= (state_register & 0x1000) >> 9;
   }

   out->length = length;
   return true;
}

enum agx_isa_encoding
agx3_continue_ray_witnessed_chip_encoding(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G17P:
      return AGX_ISA_ENCODING_AGX3B;
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
      return AGX_ISA_ENCODING_AGX3C;
   default:
      return AGX_ISA_ENCODING_UNKNOWN;
   }
}

bool
agx3_pack_continue_ray_fixed_first_site_witness(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_continue_ray_fixed_first_site_witness *out)
{
   if (out == NULL)
      return false;

   struct agx3_encoded_continue_ray canonical;
   if (!agx3_pack_continue_ray(encoding, 0, &canonical))
      return false;

   memset(out, 0, sizeof(*out));
   memcpy(out->bytes, canonical.bytes, canonical.length);
   out->length = canonical.length;
   return true;
}

static const uint8_t
   inc_stk_fixed_witness_agx3b[AGX3_INC_STK_FIXED_WITNESS_LENGTH] = {
      0x07, 0x00, 0x54, 0x12, 0x80, 0x00, 0x00, 0x00,
   };

static const uint8_t
   inc_stk_fixed_witness_agx3c[AGX3_INC_STK_FIXED_WITNESS_LENGTH] = {
      0x9f, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };

static const uint8_t *
inc_stk_fixed_witness_encoding(enum agx_isa_encoding encoding)
{
   switch (encoding) {
   case AGX_ISA_ENCODING_AGX3B:
      return inc_stk_fixed_witness_agx3b;
   case AGX_ISA_ENCODING_AGX3C:
      return inc_stk_fixed_witness_agx3c;
   default:
      return NULL;
   }
}

bool
agx3_pack_inc_stk_fixed_witness(
   enum agx_isa_encoding encoding,
   struct agx3_encoded_inc_stk_fixed_witness *out)
{
   const uint8_t *witness = inc_stk_fixed_witness_encoding(encoding);
   if (witness == NULL || out == NULL)
      return false;

   memset(out, 0, sizeof(*out));
   memcpy(out->bytes, witness, AGX3_INC_STK_FIXED_WITNESS_LENGTH);
   out->length = AGX3_INC_STK_FIXED_WITNESS_LENGTH;
   return true;
}

enum agx_isa_encoding
agx3_inc_stk_witnessed_chip_encoding(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G17P:
      return AGX_ISA_ENCODING_AGX3B;
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
      return AGX_ISA_ENCODING_AGX3C;
   default:
      return AGX_ISA_ENCODING_UNKNOWN;
   }
}
