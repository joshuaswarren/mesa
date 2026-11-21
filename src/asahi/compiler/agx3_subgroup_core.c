/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_subgroup_core.h"

#include "agx3_binder.h"

#define ADD_FIELD(F, V)                                                        \
   do {                                                                        \
      base[nb++] = (struct agx3_field_kv){(F), (V)};                           \
   } while (0)

static bool
is_c(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3C;
}

static bool
valid_reg(enum agx_isa_encoding enc, uint16_t reg)
{
   const uint16_t limit = enc == AGX_ISA_ENCODING_AGX3B   ? 192
                          : enc == AGX_ISA_ENCODING_AGX3C ? 256
                                                          : 0;
   return limit != 0 && (reg & 1) == 0 && reg < limit;
}

static enum agx3_mnemonic
shuffle_mnemonic(bool quad, enum agx3_subgroup_shuffle_op op)
{
   static const enum agx3_mnemonic simd[] = {
      AGX3_MN_simdbcast,
      AGX3_MN_simdbfly,
      AGX3_MN_simdshftup,
      AGX3_MN_simdshftdn,
   };
   static const enum agx3_mnemonic q[] = {
      AGX3_MN_quadshfl,
      AGX3_MN_quadbfly,
      AGX3_MN_quadshftup,
      AGX3_MN_quadshftdn,
   };
   return (unsigned)op < AGX3_SUBGROUP_SHUFFLE_NUM
             ? (quad ? q[op] : simd[op])
             : AGX3_MN_NONE;
}

bool
agx3_subgroup_encode_active_mask(uint16_t dest, bool dest_cache,
                                 uint8_t wait_mask,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length)
{
   if (!out || !length || dest >= 256 || (dest & 1))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,1},{24,1},{38,1},{39,1},{44,3},{45,157},
   };
   struct agx3_field_kv fields[14] = {
      {2,dest},{11,dest_cache},{15,1},{30,0},{54,0},{55,1},
   };
   unsigned count = 6;
   for (unsigned bit = 0; bit < 8; ++bit)
      if (wait_mask & (1u << bit))
         fields[count++] = (struct agx3_field_kv){46 + bit, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,AGX3_MN_simdiballot,0,
                           base,8,fields,count,out,length);
}

/*
 * The 16-bit quad ballot of a boolean immediate against zero. Native
 * native G17G (lane-mesh/cwit-quad, cwit-quad2) keeps a quad vote whose
 * operand is a constant as a real ballot rather than folding it:
 *
 *   quad_any(false), result into a store    370020182d00a61a30000840  (r7l)
 *   ballot(1 != 0) in q_vote (quad votes)    370120182d00a60230000840  (r1l)
 *
 * f23/f24 = 1/1 name an immediate 16-bit source whose value is f15, f38/f39
 * = 1/1 the immediate zero it is compared with, f54 = 0 the equality and f55
 * its inversion: the quad counterpart of agx3_subgroup_encode_active_mask().
 * Apple adds f64 only when the result reaches nothing but the colour output;
 * it stays clear, as on the register-source quad ballots.
 */
bool
agx3_subgroup_encode_quad_ballot_imm(uint16_t dest, bool dest_cache,
                                     uint16_t value, uint8_t wait_mask,
                                     uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                     uint8_t *length)
{
   if (!out || !length || dest >= 256 || value > 1)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 1}, {24, 1}, {38, 1}, {39, 1}, {44, 3}, {45, 139},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {11, dest_cache}, {15, value}, {54, 0}, {55, 1},
   };
   unsigned count = 5;
   for (unsigned bit = 0; bit < 8; ++bit)
      if (wait_mask & (1u << bit))
         fields[count++] = (struct agx3_field_kv){(uint16_t)(46 + bit), 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_quadiballot, 0, base, 8,
                            fields, count, out, length);
}

/* `g18g` selects the G18G layer's forms; see agx3_subgroup_encode_*_g18g(). */
static bool
encode_shuffle(enum agx_isa_encoding enc,
               const struct agx3_subgroup_shuffle_desc *d, bool g18g,
               uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], uint8_t *length)
{
   if (d == NULL || out == NULL || length == NULL)
      return false;
   /*
    * 16-bit data narrows the width pair, and the quad selector is a different
    * field of the same immediate-lane encoding point, so the two compose.
    * metal-nt -arch applegpu_g17p, one kernel per operation on `ushort`,
    * lane immediate, same registers, nothing else moving:
    *
    *   quad_broadcast(a, 3)      47005602020406200400   quadshfl
    *   simd_broadcast(a, 3)      47045602020406200400   simdbcast
    *   quad_shuffle_xor(a, 1)    c7005602020402200400   quadbfly
    *   simd_shuffle_xor(a, 1)    c7045602020402200400   simdbfly
    *
    * Only byte 1 moves between a quad word and its SIMD control, which is the
    * same selector the 32-bit words in the immediate-lane block below carry.
    * The shifts are NOT admitted here: Apple's 16-bit simd_shuffle_up is
    * 47055603030402200200, whose f9/f10 pair is 3/3 rather than the 2/1 this
    * composition would deposit, so that form needs its own mapping.
    */
   const bool half_shuffle_op = d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST ||
                                d->op == AGX3_SUBGROUP_SHUFFLE_XOR;
   if (d->half ? (enc != AGX_ISA_ENCODING_AGX3B && !is_c(enc)) ||
                    (d->lane_kind != AGX3_SUBGROUP_LANE_REGISTER &&
                     !half_shuffle_op) ||
                    (d->quad && d->lane_kind != AGX3_SUBGROUP_LANE_IMMEDIATE) ||
                    d->dest >= (is_c(enc) ? 256 : 192) ||
                    d->src >= (is_c(enc) ? 256 : 192)
               : !valid_reg(enc, d->dest) || !valid_reg(enc, d->src))
      return false;

   enum agx3_mnemonic mnemonic = shuffle_mnemonic(d->quad, d->op);
   if (mnemonic == AGX3_MN_NONE)
      return false;

   /*
    * The SIMD shuffles from a register lane on G17P. metal-nt -arch
    * applegpu_g17p, `simd_broadcast(ushort d, ushort lane)` with d and lane
    * each either read again later or not:
    *
    *   d dead, lane dead    47145402020604201c00   f27 f41
    *   d live, lane live    4714540702100c201000
    *   d dead, lane live    47145405020008201400   f27
    *   d live, lane dead    47145405020800201800   f41
    *
    * simd_shuffle(), whose broadcasts reuse both soon, adds f25 (source) and
    * f39 (lane) next to f11 (destination). So f25/f27 are the source's
    * cache/discard hints and f39/f41 the lane's, with the lane in f30 and the
    * source wait in the f44 bank. The same topology holds for 32-bit data
    * (f10 = f24 = 2; a broadcast reading its lane twice is 471454040204142e1400
    * then 472454000200142c1c00) and for simd_shuffle_xor/_up/_down at both
    * widths, so the lifetime bits follow the operands rather than being
    * pinned: a pinned f41 claimed the lane was dead even when the vote and
    * shuffle lowerings read it again for the next component.
    */
   if ((enc == AGX_ISA_ENCODING_AGX3B || is_c(enc)) &&
       (!d->quad || (g18g && is_c(enc))) &&
       d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER) {
      const unsigned slots = is_c(enc) ? 8 : 6;
      if (d->lane >= (is_c(enc) ? 256 : 192) || (d->wait_mask >> slots) != 0 ||
          (d->src_cache && d->src_discard) ||
          (d->lane_cache && d->lane_discard))
         return false;

      const unsigned width = d->half ? 1 : 2;
      struct agx3_field_kv hbase[8];
      unsigned hnb = 0;
      hbase[hnb++] = (struct agx3_field_kv){9, 2};
      hbase[hnb++] = (struct agx3_field_kv){10, width};
      hbase[hnb++] = (struct agx3_field_kv){23, 2};
      hbase[hnb++] = (struct agx3_field_kv){24, width};
      hbase[hnb++] = (struct agx3_field_kv){37, 2};
      hbase[hnb++] = (struct agx3_field_kv){38, 1};

      if (is_c(enc)) {
         /* Native886 G17G/G18G dynamic scalar/vector shuffles use the same
          * lifetime fields, half-register lane address and wait bank. G18
          * additionally carries its operation selector in f42/f43; the G18G
          * layer also takes its quad shuffles from a register lane here. */
         static const uint8_t ids[2][AGX3_SUBGROUP_SHUFFLE_NUM] = {
            {154, 155, 161, 160}, {140, 137, 142, 141},
         };
         hbase[hnb++] = (struct agx3_field_kv){42, 3};
         hbase[hnb++] = (struct agx3_field_kv){43, ids[d->quad][d->op]};
      }
      struct agx3_field_kv hov[16];
      unsigned hnv = 0;
      hov[hnv++] = (struct agx3_field_kv){2, d->dest};
      hov[hnv++] = (struct agx3_field_kv){11, d->dest_hint ? 1u : 0u};
      hov[hnv++] = (struct agx3_field_kv){16, d->src};
      hov[hnv++] = (struct agx3_field_kv){25, d->src_cache ? 1u : 0u};
      hov[hnv++] = (struct agx3_field_kv){27, d->src_discard ? 1u : 0u};
      hov[hnv++] = (struct agx3_field_kv){30, d->lane};
      hov[hnv++] = (struct agx3_field_kv){39, d->lane_cache ? 1u : 0u};
      hov[hnv++] = (struct agx3_field_kv){41, d->lane_discard ? 1u : 0u};
      for (unsigned i = 0; i < slots; ++i)
         if (d->wait_mask & (1u << i))
            hov[hnv++] = (struct agx3_field_kv){(uint16_t)(44 + i), 1};

      return agx3_bind_compose(enc, mnemonic, 0, hbase, hnb, hov, hnv, out,
                               length);
   }

   /*
    * The immediate-lane form: f37 = 1 selects it and the lane index sits in
    * f29; f39 and f41, which the register form carries, are absent. The
    * per-slot wait bits are f44 + (slot - 1), so a source arriving on slot 6
    * sets f49 and nothing else.
    */
   if (d->lane_kind == AGX3_SUBGROUP_LANE_IMMEDIATE) {
      const unsigned slots = is_c(enc) ? 8 : 6;
      /*
       * f29 IS the lane, for every immediate-lane shuffle and for every lane
       * the seven-bit field holds.  This used to admit three SIMD broadcast
       * lanes (0, 3, 5) and the quad family only, so `subgroupBroadcast(v,
       * 31)` -- the last-lane read that every subgroup scan ends with, Mesa's
       * own radix sort included -- had no encoding at all and the shader had
       * to pay a materialising `mov` to reach the register-lane form instead.
       *
       * Apple's applegpu_g17p output, one kernel per lane, is a single
       * field sweep with nothing else moving:
       *
       *   simd_broadcast(a, 0)      470456000200002c0400   f29 = 0
       *   simd_broadcast(a, 1)      470456000200022c0400   f29 = 1
       *   simd_broadcast(a, 5)      4704560002000a2c0400   f29 = 5
       *   simd_broadcast(a, 31)     4704560002003e2c0400   f29 = 31
       *   simd_broadcast(a, 32)     470456000200402c0400   f29 = 32
       *   simd_broadcast(a, 63)     4704560002007e2c0400   f29 = 63
       *   simd_shuffle_xor(a, 63)   c704560002007e2c0400   simdbfly, same f29
       *   simd_shuffle_up(a, 31)    4705560403003e2c0200   simdshftup
       *   simd_shuffle_down(a, 31)  c705560403003e2c0200   simdshftdn
       *   quad_broadcast(a, 3)      470056000200062c0400   quadshfl
       *   quad_shuffle_xor(a, 3)    c70056000200062c0400   quadbfly
       *
       * Every one of those also sets f49, the slot-6 wait for the device load
       * that feeds it -- including the quad butterfly, which this encoder
       * previously refused to give a wait at all.
       *
       * The field is seven bits (half 0xff spills the eighth into f16, the
       * source register), so lanes 0..127 encode and 128 up do not.
       */
      if (enc != AGX_ISA_ENCODING_AGX3B && !is_c(enc))
         return false;
      if (d->lane > 0x7f || (d->wait_mask >> slots) != 0)
         return false;

      struct agx3_field_kv ibase[10];
      unsigned inb = 0;
      /* 16-bit data only narrows the width pair: simd_broadcast(ushort, 5)
       * is 4704560202040a200400 next to the 32-bit words above. */
      ibase[inb++] = (struct agx3_field_kv){9, 2};
      ibase[inb++] = (struct agx3_field_kv){10, d->half ? 1u : 2u};
      ibase[inb++] = (struct agx3_field_kv){23, 2};
      ibase[inb++] = (struct agx3_field_kv){24, d->half ? 1u : 2u};
      /* One liveness selector for the whole family: f25 while the source
       * stays live, f27 on its final read.  Apple's simd shift witnesses
       * carry f25 (the shifted result lands in a different register and the
       * source is read again); its broadcasts carry f27. */
      ibase[inb++] = (struct agx3_field_kv){d->src_live ? 25 : 27, 1};
      ibase[inb++] = (struct agx3_field_kv){37, 1};
      ibase[inb++] = (struct agx3_field_kv){38, 1};

      if (is_c(enc)) {
         static const uint8_t ids[2][AGX3_SUBGROUP_SHUFFLE_NUM] = {
            {154,155,161,160},{140,137,142,141},
         };
         ibase[inb++] = (struct agx3_field_kv){42,3};
         ibase[inb++] = (struct agx3_field_kv){43,ids[d->quad][d->op]};
      }

      struct agx3_field_kv iov[14];
      unsigned inv = 0;
      iov[inv++] = (struct agx3_field_kv){2, d->dest};
      iov[inv++] = (struct agx3_field_kv){11, d->dest_hint ? 1u : 0u};
      iov[inv++] = (struct agx3_field_kv){16, d->src};
      iov[inv++] = (struct agx3_field_kv){29, d->lane};
      for (unsigned i = 0; i < slots; ++i)
         if (d->wait_mask & (1u << i))
            iov[inv++] = (struct agx3_field_kv){(uint16_t)(44 + i), 1};

      return agx3_bind_compose(enc, mnemonic, 0, ibase, inb, iov, inv, out,
                               length);
   }

   if (d->half || d->wait_mask != 0 ||
       (d->lane_kind != AGX3_SUBGROUP_LANE_REGISTER &&
        d->lane_kind != AGX3_SUBGROUP_LANE_UNIFORM) ||
       (d->lane & 1) != 0 ||
       (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER ? !valid_reg(enc, d->lane)
                                                    : d->lane >= 512))
      return false;

   struct agx3_field_kv base[24];
   unsigned nb = 0;
   ADD_FIELD(9, 2);
   ADD_FIELD(10, 2);
   ADD_FIELD(23, 2);
   ADD_FIELD(24, 2);
   ADD_FIELD(37, d->lane_kind);
   ADD_FIELD(38, 1);

   if (!is_c(enc)) {
      if (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER) {
         if (d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST) {
            ADD_FIELD(11, d->dest_hint);
            ADD_FIELD(d->src_live ? 25 : 27, 1);
            ADD_FIELD(d->lane_live ? 39 : 41, 1);
         } else if (d->op == AGX3_SUBGROUP_SHUFFLE_UP ||
                    d->op == AGX3_SUBGROUP_SHUFFLE_DOWN) {
            ADD_FIELD(11, 1);
            ADD_FIELD(25, 1);
            ADD_FIELD(39, 1);
         } else {
            ADD_FIELD(27, 1);
            ADD_FIELD(41, 1);
         }
      } else if (d->op == AGX3_SUBGROUP_SHUFFLE_UP ||
                 d->op == AGX3_SUBGROUP_SHUFFLE_DOWN) {
         ADD_FIELD(11, 1);
         ADD_FIELD(25, 1);
      } else {
         ADD_FIELD(27, 1);
      }
   } else {
      if (d->quad && d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST) {
         ADD_FIELD(11, d->dest_hint);
         ADD_FIELD(d->src_live ? 25 : 27, 1);
      } else if (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER && !d->quad &&
                 d->op == AGX3_SUBGROUP_SHUFFLE_BROADCAST) {
         ADD_FIELD(11, 1);
         ADD_FIELD(25, 1);
      } else {
         ADD_FIELD(27, 1);
      }
      if (d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER)
         ADD_FIELD(d->quad && d->lane_live ? 39 : 41, 1);
      static const uint8_t ids[2][AGX3_SUBGROUP_SHUFFLE_NUM] = {
         {154, 155, 161, 160},
         {140, 137, 142, 141},
      };
      ADD_FIELD(42, 3);
      ADD_FIELD(43, ids[d->quad][d->op]);
   }

   struct agx3_field_kv ov[] = {
      {2, d->dest},
      {16, d->src},
      {d->lane_kind == AGX3_SUBGROUP_LANE_REGISTER ? 30 : 31, d->lane},
   };
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, ov, 3, out, length);
}

bool
agx3_subgroup_encode_shuffle(enum agx_isa_encoding enc,
                             const struct agx3_subgroup_shuffle_desc *d,
                             uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                             uint8_t *length)
{
   return encode_shuffle(enc, d, false, out, length);
}

/* G18G only (AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d afterwards): AGX3c
 * quad shuffles from a register lane take the register-lane form with the
 * quad selectors. */
bool
agx3_subgroup_encode_shuffle_g18g(enum agx_isa_encoding enc,
                                  const struct agx3_subgroup_shuffle_desc *d,
                                  uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                  uint8_t *length)
{
   return encode_shuffle(enc, d, true, out, length);
}

static bool
encode_ballot(enum agx_isa_encoding enc,
              const struct agx3_subgroup_ballot_desc *d, bool g18g,
              uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], uint8_t *length)
{
   if (d == NULL || out == NULL || length == NULL)
      return false;

   if (d->compare_register) {
      const bool is_float = d->type == AGX3_SUBGROUP_BALLOT_FLOAT;
      if (!is_c(enc) || d->quad || d->dest_half || d->compare_immediate ||
          (unsigned)d->type > AGX3_SUBGROUP_BALLOT_FLOAT ||
          (is_float ? (d->condition != 0 && d->condition != 1 &&
                       d->condition != 2 && d->condition != 5 && d->condition != 6)
                    : (d->condition != 1 && d->condition != 2 &&
                       d->condition != 4 && d->condition != 5 &&
                       d->condition != 6)) ||
          !valid_reg(enc, d->dest) || d->wait_mask > 255 ||
          (d->src_half ? d->src >= 256 : !valid_reg(enc, d->src)) ||
          (d->compare_half ? d->compare_value >= 256 :
                             !valid_reg(enc, d->compare_value)) ||
          (d->src_cache && !d->src_live) ||
          (d->compare_cache && d->compare_discard))
         return false;

      struct agx3_field_kv base[24];
      unsigned nb = 0;
      ADD_FIELD(9, 2);
      ADD_FIELD(10, 2);
      ADD_FIELD(11, d->dest_cache);
      ADD_FIELD(23, 2);
      ADD_FIELD(24, d->src_half ? 1 : 2);
      ADD_FIELD(25, d->src_cache);
      ADD_FIELD(27, !d->src_live);
      ADD_FIELD(is_float ? 40 : 38, 2);
      ADD_FIELD(is_float ? 41 : 39, d->compare_half ? 1 : 2);
      ADD_FIELD(is_float ? 42 : 40, d->compare_cache);
      ADD_FIELD(is_float ? 44 : 42, d->compare_discard);
      ADD_FIELD(is_float ? 48 : 44, 3);
      ADD_FIELD(is_float ? 49 : 45, is_float ? 156 : 157);
      for (unsigned bit = 0; bit < 8; bit++)
         if (d->wait_mask & (1u << bit))
            ADD_FIELD((is_float ? 50 : 46) + bit, 1);
      const struct agx3_field_kv ov[] = {
         {2, d->dest}, {16, d->src}, {is_float ? 33 : 31, d->compare_value},
         {is_float ? 58 : 54, d->condition}, {is_float ? 59 : 55, d->invert},
      };
      return agx3_bind_compose(enc, is_float ? AGX3_MN_simdfballot : AGX3_MN_simdiballot,
                               0, base, nb, ov, sizeof(ov) / sizeof(ov[0]),
                               out, length);
   }

   /*
    * The 16-bit quad ballot of a register against zero. metal-nt -arch
    * applegpu_g17p, quad_any(x != 0) and quad_all(x == 0) of a buffer-loaded
    * ushort, with the result fed on to dfdx():
    *
    *   x != 0, x dead   17025600030000402212
    *   x == 0, x dead   17025601030200402202
    *   x != 0, x live   17025600030202002212
    *
    * f25/f27 carry the source's liveness, f11 the destination hint and f46
    * the source wait. Apple adds f64 when the result goes nowhere but the
    * colour output; like f62 on the quad reductions, that stays clear.
    */
   if (d->quad && d->dest_half) {
      const unsigned slots = is_c(enc) ? 8 : 6;
      if ((enc != AGX_ISA_ENCODING_AGX3B && !is_c(enc)) ||
          d->type != AGX3_SUBGROUP_BALLOT_INTEGER || d->compare_immediate ||
          d->condition != 4 || d->dest >= (is_c(enc) ? 256 : 192) ||
          (d->src_half ? d->src >= (is_c(enc) ? 256 : 192) : !valid_reg(enc, d->src)) ||
          (d->wait_mask >> slots) != 0)
         return false;

      /* A uint source only widens f24: quad_any(x != 0) of a uint is
       * 17025600030000502212. */
      struct agx3_field_kv qbase[24];
      unsigned qnb = 0;
      qbase[qnb++] = (struct agx3_field_kv){9, 2};
      qbase[qnb++] = (struct agx3_field_kv){10, 1};
      qbase[qnb++] = (struct agx3_field_kv){23, 2};
      qbase[qnb++] = (struct agx3_field_kv){24, d->src_half ? 1u : 2u};
      qbase[qnb++] = (struct agx3_field_kv){d->src_live ? 25 : 27, 1};
      qbase[qnb++] = (struct agx3_field_kv){38, 1};
      qbase[qnb++] = (struct agx3_field_kv){39, 1};
      if (is_c(enc)) {
         qbase[qnb++] = (struct agx3_field_kv){44, 3};
         qbase[qnb++] = (struct agx3_field_kv){45, 139};
      }
      for (unsigned i = 0; i < slots; ++i)
         if (d->wait_mask & (1u << i))
            qbase[qnb++] = (struct agx3_field_kv){(uint16_t)(46 + i), 1};

      const struct agx3_field_kv qov[] = {
         {2, d->dest},
         {11, d->dest_cache ? 1u : 0u},
         {16, d->src},
         {54, d->condition},
         {55, d->invert},
      };
      return agx3_bind_compose(enc, AGX3_MN_quadiballot, 0, qbase, qnb, qov,
                               sizeof(qov) / sizeof(qov[0]), out, length);
   }

   if ((!is_c(enc) && d->wait_mask != 0) || (is_c(enc) && d->wait_mask > 255) ||
       d->dest_half || !valid_reg(enc, d->dest) ||
       (unsigned)d->type > AGX3_SUBGROUP_BALLOT_FLOAT)
      return false;

   /* Only the witnessed 16-bit source shape: equal to zero. */
   if (d->src_half ? !d->compare_immediate || d->condition != 4 ||
                        d->compare_value != 0 || d->src >= (is_c(enc) ? 256 : 192)
                   : !valid_reg(enc, d->src))
      return false;

   if (d->compare_immediate) {
      /* Fresh applegpu-nt G17P witnesses use this exact form for unsigned
       * equality/less-than/greater-than against immediates.  The latter is
       * how Apple canonicalizes >= N (as > N - 1). */
      if ((enc != AGX_ISA_ENCODING_AGX3B && !is_c(enc)) || d->quad ||
          d->type != AGX3_SUBGROUP_BALLOT_INTEGER ||
          (d->condition != 1 && d->condition != 2 && d->condition != 4) ||
          (d->invert && !is_c(enc)))
         return false;
   } else if (d->type == AGX3_SUBGROUP_BALLOT_INTEGER) {
      if (d->condition != 4 && d->condition != 5 && d->condition != 6)
         return false;
   } else if (d->condition != 0 && d->condition != 1 && d->condition != 2 &&
              d->condition != 5 && d->condition != 6) {
      return false;
   }

   struct agx3_field_kv base[24];
   unsigned nb = 0;
   ADD_FIELD(9, 2);
   ADD_FIELD(10, 2);
   if (d->quad || d->dest_cache)
      ADD_FIELD(11, 1);
   ADD_FIELD(23, 2);
   ADD_FIELD(24, d->src_half ? 1 : 2);
   /* G17P simd_ballot(x != 0), metal-nt 32023.917:
    *   dead x: 17075600020000582212
    *   live x: 17075600020400182212
    * The live witness reads half 2 instead of half 0 and clears f27.
    * This source discard bit also applies to our register-zero comparison.
    * In restart unrolling, its input is a shared ballot mask that is still
    * needed by the broadcast selecting the first nonempty subgroup.
    */
   const bool live_integer_ballot =
      enc == AGX_ISA_ENCODING_AGX3B && !d->quad &&
      d->type == AGX3_SUBGROUP_BALLOT_INTEGER && d->src_live;
   if ((!d->compare_immediate && !live_integer_ballot) || !d->src_live)
      ADD_FIELD(27, 1);
   /* G18G: AGX3c's immediate form marks a source that stays live with f25:
    * simd_ballot(x != 0) + x is f25 where the dead-x form has f27
    * (m6-libagx-forms2 lk_ballot_ne16_live / ne32_live). */
   if (g18g && is_c(enc) && d->compare_immediate && d->src_live)
      ADD_FIELD(25, 1);

   enum agx3_mnemonic mnemonic;
   uint16_t cond_field, invert_field;
   if (d->type == AGX3_SUBGROUP_BALLOT_INTEGER) {
      mnemonic = d->quad ? AGX3_MN_quadiballot : AGX3_MN_simdiballot;
      if (d->compare_immediate) {
         ADD_FIELD(38, 1);
         ADD_FIELD(39, 1);
         if (!is_c(enc))
            ADD_FIELD(46, 1);
      } else if (d->condition == 4) {
         ADD_FIELD(38, 2);
         ADD_FIELD(39, 2);
         ADD_FIELD(42, 1);
      } else {
         ADD_FIELD(38, 1);
         ADD_FIELD(39, 1);
      }
      cond_field = 54;
      invert_field = 55;
      if (is_c(enc)) {
         ADD_FIELD(44, 3);
         ADD_FIELD(45, d->quad ? 139 : 157);
      }
   } else {
      mnemonic = d->quad ? AGX3_MN_quadfballot : AGX3_MN_simdfballot;
      ADD_FIELD(40, 1);
      ADD_FIELD(41, 2);
      cond_field = 58;
      invert_field = 59;
      if (is_c(enc)) {
         ADD_FIELD(48, 3);
         ADD_FIELD(49, d->quad ? 138 : 156);
      }
   }

   if (is_c(enc)) {
      /* G18G keeps its quad and float ballots, which carry no wait. */
      if ((d->quad || d->type != AGX3_SUBGROUP_BALLOT_INTEGER) &&
          (!g18g || d->wait_mask))
         return false;
      for (unsigned i = 0; i < 8; ++i)
         if (d->wait_mask & (1u << i))
            ADD_FIELD(46 + i,1);
   }

   struct agx3_field_kv ov[5] = {
      {2, d->dest},
      {16, d->src},
      {cond_field, d->condition},
      {invert_field, d->invert},
   };
   unsigned no = 4;
   if (d->compare_immediate)
      ov[no++] = (struct agx3_field_kv){30, d->compare_value};
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, ov, no, out, length);
}

bool
agx3_subgroup_encode_ballot(enum agx_isa_encoding enc,
                            const struct agx3_subgroup_ballot_desc *d,
                            uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                            uint8_t *length)
{
   return encode_ballot(enc, d, false, out, length);
}

/* G18G only (AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d afterwards): the
 * AGX3c immediate ballot of a source that stays live carries f25 as the
 * G18G layer emits it. */
bool
agx3_subgroup_encode_ballot_g18g(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_ballot_desc *d,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length)
{
   return encode_ballot(enc, d, true, out, length);
}

/*
 * simd_ballot(true). agx_lower_pseudo() spells a ballot of a boolean as
 * `icmp_ballot b, #0, ueq, invert`, and a constant-true predicate leaves
 * `icmp_ballot #1, #0, ueq, invert` -- both operands immediate, which the
 * register forms above cannot take. Apple compiles simd_ballot(true) to one
 * instruction whose bytes are identical for applegpu_g17g and applegpu_g18g
 * (g18ref-out/probe/m6-alu-misc, ballot_true):
 *
 *   simdiballot r0 = ballot(!(#1 ?0 #0))   27 01 20 1a 25 00 ae 02 30 00 08 40
 *
 * slot A the immediate 1 (f15, kind f23 = 1, width f24 = 1), slot B the
 * immediate 0 (f38/f39 = 1), f54 = 0 and the invert bit f55 set. That is the
 * very instruction Mesa's operands name, so it is emitted as witnessed, with
 * only the destination (and its f11 cache hint, the same field as on the
 * other simdiballot forms) varying. The witness carries no wait, and the
 * caller retires any before.
 */
bool
agx3_subgroup_encode_ballot_true_gpu1(uint16_t dest, bool dest_cache,
                                      uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                      uint8_t *length)
{
   if (out == NULL || length == NULL ||
       !valid_reg(AGX_ISA_ENCODING_AGX3C, dest))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {15, 1}, {23, 1}, {24, 1}, {38, 1}, {39, 1},
      {44, 3}, {45, 157}, {55, 1},
   };
   const struct agx3_field_kv ov[] = {
      {2, dest},
      {11, dest_cache ? 1u : 0u},
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_simdiballot, 0, base,
                            sizeof(base) / sizeof(base[0]), ov,
                            sizeof(ov) / sizeof(ov[0]), out, length);
}

/*
 * simd_ballot(a < b) of two 32-bit registers, unsigned: slot B is a register
 * (f31, kind f38 = 2, width f39 = 2) and f54 is the condition, 1 = ult as in
 * enum agx_icond. applegpu_g18g (g18ref-out/probe/m6-alu-misc ballot_ult and
 * m6-alu-hints h_ballot_ult_live), both operands read for the last time:
 *
 *   simdiballot r0 = ^r0 < ^r2          f16 0, f27, f31 4, f42, f54 = 1
 *   simdiballot r2 = ^r2 < ^r3          f16 4, f27, f31 6, f42, f54 = 1
 *
 * Only that lifetime shape is witnessed (Apple multiplies first when the
 * operands stay live), so live operands are refused, as is every other
 * condition.
 */
bool
agx3_subgroup_encode_ballot_ult_reg_gpu1(
   uint16_t dest, bool dest_cache, uint16_t src0, uint16_t src1,
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], uint8_t *length)
{
   if (out == NULL || length == NULL ||
       !valid_reg(AGX_ISA_ENCODING_AGX3C, dest) ||
       !valid_reg(AGX_ISA_ENCODING_AGX3C, src0) ||
       !valid_reg(AGX_ISA_ENCODING_AGX3C, src1))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, 2}, {27, 1}, {38, 2}, {39, 2},
      {42, 1}, {44, 3}, {45, 157}, {54, 1},
   };
   const struct agx3_field_kv ov[] = {
      {2, dest},
      {11, dest_cache ? 1u : 0u},
      {16, src0},
      {31, src1},
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_simdiballot, 0, base,
                            sizeof(base) / sizeof(base[0]), ov,
                            sizeof(ov) / sizeof(ov[0]), out, length);
}

struct reduce_form {
   enum agx3_mnemonic simd;
   enum agx3_mnemonic quad;
   uint8_t c_simd;
   uint8_t c_quad;
   bool is_float;
   uint8_t integer_class;
};

static const struct reduce_form reduce_forms[AGX3_SUBGROUP_REDUCE_NUM] = {
   [AGX3_SUBGROUP_REDUCE_AND] = {AGX3_MN_sRedAND, AGX3_MN_qRedAND, 169, 123, false, 2},
   [AGX3_SUBGROUP_REDUCE_FADD] = {AGX3_MN_sRedf32add, AGX3_MN_qRedf32add, 172, 126, true,
                                  0},
   [AGX3_SUBGROUP_REDUCE_OR] = {AGX3_MN_sRedOR, AGX3_MN_qRedOR, 177, 131, false, 2},
   [AGX3_SUBGROUP_REDUCE_FMUL] = {AGX3_MN_sRedf32mul, AGX3_MN_qRedf32mul, 173, 127, true,
                                  0},
   [AGX3_SUBGROUP_REDUCE_XOR] = {AGX3_MN_sRedXOR, AGX3_MN_qRedXOR, 182, 136, false, 2},
   [AGX3_SUBGROUP_REDUCE_FMIN] = {AGX3_MN_sRedfmin, AGX3_MN_qRedfmin, 175, 129, true, 0},
   [AGX3_SUBGROUP_REDUCE_FMAX] = {AGX3_MN_sRedfmax, AGX3_MN_qRedfmax, 174, 128, true, 0},
   [AGX3_SUBGROUP_REDUCE_IADD] = {AGX3_MN_sRediadd, AGX3_MN_qRediadd, 176, 130, false, 0},
   [AGX3_SUBGROUP_REDUCE_SMIN] = {AGX3_MN_sRedsmin, AGX3_MN_qRedsmin, 179, 133, false, 1},
   [AGX3_SUBGROUP_REDUCE_SMAX] = {AGX3_MN_sRedsmax, AGX3_MN_qRedsmax, 178, 132, false, 1},
   [AGX3_SUBGROUP_REDUCE_UMIN] = {AGX3_MN_sRedumin, AGX3_MN_qRedumin, 181, 135, false, 2},
   [AGX3_SUBGROUP_REDUCE_UMAX] = {AGX3_MN_sRedumax, AGX3_MN_qRedumax, 180, 134, false, 2},
};

static bool
encode_reduce(enum agx_isa_encoding enc,
              const struct agx3_subgroup_reduce_desc *d, bool g18g,
              uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], uint8_t *length)
{
   if (d == NULL || out == NULL || length == NULL ||
       (unsigned)d->op >= AGX3_SUBGROUP_REDUCE_NUM)
      return false;
   const unsigned reg_limit = is_c(enc) ? 256 : 192;
   if (d->half ? d->dest >= reg_limit || d->src >= reg_limit
               : !valid_reg(enc, d->dest) || !valid_reg(enc, d->src))
      return false;

   const struct reduce_form *f = &reduce_forms[d->op];
   /*
    * Half-precision add and multiply are their own instructions, laid out
    * like the f32 forms with every field from the source on shifted down by
    * one. metal-nt -arch applegpu_g17p:
    *
    *   quad_sum(half), result reduced again  3704560003008808  qRedf16add
    *   simd_prefix_exclusive_product(half)   bf04560202048818  sRedf16mul
    *   simd_prefix_inclusive_product(half)   bf04560303048a10  sRedf16mul
    */
   const bool f16_arith = d->half && (d->op == AGX3_SUBGROUP_REDUCE_FADD ||
                                      d->op == AGX3_SUBGROUP_REDUCE_FMUL);
   enum agx3_mnemonic mnemonic;
   if (f16_arith) {
      const bool add = d->op == AGX3_SUBGROUP_REDUCE_FADD;
      mnemonic = d->quad ? (add ? AGX3_MN_qRedf16add : AGX3_MN_qRedf16mul)
                         : (add ? AGX3_MN_sRedf16add : AGX3_MN_sRedf16mul);
   } else {
      mnemonic = d->quad ? f->quad : f->simd;
   }
   if (mnemonic == AGX3_MN_NONE)
      return false;
   /*
    * Metal has scans for sums and products only, so those are the witnessed
    * prefix forms: f53 on the integer add, f57 on the f32 and f56 on the f16
    * instructions. The bitwise and min/max instructions carry the same
    * operand in the encoder tables -- f53 on signed min/max, whose fields
    * are numbered like the integer add, and f52 on the bitwise and unsigned
    * forms, which lack the signedness field f28 and number everything above
    * it one lower -- exactly as the AGX2 SIMD instructions take every
    * reduction as a scan. Apple's compiler has no source-level scan to emit
    * them from (it reports air.simd_prefix_*_min and friends as unlowered on
    * every applegpu target), so those follow the tables.
    */
   if (d->prefix && d->quad)
      return false;
   if (d->prefix && f->is_float && d->op != AGX3_SUBGROUP_REDUCE_FADD &&
       d->op != AGX3_SUBGROUP_REDUCE_FMUL)
      return false;

   struct agx3_field_kv base[24];
   unsigned nb = 0;
   ADD_FIELD(9, 2);
   ADD_FIELD(10, d->half ? 1 : 2);
   uint16_t src_field, prefix_field, wait_base;
   if (f16_arith) {
      ADD_FIELD(24, 2);
      ADD_FIELD(25, 1);
      ADD_FIELD(d->src_live ? 26 : 28, 1);
      if (is_c(enc)) {
         ADD_FIELD(46, 3);
         ADD_FIELD(47, d->quad ? (d->op == AGX3_SUBGROUP_REDUCE_FADD ? 124 : 125)
                              : (d->op == AGX3_SUBGROUP_REDUCE_FADD ? 170 : 171));
      } else {
         ADD_FIELD(33, 1);
         ADD_FIELD(41, 1);
         ADD_FIELD(42, 1);
      }
      src_field = 17;
      prefix_field = 56;
      wait_base = 48;
   } else if (f->is_float) {
      ADD_FIELD(25, 2);
      ADD_FIELD(26, d->half ? 1 : 2);
      if (!d->src_live)
         ADD_FIELD(29, 1);
      if (is_c(enc)) {
         ADD_FIELD(47, 3);
         ADD_FIELD(48, d->quad ? f->c_quad : f->c_simd);
      } else {
         ADD_FIELD(34, 1);
         ADD_FIELD(42, 1);
         ADD_FIELD(43, 1);
      }
      src_field = 18;
      prefix_field = 57;
      wait_base = 49;
   } else {
      ADD_FIELD(23, 2);
      ADD_FIELD(24, d->half ? 1 : 2);
      /* G17P picks f25 while the source stays live and f27 on its final
       * read, for SIMD and quad forms alike: simd_prefix_inclusive_sum(),
       * which adds the source back afterwards, is bf01560403001609 (f25)
       * next to the exclusive bf0156020204100b (f27). */
      if (g18g && is_c(enc)) {
         /* AGX3c's SIMD integer add does the same: metal-nt
          * applegpu_g17g/g18g give the inclusive sum f25 and the exclusive
          * one f27 (m6-libagx-forms lk_prefix_incl_load /
          * lk_prefix_excl_load). */
         const bool c_simd_iadd = !d->quad && !d->half &&
                                  d->op == AGX3_SUBGROUP_REDUCE_IADD;
         /* Native999's live/dead pairs on both G17G and G18G use no
          * source hint for a live full-width min/max/bitwise operand. In
          * particular, f27 is a final-read hint, not the generic AGX3c
          * reduction default. Preserve the existing IADD/quad/half paths. */
         const bool c_keep_source = !d->quad && !d->half && d->src_live &&
            (d->op == AGX3_SUBGROUP_REDUCE_SMIN ||
             d->op == AGX3_SUBGROUP_REDUCE_SMAX ||
             d->op == AGX3_SUBGROUP_REDUCE_AND ||
             d->op == AGX3_SUBGROUP_REDUCE_OR ||
             d->op == AGX3_SUBGROUP_REDUCE_XOR ||
             d->op == AGX3_SUBGROUP_REDUCE_UMIN ||
             d->op == AGX3_SUBGROUP_REDUCE_UMAX);
         if (!c_keep_source)
            ADD_FIELD(c_simd_iadd && d->src_live ? 25 : 27, 1);
      } else {
         ADD_FIELD(d->src_live ? 25 : 27, 1);
      }
      if (f->integer_class == 1)
         ADD_FIELD(28, 1);
      if (is_c(enc)) {
         if (f->integer_class == 2) {
            ADD_FIELD(42, 3);
            ADD_FIELD(43, d->quad ? f->c_quad : f->c_simd);
         } else {
            ADD_FIELD(43, 3);
            ADD_FIELD(44, d->quad ? f->c_quad : f->c_simd);
         }
      } else if (f->integer_class == 2) {
         ADD_FIELD(29, 1);
         ADD_FIELD(37, 1);
         ADD_FIELD(38, 1);
      } else {
         ADD_FIELD(30, 1);
         ADD_FIELD(38, 1);
         ADD_FIELD(39, 1);
      }
      src_field = 16;
      prefix_field = f->integer_class == 2 ? 52 : 53;
      wait_base = f->integer_class == 2 ? 44 : 45;
   }

   if (d->prefix)
      ADD_FIELD(prefix_field, 1);

   /*
    * Per-slot wait bits. Their base differs by operand family, each pinned by
    * a g17p probe whose source arrives on scoreboard slot 6:
    *   float reductions            f49 + (slot - 1)   (sRedf32add, sRedfmax)
    *   half add / multiply         f48 + (slot - 1)   (qRedf16add, sRedf16mul)
    *   integer add / signed min-max
    *                               f45 + (slot - 1)   (sRediadd, sRedimin,
    *                                                   sRedimax; also catomic
    *                                                   at slot 1)
    *   bitwise / unsigned min-max  f44 + (slot - 1)   (sRedAND, sRedumax)
    * Own-shader A18 Pro captures pin slot-6 signed min/max to this family:
    *   sRedimin 3f02560002001407, sRedimax bf02560002001407.
    */
   /* AGX3c: the SIMD integer add waits at f45 + (slot - 1) on all eight
    * slots, as on AGX3b. metal-nt applegpu_g17g puts a load on slot 8 and
    * waits with f52; applegpu_g18g puts it on slot 1 and waits with f45
    * (m6-libagx-waits2 lw2_prefix_*, m6-libagx-forms lk_local_atomic_*). */
   const unsigned slots = is_c(enc) ? 8 : 6;
   if (d->wait_mask != 0) {
      if ((d->wait_mask >> slots) != 0)
         return false;
      for (unsigned i = 0; i < slots; ++i)
         if (d->wait_mask & (1u << i))
            ADD_FIELD((uint16_t)(wait_base + i), 1);
   }

   struct agx3_field_kv ov[4];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv){2, d->dest};
   ov[n++] = (struct agx3_field_kv){src_field, d->src};
   ov[n++] = (struct agx3_field_kv){11, d->dest_hint ? 1u : 0u};
   if (f->is_float && !f16_arith && d->src_live)
      ov[n++] = (struct agx3_field_kv){27, 1};
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, ov, n, out, length);
}

bool
agx3_subgroup_encode_reduce(enum agx_isa_encoding enc,
                            const struct agx3_subgroup_reduce_desc *d,
                            uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                            uint8_t *length)
{
   return encode_reduce(enc, d, false, out, length);
}

/* G18G only (AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d afterwards): the
 * source-lifetime hints of the AGX3c SIMD integer reductions as the G18G
 * layer emits them (native999). */
bool
agx3_subgroup_encode_reduce_g18g(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_reduce_desc *d,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length)
{
   return encode_reduce(enc, d, true, out, length);
}

#undef ADD_FIELD
