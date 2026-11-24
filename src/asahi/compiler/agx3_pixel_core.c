/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_pixel_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}
#define NELEM(a) (sizeof(a) / sizeof((a)[0]))

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

/*
 * G18G (agx3d) keeps pixwait and pixrel bit for bit: all 212 aligned
 * applegpu_g17g/applegpu_g18g pixwait words and all 106 pixrel words of the
 * j813-linux gpu/m5-*-reference corpus are byte-identical, and so are the
 * oracle and glxgears fragment epilogues (07 30 00 05 00 00, 67 40 00 05 00 00,
 * 67 10 00 04 00 00). Their wait field stays a mask on agx3d.
 */
static bool
enc_ok(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B || agx_encoding_is_g17_class(e);
}

static unsigned
nslots(enum agx_isa_encoding e)
{
   return is_b(e) ? 6 : 8;
}

static bool
wait_mask_ok(enum agx_isa_encoding e, uint16_t wait_mask)
{
   return (wait_mask & ~(uint16_t)((1u << nslots(e)) - 1)) == 0;
}

/*
 * The scoreboard wait bits are consecutive fields starting at `first`; pixwait
 * numbers them from f18 and pixrel from f17, because pixwait carries the extra
 * f15 gate bit ahead of them.
 */
static unsigned
wait_overrides(enum agx_isa_encoding e, uint16_t wait_mask, unsigned first,
               struct agx3_field_kv *ov, unsigned n)
{
   for (unsigned i = 0; i < nslots(e); ++i) {
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(first + i, 1);
   }
   return n;
}

static bool
in_set(uint32_t value, const uint32_t *set, unsigned n)
{
   for (unsigned i = 0; i < n; ++i) {
      if (set[i] == value)
         return true;
   }
   return false;
}

bool
agx3_pixel_colour_entry(enum agx_isa_encoding enc,
                        uint8_t out[AGX3_PIXEL_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
   *len = 0;

   /* G17P is AGX3b. The pair below is Apple's normalized composer input for
    * the exact 87 02 54 00 06 00 entry gate. */
   if (!enc_ok(enc))
      return false;

   const struct agx3_field_kv fields[] = {KV(14, 512), KV(15, 1)};
   const struct agx3_field_kv base_c[] = {KV(16, 3), KV(17, 122)};
   if (!agx3_bind_compose(enc, AGX3_MN_pixwait, agx3_bind_ep(enc, 0), fields,
                          NELEM(fields), is_b(enc) ? NULL : base_c,
                          is_b(enc) ? 0 : NELEM(base_c),
                          out, len)) {
      memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
      *len = 0;
      return false;
   }

   return true;
}

bool
agx3_pixel_colour_entry_zs_unreleased(enum agx_isa_encoding enc,
                                      uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                                      uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3C)
      return false;

   const struct agx3_field_kv fields[] = {KV(14, 512), KV(15, 1), KV(34, 1)};
   const struct agx3_field_kv base_c[] = {KV(16, 3), KV(17, 122)};
   if (!agx3_bind_compose(enc, AGX3_MN_pixwait, 0, fields, NELEM(fields), base_c,
                          NELEM(base_c), out, len)) {
      memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

/* metal-nt -arch applegpu_g17p, a fragment function with no colour writes
 * storing the result of atomic_fetch_add_explicit():
 *
 *   870254040000                  pixwait, class 4
 *   6721540000810000004600002000  atomicdev
 *   2c80192160800000              rld of the returned value
 *   070454c40800                  barrier
 *
 * The same function discarding the result emits the atomicdev alone. */
bool
agx3_pixel_atomic_acquire(enum agx_isa_encoding enc,
                          uint8_t out[AGX3_PIXEL_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
   *len = 0;
   if (!enc_ok(enc))
      return false;

   const struct agx3_field_kv fields[] = {KV(14, 4), KV(34, 0)};
   const struct agx3_field_kv base_c[] = {KV(16, 3), KV(17, 122)};
   if (!agx3_bind_compose(enc, AGX3_MN_pixwait, 0,
                          is_b(enc) ? fields : base_c,
                          is_b(enc) ? 1 : NELEM(base_c),
                          is_b(enc) ? NULL : fields,
                          is_b(enc) ? 0 : NELEM(fields), out, len)) {
      memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_pixel_atomic_barrier(enum agx_isa_encoding enc,
                          uint8_t out[AGX3_PIXEL_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
   *len = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B)
      return false;

   const struct agx3_field_kv fields[] = {KV(24, 1), KV(30, 3), KV(39, 1)};
   if (!agx3_bind_compose(enc, AGX3_MN_barrier, 0, fields, NELEM(fields), NULL, 0,
                          out, len) ||
       *len != AGX3_PIXEL_MAX_LENGTH) {
      memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_pixel_colour_partition_classes(unsigned partition,
                                    uint32_t *read_class,
                                    uint32_t *write_class,
                                    uint32_t *combined_class)
{
   if (partition >= 3 || read_class == NULL || write_class == NULL ||
       combined_class == NULL)
      return false;

   const uint32_t shift = 2 * partition;
   *read_class = 8u << shift;
   *write_class = 4u << shift;
   *combined_class = 12u << shift;
   return true;
}

bool
agx3_pixel_wait(enum agx_isa_encoding enc, uint32_t writeout,
                uint16_t wait_mask, uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                uint8_t *len)
{
   /* G17P/G17G sparse 4x MRT adds 204 (RT0+RT2) and 240 (RT1+RT2),
    * with matching final releases. Keep their single native instruction. */
   /* Every writeout mask Apple's fragment corpus emitted on pixwait, less the
    * f15-gated shader-entry form (512). */
   static const uint32_t witnessed[] = {1,   4,   8,   12, 16, 32, 48,  60, 64,
                                        128, 192, 252, 13, 49, 61, 193, 253};

   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
   *len = 0;

   if (!enc_ok(enc) || !wait_mask_ok(enc, wait_mask) ||
       !(in_set(writeout, witnessed, NELEM(witnessed)) ||
         (writeout == 204 || writeout == 240)))
      return false;

   /* f34 separates the colour writeout classes from the depth/coverage class:
    * set on every witnessed mask except 1. */
   const struct agx3_field_kv base_c[] = {KV(16, 3), KV(17, 122)};

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(14, writeout);
   ov[n++] = (struct agx3_field_kv)KV(34, writeout == 1 ? 0 : 1);
   n = wait_overrides(enc, wait_mask, 18, ov, n);

   if (!agx3_bind_compose(enc, AGX3_MN_pixwait, agx3_bind_ep(enc, 0),
                          is_b(enc) ? NULL : base_c,
                          is_b(enc) ? 0 : NELEM(base_c), ov, n, out,
                          len)) {
      memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_pixel_release(enum agx_isa_encoding enc, uint32_t writeout, bool last,
                   uint16_t wait_mask, uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                   uint8_t *len)
{
   /* The witnessed pixrel descriptors are exactly {0,1,12,48,60,192,252}
    * with an optional 0x200 "final release" bit. */
   static const uint32_t witnessed[] = {0, 1, 12, 48, 60, 192, 252};

   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
   *len = 0;

   if (!enc_ok(enc) || !wait_mask_ok(enc, wait_mask) ||
       !(in_set(writeout, witnessed, NELEM(witnessed)) ||
         (enc == AGX_ISA_ENCODING_AGX3C && writeout == 4) ||
         (writeout == 204 || writeout == 240)))
      return false;

   const struct agx3_field_kv base_c[] = {KV(15, 1), KV(16, 121)};

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(14, writeout | (last ? 0x200u : 0u));
   n = wait_overrides(enc, wait_mask, 17, ov, n);

   if (!agx3_bind_compose(enc, AGX3_MN_pixrel, agx3_bind_ep(enc, 0),
                          is_b(enc) ? NULL : base_c,
                          is_b(enc) ? 0 : NELEM(base_c), ov, n, out,
                          len)) {
      memset(out, 0, AGX3_PIXEL_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}
