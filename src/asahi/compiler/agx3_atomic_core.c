/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_atomic_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
encoding_ok(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B ||
          enc == AGX_ISA_ENCODING_AGX3C;
}
/* Scoreboard slots are machine behaviour: eight on every G17-class part.
 * Device atomics retain the agx3b/agx3c encoding_ok boundary. */
static unsigned
slot_count(enum agx_isa_encoding enc)
{
   return agx_encoding_is_g17_class(enc) ? 8 : 6;
}

static bool
op_ok(enum agx3_atomic_op op)
{
   return op >= AGX3_ATOMIC_ADD && op <= AGX3_ATOMIC_FADD;
}

static bool
standalone_return_form(const struct agx3_device_atomic_desc *d)
{
   return d->op == AGX3_ATOMIC_XCHG ||
          (d->returns_value && d->op >= AGX3_ATOMIC_UMIN &&
           d->op <= AGX3_ATOMIC_IMAX);
}

static unsigned
append_waits(uint16_t wait_mask, struct agx3_field_kv *fields, unsigned n)
{
   for (unsigned i = 0; i < 8; ++i) {
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv)KV(76 + i, 1);
   }

   return n;
}

static bool
compose_first(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
              const struct agx3_field_kv *fields, unsigned nfields,
              uint8_t out[AGX3_ATOMIC_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3D)
      return agx3_bind_compose(enc, mnemonic, AGX3_BIND_APPLE_ORDER,
                              fields, nfields, NULL, 0, out, len);

   /* The encoding points to try are a per-table layout: twelve agx3c
    * atomicdev/atomictg forms, one agx3b form, none elsewhere. */
   const unsigned count = enc == AGX_ISA_ENCODING_AGX3C   ? 12
                          : enc == AGX_ISA_ENCODING_AGX3B ? 1
                                                          : 0;

   for (unsigned ep = 0; ep < count; ++ep) {
      if (agx3_bind_compose(enc, mnemonic, ep, fields, nfields, NULL, 0,
                            out, len))
         return true;
   }

   return false;
}

bool
agx3_atomic_encode_device(enum agx_isa_encoding enc,
                          const struct agx3_device_atomic_desc *d,
                          uint8_t out[AGX3_ATOMIC_MAX_LENGTH], uint8_t *len)
{
   if (!encoding_ok(enc) || d == NULL || out == NULL || len == NULL ||
       !op_ok(d->op) || d->value > 191 || d->result > 191 ||
       d->index_shift > 4 || d->byte_offset < -32768 ||
       d->byte_offset > 32767 ||
       (d->wait_mask & ~((1u << slot_count(enc)) - 1u)))
      return false;

   if (d->returns_value) {
      if (d->slot < 1 || d->slot > slot_count(enc))
         return false;
      if (enc == AGX_ISA_ENCODING_AGX3B && d->result != d->value)
         return false;
      if ((d->result_tps_half & 1) || d->result_tps_half > 0x1fffe ||
          (enc != AGX_ISA_ENCODING_AGX3B && d->result_tps_half))
         return false;
   } else if (d->slot != 0 || d->result_tps_half) {
      return false;
   }

   if (d->base_kind == AGX3_ATOMIC_BASE_UNIFORM) {
      if ((d->base & 3) != 0 || d->base > 508)
         return false;
   } else if (d->base_kind == AGX3_ATOMIC_BASE_REGISTER) {
      if ((d->base & 1) != 0 || d->base > 188)
         return false;
   } else {
      return false;
   }

   switch (d->index_kind) {
   case AGX3_ATOMIC_INDEX_IMMEDIATE:
      break;
   case AGX3_ATOMIC_INDEX_REGISTER:
      if ((d->index & 1) != 0 ||
          d->index >
             (enc == AGX_ISA_ENCODING_AGX3C ? 254 : 190))
         return false;
      break;
   case AGX3_ATOMIC_INDEX_UNIFORM:
      if (d->index > 511)
         return false;
      break;
   default:
      return false;
   }

   struct agx3_field_kv f[48];
   unsigned n = 0;

   /* Cross-generation constants and named roles. */
   f[n++] = (struct agx3_field_kv)KV(10, 2); /* witnessed 32-bit form */
   f[n++] = (struct agx3_field_kv)KV(16, d->value);
   f[n++] = (struct agx3_field_kv)KV(23, 2);
   f[n++] = (struct agx3_field_kv)KV(24, 2);
   f[n++] = (struct agx3_field_kv)KV(27, d->value_kill ? 1 : 0);
   f[n++] = (struct agx3_field_kv)KV(37, d->index_kind);
   f[n++] = (struct agx3_field_kv)KV(
      38, d->index_kind == AGX3_ATOMIC_INDEX_IMMEDIATE && d->index == 0
             ? 1
             : 2);
   f[n++] = (struct agx3_field_kv)KV(52, d->base_kind);
   f[n++] = (struct agx3_field_kv)KV(53, 3);
   f[n++] = (struct agx3_field_kv)KV(68, 1);
   f[n++] = (struct agx3_field_kv)KV(88, d->op);
   f[n++] = (struct agx3_field_kv)KV(90, 1);
   f[n++] = (struct agx3_field_kv)KV(92, d->index_shift);
   f[n++] =
      (struct agx3_field_kv)KV(93, (uint16_t)d->byte_offset);

   f[n++] = (struct agx3_field_kv)KV(
      d->base_kind == AGX3_ATOMIC_BASE_UNIFORM ? 46 : 45, d->base);
   f[n++] = (struct agx3_field_kv)KV(
      d->index_kind == AGX3_ATOMIC_INDEX_IMMEDIATE
         ? 29
         : (d->index_kind == AGX3_ATOMIC_INDEX_REGISTER ? 30 : 31),
      d->index);
   if (d->index_kill)
      f[n++] = (struct agx3_field_kv)KV(41, 1);

   /* The same physical field is also scoreboard slot 6/8.  Xchg always uses
    * the standalone family. Returned min/max join that family, while returned
    * cmpxchg uses the masked arithmetic family. Non-returning cmpxchg remains
    * independently witnessed in the standalone family. */
   if (standalone_return_form(d) ||
       (!d->returns_value && d->op == AGX3_ATOMIC_CMPXCHG))
      f[n++] = (struct agx3_field_kv)KV(
         enc == AGX_ISA_ENCODING_AGX3C ? 83 : 81, 1);

   if (d->returns_value) {
      f[n++] = (struct agx3_field_kv)KV(9,
         enc == AGX_ISA_ENCODING_AGX3C ? 2 : 4);
      f[n++] = (struct agx3_field_kv)KV(12, d->slot);
      if (enc == AGX_ISA_ENCODING_AGX3B)
         f[n++] = (struct agx3_field_kv)KV(4, d->result_tps_half);
      f[n++] = (struct agx3_field_kv)KV(89, 1);
      if (enc == AGX_ISA_ENCODING_AGX3C) {
         f[n++] = (struct agx3_field_kv)KV(2, d->result);
         f[n++] = (struct agx3_field_kv)KV(75, 1);
         f[n++] = (struct agx3_field_kv)KV(102, 1);
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C) {
      f[n++] = (struct agx3_field_kv)KV(59, 15);
      f[n++] = (struct agx3_field_kv)KV(67, 1);
      f[n++] = (struct agx3_field_kv)KV(73, 6);
      f[n++] = (struct agx3_field_kv)KV(
         74, d->op == AGX3_ATOMIC_CMPXCHG ? 3 : 1);
   }

   /* On AGX3b, f76 also selects the elected/partial-execution arithmetic
    * family. Native returned xchg/min/max keep their standalone form even
    * inside the same compiler-generated first-lane block. */
   const uint16_t waits =
      d->wait_mask |
      ((enc == AGX_ISA_ENCODING_AGX3B && d->exec_masked &&
        !standalone_return_form(d))
          ? 1
          : 0);
   n = append_waits(waits, f, n);
   return compose_first(enc, AGX3_MN_atomicdev, f, n, out, len);
}

/* g18g admits the G18G-witnessed returning form that kills its register
 * byte address; agx3d is G18G only, so it always does. */
static bool
encode_local(enum agx_isa_encoding enc, const struct agx3_local_atomic_desc *d,
             bool g18g, uint8_t out[AGX3_ATOMIC_MAX_LENGTH], uint8_t *len)
{
   g18g = g18g || enc == AGX_ISA_ENCODING_AGX3D;

   if ((!encoding_ok(enc) && enc != AGX_ISA_ENCODING_AGX3D) ||
       d == NULL || out == NULL || len == NULL ||
       !op_ok(d->op) || d->op == AGX3_ATOMIC_FADD || d->value > 191 ||
       (d->wait_mask & ~((1u << slot_count(enc)) - 1u)))
      return false;

   if (d->returns_value) {
      if (d->result > 191 || d->slot < 1 || d->slot > slot_count(enc) ||
          (d->address_kind != AGX3_ATOMIC_INDEX_IMMEDIATE &&
           d->address_kind != AGX3_ATOMIC_INDEX_REGISTER) ||
          (d->address_kind == AGX3_ATOMIC_INDEX_IMMEDIATE &&
           (d->address & 3) != 0) ||
          (d->address_kind == AGX3_ATOMIC_INDEX_REGISTER && d->address > 191) ||
          (d->address_kill &&
           (!g18g || enc == AGX_ISA_ENCODING_AGX3B ||
            d->address_kind != AGX3_ATOMIC_INDEX_REGISTER)))
         return false;
   } else {
      /* G17G-family (agx3c) also has a no-result form at an immediate byte
       * address. G17G/G18G programs selecting between two threadgroup
       * pointers establish the register byte-base form for all eleven
       * operations, with and without a result; resultless forms omit the
       * result registers and producer slot. */
      if (d->slot != 0)
         return false;

      if (d->address_kind == AGX3_ATOMIC_INDEX_IMMEDIATE) {
         if (d->address_kill || (d->address & 3) != 0)
            return false;
      } else if (d->address_kind == AGX3_ATOMIC_INDEX_REGISTER) {
         if (d->address > 191)
            return false;
      } else {
         return false;
      }
   }

   struct agx3_field_kv f[40];
   unsigned n = 0;
   f[n++] = (struct agx3_field_kv)KV(10, 2);
   f[n++] = (struct agx3_field_kv)KV(16, d->value);
   f[n++] = (struct agx3_field_kv)KV(23, 2);
   f[n++] = (struct agx3_field_kv)KV(24, 2);
   f[n++] = (struct agx3_field_kv)KV(27, d->value_kill ? 1 : 0);
   f[n++] = (struct agx3_field_kv)KV(37, 1);
   f[n++] = (struct agx3_field_kv)KV(38, 1);
   f[n++] = (struct agx3_field_kv)KV(53, 1);
   f[n++] = (struct agx3_field_kv)KV(68, 1);
   f[n++] = (struct agx3_field_kv)KV(84, d->op);
   f[n++] = (struct agx3_field_kv)KV(86, 1);

   if (d->address_kind == AGX3_ATOMIC_INDEX_REGISTER) {
      f[n++] = (struct agx3_field_kv)KV(52, 2);
      f[n++] = (struct agx3_field_kv)KV(45, d->address);
      if (d->address_kill)
         f[n++] = (struct agx3_field_kv)KV(56, 1);
   } else {
      f[n++] = (struct agx3_field_kv)KV(52, 1);
      f[n++] = (struct agx3_field_kv)KV(89, d->address);
   }

   if (d->returns_value) {
      f[n++] = (struct agx3_field_kv)KV(2, d->result);
      f[n++] = (struct agx3_field_kv)KV(9, 2);
      f[n++] = (struct agx3_field_kv)KV(12, d->slot);
      f[n++] = (struct agx3_field_kv)KV(85, 1);
   }

   if ((d->op == AGX3_ATOMIC_XCHG || d->op == AGX3_ATOMIC_CMPXCHG) &&
       enc != AGX_ISA_ENCODING_AGX3D)
      f[n++] = (struct agx3_field_kv)KV(
         enc == AGX_ISA_ENCODING_AGX3C ? 83 : 81, 1);

   if (agx_encoding_is_g17_class(enc)) {
      if (d->returns_value)
         f[n++] = (struct agx3_field_kv)KV(11, 1);
      f[n++] = (struct agx3_field_kv)KV(59, 15);
      f[n++] = (struct agx3_field_kv)KV(67, 1);
      f[n++] = (struct agx3_field_kv)KV(73, 7);
      f[n++] = (struct agx3_field_kv)KV(
         74, d->op == AGX3_ATOMIC_CMPXCHG ? 3 : 1);
      if (d->returns_value)
         f[n++] = (struct agx3_field_kv)KV(75, 1);
      f[n++] = (struct agx3_field_kv)KV(95, 2);
      f[n++] = (struct agx3_field_kv)KV(98, 1);
   }

   n = append_waits(d->wait_mask, f, n);
   return compose_first(enc, AGX3_MN_atomictg, f, n, out, len);
}

bool
agx3_atomic_encode_local(enum agx_isa_encoding enc,
                         const struct agx3_local_atomic_desc *d,
                         uint8_t out[AGX3_ATOMIC_MAX_LENGTH], uint8_t *len)
{
   return encode_local(enc, d, false, out, len);
}

bool
agx3_atomic_encode_local_g18g(enum agx_isa_encoding enc,
                              const struct agx3_local_atomic_desc *d,
                              uint8_t out[AGX3_ATOMIC_MAX_LENGTH],
                              uint8_t *len)
{
   return encode_local(enc, d, true, out, len);
}
