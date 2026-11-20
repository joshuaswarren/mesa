/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 ray-traversal barrier.  See agx3_rt_core.h; the composer fails closed
 * on any contradiction, and the test replays the exact bytes for every
 * accepted scope.
 */

#include "agx3_rt_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}
#define NELEM(a) (sizeof(a) / sizeof((a)[0]))

static bool
enc_ok(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B || e == AGX_ISA_ENCODING_AGX3C;
}

static bool
args_ok(const uint8_t *out, const uint8_t *len)
{
   return out != NULL && len != NULL;
}

/*
 * Compose `mnemonic` from a fixed baseline plus overrides, zeroing the
 * caller's buffer on refusal so speculative callers stay transactional.
 */
static bool
rt_compose(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
           const struct agx3_field_kv *baseline, unsigned nbaseline,
           const struct agx3_field_kv *overrides, unsigned noverrides,
           uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len)
{
   if (!args_ok(out, len))
      return false;
   memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
   *len = 0;

   if (!enc_ok(enc))
      return false;

   if (!agx3_bind_compose(enc, mnemonic, 0, baseline, nbaseline, overrides,
                          noverrides, out, len)) {
      memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_rt_traversal_barrier(enum agx_isa_encoding enc,
                          enum agx3_rt_barrier_scope scope,
                          uint8_t out[AGX3_RT_CORE_MAX_LENGTH], uint8_t *len)
{
   static const struct agx3_field_kv b_base[] = {
      KV(25, 1), KV(30, 1), KV(39, 1),
   };
   static const struct agx3_field_kv c_base[] = {
      KV(14, 3), KV(15, 9), KV(25, 1), KV(30, 1), KV(39, 1),
   };

   if (!args_ok(out, len))
      return false;

   if (scope != AGX3_RT_BARRIER_IN_LOOP && scope != AGX3_RT_BARRIER_ENTRY) {
      memset(out, 0, AGX3_RT_CORE_MAX_LENGTH);
      *len = 0;
      return false;
   }

   const bool is_b = enc == AGX_ISA_ENCODING_AGX3B;
   const struct agx3_field_kv ov[] = {KV(30, scope)};
   return rt_compose(enc, AGX3_MN_barrier, is_b ? b_base : c_base,
                     is_b ? NELEM(b_base) : NELEM(c_base),
                     ov, NELEM(ov), out, len);
}
