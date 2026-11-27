/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_binder.h"

static const agx3_table *agx3d_table_or_null(void);

static const agx3_table *
agx3_bind_table(enum agx_isa_encoding encoding)
{
   if (encoding == AGX_ISA_ENCODING_AGX3B)
      return &agx3b_table;
   else if (encoding == AGX_ISA_ENCODING_AGX3C)
      return &agx3c_table;
   else if (encoding == AGX_ISA_ENCODING_AGX3D)
      return agx3d_table_or_null();
   else
      return NULL;
}

unsigned
agx3_bind_form_size(enum agx_isa_encoding encoding,
                    enum agx3_mnemonic mnemonic, int ep)
{
   const agx3_table *t = agx3_bind_table(encoding);
   const int32_t form = t ? agx3_form_index(t, mnemonic, (uint8_t)ep) : -1;
   return form < 0 ? 0 : t->forms[form].size;
}

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "c11/threads.h"
#include "util/bitscan.h"

/*
 * Only the builds that encode G18G define AGX3_HAVE_AGX3D_TABLE and so link
 * the agx3d table; everywhere else an AGX3D request fails closed here.
 */
static const agx3_table *
agx3d_table_or_null(void)
{
#ifdef AGX3_HAVE_AGX3D_TABLE
   return &agx3d_table;
#else
   return NULL;
#endif
}

/* Per-thread zeroed scratch vector: only the fields touched by baseline and
 * overrides are written and cleared back to zero before returning, avoiding
 * a 1.5 KiB memset on every composed instruction.
 *
 * Thread-specific storage rather than a _Thread_local array: libgallium and
 * libvulkan_asahi build with the initial-exec TLS model, and a dlopen() of
 * either only gets glibc's small static TLS surplus, which a 1280-byte
 * static TLS array exhausts (EGL then fails to load with "cannot allocate
 * memory in static TLS block"). */
static tss_t agx3_bind_scratch_key;
static once_flag agx3_bind_scratch_once = ONCE_FLAG_INIT;
static bool agx3_bind_scratch_key_valid;
/* Only an 8-byte pointer in static TLS (not the 1280-byte scratch array);
 * tss_set() owns cleanup on thread exit while hot lookups avoid call_once()
 * and tss_get() on every composed instruction. */
static _Thread_local uint64_t *agx3_bind_scratch_tls;

static void
agx3_bind_scratch_key_init(void)
{
   agx3_bind_scratch_key_valid =
      tss_create(&agx3_bind_scratch_key, free) == thrd_success;
}

static uint64_t *
agx3_bind_scratch_buffer(void)
{
   uint64_t *buf = agx3_bind_scratch_tls;
   if (buf)
      return buf;
   call_once(&agx3_bind_scratch_once, agx3_bind_scratch_key_init);
   if (!agx3_bind_scratch_key_valid)
      return NULL;
   buf = tss_get(agx3_bind_scratch_key);
   if (!buf) {
      buf = calloc(AGX3_MAX_FIELDS, sizeof(uint64_t));
      if (buf && tss_set(agx3_bind_scratch_key, buf) != thrd_success) {
         free(buf);
         buf = NULL;
      }
   }
   agx3_bind_scratch_tls = buf;
   return buf;
}

static void
agx3_bind_fields_clear_nonzero(uint64_t fields[AGX3_MAX_FIELDS],
                               const uint64_t nonzero[3])
{
   for (unsigned w = 0; w < 3; ++w) {
      uint64_t bits = nonzero[w];
      while (bits) {
         const unsigned field = w * 64 + u_bit_scan64(&bits);
         fields[field] = 0;
      }
   }
}

static bool
agx3_bind_fields_populate(const struct agx3_field_kv *baseline,
                          unsigned nbaseline,
                          const struct agx3_field_kv *overrides,
                          unsigned noverrides,
                          uint64_t fields[AGX3_MAX_FIELDS],
                          uint64_t nonzero[3])
{
   for (unsigned i = 0; i < nbaseline; ++i) {
      const unsigned field = baseline[i].field;
      if (unlikely(field >= AGX3_MAX_FIELDS)) {
         agx3_bind_fields_clear_nonzero(fields, nonzero);
         return false;
      }
      const uint64_t val = baseline[i].value;
      if (val) {
         fields[field] = val;
         nonzero[field >> 6] |= UINT64_C(1) << (field & 63);
      } else if (unlikely(fields[field] != 0)) {
         fields[field] = 0;
         nonzero[field >> 6] &= ~(UINT64_C(1) << (field & 63));
      }
   }
   for (unsigned i = 0; i < noverrides; ++i) {
      const unsigned field = overrides[i].field;
      if (unlikely(field >= AGX3_MAX_FIELDS)) {
         agx3_bind_fields_clear_nonzero(fields, nonzero);
         return false;
      }
      const uint64_t val = overrides[i].value;
      if (val) {
         fields[field] = val;
         nonzero[field >> 6] |= UINT64_C(1) << (field & 63);
      } else if (fields[field] != 0) {
         fields[field] = 0;
         nonzero[field >> 6] &= ~(UINT64_C(1) << (field & 63));
      }
   }
   return true;
}


/*
 * Composition is a pure function of the table, the form (a mnemonic with an
 * encoding point, or Apple's order over its forms) and the field vector, and
 * the Ioto program generators compose the same instructions for every draw:
 * on the M6 the table walk was about half of Chromium's GPU-process time
 * while dragging Google Maps. Remember recent compositions, keyed by exactly
 * those inputs. Only nonzero fields enter the vector (the scratch vector is
 * kept zero-filled), so the key is the nonzero fields in field order. Only
 * successful compositions are kept; a hit returns the bytes the table walk
 * returned for the same inputs.
 */
#define AGX3_BIND_MEMO_SLOTS  1024
#define AGX3_BIND_MEMO_FIELDS 24

struct agx3_bind_memo {
   uint64_t hash;
   const agx3_table *table;
   enum agx3_mnemonic mnemonic;
   int ep;
   uint8_t nfields, len;
   uint8_t field[AGX3_BIND_MEMO_FIELDS];
   uint64_t value[AGX3_BIND_MEMO_FIELDS];
   uint8_t out[2 * AGX3_MAX_WORDS];
};

/* Composition runs on parallel shader-compiler threads too. The memo is a
 * try-lock: a thread that finds it busy composes without it (and does not
 * store), so nobody waits, and the binder needs no threading library. */
static struct agx3_bind_memo agx3_bind_memo[AGX3_BIND_MEMO_SLOTS];
static atomic_flag agx3_bind_memo_busy = ATOMIC_FLAG_INIT;

static bool
agx3_bind_memo_lock(void)
{
   return !atomic_flag_test_and_set_explicit(&agx3_bind_memo_busy,
                                             memory_order_acquire);
}

static void
agx3_bind_memo_unlock(void)
{
   atomic_flag_clear_explicit(&agx3_bind_memo_busy, memory_order_release);
}

/* The memo key for this composition, false if it has more nonzero fields
 * than an entry holds. */
static bool
agx3_bind_memo_key(const agx3_table *t, enum agx3_mnemonic mnemonic, int ep,
                   const uint64_t fields[AGX3_MAX_FIELDS],
                   const uint64_t nonzero[3], struct agx3_bind_memo *key)
{
   key->table = t;
   key->mnemonic = mnemonic;
   key->ep = ep;
   key->nfields = 0;
   uint64_t hash = ((uint64_t)(uintptr_t)t) * UINT64_C(0x9e3779b97f4a7c15);
   hash ^= ((uint64_t)(uint32_t)mnemonic << 32) | (uint32_t)ep;
   hash *= UINT64_C(0xbf58476d1ce4e5b9);
   for (unsigned w = 0; w < 3; ++w) {
      uint64_t bits = nonzero[w];
      while (bits) {
         const unsigned field = w * 64 + u_bit_scan64(&bits);
         if (key->nfields == AGX3_BIND_MEMO_FIELDS)
            return false;
         const uint64_t val = fields[field];
         key->field[key->nfields] = field;
         key->value[key->nfields++] = val;
         hash = (hash ^ ((uint64_t)field << 48) ^ val) *
                UINT64_C(0x9e3779b97f4a7c15);
      }
   }
   hash ^= hash >> 33;
   hash *= UINT64_C(0xff51afd7ed558ccd);
   hash ^= hash >> 33;
   key->hash = hash;
   return true;
}

static struct agx3_bind_memo *
agx3_bind_memo_slot(const struct agx3_bind_memo *key)
{
   return &agx3_bind_memo[key->hash & (AGX3_BIND_MEMO_SLOTS - 1)];
}

static bool
agx3_bind_memo_get(const struct agx3_bind_memo *key, uint8_t *out,
                   uint8_t *len)
{
   if (!agx3_bind_memo_lock())
      return false;
   bool hit = false;
   const struct agx3_bind_memo *e = agx3_bind_memo_slot(key);
   if (e->len && e->hash == key->hash && e->table == key->table &&
       e->mnemonic == key->mnemonic && e->ep == key->ep &&
       e->nfields == key->nfields &&
       !memcmp(e->field, key->field, key->nfields * sizeof(key->field[0])) &&
       !memcmp(e->value, key->value, key->nfields * sizeof(key->value[0]))) {
      memcpy(out, e->out, e->len);
      *len = e->len;
      hit = true;
   }
   agx3_bind_memo_unlock();
   return hit;
}

static void
agx3_bind_memo_put(const struct agx3_bind_memo *key, const uint8_t *out,
                   uint8_t len)
{
   if (len == 0 || len > sizeof(key->out) || !agx3_bind_memo_lock())
      return;
   struct agx3_bind_memo *e = agx3_bind_memo_slot(key);
   e->hash = key->hash;
   e->table = key->table;
   e->mnemonic = key->mnemonic;
   e->ep = key->ep;
   e->nfields = key->nfields;
   if (key->nfields) {
      memcpy(e->field, key->field, key->nfields * sizeof(key->field[0]));
      memcpy(e->value, key->value, key->nfields * sizeof(key->value[0]));
   }
   memcpy(e->out, out, len);
   e->len = len;
   agx3_bind_memo_unlock();
}

bool
agx3_bind_compose(enum agx_isa_encoding encoding, enum agx3_mnemonic mnemonic,
                  int ep,
                  const struct agx3_field_kv *baseline, unsigned nbaseline,
                  const struct agx3_field_kv *overrides, unsigned noverrides,
                  uint8_t *out, uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;

   /*
    * agx3d forms are chosen the way Apple's encoder chooses them, never by an
    * encoding-point number: agx3b/agx3c ep numbers do not name agx3d forms.
    * A caller has to ask for that explicitly, so that no agx3c field builder
    * reaches the agx3d table by forwarding its encoding.
    */
   const agx3_table *t;
   if (encoding == AGX_ISA_ENCODING_AGX3D) {
      t = agx3d_table_or_null();
      if (ep != AGX3_BIND_APPLE_ORDER || t == NULL)
         return false;
   } else if (encoding == AGX_ISA_ENCODING_AGX3B) {
      t = &agx3b_table;
   } else if (encoding == AGX_ISA_ENCODING_AGX3C) {
      t = &agx3c_table;
   } else {
      return false;
   }

   int32_t form = -1;
   if (ep != AGX3_BIND_APPLE_ORDER) {
      form = agx3_form_index(t, mnemonic, (uint8_t)ep);
      if (form < 0)
         return false;
   }

   uint64_t stack_fields[AGX3_MAX_FIELDS];
   uint64_t *fields = agx3_bind_scratch_buffer();
   if (!fields) {
      memset(stack_fields, 0, sizeof(stack_fields));
      fields = stack_fields;
   }
   uint64_t nonzero[3] = {0};
   if (!agx3_bind_fields_populate(baseline, nbaseline, overrides, noverrides,
                                  fields, nonzero))
      return false;

   /* Reject mismatched explicit encoding-point probes before building a memo
    * key or taking the global memo lock. */
   if (ep != AGX3_BIND_APPLE_ORDER &&
       !agx3_form_preds_match(t, (uint32_t)form, fields)) {
      agx3_bind_fields_clear_nonzero(fields, nonzero);
      return false;
   }

   struct agx3_bind_memo key;
   const bool memo = agx3_bind_memo_key(t, mnemonic, ep, fields, nonzero, &key);
   if (memo && agx3_bind_memo_get(&key, out, len)) {
      agx3_bind_fields_clear_nonzero(fields, nonzero);
      return true;
   }

   bool ok = true;
   if (ep == AGX3_BIND_APPLE_ORDER) {
      form = agx3_compose_apple_order_known(t, mnemonic, fields, nonzero, out);
      if (form < 0)
         ok = false;
   } else if (!agx3_compose_known(t, (uint32_t)form, fields, nonzero, out)) {
      /* agx3_compose keeps its words invocation-local and does not touch
       * out until every predicate, deposit and field-consumption check
       * succeeds, so it encodes directly into the caller's buffer. */
      ok = false;
   }

   agx3_bind_fields_clear_nonzero(fields, nonzero);
   if (!ok)
      return false;

   *len = t->forms[form].size;
   if (memo)
      agx3_bind_memo_put(&key, out, *len);
   return true;
}

bool
agx3_bind_compose_ep_range(enum agx_isa_encoding encoding,
                           enum agx3_mnemonic mnemonic,
                           unsigned first_ep, unsigned last_ep,
                           int skip_ep, unsigned max_length,
                           bool monotonic_size,
                           const struct agx3_field_kv *baseline,
                           unsigned nbaseline,
                           const struct agx3_field_kv *overrides,
                           unsigned noverrides, uint8_t *out, uint8_t *len)
{
   if (out == NULL || len == NULL || first_ep > last_ep)
      return false;

   const agx3_table *t;
   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      t = &agx3b_table;
   } else if (encoding == AGX_ISA_ENCODING_AGX3C) {
      t = &agx3c_table;
   } else {
      return false;
   }

   uint64_t stack_fields[AGX3_MAX_FIELDS];
   uint64_t *fields = agx3_bind_scratch_buffer();
   if (!fields) {
      memset(stack_fields, 0, sizeof(stack_fields));
      fields = stack_fields;
   }
   uint64_t nonzero[3] = {0};
   if (!agx3_bind_fields_populate(baseline, nbaseline, overrides, noverrides,
                                  fields, nonzero))
      return false;

   bool ok = false;
   for (unsigned ep = first_ep; ep <= last_ep; ++ep) {
      if ((int)ep == skip_ep)
         continue;
      const int32_t form = agx3_form_index(t, mnemonic, (uint8_t)ep);
      if (form < 0)
         continue;
      const uint8_t fsize = t->forms[form].size;
      if (max_length && fsize >= max_length) {
         if (monotonic_size)
            break;
         continue;
      }
      if (!agx3_form_preds_match(t, (uint32_t)form, fields))
         continue;

      struct agx3_bind_memo key;
      const bool memo = agx3_bind_memo_key(t, mnemonic, (int)ep, fields,
                                           nonzero, &key);
      if (memo && agx3_bind_memo_get(&key, out, len)) {
         ok = true;
         break;
      }

      if (agx3_compose_known(t, (uint32_t)form, fields, nonzero, out)) {
         *len = fsize;
         if (memo)
            agx3_bind_memo_put(&key, out, fsize);
         ok = true;
         break;
      }
   }

   agx3_bind_fields_clear_nonzero(fields, nonzero);
   return ok;
}
