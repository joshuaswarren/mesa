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
agx3_bind_form_size(enum agx_isa_encoding encoding, const char *mnemonic,
                    int ep)
{
   const agx3_table *t = agx3_bind_table(encoding);
   const int32_t form = t ? agx3_form_index(t, mnemonic, (uint8_t)ep) : -1;
   return form < 0 ? 0 : t->forms[form].size;
}

#include <string.h>

/*
 * The agx3d (G18G) table is 10 MB of generated data. Only the builds that
 * link agx3_compose/agx3d_tables.c define AGX3_HAVE_AGX3D_TABLE; everywhere
 * else an AGX3D request fails closed here instead of failing to link.
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

static bool
agx3_bind_fields(const struct agx3_field_kv *baseline, unsigned nbaseline,
                 const struct agx3_field_kv *overrides, unsigned noverrides,
                 uint64_t fields[AGX3_MAX_FIELDS])
{
   memset(fields, 0, sizeof(uint64_t) * AGX3_MAX_FIELDS);
   for (unsigned i = 0; i < nbaseline; ++i) {
      if (baseline[i].field >= AGX3_MAX_FIELDS)
         return false;
      fields[baseline[i].field] = baseline[i].value;
   }
   for (unsigned i = 0; i < noverrides; ++i) {
      if (overrides[i].field >= AGX3_MAX_FIELDS)
         return false;
      fields[overrides[i].field] = overrides[i].value;
   }
   return true;
}


bool
agx3_bind_compose(enum agx_isa_encoding encoding, const char *mnemonic, int ep,
                  const struct agx3_field_kv *baseline, unsigned nbaseline,
                  const struct agx3_field_kv *overrides, unsigned noverrides,
                  uint8_t *out, uint8_t *len)
{
   if (out == NULL || len == NULL || mnemonic == NULL)
      return false;

   /*
    * agx3d forms are chosen the way Apple's encoder chooses them, never by an
    * encoding-point number: agx3b/agx3c ep numbers do not name agx3d forms.
    * A caller has to ask for that explicitly, so that no agx3c field builder
    * reaches the agx3d table by forwarding its encoding.
    */
   if (encoding == AGX_ISA_ENCODING_AGX3D) {
      const agx3_table *t = agx3d_table_or_null();
      uint64_t fields[AGX3_MAX_FIELDS];
      if (ep != AGX3_BIND_APPLE_ORDER || t == NULL ||
          !agx3_bind_fields(baseline, nbaseline, overrides, noverrides, fields))
         return false;

      const int32_t form = agx3_compose_apple_order(t, mnemonic, fields, out);
      if (form < 0)
         return false;
      *len = t->forms[form].size;
      return true;
   }

   const agx3_table *t;
   if (encoding == AGX_ISA_ENCODING_AGX3B)
      t = &agx3b_table;
   else if (encoding == AGX_ISA_ENCODING_AGX3C)
      t = &agx3c_table;
   else
      return false;

   uint64_t fields[AGX3_MAX_FIELDS];
   if (ep == AGX3_BIND_APPLE_ORDER) {
      if (!agx3_bind_fields(baseline, nbaseline, overrides, noverrides, fields))
         return false;
      const int32_t form = agx3_compose_apple_order(t, mnemonic, fields, out);
      if (form < 0)
         return false;
      *len = t->forms[form].size;
      return true;
   }

   const int32_t form = agx3_form_index(t, mnemonic, (uint8_t)ep);
   if (form < 0)
      return false;

   if (!agx3_bind_fields(baseline, nbaseline, overrides, noverrides, fields))
      return false;

   /* Only the listed fields can be nonzero. */
   uint64_t nonzero[3] = {0};
   for (unsigned i = 0; i < nbaseline; ++i) {
      const unsigned field = baseline[i].field;
      if (fields[field])
         nonzero[field / 64] |= UINT64_C(1) << (field % 64);
   }
   for (unsigned i = 0; i < noverrides; ++i) {
      const unsigned field = overrides[i].field;
      if (fields[field])
         nonzero[field / 64] |= UINT64_C(1) << (field % 64);
   }

   /* agx3_compose keeps its words invocation-local and does not touch out
    * until every predicate, deposit and field-consumption check succeeds.
    * Encode directly into the caller's buffer instead of copying the same
    * validated instruction through a second 32-byte stack buffer. */
   if (!agx3_compose_known(t, (uint32_t)form, fields, nonzero, out))
      return false;

   const uint8_t size = t->forms[form].size;
   *len = size;
   return true;
}
