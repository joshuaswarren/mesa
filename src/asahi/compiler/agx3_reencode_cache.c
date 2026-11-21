/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * G18G re-encoding cache. The generators emit the same program for every
 * draw of a shader but for the uniform table addresses, which the long (8-byte)
 * agx3c LDIMMs carry, and re-encoding one as agx3d costs tens of microseconds:
 * on the M6 it took over half the CPU time of a 4K60 video player. An entry
 * keys the agx3c program with those immediates cleared and keeps its agx3d
 * form plus, for each such LDIMM, where it went and its agx3d field vector. A
 * hit composes those LDIMMs with this draw's immediates the way the
 * transcoder does (agx3_compose_apple_order(), whose form, and so size, may
 * depend on the value) and copies the rest. A size the entry was not made with
 * re-encodes the program in full and replaces the entry.
 *
 * The transcoder maps an LDIMM's fields one to one, but for the USC helper
 * base, which it doubles (0x100 and 0x200 after an SR14 rld): such an
 * immediate stays in the key. Every other agx3d instruction only depends on
 * its own agx3c fields and, for branches, on the layout, which a hit keeps.
 */

#include "agx3_reencode_cache.h"

#include <stdlib.h>
#include <string.h>

#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decode.h"
#include "asahi/isa/disasm.h"
#include "asahi/lib/agx_ioto_load_core.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#define XXH_INLINE_ALL
#include "util/xxhash.h"

#define AGX3_REENCODE_MAX_LDIMMS 32
/* The agx3c/agx3d field vector index of an LDIMM's immediate */
#define AGX3_LDIMM_IMMEDIATE_FIELD 14

struct agx3_reencode_ldimm {
   uint16_t offset;
   uint8_t size;
   uint64_t fields[AGX3_MAX_FIELDS];
};

struct agx3_reencode_entry {
   struct agx3_reencode_entry *next;
   uint64_t hash;
   bool api_program;
   uint16_t shader_gprs;
   uint16_t in_length, out_length;
   uint8_t ldimm_count;
   uint8_t key[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint8_t out[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   struct agx3_reencode_ldimm ldimms[];
};

/* The bits of a long LDIMM word that are neither its destination nor its
 * immediate, and their value. */
static inline void
agx3_ldimm_long_pattern(uint64_t *fixed_mask, uint64_t *opcode)
{
   uint64_t dest = 0;
   for (unsigned half = 0; half < 256; ++half)
      dest |= agx_ioto_load_ldimm_long_dest(half) ^
              agx_ioto_load_ldimm_long_dest(0);
   *fixed_mask = ~(dest | agx_ioto_load_ldimm_long_immediate(~0u));
   *opcode = agx_ioto_load_ldimm_long_dest(0) & *fixed_mask;
}

/* The inverse of agx_ioto_load_ldimm_long_immediate() */
static inline uint32_t
agx3_ldimm_long_immediate_of(uint64_t word)
{
   return (uint32_t)(((word >> 8) & 0x7f) | (((word >> 25) & 0x7f) << 25) |
                     (((word >> 33) & 0xf) << 7) | (((word >> 42) & 3) << 11) |
                     (((word >> 48) & 0xf) << 13) |
                     (((word >> 52) & 0xff) << 17));
}

/* Split program into its key (immediates cleared) and the long LDIMMs that
 * were cleared. False if the bytes are not a whole agx3c instruction stream
 * or have more LDIMMs than an entry holds. */
static bool
agx3_reencode_key(const struct agx3_g17p_graphics_program *program,
                  uint8_t *key, unsigned *ldimm_in, uint32_t *immediates,
                  unsigned *ldimm_count)
{
   uint64_t fixed_mask, opcode;
   agx3_ldimm_long_pattern(&fixed_mask, &opcode);
   const uint64_t immediate_mask = agx_ioto_load_ldimm_long_immediate(~0u);

   memcpy(key, program->bytes, program->length);
   unsigned count = 0;
   for (unsigned i = 0; i < program->length;) {
      const signed n =
         agx3c_instr_length(program->bytes + i, program->length - i, NULL);
      if (n <= 0)
         return false;
      if (n == 8) {
         uint64_t word;
         memcpy(&word, program->bytes + i, sizeof(word));
         word = util_le64_to_cpu(word);
         const uint32_t immediate = agx3_ldimm_long_immediate_of(word);
         if ((word & fixed_mask) == opcode && immediate != 0x100 &&
             immediate != 0x200) {
            if (count == AGX3_REENCODE_MAX_LDIMMS)
               return false;
            ldimm_in[count] = i;
            immediates[count++] = immediate;
            word = util_cpu_to_le64(word & ~immediate_mask);
            memcpy(key + i, &word, sizeof(word));
         }
      }
      i += n;
   }
   *ldimm_count = count;
   return true;
}

static bool
agx3_reencode_transcode(struct agx3_g17p_graphics_program *program,
                        const struct agx3_transcode_ioto *ioto,
                        unsigned *offsets, unsigned offset_count,
                        const char **why, unsigned *at)
{
   struct util_dynarray code;
   util_dynarray_init(&code, NULL);
   memcpy(util_dynarray_grow_bytes(&code, 1, program->length), program->bytes,
          program->length);
   const bool ok = agx3_transcode_agx3d_ioto_with_offsets(
                      &code, 0, ioto, offsets, offset_count, why, at) &&
                   code.size <= sizeof(program->bytes);
   if (ok) {
      memcpy(program->bytes, code.data, code.size);
      program->length = code.size;
   } else if (!*why) {
      *why = "longer than the program buffer";
   }
   util_dynarray_fini(&code);
   return ok;
}

bool
agx3_reencode_program(struct agx3_g17p_graphics_program *program,
                      const struct agx3_transcode_ioto *ioto, const char **why,
                      unsigned *at)
{
   return agx3_reencode_transcode(program, ioto, NULL, 0, why, at);
}

static struct agx3_reencode_entry *
agx3_reencode_lookup(struct agx3_reencode_cache *cache, uint64_t hash,
                     const struct agx3_transcode_ioto *ioto, const uint8_t *key,
                     unsigned length)
{
   struct agx3_reencode_entry **slot = &cache->entries;
   for (; *slot; slot = &(*slot)->next) {
      struct agx3_reencode_entry *e = *slot;
      if (e->hash == hash && e->api_program == ioto->api_program &&
          e->shader_gprs == ioto->shader_gprs && e->in_length == length &&
          !memcmp(e->key, key, length)) {
         /* Most recently used first */
         *slot = e->next;
         e->next = cache->entries;
         cache->entries = e;
         return e;
      }
   }
   return NULL;
}

/* Fill e from a program just re-encoded from key, its LDIMMs (as numbered by
 * agx3_reencode_key()) now at ldimm_out. False if each of them does not
 * decode to its immediate and compose back to the same bytes. */
static bool
agx3_reencode_fill(struct agx3_reencode_entry *e, uint64_t hash,
                   const struct agx3_transcode_ioto *ioto, const uint8_t *key,
                   unsigned in_length,
                   const struct agx3_g17p_graphics_program *encoded,
                   const unsigned *ldimm_out, const uint32_t *immediates,
                   unsigned ldimm_count)
{
   e->hash = hash;
   e->api_program = ioto->api_program;
   e->shader_gprs = ioto->shader_gprs;
   e->in_length = in_length;
   e->out_length = encoded->length;
   e->ldimm_count = ldimm_count;
   memcpy(e->key, key, in_length);
   memcpy(e->out, encoded->bytes, encoded->length);
   for (unsigned k = 0; k < ldimm_count; ++k) {
      struct agx3_reencode_ldimm *l = &e->ldimms[k];
      const unsigned at = ldimm_out[k];
      if (at >= encoded->length)
         return false;
      const signed n =
         agx3d_instr_length(encoded->bytes + at, encoded->length - at, NULL);
      uint8_t bytes[2 * AGX3_MAX_WORDS];
      if (n <= 0 ||
          agx3_decode(&agx3d_table, encoded->bytes + at, n, "ldimm",
                      l->fields) < 0 ||
          l->fields[AGX3_LDIMM_IMMEDIATE_FIELD] != immediates[k])
         return false;
      const int32_t form =
         agx3_compose_apple_order(&agx3d_table, "ldimm", l->fields, bytes);
      if (form < 0 || agx3d_table.forms[form].size != (unsigned)n ||
          memcmp(bytes, encoded->bytes + at, n))
         return false;
      l->offset = at;
      l->size = n;
   }
   return true;
}

/* The agx3d program for the immediates from e, false if one of them takes
 * another size than e was made with. */
static bool
agx3_reencode_emit(struct agx3_reencode_entry *e, const uint32_t *immediates,
                   struct agx3_g17p_graphics_program *program)
{
   memcpy(program->bytes, e->out, e->out_length);
   program->length = e->out_length;
   for (unsigned k = 0; k < e->ldimm_count; ++k) {
      struct agx3_reencode_ldimm *l = &e->ldimms[k];
      uint8_t bytes[2 * AGX3_MAX_WORDS];
      l->fields[AGX3_LDIMM_IMMEDIATE_FIELD] = immediates[k];
      const int32_t form =
         agx3_compose_apple_order(&agx3d_table, "ldimm", l->fields, bytes);
      if (form < 0 || agx3d_table.forms[form].size != l->size)
         return false;
      memcpy(program->bytes + l->offset, bytes, l->size);
   }
   return true;
}

enum agx3_reencode_result
agx3_reencode_cached(struct agx3_reencode_cache *cache,
                     struct agx3_g17p_graphics_program *program,
                     const struct agx3_transcode_ioto *ioto, const char **why,
                     unsigned *at)
{
   uint8_t key[sizeof(program->bytes)];
   unsigned ldimm_in[AGX3_REENCODE_MAX_LDIMMS];
   uint32_t immediates[AGX3_REENCODE_MAX_LDIMMS];
   unsigned ldimm_count;

   if (program->length > sizeof(program->bytes) ||
       !agx3_reencode_key(program, key, ldimm_in, immediates, &ldimm_count))
      return AGX3_REENCODE_UNCACHEABLE;

   const unsigned length = program->length;
   const uint64_t seed =
      ioto->shader_gprs | ((uint64_t)ioto->api_program << 16);
   const uint64_t hash = XXH64(key, length, seed);
   const struct agx3_g17p_graphics_program agx3c = *program;
   bool hit = false;
   simple_mtx_lock(&cache->lock);
   struct agx3_reencode_entry *e =
      agx3_reencode_lookup(cache, hash, ioto, key, length);
   if (e)
      hit = agx3_reencode_emit(e, immediates, program);
   simple_mtx_unlock(&cache->lock);
   if (hit)
      return AGX3_REENCODE_HIT;

   /* Re-encode in full, tracking where the LDIMMs go */
   unsigned ldimm_out[AGX3_REENCODE_MAX_LDIMMS];
   memcpy(ldimm_out, ldimm_in, sizeof(ldimm_in[0]) * ldimm_count);
   *program = agx3c;
   if (!agx3_reencode_transcode(program, ioto, ldimm_out, ldimm_count, why,
                                at))
      return AGX3_REENCODE_REFUSED;

   struct agx3_reencode_entry *made =
      malloc(sizeof(*made) + ldimm_count * sizeof(made->ldimms[0]));
   if (made &&
       !agx3_reencode_fill(made, hash, ioto, key, length, program, ldimm_out,
                           immediates, ldimm_count)) {
      free(made);
      made = NULL;
   }
   if (made) {
      simple_mtx_lock(&cache->lock);
      /* Replaces an entry of another LDIMM size (or a concurrent fill) */
      struct agx3_reencode_entry *old =
         agx3_reencode_lookup(cache, hash, ioto, key, length);
      if (old) {
         cache->entries = old->next;
         free(old);
         --cache->count;
      }
      made->next = cache->entries;
      cache->entries = made;
      if (++cache->count > AGX3_REENCODE_CACHE_LIMIT) {
         struct agx3_reencode_entry **slot = &cache->entries;
         while ((*slot)->next)
            slot = &(*slot)->next;
         free(*slot);
         *slot = NULL;
         --cache->count;
      }
      simple_mtx_unlock(&cache->lock);
   }
   return AGX3_REENCODE_MISS;
}

void
agx3_reencode_cache_clear(struct agx3_reencode_cache *cache)
{
   simple_mtx_lock(&cache->lock);
   while (cache->entries) {
      struct agx3_reencode_entry *e = cache->entries;
      cache->entries = e->next;
      free(e);
   }
   cache->count = 0;
   simple_mtx_unlock(&cache->lock);
}
