/* agx3_compose.c -- generic AGX3 encoder over the generated tables.
 * Mirrors composer.py encode semantics; see agx3_compose.h. */
#include "agx3_compose.h"
#include <stdint.h>
#include <string.h>

/* The nodes of one traversal: the root and one alternative of every group it
 * comes to.  The descriptions need at most 34 (agx3b), 38 (agx3c) and 18
 * (agx3d); a table that needs more fails closed. */
#define AGX3_MAX_PATH 64

typedef struct {
   const agx3_table *t;
   const uint64_t *fields;
   uint16_t words[AGX3_MAX_WORDS], written[AGX3_MAX_WORDS];
   uint8_t nwords;
   /* The nodes applied so far, the root first.  What they consume is
    * checked once, after the walk, so an alternative that does not apply
    * leaves nothing to undo but path_n and the words.  The array is the
    * caller's and stays out of the snapshot of an alternative. */
   uint8_t path_n;
   const agx3_node **path;
   /* Fields with a direct deposit and a value that has other bits set. */
   uint64_t wide[3];
} ctx;

static inline uint64_t wmask(unsigned width) {
   return width >= 64 ? ~0ull : (1ull << width) - 1;
}

/* A predicate that is not "field == small value". */
static bool wide_pred_ok(const agx3_table *t, const uint64_t *fields,
                         const agx3_pred *p)
{
   uint64_t v = fields[p->field];
   switch (p->kind) {
   case 0: return v == p->a;
   case 1: {
      for (uint16_t i = 0; i < p->b; i++)
         if (t->setpool[p->a + i] == v) return true;
      return false;
   }
   case 2: return p->a <= v && v <= p->b;
   case 3: return ((v >> p->shift) & wmask(p->width)) == p->a;
   default: return false;
   }
}

/* Inline, and on the table and the fields rather than the ctx, so that the
 * walk over a group's alternatives keeps its state in registers. */
static inline bool preds_test(const agx3_table *t, const uint64_t *fields,
                              const agx3_node *n)
{
   const agx3_predref *r = &t->predrefs[n->pred_off];
   for (unsigned i = 0; i < n->pred_n; i++) {
      if (r[i].field < AGX3_MAX_FIELDS) {
         if (fields[r[i].field] != r[i].value) return false;
      } else if (!wide_pred_ok(t, fields, agx3_wide_pred(t, r[i]))) {
         return false;
      }
   }
   return true;
}

static bool deposit(ctx *c, uint8_t word, uint16_t mask, uint16_t value)
{
   if (c->written[word] & mask & (c->words[word] ^ value)) return false;
   c->words[word] = (uint16_t)((c->words[word] & ~mask) | (value & mask));
   c->written[word] |= mask;
   return true;
}

static bool apply_node(ctx *c, const agx3_node *n);

/* A group is walked this far before its key index is used.  The first
 * alternatives are the ones most instructions take, and testing a few of
 * them costs less than the search.  gen_agx3_tables.py stores no index for a
 * group no longer than its KEY_WALK.  The two numbers only have to agree for
 * speed and size: the alternative selected is the same with any. */
#define AGX3_KEY_WALK 8

/* The alternative of a keyed group whose row has these field values, or
 * NULL. */
static const agx3_node *keyed_alt(const agx3_table *t, const uint64_t *fields,
                                  const agx3_group *g)
{
   const uint8_t *key_fields = &t->keys[g->key_off];
   const uint8_t *rows = key_fields + g->key_n;
   unsigned lo = 0, hi = g->alt_n;
   while (lo < hi) {
      const unsigned mid = (lo + hi) / 2;
      const uint8_t *row = rows + mid * (g->key_n + 2u);
      unsigned i = 0;
      while (i < g->key_n && row[i] == fields[key_fields[i]])
         i++;
      if (i == g->key_n) {
         const unsigned pos = row[i] << 8 | row[i + 1];
         return &t->nodes[g->alt_off + pos];
      }
      if (row[i] < fields[key_fields[i]]) lo = mid + 1;
      else hi = mid;
   }
   return NULL;
}

/* The first of the alternatives alt .. end whose predicates hold, or end. */
static inline const agx3_node *find_alt(const agx3_table *t,
                                        const uint64_t *fields,
                                        const agx3_node *alt,
                                        const agx3_node *end)
{
   while (alt < end && !preds_test(t, fields, alt))
      alt++;
   return alt;
}

static bool apply_group(ctx *c, const agx3_group *g)
{
   const agx3_table *t = c->t;
   const uint64_t *fields = c->fields;
   const agx3_node *alt = &t->nodes[g->alt_off];
   const agx3_node *end = alt + g->alt_n;

   if (g->exclusive) {
      /* Only one alternative can hold: there is nothing to go back to when
       * it does not apply, so no snapshot, and it may be looked up. */
      const agx3_node *walk_end =
         g->key_n && g->alt_n > AGX3_KEY_WALK ? alt + AGX3_KEY_WALK : end;
      alt = find_alt(t, fields, alt, walk_end);
      if (alt == end) return false;
      if (alt == walk_end) {
         alt = keyed_alt(t, fields, g);
         if (!alt || !preds_test(t, fields, alt)) return false;
      }
      return apply_node(c, alt);
   }

   for (;; alt++) {
      /* Most alternatives fail their predicates without modifying state.
       * Only snapshot the encoder once an alternative can actually apply. */
      alt = find_alt(t, fields, alt, end);
      if (alt == end) return false;
      ctx saved = *c;
      if (apply_node(c, alt))
         return true;
      *c = saved;
   }
}

/* Apply a node whose predicates hold. */
static bool apply_node(ctx *c, const agx3_node *n)
{
   if (c->path_n == AGX3_MAX_PATH) return false;
   c->path[c->path_n++] = n;
   for (uint16_t i = 0; i < n->fixed_n; i++) {
      const agx3_fixed *f = &c->t->fixeds[n->fixed_off + i];
      if (!deposit(c, f->word, f->mask, f->value)) return false;
   }
   for (uint16_t i = 0; i < n->cmd_n; i++) {
      const agx3_cmd *cmd = &c->t->cmds[n->cmd_off + i];
      uint64_t value = c->fields[cmd->field], transformed;
      if (cmd->lut_n) {
         if (value >= cmd->lut_n) return false;
         transformed = c->t->lutpool[cmd->lut_off + value];
      } else {
         transformed = value;
         if (value & ~c->t->slices[cmd->lut_off])
            c->wide[cmd->field >> 6] |= 1ull << (cmd->field & 63);
      }
      const unsigned mask = (1u << cmd->width) - 1;
      const unsigned bits = (unsigned)(transformed >> cmd->src_shift) & mask;
      if (!deposit(c, cmd->word, (uint16_t)(mask << cmd->dst_shift),
                   (uint16_t)(bits << cmd->dst_shift)))
         return false;
   }
   for (uint16_t i = 0; i < n->grp_n; i++)
      if (!apply_group(c, &c->t->groups[n->grp_off + i]))
         return false;
   return true;
}

bool agx3_compose_known(const agx3_table *t, uint32_t form,
                        const uint64_t fields[AGX3_MAX_FIELDS],
                        const uint64_t nonzero[3], uint8_t *out)
{
   if (form >= t->num_forms) return false;
   const agx3_form *f = &t->forms[form];
   if (f->size % 2 || f->size / 2 > AGX3_MAX_WORDS) return false;

   /* Composition is used by parallel shader compilers.  Keep all mutable
    * traversal state on this invocation; the generated tables are immutable. */
   const agx3_node *path[AGX3_MAX_PATH];
   ctx c = {0};
   ctx *cp = &c;
   cp->t = t;
   cp->fields = fields;
   cp->nwords = f->size / 2;
   cp->path = path;

   const agx3_node *root = &t->nodes[f->node];
   if (!preds_test(t, fields, root)) return false;
   if (!apply_node(cp, root)) return false;

   /* strict: every nonzero field must have been consumed, and a directly
    * deposited value must fit its slices unless a predicate or a LUT of an
    * applied node covers the field. */
   uint64_t use[6] = {0};
   for (unsigned i = 0; i < cp->path_n; i++) {
      for (unsigned w = 0; w < 6; w++)
         use[w] |= t->uses[path[i]->use_off + w];
   }
   for (unsigned w = 0; w < 3; w++) {
      if ((nonzero[w] & ~use[w]) | (cp->wide[w] & ~use[3 + w])) return false;
   }

   for (unsigned w = 0; w < cp->nwords; w++) {
      out[2 * w] = (uint8_t)cp->words[w];
      out[2 * w + 1] = (uint8_t)(cp->words[w] >> 8);
   }
   return true;
}

/* Which fields are nonzero. */
static void nonzero_fields(const uint64_t fields[AGX3_MAX_FIELDS],
                           uint64_t nonzero[3])
{
   memset(nonzero, 0, 3 * sizeof(nonzero[0]));
   for (unsigned i = 0; i < AGX3_MAX_FIELDS; i++) {
      if (fields[i])
         nonzero[i >> 6] |= 1ull << (i & 63);
   }
}

bool agx3_compose(const agx3_table *t, uint32_t form,
                  const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out)
{
   uint64_t nonzero[3];
   nonzero_fields(fields, nonzero);
   return agx3_compose_known(t, form, fields, nonzero, out);
}

static const char *const mnemonic_names[AGX3_MN_COUNT] = {
   [AGX3_MN_NONE] = "?",
#define X(name) [AGX3_MN_##name] = #name,
   AGX3_MNEMONICS(X)
#undef X
};

const char *agx3_mnemonic_name(enum agx3_mnemonic mnemonic)
{
   return mnemonic_names[(unsigned)mnemonic < AGX3_MN_COUNT ? mnemonic
                                                            : AGX3_MN_NONE];
}

enum agx3_mnemonic agx3_mnemonic_from_name(const char *name)
{
   /* Only tests and tools start from a spelling. */
   for (unsigned m = AGX3_MN_NONE + 1; m < AGX3_MN_COUNT; m++)
      if (!strcmp(mnemonic_names[m], name)) return (enum agx3_mnemonic)m;
   return AGX3_MN_NONE;
}

/* The first form of mnemonic in t, t->num_forms if it has none. */
static uint32_t first_form(const agx3_table *t, enum agx3_mnemonic mnemonic)
{
   if ((unsigned)mnemonic >= AGX3_MN_COUNT || !t->mnemonic_forms[mnemonic])
      return t->num_forms;
   return t->mnemonic_forms[mnemonic] - 1u;
}

int32_t agx3_form_index(const agx3_table *t, enum agx3_mnemonic mnemonic,
                        uint8_t ep)
{
   for (uint32_t f = first_form(t, mnemonic);
        f < t->num_forms && t->forms[f].mnemonic == mnemonic; f++)
      if (t->forms[f].ep == ep) return (int32_t)f;
   return -1;
}

/*
 * Apple's encoder tries a mnemonic's encoding points by ascending size, the
 * ep number breaking ties, and keeps the first that encodes the field vector.
 * The forms of one mnemonic are contiguous in a table, and apple_order lists
 * them in that order.
 */
int32_t
agx3_compose_apple_order_known(const agx3_table *t,
                               enum agx3_mnemonic mnemonic,
                               const uint64_t fields[AGX3_MAX_FIELDS],
                               const uint64_t nonzero[3], uint8_t *out)
{
   for (uint32_t i = first_form(t, mnemonic);
        i < t->num_forms && t->forms[i].mnemonic == mnemonic; i++) {
      const uint32_t form = t->apple_order[i];
      if (agx3_compose_known(t, form, fields, nonzero, out))
         return (int32_t)form;
   }
   return -1;
}

int32_t
agx3_compose_apple_order(const agx3_table *t, enum agx3_mnemonic mnemonic,
                         const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out)
{
   uint64_t nonzero[3];
   nonzero_fields(fields, nonzero);
   return agx3_compose_apple_order_known(t, mnemonic, fields, nonzero, out);
}
