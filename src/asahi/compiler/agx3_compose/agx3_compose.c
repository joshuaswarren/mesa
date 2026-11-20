/* agx3_compose.c -- generic AGX3 encoder over the generated tables.
 * Mirrors composer.py encode semantics; see agx3_compose.h. */
#include "agx3_compose.h"
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

struct direct_cmd {
   uint8_t field, width, src_shift;
};

typedef struct {
   const agx3_table *t;
   const uint64_t *fields;
   uint16_t words[AGX3_MAX_WORDS], written[AGX3_MAX_WORDS];
   uint8_t nwords;
   uint64_t consumed[3];              /* field ids consumed by cmds/preds */
   uint64_t is_lut[3], is_pred[3];    /* fields seen through a LUT / predicate */
   /* Direct commands selected along the successful traversal.  Keeping this
    * as an external append-only journal makes an alternative checkpoint
    * small: rollback restores direct_count instead of copying a 1.25 KiB
    * per-field source-mask array at every table group. */
   struct direct_cmd *direct_cmds;
   uint16_t direct_count, direct_capacity;
} ctx;

static inline void bset(uint64_t *bm, unsigned i) { bm[i >> 6] |= 1ull << (i & 63); }
static inline bool bget(const uint64_t *bm, unsigned i) { return (bm[i >> 6] >> (i & 63)) & 1; }
static inline uint64_t wmask(unsigned width) {
   return width >= 64 ? ~0ull : (1ull << width) - 1;
}

/* A predicate that is not "field == small value". */
static bool wide_pred_ok(const agx3_table *t, const uint64_t *fields,
                         const agx3_pred *p)
{
   if (p->field >= AGX3_MAX_FIELDS) return false;
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

static void preds_mark(ctx *c, const agx3_node *n)
{
   /* Only once all of them hold. */
   for (unsigned i = 0; i < n->pred_n; i++) {
      const unsigned field = agx3_pred_at(c->t, n->pred_off + i).field;
      bset(c->consumed, field);
      bset(c->is_pred, field);
   }
}

static bool deposit(ctx *c, uint8_t word, uint16_t mask, uint16_t value)
{
   if (word >= c->nwords) return false;
   if (c->written[word] & mask & (c->words[word] ^ value)) return false;
   c->words[word] = (uint16_t)((c->words[word] & ~mask) | (value & mask));
   c->written[word] |= mask;
   return true;
}

static bool apply_node(ctx *c, uint32_t node_id);

static bool apply_group(ctx *c, const agx3_group *g)
{
   const agx3_table *t = c->t;
   const uint64_t *fields = c->fields;
   const agx3_node *alt = &t->nodes[g->alt_off];
   for (const agx3_node *end = alt + g->alt_n; alt < end; alt++) {
      /* Most alternatives fail their predicates without modifying state.
       * Only snapshot the encoder once an alternative can actually apply. */
      if (!preds_test(t, fields, alt))
         continue;
      ctx saved = *c;
      preds_mark(c, alt);
      if (apply_node(c, (uint32_t)(alt - t->nodes)))
         return true;
      *c = saved;
   }
   return false;
}

static bool apply_node(ctx *c, uint32_t node_id)
{
   const agx3_node *n = &c->t->nodes[node_id];
   for (uint16_t i = 0; i < n->fixed_n; i++) {
      const agx3_fixed *f = &c->t->fixeds[n->fixed_off + i];
      if (!deposit(c, f->word, f->mask, f->value)) return false;
   }
   for (uint16_t i = 0; i < n->cmd_n; i++) {
      const agx3_cmd *cmd = &c->t->cmds[n->cmd_off + i];
      if (cmd->field >= AGX3_MAX_FIELDS) return false;
      uint64_t value = c->fields[cmd->field], transformed;
      bset(c->consumed, cmd->field);
      if (cmd->lut_n) {
         bset(c->is_lut, cmd->field);
         if (value >= cmd->lut_n) return false;
         transformed = c->t->lutpool[cmd->lut_off + value];
      } else {
         transformed = value;
         if (c->direct_count >= c->direct_capacity)
            return false;
         c->direct_cmds[c->direct_count++] = (struct direct_cmd){
            .field = cmd->field,
            .width = cmd->width,
            .src_shift = cmd->src_shift,
         };
      }
      uint16_t bits = (uint16_t)((transformed >> cmd->src_shift) & wmask(cmd->width));
      if (!deposit(c, cmd->word, (uint16_t)(wmask(cmd->width) << cmd->dst_shift),
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
   /* Generated AGX3B/C tables currently select at most 73 commands along any
    * form path, and AGX3D at most 68.  One slot per field leaves more than
    * twice that headroom and fails closed if a future table exceeds it. */
   struct direct_cmd direct_cmds[AGX3_MAX_FIELDS];
   ctx c = {0};
   ctx *cp = &c;
   cp->t = t;
   cp->fields = fields;
   cp->nwords = f->size / 2;
   cp->direct_cmds = direct_cmds;
   cp->direct_capacity = AGX3_MAX_FIELDS;

   const agx3_node *root = &t->nodes[f->node];
   if (!preds_test(t, fields, root)) return false;
   preds_mark(cp, root);
   if (!apply_node(cp, f->node)) return false;

   /* strict: every nonzero field must have been consumed */
   for (unsigned w = 0; w < 3; w++) {
      if (nonzero[w] & ~cp->consumed[w]) return false;
   }

   /* ... and a direct-deposited non-selector field must fit the union of its
    * source slices.  Only fields with a direct command can fail this, so
    * build the unions for those alone. */
   uint64_t srcmask[AGX3_MAX_FIELDS];
   for (uint16_t i = 0; i < cp->direct_count; ++i)
      srcmask[cp->direct_cmds[i].field] = 0;
   for (uint16_t i = 0; i < cp->direct_count; ++i) {
      const struct direct_cmd *cmd = &cp->direct_cmds[i];
      srcmask[cmd->field] |= wmask(cmd->width) << cmd->src_shift;
   }
   for (uint16_t i = 0; i < cp->direct_count; ++i) {
      const unsigned field = cp->direct_cmds[i].field;
      if (!bget(cp->is_lut, field) && !bget(cp->is_pred, field) &&
          (fields[field] & ~srcmask[field]))
         return false;
   }

   for (unsigned w = 0; w < cp->nwords; w++) {
      out[2 * w] = (uint8_t)cp->words[w];
      out[2 * w + 1] = (uint8_t)(cp->words[w] >> 8);
   }
   return true;
}

bool agx3_compose(const agx3_table *t, uint32_t form,
                  const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out)
{
   uint64_t nonzero[3] = {0};
   for (unsigned i = 0; i < AGX3_MAX_FIELDS; i++) {
      if (fields[i])
         bset(nonzero, i);
   }
   return agx3_compose_known(t, form, fields, nonzero, out);
}

/* The packer looks the same few forms up for every instruction it tries.
 * Remember recent results in a small process-wide table of 64-bit words:
 * the form index in the low 16 bits and the low 48 bits of the mnemonic
 * string's address above it.  A slot is read and written whole with relaxed
 * atomics, so concurrent compiles may overwrite each other's entries but
 * never see a torn one.  The slot does not record the table, so every hit
 * is confirmed against the form itself (index in range, same entry point,
 * same mnemonic, and no earlier form of both): a stale or foreign entry only
 * costs the search, and a hit is always the search's answer.
 *
 * Not thread-local: libgallium is loaded with initial-exec TLS, where every
 * byte of .tbss counts against the static TLS block of the process that
 * dlopens it. */
#define AGX3_FORM_CACHE_SLOTS 256
static _Atomic uint64_t agx3_form_cache[AGX3_FORM_CACHE_SLOTS];

int32_t agx3_form_index(const agx3_table *t, const char *mnemonic, uint8_t ep)
{
   const uint64_t key = (uint64_t)(uintptr_t)mnemonic & ((UINT64_C(1) << 48) - 1);
   const unsigned slot =
      ((((uintptr_t)mnemonic >> 3) ^ ((uintptr_t)t >> 6)) * 4u + ep) %
      AGX3_FORM_CACHE_SLOTS;
   const uint64_t entry =
      atomic_load_explicit(&agx3_form_cache[slot], memory_order_relaxed);
   if (entry && (entry >> 16) == key) {
      const uint32_t form = (uint32_t)(entry & 0xffff);
      /* Only the first form of (mnemonic, entry point) in table order is
       * what the search below returns. */
      if (form < t->num_forms && t->forms[form].ep == ep &&
          !strcmp(t->forms[form].mnemonic, mnemonic) &&
          (form == 0 || t->forms[form - 1].ep != ep ||
           strcmp(t->forms[form - 1].mnemonic, mnemonic)))
         return (int32_t)form;
   }

   uint32_t lo = 0, hi = t->num_forms;

   /* The forms are in (mnemonic, entry-point) order.  This lookup is on
    * every composed instruction, so do not linearly rescan the complete AGX3
    * form table for each instruction. */
   while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      const agx3_form *f = &t->forms[mid];
      const int mnemonic_cmp = strcmp(f->mnemonic, mnemonic);

      if (mnemonic_cmp < 0 || (mnemonic_cmp == 0 && f->ep < ep))
         lo = mid + 1;
      else
         hi = mid;
   }

   if (lo < t->num_forms && t->forms[lo].ep == ep &&
       !strcmp(t->forms[lo].mnemonic, mnemonic)) {
      if (lo <= 0xffff)
         atomic_store_explicit(&agx3_form_cache[slot], (key << 16) | lo,
                               memory_order_relaxed);
      return (int32_t)lo;
   }

   return -1;
}

/*
 * Apple's encoder tries a mnemonic's encoding points by ascending size, the
 * ep number breaking ties, and keeps the first that encodes the field vector
 * (the forms[].size order of the generated table; see agx3_compose.h). The
 * forms of one mnemonic are contiguous in a table, which is sorted by
 * (mnemonic, ep).
 */
int32_t
agx3_compose_apple_order(const agx3_table *t, const char *mnemonic,
                         const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out)
{
   int32_t lo = -1;
   for (uint8_t ep = 0; ep < 64 && lo < 0; ++ep)
      lo = agx3_form_index(t, mnemonic, ep);
   if (lo < 0)
      return -1;

   uint32_t first = (uint32_t)lo;
   while (first > 0 && strcmp(t->forms[first - 1].mnemonic, mnemonic) == 0)
      first--;
   uint32_t end = (uint32_t)lo;
   while (end < t->num_forms && strcmp(t->forms[end].mnemonic, mnemonic) == 0)
      end++;

   /* At most a few dozen forms per mnemonic: try them smallest first. */
   bool tried[64] = {false};
   if (end - first > 64)
      return -1;
   for (unsigned n = 0; n < end - first; ++n) {
      int32_t best = -1;
      for (uint32_t f = first; f < end; ++f) {
         if (tried[f - first])
            continue;
         if (best < 0 || t->forms[f].size < t->forms[best].size ||
             (t->forms[f].size == t->forms[best].size &&
              t->forms[f].ep < t->forms[best].ep))
            best = (int32_t)f;
      }
      tried[best - first] = true;
      if (agx3_compose(t, (uint32_t)best, fields, out))
         return best;
   }
   return -1;
}
