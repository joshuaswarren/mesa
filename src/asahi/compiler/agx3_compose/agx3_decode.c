/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx3_decode.c -- agx3_table.py's Decoder, in C. See agx3_decode.h.
 *
 * Every bound below fails closed: a path that would overflow one is treated
 * as not decoding, never truncated.
 */
#include "agx3_decode.h"

#include <stdlib.h>
#include <string.h>

#define DEC_MAX_NODES   96
#define DEC_MAX_PENDING 96
#define DEC_MAX_LUTS    8
#define DEC_MAX_PREDS   24
#define DEC_PATH_LIMIT  20000

struct dec_lut {
   int32_t off;
   uint16_t n;
   uint8_t src, width;
   uint64_t bits;
};

struct dec_field {
   uint64_t known_mask, known_value;
   uint8_t nluts, npreds;
   struct dec_lut luts[DEC_MAX_LUTS];
   uint32_t preds[DEC_MAX_PREDS];   /* in predrefs */
};

struct dec {
   const agx3_table *t;
   const uint8_t *data;
   unsigned size;
   uint32_t form;
   uint32_t nodes[DEC_MAX_NODES];
   unsigned nnodes;
   uint32_t pending[DEC_MAX_PENDING];
   unsigned npending;
   unsigned tried;
   bool aborted;
   uint64_t *fields;
   struct dec_field f[AGX3_MAX_FIELDS];
};

static inline uint64_t
wmask(unsigned width)
{
   return width >= 64 ? ~0ull : ((1ull << width) - 1);
}

static inline bool
word_at(const struct dec *d, unsigned w, uint16_t *out)
{
   if (2 * w + 1 >= d->size)
      return false;
   *out = d->data[2 * w] | (d->data[2 * w + 1] << 8);
   return true;
}

static bool
fixed_ok(const struct dec *d, uint32_t node)
{
   const agx3_node *n = &d->t->nodes[node];
   for (unsigned i = 0; i < n->fixed_n; ++i) {
      const agx3_fixed *fx = &d->t->fixeds[n->fixed_off + i];
      uint16_t w;
      if (!word_at(d, fx->word, &w) || (w & fx->mask) != fx->value)
         return false;
   }
   return true;
}

static bool
pred_holds(const struct dec *d, const agx3_pred *p, uint64_t v)
{
   switch (p->kind) {
   case 0:
      return v == p->a;
   case 1:
      for (uint64_t i = 0; i < p->b; ++i) {
         if (d->t->setpool[p->a + i] == v)
            return true;
      }
      return false;
   case 2:
      return p->a <= v && v <= p->b;
   case 3:
      return ((v >> p->shift) & wmask(p->width)) == p->a;
   default:
      return false;
   }
}

/* Field values that satisfy the deposits and predicates of d->nodes. */
static bool
solve(struct dec *d)
{
   const agx3_table *t = d->t;
   uint8_t touched[AGX3_MAX_FIELDS] = {0};
   memset(d->f, 0, sizeof(d->f));

   for (unsigned i = 0; i < d->nnodes; ++i) {
      const agx3_node *n = &t->nodes[d->nodes[i]];
      for (unsigned k = 0; k < n->pred_n; ++k) {
         const agx3_pred p = agx3_pred_at(t, n->pred_off + k);
         if (p.field >= AGX3_MAX_FIELDS)
            return false;
         struct dec_field *f = &d->f[p.field];
         if (f->npreds == DEC_MAX_PREDS)
            return false;
         f->preds[f->npreds++] = n->pred_off + k;
         touched[p.field] = 1;
      }
      for (unsigned k = 0; k < n->cmd_n; ++k) {
         const agx3_cmd *c = &t->cmds[n->cmd_off + k];
         uint16_t w;
         if (c->field >= AGX3_MAX_FIELDS || !word_at(d, c->word, &w))
            return false;
         struct dec_field *f = &d->f[c->field];
         const uint64_t bits = (w >> c->dst_shift) & wmask(c->width);
         touched[c->field] = 1;
         if (c->lut_n) {
            if (f->nluts == DEC_MAX_LUTS)
               return false;
            f->luts[f->nluts++] = (struct dec_lut){
               .off = c->lut_off, .n = c->lut_n,
               .src = c->src_shift, .width = c->width, .bits = bits,
            };
         } else {
            const uint64_t m = wmask(c->width) << c->src_shift;
            if (f->known_mask & m & (f->known_value ^ (bits << c->src_shift)))
               return false;
            f->known_mask |= m;
            f->known_value = (f->known_value & ~m) | (bits << c->src_shift);
         }
      }
   }

   memset(d->fields, 0, sizeof(uint64_t) * AGX3_MAX_FIELDS);
   for (unsigned fi = 0; fi < AGX3_MAX_FIELDS; ++fi) {
      if (!touched[fi])
         continue;
      struct dec_field *f = &d->f[fi];
      uint64_t km = f->known_mask, kv = f->known_value;

      /* Bitfield predicates fix slices of the value. */
      for (unsigned k = 0; k < f->npreds; ++k) {
         const agx3_pred p = agx3_pred_at(t, f->preds[k]);
         if (p.kind != 3)
            continue;
         const uint64_t m = wmask(p.width) << p.shift;
         if (km & m & (kv ^ ((uint64_t)p.a << p.shift)))
            return false;
         km |= m;
         kv = (kv & ~m) | ((uint64_t)p.a << p.shift);
      }

      /* Candidate values: an equality predicate's, else every LUT index up
       * to the shortest LUT, else the known bits. */
      uint64_t first = kv, count = 1;
      bool have_eq = false;
      for (unsigned k = 0; k < f->npreds; ++k) {
         const agx3_pred p = agx3_pred_at(t, f->preds[k]);
         if (p.kind == 0) {
            first = p.a;
            have_eq = true;
            break;
         }
      }
      if (!have_eq && f->nluts) {
         first = 0;
         count = UINT64_MAX;
         for (unsigned k = 0; k < f->nluts; ++k)
            count = f->luts[k].n < count ? f->luts[k].n : count;
      }

      bool found = false;
      for (uint64_t c = 0; c < count && !found; ++c) {
         const uint64_t v = first + c;
         if ((v & km) != kv)
            continue;
         bool good = true;
         for (unsigned k = 0; k < f->nluts && good; ++k) {
            const struct dec_lut *l = &f->luts[k];
            good = v < l->n &&
                   ((t->lutpool[l->off + v] >> l->src) & wmask(l->width)) ==
                      l->bits;
         }
         for (unsigned k = 0; k < f->npreds && good; ++k) {
            const agx3_pred p = agx3_pred_at(t, f->preds[k]);
            good = pred_holds(d, &p, v);
         }
         if (good) {
            d->fields[fi] = v;
            found = true;
         }
      }
      if (!found)
         return false;
   }
   return true;
}

/* Depth-first over the pending choice groups, in agx3_table.py's order: a
 * group's chosen alternative expands its own groups before the groups after
 * it, and later groups vary fastest. Pending is a stack, so groups are pushed
 * in reverse. When two paths reproduce the same bytes this picks the one the
 * Python decoder picks. Returns true to stop (found or aborted). */
static bool
expand(struct dec *d)
{
   if (d->npending == 0) {
      if (++d->tried > DEC_PATH_LIMIT) {
         d->aborted = true;
         return true;
      }
      uint8_t out[2 * AGX3_MAX_WORDS];
      return solve(d) && agx3_compose(d->t, d->form, d->fields, out) &&
             !memcmp(out, d->data, d->size);
   }

   const uint32_t g = d->pending[--d->npending];
   const agx3_group *grp = &d->t->groups[g];
   for (unsigned i = 0; i < grp->alt_n; ++i) {
      const uint32_t alt = grp->alt_off + i;
      if (!fixed_ok(d, alt))
         continue;
      const agx3_node *an = &d->t->nodes[alt];
      if (d->nnodes == DEC_MAX_NODES ||
          d->npending + an->grp_n > DEC_MAX_PENDING)
         continue;
      d->nodes[d->nnodes++] = alt;
      for (unsigned k = an->grp_n; k-- > 0;)
         d->pending[d->npending++] = an->grp_off + k;
      if (expand(d))
         return true;
      d->npending -= an->grp_n;
      d->nnodes--;
   }
   d->pending[d->npending++] = g;
   return false;
}

int32_t
agx3_decode(const agx3_table *t, const uint8_t *data, unsigned size,
            enum agx3_mnemonic mnemonic, uint64_t fields[AGX3_MAX_FIELDS])
{
   if (!t || !data || !fields || size < 2 || size > 2 * AGX3_MAX_WORDS)
      return -1;

   struct dec *d = calloc(1, sizeof(*d));
   if (!d)
      return -1;
   d->t = t;
   d->data = data;
   d->size = size;
   d->fields = fields;

   int32_t result = -1;
   for (uint32_t form = 0; form < t->num_forms && !d->aborted; ++form) {
      const agx3_form *fm = &t->forms[form];
      if (fm->size != size || (mnemonic && fm->mnemonic != mnemonic))
         continue;
      if (!fixed_ok(d, fm->node))
         continue;
      const agx3_node *root = &t->nodes[fm->node];
      if (root->grp_n > DEC_MAX_PENDING)
         continue;
      d->form = form;
      d->nnodes = 0;
      d->nodes[d->nnodes++] = fm->node;
      d->npending = 0;
      for (unsigned k = root->grp_n; k-- > 0;)
         d->pending[d->npending++] = root->grp_off + k;
      if (expand(d) && !d->aborted) {
         result = (int32_t)form;
         break;
      }
   }
   free(d);
   return result;
}
