/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The encoder selects an alternative of an exclusive choice group without
 * going back, and finds it through the group's key index when it has one.
 * That is only the old selection, the first alternative in table order whose
 * predicates hold, while no field vector satisfies two alternatives and the
 * index agrees with the predicates. gen_agx3_tables.py checks both when it
 * generates a table; this runs the encoder's own apply_group() on the
 * generated arrays, for every exclusive group of the three tables.
 *
 * A group is tried with the field vector of each of its alternatives, and
 * with every field an alternative tests moved one up and one down, which
 * reaches its neighbours in the index and vectors that no alternative takes.
 */

/* For apply_group(), which is static. */
#include "agx3_compose/agx3_compose.c"

#include <stdio.h>
#include <stdlib.h>

/* What apply_group() selects in g, as a position; g->alt_n for none. It runs
 * on a copy of the group whose alternative i only writes i. */
static unsigned
selected(const agx3_table *t, const agx3_group *g, agx3_node *nodes,
         const agx3_fixed *positions, const uint64_t *fields)
{
   agx3_table copy = *t;
   agx3_group group = *g;
   copy.nodes = nodes;
   copy.fixeds = positions;
   group.alt_off = 0;
   for (unsigned i = 0; i < g->alt_n; i++) {
      nodes[i] = (agx3_node){
         .pred_off = t->nodes[g->alt_off + i].pred_off,
         .pred_n = t->nodes[g->alt_off + i].pred_n,
         .fixed_off = (uint16_t)i,
         .fixed_n = 1,
      };
   }

   ctx c = {.t = &copy, .fields = fields, .nwords = 1};
   return apply_group(&c, &group) ? c.words[0] : g->alt_n;
}

/* The first alternative of g whose predicates hold, as a position. */
static unsigned
first_match(const agx3_table *t, const agx3_group *g, const uint64_t *fields,
            unsigned from)
{
   const agx3_node *alts = &t->nodes[g->alt_off];
   return (unsigned)(find_alt(t, fields, alts + from, alts + g->alt_n) - alts);
}

int
main(void)
{
   const agx3_table *tables[] = {&agx3b_table, &agx3c_table, &agx3d_table};
   unsigned long groups = 0, keyed = 0, selections = 0, failures = 0;

   static agx3_fixed positions[UINT16_MAX];
   static agx3_node nodes[UINT16_MAX];
   for (unsigned i = 0; i < UINT16_MAX; i++)
      positions[i] = (agx3_fixed){.mask = UINT16_MAX, .value = (uint16_t)i};

   for (unsigned table = 0; table < 3; table++) {
      const agx3_table *t = tables[table];
      /* The roots of the forms are the last nodes. */
      uint32_t num_nodes = 0;
      for (uint32_t f = 0; f < t->num_forms; f++)
         num_nodes = t->forms[f].node >= num_nodes ? t->forms[f].node + 1
                                                   : num_nodes;
      bool *seen = calloc(num_nodes, sizeof(bool));

      for (uint32_t n = 0; n < num_nodes; n++) {
         for (unsigned k = 0; k < t->nodes[n].grp_n; k++) {
            const agx3_group *g = &t->groups[t->nodes[n].grp_off + k];
            if (g->alt_off + g->alt_n > num_nodes) {
               fprintf(stderr, "FAIL agx3%c: alternatives after the forms\n",
                       "bcd"[table]);
               failures++;
               continue;
            }
            if (!g->exclusive || seen[g->alt_off])
               continue;
            seen[g->alt_off] = true;
            groups++;
            keyed += g->key_n != 0;

            for (unsigned i = 0; i < g->alt_n; i++) {
               const agx3_node *alt = &t->nodes[g->alt_off + i];
               uint64_t fields[AGX3_MAX_FIELDS] = {0};
               /* A vector the alternative holds for: bit fields last, into
                * what the other predicates of their field left. */
               for (unsigned pass = 0; pass < 2; pass++) {
                  for (unsigned j = 0; j < alt->pred_n; j++) {
                     const agx3_pred p = agx3_pred_at(t, alt->pred_off + j);
                     if (p.kind == 3 && pass == 1)
                        fields[p.field] |= (uint64_t)p.a << p.shift;
                     else if (p.kind != 3 && pass == 0)
                        fields[p.field] = p.kind == 1 ? t->setpool[p.a] : p.a;
                  }
               }

               /* The vector itself, then each tested field one up and one
                * down. */
               for (unsigned v = 0; v < 1u + 2u * alt->pred_n; v++) {
                  unsigned field = 0;
                  uint64_t was = 0;
                  if (v > 0) {
                     field = agx3_pred_at(t, alt->pred_off + (v - 1) / 2).field;
                     was = fields[field];
                     fields[field] += v % 2 ? 1 : -1;
                  }

                  const unsigned want = first_match(t, g, fields, 0);
                  const unsigned got = selected(t, g, nodes, positions, fields);
                  const bool two =
                     want < g->alt_n &&
                     first_match(t, g, fields, want + 1) < g->alt_n;
                  const char *why =
                     got != want ? "is not the first that holds" :
                     two ? "is not the only one that holds" :
                     v == 0 && want != i ? "is not the alternative the "
                                           "vector was made for" : NULL;
                  selections++;
                  if (why && failures++ < 20)
                     fprintf(stderr,
                             "FAIL agx3%c group at node %u, vector %u of "
                             "alternative %u: selects %u, which %s; the "
                             "first that holds is %u\n",
                             "bcd"[table], g->alt_off, v, i, got, why, want);
                  if (v > 0)
                     fields[field] = was;
               }
            }
         }
      }
      free(seen);
   }

   if (!keyed) {
      fprintf(stderr, "FAIL no group has a key index\n");
      failures++;
   }
   printf("%lu exclusive groups, %lu with a key index: %lu selections, "
          "%lu failures\n", groups, keyed, selections, failures);
   return failures ? 1 : 0;
}
