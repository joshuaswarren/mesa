/*
 * agx3_compose.h -- generic table-driven AGX3 instruction encoder.
 *
 * The tables are the AGX3 instruction encoding tables, one per encoding.
 * gen_agx3_tables.py generates agx3{b,c,d}_tables.c from the checked-in
 * agx3{b,c,d}_tables.txt at build time; see README.md.  One encoder packs
 * every form of all three encodings; there is no per-opcode code.
 *
 * Field ids are shared by the three tables, but ep numbering, form sizes and
 * the set of fields each mnemonic consumes are not: a G17G field vector is
 * not a G18G one.  Apple's encoder tries a mnemonic's forms by ascending
 * size, ep breaking ties, which differs from ep order for some mnemonics of
 * each table (agx3d: the branches, rld, f32fma, f32mul, ...), so callers that need Apple's choice pick the form themselves.
 *
 * Semantics mirror composer.py exactly and fail closed:
 *  - the caller supplies the tables' anonymous field values
 *    (u64 fields[160]);
 *  - root predicates gate the form; each choice group selects the first
 *    alternative, in table order, whose predicates hold and whose nested
 *    groups resolve (looked up in the group's key index when it has one and
 *    the first few alternatives do not hold);
 *  - all fixed-bit and field deposits share a written-bit shadow mask and
 *    must agree, otherwise the encode is rejected;
 *  - a nonzero field that no selected command or predicate consumes is
 *    rejected; a direct-deposited field with value bits outside the union of
 *    its source slices is rejected.
 */
#ifndef AGX3_COMPOSE_H
#define AGX3_COMPOSE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define AGX3_MAX_FIELDS 160
#define AGX3_MAX_WORDS  16

/* A node's rows are runs in the row arrays: fixeds[fixed_off .. fixed_off +
 * fixed_n), and the same for predrefs, cmds and groups.  Identical runs are
 * stored once.  The members are as wide as the data needs: gen_agx3_tables.py
 * refuses a value that does not fit. */
typedef struct { uint16_t mask, value; uint8_t word; } agx3_fixed;
/* kind: 0=eq(a) 1=set(pool a, n=b) 2=range[a,b] 3=bitfield_eq(shift,width,a) */
typedef struct { uint64_t b; uint16_t a; uint8_t field, kind, shift, width; } agx3_pred;
/* One predicate of a node.  With field < AGX3_MAX_FIELDS it is the predicate
 * itself, fields[field] == value, which nearly all are.  Any other one is
 * preds[(field - AGX3_MAX_FIELDS) << 8 | value]. */
typedef struct { uint8_t field, value; } agx3_predref;
typedef struct {
   uint16_t lut_off;
   uint8_t lut_n;                /* 0: direct deposit */
   uint8_t field, word, dst_shift, width, src_shift;
} agx3_cmd;
/* The alternatives are nodes[alt_off .. alt_off + alt_n), in table order.
 * exclusive: no field vector satisfies the predicates of two of them.
 * With key_n != 0 the group is exclusive and has a key index at
 * keys[key_off]: key_n field ids, then alt_n rows sorted by their first key_n
 * bytes.  A row is the values those fields must have for one alternative to
 * hold, then that alternative's position in the group, high byte first.  No
 * two rows have the same values. */
typedef struct {
   uint32_t alt_off, key_off;
   uint16_t alt_n;
   uint8_t key_n;
   bool exclusive;
} agx3_group;
typedef struct {
   uint32_t pred_off;
   uint16_t fixed_off, cmd_off, grp_off;
   uint8_t fixed_n, pred_n, cmd_n, grp_n;
} agx3_node;
typedef struct { const char *mnemonic; uint8_t ep, size; uint32_t node; } agx3_form;

typedef struct {
   const agx3_form *forms; uint32_t num_forms;
   const agx3_node *nodes;
   const agx3_group *groups;
   const agx3_fixed *fixeds;
   const agx3_predref *predrefs;
   const agx3_pred *preds;
   const agx3_cmd *cmds;
   const uint8_t *lutpool;
   const uint8_t *setpool;
   const uint8_t *keys;
} agx3_table;

/* The preds row that a predref with field >= AGX3_MAX_FIELDS names. */
static inline const agx3_pred *
agx3_wide_pred(const agx3_table *t, agx3_predref r)
{
   return &t->preds[(r.field - AGX3_MAX_FIELDS) << 8 | r.value];
}

/* Predicate i of the predrefs array, in full. */
static inline agx3_pred
agx3_pred_at(const agx3_table *t, uint32_t i)
{
   const agx3_predref r = t->predrefs[i];
   if (r.field >= AGX3_MAX_FIELDS)
      return *agx3_wide_pred(t, r);
   agx3_pred p = {0};
   p.field = r.field;
   p.a = r.value;
   return p;
}

extern const agx3_table agx3b_table;
extern const agx3_table agx3c_table;
extern const agx3_table agx3d_table;

/* Pack forms[form] from fields[0..159].  On success writes size bytes into
 * out and returns true; any violation of the contract returns false with out
 * untouched. */
bool agx3_compose(const agx3_table *t, uint32_t form,
                  const uint64_t fields[AGX3_MAX_FIELDS], uint8_t *out);

/* agx3_compose() for a caller that knows which fields are nonzero: bit i of
 * nonzero is set exactly when fields[i] != 0. */
bool agx3_compose_known(const agx3_table *t, uint32_t form,
                        const uint64_t fields[AGX3_MAX_FIELDS],
                        const uint64_t nonzero[3], uint8_t *out);

int32_t agx3_form_index(const agx3_table *t, const char *mnemonic, uint8_t ep);

/* The form Apple's encoder picks: this mnemonic's forms by ascending size,
 * ep breaking ties, first that encodes. Writes the bytes to out; -1 if none. */
int32_t agx3_compose_apple_order(const agx3_table *t, const char *mnemonic,
                                 const uint64_t fields[AGX3_MAX_FIELDS],
                                 uint8_t *out);

#endif
