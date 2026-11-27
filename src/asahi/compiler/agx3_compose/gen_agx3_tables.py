#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Generate agx3{b,c,d}_tables.c from agx3{b,c,d}_tables.txt.

A node of the C tables names its fixed bits, predicates, commands and choice
groups as runs in the row arrays, and identical runs are stored once. A choice
group's alternatives are consecutive nodes. A predicate that tests a field
for a small value, as nearly all do, is two bytes in its run; any other is a
row of the preds array that the run names. A large group whose alternatives
can be told apart by field values has them indexed by those values (see
key_index()).

--verify expands the arrays this prints back into one node tree per form and
compares them with the tables these descriptions were made from, the
agx3{b,c,d}_tables.c of commit afcc08b71: same rows, same order, by the
SHA-256 of that file from its #include line on. A description that is changed
on purpose no longer has those rows: delete its line of LEGACY_SHA256 in the
same change, and --verify has nothing to compare that table with.

Every run, with or without --verify, checks the exclusive flags and the key
indexes it prints against the predicates of the alternatives (check_keys()),
and what it prints of each node's consumption against the node's rows
(check_uses(), check_slices()), and the shape of every row (check_rows()).
The encoder trusts them.
"""

import argparse
import hashlib
import itertools

import agx3_table

LEGACY_SHA256 = {
    'agx3b': 'ceb8a1490a8fa996324516291225139a803d219ca70b753dfbbf8a426fa1289a',
    'agx3c': 'ac49d4a76f0e47d28c2245c3bd09b9e9a6964fef9d68eaf56e5ba03e790733a3',
    'agx3d': 'edc9cd6548ff68f3851d7fb69b3394573c1e76bf0c7282a2998cd6114045ce2e',
}


def require(ok, what, *args):
    """Refuse the description. Not an assert, which python -O skips."""
    if not ok:
        raise ValueError(what % args)


class Pool:
    """Rows or runs stored once: add() returns the offset of the first copy."""

    def __init__(self):
        self.rows, self.offsets = [], {}

    def add(self, run):
        if run not in self.offsets:
            self.offsets[run] = len(self.rows)
            self.rows += run
        return self.offsets[run]


# AGX3_KEY_WALK of agx3_compose.c: the encoder walks that many alternatives of
# a group before it uses the key index, so a group no longer than that gets
# none. The two numbers only have to agree for speed and size. With different
# ones the encoder walks a group to its end where it could have searched, or
# has no index where it would have used one, and selects the same alternative.
KEY_WALK = 8


def key_index(alts):
    """(fields, rows) that tell the alternatives of a group apart, or None.

    fields are the fields every alternative tests for equality, less those
    all of them test for the same value. A row is an alternative's values for
    them, then its position in the group; rows are sorted. An alternative
    holds only if the field vector has exactly its values. So when no two
    rows have the same values, at most one alternative can hold for any field
    vector: the group is exclusive. The encoder then has nothing to go back
    to when that alternative does not apply, and may find it in any order.
    """
    eqs = []
    for preds, *_ in alts:
        eq = {}
        for field, kind, _, _, a, _ in preds:
            if kind == 0:
                require(eq.setdefault(field, a) == a,
                        'an alternative tests field %d for %d and for %d',
                        field, eq[field], a)
        eqs.append(eq)
    fields = sorted(f for f in set(eqs[0]).intersection(*eqs[1:])
                    if len(set(eq[f] for eq in eqs)) > 1)
    rows = sorted((tuple(eq[f] for f in fields), i) for i, eq in enumerate(eqs))
    if not fields or len(set(k for k, _ in rows)) < len(rows):
        return None
    require(len(fields) < 256, 'a group with %d key fields', len(fields))
    require(fields[-1] < agx3_table.NUM_FIELDS, 'no field %d', fields[-1])
    return fields, rows


def direct_slices(node):
    """The bits of each field that the direct deposits of a node take."""
    slices = {}
    for field, _, _, width, src_shift, lut in node[2]:
        if lut is None:
            slices[field] = (slices.get(field, 0) |
                             agx3_table.wmask(width) << src_shift)
    return slices


def check_slices(groups, forms):
    """No two nodes that one instruction can apply deposit the same field
    directly.

    The encoder tests a value against the slices of the node that deposits
    it (agx3_cmd in agx3_compose.h), which is the test against all direct
    deposits of the field only then. Checked without the predicates: two
    groups of a node, or a node and a group under it, count as applied
    together whatever their alternatives test.
    """
    under = {}

    def fields(node, what):
        seen = set(direct_slices(node))
        for g in node[3]:
            if g not in under:
                under[g] = set().union(*(fields(alt, 'group %d' % g)
                                         for alt in groups[g]))
            require(not seen & under[g],
                    '%s: fields %s are deposited directly in two nodes',
                    what, sorted(seen & under[g]))
            seen |= under[g]
        return seen

    for mnemonic, ep, _, root in forms:
        fields(root, 'form %s %d' % (mnemonic, ep))


def check_rows(groups, forms):
    """What the encoder does not test when it applies a row: a field id is
    one, a deposit stays inside its 16-bit word, and a form reaches no
    deposit into a word it does not have."""
    for node in [alt for alts in groups for alt in alts] + [f[3] for f in forms]:
        preds, fixeds, cmds, _ = node
        for field in [p[0] for p in preds] + [c[0] for c in cmds]:
            require(0 <= field < agx3_table.NUM_FIELDS, 'no field %d', field)
        for _, _, dst_shift, width, src_shift, _ in cmds:
            require(width >= 1 and dst_shift + width <= 16 and src_shift < 64,
                    'a deposit of %d bits from bit %d to bit %d',
                    width, src_shift, dst_shift)

    under = {}

    def last_word(node):
        words = [f[0] for f in node[1]] + [c[1] for c in node[2]]
        for g in node[3]:
            if g not in under:
                under[g] = max(last_word(alt) for alt in groups[g])
            words.append(under[g])
        return max(words, default=-1)

    for mnemonic, ep, size, root in forms:
        require(size % 2 == 0 and last_word(root) < size // 2,
                'form %s %d of %d bytes deposits into word %d',
                mnemonic, ep, size, last_word(root))


def dense(name, groups, forms):
    """The arrays of the C table, as lists of tuples in struct field order."""
    pools = {k: Pool() for k in ('fixeds', 'predrefs', 'preds', 'cmds',
                                 'groups', 'lutpool', 'setpool', 'keys',
                                 'uses', 'slices')}
    D = {k: p.rows for k, p in pools.items()}
    # A group's alternatives are the nodes from here on.
    alt_off = [0]
    for alts in groups:
        alt_off.append(alt_off[-1] + len(alts))

    def group(g):
        index = key_index(groups[g])
        # The encoder walks a group of up to KEY_WALK alternatives to its
        # end, so only a longer one gets its index stored.
        stored = index and len(groups[g]) > KEY_WALK
        fields, rows = index if stored else ((), ())
        # The key fields, then each row: its values and its position, high
        # byte first.
        key = tuple(fields) + tuple(
            b for values, i in rows for b in values + (i >> 8, i & 0xff))
        return (alt_off[g], pools['keys'].add(key), len(groups[g]),
                len(fields), int(index is not None))

    group_rows = [group(g) for g in range(len(groups))]

    def predref(field, kind, shift, width, a, b):
        if kind == 0 and a < 256:
            require(field < agx3_table.NUM_FIELDS, 'no field %d', field)
            return (field, a)
        row = ((b, a, field, kind, shift, width) if kind != 1 else
               (len(a), pools['setpool'].add(a), field, kind, shift, width))
        i = pools['preds'].add((row,))
        # The row number has to fit in the field ids no field has, and a byte.
        require(agx3_table.NUM_FIELDS + (i >> 8) < 256,
                'predicate row %d has no predref', i)
        return (agx3_table.NUM_FIELDS + (i >> 8), i & 0xff)

    def node(n):
        preds, fixeds, cmds, grefs = n
        # lut_n 0 is a direct deposit.
        require(all(lut is None or len(lut) for *_, lut in cmds),
                'a deposit with an empty value list')
        # What the encoder checks after its walk, resolved here: the bits of
        # each field that this node's direct deposits take, and the fields
        # the node consumes.
        slices = direct_slices(n)
        whole = set(p[0] for p in preds) | set(c[0] for c in cmds if c[5])
        consumed = whole | set(slices)
        use = tuple(sum(1 << (f - w) for f in fields if w <= f < w + 64)
                    for fields in (consumed, whole) for w in (0, 64, 128))
        cmds = [((pools['lutpool'].add(lut), len(lut)) if lut else
                 (pools['slices'].add((slices[field],)), 0)) +
                (field, word, dst_shift, width, src_shift)
                for field, word, dst_shift, width, src_shift, lut in cmds]
        return (pools['predrefs'].add(tuple(predref(*p) for p in preds)),
                pools['fixeds'].add(tuple((mask, value, word)
                                          for word, mask, value in fixeds)),
                pools['cmds'].add(tuple(cmds)),
                pools['groups'].add(tuple(group_rows[g] for g in grefs)),
                pools['uses'].add(use),
                len(fixeds), len(preds), len(cmds), len(grefs))

    D['nodes'] = [node(a) for alts in groups for a in alts]
    D['forms'] = []
    for mnemonic, ep, size, root in forms:
        D['forms'].append((len(D['nodes']), mnemonic, ep, size))
        D['nodes'].append(node(root))
    # agx3_form_index() relies on this order.
    require(forms == sorted(forms, key=lambda f: (f[0].encode(), f[1])),
            'the forms are not in (mnemonic, ep) order')
    return D


def expand(D):
    """dense() backwards, from the arrays alone."""
    # The groups that a node names, numbered in the order of their first
    # alternative, which is the order of the description.
    starts = sorted(set(D['groups']))
    ids = {g: i for i, g in enumerate(starts)}

    def pred(field, value):
        if field < agx3_table.NUM_FIELDS:
            return (field, 0, 0, 0, value, 0)
        b, a, field, kind, shift, width = D['preds'][
            (field - agx3_table.NUM_FIELDS) << 8 | value]
        return (field, kind, shift, width) + (
            (a, b) if kind != 1 else (tuple(D['setpool'][a:a + b]), 0))

    def node(i):
        (pred_off, fixed_off, cmd_off, grp_off, _,
         fixed_n, pred_n, cmd_n, grp_n) = D['nodes'][i]
        return (
            tuple(pred(*r) for r in D['predrefs'][pred_off:pred_off + pred_n]),
            tuple((word, mask, value) for mask, value, word
                  in D['fixeds'][fixed_off:fixed_off + fixed_n]),
            tuple((field, word, dst_shift, width, src_shift,
                   tuple(D['lutpool'][lut_off:lut_off + lut_n]) or None)
                  for lut_off, lut_n, field, word, dst_shift, width, src_shift
                  in D['cmds'][cmd_off:cmd_off + cmd_n]),
            tuple(ids[g] for g in D['groups'][grp_off:grp_off + grp_n]))

    return ([[node(off + i) for i in range(n)] for off, _, n, *_ in starts],
            [(mnemonic, ep, size, node(root))
             for root, mnemonic, ep, size in D['forms']])


def check_keys(D, groups):
    """The exclusive flags and key indexes of D against the alternatives of
    groups."""
    alt_off = 0
    rows_of = {(g[0], g[2]): g for g in D['groups']}
    for g, alts in enumerate(groups):
        alt_n = len(alts)
        # A group that no node names has no row.
        row = rows_of.get((alt_off, alt_n))
        alt_off += alt_n
        if row is None:
            continue
        _, key_off, _, key_n, exclusive = row
        if not exclusive:
            require(not key_n, 'group %d: a key index, but not exclusive', g)
        elif not key_n:
            # Every two alternatives test some field for different values.
            eqs = [{field: a for field, kind, _, _, a, _ in preds if kind == 0}
                   for preds, *_ in alts]
            require(all(any(f in y and y[f] != v for f, v in x.items())
                        for x, y in itertools.combinations(eqs, 2)),
                    'group %d: exclusive, but two alternatives can hold', g)
        else:
            first, width = key_off + key_n, key_n + 2
            fields = D['keys'][key_off:first]
            rows = [tuple(D['keys'][first + r * width:first + (r + 1) * width])
                    for r in range(alt_n)]
            # Strictly ascending keys: no two alternatives share one.
            require(all(a[:key_n] < b[:key_n] for a, b in zip(rows, rows[1:])),
                    'group %d: its key rows do not ascend', g)
            require(sorted(r[-2] << 8 | r[-1] for r in rows) ==
                    list(range(alt_n)),
                    'group %d: its key rows do not name every alternative', g)
            require(all((field, 0, 0, 0, value, 0)
                        in alts[r[-2] << 8 | r[-1]][0]
                        for r in rows for field, value in zip(fields, r)),
                    'group %d: a key value its alternative does not test', g)


def check_uses(D):
    """The use masks and slice rows of D against the rows of each node."""
    def bits(fields):
        return [sum(1 << (f - w) for f in fields if w <= f < w + 64)
                for w in (0, 64, 128)]

    for pred_off, _, cmd_off, _, use_off, _, pred_n, cmd_n, _ in D['nodes']:
        whole, slices = set(), {}
        for field, value in D['predrefs'][pred_off:pred_off + pred_n]:
            if field >= agx3_table.NUM_FIELDS:
                field = D['preds'][(field - agx3_table.NUM_FIELDS) << 8
                                   | value][2]
            whole.add(field)
        cmds = D['cmds'][cmd_off:cmd_off + cmd_n]
        for _, lut_n, field, _, _, width, src_shift in cmds:
            if lut_n:
                whole.add(field)
            else:
                slices[field] = (slices.get(field, 0) |
                                 agx3_table.wmask(width) << src_shift)
        require(D['uses'][use_off:use_off + 6] ==
                bits(whole | set(slices)) + bits(whole),
                'a node whose use masks are not what its rows consume')
        require(all(lut_n or D['slices'][lut_off] == slices[field]
                    for lut_off, lut_n, field, *_ in cmds),
                'a direct deposit whose slices are not its node\'s')


# C type of each array, and the bits each struct member holds.
LAYOUT = {
    'fixeds': ('agx3_fixed', (16, 16, 8)),
    'predrefs': ('agx3_predref', (8, 8)),
    'preds': ('agx3_pred', (64, 16, 8, 8, 8, 8)),
    'cmds': ('agx3_cmd', (16, 8, 8, 8, 8, 8, 8)),
    'groups': ('agx3_group', (32, 32, 16, 8, 1)),
    'nodes': ('agx3_node', (32, 16, 16, 16, 16, 8, 8, 8, 8)),
    'forms': ('agx3_form', (32, None, 8, 8)),
    'lutpool': ('uint8_t', 8),
    'setpool': ('uint8_t', 8),
    'keys': ('uint8_t', 8),
    'uses': ('uint64_t', 64),
    'slices': ('uint64_t', 64),
}


def cell(name, value, bits):
    if bits is None:
        return 'AGX3_MN_' + value
    # A value that outgrows its member needs a wider type in agx3_compose.h.
    require(0 <= value < 1 << bits, '%s: %d does not fit in %d bits',
            name, value, bits)
    return '%d' % value if bits < 64 else '0x%xull' % value


def print_c(name, D, out):
    out.write('/* Generated by gen_agx3_tables.py from %s_tables.txt; '
              'do not edit. */\n#include "agx3_compose.h"\n' % name)
    for array, (ctype, bits) in LAYOUT.items():
        out.write('static const %s %s[] = {\n' % (ctype, array))
        # An empty array still needs an element to be valid C.
        if isinstance(bits, tuple):
            for row in D[array] or [(0,) * len(bits)]:
                out.write('{%s},\n' % ','.join(
                    cell(array, v, b) for v, b in zip(row, bits)))
        else:
            vals = [cell(array, v, bits) for v in D[array]] or ['0']
            for i in range(0, len(vals), 16):
                out.write(','.join(vals[i:i + 16]) + ',\n')
        out.write('};\n')
    # One more than the index of each mnemonic's first form.
    out.write('static const uint16_t mnemonic_forms[AGX3_MN_COUNT] = {\n')
    for i, form in enumerate(D['forms']):
        if i == 0 or D['forms'][i - 1][1] != form[1]:
            out.write('[AGX3_MN_%s] = %s,\n' %
                      (form[1], cell('mnemonic_forms', i + 1, 16)))
    out.write('};\n')
    # Each mnemonic's forms by ascending size, ep breaking ties.
    order = sorted(range(len(D['forms'])), key=lambda i: (
        D['forms'][i][1].encode(), D['forms'][i][3], D['forms'][i][2]))
    out.write('static const uint16_t apple_order[] = {\n%s,\n};\n' % ','.join(
        cell('apple_order', i, 16) for i in order))
    out.write('const agx3_table %s_table = {\n'
              '   forms, %d, mnemonic_forms, apple_order, nodes, groups, fixeds,\n'
              '   predrefs,\n'
              '   preds, cmds, lutpool, setpool, keys, uses, slices,\n'
              '};\n' % (name, len(D['forms'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('description')
    parser.add_argument('-o', '--output')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    name, groups, forms = agx3_table.parse(args.description)
    check_rows(groups, forms)
    check_slices(groups, forms)
    D = dense(name, groups, forms)
    check_keys(D, groups)
    check_uses(D)
    if args.verify and name in LEGACY_SHA256:
        rows = agx3_table.legacy_c(agx3_table.unfold(name, *expand(D)))
        got = hashlib.sha256(rows.encode()).hexdigest()
        if got != LEGACY_SHA256[name]:
            raise SystemExit('%s: the dense table expands to rows with '
                             'SHA-256 %s, not %s' %
                             (name, got, LEGACY_SHA256[name]))
    if args.output:
        with open(args.output, 'w') as out:
            print_c(name, D, out)


if __name__ == '__main__':
    main()
