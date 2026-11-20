#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Generate agx3{b,c,d}_tables.c from agx3{b,c,d}_tables.txt.

A node of the C tables names its fixed bits, predicates, commands and choice
groups as runs in the row arrays, and identical runs are stored once. A choice
group's alternatives are consecutive nodes. A predicate that tests a field
for a small value, as nearly all do, is two bytes in its run; any other is a
row of the preds array that the run names.

--verify expands the arrays this prints back into one node tree per form and
compares them with the tables these descriptions were made from, the
agx3{b,c,d}_tables.c of commit afcc08b71: same rows, same order, by the
SHA-256 of that file from its #include line on. A description that is changed
on purpose no longer has those rows: delete its line of LEGACY_SHA256 in the
same change, and --verify has nothing to compare that table with.
"""

import argparse
import hashlib

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


def dense(name, groups, forms):
    """The arrays of the C table, as lists of tuples in struct field order."""
    pools = {k: Pool() for k in ('fixeds', 'predrefs', 'preds', 'cmds',
                                 'groups', 'lutpool', 'setpool')}
    D = {k: p.rows for k, p in pools.items()}
    # A group's alternatives are the nodes from here on.
    alt_off = [0]
    for alts in groups:
        alt_off.append(alt_off[-1] + len(alts))

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
        cmds = [((pools['lutpool'].add(lut), len(lut)) if lut else (0, 0)) +
                (field, word, dst_shift, width, src_shift)
                for field, word, dst_shift, width, src_shift, lut in cmds]
        # lut_n 0 is a direct deposit.
        require(all(lut is None or len(lut) for *_, lut in n[2]),
                'a deposit with an empty value list')
        return (pools['predrefs'].add(tuple(predref(*p) for p in preds)),
                pools['fixeds'].add(tuple((mask, value, word)
                                          for word, mask, value in fixeds)),
                pools['cmds'].add(tuple(cmds)),
                pools['groups'].add(tuple((alt_off[g], len(groups[g]))
                                          for g in grefs)),
                len(fixeds), len(preds), len(cmds), len(grefs))

    D['nodes'] = [node(a) for alts in groups for a in alts]
    D['forms'] = []
    for mnemonic, ep, size, root in forms:
        D['forms'].append((mnemonic, ep, size, len(D['nodes'])))
        D['nodes'].append(node(root))
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
        (pred_off, fixed_off, cmd_off, grp_off,
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

    return ([[node(off + i) for i in range(n)] for off, n in starts],
            [(mnemonic, ep, size, node(root))
             for mnemonic, ep, size, root in D['forms']])


# C type of each array, and the bits each struct member holds.
LAYOUT = {
    'fixeds': ('agx3_fixed', (16, 16, 8)),
    'predrefs': ('agx3_predref', (8, 8)),
    'preds': ('agx3_pred', (64, 16, 8, 8, 8, 8)),
    'cmds': ('agx3_cmd', (16, 8, 8, 8, 8, 8, 8)),
    'groups': ('agx3_group', (32, 16)),
    'nodes': ('agx3_node', (32, 16, 16, 16, 8, 8, 8, 8)),
    'forms': ('agx3_form', (None, 8, 8, 32)),
    'lutpool': ('uint8_t', 8),
    'setpool': ('uint8_t', 8),
}


def cell(name, value, bits):
    if bits is None:
        return '"%s"' % value
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
    out.write('const agx3_table %s_table = {\n'
              '   forms, %d, nodes, groups, fixeds, predrefs, preds, cmds,\n'
              '   lutpool, setpool,\n'
              '};\n' % (name, len(D['forms'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('description')
    parser.add_argument('-o', '--output')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    name, groups, forms = agx3_table.parse(args.description)
    D = dense(name, groups, forms)
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
