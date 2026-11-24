#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Generate agx3d_transcode_vectors.c from the committed witness texts.

  gen_agx3d_transcode_vectors.py <out.c>

Every G17G/G18G pair of applegpu texts in native/g18g (Apple's compiler output
for the same Metal source on M5 and M6) is split into instructions, decoded
with each encoding's table, and aligned by mnemonic. A pair becomes a vector
when the G17G field vector, after agx3_transcode.c's G18G rules, equals the
G18G one: Apple's compiler chose the same instruction on both, so re-encoding
the G17G bytes must reproduce Apple's G18G bytes exactly. Where the two
differ only in the scoreboard slot (field 12, which each compiler allocates on
its own; this is how every lddev pair looks), the vector is marked so the test
compares the decoded G18G fields with field 12 left out. The expected values
are Apple's, never the encoder's own output.

b/bn are never vectors on their own: a displacement belongs to its program.
Each G17G text with branches instead becomes a program vector listing, per
branch, the instruction Apple's G18G counterpart branch lands on (mapped back
through the alignment); the re-encoded program must land on the same one.
"""

import collections
import difflib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(HERE, 'native', 'g18g')
COMPOSE = os.path.join(HERE, '..', 'agx3_compose')
ISA_DIR = os.path.join(HERE, '..', '..', 'isa')
sys.path[:0] = [HERE, COMPOSE, ISA_DIR]
import agx3_table  # noqa: E402
from isa import load_isa  # noqa: E402
from gen_agx3d_compose_corpus_vectors import segment  # noqa: E402

ARCH = {'applegpu_g17g': ('AGX3c.xml', 'agx3c_tables.txt'),
        'applegpu_g18g': ('AGX3d.xml', 'agx3d_tables.txt')}
CODEC = {}
for arch, (xml, table) in ARCH.items():
    T = agx3_table.load(os.path.join(COMPOSE, table))
    CODEC[arch] = (load_isa(os.path.join(ISA_DIR, xml)), T,
                   agx3_table.Decoder(T))


FLAG_SLOT_FIELD = {'icmp': 31, 'fcmp': 35, 'cndst': 1, 'cndlt': 1,
                   'cndcf': 1, 'cndef': 1}
BRANCHES = ('b', 'bn')


def g18g_rules(mnemonic, fields):
    """agx3_transcode.c's shader-code rules, in Python. Only the flag slots
    the witnesses use (0, 1 and the always-true 6) are modelled; the C side's
    fold of slots 2..5 needs a whole program."""
    v = list(fields)
    field = FLAG_SLOT_FIELD.get(mnemonic)
    if field is not None:
        if v[field] == 6 and mnemonic in ('cndst', 'cndef'):
            v[1], v[0] = 0, v[0] ^ 1
        elif v[field] <= 1:
            v[field] += 1
        else:
            return None
    if mnemonic == 'lddev':
        v[87] = 2 if v[75] else 0
    elif mnemonic == 'smp':
        if v[77] not in (0, 2):
            return None
        v[57], v[158] = 1, 4
        if v[77] == 0:
            v[56] = 2
        else:
            v[19] = 4
        v[62] |= v[69]
        v[69] = 0
    return v


def decode(arch, data, symbols, size):
    isa, T, D = CODEC[arch]
    out = []
    syms = sorted(symbols.values())
    for start, end in zip(syms, syms[1:] + [size]):
        for off, b, mn in segment(isa, data[start:end]):
            cands = D.candidates(b)
            pref = ([f for f in cands if T['forms'][f][0] == mn] +
                    [f for f in cands if T['forms'][f][0] != mn])
            res = D.decode(b, pref)
            if res is None:
                sys.exit('%s+0x%x does not decode' % (arch, start + off))
            out.append((b, T['forms'][res[0]][0], list(res[1]), start + off))
    return out


def sext48(x):
    return x - (1 << 48) if x & (1 << 47) else x


def branch_targets(text):
    """{instruction index of a b/bn: instruction index it lands on}."""
    index = {x[3]: k for k, x in enumerate(text)}
    index[text[-1][3] + len(text[-1][0])] = len(text)
    return {k: index.get(x[3] + sext48(x[2][0]))
            for k, x in enumerate(text) if x[1] in BRANCHES}


def main():
    manifest = json.load(open(os.path.join(NATIVE, 'manifest.json')))['texts']
    pairs = collections.defaultdict(dict)
    for fn, m in manifest.items():
        pairs[fn.rsplit('.', 2)[0]][m['arch']] = (fn, m)

    vectors = {}
    programs = []
    for stem in sorted(pairs):
        p = pairs[stem]
        if set(p) != set(ARCH):
            continue
        t = {}
        raw = {}
        for arch, (fn, m) in p.items():
            raw[arch] = open(os.path.join(NATIVE, fn), 'rb').read()
            t[arch] = decode(arch, raw[arch], m['symbols'], m['size'])
        c, d = t['applegpu_g17g'], t['applegpu_g18g']
        sm = difflib.SequenceMatcher(a=[x[1] for x in c], b=[x[1] for x in d],
                                     autojunk=False)
        aligned = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                aligned.update(zip(range(i1, i2), range(j1, j2)))

        # Branches: where Apple's G18G branch lands on the counterpart of the
        # instruction the G17G branch lands on, the re-encoded program must
        # land there too. The expected index is read off Apple's G18G text.
        back = {j: i for i, j in aligned.items()}
        back[len(d)] = len(c)
        ct, dt = branch_targets(c), branch_targets(d)
        checks = []
        for i, target in sorted(ct.items()):
            j = aligned.get(i)
            if j is None or dt.get(j) is None:
                continue
            expected = back.get(dt[j])
            if expected is not None and expected == target:
                checks.append((i, expected))
        if checks:
            programs.append((stem, raw['applegpu_g17g'], checks))

        # A branch's displacement is a property of the whole program, so
        # branches are only checked above, never as lone instructions.
        for i, j in aligned.items():
            if c[i][1] in BRANCHES:
                continue
            ruled = g18g_rules(c[i][1], c[i][2])
            if ruled is None:
                continue
            if ruled == d[j][2]:
                vectors.setdefault((c[i][0], d[j][0]), (c[i][1], stem, 0xff))
            elif [x for k, x in enumerate(ruled) if k != 12] == \
                 [x for k, x in enumerate(d[j][2]) if k != 12]:
                vectors.setdefault((c[i][0], d[j][0]), (c[i][1], stem, 12))

    out = ['/*',
           ' * Copyright 2026 Aurora Silicon',
           ' * SPDX-License-Identifier: MIT',
           ' *',
           ' * GENERATED by gen_agx3d_transcode_vectors.py -- do not edit.',
           ' * %d distinct G17G instructions whose G18G re-encoding must equal'
           % len(vectors),
           ' * the instruction Apple compiled for the same Metal source on G18G.',
           ' */',
           '#include <stdint.h>',
           '',
           'struct agx3d_transcode_vector {',
           '   const char *mnemonic;',
           '   const char *text;',
           '   uint8_t ignore_field; /* 0xff: compare bytes */',
           '   uint8_t g17g_len, g18g_len;',
           '   uint8_t g17g[32], g18g[32];',
           '};',
           '',
           'static const struct agx3d_transcode_vector agx3d_transcode_vectors[] = {']
    for (bc, bd), (mn, stem, ignore) in sorted(vectors.items(),
                                               key=lambda kv: (kv[1][0], kv[0])):
        hexlist = lambda b: ','.join('0x%02x' % x for x in b)
        out.append('   {"%s", "%s", 0x%02x, %d, %d, {%s}, {%s}},' %
                   (mn, stem, ignore, len(bc), len(bd), hexlist(bc),
                    hexlist(bd)))
    out += ['};', '']

    # Whole G17G programs with branches: after re-encoding, branch number
    # `branch` (an instruction index) must land on instruction `target`, the
    # counterpart of the one Apple's G18G branch lands on.
    max_checks = max((len(p[2]) for p in programs), default=1)
    for n, (stem, code, checks) in enumerate(programs):
        out.append('static const uint8_t agx3d_branch_program_%d[] = {' % n)
        for k in range(0, len(code), 16):
            out.append('   ' + ','.join('0x%02x' % x for x in code[k:k + 16]) + ',')
        out.append('};')
    out += ['',
            'struct agx3d_transcode_branch_program {',
            '   const char *text;',
            '   const uint8_t *g17g;',
            '   unsigned g17g_len, count;',
            '   struct { unsigned branch, target; } checks[%d];' % max_checks,
            '};',
            '',
            'static const struct agx3d_transcode_branch_program '
            'agx3d_transcode_branch_programs[] = {']
    for n, (stem, code, checks) in enumerate(programs):
        out.append('   {"%s", agx3d_branch_program_%d, %d, %d, {%s}},' %
                   (stem, n, len(code), len(checks),
                    ', '.join('{%d, %d}' % c for c in checks)))
    out += ['};', '']
    open(sys.argv[1], 'w').write('\n'.join(out))
    print('branch programs', len(programs), 'branches',
          sum(len(p[2]) for p in programs))
    print('vectors', len(vectors), 'slot-normalised',
          sum(1 for v in vectors.values() if v[2] == 12), 'mnemonics',
          len({v[0] for v in vectors.values()}))


if __name__ == '__main__':
    main()
