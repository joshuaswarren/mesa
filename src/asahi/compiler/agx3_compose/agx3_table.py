# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Read an agx3{b,c,d}_tables.txt description, and encode or decode with it.

load() parses a description into the row arrays the tables were first
checked in as (the legacy layout), as Python tuples:

  fixeds  (word, mask, value)
  preds   (field, kind, shift, width, a, b)
          kind 0 eq a, 1 in setpool[a:a+b], 2 range [a,b], 3 bitfield eq a
  cmds    (field, word, dst_shift, width, src_shift, lut_off, lut_n)
          lut_off -1 deposits the field directly
  groups  (alt_off, alt_n) into altpool, a list of node ids
  nodes   (fixed_off, fixed_n, pred_off, pred_n, cmd_off, cmd_n, grp_off, grp_n)
  forms   (mnemonic, ep, size, node)

A description names each distinct choice group once; load() unfolds it into
one node tree per form again. legacy_c() prints the rows as the C file they
were checked in as, which is how gen_agx3_tables.py --verify proves that a
description still holds exactly those rows.

Composer agrees with agx3_compose.c, including every fail-closed rule, on
every description gen_agx3_tables.py accepts.

Decoder is the inverse: for an instruction's bytes it walks the candidate
forms' choice groups (alternatives in table order, pruned by their constant
bits), solves each path's field constraints, and accepts the first field
vector that Composer turns back into exactly those bytes. A decoded vector is
therefore always one the composer reproduces; it is not independent evidence
that the table's field-to-bit mapping is right.
"""

import os

NUM_FIELDS = 160

ARRAYS = ('fixeds', 'preds', 'cmds', 'groups', 'nodes', 'forms', 'altpool',
          'lutpool', 'setpool')


def wmask(width):
    return (1 << width) - 1


def _pred(tok):
    lhs, rhs = tok.split('=')
    if lhs.endswith(']'):
        field, bits = lhs[:-1].split('[')
        shift, width = bits.split('+')
        return (int(field), 3, int(shift), int(width), int(rhs), 0)
    if rhs.startswith('{'):
        return (int(lhs), 1, 0, 0, tuple(int(v) for v in rhs[1:-1].split(',')), 0)
    if '..' in rhs:
        lo, hi = rhs.split('..')
        return (int(lhs), 2, 0, 0, int(lo), int(hi))
    return (int(lhs), 0, 0, 0, int(rhs), 0)


def _fixed(tok):
    word, bits = tok.split('&')
    mask, value = bits.split('=')
    return (int(word), int(mask, 16), int(value, 16))


def _cmd(tok):
    tok, _, lut = tok.partition(':')
    src, dst = tok.split('>')
    field, bits = src[:-1].split('[')
    src_shift, width = bits.split('+')
    word, dst_shift = dst.split('.')
    return (int(field), int(word), int(dst_shift), int(width), int(src_shift),
            tuple(int(v) for v in lut.split(',')) if lut else None)


def _node(line):
    """A node line: preds ; fixeds ; cmds ; groups."""
    part = [p.split() for p in line.split(';')] + [[]] * 3
    return (tuple(_pred(t) for t in part[0]), tuple(_fixed(t) for t in part[1]),
            tuple(_cmd(t) for t in part[2]), tuple(int(t[1:]) for t in part[3]))


def parse(path):
    """(name, groups, forms): groups[id] is a list of alternative nodes and
    forms a list of (mnemonic, ep, size, node). A node is (preds, fixeds,
    cmds, group ids), with set and LUT contents in place of pool offsets."""
    groups, forms = [], []
    shape = None
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('group'):
            head, _, shape = line.partition(':')
            if int(head.split()[1]) != len(groups):
                raise ValueError('groups are numbered in file order: ' + line)
            if shape:
                preds, fixeds = shape.split(';')
                shape = ([_pred(t + '=0')[:4] for t in preds.split()],
                         [_fixed(t + '=0')[:2] for t in fixeds.split()])
            groups.append([])
        elif line.startswith('form'):
            _, mnemonic, ep, size = line.split()
            forms.append([mnemonic, int(ep), int(size)])
        elif len(forms) and len(forms[-1]) == 3:
            forms[-1].append(_node(line))
        elif shape:
            # A row of a table-form group: the values of the header's shape.
            a, _, values = line.partition(';')
            a, values = a.split(), values.split()
            if (len(a), len(values)) != (len(shape[0]), len(shape[1])):
                raise ValueError('not a row of its group header: ' + line)
            groups[-1].append((
                tuple(p + (int(v), 0) for p, v in zip(shape[0], a)),
                tuple(f + (int(v, 16),) for f, v in zip(shape[1], values)),
                (), ()))
        else:
            groups[-1].append(_node(line))
    name = os.path.basename(path).split('_')[0]
    return name, groups, [tuple(f) for f in forms]


def load(path):
    return unfold(*parse(path))


def unfold(name, groups, forms):
    """One node tree per form, in the legacy row order."""
    T = {a: [] for a in ARRAYS}
    T['name'] = name + '_table'
    pooled = {}

    def pool(kind, values):
        """Offset of values in the LUT or set pool, appended on first use."""
        if (kind, values) not in pooled:
            pooled[kind, values] = len(T[kind])
            T[kind] += values
        return pooled[kind, values]

    def emit(node):
        preds, fixeds, cmds, grefs = node
        own = []
        for g in grefs:
            alts = [emit(a) for a in groups[g]]
            own.append((len(T['altpool']), len(alts)))
            T['altpool'] += alts
        T['nodes'].append((len(T['fixeds']), len(fixeds), len(T['preds']),
                           len(preds), len(T['cmds']), len(cmds),
                           len(T['groups']), len(own)))
        T['fixeds'] += fixeds
        T['preds'] += [p if p[1] != 1 else
                       p[:4] + (pool('setpool', p[4]), len(p[4])) for p in preds]
        T['cmds'] += [c[:5] + ((-1, 0) if c[5] is None else
                               (pool('lutpool', c[5]), len(c[5]))) for c in cmds]
        T['groups'] += own
        return len(T['nodes']) - 1

    for mnemonic, ep, size, node in forms:
        T['forms'].append((mnemonic, ep, size, emit(node)))
    T['num_forms'] = len(forms)
    return T


def legacy_c(T):
    """The rows as the checked-in C files had them, from the #include line on."""
    def rows(ctype, name, fmt):
        return ('static const %s %s[] = {\n' % (ctype, name) +
                ',\n'.join(fmt % r for r in T[name]) + '\n};\n')

    def pool(ctype, name, fmt):
        return ('static const %s %s[] = {\n' % (ctype, name) +
                ','.join(fmt % v for v in T[name]) + '\n};\n')

    return ('#include "agx3_compose.h"\n' +
            rows('agx3_fixed', 'fixeds', '{%d,0x%x,0x%x}') +
            rows('agx3_pred', 'preds', '{%d,%d,%d,%d,0x%xull,0x%xull}') +
            rows('agx3_cmd', 'cmds', '{%d,%d,%d,%d,%d,%d,%d}') +
            rows('agx3_group', 'groups', '{%d,%d}') +
            rows('agx3_node', 'nodes', '{%d,%d,%d,%d,%d,%d,%d,%d}') +
            rows('agx3_form', 'forms', '{"%s",%d,%d,%d}') +
            pool('uint32_t', 'altpool', '%d') +
            pool('uint16_t', 'lutpool', '%d') +
            pool('uint64_t', 'setpool', '0x%xull') +
            'const agx3_table %s = {\n  forms, %d, nodes, groups, altpool, '
            'fixeds, preds, cmds, lutpool, setpool\n};\n' %
            (T['name'], T['num_forms']))


class Composer:
    """agx3_compose.c in Python: same contract, same failures."""

    def __init__(self, T):
        self.T = T

    def compose(self, form, fields):
        T = self.T
        _, _, size, root = T['forms'][form]
        self.nwords = size // 2
        self.words = [0] * self.nwords
        self.written = [0] * self.nwords
        self.consumed = set()
        self.is_lut = set()
        self.is_pred = set()
        self.direct = []
        self.fields = fields

        _, _, po, pn, _, _, _, _ = T['nodes'][root]
        if not self.preds_test(po, pn):
            return None
        self.preds_mark(po, pn)
        if not self.apply_node(root):
            return None

        src = [0] * NUM_FIELDS
        for f, w, s in self.direct:
            src[f] |= wmask(w) << s
        for i in range(NUM_FIELDS):
            if fields[i] and i not in self.consumed:
                return None
            if (i in self.consumed and i not in self.is_lut and
                    i not in self.is_pred and (fields[i] & ~src[i])):
                return None

        out = bytearray()
        for w in self.words:
            out += bytes((w & 0xff, w >> 8))
        return bytes(out)

    def pred_ok(self, p):
        f, k, sh, w, a, b = p
        v = self.fields[f]
        if k == 0:
            return v == a
        if k == 1:
            return v < 0x10000 and v in self.T['setpool'][a:a + b]
        if k == 2:
            return a <= v <= b
        if k == 3:
            return ((v >> sh) & wmask(w)) == a
        return False

    def preds_test(self, off, n):
        return all(self.pred_ok(p) for p in self.T['preds'][off:off + n])

    def preds_mark(self, off, n):
        for p in self.T['preds'][off:off + n]:
            self.consumed.add(p[0])
            self.is_pred.add(p[0])

    def deposit(self, word, mask, value):
        if word >= self.nwords:
            return False
        if self.written[word] & mask & (self.words[word] ^ value):
            return False
        self.words[word] = (self.words[word] & ~mask) | (value & mask)
        self.written[word] |= mask
        return True

    def snapshot(self):
        return (list(self.words), list(self.written), set(self.consumed),
                set(self.is_lut), set(self.is_pred), len(self.direct))

    def restore(self, s):
        self.words, self.written, self.consumed, self.is_lut, self.is_pred, nd = s
        del self.direct[nd:]

    def apply_group(self, g):
        T = self.T
        ao, an = T['groups'][g]
        for a in T['altpool'][ao:ao + an]:
            _, _, po, pn, _, _, _, _ = T['nodes'][a]
            if not self.preds_test(po, pn):
                continue
            s = self.snapshot()
            self.preds_mark(po, pn)
            if self.apply_node(a):
                return True
            self.restore(s)
        return False

    def apply_node(self, n):
        T = self.T
        fo, fn, _, _, co, cn, go, gn = T['nodes'][n]
        for w, m, v in T['fixeds'][fo:fo + fn]:
            if not self.deposit(w, m, v):
                return False
        for f, word, dst, width, src, lo, ln in T['cmds'][co:co + cn]:
            v = self.fields[f]
            self.consumed.add(f)
            if lo >= 0:
                self.is_lut.add(f)
                if v >= ln:
                    return False
                v = T['lutpool'][lo + v]
            else:
                self.direct.append((f, width, src))
            bits = (v >> src) & wmask(width)
            if not self.deposit(word, (wmask(width) << dst) & 0xffff,
                                (bits << dst) & 0xffff):
                return False
        for g in range(go, go + gn):
            if not self.apply_group(g):
                return False
        return True


class Decoder:
    def __init__(self, T):
        self.T = T
        self.C = Composer(T)
        self.by_size = {}
        for i, (_, _, size, _) in enumerate(T['forms']):
            self.by_size.setdefault(size, []).append(i)

    @staticmethod
    def word(data, w):
        return data[2 * w] | (data[2 * w + 1] << 8)

    def fixed_ok(self, node, data):
        fo, fn = self.T['nodes'][node][:2]
        for w, m, v in self.T['fixeds'][fo:fo + fn]:
            if 2 * w + 1 >= len(data) or (self.word(data, w) & m) != v:
                return False
        return True

    def candidates(self, data):
        """Forms of this size whose own constant bits match."""
        return [f for f in self.by_size.get(len(data), [])
                if self.fixed_ok(self.T['forms'][f][3], data)]

    def paths(self, node, data):
        """Yield node lists (node plus one alternative per choice group,
        recursively) whose constant bits all match data."""
        T = self.T
        _, _, _, _, _, _, go, gn = T['nodes'][node]
        groups = list(range(go, go + gn))

        def rec(gi):
            if gi == len(groups):
                yield []
                return
            ao, an = T['groups'][groups[gi]]
            for a in T['altpool'][ao:ao + an]:
                if not self.fixed_ok(a, data):
                    continue
                for sub in self.paths(a, data):
                    for rest in rec(gi + 1):
                        yield sub + rest

        for r in rec(0):
            yield [node] + r

    def solve(self, nodes, data):
        """Field values that satisfy the deposits and predicates of nodes."""
        T = self.T
        known = {}   # field -> [mask, value]
        luts = {}    # field -> [(lut, src, width, bits)]
        preds = {}   # field -> [pred]
        for n in nodes:
            _, _, po, pn, co, cn, _, _ = T['nodes'][n]
            for p in T['preds'][po:po + pn]:
                preds.setdefault(p[0], []).append(p)
            for f, word, dst, width, src, lo, ln in T['cmds'][co:co + cn]:
                bits = (self.word(data, word) >> dst) & wmask(width)
                if lo >= 0:
                    luts.setdefault(f, []).append(
                        (tuple(T['lutpool'][lo:lo + ln]), src, width, bits))
                else:
                    k = known.setdefault(f, [0, 0])
                    m = wmask(width) << src
                    if k[0] & m & (k[1] ^ (bits << src)):
                        return None
                    k[0] |= m
                    k[1] = (k[1] & ~m) | (bits << src)

        fields = [0] * NUM_FIELDS
        for f in set(known) | set(luts) | set(preds):
            km, kv = known.get(f, [0, 0])
            ps = preds.get(f, [])
            # Bitfield predicates fix slices of the value.
            for p in ps:
                if p[1] == 3:
                    m = wmask(p[3]) << p[2]
                    if km & m & (kv ^ (p[4] << p[2])):
                        return None
                    km |= m
                    kv = (kv & ~m) | (p[4] << p[2])
            eqs = [p[4] for p in ps if p[1] == 0]
            if eqs:
                cands = [eqs[0]]
            elif f in luts:
                cands = range(min(len(l[0]) for l in luts[f]))
            else:
                cands = [kv]

            ok = None
            for v in cands:
                if (v & km) != kv:
                    continue
                if any(v >= len(l[0]) or
                       ((l[0][v] >> l[1]) & wmask(l[2])) != l[3]
                       for l in luts.get(f, [])):
                    continue
                good = True
                for p in ps:
                    k = p[1]
                    if k == 0 and v != p[4]:
                        good = False
                    elif k == 1 and v not in T['setpool'][p[4]:p[4] + p[5]]:
                        good = False
                    elif k == 2 and not (p[4] <= v <= p[5]):
                        good = False
                    elif k == 3 and ((v >> p[2]) & wmask(p[3])) != p[4]:
                        good = False
                if good:
                    ok = v
                    break
            if ok is None:
                return None
            fields[f] = ok
        return fields

    def decode(self, data, forms=None, limit=20000):
        """(form, fields) that compose back to exactly data, or None."""
        tried = 0
        for f in (forms if forms is not None else self.candidates(data)):
            root = self.T['forms'][f][3]
            for nodes in self.paths(root, data):
                tried += 1
                if tried > limit:
                    return None
                fields = self.solve(nodes, data)
                if fields is None:
                    continue
                if self.C.compose(f, fields) == bytes(data):
                    return f, fields
        return None
