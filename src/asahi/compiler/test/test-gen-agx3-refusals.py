#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""gen_agx3_tables.py must refuse a description with rows the encoder no
longer tests when it applies them: agx3_compose.c indexes its word and field
arrays with what the table holds.

usage: test-gen-agx3-refusals.py <gen_agx3_tables.py>
"""
import os
import subprocess
import sys
import tempfile

HEAD = '# AGX3 instruction encoding table, variant 0: test.\n'

# A description, and what the refusal must say; None: it is accepted.
CASES = [
    ('group 0\n ; ; 5[0+1]>0.1\n'
     'form mov 0 4\n ; 0&0001=0001 ; 6[0+1]>0.2 ; g0\n', None),
    # A fixed row and a deposit in word 2 of a form of two words.
    ('form mov 0 4\n ; 2&0001=0001 ;\n', 'deposits into word 2'),
    ('form mov 0 4\n ; 0&0001=0001 ; 6[0+1]>2.0\n', 'deposits into word 2'),
    # The same, in an alternative of a group under the form.
    ('group 0\n ; 3&0001=0001 ;\n'
     'form mov 0 4\n ; 0&0001=0001 ; ; g0\n', 'deposits into word 3'),
    ('form mov 0 4\n ; 0&0001=0001 ; 160[0+1]>0.1\n', 'no field 160'),
    ('form mov 0 4\n 160=0 ; 0&0001=0001 ;\n', 'no field 160'),
    # A deposit that leaves its 16-bit word.
    ('form mov 0 4\n ; ; 6[0+2]>0.15\n', 'a deposit of 2 bits'),
    # One field deposited directly by a node and by a group under it.
    ('group 0\n ; ; 5[0+1]>0.1\n'
     'form mov 0 4\n ; ; 5[1+1]>0.2 ; g0\n', 'deposited directly in two'),
]


def main():
    gen = sys.argv[1]
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (text, refusal) in enumerate(CASES):
            path = os.path.join(tmp, 'case%d.txt' % i)
            with open(path, 'w') as f:
                f.write(HEAD + text)
            run = subprocess.run([sys.executable, '-B', gen, path],
                                 capture_output=True, text=True)
            if refusal is None:
                ok = run.returncode == 0
            else:
                ok = run.returncode != 0 and refusal in run.stderr
            failed += not ok
            print('%s case %d: rc=%d, expected %s' %
                  ('ok  ' if ok else 'FAIL', i, run.returncode,
                   repr(refusal) if refusal else 'acceptance'))
            if not ok:
                sys.stdout.write(run.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
