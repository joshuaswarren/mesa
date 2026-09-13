#!/usr/bin/env python3
"""Usage: precise_math.py ICD COMPUTE_RUNNER; glslangValidator and numpy are required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

icd, runner = map(os.path.abspath, sys.argv[1:])
here = Path(__file__).resolve().parent
rng = np.random.default_rng(20260913)
n = 262144
inp = rng.integers(0, 2**32, (n, 4), dtype=np.uint32)
edges = np.array([0, 0x80000000, 1, 0x7fffff, 0x800000, 0x800001,
                  0x80800000, 0x3f800000, 0x3f7fffff, 0x3f800001,
                  0x40000000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
                  0xff800000, 0x7fc00000], dtype=np.uint32)
inp[:256, 0] = np.repeat(edges, len(edges))
inp[:256, 1] = np.tile(edges, len(edges))
inp[256:65536, 0] = (inp[256:65536, 0] & 0x807fffff) | (rng.integers(1, 40, 65280, dtype=np.uint32) << 23)
inp[256:65536, 1] = (inp[256:65536, 1] & 0x807fffff) | (rng.integers(80, 160, 65280, dtype=np.uint32) << 23)
inp[:, 2] = 0x3f317218
inp[:, 3] = rng.integers(0x00800000, 0x7f800000, n, dtype=np.uint32)
inp[:7, 3] = [0x3f800000, 0x40000000, 0x40400000, 0x3f7fffff, 0x3f800001, 0x00800000, 0x7f7fffff]

def quotient(a, b):
    a = a.copy()
    b = b.copy()
    a[(a & 0x7fffffff) < 0x800000] &= 0x80000000
    b[(b & 0x7fffffff) < 0x800000] &= 0x80000000
    with np.errstate(all="ignore"):
        exact = a.view(np.float32).astype(np.float64) / b.view(np.float32).astype(np.float64)
        rounded = exact.astype(np.float32).view(np.uint32)
    flushed = np.abs(exact) < 2.0**-126
    rounded[flushed] = (a[flushed] ^ b[flushed]) & 0x80000000
    return rounded

def mismatch(got, want):
    both_nan = ((got & 0x7fffffff) > 0x7f800000) & ((want & 0x7fffffff) > 0x7f800000)
    return np.flatnonzero((got != want) & ~both_nan)

with tempfile.TemporaryDirectory(prefix="mesa-math-") as td:
    td = Path(td)
    inp.tofile(td / "in.bin")
    subprocess.run(["glslangValidator", "-V", str(here / "precise_math.comp"), "-o", str(td / "probe.spv")], check=True, timeout=30)
    env = dict(os.environ, VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd, MESA_SHADER_CACHE_DISABLE="true")
    subprocess.run(["flock", "-w", "30", "/tmp/m1-gpu.lock", runner,
                    str(td / "probe.spv"), str(td / "in.bin"), str(td / "out.bin"),
                    str(n * 6 * 4), str(n // 64)], env=env, check=True, timeout=120)
    out = np.fromfile(td / "out.bin", dtype=np.uint32).reshape(n, 6)

checks = {
    "division": mismatch(out[:, 0], quotient(inp[:, 0], inp[:, 1])),
    "reciprocal": mismatch(out[:, 1], quotient(np.full(n, 0x3f800000, dtype=np.uint32), inp[:, 1])),
    "literal_vs_buffer": mismatch(out[:, 2], out[:, 3]),
    "precise_multiply": mismatch(out[:, 2], (out[:, 4].copy().view(np.float32) * np.float32(np.log(2.0))).view(np.uint32)),
}
x = inp[:, 3].copy().view(np.float32).astype(np.float64)
for name, column, ref in [("log", 5, np.log(x)), ("log2", 4, np.log2(x))]:
    want = ref.astype(np.float32).view(np.int32).astype(np.int64)
    got = out[:, column].copy().view(np.int32).astype(np.int64)
    ulp = np.abs(want - got)
    checks[name + "_over_1ulp"] = np.flatnonzero(ulp > 1)
    print(name, "max_ulp", int(ulp.max()))
result = {"pairs": n, "failures": {k: len(v) for k, v in checks.items()}}
for name, indices in checks.items():
    if len(indices):
        print(name, [{"input": [hex(int(v)) for v in inp[i]], "output": [hex(int(v)) for v in out[i]]} for i in indices[:5]])
print(json.dumps(result, sort_keys=True))
sys.exit(any(len(v) for v in checks.values()))
