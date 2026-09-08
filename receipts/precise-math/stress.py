#!/usr/bin/env python3
"""Bulk random and IEEE edge checks through probe.spv. usage: stress.py <icd> <tag>"""
import json, math, os, subprocess, sys
import numpy as np

icd, tag = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
N = 1 << 20
inf, nan = np.float32(np.inf), np.float32(np.nan)

def rand_floats(n, lo_exp, hi_exp):
    v = (10.0 ** rng.uniform(lo_exp, hi_exp, n)) * rng.choice([-1.0, 1.0], n)
    return v.astype(np.float32)

inp = np.zeros((N, 4), dtype=np.float32)
inp[:, 0] = rand_floats(N, -30, 30)          # a (div), x (sin/cos)
inp[:, 1] = rand_floats(N, -30, 30)          # b (div)
inp[:, 2] = rand_floats(N, -10, 10)          # c (fma)
inp[:, 3] = np.abs(rand_floats(N, -37, 38))  # d (log)
# sin/cos want a mixed range: overwrite column 0 for half the rows
half = N // 2
inp[:half, 0] = rand_floats(half, -3, 9)
inp[half:half + half // 2, 0] = rng.uniform(-10, 10, half // 2).astype(np.float32)

# edge rows at the end
edges = [
    # a, b, c, d
    (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, -0.0), (0.0, 0.0, 0.0, -1.0),
    (np.inf, 1.0, 0.0, np.inf), (1.0, np.inf, 0.0, np.nan), (np.inf, np.inf, 0.0, 1e-40),
    (np.nan, 1.0, 0.0, 1.1754944e-38), (1e38, 1e-38, 0.0, 3.4e38), (1e-38, 1e38, 0.0, 1e-45),
    (3.0, 1e-45, 0.0, 0.7071067), (1e-45, 3.0, 0.0, 1.4142135), (-0.0, 5.0, 0.0, 2.0**-126),
    (5.0, -0.0, 0.0, 1.0000001), (2.0**120, 2.0**-10, 0.0, 0.99999994), (1.0, 3.0, 0.0, 1.0),
]
E = len(edges)
for i, row in enumerate(edges):
    inp[N - E + i] = row
inp.tofile("/tmp/pm/in2.bin")

env = dict(os.environ, VK_ICD_FILENAMES=icd)
cmd = ["flock", "/tmp/m1-gpu.lock", os.path.join(here, "mathrepro"),
       os.path.join(here, "probe.spv"), "/tmp/pm/in2.bin", "/tmp/pm/out2.bin",
       str(N * 8 * 4), str(N // 64)]
r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
if r.returncode != 0:
    print(r.stdout, r.stderr); sys.exit(r.returncode)
out = np.fromfile("/tmp/pm/out2.bin", dtype=np.float32).reshape(N, 8)

def ulps(got, want):
    g = np.asarray(got, np.float32).view(np.int32).astype(np.int64)
    w = np.asarray(want, np.float32).view(np.int32).astype(np.int64)
    g = np.where(g < 0, -(g & 0x7fffffff), g)
    w = np.where(w < 0, -(w & 0x7fffffff), w)
    both_nan = np.isnan(got) & np.isnan(want)
    return np.where(both_nan, 0, np.abs(g - w))

def hist(u, cap=8):
    u = np.minimum(np.asarray(u), cap)
    return {("%d" % k if k < cap else ">=%d" % cap): int((u == k).sum()) for k in sorted(set(int(v) for v in u))}

A, B, C, D = (inp[:N - E, i] for i in range(4))
res = {"tag": tag, "n": N - E}
with np.errstate(all="ignore"):
    want = (A / B).astype(np.float32)
    # FTZ: the driver flushes subnormal results and inputs
    want = np.where(np.abs(want) < np.float32(2.0**-126), np.float32(0.0) * np.sign(want), want).astype(np.float32)
    res["div"] = hist(ulps(out[:N - E, 0], want))
    want = (np.float32(1.0) / B).astype(np.float32)
    want = np.where(np.abs(want) < np.float32(2.0**-126), np.float32(0.0) * np.sign(want), want).astype(np.float32)
    res["rcp"] = hist(ulps(out[:N - E, 6], want))
    want = np.log(D.astype(np.float64)).astype(np.float32)
    res["log"] = hist(ulps(out[:N - E, 2], want))
    want = np.log2(D.astype(np.float64)).astype(np.float32)
    res["log2"] = hist(ulps(out[:N - E, 3], want))
    res["sin"] = hist(ulps(out[:N - E, 4], np.sin(A.astype(np.float64)).astype(np.float32)))
    res["cos"] = hist(ulps(out[:N - E, 5], np.cos(A.astype(np.float64)).astype(np.float32)))
    res["sin_|x|>=2^22"] = hist(ulps(out[:N - E, 4][np.abs(A) >= 2**22], np.sin(A[np.abs(A) >= 2**22].astype(np.float64)).astype(np.float32)))
    res["n_|x|>=2^22"] = int((np.abs(A) >= 2**22).sum())

res["edges"] = []
for i, (a, b, c, d) in enumerate(edges):
    o = out[N - E + i]
    with np.errstate(all="ignore"):
        res["edges"].append({
            "a": repr(float(a)), "b": repr(float(b)), "d": repr(float(d)),
            "a/b": repr(float(o[0])), "host_a/b": repr(float(np.float32(a) / np.float32(b))) if b != 0 or True else "",
            "1/b": repr(float(o[6])),
            "log(d)": repr(float(o[2])), "host_log": repr(float(np.float32(np.log(np.float64(d))))),
            "log2(d)": repr(float(o[3])), "sin(a)": repr(float(o[4])), "cos(a)": repr(float(o[5])),
        })
print(json.dumps(res, indent=1))
