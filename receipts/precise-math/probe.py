#!/usr/bin/env python3
"""Precise-math reproducer driver. Builds inputs, runs probe.spv through
mathrepro under the GPU lock, compares against host references, prints JSON.
usage: probe.py <icd.json> <tag> [--env K=V ...]
"""
import json, math, os, struct, subprocess, sys
import numpy as np

icd, tag = sys.argv[1], sys.argv[2]
extra_env = dict(kv.split("=", 1) for kv in sys.argv[3:])
here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260908)

# --- inputs ---------------------------------------------------------------
div_pairs = [(80.0, 25.0), (10.0, 25.0), (6.55859375, 0.030016447), (1.0, 3.0),
             (2.0, 3.0), (1.0, 10.0), (7.0, 9.0), (1e-3, 3.0)]
for _ in range(1000):
    a = float(rng.choice([-1, 1]) * 10.0 ** rng.uniform(-10, 10))
    b = float(rng.choice([-1, 1]) * 10.0 ** rng.uniform(-10, 10))
    div_pairs.append((a, b))
# fma inputs: random, some with cancellation
fma_rows = [(float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)),
             float(rng.uniform(-16, 16))) for _ in range(1000)]
log_xs = np.exp(np.linspace(np.log(0.5), np.log(64.0), 4096)).astype(np.float32)
log_xs[0] = 0.5
log_exact = [0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 10.0, 100.0 / 100.0]
sin_xs = [1e5, 5e5, 1e6, 5e6, 1e8, 3.0, 1e3, 1e4, 2.5e4, 6e4, 1.2e5, 2e6,
          3e7, 1e9, 1e10, 1e20, 3.4e38, -1e5, -1e6, -1e8, 0.5, 1.5, 2.9,
          3.1415925, 3.1415927, 6.2831855, 1e-3, -1e-3, 0.0, -0.0]
sin_xs += [float(v) for v in rng.uniform(-8, 8, 512)]
sin_xs += [float(10.0 ** v) for v in rng.uniform(2, 8, 512)]
sin_xs += [float(-(10.0 ** v)) for v in rng.uniform(2, 8, 256)]

off_div = len(fma_rows)
off_sin = off_div + len(div_pairs)
all_log = list(log_xs) + log_exact
n = max(off_sin + len(sin_xs), len(all_log))
n = (n + 63) // 64 * 64
inp = np.zeros((n, 4), dtype=np.float32)
for i, (a, b, c) in enumerate(fma_rows):
    inp[i, 0], inp[i, 1], inp[i, 2] = a, b, c
for i, (a, b) in enumerate(div_pairs):
    inp[off_div + i, 0], inp[off_div + i, 1] = a, b
for i, v in enumerate(sin_xs):
    inp[off_sin + i, 0], inp[off_sin + i, 1] = v, 1.0
for i, d in enumerate(all_log):
    inp[i, 3] = d

inp.tofile("/tmp/pm/in.bin")
env = dict(os.environ, VK_ICD_FILENAMES=icd, **extra_env)
cmd = ["flock", "/tmp/m1-gpu.lock", os.path.join(here, "mathrepro"),
       os.path.join(here, "probe.spv"), "/tmp/pm/in.bin", "/tmp/pm/out.bin",
       str(n * 8 * 4), str(n // 64)]
r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
if r.returncode != 0:
    print(r.stdout, r.stderr)
    sys.exit(r.returncode)
device_line = r.stderr.strip().splitlines()[-1]
out = np.fromfile("/tmp/pm/out.bin", dtype=np.float32).reshape(n, 8)

def ulps(got, want):
    g = np.asarray(got, np.float32).view(np.int32).astype(np.int64)
    w = np.asarray(want, np.float32).view(np.int32).astype(np.int64)
    # map to monotone integer line
    g = np.where(g < 0, -(g & 0x7fffffff), g)
    w = np.where(w < 0, -(w & 0x7fffffff), w)
    return np.abs(g - w)

def hist(u):
    u = np.asarray(u)
    keys = sorted(set(int(v) for v in u))
    return {str(k): int((u == k).sum()) for k in keys}

res = {"tag": tag, "device": device_line, "env": extra_env}

# --- division ---------------------------------------------------------------
A = inp[off_div: off_div + len(div_pairs), 0]
B = inp[off_div: off_div + len(div_pairs), 1]
want = (A / B).astype(np.float32)
got = out[off_div: off_div + len(div_pairs), 0]
u = ulps(got, want)
res["div"] = {
    "n": len(div_pairs), "mismatches": int((u != 0).sum()), "ulp_hist": hist(u),
    "cases": [{"a": float(a), "b": float(b), "got": repr(float(g)),
               "want": repr(float(w)), "ulp": int(x)}
              for (a, b), g, w, x in list(zip(div_pairs, got, want, u))[:8]],
}
want_r = (np.float32(1.0) / B).astype(np.float32)
got_r = out[off_div: off_div + len(div_pairs), 6]
u = ulps(got_r, want_r)
res["rcp"] = {"n": len(div_pairs), "mismatches": int((u != 0).sum()), "ulp_hist": hist(u)}

# --- fma ------------------------------------------------------------------
fa = inp[:len(fma_rows), 0]; fb = inp[:len(fma_rows), 1]; fc = inp[:len(fma_rows), 2]
if hasattr(math, "fma"):
    want = np.array([math.fma(float(x), float(y), float(z)) for x, y, z in zip(fa, fb, fc)], np.float64).astype(np.float32)
else:
    from fractions import Fraction
    want = np.array([float(Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z))) for x, y, z in zip(fa, fb, fc)], np.float64).astype(np.float32)
got = out[:len(fma_rows), 1]
u = ulps(got, want)
res["fma"] = {"n": len(fma_rows), "mismatches": int((u != 0).sum()), "ulp_hist": hist(u)}
got = out[:len(fma_rows), 7]
u = ulps(got, want)
res["mul_add_contracted"] = {"n": len(fma_rows), "mismatches_vs_fma": int((u != 0).sum()), "ulp_hist": hist(u)}

# --- log ------------------------------------------------------------------
D = np.array(all_log, np.float32)
want = np.log(D.astype(np.float64)).astype(np.float32)
got = out[:len(all_log), 2]
u = ulps(got, want)
res["log"] = {
    "n": len(all_log), "mismatches": int((u != 0).sum()), "ulp_hist": hist(u),
    "exact_cases": [{"x": float(x), "got": repr(float(g)), "want": repr(float(w)), "ulp": int(z)}
                    for x, g, w, z in zip(D[len(log_xs):], got[len(log_xs):], want[len(log_xs):], u[len(log_xs):])],
}
want2 = np.log2(D.astype(np.float64)).astype(np.float32)
got2 = out[:len(all_log), 3]
u2 = ulps(got2, want2)
res["log2"] = {
    "n": len(all_log), "mismatches": int((u2 != 0).sum()), "ulp_hist": hist(u2),
    "exact_cases": [{"x": float(x), "got": repr(float(g)), "want": repr(float(w)), "ulp": int(z)}
                    for x, g, w, z in zip(D[len(log_xs):], got2[len(log_xs):], want2[len(log_xs):], u2[len(log_xs):])],
}

# --- sin/cos --------------------------------------------------------------
X = inp[off_sin: off_sin + len(sin_xs), 0]
ws = np.sin(X.astype(np.float64)); wc = np.cos(X.astype(np.float64))
gs = out[off_sin: off_sin + len(sin_xs), 4]; gc = out[off_sin: off_sin + len(sin_xs), 5]
us = ulps(gs, ws.astype(np.float32)); uc = ulps(gc, wc.astype(np.float32))
pinned = 30
res["sin"] = {
    "n": len(sin_xs),
    "pinned": [{"x": float(x), "got": repr(float(g)), "want": repr(float(w)),
                "abs_err": float(abs(float(g) - w)), "ulp": int(z)}
               for x, g, w, z in zip(X[:pinned], gs[:pinned], ws[:pinned], us[:pinned])],
    "ulp_hist_small_|x|<8": hist(us[pinned:pinned + 512]),
    "ulp_hist_large_1e2..1e8": hist(np.minimum(us[pinned + 512:], 1000)),
    "max_ulp_large": int(us[pinned + 512:].max()),
}
res["cos"] = {
    "pinned": [{"x": float(x), "got": repr(float(g)), "want": repr(float(w)),
                "abs_err": float(abs(float(g) - w)), "ulp": int(z)}
               for x, g, w, z in zip(X[:pinned], gc[:pinned], wc[:pinned], uc[:pinned])],
    "ulp_hist_small_|x|<8": hist(uc[pinned:pinned + 512]),
    "ulp_hist_large_1e2..1e8": hist(np.minimum(uc[pinned + 512:], 1000)),
    "max_ulp_large": int(uc[pinned + 512:].max()),
    "outliers": [{"x": float(x), "got": repr(float(g)), "want": repr(float(w)), "ulp": int(z)}
                 for x, g, w, z in zip(X[pinned:], gc[pinned:], wc[pinned:], uc[pinned:]) if z > 1][:10],
}
res["sin"]["outliers"] = [{"x": float(x), "got": repr(float(g)), "want": repr(float(w)), "ulp": int(z)}
                 for x, g, w, z in zip(X[pinned:], gs[pinned:], ws[pinned:], us[pinned:]) if z > 1][:10]
print(json.dumps(res, indent=1))
