
import json, os, sys, time
import mlx.core as mx
n = int(sys.argv[1])
H, KVH, D = 14, 2, 64
mx.random.seed(0)
q = mx.random.normal((1, H, n, D)); k = mx.random.normal((1, KVH, n, D)); v = mx.random.normal((1, KVH, n, D))
mx.eval(q, k, v)
print("inputs ready", flush=True)
t0 = time.monotonic()
try:
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5), mask=None)
    mx.eval(out)
    print(json.dumps({"status": "ok", "wall_s": time.monotonic() - t0, "n": n, "sum": float(out.sum())}), flush=True)
except Exception as e:
    print(json.dumps({"status": "error", "wall_s": time.monotonic() - t0, "n": n, "error": str(e)[:300]}), flush=True)
    sys.exit(2)
