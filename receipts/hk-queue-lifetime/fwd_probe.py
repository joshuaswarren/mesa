
import json, os, sys, time
import mlx.core as mx
n = int(sys.argv[1])
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
if os.environ.get("PROBE_DISABLE_COMPILE", "1") == "1":
    mx.disable_compile()
from mlx_lm.utils import load
model, tok = load(model_path)
ids = tok.encode(open(os.path.expanduser("~/benchq/long-prompt-2k.txt")).read())
while len(ids) < n:
    ids = ids + ids
ids = ids[:n]
print("tokens", len(ids), flush=True)
t0 = time.monotonic()
try:
    logits = model(mx.array([ids]))
    last = logits[:, -1, :]
    mx.eval(last)
    print(json.dumps({"status": "ok", "wall_s": time.monotonic() - t0, "n": n, "argmax": int(mx.argmax(last))}), flush=True)
except Exception as e:
    print(json.dumps({"status": "error", "wall_s": time.monotonic() - t0, "n": n, "error": str(e)[:400]}), flush=True)
    sys.exit(2)
