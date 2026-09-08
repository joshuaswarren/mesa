#!/bin/bash
# Elementwise 16M-float dispatches (vec4 per thread), 15 reps each, median wall
# time around submit+fence on the GPU lock. Stock vs precise-math ICD.
set -e
cd /tmp/pm
declare -A OPS=([sin]="sin(v)" [cos]="cos(v)" [div]="v / w" [log]="log(v)" [mul]="v * w")
for name in sin cos div log mul; do
  op=${OPS[$name]}
  cat > bench_$name.comp <<EOF
#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer In { vec4 v[]; } inp;
layout(std430, binding = 1) writeonly buffer Out { vec4 o[]; } outp;
void main() {
  uint i = gl_GlobalInvocationID.x;
  vec4 v = inp.v[i];
  vec4 w = inp.v[(i + 1u) & 4194303u];
  outp.o[i] = $op;
}
EOF
  glslangValidator -V --target-env vulkan1.3 bench_$name.comp -o bench_$name.spv > /dev/null
done
python3 - <<'EOF'
import numpy as np
rng = np.random.default_rng(1)
a = rng.uniform(-6.0, 6.0, 1 << 24).astype(np.float32)
a[a == 0] = 1.0
a.tofile('/tmp/pm/bench_in.bin')
EOF
for icd in /usr/share/vulkan/icd.d/asahi_icd.aarch64.json /tmp/pm/icd-precise-math.json; do
  for name in sin cos div log mul; do
    echo -n "$(basename $icd) $name: "
    VK_ICD_FILENAMES=$icd flock /tmp/m1-gpu.lock ./mathrepro bench_$name.spv bench_in.bin /tmp/pm/bench_out.bin 67108864 65536 15 2>/dev/null
  done
done
