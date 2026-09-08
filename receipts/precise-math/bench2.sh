#!/bin/bash
set -e
cd /tmp/pm
declare -A OPS=([sin]="v = sin(v) + 0.5;" [log]="v = log(abs(v) + 1.5);" [div]="v = v / (w + v);" [mul]="v = v * w + 0.5;")
for name in sin log div mul; do
  cat > bench2_$name.comp <<EOF
#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer In { vec4 v[]; } inp;
layout(std430, binding = 1) writeonly buffer Out { vec4 o[]; } outp;
void main() {
  uint i = gl_GlobalInvocationID.x;
  vec4 v = inp.v[i];
  vec4 w = inp.v[(i + 1u) & 4194303u];
  for (int k = 0; k < 32; ++k) { ${OPS[$name]} }
  outp.o[i] = v;
}
EOF
  glslangValidator -V --target-env vulkan1.3 bench2_$name.comp -o bench2_$name.spv > /dev/null
done
for icd in /usr/share/vulkan/icd.d/asahi_icd.aarch64.json /tmp/pm/icd-precise-math.json; do
  for name in sin log div mul; do
    echo -n "$(basename $icd) chain32 $name: "
    VK_ICD_FILENAMES=$icd flock /tmp/m1-gpu.lock ./mathrepro bench2_$name.spv bench_in.bin /tmp/pm/bench_out.bin 67108864 65536 9 2>/dev/null
  done
done
