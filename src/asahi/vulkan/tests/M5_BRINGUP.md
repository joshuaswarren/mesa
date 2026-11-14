# M5 Honeykrisp integration

The driver admits the exact single-die AGX3 USC/HAL300 kernel contracts for
T8142/G17G (one cluster), T6050/G17S M5 Pro (two clusters) and T6050/G17C M5 Max
(four clusters) by default. There is no device-admission environment override.
The implemented native acceleration-structure builder, cooperative matrices,
bfloat16 and native mesh/task paths are selected automatically by the chip and
kernel capabilities. Diagnostics and hardware probes remain explicit tools.
These defaults do not establish Vulkan conformance; the experimental device
reports conformance version 0.0.0.0.

The driver uses allocator-owned printf storage because the AGX2 fixed address
overlaps M5's compact driver-owned TA arena. Honeykrisp loads its actual
address from the runtime root table.

## Hardware results, 2026-09-12

On Linux J813 with the M5 application DRM adapter:

- `vulkaninfo --summary` enumerates Apple M5 (G17G C0), device ID 0x8142,
  using Honeykrisp.
- Vulkan instance/device creation, graphics-queue acquisition and idle
  destruction succeed.
- The existing render-only Vulkan fixture creates its graphics pipeline.
- Background clear and reload shaders now compile. The reload path uses
  32-bit coordinates and the AGX3c register COPY/CLR encodings independently
  present in the instruction corpus, including the widening copy-and-clear
  sequence. These instructions have byte-exact compiler tests; this is not a
  hardware execution qualification of the complete reload program.
- The single-sample RGBA8 tile-store shader compiles to the exact 78-byte
  native G17G builtin. Its independently generated text, resource metadata,
  and rejection tests are documented in the compiler's native EOT probe.
- Command recording now reaches vertex launch construction and fails with
  `VK_ERROR_FEATURE_NOT_PRESENT`: the M5 vertex pipeline address is zero.
  Native Ioto resource programs, vertex launch state, and background/EOT
  launch integration remain incomplete. Shader compilation failure is
  propagated instead of dereferencing a null shader.
- No Honeykrisp Vulkan draw has completed. Hardware glxgears, repeated draws
  and combined GPU/DCP presentation remain unqualified.

The separately qualified application triangle used captured runtime programs
and resource state. Substituting a newly generated Mesa fragment shader into
that workload also passed pixel validation. Those results do not establish a
complete Honeykrisp draw.

## Reproduce pipeline and recording checks

Build the existing Vulkan fixture against the Vulkan headers:

```sh
cc -DAGX_TEST_DEVICE_ID=0x8142 -DIMAGE_WIDTH=256 -DIMAGE_HEIGHT=256 \
   g17p_render_only.c -ldl -o m5-vulkan-render-only
```

Run with an isolated Honeykrisp ICD and `AGX_RENDER_RECORD_ONLY=1`. This exercises real device/pipeline/recording code
without submitting GPU work. The fixture returns failure when recording is
unsupported. A successful record-only run would still not prove rendering.

The tested kernel currently admits one restricted application draw per GPU
session and requires complete resource ownership. Do not bypass those guards
or reset the live GPU to turn an error into an apparent rendering success.

Host regression checks: `asahi-agx-chip`, `asahi-agx-m5-va`,
`asahi-agx-ioto-program-core`, `hk_printf_root`, `agx3_copy_core`,
`agx3_g17p_producers`, `agx3_compose_corpus`, `agx3_graphics_compile`, and
`agx_tests` all pass.
