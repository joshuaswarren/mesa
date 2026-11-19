# M5 instruction reference probes

These Metal sources reproduce native G17G instruction witnesses used by the
AGX3c compiler tests. They are reference inputs only; the Mesa hardware tests
compile their own shaders and do not substitute these native binaries.

The references were compiled with Apple Metal toolchain 27.1 (27A5194o):

```sh
probe=m5-half-select-reference
xcrun metal -std=metal3.0 -mmacosx-version-min=26.0 \
   -c "$probe.metal" -o "$probe.air"
xcrun metallib "$probe.air" -o "$probe.metallib"
xcrun metal-nt -arch applegpu_g17g -platform_version macos 26.0 26.0 \
   -N "$probe.mtlp-json" "$probe.metallib" -save-temps -o "$probe.gpubin"
```

Use the `.gpubin` output suffix. Compare the embedded Mach-O shader text
against the byte witnesses in `test-agx3-*.c`; instruction scheduling and
register choices may change with a different toolchain.

- Array sampler probes cover uniform and dynamically indexed descriptor
  resources, explicit LOD zero, and native array-layer staging.
- Float probes keep operands live to expose cache/discard and dependency
  waits for ADD, MUL, and FMA.
- IMAD probes cover register and immediate multiplier/addend forms.
- Half-select probes cover unsigned 32-bit comparisons producing half
  results, including nested bounds conditions and mixed selected operands.
- Texture-read probes cover uniform and register heap indices with explicit
  zero LOD and integer array-layer staging.
- Buffer-store probes retain the same value across two writes through a
  dynamically loaded pointer, exposing the store's extended dependency form.

Native compilation of these probes is encoding evidence only, not evidence
of how Mesa's own programs behave on M5 hardware.
