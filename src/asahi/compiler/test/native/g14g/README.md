# G14G native instruction witness

Apple's Metal toolchain output for the `scoreprobe` multiply kernel on
`applegpu_g14g` (M2 Max class), the AGX2 encoding. Reference data only:
nothing here ran on a GPU, and Mesa does not produce any of it.

## Contents

| file | what it is |
| --- | --- |
| `probe.scoreprobe-reference.scoreprobe.g14g.bin` | `_agc.main` of the compiled kernel (30 bytes). The archive's full `__TEXT,__text` is 94 bytes: a 64-byte `_agc.main.constant_program` ahead of `_agc.main` at 0x40. Only Apple-emitted bytes are checked in; the constant program is arch-specific and is not reconstructed here. |
| `scoreprobe-reference.metal` | the MSL source the witness was compiled from |
| `scoreprobe-reference.mtlp-json` | the pipeline script `metal-nt` consumes |
| `manifest.json` | provenance: archive and inner-object SHA-256, symbol offsets, and the expected `_agc.main` decode |

`test-agx2-g14g-witness.cpp` decodes `_agc.main` with `agx2_disassemble_instr`
and asserts the instruction form, operands, register domains, wait slot and
staging order. It reads from offset 0 of the checked-in fixture, or from 0x40
if the full 94-byte section is dropped in:

```
0040: 72051004         mov      r1, thread_in_grid_x
0044: 0501200e00c01200 load     r0, du0, r1, i32, x, a
004c: 3800             wait     a
004e: 1a81c0020001     fmul     r0, ^r0, 2.0
0054: 4501240e00c01200 store    r0, du2, r1, i32, x, a
005c: 8800             stop
```

## Regenerating

Needs Xcode's Metal toolchain with `metal-nt` listing `applegpu_g14g`, on the
same method `src/asahi/isa/README-metal-witness.md` documents:

```sh
xcrun metal -c scoreprobe-reference.metal -o scoreprobe.air
xcrun metallib scoreprobe.air -o scoreprobe.metallib
"$MT/metal-nt" -arch applegpu_g14g -platform_version macos 27.0 27.0 \
    -N scoreprobe-reference.mtlp-json scoreprobe.metallib -o out.bin
```

Take the `__TEXT,__text` bytes of the scoreprobe GPU object (`otool -s __TEXT
__text` or the archive's slice dump). The manifest records the archive and
inner-object SHA-256 the checked-in fixture was taken from.
