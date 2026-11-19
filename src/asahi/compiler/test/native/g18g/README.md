# G17G/G18G native instruction witnesses

Apple's compiler output for the same Metal inputs on M5 (`applegpu_g17g`)
and M6 (`applegpu_g18g`). These are byte witnesses for the G18G encoding
work. They are reference data only: nothing here ran on a GPU, and Mesa does
not produce any of it.

## Contents

| file | what it is |
| --- | --- |
| `probe.<probe>.<function>.<g17g\|g18g>.bin` | one compute function of a `../g17g/<probe>.metal` probe |
| `oracle.<mode>.<stage>.<g17g\|g18g>.bin` | one stage of a native Metal oracle pipeline (`oracle/`) |
| `manifest.json` | per text: corpus, source, pipeline script, function or mode and stage, arch, symbol offsets, size, SHA-256, and for G18G oracle texts the device captures that match it |
| `oracle/oracle.metal` | the oracle's shaders, a byte-identical copy of the native capture oracle's `oracle.metal` |
| `oracle/<mode>.mtlp-json` | the pipeline script of each of the 13 oracle modes, as `metal-source` extracted it from the oracle's binary archive, with the library path set to `oracle.metallib` |
| `build-reference.py` | compiles both corpora and extracts, checks or replaces these files |

Each `.bin` is the stage object's raw `__TEXT,__text`:
`_agc.main.constant_program` at offset 0 (64 bytes) followed by `_agc.main`
at offset 0x40, as `manifest.json` records under `symbols`. There are 128
shaders (103 probe functions and 25 oracle stages), each for both
architectures, so 256 texts. Several shaders share a text, which leaves
111 distinct G17G/G18G pairs.

The texts were produced with Apple metal 32023.921 (metalfe-32023.921.6),
AIR-NT 32023.921 and Metal toolchain 27.1.266.1 on macOS 27.0 (26A428), as
`manifest.json` records under `toolchain`. A different toolchain may
schedule or allocate differently.

## Device witness

The J873 (Apple M6) compiled the oracle pipelines itself when it ran the
native capture sessions `mini-26A428/sessions-20260925-a` and `-b`. Each
mode's `MTLBinaryArchive` there holds an `applegpu_g18g` slice. All 25
G18G oracle texts are byte-identical to the host's offline `metal-nt` output
in both sessions (`device_witness` in `manifest.json`), so offline
`metal-nt -arch applegpu_g18g` is a valid G18G oracle. The probe corpus has
no device counterpart.

## What the corpus shows

- 0 of 128 G18G texts equal their G17G build. 7 have the same length:
  `oracle.compute-add.compute`, `m5_sample_array_level_zero`,
  `glm_fsel16_regs`, `cf_float_if`, `m5_half_full_add`, `and16_of32_live`
  and `ld_u8_ptr`.
- Over the 111 distinct pairs, `_agc.main` is 12128 bytes on G17G and 11744
  bytes on G18G.
- G17G texts use the AGX3c encoding (`src/asahi/isa/AGX3c.xml`); G18G texts
  use AGX3d (`src/asahi/isa/AGX3d.xml`).

## Regenerating

Compiling needs Xcode's Metal toolchain with `metal-nt` listing both
`applegpu_g17g` and `applegpu_g18g`. Build outside the source tree:

```sh
./build-reference.py --output /tmp/g18g-ref --check \
   --device-session .../runs/mini-26A428/sessions-20260925-a \
   --device-session .../runs/mini-26A428/sessions-20260925-b
```

`--check` fails unless every compiled text equals the committed one.
`--update` replaces the committed texts and `manifest.json` instead, and
refuses a partial `--probe`/`--mode` selection. Give `--device-session` with
`--update` so that `manifest.json` keeps its `device_witness` entries. The
oracle's metallib is rebuilt with the recipe of the oracle's `build.sh`
(`-std=metal3.1 -mmacosx-version-min=14.0`), and its SHA-256 must equal the
library label in the pipeline scripts
(`7d3c8bbc1ca51eecf56f07f55bb872269483718fa1dbf25af78d4645b9228476`).

Without the toolchain, `./build-reference.py --verify` checks the committed
texts against `manifest.json`. Add `--device-session` to also compare them with a capture session's
device-compiled archives.
