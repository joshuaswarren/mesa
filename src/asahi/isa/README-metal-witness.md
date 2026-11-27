# Getting Apple's own G17P encoding for a shader shape

Apple's Metal toolchain targets the G17P directly, so an ISA question can be
answered with Apple's byte-exact machine code instead of inference: write the
shape in MSL, compile it for `applegpu_g17p`, and compare against ours.

Use this for any AGX3B change to instruction selection, operand packing,
staging, register allocation, scheduling or the program ABI, and before adding
a G16G/G17P chip gate.  G16G and G17P share the AGX3B compiler path; keep a
chip-specific branch only when Apple's output, a hardware result or a device
limit shows a real difference.  A host test only shows what our compiler
emitted; it is not an architecture oracle.

1. Reduce the shader to the smallest MSL that keeps the operation and its live
   operands.
2. Compile it for `applegpu_g17p` (and a G16G target when one is available).
3. Decode the relevant instructions to field vectors (see below).
4. Compare instruction form, operands, register domains, staging order and
   waits with Mesa's output for the equivalent NIR -- not just size or a hash.

## Toolchain

    MT=$(dirname $(xcrun -f metal))     # Metal.xctoolchain/usr/bin
    "$MT/metal-nt" -archs               # applegpu_g17p is in the list

`metal-nt` needs a pipeline script, and the script's `.mtlp-json` suffix is
required (any other name fails with "malformed flatbuffer magic").  `pipelines`
is a table with the keys `compute_pipelines`, `render_pipelines`,
`tile_render_pipelines` and `mesh_render_pipelines`; each entry names functions
with bare strings.  Verified with MetalToolchain 27.1.5194.15:

    printf '{"pipelines":{"compute_pipelines":[{"compute_function":"K"}]}}' \
        > p.mtlp-json
    xcrun metal -c w.metal -o w.air
    xcrun metallib w.air -o w.metallib
    "$MT/metal-nt" -arch applegpu_g17p -platform_version macos 27.0 27.0 \
        -N p.mtlp-json w.metallib -o out.bin

`-platform_version macos 27.0 27.0` is needed: older targets emit an AIR
version `metal-nt` refuses, and `-std=metalN.M` does not change it.
`xcrun metal -arch applegpu_g17p -c` is not a shortcut; it fails in
`AIRNTEmitObject`.

The output is a Mach-O GPU executable.  Compute code is in `__TEXT,__compute`
(`otool -s __TEXT __compute out.bin`).  The first 0x40 bytes before the entry
point are the constant program (`0e 00 00 00`, then `06 00` filler), the same
container layout the driver builds.

A render pipeline's fragment function is compiled without attachment
information, so a `fragment float4` whose result only reaches a colour output
is optimised away to an 80-byte stub.  Declare it `fragment void` and store to
a `device` buffer instead.  Implicit-LOD sampling is illegal in a compute
kernel (`metal-nt` aborts with "Failed to encode"), so bias witnesses must come
from the render path.

## Decoding

Decode an instruction by finding the AGX3b/AGX3c form whose field vector
re-composes to the same bytes through `agx3_compose` (in-tree tables under
`src/asahi/compiler/agx3_compose/`): enumerate the forms of that length, derive
the fields the bytes imply, re-compose and compare.  Only exact round trips are
facts about the encoding; field names from a disassembler are not.  Start from
a real instruction as the baseline -- an all-zero field vector does not satisfy
a form's predicates.

Field positions differ per instruction form.  For example, the memory forms:

    base kind :  lddev = bit 101      stdev = bit 99
    index kind:  stdev = bit 100
    access kind: lddev = f75 @97:99   stdev = f73 @95:97
    16-bit off : lddev = f76 @77:92   stdev = f74 @75:90

`lddev.f75` and `stdev.f75` are different physical fields, so a rule found on
one form does not transfer to another.  Map each form separately by varying
one field at a time from a known-good baseline.  Common to both memory forms:
wait mask = bits 12..17 (wait slots 1..6), lddev signal slot = bits 70..72,
lddev uniform base = byte 4 = base / 4.
## g14g (AGX2) fixtures and the encoded-output comparison

The same procedure applies to AGX2 chips (`applegpu_g14g`, M2 Max class)
with one decoder difference: the instruction stream is AGX2, so
`agx2_disassemble_instr` (generated tables in `src/asahi/isa/`) is the
authority on instruction boundaries and forms, not `agx3_compose` round
trips.  Only exact round trips are still the facts; decoded text names what
the decoder establishes and nothing more.

Fixture layout follows the existing native corpus under
`src/asahi/compiler/test/native/`:

    native/g14g/
      scoreprobe-reference.metal             MSL source (smallest shape that
                                             keeps the operation and its live
                                             operands)
      scoreprobe-reference.mtlp-json         metal-nt pipeline script
      probe.scoreprobe-reference.scoreprobe.g14g.bin
                                             the _agc.main bytes exactly as
                                             Apple emitted them; the full
                                             __TEXT,__text section may be
                                             dropped in instead, the tests
                                             read from the entry offset
      manifest.json                          provenance (archive/inner-object
                                             SHA-256), symbol offsets, the
                                             section layout and the expected
                                             decode, in the g18g schema
      README.md                              file guide and regeneration steps

Two gtests in `agx_tests` consume the fixture:

  * `test-agx2-g14g-witness.cpp` (AGX2G14GWitness) decodes `_agc.main` with
    `agx2_disassemble_instr` and asserts instruction length and decoded text
    against the manifest's expected listing.
  * `test-agx2-witness-compare.cpp` (Agx2WitnessCompare) builds the
    equivalent NIR with `nir_builder`, compiles it through
    `agx_compile_shader_nir` with `key.dev.chip = AGX_CHIP_G14G`,
    `key.dev.encoding = AGX_ISA_ENCODING_AGX2` and `key.no_preamble = true`,
    then compares the encoded bytes against the witness instruction by
    instruction through `agx2_disassemble_instr`.

Per instruction pair the comparison reports one of:

  * byte-identical
  * same decoded text, different encoding
  * same form, operands differ
  * form differs

and per program it reports the ABI fields it can check (binary/main/preamble
sizes and offsets, constant-program presence per the manifest, instruction
count, termination form).  Anything the decoder cannot settle -- for example
staging order once the streams' instruction counts diverge, or the meaning of
bytes the decoder leaves unnamed -- is printed as unestablished, never
inferred.  Buffer bases read through `load_preamble` encode as `du` slots in
the driver uniform area; established on our side: our `du` index is half the
preamble base (preamble base 0 loads as `du0`, base 4 as `du2`).  On Apple's
side the witness shows the `load` on `du0` and the `store` on `du2`; whether
Apple derives the index the same way is unestablished -- its base values
live in the constant program, which the checked-in fixture does not carry.

To run just this comparison:

    meson test -C build agx_tests --test-args='--gtest_filter=Agx2WitnessCompare.*'
