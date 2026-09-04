Standalone reproductions for five Honeykrisp shader miscompiles.

Each case is a Vulkan compute shader plus a check that runs on any
Vulkan 1.1 device. On conformant drivers all cases pass. On Apple M1
under Honeykrisp (Vulkan API 1.4.354, 2026-09-03, three identical
passes per case), cases 2, 3, and 5 reproduce with the signatures the
mlx-omarchy project measured, and cases 1 and 4 do not reproduce -
both outcomes are reported below and in each case header.

Build and run:

    make
    ./repro          # run every case
    ./repro 3        # run case 3 (all four matrix arms)
    ./repro 5        # run the n=64 reduce arm; use 50 for n=33

Select a driver with VK_DRIVER_FILES (or VK_ICD_FILENAMES on older
loaders). These programs are not part of Mesa's meson build and not run
by CI. glslangValidator compiles the shaders; the harness in vk.c and
repro.c is about 500 lines and does one dispatch per arm.

Verification of record for this branch: llvmpipe (Mesa LLVM 15.0.6,
Vulkan 1.3.230), all nine case runs pass (eleven PASS checks), exit 0.

## The five defects

| # | Case | Trigger | Expected | Observed on M1 |
| - | ---- | ------- | -------- | -------------- |
| 1 | 01-word-load-divergent-loop | SSBO word load hoisted into a multi-iteration grid-stride loop | out[w] == in[w] for 200 words | only words with (w & 3) == 0 read back; the rest read 0 |
| 2 | 02-scan-dynamic-shift | data-dependent shift `(w >> ((lane & 3) * 8)) & 0xFF` feeding a shared-memory Hillis-Steele scan | out[255] == 128 | out[255] == 32, a window-64 scan; later steps stop propagating |
| 3 | 03 matrix, arms A-D | bool comparison under a wide op selector over a per-byte path | all 33 outputs == 1 in every arm | suite-level wrong values in the combined shader; exact wrong bytes not recorded |
| 4 | 04-packed-bool-and | the same dynamic shift in packed-bool logical AND | all 33 outputs == 1 | 13 of 33 bytes wrong; 15 of 33 against a scalar true |
| 5 | 05-reduce-load-truthy | the same dynamic shift in reduce_general's load_truthy | reduce == 1 over all-true input | reduce == 0 from n >= 5; mx.any drops a True at index >= 4 |

Every case file carries the same information in its header comment,
with the receipt that recorded the observed values, plus the hardware
verdict from the run below.

## Hardware verdicts (2026-09-03)

Run by the mlx-omarchy project on jwm1 (Apple M1, G13G B1), Mesa
Honeykrisp, Vulkan API 1.4.354, three passes per case, byte-identical
across passes. Raw logs: ~/benchq/logs/mesa-pass{1,2,3}.log on that
machine; AGX_MESA_DEBUG=shaders dumps for every case in
~/benchq/logs/mesa-dump-case{1,2,3,4,5,50}.log.

| Case | Verdict | Observed on hardware |
| ---- | ------- | -------------------- |
| 1 word load | does NOT reproduce | copy matches all 200 words |
| 2 scan | REPRODUCES | out[255] == 32; 252/256 wrong |
| 3 arm A (selector + dynamic shift) | REPRODUCES | 24/33 wrong (got 0, want 1) |
| 3 arm B (no selector + dynamic shift) | REPRODUCES | 24/33 wrong, identical to arm A |
| 3 arm C (selector + constant shift) | PASS | 33/33 correct |
| 3 arm D (no selector + constant shift) | PASS | 33/33 correct |
| 4 packed bool AND | does NOT reproduce | 33/33 correct |
| 5 reduce n=64 | REPRODUCES | reduce[0] == 0; 48/64 truthy reads wrong |
| 5 reduce n=33 | REPRODUCES | 24/33 wrong; truthy correct only at 0-3 and 16-19, matching the recorded map |

Two readings on each miss, and we cannot separate them from this run:

- Case 1: either the reconstruction lost the trigger (the original
  probe's exact shape is not preserved), or the driver fixed the load
  defect. Cases 2/5 still failing on the same build says the extraction
  family is alive; it says nothing about the separate load path.
- Case 4: the same construct DOES fail in arm B's shape (case 3) on
  the same build, so the extraction defect is real here; this kernel
  shape computes correctly. The receipts warned that byte-extraction
  sites are shape- and context-sensitive on this hardware; case 4 is
  the demonstration. The receipt's 13/33 signature came from the full
  logical_or.comp shader, not from this reduced form.

The receipts' observed values in the table above were measured on the
1.4.354/1.4.357 builds of August/September 2026; those receipts do not
record the Mesa commit, and the same-day receipts disagree on the API
version (1.4.354 vs 1.4.357). Today's build reports 1.4.354.

## What each defect shipped as a workaround

mlx-omarchy works around all five in its own shaders. The workarounds
are the evidence that each trigger is real: each one swaps the
miscompiling construct for a construct that computes correctly on the
same hardware, and the affected suite goes green.

1. One straight-line word load per thread at the top of the function,
   byte skew for misalignment, no loop-carried loads. The probe ladder
   in the receipt shows the same load reads correctly only in that
   shape.
   Receipt: https://github.com/joshuaswarren/mlx-omarchy/blob/main/receipts/2026-08-31-m1-mlxlm-fp16-smoke.md
2. Four constant shifts picked by a `(pos & 3)` select chain instead of
   one dynamic shift. Constant per-lane shifts, a byte-table select, and
   a uvec4 swizzle all compute correctly on the same hardware.
   Receipt: https://github.com/joshuaswarren/mlx-omarchy/blob/main/receipts/2026-09-02-masked-scatter-m1-fix.md
3. The bool comparisons moved to their own selector-free shader
   (compare_bool.comp).
   Commit: https://github.com/joshuaswarren/mlx-omarchy/commit/3c7d25706f6200c6af0257b56d5201ea49625a54
4. The same constant-shift select chain (BYTE_AT) in five shaders.
   WARNING: the workaround is context-sensitive. In select.comp's
   packed-bool path the select-chain macro is WRONG and the plain
   shift-then-mask helper is CORRECT - the exact inverse of every other
   site. An eight-variant device probe pinned it: macro form wrong at 5
   of 17 positions, helper form wrong nowhere. Neither form is safe by
   default on this hardware; each site needs its own probe.
   Receipt: https://github.com/joshuaswarren/mlx-omarchy/blob/main/receipts/2026-09-02-m1-red-suites-root-cause.md
   Commit: https://github.com/joshuaswarren/mlx-omarchy/commit/959c7a0e14d11cb81b1888ad2215f920ce02a3f0
5. The same constant-shift select chain in load_truthy, verified with
   366 boundary checks plus the full battery, twice, on device.
   Commit: https://github.com/joshuaswarren/mlx-omarchy/commit/cf68e7dedf5c6c86428c77e3c86cddde0ca0091b
   Probe artifacts: https://github.com/joshuaswarren/mlx-omarchy/tree/main/receipts/boolall-2026-09-03

## The family claim: confirmed on device for these shaders

CLAIM: cases 2, 3, and 5 are one defect - Honeykrisp miscompiles
dynamic byte extraction, a shift by a data-dependent amount feeding a
mask. The claim now rests on paired controls run in THIS directory's
shaders on the affected hardware, not only on mlx-omarchy's kernels:
on jwm1 (2026-09-03), arms A and B (dynamic shift, with and without
the wide selector) both fail 24/33 identically, while arms C and D
(constant-shift chain) compute 33/33 correctly on the same device in
the same runs.

Evidence, strongest first:

- The case 3 matrix resolves its own confound. Arms A and B failing
  identically means the wide selector is neither necessary nor the
  trigger: the dynamic shift alone explains both. Arms C and D passing
  on the same device and run is the paired-control A/B result for the
  extraction hypothesis in these exact shaders.
- Case 2 pins the shift amount itself: the input word is a constant
  0x00010001 in every variant, so the input cannot be the variable. On
  hardware the scan produces 32, the receipt's window-64 signature.
- Case 5 reproduces with the recorded 33-element map: truthy reads
  correct only at positions 0-3 and 16-19, falsy elsewhere. The
  raw-word control in the same run matches the input, so the buffer
  content is correct and the reads are what fail.
- mlx-omarchy's original A/B results stand: in cases 4 and 5's
  kernels, swapping only the extraction form flipped suite results,
  repeated on device.

Honest misses, reported not dropped:

- Case 4 does not reproduce on this build, while the same construct
  in arm B's shape does. The receipts predicted exactly this:
  byte-extraction sites on this hardware are shape- and
  context-sensitive, and no form is safe by default. Case 4 is kept
  as the demonstration of that context sensitivity, and as a
  re-derivation target.
- Case 1 does not reproduce. Either the reconstruction lost the
  original probe's trigger or the load defect was fixed; the run
  cannot separate these. Cases 2 and 5 still failing on the same
  build says the extraction family is alive, and case 1 was always a
  separate signature.

What we still cannot claim:

- One single compiler-level origin is named only as a hypothesis (next
  section). The A/B results prove the trigger construct per shader,
  not that the compiler folds them through one faulty path.
- Case 5's hardware map (correct only at the 16-byte-aligned words)
  echoes case 1's signature inside case 5's extraction defect. Whether
  the two share a deeper origin in addressing remains open; the
  single-shader probe crossing load alignment against shift
  dynamicness would settle it.

Falsification: any case that miscompiles with a dynamic shift but
computes correctly when the SAME value reaches it through a
constant-shift select chain would refute the extraction hypothesis for
that case. Every probed site so far reports the opposite result - the
mlx-omarchy receipts across their kernels, and arms C/D of this branch
on jwm1 today.

## Where the origin may live in Mesa: one hypothesis, no patch

Reconstructing the construct in Mesa's AGX backend points at one named
candidate. In src/asahi/compiler/agx_nir_algebraic.py, the late
`agx_nir_fuse_algebraic_late` pass contains `fuse_ubfe`, which rewrites
`(ushr a, b) & ((1 << bits) - 1)` into `ubitfield_extract(a, b, bits)`
with NO constant requirement on the offset b. Codegen
(src/asahi/compiler/agx_compile.c, `nir_op_ubitfield_extract`) emits
hardware `bfeil` for that. So in Mesa:

- `(w >> s) & 0xFF` with a CONSTANT s fuses to `bfeil` with an
  immediate offset.
- `(w >> s) & 0xFF` with a DYNAMIC s also fuses to `bfeil`, with the
  offset in a register.

That const/dynamic split matches the hardware behavior exactly:
constant-shift forms compute correctly on the M1, dynamic-shift forms
miscompile - now measured in this directory's own shaders, not only in
the receipts. HYPOTHESIS: `bfeil` with a register offset is broken on
G13 - either the hardware does not support it the way the compiler
assumes, or the encoding path for register offsets is wrong. This is
inference from the lowering shape plus the hardware results; it is not
a documented upstream limitation, and the in-tree comment about
constant bitfield extracts (agx_nir_algebraic.py:81) is about constant
WIDTH, not constant offset. `fuse_ubfe` fuses dynamic offsets with
constant width by design ("get the win everywhere", 7193849f302). The
AGX_MESA_DEBUG=shaders dumps captured alongside the hardware run can
confirm or kill this directly: if the failing dynamic-shift paths
carry `bfeil` with a register offset while the passing constant-shift
paths use an immediate, the hypothesis is confirmed at the ISA level.

If this hypothesis is right, the minimal change is to gate `fuse_ubfe`
to constant offsets, leaving `ushr + iand`, which lowers to the
well-exercised `shr`:

    (('iand', ('ushr', 'a@32', '#b'), (1 << bits) - 1),
     ('ubitfield_extract', a, b, bits))

The `#b` marks the offset constant in NIR algebraic syntax. THIS PATCH
IS NOT APPLIED IN THIS BRANCH AND IS UNTESTED: not against hardware,
not against Mesa's test suites (no LLVM 15 + LLVMSPIRVLib toolchain was
available on the authoring box to build agx_tests), and performance on
other gens is unexamined. It is a pointer for bisection, not a
contribution. Case 1 (word loads) is NOT explained by this hypothesis;
it has no shift, and its 16-byte alignment signature points at the
load path.

Capturing the miscompiled ISA settles the origin hypothesis: the
hardware run above already captured AGX_MESA_DEBUG=shaders dumps for
every case (paths in the Hardware verdicts section). Diffing arm A
against arm C, or case 5 against a constant-shift variant, shows
whether the failing dynamic-shift paths carry bfeil with a register
offset. The flag is the compiler's debug option list in
src/asahi/compiler/agx_compile.c; the Vulkan driver applies it at
shader compile time in src/asahi/vulkan/hk_shader.c.

## Provenance and limits

- Written by the mlx-omarchy project (Joshua Warren), September 2026,
  from its receipts. The original probe programs were throwaway files
  in /tmp on the M1 and are not preserved; these shaders are
  reconstructions of the documented probe shapes, not copies.
- Verified on llvmpipe (Mesa LLVM 15.0.6, Vulkan 1.3.230): all nine
  case runs pass (eleven PASS checks), exit 0, proving the shaders and
  checks are correct on a conformant driver.
- Verified on Apple M1 under Honeykrisp (Vulkan API 1.4.354) on
  2026-09-03: cases 2, 3, and 5 reproduce with the receipt signatures;
  cases 1 and 4 do not reproduce, and are kept and labeled as misses
  with both readings. See "Hardware verdicts" above.
- The receipts' driver versions and wrong values are quoted from the
  mlx-omarchy receipts; the 2026-09-03 hardware verdicts were measured
  by this project on jwm1 and are quoted from that run's logs.
