Standalone reproductions for five Honeykrisp shader miscompiles.

Each case is a Vulkan compute shader plus a check that runs on any
Vulkan 1.1 device. On conformant drivers all cases pass. On the affected
Honeykrisp build they fail with the wrong values listed below, which the
mlx-omarchy project measured on real Apple M1 hardware.

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
with the receipt that recorded the observed values.

## Hardware and driver

All observed values come from the mlx-omarchy project's receipts,
measured on one machine: jwm1-linux, Apple M1 (G13G B1, 8 GB), running
Omarchy with Mesa's Honeykrisp Vulkan driver.

- Cases 1 and 4: Vulkan API 1.4.354 per the receipts. The 2026-09-02
  receipts disagree with each other (1.4.354 in one, 1.4.357 in
  another, both dated the same day), so the exact driver build per case
  is uncertain. Case 2: 1.4.357 per its receipt. Case 5: no API version
  recorded.
- The mlx-omarchy receipts do not record the Mesa commit. The defects
  are present in the 1.4.354/1.4.357 builds shipped by Omarchy in
  August/September 2026. Nobody has re-run these exact shaders on
  Apple hardware yet; that is the first thing to do with this branch.

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

## The family claim: hypothesis, with a named confound

CLAIM (hypothesis): cases 2, 4, and 5 are one defect - Honeykrisp
miscompiles dynamic byte extraction, a shift by a data-dependent amount
feeding a mask. Case 3 is "observed with" the family, not proven part
of it. Case 1 has a different signature.

Evidence for the claim, strongest first:

- Cases 4 and 5 hold everything constant except the extraction form and
  swap results: dynamic shift wrong, constant-shift select chain
  correct, on the same device, in repeated runs. That is an A/B result,
  not a correlation.
- Case 2 pins the shift amount itself: the input word is a constant
  0x00010001 in every variant, so the input cannot be the variable. The
  same scan fed by constant shifts computes 128 correctly; fed by the
  dynamic shift it computes 32.
- Case 5's fix was applied without any other change to the reduction
  and turned 85 of 366 failing boundary checks into 0, twice on device.
  The cause is pinned per the workaround, not guessed.

The named confound, and why case 3 is demoted:

- The original evidence for case 3 is a pair: a wide-selector shader
  was red and a selector-free control was green. But the control shared
  the same dynamic byte extraction, so if the extraction alone
  miscompiles, both results are explained without any selector effect.
  The same mlx-omarchy commit also fixed an unrelated
  output-initialization defect (missing ClearU32 before atomicOr),
  which poisoned composed bool operations on its own.
- Case 3 therefore ships as a four-arm matrix instead of a claim. Arms
  A and C run under the wide selector with dynamic and constant
  extraction; arms B and D run selector-free with the same two
  extractions. The decision table is in the shader header. Running the
  matrix on affected hardware either separates the selector from the
  shift or collapses case 3 into the family; both outcomes sharpen the
  picture, and no outcome is claimed in advance.

What we cannot claim:

- One single compiler-level origin is NOT established by the receipts.
  The A/B results prove the trigger construct per site. They do not
  prove the compiler folds all sites through one faulty path.
- Case 5's recorded 33-element map shows truthy reads correct exactly
  at positions 0-3 and 16-19 - the words whose byte addresses are
  16-byte aligned - and falsy elsewhere. That is case 1's signature
  (word loads read correctly only at 16-byte-aligned addresses)
  appearing inside case 5's extraction defect. Two readings fit: the
  extraction defect and the load defect share a deeper origin in
  addressing, or two distinct defects produce matching shadows at this
  shape. A single-shader probe crossing load alignment against shift
  dynamicness, per word, would separate them.

Falsification: any case that miscompiles with a dynamic shift but
computes correctly when the SAME value reaches it through a
constant-shift select chain would refute the extraction hypothesis for
that case. The mlx-omarchy receipts report the opposite result at every
site probed so far.

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
miscompile. HYPOTHESIS: `bfeil` with a register offset is broken on
G13 - either the hardware does not support it the way the compiler
assumes, or the encoding path for register offsets is wrong. This is
inference from the lowering shape plus the receipts; it is not a
documented upstream limitation, and the in-tree comment about constant
bitfield extracts (agx_nir_algebraic.py:81) is about constant WIDTH,
not constant offset. `fuse_ubfe` fuses dynamic offsets with constant
width by design ("get the win everywhere", 7193849f302).

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

Capturing the miscompiled ISA on Apple hardware settles it: run any
case under the Honeykrisp driver with `AGX_MESA_DEBUG=shaders` to dump
NIR and AGX IR for the case shaders (the option list and the
`AGX_MESA_DEBUG` lookup live in src/asahi/compiler/agx_compile.c; the
Vulkan driver applies the flags at shader compile time in
src/asahi/vulkan/hk_shader.c), and
compare the dynamic-shift dispatch against the constant-shift control
in the same run. The case pair to diff is arm A against arm C, or case
5 against a constant-shift variant.

## Provenance and limits

- Written by the mlx-omarchy project (Joshua Warren), September 2026,
  from its receipts. The original probe programs were throwaway files
  in /tmp on the M1 and are not preserved; these shaders are
  reconstructions of the documented probe shapes, not copies.
- Verified here only on llvmpipe (Mesa LLVM 15.0.6, Vulkan 1.3.230):
  all nine case runs pass (eleven PASS checks), exit 0. llvmpipe computing the expected
  values proves the shaders and checks are correct on a conformant
  driver; it proves nothing about Honeykrisp, which is exactly the
  point - every observed failure above was invisible on llvmpipe.
- Case 3's observed column is the weakest: see the confound section.
- The driver versions and the wrong values are quoted from the
  receipts. Nothing on this branch has been run on Apple hardware.
