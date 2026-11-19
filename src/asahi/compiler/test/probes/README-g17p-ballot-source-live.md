# G17P ballot source lifetime

`g17p-ballot-source-live.metal` and its pipeline script compile two kernels
which differ only in whether the ballot's input is used afterwards:

```c
int x = as_type<int>(tid * tid);
uint b = uint(simd_vote::vote_t(simd_ballot(x < 0)));
out[tid] = b;      // ballot_dead
out[tid] = b + x;  // ballot_live
```

Metal toolchain v27.1.5194.15, `metal-nt -arch applegpu_g17p
-platform_version macos 27.0 27.0`, produces these ten-byte instructions:

| Kernel | Native ballot | Encoder descriptor |
|---|---|---|
| dead | `17075400020000582206` | signed less-than, src 0, dst 0, source dead |
| live | `17075404030002182206` | signed less-than, src 0, dst 4, destination cache hint, source live |

`test-agx3-subgroup-core.c` reproduces the dead instruction exactly
(`int_lt_b`) and checks the live-source form of a register comparison
(`int_live_b`). The register comparison uses source field 25 for the live read and field 27 for
the final read. Previously `agx3_subgroup_encode_ballot` always selected field
27 for register comparisons, ignoring `src_live`.

On G17P hardware this destroyed the ballot word in
`poly_work_group_first_true` before its subsequent broadcast. The subgroup
ballot and shared-memory copy were correct (e.g. `0xffffff00`), but the
selected word became zero and the result was 32 instead of 8. The corrected
encoder preserves the word and produces first positions 0, 4, 8 and 12 in
controlled restart probes. The unmodified production restart kernel then
passes `KHR-GL33.primitive_restart.restart_mode`; token-free and
restart-enabled adjacency rendering match llvmpipe for direct and indirect
draws, including leading and trailing restart tokens.

The fix changes source lifetime metadata. It adds no instructions, waits,
barriers or draw calls. The unrelated late-readback visibility defect remains
unresolved.

To reproduce the native code, compile the MSL with `xcrun metal -c`, package
the AIR with `xcrun metallib`, and pass the `.mtlp-json` to `metal-nt -N` with
the target flags above. Each pipeline is a nested Mach-O inside the outer
`__TEXT,__compute` section; its code is in its own `__TEXT,__text` section.
