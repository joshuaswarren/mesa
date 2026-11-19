# G17G/G18G RGB10A2 fragment packing

Fresh Metal compiler output from `rgb10a2-pack.metal` and its pipeline descriptor
establishes native RGB10A2Unorm `pack` on G17P, G17G and G18G. Each target has seven
fragment objects: four register lanes, immediate alpha 1/0, float modifiers,
sources retained by a second attachment, texture input with a scoreboard wait,
and an all-immediate constant shader.

G17G/G18G retain the G17P operand fields and add the sub-opcode pair f82=3,
f83=116. Register operands use half-register indices; f27+17i discards each
source, while the retained-source witness leaves both hint bits clear. The
texture witness waits at f84. Immediate alpha 1 is minifloat 0x30 at f66.

The production encoder reproduces all seven G17P instructions and all six
non-leading G17G/G18G instructions byte-for-byte. The two leading constant
packs remain refused: their f102 marker needs a separately qualified G17G/G18G
shader-part relocation path. Packed float formats remain G17P-only; RGB10A2
unpack has its own G17G/G18G form (`agx3_rgb10a2_unpack_encode_g18g`).

Repeatable native extraction and original CTS replay artifacts are retained in
the J873 workbench (run 277). `compile-native.py --attempt NAME`
invokes `xcrun metal`, `metallib` and `metal-nt` for macOS 27.0, extracts each
fragment object, and verifies every instruction by decode/recompose. The saved
native-03 receipt pins Metal toolchain, inputs, tables and output bytes.
`verify-native.py --phase after --output DIRECTORY` feeds those independent
native operands through the production encoder and also checks ten invalid or
unqualified inputs are refused.

These are compiler witnesses, not GPU execution or multisample pixel proof.
