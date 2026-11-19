# G17G native RT0 EOT witness

The existing `g17p-native-rt0-eot.btnf.hex` is a hardware-independent builtin
factory request. Generate an independent M5 witness with the G17G target:

```sh
xxd -r -p g17p-native-rt0-eot.btnf.hex > eot_rt0.btnf
xcrun metal-nt -N eot_rt0.btnf -arch applegpu_g17g \
  -platform_version macos 26.0 26.0 -o eot_rt0.nat
```

Metal toolchain 27A5194o produced a 2632-byte native object with SHA-256
`bd9e36505453ed7239cc4b8fc5963bd9551a07dc6f88a7804da202e2be60be2c`.
Its nested fragment `__text` is 78 bytes at outer offset 0x770, SHA-256
`993b96ceeb6ca1622d34811dbe603fe3502768892775c7cb70805637c72db6df`:

```text
1c9c100624a814060c9d1026248101402801a00a2706040a2100a102680120000102
3708040a2100a1022801200001022700202a2100ab0a00004a301800250415002400
a02200e000550e000000
```

The builtin request is kind5, enabled RT mask1. This proves only an unspilled,
single-sample, single-layer RT0 store.

## The five imageblock format classes

The request is a description of the attachment, so the same factory emits the
program for any other colour format when three bytes of it are changed: the
internal format at file offset 0x75, the texture dimension at 0x76 and the
component count at 0x79. The component count never changes the text; the
format byte selects one of five imageblock format classes, exactly the five
that `g17p_rt0_eot` already tabulates for applegpu_g17p. G17G spells the class
in the two text bytes at offsets 69 and 71 instead of G17P's single selector
byte, and nothing else in the 78 bytes moves.

Generating the whole cross-product against applegpu_g17p reproduces the
existing G17P tables byte for byte, including the documented selector-0x75 text
(`e641858631bd55fb...`) and the R5G6B5 selector-0x35 text
(`60beeb253d8f81af...`), which is what qualifies the same request for G17G.

| G17P selector | format byte | components | key | applegpu_g17g SHA-256 (first 16 bytes) | covers |
|---|---|---|---|---|---|
| 0x75 | 3 | 4 | `0x10020300` | `993b96ceeb6ca1622d34811dbe603fe3` | U8NORM, S8NORM, SRGBA8, RGB10A2 |
| 0x15 | 8 | 2 | `0x08020800` | `dc11621c05dcc10b67fc8662f562091f` | I16, F16, the two-word U16NORM pair |
| 0x35 | 11 | 3 | `0x0c020b00` | `1e018256e20eb3d391b7cd522583eb9c` | I32 words, the three-lane R5G6B5 store |
| 0x55 | 4 | 1 | `0x04020400` | `f2aecada275a6c952f2e2dcf48b7f72e` | the one-word I8 integer keys |
| 0x95 | 6 | 4 | `0x10020600` | `788b7d3b692bb5d630bd9aa6e7bf9a6b` | the normalized-16 keys |

These are the `g17g_rt0_eot` table in `agx3_pack_binary.c`, in that order. The
84-byte signed I16 and signed I32 builtins that G17P reaches separately have no
G17G sibling here, so `agx3_rt0_eot_kind` refuses them on G17G rather than
storing a neighbouring class's bytes.

## Metadata and bindings

Native common metadata field0=3 GPR dwords and fields38/42 D=16 require
six GPR halves and eight staging halves. Mesa conservatively allocates eight
GPR halves. Native loader RT-store mask is1.

The native ESL matches the independently decoded G17P binding contract:
qword binding0x32 loads the PBE-table pointer into uniform halves0..3;
driver argument kind5/index0 loads the tile byte offset into halves4..5.
The admitted tile byte offset is0.

The compiler recognizes the complete explicit-coordinate post-RA graph,
requires an explicit G17G key, and rejects changed attachment index, offset,
dimension, format, and conflicting G17P selectors. Tests compare complete
native text and register metadata and exercise these rejection boundaries.
