# G17G eight-target RGBA8 end-of-tile witness

`KHR-GL30.draw_buffers.draw_buffers_1` binds eight colour attachments, and the
G17G draw profile admits one. The store program for eight targets is not
captured, it is *generated*, exactly as the single-target one is (see
`README-g17g-native-rt0-eot.md`).

The builtin request is a flatbuffer whose per-attachment record simply
repeats. The three requests it was read from (`rt0-eot`, `raw32-mrt2` and
`raw32-mrt3`, not kept in this tree) give the rule, and it reproduces all three
byte for byte:

* offset 0x50 — enabled RT mask, `(1 << n) - 1`
* offset 0x54 — attachment count `n`
* offset 0x58 — `n` uoffsets; element `i` at `V + 4i` holds
  `(B + 32i + 0x14) - (V + 4i)` where `V = 0x58` and `B = V + 4n`
* then `n` per-attachment records of 31 bytes, each but the last padded to 32

The record differs between classes in one byte only, the internal format at
its offset 0x19: `0x03` is RGBA8 (the `rt0-eot` request) and `0x0a` is raw32
(the `raw32-mrt2`/`raw32-mrt3` requests). `g17g-native-rgba8-mrt8.btnf.hex`
is the RGBA8 record repeated eight times.

```sh
xxd -r -p g17g-native-rgba8-mrt8.btnf.hex > eot8.btnf
xcrun metal-nt -N eot8.btnf -arch applegpu_g17g \
  -platform_version macos 26.0 26.0 -o eot8.nat
python3 ../../../../../gpu/extract-m5-native-text.py eot8.nat eot8-text
```

Metal toolchain 27A5194o accepts it and produces a 3096-byte native object
whose nested fragment `__text` is the 380 bytes in
`g17g-native-rgba8-mrt8.text.bin`: four `rld`, two `imadd32`, then eight
`pack_imgwir` publications with attachment indices 0..7 and their coordinate
copies, interleaved with `imgwblk` block writes. The raw32 form of the same
eight-target program differs only in the `imgwblk` format field (`00e000`
for RGBA8 against `006000` for raw32).

This is the store program alone. Driving eight attachments also needs the
tile buffer layout, a PBE argument table of eight 32-byte entries, the
fragment record's resource counts and the Gallium draw profile's colour
attachment shape. The kernel already permits it: `UAPI_MAX_ATTACHMENTS` is
16 and `admit_reference_render` does not constrain the colour count.
