# AGX3 generic instruction encoder

`agx3_compose.c` is one table-driven encoder for every form of the three AGX3
encodings: `agx3b` (G15, G16, G17P), `agx3c` (G17G, G17S, G17D, G18P) and
`agx3d` (G18G, G18M, G19P), the AGX3 instruction encoding tables, variants 0,
1 and 2.

## The tables

`agx3{b,c,d}_tables.txt` are the checked-in descriptions.
`gen_agx3_tables.py` turns each into `agx3{b,c,d}_tables.c` at build time.

A form is a tree. Its root node carries the predicates that gate the form,
the constant bits and field deposits it always makes, and its choice groups.
The encoder takes the first alternative of each group, in table order, whose
predicates hold and whose own groups resolve. An alternative is a node too.

The descriptions were made once from the C tables that used to be checked
in. Those repeated every choice group in every form that uses it. A
description names each distinct group once and is otherwise the same data:
`gen_agx3_tables.py --verify`, which the tests run, unfolds the generated
arrays into the old rows again and compares them with the old files by hash.
`src/asahi/isa/AGX3d.xml`, the disassembler's description of the same
encoding, is transcribed from the `agx3d` table.

### Description grammar

A description is lines. `#` starts a comment line. An example, with made-up
rows:

```
group 7
 67=0 57=0 68=0..2 ; 6&0001=0000 ; 68[0+2]>4.7:2,0,1,3,4 ; g4
group 12 : 84 83 29[0+4] ; 8&f0f0 3&0003
 0 0 2 ; 0010 0001
form iadd32 0 8
 23=2 79={1,2} ; 0&07ff=0357 ; 16[0+8]>2.1 ; g7 g12
```

`group N` starts choice group N; groups are numbered in file order. Each
indented line under it is one alternative, in table order. `form MNEMONIC EP
SIZE` starts a form of SIZE bytes; the one indented line under it is its root
node. Forms are in (mnemonic, ep) order.

A node line has up to four parts separated by `;`:

1. predicates on field values: `F=A` equals, `F=A..B` in the inclusive
   range, `F={A,B,C}` one of, `F[S+W]=A` the W bits from bit S equal A;
2. constant bits `WORD&MASK=VALUE`, in hex, on a 16-bit instruction word;
3. deposits `F[S+W]>WORD.D`: W bits of field F from bit S go to bit D of the
   word. With `:V0,V1,...` the field indexes that list first and the bits
   come from the listed value;
4. the node's choice groups, `gN`.

Trailing empty parts are left out. A node with nothing in it is a lone `;`.

A group whose alternatives have the same predicates and constant-bit masks
and differ only in the values is written as a table. `group N : SHAPE ;
MASKS` gives the predicates without their value (`F` or `F[S+W]`) and the
`WORD&MASK` list, and each row gives the values: decimal for the predicates,
hex for the bits.

Field ids are the tables' anonymous ones. Do not reorder the predicates or
the alternatives of an existing group: `--verify` compares row order too.
A description that is changed on purpose no longer holds the old rows:
delete its line of `LEGACY_SHA256` in `gen_agx3_tables.py` in the same
change. `--verify` then has nothing to compare that table with and passes.

## The encoder

The consumer supplies the tables' anonymous field values (`fields[160]`); the
encoder gates the form on its predicates, selects each choice group, and
deposits every fixed bit and field under a shared written-bit shadow, failing
closed on any contradiction, unconsumed nonzero field, or over-wide value.
`agx3_table.py` is the same encoder in Python, plus the decoder that turns an
instruction's bytes back into a field vector.

## Regenerating the G18G replay vectors

The vectors are derived from the committed witness texts in
`../test/native/g18g` with the in-tree table and XML. Write them outside the
tree and compare:

```sh
OUT=$(mktemp -d)
../test/gen_agx3d_compose_corpus_vectors.py $OUT/vectors.c
cmp $OUT/vectors.c ../test/agx3d_compose_corpus_vectors.c
```

## Validation

`test/test-agx3-compose-corpus.c` replays 1,494 real G16G/G17S corpus
instructions (decoded by the Python composer) and 694 distinct G18G
instructions from the witness texts, and the C encoder reproduces every one
byte-exactly.

The G18G replay is a round trip through the `agx3d` table: the field vectors
were decoded with the same table the encoder uses. It shows that the C
encoder, the Python decoder and Apple's bytes agree, but it cannot catch a
field-to-bit mapping that is wrong and self-consistent. The evidence for the
mapping itself is that the generator behind the `agx3d` table reproduces the
independently generated `agx3b` and `agx3c` tables byte-exactly.
