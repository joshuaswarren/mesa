# Copyright 2022 Alyssa Rosenzweig
# Copyright 2021 Collabora, Ltd.
# Copyright 2016 Intel Corporation
# SPDX-License-Identifier: MIT

import argparse
import sys
import math

a = 'a'
b = 'b'
c = 'c'
d = 'd'
e = 'e'
f = 'f'

lower_sm5_shift = []

# Our shifts differ from SM5 for the upper bits. Mask to match the NIR
# behaviour. Because this happens as a late lowering, NIR won't optimize the
# masking back out (that happens in the main nir_opt_algebraic).
for s in [8, 16, 32, 64]:
    for shift in ["ishl", "ishr", "ushr"]:
        lower_sm5_shift += [((shift, f'a@{s}', b),
                             (shift, a, ('iand', b, s - 1)))]

    for shift in ["imadshl_agx", "imsubshl_agx"]:
        lower_sm5_shift += [((shift, f'a@{s}', b, c, d),
                             (shift, a, b, c, ('iand', d, s - 1)))]

    # bitfield_extracts do a shift internally, so we need to mask the offset parameter
    lower_sm5_shift += [
        # These are based on the lowerings from nir_opt_algebraic, but conditioned
        # on the number of bits not being constant. If the bit count is constant
        # (the happy path) we can use our native instruction instead.
        (('ibitfield_extract', f'value@{s}', 'offset', 'bits(is_not_const)'),
         ('bcsel', ('ieq', 0, 'bits'),
          0,
          ('ishr',
           ('ishl', 'value', ('isub', ('isub', 32, 'bits'), ('iand', 'offset', s - 1))),
           ('isub', 32, 'bits')))),
 
        (('ubitfield_extract', f'value@{s}', 'offset', 'bits(is_not_const)'),
         ('iand',
          ('ushr', 'value', ('iand', 'offset', s - 1)),
          ('bcsel', ('ieq', 'bits', 32),
           0xffffffff,
           ('isub', ('ishl', 1, 'bits'), 1)))),

        # At this point, bitfield extracts are constant. We can only do constant
        # unsigned bitfield extract, so lower signed to unsigned + sign extend.
        (('ibitfield_extract', f'a@{s}', b, '#bits'),
         ('ishr', ('ishl', ('ubitfield_extract', a, ('iand', b, s - 1), 'bits'), ('isub', 32, 'bits')),
          ('isub', 32, 'bits'))),

        (("ibitfield_extract", f'a@{s}', 'offset', '#bits'),
         ("ibitfield_extract", a, ('iand', 'offset', s - 1), 'bits')),
        (("ubitfield_extract", f'value@{s}', 'offset', '#bits'),
         ("ubitfield_extract", 'value', ('iand', 'offset', s - 1), 'bits'))
    ]

# Based on the VIR lowering: step back one ulp when RTNE rounded away from zero.
def f2f16_rtz(x):
    return ('bcsel', ('flt', ('fabs', x), ('fabs', ('f2f32', ('f2f16_rtne', x)))),
            ('isub', ('f2f16_rtne', x), 1), ('f2f16_rtne', x))

lower_pack = [
    # The hardware converts with RTNE; D3D's f32tof16 (and AMD's packHalf)
    # truncate, and GLSL leaves the rounding to the implementation.
    (('pack_half_2x16_split', a, b),
     ('pack_32_2x16_split', f2f16_rtz(a), f2f16_rtz(b))),
    (('pack_half_2x16_rtz_split', a, b),
     ('pack_32_2x16_split', f2f16_rtz(a), f2f16_rtz(b))),

    # We don't have 8-bit ALU, so we need to lower this. But if we lower it like
    # this, we can at least coalesce the pack_32_2x16_split and only pay the
    # cost of the iors and ishl. (u2u16 of 8-bit is assumed free.)
    (('pack_32_4x8_split', a, b, c, d),
     ('pack_32_2x16_split', ('ior', ('u2u16', a), ('ishl', ('u2u16', b), 8)),
                            ('ior', ('u2u16', c), ('ishl', ('u2u16', d), 8)))),

    # Recognize open-coded 4x8 -> 32-bit byte packing of zero-extended bytes
    # and lower identically to pack_32_4x8_split, avoiding << 16 / << 24 shift
    # immediates and halving live register width. Inner iors require
    # is_used_once so multi-use partial words do not stay live alongside the
    # rewritten 16-bit chain. Covers left-associative (including commuted
    # d<<24 | (c<<16 | (b<<8 | a))), balanced, and right-associative
    # a | (b<<8 | (c<<16 | d<<24)) trees.
    (('ior', ('ior(is_used_once)',
              ('ior(is_used_once)', ('u2u32', 'a@8'),
                                    ('ishl', ('u2u32', 'b@8'), 8)),
              ('ishl', ('u2u32', 'c@8'), 16)),
             ('ishl', ('u2u32', 'd@8'), 24)),
     ('pack_32_2x16_split', ('ior', ('u2u16', a), ('ishl', ('u2u16', b), 8)),
                            ('ior', ('u2u16', c), ('ishl', ('u2u16', d), 8)))),
    (('ior', ('ior(is_used_once)', ('u2u32', 'a@8'),
                                   ('ishl', ('u2u32', 'b@8'), 8)),
             ('ior(is_used_once)', ('ishl', ('u2u32', 'c@8'), 16),
                                   ('ishl', ('u2u32', 'd@8'), 24))),
     ('pack_32_2x16_split', ('ior', ('u2u16', a), ('ishl', ('u2u16', b), 8)),
                            ('ior', ('u2u16', c), ('ishl', ('u2u16', d), 8)))),
    (('ior', ('u2u32', 'a@8'),
             ('ior(is_used_once)',
              ('ishl', ('u2u32', 'b@8'), 8),
              ('ior(is_used_once)', ('ishl', ('u2u32', 'c@8'), 16),
                                    ('ishl', ('u2u32', 'd@8'), 24)))),
     ('pack_32_2x16_split', ('ior', ('u2u16', a), ('ishl', ('u2u16', b), 8)),
                            ('ior', ('u2u16', c), ('ishl', ('u2u16', d), 8)))),

    # A half taken from a word and narrowed again (u2u16(extract_u16(x, 1)),
    # as the cooperative matrix lowering builds an operand from a 32-bit
    # load) is the half itself. Without this the u2u32 of the rewrite below
    # stays: a 16-to-32-bit move and a split per half, which register
    # allocation turned into two moves per operand half.
    (('u2u16', ('u2u32', 'a@16')), a),
    (('u2u16', ('i2i32', 'a@16')), a),

    (('extract_u16', 'a@32', 0), ('u2u32', ('unpack_32_2x16_split_x', a))),
    (('extract_u16', 'a@32', 1), ('u2u32', ('unpack_32_2x16_split_y', a))),
    (('extract_i16', 'a@32', 0), ('i2i32', ('unpack_32_2x16_split_x', a))),
    (('extract_i16', 'a@32', 1), ('i2i32', ('unpack_32_2x16_split_y', a))),

    # For optimizing extract->convert sequences for unpack/pack norm
    (('u2f32', ('u2u32', a)), ('u2f32', a)),
    (('i2f32', ('i2i32', a)), ('i2f32', a)),

    # A value zero-extended from 8 or 16 bits is non-negative as a signed
    # integer, so its signed conversion is the unsigned conversion of the
    # narrow value, which reads the half register directly instead of
    # widening it first (SPIR-V's OpUConvert of a byte, then OpConvertSToF).
    # Only for a byte loaded on its own: one taken from a word, a four-byte
    # load included, converts better through the word (fold_ubyte_to_float,
    # fold_w8_word_unpack()).
    (('i2f32', ('u2u32', 'a@8(is_scalar_intrinsic_result)')), ('u2f32', a)),
    (('i2f32', ('u2u32', 'a@16')), ('u2f32', a)),
    (('i2f16', ('u2u16', 'a@8(is_scalar_intrinsic_result)')), ('u2f16', a)),

    # Chew through some 8-bit before the backend has to deal with it
    (('f2u8', a), ('u2u8', ('f2u16', a))),
    (('f2i8', a), ('u2u8', ('f2i16', a))),

    # Duplicated from nir_opt_algebraic since this pattern is generated by our
    # bounds checking optimization which needs to run relatively late.
    (('unpack_64_2x32_split_x', ('pack_64_2x32_split', a, b)), a),
    (('unpack_64_2x32_split_y', ('pack_64_2x32_split', a, b)), b),

    (('f2f16_rtz', 'a@32'), f2f16_rtz(a)),

    # Codegen depends on this trivial case being optimized out.
    (('ubitfield_extract', 'value', 'offset', 0), 0),
    (('ibitfield_extract', 'value', 'offset', 0), 0),
]

# Vulkan leaves the rounding of a float to bfloat16 conversion to the
# implementation unless it has an FPRoundingMode decoration (spirv_to_nir
# builds those from integer operations, so f2bf is never decorated), and the
# shared lower_bfloat16_conversions rule truncates. Round to nearest even
# instead, as constant folding of f2bf does (_mesa_float_to_bfloat16_bits_rte):
# add 0x7fff plus the lowest kept bit and keep the high half. A NaN would carry
# into the exponent or the sign, so it keeps its sign and top payload bits and
# becomes quiet. Only bf2f is left to the shared rule, which is exact.
lower_f2bf_rtne = [
    (('f2bf', 'a@32'),
     ('bcsel', ('fneu(preserve_nan_inf)', a, a),
               ('ior', ('unpack_32_2x16_split_y', a), 0x40),
               ('unpack_32_2x16_split_y',
                ('iadd', ('iadd', a, 0x7fff), ('iand', ('ushr', a, 16), 1))))),
]

lower_selects = []
for T, sizes, one in [('f', [16, 32], 1.0),
                      ('i', [8, 16, 32], 1),
                      ('b', [32], -1)]:
    for size in sizes:
        lower_selects.extend([
            ((f'b2{T}{size}', ('inot', 'a@1')), ('bcsel', a, 0, one)),
            ((f'b2{T}{size}', 'a@1'), ('bcsel', a, one, 0)),
        ])

# Rewriting bcsel(a || b, ...) in terms of bcsel(a, ...) and bcsel(b, ...) lets
# our rules to fuse compare-and-select do a better job, assuming that a and b
# are comparisons themselves.
#
# This needs to be a separate pass that runs after lower_selects, in order to
# pick up patterns like b2f32(iand(...))
opt_selects = [
        # Direct condition dominance in nested selects (only when the
        # rewritten intermediate select becomes dead, avoiding live-range
        # extension).
        (('bcsel', a, ('bcsel(is_used_once)', a, b, c), d), ('bcsel', a, b, d)),
        (('bcsel', a, b, ('bcsel(is_used_once)', a, c, d)), ('bcsel', a, b, d)),
        (('bcsel', a, c, ('bcsel(is_used_once)', b, ('bcsel', a, 'x', d), e)),
         ('bcsel', a, c, ('bcsel', b, d, e))),
        (('bcsel', a, c, ('bcsel(is_used_once)', b, d, ('bcsel', a, 'x', e))),
         ('bcsel', a, c, ('bcsel', b, d, e))),
        (('bcsel', a, ('bcsel(is_used_once)', b, ('bcsel', a, c, 'x'), d), e),
         ('bcsel', a, ('bcsel', b, c, d), e)),
        (('bcsel', a, ('bcsel(is_used_once)', b, d, ('bcsel', a, c, 'x')), e),
         ('bcsel', a, ('bcsel', b, d, c), e)),

        # Collapse selects whose boolean condition (ior/iand or boolean bcsel)
        # shares a predicate with an arm select, provided every use of the
        # boolean condition collapses so it becomes dead.
        (('bcsel', ('ior(agx_bcsel_cond_collapses)', a, b), ('bcsel(is_used_once)', a, c, d), e),
         ('bcsel', a, c, ('bcsel', b, d, e))),
        (('bcsel', ('ior(agx_bcsel_cond_collapses)', a, b), c, ('bcsel(is_used_once)', a, d, e)),
         ('bcsel', a, c, ('bcsel', b, c, e))),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, True, b), ('bcsel(is_used_once)', a, c, d), e),
         ('bcsel', a, c, ('bcsel', b, d, e))),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, True, b), c, ('bcsel(is_used_once)', a, d, e)),
         ('bcsel', a, c, ('bcsel', b, c, e))),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, b, True), ('bcsel(is_used_once)', a, c, d), e),
         ('bcsel', a, ('bcsel', b, c, e), d)),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, b, True), c, ('bcsel(is_used_once)', a, d, e)),
         ('bcsel', a, ('bcsel', b, c, d), c)),
        (('bcsel', ('iand(agx_bcsel_cond_collapses)', a, b), ('bcsel(is_used_once)', a, c, d), e),
         ('bcsel', a, ('bcsel', b, c, e), e)),
        (('bcsel', ('iand(agx_bcsel_cond_collapses)', a, b), c, ('bcsel(is_used_once)', a, d, e)),
         ('bcsel', a, ('bcsel', b, c, d), e)),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, b, False), ('bcsel(is_used_once)', a, c, d), e),
         ('bcsel', a, ('bcsel', b, c, e), e)),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, b, False), c, ('bcsel(is_used_once)', a, d, e)),
         ('bcsel', a, ('bcsel', b, c, d), e)),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, False, b), ('bcsel(is_used_once)', a, c, d), e),
         ('bcsel', a, e, ('bcsel', b, d, e))),
        (('bcsel', ('bcsel(agx_bcsel_cond_collapses)', a, False, b), c, ('bcsel(is_used_once)', a, d, e)),
         ('bcsel', a, d, ('bcsel', b, c, e))),

        # Hoist single-use negation out of select trees to cut live values and
        # let the final negation fold into a downstream add/sub.
        (('bcsel', a, ('ineg(is_used_once)', b), ('ineg(is_used_once)', c)),
         ('ineg', ('bcsel', a, b, c))),
        (('bcsel', a, ('u2u32', 'b@8'), ('u2u32', 'c@8')),
         ('u2u32', ('bcsel', a, b, c))),
        (('bcsel', a, ('u2u32', 'b@16'), ('u2u32', 'c@16')),
         ('u2u32', ('bcsel', a, b, c))),

        (('bcsel', ('ior(is_used_once)', ('inot', a), b), c, d),
         ('bcsel', a, ('bcsel', b, c, d), c)),

        (('bcsel', ('ior(is_used_once)', a, b), c, d),
         ('bcsel', a, c, ('bcsel', b, c, d))),

        (('bcsel', ('iand(is_used_once)', ('inot', a), b), c, d),
         ('bcsel', a, d, ('bcsel', b, c, d))),

        (('bcsel', ('iand(is_used_once)', a, b), c, d),
         ('bcsel', a, ('bcsel', b, c, d), d)),

]

# When the ior/iand is used multiple times, we can instead fuse the other way.
# Prefer placing a fusable comparison on the bcsel condition when the other
# operand is not a fusable comparison (e.g. an accumulated bcsel/ior chain),
# so `acc || (x < bound)` lowers to one icmpsel/fcmpsel instead of two.
opt_selects.extend([
        (('iand', ('inot', 'a@1(agx_is_fusable_cmp)'), 'b@1(agx_is_not_fusable_cmp)'),
         ('bcsel', a, False, b)),
        (('iand', 'a@1(agx_is_fusable_cmp)', 'b@1(agx_is_not_fusable_cmp)'),
         ('bcsel', a, b, False)),

        (('ior', ('inot', 'a@1(agx_is_fusable_cmp)'), 'b@1(agx_is_not_fusable_cmp)'),
         ('bcsel', a, b, True)),
        (('ior', 'a@1(agx_is_fusable_cmp)', 'b@1(agx_is_not_fusable_cmp)'),
         ('bcsel', a, True, b)),

        (('iand', ('inot', 'a@1'), b), ('bcsel', a, False, b)),
        (('iand', 'a@1', b), ('bcsel', a, b, False)),

        (('ior', ('inot', 'a@1'), b), ('bcsel', a, b, True)),
        (('ior', 'a@1', b), ('bcsel', a, True, b)),
])

# Rewriting fneg to ixor costs a uniform but lets us CSE the bcsel if this is
# applied to a whole vector, hence the has_multiple_uses heuristic.
opt_selects.extend([
        (('bcsel', 'a(has_multiple_uses)', ('fneg(is_used_once)', 'b@32'), b),
         ('ixor', b, ('bcsel', a, 1 << 31, 0))),

        (('bcsel', 'a(has_multiple_uses)', b, ('fneg(is_used_once)', 'b@32')),
         ('ixor', b, ('bcsel', a, 0, 1 << 31))),
])

fuse_extr = []
for start in range(32):
    fuse_extr.extend([
        (('ior', ('ushr', 'a@32', start), ('ishl', 'b@32', 32 - start)),
         ('extr_agx', a, b, start, 0)),
    ])

fuse_ubfe = []
for bits in range(1, 32):
    fuse_ubfe.extend([
        (('iand', ('ushr', 'a@32', b), (1 << bits) - 1),
         ('ubitfield_extract', a, b, bits))
    ])

fold_byte_swap = [
    (('ior', ('ushr(is_used_once)', 'a@32', 24),
             ('ior(is_used_once)',
                ('iand(is_used_once)', ('ushr(is_used_once)', a, 8), 0xff00),
                ('ior(is_used_once)', ('ishl(is_used_once)', a, 24),
                   ('iand(is_used_once)', ('ishl(is_used_once)', a, 8), 0xff0000)))),
     ('ior', ('iand', ('extr_agx', a, a, 8, 0), 0xff00ff00),
             ('iand', ('extr_agx', a, a, 24, 0), 0x00ff00ff))),
]

# (x * y) + s = (x * y) + (s << 0)
def imad(x, y, z):
    return ('imadshl_agx', x, y, z, 0)

# (x * y) - s = (x * y) - (s << 0)
def imsub(x, y, z):
    return ('imsubshl_agx', x, y, z, 0)

# x + (y << s) = (x * 1) + (y << s)
def iaddshl(x, y, s):
    return ('imadshl_agx', x, 1, y, s)

# x - (y << s) = (x * 1) - (y << s)
def isubshl(x, y, s):
    return ('imsubshl_agx', x, 1, y, s)

fuse_imad = [
    # Reassociate imul+iadd chain in order to fuse imads. This pattern comes up
    # in compute shader lowering.
    (('iadd', ('iadd(is_used_once)', ('imul(is_used_once)', a, b),
              ('imul(is_used_once)', c, d)), e),
     imad(a, b, imad(c, d, e))),

    # Fuse regular imad
    (('iadd', ('imul(is_used_once)', a, b), c), imad(a, b, c)),
    (('isub', ('imul(is_used_once)', a, b), c), imsub(a, b, c)),
]

for s in range(1, 5):
    fuse_imad += [
        # Definitions
        (('iadd', a, ('ishl(is_used_once)', b, s)), iaddshl(a, b, s)),
        (('isub', a, ('ishl(is_used_once)', b, s)), isubshl(a, b, s)),

        # ineg(x) is 0 - x
        (('ineg', ('ishl(is_used_once)', b, s)), isubshl(0, b, s)),

        # Definitions
        (imad(a, b, ('ishl(is_used_once)', c, s)), ('imadshl_agx', a, b, c, s)),
        (imsub(a, b, ('ishl(is_used_once)', c, s)), ('imsubshl_agx', a, b, c, s)),

        # The above but after the below shift lowering
        (imad(a, b, ('imadshl_agx(is_used_once)', 0, 1, c, s)), ('imadshl_agx', a, b, c, s)),
        (imsub(a, b, ('imadshl_agx(is_used_once)', 0, 1, c, s)), ('imsubshl_agx', a, b, c, s)),

        # a + (a << s) = a + a * (1 << s) = a * (1 + (1 << s))
        (('imul', a, 1 + (1 << s)), iaddshl(a, a, s)),

        # a - (a << s) = a - a * (1 << s) = a * (1 - (1 << s))
        (('imul', a, 1 - (1 << s)), isubshl(a, a, s)),

        # a - (a << s) = a * (1 - (1 << s)) = -(a * (1 << s) - 1)
        (('ineg', ('imul(is_used_once)', a, (1 << s) - 1)), isubshl(a, a, s)),

        # iadd is SCIB, general shfit is IC (slower)
        (('ishl', a, s), iaddshl(0, a, s)),
    ]

# If the above rules failed, we have a large constant shift on the IC unit.
# Might as well fuse an add to form an imad, if we're on the IC anyway.
fuse_imad += [
    (('iadd', a, ('ishl(is_used_once)', b, '#c')), imad(b, ('ishl', 1, c), a)),
]

# A value added twice is one shifted add: reassociation groups the duplicated
# sources of a chain as x + (y + y), an accumulation it leaves alone adds
# them in turn as (x + y) + y.
fuse_doubled_add = [
    (('iadd', 'a@32', ('iadd(is_used_once)', 'b@32', b)), iaddshl(a, b, 1)),
    (('iadd', ('iadd(is_used_once)', 'a@32', 'b@32'), b), iaddshl(a, b, 1)),
]

# Discard lowering generates this pattern, clean it up
ixor_bcsel = [
   (('ixor', ('bcsel', a, '#b', '#c'), '#d'),
    ('bcsel', a, ('ixor', b, d), ('ixor', c, d))),
]

# The main NIR optimizer works on imul, not iadd. We need just enough patterns
# for amul to let us fuse lea.
cleanup_amul = [
   # Neither operation overflows so we can keep the amul.
   (('amul', ('amul', a, '#b'), '#c'), ('amul', a, ('imul', b, c))),

   # Result of u2u64 has zero in upper half, so the shift doesn't overflow, so
   # neither multiplication overflows.
   (('amul', ('ishl', ('u2u64', 'a@32'), '#b(is_ult_32)'), '#c'),
    ('amul', ('u2u64', a), ('ishl', c, b))),

   # We depending on simplifying this.
   (('iadd', a, 0), a),
]

fuse_lea = []

# Handle 64-bit address arithmetic (OpenCL)
for s in range(1, 5):
    pot = 1 << s

    fuse_lea += [
        # A + (#b + c) 2^s = (A + c 2^s) + #b 2^s
        (('iadd', 'a@64', ('amul', pot, ('iadd', '#b(is_upper_half_zero)', ('u2u64', 'c@32')))),
         ('ulea_agx', ('ulea_agx', a, c, s), ('u2u32', b), s)),

        # A + (B + c) 2^s = (A + B 2^s) + c 2^s
        (('iadd', 'a@64', ('amul', ('iadd', 'b@64', ('i2i64', 'c@32')), pot)),
         ('ilea_agx', ('iadd', a, ('ishl', b, s)), c, s)),

        # A + 2^s (B + (C + d)) = (A + (B + C)2^s) + d 2^s
        (('iadd', 'a@64', ('amul', ('iadd', 'b@64',
                                   ('iadd', 'c@64', ('u2u64', 'd@32'))), pot)),
         ('ulea_agx', ('iadd', a, ('ishl', ('iadd', b, c), s)), d, s)),
    ]

    for sgn in ["u", "i"]:
        upconv = f'{sgn}2{sgn}64'
        lea = f'{sgn}lea_agx'

        fuse_lea += [
            # Basic pattern match
            (('iadd', 'a@64', ('amul', (upconv, 'b@32'), pot)), (lea, a, b, s)),
            (('iadd', 'a@64', ('ishl', (upconv, 'b@32'), s)), (lea, a, b, s)),
        ]

# Handle relaxed 32-bit address arithmetic (OpenGL, Vulkan)
for s_ in range(1, 5):
    # Iterate backwards
    s = 5 - s_

    v = 1 << s
    is_mult = f'(is_unsigned_multiple_of_{v})'
    is_mult_below_2g = f'(is_unsigned_multiple_of_{v}_below_2g)'

    fuse_lea += [
        # A + b * s = A + B * s with relaxed multiply
        (('iadd', 'a@64', ('u2u64', ('amul', 'b@32', v))),
         ('ulea_agx', a, b, s)),

        # A + (b * c 2^s) = A + (b * c) 2^s with relaxed multiply
        (('iadd', 'a@64', ('u2u64', ('amul', 'b@32', f'#c{is_mult}'))),
         ('ulea_agx', a, ('imul', b, ('ushr', c, s)), s)),

        # A + (b + c d) = A + ((b + c d) / 2^s) 2^s with relaxation, for b and
        # d multiples of 2^s.
        #
        # The offset is the 32-bit sum X = (b + c d) mod 2^32, zero-extended;
        # amul does not wrap. With unsigned shifts,
        #
        #   X / 2^s = ((b >> s) + c (d >> s)) mod 2^(32 - s)
        #
        # because 2^s ((b >> s) + c (d >> s)) = b + c d exactly. When b is below
        # 2^31 the sum wraps only for an offset past 4 GiB, outside every buffer,
        # so the mod is free. A larger b is either an offset past 2 GiB or a
        # negative one, b - 2^32 (buf[i - 1]); there the sum wraps, so mask it.
        (('iadd', 'a@64', ('u2u64', ('iadd', f'#b{is_mult_below_2g}',
                                     ('amul', 'c@32', f'#d{is_mult}')))),
         ('ulea_agx', a, ('iadd', ('ushr', b, s),
                                  ('amul', 'c@32', ('ushr', d, s))), s)),
        (('iadd', 'a@64', ('u2u64', ('iadd', f'#b{is_mult}',
                                     ('amul', 'c@32', f'#d{is_mult}')))),
         ('ulea_agx', a, ('iand', ('iadd', ('ushr', b, s),
                                           ('amul', 'c@32', ('ushr', d, s))),
                                  (1 << (32 - s)) - 1), s)),
    ]

# 8-bit rules are lowest precedence since we really like to fuse shifts
fuse_lea += [
    (('iadd', a, ('u2u64', 'b@32')), ('ulea_agx', a, b, 0)),
    (('iadd', a, ('i2i64', 'b@32')), ('ilea_agx', a, b, 0)),

    (('iadd', a, ('iadd', ('u2u64', 'b@32'), c)),
     ('ulea_agx', ('iadd', a, c), b, 0)),
    (('iadd', a, ('iadd', ('i2i64', 'b@32'), c)),
     ('ilea_agx', ('iadd', a, c), b, 0)),
]

# After lowering address arithmetic, the various address arithmetic opcodes are
# no longer useful. Lower them to regular arithmetic to let nir_opt_algebraic
# take over.
lower_lea = [
    (('amul', a, b), ('imul', a, b)),
    (('ulea_agx', a, b, c), ('iadd', a, ('ishl', ('u2u64', b), c))),
    (('ilea_agx', a, b, c), ('iadd', a, ('ishl', ('i2i64', b), c))),
]

# A fragment's integer pixel coordinate round trip: the 16-bit coordinate
# converted to float, offset to the pixel centre and truncated back. Exact for
# every 16-bit value (17 significant bits fit fp32), so fold it to the integer.
fold_pixel_coord = [
    (('f2u32', ('fadd', ('u2f32', 'a@16'), 0.5)), ('u2u32', a)),
    (('f2i32', ('fadd', ('u2f32', 'a@16'), 0.5)), ('u2u32', a)),
]

# AGX_G17P_EXTRACT16_BFE: nir_opt_algebraic turns `x >> 16` into
# extract_u16(x, 1), so the generic iand(ushr(a, b), mask) -> ubfe fusion never
# sees `(x >> 16) & mask`; the late lowering then produces
# u2u32(unpack_32_2x16_split_y(x)) & mask, which the G17P backend emits as a
# half copy, a half-to-word copy/clear pair, an ldimm and an AND (five
# instructions in the throughput fragment's exit block). A constant-width
# extract of bits [16, 16 + n) is the same value for every n < 16.
fold_extract16_bfe = []
for bits in range(1, 16):
    fold_extract16_bfe.extend([
        (('iand', ('extract_u16', 'a@32', 1), (1 << bits) - 1),
         ('ubitfield_extract', a, 16, bits)),
        (('iand', ('u2u32', ('unpack_32_2x16_split_y', 'a@32')), (1 << bits) - 1),
         ('ubitfield_extract', a, 16, bits)),
    ])

# Computing (2*b)*b first can overflow even when a - 2*a*b*b is finite.
# Multiplying a first bounds the intermediates when the final result is finite;
# keeping the factor of two in the final FMA also avoids overflow before the
# subtraction. Only reassociate non-precise FP32 operations with no shared
# intermediate. Run after the generic algebraic passes to preserve this order.
avoid_square_overflow = [
    (('~ffma@32', ('fneg', a),
      ('~fmul(is_used_once)', ('~fmul(is_used_once)', 2.0, b), b), a),
     ('ffma', -2.0, ('fmul', ('fmul', a, b), b), a)),
]

fold_aligned_bounds = []
for shift in (1, 2):
    size = 1 << shift
    for component in range(4):
        delta = component * size
        folded_offset = a if delta == 0 else ('iadd', a, delta)
        fold_aligned_bounds.append(
            (('ult(agx_has_reusable_bound_store_index)',
               ('iadd', ('iand', 'a@32', -size),
                              delta + size - 1), 'b@32'),
             ('ult', ('ushr', folded_offset, shift), ('ushr', b, shift))))

# D3D float-to-integer conversions saturate and map NaN to 0. DXVK spells them
# out as a conversion plus clamping selects; the AGX3 hardware conversion
# already does exactly this (verified for both signs of NaN, infinities and
# every boundary). Fold the selects into NIR's saturating conversions
# (f2i32_sat/f2u32_sat) so subsequent constant folding preserves saturation and
# NaN-to-zero semantics rather than treating out-of-range inputs as poison.
fold_d3d_float_to_int = [
    (('bcsel', ('fneu', 'a@32', a), 0,
      ('bcsel', ('flt', a, -2147483648.0), -0x80000000,
       ('bcsel', ('flt', 2147483520.0, a), 0x7fffffff, ('f2i32', a)))),
     ('f2i32_sat', a)),
    (('bcsel', ('flt', 4294967040.0, 'a@32'), 0xffffffff,
      ('f2u32', ('bcsel', ('fneu', a, a), 0.0, ('fmax', a, 0.0)))),
     ('f2u32_sat', a)),
    (('bcsel', ('flt', 4294967040.0, 'a@32'), 0xffffffff,
      ('f2u32', ('fmax', a, 0.0))),
     ('f2u32_sat', a)),
]

fold_selected_f2i32 = [
    (('bcsel@32(agx_selected_f2i32_controls_match)', 'a@1',
      ('f2i32(is_used_once)', 'b@32'), ('f2i32(is_used_once)', 'c@32')),
     ('f2i32', ('bcsel', a, b, c))),
]

# Contract inexact linear interpolation a*(1 - t) + b*t (and its FMA form
# ffma(1 - t, a, b*t)) into ffma(t, b - a, a), cutting 4 ops to 2 (fneg is a
# free source modifier on AGX3 f32add). Only non-exact (~) ops without
# SignedZeroInfNanPreserve (nnan,ninf,nsz) match, preserving precise /
# NoContraction and VK_KHR_shader_float_controls semantics.
opt_flrp_contract = [
    (('~ffma(nnan,ninf,nsz)',
      ('~fadd(nnan,ninf,nsz)', 1.0, ('fneg', 't')), 'a',
      ('~fmul(is_used_once,nnan,ninf,nsz)', 't', 'b')),
     ('ffma', 't', ('fadd', 'b', ('fneg', 'a')), 'a')),
    (('~fadd(nnan,ninf,nsz)',
      ('~fmul(is_used_once,nnan,ninf,nsz)',
       ('~fadd(nnan,ninf,nsz)', 1.0, ('fneg', 't')), 'a'),
      ('~fmul(is_used_once,nnan,ninf,nsz)', 't', 'b')),
     ('ffma', 't', ('fadd', 'b', ('fneg', 'a')), 'a')),
]

# A sign-extended byte converted to float: (x << (24 - 8k)) >> 24 is three
# instructions on M5 (insert, arithmetic shift, word conversion). int2flt has a
# signed-byte form that reads the low eight bits of a half register, so convert
# the byte from the half that holds it: free for bytes 0 and 2 of a word, one
# 16-bit shift for bytes 1 and 3 of a word. The emitter feeds the conversion
# the u2u8's source directly (agx_emit_alu, i2f of an 8-bit value).
fold_byte_to_float = []
for conv in ['i2f32', 'i2f16']:
    fold_byte_to_float.extend([
        ((conv, ('ishr', ('ishl', ('u2u32', 'a@16'), 24), 24)),
         (conv, ('u2u8', a))),
        ((conv, ('ishr', ('ishl', ('u2u32', 'a@16'), 16), 24)),
         (conv, ('u2u8', ('ushr', a, 8)))),
        ((conv, ('ishr', ('ishl', 'a@32', 24), 24)),
         (conv, ('u2u8', ('unpack_32_2x16_split_x', a)))),
        ((conv, ('ishr', ('ishl', 'a@32', 16), 24)),
         (conv, ('u2u8', ('ushr', ('unpack_32_2x16_split_x', a), 8)))),
        ((conv, ('ishr', ('ishl', 'a@32', 8), 24)),
         (conv, ('u2u8', ('unpack_32_2x16_split_y', a)))),
        ((conv, ('ibitfield_extract', 'a@32', 0, 8)),
         (conv, ('u2u8', ('unpack_32_2x16_split_x', a)))),
        ((conv, ('ibitfield_extract', 'a@32', 8, 8)),
         (conv, ('u2u8', ('ushr', ('unpack_32_2x16_split_x', a), 8)))),
        ((conv, ('ibitfield_extract', 'a@32', 16, 8)),
         (conv, ('u2u8', ('unpack_32_2x16_split_y', a)))),
    ])

# The same for unsigned bytes 0 and 2 of a word ((x >> 8k) & 0xff: an AND or
# a bitfield extract, then a word conversion), into the unsigned-byte form of
# int2flt (float32 results only). Bytes 1 and 3 cost two either way.
fold_ubyte_to_float = [
    (('u2f32', ('iand', ('u2u32', 'a@16'), 0xff)), ('u2f32', ('u2u8', a))),
    (('u2f32', ('iand', 'a@32', 0xff)),
     ('u2f32', ('u2u8', ('unpack_32_2x16_split_x', a)))),
    (('u2f32', ('iand', ('ushr', 'a@32', 16), 0xff)),
     ('u2f32', ('u2u8', ('unpack_32_2x16_split_y', a)))),
    (('u2f32', ('ubitfield_extract', 'a@32', 0, 8)),
     ('u2f32', ('u2u8', ('unpack_32_2x16_split_x', a)))),
    (('u2f32', ('ubitfield_extract', 'a@32', 16, 8)),
     ('u2f32', ('u2u8', ('unpack_32_2x16_split_y', a)))),
]

# |a - b| of two unsigned bytes: their 16-bit differences a - b and b - a are
# opt_selects hoists single-use negations out of select trees so the final
# negation can fold into a downstream add, but it runs after
# nir_opt_algebraic_late, which is what forms isub. Fold it here: a separate
# ineg costs a zero and a subtract on AGX3, the isub is one instruction.
fold_ineg_add = [
    (('iadd', a, ('ineg', b)), ('isub', a, b)),
]

# |a - b| and 2^16 - |a - b| > 255 (or both 0), so the smaller one is the
# magnitude. Two independent 16-bit subtracts and an unsigned minimum replace
# the widened subtract, its negation and the signed maximum.
def u8_absdiff():
    left, right = ('u2u16', 'a@8'), ('u2u16', 'b@8')
    absdiff = ('u2u32', ('umin@16', ('isub@16', left, right),
                         ('isub@16', right, left)))
    rules = []
    for delta in (('isub@32(is_used_once)', ('u2u32', 'a@8'), ('u2u32', 'b@8')),
                  ('iadd@32(is_used_once)', ('u2u32', 'a@8'),
                   ('ineg', ('u2u32', 'b@8')))):
        rules += [(('iabs', delta), absdiff)]
    for delta in (('isub@32', ('u2u32', 'a@8'), ('u2u32', 'b@8')),
                  ('iadd@32', ('u2u32', 'a@8'), ('ineg', ('u2u32', 'b@8')))):
        rules += [(('imax@32(agx_u8_absdiff_uses_match)', delta, ('ineg', delta)),
                   absdiff),
                  (('imax@32(agx_u8_absdiff_uses_match)', ('ineg', delta), delta),
                   absdiff)]
    return rules

# M5 converts an unsigned byte or half in a half register to f32 directly. A
# zero-extended value converts exactly as signed or unsigned, so the widening
# copy is not needed.
# Canonicalize constant offsets across integer subtractions so unrolled
# diagonal address expressions like (a + c1) - (b + c2) CSE onto (a - b) + (c1 - c2).
reassoc_const_offsets = [
    (('ineg@32', ('iadd@32(is_used_once)', 'a(is_not_const)', '#c')),
     ('iadd', ('ineg', a), ('ineg', c))),
    (('iadd@32', ('iadd@32', 'a(is_not_const)', '#c'), ('ineg', ('iadd@32', 'b(is_not_const)', '#d'))),
     ('iadd', ('isub', a, b), ('isub', c, d))),
    (('iadd@32', 'a(is_not_const)', ('ineg', ('iadd@32', 'b(is_not_const)', '#d'))),
     ('iadd', ('isub', a, b), ('ineg', d))),
    (('iadd@32', ('iadd@32', 'a(is_not_const)', '#c'), ('ineg', 'b(is_not_const)')),
     ('iadd', ('isub', a, b), c)),
    (('iadd@32', ('iadd@32(is_used_once)', 'a(is_not_const)', '#c'), ('iadd@32(is_used_once)', 'b(is_not_const)', '#d')),
     ('iadd', ('iadd', a, b), ('iadd', c, d))),
    (('iadd@32', ('iadd@32(is_used_once)', 'a(is_not_const)', '#c'), 'b(is_not_const)'),
     ('iadd', ('iadd', a, b), c)),
    (('isub@32', ('iadd@32', 'a(is_not_const)', '#c'), ('iadd@32', 'b(is_not_const)', '#d')),
     ('iadd', ('isub', a, b), ('isub', c, d))),
    (('isub@32', 'a(is_not_const)', ('iadd@32', 'b(is_not_const)', '#c')),
     ('iadd', ('isub', a, b), ('ineg', c))),
    (('isub@32', ('iadd@32', 'a(is_not_const)', '#c'), 'b(is_not_const)'),
     ('iadd', ('isub', a, b), c)),
]

# Drop aligned addends under low-bit masks ((a + b) & (2^k - 1) == b & (2^k - 1)
# when (a & (2^k - 1)) == 0) and merge sums of disjoint masks of the same value
# ((x & m1) + (x & m2) == x & (m1 | m2)). Collapses redundant byte-in-word
# shared/buffer address splits before imad fusion.
fold_aligned_masks = [
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_2)', 'b@32'), 1), ('iand', 'b', 1)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_4)', 'b@32'), 3), ('iand', 'b', 3)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_8)', 'b@32'), 7), ('iand', 'b', 7)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_16)', 'b@32'), 15), ('iand', 'b', 15)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_32)', 'b@32'), 31), ('iand', 'b', 31)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_64)', 'b@32'), 63), ('iand', 'b', 63)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_128)', 'b@32'), 127), ('iand', 'b', 127)),
    (('iand', ('iadd', 'a@32(agx_is_multiple_of_256)', 'b@32'), 255), ('iand', 'b', 255)),
    (('ishl', ('ushr', 'a@32(agx_is_multiple_of_4)', 2), 1), ('ushr', 'a', 1)),
    (('iand', ('ushr', 'a@32(agx_is_multiple_of_4)', 1), 0xfffffffe), ('ushr', 'a', 1)),
    (('ishl', ('ushr', 'a@32(agx_is_multiple_of_2)', 1), 1), 'a'),
    (('ishl', ('ushr', 'a@32(agx_is_multiple_of_4)', 2), 2), 'a'),
    (('iand', ('iand', 'x@32', '#c'), '#d'), ('iand', 'x', ('iand', c, d))),
    (('iand', 'x@32', 0xffffffff), 'x'),
    (('iadd(agx_iadd_disjoint_masks)', ('iand', 'x@32', '#c'), ('iand', 'x@32', '#d')),
     ('iand', 'x', ('ior', c, d))),
    (('iadd(agx_iadd_disjoint_masks_nested)', ('iand', 'x@32', '#c'),
      ('iadd', 'a@32', ('iand', 'x@32', '#d'))),
     ('iadd', 'a', ('iand', 'x', ('ior', c, d)))),
]

m5_narrow_to_float = [
    (('i2f32', ('u2u32', 'a@8')), ('u2f32', a)),
    (('u2f32', ('u2u32', 'a@8')), ('u2f32', a)),
    (('i2f32', ('u2u32', 'a@16')), ('u2f32', a)),
    (('u2f32', ('u2u32', 'a@16')), ('u2f32', a)),
    (('u2f32', ('iand', ('u2u32', 'a@16'), '#c(agx_is_non_byte_mask16)')),
     ('u2f32', ('iand@16', 'a', ('u2u16', 'c')))),
    (('i2f32', ('iand', ('u2u32', 'a@16'), '#c(agx_is_non_byte_mask16)')),
     ('u2f32', ('iand@16', 'a', ('u2u16', 'c')))),
    (('u2f32', ('ushr', ('u2u32', 'a@16'), '#k(agx_is_shift_1_to_15)')),
     ('u2f32', ('ushr@16', 'a', 'k'))),
    (('i2f32', ('ushr', ('u2u32', 'a@16'), '#k(agx_is_shift_1_to_15)')),
     ('u2f32', ('ushr@16', 'a', 'k'))),
    (('u2f32', ('ubitfield_extract(agx_is_top_bits_ubfe16)', ('u2u32', 'a@16'), '#o', '#b')),
     ('u2f32', ('ushr@16', 'a', 'o'))),
    (('i2f32', ('ubitfield_extract(agx_is_top_bits_ubfe16)', ('u2u32', 'a@16'), '#o', '#b')),
     ('u2f32', ('ushr@16', 'a', 'o'))),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--import-path', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.import_path)
    run()

def run():
    import nir_algebraic  # pylint: disable=import-error

    print('#include "agx_nir.h"')
    print('#include "agx_nir_aligned_bounds.h"')
    print('#include "agx_nir_algebraic_helpers.h"')
    print('#include "nir.h"')
    print('#include "nir_search.h"')
    print("""
static bool
agx_is_fusable_cmp(UNUSED const nir_search_state *state,
                   const nir_alu_instr *instr, unsigned src,
                   UNUSED unsigned num_components,
                   UNUSED const uint8_t *swizzle)
{
   nir_alu_instr *src_alu = nir_src_as_alu(instr->src[src].src);
   if (!src_alu || src_alu->def.num_components != 1)
      return false;
   switch (src_alu->op) {
   case nir_op_ilt:
   case nir_op_ige:
   case nir_op_ult:
   case nir_op_uge:
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_feq:
   case nir_op_fneu:
      return true;
   default:
      return false;
   }
}

static bool
agx_is_not_fusable_cmp(UNUSED const nir_search_state *state,
                       const nir_alu_instr *instr, unsigned src,
                       unsigned num_components, const uint8_t *swizzle)
{
   return !agx_is_fusable_cmp(state, instr, src, num_components, swizzle);
}

static bool
agx_selected_f2i32_controls_match(const nir_alu_instr *instr)
{
   if (instr->op != nir_op_bcsel || instr->def.bit_size != 32 ||
       instr->def.num_components != 1)
      return false;
   nir_instr *left_instr = nir_def_instr(instr->src[1].src.ssa);
   nir_instr *right_instr = nir_def_instr(instr->src[2].src.ssa);
   if (left_instr->type != nir_instr_type_alu ||
       right_instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *left = nir_instr_as_alu(left_instr);
   const nir_alu_instr *right = nir_instr_as_alu(right_instr);
   return left->op == nir_op_f2i32 && right->op == nir_op_f2i32 &&
          left->def.num_components == 1 && right->def.num_components == 1 &&
          left->fp_math_ctrl == right->fp_math_ctrl;
}
""")

    print("""
/* imax(d, -d) where d has no other use than this maximum and its negation. */
/* True when every use of a 1-bit boolean op is the condition of a bcsel whose
 * true or false arm is a single-use bcsel sharing one of this op's predicates,
 * so collapsing all uses kills the boolean op with zero live-value increase. */
static bool
agx_bcsel_cond_collapses(const nir_alu_instr *instr)
{
   if (instr->def.bit_size != 1 || instr->def.num_components != 1)
      return false;
   if (list_is_singular(&instr->def.uses))
      return true;
   nir_def *op0 = instr->src[0].src.ssa;
   nir_def *op1 = NULL;
   if (instr->op == nir_op_ior || instr->op == nir_op_iand) {
      op1 = instr->src[1].src.ssa;
   } else if (instr->op == nir_op_bcsel) {
      if (nir_src_is_const(instr->src[1].src))
         op1 = instr->src[2].src.ssa;
      else if (nir_src_is_const(instr->src[2].src))
         op1 = instr->src[1].src.ssa;
      else
         return false;
   } else {
      return false;
   }
   nir_foreach_use_including_if(src, &instr->def) {
      if (nir_src_is_if(src))
         return false;
      nir_instr *use = nir_src_use_instr(src);
      if (use->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *ualu = nir_instr_as_alu(use);
      if (ualu->op != nir_op_bcsel || src != &ualu->src[0].src)
         return false;
      bool matched = false;
      for (unsigned arm = 1; arm <= 2; arm++) {
         nir_alu_instr *arm_alu = nir_src_as_alu(ualu->src[arm].src);
         if (arm_alu && arm_alu->op == nir_op_bcsel &&
             list_is_singular(&arm_alu->def.uses) &&
             (arm_alu->src[0].src.ssa == op0 ||
              arm_alu->src[0].src.ssa == op1)) {
            matched = true;
            break;
         }
      }
      if (!matched)
         return false;
   }
   return true;
}

static bool
agx_u8_absdiff_uses_match(const nir_alu_instr *instr)
{
   const nir_alu_instr *x = nir_src_as_alu(instr->src[0].src);
   const nir_alu_instr *y = nir_src_as_alu(instr->src[1].src);
   if (!x || !y)
      return false;
   const nir_alu_instr *neg = x->op == nir_op_ineg ? x : y;
   const nir_alu_instr *delta = x->op == nir_op_ineg ? y : x;
   return neg->op == nir_op_ineg && neg != delta &&
          neg->src[0].src.ssa == &delta->def &&
          neg->instr.block == instr->instr.block &&
          delta->instr.block == instr->instr.block &&
          list_is_singular(&neg->def.uses) &&
          list_length(&delta->def.uses) == 2;
}
""")

    print(nir_algebraic.AlgebraicPass("agx_nir_cleanup_amul", cleanup_amul).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fuse_lea", fuse_lea).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_lower_lea", lower_lea).render())

    print(nir_algebraic.AlgebraicPass("agx_nir_lower_algebraic_late",
                                      lower_sm5_shift + lower_pack +
                                      lower_selects).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_lower_f2bf_rtne",
                                      lower_f2bf_rtne).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_avoid_square_overflow",
                                      avoid_square_overflow).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fuse_selects",
                                      opt_selects).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_pixel_coord",
                                      fold_pixel_coord).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_ineg_add",
                                      fold_ineg_add).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_extract16_bfe",
                                      fold_extract16_bfe).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fuse_doubled_add",
                                      fuse_doubled_add).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_byte_to_float",
                                      fold_byte_to_float).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_ubyte_to_float",
                                      fold_ubyte_to_float).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_d3d_float_to_int",
                                      fold_d3d_float_to_int).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_opt_u8_absdiff",
                                      u8_absdiff()).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_opt_m5_narrow_to_float",
                                      m5_narrow_to_float).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_reassoc_const_offsets",
                                      reassoc_const_offsets).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fold_aligned_masks",
                                      fold_aligned_masks).render())
    print(nir_algebraic.AlgebraicPass("agx_nir_fuse_algebraic_late",
                                      fuse_extr + fuse_ubfe +
                                      fuse_imad + ixor_bcsel +
                                      fold_aligned_bounds + fold_byte_swap +
                                      fold_selected_f2i32 +
                                      opt_flrp_contract).render())


if __name__ == '__main__':
    main()
