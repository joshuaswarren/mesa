# Copyright 2016 Intel Corporation
# Copyright 2016 Broadcom
# Copyright 2020 Collabora, Ltd.
# Copyright 2025 Valve Corporation
# SPDX-License-Identifier: MIT

from mako.template import Template
from mako import exceptions
import sys
from isa import isa_prefix_for_file, load_isa

xmlfile = sys.argv[1]
isa = load_isa(xmlfile)
isa_prefix = isa_prefix_for_file(xmlfile)

# XXX: deduplicate
def to_alphanum(name):
    substitutions = {
        ' ': '_',
        '/': '_',
        '[': '',
        ']': '',
        '(': '',
        ')': '',
        '-': '_',
        ':': '',
        '.': '',
        ',': '',
        '=': '',
        '>': '',
        '#': '',
        '&': '',
        '*': '',
        '"': '',
        '+': '',
        '\'': '',
    }

    for i, j in substitutions.items():
        name = name.replace(i, j)

    return name

def safe_name(name):
    name = to_alphanum(name)
    if not name[0].isalpha():
        name = '_' + name

    return name.lower()

# Must match AGX_DISASM_WORDS in disasm-internal.h: 6 words cover the 24-byte
# AGX3d forms. At 4, no exact or known bit at or above bit 128 was emitted, so
# fields of 20- to 24-byte instructions above bit 127 were never matched and
# were reported as unknown bits.
DISASM_WORDS = 6

def chunk(x):
    assert x < (1 << (32 * DISASM_WORDS))
    chunks = [hex((x >> (32 * i)) & ((1 << 32) - 1)) for i in range(DISASM_WORDS)]
    while len(chunks) > 1 and chunks[-1] == '0x0':
        chunks.pop()

    return f'{{{", ".join(chunks)}}}'

def instr_is_trivial(I):
    return len(I.dests + I.sources + I.immediates + I.modifiers) == 0

# Instruction length lookups (the transcoder and the G18G re-encoding cache
# segment every program with them) test the forms in table order until one
# matches. Index the forms by LENGTH_INDEX_BITS of the instruction's first 64
# bits, the bits most forms fix: a bucket lists, in table order, the forms
# whose fixed bits agree with the bucket's. A form whose shortest length does
# not reach a bit does not fix it. So the first match within the bucket is the
# first match in the whole table.
LENGTH_INDEX_BITS = 10

def length_index(instrs):
    lengths = [min(I.length[0], I.length[-1]) for I in instrs]
    def fixes(I, n, b):
        return b < 8 * n and (I.exact.mask >> b) & 1
    count = [sum(1 for I, n in zip(instrs, lengths) if fixes(I, n, b))
             for b in range(64)]
    bits = sorted(sorted(range(64), key=lambda b: (-count[b], b))[:LENGTH_INDEX_BITS])

    buckets = [[] for _ in range(1 << LENGTH_INDEX_BITS)]
    for i, (I, n) in enumerate(zip(instrs, lengths)):
        mask = value = 0
        for k, b in enumerate(bits):
            if fixes(I, n, b):
                mask |= 1 << k
                value |= ((I.exact.value >> b) & 1) << k
        for key in range(1 << LENGTH_INDEX_BITS):
            if key & mask == value:
                buckets[key].append(i)

    start, forms = [0], []
    for bucket in buckets:
        forms += bucket
        start.append(len(forms))
    assert len(instrs) < (1 << 16) and len(forms) < (1 << 16)
    return bits, start, forms

def c_list(values, per_line=16):
    lines = [', '.join(str(v) for v in values[i:i + per_line])
             for i in range(0, len(values), per_line)]
    return ',\n   '.join(lines)

def extract_bits(x, bitset="r"):
    parts = []
    n = 0

    for r in x.bits.ranges:
        parts.append(f'bits({bitset}, {r.start}, {r.size}, {n})')
        n += r.size

    return ' | '.join(parts)

template = """
#include "disasm-internal.h"

% for en in isa.enums:
UNUSED static const char *enum_${safe_name(isa.enums[en].kind)}[] = {
% for n, v in isa.enums[en].values.items():
    [${n}] = "${v.display or ''}",
% endfor
};
% endfor

% for k, enc in isa.encodings.items():
static void
print_${k}(struct disasm_ctx *ctx, BITSET_WORD *orig, uint64_t v, bool source)
{
   BITSET_WORD r[] = { v, v >> 32 };
% for i, case in enumerate(enc.cases):
   ${"} else " if i > 0 else ""}if ((v & ${hex(case.exact.mask)}ull) == ${hex(case.exact.value)}ull) {
% for mod in case.modifiers:
% if mod.name == 'hint':
       if (${extract_bits(mod)} == 0) {
          fprintf(ctx->fp, "# missing hint");
       }

% endif
% endfor
       _print_operand(ctx, (struct operand_desc){
% if case.kind is not None:
          .kind = KIND_${case.kind.name.upper()},
% if case.kind.kind == 'signed':
          .value = util_sign_extend(${extract_bits(case.kind)} << ${case.kind.shift}, ${case.kind.bits.size()}),
% else:
          .value = ${extract_bits(case.kind)} << ${case.kind.shift},
% endif
% endif
% for mod in case.modifiers:
% if mod.name == 'mask':
          .count = util_bitcount(${extract_bits(mod)}),
% elif mod.name == 'count':
          .count = (${extract_bits(mod)}) ?: 4,
% elif mod.name == 'zx':
          .sx = ${extract_bits(mod)} ^ 1,
% else:
          .${mod.name} = ${extract_bits(mod)},
% endif
% endfor
% for prop, v in case.properties.items():
          .${prop} = ${v},
% endfor
          .optional = source,
       });
% endfor
   } else {
       fprintf(ctx->fp, "# XXX: Invalid value 0x%"PRIx64" for ${k}", v);
       ctx->error = true;
   }
}

% endfor

% for k, instr in isa.instrs.items():
% if not instr_is_trivial(instr):
static void
print_${instr.name}(struct disasm_ctx *ctx, BITSET_WORD *r)
{
% for i, op in enumerate(instr.dests):
   print_${op.kind}(ctx, r, ${extract_bits(op)}, false);
% endfor
% for i, op in enumerate(instr.sources):
   print_${op.kind}(ctx, r, ${extract_bits(op)}, true);
% endfor
% for imm in instr.immediates:
   print_immediate(ctx, ${'true' if imm.kind == 'signed' else 'false'}, ${extract_bits(imm)});
% endfor
% for mod in instr.modifiers:
% if mod.name != 'length':
% if mod.kind in isa.enums:
   print_enum(ctx, enum_${mod.kind}, ${extract_bits(mod)});
% else:
   print_modifier(ctx, "${mod.display}", ${extract_bits(mod)});
% endif
% endif
% endfor
}

% endif
% endfor
static struct disasm_spec decodings[] = {
% for k, instr in isa.instrs.items():
    { "${instr.display if instr_is_trivial(instr) else f'{instr.display:<8}'}", ${'NULL' if instr_is_trivial(instr) else f'print_{instr.name}'}, ${instr.length_bit}, ${instr.length[0]}, ${instr.length[-1]}, ${chunk(instr.exact.mask)}, ${chunk(instr.exact.value)}, ${chunk(instr.mask())} },
% endfor
};

signed
${isa_prefix}_disassemble_instr(BITSET_WORD *code, FILE *fp, unsigned offset, size_t available, bool verbose)
{
  return _disassemble_instr(code, fp, decodings, ARRAY_SIZE(decodings), available, offset, verbose);
}

<% index_bits, index_start, index_forms = length_index(list(isa.instrs.values())) %>
static const uint8_t length_index_bits[] = {
   ${c_list(index_bits)}
};

static const uint16_t length_index_start[] = {
   ${c_list(index_start)}
};

static const uint16_t length_index_forms[] = {
   ${c_list(index_forms)}
};

static const struct disasm_length_index length_index = {
   .nr_bits = ARRAY_SIZE(length_index_bits),
   .bits = length_index_bits,
   .start = length_index_start,
   .forms = length_index_forms,
};

signed
${isa_prefix}_instr_length(const uint8_t *code, size_t available, const char **display)
{
  return _instr_length(code, decodings, &length_index, available, display);
}

signed
${isa_prefix}_instr_length_linear(const uint8_t *code, size_t available, const char **display)
{
  return _instr_length_linear(code, decodings, ARRAY_SIZE(decodings), available, display);
}"""

try:
    print(Template(template).render(isa = isa, isa_prefix = isa_prefix, safe_name = safe_name, chunk = chunk, instr_is_trivial = instr_is_trivial, extract_bits = extract_bits, length_index = length_index, c_list = c_list))
except:
    print(exceptions.text_error_template().render())
