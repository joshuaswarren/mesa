"""
Copyright 2021 Alyssa Rosenzweig
# SPDX-License-Identifier: MIT
"""

opcodes = {}
immediates = {}
enums = {}

VARIABLE = ~0

class Opcode(object):
   def __init__(self, name, dests, srcs, imms, is_float, can_eliminate,
                can_reorder, schedule_class, encoding, agx3_mnemonic):
      self.name = name
      self.dests = dests if dests != VARIABLE else 0
      self.srcs = srcs if srcs != VARIABLE else 0
      self.variable_srcs = (srcs == VARIABLE)
      self.variable_dests = (dests == VARIABLE)
      self.imms = imms
      self.is_float = is_float
      self.can_eliminate = can_eliminate
      self.can_reorder = can_reorder
      self.schedule_class = schedule_class
      self.encoding = encoding
      self.agx3_mnemonic = agx3_mnemonic

class Immediate(object):
   def __init__(self, name, ctype):
      self.name = name
      self.ctype = ctype

class Encoding(object):
   def __init__(self, description):
      (exact, length_short, length_long) = description

      # Convenience
      if length_long is None:
         length_long = length_short

      self.exact = exact
      self.length_short = length_short
      self.extensible = length_short != length_long

      if self.extensible:
         assert(length_long == length_short + (4 if length_short > 8 else 2))

def op(name, encoding, dests = 1, srcs = 0, imms = [], is_float = False,
        can_eliminate = True, can_reorder = True, schedule_class = "none",
        agx3_mnemonic = None):
   encoding = Encoding(encoding) if encoding is not None else None

   opcodes[name] = Opcode(name, dests, srcs, imms, is_float, can_eliminate,
                          can_reorder, schedule_class, encoding, agx3_mnemonic)

def immediate(name, ctype = "uint32_t"):
   imm = Immediate(name, ctype)
   immediates[name] = imm
   return imm

def enum(name, value_dict):
   enums[name] = value_dict
   return immediate(name, "enum agx_" + name)

L = (1 << 15)
_ = None

FORMAT = immediate("format", "enum agx_format")
MATRIX_TYPE = immediate("matrix_type")
IMM = immediate("imm", "uint64_t")
WRITEOUT = immediate("writeout")
INDEX = immediate("index")
COMPONENT = immediate("component")
CHANNELS = immediate("channels")
TRUTH_TABLE = immediate("truth_table")
ROUND = immediate("round", "enum agx_round")
SHIFT = immediate("shift")
MASK = immediate("mask")
BFI_MASK = immediate("bfi_mask")
LOD_MODE = immediate("lod_mode", "enum agx_lod_mode")
PIXEL_OFFSET = immediate("pixel_offset")
COLOUR_TARGET = immediate("colour_target", "uint8_t")
STACK_SIZE = immediate("stack_size", 'int16_t')
EXPLICIT_COORDS = immediate("explicit_coords", "bool")
PERSPECTIVE = immediate("perspective", "bool")

DIM = enum("dim", {
    0: '1d',
    1: '1d_array',
    2: '2d',
    3: '2d_array',
    4: '2d_ms',
    5: '3d',
    6: 'cube',
    7: 'cube_array',
    8: '2d_ms_array',
})

GATHER = enum("gather", {
	0b000: "none",
	0b001: "r",
	0b011: "g",
	0b101: "b",
	0b111: "a",
})

OFFSET = immediate("offset", "bool")
SHADOW = immediate("shadow", "bool")
SPARSE = immediate("sparse", "bool")
QUERY_LOD = immediate("query_lod", "bool")
COHERENT = immediate("coherent", "bool")
SCS_ZERO_EXTEND = immediate("scs_zero_extend", "bool")
SCS_C_STATUS_LOAD = immediate("scs_c_status_load", "bool")
SCS_C_LOAD_MARKER = immediate("scs_c_load_marker", "bool")
SCS_RT_COMPLETION = immediate("scs_rt_completion", "bool")
SCS_ADDRESS_SELECTOR = immediate("scs_address_selector", "uint8_t")
SCOREBOARD = immediate("scoreboard")
ICOND = immediate("icond", "enum agx_icond")
FCOND = immediate("fcond", "enum agx_fcond")
NEST = immediate("nest")
INVERT_COND = immediate("invert_cond")
NEST = immediate("nest")
TARGET = immediate("target", "agx_block *")
ZS = immediate("zs")
SR = enum("sr", {
   235: 'mesh_payload_lo',
   236: 'mesh_payload_hi',
   237: 'mesh_primitive_storage_lo',
   238: 'mesh_primitive_storage_hi',
   0:  'threadgroup_position_in_grid.x',
   1:  'threadgroup_position_in_grid.y',
   2:  'threadgroup_position_in_grid.z',
   4:  'threads_per_threadgroup.x',
   5:  'threads_per_threadgroup.y',
   6:  'threads_per_threadgroup.z',
   8:  'dispatch_threads_per_threadgroup.x',
   9:  'dispatch_threads_per_threadgroup.y',
   10: 'dispatch_threads_per_threadgroup.z',
   14: 'samples_log2',
   20: 'core_id',
   21: 'vm_slot',
   48: 'thread_position_in_threadgroup.x',
   49: 'thread_position_in_threadgroup.y',
   50: 'thread_position_in_threadgroup.z',
   51: 'thread_index_in_threadgroup',
   52: 'thread_index_in_subgroup',
   53: 'subgroup_index_in_threadgroup',
   56: 'active_thread_index_in_quad',
   57: 'total_active_threads_in_quad',
   58: 'active_thread_index_in_subgroup',
   59: 'total_active_threads_in_subgroup',
   60: 'coverage_mask',
   62: 'backfacing',
   63: 'is_active_thread',
   80: 'thread_position_in_grid.x',
   81: 'thread_position_in_grid.y',
   82: 'thread_position_in_grid.z',
   124: 'input_sample_mask',
   144: 'helper_op',
   146: 'helper_arg_l',
   147: 'helper_arg_h',
   # G17P builtin background reload: native half-register layer input.
   195: 'background_layer_id',
   # Native M5 ray shader reads normalized SR59 (hardware byte0xea).
   234: 'scs_thread_index',
   # Native object_amplify and mesh_amplified payload pointer witness.
   235: 'mesh_payload_lo',
   236: 'mesh_payload_hi',
   # Native primitive_color/primitive_id mesh witnesses read the per-workgroup
   # per-primitive record base from normalized SR64 (W32) and SR65 (W16).
   237: 'mesh_primitive_storage_lo',
   238: 'mesh_primitive_storage_hi',
   # AGX3 hardware-vertex inputs. Apple emits these as explicit 32-bit rld
   # instructions instead of the AGX2 preloaded-input ABI.
   216: 'instance_id',
   217: 'tess_patch_id',
   218: 'tess_coord_u',
   219: 'tess_coord_v',
   221: 'vertex_id',
})

ATOMIC_OPC = enum("atomic_opc", {
	0: 'add',
	1: 'sub',
	2: 'xchg',
	3: 'cmpxchg',
	4: 'umin',
	5: 'imin',
	6: 'umax',
	7: 'imax',
	8: 'and',
	9: 'or',
	10: 'xor',
	11: 'fadd',
})

INTERPOLATION = enum("interpolation", {
    0: 'center',
    1: 'sample',
    2: 'centroid',
    # We translate sample -> sample_register at pack time for simplicity
    3: 'sample_register',
})

SIMD_OP = enum("simd_op", {
    0b00000: 'and',
    0b00001: 'fadd',
    0b00010: 'or',
    0b00011: 'fmul',
    0b00100: 'xor',
    0b00101: 'fmin',
    0b00111: 'fmax',
    0b10000: 'iadd',
    0b10100: 'smin',
    0b10110: 'smax',
    0b11100: 'umin',
    0b11110: 'umax',
})

FUNOP = lambda x: (x << 28)
FUNOP_MASK = FUNOP((1 << 14) - 1)

def funop(name, opcode, schedule_class = "none"):
   op(name, (0x0A | (opcode << 28), 4, 6),
      srcs = 1, is_float = True, schedule_class = schedule_class)

def iunop(name, opcode):
    assert(opcode < 4)
    op(name, (0x3E | (opcode << 26), 6, _), srcs = 1)

# Listing of opcodes
funop("floor",     0b000000)
funop("srsqrt",    0b000001)
funop("dfdx",      0b000100, schedule_class = "coverage")
funop("dfdy",      0b000110, schedule_class = "coverage")
funop("rcp",       0b001000)
funop("rsqrt",     0b001001)
funop("sin_pt_1",  0b001010)
funop("log2",      0b001100)
funop("exp2",      0b001101)
funop("sin_pt_2",  0b001110)
op("sinc_agx3", None, srcs=1, is_float=True, agx3_mnemonic="f32sinc")
# Sources: quadrant bits, selected true float, selected false float.
op("bittest_select_agx3", None, srcs=3, imms=[IMM], is_float=True,
   agx3_mnemonic="isel")
funop("ceil",      0b010000)
funop("trunc",     0b100000)
funop("roundeven", 0b110000)

iunop("bitrev",    0b01)
iunop("popcount",  0b10)
iunop("ffs",       0b11)

op("fadd", encoding = (0x2A, 4, 6), srcs = 2, is_float = True)
op("ffma", encoding = (0x3A, 6, 8), srcs = 3, is_float = True)
op("fmul", encoding = (0x1A, 4, 6), srcs = 2, is_float = True)
op("hadd", encoding = (0x26, 4, 6), srcs = 2, is_float = True)
op("hfma", encoding = (0x36, 6, 8), srcs = 3, is_float = True)
op("hmul", encoding = (0x16, 4, 6), srcs = 2, is_float = True)
op("matmad8_agx3", None, srcs = 3, imms = [MATRIX_TYPE], can_reorder = False)
op("matmad16_agx3", None, srcs = 3, imms = [MATRIX_TYPE], can_reorder = False)
# Apple's AGX3 backend calls the same semantic operation `ldimm`.  This is the
# first source-grounded AGX2 IR -> AGX3 mnemonic mapping: the native immediate
# field reproduces all 14 probe constants exactly on G15, G16 and G17, while
# the destination field follows the register-pressure ladder in 16-bit units.
op("mov_imm", encoding = (0x62, 6, 8), imms = [IMM],
   agx3_mnemonic = "ldimm")
op("iadd", encoding = (0x0E, 8, _), srcs = 2, imms = [SHIFT])
op("imad", encoding = (0x1E, 8, _), srcs = 3, imms = [SHIFT])
op("bfi", encoding = (0x2E, 8, _), srcs = 3, imms = [BFI_MASK])
op("bfeil", encoding = (0x2E | L, 8, _), srcs = 3, imms = [BFI_MASK])
op("extr", encoding = (0x2E | (0x1 << 26), 8, _), srcs = 3, imms = [BFI_MASK])
op("asr", encoding = (0x2E | L | (0x1 << 26), 8, _), srcs = 2)
op("icmpsel", encoding = (0x12, 8, 10), srcs = 4, imms = [ICOND])
op("fcmpsel", encoding = (0x02, 8, 10), srcs = 4, imms = [FCOND])

def subgroup_op(name, opc):
    exact      = 0b01101111 | L | (opc << 29)
    op(name, encoding = (exact, 6, _), srcs = 1, imms = [SIMD_OP])

subgroup_op("quad_reduce", 0x0)
subgroup_op("simd_reduce", 0x1)
subgroup_op("quad_prefix", 0x2)
subgroup_op("simd_prefix", 0x3)

for window, w_bit in [('quad_', 0), ('', 1)]:
    for s, shuffle in enumerate(['', '_xor', '_up', '_down']):
        op(f"{window}shuffle{shuffle}",
            encoding = (0b01101111 | (w_bit << 26) | (s << 38), 6, _),
            srcs = 2)

    # Pseudo-instruction ballotting a boolean
    op(f"{window}ballot", _, srcs = 1)

    for T, T_bit, cond in [('f', 0, FCOND), ('i', 1, ICOND)]:
        op(f"{T}cmp_{window}ballot",
           encoding = (0b0100010 | (T_bit << 4) | (w_bit << 48), 8, _),
           srcs = 2, imms = [cond, INVERT_COND])

# Pseudo-instructions for compares returning 1/0
op("icmp", _, srcs = 2, imms = [ICOND, INVERT_COND])
op("fcmp", _, srcs = 2, imms = [FCOND, INVERT_COND])

# sources are coordinates, LOD, texture bindless base (zero for texture state
# registers), texture, sampler, shadow/offset
op("texture_sample",
      encoding = (0x31, 8, 10), # XXX WRONG SIZE
      srcs = 6, imms = [DIM, LOD_MODE, MASK, SCOREBOARD, OFFSET, SHADOW,
                        QUERY_LOD, GATHER, SPARSE])

# Post-RA native publication into the sampler-private file. IMM names an
# immutable resource value; it is not a GPR destination. Sources are the
# coordinate pair and optional comparison reference.
op("publish_texture_agx3", None, dests = 0, srcs = 2, imms = [IMM],
   can_eliminate = False, can_reorder = False)

# Pre-RA publication has independent scalar x/y/reference inputs. A native
# coordinate COPY does not require the GPR inputs to form a contiguous vector.
op("publish_texture_scalar_agx3", None, dests = 0, srcs = 3, imms = [IMM],
   can_eliminate = False, can_reorder = False)

# Immutable constant-program input. IMM names the logical uniform half in
# the frozen loader interface, not a GPR or an already assigned TPS offset.
op("load_cprog_agx3", None, srcs = 0, imms = [IMM])
# Constant-program arithmetic with a native uniform-file destination (IMM,
# in half-register units). No temporary GPR result or following COPY.
for name, srcs in [("fadd", 2), ("fmul", 2), ("ffma", 3)]:
   op(f"uniform_{name}_agx3", None, dests = 0, srcs = srcs, imms = [IMM],
      is_float = True, can_eliminate = False)
for memory, can_reorder in [("texture", True), ("image", False)]:
    coherency = [COHERENT] if not can_reorder else []
    op(f"{memory}_load", encoding = (0x71, 8, 10), # XXX WRONG SIZE
       srcs = 6, imms = [DIM, LOD_MODE, MASK, SCOREBOARD, OFFSET, SPARSE] + coherency,
       can_reorder = can_reorder,
       schedule_class = "none" if can_reorder else "load")

# sources are base, index
op("device_load",
      encoding = (0x05, 6, 8),
      srcs = 2, imms = [FORMAT, MASK, SHIFT, SCOREBOARD, COHERENT], can_reorder = False,
      schedule_class = "load")

# sources are base (relative to workgroup memory), index
op("local_load",
      encoding = (0b1101001, 6, 8),
      srcs = 2, imms = [FORMAT, MASK], can_reorder = False,
      schedule_class = "load")

# sources are value, base, index
# TODO: Consider permitting the short form
op("device_store",
      encoding = (0x45 | (1 << 47), 8, _),
      dests = 0, srcs = 3, imms = [FORMAT, MASK, SHIFT, SCOREBOARD, COHERENT], can_eliminate = False,
      schedule_class = "store")

# sources are value, base, index
op("local_store",
      encoding = (0b0101001, 6, 8),
      dests = 0, srcs = 3, imms = [FORMAT, MASK],
      can_eliminate=False, schedule_class = "store")

# sources are value, index
# TODO: Consider permitting the short form
op("uniform_store",
      encoding = ((0b111 << 27) | 0b1000101 | (1 << 47), 8, _),
      dests = 0, srcs = 2, imms = [MASK], can_eliminate = False)

# sources are base, offset
for kind, bit in [("tex", 1 << 6), ("sampler", 0)]:
    op(f"{kind}_state_store", encoding = (0b10101101 | bit | (1 << 20), 8, _),
          dests = 0, srcs = 2, imms = [IMM, SCOREBOARD], can_eliminate = False,
          schedule_class = "store")

# sources are value, base, index
op("atomic",
      encoding = (0x15 | (1 << 26) | (1 << 31) | (5 << 44), 8, _),
      dests = 1, srcs = 3, imms = [ATOMIC_OPC, SCOREBOARD],
      can_eliminate = False, schedule_class = "atomic")

# XXX: stop hardcoding the long form
op("local_atomic",
      encoding = (0x19 | (1 << 15) | (1 << 36) | (1 << 47), 10, _),
      dests = 1, srcs = 3, imms = [ATOMIC_OPC], schedule_class = "atomic",
      can_eliminate = False)

op("wait", (0x38, 2, _), dests = 0,
      can_eliminate = False, imms = [SCOREBOARD], schedule_class = "invalid")

for (suffix, schedule_class) in [("", "none"), ("_coverage", "coverage"), ("_barrier", "barrier")]:
    op(f"get_sr{suffix}", (0x72, 4, _), dests = 1, imms = [SR],
       schedule_class = schedule_class, can_reorder = schedule_class == "none")

op("sample_mask", (0x7fc1, 6, _), dests = 0, srcs = 2,
    can_eliminate = False, schedule_class = "coverage")

# Sources: sample mask, combined depth/stencil
op("zs_emit", (0x41, 4, _), dests = 0, srcs = 2,
              can_eliminate = False, imms = [ZS], schedule_class = "coverage")

# AGX3's tile store carries a *width*, not a format: `lstcol` cannot normalize
# on the way out the way AGX2's st_tile could. A normalized 8-bit target is
# therefore packed first -- `fltnorm` takes two 32-bit floats and writes them,
# clamped and scaled, into the two halves of one 16-bit register, so a
# four-channel 8-bit store costs exactly two of them. Sources: the two
# channels; the second is null for an odd trailing channel. FORMAT selects
# unsigned or signed normalization.
op("fltnorm_agx3", _, dests = 1, srcs = 2, imms = [FORMAT], is_float = True,
   agx3_mnemonic = "fltnorm")

# The tilebuffer-read inverse: one 16-bit register of packed normalized lanes
# in, a 32-bit float pair out.
op("normflt_agx3", _, dests = 1, srcs = 1, imms = [FORMAT], is_float = True,
   agx3_mnemonic = "normflt")

# Sources: sample mask, explicit coords (if present)
op("ld_tile", (0x49, 8, _), dests = 1, srcs = 2,
        imms = [FORMAT, MASK, PIXEL_OFFSET, EXPLICIT_COORDS, COLOUR_TARGET], can_reorder = False,
        schedule_class = "coverage")

# Sources: value, sample mask, explicit coords (if present)
op("st_tile", (0x09, 8, _), dests = 0, srcs = 3,
      can_eliminate = False, imms = [FORMAT, MASK, PIXEL_OFFSET, EXPLICIT_COORDS, COLOUR_TARGET],
      schedule_class = "coverage")

for (name, exact) in [("any", 0xC000), ("none", 0xC020), ("none_after", 0xC020)]:
   op("jmp_exec_" + name, (exact, 6, _), dests = 0, srcs = 0,
         can_eliminate = False, schedule_class = "invalid", imms = [TARGET])

# TODO: model implicit r0l destinations
op("pop_exec", (0x52 | (0x3 << 9), 6, _),
      dests = 0, srcs = 0, can_eliminate = False, schedule_class = "invalid",
      imms = [NEST])

for is_float in [False, True]:
   mod_mask = 0 if is_float else (0x3 << 26) | (0x3 << 38)

   for (cf, cf_op) in [("if", 0), ("else", 1), ("while", 2)]:
      name = "{}_{}cmp".format(cf, "f" if is_float else "i")
      exact = 0x42 | (0x0 if is_float else 0x10) | (cf_op << 9)
      imms = [NEST, FCOND if is_float else ICOND, INVERT_COND, TARGET]

      op(name, (exact, 6, _), dests = 0, srcs = 2, can_eliminate = False,
            imms = imms, is_float = is_float,
            schedule_class = "preload" if cf == "else" else "invalid")

op("bitop", (0x7E, 6, _), srcs = 2, imms = [TRUTH_TABLE])
op("intl", (0x3E, 6, _), srcs = 2, imms = [])
op("convert", (0x3E | L, 6, _), srcs = 2, imms = [ROUND]) 

# Sources are the coeffient register and the sample index (if applicable)
op("iter", (0x21, 8, _), srcs = 2, imms = [CHANNELS, INTERPOLATION])

# AGX3's f32interp is a scalar linear iterator. Perspective correction is a
# software sequence around this operation, so retain the iterator's perspective
# selector independently instead of pretending AGX2's fused iterproj exists.
op("interp_agx3", None, srcs = 2,
   imms = [INTERPOLATION, PERSPECTIVE])

# G17P computes explicit interpolation coordinates in a 32-bit token consumed
# by one or more scalar f32interp instructions. For sample interpolation the
# input is the low 16 bits of the API sample index; for centroid interpolation
# it is the native 16-bit coverage word.
op("itrcoords_agx3", None, srcs = 1, imms = [INTERPOLATION])

# Native projective-only RHW producer. A distinct opcode prevents CSE from
# merging it with ordinary position.w, whose f32interp f54 must remain clear.
op("rhw_interp_agx3", None, srcs = 2, imms = [INTERPOLATION])

# G17P's native projective multiply carries the originating coefficient index.
# Keep that provenance through SSA/RA; an ordinary fmul cannot encode it.
op("persp_mul_agx3", None, srcs = 2, imms = [IMM], is_float = True)

# Projective RCP also carries the RHW coefficient; preserve it independently.
op("persp_rcp_agx3", None, srcs = 1, imms = [IMM], is_float = True)


# Sources are the coeffient register for the varying, the coefficient register
# for W, and the sample index (if applicable)
op("iterproj", (0x21, 8, _), srcs = 3, imms = [CHANNELS, INTERPOLATION])

op("ldcf", (0xA1, 8, _), srcs = 1, imms = [CHANNELS])
# Native mesh output allocation and two-word UVB descriptor publication.
# Keep their source dependencies explicit through scheduling and allocation.
op("allocate_uvb_m5", _, dests=0, srcs=2, can_eliminate=False,
   can_reorder=False, schedule_class="store", agx3_mnemonic="allocate_uvb")
op("store_uvb_m5", _, dests=0, srcs=1, imms=[IMM], can_eliminate=False,
   can_reorder=False, schedule_class="store", agx3_mnemonic="uvst")
op("fence_uvb_gen1", _, dests=0, srcs=0, imms=[IMM], can_eliminate=False,
   can_reorder=False, schedule_class="barrier")

op("store_object_childcount_m5", _, dests=0, srcs=1, can_eliminate=False,
   can_reorder=False, schedule_class="store", agx3_mnemonic="uvsw")

op("st_vary", None, dests = 0, srcs = 2, can_eliminate = False)
op("st_vary_mul_agx3", None, dests = 0, srcs = 3, can_eliminate = False)
op("st_vary_fma_agx3", None, dests = 0, srcs = 4, can_eliminate = False)
op("no_varyings", (0x80000051, 4, _), dests = 0, can_eliminate = False)
# Both AGX3 encodings end every hash-pinned shader region with the canonical
# four-byte `end`.  This maps the compiler's unconditional terminator only;
# conditional termination remains unmapped.
op("stop", (0x88, 2, _), dests = 0, can_eliminate = False,
   schedule_class = "invalid", agx3_mnemonic = "end")
op("trap", (0x08, 2, _), dests = 0, can_eliminate = False,
   schedule_class = "invalid")

# These are modelled as total barriers since they can guard global memory
# access too, and even need to be properly ordered with loads.
op("wait_pix", (0x48, 4, _), dests = 0, imms = [WRITEOUT],
   can_eliminate = False, schedule_class = "barrier")
op("signal_pix", (0x58, 4, _), dests = 0, imms = [WRITEOUT],
   can_eliminate = False, schedule_class = "barrier")

# Sources are the data vector, the coordinate vector, the LOD, the bindless
# table if present (zero for texture state registers), and texture index.
op("image_write", (0xF1 | (1 << 23), 6, 8), dests = 0, srcs = 5, imms
   = [DIM, COHERENT], can_eliminate = False, schedule_class = "store")

# Sources are the image base, image index, the offset within shared memory, and
# the coordinates (or just the layer if implicit).
# TODO: Do we need the short encoding?
op("block_image_store", (0xB1, 10, _), dests = 0, srcs = 4,
   imms = [FORMAT, DIM, EXPLICIT_COORDS], can_eliminate = False, schedule_class = "store")

# Barriers
# Apple's AGX3 backend calls the same field-free semantic operation `barrier`.
# Exact occurrence ladders ground both the six-byte AGX3b and AGX3c encodings;
# scheduling/packing still stays in the bounded AGX3 packer.
op("threadgroup_barrier", (0x0068, 2, _), dests = 0, srcs = 0,
   can_eliminate = False, schedule_class = "barrier",
   agx3_mnemonic = "barrier")

# Semantic AGX3 fence. IMM is enum agx3_barrier_scope; the exact scope may
# expand to one or two barrier instructions in the AGX3 packer. This pseudo is
# emitted only for AGX3 device keys, leaving the AGX2 barrier sequence intact.
op("scoped_barrier_agx3", _, dests = 0, srcs = 0, imms = [IMM],
   can_eliminate = False, schedule_class = "barrier",
   agx3_mnemonic = "barrier")

# Direct-absolute coverage-stack access (word vectors or narrow scalars). These internal pseudos are the
# bounded landing point for future ray-query lowering; no generic address-space
# operation is silently reinterpreted as SCS.
op("scs_load_agx3", _, dests = 1, srcs = 0, imms = [IMM, FORMAT, MASK],
   can_reorder = False, schedule_class = "load", agx3_mnemonic = "ldscs")
op("scs_store_agx3", _, dests = 0, srcs = 1, imms = [IMM, FORMAT, MASK],
   can_eliminate = False, can_reorder = False, schedule_class = "store", agx3_mnemonic = "stscs")

# Uniform-base plus scalar-GPR-index SCS access.  The source order keeps the
# address provenance explicit instead of recovering it from generic pointer
# arithmetic after the fact: load(base, index), store(value, base, index).
# The native address selector is explicit and independent of FORMAT; the
# backend refuses other address forms.
op("scs_load_uniform_indexed_agx3", _, dests = 1, srcs = 2,
   imms = [IMM, FORMAT, MASK, SCS_ADDRESS_SELECTOR, SCS_ZERO_EXTEND, SCS_C_LOAD_MARKER],
   can_reorder = False, schedule_class = "load", agx3_mnemonic = "ldscs")
op("scs_store_uniform_indexed_agx3", _, dests = 0, srcs = 3,
   imms = [IMM, FORMAT, MASK, SCS_ADDRESS_SELECTOR], can_eliminate = False, can_reorder = False, schedule_class = "store",
   agx3_mnemonic = "stscs")

# Explicit two-GPR M5 SCS addressing, including the instanced traversal form.
# Source order matches the uniform-indexed pseudos: (base, index) and
# (value, base, index). The native address selector is preserved explicitly.
op("scs_load_register_indexed_agx3", _, dests = 1, srcs = 2,
   imms = [IMM, FORMAT, MASK, SCS_ADDRESS_SELECTOR, SCS_C_LOAD_MARKER, SCS_RT_COMPLETION],
   can_reorder = False, schedule_class = "load", agx3_mnemonic = "ldscs")
op("scs_store_register_indexed_agx3", _, dests = 0, srcs = 3,
   imms = [IMM, FORMAT, MASK, SCS_ADDRESS_SELECTOR],
   can_eliminate = False, can_reorder = False, schedule_class = "store", agx3_mnemonic = "stscs")

# Register-base form has no index operand. Its special status load cannot be
# represented as a normal byte load: both native control fields are explicit.
op("scs_load_register_base_agx3", _, dests = 1, srcs = 1,
   imms = [IMM, FORMAT, MASK, SCS_ADDRESS_SELECTOR, SCS_C_LOAD_MARKER,
           SCS_C_STATUS_LOAD],
   can_reorder = False, schedule_class = "load", agx3_mnemonic = "ldscs")
op("scs_store_register_base_agx3", _, dests = 0, srcs = 2,
   imms = [IMM, FORMAT, MASK, SCS_ADDRESS_SELECTOR],
   can_eliminate = False, can_reorder = False, schedule_class = "store",
   agx3_mnemonic = "stscs")


op("intersect_ray_gen1", _, dests=0, srcs=4, imms=[IMM],
   can_eliminate=False, can_reorder=False, schedule_class="barrier",
   agx3_mnemonic="intersect_ray")
op("continue_ray_gen1", _, dests=0, srcs=1, imms=[IMM],
   can_eliminate=False, can_reorder=False, schedule_class="barrier",
   agx3_mnemonic="continue_ray")
# M5 native traversal entry. Two native operands are 64-bit; the BVH operand
# may remain uniform. Three explicit 32-bit launch words are published to
# staging halves 0/2/4 immediately before the instruction. Completion is
# observed by the SCS polling protocol, not a fabricated GPR result.
op("intersect_ray_m5", _, dests = 0, srcs = 5,
   can_eliminate = False, can_reorder = False, schedule_class = "barrier",
   agx3_mnemonic = "intersect_ray")
op("enter_query_m5", _, dests = 1, srcs = 0, imms = [IMM],
   can_eliminate = False, can_reorder = False, schedule_class = "barrier",
   agx3_mnemonic = "incStk")
op("continue_ray_m5", _, dests = 0, srcs = 1,
   can_eliminate = False, can_reorder = False, schedule_class = "barrier",
   agx3_mnemonic = "continue_ray")

def memory_barrier(name, a, b, c):
    op(name, (0xF5 | (a << 10) | (b << 8) | (c << 12), 2, _), dests = 0, srcs = 0,
       can_eliminate = False, schedule_class = "barrier")

memory_barrier("memory_barrier", 1, 2, 9)

# TODO: Not clear what these individually are. Some might be cache flushes?
memory_barrier("image_barrier_1", 2, 2, 10)
memory_barrier("image_barrier_2", 3, 2, 10)
memory_barrier("image_barrier_3", 2, 1, 10)
memory_barrier("image_barrier_4", 3, 1, 10)

memory_barrier("flush_memory_to_texture", 0, 0, 4)

memory_barrier("memory_barrier_2", 2, 2, 9)
memory_barrier("memory_barrier_3", 2, 1, 9)
memory_barrier("unknown_barrier_1", 0, 3, 3)
memory_barrier("unknown_barrier_2", 0, 3, 0)

# Seen with device-scope memory barriers. Again not clear what's what.
memory_barrier("device_barrier_1", 3, 1, 9)
memory_barrier("device_barrier_2", 3, 2, 9)

op("doorbell", (0x60020 | 0x28 << 32, 6, _), dests = 0,
      can_eliminate = False, can_reorder = False, imms = [IMM])

op("stack_unmap", (0x00075, 8, _), dests = 1, srcs = 0, can_eliminate = False, can_reorder = False, imms = [IMM])
op("stack_map",   (0x10075, 8, _), dests = 0, srcs = 1, can_eliminate = False, can_reorder = False, imms = [IMM])

op("stack_adjust", encoding = (0x10100b5, 8, _),
      dests = 0, srcs = 0, can_eliminate = False, can_reorder = False,
      imms = [STACK_SIZE], schedule_class = "store")

# source is offset
op("stack_load", encoding = (0x35, 6, 8),
      srcs = 1, imms = [FORMAT, MASK, SCOREBOARD], can_reorder = False,
      schedule_class = "load")

# sources are value and offset
op("stack_store", encoding = (0xb5, 6, 8),
      dests = 0, srcs = 2, imms = [FORMAT, MASK, SCOREBOARD],
      can_eliminate=False, schedule_class = "store")

# Convenient aliases.
op("mov", _, srcs = 1)
op("not", _, srcs = 1)
op("signext", _, srcs = 1)

op("collect", _, srcs = VARIABLE)
op("split", _, srcs = 1, dests = VARIABLE)
op("phi", _, srcs = VARIABLE, schedule_class = "preload")

# The srcs double as destinations. Only deals in registers. This is generated by
# parallel copy lowering and lowered soon after. We need this as a dedicated
# instruction only for RA validation.
op("swap", _, dests = 0, srcs = 2)

op("unit_test", _, dests = 0, srcs = 1, can_eliminate = False)

# Like mov, but takes a register and can only appear at the start. Guaranteed
# to be coalesced during RA, rather than lowered to a real move. 
op("preload", _, srcs = 1, schedule_class = "preload")

# Opposite of preload. Exports a scalar value to a particular register at the
# end of the shader part. Must only appear after the logical end of the exit
# block, this avoids special casing the source's liveness. Logically all exports
# happen in parallel at the end of the shader part.
op("export", _, dests = 0, srcs = 1, imms = [IMM], can_eliminate = False,
   schedule_class = "invalid")

# Pseudo-instructions to set the nesting counter. Lowers to r0l writes after RA.
op("begin_cf", _, dests = 0, can_eliminate = False)
op("break", _, dests = 0, imms = [NEST, TARGET], can_eliminate = False,
   schedule_class = "invalid")

for (name, is_float) in [("break_if_icmp", False), ("break_if_fcmp", True)]:
    op(name, _, dests = 0, srcs = 2,
       imms = [NEST, INVERT_COND, FCOND if is_float else ICOND, TARGET],
       can_eliminate = False, schedule_class = "invalid")

# Native normalized RGB10A2 conversion, four floats to/from one packed word.
op("pack_rgb10a2_agx3", _, dests=1, srcs=4, is_float=True, agx3_mnemonic="pack")
op("unpack_rgb10a2_agx3", _, dests=1, srcs=1, agx3_mnemonic="unpack")
# G17G packs reusable half-float LOD/bias values into normal register words.
# Pure SSA producers can be CSE'd before each sample publishes its own COPY.
op("pack_sampler_lod_agx3", _, dests=1, srcs=1, agx3_mnemonic="pack_smpir")
# G17P's native coverage word is prepared before RA, from SR2 at one sample.
# Keep the general sample-mask target explicit while selecting its native zero
# no-op.
op("get_coverage_1x_agx3", _, dests = 1, schedule_class = "coverage")
op("sample_mask_agx3", _, dests = 0, srcs = 2, imms = [IMM],
   can_eliminate = False, schedule_class = "coverage")
# G17P depthf consumes the coverage dword, carrying any stencil reference in
# its upper half, and full-width float32 depth when written. The tuple is
# assembled before RA and published at staging halves 0 and 2.
op("zs_emit_agx3", _, dests = 0, srcs = 1, imms = [ZS],
   can_eliminate = False, schedule_class = "coverage")

# G17P destination-less SCS load of MASK 32-bit words at (base, index) or
# (base). It writes no register and takes no scoreboard slot, so it has no
# destination to allocate and nothing can wait on it. The low half of the
# index register is read and scaled by the address selector. Ordered like a
# barrier: it stays where the lowering put it.
op("scs_touch_register_indexed_agx3", _, dests = 0, srcs = 2,
   imms = [IMM, MASK, SCS_ADDRESS_SELECTOR],
   can_eliminate = False, can_reorder = False, schedule_class = "barrier",
   agx3_mnemonic = "ldscs")
op("scs_touch_register_base_agx3", _, dests = 0, srcs = 1,
   imms = [IMM, MASK],
   can_eliminate = False, can_reorder = False, schedule_class = "barrier",
   agx3_mnemonic = "ldscs")

# The G17P traversal barrier on its own. IMM is enum agx3_rt_barrier_scope.
op("rt_barrier_gen1", _, dests = 0, srcs = 0, imms = [IMM],
   can_eliminate = False, can_reorder = False, schedule_class = "barrier",
   agx3_mnemonic = "barrier")
