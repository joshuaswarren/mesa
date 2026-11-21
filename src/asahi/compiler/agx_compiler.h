/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/isa/agx_minifloat.h"
#include "compiler/nir/nir.h"
#include "util/bitset.h"
#include "util/half_float.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_worklist.h"
#include "agx_compile.h"
#include "agx_opcodes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* r0-r127 inclusive, as pairs of 16-bits, gives 256 registers */
#define AGX_NUM_REGS (256)


/* G17G/S native allocation ends at r125; r126/r127 are not GPRs. */
#define AGX3C_NUM_REGS (252)

/* G18G's ordinary register file ends at r125 (half 251). */
#define AGX3D_NUM_REGS (252)

/* u0-u255 inclusive, as pairs of 16-bits */
#define AGX_NUM_UNIFORMS (512)

/* Where an AGX3 fragment shader's uniforms must end. A spilling epilog places
 * its eMRT heap pointer after them, and the spill store's heap operand names
 * uniform halves up to 252 (agx3_tex_plain_uniform64). */
#define AGX3_FRAGMENT_UNIFORM_END (252)

/* Semi-arbitrary limit for spill slot allocation */
#define AGX_NUM_MODELED_REGS_LOG2 (11)
#define AGX_NUM_MODELED_REGS      (1 << AGX_NUM_MODELED_REGS_LOG2)

/* Limit on number of sources for non-phi instructions */
#define AGX_MAX_NORMAL_SOURCES (16)

enum agx_index_type {
   AGX_INDEX_NULL = 0,
   AGX_INDEX_NORMAL = 1,
   AGX_INDEX_IMMEDIATE = 2,
   AGX_INDEX_UNIFORM = 3,
   AGX_INDEX_REGISTER = 4,
   AGX_INDEX_UNDEF = 5,
   /* Post-RA value in the sampler-private file, independent of GPRs. */
   AGX_INDEX_RESOURCE = 6,
};

enum agx_size { AGX_SIZE_16 = 0, AGX_SIZE_32 = 1, AGX_SIZE_64 = 2 };

static inline unsigned
agx_size_align_16(enum agx_size size)
{
   switch (size) {
   case AGX_SIZE_16:
      return 1;
   case AGX_SIZE_32:
      return 2;
   case AGX_SIZE_64:
      return 4;
   }

   UNREACHABLE("Invalid size");
}

/* Keep synced with hash_index */
typedef struct {
   /* Sufficient for as many SSA values, immediates, and uniforms as we need. */
   uint32_t value;

   /* Indicates that this source kills the referenced value (because it is the
    * last use in a block and the source is not live after the block). Set by
    * liveness analysis.
    */
   bool kill : 1;

   /* Cache hints */
   bool cache   : 1;
   bool discard : 1;

   /* src - float modifiers */
   bool abs : 1;
   bool neg : 1;

   /* Register class */
   bool memory : 1;

   unsigned channels_m1     : 3;
   enum agx_size size       : 2;
   enum agx_index_type type : 3;

   /* If has_reg is set (during register allocation), the register assigned to
    * this SSA value This is used with NORMAL. Contrast REGISTER which uses
    * value instead.
    *
    * TODO: Unify.
    */
   unsigned reg : AGX_NUM_MODELED_REGS_LOG2;
   bool has_reg     : 1;
   unsigned padding : 6;
} agx_index;
static_assert(sizeof(agx_index) == 8, "packed");

static inline unsigned
agx_channels(agx_index idx)
{
   return idx.channels_m1 + 1;
}

static inline unsigned
agx_index_size_16(agx_index idx)
{
   return agx_size_align_16(idx.size) * agx_channels(idx);
}

static inline agx_index
agx_get_vec_index(unsigned value, enum agx_size size, unsigned channels)
{
   return (agx_index){
      .value = value,
      .channels_m1 = channels - 1,
      .size = size,
      .type = AGX_INDEX_NORMAL,
   };
}

static inline agx_index
agx_get_index(unsigned value, enum agx_size size)
{
   return agx_get_vec_index(value, size, 1);
}

static inline agx_index
agx_immediate(uint32_t imm)
{
   assert(imm < (1 << 16) && "overflowed immediate");

   return (agx_index){
      .value = imm,
      .size = AGX_SIZE_16,
      .type = AGX_INDEX_IMMEDIATE,
   };
}

static inline agx_index
agx_immediate_f(float f)
{
   assert(agx_minifloat_exact(f));
   return agx_immediate(agx_minifloat_encode(f));
}

/* in half-words, specify r0h as 1, r1 as 2... */
static inline agx_index
agx_register(uint32_t imm, enum agx_size size)
{
   assert(imm < AGX_NUM_REGS);

   return (agx_index){
      .value = imm,
      .size = size,
      .type = AGX_INDEX_REGISTER,
   };
}

static inline agx_index
agx_memory_register(uint32_t imm, enum agx_size size)
{
   return (agx_index){
      .value = imm,
      .memory = true,
      .size = size,
      .type = AGX_INDEX_REGISTER,
   };
}

static inline agx_index
agx_register_like(uint32_t imm, agx_index like)
{
   return (agx_index){
      .value = imm,
      .memory = like.memory,
      .channels_m1 = like.channels_m1,
      .size = like.size,
      .type = AGX_INDEX_REGISTER,
   };
}

static inline agx_index
agx_as_register(agx_index x)
{
   assert(x.has_reg);
   return agx_register_like(x.reg, x);
}

static inline agx_index
agx_undef(enum agx_size size)
{
   return (agx_index){
      .size = size,
      .type = AGX_INDEX_UNDEF,
   };
}

/* Also in half-words */
static inline agx_index
agx_uniform(uint32_t imm, enum agx_size size)
{
   assert(imm < AGX_NUM_UNIFORMS);

   return (agx_index){
      .value = imm,
      .size = size,
      .type = AGX_INDEX_UNIFORM,
   };
}

static inline agx_index
agx_null()
{
   return (agx_index){.type = AGX_INDEX_NULL};
}

static inline agx_index
agx_zero()
{
   return agx_immediate(0);
}

/* IEEE 754 additive identity -0.0, stored as an 8-bit AGX minifloat: mantissa
 * = exponent = 0, sign bit set */

static inline agx_index
agx_negzero()
{
   return agx_immediate(0x80);
}

static inline agx_index
agx_abs(agx_index idx)
{
   idx.abs = true;
   idx.neg = false;
   return idx;
}

static inline agx_index
agx_neg(agx_index idx)
{
   idx.neg ^= true;
   return idx;
}

/* Replaces an index, preserving any modifiers */

static inline agx_index
agx_replace_index(agx_index old, agx_index replacement)
{
   replacement.abs = old.abs;
   replacement.neg = old.neg;
   return replacement;
}

static inline bool
agx_is_null(agx_index idx)
{
   return idx.type == AGX_INDEX_NULL;
}

/* Compares equivalence as references */

static inline bool
agx_is_equiv(agx_index left, agx_index right)
{
   return (left.type == right.type) && (left.value == right.value);
}

enum ra_class {
   /* General purpose register */
   RA_GPR,

   /* Memory, used to assign stack slots */
   RA_MEM,

   /* Keep last */
   RA_CLASSES,
};

static inline enum ra_class
ra_class_for_index(agx_index idx)
{
   return idx.memory ? RA_MEM : RA_GPR;
}

enum agx_icond {
   AGX_ICOND_UEQ = 0,
   AGX_ICOND_ULT = 1,
   AGX_ICOND_UGT = 2,
   /* unknown */
   AGX_ICOND_SEQ = 4,
   AGX_ICOND_SLT = 5,
   AGX_ICOND_SGT = 6,
   /* unknown */
};

enum agx_fcond {
   AGX_FCOND_EQ = 0,
   AGX_FCOND_LT = 1,
   AGX_FCOND_GT = 2,
   AGX_FCOND_LTN = 3,
   /* unknown */
   AGX_FCOND_GE = 5,
   AGX_FCOND_LE = 6,
   AGX_FCOND_GTN = 7,
};

enum agx_round {
   AGX_ROUND_RTZ = 0,
   AGX_ROUND_RTE = 1,
};

enum agx_convert {
   AGX_CONVERT_U8_TO_F = 0,
   AGX_CONVERT_S8_TO_F = 1,
   AGX_CONVERT_F_TO_U16 = 4,
   AGX_CONVERT_F_TO_S16 = 5,
   AGX_CONVERT_U16_TO_F = 6,
   AGX_CONVERT_S16_TO_F = 7,
   AGX_CONVERT_F_TO_U32 = 8,
   AGX_CONVERT_F_TO_S32 = 9,
   AGX_CONVERT_U32_TO_F = 10,
   AGX_CONVERT_S32_TO_F = 11
};

enum agx_lod_mode {
   AGX_LOD_MODE_AUTO_LOD = 0,
   AGX_LOD_MODE_AUTO_LOD_BIAS_UNIFORM = 1,
   AGX_LOD_MODE_LOD_MIN_UNIFORM = 2,
   AGX_LOD_MODE_AUTO_LOD_BIAS = 5,
   AGX_LOD_MODE_LOD_GRAD = 4,
   AGX_LOD_MODE_LOD_MIN = 6,
   AGX_LOD_MODE_AUTO_LOD_BIAS_MIN_UNIFORM = 9,
   AGX_LOD_MODE_LOD_GRAD_MIN = 12,
   AGX_LOD_MODE_AUTO_LOD_BIAS_MIN = 13,
};

/* Forward declare for branch target */
struct agx_block;

/* Keep synced with hash_instr */
typedef struct {
   /* Must be first */
   struct list_head link;

   /* The sources list. */
   agx_index *src;

   /* Data flow */
   agx_index *dest;

   enum agx_opcode op;

   uint8_t nr_dests;
   uint8_t nr_srcs;

   /* TODO: More efficient */
   union {
      enum agx_icond icond;
      enum agx_fcond fcond;
   };

   union {
      uint64_t imm;
      uint32_t writeout;
      uint32_t truth_table;
      uint32_t matrix_type;
      uint32_t component;
      uint32_t channels;
      uint32_t bfi_mask;
      uint16_t pixel_offset;
      uint16_t zs;
      int16_t stack_size;
      enum agx_sr sr;
      enum agx_round round;
      enum agx_atomic_opc atomic_opc;
      enum agx_lod_mode lod_mode;
      enum agx_simd_op simd_op;
      struct agx_block *target;

      /* As a special case to workaround ordering issues when translating phis,
       * if nr_srcs == 0 and the opcode is PHI, points to the NIR phi.
       */
      nir_phi_instr *phi;
   };

   /* For local access */
   enum agx_format format;

   /* Number of nested control flow layers to jump by. TODO: Optimize */
   uint32_t nest;

   /* Invert icond/fcond */
   bool invert_cond : 1;

   /*
    * G17P carries a texture coordinate offset as an immediate field in the
    * sample instruction (f43), while the depth comparator is a staged
    * register. agx_emit_tex() cannot hand the packer both through src[5],
    * which is one operand, so the offset rides here instead.
    *
    * Signed x/y/z nibbles, exactly as agx_nir_lower_texture()
    * packs nir_tex_src_offset. Only meaningful when `offset` is set.
    *
    * NOTE: agx_opt_cse.c's instrs_equal() is hand-written and enumerates
    * fields one at a time. This field is compared there. Anything added
    * beside it must be too, or two samples differing only in their offset
    * will CSE into one.
    */
   uint16_t offset_xyz;

   /* TODO: Handle tex ops more efficient */
   enum agx_dim dim       : 4;
   bool offset            : 1;
   bool shadow            : 1;
   bool query_lod         : 1;
   bool sparse            : 1;
   enum agx_gather gather : 3;

   /* TODO: Handle tilebuffer ops more efficient */
   bool explicit_coords : 1;

   /* Logical attachment identity, independent of the physical pixel_offset. */
   uint8_t colour_target;

   /* TODO: Handle iter ops more efficient */
   enum agx_interpolation interpolation : 2;
   bool perspective                    : 1;

   /* TODO: Handle loads more efficiently */
   bool coherent : 1;
   /* M5 image load from memory no store of this shader can write (NIR
    * ACCESS_CAN_REORDER and ACCESS_RESTRICT, not coherent): the M5 copy
    * schedule may move it across memory stores. */
   bool memory_can_reorder : 1;
   /* AGX3 memory index scale excludes the access-format size. */
   bool memory_index_is_bytes : 1;
   /* Raw subword memory elements use packed payloads (device loads use
    * 32-bit SSA words; byte stores and shared loads use 16-bit halves). */
   bool memory_is_packed : 1;
   /* AGX3C device memory: signed byte offset added after scaling the index. */
   int16_t memory_byte_offset;

   /* Explicit AGX3 SCS load semantics.  These stay on the internal SCS pseudo
    * and are never inferred from generic memory IR. */
   bool scs_zero_extend : 1;
   bool scs_c_load_marker : 1;
   bool scs_c_status_load : 1;
   bool scs_rt_wait : 1;
   bool scs_rt_completion : 1;
   uint8_t scs_address_selector;

   /* Final st_vary op */
   bool last : 1;

   /* Shift for a bitwise or memory op (conflicts with format for memory ops) */
   unsigned shift : 4;

   /* Scoreboard index, 0 or 1. Leave as 0 for instructions that do not require
    * scoreboarding (everything but memory load/store and texturing). */
   unsigned scoreboard : 1;

   /* NIR permits either sign for a zero result of this floating add. */
   bool no_signed_zero : 1;

   /* M5 FCMPSEL: a selected immediate is a minifloat (native fself32), not
    * raw integer bits. */
   bool fcmpsel_minifloat_select : 1;

   /* Output modifiers */
   bool saturate : 1;
   unsigned mask : 4;

   /* G17P implicit fragment samples do not copy their coordinates into the
    * texture staging file after the arithmetic is complete.  Apple's compiler
    * writes the final scalar operations directly to staging so helper lanes
    * survive an odd-aligned hardware scissor.  The pre-RA sink pass marks the
    * two scalar producers and their consuming sample here; the packer assigns
    * their shared scoreboard slot after physical register allocation. */
   bool agx3_tex_coord_stage : 1;
   bool agx3_tex_coord_component : 1;
   bool agx3_tex_coord_start : 1;
   bool agx3_tex_coords_staged : 1;
   bool agx3_cube_scalar_coords  : 1;

   /* M5 texel fetch or 2D sample with independent scalar coordinates:
    * src[0] is x, src[6] is y and src[7] is the sample index of a
    * multisampled read or the layer of an array read (null otherwise). The
    * staging COPYs read each component where it lives, so no contiguous
    * coordinate vector has to be collected for the read. */
   bool agx3_split_coords : 1;

} agx_instr;

static inline uint32_t
agx_texture_resource(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->nr_srcs &&
          I->src[0].type == AGX_INDEX_RESOURCE ? I->src[0].value : 0;
}

static inline bool
agx_is_texture_publication(const agx_instr *I)
{
   return I->op == AGX_OPCODE_PUBLISH_TEXTURE_AGX3 ||
          I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3;
}

static inline uint32_t
agx_publication_resource(const agx_instr *I)
{
   return agx_is_texture_publication(I) ? I->imm : 0;
}

static inline bool
agx_is_vertex_store(const agx_instr *I)
{
   return I->op == AGX_OPCODE_ST_VARY ||
          I->op == AGX_OPCODE_ST_VARY_MUL_AGX3 ||
          I->op == AGX_OPCODE_ST_VARY_FMA_AGX3;
}

static inline void
agx_replace_src(agx_instr *I, unsigned src_index, agx_index replacement)
{
   I->src[src_index] = agx_replace_index(I->src[src_index], replacement);
}

struct agx_block;

typedef struct agx_block {
   /* Link to next block. Must be first */
   struct list_head link;

   /* List of instructions emitted for the current block */
   struct list_head instructions;

   /* Index of the block in source order */
   unsigned index;

   /* Control flow graph */
   struct agx_block *successors[2];
   struct util_dynarray predecessors;
   bool unconditional_jumps;

   /* Could there be masked execution? */
   bool divergent;

   /* Some enclosing if or loop may diverge within the subgroup (NIR
    * divergence analysis on AGX3; equal to divergent on AGX2). Used only for
    * register-cache discard hints.
    */
   bool cf_divergent;

   /* Liveness analysis results */
   struct u_sparse_bitset live_in;
   struct u_sparse_bitset live_out;

   BITSET_DECLARE(reg_live_in, AGX_NUM_REGS);
   BITSET_DECLARE(reg_live_out, AGX_NUM_REGS);

   /* For visited blocks during register assignment and live-out registers, the
    * mapping of registers to SSA names at the end of the block. This is dense,
    * unlike its inverse.
    */
   uint32_t *reg_to_ssa_out[2];

   /* Is this block a loop header? If not, all of its predecessors precede it in
    * source order.
    */
   bool loop_header;

   /* Offsets of the block in the emitted binary.  On AGX3, loop_exit_offset
    * names the first CNDEND after a loop backedge, before any enclosing-scope
    * CNDENDs that happen to share the block. */
   off_t offset, last_offset, loop_exit_offset;

   /** Available for passes to use for metadata */
   uint8_t pass_flags;
} agx_block;

typedef struct {
   nir_shader *nir;
   mesa_shader_stage stage;
   struct util_dynarray collect_split_hints;
   bool is_preamble;

   /* G17G second attempt after a packer refusal: keep float constants in
    * registers instead of inlining them as minifloat immediates. The
    * register forms of the float ALU, compare-select and compare-branch are
    * grounded; many immediate shapes are not. */
   bool agx3_reg_float_consts;
   bool uniforms_written;
   /* The spiller reloaded a value by repeating its device load */
   bool spilled_by_load;
   unsigned scratch_size_B;
   /* The input mask/compact ranks are frozen before native import selection.
    * Later DCE may remove a read without changing this external interface. */
   bool agx3_cprog_inputs_lowered;

   /* AGX3b: thread-private storage half a returned device atomic publishes
    * its old value to, above the spill frame.
    */
   unsigned agx3_atomic_return_tps_half;

   struct list_head blocks; /* list of agx_block */
   struct agx_shader_info *out;
   struct agx_shader_key *key;

   /* Maximum block index */
   unsigned num_blocks;

   /* For creating temporaries */
   unsigned alloc;

   /* Does the shader statically use scratch memory? */
   bool any_scratch;

   /* Interlock waits guard memory, not the tilebuffer, so keep them. */
   bool any_interlock;

   /* Mask of pixel fences we've definitely already waited for. */
   uint16_t already_pixel_waited;

   /* G17P depth/stencil writes not yet emitted. Every sample's write shares
    * one class-1 transaction, released after the last. */
   unsigned zs_writes_left;

   /* The first coverage update in an explicit-early fragment is the one
    * per-pixel Z/S test.  It has a distinct G17P acquire/retain protocol. */
   bool force_early_tests_pending;

   /* Has r0l been zeroed yet due to control flow? */
   bool any_cf;

   /* Do we need a reserved zero register throughout the program to handle
    * quad-divergent shuffle?
    */
   bool any_quad_divergent_shuffle;

   /* Number of nested control flow structures within the innermost loop. Since
    * NIR is just loop and if-else, this is the number of nested if-else
    * statements in the loop */
   unsigned loop_nesting;

   /* Total nesting across all loops, to determine if we need push_exec */
   unsigned total_nesting;

   /* Enclosing ifs/loops that may diverge; see agx_block::cf_divergent. */
   unsigned divergent_nesting;

   /* During instruction selection, for inserting control flow */
   agx_block *current_block;
   agx_block *break_block;
   agx_block *after_block;
   agx_block **indexed_nir_blocks;

   /* During instruction selection, map from vector agx_index to its scalar
    * components, populated by a split. */
   struct hash_table_u64 *allocated_vec;

   /* During instruction selection, preloaded values or NULL if it hasn't been
    * preloaded.
    */
   agx_index preloaded[AGX_NUM_REGS];

   /* AGX3B pixel-centre 1/W, computed once in the start block. */
   agx_index agx3_center_rcp_w;

   /* AGX_G17P_SR32_PIXEL_COORD: the 32-bit native pixel-coordinate special
    * registers read by load_pixel_coord and the 16-bit halves NIR sees, so a
    * zero extension of a half can reuse the register (Apple reads SR80/81 as
    * zero-extended 32-bit values: native fragment 0x40/0x48 feed 32-bit XORs
    * directly). Null unless the pixel coordinate was loaded. */
   agx_index agx3_pixel_coord_native[2];
   agx_index agx3_pixel_coord_half[2];

   /* AGX_G17P_SHARED_RHW: the centre-interpolated projective 1/W computed in
    * the entry block by the first perspective input load, reused by later
    * loads instead of a fresh RHW f32interp + f32rcp pair. Null until then. */

   /* Beginning of our stack allocation used for spilling, below that is
    * NIR-level scratch.
    */
   unsigned spill_base_B;

   /* Beginning of stack allocation used for parallel copy lowering */
   bool has_spill_pcopy_reserved;

   /* M5 CEM owns halves 8..15, beyond CF/shuffle/parallel-copy scratch. */
   bool agx3_cem_low_scratch;

   /* The remat-only spiller had to spill a value to scratch; register
    * allocation must finish with real spilling under the full budget. */
   bool remat_fallback_spilled;
   unsigned spill_pcopy_base;

   /* Stats for shader-db */
   unsigned loop_count;
   unsigned max_reg;

   /* Half of the AGX3 compiler staging footprint. Unlike max_reg, this
    * describes the separate file consumed by UVSW and texture inputs. */
   unsigned agx3_staging_halfs;
   bool agx3_staging_known;
   bool agx3_fltnorm_start;
   bool agx3_mrt_tile_ops_ordered;
   /* The packer emitted this fragment's tile accesses through the native1130
    * mixed-format protocol, inside its 0x200 entry and final release (run
    * 1139); its reads then take the long LLDCol form. */
   bool agx3_g18_mixed_protocol;
   struct agx3_colour_link_info agx3_colour_link;

   /* Byte-packed struct agx3_uvs_reloc entries the packer records for a
    * relocatable vertex program (agx_shader_key::g17p_uvs_reloc_size), their
    * offsets from the start of the emission, and how many it recorded. */
   struct util_dynarray agx3_uvs_relocs;
   unsigned agx3_uvs_reloc_count;

   /* Promoted constants. These will be appended to the binary at the end. */
   uint16_t rodata[512];
   /* An intrinsic or ALU op this backend does not implement reached
    * instruction selection; the compile is refused right after emission. */
   bool unsupported_instruction;
   /* Register allocation found no legal placement for a constrained value
    * (agx_register_allocate.c); the compile is refused after agx_ra(). */
   bool ra_failed;
   /* Halves taken off the spill budget after a failed allocation, so the
    * live-range shuffle gets free room (agx_compile_function_nir retries). */
   unsigned ra_budget_cut;
   /* Set by agx_ra_impl() while it assigns registers: a failed allocation
    * that has no legal region left at all leaves RA through it (a
    * jmp_buf *), see find_best_region_to_evict(). */
   void *ra_abort;
} agx_context;

static inline void
agx_remove_instruction(agx_instr *ins)
{
   list_del(&ins->link);
}

static inline agx_index
agx_vec_temp(agx_context *ctx, enum agx_size size, unsigned channels)
{
   return agx_get_vec_index(ctx->alloc++, size, channels);
}

static inline agx_index
agx_temp(agx_context *ctx, enum agx_size size)
{
   return agx_get_index(ctx->alloc++, size);
}

static inline agx_index
agx_temp_like(agx_context *ctx, agx_index idx)
{
   idx.value = ctx->alloc++;
   return idx;
}

static enum agx_size
agx_size_for_bits(unsigned bits)
{
   switch (bits) {
   case 1:
   case 8:
   case 16:
      return AGX_SIZE_16;
   case 32:
      return AGX_SIZE_32;
   case 64:
      return AGX_SIZE_64;
   default:
      UNREACHABLE("Invalid bitsize");
   }
}

static inline agx_index
agx_def_index(nir_def *ssa)
{
   return agx_get_vec_index(ssa->index, agx_size_for_bits(ssa->bit_size),
                            ssa->num_components);
}

static inline agx_index
agx_src_index(nir_src *src)
{
   return agx_def_index(src->ssa);
}

static inline agx_index
agx_vec_for_def(agx_context *ctx, nir_def *def)
{
   return agx_vec_temp(ctx, agx_size_for_bits(def->bit_size),
                       def->num_components);
}

static inline agx_index
agx_vec_for_intr(agx_context *ctx, nir_intrinsic_instr *instr)
{
   return agx_vec_for_def(ctx, &instr->def);
}

static inline unsigned
agx_num_predecessors(agx_block *block)
{
   return util_dynarray_num_elements(&block->predecessors, agx_block *);
}

static inline unsigned
agx_num_successors(agx_block *block)
{
   STATIC_ASSERT(ARRAY_SIZE(block->successors) == 2);
   return (block->successors[0] ? 1 : 0) + (block->successors[1] ? 1 : 0);
}

static inline agx_block *
agx_start_block(agx_context *ctx)
{
   agx_block *first = list_first_entry(&ctx->blocks, agx_block, link);
   assert(agx_num_predecessors(first) == 0);
   return first;
}

static inline agx_block *
agx_end_block(agx_context *ctx)
{
   agx_block *last = list_last_entry(&ctx->blocks, agx_block, link);
   assert(agx_num_successors(last) == 0);
   return last;
}

void agx_block_add_successor(agx_block *block, agx_block *successor);

/* Iterators for AGX IR */

#define agx_foreach_block(ctx, v)                                              \
   list_for_each_entry(agx_block, v, &ctx->blocks, link)

#define agx_foreach_block_safe(ctx, v)                                         \
   list_for_each_entry_safe(agx_block, v, &ctx->blocks, link)

#define agx_foreach_block_rev(ctx, v)                                          \
   list_for_each_entry_rev(agx_block, v, &ctx->blocks, link)

#define agx_foreach_block_from(ctx, from, v)                                   \
   list_for_each_entry_from(agx_block, v, from, &ctx->blocks, link)

#define agx_foreach_block_from_rev(ctx, from, v)                               \
   list_for_each_entry_from_rev(agx_block, v, from, &ctx->blocks, link)

#define agx_foreach_instr_in_block(block, v)                                   \
   list_for_each_entry(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_rev(block, v)                               \
   list_for_each_entry_rev(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_safe(block, v)                              \
   list_for_each_entry_safe(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_safe_rev(block, v)                          \
   list_for_each_entry_safe_rev(agx_instr, v, &(block)->instructions, link)

#define agx_foreach_instr_in_block_from(block, v, from)                        \
   list_for_each_entry_from(agx_instr, v, from, &(block)->instructions, link)

#define agx_foreach_instr_in_block_from_rev(block, v, from)                    \
   list_for_each_entry_from_rev(agx_instr, v, from, &(block)->instructions,    \
                                link)

#define agx_foreach_instr_global(ctx, v)                                       \
   agx_foreach_block(ctx, v_block)                                             \
      agx_foreach_instr_in_block(v_block, v)

#define agx_foreach_instr_global_rev(ctx, v)                                   \
   agx_foreach_block_rev(ctx, v_block)                                         \
      agx_foreach_instr_in_block_rev(v_block, v)

#define agx_foreach_instr_global_safe(ctx, v)                                  \
   agx_foreach_block(ctx, v_block)                                             \
      agx_foreach_instr_in_block_safe(v_block, v)

#define agx_foreach_instr_global_safe_rev(ctx, v)                              \
   agx_foreach_block_rev(ctx, v_block)                                         \
      agx_foreach_instr_in_block_safe_rev(v_block, v)

/* Based on set_foreach, expanded with automatic type casts */

#define agx_foreach_successor(blk, v)                                          \
   agx_block *v;                                                               \
   agx_block **_v;                                                             \
   for (_v = (agx_block **)&blk->successors[0], v = *_v;                       \
        v != NULL && _v < (agx_block **)&blk->successors[2]; _v++, v = *_v)

#define agx_foreach_predecessor(blk, v)                                        \
   util_dynarray_foreach(&blk->predecessors, agx_block *, v)

#define agx_foreach_src(ins, v) for (unsigned v = 0; v < ins->nr_srcs; ++v)

#define agx_foreach_src_rev(ins, v)                                            \
   for (signed v = ins->nr_srcs - 1; v >= 0; --v)

#define agx_foreach_dest(ins, v) for (unsigned v = 0; v < ins->nr_dests; ++v)

#define agx_foreach_dest_rev(ins, v)                                           \
   for (signed v = ins->nr_dests - 1; v >= 0; --v)

#define agx_foreach_ssa_src(ins, v)                                            \
   agx_foreach_src(ins, v)                                                     \
      if (ins->src[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_ssa_src_rev(ins, v)                                        \
   agx_foreach_src_rev(ins, v)                                                 \
      if (ins->src[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_reg_src(ins, v)                                            \
   agx_foreach_src(ins, v)                                                     \
      if (ins->src[v].type == AGX_INDEX_REGISTER)

#define agx_foreach_ssa_dest(ins, v)                                           \
   agx_foreach_dest(ins, v)                                                    \
      if (ins->dest[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_ssa_dest_rev(ins, v)                                       \
   agx_foreach_dest_rev(ins, v)                                                \
      if (ins->dest[v].type == AGX_INDEX_NORMAL)

#define agx_foreach_reg_dest(ins, v)                                           \
   agx_foreach_dest(ins, v)                                                    \
      if (ins->dest[v].type == AGX_INDEX_REGISTER)

/* Phis only come at the start (after else instructions) so we stop as soon as
 * we hit a non-phi
 */
#define agx_foreach_phi_in_block(block, v)                                     \
   agx_foreach_instr_in_block(block, v)                                        \
      if (v->op == AGX_OPCODE_ELSE_ICMP || v->op == AGX_OPCODE_ELSE_FCMP)      \
         continue;                                                             \
      else if (v->op != AGX_OPCODE_PHI)                                        \
         break;                                                                \
      else

#define agx_foreach_phi_in_block_safe(block, v)                                \
   agx_foreach_instr_in_block_safe(block, v)                                   \
      if (v->op == AGX_OPCODE_ELSE_ICMP || v->op == AGX_OPCODE_ELSE_FCMP)      \
         continue;                                                             \
      else if (v->op != AGX_OPCODE_PHI)                                        \
         break;                                                                \
      else

/*
 * Find the index of a predecessor, used as the implicit order of phi sources.
 */
static inline unsigned
agx_predecessor_index(agx_block *succ, agx_block *pred)
{
   unsigned index = 0;

   agx_foreach_predecessor(succ, x) {
      if (*x == pred)
         return index;

      index++;
   }

   UNREACHABLE("Invalid predecessor");
}

static inline agx_block *
agx_prev_block(agx_block *ins)
{
   return list_last_entry(&(ins->link), agx_block, link);
}

static inline agx_instr *
agx_prev_op(agx_instr *ins)
{
   return list_last_entry(&(ins->link), agx_instr, link);
}

static inline agx_instr *
agx_first_instr(agx_block *block)
{
   if (list_is_empty(&block->instructions))
      return NULL;
   else
      return list_first_entry(&block->instructions, agx_instr, link);
}

static inline agx_instr *
agx_last_instr(agx_block *block)
{
   if (list_is_empty(&block->instructions))
      return NULL;
   else
      return list_last_entry(&block->instructions, agx_instr, link);
}

static inline agx_instr *
agx_next_op(agx_instr *ins)
{
   return list_first_entry(&(ins->link), agx_instr, link);
}

static inline agx_block *
agx_next_block(agx_block *block)
{
   return list_first_entry(&(block->link), agx_block, link);
}

static inline agx_block *
agx_exit_block(agx_context *ctx)
{
   agx_block *last = list_last_entry(&ctx->blocks, agx_block, link);
   assert(!last->successors[0] && !last->successors[1]);
   return last;
}

#define agx_worklist_init(ctx, w)        u_worklist_init(w, ctx->num_blocks, ctx)
#define agx_worklist_push_head(w, block) u_worklist_push_head(w, block, index)
#define agx_worklist_push_tail(w, block) u_worklist_push_tail(w, block, index)
#define agx_worklist_peek_head(w)        u_worklist_peek_head(w, agx_block, index)
#define agx_worklist_pop_head(w)         u_worklist_pop_head(w, agx_block, index)
#define agx_worklist_peek_tail(w)        u_worklist_peek_tail(w, agx_block, index)
#define agx_worklist_pop_tail(w)         u_worklist_pop_tail(w, agx_block, index)

/* Like in NIR, for use with the builder */

enum agx_cursor_option {
   agx_cursor_after_block,
   agx_cursor_before_instr,
   agx_cursor_after_instr
};

/* The stages whose AGX3 samples take an explicit level the vertex way: the
 * packer's vertex_explicit_lod (agx3_pack_binary.c), which also lets an array
 * comparison keep an immediate level (agx_optimizer_inline_imm). A G17P mesh
 * program is VDM-launched like a vertex program. */
static inline bool
agx3_vertex_explicit_lod_stage(mesa_shader_stage stage,
                               enum agx_isa_encoding enc)
{
   return stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY ||
          stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_TESS_EVAL ||
          (enc == AGX_ISA_ENCODING_AGX3B && stage == MESA_SHADER_MESH);
}

typedef struct {
   enum agx_cursor_option option;

   union {
      agx_block *block;
      agx_instr *instr;
   };
} agx_cursor;

static inline bool
agx_cursors_equal(agx_cursor a, agx_cursor b)
{
   if (a.option != b.option)
      return false;

   if (a.option == agx_cursor_after_block)
      return a.block == b.block;
   else
      return a.instr == b.instr;
}

static inline agx_cursor
agx_after_block(agx_block *block)
{
   return (agx_cursor){
      .option = agx_cursor_after_block,
      .block = block,
   };
}

static inline agx_cursor
agx_before_instr(agx_instr *instr)
{
   return (agx_cursor){
      .option = agx_cursor_before_instr,
      .instr = instr,
   };
}

static inline agx_cursor
agx_after_instr(agx_instr *instr)
{
   return (agx_cursor){
      .option = agx_cursor_after_instr,
      .instr = instr,
   };
}

static inline agx_cursor
agx_before_nonempty_block(agx_block *block)
{
   agx_instr *I = list_first_entry(&block->instructions, agx_instr, link);
   assert(I != NULL);

   return agx_before_instr(I);
}

static inline agx_cursor
agx_before_block(agx_block *block)
{
   if (list_is_empty(&block->instructions))
      return agx_after_block(block);
   else
      return agx_before_nonempty_block(block);
}

static inline bool
instr_after_logical_end(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_JMP_EXEC_ANY:
   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_POP_EXEC:
   case AGX_OPCODE_BREAK:
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_IF_FCMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_STOP:
   case AGX_OPCODE_EXPORT:
      return true;
   default:
      return false;
   }
}

/*
 * Get a cursor inserting at the logical end of the block. In particular, this
 * is before branches or control flow instructions, which occur after the
 * logical end but before the physical end.
 */
static inline agx_cursor
agx_after_block_logical(agx_block *block)
{
   /* Search for the first instruction that's not past the logical end */
   agx_foreach_instr_in_block_rev(block, I) {
      if (!instr_after_logical_end(I))
         return agx_after_instr(I);
   }

   /* If we got here, the block is either empty or entirely control flow */
   return agx_before_block(block);
}

/* Get a cursor at the start of a function, after any preloads */
static inline agx_cursor
agx_before_function(agx_context *ctx)
{
   agx_block *block = agx_start_block(ctx);

   agx_foreach_instr_in_block(block, I) {
      if (I->op != AGX_OPCODE_PRELOAD)
         return agx_before_instr(I);
   }

   /* The whole block is preloads, so insert at the end */
   return agx_after_block(block);
}

/* IR builder in terms of cursor infrastructure */

typedef struct {
   agx_context *shader;
   agx_cursor cursor;
} agx_builder;

static inline agx_builder
agx_init_builder(agx_context *ctx, agx_cursor cursor)
{
   return (agx_builder){
      .shader = ctx,
      .cursor = cursor,
   };
}

/* Insert an instruction at the cursor and move the cursor */

static inline void
agx_builder_insert(agx_cursor *cursor, agx_instr *I)
{
   switch (cursor->option) {
   case agx_cursor_after_instr:
      list_add(&I->link, &cursor->instr->link);
      cursor->instr = I;
      return;

   case agx_cursor_after_block:
      list_addtail(&I->link, &cursor->block->instructions);
      cursor->option = agx_cursor_after_instr;
      cursor->instr = I;
      return;

   case agx_cursor_before_instr:
      list_addtail(&I->link, &cursor->instr->link);
      cursor->option = agx_cursor_after_instr;
      cursor->instr = I;
      return;
   }

   UNREACHABLE("Invalid cursor option");
}

bool agx3_tex_uniform_pair_unwitnessed(const agx_context *ctx,
                                       const agx_instr *I, agx_index texture,
                                       agx_index sampler);
/* Whether replacing source s of an M5 main-program FADD/FMUL/FFMA with
 * `uniform` gives a native uniform-operand form (agx3_uniform_float_encode). */
bool agx3_m5_float_accepts_uniform(const agx_context *ctx, const agx_instr *I,
                                   unsigned s, unsigned uniform,
                                   enum agx_size size);

bool agx_instr_accepts_uniform(enum agx_isa_encoding encoding,
                               enum agx_opcode op, unsigned src_index,
                               unsigned value, enum agx_size size);

/* Routines defined for AIR */
void agx_print_index(agx_index index, bool is_float, FILE *fp);
void agx_print_instr(const agx_instr *I, FILE *fp);
void agx_print_block(const agx_block *block, FILE *fp);
void agx_print_shader(const agx_context *ctx, FILE *fp);
void agx_optimizer_forward(agx_context *ctx);
bool agx_eval_immediate_icond(agx_context *ctx, enum agx_icond icond,
                              agx_index x, agx_index y, bool *result);
void agx_optimizer_backward(agx_context *ctx);
void agx_lower_divergent_shuffle(agx_context *ctx);
void agx_lower_pseudo(agx_context *ctx);
void agx_lower_spill(agx_context *ctx);
void agx3_split_vector_stack_access(agx_context *ctx);
bool agx3_lower_texture_publications(agx_context *ctx);
bool agx3_fuse_fma_texture_publications(agx_context *ctx);
bool agx3_lower_scalar_texture_publications(agx_context *ctx);
bool agx3_lower_scalar_cube_coordinates(agx_context *ctx);
bool agx3_split_texture_coordinates(agx_context *ctx);
bool agx3_reserve_cube_scratch(agx_context *ctx);
void agx_lower_uniform_sources(agx_context *ctx);
void agx_lower_agx3_fma_sources(agx_context *ctx);
void agx_lower_agx3_immediates(agx_context *ctx);
void agx_materialize_agx3_cf_constants(agx_context *ctx);
void agx_opt_cse(agx_context *ctx);
bool agx_opt_collect_split(agx_context *ctx);
void agx_opt_cse_pressure(agx_context *ctx);
void agx_opt_cse_bounded(agx_context *ctx);
bool agx_can_rematerialize_uniform(const agx_context *ctx, const agx_instr *I);
bool agx_has_uniform_store(const agx_context *ctx);
void agx_opt_compact_constants(agx_context *ctx);
void agx_opt_promote_constants(agx_context *ctx);
void agx_opt_promote_m5_float_constants(agx_context *ctx);
void agx_dce(agx_context *ctx, bool partial);
void agx_pressure_schedule(agx_context *ctx);
void agx_ilp_schedule(agx_context *ctx);
int agx_ilp_m5_device_load_cap(int program_peak);
bool agx3_lower_uniform_program(agx_context *ctx);
/* Success/refusal, not a progress flag. Call only for AGX3B preambles. */
bool agx3_lower_cprog_inputs(agx_context *ctx);
bool agx3_cprog_spill_provenance(const agx_context *ctx);
bool agx3_cprog_certify_static_spill(const agx_context *ctx,
                                      bool imports_only_before_ra);
bool agx3_lower_uniform_half_exports(agx_context *ctx);
bool agx3_analyze_preamble_uniform_inputs(agx_context *ctx, bool *has_inputs);
/* UVSW carries the UVS index and the staging address in two independent
 * eight-bit operands, so the hardware addresses 256 vertex outputs while only
 * 128 distinct staging pairs exist.
 *
 * Apple's G17P compiler emits position plus thirty-one float4 user locations
 * as 128 UVSWs for UVS indices 0..127, and __GPU_LD_MD root.field1.field3 is
 * 128; a thirty-second float4 exceeds the fragment's 127 coefficients. Each
 * of those outputs stages at half 2k, and that scheme is kept verbatim for
 * the first AGX3_G17P_STAGING_OUTPUTS of any program.
 *
 * GL asks for more than Apple's own compiler ever emits: a points geometry
 * shader may write the full 128 interleaved transform-feedback components as
 * thirty-one float4 locations and still need position (four) and point size
 * (one), which is 129 outputs. Those past the staging capacity reuse the
 * addresses of batches that have already drained -- every batch boundary
 * waits on all six scoreboard slots, so the COPY/UVSW pair that last owned a
 * staging half has retired long before it is written again, twenty-one
 * batches later. Nothing about the UVS index itself wraps. */
#define AGX3_MAX_VERTEX_OUTPUTS 256u
#define AGX3_G17P_STAGING_OUTPUTS 128u

/* UVS_POSITION is the first group in the hardware's UVS order (see
 * agx_uvs.h) and agx_nir_lower_uvs.c gives it four entries whether or not
 * the shader writes gl_Position. A vertex program that writes no position
 * therefore starts its outputs here, and nowhere else. */
#define AGX3_UVS_POSITION_ENTRIES 4u

/* Number of 16-bit registers RA keeps reserved at the bottom of the register
 * file for the whole shader.
 */
static inline unsigned
agx_reserved_registers(const agx_context *ctx)
{
   if (ctx->agx3_cem_low_scratch)
      return 16;
   else if (ctx->has_spill_pcopy_reserved)
      return 8;
   else if (ctx->any_quad_divergent_shuffle)
      return ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2 ? 2 : 4;
   else if (ctx->any_cf)
      return 1;
   else
      return 0;
}

void agx_spill(agx_context *ctx, unsigned k, bool remat_only);
bool agx_fold_m5_spill_constant_phis(agx_context *ctx);
void agx_repair_ssa(agx_context *ctx);
void agx_reindex_ssa(agx_context *ctx);
void agx_ra(agx_context *ctx);
void agx_ra_without_remat(agx_context *ctx);
bool agx3_is_async(const agx_instr *I);
unsigned agx_calc_register_demand(agx_context *ctx, bool remat);
unsigned agx3_tile_store_bank(agx_context *ctx);

void agx_lower_64bit_postra(agx_context *ctx);
void agx_insert_waits(agx_context *ctx);
void agx_opt_empty_else(agx_context *ctx);
void agx_opt_register_cache(agx_context *ctx);
void agx_opt_break_if(agx_context *ctx);
void agx_opt_jmp_none(agx_context *ctx);
void agx_pack_binary(agx_context *ctx, struct util_dynarray *emission);

/* AGX3 (M3/M4/M5) binary emission. Returns false -- failing the shader,
 * never aborting -- on any op or operand outside the grounded subset.
 * The AGX2 path above is unaffected; callers select on the device
 * encoding. */
bool agx3_g17p_interleaved_zs(const agx_context *ctx);
bool agx3_pack_binary(agx_context *ctx, struct util_dynarray *emission);
bool agx3_interp_coords_from_itrcoords(const agx_context *ctx);
bool agx3_merge_unread_pixel_waits(agx_context *ctx);
const char *agx3_last_pack_refusal(void);

#ifndef NDEBUG
void agx_validate(agx_context *ctx, const char *after_str);
void agx_validate_ra(agx_context *ctx);
#else
static inline void
agx_validate(UNUSED agx_context *ctx, UNUSED const char *after_str)
{
}

static inline void
agx_validate_ra(UNUSED agx_context *ctx)
{
}
#endif

enum agx_size agx_split_width(const agx_instr *I);
bool agx_allows_16bit_immediate(agx_instr *I);
unsigned agx_negate_src_index(agx_instr *I);

static inline bool
agx_is_float_src(const agx_instr *I, unsigned s)
{
   struct agx_opcode_info info = agx_opcodes_info[I->op];
   bool fcmp = (I->op == AGX_OPCODE_FCMPSEL || I->op == AGX_OPCODE_FCMP);

   /* fcmp takes first 2 as floats but returns an integer */
   return info.is_float || (s < 2 && fcmp);
}

/* The M5 select with a uniform first comparand that native G17G
 * fragments use (agx3_alu_fcmp_select32_gpu1 cmp0_uniform): a 32-bit
 * result choosing a zero immediate or a 32-bit register after comparing an
 * aligned 32-bit uniform with a 32-bit register, by GE or LE once the
 * NaN-ignoring conditions are normalized as the packer does. Returns the
 * normalized condition and whether the selected operands are exchanged. */
static inline bool
agx3_m5_fcmpsel_uniform_cmp0(const agx_instr *I, unsigned *condition,
                             bool *swap)
{
   if (I->op != AGX_OPCODE_FCMPSEL || I->nr_dests != 1 || I->nr_srcs != 4 ||
       I->saturate || I->dest[0].size != AGX_SIZE_32 ||
       agx_channels(I->dest[0]) != 1)
      return false;
   const agx_index u = I->src[0], r = I->src[1];
   if (u.type != AGX_INDEX_UNIFORM || u.size != AGX_SIZE_32 || (u.value & 1) ||
       u.value >= 256 || u.abs || u.neg || u.memory ||
       r.type == AGX_INDEX_UNIFORM || r.type == AGX_INDEX_IMMEDIATE ||
       r.size != AGX_SIZE_32 || agx_channels(r) != 1 || r.abs || r.neg ||
       r.memory)
      return false;
   unsigned cond = I->fcond;
   bool exchanged = I->invert_cond;
   if (cond == AGX_FCOND_LTN) {
      cond = AGX_FCOND_GE;
      exchanged = !exchanged;
   } else if (cond == AGX_FCOND_GTN) {
      cond = AGX_FCOND_LE;
      exchanged = !exchanged;
   }
   if (cond != AGX_FCOND_GE && cond != AGX_FCOND_LE)
      return false;
   const agx_index imm = I->src[exchanged ? 3 : 2];
   const agx_index reg = I->src[exchanged ? 2 : 3];
   if (imm.type != AGX_INDEX_IMMEDIATE || imm.value != 0 || imm.abs ||
       imm.neg || reg.type == AGX_INDEX_UNIFORM ||
       reg.type == AGX_INDEX_IMMEDIATE || reg.size != AGX_SIZE_32 ||
       agx_channels(reg) != 1 || reg.abs || reg.neg || reg.memory)
      return false;
   *condition = cond;
   *swap = exchanged;
   return true;
}

struct agx_copy {
   /* Base register destination of the copy */
   unsigned dest;

   /* Destination is memory */
   bool dest_mem;

   /* Source of the copy */
   agx_index src;

   /* Whether the copy has been handled. Callers must leave to false. */
   bool done;
};

void agx_emit_parallel_copies(agx_builder *b, struct agx_copy *copies,
                              unsigned n);

/* As above, with an optional free 32-bit GPR (two aligned halves that no
 * copy reads or writes and that holds no live value at this point).  Register
 * cycles are then broken with one move through it instead of a three-XOR
 * swap per element.  Pass agx_null() when no such register is known. */
void agx_emit_parallel_copies_temp(agx_builder *b, struct agx_copy *copies,
                                   unsigned num_copies, agx_index temp);

void agx_compute_liveness(agx_context *ctx);
void agx_liveness_ins_update(struct u_sparse_bitset *live, agx_instr *I);

bool agx_nir_opt_preamble(nir_shader *s, unsigned *sizes, bool g17p,
                          bool agx3c,
                          const nir_opt_peephole_select_options *flatten);
bool agx_nir_export_select_flags_as_halves(nir_shader *s, bool branches);
bool agx_nir_remove_unused_preamble_exports(nir_shader *s, unsigned reserved);
bool agx_nir_coalesce_preamble_half_exports(nir_shader *s, unsigned reserved,
                                          unsigned allocated);
/* Highest uniform half a G17P sample can name directly as its texture or
 * sampler handle (agx_instr_accepts_uniform): handles above it are copied to
 * registers before every sample.
 */
#define AGX3_SMP_HANDLE_MAX_HALF 252

bool agx_nir_place_preamble_handles_low(nir_shader *s, unsigned reserved,
                                         unsigned exports_end,
                                         unsigned *allocated,
                                        bool load_bases);
agx_block *agx_reconverged_mrt_tail(agx_context *ctx);
bool agx_nir_lower_load_mask(nir_shader *shader);
bool agx_nir_lower_ubo(nir_shader *shader);
bool agx_nir_lower_shared_bitsize(nir_shader *shader, bool element_indices);
bool agx_nir_lower_frag_sidefx(nir_shader *s, bool g17p);
void agx_lower_agx3_fma_sources(agx_context *ctx);
void agx_materialize_agx3_alu_constants(agx_context *ctx);
void agx_lower_agx3_mixed_width_unary(agx_context *ctx);
bool agx_lower_g17g_retry_float_operands(agx_context *ctx);

struct agx_cycle_estimate {
   /* ALU throughput */
   unsigned alu;

   /* Floating point and SCIB (select, conditional, integer, and boolean)
    * throughput.
    */
   unsigned f_scib;

   /* IC (Integer and complex) throughput */
   unsigned ic;
};

struct agx_cycle_estimate agx_estimate_cycles(agx_context *ctx);

bool agx_is_alu(const agx_instr *I);

extern int agx_compiler_debug;

bool agx_instrs_equal_for_cse(const agx_instr *i1, const agx_instr *i2);

bool agx3_sink_vertex_outputs(agx_context *ctx);

bool agx3_coalesce_uniform_half_stores(agx_context *ctx);

bool agx_lower_agx3_device_atomic_sources(agx_context *ctx);

#ifdef __cplusplus
} /* extern C */
#endif
