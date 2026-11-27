/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "util/half_float.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/u_debug.h"
#include "agx3_uniform_integer_instr.h"
#include "util/u_debug.h"
#include "agx3_uniform_alu.h"
#include "agx3_isel_core.h"

/*
 * Not all instructions can take uniforms. Memory instructions can take
 * uniforms for their base source. Most forms use the lower half of the
 * uniform file; M5 device loads also accept aligned bases in the upper half.
 *
 * This pass lowers invalid uniforms.
 */
bool
agx_instr_accepts_uniform(enum agx_isa_encoding encoding, enum agx_opcode op,
                          unsigned src_index, unsigned value,
                          enum agx_size size)
{
   /* Some instructions only seem able to access uniforms in the low half */
   bool high = value >= 256;

   /* ALU cannot access 64-bit uniforms */
   bool is_64 = size == AGX_SIZE_64;
   /* AGX3 memory bases address four-half uniform groups. A pointer packed
    * at a two-half boundary must be copied to a GPR pair before use. */
   const bool m5_high_load_base =
      encoding == AGX_ISA_ENCODING_AGX3C && op == AGX_OPCODE_DEVICE_LOAD &&
      size == AGX_SIZE_64 && value <= 508;
   bool memory_base = (!high || m5_high_load_base) &&
      (encoding == AGX_ISA_ENCODING_AGX2 || !(value & 3));

   switch (op) {
   case AGX_OPCODE_FCMPSEL:
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return src_index < 4 && value < 512 &&
                (size == AGX_SIZE_16 || (size == AGX_SIZE_32 && !(value & 1)));
      return encoding == AGX_ISA_ENCODING_AGX2 && !is_64;
   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return size == AGX_SIZE_16 && value < 512 &&
                src_index < (op == AGX_OPCODE_HFMA ? 3 : 2);
      return encoding == AGX_ISA_ENCODING_AGX2 && !is_64;
   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_FFMA:
      /* Apple G17P/G16G arithmetic reads aligned f32 uniforms directly.
       * The instruction-level check below retains copies for mixed widths. */
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return (size == AGX_SIZE_16 ||
                 (size == AGX_SIZE_32 && !(value & 1))) &&
                value < 512 && src_index < (op == AGX_OPCODE_FFMA ? 3 : 2);
      return encoding == AGX_ISA_ENCODING_AGX2 && !is_64;
   case AGX_OPCODE_FLTNORM_AGX3:
      /* G17P colour packing has native uniform operands in both lanes.
       * Preserve aligned float32 sources instead of allocating temporary
       * GPRs and copying the constant buffer before every pack. */
      return encoding == AGX_ISA_ENCODING_AGX3B && src_index < 2 &&
             size == AGX_SIZE_32 && value < 512 && !(value & 1);
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_TEXTURE_LOAD:
      /* The saved G17S Atrous programs read texture handles directly from
       * aligned 64-bit uniforms (SMP kind 3), without per-lane copies. */
      if (encoding == AGX_ISA_ENCODING_AGX3C && op == AGX_OPCODE_TEXTURE_LOAD &&
          src_index == 3)
         return !high && !(value & 3) && value <= 252 && size == AGX_SIZE_64;
      /* G17P publishes a dynamic integer LOD through PACK_SMPIR from a GPR;
       * there is no witnessed uniform-file source for that operand.  Keep
       * the heap in uniforms, but materialize the LOD into a register. */
      assert(!(src_index == 2 && high) && "texture heap always low");
      return !high &&
             (src_index == 2 ||
              (encoding == AGX_ISA_ENCODING_AGX2 && src_index == 1));
   case AGX_OPCODE_TEXTURE_SAMPLE:
      /* The texture (3) and sampler (4)
       * index operands read the uniform file directly, as Apple's G17P
       * argument-buffer fragments do (kind 3, u12/u16). Keep Apple's
       * four-half alignment and the encoder's low-file range. */
      if (encoding == AGX_ISA_ENCODING_AGX3B &&
          (src_index == 3 || src_index == 4))
         return !high && !(value & 3) && value <= 252 &&
                (size == AGX_SIZE_64 || size == AGX_SIZE_32);
      /* G17P's LOD operand is an immediate level or a packed GPR value, not a
       * uniform-file operand. Source 2 is still the bindless texture heap. */
      assert(!(src_index == 2 && high) && "texture heap always low");
      return !high && (src_index == 2 ||
                       (encoding == AGX_ISA_ENCODING_AGX2 && src_index == 1));

   case AGX_OPCODE_TEX_STATE_STORE:
   case AGX_OPCODE_SAMPLER_STATE_STORE:
      assert(!high && "texture heap always low");
      return !high && src_index == 0;

   case AGX_OPCODE_INTERSECT_RAY_GEN1:
      return encoding == AGX_ISA_ENCODING_AGX3B && src_index < 2 &&
             !high && !(value & 1) && value <= 508 && size == AGX_SIZE_64;
   case AGX_OPCODE_INTERSECT_RAY_M5:
      return encoding == AGX_ISA_ENCODING_AGX3C && src_index == 1 &&
             size == AGX_SIZE_64 && !(value & 1) && value < 512;
   case AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3:
   case AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3:
      return (encoding == AGX_ISA_ENCODING_AGX3C ||
              (encoding == AGX_ISA_ENCODING_AGX3B && !high)) &&
             src_index == (op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ? 0 : 1) &&
             size == AGX_SIZE_64 && !(value & 1) && value < 512;
   case AGX_OPCODE_DEVICE_LOAD:
      return src_index == 0 && memory_base;
   case AGX_OPCODE_DEVICE_STORE:
   case AGX_OPCODE_ATOMIC:
      return src_index == 1 && memory_base;
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
      /* AGX3 threadgroup accesses with a promoted-constant uniform base
       * silently miss on G16G once the shader carries many of them
       * (tools/shared-atomic-probe.c mode 4: a 145-word shared struct copy
       * left every result untouched; AGX_MESA_DEBUG=nopromote passes). The
       * only witnessed uniform base is u4 in one small kernel, so keep
       * threadgroup bases in GPRs on AGX3B and let this pass copy them.
       * AGX3C still uses the fixed u8 base and has no grounded half-uniform
       * copy; retain that existing encoding's local-memory contract. */
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return false;
      return src_index == (op == AGX_OPCODE_LOCAL_LOAD ? 0 : 1);
   case AGX_OPCODE_IMAGE_WRITE:
      return src_index == 3;
   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      /* AGX3B compute block stores read their PBE heap from a uniform, as
       * image writes do. */
      if (encoding == AGX_ISA_ENCODING_AGX3B)
         return src_index == 0 && is_64 && memory_base;
      return false;
   case AGX_OPCODE_ST_VARY:
      /* G17P's vertex output store reads a 16-bit index from the uniform
       * file (agx3_vary_uvsw_uniform_index), as an unlinked vertex shader's
       * per-draw output indices are. */
      return encoding == AGX_ISA_ENCODING_AGX3B && src_index == 0 &&
             size == AGX_SIZE_16;
   case AGX_OPCODE_ZS_EMIT:
   case AGX_OPCODE_ZS_EMIT_AGX3:
   case AGX_OPCODE_ST_TILE:
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_UNIFORM_STORE:
   case AGX_OPCODE_LOCAL_ATOMIC:
   case AGX_OPCODE_SAMPLE_MASK:
   case AGX_OPCODE_ITER:
   case AGX_OPCODE_ITERPROJ:
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
   case AGX_OPCODE_BALLOT:
   case AGX_OPCODE_FCMP_BALLOT:
   case AGX_OPCODE_ICMP_BALLOT:
   case AGX_OPCODE_QUAD_BALLOT:
   case AGX_OPCODE_FCMP_QUAD_BALLOT:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
      return false;
   case AGX_OPCODE_EXPORT:
      /* Register allocation copies an export's uniform source into its
       * output register. AGX3 grounds that copy only for whole words; a
       * half uniform (a 16-bit constant such as a stencil reference) has no
       * witnessed uniform-to-half copy, so it is split from its aligned word
       * below, ahead of the export run. */
      return !(encoding == AGX_ISA_ENCODING_AGX3B && size == AGX_SIZE_16);
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
      /* AGX2 control-flow compares read 32-bit uniforms in either operand,
       * as they always have upstream. */
      if (encoding == AGX_ISA_ENCODING_AGX2)
         return !is_64;

      /* Apple's G17P loop
       * test compares a GPR with an aligned 32-bit uniform in one icmp
       * (the uniform's half index in f18, operand class f24 = 3), so
       * keep the uniform as the second operand instead of copying it into
       * a GPR on every iteration. */
      return agx_isa_is_agx3(encoding) &&
             src_index == 1 && size == AGX_SIZE_32 && !(value & 1) &&
             value < 256;
   case AGX_OPCODE_PHI:
      /* We would fail validation otherwise */
      return true;
   default:
      /* Keep copies for operands not yet supported by the AGX3 packer.
       * This is a compiler limitation, not an ISA-wide uniform restriction. */
      if (encoding != AGX_ISA_ENCODING_AGX2)
         return false;

      return !is_64;
   }
}

/*
 * SMP texture/sampler descriptor operands kept in the uniform file
 * (AGX_G17P_TEX_UNIFORM_OPERANDS, or the vertex-only qualification) are
 * restricted to the operand-kind pairs Apple's G17P compiler emits:
 * uniform/uniform and, outside cube sampling, a uniform texture with an
 * immediate sampler (native corpus: f103/f132 = 3/3 and 3/1 only, never a
 * register beside a uniform). Cube pairs are grounded only as same-kind pairs
 * (agx3_tex_core.c). Returns true when `texture`/`sampler`, as they would be
 * encoded, contain a retained uniform in any other pairing.
 */
static bool
agx_tex_uniform_stays(const agx_context *ctx, agx_index src, unsigned s)
{
   return src.type == AGX_INDEX_UNIFORM &&
          agx_instr_accepts_uniform(ctx->key->dev.encoding,
                                    AGX_OPCODE_TEXTURE_SAMPLE, s, src.value,
                                    src.size);
}

bool
agx3_tex_uniform_pair_unwitnessed(const agx_context *ctx, const agx_instr *I,
                                  agx_index texture, agx_index sampler)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       I->op != AGX_OPCODE_TEXTURE_SAMPLE)
      return false;
   const agx_index src[2] = {texture, sampler};
   enum agx_index_type kind[2];
   bool any_uniform = false;
   for (unsigned j = 0; j < 2; ++j) {
      const bool stays = agx_tex_uniform_stays(ctx, src[j], 3 + j);
      kind[j] = stays ? AGX_INDEX_UNIFORM :
                src[j].type == AGX_INDEX_IMMEDIATE ? AGX_INDEX_IMMEDIATE :
                                                     AGX_INDEX_NORMAL;
      any_uniform |= stays;
   }
   const bool cube = I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY;
   const bool witnessed =
      kind[0] == kind[1] ||
      (!cube && kind[0] == AGX_INDEX_UNIFORM &&
       kind[1] == AGX_INDEX_IMMEDIATE);
   return any_uniform && !witnessed;
}

/*
 * A G17P float operation reads a negated uniform only as an FMA addend.
 * Elsewhere, move the negation to an operand that may carry it instead of
 * copying the uniform into a register: -u * x == u * -x exactly, and
 * x - u == fma(x, 1.0, -u) with the same single rounding.
 */
static agx_instr *
agx3_move_uniform_negation(agx_context *ctx, agx_instr *I)
{
   if (I->op == AGX_OPCODE_FMUL || I->op == AGX_OPCODE_FFMA) {
      for (unsigned s = 0; s < 2; ++s) {
         agx_index *u = &I->src[s], *x = &I->src[1 - s];
         if (u->type == AGX_INDEX_UNIFORM && u->neg && !u->abs &&
             u->size == AGX_SIZE_32 && x->type == AGX_INDEX_NORMAL &&
             x->size == AGX_SIZE_32) {
            u->neg = false;
            x->neg = !x->neg;
         }
      }
      return I;
   }

   if (I->op != AGX_OPCODE_FADD || I->nr_dests != 1 ||
       I->dest[0].size != AGX_SIZE_32)
      return I;

   for (unsigned s = 0; s < 2; ++s) {
      const agx_index u = I->src[s], x = I->src[1 - s];
      if (u.type == AGX_INDEX_UNIFORM && u.neg && !u.abs &&
          u.size == AGX_SIZE_32 && x.type == AGX_INDEX_NORMAL &&
          x.size == AGX_SIZE_32) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_instr *fma =
            agx_ffma_to(&b, I->dest[0], x, agx_immediate_f(1.0f), u);
         fma->saturate = I->saturate;
         agx_remove_instruction(I);
         return fma;
      }
   }
   return I;
}

/* Modified M5 FMA uses the witnessed register-only modifier encoder.
 * Convert minifloats exactly before RA, preserving the one fused operation.
 * Unmodified immediate FMA keeps its existing compact operand forms. */
static void
agx3_lower_m5_modified_fma_immediates(agx_context *ctx, agx_instr *I)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       I->op != AGX_OPCODE_FFMA || I->nr_dests != 1 || I->nr_srcs != 3)
      return;

   bool modified = I->saturate;
   agx_foreach_src(I, s)
      modified |= I->src[s].abs || I->src[s].neg;
   if (!modified)
      return;

   /* Native G17G terrain fragments saturate a*b + 0.5 with the
    * minifloat addend in place (f51 = 32, f14 = 1, both multipliers plain
    * GPRs): agx3_falu_fma_addend_imm_width_mod encodes exactly that, so
    * keep the literal instead of rematerializing it before every use. */
   if (I->saturate && I->dest[0].size == AGX_SIZE_32 &&
       I->src[2].type == AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
       !I->src[2].neg) {
      bool plain = true;
      for (unsigned s = 0; s < 2; ++s) {
         const agx_index x = I->src[s];
         plain &= x.type == AGX_INDEX_NORMAL && x.size == AGX_SIZE_32 &&
                  agx_channels(x) == 1 && !x.abs && !x.neg && !x.memory;
      }
      if (plain)
         return;
   }

   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
   agx_foreach_src(I, s) {
      const agx_index literal = I->src[s];
      if (literal.type != AGX_INDEX_IMMEDIATE)
         continue;

      agx_index value = agx_mov_imm(
         &b, 32, fui(agx_minifloat_decode(literal.value)));
      value.abs = literal.abs;
      value.neg = literal.neg;
      I->src[s] = value;
   }
}

/* The direct M5 scalar LOD publisher consumes its half-float source.
 * Final physical divergence disables discard hints, and post-optimizer SSA
 * reuse may differ from the original NIR use count. Pack only those operands
 * that cannot use the consuming publisher, after optimization and before RA. */

static bool
agx3_m5_exact_lod_immediate(float value, bool explicit_lod, uint8_t *immediate)
{
   if (!isfinite(value) || !agx_minifloat_exact(value) ||
       value * 64.0f < -2048.0f || value * 64.0f > 2047.0f ||
       (explicit_lod && value < 0.0f))
      return false;

   const uint8_t imm = agx_minifloat_encode(value);
   const float decoded = agx_minifloat_decode(imm);
   const int encoded = (int)(decoded * 64.0f);
   /* Reject signed zero or fractions lost by the hardware immediate. */
   if (fui(decoded) != fui(value) ||
       fui((float)encoded / 64.0f) != fui(value))
      return false;

   *immediate = imm;
   return true;
}

/* Recover an exactly representable immediate before live-parameter
 * legalization gives a constant a register lifetime. */
static void
agx3_fold_m5_constant_lods(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->stage != MESA_SHADER_FRAGMENT)
      return;

   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   if (!defs)
      return;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }

   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || I->nr_srcs != 6 ||
          (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_3D) ||
          I->shadow || I->offset || I->sparse || I->query_lod ||
          I->gather != AGX_GATHER_NONE ||
          (I->lod_mode != AGX_LOD_MODE_LOD_MIN &&
           I->lod_mode != AGX_LOD_MODE_AUTO_LOD_BIAS))
         continue;

      agx_index lod = I->src[1];
      if (lod.type != AGX_INDEX_NORMAL || lod.abs || lod.neg || lod.memory ||
          agx_channels(lod) != 1)
         continue;
      agx_instr *def = defs[lod.value];
      if (I->memory_is_packed) {
         if (!def || def->op != AGX_OPCODE_PACK_SAMPLER_LOD_AGX3 ||
             def->nr_srcs != 1 || lod.size != AGX_SIZE_32)
            continue;
         lod = def->src[0];
         if (lod.type != AGX_INDEX_NORMAL || lod.abs || lod.neg || lod.memory ||
             agx_channels(lod) != 1)
            continue;
         def = defs[lod.value];
      }
      if (!def || def->op != AGX_OPCODE_MOV_IMM ||
          lod.size != AGX_SIZE_16 || def->nr_dests != 1 ||
          def->dest[0].size != AGX_SIZE_16 || def->imm > UINT16_MAX)
         continue;

      uint8_t immediate;
      if (!agx3_m5_exact_lod_immediate(_mesa_half_to_float(def->imm),
                                      I->lod_mode == AGX_LOD_MODE_LOD_MIN,
                                      &immediate))
         continue;

      I->src[1] = agx_immediate(immediate);
      I->memory_is_packed = false;
   }
   free(defs);
}

void
agx3_lower_m5_live_scalar_lods(agx_context *ctx)
{
   /* G18G samples through j873's G17G-family sampler
    * (agx3_emit_texture_sample_g17g), which takes a per-lane level or bias
    * only as the 16-bit scalar itself, not as a packed word.
    */
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       agx_is_g18g(&ctx->key->dev))
      return;

   bool needed = false;
   agx_foreach_instr_global(ctx, I) {
      needed |= I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
                (I->nr_srcs == 6 || I->agx3_split_coords) &&
                !I->memory_is_packed &&
                (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_3D) &&
                (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                 I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
                I->src[1].type == AGX_INDEX_NORMAL &&
                I->src[1].size == AGX_SIZE_16;
   }
   if (!needed)
      return;

   /* A single static use inside a uniform loop whose LOD is defined outside
    * the loop is still live across the back-edge, so agx_opt_register_cache()
    * cannot mark it discard. Compute SSA liveness so only a killed single-use
    * scalar in uniform control flow retains the direct unpacked path. */
   agx_compute_liveness(ctx);

   const unsigned alloc = ctx->alloc;
   unsigned *uses = calloc(alloc, sizeof(*uses));
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         uses[I->src[s].value]++;
   }

   agx_index *packed = NULL;
   agx_block **owner = NULL;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE ||
          (I->nr_srcs != 6 && !I->agx3_split_coords) ||
          I->memory_is_packed ||
          (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_3D) ||
          (I->lod_mode != AGX_LOD_MODE_LOD_MIN &&
           I->lod_mode != AGX_LOD_MODE_AUTO_LOD_BIAS))
         continue;

      const agx_index lod = I->src[1];
      if (lod.type != AGX_INDEX_NORMAL || lod.size != AGX_SIZE_16 ||
          agx_channels(lod) != 1 ||
          (!v_block->cf_divergent && !ctx->any_quad_divergent_shuffle &&
           uses[lod.value] == 1 && lod.kill))
         continue;

      if (!owner) {
         packed = calloc(alloc, sizeof(*packed));
         owner = calloc(alloc, sizeof(*owner));
      }

      /* Share identical pure producers within this block. Sibling blocks
       * cannot share a producer inserted at a use without dominance proof. */
      if (owner[lod.value] != v_block) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         packed[lod.value] = agx_temp(ctx, AGX_SIZE_32);
         agx_pack_sampler_lod_agx3_to(&b, packed[lod.value], lod);
         owner[lod.value] = v_block;
      }
      I->src[1] = packed[lod.value];
      I->memory_is_packed = true;
   }
   free(owner);
   free(packed);
   free(uses);
}

/* Validate the narrow native M5 FALU forms before retaining uniform
 * operands. The encoder owns the exact source/modifier domain; dummy GPRs
 * suffice here because allocation has not assigned their indices yet. */
static bool
m5_native_float_uniform(const agx_context *ctx, const agx_instr *I)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->is_preamble || I->nr_dests != 1 ||
       I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
       (I->op != AGX_OPCODE_FADD && I->op != AGX_OPCODE_FMUL &&
        I->op != AGX_OPCODE_FFMA))
      return false;
   const bool fma = I->op == AGX_OPCODE_FFMA;
   if (I->nr_srcs != (fma ? 3 : 2))
      return false;
   struct agx3_uniform_float_src src[3] = {0};
   agx_foreach_src(I, s) {
      const agx_index v = I->src[s];
      const bool uniform = v.type == AGX_INDEX_UNIFORM;
      if (v.memory || agx_channels(v) != 1 ||
          (v.type == AGX_INDEX_IMMEDIATE ? v.value > 255
           : v.type != AGX_INDEX_NORMAL && !uniform) ||
          (v.type != AGX_INDEX_IMMEDIATE && v.size != AGX_SIZE_32 &&
           !(uniform && v.size == AGX_SIZE_16)))
         return false;
      src[s] = (struct agx3_uniform_float_src){
         .kind = uniform ? AGX3_FLOAT_UNIFORM
                 : v.type == AGX_INDEX_IMMEDIATE ? AGX3_FLOAT_IMMEDIATE
                                                 : AGX3_FLOAT_REGISTER,
         .value = v.type == AGX_INDEX_NORMAL ? 0 : v.value,
         .abs = v.abs, .neg = v.neg,
         .half = uniform && v.size == AGX_SIZE_16,
      };
   }
   uint8_t out[16], length;
   return agx3_uniform_float_encode(AGX_ISA_ENCODING_AGX3C,
      fma ? AGX3_UNIFORM_FMA
          : I->op == AGX_OPCODE_FMUL ? AGX3_UNIFORM_FMUL : AGX3_UNIFORM_FADD,
      0, src, I->saturate, false, 0, out, &length);
}

/*
 * Every native M5 float form with a uniform operand also has a GPR operand,
 * so an operation on uniforms and immediates alone has to copy a uniform. One
 * copy suffices whenever the rest is then a native form: u0 - u1 is R + -U
 * after copying u0, whereas copying u1 would leave U - R, which has none.
 * Mark the uniforms that can stay when one is copied.
 */
static bool
m5_float_uniforms_to_keep(const agx_context *ctx, const agx_instr *I,
                          bool keep[3])
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->is_preamble || I->nr_srcs > 3 ||
       (I->op != AGX_OPCODE_FADD && I->op != AGX_OPCODE_FMUL &&
        I->op != AGX_OPCODE_FFMA))
      return false;

   unsigned uniforms = 0;
   agx_foreach_src(I, s)
      uniforms += I->src[s].type == AGX_INDEX_UNIFORM;
   if (uniforms < 2)
      return false;

   agx_foreach_src(I, s) {
      if (I->src[s].type != AGX_INDEX_UNIFORM)
         continue;

      /* The copy is a register of the uniform's size. The encoder query
       * looks at a register's size and modifiers, not at its index. */
      agx_instr candidate = *I;
      agx_index sources[3];
      memcpy(sources, I->src, I->nr_srcs * sizeof(*sources));
      candidate.src = sources;
      sources[s] =
         agx_replace_index(sources[s], agx_get_index(0, I->src[s].size));

      if (m5_native_float_uniform(ctx, &candidate)) {
         agx_foreach_src(I, t)
            keep[t] = t != s && I->src[t].type == AGX_INDEX_UNIFORM;
         return true;
      }
   }

   return false;
}

bool
agx3_m5_float_accepts_uniform(const agx_context *ctx, const agx_instr *I,
                              unsigned s, unsigned uniform,
                              enum agx_size size)
{
   if (s >= I->nr_srcs || I->nr_srcs > 3 || uniform >= 512)
      return false;
   agx_instr candidate = *I;
   agx_index sources[3];
   memcpy(sources, I->src, I->nr_srcs * sizeof(*sources));
   candidate.src = sources;
   sources[s] = agx_replace_index(sources[s], agx_uniform(uniform, size));
   return m5_native_float_uniform(ctx, &candidate);
}

/*
 * M5 adds a uniform to a register with the uniform optionally negated, not
 * the register: u - r has no native form and would copy u into a GPR. When
 * every use of u - r is a float operation reading it as a plain register
 * operand, compute r - u instead (a native R + -U) and negate it at each use,
 * only when NIR permits a zero result of either sign. Equal finite operands
 * otherwise change from +0 to -0. Minecraft's fog mixes the fragment colour
 * toward the fog colour as (fog - c) * t + c, three times per fragment.
 */
static bool
agx3_m5_negate_uniform_difference(agx_context *ctx, agx_instr *I)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C || ctx->is_preamble ||
       I->op != AGX_OPCODE_FADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
       I->saturate || !I->no_signed_zero ||
       I->dest[0].type != AGX_INDEX_NORMAL || I->dest[0].size != AGX_SIZE_32)
      return false;
   unsigned u = I->src[0].type == AGX_INDEX_UNIFORM ? 0 : 1;
   const agx_index uni = I->src[u], reg = I->src[!u];
   if (uni.type != AGX_INDEX_UNIFORM || uni.neg || uni.abs ||
       reg.type != AGX_INDEX_NORMAL || !reg.neg || reg.abs)
      return false;

   const unsigned value = I->dest[0].value;
   agx_foreach_instr_global(ctx, J) {
      agx_foreach_ssa_src(J, s) {
         if (J->src[s].value != value)
            continue;
         if ((J->op != AGX_OPCODE_FADD && J->op != AGX_OPCODE_FMUL &&
              J->op != AGX_OPCODE_FFMA) || J->src[s].abs)
            return false;
      }
   }

   agx_index r = reg, n = uni;
   r.neg = false;
   n.neg = true;
   I->src[0] = r;
   I->src[1] = n;
   agx_foreach_instr_global(ctx, J) {
      agx_foreach_ssa_src(J, s) {
         if (J->src[s].value == value)
            J->src[s].neg = !J->src[s].neg;
      }
   }
   return true;
}

/* Native M5 bounded-buffer code uses uniform compare/select operands.
 * Query the encoder's full32 contract before retaining any of them. */
static bool
m5_native_isel_uniform(const agx_context *ctx, const agx_instr *I)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->is_preamble || I->op != AGX_OPCODE_ICMPSEL || I->saturate ||
       I->invert_cond || I->nr_dests != 1 ||
       I->nr_srcs != 4 || I->dest[0].size != AGX_SIZE_32 ||
       agx_channels(I->dest[0]) != 1 || I->src[0].type != AGX_INDEX_NORMAL)
      return false;
   bool uniform = false;
   for (unsigned i = 0; i < 4; ++i) {
      const agx_index x = I->src[i];
      uniform |= x.type == AGX_INDEX_UNIFORM;
      if (x.memory || x.abs || x.neg || agx_channels(x) != 1 ||
          (x.type != AGX_INDEX_IMMEDIATE && x.size != AGX_SIZE_32) ||
          (x.type != AGX_INDEX_NORMAL && x.type != AGX_INDEX_UNIFORM &&
           x.type != AGX_INDEX_IMMEDIATE))
         return false;
   }
   if (!uniform)
      return false;
   struct agx3_isel_desc d = {
      .dest = 0, .cmp0 = 0,
      .cmp1 = I->src[1].type == AGX_INDEX_NORMAL ? 0 : I->src[1].value,
      .cmp1_immediate = I->src[1].type == AGX_INDEX_IMMEDIATE,
      .cmp1_uniform = I->src[1].type == AGX_INDEX_UNIFORM,
      .cond = I->icond == AGX_ICOND_UEQ ? AGX3_ISEL_ICOND_EQ :
                                       (enum agx3_isel_icond)I->icond,
      .explicit_hints = true,
   };
   enum agx3_isel_operand *kind[2] = {&d.true_kind, &d.false_kind};
   uint16_t *value[2] = {&d.sel_true, &d.sel_false};
   for (unsigned i = 0; i < 2; ++i) {
      const agx_index x = I->src[2 + i];
      *kind[i] = x.type == AGX_INDEX_UNIFORM ? AGX3_ISEL_UNIFORM :
                 x.type == AGX_INDEX_IMMEDIATE ? AGX3_ISEL_IMMEDIATE :
                                                AGX3_ISEL_REGISTER;
      *value[i] = x.type == AGX_INDEX_NORMAL ? 0 : x.value;
   }
   uint8_t bytes[AGX3_ISEL_MAX_LENGTH], length;
   return agx3_isel_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length);
}

bool
agx3_m5_isel_accepts_uniform(const agx_context *ctx, const agx_instr *I,
                             unsigned s, unsigned uniform, enum agx_size size)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->key->dev.emit_agx3d || ctx->is_preamble ||
       I->op != AGX_OPCODE_ICMPSEL || s == 0 || s >= 4 || I->nr_srcs != 4 ||
       size != AGX_SIZE_32 || (uniform & 1) || uniform >= 512)
      return false;

   agx_instr candidate = *I;
   agx_index sources[4];
   memcpy(sources, I->src, sizeof(sources));
   candidate.src = sources;
   sources[s] = agx_replace_index(sources[s], agx_uniform(uniform, size));
   return m5_native_isel_uniform(ctx, &candidate);
}

/* On AGX3C, native uniform ISEL takes a uniform right comparand (cmp1_uniform)
 * beside a 32-bit register left comparand. Canonicalize `u op r` to `r op' u`
 * when that yields a native form. */
static void
m5_canonicalize_isel_uniform_cmp(const agx_context *ctx, agx_instr *I)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->key->dev.emit_agx3d || ctx->is_preamble ||
       I->op != AGX_OPCODE_ICMPSEL || I->nr_srcs != 4 || I->saturate ||
       I->invert_cond || I->src[0].type != AGX_INDEX_UNIFORM ||
       I->src[1].type != AGX_INDEX_NORMAL)
      return;

   enum agx_icond rev;
   switch (I->icond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      rev = I->icond;
      break;
   case AGX_ICOND_ULT: rev = AGX_ICOND_UGT; break;
   case AGX_ICOND_UGT: rev = AGX_ICOND_ULT; break;
   case AGX_ICOND_SLT: rev = AGX_ICOND_SGT; break;
   case AGX_ICOND_SGT: rev = AGX_ICOND_SLT; break;
   default:
      return;
   }

   agx_instr candidate = *I;
   agx_index sources[4];
   memcpy(sources, I->src, sizeof(sources));
   sources[0] = I->src[1];
   sources[1] = I->src[0];
   candidate.src = sources;
   candidate.icond = rev;
   if (!m5_native_isel_uniform(ctx, &candidate))
      return;

   I->src[0] = sources[0];
   I->src[1] = sources[1];
   I->icond = rev;
}

/*
 * The ICMPSEL operands that read the uniform file in place. G17P reads an
 * aligned 32-bit uniform directly as one comparand beside a 32-bit register
 * (f36/f42 = 3, f51/f57 = 3) or as either select (f66/f72 = 3, f83/f89 = 3),
 * 32-bit result (agx3_isel_core.c witnesses); M5's forms are queried from its
 * encoder. Prefer the selects when both kinds are present, since a uniform
 * comparand is only witnessed beside register selects.
 */
static void
agx3_isel_uniforms_kept(const agx_context *ctx, const agx_instr *I,
                        bool keep[4])
{
   memset(keep, 0, 4 * sizeof(*keep));
   if (m5_native_isel_uniform(ctx, I)) {
      for (unsigned i = 0; i < 4; ++i)
         keep[i] = I->src[i].type == AGX_INDEX_UNIFORM;
      return;
   }

   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->is_preamble || I->op != AGX_OPCODE_ICMPSEL ||
       I->nr_dests != 1 || I->nr_srcs != 4 ||
       I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
       I->src[0].size != AGX_SIZE_32 || I->src[1].size != AGX_SIZE_32)
      return;

   bool uniform_ok[4], reg32[4];
   for (unsigned j = 0; j < 4; ++j) {
      const agx_index x = I->src[j];
      uniform_ok[j] = x.type == AGX_INDEX_UNIFORM &&
                      x.size == AGX_SIZE_32 && !(x.value & 1) &&
                      x.value + 2 <= 256 && !x.abs && !x.neg && !x.memory;
      reg32[j] = x.type == AGX_INDEX_NORMAL && x.size == AGX_SIZE_32 &&
                 !x.abs && !x.neg && agx_channels(x) == 1;
   }
   if (uniform_ok[2] || uniform_ok[3]) {
      keep[2] = uniform_ok[2];
      keep[3] = uniform_ok[3];
   } else if (uniform_ok[0] && reg32[1] && reg32[2] && reg32[3]) {
      keep[0] = true;
   } else if (uniform_ok[1] && reg32[0] && reg32[2] && reg32[3]) {
      keep[1] = true;
   }
}

/* A select alternative that equals its value zero-extended to 32 bits. */
static bool
agx3_zext_select(agx_index x, const agx_index *widened, unsigned nr_widened,
                 agx_index *wide)
{
   if (x.abs || x.neg || x.memory)
      return false;
   if (x.type == AGX_INDEX_IMMEDIATE && x.value <= UINT8_MAX) {
      *wide = x;
      return true;
   }
   if (x.type == AGX_INDEX_NORMAL && x.value < nr_widened &&
       widened[x.value].type == AGX_INDEX_NORMAL) {
      *wide = widened[x.value];
      return true;
   }
   return false;
}

/*
 * Every native uniform ISEL form writes a 32-bit result, so a comparison
 * producing a 16-bit boolean (an ICMP, or an ICMPSEL selecting halves) copied
 * its uniform comparand into a GPR at each use. When the 32-bit select of the
 * same comparison reads that uniform in place, compute it instead and take
 * its low half. Its alternatives are 8-bit immediates or 16-bit results
 * widened here, so its upper half is zero and its lower half is exactly the
 * 16-bit result; they are the same register, so allocation coalesces the
 * split and no copy remains. `widened` maps those results to the 32-bit
 * values, for comparisons selecting them.
 */
static agx_instr *
agx3_widen_uniform_compare(agx_context *ctx, agx_instr *I, agx_index *widened,
                           unsigned nr_widened)
{
   const bool icmp = I->op == AGX_OPCODE_ICMP && I->nr_srcs == 2;
   if ((!icmp && !(I->op == AGX_OPCODE_ICMPSEL && I->nr_srcs == 4)) ||
       I->nr_dests != 1 || I->saturate ||
       I->dest[0].type != AGX_INDEX_NORMAL || I->dest[0].memory ||
       I->dest[0].size != AGX_SIZE_16 || agx_channels(I->dest[0]) != 1 ||
       (I->src[0].type != AGX_INDEX_UNIFORM &&
        I->src[1].type != AGX_INDEX_UNIFORM))
      return I;

   /* agx_lower_pseudo() selects 1 or 0 for a comparison; an inverted
    * condition exchanges the alternatives. */
   agx_index src[4] = {I->src[0], I->src[1],
                       icmp ? agx_immediate(1) : I->src[2],
                       icmp ? agx_immediate(0) : I->src[3]};
   if (I->invert_cond) {
      const agx_index t = src[2];
      src[2] = src[3];
      src[3] = t;
   }
   for (unsigned i = 2; i < 4; ++i) {
      if (!agx3_zext_select(src[i], widened, nr_widened, &src[i]))
         return I;
   }

   agx_index dest = I->dest[0];
   dest.size = AGX_SIZE_32;
   agx_instr candidate = *I;
   candidate.op = AGX_OPCODE_ICMPSEL;
   candidate.nr_srcs = 4;
   candidate.invert_cond = false;
   candidate.dest = &dest;
   candidate.src = src;
   m5_canonicalize_isel_uniform_cmp(ctx, &candidate);

   bool keep[4];
   agx3_isel_uniforms_kept(ctx, &candidate, keep);
   for (unsigned i = 0; i < 2; ++i) {
      if (src[i].type == AGX_INDEX_UNIFORM && !keep[i])
         return I;
   }

   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
   const agx_index wide = agx_temp(ctx, AGX_SIZE_32);
   agx_instr *select = agx_icmpsel_to(&b, wide, src[0], src[1], src[2],
                                      src[3], candidate.icond);
   agx_instr *split = agx_split(&b, 2, wide);
   split->dest[0] = I->dest[0];
   split->dest[1] = agx_temp(ctx, AGX_SIZE_16);
   if (I->dest[0].value < nr_widened)
      widened[I->dest[0].value] = wide;
   agx_remove_instruction(I);
   return select;
}

void
agx_lower_uniform_sources(agx_context *ctx)
{
   const unsigned nr_widened = ctx->alloc;
   agx_index *widened =
      agx_isa_is_agx3(ctx->key->dev.encoding)
         ? calloc(nr_widened, sizeof(*widened)) : NULL;

   agx_foreach_instr_global_safe(ctx, I) {
      agx3_lower_m5_modified_fma_immediates(ctx, I);

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         I = agx3_move_uniform_negation(ctx, I);

      if (widened)
         I = agx3_widen_uniform_compare(ctx, I, widened, nr_widened);

      agx_instr *at = I;
      struct agx3_uniform_integer_desc integer_desc;
      const bool native_integer =
         agx3_uniform_integer_for_encoding(ctx->key->dev.encoding,
                                           I, false, &integer_desc);
      if (!m5_native_float_uniform(ctx, I))
         agx3_m5_negate_uniform_difference(ctx, I);
      const bool native_m5_float = m5_native_float_uniform(ctx, I);
      bool keep_m5_float_uniform[3] = {false};
      if (!native_m5_float)
         m5_float_uniforms_to_keep(ctx, I, keep_m5_float_uniform);
      bool copy_float_uniforms = false;
      unsigned first_uniform = UINT_MAX;
      const bool half_float = I->op == AGX_OPCODE_HADD ||
                              I->op == AGX_OPCODE_HMUL || I->op == AGX_OPCODE_HFMA;
      const bool full_float = I->op == AGX_OPCODE_FADD ||
                              I->op == AGX_OPCODE_FMUL || I->op == AGX_OPCODE_FFMA;
      bool native_minmax = I->op == AGX_OPCODE_FCMPSEL && I->nr_dests == 1 &&
                           I->nr_srcs == 4 && !I->saturate && !I->invert_cond &&
                           (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
                           agx_is_equiv(I->src[0], I->src[2]) &&
                           agx_is_equiv(I->src[1], I->src[3]);
      if (native_minmax) {
         agx_foreach_src(I, s) {
            native_minmax &= I->src[s].type != AGX_INDEX_IMMEDIATE &&
                             I->src[s].size == I->dest[0].size &&
                             !I->src[s].abs && !I->src[s].neg;
         }
      }
      const bool both_minmax_uniform = native_minmax &&
         I->src[0].type == AGX_INDEX_UNIFORM && I->src[1].type == AGX_INDEX_UNIFORM;
      /* Native AGX3B uniform-comparand fsel witnesses only select registers
       * or immediate zero (never non-zero integer immediates such as the
       * #1/#0 produced when agx_lower_pseudo() lowers AGX_OPCODE_FCMP). */
      const bool fsel_shape =
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
         !(I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value != 0) &&
         !(I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value != 0);
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (half_float || full_float)) {
         const enum agx_size size = half_float ? AGX_SIZE_16 : AGX_SIZE_32;
         copy_float_uniforms = I->nr_dests != 1 ||
                               I->dest[0].size != size || (half_float && I->saturate);
         bool has_gpr = false;
         unsigned half_uniforms = 0;
         agx_foreach_src(I, s) {
            const bool native_half_uniform =
               full_float && I->src[s].type == AGX_INDEX_UNIFORM &&
               I->src[s].size == AGX_SIZE_16 && !I->src[s].abs &&
               !I->src[s].neg;
            half_uniforms += native_half_uniform;
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               copy_float_uniforms |=
                  I->src[s].size != size && !native_half_uniform;
            if (half_float)
               copy_float_uniforms |= I->src[s].abs || I->src[s].neg;
            has_gpr |= I->src[s].type != AGX_INDEX_UNIFORM &&
                       I->src[s].type != AGX_INDEX_IMMEDIATE;
            if (first_uniform == UINT_MAX && I->src[s].type == AGX_INDEX_UNIFORM)
               first_uniform = s;
         }
         if (has_gpr)
            first_uniform = UINT_MAX;
         else if (half_uniforms)
            copy_float_uniforms = true;
         copy_float_uniforms |=
            half_uniforms > 1 || (half_uniforms && I->saturate);
      }

      bool keep_isel_uniform[4];
      unsigned m5_fsel_cond;
      bool m5_fsel_swap;
      const bool m5_fsel_uniform =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
         !ctx->is_preamble &&
         agx3_m5_fcmpsel_uniform_cmp0(I, &m5_fsel_cond, &m5_fsel_swap);
      m5_canonicalize_isel_uniform_cmp(ctx, I);
      agx3_isel_uniforms_kept(ctx, I, keep_isel_uniform);

      /* An unwitnessed descriptor pair (agx3_tex_uniform_pair_unwitnessed)
       * materializes its uniform operand, keeping the register form. */
      const bool copy_tex_uniform =
         I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
         agx3_tex_uniform_pair_unwitnessed(ctx, I, I->src[3], I->src[4]);

      /* M5 fragment and compute SMP read both the texture and the sampler
       * handle from aligned 64-bit uniforms (f97/f103=3 and f126/f132=3)
       * when Metal passes them through an argument buffer; the native
       * G17G compiler compiles texture2d::sample, level and bias that way,
       * and so samples in compute kernels.
       * Only that pair is witnessed: keep both or copy both. */
      bool m5_tex_uniform_pair =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
         (ctx->stage == MESA_SHADER_FRAGMENT ||
          ctx->stage == MESA_SHADER_COMPUTE) && !ctx->is_preamble &&
         I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->nr_srcs >= 5;
      for (unsigned j = 3; m5_tex_uniform_pair && j < 5; ++j) {
         const agx_index x = I->src[j];
         m5_tex_uniform_pair = x.type == AGX_INDEX_UNIFORM &&
            x.size == AGX_SIZE_64 && !(x.value & 3) && x.value <= 252 &&
            !x.abs && !x.neg && !x.memory;
      }

      /* Exports end the block and admit no instruction between them; place
       * their materialized sources before the first export of the run. */
      if (I->op == AGX_OPCODE_EXPORT) {
         while (at->link.prev != &v_block->instructions) {
            agx_instr *prev = list_entry(at->link.prev, agx_instr, link);
            if (prev->op != AGX_OPCODE_EXPORT)
               break;
            at = prev;
         }
      }

      agx_builder b = agx_init_builder(ctx, agx_before_instr(at));

      /* A tied compare-select (fmin/fmax lowered to fcmpsel(a, b, a, b)) is
       * grounded on AGX3 only while its compared and selected sources stay the
       * same register. Materialize a uniform once per instruction and reuse
       * the copy for every source that named the same uniform. */
      agx_index materialized_from[8] = {0};
      agx_index materialized_to[8] = {0};
      unsigned materialized = 0;

      agx_foreach_src(I, s) {
         /* A selected uniform may already have been copied for the
          * comparison. Reuse that value even when the selected operand
          * could read the uniform directly, preserving min/max operand ties. */
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_FCMPSEL &&
             I->src[s].type == AGX_INDEX_UNIFORM) {
            agx_index idx = I->src[s];
            idx.abs = idx.neg = false;
            bool reused = false;
            for (unsigned m = 0; m < materialized; ++m) {
               if (agx_is_equiv(materialized_from[m], idx) &&
                   materialized_from[m].size == idx.size) {
                  agx_replace_src(I, s, materialized_to[m]);
                  reused = true;
                  break;
               }
            }
            if (reused)
               continue;
         }
         const bool copy_minmax = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            I->op == AGX_OPCODE_FCMPSEL &&
            (!native_minmax || (both_minmax_uniform && !(s & 1)));
         /* Apple may precompute abs(uniform) in its constant program. The
          * native main-program witness proves negation only on an FMA addend;
          * retain materialization for the other modified uniform operands. */
         const bool copy_float_modifier =
            ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && full_float &&
            (I->src[s].abs ||
             (I->src[s].neg && !(I->op == AGX_OPCODE_FFMA && s == 2)));
         /* The reg/uniform compare (AGX_G17P_UNIFORM_COMPARE) exists only
          * with a 32-bit register first operand. With an immediate, uniform
          * or 16-bit first operand keep copying the uniform, so the existing
          * register forms still encode the compare. AGX2 has no such
          * restriction. */
         const bool is_int_compare =
            I->op == AGX_OPCODE_IF_ICMP || I->op == AGX_OPCODE_WHILE_ICMP ||
            I->op == AGX_OPCODE_BREAK_IF_ICMP;
         /* An aligned 32-bit
          * uniform against an 8-bit immediate is one native icmp (Apple's
          * G17P loop guard, e.g. u6 == 0). Keep the uniform
          * in place instead of copying it into a GPR first; the packer
          * emits agx3_cf_icmp_uniform_imm() for either operand order. */
         const bool uniform_imm_compare =
            (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
             (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
              ctx->stage == MESA_SHADER_COMPUTE && !ctx->is_preamble)) &&
            is_int_compare && I->nr_srcs == 2 &&
            I->src[s].type == AGX_INDEX_UNIFORM &&
            I->src[s].size == AGX_SIZE_32 && !(I->src[s].value & 1) &&
            I->src[s].value < 256 && !I->src[s].abs && !I->src[s].neg &&
            I->src[1 - s].type == AGX_INDEX_IMMEDIATE &&
            I->src[1 - s].value <= UINT8_MAX && !I->src[1 - s].abs &&
            !I->src[1 - s].neg;
         const bool copy_compare_uniform =
            ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
            is_int_compare && !uniform_imm_compare &&
            (I->src[0].type != AGX_INDEX_NORMAL ||
             I->src[0].size != AGX_SIZE_32 || I->src[0].abs ||
             I->src[0].neg ||
             (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
              (ctx->stage != MESA_SHADER_COMPUTE || ctx->is_preamble)));
         const bool copy_tex = copy_tex_uniform && (s == 3 || s == 4);
         /* A G17P select compares a uniform (a half, or an aligned word)
          * against an immediate zero directly, as Apple does for constant
          * buffer values and constant-program flags (agx3_alu_fcmp_select,
          * f42 = 3). */
         const bool select_uniform_flag =
            ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            !ctx->is_preamble && fsel_shape &&
            I->nr_dests == 1 && !I->saturate &&
            I->fcond != AGX_FCOND_LTN && I->fcond != AGX_FCOND_GTN &&
            (I->dest[0].size == AGX_SIZE_16 ||
             I->dest[0].size == AGX_SIZE_32) &&
            agx_channels(I->dest[0]) == 1 && !native_minmax && s < 2 &&
            I->src[s].type == AGX_INDEX_UNIFORM && I->src[s].value < 256 &&
            (I->src[s].size == AGX_SIZE_16 ||
             (I->src[s].size == AGX_SIZE_32 && !(I->src[s].value & 1))) &&
            !I->src[s].abs && !I->src[s].neg &&
            I->src[1 - s].type == AGX_INDEX_IMMEDIATE &&
            I->src[1 - s].value == 0 && !I->src[1 - s].abs &&
            !I->src[1 - s].neg;
         /* A G17P select also compares a register (slot 0, no negate; an
          * absolute value is witnessed) against a uniform in slot 1 directly
          * (agx3_alu_fcmp_select, f59 = 3): Apple's r < u shape, 14,500
          * times in the Subnautica/Fallout NV native corpus. A uniform first
          * comparand beside a register is exchanged into that shape below. */
         const bool select_uniform_comparand =
            ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            !ctx->is_preamble && fsel_shape &&
            I->nr_dests == 1 && !I->saturate &&
            (I->dest[0].size == AGX_SIZE_16 ||
             I->dest[0].size == AGX_SIZE_32) &&
            agx_channels(I->dest[0]) == 1 && !native_minmax && s < 2 &&
            I->src[s].type == AGX_INDEX_UNIFORM &&
            /* slot 1 (f53) holds half indices up to 511; a slot-0 uniform
             * is always exchanged into it below */
            I->src[s].value < 512 && !I->src[s].memory &&
            (I->src[s].size == AGX_SIZE_16 ||
             (I->src[s].size == AGX_SIZE_32 && !(I->src[s].value & 1))) &&
            !I->src[s].abs && !I->src[s].neg &&
            I->src[1 - s].type == AGX_INDEX_NORMAL &&
            agx_channels(I->src[1 - s]) == 1 && !I->src[1 - s].memory &&
            !I->src[1 - s].neg &&
            (I->src[1 - s].size == AGX_SIZE_32 ||
             (I->src[1 - s].size == AGX_SIZE_16 && !I->src[1 - s].abs)) &&
            (s == 1 || I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
             I->fcond == AGX_FCOND_GT || I->fcond == AGX_FCOND_LE ||
             I->fcond == AGX_FCOND_GE);
         if (select_uniform_comparand && s == 0) {
            /* Exchange the comparands and reverse the relation: exact for
             * every ordered condition, NaN compares false either way. */
            const agx_index u = I->src[0];
            I->src[0] = I->src[1];
            I->src[1] = u;
            switch (I->fcond) {
            case AGX_FCOND_LT: I->fcond = AGX_FCOND_GT; break;
            case AGX_FCOND_GT: I->fcond = AGX_FCOND_LT; break;
            case AGX_FCOND_LE: I->fcond = AGX_FCOND_GE; break;
            case AGX_FCOND_GE: I->fcond = AGX_FCOND_LE; break;
            default: break;
            }
         }
         /* A G17P select reads a uniform alternative of the result's width
          * directly (f70 / f87, kind 3; native corpus form 121 beside a
          * slot-1 uniform comparand, forms 127/136/139 beside registers). */
         const bool select_uniform_alternative =
            ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            !ctx->is_preamble && I->op == AGX_OPCODE_FCMPSEL &&
            I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
            I->fcond != AGX_FCOND_LTN && I->fcond != AGX_FCOND_GTN &&
            (I->dest[0].size == AGX_SIZE_16 ||
             I->dest[0].size == AGX_SIZE_32) &&
            agx_channels(I->dest[0]) == 1 && !native_minmax && s >= 2 &&
            I->src[s].type == AGX_INDEX_UNIFORM && I->src[s].value < 256 &&
            !I->src[s].memory && !I->src[s].abs && !I->src[s].neg &&
            I->src[s].size == I->dest[0].size &&
            (I->src[s].size == AGX_SIZE_16 || !(I->src[s].value & 1)) &&
            !(I->src[0].type == AGX_INDEX_IMMEDIATE &&
              I->src[1].type == AGX_INDEX_IMMEDIATE);
         /* G18G: export coalescing can introduce copies during register
          * allocation. Materialize half uniforms while virtual registers are
          * available, rather than asking the late parallel-copy pass to
          * encode an unwitnessed uniform-to-half move. */
         const bool copy_g18g_half_export =
            agx_is_g18g(&ctx->key->dev) && I->op == AGX_OPCODE_EXPORT &&
            I->src[s].size == AGX_SIZE_16;
         if (I->src[s].type == AGX_INDEX_UNIFORM && !select_uniform_flag &&
             !select_uniform_comparand && !select_uniform_alternative &&
             (s == first_uniform ||
              copy_minmax || copy_float_uniforms || copy_float_modifier ||
              copy_compare_uniform || copy_tex || copy_g18g_half_export ||
              !(native_integer || native_m5_float || uniform_imm_compare ||
                (s < 3 && keep_m5_float_uniform[s]) ||
                (s < 4 && keep_isel_uniform[s]) ||
                (m5_tex_uniform_pair && (s == 3 || s == 4)) ||
                (m5_fsel_uniform && s == 0) ||
                agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s,
                                         I->src[s].value, I->src[s].size)))) {
            agx_index idx = I->src[s];
            idx.abs = idx.neg = false;
            bool reused = false;
            for (unsigned m = 0; m < materialized; ++m) {
               if (agx_is_equiv(materialized_from[m], idx) &&
                   materialized_from[m].size == idx.size) {
                  agx_replace_src(I, s, materialized_to[m]);
                  reused = true;
                  break;
               }
            }
            if (reused)
               continue;
            if ((ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
                 agx_is_g18g(&ctx->key->dev)) &&
                idx.size == AGX_SIZE_16) {
               /* Native uniform copies fetch an aligned word. Extract the
                * requested half through an SSA split, so RA can coalesce
                * it without reading the neighbouring half as the value.
                */
               agx_index word = agx_mov(&b, agx_uniform(idx.value & ~1u,
                                                      AGX_SIZE_32));
               agx_instr *split = agx_split(&b, 2, word);
               split->dest[0] = agx_temp(ctx, AGX_SIZE_16);
               split->dest[1] = agx_temp(ctx, AGX_SIZE_16);
               agx_replace_src(I, s, split->dest[idx.value & 1]);
               if (materialized < ARRAY_SIZE(materialized_from)) {
                  materialized_from[materialized] = idx;
                  materialized_to[materialized++] = split->dest[idx.value & 1];
               }
            } else {
               agx_index copy = agx_mov(&b, idx);
               agx_replace_src(I, s, copy);
               if (materialized < ARRAY_SIZE(materialized_from)) {
                  materialized_from[materialized] = idx;
                  materialized_to[materialized++] = copy;
               }
            }
         }
      }
   }
   free(widened);

   agx3_fold_m5_constant_lods(ctx);

   /* Uniform materialization can also create a scalar LOD SSA source. */
   agx3_lower_m5_live_scalar_lods(ctx);
}
