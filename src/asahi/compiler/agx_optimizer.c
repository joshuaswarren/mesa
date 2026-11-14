/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/lut.h"
#include "util/macros.h"
#include "agx_builder.h"
#include "agx3_select_core.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/u_debug.h"
#include "agx_opcodes.h"

/* AGX peephole optimizer responsible for instruction combining. It operates in
 * a forward direction and a backward direction, in each case traversing in
 * source order. SSA means the forward pass satisfies the invariant:
 *
 *    Every def is visited before any of its uses.
 *
 * Dually, the backend pass satisfies the invariant:
 *
 *    Every use of a def is visited before the def.
 *
 * This means the forward pass can propagate modifiers forward, whereas the
 * backwards pass propagates modifiers backward. Consider an example:
 *
 *    1 = fabs 0
 *    2 = fround 1
 *    3 = fsat 1
 *
 * The forwards pass would propagate the fabs to the fround (since we can
 * lookup the fabs from the fround source and do the replacement). By contrast
 * the backwards pass would propagate the fsat back to the fround (since when
 * we see the fround we know it has only a single user, fsat).  Propagatable
 * instruction have natural directions (like pushforwards and pullbacks).
 *
 * We are careful to update the tracked state whenever we modify an instruction
 * to ensure the passes are linear-time and converge in a single iteration.
 *
 * Size conversions are worth special discussion. Consider the snippet:
 *
 *    2 = fadd 0, 1
 *    3 = f2f16 2
 *    4 = fround 3
 *
 * A priori, we can move the f2f16 in either direction. But it's not equal --
 * if we move it up to the fadd, we get FP16 for two instructions, whereas if
 * we push it into the fround, we effectively get FP32 for two instructions. So
 * f2f16 is backwards. Likewise, consider
 *
 *    2 = fadd 0, 1
 *    3 = f2f32 1
 *    4 = fround 3
 *
 * This time if we move f2f32 up to the fadd, we get FP32 for two, but if we
 * move it down to the fround, we get FP16 to too. So f2f32 is backwards.
 */

static bool
agx_is_fmov(agx_instr *def)
{
   return (def->op == AGX_OPCODE_FADD || def->op == AGX_OPCODE_HADD) &&
          agx_is_equiv(def->src[1], agx_negzero());
}

/* Compose floating-point modifiers with floating-point sources */

static agx_index
agx_compose_float_src(agx_index to, agx_index from)
{
   if (to.abs) {
      from.neg = false;
      from.abs = true;
   }

   from.neg ^= to.neg;

   return from;
}

/* AGX3's grounded fmin/fmax encoding requires the compared values to remain
 * tied to the selected values. Inlining an immediate into only one side (a
 * minifloat into the comparison, or raw bits that fit an integer immediate,
 * such as +0.0, into the selection) breaks that identity, and the resulting
 * general FSEL has no NaN-correct grounded form. Keep the shared SSA value
 * explicit on both sides through register allocation. */
static bool
agx_preserve_g17p_tied_fcmpsel(agx_context *ctx, agx_instr *I, unsigned src)
{
   return ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.emit_agx3d ||
           (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
            (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN))) &&
          I->op == AGX_OPCODE_FCMPSEL && src < 4 &&
          agx_is_equiv(I->src[src], I->src[src ^ 2]);
}

/*
 * A tied FCMPSEL whose condition is GTN or LTN is a minimum or a maximum, and
 * G17P encodes exactly that as a compact six-byte fsel whose second operand is
 * either a register pair or a single minifloat deposited into both the compare
 * and the select slot. The tie is what makes it a min/max, so an immediate has
 * to reach both slots or neither: folding it into just one leaves the packer a
 * general mixed-width select it has no encoding for, which is how max(x, 0.0)
 * -- normalize(), clamp(), saturate(), discard -- used to fail the shader
 * closed.  Sources 2 and 3 are not float sources in general (a select can
 * carry an integer), so the float interpretation is only taken here, where the
 * tie proves the operand is the compared float.
 *
 * Either pair can hold the constant: max(0.0, x) puts it in sources 0 and 2,
 * max(x, 0.0) in 1 and 3. Half-folding the first pair is worse than the
 * second, because agx_opt_compact_constants then sees the surviving reference
 * as the constant's only float use and narrows it to 16 bits, handing the
 * packer a 32-bit maximum against a 16-bit register.
 */
static bool
agx_g17p_tied_minmax(agx_context *ctx, const agx_instr *I)
{
   if (!(ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 && !I->invert_cond &&
         !I->saturate &&
         (I->fcond == AGX_FCOND_GTN || I->fcond == AGX_FCOND_LTN)))
      return false;

   /* agx_is_equiv() ignores modifiers, and a modifier on one half of a tie
    * would make the two operands different values. Require them identical. */
   for (unsigned i = 0; i < 2; ++i) {
      const agx_index a = I->src[i], b = I->src[i + 2];
      if (!agx_is_equiv(a, b) || a.size != b.size || a.abs != b.abs ||
          a.neg != b.neg || a.memory != b.memory)
         return false;
   }

   return true;
}

/*
 * (0 < x) ? x : 0 and (0 > x) ? x : 0 are exactly G17P's tied maximum
 * (x GTN 0) ? x : 0 and minimum (x LTN 0) ? x : 0: a NaN x compares false
 * either way, and so does a zero of either sign. The same holds for
 * (x < 0) ? x : 0 and (x > 0) ? x : 0. Fusion leaves the comparator zero as a
 * minifloat and the selected zero as a bit pattern, which are the same value.
 * Rewrite these clamps to the tied form, which has the compact native
 * encoding Apple uses for max(0.0, x).
 */
static void
agx_g17p_tie_zero_minmax(agx_context *ctx, agx_instr *I)
{
   if (!(ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 && !I->invert_cond &&
         !I->saturate &&
         (I->fcond == AGX_FCOND_LT || I->fcond == AGX_FCOND_GT)))
      return;

   for (unsigned s = 0; s < 4; ++s) {
      if (I->src[s].abs || I->src[s].neg)
         return;
   }

   const bool zero[4] = {
      agx_is_equiv(I->src[0], agx_immediate(0)),
      agx_is_equiv(I->src[1], agx_immediate(0)),
      agx_is_equiv(I->src[2], agx_immediate(0)),
      agx_is_equiv(I->src[3], agx_immediate(0)),
   };
   const bool crossed = zero[0] && zero[3] && !zero[1] &&
                        agx_is_equiv(I->src[1], I->src[2]);
   const bool tied = zero[1] && zero[3] && !zero[0] &&
                     agx_is_equiv(I->src[0], I->src[2]);
   if (!crossed && !tied)
      return;

   const agx_index x = crossed ? I->src[1] : I->src[0];
   if (x.type != AGX_INDEX_NORMAL || x.size != I->dest[0].size)
      return;

   const bool max = crossed ? I->fcond == AGX_FCOND_LT
                            : I->fcond == AGX_FCOND_GT;
   I->src[0] = I->src[2] = x;
   I->src[1] = I->src[3] = agx_immediate(0);
   I->fcond = max ? AGX_FCOND_GTN : AGX_FCOND_LTN;
}

static bool
agx_can_fold_half_uniform(const agx_instr *I, agx_instr **defs)
{
   if (I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_32 ||
       agx_channels(I->dest[0]) != 1 || I->saturate)
      return false;

   unsigned half_uniforms = 0;
   bool full_gpr = false;
   agx_foreach_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = src.type == AGX_INDEX_NORMAL ? defs[src.value] : NULL;
      if (def && agx_is_fmov(def) && !def->saturate && !src.abs && !src.neg &&
          !def->src[0].abs && !def->src[0].neg) {
         if (def->src[0].type == AGX_INDEX_UNIFORM ||
             def->src[0].type == AGX_INDEX_IMMEDIATE)
            src = def->src[0];
      }
      half_uniforms += src.type == AGX_INDEX_UNIFORM && src.size == AGX_SIZE_16;
      full_gpr |=
         (src.type == AGX_INDEX_NORMAL || src.type == AGX_INDEX_REGISTER) &&
         src.size == AGX_SIZE_32 && !src.memory && agx_channels(src) == 1;
   }
   return half_uniforms == 1 && full_gpr;
}

static void
agx_optimizer_fmov(agx_context *ctx, agx_instr **defs, agx_instr *ins)
{
   agx_foreach_ssa_src(ins, s) {
      agx_index src = ins->src[s];
      if (agx_preserve_g17p_tied_fcmpsel(ctx, ins, s))
         continue;
      agx_instr *def = defs[src.value];

      if (def == NULL)
         continue; /* happens for phis in loops */
      if (!agx_is_fmov(def))
         continue;
      if (def->saturate)
         continue;
      if (ins->op == AGX_OPCODE_FCMPSEL && s >= 2)
         continue;

      /* We can fold f2f32 into 32-bit instructions, but we can't fold f2f16
       * into 16-bit instructions, since the latter would implicitly promote to
       * a 32-bit instruction which is not exact.
       */
      assert(def->src[0].size == AGX_SIZE_32 ||
             def->src[0].size == AGX_SIZE_16);
      assert(src.size == AGX_SIZE_32 || src.size == AGX_SIZE_16);

      if (src.size == AGX_SIZE_16 && def->src[0].size == AGX_SIZE_32)
         continue;

      /* G17P and G18G do not have generally grounded f32 arithmetic over an f16
       * register.  FMUL has a few native mixed endpoints, but their combined
       * producer-wait/lifetime domain is intentionally narrow.  Preserve the
       * explicit exact widening for ordinary full-width arithmetic; native
       * patterns such as the scalar-half sqrt are emitted without an fmov and
       * therefore remain available. */
      if (ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           agx_is_g18g(&ctx->key->dev)) &&
          (ins->op == AGX_OPCODE_FADD || ins->op == AGX_OPCODE_FMUL ||
           ins->op == AGX_OPCODE_FFMA) &&
          src.size == AGX_SIZE_32 && def->src[0].size == AGX_SIZE_16 &&
          !(def->src[0].type == AGX_INDEX_UNIFORM && !def->src[0].memory &&
            !def->src[0].abs && !def->src[0].neg && !src.abs && !src.neg &&
            agx_can_fold_half_uniform(ins, defs)))
         continue;

      /* M5's f32 add takes at most one half register (f32 + f16, or a half
       * plus a minifloat immediate); two half registers into an f32 result
       * have no encoding. Keep the second widening explicit. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          ins->op == AGX_OPCODE_FADD && ins->nr_srcs == 2 &&
          ins->dest[0].size == AGX_SIZE_32 && src.size == AGX_SIZE_32 &&
          def->src[0].type == AGX_INDEX_NORMAL &&
          def->src[0].size == AGX_SIZE_16) {
         const agx_index other = ins->src[1 - s];
         if (other.type == AGX_INDEX_NORMAL && other.size == AGX_SIZE_16)
            continue;
      }

      /* The reciprocal square root takes one width for source and result
       * (agx3_alu_hrsqrt and agx3_alu_f32rsqrt_mod). Folding an f16 widening
       * into an f32 rsqrt would leave a half register feeding a word result,
       * which has no encoding and fails the whole shader (an elementwise
       * half-precision kernel). Keep the explicit widening. */
      if (ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
          ins->op == AGX_OPCODE_RSQRT && src.size == AGX_SIZE_32 &&
          def->src[0].size == AGX_SIZE_16)
         continue;

      /* FLTNORM and RGB10A2 packing have only captured f32 sources. Keep
       * the explicit widening used by linked half-width fragment outputs
       * before normalized packing. */
      if (ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
          (ins->op == AGX_OPCODE_FLTNORM_AGX3 ||
           ins->op == AGX_OPCODE_PACK_RGB10A2_AGX3) &&
          src.size == AGX_SIZE_32 &&
          def->src[0].size == AGX_SIZE_16)
         continue;

      /* G17P's grounded f32 multiply into a half destination takes plain
       * registers only. Keep |x| or -x as its own move in front of such a
       * multiply rather than inventing an unencodable modified form. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          ins->op == AGX_OPCODE_FMUL && ins->dest[0].size == AGX_SIZE_16 &&
          src.size == AGX_SIZE_32 && (def->src[0].abs || def->src[0].neg))
         continue;

      /* The half-width TRUNC/ROUNDEVEN forms are grounded for plain
       * registers only; the packer refuses -x or |x| there, which fails the
       * whole pipeline (vkQuake's ssao_evaluate: roundeven of a negated
       * half). Keep the modifier on its own move in front instead. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (ins->op == AGX_OPCODE_TRUNC || ins->op == AGX_OPCODE_ROUNDEVEN) &&
          src.size == AGX_SIZE_16 && (def->src[0].abs || def->src[0].neg))
         continue;

      ins->src[s] = agx_compose_float_src(src, def->src[0]);
   }
}

static bool
image_write_source_can_be_immediate(agx_instr *I, unsigned s)
{
   bool block = I->op == AGX_OPCODE_BLOCK_IMAGE_STORE;
   assert(I->op == AGX_OPCODE_IMAGE_WRITE || block);

   /* LOD can always be immediate. Actually, it's just zero so far, we don't
    * support nonzero LOD for images yet.
    */
   if (s == 2 && !block)
      return true;

   /* If the "bindless" source (source 3) is an immediate, it means we don't
    * have a bindless image, instead we have a texture state index. We're
    * allowed to have immediate texture state registers (source 4). However,
    * we're not allowed to have immediate bindless offsets (also source 4).
    */
   unsigned base = block ? 0 : 3;
   bool is_texture_state = (I->src[base].type == AGX_INDEX_IMMEDIATE);
   if (s == (base + 1) && is_texture_state)
      return true;

   /* Otherwise, must be from a register */
   return false;
}

/* M5 FCMPSEL also selects a minifloat immediate (fself32), which is how
 * native G17G code selects constants such as 1.0. Fold one whose bits
 * do not fit the integer immediate, opposite a selected register only: the
 * two-immediate form takes raw bits. Compare-select fusion creates most of
 * these selects after immediate inlining, so run it after fusing as well. */
static void
agx_optimizer_minifloat_select(agx_context *ctx, agx_instr **defs,
                               agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->agx3_reg_float_consts ||
       I->op != AGX_OPCODE_FCMPSEL || I->nr_srcs != 4 ||
       I->fcmpsel_minifloat_select || I->saturate ||
       I->dest[0].size != AGX_SIZE_32)
      return;

   for (unsigned s = 2; s < 4; ++s) {
      const agx_index src = I->src[s], other = I->src[s ^ 1];
      if (src.type != AGX_INDEX_NORMAL || src.abs || src.neg ||
          agx_preserve_g17p_tied_fcmpsel(ctx, I, s))
         continue;

      const agx_instr *def = defs[src.value];
      if (!def || def->op != AGX_OPCODE_MOV_IMM ||
          def->dest[0].size != AGX_SIZE_32 || def->imm <= UINT8_MAX ||
          !agx_minifloat_exact(uif(def->imm)))
         continue;

      if (other.type == AGX_INDEX_IMMEDIATE ||
          (other.type == AGX_INDEX_NORMAL && defs[other.value] &&
           defs[other.value]->op == AGX_OPCODE_MOV_IMM))
         return;

      I->src[s] = agx_immediate_f(uif(def->imm));
      I->fcmpsel_minifloat_select = true;
      return;
   }
}

static void
agx_optimizer_inline_imm(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (I->op == AGX_OPCODE_MATMAD8_AGX3 ||
       I->op == AGX_OPCODE_MATMAD16_AGX3)
      return;
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      if (I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 ||
          I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
          I->op == AGX_OPCODE_SCS_TOUCH_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_TOUCH_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3)
         continue;

      /* Either compared source of a tied minimum/maximum may be inlined,
       * because the fold below carries it into its partner as well and so
       * keeps the tie. */
      if (agx_preserve_g17p_tied_fcmpsel(ctx, I, s) &&
          !(s < 2 && agx_g17p_tied_minmax(ctx, I)))
         continue;
      if (src.neg)
         continue;
      if (I->op == AGX_OPCODE_ALLOCATE_UVB_M5 ||
          I->op == AGX_OPCODE_STORE_UVB_M5 ||
          I->op == AGX_OPCODE_STORE_OBJECT_CHILDCOUNT_M5)
         continue;

      /* AGX3 reductions and scans read a register even when every active
       * lane contributes the same constant. Uniform-address atomics can
       * introduce exclusive scans of constants; inlining those constants
       * leaves an operand the subgroup encoder cannot encode. */
      if (ctx->key && agx_isa_is_agx3(ctx->key->dev.encoding) &&
          (I->op == AGX_OPCODE_SIMD_REDUCE ||
           I->op == AGX_OPCODE_SIMD_PREFIX ||
           I->op == AGX_OPCODE_QUAD_REDUCE ||
           I->op == AGX_OPCODE_QUAD_PREFIX))
         continue;

      /* Coverage-stack address/data operands, traversal addresses, and UVB
       * allocation/descriptor and child-count operands have register forms
       * only. Keep constant materialization intact. */
      if (I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_INTERSECT_RAY_M5 ||
          I->op == AGX_OPCODE_ALLOCATE_UVB_M5 ||
          I->op == AGX_OPCODE_STORE_UVB_M5 ||
          I->op == AGX_OPCODE_STORE_OBJECT_CHILDCOUNT_M5)
         continue;

      /* PACK_SMPIR-to-GPR takes a half register, not an integer literal. */
      if (I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3)
         continue;

      /* G17P's native subtraction takes two registers. Instruction selection
       * deliberately materializes an immediate minuend for `c - x`; folding
       * it back here would leave immediate-plus-negated-register, whose iadd
       * modifier means narrow sign extension rather than arithmetic negate.
       */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_IADD &&
          ((s == 0 && I->src[1].neg) || (s == 1 && I->src[0].neg)))
         continue;

      /* G17P has no iadd of two immediates. The ushr count copy
       * `iadd #0, count` must keep its register count. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_IADD && I->nr_srcs == 2 &&
          I->src[1 - s].type == AGX_INDEX_IMMEDIATE)
         continue;

      /* G17P's arithmetic shift right takes its shifted VALUE in a register.
       * Only the count (source 1) has an immediate form -- `sra` is what
       * `metal-nt -arch applegpu_g17p` emits for `(-7) >> x`, and it first
       * spends an LDIMM to put the -7 in a register (`1cf922fe1e0cff0f`
       * ldimm then `a7015408000400d8e200` sra reg,reg).
       * Folding the constant into source 0 leaves a shape with no encoding
       * at all -- agx3_select_core.c's AGX3_SEL_OP_ASR requires two plain
       * 32-bit registers -- and the packer then refuses the whole shader.
       * Keep the MOV so selection reproduces Apple's ldimm+sra pair. */
      if (ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
          I->op == AGX_OPCODE_ASR && s == 0)
         continue;

      /* G18G's half-precision compare-select is grounded only with every
       * source in a half register (agx3_alu_fcmp_select_half_gpu1). An
       * inlined immediate leaves a form the packer cannot encode, so keep
       * the constants in registers (alacritty's discard test, bg.a == 0).
       * M6 only. */
      if (ctx->key && agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_FCMPSEL && I->dest[0].size == AGX_SIZE_16)
         continue;

      /* AGX3's FMA takes a minifloat for one multiplicand (beside an optional
       * minifloat addend) and has no form that multiplies two immediates, so
       * the packer refused the whole shader (Cyberpunk 2077's benchmark
       * vertex shaders: ffma r32, #k16, #k16, r32). Keep the second constant
       * multiplicand in its register. */
      if (ctx->key && agx_encoding_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FFMA && s < 2 &&
          I->src[1 - s].type == AGX_INDEX_IMMEDIATE)
         continue;

      agx_instr *def = defs[src.value];
      if (!def || def->op != AGX_OPCODE_MOV_IMM)
         continue;

      uint8_t value = def->imm;
      uint16_t value_u16 = def->imm;

      /* G17P's sample names its explicit level and its implicit-LOD bias in
       * the same immediate operand, and the packer decodes that operand as a
       * minifloat.  agx_is_float_src() reports the whole instruction, which
       * is integer-typed, so name this one source explicitly rather than
       * folding a float bit pattern in as an integer. */
      bool float_src =
         agx_is_float_src(I, s) ||
         (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_TEXTURE_SAMPLE && s == 1 &&
          (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS));

      if (I->op == AGX_OPCODE_ST_TILE && s == 0)
         continue;
      /* G17P tile stores only encode the whole-pixel 0xff mask as an
       * immediate. Keep all other masks, including zero coverage, in the
       * 16-bit register already produced by instruction selection.
       */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_ST_TILE && s == 1 && def->imm != 0xff)
         continue;
      /* Tile loads likewise: a per-sample load (multisample framebuffer
       * fetch) names its sample mask in a register, as Apple's multisample
       * LLDCol does; only the whole-pixel 0xff mask is an immediate. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_LD_TILE && s == 0 && def->imm != 0xff)
         continue;
      if (I->op == AGX_OPCODE_ZS_EMIT && s != 0)
         continue;
      /* G17P names an explicit level and an implicit-LOD bias in the same
       * 12-bit immediate operand, so both fold; every other sample source
       * has to stay a register. */
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && s != 4 &&
          !(ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && s == 1 &&
            (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
             I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS)))
         continue;
      /* AGX3 sampler indices past 15 have only a register form. */
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && s == 4 &&
          ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && def->imm > 15)
         continue;
      if ((I->op == AGX_OPCODE_DEVICE_STORE ||
           I->op == AGX_OPCODE_LOCAL_STORE || I->op == AGX_OPCODE_ATOMIC ||
           I->op == AGX_OPCODE_LOCAL_ATOMIC) &&
          s != 2)
         continue;
      if (I->op == AGX_OPCODE_ST_VARY && s != 0)
         continue;
      if ((I->op == AGX_OPCODE_LOCAL_LOAD || I->op == AGX_OPCODE_DEVICE_LOAD ||
           I->op == AGX_OPCODE_STACK_STORE) &&
          s != 1)
         continue;
      if (I->op == AGX_OPCODE_SPLIT)
         continue;

      /*
       * AGX3's immediate-lane shuffle holds the lane in a seven-bit field,
       * so every lane a subgroup can name folds in.  Only an out-of-range
       * constant -- undefined behaviour, but not a reason to refuse a shader
       * -- has to keep its mov and go through the register-lane form.
       */
      if (I->op == AGX_OPCODE_SHUFFLE && s == 1 && ctx->key &&
          ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && def->imm > 0x7f)
         continue;

      if ((I->op == AGX_OPCODE_IMAGE_WRITE ||
           I->op == AGX_OPCODE_BLOCK_IMAGE_STORE) &&
          !image_write_source_can_be_immediate(I, s))
         continue;

      /* The minifloat form of an M5 select is folded separately below. */
      if (I->op == AGX_OPCODE_FCMPSEL && I->fcmpsel_minifloat_select && s >= 2)
         continue;

      if (float_src) {
         if (ctx->agx3_reg_float_consts)
            continue;

         bool fp16 = (def->dest[0].size == AGX_SIZE_16);
         assert(fp16 || (def->dest[0].size == AGX_SIZE_32));

         float f = fp16 ? _mesa_half_to_float(def->imm) : uif(def->imm);
         if (!agx_minifloat_exact(f))
            continue;

         /* Both halves of a tied minimum/maximum, or neither. */
         if (s < 2 && agx_g17p_tied_minmax(ctx, I))
            I->src[s + 2] = agx_immediate_f(f);

         I->src[s] = agx_immediate_f(f);
      } else if (value == def->imm) {
         I->src[s] = agx_immediate(value);
      } else if (value_u16 == def->imm && agx_allows_16bit_immediate(I)) {
         I->src[s] = agx_abs(agx_immediate(value_u16));
      } else if ((I->op == AGX_OPCODE_IADD || I->op == AGX_OPCODE_IMAD) &&
                 s == agx_negate_src_index(I)) {
         unsigned bits = agx_size_align_16(def->dest[0].size) * 16;
         uint64_t mask = BITFIELD64_MASK(bits);
         uint64_t negated = (-def->imm) & mask;
         value = negated;

         /* Try to negate the immediate */
         if (value == negated) {
            I->src[s] = agx_neg(agx_immediate(value));
         }
      }
   }
}

/*
 * Fuse not into and/or/xor. Specifically, acts on not and fuses:
 *
 *    not(and(x, y) -> nand(x, y)
 *    not(or(x, y) -> nor(x, y)
 *    not(xor(x, y) -> xnor(x, y)
 */
static bool
agx_optimizer_not(agx_context *ctx, agx_instr *I, agx_instr *use)
{
   /* Check for bit op and use of not op */
   if (I->op != AGX_OPCODE_BITOP || use->op != AGX_OPCODE_NOT)
      return false;

   /* G17P has native word-wide NAND/NOR/XNOR, but the half-width forms
    * still need the separate complement. Do not create an unencodable
    * truth table or disable the supported word-wide fusion. */
   const unsigned truth_table = I->truth_table ^ 0xf;
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->dest[0].size != AGX_SIZE_32 &&
       truth_table != AGX3_SEL_TT_AND && truth_table != AGX3_SEL_TT_OR &&
       truth_table != AGX3_SEL_TT_XOR && truth_table != AGX3_SEL_TT_NOT)
      return false;

   /* Remap operation to the appropriate one */
   I->truth_table ^= 0xF;
   I->dest[0] = use->dest[0];

   return true;
}

static bool
agx_optimizer_fmov_rev(agx_context *ctx, agx_instr *I, agx_instr *use)
{
   if (!agx_is_fmov(use))
      return false;
   if (use->src[0].neg || use->src[0].abs)
      return false;

   /* G17P's native saturate is a standalone f32add.ep2 of the value and -0.
    * G18G keeps the same rule for its native narrowing saturate.
    * Pulling that modifier into its producer invents saturated forms for
    * operations such as fmul and transcendental functions. Keep the exact
    * Apple sequence intact for every producer. */
   if (ctx->key &&
       (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        agx_is_g18g(&ctx->key->dev)) &&
       use->saturate)
      return false;

   /* M5 (AGX3C) encodes reciprocal and reciprocal square root without a
    * saturating form, so a saturate folded into one made the shader
    * unencodable (Skyrim SE's fragment shaders clamp a reciprocal). Keep the
    * separate saturating move there. */
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       use->saturate &&
       (I->op == AGX_OPCODE_RCP || I->op == AGX_OPCODE_RSQRT))
      return false;

   /* AGX3's projective reciprocal and multiply are the native f32rcp and
    * f32mul.ep4 forms that carry a coefficient index. Their only grounded
    * destination is a plain 32-bit register and neither has a saturating
    * form, so the packer refuses anything else and the whole shader with it.
    * AGX3B already keeps the move (only FMUL narrows there, and no producer
    * absorbs a saturate); AGX3C did not, so on G17S every perspective
    * fragment input narrowed to 16 bits -- each mediump varying of a GLES
    * shader -- was refused as `persp_mul_agx3 d16, r32, r32` and its draws
    * were skipped (es2gears, Firefox WebRender text). Keep the explicit
    * f32-to-f16 or saturating move on every encoding. */
   if (I->op == AGX_OPCODE_PERSP_MUL_AGX3 ||
       I->op == AGX_OPCODE_PERSP_RCP_AGX3)
      return false;

   /* We can fold f2f16 into 32-bit instructions, but we can't fold f2f32 into
    * 16-bit instructions, since the latter would implicitly promote to a 32-bit
    * instruction which is not exact.
    */
   assert(use->dest[0].size == AGX_SIZE_32 || use->dest[0].size == AGX_SIZE_16);
   assert(I->dest[0].size == AGX_SIZE_32 || I->dest[0].size == AGX_SIZE_16);

   if (I->dest[0].size == AGX_SIZE_16 && use->dest[0].size == AGX_SIZE_32)
      return false;

   /* G17P and G18G only have a corpus-grounded direct f32 operation with an f16
    * destination for FMUL.  Keep an explicit f32-to-f16 move after ADD/FMA
    * and the remaining float operations so each rounds exactly once through
    * the native narrowing form instead of creating an unencodable mixed-width
    * producer. */
   if (ctx->key &&
       (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        agx_is_g18g(&ctx->key->dev)) &&
       I->dest[0].size == AGX_SIZE_32 && use->dest[0].size == AGX_SIZE_16 &&
       I->op != AGX_OPCODE_FMUL)
      return false;

   /* M5 (AGX3C): an f16-destination ADD/FMA fed by an immediate has no
    * grounded encoding -- the packer refuses `ffma d16, r32, #imm16, r32`
    * (RayTracedGL1's IndirectInit raygen). Keep the explicit narrowing move in
    * that case only; register-only producers still fold. */
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       I->dest[0].size == AGX_SIZE_32 && use->dest[0].size == AGX_SIZE_16 &&
       I->op != AGX_OPCODE_FMUL) {
      agx_foreach_src(I, s) {
         if (I->src[s].type == AGX_INDEX_IMMEDIATE)
            return false;
      }
   }

   /* ...and that narrowing FMUL takes plain registers only: a multiply that
    * already carries |x| or -x on a source keeps its explicit f32-to-f16
    * move (Skia's conic/quad edge shaders narrow abs(f) * rsqrt(g)). */
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_FMUL && I->dest[0].size == AGX_SIZE_32 &&
       use->dest[0].size == AGX_SIZE_16 &&
       (I->src[0].abs || I->src[0].neg || I->src[1].abs || I->src[1].neg))
      return false;

   /* M5 (AGX3C): the rounding/log/exp funops have no half-destination form
    * over a word source (agx3_alu_unary_m5 refuses it), and rsqrt and the
    * gradients take a single width. Folding the narrowing move into one made
    * exp/log/floor/ceil/trunc/rsqrt shaders unencodable and the draw was
    * skipped, so keep the explicit f32-to-f16 move and compute at word
    * width. */
   if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       I->dest[0].size == AGX_SIZE_32 && use->dest[0].size == AGX_SIZE_16 &&
       I->src[0].size == AGX_SIZE_32 &&
       (I->op == AGX_OPCODE_EXP2 || I->op == AGX_OPCODE_LOG2 ||
        I->op == AGX_OPCODE_FLOOR || I->op == AGX_OPCODE_CEIL ||
        I->op == AGX_OPCODE_TRUNC || I->op == AGX_OPCODE_ROUNDEVEN ||
        I->op == AGX_OPCODE_RSQRT || I->op == AGX_OPCODE_DFDX ||
        I->op == AGX_OPCODE_DFDY))
      return false;

   /* saturate(saturate(x)) = saturate(x) */
   I->saturate |= use->saturate;
   I->dest[0] = use->dest[0];
   return true;
}

static bool
agx_icond_is_unsigned(enum agx_icond cond)
{
   switch (cond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_ULT:
   case AGX_ICOND_UGT:
      return true;

   case AGX_ICOND_SEQ:
   case AGX_ICOND_SLT:
   case AGX_ICOND_SGT:
      return false;
   }

   UNREACHABLE("invalid condition");
}

static bool
agx_supports_zext(agx_instr *I, unsigned s)
{
   switch (I->op) {
   case AGX_OPCODE_IADD:
   case AGX_OPCODE_IMAD:
   case AGX_OPCODE_INTL:
   case AGX_OPCODE_FFS:
   case AGX_OPCODE_BITREV:
   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_EXTR:
   case AGX_OPCODE_BITOP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_ICMP_BALLOT:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
      return true;

   case AGX_OPCODE_ICMP:
   case AGX_OPCODE_ICMPSEL:
      /* Only the comparisons can be extended, not the selection. And we can
       * only zero-extend with unsigned comparison. Presumably the hardware
       * sign-extends with signed comparisons but we don't handle that yet.
       */
      return (s < 2) && agx_icond_is_unsigned(I->icond);

   default:
      return false;
   }
}

bool
agx_opt_collect_split(agx_context *ctx)
{
   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   agx_index *aliases = calloc(ctx->alloc, sizeof(*aliases));
   bool progress = false;
   if (!defs || !aliases)
      goto done;
   for (unsigned i = 0; i < ctx->alloc; ++i)
      aliases[i] = agx_null();
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_SPLIT || I->nr_srcs != 1 || I->saturate ||
          I->src[0].type != AGX_INDEX_NORMAL || I->src[0].memory ||
          I->src[0].abs || I->src[0].neg)
         continue;
      agx_instr *collect = defs[I->src[0].value];
      if (!collect || collect->op != AGX_OPCODE_COLLECT ||
          collect->nr_dests != 1 || collect->saturate ||
          collect->dest[0].memory ||
          collect->dest[0].size != I->src[0].size ||
          collect->nr_srcs != I->nr_dests ||
          agx_channels(collect->dest[0]) != collect->nr_srcs ||
          agx_channels(I->src[0]) != I->nr_dests)
         continue;
      bool hinted = false;
      agx_foreach_ssa_dest(I, d) {
         const agx_index dest = I->dest[d], src = collect->src[d];
         if (dest.size != I->src[0].size || dest.memory ||
             agx_channels(dest) != 1 || src.type != AGX_INDEX_NORMAL ||
             src.size != dest.size || src.memory || src.abs || src.neg ||
             agx_channels(src) != 1)
            continue;
         aliases[dest.value] = src;
         if (!hinted) {
            if (!ctx->collect_split_hints.data)
               util_dynarray_init(&ctx->collect_split_hints, ctx);
            agx_builder b = agx_init_builder(ctx, agx_before_instr(collect));
            agx_instr *hint = agx_alloc_instr(
               &b, AGX_OPCODE_COLLECT, 1, collect->nr_srcs);
            hint->dest[0] = collect->dest[0];
            agx_foreach_src(collect, s)
               hint->src[s] = collect->src[s];
            util_dynarray_append(&ctx->collect_split_hints, hint);
            hinted = true;
         }
      }
   }
   for (unsigned i = 0; i < ctx->alloc; ++i) {
      if (agx_is_null(aliases[i]))
         continue;
      agx_index root = aliases[i];
      unsigned steps = 0;
      while (!agx_is_null(aliases[root.value])) {
         if (++steps >= ctx->alloc)
            goto done;
         root = aliases[root.value];
      }
      unsigned value = i;
      while (!agx_is_null(aliases[value])) {
         unsigned next = aliases[value].value;
         aliases[value] = root;
         value = next;
      }
   }
   util_dynarray_foreach(&ctx->collect_split_hints, agx_instr *, hint) {
      agx_instr *I = *hint;
      agx_foreach_ssa_src(I, s) {
         agx_index src = I->src[s], alias = aliases[src.value];
         if (!agx_is_null(alias) && src.size == alias.size && !src.memory &&
             agx_channels(src) == 1)
            agx_replace_src(I, s, alias);
      }
   }
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s) {
         const agx_index src = I->src[s], alias = aliases[src.value];
         if (agx_is_null(alias) || src.size != alias.size || src.memory ||
             agx_channels(src) != 1)
            continue;
         agx_replace_src(I, s, alias);
         progress = true;
      }
   }
done:
   free(aliases);
   free(defs);
   return progress;
}

static void
agx_optimizer_copyprop(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = defs[src.value];

      if (def == NULL)
         continue; /* happens for phis in loops */

      /* Preserve a pointer assembled from adjacent uniform words instead of
       * materializing its two words in registers. M5 loads can address the
       * complete uniform file with an aligned 64-bit base. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          I->op == AGX_OPCODE_DEVICE_LOAD && s == 0 &&
          src.size == AGX_SIZE_64 && agx_channels(src) == 1 &&
          !src.abs && !src.neg && !src.memory &&
          def->op == AGX_OPCODE_COLLECT && def->nr_srcs == 2 &&
          def->nr_dests == 1 && def->dest[0].size == AGX_SIZE_64 &&
          agx_channels(def->dest[0]) == 1 && !def->dest[0].memory &&
          !def->saturate &&
          debug_get_bool_option("ASAHI_M5_HIGH_UNIFORM_LOAD", true)) {
         const agx_index lo = def->src[0], hi = def->src[1];
         if (lo.type == AGX_INDEX_UNIFORM && hi.type == AGX_INDEX_UNIFORM &&
             lo.size == AGX_SIZE_32 && hi.size == AGX_SIZE_32 &&
             agx_channels(lo) == 1 && agx_channels(hi) == 1 &&
             !lo.abs && !lo.neg && !lo.memory &&
             !hi.abs && !hi.neg && !hi.memory &&
             !(lo.value & 3) && hi.value == lo.value + 2 &&
             agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s,
                                       lo.value, AGX_SIZE_64)) {
            agx_replace_src(I, s, agx_uniform(lo.value, AGX_SIZE_64));
            continue;
         }
      }
      if (def->op != AGX_OPCODE_MOV)
         continue;

      /* At the moment, not all instructions support size conversions. Notably
       * RA pseudo instructions don't handle size conversions. This should be
       * refined in the future.
       */
      if (def->src[0].size != src.size &&
          !(def->src[0].size < src.size && agx_supports_zext(I, s)))
         continue;

      /* M5's mixed-width compare path does not preserve a NIR unsigned
       * half-to-word conversion. A native SCS u16 load correctly returned
       * 0x0000c56d, but comparison against a folded half operand rejected it;
       * the same compiled test passed below bit15. Keep the explicit widening
       * before comparing. This is the value conversion required by NIR, not
       * an alteration of the SCS load format or its tested extension bit. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          def->src[0].size == AGX_SIZE_16 && src.size == AGX_SIZE_32 && s < 2 &&
          (I->op == AGX_OPCODE_ICMP || I->op == AGX_OPCODE_ICMPSEL))
         continue;

      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          I->op == AGX_OPCODE_BITOP && I->dest[0].size == AGX_SIZE_32 &&
          def->src[0].size == AGX_SIZE_16 && src.size == AGX_SIZE_32 &&
          def->src[0].type != AGX_INDEX_IMMEDIATE)
         continue;

      /* AGX2 can carry the zero-extension modifier on integer arithmetic and
       * compare-select, but the grounded G17P forms are width-matched.
       * Preserve the explicit 16-to-32 move so the AGX3 packer emits its
       * witnessed COPY-low plus CLR-high sequence first. G18G keeps it too.
       * j873 reached this through the Gen10 texture-descriptor crawl, whose
       * level field is a half extract: folding the widening left an
       * `and d32, h5, r8` the packer has no encoding for, and the whole
       * fragment program -- the depth blitter's -- was dropped. */
      if (ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           agx_is_g18g(&ctx->key->dev)) &&
          def->src[0].size == AGX_SIZE_16 && src.size == AGX_SIZE_32 &&
          (I->op == AGX_OPCODE_IADD || I->op == AGX_OPCODE_IMAD ||
           I->op == AGX_OPCODE_ICMP || I->op == AGX_OPCODE_ICMPSEL ||
           I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL ||
           (I->op == AGX_OPCODE_EXTR && s < 2) ||
           I->op == AGX_OPCODE_BITOP || I->op == AGX_OPCODE_BALLOT ||
           I->op == AGX_OPCODE_QUAD_BALLOT ||
           I->op == AGX_OPCODE_ICMP_BALLOT ||
           I->op == AGX_OPCODE_ICMP_QUAD_BALLOT))
         continue;

      /* M5's funnel extract takes two word registers. Only a zero-extended
       * low half shifted out entirely (an immediate shift of 16 or more,
       * which legalization turns into a bitfield insert) may stay narrow;
       * keep the explicit extension move for every other half source. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          def->src[0].size == AGX_SIZE_16 && src.size == AGX_SIZE_32 &&
          I->op == AGX_OPCODE_EXTR && s < 2 &&
          !(s == 0 && I->src[2].type == AGX_INDEX_IMMEDIATE &&
            I->src[2].value >= 16 && I->src[2].value <= 31))
         continue;

      /* The AGX3C saturating add has only a witnessed 32-bit register form
       * (iadd/isub with f53, agx3_pack_binary.c); a zero-extended 16-bit
       * source is refused by the packer. Keep the explicit extension move,
       * as the AGX3B rule above does for all of these ops. */
      if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          def->src[0].size == AGX_SIZE_16 && src.size == AGX_SIZE_32 &&
          I->op == AGX_OPCODE_IADD && I->saturate)
         continue;

      /* Optimize split(64-bit uniform) so we can get better copyprop of the
       * 32-bit uniform parts. This helps reduce moves with 64-bit uniforms.
       */
      if (I->op == AGX_OPCODE_SPLIT && def->src[0].type == AGX_INDEX_UNIFORM &&
          src.size == AGX_SIZE_64 && I->dest[0].size == AGX_SIZE_32) {

         assert(I->nr_dests == 2 && "decomposing a 64-bit scalar");
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

         agx_index lo = def->src[0];
         lo.size = AGX_SIZE_32;

         agx_index hi = lo;
         hi.value += 2 /* half of 64-bits = 32-bits = 2 x 16-bits */;

         defs[I->dest[0].value] = agx_mov_to(&b, I->dest[0], lo);
         defs[I->dest[1].value] = agx_mov_to(&b, I->dest[1], hi);

         agx_remove_instruction(I);
         continue;
      }

      /* Immediate inlining happens elsewhere */
      if (def->src[0].type == AGX_INDEX_IMMEDIATE)
         continue;

      /* Native G17S texel fetches can read a static descriptor directly
       * from CProg uniforms. Keep it through copy propagation, then let
       * uniform-source validation enforce the witnessed range/alignment. */
      const bool m5_texture_uniform =
         ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
         I->op == AGX_OPCODE_TEXTURE_LOAD && s == 3 &&
         def->src[0].type == AGX_INDEX_UNIFORM &&
         debug_get_bool_option("ASAHI_M5_UNIFORM_TEXTURE_OPERANDS", true) &&
         agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s,
                                  def->src[0].value, def->src[0].size);

      /* An M5 fragment or compute SMP reads its texture and sampler handles
       * from a pair of aligned 64-bit uniforms (native G17G compiler
       * argument-buffer samples); forwarded below only as a whole pair. */
      const bool m5_sample_uniform =
         ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
         (ctx->stage == MESA_SHADER_FRAGMENT ||
          ctx->stage == MESA_SHADER_COMPUTE) && !ctx->is_preamble &&
         I->op == AGX_OPCODE_TEXTURE_SAMPLE && (s == 3 || s == 4) &&
         def->src[0].type == AGX_INDEX_UNIFORM;

      /* ALU instructions cannot take 64-bit. G17P SMP's texture/sampler index
       * operands can (agx_lower_uniform_sources
       * copies back any placement the native form cannot encode). */
      if (def->src[0].size == AGX_SIZE_64 && !m5_texture_uniform &&
          !m5_sample_uniform &&
          !(I->op == AGX_OPCODE_DEVICE_LOAD && s == 0) &&
          !(I->op == AGX_OPCODE_DEVICE_STORE && s == 1) &&
          !(I->op == AGX_OPCODE_ATOMIC && s == 1) &&
          !(I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 && s < 2) &&
          !((I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 && s == 0) ||
            (I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3 && s == 1) ||
            (I->op == AGX_OPCODE_INTERSECT_RAY_M5 && s == 1)) &&
          !(I->op == AGX_OPCODE_TEXTURE_SAMPLE && (s == 3 || s == 4) &&
            ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            def->src[0].type == AGX_INDEX_UNIFORM))
         continue;

      /* Forward a descriptor uniform into SMP only when the pair it forms is
       * a native one. Otherwise agx_lower_uniform_sources would copy it back
       * per instruction, where the shared SSA move serves every sample. */
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && (s == 3 || s == 4) &&
          def->src[0].type == AGX_INDEX_UNIFORM) {
         agx_index pair[2] = {I->src[3], I->src[4]};
         pair[s - 3] = def->src[0];
         agx_index *other = &pair[4 - s];
         if (other->type == AGX_INDEX_NORMAL) {
            agx_instr *odef = defs[other->value];
            if (odef && odef->op == AGX_OPCODE_MOV &&
                odef->src[0].type == AGX_INDEX_UNIFORM &&
                odef->src[0].size == other->size)
               *other = odef->src[0];
         }
         if (agx3_tex_uniform_pair_unwitnessed(ctx, I, pair[0], pair[1]))
            continue;
         if (m5_sample_uniform) {
            bool both = true;
            for (unsigned j = 0; j < 2; ++j) {
               both &= pair[j].type == AGX_INDEX_UNIFORM &&
                       pair[j].size == AGX_SIZE_64 && !(pair[j].value & 3) &&
                       pair[j].value <= 252 && !pair[j].abs && !pair[j].neg;
            }
            if (!both)
               continue;
         }
      }

      agx_replace_src(I, s, def->src[0]);

      /* If we are zero-extending into an instruction that distinguishes sign
       * and zero extend, make sure we pick zero-extend.
       */
      if (def->src[0].size < src.size &&
          (I->op == AGX_OPCODE_IMAD || I->op == AGX_OPCODE_IADD)) {

         assert(agx_supports_zext(I, s));
         I->src[s].abs = true;
      }
   }
}

/*
 * Fuse conditions into if. Specifically, acts on if_icmp and fuses:
 *
 *    if_icmp(cmp(x, y, *), 0, ne/eq) -> if_cmp(x, y, *)
 */
static void
agx_optimizer_if_cmp(agx_instr **defs, agx_instr *I)
{
   /* Check for unfused if */
   if (!agx_is_equiv(I->src[1], agx_zero()) || I->icond != AGX_ICOND_UEQ ||
       I->src[0].type != AGX_INDEX_NORMAL)
      return;

   /* Check for condition */
   agx_instr *def = defs[I->src[0].value];
   if (def->op != AGX_OPCODE_ICMP && def->op != AGX_OPCODE_FCMP)
      return;

   /* Fuse */
   I->src[0] = def->src[0];
   I->src[1] = def->src[1];
   I->invert_cond = def->invert_cond ^ !I->invert_cond;

   if (def->op == AGX_OPCODE_ICMP) {
      I->op = AGX_OPCODE_IF_ICMP;
      I->icond = def->icond;
   } else {
      I->op = AGX_OPCODE_IF_FCMP;
      I->fcond = def->fcond;
   }
}

/*
 * Fuse invert into if. Acts on if_icmp and fuses:
 *
 *    if_icmp(xor(x, 1), 0, ne) -> if_cmp(x, 0, eq)
 */
static void
agx_optimizer_if_not(agx_instr **defs, agx_instr *I)
{
   /* Check for unfused if */
   if (!agx_is_equiv(I->src[1], agx_zero()) || I->icond != AGX_ICOND_UEQ ||
       I->src[0].type != AGX_INDEX_NORMAL)
      return;

   /* Check for invert */
   agx_instr *def = defs[I->src[0].value];
   if (def->op != AGX_OPCODE_BITOP ||
       !agx_is_equiv(def->src[1], agx_immediate(1)) ||
       def->truth_table != UTIL_LUT2(a ^ b))
      return;

   /* Fuse */
   I->src[0] = def->src[0];
   I->invert_cond = !I->invert_cond;
}

/*
 * Evaluate an integer compare of two immediates on AGX3, which has no compare
 * or select form without a register compare operand. Immediate inlining
 * leaves one behind for e.g. umax(undef, 1), a constant compare can feed a
 * select as a value, and fusing a compare of two constants into a select
 * produces one; robust cube image coordinates do all of these.
 */
bool
agx_eval_immediate_icond(agx_context *ctx, enum agx_icond icond, agx_index x,
                         agx_index y, bool *result)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       x.type != AGX_INDEX_IMMEDIATE || y.type != AGX_INDEX_IMMEDIATE ||
       x.neg || y.neg)
      return false;

   /* A 16-bit immediate with its top bit set is negative in a 16-bit signed
    * comparison but not in a 32-bit one, and the comparison width is no
    * longer known here. Below that bit, signed and unsigned orderings agree.
    */
   const bool is_signed = icond == AGX_ICOND_SEQ || icond == AGX_ICOND_SLT ||
                          icond == AGX_ICOND_SGT;
   if (is_signed && (x.value >= 0x8000 || y.value >= 0x8000))
      return false;

   switch (icond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      *result = x.value == y.value;
      return true;
   case AGX_ICOND_ULT:
   case AGX_ICOND_SLT:
      *result = x.value < y.value;
      return true;
   case AGX_ICOND_UGT:
   case AGX_ICOND_SGT:
      *result = x.value > y.value;
      return true;
   default:
      return false;
   }
}

static agx_instr *
agx_optimizer_fold_icmp(agx_context *ctx, agx_instr *I)
{
   bool cond;
   if (!agx_eval_immediate_icond(ctx, I->icond, I->src[0], I->src[1], &cond))
      return NULL;

   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
   agx_instr *folded =
      agx_mov_imm_to(&b, I->dest[0], cond != I->invert_cond);
   agx_remove_instruction(I);
   return folded;
}

static agx_instr *
agx_optimizer_fold_icmpsel(agx_context *ctx, agx_instr *I)
{
   bool cond;
   if (!agx_eval_immediate_icond(ctx, I->icond, I->src[0], I->src[1], &cond))
      return NULL;

   const agx_index chosen = cond ? I->src[2] : I->src[3];
   if (chosen.neg ||
       !(chosen.type == AGX_INDEX_IMMEDIATE ||
         ((chosen.type == AGX_INDEX_NORMAL ||
           chosen.type == AGX_INDEX_REGISTER) &&
          !chosen.abs)))
      return NULL;

   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
   agx_instr *folded = chosen.type == AGX_INDEX_IMMEDIATE
                          ? agx_mov_imm_to(&b, I->dest[0], chosen.value)
                          : agx_mov_to(&b, I->dest[0], chosen);
   agx_remove_instruction(I);
   return folded;
}

static bool
agx_same_select_value(agx_index a, agx_index b)
{
   return agx_is_equiv(a, b) && a.size == b.size && a.abs == b.abs &&
          a.neg == b.neg && a.memory == b.memory &&
          a.channels_m1 == b.channels_m1;
}

struct agx_select_domain {
   agx_instr **defs;
   agx_index variable;
   uint64_t cuts[32];
   unsigned count;
};

static bool agx_select_collect_value(struct agx_select_domain *d,
                                     agx_index value, unsigned depth);

static bool
agx_select_plain(agx_index value)
{
   return !value.abs && !value.neg && !value.memory &&
          agx_channels(value) == 1 && value.size <= AGX_SIZE_32;
}

static bool
agx_select_collect_condition(struct agx_select_domain *d, const agx_instr *I,
                             unsigned depth)
{
   if (!depth || !agx_select_plain(I->src[0]) || !agx_select_plain(I->src[1]) ||
       I->icond == 3 || I->icond == 7)
      return false;
   for (unsigned side = 0; side < 2; ++side) {
      if (!agx_same_select_value(I->src[side], d->variable))
         continue;
      const agx_index other = I->src[side ^ 1];
      if (other.type != AGX_INDEX_IMMEDIATE || d->count > 30)
         return false;
      d->cuts[d->count++] = other.value;
      d->cuts[d->count++] = (uint64_t)other.value + 1;
      return true;
   }
   return agx_select_collect_value(d, I->src[0], depth - 1) &&
          agx_select_collect_value(d, I->src[1], depth - 1);
}

static bool
agx_select_collect_value(struct agx_select_domain *d, agx_index value,
                         unsigned depth)
{
   if (!agx_select_plain(value))
      return false;
   if (value.type == AGX_INDEX_IMMEDIATE)
      return true;
   if (!depth || value.type != AGX_INDEX_NORMAL ||
       agx_same_select_value(value, d->variable))
      return false;
   const agx_instr *I = d->defs[value.value];
   if (!I || (I->op != AGX_OPCODE_ICMPSEL && I->op != AGX_OPCODE_ICMP) ||
       I->saturate || (I->op == AGX_OPCODE_ICMPSEL && I->invert_cond) ||
       value.size != I->dest[0].size || !agx_select_plain(I->dest[0]) ||
       !agx_select_collect_condition(d, I, depth - 1))
      return false;
   return I->op == AGX_OPCODE_ICMP ||
          (agx_select_collect_value(d, I->src[2], depth - 1) &&
           agx_select_collect_value(d, I->src[3], depth - 1));
}

static uint32_t agx_select_evaluate_value(const struct agx_select_domain *d,
                                          agx_index value, uint32_t input);

static bool
agx_select_evaluate_condition(const struct agx_select_domain *d,
                              const agx_instr *I, uint32_t input)
{
   uint32_t x = agx_select_evaluate_value(d, I->src[0], input);
   uint32_t y = agx_select_evaluate_value(d, I->src[1], input);
   const bool half =
      I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16;
   if (half) {
      x &= 0xffff;
      y &= 0xffff;
   }
   const int32_t sx = half ? (int16_t)x : (int32_t)x;
   const int32_t sy = half ? (int16_t)y : (int32_t)y;
   switch (I->icond) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      return x == y;
   case AGX_ICOND_ULT:
      return x < y;
   case AGX_ICOND_UGT:
      return x > y;
   case AGX_ICOND_SLT:
      return sx < sy;
   case AGX_ICOND_SGT:
      return sx > sy;
   default:
      UNREACHABLE("invalid select condition");
   }
}

static uint32_t
agx_select_evaluate_value(const struct agx_select_domain *d, agx_index value,
                          uint32_t input)
{
   if (value.type == AGX_INDEX_IMMEDIATE)
      return value.value;
   if (agx_same_select_value(value, d->variable))
      return input;
   const agx_instr *I = d->defs[value.value];
   const bool condition = agx_select_evaluate_condition(d, I, input);
   if (I->op == AGX_OPCODE_ICMP)
      return condition != I->invert_cond;
   return agx_select_evaluate_value(d, I->src[condition ? 2 : 3], input);
}

static int
agx_select_compare_cut(const void *a, const void *b)
{
   const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
   return (x > y) - (x < y);
}

static void
agx_optimizer_select_guard(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   if (!ctx->key || ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       I->op != AGX_OPCODE_ICMPSEL || I->saturate || I->invert_cond ||
       agx_channels(I->dest[0]) != 1)
      return;
   for (unsigned branch = 2; branch < 4; ++branch) {
      const agx_index value = I->src[branch];
      if (value.type != AGX_INDEX_NORMAL || !agx_select_plain(value) ||
          value.size != I->dest[0].size)
         continue;
      const agx_instr *inner = defs[value.value];
      if (!inner ||
          (inner->op != AGX_OPCODE_ICMPSEL && inner->op != AGX_OPCODE_ICMP) ||
          inner->saturate ||
          (inner->op == AGX_OPCODE_ICMPSEL && inner->invert_cond))
         continue;
      agx_instr boolean_select;
      agx_index boolean_sources[4];
      if (inner->op == AGX_OPCODE_ICMP) {
         boolean_select = *inner;
         boolean_sources[0] = inner->src[0];
         boolean_sources[1] = inner->src[1];
         boolean_sources[2] = agx_immediate(!inner->invert_cond);
         boolean_sources[3] = agx_immediate(inner->invert_cond);
         boolean_select.src = boolean_sources;
         boolean_select.nr_srcs = 4;
         boolean_select.op = AGX_OPCODE_ICMPSEL;
         boolean_select.invert_cond = false;
         inner = &boolean_select;
      }
      for (unsigned base = 2; base < 4; ++base) {
         if (!agx_same_select_value(inner->src[base], I->src[branch ^ 1]))
            continue;
         for (unsigned side = 0; side < 4; ++side) {
            const agx_index variable =
               side < 2 ? inner->src[side] : I->src[side - 2];
            if (variable.type != AGX_INDEX_NORMAL ||
                !agx_select_plain(variable))
               continue;
            const uint64_t limit = UINT64_C(1)
                                   << (variable.size == AGX_SIZE_16 ? 16 : 32);
            struct agx_select_domain d = {
               .defs = defs,
               .variable = variable,
               .cuts = {0, limit / 2, limit},
               .count = 3,
            };
            if (!agx_select_collect_condition(&d, I, 8) ||
                !agx_select_collect_condition(&d, inner, 8))
               continue;
            qsort(d.cuts, d.count, sizeof(d.cuts[0]), agx_select_compare_cut);
            bool active[32] = {false};
            for (unsigned c = 0; c + 1 < d.count; ++c) {
               active[c] = agx_select_evaluate_condition(&d, I, d.cuts[c]) ==
                              (branch == 2) &&
                           agx_select_evaluate_condition(
                              &d, inner, d.cuts[c]) == (base == 3);
            }
            bool any = false, all = true;
            for (unsigned c = 0; c + 1 < d.count; ++c) {
               if (d.cuts[c] == d.cuts[c + 1])
                  continue;
               any |= active[c];
               all &= active[c];
            }
            if (!any || all) {
               I->src[0] = I->src[1] = agx_zero();
               I->src[2] = I->src[3] = inner->src[all ? (base ^ 1) : base];
               I->icond = AGX_ICOND_UEQ;
               return;
            }
            const enum agx_icond conditions[] = {
               AGX_ICOND_UEQ, AGX_ICOND_ULT, AGX_ICOND_UGT,
               AGX_ICOND_SLT, AGX_ICOND_SGT,
            };
            for (unsigned op = 0; op < ARRAY_SIZE(conditions); ++op) {
               for (unsigned c = 0; c < d.count; ++c) {
                  for (unsigned delta = 0; delta < 2; ++delta) {
                     const uint64_t constant = d.cuts[c] - delta;
                     if (constant >= 0x100)
                        continue;
                     agx_index sources[2] = {variable, agx_immediate(constant)};
                     agx_instr comparison = {
                        .src = sources,
                        .icond = conditions[op],
                     };
                     bool match = true, inverse = true;
                     for (unsigned cell = 0; cell + 1 < d.count; ++cell) {
                        if (d.cuts[cell] == d.cuts[cell + 1])
                           continue;
                        const bool first = agx_select_evaluate_condition(
                           &d, &comparison, d.cuts[cell]);
                        const bool last = agx_select_evaluate_condition(
                           &d, &comparison, d.cuts[cell + 1] - 1);
                        match &= first == active[cell] && last == active[cell];
                        inverse &=
                           first != active[cell] && last != active[cell];
                        if (!match && !inverse)
                           break;
                     }
                     if (!match && !inverse)
                        continue;
                     I->src[0] = sources[0];
                     I->src[1] = sources[1];
                     I->src[2] = inner->src[inverse ? base : (base ^ 1)];
                     I->src[3] = inner->src[inverse ? (base ^ 1) : base];
                     I->icond = comparison.icond;
                     return;
                  }
               }
            }
         }
      }
   }
}

/*
 * Fuse conditions into select. Specifically, acts on icmpsel and fuses:
 *
 *    icmpsel(cmp(x, y, *), 0, z, w, eq) -> cmpsel(x, y, w, z, *)
 *
 * Care must be taken to invert the condition by swapping cmpsel arguments.
 */
static void
agx_optimizer_cmpsel(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   /* Check for unfused select */
   if (!agx_is_equiv(I->src[1], agx_zero()) || I->icond != AGX_ICOND_UEQ ||
       I->src[0].type != AGX_INDEX_NORMAL)
      return;

   /* Check for condition */
   agx_instr *def = defs[I->src[0].value];
   if (!def)
      return;
   if (def->op == AGX_OPCODE_ICMPSEL && !def->saturate && !def->invert_cond &&
       agx_select_plain(I->src[0]) && I->src[0].size == def->dest[0].size &&
       agx_select_plain(def->src[2]) && agx_select_plain(def->src[3])) {
      bool yes, no;
      if (!agx_eval_immediate_icond(ctx, I->icond, def->src[2], I->src[1],
                                    &yes) ||
          !agx_eval_immediate_icond(ctx, I->icond, def->src[3], I->src[1], &no))
         return;
      const agx_index a = I->src[yes ? 2 : 3];
      const agx_index b = I->src[no ? 2 : 3];
      I->src[0] = yes == no ? agx_zero() : def->src[0];
      I->src[1] = yes == no ? agx_zero() : def->src[1];
      I->src[2] = a;
      I->src[3] = b;
      I->icond = yes == no ? AGX_ICOND_UEQ : def->icond;
      return;
   }
   if (def->op != AGX_OPCODE_ICMP && def->op != AGX_OPCODE_FCMP)
      return;

   /* Fuse */
   I->src[0] = def->src[0];
   I->src[1] = def->src[1];

   /* In the unfused select, the condition is inverted due to the form:
    *
    *    (cond == 0) ? x : y
    *
    * So we need to swap the arguments when fusing to become cond ? y : x. If
    * the condition was supposed to be inverted, we don't swap since it's
    * already inverted. cmpsel does not have an invert_cond bit to use.
    */
   if (!def->invert_cond) {
      agx_index temp = I->src[2];
      I->src[2] = I->src[3];
      I->src[3] = temp;
   }

   if (def->op == AGX_OPCODE_ICMP) {
      I->op = AGX_OPCODE_ICMPSEL;
      I->icond = def->icond;
   } else {
      I->op = AGX_OPCODE_FCMPSEL;
      I->fcond = def->fcond;
   }
}

/*
 * Fuse conditions into ballots:
 *
 *    ballot(cmp(x, y)) -> ballot_cmp(x, y)
 */

/* Whether the AGX3B packer has an immediate form for a ballot of this
 * comparison (agx3_emit_subgroup_ballot, non-quad). */
static bool
agx3b_ballot_compare_encodable(const agx_instr *def)
{
   if (def->src[1].type != AGX_INDEX_IMMEDIATE)
      return false;
   const uint32_t value = def->src[1].value;
   /* A 16-bit source has only the equal-to-zero form. */
   if (def->src[0].size == AGX_SIZE_16)
      return def->op == AGX_OPCODE_ICMP && !def->invert_cond && value == 0 &&
             (def->icond == AGX_ICOND_UEQ || def->icond == AGX_ICOND_SEQ);
   if (def->op == AGX_OPCODE_FCMP)
      return value == 0;
   switch (def->icond) {
   case AGX_ICOND_UEQ: return !def->invert_cond || value == 0;
   case AGX_ICOND_SEQ:
   case AGX_ICOND_SLT:
   case AGX_ICOND_SGT: return value == 0;
   case AGX_ICOND_ULT: return !def->invert_cond || value != 0;
   case AGX_ICOND_UGT: return !def->invert_cond || value != UINT16_MAX;
   default: return false;
   }
}

static void
agx_optimizer_ballot(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   /* The grounded G17P ballot form compares a full 32-bit register against
    * zero.  Common subgroup reduction lowering legitimately creates
    * ballot(true) to distinguish full from partial active subgroups, and the
    * immediate inliner runs before this fusion pass.  Materialize that exact
    * boolean here so lower_pseudo reaches the witnessed register ballot
    * instead of leaving an unencodable immediate predicate after RA. */
   if (I->op == AGX_OPCODE_BALLOT && ctx->key &&
      ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      I->src[0].type == AGX_INDEX_IMMEDIATE) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index predicate = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_imm_to(&b, predicate, I->src[0].value);
      I->src[0] = predicate;
      return;
   }

   /* Likewise for a quad ballot, whose witnessed register form takes a 16-bit
    * predicate. */
   if (I->op == AGX_OPCODE_QUAD_BALLOT && ctx->key &&
      ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      I->src[0].type == AGX_INDEX_IMMEDIATE) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index predicate = agx_temp(ctx, AGX_SIZE_16);
      agx_mov_imm_to(&b, predicate, I->src[0].value);
      I->src[0] = predicate;
      return;
   }

   agx_instr *def =
      I->src[0].type == AGX_INDEX_NORMAL ? defs[I->src[0].value] : NULL;

   /* The same register-against-zero ballot is witnessed on G17P for a 16-bit
    * source only in its equal-to-zero form, while an unfused ballot counts
    * the lanes where its predicate is nonzero. Widen a half predicate so it
    * reaches the 32-bit form instead. That includes a uniform predicate
    * (one the constant program computed), which register allocation would
    * otherwise copy into a half register.
    */
   if (I->op == AGX_OPCODE_BALLOT && ctx->key &&
       ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->src[0].size == AGX_SIZE_16 &&
       !(def && (def->op == AGX_OPCODE_ICMP || def->op == AGX_OPCODE_FCMP))) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index wide = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_to(&b, wide, I->src[0]);
      I->src[0] = wide;
      return;
   }

   if (!def || (def->op != AGX_OPCODE_ICMP && def->op != AGX_OPCODE_FCMP))
      return;

   bool quad = I->op == AGX_OPCODE_QUAD_BALLOT;
   assert(quad || I->op == AGX_OPCODE_BALLOT);

   /* G17P's half-width quad ballot compares a register against zero. AGX2
    * can fuse general integer/float comparisons, but those forms do not
    * encode an AGX3 half-width quad result. Keep the predicate calculation
    * for quad any/all and let lower_pseudo ballot its nonzero boolean.
    */
   if (quad && I->dest[0].size == AGX_SIZE_16 && ctx->key &&
       ctx->key->dev.chip == AGX_CHIP_G17P &&
       ctx->key->dev.shader_quad_control &&
       (def->op != AGX_OPCODE_ICMP ||
        def->src[1].type != AGX_INDEX_IMMEDIATE || def->src[1].value != 0 ||
        (def->icond != AGX_ICOND_UEQ && def->icond != AGX_ICOND_SEQ)))
      return;

   /* G18G's general ballot form tests a predicate against an immediate. Keep
    * register comparisons separate instead of fusing an unsupported shape.
    * M6 only: the M5 keeps the fused register form. */
   if (!quad && ctx->key && agx_is_g18g(&ctx->key->dev) &&
       def->src[1].type != AGX_INDEX_IMMEDIATE)
      return;

   /* Replace with a fused instruction since the # of sources changes */
   agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

   /* Every G17P SIMD ballot Apple's compiler emits compares one register
    * against a constant, so a ballot of two registers has no witnessed
    * encoding. Keep the comparison, with the opposite sense, as a boolean and
    * ballot the lanes where it is zero: the witnessed simd_broadcast_first()
    * form.
    */
   /* Likewise for a constant the immediate form cannot take (see
    * agx3_emit_subgroup_ballot): `x != N` for a nonzero N (the register-zero
    * equality form it would otherwise take is measured wrong on G17P), signed
    * and float comparisons against anything but zero, `!(x < 0)`,
    * `!(x > 0xffff)`, and any 16-bit source but `x == 0`. Ballot the opposite compare instead, exactly as for a
    * register operand below. */
   if (!quad && ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       !agx3b_ballot_compare_encodable(def)) {
      agx_index opposite = agx_temp(ctx, AGX_SIZE_16);

      if (def->op == AGX_OPCODE_ICMP) {
         agx_icmp_to(&b, opposite, def->src[0], def->src[1], def->icond,
                     !def->invert_cond);
      } else {
         agx_fcmp_to(&b, opposite, def->src[0], def->src[1], def->fcond,
                     !def->invert_cond);
      }

      agx_icmp_ballot_to(&b, I->dest[0], opposite, agx_zero(), AGX_ICOND_UEQ,
                         false);
      agx_remove_instruction(I);
      return;
   }

   agx_instr *fused = agx_icmp_ballot_to(
      &b, I->dest[0], def->src[0], def->src[1], def->icond, def->invert_cond);

   if (def->op == AGX_OPCODE_ICMP) {
      fused->op = quad ? AGX_OPCODE_ICMP_QUAD_BALLOT : AGX_OPCODE_ICMP_BALLOT;
   } else {
      fused->op = quad ? AGX_OPCODE_FCMP_QUAD_BALLOT : AGX_OPCODE_FCMP_BALLOT;
   }

   agx_remove_instruction(I);
}

/*
 * Fuse not srcs into bitop.
 */
static void
agx_optimizer_bitop(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = defs[src.value];

      /* If we find a not, select new operation and fuse */
      if (def->op == AGX_OPCODE_NOT) {
         const unsigned truth_table =
            util_lut2_invert_source(I->truth_table, s);
         agx_index other = I->src[1 - s];
         if (other.type == AGX_INDEX_NORMAL) {
            const agx_instr *other_def = defs[other.value];
            if (other_def && other_def->op == AGX_OPCODE_NOT &&
                other_def->src[0].size == AGX_SIZE_32)
               other = other_def->src[0];
         }
         /* Enable the witnessed uniform forms. Keep the existing restriction
          * for register-only expressions until their generated code is
          * profitable as well as their encoding being supported.
          */
         const bool native_uniform_word =
            I->dest[0].size == AGX_SIZE_32 &&
            I->src[0].size == AGX_SIZE_32 &&
            I->src[1].size == AGX_SIZE_32 &&
            def->src[0].size == AGX_SIZE_32 &&
            (def->src[0].type == AGX_INDEX_UNIFORM ||
             other.type == AGX_INDEX_UNIFORM);
         if (ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             !native_uniform_word &&
             truth_table != AGX3_SEL_TT_AND &&
             truth_table != AGX3_SEL_TT_OR &&
             truth_table != AGX3_SEL_TT_XOR &&
             truth_table != AGX3_SEL_TT_NOT)
            continue;

         I->truth_table = truth_table;
         I->src[s] = def->src[0];
      }
   }
}

/*
 * Fuse sign-extends into addition-like instructions.
 */
static void
agx_optimizer_signext(agx_context *ctx, agx_instr **defs, agx_instr *I)
{
   agx_foreach_ssa_src(I, s) {
      agx_index src = I->src[s];
      agx_instr *def = defs[src.value];

      /* Integer abs means zero extension. Keep the intermediate signed
       * word when the operation later zero-extends it (the unsigned 64-bit
       * product of a sign-extended half): narrowing that operand would
       * change the value for negative halves. The native legalization pass
       * already knows how to materialize SIGNEXT without a signed-half
       * source modifier. */
      if (def == NULL || def->op != AGX_OPCODE_SIGNEXT || src.abs)
         continue;

      agx_replace_src(I, s, def->src[0]);
   }
}


void
agx_optimizer_forward(agx_context *ctx)
{
   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));

   agx_foreach_instr_global_safe(ctx, I) {
      struct agx_opcode_info info = agx_opcodes_info[I->op];

      agx_foreach_ssa_dest(I, d) {
         defs[I->dest[d].value] = I;
      }

      /* Optimize moves */
      agx_optimizer_copyprop(ctx, defs, I);

      /* Propagate fmov down */
      if (info.is_float || I->op == AGX_OPCODE_FCMPSEL ||
          I->op == AGX_OPCODE_FCMP)
         agx_optimizer_fmov(ctx, defs, I);

      /* Inline immediates if we can. TODO: systematic */
      if (I->op != AGX_OPCODE_COLLECT && I->op != AGX_OPCODE_IMAGE_LOAD &&
          I->op != AGX_OPCODE_TEXTURE_LOAD &&
          I->op != AGX_OPCODE_TEX_STATE_STORE &&
          I->op != AGX_OPCODE_SAMPLER_STATE_STORE &&
          I->op != AGX_OPCODE_UNIFORM_STORE && I->op != AGX_OPCODE_EXPORT)
         agx_optimizer_inline_imm(ctx, defs, I);

      agx_optimizer_minifloat_select(ctx, defs, I);

      /* Inlining a constant into fmov can leave two minifloat operands.
       * Fold the conversion exactly, including negative zero, before RA. */
      if (agx_is_fmov(I) && I->src[0].type == AGX_INDEX_IMMEDIATE &&
          !I->saturate) {
         float value = agx_minifloat_decode(I->src[0].value);
         if (I->src[0].abs)
            value = fabsf(value);
         if (I->src[0].neg)
            value = -value;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_instr *mov = agx_mov_imm_to(&b, I->dest[0],
            I->dest[0].size == AGX_SIZE_16 ? _mesa_float_to_half(value) : fui(value));
         defs[I->dest[0].value] = mov;
         agx_remove_instruction(I);
         continue;
      }

      if (I->op == AGX_OPCODE_IF_ICMP) {
         agx_optimizer_if_not(defs, I);
         agx_optimizer_if_cmp(defs, I);
      } else if (I->op == AGX_OPCODE_ICMP) {
         agx_instr *folded = agx_optimizer_fold_icmp(ctx, I);
         if (folded)
            defs[folded->dest[0].value] = folded;
      } else if (I->op == AGX_OPCODE_ICMPSEL) {
         /* Fusing a compare of two inlined immediates yields a select the
          * fold below can resolve. */
         agx_optimizer_cmpsel(ctx, defs, I);
         agx_optimizer_minifloat_select(ctx, defs, I);
         agx_g17p_tie_zero_minmax(ctx, I);
         agx_optimizer_select_guard(ctx, defs, I);
         agx_instr *folded = agx_optimizer_fold_icmpsel(ctx, I);
         if (folded)
            defs[folded->dest[0].value] = folded;
      } else if (I->op == AGX_OPCODE_BALLOT ||
                 I->op == AGX_OPCODE_QUAD_BALLOT) {
         agx_optimizer_ballot(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_BITOP) {
         agx_optimizer_bitop(ctx, defs, I);
      } else if (I->op == AGX_OPCODE_IADD || I->op == AGX_OPCODE_IMAD) {
         agx_optimizer_signext(ctx, defs, I);
      }
   }

   free(defs);
}

static void
record_use(agx_instr **uses, BITSET_WORD *multiple, agx_instr *I, unsigned src)
{
   unsigned v = I->src[src].value;

   if (uses[v])
      BITSET_SET(multiple, v);
   else
      uses[v] = I;
}

void
agx_optimizer_backward(agx_context *ctx)
{
   agx_instr **uses = calloc(ctx->alloc, sizeof(*uses));
   BITSET_WORD *multiple = BITSET_CALLOC(ctx->alloc);

   agx_foreach_block_rev(ctx, block) {
      /* Phi sources are logically read at the end of predecessor, so process
       * our source in our successors' phis firsts. This ensures we set
       * `multiple` correctly with phi sources.
       */
      agx_foreach_successor(block, succ) {
         unsigned s = agx_predecessor_index(succ, block);

         agx_foreach_phi_in_block(succ, phi) {
            record_use(uses, multiple, phi, s);
         }
      }

      agx_foreach_instr_in_block_rev(block, I) {
         struct agx_opcode_info info = agx_opcodes_info[I->op];

         /* Skip phis, they're handled specially */
         if (I->op == AGX_OPCODE_PHI) {
            continue;
         }

         agx_foreach_ssa_src(I, s) {
            record_use(uses, multiple, I, s);
         }

         if (info.nr_dests != 1)
            continue;

         if (I->dest[0].type != AGX_INDEX_NORMAL)
            continue;

         agx_instr *use = uses[I->dest[0].value];

         if (!use || BITSET_TEST(multiple, I->dest[0].value))
            continue;

         if (agx_optimizer_not(ctx, I, use)) {
            agx_remove_instruction(use);
            continue;
         }

         /* Destination has a single use, try to propagate */
         if (info.is_float && agx_optimizer_fmov_rev(ctx, I, use)) {
            agx_remove_instruction(use);
            continue;
         }
      }
   }

   free(uses);
   free(multiple);
}
