/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "util/lut.h"
#include "agx_builder.h"
#include "asahi/isa/agx_minifloat.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/* Lower pseudo instructions created during optimization. */

static agx_instr *
while_for_break_if(agx_builder *b, agx_instr *I)
{
   if (I->op == AGX_OPCODE_BREAK_IF_FCMP) {
      return agx_while_fcmp(b, I->src[0], I->src[1], I->nest, I->fcond,
                            !I->invert_cond, NULL);
   } else {
      return agx_while_icmp(b, I->src[0], I->src[1], I->nest, I->icond,
                            !I->invert_cond, NULL);
   }
}

static agx_instr *
cmpsel_for_break_if(agx_builder *b, agx_instr *I)
{
   agx_index r0l = agx_register(0, AGX_SIZE_16);

   /* If the condition is true, set r0l to nest to break */
   agx_index t = agx_immediate(I->nest);
   agx_index f = r0l;

   if (I->invert_cond) {
      SWAP(t, f);
   }

   if (I->op == AGX_OPCODE_BREAK_IF_FCMP)
      agx_fcmpsel_to(b, r0l, I->src[0], I->src[1], t, f, I->fcond);
   else
      agx_icmpsel_to(b, r0l, I->src[0], I->src[1], t, f, I->icond);

   return agx_push_exec(b, 0);
}

static void
swap(agx_builder *b, agx_index x, agx_index y)
{
   assert(!x.memory && "already lowered");
   assert(!y.memory && "already lowered");

   /* We can swap lo/hi halves of a 32-bit register with a 32-bit extr */
   if (x.size == AGX_SIZE_16 && (x.value >> 1) == (y.value >> 1)) {

      assert(((x.value & 1) == (1 - (y.value & 1))) &&
             "no trivial swaps, and only 2 halves of a register");

      /* r0 = extr r0, r0, #16
       *    = (((r0 << 32) | r0) >> 16) & 0xFFFFFFFF
       *    = (((r0 << 32) >> 16) & 0xFFFFFFFF) | (r0 >> 16)
       *    = (r0l << 16) | r0h
       */
      agx_index reg32 = agx_register(x.value & ~1, AGX_SIZE_32);
      agx_extr_to(b, reg32, reg32, reg32, agx_immediate(16), 0);
   } else {
      /* We're swapping GPRs and fall back on a XOR swap. */
      agx_xor_to(b, x, x, y);
      agx_xor_to(b, y, x, y);
      agx_xor_to(b, x, x, y);
   }
}

/*
 * G17P has no compare of two immediates. NIR folds constant comparisons, but
 * backend lowering can still produce one after register allocation (an
 * always-taken else of the structurizer, for instance), and the packer then
 * refused the whole shader. Evaluate such a compare here: float immediates are
 * minifloats, which have no NaN or infinity, so every known condition is exact;
 * integer immediates go through agx_eval_immediate_icond(). Returns the truth
 * value, or -1 to leave the instruction alone.
 */
static int
agx3_immediate_compare(agx_context *ctx, const agx_instr *I)
{
   const agx_index a = I->src[0], c = I->src[1];
   if (a.type != AGX_INDEX_IMMEDIATE || c.type != AGX_INDEX_IMMEDIATE ||
       a.abs || a.neg || c.abs || c.neg)
      return -1;

   if (I->op == AGX_OPCODE_FCMP || I->op == AGX_OPCODE_FCMPSEL) {
      const float x = agx_minifloat_decode(a.value);
      const float y = agx_minifloat_decode(c.value);
      switch (I->fcond) {
      case AGX_FCOND_EQ: return x == y;
      case AGX_FCOND_LT: return x < y;
      case AGX_FCOND_GT: return x > y;
      case AGX_FCOND_GE: return x >= y;
      case AGX_FCOND_LE: return x <= y;
      default: return -1;
      }
   }

   bool cond;
   return agx_eval_immediate_icond(ctx, I->icond, a, c, &cond) ? cond : -1;
}

static agx_instr *
lower(agx_builder *b, agx_instr *I)
{
   const bool agx3 = b->shader->key != NULL &&
                     b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;

   if (agx3 && (I->op == AGX_OPCODE_FCMP || I->op == AGX_OPCODE_ICMP)) {
      const int truth = agx3_immediate_compare(b->shader, I);
      if (truth >= 0)
         return agx_mov_imm_to(b, I->dest[0], truth != I->invert_cond);
   }

   if (agx3 && !I->invert_cond &&
       (I->op == AGX_OPCODE_FCMPSEL || I->op == AGX_OPCODE_ICMPSEL)) {
      const int truth = agx3_immediate_compare(b->shader, I);
      const agx_index chosen = I->src[truth ? 2 : 3];
      if (truth >= 0 && chosen.type == AGX_INDEX_IMMEDIATE)
         return agx_mov_imm_to(b, I->dest[0],
                               I->fcmpsel_minifloat_select
                                  ? fui(agx_minifloat_decode(chosen.value))
                                  : chosen.value);
      if (truth >= 0 && !chosen.abs && !chosen.neg &&
          chosen.size == I->dest[0].size)
         return agx_bitop_to(b, I->dest[0], chosen, agx_zero(),
                             UTIL_LUT2(a));
   }

   switch (I->op) {

   case AGX_OPCODE_FADD:
      /* Constant propagation can leave two minifloats at a floating
       * move/conversion. AGX2 encodes an immediate in either ALU source, but
       * none of the AGX3 tables (agx3b, agx3c, agx3d) has an f32 add whose two
       * operands are both immediates: every AGX3 packer refuses the pair
       * (agx3c and agx3d always; agx3b folded only the unmodified,
       * non-saturating pair itself). So this is an AGX3 encoding property,
       * not a G18G one.
       */
      if (agx3 && agx_encoding_is_agx3(b->shader->key->dev.encoding) &&
          I->nr_dests == 1 && I->nr_srcs == 2 &&
          I->dest[0].type == AGX_INDEX_REGISTER && !I->dest[0].memory &&
          I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value <= UINT8_MAX &&
          I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value <= UINT8_MAX &&
          !I->src[0].memory && !I->src[1].memory &&
          agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1) {
         /* Unmodified and unclamped, the sum of two minifloats is exact in
          * f32 (including the sign of a zero sum under round-to-nearest), so
          * fold it to one literal move, the form agx3b already emitted. */
         if (!I->saturate && !I->src[0].abs && !I->src[0].neg &&
             !I->src[1].abs && !I->src[1].neg) {
            const float sum = agx_minifloat_decode(I->src[0].value) +
                              agx_minifloat_decode(I->src[1].value);
            return agx_mov_imm_to(b, I->dest[0], fui(sum));
         }

         /* Otherwise keep the arithmetic, modifiers and clamp on the GPU:
          * reuse the assigned destination for the exact first constant. */
         const agx_index immediate = I->src[0];
         agx_mov_imm_to(b, I->dest[0], fui(agx_minifloat_decode(immediate.value)));
         I->src[0] = agx_register(I->dest[0].value, AGX_SIZE_32);
         I->src[0].abs = immediate.abs;
         I->src[0].neg = immediate.neg;
      }
      return NULL;

   /* Various instructions are implemented as bitwise truth tables */
   case AGX_OPCODE_MOV:
      if (agx3 && agx_channels(I->dest[0]) > 1) {
         const unsigned channels = agx_channels(I->dest[0]);
         const unsigned stride = agx_size_align_16(I->dest[0].size);
         const agx_index dest = I->dest[0], src = I->src[0];
         assert(!dest.memory && !src.memory);
         assert(src.type == AGX_INDEX_IMMEDIATE ||
                (src.size == dest.size && agx_channels(src) == channels));
         const bool backwards = src.type == AGX_INDEX_REGISTER &&
            dest.value > src.value &&
            dest.value < src.value + channels * stride;

         for (unsigned i = 0; i < channels; ++i) {
            const unsigned c = backwards ? channels - 1 - i : i;
            agx_index d = dest, s = src;
            d.channels_m1 = s.channels_m1 = 0;
            d.value += c * stride;
            if (s.type != AGX_INDEX_IMMEDIATE)
               s.value += c * stride;
            agx_instr *copy = agx_mov_to(b, d, s);
            agx_instr *lowered = lower(b, copy);
            assert(lowered);
            (void)lowered;
            agx_remove_instruction(copy);
         }
         return I;
      }
      if (agx3 && I->src[0].type == AGX_INDEX_IMMEDIATE)
         return agx_mov_imm_to(b, I->dest[0], I->src[0].value);
      /*
       * A modifier on a move is agx_optimizer_fmov() having composed an fneg
       * or fabs into the copy's source: agx_is_fmov() matches `fadd(x, -0.0)`,
       * and agx_compose_float_src() carries its abs/neg across. The bitwise
       * move below has no field for one, so the packer refused the whole
       * shader for `-x` or `|x|` crossing a copy -- 19 of the 307 refusals in
       * a three.js frame. Put the float move back instead: Apple spells a move
       * as an add of the exact negative-zero minifloat, and that form carries
       * both modifier bits (agx3_alu_fadd_minifloat_mod, agx3_alu_fadd_abs_*),
       * so this is the pre-fold program rather than a new encoding.
       */
      if (agx3 && (I->src[0].abs || I->src[0].neg) &&
          I->src[0].size == AGX_SIZE_32 &&
          I->dest[0].size == AGX_SIZE_32)
         return agx_fadd_to(b, I->dest[0], I->src[0], agx_negzero());
      return agx_bitop_to(b, I->dest[0], I->src[0], agx_zero(), UTIL_LUT2(a));

   case AGX_OPCODE_NOT:
      return agx_bitop_to(b, I->dest[0], I->src[0], agx_zero(), UTIL_LUT2(~a));

   /* We can sign-extend with an add */
   case AGX_OPCODE_SIGNEXT:
      return agx_iadd_to(b, I->dest[0], I->src[0], agx_zero(), 0);

   /* Unfused comparisons are fused with a 0/1 select */
   case AGX_OPCODE_ICMP:
      return agx_icmpsel_to(b, I->dest[0], I->src[0], I->src[1],
                            agx_immediate(I->invert_cond ? 0 : 1),
                            agx_immediate(I->invert_cond ? 1 : 0), I->icond);

   case AGX_OPCODE_FCMP:
      return agx_fcmpsel_to(b, I->dest[0], I->src[0], I->src[1],
                            agx_immediate(I->invert_cond ? 0 : 1),
                            agx_immediate(I->invert_cond ? 1 : 0), I->fcond);

   case AGX_OPCODE_BALLOT:
      return agx_icmp_ballot_to(b, I->dest[0], I->src[0], agx_zero(),
                                AGX_ICOND_UEQ, true /* invert */);

   case AGX_OPCODE_QUAD_BALLOT:
      return agx_icmp_quad_ballot_to(b, I->dest[0], I->src[0], agx_zero(),
                                     AGX_ICOND_UEQ, true /* invert */);

   /* Writes to the nesting counter lowered to the real register */
   case AGX_OPCODE_BEGIN_CF:
      /* AGX3 has an architectural exec-mask stack and no witnessed r0l
       * nesting-counter write. Keep the pseudo for its zero-byte AGX3 packer
       * mapping instead of silently emitting the AGX2 contract. */
      if (agx3)
         return NULL;
      return agx_mov_imm_to(b, agx_register(0, AGX_SIZE_16), 0);

   case AGX_OPCODE_BREAK:
      if (agx3) {
         /* AGX3 has no unconditional break instruction. Break on the
          * integer identity r0l == r0l instead, using the compare+cnd
          * sequence without clobbering a post-RA register.
          *
          * NEST carries over unchanged: it is the number of exec-mask levels
          * the break unwinds in Mesa's model either way, and agx_opt_break_if
          * only subtracts one when it actually folds an enclosing if away.
          * The packer turns that into the native level count (nest + 1,
          * because a loop owns two architectural levels), which CNDCF
          * encodes at any depth. */
         agx_index r0l = agx_register(0, AGX_SIZE_16);
         return agx_break_if_icmp(b, r0l, r0l, I->nest, false, AGX_ICOND_UEQ,
                                  I->target);
      }
      agx_mov_imm_to(b, agx_register(0, AGX_SIZE_16), I->nest);
      return agx_pop_exec(b, 0);

   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP: {
      /* AGX3 lowers these directly to compare + cndlt, retaining the
       * condition and avoiding AGX2's r0l/cmpsel fallback. */
      if (agx3)
         return NULL;
      if (I->nest == 1)
         return while_for_break_if(b, I);
      else
         return cmpsel_for_break_if(b, I);
   }

   case AGX_OPCODE_SWAP:
      swap(b, I->src[0], I->src[1]);
      return (void *)true;

   default:
      return NULL;
   }
}

/*
 * AGX3 has no control-flow compare with an immediate on both sides. A constant
 * condition can still reach one: SPIR-V such as dEQP-VK.reconvergence's keeps
 * `if (true) { break; }` inside loops, and immediate inlining turns the
 * condition into an immediate. The construct itself must stay, since it owns
 * an exec-mask level, so keep the instruction and branch on the integer
 * identity r0l == r0l, as an AGX3 unconditional break already does, inverted
 * to preserve the constant outcome.
 */
static void
agx3_lower_constant_cf_compare(agx_context *ctx, agx_instr *I)
{
   if (I->op != AGX_OPCODE_IF_ICMP && I->op != AGX_OPCODE_ELSE_ICMP &&
       I->op != AGX_OPCODE_WHILE_ICMP && I->op != AGX_OPCODE_BREAK_IF_ICMP)
      return;

   bool cond;
   if (!agx_eval_immediate_icond(ctx, I->icond, I->src[0], I->src[1], &cond))
      return;

   agx_index r0l = agx_register(0, AGX_SIZE_16);
   I->src[0] = r0l;
   I->src[1] = r0l;
   I->icond = AGX_ICOND_UEQ;
   I->invert_cond = cond == I->invert_cond;
}

void
agx_lower_pseudo(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      if (ctx->key && agx_isa_is_agx3(ctx->key->dev.encoding))
         agx3_lower_constant_cf_compare(ctx, I);

      if (lower(&b, I))
         agx_remove_instruction(I);
   }
}
