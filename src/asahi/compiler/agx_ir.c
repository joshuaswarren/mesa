/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_compiler.h"
#include "agx_opcodes.h"

bool
agx_allows_16bit_immediate(agx_instr *I, enum agx_isa_encoding enc)
{
   /* The local load/store index immediate exists only in the AGX2 local
    * form; AGX3 keeps the offset in the base operand. */
   const bool local = (I->op == AGX_OPCODE_LOCAL_LOAD) ||
                      (I->op == AGX_OPCODE_LOCAL_STORE);
   return (I->op == AGX_OPCODE_DEVICE_LOAD) ||
          (I->op == AGX_OPCODE_DEVICE_STORE) ||
          (local && enc == AGX_ISA_ENCODING_AGX2) ||
          (I->op == AGX_OPCODE_STACK_LOAD) ||
          (I->op == AGX_OPCODE_STACK_STORE) ||
          (I->op == AGX_OPCODE_UNIFORM_STORE) || (I->op == AGX_OPCODE_ATOMIC) ||
          (I->op == AGX_OPCODE_LOCAL_ATOMIC) || (I->op == AGX_OPCODE_PHI) ||
          (I->op == AGX_OPCODE_TEX_STATE_STORE) ||
          (I->op == AGX_OPCODE_SAMPLER_STATE_STORE);
}

unsigned
agx_negate_src_index(agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_IMAD:
      return 2;
   case AGX_OPCODE_IADD:
      return 1;
   default:
      UNREACHABLE("not allowed");
   }
}

/* A complete fragment may sample through structured control flow and finish
 * with an ordinary MRT epilog. Recognize only a reconverged, colour-only tail;
 * Prefix colour accesses stay closed. A leading force-early depth test may
 * protect memory effects that remain entirely before the colour transaction.
 * Both ordering (SSA) and packing (physical registers) use this same proof.
 */
agx_block *
agx_reconverged_mrt_tail(agx_context *ctx)
{
   const bool multisample_epilog = ctx->key->secondary &&
      (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4);
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       (ctx->key->secondary && !multisample_epilog) || ctx->key->no_stop ||
       ctx->key->fs.ignore_tib_dependencies ||
       (ctx->key->g17p_emrt && !multisample_epilog) ||
       (ctx->key->fs.nr_samples > 1 && !multisample_epilog) ||
       list_length(&ctx->blocks) <= 1 ||
       !ctx->nir || !ctx->indexed_nir_blocks)
      return NULL;

   agx_block *tail = list_last_entry(&ctx->blocks, agx_block, link);
   /* The compiler supports only entrypoint and preamble functions. The
    * preamble was excluded above, so this is the map for the entrypoint. */
   nir_function_impl *impl = nir_shader_get_entrypoint(ctx->nir);
   nir_block *nir_tail = nir_impl_last_block(impl);
   /* The last top-level NIR block is outside every if/loop. Do not use
    * agx_block::divergent here: after-blocks are conservatively allocated
    * before emit_if decrements the nesting counter.
    */
   if (ctx->indexed_nir_blocks[nir_tail->index] != tail ||
       agx_num_successors(tail) != 0)
      return NULL;

   bool tile = false, stop = false, early_wait = false, early_tested = false;
   agx_foreach_block(ctx, block) {
      /* A unique CFG exit means every normally terminating path reaches
       * the reconverged tail, even when the prefix contains a loop.
       */
      if (block != tail && !agx_num_successors(block))
         return NULL;
      agx_foreach_instr_in_block(block, I) {
         switch (I->op) {
         case AGX_OPCODE_LD_TILE:
         case AGX_OPCODE_ST_TILE:
            if (block != tail)
               return NULL;
            tile = true;
            break;
         case AGX_OPCODE_WAIT_PIX:
            if (I->writeout == 1) {
               /* Force-early tests begin class 1 in the prefix and the
                * following mask update retains it. Colour ordering does
                * not move or otherwise alter that transaction. */
               if (block != agx_start_block(ctx) || tile || early_wait ||
                   !ctx->nir->info.fs.early_fragment_tests)
                  return NULL;
               early_wait = true;
            } else if (block != tail) {
               return NULL;
            }
            break;
         case AGX_OPCODE_PUBLISH_TEXTURE_AGX3:
            /* RA publishes the prefix's texture results into the staging
             * file. This compiler-generated operation has no pixel class. */
            if (tile)
               return NULL;
            break;
         case AGX_OPCODE_SAMPLE_MASK_AGX3:
            /* Pure kills do not acquire a pixel class. A testing update
             * acquires and releases class 1 within its own instruction, so
             * either may precede the reconverged colour transaction. */
            if (tile || (early_wait && I->imm != 3) ||
                (multisample_epilog && I->imm != 0 && I->imm != 2))
               return NULL;
            early_tested |= early_wait && I->imm == 3;
            early_wait = false;
            break;
         case AGX_OPCODE_ATOMIC:
         case AGX_OPCODE_DEVICE_STORE:
            /* Force-early testing already protects these API side effects.
             * They remain in the prefix, before every colour transaction. */
            if (block == tail || tile || !early_tested)
               return NULL;
            break;
         case AGX_OPCODE_STOP:
            if (block != tail || I->link.next != &tail->instructions)
               return NULL;
            stop = true;
            break;
         case AGX_OPCODE_BEGIN_CF:
         case AGX_OPCODE_IF_ICMP:
         case AGX_OPCODE_IF_FCMP:
         case AGX_OPCODE_WHILE_ICMP:
         case AGX_OPCODE_WHILE_FCMP:
         case AGX_OPCODE_BREAK:
         case AGX_OPCODE_BREAK_IF_ICMP:
         case AGX_OPCODE_BREAK_IF_FCMP:
         case AGX_OPCODE_ELSE_ICMP:
         case AGX_OPCODE_ELSE_FCMP:
         case AGX_OPCODE_POP_EXEC:
         case AGX_OPCODE_JMP_EXEC_ANY:
         case AGX_OPCODE_JMP_EXEC_NONE:
         case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
            if (block == tail)
               return NULL;
            break;
         default:
            /* Includes depth/coverage writes, pixel releases, atomics,
             * stores and barriers. Reads and pure ALU remain in place.
             */
            if (!agx_opcodes_info[I->op].can_eliminate)
               return NULL;
            break;
         }
      }
   }
   return tile && stop && !early_wait ? tail : NULL;
}
