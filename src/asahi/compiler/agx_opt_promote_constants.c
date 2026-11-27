/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx3_uniform_integer_instr.h"
#include "util/u_debug.h"
#include "agx_opcodes.h"

/*
 * Information about a constant, indexed by its 64-bit value. This describes the
 * value, not the move that generated it. If there are multiple moves in the
 * shader with the same immediate value, they resolve to the same constant.
 */
struct constant_info {
   uint64_t value;

   /* Number of uses of the constant that could be promoted */
   unsigned nr_promotable_uses;

   /* Order of the constant's first promotable use in the shader */
   unsigned first_use;

   /* If we push, the uniform used */
   uint16_t uniform;

   /* Alignment in 16-bit units needed for the constant */
   uint8_t align_16;

   /* True if the constant was promoted to a uniform */
   bool promoted;
};

/*
 * Choosing constants to promote is similar to the 0-1 knapsack problem. We use
 * a well-known heuristic: sort by benefit divided by size. We approximate
 * benefit by use count.
 */
static int
constant_priority(const struct constant_info *const info)
{
   int size = info->align_16;
   assert(size == 1 || size == 2 || size == 4);
   int inverse_size = (size == 1) ? 4 : (size == 2) ? 2 : 1;

   return info->nr_promotable_uses * inverse_size;
}

static int
priority_compare(const void *A_, const void *B_)
{
   const struct constant_info *const *A = A_;
   const struct constant_info *const *B = B_;

   /* This is backwards from qsort's documentation, because we want descending
    * order and qsort returns ascending.
    */
   int delta = constant_priority(*B) - constant_priority(*A);
   if (delta)
      return delta;

   /* Break ties by first use. The array is gathered from a hash table whose
    * iteration order depends on the host's pointer width and qsort is not
    * stable, so without this the uniform assignment, and therefore the
    * shader binary, would depend on the host the compiler runs on.
    */
   return (int)(*A)->first_use - (int)(*B)->first_use;
}

static void
record_use(void *memctx, struct hash_table_u64 *constants, uint64_t imm,
           enum agx_size size)
{
   struct constant_info *info = _mesa_hash_table_u64_search(constants, imm);

   if (!info) {
      info = rzalloc(memctx, struct constant_info);
      info->value = imm;
      info->first_use = _mesa_hash_table_u64_num_entries(constants);
      _mesa_hash_table_u64_insert(constants, imm, info);
   }

   info->nr_promotable_uses++;
   info->align_16 = MAX2(info->align_16, agx_size_align_16(size));
}

/*
 * On AGX3B, a 16-bit float ALU operand read from the uniform file inside a
 * loop runs slower than reading it from a register (matching
 * agx_nir_opt_preamble's keep_half_loop_operand_in_registers). Keep 16-bit
 * constants in GPRs for in-loop half-float ALU uses so late compiler passes
 * such as agx_opt_agx3_normflt_byte_pairs do not re-introduce uniform-file
 * half-float reads inside hot loops.
 */
static bool
agx3b_avoids_half_float_uniform_in_loop(const agx_context *ctx,
                                        const agx_instr *I, enum agx_size size,
                                        bool block_in_loop)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B || !block_in_loop ||
       size != AGX_SIZE_16)
      return false;

   switch (I->op) {
   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_FFMA:
      return true;
   default:
      return false;
   }
}

static bool *
build_in_loop_map(agx_context *ctx, void *memctx, unsigned *nr_blocks_out)
{
   unsigned nr_blocks = 0;
   agx_foreach_block(ctx, block)
      nr_blocks = MAX2(nr_blocks, block->index + 1);
   *nr_blocks_out = nr_blocks;

   bool *in_loop = rzalloc_array(memctx, bool, MAX2(nr_blocks, 1u));
   agx_foreach_block(ctx, block) {
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index) {
            for (unsigned i = succ->index; i <= block->index; ++i)
               in_loop[i] = true;
         }
      }
      if (block->loop_header)
         in_loop[block->index] = true;
   }
   return in_loop;
}

/* Reuse the packer's full instruction-domain proof for AGX3 integer operands.
 * Do not broaden the opcode-only predicate: modifiers, operand position and
 * the opposite source's type are part of the native instruction contract.
 * Private qualification switch; shader cache must be disabled in both arms.
 */
static bool
constant_use_accepts_uniform(agx_context *ctx, const agx_instr *I, unsigned s,
                             unsigned uniform, enum agx_size size,
                             bool block_in_loop)
{
   if (agx3b_avoids_half_float_uniform_in_loop(ctx, I, size, block_in_loop))
      return false;

   if (agx_instr_accepts_uniform(ctx->key->dev.encoding, I->op, s, uniform, size))
      return true;

   /* M5 floating-point operations read an aligned f32 uniform exactly where
    * the native uniform-operand encoder admits it. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       agx3_m5_float_accepts_uniform(ctx, I, s, uniform, size))
      return true;

   /* M5 32-bit integer compare-select reads an aligned 32-bit uniform in its
    * right comparand and/or either selected operand natively. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       agx3_m5_isel_accepts_uniform(ctx, I, s, uniform, size))
      return true;

   if (I->nr_srcs > 3 ||
       uniform >= AGX_NUM_UNIFORMS)
      return false;

   agx_instr candidate = *I;
   agx_index sources[3];
   memcpy(sources, I->src, I->nr_srcs * sizeof(*sources));
   candidate.src = sources;
   sources[s] = agx_replace_index(sources[s], agx_uniform(uniform, size));
   struct agx3_uniform_integer_desc desc;
   return agx3_uniform_integer_for_encoding(ctx->key->dev.encoding,
                                            &candidate, false, &desc);
}

/* The native R/U/U IMADD endpoint can promote two constants together.
 * Probing C alone would falsely reject it before the multiplier is promoted.
 * Do not admit the unwitnessed R/R/U endpoint to make that probe pass. */
static bool
m5_constant_mad_pair(agx_context *ctx, const agx_instr *I,
                     const BITSET_WORD *is_immediate, unsigned uniform,
                     unsigned *multiplier)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       I->op != AGX_OPCODE_IMAD || I->nr_srcs != 3 || uniform > 508 ||
       I->src[0].type != AGX_INDEX_NORMAL ||
       I->src[1].type != AGX_INDEX_NORMAL ||
       I->src[2].type != AGX_INDEX_NORMAL ||
       !BITSET_TEST(is_immediate, I->src[2].value))
      return false;
   const bool a = BITSET_TEST(is_immediate, I->src[0].value);
   const bool b = BITSET_TEST(is_immediate, I->src[1].value);
   if (a == b)
      return false;
   *multiplier = a ? 0 : 1;
   agx_instr candidate = *I;
   agx_index sources[3];
   memcpy(sources, I->src, sizeof(sources));
   candidate.src = sources;
   sources[*multiplier] = agx_replace_index(
      sources[*multiplier], agx_uniform(uniform, I->src[*multiplier].size));
   sources[2] = agx_replace_index(sources[2],
                                agx_uniform(uniform + 2, I->src[2].size));
   struct agx3_uniform_integer_desc desc;
   return agx3_uniform_integer_for_encoding(ctx->key->dev.encoding,
                                            &candidate, false, &desc);
}

/* The late M5 round promotes only float constants that M5 constant
 * materialization left in GPRs for native uniform-operand forms. */
static bool
late_m5_float_use(agx_context *ctx, const agx_instr *I, unsigned s,
                  unsigned uniform, enum agx_size size)
{
   return (I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL ||
           I->op == AGX_OPCODE_FFMA) &&
          agx3_m5_float_accepts_uniform(ctx, I, s, uniform, size);
}

static void
pass(agx_context *ctx, void *memctx, bool late_m5_float)
{
   /* Map from SSA indices to struct constant_info */
   struct hash_table_u64 *constants = _mesa_hash_table_u64_create(memctx);

   /* Map from SSA indices to immediate values */
   uint64_t *values = rzalloc_array(memctx, uint64_t, ctx->alloc);

   /* Set of SSA indices that map to immediate values */
   BITSET_WORD *is_immediate = BITSET_RZALLOC(memctx, ctx->alloc);

   /* Probe uniform acceptance where the first promoted constant will land:
    * the range starts 32-bit aligned on AGX3, and aligned f32 operands
    * (FFMA, FADD, FMUL, FCMPSEL) refuse an odd index. Probing the raw count
    * after a preamble that ends on a 16-bit export refused every such use.
    */
   const unsigned probe_uniform =
      ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2
         ? ALIGN_POT(ctx->out->push_count, 2)
         : ctx->out->push_count;
   const unsigned push_before = ctx->out->push_count;

   unsigned nr_blocks = 0;
   bool *in_loop = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B
                      ? build_in_loop_map(ctx, memctx, &nr_blocks)
                      : NULL;

   /* Gather constant definitions and use */
   agx_foreach_block(ctx, block) {
      const bool block_in_loop =
         in_loop && block->index < nr_blocks && in_loop[block->index];
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_MOV_IMM) {
            assert(I->dest[0].type == AGX_INDEX_NORMAL);
            BITSET_SET(is_immediate, I->dest[0].value);
            values[I->dest[0].value] = I->imm;
         } else if (late_m5_float) {
            agx_foreach_ssa_src(I, s) {
               if (BITSET_TEST(is_immediate, I->src[s].value) &&
                   late_m5_float_use(ctx, I, s, probe_uniform, I->src[s].size))
                  record_use(memctx, constants, values[I->src[s].value],
                             I->src[s].size);
            }
         } else {
            unsigned multiplier;
            if (m5_constant_mad_pair(ctx, I, is_immediate, probe_uniform,
                                     &multiplier)) {
               record_use(memctx, constants, values[I->src[multiplier].value],
                          I->src[multiplier].size);
               record_use(memctx, constants, values[I->src[2].value],
                          I->src[2].size);
               continue;
            }
            agx_foreach_ssa_src(I, s) {
               if (BITSET_TEST(is_immediate, I->src[s].value) &&
                   constant_use_accepts_uniform(ctx, I, s, probe_uniform,
                                                I->src[s].size,
                                                block_in_loop)) {

                  record_use(memctx, constants, values[I->src[s].value],
                             I->src[s].size);
               }
            }
         }
      }
   }

   /* Early exit if there were no constants */
   unsigned nr_nodes = _mesa_hash_table_u64_num_entries(constants);
   if (nr_nodes == 0)
      return;

   /* Collect nodes that are promotable */
   struct constant_info **flat =
      rzalloc_array(memctx, struct constant_info *, nr_nodes);

   unsigned flat_count = 0;
   hash_table_u64_foreach(constants, entry) {
      flat[flat_count++] = entry.data;
   }

   /* Select constants. Even when we can promote everything, sorting keeps hot
    * constants in lower uniforms, required by some instructions.
    */
   qsort(flat, flat_count, sizeof(*flat), priority_compare);

   const unsigned max_uniforms =
      ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
            ctx->stage == MESA_SHADER_FRAGMENT
         ? AGX3_FRAGMENT_UNIFORM_END
         : AGX_NUM_UNIFORMS;

   /* Promote as many constants as we can */
   for (unsigned i = 0; i < flat_count; ++i) {
      struct constant_info *info = flat[i];
      assert(info->nr_promotable_uses > 0);

      /* Try to assign a uniform */
      /* Native Ioto publishes 32-bit uniform words. Align the beginning of
       * the constant range even when its first constant is only 16 bits. */
      unsigned range_align = ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
                             i == 0 ? MAX2(info->align_16, 2) : info->align_16;
      unsigned uniform = ALIGN_POT(ctx->out->push_count, range_align);
      unsigned new_count = uniform + info->align_16;
      if (new_count > max_uniforms)
         break;

      /* Set the base for the first constant. This avoids uploading an extra
       * zero if we needed to pad, compared to doing this before the loop. Same
       * shader code, but slightly more efficient USC record.
       */
      if (i == 0 && !(late_m5_float && ctx->out->rodata.size_16)) {
         ctx->out->rodata.base_uniform = uniform;
      }

      info->uniform = uniform;
      info->promoted = true;
      ctx->out->push_count = new_count;

      unsigned size_B = info->align_16 * 2;
      unsigned uniform_offs = uniform - ctx->out->rodata.base_uniform;
      assert(uniform_offs < ARRAY_SIZE(ctx->rodata));
      memcpy(&ctx->rodata[uniform_offs], &info->value, size_B);

      ctx->out->rodata.size_16 = new_count - ctx->out->rodata.base_uniform;
   }

   /* Keep the trailing half inside the owned constant blob and uniform
    * allocation; the native uploader cannot load half of a 32-bit word. */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       (ctx->out->rodata.size_16 & 1)) {
      ctx->rodata[ctx->out->rodata.size_16++] = 0;
      ctx->out->push_count++;
   }

   /* An inline-blend recompile must leave its constants where the split
    * main program has them (see agx_shader_key::g17p_rodata_hint). Keep the
    * ordinary placement whenever it already does; otherwise take each
    * constant's slot from the main's table. */
   const uint16_t *hint = ctx->key->g17p_rodata_hint;
   const unsigned hint_base = ctx->key->g17p_rodata_hint_base;
   const unsigned hint_size = ctx->key->g17p_rodata_hint_size_16;
   if (hint && hint_size && ctx->out->rodata.size_16 &&
       !(ctx->out->rodata.base_uniform == hint_base &&
         ctx->out->rodata.size_16 <= hint_size &&
         !memcmp(ctx->rodata, hint, ctx->out->rodata.size_16 * 2)) &&
       push_before <= hint_base) {
      unsigned end = 0;
      for (unsigned i = 0; i < flat_count; ++i) {
         struct constant_info *info = flat[i];
         info->promoted = false;
         const unsigned align = info->align_16;
         for (unsigned k = 0; k + align <= hint_size; ++k) {
            if ((hint_base + k) % align ||
                memcmp(&hint[k], &info->value, align * 2))
               continue;
            info->uniform = hint_base + k;
            info->promoted = true;
            end = MAX2(end, k + align);
            break;
         }
      }

      /* Whole 32-bit words, as above, taken from the main's table. */
      if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && (end & 1))
         end++;
      assert(end <= hint_size && end <= ARRAY_SIZE(ctx->rodata));
      memcpy(ctx->rodata, hint, end * 2);
      ctx->out->rodata.base_uniform = end ? hint_base : 0;
      ctx->out->rodata.size_16 = end;
      ctx->out->push_count = end ? hint_base + end : push_before;
   }

   /* Promote in the IR */
   agx_foreach_block(ctx, block) {
      const bool block_in_loop =
         in_loop && block->index < nr_blocks && in_loop[block->index];
      agx_foreach_instr_in_block(block, I) {
         if (late_m5_float) {
            agx_foreach_ssa_src(I, s) {
               if (!BITSET_TEST(is_immediate, I->src[s].value))
                  continue;
               struct constant_info *info =
                  _mesa_hash_table_u64_search(constants, values[I->src[s].value]);
               if (info && info->promoted &&
                   late_m5_float_use(ctx, I, s, info->uniform, I->src[s].size))
                  agx_replace_src(I, s,
                                  agx_uniform(info->uniform, I->src[s].size));
            }
            continue;
         }
         unsigned multiplier;
         if (m5_constant_mad_pair(ctx, I, is_immediate, probe_uniform,
                                  &multiplier)) {
            struct constant_info *m = _mesa_hash_table_u64_search(
               constants, values[I->src[multiplier].value]);
            struct constant_info *c = _mesa_hash_table_u64_search(
               constants, values[I->src[2].value]);
            if (m && c && m->promoted && c->promoted) {
               /* Both values own aligned uniforms before either source changes. */
               agx_replace_src(I, multiplier,
                               agx_uniform(m->uniform, I->src[multiplier].size));
               agx_replace_src(I, 2, agx_uniform(c->uniform, I->src[2].size));
               continue;
            }
         }
         agx_foreach_ssa_src(I, s) {
            if (!BITSET_TEST(is_immediate, I->src[s].value))
               continue;

            struct constant_info *info =
               _mesa_hash_table_u64_search(constants, values[I->src[s].value]);

            if (info && info->promoted &&
                constant_use_accepts_uniform(ctx, I, s,
                                             info->uniform, I->src[s].size,
                                             block_in_loop)) {

               agx_replace_src(I, s, agx_uniform(info->uniform, I->src[s].size));
            }
         }
      }
   }
}

void
agx_opt_promote_constants(agx_context *ctx)
{
   /* We do not promote constants in preambles since it's pointless and wastes
    * uniform slots.
    */
   if (ctx->is_preamble)
      return;

   void *memctx = ralloc_context(NULL);
   pass(ctx, memctx, false);
   ralloc_free(memctx);
}

/*
 * M5 constant materialization (agx_materialize_agx3_alu_constants) runs after
 * the ordinary round and moves immediates that no M5 form can read into GPRs,
 * e.g. the 0.25 multiplicand of Minecraft's RGSS fma(x, 0.25, -y). Apple keeps
 * such constants in uniforms. Promote those feeding a native M5 float
 * uniform-operand form, appending to the ordinary round's constant range so
 * the program keeps one contiguous rodata upload.
 */
void
agx_opt_promote_m5_float_constants(agx_context *ctx)
{
   const struct agx_rodata *rodata = &ctx->out->rodata;
   if (ctx->is_preamble ||
       ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       (ctx->key->g17p_rodata_hint && ctx->key->g17p_rodata_hint_size_16) ||
       (rodata->size_16 &&
        ctx->out->push_count != rodata->base_uniform + rodata->size_16))
      return;

   void *memctx = ralloc_context(NULL);
   pass(ctx, memctx, true);
   ralloc_free(memctx);
}
