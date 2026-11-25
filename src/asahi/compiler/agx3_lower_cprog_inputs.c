/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"

static bool
mark(uint64_t *mask, unsigned first, unsigned count)
{
   if (!count || first >= 512 || count > 512 - first)
      return false;
   for (unsigned h = first; h < first + count; ++h)
      mask[h / 64] |= UINT64_C(1) << (h % 64);
   return true;
}

static agx_cursor
import_cursor(agx_block *block, agx_instr *I, unsigned source)
{
   if (I->op == AGX_OPCODE_PHI) {
      agx_block **preds = util_dynarray_begin(&block->predecessors);
      return agx_after_block_logical(preds[source]);
   }
   /* Keep the terminal export run contiguous. */
   if (I->op == AGX_OPCODE_EXPORT) {
      while (I->link.prev != &block->instructions) {
         agx_instr *prev = list_entry(I->link.prev, agx_instr, link);
         if (prev->op != AGX_OPCODE_EXPORT)
            break;
         I = prev;
      }
   }
   return agx_before_instr(I);
}

bool
agx3_lower_cprog_inputs(agx_context *ctx)
{
   if (!ctx->is_preamble || !ctx->out || ctx->agx3_cprog_inputs_lowered ||
       !agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;

   uint64_t reads[8] = {0}, writes[8] = {0};
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_PHI &&
             I->nr_srcs != agx_num_predecessors(block))
            return false;
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type == AGX_INDEX_UNIFORM &&
                (src.memory || !mark(reads, src.value, agx_index_size_16(src))))
               return false;
         }
         if (I->op == AGX_OPCODE_UNIFORM_STORE) {
            if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE ||
                I->src[1].abs || I->src[1].neg || (I->mask & ~15u))
               return false;
            unsigned stride = agx_size_align_16(I->src[0].size);
            for (unsigned c = 0; c < 4; ++c) {
               if ((I->mask & (1u << c)) &&
                   !mark(writes, I->src[1].value + c * stride, stride))
                  return false;
            }
         } else if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
            if (I->imm >= 512 || !mark(writes, I->imm, 2))
               return false;
         }
      }
   }
   /* Extracted NIR keeps intermediate values in SSA. Its imported uniform
    * arguments are immutable and disjoint from exported uniform slots.
    * Refuse noncanonical physical-uniform read/modify/write instead of
    * replacing an updated value with an invocation-entry snapshot. */
   for (unsigned w = 0; w < 8; ++w) {
      if (reads[w] & writes[w])
         return false;
   }
   struct agx3_cprog_inputs inputs;
   if (!agx3_cprog_inputs_init(reads, &inputs))
      return false;
   if (!inputs.count)
      return true;

   const unsigned prefix = inputs.storage_bytes;
   if (ctx->scratch_size_B > 65536u - prefix)
      return false;
   /* Validate all existing scratch rebases before mutating the IR. Spiller
    * allocations subsequently start after both this prefix and local data. */
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_STACK_LOAD && I->op != AGX_OPCODE_STACK_STORE)
         continue;
      unsigned s = I->op == AGX_OPCODE_STACK_STORE ? 1 : 0;
      if (I->nr_srcs <= s)
         return false;
      agx_index offset = I->src[s];
      if (offset.memory || offset.neg || offset.cache || offset.discard ||
          offset.kill || agx_channels(offset) != 1)
         return false;
      if (offset.type == AGX_INDEX_IMMEDIATE) {
         if (offset.size != AGX_SIZE_16 || offset.value > UINT16_MAX - prefix)
            return false;
      } else if ((offset.type != AGX_INDEX_NORMAL &&
                  offset.type != AGX_INDEX_REGISTER &&
                  offset.type != AGX_INDEX_UNIFORM) ||
                 offset.size != AGX_SIZE_32 || offset.abs) {
         return false;
      }
   }

   ctx->out->agx3_preamble_inputs = inputs;
   ctx->out->agx3_preamble_uniform_inputs = true;
   ctx->out->agx3_preamble_tps_inputs = true;
   ctx->agx3_cprog_inputs_lowered = true;
   ctx->scratch_size_B += prefix;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, I) {
         agx_foreach_src(I, s) {
            agx_index src = I->src[s];
            if (src.type != AGX_INDEX_UNIFORM)
               continue;
            agx_builder b = agx_init_builder(ctx, import_cursor(block, I, s));
            uint16_t tps;
            bool found = agx3_cprog_input_tps_half(&inputs, src.value, &tps);
            assert(found);
            (void)found;
            /* A compact word can begin at an odd half after a sparse import.
             * Use two real half loads in that case, with independent SSA
             * results/dependencies, then let RA coalesce the collection. */
            unsigned piece = src.size != AGX_SIZE_16 && !(tps & 1) ? 2 : 1;
            unsigned count = agx_index_size_16(src) / piece;
            agx_index value = agx_vec_temp(ctx, src.size, agx_channels(src));
            agx_index parts[32];
            assert(count <= ARRAY_SIZE(parts));
            for (unsigned p = 0; p < count; ++p) {
               parts[p] = count == 1 ? value :
                  agx_temp(ctx, piece == 2 ? AGX_SIZE_32 : AGX_SIZE_16);
               agx_instr *load = agx_alloc_instr(&b, AGX_OPCODE_LOAD_CPROG_AGX3, 1, 0);
               load->dest[0] = parts[p];
               load->imm = src.value + p * piece;
               agx_builder_insert(&b.cursor, load);
            }
            if (count > 1) {
               agx_instr *collect = agx_collect_to(&b, value, count);
               for (unsigned p = 0; p < count; ++p)
                  collect->src[p] = parts[p];
            }
            agx_replace_src(I, s, value);
         }
      }
   }

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_STACK_LOAD && I->op != AGX_OPCODE_STACK_STORE)
         continue;
      unsigned s = I->op == AGX_OPCODE_STACK_STORE ? 1 : 0;
      agx_index offset = I->src[s];
      if (offset.type == AGX_INDEX_IMMEDIATE) {
         I->src[s] = agx_immediate(offset.value + prefix);
      } else {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[s] = agx_iadd(&b, offset, agx_mov_imm(&b, 32, prefix), 0);
      }
   }
   return true;
}

/* A proof version names this complete provenance + final-IR contract. New TPS
 * instruction families must be classified here before this contract can cover
 * them. In particular, device atomics can hide a TPS return slot. */
static bool
cprog_other_storage(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_ATOMIC:
   case AGX_OPCODE_LOCAL_ATOMIC:
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
   case AGX_OPCODE_STACK_MAP:
   case AGX_OPCODE_STACK_UNMAP:
   case AGX_OPCODE_STACK_ADJUST:
   case AGX_OPCODE_SCS_LOAD_AGX3:
   case AGX_OPCODE_SCS_STORE_AGX3:
   case AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3:
   case AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3:
   case AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3:
   case AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3:
   case AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3:
   case AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3:
   case AGX_OPCODE_SCS_TOUCH_REGISTER_INDEXED_AGX3:
   case AGX_OPCODE_SCS_TOUCH_REGISTER_BASE_AGX3:
   case AGX_OPCODE_INTERSECT_RAY_M5:
   case AGX_OPCODE_CONTINUE_RAY_M5:
      return true;
   default:
      return false;
   }
}

static bool
cprog_canonical_inputs(const struct agx_shader_info *info,
                      struct agx3_cprog_inputs *canonical)
{
   return info && info->agx3_preamble_uniform_inputs &&
          info->agx3_preamble_tps_inputs &&
          agx3_cprog_inputs_init(info->agx3_preamble_inputs.halves, canonical) &&
          canonical->count && canonical->count == info->agx3_preamble_inputs.count &&
          canonical->storage_bytes == info->agx3_preamble_inputs.storage_bytes;
}

static unsigned
cprog_static_spill_limit(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return AGX3_CPROG_STATIC_SPILL_MAX_B;
   case MESA_SHADER_FRAGMENT:
      return AGX3_CPROG_FS_STATIC_SPILL_MAX_B;
   default:
      return 0;
   }
}

bool
agx3_cprog_static_spill_info_valid(const struct agx_shader_info *info)
{
   struct agx3_cprog_inputs canonical;
   if (!info || !cprog_static_spill_limit(info->stage))
      return false;
   const unsigned version = info->stage == MESA_SHADER_FRAGMENT
      ? AGX3_CPROG_FS_STATIC_SPILL_VERSION : AGX3_CPROG_STATIC_SPILL_VERSION;
   return info->agx3_preamble_static_spill_version == version &&
          cprog_canonical_inputs(info, &canonical) &&
          info->preamble_scratch_size > canonical.storage_bytes &&
          info->preamble_scratch_size <= cprog_static_spill_limit(info->stage) &&
          !(info->preamble_scratch_size & 3);
}

bool
agx3_cprog_spill_provenance(const agx_context *ctx)
{
   struct agx3_cprog_inputs canonical;
   if (!ctx || !ctx->key || !ctx->is_preamble ||
       !cprog_static_spill_limit(ctx->stage) || !ctx->out ||
       !ctx->nir || ctx->nir->info.stage != ctx->stage || ctx->any_scratch ||
       ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       !ctx->agx3_cprog_inputs_lowered ||
       !cprog_canonical_inputs(ctx->out, &canonical) ||
       ctx->scratch_size_B != canonical.storage_bytes)
      return false;

   agx_foreach_instr_global(ctx, I) {
      if (cprog_other_storage(I) || I->op == AGX_OPCODE_STACK_LOAD ||
          I->op == AGX_OPCODE_STACK_STORE)
         return false;
      agx_foreach_src(I, s) {
         if (I->src[s].memory)
            return false;
      }
      agx_foreach_dest(I, d) {
         if (I->dest[d].memory)
            return false;
      }
   }
   return true;
}

bool
agx3_cprog_certify_static_spill(const agx_context *ctx,
                                bool imports_only_before_ra)
{
   struct agx3_cprog_inputs canonical;
   if (!imports_only_before_ra || !ctx || !ctx->key || !ctx->is_preamble ||
       !cprog_static_spill_limit(ctx->stage) || !ctx->out ||
       !ctx->nir || ctx->nir->info.stage != ctx->stage || ctx->any_scratch ||
       ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       !ctx->agx3_cprog_inputs_lowered ||
       !cprog_canonical_inputs(ctx->out, &canonical) ||
       ctx->spill_base_B != canonical.storage_bytes ||
       ctx->scratch_size_B <= ctx->spill_base_B ||
       ctx->scratch_size_B > cprog_static_spill_limit(ctx->stage) ||
       ctx->out->preamble_scratch_size != align(ctx->scratch_size_B, 4))
      return false;

   bool saw_import = false, saw_store = false, saw_load = false;
   uint64_t writes[8] = {0};
   agx_foreach_instr_global(ctx, I) {
      if (cprog_other_storage(I))
         return false;
      agx_foreach_src(I, s) {
         if (I->src[s].memory || I->src[s].type == AGX_INDEX_UNIFORM)
            return false;
      }
      agx_foreach_dest(I, d) {
         if (I->dest[d].memory)
            return false;
      }
      if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
         if (I->nr_dests != 1 || I->nr_srcs || I->saturate || I->imm >= 512)
            return false;
         agx_index dest = I->dest[0];
         unsigned halves = dest.size == AGX_SIZE_16 ? 1 : 2;
         uint16_t first, last;
         if (dest.type != AGX_INDEX_REGISTER || agx_channels(dest) != 1 ||
             (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
             dest.abs || dest.neg || dest.kill || dest.discard ||
             !agx3_cprog_input_tps_half(&canonical, I->imm, &first) ||
             !agx3_cprog_input_tps_half(&canonical, I->imm + halves - 1, &last) ||
             last != first + halves - 1 || (halves == 2 && (first & 1)) ||
             first < AGX3_CPROG_INPUT_TPS_BASE ||
             last >= AGX3_CPROG_INPUT_TPS_BASE + canonical.count)
            return false;
         saw_import = true;
      } else if (I->op == AGX_OPCODE_STACK_LOAD || I->op == AGX_OPCODE_STACK_STORE) {
         bool store = I->op == AGX_OPCODE_STACK_STORE;
         if (I->nr_dests != (store ? 0 : 1) || I->nr_srcs != (store ? 2 : 1) ||
             I->mask != 1 || (I->format != AGX_FORMAT_I16 && I->format != AGX_FORMAT_I32))
            return false;
         const unsigned bytes = I->format == AGX_FORMAT_I16 ? 2 : 4;
         agx_index offset = I->src[store ? 1 : 0];
         agx_index data = store ? I->src[0] : I->dest[0];
         if (offset.type != AGX_INDEX_IMMEDIATE || offset.size != AGX_SIZE_16 ||
             agx_channels(offset) != 1 || offset.memory || offset.neg ||
             offset.cache || offset.discard || offset.kill ||
             (offset.value & (bytes - 1)) || offset.value < ctx->spill_base_B ||
             offset.value > ctx->scratch_size_B - bytes ||
             data.type != AGX_INDEX_REGISTER || data.memory || data.abs || data.neg ||
             data.size != (bytes == 2 ? AGX_SIZE_16 : AGX_SIZE_32) ||
             agx_channels(data) != 1 || (data.cache && data.discard) ||
             (!store && (data.kill || data.discard)) ||
             data.value > (bytes == 2 ? 255 : 254) ||
             (bytes == 4 && (data.value & 1)))
            return false;
         saw_store |= store;
         saw_load |= !store;
      } else if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE ||
             I->src[1].abs || I->src[1].neg || (I->mask & ~15u))
            return false;
         unsigned stride = agx_size_align_16(I->src[0].size);
         for (unsigned c = 0; c < 4; ++c) {
            if ((I->mask & (1u << c)) &&
                !mark(writes, I->src[1].value + c * stride, stride))
               return false;
         }
      } else if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                 I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                 I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
         if (!mark(writes, I->imm, 2))
            return false;
      }
   }
   for (unsigned w = 0; w < 8; ++w) {
      if (writes[w] & canonical.halves[w])
         return false;
   }
   return saw_import && saw_store && saw_load;
}
