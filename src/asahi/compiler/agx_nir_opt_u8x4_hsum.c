/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>

#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"
#include "nir_builder.h"

struct u8x4_hsum_load {
   nir_intrinsic_instr *load;
   unsigned order;
   uint8_t chan_mask;
   unsigned count;
};

static bool
is_single_use_iadd_in_block(nir_def *def, nir_block *block)
{
   if (def->num_components != 1 || def->bit_size != 32)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->block != block || instr->type != nir_instr_type_alu)
      return false;
   if (nir_instr_as_alu(instr)->op != nir_op_iadd)
      return false;
   return list_is_singular(&def->uses);
}

static bool
is_iadd_root_in_block(nir_alu_instr *alu, nir_block *block)
{
   if (alu->op != nir_op_iadd || alu->def.num_components != 1 ||
       alu->def.bit_size != 32)
      return false;

   if (!list_is_singular(&alu->def.uses))
      return true;

   nir_src *use = list_first_entry(&alu->def.uses, nir_src, use_link);
   if (nir_src_is_if(use))
      return true;
   nir_instr *parent = nir_src_use_instr(use);
   if (parent->block != block || parent->type != nir_instr_type_alu)
      return true;
   nir_alu_instr *palu = nir_instr_as_alu(parent);
   return !(palu->op == nir_op_iadd && palu->def.num_components == 1 &&
            palu->def.bit_size == 32);
}

static nir_intrinsic_instr *
match_u8x4_load_elem(nir_scalar leaf, nir_block *block, uint8_t *out_comp)
{
   if (leaf.def->num_components != 1 || leaf.def->bit_size != 32)
      return NULL;
   nir_instr *instr = nir_def_instr(leaf.def);
   if (instr->type != nir_instr_type_alu)
      return NULL;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_u2u32 || !list_is_singular(&alu->def.uses))
      return NULL;

   nir_scalar sc = nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
   nir_def *src = sc.def;
   if (src->bit_size != 8 || src->num_components != 4)
      return NULL;
   nir_instr *src_instr = nir_def_instr(src);
   if (src_instr->block != block || src_instr->type != nir_instr_type_intrinsic)
      return NULL;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(src_instr);
   if (intr->intrinsic != nir_intrinsic_load_agx ||
       nir_intrinsic_format(intr) != PIPE_FORMAT_R8_UINT ||
       nir_intrinsic_sign_extend(intr) ||
       (nir_intrinsic_base(intr) & (AGX_NIR_MEMORY_PACKED_COUNT_MASK |
                                    AGX_NIR_MEMORY_UNPACK_W8_WORD)))
      return NULL;

   *out_comp = sc.comp;
   return intr;
}

/* Leaves are scalars: a 32-bit iadd source may read any component of a
 * vector def, so the component is kept alongside the def. */
static void
collect_iadd_leaves(nir_scalar leaf, nir_block *block,
                    struct util_dynarray *leaves, unsigned depth)
{
   if (depth > 0 && is_single_use_iadd_in_block(leaf.def, block)) {
      nir_alu_instr *alu = nir_def_as_alu(leaf.def);
      for (unsigned i = 0; i < 2; ++i) {
         collect_iadd_leaves(nir_get_scalar(alu->src[i].src.ssa,
                                            alu->src[i].swizzle[0]),
                             block, leaves, depth - 1);
      }
      return;
   }
   util_dynarray_append(leaves, leaf);
}

static nir_cf_node *
outermost_enclosing_loop(nir_block *block)
{
   nir_cf_node *outer = NULL;
   for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n->type == nir_cf_node_loop)
         outer = n;
   }
   return outer;
}

static int
cmp_u8x4_hsum_load(const void *a, const void *b)
{
   const struct u8x4_hsum_load *la = a;
   const struct u8x4_hsum_load *lb = b;
   return (la->order > lb->order) - (la->order < lb->order);
}

/*
 * Lower associative 32-bit sums of all 4 unsigned bytes from packed R8_UINT
 * 4-byte load_agx intrinsics using SWAR 16x2 horizontal byte-pair sums:
 *
 *   w    = pack_32_4x8(load_agx 8x4)   // zero-copy raw packed word
 *   pair = (w & 0x00ff00ff) + ((w >> 8) & 0x00ff00ff)
 *
 * Each 16-bit lane of `pair` is <= 510, so up to 128 pairs (<= 65280 < 2^16)
 * can be summed directly in 32-bit before unpacking the two 16-bit lanes once
 * and adding to the 32-bit accumulator. Exact modulo 2^32 for all inputs.
 */
bool
agx_nir_opt_u8x4_hsum(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = false;
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         /* Assign per-block instruction order for dominance-preserving sorting. */
         unsigned idx = 0;
         nir_foreach_instr(instr, block)
            instr->index = idx++;

         nir_def *block_mask = NULL;

         nir_foreach_instr_reverse_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *root = nir_instr_as_alu(instr);
            if (!is_iadd_root_in_block(root, block))
               continue;

            struct util_dynarray leaves;
            util_dynarray_init(&leaves, NULL);
            for (unsigned i = 0; i < 2; ++i) {
               collect_iadd_leaves(nir_get_scalar(root->src[i].src.ssa,
                                                  root->src[i].swizzle[0]),
                                   block, &leaves, 128);
            }

            if (util_dynarray_num_elements(&leaves, nir_scalar) < 4) {
               util_dynarray_fini(&leaves);
               continue;
            }

            struct util_dynarray loads;
            util_dynarray_init(&loads, NULL);

            util_dynarray_foreach(&leaves, nir_scalar, lp) {
               uint8_t comp = 0;
               nir_intrinsic_instr *ld = match_u8x4_load_elem(*lp, block, &comp);
               if (!ld)
                  continue;
               bool found = false;
               util_dynarray_foreach(&loads, struct u8x4_hsum_load, e) {
                  if (e->load == ld) {
                     e->chan_mask |= (1u << comp);
                     e->count++;
                     found = true;
                     break;
                  }
               }
               if (!found) {
                  struct u8x4_hsum_load e = {
                     .load = ld,
                     .order = ld->instr.index,
                     .chan_mask = (uint8_t)(1u << comp),
                     .count = 1,
                  };
                  util_dynarray_append(&loads, e);
               }
            }

            /* Keep only complete 4-channel loads whose sole uses in the entire
             * shader are these 4 u2u32 leaves (so unpacking is eliminated). */
            struct util_dynarray valid_loads;
            util_dynarray_init(&valid_loads, NULL);
            util_dynarray_foreach(&loads, struct u8x4_hsum_load, e) {
               if (e->chan_mask != 0xf || e->count != 4)
                  continue;
               unsigned total_uses = 0;
               nir_foreach_use(u, &e->load->def)
                  total_uses++;
               if (total_uses == 4)
                  util_dynarray_append(&valid_loads, *e);
            }
            util_dynarray_fini(&loads);

            unsigned n_valid =
               util_dynarray_num_elements(&valid_loads, struct u8x4_hsum_load);
            if (n_valid == 0) {
               util_dynarray_fini(&valid_loads);
               util_dynarray_fini(&leaves);
               continue;
            }

            qsort(valid_loads.data, n_valid, sizeof(struct u8x4_hsum_load),
                  cmp_u8x4_hsum_load);

            /* Emit mask once before the outermost enclosing loop, or after the
             * phis that start the block: phis must stay first in a block. */
            if (!block_mask) {
               nir_cf_node *outer_loop = outermost_enclosing_loop(block);
               b.cursor = outer_loop ? nir_before_cf_node(outer_loop)
                                     : nir_after_phis(block);
               block_mask = nir_imm_int(&b, 0x00ff00ff);
            }

            /* Accumulate SWAR pairs in chunks of at most 128 loads (128 * 510
             * = 65280 <= 65535), emitting each pair right after its load to
             * keep live ranges minimal. */
            nir_def *acc = NULL;
            util_dynarray_foreach(&leaves, nir_scalar, lp) {
               uint8_t comp = 0;
               nir_intrinsic_instr *ld = match_u8x4_load_elem(*lp, block, &comp);
               bool consumed = false;
               if (ld) {
                  util_dynarray_foreach(&valid_loads, struct u8x4_hsum_load, e) {
                     if (e->load == ld) {
                        consumed = true;
                        break;
                     }
                  }
               }
               if (!consumed) {
                  b.cursor = nir_before_instr(&root->instr);
                  nir_def *leaf = nir_channel(&b, lp->def, lp->comp);
                  acc = acc ? nir_iadd(&b, acc, leaf) : leaf;
               }
            }

            nir_def *pair_sum = NULL;
            unsigned chunk_cnt = 0;
            util_dynarray_foreach(&valid_loads, struct u8x4_hsum_load, e) {
               b.cursor = nir_after_instr(&e->load->instr);
               nir_def *w = nir_pack_32_4x8(&b, &e->load->def);
               nir_def *even = nir_iand(&b, w, block_mask);
               nir_def *odd = nir_iand(&b, nir_ushr_imm(&b, w, 8), block_mask);
               nir_def *pair = nir_iadd(&b, even, odd);

               if (chunk_cnt == 128) {
                  b.cursor = nir_before_instr(&root->instr);
                  nir_def *lo = nir_u2u32(&b, nir_unpack_32_2x16_split_x(&b, pair_sum));
                  nir_def *hi = nir_u2u32(&b, nir_unpack_32_2x16_split_y(&b, pair_sum));
                  acc = acc ? nir_iadd(&b, acc, lo) : lo;
                  acc = nir_iadd(&b, acc, hi);
                  pair_sum = NULL;
                  chunk_cnt = 0;
               }

               b.cursor = nir_after_instr(nir_def_instr(pair));
               pair_sum = pair_sum ? nir_iadd(&b, pair_sum, pair) : pair;
               chunk_cnt++;
            }

            b.cursor = nir_before_instr(&root->instr);
            nir_def *lo = nir_u2u32(&b, nir_unpack_32_2x16_split_x(&b, pair_sum));
            nir_def *hi = nir_u2u32(&b, nir_unpack_32_2x16_split_y(&b, pair_sum));
            acc = acc ? nir_iadd(&b, acc, lo) : lo;
            acc = nir_iadd(&b, acc, hi);

            nir_def_rewrite_uses(&root->def, acc);
            impl_progress = true;

            util_dynarray_fini(&valid_loads);
            util_dynarray_fini(&leaves);
         }
      }

      if (impl_progress) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      } else {
         nir_no_progress(impl);
      }
   }

   return progress;
}
