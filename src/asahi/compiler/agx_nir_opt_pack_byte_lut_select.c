/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Pack loop-invariant byte lookup select trees into 32-bit words.
 *
 * Indexing a small local byte array (8..16 elements) by a dynamic index inside
 * a loop lowers to a binary decision tree of N - 1 compare-selects that keeps
 * all N elements live across the loop. When every leaf is a loop-invariant
 * unsigned byte (or zero-extended unsigned byte) and the index is bounded to
 * [0, N - 1], pack the N bytes into ceil(N / 4) 32-bit words in the loop
 * preheader and replace the tree inside the loop with 1..3 word selects plus a
 * dynamic byte extract:
 *
 *    w   = select_word(idx, w0 .. w_{W-1})
 *    val = ubitfield_extract(w, (idx & 3) << 3, 8)
 *
 * This cuts both live register pressure across the loop (up to 4x) and dynamic
 * select count inside the loop body.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/set.h"

#define MAX_LUT_BYTES 16u
#define MIN_LUT_BYTES 8u
#define MIN_TREE_SELS 7u
#define MAX_TREE_DEPTH 16u

static bool
block_in_loop(const nir_block *block, const nir_loop *loop)
{
   for (const nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }
   return false;
}

/* Return an upper bound `max_idx` (in [MIN_LUT_BYTES - 1, MAX_LUT_BYTES - 1])
 * if `idx` is provably in [0, max_idx] for every input. */
static bool
proven_index_max(nir_scalar idx, unsigned *max_out)
{
   idx = nir_scalar_chase_movs(idx);
   if (idx.def->bit_size != 32 || !nir_scalar_is_alu(idx))
      return false;

   const nir_op op = nir_scalar_alu_op(idx);
   if (op == nir_op_iand || op == nir_op_umin) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar c = nir_scalar_chase_alu_src(idx, i);
         if (!nir_scalar_is_const(c))
            continue;
         const uint64_t val = nir_scalar_as_uint(c);
         if (val >= MIN_LUT_BYTES - 1 && val <= MAX_LUT_BYTES - 1) {
            *max_out = (unsigned)val;
            return true;
         }
      }
      return false;
   }

   if (op == nir_op_ubitfield_extract || op == nir_op_ubfe) {
      nir_scalar bits = nir_scalar_chase_alu_src(idx, 2);
      if (nir_scalar_is_const(bits)) {
         const uint64_t w = nir_scalar_as_uint(bits);
         if (w == 3 || w == 4) {
            *max_out = (1u << w) - 1u;
            return true;
         }
      }
      return false;
   }

   if (op == nir_op_ushr) {
      nir_scalar sh = nir_scalar_chase_alu_src(idx, 1);
      if (nir_scalar_is_const(sh)) {
         const unsigned s = nir_scalar_as_uint(sh) & 31u;
         if (s == 28 || s == 29) {
            *max_out = (1u << (32u - s)) - 1u;
            return true;
         }
      }
   }

   return false;
}

struct cmp_info {
   nir_op op;
   nir_scalar idx;
   int64_t imm;
   bool swapped; /* true when form is op(imm, idx) instead of op(idx, imm) */
};

static bool
parse_index_cmp(nir_scalar cond, struct cmp_info *out)
{
   cond = nir_scalar_chase_movs(cond);
   if (cond.def->bit_size != 1 || !nir_scalar_is_alu(cond))
      return false;

   const nir_op op = nir_scalar_alu_op(cond);
   switch (op) {
   case nir_op_ilt:
   case nir_op_ige:
   case nir_op_ult:
   case nir_op_uge:
   case nir_op_ieq:
   case nir_op_ine:
      break;
   default:
      return false;
   }

   nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
      return false;

   const bool is_unsigned = (op == nir_op_ult || op == nir_op_uge);
   if (nir_scalar_is_const(s1) && !nir_scalar_is_const(s0)) {
      out->op = op;
      out->idx = s0;
      out->imm = is_unsigned ? (int64_t)(uint32_t)nir_scalar_as_uint(s1)
                             : (int64_t)(int32_t)nir_scalar_as_uint(s1);
      out->swapped = false;
      return true;
   }
   if (nir_scalar_is_const(s0) && !nir_scalar_is_const(s1)) {
      out->op = op;
      out->idx = s1;
      out->imm = is_unsigned ? (int64_t)(uint32_t)nir_scalar_as_uint(s0)
                             : (int64_t)(int32_t)nir_scalar_as_uint(s0);
      out->swapped = true;
      return true;
   }

   return false;
}

static bool
eval_cmp(const struct cmp_info *ci, uint32_t k)
{
   const uint32_t u_lhs = ci->swapped ? (uint32_t)ci->imm : k;
   const uint32_t u_rhs = ci->swapped ? k : (uint32_t)ci->imm;
   const int32_t s_lhs = (int32_t)u_lhs;
   const int32_t s_rhs = (int32_t)u_rhs;

   switch (ci->op) {
   case nir_op_ilt: return s_lhs < s_rhs;
   case nir_op_ige: return s_lhs >= s_rhs;
   case nir_op_ult: return u_lhs < u_rhs;
   case nir_op_uge: return u_lhs >= u_rhs;
   case nir_op_ieq: return u_lhs == u_rhs;
   case nir_op_ine: return u_lhs != u_rhs;
   default:         UNREACHABLE("invalid cmp");
   }
}

/* Extract an unsigned 8-bit leaf (or a constant/masked value in [0, 255])
 * defined outside `loop` and dominating `pre`. */
static bool
extract_u8_invariant_leaf(nir_scalar s, nir_loop *loop, nir_block *pre,
                          nir_scalar *byte_out)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      const uint64_t v = nir_scalar_as_uint(s);
      if (v > 0xff)
         return false;
      *byte_out = s;
      return true;
   }

   if (s.def->bit_size == 8) {
      if (block_in_loop(nir_def_block(s.def), loop) ||
          !nir_block_dominates(nir_def_block(s.def), pre))
         return false;
      *byte_out = s;
      return true;
   }

   if (nir_scalar_is_alu(s)) {
      const nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_u2u32 || op == nir_op_u2u16) {
         nir_scalar src = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         if (src.def->bit_size == 8 &&
             !block_in_loop(nir_def_block(src.def), loop) &&
             nir_block_dominates(nir_def_block(src.def), pre)) {
            *byte_out = src;
            return true;
         }
      } else if (op == nir_op_iand) {
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar c = nir_scalar_chase_alu_src(s, i);
            if (nir_scalar_is_const(c) && (nir_scalar_as_uint(c) & ~0xffull) == 0 &&
                !block_in_loop(nir_def_block(s.def), loop) &&
                nir_block_dominates(nir_def_block(s.def), pre)) {
               *byte_out = s;
               return true;
            }
         }
      }
   }

   return false;
}

static bool
is_matching_bcsel(nir_scalar s, nir_block *block, nir_scalar idx,
                  struct cmp_info *ci_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_bcsel)
      return false;
   if (nir_def_block(s.def) != block || s.def->num_components != 1)
      return false;
   if (!parse_index_cmp(nir_scalar_chase_alu_src(s, 0), ci_out))
      return false;
   return nir_scalar_equal(ci_out->idx, idx);
}

static bool
collect_tree(nir_scalar node, nir_block *block, nir_loop *loop, nir_block *pre,
             nir_scalar idx, struct set *internal_sels, unsigned depth)
{
   if (depth > MAX_TREE_DEPTH)
      return false;

   node = nir_scalar_chase_movs(node);
   struct cmp_info ci;
   if (is_matching_bcsel(node, block, idx, &ci)) {
      if (_mesa_set_search(internal_sels, node.def))
         return true;
      _mesa_set_add(internal_sels, node.def);
      return collect_tree(nir_scalar_chase_alu_src(node, 1), block, loop, pre,
                          idx, internal_sels, depth + 1) &&
             collect_tree(nir_scalar_chase_alu_src(node, 2), block, loop, pre,
                          idx, internal_sels, depth + 1);
   }

   nir_scalar byte;
   return extract_u8_invariant_leaf(node, loop, pre, &byte);
}

static bool
eval_tree(nir_scalar node, nir_block *block, nir_loop *loop, nir_block *pre,
          nir_scalar idx, uint32_t k, nir_scalar *leaf_out, unsigned depth)
{
   if (depth > MAX_TREE_DEPTH)
      return false;

   node = nir_scalar_chase_movs(node);
   struct cmp_info ci;
   if (is_matching_bcsel(node, block, idx, &ci)) {
      const unsigned arm = eval_cmp(&ci, k) ? 1u : 2u;
      return eval_tree(nir_scalar_chase_alu_src(node, arm), block, loop, pre,
                       idx, k, leaf_out, depth + 1);
   }

   return extract_u8_invariant_leaf(node, loop, pre, leaf_out);
}

static nir_def *
emit_byte_as_u16(nir_builder *b, nir_scalar leaf)
{
   if (nir_scalar_is_const(leaf))
      return nir_imm_intN_t(b, (uint16_t)(nir_scalar_as_uint(leaf) & 0xffu), 16);

   nir_def *v = nir_channel(b, leaf.def, leaf.comp);
   if (v->bit_size == 8)
      return nir_u2u16(b, v);
   if (v->bit_size == 16)
      return v;
   return nir_u2u16(b, v);
}

/* idx & 3, the byte of the selected word. When the index is itself a
 * constant mask of a wider value (the usual idx = i & 15), mask that value
 * directly: (x & c) & 3 == x & (c & 3). */
static nir_def *
byte_lane_of_index(nir_builder *b, nir_scalar idx)
{
   nir_scalar s = nir_scalar_chase_movs(idx);
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iand) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar c = nir_scalar_chase_alu_src(s, i);
         if (nir_scalar_is_const(c)) {
            nir_scalar x = nir_scalar_chase_alu_src(s, 1 - i);
            return nir_iand_imm(b, nir_channel(b, x.def, x.comp),
                                nir_scalar_as_uint(c) & 3);
         }
      }
   }

   return nir_iand_imm(b, nir_channel(b, idx.def, idx.comp), 3);
}

static bool
try_pack_lut_select(nir_alu_instr *root, nir_loop *loop, nir_block *pre,
                    struct set *visited)
{
   if (_mesa_set_search(visited, &root->def) ||
       list_is_empty(&root->def.uses))
      return false;
   if (root->op != nir_op_bcsel || root->def.num_components != 1)
      return false;
   const unsigned bits = root->def.bit_size;
   if (bits != 8 && bits != 16 && bits != 32)
      return false;

   struct cmp_info root_ci;
   if (!parse_index_cmp(nir_get_scalar(root->src[0].src.ssa,
                                       root->src[0].swizzle[0]),
                        &root_ci))
      return false;

   nir_scalar idx = root_ci.idx;
   unsigned max_idx = 0;
   if (!proven_index_max(idx, &max_idx))
      return false;

   const unsigned n = max_idx + 1u;
   assert(n >= MIN_LUT_BYTES && n <= MAX_LUT_BYTES);

   /* Skip non-root internal nodes of a larger matching select tree. */
   nir_block *block = root->instr.block;
   nir_foreach_use(use, &root->def) {
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->block == block && parent->type == nir_instr_type_alu) {
         nir_alu_instr *palu = nir_instr_as_alu(parent);
         struct cmp_info pci;
         if (palu->op == nir_op_bcsel && palu->def.num_components == 1 &&
             parse_index_cmp(nir_get_scalar(palu->src[0].src.ssa,
                                            palu->src[0].swizzle[0]),
                             &pci) &&
             nir_scalar_equal(pci.idx, idx))
            return false;
      }
   }

   struct set *internal_sels = _mesa_pointer_set_create(NULL);
   nir_scalar root_s = nir_get_scalar(&root->def, 0);
   bool ok = collect_tree(root_s, block, loop, pre, idx, internal_sels, 0);
   const unsigned sel_count = internal_sels->entries;

   /* Require every internal select (other than `root`) to be consumed only by
    * the tree so replacing `root` actually eliminates the tree. */
   if (ok && sel_count >= MIN_TREE_SELS) {
      set_foreach(internal_sels, entry) {
         nir_def *def = (nir_def *)entry->key;
         if (def == &root->def)
            continue;
         nir_foreach_use_including_if(use, def) {
            if (nir_src_is_if(use)) {
               ok = false;
               break;
            }
            nir_def *udef = nir_instr_def(nir_src_use_instr(use));
            if (!udef || !_mesa_set_search(internal_sels, udef)) {
               ok = false;
               break;
            }
         }
         if (!ok)
            break;
      }
   } else {
      ok = false;
   }

   if (!ok) {
      _mesa_set_destroy(internal_sels, NULL);
      return false;
   }

   nir_scalar leaves[MAX_LUT_BYTES];
   unsigned distinct = 0;
   for (unsigned k = 0; k < n; ++k) {
      if (!eval_tree(root_s, block, loop, pre, idx, k, &leaves[k], 0)) {
         _mesa_set_destroy(internal_sels, NULL);
         return false;
      }
      bool seen = false;
      for (unsigned j = 0; j < k; ++j) {
         if (nir_scalar_equal(leaves[j], leaves[k])) {
            seen = true;
            break;
         }
      }
      if (!seen)
         distinct++;
   }

   /* Require at least 4 distinct leaves so trivial trees stay untouched. */
   if (distinct < 4) {
      _mesa_set_destroy(internal_sels, NULL);
      return false;
   }

   /* Mark all internal selects so dead subtrees are not revisited in this pass. */
   set_foreach(internal_sels, entry)
      _mesa_set_add(visited, entry->key);
   _mesa_set_destroy(internal_sels, NULL);

   /* Pack 4 bytes per 32-bit word via 16-bit half pairs in the loop preheader
    * so the backend coalesces pack_32_2x16_split without << 16 / << 24 shifts. */
   const unsigned num_words = DIV_ROUND_UP(n, 4u);
   nir_def *words[4] = {NULL};
   nir_builder b_pre = nir_builder_at(nir_after_block_before_jump(pre));
   for (unsigned g = 0; g < num_words; ++g) {
      const unsigned base = 4u * g;
      nir_def *lo = emit_byte_as_u16(&b_pre, leaves[base]);
      if (base + 1u < n) {
         nir_def *b1 = emit_byte_as_u16(&b_pre, leaves[base + 1u]);
         lo = nir_ior(&b_pre, lo, nir_ishl_imm(&b_pre, b1, 8));
      }
      nir_def *hi = base + 2u < n
                       ? emit_byte_as_u16(&b_pre, leaves[base + 2u])
                       : nir_imm_intN_t(&b_pre, 0, 16);
      if (base + 3u < n) {
         nir_def *b3 = emit_byte_as_u16(&b_pre, leaves[base + 3u]);
         hi = nir_ior(&b_pre, hi, nir_ishl_imm(&b_pre, b3, 8));
      }
      words[g] = nir_pack_32_2x16_split(&b_pre, lo, hi);
   }

   /* Select the word and extract byte (idx & 3) inside the loop. */
   nir_builder b = nir_builder_at(nir_before_instr(&root->instr));
   nir_def *idx_val = nir_channel(&b, idx.def, idx.comp);
   nir_def *word;
   if (num_words == 2) {
      word = nir_bcsel(&b, nir_ult_imm(&b, idx_val, 4), words[0], words[1]);
   } else if (num_words == 3) {
      nir_def *w01 =
         nir_bcsel(&b, nir_ult_imm(&b, idx_val, 4), words[0], words[1]);
      word = nir_bcsel(&b, nir_ult_imm(&b, idx_val, 8), w01, words[2]);
   } else {
      assert(num_words == 4);
      nir_def *w01 =
         nir_bcsel(&b, nir_ult_imm(&b, idx_val, 4), words[0], words[1]);
      nir_def *w23 =
         nir_bcsel(&b, nir_ult_imm(&b, idx_val, 12), words[2], words[3]);
      word = nir_bcsel(&b, nir_ult_imm(&b, idx_val, 8), w01, w23);
   }

   /* The byte offset is 0, 8, 16 or 24: always inside the word, so the
    * extract needs no modulo-32 count and is one masked bitfield extract,
    * where a shift and a separate 0xff mask would be two operations plus the
    * count wrap a variable shift needs. */
   nir_def *offset = nir_ishl_imm(&b, byte_lane_of_index(&b, idx), 3);
   nir_def *val32 = nir_ubitfield_extract(&b, word, offset, nir_imm_int(&b, 8));

   /* When a narrowed select tree feeds u2u32 directly, bypass the narrow->wide
    * round-trip since `val32` is already zero-extended in [0, 255]. */
   if (bits < 32) {
      nir_foreach_use_safe(use, &root->def) {
         if (nir_src_is_if(use))
            continue;
         nir_instr *ui = nir_src_use_instr(use);
         if (ui->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *ualu = nir_instr_as_alu(ui);
         if (ualu->op == nir_op_u2u32 && ualu->def.num_components == 1 &&
             ualu->src[0].swizzle[0] == 0) {
            nir_def_replace(&ualu->def, val32);
         }
      }
   }

   if (!list_is_empty(&root->def.uses)) {
      nir_def *res = bits == 32 ? val32
                                : (bits == 16 ? nir_u2u16(&b, val32)
                                              : nir_u2u8(&b, val32));
      nir_def_replace(&root->def, res);
   }
   return true;
}

static bool
process_cf_list(struct exec_list *list, nir_loop *innermost_loop)
{
   bool progress = false;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         if (!innermost_loop)
            break;
         nir_block *block = nir_cf_node_as_block(node);
         nir_block *pre = nir_loop_predecessor_block(innermost_loop);
         struct set *visited = _mesa_pointer_set_create(NULL);
         nir_foreach_instr_reverse_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            progress |= try_pack_lut_select(nir_instr_as_alu(instr),
                                            innermost_loop, pre, visited);
         }
         _mesa_set_destroy(visited, NULL);
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= process_cf_list(&nif->then_list, innermost_loop);
         progress |= process_cf_list(&nif->else_list, innermost_loop);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= process_cf_list(&loop->body,
                                     nir_loop_has_back_edge(loop) ? loop : NULL);
         break;
      }
      default:
         break;
      }
   }

   return progress;
}

bool
agx_nir_opt_pack_byte_lut_select(nir_shader *shader)
{
   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_dominance);
      bool progress = process_cf_list(&impl->body, NULL);
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }

   return any;
}
