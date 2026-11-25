/* SPDX-License-Identifier: MIT */
#include "agx_nir_match_gemm.h"
#include "agx_nir.h"

#include "compiler/nir/nir_builder.h"

#define TILE_BYTES (4 * AGX_GEMM_TILE_WORDS)
#define TILE_MATH (AGX_GEMM_WIDTH * AGX_GEMM_ROWS_PER_LANE)

static nir_loop *
innermost_loop(nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(node);
   }
   return NULL;
}

/* Model only known, bounded local index expressions. Reject unsupported
 * system values, uniform/buffer loads, phis and unproven arithmetic.
 */
static bool
local_uint_rec(nir_scalar scalar, unsigned x, unsigned y, unsigned depth,
               uint32_t *value, unsigned *budget)
{
   if (!depth || !*budget || !scalar.def || scalar.def->bit_size > 32)
      return false;
   --*budget;

   if (nir_scalar_is_const(scalar)) {
      *value = nir_scalar_as_uint(scalar);
      return true;
   }

   if (nir_scalar_is_intrinsic(scalar)) {
      if (nir_scalar_as_intrinsic(scalar)->intrinsic !=
            nir_intrinsic_load_local_invocation_id || scalar.comp > 2)
         return false;
      const unsigned xyz[] = { x, y, 0 };
      *value = xyz[scalar.comp];
      return true;
   }

   if (!nir_scalar_is_alu(scalar))
      return false;

   const nir_op op = nir_scalar_alu_op(scalar);
   switch (op) {
   case nir_op_mov:
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_iadd:
   case nir_op_imul:
   case nir_op_ishl:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_imadshl_agx:
      break;
   default:
      return false;
   }

   uint32_t lhs, rhs = 0;
   if (!local_uint_rec(nir_scalar_chase_alu_src(scalar, 0), x, y, depth - 1, &lhs,
                       budget))
      return false;
   if (op != nir_op_mov && op != nir_op_u2u16 && op != nir_op_u2u32 &&
       !local_uint_rec(nir_scalar_chase_alu_src(scalar, 1), x, y, depth - 1, &rhs,
                       budget))
      return false;

   uint64_t wide;
   switch (op) {
   case nir_op_mov:
   case nir_op_u2u16:
   case nir_op_u2u32:
      wide = lhs;
      break;
   case nir_op_iadd:
      wide = (uint64_t)lhs + rhs;
      break;
   case nir_op_imul:
      wide = (uint64_t)lhs * rhs;
      break;
   case nir_op_ishl:
      if (rhs >= scalar.def->bit_size)
         return false;
      wide = (uint64_t)lhs << rhs;
      break;
   case nir_op_iand:
      wide = lhs & rhs;
      break;
   case nir_op_ior:
      wide = lhs | rhs;
      break;
   case nir_op_imadshl_agx: {
      uint32_t add, shift;
      if (!local_uint_rec(nir_scalar_chase_alu_src(scalar, 2), x, y, depth - 1, &add,
                          budget) ||
          !local_uint_rec(nir_scalar_chase_alu_src(scalar, 3), x, y, depth - 1, &shift,
                          budget) ||
          shift >= scalar.def->bit_size ||
          (uint64_t)lhs * rhs > UINT32_MAX ||
          ((uint64_t)add << shift) > UINT32_MAX)
         return false;
      wide = (uint64_t)lhs * rhs + ((uint64_t)add << shift);
      break;
   }
   default:
      return false;
   }

   if (wide > BITFIELD64_MASK(scalar.def->bit_size))
      return false;
   *value = wide;
   return true;
}

/* Index expressions are a few dozen nodes; the budget bounds the walk of a
 * shared subexpression reached along many paths. */
static bool
local_uint(nir_scalar scalar, unsigned x, unsigned y, unsigned depth,
           uint32_t *value)
{
   unsigned budget = 512;
   return local_uint_rec(scalar, x, y, depth, value, &budget);
}

static bool
shared_word(nir_intrinsic_instr *intr, unsigned x, unsigned y, unsigned *address)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_load_shared;
   if (!is_load && intr->intrinsic != nir_intrinsic_store_shared)
      return false;

   nir_def *data = is_load ? &intr->def : intr->src[0].ssa;
   nir_def *offset = intr->src[is_load ? 0 : 1].ssa;
   const int base = nir_intrinsic_base(intr);
   if (data->bit_size != 32 || data->num_components != 1 ||
       offset->bit_size != 16 || offset->num_components != 1 ||
       nir_intrinsic_access(intr) != 0 || nir_intrinsic_align_mul(intr) < 4 ||
       nir_intrinsic_align_offset(intr) % 4 || base < 0 || base % 4 ||
       (!is_load && nir_intrinsic_write_mask(intr) != 1))
      return false;

   uint32_t index;
   if (!local_uint(nir_scalar_resolved(offset, 0), x, y, 12, &index))
      return false;
   *address = (unsigned)base + 4 * index;
   return true;
}

struct operands {
   nir_intrinsic_instr *a;
   nir_intrinsic_instr *b;
   unsigned a0;
   unsigned b0;
};

static bool
fma_operands(nir_alu_instr *math, nir_instr *begin, nir_instr *end,
             struct operands *out)
{
   nir_scalar def = nir_get_scalar(&math->def, 0);
   nir_scalar lhs = nir_scalar_chase_alu_src(def, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(def, 1);
   if (!nir_scalar_is_intrinsic(lhs) || !nir_scalar_is_intrinsic(rhs))
      return false;
   nir_intrinsic_instr *left = nir_scalar_as_intrinsic(lhs);
   nir_intrinsic_instr *right = nir_scalar_as_intrinsic(rhs);
   if (lhs.comp != 0 || rhs.comp != 0 ||
       left->intrinsic != nir_intrinsic_load_shared ||
       right->intrinsic != nir_intrinsic_load_shared)
      return false;

   /* Match in the convergent fence region and establish the axis roles from
    * actual addresses. Either multiplicand order is legal.
    */
   unsigned present = 0;
   for (nir_instr *instr = nir_instr_next(begin); instr != end; instr = nir_instr_next(instr)) {
      present += instr == &left->instr || instr == &right->instr;
      if (instr == &math->instr)
         break;
   }
   if (present != 2)
      return false;

   unsigned l0, r0, lx, rx;
   if (!shared_word(left, 0, 0, &l0) || !shared_word(right, 0, 0, &r0) ||
       !shared_word(left, 1, 0, &lx) || !shared_word(right, 1, 0, &rx))
      return false;
   if (lx == l0 && rx == r0 + 4) {
      *out = (struct operands){ left, right, l0, r0 };
      return true;
   }
   if (rx == r0 && lx == l0 + 4) {
      *out = (struct operands){ right, left, r0, l0 };
      return true;
   }
   return false;
}

static bool
collect_phase(nir_shader *shader, struct agx_gemm_match *m)
{
   const unsigned W = AGX_GEMM_WIDTH;
   nir_alu_instr *math[TILE_MATH];
   struct operands args[TILE_MATH];
   unsigned count = 0;
   m->a_bytes = UINT_MAX;
   m->b_bytes = UINT_MAX;

   for (nir_instr *ins = nir_instr_next(m->ready); ins != m->consumed;
        ins = nir_instr_next(ins)) {
      if (ins->type == nir_instr_type_load_const)
         continue;
      if (ins->type == nir_instr_type_intrinsic) {
         unsigned address;
         if (nir_instr_as_intrinsic(ins)->intrinsic != nir_intrinsic_load_shared ||
             !shared_word(nir_instr_as_intrinsic(ins), 0, 0, &address))
            return false;
         continue;
      }
      if (ins->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(ins);
      if (alu->op != nir_op_ffma)
         continue;
      if (count == TILE_MATH || alu->def.bit_size != 32 ||
          alu->def.num_components != 1 || alu->fp_math_ctrl != nir_fp_fast_math ||
          !fma_operands(alu, m->ready, m->consumed, &args[count]))
         return false;
      m->a_bytes = MIN2(m->a_bytes, args[count].a0);
      m->b_bytes = MIN2(m->b_bytes, args[count].b0);
      math[count++] = alu;
   }

   if (count != TILE_MATH || ((m->a_bytes | m->b_bytes) & 7) ||
       m->a_bytes > 4 * (UINT16_MAX + 1u) - TILE_BYTES ||
       m->b_bytes > 4 * (UINT16_MAX + 1u) - TILE_BYTES ||
       m->a_bytes > shader->info.shared_size - TILE_BYTES ||
       m->b_bytes > shader->info.shared_size - TILE_BYTES ||
       (m->a_bytes < m->b_bytes ? m->b_bytes - m->a_bytes :
                                m->a_bytes - m->b_bytes) < TILE_BYTES)
      return false;

   for (unsigned i = 0; i < count; ++i) {
      const struct operands *f = &args[i];
      const unsigned b = (f->b0 - m->b_bytes) / 4;
      const unsigned a = (f->a0 - m->a_bytes) / 4;
      const unsigned k = b / W;
      const unsigned row = a / W;
      if (b % W || k >= W || a % W != k || row >= AGX_GEMM_ROWS_PER_LANE ||
          m->fma[k][row])
         return false;
      m->fma[k][row] = math[i];

      for (unsigned y = 0; y < AGX_GEMM_SUBGROUPS; ++y) {
         for (unsigned x = 0; x < W; ++x) {
            unsigned actual_a, actual_b;
            if (!shared_word(f->a, x, y, &actual_a) ||
                actual_a != m->a_bytes + 4 * ((8 * y + row) * W + k) ||
                !shared_word(f->b, x, y, &actual_b) ||
                actual_b != m->b_bytes + 4 * (k * W + x))
               return false;
         }
      }
   }
   return true;
}

static bool
phi_has_zero_and_end(nir_phi_instr *phi, nir_def *result)
{
   bool has_zero = false;
   bool has_result = false;
   unsigned count = 0;
   nir_foreach_phi_src(src, phi) {
      if (src->src.ssa == result)
         has_result = true;
      else if (nir_scalar_is_zero(nir_get_scalar(src->src.ssa, 0)))
         has_zero = true;
      ++count;
   }
   return has_zero && has_result && count == 2;
}

/* The value after the loop must be the merge of the uniform if that holds
 * only the loop's branch, as loop rotation leaves it: the chain's last value
 * from the branch with the loop, +0.0 from the other (the loop did not run).
 * The lowered chain's value is in fragment layout, spread over the lanes; a
 * merge of any other if could take +0.0 in some lanes only. The if must be
 * at the top level of the function: the epilogue's workgroup barrier at its
 * merge then runs in every invocation, also when the loop did not run (an
 * enclosing if could be uniform per subgroup but not per workgroup). */
static bool
guarded_exit(nir_loop *loop, nir_phi_instr *phi, nir_def *last)
{
   nir_cf_node *parent = loop->cf_node.parent;
   if (!parent || parent->type != nir_cf_node_if)
      return false;
   nir_if *nif = nir_cf_node_as_if(parent);
   if (parent->parent != &nir_cf_node_get_function(parent)->cf_node)
      return false;
   nir_cf_node *next = nir_cf_node_next(parent);
   if (nir_src_is_divergent(&nif->condition) || !next ||
       next->type != nir_cf_node_block || phi->instr.block != nir_cf_node_as_block(next))
      return false;

   bool in_then = false;
   foreach_list_typed(nir_cf_node, node, node, &nif->then_list)
      in_then |= node == &loop->cf_node;
   nir_block *loop_pred = in_then ? nir_if_last_then_block(nif)
                                  : nir_if_last_else_block(nif);
   unsigned count = 0;
   nir_foreach_phi_src(src, phi) {
      if (src->pred == loop_pred ? src->src.ssa != last
                                 : !nir_scalar_is_zero(nir_get_scalar(src->src.ssa, 0)))
         return false;
      ++count;
   }
   return count == 2;
}

static bool
only_user(nir_def *def, nir_instr *expected)
{
   unsigned count = 0;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src) || nir_src_use_instr(src) != expected)
         return false;
      ++count;
   }
   return count == 1;
}

static bool
recurrences(struct agx_gemm_match *m)
{
   nir_block *header = nir_loop_first_block(m->loop);
   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_alu_instr *seed = m->fma[0][row];
      nir_alu_instr *last = m->fma[AGX_GEMM_WIDTH - 1][row];
      nir_scalar initial =
         nir_scalar_chase_alu_src(nir_get_scalar(&seed->def, 0), 2);
      if (initial.comp != 0 || !nir_scalar_is_phi(initial))
         return false;

      nir_phi_instr *start = nir_scalar_as_phi(initial);
      if (start->instr.block != header || !phi_has_zero_and_end(start, &last->def) ||
          !only_user(&start->def, &seed->instr))
         return false;
      m->initial[row] = start;

      for (unsigned k = 1; k < AGX_GEMM_WIDTH; ++k) {
         nir_alu_instr *alu = m->fma[k][row];
         nir_alu_instr *prev = m->fma[k - 1][row];
         nir_scalar old = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 2);
         if (!nir_scalar_equal(old, nir_get_scalar(&prev->def, 0)) ||
             !only_user(&prev->def, &alu->instr))
            return false;
      }

      unsigned count = 0;
      nir_foreach_use_including_if(src, &last->def) {
         if (nir_src_is_if(src) || nir_src_use_instr(src)->type != nir_instr_type_phi)
            return false;
         nir_phi_instr *phi = nir_instr_as_phi(nir_src_use_instr(src));
         if (phi != start) {
            if (phi->instr.block->divergent || innermost_loop(phi->instr.block) ||
                !guarded_exit(m->loop, phi, &last->def))
               return false;
            m->exit[row] = phi;
         }
         ++count;
      }
      if (count != 2 || !m->exit[row] ||
          m->exit[row]->instr.block != m->exit[0]->instr.block)
         return false;
   }
   return true;
}

enum visit_state {
   UNSEEN,
   ACTIVE,
   DONE,
};

struct writes {
   enum visit_state state;
   uint16_t slots;
};

static bool
producer_slot(const struct agx_gemm_match *m, nir_intrinsic_instr *store,
              uint16_t *slot)
{
   unsigned first;
   if (!shared_word(store, 0, 0, &first))
      return false;

   const bool a = first >= m->a_bytes && first - m->a_bytes < TILE_BYTES;
   const unsigned base = a ? m->a_bytes : m->b_bytes;
   if (first < base || (first - base) % (4 * AGX_GEMM_WIDTH))
      return false;

   const unsigned row = (first - base) / (4 * AGX_GEMM_WIDTH);
   if (row >= AGX_GEMM_ROWS_PER_LANE)
      return false;

   for (unsigned y = 0; y < AGX_GEMM_SUBGROUPS; ++y) {
      for (unsigned x = 0; x < AGX_GEMM_WIDTH; ++x) {
         unsigned address;
         const unsigned i = (8 * y + row) * AGX_GEMM_WIDTH + x;
         if (!shared_word(store, x, y, &address) || address != base + 4 * i)
            return false;
      }
   }
   *slot = 1u << (row + (a ? 0 : AGX_GEMM_ROWS_PER_LANE));
   return true;
}

/*
 * Intersect writes at joins: a slot must be initialized on all incoming
 * paths, including predicated zero stores in the edge producer.
 */
static bool
producer_writes(const struct agx_gemm_match *m, nir_block *block,
                struct writes *blocks, unsigned depth)
{
   struct writes *record = &blocks[block->index];
   if (record->state == DONE)
      return true;
   if (!depth || record->state == ACTIVE || innermost_loop(block) != m->loop)
      return false;
   record->state = ACTIVE;

   uint16_t slots = 0;
   if (block != nir_loop_first_block(m->loop)) {
      if (!nir_block_num_preds(block))
         return false;
      slots = UINT16_MAX;
      nir_foreach_pred(pred, block) {
         if (!producer_writes(m, pred, blocks, depth - 1))
            return false;
         slots &= blocks[pred->index].slots;
      }
   }

   nir_foreach_instr(ins, block) {
      if (ins == m->ready)
         break;
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ins);
      if (intr->intrinsic == nir_intrinsic_barrier)
         return false;
      if (nir_is_shared_access(intr)) {
         uint16_t slot;
         if (intr->intrinsic != nir_intrinsic_store_shared ||
             !producer_slot(m, intr, &slot))
            return false;
         slots |= slot;
      }
   }
   *record = (struct writes){ DONE, slots };
   return true;
}

static bool
validate_visibility(struct agx_gemm_match *m)
{
   nir_function_impl *impl = m->ready->block->impl;
   struct writes *blocks = calloc(impl->num_blocks, sizeof(*blocks));
   if (!blocks)
      return false;
   bool match = producer_writes(m, m->ready->block, blocks, impl->num_blocks) &&
                blocks[m->ready->block->index].slots == UINT16_MAX;

   /* Reject surviving observers of the scratch tile and nonuniform exits.
    * All shared uses must belong to the traversed producers or the phase.
    */
   unsigned breaks = 0;
   nir_foreach_block(block, impl) {
      bool in_phase = false;
      bool passed_phase = false;
      nir_foreach_instr(ins, block) {
         if (ins == m->ready) {
            in_phase = true;
         } else if (ins == m->consumed) {
            in_phase = false;
            passed_phase = true;
         } else if (ins->type == nir_instr_type_jump) {
            nir_jump_type jump = nir_instr_as_jump(ins)->type;
            if (innermost_loop(block) == m->loop) {
               match &= !block->divergent && jump == nir_jump_break &&
                        nir_block_dominates(m->ready->block, block);
               ++breaks;
            } else if (jump == nir_jump_return || jump == nir_jump_halt) {
               match &= nir_block_dominates(m->exit[0]->instr.block, block);
            }
         } else if (ins->type == nir_instr_type_call ||
                    ins->type == nir_instr_type_cmat_call ||
                    (ins->type == nir_instr_type_deref &&
                     (nir_instr_as_deref(ins)->modes & nir_var_mem_shared))) {
            match = false;
         } else if (ins->type == nir_instr_type_intrinsic &&
                    nir_is_shared_access(nir_instr_as_intrinsic(ins))) {
            match &= in_phase || (!passed_phase && blocks[block->index].state == DONE);
         }
      }
   }
   free(blocks);
   return match && breaks == 1;
}

bool
agx_nir_match_gemm(nir_shader *shader, struct agx_gemm_match *result)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE || !shader->info.io_lowered ||
       shader->info.workgroup_size_variable || !shader->info.shared_element_index_agx ||
       shader->info.workgroup_size[0] != AGX_GEMM_WIDTH ||
       shader->info.workgroup_size[1] != AGX_GEMM_SUBGROUPS ||
       shader->info.workgroup_size[2] != 1 ||
       shader->info.min_subgroup_size != AGX_GEMM_WIDTH ||
       shader->info.max_subgroup_size != AGX_GEMM_WIDTH ||
       shader->info.shared_size < 2 * TILE_BYTES ||
       shader->info.float_controls_execution_mode)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_divergence |
                              nir_metadata_dominance);
   nir_foreach_block(block, impl) {
      if (block->divergent)
         continue;
      nir_loop *loop = innermost_loop(block);
      if (!loop || nir_loop_has_continue_construct(loop) ||
          innermost_loop(nir_loop_predecessor_block(loop)))
         continue;

      nir_instr *previous = NULL;
      nir_foreach_instr(ins, block) {
         if (!agx_nir_is_shared_phase_barrier(ins))
            continue;
         if (previous &&
             nir_intrinsic_memory_scope(nir_instr_as_intrinsic(previous)) ==
                nir_intrinsic_memory_scope(nir_instr_as_intrinsic(ins))) {
            struct agx_gemm_match match = { .loop = loop, .ready = previous, .consumed = ins };
            if (collect_phase(shader, &match) && recurrences(&match) &&
                validate_visibility(&match)) {
               *result = match;
               return true;
            }
         }
         previous = ins;
      }
   }
   return false;
}

static nir_def *
shared_index(nir_builder *b, nir_def *row, nir_def *col, unsigned byte_base)
{
   nir_def *word = nir_iadd(b, nir_ishl_imm(b, row, 5), col);
   return nir_u2u16(b, nir_iadd_imm(b, word, byte_base / 4));
}

static void
replace_phi_source(nir_phi_instr *phi, nir_def *old, nir_def *value)
{
   nir_foreach_phi_src(src, phi) {
      if (src->src.ssa == old) {
         nir_src_rewrite(&src->src, value);
         return;
      }
   }
   assert(!"validated phi is missing its source");
}

struct fragment_position {
   nir_def *row;
   nir_def *subgroup_row;
   nir_def *col;
};

static struct fragment_position
fragment_position(nir_builder *b)
{
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *offset = nir_ishl_imm(b, nir_load_subgroup_id(b), 3);
   nir_def *row =
      nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3),
                 nir_iand_imm(b, nir_ushr_imm(b, lane, 2), 4));
   nir_def *col =
      nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 1),
                 nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 4));
   return (struct fragment_position){ nir_iadd(b, offset, row), row, col };
}

bool
agx_nir_lower_gemm(nir_shader *shader, const struct agx_gemm_match *m)
{
   nir_function_impl *impl = m->ready->block->impl;
   nir_builder b = nir_builder_create(impl);

   /* The fragment position and the shared indices depend only on the lane
    * and the subgroup: compute them once, before the K loop. */
   b.cursor = nir_after_block_before_jump(nir_loop_predecessor_block(m->loop));
   const struct fragment_position input = fragment_position(&b);
   /* Use the AGX3 byte immediate for the small contiguous K and N
    * offsets. The per-lane register index still counts scalar words.
    */
   nir_def *left_index = shared_index(&b, input.row, input.col, m->a_bytes);
   nir_def *right_indices[AGX_GEMM_WIDTH / 8];
   for (unsigned k = 0; k < AGX_GEMM_WIDTH; k += 8) {
      nir_def *k_row = nir_iadd_imm(&b, input.subgroup_row, k);
      right_indices[k / 8] = shared_index(&b, k_row, input.col, m->b_bytes);
   }

   b.cursor = nir_after_instr(m->ready);
   nir_def *acc[4];
   for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile)
      acc[tile] = nir_vec2(&b, &m->initial[2 * tile]->def,
                              &m->initial[2 * tile + 1]->def);

   for (unsigned k = 0; k < AGX_GEMM_WIDTH; k += 8) {
      nir_def *left = nir_load_shared(&b, 2, 32, left_index,
                                      .base = 4 * k, .align_mul = 8);
      nir_def *right_index = right_indices[k / 8];
      for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile) {
         nir_def *right = nir_load_shared(&b, 2, 32, right_index,
                                           .base = 32 * tile, .align_mul = 8);
         acc[tile] = nir_cmat_muladd_agx(&b, left, right, acc[tile],
            .src_type = nir_type_float32, .src_type2 = nir_type_float32);
      }
   }

   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_def *old = &m->fma[AGX_GEMM_WIDTH - 1][row]->def;
      nir_def *part = nir_channel(&b, acc[row / 2], row % 2);
      replace_phi_source(m->initial[row], old, part);
      replace_phi_source(m->exit[row], old, part);
   }

   /* The input barrier at the end of the last K iteration completed all
    * uses of A. Store the subgroup-owned C fragments into its allocation,
    * then resume the unchanged per-invocation tail and bias epilogue.
    */
   b.cursor = nir_after_phis(m->exit[0]->instr.block);
   const struct fragment_position output = fragment_position(&b);
   nir_def *staged = shared_index(&b, output.row, output.col, m->a_bytes);
   for (unsigned tile = 0; tile < ARRAY_SIZE(acc); ++tile) {
      nir_def *part = nir_vec2(&b, &m->exit[2 * tile]->def,
                                  &m->exit[2 * tile + 1]->def);
      nir_store_shared(&b, part, staged, .base = 32 * tile,
                        .write_mask = 3, .align_mul = 8);
   }
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP, .memory_modes = nir_var_mem_shared,
               .memory_semantics = NIR_MEMORY_ACQ_REL);
   nir_def *local = nir_load_local_invocation_id(&b);
   nir_def *column = nir_channel(&b, local, 0);
   nir_def *row_base = nir_ishl_imm(&b, nir_channel(&b, local, 1), 3);
   for (unsigned row = 0; row < AGX_GEMM_ROWS_PER_LANE; ++row) {
      nir_def *offset = shared_index(&b, nir_iadd_imm(&b, row_base, row),
                                     column, m->a_bytes);
      nir_def *result = nir_load_shared(&b, 1, 32, offset, .align_mul = 4);
      nir_def_rewrite_uses_after(&m->exit[row]->def, result);
   }

   nir_progress(true, impl, nir_metadata_control_flow);
   nir_opt_dce(shader);
   nir_divergence_analysis(shader);
   return true;
}
