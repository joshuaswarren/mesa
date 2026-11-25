/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_shader.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

struct shader_counts {
   unsigned uadd_sat;
   unsigned iadd_nuw;
   unsigned usub_sat;
   unsigned bounds_agx;
   unsigned load_global;
   unsigned load_global_max_comps;
   unsigned load_global_bounded;
};

static struct shader_counts
count_ops(nir_shader *nir)
{
   struct shader_counts c = {0};
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_uadd_sat)
                  c.uadd_sat++;
               else if (alu->op == nir_op_iadd && alu->no_unsigned_wrap)
                  c.iadd_nuw++;
               else if (alu->op == nir_op_usub_sat)
                  c.usub_sat++;
               else if (alu->op == nir_op_bounds_agx)
                  c.bounds_agx++;
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_global) {
                  c.load_global++;
                  if (intr->num_components > c.load_global_max_comps)
                     c.load_global_max_comps = intr->num_components;
               } else if (intr->intrinsic == nir_intrinsic_load_global_bounded) {
                  c.load_global_bounded++;
               }
            }
         }
      }
   }
   return c;
}

static nir_def *
make_input_u32(nir_builder *b, unsigned offset)
{
   return nir_load_global_constant(b, 1, 32,
                                   nir_imm_int64(b, 0x100000000ull + offset),
                                   .align_mul = 4);
}

static nir_def *
make_input_u64(nir_builder *b, unsigned offset)
{
   return nir_load_global_constant(b, 1, 64,
                                   nir_imm_int64(b, 0x100000000ull + offset),
                                   .align_mul = 8);
}

static void
store_at(nir_builder *b, uint64_t addr, nir_def *val)
{
   nir_store_global(b, val, nir_imm_int64(b, addr),
                    .align_mul = 4,
                    .write_mask = (1u << val->num_components) - 1);
}

static nir_intrinsic_instr *
find_store_at(nir_shader *nir, uint64_t addr)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_global)
               continue;
            nir_scalar a = nir_scalar_resolved(intr->src[1].ssa, 0);
            if (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == addr)
               return intr;
         }
      }
   }
   return NULL;
}

static nir_def *
find_input_load(nir_shader *nir, unsigned offset)
{
   const uint64_t target = 0x100000000ull + offset;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_global_constant)
               continue;
            nir_scalar a = nir_scalar_resolved(intr->src[0].ssa, 0);
            if (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == target)
               return &intr->def;
         }
      }
   }
   return NULL;
}

static bool
is_binop_with_const(nir_scalar s, nir_op op, bool require_nuw,
                    nir_scalar *other_out, uint32_t expected_const)
{
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != op)
      return false;
   if (require_nuw && !nir_instr_as_alu(nir_def_instr(s.def))->no_unsigned_wrap)
      return false;
   nir_scalar a = nir_scalar_chase_alu_src(s, 0);
   nir_scalar b = nir_scalar_chase_alu_src(s, 1);
   if (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == expected_const) {
      *other_out = b;
      return true;
   }
   if (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == expected_const) {
      *other_out = a;
      return true;
   }
   return false;
}

/* Aligned shifted/multiplied/masked/selected terms with k below the proven
 * alignment lower to iadd with no_unsigned_wrap. */
static void
test_aligned_lowers_to_iadd_nuw(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "aligned_lowers");
   nir_def *idx = make_input_u32(&b, 0);
   nir_def *cond = nir_ine_imm(&b, make_input_u32(&b, 4), 0);

   nir_def *shifted = nir_ishl_imm(&b, idx, 4);
   nir_def *r0 = nir_uadd_sat(&b, shifted, nir_imm_int(&b, 12));
   /* Constant on LHS commutes identically. */
   nir_def *r1 = nir_uadd_sat(&b, nir_imm_int(&b, 4), nir_iadd_imm(&b, shifted, 8));
   /* Multiplied by 16 is 16-byte aligned; +15 cannot carry out of bit 3. */
   nir_def *r2 = nir_uadd_sat(&b, nir_imul_imm(&b, idx, 16), nir_imm_int(&b, 15));
   /* Masked low bits and aligned bcsel arms both prove alignment. */
   nir_def *masked = nir_iand_imm(&b, idx, ~0x7u);
   nir_def *sel = nir_bcsel(&b, cond, masked, nir_ishl_imm(&b, idx, 3));
   nir_def *r3 = nir_uadd_sat(&b, sel, nir_imm_int(&b, 4));

   store_at(&b, 0x200000000ull, nir_vec4(&b, r0, r1, r2, r3));

   bool progress = false;
   NIR_PASS(progress, b.shader, hk_nir_lower_aligned_uadd_sat);
   CHECK(progress);
   nir_validate_shader(b.shader, "aligned_lowers");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.uadd_sat == 0);
   CHECK(c.iadd_nuw == 4);
   ralloc_free(b.shader);
}

/* Boundary and unaligned cases that could wrap must remain uadd_sat. */
static void
test_unaligned_and_boundary_stay_uadd_sat(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "unaligned_stays");
   nir_def *idx = make_input_u32(&b, 0);
   nir_def *cond = nir_ine_imm(&b, make_input_u32(&b, 4), 0);
   nir_def *shifted16 = nir_ishl_imm(&b, idx, 4);

   /* 16-byte aligned + 16 can overflow at 0xfffffff0 + 16. */
   nir_def *r0 = nir_uadd_sat(&b, shifted16, nir_imm_int(&b, 16));
   /* (idx << 4) + 12 is only 4-byte aligned, so +4 can overflow at 0xfffffffc + 4. */
   nir_def *r1 = nir_uadd_sat(&b, nir_iadd_imm(&b, shifted16, 12), nir_imm_int(&b, 4));
   /* One unaligned bcsel arm disqualifies the select. */
   nir_def *sel = nir_bcsel(&b, cond, shifted16, nir_ishl_imm(&b, idx, 2));
   nir_def *r2 = nir_uadd_sat(&b, sel, nir_imm_int(&b, 12));
   /* Top bit set in k (mask = UINT32_MAX) must not lower unknown (idx << 31). */
   nir_def *r3 = nir_uadd_sat(&b, nir_ishl_imm(&b, idx, 31),
                              nir_imm_int(&b, 0x80000000u));

   store_at(&b, 0x200000000ull, nir_vec4(&b, r0, r1, r2, r3));

   bool progress = false;
   NIR_PASS(progress, b.shader, hk_nir_lower_aligned_uadd_sat);
   CHECK(!progress);
   nir_validate_shader(b.shader, "unaligned_stays");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.uadd_sat == 4);
   CHECK(c.iadd_nuw == 0);
   ralloc_free(b.shader);
}

/* Edge cases: k == 0 simplifies directly to the exact non-constant operand;
 * swizzled vector lanes, two-constant, vector, and 16-bit uadd_sat preserve
 * their exact operands and opcodes. */
static void
test_edge_cases(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "edge_cases");
   nir_def *idx = make_input_u32(&b, 0);
   nir_def *other = make_input_u32(&b, 4);
   nir_def *shifted16 = nir_ishl_imm(&b, idx, 4);

   /* k == 0 replaces with other directly without emitting iadd. */
   nir_def *zero_add = nir_uadd_sat(&b, other, nir_imm_int(&b, 0));

   /* Swizzle into a vector must inspect the selected component, not .x. */
   nir_def *pair = nir_vec2(&b, shifted16, other);
   nir_def *bad_lane = nir_uadd_sat(&b, nir_channel(&b, pair, 1), nir_imm_int(&b, 4));
   nir_def *good_lane = nir_uadd_sat(&b, nir_channel(&b, pair, 0), nir_imm_int(&b, 12));

   /* Both constants: left to constant folding. */
   nir_def *both_const = nir_uadd_sat(&b, nir_imm_int(&b, 0xfffffff0u),
                                      nir_imm_int(&b, 16));

   /* Non-scalar, non-32-bit, and vector-input scalar ALU (pack_32_2x16) are
    * left untouched without asserting in m5_offset_aligned(). */
   nir_def *vec = nir_uadd_sat(&b, nir_vec2(&b, shifted16, shifted16),
                               nir_imm_ivec2(&b, 4, 4));
   nir_def *narrow = nir_uadd_sat(&b, nir_ishl_imm(&b, nir_u2u16(&b, idx), 4),
                                  nir_imm_intN_t(&b, 4, 16));
   nir_def *packed = nir_pack_32_2x16(&b, nir_vec2(&b, nir_u2u16(&b, idx),
                                                   nir_u2u16(&b, other)));
   nir_def *packed_add = nir_uadd_sat(&b, packed, nir_imm_int(&b, 4));

   store_at(&b, 0x200000000ull, zero_add);
   store_at(&b, 0x200000004ull, bad_lane);
   store_at(&b, 0x200000008ull, good_lane);
   store_at(&b, 0x20000000cull, both_const);
   store_at(&b, 0x200000010ull, vec);
   store_at(&b, 0x200000018ull, nir_u2u32(&b, narrow));
   store_at(&b, 0x20000001cull, packed_add);

   bool progress = false;
   NIR_PASS(progress, b.shader, hk_nir_lower_aligned_uadd_sat);
   CHECK(progress);
   nir_validate_shader(b.shader, "edge_cases");

   nir_def *idx_after = find_input_load(b.shader, 0);
   nir_def *other_after = find_input_load(b.shader, 4);
   CHECK(idx_after && other_after);

   nir_intrinsic_instr *st_zero = find_store_at(b.shader, 0x200000000ull);
   nir_intrinsic_instr *st_bad = find_store_at(b.shader, 0x200000004ull);
   nir_intrinsic_instr *st_good = find_store_at(b.shader, 0x200000008ull);
   nir_intrinsic_instr *st_const = find_store_at(b.shader, 0x20000000cull);
   nir_intrinsic_instr *st_vec = find_store_at(b.shader, 0x200000010ull);
   nir_intrinsic_instr *st_narrow = find_store_at(b.shader, 0x200000018ull);
   nir_intrinsic_instr *st_packed = find_store_at(b.shader, 0x20000001cull);
   CHECK(st_zero && st_bad && st_good && st_const && st_vec && st_narrow &&
         st_packed);

   /* zero_add resolved to `other`, not `idx`. */
   nir_scalar s_zero = nir_scalar_resolved(st_zero->src[0].ssa, 0);
   CHECK(s_zero.def == other_after && s_zero.comp == 0);

   /* bad_lane stayed uadd_sat(other, 4). */
   nir_scalar bad_term;
   CHECK(is_binop_with_const(nir_scalar_resolved(st_bad->src[0].ssa, 0),
                             nir_op_uadd_sat, false, &bad_term, 4));
   CHECK(bad_term.def == other_after && bad_term.comp == 0);

   /* good_lane became iadd.nuw(idx << 4, 12). */
   nir_scalar good_term;
   CHECK(is_binop_with_const(nir_scalar_resolved(st_good->src[0].ssa, 0),
                             nir_op_iadd, true, &good_term, 12));
   CHECK(nir_scalar_is_alu(good_term) &&
         nir_scalar_alu_op(good_term) == nir_op_ishl);
   CHECK(nir_scalar_chase_alu_src(good_term, 0).def == idx_after);
   nir_scalar good_shift = nir_scalar_chase_alu_src(good_term, 1);
   CHECK(nir_scalar_is_const(good_shift) && nir_scalar_as_uint(good_shift) == 4);

   /* both_const, vec, narrow, and packed_add stayed uadd_sat. */
   nir_scalar s_const = nir_scalar_resolved(st_const->src[0].ssa, 0);
   CHECK(nir_scalar_is_alu(s_const) && nir_scalar_alu_op(s_const) == nir_op_uadd_sat);
   nir_alu_instr *vec_alu = nir_def_as_alu(st_vec->src[0].ssa);
   CHECK(vec_alu->op == nir_op_uadd_sat && vec_alu->def.num_components == 2);
   nir_scalar s_narrow = nir_scalar_resolved(st_narrow->src[0].ssa, 0);
   CHECK(nir_scalar_is_alu(s_narrow) && nir_scalar_alu_op(s_narrow) == nir_op_u2u32);
   nir_scalar narrow_src = nir_scalar_chase_alu_src(s_narrow, 0);
   CHECK(nir_scalar_is_alu(narrow_src) &&
         nir_scalar_alu_op(narrow_src) == nir_op_uadd_sat &&
         narrow_src.def->bit_size == 16);
   nir_scalar packed_term;
   CHECK(is_binop_with_const(nir_scalar_resolved(st_packed->src[0].ssa, 0),
                             nir_op_uadd_sat, false, &packed_term, 4));
   CHECK(nir_scalar_is_alu(packed_term) &&
         nir_scalar_alu_op(packed_term) == nir_op_pack_32_2x16);

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.uadd_sat == 5);
   CHECK(c.iadd_nuw == 1);
   ralloc_free(b.shader);
}

/* Lowering aligned uadd_sat to iadd.nuw feeds agx_nir_share_bounds even without
 * size_below_u32_max, producing exact immediates 4 and 12 against one shared
 * usub_sat(size, idx << 4). */
static void
test_feeds_share_bounds(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "feeds_share_bounds");
   nir_def *idx = make_input_u32(&b, 0);
   nir_def *size = make_input_u32(&b, 4);
   nir_def *d0 = make_input_u32(&b, 8);
   nir_def *d1 = make_input_u32(&b, 12);

   nir_def *base = nir_ishl_imm(&b, idx, 4);
   nir_def *end0 = nir_uadd_sat(&b, base, nir_imm_int(&b, 4));
   nir_def *off1 = nir_uadd_sat(&b, base, nir_imm_int(&b, 8));
   nir_def *end1 = nir_uadd_sat(&b, off1, nir_imm_int(&b, 4));

   nir_def *b0 = nir_bounds_agx(&b, d0, end0, size);
   nir_def *b1 = nir_bounds_agx(&b, d1, end1, size);
   store_at(&b, 0x200000000ull, b0);
   store_at(&b, 0x200000004ull, b1);

   bool p1 = false, p2 = false;
   NIR_PASS(p1, b.shader, hk_nir_lower_aligned_uadd_sat);
   NIR_PASS(p2, b.shader, agx_nir_share_bounds, false);
   CHECK(p1 && p2);
   nir_validate_shader(b.shader, "feeds_share_bounds");

   nir_def *idx_after = find_input_load(b.shader, 0);
   nir_def *size_after = find_input_load(b.shader, 4);
   nir_def *d0_after = find_input_load(b.shader, 8);
   nir_def *d1_after = find_input_load(b.shader, 12);
   nir_intrinsic_instr *st0 = find_store_at(b.shader, 0x200000000ull);
   nir_intrinsic_instr *st1 = find_store_at(b.shader, 0x200000004ull);
   CHECK(idx_after && size_after && d0_after && d1_after && st0 && st1);

   nir_scalar s0 = nir_scalar_resolved(st0->src[0].ssa, 0);
   nir_scalar s1 = nir_scalar_resolved(st1->src[0].ssa, 0);
   CHECK(nir_scalar_is_alu(s0) && nir_scalar_alu_op(s0) == nir_op_bounds_agx);
   CHECK(nir_scalar_is_alu(s1) && nir_scalar_alu_op(s1) == nir_op_bounds_agx);

   CHECK(nir_scalar_chase_alu_src(s0, 0).def == d0_after);
   CHECK(nir_scalar_chase_alu_src(s1, 0).def == d1_after);
   nir_scalar imm0 = nir_scalar_chase_alu_src(s0, 1);
   nir_scalar imm1 = nir_scalar_chase_alu_src(s1, 1);
   CHECK(nir_scalar_is_const(imm0) && nir_scalar_as_uint(imm0) == 4);
   CHECK(nir_scalar_is_const(imm1) && nir_scalar_as_uint(imm1) == 12);

   nir_scalar rem0 = nir_scalar_chase_alu_src(s0, 2);
   nir_scalar rem1 = nir_scalar_chase_alu_src(s1, 2);
   CHECK(nir_scalar_equal(rem0, rem1));
   CHECK(nir_scalar_is_alu(rem0) && nir_scalar_alu_op(rem0) == nir_op_usub_sat);
   CHECK(nir_scalar_chase_alu_src(rem0, 0).def == size_after);
   nir_scalar sub_base = nir_scalar_chase_alu_src(rem0, 1);
   CHECK(nir_scalar_is_alu(sub_base) && nir_scalar_alu_op(sub_base) == nir_op_ishl);
   CHECK(nir_scalar_chase_alu_src(sub_base, 0).def == idx_after);
   nir_scalar sub_shift = nir_scalar_chase_alu_src(sub_base, 1);
   CHECK(nir_scalar_is_const(sub_shift) && nir_scalar_as_uint(sub_shift) == 4);

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.uadd_sat == 0);
   CHECK(c.usub_sat == 1);
   CHECK(c.bounds_agx == 2);
   ralloc_free(b.shader);
}

/* Regression: lowering uadd_sat(X, 12) to iadd.nuw must NOT let
 * hk_nir_merge_buffer_words merge it with wrapping iadd(X, 16) when X is only
 * 16-byte aligned (at X = 0xfffffff0 and size = 16, the second load wraps to
 * offset 0 and is in bounds, while a merged lane would compute end = UINT32_MAX
 * and zero it). */
static void
test_merge_rejects_wrapping_boundary(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "merge_rejects_wrap");
   nir_def *base = make_input_u64(&b, 0);
   nir_def *idx = make_input_u32(&b, 8);
   nir_def *size = make_input_u32(&b, 12);
   nir_def *x = nir_ishl_imm(&b, idx, 4);

   nir_def *off0 = nir_uadd_sat(&b, x, nir_imm_int(&b, 12));
   nir_def *off1 = nir_iadd_imm(&b, x, 16);
   nir_def *l0 = nir_load_global_bounded(&b, 1, 32, base, off0, size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *l1 = nir_load_global_bounded(&b, 1, 32, base, off1, size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   store_at(&b, 0x200000000ull, l0);
   store_at(&b, 0x200000004ull, l1);

   bool lowered = false, merged = false;
   NIR_PASS(lowered, b.shader, hk_nir_lower_aligned_uadd_sat);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(merged, b.shader, hk_nir_merge_buffer_words);
   CHECK(lowered);
   CHECK(!merged);
   nir_validate_shader(b.shader, "merge_rejects_wrap");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.load_global_bounded == 2);
   CHECK(c.load_global == 0);
   CHECK(c.bounds_agx == 0);
   ralloc_free(b.shader);
}

/* Positive merge case: aligned uadd_sat structured-buffer offsets within the
 * same 16-byte record (X, uadd_sat(X, 4), uadd_sat(X, 8), uadd_sat(X, 12))
 * lower to iadd.nuw, merge into one vec4 load_global, and the subsequent
 * aligned uadd_sat pass lowers the first three per-word bounds ends while
 * leaving uadd_sat(X, 16) intact. */
static void
test_merge_aligned_record_succeeds(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "merge_aligned_record");
   nir_def *base = make_input_u64(&b, 0);
   nir_def *idx = make_input_u32(&b, 8);
   nir_def *size = make_input_u32(&b, 12);
   nir_def *x = nir_ishl_imm(&b, idx, 4);

   nir_def *l0 = nir_load_global_bounded(&b, 1, 32, base, x, size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *l1 = nir_load_global_bounded(&b, 1, 32, base,
                                         nir_uadd_sat(&b, x, nir_imm_int(&b, 4)),
                                         size, .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *l2 = nir_load_global_bounded(&b, 1, 32, base,
                                         nir_uadd_sat(&b, x, nir_imm_int(&b, 8)),
                                         size, .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *l3 = nir_load_global_bounded(&b, 1, 32, base,
                                         nir_uadd_sat(&b, x, nir_imm_int(&b, 12)),
                                         size, .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   store_at(&b, 0x200000000ull, nir_vec4(&b, l0, l1, l2, l3));

   bool lowered1 = false, merged = false, lowered2 = false;
   NIR_PASS(lowered1, b.shader, hk_nir_lower_aligned_uadd_sat);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(merged, b.shader, hk_nir_merge_buffer_words);
   NIR_PASS(lowered2, b.shader, hk_nir_lower_aligned_uadd_sat);
   CHECK(lowered1 && merged && lowered2);
   nir_validate_shader(b.shader, "merge_aligned_record");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.load_global_bounded == 0);
   CHECK(c.load_global == 1);
   CHECK(c.load_global_max_comps == 4);
   CHECK(c.bounds_agx == 4);
   /* Ends +4, +8, +12 lower to iadd.nuw; end +16 stays uadd_sat. */
   CHECK(c.uadd_sat == 1);
   ralloc_free(b.shader);
}

/* Regression: merging an earlier group whose load supplies the bound of a
 * later mergeable group must rewrite the later group's bounds_agx to the
 * live merged value instead of a removed SSA definition. */
static void
test_merge_bound_from_earlier_group(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "merge_chained_bound");
   nir_def *base0 = make_input_u64(&b, 0);
   nir_def *base1 = make_input_u64(&b, 8);
   nir_def *idx = make_input_u32(&b, 16);
   nir_def *size0 = make_input_u32(&b, 20);
   nir_def *x = nir_ishl_imm(&b, idx, 4);

   /* Group 0: two bounded words made mergeable by aligned uadd_sat lowering. */
   nir_def *a0 = nir_load_global_bounded(&b, 1, 32, base0, x, size0,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *a1 = nir_load_global_bounded(&b, 1, 32, base0,
                                         nir_uadd_sat(&b, x, nir_imm_int(&b, 4)),
                                         size0, .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);

   /* Group 1: uses a1 (from Group 0) as its descriptor bound. */
   nir_def *b0 = nir_load_global_bounded(&b, 1, 32, base1, x, a1,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *b1 = nir_load_global_bounded(&b, 1, 32, base1,
                                         nir_uadd_sat(&b, x, nir_imm_int(&b, 4)),
                                         a1, .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   store_at(&b, 0x200000000ull, nir_vec4(&b, a0, a1, b0, b1));

   bool lowered = false, merged = false;
   NIR_PASS(lowered, b.shader, hk_nir_lower_aligned_uadd_sat);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(merged, b.shader, hk_nir_merge_buffer_words);
   CHECK(lowered && merged);
   nir_validate_shader(b.shader, "merge_chained_bound");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.load_global_bounded == 0);
   CHECK(c.load_global == 2);
   CHECK(c.bounds_agx == 4);
   ralloc_free(b.shader);
}

/* Regression: constant-offset bounded loads at 0xfffffff8 and 0xfffffffc
 * must not merge because 0xfffffffc + 4 > UINT32_MAX returns zero in
 * separate lowering, whereas merged uadd_sat saturates to UINT32_MAX. */
static void
test_merge_rejects_const_end_overflow(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "merge_const_end_overflow");
   nir_def *base = make_input_u64(&b, 0);
   nir_def *size = make_input_u32(&b, 8);

   nir_def *l0 = nir_load_global_bounded(&b, 1, 32, base,
                                         nir_imm_int(&b, 0xfffffff8u), size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *l1 = nir_load_global_bounded(&b, 1, 32, base,
                                         nir_imm_int(&b, 0xfffffffcu), size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   store_at(&b, 0x200000000ull, nir_vec2(&b, l0, l1));

   bool merged = false;
   NIR_PASS(merged, b.shader, hk_nir_merge_buffer_words);
   CHECK(!merged);
   nir_validate_shader(b.shader, "merge_const_end_overflow");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.load_global_bounded == 2);
   CHECK(c.load_global == 0);
   CHECK(c.bounds_agx == 0);
   ralloc_free(b.shader);
}

/* Shared-base iadd.nuw(base, k0) + iadd.nuw(base, k0 + 4) on a non-power-of-two
 * stride proves no wrap without power-of-two alignment and still merges. */
static void
test_merge_shared_base_iadd_nuw(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "merge_shared_iadd_nuw");
   nir_def *base = make_input_u64(&b, 0);
   nir_def *idx = make_input_u32(&b, 8);
   nir_def *size = make_input_u32(&b, 12);
   nir_def *row = nir_imul_imm(&b, idx, 12);

   nir_def *off0 = nir_iadd(&b, row, nir_imm_int(&b, 4));
   nir_def_as_alu(off0)->no_unsigned_wrap = true;
   nir_def *off1 = nir_iadd(&b, row, nir_imm_int(&b, 8));
   nir_def_as_alu(off1)->no_unsigned_wrap = true;

   nir_def *l0 = nir_load_global_bounded(&b, 1, 32, base, off0, size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   nir_def *l1 = nir_load_global_bounded(&b, 1, 32, base, off1, size,
                                         .align_mul = 4,
                                         .access = ACCESS_CAN_REORDER);
   store_at(&b, 0x200000000ull, nir_vec2(&b, l0, l1));

   bool merged = false;
   NIR_PASS(merged, b.shader, hk_nir_merge_buffer_words);
   CHECK(merged);
   nir_validate_shader(b.shader, "merge_shared_iadd_nuw");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.load_global_bounded == 0);
   CHECK(c.load_global == 1);
   CHECK(c.load_global_max_comps == 2);
   CHECK(c.bounds_agx == 2);
   ralloc_free(b.shader);
}

/* Regression: raw physical load_global at base + u2u64(off) and
 * base + u2u64(off + 4) must not merge without a no-wrap proof because
 * off = 0xfffffffc wraps the second 32-bit offset to 0, whereas a merged vec2
 * would read at base + 0x100000000. */
static void
test_merge_rejects_raw_global_wrap(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "merge_raw_global_wrap");
   nir_def *base = make_input_u64(&b, 0);
   nir_def *idx = make_input_u32(&b, 8);
   nir_def *x = nir_ishl_imm(&b, idx, 4);

   nir_def *a0 = nir_iadd(&b, base, nir_u2u64(&b, nir_uadd_sat(&b, x, nir_imm_int(&b, 12))));
   nir_def *a1 = nir_iadd(&b, base, nir_u2u64(&b, nir_iadd_imm(&b, x, 16)));
   nir_def *l0 = nir_load_global(&b, 1, 32, a0, .align_mul = 4,
                                 .access = ACCESS_CAN_REORDER);
   nir_def *l1 = nir_load_global(&b, 1, 32, a1, .align_mul = 4,
                                 .access = ACCESS_CAN_REORDER);
   store_at(&b, 0x200000000ull, nir_vec2(&b, l0, l1));

   bool lowered = false, merged = false;
   NIR_PASS(lowered, b.shader, hk_nir_lower_aligned_uadd_sat);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(merged, b.shader, hk_nir_merge_buffer_words);
   CHECK(lowered);
   CHECK(!merged);
   nir_validate_shader(b.shader, "merge_raw_global_wrap");

   struct shader_counts c = count_ops(b.shader);
   CHECK(c.load_global == 2);
   CHECK(c.load_global_max_comps == 1);
   ralloc_free(b.shader);
}

int
main(void)
{
   test_aligned_lowers_to_iadd_nuw();
   test_unaligned_and_boundary_stay_uadd_sat();
   test_edge_cases();
   test_feeds_share_bounds();
   test_merge_rejects_wrapping_boundary();
   test_merge_aligned_record_succeeds();
   test_merge_bound_from_earlier_group();
   test_merge_rejects_const_end_overflow();
   test_merge_shared_base_iadd_nuw();
   test_merge_rejects_raw_global_wrap();
   return 0;
}
