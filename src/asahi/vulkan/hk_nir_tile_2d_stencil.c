/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_2d_stencil.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"

#include "compiler/glsl_types.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "util/u_math.h"

/* ---------------------------------------------------------------------------
 * Canonicalization & structural matching helpers
 * --------------------------------------------------------------------------- */

static nir_def *
strip_mov(nir_def *def)
{
   while (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov && alu->src[0].src.ssa->num_components == 1) {
         def = alu->src[0].src.ssa;
         continue;
      }
      break;
   }
   return def;
}

static bool
match_const_i32(nir_def *def, int32_t *val_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   *val_out = nir_src_as_int(nir_src_for_ssa(def));
   return true;
}

static bool
is_const_i32(nir_def *def, int32_t expected)
{
   int32_t v = 0;
   return match_const_i32(def, &v) && v == expected;
}

static bool
is_const_bool(nir_def *def, bool expected)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 1 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   return nir_src_as_bool(nir_src_for_ssa(def)) == expected;
}

static bool
match_push_const_u32(nir_def *def, uint16_t *offset_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;

   unsigned comp = 0;
   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op != nir_op_mov)
         return false;
      comp = alu->src[0].swizzle[0];
      def = alu->src[0].src.ssa;
   }

   if (!def || def->bit_size != 32 ||
       nir_def_instr(def)->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(def));
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   if (!nir_src_is_const(intr->src[0]))
      return false;

   uint64_t off = (uint64_t)nir_intrinsic_base(intr) +
                  nir_src_as_uint(intr->src[0]) + (uint64_t)comp * 4u;
   if ((off & 3u) != 0 || off + 4u > HK_MAX_PUSH_SIZE)
      return false;

   *offset_out = (uint16_t)off;
   return true;
}

static bool
same_push_const(nir_def *a, nir_def *b)
{
   if (strip_mov(a) == strip_mov(b))
      return true;
   uint16_t oa = 0, ob = 0;
   return match_push_const_u32(a, &oa) && match_push_const_u32(b, &ob) &&
          oa == ob;
}

static bool
match_sysval_comp(nir_def *def, nir_intrinsic_op op, unsigned *comp_out)
{
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov) {
         nir_def *src = alu->src[0].src.ssa;
         if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op) {
            *comp_out = alu->src[0].swizzle[0];
            return true;
         }
      }
   }
   return false;
}

static bool
match_sysval_alu_src(const nir_alu_src *asrc, nir_intrinsic_op op,
                     unsigned *comp_out)
{
   nir_def *src = asrc->src.ssa;
   if (!src || src->bit_size != 32)
      return false;
   if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op) {
      *comp_out = asrc->swizzle[0];
      return true;
   }
   return match_sysval_comp(src, op, comp_out);
}

static bool
match_scaled_sysval_comp(const nir_alu_src *asrc, nir_intrinsic_op op,
                         const uint32_t wg_size[3], unsigned comp)
{
   if (comp >= 3 || wg_size[comp] == 0)
      return false;

   unsigned sc = 0;
   if (wg_size[comp] == 1 && match_sysval_alu_src(asrc, op, &sc) && sc == comp)
      return true;

   nir_def *bm = strip_mov(asrc->src.ssa);
   if (!bm || nir_def_instr(bm)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(bm));

   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      for (unsigned side = 0; side < 2; side++) {
         int32_t c = 0;
         if (match_sysval_alu_src(&alu->src[side], op, &sc) && sc == comp &&
             match_const_i32(alu->src[1 - side].src.ssa, &c) &&
             (uint32_t)c == wg_size[comp])
            return true;
      }
   } else if (alu->op == nir_op_ishl) {
      int32_t sh = 0;
      if (match_sysval_alu_src(&alu->src[0], op, &sc) && sc == comp &&
          match_const_i32(alu->src[1].src.ssa, &sh) && sh >= 0 && sh < 31 &&
          (1u << sh) == wg_size[comp])
         return true;
   } else if (alu->op == nir_op_iadd && wg_size[comp] == 2) {
      unsigned sc0 = 0, sc1 = 0;
      if (match_sysval_alu_src(&alu->src[0], op, &sc0) && sc0 == comp &&
          match_sysval_alu_src(&alu->src[1], op, &sc1) && sc1 == comp)
         return true;
   }

   return false;
}

/* Match global invocation coordinate `comp` (0 for X, 1 for Y), handling:
 *   - direct `load_global_invocation_id[comp]`
 *   - `load_global_invocation_id[comp] + load_base_workgroup_id[comp] * wg[comp]`
 *   - `load_global_invocation_id[comp] + load_base_global_invocation_id[comp]`
 *   - `load_workgroup_id[comp] * wg[comp] + load_local_invocation_id[comp]`
 */
static bool
match_global_id_comp(nir_def *def, const uint32_t wg_size[3],
                     unsigned *comp_out)
{
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;

   unsigned c = 0;
   if (match_sysval_comp(def, nir_intrinsic_load_global_invocation_id, &c)) {
      *comp_out = c;
      return true;
   }

   nir_def *s = strip_mov(def);
   if (!s || nir_def_instr(s)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(s));
   if (add->op != nir_op_iadd)
      return false;

   for (unsigned side = 0; side < 2; side++) {
      if (match_sysval_alu_src(&add->src[side],
                               nir_intrinsic_load_global_invocation_id, &c)) {
         unsigned bc = 0;
         if (match_sysval_alu_src(
                &add->src[1 - side],
                nir_intrinsic_load_base_global_invocation_id, &bc) &&
             bc == c) {
            *comp_out = c;
            return true;
         }
         if (match_scaled_sysval_comp(&add->src[1 - side],
                                      nir_intrinsic_load_base_workgroup_id,
                                      wg_size, c)) {
            *comp_out = c;
            return true;
         }
      }
      if (match_sysval_alu_src(&add->src[side],
                               nir_intrinsic_load_local_invocation_id, &c) &&
          match_scaled_sysval_comp(&add->src[1 - side],
                                   nir_intrinsic_load_workgroup_id, wg_size,
                                   c)) {
         *comp_out = c;
         return true;
      }
   }

   return false;
}

static bool
same_coord(nir_def *a, nir_def *b, const uint32_t wg_size[3])
{
   if (strip_mov(a) == strip_mov(b))
      return true;
   unsigned ca = 0, cb = 0;
   return match_global_id_comp(a, wg_size, &ca) &&
          match_global_id_comp(b, wg_size, &cb) && ca == cb;
}

static bool
same_val(nir_def *a, nir_def *b, const uint32_t wg_size[3])
{
   if (strip_mov(a) == strip_mov(b))
      return true;
   if (same_push_const(a, b))
      return true;
   if (same_coord(a, b, wg_size))
      return true;
   int32_t ca = 0, cb = 0;
   return match_const_i32(a, &ca) && match_const_i32(b, &cb) && ca == cb;
}

/* Match `a + b` (or `a - (-b)`) in either operand order. */
static bool
match_iadd_pair(nir_def *def, nir_def *a, nir_def *b,
                const uint32_t wg_size[3])
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_iadd)
      return false;
   nir_def *s0 = alu->src[0].src.ssa;
   nir_def *s1 = alu->src[1].src.ssa;
   return (same_val(s0, a, wg_size) && same_val(s1, b, wg_size)) ||
          (same_val(s0, b, wg_size) && same_val(s1, a, wg_size));
}

/* Match `a - b` (or `a + ineg(b)`). */
static bool
match_isub_pair(nir_def *def, nir_def *a, nir_def *b,
                const uint32_t wg_size[3])
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_isub) {
      return same_val(alu->src[0].src.ssa, a, wg_size) &&
             same_val(alu->src[1].src.ssa, b, wg_size);
   }
   if (alu->op == nir_op_iadd) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *pos = alu->src[side].src.ssa;
         nir_def *neg = strip_mov(alu->src[1 - side].src.ssa);
         if (!same_val(pos, a, wg_size) || !neg ||
             nir_def_instr(neg)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *na = nir_instr_as_alu(nir_def_instr(neg));
         if (na->op == nir_op_ineg && same_val(na->src[0].src.ssa, b, wg_size))
            return true;
      }
      int32_t cb = 0;
      if (match_const_i32(b, &cb)) {
         for (unsigned side = 0; side < 2; side++) {
            int32_t c = 0;
            if (same_val(alu->src[side].src.ssa, a, wg_size) &&
                match_const_i32(alu->src[1 - side].src.ssa, &c) && c == -cb)
               return true;
         }
      }
   }
   return false;
}

/* Match `row * stride + col` in either add/mul operand order. */
static bool
match_linear_index(nir_def *idx, nir_def *row, nir_def *col,
                   uint16_t *stride_pc_out, const uint32_t wg_size[3])
{
   idx = strip_mov(idx);
   if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(idx));
   if (add->op != nir_op_iadd)
      return false;

   for (unsigned side = 0; side < 2; side++) {
      if (!same_val(add->src[side].src.ssa, col, wg_size))
         continue;
      nir_def *prod = strip_mov(add->src[1 - side].src.ssa);
      if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
      if (mul->op != nir_op_imul && mul->op != nir_op_amul)
         continue;
      for (unsigned ms = 0; ms < 2; ms++) {
         uint16_t spc = 0;
         if (same_val(mul->src[ms].src.ssa, row, wg_size) &&
             match_push_const_u32(mul->src[1 - ms].src.ssa, &spc)) {
            *stride_pc_out = spc;
            return true;
         }
      }
   }
   return false;
}

/* Decompose a boolean conjunction into leaf terms (flattening `iand`,
 * `bcsel(a, b, false)`, `inot(ior(a, b))`, `inot(bcsel(a, true, b))`, etc.).
 * Each output entry carries `(def, negated)`. */
struct bool_term {
   nir_def *def;
   bool negated;
};

static bool
collect_and_terms(nir_def *cond, bool negated, struct bool_term *out,
                  unsigned *count, unsigned max_terms)
{
   cond = strip_mov(cond);
   if (!cond || cond->bit_size != 1 || cond->num_components != 1)
      return false;

   if (nir_def_instr(cond)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
      if (alu->op == nir_op_inot)
         return collect_and_terms(alu->src[0].src.ssa, !negated, out, count,
                                  max_terms);

      if (!negated && alu->op == nir_op_iand) {
         return collect_and_terms(alu->src[0].src.ssa, false, out, count,
                                  max_terms) &&
                collect_and_terms(alu->src[1].src.ssa, false, out, count,
                                  max_terms);
      }
      if (negated && alu->op == nir_op_ior) {
         return collect_and_terms(alu->src[0].src.ssa, true, out, count,
                                  max_terms) &&
                collect_and_terms(alu->src[1].src.ssa, true, out, count,
                                  max_terms);
      }
      if (alu->op == nir_op_bcsel) {
         nir_def *c = alu->src[0].src.ssa;
         nir_def *t = alu->src[1].src.ssa;
         nir_def *e = alu->src[2].src.ssa;
         if (!negated) {
            if (is_const_bool(e, false))
               return collect_and_terms(c, false, out, count, max_terms) &&
                      collect_and_terms(t, false, out, count, max_terms);
            if (is_const_bool(t, false))
               return collect_and_terms(c, true, out, count, max_terms) &&
                      collect_and_terms(e, false, out, count, max_terms);
         } else {
            if (is_const_bool(t, true))
               return collect_and_terms(c, true, out, count, max_terms) &&
                      collect_and_terms(e, true, out, count, max_terms);
            if (is_const_bool(e, true))
               return collect_and_terms(c, false, out, count, max_terms) &&
                      collect_and_terms(t, true, out, count, max_terms);
         }
      }
   }

   if (*count >= max_terms)
      return false;
   out[(*count)++] = (struct bool_term){.def = cond, .negated = negated};
   return true;
}

/* Normalize a signed/unsigned integer comparison term to either:
 *   `lhs < rhs` (`is_lt = true`) or `lhs >= rhs` (`is_lt = false`). */
static bool
decompose_cmp_term(struct bool_term t, nir_def **lhs_out, nir_def **rhs_out,
                   bool *is_lt_out, bool *is_unsigned_out)
{
   if (nir_def_instr(t.def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(t.def));
   nir_def *a = alu->src[0].src.ssa;
   nir_def *b = alu->src[1].src.ssa;

   switch (alu->op) {
   case nir_op_ilt:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = !t.negated;
      *is_unsigned_out = false;
      return true;
   case nir_op_ige:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = t.negated;
      *is_unsigned_out = false;
      return true;
   case nir_op_ult:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = !t.negated;
      *is_unsigned_out = true;
      return true;
   case nir_op_uge:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = t.negated;
      *is_unsigned_out = true;
      return true;
   default:
      return false;
   }
}

/* Match `x < width && y < height` (returning `x`, `y`, `width_pc`, `height_pc`). */
static bool
match_xy_bounds_cond(nir_def *cond, bool negated, const uint32_t wg_size[3],
                     nir_def **x_out, nir_def **y_out,
                     nir_def **w_pc_out, nir_def **h_pc_out)
{
   struct bool_term terms[4];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 4) || n != 2)
      return false;

   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   for (unsigned i = 0; i < 2; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u))
         return false;
      if (!is_lt)
         continue;
      unsigned comp = 0;
      uint16_t pc = 0;
      if (!match_global_id_comp(lhs, wg_size, &comp) || comp > 1 ||
          !match_push_const_u32(rhs, &pc))
         return false;
      if (comp == 0) {
         if (x)
            return false;
         x = lhs;
         w_pc = rhs;
      } else {
         if (y)
            return false;
         y = lhs;
         h_pc = rhs;
      }
   }

   if (!x || !y || !w_pc || !h_pc)
      return false;

   *x_out = x;
   *y_out = y;
   *w_pc_out = w_pc;
   *h_pc_out = h_pc;
   return true;
}

/* Match SSBO array deref `(ssbo elem_type[])[idx]` returning set, binding,
 * element bit size, and element index SSA def. */
static bool
match_ssbo_deref_access(nir_deref_instr *deref, unsigned expected_bits,
                        uint32_t *set_out, uint32_t *binding_out,
                        nir_def **idx_out)
{
   if (!deref ||
       (deref->deref_type != nir_deref_type_array &&
        deref->deref_type != nir_deref_type_ptr_as_array))
      return false;
   if (!nir_deref_mode_is(deref, nir_var_mem_ssbo))
      return false;
   if (!glsl_type_is_scalar(deref->type) ||
       glsl_get_bit_size(deref->type) != expected_bits)
      return false;
   if (nir_deref_instr_array_stride(deref) != expected_bits / 8u)
      return false;

   nir_def *idx = deref->arr.index.ssa;
   if (!idx || idx->bit_size != 32 || idx->num_components != 1)
      return false;

   /* Walk up deref_struct / deref_cast to load_vulkan_descriptor ->
    * vulkan_resource_index, verifying zero struct field offset. */
   nir_deref_instr *cur = nir_deref_instr_parent(deref);
   while (cur) {
      if (cur->deref_type == nir_deref_type_struct) {
         nir_deref_instr *parent = nir_deref_instr_parent(cur);
         if (!parent || !glsl_type_is_struct_or_ifc(parent->type))
            return false;
         if (glsl_get_struct_field_offset(parent->type, cur->strct.index) != 0)
            return false;
         cur = parent;
      } else if (cur->deref_type == nir_deref_type_cast) {
         if (nir_def_instr(cur->parent.ssa)->type == nir_instr_type_deref) {
            cur = nir_src_as_deref(cur->parent);
            continue;
         }
         if (cur->cast.ptr_stride != 0 &&
             cur->cast.ptr_stride != expected_bits / 8u)
            return false;
         nir_def *src = cur->parent.ssa;
         if (!src || nir_def_instr(src)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *lvd = nir_instr_as_intrinsic(nir_def_instr(src));
         if (lvd->intrinsic != nir_intrinsic_load_vulkan_descriptor)
            return false;
         nir_def *rsrc = lvd->src[0].ssa;
         if (!rsrc || nir_def_instr(rsrc)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *vri = nir_instr_as_intrinsic(nir_def_instr(rsrc));
         if (vri->intrinsic != nir_intrinsic_vulkan_resource_index)
            return false;
         if (!nir_src_is_const(vri->src[0]) || nir_src_as_uint(vri->src[0]) != 0)
            return false;
         *set_out = nir_intrinsic_desc_set(vri);
         *binding_out = nir_intrinsic_binding(vri);
         *idx_out = idx;
         return true;
      } else {
         return false;
      }
   }

   return false;
}

/* Match SSBO array deref `(ssbo vecN[])[idx]` (or vectorized cast
 * `(vecN *)&(ssbo struct { scalar x, y; }[])[idx].x`) returning set, binding,
 * and element index SSA def. */
static bool
match_ssbo_deref_vec_access(nir_deref_instr *deref, unsigned comps,
                            unsigned expected_bits, uint32_t *set_out,
                            uint32_t *binding_out, nir_def **idx_out)
{
   if (!deref || !nir_deref_mode_is(deref, nir_var_mem_ssbo))
      return false;
   if (!glsl_type_is_vector(deref->type) ||
       glsl_get_vector_elements(deref->type) != comps ||
       glsl_get_bit_size(deref->type) != expected_bits)
      return false;
   const unsigned stride = comps * (expected_bits / 8u);

   /* Handle `nir_opt_load_store_vectorize` wrapping `&arr[idx].x` in a vector cast. */
   if (deref->deref_type == nir_deref_type_cast) {
      nir_deref_instr *p = nir_deref_instr_parent(deref);
      if (p && p->deref_type == nir_deref_type_struct) {
         nir_deref_instr *sp = nir_deref_instr_parent(p);
         if (!sp || !glsl_type_is_struct_or_ifc(sp->type) ||
             glsl_get_struct_field_offset(sp->type, p->strct.index) != 0)
            return false;
         deref = sp;
      } else if (p) {
         deref = p;
      }
   }

   if (!deref ||
       (deref->deref_type != nir_deref_type_array &&
        deref->deref_type != nir_deref_type_ptr_as_array))
      return false;
   const unsigned arr_stride = nir_deref_instr_array_stride(deref);
   const unsigned scalar_stride = expected_bits / 8u;
   nir_def *idx = deref->arr.index.ssa;
   if (!idx || idx->bit_size != 32 || idx->num_components != 1)
      return false;

   if (arr_stride == stride) {
      /* Direct vecN[] or struct { scalar[N] }[] indexing. */
   } else if (comps == 2 && arr_stride == scalar_stride) {
      /* Vectorized store to scalar[idx * 2]. */
      nir_def *is = strip_mov(idx);
      if (!is || nir_def_instr(is)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *ia = nir_instr_as_alu(nir_def_instr(is));
      if (ia->op == nir_op_ishl && is_const_i32(ia->src[1].src.ssa, 1)) {
         idx = ia->src[0].src.ssa;
      } else if ((ia->op == nir_op_imul || ia->op == nir_op_amul) &&
                 is_const_i32(ia->src[1].src.ssa, 2)) {
         idx = ia->src[0].src.ssa;
      } else if ((ia->op == nir_op_imul || ia->op == nir_op_amul) &&
                 is_const_i32(ia->src[0].src.ssa, 2)) {
         idx = ia->src[1].src.ssa;
      } else if (ia->op == nir_op_iadd &&
                 strip_mov(ia->src[0].src.ssa) == strip_mov(ia->src[1].src.ssa)) {
         idx = ia->src[0].src.ssa;
      } else {
         return false;
      }
   } else {
      return false;
   }

   nir_deref_instr *cur = nir_deref_instr_parent(deref);
   while (cur) {
      if (cur->deref_type == nir_deref_type_struct) {
         nir_deref_instr *parent = nir_deref_instr_parent(cur);
         if (!parent || !glsl_type_is_struct_or_ifc(parent->type))
            return false;
         if (glsl_get_struct_field_offset(parent->type, cur->strct.index) != 0)
            return false;
         cur = parent;
      } else if (cur->deref_type == nir_deref_type_cast) {
         if (nir_def_instr(cur->parent.ssa)->type == nir_instr_type_deref) {
            cur = nir_src_as_deref(cur->parent);
            continue;
         }
         if (cur->cast.ptr_stride != 0 && cur->cast.ptr_stride != arr_stride)
            return false;
         nir_def *src = cur->parent.ssa;
         if (!src || nir_def_instr(src)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *lvd = nir_instr_as_intrinsic(nir_def_instr(src));
         if (lvd->intrinsic != nir_intrinsic_load_vulkan_descriptor)
            return false;
         nir_def *rsrc = lvd->src[0].ssa;
         if (!rsrc || nir_def_instr(rsrc)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *vri = nir_instr_as_intrinsic(nir_def_instr(rsrc));
         if (vri->intrinsic != nir_intrinsic_vulkan_resource_index)
            return false;
         if (!nir_src_is_const(vri->src[0]) || nir_src_as_uint(vri->src[0]) != 0)
            return false;
         *set_out = nir_intrinsic_desc_set(vri);
         *binding_out = nir_intrinsic_binding(vri);
         *idx_out = idx;
         return true;
      } else {
         return false;
      }
   }

   return false;
}

/* Match scalar 32-bit SSBO store targeting element `idx` component `comp`
 * (0 or 1) of a 2x32-bit point array, supporting either:
 *   - `struct { int32_t x, y; } arr[idx].x/.y` (array stride 8, field offset 0/4)
 *   - `int32_t arr[idx * 2 + comp]` (array stride 4)
 */
static bool
match_ssbo_i32_pair_store_deref(nir_deref_instr *deref, uint32_t *set_out,
                                uint32_t *binding_out, nir_def **idx_out,
                                unsigned *comp_out)
{
   if (!deref || !nir_deref_mode_is(deref, nir_var_mem_ssbo) ||
       !glsl_type_is_scalar(deref->type) ||
       glsl_get_bit_size(deref->type) != 32)
      return false;

   /* Form B1: `deref_struct(field 0/1, offset 0/4)` -> `deref_array(idx)` (stride 8). */
   if (deref->deref_type == nir_deref_type_struct) {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      if (!parent || !glsl_type_is_struct_or_ifc(parent->type))
         return false;
      int off = glsl_get_struct_field_offset(parent->type, deref->strct.index);
      if (off != 0 && off != 4)
         return false;
      if (parent->deref_type != nir_deref_type_array &&
          parent->deref_type != nir_deref_type_ptr_as_array)
         return false;
      if (nir_deref_instr_array_stride(parent) != 8)
         return false;
      nir_def *idx = parent->arr.index.ssa;
      if (!idx || idx->bit_size != 32 || idx->num_components != 1)
         return false;

      nir_deref_instr *cur = nir_deref_instr_parent(parent);
      while (cur) {
         if (cur->deref_type == nir_deref_type_struct) {
            nir_deref_instr *p = nir_deref_instr_parent(cur);
            if (!p || !glsl_type_is_struct_or_ifc(p->type) ||
                glsl_get_struct_field_offset(p->type, cur->strct.index) != 0)
               return false;
            cur = p;
         } else if (cur->deref_type == nir_deref_type_cast) {
            if (nir_def_instr(cur->parent.ssa)->type == nir_instr_type_deref) {
               cur = nir_src_as_deref(cur->parent);
               continue;
            }
            nir_def *src = cur->parent.ssa;
            if (!src || nir_def_instr(src)->type != nir_instr_type_intrinsic)
               return false;
            nir_intrinsic_instr *lvd =
               nir_instr_as_intrinsic(nir_def_instr(src));
            if (lvd->intrinsic != nir_intrinsic_load_vulkan_descriptor)
               return false;
            nir_def *rsrc = lvd->src[0].ssa;
            if (!rsrc || nir_def_instr(rsrc)->type != nir_instr_type_intrinsic)
               return false;
            nir_intrinsic_instr *vri =
               nir_instr_as_intrinsic(nir_def_instr(rsrc));
            if (vri->intrinsic != nir_intrinsic_vulkan_resource_index ||
                !nir_src_is_const(vri->src[0]) ||
                nir_src_as_uint(vri->src[0]) != 0)
               return false;
            *set_out = nir_intrinsic_desc_set(vri);
            *binding_out = nir_intrinsic_binding(vri);
            *idx_out = idx;
            *comp_out = (off == 4) ? 1u : 0u;
            return true;
         } else {
            return false;
         }
      }
      return false;
   }

   /* Form B2: `deref_array(idx * 2 [+ 1])` (stride 4). */
   uint32_t set = 0, bind = 0;
   nir_def *raw_idx = NULL;
   if (!match_ssbo_deref_access(deref, 32, &set, &bind, &raw_idx))
      return false;

   nir_def *base = strip_mov(raw_idx);
   unsigned comp = 0;
   if (base && nir_def_instr(base)->type == nir_instr_type_alu) {
      nir_alu_instr *a = nir_instr_as_alu(nir_def_instr(base));
      if (a->op == nir_op_iadd || a->op == nir_op_ior) {
         if (is_const_i32(a->src[1].src.ssa, 1)) {
            base = strip_mov(a->src[0].src.ssa);
            comp = 1;
         } else if (is_const_i32(a->src[0].src.ssa, 1)) {
            base = strip_mov(a->src[1].src.ssa);
            comp = 1;
         }
      }
   }
   if (!base || nir_def_instr(base)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *m = nir_instr_as_alu(nir_def_instr(base));
   nir_def *pt_idx = NULL;
   if (m->op == nir_op_ishl && is_const_i32(m->src[1].src.ssa, 1)) {
      pt_idx = m->src[0].src.ssa;
   } else if ((m->op == nir_op_imul || m->op == nir_op_amul) &&
              is_const_i32(m->src[1].src.ssa, 2)) {
      pt_idx = m->src[0].src.ssa;
   } else if ((m->op == nir_op_imul || m->op == nir_op_amul) &&
              is_const_i32(m->src[0].src.ssa, 2)) {
      pt_idx = m->src[1].src.ssa;
   } else if (m->op == nir_op_iadd &&
              strip_mov(m->src[0].src.ssa) == strip_mov(m->src[1].src.ssa)) {
      pt_idx = m->src[0].src.ssa;
   } else {
      return false;
   }

   *set_out = set;
   *binding_out = bind;
   *idx_out = pt_idx;
   *comp_out = comp;
   return true;
}

/* Match scalar 32-bit SSBO access at byte offset 0 (either `arr[0]` or a
 * scalar struct member at offset 0). */
static bool
match_ssbo_u32_zero_deref(nir_deref_instr *deref, uint32_t *set_out,
                          uint32_t *binding_out)
{
   nir_def *idx = NULL;
   if (match_ssbo_deref_access(deref, 32, set_out, binding_out, &idx))
      return is_const_i32(idx, 0);

   if (!deref || !nir_deref_mode_is(deref, nir_var_mem_ssbo) ||
       !glsl_type_is_scalar(deref->type) ||
       glsl_get_bit_size(deref->type) != 32)
      return false;

   nir_deref_instr *cur = deref;
   while (cur) {
      if (cur->deref_type == nir_deref_type_struct) {
         nir_deref_instr *parent = nir_deref_instr_parent(cur);
         if (!parent || !glsl_type_is_struct_or_ifc(parent->type))
            return false;
         if (glsl_get_struct_field_offset(parent->type, cur->strct.index) != 0)
            return false;
         cur = parent;
      } else if (cur->deref_type == nir_deref_type_cast) {
         if (nir_def_instr(cur->parent.ssa)->type == nir_instr_type_deref) {
            cur = nir_src_as_deref(cur->parent);
            continue;
         }
         nir_def *src = cur->parent.ssa;
         if (!src || nir_def_instr(src)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *lvd = nir_instr_as_intrinsic(nir_def_instr(src));
         if (lvd->intrinsic != nir_intrinsic_load_vulkan_descriptor)
            return false;
         nir_def *rsrc = lvd->src[0].ssa;
         if (!rsrc || nir_def_instr(rsrc)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *vri = nir_instr_as_intrinsic(nir_def_instr(rsrc));
         if (vri->intrinsic != nir_intrinsic_vulkan_resource_index)
            return false;
         if (!nir_src_is_const(vri->src[0]) || nir_src_as_uint(vri->src[0]) != 0)
            return false;
         *set_out = nir_intrinsic_desc_set(vri);
         *binding_out = nir_intrinsic_binding(vri);
         return true;
      } else {
         return false;
      }
   }

   return false;
}

static bool
populate_binding_info(uint32_t set_layout_count,
                      struct vk_descriptor_set_layout *const *set_layouts,
                      uint32_t set, uint32_t binding,
                      struct hk_2d_stencil_binding *out)
{
   if (set >= HK_MAX_SETS || set >= set_layout_count || !set_layouts ||
       !set_layouts[set])
      return false;

   const struct hk_descriptor_set_layout *layout =
      vk_to_hk_descriptor_set_layout(set_layouts[set]);
   if (layout->vk.flags &
       (VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT |
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT))
      return false;
   if (binding >= layout->binding_count)
      return false;

   const struct hk_descriptor_set_binding_layout *bl = &layout->binding[binding];
   if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || bl->array_size != 1)
      return false;
   if (bl->flags &
       (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT))
      return false;

   out->set = set;
   out->binding = binding;
   out->descriptor_offset_B = bl->offset;
   memcpy(out->set_layout_hash, layout->vk.blake3, sizeof(out->set_layout_hash));
   return true;
}

/* Expand AGX-specific fused integer ops on the scratch matcher clone. */
static bool
expand_agx_ops_instr(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *rep = NULL;

   switch (alu->op) {
   case nir_op_imadshl_agx: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *m = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 2);
      nir_def *sh = nir_ssa_for_alu_src(b, alu, 3);
      rep = nir_iadd(b, nir_imul(b, a, m), nir_ishl(b, c, sh));
      break;
   }
   case nir_op_imsubshl_agx: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *m = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 2);
      nir_def *sh = nir_ssa_for_alu_src(b, alu, 3);
      rep = nir_isub(b, nir_imul(b, a, m), nir_ishl(b, c, sh));
      break;
   }
   case nir_op_amul: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *m = nir_ssa_for_alu_src(b, alu, 1);
      rep = nir_imul(b, a, m);
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&alu->def, rep);
   return true;
}

/* Hoist pure constant/push-constant/sysval loads into the entry block so
 * glslang's short-circuit boolean control flow flattens via peephole_select. */
static bool
hoist_matcher_invariants(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_block *start = nir_start_block(impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      if (block == start)
         continue;
      nir_foreach_instr_safe(instr, block) {
         bool hoist = false;
         if (instr->type == nir_instr_type_load_const) {
            hoist = true;
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_push_constant:
               hoist = nir_src_is_const(intr->src[0]);
               break;
            case nir_intrinsic_load_global_invocation_id:
            case nir_intrinsic_load_base_global_invocation_id:
            case nir_intrinsic_load_base_workgroup_id:
            case nir_intrinsic_load_local_invocation_id:
            case nir_intrinsic_load_workgroup_id:
            case nir_intrinsic_load_num_workgroups:
            case nir_intrinsic_load_workgroup_size:
               hoist = true;
               break;
            default:
               break;
            }
         }
         if (hoist) {
            nir_instr_remove(instr);
            nir_instr_insert_after_block(start, instr);
            progress = true;
         }
      }
   }

   if (progress)
      nir_progress(true, impl, nir_metadata_control_flow);
   return progress;
}

static bool
match_loop_phi(nir_loop *loop, nir_phi_instr *phi, nir_def **init_out,
               nir_def **next_out)
{
   nir_def *init = NULL, *next = NULL;
   nir_block *first = nir_loop_first_block(loop);
   nir_block *last = nir_loop_last_block(loop);

   nir_foreach_phi_src(src, phi) {
      if (src->pred->index < first->index) {
         if (init)
            return false;
         init = src->src.ssa;
      } else if (src->pred->index >= first->index &&
                 src->pred->index <= last->index) {
         if (next)
            return false;
         next = src->src.ssa;
      } else {
         return false;
      }
   }
   if (!init || !next)
      return false;
   *init_out = init;
   *next_out = next;
   return true;
}

static bool
cf_list_is_empty_block(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   if (n->type != nir_cf_node_block)
      return false;
   return exec_list_is_empty(&nir_cf_node_as_block(n)->instr_list);
}

static bool
cf_list_is_single_break(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *blk = nir_cf_node_as_block(n);
   if (!exec_list_is_singular(&blk->instr_list))
      return false;
   nir_instr *i = nir_block_first_instr(blk);
   return i->type == nir_instr_type_jump &&
          nir_instr_as_jump(i)->type == nir_jump_break;
}

static bool
cf_list_is_single_return(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *blk = nir_cf_node_as_block(n);
   if (!exec_list_is_singular(&blk->instr_list))
      return false;
   nir_instr *i = nir_block_first_instr(blk);
   return i->type == nir_instr_type_jump &&
          nir_instr_as_jump(i)->type == nir_jump_return;
}

/* Match a loop header break `if (iv > hi) break;` (or `if (iv <= hi) { body } else { break; }`),
 * returning the loop body CF list and break `nir_if`. */
static bool
match_inclusive_range_loop_break(nir_loop *loop, nir_def *iv, nir_def *hi,
                                 const uint32_t wg_size[3],
                                 nir_if **break_if_out,
                                 struct exec_list **body_list_out)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_cf_node *next_node = nir_cf_node_next(&header->cf_node);
   if (!next_node || next_node->type != nir_cf_node_if)
      return false;

   nir_if *nif = nir_cf_node_as_if(next_node);
   bool break_in_then = cf_list_is_single_break(&nif->then_list);
   bool break_in_else = cf_list_is_single_break(&nif->else_list);
   if (break_in_then == break_in_else)
      return false;

   /* Break condition must hold iff `iv > hi` (i.e. `hi < iv` or `iv >= hi + 1`). */
   struct bool_term t = {.def = nif->condition.ssa, .negated = break_in_else};
   while (nir_def_instr(t.def)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(t.def))->op == nir_op_inot) {
      t.def = nir_instr_as_alu(nir_def_instr(t.def))->src[0].src.ssa;
      t.negated = !t.negated;
   }

   nir_def *lhs = NULL, *rhs = NULL;
   bool is_lt = false, is_u = false;
   if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u)
      return false;

   bool cond_ok = false;
   if (is_lt && same_val(lhs, hi, wg_size) && same_val(rhs, iv, wg_size)) {
      /* `hi < iv` */
      cond_ok = true;
   } else if (!is_lt && same_val(lhs, iv, wg_size)) {
      /* `iv >= hi + 1` */
      if (nir_def_instr(rhs)->type == nir_instr_type_alu) {
         nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(rhs));
         if (add->op == nir_op_iadd) {
            for (unsigned side = 0; side < 2; side++) {
               if (same_val(add->src[side].src.ssa, hi, wg_size) &&
                   is_const_i32(add->src[1 - side].src.ssa, 1))
                  cond_ok = true;
            }
         }
      }
   }
   if (!cond_ok)
      return false;

   if (break_in_then) {
      if (!cf_list_is_empty_block(&nif->else_list))
         return false;
      *body_list_out = &loop->body;
   } else {
      /* Ensure nothing follows `nif` in `loop->body` except an empty block. */
      nir_cf_node *after = nir_cf_node_next(&nif->cf_node);
      if (!after || after->type != nir_cf_node_block ||
          nir_cf_node_next(after) != NULL ||
          !exec_list_is_empty(&nir_cf_node_as_block(after)->instr_list))
         return false;
      *body_list_out = &nif->then_list;
   }

   *break_if_out = nif;
   return true;
}

static nir_loop *
find_single_loop_in_list(struct exec_list *cf_list)
{
   nir_loop *found = NULL;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_loop) {
         if (found)
            return NULL;
         found = nir_cf_node_as_loop(node);
      }
   }
   return found;
}

/* Match `val = (2 * r + 1) * (2 * r + 1)`. */
static bool
match_box_area(nir_def *def, nir_def *r_pc, const uint32_t wg_size[3])
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(def));
   if (mul->op != nir_op_imul && mul->op != nir_op_amul)
      return false;

   nir_def *d0 = strip_mov(mul->src[0].src.ssa);
   nir_def *d1 = strip_mov(mul->src[1].src.ssa);
   if (!d0 || !d1 || d0 != d1) {
      /* Also allow two separate identical `2*r + 1` expressions. */
   }

   nir_def *sides[2] = {d0, d1};
   for (unsigned s = 0; s < 2; s++) {
      nir_def *d = sides[s];
      if (!d || nir_def_instr(d)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(d));
      if (add->op != nir_op_iadd)
         return false;

      bool ok = false;
      for (unsigned side = 0; side < 2; side++) {
         if (!is_const_i32(add->src[side].src.ssa, 1))
            continue;
         nir_def *two_r = strip_mov(add->src[1 - side].src.ssa);
         if (!two_r || nir_def_instr(two_r)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *ta = nir_instr_as_alu(nir_def_instr(two_r));
         if (ta->op == nir_op_ishl &&
             same_val(ta->src[0].src.ssa, r_pc, wg_size) &&
             is_const_i32(ta->src[1].src.ssa, 1)) {
            ok = true;
         } else if ((ta->op == nir_op_imul || ta->op == nir_op_amul) &&
                    ((same_val(ta->src[0].src.ssa, r_pc, wg_size) &&
                      is_const_i32(ta->src[1].src.ssa, 2)) ||
                     (same_val(ta->src[1].src.ssa, r_pc, wg_size) &&
                      is_const_i32(ta->src[0].src.ssa, 2)))) {
            ok = true;
         } else if (ta->op == nir_op_iadd &&
                    same_val(ta->src[0].src.ssa, r_pc, wg_size) &&
                    same_val(ta->src[1].src.ssa, r_pc, wg_size)) {
            ok = true;
         }
      }
      if (!ok)
         return false;
   }
   return true;
}

/* Match `x >= r && x < w - r && y >= r && y < h - r` in any term order. */
static bool
match_box_interior_guard(nir_def *cond, bool negated, nir_def *x, nir_def *y,
                         nir_def *w_pc, nir_def *h_pc,
                         const uint32_t wg_size[3], nir_def **r_pc_out)
{
   struct bool_term terms[8];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 8) || n != 4)
      return false;

   nir_def *r_pc = NULL;
   bool got_x_lo = false, got_x_hi = false, got_y_lo = false, got_y_hi = false;

   /* First pass: locate `r_pc` from `x >= r` or `y >= r`. */
   for (unsigned i = 0; i < 4; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;
      uint16_t pc = 0;
      if (!is_lt &&
          (same_val(lhs, x, wg_size) || same_val(lhs, y, wg_size)) &&
          match_push_const_u32(rhs, &pc)) {
         r_pc = rhs;
         break;
      }
   }
   if (!r_pc)
      return false;

   for (unsigned i = 0; i < 4; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;
      if (!is_lt && same_val(lhs, x, wg_size) && same_val(rhs, r_pc, wg_size)) {
         got_x_lo = true;
      } else if (!is_lt && same_val(lhs, y, wg_size) &&
                 same_val(rhs, r_pc, wg_size)) {
         got_y_lo = true;
      } else if (is_lt && same_val(lhs, x, wg_size) &&
                 match_isub_pair(rhs, w_pc, r_pc, wg_size)) {
         got_x_hi = true;
      } else if (is_lt && same_val(lhs, y, wg_size) &&
                 match_isub_pair(rhs, h_pc, r_pc, wg_size)) {
         got_y_hi = true;
      } else {
         return false;
      }
   }

   if (!got_x_lo || !got_x_hi || !got_y_lo || !got_y_hi)
      return false;

   *r_pc_out = r_pc;
   return true;
}

/* Match `val = (uint8_t)(u2u32_sum / area)` where `area = (2*r + 1)^2`. */
static bool
match_box_store_val(nir_def *val, nir_def *sum_def, nir_def *r_pc,
                    const uint32_t wg_size[3])
{
   val = strip_mov(val);
   if (!val || val->bit_size != 8 || val->num_components != 1)
      return false;
   if (nir_def_instr(val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *trunc = nir_instr_as_alu(nir_def_instr(val));
   if (trunc->op != nir_op_u2u8 && trunc->op != nir_op_i2i8)
      return false;

   nir_def *q = strip_mov(trunc->src[0].src.ssa);
   if (!q || nir_def_instr(q)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *div = nir_instr_as_alu(nir_def_instr(q));
   if (div->op != nir_op_idiv && div->op != nir_op_udiv)
      return false;
   if (strip_mov(div->src[0].src.ssa) != strip_mov(sum_def))
      return false;
   return match_box_area(div->src[1].src.ssa, r_pc, wg_size);
}

/* ---------------------------------------------------------------------------
 * Matcher 1: 2D Byte Box Blur (R8 -> R8, interior average, border copy)
 * --------------------------------------------------------------------------- */

struct box_blur_cf {
   nir_if *outer_if;
   nir_if *interior_if;
   nir_loop *outer_loop;
   nir_loop *inner_loop;
   nir_if *outer_break;
   nir_if *inner_break;
};

static bool
verify_box_blur_cf_list(struct exec_list *cf_list, const struct box_blur_cf *cf)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_jump)
               continue;
            nir_jump_instr *j = nir_instr_as_jump(instr);
            if (j->type == nir_jump_return && cf->outer_if &&
                (cf_list == &cf->outer_if->then_list ||
                 cf_list == &cf->outer_if->else_list))
               continue;
            if (j->type == nir_jump_break &&
                ((cf->outer_break &&
                  (cf_list == &cf->outer_break->then_list ||
                   cf_list == &cf->outer_break->else_list)) ||
                 (cf->inner_break &&
                  (cf_list == &cf->inner_break->then_list ||
                   cf_list == &cf->inner_break->else_list))))
               continue;
            return false;
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (nif != cf->outer_if && nif != cf->interior_if &&
             nif != cf->outer_break && nif != cf->inner_break)
            return false;
         if (!verify_box_blur_cf_list(&nif->then_list, cf) ||
             !verify_box_blur_cf_list(&nif->else_list, cf))
            return false;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (loop != cf->outer_loop && loop != cf->inner_loop)
            return false;
         if (!verify_box_blur_cf_list(&loop->body, cf))
            return false;
         break;
      }
      default:
         return false;
      }
   }
   return true;
}

static bool
verify_box_blur_completeness(nir_shader *s, const struct box_blur_cf *cf,
                             const nir_intrinsic_instr *int_load,
                             const nir_intrinsic_instr *int_store,
                             const nir_intrinsic_instr *border_load,
                             const nir_intrinsic_instr *border_store)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!verify_box_blur_cf_list(&impl->body, cf))
      return false;

   unsigned seen_loads = 0, seen_stores = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref:
            if (intr == int_load || intr == border_load) {
               seen_loads++;
               break;
            }
            return false;
         case nir_intrinsic_store_deref:
            if (intr == int_store || intr == border_store) {
               seen_stores++;
               break;
            }
            return false;
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            return false;
         }
      }
   }

   return seen_loads == 2 && seen_stores == 2;
}

static bool
match_box_blur_u8(nir_shader *s, uint32_t set_layout_count,
                  struct vk_descriptor_set_layout *const *set_layouts,
                  struct hk_2d_stencil_plan *plan)
{
   const uint32_t *wg = plan->orig_wg_size;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   /* Find outer bounds guard: either
    *   `if (x < w && y < h) { <body_list> }` (empty else)
    * or
    *   `if (x >= w || y >= h) return; <body_list>` */
   nir_if *outer_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if) {
         outer_if = nir_cf_node_as_if(node);
         break;
      }
      if (node->type != nir_cf_node_block)
         return false;
   }
   if (!outer_if)
      return false;

   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   struct exec_list *in_bounds_list = NULL;
   nir_cf_node *start_after = NULL;

   if (cf_list_is_empty_block(&outer_if->else_list) &&
       match_xy_bounds_cond(outer_if->condition.ssa, false, wg, &x, &y, &w_pc,
                            &h_pc)) {
      in_bounds_list = &outer_if->then_list;
   } else if (cf_list_is_empty_block(&outer_if->then_list) &&
              match_xy_bounds_cond(outer_if->condition.ssa, true, wg, &x, &y,
                                   &w_pc, &h_pc)) {
      in_bounds_list = &outer_if->else_list;
   } else if (cf_list_is_single_return(&outer_if->then_list) &&
              cf_list_is_empty_block(&outer_if->else_list) &&
              match_xy_bounds_cond(outer_if->condition.ssa, true, wg, &x, &y,
                                   &w_pc, &h_pc)) {
      in_bounds_list = &impl->body;
      start_after = &outer_if->cf_node;
   } else if (cf_list_is_single_return(&outer_if->else_list) &&
              cf_list_is_empty_block(&outer_if->then_list) &&
              match_xy_bounds_cond(outer_if->condition.ssa, false, wg, &x, &y,
                                   &w_pc, &h_pc)) {
      in_bounds_list = &impl->body;
      start_after = &outer_if->cf_node;
   } else {
      return false;
   }

   /* Locate the interior/border `if` inside `in_bounds_list`. */
   nir_if *interior_if = NULL;
   bool past_start = (start_after == NULL);
   foreach_list_typed(nir_cf_node, node, node, in_bounds_list) {
      if (!past_start) {
         if (node == start_after)
            past_start = true;
         continue;
      }
      if (node->type == nir_cf_node_if) {
         if (interior_if)
            return false;
         interior_if = nir_cf_node_as_if(node);
      } else if (node->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!interior_if)
      return false;

   nir_def *r_pc = NULL;
   struct exec_list *int_list = NULL;
   struct exec_list *border_list = NULL;
   if (match_box_interior_guard(interior_if->condition.ssa, false, x, y, w_pc,
                                h_pc, wg, &r_pc)) {
      int_list = &interior_if->then_list;
      border_list = &interior_if->else_list;
   } else if (match_box_interior_guard(interior_if->condition.ssa, true, x, y,
                                       w_pc, h_pc, wg, &r_pc)) {
      int_list = &interior_if->else_list;
      border_list = &interior_if->then_list;
   } else {
      return false;
   }

   /* Match nested 2D loops inside `int_list`. */
   nir_loop *outer_loop = find_single_loop_in_list(int_list);
   if (!outer_loop)
      return false;

   nir_block *o_hdr = nir_loop_first_block(outer_loop);
   nir_phi_instr *o_iv_phi = NULL, *o_acc_phi = NULL;
   nir_def *o_iv_init = NULL, *o_iv_next = NULL;
   nir_def *o_acc_next = NULL;

   nir_foreach_phi(phi, o_hdr) {
      nir_def *init = NULL, *next = NULL;
      if (phi->def.bit_size != 32 || phi->def.num_components != 1 ||
          !match_loop_phi(outer_loop, phi, &init, &next))
         return false;
      if (is_const_i32(init, 0)) {
         if (o_acc_phi)
            return false;
         o_acc_phi = phi;
         o_acc_next = next;
      } else {
         if (o_iv_phi)
            return false;
         o_iv_phi = phi;
         o_iv_init = init;
         o_iv_next = next;
      }
   }
   if (!o_iv_phi || !o_acc_phi)
      return false;

   /* Determine whether outer loop walks Y or X. */
   bool outer_is_y = false;
   if (match_isub_pair(o_iv_init, y, r_pc, wg)) {
      outer_is_y = true;
   } else if (match_isub_pair(o_iv_init, x, r_pc, wg)) {
      outer_is_y = false;
   } else {
      return false;
   }

   /* Outer IV step must be `o_iv + 1`. */
   {
      nir_def *step = strip_mov(o_iv_next);
      if (!step || nir_def_instr(step)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(step));
      if (sa->op != nir_op_iadd)
         return false;
      bool step_ok =
         (strip_mov(sa->src[0].src.ssa) == &o_iv_phi->def &&
          is_const_i32(sa->src[1].src.ssa, 1)) ||
         (strip_mov(sa->src[1].src.ssa) == &o_iv_phi->def &&
          is_const_i32(sa->src[0].src.ssa, 1));
      if (!step_ok)
         return false;
   }

   nir_def *o_coord = outer_is_y ? y : x;
   nir_def *i_coord = outer_is_y ? x : y;

   /* Match outer loop break on `o_iv > o_coord + r`. */
   nir_if *outer_break = NULL;
   struct exec_list *outer_body = NULL;
   {
      /* Extract `hi` from break condition and check it equals `o_coord + r`. */
      nir_cf_node *nn = nir_cf_node_next(&o_hdr->cf_node);
      if (!nn || nn->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(nn);
      bool break_then = cf_list_is_single_break(&nif->then_list);
      bool break_else = cf_list_is_single_break(&nif->else_list);
      if (break_then == break_else)
         return false;
      struct bool_term t = {.def = nif->condition.ssa, .negated = break_else};
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;
      nir_def *o_hi = is_lt ? lhs : rhs;
      if (!match_iadd_pair(o_hi, o_coord, r_pc, wg))
         return false;
      if (!match_inclusive_range_loop_break(outer_loop, &o_iv_phi->def, o_hi,
                                            wg, &outer_break, &outer_body))
         return false;
   }

   /* Match inner loop inside `outer_body`. */
   nir_loop *inner_loop = find_single_loop_in_list(outer_body);
   if (!inner_loop)
      return false;

   nir_block *i_hdr = nir_loop_first_block(inner_loop);
   nir_phi_instr *i_iv_phi = NULL, *i_acc_phi = NULL;
   nir_def *i_iv_init = NULL, *i_iv_next = NULL;
   nir_def *i_acc_next = NULL;

   nir_foreach_phi(phi, i_hdr) {
      nir_def *init = NULL, *next = NULL;
      if (phi->def.bit_size != 32 || phi->def.num_components != 1 ||
          !match_loop_phi(inner_loop, phi, &init, &next))
         return false;
      if (strip_mov(init) == &o_acc_phi->def) {
         if (i_acc_phi)
            return false;
         i_acc_phi = phi;
         i_acc_next = next;
      } else {
         if (i_iv_phi)
            return false;
         i_iv_phi = phi;
         i_iv_init = init;
         i_iv_next = next;
      }
   }
   if (!i_iv_phi || !i_acc_phi)
      return false;

   /* Outer accumulator must receive inner accumulator phi on loop exit. */
   if (strip_mov(o_acc_next) != &i_acc_phi->def)
      return false;

   if (!match_isub_pair(i_iv_init, i_coord, r_pc, wg))
      return false;

   /* Inner IV step must be `i_iv + 1`. */
   {
      nir_def *step = strip_mov(i_iv_next);
      if (!step || nir_def_instr(step)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(step));
      if (sa->op != nir_op_iadd)
         return false;
      bool step_ok =
         (strip_mov(sa->src[0].src.ssa) == &i_iv_phi->def &&
          is_const_i32(sa->src[1].src.ssa, 1)) ||
         (strip_mov(sa->src[1].src.ssa) == &i_iv_phi->def &&
          is_const_i32(sa->src[0].src.ssa, 1));
      if (!step_ok)
         return false;
   }

   /* Match inner loop break on `i_iv > i_coord + r`. */
   nir_if *inner_break = NULL;
   struct exec_list *inner_body = NULL;
   {
      nir_cf_node *nn = nir_cf_node_next(&i_hdr->cf_node);
      if (!nn || nn->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(nn);
      bool break_then = cf_list_is_single_break(&nif->then_list);
      bool break_else = cf_list_is_single_break(&nif->else_list);
      if (break_then == break_else)
         return false;
      struct bool_term t = {.def = nif->condition.ssa, .negated = break_else};
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;
      nir_def *i_hi = is_lt ? lhs : rhs;
      if (!match_iadd_pair(i_hi, i_coord, r_pc, wg))
         return false;
      if (!match_inclusive_range_loop_break(inner_loop, &i_iv_phi->def, i_hi,
                                            wg, &inner_break, &inner_body))
         return false;
   }

   /* Inner accumulation must be `i_acc + u2u32(src[yy * stride + xx])`. */
   nir_def *add_def = strip_mov(i_acc_next);
   if (!add_def || nir_def_instr(add_def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *acc_add = nir_instr_as_alu(nir_def_instr(add_def));
   if (acc_add->op != nir_op_iadd)
      return false;

   nir_def *term = NULL;
   if (strip_mov(acc_add->src[0].src.ssa) == &i_acc_phi->def)
      term = strip_mov(acc_add->src[1].src.ssa);
   else if (strip_mov(acc_add->src[1].src.ssa) == &i_acc_phi->def)
      term = strip_mov(acc_add->src[0].src.ssa);
   else
      return false;

   if (!term || nir_def_instr(term)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *uext = nir_instr_as_alu(nir_def_instr(term));
   if (uext->op != nir_op_u2u32)
      return false;

   nir_def *loaded_u8 = strip_mov(uext->src[0].src.ssa);
   if (!loaded_u8 || nir_def_instr(loaded_u8)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *int_load =
      nir_instr_as_intrinsic(nir_def_instr(loaded_u8));
   if (int_load->intrinsic != nir_intrinsic_load_deref)
      return false;
   if (nir_intrinsic_access(int_load) &
       (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
      return false;

   uint32_t src_set = 0, src_bind = 0;
   nir_def *int_load_idx = NULL;
   if (!match_ssbo_deref_access(nir_src_as_deref(int_load->src[0]), 8, &src_set,
                                &src_bind, &int_load_idx))
      return false;

   nir_def *yy_iv = outer_is_y ? &o_iv_phi->def : &i_iv_phi->def;
   nir_def *xx_iv = outer_is_y ? &i_iv_phi->def : &o_iv_phi->def;
   uint16_t stride_pc = 0;
   if (!match_linear_index(int_load_idx, yy_iv, xx_iv, &stride_pc, wg))
      return false;

   /* Find interior store after `outer_loop`. */
   nir_cf_node *post_loop_node = nir_cf_node_next(&outer_loop->cf_node);
   if (!post_loop_node || post_loop_node->type != nir_cf_node_block ||
       nir_cf_node_next(post_loop_node) != NULL)
      return false;
   nir_block *post_loop_blk = nir_cf_node_as_block(post_loop_node);

   nir_intrinsic_instr *int_store = NULL;
   nir_foreach_instr(instr, post_loop_blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_store_deref) {
         if (int_store)
            return false;
         int_store = intr;
      }
   }
   if (!int_store || nir_intrinsic_write_mask(int_store) != 1 ||
       (nir_intrinsic_access(int_store) &
        (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL)))
      return false;

   uint32_t dst_set = 0, dst_bind = 0;
   nir_def *int_store_idx = NULL;
   if (!match_ssbo_deref_access(nir_src_as_deref(int_store->src[0]), 8, &dst_set,
                                &dst_bind, &int_store_idx))
      return false;
   uint16_t dst_stride_pc = 0;
   if (!match_linear_index(int_store_idx, y, x, &dst_stride_pc, wg) ||
       dst_stride_pc != stride_pc)
      return false;
   if (!match_box_store_val(int_store->src[1].ssa, &o_acc_phi->def, r_pc, wg))
      return false;

   /* Match border copy in `border_list`: single block loading src[y*stride+x]
    * and storing to dst[y*stride+x]. */
   if (!exec_list_is_singular(border_list))
      return false;
   nir_cf_node *bnode =
      exec_node_data(nir_cf_node, exec_list_get_head(border_list), node);
   if (bnode->type != nir_cf_node_block)
      return false;
   nir_block *border_blk = nir_cf_node_as_block(bnode);

   nir_intrinsic_instr *border_load = NULL, *border_store = NULL;
   nir_foreach_instr(instr, border_blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_deref) {
         if (border_load)
            return false;
         border_load = intr;
      } else if (intr->intrinsic == nir_intrinsic_store_deref) {
         if (border_store)
            return false;
         border_store = intr;
      }
   }
   if (!border_load || !border_store ||
       nir_intrinsic_write_mask(border_store) != 1 ||
       strip_mov(border_store->src[1].ssa) != &border_load->def)
      return false;
   if ((nir_intrinsic_access(border_load) | nir_intrinsic_access(border_store)) &
       (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
      return false;

   uint32_t bl_set = 0, bl_bind = 0, bs_set = 0, bs_bind = 0;
   nir_def *bl_idx = NULL, *bs_idx = NULL;
   uint16_t bl_stride = 0, bs_stride = 0;
   if (!match_ssbo_deref_access(nir_src_as_deref(border_load->src[0]), 8,
                                &bl_set, &bl_bind, &bl_idx) ||
       !match_ssbo_deref_access(nir_src_as_deref(border_store->src[0]), 8,
                                &bs_set, &bs_bind, &bs_idx))
      return false;
   if (bl_set != src_set || bl_bind != src_bind || bs_set != dst_set ||
       bs_bind != dst_bind)
      return false;
   if (!match_linear_index(bl_idx, y, x, &bl_stride, wg) ||
       !match_linear_index(bs_idx, y, x, &bs_stride, wg) ||
       bl_stride != stride_pc || bs_stride != stride_pc)
      return false;

   if (src_set == dst_set && src_bind == dst_bind)
      return false;

   const struct box_blur_cf cf = {
      .outer_if = outer_if,
      .interior_if = interior_if,
      .outer_loop = outer_loop,
      .inner_loop = inner_loop,
      .outer_break = outer_break,
      .inner_break = inner_break,
   };
   if (!verify_box_blur_completeness(s, &cf, int_load, int_store, border_load,
                                     border_store))
      return false;

   uint16_t w_off = 0, h_off = 0, r_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off) ||
       !match_push_const_u32(r_pc, &r_off))
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[1]))
      return false;

   plan->kind = HK_2D_STENCIL_BOX_BLUR_U8;
   plan->tile_w = HK_BOX_BLUR_STRIP_W;
   plan->tile_h = HK_BOX_BLUR_TILE_H;
   plan->width_pc_offset_B = w_off;
   plan->height_pc_offset_B = h_off;
   plan->stride_pc_offset_B = stride_pc;
   plan->radius_pc_offset_B = r_off;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 2: 2D Stereo SAD Block Matching (R8 left/right -> disp u8 + cost u32)
 * --------------------------------------------------------------------------- */

static bool
match_iv_plus_one(nir_def *next, nir_def *iv)
{
   nir_def *step = strip_mov(next);
   if (!step || nir_def_instr(step)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(step));
   if (sa->op != nir_op_iadd)
      return false;
   return (strip_mov(sa->src[0].src.ssa) == strip_mov(iv) &&
           is_const_i32(sa->src[1].src.ssa, 1)) ||
          (strip_mov(sa->src[1].src.ssa) == strip_mov(iv) &&
           is_const_i32(sa->src[0].src.ssa, 1));
}

static bool
match_iabs_of(nir_def *def, nir_def **inner_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iabs) {
      *inner_out = strip_mov(alu->src[0].src.ssa);
      return true;
   }
   if (alu->op == nir_op_imax) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *a = strip_mov(alu->src[side].src.ssa);
         nir_def *b = strip_mov(alu->src[1 - side].src.ssa);
         if (b && nir_def_instr(b)->type == nir_instr_type_alu) {
            nir_alu_instr *na = nir_instr_as_alu(nir_def_instr(b));
            if (na->op == nir_op_ineg && strip_mov(na->src[0].src.ssa) == a) {
               *inner_out = a;
               return true;
            }
         }
      }
   }
   return false;
}

static bool
match_isign_of_pc(nir_def *def, uint16_t pc_off)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_isign) {
      uint16_t o = 0;
      return match_push_const_u32(alu->src[0].src.ssa, &o) && o == pc_off;
   }
   /* imin(imax(pc, -1), 1) or imax(imin(pc, 1), -1) */
   if (alu->op == nir_op_imin || alu->op == nir_op_imax) {
      int32_t outer_c = (alu->op == nir_op_imin) ? 1 : -1;
      nir_op inner_op = (alu->op == nir_op_imin) ? nir_op_imax : nir_op_imin;
      int32_t inner_c = -outer_c;
      for (unsigned side = 0; side < 2; side++) {
         if (!is_const_i32(alu->src[side].src.ssa, outer_c))
            continue;
         nir_def *in = strip_mov(alu->src[1 - side].src.ssa);
         if (!in || nir_def_instr(in)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *ia = nir_instr_as_alu(nir_def_instr(in));
         if (ia->op != inner_op)
            continue;
         for (unsigned is = 0; is < 2; is++) {
            uint16_t o = 0;
            if (is_const_i32(ia->src[is].src.ssa, inner_c) &&
                match_push_const_u32(ia->src[1 - is].src.ssa, &o) &&
                o == pc_off)
               return true;
         }
      }
   }
   return false;
}

/* Match `r = window_size / 2` (either direct `idiv/udiv/ishr/ushr` or
 * algebraically lowered signed `idiv(ws, 2) = isign(ws) * (iabs(ws) >> 1)`). */
static bool
match_window_half_pc(nir_def *def, uint16_t *ws_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if ((alu->op == nir_op_idiv || alu->op == nir_op_udiv) &&
       is_const_i32(alu->src[1].src.ssa, 2))
      return match_push_const_u32(alu->src[0].src.ssa, ws_pc_out);

   if ((alu->op == nir_op_ishr || alu->op == nir_op_ushr) &&
       is_const_i32(alu->src[1].src.ssa, 1))
      return match_push_const_u32(alu->src[0].src.ssa, ws_pc_out);

   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *sh = strip_mov(alu->src[side].src.ssa);
         if (!sh || nir_def_instr(sh)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(sh));
         if ((sa->op != nir_op_ushr && sa->op != nir_op_ishr) ||
             !is_const_i32(sa->src[1].src.ssa, 1))
            continue;
         nir_def *inner = NULL;
         uint16_t pc_off = 0;
         if (match_iabs_of(sa->src[0].src.ssa, &inner) &&
             match_push_const_u32(inner, &pc_off) &&
             match_isign_of_pc(alu->src[1 - side].src.ssa, pc_off)) {
            *ws_pc_out = pc_off;
            return true;
         }
      }
   }

   return false;
}

/* Match `clamp(coord + sign * half_ws, 0, dim - 1)` where `sign` is +1 or -1. */
static bool
match_clamped_window_bound(nir_def *def, nir_def *coord, nir_def *dim_pc,
                           bool is_hi, const uint32_t wg_size[3],
                           uint16_t *ws_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *outer = nir_instr_as_alu(nir_def_instr(def));
   if (outer->op != nir_op_imin && outer->op != nir_op_imax)
      return false;

   nir_def *lo_val = NULL, *hi_val = NULL, *mid = NULL;
   if (outer->op == nir_op_imin) {
      /* imin(imax(expr, 0), dim - 1) */
      for (unsigned side = 0; side < 2; side++) {
         nir_def *in = strip_mov(outer->src[side].src.ssa);
         if (in && nir_def_instr(in)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(in))->op == nir_op_imax) {
            hi_val = outer->src[1 - side].src.ssa;
            nir_alu_instr *ia = nir_instr_as_alu(nir_def_instr(in));
            for (unsigned is = 0; is < 2; is++) {
               if (is_const_i32(ia->src[is].src.ssa, 0)) {
                  lo_val = ia->src[is].src.ssa;
                  mid = ia->src[1 - is].src.ssa;
                  break;
               }
            }
            if (mid)
               break;
         }
      }
   } else {
      /* imax(imin(expr, dim - 1), 0) */
      for (unsigned side = 0; side < 2; side++) {
         if (!is_const_i32(outer->src[side].src.ssa, 0))
            continue;
         lo_val = outer->src[side].src.ssa;
         nir_def *in = strip_mov(outer->src[1 - side].src.ssa);
         if (in && nir_def_instr(in)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(in))->op == nir_op_imin) {
            nir_alu_instr *ia = nir_instr_as_alu(nir_def_instr(in));
            for (unsigned is = 0; is < 2; is++) {
               nir_def *cand_hi = ia->src[is].src.ssa;
               nir_def *cand_mid = ia->src[1 - is].src.ssa;
               nir_def *ch = strip_mov(cand_hi);
               if (ch && nir_def_instr(ch)->type == nir_instr_type_alu) {
                  nir_alu_instr *ha = nir_instr_as_alu(nir_def_instr(ch));
                  bool is_dim_m1 =
                     (ha->op == nir_op_iadd &&
                      ((same_val(ha->src[0].src.ssa, dim_pc, wg_size) &&
                        is_const_i32(ha->src[1].src.ssa, -1)) ||
                       (same_val(ha->src[1].src.ssa, dim_pc, wg_size) &&
                        is_const_i32(ha->src[0].src.ssa, -1)))) ||
                     (ha->op == nir_op_isub &&
                      same_val(ha->src[0].src.ssa, dim_pc, wg_size) &&
                      is_const_i32(ha->src[1].src.ssa, 1));
                  if (is_dim_m1) {
                     hi_val = cand_hi;
                     mid = cand_mid;
                     break;
                  }
               }
            }
         }
      }
   }
   if (!lo_val || !hi_val || !mid)
      return false;

   /* Verify `hi_val == dim_pc - 1`. */
   nir_def *hv = strip_mov(hi_val);
   if (!hv || nir_def_instr(hv)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ha = nir_instr_as_alu(nir_def_instr(hv));
   bool hi_ok =
      (ha->op == nir_op_iadd &&
       ((same_val(ha->src[0].src.ssa, dim_pc, wg_size) &&
         is_const_i32(ha->src[1].src.ssa, -1)) ||
        (same_val(ha->src[1].src.ssa, dim_pc, wg_size) &&
         is_const_i32(ha->src[0].src.ssa, -1)))) ||
      (ha->op == nir_op_isub && same_val(ha->src[0].src.ssa, dim_pc, wg_size) &&
       is_const_i32(ha->src[1].src.ssa, 1));
   if (!hi_ok)
      return false;

   /* Verify `mid == coord + half_ws` (if `is_hi`) or `coord - half_ws`. */
   mid = strip_mov(mid);
   if (!mid || nir_def_instr(mid)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ma = nir_instr_as_alu(nir_def_instr(mid));
   if (is_hi) {
      if (ma->op != nir_op_iadd)
         return false;
      for (unsigned side = 0; side < 2; side++) {
         if (same_val(ma->src[side].src.ssa, coord, wg_size) &&
             match_window_half_pc(ma->src[1 - side].src.ssa, ws_pc_out))
            return true;
      }
      return false;
   } else {
      if (ma->op == nir_op_isub) {
         return same_val(ma->src[0].src.ssa, coord, wg_size) &&
                match_window_half_pc(ma->src[1].src.ssa, ws_pc_out);
      }
      if (ma->op == nir_op_iadd) {
         for (unsigned side = 0; side < 2; side++) {
            if (!same_val(ma->src[side].src.ssa, coord, wg_size))
               continue;
            nir_def *neg = strip_mov(ma->src[1 - side].src.ssa);
            if (neg && nir_def_instr(neg)->type == nir_instr_type_alu &&
                nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg &&
                match_window_half_pc(
                   nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa,
                   ws_pc_out))
               return true;
         }
      }
      return false;
   }
}

/* Match `x0 - d >= 0 && x1 - d < width` returning `x0` and `x1`. */
static bool
match_stereo_disp_guard(nir_def *cond, bool negated, nir_def *d_iv,
                        nir_def *w_pc, const uint32_t wg_size[3],
                        nir_def **x0_out, nir_def **x1_out)
{
   struct bool_term terms[4];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 4) || n != 2)
      return false;

   nir_def *x0 = NULL, *x1 = NULL;
   for (unsigned i = 0; i < 2; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;

      if (!is_lt) {
         /* `x0 - d >= 0` or `x0 >= d` */
         if (same_val(rhs, d_iv, wg_size)) {
            x0 = lhs;
         } else if (is_const_i32(rhs, 0)) {
            nir_def *sub = strip_mov(lhs);
            if (!sub || nir_def_instr(sub)->type != nir_instr_type_alu)
               return false;
            nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(sub));
            if (sa->op == nir_op_isub &&
                same_val(sa->src[1].src.ssa, d_iv, wg_size)) {
               x0 = sa->src[0].src.ssa;
            } else if (sa->op == nir_op_iadd) {
               for (unsigned s = 0; s < 2; s++) {
                  nir_def *neg = strip_mov(sa->src[s].src.ssa);
                  if (neg && nir_def_instr(neg)->type == nir_instr_type_alu &&
                      nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg &&
                      same_val(nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa,
                               d_iv, wg_size)) {
                     x0 = sa->src[1 - s].src.ssa;
                     break;
                  }
               }
            }
         }
      } else {
         /* `x1 - d < width` */
         if (same_val(rhs, w_pc, wg_size)) {
            nir_def *sub = strip_mov(lhs);
            if (!sub || nir_def_instr(sub)->type != nir_instr_type_alu)
               return false;
            nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(sub));
            if (sa->op == nir_op_isub &&
                same_val(sa->src[1].src.ssa, d_iv, wg_size)) {
               x1 = sa->src[0].src.ssa;
            } else if (sa->op == nir_op_iadd) {
               for (unsigned s = 0; s < 2; s++) {
                  nir_def *neg = strip_mov(sa->src[s].src.ssa);
                  if (neg && nir_def_instr(neg)->type == nir_instr_type_alu &&
                      nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg &&
                      same_val(nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa,
                               d_iv, wg_size)) {
                     x1 = sa->src[1 - s].src.ssa;
                     break;
                  }
               }
            }
         }
      }
   }

   if (!x0 || !x1)
      return false;
   *x0_out = x0;
   *x1_out = x1;
   return true;
}

/* Match `val = (yy * stride + xx) - d` or `yy * stride + (xx - d)`. */
static bool
match_shifted_linear_index(nir_def *idx, nir_def *yy, nir_def *xx, nir_def *d,
                           uint16_t stride_pc, const uint32_t wg_size[3])
{
   idx = strip_mov(idx);
   if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(idx));

   /* Form 1: `base_idx - d` */
   if (alu->op == nir_op_isub && same_val(alu->src[1].src.ssa, d, wg_size)) {
      uint16_t sp = 0;
      return match_linear_index(alu->src[0].src.ssa, yy, xx, &sp, wg_size) &&
             sp == stride_pc;
   }
   /* Form 2: `base_idx + (-d)` OR `yy * stride + (xx - d)` */
   if (alu->op == nir_op_iadd) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *other = strip_mov(alu->src[1 - side].src.ssa);
         if (other && nir_def_instr(other)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(other))->op == nir_op_ineg &&
             same_val(nir_instr_as_alu(nir_def_instr(other))->src[0].src.ssa, d,
                      wg_size)) {
            uint16_t sp = 0;
            if (match_linear_index(alu->src[side].src.ssa, yy, xx, &sp,
                                   wg_size) &&
                sp == stride_pc)
               return true;
         }
         /* `yy * stride + (xx - d)` */
         if (match_isub_pair(alu->src[side].src.ssa, xx, d, wg_size)) {
            nir_def *prod = strip_mov(alu->src[1 - side].src.ssa);
            if (prod && nir_def_instr(prod)->type == nir_instr_type_alu) {
               nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
               if (mul->op == nir_op_imul || mul->op == nir_op_amul) {
                  for (unsigned ms = 0; ms < 2; ms++) {
                     uint16_t sp = 0;
                     if (same_val(mul->src[ms].src.ssa, yy, wg_size) &&
                         match_push_const_u32(mul->src[1 - ms].src.ssa, &sp) &&
                         sp == stride_pc)
                        return true;
                  }
               }
            }
         }
      }
   }
   return false;
}

struct stereo_sad_cf {
   nir_if *outer_if;
   nir_if *d_guard_if; /* optional rotated `if (d_max >= d_min)` */
   nir_loop *d_loop;
   nir_if *d_break;
   nir_if *disp_if;
   nir_loop *win_outer_loop;
   nir_if *win_outer_break;
   nir_loop *win_inner_loop;
   nir_if *win_inner_break;
};

static bool
verify_stereo_cf_list(struct exec_list *cf_list, const struct stereo_sad_cf *cf)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_jump)
               continue;
            nir_jump_instr *j = nir_instr_as_jump(instr);
            if (j->type == nir_jump_return && cf->outer_if &&
                (cf_list == &cf->outer_if->then_list ||
                 cf_list == &cf->outer_if->else_list))
               continue;
            if (j->type == nir_jump_break &&
                ((cf->d_break &&
                  (cf_list == &cf->d_break->then_list ||
                   cf_list == &cf->d_break->else_list)) ||
                 (cf->win_outer_break &&
                  (cf_list == &cf->win_outer_break->then_list ||
                   cf_list == &cf->win_outer_break->else_list)) ||
                 (cf->win_inner_break &&
                  (cf_list == &cf->win_inner_break->then_list ||
                   cf_list == &cf->win_inner_break->else_list))))
               continue;
            return false;
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (nif != cf->outer_if && nif != cf->d_guard_if &&
             nif != cf->d_break && nif != cf->disp_if &&
             nif != cf->win_outer_break && nif != cf->win_inner_break)
            return false;
         if (!verify_stereo_cf_list(&nif->then_list, cf) ||
             !verify_stereo_cf_list(&nif->else_list, cf))
            return false;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (loop != cf->d_loop && loop != cf->win_outer_loop &&
             loop != cf->win_inner_loop)
            return false;
         if (!verify_stereo_cf_list(&loop->body, cf))
            return false;
         break;
      }
      default:
         return false;
      }
   }
   return true;
}

static bool
verify_stereo_completeness(nir_shader *s, const struct stereo_sad_cf *cf,
                           const nir_intrinsic_instr *prev_cost_ld,
                           const nir_intrinsic_instr *prev_disp_ld,
                           const nir_intrinsic_instr *left_ld,
                           const nir_intrinsic_instr *right_ld,
                           const nir_intrinsic_instr *dst_disp_st,
                           const nir_intrinsic_instr *dst_cost_st)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!verify_stereo_cf_list(&impl->body, cf))
      return false;

   unsigned seen_loads = 0, seen_stores = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref:
            if (intr == prev_cost_ld || intr == prev_disp_ld ||
                intr == left_ld || intr == right_ld) {
               seen_loads++;
               break;
            }
            return false;
         case nir_intrinsic_store_deref:
            if (intr == dst_disp_st || intr == dst_cost_st) {
               seen_stores++;
               break;
            }
            return false;
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            return false;
         }
      }
   }

   return seen_loads == 4 && seen_stores == 2;
}

static bool
match_stereo_sad_u8(nir_shader *s, uint32_t set_layout_count,
                    struct vk_descriptor_set_layout *const *set_layouts,
                    struct hk_2d_stencil_plan *plan)
{
   const uint32_t *wg = plan->orig_wg_size;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   nir_if *outer_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if) {
         outer_if = nir_cf_node_as_if(node);
         break;
      }
      if (node->type != nir_cf_node_block)
         return false;
   }
   if (!outer_if)
      return false;

   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   struct exec_list *in_bounds = NULL;
   nir_cf_node *start_after = NULL;

   if (cf_list_is_empty_block(&outer_if->else_list) &&
       match_xy_bounds_cond(outer_if->condition.ssa, false, wg, &x, &y, &w_pc,
                            &h_pc)) {
      in_bounds = &outer_if->then_list;
   } else if (cf_list_is_empty_block(&outer_if->then_list) &&
              match_xy_bounds_cond(outer_if->condition.ssa, true, wg, &x, &y,
                                   &w_pc, &h_pc)) {
      in_bounds = &outer_if->else_list;
   } else if (cf_list_is_single_return(&outer_if->then_list) &&
              cf_list_is_empty_block(&outer_if->else_list) &&
              match_xy_bounds_cond(outer_if->condition.ssa, true, wg, &x, &y,
                                   &w_pc, &h_pc)) {
      in_bounds = &impl->body;
      start_after = &outer_if->cf_node;
   } else if (cf_list_is_single_return(&outer_if->else_list) &&
              cf_list_is_empty_block(&outer_if->then_list) &&
              match_xy_bounds_cond(outer_if->condition.ssa, false, wg, &x, &y,
                                   &w_pc, &h_pc)) {
      in_bounds = &impl->body;
      start_after = &outer_if->cf_node;
   } else {
      return false;
   }

   /* Locate the `d` loop — either directly in `in_bounds` or wrapped in a
    * rotated `if (d_max >= d_min) { loop { ... } }` guard. */
   nir_if *d_guard_if = NULL;
   nir_loop *d_loop = NULL;
   nir_cf_node *d_container = NULL;
   bool past_start = (start_after == NULL);

   foreach_list_typed(nir_cf_node, node, node, in_bounds) {
      if (!past_start) {
         if (node == start_after)
            past_start = true;
         continue;
      }
      if (node->type == nir_cf_node_loop) {
         if (d_loop || d_guard_if)
            return false;
         d_loop = nir_cf_node_as_loop(node);
         d_container = node;
      } else if (node->type == nir_cf_node_if) {
         if (d_loop || d_guard_if)
            return false;
         d_guard_if = nir_cf_node_as_if(node);
         d_container = node;
      } else if (node->type != nir_cf_node_block) {
         return false;
      }
   }

   nir_def *guard_d_min = NULL, *guard_d_max = NULL;
   bool d_loop_rotated = false;
   if (d_guard_if) {
      struct exec_list *g_body = NULL;
      bool g_neg = false;
      if (cf_list_is_empty_block(&d_guard_if->else_list)) {
         g_body = &d_guard_if->then_list;
         g_neg = false;
      } else if (cf_list_is_empty_block(&d_guard_if->then_list)) {
         g_body = &d_guard_if->else_list;
         g_neg = true;
      } else {
         return false;
      }
      struct bool_term t = {.def = d_guard_if->condition.ssa, .negated = g_neg};
      while (nir_def_instr(t.def)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(t.def))->op == nir_op_inot) {
         t.def = nir_instr_as_alu(nir_def_instr(t.def))->src[0].src.ssa;
         t.negated = !t.negated;
      }
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u || is_lt)
         return false;
      /* `d_max >= d_min` */
      uint16_t max_pc = 0, min_pc = 0;
      if (!match_push_const_u32(lhs, &max_pc) ||
          !match_push_const_u32(rhs, &min_pc))
         return false;
      guard_d_max = lhs;
      guard_d_min = rhs;
      d_loop = find_single_loop_in_list(g_body);
      if (!d_loop)
         return false;
      d_loop_rotated = true;
   }
   if (!d_loop)
      return false;

   /* Find pre-loop `prev_cost` (32-bit) and `prev_disp` (8-bit) loads in block
    * preceding `d_container`. */
   nir_cf_node *pre_node = nir_cf_node_prev(d_container);
   if (!pre_node || pre_node->type != nir_cf_node_block)
      return false;
   nir_block *pre_blk = nir_cf_node_as_block(pre_node);

   nir_intrinsic_instr *prev_cost_ld = NULL, *prev_disp_ld = NULL;
   uint32_t pcost_set = 0, pcost_bind = 0, pdisp_set = 0, pdisp_bind = 0;
   uint16_t cost_stride_pc = 0, stride_pc = 0;
   nir_foreach_instr(instr, pre_blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_deref)
         continue;
      if (nir_intrinsic_access(intr) &
          (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
         return false;
      nir_deref_instr *drf = nir_src_as_deref(intr->src[0]);
      uint32_t set = 0, bind = 0;
      nir_def *idx = NULL;
      if (match_ssbo_deref_access(drf, 32, &set, &bind, &idx)) {
         if (prev_cost_ld ||
             !match_linear_index(idx, y, x, &cost_stride_pc, wg))
            return false;
         prev_cost_ld = intr;
         pcost_set = set;
         pcost_bind = bind;
      } else if (match_ssbo_deref_access(drf, 8, &set, &bind, &idx)) {
         if (prev_disp_ld || !match_linear_index(idx, y, x, &stride_pc, wg))
            return false;
         prev_disp_ld = intr;
         pdisp_set = set;
         pdisp_bind = bind;
      } else {
         return false;
      }
   }
   if (!prev_cost_ld || !prev_disp_ld)
      return false;

   /* Classify the 3 phis of `d_loop`: `d`, `best_disp`, `best_cost`. */
   nir_block *d_hdr = nir_loop_first_block(d_loop);
   nir_phi_instr *d_phi = NULL, *bd_phi = NULL, *bc_phi = NULL;
   nir_def *d_init = NULL, *d_next = NULL;
   nir_def *bd_init = NULL, *bd_next = NULL;
   nir_def *bc_init = NULL, *bc_next = NULL;

   nir_foreach_phi(phi, d_hdr) {
      nir_def *init = NULL, *next = NULL;
      if (phi->def.bit_size != 32 || phi->def.num_components != 1 ||
          !match_loop_phi(d_loop, phi, &init, &next))
         return false;
      uint16_t pc = 0;
      nir_def *s_init = strip_mov(init);
      if (s_init == &prev_cost_ld->def) {
         if (bc_phi)
            return false;
         bc_phi = phi;
         bc_init = init;
         bc_next = next;
      } else if (s_init && nir_def_instr(s_init)->type == nir_instr_type_alu &&
                 nir_instr_as_alu(nir_def_instr(s_init))->op == nir_op_u2u32 &&
                 strip_mov(nir_instr_as_alu(nir_def_instr(s_init))->src[0].src.ssa) ==
                    &prev_disp_ld->def) {
         if (bd_phi)
            return false;
         bd_phi = phi;
         bd_init = init;
         bd_next = next;
      } else if (match_push_const_u32(init, &pc)) {
         if (d_phi)
            return false;
         d_phi = phi;
         d_init = init;
         d_next = next;
      } else {
         return false;
      }
   }
   if (!d_phi || !bd_phi || !bc_phi)
      return false;
   if (!match_iv_plus_one(d_next, &d_phi->def))
      return false;
   if (d_loop_rotated && !same_val(d_init, guard_d_min, wg))
      return false;

   /* Identify `d_break` and `disp_if` inside `d_loop->body`. */
   nir_if *d_break = NULL, *disp_if = NULL;
   nir_def *d_max_pc = NULL;
   if (d_loop_rotated) {
      /* Expect: block -> `disp_if` -> block -> `d_break` -> block. */
      nir_cf_node *n1 = nir_cf_node_next(&d_hdr->cf_node);
      if (!n1 || n1->type != nir_cf_node_if)
         return false;
      disp_if = nir_cf_node_as_if(n1);
      nir_cf_node *n2 = nir_cf_node_next(&disp_if->cf_node);
      if (!n2 || n2->type != nir_cf_node_block)
         return false;
      nir_cf_node *n3 = nir_cf_node_next(n2);
      if (!n3 || n3->type != nir_cf_node_if)
         return false;
      d_break = nir_cf_node_as_if(n3);

      bool bt = cf_list_is_single_break(&d_break->then_list);
      bool be = cf_list_is_single_break(&d_break->else_list);
      if (bt == be || !(bt ? cf_list_is_empty_block(&d_break->else_list)
                           : cf_list_is_empty_block(&d_break->then_list)))
         return false;

      /* Break iff `d_max < d_next` (i.e. `d_next > d_max`). */
      struct bool_term t = {.def = d_break->condition.ssa, .negated = be};
      while (nir_def_instr(t.def)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(t.def))->op == nir_op_inot) {
         t.def = nir_instr_as_alu(nir_def_instr(t.def))->src[0].src.ssa;
         t.negated = !t.negated;
      }
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;
      if (!is_lt || !same_val(lhs, guard_d_max, wg) ||
          strip_mov(rhs) != strip_mov(d_next))
         return false;
      d_max_pc = guard_d_max;
   } else {
      /* Unrotated: header break `if (d > d_max) break;`, then `disp_if`. */
      nir_cf_node *n1 = nir_cf_node_next(&d_hdr->cf_node);
      if (!n1 || n1->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(n1);
      bool bt = cf_list_is_single_break(&nif->then_list);
      bool be = cf_list_is_single_break(&nif->else_list);
      if (bt == be)
         return false;
      struct bool_term t = {.def = nif->condition.ssa, .negated = be};
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u || !is_lt)
         return false;
      uint16_t max_pc = 0;
      if (!match_push_const_u32(lhs, &max_pc) ||
          strip_mov(rhs) != &d_phi->def)
         return false;
      d_max_pc = lhs;
      struct exec_list *d_body = NULL;
      if (!match_inclusive_range_loop_break(d_loop, &d_phi->def, d_max_pc, wg,
                                            &d_break, &d_body))
         return false;
      foreach_list_typed(nir_cf_node, node, node, d_body) {
         if (node == &d_break->cf_node)
            continue;
         if (node->type == nir_cf_node_if) {
            if (disp_if)
               return false;
            disp_if = nir_cf_node_as_if(node);
         }
      }
      if (!disp_if)
         return false;
   }

   /* Match `disp_if`: `if (x0 - d >= 0 && x1 - d < width) { ... }` with empty else. */
   struct exec_list *disp_body = NULL;
   nir_def *x0 = NULL, *x1 = NULL;
   if (cf_list_is_empty_block(&disp_if->else_list) &&
       match_stereo_disp_guard(disp_if->condition.ssa, false, &d_phi->def, w_pc,
                               wg, &x0, &x1)) {
      disp_body = &disp_if->then_list;
   } else if (cf_list_is_empty_block(&disp_if->then_list) &&
              match_stereo_disp_guard(disp_if->condition.ssa, true, &d_phi->def,
                                      w_pc, wg, &x0, &x1)) {
      disp_body = &disp_if->else_list;
   } else {
      return false;
   }

   uint16_t ws_pc_x0 = 0, ws_pc_x1 = 0;
   if (!match_clamped_window_bound(x0, x, w_pc, false, wg, &ws_pc_x0) ||
       !match_clamped_window_bound(x1, x, w_pc, true, wg, &ws_pc_x1) ||
       ws_pc_x0 != ws_pc_x1)
      return false;

   /* Match nested 2D SAD loops inside `disp_body`. */
   nir_loop *win_outer = find_single_loop_in_list(disp_body);
   if (!win_outer)
      return false;

   nir_block *wo_hdr = nir_loop_first_block(win_outer);
   nir_phi_instr *wo_iv = NULL, *wo_acc = NULL;
   nir_def *wo_iv_init = NULL, *wo_iv_next = NULL;
   nir_def *wo_acc_next = NULL;
   nir_foreach_phi(phi, wo_hdr) {
      nir_def *init = NULL, *next = NULL;
      if (phi->def.bit_size != 32 || phi->def.num_components != 1 ||
          !match_loop_phi(win_outer, phi, &init, &next))
         return false;
      if (is_const_i32(init, 0)) {
         if (wo_acc)
            return false;
         wo_acc = phi;
         wo_acc_next = next;
      } else {
         if (wo_iv)
            return false;
         wo_iv = phi;
         wo_iv_init = init;
         wo_iv_next = next;
      }
   }
   if (!wo_iv || !wo_acc || !match_iv_plus_one(wo_iv_next, &wo_iv->def))
      return false;

   bool outer_is_y = false;
   nir_def *y0 = NULL, *y1 = NULL;
   uint16_t ws_pc_y0 = 0, ws_pc_y1 = 0;
   if (match_clamped_window_bound(wo_iv_init, y, h_pc, false, wg, &ws_pc_y0) &&
       ws_pc_y0 == ws_pc_x0) {
      outer_is_y = true;
      y0 = wo_iv_init;
   } else if (strip_mov(wo_iv_init) == strip_mov(x0)) {
      outer_is_y = false;
   } else {
      return false;
   }

   /* Match outer window loop break. */
   nir_if *wo_break = NULL;
   struct exec_list *wo_body = NULL;
   {
      nir_cf_node *nn = nir_cf_node_next(&wo_hdr->cf_node);
      if (!nn || nn->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(nn);
      bool bt = cf_list_is_single_break(&nif->then_list);
      bool be = cf_list_is_single_break(&nif->else_list);
      if (bt == be)
         return false;
      struct bool_term t = {.def = nif->condition.ssa, .negated = be};
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u || !is_lt)
         return false;
      nir_def *wo_hi = lhs;
      if (outer_is_y) {
         if (!match_clamped_window_bound(wo_hi, y, h_pc, true, wg, &ws_pc_y1) ||
             ws_pc_y1 != ws_pc_x0)
            return false;
         y1 = wo_hi;
      } else {
         if (strip_mov(wo_hi) != strip_mov(x1))
            return false;
      }
      if (!match_inclusive_range_loop_break(win_outer, &wo_iv->def, wo_hi, wg,
                                            &wo_break, &wo_body))
         return false;
   }

   /* Match inner window loop inside `wo_body`. */
   nir_loop *win_inner = find_single_loop_in_list(wo_body);
   if (!win_inner)
      return false;

   nir_block *wi_hdr = nir_loop_first_block(win_inner);
   nir_phi_instr *wi_iv = NULL, *wi_acc = NULL;
   nir_def *wi_iv_init = NULL, *wi_iv_next = NULL;
   nir_def *wi_acc_next = NULL;
   nir_foreach_phi(phi, wi_hdr) {
      nir_def *init = NULL, *next = NULL;
      if (phi->def.bit_size != 32 || phi->def.num_components != 1 ||
          !match_loop_phi(win_inner, phi, &init, &next))
         return false;
      if (strip_mov(init) == &wo_acc->def) {
         if (wi_acc)
            return false;
         wi_acc = phi;
         wi_acc_next = next;
      } else {
         if (wi_iv)
            return false;
         wi_iv = phi;
         wi_iv_init = init;
         wi_iv_next = next;
      }
   }
   if (!wi_iv || !wi_acc || strip_mov(wo_acc_next) != &wi_acc->def ||
       !match_iv_plus_one(wi_iv_next, &wi_iv->def))
      return false;

   if (outer_is_y) {
      if (strip_mov(wi_iv_init) != strip_mov(x0))
         return false;
   } else {
      if (!match_clamped_window_bound(wi_iv_init, y, h_pc, false, wg, &ws_pc_y0) ||
          ws_pc_y0 != ws_pc_x0)
         return false;
      y0 = wi_iv_init;
   }

   nir_if *wi_break = NULL;
   struct exec_list *wi_body = NULL;
   {
      nir_cf_node *nn = nir_cf_node_next(&wi_hdr->cf_node);
      if (!nn || nn->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(nn);
      bool bt = cf_list_is_single_break(&nif->then_list);
      bool be = cf_list_is_single_break(&nif->else_list);
      if (bt == be)
         return false;
      struct bool_term t = {.def = nif->condition.ssa, .negated = be};
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u) || is_u || !is_lt)
         return false;
      nir_def *wi_hi = lhs;
      if (outer_is_y) {
         if (strip_mov(wi_hi) != strip_mov(x1))
            return false;
      } else {
         if (!match_clamped_window_bound(wi_hi, y, h_pc, true, wg, &ws_pc_y1) ||
             ws_pc_y1 != ws_pc_x0)
            return false;
         y1 = wi_hi;
      }
      if (!match_inclusive_range_loop_break(win_inner, &wi_iv->def, wi_hi, wg,
                                            &wi_break, &wi_body))
         return false;
   }

   /* Match inner SAD accumulation:
    * `wi_acc_next = wi_acc + iabs(u2u32(left[yy*stride+xx]) - u2u32(right[yy*stride+xx-d]))`. */
   nir_def *sad_add = strip_mov(wi_acc_next);
   if (!sad_add || nir_def_instr(sad_add)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *saa = nir_instr_as_alu(nir_def_instr(sad_add));
   if (saa->op != nir_op_iadd)
      return false;

   nir_def *abs_term = NULL;
   if (strip_mov(saa->src[0].src.ssa) == &wi_acc->def)
      abs_term = strip_mov(saa->src[1].src.ssa);
   else if (strip_mov(saa->src[1].src.ssa) == &wi_acc->def)
      abs_term = strip_mov(saa->src[0].src.ssa);
   else
      return false;

   nir_def *diff = NULL;
   if (!match_iabs_of(abs_term, &diff) || !diff ||
       nir_def_instr(diff)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *da = nir_instr_as_alu(nir_def_instr(diff));
   nir_def *op_a = NULL, *op_b = NULL;
   if (da->op == nir_op_isub) {
      op_a = strip_mov(da->src[0].src.ssa);
      op_b = strip_mov(da->src[1].src.ssa);
   } else if (da->op == nir_op_iadd) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *neg = strip_mov(da->src[side].src.ssa);
         if (neg && nir_def_instr(neg)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg) {
            op_a = strip_mov(da->src[1 - side].src.ssa);
            op_b = strip_mov(nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa);
            break;
         }
      }
   }
   if (!op_a || !op_b || nir_def_instr(op_a)->type != nir_instr_type_alu ||
       nir_def_instr(op_b)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *ua = nir_instr_as_alu(nir_def_instr(op_a));
   nir_alu_instr *ub = nir_instr_as_alu(nir_def_instr(op_b));
   if (ua->op != nir_op_u2u32 || ub->op != nir_op_u2u32)
      return false;

   nir_def *la = strip_mov(ua->src[0].src.ssa);
   nir_def *lb = strip_mov(ub->src[0].src.ssa);
   if (!la || !lb || nir_def_instr(la)->type != nir_instr_type_intrinsic ||
       nir_def_instr(lb)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *ld_a = nir_instr_as_intrinsic(nir_def_instr(la));
   nir_intrinsic_instr *ld_b = nir_instr_as_intrinsic(nir_def_instr(lb));
   if (ld_a->intrinsic != nir_intrinsic_load_deref ||
       ld_b->intrinsic != nir_intrinsic_load_deref)
      return false;
   if ((nir_intrinsic_access(ld_a) | nir_intrinsic_access(ld_b)) &
       (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
      return false;

   uint32_t sa_set = 0, sa_bind = 0, sb_set = 0, sb_bind = 0;
   nir_def *idx_a = NULL, *idx_b = NULL;
   if (!match_ssbo_deref_access(nir_src_as_deref(ld_a->src[0]), 8, &sa_set,
                                &sa_bind, &idx_a) ||
       !match_ssbo_deref_access(nir_src_as_deref(ld_b->src[0]), 8, &sb_set,
                                &sb_bind, &idx_b))
      return false;

   nir_def *yy_iv = outer_is_y ? &wo_iv->def : &wi_iv->def;
   nir_def *xx_iv = outer_is_y ? &wi_iv->def : &wo_iv->def;
   uint16_t sp_tmp = 0;
   nir_intrinsic_instr *left_ld = NULL, *right_ld = NULL;
   uint32_t left_set = 0, left_bind = 0, right_set = 0, right_bind = 0;
   if (match_linear_index(idx_a, yy_iv, xx_iv, &sp_tmp, wg) &&
       sp_tmp == stride_pc &&
       match_shifted_linear_index(idx_b, yy_iv, xx_iv, &d_phi->def, stride_pc,
                                  wg)) {
      left_ld = ld_a;
      right_ld = ld_b;
      left_set = sa_set;
      left_bind = sa_bind;
      right_set = sb_set;
      right_bind = sb_bind;
   } else if (match_linear_index(idx_b, yy_iv, xx_iv, &sp_tmp, wg) &&
              sp_tmp == stride_pc &&
              match_shifted_linear_index(idx_a, yy_iv, xx_iv, &d_phi->def,
                                         stride_pc, wg)) {
      left_ld = ld_b;
      right_ld = ld_a;
      left_set = sb_set;
      left_bind = sb_bind;
      right_set = sa_set;
      right_bind = sa_bind;
   } else {
      return false;
   }

   /* Verify min-cost update after `disp_if`: `bd_next` and `bc_next` are phis
    * in the block after `disp_if` merging `(upd_disp, bd_phi)` and
    * `(upd_cost, bc_phi)`. */
   nir_def *bd_s = strip_mov(bd_next);
   nir_def *bc_s = strip_mov(bc_next);
   if (!bd_s || !bc_s || nir_def_instr(bd_s)->type != nir_instr_type_phi ||
       nir_def_instr(bc_s)->type != nir_instr_type_phi)
      return false;

   nir_phi_instr *bd_join = nir_instr_as_phi(nir_def_instr(bd_s));
   nir_phi_instr *bc_join = nir_instr_as_phi(nir_def_instr(bc_s));
   nir_def *upd_disp = NULL, *upd_cost = NULL;
   bool bd_else_ok = false, bc_else_ok = false;

   nir_foreach_phi_src(ps, bd_join) {
      if (strip_mov(ps->src.ssa) == &bd_phi->def)
         bd_else_ok = true;
      else
         upd_disp = strip_mov(ps->src.ssa);
   }
   nir_foreach_phi_src(ps, bc_join) {
      if (strip_mov(ps->src.ssa) == &bc_phi->def)
         bc_else_ok = true;
      else
         upd_cost = strip_mov(ps->src.ssa);
   }
   if (!bd_else_ok || !bc_else_ok || !upd_disp || !upd_cost)
      return false;

   /* `upd_disp = bcsel(sad < best_cost, d, best_disp)` */
   if (nir_def_instr(upd_disp)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *uds = nir_instr_as_alu(nir_def_instr(upd_disp));
   if (uds->op != nir_op_bcsel)
      return false;
   bool flip_disp = false;
   if (strip_mov(uds->src[1].src.ssa) == &d_phi->def &&
       strip_mov(uds->src[2].src.ssa) == &bd_phi->def) {
      flip_disp = false;
   } else if (strip_mov(uds->src[2].src.ssa) == &d_phi->def &&
              strip_mov(uds->src[1].src.ssa) == &bd_phi->def) {
      flip_disp = true;
   } else {
      return false;
   }

   struct bool_term ct = {.def = uds->src[0].src.ssa, .negated = flip_disp};
   nir_def *cl = NULL, *cr = NULL;
   bool c_lt = false, c_u = false;
   if (!decompose_cmp_term(ct, &cl, &cr, &c_lt, &c_u) || c_u || !c_lt)
      return false;
   if (strip_mov(cl) != &wo_acc->def || strip_mov(cr) != &bc_phi->def)
      return false;

   /* `upd_cost = imin(best_cost, sad)` OR `bcsel(sad < best_cost, sad, best_cost)`. */
   if (nir_def_instr(upd_cost)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ucs = nir_instr_as_alu(nir_def_instr(upd_cost));
   bool cost_upd_ok = false;
   if (ucs->op == nir_op_imin) {
      nir_def *u0 = strip_mov(ucs->src[0].src.ssa);
      nir_def *u1 = strip_mov(ucs->src[1].src.ssa);
      cost_upd_ok = (u0 == &bc_phi->def && u1 == &wo_acc->def) ||
                    (u1 == &bc_phi->def && u0 == &wo_acc->def);
   } else if (ucs->op == nir_op_bcsel) {
      if (strip_mov(ucs->src[0].src.ssa) == strip_mov(uds->src[0].src.ssa)) {
         nir_def *tval = flip_disp ? strip_mov(ucs->src[2].src.ssa)
                                   : strip_mov(ucs->src[1].src.ssa);
         nir_def *eval = flip_disp ? strip_mov(ucs->src[1].src.ssa)
                                   : strip_mov(ucs->src[2].src.ssa);
         cost_upd_ok = (tval == &wo_acc->def && eval == &bc_phi->def);
      }
   }
   if (!cost_upd_ok)
      return false;

   /* Find final stores after `d_container`. */
   nir_cf_node *post_node = nir_cf_node_next(d_container);
   if (!post_node || post_node->type != nir_cf_node_block ||
       nir_cf_node_next(post_node) != NULL)
      return false;
   nir_block *post_blk = nir_cf_node_as_block(post_node);

   nir_def *expect_disp = d_loop_rotated ? bd_s : &bd_phi->def;
   nir_def *expect_cost = d_loop_rotated ? bc_s : &bc_phi->def;

   if (d_loop_rotated) {
      /* Match the 2 merge phis in `post_blk`. */
      nir_def *m_disp = NULL, *m_cost = NULL;
      nir_foreach_phi(phi, post_blk) {
         bool got_loop = false, got_init = false;
         bool is_disp = false, is_cost = false;
         nir_foreach_phi_src(ps, phi) {
            nir_def *sv = strip_mov(ps->src.ssa);
            if (sv == bd_s) {
               got_loop = true;
               is_disp = true;
            } else if (sv == strip_mov(bd_init)) {
               got_init = true;
               is_disp = true;
            } else if (sv == bc_s) {
               got_loop = true;
               is_cost = true;
            } else if (sv == strip_mov(bc_init)) {
               got_init = true;
               is_cost = true;
            }
         }
         if (got_loop && got_init && is_disp && !is_cost)
            m_disp = &phi->def;
         else if (got_loop && got_init && is_cost && !is_disp)
            m_cost = &phi->def;
         else
            return false;
      }
      if (!m_disp || !m_cost)
         return false;
      expect_disp = m_disp;
      expect_cost = m_cost;
   }

   nir_intrinsic_instr *dst_disp_st = NULL, *dst_cost_st = NULL;
   uint32_t ddisp_set = 0, ddisp_bind = 0, dcost_set = 0, dcost_bind = 0;
   nir_foreach_instr(instr, post_blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_store_deref)
         continue;
      if (nir_intrinsic_write_mask(intr) != 1 ||
          (nir_intrinsic_access(intr) &
           (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL)))
         return false;

      nir_deref_instr *drf = nir_src_as_deref(intr->src[0]);
      uint32_t set = 0, bind = 0;
      nir_def *idx = NULL;
      uint16_t sp = 0;
      if (match_ssbo_deref_access(drf, 8, &set, &bind, &idx)) {
         if (dst_disp_st || !match_linear_index(idx, y, x, &sp, wg) ||
             sp != stride_pc)
            return false;
         nir_def *sv = strip_mov(intr->src[1].ssa);
         if (!sv || nir_def_instr(sv)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *ta = nir_instr_as_alu(nir_def_instr(sv));
         if ((ta->op != nir_op_u2u8 && ta->op != nir_op_i2i8) ||
             strip_mov(ta->src[0].src.ssa) != expect_disp)
            return false;
         dst_disp_st = intr;
         ddisp_set = set;
         ddisp_bind = bind;
      } else if (match_ssbo_deref_access(drf, 32, &set, &bind, &idx)) {
         if (dst_cost_st || !match_linear_index(idx, y, x, &sp, wg) ||
             sp != cost_stride_pc)
            return false;
         if (strip_mov(intr->src[1].ssa) != expect_cost)
            return false;
         dst_cost_st = intr;
         dcost_set = set;
         dcost_bind = bind;
      } else {
         return false;
      }
   }
   if (!dst_disp_st || !dst_cost_st)
      return false;

   /* Reject compile-time obvious aliasing between outputs and read-only inputs. */
   if ((ddisp_set == dcost_set && ddisp_bind == dcost_bind) ||
       (ddisp_set == left_set && ddisp_bind == left_bind) ||
       (ddisp_set == right_set && ddisp_bind == right_bind) ||
       (dcost_set == left_set && dcost_bind == left_bind) ||
       (dcost_set == right_set && dcost_bind == right_bind))
      return false;

   const struct stereo_sad_cf cf = {
      .outer_if = outer_if,
      .d_guard_if = d_guard_if,
      .d_loop = d_loop,
      .d_break = d_break,
      .disp_if = disp_if,
      .win_outer_loop = win_outer,
      .win_outer_break = wo_break,
      .win_inner_loop = win_inner,
      .win_inner_break = wi_break,
   };
   if (!verify_stereo_completeness(s, &cf, prev_cost_ld, prev_disp_ld, left_ld,
                                   right_ld, dst_disp_st, dst_cost_st))
      return false;

   uint16_t w_off = 0, h_off = 0, dmin_off = 0, dmax_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off) ||
       !match_push_const_u32(d_init, &dmin_off) ||
       !match_push_const_u32(d_max_pc, &dmax_off))
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, left_set, left_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, right_set,
                              right_bind, &plan->bufs[1]) ||
       !populate_binding_info(set_layout_count, set_layouts, pdisp_set,
                              pdisp_bind, &plan->bufs[2]) ||
       !populate_binding_info(set_layout_count, set_layouts, pcost_set,
                              pcost_bind, &plan->bufs[3]) ||
       !populate_binding_info(set_layout_count, set_layouts, ddisp_set,
                              ddisp_bind, &plan->bufs[4]) ||
       !populate_binding_info(set_layout_count, set_layouts, dcost_set,
                              dcost_bind, &plan->bufs[5]))
      return false;

   plan->kind = HK_2D_STENCIL_STEREO_SAD_U8;
   plan->tile_w = 64;
   plan->tile_h = 16;
   plan->width_pc_offset_B = w_off;
   plan->height_pc_offset_B = h_off;
   plan->stride_pc_offset_B = stride_pc;
   plan->radius_pc_offset_B = ws_pc_x0;
   plan->d_min_pc_offset_B = dmin_off;
   plan->d_max_pc_offset_B = dmax_off;
   plan->cost_stride_pc_offset_B = cost_stride_pc;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 3: 2D 3x3 Sobel Gradient Magnitude + Quantized Direction
 *            (R8 src -> i32 dir + f32 mag)
 * --------------------------------------------------------------------------- */

static int
match_coord_offset_1(nir_def *def, nir_def *coord, const uint32_t wg_size[3])
{
   if (same_val(def, coord, wg_size))
      return 1;
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return -1;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iadd) {
      for (unsigned side = 0; side < 2; side++) {
         int32_t c = 0;
         if (same_val(alu->src[side].src.ssa, coord, wg_size) &&
             match_const_i32(alu->src[1 - side].src.ssa, &c)) {
            if (c == -1)
               return 0;
            if (c == 1)
               return 2;
         }
      }
   } else if (alu->op == nir_op_isub &&
              same_val(alu->src[0].src.ssa, coord, wg_size)) {
      int32_t c = 0;
      if (match_const_i32(alu->src[1].src.ssa, &c)) {
         if (c == 1)
            return 0;
         if (c == -1)
            return 2;
      }
   }
   return -1;
}

static bool
match_const_f32_bits(nir_def *def, uint32_t *bits_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   *bits_out = nir_src_as_uint(nir_src_for_ssa(def));
   return true;
}

static bool
match_const_f32_int(nir_def *def, int *val_out)
{
   uint32_t bits = 0;
   if (!match_const_f32_bits(def, &bits))
      return false;
   float f;
   memcpy(&f, &bits, sizeof(f));
   if (f == 2.0f) {
      *val_out = 2;
      return true;
   }
   if (f == -2.0f) {
      *val_out = -2;
      return true;
   }
   if (f == 1.0f) {
      *val_out = 1;
      return true;
   }
   if (f == -1.0f) {
      *val_out = -1;
      return true;
   }
   return false;
}

/* Recursively evaluate a linear combination of the 8 Sobel u8 samples. */
static bool
eval_sobel_linear(nir_def *def, nir_intrinsic_instr *const samples[9],
                  int coeffs[9], unsigned depth)
{
   if (depth > 24)
      return false;
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1 ||
       nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (nir_alu_instr_is_exact(alu))
      return false;

   if (alu->op == nir_op_u2f32 || alu->op == nir_op_i2f32) {
      nir_def *s = strip_mov(alu->src[0].src.ssa);
      if (alu->op == nir_op_i2f32 && s &&
          nir_def_instr(s)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(s))->op == nir_op_u2u32) {
         s = strip_mov(nir_instr_as_alu(nir_def_instr(s))->src[0].src.ssa);
      } else if (alu->op == nir_op_i2f32) {
         return false;
      }
      if (!s || nir_def_instr(s)->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *ld = nir_instr_as_intrinsic(nir_def_instr(s));
      for (unsigned k = 0; k < 9; k++) {
         if (samples[k] == ld) {
            memset(coeffs, 0, 9 * sizeof(int));
            coeffs[k] = 1;
            return true;
         }
      }
      return false;
   }

   if (alu->op == nir_op_fneg) {
      if (!eval_sobel_linear(alu->src[0].src.ssa, samples, coeffs, depth + 1))
         return false;
      for (unsigned k = 0; k < 9; k++)
         coeffs[k] = -coeffs[k];
      return true;
   }

   if (alu->op == nir_op_fadd || alu->op == nir_op_fsub) {
      int a[9], b[9];
      if (!eval_sobel_linear(alu->src[0].src.ssa, samples, a, depth + 1) ||
          !eval_sobel_linear(alu->src[1].src.ssa, samples, b, depth + 1))
         return false;
      int sign = (alu->op == nir_op_fsub) ? -1 : 1;
      for (unsigned k = 0; k < 9; k++)
         coeffs[k] = a[k] + sign * b[k];
      return true;
   }

   if (alu->op == nir_op_fmul) {
      int sc = 0;
      nir_def *var = NULL;
      if (match_const_f32_int(alu->src[0].src.ssa, &sc))
         var = alu->src[1].src.ssa;
      else if (match_const_f32_int(alu->src[1].src.ssa, &sc))
         var = alu->src[0].src.ssa;
      else
         return false;
      if (!eval_sobel_linear(var, samples, coeffs, depth + 1))
         return false;
      for (unsigned k = 0; k < 9; k++)
         coeffs[k] *= sc;
      return true;
   }

   if (alu->op == nir_op_ffma) {
      int sc = 0;
      nir_def *var = NULL;
      if (match_const_f32_int(alu->src[0].src.ssa, &sc))
         var = alu->src[1].src.ssa;
      else if (match_const_f32_int(alu->src[1].src.ssa, &sc))
         var = alu->src[0].src.ssa;
      else
         return false;
      int a[9], c[9];
      if (!eval_sobel_linear(var, samples, a, depth + 1) ||
          !eval_sobel_linear(alu->src[2].src.ssa, samples, c, depth + 1))
         return false;
      for (unsigned k = 0; k < 9; k++)
         coeffs[k] = sc * a[k] + c[k];
      return true;
   }

   return false;
}

static bool
match_flt_zero(nir_def *cond, nir_def *val)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   uint32_t bits = 0;
   return alu->op == nir_op_flt &&
          strip_mov(alu->src[0].src.ssa) == strip_mov(val) &&
          match_const_f32_bits(alu->src[1].src.ssa, &bits) && bits == 0u;
}

static bool
match_bcsel_const_pair(nir_def *def, nir_def *neg_val, int32_t t_const,
                       int32_t e_const)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   return alu->op == nir_op_bcsel &&
          match_flt_zero(alu->src[0].src.ssa, neg_val) &&
          is_const_i32(alu->src[1].src.ssa, t_const) &&
          is_const_i32(alu->src[2].src.ssa, e_const);
}

/* Match the exact 8-sector Sobel direction quantization:
 *   ratio = fabs(gx * frcp(gy))  (or fabs(fdiv(gx, gy)))
 *   diag  = (gy < 0) ? ((gx < 0) ? 5 : 3) : ((gx < 0) ? 7 : 1)
 *   q1    = (ratio < 0.41421357f) ? ((gy < 0) ? 4 : 0) : diag
 *   q2    = (2.4142137f < ratio) ? ((gx < 0) ? 6 : 2) : q1
 *   dir   = (q2 + 6) & 7 */
static bool
match_sobel_dir_expr(nir_def *dir_def, nir_def *gy, nir_def *gx)
{
   dir_def = strip_mov(dir_def);
   if (!dir_def || nir_def_instr(dir_def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *a_and = nir_instr_as_alu(nir_def_instr(dir_def));
   if (a_and->op != nir_op_iand)
      return false;

   nir_def *add_def = NULL;
   if (is_const_i32(a_and->src[1].src.ssa, 7))
      add_def = strip_mov(a_and->src[0].src.ssa);
   else if (is_const_i32(a_and->src[0].src.ssa, 7))
      add_def = strip_mov(a_and->src[1].src.ssa);
   else
      return false;

   if (!add_def || nir_def_instr(add_def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *a_add = nir_instr_as_alu(nir_def_instr(add_def));
   if (a_add->op != nir_op_iadd)
      return false;

   nir_def *q2 = NULL;
   if (is_const_i32(a_add->src[1].src.ssa, 6))
      q2 = strip_mov(a_add->src[0].src.ssa);
   else if (is_const_i32(a_add->src[0].src.ssa, 6))
      q2 = strip_mov(a_add->src[1].src.ssa);
   else
      return false;

   if (!q2 || nir_def_instr(q2)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *s2 = nir_instr_as_alu(nir_def_instr(q2));
   if (s2->op != nir_op_bcsel)
      return false;

   /* s2->src[0] must be `2.4142137f (0x401a827a) < ratio`. */
   nir_def *c2 = strip_mov(s2->src[0].src.ssa);
   if (!c2 || nir_def_instr(c2)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cmp2 = nir_instr_as_alu(nir_def_instr(c2));
   uint32_t hi_bits = 0;
   if (cmp2->op != nir_op_flt ||
       !match_const_f32_bits(cmp2->src[0].src.ssa, &hi_bits) ||
       hi_bits != 0x401a827au)
      return false;
   nir_def *ratio = strip_mov(cmp2->src[1].src.ssa);

   /* Verify `ratio == fabs(gx * frcp(gy))` or `fabs(fdiv(gx, gy))`. */
   if (!ratio || nir_def_instr(ratio)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *rabs = nir_instr_as_alu(nir_def_instr(ratio));
   if (rabs->op != nir_op_fabs)
      return false;
   nir_def *quot = strip_mov(rabs->src[0].src.ssa);
   if (!quot || nir_def_instr(quot)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *qa = nir_instr_as_alu(nir_def_instr(quot));
   bool quot_ok = false;
   if (qa->op == nir_op_fdiv &&
       strip_mov(qa->src[0].src.ssa) == strip_mov(gx) &&
       strip_mov(qa->src[1].src.ssa) == strip_mov(gy)) {
      quot_ok = true;
   } else if (qa->op == nir_op_fmul) {
      for (unsigned side = 0; side < 2; side++) {
         if (strip_mov(qa->src[side].src.ssa) != strip_mov(gx))
            continue;
         nir_def *rcp = strip_mov(qa->src[1 - side].src.ssa);
         if (rcp && nir_def_instr(rcp)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(rcp))->op == nir_op_frcp &&
             strip_mov(nir_instr_as_alu(nir_def_instr(rcp))->src[0].src.ssa) ==
                strip_mov(gy))
            quot_ok = true;
      }
   }
   if (!quot_ok)
      return false;

   /* s2->src[1] must be `(gx < 0) ? 6 : 2`. */
   if (!match_bcsel_const_pair(s2->src[1].src.ssa, gx, 6, 2))
      return false;

   /* s2->src[2] is `q1 = (ratio < 0.41421357f) ? ((gy < 0) ? 4 : 0) : diag`. */
   nir_def *q1 = strip_mov(s2->src[2].src.ssa);
   if (!q1 || nir_def_instr(q1)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *s1 = nir_instr_as_alu(nir_def_instr(q1));
   if (s1->op != nir_op_bcsel)
      return false;
   nir_def *c1 = strip_mov(s1->src[0].src.ssa);
   if (!c1 || nir_def_instr(c1)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cmp1 = nir_instr_as_alu(nir_def_instr(c1));
   uint32_t lo_bits = 0;
   if (cmp1->op != nir_op_flt || strip_mov(cmp1->src[0].src.ssa) != ratio ||
       !match_const_f32_bits(cmp1->src[1].src.ssa, &lo_bits) ||
       lo_bits != 0x3ed413cdu)
      return false;

   if (!match_bcsel_const_pair(s1->src[1].src.ssa, gy, 4, 0))
      return false;

   /* `diag = (gy < 0) ? ((gx < 0) ? 5 : 3) : ((gx < 0) ? 7 : 1)`. */
   nir_def *diag = strip_mov(s1->src[2].src.ssa);
   if (!diag || nir_def_instr(diag)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *sd = nir_instr_as_alu(nir_def_instr(diag));
   return sd->op == nir_op_bcsel && match_flt_zero(sd->src[0].src.ssa, gy) &&
          match_bcsel_const_pair(sd->src[1].src.ssa, gx, 5, 3) &&
          match_bcsel_const_pair(sd->src[2].src.ssa, gx, 7, 1);
}

/* Match `x >= 1 && x < w - 1 && y >= 1 && y < h - 1` returning x, y, w, h. */
static bool
match_sobel_interior_guard(nir_def *cond, bool negated,
                           const uint32_t wg_size[3], nir_def **x_out,
                           nir_def **y_out, nir_def **w_pc_out,
                           nir_def **h_pc_out)
{
   struct bool_term terms[8];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 8) || n != 4)
      return false;

   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   bool got_x_lo = false, got_y_lo = false;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;
      unsigned comp = 0;
      if (!match_global_id_comp(lhs, wg_size, &comp) || comp > 1)
         return false;
      if (!is_lt) {
         if (!is_const_i32(rhs, 1))
            return false;
         if (comp == 0) {
            x = lhs;
            got_x_lo = true;
         } else {
            y = lhs;
            got_y_lo = true;
         }
      } else {
         /* `coord < dim - 1` */
         nir_def *r = strip_mov(rhs);
         if (!r || nir_def_instr(r)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *ra = nir_instr_as_alu(nir_def_instr(r));
         nir_def *dim = NULL;
         if (ra->op == nir_op_iadd) {
            if (is_const_i32(ra->src[1].src.ssa, -1))
               dim = ra->src[0].src.ssa;
            else if (is_const_i32(ra->src[0].src.ssa, -1))
               dim = ra->src[1].src.ssa;
         } else if (ra->op == nir_op_isub &&
                    is_const_i32(ra->src[1].src.ssa, 1)) {
            dim = ra->src[0].src.ssa;
         }
         uint16_t pc = 0;
         if (!dim || !match_push_const_u32(dim, &pc))
            return false;
         if (comp == 0) {
            x = lhs;
            w_pc = dim;
         } else {
            y = lhs;
            h_pc = dim;
         }
      }
   }

   if (!got_x_lo || !got_y_lo || !x || !y || !w_pc || !h_pc)
      return false;
   *x_out = x;
   *y_out = y;
   *w_pc_out = w_pc;
   *h_pc_out = h_pc;
   return true;
}

static bool
verify_sobel_cf_list(struct exec_list *cf_list, nir_if *int_if,
                     nir_if *border_if)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_jump)
               return false;
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (nif != int_if && nif != border_if)
            return false;
         if (!verify_sobel_cf_list(&nif->then_list, int_if, border_if) ||
             !verify_sobel_cf_list(&nif->else_list, int_if, border_if))
            return false;
         break;
      }
      default:
         return false;
      }
   }
   return true;
}

static bool
match_sobel_3x3_u8(nir_shader *s, uint32_t set_layout_count,
                   struct vk_descriptor_set_layout *const *set_layouts,
                   struct hk_2d_stencil_plan *plan)
{
   const uint32_t *wg = plan->orig_wg_size;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   nir_if *int_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if) {
         if (int_if)
            return false;
         int_if = nir_cf_node_as_if(node);
      } else if (node->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!int_if)
      return false;

   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   struct exec_list *int_list = NULL, *else_list = NULL;
   if (match_sobel_interior_guard(int_if->condition.ssa, false, wg, &x, &y,
                                  &w_pc, &h_pc)) {
      int_list = &int_if->then_list;
      else_list = &int_if->else_list;
   } else if (match_sobel_interior_guard(int_if->condition.ssa, true, wg, &x,
                                         &y, &w_pc, &h_pc)) {
      int_list = &int_if->else_list;
      else_list = &int_if->then_list;
   } else {
      return false;
   }

   /* Interior branch must be a single block with 8 src loads and 2 stores. */
   if (!exec_list_is_singular(int_list))
      return false;
   nir_cf_node *in_node =
      exec_node_data(nir_cf_node, exec_list_get_head(int_list), node);
   if (in_node->type != nir_cf_node_block)
      return false;
   nir_block *int_blk = nir_cf_node_as_block(in_node);

   nir_intrinsic_instr *samples[9] = {NULL};
   unsigned sample_cnt = 0;
   uint32_t src_set = 0, src_bind = 0;
   uint16_t stride_pc = 0;

   nir_intrinsic_instr *int_stores[2] = {NULL, NULL};
   unsigned int_store_cnt = 0;

   nir_foreach_instr(instr, int_blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_deref) {
         if (nir_intrinsic_access(intr) &
             (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
            return false;
         uint32_t set = 0, bind = 0;
         nir_def *idx = NULL;
         if (!match_ssbo_deref_access(nir_src_as_deref(intr->src[0]), 8, &set,
                                      &bind, &idx))
            return false;
         /* idx must be `(y + dy) * stride + (x + dx)`. */
         idx = strip_mov(idx);
         if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(idx));
         if (add->op != nir_op_iadd)
            return false;
         int dx = -1, dy = -1;
         uint16_t sp = 0;
         for (unsigned side = 0; side < 2; side++) {
            int cx = match_coord_offset_1(add->src[side].src.ssa, x, wg);
            if (cx < 0)
               continue;
            nir_def *prod = strip_mov(add->src[1 - side].src.ssa);
            if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
            if (mul->op != nir_op_imul && mul->op != nir_op_amul)
               continue;
            for (unsigned ms = 0; ms < 2; ms++) {
               int cy = match_coord_offset_1(mul->src[ms].src.ssa, y, wg);
               uint16_t p = 0;
               if (cy >= 0 &&
                   match_push_const_u32(mul->src[1 - ms].src.ssa, &p)) {
                  dx = cx;
                  dy = cy;
                  sp = p;
                  break;
               }
            }
            if (dy >= 0)
               break;
         }
         if (dx < 0 || dy < 0 || (dx == 1 && dy == 1))
            return false;
         if (sample_cnt == 0) {
            src_set = set;
            src_bind = bind;
            stride_pc = sp;
         } else if (set != src_set || bind != src_bind || sp != stride_pc) {
            return false;
         }
         if (samples[dy * 3 + dx])
            return false;
         samples[dy * 3 + dx] = intr;
         sample_cnt++;
      } else if (intr->intrinsic == nir_intrinsic_store_deref) {
         if (int_store_cnt >= 2 || nir_intrinsic_write_mask(intr) != 1 ||
             (nir_intrinsic_access(intr) &
              (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL)))
            return false;
         int_stores[int_store_cnt++] = intr;
      }
   }

   if (sample_cnt != 8 || int_store_cnt != 2)
      return false;

   /* Classify the 2 interior stores into `mag` (`fabs(gy) + fabs(gx)`) and
    * `dir` (`(q + 6) & 7`). */
   nir_intrinsic_instr *mag_st = NULL, *dir_st = NULL;
   uint32_t mag_set = 0, mag_bind = 0, dir_set = 0, dir_bind = 0;
   nir_def *gy_def = NULL, *gx_def = NULL;
   uint16_t w_off = 0, h_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off))
      return false;

   static const int expect_gy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
   static const int expect_gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

   for (unsigned i = 0; i < 2; i++) {
      uint32_t set = 0, bind = 0;
      nir_def *idx = NULL;
      uint16_t sp = 0;
      if (!match_ssbo_deref_access(nir_src_as_deref(int_stores[i]->src[0]), 32,
                                   &set, &bind, &idx) ||
          !match_linear_index(idx, y, x, &sp, wg) || sp != w_off)
         return false;

      nir_def *val = strip_mov(int_stores[i]->src[1].ssa);
      if (val && nir_def_instr(val)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(val))->op == nir_op_fadd) {
         nir_alu_instr *fa = nir_instr_as_alu(nir_def_instr(val));
         nir_def *a0 = strip_mov(fa->src[0].src.ssa);
         nir_def *a1 = strip_mov(fa->src[1].src.ssa);
         if (a0 && a1 && nir_def_instr(a0)->type == nir_instr_type_alu &&
             nir_def_instr(a1)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(a0))->op == nir_op_fabs &&
             nir_instr_as_alu(nir_def_instr(a1))->op == nir_op_fabs) {
            nir_def *v0 = nir_instr_as_alu(nir_def_instr(a0))->src[0].src.ssa;
            nir_def *v1 = nir_instr_as_alu(nir_def_instr(a1))->src[0].src.ssa;
            int c0[9], c1[9];
            if (eval_sobel_linear(v0, samples, c0, 0) &&
                eval_sobel_linear(v1, samples, c1, 0)) {
               if (memcmp(c0, expect_gy, sizeof(c0)) == 0 &&
                   memcmp(c1, expect_gx, sizeof(c1)) == 0) {
                  gy_def = v0;
                  gx_def = v1;
                  mag_st = int_stores[i];
                  mag_set = set;
                  mag_bind = bind;
                  continue;
               }
               if (memcmp(c1, expect_gy, sizeof(c1)) == 0 &&
                   memcmp(c0, expect_gx, sizeof(c0)) == 0) {
                  gy_def = v1;
                  gx_def = v0;
                  mag_st = int_stores[i];
                  mag_set = set;
                  mag_bind = bind;
                  continue;
               }
            }
         }
      }
      if (!dir_st) {
         dir_st = int_stores[i];
         dir_set = set;
         dir_bind = bind;
      }
   }

   if (!mag_st || !dir_st || !gy_def || !gx_def)
      return false;
   if (!match_sobel_dir_expr(dir_st->src[1].ssa, gy_def, gx_def))
      return false;

   /* Match border else branch: `if (x < width && y < height) { mag[idx] = 0; }`. */
   nir_if *border_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, else_list) {
      if (node->type == nir_cf_node_if) {
         if (border_if)
            return false;
         border_if = nir_cf_node_as_if(node);
      } else if (node->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!border_if)
      return false;

   nir_def *bx = NULL, *by = NULL, *bw = NULL, *bh = NULL;
   struct exec_list *bstore_list = NULL;
   if (cf_list_is_empty_block(&border_if->else_list) &&
       match_xy_bounds_cond(border_if->condition.ssa, false, wg, &bx, &by, &bw,
                            &bh)) {
      bstore_list = &border_if->then_list;
   } else if (cf_list_is_empty_block(&border_if->then_list) &&
              match_xy_bounds_cond(border_if->condition.ssa, true, wg, &bx, &by,
                                   &bw, &bh)) {
      bstore_list = &border_if->else_list;
   } else {
      return false;
   }
   if (!same_val(bw, w_pc, wg) || !same_val(bh, h_pc, wg))
      return false;

   if (!exec_list_is_singular(bstore_list))
      return false;
   nir_cf_node *bnode =
      exec_node_data(nir_cf_node, exec_list_get_head(bstore_list), node);
   if (bnode->type != nir_cf_node_block)
      return false;
   nir_block *bblk = nir_cf_node_as_block(bnode);

   nir_intrinsic_instr *border_mag_st = NULL;
   nir_foreach_instr(instr, bblk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_store_deref) {
         if (border_mag_st)
            return false;
         border_mag_st = intr;
      }
   }
   if (!border_mag_st || nir_intrinsic_write_mask(border_mag_st) != 1 ||
       !is_const_i32(border_mag_st->src[1].ssa, 0) ||
       (nir_intrinsic_access(border_mag_st) &
        (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL)))
      return false;

   uint32_t bm_set = 0, bm_bind = 0;
   nir_def *bm_idx = NULL;
   uint16_t bm_sp = 0;
   if (!match_ssbo_deref_access(nir_src_as_deref(border_mag_st->src[0]), 32,
                                &bm_set, &bm_bind, &bm_idx) ||
       bm_set != mag_set || bm_bind != mag_bind ||
       !match_linear_index(bm_idx, y, x, &bm_sp, wg) || bm_sp != w_off)
      return false;

   if ((src_set == dir_set && src_bind == dir_bind) ||
       (src_set == mag_set && src_bind == mag_bind) ||
       (dir_set == mag_set && dir_bind == mag_bind))
      return false;

   /* Completeness check across the whole shader. */
   if (!verify_sobel_cf_list(&impl->body, int_if, border_if))
      return false;

   unsigned seen_loads = 0, seen_stores = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref: {
            bool ok = false;
            for (unsigned k = 0; k < 9; k++)
               if (samples[k] == intr)
                  ok = true;
            if (!ok)
               return false;
            seen_loads++;
            break;
         }
         case nir_intrinsic_store_deref:
            if (intr == mag_st || intr == dir_st || intr == border_mag_st) {
               seen_stores++;
               break;
            }
            return false;
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            return false;
         }
      }
   }
   if (seen_loads != 8 || seen_stores != 3)
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, dir_set, dir_bind,
                              &plan->bufs[1]) ||
       !populate_binding_info(set_layout_count, set_layouts, mag_set, mag_bind,
                              &plan->bufs[2]))
      return false;

   plan->kind = HK_2D_STENCIL_SOBEL_3X3_U8;
   plan->tile_w = 128;
   plan->tile_h = 8;
   plan->width_pc_offset_B = w_off;
   plan->height_pc_offset_B = h_off;
   plan->stride_pc_offset_B = stride_pc;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 4: Grid-stride FAST-9 u8 corner detector (radius 3, 16-pt circle)
 * --------------------------------------------------------------------------- */

static const int8_t fast9_dx[16] = {
   0,  1,  2,  3,  3,  3,  2,  1,
   0, -1, -2, -3, -3, -3, -2, -1,
};

static const int8_t fast9_dy[16] = {
   3,  3,  2,  1,  0, -1, -2, -3,
  -3, -3, -2, -1,  0,  1,  2,  3,
};

static bool
match_fast9_grid_loop(nir_loop *loop, unsigned comp, const uint32_t wg[3],
                      nir_def **coord_out, nir_def **limit_pc_out,
                      nir_if **exit_if_out)
{
   nir_block *hdr = nir_loop_first_block(loop);
   nir_phi_instr *phi = NULL;

   /* Find the unique grid-stride induction phi (`init = gid[comp]`,
    * `latch = phi + num_wg[comp] * wg[comp]`). Uninitialized scratch locals
    * hoisted before the loop may introduce extra loop-carried phis that are
    * ignored here because all stores/atomics/branches are verified directly. */
   nir_foreach_phi(p, hdr) {
      if (p->def.bit_size != 32 || p->def.num_components != 1)
         continue;
      nir_def *init = NULL, *latch = NULL;
      bool bad = false;
      nir_foreach_phi_src(ps, p) {
         if (ps->pred->index < hdr->index) {
            if (init)
               bad = true;
            init = ps->src.ssa;
         } else {
            if (latch)
               bad = true;
            latch = ps->src.ssa;
         }
      }
      if (bad || !init || !latch)
         continue;
      unsigned init_c = 0;
      if (!match_global_id_comp(init, wg, &init_c) || init_c != comp)
         continue;
      nir_def *ls = strip_mov(latch);
      if (!ls || nir_def_instr(ls)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *ladd = nir_instr_as_alu(nir_def_instr(ls));
      if (ladd->op != nir_op_iadd)
         continue;
      bool step_ok = false;
      for (unsigned side = 0; side < 2; side++) {
         if (strip_mov(ladd->src[side].src.ssa) == &p->def &&
             match_scaled_sysval_comp(&ladd->src[1 - side],
                                      nir_intrinsic_load_num_workgroups, wg, comp)) {
            step_ok = true;
            break;
         }
      }
      if (!step_ok)
         continue;
      if (phi)
         return false;
      phi = p;
   }
   if (!phi)
      return false;

   /* Find the first `if` in loop body: must be `if (coord >= limit) break;`. */
   nir_if *exit_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, &loop->body) {
      if (n->type == nir_cf_node_if) {
         exit_if = nir_cf_node_as_if(n);
         break;
      }
      if (n->type != nir_cf_node_block)
         return false;
   }
   if (!exit_if)
      return false;

   bool exit_on_true = false;
   if (cf_list_is_single_break(&exit_if->then_list) &&
       cf_list_is_empty_block(&exit_if->else_list)) {
      exit_on_true = true;
   } else if (cf_list_is_single_break(&exit_if->else_list) &&
              cf_list_is_empty_block(&exit_if->then_list)) {
      exit_on_true = false;
   } else {
      return false;
   }

   struct bool_term terms[2];
   unsigned nt = 0;
   if (!collect_and_terms(exit_if->condition.ssa, !exit_on_true, terms, &nt, 2) ||
       nt != 1)
      return false;

   nir_def *lhs = NULL, *rhs = NULL;
   bool is_lt = false, is_u = false;
   if (!decompose_cmp_term(terms[0], &lhs, &rhs, &is_lt, &is_u))
      return false;

   /* Break condition: `coord >= limit_pc` (i.e. `!is_lt && lhs == coord`). */
   uint16_t lim_off = 0;
   if (is_lt || strip_mov(lhs) != &phi->def ||
       !match_push_const_u32(rhs, &lim_off))
      return false;

   *coord_out = &phi->def;
   *limit_pc_out = rhs;
   *exit_if_out = exit_if;
   return true;
}

/* Match `imax(3, border_pc)` or `imax(border_pc, 3)`. */
static bool
match_imax3_border(nir_def *def, uint16_t *border_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_imax)
      return false;
   for (unsigned side = 0; side < 2; side++) {
      uint16_t off = 0;
      if (is_const_i32(alu->src[side].src.ssa, 3) &&
          match_push_const_u32(alu->src[1 - side].src.ssa, &off)) {
         *border_pc_out = off;
         return true;
      }
   }
   return false;
}

enum fast9_border_bit {
   F9_X_LO_3 = 1u << 0,
   F9_X_LO_B = 1u << 1,
   F9_X_HI_3 = 1u << 2,
   F9_X_HI_B = 1u << 3,
   F9_Y_LO_3 = 1u << 4,
   F9_Y_LO_B = 1u << 5,
   F9_Y_HI_3 = 1u << 6,
   F9_Y_HI_B = 1u << 7,
   F9_BORDER_ALL = 0xffu,
};

static bool
match_imin_xy(nir_def *def, nir_def *x, nir_def *y)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_imin)
      return false;
   nir_def *a = strip_mov(alu->src[0].src.ssa);
   nir_def *b = strip_mov(alu->src[1].src.ssa);
   return (a == x && b == y) || (a == y && b == x);
}

static bool
collect_fast9_border_flags(nir_def *cond, bool negated, nir_def *x, nir_def *y,
                           nir_def *w_pc, nir_def *h_pc, const uint32_t wg[3],
                           unsigned *flags_inout, bool *has_b_inout,
                           uint16_t *border_pc_inout)
{
   struct bool_term terms[16];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 16) || n < 2)
      return false;

   unsigned flags = *flags_inout;
   bool has_b = *has_b_inout;
   uint16_t b_off = *border_pc_inout;

   for (unsigned i = 0; i < n; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || is_u)
         return false;

      uint16_t off = 0;
      if (!is_lt) {
         /* `coord >= max(3, border)`, `coord >= 3`, `coord >= border`,
          * `imin(x, y) >= ...`, or `dim - border >= coord`. */
         if (match_imin_xy(lhs, x, y)) {
            if (match_imax3_border(rhs, &off)) {
               if (has_b && off != b_off)
                  return false;
               has_b = true;
               b_off = off;
               flags |= F9_X_LO_3 | F9_X_LO_B | F9_Y_LO_3 | F9_Y_LO_B;
               continue;
            }
            if (is_const_i32(rhs, 3)) {
               flags |= F9_X_LO_3 | F9_Y_LO_3;
               continue;
            }
            if (match_push_const_u32(rhs, &off)) {
               if (has_b && off != b_off)
                  return false;
               has_b = true;
               b_off = off;
               flags |= F9_X_LO_B | F9_Y_LO_B;
               continue;
            }
         }
         if (strip_mov(lhs) == x) {
            if (match_imax3_border(rhs, &off)) {
               if (has_b && off != b_off)
                  return false;
               has_b = true;
               b_off = off;
               flags |= F9_X_LO_3 | F9_X_LO_B;
               continue;
            }
            if (is_const_i32(rhs, 3)) {
               flags |= F9_X_LO_3;
               continue;
            }
            if (match_push_const_u32(rhs, &off)) {
               if (has_b && off != b_off)
                  return false;
               has_b = true;
               b_off = off;
               flags |= F9_X_LO_B;
               continue;
            }
         }
         if (strip_mov(lhs) == y) {
            if (match_imax3_border(rhs, &off)) {
               if (has_b && off != b_off)
                  return false;
               has_b = true;
               b_off = off;
               flags |= F9_Y_LO_3 | F9_Y_LO_B;
               continue;
            }
            if (is_const_i32(rhs, 3)) {
               flags |= F9_Y_LO_3;
               continue;
            }
            if (match_push_const_u32(rhs, &off)) {
               if (has_b && off != b_off)
                  return false;
               has_b = true;
               b_off = off;
               flags |= F9_Y_LO_B;
               continue;
            }
         }
         if (strip_mov(rhs) == x) {
            /* `w - border >= x` (either `isub(w, border)` or `iadd(w, ineg(border))`) */
            nir_def *ls = strip_mov(lhs);
            if (ls && nir_def_instr(ls)->type == nir_instr_type_alu) {
               nir_alu_instr *al = nir_instr_as_alu(nir_def_instr(ls));
               if (al->op == nir_op_isub &&
                   same_val(al->src[0].src.ssa, w_pc, wg) &&
                   match_push_const_u32(al->src[1].src.ssa, &off)) {
                  if (has_b && off != b_off)
                     return false;
                  has_b = true;
                  b_off = off;
                  flags |= F9_X_HI_B;
                  continue;
               }
               if (al->op == nir_op_iadd) {
                  bool ok = false;
                  for (unsigned side = 0; side < 2; side++) {
                     nir_def *neg = strip_mov(al->src[1 - side].src.ssa);
                     if (same_val(al->src[side].src.ssa, w_pc, wg) && neg &&
                         nir_def_instr(neg)->type == nir_instr_type_alu &&
                         nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg &&
                         match_push_const_u32(
                            nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa,
                            &off)) {
                        if (has_b && off != b_off)
                           return false;
                        has_b = true;
                        b_off = off;
                        flags |= F9_X_HI_B;
                        ok = true;
                        break;
                     }
                  }
                  if (ok)
                     continue;
               }
            }
         }
         if (strip_mov(rhs) == y) {
            /* `h - border >= y` */
            nir_def *ls = strip_mov(lhs);
            if (ls && nir_def_instr(ls)->type == nir_instr_type_alu) {
               nir_alu_instr *al = nir_instr_as_alu(nir_def_instr(ls));
               if (al->op == nir_op_isub &&
                   same_val(al->src[0].src.ssa, h_pc, wg) &&
                   match_push_const_u32(al->src[1].src.ssa, &off)) {
                  if (has_b && off != b_off)
                     return false;
                  has_b = true;
                  b_off = off;
                  flags |= F9_Y_HI_B;
                  continue;
               }
               if (al->op == nir_op_iadd) {
                  bool ok = false;
                  for (unsigned side = 0; side < 2; side++) {
                     nir_def *neg = strip_mov(al->src[1 - side].src.ssa);
                     if (same_val(al->src[side].src.ssa, h_pc, wg) && neg &&
                         nir_def_instr(neg)->type == nir_instr_type_alu &&
                         nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_ineg &&
                         match_push_const_u32(
                            nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa,
                            &off)) {
                        if (has_b && off != b_off)
                           return false;
                        has_b = true;
                        b_off = off;
                        flags |= F9_Y_HI_B;
                        ok = true;
                        break;
                     }
                  }
                  if (ok)
                     continue;
               }
            }
         }
         return false;
      } else {
         /* `2 < imin(x, y)`, `2 < x`, `2 < y`, `x < w - 3`, or `y < h - 3`. */
         if (is_const_i32(lhs, 2) && match_imin_xy(rhs, x, y)) {
            flags |= F9_X_LO_3 | F9_Y_LO_3;
            continue;
         }
         if (is_const_i32(lhs, 2) && strip_mov(rhs) == x) {
            flags |= F9_X_LO_3;
            continue;
         }
         if (is_const_i32(lhs, 2) && strip_mov(rhs) == y) {
            flags |= F9_Y_LO_3;
            continue;
         }
         nir_def *rs = strip_mov(rhs);
         if (!rs || nir_def_instr(rs)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *ar = nir_instr_as_alu(nir_def_instr(rs));
         bool is_sub3 = false;
         nir_def *dim = NULL;
         if (ar->op == nir_op_iadd) {
            if (is_const_i32(ar->src[1].src.ssa, -3)) {
               dim = ar->src[0].src.ssa;
               is_sub3 = true;
            } else if (is_const_i32(ar->src[0].src.ssa, -3)) {
               dim = ar->src[1].src.ssa;
               is_sub3 = true;
            }
         } else if (ar->op == nir_op_isub &&
                    is_const_i32(ar->src[1].src.ssa, 3)) {
            dim = ar->src[0].src.ssa;
            is_sub3 = true;
         }
         if (!is_sub3)
            return false;
         if (strip_mov(lhs) == x && same_val(dim, w_pc, wg)) {
            flags |= F9_X_HI_3;
            continue;
         }
         if (strip_mov(lhs) == y && same_val(dim, h_pc, wg)) {
            flags |= F9_Y_HI_3;
            continue;
         }
         return false;
      }
   }

   *flags_inout = flags;
   *has_b_inout = has_b;
   *border_pc_inout = b_off;
   return true;
}

struct fast9_addr_lin {
   int32_t ax, ay, ab, a1;
   int32_t sx, sy, sb, s1;
};

static bool
eval_fast9_addr(nir_def *def, nir_def *x, nir_def *y, uint16_t border_pc,
                uint16_t *stride_pc_inout, bool *has_stride,
                struct fast9_addr_lin *out, unsigned depth)
{
   if (depth > 20 || !def || def->bit_size != 32 || def->num_components != 1)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;

   memset(out, 0, sizeof(*out));
   if (def == x) {
      out->ax = 1;
      return true;
   }
   if (def == y) {
      out->ay = 1;
      return true;
   }

   int32_t c = 0;
   if (match_const_i32(def, &c)) {
      out->a1 = c;
      return true;
   }

   uint16_t pc = 0;
   if (match_push_const_u32(def, &pc)) {
      if (pc == border_pc) {
         out->ab = 1;
         return true;
      }
      if (!*has_stride) {
         *stride_pc_inout = pc;
         *has_stride = true;
      } else if (pc != *stride_pc_inout) {
         return false;
      }
      out->s1 = 1;
      return true;
   }

   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_ineg) {
      struct fast9_addr_lin a;
      if (!eval_fast9_addr(alu->src[0].src.ssa, x, y, border_pc,
                           stride_pc_inout, has_stride, &a, depth + 1))
         return false;
      out->ax = -a.ax;
      out->ay = -a.ay;
      out->ab = -a.ab;
      out->a1 = -a.a1;
      out->sx = -a.sx;
      out->sy = -a.sy;
      out->sb = -a.sb;
      out->s1 = -a.s1;
      return true;
   }

   if (alu->op == nir_op_iadd || alu->op == nir_op_isub) {
      struct fast9_addr_lin a, b;
      if (!eval_fast9_addr(alu->src[0].src.ssa, x, y, border_pc,
                           stride_pc_inout, has_stride, &a, depth + 1) ||
          !eval_fast9_addr(alu->src[1].src.ssa, x, y, border_pc,
                           stride_pc_inout, has_stride, &b, depth + 1))
         return false;
      int32_t sgn = (alu->op == nir_op_iadd) ? 1 : -1;
      out->ax = a.ax + sgn * b.ax;
      out->ay = a.ay + sgn * b.ay;
      out->ab = a.ab + sgn * b.ab;
      out->a1 = a.a1 + sgn * b.a1;
      out->sx = a.sx + sgn * b.sx;
      out->sy = a.sy + sgn * b.sy;
      out->sb = a.sb + sgn * b.sb;
      out->s1 = a.s1 + sgn * b.s1;
      return true;
   }

   if (alu->op == nir_op_ishl) {
      int32_t sh = 0;
      struct fast9_addr_lin a;
      if (!match_const_i32(alu->src[1].src.ssa, &sh) || sh < 0 || sh > 8 ||
          !eval_fast9_addr(alu->src[0].src.ssa, x, y, border_pc,
                           stride_pc_inout, has_stride, &a, depth + 1))
         return false;
      int32_t k = 1 << sh;
      out->ax = a.ax * k;
      out->ay = a.ay * k;
      out->ab = a.ab * k;
      out->a1 = a.a1 * k;
      out->sx = a.sx * k;
      out->sy = a.sy * k;
      out->sb = a.sb * k;
      out->s1 = a.s1 * k;
      return true;
   }

   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      struct fast9_addr_lin a, b;
      if (!eval_fast9_addr(alu->src[0].src.ssa, x, y, border_pc,
                           stride_pc_inout, has_stride, &a, depth + 1) ||
          !eval_fast9_addr(alu->src[1].src.ssa, x, y, border_pc,
                           stride_pc_inout, has_stride, &b, depth + 1))
         return false;
      bool a_const =
         (a.ax == 0 && a.ay == 0 && a.ab == 0 && a.sx == 0 && a.sy == 0 &&
          a.sb == 0 && a.s1 == 0);
      bool b_const =
         (b.ax == 0 && b.ay == 0 && b.ab == 0 && b.sx == 0 && b.sy == 0 &&
          b.sb == 0 && b.s1 == 0);
      if (a_const || b_const) {
         struct fast9_addr_lin v = a_const ? b : a;
         int32_t k = a_const ? a.a1 : b.a1;
         out->ax = v.ax * k;
         out->ay = v.ay * k;
         out->ab = v.ab * k;
         out->a1 = v.a1 * k;
         out->sx = v.sx * k;
         out->sy = v.sy * k;
         out->sb = v.sb * k;
         out->s1 = v.s1 * k;
         return true;
      }
      bool a_stride =
         (a.ax == 0 && a.ay == 0 && a.ab == 0 && a.a1 == 0 && a.sx == 0 &&
          a.sy == 0 && a.sb == 0 && a.s1 != 0);
      bool b_stride =
         (b.ax == 0 && b.ay == 0 && b.ab == 0 && b.a1 == 0 && b.sx == 0 &&
          b.sy == 0 && b.sb == 0 && b.s1 != 0);
      if (a_stride && b.sx == 0 && b.sy == 0 && b.sb == 0 && b.s1 == 0) {
         out->sx = b.ax * a.s1;
         out->sy = b.ay * a.s1;
         out->sb = b.ab * a.s1;
         out->s1 = b.a1 * a.s1;
         return true;
      }
      if (b_stride && a.sx == 0 && a.sy == 0 && a.sb == 0 && a.s1 == 0) {
         out->sx = a.ax * b.s1;
         out->sy = a.ay * b.s1;
         out->sb = a.ab * b.s1;
         out->s1 = a.a1 * b.s1;
         return true;
      }
   }

   return false;
}

/* Table of all classified readonly u8 loads from `src`:
 * `kind == -1` denotes center `(0, 0)`, `kind == 0..15` denotes `circle[kind]`.
 * Pre-CSE NIR may load the same readonly coordinate multiple times. */
#define FAST9_MAX_SRC_LOADS 128

struct fast9_src_table {
   nir_intrinsic_instr *loads[FAST9_MAX_SRC_LOADS];
   int8_t kinds[FAST9_MAX_SRC_LOADS];
   unsigned count;
   uint32_t seen_mask; /* bit 0 = center, bits 1..16 = circle[0..15] */
};

static bool
fast9_lookup_load(const struct fast9_src_table *tbl, nir_intrinsic_instr *intr,
                  int8_t *kind_out)
{
   for (unsigned i = 0; i < tbl->count; i++) {
      if (tbl->loads[i] == intr) {
         *kind_out = tbl->kinds[i];
         return true;
      }
   }
   return false;
}

static bool
match_u2u32_src_kind(nir_def *def, const struct fast9_src_table *tbl,
                     int8_t *kind_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_u2u32)
      return false;
   nir_def *s = strip_mov(alu->src[0].src.ssa);
   if (!s || s->bit_size != 8 || nir_def_instr(s)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(s));
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   return fast9_lookup_load(tbl, intr, kind_out);
}

/* Match `c + thresh` (sign = +1) or `c - thresh` (sign = -1). */
static bool
match_c_plus_minus_thresh(nir_def *def, const struct fast9_src_table *tbl,
                          int sign, uint16_t *thresh_pc_inout, bool *has_thresh)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (sign > 0) {
      if (alu->op != nir_op_iadd)
         return false;
      for (unsigned side = 0; side < 2; side++) {
         int8_t k = 0;
         uint16_t off = 0;
         if (match_u2u32_src_kind(alu->src[side].src.ssa, tbl, &k) &&
             k == -1 &&
             match_push_const_u32(alu->src[1 - side].src.ssa, &off)) {
            if (*has_thresh && off != *thresh_pc_inout)
               return false;
            *thresh_pc_inout = off;
            *has_thresh = true;
            return true;
         }
      }
      return false;
   }

   /* sign < 0: `isub(c, thresh)` or `iadd(c, ineg(thresh))` */
   if (alu->op == nir_op_isub) {
      int8_t k = 0;
      uint16_t off = 0;
      if (match_u2u32_src_kind(alu->src[0].src.ssa, tbl, &k) && k == -1 &&
          match_push_const_u32(alu->src[1].src.ssa, &off)) {
         if (*has_thresh && off != *thresh_pc_inout)
            return false;
         *thresh_pc_inout = off;
         *has_thresh = true;
         return true;
      }
      return false;
   }
   if (alu->op == nir_op_iadd) {
      for (unsigned side = 0; side < 2; side++) {
         int8_t k = 0;
         nir_def *neg = strip_mov(alu->src[1 - side].src.ssa);
         if (!match_u2u32_src_kind(alu->src[side].src.ssa, tbl, &k) ||
             k != -1 || !neg ||
             nir_def_instr(neg)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *na = nir_instr_as_alu(nir_def_instr(neg));
         uint16_t off = 0;
         if (na->op == nir_op_ineg &&
             match_push_const_u32(na->src[0].src.ssa, &off)) {
            if (*has_thresh && off != *thresh_pc_inout)
               return false;
            *thresh_pc_inout = off;
            *has_thresh = true;
            return true;
         }
      }
   }
   return false;
}

/* Match `s(p) = (p > c + t) ? 1 : ((p < c - t) ? -1 : 0)` across canonical
 * bcsel / b2i32 / undef-guarded algebraic forms, returning circle index `0..15`. */
static bool
match_fast9_sign(nir_def *def, const struct fast9_src_table *tbl,
                 uint16_t *thresh_pc_inout, bool *has_thresh,
                 int8_t *sample_kind_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *outer = nir_instr_as_alu(nir_def_instr(def));

   nir_def *c_hi = NULL, *c_lo = NULL;
   if (outer->op == nir_op_bcsel && is_const_i32(outer->src[1].src.ssa, -1)) {
      /* Form 1: bcsel(p < c - t, -1, bcsel(c + t < p, 1, 0) | b2i32(c + t < p)) */
      nir_def *inner_d = strip_mov(outer->src[2].src.ssa);
      if (!inner_d || nir_def_instr(inner_d)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *inner = nir_instr_as_alu(nir_def_instr(inner_d));
      if (inner->op == nir_op_bcsel &&
          is_const_i32(inner->src[1].src.ssa, 1) &&
          is_const_i32(inner->src[2].src.ssa, 0)) {
         c_hi = inner->src[0].src.ssa;
      } else if (inner->op == nir_op_b2i32) {
         c_hi = inner->src[0].src.ssa;
      } else {
         return false;
      }
      c_lo = outer->src[0].src.ssa;
   } else if (outer->op == nir_op_bcsel && is_const_i32(outer->src[1].src.ssa, 1)) {
      /* Form 2: bcsel(c + t < p, 1, bcsel(p < c - t, -1, 0) | ineg(b2i32(p < c - t))) */
      nir_def *inner_d = strip_mov(outer->src[2].src.ssa);
      if (!inner_d || nir_def_instr(inner_d)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *inner = nir_instr_as_alu(nir_def_instr(inner_d));
      if (inner->op == nir_op_bcsel &&
          is_const_i32(inner->src[1].src.ssa, -1) &&
          is_const_i32(inner->src[2].src.ssa, 0)) {
         c_lo = inner->src[0].src.ssa;
      } else if (inner->op == nir_op_ineg) {
         nir_def *b2 = strip_mov(inner->src[0].src.ssa);
         if (!b2 || nir_def_instr(b2)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(b2))->op != nir_op_b2i32)
            return false;
         c_lo = nir_instr_as_alu(nir_def_instr(b2))->src[0].src.ssa;
      } else {
         return false;
      }
      c_hi = outer->src[0].src.ssa;
   } else if (outer->op == nir_op_bcsel && is_const_i32(outer->src[2].src.ssa, 0)) {
      /* Form 5: bcsel(ior(c_lo, c_hi), inner, 0) where inner is either:
       *   - bcsel(c_lo, -1, bcsel(c_hi, 1, ANY))   (or commuted)
       *   - bcsel(c_lo, -1, 1)                     (after undef-folding inner bcsel)
       *   - bcsel(c_hi, 1, -1)
       * Since `ior(c_lo, c_hi)` guards `inner`, when one condition is false
       * the other is guaranteed true. */
      nir_def *g = strip_mov(outer->src[0].src.ssa);
      nir_def *in1 = strip_mov(outer->src[1].src.ssa);
      if (!g || !in1 || nir_def_instr(g)->type != nir_instr_type_alu ||
          nir_def_instr(in1)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *ga = nir_instr_as_alu(nir_def_instr(g));
      nir_alu_instr *a1 = nir_instr_as_alu(nir_def_instr(in1));
      if (ga->op != nir_op_ior || a1->op != nir_op_bcsel)
         return false;
      nir_def *g0 = strip_mov(ga->src[0].src.ssa);
      nir_def *g1 = strip_mov(ga->src[1].src.ssa);
      nir_def *a1c = strip_mov(a1->src[0].src.ssa);
      nir_def *in2 = strip_mov(a1->src[2].src.ssa);
      if (is_const_i32(a1->src[1].src.ssa, -1) && is_const_i32(in2, 1)) {
         c_lo = a1c;
         c_hi = (a1c == g0) ? g1 : (a1c == g1) ? g0 : NULL;
      } else if (is_const_i32(a1->src[1].src.ssa, 1) && is_const_i32(in2, -1)) {
         c_hi = a1c;
         c_lo = (a1c == g0) ? g1 : (a1c == g1) ? g0 : NULL;
      } else {
         if (!in2 || nir_def_instr(in2)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *a2 = nir_instr_as_alu(nir_def_instr(in2));
         if (a2->op != nir_op_bcsel)
            return false;
         if (is_const_i32(a1->src[1].src.ssa, -1) &&
             is_const_i32(a2->src[1].src.ssa, 1)) {
            c_lo = a1c;
            c_hi = strip_mov(a2->src[0].src.ssa);
         } else if (is_const_i32(a1->src[1].src.ssa, 1) &&
                    is_const_i32(a2->src[1].src.ssa, -1)) {
            c_hi = a1c;
            c_lo = strip_mov(a2->src[0].src.ssa);
         } else {
            return false;
         }
         if (!((g0 == c_lo && g1 == c_hi) || (g0 == c_hi && g1 == c_lo)))
            return false;
      }
      if (!c_lo || !c_hi)
         return false;
   } else if (outer->op == nir_op_isub) {
      /* Form 3: b2i32(c + t < p) - b2i32(p < c - t) */
      nir_def *s0 = strip_mov(outer->src[0].src.ssa);
      nir_def *s1 = strip_mov(outer->src[1].src.ssa);
      if (!s0 || !s1 || nir_def_instr(s0)->type != nir_instr_type_alu ||
          nir_def_instr(s1)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *a0 = nir_instr_as_alu(nir_def_instr(s0));
      nir_alu_instr *a1 = nir_instr_as_alu(nir_def_instr(s1));
      if (a0->op != nir_op_b2i32 || a1->op != nir_op_b2i32)
         return false;
      c_hi = a0->src[0].src.ssa;
      c_lo = a1->src[0].src.ssa;
   } else if (outer->op == nir_op_iadd) {
      /* Form 4: b2i32(c + t < p) + ineg(b2i32(p < c - t)) */
      for (unsigned side = 0; side < 2; side++) {
         nir_def *p = strip_mov(outer->src[side].src.ssa);
         nir_def *n = strip_mov(outer->src[1 - side].src.ssa);
         if (!p || !n || nir_def_instr(p)->type != nir_instr_type_alu ||
             nir_def_instr(n)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *pa = nir_instr_as_alu(nir_def_instr(p));
         nir_alu_instr *na = nir_instr_as_alu(nir_def_instr(n));
         if (pa->op != nir_op_b2i32 || na->op != nir_op_ineg)
            continue;
         nir_def *nb = strip_mov(na->src[0].src.ssa);
         if (!nb || nir_def_instr(nb)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(nb))->op != nir_op_b2i32)
            continue;
         c_hi = pa->src[0].src.ssa;
         c_lo = nir_instr_as_alu(nir_def_instr(nb))->src[0].src.ssa;
         break;
      }
      if (!c_hi || !c_lo)
         return false;
   } else {
      return false;
   }

   c_hi = strip_mov(c_hi);
   c_lo = strip_mov(c_lo);
   if (!c_hi || !c_lo || nir_def_instr(c_hi)->type != nir_instr_type_alu ||
       nir_def_instr(c_lo)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *ahi = nir_instr_as_alu(nir_def_instr(c_hi));
   nir_alu_instr *alo = nir_instr_as_alu(nir_def_instr(c_lo));
   if (ahi->op != nir_op_ilt || alo->op != nir_op_ilt)
      return false;

   int8_t k_hi = -2, k_lo = -2;
   if (!match_u2u32_src_kind(ahi->src[1].src.ssa, tbl, &k_hi) ||
       !match_u2u32_src_kind(alo->src[0].src.ssa, tbl, &k_lo) ||
       k_hi < 0 || k_hi != k_lo)
      return false;

   if (!match_c_plus_minus_thresh(ahi->src[0].src.ssa, tbl, +1,
                                  thresh_pc_inout, has_thresh) ||
       !match_c_plus_minus_thresh(alo->src[1].src.ssa, tbl, -1,
                                  thresh_pc_inout, has_thresh))
      return false;

   *sample_kind_out = k_hi;
   return true;
}

/* Match `s_a == 0 && s_b == 0` (expressed via `umax(a, b) == 0`,
 * `ior(a, b) == 0`, or conjunction of `== 0`), returning whether the
 * condition is true on reject. */
static bool
match_fast9_pair_zero(nir_def *cond, const struct fast9_src_table *tbl,
                      int8_t want_a, int8_t want_b,
                      uint16_t *thresh_pc_inout, bool *has_thresh,
                      bool *reject_on_true_out)
{
   cond = strip_mov(cond);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));

   bool neg = false;
   if (alu->op == nir_op_inot) {
      neg = true;
      cond = strip_mov(alu->src[0].src.ssa);
      if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
         return false;
      alu = nir_instr_as_alu(nir_def_instr(cond));
   }

   if (alu->op == nir_op_ieq || alu->op == nir_op_ine) {
      bool is_eq = (alu->op == nir_op_ieq) ^ neg;
      nir_def *nz = NULL;
      if (is_const_i32(alu->src[1].src.ssa, 0))
         nz = strip_mov(alu->src[0].src.ssa);
      else if (is_const_i32(alu->src[0].src.ssa, 0))
         nz = strip_mov(alu->src[1].src.ssa);
      if (!nz || nir_def_instr(nz)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *comb = nir_instr_as_alu(nir_def_instr(nz));
      if (comb->op != nir_op_umax && comb->op != nir_op_ior)
         return false;
      int8_t pa = -2, pb = -2;
      if (!match_fast9_sign(comb->src[0].src.ssa, tbl, thresh_pc_inout,
                            has_thresh, &pa) ||
          !match_fast9_sign(comb->src[1].src.ssa, tbl, thresh_pc_inout,
                            has_thresh, &pb))
         return false;
      if ((pa == want_a && pb == want_b) || (pa == want_b && pb == want_a)) {
         *reject_on_true_out = is_eq;
         return true;
      }
      return false;
   }

   /* Also accept `s_a == 0 && s_b == 0` / `s_a != 0 || s_b != 0`. */
   for (unsigned try_neg = 0; try_neg < 2; try_neg++) {
      struct bool_term terms[4];
      unsigned nt = 0;
      if (!collect_and_terms(cond, try_neg != 0, terms, &nt, 4) || nt != 2)
         continue;
      int8_t ps[2] = {-2, -2};
      bool ok = true;
      for (unsigned k = 0; k < 2; k++) {
         if (nir_def_instr(terms[k].def)->type != nir_instr_type_alu) {
            ok = false;
            break;
         }
         nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(terms[k].def));
         bool is_eq = (ca->op == nir_op_ieq && !terms[k].negated) ||
                      (ca->op == nir_op_ine && terms[k].negated);
         if (!is_eq) {
            ok = false;
            break;
         }
         nir_def *s = is_const_i32(ca->src[1].src.ssa, 0) ? ca->src[0].src.ssa
                      : is_const_i32(ca->src[0].src.ssa, 0) ? ca->src[1].src.ssa
                                                            : NULL;
         if (!s || !match_fast9_sign(s, tbl, thresh_pc_inout,
                                     has_thresh, &ps[k])) {
            ok = false;
            break;
         }
      }
      if (ok && ((ps[0] == want_a && ps[1] == want_b) ||
                 (ps[0] == want_b && ps[1] == want_a))) {
         *reject_on_true_out = ((try_neg == 0) ^ neg);
         return true;
      }
   }

   return false;
}

static bool
collect_fast9_sign_sum(nir_def *def, const struct fast9_src_table *tbl,
                       uint16_t *thresh_pc_inout, bool *has_thresh,
                       unsigned counts[16], unsigned depth)
{
   if (depth > 32 || !def)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;

   int8_t k = -2;
   if (match_fast9_sign(def, tbl, thresh_pc_inout, has_thresh, &k)) {
      if (k >= 0 && k < 16) {
         counts[k]++;
         return true;
      }
      return false;
   }

   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_iadd)
      return false;

   return collect_fast9_sign_sum(alu->src[0].src.ssa, tbl,
                                 thresh_pc_inout, has_thresh, counts,
                                 depth + 1) &&
          collect_fast9_sign_sum(alu->src[1].src.ssa, tbl,
                                 thresh_pc_inout, has_thresh, counts,
                                 depth + 1);
}

enum fast9_eval_kind {
   F9E_CONST,
   F9E_LANES, /* 4 byte lanes holding sample kind (-1 = center, 0..15 = circle, -2 = empty) */
   F9E_NEG_SAMPLE, /* -u2u32(sample) */
   F9E_DIFF,       /* diff_sign * (center - circle[sample]) */
};

struct fast9_eval_val {
   enum fast9_eval_kind kind;
   uint32_t cval;
   int8_t bytes[4];
   int8_t sample;
   int8_t diff_sign; /* +1 => center - p, -1 => p - center */
};

static bool
eval_fast9_expr(nir_def *def, nir_def *i_phi, uint32_t i_val,
                const struct fast9_src_table *tbl,
                struct fast9_eval_val *out, unsigned depth)
{
   if (depth > 32 || !def || def->num_components != 1)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;

   memset(out, 0, sizeof(*out));
   for (unsigned j = 0; j < 4; j++)
      out->bytes[j] = -2;

   if (def == i_phi) {
      out->kind = F9E_CONST;
      out->cval = i_val;
      return true;
   }

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const) {
      out->kind = F9E_CONST;
      out->cval = (uint32_t)nir_src_as_uint(nir_src_for_ssa(def));
      return true;
   }

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      int8_t k = -2;
      if (intr->intrinsic == nir_intrinsic_load_deref &&
          fast9_lookup_load(tbl, intr, &k)) {
         out->kind = F9E_LANES;
         out->bytes[0] = k;
         return true;
      }
      return false;
   }

   /* Follow if-merge phis by evaluating the preceding `nir_if` condition at `i_val`. */
   if (instr->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
      if (!prev || prev->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(prev);
      struct fast9_eval_val c;
      if (!eval_fast9_expr(nif->condition.ssa, i_phi, i_val, tbl, &c,
                           depth + 1) ||
          c.kind != F9E_CONST)
         return false;
      nir_block *want = c.cval ? nir_if_last_then_block(nif)
                               : nir_if_last_else_block(nif);
      nir_foreach_phi_src(ps, phi) {
         if (ps->pred == want)
            return eval_fast9_expr(ps->src.ssa, i_phi, i_val, tbl, out,
                                   depth + 1);
      }
      return false;
   }

   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);

   switch (alu->op) {
   case nir_op_u2u32:
   case nir_op_u2u16:
   case nir_op_u2u8: {
      struct fast9_eval_val a;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &a,
                           depth + 1))
         return false;
      if (a.kind == F9E_CONST) {
         uint32_t mask = def->bit_size == 8    ? 0xffu
                         : def->bit_size == 16 ? 0xffffu
                                               : 0xffffffffu;
         out->kind = F9E_CONST;
         out->cval = a.cval & mask;
         return true;
      }
      if (a.kind == F9E_LANES) {
         unsigned keep = def->bit_size / 8u;
         out->kind = F9E_LANES;
         for (unsigned j = 0; j < keep && j < 4; j++)
            out->bytes[j] = a.bytes[j];
         return true;
      }
      return false;
   }
   case nir_op_ineg: {
      struct fast9_eval_val a;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &a,
                           depth + 1))
         return false;
      if (a.kind == F9E_CONST) {
         out->kind = F9E_CONST;
         out->cval = (uint32_t)(-(int32_t)a.cval);
         return true;
      }
      if (a.kind == F9E_LANES && a.bytes[0] >= -1 && a.bytes[1] == -2 &&
          a.bytes[2] == -2 && a.bytes[3] == -2) {
         out->kind = F9E_NEG_SAMPLE;
         out->sample = a.bytes[0];
         return true;
      }
      if (a.kind == F9E_DIFF) {
         out->kind = F9E_DIFF;
         out->sample = a.sample;
         out->diff_sign = -a.diff_sign;
         return true;
      }
      return false;
   }
   case nir_op_isub:
   case nir_op_iadd: {
      struct fast9_eval_val a, b;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &a,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[1].src.ssa, i_phi, i_val, tbl, &b,
                           depth + 1))
         return false;
      if (a.kind == F9E_CONST && b.kind == F9E_CONST) {
         out->kind = F9E_CONST;
         out->cval = (alu->op == nir_op_iadd) ? (a.cval + b.cval)
                                              : (a.cval - b.cval);
         return true;
      }
      if (alu->op == nir_op_isub && a.kind == F9E_LANES && b.kind == F9E_LANES &&
          a.bytes[1] == -2 && a.bytes[2] == -2 && a.bytes[3] == -2 &&
          b.bytes[1] == -2 && b.bytes[2] == -2 && b.bytes[3] == -2) {
         if (a.bytes[0] == -1 && b.bytes[0] >= 0) {
            out->kind = F9E_DIFF;
            out->sample = b.bytes[0];
            out->diff_sign = +1;
            return true;
         }
         if (b.bytes[0] == -1 && a.bytes[0] >= 0) {
            out->kind = F9E_DIFF;
            out->sample = a.bytes[0];
            out->diff_sign = -1;
            return true;
         }
      }
      if (alu->op == nir_op_iadd) {
         for (unsigned side = 0; side < 2; side++) {
            const struct fast9_eval_val *pos = side == 0 ? &a : &b;
            const struct fast9_eval_val *neg = side == 0 ? &b : &a;
            if (pos->kind == F9E_LANES && neg->kind == F9E_NEG_SAMPLE &&
                pos->bytes[1] == -2 && pos->bytes[2] == -2 &&
                pos->bytes[3] == -2) {
               if (pos->bytes[0] == -1 && neg->sample >= 0) {
                  out->kind = F9E_DIFF;
                  out->sample = neg->sample;
                  out->diff_sign = +1;
                  return true;
               }
               if (neg->sample == -1 && pos->bytes[0] >= 0) {
                  out->kind = F9E_DIFF;
                  out->sample = pos->bytes[0];
                  out->diff_sign = -1;
                  return true;
               }
            }
         }
      }
      return false;
   }
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ishl:
   case nir_op_ushr:
   case nir_op_ult:
   case nir_op_ilt:
   case nir_op_uge:
   case nir_op_ige:
   case nir_op_ieq:
   case nir_op_ine: {
      struct fast9_eval_val a, b;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &a,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[1].src.ssa, i_phi, i_val, tbl, &b,
                           depth + 1))
         return false;
      if (a.kind == F9E_CONST && b.kind == F9E_CONST) {
         out->kind = F9E_CONST;
         switch (alu->op) {
         case nir_op_iand: out->cval = a.cval & b.cval; break;
         case nir_op_ior:  out->cval = a.cval | b.cval; break;
         case nir_op_ishl: out->cval = a.cval << (b.cval & 31u); break;
         case nir_op_ushr: out->cval = a.cval >> (b.cval & 31u); break;
         case nir_op_ult:  out->cval = a.cval < b.cval; break;
         case nir_op_ilt:  out->cval = (int32_t)a.cval < (int32_t)b.cval; break;
         case nir_op_uge:  out->cval = a.cval >= b.cval; break;
         case nir_op_ige:  out->cval = (int32_t)a.cval >= (int32_t)b.cval; break;
         case nir_op_ieq:  out->cval = a.cval == b.cval; break;
         case nir_op_ine:  out->cval = a.cval != b.cval; break;
         default:          return false;
         }
         return true;
      }
      if (alu->op == nir_op_ishl && a.kind == F9E_LANES && b.kind == F9E_CONST &&
          (b.cval & 7u) == 0 && b.cval <= 24u) {
         unsigned s = b.cval >> 3;
         out->kind = F9E_LANES;
         for (unsigned j = s; j < 4; j++)
            out->bytes[j] = a.bytes[j - s];
         return true;
      }
      if (alu->op == nir_op_ushr && a.kind == F9E_LANES && b.kind == F9E_CONST &&
          (b.cval & 7u) == 0 && b.cval <= 24u) {
         unsigned s = b.cval >> 3;
         out->kind = F9E_LANES;
         for (unsigned j = 0; j + s < 4; j++)
            out->bytes[j] = a.bytes[j + s];
         return true;
      }
      if (alu->op == nir_op_ior && a.kind == F9E_LANES && b.kind == F9E_LANES) {
         out->kind = F9E_LANES;
         for (unsigned j = 0; j < 4; j++) {
            if (a.bytes[j] >= -1 && b.bytes[j] >= -1)
               return false;
            out->bytes[j] = (a.bytes[j] >= -1) ? a.bytes[j] : b.bytes[j];
         }
         return true;
      }
      if (alu->op == nir_op_iand) {
         if (a.kind == F9E_LANES && b.kind == F9E_CONST && b.cval == 0xffu) {
            out->kind = F9E_LANES;
            out->bytes[0] = a.bytes[0];
            return true;
         }
         if (b.kind == F9E_LANES && a.kind == F9E_CONST && a.cval == 0xffu) {
            out->kind = F9E_LANES;
            out->bytes[0] = b.bytes[0];
            return true;
         }
      }
      return false;
   }
   case nir_op_pack_32_2x16_split: {
      struct fast9_eval_val lo, hi;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &lo,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[1].src.ssa, i_phi, i_val, tbl, &hi,
                           depth + 1))
         return false;
      if (lo.kind == F9E_LANES && hi.kind == F9E_LANES) {
         out->kind = F9E_LANES;
         out->bytes[0] = lo.bytes[0];
         out->bytes[1] = lo.bytes[1];
         out->bytes[2] = hi.bytes[0];
         out->bytes[3] = hi.bytes[1];
         return true;
      }
      return false;
   }
   case nir_op_pack_32_4x8_split: {
      out->kind = F9E_LANES;
      for (unsigned j = 0; j < 4; j++) {
         struct fast9_eval_val v;
         if (!eval_fast9_expr(alu->src[j].src.ssa, i_phi, i_val, tbl, &v,
                              depth + 1) ||
             v.kind != F9E_LANES)
            return false;
         out->bytes[j] = v.bytes[0];
      }
      return true;
   }
   case nir_op_ubitfield_extract: {
      struct fast9_eval_val word, off, bits;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &word,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[1].src.ssa, i_phi, i_val, tbl, &off,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[2].src.ssa, i_phi, i_val, tbl, &bits,
                           depth + 1))
         return false;
      if (word.kind == F9E_LANES && off.kind == F9E_CONST &&
          bits.kind == F9E_CONST && bits.cval == 8 &&
          (off.cval & 7u) == 0 && (off.cval >> 3) < 4 &&
          word.bytes[off.cval >> 3] >= -1) {
         out->kind = F9E_LANES;
         out->bytes[0] = word.bytes[off.cval >> 3];
         return true;
      }
      return false;
   }
   case nir_op_bitfield_insert: {
      struct fast9_eval_val base, ins, off, bits;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &base,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[1].src.ssa, i_phi, i_val, tbl, &ins,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[2].src.ssa, i_phi, i_val, tbl, &off,
                           depth + 1) ||
          !eval_fast9_expr(alu->src[3].src.ssa, i_phi, i_val, tbl, &bits,
                           depth + 1))
         return false;
      if (ins.kind == F9E_LANES && off.kind == F9E_CONST &&
          bits.kind == F9E_CONST && bits.cval == 8 &&
          (off.cval & 7u) == 0 && (off.cval >> 3) < 4) {
         out->kind = F9E_LANES;
         if (base.kind == F9E_LANES)
            memcpy(out->bytes, base.bytes, sizeof(out->bytes));
         out->bytes[off.cval >> 3] = ins.bytes[0];
         return true;
      }
      return false;
   }
   case nir_op_bcsel: {
      struct fast9_eval_val c;
      if (!eval_fast9_expr(alu->src[0].src.ssa, i_phi, i_val, tbl, &c,
                           depth + 1) ||
          c.kind != F9E_CONST)
         return false;
      return eval_fast9_expr(
         c.cval ? alu->src[1].src.ssa : alu->src[2].src.ssa, i_phi, i_val,
         tbl, out, depth + 1);
   }
   default:
      return false;
   }
}

/* Verify a block/cf_list contains only a single `dst_score[y * w + x] = 0`
 * store and no other side effects. */
static bool
match_fast9_zero_score_list(struct exec_list *list, nir_def *x, nir_def *y,
                            uint16_t w_off, const uint32_t wg[3],
                            uint32_t *score_set_inout,
                            uint32_t *score_bind_inout, bool *has_score,
                            nir_intrinsic_instr **store_out)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *n = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *blk = nir_cf_node_as_block(n);

   nir_intrinsic_instr *st = NULL;
   nir_foreach_instr(instr, blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_store_deref) {
         if (st)
            return false;
         st = intr;
      } else if (intr->intrinsic == nir_intrinsic_load_deref ||
                 intr->intrinsic == nir_intrinsic_deref_atomic) {
         return false;
      }
   }
   if (!st || nir_intrinsic_write_mask(st) != 1 ||
       !is_const_i32(st->src[1].ssa, 0) ||
       (nir_intrinsic_access(st) &
        (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL)))
      return false;

   uint32_t set = 0, bind = 0;
   nir_def *idx = NULL;
   uint16_t sp = 0;
   if (!match_ssbo_deref_access(nir_src_as_deref(st->src[0]), 32, &set, &bind,
                                &idx) ||
       !match_linear_index(idx, y, x, &sp, wg) || sp != w_off)
      return false;

   if (!*has_score) {
      *score_set_inout = set;
      *score_bind_inout = bind;
      *has_score = true;
   } else if (set != *score_set_inout || bind != *score_bind_inout) {
      return false;
   }

   *store_out = st;
   return true;
}

/* Match the 24-step FAST-9 arc scoring loop. */
static bool
match_fast9_arc_loop(nir_loop *loop, const struct fast9_src_table *tbl,
                     uint16_t thresh_pc, const uint32_t wg[3],
                     nir_def **best_phi_out, nir_def **found_phi_out)
{
   nir_block *hdr = nir_loop_first_block(loop);

   nir_phi_instr *i_phi = NULL, *arc_min_phi = NULL, *best_phi = NULL;
   nir_phi_instr *rem_phi = NULL, *found_phi = NULL;
   nir_phi_instr *zero_phis[2] = {NULL, NULL};
   unsigned zero_cnt = 0;

   /* Loop body: `exit_if` (`if (i >= 24) break;`), plus optionally one
    * `diff_if` (`if (0 < sum_s)`) selecting between `p[i&15] - c` and `c - p[i&15]`. */
   nir_if *exit_if = NULL, *diff_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, &loop->body) {
      if (n->type == nir_cf_node_if) {
         if (!exit_if)
            exit_if = nir_cf_node_as_if(n);
         else if (!diff_if)
            diff_if = nir_cf_node_as_if(n);
         else
            return false;
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!exit_if)
      return false;

   bool exit_on_true = false;
   if (cf_list_is_single_break(&exit_if->then_list) &&
       cf_list_is_empty_block(&exit_if->else_list)) {
      exit_on_true = true;
   } else if (cf_list_is_single_break(&exit_if->else_list) &&
              cf_list_is_empty_block(&exit_if->then_list)) {
      exit_on_true = false;
   } else {
      return false;
   }

   struct bool_term ex_terms[2];
   unsigned ex_n = 0;
   if (!collect_and_terms(exit_if->condition.ssa, !exit_on_true, ex_terms,
                          &ex_n, 2) ||
       ex_n != 1)
      return false;
   nir_def *ex_lhs = NULL, *ex_rhs = NULL;
   bool ex_lt = false, ex_u = false;
   if (!decompose_cmp_term(ex_terms[0], &ex_lhs, &ex_rhs, &ex_lt, &ex_u) ||
       ex_lt || !is_const_i32(ex_rhs, 24))
      return false;
   nir_def *i_def_expected = strip_mov(ex_lhs);

   nir_def *inits[5] = {NULL}, *latches[5] = {NULL};
   nir_phi_instr *phis[5] = {NULL};
   unsigned nphi = 0;
   nir_foreach_phi(p, hdr) {
      if (nphi >= 5)
         return false;
      nir_def *init = NULL, *latch = NULL;
      nir_foreach_phi_src(ps, p) {
         if (ps->pred->index < hdr->index) {
            if (init)
               return false;
            init = ps->src.ssa;
         } else {
            if (latch)
               return false;
            latch = ps->src.ssa;
         }
      }
      if (!init || !latch)
         return false;
      phis[nphi] = p;
      inits[nphi] = init;
      latches[nphi] = latch;
      nphi++;
   }
   if (nphi != 5)
      return false;

   nir_def *i_latch = NULL, *arc_min_latch = NULL, *best_latch = NULL;
   nir_def *rem_latch = NULL, *found_latch = NULL;
   nir_def *thresh_def = NULL;

   for (unsigned k = 0; k < 5; k++) {
      uint16_t pc = 0;
      if (phis[k]->def.bit_size == 1 && is_const_bool(inits[k], false)) {
         if (found_phi)
            return false;
         found_phi = phis[k];
         found_latch = latches[k];
      } else if (is_const_i32(inits[k], 9)) {
         if (rem_phi)
            return false;
         rem_phi = phis[k];
         rem_latch = latches[k];
      } else if (match_push_const_u32(inits[k], &pc) && pc == thresh_pc) {
         if (arc_min_phi)
            return false;
         arc_min_phi = phis[k];
         arc_min_latch = latches[k];
         thresh_def = inits[k];
      } else if (is_const_i32(inits[k], 0)) {
         if (&phis[k]->def == i_def_expected) {
            i_phi = phis[k];
            i_latch = latches[k];
         } else {
            if (zero_cnt >= 2)
               return false;
            zero_phis[zero_cnt++] = phis[k];
            best_phi = phis[k];
            best_latch = latches[k];
         }
      } else {
         return false;
      }
   }
   if (!i_phi || !arc_min_phi || !best_phi || !rem_phi || !found_phi ||
       zero_cnt != 1)
      return false;

   /* Inspect `found_latch`: `rem_zero ? true : found` (or `rem_zero || found`). */
   nir_def *rem_zero = NULL;
   bool rem_zero_neg = false;
   nir_def *fl = strip_mov(found_latch);
   if (!fl || nir_def_instr(fl)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *fl_alu = nir_instr_as_alu(nir_def_instr(fl));
   if (fl_alu->op == nir_op_bcsel) {
      if (is_const_bool(fl_alu->src[1].src.ssa, true) &&
          strip_mov(fl_alu->src[2].src.ssa) == &found_phi->def) {
         rem_zero = strip_mov(fl_alu->src[0].src.ssa);
         rem_zero_neg = false;
      } else if (is_const_bool(fl_alu->src[2].src.ssa, true) &&
                 strip_mov(fl_alu->src[1].src.ssa) == &found_phi->def) {
         rem_zero = strip_mov(fl_alu->src[0].src.ssa);
         rem_zero_neg = true;
      } else {
         return false;
      }
   } else if (fl_alu->op == nir_op_ior) {
      if (strip_mov(fl_alu->src[0].src.ssa) == &found_phi->def)
         rem_zero = strip_mov(fl_alu->src[1].src.ssa);
      else if (strip_mov(fl_alu->src[1].src.ssa) == &found_phi->def)
         rem_zero = strip_mov(fl_alu->src[0].src.ssa);
      else
         return false;
   } else {
      return false;
   }

   /* Verify `rem_zero` is `rem_next == 0` (or `!= 0` when negated). */
   if (!rem_zero || nir_def_instr(rem_zero)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *rz_alu = nir_instr_as_alu(nir_def_instr(rem_zero));
   bool rz_is_eq = (rz_alu->op == nir_op_ieq && !rem_zero_neg) ||
                   (rz_alu->op == nir_op_ine && rem_zero_neg);
   if (!rz_is_eq)
      return false;
   nir_def *rem_next = is_const_i32(rz_alu->src[1].src.ssa, 0)
                          ? strip_mov(rz_alu->src[0].src.ssa)
                       : is_const_i32(rz_alu->src[0].src.ssa, 0)
                          ? strip_mov(rz_alu->src[1].src.ssa)
                          : NULL;
   if (!rem_next || nir_def_instr(rem_next)->type != nir_instr_type_alu)
      return false;

   /* Verify `rem_next == pass ? (rem - 1) : 9`. */
   nir_alu_instr *rn_alu = nir_instr_as_alu(nir_def_instr(rem_next));
   if (rn_alu->op != nir_op_bcsel)
      return false;
   nir_def *pass_cond = strip_mov(rn_alu->src[0].src.ssa);
   nir_def *rem_dec = NULL;
   bool pass_neg = false;
   if (is_const_i32(rn_alu->src[2].src.ssa, 9)) {
      rem_dec = strip_mov(rn_alu->src[1].src.ssa);
      pass_neg = false;
   } else if (is_const_i32(rn_alu->src[1].src.ssa, 9)) {
      rem_dec = strip_mov(rn_alu->src[2].src.ssa);
      pass_neg = true;
   } else {
      return false;
   }
   /* Check `rem_dec == rem - 1`. */
   if (!rem_dec || nir_def_instr(rem_dec)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *rd_alu = nir_instr_as_alu(nir_def_instr(rem_dec));
   bool rd_ok =
      (rd_alu->op == nir_op_iadd &&
       ((strip_mov(rd_alu->src[0].src.ssa) == &rem_phi->def &&
         is_const_i32(rd_alu->src[1].src.ssa, -1)) ||
        (strip_mov(rd_alu->src[1].src.ssa) == &rem_phi->def &&
         is_const_i32(rd_alu->src[0].src.ssa, -1)))) ||
      (rd_alu->op == nir_op_isub &&
       strip_mov(rd_alu->src[0].src.ssa) == &rem_phi->def &&
       is_const_i32(rd_alu->src[1].src.ssa, 1));
   if (!rd_ok)
      return false;

   /* Verify `rem_latch == rem_zero ? 9 : rem_next`. */
   {
      nir_def *rl = strip_mov(rem_latch);
      if (!rl || nir_def_instr(rl)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *rla = nir_instr_as_alu(nir_def_instr(rl));
      if (rla->op != nir_op_bcsel ||
          strip_mov(rla->src[0].src.ssa) != rem_zero)
         return false;
      nir_def *t = rem_zero_neg ? rla->src[2].src.ssa : rla->src[1].src.ssa;
      nir_def *e = rem_zero_neg ? rla->src[1].src.ssa : rla->src[2].src.ssa;
      if (!is_const_i32(t, 9) || strip_mov(e) != rem_next)
         return false;
   }

   /* Verify `i_latch == (rem_zero ? (i - 8) : i) + 1` or
    * `rem_zero ? (i - 7) : (i + 1)` or `i + (rem_zero ? -7 : 1)`. */
   {
      nir_def *il = strip_mov(i_latch);
      if (!il || nir_def_instr(il)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *ila = nir_instr_as_alu(nir_def_instr(il));
      bool i_ok = false;
      if (ila->op == nir_op_iadd) {
         /* Form 1: `(rz ? i - 8 : i) + 1` */
         nir_def *base_i = is_const_i32(ila->src[1].src.ssa, 1)
                              ? strip_mov(ila->src[0].src.ssa)
                           : is_const_i32(ila->src[0].src.ssa, 1)
                              ? strip_mov(ila->src[1].src.ssa)
                              : NULL;
         if (base_i && nir_def_instr(base_i)->type == nir_instr_type_alu) {
            nir_alu_instr *bi = nir_instr_as_alu(nir_def_instr(base_i));
            if (bi->op == nir_op_bcsel &&
                strip_mov(bi->src[0].src.ssa) == rem_zero) {
               nir_def *t = strip_mov(rem_zero_neg ? bi->src[2].src.ssa
                                                   : bi->src[1].src.ssa);
               nir_def *e = strip_mov(rem_zero_neg ? bi->src[1].src.ssa
                                                   : bi->src[2].src.ssa);
               if (e == &i_phi->def && t &&
                   nir_def_instr(t)->type == nir_instr_type_alu) {
                  nir_alu_instr *ta = nir_instr_as_alu(nir_def_instr(t));
                  if ((ta->op == nir_op_iadd &&
                       ((strip_mov(ta->src[0].src.ssa) == &i_phi->def &&
                         is_const_i32(ta->src[1].src.ssa, -8)) ||
                        (strip_mov(ta->src[1].src.ssa) == &i_phi->def &&
                         is_const_i32(ta->src[0].src.ssa, -8)))) ||
                      (ta->op == nir_op_isub &&
                       strip_mov(ta->src[0].src.ssa) == &i_phi->def &&
                       is_const_i32(ta->src[1].src.ssa, 8)))
                     i_ok = true;
               }
            }
         }
         /* Form 2: `i + (rz ? -7 : 1)` */
         if (!i_ok) {
            nir_def *step =
               strip_mov(ila->src[0].src.ssa) == &i_phi->def
                  ? strip_mov(ila->src[1].src.ssa)
               : strip_mov(ila->src[1].src.ssa) == &i_phi->def
                  ? strip_mov(ila->src[0].src.ssa)
                  : NULL;
            if (step && nir_def_instr(step)->type == nir_instr_type_alu) {
               nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(step));
               if (sa->op == nir_op_bcsel &&
                   strip_mov(sa->src[0].src.ssa) == rem_zero) {
                  nir_def *t = rem_zero_neg ? sa->src[2].src.ssa : sa->src[1].src.ssa;
                  nir_def *e = rem_zero_neg ? sa->src[1].src.ssa : sa->src[2].src.ssa;
                  if (is_const_i32(t, -7) && is_const_i32(e, 1))
                     i_ok = true;
               }
            }
         }
      } else if (ila->op == nir_op_bcsel &&
                 strip_mov(ila->src[0].src.ssa) == rem_zero) {
         /* Form 3: `rz ? (i - 7) : (i + 1)` */
         nir_def *t = strip_mov(rem_zero_neg ? ila->src[2].src.ssa
                                             : ila->src[1].src.ssa);
         nir_def *e = strip_mov(rem_zero_neg ? ila->src[1].src.ssa
                                             : ila->src[2].src.ssa);
         if (t && e && nir_def_instr(t)->type == nir_instr_type_alu &&
             nir_def_instr(e)->type == nir_instr_type_alu) {
            nir_alu_instr *ta = nir_instr_as_alu(nir_def_instr(t));
            nir_alu_instr *ea = nir_instr_as_alu(nir_def_instr(e));
            bool t_ok = (ta->op == nir_op_iadd &&
                         ((strip_mov(ta->src[0].src.ssa) == &i_phi->def &&
                           is_const_i32(ta->src[1].src.ssa, -7)) ||
                          (strip_mov(ta->src[1].src.ssa) == &i_phi->def &&
                           is_const_i32(ta->src[0].src.ssa, -7)))) ||
                        (ta->op == nir_op_isub &&
                         strip_mov(ta->src[0].src.ssa) == &i_phi->def &&
                         is_const_i32(ta->src[1].src.ssa, 7));
            bool e_ok = (ea->op == nir_op_iadd &&
                         ((strip_mov(ea->src[0].src.ssa) == &i_phi->def &&
                           is_const_i32(ea->src[1].src.ssa, 1)) ||
                          (strip_mov(ea->src[1].src.ssa) == &i_phi->def &&
                           is_const_i32(ea->src[0].src.ssa, 1))));
            if (t_ok && e_ok)
               i_ok = true;
         }
      }
      if (!i_ok)
         return false;
   }

   /* Verify `arc_min_latch == rem_zero ? thresh : min_cand` and extract `min_cand`. */
   nir_def *min_cand = NULL;
   {
      nir_def *al = strip_mov(arc_min_latch);
      if (!al || nir_def_instr(al)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *ala = nir_instr_as_alu(nir_def_instr(al));
      if (ala->op != nir_op_bcsel || strip_mov(ala->src[0].src.ssa) != rem_zero)
         return false;
      nir_def *t = rem_zero_neg ? ala->src[2].src.ssa : ala->src[1].src.ssa;
      nir_def *e = rem_zero_neg ? ala->src[1].src.ssa : ala->src[2].src.ssa;
      if (!same_val(t, thresh_def, wg))
         return false;
      min_cand = strip_mov(e);
   }

   /* Verify `best_latch == rem_zero ? imax(best, min_cand) : best`. */
   {
      nir_def *bl = strip_mov(best_latch);
      if (!bl || nir_def_instr(bl)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *bla = nir_instr_as_alu(nir_def_instr(bl));
      if (bla->op != nir_op_bcsel || strip_mov(bla->src[0].src.ssa) != rem_zero)
         return false;
      nir_def *t = strip_mov(rem_zero_neg ? bla->src[2].src.ssa : bla->src[1].src.ssa);
      nir_def *e = strip_mov(rem_zero_neg ? bla->src[1].src.ssa : bla->src[2].src.ssa);
      if (e != &best_phi->def || !t || nir_def_instr(t)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *mx = nir_instr_as_alu(nir_def_instr(t));
      if (mx->op != nir_op_imax)
         return false;
      bool mx_ok =
         (strip_mov(mx->src[0].src.ssa) == &best_phi->def &&
          strip_mov(mx->src[1].src.ssa) == min_cand) ||
         (strip_mov(mx->src[1].src.ssa) == &best_phi->def &&
          strip_mov(mx->src[0].src.ssa) == min_cand);
      if (!mx_ok)
         return false;
   }

   /* Verify `pass_cond` is `thresh < diff` (or `!(diff <= thresh)`), extract `diff`. */
   struct bool_term pt[2];
   unsigned ptn = 0;
   if (!collect_and_terms(pass_cond, pass_neg, pt, &ptn, 2) || ptn != 1)
      return false;
   nir_def *plhs = NULL, *prhs = NULL;
   bool plt = false, pu = false;
   if (!decompose_cmp_term(pt[0], &plhs, &prhs, &plt, &pu) || pu || !plt ||
       !same_val(plhs, thresh_def, wg))
      return false;
   nir_def *diff = strip_mov(prhs);

   /* Verify `min_cand == pass ? (rem == 9 ? diff : imin(arc_min, diff)) : thresh`. */
   if (!min_cand || nir_def_instr(min_cand)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mca = nir_instr_as_alu(nir_def_instr(min_cand));
   if (mca->op != nir_op_bcsel || strip_mov(mca->src[0].src.ssa) != pass_cond)
      return false;
   nir_def *min_val = strip_mov(pass_neg ? mca->src[2].src.ssa : mca->src[1].src.ssa);
   nir_def *min_else = pass_neg ? mca->src[1].src.ssa : mca->src[2].src.ssa;
   if (!same_val(min_else, thresh_def, wg) || !min_val ||
       nir_def_instr(min_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mva = nir_instr_as_alu(nir_def_instr(min_val));
   if (mva->op != nir_op_bcsel)
      return false;
   nir_def *rem9 = strip_mov(mva->src[0].src.ssa);
   if (!rem9 || nir_def_instr(rem9)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *r9a = nir_instr_as_alu(nir_def_instr(rem9));
   bool r9_eq = false;
   if (r9a->op == nir_op_ieq)
      r9_eq = true;
   else if (r9a->op == nir_op_ine)
      r9_eq = false;
   else
      return false;
   bool r9_match =
      (strip_mov(r9a->src[0].src.ssa) == &rem_phi->def &&
       is_const_i32(r9a->src[1].src.ssa, 9)) ||
      (strip_mov(r9a->src[1].src.ssa) == &rem_phi->def &&
       is_const_i32(r9a->src[0].src.ssa, 9));
   if (!r9_match)
      return false;
   nir_def *mv_t = strip_mov(r9_eq ? mva->src[1].src.ssa : mva->src[2].src.ssa);
   nir_def *mv_e = strip_mov(r9_eq ? mva->src[2].src.ssa : mva->src[1].src.ssa);
   if (mv_t != diff || !mv_e || nir_def_instr(mv_e)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *imn = nir_instr_as_alu(nir_def_instr(mv_e));
   if (imn->op != nir_op_imin)
      return false;
   bool imn_ok =
      (strip_mov(imn->src[0].src.ssa) == &arc_min_phi->def &&
       strip_mov(imn->src[1].src.ssa) == diff) ||
      (strip_mov(imn->src[1].src.ssa) == &arc_min_phi->def &&
       strip_mov(imn->src[0].src.ssa) == diff);
   if (!imn_ok)
      return false;

   /* Verify `diff == (0 < sum_s) ? (p_i - c) : (c - p_i)`, either as a flat
    * `bcsel` or as an `if (0 < sum_s)` merge phi with binary lookup trees. */
   if (!diff)
      return false;
   nir_def *diff_cond = NULL, *true_arm = NULL, *false_arm = NULL;
   if (nir_def_instr(diff)->type == nir_instr_type_alu) {
      nir_alu_instr *da = nir_instr_as_alu(nir_def_instr(diff));
      if (da->op != nir_op_bcsel)
         return false;
      diff_cond = da->src[0].src.ssa;
      true_arm = strip_mov(da->src[1].src.ssa);
      false_arm = strip_mov(da->src[2].src.ssa);
   } else if (nir_def_instr(diff)->type == nir_instr_type_phi && diff_if) {
      nir_phi_instr *dp = nir_instr_as_phi(nir_def_instr(diff));
      nir_cf_node *prev = nir_cf_node_prev(&dp->instr.block->cf_node);
      if (prev != &diff_if->cf_node)
         return false;
      diff_cond = diff_if->condition.ssa;
      nir_block *t_blk = nir_if_last_then_block(diff_if);
      nir_block *e_blk = nir_if_last_else_block(diff_if);
      nir_foreach_phi_src(ps, dp) {
         if (ps->pred == t_blk)
            true_arm = strip_mov(ps->src.ssa);
         else if (ps->pred == e_blk)
            false_arm = strip_mov(ps->src.ssa);
      }
   } else {
      return false;
   }
   if (!diff_cond || !true_arm || !false_arm)
      return false;

   struct bool_term dt[2];
   unsigned dtn = 0;
   if (!collect_and_terms(diff_cond, false, dt, &dtn, 2) || dtn != 1)
      return false;
   nir_def *dlhs = NULL, *drhs = NULL;
   bool dlt = false, du = false;
   if (!decompose_cmp_term(dt[0], &dlhs, &drhs, &dlt, &du) || du)
      return false;
   nir_def *sum_s_def = NULL;
   bool pos_is_true = false;
   if (dlt && is_const_i32(dlhs, 0)) {
      /* `0 < sum_s` */
      sum_s_def = strip_mov(drhs);
      pos_is_true = true;
   } else if (!dlt && is_const_i32(dlhs, 0)) {
      /* `0 >= sum_s` (i.e. `sum_s <= 0`) */
      sum_s_def = strip_mov(drhs);
      pos_is_true = false;
   } else {
      return false;
   }

   /* Verify `sum_s_def` sums all 16 circle signs `s[0..15]` once each. */
   unsigned sign_counts[16] = {0};
   uint16_t check_thresh = thresh_pc;
   bool has_t = true;
   if (!collect_fast9_sign_sum(sum_s_def, tbl, &check_thresh,
                               &has_t, sign_counts, 0))
      return false;
   for (unsigned k = 0; k < 16; k++) {
      if (sign_counts[k] != 1)
         return false;
   }

   nir_def *pos_arm = pos_is_true ? true_arm : false_arm;
   nir_def *neg_arm = pos_is_true ? false_arm : true_arm;

   /* Symbolically verify for every `k = 0..15`:
    *   `neg_arm` evaluates to `+(center - circle[k])`
    *   `pos_arm` evaluates to `-(center - circle[k])` (`circle[k] - center`). */
   for (unsigned k = 0; k < 16; k++) {
      struct fast9_eval_val vn, vp;
      if (!eval_fast9_expr(neg_arm, &i_phi->def, k, tbl, &vn, 0) ||
          vn.kind != F9E_DIFF || vn.sample != (int8_t)k || vn.diff_sign != +1)
         return false;
      if (!eval_fast9_expr(pos_arm, &i_phi->def, k, tbl, &vp, 0) ||
          vp.kind != F9E_DIFF || vp.sample != (int8_t)k || vp.diff_sign != -1)
         return false;
   }

   *best_phi_out = &best_phi->def;
   *found_phi_out = &found_phi->def;
   return true;
}

static bool
match_fast9_u8(nir_shader *s, uint32_t set_layout_count,
               struct vk_descriptor_set_layout *const *set_layouts,
               struct hk_2d_stencil_plan *plan)
{
   const uint32_t *wg = plan->orig_wg_size;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   /* Top-level body: single outer `y` grid-stride loop. */
   nir_loop *y_loop = NULL;
   foreach_list_typed(nir_cf_node, n, node, &impl->body) {
      if (n->type == nir_cf_node_loop) {
         if (y_loop)
            return false;
         y_loop = nir_cf_node_as_loop(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!y_loop)
      return false;

   nir_def *y = NULL, *h_pc = NULL;
   nir_if *y_exit = NULL;
   if (!match_fast9_grid_loop(y_loop, 1, wg, &y, &h_pc, &y_exit))
      return false;

   /* Inside `y_loop`: `y_exit` followed by single inner `x` grid-stride loop. */
   nir_loop *x_loop = NULL;
   foreach_list_typed(nir_cf_node, n, node, &y_loop->body) {
      if (n->type == nir_cf_node_if) {
         if (nir_cf_node_as_if(n) != y_exit)
            return false;
      } else if (n->type == nir_cf_node_loop) {
         if (x_loop)
            return false;
         x_loop = nir_cf_node_as_loop(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!x_loop)
      return false;

   nir_def *x = NULL, *w_pc = NULL;
   nir_if *x_exit = NULL;
   if (!match_fast9_grid_loop(x_loop, 0, wg, &x, &w_pc, &x_exit))
      return false;

   uint16_t w_off = 0, h_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off))
      return false;

   /* Inside `x_loop`: `x_exit` followed by `border_if`. */
   nir_if *border_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, &x_loop->body) {
      if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         if (nif == x_exit)
            continue;
         if (border_if)
            return false;
         border_if = nif;
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!border_if)
      return false;

   uint16_t border_pc = 0;
   bool has_border = false;
   unsigned border_flags = 0;
   struct exec_list *border_list = NULL, *int_list = NULL;
   if (collect_fast9_border_flags(border_if->condition.ssa, true, x, y, w_pc,
                                  h_pc, wg, &border_flags, &has_border,
                                  &border_pc)) {
      border_list = &border_if->then_list;
      int_list = &border_if->else_list;
   } else if (collect_fast9_border_flags(border_if->condition.ssa, false, x, y,
                                         w_pc, h_pc, wg, &border_flags,
                                         &has_border, &border_pc)) {
      border_list = &border_if->else_list;
      int_list = &border_if->then_list;
   } else {
      return false;
   }

   uint32_t score_set = 0, score_bind = 0;
   bool has_score = false;
   nir_intrinsic_instr *zero_stores[5] = {NULL};
   unsigned num_zero_stores = 0;
   if (!match_fast9_zero_score_list(border_list, x, y, w_off, wg, &score_set,
                                    &score_bind, &has_score,
                                    &zero_stores[num_zero_stores++]))
      return false;

   /* Accept either a single merged border `if` (all 8 border bits covered) or
    * two nested border `if`s (e.g. radius-3 check followed by `border` check),
    * both writing `dst_score[idx] = 0`. */
   if (border_flags != F9_BORDER_ALL || !has_border) {
      nir_if *border_if2 = NULL;
      foreach_list_typed(nir_cf_node, n, node, int_list) {
         if (n->type == nir_cf_node_if) {
            if (border_if2)
               return false;
            border_if2 = nir_cf_node_as_if(n);
         } else if (n->type != nir_cf_node_block) {
            return false;
         }
      }
      if (!border_if2)
         return false;
      struct exec_list *b2_list = NULL, *i2_list = NULL;
      if (collect_fast9_border_flags(border_if2->condition.ssa, true, x, y,
                                     w_pc, h_pc, wg, &border_flags, &has_border,
                                     &border_pc)) {
         b2_list = &border_if2->then_list;
         i2_list = &border_if2->else_list;
      } else if (collect_fast9_border_flags(border_if2->condition.ssa, false, x,
                                            y, w_pc, h_pc, wg, &border_flags,
                                            &has_border, &border_pc)) {
         b2_list = &border_if2->else_list;
         i2_list = &border_if2->then_list;
      } else {
         return false;
      }
      if (border_flags != F9_BORDER_ALL || !has_border)
         return false;
      if (!match_fast9_zero_score_list(b2_list, x, y, w_off, wg, &score_set,
                                       &score_bind, &has_score,
                                       &zero_stores[num_zero_stores++]))
         return false;
      int_list = i2_list;
   }

   /* Inside `int_list`: single `ew_if`. */
   nir_if *ew_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, int_list) {
      if (n->type == nir_cf_node_if) {
         if (ew_if)
            return false;
         ew_if = nir_cf_node_as_if(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!ew_if)
      return false;

   /* Collect all `src` u8 loads across the shader and classify each one as
    * center (`-1`) or `circle[k]` (`0..15`). Duplicate readonly loads of the
    * same coordinate (pre-CSE) are permitted. */
   struct fast9_src_table src_tbl = {0};
   uint32_t src_set = 0, src_bind = 0;
   bool has_src = false;
   uint16_t stride_pc = 0;
   bool has_stride = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref)
            continue;
         if (nir_intrinsic_access(intr) &
             (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
            return false;
         uint32_t set = 0, bind = 0;
         nir_def *idx = NULL;
         if (!match_ssbo_deref_access(nir_src_as_deref(intr->src[0]), 8, &set,
                                      &bind, &idx))
            return false;
         if (!has_src) {
            src_set = set;
            src_bind = bind;
            has_src = true;
         } else if (set != src_set || bind != src_bind) {
            return false;
         }
         struct fast9_addr_lin lin;
         if (!eval_fast9_addr(idx, x, y, border_pc, &stride_pc, &has_stride,
                              &lin, 0))
            return false;
         if (lin.ax != 1 || lin.ay != 0 || lin.ab != 1 || lin.sx != 0 ||
             lin.sy != 1 || lin.sb != 1)
            return false;
         int8_t kind = -2;
         if (lin.a1 == 0 && lin.s1 == 0) {
            kind = -1;
            src_tbl.seen_mask |= 1u;
         } else {
            for (unsigned k = 0; k < 16; k++) {
               if (lin.a1 == fast9_dx[k] && lin.s1 == fast9_dy[k]) {
                  kind = (int8_t)k;
                  src_tbl.seen_mask |= 1u << (k + 1);
                  break;
               }
            }
         }
         if (kind < -1 || src_tbl.count >= FAST9_MAX_SRC_LOADS)
            return false;
         src_tbl.loads[src_tbl.count] = intr;
         src_tbl.kinds[src_tbl.count] = kind;
         src_tbl.count++;
      }
   }
   if (src_tbl.seen_mask != 0x1ffffu || !has_stride)
      return false;

   /* Match `ew_if`: rejects when East (4) and West (12) are both 0. */
   uint16_t thresh_pc = 0;
   bool has_thresh = false;
   bool ew_rej_true = false;
   if (!match_fast9_pair_zero(ew_if->condition.ssa, &src_tbl, 4, 12,
                              &thresh_pc, &has_thresh, &ew_rej_true))
      return false;

   struct exec_list *ew_rej = ew_rej_true ? &ew_if->then_list : &ew_if->else_list;
   struct exec_list *ew_pass = ew_rej_true ? &ew_if->else_list : &ew_if->then_list;
   if (!match_fast9_zero_score_list(ew_rej, x, y, w_off, wg, &score_set,
                                    &score_bind, &has_score,
                                    &zero_stores[num_zero_stores++]))
      return false;

   /* Inside `ew_pass`: single `ns_if`. */
   nir_if *ns_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, ew_pass) {
      if (n->type == nir_cf_node_if) {
         if (ns_if)
            return false;
         ns_if = nir_cf_node_as_if(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!ns_if)
      return false;

   bool ns_rej_true = false;
   if (!match_fast9_pair_zero(ns_if->condition.ssa, &src_tbl, 0, 8,
                              &thresh_pc, &has_thresh, &ns_rej_true))
      return false;

   struct exec_list *ns_rej = ns_rej_true ? &ns_if->then_list : &ns_if->else_list;
   struct exec_list *ns_pass = ns_rej_true ? &ns_if->else_list : &ns_if->then_list;
   if (!match_fast9_zero_score_list(ns_rej, x, y, w_off, wg, &score_set,
                                    &score_bind, &has_score,
                                    &zero_stores[num_zero_stores++]))
      return false;

   /* Inside `ns_pass`: `arc_loop` followed by `corner_if`. */
   nir_loop *arc_loop = NULL;
   nir_if *corner_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, ns_pass) {
      if (n->type == nir_cf_node_loop) {
         if (arc_loop || corner_if)
            return false;
         arc_loop = nir_cf_node_as_loop(n);
      } else if (n->type == nir_cf_node_if) {
         if (!arc_loop || corner_if)
            return false;
         corner_if = nir_cf_node_as_if(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!arc_loop || !corner_if)
      return false;

   nir_def *best_phi = NULL, *found_phi = NULL;
   if (!match_fast9_arc_loop(arc_loop, &src_tbl, thresh_pc, wg,
                             &best_phi, &found_phi))
      return false;

   /* `corner_if` branches on `found_phi` (or `!found_phi`). */
   struct exec_list *corner_hit = NULL, *corner_miss = NULL;
   nir_def *cc = strip_mov(corner_if->condition.ssa);
   if (cc == found_phi) {
      corner_hit = &corner_if->then_list;
      corner_miss = &corner_if->else_list;
   } else if (cc && nir_def_instr(cc)->type == nir_instr_type_alu &&
              nir_instr_as_alu(nir_def_instr(cc))->op == nir_op_inot &&
              strip_mov(nir_instr_as_alu(nir_def_instr(cc))->src[0].src.ssa) ==
                 found_phi) {
      corner_hit = &corner_if->else_list;
      corner_miss = &corner_if->then_list;
   } else {
      return false;
   }

   if (!match_fast9_zero_score_list(corner_miss, x, y, w_off, wg, &score_set,
                                    &score_bind, &has_score,
                                    &zero_stores[num_zero_stores++]))
      return false;

   /* `corner_hit`: single block with `dst_score[idx] = i2f32(best - 1)`,
    * `pt_idx = atomicAdd(dst_count[0], 1)`, `dst_pts[pt_idx] = ivec2(x, y)`. */
   if (!exec_list_is_singular(corner_hit))
      return false;
   nir_cf_node *chn =
      exec_node_data(nir_cf_node, exec_list_get_head(corner_hit), node);
   if (chn->type != nir_cf_node_block)
      return false;
   nir_block *ch_blk = nir_cf_node_as_block(chn);

   nir_intrinsic_instr *score_hit_st = NULL;
   nir_intrinsic_instr *cnt_atomic = NULL;
   nir_intrinsic_instr *pts_st = NULL;
   nir_intrinsic_instr *pts_x_st = NULL, *pts_y_st = NULL;
   uint32_t cnt_set = 0, cnt_bind = 0, pts_set = 0, pts_bind = 0;
   bool has_pts_bind = false;

   nir_foreach_instr(instr, ch_blk) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_deref_atomic) {
         if (cnt_atomic ||
             nir_intrinsic_atomic_op(intr) != nir_atomic_op_iadd ||
             !is_const_i32(intr->src[1].ssa, 1) ||
             !match_ssbo_u32_zero_deref(nir_src_as_deref(intr->src[0]),
                                        &cnt_set, &cnt_bind))
            return false;
         cnt_atomic = intr;
      } else if (intr->intrinsic == nir_intrinsic_store_deref) {
         if (nir_intrinsic_access(intr) &
             (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
            return false;
         nir_deref_instr *dr = nir_src_as_deref(intr->src[0]);
         uint32_t set = 0, bind = 0;
         nir_def *idx = NULL;
         unsigned pair_comp = 0;
         if (nir_intrinsic_write_mask(intr) == 1 &&
             match_ssbo_deref_access(dr, 32, &set, &bind, &idx) &&
             set == score_set && bind == score_bind) {
            if (score_hit_st)
               return false;
            uint16_t sp = 0;
            if (!match_linear_index(idx, y, x, &sp, wg) || sp != w_off)
               return false;
            /* Value must be `i2f32(best_phi - 1)`. */
            nir_def *val = strip_mov(intr->src[1].ssa);
            if (!val || nir_def_instr(val)->type != nir_instr_type_alu)
               return false;
            nir_alu_instr *va = nir_instr_as_alu(nir_def_instr(val));
            if (va->op != nir_op_i2f32)
               return false;
            nir_def *bm1 = strip_mov(va->src[0].src.ssa);
            if (!bm1 || nir_def_instr(bm1)->type != nir_instr_type_alu)
               return false;
            nir_alu_instr *ba = nir_instr_as_alu(nir_def_instr(bm1));
            bool bm1_ok =
               (ba->op == nir_op_iadd &&
                ((strip_mov(ba->src[0].src.ssa) == best_phi &&
                  is_const_i32(ba->src[1].src.ssa, -1)) ||
                 (strip_mov(ba->src[1].src.ssa) == best_phi &&
                  is_const_i32(ba->src[0].src.ssa, -1)))) ||
               (ba->op == nir_op_isub &&
                strip_mov(ba->src[0].src.ssa) == best_phi &&
                is_const_i32(ba->src[1].src.ssa, 1));
            if (!bm1_ok)
               return false;
            score_hit_st = intr;
         } else if (nir_intrinsic_write_mask(intr) == 0x3 &&
                    match_ssbo_deref_vec_access(dr, 2, 32, &set, &bind, &idx)) {
            if (pts_st || pts_x_st || pts_y_st || !cnt_atomic ||
                strip_mov(idx) != &cnt_atomic->def)
               return false;
            nir_def *pv = intr->src[1].ssa;
            if (!pv || pv->num_components != 2 || pv->bit_size != 32 ||
                nir_def_instr(pv)->type != nir_instr_type_alu)
               return false;
            nir_alu_instr *pa = nir_instr_as_alu(nir_def_instr(pv));
            if (pa->op != nir_op_vec2 ||
                strip_mov(pa->src[0].src.ssa) != x ||
                strip_mov(pa->src[1].src.ssa) != y)
               return false;
            pts_set = set;
            pts_bind = bind;
            has_pts_bind = true;
            pts_st = intr;
         } else if (nir_intrinsic_write_mask(intr) == 1 &&
                    match_ssbo_i32_pair_store_deref(dr, &set, &bind, &idx,
                                                    &pair_comp)) {
            if (pts_st || !cnt_atomic || strip_mov(idx) != &cnt_atomic->def)
               return false;
            if (!has_pts_bind) {
               pts_set = set;
               pts_bind = bind;
               has_pts_bind = true;
            } else if (set != pts_set || bind != pts_bind) {
               return false;
            }
            if (pair_comp == 0) {
               if (pts_x_st || strip_mov(intr->src[1].ssa) != x)
                  return false;
               pts_x_st = intr;
            } else {
               if (pts_y_st || strip_mov(intr->src[1].ssa) != y)
                  return false;
               pts_y_st = intr;
            }
         } else {
            return false;
         }
      }
   }
   const bool pts_ok = (pts_st != NULL) || (pts_x_st != NULL && pts_y_st != NULL);
   if (!score_hit_st || !cnt_atomic || !pts_ok)
      return false;

   /* All 4 SSBO bindings must be pairwise distinct. */
   const uint32_t sets[4] = {src_set, score_set, cnt_set, pts_set};
   const uint32_t binds[4] = {src_bind, score_bind, cnt_bind, pts_bind};
   for (unsigned i = 0; i < 4; i++) {
      for (unsigned j = i + 1; j < 4; j++) {
         if (sets[i] == sets[j] && binds[i] == binds[j])
            return false;
      }
   }

   /* Whole-shader completeness check: no stray loads/stores/atomics/tex/calls. */
   unsigned seen_loads = 0, seen_stores = 0, seen_atomics = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_deref: {
            int8_t k = -2;
            if (!fast9_lookup_load(&src_tbl, intr, &k))
               return false;
            seen_loads++;
            break;
         }
         case nir_intrinsic_store_deref: {
            bool ok = (intr == score_hit_st || intr == pts_st ||
                       intr == pts_x_st || intr == pts_y_st);
            for (unsigned k = 0; k < num_zero_stores && !ok; k++)
               ok = (intr == zero_stores[k]);
            if (!ok)
               return false;
            seen_stores++;
            break;
         }
         case nir_intrinsic_deref_atomic:
            if (intr != cnt_atomic)
               return false;
            seen_atomics++;
            break;
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            return false;
         }
      }
   }
   const unsigned expect_stores = num_zero_stores + (pts_st ? 2u : 3u);
   if (seen_loads != src_tbl.count || seen_stores != expect_stores ||
       seen_atomics != 1)
      return false;

   for (unsigned i = 0; i < 4; i++) {
      if (!populate_binding_info(set_layout_count, set_layouts, sets[i],
                                 binds[i], &plan->bufs[i]))
         return false;
   }

   plan->kind = HK_2D_STENCIL_FAST9_U8;
   plan->tile_w = 32;
   plan->tile_h = 8;
   plan->width_pc_offset_B = w_off;
   plan->height_pc_offset_B = h_off;
   plan->stride_pc_offset_B = stride_pc;
   plan->border_pc_offset_B = border_pc;
   plan->threshold_pc_offset_B = thresh_pc;
   return true;
}

/* ---------------------------------------------------------------------------
 * Top-level matcher entry point
 * --------------------------------------------------------------------------- */

bool
hk_nir_match_2d_stencil(const nir_shader *orig_nir, uint32_t set_layout_count,
                        struct vk_descriptor_set_layout *const *set_layouts,
                        struct hk_2d_stencil_plan *plan_out)
{
   if (!orig_nir || orig_nir->info.stage != MESA_SHADER_COMPUTE ||
       orig_nir->info.internal || orig_nir->info.shared_size != 0 ||
       orig_nir->info.workgroup_size_variable ||
       orig_nir->info.workgroup_size[0] == 0 ||
       orig_nir->info.workgroup_size[1] == 0 ||
       orig_nir->info.workgroup_size[2] != 1 ||
       orig_nir->info.ray_queries > 0 ||
       orig_nir->info.num_images > 0 || orig_nir->info.num_textures > 0 ||
       orig_nir->info.num_ubos > 0)
      return false;

   nir_foreach_variable_in_shader(var, orig_nir) {
      if (var->data.mode &
          (nir_var_mem_shared | nir_var_image | nir_var_uniform))
         return false;
   }

   if (!exec_list_is_singular(&orig_nir->functions))
      return false;

   /* Quick SSBO count pre-filter before cloning. */
   if (orig_nir->info.num_ssbos != 2 && orig_nir->info.num_ssbos != 3 &&
       orig_nir->info.num_ssbos != 4 && orig_nir->info.num_ssbos != 6)
      return false;

   nir_shader *s = nir_shader_clone(NULL, orig_nir);
   if (!s)
      return false;

   nir_lower_vars_to_ssa(s);
   nir_lower_system_values(s);
   nir_lower_load_const_to_scalar(s);
   nir_lower_explicit_io(s, nir_var_mem_push_const,
                         nir_address_format_32bit_offset);
   nir_lower_alu_to_scalar(s, NULL, NULL);
   nir_shader_alu_pass(s, expand_agx_ops_instr, nir_metadata_control_flow, NULL);

   bool progress;
   unsigned iter = 0;
   do {
      progress = false;
      progress |= hoist_matcher_invariants(s);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_undef);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      const nir_opt_peephole_select_options peep_opts = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, s, nir_opt_peephole_select, &peep_opts);
      NIR_PASS(progress, s, nir_opt_if, nir_opt_if_optimize_phi_true_false);
   } while (progress && ++iter < 8);

   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   /* Materialize explicit scalar channel extractions for vector system values
    * so copy-prop'd swizzled ALU uses expose uniform 1-component `nir_def *`. */
   nir_builder sb = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->def.num_components <= 1)
            continue;
         switch (intr->intrinsic) {
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_base_global_invocation_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_size:
            break;
         default:
            continue;
         }
         sb.cursor = nir_after_instr(instr);
         nir_def *ch[4] = {NULL};
         nir_foreach_use_safe(use, &intr->def) {
            if (nir_src_is_if(use))
               continue;
            nir_instr *parent = nir_src_use_instr(use);
            if (parent->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(parent);
            if (alu->op == nir_op_mov && alu->def.num_components == 1)
               continue;
            nir_alu_src *asrc = container_of(use, nir_alu_src, src);
            if (alu->def.num_components != 1)
               continue;
            unsigned c = asrc->swizzle[0];
            if (c >= intr->def.num_components)
               continue;
            if (!ch[c])
               ch[c] = nir_channel(&sb, &intr->def, c);
            nir_src_rewrite(use, ch[c]);
            asrc->swizzle[0] = 0;
         }
      }
   }

   nir_metadata_require(impl, nir_metadata_block_index);

   struct hk_2d_stencil_plan plan = {0};
   plan.orig_wg_size[0] = orig_nir->info.workgroup_size[0];
   plan.orig_wg_size[1] = orig_nir->info.workgroup_size[1];
   plan.orig_wg_size[2] = orig_nir->info.workgroup_size[2];

   bool matched = false;
   if (orig_nir->info.num_ssbos == 2)
      matched = match_box_blur_u8(s, set_layout_count, set_layouts, &plan);
   else if (orig_nir->info.num_ssbos == 3)
      matched = match_sobel_3x3_u8(s, set_layout_count, set_layouts, &plan);
   else if (orig_nir->info.num_ssbos == 4)
      matched = match_fast9_u8(s, set_layout_count, set_layouts, &plan);
   else if (orig_nir->info.num_ssbos == 6)
      matched = match_stereo_sad_u8(s, set_layout_count, set_layouts, &plan);

   ralloc_free(s);

   if (!matched)
      return false;

   *plan_out = plan;
   return true;
}

/* ---------------------------------------------------------------------------
 * Tiled NIR builders
 * --------------------------------------------------------------------------- */

static nir_def *
load_pc_i32(nir_builder *b, uint16_t offset_B)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = offset_B,
                                 .range = 4);
}

static nir_deref_instr *
build_ssbo_base_deref(nir_builder *b, const struct hk_2d_stencil_binding *bind,
                      nir_address_format ssbo_addr_format,
                      const struct glsl_type *elem_type, unsigned elem_stride)
{
   unsigned comps = nir_address_format_num_components(ssbo_addr_format);
   unsigned bits = nir_address_format_bit_size(ssbo_addr_format);
   nir_def *ridx = nir_vulkan_resource_index(
      b, comps, bits, nir_imm_int(b, 0), .desc_set = bind->set,
      .binding = bind->binding, .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, comps, bits, ridx, .desc_type = nir_descriptor_type_storage_buffer);
   return nir_build_deref_cast(b, desc, nir_var_mem_ssbo, elem_type,
                               elem_stride);
}

static nir_def *
load_ssbo_u8(nir_builder *b, nir_deref_instr *base, nir_def *idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_ssbo_u8x4(nir_builder *b, nir_deref_instr *base_u8x4, nir_def *byte_idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_u8x4, byte_idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static void
store_ssbo_u8(nir_builder *b, nir_deref_instr *base, nir_def *idx, nir_def *val)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   nir_store_deref_with_access(b, elem, val, 0x1, ACCESS_NON_READABLE);
}

static void
store_ssbo_u8x4(nir_builder *b, nir_deref_instr *base_u8x4, nir_def *byte_idx,
                nir_def *val_u8x4)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base_u8x4, byte_idx);
   nir_store_deref_with_access(b, elem, val_u8x4, 0xf, ACCESS_NON_READABLE);
}

static nir_def *
load_ssbo_u32(nir_builder *b, nir_deref_instr *base, nir_def *idx)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_ssbo_u32x4(nir_builder *b, nir_deref_instr *base_u32x4, nir_def *elem_idx)
{
   nir_deref_instr *elem =
      nir_build_deref_ptr_as_array(b, base_u32x4, elem_idx);
   return nir_load_deref_with_access(b, elem,
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static void
store_ssbo_u32(nir_builder *b, nir_deref_instr *base, nir_def *idx,
               nir_def *val)
{
   nir_deref_instr *elem = nir_build_deref_ptr_as_array(b, base, idx);
   nir_store_deref_with_access(b, elem, val, 0x1, ACCESS_NON_READABLE);
}

static void
store_ssbo_u32x4(nir_builder *b, nir_deref_instr *base_u32x4, nir_def *elem_idx,
                 nir_def *val_u32x4)
{
   nir_deref_instr *elem =
      nir_build_deref_ptr_as_array(b, base_u32x4, elem_idx);
   nir_store_deref_with_access(b, elem, val_u32x4, 0xf, ACCESS_NON_READABLE);
}

static void
emit_shared_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

static void
store_sh_u8x4(nir_builder *b, nir_deref_instr *arr, nir_def *idx4,
              nir_def *val_u8x4)
{
   nir_store_deref(b, nir_build_deref_array(b, arr, idx4), val_u8x4, 0xf);
}

static nir_def *
load_sh_u8x4(nir_builder *b, nir_deref_instr *arr, nir_def *idx4)
{
   return nir_load_deref(b, nir_build_deref_array(b, arr, idx4));
}

static nir_def *
load_sh_u8x4_byte(nir_builder *b, nir_deref_instr *arr, nir_def *byte_idx)
{
   nir_deref_instr *vec =
      nir_build_deref_array(b, arr, nir_ushr_imm(b, byte_idx, 2));
   return nir_load_deref(
      b, nir_build_deref_array(b, vec, nir_iand_imm(b, byte_idx, 3)));
}

static void
store_sh_u8x4_byte(nir_builder *b, nir_deref_instr *arr, nir_def *byte_idx,
                   nir_def *val)
{
   nir_deref_instr *vec =
      nir_build_deref_array(b, arr, nir_ushr_imm(b, byte_idx, 2));
   nir_store_deref(
      b, nir_build_deref_array(b, vec, nir_iand_imm(b, byte_idx, 3)), val, 0x1);
}

static nir_def *
load_sh_u16(nir_builder *b, nir_deref_instr *arr, nir_def *idx)
{
   return nir_load_deref(b, nir_build_deref_array(b, arr, idx));
}

static void
store_sh_u16(nir_builder *b, nir_deref_instr *arr, nir_def *idx, nir_def *val)
{
   nir_store_deref(b, nir_build_deref_array(b, arr, idx), val, 0x1);
}

/* R8 box blur, one subgroup per 128-column strip:
 *   Workgroup 256x1x1 = 8 subgroups stacked on one strip, 16 rows each
 *   (HK_BOX_BLUR_TILE_H = 128 rows), hk_box_blur_out_w(r) output columns.
 *   Each lane owns 4 source columns and keeps their vertical (2r+1)-row sums
 *   in registers, sliding them one row at a time (one u8x4 row added, one
 *   subtracted). Per row, an in-lane prefix plus a subgroup exclusive scan
 *   gives the strip's horizontal prefix sum P; output x is
 *   (P[x + r] - P[x - r - 1]) / (2r+1)^2. P goes through a 512-byte
 *   subgroup-private slice of shared memory, so no workgroup barrier is
 *   needed. Border pixels copy the source, as in the original shader.
 */
static void
box_blur_load_row(nir_builder *b, nir_deref_instr *src_u8,
                  nir_deref_instr *src_u8x4, nir_def *row, nir_def *hm1,
                  nir_def *wm1, nir_def *stride, nir_def *col, nir_def *col_ld,
                  nir_def *out[4])
{
   nir_def *rowc = nir_imin(b, nir_imax(b, row, nir_imm_int(b, 0)), hm1);
   nir_def *roff = nir_imul(b, rowc, stride);

   /* A u8x4 read of a group that crosses the right edge runs up to 3 bytes
    * past the row: inside the next row, but past the buffer on the last row,
    * which loads byte by byte instead. */
   nir_def *fast, *slow;
   nir_push_if(b, nir_ilt(b, rowc, hm1));
   {
      nir_def *v = load_ssbo_u8x4(b, src_u8x4, nir_iadd(b, roff, col_ld));
      fast = nir_u2u32(b, v);
   }
   nir_push_else(b, NULL);
   {
      nir_def *s[4];
      for (unsigned j = 0; j < 4; j++) {
         nir_def *cj = nir_imin(
            b, nir_imax(b, nir_iadd_imm(b, col, j), nir_imm_int(b, 0)), wm1);
         s[j] = nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd(b, roff, cj)));
      }
      slow = nir_vec(b, s, 4);
   }
   nir_pop_if(b, NULL);
   nir_def *v = nir_if_phi(b, fast, slow);
   for (unsigned j = 0; j < 4; j++)
      out[j] = nir_channel(b, v, j);
}

static void
subgroup_shared_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_SUBGROUP,
               .memory_scope = SCOPE_SUBGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

static void
build_box_blur_u8_kernel(nir_builder *b, const struct hk_2d_stencil_plan *plan,
                         nir_address_format ssbo_addr_format)
{
   const unsigned STRIP_W = HK_BOX_BLUR_STRIP_W;
   const unsigned ROWS = HK_BOX_BLUR_SUBGROUP_ROWS;

   b->shader->info.workgroup_size[0] = HK_BOX_BLUR_SUBGROUPS * 32;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;
   b->shader->info.api_subgroup_size = 32;
   b->shader->info.min_subgroup_size = 32;
   b->shader->info.max_subgroup_size = 32;

   nir_variable *sh_p_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint_type(), HK_BOX_BLUR_SUBGROUPS * STRIP_W, 0),
      "sh_prefix");
   nir_deref_instr *sh_p = nir_build_deref_var(b, sh_p_var);

   nir_deref_instr *src_u8 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *src_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *dst_u8 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *dst_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *radius = load_pc_i32(b, plan->radius_pc_offset_B);

   nir_def *diam = nir_iadd_imm(b, nir_ishl_imm(b, radius, 1), 1);
   nir_def *area = nir_imul(b, diam, diam);
   nir_def *halo = nir_iand_imm(b, nir_iadd_imm(b, radius, 4), ~3);
   nir_def *out_lanes = nir_iadd_imm(
      b,
      nir_ushr_imm(b, nir_isub(b, nir_imm_int(b, STRIP_W - 4),
                               nir_iadd(b, halo, radius)),
                   2),
      1);
   nir_def *wm1 = nir_iadd_imm(b, width, -1);
   nir_def *hm1 = nir_iadd_imm(b, height, -1);

   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *sg = nir_load_subgroup_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tile_x =
      nir_imul(b, nir_channel(b, wgid, 0), nir_ishl_imm(b, out_lanes, 2));
   nir_def *y0 = nir_iadd(b, nir_imul_imm(b, nir_channel(b, wgid, 1),
                                          HK_BOX_BLUR_TILE_H),
                          nir_imul_imm(b, sg, ROWS));

   /* First of this lane's 4 source columns. Strips start at a multiple of
    * 4, so a group never straddles column 0. Groups wholly outside the image
    * read columns 0..3 or the last 4; their sums only reach border outputs,
    * which copy the source instead. */
   nir_def *col =
      nir_iadd(b, nir_isub(b, tile_x, halo), nir_ishl_imm(b, lane, 2));
   nir_def *col_ld =
      nir_bcsel(b, nir_ige(b, col, width), nir_iadd_imm(b, width, -4),
                nir_imax(b, col, nir_imm_int(b, 0)));

   nir_def *sh_lane = nir_iadd(b, nir_imul_imm(b, sg, STRIP_W),
                               nir_ishl_imm(b, lane, 2));
   nir_def *sh_hi = nir_iadd(b, sh_lane, nir_iadd(b, halo, radius));
   nir_def *sh_lo = nir_isub(b, nir_iadd(b, sh_lane, halo),
                             nir_iadd_imm(b, radius, 1));
   nir_def *x0 = nir_iadd(b, tile_x, nir_ishl_imm(b, lane, 2));
   nir_def *out_lane =
      nir_iand(b, nir_ult(b, lane, out_lanes), nir_ilt(b, x0, width));
   nir_def *x_int = nir_iand(b, nir_ige(b, x0, radius),
                             nir_ilt(b, nir_iadd_imm(b, x0, 3),
                                     nir_isub(b, width, radius)));
   nir_def *x_lim = nir_isub(b, width, radius);

   nir_push_if(b, nir_ilt(b, y0, height));
   {
      /* Vertical sums of rows y0 - r .. y0 + r (clamped rows only ever feed
       * border outputs). */
      nir_variable *cs_var[4];
      for (unsigned j = 0; j < 4; j++) {
         cs_var[j] = nir_local_variable_create(b->impl, glsl_uint_type(), "cs");
         nir_store_var(b, cs_var[j], nir_imm_int(b, 0), 0x1);
      }
      nir_variable *d_var =
         nir_local_variable_create(b->impl, glsl_int_type(), "bd");
      nir_store_var(b, d_var, nir_ineg(b, radius), 0x1);
      nir_loop *il = nir_push_loop(b);
      il->control = nir_loop_control_dont_unroll;
      {
         nir_def *d = nir_load_var(b, d_var);
         nir_push_if(b, nir_ilt(b, radius, d));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         nir_def *v[4];
         box_blur_load_row(b, src_u8, src_u8x4, nir_iadd(b, y0, d), hm1, wm1,
                           stride, col, col_ld, v);
         for (unsigned j = 0; j < 4; j++)
            nir_store_var(b, cs_var[j],
                          nir_iadd(b, nir_load_var(b, cs_var[j]), v[j]), 0x1);
         nir_store_var(b, d_var, nir_iadd_imm(b, d, 1), 0x1);
      }
      nir_pop_loop(b, NULL);

      nir_variable *i_var =
         nir_local_variable_create(b->impl, glsl_int_type(), "bi");
      nir_store_var(b, i_var, nir_imm_int(b, 0), 0x1);
      nir_loop *rl = nir_push_loop(b);
      rl->control = nir_loop_control_dont_unroll;
      {
         nir_def *i = nir_load_var(b, i_var);
         nir_def *y = nir_iadd(b, y0, i);
         nir_push_if(b, nir_ior(b, nir_ige_imm(b, i, ROWS),
                                nir_ige(b, y, height)));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         /* Next row's window: add row y + r + 1, drop row y - r. Issued
          * here so the loads overlap this row's output work. */
         nir_def *add[4], *sub[4];
         box_blur_load_row(b, src_u8, src_u8x4,
                           nir_iadd(b, y, nir_iadd_imm(b, radius, 1)), hm1,
                           wm1, stride, col, col_ld, add);
         box_blur_load_row(b, src_u8, src_u8x4, nir_isub(b, y, radius), hm1,
                           wm1, stride, col, col_ld, sub);

         nir_def *cs[4], *p[4];
         for (unsigned j = 0; j < 4; j++)
            cs[j] = nir_load_var(b, cs_var[j]);
         p[0] = cs[0];
         for (unsigned j = 1; j < 4; j++)
            p[j] = nir_iadd(b, p[j - 1], cs[j]);
         nir_def *ex =
            nir_exclusive_scan(b, p[3], .reduction_op = nir_op_iadd);

         /* The previous row's reads of this slice are done before it is
          * overwritten, and this row's writes land before it is read. */
         subgroup_shared_barrier(b);
         for (unsigned j = 0; j < 4; j++) {
            nir_store_deref(
               b, nir_build_deref_array(b, sh_p, nir_iadd_imm(b, sh_lane, j)),
               nir_iadd(b, ex, p[j]), 0x1);
         }
         subgroup_shared_barrier(b);

         nir_push_if(b, out_lane);
         {
            nir_def *o[4];
            for (unsigned j = 0; j < 4; j++) {
               nir_def *hi = nir_load_deref(
                  b, nir_build_deref_array(b, sh_p, nir_iadd_imm(b, sh_hi, j)));
               nir_def *lo = nir_load_deref(
                  b, nir_build_deref_array(b, sh_p, nir_iadd_imm(b, sh_lo, j)));
               o[j] = nir_u2u8(b, nir_udiv(b, nir_isub(b, hi, lo), area));
            }

            nir_def *y_int = nir_iand(b, nir_ige(b, y, radius),
                                      nir_ilt(b, y, nir_isub(b, height, radius)));
            nir_def *off = nir_iadd(b, nir_imul(b, y, stride), x0);
            nir_push_if(b, nir_iand(b, y_int, x_int));
            {
               store_ssbo_u8x4(b, dst_u8x4, off, nir_vec(b, o, 4));
            }
            nir_push_else(b, NULL);
            {
               for (unsigned j = 0; j < 4; j++) {
                  nir_def *px = nir_iadd_imm(b, x0, j);
                  nir_def *offj = nir_iadd_imm(b, off, j);
                  nir_push_if(b, nir_ilt(b, px, width));
                  {
                     nir_push_if(b, nir_iand(b, y_int,
                                             nir_iand(b, nir_ige(b, px, radius),
                                                      nir_ilt(b, px, x_lim))));
                     {
                        store_ssbo_u8(b, dst_u8, offj, o[j]);
                     }
                     nir_push_else(b, NULL);
                     {
                        store_ssbo_u8(b, dst_u8, offj,
                                      load_ssbo_u8(b, src_u8, offj));
                     }
                     nir_pop_if(b, NULL);
                  }
                  nir_pop_if(b, NULL);
               }
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);

         for (unsigned j = 0; j < 4; j++)
            nir_store_var(b, cs_var[j],
                          nir_isub(b, nir_iadd(b, cs[j], add[j]), sub[j]),
                          0x1);
         nir_store_var(b, i_var, nir_iadd_imm(b, i, 1), 0x1);
      }
      nir_pop_loop(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Separable 2D shared-memory stereo SAD block matching:
 *   Tile 64x16 (1024 px/WG), WG 256x1x1 (8 simdgroups), disparity chunk 32.
 *   Shared memory:
 *     sh_left:  48 x 25 u8x4 (= 100 B/row, bank-conflict-free) = 4,800 B
 *     sh_right: 48 x 33 u8x4 (= 132 B/row, bank-conflict-free) = 6,336 B
 *     sh_col:   16 x 97 u16  (odd stride, bank-conflict-free)  = 3,104 B
 *     Total: 14,240 B (13.9 KiB).
 *   Loads `sh_left` once per tile, loads `sh_right` once per 32-disparity
 *   chunk, then for each disparity runs an O(1)-amortized 2-pass separable
 *   vertical + horizontal sliding-window SAD in shared memory, bit-exact on
 *   interior and all border pixels. */
static nir_def *
load_stereo_absdiff(nir_builder *b, nir_deref_instr *sh_left,
                    nir_deref_instr *sh_right, nir_def *sy, nir_def *cx,
                    nir_def *rx, unsigned sh_l_b, unsigned sh_r_b)
{
   nir_def *la = nir_u2u32(
      b, load_sh_u8x4_byte(b, sh_left,
                           nir_iadd(b, nir_imul_imm(b, sy, sh_l_b), cx)));
   nir_def *ra = nir_u2u32(
      b, load_sh_u8x4_byte(b, sh_right,
                           nir_iadd(b, nir_imul_imm(b, sy, sh_r_b), rx)));
   return nir_iabs(b, nir_isub(b, la, ra));
}

static void
build_stereo_sad_u8_kernel(nir_builder *b,
                           const struct hk_2d_stencil_plan *plan,
                           nir_address_format ssbo_addr_format)
{
   const unsigned WG = 256;
   const unsigned TILE_W = 64;
   const unsigned TILE_H = 16;
   const unsigned D_CHUNK = 32;
   const unsigned MAX_H_EXT = TILE_H + 2 * HK_2D_STENCIL_MAX_RADIUS; /* 48 */
   const unsigned SH_L_U4 = 25; /* 100 B >= 64 + 32 = 96 */
   const unsigned SH_L_B = SH_L_U4 * 4;
   const unsigned SH_R_U4 = 33; /* 132 B >= 64 + 32 + 31 = 127 */
   const unsigned SH_R_B = SH_R_U4 * 4;
   const unsigned SH_COL_STRIDE = 97;

   b->shader->info.workgroup_size[0] = WG;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_left_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT8, 4),
                      MAX_H_EXT * SH_L_U4, 0),
      "sh_left");
   nir_variable *sh_right_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT8, 4),
                      MAX_H_EXT * SH_R_U4, 0),
      "sh_right");
   nir_variable *sh_col_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint16_t_type(), TILE_H * SH_COL_STRIDE, 0),
      "sh_col");
   nir_deref_instr *sh_left = nir_build_deref_var(b, sh_left_var);
   nir_deref_instr *sh_right = nir_build_deref_var(b, sh_right_var);
   nir_deref_instr *sh_col = nir_build_deref_var(b, sh_col_var);

   nir_deref_instr *left_u8 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *left_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *right_u8 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *right_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *pdisp_u8 = build_ssbo_base_deref(
      b, &plan->bufs[2], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *pdisp_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[2], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *pcost_u32 = build_ssbo_base_deref(
      b, &plan->bufs[3], ssbo_addr_format, glsl_uint_type(), 4);
   nir_deref_instr *pcost_u32x4 = build_ssbo_base_deref(
      b, &plan->bufs[3], ssbo_addr_format, glsl_uvec4_type(), 4);
   nir_deref_instr *ddisp_u8 = build_ssbo_base_deref(
      b, &plan->bufs[4], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *ddisp_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[4], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *dcost_u32 = build_ssbo_base_deref(
      b, &plan->bufs[5], ssbo_addr_format, glsl_uint_type(), 4);
   nir_deref_instr *dcost_u32x4 = build_ssbo_base_deref(
      b, &plan->bufs[5], ssbo_addr_format, glsl_uvec4_type(), 4);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *cost_stride = load_pc_i32(b, plan->cost_stride_pc_offset_B);
   nir_def *win_size = load_pc_i32(b, plan->radius_pc_offset_B);
   nir_def *d_min = load_pc_i32(b, plan->d_min_pc_offset_B);
   nir_def *d_max = load_pc_i32(b, plan->d_max_pc_offset_B);

   /* Dispatch validator guarantees `win_size >= 1`. */
   nir_def *radius = nir_ushr_imm(b, win_size, 1);
   nir_def *two_r = nir_ishl_imm(b, radius, 1);
   nir_def *diam = nir_iadd_imm(b, two_r, 1);
   nir_def *w_ext = nir_iadd_imm(b, two_r, TILE_W);
   nir_def *h_ext = nir_iadd_imm(b, two_r, TILE_H);
   nir_def *w4 = nir_ushr_imm(b, nir_iadd_imm(b, w_ext, 3), 2);
   nir_def *w_m1 = nir_iadd_imm(b, width, -1);

   nir_def *tid = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tile_x = nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W);
   nir_def *tile_y = nir_imul_imm(b, nir_channel(b, wgid, 1), TILE_H);
   nir_def *x_start = nir_isub(b, tile_x, radius);
   nir_def *y_start = nir_isub(b, tile_y, radius);

   nir_def *lane32 = nir_iand_imm(b, tid, 31);
   nir_def *warp8 = nir_ushr_imm(b, tid, 5);
   nir_def *bx = nir_iand_imm(b, tid, 15);
   nir_def *row = nir_ushr_imm(b, tid, 4);
   nir_def *lx0 = nir_ishl_imm(b, bx, 2);
   nir_def *x0 = nir_iadd(b, tile_x, lx0);
   nir_def *py = nir_iadd(b, tile_y, row);

   /* Load initial `(best_cost, best_disp)` for the 4 pixels owned by this thread. */
   nir_variable *bc_var[4];
   nir_variable *bd_var[4];
   nir_def *px_lo[4];
   nir_def *px_hi[4];
   for (unsigned t = 0; t < 4; t++) {
      bc_var[t] = nir_local_variable_create(b->impl, glsl_int_type(), "bc");
      bd_var[t] = nir_local_variable_create(b->impl, glsl_uint_type(), "bd");
      nir_store_var(b, bc_var[t], nir_imm_int(b, 0), 0x1);
      nir_store_var(b, bd_var[t], nir_imm_int(b, 0), 0x1);
      nir_def *px = nir_iadd_imm(b, x0, t);
      px_lo[t] =
         nir_imin(b, nir_imax(b, nir_isub(b, px, radius), nir_imm_int(b, 0)),
                  w_m1);
      px_hi[t] =
         nir_imin(b, nir_imax(b, nir_iadd(b, px, radius), nir_imm_int(b, 0)),
                  w_m1);
   }

   nir_def *disp_row = nir_imul(b, py, stride);
   nir_def *cost_row = nir_imul(b, py, cost_stride);
   nir_push_if(b, nir_ilt(b, py, height));
   {
      nir_push_if(b, nir_ilt(b, nir_iadd_imm(b, x0, 3), width));
      {
         nir_def *c4 = load_ssbo_u32x4(b, pcost_u32x4, nir_iadd(b, cost_row, x0));
         nir_def *d4 = load_ssbo_u8x4(b, pdisp_u8x4, nir_iadd(b, disp_row, x0));
         for (unsigned t = 0; t < 4; t++) {
            nir_store_var(b, bc_var[t], nir_channel(b, c4, t), 0x1);
            nir_store_var(b, bd_var[t], nir_u2u32(b, nir_channel(b, d4, t)),
                          0x1);
         }
      }
      nir_push_else(b, NULL);
      {
         for (unsigned t = 0; t < 4; t++) {
            nir_def *px = nir_iadd_imm(b, x0, t);
            nir_push_if(b, nir_ilt(b, px, width));
            {
               nir_store_var(
                  b, bc_var[t],
                  load_ssbo_u32(b, pcost_u32, nir_iadd(b, cost_row, px)), 0x1);
               nir_store_var(
                  b, bd_var[t],
                  nir_u2u32(b, load_ssbo_u8(b, pdisp_u8,
                                            nir_iadd(b, disp_row, px))),
                  0x1);
            }
            nir_pop_if(b, NULL);
         }
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   /* Phase 0: Preload `sh_left` ONCE for the entire disparity sweep. */
   nir_push_if(b, nir_ige(b, d_max, d_min));
   {
      nir_def *left_fast = nir_iand(
         b,
         nir_iand(b, nir_ige_imm(b, x_start, 0),
                  nir_ige(b, width,
                          nir_iadd(b, x_start, nir_ishl_imm(b, w4, 2)))),
         nir_iand(b, nir_ige_imm(b, y_start, 0),
                  nir_ige(b, height, nir_iadd(b, y_start, h_ext))));

      nir_push_if(b, left_fast);
      {
         nir_push_if(b, nir_ilt(b, lane32, w4));
         {
            nir_variable *sy_v =
               nir_local_variable_create(b->impl, glsl_int_type(), "lsy");
            nir_store_var(b, sy_v, warp8, 0x1);
            nir_def *gx = nir_iadd(b, x_start, nir_ishl_imm(b, lane32, 2));
            nir_loop *ll = nir_push_loop(b);
            ll->control = nir_loop_control_dont_unroll;
            {
               nir_def *sy = nir_load_var(b, sy_v);
               nir_push_if(b, nir_ige(b, sy, h_ext));
               nir_jump(b, nir_jump_break);
               nir_pop_if(b, NULL);

               nir_def *gy = nir_iadd(b, y_start, sy);
               nir_def *v = load_ssbo_u8x4(
                  b, left_u8x4, nir_iadd(b, nir_imul(b, gy, stride), gx));
               store_sh_u8x4(
                  b, sh_left, nir_iadd(b, nir_imul_imm(b, sy, SH_L_U4), lane32),
                  v);
               nir_store_var(b, sy_v, nir_iadd_imm(b, sy, 8), 0x1);
            }
            nir_pop_loop(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         nir_variable *sy_v =
            nir_local_variable_create(b->impl, glsl_int_type(), "lsyb");
         nir_store_var(b, sy_v, warp8, 0x1);
         nir_loop *ll = nir_push_loop(b);
         ll->control = nir_loop_control_dont_unroll;
         {
            nir_def *sy = nir_load_var(b, sy_v);
            nir_push_if(b, nir_ige(b, sy, h_ext));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *gy = nir_iadd(b, y_start, sy);
            nir_def *y_ok =
               nir_iand(b, nir_ige_imm(b, gy, 0), nir_ilt(b, gy, height));
            nir_def *row_off = nir_imul(b, gy, stride);
            nir_def *sh_row = nir_imul_imm(b, sy, SH_L_B);

            nir_variable *cx_v =
               nir_local_variable_create(b->impl, glsl_int_type(), "lcxb");
            nir_store_var(b, cx_v, lane32, 0x1);
            nir_loop *cl = nir_push_loop(b);
            cl->control = nir_loop_control_dont_unroll;
            {
               nir_def *cx = nir_load_var(b, cx_v);
               nir_push_if(b, nir_ige(b, cx, w_ext));
               nir_jump(b, nir_jump_break);
               nir_pop_if(b, NULL);

               nir_def *gx = nir_iadd(b, x_start, cx);
               nir_def *in = nir_iand(
                  b, y_ok,
                  nir_iand(b, nir_ige_imm(b, gx, 0), nir_ilt(b, gx, width)));
               nir_variable *bv =
                  nir_local_variable_create(b->impl, glsl_uint8_t_type(), "lb");
               nir_store_var(b, bv, nir_imm_intN_t(b, 0, 8), 0x1);
               nir_push_if(b, in);
               {
                  nir_store_var(
                     b, bv, load_ssbo_u8(b, left_u8, nir_iadd(b, row_off, gx)),
                     0x1);
               }
               nir_pop_if(b, NULL);
               store_sh_u8x4_byte(b, sh_left, nir_iadd(b, sh_row, cx),
                                  nir_load_var(b, bv));
               nir_store_var(b, cx_v, nir_iadd_imm(b, cx, 32), 0x1);
            }
            nir_pop_loop(b, NULL);
            nir_store_var(b, sy_v, nir_iadd_imm(b, sy, 8), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);

      /* Outer disparity chunk loop: `d0 = d_min, d_min + 32, ...` */
      nir_variable *d0_var =
         nir_local_variable_create(b->impl, glsl_int_type(), "d0");
      nir_store_var(b, d0_var, d_min, 0x1);
      nir_loop *chunk_loop = nir_push_loop(b);
      chunk_loop->control = nir_loop_control_dont_unroll;
      {
         nir_def *d0 = nir_load_var(b, d0_var);
         nir_push_if(b, nir_ilt(b, d_max, d0));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         nir_def *d1 = nir_imin(b, nir_iadd_imm(b, d0, D_CHUNK - 1), d_max);

         /* Skip the entire chunk if no pixel in the tile can be in-bounds. */
         nir_def *chunk_active = nir_iand(
            b,
            nir_ige(b, nir_iadd_imm(b, nir_iadd(b, tile_x, radius), TILE_W - 1),
                    d0),
            nir_ilt(b, nir_isub(b, x_start, d1), width));

         nir_push_if(b, chunk_active);
         {
            nir_def *w_right = nir_iadd(b, w_ext, nir_isub(b, d1, d0));
            nir_def *wr4 = nir_ushr_imm(b, nir_iadd_imm(b, w_right, 3), 2);
            nir_def *grx_start = nir_isub(b, x_start, d1);

            /* Load `sh_right` for chunk `[d0..d1]`. */
            nir_def *right_fast = nir_iand(
               b,
               nir_iand(b, nir_ige_imm(b, grx_start, 0),
                        nir_ige(b, width,
                                nir_iadd(b, grx_start,
                                         nir_ishl_imm(b, wr4, 2)))),
               nir_iand(b, nir_ige_imm(b, y_start, 0),
                        nir_ige(b, height, nir_iadd(b, y_start, h_ext))));

            nir_push_if(b, right_fast);
            {
               nir_variable *sy_v =
                  nir_local_variable_create(b->impl, glsl_int_type(), "rsy");
               nir_store_var(b, sy_v, warp8, 0x1);
               nir_loop *rl = nir_push_loop(b);
               rl->control = nir_loop_control_dont_unroll;
               {
                  nir_def *sy = nir_load_var(b, sy_v);
                  nir_push_if(b, nir_ige(b, sy, h_ext));
                  nir_jump(b, nir_jump_break);
                  nir_pop_if(b, NULL);

                  nir_def *gy = nir_iadd(b, y_start, sy);
                  nir_def *row_off = nir_imul(b, gy, stride);
                  nir_def *sh_u4_row = nir_imul_imm(b, sy, SH_R_U4);

                  nir_push_if(b, nir_ilt(b, lane32, wr4));
                  {
                     nir_def *grx =
                        nir_iadd(b, grx_start, nir_ishl_imm(b, lane32, 2));
                     nir_def *v = load_ssbo_u8x4(b, right_u8x4,
                                                 nir_iadd(b, row_off, grx));
                     store_sh_u8x4(b, sh_right, nir_iadd(b, sh_u4_row, lane32),
                                   v);
                  }
                  nir_pop_if(b, NULL);
                  /* wr4 can be 32 (max w_right=127 -> (127+3)>>2 = 32); handle
                   * any extra word if wr4 > 32 (never happens for MAX_R=16). */
                  nir_store_var(b, sy_v, nir_iadd_imm(b, sy, 8), 0x1);
               }
               nir_pop_loop(b, NULL);
            }
            nir_push_else(b, NULL);
            {
               nir_variable *sy_v =
                  nir_local_variable_create(b->impl, glsl_int_type(), "rsyb");
               nir_store_var(b, sy_v, warp8, 0x1);
               nir_loop *rl = nir_push_loop(b);
               rl->control = nir_loop_control_dont_unroll;
               {
                  nir_def *sy = nir_load_var(b, sy_v);
                  nir_push_if(b, nir_ige(b, sy, h_ext));
                  nir_jump(b, nir_jump_break);
                  nir_pop_if(b, NULL);

                  nir_def *gy = nir_iadd(b, y_start, sy);
                  nir_def *y_ok = nir_iand(b, nir_ige_imm(b, gy, 0),
                                           nir_ilt(b, gy, height));
                  nir_def *row_off = nir_imul(b, gy, stride);
                  nir_def *sh_row = nir_imul_imm(b, sy, SH_R_B);

                  nir_variable *rx_v =
                     nir_local_variable_create(b->impl, glsl_int_type(), "rrx");
                  nir_store_var(b, rx_v, lane32, 0x1);
                  nir_loop *cl = nir_push_loop(b);
                  cl->control = nir_loop_control_dont_unroll;
                  {
                     nir_def *rx = nir_load_var(b, rx_v);
                     nir_push_if(b, nir_ige(b, rx, w_right));
                     nir_jump(b, nir_jump_break);
                     nir_pop_if(b, NULL);

                     nir_def *grx = nir_iadd(b, grx_start, rx);
                     nir_def *in = nir_iand(
                        b, y_ok,
                        nir_iand(b, nir_ige_imm(b, grx, 0),
                                 nir_ilt(b, grx, width)));
                     nir_variable *bv = nir_local_variable_create(
                        b->impl, glsl_uint8_t_type(), "rb");
                     nir_store_var(b, bv, nir_imm_intN_t(b, 0, 8), 0x1);
                     nir_push_if(b, in);
                     {
                        nir_store_var(
                           b, bv,
                           load_ssbo_u8(b, right_u8, nir_iadd(b, row_off, grx)),
                           0x1);
                     }
                     nir_pop_if(b, NULL);
                     store_sh_u8x4_byte(b, sh_right, nir_iadd(b, sh_row, rx),
                                        nir_load_var(b, bv));
                     nir_store_var(b, rx_v, nir_iadd_imm(b, rx, 32), 0x1);
                  }
                  nir_pop_loop(b, NULL);
                  nir_store_var(b, sy_v, nir_iadd_imm(b, sy, 8), 0x1);
               }
               nir_pop_loop(b, NULL);
            }
            nir_pop_if(b, NULL);

            emit_shared_barrier(b);

            /* Interior threads have all 4 pixels in bounds for every `d` in
             * `[d0..d1]` because `px_lo` and `px_hi` are monotonic in `t`. */
            nir_def *chunk_all_valid = nir_iand(
               b, nir_ige(b, px_lo[0], d1),
               nir_ilt(b, nir_isub(b, px_hi[3], d0), width));

            /* Inner disparity loop: `for (d = d0; d <= d1; d++)`. */
            nir_variable *d_var =
               nir_local_variable_create(b->impl, glsl_int_type(), "d");
            nir_store_var(b, d_var, d0, 0x1);
            nir_loop *d_loop = nir_push_loop(b);
            d_loop->control = nir_loop_control_dont_unroll;
            {
               nir_def *d = nir_load_var(b, d_var);
               nir_push_if(b, nir_ilt(b, d1, d));
               nir_jump(b, nir_jump_break);
               nir_pop_if(b, NULL);

               nir_def *r_off = nir_isub(b, d1, d);

               /* Phase 1: Vertical sliding-window column SADs (2 halves x 8 rows). */
               nir_def *half = nir_ushr_imm(b, tid, 7);
               nir_def *cx = nir_iand_imm(b, tid, 127);
               nir_push_if(b, nir_ilt(b, cx, w_ext));
               {
                  nir_def *row0 = nir_ishl_imm(b, half, 3);
                  nir_def *gx = nir_iadd(b, x_start, cx);
                  nir_def *rx = nir_iadd(b, cx, r_off);
                  nir_def *col_in =
                     nir_iand(b, nir_ige_imm(b, gx, 0), nir_ilt(b, gx, width));

                  nir_push_if(b, col_in);
                  {
                     nir_push_if(b, nir_ige_imm(b, diam, 7));
                     {
                        /* Cache the first 7 vertical absolute differences and
                         * reuse them as the outgoing rows for `j = 1..7`,
                         * eliminating 14 shared loads and 7 absdiffs per
                         * disparity. `diam` is odd, so `diam - 7` is even. */
                        nir_def *ad_top[7];
                        nir_def *init_acc = nir_imm_int(b, 0);
                        for (unsigned k = 0; k < 7; k++) {
                           nir_def *ad = load_stereo_absdiff(
                              b, sh_left, sh_right, nir_iadd_imm(b, row0, k),
                              cx, rx, SH_L_B, SH_R_B);
                           ad_top[k] = nir_u2u16(b, ad);
                           init_acc = nir_iadd(b, init_acc, ad);
                        }

                        nir_variable *acc_v = nir_local_variable_create(
                           b->impl, glsl_int_type(), "vacc");
                        nir_variable *k_v = nir_local_variable_create(
                           b->impl, glsl_int_type(), "vk");
                        nir_store_var(b, acc_v, init_acc, 0x1);
                        nir_store_var(b, k_v, nir_imm_int(b, 7), 0x1);
                        nir_loop *vl = nir_push_loop(b);
                        vl->control = nir_loop_control_dont_unroll;
                        {
                           nir_def *k = nir_load_var(b, k_v);
                           nir_push_if(b, nir_ige(b, k, diam));
                           nir_jump(b, nir_jump_break);
                           nir_pop_if(b, NULL);

                           nir_def *sy0 = nir_iadd(b, row0, k);
                           nir_def *ad0 = load_stereo_absdiff(
                              b, sh_left, sh_right, sy0, cx, rx, SH_L_B,
                              SH_R_B);
                           nir_def *ad1 = load_stereo_absdiff(
                              b, sh_left, sh_right, nir_iadd_imm(b, sy0, 1), cx,
                              rx, SH_L_B, SH_R_B);
                           nir_def *acc_next = nir_iadd(
                              b, nir_load_var(b, acc_v),
                              nir_iadd(b, ad0, ad1));
                           nir_store_var(b, acc_v, acc_next, 0x1);
                           nir_store_var(b, k_v, nir_iadd_imm(b, k, 2), 0x1);
                        }
                        nir_pop_loop(b, NULL);

                        nir_def *acc = nir_load_var(b, acc_v);
                        store_sh_u16(
                           b, sh_col,
                           nir_iadd(b, nir_imul_imm(b, row0, SH_COL_STRIDE), cx),
                           nir_u2u16(b, acc));

                        for (unsigned j = 1; j < 8; j++) {
                           nir_def *sy_add =
                              nir_iadd_imm(b, nir_iadd(b, row0, two_r), j);
                           nir_def *ad_add = load_stereo_absdiff(
                              b, sh_left, sh_right, sy_add, cx, rx, SH_L_B,
                              SH_R_B);
                           acc = nir_isub(b, nir_iadd(b, acc, ad_add),
                                          nir_u2u32(b, ad_top[j - 1]));
                           store_sh_u16(
                              b, sh_col,
                              nir_iadd(b,
                                       nir_imul_imm(b, nir_iadd_imm(b, row0, j),
                                                    SH_COL_STRIDE),
                                       cx),
                              nir_u2u16(b, acc));
                        }
                     }
                     nir_push_else(b, NULL);
                     {
                        nir_variable *acc_v = nir_local_variable_create(
                           b->impl, glsl_int_type(), "vacc_s");
                        nir_variable *k_v = nir_local_variable_create(
                           b->impl, glsl_int_type(), "vk_s");
                        nir_store_var(b, acc_v, nir_imm_int(b, 0), 0x1);
                        nir_store_var(b, k_v, nir_imm_int(b, 0), 0x1);
                        nir_loop *vl = nir_push_loop(b);
                        vl->control = nir_loop_control_dont_unroll;
                        {
                           nir_def *k = nir_load_var(b, k_v);
                           nir_push_if(b, nir_ige(b, k, diam));
                           nir_jump(b, nir_jump_break);
                           nir_pop_if(b, NULL);

                           nir_def *sy = nir_iadd(b, row0, k);
                           nir_def *ad = load_stereo_absdiff(
                              b, sh_left, sh_right, sy, cx, rx, SH_L_B, SH_R_B);
                           nir_store_var(b, acc_v,
                                         nir_iadd(b, nir_load_var(b, acc_v), ad),
                                         0x1);
                           nir_store_var(b, k_v, nir_iadd_imm(b, k, 1), 0x1);
                        }
                        nir_pop_loop(b, NULL);

                        nir_def *acc = nir_load_var(b, acc_v);
                        store_sh_u16(
                           b, sh_col,
                           nir_iadd(b, nir_imul_imm(b, row0, SH_COL_STRIDE), cx),
                           nir_u2u16(b, acc));

                        for (unsigned j = 1; j < 8; j++) {
                           nir_def *sy_add =
                              nir_iadd_imm(b, nir_iadd(b, row0, two_r), j);
                           nir_def *sy_sub = nir_iadd_imm(b, row0, j - 1);
                           nir_def *ad_add = load_stereo_absdiff(
                              b, sh_left, sh_right, sy_add, cx, rx, SH_L_B,
                              SH_R_B);
                           nir_def *ad_sub = load_stereo_absdiff(
                              b, sh_left, sh_right, sy_sub, cx, rx, SH_L_B,
                              SH_R_B);
                           acc = nir_isub(b, nir_iadd(b, acc, ad_add), ad_sub);
                           store_sh_u16(
                              b, sh_col,
                              nir_iadd(b,
                                       nir_imul_imm(b, nir_iadd_imm(b, row0, j),
                                                    SH_COL_STRIDE),
                                       cx),
                              nir_u2u16(b, acc));
                        }
                     }
                     nir_pop_if(b, NULL);
                  }
                  nir_push_else(b, NULL);
                  {
                     for (unsigned j = 0; j < 8; j++) {
                        store_sh_u16(
                           b, sh_col,
                           nir_iadd(b,
                                    nir_imul_imm(b, nir_iadd_imm(b, row0, j),
                                                 SH_COL_STRIDE),
                                    cx),
                           nir_imm_intN_t(b, 0, 16));
                     }
                  }
                  nir_pop_if(b, NULL);
               }
               nir_pop_if(b, NULL);

               emit_shared_barrier(b);

               /* Phase 2: Horizontal sliding window + min-cost update (4 px/thread). */
               nir_push_if(b, nir_ilt(b, py, height));
               {
                  nir_def *col_row = nir_imul_imm(b, row, SH_COL_STRIDE);
                  nir_def *col_base = nir_iadd(b, col_row, lx0);
                  nir_def *n4 = nir_ishr_imm(b, diam, 2);
                  nir_variable *hsum_v = nir_local_variable_create(
                     b->impl, glsl_int_type(), "hsum");
                  nir_variable *hk4_v = nir_local_variable_create(
                     b->impl, glsl_int_type(), "hk4");
                  nir_store_var(b, hsum_v, nir_imm_int(b, 0), 0x1);
                  nir_store_var(b, hk4_v, nir_imm_int(b, 0), 0x1);
                  /* 4 adjacent u16 column sums per iteration vectorize into a
                   * single 8-byte shared load. */
                  nir_loop *hl = nir_push_loop(b);
                  hl->control = nir_loop_control_dont_unroll;
                  {
                     nir_def *k4 = nir_load_var(b, hk4_v);
                     nir_push_if(b, nir_ige(b, k4, n4));
                     nir_jump(b, nir_jump_break);
                     nir_pop_if(b, NULL);

                     nir_def *off =
                        nir_iadd(b, col_base, nir_ishl_imm(b, k4, 2));
                     nir_def *s = nir_load_var(b, hsum_v);
                     for (unsigned lane = 0; lane < 4; lane++) {
                        nir_def *c = load_sh_u16(
                           b, sh_col, nir_iadd_imm(b, off, lane));
                        s = nir_iadd(b, s, nir_u2u32(b, c));
                     }
                     nir_store_var(b, hsum_v, s, 0x1);
                     nir_store_var(b, hk4_v, nir_iadd_imm(b, k4, 1), 0x1);
                  }
                  nir_pop_loop(b, NULL);

                  /* `diam = 2*radius + 1` is always odd, leaving 1 or 3 taps. */
                  nir_def *last = nir_iand_imm(b, diam, ~3);
                  nir_def *tail_off = nir_iadd(b, col_base, last);
                  nir_def *hsum = nir_iadd(
                     b, nir_load_var(b, hsum_v),
                     nir_u2u32(b, load_sh_u16(b, sh_col, tail_off)));
                  nir_store_var(b, hsum_v, hsum, 0x1);
                  nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, diam, 2), 0));
                  {
                     nir_def *c1 = load_sh_u16(
                        b, sh_col, nir_iadd_imm(b, tail_off, 1));
                     nir_def *c2 = load_sh_u16(
                        b, sh_col, nir_iadd_imm(b, tail_off, 2));
                     nir_def *s = nir_iadd(
                        b, nir_load_var(b, hsum_v),
                        nir_iadd(b, nir_u2u32(b, c1), nir_u2u32(b, c2)));
                     nir_store_var(b, hsum_v, s, 0x1);
                  }
                  nir_pop_if(b, NULL);

                  nir_def *hsums[4];
                  hsums[0] = nir_load_var(b, hsum_v);
                  for (unsigned t = 1; t < 4; t++) {
                     nir_def *c_add = load_sh_u16(
                        b, sh_col,
                        nir_iadd_imm(b, nir_iadd(b, col_base, two_r), t));
                     nir_def *c_sub = load_sh_u16(
                        b, sh_col, nir_iadd_imm(b, col_base, t - 1));
                     hsums[t] = nir_isub(
                        b, nir_iadd(b, hsums[t - 1], nir_u2u32(b, c_add)),
                        nir_u2u32(b, c_sub));
                  }

                  nir_push_if(b, chunk_all_valid);
                  {
                     for (unsigned t = 0; t < 4; t++) {
                        nir_def *cur_bc = nir_load_var(b, bc_var[t]);
                        nir_def *cur_bd = nir_load_var(b, bd_var[t]);
                        nir_def *better = nir_ilt(b, hsums[t], cur_bc);
                        nir_store_var(b, bc_var[t],
                                      nir_bcsel(b, better, hsums[t], cur_bc),
                                      0x1);
                        nir_store_var(b, bd_var[t],
                                      nir_bcsel(b, better, d, cur_bd), 0x1);
                     }
                  }
                  nir_push_else(b, NULL);
                  {
                     for (unsigned t = 0; t < 4; t++) {
                        nir_def *valid_d = nir_iand(
                           b, nir_ige(b, px_lo[t], d),
                           nir_ilt(b, nir_isub(b, px_hi[t], d), width));
                        nir_def *cur_bc = nir_load_var(b, bc_var[t]);
                        nir_def *cur_bd = nir_load_var(b, bd_var[t]);
                        nir_def *better =
                           nir_iand(b, valid_d, nir_ilt(b, hsums[t], cur_bc));
                        nir_store_var(b, bc_var[t],
                                      nir_bcsel(b, better, hsums[t], cur_bc),
                                      0x1);
                        nir_store_var(b, bd_var[t],
                                      nir_bcsel(b, better, d, cur_bd), 0x1);
                     }
                  }
                  nir_pop_if(b, NULL);
               }
               nir_pop_if(b, NULL);

               emit_shared_barrier(b);

               nir_store_var(b, d_var, nir_iadd_imm(b, d, 1), 0x1);
            }
            nir_pop_loop(b, NULL);
         }
         nir_pop_if(b, NULL);

         nir_store_var(b, d0_var, nir_iadd_imm(b, d0, D_CHUNK), 0x1);
      }
      nir_pop_loop(b, NULL);
   }
   nir_pop_if(b, NULL);

   /* Final coalesced stores of `dst_disp` (u8x4) and `dst_cost` (u32x4). */
   nir_push_if(b, nir_ilt(b, py, height));
   {
      nir_push_if(b, nir_ilt(b, nir_iadd_imm(b, x0, 3), width));
      {
         nir_def *d_out[4], *c_out[4];
         for (unsigned t = 0; t < 4; t++) {
            d_out[t] = nir_u2u8(b, nir_load_var(b, bd_var[t]));
            c_out[t] = nir_load_var(b, bc_var[t]);
         }
         store_ssbo_u8x4(b, ddisp_u8x4, nir_iadd(b, disp_row, x0),
                         nir_vec(b, d_out, 4));
         store_ssbo_u32x4(b, dcost_u32x4, nir_iadd(b, cost_row, x0),
                          nir_vec(b, c_out, 4));
      }
      nir_push_else(b, NULL);
      {
         for (unsigned t = 0; t < 4; t++) {
            nir_def *px = nir_iadd_imm(b, x0, t);
            nir_push_if(b, nir_ilt(b, px, width));
            {
               store_ssbo_u8(b, ddisp_u8, nir_iadd(b, disp_row, px),
                             nir_u2u8(b, nir_load_var(b, bd_var[t])));
               store_ssbo_u32(b, dcost_u32, nir_iadd(b, cost_row, px),
                              nir_load_var(b, bc_var[t]));
            }
            nir_pop_if(b, NULL);
         }
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* 2D shared-memory 3x3 Sobel gradient magnitude + quantized direction:
 *   Tile 128x8 (1024 px/WG), WG 256x1x1 (32x8 logical, 4 horizontal px/thread).
 *   Shared memory: 10 rows x 35 u8x4 (= 140 B/row >= 130 B, odd u32 stride)
 *   = 1,400 B (1.37 KiB -> full occupancy).
 *   Cooperative coalesced u8x4 preload, 6 vectorized u8x4 shared loads per
 *   4 pixels, exact integer Gx/Gy -> f32 + bit-identical angle quantization,
 *   and coalesced u32x4 interior stores to dir and mag. */
static void
build_sobel_3x3_u8_kernel(nir_builder *b,
                          const struct hk_2d_stencil_plan *plan,
                          nir_address_format ssbo_addr_format)
{
   const unsigned WG = 256;
   const unsigned TILE_W = 128;
   const unsigned TILE_H = 8;
   const unsigned H_EXT = TILE_H + 2; /* 10 */
   const unsigned W_EXT = TILE_W + 2; /* 130 */
   const unsigned SH_U4 = 35;         /* 140 B >= 132 B, odd -> conflict-free */
   const unsigned SH_B = SH_U4 * 4;

   b->shader->info.workgroup_size[0] = WG;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_src_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT8, 4), H_EXT * SH_U4, 0),
      "sh_sobel");
   nir_deref_instr *sh_src = nir_build_deref_var(b, sh_src_var);

   nir_deref_instr *src_u8 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *src_u8x4 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1);
   nir_deref_instr *dir_u32 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint_type(), 4);
   nir_deref_instr *dir_u32x4 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uvec4_type(), 4);
   nir_deref_instr *mag_u32 = build_ssbo_base_deref(
      b, &plan->bufs[2], ssbo_addr_format, glsl_uint_type(), 4);
   nir_deref_instr *mag_u32x4 = build_ssbo_base_deref(
      b, &plan->bufs[2], ssbo_addr_format, glsl_uvec4_type(), 4);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);

   nir_def *tid = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tile_x = nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W);
   nir_def *tile_y = nir_imul_imm(b, nir_channel(b, wgid, 1), TILE_H);
   nir_def *x_start = nir_iadd_imm(b, tile_x, -1);
   nir_def *y_start = nir_iadd_imm(b, tile_y, -1);

   nir_def *lane32 = nir_iand_imm(b, tid, 31);
   nir_def *warp8 = nir_ushr_imm(b, tid, 5);

   /* Phase 0: Preload 10 x 130 halo bytes into `sh_src`. */
   nir_def *tile_fast = nir_iand(
      b,
      nir_iand(b, nir_ige_imm(b, x_start, 0),
               nir_ige(b, width, nir_iadd_imm(b, x_start, 132))),
      nir_iand(b, nir_ige_imm(b, y_start, 0),
               nir_ige(b, height, nir_iadd_imm(b, y_start, H_EXT))));

   nir_push_if(b, tile_fast);
   {
      /* Fast interior preload: 8 warps load rows 0..7, warps 0..1 load 8..9. */
      nir_def *gx_main = nir_iadd(b, x_start, nir_ishl_imm(b, lane32, 2));
      nir_def *gx_tail = nir_iadd_imm(b, x_start, 128);
      for (unsigned pass = 0; pass < 2; pass++) {
         nir_def *sy = pass == 0 ? warp8 : nir_iadd_imm(b, warp8, 8);
         if (pass == 1)
            nir_push_if(b, nir_ilt_imm(b, sy, H_EXT));
         nir_def *gy = nir_iadd(b, y_start, sy);
         nir_def *row_off = nir_imul(b, gy, stride);
         nir_def *sh_u4_row = nir_imul_imm(b, sy, SH_U4);
         store_sh_u8x4(b, sh_src, nir_iadd(b, sh_u4_row, lane32),
                       load_ssbo_u8x4(b, src_u8x4, nir_iadd(b, row_off, gx_main)));
         nir_push_if(b, nir_ieq_imm(b, lane32, 0));
         {
            store_sh_u8x4(
               b, sh_src, nir_iadd_imm(b, sh_u4_row, 32),
               load_ssbo_u8x4(b, src_u8x4, nir_iadd(b, row_off, gx_tail)));
         }
         nir_pop_if(b, NULL);
         if (pass == 1)
            nir_pop_if(b, NULL);
      }
   }
   nir_push_else(b, NULL);
   {
      nir_variable *sy_v =
         nir_local_variable_create(b->impl, glsl_int_type(), "ssy");
      nir_store_var(b, sy_v, warp8, 0x1);
      nir_loop *rl = nir_push_loop(b);
      rl->control = nir_loop_control_dont_unroll;
      {
         nir_def *sy = nir_load_var(b, sy_v);
         nir_push_if(b, nir_ige_imm(b, sy, H_EXT));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         nir_def *gy = nir_iadd(b, y_start, sy);
         nir_push_if(b, nir_iand(b, nir_ige_imm(b, gy, 0),
                                 nir_ilt(b, gy, height)));
         {
            nir_def *row_off = nir_imul(b, gy, stride);
            nir_def *sh_row = nir_imul_imm(b, sy, SH_B);
            nir_variable *cx_v =
               nir_local_variable_create(b->impl, glsl_int_type(), "scx");
            nir_store_var(b, cx_v, lane32, 0x1);
            nir_loop *cl = nir_push_loop(b);
            cl->control = nir_loop_control_dont_unroll;
            {
               nir_def *cx = nir_load_var(b, cx_v);
               nir_push_if(b, nir_ige_imm(b, cx, W_EXT));
               nir_jump(b, nir_jump_break);
               nir_pop_if(b, NULL);

               nir_def *gx = nir_iadd(b, x_start, cx);
               nir_push_if(b, nir_iand(b, nir_ige_imm(b, gx, 0),
                                       nir_ilt(b, gx, width)));
               {
                  store_sh_u8x4_byte(
                     b, sh_src, nir_iadd(b, sh_row, cx),
                     load_ssbo_u8(b, src_u8, nir_iadd(b, row_off, gx)));
               }
               nir_pop_if(b, NULL);
               nir_store_var(b, cx_v, nir_iadd_imm(b, cx, 32), 0x1);
            }
            nir_pop_loop(b, NULL);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, sy_v, nir_iadd_imm(b, sy, 8), 0x1);
      }
      nir_pop_loop(b, NULL);
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   /* Phase 1: 32x8 threads compute 4 consecutive horizontal pixels each. */
   nir_def *bx = lane32;
   nir_def *row = warp8;
   nir_def *x0 = nir_iadd(b, tile_x, nir_ishl_imm(b, bx, 2));
   nir_def *py = nir_iadd(b, tile_y, row);

   nir_push_if(b, nir_ilt(b, py, height));
   {
      /* Load 3 rows x 6 bytes via 6 coalesced u8x4 shared loads. Keep samples
       * in u16/i16 (Sobel sums are in [-1020, 1020], L1 mag in [0, 2040]) to
       * halve live register footprint across the 4-pixel stencil body. */
      nir_def *p[3][6];
      for (unsigned dy = 0; dy < 3; dy++) {
         nir_def *sh_u4 =
            nir_iadd(b, nir_imul_imm(b, nir_iadd_imm(b, row, dy), SH_U4), bx);
         nir_def *v0 = load_sh_u8x4(b, sh_src, sh_u4);
         nir_def *v1 = load_sh_u8x4(b, sh_src, nir_iadd_imm(b, sh_u4, 1));
         p[dy][0] = nir_u2u16(b, nir_channel(b, v0, 0));
         p[dy][1] = nir_u2u16(b, nir_channel(b, v0, 1));
         p[dy][2] = nir_u2u16(b, nir_channel(b, v0, 2));
         p[dy][3] = nir_u2u16(b, nir_channel(b, v0, 3));
         p[dy][4] = nir_u2u16(b, nir_channel(b, v1, 0));
         p[dy][5] = nir_u2u16(b, nir_channel(b, v1, 1));
      }

      nir_def *c_lo = nir_imm_int(b, 0x3ed413cdu);
      nir_def *c_hi = nir_imm_int(b, 0x401a827au);

      nir_def *dir_out[4];
      nir_def *mag_out[4];
      for (unsigned t = 0; t < 4; t++) {
         nir_def *top = nir_iadd(
            b, nir_iadd(b, p[0][t], p[0][t + 2]),
            nir_ishl_imm(b, p[0][t + 1], 1));
         nir_def *bot = nir_iadd(
            b, nir_iadd(b, p[2][t], p[2][t + 2]),
            nir_ishl_imm(b, p[2][t + 1], 1));
         nir_def *lft = nir_iadd(
            b, nir_iadd(b, p[0][t], p[2][t]),
            nir_ishl_imm(b, p[1][t], 1));
         nir_def *rgt = nir_iadd(
            b, nir_iadd(b, p[0][t + 2], p[2][t + 2]),
            nir_ishl_imm(b, p[1][t + 2], 1));

         nir_def *gy_i = nir_isub(b, top, bot);
         nir_def *gx_i = nir_isub(b, lft, rgt);
         nir_def *agy_i = nir_iabs(b, gy_i);
         nir_def *agx_i = nir_iabs(b, gx_i);

         /* Exact L1 magnitude in i16 ([0, 2040]) -> f32. */
         mag_out[t] = nir_u2f32(b, nir_iadd(b, agy_i, agx_i));

         /* Replace SFU frcp(gy) with cross-multiplied threshold comparisons:
          * |gx / gy| < c_lo  <=>  |gx| < |gy| * c_lo
          * c_hi < |gx / gy|  <=>  |gy| * c_hi < |gx|
          * Bit-exact across all (gx, gy) in [-1020, 1020]^2 including gy == 0
          * (both (0, 0) -> false/false and (gx != 0, 0) -> false/true).
          * Direction constants include the final (q + 6) & 7 rotation. */
         nir_def *agx = nir_u2f32(b, agx_i);
         nir_def *agy = nir_u2f32(b, agy_i);
         nir_def *lt_lo = nir_flt(b, agx, nir_fmul(b, agy, c_lo));
         nir_def *gt_hi = nir_flt(b, nir_fmul(b, agy, c_hi), agx);

         nir_def *gx_neg = nir_ilt_imm(b, gx_i, 0);
         nir_def *gy_neg = nir_ilt_imm(b, gy_i, 0);
         nir_def *d_neg_gy =
            nir_bcsel(b, gx_neg, nir_imm_int(b, 3), nir_imm_int(b, 1));
         nir_def *d_pos_gy =
            nir_bcsel(b, gx_neg, nir_imm_int(b, 5), nir_imm_int(b, 7));
         nir_def *diag = nir_bcsel(b, gy_neg, d_neg_gy, d_pos_gy);
         nir_def *q0 = nir_bcsel(
            b, lt_lo,
            nir_bcsel(b, gy_neg, nir_imm_int(b, 2), nir_imm_int(b, 6)), diag);
         dir_out[t] = nir_bcsel(
            b, gt_hi,
            nir_bcsel(b, gx_neg, nir_imm_int(b, 4), nir_imm_int(b, 0)), q0);
      }

      nir_def *wm1 = nir_iadd_imm(b, width, -1);
      nir_def *hm1 = nir_iadd_imm(b, height, -1);
      nir_def *y_int = nir_iand(b, nir_ige_imm(b, py, 1), nir_ilt(b, py, hm1));
      nir_def *all_int = nir_iand(
         b, y_int,
         nir_iand(b, nir_ige_imm(b, x0, 1),
                  nir_ilt(b, nir_iadd_imm(b, x0, 3), wm1)));
      nir_def *out_row = nir_imul(b, py, width);

      nir_push_if(b, all_int);
      {
         nir_def *off = nir_iadd(b, out_row, x0);
         store_ssbo_u32x4(b, dir_u32x4, off, nir_vec(b, dir_out, 4));
         store_ssbo_u32x4(b, mag_u32x4, off, nir_vec(b, mag_out, 4));
      }
      nir_push_else(b, NULL);
      {
         for (unsigned t = 0; t < 4; t++) {
            nir_def *px = nir_iadd_imm(b, x0, t);
            nir_push_if(b, nir_ilt(b, px, width));
            {
               nir_def *off = nir_iadd(b, out_row, px);
               nir_def *is_int = nir_iand(
                  b, y_int,
                  nir_iand(b, nir_ige_imm(b, px, 1), nir_ilt(b, px, wm1)));
               nir_push_if(b, is_int);
               {
                  store_ssbo_u32(b, dir_u32, off, dir_out[t]);
                  store_ssbo_u32(b, mag_u32, off, mag_out[t]);
               }
               nir_push_else(b, NULL);
               {
                  store_ssbo_u32(b, mag_u32, off, nir_imm_int(b, 0));
               }
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);
         }
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Shared-memory 128x8 tiled FAST-9 corner detector (radius 3, 16-pt circle):
 *   Tile 128x8 (1024 px/WG), WG 256x1x1 (8 simdgroups, 4 horizontal px/lane).
 *   Shared memory:
 *     sh_src: 14 rows x 35 u8x4 (= 140 B/row, odd stride) = 1,960 B (< 2 KiB).
 *   Phase 0: coalesced u8x4 cooperative preload of 134x14 halo into sh_src.
 *   Phase 1: 7 coalesced u8x4 shared loads evaluate cardinal quick-rejects for
 *            4 pixels at once; quads with no cardinal pass skip directly to a
 *            single coalesced u32x4(0) store. Passing pixels load the remaining
 *            circle bytes from sh_src, run a 5-op bitmask 9-in-a-row reject,
 *            and only enter the exact 24-step arc loop + atomic append when a
 *            true 9-pixel arc exists. */
/* Helper: 5-op bitmask check returning a 16-bit mask whose bit `k` is 1 iff
 * bits `k..(k+8) mod 16` are all set in `m` (16-bit). */
static nir_def *
fast9_run9_mask(nir_builder *b, nir_def *m)
{
   nir_def *m2 = nir_ior(b, m, nir_ishl_imm(b, m, 16));
   nir_def *a = nir_iand(b, m2, nir_ushr_imm(b, m2, 1));
   a = nir_iand(b, a, nir_ushr_imm(b, a, 2));
   a = nir_iand(b, a, nir_ushr_imm(b, a, 4));
   return nir_iand_imm(b, nir_iand(b, a, nir_ushr_imm(b, m2, 8)), 0xffff);
}

/* Helper: 4-op bitmask check for 4 consecutive set even bits (`0,2,...,14`)
 * in a 16-bit circular mask (necessary condition for any 9-pixel arc). */
static nir_def *
fast9_has_run4_even16(nir_builder *b, nir_def *m)
{
   nir_def *m2 = nir_ior(b, m, nir_ishl_imm(b, m, 16));
   nir_def *a = nir_iand(b, m2, nir_ushr_imm(b, m2, 2));
   return nir_ine_imm(
      b, nir_iand_imm(b, nir_iand(b, a, nir_ushr_imm(b, a, 4)), 0x5555), 0);
}

/* Direct-L1 32x8 FAST-9 corner detector (WG 32x8 = 256, 1 pixel/lane, 0 B
 * shared memory, no workgroup barrier):
 *  - Stage 1: parallel coalesced L1 loads of center + 4 cardinals (S, E, N, W)
 *    with same-sign adjacent-pair filter `(S|N) & (E|W)` for bright/dark
 *    (necessary condition for any 9-pixel arc; rejects 1D edges immediately).
 *  - Stage 2: 4 diagonal loads `{2, 6, 10, 14}` + 4-op circular run-of-4 check
 *    on the 8 even circle points (rejects ~85-90% of Stage 1 survivors before
 *    loading the remaining 8 odd points).
 *  - Stage 3: 8 odd circle loads (row-grouped to minimize live row registers)
 *    + 5-op circular run-of-9 bitmask check before the 24-step arc loop.
 *  - Stage 4 (<0.5% actual corners): nibble-LUT 24-step exact score directly
 *    from hot L1 cache, keeping peak GPR pressure low for high occupancy.
 *  - Single converged coalesced score store per pixel. */
static void
build_fast9_u8_kernel(nir_builder *b, const struct hk_2d_stencil_plan *plan,
                      nir_address_format ssbo_addr_format)
{
   const unsigned TILE_W = 32;
   const unsigned TILE_H = 8;

   b->shader->info.workgroup_size[0] = TILE_W;
   b->shader->info.workgroup_size[1] = TILE_H;
   b->shader->info.workgroup_size[2] = 1;

   nir_deref_instr *src_u8 = build_ssbo_base_deref(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1);
   nir_deref_instr *score_u32 = build_ssbo_base_deref(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint_type(), 4);
   nir_deref_instr *cnt_u32 = build_ssbo_base_deref(
      b, &plan->bufs[2], ssbo_addr_format, glsl_uint_type(), 4);
   nir_deref_instr *pts_i32x2 = build_ssbo_base_deref(
      b, &plan->bufs[3], ssbo_addr_format, glsl_ivec2_type(), 8);

   nir_def *width = load_pc_i32(b, plan->width_pc_offset_B);
   nir_def *height = load_pc_i32(b, plan->height_pc_offset_B);
   nir_def *stride = load_pc_i32(b, plan->stride_pc_offset_B);
   nir_def *border = load_pc_i32(b, plan->border_pc_offset_B);
   nir_def *thresh = load_pc_i32(b, plan->threshold_pc_offset_B);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *px = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 0), 5),
                          nir_channel(b, lid, 0));
   nir_def *py = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 1), 3),
                          nir_channel(b, lid, 1));

   nir_push_if(b, nir_iand(b, nir_ilt(b, px, width), nir_ilt(b, py, height)));
   {
      nir_variable *score_v =
         nir_local_variable_create(b->impl, glsl_uint_type(), "fsc");
      nir_store_var(b, score_v, nir_imm_int(b, 0), 0x1);

      nir_def *min_xy = nir_imax(b, nir_imm_int(b, 3), border);
      nir_def *x_max =
         nir_imin(b, nir_iadd_imm(b, width, -4), nir_isub(b, width, border));
      nir_def *y_max =
         nir_imin(b, nir_iadd_imm(b, height, -4), nir_isub(b, height, border));
      nir_def *is_int = nir_iand(
         b, nir_iand(b, nir_ige(b, px, min_xy), nir_ige(b, x_max, px)),
         nir_iand(b, nir_ige(b, py, min_xy), nir_ige(b, y_max, py)));

      nir_push_if(b, is_int);
      {
         nir_def *base_idx =
            nir_iadd(b, nir_imul(b, nir_iadd(b, py, border), stride),
                     nir_iadd(b, px, border));
         nir_def *s3 = nir_imul_imm(b, stride, 3);

         /* Stage 1: Load center + 4 cardinals (E=4, W=12, S=0, N=8) in
          * parallel. Every 9-pixel arc must contain two adjacent same-sign
          * cardinals: `(E|W) & (S|N)` bright OR `(E|W) & (S|N)` dark. */
         nir_def *c_val = nir_u2u32(b, load_ssbo_u8(b, src_u8, base_idx));
         nir_def *pe =
            nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd_imm(b, base_idx, 3)));
         nir_def *pw = nir_u2u32(
            b, load_ssbo_u8(b, src_u8, nir_iadd_imm(b, base_idx, -3)));
         nir_def *ps =
            nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd(b, base_idx, s3)));
         nir_def *pn =
            nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_isub(b, base_idx, s3)));

         nir_def *c_hi = nir_iadd(b, c_val, thresh);
         nir_def *c_lo = nir_isub(b, c_val, thresh);

         nir_def *e_hi = nir_ilt(b, c_hi, pe);
         nir_def *w_hi = nir_ilt(b, c_hi, pw);
         nir_def *s_hi = nir_ilt(b, c_hi, ps);
         nir_def *n_hi = nir_ilt(b, c_hi, pn);
         nir_def *bright_card =
            nir_iand(b, nir_ior(b, e_hi, w_hi), nir_ior(b, s_hi, n_hi));

         nir_def *e_lo = nir_ilt(b, pe, c_lo);
         nir_def *w_lo = nir_ilt(b, pw, c_lo);
         nir_def *s_lo = nir_ilt(b, ps, c_lo);
         nir_def *n_lo = nir_ilt(b, pn, c_lo);
         nir_def *dark_card =
            nir_iand(b, nir_ior(b, e_lo, w_lo), nir_ior(b, s_lo, n_lo));

         nir_push_if(b, nir_ior(b, bright_card, dark_card));
         {
            /* Seed 16-bit circle masks with the 4 cardinals (0, 4, 8, 12). */
            nir_def *m_hi = nir_ior(
               b,
               nir_ior(b, nir_bcsel(b, s_hi, nir_imm_int(b, 1u << 0), nir_imm_int(b, 0)),
                       nir_bcsel(b, e_hi, nir_imm_int(b, 1u << 4), nir_imm_int(b, 0))),
               nir_ior(b, nir_bcsel(b, n_hi, nir_imm_int(b, 1u << 8), nir_imm_int(b, 0)),
                       nir_bcsel(b, w_hi, nir_imm_int(b, 1u << 12), nir_imm_int(b, 0))));
            nir_def *m_lo = nir_ior(
               b,
               nir_ior(b, nir_bcsel(b, s_lo, nir_imm_int(b, 1u << 0), nir_imm_int(b, 0)),
                       nir_bcsel(b, e_lo, nir_imm_int(b, 1u << 4), nir_imm_int(b, 0))),
               nir_ior(b, nir_bcsel(b, n_lo, nir_imm_int(b, 1u << 8), nir_imm_int(b, 0)),
                       nir_bcsel(b, w_lo, nir_imm_int(b, 1u << 12), nir_imm_int(b, 0))));

            /* Stage 2: Load the 4 diagonal points (2, 14 on row +2; 6, 10 on
             * row -2) and check for 4 consecutive set even bits (necessary for
             * any 9-pixel arc). */
            nir_def *s2 = nir_ishl_imm(b, stride, 1);
            nir_def *r_p2 = nir_iadd(b, base_idx, s2);
            nir_def *p2 =
               nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd_imm(b, r_p2, 2)));
            nir_def *p14 =
               nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd_imm(b, r_p2, -2)));
            nir_def *r_m2 = nir_isub(b, base_idx, s2);
            nir_def *p6 =
               nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd_imm(b, r_m2, 2)));
            nir_def *p10 =
               nir_u2u32(b, load_ssbo_u8(b, src_u8, nir_iadd_imm(b, r_m2, -2)));

            m_hi = nir_ior(
               b, m_hi,
               nir_ior(
                  b,
                  nir_ior(b, nir_bcsel(b, nir_ilt(b, c_hi, p2), nir_imm_int(b, 1u << 2), nir_imm_int(b, 0)),
                          nir_bcsel(b, nir_ilt(b, c_hi, p6), nir_imm_int(b, 1u << 6), nir_imm_int(b, 0))),
                  nir_ior(b, nir_bcsel(b, nir_ilt(b, c_hi, p10), nir_imm_int(b, 1u << 10), nir_imm_int(b, 0)),
                          nir_bcsel(b, nir_ilt(b, c_hi, p14), nir_imm_int(b, 1u << 14), nir_imm_int(b, 0)))));
            m_lo = nir_ior(
               b, m_lo,
               nir_ior(
                  b,
                  nir_ior(b, nir_bcsel(b, nir_ilt(b, p2, c_lo), nir_imm_int(b, 1u << 2), nir_imm_int(b, 0)),
                          nir_bcsel(b, nir_ilt(b, p6, c_lo), nir_imm_int(b, 1u << 6), nir_imm_int(b, 0))),
                  nir_ior(b, nir_bcsel(b, nir_ilt(b, p10, c_lo), nir_imm_int(b, 1u << 10), nir_imm_int(b, 0)),
                          nir_bcsel(b, nir_ilt(b, p14, c_lo), nir_imm_int(b, 1u << 14), nir_imm_int(b, 0)))));

            nir_def *has4_hi = fast9_has_run4_even16(b, m_hi);
            nir_def *has4_lo = fast9_has_run4_even16(b, m_lo);

            nir_push_if(b, nir_ior(b, has4_hi, has4_lo));
            {
               /* Stage 3: Load remaining 8 odd points grouped by row so at
                * most one divergent row address is live at a time:
                *   row +3: 1(+1), 15(-1)
                *   row +1: 3(+3), 13(-3)
                *   row -1: 5(+3), 11(-3)
                *   row -3: 7(+1),  9(-1) */
               static const int8_t odd_rows[4] = {3, 1, -1, -3};
               static const uint8_t odd_idx[4][2] = {
                  {1, 15}, {3, 13}, {5, 11}, {7, 9},
               };
               for (unsigned r = 0; r < 4; r++) {
                  nir_def *rb = nir_iadd(b, base_idx,
                                         nir_imul_imm(b, stride, odd_rows[r]));
                  for (unsigned s = 0; s < 2; s++) {
                     unsigned j = odd_idx[r][s];
                     nir_def *pv = nir_u2u32(
                        b, load_ssbo_u8(b, src_u8,
                                        nir_iadd_imm(b, rb, fast9_dx[j])));
                     nir_def *jb = nir_imm_int(b, 1u << j);
                     m_hi = nir_ior(
                        b, m_hi,
                        nir_bcsel(b, nir_ilt(b, c_hi, pv), jb,
                                  nir_imm_int(b, 0)));
                     m_lo = nir_ior(
                        b, m_lo,
                        nir_bcsel(b, nir_ilt(b, pv, c_lo), jb,
                                  nir_imm_int(b, 0)));
                  }
               }

               /* Since thresh >= 0, m_hi & m_lo == 0, so at most one can hold
                * >= 9 bits; when one has a 9-bit run, pop(hi)-pop(lo) has the
                * exact same sign (> 0 for bright, < 0 for dark). */
               nir_def *r9_hi = fast9_run9_mask(b, m_hi);
               nir_def *r9_lo = fast9_run9_mask(b, m_lo);
               nir_push_if(b, nir_ine_imm(b, nir_ior(b, r9_hi, r9_lo), 0));
               {
                  /* Stage 4 (<0.5% actual corners): exact 24-step arc score
                   * via 4-bit biased nibble LUT (+3) reading hot L1 cache. */
                  nir_def *is_bright = nir_ine_imm(b, r9_hi, 0);
                  nir_variable *ai_v = nir_local_variable_create(
                     b->impl, glsl_int_type(), "fai");
                  nir_variable *amin_v = nir_local_variable_create(
                     b->impl, glsl_int_type(), "famin");
                  nir_variable *best_v = nir_local_variable_create(
                     b->impl, glsl_int_type(), "fbest");
                  nir_variable *rem_v = nir_local_variable_create(
                     b->impl, glsl_int_type(), "frem");
                  nir_store_var(b, ai_v, nir_imm_int(b, 0), 0x1);
                  nir_store_var(b, amin_v, thresh, 0x1);
                  nir_store_var(b, best_v, nir_imm_int(b, 0), 0x1);
                  nir_store_var(b, rem_v, nir_imm_int(b, 9), 0x1);

                  nir_loop *al = nir_push_loop(b);
                  al->control = nir_loop_control_dont_unroll;
                  {
                     nir_def *ai = nir_load_var(b, ai_v);
                     nir_push_if(b, nir_ige_imm(b, ai, 24));
                     nir_jump(b, nir_jump_break);
                     nir_pop_if(b, NULL);

                     nir_def *idx16 = nir_iand_imm(b, ai, 15);
                     nir_def *lo8 = nir_ult_imm(b, idx16, 8);
                     nir_def *sh = nir_ishl_imm(b, nir_iand_imm(b, idx16, 7), 2);
                     nir_def *dx_w = nir_bcsel(b, lo8, nir_imm_int(b, 0x45666543u),
                                               nir_imm_int(b, 0x21000123u));
                     nir_def *dy_w = nir_bcsel(b, lo8, nir_imm_int(b, 0x01234566u),
                                               nir_imm_int(b, 0x65432100u));
                     nir_def *dx_i = nir_iadd_imm(
                        b, nir_iand_imm(b, nir_ushr(b, dx_w, sh), 0xf), -3);
                     nir_def *dy_i = nir_iadd_imm(
                        b, nir_iand_imm(b, nir_ushr(b, dy_w, sh), 0xf), -3);
                     nir_def *addr = nir_iadd(
                        b, nir_iadd(b, base_idx, nir_imul(b, dy_i, stride)),
                        dx_i);
                     nir_def *pv = nir_u2u32(b, load_ssbo_u8(b, src_u8, addr));
                     nir_def *sub_cp = nir_isub(b, c_val, pv);
                     nir_def *diff =
                        nir_bcsel(b, is_bright, nir_ineg(b, sub_cp), sub_cp);

                     nir_def *amin = nir_load_var(b, amin_v);
                     nir_def *best = nir_load_var(b, best_v);
                     nir_def *rem = nir_load_var(b, rem_v);

                     nir_def *pass = nir_ilt(b, thresh, diff);
                     nir_def *mv = nir_bcsel(b, nir_ieq_imm(b, rem, 9), diff,
                                             nir_imin(b, amin, diff));
                     nir_def *mc = nir_bcsel(b, pass, mv, thresh);
                     nir_def *rn =
                        nir_bcsel(b, pass, nir_iadd_imm(b, rem, -1),
                                  nir_imm_int(b, 9));
                     nir_def *rz = nir_ieq_imm(b, rn, 0);

                     nir_store_var(
                        b, ai_v,
                        nir_iadd_imm(
                           b, nir_bcsel(b, rz, nir_iadd_imm(b, ai, -8), ai), 1),
                        0x1);
                     nir_store_var(b, amin_v, nir_bcsel(b, rz, thresh, mc), 0x1);
                     nir_store_var(
                        b, best_v,
                        nir_bcsel(b, rz, nir_imax(b, best, mc), best), 0x1);
                     nir_store_var(b, rem_v,
                                   nir_bcsel(b, rz, nir_imm_int(b, 9), rn), 0x1);
                  }
                  nir_pop_loop(b, NULL);

                  nir_store_var(
                     b, score_v,
                     nir_i2f32(b, nir_iadd_imm(b, nir_load_var(b, best_v), -1)),
                     0x1);

                  nir_deref_instr *cnt_elem = nir_build_deref_ptr_as_array(
                     b, cnt_u32, nir_imm_int(b, 0));
                  nir_def *pt_idx = nir_deref_atomic(
                     b, 32, &cnt_elem->def, nir_imm_int(b, 1),
                     .atomic_op = nir_atomic_op_iadd);
                  nir_deref_instr *pt_elem =
                     nir_build_deref_ptr_as_array(b, pts_i32x2, pt_idx);
                  nir_store_deref_with_access(b, pt_elem, nir_vec2(b, px, py),
                                              0x3, ACCESS_NON_READABLE);
               }
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);

      /* Single converged coalesced score store. */
      store_ssbo_u32(b, score_u32, nir_iadd(b, nir_imul(b, py, width), px),
                     nir_load_var(b, score_v));
   }
   nir_pop_if(b, NULL);
}

nir_shader *
hk_nir_build_tiled_2d_stencil(void *mem_ctx, const nir_shader *orig_nir,
                              const struct hk_2d_stencil_plan *plan,
                              nir_address_format ssbo_addr_format)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, orig_nir->options, "%s_tiled_2d",
      orig_nir->info.name ? orig_nir->info.name : "stencil_2d");
   b.shader->info.internal = false;
   ralloc_steal(mem_ctx, b.shader);

   b.shader->info.zero_initialize_shared_memory =
      orig_nir->info.zero_initialize_shared_memory;

   switch (plan->kind) {
   case HK_2D_STENCIL_BOX_BLUR_U8:
      build_box_blur_u8_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_2D_STENCIL_STEREO_SAD_U8:
      build_stereo_sad_u8_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_2D_STENCIL_SOBEL_3X3_U8:
      build_sobel_3x3_u8_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_2D_STENCIL_FAST9_U8:
      build_fast9_u8_kernel(&b, plan, ssbo_addr_format);
      break;
   default:
      ralloc_free(b.shader);
      return NULL;
   }

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "after hk_nir_build_tiled_2d_stencil");
   return b.shader;
}

/* ---------------------------------------------------------------------------
 * Dispatch-time validation
 * --------------------------------------------------------------------------- */

static bool
resolve_ssbo_binding(const struct hk_descriptor_state *desc,
                     const struct hk_2d_stencil_binding *b,
                     struct hk_buffer_address *addr_out)
{
   if (b->set >= HK_MAX_SETS)
      return false;
   if (desc->buffer_active & BITFIELD_BIT(b->set))
      return false;

   if (desc->push_active & BITFIELD_BIT(b->set)) {
      const struct hk_push_descriptor_set *push = desc->push[b->set];
      if (!push)
         return false;
      if (memcmp(push->layout_hash, b->set_layout_hash,
                 sizeof(push->layout_hash)) != 0)
         return false;
      if ((uint64_t)b->descriptor_offset_B + sizeof(*addr_out) > push->size)
         return false;
      memcpy(addr_out, push->data + b->descriptor_offset_B, sizeof(*addr_out));
   } else {
      const struct hk_descriptor_set *ds = desc->sets[b->set];
      if (!ds || !ds->layout || !ds->mapped_ptr)
         return false;
      if (b->binding >= ds->layout->binding_count)
         return false;
      const struct hk_descriptor_set_binding_layout *bl =
         &ds->layout->binding[b->binding];
      if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          bl->offset != b->descriptor_offset_B)
         return false;
      if (bl->flags &
          (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT))
         return false;
      if ((uint64_t)b->descriptor_offset_B + sizeof(*addr_out) > ds->size)
         return false;
      memcpy(addr_out, (const uint8_t *)ds->mapped_ptr + b->descriptor_offset_B,
             sizeof(*addr_out));
   }

   if (addr_out->base_addr == 0 || addr_out->sparse_map_page != 0 ||
       addr_out->size == 0)
      return false;
   if (addr_out->base_addr > UINT64_MAX - addr_out->size)
      return false;

   return true;
}

static bool
va_ranges_disjoint(uint64_t a_base, uint64_t a_len, uint64_t b_base,
                   uint64_t b_len)
{
   if (a_base > UINT64_MAX - a_len || b_base > UINT64_MAX - b_len)
      return false;
   return (a_base + a_len <= b_base) || (b_base + b_len <= a_base);
}

static bool
read_written_push_i32(const struct hk_descriptor_state *desc, uint16_t offset_B,
                      int32_t *val_out)
{
   if ((offset_B & 3u) != 0 || offset_B + 4u > HK_MAX_PUSH_SIZE)
      return false;
   unsigned dword = offset_B >> 2;
   if (dword >= 64 || !(desc->push_written_dwords & (UINT64_C(1) << dword)))
      return false;
   memcpy(val_out, desc->root.push + offset_B, sizeof(*val_out));
   return true;
}

bool
hk_2d_stencil_validate_dispatch_core(
   bool alias_free, const struct hk_descriptor_state *desc,
   const struct hk_2d_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y)
{
   if (!alias_free || !desc || !plan)
      return false;

   if (base_x != 0 || base_y != 0 || base_z != 0)
      return false;
   if (groups_x == 0 || groups_y == 0 || groups_z != 1)
      return false;
   if (plan->tile_w == 0 || plan->tile_h == 0)
      return false;

   int32_t width = 0, height = 0, stride = 0;
   if (!read_written_push_i32(desc, plan->width_pc_offset_B, &width) ||
       !read_written_push_i32(desc, plan->height_pc_offset_B, &height) ||
       !read_written_push_i32(desc, plan->stride_pc_offset_B, &stride))
      return false;

   if (width < 1 || height < 1 || stride < width)
      return false;

   if (plan->kind != HK_2D_STENCIL_FAST9_U8) {
      if ((uint64_t)groups_x * plan->orig_wg_size[0] < (uint64_t)width ||
          (uint64_t)groups_y * plan->orig_wg_size[1] < (uint64_t)height)
         return false;
   }

   const uint64_t img_bytes =
      (uint64_t)(height - 1) * (uint64_t)stride + (uint64_t)width;
   if (img_bytes > (uint64_t)INT32_MAX)
      return false;

   switch (plan->kind) {
   case HK_2D_STENCIL_BOX_BLUR_U8: {
      int32_t radius = 0;
      if (!read_written_push_i32(desc, plan->radius_pc_offset_B, &radius))
         return false;
      if (radius < 0 || (uint32_t)radius > HK_2D_STENCIL_MAX_RADIUS)
         return false;

      if ((uint32_t)width >
             (uint32_t)INT32_MAX - (plan->tile_w + 2u * HK_2D_STENCIL_MAX_RADIUS + 4u) ||
          (uint32_t)height >
             (uint32_t)INT32_MAX - (plan->tile_h + 2u * HK_2D_STENCIL_MAX_RADIUS))
         return false;

      struct hk_buffer_address src_addr = {0}, dst_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &src_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &dst_addr))
         return false;
      if (src_addr.size < img_bytes || dst_addr.size < img_bytes)
         return false;
      if (!va_ranges_disjoint(dst_addr.base_addr, img_bytes, src_addr.base_addr,
                              img_bytes))
         return false;

      /* The u8x4 row reads clamp to the last 4 columns. */
      if (width < 4)
         return false;

      *out_groups_x =
         DIV_ROUND_UP((uint32_t)width, hk_box_blur_out_w((uint32_t)radius));
      *out_groups_y = DIV_ROUND_UP((uint32_t)height, (uint32_t)plan->tile_h);
      return true;
   }
   case HK_2D_STENCIL_STEREO_SAD_U8: {
      int32_t win_size = 0, d_min = 0, d_max = 0, cost_stride = 0;
      if (!read_written_push_i32(desc, plan->radius_pc_offset_B, &win_size) ||
          !read_written_push_i32(desc, plan->d_min_pc_offset_B, &d_min) ||
          !read_written_push_i32(desc, plan->d_max_pc_offset_B, &d_max) ||
          !read_written_push_i32(desc, plan->cost_stride_pc_offset_B,
                                 &cost_stride))
         return false;

      if (win_size < 1 || cost_stride < width)
         return false;
      const uint32_t radius = (uint32_t)win_size >> 1;
      if (radius > HK_2D_STENCIL_MAX_RADIUS)
         return false;

      /* Prevent signed 32-bit wrap in `d0 + 32`, `x_start - d1`, and
       * `tile_x + radius + TILE_W`. */
      if (d_min < INT32_MIN + 1024 || d_max > INT32_MAX - 1024)
         return false;
      if ((uint32_t)width >
             (uint32_t)INT32_MAX - (plan->tile_w + 2u * HK_2D_STENCIL_MAX_RADIUS + 64u) ||
          (uint32_t)height >
             (uint32_t)INT32_MAX - (plan->tile_h + 2u * HK_2D_STENCIL_MAX_RADIUS))
         return false;
      const int64_t abs_dmax = d_max < 0 ? -(int64_t)d_max : (int64_t)d_max;
      const int64_t abs_dmin = d_min < 0 ? -(int64_t)d_min : (int64_t)d_min;
      if ((int64_t)width + abs_dmax + 256 > INT32_MAX ||
          (int64_t)width + abs_dmin + 256 > INT32_MAX)
         return false;

      const uint64_t cost_elems =
         (uint64_t)(height - 1) * (uint64_t)cost_stride + (uint64_t)width;
      if (cost_elems > (uint64_t)INT32_MAX / 4u)
         return false;
      const uint64_t cost_bytes = cost_elems * 4u;

      struct hk_buffer_address a[6] = {0};
      for (unsigned i = 0; i < 6; i++) {
         if (!resolve_ssbo_binding(desc, &plan->bufs[i], &a[i]))
            return false;
      }

      /* left(0), right(1), prev_disp(2), dst_disp(4) are u8;
       * prev_cost(3), dst_cost(5) are u32 (must be 4-byte aligned). */
      if (a[0].size < img_bytes || a[1].size < img_bytes ||
          a[2].size < img_bytes || a[4].size < img_bytes ||
          a[3].size < cost_bytes || a[5].size < cost_bytes ||
          (a[3].base_addr & 3u) != 0 || (a[5].base_addr & 3u) != 0)
         return false;

      /* dst_disp(4) and dst_cost(5) must be disjoint from left(0), right(1),
       * and each other, and cross-disjoint from prev_cost(3)/prev_disp(2).
       * In-place updates (prev_disp == dst_disp or prev_cost == dst_cost with
       * identical base) are sound because each pixel reads before writing its
       * own index exclusively. */
      if (!va_ranges_disjoint(a[4].base_addr, img_bytes, a[0].base_addr,
                              img_bytes) ||
          !va_ranges_disjoint(a[4].base_addr, img_bytes, a[1].base_addr,
                              img_bytes) ||
          !va_ranges_disjoint(a[4].base_addr, img_bytes, a[3].base_addr,
                              cost_bytes) ||
          !va_ranges_disjoint(a[4].base_addr, img_bytes, a[5].base_addr,
                              cost_bytes) ||
          !va_ranges_disjoint(a[5].base_addr, cost_bytes, a[0].base_addr,
                              img_bytes) ||
          !va_ranges_disjoint(a[5].base_addr, cost_bytes, a[1].base_addr,
                              img_bytes) ||
          !va_ranges_disjoint(a[5].base_addr, cost_bytes, a[2].base_addr,
                              img_bytes))
         return false;

      const bool disp_same = (a[2].base_addr == a[4].base_addr);
      const bool cost_same = (a[3].base_addr == a[5].base_addr);
      if (!disp_same && !va_ranges_disjoint(a[4].base_addr, img_bytes,
                                            a[2].base_addr, img_bytes))
         return false;
      if (!cost_same && !va_ranges_disjoint(a[5].base_addr, cost_bytes,
                                            a[3].base_addr, cost_bytes))
         return false;

      *out_groups_x = DIV_ROUND_UP((uint32_t)width, (uint32_t)plan->tile_w);
      *out_groups_y = DIV_ROUND_UP((uint32_t)height, (uint32_t)plan->tile_h);
      return true;
   }
   case HK_2D_STENCIL_SOBEL_3X3_U8: {
      if ((uint32_t)width > (uint32_t)INT32_MAX - (plan->tile_w + 16u) ||
          (uint32_t)height > (uint32_t)INT32_MAX - (plan->tile_h + 16u))
         return false;

      const uint64_t out_elems = (uint64_t)width * (uint64_t)height;
      if (out_elems > (uint64_t)INT32_MAX / 4u)
         return false;
      const uint64_t out_bytes = out_elems * 4u;

      struct hk_buffer_address src_addr = {0}, dir_addr = {0}, mag_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &src_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &dir_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[2], &mag_addr))
         return false;

      if (src_addr.size < img_bytes || dir_addr.size < out_bytes ||
          mag_addr.size < out_bytes || (dir_addr.base_addr & 3u) != 0 ||
          (mag_addr.base_addr & 3u) != 0)
         return false;

      if (!va_ranges_disjoint(dir_addr.base_addr, out_bytes, src_addr.base_addr,
                              img_bytes) ||
          !va_ranges_disjoint(mag_addr.base_addr, out_bytes, src_addr.base_addr,
                              img_bytes) ||
          !va_ranges_disjoint(dir_addr.base_addr, out_bytes, mag_addr.base_addr,
                              out_bytes))
         return false;

      *out_groups_x = DIV_ROUND_UP((uint32_t)width, (uint32_t)plan->tile_w);
      *out_groups_y = DIV_ROUND_UP((uint32_t)height, (uint32_t)plan->tile_h);
      return true;
   }
   case HK_2D_STENCIL_FAST9_U8: {
      int32_t border = 0, threshold = 0;
      if (!read_written_push_i32(desc, plan->border_pc_offset_B, &border) ||
          !read_written_push_i32(desc, plan->threshold_pc_offset_B, &threshold))
         return false;

      if (border < 0 || border > 1024 || threshold < 0 || threshold > 255)
         return false;

      if ((uint32_t)width >
             (uint32_t)INT32_MAX - (plan->tile_w + (uint32_t)border + 16u) ||
          (uint32_t)height >
             (uint32_t)INT32_MAX - (plan->tile_h + (uint32_t)border + 16u))
         return false;

      const uint64_t score_elems = (uint64_t)width * (uint64_t)height;
      if (score_elems > (uint64_t)INT32_MAX / 4u)
         return false;
      const uint64_t score_bytes = score_elems * 4u;

      struct hk_buffer_address a[4] = {0};
      for (unsigned i = 0; i < 4; i++) {
         if (!resolve_ssbo_binding(desc, &plan->bufs[i], &a[i]))
            return false;
      }

      /* Verify `src` covers the maximum interior circle coordinate. */
      const int32_t min_xy = border > 3 ? border : 3;
      const int32_t x_max =
         (width - 4 < width - border) ? (width - 4) : (width - border);
      const int32_t y_max =
         (height - 4 < height - border) ? (height - 4) : (height - border);
      if (x_max >= min_xy && y_max >= min_xy) {
         const uint64_t max_src_x = (uint64_t)x_max + (uint64_t)border + 3u;
         const uint64_t max_src_y = (uint64_t)y_max + (uint64_t)border + 3u;
         if (max_src_x >= (uint64_t)stride)
            return false;
         const uint64_t need_src =
            max_src_y * (uint64_t)stride + max_src_x + 1u;
         if (need_src > (uint64_t)INT32_MAX || a[0].size < need_src)
            return false;
      }

      if (a[1].size < score_bytes || (a[1].base_addr & 3u) != 0 ||
          a[2].size < 4u || (a[2].base_addr & 3u) != 0 ||
          a[3].size < 8u || (a[3].base_addr & 3u) != 0)
         return false;

      for (unsigned i = 0; i < 4; i++) {
         for (unsigned j = i + 1; j < 4; j++) {
            if (!va_ranges_disjoint(a[i].base_addr, a[i].size,
                                    a[j].base_addr, a[j].size))
               return false;
         }
      }

      *out_groups_x = DIV_ROUND_UP((uint32_t)width, (uint32_t)plan->tile_w);
      *out_groups_y = DIV_ROUND_UP((uint32_t)height, (uint32_t)plan->tile_h);
      return true;
   }
   default:
      return false;
   }
}

bool
hk_2d_stencil_validate_dispatch(const struct hk_device *dev,
                                const struct hk_descriptor_state *desc,
                                const struct hk_2d_stencil_plan *plan,
                                uint32_t base_x, uint32_t base_y,
                                uint32_t base_z, uint32_t groups_x,
                                uint32_t groups_y, uint32_t groups_z,
                                uint32_t *out_groups_x, uint32_t *out_groups_y)
{
   if (!dev)
      return false;

   const bool alias_free =
      p_atomic_read(&dev->va_alias_resources) == 0 &&
      p_atomic_read(&dev->alias_live) == 0 &&
      !hk_cdm_alias_scope(dev);

   return hk_2d_stencil_validate_dispatch_core(
      alias_free, desc, plan, base_x, base_y, base_z, groups_x, groups_y,
      groups_z, out_groups_x, out_groups_y);
}
