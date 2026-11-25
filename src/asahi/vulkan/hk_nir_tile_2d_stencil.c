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
   if (nir_def_instr(def)->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(def));
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   if (!nir_src_is_const(intr->src[0]))
      return false;

   uint64_t off = (uint64_t)nir_intrinsic_base(intr) +
                  nir_src_as_uint(intr->src[0]);
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
   plan->tile_w = 64;
   plan->tile_h = 16;
   plan->width_pc_offset_B = w_off;
   plan->height_pc_offset_B = h_off;
   plan->stride_pc_offset_B = stride_pc;
   plan->radius_pc_offset_B = r_off;
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
       orig_nir->info.ray_queries > 0)
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
       orig_nir->info.num_ssbos != 6)
      return false;

   nir_shader *s = nir_shader_clone(NULL, orig_nir);
   if (!s)
      return false;

   nir_lower_vars_to_ssa(s);
   nir_lower_system_values(s);
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

/* Separable 2D shared-memory R8 box blur:
 *   Tile 64x16 (1024 px/WG), WG 256x1x1 (8 simdgroups).
 *   Shared memory:
 *     sh_src: 48 rows x 25 u8x4 (= 100 bytes/row, bank-conflict-free) = 4,800 B
 *     sh_col: 16 rows x 97 u16  (odd stride, bank-conflict-free)     = 3,104 B
 *     Total: 7,904 B (< 8 KiB -> full threadgroup occupancy).
 *   Phase 0: coalesced u8x4 cooperative preload into sh_src.
 *   Phase 1: 2*W_EXT threads run vertical sliding window (8 rows/thread) -> sh_col.
 *   Phase 2: 16x16 threads run horizontal sliding window (4 px/thread) + udiv
 *            by uniform (2r+1)^2 + coalesced u8x4 interior store. */
static void
build_box_blur_u8_kernel(nir_builder *b, const struct hk_2d_stencil_plan *plan,
                         nir_address_format ssbo_addr_format)
{
   const unsigned WG = 256;
   const unsigned TILE_W = 64;
   const unsigned TILE_H = 16;
   const unsigned MAX_H_EXT = TILE_H + 2 * HK_2D_STENCIL_MAX_RADIUS; /* 48 */
   const unsigned SH_SRC_U4_STRIDE = 25; /* 100 B >= 64 + 32 = 96 B */
   const unsigned SH_SRC_B_STRIDE = SH_SRC_U4_STRIDE * 4;
   const unsigned SH_COL_STRIDE = 97; /* odd >= 96 */

   b->shader->info.workgroup_size[0] = WG;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_variable *sh_src_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT8, 4),
                      MAX_H_EXT * SH_SRC_U4_STRIDE, 0),
      "sh_src");
   nir_variable *sh_col_var = nir_variable_create(
      b->shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint16_t_type(), TILE_H * SH_COL_STRIDE, 0),
      "sh_col");
   nir_deref_instr *sh_src = nir_build_deref_var(b, sh_src_var);
   nir_deref_instr *sh_col = nir_build_deref_var(b, sh_col_var);

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

   nir_def *two_r = nir_ishl_imm(b, radius, 1);
   nir_def *diam = nir_iadd_imm(b, two_r, 1);
   nir_def *area = nir_imul(b, diam, diam);
   nir_def *w_ext = nir_iadd_imm(b, two_r, TILE_W);
   nir_def *h_ext = nir_iadd_imm(b, two_r, TILE_H);
   nir_def *w4 = nir_ushr_imm(b, nir_iadd_imm(b, w_ext, 3), 2);

   nir_def *tid = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *tile_x = nir_imul_imm(b, nir_channel(b, wgid, 0), TILE_W);
   nir_def *tile_y = nir_imul_imm(b, nir_channel(b, wgid, 1), TILE_H);
   nir_def *x_start = nir_isub(b, tile_x, radius);
   nir_def *y_start = nir_isub(b, tile_y, radius);

   /* Phase 0: Cooperative load of `(TILE_W + 2r) x (TILE_H + 2r)` bytes. */
   nir_def *tile_in_bounds = nir_iand(
      b,
      nir_iand(b, nir_ige_imm(b, x_start, 0),
               nir_ige(b, width, nir_iadd(b, x_start, nir_ishl_imm(b, w4, 2)))),
      nir_iand(b, nir_ige_imm(b, y_start, 0),
               nir_ige(b, height, nir_iadd(b, y_start, h_ext))));

   nir_def *lane32 = nir_iand_imm(b, tid, 31);
   nir_def *warp8 = nir_ushr_imm(b, tid, 5);

   nir_push_if(b, tile_in_bounds);
   {
      /* Fast interior path: branch-free coalesced u8x4 loads per warp. */
      nir_push_if(b, nir_ilt(b, lane32, w4));
      {
         nir_variable *sy_var =
            nir_local_variable_create(b->impl, glsl_int_type(), "sy_fast");
         nir_store_var(b, sy_var, warp8, 0x1);
         nir_def *gx = nir_iadd(b, x_start, nir_ishl_imm(b, lane32, 2));
         nir_loop *rl = nir_push_loop(b);
         rl->control = nir_loop_control_dont_unroll;
         {
            nir_def *sy = nir_load_var(b, sy_var);
            nir_push_if(b, nir_ige(b, sy, h_ext));
            nir_jump(b, nir_jump_break);
            nir_pop_if(b, NULL);

            nir_def *gy = nir_iadd(b, y_start, sy);
            nir_def *v =
               load_ssbo_u8x4(b, src_u8x4, nir_iadd(b, nir_imul(b, gy, stride), gx));
            store_sh_u8x4(
               b, sh_src,
               nir_iadd(b, nir_imul_imm(b, sy, SH_SRC_U4_STRIDE), lane32), v);

            nir_store_var(b, sy_var, nir_iadd_imm(b, sy, 8), 0x1);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      /* Cold border tile path: bounded scalar byte loads. */
      nir_variable *sy_var =
         nir_local_variable_create(b->impl, glsl_int_type(), "sy_bord");
      nir_store_var(b, sy_var, warp8, 0x1);
      nir_loop *rl = nir_push_loop(b);
      rl->control = nir_loop_control_dont_unroll;
      {
         nir_def *sy = nir_load_var(b, sy_var);
         nir_push_if(b, nir_ige(b, sy, h_ext));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         nir_def *gy = nir_iadd(b, y_start, sy);
         nir_push_if(b, nir_iand(b, nir_ige_imm(b, gy, 0),
                                 nir_ilt(b, gy, height)));
         {
            nir_def *row_off = nir_imul(b, gy, stride);
            nir_def *sh_row = nir_imul_imm(b, sy, SH_SRC_B_STRIDE);
            nir_variable *cx_var =
               nir_local_variable_create(b->impl, glsl_int_type(), "cx_bord");
            nir_store_var(b, cx_var, lane32, 0x1);
            nir_loop *cl = nir_push_loop(b);
            cl->control = nir_loop_control_dont_unroll;
            {
               nir_def *cx = nir_load_var(b, cx_var);
               nir_push_if(b, nir_ige(b, cx, w_ext));
               nir_jump(b, nir_jump_break);
               nir_pop_if(b, NULL);

               nir_def *gx = nir_iadd(b, x_start, cx);
               nir_push_if(b, nir_iand(b, nir_ige_imm(b, gx, 0),
                                       nir_ilt(b, gx, width)));
               {
                  nir_def *val = load_ssbo_u8(b, src_u8, nir_iadd(b, row_off, gx));
                  store_sh_u8x4_byte(b, sh_src, nir_iadd(b, sh_row, cx), val);
               }
               nir_pop_if(b, NULL);

               nir_store_var(b, cx_var, nir_iadd_imm(b, cx, 32), 0x1);
            }
            nir_pop_loop(b, NULL);
         }
         nir_pop_if(b, NULL);

         nir_store_var(b, sy_var, nir_iadd_imm(b, sy, 8), 0x1);
      }
      nir_pop_loop(b, NULL);
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   /* Phase 1: Vertical sliding window -> `sh_col[16][97]` (2 halves x 8 rows). */
   nir_def *half = nir_ushr_imm(b, tid, 7);
   nir_def *cx = nir_iand_imm(b, tid, 127);
   nir_push_if(b, nir_ilt(b, cx, w_ext));
   {
      nir_def *row0 = nir_ishl_imm(b, half, 3); /* 0 or 8 */
      nir_variable *acc_var =
         nir_local_variable_create(b->impl, glsl_uint_type(), "vacc");
      nir_variable *k_var =
         nir_local_variable_create(b->impl, glsl_int_type(), "vk");
      nir_store_var(b, acc_var, nir_imm_int(b, 0), 0x1);
      nir_store_var(b, k_var, nir_imm_int(b, 0), 0x1);

      nir_loop *vl = nir_push_loop(b);
      vl->control = nir_loop_control_dont_unroll;
      {
         nir_def *k = nir_load_var(b, k_var);
         nir_push_if(b, nir_ige(b, k, diam));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         nir_def *sy = nir_iadd(b, row0, k);
         nir_def *byte = load_sh_u8x4_byte(
            b, sh_src, nir_iadd(b, nir_imul_imm(b, sy, SH_SRC_B_STRIDE), cx));
         nir_store_var(b, acc_var,
                       nir_iadd(b, nir_load_var(b, acc_var), nir_u2u32(b, byte)),
                       0x1);
         nir_store_var(b, k_var, nir_iadd_imm(b, k, 1), 0x1);
      }
      nir_pop_loop(b, NULL);

      nir_def *acc = nir_load_var(b, acc_var);
      store_sh_u16(b, sh_col,
                   nir_iadd(b, nir_imul_imm(b, row0, SH_COL_STRIDE), cx),
                   nir_u2u16(b, acc));

      for (unsigned j = 1; j < 8; j++) {
         nir_def *sy_add = nir_iadd_imm(b, nir_iadd(b, row0, two_r), j);
         nir_def *sy_sub = nir_iadd_imm(b, row0, j - 1);
         nir_def *b_add = load_sh_u8x4_byte(
            b, sh_src,
            nir_iadd(b, nir_imul_imm(b, sy_add, SH_SRC_B_STRIDE), cx));
         nir_def *b_sub = load_sh_u8x4_byte(
            b, sh_src,
            nir_iadd(b, nir_imul_imm(b, sy_sub, SH_SRC_B_STRIDE), cx));
         acc = nir_isub(b, nir_iadd(b, acc, nir_u2u32(b, b_add)),
                        nir_u2u32(b, b_sub));
         store_sh_u16(
            b, sh_col,
            nir_iadd(b, nir_imul_imm(b, nir_iadd_imm(b, row0, j), SH_COL_STRIDE),
                     cx),
            nir_u2u16(b, acc));
      }
   }
   nir_pop_if(b, NULL);

   emit_shared_barrier(b);

   /* Phase 2: Horizontal sliding window (16x16 threads, 4 px/thread) + store. */
   nir_def *bx = nir_iand_imm(b, tid, 15);
   nir_def *row = nir_ushr_imm(b, tid, 4);
   nir_def *lx0 = nir_ishl_imm(b, bx, 2);
   nir_def *x0 = nir_iadd(b, tile_x, lx0);
   nir_def *py = nir_iadd(b, tile_y, row);

   nir_push_if(b, nir_ilt(b, py, height));
   {
      nir_def *col_row = nir_imul_imm(b, row, SH_COL_STRIDE);
      nir_variable *hsum_var =
         nir_local_variable_create(b->impl, glsl_uint_type(), "hsum");
      nir_variable *hk_var =
         nir_local_variable_create(b->impl, glsl_int_type(), "hk");
      nir_store_var(b, hsum_var, nir_imm_int(b, 0), 0x1);
      nir_store_var(b, hk_var, nir_imm_int(b, 0), 0x1);

      nir_loop *hl = nir_push_loop(b);
      hl->control = nir_loop_control_dont_unroll;
      {
         nir_def *k = nir_load_var(b, hk_var);
         nir_push_if(b, nir_ige(b, k, diam));
         nir_jump(b, nir_jump_break);
         nir_pop_if(b, NULL);

         nir_def *c = load_sh_u16(
            b, sh_col, nir_iadd(b, col_row, nir_iadd(b, lx0, k)));
         nir_store_var(b, hsum_var,
                       nir_iadd(b, nir_load_var(b, hsum_var), nir_u2u32(b, c)),
                       0x1);
         nir_store_var(b, hk_var, nir_iadd_imm(b, k, 1), 0x1);
      }
      nir_pop_loop(b, NULL);

      nir_def *hsum = nir_load_var(b, hsum_var);
      nir_def *out_p[4];
      out_p[0] = nir_u2u8(b, nir_udiv(b, hsum, area));
      for (unsigned t = 1; t < 4; t++) {
         nir_def *c_add = load_sh_u16(
            b, sh_col,
            nir_iadd(b, col_row,
                     nir_iadd_imm(b, nir_iadd(b, lx0, two_r), t)));
         nir_def *c_sub = load_sh_u16(
            b, sh_col, nir_iadd(b, col_row, nir_iadd_imm(b, lx0, t - 1)));
         hsum = nir_isub(b, nir_iadd(b, hsum, nir_u2u32(b, c_add)),
                         nir_u2u32(b, c_sub));
         out_p[t] = nir_u2u8(b, nir_udiv(b, hsum, area));
      }

      nir_def *max_int_x = nir_isub(b, width, radius);
      nir_def *max_int_y = nir_isub(b, height, radius);
      nir_def *y_interior =
         nir_iand(b, nir_ige(b, py, radius), nir_ilt(b, py, max_int_y));
      nir_def *all_interior = nir_iand(
         b, y_interior,
         nir_iand(b, nir_ige(b, x0, radius),
                  nir_ilt(b, nir_iadd_imm(b, x0, 3), max_int_x)));
      nir_def *dst_row = nir_imul(b, py, stride);

      nir_push_if(b, all_interior);
      {
         nir_def *packed = nir_vec(b, out_p, 4);
         store_ssbo_u8x4(b, dst_u8x4, nir_iadd(b, dst_row, x0), packed);
      }
      nir_push_else(b, NULL);
      {
         for (unsigned t = 0; t < 4; t++) {
            nir_def *px = nir_iadd_imm(b, x0, t);
            nir_push_if(b, nir_ilt(b, px, width));
            {
               nir_def *off = nir_iadd(b, dst_row, px);
               nir_def *is_int = nir_iand(
                  b, y_interior,
                  nir_iand(b, nir_ige(b, px, radius), nir_ilt(b, px, max_int_x)));
               nir_push_if(b, is_int);
               {
                  store_ssbo_u8(b, dst_u8, off, out_p[t]);
               }
               nir_push_else(b, NULL);
               {
                  store_ssbo_u8(b, dst_u8, off, load_ssbo_u8(b, src_u8, off));
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

   if ((uint64_t)groups_x * plan->orig_wg_size[0] < (uint64_t)width ||
       (uint64_t)groups_y * plan->orig_wg_size[1] < (uint64_t)height)
      return false;

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
