/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 *
 * Software lowering for VK_KHR_cooperative_matrix on AGX2. Used for shaders
 * that the hardware path (agx_nir_lower_cmat.c -> simd_matrix_fmadd16/32)
 * cannot serve: the AGX2 matrix unit requires all 32 lanes of a subgroup to
 * be active, so workgroups whose static size is not a multiple of 32 (or with
 * a runtime workgroup size) lower here using subgroup shuffles and scalar ALU.
 *
 * Model (distributed authority): each active invocation holds a private
 * array of rows*cols elements (spilled to scratch if register pressure
 * demands), while logical element e is authoritatively owned by invocation
 * e % A where A = min(32, W - 32*subgroup_id) and W is the workgroup thread
 * count. Tangled operations gather non-owned inputs from the owner lane via
 * nir_shuffle(); untangled operations (insert/extract/copy/bitcast/length) act
 * only on locally owned elements (lane + t*A).
 */

#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "util/hash_table.h"
#include "util/macros.h"

#define AGX_CMAT_MAX_SUBGROUP 32
#define AGX_CMAT_MAX_ELEMS    (16 * 16)

static struct glsl_cmat_description
src_desc(struct hash_table *orig, nir_src s)
{
   nir_deref_instr *deref = nir_src_as_deref(s);
   struct hash_entry *e = _mesa_hash_table_search(orig, deref);
   const struct glsl_type *type = e ? e->data : deref->type;
   assert(glsl_type_is_cmat(type));
   return *glsl_get_cmat_description(type);
}

static const struct glsl_type *
remap_type(struct hash_table *tm, const struct glsl_type *orig)
{
   struct hash_entry *e = _mesa_hash_table_search(tm, orig);
   if (e)
      return e->data;

   const struct glsl_type *nt = orig;
   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description d = *glsl_get_cmat_description(orig);
      nt = glsl_array_type(glsl_scalar_type(d.element_type), d.rows * d.cols, 0);
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *el = glsl_get_array_element(orig);
      const struct glsl_type *ne = remap_type(tm, el);
      if (ne != el) {
         nt = glsl_array_type(ne, glsl_get_length(orig),
                              glsl_get_explicit_stride(orig));
      }
   } else if (glsl_type_is_struct(orig)) {
      unsigned i;
      for (i = 0; i < orig->length; i++) {
         const struct glsl_type *ft = glsl_get_struct_field(orig, i);
         if (remap_type(tm, ft) != ft)
            break;
      }

      if (i < orig->length) {
         struct glsl_struct_field *fields =
            malloc(sizeof(struct glsl_struct_field) * orig->length);
         memcpy(fields, orig->fields.structure,
                sizeof(struct glsl_struct_field) * i);
         for (; i < orig->length; i++) {
            fields[i] = *glsl_get_struct_field_data(orig, i);
            fields[i].type = remap_type(tm, fields[i].type);
         }
         nt = glsl_struct_type(fields, orig->length, glsl_get_type_name(orig),
                               glsl_struct_type_is_packed(orig));
         free(fields);
      }
   }

   _mesa_hash_table_insert(tm, orig, (void *)nt);
   return nt;
}

static nir_deref_instr *
elem_deref(nir_builder *b, nir_deref_instr *mat, nir_def *idx)
{
   return nir_build_deref_array(b, mat, idx);
}

static nir_def *
load_elem(nir_builder *b, nir_deref_instr *mat, unsigned e)
{
   return nir_load_deref(b, elem_deref(b, mat, nir_imm_int(b, e)));
}

static void
store_elem(nir_builder *b, nir_deref_instr *mat, unsigned e, nir_def *v)
{
   nir_store_deref(b, elem_deref(b, mat, nir_imm_int(b, e)), v, 0x1);
}

static nir_def *
cmat_convert_elem(nir_builder *b, nir_def *v,
                  enum glsl_base_type src_type, enum glsl_base_type dst_type)
{
   if (src_type == dst_type)
      return v;

   if (src_type == GLSL_TYPE_BFLOAT16) {
      v = nir_bf2f(b, v);
      if (dst_type == GLSL_TYPE_FLOAT)
         return v;
      src_type = GLSL_TYPE_FLOAT;
   }

   if (dst_type == GLSL_TYPE_BFLOAT16) {
      if (src_type != GLSL_TYPE_FLOAT)
         v = cmat_convert_elem(b, v, src_type, GLSL_TYPE_FLOAT);
      return nir_f2bf(b, v);
   }

   nir_alu_type src_alu = nir_get_nir_type_for_glsl_base_type(src_type);
   nir_alu_type dst_alu = nir_get_nir_type_for_glsl_base_type(dst_type);
   bool is_float_src = nir_alu_type_get_base_type(src_alu) == nir_type_float;
   bool is_float_dst = nir_alu_type_get_base_type(dst_alu) == nir_type_float;
   unsigned src_bits = nir_alu_type_get_type_size(src_alu);
   unsigned dst_bits = nir_alu_type_get_type_size(dst_alu);
   nir_rounding_mode round =
      (is_float_src && is_float_dst && dst_bits < src_bits)
         ? nir_rounding_mode_rtne
         : nir_rounding_mode_undef;
   nir_op op = nir_type_conversion_op(src_alu, dst_alu, round);
   return nir_build_alu1(b, op, v);
}

static nir_def *
cmat_alu_elem(nir_builder *b, nir_op op, enum glsl_base_type type,
              nir_def *x, nir_def *y)
{
   if (type == GLSL_TYPE_BFLOAT16) {
      x = nir_bf2f(b, x);
      if (y)
         y = nir_bf2f(b, y);
   }
   nir_def *res = y ? nir_build_alu2(b, op, x, y) : nir_build_alu1(b, op, x);
   if (type == GLSL_TYPE_BFLOAT16)
      res = nir_f2bf(b, res);
   return res;
}

static nir_def *
active_count(nir_builder *b)
{
   nir_def *sz = nir_load_workgroup_size(b);
   nir_def *w = nir_imul(
      b, nir_channel(b, sz, 0),
      nir_imul(b, nir_channel(b, sz, 1), nir_channel(b, sz, 2)));
   nir_def *before =
      nir_imul_imm(b, nir_load_subgroup_id(b), AGX_CMAT_MAX_SUBGROUP);
   return nir_imin(b, nir_imm_int(b, AGX_CMAT_MAX_SUBGROUP),
                   nir_isub(b, w, before));
}

static nir_def *
owner_next(nir_builder *b, nir_def *owner, nir_def *a)
{
   nir_def *next = nir_iadd_imm(b, owner, 1);
   return nir_bcsel(b, nir_ieq(b, next, a), nir_imm_int(b, 0), next);
}

static nir_def *
gather_elem(nir_builder *b, nir_deref_instr *mat, unsigned e, nir_def *owner)
{
   return nir_shuffle(b, load_elem(b, mat, e), owner);
}

static nir_deref_instr *
mem_deref(nir_builder *b, nir_deref_instr *ptr, nir_def *stride,
          nir_def *major, nir_def *minor, enum glsl_base_type elem,
          unsigned nvec, unsigned scalar_bytes, unsigned tsz)
{
   const unsigned bits = ptr->def.bit_size;
   nir_def *row_comps = nir_udiv_imm(
      b,
      nir_imul_imm(b,
                   nir_imul(b, nir_u2uN(b, major, bits),
                            nir_u2uN(b, stride, bits)),
                   scalar_bytes * nvec),
      tsz);
   nir_deref_instr *c = nir_build_deref_cast(b, &ptr->def, ptr->modes,
                                             glsl_scalar_type(elem), tsz);
   return nir_build_deref_ptr_as_array(
      b, c, nir_iadd(b, row_comps, nir_u2uN(b, minor, bits)));
}

static bool
lower_load(nir_builder *b, struct hash_table *orig, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description desc = src_desc(orig, intr->src[0]);
   enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
   nir_deref_instr *mat = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *ptr = nir_src_as_deref(intr->src[1]);
   nir_def *stride = intr->src[2].ssa;

   unsigned tsz = glsl_base_type_bit_size(desc.element_type) / 8;
   const unsigned nvec = glsl_get_vector_elements(ptr->type);
   const unsigned scalar_bytes = glsl_get_bit_size(ptr->type) / 8;
   bool colmaj = (layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR);

   for (unsigned e = 0; e < desc.rows * desc.cols; e++) {
      unsigned r = e / desc.cols, c = e % desc.cols;
      nir_deref_instr *at = mem_deref(b, ptr, stride,
                                      nir_imm_int(b, colmaj ? c : r),
                                      nir_imm_int(b, colmaj ? r : c),
                                      desc.element_type, nvec,
                                      scalar_bytes, tsz);
      store_elem(b, mat, e, nir_load_deref(b, at));
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_store(nir_builder *b, struct hash_table *orig, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description desc = src_desc(orig, intr->src[1]);
   enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
   nir_deref_instr *mat = nir_src_as_deref(intr->src[1]);
   nir_deref_instr *ptr = nir_src_as_deref(intr->src[0]);
   nir_def *stride = intr->src[2].ssa;

   unsigned tsz = glsl_base_type_bit_size(desc.element_type) / 8;
   const unsigned nvec = glsl_get_vector_elements(ptr->type);
   const unsigned scalar_bytes = glsl_get_bit_size(ptr->type) / 8;
   bool colmaj = (layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR);

   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *a = active_count(b);
   for (unsigned t = 0; t < desc.rows * desc.cols; t++) {
      nir_def *e = nir_iadd(b, lane, nir_imul_imm(b, a, t));
      nir_if *nif =
         nir_push_if(b, nir_ult(b, e, nir_imm_int(b, desc.rows * desc.cols)));
      nir_def *r = nir_udiv_imm(b, e, desc.cols);
      nir_def *c = nir_umod_imm(b, e, desc.cols);
      nir_deref_instr *at = mem_deref(b, ptr, stride,
                                      colmaj ? c : r, colmaj ? r : c,
                                      desc.element_type, nvec,
                                      scalar_bytes, tsz);
      nir_store_deref(b, at,
                      nir_load_deref(b, elem_deref(b, mat, e)), 0x1);
      nir_pop_if(b, nif);
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_muladd(nir_builder *b, struct hash_table *orig, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description da = src_desc(orig, intr->src[1]);
   struct glsl_cmat_description db = src_desc(orig, intr->src[2]);
   struct glsl_cmat_description dc = src_desc(orig, intr->src[3]);
   struct glsl_cmat_description dd = src_desc(orig, intr->src[0]);
   nir_deref_instr *d = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *a = nir_src_as_deref(intr->src[1]);
   nir_deref_instr *mat_b = nir_src_as_deref(intr->src[2]);
   nir_deref_instr *c = nir_src_as_deref(intr->src[3]);

   unsigned m = da.rows, k = da.cols, n = db.cols;
   assert(da.cols == db.rows && da.rows == dc.rows && db.cols == dc.cols);

   if (m * k > AGX_CMAT_MAX_ELEMS || k * n > AGX_CMAT_MAX_ELEMS ||
       m * n > AGX_CMAT_MAX_ELEMS)
      return false;

   /* bf16 inputs/accumulators compute in fp32 and round the result back. */
   const enum glsl_base_type compute_type =
      dc.element_type == GLSL_TYPE_BFLOAT16 ? GLSL_TYPE_FLOAT : dc.element_type;

   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_exact;

   nir_def *a_cnt = active_count(b);
   nir_def *av[AGX_CMAT_MAX_ELEMS], *bv[AGX_CMAT_MAX_ELEMS];
   nir_def *owner = nir_imm_int(b, 0);
   for (unsigned e = 0; e < m * k; e++) {
      av[e] = cmat_convert_elem(b, gather_elem(b, a, e, owner),
                                da.element_type, compute_type);
      owner = owner_next(b, owner, a_cnt);
   }
   owner = nir_imm_int(b, 0);
   for (unsigned e = 0; e < k * n; e++) {
      bv[e] = cmat_convert_elem(b, gather_elem(b, mat_b, e, owner),
                                db.element_type, compute_type);
      owner = owner_next(b, owner, a_cnt);
   }

   for (unsigned e = 0; e < m * n; e++) {
      unsigned i = e / n, j = e % n;
      nir_def *acc = cmat_convert_elem(b, load_elem(b, c, e),
                                       dc.element_type, compute_type);
      for (unsigned kk = 0; kk < k; kk++) {
         if (glsl_base_type_is_integer(compute_type))
            acc = nir_iadd(b, acc, nir_imul(b, av[i * k + kk], bv[kk * n + j]));
         else
            acc = nir_ffma(b, av[i * k + kk], bv[kk * n + j], acc);
      }
      store_elem(b, d, e,
                 cmat_convert_elem(b, acc, compute_type, dd.element_type));
   }

   b->fp_math_ctrl = save;
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_construct(nir_builder *b, struct hash_table *orig,
                nir_intrinsic_instr *intr)
{
   nir_deref_instr *mat = nir_src_as_deref(intr->src[0]);
   unsigned elems = src_desc(orig, intr->src[0]).rows *
                    src_desc(orig, intr->src[0]).cols;

   for (unsigned e = 0; e < elems; e++)
      store_elem(b, mat, e, intr->src[1].ssa);

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_length(nir_builder *b, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description desc = nir_intrinsic_cmat_desc(intr);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *a = active_count(b);
   nir_def *len = nir_udiv(b,
      nir_iadd(b, nir_isub(b, nir_imm_int(b, desc.rows * desc.cols), lane),
                  nir_iadd_imm(b, a, -1)),
      a);
   nir_def_replace(&intr->def, len);
   return true;
}

static bool
lower_alu_op(nir_builder *b, struct hash_table *orig,
             nir_intrinsic_instr *intr, unsigned kind)
{
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *a = nir_src_as_deref(intr->src[1]);
   nir_op op = nir_intrinsic_alu_op(intr);
   struct glsl_cmat_description desc = src_desc(orig, intr->src[1]);
   unsigned elems = desc.rows * desc.cols;

   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   nir_deref_instr *c = kind == 2 ? nir_src_as_deref(intr->src[2]) : NULL;
   for (unsigned e = 0; e < elems; e++) {
      nir_def *y = kind == 0 ? NULL
                   : kind == 1 ? intr->src[2].ssa
                               : load_elem(b, c, e);
      store_elem(b, dst, e,
                 cmat_alu_elem(b, op, desc.element_type, load_elem(b, a, e), y));
   }
   b->fp_math_ctrl = save;

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_extract(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_deref_instr *mat = nir_src_as_deref(intr->src[0]);
   nir_def *e = nir_iadd(b, nir_load_subgroup_invocation(b),
                         nir_imul(b, intr->src[1].ssa, active_count(b)));
   nir_def *v = nir_load_deref(b, elem_deref(b, mat, e));
   if (intr->def.bit_size != v->bit_size)
      v = nir_u2uN(b, v, intr->def.bit_size);
   nir_def_replace(&intr->def, v);
   return true;
}

static bool
lower_insert(nir_builder *b, struct hash_table *orig, nir_intrinsic_instr *intr)
{
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *src = nir_src_as_deref(intr->src[2]);
   nir_def *v = intr->src[1].ssa;
   nir_def *idx = intr->src[3].ssa;
   unsigned elems = src_desc(orig, intr->src[0]).rows *
                    src_desc(orig, intr->src[0]).cols;

   for (unsigned e = 0; e < elems; e++)
      store_elem(b, dst, e, load_elem(b, src, e));

   nir_def *e = nir_iadd(b, nir_load_subgroup_invocation(b),
                         nir_imul(b, idx, active_count(b)));
   nir_store_deref(b, elem_deref(b, dst, e), v, 0x1);

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_copy_bitcast(nir_builder *b, struct hash_table *orig,
                   nir_intrinsic_instr *intr)
{
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *src = nir_src_as_deref(intr->src[1]);
   unsigned elems = src_desc(orig, intr->src[0]).rows *
                    src_desc(orig, intr->src[0]).cols;

   for (unsigned e = 0; e < elems; e++)
      store_elem(b, dst, e, load_elem(b, src, e));

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_convert(nir_builder *b, struct hash_table *orig,
              nir_intrinsic_instr *intr)
{
   bool trans = intr->intrinsic == nir_intrinsic_cmat_transpose;
   struct glsl_cmat_description dd = src_desc(orig, intr->src[0]);
   struct glsl_cmat_description sd = src_desc(orig, intr->src[1]);
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *src = nir_src_as_deref(intr->src[1]);

   if (dd.rows * dd.cols > AGX_CMAT_MAX_ELEMS)
      return false;

   nir_def *a = active_count(b);
   nir_def *sv[AGX_CMAT_MAX_ELEMS];
   nir_def *owner = nir_imm_int(b, 0);
   for (unsigned e = 0; e < dd.rows * dd.cols; e++) {
      unsigned se = trans ? ((e % dd.cols) * sd.cols + e / dd.cols) : e;
      nir_def *own = trans ? nir_umod(b, nir_imm_int(b, se), a) : owner;
      sv[e] = gather_elem(b, src, se, own);
      if (!trans)
         owner = owner_next(b, owner, a);
   }

   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   for (unsigned e = 0; e < dd.rows * dd.cols; e++)
      store_elem(b, dst, e,
                 cmat_convert_elem(b, sv[e], sd.element_type, dd.element_type));
   b->fp_math_ctrl = save;

   nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_nir_lower_cmat_sw(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       !shader->info.cs.has_cooperative_matrix)
      return false;

   bool progress = false;
   struct hash_table *tm = _mesa_pointer_hash_table_create(NULL);
   struct hash_table *orig = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (glsl_type_is_cmat(deref->type))
               _mesa_hash_table_insert(orig, deref, (void *)deref->type);
         }
      }
   }

   nir_foreach_variable_with_modes(var, shader, nir_var_shader_temp)
      var->type = remap_type(tm, var->type);

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_function_temp_variable(var, impl)
         var->type = remap_type(tm, var->type);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref) {
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               deref->type = remap_type(tm, deref->type);
            }
         }
      }

      nir_builder bld = nir_builder_create(impl);
      nir_foreach_block_reverse_safe(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            nir_builder *b = &bld;
            bld.cursor = nir_before_instr(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_cmat_length:
               progress |= lower_length(b, intr);
               break;
            case nir_intrinsic_cmat_construct:
               progress |= lower_construct(b, orig, intr);
               break;
            case nir_intrinsic_cmat_load:
               progress |= lower_load(b, orig, intr);
               break;
            case nir_intrinsic_cmat_store:
               progress |= lower_store(b, orig, intr);
               break;
            case nir_intrinsic_cmat_muladd:
               progress |= lower_muladd(b, orig, intr);
               break;
            case nir_intrinsic_cmat_copy:
            case nir_intrinsic_cmat_bitcast:
               progress |= lower_copy_bitcast(b, orig, intr);
               break;
            case nir_intrinsic_cmat_binary_op:
               progress |= lower_alu_op(b, orig, intr, 2);
               break;
            case nir_intrinsic_cmat_scalar_op:
               progress |= lower_alu_op(b, orig, intr, 1);
               break;
            case nir_intrinsic_cmat_unary_op:
               progress |= lower_alu_op(b, orig, intr, 0);
               break;
            case nir_intrinsic_cmat_extract:
               progress |= lower_extract(b, intr);
               break;
            case nir_intrinsic_cmat_insert:
               progress |= lower_insert(b, orig, intr);
               break;
            case nir_intrinsic_cmat_convert:
            case nir_intrinsic_cmat_transpose:
               progress |= lower_convert(b, orig, intr);
               break;
            default:
               break;
            }
         }
      }
      nir_progress(progress, impl, nir_metadata_none);
   }

   _mesa_hash_table_destroy(tm, NULL);
   _mesa_hash_table_destroy(orig, NULL);
   return progress;
}
