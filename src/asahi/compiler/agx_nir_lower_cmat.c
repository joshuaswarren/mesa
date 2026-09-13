/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 *
 * Software lowering for VK_KHR_cooperative_matrix on AGX. Used for shaders
 * the hardware path (agx_nir_lower_simdmat.c) cannot serve: the G13 matrix
 * tile needs all 32 lanes of a subgroup, so any workgroup whose size is not
 * a static multiple of 32 leaves the last subgroup partially populated.
 * Those are valid shaders; they lower here instead of being rejected.
 *
 * MODEL (distributed authority): every active invocation keeps a private
 * array of rows*cols elements (<= 256 scalars; spills to per-invocation
 * scratch if register pressure demands), but only the slots the invocation
 * OWNS are authoritative: logical element e is owned by invocation e % A,
 * where A = min(32, W - 32*subgroup_id) is the active invocation count of
 * this subgroup (W = dispatched workgroup size, a system value, so runtime
 * sizes and partial tail subgroups are handled).
 *
 * The split follows the SPV_KHR_cooperative_matrix tangled-instruction list
 * (rev 10, sections 2.2.5 / Composite): load, store, muladd, conversions and
 * cooperative-matrix arithmetic are tangled (uniform execution across the
 * scope required); CompositeInsert, Extract and CopyObject are NOT and may
 * execute on a subset of lanes.
 *
 *  - Tangled ops execute on every lane with uniform loop bounds.
 *    load: redundant uniform reads fill every slot identically. muladd and
 *    element-wise arithmetic compute locally, gathering any non-owned source
 *    element from its owner with nir_shuffle(load_elem(src, e), owner(e))
 *    (incremental owner, no division). convert/transpose gather their source
 *    element the same way. Results are written to the local array; only
 *    owned results are ever consumed downstream.
 *  - Untangled ops (insert/extract/copy/bitcast/length) are purely LOCAL:
 *    they touch this invocation's slots only and never shuffle, so they
 *    remain valid even when executed by a subset of lanes. Their local
 *    fragment index i covers owned element lane + i*A; length is
 *    ceil((rows*cols - lane) / A).
 *  - coopMatStore writes only owned elements, e = lane + t*A.
 *
 * Internal element indices are always LOGICAL ROW-MAJOR (e = r*cols + c);
 * the cooperative-matrix memory layout only affects the external address.
 *
 * Precision: NoContraction on element-wise ops is carried onto the rebuilt
 * ALU (the fp_math_ctrl transfer); muladd is a single SPIR-V operation and
 * lowers to ffma by definition. Unlike the hardware path, transpose and
 * use-changing converts are supported: the layout is logical, so they are
 * pure index/type remaps.
 *
 * Memory address contract (same as the lvp lowering): the Stride operand and
 * the row axis count POINTER elements (scalar_bytes * nvec bytes each); the
 * contiguous column axis counts MATRIX components (tsz bytes each).
 *
 * Types are remapped before the intrinsic handlers run (variables and every
 * deref: cmat -> elem[rows*cols] array), so handlers read the ORIGINAL cmat
 * descriptions from the `orig` map captured during that sweep.
 *
 * ponytail: ops unroll to rows*cols (muladd to rows*cols*K) instructions, so
 * a 16x16x16 matrix emits a few thousand nodes. Fine for an opt-in fallback;
 * fixed 32-multiple workgroups route to the hardware path instead.
 */

#include "util/macros.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"

#define AGX_CMAT_MAX_SUBGROUP 32
#define AGX_CMAT_MAX_ELEMS (16 * 16)

/* Original cmat description of a deref, captured before retyping. */
static struct glsl_cmat_description
src_desc(struct hash_table *orig, nir_src s)
{
   nir_deref_instr *deref = nir_src_as_deref(s);
   struct hash_entry *e = _mesa_hash_table_search(orig, deref);
   const struct glsl_type *type = e ? e->data : deref->type;
   assert(glsl_type_is_cmat(type));
   return *glsl_get_cmat_description(type);
}

/* Cooperative matrices lower to a private element array. Types are remapped
 * in place (variables and every deref keep their identity), mirroring the
 * hardware pass, so arrays/structs of cooperative matrices keep one array
 * per element. Internal indices are logical row-major. */
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
      if (ne != el)
         nt = glsl_array_type(ne, glsl_get_length(orig),
                              glsl_get_explicit_stride(orig));
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

/* Active invocation count of the current subgroup. */
static nir_def *
active_count(nir_builder *b)
{
   nir_def *sz = nir_load_workgroup_size(b);
   nir_def *w = nir_imul(b, nir_channel(b, sz, 0),
                            nir_imul(b, nir_channel(b, sz, 1),
                                        nir_channel(b, sz, 2)));
   nir_def *before = nir_imul_imm(b, nir_load_subgroup_id(b),
                                  AGX_CMAT_MAX_SUBGROUP);
   return nir_imin(b, nir_imm_int(b, AGX_CMAT_MAX_SUBGROUP),
                   nir_isub(b, w, before));
}

/* Owner of element e (e % A) as a running value: e increases by one per
 * iteration, so the owner cycles 0,1,..,A-1 without a division. */
static nir_def *
owner_next(nir_builder *b, nir_def *owner, nir_def *a)
{
   nir_def *next = nir_iadd_imm(b, owner, 1);
   return nir_bcsel(b, nir_ieq(b, next, a), nir_imm_int(b, 0), next);
}

/* Authoritative value of element e: gather from the owner lane. The local
 * load may read a non-owned (stale) slot; the shuffle replaces it with the
 * owner's authoritative copy. */
static nir_def *
gather_elem(nir_builder *b, nir_deref_instr *mat, unsigned e, nir_def *owner)
{
   return nir_shuffle(b, load_elem(b, mat, e), owner);
}

/* Memory address of matrix element (row, col) as a component-typed deref.
 * Pointer contract (as in the lvp lowering): the Stride operand and the row
 * axis count pointer elements (scalar_bytes * nvec bytes each); the
 * contiguous column axis counts matrix components (tsz bytes each). The
 * component index = row_bytes / tsz + minor: division by a constant, exact
 * whenever the row stride is component-aligned as the cooperative matrix
 * memory alignment rules require; for width-matched buffers it folds to
 * major * stride * nvec + minor. No byte-width accesses are emitted. */
static nir_deref_instr *
mem_deref(nir_builder *b, nir_deref_instr *ptr, nir_def *stride,
          nir_def *major, nir_def *minor, enum glsl_base_type elem,
          unsigned nvec, unsigned scalar_bytes, unsigned tsz)
{
   nir_def *row_comps = nir_udiv_imm(
      b, nir_imul_imm(b, nir_imul(b, major, stride), scalar_bytes * nvec),
      tsz);
   nir_deref_instr *c = nir_build_deref_cast(b, &ptr->def, ptr->modes,
                                             glsl_scalar_type(elem), tsz);
   return nir_build_deref_ptr_as_array(b, c, nir_iadd(b, row_comps, minor));
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

   /* Redundant uniform reads: every lane fills every slot from the same
    * memory, so owned slots are authoritative. */
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

   /* Only owned elements (e = lane + t*A) are written; each logical element
    * has exactly one owner, so there are no write races. */
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
   struct glsl_cmat_description da = src_desc(orig, intr->src[1]); /* A: MxK */
   nir_deref_instr *d = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *a = nir_src_as_deref(intr->src[1]);
   nir_deref_instr *mat_b = nir_src_as_deref(intr->src[2]);
   nir_deref_instr *c = nir_src_as_deref(intr->src[3]);

   unsigned m = da.rows, k = da.cols, n = src_desc(orig, intr->src[2]).cols;
   struct glsl_cmat_description dc = src_desc(orig, intr->src[3]);

   /* Only N x N x N, N in {8, 16}, is advertised; anything larger would
    * overflow the fixed SSA hoist arrays. Leave it unlowered so it fails
    * loudly instead of corrupting memory. */
   if (m * k > AGX_CMAT_MAX_ELEMS || k * n > AGX_CMAT_MAX_ELEMS ||
       m * n > AGX_CMAT_MAX_ELEMS)
      return false;

   unsigned acc_bits = glsl_base_type_bit_size(dc.element_type);
   unsigned in_bits = glsl_base_type_bit_size(da.element_type);
   nir_op conv = nir_op_mov;
   if (in_bits != acc_bits) {
      /* fp16 A/B with an fp32 accumulator: the HW form computes the products
       * at accumulator width; mirror that. */
      conv = nir_type_conversion_op(
         nir_get_nir_type_for_glsl_base_type(da.element_type),
         nir_get_nir_type_for_glsl_base_type(dc.element_type),
         nir_rounding_mode_undef);
   }

   /* Tangled op: uniform loop, gather each A/B element from its owner lane.
    * Hoisted before any D store so D aliasing an input stays correct. C is
    * local: only the owner's result of each element is consumed. */
   nir_def *a_cnt = active_count(b);
   nir_def *av[AGX_CMAT_MAX_ELEMS], *bv[AGX_CMAT_MAX_ELEMS];
   nir_def *owner = nir_imm_int(b, 0);
   for (unsigned e = 0; e < m * k; e++) {
      av[e] = gather_elem(b, a, e, owner);
      owner = owner_next(b, owner, a_cnt);
   }
   owner = nir_imm_int(b, 0);
   for (unsigned e = 0; e < k * n; e++) {
      bv[e] = gather_elem(b, mat_b, e, owner);
      owner = owner_next(b, owner, a_cnt);
   }

   for (unsigned e = 0; e < m * n; e++) {
      unsigned i = e / n, j = e % n;
      nir_def *acc = load_elem(b, c, e);
      for (unsigned kk = 0; kk < k; kk++) {
         nir_def *x = av[i * k + kk], *y = bv[kk * n + j];
         if (conv != nir_op_mov) {
            x = nir_build_alu1(b, conv, x);
            y = nir_build_alu1(b, conv, y);
         }
         acc = nir_ffma(b, x, y, acc);
      }
      store_elem(b, d, e, acc);
   }

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

   /* Tangled (uniform); x is scalar per the contract. Fill locally so owned
    * slots are authoritative. */
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

/* Element-wise arithmetic acts per element on the local copy. NoContraction
 * is carried from the intrinsic onto the rebuilt ALU so a precise X*s+Y
 * stays unfused. kind: 0 = unary, 1 = scalar (src[2] is a plain value),
 * 2 = binary (src[2] is a matrix). */
static bool
lower_alu_op(nir_builder *b, struct hash_table *orig,
             nir_intrinsic_instr *intr, unsigned kind)
{
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   nir_deref_instr *a = nir_src_as_deref(intr->src[1]);
   nir_op op = nir_intrinsic_alu_op(intr);
   unsigned elems = src_desc(orig, intr->src[0]).rows *
                    src_desc(orig, intr->src[0]).cols;

   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   if (kind == 0) {
      for (unsigned e = 0; e < elems; e++)
         store_elem(b, dst, e, nir_build_alu1(b, op, load_elem(b, a, e)));
   } else if (kind == 1) {
      for (unsigned e = 0; e < elems; e++)
         store_elem(b, dst, e,
                    nir_build_alu2(b, op, load_elem(b, a, e), intr->src[2].ssa));
   } else {
      nir_deref_instr *c = nir_src_as_deref(intr->src[2]);
      for (unsigned e = 0; e < elems; e++)
         store_elem(b, dst, e,
                    nir_build_alu2(b, op, load_elem(b, a, e),
                                   load_elem(b, c, e)));
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

   /* Untangled: purely local. Copy, then write this invocation's owned slot
    * for local index i (logical element lane + i*A). */
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

   /* Untangled (CopyObject): purely local, in-place safe. */
   for (unsigned e = 0; e < elems; e++)
      store_elem(b, dst, e, load_elem(b, src, e));

   nir_instr_remove(&intr->instr);
   return true;
}

/* convert/transpose (tangled): the layout is logical, so a use change is
 * identity and a transpose is a pure index remap: dst (r, c) = src (c, r).
 * Non-owned source slots are gathered from their owner. Sources are hoisted
 * first so D aliasing an input stays correct. */
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
      return false; /* unadvertised shape: fail loudly, not corrupt */

   nir_op op = nir_op_mov;
   if (dd.element_type != sd.element_type) {
      op = nir_type_conversion_op(
         nir_get_nir_type_for_glsl_base_type(sd.element_type),
         nir_get_nir_type_for_glsl_base_type(dd.element_type),
         nir_rounding_mode_undef);
   }

   nir_def *a = active_count(b);
   nir_def *sv[AGX_CMAT_MAX_ELEMS];
   nir_def *owner = nir_imm_int(b, 0);
   for (unsigned e = 0; e < dd.rows * dd.cols; e++) {
      unsigned se = trans ? ((e % dd.cols) * sd.cols + e / dd.cols) : e;
      /* Non-transpose source indices are sequential (running owner); a
       * transpose jumps, so compute the owner directly. */
      nir_def *own = trans ? nir_umod(b, nir_imm_int(b, se), a) : owner;
      sv[e] = gather_elem(b, src, se, own);
      if (!trans)
         owner = owner_next(b, owner, a);
   }

   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   for (unsigned e = 0; e < dd.rows * dd.cols; e++) {
      if (op == nir_op_mov)
         store_elem(b, dst, e, sv[e]);
      else
         store_elem(b, dst, e, nir_build_alu1(b, op, sv[e]));
   }
   b->fp_math_ctrl = save;

   nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_nir_lower_cmat(nir_shader *shader)
{
   bool progress = false;
   struct hash_table *tm = _mesa_pointer_hash_table_create(NULL);
   struct hash_table *orig = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_function_impl(impl, shader) {
      /* Capture original cmat types, then retype variables and every deref,
       * so handlers can index the element arrays through the original deref
       * chains while still reading the cmat descriptions. */
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (glsl_type_is_cmat(deref->type))
               _mesa_hash_table_insert(orig, deref, (void *)deref->type);
         }
      }
      nir_foreach_function_temp_variable(var, impl) {
         const struct glsl_type *nt = remap_type(tm, var->type);
         var->type = nt;
      }
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
