/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 *
 * Lower cooperative-matrix to the Apple G13 HARDWARE matrix instruction
 * (simd_matrix_fmadd16/32), instead of the software shuffle+FMA path
 * (agx_nir_lower_cmat.c, ~12x slower). One SIMD-group-wide 8x8x8 tile = ONE
 * instruction: D = A*B + C, row-major, no transpose.
 *
 * The instruction was RE'd from Metal's simdgroup_multiply_accumulate codegen
 * (dougallj/applegpu SimdMatrixFMadd32InstructionDesc; macOS phase). Encoding +
 * packer are in agx_opcodes.py / agx_pack.c (byte-exact verified). This pass
 * provides the NIR->instruction lowering.
 *
 * FRAGMENT LAYOUT (one 8x8 sub-block across 32 lanes, 2 consecutive elems/lane):
 *   row = ((lane>>4)&1)*4 + ((lane>>1)&3)
 *   col = ((lane>>3)&1)*4 + (lane&1)*2    (lane holds (row,col) and (row,col+1))
 * fp32: the 2 elements are a register pair (vec2). VERIFIED 2/2 vs golden_vectors
 * on GPU (8x8 fp32). See FRAGMENT_LAYOUT.md.
 *
 * SIZES: N x N x N for N in {8, 16}. 8x8 = one HW tile. 16x16 is tiled into a
 * 2x2 grid of 8x8 sub-blocks (nblk = N/8): each lane holds nblk*nblk register
 * pairs (vec(N*N/32): vec2 for 8x8, vec8 for 16x16). Sub-block (bi,bj) lives in
 * vec channels [sb*2, sb*2+1] (sb = bi*nblk+bj) and maps to matrix rows
 * [bi*8,bi*8+8) x cols [bj*8,bj*8+8) using the verified 8x8 layout above.
 * muladd does blocked GEMM: D[di][dj] = C[di][dj] + sum_k A[di][k]*B[k][dj],
 * i.e. nblk^3 HW fmadds (8 for 16x16).
 *
 * The M1 integer-input battery covered 48 shape/type/layout cases, including
 * live-C and D != C forms. It does not establish bit-exact results for
 * arbitrary floating-point inputs or other GPU generations.
 *
 * Precision: the accumulator type selects the opcode (fp16 C -> fmadd16, fp32
 * C -> fmadd32, in agx_compile.c); A/B are passed at their own width. Verified
 * operand forms: fmadd32 with fp32 or fp16 A/B (the fp16 A/B, fp32 C form is
 * what Apple's own mul_mm emits), fmadd16 with fp16 A/B. fp32 A/B with an fp16
 * accumulator is not advertised and not lowered.
 *
 * Opt-in via AGX_SIMDMAT=1 (default off). The hardware tile needs ALL 32
 * lanes of a subgroup, so this path is only correct when every subgroup of
 * the dispatch is fully populated: the caller (agx_preprocess_nir) routes
 * here only for compute shaders with a static workgroup size that is a
 * multiple of 32; everything else (partial tail subgroup, runtime-variable
 * size) lowers through the software path in agx_nir_lower_cmat.c instead.
 * A valid shader is never rejected.
 */

#include "util/macros.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"

#define SIMDMAT_SUBGROUP 32

/* rows*cols / subgroup_size. For 8x8 over 32 lanes = 2 elements/lane. */
static unsigned
simdmat_len(struct glsl_cmat_description desc)
{
   return (desc.rows * desc.cols) / SIMDMAT_SUBGROUP;
}

static const struct glsl_type *
remap_type(struct hash_table *mapping, const struct glsl_type *orig)
{
   struct hash_entry *e = _mesa_hash_table_search(mapping, orig);
   if (e)
      return e->data;

   const struct glsl_type *nt = orig;
   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description d = *glsl_get_cmat_description(orig);
      nt = glsl_vector_type(d.element_type, simdmat_len(d)); /* vec2 fp32 */
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *el = glsl_get_array_element(orig);
      const struct glsl_type *ne = remap_type(mapping, el);
      if (ne != el)
         nt = glsl_array_type(ne, glsl_get_length(orig),
                              glsl_get_explicit_stride(orig));
   }
   _mesa_hash_table_insert(mapping, orig, (void *)nt);
   return nt;
}

static struct glsl_cmat_description
src_desc(nir_src s)
{
   return *glsl_get_cmat_description(nir_src_as_deref(s)->type);
}

static nir_def *
load_src(nir_builder *b, nir_src s)
{
   nir_deref_instr *d = nir_src_as_deref(s);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(d->type);
   return nir_build_load_deref(b, simdmat_len(desc),
                               glsl_base_type_bit_size(desc.element_type),
                               &d->def, 0);
}

static void
store_src(nir_builder *b, nir_src s, nir_def *v)
{
   nir_store_deref(b, nir_src_as_deref(s), v,
                   nir_component_mask(v->num_components));
}

/* lane -> logical (row,col) of the fragment's first element (2nd is col+1). */
static void
frag_rc(nir_builder *b, nir_def *lane, nir_def **row, nir_def **col)
{
   *row = nir_iadd(b,
                   nir_imul_imm(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 4), 1), 4),
                   nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3));
   *col = nir_iadd(b,
                   nir_imul_imm(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 3), 1), 4),
                   nir_imul_imm(b, nir_iand_imm(b, lane, 1), 2));
}

static bool
lower_load_store(nir_builder *b, struct hash_table *tm, nir_intrinsic_instr *intr)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
   struct glsl_cmat_description desc = src_desc(intr->src[!is_load]);
   enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
   nir_deref_instr *cmat_deref = nir_src_as_deref(intr->src[!is_load]);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[is_load]);
   nir_def *stride = intr->src[2].ssa;

   /* For inline Q4 decode: the coopMatLoad pointer already points at the tile's
    * base element a[base_off]; capture base_off (buffer-element units) so the
    * decode can form the ABSOLUTE linear weight index = base_off + flat. */
   nir_def *base_off = NULL;
   {
      nir_deref_instr *od = nir_src_as_deref(intr->src[is_load]);
      while (od && od->deref_type == nir_deref_type_cast)
         od = nir_deref_instr_parent(od);
      if (od && (od->deref_type == nir_deref_type_ptr_as_array ||
                 od->deref_type == nir_deref_type_array))
         base_off = od->arr.index.ssa;
   }

   unsigned tsz = glsl_base_type_bit_size(desc.element_type) / 8;
   /* Keep the buffer's OWN element type (may be a vecN -- e.g. mul_mm's vec2
    * shared tiles) so the caller-supplied stride/index, which are in
    * buffer-element units, address correctly. We descend to a scalar only for
    * the contiguous within-row index below. Casting to scalar here (as a
    * previous version did) made the major stride off by the buffer vec width,
    * which silently corrupted every vec2-backed load/store. */
   const unsigned nvec = glsl_get_vector_elements(deref->type);
   const unsigned scalar_bytes = glsl_get_bit_size(deref->type) / 8;
   const uint32_t ptr_stride = scalar_bytes * nvec;
   deref = nir_build_deref_cast(b, &deref->def, deref->modes, deref->type,
                                ptr_stride);
   const struct glsl_type *ct = remap_type(tm, cmat_deref->type);
   cmat_deref =
      nir_build_deref_cast(b, &cmat_deref->def, cmat_deref->modes, ct, 0);

   /* The buffer element width need not match the matrix component width (an
    * fp32 matrix behind a uint16_t* / uint8_t*). Addressing below counts
    * buffer elements; the component-typed sdref steps by tsz, so indices are
    * rescaled exactly to component steps -- matrix data alignment makes the
    * division exact. */
   const bool width_matched = scalar_bytes == tsz;


   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *row, *col;
   frag_rc(b, lane, &row, &col);

   unsigned idx_bits = deref->def.bit_size;
   nir_def *str = nir_u2uN(b, stride, idx_bits);
   unsigned nblk = desc.rows / 8;    /* 1 for 8x8, 2 for 16x16 */
   unsigned len = simdmat_len(desc); /* nblk*nblk*2 */
   nir_def *vals[8];

   if (!is_load) {
      nir_def *src = load_src(b, intr->src[!is_load]);
      for (unsigned e = 0; e < len; e++)
         vals[e] = nir_channel(b, src, e);
   }

   /* Flatten the two-level (major ptr_as_array -> cast -> minor ptr_as_array)
    * address into a single scalar index = major*nvec + minor on a scalar-cast
    * deref built ONCE. Same byte address (major steps whole vecN elements =
    * nvec scalars; minor steps scalars), but one address computation per
    * element instead of two -- the lowering's address arithmetic is the
    * matmul's #1 cost (pt7). Verified MUL_MAT 947/947 + PPL 1.0162,
    * pp512 177.1 -> 179.4. */
   nir_deref_instr *sdref =
      nir_build_deref_cast(b, &deref->def, deref->modes,
                           glsl_scalar_type(desc.element_type), tsz);

   /* VEC2 fragment loads (opt-in via AGX_HWMAT_VEC2): load the contiguous
    * (col,col+1) row-major pair as ONE i16,xy load instead of two i16,x scalars.
    * pt11 found this perf-neutral ALONE, but macOS RE of Apple's pp512=270 kernel
    * (RECIPE 2026-06-22) shows Apple emits paired i16,xy loads, ALL hoisted into one
    * block before a tight 8x4 simd_matrix burst. The pairing is the ENABLER: it gives
    * 24 distinct fragment regs (vs ~12 reused) so the post-RA scheduler can hoist all
    * loads (kills the 42 waits -> ~1). flat is even (frag_rc col is always x2) so the
    * pair is vec2-aligned. Row-major (A) only; col-major (B) pair is strided. */
   const bool vec2ld = getenv("AGX_HWMAT_VEC2") != NULL && width_matched;
   nir_deref_instr *v2dref =
      vec2ld
         ? nir_build_deref_cast(b, &deref->def, deref->modes,
                                glsl_vector_type(desc.element_type, 2), 2 * tsz)
         : NULL;

   /* Walk the nblk x nblk grid of 8x8 sub-blocks; each lane handles its 2
    * fragment elements per sub-block. Sub-block (bi,bj) -> matrix rows/cols
    * [bi*8,..) x [bj*8,..), stored at vec channels [sb*2, sb*2+1]. */
   for (unsigned bi = 0; bi < nblk; bi++) {
      for (unsigned bj = 0; bj < nblk; bj++) {
         unsigned sb = bi * nblk + bj;
         for (unsigned e = 0; e < 2; e++) {
            /* logical element (bi*8+row, bj*8+col+e). Pointer contract (as
             * in the lvp lowering): the Stride operand and the major axis
             * count pointer elements (scalar_bytes*nvec bytes each); the
             * contiguous minor axis counts matrix components (tsz each). */
            bool colmaj = (layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR);
            nir_def *lr = nir_u2uN(b, nir_iadd_imm(b, row, bi * 8), idx_bits);
            nir_def *lc =
               nir_u2uN(b, nir_iadd_imm(b, col, bj * 8 + e), idx_bits);
            nir_def *major = nir_imul(b, colmaj ? lc : lr, str);
            nir_def *minor = colmaj ? lr : lc;
            nir_deref_instr *it;
            /* Component index = row_bytes / tsz + minor, with the row counted
             * in pointer elements (major includes the Stride multiply). The
             * division is by a constant and exact whenever the row stride is
             * component-aligned, which the cooperative matrix memory alignment
             * rules require; for width-matched buffers it folds to
             * major*nvec + minor. No byte-width accesses are emitted. */
            nir_def *row_comps = nir_udiv_imm(
               b, nir_imul_imm(b, major, scalar_bytes * nvec), tsz);
            nir_def *flat = nir_iadd(b, row_comps, minor);
            it = nir_build_deref_ptr_as_array(b, sdref, flat);
            /* INLINE Q4_0 DECODE (env-gated, A-operand): instead of loading f16,
             * decode a quantized weight directly from a 4-byte-aligned padded-Q4_0
             * buffer (block = 20 bytes = 10 u16: [d:f16][16B qs][2B pad]) using the
             * linear weight index `flat`. Eliminates the f16 dequant-staging phase
             * that bounds the Q4 matmul (LAB_NOTEBOOK pt43-45). PROTOTYPE: assumes
             * deref base = buffer start (flat = absolute linear index). */
            if (is_load && getenv("AGX_DECODE_Q4") &&
                desc.use == GLSL_CMAT_USE_A && width_matched) {
               /* AMORTIZED: decode BOTH elements (e=0,1) of this sub-block from ONE
                * scale + ONE qbyte load (they are the lo/hi nibble of the same byte
                * in the padded-Q4 format). Only run at e==0; e==1 is filled here. */
               if (e == 0) {
                  nir_deref_instr *u16d = nir_build_deref_cast(
                     b, &deref->def, deref->modes, glsl_uint16_t_type(), 2);
                  nir_def *aidx = base_off
                     ? nir_iadd(b, nir_u2uN(b, base_off, idx_bits), flat) : flat;
                  nir_def *blk = nir_ushr_imm(b, aidx, 5);       /* /32 */
                  nir_def *within = nir_iand_imm(b, aidx, 31);   /* %32 (even) */
                  nir_def *base = nir_imul_imm(b, blk, 10);      /* u16 units */
                  nir_def *d = nir_load_deref(
                     b, nir_build_deref_ptr_as_array(b, sdref, base));
                  nir_def *bb = nir_ushr_imm(b, within, 1);      /* byte index 0..15 */
                  nir_def *qoff = nir_iadd(b, nir_iadd_imm(b, base, 1),
                                           nir_u2uN(b, nir_ushr_imm(b, bb, 1), idx_bits));
                  nir_def *qv32 = nir_u2u32(b, nir_load_deref(
                     b, nir_build_deref_ptr_as_array(b, u16d, qoff)));
                  /* byte = (bb&1)? hi8 : lo8 */
                  nir_def *byte = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, bb, 1), 0),
                                            nir_ushr_imm(b, qv32, 8),
                                            nir_iand_imm(b, qv32, 0xFF));
                  nir_def *nlo = nir_iand_imm(b, byte, 0xF);          /* e=0 */
                  nir_def *nhi = nir_iand_imm(b, nir_ushr_imm(b, byte, 4), 0xF); /* e=1 */
                  nir_def *flo = nir_i2f16(b, nir_iadd_imm(b, nir_u2u32(b, nlo), -8));
                  nir_def *fhi = nir_i2f16(b, nir_iadd_imm(b, nir_u2u32(b, nhi), -8));
                  vals[sb * 2 + 0] = nir_fmul(b, flo, d);
                  vals[sb * 2 + 1] = nir_fmul(b, fhi, d);
               }
            } else if (is_load && v2dref && !colmaj) {
               /* VEC2: load the contiguous (col,col+1) row-major pair as one
                * i16,xy. flat is even -> vec2 index = flat>>1. Fill both elems at
                * e==0; e==1 is a no-op. Gives paired loads + distinct regs so the
                * post-RA scheduler can hoist them (RECIPE 2026-06-22). */
               if (e == 0) {
                  nir_def *pv = nir_load_deref(
                     b, nir_build_deref_ptr_as_array(b, v2dref,
                                                     nir_ushr_imm(b, flat, 1)));
                  vals[sb * 2 + 0] = nir_channel(b, pv, 0);
                  vals[sb * 2 + 1] = nir_channel(b, pv, 1);
               }
            } else if (is_load)
               vals[sb * 2 + e] = nir_load_deref(b, it);
            else
               nir_store_deref(b, it, vals[sb * 2 + e], 0x1);
         }
      }
   }

   if (is_load)
      nir_store_deref(b, cmat_deref, nir_vec(b, vals, len),
                      nir_component_mask(len));
   nir_instr_remove(&intr->instr);
   return true;
}

/* Extract 8x8 sub-block sb (its vec2 register pair) from a packed fragment. */
static nir_def *
subblk(nir_builder *b, nir_def *v, unsigned sb)
{
   return nir_vec2(b, nir_channel(b, v, sb * 2), nir_channel(b, v, sb * 2 + 1));
}

static bool
lower_muladd(nir_builder *b, nir_intrinsic_instr *intr)
{
   /* src order: dst, A, B, C. A/B element type may differ from the accumulator
    * (f32acc shape: fp16 A/B, fp32 C/D). */
   struct glsl_cmat_description da = src_desc(intr->src[1]); /* A */
   struct glsl_cmat_description dc = src_desc(intr->src[3]); /* C / accumulator */
   unsigned nblk = da.rows / 8;
   unsigned acc_bits = glsl_base_type_bit_size(dc.element_type);
   unsigned in_bits = glsl_base_type_bit_size(da.element_type);

   nir_def *a = load_src(b, intr->src[1]);
   nir_def *bb = load_src(b, intr->src[2]);
   nir_def *c = load_src(b, intr->src[3]);

   /* The HW op accumulates in the C type and takes A/B at their own width
    * (fp16 A/B with fp32 C is the mixed-precision form Apple's own mul_mm
    * emits: simd_matrix_fmadd32 rN_rN+1, rMl_rMh, rKl_rKh, rN_rN+1). Only a
    * wider A/B than C (fp32 A/B, fp16 acc) needs a convert, and no shape
    * advertises that. */
   assert(in_bits <= acc_bits);

   /* Blocked GEMM over the 8x8 sub-blocks: D[di][dj] = C[di][dj] +
    * sum_k A[di][k] * B[k][dj]. Each term is ONE HW matrix fmadd
    * (nblk^3 total: 1 for 8x8, 8 for 16x16). */
   unsigned len = simdmat_len(dc);
   nir_def *dch[8];
   for (unsigned di = 0; di < nblk; di++) {
      for (unsigned dj = 0; dj < nblk; dj++) {
         unsigned sb = di * nblk + dj;
         nir_def *acc = subblk(b, c, sb);
         for (unsigned k = 0; k < nblk; k++) {
            nir_def *as = subblk(b, a, di * nblk + k);
            nir_def *bs = subblk(b, bb, k * nblk + dj);
            acc = nir_simd_matrix_fmadd_agx(b, acc_bits, as, bs, acc);
         }
         dch[sb * 2] = nir_channel(b, acc, 0);
         dch[sb * 2 + 1] = nir_channel(b, acc, 1);
      }
   }

   store_src(b, intr->src[0], nir_vec(b, dch, len));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_construct(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(dst->type);
   nir_def *r = nir_replicate(b, intr->src[1].ssa, simdmat_len(desc));
   nir_store_deref(b, dst, r, nir_component_mask(r->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_length(nir_builder *b, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description desc = nir_intrinsic_cmat_desc(intr);
   nir_def_replace(&intr->def, nir_imm_int(b, simdmat_len(desc)));
   return true;
}

/* Element-wise ops on the packed fragment. binary/scalar/unary/extract/insert/
 * bitcast never move data between lanes, so they act identically on the SW and
 * the HW (frag_rc) layouts: just operate per-vec-component. The intrinsic's
 * fp_math_ctrl (NoContraction from SPIR-V, which survives NIR only because
 * nir_builder_opcodes_h now defaults it from the builder) is carried onto the
 * rebuilt ALU so a precise X*s+Y is not contracted into ffma later. */
static bool
lower_binary_op(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *a = load_src(b, intr->src[1]);
   nir_def *c = load_src(b, intr->src[2]);
   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   store_src(b, intr->src[0], nir_build_alu2(b, nir_intrinsic_alu_op(intr), a, c));
   b->fp_math_ctrl = save;
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_unary_op(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *a = load_src(b, intr->src[1]);
   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   store_src(b, intr->src[0], nir_build_alu1(b, nir_intrinsic_alu_op(intr), a));
   b->fp_math_ctrl = save;
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_scalar_op(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *a = load_src(b, intr->src[1]);
   unsigned save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   store_src(b, intr->src[0],
             nir_build_alu2(b, nir_intrinsic_alu_op(intr), a, intr->src[2].ssa));
   b->fp_math_ctrl = save;
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_extract(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *mat = load_src(b, intr->src[0]);
   nir_def_replace(&intr->def, nir_vector_extract(b, mat, intr->src[1].ssa));
   return true;
}

static bool
lower_insert(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *mat = load_src(b, intr->src[2]);
   store_src(b, intr->src[0],
             nir_vector_insert(b, mat, intr->src[1].ssa, intr->src[3].ssa));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_bitcast(nir_builder *b, nir_intrinsic_instr *intr)
{
   store_src(b, intr->src[0], load_src(b, intr->src[1]));
   nir_instr_remove(&intr->instr);
   return true;
}

/* convert: base-type conversion is element-wise (layout-safe). A use-change
 * (A<->B<->Accumulator) or transpose would need a HW-layout-aware cross-lane
 * relayout, which we don't implement yet -- assert so it's loud, not silent
 * corruption. mul_mm's accumulator convert is base-type only (same use). */
static bool
lower_convert(nir_builder *b, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description dd = src_desc(intr->src[0]);
   struct glsl_cmat_description sd = src_desc(intr->src[1]);
   enum glsl_cmat_use du = dd.use, su = sd.use;
   if (du == GLSL_CMAT_USE_ACCUMULATOR)
      du = GLSL_CMAT_USE_A;
   if (su == GLSL_CMAT_USE_ACCUMULATOR)
      su = GLSL_CMAT_USE_A;
   assert(du == su && intr->intrinsic != nir_intrinsic_cmat_transpose &&
          "HWMAT: cmat use-change/transpose not implemented");

   nir_def *src = load_src(b, intr->src[1]);
   nir_def *ret = src;
   if (dd.element_type != sd.element_type) {
      nir_op op = nir_type_conversion_op(
         nir_get_nir_type_for_glsl_base_type(sd.element_type),
         nir_get_nir_type_for_glsl_base_type(dd.element_type),
         nir_rounding_mode_undef);
      unsigned save = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      ret = nir_build_alu1(b, op, src);
      b->fp_math_ctrl = save;
   }
   store_src(b, intr->src[0], ret);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_impl(nir_function_impl *impl, struct hash_table *tm)
{
   bool progress = false;

   nir_foreach_function_temp_variable(var, impl) {
      const struct glsl_type *nt = remap_type(tm, var->type);
      if (nt != var->type) {
         var->type = nt;
         progress = true;
      }
   }

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         b.cursor = nir_before_instr(instr);

         if (instr->type == nir_instr_type_deref) {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            const struct glsl_type *nt = remap_type(tm, deref->type);
            if (nt != deref->type) {
               deref->type = nt;
               progress = true;
            }
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_cmat_length:
            progress |= lower_length(&b, intr);
            break;
         case nir_intrinsic_cmat_construct:
            progress |= lower_construct(&b, intr);
            break;
         case nir_intrinsic_cmat_load:
         case nir_intrinsic_cmat_store:
            progress |= lower_load_store(&b, tm, intr);
            break;
         case nir_intrinsic_cmat_muladd:
            progress |= lower_muladd(&b, intr);
            break;
         case nir_intrinsic_cmat_copy:
            nir_build_copy_deref(&b, intr->src[0].ssa, intr->src[1].ssa);
            nir_instr_remove(instr);
            progress = true;
            break;
         case nir_intrinsic_cmat_binary_op:
            progress |= lower_binary_op(&b, intr);
            break;
         case nir_intrinsic_cmat_unary_op:
            progress |= lower_unary_op(&b, intr);
            break;
         case nir_intrinsic_cmat_scalar_op:
            progress |= lower_scalar_op(&b, intr);
            break;
         case nir_intrinsic_cmat_extract:
            progress |= lower_extract(&b, intr);
            break;
         case nir_intrinsic_cmat_insert:
            progress |= lower_insert(&b, intr);
            break;
         case nir_intrinsic_cmat_bitcast:
            progress |= lower_bitcast(&b, intr);
            break;
         case nir_intrinsic_cmat_convert:
         case nir_intrinsic_cmat_transpose:
            progress |= lower_convert(&b, intr);
            break;
         default:
            break;
         }
      }
   }

   return nir_progress(progress, impl, nir_metadata_none);
}

bool
agx_nir_lower_simdmat(nir_shader *shader, unsigned subgroup_size)
{
   assert(subgroup_size == SIMDMAT_SUBGROUP);
   bool progress = false;
   struct hash_table *tm = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_function_impl(impl, shader) {
      if (lower_impl(impl, tm))
         progress = true;
   }

   _mesa_hash_table_destroy(tm, NULL);
   return progress;
}
