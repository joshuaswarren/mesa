/*
 * Copyright © 2023 Bas Nieuwenhuizen
 * Copyright © 2024 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */


#include "asahi/lib/agx_abi.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "util/set.h"
#include "util/stack_array.h"
#include "util/u_dynarray.h"

static const struct glsl_type *
remap_matrix_type(struct hash_table *mapping, const struct glsl_type *orig)
{
   struct hash_entry *entry = _mesa_hash_table_search(mapping, orig);

   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;

   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description matrix_desc =
         *glsl_get_cmat_description(orig);

      new_type = glsl_vector_type(matrix_desc.element_type,
                                  matrix_desc.rows * matrix_desc.cols / 32);
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *elem_type = glsl_get_array_element(orig);
      const struct glsl_type *new_elem_type =
         remap_matrix_type(mapping, elem_type);

      if (elem_type != new_elem_type) {
         new_type = glsl_array_type(new_elem_type, glsl_get_length(orig),
                                    glsl_get_explicit_stride(orig));
      }
   } else if (glsl_type_is_struct(orig)) {
      unsigned i;
      for (i = 0; i < orig->length; i++) {
         const struct glsl_type *field_type = glsl_get_struct_field(orig, i);
         const struct glsl_type *new_field_type =
            remap_matrix_type(mapping, field_type);

         if (field_type != new_field_type) {
            break;
         }
      }

      
      if (i < orig->length) {
         struct glsl_struct_field *fields =
            malloc(sizeof(struct glsl_struct_field) * orig->length);

         
         memcpy(fields, orig->fields.structure,
                sizeof(struct glsl_struct_field) * i);

         
         for (; i < orig->length; i++) {
            fields[i] = *glsl_get_struct_field_data(orig, i);
            fields[i].type = remap_matrix_type(mapping, fields[i].type);
         }

         new_type =
            glsl_struct_type(fields, orig->length, glsl_get_type_name(orig),
                             glsl_struct_type_is_packed(orig));

         free(fields);
      }
   }

   _mesa_hash_table_insert(mapping, orig, (void *)new_type);
   return new_type;
}

static struct glsl_cmat_description
matrix_desc(nir_src src)
{
   return *glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

static nir_def *
load_matrix(nir_builder *b, nir_src src)
{
   struct glsl_cmat_description desc = matrix_desc(src);
   assert((desc.rows == 8 || desc.rows == 16) &&
          desc.cols == desc.rows && desc.scope == SCOPE_SUBGROUP);
   return nir_build_load_deref(b, desc.rows * desc.cols / 32, glsl_base_type_bit_size(desc.element_type),
                               src.ssa, 0);
}

static void
store_matrix(nir_builder *b, nir_src dst, nir_def *value)
{
   nir_build_store_deref(b, dst.ssa, value, .write_mask = BITFIELD_MASK(value->num_components));
}

static nir_def *
matrix_row(nir_builder *b, nir_def *lane,
           struct glsl_cmat_description desc, unsigned element)
{
   if (desc.rows == 16)
      return nir_ior_imm(b, nir_ior(b, nir_iand_imm(b, lane, 6),
                 nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 8)), element >> 2);
   return nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3),
                    nir_iand_imm(b, nir_ushr_imm(b, lane, 2), 4));
}

static nir_def *
matrix_col(nir_builder *b, nir_def *lane,
           struct glsl_cmat_description desc, unsigned element)
{
   if (desc.cols == 16) {
      if (desc.use == GLSL_CMAT_USE_A)
         return nir_ior_imm(b, nir_ior(b,
            nir_iand_imm(b, nir_ushr_imm(b, lane, 3), 1),
            nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 3)),
            (element & 3) << 1);
      return nir_ior_imm(b, nir_ior(b,
         nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 2),
         nir_iand_imm(b, lane, 8)), element & 3);
   }
   return nir_ior_imm(b, nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 1),
                                   nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 4)),
                         element);
}

/* Where a matrix's elements are: bit i of an element's index within its lane
 * and bit j of its lane's subgroup invocation each hold one bit of its row
 * (ROW(k)) or of its column (COL(k)), as matrix_row() and matrix_col() place
 * them. The accumulator and B layouts of a 16x16 matrix are the same; A
 * differs only in where its column bits are. All uses of an 8x8 matrix share
 * one layout.
 */
#define ROW(k)       (k)
#define COL(k)       (8 + (k))
#define COORD_BIT(c) ((c) & 7)
#define TRANSPOSED(c) ((c) ^ 8)

struct cmat_layout {
   unsigned element_bits;
   uint8_t element[3];
   uint8_t lane[5];
};

static const struct cmat_layout layout_16_a = {
   3, {COL(1), COL(2), ROW(0)}, {COL(3), ROW(1), ROW(2), COL(0), ROW(3)},
};
static const struct cmat_layout layout_16 = {
   3, {COL(0), COL(1), ROW(0)}, {COL(2), ROW(1), ROW(2), COL(3), ROW(3)},
};
static const struct cmat_layout layout_8 = {
   1, {COL(0)}, {COL(1), ROW(0), ROW(1), COL(2), ROW(2)},
};

static const struct cmat_layout *
matrix_layout(struct glsl_cmat_description desc)
{
   assert((desc.rows == 8 || desc.rows == 16) && desc.cols == desc.rows);
   if (desc.rows == 8)
      return &layout_8;
   return desc.use == GLSL_CMAT_USE_A ? &layout_16_a : &layout_16;
}

static int
find_element_bit(const struct cmat_layout *l, unsigned coord)
{
   for (unsigned i = 0; i < l->element_bits; ++i) {
      if (l->element[i] == coord)
         return i;
   }
   return -1;
}

static int
find_lane_bit(const struct cmat_layout *l, unsigned coord)
{
   for (unsigned j = 0; j < 5; ++j) {
      if (l->lane[j] == coord)
         return j;
   }
   return -1;
}

/* The row (or column) of a lane's element `element` */
static nir_def *
layout_coord(nir_builder *b, const struct cmat_layout *l, nir_def *lane,
             unsigned element, bool col)
{
   unsigned imm = 0;
   for (unsigned i = 0; i < l->element_bits; ++i) {
      if ((l->element[i] >= COL(0)) == col && (element & (1 << i)))
         imm |= 1 << COORD_BIT(l->element[i]);
   }
   nir_def *coord = nir_imm_int(b, imm);
   for (unsigned j = 0; j < 5; ++j) {
      if ((l->lane[j] >= COL(0)) != col)
         continue;
      nir_def *bit = nir_iand_imm(b, lane, 1 << j);
      unsigned to = COORD_BIT(l->lane[j]);
      bit = to >= j ? nir_ishl_imm(b, bit, to - j) : nir_ushr_imm(b, bit, j - to);
      coord = nir_ior(b, coord, bit);
   }
   return coord;
}

/* out[i] = subgroupShuffleXor(in[i], mask). The 16-bit values go two to a
 * 32-bit shuffle, which is one instruction for a constant mask.
 */
static void
shuffle_xor_values(nir_builder *b, nir_def **in, nir_def **out, unsigned count,
                   unsigned mask)
{
   nir_def *m = nir_imm_int(b, mask);
   for (unsigned i = 0; i < count; ++i) {
      if (in[i]->bit_size == 32) {
         out[i] = nir_shuffle_xor(b, in[i], m);
      } else if (i + 1 < count) {
         nir_def *pair = nir_shuffle_xor(b, nir_pack_32_2x16_split(b, in[i], in[i + 1]), m);
         out[i] = nir_unpack_32_2x16_split_x(b, pair);
         out[++i] = nir_unpack_32_2x16_split_y(b, pair);
      } else {
         out[i] = nir_u2u16(b, nir_shuffle_xor(b, nir_u2u32(b, in[i]), m));
      }
   }
}

/* Swap what element bit i and lane bit j of the matrix in v hold: of each pair
 * of elements differing in bit i, a lane keeps the one that stays and trades
 * the other with the lane differing in bit j.
 */
static void
exchange_bits(nir_builder *b, nir_def **v, struct cmat_layout *l, unsigned i,
              unsigned j, nir_def *lane)
{
   unsigned count = 1 << l->element_bits, pairs = 0;
   nir_def *upper = nir_i2b(b, nir_iand_imm(b, lane, 1 << j));
   nir_def *send[4], *recv[4];
   for (unsigned e = 0; e < count; ++e) {
      if (!(e & (1 << i)))
         send[pairs++] = nir_bcsel(b, upper, v[e], v[e | (1 << i)]);
   }
   shuffle_xor_values(b, send, recv, pairs, 1 << j);
   pairs = 0;
   for (unsigned e = 0; e < count; ++e) {
      if (e & (1 << i))
         continue;
      v[e] = nir_bcsel(b, upper, recv[pairs], v[e]);
      v[e | (1 << i)] = nir_bcsel(b, upper, v[e | (1 << i)], recv[pairs]);
      pairs++;
   }
   uint8_t tmp = l->element[i];
   l->element[i] = l->lane[j];
   l->lane[j] = tmp;
}

/* Move a matrix's elements from layout `from` to layout `to` (with the same
 * coordinates). Each lane bit that holds the wrong coordinate trades it with
 * the element bit that holds the right one; a coordinate that is in another
 * lane bit goes through an element bit first. What is left is a renaming of
 * the elements within each lane.
 */
static nir_def *
permute_matrix(nir_builder *b, nir_def *value, struct cmat_layout from,
               const struct cmat_layout *to)
{
   nir_def *v[8], *out[8];
   unsigned count = 1 << from.element_bits;
   assert(value->num_components == count && to->element_bits == from.element_bits);
   for (unsigned e = 0; e < count; ++e)
      v[e] = nir_channel(b, value, e);

   nir_def *lane = nir_load_subgroup_invocation(b);
   for (unsigned j = 0; j < 5; ++j) {
      while (from.lane[j] != to->lane[j]) {
         int i = find_element_bit(&from, to->lane[j]);
         if (i >= 0) {
            exchange_bits(b, v, &from, i, j, lane);
            continue;
         }
         /* Bring it in from lane bit k, through the element bit whose
          * coordinate lane bit k wants if there is one.
          */
         int k = find_lane_bit(&from, to->lane[j]);
         assert(k > (int)j);
         i = find_element_bit(&from, to->lane[k]);
         exchange_bits(b, v, &from, i >= 0 ? i : 0, k, lane);
      }
   }
   for (unsigned e = 0; e < count; ++e) {
      unsigned src = 0;
      for (unsigned i = 0; i < from.element_bits; ++i) {
         if (e & (1 << i))
            src |= 1 << find_element_bit(&from, to->element[i]);
      }
      out[e] = v[src];
   }
   return nir_vec(b, out, count);
}

/* The memory of matrix element `element` of this lane, or of the one `skip`
 * elements after it along its row (a column, for a column-major layout).
 */
static nir_deref_instr *
matrix_memory_at(nir_builder *b, nir_intrinsic_instr *intr, unsigned element,
                 unsigned skip)
{
   const bool store = intr->intrinsic == nir_intrinsic_cmat_store;
   struct glsl_cmat_description desc = matrix_desc(intr->src[store ? 1 : 0]);
   nir_deref_instr *ptr = nir_src_as_deref(intr->src[store ? 0 : 1]);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *row = matrix_row(b, lane, desc, element),
           *col = matrix_col(b, lane, desc, element);
   if (nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
      nir_def *tmp = row; row = col; col = tmp;
   }
   unsigned bits = ptr->def.bit_size;
   nir_def *stride = nir_u2uN(b, intr->src[2].ssa, bits);
   nir_def *row_offset = nir_imul(b, nir_u2uN(b, row, bits), stride);
   ptr = nir_build_deref_cast(b, &ptr->def, ptr->modes, ptr->type,
                              glsl_get_vector_elements(ptr->type) *
                              glsl_get_bit_size(ptr->type) / 8);
   ptr = nir_build_deref_ptr_as_array(b, ptr, row_offset);
   ptr = nir_build_deref_cast(b, &ptr->def, ptr->modes,
                              glsl_scalar_type(desc.element_type),
                              glsl_base_type_bit_size(desc.element_type) / 8);
   return nir_build_deref_ptr_as_array(
      b, ptr, nir_u2uN(b, nir_iadd_imm(b, col, skip), bits));
}

static nir_deref_instr *
matrix_memory(nir_builder *b, nir_intrinsic_instr *intr, unsigned element)
{
   return matrix_memory_at(b, intr, element, 0);
}

static nir_def *
convert_matrix_value(nir_builder *b, nir_def *value,
                     enum glsl_base_type src, enum glsl_base_type dst)
{
   if (src == dst)
      return value;
   if (src == GLSL_TYPE_BFLOAT16) {
      value = nir_bf2f(b, value);
      src = GLSL_TYPE_FLOAT;
   }
   if (dst == GLSL_TYPE_BFLOAT16) {
      if (src != GLSL_TYPE_FLOAT)
         value = nir_f2f32(b, value);
      return nir_f2bf(b, value);
   }
   nir_op op = nir_type_conversion_op(nir_get_nir_type_for_glsl_base_type(src),
                                      nir_get_nir_type_for_glsl_base_type(dst),
                                      nir_rounding_mode_undef);
   return nir_build_alu1(b, op, value);
}

static bool
lower_matrix_instr(nir_builder *b, nir_instr *instr, struct hash_table *types)
{
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      const glsl_type *type = remap_matrix_type(types, deref->type);
      bool changed = type != deref->type;
      deref->type = type;
      return changed;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct:
      store_matrix(b, intr->src[0], nir_replicate(b, intr->src[1].ssa,
         matrix_desc(intr->src[0]).rows * matrix_desc(intr->src[0]).cols / 32));
      break;
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store: {
      bool store = intr->intrinsic == nir_intrinsic_cmat_store;
      struct glsl_cmat_description desc = matrix_desc(intr->src[store ? 1 : 0]);
      nir_def *data = store ? load_matrix(b, intr->src[1]) : NULL;
      nir_def *values[8];
      unsigned count = desc.rows * desc.cols / 32;
      bool row_major = nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_ROW_MAJOR;
      unsigned width = desc.rows == 8 ? (row_major ? 2 : 1)
         : (!row_major ? 2 : desc.use == GLSL_CMAT_USE_A ? 1 : 4);
      unsigned component_step = desc.rows == 16 && !row_major ? 4 : 1;
      unsigned component_mask = (width - 1) * component_step;
      /* A lane's four elements of a row of a 16x16 A operand are every other
       * one of seven consecutive columns (matrix_col), so they are in two
       * four-element loads that start at the first and at the fourth of the
       * seven: the first load's elements 0 and 2, the second's 1 and 3.
       * Neither reads outside the lane's seven columns. A store cannot
       * overlap its neighbours' columns and stays scalar.
       */
      if (!store && desc.rows == 16 && row_major &&
          desc.use == GLSL_CMAT_USE_A) {
         unsigned bytes = glsl_base_type_bit_size(desc.element_type) / 8;
         for (unsigned i = 0; i < count; i += 4) {
            for (unsigned half = 0; half < 2; ++half) {
               nir_deref_instr *ptr = matrix_memory_at(b, intr, i, half * 3);
               ptr = nir_build_deref_cast_with_alignment(
                  b, &ptr->def, ptr->modes,
                  glsl_vector_type(desc.element_type, 4), 0, bytes, 0);
               nir_def *loaded = nir_load_deref(b, ptr);
               values[i + half * 2] = nir_channel(b, loaded, half);
               values[i + half * 2 + 1] = nir_channel(b, loaded, half + 2);
            }
         }
         store_matrix(b, intr->src[0], nir_vec(b, values, count));
         break;
      }

      for (unsigned i = 0; i < count; ++i) {
         if (i & component_mask)
            continue;
         nir_deref_instr *ptr = matrix_memory(b, intr, i);
         if (width > 1) {
            unsigned align = width * glsl_base_type_bit_size(desc.element_type) / 8;
            ptr = nir_build_deref_cast_with_alignment(b, &ptr->def, ptr->modes,
               glsl_vector_type(desc.element_type, width), 0, align, 0);
         }
         if (store) {
            nir_def *channels[4];
            for (unsigned j = 0; j < width; ++j)
               channels[j] = nir_channel(b, data, i + j * component_step);
            nir_store_deref(b, ptr, nir_vec(b, channels, width), ~0);
         } else {
            nir_def *loaded = nir_load_deref(b, ptr);
            for (unsigned j = 0; j < width; ++j)
               values[i + j * component_step] = nir_channel(b, loaded, j);
         }
      }
      if (!store)
         store_matrix(b, intr->src[0], nir_vec(b, values, count));
      break;
   }
   case nir_intrinsic_cmat_length: {
      struct glsl_cmat_description desc = nir_intrinsic_cmat_desc(intr);
      nir_def_replace(&intr->def, nir_imm_int(b, desc.rows * desc.cols / 32));
      return true;
   }
   case nir_intrinsic_cmat_muladd: {
      assert(!nir_intrinsic_saturate(intr));
      enum glsl_base_type input = matrix_desc(intr->src[1]).element_type;
      enum glsl_base_type input_b = matrix_desc(intr->src[2]).element_type;
      enum glsl_base_type accum = matrix_desc(intr->src[3]).element_type;
      enum glsl_base_type result = matrix_desc(intr->src[0]).element_type;
      const unsigned rows = matrix_desc(intr->src[1]).rows;
      /* 8x8 matmad has a single input-type field, so mixed A/B types
       * cannot be encoded. f16 and bf16 both widen to f32 exactly;
       * promote both operands to f32 and use that form. 16x16 gemm
       * keeps mixed f16/bf16 (it has separate type fields). */
      const bool mixed_8x8 = rows == 8 && input != input_b;
      assert((input == GLSL_TYPE_FLOAT16 || input == GLSL_TYPE_BFLOAT16 ||
              (input == GLSL_TYPE_FLOAT && rows == 8)) &&
             (input_b == GLSL_TYPE_FLOAT16 || input_b == GLSL_TYPE_BFLOAT16 ||
              (input_b == GLSL_TYPE_FLOAT && rows == 8)) &&
             (input_b == input || mixed_8x8 ||
              (rows == 16 &&
                ((input == GLSL_TYPE_FLOAT16 && input_b == GLSL_TYPE_BFLOAT16) ||
                 (input == GLSL_TYPE_BFLOAT16 && input_b == GLSL_TYPE_FLOAT16)))) &&
             (accum == GLSL_TYPE_FLOAT || accum == GLSL_TYPE_FLOAT16 ||
              accum == GLSL_TYPE_BFLOAT16) && result == accum);
      nir_def *a = load_matrix(b, intr->src[1]);
      nir_def *bb = load_matrix(b, intr->src[2]);
      if (mixed_8x8) {
         a = convert_matrix_value(b, a, input, GLSL_TYPE_FLOAT);
         bb = convert_matrix_value(b, bb, input_b, GLSL_TYPE_FLOAT);
         input = input_b = GLSL_TYPE_FLOAT;
      }
      /* The matmad accumulates in f32: a 16-bit accumulator is widened into
       * it and the result rounded back once per product. The rounding is
       * the accumulator's type, so keep it exact: fast-math folding of
       * f2f32(f2f16(x)) to x would carry f32 accumulators of twice the
       * registers from one product to the next. */
      b->fp_math_ctrl = nir_fp_exact;
      nir_def *c = convert_matrix_value(b, load_matrix(b, intr->src[3]),
                                         accum, GLSL_TYPE_FLOAT);
      nir_def *value = nir_cmat_muladd_agx(b, a, bb, c, .src_type =
         input == GLSL_TYPE_BFLOAT16 ? nir_type_uint16 :
         input == GLSL_TYPE_FLOAT ? nir_type_float32 : nir_type_float16,
         .src_type2 = input_b == GLSL_TYPE_BFLOAT16 ? nir_type_uint16 :
         input_b == GLSL_TYPE_FLOAT ? nir_type_float32 : nir_type_float16);
      store_matrix(b, intr->src[0], convert_matrix_value(b, value,
                                                      GLSL_TYPE_FLOAT, result));
      b->fp_math_ctrl = nir_fp_fast_math;
      break;
   }
   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op: {
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      nir_op op = nir_intrinsic_alu_op(intr);
      enum glsl_base_type type = matrix_desc(intr->src[1]).element_type;
      nir_def *a = load_matrix(b, intr->src[1]);
      if (type == GLSL_TYPE_BFLOAT16)
         a = nir_bf2f(b, a);
      nir_def *value;
      if (intr->intrinsic == nir_intrinsic_cmat_unary_op) {
         value = nir_build_alu1(b, op, a);
      } else {
         nir_def *bb = intr->intrinsic == nir_intrinsic_cmat_scalar_op
            ? intr->src[2].ssa : load_matrix(b, intr->src[2]);
         if (type == GLSL_TYPE_BFLOAT16)
            bb = nir_bf2f(b, bb);
         value = nir_build_alu2(b, op, a, bb);
      }
      if (type == GLSL_TYPE_BFLOAT16)
         value = nir_f2bf(b, value);
      store_matrix(b, intr->src[0], value);
      b->fp_math_ctrl = nir_fp_fast_math;
      break;
   }
   case nir_intrinsic_cmat_convert:
   case nir_intrinsic_cmat_transpose: {
      struct glsl_cmat_description src = matrix_desc(intr->src[1]);
      struct glsl_cmat_description dst = matrix_desc(intr->src[0]);
      struct cmat_layout from = *matrix_layout(src);
      const struct cmat_layout *to = matrix_layout(dst);
      bool transpose = intr->intrinsic == nir_intrinsic_cmat_transpose;
      if (transpose) {
         for (unsigned i = 0; i < from.element_bits; ++i)
            from.element[i] = TRANSPOSED(from.element[i]);
         for (unsigned j = 0; j < 5; ++j)
            from.lane[j] = TRANSPOSED(from.lane[j]);
      }
      bool move = memcmp(&from, to, sizeof(from)) != 0;
      /* Convert where the elements are narrower */
      bool convert_first = glsl_base_type_bit_size(dst.element_type) <
                           glsl_base_type_bit_size(src.element_type);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      nir_def *value = load_matrix(b, intr->src[1]);
      if (convert_first)
         value = convert_matrix_value(b, value, src.element_type, dst.element_type);
      if (move)
         value = permute_matrix(b, value, from, to);
      if (!convert_first)
         value = convert_matrix_value(b, value, src.element_type, dst.element_type);
      store_matrix(b, intr->src[0], value);
      b->fp_math_ctrl = nir_fp_fast_math;
      break;
   }
   case nir_intrinsic_cmat_bitcast:
      store_matrix(b, intr->src[0], load_matrix(b, intr->src[1]));
      break;
   case nir_intrinsic_cmat_extract:
      nir_def_replace(&intr->def, nir_vector_extract(b, load_matrix(b, intr->src[0]), intr->src[1].ssa));
      return true;
   case nir_intrinsic_cmat_insert:
      store_matrix(b, intr->src[0], nir_vector_insert(b, load_matrix(b, intr->src[2]),
                                                     intr->src[1].ssa, intr->src[3].ssa));
      break;
   case nir_intrinsic_cmat_copy:
      store_matrix(b, intr->src[0], load_matrix(b, intr->src[1]));
      break;
   default:
      return false;
   }
   nir_instr_remove(instr);
   return true;
}

bool
agx_nir_lower_cmat(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || !nir->info.cs.has_cooperative_matrix)
      return false;
   struct hash_table *types = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp)
      var->type = remap_matrix_type(types, var->type);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl)
         var->type = remap_matrix_type(types, var->type);
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block_reverse_safe(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block)
            lower_matrix_instr(&b, instr, types);
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   _mesa_hash_table_destroy(types, NULL);
   return true;
}

/* Calls `callee`, whose first parameter is where it returns a value of type
 * `type`, with the rest of `args`.
 */
static nir_def *
call_returning(nir_builder *b, nir_function *callee, const struct glsl_type *type,
               nir_def **args, unsigned count)
{
   nir_variable *ret = nir_local_variable_create(b->impl, type, "cmat_call_ret");
   nir_deref_instr *deref = nir_build_deref_var(b, ret);
   nir_call_instr *call = nir_call_instr_create(b->shader, callee);
   call->params[0] = nir_src_for_ssa(&deref->def);
   for (unsigned i = 0; i < count; ++i)
      call->params[i + 1] = nir_src_for_ssa(args[i]);
   nir_builder_instr_insert(b, &call->instr);
   return nir_load_deref(b, deref);
}

/* Store the elements in `values` as the matrix `dst` */
static void
build_matrix(nir_builder *b, nir_deref_instr *dst, nir_def **values, unsigned count)
{
   nir_cmat_construct(b, &dst->def, values[0]);
   for (unsigned e = 1; e < count; ++e)
      nir_cmat_insert(b, &dst->def, values[e], &dst->def, nir_imm_int(b, e));
}

/* coopMatPerElementNV: the function of each element, its row, its column and
 * the same element of any matrix among the extra arguments.
 */
static void
lower_per_element(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *dst = nir_src_as_deref(call->params[0]);
   nir_deref_instr *src = nir_src_as_deref(call->params[3]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(src->type);
   const struct cmat_layout *l = matrix_layout(desc);
   unsigned count = 1 << l->element_bits;
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *values[8];
   STACK_ARRAY(nir_def *, args, call->num_params - 1);

   for (unsigned e = 0; e < count; ++e) {
      args[0] = nir_iadd(b, layout_coord(b, l, lane, e, false), call->params[1].ssa);
      args[1] = nir_iadd(b, layout_coord(b, l, lane, e, true), call->params[2].ssa);
      args[2] = nir_cmat_extract(b, bits, &src->def, nir_imm_int(b, e));
      for (unsigned p = 4; p < call->num_params; ++p) {
         nir_deref_instr *arg = nir_src_as_deref(call->params[p]);
         if (arg && glsl_type_is_cmat(arg->type)) {
            struct glsl_cmat_description arg_desc = *glsl_get_cmat_description(arg->type);
            args[p - 1] = nir_cmat_extract(b, glsl_base_type_bit_size(arg_desc.element_type),
                                           &arg->def, nir_imm_int(b, e));
         } else {
            args[p - 1] = call->params[p].ssa;
         }
      }
      values[e] = call_returning(b, call->callee, glsl_scalar_type(desc.element_type),
                                 args, call->num_params - 1);
   }
   build_matrix(b, dst, values, count);
   STACK_ARRAY_FINISH(args);
}

/* Combine with `callee` the elements of the matrix `src` (layout `sl`) whose
 * coordinates differ only in the bits of `combined`: partly element bits,
 * combined within the lane, and partly lane bits, combined with the lane
 * across each in turn. Element e of v then holds the result for every
 * element of every lane that differs from it only in those bits, for each e
 * with none of the returned element bits set.
 */
static unsigned
reduce_elements(nir_builder *b, nir_function *callee, nir_deref_instr *src,
                const struct cmat_layout *sl, uint32_t combined, nir_def **v)
{
   struct glsl_cmat_description sd = *glsl_get_cmat_description(src->type);
   const struct glsl_type *type = glsl_scalar_type(sd.element_type);
   unsigned bits = glsl_base_type_bit_size(sd.element_type);
   unsigned count = 1 << sl->element_bits;

   for (unsigned e = 0; e < count; ++e)
      v[e] = nir_cmat_extract(b, bits, &src->def, nir_imm_int(b, e));

   /* Within the lane */
   unsigned done = 0;
   for (unsigned i = 0; i < sl->element_bits; ++i) {
      if (!(combined & BITFIELD_BIT(sl->element[i])))
         continue;
      for (unsigned e = 0; e < count; ++e) {
         if (!(e & (done | (1 << i)))) {
            nir_def *args[2] = {v[e], v[e | (1 << i)]};
            v[e] = call_returning(b, callee, type, args, 2);
         }
      }
      done |= 1 << i;
   }

   /* Across lanes */
   for (unsigned j = 0; j < 5; ++j) {
      if (!(combined & BITFIELD_BIT(sl->lane[j])))
         continue;
      nir_def *mine[8], *other[8];
      unsigned n = 0;
      for (unsigned e = 0; e < count; ++e) {
         if (!(e & done))
            mine[n++] = v[e];
      }
      shuffle_xor_values(b, mine, other, n, 1 << j);
      n = 0;
      for (unsigned e = 0; e < count; ++e) {
         if (!(e & done)) {
            nir_def *args[2] = {mine[n], other[n]};
            v[e] = call_returning(b, callee, type, args, 2);
            n++;
         }
      }
   }
   return done;
}

/* coopMatReduceNV. Each lane has the result of every row (column, 2x2
 * block) it holds elements of, and holds the same ones in the result matrix
 * (the result of a 2x2 reduction is half the size, its row k and column k
 * bits where the source's k + 1 bits are).
 */
static void
lower_reduce(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *dst = nir_src_as_deref(call->params[0]);
   nir_deref_instr *src = nir_src_as_deref(call->params[1]);
   struct glsl_cmat_description sd = *glsl_get_cmat_description(src->type);
   struct glsl_cmat_description dd = *glsl_get_cmat_description(dst->type);
   const struct cmat_layout *sl = matrix_layout(sd), *dl = matrix_layout(dd);
   nir_cmat_reduce reduce = nir_cmat_call_reduce_flags(call);

   /* The source coordinates combined, and the shift from a result
    * coordinate's bit to its source's.
    */
   uint32_t combined = 0;
   unsigned shift = 0;
   if (reduce & NIR_CMAT_REDUCE_2X2) {
      combined = BITFIELD_BIT(ROW(0)) | BITFIELD_BIT(COL(0));
      shift = 1;
   }
   if (reduce & NIR_CMAT_REDUCE_ROW)
      combined |= BITFIELD_RANGE(COL(0), util_logbase2(sd.cols));
   if (reduce & NIR_CMAT_REDUCE_COLUMN)
      combined |= BITFIELD_RANGE(ROW(0), util_logbase2(sd.rows));

   nir_def *v[8];
   unsigned done = reduce_elements(b, call->callee, src, sl, combined, v);

   unsigned dst_count = 1 << dl->element_bits;
   nir_def *values[8] = {NULL};
   for (unsigned e = 0; e < dst_count; ++e) {
      unsigned from = 0;
      for (unsigned i = 0; i < dl->element_bits; ++i) {
         unsigned coord = dl->element[i] + shift;
         if (!(e & (1 << i)) || (combined & BITFIELD_BIT(coord)))
            continue;
         int k = find_element_bit(sl, coord);
         if (k >= 0)
            from |= 1 << k;
      }
      values[e] = v[from & ~done];
   }
   build_matrix(b, dst, values, dst_count);
}

/* What nir_lower_cooperative_matrix_flexible_dimensions() makes of a
 * reduction of a matrix it splits into 16x16 pieces: each piece is reduced
 * alone, then a reduce_finish combines two pieces' results element by
 * element (same type: an element of each holds the reduction of its row or
 * column over its own piece).
 */
static void
lower_reduce_finish(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *dst = nir_src_as_deref(call->params[0]);
   nir_deref_instr *x = nir_src_as_deref(call->params[1]);
   nir_deref_instr *y = nir_src_as_deref(call->params[2]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(x->type);
   unsigned count = 1 << matrix_layout(desc)->element_bits;
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   nir_def *values[8];

   for (unsigned e = 0; e < count; ++e) {
      nir_def *args[2] = {
         nir_cmat_extract(b, bits, &x->def, nir_imm_int(b, e)),
         nir_cmat_extract(b, bits, &y->def, nir_imm_int(b, e)),
      };
      values[e] = call_returning(b, call->callee, glsl_scalar_type(desc.element_type),
                                 args, 2);
   }
   build_matrix(b, dst, values, count);
}

/* The split's 2x2 reduction: four 16x16 pieces, top left, top right, bottom
 * left, bottom right, into one 16x16 result. Each piece reduces 2x2 into an
 * 8x8 quarter of the result held as lower_reduce() holds it (its lane bits
 * those of the piece, one coordinate down; one element bit left); the four
 * quarters' elements together are a matrix whose other two element bits are
 * the result's row 3 and column 3, moved to the result's layout.
 */
static void
lower_reduce_2x2(nir_builder *b, nir_cmat_call_instr *call)
{
   nir_deref_instr *dst = nir_src_as_deref(call->params[0]);
   const struct cmat_layout *sl =
      matrix_layout(*glsl_get_cmat_description(nir_src_as_deref(call->params[1])->type));
   const struct cmat_layout *dl =
      matrix_layout(*glsl_get_cmat_description(dst->type));
   uint32_t combined = BITFIELD_BIT(ROW(0)) | BITFIELD_BIT(COL(0));
   struct cmat_layout quarters = {.element_bits = 3};
   nir_def *w[8];

   for (unsigned q = 0; q < 4; ++q) {
      nir_def *v[8];
      unsigned done = reduce_elements(b, call->callee,
                                      nir_src_as_deref(call->params[1 + q]),
                                      sl, combined, v);
      unsigned k = ffs(~done & BITFIELD_MASK(sl->element_bits)) - 1;
      w[2 * q] = v[0];
      w[2 * q + 1] = v[1 << k];
      quarters.element[0] = sl->element[k] - 1;
   }
   quarters.element[1] = COL(3);
   quarters.element[2] = ROW(3);
   for (unsigned j = 0; j < 5; ++j)
      quarters.lane[j] = sl->lane[j] - 1;

   nir_def *value = permute_matrix(b, nir_vec(b, w, 8), quarters, dl);
   nir_def *values[8];
   for (unsigned e = 0; e < 8; ++e)
      values[e] = nir_channel(b, value, e);
   build_matrix(b, dst, values, 8);
}

static bool
type_is_supported(const struct glsl_type *type)
{
   if (glsl_type_is_cmat(type)) {
      const struct glsl_cmat_description *desc = glsl_get_cmat_description(type);
      return (desc->rows == 8 || desc->rows == 16) && desc->cols == desc->rows &&
             desc->scope == SCOPE_SUBGROUP;
   } else if (glsl_type_is_array(type)) {
      return type_is_supported(glsl_get_array_element(type));
   } else if (glsl_type_is_struct(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         if (!type_is_supported(glsl_get_struct_field(type, i)))
            return false;
      }
   }
   return true;
}

static bool
matrices_supported(nir_cmat_call_instr *call)
{
   for (unsigned i = 0; i < call->num_params; ++i) {
      nir_deref_instr *deref = nir_src_as_deref(call->params[i]);
      if (deref && !type_is_supported(deref->type))
         return false;
   }
   return true;
}

/* Whether the lowering can take every cooperative matrix of the shader: only
 * subgroup-scope 8x8 and 16x16 ones, and no matrix call left
 * (agx_nir_lower_cmat_calls() leaves those it cannot lower, and none of the
 * others is lowered at all).
 */
bool
agx_nir_cmat_supported(nir_shader *nir)
{
   if (!nir->info.cs.has_cooperative_matrix)
      return true;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref &&
                !type_is_supported(nir_instr_as_deref(instr)->type))
               return false;
            if (instr->type == nir_instr_type_cmat_call)
               return false;
         }
      }
   }
   return true;
}

/* coopMatPerElementNV passes element e of each matrix argument with element
 * e of the source, so every matrix of the call must lie over the lanes as the
 * source does.
 */
static bool
per_element_layouts_match(nir_cmat_call_instr *call)
{
   const struct cmat_layout *l =
      matrix_layout(*glsl_get_cmat_description(nir_src_as_deref(call->params[3])->type));
   for (unsigned p = 0; p < call->num_params; ++p) {
      nir_deref_instr *deref = nir_src_as_deref(call->params[p]);
      if (deref && glsl_type_is_cmat(deref->type) &&
          matrix_layout(*glsl_get_cmat_description(deref->type)) != l)
         return false;
   }
   return true;
}

static bool
reduce_layouts_match(nir_cmat_call_instr *call)
{
   nir_deref_instr *dst = nir_src_as_deref(call->params[0]);
   nir_deref_instr *src = nir_src_as_deref(call->params[1]);
   const struct cmat_layout *sl = matrix_layout(*glsl_get_cmat_description(src->type));
   const struct cmat_layout *dl = matrix_layout(*glsl_get_cmat_description(dst->type));
   nir_cmat_reduce reduce = nir_cmat_call_reduce_flags(call);
   if ((reduce & NIR_CMAT_REDUCE_ROW) && (reduce & NIR_CMAT_REDUCE_COLUMN))
      return true;

   /* What the result needs is where it goes: each lane bit of the result
    * matrix holds the coordinate it holds in the source, after reduction.
    */
   unsigned shift = (reduce & NIR_CMAT_REDUCE_2X2) ? 1 : 0;
   for (unsigned j = 0; j < 5; ++j) {
      bool src_kept = (reduce & NIR_CMAT_REDUCE_2X2)
         ? COORD_BIT(sl->lane[j]) != 0
         : (reduce & NIR_CMAT_REDUCE_ROW) ? sl->lane[j] < COL(0)
                                          : sl->lane[j] >= COL(0);
      bool dst_kept = (reduce & NIR_CMAT_REDUCE_2X2) ||
         ((reduce & NIR_CMAT_REDUCE_ROW) ? dl->lane[j] < COL(0)
                                         : dl->lane[j] >= COL(0));
      if (src_kept != dst_kept || (src_kept && dl->lane[j] + shift != sl->lane[j]))
         return false;
   }
   return true;
}

/* reduce_finish combines the same element of two pieces of one type */
static bool
reduce_finish_layouts_match(nir_cmat_call_instr *call)
{
   const struct glsl_type *type = nir_src_as_deref(call->params[1])->type;
   return nir_src_as_deref(call->params[0])->type == type &&
          nir_src_as_deref(call->params[2])->type == type;
}

/* reduce_2x2 takes four 16x16 pieces of one layout whose 2x2 reduction
 * lower_reduce_2x2() can place: none of their lane bits holds row 0 or
 * column 0, and one element bit is left after them.
 */
static bool
reduce_2x2_layouts_match(nir_cmat_call_instr *call)
{
   const struct glsl_type *type = nir_src_as_deref(call->params[1])->type;
   struct glsl_cmat_description sd = *glsl_get_cmat_description(type);
   struct glsl_cmat_description dd =
      *glsl_get_cmat_description(nir_src_as_deref(call->params[0])->type);
   if (sd.rows != 16 || dd.rows != 16)
      return false;
   for (unsigned q = 2; q < 5; ++q) {
      if (nir_src_as_deref(call->params[q])->type != type)
         return false;
   }
   const struct cmat_layout *sl = matrix_layout(sd);
   unsigned left = 0;
   for (unsigned i = 0; i < sl->element_bits; ++i)
      left += COORD_BIT(sl->element[i]) != 0;
   for (unsigned j = 0; j < 5; ++j) {
      if (COORD_BIT(sl->lane[j]) == 0)
         return false;
   }
   return left == 1;
}

/* Lower coopMatPerElementNV and coopMatReduceNV (and the partial reductions
 * of split matrices) to calls of their functions on the elements, and inline
 * those. This runs before anything else lowers the
 * shader, while the functions are still separate.
 */
bool
agx_nir_lower_cmat_calls(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || !nir->info.cs.has_cooperative_matrix)
      return false;

   bool progress = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_cmat_call)
            continue;
         nir_cmat_call_instr *call = nir_instr_as_cmat_call(instr);
         b.cursor = nir_before_instr(instr);
         if (!matrices_supported(call)) {
            continue;
         } else if (call->op == nir_cmat_call_op_per_element_op &&
                    per_element_layouts_match(call)) {
            lower_per_element(&b, call);
         } else if (call->op == nir_cmat_call_op_reduce && reduce_layouts_match(call)) {
            lower_reduce(&b, call);
         } else if (call->op == nir_cmat_call_op_reduce_finish &&
                    reduce_finish_layouts_match(call)) {
            lower_reduce_finish(&b, call);
         } else if (call->op == nir_cmat_call_op_reduce_2x2 &&
                    reduce_2x2_layouts_match(call)) {
            lower_reduce_2x2(&b, call);
         } else {
            continue;
         }
         nir_instr_remove(instr);
         progress = true;
      }
   }
   if (!progress)
      return nir_no_progress(impl);

   nir_progress(true, impl, nir_metadata_none);
   nir_inline_functions(nir);

   /* Keep only the functions other matrix calls still name */
   struct set *live = _mesa_pointer_set_create(NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_cmat_call &&
             nir_instr_as_cmat_call(instr)->callee)
            _mesa_set_add(live, nir_instr_as_cmat_call(instr)->callee);
      }
   }
   nir_foreach_function_safe(func, nir) {
      if (!func->is_entrypoint && !_mesa_set_search(live, func))
         exec_node_remove(&func->node);
   }
   _mesa_set_destroy(live, NULL);
   return true;
}

/*
 * Tensor loads and stores of VK_NV_cooperative_matrix2 (cmat_call
 * tensor_load/tensor_store), lowered to memory accesses and cmat_insert or
 * cmat_extract while the matrices are still cooperative matrix types, so that
 * a decode function's calls can be inlined before the shader's other lowering.
 *
 * The common case, a 2D tensor layout with element strides (1 along the inner
 * dimension), unit blocks and either no view or a view that only transposes,
 * reads a lane's elements as runs along the tensor's inner dimension: a lane
 * holds four consecutive columns of each of two rows of an accumulator or B
 * operand (two loads of four), every other one of seven columns of each of
 * two rows of an A operand (two loads of three per row), or a pair of
 * consecutive rows of each of four columns of a transposed operand (four
 * loads of two).
 * A lane whose elements reach past the tensor's edge reads each of them at
 * its clamped coordinates, so no address outside the tensor is read; with the
 * clamp mode undefined there is no bounds test. An element index is 32 bits
 * wide and wraps, which a run's does not: a lane with a run across the wrap
 * reads each element at its own index too. With a constant clamp, a lane
 * whose rows are all past the tensor reads the zero page instead (global
 * memory). Everything else goes through the common per-element addressing
 * (nir_calc_tensor_derefs).
 *
 * K paired 64 deep. A product's sum over K may be taken in any order, so the
 * 16-deep pieces of an A (M x K) or B (K x N) matrix of flexible dimensions
 * whose K is a multiple of 64 need not each hold 16 consecutive K: piece
 * 4 g + s holds, for its column (A) or row (B) c,
 *
 *    k = 64 g + 8 (c >> 1) + 2 s + (c & 1),
 *
 * the same for A and B, so their products pair as before. A lane's elements
 * of the four pieces of a 64-deep group then lie in whole 8-element vectors
 * of K: a transposed B operand (B^T row-major, K contiguous) is one 16-byte
 * load per column for four multiplies where K order needs four 4-byte loads
 * for each, an A operand one 16-byte load per row for each multiply, all of
 * it used, and a decoded block of eight codes along K is one block per lane
 * and column for four multiplies. The pieces' tensor loads are lowered
 * together. The pairing is only taken where nothing can tell it from K order
 * (agx_nir_lower_cmat_flexible): no coopMatLoad or coopMatStore of such a
 * matrix, no operation that gives an element's place.
 */

/* The offset of element e of a lane from its element 0, in matrix rows and
 * columns (matrix_row, matrix_col).
 */
static void
element_offset(struct glsl_cmat_description desc, unsigned e,
               unsigned *dr, unsigned *dc)
{
   if (desc.rows == 16) {
      *dr = e >> 2;
      *dc = (desc.use == GLSL_CMAT_USE_A ? 2 : 1) * (e & 3);
   } else {
      *dr = 0;
      *dc = e;
   }
}

/* The K of a piece's column (A) or row (B) c, less 64 g + 2 s (above). */
static nir_def *
paired_k(nir_builder *b, nir_def *c)
{
   return nir_ior(b, nir_ishl_imm(b, nir_ushr_imm(b, c, 1), 3),
                  nir_iand_imm(b, c, 1));
}

/* Whether a piece of a matrix of K paired 64 deep, and its group and step. */
static bool
piece_paired(nir_cmat_call_instr *call, struct glsl_cmat_description piece,
             bool pairing, unsigned *g, unsigned *s)
{
   struct glsl_cmat_description whole = nir_cmat_call_cmat_desc(call);
   if (!pairing || piece.rows != 16 || piece.cols != 16 ||
       !((whole.use == GLSL_CMAT_USE_A && whole.cols % 64 == 0) ||
         (whole.use == GLSL_CMAT_USE_B && whole.rows % 64 == 0)))
      return false;
   struct nir_cmat_tensor_load tl = nir_cmat_call_tensor_load_info(call);
   unsigned t = whole.use == GLSL_CMAT_USE_A ? tl.split_col_index
                                             : tl.split_row_index;
   *g = t / 4;
   *s = t % 4;
   return true;
}

/* A clamp mode applied to tensor coordinate c of a dimension of size dim.
 * A coordinate is signed and a dimension unsigned, so the dimension may be
 * longer than any coordinate, and the repeating modes divide the
 * coordinate's magnitude (2^31 for INT32_MIN).
 */
static nir_def *
clamp_coord(nir_builder *b, nir_tensor_clamp_mode mode, nir_def *c,
            nir_def *dim)
{
   nir_def *negative = nir_ilt_imm(b, c, 0);
   nir_def *magnitude = nir_bcsel(b, negative, nir_ineg(b, c), c);

   switch (mode) {
   case NIR_TENSOR_CLAMP_EDGE:
      return nir_umin(b, nir_imax_imm(b, c, 0), nir_iadd_imm(b, dim, -1));
   case NIR_TENSOR_CLAMP_REPEAT: {
      /* The remainder of the floored division: dim - rem below zero. */
      nir_def *rem = nir_umod(b, magnitude, dim);
      nir_def *below = nir_iand(b, negative, nir_ine_imm(b, rem, 0));
      return nir_bcsel(b, below, nir_isub(b, dim, rem), rem);
   }
   case NIR_TENSOR_CLAMP_REPEAT_MIRRORED: {
      /* The period, 2 dim - 2, may not fit in 32 bits. The reflection of c
       * is that of -c, and |c| lies in pass |c| / (dim - 1) over the
       * dimension: forwards in the even passes, backwards in the odd ones.
       * A dimension of one repeats its only element.
       */
      nir_def *last = nir_iadd_imm(b, dim, -1);
      nir_def *length = nir_umax_imm(b, last, 1);
      nir_def *pass = nir_udiv(b, magnitude, length);
      nir_def *rem = nir_isub(b, magnitude, nir_imul(b, pass, length));
      return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, pass, 1), 0),
                       nir_isub(b, last, rem), rem);
   }
   default:
      /* Constant: any address inside the tensor; the value is replaced. */
      return nir_imm_int(b, 0);
   }
}

/* Whether coordinate c is inside a dimension of size dim: not negative, and
 * then less than the dimension, which is unsigned.
 */
static nir_def *
coord_inside(nir_builder *b, nir_def *c, nir_def *dim)
{
   return nir_iand(b, nir_ige_imm(b, c, 0), nir_ult(b, c, dim));
}

/* Whether the coordinates c + lo to c + hi (lo <= hi) are all inside a
 * dimension of size dim. The sums wrap, so c is first held to where they do
 * not: a coordinate past INT32_MAX is outside the tensor.
 */
static nir_def *
coords_inside(nir_builder *b, nir_def *c, unsigned lo, unsigned hi,
              nir_def *dim)
{
   nir_def *no_wrap = nir_ult_imm(b, nir_iadd_imm(b, c, lo),
                                  (uint64_t)INT32_MAX - hi + lo + 1);
   return nir_iand(b, no_wrap, nir_ult(b, nir_iadd_imm(b, c, hi), dim));
}

/* The deref of `count` elements at element index idx of the tensor. A byte
 * offset in memory other than global is 32 bits wide: where the elements'
 * end in the binding (with the offset of the tensor's pointer in it) passes
 * 2^32, idx is replaced by one past any binding (nir_cap_tensor_index), so
 * that they do not wrap into the binding.
 */
static nir_deref_instr *
tensor_elements(nir_builder *b, nir_deref_instr *ptr,
                struct glsl_cmat_description desc, nir_def *idx,
                unsigned count)
{
   unsigned bytes = glsl_base_type_bit_size(desc.element_type) / 8;
   if (!nir_deref_mode_is(ptr, nir_var_mem_global))
      idx = nir_cap_tensor_index(b, ptr, idx, bytes, count);
   nir_deref_instr *d =
      nir_build_deref_cast(b, &ptr->def, ptr->modes,
                           glsl_scalar_type(desc.element_type), bytes);
   d = nir_build_deref_ptr_as_array(b, d, nir_u2uN(b, idx, ptr->def.bit_size));
   if (count > 1) {
      d = nir_build_deref_cast_with_alignment(
         b, &d->def, d->modes, glsl_vector_type(desc.element_type, count), 0,
         bytes, 0);
   }
   return d;
}

/* The address of a deref of global memory. */
static nir_def *
global_address(nir_builder *b, nir_deref_instr *deref)
{
   if (deref->deref_type == nir_deref_type_cast &&
       !nir_src_is_deref(deref->parent))
      return deref->parent.ssa;
   return nir_explicit_io_address_from_deref(
      b, deref, global_address(b, nir_deref_instr_parent(deref)),
      nir_address_format_64bit_global);
}

/* Where an element at clamped coordinates must not be read, given whether
 * its own coordinates are inside the tensor: with a constant clamp, outside
 * it; with the other clamp modes, in a tensor with a dimension of zero,
 * which has no element to clamp to. NULL with the clamp mode undefined.
 */
static nir_def *
clamp_skip(nir_builder *b, struct nir_calc_tensor_info *info, nir_def *in,
           nir_def *dim0, nir_def *dim1)
{
   switch (info->clamp_mode) {
   case NIR_TENSOR_CLAMP_UNDEFINED:
      return NULL;
   case NIR_TENSOR_CLAMP_CONSTANT:
      return nir_inot(b, in);
   default:
      return nir_ior(b, nir_ieq_imm(b, dim0, 0), nir_ieq_imm(b, dim1, 0));
   }
}

/* A load's value where it reads nothing: the clamp value, or zero. */
static nir_def *
clamp_skip_value(nir_builder *b, struct nir_calc_tensor_info *info,
                 unsigned bits)
{
   return info->clamp_mode == NIR_TENSOR_CLAMP_CONSTANT
             ? nir_u2uN(b, info->clamp_value, bits)
             : nir_imm_intN_t(b, 0, bits);
}

/* The element at d, or `skipped` without a read where skip is true: global
 * memory reads the zero page instead, other memory reads under a branch.
 */
static nir_def *
load_unless(nir_builder *b, nir_deref_instr *d, nir_def *skip,
            nir_def *skipped)
{
   if (!skip)
      return nir_load_deref(b, d);
   if (nir_deref_mode_is(d, nir_var_mem_global)) {
      unsigned bytes = skipped->bit_size / 8;
      nir_def *addr = nir_bcsel(b, skip, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS),
                                global_address(b, d));
      d = nir_build_deref_cast_with_alignment(b, addr, nir_var_mem_global,
                                              d->type, 0, bytes, 0);
      return nir_bcsel(b, skip, skipped, nir_load_deref(b, d));
   }
   nir_if *nif = nir_push_if(b, nir_inot(b, skip));
   nir_def *v = nir_load_deref(b, d);
   nir_pop_if(b, nif);
   return nir_if_phi(b, v, skipped);
}

struct tensor_fast {
   bool transposed;
   unsigned row_offset, col_offset;
   nir_def *stride, *dim0, *dim1;
};

/* Whether the common case applies, as a condition that folds when the
 * layout's spans, strides and block sizes are constants; NULL if it cannot.
 */
static nir_def *
tensor_fast_condition(nir_builder *b, struct nir_calc_tensor_info *info,
                      struct glsl_cmat_description desc,
                      struct tensor_fast *fast)
{
   if (info->spans->num_components != 2)
      return NULL;

   fast->transposed = false;
   fast->row_offset = info->row_imm_offset;
   fast->col_offset = info->col_imm_offset;
   nir_def *cond = nir_imm_true(b);
   if (info->view) {
      if (info->view_has_dims)
         return NULL;
      if (info->view_permutations[0] == 1 && info->view_permutations[1] == 0)
         fast->transposed = true;
      else if (info->view_permutations[0] != 0 || info->view_permutations[1] != 1)
         return NULL;
      cond = nir_iand(b, cond, nir_ieq_imm(b, info->clip_row_offset, 0));
      cond = nir_iand(b, cond, nir_ieq_imm(b, info->clip_col_offset, 0));
      cond = nir_iand(b, cond, nir_uge_imm(b, info->clip_row_span, desc.rows));
      cond = nir_iand(b, cond, nir_uge_imm(b, info->clip_col_span, desc.cols));
   }

   /* A row of the matrix lies along the span's inner dimension (its outer
    * one, transposed), which must be exactly a row long; the other must hold
    * all of the matrix's rows.
    */
   unsigned inner = fast->transposed ? 0 : 1;
   cond = nir_iand(b, cond, nir_ieq_imm(b, nir_channel(b, info->spans, inner),
                                        desc.cols));
   cond = nir_iand(b, cond, nir_uge_imm(b, nir_channel(b, info->spans, 1 - inner),
                                        desc.rows));
   /* A transposed element's index in the span, its column times the span's
    * inner dimension plus its row, is 32 bits wide: it must not wrap.
    */
   if (fast->transposed && desc.cols > 1) {
      uint32_t longest = (UINT32_MAX - (desc.rows - 1)) / (desc.cols - 1);
      cond = nir_iand(b, cond, nir_ult_imm(b, nir_channel(b, info->spans, 1),
                                           (uint64_t)longest + 1));
   }
   /* Runs of elements need unit blocks and strides; decoded elements are
    * addressed one by one, through their blocks.
    */
   if (!info->decode_fnptr) {
      cond = nir_iand(b, cond, nir_ieq_imm(b, nir_channel(b, info->block_sizes, 0), 1));
      cond = nir_iand(b, cond, nir_ieq_imm(b, nir_channel(b, info->block_sizes, 1), 1));
      cond = nir_iand(b, cond, nir_ieq_imm(b, nir_channel(b, info->strides, 1), 1));
   }
   fast->stride = nir_channel(b, info->strides, 0);
   if (info->layout_dims) {
      fast->dim0 = nir_channel(b, info->layout_dims, 0);
      fast->dim1 = nir_channel(b, info->layout_dims, 1);
   }
   return cond;
}

/* A lane's elements of one load (of the four pieces of a group, with K
 * paired), in tensor coordinates: element i at (y0 + dy[i], x0 + dx[i]), x
 * along the tensor's inner dimension. Element i is element i % per_piece of
 * piece i / per_piece.
 */
#define TENSOR_MAX_ELEMENTS 32

struct tensor_items {
   unsigned count, per_piece;
   unsigned dy[TENSOR_MAX_ELEMENTS], dx[TENSOR_MAX_ELEMENTS];
   unsigned min_dx, max_dx, max_dy;
   nir_def *y0, *x0;
   /* An A operand's lane holds every other column from an odd or an even
    * one: 1 if x0 is the odd one, for loads that begin at the even one.
    */
   nir_def *parity;
};

static void
tensor_items(nir_builder *b, struct nir_calc_tensor_info *info,
             struct tensor_fast *fast, struct glsl_cmat_description desc,
             unsigned pieces, unsigned group, struct tensor_items *it)
{
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *r0 = matrix_row(b, lane, desc, 0), *c0 = matrix_col(b, lane, desc, 0);
   bool a = desc.use == GLSL_CMAT_USE_A;
   nir_def *row, *col;
   if (pieces > 1 && a) {
      row = nir_iadd_imm(b, r0, fast->row_offset);
      col = nir_iadd_imm(b, paired_k(b, c0), 64 * group);
   } else if (pieces > 1) {
      row = nir_iadd_imm(b, paired_k(b, r0), 64 * group);
      col = nir_iadd_imm(b, c0, fast->col_offset);
   } else {
      row = nir_iadd_imm(b, r0, fast->row_offset);
      col = nir_iadd_imm(b, c0, fast->col_offset);
   }

   it->per_piece = desc.rows * desc.cols / 32;
   it->count = pieces * it->per_piece;
   it->min_dx = ~0u;
   it->max_dx = it->max_dy = 0;
   for (unsigned p = 0; p < pieces; ++p) {
      for (unsigned e = 0; e < it->per_piece; ++e) {
         unsigned dr, dc, i = p * it->per_piece + e;
         element_offset(desc, e, &dr, &dc);
         /* A's columns c0 + dc, dc even, and B's rows r0 + dr, r0 even. */
         if (pieces > 1 && a)
            dc = 2 * p + 4 * dc;
         else if (pieces > 1)
            dr = 2 * p + dr;
         it->dy[i] = fast->transposed ? dc : dr;
         it->dx[i] = fast->transposed ? dr : dc;
         it->min_dx = MIN2(it->min_dx, it->dx[i]);
         it->max_dx = MAX2(it->max_dx, it->dx[i]);
         it->max_dy = MAX2(it->max_dy, it->dy[i]);
      }
   }
   it->y0 = nir_iadd(b, nir_channel(b, info->offsets, 0),
                     fast->transposed ? col : row);
   it->x0 = nir_iadd(b, nir_channel(b, info->offsets, 1),
                     fast->transposed ? row : col);
   it->parity = a && !fast->transposed && desc.rows == 16
                   ? nir_iand_imm(b, c0, 1) : NULL;
}

/* A run: the lane's unread elements of one tensor row within four columns of
 * the lowest of them, which need not include `first` (with K paired a row's
 * elements interleave the pieces'): callers take runs until `first` is read.
 */
static unsigned
next_run(struct tensor_items *it, unsigned first, uint32_t *done,
         unsigned *members, unsigned *places, unsigned *start, unsigned *width)
{
   *start = ~0u;
   for (unsigned i = first; i < it->count; ++i) {
      if (it->dy[i] == it->dy[first] && !(*done & BITFIELD_BIT(i)))
         *start = MIN2(*start, it->dx[i]);
   }
   unsigned n = 0;
   *width = 0;
   for (unsigned i = first; i < it->count; ++i) {
      if (it->dy[i] == it->dy[first] && !(*done & BITFIELD_BIT(i)) &&
          it->dx[i] < *start + 4) {
         members[n] = i;
         places[n++] = it->dx[i] - *start;
         *width = MAX2(*width, it->dx[i] - *start + 1);
         *done |= BITFIELD_BIT(i);
      }
   }
   return n;
}

static unsigned
count_runs(struct tensor_items *it)
{
   uint32_t done = 0;
   unsigned runs = 0, members[4], places[4], start, width;
   for (unsigned i = 0; i < it->count;) {
      if (done & BITFIELD_BIT(i)) {
         ++i;
         continue;
      }
      next_run(it, i, &done, members, places, &start, &width);
      runs++;
   }
   return runs;
}

/* The 16-byte vectors (eight 16-bit elements) holding the lane's elements,
 * from the even column at or before x0, or 0 if one of them cannot be read
 * so: a lane with odd columns takes the high halves of the vectors' words.
 * Returns their count and the highest vector.
 */
static unsigned
count_vectors(struct tensor_items *it, unsigned *last)
{
   uint32_t seen[TENSOR_MAX_ELEMENTS] = {0};
   unsigned vectors = 0;
   *last = 0;
   for (unsigned i = 0; i < it->count; ++i) {
      if (it->parity && (it->dx[i] & 1))
         return 0;
      unsigned v = it->dx[i] / 8;
      if (it->dy[i] >= TENSOR_MAX_ELEMENTS || v >= 32)
         return 0;
      if (!(seen[it->dy[i]] & BITFIELD_BIT(v))) {
         seen[it->dy[i]] |= BITFIELD_BIT(v);
         vectors++;
      }
      *last = MAX2(*last, v);
   }
   return vectors;
}

/* Whether the element indices of the lane's elements of each tensor row
 * count up from the row's first without wrapping, so that they can be read
 * or written as runs (as the 16-byte vectors of count_vectors, with
 * `vectors`): an index is 32 bits wide, and each element's wraps alone. One
 * comparison, of the highest of the rows' first indices with the last index
 * that leaves room for the longest row.
 */
static nir_def *
rows_unwrapped(nir_builder *b, struct tensor_items *it, nir_def *stride,
               bool vectors)
{
   nir_def *x = vectors && it->parity ? nir_isub(b, it->x0, it->parity) : it->x0;
   nir_def *highest = NULL;
   unsigned longest = 0;
   uint32_t done = 0;
   for (unsigned first = 0; first < it->count; ++first) {
      if (done & BITFIELD_BIT(first))
         continue;
      unsigned lo = ~0u, hi = 0;
      for (unsigned i = first; i < it->count; ++i) {
         if (it->dy[i] != it->dy[first])
            continue;
         done |= BITFIELD_BIT(i);
         lo = MIN2(lo, vectors ? it->dx[i] & ~7u : it->dx[i]);
         hi = MAX2(hi, vectors ? it->dx[i] | 7u : it->dx[i]);
      }
      if (hi == lo)
         continue;
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[first]);
      nir_def *idx = nir_iadd(b, nir_imul(b, y, stride), nir_iadd_imm(b, x, lo));
      highest = highest ? nir_umax(b, highest, idx) : idx;
      longest = MAX2(longest, hi - lo);
   }
   if (!highest)
      return nir_imm_true(b);
   return nir_ult_imm(b, highest, (uint64_t)UINT32_MAX - longest + 1);
}

/* The lane's elements one by one, each at its own index. */
static void
load_each(nir_builder *b, struct tensor_items *it, nir_def *stride,
          nir_deref_instr *ptr, struct glsl_cmat_description desc,
          nir_def **values)
{
   for (unsigned i = 0; i < it->count; ++i) {
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[i]);
      nir_def *x = nir_iadd_imm(b, it->x0, it->dx[i]);
      nir_def *idx = nir_iadd(b, nir_imul(b, y, stride), x);
      values[i] = nir_load_deref(b, tensor_elements(b, ptr, desc, idx, 1));
   }
}

/* The lane's elements in runs of up to four along the tensor's rows. */
static void
load_runs(nir_builder *b, struct tensor_items *it, nir_def *stride,
          nir_deref_instr *ptr, struct glsl_cmat_description desc,
          nir_def **values)
{
   uint32_t done = 0;
   unsigned members[4], places[4], start, width;
   for (unsigned first = 0; first < it->count;) {
      if (done & BITFIELD_BIT(first)) {
         ++first;
         continue;
      }
      unsigned n = next_run(it, first, &done, members, places, &start, &width);
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[first]);
      nir_def *idx = nir_iadd(b, nir_imul(b, y, stride),
                              nir_iadd_imm(b, it->x0, start));
      nir_def *v = nir_load_deref(b, tensor_elements(b, ptr, desc, idx, width));
      for (unsigned i = 0; i < n; ++i)
         values[members[i]] = nir_channel(b, v, places[i]);
   }
}

/* The lane's 16-bit elements by 16-byte loads (count_vectors), the tensor's
 * rows 4-byte aligned. All loads are issued before any element is taken
 * from them: a thread waits at the first use of a loaded value.
 */
static void
load_vectors(nir_builder *b, struct tensor_items *it, nir_def *stride,
             nir_deref_instr *ptr, nir_def **values)
{
   nir_def *x = it->parity ? nir_isub(b, it->x0, it->parity) : it->x0;
   nir_def *shift = it->parity ? nir_ishl_imm(b, it->parity, 4) : NULL;
   nir_def *words[TENSOR_MAX_ELEMENTS];
   unsigned vector_of[TENSOR_MAX_ELEMENTS], vectors = 0;
   uint32_t done = 0;
   for (unsigned first = 0; first < it->count; ++first) {
      if (done & BITFIELD_BIT(first))
         continue;
      unsigned v = it->dx[first] / 8;
      /* Indexed by words, as the load scales its index. */
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[first]);
      nir_def *idx = nir_iadd(b, nir_imul(b, y, stride), nir_iadd_imm(b, x, 8 * v));
      nir_deref_instr *d =
         nir_build_deref_cast(b, &ptr->def, ptr->modes, glsl_uint_type(), 4);
      d = nir_build_deref_ptr_as_array(
         b, d, nir_u2uN(b, nir_ushr_imm(b, idx, 1), ptr->def.bit_size));
      d = nir_build_deref_cast_with_alignment(b, &d->def, d->modes,
                                              glsl_vector_type(GLSL_TYPE_UINT, 4),
                                              0, 4, 0);
      words[vectors] = nir_load_deref(b, d);
      for (unsigned i = first; i < it->count; ++i) {
         if (it->dy[i] == it->dy[first] && it->dx[i] / 8 == v) {
            done |= BITFIELD_BIT(i);
            vector_of[i] = vectors;
         }
      }
      vectors++;
   }
   /* Vector by vector, so that each is dead once taken apart. */
   for (unsigned v = 0; v < vectors; ++v) {
      for (unsigned i = 0; i < it->count; ++i) {
         if (vector_of[i] != v)
            continue;
         nir_def *word = nir_channel(b, words[v], (it->dx[i] % 8) / 2);
         if (shift)
            values[i] = nir_unpack_32_2x16_split_x(b, nir_ushr(b, word, shift));
         else if (it->dx[i] & 1)
            values[i] = nir_unpack_32_2x16_split_y(b, word);
         else
            values[i] = nir_unpack_32_2x16_split_x(b, word);
      }
   }
}

/* The common case's loads: a lane inside the tensor reads its runs, a lane
 * reaching past the tensor's edge each element at its clamped coordinates.
 * One test covers all of a lane's elements, so that its loads issue
 * together. Where 16-byte vectors take fewer loads, a lane reads them if the
 * tensor's rows are 4-byte aligned (the same test), its runs otherwise. A
 * lane whose indices wrap within a row (the same test again) reads each
 * element at its own index.
 */
static void
tensor_fast_load(nir_builder *b, struct nir_calc_tensor_info *info,
                 struct tensor_fast *fast, nir_deref_instr *ptr,
                 struct glsl_cmat_description desc, struct tensor_items *it,
                 nir_def **values)
{
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   bool checked = info->clamp_mode != NIR_TENSOR_CLAMP_UNDEFINED;

   unsigned last_vector = 0;
   unsigned vectors = bits == 16 && nir_deref_mode_is(ptr, nir_var_mem_global)
                         ? count_vectors(it, &last_vector) : 0;
   /* Only a group's vectors are all used: a lane of an A operand in K order
    * uses half of each, and two loads of three elements are faster.
    */
   bool wide = it->count > it->per_piece && vectors && vectors < count_runs(it);

   nir_def *fast_ok = nir_imm_true(b);
   if (checked) {
      fast_ok = nir_iand(b, coords_inside(b, it->y0, 0, it->max_dy, fast->dim0),
                         coords_inside(b, it->x0, it->min_dx, it->max_dx, fast->dim1));
   }
   if (wide) {
      /* The vectors begin at the even column at or before x0, and end
       * within the tensor where it has bounds (with none, within the slice
       * the load reads, by the vectors' place in it).
       */
      nir_def *x = it->parity ? nir_isub(b, it->x0, it->parity) : it->x0;
      nir_def *addr = global_address(b, ptr);
      fast_ok = nir_iand(b, fast_ok, nir_ieq_imm(
         b, nir_iand_imm(b, nir_ior(b, nir_unpack_64_2x32_split_x(b, addr),
                                    nir_ishl_imm(b, nir_ior(b, fast->stride, x), 1)),
                         3), 0));
      if (checked) {
         fast_ok = nir_iand(b, fast_ok,
                            coords_inside(b, x, 0, 8 * last_vector + 7,
                                          fast->dim1));
      }
   }
   fast_ok = nir_iand(b, fast_ok, rows_unwrapped(b, it, fast->stride, wide));

   /* With a constant clamp in global memory, a lane whose rows are all past
    * the tensor's (the rows of a tile past a matrix's) takes the clamp value
    * where its runs read the zero page, instead of each element apart: its
    * first row is at or after the tensor's last, or its last row before the
    * tensor's first.
    */
   struct tensor_items read = *it;
   nir_deref_instr *tensor = ptr;
   nir_def *stride = fast->stride, *rows_out = NULL;
   if (info->clamp_mode == NIR_TENSOR_CLAMP_CONSTANT &&
       nir_deref_mode_is(ptr, nir_var_mem_global)) {
      nir_def *reads = fast_ok;
      rows_out = nir_ior(b, nir_iand(b, nir_ige_imm(b, it->y0, 0),
                                     nir_uge(b, it->y0, fast->dim0)),
                         nir_ilt_imm(b, it->y0, -(int)it->max_dy));
      fast_ok = nir_ior(b, fast_ok, rows_out);
      nir_def *zero = nir_imm_int(b, 0);
      tensor = nir_build_deref_cast(
         b, nir_bcsel(b, reads, global_address(b, ptr),
                      nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS)),
         nir_var_mem_global, glsl_scalar_type(desc.element_type), bits / 8);
      stride = nir_bcsel(b, reads, stride, zero);
      read.y0 = nir_bcsel(b, reads, it->y0, zero);
      read.x0 = nir_bcsel(b, reads, it->x0, it->parity ? it->parity : zero);
   }

   nir_def *fast_values[TENSOR_MAX_ELEMENTS], *other_values[TENSOR_MAX_ELEMENTS];
   nir_if *nif = nir_push_if(b, fast_ok);
   if (wide)
      load_vectors(b, &read, stride, tensor, fast_values);
   else
      load_runs(b, &read, stride, tensor, desc, fast_values);
   if (rows_out) {
      nir_def *replace = nir_iand(b, rows_out, nir_ine_imm(b, info->clamp_value, 0));
      for (unsigned i = 0; i < it->count; ++i) {
         fast_values[i] = nir_bcsel(b, replace, nir_u2uN(b, info->clamp_value, bits),
                                    fast_values[i]);
      }
   }
   nir_push_else(b, nif);
   if (!checked && wide) {
      /* Rows not 4-byte aligned, or indices that wrap within a vector. */
      nir_def *run_values[TENSOR_MAX_ELEMENTS];
      nir_if *runs = nir_push_if(b, rows_unwrapped(b, it, fast->stride, false));
      load_runs(b, it, fast->stride, ptr, desc, run_values);
      nir_push_else(b, runs);
      load_each(b, it, fast->stride, ptr, desc, other_values);
      nir_pop_if(b, runs);
      for (unsigned i = 0; i < it->count; ++i)
         other_values[i] = nir_if_phi(b, run_values[i], other_values[i]);
   } else if (!checked) {
      load_each(b, it, fast->stride, ptr, desc, other_values);
   } else {
      for (unsigned i = 0; i < it->count; ++i) {
         nir_def *y = nir_iadd_imm(b, it->y0, it->dy[i]);
         nir_def *x = nir_iadd_imm(b, it->x0, it->dx[i]);
         nir_def *in = nir_iand(b, coord_inside(b, y, fast->dim0),
                                coord_inside(b, x, fast->dim1));
         nir_def *cy = nir_bcsel(b, in, y, clamp_coord(b, info->clamp_mode, y, fast->dim0));
         nir_def *cx = nir_bcsel(b, in, x, clamp_coord(b, info->clamp_mode, x, fast->dim1));
         nir_deref_instr *d = tensor_elements(
            b, ptr, desc, nir_iadd(b, nir_imul(b, cy, fast->stride), cx), 1);
         other_values[i] = load_unless(
            b, d, clamp_skip(b, info, in, fast->dim0, fast->dim1),
            clamp_skip_value(b, info, bits));
      }
   }
   nir_pop_if(b, nif);
   for (unsigned i = 0; i < it->count; ++i)
      values[i] = nir_if_phi(b, fast_values[i], other_values[i]);
}

/* The common case's store: inside the tensor, the lane's runs; past its
 * edge, each element inside it. A lane whose indices wrap within a row
 * stores each element at its own index too.
 */
static void
tensor_fast_store(nir_builder *b, struct nir_calc_tensor_info *info,
                  struct tensor_fast *fast, nir_deref_instr *ptr,
                  struct glsl_cmat_description desc, struct tensor_items *it,
                  nir_def **values)
{
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   bool checked = info->clamp_mode != NIR_TENSOR_CLAMP_UNDEFINED;

   nir_def *runs_ok = rows_unwrapped(b, it, fast->stride, false);
   if (checked) {
      nir_def *inside =
         nir_iand(b, coords_inside(b, it->y0, 0, it->max_dy, fast->dim0),
                  coords_inside(b, it->x0, it->min_dx, it->max_dx, fast->dim1));
      runs_ok = nir_iand(b, inside, runs_ok);
   }
   nir_if *edge = nir_push_if(b, runs_ok);

   uint32_t done = 0;
   unsigned members[4], places[4], start, width;
   for (unsigned first = 0; first < it->count;) {
      if (done & BITFIELD_BIT(first)) {
         ++first;
         continue;
      }
      unsigned n = next_run(it, first, &done, members, places, &start, &width);
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[first]);
      nir_def *idx = nir_iadd(b, nir_imul(b, y, fast->stride),
                              nir_iadd_imm(b, it->x0, start));
      nir_deref_instr *d = tensor_elements(b, ptr, desc, idx, width);
      nir_def *channels[4];
      unsigned mask = 0;
      for (unsigned i = 0; i < width; ++i)
         channels[i] = nir_undef(b, 1, bits);
      for (unsigned i = 0; i < n; ++i) {
         channels[places[i]] = values[members[i]];
         mask |= BITFIELD_BIT(places[i]);
      }
      nir_store_deref(b, d, nir_vec(b, channels, width), mask);
   }

   nir_push_else(b, edge);
   for (unsigned i = 0; i < it->count; ++i) {
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[i]);
      nir_def *x = nir_iadd_imm(b, it->x0, it->dx[i]);
      nir_if *in = NULL;
      if (checked) {
         in = nir_push_if(b, nir_iand(b, coord_inside(b, y, fast->dim0),
                                      coord_inside(b, x, fast->dim1)));
      }
      nir_deref_instr *d = tensor_elements(
         b, ptr, desc, nir_iadd(b, nir_imul(b, y, fast->stride), x), 1);
      nir_store_deref(b, d, values[i], 1);
      if (checked)
         nir_pop_if(b, in);
   }
   nir_pop_if(b, edge);
}

/* The cast of the decode function's block pointer parameter
 * (nir_calc_tensor_derefs does the same), whose type is the block's.
 */
static nir_deref_instr *
decode_block_deref(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type == nir_deref_type_cast &&
             nir_def_is_intrinsic(deref->parent.ssa)) {
            nir_intrinsic_instr *param = nir_def_as_intrinsic(deref->parent.ssa);
            if (param->intrinsic == nir_intrinsic_load_param &&
                nir_intrinsic_param_idx(param) == 1)
               return deref;
         }
      }
   }
   return NULL;
}

/* One call of the decode function, for the element at `at` in the block at
 * block coordinates `block`, whose index in the tensor is idx.
 */
static nir_def *
decode_element(nir_builder *b, nir_function *fn, nir_deref_instr *param,
               nir_deref_instr *ptr, nir_def *idx, nir_def **block,
               nir_def **at)
{
   nir_def *block_ptr;
   if (param) {
      nir_deref_instr *d = nir_build_deref_cast(
         b, &ptr->def, ptr->modes, param->type,
         glsl_get_explicit_size(param->type, true));
      d = nir_build_deref_ptr_as_array(b, d, nir_u2uN(b, idx, ptr->def.bit_size));
      block_ptr = &d->def;
   } else {
      block_ptr = nir_undef(b, fn->params[1].num_components,
                            fn->params[1].bit_size);
   }

   nir_variable *out = nir_local_variable_create(
      b->impl, glsl_get_bare_type(fn->params[0].type), "decoded");
   nir_deref_instr *out_deref = nir_build_deref_var(b, out);
   nir_call_instr *call = nir_call_instr_create(b->shader, fn);
   call->params[0] = nir_src_for_ssa(&out_deref->def);
   call->params[1] = nir_src_for_ssa(block_ptr);
   for (unsigned dim = 0; dim < 2; ++dim) {
      call->params[2 + dim] = nir_src_for_ssa(block[dim]);
      call->params[4 + dim] = nir_src_for_ssa(at[dim]);
   }
   nir_builder_instr_insert(b, &call->instr);
   return nir_load_deref(b, out_deref);
}

/* The common case with a decode function: each element's block found
 * directly from its tensor coordinates, with no branch around its call.
 * When the lane's elements are inside the tensor and those of each tensor
 * row fall in one block, they share the block's address and coordinates, so
 * the decode function's loads of the block, and of what it finds by the
 * block's coordinates (a scale, say), are the same loads, which the optimizer
 * merges when the memory is read-only: with K paired, the eight elements of
 * a lane's column of a 64-deep group are one block of eight. Otherwise each
 * element is clamped as the layout says and finds its own block.
 */
static void
tensor_fast_decode(nir_builder *b, struct nir_calc_tensor_info *info,
                   struct tensor_fast *fast, nir_deref_instr *ptr,
                   struct glsl_cmat_description desc, struct tensor_items *it,
                   nir_def **values)
{
   nir_function *fn = info->decode_fnptr;
   nir_deref_instr *param = decode_block_deref(fn->impl);
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   bool checked = info->clamp_mode != NIR_TENSOR_CLAMP_UNDEFINED;
   nir_def *bs0 = nir_channel(b, info->block_sizes, 0);
   nir_def *bs1 = nir_channel(b, info->block_sizes, 1);
   nir_def *stride1 = nir_channel(b, info->strides, 1);

   nir_def *x_first = nir_iadd_imm(b, it->x0, it->min_dx);
   nir_def *x_last = nir_iadd_imm(b, it->x0, it->max_dx);
   nir_def *bx = nir_udiv(b, x_first, bs1);
   nir_def *shared = nir_ieq(b, bx, nir_udiv(b, x_last, bs1));
   if (checked) {
      shared = nir_iand(b, shared,
                        coords_inside(b, it->y0, 0, it->max_dy, fast->dim0));
      shared = nir_iand(b, shared,
                        coords_inside(b, it->x0, it->min_dx, it->max_dx, fast->dim1));
   }

   nir_def *shared_values[TENSOR_MAX_ELEMENTS], *own_values[TENSOR_MAX_ELEMENTS];
   nir_if *nif = nir_push_if(b, shared);
   {
      nir_def *ix = nir_umod(b, x_first, bs1);
      for (unsigned i = 0; i < it->count; ++i) {
         nir_def *y = nir_iadd_imm(b, it->y0, it->dy[i]);
         nir_def *block[2] = {nir_udiv(b, y, bs0), bx};
         nir_def *at[2] = {nir_umod(b, y, bs0),
                           nir_iadd_imm(b, ix, it->dx[i] - it->min_dx)};
         nir_def *idx = nir_iadd(b, nir_imul(b, block[0], fast->stride),
                                 nir_imul(b, bx, stride1));
         shared_values[i] = decode_element(b, fn, param, ptr, idx, block, at);
      }
   }
   nir_push_else(b, nif);
   for (unsigned i = 0; i < it->count; ++i) {
      nir_def *y = nir_iadd_imm(b, it->y0, it->dy[i]);
      nir_def *x = nir_iadd_imm(b, it->x0, it->dx[i]);
      nir_def *in = NULL;
      if (checked) {
         in = nir_iand(b, coord_inside(b, y, fast->dim0),
                       coord_inside(b, x, fast->dim1));
         y = nir_bcsel(b, in, y, clamp_coord(b, info->clamp_mode, y, fast->dim0));
         x = nir_bcsel(b, in, x, clamp_coord(b, info->clamp_mode, x, fast->dim1));
      }
      nir_def *block[2] = {nir_udiv(b, y, bs0), nir_udiv(b, x, bs1)};
      nir_def *at[2] = {nir_umod(b, y, bs0), nir_umod(b, x, bs1)};
      nir_def *idx = nir_iadd(b, nir_imul(b, block[0], fast->stride),
                              nir_imul(b, block[1], stride1));
      nir_def *skip = checked ? clamp_skip(b, info, in, fast->dim0, fast->dim1) : NULL;
      if (skip) {
         /* No call: the decode function would read the block. */
         nir_def *skipped = clamp_skip_value(b, info, bits);
         nir_if *call = nir_push_if(b, nir_inot(b, skip));
         nir_def *v = decode_element(b, fn, param, ptr, idx, block, at);
         nir_pop_if(b, call);
         own_values[i] = nir_if_phi(b, v, skipped);
      } else {
         own_values[i] = decode_element(b, fn, param, ptr, idx, block, at);
      }
   }
   nir_pop_if(b, nif);
   for (unsigned i = 0; i < it->count; ++i)
      values[i] = nir_if_phi(b, shared_values[i], own_values[i]);
}

/* Any other layout: each element through the common addressing. A piece of
 * a matrix of K paired 64 deep (step s of group g) gives the common
 * addressing its elements' whole coordinates.
 */
static void
tensor_generic_access(nir_builder *b, struct nir_calc_tensor_info *info,
                      nir_deref_instr *ptr, nir_deref_instr *clip,
                      struct glsl_cmat_description desc, bool load,
                      nir_def **values, unsigned count, bool paired,
                      unsigned g, unsigned s)
{
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *r0 = matrix_row(b, lane, desc, 0), *c0 = matrix_col(b, lane, desc, 0);
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   bool a = desc.use == GLSL_CMAT_USE_A;
   if (paired) {
      if (a)
         info->col_imm_offset = 0;
      else
         info->row_imm_offset = 0;
   }

   for (unsigned e = 0; e < count; ++e) {
      unsigned dr, dc;
      element_offset(desc, e, &dr, &dc);
      nir_def *clip_value = load && info->view
         ? nir_cmat_extract(b, bits, &clip->def, nir_imm_int(b, e)) : NULL;
      nir_def *row = nir_iadd_imm(b, r0, dr), *col = nir_iadd_imm(b, c0, dc);
      if (paired && a)
         col = nir_iadd_imm(b, paired_k(b, col), 64 * g + 2 * s);
      else if (paired)
         row = nir_iadd_imm(b, paired_k(b, row), 64 * g + 2 * s);

      info->clipped_if = NULL;
      nir_deref_instr *d = ptr;
      nir_def *v = nir_calc_tensor_derefs(b, info, row, col, &d);
      if (load) {
         /* A decoded element is skipped by the common addressing. */
         if (!info->decode_fnptr) {
            v = load_unless(b, d, info->skip,
                            info->skip ? nir_u2uN(b, info->skip_value, bits) : NULL);
         }
      } else {
         nir_push_if(b, nir_inot(b, info->do_clamp));
         nir_store_deref(b, d, values[e], 1);
         nir_pop_if(b, NULL);
      }

      if (info->clipped_if) {
         nir_push_else(b, info->clipped_if);
         nir_pop_if(b, info->clipped_if);
         if (load)
            v = nir_if_phi(b, v, clip_value);
      }
      if (load)
         values[e] = v;
   }
}

/* A tensor load or store, of one matrix or (K paired) of the four pieces
 * calls[0..pieces) of one 64-deep group of a matrix, steps 0 to 3 in order:
 * lowered at the last call.
 */
static bool
lower_tensor_calls(nir_builder *b, nir_cmat_call_instr **calls,
                   unsigned pieces, bool pairing)
{
   nir_cmat_call_instr *call = calls[pieces - 1];
   const bool load = call->op == nir_cmat_call_op_tensor_load;
   nir_deref_instr *mats[TENSOR_MAX_ELEMENTS / 8], *clips[TENSOR_MAX_ELEMENTS / 8];
   for (unsigned p = 0; p < pieces; ++p) {
      mats[p] = nir_src_as_deref(calls[p]->params[load ? 0 : 1]);
      clips[p] = load ? nir_src_as_deref(calls[p]->params[4]) : NULL;
   }
   nir_deref_instr *ptr = nir_src_as_deref(call->params[load ? 1 : 0]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(mats[0]->type);
   unsigned count = desc.rows * desc.cols / 32;
   unsigned bits = glsl_base_type_bit_size(desc.element_type);
   unsigned g = 0, s = 0;
   bool paired = piece_paired(calls[0], desc, pairing, &g, &s);

   b->cursor = nir_before_instr(&call->instr);

   struct nir_calc_tensor_info info = {0};
   nir_calc_tensor_derefs_init(b, &info, calls[0]);

   /* A decode function takes a global pointer to its block. */
   if (info.decode_fnptr && nir_deref_mode_is(ptr, nir_var_mem_ssbo)) {
      nir_def *addr = nir_deref_buffer_address(b, 1, 64, &ptr->def);
      ptr = nir_build_deref_cast(b, addr, nir_var_mem_global, ptr->type,
                                 bits / 8);
   }

   nir_def *values[TENSOR_MAX_ELEMENTS] = {NULL};
   if (!load) {
      for (unsigned p = 0; p < pieces; ++p) {
         for (unsigned e = 0; e < count; ++e) {
            values[p * count + e] =
               nir_cmat_extract(b, bits, &mats[p]->def, nir_imm_int(b, e));
         }
      }
   }

   /* The shape tests are of the whole matrix; desc is this piece's. */
   struct glsl_cmat_description whole = nir_cmat_call_cmat_desc(call);
   struct tensor_fast fast = {0};
   nir_def *cond = tensor_fast_condition(b, &info, whole, &fast);
   /* A piece of K paired 64 deep has the common case only with its group. */
   if (cond && (paired ? load && pieces == 4 : true)) {
      struct tensor_items it;
      tensor_items(b, &info, &fast, desc, pieces, g, &it);
      nir_def *fast_values[TENSOR_MAX_ELEMENTS], *slow_values[TENSOR_MAX_ELEMENTS];
      memcpy(fast_values, values, sizeof(values));
      memcpy(slow_values, values, sizeof(values));
      nir_if *nif = nir_push_if(b, cond);
      if (!load)
         tensor_fast_store(b, &info, &fast, ptr, desc, &it, fast_values);
      else if (info.decode_fnptr)
         tensor_fast_decode(b, &info, &fast, ptr, desc, &it, fast_values);
      else
         tensor_fast_load(b, &info, &fast, ptr, desc, &it, fast_values);
      nir_push_else(b, nif);
      for (unsigned p = 0; p < pieces; ++p) {
         struct nir_calc_tensor_info piece_info = info;
         nir_calc_tensor_derefs_init(b, &piece_info, calls[p]);
         piece_paired(calls[p], desc, pairing, &g, &s);
         tensor_generic_access(b, &piece_info, ptr, clips[p], desc, load,
                               slow_values + p * count, count, paired, g, s);
      }
      nir_pop_if(b, nif);
      if (load) {
         for (unsigned i = 0; i < pieces * count; ++i)
            values[i] = nir_if_phi(b, fast_values[i], slow_values[i]);
      }
   } else {
      for (unsigned p = 0; p < pieces; ++p) {
         struct nir_calc_tensor_info piece_info = info;
         if (p > 0)
            nir_calc_tensor_derefs_init(b, &piece_info, calls[p]);
         piece_paired(calls[p], desc, pairing, &g, &s);
         tensor_generic_access(b, &piece_info, ptr, clips[p], desc, load,
                               values + p * count, count, paired, g, s);
      }
   }

   if (load) {
      for (unsigned p = 0; p < pieces; ++p) {
         for (unsigned e = 0; e < count; ++e) {
            nir_cmat_insert(b, &mats[p]->def, values[p * count + e],
                            &mats[p]->def, nir_imm_int(b, e));
         }
      }
   }

   for (unsigned p = 0; p < pieces; ++p)
      nir_instr_remove(&calls[p]->instr);
   return true;
}

static bool
function_is_called(nir_shader *nir, nir_function *func)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_cmat_call &&
                nir_instr_as_cmat_call(instr)->callee == func)
               return true;
         }
      }
   }
   return false;
}

/* Whether two tensor loads are pieces of one load (of a matrix split into
 * 16x16 pieces), differing only in their piece.
 */
static bool
same_tensor_load(nir_cmat_call_instr *a, nir_cmat_call_instr *b)
{
   if (a->op != nir_cmat_call_op_tensor_load || b->op != a->op ||
       a->callee != b->callee)
      return false;
   for (unsigned i = 1; i <= 3; ++i) {
      if (a->params[i].ssa != b->params[i].ssa)
         return false;
   }
   struct nir_cmat_tensor_load ta = nir_cmat_call_tensor_load_info(a);
   struct nir_cmat_tensor_load tb = nir_cmat_call_tensor_load_info(b);
   ta.split_row_index = tb.split_row_index = 0;
   ta.split_col_index = tb.split_col_index = 0;
   struct glsl_cmat_description da = nir_cmat_call_cmat_desc(a);
   struct glsl_cmat_description db = nir_cmat_call_cmat_desc(b);
   return !memcmp(&ta, &tb, sizeof(ta)) && !memcmp(&da, &db, sizeof(da));
}

/* Whether only derefs and arithmetic lie between two instructions of a
 * block: nothing that reads a matrix or touches memory.
 */
static bool
only_addressing_between(nir_instr *from, nir_instr *to)
{
   for (nir_instr *instr = nir_instr_next(from); instr != to;
        instr = nir_instr_next(instr)) {
      if (!instr || (instr->type != nir_instr_type_deref &&
                     instr->type != nir_instr_type_alu &&
                     instr->type != nir_instr_type_load_const &&
                     instr->type != nir_instr_type_undef))
         return false;
   }
   return true;
}

/* The piece's index across K (with g and s) and along the other dimension. */
static bool
paired_piece(nir_cmat_call_instr *call, bool pairing, unsigned *g, unsigned *s,
             unsigned *other)
{
   if (call->op != nir_cmat_call_op_tensor_load)
      return false;
   nir_deref_instr *mat = nir_src_as_deref(call->params[0]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(mat->type);
   if (!piece_paired(call, desc, pairing, g, s))
      return false;
   struct nir_cmat_tensor_load tl = nir_cmat_call_tensor_load_info(call);
   *other = desc.use == GLSL_CMAT_USE_A ? tl.split_row_index : tl.split_col_index;
   return true;
}

static bool
lower_tensor_block(nir_builder *b, nir_block *block, bool pairing)
{
   struct util_dynarray calls;
   util_dynarray_init(&calls, NULL);
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_cmat_call)
         continue;
      nir_cmat_call_instr *call = nir_instr_as_cmat_call(instr);
      /* A matrix of a size the lowering lacks stays a call, which
       * agx_nir_cmat_supported() refuses.
       */
      if ((call->op == nir_cmat_call_op_tensor_load ||
           call->op == nir_cmat_call_op_tensor_store) &&
          matrices_supported(call))
         util_dynarray_append(&calls, call);
   }
   unsigned n = util_dynarray_num_elements(&calls, nir_cmat_call_instr *);
   nir_cmat_call_instr **list = util_dynarray_begin(&calls);

   for (unsigned i = 0; i < n;) {
      /* The run of pieces of one load from list[i]. */
      unsigned end = i + 1;
      while (pairing && end < n && same_tensor_load(list[i], list[end]) &&
             only_addressing_between(&list[end - 1]->instr, &list[end]->instr))
         end++;

      /* The four steps of each 64-deep group, together. */
      bool done[64] = {false};
      if (end - i > 64)
         end = i + 64;
      for (unsigned j = i; j < end; ++j) {
         unsigned g, s, other;
         if (!paired_piece(list[j], pairing, &g, &s, &other) || s != 0)
            continue;
         nir_cmat_call_instr *group[4] = {list[j]};
         unsigned found = 1, at[4] = {j};
         for (unsigned k = j + 1; k < end && found < 4; ++k) {
            unsigned g2, s2, other2;
            if (!done[k - i] && paired_piece(list[k], pairing, &g2, &s2, &other2) &&
                g2 == g && other2 == other && s2 == found) {
               group[found] = list[k];
               at[found++] = k;
            }
         }
         if (found == 4) {
            lower_tensor_calls(b, group, 4, pairing);
            for (unsigned p = 0; p < 4; ++p)
               done[at[p] - i] = true;
         }
      }
      for (unsigned j = i; j < end; ++j) {
         if (!done[j - i])
            lower_tensor_calls(b, &list[j], 1, pairing);
      }
      i = end;
   }

   util_dynarray_fini(&calls);
   return n > 0;
}

bool
agx_nir_lower_cmat_tensor(nir_shader *nir, bool pairing)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || !nir->info.cs.has_cooperative_matrix)
      return false;

   bool progress = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_safe(block, impl)
      progress |= lower_tensor_block(&b, block, pairing);
   if (!progress)
      return nir_no_progress(impl);

   nir_progress(true, impl, nir_metadata_none);

   /* Inline the decode functions' calls, then drop the functions no
    * cooperative matrix call refers to any more.
    */
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_foreach_function_safe(func, nir) {
      if (!func->is_entrypoint && !function_is_called(nir, func))
         exec_node_remove(&func->node);
   }
   return true;
}

/* Whether a type holds a matrix larger than 16x16, and whether its matrices
 * split into 16x16 pieces (*splittable) or into 8x8 ones (*splittable_8).
 */
static bool
type_needs_split(const struct glsl_type *type, bool *splittable,
                 bool *splittable_8)
{
   type = glsl_without_array(type);
   if (glsl_type_is_struct(type)) {
      bool split = false;
      for (unsigned i = 0; i < glsl_get_length(type); ++i)
         split |= type_needs_split(glsl_get_struct_field(type, i), splittable,
                                   splittable_8);
      return split;
   }
   if (!glsl_type_is_cmat(type))
      return false;
   struct glsl_cmat_description desc = *glsl_get_cmat_description(type);
   /* 8x8 pieces split every dimension longer than 8. */
   if ((desc.rows > 8 && desc.rows % 8) || (desc.cols > 8 && desc.cols % 8))
      *splittable_8 = false;
   /* Up to 16x16 needs no split of its own (8x8 pieces still divide it). */
   if (desc.rows <= 16 && desc.cols <= 16)
      return false;
   if (desc.rows % 16 || desc.cols % 16)
      *splittable = false;
   return true;
}

/* Whether a type holds an A or B matrix whose K is a multiple of 64. */
static bool
type_pairs_k(const struct glsl_type *type)
{
   type = glsl_without_array(type);
   if (glsl_type_is_struct(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); ++i) {
         if (type_pairs_k(glsl_get_struct_field(type, i)))
            return true;
      }
      return false;
   }
   if (!glsl_type_is_cmat(type))
      return false;
   struct glsl_cmat_description desc = *glsl_get_cmat_description(type);
   return (desc.use == GLSL_CMAT_USE_A && desc.cols % 64 == 0) ||
          (desc.use == GLSL_CMAT_USE_B && desc.rows % 64 == 0);
}

static bool
src_pairs_k(nir_src *src, void *data)
{
   if (nir_src_is_deref(*src) && type_pairs_k(nir_src_as_deref(*src)->type)) {
      *(bool *)data = true;
      return false;
   }
   return true;
}

/* Whether the shader could tell K paired 64 deep (the tensor loads' comment)
 * from K in order: a matrix of such a K loaded or stored by coopMatLoad or
 * coopMatStore, or under an operation that gives an element's place or moves
 * elements between places. Elementwise operations, element access by index,
 * products and tensor loads and stores cannot.
 */
static bool
k_order_visible(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            bool visible = false;
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               switch (intr->intrinsic) {
               case nir_intrinsic_cmat_convert:
                  /* A conversion between uses (an accumulator to an A
                   * operand, say) puts its source's elements in K order.
                   */
                  if (matrix_desc(intr->src[0]).use != matrix_desc(intr->src[1]).use)
                     nir_foreach_src(instr, src_pairs_k, &visible);
                  break;
               case nir_intrinsic_cmat_construct:
               case nir_intrinsic_cmat_copy:
               case nir_intrinsic_cmat_length:
               case nir_intrinsic_cmat_insert:
               case nir_intrinsic_cmat_extract:
               case nir_intrinsic_cmat_unary_op:
               case nir_intrinsic_cmat_binary_op:
               case nir_intrinsic_cmat_scalar_op:
               case nir_intrinsic_cmat_bitcast:
               case nir_intrinsic_cmat_muladd:
               case nir_intrinsic_load_deref:
               case nir_intrinsic_store_deref:
               case nir_intrinsic_copy_deref:
                  break;
               default:
                  nir_foreach_src(instr, src_pairs_k, &visible);
                  break;
               }
            } else if (instr->type == nir_instr_type_cmat_call) {
               nir_cmat_call_instr *call = nir_instr_as_cmat_call(instr);
               if (call->op != nir_cmat_call_op_tensor_load &&
                   call->op != nir_cmat_call_op_tensor_store)
                  nir_foreach_src(instr, src_pairs_k, &visible);
            } else if (instr->type == nir_instr_type_call) {
               nir_foreach_src(instr, src_pairs_k, &visible);
            }
            if (visible)
               return true;
         }
      }
   }
   return false;
}

/* Whether the shader has an operation only 16x16 matrices take: a product
 * of A and B operands of different types, or a 2x2 reduction.
 */
static bool
needs_16x16_pieces(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_cmat_call &&
                nir_instr_as_cmat_call(instr)->op == nir_cmat_call_op_reduce_2x2)
               return true;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_cmat_muladd &&
                matrix_desc(intr->src[1]).element_type !=
                   matrix_desc(intr->src[2]).element_type)
               return true;
         }
      }
   }
   return false;
}

/*
 * Matrices of flexible dimensions (VK_NV_cooperative_matrix2
 * cooperativeMatrixFlexibleDimensions), multiples of 16 in each dimension at
 * subgroup scope, as 16x16 pieces (nir_lower_cooperative_matrix_flexible_
 * dimensions). A shader whose matrices are all 16x16 or 8x8 is left alone,
 * and with 16x16 pieces such matrices stay whole beside larger ones.
 * A shader with a larger matrix that is a multiple of 8 but not of 16 (the
 * result of a reduction of an 8x8 matrix that changes its dimensions, say)
 * is split into 8x8 pieces instead, unless it has products of mixed types
 * or 2x2 reductions, which need 16x16 matrices.
 * *pairing is whether the pieces of A and B matrices whose K is a multiple of
 * 64 take K paired 64 deep (the tensor loads' comment).
 */
bool
agx_nir_lower_cmat_flexible(nir_shader *nir, bool *pairing)
{
   *pairing = false;
   if (nir->info.stage != MESA_SHADER_COMPUTE || !nir->info.cs.has_cooperative_matrix)
      return false;

   bool split = false, splittable = true, splittable_8 = true;
   nir_foreach_variable_in_shader(var, nir)
      split |= type_needs_split(var->type, &splittable, &splittable_8);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl)
         split |= type_needs_split(var->type, &splittable, &splittable_8);
   }
   if (!split)
      return false;

   unsigned gran = 16;
   if (!splittable) {
      if (!splittable_8 || needs_16x16_pieces(nir))
         return false;
      gran = 8;
   }

   *pairing = gran == 16 && !k_order_visible(nir);

   const struct nir_lower_coopmat_args args = {
      .m_gran = gran,
      .n_gran = gran,
      .k_gran = gran,
   };
   return nir_lower_cooperative_matrix_flexible_dimensions(nir, &args);
}
