/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"
#include "agx_nir.h"
#include "nir.h"

/*
 * Factor common constant multipliers in scalar integer linear combinations:
 *
 *    a * C + b * C  ->  (a + b) * C
 *    (a << k) + (b << k)  ->  (a + b) << k
 *
 * In Z / 2^B Z (16- and 32-bit modular integer arithmetic), addition is
 * associative and commutative, multiplication distributes over addition, and
 * left-shifting by 0 < k < B is exact multiplication by 2^k, so this rewrite
 * is bit-exact for all signed and unsigned integer inputs regardless of
 * wrap-around.
 *
 * Symmetric separable convolution / FIR filters (for example 17-tap Gaussian
 * kernels) repeat weights in pairs; factoring halves the number of integer
 * multiplies (which run on AGX's slower IC pipe) in exchange for cheap SCIB
 * additions.
 */

#define MAX_LEAVES 64

static bool
is_scalar_int_iadd(nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_iadd && alu->def.num_components == 1 &&
          (alu->def.bit_size == 16 || alu->def.bit_size == 32);
}

static bool
is_interior_iadd(nir_alu_instr *alu)
{
   if (!list_is_singular(&alu->def.uses))
      return false;

   nir_src *use = list_first_entry(&alu->def.uses, nir_src, use_link);
   if (nir_src_is_if(use))
      return false;

   nir_instr *user = nir_src_use_instr(use);
   if (user->block != alu->instr.block || !is_scalar_int_iadd(user))
      return false;

   return nir_instr_as_alu(user)->def.bit_size == alu->def.bit_size;
}

static bool
is_single_use_in_block(nir_scalar s, nir_block *block)
{
   return nir_scalar_is_alu(s) && s.def->num_components == 1 &&
          list_is_singular(&s.def->uses) &&
          nir_def_instr(s.def)->block == block;
}

static nir_scalar
alu_src_scalar(nir_scalar s, unsigned alu_src_idx)
{
   nir_alu_instr *alu = nir_scalar_as_alu(s);
   nir_scalar out = {
      .def = alu->src[alu_src_idx].src.ssa,
      .comp = alu->src[alu_src_idx].swizzle[s.comp],
   };
   return out;
}

static nir_scalar
chase_single_use_movs(nir_scalar s, nir_block *block)
{
   while (is_single_use_in_block(s, block) &&
          nir_scalar_alu_op(s) == nir_op_mov) {
      s = alu_src_scalar(s, 0);
   }
   return s;
}

struct scaled_group {
   uint64_t factor;
   unsigned count;
   nir_scalar items[MAX_LEAVES];
};

static bool
factor_iadd_tree(nir_builder *b, nir_alu_instr *root)
{
   nir_block *block = root->instr.block;
   const unsigned bit_size = root->def.bit_size;
   const uint64_t mask = BITFIELD64_MASK(bit_size);

   /* Flatten single-use in-block iadd interior nodes into leaves. */
   nir_scalar stack[MAX_LEAVES];
   nir_scalar leaves[MAX_LEAVES];
   unsigned sp = 0;
   unsigned num_leaves = 0;

   nir_scalar root_s = nir_get_scalar(&root->def, 0);
   stack[sp++] = alu_src_scalar(root_s, 1);
   stack[sp++] = alu_src_scalar(root_s, 0);

   while (sp > 0) {
      nir_scalar s = chase_single_use_movs(stack[--sp], block);
      if (is_single_use_in_block(s, block) &&
          nir_scalar_alu_op(s) == nir_op_iadd) {
         if (num_leaves + sp + 2 > MAX_LEAVES)
            return false;
         stack[sp++] = alu_src_scalar(s, 1);
         stack[sp++] = alu_src_scalar(s, 0);
      } else {
         if (num_leaves >= MAX_LEAVES)
            return false;
         leaves[num_leaves++] = s;
      }
   }

   if (num_leaves < 2)
      return false;

   uint64_t const_sum = 0;
   nir_scalar plain[MAX_LEAVES];
   unsigned num_plain = 0;

   struct scaled_group groups[MAX_LEAVES];
   unsigned num_groups = 0;

   for (unsigned i = 0; i < num_leaves; ++i) {
      nir_scalar s = leaves[i];
      nir_scalar resolved = nir_scalar_chase_movs(s);
      if (nir_scalar_is_const(resolved)) {
         const_sum = (const_sum + nir_scalar_as_uint(resolved)) & mask;
         continue;
      }

      uint64_t factor = 1;
      nir_scalar base = s;

      if (is_single_use_in_block(s, block)) {
         nir_op op = nir_scalar_alu_op(s);
         if (op == nir_op_imul) {
            nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
            nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
            if (nir_scalar_is_const(s1)) {
               uint64_t c = nir_scalar_as_uint(s1) & mask;
               if (c > 1) {
                  factor = c;
                  base = s0;
               }
            } else if (nir_scalar_is_const(s0)) {
               uint64_t c = nir_scalar_as_uint(s0) & mask;
               if (c > 1) {
                  factor = c;
                  base = s1;
               }
            }
         } else if (op == nir_op_ishl) {
            nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
            nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
            if (nir_scalar_is_const(s1)) {
               unsigned k = nir_scalar_as_uint(s1) & (bit_size - 1);
               if (k > 0) {
                  factor = (UINT64_C(1) << k) & mask;
                  base = s0;
               }
            }
         }
      }

      if (factor <= 1) {
         plain[num_plain++] = base;
         continue;
      }

      unsigned idx = num_groups;
      for (unsigned g = 0; g < num_groups; ++g) {
         if (groups[g].factor == factor) {
            idx = g;
            break;
         }
      }
      if (idx == num_groups) {
         groups[num_groups].factor = factor;
         groups[num_groups].count = 0;
         num_groups++;
      }
      groups[idx].items[groups[idx].count++] = base;
   }

   /* Require at least one repeated factor whose elimination saves a real
    * multiply or a shift beyond AGX's single-instruction iadd shift (1..4).
    */
   bool profitable = false;
   for (unsigned g = 0; g < num_groups; ++g) {
      if (groups[g].count < 2)
         continue;
      const uint64_t f = groups[g].factor;
      if (!util_is_power_of_two_nonzero(f) || (unsigned)(ffsll(f) - 1) > 4) {
         profitable = true;
         break;
      }
   }
   if (!profitable)
      return false;

   b->cursor = nir_before_instr(&root->instr);

   nir_def *acc = NULL;
   if (const_sum != 0)
      acc = nir_imm_intN_t(b, const_sum, bit_size);

   for (unsigned i = 0; i < num_plain; ++i) {
      nir_def *v = nir_mov_scalar(b, plain[i]);
      acc = acc ? nir_iadd(b, acc, v) : v;
   }

   /* Emit multiplies and large shifts before small shifts (1..4) so small
    * shifts accumulate onto `acc` via a single shifted iadd instead of
    * starting the chain with an unfused ishl.
    */
   for (unsigned pass = 0; pass < 2; ++pass) {
      for (unsigned g = 0; g < num_groups; ++g) {
         const uint64_t f = groups[g].factor;
         const bool small_shift =
            util_is_power_of_two_nonzero(f) && (unsigned)(ffsll(f) - 1) <= 4;
         if (small_shift != (pass == 1))
            continue;

         nir_def *sum = nir_mov_scalar(b, groups[g].items[0]);
         for (unsigned j = 1; j < groups[g].count; ++j)
            sum = nir_iadd(b, sum, nir_mov_scalar(b, groups[g].items[j]));

         nir_def *scaled;
         if (util_is_power_of_two_nonzero(f)) {
            scaled = nir_ishl_imm(b, sum, (unsigned)(ffsll(f) - 1));
         } else {
            scaled = nir_imul(b, sum, nir_imm_intN_t(b, f, bit_size));
         }

         acc = acc ? nir_iadd(b, acc, scaled) : scaled;
      }
   }

   if (!acc)
      acc = nir_imm_intN_t(b, 0, bit_size);

   nir_def_replace(&root->def, acc);
   return true;
}

bool
agx_nir_opt_factor_const_mul(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block) {
            if (!is_scalar_int_iadd(instr))
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (list_is_empty(&alu->def.uses) || is_interior_iadd(alu))
               continue;

            impl_progress |= factor_iadd_tree(&b, alu);
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}
