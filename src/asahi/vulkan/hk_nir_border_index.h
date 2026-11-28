#pragma once
#include "compiler/nir/nir.h"

struct hk_nir_border_addr {
   nir_scalar base;
   nir_scalar dyn;
   uint64_t offset;
   bool has_dyn;
};

static inline bool
hk_nir_border_dyn_offset(nir_scalar value, bool require_const, nir_scalar *dyn,
                         uint64_t *offset)
{
   value = nir_scalar_chase_movs(value);
   if (!nir_scalar_is_alu(value) || nir_scalar_alu_op(value) != nir_op_u2u64)
      return false;

   nir_scalar off32 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(value, 0));
   if (off32.def->bit_size != 32)
      return false;

   if (nir_scalar_is_alu(off32)) {
      nir_op op = nir_scalar_alu_op(off32);
      if (op == nir_op_iadd || op == nir_op_ior) {
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar constant =
               nir_scalar_chase_movs(nir_scalar_chase_alu_src(off32, i));
            if (!nir_scalar_is_const(constant))
               continue;
            nir_scalar other =
               nir_scalar_chase_movs(nir_scalar_chase_alu_src(off32, i ^ 1));
            uint64_t c = nir_scalar_as_uint(constant);
            if (op == nir_op_ior) {
               if (!nir_scalar_is_alu(other) ||
                   nir_scalar_alu_op(other) != nir_op_ishl)
                  continue;
               nir_scalar shift =
                  nir_scalar_chase_movs(nir_scalar_chase_alu_src(other, 1));
               if (!nir_scalar_is_const(shift))
                  continue;
               uint64_t s = nir_scalar_as_uint(shift);
               if (s >= 32 || c >= (UINT64_C(1) << s))
                  continue;
            }
            *dyn = other;
            *offset = c;
            return true;
         }
      }
   }

   if (require_const)
      return false;

   *dyn = off32;
   *offset = 0;
   return true;
}

static inline unsigned
hk_nir_border_address(nir_scalar value, struct hk_nir_border_addr out[2])
{
   value = nir_scalar_chase_movs(value);
   nir_intrinsic_instr *load = nir_scalar_as_intrinsic(value);
   if (!load || value.comp || load->intrinsic != nir_intrinsic_load_global_constant ||
       load->def.bit_size != 16 || load->def.num_components != 1 ||
       (nir_intrinsic_access(load) & ACCESS_VOLATILE))
      return 0;
   nir_scalar addr = nir_scalar_chase_movs(nir_get_scalar(load->src[0].ssa, 0));
   if (addr.def->bit_size != 64 || !nir_scalar_is_alu(addr) ||
       nir_scalar_alu_op(addr) != nir_op_iadd)
      return 0;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar constant = nir_scalar_chase_movs(nir_scalar_chase_alu_src(addr, i));
      if (nir_scalar_is_const(constant)) {
         out[0].offset = nir_scalar_as_uint(constant);
         out[0].base = nir_scalar_chase_movs(nir_scalar_chase_alu_src(addr, i ^ 1));
         out[0].has_dyn = false;
         return 1;
      }
   }
   unsigned count = 0;
   for (int pass = 0; pass < 2 && count == 0; ++pass) {
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar term = nir_scalar_chase_alu_src(addr, i);
         if (hk_nir_border_dyn_offset(term, pass == 0, &out[count].dyn,
                                      &out[count].offset)) {
            out[count].base =
               nir_scalar_chase_movs(nir_scalar_chase_alu_src(addr, i ^ 1));
            out[count].has_dyn = true;
            count++;
         }
      }
   }
   return count;
}

static inline bool
hk_nir_border_indices_same_descriptor(nir_scalar original, nir_scalar unbiased,
                                      uint64_t delta)
{
   struct hk_nir_border_addr orig[2], unb[2];
   unsigned orig_n = hk_nir_border_address(original, orig);
   unsigned unb_n = hk_nir_border_address(unbiased, unb);
   for (unsigned i = 0; i < orig_n; ++i) {
      for (unsigned j = 0; j < unb_n; ++j) {
         if (nir_scalar_equal(orig[i].base, unb[j].base) &&
             orig[i].has_dyn == unb[j].has_dyn &&
             (!orig[i].has_dyn || nir_scalar_equal(orig[i].dyn, unb[j].dyn)) &&
             orig[i].offset >= unb[j].offset &&
             orig[i].offset - unb[j].offset == delta)
            return true;
      }
   }
   return false;
}
