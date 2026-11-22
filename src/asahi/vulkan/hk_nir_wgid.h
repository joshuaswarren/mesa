/* SPDX-License-Identifier: MIT */
#ifndef HK_NIR_WGID_H
#define HK_NIR_WGID_H

#include <stdbool.h>
#include <stdint.h>

#include "compiler/nir/nir.h"
#include "util/bitscan.h"
#include "util/u_math.h"

/* Direct compute alternates and producer/clamp fusion assume zero base
 * workgroup offsets (no vkCmdDispatchBase shift), which is also required for
 * the bare unit-axis load_workgroup_id[axis] == GlobalInvocationID[axis]
 * equivalence. */
static inline bool
hk_nir_dispatch_zero_base(const uint32_t base_group[3])
{
   return !base_group[0] && !base_group[1] && !base_group[2];
}

/* Match 32-bit base_workgroup_id[axis] * local_size[axis] (scalar or vector
 * imul/ishl, or bare base_workgroup_id[axis] when local_size == 1).
 * nir_lower_compute_system_values emits vec3 imul(load_base_workgroup_id,
 * vec3(workgroup_size)) even when an axis has local_size == 1, before any
 * scalarization pass runs, and SPIR-V WorkgroupId lowers directly to
 * load_workgroup_id + load_base_workgroup_id. */
static inline bool
hk_nir_match_base_wg_scaled(nir_scalar s, unsigned axis, uint32_t local_size)
{
   s = nir_scalar_chase_movs(s);
   if (!s.def || !local_size || s.def->bit_size != 32)
      return false;

   if (local_size == 1 && s.comp == axis &&
       nir_def_is_intrinsic(s.def) &&
       nir_def_as_intrinsic(s.def)->intrinsic ==
          nir_intrinsic_load_base_workgroup_id)
      return true;

   if (nir_def_instr(s.def)->type != nir_instr_type_alu)
      return false;

   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_imul) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (a.def && b.def && a.def->bit_size == 32 && b.def->bit_size == 32) {
         if (a.comp == axis && nir_def_is_intrinsic(a.def) &&
             nir_def_as_intrinsic(a.def)->intrinsic ==
                nir_intrinsic_load_base_workgroup_id &&
             nir_scalar_is_const(b) && nir_scalar_as_uint(b) == local_size)
            return true;
         if (b.comp == axis && nir_def_is_intrinsic(b.def) &&
             nir_def_as_intrinsic(b.def)->intrinsic ==
                nir_intrinsic_load_base_workgroup_id &&
             nir_scalar_is_const(a) && nir_scalar_as_uint(a) == local_size)
            return true;
      }
   }

   if (util_is_power_of_two_nonzero(local_size) && op == nir_op_ishl) {
      const unsigned shift = ffs(local_size) - 1;
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (a.def && b.def && a.def->bit_size == 32 &&
          a.comp == axis && nir_def_is_intrinsic(a.def) &&
          nir_def_as_intrinsic(a.def)->intrinsic ==
             nir_intrinsic_load_base_workgroup_id &&
          nir_scalar_is_const(b) && nir_scalar_as_uint(b) == shift)
         return true;
   }

   return false;
}

#endif
