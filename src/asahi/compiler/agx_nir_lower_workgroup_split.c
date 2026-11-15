/*
 * Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/bitscan.h"
#include "agx_nir.h"

/*
 * Let the driver launch a big API workgroup as several smaller hardware
 * workgroups.
 *
 * G17P runs the same invocations measurably slower in 512- and
 * 1024-invocation workgroups than in 128-256-invocation ones. A compute
 * shader that shares nothing across its workgroup (no workgroup memory, no
 * workgroup-scope barrier) observes the workgroup only through its builtins.
 * Derive those from the invocation's position in the grid instead of from the
 * hardware workgroup, and they are the same for every hardware workgroup shape
 * that tiles the grid along one axis: the driver may then launch a direct
 * dispatch in the smaller groups agx_workgroup_split_size() returns, and any
 * other launch keeps the API shape.
 *
 * The split axis is the outermost one with an extent above one, so every
 * hardware workgroup is a contiguous range of API local invocation indices,
 * a multiple of 32 long: every subgroup holds exactly the invocations it held
 * before, and subgroup and quad operations are unchanged.
 */

#define AGX_SPLIT_MAX_THREADS 256

static bool
refuses_split(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_barrier:
      /* Subgroup-scope execution and device-scope memory barriers do not
       * depend on the workgroup; workgroup scope does. */
      return nir_intrinsic_execution_scope(intr) >= SCOPE_WORKGROUP ||
             nir_intrinsic_memory_scope(intr) == SCOPE_WORKGROUP;
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_num_subgroups:
   case nir_intrinsic_load_workgroup_index:
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   /* Tile memory is shared by the workgroup like workgroup memory. */
   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
   case nir_intrinsic_image_deref_store_block_agx:
      return true;
   default:
      return false;
   }
}

/* The hardware workgroup size a direct launch of this shader may use, or false
 * when it must keep the API size. */
bool
agx_workgroup_split_size(const nir_shader *nir, uint16_t size[3])
{
   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       nir->info.workgroup_size_variable || nir->info.shared_size ||
       nir->info.derivative_group != DERIVATIVE_GROUP_NONE ||
       nir->info.ray_queries || nir->info.cs.image_block_size_per_thread_agx)
      return false;

   const uint16_t *api = nir->info.workgroup_size;
   const unsigned threads = api[0] * api[1] * api[2];
   if (threads <= AGX_SPLIT_MAX_THREADS)
      return false;

   unsigned axis = api[2] > 1 ? 2 : api[1] > 1 ? 1 : 0;
   /* The derived builtins divide the 32-bit thread position in the grid. Along
    * x that can wrap (an indirect dispatch may launch up to 0x7fffffff groups),
    * along y and z it cannot (65535 groups of at most 1024). */
   if (axis == 0)
      return false;
   unsigned k = 2;
   while ((api[axis] % k) == 0 && threads / k > AGX_SPLIT_MAX_THREADS)
      k *= 2;
   if ((api[axis] % k) != 0 || ((threads / k) % 32) != 0)
      return false;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                refuses_split(nir_instr_as_intrinsic(instr)))
               return false;
         }
      }
   }

   for (unsigned i = 0; i < 3; ++i)
      size[i] = api[i];
   size[axis] /= k;
   return true;
}

static nir_def *
grid_div(nir_builder *b, nir_def *x, unsigned d)
{
   return util_is_power_of_two_nonzero(d) ? nir_ushr_imm(b, x, util_logbase2(d))
                                          : nir_udiv_imm(b, x, d);
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned axis = *(const unsigned *)data;
   const uint16_t *api = b->shader->info.workgroup_size;

   if (intr->intrinsic != nir_intrinsic_load_workgroup_id &&
       intr->intrinsic != nir_intrinsic_load_local_invocation_id &&
       intr->intrinsic != nir_intrinsic_load_local_invocation_index &&
       intr->intrinsic != nir_intrinsic_load_workgroup_size)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   const unsigned bit_size = intr->def.bit_size;

   if (intr->intrinsic == nir_intrinsic_load_workgroup_size) {
      nir_def *v = nir_imm_ivec3(b, api[0], api[1], api[2]);
      nir_def_replace(&intr->def, nir_u2uN(b, v, bit_size));
      return true;
   }

   /* Only the split axis differs between hardware shapes; the hardware values
    * of the other axes are the API ones. */
   nir_def *g = nir_channel(b, nir_load_global_invocation_id(b, 32), axis);
   nir_def *wg_a = grid_div(b, g, api[axis]);

   nir_def *v;
   if (intr->intrinsic == nir_intrinsic_load_workgroup_id) {
      v = nir_vector_insert_imm(b, nir_u2u32(b, nir_load_workgroup_id(b)),
                                wg_a, axis);
   } else {
      nir_def *lid_a = nir_isub(b, g, nir_imul_imm(b, wg_a, api[axis]));
      nir_def *lid = nir_vector_insert_imm(
         b, nir_u2u32(b, nir_load_local_invocation_id(b)), lid_a, axis);

      if (intr->intrinsic == nir_intrinsic_load_local_invocation_id) {
         v = lid;
      } else {
         v = nir_iadd(b, nir_channel(b, lid, 0),
                      nir_imul_imm(b,
                                   nir_iadd(b, nir_channel(b, lid, 1),
                                            nir_imul_imm(b, nir_channel(b, lid, 2),
                                                         api[1])),
                                   api[0]));
      }
   }

   nir_def_replace(&intr->def, nir_u2uN(b, v, bit_size));
   return true;
}

/* Rewrite the workgroup builtins of a shader agx_workgroup_split_size()
 * accepted so they do not depend on the hardware workgroup shape. */
bool
agx_nir_lower_workgroup_split(nir_shader *nir, const uint16_t size[3])
{
   unsigned axis = 0;
   for (unsigned i = 0; i < 3; ++i) {
      if (size[i] != nir->info.workgroup_size[i])
         axis = i;
   }

   /* New loads go before the instruction they replace, behind the pass's
    * iterator, so they are not revisited. */
   return nir_shader_intrinsics_pass(nir, lower, nir_metadata_control_flow,
                                     &axis);
}
