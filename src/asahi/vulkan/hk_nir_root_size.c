/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_root_size.h"

/* Follow only constant pointer additions to ordinary root loads. Descriptor
 * indexing, reflection, casts, stores and any unrecognized use keep the whole
 * table. This is a footprint proof, not an assumption about a shader's stage
 * or API descriptor layout. Bound recursion even for adversarial NIR. */
static unsigned
root_extent(nir_def *def, uint64_t offset, unsigned full_size, unsigned depth)
{
   if (depth > 8 || offset >= full_size)
      return full_size;

   unsigned end = 0;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src))
         return full_size;
      nir_instr *use = nir_src_use_instr(src);
      unsigned size;
      if (use->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(use);
         if (alu->op != nir_op_iadd || alu->def.bit_size != 64 ||
             alu->def.num_components != 1)
            return full_size;
         const unsigned other_idx = alu->src[0].src.ssa == def ? 1 : 0;
         nir_src other = alu->src[other_idx].src;
         const unsigned comp = alu->src[other_idx].swizzle[0];
         if (!nir_src_is_const(other) || comp >= nir_src_num_components(other))
            return full_size;
         const uint64_t add = nir_src_comp_as_uint(other, comp);
         if (add >= full_size || offset + add >= full_size)
            return full_size;
         size = root_extent(&alu->def, offset + add, full_size, depth + 1);
      } else if (use->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(use);
         if (intr->intrinsic != nir_intrinsic_load_global_constant ||
             src != &intr->src[0])
            return full_size;
         /* The first field is a self pointer used for descriptor reflection.
          * A load of it can reach any later root field through that pointer. */
         if (offset < sizeof(uint64_t))
            return full_size;
         const unsigned bytes =
            DIV_ROUND_UP(intr->def.num_components * intr->def.bit_size, 8);
         if (bytes > full_size - offset)
            return full_size;
         size = offset + bytes;
      } else {
         return full_size;
      }
      end = MAX2(end, size);
   }
   return end;
}

unsigned
hk_nir_root_size(const nir_shader *nir, unsigned full_size)
{
   unsigned end = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_root_agx)
               end = MAX2(end, root_extent(&intr->def, 0, full_size, 0));
         }
      }
   }
   return end;
}

static bool
root_may_read_range(nir_def *def, uint64_t offset, uint64_t start_B,
                    uint64_t end_B, unsigned full_size, unsigned depth)
{
   if (depth > 8 || offset >= full_size)
      return true;

   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src))
         return true;
      nir_instr *use = nir_src_use_instr(src);
      if (use->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(use);
         if (alu->op != nir_op_iadd || alu->def.bit_size != 64 ||
             alu->def.num_components != 1)
            return true;
         const unsigned other_idx = alu->src[0].src.ssa == def ? 1 : 0;
         nir_src other = alu->src[other_idx].src;
         const unsigned comp = alu->src[other_idx].swizzle[0];
         if (!nir_src_is_const(other) || comp >= nir_src_num_components(other))
            return true;
         const uint64_t add = nir_src_comp_as_uint(other, comp);
         if (add >= full_size || offset + add >= full_size)
            return true;
         if (root_may_read_range(&alu->def, offset + add, start_B, end_B,
                                 full_size, depth + 1))
            return true;
      } else if (use->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(use);
         if ((intr->intrinsic != nir_intrinsic_load_global_constant &&
              intr->intrinsic != nir_intrinsic_load_global) ||
             src != &intr->src[0])
            return true;
         /* Loading root_desc_addr (bytes 0..7) can reflect any root field. */
         if (offset < sizeof(uint64_t))
            return true;
         const uint64_t bytes =
            DIV_ROUND_UP((uint64_t)intr->def.num_components * intr->def.bit_size,
                         8);
         if (bytes > full_size - offset)
            return true;
         if (offset < end_B && offset + bytes > start_B)
            return true;
      } else {
         return true;
      }
   }
   return false;
}

bool
hk_nir_root_reads_range(const nir_shader *nir, uint64_t start_B, uint64_t end_B,
                        unsigned full_size)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_root_agx)
               continue;
            if (root_may_read_range(&intr->def, 0, start_B, end_B, full_size,
                                    0))
               return true;
         }
      }
   }
   return false;
}

static bool
hk_nir_intrinsic_refuses_wg_coalesce(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   /* Any barrier (execution or memory) fails closed: workgroup-scoped barriers
    * depend on workgroup boundaries, and subgroup barriers are excluded along
    * with all subgroup ops below. */
   case nir_intrinsic_barrier:
   /* Workgroup / local / subgroup identity builtins. load_num_workgroups reads
    * the API group_count_addr on direct dispatches and load_workgroup_size
    * folds to the compile-time API local size, so neither depends on the
    * reshaped launch geometry. */
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_local_invocation_index:
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_workgroup_index:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_num_subgroups:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_load_active_subgroup_count_agx:
   case nir_intrinsic_load_active_subgroup_invocation_agx:
   /* Subgroup and quad communication ops: coalescing a 2D/3D workgroup can
    * regroup invocations across subgroups, so reject all of them. */
   case nir_intrinsic_elect:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_last_invocation:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_ballot:
   case nir_intrinsic_ballot_relaxed:
   case nir_intrinsic_inverse_ballot:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_bit_count_inclusive:
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_shuffle_up_intel:
   case nir_intrinsic_shuffle_down_intel:
   case nir_intrinsic_rotate:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_quad_vote_all:
   case nir_intrinsic_quad_vote_any:
   case nir_intrinsic_quad_ballot_agx:
   /* Shared and tile memory. */
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
   case nir_intrinsic_image_deref_store_block_agx:
   /* Ray queries, native AGX SCS/RT ops, and cooperative matrices. */
   case nir_intrinsic_rq_initialize:
   case nir_intrinsic_rq_terminate:
   case nir_intrinsic_rq_proceed:
   case nir_intrinsic_rq_generate_intersection:
   case nir_intrinsic_rq_confirm_intersection:
   case nir_intrinsic_rq_load:
   case nir_intrinsic_load_scs_thread_index_agx:
   case nir_intrinsic_intersect_ray_gen1_agx:
   case nir_intrinsic_continue_ray_gen1_agx:
   case nir_intrinsic_intersect_ray_agx:
   case nir_intrinsic_continue_ray_agx:
   case nir_intrinsic_enter_query_agx:
   case nir_intrinsic_cmat_construct:
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
   case nir_intrinsic_cmat_muladd:
   case nir_intrinsic_cmat_muladd_agx:
   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op:
   case nir_intrinsic_cmat_bitcast:
   case nir_intrinsic_cmat_extract:
   case nir_intrinsic_cmat_insert:
   case nir_intrinsic_cmat_copy:
   case nir_intrinsic_cmat_convert:
   case nir_intrinsic_cmat_length:
   case nir_intrinsic_cmat_transpose:
   case nir_intrinsic_cmat_get_coordinate:
   /* Implicit/explicit quad derivatives. */
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      return true;
   default:
      return false;
   }
}

bool
hk_nir_can_coalesce_workgroup(const nir_shader *nir)
{
   if (!nir || nir->info.stage != MESA_SHADER_COMPUTE || nir->info.internal ||
       nir->info.workgroup_size_variable || nir->info.shared_size ||
       nir->info.derivative_group != DERIVATIVE_GROUP_NONE ||
       nir->info.ray_queries || nir->info.cs.has_cooperative_matrix ||
       nir->info.cs.image_block_size_per_thread_agx)
      return false;

   const uint16_t *wg = nir->info.workgroup_size;
   if (!wg[0] || !wg[1] || !wg[2])
      return false;

   const uint64_t threads = (uint64_t)wg[0] * wg[1] * wg[2];
   if (threads > 1024 || (threads % 32) != 0)
      return false;

   unsigned sv;
   BITSET_FOREACH_SET(sv, nir->info.system_values_read, SYSTEM_VALUE_MAX) {
      if (sv != SYSTEM_VALUE_GLOBAL_INVOCATION_ID &&
          sv != SYSTEM_VALUE_BASE_WORKGROUP_ID &&
          sv != SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID &&
          sv != SYSTEM_VALUE_NUM_WORKGROUPS &&
          sv != SYSTEM_VALUE_WORKGROUP_SIZE)
         return false;
   }

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               return false;
            if (instr->type == nir_instr_type_tex) {
               switch (nir_instr_as_tex(instr)->op) {
               case nir_texop_tex:
               case nir_texop_txb:
               case nir_texop_lod:
                  return false;
               default:
                  break;
               }
            } else if (instr->type == nir_instr_type_intrinsic &&
                       hk_nir_intrinsic_refuses_wg_coalesce(
                          nir_instr_as_intrinsic(instr))) {
               return false;
            }
         }
      }
   }

   return true;
}

static bool
hk_coalesce_partial_padded(const uint32_t grid_threads[3],
                           const uint16_t local[3],
                           uint64_t total_threads,
                           uint64_t req_wgs,
                           uint64_t *padded_out)
{
   uint64_t wgs = 1, padded = 1;
   for (unsigned i = 0; i < 3; ++i) {
      const uint64_t g = DIV_ROUND_UP((uint64_t)grid_threads[i], local[i]);
      uint64_t p_axis = 0;
      if (__builtin_mul_overflow(g, (uint64_t)local[i], &p_axis) ||
          __builtin_mul_overflow(wgs, g, &wgs) ||
          __builtin_mul_overflow(padded, p_axis, &padded))
         return false;
   }

   /* Keep core saturation and cap inactive edge lanes at <= 6.25% (1/16). */
   if (wgs < req_wgs || padded < total_threads ||
       padded - total_threads > total_threads / 16u)
      return false;

   *padded_out = padded;
   return true;
}

bool
hk_coalesce_workgroup_size_ex(const uint16_t api_local[3],
                              const uint32_t grid_threads[3],
                              const uint32_t max_wg_size[3],
                              uint32_t max_wg_threads,
                              uint32_t target_wg_threads,
                              uint64_t min_workgroups,
                              uint16_t out_local[3])
{
   out_local[0] = api_local[0];
   out_local[1] = api_local[1];
   out_local[2] = api_local[2];

   if (!api_local[0] || !api_local[1] || !api_local[2] ||
       !grid_threads[0] || !grid_threads[1] || !grid_threads[2] ||
       !max_wg_size || !max_wg_size[0] || !max_wg_size[1] || !max_wg_size[2])
      return false;

   uint64_t cur_threads =
      (uint64_t)api_local[0] * api_local[1] * api_local[2];
   const uint32_t target_max = MIN2(MIN2(target_wg_threads, 512u),
                                    max_wg_threads);
   if (cur_threads > 256 || (cur_threads % 32) != 0 ||
       target_max < cur_threads * 2)
      return false;

   if ((grid_threads[0] % api_local[0]) != 0 ||
       (grid_threads[1] % api_local[1]) != 0 ||
       (grid_threads[2] % api_local[2]) != 0)
      return false;

   uint64_t xy_threads = 0, total_threads = 0;
   if (__builtin_mul_overflow((uint64_t)grid_threads[0],
                              (uint64_t)grid_threads[1], &xy_threads) ||
       __builtin_mul_overflow(xy_threads, (uint64_t)grid_threads[2],
                              &total_threads))
      return false;

   bool changed = false;

   /* Pass 1: exact-divisor doubling (zero partial workgroups). */
   for (unsigned axis = 0; axis < 3; ++axis) {
      while (cur_threads * 2 <= target_max) {
         const uint32_t next_dim = (uint32_t)out_local[axis] * 2;
         if (next_dim > max_wg_size[axis] ||
             (grid_threads[axis] % next_dim) != 0)
            break;
         /* Keep at least 256 workgroups up to 256 threads/WG, and at least
          * 1024 workgroups when stepping up to 512 threads/WG, so wide
          * multi-cluster GPUs never lose occupancy or suffer tail imbalance. */
         const uint64_t next_threads = cur_threads * 2;
         const uint64_t req_wgs =
            next_threads > 256u ? MAX2(min_workgroups, 1024ull)
                                : MAX2(min_workgroups, 256ull);
         uint64_t min_threads = 0;
         if (__builtin_mul_overflow(next_threads, req_wgs, &min_threads) ||
             total_threads < min_threads)
            break;
         out_local[axis] = (uint16_t)next_dim;
         cur_threads = next_threads;
         changed = true;
      }
   }

   /* Pass 2: bounded partial-edge fallback for large odd/prime grids (e.g.
    * 1512x984 at 8x8 -> 16x16). AGX3 direct CDM launches take global thread
    * counts and clip trailing partial workgroups in hardware; pick the axis
    * with minimal padded invocations while keeping edge waste <= 6.25%. */
   while (cur_threads * 2 <= target_max) {
      const uint64_t next_threads = cur_threads * 2;
      const uint64_t req_wgs =
         next_threads > 256u ? MAX2(min_workgroups, 1024ull)
                             : MAX2(min_workgroups, 256ull);
      unsigned best_axis = 3;
      uint64_t best_padded = UINT64_MAX;
      for (unsigned axis = 0; axis < 3; ++axis) {
         const uint32_t next_dim = (uint32_t)out_local[axis] * 2;
         if (next_dim > max_wg_size[axis] || grid_threads[axis] < next_dim)
            continue;
         uint16_t cand[3] = {out_local[0], out_local[1], out_local[2]};
         cand[axis] = (uint16_t)next_dim;
         uint64_t padded = 0;
         if (!hk_coalesce_partial_padded(grid_threads, cand, total_threads,
                                         req_wgs, &padded))
            continue;
         if (padded < best_padded) {
            best_padded = padded;
            best_axis = axis;
         }
      }
      if (best_axis == 3)
         break;
      out_local[best_axis] = (uint16_t)(out_local[best_axis] * 2);
      cur_threads = next_threads;
      changed = true;
   }

   return changed;
}

bool
hk_coalesce_workgroup_size(const uint16_t api_local[3],
                           const uint32_t grid_threads[3],
                           uint32_t max_wg_threads,
                           uint16_t out_local[3])
{
   static const uint32_t default_max_wg_size[3] = {1024u, 1024u, 64u};
   return hk_coalesce_workgroup_size_ex(api_local, grid_threads,
                                        default_max_wg_size, max_wg_threads,
                                        256u, 256ull, out_local);
}

bool
hk_split_workgroup_size_ex(const uint16_t api_local[3],
                           const uint32_t grid_threads[3],
                           uint64_t min_workgroups,
                           uint16_t out_local[3])
{
   out_local[0] = api_local[0];
   out_local[1] = api_local[1];
   out_local[2] = api_local[2];

   if (!api_local[0] || !api_local[1] || !api_local[2] ||
       !grid_threads[0] || !grid_threads[1] || !grid_threads[2] ||
       min_workgroups <= 1ull)
      return false;

   uint64_t cur_threads =
      (uint64_t)api_local[0] * api_local[1] * api_local[2];
   if (cur_threads < 64ull || (cur_threads % 32ull) != 0)
      return false;

   if ((grid_threads[0] % api_local[0]) != 0 ||
       (grid_threads[1] % api_local[1]) != 0 ||
       (grid_threads[2] % api_local[2]) != 0)
      return false;

   uint64_t xy_threads = 0, total_threads = 0;
   if (__builtin_mul_overflow((uint64_t)grid_threads[0],
                              (uint64_t)grid_threads[1], &xy_threads) ||
       __builtin_mul_overflow(xy_threads, (uint64_t)grid_threads[2],
                              &total_threads))
      return false;

   uint64_t cur_wgs = total_threads / cur_threads;
   if (cur_wgs >= min_workgroups)
      return false;

   bool changed = false;
   while (cur_wgs < min_workgroups && cur_threads >= 64ull) {
      const unsigned axis = out_local[2] > 1 ? 2 : out_local[1] > 1 ? 1 : 0;
      if ((out_local[axis] % 2u) != 0)
         break;
      const uint16_t next_dim = (uint16_t)(out_local[axis] / 2u);
      const uint64_t next_threads = cur_threads / 2ull;
      if ((next_threads % 32ull) != 0)
         break;
      if ((grid_threads[axis] / next_dim) > 65535u)
         break;
      out_local[axis] = next_dim;
      cur_threads = next_threads;
      cur_wgs *= 2ull;
      changed = true;
   }

   return changed;
}
