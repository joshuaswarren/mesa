/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Narrow workgroup shared-memory barriers to subgroup scope when the barrier
 * provably orders only invocations within a single 32-wide AGX subgroup.
 *
 * 1. Single-subgroup workgroups (1..32 fixed invocations): workgroup scope and
 *    subgroup scope name the same invocations. Narrow workgroup execution and
 *    workgroup/device memory scopes on barriers that order only shared memory
 *    (or execution only) to subgroup scope, avoiding a threadgroup rendezvous
 *    while preserving compiler ordering. Barriers that order global/device
 *    memory or images keep their original scopes.
 *
 * 2. Larger fixed workgroups (>32 invocations): a workgroup shared-memory
 *    rendezvous can act at subgroup scope when the shared-memory traffic it
 *    orders stays inside subgroup 0 (local invocation index < 32; AGX assigns
 *    invocations to 32-wide subgroups in local index order), such as the tail
 *    steps of a tree reduction (`if (tid < off) s[tid] += s[tid + off];
 *    barrier();` for off <= 16).
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

static bool
narrow_single_subgroup_instr(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_barrier ||
       (nir_intrinsic_memory_modes(intr) & ~nir_var_mem_shared))
      return false;

   bool progress = false;
   if (nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
      nir_intrinsic_set_execution_scope(intr, SCOPE_SUBGROUP);
      progress = true;
   }
   if (nir_intrinsic_memory_scope(intr) >= SCOPE_WORKGROUP) {
      nir_intrinsic_set_memory_scope(intr, SCOPE_SUBGROUP);
      progress = true;
   }
   return progress;
}

bool
agx_nir_narrow_single_subgroup_barriers(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable)
      return false;

   const uint64_t invocations = (uint64_t)shader->info.workgroup_size[0] *
                                (uint64_t)shader->info.workgroup_size[1] *
                                (uint64_t)shader->info.workgroup_size[2];
   if (!invocations || invocations > 32)
      return false;

   return nir_shader_intrinsics_pass(shader, narrow_single_subgroup_instr,
                                     nir_metadata_all, NULL);
}

/*
 * Larger fixed workgroups: phases are the code between kept barriers. Barrier B
 * narrows when the phase before it (back to the last kept barrier) or the phase
 * after it (to the next barrier) has no shared-memory access at all -- the
 * neighbouring kept barriers then order everything B would -- or when every
 * shared access in both is guarded by a condition implying local index < 32,
 * and B does not order non-shared memory traffic across subgroups. Narrowed
 * barriers merge their phases, so a later decision sees the merged phase.
 *
 * Only runs when every barrier of the shader is a plain shared-memory workgroup
 * rendezvous (plain_shared_barrier) and all lie in one non-divergent
 * control-flow list outside any loop: phases are then program-order ranges
 * between rendezvous (conservative supersets for nested if arms). Tile-memory
 * accesses count as unguarded traffic.
 */

struct sg0 {
   uint32_t wg[3];
};

/* Express a non-overflowing linear combination of local_invocation_id.{x,y,z}
 * as coeff[0]*x + coeff[1]*y + coeff[2]*z. Handles load_local_invocation_index
 * as well as lowered 1D/2D/3D linearizations (iadd, disjoint ior, imul, ishl,
 * imadshl_agx). */
static bool
eval_local_id_coeffs(const struct sg0 *t, nir_scalar s, uint64_t coeff[3],
                     unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (s.def->bit_size < 16)
      return false;

   while (nir_scalar_is_alu(s) && (nir_scalar_alu_op(s) == nir_op_u2u32 ||
                                   nir_scalar_alu_op(s) == nir_op_u2u16 ||
                                   nir_scalar_alu_op(s) == nir_op_i2i32 ||
                                   nir_scalar_alu_op(s) == nir_op_i2i16)) {
      nir_scalar src = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      if (src.def->bit_size < 16)
         return false;
      s = src;
   }

   coeff[0] = coeff[1] = coeff[2] = 0;

   if (nir_scalar_is_intrinsic(s)) {
      nir_intrinsic_op op = nir_scalar_intrinsic_op(s);
      if (op == nir_intrinsic_load_local_invocation_index) {
         coeff[0] = 1;
         coeff[1] = t->wg[0];
         coeff[2] = (uint64_t)t->wg[0] * t->wg[1];
         return true;
      }
      if (op == nir_intrinsic_load_local_invocation_id && s.comp < 3) {
         coeff[s.comp] = 1;
         return true;
      }
      return false;
   }

   if (!depth || !nir_scalar_is_alu(s))
      return false;

   const nir_op op = nir_scalar_alu_op(s);
   switch (op) {
   case nir_op_iadd:
   case nir_op_ior: {
      uint64_t ca[3], cb[3];
      if (!eval_local_id_coeffs(t, nir_scalar_chase_alu_src(s, 0), ca,
                                depth - 1) ||
          !eval_local_id_coeffs(t, nir_scalar_chase_alu_src(s, 1), cb,
                                depth - 1))
         return false;
      if (op == nir_op_ior) {
         /* Bitwise OR equals addition only when every bit that can be set by
          * one operand is guaranteed zero in the other. */
         uint64_t max_a = 0, max_b = 0;
         unsigned tz_a = 64, tz_b = 64;
         for (unsigned i = 0; i < 3; ++i) {
            if (ca[i] && t->wg[i] > 1) {
               max_a += ca[i] * (uint64_t)(t->wg[i] - 1);
               tz_a = MIN2(tz_a, (unsigned)__builtin_ctzll(ca[i]));
            }
            if (cb[i] && t->wg[i] > 1) {
               max_b += cb[i] * (uint64_t)(t->wg[i] - 1);
               tz_b = MIN2(tz_b, (unsigned)__builtin_ctzll(cb[i]));
            }
         }
         const bool a_below_b = max_a == 0 || (tz_b < 64 && max_a < (1ull << tz_b));
         const bool b_below_a = max_b == 0 || (tz_a < 64 && max_b < (1ull << tz_a));
         if (!a_below_b && !b_below_a)
            return false;
      }
      for (unsigned i = 0; i < 3; ++i) {
         if (ca[i] > 1024 || cb[i] > 1024)
            return false;
         coeff[i] = ca[i] + cb[i];
      }
      return true;
   }
   case nir_op_imul: {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(a) == nir_scalar_is_const(b))
         return false;
      nir_scalar c = nir_scalar_is_const(a) ? a : b;
      nir_scalar v = nir_scalar_is_const(a) ? b : a;
      const uint64_t k = nir_scalar_as_uint(c);
      if (k > 1024 || !eval_local_id_coeffs(t, v, coeff, depth - 1))
         return false;
      for (unsigned i = 0; i < 3; ++i) {
         if (coeff[i] > 1024)
            return false;
         coeff[i] *= k;
      }
      return true;
   }
   case nir_op_ishl: {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      if (!nir_scalar_is_const(sh))
         return false;
      const uint64_t shift = nir_scalar_as_uint(sh);
      if (shift > 10 || !eval_local_id_coeffs(t, a, coeff, depth - 1))
         return false;
      for (unsigned i = 0; i < 3; ++i) {
         if (coeff[i] > 1024)
            return false;
         coeff[i] <<= shift;
      }
      return true;
   }
   case nir_op_imadshl_agx: {
      /* (src0 * src1) + (src2 << src3) */
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      nir_scalar s2 = nir_scalar_chase_alu_src(s, 2);
      nir_scalar s3 = nir_scalar_chase_alu_src(s, 3);
      if (!nir_scalar_is_const(s3) ||
          nir_scalar_is_const(s0) == nir_scalar_is_const(s1))
         return false;
      const uint64_t shift = nir_scalar_as_uint(s3);
      nir_scalar mul_c = nir_scalar_is_const(s0) ? s0 : s1;
      nir_scalar mul_v = nir_scalar_is_const(s0) ? s1 : s0;
      const uint64_t k = nir_scalar_as_uint(mul_c);
      uint64_t ca[3], cb[3];
      if (shift > 10 || k > 1024 ||
          !eval_local_id_coeffs(t, mul_v, ca, depth - 1) ||
          !eval_local_id_coeffs(t, s2, cb, depth - 1))
         return false;
      for (unsigned i = 0; i < 3; ++i) {
         if (ca[i] > 1024 || cb[i] > 1024)
            return false;
         coeff[i] = ca[i] * k + (cb[i] << shift);
      }
      return true;
   }
   default:
      return false;
   }
}

static bool
is_tid(const struct sg0 *t, nir_scalar s)
{
   uint64_t coeff[3];
   if (!eval_local_id_coeffs(t, s, coeff, 6))
      return false;
   if (t->wg[0] > 1 && coeff[0] != 1)
      return false;
   if (t->wg[1] > 1 && coeff[1] != t->wg[0])
      return false;
   if (t->wg[2] > 1 && coeff[2] != (uint64_t)t->wg[0] * t->wg[1])
      return false;
   return true;
}

/* Whether cond being `value` implies local index < 32. */
static bool
implies_subgroup0(const struct sg0 *t, nir_scalar cond, bool value,
                  unsigned depth)
{
   cond = nir_scalar_chase_movs(cond);
   if (!depth || !nir_scalar_is_alu(cond))
      return false;
   nir_op op = nir_scalar_alu_op(cond);
   if (op == nir_op_inot)
      return implies_subgroup0(t, nir_scalar_chase_alu_src(cond, 0), !value,
                               depth - 1);
   if (((op == nir_op_iand && value) || (op == nir_op_ior && !value)) &&
       nir_op_infos[op].num_inputs == 2) {
      /* Both hold (or both fail): either one implying it suffices. */
      return implies_subgroup0(t, nir_scalar_chase_alu_src(cond, 0), value,
                               depth - 1) ||
             implies_subgroup0(t, nir_scalar_chase_alu_src(cond, 1), value,
                               depth - 1);
   }

   /* Only binary comparisons from here (b2b1 and other unary ops have no
    * second source). */
   if (nir_op_infos[op].num_inputs != 2)
      return false;
   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   const bool tid_a = is_tid(t, a), tid_b = is_tid(t, b);
   if ((!tid_a || !nir_scalar_is_const(b)) &&
       (!tid_b || !nir_scalar_is_const(a)))
      return false;

   /* Normalize to tid OP c. */
   nir_scalar cs = tid_a ? b : a;
   const uint64_t c = nir_scalar_as_uint(cs);
   switch (op) {
   case nir_op_ieq:
      return value && c < 32;
   case nir_op_ine:
      return !value && c < 32;
   case nir_op_ilt:
      if (nir_scalar_as_int(cs) < 0)
         return false;
      FALLTHROUGH;
   case nir_op_ult:
      /* tid < c true, or c < tid false (tid <= c). The index is nonnegative
       * and <= 1023, so the signed compare matches for nonnegative c. */
      if (c > INT32_MAX)
         return false;
      return tid_a ? (value && c <= 32) : (!value && c < 32);
   case nir_op_ige:
      if (nir_scalar_as_int(cs) < 0)
         return false;
      FALLTHROUGH;
   case nir_op_uge:
      /* tid >= c false (tid < c), or c >= tid true (tid <= c). */
      if (c > INT32_MAX)
         return false;
      return tid_a ? (!value && c <= 32) : (value && c < 32);
   default:
      return false;
   }
}

static bool
guarded_subgroup0(const struct sg0 *t, nir_instr *instr)
{
   nir_cf_node *child = &instr->block->cf_node;
   for (nir_cf_node *node = child->parent; node;
        child = node, node = node->parent) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node);
      bool in_then = false;
      foreach_list_typed(nir_cf_node, n, node, &nif->then_list)
         in_then |= n == child;
      if (implies_subgroup0(t, nir_get_scalar(nif->condition.ssa, 0), in_then,
                            6))
         return true;
   }
   return false;
}

static bool
inside_loop_or_divergent_if(nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if &&
          nir_src_is_divergent(&nir_cf_node_as_if(node)->condition))
         return true;
   }
   return false;
}

/* The list (then, else or function body) a block is a direct member of. */
static const struct exec_list *
member_list(nir_block *block)
{
   nir_cf_node *parent = block->cf_node.parent;
   if (parent->type == nir_cf_node_function)
      return &nir_cf_node_as_function(parent)->body;
   if (parent->type == nir_cf_node_if) {
      nir_if *nif = nir_cf_node_as_if(parent);
      foreach_list_typed(nir_cf_node, n, node, &nif->then_list)
         if (n == &block->cf_node)
            return &nif->then_list;
      return &nif->else_list;
   }
   return NULL;
}

/* Workgroup memory: shared variables and the tile memory a compute shader can
 * reach. */
static bool
workgroup_memory_access(nir_intrinsic_instr *intr, bool *tile)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
   case nir_intrinsic_image_deref_store_block_agx:
      *tile = true;
      return true;
   default:
      *tile = false;
      return nir_is_shared_access(intr);
   }
}

/* A barrier this pass understands: a workgroup rendezvous that orders shared
 * memory and nothing else. Any other barrier in the shader -- memory-only
 * fences (whose ordering of buffers and images relies on a separate control
 * barrier's rendezvous), subgroup barriers, execution-only barriers, or
 * barriers that also order device memory -- makes the pass leave the shader
 * alone: phases are then not delimited by rendezvous alone. */
static bool
plain_shared_barrier(nir_intrinsic_instr *intr)
{
   return nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP &&
          nir_intrinsic_memory_scope(intr) >= SCOPE_WORKGROUP &&
          nir_intrinsic_memory_modes(intr) == nir_var_mem_shared &&
          nir_intrinsic_memory_semantics(intr) != 0;
}

struct phase {
   bool any;     /* a shared-memory access */
   bool outside; /* one not limited to subgroup 0 */
   bool writes;  /* a write to other memory (stores, atomics, ...) */
   bool reads;   /* a read of other memory that a write may change */
};

/* Memory other than workgroup memory: a barrier's rendezvous orders it
 * between subgroups even when the barrier's own memory semantics cover only
 * shared memory (a flag read before the barrier by one subgroup and written
 * after it by another must read the old value). Intrinsics with side effects
 * count as writes; reads that may not be reordered (from memory the shader
 * or another invocation may write) count as reads. */
static void
other_memory_access(nir_intrinsic_instr *intr, struct phase *p)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   if (!(info->flags & NIR_INTRINSIC_CAN_ELIMINATE))
      p->writes = true;
   else if (info->has_dest && !nir_intrinsic_can_reorder(intr))
      p->reads = true;
}

bool
agx_nir_narrow_subgroup0_barriers(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable)
      return false;
   const uint64_t invocations = (uint64_t)shader->info.workgroup_size[0] *
                                (uint64_t)shader->info.workgroup_size[1] *
                                (uint64_t)shader->info.workgroup_size[2];
   if (invocations <= 32)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   nir_divergence_analysis(shader);

   /* Every barrier, in program order; all in one uniform list outside loops. */
   struct util_dynarray barriers;
   util_dynarray_init(&barriers, NULL);
   const struct exec_list *list = NULL;
   bool usable = true;
   const struct sg0 t = {
      .wg = {shader->info.workgroup_size[0], shader->info.workgroup_size[1],
             shader->info.workgroup_size[2]},
   };
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            usable = false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_barrier)
            continue;
         const struct exec_list *l = member_list(block);
         if (inside_loop_or_divergent_if(block) || !l || (list && l != list) ||
             !plain_shared_barrier(intr))
            usable = false;
         list = l;
         util_dynarray_append(&barriers, intr);
      }
   }
   const unsigned count =
      util_dynarray_num_elements(&barriers, nir_intrinsic_instr *);
   if (!usable || count == 0) {
      util_dynarray_fini(&barriers);
      return false;
   }
   nir_intrinsic_instr **bar = barriers.data;

   /* The shared and non-shared memory accesses of each gap between barriers. */
   struct phase *gaps = calloc(count + 1, sizeof(*gaps));
   unsigned gap = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (gap < count && instr == &bar[gap]->instr) {
            ++gap;
            continue;
         }
         bool tile;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         if (!workgroup_memory_access(nir_instr_as_intrinsic(instr), &tile)) {
            other_memory_access(nir_instr_as_intrinsic(instr), &gaps[gap]);
            continue;
         }
         gaps[gap].any = true;
         if (tile || !guarded_subgroup0(&t, instr))
            gaps[gap].outside = true;
      }
   }

   bool progress = false;
   struct phase before = gaps[0];
   for (unsigned i = 0; i < count; ++i) {
      nir_intrinsic_instr *b = bar[i];
      const struct phase after = gaps[i + 1];
      /* Every barrier is a plain shared rendezvous here. Its rendezvous also
       * orders other memory between subgroups: keep it if one side writes
       * such memory and the other side reads or writes it. */
      const bool ordered = (before.writes && (after.writes || after.reads)) ||
                           (after.writes && before.reads);
      const bool narrow = !ordered && (!before.any || !after.any ||
                                       (!before.outside && !after.outside));
      if (narrow) {
         nir_intrinsic_set_execution_scope(b, SCOPE_SUBGROUP);
         nir_intrinsic_set_memory_scope(b, SCOPE_SUBGROUP);
         before.any |= after.any;
         before.outside |= after.outside;
         before.writes |= after.writes;
         before.reads |= after.reads;
         progress = true;
      } else {
         before = after;
      }
   }

   free(gaps);
   util_dynarray_fini(&barriers);
   /* Other functions (the preamble) are left alone. */
   nir_foreach_function_impl(other, shader) {
      if (other != impl)
         nir_no_progress(other);
   }
   return nir_progress(progress, impl, nir_metadata_all);
}
