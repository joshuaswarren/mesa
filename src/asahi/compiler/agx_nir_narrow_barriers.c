/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * A compute workgroup of at most 32 invocations is one AGX subgroup, so its
 * workgroup scope and subgroup scope name the same invocations. Narrow its
 * workgroup execution and workgroup memory scopes to subgroup scope on
 * barriers that order only shared memory (or nothing): the backend then
 * emits the SIMD-group shared-memory fence instead of the threadgroup
 * rendezvous. Barriers that order device memory or images keep their
 * scopes. Metal's compiler removes these barriers from 32-wide variants of
 * the same kernels on M5 (Geekbench's particle kernel).
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

static bool
narrow(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_barrier ||
       (nir_intrinsic_memory_modes(intr) & ~nir_var_mem_shared))
      return false;

   bool progress = false;
   if (nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
      nir_intrinsic_set_execution_scope(intr, SCOPE_SUBGROUP);
      progress = true;
   }
   if (nir_intrinsic_memory_scope(intr) == SCOPE_WORKGROUP) {
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

   const unsigned invocations = shader->info.workgroup_size[0] *
                                shader->info.workgroup_size[1] *
                                shader->info.workgroup_size[2];
   if (!invocations || invocations > 32)
      return false;

   return nir_shader_intrinsics_pass(shader, narrow, nir_metadata_all, NULL);
}

/*
 * Larger fixed workgroups: a workgroup barrier that orders only shared
 * memory can also act at subgroup scope when the shared memory traffic it
 * orders stays inside subgroup 0, the invocations with local index < 32 (AGX
 * assigns invocations to 32-wide subgroups in local index order). That is
 * the tail of a tree reduction: `if (tid < off) s[tid] += s[tid + off];
 * barrier();` with off <= 16. Metal's compiler leaves those steps without a
 * barrier on M5 (Geekbench's particle kernel at 64 and 128 invocations).
 *
 * Phases are the code between kept barriers. Barrier B narrows when the
 * phase before it (back to the last kept barrier) or the phase after it (to
 * the next barrier) has no shared-memory access at all -- the neighbouring
 * kept barriers then order everything B would -- or when every shared access
 * in both is guarded by a condition implying local index < 32. Narrowed
 * barriers merge their phases, so a later decision sees the merged phase.
 *
 * Only when every barrier of the shader is a plain shared-memory workgroup
 * rendezvous (plain_shared_barrier) and all lie in one control-flow list
 * outside any loop: the phases are then program-order ranges between
 * rendezvous (supersets, for code in the branches of an if, which is
 * conservative). Tile-memory accesses count as unguarded traffic.
 */

struct sg0 {
   nir_def *index;    /* load_local_invocation_index, or NULL */
   nir_def *id;       /* load_local_invocation_id, used as .x when 1D */
};

static bool
is_tid(const struct sg0 *t, nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   while (nir_scalar_is_alu(s) && (nir_scalar_alu_op(s) == nir_op_u2u32 ||
                                   nir_scalar_alu_op(s) == nir_op_u2u16 ||
                                   nir_scalar_alu_op(s) == nir_op_i2i32))
      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   if (!nir_scalar_is_intrinsic(s))
      return false;
   nir_intrinsic_op op = nir_scalar_intrinsic_op(s);
   if (op == nir_intrinsic_load_local_invocation_index)
      return true;
   return op == nir_intrinsic_load_local_invocation_id && t->id && s.comp == 0;
}

/* Whether cond being `value` implies local index < 32. */
static bool
implies_subgroup0(const struct sg0 *t, nir_scalar cond, bool value, unsigned depth)
{
   cond = nir_scalar_chase_movs(cond);
   if (!depth || !nir_scalar_is_alu(cond))
      return false;
   nir_op op = nir_scalar_alu_op(cond);
   if (op == nir_op_inot)
      return implies_subgroup0(t, nir_scalar_chase_alu_src(cond, 0), !value, depth - 1);
   if (((op == nir_op_iand && value) || (op == nir_op_ior && !value)) &&
       nir_op_infos[op].num_inputs == 2) {
      /* Both hold (or both fail): either one implying it suffices. */
      return implies_subgroup0(t, nir_scalar_chase_alu_src(cond, 0), value, depth - 1) ||
             implies_subgroup0(t, nir_scalar_chase_alu_src(cond, 1), value, depth - 1);
   }

   /* Only binary comparisons from here (b2b1 and other unary ops have no
    * second source). */
   if (nir_op_infos[op].num_inputs != 2)
      return false;
   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   const bool tid_a = is_tid(t, a), tid_b = is_tid(t, b);
   if ((!tid_a || !nir_scalar_is_const(b)) && (!tid_b || !nir_scalar_is_const(a)))
      return false;
   /* Normalize to tid OP c. */
   const uint64_t c = nir_scalar_as_uint(tid_a ? b : a);
   switch (op) {
   case nir_op_ieq:
      return value && c < 32;
   case nir_op_ine:
      return !value && c < 32;
   case nir_op_ult:
   case nir_op_ilt:
      /* tid < c true, or c < tid false (tid <= c). The index is
       * nonnegative, so the signed compare is the same for c < 2^31. */
      if (c > INT32_MAX)
         return false;
      return tid_a ? (value && c <= 32) : (!value && c < 32);
   case nir_op_uge:
   case nir_op_ige:
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
   for (nir_cf_node *node = child->parent; node; child = node, node = node->parent) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node);
      bool in_then = false;
      foreach_list_typed(nir_cf_node, n, node, &nif->then_list)
         in_then |= n == child;
      if (implies_subgroup0(t, nir_get_scalar(nif->condition.ssa, 0), in_then, 6))
         return true;
   }
   return false;
}

static bool
inside_loop(nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent)
      if (node->type == nir_cf_node_loop)
         return true;
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
 * reach (as agx_nir_lower_workgroup_split counts it). */
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
   bool any;       /* a shared-memory access */
   bool outside;   /* one not limited to subgroup 0 */
   bool writes;    /* a write to other memory (stores, atomics, ...) */
   bool reads;     /* a read of other memory that a write may change */
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
   const unsigned invocations = shader->info.workgroup_size[0] *
                                shader->info.workgroup_size[1] *
                                shader->info.workgroup_size[2];
   if (invocations <= 32)
      return false;
   const bool one_d = shader->info.workgroup_size[1] == 1 &&
                      shader->info.workgroup_size[2] == 1;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   /* Every barrier, in program order; all in one list outside loops. */
   struct util_dynarray barriers;
   util_dynarray_init(&barriers, NULL);
   const struct exec_list *list = NULL;
   bool usable = true;
   struct sg0 t = {0};
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            usable = false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_local_invocation_id && one_d)
            t.id = &intr->def;
         if (intr->intrinsic != nir_intrinsic_barrier)
            continue;
         const struct exec_list *l = member_list(block);
         if (inside_loop(block) || !l || (list && l != list) ||
             !plain_shared_barrier(intr))
            usable = false;
         list = l;
         util_dynarray_append(&barriers, intr);
      }
   }
   const unsigned count = util_dynarray_num_elements(&barriers, nir_intrinsic_instr *);
   if (!usable || count == 0) {
      util_dynarray_fini(&barriers);
      return false;
   }
   nir_intrinsic_instr **bar = barriers.data;

   /* The shared accesses of each gap between consecutive barriers. */
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
