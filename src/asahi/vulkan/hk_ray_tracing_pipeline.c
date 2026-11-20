/*
 * Copyright © 2021 Google
 * Copyright © 2024 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_ray_tracing_pipeline for Honeykrisp.
 *
 * Ported from lavapipe's src/gallium/frontends/lavapipe/lvp_ray_tracing_pipeline.c,
 * which is the same monolithic shape: every stage is inlined into one compute
 * megakernel and the shader binding table picks between the inlined copies with
 * an equality test on a group index.  hk's software traversal
 * (hk_nir_ray_tracing.c) is already a port of lavapipe's, so the two halves fit
 * without reinterpretation.
 *
 * On M5, two departures from lavapipe's single megakernel, both for
 * throughput:
 *
 *  - One kernel is compiled PER RAY-GENERATION GROUP rather than one for the
 *    whole pipeline.  A trace runs exactly one raygen, so inlining every raygen
 *    into one kernel only makes each dispatch pay for the union: its code
 *    size, its register count (the occupancy of every trace was that of the
 *    worst raygen) and its scratch (the stack was sized for the largest raygen
 *    frame).  hk_cmd_trace_rays.c selects the kernel on the GPU from the SBT.
 *
 *  - A stage that several groups name (a hit group with and without an
 *    any-hit shader commonly share one closest-hit) is inlined once per call
 *    site, guarded by the disjunction of the groups' handle tests, instead of
 *    once per group.
 *
 * The one real divergence is how the trace-rays record reaches the shader.
 * lavapipe binds it as an internal SSBO; hk has no internal descriptor set, so
 * the record's device address is published in the compute root descriptor table
 * (hk_root_descriptor_table::cs.trace_rays_addr) and the shader reads it with
 * the same load_root_agx + load_global construction hk_nir_lower_descriptors.c
 * uses for every other root sysval.
 */

#include "hk_ray_tracing_pipeline.h"
#include "hk_rt_inlining.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"

#include "hk_buffer.h"
#include "hk_bvh.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_nir_ray_tracing.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader_module.h"

#include "asahi/compiler/agx_nir.h"
#include "util/blob.h"
#include "util/hash_table.h"
#include "util/mesa-blake3.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_serialize.h"
#include "vk_descriptor_set_layout.h"

#include "spirv/spirv.h"

#include "util/u_debug.h"
#include "util/set.h"
#include "util/u_dynarray.h"

/* ------------------------------------------------------------------------ */
/* Stage NIR refcounting                                                     */
/* ------------------------------------------------------------------------ */

static struct hk_rt_stage_nir *
hk_rt_stage_nir_create(nir_shader *nir)
{
   struct hk_rt_stage_nir *stage = calloc(1, sizeof(*stage));
   if (!stage)
      return NULL;

   stage->ref_cnt = 1;
   stage->nir = nir;
   return stage;
}

static void
hk_rt_stage_nir_ref(struct hk_rt_stage_nir **dst, struct hk_rt_stage_nir *src)
{
   if (src)
      p_atomic_inc(&src->ref_cnt);
   *dst = src;
}

static void
hk_rt_stage_nir_unref(struct hk_rt_stage_nir *stage)
{
   if (!stage)
      return;

   if (p_atomic_dec_zero(&stage->ref_cnt)) {
      ralloc_free(stage->nir);
      free(stage);
   }
}

/* ------------------------------------------------------------------------ */
/* Reading the trace-rays record                                             */
/* ------------------------------------------------------------------------ */

/*
 * The megakernel's only driver-private input.  hk_cmd_trace_rays.c publishes a
 * VkTraceRaysIndirectCommand2KHR and puts its address here; every shader-binding
 * table pointer, stride and the launch size come out of that one record, so the
 * indirect entry points can hand the shader the application's own buffer
 * unchanged.
 */
static nir_def *
hk_load_trace_rays_record(nir_builder *b)
{
   nir_def *root = nir_load_root_agx(b);
   nir_def *addr = nir_iadd_imm(
      b, root, hk_root_descriptor_offset(cs.trace_rays_addr));

   return nir_load_global_constant(b, 1, 64, addr, .align_mul = 8,
                                   .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
hk_load_trace_ray_command_field(nir_builder *b, uint32_t command_offset,
                                uint32_t num_components, uint32_t bit_size)
{
   nir_def *record = hk_load_trace_rays_record(b);

   return nir_load_global_constant(b, num_components, bit_size,
                                   nir_iadd_imm(b, record, command_offset),
                                   .align_mul = MIN2(bit_size / 8, 8));
}

struct hk_sbt_entry {
   nir_def *value;
   nir_def *shader_record_ptr;
};

/*
 * Fetch one shader binding table record.  `command_offset` names the region
 * inside VkTraceRaysIndirectCommand2KHR; `index` is the record index inside it,
 * or NULL for the ray-generation region, which has exactly one record and no
 * stride.
 *
 * The record is laid out as the specification requires: the opaque group handle
 * first (HK_RAY_TRACING_GROUP_HANDLE_SIZE bytes), the application's shader
 * record data immediately after it.
 */
static struct hk_sbt_entry
hk_load_sbt_entry(nir_builder *b, nir_def *index, uint32_t command_offset)
{
   nir_def *addr = hk_load_trace_ray_command_field(b, command_offset, 1, 64);

   if (index) {
      /* A stride is a VkDeviceSize but is bounded by maxShaderGroupStride, so
       * the high 32 bits cannot contribute. */
      nir_def *stride = hk_load_trace_ray_command_field(
         b, hk_rt_sbt_stride_offset(command_offset), 1, 32);
      addr = nir_iadd(b, addr, nir_u2u64(b, nir_imul(b, index, stride)));
   }

   /* A record is only aligned to shaderGroupHandleAlignment. Read the aligned
    * eight bytes holding the index and shift it out, without control flow:
    * callers build around this load while other passes rewrite the shader,
    * so it must not split their blocks. The second word is still inside the
    * 32-byte handle. */
   nir_def *handle_addr = nir_iadd_imm(
      b, addr, offsetof(struct hk_ray_tracing_group_handle, index));
   nir_def *words = nir_load_global_constant(
      b, 2, 32, nir_iand_imm(b, handle_addr, ~UINT64_C(3)), .align_mul = 4);
   nir_def *shift = nir_ishl_imm(
      b, nir_iand_imm(b, nir_unpack_64_2x32_split_x(b, handle_addr), 3), 3);

   return (struct hk_sbt_entry){
      .value = nir_unpack_64_2x32_split_x(
         b, nir_ushr(b, nir_pack_64_2x32(b, words), shift)),
      .shader_record_ptr =
         nir_iadd_imm(b, addr, HK_RAY_TRACING_GROUP_HANDLE_SIZE),
   };
}

/* ------------------------------------------------------------------------ */
/* Per-stage preparation                                                     */
/* ------------------------------------------------------------------------ */

/*
 * Ray payloads (nir_var_shader_call_data) and hit attributes
 * (nir_var_ray_hit_attrib) both live in the invocation's scratch stack.  A
 * payload is addressed relative to the caller's frame, which the megakernel
 * tracks in state->shader_call_data_offset; a hit attribute always sits at the
 * current stack pointer.  Rewriting both to function_temp derefs lets
 * nir_lower_explicit_io turn them into plain scratch access, which the final
 * lowering then biases by the live stack pointer.
 */
static bool
hk_lower_ray_tracing_derefs(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   bool progress = false;

   nir_builder _b = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &_b;

   nir_def *arg_offset = nir_load_shader_call_data_offset_lvp(b);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is_one_of(deref, nir_var_shader_call_data |
                                                 nir_var_ray_hit_attrib))
            continue;

         const bool is_shader_call_data =
            nir_deref_mode_is(deref, nir_var_shader_call_data);

         deref->modes = nir_var_function_temp;
         progress = true;

         if (deref->deref_type == nir_deref_type_var) {
            b->cursor = nir_before_instr(&deref->instr);
            nir_def *offset =
               is_shader_call_data ? arg_offset : nir_imm_int(b, 0);
            nir_deref_instr *replacement = nir_build_deref_cast(
               b, offset, nir_var_function_temp, deref->var->type, 0);
            nir_def_replace(&deref->def, &replacement->def);
         }
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/*
 * The ray state a stage reads is the state of the trace that invoked it, and
 * the megakernel restores that state after the inlined call returns.  Hoisting
 * the reads to the top of the stage guarantees they happen before anything the
 * stage itself does can perturb it.
 */
static bool
hk_move_ray_tracing_intrinsic(nir_builder *b, nir_intrinsic_instr *instr,
                              void *data)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_t_max:
      nir_instr_move(nir_before_impl(b->impl), &instr->instr);
      return true;
   default:
      return false;
   }
}

/* ------------------------------------------------------------------------ */
/* The megakernel compiler                                                   */
/* ------------------------------------------------------------------------ */

struct hk_ray_traversal_state {
   nir_variable *origin;
   nir_variable *dir;
   nir_variable *inv_dir;
   nir_variable *bvh_base;
   nir_variable *current_node;
   nir_variable *stack_base;
   nir_variable *stack_ptr;
   nir_variable *stack;
   nir_variable *hit;

   nir_variable *instance_addr;
   nir_variable *sbt_offset_and_flags;
};

struct hk_rt_shader_state {
   nir_variable *bvh_base;
   nir_variable *flags;
   nir_variable *cull_mask;
   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *miss_index;
   nir_variable *origin;
   nir_variable *tmin;
   nir_variable *dir;
   nir_variable *tmax;

   nir_variable *instance_addr;
   nir_variable *primitive_addr;
   nir_variable *primitive_id;
   nir_variable *geometry_id_and_flags;
   nir_variable *hit_kind;
   nir_variable *sbt_index;

   nir_variable *shader_record_ptr;
   nir_variable *stack_ptr;
   nir_variable *shader_call_data_offset;

   nir_variable *accept;
   nir_variable *terminate;
   nir_variable *opaque;

   struct hk_ray_traversal_state traversal;
};

struct hk_rt_callable_context;
struct hk_rt_callable_part {
   nir_shader *nir;
   struct hk_rt_callable_context *context;
   unsigned pc, resume_base;
};

struct hk_rt_callable_context {
   nir_variable *pc, *called;
   struct util_dynarray parts;
   struct hash_table *entries;
};

/* A callable return record precedes the callee's scratch frame. */
#define HK_RT_CALLABLE_RETURN_SIZE 32

struct hk_rt_compiler {
   struct hk_ray_tracing_pipeline *pipeline;
   VkPipelineCreateFlags2KHR flags;
   bool position_fetch;
   bool native_agx;
   /* M5 compiles one kernel per ray-generation group and shares a stage's
    * call site between the groups naming it. Other chips keep the single
    * megakernel with one call site per group. */
   bool m5;
   bool callable_dispatch;

   struct hk_rt_shader_state state;
   nir_variable *native_query;
   /* Hit kind reported by the intersection shader of a committed generated
    * (procedural) hit; triangle hits derive theirs from the facing bit. */
   nir_variable *native_hit_kind;
   nir_variable *native_hit_attribs;

   struct hash_table **functions;
   struct hash_table *function_depths;
   struct hash_table *callable_parts;
   struct hash_table *callable_functions;
   struct hk_rt_callable_context **callable_contexts;
   /* Megakernel variables backed by a descriptor binding (hk_rt_binding_var). */
   struct set *binding_vars;
   uint32_t max_depth;

   /* Continuation mode (hk_rt_cont_split): every stage is called at depth 0
    * and only from the scheduler loop, whose pc and parts live in `cont`. */
   bool continuation;
   /* Some part traces a ray, so the scheduler needs its traversal state. */
   bool cont_traces;
   struct hk_rt_callable_context *cont;
   nir_variable *cont_stack_limit;
   uint32_t cont_stack_size, cont_max_frame;

   struct hk_rt_stack_sizes sizes;
};

static uint32_t
hk_rt_function_stack_size(struct hk_rt_compiler *compiler,
                          nir_function *function)
{
   for (unsigned depth = 0; depth <= compiler->max_depth; ++depth) {
      hash_table_foreach (compiler->functions[depth], entry) {
         if (entry->data == function) {
            const nir_shader *shader = entry->key;
            return shader->scratch_size;
         }
      }
   }
   return 0;
}

static void
hk_ray_tracing_state_init(nir_shader *nir, struct hk_rt_compiler *compiler)
{
   struct hk_rt_shader_state *state = &compiler->state;

#define VAR(field, type, name)                                                 \
   state->field = nir_variable_create(nir, nir_var_shader_temp, type, name)

   VAR(bvh_base, glsl_uint64_t_type(), "bvh_base");
   VAR(flags, glsl_uint_type(), "flags");
   VAR(cull_mask, glsl_uint_type(), "cull_mask");
   VAR(sbt_offset, glsl_uint_type(), "sbt_offset");
   VAR(sbt_stride, glsl_uint_type(), "sbt_stride");
   VAR(miss_index, glsl_uint_type(), "miss_index");
   VAR(origin, glsl_vec_type(3), "origin");
   VAR(tmin, glsl_float_type(), "tmin");
   VAR(dir, glsl_vec_type(3), "dir");
   VAR(tmax, glsl_float_type(), "tmax");

   VAR(instance_addr, glsl_uint64_t_type(), "instance_addr");
   VAR(primitive_id, glsl_uint_type(), "primitive_id");
   VAR(geometry_id_and_flags, glsl_uint_type(), "geometry_id_and_flags");
   VAR(hit_kind, glsl_uint_type(), "hit_kind");
   VAR(sbt_index, glsl_uint_type(), "sbt_index");

   VAR(shader_record_ptr, glsl_uint64_t_type(), "shader_record_ptr");
   VAR(stack_ptr, glsl_uint_type(), "stack_ptr");
   VAR(shader_call_data_offset, glsl_uint_type(), "shader_call_data_offset");

   VAR(accept, glsl_bool_type(), "accept");
   VAR(terminate, glsl_bool_type(), "terminate");
   VAR(opaque, glsl_bool_type(), "opaque");

   if (compiler->position_fetch)
      VAR(primitive_addr, glsl_uint64_t_type(), "primitive_addr");

#undef VAR
}

static void
hk_ray_traversal_state_init(nir_function_impl *impl,
                            struct hk_ray_traversal_state *state)
{
   state->origin =
      nir_local_variable_create(impl, glsl_vec_type(3), "traversal.origin");
   state->dir =
      nir_local_variable_create(impl, glsl_vec_type(3), "traversal.dir");
   state->inv_dir =
      nir_local_variable_create(impl, glsl_vec_type(3), "traversal.inv_dir");
   state->bvh_base = nir_local_variable_create(impl, glsl_uint64_t_type(),
                                               "traversal.bvh_base");
   state->current_node = nir_local_variable_create(impl, glsl_uint_type(),
                                                   "traversal.current_node");
   state->stack_base =
      nir_local_variable_create(impl, glsl_uint_type(), "traversal.stack_base");
   state->stack_ptr =
      nir_local_variable_create(impl, glsl_uint_type(), "traversal.stack_ptr");
   state->stack = nir_local_variable_create(
      impl,
      glsl_array_type(glsl_uint_type(), HK_MAX_TLAS_DEPTH + HK_MAX_BLAS_DEPTH,
                      0),
      "traversal.stack");
   state->hit =
      nir_local_variable_create(impl, glsl_bool_type(), "traversal.hit");

   state->instance_addr = nir_local_variable_create(impl, glsl_uint64_t_type(),
                                                    "traversal.instance_addr");
   state->sbt_offset_and_flags = nir_local_variable_create(
      impl, glsl_uint_type(), "traversal.sbt_offset_and_flags");
}

/*
 * Inline one application stage as a NIR function of the megakernel, creating
 * the function on first use so a stage referenced by several groups is emitted
 * once.
 */
static bool
hk_rt_has_binding(const nir_variable *var)
{
   return var->data.mode & (nir_var_image | nir_var_uniform |
                            nir_var_mem_ssbo | nir_var_mem_ubo);
}

/* Every inlined function that names a descriptor binding shares one megakernel
 * variable for it: the stages, and the parts a callable is split into at its
 * calls, access the same memory, and both NIR's access inference and
 * hk_nir_forward_image_stores reason per variable. Declarations of a binding
 * with another type keep their own variable. */
static nir_variable *
hk_rt_binding_var(struct hk_rt_compiler *compiler, const nir_variable *var)
{
   if (!hk_rt_has_binding(var))
      return NULL;
   set_foreach(compiler->binding_vars, entry) {
      nir_variable *shared = (nir_variable *)entry->key;
      if (shared->data.mode == var->data.mode &&
          shared->data.descriptor_set == var->data.descriptor_set &&
          shared->data.binding == var->data.binding &&
          shared->type == var->type &&
          shared->data.image.format == var->data.image.format) {
         /* Only the qualifiers every declaration has; whether accesses can
          * be reordered is decided for the whole megakernel. */
         shared->data.access &= var->data.access & ~ACCESS_CAN_REORDER;
         return shared;
      }
   }
   return NULL;
}

static void
hk_call_ray_tracing_stage(nir_builder *b, struct hk_rt_compiler *compiler,
                          nir_shader *stage, bool trace_call)
{
   /* The backend has no recursive call instruction. Reuse each stage only
    * within the same trace depth so the inliner sees an acyclic call graph.
    * Callable and any-hit calls stay at the invoking trace depth. Continuation
    * parts are leaves called once from the scheduler, so one copy serves
    * every depth. */
   unsigned depth = 0;
   if (!compiler->continuation) {
      struct hash_entry *caller = _mesa_hash_table_search(
         compiler->function_depths, b->impl->function);
      depth = caller ? (uintptr_t)caller->data : 0;
      depth += trace_call;
      if (depth > compiler->max_depth)
         return; /* Execution beyond the declared maximum depth is undefined. */
   }
   nir_function *function;
   struct hash_entry *entry =
      _mesa_hash_table_search(compiler->functions[depth], stage);
   if (entry) {
      function = entry->data;
   } else {
      nir_function_impl *stage_entrypoint = nir_shader_get_entrypoint(stage);
      nir_function_impl *copy = nir_function_impl_clone(b->shader,
                                                        stage_entrypoint);

      struct hash_table *var_remap = _mesa_pointer_hash_table_create(NULL);

      nir_foreach_block (block, copy) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var ||
                deref->var->data.mode == nir_var_function_temp)
               continue;

            struct hash_entry *var_entry =
               _mesa_hash_table_search(var_remap, deref->var);
            if (!var_entry) {
               nir_variable *new_var = hk_rt_binding_var(compiler, deref->var);
               if (!new_var) {
                  new_var = nir_variable_clone(deref->var, b->shader);
                  nir_shader_add_variable(b->shader, new_var);
                  if (hk_rt_has_binding(new_var)) {
                     new_var->data.access &= ~ACCESS_CAN_REORDER;
                     _mesa_set_add(compiler->binding_vars, new_var);
                  }
               }
               var_entry = _mesa_hash_table_insert(var_remap, deref->var,
                                                   new_var);
            }
            deref->var = var_entry->data;
         }
      }

      function = nir_function_create(
         b->shader, ralloc_asprintf(b->shader, "%s.depth%u",
            _mesa_shader_stage_to_string(stage->info.stage), depth));
      nir_function_set_impl(function, copy);

      ralloc_free(var_remap);

      _mesa_hash_table_insert(compiler->functions[depth], stage, function);
      _mesa_hash_table_insert(compiler->function_depths, function,
                             (void *)(uintptr_t)depth);
      struct hash_entry *part =
         _mesa_hash_table_search(compiler->callable_parts, stage);
      if (part)
         _mesa_hash_table_insert(compiler->callable_functions, function,
                                part->data);
   }

   nir_build_call(b, function, 0, NULL);

   switch (stage->info.stage) {
   case MESA_SHADER_RAYGEN:
      compiler->sizes.raygen =
         MAX2(compiler->sizes.raygen, stage->scratch_size);
      break;
   case MESA_SHADER_ANY_HIT:
      compiler->sizes.any_hit =
         MAX2(compiler->sizes.any_hit, stage->scratch_size);
      break;
   case MESA_SHADER_CLOSEST_HIT:
      compiler->sizes.closest_hit =
         MAX2(compiler->sizes.closest_hit, stage->scratch_size);
      break;
   case MESA_SHADER_MISS:
      compiler->sizes.miss = MAX2(compiler->sizes.miss, stage->scratch_size);
      break;
   case MESA_SHADER_INTERSECTION:
      compiler->sizes.intersection =
         MAX2(compiler->sizes.intersection, stage->scratch_size);
      break;
   case MESA_SHADER_CALLABLE:
      compiler->sizes.callable =
         MAX2(compiler->sizes.callable,
              stage->scratch_size + (compiler->callable_dispatch
                 ? HK_RT_CALLABLE_RETURN_SIZE : 0));
      break;
   default:
      UNREACHABLE("Invalid ray tracing stage");
   }
}

/*
 * Which stage of a shader group a call site dispatches through.
 * HK_RT_GROUP_RECURSIVE is the group's callable stage (general or closest-hit),
 * filtered by `stage_filter`; the other two are reached from inside traversal.
 */
enum hk_rt_group_stage {
   HK_RT_GROUP_RECURSIVE,
   HK_RT_GROUP_AHIT,
   HK_RT_GROUP_ISEC,
};

struct hk_rt_stage_call {
   nir_shader *stage;
   /* The any-hit shader paired with an intersection shader's group, else NULL:
    * reportIntersectionEXT lowering binds it to the intersection body. */
   nir_shader *ahit;
   nir_def *cond;
   /* Without merged call sites (cond == NULL), the group's handle test is
    * emitted where the call is. */
   nir_def *handle;
   uint32_t handle_index;
};

/*
 * Collect the distinct stage bodies the pipeline's groups select for `handle`
 * through `which`, each guarded by the disjunction of its groups' handle
 * tests.  Several groups commonly name the same stage; emitting one inlined
 * copy per call site instead of one per group keeps a closest-hit shared by N
 * hit groups from appearing N times at every traceRayEXT.  The groups are
 * indistinguishable inside the stage: the shader record pointer was already
 * taken from the SBT record itself, not from the group.  `calls` must hold
 * group_count entries.
 */
static unsigned
hk_rt_collect_stage_calls(nir_builder *b, struct hk_rt_compiler *compiler,
                          nir_def *handle, enum hk_rt_group_stage which,
                          mesa_shader_stage stage_filter,
                          struct hk_rt_stage_call *calls)
{
   unsigned count = 0;

   for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
      const struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
      const uint32_t index = which == HK_RT_GROUP_AHIT   ? group->ahit_index
                             : which == HK_RT_GROUP_ISEC ? group->isec_index
                                                         : group->recursive_index;
      if (index == VK_SHADER_UNUSED_KHR)
         continue;

      nir_shader *stage = compiler->pipeline->stages[index]->nir;
      if (which == HK_RT_GROUP_RECURSIVE && stage->info.stage != stage_filter)
         continue;

      nir_shader *ahit = NULL;
      if (which == HK_RT_GROUP_ISEC && group->ahit_index != VK_SHADER_UNUSED_KHR)
         ahit = compiler->pipeline->stages[group->ahit_index]->nir;

      if (!compiler->m5) {
         calls[count++] = (struct hk_rt_stage_call){
            .stage = stage, .ahit = ahit, .handle = handle,
            .handle_index = group->handle.index,
         };
         continue;
      }

      nir_def *test = nir_ieq_imm(b, handle, group->handle.index);

      unsigned j;
      for (j = 0; j < count; ++j) {
         if (calls[j].stage == stage && calls[j].ahit == ahit)
            break;
      }

      if (j == count) {
         calls[count++] = (struct hk_rt_stage_call){
            .stage = stage, .ahit = ahit, .cond = test,
         };
      } else {
         calls[j].cond = nir_ior(b, calls[j].cond, test);
      }
   }

   return count;
}

static nir_def *
hk_rt_stage_call_cond(nir_builder *b, const struct hk_rt_stage_call *call)
{
   return call->cond ? call->cond
                     : nir_ieq_imm(b, call->handle, call->handle_index);
}

/* Preserve invocation state across nested calls. SSA/DCE eliminates fields
 * that the caller never reads after the return; no fixed spill frame is added. */
struct hk_rt_saved_state {
   nir_variable *vars[22];
   nir_def *values[22];
   unsigned count;
};

static struct hk_rt_saved_state
hk_rt_save_state(nir_builder *b, struct hk_rt_shader_state *state)
{
   struct hk_rt_saved_state saved = {0};
   nir_variable *vars[] = {
      state->bvh_base,
      state->flags,
      state->cull_mask,
      state->sbt_offset,
      state->sbt_stride,
      state->miss_index,
      state->origin,
      state->tmin,
      state->dir,
      state->tmax,
      state->instance_addr,
      state->primitive_id,
      state->geometry_id_and_flags,
      state->hit_kind,
      state->sbt_index,
      state->shader_record_ptr,
      state->stack_ptr,
      state->shader_call_data_offset,
      state->accept,
      state->terminate,
      state->opaque,
      state->primitive_addr,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(vars); ++i) {
      if (!vars[i])
         continue;
      saved.vars[saved.count] = vars[i];
      saved.values[saved.count++] = nir_load_var(b, vars[i]);
   }
   return saved;
}

static void
hk_rt_restore_state(nir_builder *b, const struct hk_rt_saved_state *saved)
{
   for (unsigned i = 0; i < saved->count; ++i)
      nir_store_var(b, saved->vars[i], saved->values[i],
                    BITFIELD_MASK(saved->values[i]->num_components));
}

static void
hk_execute_callable_inline(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_intrinsic_instr *instr)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_rt_saved_state saved = hk_rt_save_state(b, state);

   nir_def *sbt_index = instr->src[0].ssa;
   nir_def *payload = instr->src[1].ssa;

   struct hk_sbt_entry callable_entry =
      hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_CALLABLE_ADDRESS);
   nir_store_var(b, state->shader_record_ptr,
                 callable_entry.shader_record_ptr, 0x1);

   const uint32_t stack_size =
      hk_rt_function_stack_size(compiler, b->impl->function);
   nir_def *stack_ptr = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr, nir_iadd_imm(b, stack_ptr, stack_size),
                 0x1);

   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int32_t)stack_size), 0x1);

   struct hk_rt_stage_call calls[compiler->pipeline->group_count];
   const unsigned count = hk_rt_collect_stage_calls(
      b, compiler, callable_entry.value, HK_RT_GROUP_RECURSIVE,
      MESA_SHADER_CALLABLE, calls);
   for (unsigned i = 0; i < count; i++) {
      nir_push_if(b, hk_rt_stage_call_cond(b, &calls[i]));
      hk_call_ray_tracing_stage(b, compiler, calls[i].stage, false);
      nir_pop_if(b, NULL);
   }

   hk_rt_restore_state(b, &saved);
}

static bool
hk_callable_dispatch_needed(const struct hk_ray_tracing_pipeline *pipeline)
{
   for (unsigned i = 0; i < pipeline->stage_count; ++i) {
      nir_shader *stage = pipeline->stages[i]->nir;
      if (stage->info.stage != MESA_SHADER_CALLABLE)
         continue;
      nir_foreach_block(block, nir_shader_get_entrypoint(stage)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_execute_callable)
               return true;
         }
      }
   }
   return false;
}

/* Split callable stages at shader calls. The resulting functions are leaves
 * of the C/NIR call graph; a per-invocation dispatcher selects their entry and
 * resume parts. In particular, a dynamic SBT index must not make the inliner
 * recurse through every callable (including itself). */
static struct hk_rt_callable_context *
hk_callable_context(nir_builder *b, struct hk_rt_compiler *compiler)
{
   struct hash_entry *caller = _mesa_hash_table_search(
      compiler->function_depths, b->impl->function);
   unsigned depth = caller ? (uintptr_t)caller->data : 0;
   struct hk_rt_callable_context **slot = &compiler->callable_contexts[depth];
   if (*slot)
      return *slot;

   struct hk_rt_callable_context *ctx =
      rzalloc(b->shader, struct hk_rt_callable_context);
   *slot = ctx;
   ctx->pc = nir_variable_create(b->shader, nir_var_shader_temp,
                                 glsl_uint_type(), "callable.pc");
   ctx->called = nir_variable_create(b->shader, nir_var_shader_temp,
                                     glsl_bool_type(), "callable.called");
   util_dynarray_init(&ctx->parts, ctx);
   ctx->entries = _mesa_pointer_hash_table_create(ctx);
   const nir_lower_shader_calls_options options = {
      .address_format = nir_address_format_32bit_offset,
      .stack_alignment = 16,
      .localized_loads = true,
   };
   for (unsigned i = 0; i < compiler->pipeline->stage_count; ++i) {
      nir_shader *stage = compiler->pipeline->stages[i]->nir;
      if (stage->info.stage != MESA_SHADER_CALLABLE ||
          _mesa_hash_table_search(ctx->entries, stage))
         continue;
      nir_shader *entry = nir_shader_clone(ctx, stage);
      nir_shader **resumes = NULL;
      unsigned count = 0;
      nir_lower_shader_calls(entry, &options, &resumes, &count, ctx);
      unsigned first = util_dynarray_num_elements(&ctx->parts,
                                                  struct hk_rt_callable_part *) + 1;
      _mesa_hash_table_insert(ctx->entries, stage, (void *)(uintptr_t)first);
      for (unsigned j = 0; j <= count; ++j) {
         struct hk_rt_callable_part *part =
            rzalloc(ctx, struct hk_rt_callable_part);
         *part = (struct hk_rt_callable_part){
            .nir = j ? resumes[j - 1] : entry,
            .context = ctx, .pc = first + j, .resume_base = first + 1,
         };
         util_dynarray_append(&ctx->parts, part);
         _mesa_hash_table_insert(compiler->callable_parts, part->nir, part);
      }
   }
   return ctx;
}

static nir_def *
hk_callable_pc(nir_builder *b, struct hk_rt_compiler *compiler,
                struct hk_rt_callable_context *ctx, nir_def *handle)
{
   nir_def *pc = nir_imm_int(b, 0);
   for (unsigned i = 0; i < compiler->pipeline->group_count; ++i) {
      const struct hk_ray_tracing_group *g = compiler->pipeline->groups + i;
      if (g->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      nir_shader *stage = compiler->pipeline->stages[g->recursive_index]->nir;
      struct hash_entry *entry = _mesa_hash_table_search(ctx->entries, stage);
      if (entry)
         pc = nir_bcsel(b, nir_ieq_imm(b, handle, g->handle.index),
                        nir_imm_int(b, (uintptr_t)entry->data), pc);
   }
   return pc;
}

static void
hk_callable_push(nir_builder *b, struct hk_rt_compiler *compiler,
                  struct hk_rt_callable_context *ctx, nir_def *sbt_index,
                  nir_def *payload, unsigned frame_size, unsigned resume_pc)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_sbt_entry entry =
      hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_CALLABLE_ADDRESS);
   nir_def *sp = nir_load_var(b, state->stack_ptr);
   unsigned delta = ALIGN_POT(frame_size, 16) + HK_RT_CALLABLE_RETURN_SIZE;
   nir_def *next_sp = nir_iadd_imm(b, sp, delta);
   nir_def *record = nir_iadd_imm(b, next_sp, -HK_RT_CALLABLE_RETURN_SIZE);
   nir_store_scratch(b,
      nir_vec4(b, nir_imm_int(b, resume_pc), sp,
               nir_load_var(b, state->shader_call_data_offset), nir_imm_int(b, 0)),
      record, .align_mul = 16, .write_mask = 15);
   nir_store_scratch(b, nir_load_var(b, state->shader_record_ptr),
      nir_iadd_imm(b, record, 16), .align_mul = 8, .write_mask = 1);
   nir_store_var(b, state->stack_ptr, next_sp, 1);
   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int)delta), 1);
   nir_store_var(b, state->shader_record_ptr, entry.shader_record_ptr, 1);
   nir_store_var(b, ctx->pc, hk_callable_pc(b, compiler, ctx, entry.value), 1);
   nir_store_var(b, ctx->called, nir_imm_true(b), 1);
}

static void
hk_callable_pop(nir_builder *b, struct hk_rt_compiler *compiler,
                 struct hk_rt_callable_context *ctx)
{
   struct hk_rt_shader_state *state = &compiler->state;
   nir_def *record = nir_iadd_imm(b, nir_load_var(b, state->stack_ptr),
                                 -HK_RT_CALLABLE_RETURN_SIZE);
   nir_def *saved = nir_load_scratch(b, 4, 32, record, .align_mul = 16);
   nir_def *shader_record = nir_load_scratch(b, 1, 64,
      nir_iadd_imm(b, record, 16), .align_mul = 8);
   nir_store_var(b, ctx->pc, nir_channel(b, saved, 0), 1);
   nir_store_var(b, state->stack_ptr, nir_channel(b, saved, 1), 1);
   nir_store_var(b, state->shader_call_data_offset, nir_channel(b, saved, 2), 1);
   nir_store_var(b, state->shader_record_ptr, shader_record, 1);
}

static void
hk_execute_callable(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_intrinsic_instr *instr)
{
   if (!compiler->callable_dispatch) {
      hk_execute_callable_inline(b, compiler, instr);
      return;
   }

   struct hk_rt_saved_state saved = hk_rt_save_state(b, &compiler->state);
   struct hk_rt_callable_context *ctx = hk_callable_context(b, compiler);
   hk_callable_push(b, compiler, ctx, instr->src[0].ssa, instr->src[1].ssa,
      hk_rt_function_stack_size(compiler, b->impl->function), 0);
   nir_push_loop(b);
   nir_def *pc = nir_load_var(b, ctx->pc);
   nir_break_if(b, nir_ieq_imm(b, pc, 0));
   nir_store_var(b, ctx->called, nir_imm_false(b), 1);
   util_dynarray_foreach(&ctx->parts, struct hk_rt_callable_part *, part) {
      nir_push_if(b, nir_ieq_imm(b, pc, (*part)->pc));
      hk_call_ray_tracing_stage(b, compiler, (*part)->nir, false);
      nir_pop_if(b, NULL);
   }
   nir_push_if(b, nir_inot(b, nir_load_var(b, ctx->called)));
   hk_callable_pop(b, compiler, ctx);
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
   hk_rt_restore_state(b, &saved);
}

struct hk_lower_isec_state {
   struct hk_rt_compiler *compiler;
   nir_shader *ahit;
};

static nir_deref_instr *
hk_native_hit_attrib(nir_builder *b, struct hk_rt_compiler *compiler, unsigned word)
{
   return nir_build_deref_array_imm(
      b, nir_build_deref_var(b, compiler->native_hit_attribs), word);
}

/*
 * reportIntersectionEXT inside an intersection shader.  The hit is accepted
 * only if it is inside the current [tmin, tmax] window and the ray has not
 * already been terminated; an any-hit shader in the same group then gets to
 * veto it, and ignoreIntersectionEXT rolls the state back.
 */
static void
hk_native_commit_report(nir_builder *b, struct hk_rt_compiler *compiler,
                        nir_def *distance, nir_def *hit_kind)
{
   nir_def *q = &nir_build_deref_var(b, compiler->native_query)->def;
   nir_rq_generate_intersection(b, q, distance);
   nir_store_var(b, compiler->native_hit_kind, hit_kind, 1);

   /* Later candidates overwrite the frame whether or not they are committed,
    * so the committed hit keeps its own copy of the attributes. This runs in
    * the intersection stage, whose scratch hk_lower_ray_tracing_instr() biases
    * by the stack pointer, so the offsets are frame-relative. */
   for (unsigned i = 0; i < HK_RAY_HIT_ATTRIBS_SIZE / 4; ++i) {
      nir_store_deref(b, hk_native_hit_attrib(b, compiler, i),
                      nir_load_scratch(b, 1, 32, nir_imm_int(b, 4 * i),
                                       .align_mul = 4), 0x1);
   }
}

static bool
hk_lower_isec_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   struct hk_lower_isec_state *isec_state = data;
   struct hk_rt_compiler *compiler = isec_state->compiler;
   struct hk_rt_shader_state *state = &compiler->state;

   b->cursor = nir_after_instr(&instr->instr);

   nir_def *t = instr->src[0].ssa;
   nir_def *hit_kind = instr->src[1].ssa;

   nir_def *prev_accept = nir_load_var(b, state->accept);
   nir_def *prev_tmax = nir_load_var(b, state->tmax);
   nir_def *prev_hit_kind = nir_load_var(b, state->hit_kind);

   nir_variable *commit =
      nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 0x1);

   nir_def *in_range =
      nir_iand(b, nir_fge(b, t, nir_load_var(b, state->tmin)),
               nir_fge(b, nir_load_var(b, state->tmax), t));
   nir_def *terminated = nir_iand(b, nir_load_var(b, state->terminate),
                                  nir_load_var(b, state->accept));
   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, state->accept, nir_imm_true(b), 0x1);

      nir_store_var(b, state->tmax, t, 1);
      nir_store_var(b, state->hit_kind, hit_kind, 1);

      if (isec_state->ahit) {
         nir_def *prev_terminate = nir_load_var(b, state->terminate);
         nir_store_var(b, state->terminate, nir_imm_false(b), 0x1);

         nir_push_if(b, nir_inot(b, nir_load_var(b, state->opaque)));
         {
            hk_call_ray_tracing_stage(b, compiler, isec_state->ahit, false);
         }
         nir_pop_if(b, NULL);

         nir_def *terminate = nir_load_var(b, state->terminate);
         nir_store_var(b, state->terminate,
                       nir_ior(b, terminate, prev_terminate), 0x1);

         /* Commit before terminateRay returns from the intersection stage. */
         if (compiler->native_agx) {
            nir_push_if(b, nir_load_var(b, state->accept));
            hk_native_commit_report(b, compiler, t, hit_kind);
            nir_pop_if(b, NULL);
         }
         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, nir_load_var(b, state->accept));
      {
         if (compiler->native_agx && !isec_state->ahit)
            hk_native_commit_report(b, compiler, t, hit_kind);
         nir_store_var(b, commit, nir_imm_true(b), 0x1);
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, state->accept, prev_accept, 0x1);
         nir_store_var(b, state->tmax, prev_tmax, 1);
         nir_store_var(b, state->hit_kind, prev_hit_kind, 1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&instr->def, nir_load_var(b, commit));

   return true;
}

static nir_def *
hk_hit_sbt_index(nir_builder *b, struct hk_rt_shader_state *state,
                 nir_def *geometry_id_and_flags)
{
   nir_def *geometry_id = nir_iand_imm(b, geometry_id_and_flags, 0xfffffff);

   return nir_iadd(
      b,
      nir_iadd(b, nir_load_var(b, state->sbt_offset),
               nir_iand_imm(b, nir_load_var(b, state->traversal.sbt_offset_and_flags),
                            0xffffff)),
      nir_imul(b, nir_load_var(b, state->sbt_stride), geometry_id));
}

static void
hk_handle_aabb_intersection(nir_builder *b,
                            struct hk_leaf_intersection *intersection,
                            const struct hk_ray_traversal_args *args,
                            const struct hk_ray_flags *ray_flags)
{
   struct hk_rt_compiler *compiler = args->data;
   struct hk_rt_shader_state *state = &compiler->state;

   bool has_isec = false;
   for (uint32_t i = 0; i < compiler->pipeline->group_count; i++) {
      if (compiler->pipeline->groups[i].isec_index != VK_SHADER_UNUSED_KHR) {
         has_isec = true;
         break;
      }
   }

   /* No intersection shader anywhere in the pipeline means an AABB can never
    * produce a hit.  Emitting nothing is not a silent drop: a procedural
    * geometry with no intersection shader has no defined intersection. */
   if (!has_isec)
      return;

   nir_store_var(b, state->accept, nir_imm_false(b), 0x1);
   nir_store_var(b, state->terminate, ray_flags->terminate_on_first_hit, 0x1);
   nir_store_var(b, state->opaque, intersection->opaque, 0x1);

   nir_def *prev_instance_addr = nir_load_var(b, state->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, state->primitive_id);
   nir_def *prev_geometry_id_and_flags =
      nir_load_var(b, state->geometry_id_and_flags);

   nir_store_var(b, state->instance_addr,
                 nir_load_var(b, state->traversal.instance_addr), 0x1);
   nir_store_var(b, state->primitive_id, intersection->primitive_id, 0x1);
   nir_store_var(b, state->geometry_id_and_flags,
                 intersection->geometry_id_and_flags, 0x1);

   nir_def *sbt_index =
      hk_hit_sbt_index(b, state, intersection->geometry_id_and_flags);

   struct hk_sbt_entry isec_entry =
      hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_HIT_ADDRESS);
   nir_store_var(b, state->shader_record_ptr, isec_entry.shader_record_ptr,
                 0x1);

   struct hk_rt_stage_call calls[compiler->pipeline->group_count];
   const unsigned count = hk_rt_collect_stage_calls(
      b, compiler, isec_entry.value, HK_RT_GROUP_ISEC, MESA_SHADER_NONE, calls);
   for (unsigned i = 0; i < count; i++) {
      nir_push_if(b, hk_rt_stage_call_cond(b, &calls[i]));
      hk_call_ray_tracing_stage(b, compiler, calls[i].stage, true);
      nir_pop_if(b, NULL);

      struct hk_lower_isec_state isec_state = {
         .compiler = compiler,
         .ahit = calls[i].ahit,
      };
      nir_shader_intrinsics_pass(b->shader, hk_lower_isec_intrinsic,
                                 nir_metadata_none, &isec_state);
   }

   nir_push_if(b, nir_load_var(b, state->accept));
   {
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      nir_store_var(b, state->traversal.hit, nir_imm_true(b), 0x1);

      nir_break_if(b, nir_load_var(b, state->terminate));
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, state->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, state->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, state->geometry_id_and_flags,
                    prev_geometry_id_and_flags, 0x1);
   }
   nir_pop_if(b, NULL);
}

static void
hk_handle_triangle_intersection(nir_builder *b,
                                struct hk_triangle_intersection *intersection,
                                const struct hk_ray_traversal_args *args,
                                const struct hk_ray_flags *ray_flags)
{
   struct hk_rt_compiler *compiler = args->data;
   struct hk_rt_shader_state *state = &compiler->state;

   nir_store_var(b, state->accept, nir_imm_true(b), 0x1);
   nir_store_var(b, state->terminate, ray_flags->terminate_on_first_hit, 0x1);

   /* The hit attribute lives at the top of the caller's frame, which is where
    * an any-hit shader's hitAttributeEXT deref was rewritten to point. */
   nir_def *barycentrics_offset = nir_load_var(b, state->stack_ptr);

   nir_def *prev_tmax = nir_load_var(b, state->tmax);
   nir_def *prev_instance_addr = nir_load_var(b, state->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, state->primitive_id);
   nir_def *prev_geometry_id_and_flags =
      nir_load_var(b, state->geometry_id_and_flags);
   nir_def *prev_hit_kind = nir_load_var(b, state->hit_kind);
   nir_def *prev_barycentrics =
      nir_load_scratch(b, 2, 32, barycentrics_offset);

   nir_store_var(b, state->tmax, intersection->t, 0x1);
   nir_store_var(b, state->instance_addr,
                 nir_load_var(b, state->traversal.instance_addr), 0x1);
   nir_store_var(b, state->primitive_id, intersection->base.primitive_id, 0x1);
   nir_store_var(b, state->geometry_id_and_flags,
                 intersection->base.geometry_id_and_flags, 0x1);
   nir_store_var(b, state->hit_kind,
                 nir_bcsel(b, intersection->frontface, nir_imm_int(b, 0xFE),
                           nir_imm_int(b, 0xFF)),
                 0x1);

   nir_def *prev_primitive_addr = NULL;
   if (state->primitive_addr) {
      prev_primitive_addr = nir_load_var(b, state->primitive_addr);
      nir_store_var(b, state->primitive_addr, intersection->base.node_addr,
                    0x1);
   }

   nir_store_scratch(b, intersection->barycentrics, barycentrics_offset);

   nir_def *sbt_index =
      hk_hit_sbt_index(b, state, intersection->base.geometry_id_and_flags);

   nir_push_if(b, nir_inot(b, intersection->base.opaque));
   {
      struct hk_sbt_entry ahit_entry =
         hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_HIT_ADDRESS);
      nir_store_var(b, state->shader_record_ptr,
                    ahit_entry.shader_record_ptr, 0x1);

      struct hk_rt_stage_call calls[compiler->pipeline->group_count];
      const unsigned count = hk_rt_collect_stage_calls(
         b, compiler, ahit_entry.value, HK_RT_GROUP_AHIT, MESA_SHADER_NONE,
         calls);
      for (unsigned i = 0; i < count; i++) {
         nir_push_if(b, hk_rt_stage_call_cond(b, &calls[i]));
         hk_call_ray_tracing_stage(b, compiler, calls[i].stage, true);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);

   nir_push_if(b, nir_load_var(b, state->accept));
   {
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      nir_store_var(b, state->traversal.hit, nir_imm_true(b), 0x1);

      nir_break_if(b, nir_load_var(b, state->terminate));
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, state->tmax, prev_tmax, 0x1);
      nir_store_var(b, state->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, state->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, state->geometry_id_and_flags,
                    prev_geometry_id_and_flags, 0x1);
      nir_store_var(b, state->hit_kind, prev_hit_kind, 0x1);
      nir_store_scratch(b, prev_barycentrics, barycentrics_offset);
      if (state->primitive_addr)
         nir_store_var(b, state->primitive_addr, prev_primitive_addr, 0x1);
   }
   nir_pop_if(b, NULL);
}

/* Candidate and committed state use the same native instance pointer table.
 * Any-hit must see the candidate state before application acceptance. */
static void
hk_rt_native_load_hit_state(nir_builder *b, struct hk_rt_shader_state *state,
                        nir_def *q, nir_def *as, bool committed)
{
   nir_def *instance_id = nir_rq_load(b, 1, 32, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_instance_id);
   /* Native pointer table entries contain record-relative-offset +4. */
   nir_def *table = nir_load_global(b, 1, 64, nir_iadd_imm(b, as, 128),
                                    .align_mul = 8);
   nir_def *entry = nir_load_global(b, 1, 64,
      nir_iadd(b, nir_iadd(b, as, table), nir_u2u64(b, nir_ishl_imm(b, instance_id, 4))),
      .align_mul = 8);
   nir_def *instance = nir_iadd_imm(b, nir_iadd(b, as, entry), -4);
   nir_store_var(b, state->instance_addr, instance, 1);
   nir_def *primitive = nir_rq_load(b, 1, 32, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_primitive_index);
   nir_store_var(b, state->primitive_id, primitive, 1);
   nir_def *geometry = nir_rq_load(b, 1, 32, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_geometry_index);
   nir_store_var(b, state->geometry_id_and_flags, geometry, 1);
   if (state->primitive_addr)
      nir_store_var(b, state->primitive_addr,
         agx_nir_gen1_vertex_position_address(b, instance, geometry, primitive), 1);
   nir_def *distance = nir_rq_load(b, 1, 32, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_t);
   nir_store_var(b, state->tmax, distance, 1);
   nir_def *type = nir_rq_load(b, 1, 32, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_type);
   nir_push_if(b, nir_ieq_imm(b, type, committed ? 1 : 0));
   nir_def *front = nir_rq_load(b, 1, 1, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_front_face);
   nir_store_var(b, state->hit_kind,
      nir_bcsel(b, front, nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff)), 1);
   nir_def *bary = nir_rq_load(b, 2, 32, q, .committed = committed,
      .ray_query_value = nir_ray_query_value_intersection_barycentrics);
   nir_store_scratch(b, bary, nir_load_var(b, state->stack_ptr),
                     .align_mul = 8, .write_mask = 3);
   nir_pop_if(b, NULL);
   nir_def *otw = nir_iadd(b, instance,
      nir_load_global(b, 1, 64, nir_iadd_imm(b, instance, 104), .align_mul = 8));
   nir_def *sbt = nir_load_global(b, 1, 32, nir_iadd_imm(b, otw, 48), .align_mul = 4);
   nir_store_var(b, state->traversal.sbt_offset_and_flags, sbt, 1);
   nir_store_var(b, state->sbt_index,
                 hk_hit_sbt_index(b, state, geometry), 1);
}

/* Native AS buffers use the captured Apple layout. Reuse the Vulkan query
 * lowering for traversal; software hk_bvh traversal cannot interpret them. */
static void
hk_trace_ray_native(nir_builder *b, struct hk_rt_compiler *compiler,
                nir_def *as, nir_def *flags, nir_def *mask,
                nir_def *origin, nir_def *tmin, nir_def *dir, nir_def *tmax)
{
   struct hk_rt_shader_state *state = &compiler->state;
   /* Each generated traversal drains/terminates its native queue and copies
    * the committed hit into state before invoking closest-hit or miss code.
    * Those stages therefore reuse this context for their subsequent traces.
    * Any application ray-query variables remain separate and retain the
    * lowerer's multiple-live-query guard. */
   if (!compiler->native_query) {
      compiler->native_query = nir_variable_create(b->shader, nir_var_shader_temp,
         glsl_uint64_t_type(), "rt.query");
      compiler->native_query->data.ray_query = true;
      compiler->native_hit_kind = nir_variable_create(b->shader,
         nir_var_shader_temp, glsl_uint_type(), "rt.procedural_hit_kind");
      compiler->native_hit_attribs = nir_variable_create(b->shader,
         nir_var_shader_temp,
         glsl_array_type(glsl_uint_type(), HK_RAY_HIT_ATTRIBS_SIZE / 4, 0),
         "rt.procedural_hit_attribs");
   }
   nir_variable *query = compiler->native_query;
   nir_def *q = &nir_build_deref_var(b, query)->def;
   /* SkipClosestHitShader selects a pipeline stage after traversal; it
    * does not change the native intersection query. */
   nir_def *query_flags = nir_iand_imm(b, flags,
      ~SpvRayFlagsSkipClosestHitShaderKHRMask);
   if (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
      query_flags = nir_ior_imm(b, query_flags, SpvRayFlagsSkipAABBsKHRMask);
   if (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
      query_flags = nir_ior_imm(b, query_flags, SpvRayFlagsSkipTrianglesKHRMask);
   nir_rq_initialize(b, q, as, query_flags, mask, origin, tmin, dir, tmax);
   nir_push_loop(b);
   nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, q)));
   {
      nir_def *candidate_type = nir_rq_load(b, 1, 32, q, .committed = false,
         .ray_query_value = nir_ray_query_value_intersection_type);
      nir_def *best = nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_t);
      hk_rt_native_load_hit_state(b, state, q, as, false);
      nir_store_var(b, state->terminate,
         nir_test_mask(b, flags, SpvRayFlagsTerminateOnFirstHitKHRMask), 1);
      struct hk_sbt_entry hit_entry = hk_load_sbt_entry(
         b, nir_load_var(b, state->sbt_index), HK_RT_CMD_HIT_ADDRESS);
      nir_store_var(b, state->shader_record_ptr, hit_entry.shader_record_ptr, 1);
      nir_push_if(b, nir_ieq_imm(b, candidate_type, 1));
      {
         nir_store_var(b, state->accept, nir_imm_false(b), 1);
         nir_store_var(b, state->tmax, best, 1);
         nir_store_var(b, state->opaque, nir_rq_load(b, 1, 1, q,
            .committed = false,
            .ray_query_value = nir_ray_query_value_intersection_candidate_aabb_opaque), 1);
         for (uint32_t i = 0; i < compiler->pipeline->group_count; ++i) {
            struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
            if (group->isec_index == VK_SHADER_UNUSED_KHR)
               continue;
            nir_shader *stage = compiler->pipeline->stages[group->isec_index]->nir;
            nir_push_if(b, nir_ieq_imm(b, hit_entry.value, group->handle.index));
            hk_call_ray_tracing_stage(b, compiler, stage, true);
            nir_pop_if(b, NULL);
            struct hk_lower_isec_state isec_state = {
               .compiler = compiler,
               .ahit = group->ahit_index == VK_SHADER_UNUSED_KHR ? NULL :
                  compiler->pipeline->stages[group->ahit_index]->nir,
            };
            nir_shader_intrinsics_pass(b->shader, hk_lower_isec_intrinsic,
                                       nir_metadata_none, &isec_state);
         }
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, state->accept, nir_imm_true(b), 1);
         for (uint32_t i = 0; i < compiler->pipeline->group_count; ++i) {
            struct hk_ray_tracing_group *group = compiler->pipeline->groups + i;
            if (group->ahit_index == VK_SHADER_UNUSED_KHR)
               continue;
            nir_shader *stage = compiler->pipeline->stages[group->ahit_index]->nir;
            nir_push_if(b, nir_ieq_imm(b, hit_entry.value, group->handle.index));
            hk_call_ray_tracing_stage(b, compiler, stage, true);
            nir_pop_if(b, NULL);
         }
         nir_push_if(b, nir_load_var(b, state->accept));
         nir_rq_confirm_intersection(b, q);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
      nir_push_if(b, nir_iand(b, nir_load_var(b, state->accept),
                              nir_load_var(b, state->terminate)));
      nir_rq_terminate(b, q);
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);
   /* A miss after rejected candidates keeps the original ray maximum. */
   nir_store_var(b, state->tmax, tmax, 1);
   nir_def *type = nir_rq_load(b, 1, 32, q, .committed = true,
                               .ray_query_value = nir_ray_query_value_intersection_type);
   nir_store_var(b, state->traversal.hit, nir_ine_imm(b, type, 0), 1);
   nir_push_if(b, nir_ine_imm(b, type, 0));
   hk_rt_native_load_hit_state(b, state, q, as, true);
   nir_push_if(b, nir_ieq_imm(b, type, 2));
   nir_store_var(b, state->hit_kind, nir_load_var(b, compiler->native_hit_kind), 1);
   nir_def *frame = nir_load_var(b, state->stack_ptr);
   for (unsigned i = 0; i < HK_RAY_HIT_ATTRIBS_SIZE / 4; ++i) {
      nir_store_scratch(b, nir_load_deref(b, hk_native_hit_attrib(b, compiler, i)),
                        nir_iadd_imm(b, frame, 4 * i), .align_mul = 4,
                        .write_mask = 0x1);
   }
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static void hk_rt_dispatch_hit_or_miss(nir_builder *, struct hk_rt_compiler *,
                                       nir_def *, nir_def *);
static nir_variable *hk_deref_base_var(nir_deref_instr *);

/*
 * Traverse `accel_struct` with the given ray from the current stack pointer,
 * which is the callee frame: hit attributes land at its base.  Leaves the hit
 * in the shader state and traversal.hit.
 */
static void
hk_rt_emit_traversal(nir_builder *b, struct hk_rt_compiler *compiler,
                     nir_def *accel_struct, nir_def *flags, nir_def *cull_mask,
                     nir_def *sbt_offset, nir_def *sbt_stride,
                     nir_def *miss_index, nir_def *origin, nir_def *tmin,
                     nir_def *dir, nir_def *tmax)
{
   struct hk_rt_shader_state *state = &compiler->state;

   hk_ray_traversal_state_init(b->impl, &state->traversal);

   nir_store_var(b, state->bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->flags, flags, 0x1);
   nir_store_var(b, state->cull_mask, cull_mask, 0x1);
   nir_store_var(b, state->sbt_offset, sbt_offset, 0x1);
   nir_store_var(b, state->sbt_stride, sbt_stride, 0x1);
   nir_store_var(b, state->miss_index, miss_index, 0x1);
   nir_store_var(b, state->origin, origin, 0x7);
   nir_store_var(b, state->tmin, tmin, 0x1);
   nir_store_var(b, state->dir, dir, 0x7);
   nir_store_var(b, state->tmax, tmax, 0x1);

   nir_store_var(b, state->traversal.bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->traversal.origin, origin, 0x7);
   nir_store_var(b, state->traversal.dir, dir, 0x7);
   nir_store_var(b, state->traversal.inv_dir, nir_frcp(b, dir), 0x7);
   nir_store_var(b, state->traversal.current_node,
                 nir_imm_int(b, HK_BVH_ROOT_NODE), 0x1);
   nir_store_var(b, state->traversal.stack_base, nir_imm_int(b, -1), 0x1);
   nir_store_var(b, state->traversal.stack_ptr, nir_imm_int(b, 0), 0x1);

   nir_store_var(b, state->traversal.hit, nir_imm_false(b), 0x1);

   struct hk_ray_traversal_vars vars = {
      .tmax = nir_build_deref_var(b, state->tmax),
      .origin = nir_build_deref_var(b, state->traversal.origin),
      .dir = nir_build_deref_var(b, state->traversal.dir),
      .inv_dir = nir_build_deref_var(b, state->traversal.inv_dir),
      .bvh_base = nir_build_deref_var(b, state->traversal.bvh_base),
      .current_node = nir_build_deref_var(b, state->traversal.current_node),
      .stack_base = nir_build_deref_var(b, state->traversal.stack_base),
      .stack_ptr = nir_build_deref_var(b, state->traversal.stack_ptr),
      .stack = nir_build_deref_var(b, state->traversal.stack),
      .instance_addr = nir_build_deref_var(b, state->traversal.instance_addr),
      .sbt_offset_and_flags =
         nir_build_deref_var(b, state->traversal.sbt_offset_and_flags),
   };

   struct hk_ray_traversal_args args = {
      .root_bvh_base = accel_struct,
      .flags = flags,
      .cull_mask = nir_ishl_imm(b, cull_mask, 24),
      .origin = origin,
      .tmin = tmin,
      .dir = dir,
      .vars = vars,
      .aabb_cb =
         (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
            ? NULL
            : hk_handle_aabb_intersection,
      .triangle_cb =
         (compiler->flags &
          VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
            ? NULL
            : hk_handle_triangle_intersection,
      .data = compiler,
   };

   nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
   if (compiler->native_agx)
      hk_trace_ray_native(b, compiler, accel_struct, flags, cull_mask,
                      origin, tmin, dir, tmax);
   else
      hk_build_ray_traversal(b, &args);
   nir_pop_if(b, NULL);
}

static void
hk_trace_ray(nir_builder *b, struct hk_rt_compiler *compiler,
             nir_intrinsic_instr *instr)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_rt_saved_state saved = hk_rt_save_state(b, state);

   nir_def *accel_struct = instr->src[0].ssa;
   nir_def *flags = instr->src[1].ssa;
   nir_def *cull_mask = instr->src[2].ssa;
   nir_def *sbt_offset = nir_iand_imm(b, instr->src[3].ssa, 0xF);
   nir_def *sbt_stride = nir_iand_imm(b, instr->src[4].ssa, 0xF);
   nir_def *miss_index = nir_iand_imm(b, instr->src[5].ssa, 0xFFFF);
   nir_def *origin = instr->src[6].ssa;
   nir_def *tmin = instr->src[7].ssa;
   nir_def *dir = instr->src[8].ssa;
   nir_def *tmax = instr->src[9].ssa;
   nir_def *payload = instr->src[10].ssa;

   const uint32_t stack_size =
      hk_rt_function_stack_size(compiler, b->impl->function);
   nir_def *stack_ptr = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr, nir_iadd_imm(b, stack_ptr, stack_size),
                 0x1);

   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int32_t)stack_size), 0x1);

   hk_rt_emit_traversal(b, compiler, accel_struct, flags, cull_mask,
                        sbt_offset, sbt_stride, miss_index, origin, tmin, dir,
                        tmax);

   hk_rt_dispatch_hit_or_miss(b, compiler, flags, miss_index);

   hk_rt_restore_state(b, &saved);
}

static void
hk_rt_load_wto(nir_builder *b, const struct hk_rt_compiler *compiler,
                nir_def *instance, nir_def *rows[3])
{
   if (!compiler->native_agx) {
      hk_load_wto_matrix(b, instance, NULL, rows);
      return;
   }
   nir_def *cols[4];
   for (unsigned c = 0; c < 4; ++c)
      cols[c] = nir_load_global(b, 3, 32,
         nir_iadd_imm(b, instance, 16 + c*12), .align_mul = 4);
   for (unsigned r = 0; r < 3; ++r)
      rows[r] = nir_vec4(b, nir_channel(b, cols[0], r),
         nir_channel(b, cols[1], r), nir_channel(b, cols[2], r),
         nir_channel(b, cols[3], r));
}


static void
hk_rt_dispatch_hit_or_miss(nir_builder *b, struct hk_rt_compiler *compiler,
                           nir_def *flags, nir_def *miss_index)
{
   struct hk_rt_shader_state *state = &compiler->state;

   nir_push_if(b, nir_load_var(b, state->traversal.hit));
   {
      nir_def *skip_chit =
         nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));

      struct hk_sbt_entry chit_entry = hk_load_sbt_entry(
         b, nir_load_var(b, state->sbt_index), HK_RT_CMD_HIT_ADDRESS);
      nir_store_var(b, state->shader_record_ptr,
                    chit_entry.shader_record_ptr, 0x1);

      struct hk_rt_stage_call calls[compiler->pipeline->group_count];
      const unsigned count = hk_rt_collect_stage_calls(
         b, compiler, chit_entry.value, HK_RT_GROUP_RECURSIVE,
         MESA_SHADER_CLOSEST_HIT, calls);
      for (unsigned i = 0; i < count; i++) {
         nir_push_if(b, hk_rt_stage_call_cond(b, &calls[i]));
         hk_call_ray_tracing_stage(b, compiler, calls[i].stage, true);
         nir_pop_if(b, NULL);
      }

      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct hk_sbt_entry miss_entry =
         hk_load_sbt_entry(b, miss_index, HK_RT_CMD_MISS_ADDRESS);
      nir_store_var(b, state->shader_record_ptr,
                    miss_entry.shader_record_ptr, 0x1);

      struct hk_rt_stage_call calls[compiler->pipeline->group_count];
      const unsigned count = hk_rt_collect_stage_calls(
         b, compiler, miss_entry.value, HK_RT_GROUP_RECURSIVE,
         MESA_SHADER_MISS, calls);
      for (unsigned i = 0; i < count; i++) {
         nir_push_if(b, hk_rt_stage_call_cond(b, &calls[i]));
         hk_call_ray_tracing_stage(b, compiler, calls[i].stage, true);
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);
}

enum hk_hit_object_field {
   HK_HOBJ_KIND,
   HK_HOBJ_FLAGS,
   HK_HOBJ_CULL_MASK,
   HK_HOBJ_SBT_OFFSET,
   HK_HOBJ_SBT_STRIDE,
   HK_HOBJ_SBT_INDEX,
   HK_HOBJ_ORIGIN,
   HK_HOBJ_TMIN,
   HK_HOBJ_DIR,
   HK_HOBJ_TMAX,
   HK_HOBJ_INSTANCE_ADDR,
   HK_HOBJ_PRIMITIVE_ID,
   HK_HOBJ_GEOMETRY_ID_AND_FLAGS,
   HK_HOBJ_HIT_KIND,
   HK_HOBJ_BVH_BASE,
   HK_HOBJ_PRIMITIVE_ADDR,
   HK_HOBJ_ATTRIBUTES,
   HK_HOBJ_FIELDS,
};

#define HK_HOBJ_EMPTY 0
#define HK_HOBJ_MISS  1
#define HK_HOBJ_HIT   2
#define HK_HOBJ_ATTRIBUTE_WORDS (HK_RAY_HIT_ATTRIBS_SIZE / 4)

static const struct glsl_type *
hk_hit_object_type(void)
{
   const struct glsl_struct_field fields[HK_HOBJ_FIELDS] = {
      [HK_HOBJ_KIND] = {.type = glsl_uint_type(), .name = "kind"},
      [HK_HOBJ_FLAGS] = {.type = glsl_uint_type(), .name = "flags"},
      [HK_HOBJ_CULL_MASK] = {.type = glsl_uint_type(), .name = "cull_mask"},
      [HK_HOBJ_SBT_OFFSET] = {.type = glsl_uint_type(), .name = "sbt_offset"},
      [HK_HOBJ_SBT_STRIDE] = {.type = glsl_uint_type(), .name = "sbt_stride"},
      [HK_HOBJ_SBT_INDEX] = {.type = glsl_uint_type(), .name = "sbt_index"},
      [HK_HOBJ_ORIGIN] = {.type = glsl_vec_type(3), .name = "origin"},
      [HK_HOBJ_TMIN] = {.type = glsl_float_type(), .name = "tmin"},
      [HK_HOBJ_DIR] = {.type = glsl_vec_type(3), .name = "dir"},
      [HK_HOBJ_TMAX] = {.type = glsl_float_type(), .name = "tmax"},
      [HK_HOBJ_INSTANCE_ADDR] = {.type = glsl_uint64_t_type(), .name = "instance_addr"},
      [HK_HOBJ_PRIMITIVE_ID] = {.type = glsl_uint_type(), .name = "primitive_id"},
      [HK_HOBJ_GEOMETRY_ID_AND_FLAGS] = {.type = glsl_uint_type(), .name = "geometry_id_and_flags"},
      [HK_HOBJ_HIT_KIND] = {.type = glsl_uint_type(), .name = "hit_kind"},
      [HK_HOBJ_BVH_BASE] = {.type = glsl_uint64_t_type(), .name = "bvh_base"},
      [HK_HOBJ_PRIMITIVE_ADDR] = {.type = glsl_uint64_t_type(), .name = "primitive_addr"},
      [HK_HOBJ_ATTRIBUTES] = {.type = glsl_array_type(glsl_uint_type(), HK_HOBJ_ATTRIBUTE_WORDS, 0),
                              .name = "attributes"},
   };
   return glsl_struct_type(fields, HK_HOBJ_FIELDS, "hk_hit_object", false);
}

static void
hk_retype_hit_objects(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool any = false;
   nir_foreach_variable_in_shader(var, nir) {
      if (var->data.hit_object) {
         var->type = glsl_type_wrap_in_arrays(hk_hit_object_type(), var->type);
         any = true;
      }
   }
   nir_foreach_function_temp_variable(var, impl) {
      if (var->data.hit_object) {
         var->type = glsl_type_wrap_in_arrays(hk_hit_object_type(), var->type);
         any = true;
      }
   }
   if (!any)
      return;
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         nir_variable *var = hk_deref_base_var(deref);
         if (!var || !var->data.hit_object)
            continue;
         if (deref->deref_type == nir_deref_type_var)
            deref->type = var->type;
         else if (deref->deref_type == nir_deref_type_array)
            deref->type = glsl_get_array_element(nir_deref_instr_parent(deref)->type);
      }
   }
   nir_progress(true, impl, nir_metadata_all);
}

static nir_deref_instr *
hk_hobj_field(nir_builder *b, nir_deref_instr *hobj, enum hk_hit_object_field field)
{
   return nir_build_deref_struct(b, hobj, field);
}

static nir_def *
hk_hobj_load(nir_builder *b, nir_deref_instr *hobj, enum hk_hit_object_field field)
{
   return nir_load_deref(b, hk_hobj_field(b, hobj, field));
}

static void
hk_hobj_store(nir_builder *b, nir_deref_instr *hobj, enum hk_hit_object_field field,
              nir_def *value)
{
   nir_store_deref(b, hk_hobj_field(b, hobj, field), value,
                   BITFIELD_MASK(value->num_components));
}

static nir_deref_instr *
hk_hobj_attribute(nir_builder *b, nir_deref_instr *hobj, unsigned word)
{
   return nir_build_deref_array_imm(b, hk_hobj_field(b, hobj, HK_HOBJ_ATTRIBUTES), word);
}

static void
hk_hobj_clear(nir_builder *b, nir_deref_instr *hobj, unsigned kind)
{
   hk_hobj_store(b, hobj, HK_HOBJ_KIND, nir_imm_int(b, kind));
   hk_hobj_store(b, hobj, HK_HOBJ_CULL_MASK, nir_imm_int(b, 0xFF));
   hk_hobj_store(b, hobj, HK_HOBJ_SBT_OFFSET, nir_imm_int(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_SBT_STRIDE, nir_imm_int(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_INSTANCE_ADDR, nir_imm_int64(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_PRIMITIVE_ID, nir_imm_int(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_GEOMETRY_ID_AND_FLAGS, nir_imm_int(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_HIT_KIND, nir_imm_int(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_BVH_BASE, nir_imm_int64(b, 0));
   hk_hobj_store(b, hobj, HK_HOBJ_PRIMITIVE_ADDR, nir_imm_int64(b, 0));
}

static void
hk_hobj_push_frame(nir_builder *b, struct hk_rt_compiler *compiler,
                   nir_def *payload)
{
   struct hk_rt_shader_state *state = &compiler->state;
   const uint32_t stack_size =
      hk_rt_function_stack_size(compiler, b->impl->function);
   nir_def *stack_ptr = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr, nir_iadd_imm(b, stack_ptr, stack_size), 0x1);
   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int32_t)stack_size), 0x1);
}

static void
hk_hit_object_trace(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_intrinsic_instr *intr)
{
   struct hk_rt_shader_state *state = &compiler->state;
   nir_deref_instr *hobj = nir_src_as_deref(intr->src[0]);
   struct hk_rt_saved_state saved = hk_rt_save_state(b, state);

   nir_def *accel_struct = intr->src[1].ssa;
   nir_def *flags = intr->src[2].ssa;
   nir_def *cull_mask = intr->src[3].ssa;
   nir_def *sbt_offset = nir_iand_imm(b, nir_channel(b, intr->src[4].ssa, 0), 0xF);
   nir_def *sbt_stride = nir_iand_imm(b, nir_channel(b, intr->src[4].ssa, 1), 0xF);
   nir_def *miss_index = nir_iand_imm(b, intr->src[5].ssa, 0xFFFF);
   nir_def *origin = intr->src[6].ssa;
   nir_def *tmin = intr->src[7].ssa;
   nir_def *dir = intr->src[8].ssa;
   nir_def *tmax = intr->src[9].ssa;
   nir_def *payload = intr->src[10].ssa;

   hk_hobj_push_frame(b, compiler, payload);
   hk_ray_traversal_state_init(b->impl, &state->traversal);

   nir_store_var(b, state->bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->flags, flags, 0x1);
   nir_store_var(b, state->cull_mask, cull_mask, 0x1);
   nir_store_var(b, state->sbt_offset, sbt_offset, 0x1);
   nir_store_var(b, state->sbt_stride, sbt_stride, 0x1);
   nir_store_var(b, state->miss_index, miss_index, 0x1);
   nir_store_var(b, state->origin, origin, 0x7);
   nir_store_var(b, state->tmin, tmin, 0x1);
   nir_store_var(b, state->dir, dir, 0x7);
   nir_store_var(b, state->tmax, tmax, 0x1);

   nir_store_var(b, state->traversal.bvh_base, accel_struct, 0x1);
   nir_store_var(b, state->traversal.origin, origin, 0x7);
   nir_store_var(b, state->traversal.dir, dir, 0x7);
   nir_store_var(b, state->traversal.inv_dir, nir_frcp(b, dir), 0x7);
   nir_store_var(b, state->traversal.current_node,
                 nir_imm_int(b, HK_BVH_ROOT_NODE), 0x1);
   nir_store_var(b, state->traversal.stack_base, nir_imm_int(b, -1), 0x1);
   nir_store_var(b, state->traversal.stack_ptr, nir_imm_int(b, 0), 0x1);
   nir_store_var(b, state->traversal.hit, nir_imm_false(b), 0x1);

   struct hk_ray_traversal_vars vars = {
      .tmax = nir_build_deref_var(b, state->tmax),
      .origin = nir_build_deref_var(b, state->traversal.origin),
      .dir = nir_build_deref_var(b, state->traversal.dir),
      .inv_dir = nir_build_deref_var(b, state->traversal.inv_dir),
      .bvh_base = nir_build_deref_var(b, state->traversal.bvh_base),
      .current_node = nir_build_deref_var(b, state->traversal.current_node),
      .stack_base = nir_build_deref_var(b, state->traversal.stack_base),
      .stack_ptr = nir_build_deref_var(b, state->traversal.stack_ptr),
      .stack = nir_build_deref_var(b, state->traversal.stack),
      .instance_addr = nir_build_deref_var(b, state->traversal.instance_addr),
      .sbt_offset_and_flags =
         nir_build_deref_var(b, state->traversal.sbt_offset_and_flags),
   };
   struct hk_ray_traversal_args args = {
      .root_bvh_base = accel_struct,
      .flags = flags,
      .cull_mask = nir_ishl_imm(b, cull_mask, 24),
      .origin = origin,
      .tmin = tmin,
      .dir = dir,
      .vars = vars,
      .aabb_cb =
         (compiler->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
            ? NULL
            : hk_handle_aabb_intersection,
      .triangle_cb =
         (compiler->flags &
          VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
            ? NULL
            : hk_handle_triangle_intersection,
      .data = compiler,
   };

   nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
   if (compiler->native_agx)
      hk_trace_ray_native(b, compiler, accel_struct, flags, cull_mask,
                          origin, tmin, dir, tmax);
   else
      hk_build_ray_traversal(b, &args);
   nir_pop_if(b, NULL);

   nir_def *hit = nir_load_var(b, state->traversal.hit);
   hk_hobj_store(b, hobj, HK_HOBJ_KIND,
                 nir_bcsel(b, hit, nir_imm_int(b, HK_HOBJ_HIT), nir_imm_int(b, HK_HOBJ_MISS)));
   hk_hobj_store(b, hobj, HK_HOBJ_FLAGS, flags);
   hk_hobj_store(b, hobj, HK_HOBJ_CULL_MASK, cull_mask);
   hk_hobj_store(b, hobj, HK_HOBJ_SBT_OFFSET, sbt_offset);
   hk_hobj_store(b, hobj, HK_HOBJ_SBT_STRIDE, sbt_stride);
   hk_hobj_store(b, hobj, HK_HOBJ_SBT_INDEX,
                 nir_bcsel(b, hit, nir_load_var(b, state->sbt_index), miss_index));
   hk_hobj_store(b, hobj, HK_HOBJ_ORIGIN, origin);
   hk_hobj_store(b, hobj, HK_HOBJ_TMIN, tmin);
   hk_hobj_store(b, hobj, HK_HOBJ_DIR, dir);
   hk_hobj_store(b, hobj, HK_HOBJ_TMAX, nir_load_var(b, state->tmax));
   hk_hobj_store(b, hobj, HK_HOBJ_INSTANCE_ADDR, nir_load_var(b, state->instance_addr));
   hk_hobj_store(b, hobj, HK_HOBJ_PRIMITIVE_ID, nir_load_var(b, state->primitive_id));
   hk_hobj_store(b, hobj, HK_HOBJ_GEOMETRY_ID_AND_FLAGS,
                 nir_load_var(b, state->geometry_id_and_flags));
   hk_hobj_store(b, hobj, HK_HOBJ_HIT_KIND, nir_load_var(b, state->hit_kind));
   hk_hobj_store(b, hobj, HK_HOBJ_BVH_BASE, accel_struct);
   hk_hobj_store(b, hobj, HK_HOBJ_PRIMITIVE_ADDR, state->primitive_addr ?
                 nir_load_var(b, state->primitive_addr) : nir_imm_int64(b, 0));
   nir_def *frame = nir_load_var(b, state->stack_ptr);
   for (unsigned i = 0; i < HK_HOBJ_ATTRIBUTE_WORDS; ++i) {
      nir_def *word = nir_load_scratch(b, 1, 32, nir_iadd_imm(b, frame, 4 * i),
                                       .align_mul = 4);
      nir_store_deref(b, hk_hobj_attribute(b, hobj, i), word, 0x1);
   }
   if (compiler->native_agx) {
      nir_def *q = &nir_build_deref_var(b, compiler->native_query)->def;
      nir_def *type = nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_type);
      nir_push_if(b, nir_iand(b, hit, nir_ieq_imm(b, type, 1)));
      nir_def *bary = nir_rq_load(b, 2, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_barycentrics);
      for (unsigned i = 0; i < 2; ++i)
         nir_store_deref(b, hk_hobj_attribute(b, hobj, i),
                         nir_channel(b, bary, i), 0x1);
      nir_pop_if(b, NULL);
   }

   hk_rt_restore_state(b, &saved);
}

static void
hk_hit_object_execute(nir_builder *b, struct hk_rt_compiler *compiler,
                      nir_intrinsic_instr *intr)
{
   struct hk_rt_shader_state *state = &compiler->state;
   nir_deref_instr *hobj = nir_src_as_deref(intr->src[0]);
   nir_def *payload = intr->src[1].ssa;
   struct hk_rt_saved_state saved = hk_rt_save_state(b, state);

   hk_ray_traversal_state_init(b->impl, &state->traversal);
   nir_def *kind = hk_hobj_load(b, hobj, HK_HOBJ_KIND);
   nir_push_if(b, nir_ine_imm(b, kind, HK_HOBJ_EMPTY));
   {
      hk_hobj_push_frame(b, compiler, payload);
      nir_def *flags = hk_hobj_load(b, hobj, HK_HOBJ_FLAGS);
      nir_def *sbt_index = hk_hobj_load(b, hobj, HK_HOBJ_SBT_INDEX);
      nir_store_var(b, state->bvh_base, hk_hobj_load(b, hobj, HK_HOBJ_BVH_BASE), 0x1);
      nir_store_var(b, state->flags, flags, 0x1);
      nir_store_var(b, state->cull_mask, hk_hobj_load(b, hobj, HK_HOBJ_CULL_MASK), 0x1);
      nir_store_var(b, state->sbt_offset, hk_hobj_load(b, hobj, HK_HOBJ_SBT_OFFSET), 0x1);
      nir_store_var(b, state->sbt_stride, hk_hobj_load(b, hobj, HK_HOBJ_SBT_STRIDE), 0x1);
      nir_store_var(b, state->miss_index, sbt_index, 0x1);
      nir_store_var(b, state->origin, hk_hobj_load(b, hobj, HK_HOBJ_ORIGIN), 0x7);
      nir_store_var(b, state->tmin, hk_hobj_load(b, hobj, HK_HOBJ_TMIN), 0x1);
      nir_store_var(b, state->dir, hk_hobj_load(b, hobj, HK_HOBJ_DIR), 0x7);
      nir_store_var(b, state->tmax, hk_hobj_load(b, hobj, HK_HOBJ_TMAX), 0x1);
      nir_store_var(b, state->instance_addr,
                    hk_hobj_load(b, hobj, HK_HOBJ_INSTANCE_ADDR), 0x1);
      nir_store_var(b, state->primitive_id,
                    hk_hobj_load(b, hobj, HK_HOBJ_PRIMITIVE_ID), 0x1);
      nir_store_var(b, state->geometry_id_and_flags,
                    hk_hobj_load(b, hobj, HK_HOBJ_GEOMETRY_ID_AND_FLAGS), 0x1);
      nir_store_var(b, state->hit_kind, hk_hobj_load(b, hobj, HK_HOBJ_HIT_KIND), 0x1);
      nir_store_var(b, state->sbt_index, sbt_index, 0x1);
      if (state->primitive_addr)
         nir_store_var(b, state->primitive_addr,
                       hk_hobj_load(b, hobj, HK_HOBJ_PRIMITIVE_ADDR), 0x1);
      nir_store_var(b, state->traversal.hit, nir_ieq_imm(b, kind, HK_HOBJ_HIT), 0x1);
      nir_def *frame = nir_load_var(b, state->stack_ptr);
      for (unsigned i = 0; i < HK_HOBJ_ATTRIBUTE_WORDS; ++i) {
         nir_store_scratch(b, nir_load_deref(b, hk_hobj_attribute(b, hobj, i)),
                           nir_iadd_imm(b, frame, 4 * i), .align_mul = 4,
                           .write_mask = 0x1);
      }
      hk_rt_dispatch_hit_or_miss(b, compiler, flags, sbt_index);
   }
   nir_pop_if(b, NULL);

   hk_rt_restore_state(b, &saved);
}

static void
hk_hit_object_record_miss(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_deref_instr *hobj = nir_src_as_deref(intr->src[0]);
   hk_hobj_clear(b, hobj, HK_HOBJ_MISS);
   hk_hobj_store(b, hobj, HK_HOBJ_FLAGS, intr->src[1].ssa);
   hk_hobj_store(b, hobj, HK_HOBJ_SBT_INDEX, nir_iand_imm(b, intr->src[2].ssa, 0xFFFF));
   hk_hobj_store(b, hobj, HK_HOBJ_ORIGIN, intr->src[3].ssa);
   hk_hobj_store(b, hobj, HK_HOBJ_TMIN, intr->src[4].ssa);
   hk_hobj_store(b, hobj, HK_HOBJ_DIR, intr->src[5].ssa);
   hk_hobj_store(b, hobj, HK_HOBJ_TMAX, intr->src[6].ssa);
}

static void
hk_hit_object_record_from_query(nir_builder *b, struct hk_rt_compiler *compiler,
                                nir_intrinsic_instr *intr)
{
   nir_deref_instr *hobj = nir_src_as_deref(intr->src[0]);
   nir_def *q = intr->src[1].ssa;
   nir_def *sbt_index = intr->src[2].ssa;
   nir_def *hit_kind = intr->src[3].ssa;

   nir_def *type = nir_rq_load(b, 1, 32, q, .committed = true,
                               .ray_query_value = nir_ray_query_value_intersection_type);
   nir_push_if(b, nir_ine_imm(b, type, 0));
   {
      hk_hobj_clear(b, hobj, HK_HOBJ_HIT);
      hk_hobj_store(b, hobj, HK_HOBJ_FLAGS, nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_flags));
      hk_hobj_store(b, hobj, HK_HOBJ_SBT_INDEX, sbt_index);
      hk_hobj_store(b, hobj, HK_HOBJ_ORIGIN, nir_rq_load(b, 3, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_world_ray_origin));
      hk_hobj_store(b, hobj, HK_HOBJ_TMIN, nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_tmin));
      hk_hobj_store(b, hobj, HK_HOBJ_DIR, nir_rq_load(b, 3, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_world_ray_direction));
      hk_hobj_store(b, hobj, HK_HOBJ_TMAX, nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_t));
      nir_def *instance = nir_rq_load(b, 1, 64, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_instance_address);
      nir_def *primitive = nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_primitive_index);
      nir_def *geometry = nir_rq_load(b, 1, 32, q, .committed = true,
         .ray_query_value = nir_ray_query_value_intersection_geometry_index);
      hk_hobj_store(b, hobj, HK_HOBJ_INSTANCE_ADDR, instance);
      hk_hobj_store(b, hobj, HK_HOBJ_PRIMITIVE_ID, primitive);
      hk_hobj_store(b, hobj, HK_HOBJ_GEOMETRY_ID_AND_FLAGS, geometry);
      if (compiler->native_agx)
         hk_hobj_store(b, hobj, HK_HOBJ_PRIMITIVE_ADDR,
            agx_nir_gen1_vertex_position_address(b, instance, geometry, primitive));
      /* The application's hit kind and attributes describe a generated hit.
       * A triangle already has its own. */
      nir_push_if(b, nir_ieq_imm(b, type, 1));
      {
         nir_def *front = nir_rq_load(b, 1, 1, q, .committed = true,
            .ray_query_value = nir_ray_query_value_intersection_front_face);
         hk_hobj_store(b, hobj, HK_HOBJ_HIT_KIND,
            nir_bcsel(b, front, nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff)));
         nir_def *bary = nir_rq_load(b, 2, 32, q, .committed = true,
            .ray_query_value = nir_ray_query_value_intersection_barycentrics);
         for (unsigned i = 0; i < 2; ++i)
            nir_store_deref(b, hk_hobj_attribute(b, hobj, i),
                            nir_channel(b, bary, i), 0x1);
      }
      nir_push_else(b, NULL);
      {
         hk_hobj_store(b, hobj, HK_HOBJ_HIT_KIND, hit_kind);
         nir_def *words = intr->src[4].ssa;
         for (unsigned i = 0; i < words->num_components; ++i)
            nir_store_deref(b, hk_hobj_attribute(b, hobj, i),
                            nir_channel(b, words, i), 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      hk_hobj_clear(b, hobj, HK_HOBJ_EMPTY);
   }
   nir_pop_if(b, NULL);
}

static nir_def *
hk_hit_object_record_ptr(nir_builder *b, nir_deref_instr *hobj)
{
   nir_def *kind = hk_hobj_load(b, hobj, HK_HOBJ_KIND);
   nir_def *sbt_index = hk_hobj_load(b, hobj, HK_HOBJ_SBT_INDEX);
   struct hk_sbt_entry hit = hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_HIT_ADDRESS);
   struct hk_sbt_entry miss = hk_load_sbt_entry(b, sbt_index, HK_RT_CMD_MISS_ADDRESS);
   return nir_bcsel(b, nir_ieq_imm(b, kind, HK_HOBJ_HIT),
                    hit.shader_record_ptr, miss.shader_record_ptr);
}

static nir_def *
hk_hit_object_load(nir_builder *b, struct hk_rt_compiler *compiler,
                   nir_intrinsic_instr *intr)
{
   nir_deref_instr *hobj = nir_src_as_deref(intr->src[0]);
   const unsigned column = nir_intrinsic_column(intr);
   switch (nir_intrinsic_hit_object_value(intr)) {
   case nir_hit_object_value_kind:
      return hk_hobj_load(b, hobj, HK_HOBJ_KIND);
   case nir_hit_object_value_ray_flags:
      return hk_hobj_load(b, hobj, HK_HOBJ_FLAGS);
   case nir_hit_object_value_sbt_record_index:
      return hk_hobj_load(b, hobj, HK_HOBJ_SBT_INDEX);
   case nir_hit_object_value_world_ray_origin:
      return hk_hobj_load(b, hobj, HK_HOBJ_ORIGIN);
   case nir_hit_object_value_tmin:
      return hk_hobj_load(b, hobj, HK_HOBJ_TMIN);
   case nir_hit_object_value_world_ray_direction:
      return hk_hobj_load(b, hobj, HK_HOBJ_DIR);
   case nir_hit_object_value_tmax:
      return hk_hobj_load(b, hobj, HK_HOBJ_TMAX);
   case nir_hit_object_value_primitive_index:
      return hk_hobj_load(b, hobj, HK_HOBJ_PRIMITIVE_ID);
   case nir_hit_object_value_geometry_index:
      return nir_iand_imm(b, hk_hobj_load(b, hobj, HK_HOBJ_GEOMETRY_ID_AND_FLAGS),
                          0xFFFFFFF);
   case nir_hit_object_value_instance_id: {
      nir_def *instance = hk_hobj_load(b, hobj, HK_HOBJ_INSTANCE_ADDR);
      return nir_load_global(b, 1, 32, nir_iadd_imm(b, instance,
         compiler->native_agx ? 100 : offsetof(struct hk_bvh_instance_node, instance_id)));
   }
   case nir_hit_object_value_instance_custom_index: {
      nir_def *instance = hk_hobj_load(b, hobj, HK_HOBJ_INSTANCE_ADDR);
      nir_def *custom = nir_load_global(b, 1, 32, nir_iadd_imm(b, instance,
         compiler->native_agx ? 96 :
            offsetof(struct hk_bvh_instance_node, custom_instance_and_mask)));
      return nir_iand_imm(b, custom, 0xFFFFFF);
   }
   case nir_hit_object_value_hit_kind:
      return hk_hobj_load(b, hobj, HK_HOBJ_HIT_KIND);
   case nir_hit_object_value_object_ray_origin:
   case nir_hit_object_value_object_ray_direction: {
      const bool is_origin =
         nir_intrinsic_hit_object_value(intr) == nir_hit_object_value_object_ray_origin;
      nir_def *instance = hk_hobj_load(b, hobj, HK_HOBJ_INSTANCE_ADDR);
      nir_def *wto[3];
      hk_rt_load_wto(b, compiler, instance, wto);
      return hk_mul_vec3_mat(b, hk_hobj_load(b, hobj, is_origin ? HK_HOBJ_ORIGIN : HK_HOBJ_DIR),
                             wto, is_origin);
   }
   case nir_hit_object_value_world_to_object: {
      nir_def *instance = hk_hobj_load(b, hobj, HK_HOBJ_INSTANCE_ADDR);
      nir_def *wto[3];
      hk_rt_load_wto(b, compiler, instance, wto);
      nir_def *vals[3];
      for (unsigned i = 0; i < 3; ++i)
         vals[i] = nir_channel(b, wto[i], column);
      return nir_vec(b, vals, 3);
   }
   case nir_hit_object_value_object_to_world: {
      nir_def *instance = hk_hobj_load(b, hobj, HK_HOBJ_INSTANCE_ADDR);
      if (compiler->native_agx) {
         nir_def *relative = nir_load_global(b, 1, 64,
            nir_iadd_imm(b, instance, 104), .align_mul = 8);
         return nir_load_global(b, 3, 32,
            nir_iadd_imm(b, nir_iadd(b, instance, relative), column * 12),
            .align_mul = 4);
      }
      nir_def *rows[3];
      for (unsigned r = 0; r < 3; ++r)
         rows[r] = nir_load_global(b, 4, 32,
            nir_iadd_imm(b, instance,
                         offsetof(struct hk_bvh_instance_node, otw_matrix) + r * 16));
      return nir_vec3(b, nir_channel(b, rows[0], column),
                      nir_channel(b, rows[1], column),
                      nir_channel(b, rows[2], column));
   }
   case nir_hit_object_value_shader_record_buffer_handle:
      return nir_unpack_64_2x32(b, hk_hit_object_record_ptr(b, hobj));
   case nir_hit_object_value_triangle_vertex_positions:
      return hk_load_vertex_position(b, hk_hobj_load(b, hobj, HK_HOBJ_PRIMITIVE_ADDR),
                                     column);
   default:
      UNREACHABLE("invalid hit object value");
   }
}

/* ------------------------------------------------------------------------ */
/* Continuation mode                                                         */
/* ------------------------------------------------------------------------ */

/*
 * The inlined megakernel needs one copy of a stage per trace depth, so a stage
 * with k trace sites costs k^depth copies.  Deeper pipelines split raygen,
 * closest-hit, miss and callable stages at every traceRayEXT and
 * executeCallableEXT with nir_lower_shader_calls and run the parts from one
 * scheduler loop: each part and traversal (with the intersection and any-hit
 * stages inlined into it) is emitted once, the trace depth is a runtime
 * property of the stack, and each call pushes a return record onto the
 * invocation's scratch stack:
 *
 *    [caller frame, 16-aligned][return record][callee frame ...
 *                                              ^ stack_ptr of the callee
 *
 * A part ends in a call (pc = traversal or the callee's entry, called = true)
 * or by returning, in which case the scheduler pops the record and resumes
 * the caller.  Builtins a stage reads are hoisted to its entry so a resumed
 * part takes them from its spills, not from state a callee overwrote.  The
 * instance loads they imply are valid in any closest-hit; vertex positions
 * are not (an AABB hit has none), so the record carries their address.
 */
#define HK_RT_CONT_PC_DONE       0
#define HK_RT_CONT_PC_TRAVERSE   1
#define HK_RT_CONT_PC_FIRST_PART 2

/* Keeps the stack and the backend's spills inside the 64 KiB thread-private
 * frame that static TPS offsets can address. */
#define HK_RT_CONT_MAX_STACK_SIZE (48 * 1024)

static bool
hk_rt_cont_hoist_builtin(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_shader_call_data_offset_lvp:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_cull_mask:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_t_max:
   case nir_intrinsic_load_ray_hit_kind:
   case nir_intrinsic_load_primitive_id:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_ray_instance_custom_index:
   case nir_intrinsic_load_ray_geometry_index:
   case nir_intrinsic_load_ray_world_to_object:
   case nir_intrinsic_load_ray_object_to_world:
      nir_instr_move(nir_before_impl(b->impl), &intr->instr);
      return true;
   default:
      return false;
   }
}

static bool
hk_rt_cont_splits(const nir_shader *stage)
{
   return stage->info.stage == MESA_SHADER_RAYGEN ||
          stage->info.stage == MESA_SHADER_CLOSEST_HIT ||
          stage->info.stage == MESA_SHADER_MISS ||
          stage->info.stage == MESA_SHADER_CALLABLE;
}

static bool
hk_rt_cont_has_intrinsic(nir_shader *nir, nir_intrinsic_op op)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == op)
            return true;
      }
   }
   return false;
}

/* nir_lower_shader_calls leaves scratch_size at the last call's packed
 * frame; an earlier call may spill more. */
static uint32_t
hk_rt_cont_part_frame(nir_shader *part)
{
   uint32_t frame = part->scratch_size;
   nir_foreach_block(block, nir_shader_get_entrypoint(part)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_rt_trace_ray ||
             intr->intrinsic == nir_intrinsic_rt_execute_callable)
            frame = MAX2(frame, nir_intrinsic_stack_size(intr));
      }
   }
   return frame;
}

/* Every part gets its own copy of a stage's locals.  A ray query or hit object
 * whose uses span a call would lose its state, so require each to stay inside
 * one part. */
static bool
hk_rt_cont_objects_local(nir_shader *part, unsigned pc, struct hash_table *seen)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(part)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             !(deref->var->data.ray_query || deref->var->data.hit_object))
            continue;
         struct hash_entry *entry =
            _mesa_hash_table_search(seen, deref->var->name);
         if (!entry)
            _mesa_hash_table_insert(seen, deref->var->name, (void *)(uintptr_t)pc);
         else if ((uintptr_t)entry->data != pc)
            return false;
      }
   }
   return true;
}

/*
 * Split every raygen, closest-hit, miss and callable stage into parts and size
 * the stack.  Returns false when the pipeline cannot run as continuations; it
 * then keeps the inlined megakernel.
 */
static bool
hk_rt_cont_split(struct hk_rt_compiler *compiler, void *mem_ctx)
{
   struct hk_ray_tracing_pipeline *pipeline = compiler->pipeline;

   for (unsigned i = 0; i < pipeline->stage_count; ++i) {
      if (hk_rt_cont_has_intrinsic(pipeline->stages[i]->nir,
                                   nir_intrinsic_hit_object_execute_shader))
         return false;
   }

   struct hk_rt_callable_context *ctx =
      rzalloc(mem_ctx, struct hk_rt_callable_context);
   util_dynarray_init(&ctx->parts, ctx);
   ctx->entries = _mesa_pointer_hash_table_create(ctx);
   struct hash_table *objects =
      _mesa_hash_table_create(ctx, _mesa_hash_string, _mesa_key_string_equal);

   const nir_lower_shader_calls_options options = {
      .address_format = nir_address_format_32bit_offset,
      .stack_alignment = 16,
      .localized_loads = true,
   };
   struct hk_rt_stack_sizes frames = {0};
   uint32_t traversing_frame = 0;
   bool has_callables = false;
   unsigned next_pc = HK_RT_CONT_PC_FIRST_PART, object = 0;

   for (unsigned i = 0; i < pipeline->stage_count; ++i) {
      nir_shader *stage = pipeline->stages[i]->nir;
      const uint32_t own = stage->scratch_size;

      if (stage->info.stage == MESA_SHADER_INTERSECTION) {
         frames.intersection = MAX2(frames.intersection, own);
         continue;
      } else if (stage->info.stage == MESA_SHADER_ANY_HIT) {
         frames.any_hit = MAX2(frames.any_hit, own);
         continue;
      } else if (!hk_rt_cont_splits(stage) ||
                 _mesa_hash_table_search(ctx->entries, stage)) {
         continue;
      }

      nir_shader *entry = nir_shader_clone(ctx, stage);
      NIR_PASS(_, entry, nir_shader_intrinsics_pass, hk_rt_cont_hoist_builtin,
               nir_metadata_control_flow, NULL);
      nir_foreach_function_temp_variable(var, nir_shader_get_entrypoint(entry)) {
         if (var->data.ray_query || var->data.hit_object)
            var->name = ralloc_asprintf(entry, "hk.cont.object%u", object++);
      }

      nir_shader **resumes = NULL;
      uint32_t count = 0;
      nir_lower_shader_calls(entry, &options, &resumes, &count, ctx);

      const unsigned first = next_pc;
      next_pc += count + 1;
      _mesa_hash_table_insert(ctx->entries, stage, (void *)(uintptr_t)first);

      uint32_t frame = 0;
      bool traverses = false;
      for (unsigned j = 0; j <= count; ++j) {
         nir_shader *nir = j ? resumes[j - 1] : entry;
         frame = MAX2(frame, hk_rt_cont_part_frame(nir));
         traverses |=
            hk_rt_cont_has_intrinsic(nir, nir_intrinsic_hit_object_trace_ray);
         compiler->cont_traces |=
            hk_rt_cont_has_intrinsic(nir, nir_intrinsic_rt_trace_ray);
      }
      if (traverses)
         traversing_frame = MAX2(traversing_frame, frame);
      for (unsigned j = 0; j <= count; ++j) {
         struct hk_rt_callable_part *part =
            rzalloc(ctx, struct hk_rt_callable_part);
         *part = (struct hk_rt_callable_part){
            .nir = j ? resumes[j - 1] : entry,
            .context = ctx, .pc = first + j, .resume_base = first + 1,
         };
         if (!hk_rt_cont_objects_local(part->nir, part->pc, objects))
            return false;
         /* Every part of a stage shares its frame. */
         part->nir->scratch_size = frame;
         util_dynarray_append(&ctx->parts, part);
         _mesa_hash_table_insert(compiler->callable_parts, part->nir, part);
      }

      switch (stage->info.stage) {
      case MESA_SHADER_RAYGEN:
         frames.raygen = MAX2(frames.raygen, frame);
         break;
      case MESA_SHADER_CLOSEST_HIT:
         frames.closest_hit = MAX2(frames.closest_hit, frame);
         break;
      case MESA_SHADER_MISS:
         frames.miss = MAX2(frames.miss, frame);
         break;
      default:
         frames.callable = MAX2(frames.callable, frame);
         has_callables = true;
         break;
      }
   }

   compiler->cont = ctx;
   compiler->sizes = frames;
   compiler->cont_stack_size = hk_rt_continuation_stack_size(
      &frames, compiler->max_depth, has_callables, traversing_frame,
      &compiler->cont_max_frame);
   return true;
}

struct hk_rt_cont_target {
   uint32_t handle, pc;
};

static int
hk_rt_cont_target_cmp(const void *a, const void *b)
{
   const struct hk_rt_cont_target *x = a, *y = b;
   return x->handle < y->handle ? -1 : x->handle > y->handle;
}

static void
hk_rt_cont_target_tree(nir_builder *b, nir_variable *out, nir_def *handle,
                       const struct hk_rt_cont_target *t, unsigned n)
{
   if (n <= 8) {
      nir_def *pc = nir_load_var(b, out);
      for (unsigned i = 0; i < n; ++i)
         pc = nir_bcsel(b, nir_ieq_imm(b, handle, t[i].handle),
                        nir_imm_int(b, t[i].pc), pc);
      nir_store_var(b, out, pc, 0x1);
      return;
   }
   const unsigned mid = n / 2;
   nir_push_if(b, nir_ult(b, handle, nir_imm_int(b, t[mid].handle)));
   hk_rt_cont_target_tree(b, out, handle, t, mid);
   nir_push_else(b, NULL);
   hk_rt_cont_target_tree(b, out, handle, t + mid, n - mid);
   nir_pop_if(b, NULL);
}

/*
 * Entry pc of the group whose handle is `handle`, or HK_RT_CONT_PC_DONE.  A
 * balanced search over the sorted handles: a pipeline with thousands of groups
 * costs log2(n) compares and no block holds a select per group.
 */
static nir_def *
hk_rt_cont_entry_pc(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_def *handle)
{
   const struct hk_ray_tracing_pipeline *pipeline = compiler->pipeline;
   struct hk_rt_cont_target *t =
      malloc(MAX2(pipeline->group_count, 1) * sizeof(*t));
   unsigned n = 0;

   for (unsigned i = 0; t && i < pipeline->group_count; ++i) {
      const struct hk_ray_tracing_group *g = pipeline->groups + i;
      if (g->recursive_index == VK_SHADER_UNUSED_KHR)
         continue;
      struct hash_entry *entry = _mesa_hash_table_search(
         compiler->cont->entries, pipeline->stages[g->recursive_index]->nir);
      if (entry)
         t[n++] = (struct hk_rt_cont_target){
            .handle = g->handle.index, .pc = (uintptr_t)entry->data,
         };
   }
   qsort(t, n, sizeof(*t), hk_rt_cont_target_cmp);

   nir_variable *out =
      nir_local_variable_create(b->impl, glsl_uint_type(), "rt.entry_pc");
   nir_store_var(b, out, nir_imm_int(b, HK_RT_CONT_PC_DONE), 0x1);
   hk_rt_cont_target_tree(b, out, handle, t, n);
   free(t);
   return nir_load_var(b, out);
}

/* Run the part whose pc is `cur`: a balanced search over the parts, which are
 * numbered consecutively. */
static void
hk_rt_cont_dispatch(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_def *cur, struct hk_rt_callable_part *const *parts,
                    unsigned n)
{
   if (n <= 4) {
      for (unsigned i = 0; i < n; ++i) {
         nir_push_if(b, nir_ieq_imm(b, cur, parts[i]->pc));
         hk_call_ray_tracing_stage(b, compiler, parts[i]->nir, false);
         nir_pop_if(b, NULL);
      }
      return;
   }
   const unsigned mid = n / 2;
   nir_push_if(b, nir_ult(b, cur, nir_imm_int(b, parts[mid]->pc)));
   hk_rt_cont_dispatch(b, compiler, cur, parts, mid);
   nir_push_else(b, NULL);
   hk_rt_cont_dispatch(b, compiler, cur, parts + mid, n - mid);
   nir_pop_if(b, NULL);
}

/* Push a return record above the caller's frame and enter the callee frame. */
static void
hk_rt_cont_push(nir_builder *b, struct hk_rt_compiler *compiler,
                nir_def *sp, nir_def *next_sp, unsigned resume_pc,
                nir_def *payload, unsigned delta)
{
   struct hk_rt_shader_state *state = &compiler->state;
   nir_def *record = nir_iadd_imm(b, next_sp, -HK_RT_RETURN_RECORD_SIZE);

   nir_store_scratch(b,
      nir_vec4(b, nir_imm_int(b, resume_pc), sp,
               nir_load_var(b, state->shader_call_data_offset), nir_imm_int(b, 0)),
      record, .align_mul = 16, .write_mask = 0xf);
   nir_store_scratch(b, nir_load_var(b, state->shader_record_ptr),
                     nir_iadd_imm(b, record, 16), .align_mul = 16,
                     .write_mask = 0x1);
   if (state->primitive_addr)
      nir_store_scratch(b, nir_load_var(b, state->primitive_addr),
                        nir_iadd_imm(b, record, 24), .align_mul = 8,
                        .write_mask = 0x1);
   nir_store_var(b, state->stack_ptr, next_sp, 0x1);
   nir_store_var(b, state->shader_call_data_offset,
                 nir_iadd_imm(b, payload, -(int32_t)delta), 0x1);
}

/* rt_trace_ray at the end of a part: hand the ray to the traversal state.  A
 * call that would leave the stack reservation resumes the caller instead. */
static void
hk_rt_cont_trace(nir_builder *b, struct hk_rt_compiler *compiler,
                 nir_intrinsic_instr *intr,
                 const struct hk_rt_callable_part *part)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_rt_callable_context *ctx = compiler->cont;
   const unsigned resume = part->resume_base + nir_intrinsic_call_idx(intr);
   const unsigned delta = ALIGN_POT(nir_intrinsic_stack_size(intr), 16) +
                          HK_RT_RETURN_RECORD_SIZE;

   nir_def *sp = nir_load_var(b, state->stack_ptr);
   nir_def *next_sp = nir_iadd_imm(b, sp, delta);

   nir_push_if(b, nir_uge(b, nir_load_var(b, compiler->cont_stack_limit), next_sp));
   {
      hk_rt_cont_push(b, compiler, sp, next_sp, resume, intr->src[10].ssa, delta);

      nir_store_var(b, state->bvh_base, intr->src[0].ssa, 0x1);
      nir_store_var(b, state->flags, intr->src[1].ssa, 0x1);
      nir_store_var(b, state->cull_mask, intr->src[2].ssa, 0x1);
      nir_store_var(b, state->sbt_offset,
                    nir_iand_imm(b, intr->src[3].ssa, 0xF), 0x1);
      nir_store_var(b, state->sbt_stride,
                    nir_iand_imm(b, intr->src[4].ssa, 0xF), 0x1);
      nir_store_var(b, state->miss_index,
                    nir_iand_imm(b, intr->src[5].ssa, 0xFFFF), 0x1);
      nir_store_var(b, state->origin, intr->src[6].ssa, 0x7);
      nir_store_var(b, state->tmin, intr->src[7].ssa, 0x1);
      nir_store_var(b, state->dir, intr->src[8].ssa, 0x7);
      nir_store_var(b, state->tmax, intr->src[9].ssa, 0x1);
      nir_store_var(b, ctx->pc, nir_imm_int(b, HK_RT_CONT_PC_TRAVERSE), 0x1);
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, ctx->pc, nir_imm_int(b, resume), 0x1);
   }
   nir_pop_if(b, NULL);
   nir_store_var(b, ctx->called, nir_imm_true(b), 0x1);
}

/* rt_execute_callable at the end of a part. */
static void
hk_rt_cont_execute_callable(nir_builder *b, struct hk_rt_compiler *compiler,
                            nir_intrinsic_instr *intr,
                            const struct hk_rt_callable_part *part)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_rt_callable_context *ctx = compiler->cont;
   const unsigned resume = part->resume_base + nir_intrinsic_call_idx(intr);
   const unsigned delta = ALIGN_POT(nir_intrinsic_stack_size(intr), 16) +
                          HK_RT_RETURN_RECORD_SIZE;

   struct hk_sbt_entry entry =
      hk_load_sbt_entry(b, intr->src[0].ssa, HK_RT_CMD_CALLABLE_ADDRESS);
   nir_def *target = hk_rt_cont_entry_pc(b, compiler, entry.value);
   nir_def *sp = nir_load_var(b, state->stack_ptr);
   nir_def *next_sp = nir_iadd_imm(b, sp, delta);

   nir_push_if(b, nir_iand(b, nir_ine_imm(b, target, HK_RT_CONT_PC_DONE),
                           nir_uge(b, nir_load_var(b, compiler->cont_stack_limit),
                                   next_sp)));
   {
      hk_rt_cont_push(b, compiler, sp, next_sp, resume, intr->src[1].ssa, delta);
      nir_store_var(b, state->shader_record_ptr, entry.shader_record_ptr, 0x1);
      nir_store_var(b, ctx->pc, target, 0x1);
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, ctx->pc, nir_imm_int(b, resume), 0x1);
   }
   nir_pop_if(b, NULL);
   nir_store_var(b, ctx->called, nir_imm_true(b), 0x1);
}

/*
 * The scheduler's traversal state, reached through the trace_ray marker
 * hk_rt_cont_build() leaves in it: traverse the ray the calling part stored,
 * then enter the closest-hit or miss shader the SBT names.  With no shader to
 * run, `called` stays false and the scheduler returns to the caller.
 */
static void
hk_rt_cont_traverse(nir_builder *b, struct hk_rt_compiler *compiler,
                    nir_intrinsic_instr *intr)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_rt_callable_context *ctx = compiler->cont;
   nir_def *flags = intr->src[1].ssa;
   nir_def *miss_index = intr->src[5].ssa;

   hk_rt_emit_traversal(b, compiler, intr->src[0].ssa, flags, intr->src[2].ssa,
                        intr->src[3].ssa, intr->src[4].ssa, miss_index,
                        intr->src[6].ssa, intr->src[7].ssa, intr->src[8].ssa,
                        intr->src[9].ssa);

   nir_variable *target_var = nir_local_variable_create(
      b->impl, glsl_uint_type(), "rt.target");
   nir_variable *record_var = nir_local_variable_create(
      b->impl, glsl_uint64_t_type(), "rt.target_record");
   nir_store_var(b, target_var, nir_imm_int(b, HK_RT_CONT_PC_DONE), 0x1);
   nir_store_var(b, record_var, nir_imm_int64(b, 0), 0x1);

   nir_push_if(b, nir_load_var(b, state->traversal.hit));
   {
      nir_def *skip_chit =
         nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));
      {
         struct hk_sbt_entry chit_entry = hk_load_sbt_entry(
            b, nir_load_var(b, state->sbt_index), HK_RT_CMD_HIT_ADDRESS);
         nir_store_var(b, target_var,
                       hk_rt_cont_entry_pc(b, compiler, chit_entry.value), 0x1);
         nir_store_var(b, record_var, chit_entry.shader_record_ptr, 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct hk_sbt_entry miss_entry =
         hk_load_sbt_entry(b, miss_index, HK_RT_CMD_MISS_ADDRESS);
      nir_store_var(b, target_var,
                    hk_rt_cont_entry_pc(b, compiler, miss_entry.value), 0x1);
      nir_store_var(b, record_var, miss_entry.shader_record_ptr, 0x1);
   }
   nir_pop_if(b, NULL);
   nir_def *target = nir_load_var(b, target_var);
   nir_def *record = nir_load_var(b, record_var);

   nir_push_if(b, nir_ine_imm(b, target, HK_RT_CONT_PC_DONE));
   {
      nir_store_var(b, state->shader_record_ptr, record, 0x1);
      nir_store_var(b, ctx->pc, target, 0x1);
      nir_store_var(b, ctx->called, nir_imm_true(b), 0x1);
   }
   nir_pop_if(b, NULL);
}

/*
 * Emit the scheduler loop at the megakernel's entry.  Scratch accesses here
 * are frame-relative: hk_lower_ray_tracing_instr() biases them by the stack
 * pointer like every stage's.
 */
static void
hk_rt_cont_build(nir_builder *b, struct hk_rt_compiler *compiler,
                 const struct hk_ray_tracing_group *raygen_group,
                 struct hk_sbt_entry raygen_entry)
{
   struct hk_rt_shader_state *state = &compiler->state;
   struct hk_rt_callable_context *ctx = compiler->cont;

   ctx->pc = nir_variable_create(b->shader, nir_var_shader_temp,
                                 glsl_uint_type(), "rt.pc");
   ctx->called = nir_variable_create(b->shader, nir_var_shader_temp,
                                     glsl_bool_type(), "rt.called");
   compiler->cont_stack_limit = nir_variable_create(
      b->shader, nir_var_shader_temp, glsl_uint_type(), "rt.stack_limit");

   /* Raygen's return record: popping it ends the invocation. */
   nir_store_scratch(b, nir_imm_zero(b, 4, 32), nir_imm_int(b, 0),
                     .align_mul = 16, .write_mask = 0xf);
   nir_store_scratch(b, nir_imm_int64(b, 0), nir_imm_int(b, 16),
                     .align_mul = 16, .write_mask = 0x1);
   nir_def *base = nir_load_var(b, state->stack_ptr);
   nir_store_var(b, state->stack_ptr,
                 nir_iadd_imm(b, base, HK_RT_RETURN_RECORD_SIZE), 0x1);
   nir_store_var(b, state->shader_call_data_offset, nir_imm_int(b, 0), 0x1);
   nir_store_var(b, compiler->cont_stack_limit,
                 nir_iadd_imm(b, base, compiler->cont_stack_size -
                                          compiler->cont_max_frame), 0x1);

   nir_def *pc;
   if (raygen_group) {
      nir_shader *stage = compiler->pipeline->stages[raygen_group->recursive_index]->nir;
      struct hash_entry *entry = _mesa_hash_table_search(ctx->entries, stage);
      assert(entry);
      pc = nir_bcsel(b, nir_ieq_imm(b, raygen_entry.value,
                                    raygen_group->handle.index),
                     nir_imm_int(b, (uintptr_t)entry->data),
                     nir_imm_int(b, HK_RT_CONT_PC_DONE));
   } else {
      pc = hk_rt_cont_entry_pc(b, compiler, raygen_entry.value);
   }
   nir_store_var(b, ctx->pc, pc, 0x1);

   nir_push_loop(b);
   {
      nir_def *cur = nir_load_var(b, ctx->pc);
      nir_break_if(b, nir_ieq_imm(b, cur, HK_RT_CONT_PC_DONE));
      nir_store_var(b, ctx->called, nir_imm_false(b), 0x1);

      /* Without a trace site the ray state is never written: emit no
       * traversal on undefined values. */
      if (compiler->cont_traces) {
         nir_push_if(b, nir_ieq_imm(b, cur, HK_RT_CONT_PC_TRAVERSE));
         nir_trace_ray(b, nir_load_var(b, state->bvh_base),
                       nir_load_var(b, state->flags),
                       nir_load_var(b, state->cull_mask),
                       nir_load_var(b, state->sbt_offset),
                       nir_load_var(b, state->sbt_stride),
                       nir_load_var(b, state->miss_index),
                       nir_load_var(b, state->origin),
                       nir_load_var(b, state->tmin),
                       nir_load_var(b, state->dir),
                       nir_load_var(b, state->tmax), nir_imm_int(b, 0));
         nir_push_else(b, NULL);
      }
      hk_rt_cont_dispatch(b, compiler, cur, util_dynarray_begin(&ctx->parts),
                          util_dynarray_num_elements(
                             &ctx->parts, struct hk_rt_callable_part *));
      if (compiler->cont_traces)
         nir_pop_if(b, NULL);

      /* The part returned, or the trace had no shader to run. */
      nir_push_if(b, nir_inot(b, nir_load_var(b, ctx->called)));
      {
         nir_def *record = nir_load_scratch(
            b, 4, 32, nir_imm_int(b, -HK_RT_RETURN_RECORD_SIZE), .align_mul = 16);
         nir_def *shader_record = nir_load_scratch(
            b, 1, 64, nir_imm_int(b, -HK_RT_RETURN_RECORD_SIZE + 16),
            .align_mul = 16);
         nir_def *primitive = state->primitive_addr ? nir_load_scratch(
            b, 1, 64, nir_imm_int(b, -HK_RT_RETURN_RECORD_SIZE + 24),
            .align_mul = 8) : NULL;
         nir_store_var(b, ctx->pc, nir_channel(b, record, 0), 0x1);
         nir_store_var(b, state->stack_ptr, nir_channel(b, record, 1), 0x1);
         nir_store_var(b, state->shader_call_data_offset,
                       nir_channel(b, record, 2), 0x1);
         nir_store_var(b, state->shader_record_ptr, shader_record, 0x1);
         if (primitive)
            nir_store_var(b, state->primitive_addr, primitive, 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);
}

static bool
hk_lower_ray_tracing_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct hk_rt_compiler *compiler = data;
   struct hk_rt_shader_state *state = &compiler->state;

   if (instr->type == nir_instr_type_jump) {
      nir_jump_instr *jump = nir_instr_as_jump(instr);
      if (jump->type == nir_jump_halt) {
         /* An inlined stage's terminateInvocation must end the stage, not the
          * megakernel: the caller still has state to restore. */
         jump->type = nir_jump_return;
         return true;
      }
      return false;
   } else if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_def *def = NULL;

   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   /* Ray tracing instructions */
   case nir_intrinsic_execute_callable:
      assert(!compiler->continuation);
      hk_execute_callable(b, compiler, intr);
      break;
   case nir_intrinsic_rt_execute_callable: {
      struct hash_entry *entry = _mesa_hash_table_search(
         compiler->callable_functions, b->impl->function);
      assert(entry);
      struct hk_rt_callable_part *part = entry->data;
      if (compiler->continuation) {
         hk_rt_cont_execute_callable(b, compiler, intr, part);
         break;
      }
      hk_callable_push(b, compiler, part->context,
         intr->src[0].ssa, intr->src[1].ssa, nir_intrinsic_stack_size(intr),
         part->resume_base + nir_intrinsic_call_idx(intr));
      break;
   }
   case nir_intrinsic_rt_trace_ray: {
      struct hash_entry *entry = _mesa_hash_table_search(
         compiler->callable_functions, b->impl->function);
      assert(entry);
      struct hk_rt_callable_part *part = entry->data;
      if (compiler->continuation) {
         hk_rt_cont_trace(b, compiler, intr, part);
         break;
      }
      hk_trace_ray(b, compiler, intr);
      nir_store_var(b, part->context->pc,
         nir_imm_int(b, part->resume_base + nir_intrinsic_call_idx(intr)), 1);
      nir_store_var(b, part->context->called, nir_imm_true(b), 1);
      break;
   }
   case nir_intrinsic_rt_resume:
      break;
   case nir_intrinsic_trace_ray:
      /* In continuation mode, only the scheduler's traversal marker. */
      if (compiler->continuation)
         hk_rt_cont_traverse(b, compiler, intr);
      else
         hk_trace_ray(b, compiler, intr);
      break;
   case nir_intrinsic_hit_object_trace_ray:
      hk_hit_object_trace(b, compiler, intr);
      break;
   case nir_intrinsic_hit_object_execute_shader:
      hk_hit_object_execute(b, compiler, intr);
      break;
   case nir_intrinsic_hit_object_record_miss:
      hk_hit_object_record_miss(b, intr);
      break;
   case nir_intrinsic_hit_object_record_empty:
      hk_hobj_clear(b, nir_src_as_deref(intr->src[0]), HK_HOBJ_EMPTY);
      break;
   case nir_intrinsic_hit_object_record_from_query:
      hk_hit_object_record_from_query(b, compiler, intr);
      break;
   case nir_intrinsic_hit_object_set_sbt_record_index:
      hk_hobj_store(b, nir_src_as_deref(intr->src[0]), HK_HOBJ_SBT_INDEX,
                    intr->src[1].ssa);
      break;
   case nir_intrinsic_hit_object_load:
      def = hk_hit_object_load(b, compiler, intr);
      break;
   case nir_intrinsic_hit_object_load_attributes: {
      nir_deref_instr *hobj = nir_src_as_deref(intr->src[0]);
      nir_def *words[HK_HOBJ_ATTRIBUTE_WORDS];
      for (unsigned i = 0; i < intr->def.num_components; ++i)
         words[i] = nir_load_deref(b, hk_hobj_attribute(b, hobj, i));
      def = nir_vec(b, words, intr->def.num_components);
      break;
   }
   case nir_intrinsic_ignore_ray_intersection:
      nir_store_var(b, state->accept, nir_imm_false(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   case nir_intrinsic_terminate_ray:
      nir_store_var(b, state->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, state->terminate, nir_imm_true(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;

   /* Ray tracing system values */
   case nir_intrinsic_load_ray_launch_id:
      def = nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = hk_load_trace_ray_command_field(b, HK_RT_CMD_WIDTH, 3, 32);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = nir_load_var(b, state->shader_record_ptr);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = nir_load_var(b, state->tmin);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = nir_load_var(b, state->tmax);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = nir_load_var(b, state->origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = nir_load_var(b, state->dir);
      break;
   case nir_intrinsic_load_ray_instance_custom_index: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *custom_instance_and_mask = nir_load_global(
         b, 1, 32,
         nir_iadd_imm(b, instance_node_addr,
                      compiler->native_agx ? 96 :
                         offsetof(struct hk_bvh_instance_node,
                                  custom_instance_and_mask)));
      def = nir_iand_imm(b, custom_instance_and_mask, 0xFFFFFF);
      break;
   }
   case nir_intrinsic_load_primitive_id:
      def = nir_load_var(b, state->primitive_id);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = nir_iand_imm(b, nir_load_var(b, state->geometry_id_and_flags),
                         0xFFFFFFF);
      break;
   case nir_intrinsic_load_instance_id: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      def = nir_load_global(
         b, 1, 32,
         nir_iadd_imm(b, instance_node_addr,
                      compiler->native_agx ? 100 :
                         offsetof(struct hk_bvh_instance_node, instance_id)));
      break;
   }
   case nir_intrinsic_load_ray_flags:
      def = nir_load_var(b, state->flags);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = nir_load_var(b, state->hit_kind);
      break;
   case nir_intrinsic_load_ray_world_to_object: {
      const unsigned c = nir_intrinsic_column(intr);
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      hk_rt_load_wto(b, compiler, instance_node_addr, wto_matrix);

      nir_def *vals[3];
      for (unsigned i = 0; i < 3; ++i)
         vals[i] = nir_channel(b, wto_matrix[i], c);

      def = nir_vec(b, vals, 3);
      break;
   }
   case nir_intrinsic_load_ray_object_to_world: {
      const unsigned c = nir_intrinsic_column(intr);
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      if (compiler->native_agx) {
         nir_def *relative = nir_load_global(b, 1, 64,
            nir_iadd_imm(b, instance_node_addr, 104), .align_mul = 8);
         def = nir_load_global(b, 3, 32,
            nir_iadd_imm(b, nir_iadd(b, instance_node_addr, relative), c*12),
            .align_mul = 4);
         break;
      }
      nir_def *rows[3];
      for (unsigned r = 0; r < 3; ++r)
         rows[r] = nir_load_global(
            b, 4, 32,
            nir_iadd_imm(b, instance_node_addr,
                         offsetof(struct hk_bvh_instance_node, otw_matrix) +
                            r * 16));
      def = nir_vec3(b, nir_channel(b, rows[0], c), nir_channel(b, rows[1], c),
                     nir_channel(b, rows[2], c));
      break;
   }
   case nir_intrinsic_load_ray_object_origin: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      hk_rt_load_wto(b, compiler, instance_node_addr, wto_matrix);
      def = hk_mul_vec3_mat(b, nir_load_var(b, state->origin), wto_matrix,
                            true);
      break;
   }
   case nir_intrinsic_load_ray_object_direction: {
      nir_def *instance_node_addr = nir_load_var(b, state->instance_addr);
      nir_def *wto_matrix[3];
      hk_rt_load_wto(b, compiler, instance_node_addr, wto_matrix);
      def = hk_mul_vec3_mat(b, nir_load_var(b, state->dir), wto_matrix, false);
      break;
   }
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, nir_load_var(b, state->cull_mask), 0xFF);
      break;

   /* Ray tracing stack lowering: every stage's scratch access is relative to
    * its own frame. */
   case nir_intrinsic_load_scratch:
      nir_src_rewrite(&intr->src[0],
                      nir_iadd(b, nir_load_var(b, state->stack_ptr),
                               intr->src[0].ssa));
      return true;
   case nir_intrinsic_store_scratch:
      nir_src_rewrite(&intr->src[1],
                      nir_iadd(b, nir_load_var(b, state->stack_ptr),
                               intr->src[1].ssa));
      return true;

   case nir_intrinsic_load_ray_triangle_vertex_positions:
      def = hk_load_vertex_position(b, nir_load_var(b, state->primitive_addr),
                                    nir_intrinsic_column(intr));
      break;

   /* Internal system values.  These two intrinsics are shared with lavapipe
    * rather than duplicated: they are driver-internal markers with no encoding
    * of their own, and both are fully consumed here. */
   case nir_intrinsic_load_shader_call_data_offset_lvp:
      def = nir_load_var(b, state->shader_call_data_offset);
      break;

   default:
      return false;
   }

   if (def)
      nir_def_rewrite_uses(&intr->def, def);
   nir_instr_remove(instr);

   return true;
}

static void
hk_rt_optimize_nir(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
   } while (progress);
}

/* ------------------------------------------------------------------------ */
/* Pipeline creation                                                         */
/* ------------------------------------------------------------------------ */

static void
hk_ray_tracing_pipeline_destroy(struct vk_device *vk_dev,
                                struct vk_pipeline *vk_pipeline,
                                const VkAllocationCallbacks *pAllocator)
{
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   if (pipeline->cache_object)
      vk_pipeline_cache_object_unref(vk_dev, pipeline->cache_object);

   if (pipeline->shader)
      hk_api_shader_destroy(vk_dev, &pipeline->shader->vk, pAllocator);

   if (pipeline->kernels) {
      for (uint32_t i = 0; i < pipeline->kernel_count; i++) {
         if (pipeline->kernels[i].shader)
            hk_api_shader_destroy(vk_dev, &pipeline->kernels[i].shader->vk,
                                  pAllocator);
      }
      free(pipeline->kernels);
   }

   if (pipeline->stages) {
      for (uint32_t i = 0; i < pipeline->stage_count; i++)
         hk_rt_stage_nir_unref(pipeline->stages[i]);
      free(pipeline->stages);
   }

   free(pipeline->groups);

   if (pipeline->layout)
      vk_pipeline_layout_unref(vk_dev, pipeline->layout);

   vk_pipeline_free(vk_dev, pAllocator, &pipeline->base);
}

static void
hk_ray_tracing_pipeline_cmd_bind(struct vk_command_buffer *vk_cmd,
                                 struct vk_pipeline *vk_pipeline)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd, struct hk_cmd_buffer, vk);

   cmd->state.rt.pipeline =
      vk_pipeline ? hk_pipeline_to_ray_tracing(vk_pipeline) : NULL;
}

static struct vk_shader *
hk_ray_tracing_pipeline_get_shader(struct vk_pipeline *vk_pipeline,
                                   mesa_shader_stage stage)
{
   /* The pipeline is one compute shader; there is no per-stage binary to
    * report.  Saying so is better than handing back the megakernel under a
    * stage name it is not. */
   return NULL;
}

/* All RT stages are inlined into a single compute megakernel, except on M5,
 * which compiles one kernel per ray-generation group. Report each executable
 * with the API stages it implements rather than COMPUTE. A library only
 * retains stage NIR and has no executable to enumerate.
 */
static VkResult
hk_rt_pipeline_get_executable_properties(
   struct vk_device *device, struct vk_pipeline *vk_pipeline,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   if (pipeline->kernel_count) {
      const uint32_t capacity = properties ? *executable_count : 0;
      *executable_count = properties ? MIN2(capacity, pipeline->kernel_count)
                                     : pipeline->kernel_count;
      for (uint32_t i = 0; properties && i < *executable_count; i++) {
         struct vk_shader *shader = &pipeline->kernels[i].shader->vk;
         uint32_t count = 1;
         VkResult result = shader->ops->get_executable_properties(
            device, shader, &count, &properties[i]);
         if (result != VK_SUCCESS)
            return result;
         assert(count == 1);
         properties[i].stages = pipeline->base.stages;
         snprintf(properties[i].name, sizeof(properties[i].name),
                  "Ray generation %u", pipeline->kernels[i].raygen_handle_index);
         snprintf(properties[i].description, sizeof(properties[i].description),
                  "Ray-tracing megakernel for ray-generation group %u",
                  pipeline->kernels[i].raygen_handle_index);
      }
      return properties && capacity < pipeline->kernel_count ? VK_INCOMPLETE
                                                             : VK_SUCCESS;
   }

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                         executable_count);

   if (pipeline->shader) {
      vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props) {
         props->stages = pipeline->base.stages;
         props->subgroupSize = 32;
         VK_COPY_STR(props->name, "Ray tracing megakernel");
         VK_COPY_STR(props->description,
                     "Ray tracing stages inlined into a compute shader");
      }
   }

   return vk_outarray_status(&out);
}

static const struct vk_shader *
hk_rt_pipeline_executable(const struct hk_ray_tracing_pipeline *pipeline,
                          uint32_t executable_index)
{
   if (pipeline->kernel_count) {
      assert(executable_index < pipeline->kernel_count);
      return &pipeline->kernels[executable_index].shader->vk;
   }

   assert(pipeline->shader && executable_index == 0);
   return &pipeline->shader->vk;
}

static VkResult
hk_rt_pipeline_get_executable_statistics(
   struct vk_device *device, struct vk_pipeline *vk_pipeline,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   const struct vk_shader *shader = hk_rt_pipeline_executable(
      hk_pipeline_to_ray_tracing(vk_pipeline), executable_index);

   return shader->ops->get_executable_statistics(
      device, shader, 0, statistic_count, statistics);
}

static VkResult
hk_rt_pipeline_get_internal_representations(
   struct vk_device *device, struct vk_pipeline *vk_pipeline,
   uint32_t executable_index, uint32_t *representation_count,
   VkPipelineExecutableInternalRepresentationKHR *representations)
{
   const struct vk_shader *shader = hk_rt_pipeline_executable(
      hk_pipeline_to_ray_tracing(vk_pipeline), executable_index);

   return shader->ops->get_executable_internal_representations(
      device, shader, 0, representation_count, representations);
}

const struct vk_pipeline_ops hk_ray_tracing_pipeline_ops = {
   .get_executable_properties = hk_rt_pipeline_get_executable_properties,
   .get_executable_statistics = hk_rt_pipeline_get_executable_statistics,
   .get_internal_representations = hk_rt_pipeline_get_internal_representations,
   .destroy = hk_ray_tracing_pipeline_destroy,
   .cmd_bind = hk_ray_tracing_pipeline_cmd_bind,
   .get_shader = hk_ray_tracing_pipeline_get_shader,
};

static VkResult
hk_rt_stage_to_nir(struct hk_device *dev,
                   VkPipelineCreateFlags2KHR pipeline_flags, const void *pNext,
                   const VkPipelineShaderStageCreateInfo *stage_info,
                   nir_shader **nir_out)
{
   if (pipeline_flags &
       VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR)
      return VK_PIPELINE_COMPILE_REQUIRED;

   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct vk_pipeline_robustness_state rstate;

   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rstate, pNext,
                                     stage_info->pNext);

   const mesa_shader_stage stage =
      vk_to_mesa_shader_stage(stage_info->stage);
   const struct spirv_to_nir_options spirv_options =
      hk_get_spirv_options(&pdev->vk, stage, &rstate);
   const nir_shader_compiler_options *nir_options =
      hk_get_nir_options(&pdev->vk, stage, &rstate);

   return vk_pipeline_shader_stage_to_nir(&dev->vk, pipeline_flags, stage_info,
                                          &spirv_options, nir_options, NULL,
                                          nir_out);
}


/*
 * A `rayQueryEXT` declared inside a ray-tracing stage is an ordinary
 * function_temp variable, and hk_nir_lower_ray_queries() finds it by
 * `var->data.ray_query` and then requires every rq_* intrinsic's source to
 * still be a DEREF of it (`hk_nir_lower_ray_queries.c:607-617`).  The
 * per-stage nir_lower_explicit_io(nir_var_function_temp) below rewrites that
 * deref into a 32-bit scratch offset, after which the pass aborts on
 * nir_instr_as_deref -- `nir.h:3060: nir_instr_as_deref: Assertion `parent &&
 * parent->type == nir_instr_type_deref' failed`, and a plain SIGSEGV in a
 * release build.  lavapipe has the same shape and the same hazard
 * (`lvp_ray_tracing_pipeline.c:181`).
 *
 * Ray-query lowering cannot simply move earlier: it needs rq_initialize's
 * acceleration structure to already be a resolved 64-bit address, and
 * descriptor lowering happens inside hk_lower_nir(), long after the megakernel
 * is assembled (`hk_shader.c:1764-1783`).  So park the query variables in a
 * mode the stage lowering does not touch and put them back before the stage
 * joins the megakernel.  Nothing between those two points lowers
 * nir_var_shader_temp inside a stage, and nothing after the unpark lowers
 * function_temp until hk_lower_nir() has run the ray-query pass.
 */
static nir_variable *
hk_deref_base_var(nir_deref_instr *deref)
{
   while (deref->deref_type != nir_deref_type_var) {
      if (deref->deref_type == nir_deref_type_cast)
         return NULL;

      deref = nir_deref_instr_parent(deref);
      if (!deref)
         return NULL;
   }

   return deref->var;
}

static void
hk_park_ray_query_vars(nir_shader *nir, bool park)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   const nir_variable_mode to =
      park ? nir_var_shader_temp : nir_var_function_temp;
   struct exec_list *src = park ? &impl->locals : &nir->variables;
   struct exec_list *dst = park ? &nir->variables : &impl->locals;
   bool any = false;

   foreach_list_typed_safe (nir_variable, var, node, src) {
      if (!var->data.ray_query && !var->data.hit_object)
         continue;

      exec_node_remove(&var->node);
      var->data.mode = to;
      exec_list_push_tail(dst, &var->node);
      any = true;
   }

   if (!any)
      return;

   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         nir_variable *var = hk_deref_base_var(deref);
         if (var && (var->data.ray_query || var->data.hit_object))
            deref->modes = to;
      }
   }

   nir_progress(true, impl, nir_metadata_all);
}

static VkResult
hk_compile_ray_tracing_stages(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   const VkPipelineCreateFlags2KHR pipeline_flags =
      vk_rt_pipeline_create_flags(create_info);
   uint32_t i = 0;

   for (; i < create_info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage_info =
         create_info->pStages + i;

      /* VK_EXT_shader_module_identifier: a stage given only by its module
       * identifier can only come from a pipeline cache, and ray-tracing
       * pipelines are never retrieved from one here. The application asked
       * for VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT (it is
       * required with identifiers), so the answer is that the pipeline must
       * be compiled; there is no SPIR-V to compile it from.
       */
      if (stage_info->module == VK_NULL_HANDLE &&
          !vk_find_struct_const(stage_info->pNext,
                                SHADER_MODULE_CREATE_INFO) &&
          vk_pipeline_shader_stage_has_identifier(stage_info))
         return VK_PIPELINE_COMPILE_REQUIRED;

      /* There is no RT stage cache lookup here.  Even when SPIR-V is
       * available, translating and lowering a new stage requires compilation.
       * Imported library stages below already have NIR and can be reused.
       */
      if (pipeline_flags &
          VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR)
         return VK_PIPELINE_COMPILE_REQUIRED;

      nir_shader *nir;
      VkResult result = hk_rt_stage_to_nir(dev, pipeline_flags,
                                           create_info->pNext,
                                           create_info->pStages + i, &nir);
      if (result != VK_SUCCESS)
         return result;

      /* Every stage's private storage is its frame of the megakernel's
       * scratch stack, so nothing may already be claiming scratch. */
      /* Turn system-value variables into intrinsics while the stage is still
       * its own shader.  gl_LaunchIDEXT and friends arrive from SPIR-V as
       * `decl_var system` plus a deref; hk_lower_ray_tracing_instr() lowers
       * the INTRINSIC form, and nothing downstream of the megakernel build
       * would reach a surviving deref -- the driver's own
       * nir_lower_system_values runs inside hk_preprocess_nir_internal, which
       * is after the ray-tracing lowering pass has already been and gone.
       *
       * Getting this wrong is invisible until the AGX backend, and the way it
       * announces itself is hostile: agx_emit_intrinsic()'s default prints the
       * name once and then UNREACHABLE()s, which in a release build spun out
       * 184 million lines before the process died.  Redirect to a file and
       * head it; the first line is the whole message. */
      NIR_PASS(_, nir, nir_lower_system_values);

      /* An outgoing ray payload (SpvStorageClassRayPayloadKHR) and an outgoing
       * callable payload arrive as nir_var_shader_temp
       * (`vtn_variables.c:1872-1875`), not function_temp, so they are NOT
       * covered by the explicit-type and explicit-io lowering below and their
       * derefs survive into the megakernel.  traceRayEXT's payload source must
       * be a scratch OFFSET by the time hk_trace_ray() reads it, or the
       * megakernel biases the callee's payload window by a pointer.
       *
       * Demote them to function_temp first.  lavapipe gets this for free
       * because its per-stage preprocessing already ran
       * nir_lower_global_vars_to_local; hk's does not, and the symptom is a
       * `deref_var &payload (function_temp uint)` reaching agx_emit_instr()
       * and asserting "should've been lowered" -- in a release build, a
       * segfault with no message. */
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);

      /* nir_lower_vars_to_ssa promotes most locals to SSA but leaves their
       * declarations behind, and nir_lower_vars_to_explicit_types below
       * assigns a frame slot to every surviving function_temp declaration,
       * promoted or not.  Every call site of a GLSL function was inlined
       * with its own copy of the callee's locals, so the dead declarations
       * dominate the frame: RTGL1's IndirectInit raygen declared a 28,416-byte
       * frame of which the kernel touched 24 bytes (the payload window), and
       * the per-dispatch stack reservation multiplies that by the recursion
       * depth.  Drop the dead declarations so the frame holds only what is
       * still addressed (payload, hit attributes, dynamically indexed arrays,
       * ray queries). */
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);

      assert(!nir->scratch_size);
      if (nir->info.stage == MESA_SHADER_ANY_HIT ||
          nir->info.stage == MESA_SHADER_CLOSEST_HIT ||
          nir->info.stage == MESA_SHADER_INTERSECTION)
         nir->scratch_size = HK_RAY_HIT_ATTRIBS_SIZE;

      hk_park_ray_query_vars(nir, true);

      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_function_temp | nir_var_shader_call_data |
                  nir_var_ray_hit_attrib,
               glsl_get_natural_size_align_bytes);

      NIR_PASS(_, nir, hk_lower_ray_tracing_derefs);

      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_function_temp,
               nir_address_format_32bit_offset);

      /* A shaderRecordEXT block is not a variable.  vtn builds a deref_cast of
       * nir_load_shader_record_ptr() in nir_var_mem_constant
       * (`vtn_variables.c:448-457` and `:1884-1886`), so there is no
       * nir_variable for nir_lower_vars_to_explicit_types to find and the mode
       * is one hk_lower_nir() never lowers: it lowers nir_var_mem_global,
       * nir_var_mem_ssbo, nir_var_mem_ubo and nir_var_mem_shared and no other
       * (`hk_shader.c:1767-1775`, `:1811-1814`).  Without this the deref
       * survives every pass and reaches the AGX backend, where
       * agx_extract_nir_src() reads a def that was never emitted -- in a
       * release build a SIGSEGV inside agx_emit_alu() with no message, which
       * is nowhere near the deref.  64-bit global is the right format: the
       * record address is an ordinary device address and
       * agx_nir_lower_address() already lowers load_global_constant
       * (`agx_nir_lower_address.c:110-111`, `:240`). */
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant,
               nir_address_format_64bit_global);

      hk_park_ray_query_vars(nir, false);
      hk_retype_hit_objects(nir);

      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_move_ray_tracing_intrinsic, nir_metadata_control_flow, NULL);

      pipeline->stages[i] = hk_rt_stage_nir_create(nir);
      if (!pipeline->stages[i]) {
         ralloc_free(nir);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   if (!create_info->pLibraryInfo)
      return VK_SUCCESS;

   for (uint32_t library_index = 0;
        library_index < create_info->pLibraryInfo->libraryCount;
        library_index++) {
      VK_FROM_HANDLE(vk_pipeline, vk_library,
                     create_info->pLibraryInfo->pLibraries[library_index]);
      struct hk_ray_tracing_pipeline *library =
         hk_pipeline_to_ray_tracing(vk_library);

      for (uint32_t stage_index = 0; stage_index < library->stage_count;
           stage_index++) {
         hk_rt_stage_nir_ref(pipeline->stages + i,
                             library->stages[stage_index]);
         i++;
      }
   }

   return VK_SUCCESS;
}

/* Lowest address a capture/replay handle pointer may hold: the smallest page
 * size, below which no process memory can be mapped. */
#define HK_RT_MIN_REPLAY_HANDLE_ADDRESS 4096

/* Keep new handles above one that came from elsewhere (capture/replay or a
 * pipeline cache). */
static void
hk_rt_handle_alloc_above(struct hk_device *dev, uint32_t index)
{
   uint32_t current = p_atomic_read(&dev->rt_group_handle_alloc);
   while (current < index) {
      uint32_t seen =
         p_atomic_cmpxchg(&dev->rt_group_handle_alloc, current, index);
      if (seen == current)
         break;
      current = seen;
   }
}

struct hk_rt_library_handles {
   uint8_t key[BLAKE3_OUT_LEN];
   uint32_t count;
   uint32_t index[];
};

static uint32_t
hk_rt_library_key_hash(const void *key)
{
   return _mesa_hash_data(key, BLAKE3_OUT_LEN);
}

static bool
hk_rt_library_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, BLAKE3_OUT_LEN) == 0;
}

/* Remember the own group handles of the latest library with content `key`,
 * which a library loaded from a cache with that content will take. */
static void
hk_rt_library_handles_record(struct hk_device *dev, const uint8_t *key,
                             const struct hk_ray_tracing_group *groups,
                             uint32_t count)
{
   simple_mtx_lock(&dev->rt_library_handles.lock);
   if (!dev->rt_library_handles.table) {
      dev->rt_library_handles.table = _mesa_hash_table_create(
         NULL, hk_rt_library_key_hash, hk_rt_library_key_equal);
   }
   struct hash_table *table = dev->rt_library_handles.table;
   struct hash_entry *he = table ? _mesa_hash_table_search(table, key) : NULL;
   struct hk_rt_library_handles *entry = he ? he->data : NULL;
   if (entry && entry->count != count) {
      _mesa_hash_table_remove(table, he);
      free(entry);
      entry = NULL;
   }
   if (!entry && table) {
      entry = malloc(sizeof(*entry) + count * sizeof(entry->index[0]));
      if (entry) {
         memcpy(entry->key, key, sizeof(entry->key));
         entry->count = count;
         _mesa_hash_table_insert(table, entry->key, entry);
      }
   }
   if (entry) {
      for (uint32_t i = 0; i < count; ++i)
         entry->index[i] = groups[i].handle.index;
   }
   simple_mtx_unlock(&dev->rt_library_handles.lock);
}

static bool
hk_rt_library_handles_find(struct hk_device *dev, const uint8_t *key,
                           struct hk_ray_tracing_group *groups, uint32_t count)
{
   bool found = false;
   simple_mtx_lock(&dev->rt_library_handles.lock);
   struct hash_entry *he = dev->rt_library_handles.table ?
      _mesa_hash_table_search(dev->rt_library_handles.table, key) : NULL;
   if (he) {
      const struct hk_rt_library_handles *entry = he->data;
      if (entry->count == count) {
         for (uint32_t i = 0; i < count; ++i)
            groups[i].handle.index = entry->index[i];
         found = true;
      }
   }
   simple_mtx_unlock(&dev->rt_library_handles.lock);
   return found;
}

void
hk_rt_device_finish(struct hk_device *dev)
{
   if (!dev->rt_library_handles.table)
      return;
   hash_table_foreach (dev->rt_library_handles.table, he)
      free(he->data);
   _mesa_hash_table_destroy(dev->rt_library_handles.table, NULL);
   dev->rt_library_handles.table = NULL;
}

static VkResult
hk_init_ray_tracing_groups(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   uint32_t i = 0;

   for (; i < create_info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group_info =
         create_info->pGroups + i;

      if (!hk_rt_group_from_create_info(
             pipeline->groups + i, group_info->type,
             group_info->generalShader, group_info->closestHitShader,
             group_info->anyHitShader, group_info->intersectionShader))
         return VK_ERROR_FEATURE_NOT_PRESENT;

      /* Library stages are appended to the local stage array before group
       * initialization. Newly declared groups may reference any stage in
       * that combined array, including a shader-only library's stages. */
      const struct hk_ray_tracing_group *g = pipeline->groups + i;
      const uint32_t total_stage_count = pipeline->stage_count;
      if ((g->recursive_index != VK_SHADER_UNUSED_KHR &&
           g->recursive_index >= total_stage_count) ||
          (g->ahit_index != VK_SHADER_UNUSED_KHR &&
           g->ahit_index >= total_stage_count) ||
          (g->isec_index != VK_SHADER_UNUSED_KHR &&
           g->isec_index >= total_stage_count)) {
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                          "ray tracing group %u names a shader index past "
                          "combined stageCount %u",
                          i, total_stage_count);
      }

      /* Handles are unique per device, not per pipeline, so a handle an
       * application captured from a library keeps identifying the same group
       * after it is imported. */
      if ((vk_rt_pipeline_create_flags(create_info) &
           VK_PIPELINE_CREATE_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR) &&
          group_info->pShaderGroupCaptureReplayHandle) {
         /* A replay handle points at the 4-byte value that
          * vkGetRayTracingCaptureReplayShaderGroupHandlesKHR returned. No
          * application memory lives in the zero page; a pointer there is a
          * tool's offset from a NULL base (gfxreconstruct replays a 0-byte
          * captured handle set as NULL + 4 * group). Report that the handle
          * cannot be reused instead of faulting on it. */
         if ((uintptr_t)group_info->pShaderGroupCaptureReplayHandle <
             HK_RT_MIN_REPLAY_HANDLE_ADDRESS) {
            return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                             "ray tracing group %u: capture/replay handle "
                             "pointer %p is in the zero page",
                             i, group_info->pShaderGroupCaptureReplayHandle);
         }
         uint32_t replay;
         memcpy(&replay, group_info->pShaderGroupCaptureReplayHandle,
                sizeof(replay));
         /* Handles start at 1 (p_atomic_inc_return below), so 0 was never
          * returned by vkGetRayTracingCaptureReplayShaderGroupHandlesKHR. */
         if (!replay) {
            return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                             "ray tracing group %u: capture/replay handle 0 "
                             "was never issued", i);
         }
         pipeline->groups[i].handle.index = replay;
         hk_rt_handle_alloc_above(dev, replay);
      } else {
         pipeline->groups[i].handle.index = 0;
      }
   }

   for (i = 0; i < create_info->groupCount; i++) {
      if (!pipeline->groups[i].handle.index)
         pipeline->groups[i].handle.index =
            p_atomic_inc_return(&dev->rt_group_handle_alloc);
   }

   if (!create_info->pLibraryInfo)
      return VK_SUCCESS;

   uint32_t stage_base_index = create_info->stageCount;
   for (uint32_t library_index = 0;
        library_index < create_info->pLibraryInfo->libraryCount;
        library_index++) {
      VK_FROM_HANDLE(vk_pipeline, vk_library,
                     create_info->pLibraryInfo->pLibraries[library_index]);
      struct hk_ray_tracing_pipeline *library =
         hk_pipeline_to_ray_tracing(vk_library);

      for (uint32_t group_index = 0; group_index < library->group_count;
           group_index++) {
         pipeline->groups[i] =
            hk_rt_group_rebase(library->groups[group_index], stage_base_index);
         i++;
      }

      stage_base_index += library->stage_count;
   }

   return VK_SUCCESS;
}

/*
 * Compile the kernel for one ray-generation group: that raygen, plus every
 * stage reachable from it through the shader binding table (hit, miss,
 * intersection, callable), inlined.  Without a group, compile the single
 * megakernel that selects between every raygen group.  The kernel still checks the raygen
 * record's handle against its own group, so a kernel dispatched for another
 * group's record retires without tracing; the GPU-side selection in
 * hk_cmd_trace_rays.c dispatches the others with zero threadgroups.
 */
static VkResult
hk_compile_rt_kernel(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info,
   const VkAllocationCallbacks *pAllocator,
   const struct hk_ray_tracing_group *raygen_group,
   struct hk_rt_kernel *kernel)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct vk_pipeline_robustness_state rstate;

   /* The megakernel is one shader made of many stages, so there is no
    * per-stage robustness to honour: take the pipeline-level state, which is
    * what VK_EXT_pipeline_robustness defines for a stage that does not
    * override it. */
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rstate, create_info->pNext,
                                     NULL);

   const nir_shader_compiler_options *nir_options =
      hk_get_nir_options(&pdev->vk, MESA_SHADER_COMPUTE, &rstate);

   nir_builder _b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, nir_options, "ray tracing pipeline");
   nir_builder *b = &_b;

   kernel->raygen_handle_index = raygen_group ? raygen_group->handle.index : 0;

   /* nir_builder_init_simple_shader() marks the shader internal, and
    * hk_compile_nir() turns that into `has_scratch = false`
    * (hk_shader.c:2633), which makes agx_ra() assert "internal shaders are
    * unspillable".  The megakernel's whole stack model IS scratch, so it is
    * not an internal shader in that sense: it is an application shader that
    * the driver happens to assemble.  A release build does not assert; it
    * spills into a scratch allocation that does not exist. */
   b->shader->info.internal = false;

   b->shader->info.workgroup_size[0] = HK_RT_WORKGROUP_SIZE_X;
   b->shader->info.workgroup_size[1] = HK_RT_WORKGROUP_SIZE_Y;
   b->shader->info.workgroup_size[2] = HK_RT_WORKGROUP_SIZE_Z;

   struct hk_rt_compiler compiler = {
      .pipeline = pipeline,
      .flags = vk_rt_pipeline_create_flags(create_info),
      .position_fetch = dev->vk.enabled_features.rayTracingPositionFetch,
      .native_agx = dev->gen1_native_rq && dev->gen1_rq_instances,
      .m5 = agx_chip_g17_m5(dev->dev.chip),
      .callable_dispatch = hk_callable_dispatch_needed(pipeline),
      .max_depth = create_info->maxPipelineRayRecursionDepth,
   };
   hk_ray_tracing_state_init(b->shader, &compiler);
   compiler.functions = rzalloc_array(b->shader, struct hash_table *,
                                      compiler.max_depth + 1);
   if (!compiler.functions) {
      ralloc_free(b->shader);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   for (unsigned depth = 0; depth <= compiler.max_depth; ++depth)
      compiler.functions[depth] = _mesa_pointer_hash_table_create(b->shader);
   compiler.function_depths = _mesa_pointer_hash_table_create(b->shader);
   compiler.callable_parts = _mesa_pointer_hash_table_create(b->shader);
   compiler.callable_functions = _mesa_pointer_hash_table_create(b->shader);
   compiler.callable_contexts = rzalloc_array(b->shader,
      struct hk_rt_callable_context *, compiler.max_depth + 1);
   compiler.binding_vars = _mesa_pointer_set_create(b->shader);

   /* Deeper pipelines run as continuations unless a ray query or hit object
    * lives across a call; those keep the inlined megakernel. */
   if (compiler.max_depth > HK_RT_INLINE_MAX_RAY_RECURSION_DEPTH &&
       hk_rt_cont_split(&compiler, b->shader)) {
      if (compiler.cont_stack_size > HK_RT_CONT_MAX_STACK_SIZE) {
         ralloc_free(b->shader);
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "RT continuation stack of %u bytes exceeds %u",
                          compiler.cont_stack_size, HK_RT_CONT_MAX_STACK_SIZE);
      }
      compiler.continuation = true;
      compiler.callable_dispatch = false;
   }

   /* A launch is dispatched in whole workgroups, so the trailing workgroup of
    * a width that is not a multiple of the workgroup size runs invocations the
    * application never asked for.  Retire them before anything else. */
   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *oob = nir_ige(b, nir_channel(b, launch_id, 0),
                          nir_channel(b, launch_size, 0));
   oob = nir_ior(b, oob,
                 nir_ige(b, nir_channel(b, launch_id, 1),
                         nir_channel(b, launch_size, 1)));
   oob = nir_ior(b, oob,
                 nir_ige(b, nir_channel(b, launch_id, 2),
                         nir_channel(b, launch_size, 2)));

   nir_push_if(b, oob);
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);

   nir_store_var(b, compiler.state.stack_ptr,
                 nir_load_ray_tracing_stack_base_lvp(b), 0x1);

   struct hk_sbt_entry raygen_entry =
      hk_load_sbt_entry(b, NULL, HK_RT_CMD_RAYGEN_ADDRESS);
   nir_store_var(b, compiler.state.shader_record_ptr,
                 raygen_entry.shader_record_ptr, 0x1);

   if (compiler.continuation) {
      hk_rt_cont_build(b, &compiler, raygen_group, raygen_entry);
   } else if (raygen_group) {
      nir_shader *stage = pipeline->stages[raygen_group->recursive_index]->nir;
      assert(stage->info.stage == MESA_SHADER_RAYGEN);

      nir_push_if(b, nir_ieq_imm(b, raygen_entry.value,
                                 raygen_group->handle.index));
      hk_call_ray_tracing_stage(b, &compiler, stage, false);
      nir_pop_if(b, NULL);
   } else {
      /* Single megakernel: every raygen group, selected by its handle. */
      for (uint32_t i = 0; i < pipeline->group_count; i++) {
         struct hk_ray_tracing_group *group = pipeline->groups + i;
         if (group->recursive_index == VK_SHADER_UNUSED_KHR)
            continue;

         nir_shader *stage = pipeline->stages[group->recursive_index]->nir;
         if (stage->info.stage != MESA_SHADER_RAYGEN)
            continue;

         nir_push_if(b, nir_ieq_imm(b, raygen_entry.value, group->handle.index));
         hk_call_ray_tracing_stage(b, &compiler, stage, false);
         nir_pop_if(b, NULL);
      }
   }

   nir_shader_instructions_pass(b->shader, hk_lower_ray_tracing_instr,
                                nir_metadata_none, &compiler);

   NIR_PASS(_, b->shader, nir_lower_continue_constructs);
   NIR_PASS(_, b->shader, nir_lower_variable_initializers,
            nir_var_function_temp);
   NIR_PASS(_, b->shader, nir_lower_returns);

   /* The AGX backend has no call instruction: every stage was inlined as a
    * NIR function so it could be emitted once and called from several places,
    * and the calls are flattened here.  lavapipe leaves them standing because
    * its backend executes calls directly; ours cannot, and a call that reaches
    * agx_compile_shader_nir() faults rather than refusing. */
   /* Depth alone does not bound code size: two traces per stage duplicate
    * the complete callee tree at every level, even though the pre-inline DAG
    * is small. Reject that expansion before NIR allocates gigabytes and
    * drives the system into reclaim. This is a compiler resource limit, not
    * a substitute for the advertised depth bound. Legacy generations are
    * unchanged.
    */
   if (dev->dev.chip == AGX_CHIP_G17P &&
       !hk_rt_inlining_fits(b->shader, 1u << 20)) {
      ralloc_free(b->shader);
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "RT inlining exceeds 1048576 instructions");
   }

   NIR_PASS(_, b->shader, nir_inline_functions);
   nir_remove_non_entrypoints(b->shader);

   const struct nir_lower_compute_system_values_options compute_system_values = {
      0
   };
   NIR_PASS(_, b->shader, nir_lower_compute_system_values,
            &compute_system_values);
   NIR_PASS(_, b->shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, b->shader, nir_lower_vars_to_ssa);

   hk_rt_optimize_nir(b->shader);

   NIR_PASS(_, b->shader, nir_lower_vars_to_explicit_types, nir_var_shader_temp,
            glsl_get_natural_size_align_bytes);

   NIR_PASS(_, b->shader, nir_lower_explicit_io, nir_var_shader_temp,
            nir_address_format_32bit_offset);

   /* Everything below the megakernel's own frame is the per-stage stack.
    * Continuation frames move at run time, so keep them 16-byte aligned. */
   if (compiler.continuation)
      b->shader->scratch_size = ALIGN_POT(b->shader->scratch_size, 16);
   NIR_PASS(_, b->shader, nir_inline_sysval,
            nir_intrinsic_load_ray_tracing_stack_base_lvp,
            b->shader->scratch_size);

   /* Each kernel reserves exactly what its own raygen needs; the pipeline
    * reports the maximum over its kernels. */
   const uint32_t stack_size = compiler.continuation ? compiler.cont_stack_size :
      hk_rt_pipeline_stack_size(
         &compiler.sizes, create_info->maxPipelineRayRecursionDepth);
   b->shader->scratch_size += stack_size;

   pipeline->stack_sizes.raygen =
      MAX2(pipeline->stack_sizes.raygen, compiler.sizes.raygen);
   pipeline->stack_sizes.any_hit =
      MAX2(pipeline->stack_sizes.any_hit, compiler.sizes.any_hit);
   pipeline->stack_sizes.closest_hit =
      MAX2(pipeline->stack_sizes.closest_hit, compiler.sizes.closest_hit);
   pipeline->stack_sizes.miss =
      MAX2(pipeline->stack_sizes.miss, compiler.sizes.miss);
   pipeline->stack_sizes.intersection =
      MAX2(pipeline->stack_sizes.intersection, compiler.sizes.intersection);
   pipeline->stack_sizes.callable =
      MAX2(pipeline->stack_sizes.callable, compiler.sizes.callable);
   pipeline->stack_size = MAX2(pipeline->stack_size, stack_size);

   hk_rt_optimize_nir(b->shader);

   if (agx_chip_isa_encoding(dev->dev.chip) == AGX_ISA_ENCODING_AGX3C ||
       dev->dev.chip == AGX_CHIP_G17P) {
      bool indirect = false;
      nir_foreach_function_impl(impl, b->shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_scratch)
                  indirect |= !nir_src_is_const(intr->src[0]);
               else if (intr->intrinsic == nir_intrinsic_store_scratch)
                  indirect |= !nir_src_is_const(intr->src[1]);
            }
         }
      }
      if (!indirect) {
         NIR_PASS(_, b->shader, nir_lower_scratch_to_var);
         hk_rt_optimize_nir(b->shader);
      }
   }

   for (unsigned depth = 0; depth <= compiler.max_depth; ++depth)
      _mesa_hash_table_destroy(compiler.functions[depth], NULL);
   _mesa_hash_table_destroy(compiler.function_depths, NULL);
   _mesa_hash_table_destroy(compiler.callable_parts, NULL);
   _mesa_hash_table_destroy(compiler.callable_functions, NULL);

   /* hk_lower_nir() gates hk_nir_lower_ray_queries on `info.ray_queries > 0`
    * (`hk_shader.c:1783`), and info.ray_queries is only ever set by
    * nir_shader_gather_info() -- which hk calls in exactly one place,
    * hk_preprocess_nir_internal() (`hk_shader.c:206`), the function this path
    * must NOT run (see below).  The megakernel starts from
    * nir_builder_init_simple_shader(), so its info.ray_queries is 0 no matter
    * what the stages contain, and a rayQueryEXT in any ray-tracing stage then
    * reaches agx_emit_intrinsic() as `rq_initialize`.  That is not a corner:
    * dEQP-VK.ray_query has a variant of nearly every case for each
    * ray-tracing-pipeline stage.
    *
    * Gather here, after inlining, when the stages' ray-query variables are
    * function_temp locals of the entrypoint and nir_shader_gather_info() can
    * see them (`nir_gather_info.c:1168-1182`).  This is also what every
    * ordinary shader gets; the megakernel simply reaches hk_compile_shader()
    * by a different road. */
   nir_shader_gather_info(b->shader, nir_shader_get_entrypoint(b->shader));

   /* Hand the megakernel to the ordinary compute path with `io_lowered` still
    * false, exactly as hk's own internal compute shaders do, so that
    * hk_compile_shader() runs hk_lower_nir() on it: descriptor lowering, push
    * constants and agx_preprocess_nir are all inside that one call, in that
    * order.
    *
    * Do NOT pre-run hk_preprocess_nir_internal() here.  It ends in
    * agx_preprocess_nir(), which sets info.io_lowered, and hk_compile_shader()
    * then SKIPS hk_lower_nir() entirely -- so the megakernel reaches the
    * backend with vulkan_resource_index still in it and faults.  (The plural
    * hk_compile_shaders() calls hk_lower_nir() unconditionally, which is why
    * the pipeline path does not have this hazard.) */
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_COMPUTE,
      .flags = 0,
      .rt_flags = compiler.flags & MESA_VK_PIPELINE_RAY_TRACING_FLAGS,
      .next_stage_mask = 0,
      .nir = b->shader,
      .robustness = &rstate,
      .set_layout_count = pipeline->layout ? pipeline->layout->set_count : 0,
      .set_layouts = pipeline->layout ? pipeline->layout->set_layouts : NULL,
      .push_constant_range_count = 0,
      .push_constant_ranges = NULL,
   };

   return hk_compile_shader(dev, &info, NULL, &dev->vk.enabled_features,
                            pAllocator, &kernel->shader);
}

/*
 * On M5, one kernel per ray-generation group.  A pipeline whose groups name no
 * raygen (only a library can be created that way) has nothing to compile;
 * hk_cmd_trace_rays.c refuses to trace with it.
 */
static VkResult
hk_compile_ray_tracing_pipeline(
   struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *create_info,
   const VkAllocationCallbacks *pAllocator)
{
   /* Only M5 compiles one kernel per ray-generation group; elsewhere the
    * pipeline is one megakernel. */
   if (!agx_chip_g17_m5(dev->dev.chip)) {
      struct hk_rt_kernel kernel = {0};
      VkResult result = hk_compile_rt_kernel(dev, pipeline, create_info,
                                             pAllocator, NULL, &kernel);
      pipeline->shader = kernel.shader;
      return result;
   }
   uint32_t raygen_count = 0;
   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct hk_ray_tracing_group *group = pipeline->groups + i;
      if (group->recursive_index != VK_SHADER_UNUSED_KHR &&
          pipeline->stages[group->recursive_index]->nir->info.stage ==
             MESA_SHADER_RAYGEN)
         raygen_count++;
   }

   if (raygen_count == 0)
      return VK_SUCCESS;

   if (vk_rt_pipeline_create_flags(create_info) &
       VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR)
      return VK_PIPELINE_COMPILE_REQUIRED;

   pipeline->kernels = calloc(raygen_count, sizeof(*pipeline->kernels));
   if (!pipeline->kernels)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct hk_ray_tracing_group *group = pipeline->groups + i;
      if (group->recursive_index == VK_SHADER_UNUSED_KHR ||
          pipeline->stages[group->recursive_index]->nir->info.stage !=
             MESA_SHADER_RAYGEN)
         continue;

      VkResult result = hk_compile_rt_kernel(
         dev, pipeline, create_info, pAllocator, group,
         &pipeline->kernels[pipeline->kernel_count]);
      if (result != VK_SUCCESS)
         return result;

      pipeline->kernel_count++;
   }

   assert(pipeline->kernel_count == raygen_count);
   return VK_SUCCESS;
}

/* The common binary entrypoints own vk_rt_pipeline objects.  Honeykrisp's
 * megakernel (or, on M5, per-raygen kernels) and retained library NIR have a
 * different layout and lifetime. */
#define HK_RT_BINARY_MAGIC   0x42524b48u
#define HK_RT_BINARY_VERSION 1u

/* What an RT binary carries after the stage NIR. */
enum hk_rt_binary_executable {
   HK_RT_BINARY_LIBRARY = 0,
   HK_RT_BINARY_MEGAKERNEL = 1,
   HK_RT_BINARY_RAYGEN_KERNELS = 2, /* M5 */
};

static bool
hk_rt_binaries_native(const struct hk_device *dev)
{
   return agx_chip_g17p_or_m5(dev->dev.chip);
}

static void
hk_rt_pipeline_key(struct hk_device *dev,
                   const VkRayTracingPipelineCreateInfoKHR *info,
                   uint8_t key[BLAKE3_OUT_LEN])
{
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   const VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(info);
   const VkPipelineCreateFlags2KHR semantic =
      flags & (MESA_VK_PIPELINE_RAY_TRACING_FLAGS |
               VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR);
   _mesa_blake3_update(&hash, &semantic, sizeof(semantic));
   _mesa_blake3_update(&hash, &info->maxPipelineRayRecursionDepth,
                       sizeof(uint32_t));
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   if (layout) {
      _mesa_blake3_update(&hash, &layout->set_count, sizeof(layout->set_count));
      for (uint32_t i = 0; i < layout->set_count; ++i) {
         const uint8_t absent[BLAKE3_OUT_LEN] = {0};
         _mesa_blake3_update(
            &hash,
            layout->set_layouts[i] ? layout->set_layouts[i]->blake3 : absent,
            BLAKE3_OUT_LEN);
      }
      _mesa_blake3_update(
         &hash, layout->push_ranges,
         layout->push_range_count * sizeof(layout->push_ranges[0]));
   }
   if (info->pLibraryInterface) {
      _mesa_blake3_update(&hash,
                          &info->pLibraryInterface->maxPipelineRayPayloadSize,
                          sizeof(uint32_t));
      _mesa_blake3_update(
         &hash, &info->pLibraryInterface->maxPipelineRayHitAttributeSize,
         sizeof(uint32_t));
   }
   _mesa_blake3_update(&hash, &info->stageCount, sizeof(uint32_t));
   for (uint32_t i = 0; i < info->stageCount; ++i) {
      struct vk_pipeline_robustness_state rs;
      vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rs,
                                        info->pNext, info->pStages[i].pNext);
      uint8_t stage[BLAKE3_OUT_LEN];
      vk_pipeline_hash_shader_stage(flags, &info->pStages[i], &rs, stage);
      _mesa_blake3_update(&hash, stage, sizeof(stage));
   }
   _mesa_blake3_update(&hash, &info->groupCount, sizeof(uint32_t));
   for (uint32_t i = 0; i < info->groupCount; ++i) {
      const VkRayTracingShaderGroupCreateInfoKHR *g = &info->pGroups[i];
      const uint32_t fields[] = {g->type, g->generalShader, g->closestHitShader,
                                 g->anyHitShader, g->intersectionShader};
      _mesa_blake3_update(&hash, fields, sizeof(fields));
      if (g->pShaderGroupCaptureReplayHandle)
         _mesa_blake3_update(&hash, g->pShaderGroupCaptureReplayHandle,
                             HK_RAY_TRACING_GROUP_HANDLE_SIZE);
   }
   uint32_t libraries =
      info->pLibraryInfo ? info->pLibraryInfo->libraryCount : 0;
   _mesa_blake3_update(&hash, &libraries, sizeof(libraries));
   for (uint32_t i = 0; i < libraries; ++i) {
      VK_FROM_HANDLE(hk_ray_tracing_pipeline, lib,
                     info->pLibraryInfo->pLibraries[i]);
      _mesa_blake3_update(&hash, lib->binary_key, sizeof(lib->binary_key));
   }
   _mesa_blake3_final(&hash, key);
}

static bool
hk_rt_binary_serialize(struct hk_device *dev,
                       const struct hk_ray_tracing_pipeline *pipeline,
                       struct blob *blob)
{
   blob_write_uint32(blob, HK_RT_BINARY_MAGIC);
   blob_write_uint32(blob, HK_RT_BINARY_VERSION);
   blob_write_bytes(blob, dev->vk.physical->properties.shaderBinaryUUID,
                    VK_UUID_SIZE);
   blob_write_bytes(blob, pipeline->binary_key, sizeof(pipeline->binary_key));
   blob_write_uint32(blob, pipeline->stage_count);
   blob_write_uint32(blob, pipeline->group_count);
   blob_write_uint32(blob, pipeline->base.stages);
   blob_write_uint32(blob, pipeline->kernel_count ? HK_RT_BINARY_RAYGEN_KERNELS
                           : pipeline->shader ? HK_RT_BINARY_MEGAKERNEL
                                              : HK_RT_BINARY_LIBRARY);
   blob_write_bytes(blob, &pipeline->stack_sizes,
                    sizeof(pipeline->stack_sizes));
   blob_write_uint32(blob, pipeline->stack_size);
   blob_write_bytes(blob, pipeline->groups,
                    pipeline->group_count * sizeof(pipeline->groups[0]));
   for (uint32_t i = 0; i < pipeline->stage_count; ++i) {
      struct blob nir;
      blob_init(&nir);
      nir_serialize(&nir, pipeline->stages[i]->nir, false);
      blob_write_uint32(blob, nir.size);
      blob_write_bytes(blob, nir.data, nir.size);
      bool ok = !nir.out_of_memory && nir.size <= UINT32_MAX;
      blob_finish(&nir);
      if (!ok)
         return false;
   }
   if (pipeline->shader && !pipeline->shader->vk.ops->serialize(
                              &dev->vk, &pipeline->shader->vk, blob))
      return false;
   if (pipeline->kernel_count) {
      blob_write_uint32(blob, pipeline->kernel_count);
      for (uint32_t i = 0; i < pipeline->kernel_count; ++i) {
         const struct vk_shader *shader = &pipeline->kernels[i].shader->vk;
         blob_write_uint32(blob, pipeline->kernels[i].raygen_handle_index);
         if (!shader->ops->serialize(&dev->vk, shader, blob))
            return false;
      }
   }
   return !blob->out_of_memory;
}

/* RT pipelines use the pipeline cache on G17P, where they are tested. */
static bool
hk_rt_cache_supported(const struct hk_device *dev)
{
   return dev->dev.chip == AGX_CHIP_G17P;
}

static struct vk_pipeline_cache *
hk_rt_pipeline_cache(struct hk_device *dev, VkPipelineCache handle,
                     VkPipelineCreateFlags2KHR flags)
{
   VK_FROM_HANDLE(vk_pipeline_cache, cache, handle);
   /* Cached pipelines carry no internal representations. */
   if (!hk_rt_cache_supported(dev) ||
       (flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR))
      return NULL;
   return cache ? cache : dev->vk.mem_cache;
}

/* A cache entry holds hk_rt_binary_serialize() data under the pipeline key
 * (hk_rt_pipeline_key: every input that changes the code) and the shader
 * binary UUID. An executable pipeline also has the handles of the library
 * groups it links compiled in, so those must match the libraries' handles. */
static void
hk_rt_cache_key(struct hk_device *dev, const uint8_t *binary_key,
                const VkRayTracingPipelineCreateInfoKHR *info,
                uint8_t key[BLAKE3_OUT_LEN])
{
   static const char tag[] = "hk ray tracing pipeline cache 1";
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   _mesa_blake3_update(&hash, tag, sizeof(tag));
   _mesa_blake3_update(&hash, dev->vk.physical->properties.shaderBinaryUUID,
                       VK_UUID_SIZE);
   _mesa_blake3_update(&hash, binary_key, BLAKE3_OUT_LEN);
   if (!(vk_rt_pipeline_create_flags(info) &
         VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) &&
       info->pLibraryInfo) {
      for (uint32_t l = 0; l < info->pLibraryInfo->libraryCount; ++l) {
         VK_FROM_HANDLE(hk_ray_tracing_pipeline, lib,
                        info->pLibraryInfo->pLibraries[l]);
         for (uint32_t g = 0; g < lib->group_count; ++g)
            _mesa_blake3_update(&hash, &lib->groups[g].handle.index,
                                sizeof(lib->groups[g].handle.index));
      }
   }
   _mesa_blake3_final(&hash, key);
}

/* One pipeline binary holding hk_rt_binary_serialize() data. */
static VkResult
hk_rt_create_binary(VkDevice device, const void *data, size_t size,
                    const VkAllocationCallbacks *alloc,
                    VkPipelineBinaryHandlesInfoKHR *out)
{
   VkPipelineBinaryKeyKHR key = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
      .keySize = BLAKE3_OUT_LEN};
   _mesa_blake3_compute(data, size, key.key);
   const VkPipelineBinaryDataKHR binary_data = {.dataSize = size,
                                                .pData = (void *)data};
   const VkPipelineBinaryKeysAndDataKHR keys = {
      .binaryCount = 1,
      .pPipelineBinaryKeys = &key,
      .pPipelineBinaryData = &binary_data};
   const VkPipelineBinaryCreateInfoKHR wrapped = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
      .pKeysAndDataInfo = &keys};
   return vk_common_CreatePipelineBinariesKHR(device, &wrapped, alloc, out);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreatePipelineBinariesKHR(VkDevice device,
                             const VkPipelineBinaryCreateInfoKHR *info,
                             const VkAllocationCallbacks *alloc,
                             VkPipelineBinaryHandlesInfoKHR *out)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   if (!hk_rt_binaries_native(dev))
      return vk_common_CreatePipelineBinariesKHR(device, info, alloc, out);

   const VkRayTracingPipelineCreateInfoKHR *rt =
      info->pPipelineCreateInfo
         ? vk_find_struct_const(info->pPipelineCreateInfo->pNext,
                                RAY_TRACING_PIPELINE_CREATE_INFO_KHR)
         : NULL;
   if (rt) {
      /* The internal cache is the device's memory cache, as for the common
       * graphics and compute binaries. */
      struct vk_pipeline_cache_object *object = NULL;
      if (hk_rt_cache_supported(dev) && dev->vk.mem_cache &&
          dev->vk.physical->properties.pipelineBinaryInternalCache) {
         uint8_t binary_key[BLAKE3_OUT_LEN], cache_key[BLAKE3_OUT_LEN];
         hk_rt_pipeline_key(dev, rt, binary_key);
         hk_rt_cache_key(dev, binary_key, rt, cache_key);
         object = vk_pipeline_cache_lookup_object(
            dev->vk.mem_cache, cache_key, sizeof(cache_key),
            &vk_raw_data_cache_object_ops, NULL);
      }
      if (!object) {
         out->pipelineBinaryCount = 0;
         return VK_PIPELINE_BINARY_MISSING_KHR;
      }
      VkResult result;
      if (!out->pPipelineBinaries) {
         out->pipelineBinaryCount = 1;
         result = VK_SUCCESS;
      } else if (out->pipelineBinaryCount == 0) {
         result = VK_INCOMPLETE;
      } else {
         const struct vk_raw_data_cache_object *data =
            container_of(object, struct vk_raw_data_cache_object, base);
         result =
            hk_rt_create_binary(device, data->data, data->data_size, alloc, out);
      }
      vk_pipeline_cache_object_unref(&dev->vk, object);
      return result;
   }
   VK_FROM_HANDLE(vk_pipeline, base, info->pipeline);
   if (!base || base->ops != &hk_ray_tracing_pipeline_ops)
      return vk_common_CreatePipelineBinariesKHR(device, info, alloc, out);
   if (!out->pPipelineBinaries) {
      out->pipelineBinaryCount = 1;
      return VK_SUCCESS;
   }
   if (out->pipelineBinaryCount == 0)
      return VK_INCOMPLETE;
   struct blob blob;
   blob_init(&blob);
   if (!hk_rt_binary_serialize(dev, hk_pipeline_to_ray_tracing(base), &blob)) {
      blob_finish(&blob);
      out->pipelineBinaryCount = 0;
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   VkResult result = hk_rt_create_binary(device, blob.data, blob.size, alloc, out);
   blob_finish(&blob);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetPipelineKeyKHR(VkDevice device, const VkPipelineCreateInfoKHR *info,
                     VkPipelineBinaryKeyKHR *key)
{
   if (!info ||
       !vk_find_struct_const(info->pNext, RAY_TRACING_PIPELINE_CREATE_INFO_KHR))
      return vk_common_GetPipelineKeyKHR(device, info, key);
   VK_FROM_HANDLE(hk_device, dev, device);
   if (!hk_rt_binaries_native(dev))
      return vk_common_GetPipelineKeyKHR(device, info, key);

   const VkRayTracingPipelineCreateInfoKHR *rt =
      vk_find_struct_const(info->pNext, RAY_TRACING_PIPELINE_CREATE_INFO_KHR);
   hk_rt_pipeline_key(dev, rt, key->key);
   key->keySize = BLAKE3_OUT_LEN;
   return VK_SUCCESS;
}

/* Fill a newly allocated pipeline from hk_rt_binary_serialize() data. On
 * failure the pipeline holds partial state and must be destroyed. */
static VkResult
hk_rt_binary_load(struct hk_device *dev, struct hk_ray_tracing_pipeline *pipeline,
                  const void *data, size_t size,
                  const VkAllocationCallbacks *alloc)
{
   VkResult result = VK_ERROR_UNKNOWN;
   struct blob_reader blob;
   blob_reader_init(&blob, data, size);
   if (blob_read_uint32(&blob) != HK_RT_BINARY_MAGIC ||
       blob_read_uint32(&blob) != HK_RT_BINARY_VERSION)
      goto done;
   uint8_t uuid[VK_UUID_SIZE];
   blob_copy_bytes(&blob, uuid, sizeof(uuid));
   if (blob.overrun ||
       memcmp(uuid, dev->vk.physical->properties.shaderBinaryUUID,
              sizeof(uuid)))
      goto done;
   blob_copy_bytes(&blob, pipeline->binary_key, sizeof(pipeline->binary_key));
   uint32_t stages = blob_read_uint32(&blob);
   uint32_t groups = blob_read_uint32(&blob);
   pipeline->base.stages = blob_read_uint32(&blob);
   const uint32_t executable = blob_read_uint32(&blob);
   const uint32_t kernels_kind = agx_chip_g17_m5(dev->dev.chip)
                                    ? HK_RT_BINARY_RAYGEN_KERNELS
                                    : HK_RT_BINARY_MEGAKERNEL;
   if (blob.overrun ||
       (executable != HK_RT_BINARY_LIBRARY && executable != kernels_kind) ||
       (executable != HK_RT_BINARY_LIBRARY) ==
          !!(pipeline->base.flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) ||
       stages > size / sizeof(uint32_t) ||
       groups > size / sizeof(pipeline->groups[0]))
      goto done;
   blob_copy_bytes(&blob, &pipeline->stack_sizes,
                   sizeof(pipeline->stack_sizes));
   pipeline->stack_size = blob_read_uint32(&blob);
   pipeline->stages = calloc(stages, sizeof(pipeline->stages[0]));
   pipeline->groups = calloc(groups, sizeof(pipeline->groups[0]));
   if ((stages && !pipeline->stages) || (groups && !pipeline->groups)) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto done;
   }
   pipeline->stage_count = stages;
   pipeline->group_count = groups;
   blob_copy_bytes(&blob, pipeline->groups,
                   groups * sizeof(pipeline->groups[0]));
   if (blob.overrun)
      goto done;
   for (uint32_t i = 0; i < stages; ++i) {
      uint32_t bytes = blob_read_uint32(&blob);
      const void *nir_data = blob_read_bytes(&blob, bytes);
      if (blob.overrun)
         goto done;
      struct blob_reader nir_blob;
      blob_reader_init(&nir_blob, nir_data, bytes);
      nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &nir_blob);
      if (!nir || nir_blob.overrun || nir_blob.current != nir_blob.end) {
         ralloc_free(nir);
         goto done;
      }
      pipeline->stages[i] = hk_rt_stage_nir_create(nir);
      if (!pipeline->stages[i]) {
         ralloc_free(nir);
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto done;
      }
   }
   if (executable == HK_RT_BINARY_MEGAKERNEL) {
      struct vk_shader *shader = NULL;
      result = dev->vk.shader_ops->deserialize(
         &dev->vk, &blob, dev->vk.physical->properties.shaderBinaryVersion,
         alloc, &shader);
      if (result != VK_SUCCESS)
         goto done;
      pipeline->shader = container_of(shader, struct hk_api_shader, vk);
   } else if (executable == HK_RT_BINARY_RAYGEN_KERNELS) {
      const uint32_t kernel_count = blob_read_uint32(&blob);
      if (blob.overrun || kernel_count == 0 || kernel_count > groups)
         goto done;
      pipeline->kernels = calloc(kernel_count, sizeof(*pipeline->kernels));
      if (!pipeline->kernels) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto done;
      }
      for (uint32_t i = 0; i < kernel_count; ++i) {
         const uint32_t handle = blob_read_uint32(&blob);
         struct vk_shader *shader = NULL;
         if (blob.overrun) {
            result = VK_ERROR_UNKNOWN;
            goto done;
         }
         result = dev->vk.shader_ops->deserialize(
            &dev->vk, &blob, dev->vk.physical->properties.shaderBinaryVersion,
            alloc, &shader);
         if (result != VK_SUCCESS)
            goto done;
         pipeline->kernels[i].shader =
            container_of(shader, struct hk_api_shader, vk);
         pipeline->kernels[i].raygen_handle_index = handle;
         pipeline->kernel_count++;
      }
   }
   result =
      blob.overrun || blob.current != blob.end ? VK_ERROR_UNKNOWN : VK_SUCCESS;
done:
   return result;
}

static VkResult
hk_rt_binary_deserialize(struct hk_device *dev,
                         struct hk_ray_tracing_pipeline *pipeline,
                         VkPipelineBinaryKHR binary,
                         const VkAllocationCallbacks *alloc)
{
   const VkPipelineBinaryDataInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
      .pipelineBinary = binary};
   VkPipelineBinaryKeyKHR key = {.sType =
                                    VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
   size_t size = 0;
   VkResult result = vk_common_GetPipelineBinaryDataKHR(
      hk_device_to_handle(dev), &info, &key, &size, NULL);
   if (result != VK_SUCCESS)
      return result;
   void *data = malloc(size);
   if (!data)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   result = vk_common_GetPipelineBinaryDataKHR(hk_device_to_handle(dev), &info,
                                               &key, &size, data);
   if (result == VK_SUCCESS) {
      uint8_t digest[BLAKE3_OUT_LEN];
      _mesa_blake3_compute(data, size, digest);
      if (key.keySize != sizeof(digest) ||
          memcmp(key.key, digest, sizeof(digest)))
         result = VK_ERROR_UNKNOWN;
      else
         result = hk_rt_binary_load(dev, pipeline, data, size, alloc);
   }
   free(data);
   return result;
}

/* The pipeline keeps a reference to its entry, which is what keeps it in a
 * weak cache such as the device's memory cache. */
static void
hk_rt_cache_insert(struct hk_device *dev, struct vk_pipeline_cache *cache,
                   const uint8_t *key, struct hk_ray_tracing_pipeline *pipeline)
{
   struct blob blob;
   blob_init(&blob);
   if (hk_rt_binary_serialize(dev, pipeline, &blob)) {
      struct vk_raw_data_cache_object *object = vk_raw_data_cache_object_create(
         &dev->vk, key, BLAKE3_OUT_LEN, blob.data, blob.size);
      if (object)
         pipeline->cache_object =
            vk_pipeline_cache_add_object(cache, &object->base);
   }
   blob_finish(&blob);
}

/* Group handles of a pipeline loaded from a cache. An executable keeps the
 * handles compiled into it. A library takes the handles an identical library
 * got on this device (or new ones) and those of the libraries it links now,
 * never handles issued on another device. */
static bool
hk_rt_cache_loaded_handles(struct hk_device *dev,
                           struct hk_ray_tracing_pipeline *pipeline,
                           const VkRayTracingPipelineCreateInfoKHR *info)
{
   const VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(info);
   const uint32_t libs = info->pLibraryInfo ? info->pLibraryInfo->libraryCount : 0;
   uint32_t group_count = info->groupCount;
   for (uint32_t l = 0; l < libs; ++l) {
      VK_FROM_HANDLE(hk_ray_tracing_pipeline, lib,
                     info->pLibraryInfo->pLibraries[l]);
      group_count += lib->group_count;
   }
   if (group_count != pipeline->group_count)
      return false;

   if (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      if (!hk_rt_library_handles_find(dev, pipeline->binary_key,
                                      pipeline->groups, info->groupCount)) {
         for (uint32_t i = 0; i < info->groupCount; ++i) {
            /* Replay handles are part of the key, so the entry has them. */
            if (!(flags &
                  VK_PIPELINE_CREATE_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR) ||
                !info->pGroups[i].pShaderGroupCaptureReplayHandle) {
               pipeline->groups[i].handle.index =
                  p_atomic_inc_return(&dev->rt_group_handle_alloc);
            }
         }
         hk_rt_library_handles_record(dev, pipeline->binary_key,
                                      pipeline->groups, info->groupCount);
      }
      uint32_t i = info->groupCount;
      for (uint32_t l = 0; l < libs; ++l) {
         VK_FROM_HANDLE(hk_ray_tracing_pipeline, lib,
                        info->pLibraryInfo->pLibraries[l]);
         for (uint32_t g = 0; g < lib->group_count; ++g)
            pipeline->groups[i++].handle = lib->groups[g].handle;
      }
   }

   for (uint32_t i = 0; i < pipeline->group_count; ++i)
      hk_rt_handle_alloc_above(dev, pipeline->groups[i].handle.index);
   return true;
}

static struct hk_ray_tracing_pipeline *
hk_rt_pipeline_alloc(struct hk_device *dev, VkPipelineCreateFlags2KHR flags,
                     struct vk_pipeline_layout *layout,
                     const VkAllocationCallbacks *pAllocator)
{
   struct hk_ray_tracing_pipeline *pipeline = vk_pipeline_zalloc(
      &dev->vk, &hk_ray_tracing_pipeline_ops,
      VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, flags, pAllocator,
      sizeof(*pipeline));
   if (pipeline && layout) {
      vk_pipeline_layout_ref(layout);
      pipeline->layout = layout;
   }
   return pipeline;
}

static VkResult
hk_create_ray_tracing_pipeline(
   struct hk_device *dev, struct vk_pipeline_cache *cache,
   const VkAllocationCallbacks *pAllocator,
   const VkRayTracingPipelineCreateInfoKHR *create_info,
   VkPipeline *out_pipeline)
{
   VK_FROM_HANDLE(vk_pipeline_layout, layout, create_info->layout);
   VkResult result;

   const VkPipelineCreateFlags2KHR flags =
      vk_rt_pipeline_create_flags(create_info);

   /* Keep the existing defensive ceiling before key construction or NIR
    * cloning. G17P advertises 31 levels; legacy generations three. */
   const uint32_t max_depth = dev->dev.chip == AGX_CHIP_G17P ?
      HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH : HK_MAX_RAY_RECURSION_DEPTH;
   if (create_info->maxPipelineRayRecursionDepth > max_depth)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "maxPipelineRayRecursionDepth %u exceeds %u",
                       create_info->maxPipelineRayRecursionDepth, max_depth);

   struct hk_ray_tracing_pipeline *pipeline =
      hk_rt_pipeline_alloc(dev, flags, layout, pAllocator);
   if (!pipeline)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   const VkPipelineBinaryInfoKHR *binaries =
      vk_find_struct_const(create_info->pNext, PIPELINE_BINARY_INFO_KHR);
   if (hk_rt_binaries_native(dev) && binaries && binaries->binaryCount) {
      if (binaries->binaryCount != 1) {
         result = VK_ERROR_UNKNOWN;
         goto fail;
      }
      result = hk_rt_binary_deserialize(
         dev, pipeline, binaries->pPipelineBinaries[0], pAllocator);
      if (result != VK_SUCCESS)
         goto fail;
      *out_pipeline = vk_pipeline_to_handle(&pipeline->base);
      return VK_SUCCESS;
   }
   if (hk_rt_binaries_native(dev))
      hk_rt_pipeline_key(dev, create_info, pipeline->binary_key);

   uint8_t cache_key[BLAKE3_OUT_LEN];
   if (cache) {
      hk_rt_cache_key(dev, pipeline->binary_key, create_info, cache_key);
      struct vk_pipeline_cache_object *object = vk_pipeline_cache_lookup_object(
         cache, cache_key, sizeof(cache_key), &vk_raw_data_cache_object_ops,
         NULL);
      if (object) {
         const struct vk_raw_data_cache_object *data =
            container_of(object, struct vk_raw_data_cache_object, base);
         uint8_t binary_key[BLAKE3_OUT_LEN];
         memcpy(binary_key, pipeline->binary_key, sizeof(binary_key));
         result = hk_rt_binary_load(dev, pipeline, data->data,
                                    data->data_size, pAllocator);
         pipeline->cache_object = object;
         if (result == VK_SUCCESS &&
             !memcmp(binary_key, pipeline->binary_key, sizeof(binary_key)) &&
             hk_rt_cache_loaded_handles(dev, pipeline, create_info)) {
            *out_pipeline = vk_pipeline_to_handle(&pipeline->base);
            return VK_SUCCESS;
         }
         /* Not a usable entry: build the pipeline instead. */
         hk_ray_tracing_pipeline_destroy(&dev->vk, &pipeline->base, pAllocator);
         pipeline = hk_rt_pipeline_alloc(dev, flags, layout, pAllocator);
         if (!pipeline)
            return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         memcpy(pipeline->binary_key, binary_key, sizeof(binary_key));
      }
   }

   pipeline->stage_count = create_info->stageCount;
   pipeline->group_count = create_info->groupCount;

   if (create_info->pLibraryInfo) {
      for (uint32_t i = 0; i < create_info->pLibraryInfo->libraryCount; i++) {
         VK_FROM_HANDLE(vk_pipeline, vk_library,
                        create_info->pLibraryInfo->pLibraries[i]);
         struct hk_ray_tracing_pipeline *library =
            hk_pipeline_to_ray_tracing(vk_library);

         pipeline->stage_count += library->stage_count;
         pipeline->group_count += library->group_count;
      }
   }

   if (pipeline->stage_count) {
      pipeline->stages =
         calloc(pipeline->stage_count, sizeof(*pipeline->stages));
      if (!pipeline->stages) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }
   }

   if (pipeline->group_count) {
      pipeline->groups =
         calloc(pipeline->group_count, sizeof(*pipeline->groups));
      if (!pipeline->groups) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }
   }

   result = hk_compile_ray_tracing_stages(dev, pipeline, create_info);
   if (result != VK_SUCCESS)
      goto fail;

   result = hk_init_ray_tracing_groups(dev, pipeline, create_info);
   if (result != VK_SUCCESS)
      goto fail;

   if ((flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) &&
       hk_rt_cache_supported(dev)) {
      hk_rt_library_handles_record(dev, pipeline->binary_key, pipeline->groups,
                                   create_info->groupCount);
   }

   for (uint32_t i = 0; i < pipeline->stage_count; i++)
      pipeline->base.stages |= mesa_to_vk_shader_stage(
         pipeline->stages[i]->nir->info.stage);

   /* A library is a container of stages and groups; it is never executable and
    * compiling a megakernel for it would be wasted work. */
   if (!(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      /* Libraries retain stage NIR, not a compiled megakernel.  Linking an
       * executable without a cache hit still requires backend compilation,
       * even with no new stages.
       */
      if (flags &
          VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR) {
         result = VK_PIPELINE_COMPILE_REQUIRED;
         goto fail;
      }

      result = hk_compile_ray_tracing_pipeline(dev, pipeline, create_info,
                                               pAllocator);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (cache)
      hk_rt_cache_insert(dev, cache, cache_key, pipeline);

   *out_pipeline = vk_pipeline_to_handle(&pipeline->base);
   return VK_SUCCESS;

fail:
   hk_ray_tracing_pipeline_destroy(&dev->vk, &pipeline->base, pAllocator);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateRayTracingPipelinesKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache, uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VkResult result = VK_SUCCESS;

   uint32_t i = 0;
   for (; i < createInfoCount; i++) {
      VkResult one = hk_create_ray_tracing_pipeline(
         dev,
         hk_rt_pipeline_cache(dev, pipelineCache,
                              vk_rt_pipeline_create_flags(&pCreateInfos[i])),
         pAllocator, pCreateInfos + i, pPipelines + i);

      if (one != VK_SUCCESS) {
         result = one;
         pPipelines[i] = VK_NULL_HANDLE;

         if (vk_rt_pipeline_create_flags(&pCreateInfos[i]) &
             VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR) {
            i++;
            break;
         }
      }
   }

   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetRayTracingShaderGroupHandlesKHR(VkDevice _device, VkPipeline _pipeline,
                                      uint32_t firstGroup, uint32_t groupCount,
                                      size_t dataSize, void *pData)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, _pipeline);
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   uint8_t *data = pData;

   assert(dataSize >= (size_t)groupCount * HK_RAY_TRACING_GROUP_HANDLE_SIZE);

   for (uint32_t i = 0; i < groupCount; i++) {
      hk_rt_write_group_handle(data + i * HK_RAY_TRACING_GROUP_HANDLE_SIZE,
                               pipeline->groups[firstGroup + i].handle);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice _device, VkPipeline _pipeline, uint32_t firstGroup,
   uint32_t groupCount, size_t dataSize, void *pData)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, _pipeline);
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);
   uint8_t *data = pData;

   assert(dataSize >= (size_t)groupCount * sizeof(uint32_t));

   for (uint32_t i = 0; i < groupCount; i++) {
      const uint32_t index = pipeline->groups[firstGroup + i].handle.index;
      memcpy(data + i * sizeof(uint32_t), &index, sizeof(index));
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
hk_GetRayTracingShaderGroupStackSizeKHR(VkDevice _device, VkPipeline _pipeline,
                                        uint32_t group,
                                        VkShaderGroupShaderKHR groupShader)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, _pipeline);
   struct hk_ray_tracing_pipeline *pipeline =
      hk_pipeline_to_ray_tracing(vk_pipeline);

   if (group >= pipeline->group_count)
      return 0;

   const struct hk_ray_tracing_group *g = pipeline->groups + group;
   uint32_t recursive_size = 0;

   if (g->recursive_index != VK_SHADER_UNUSED_KHR) {
      switch (pipeline->stages[g->recursive_index]->nir->info.stage) {
      case MESA_SHADER_RAYGEN:
         recursive_size = pipeline->stack_sizes.raygen;
         break;
      case MESA_SHADER_CLOSEST_HIT:
         recursive_size = pipeline->stack_sizes.closest_hit;
         break;
      case MESA_SHADER_MISS:
         recursive_size = pipeline->stack_sizes.miss;
         break;
      case MESA_SHADER_CALLABLE:
         recursive_size = pipeline->stack_sizes.callable;
         break;
      default:
         recursive_size = 0;
         break;
      }
   }

   return hk_rt_group_stack_size(&pipeline->stack_sizes, g, groupShader,
                                 recursive_size);
}
