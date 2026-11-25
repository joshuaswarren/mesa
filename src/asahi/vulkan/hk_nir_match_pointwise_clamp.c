/* SPDX-License-Identifier: MIT */
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_nir_fast_udiv.h"
#include "hk_nir_wgid.h"

#include <math.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"

bool
hk_nir_clamp_epilogue_valid(const struct hk_nir_clamp_epilogue *e)
{
   if (!e || (!e->has_lower && !e->has_upper))
      return false;
   if (e->has_lower &&
       (!isfinite(uif(e->lower_bits)) ||
        (e->max_fp_math_ctrl & ~nir_op_infos[nir_op_fmax].valid_fp_math_ctrl)))
      return false;
   if (e->has_upper &&
       (!isfinite(uif(e->upper_bits)) ||
        (e->min_fp_math_ctrl & ~nir_op_infos[nir_op_fmin].valid_fp_math_ctrl)))
      return false;
   return true;
}

static unsigned
storage_format_components(nir_address_format format)
{
   switch (format) {
   case nir_address_format_64bit_bounded_global_map:
   case nir_address_format_64bit_global_32bit_offset_map:
   case nir_address_format_64bit_bounded_global:
   case nir_address_format_64bit_global_32bit_offset:
      return nir_address_format_num_components(format);
   default:
      return 0;
   }
}

static const nir_intrinsic_instr *
intrinsic(const nir_def *def, nir_intrinsic_op op)
{
   if (!def || !nir_def_is_intrinsic(def))
      return NULL;

   const nir_intrinsic_instr *instr = nir_def_as_intrinsic(def);
   return instr->intrinsic == op ? instr : NULL;
}

static const nir_deref_instr *
deref(nir_src src, nir_deref_type type, nir_variable_mode mode)
{
   if (!src.ssa || !nir_def_is_deref(src.ssa))
      return NULL;

   const nir_deref_instr *instr = nir_def_as_deref(src.ssa);
   return instr->deref_type == type && nir_deref_mode_is(instr, mode)
          ? instr : NULL;
}

static bool
alu_op(nir_scalar value, nir_op op)
{
   return value.def && nir_scalar_is_alu(value) &&
          nir_scalar_alu_op(value) == op;
}

static bool
scalar_op(nir_scalar value, nir_op op)
{
   return value.def && value.def->num_components == 1 &&
          nir_scalar_is_alu(value) && nir_scalar_alu_op(value) == op;
}

static nir_scalar
source(nir_src src)
{
   return nir_scalar_chase_movs(nir_get_scalar(src.ssa, 0));
}

static bool
constant(nir_scalar val, uint32_t bits)
{
   val = nir_scalar_chase_movs(val);
   return val.def && nir_scalar_is_const(val) && nir_scalar_as_uint(val) == bits;
}

static nir_scalar
strip_zero_add(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   for (unsigned i = 0; i < 3 && alu_op(s, nir_op_iadd); ++i) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (constant(a, 0))
         s = nir_scalar_chase_movs(b);
      else if (constant(b, 0))
         s = nir_scalar_chase_movs(a);
      else
         break;
   }
   return s;
}

static bool
plain_load(const nir_intrinsic_instr *instr)
{
   const unsigned allowed = ACCESS_NON_WRITEABLE | ACCESS_RESTRICT |
                            ACCESS_CAN_REORDER | ACCESS_IN_BOUNDS;
   return instr && !(nir_intrinsic_access(instr) & ~allowed);
}

/* Accept load_workgroup_id[axis] as GlobalInvocationID[axis] when
 * local_size[axis] == 1 (so LocalInvocationID[axis] == 0).
 *
 * NOTE: The bare load_workgroup_id[axis] form (without + load_base_workgroup_id)
 * equals GlobalInvocationID[axis] ONLY when base_group[axis] == 0. This matcher
 * is sound only together with the dispatch-time nonzero-base refusal gates
 * (hk_nir_dispatch_zero_base() in hk_cmd_dispatch.c:hk_try_record_compute_pair()
 * and hk_select_depthwise_vec4(), plus dispatch_covers() below), all of which
 * fall back to the original shader whenever any vkCmdDispatchBase offset is
 * nonzero. */
static bool
is_global_id_intr(nir_scalar s, const uint16_t local_size[3])
{
   return s.def && s.def->bit_size == 32 && s.comp < 3 &&
          (intrinsic(s.def, nir_intrinsic_load_global_invocation_id) ||
           (local_size && local_size[s.comp] == 1 &&
            intrinsic(s.def, nir_intrinsic_load_workgroup_id)));
}

/* Recognizes a component of GlobalInvocationID or LocalInvocationID on either
 * pre-preprocess NIR (load_deref of nir_var_system_value) or post-preprocess
 * NIR (load_global_invocation_id or unit-axis load_workgroup_id
 * [+ load_base_workgroup_id * local_size], load_local_invocation_id). */
static bool
sysval_id_axis(nir_scalar value, gl_system_value loc,
               const uint16_t local_size[3], unsigned *axis_out)
{
   value = strip_zero_add(value);
   if (!value.def || value.def->bit_size != 32)
      return false;

   if (value.comp < 3 && value.def->num_components == 3) {
      const nir_intrinsic_instr *load = intrinsic(value.def, nir_intrinsic_load_deref);
      if (plain_load(load)) {
         const nir_deref_instr *var = deref(load->src[0], nir_deref_type_var,
                                            nir_var_system_value);
         if (var && var->var->data.location == loc) {
            *axis_out = value.comp;
            return true;
         }
      }
   }

   if (loc == SYSTEM_VALUE_GLOBAL_INVOCATION_ID) {
      if (is_global_id_intr(value, local_size)) {
         *axis_out = value.comp;
         return true;
      }
      if (local_size && alu_op(value, nir_op_iadd)) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(value, 0));
         nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(value, 1));
         if (is_global_id_intr(a, local_size) &&
             hk_nir_match_base_wg_scaled(b, a.comp, local_size[a.comp])) {
            *axis_out = a.comp;
            return true;
         }
         if (is_global_id_intr(b, local_size) &&
             hk_nir_match_base_wg_scaled(a, b.comp, local_size[b.comp])) {
            *axis_out = b.comp;
            return true;
         }
      }
   } else if (loc == SYSTEM_VALUE_LOCAL_INVOCATION_ID) {
      if (value.comp < 3 &&
          intrinsic(value.def, nir_intrinsic_load_local_invocation_id)) {
         *axis_out = value.comp;
         return true;
      }
   }

   return false;
}

static bool
is_global_id(nir_scalar value, const uint16_t local_size[3], unsigned axis)
{
   unsigned got = 0;
   return sysval_id_axis(value, SYSTEM_VALUE_GLOBAL_INVOCATION_ID,
                         local_size, &got) && got == axis;
}

/* Recognizes a 32-bit scalar push-constant read at a 4-byte-aligned byte
 * offset, on either pre-preprocess load_deref(nir_var_mem_push_const) or
 * lowered load_push_constant. */
static bool
push_count(nir_scalar value, uint32_t *offset_B)
{
   value = nir_scalar_chase_movs(value);
   if (!value.def || value.comp != 0 || value.def->bit_size != 32 ||
       value.def->num_components != 1)
      return false;

   const nir_intrinsic_instr *pc =
      intrinsic(value.def, nir_intrinsic_load_push_constant);
   if (pc) {
      if (nir_intrinsic_access(pc) != 0)
         return false;
      nir_scalar off = source(pc->src[0]);
      if (!nir_scalar_is_const(off))
         return false;
      uint64_t byte_off = (uint64_t)nir_intrinsic_base(pc) + nir_scalar_as_uint(off);
      if ((byte_off & 3) || byte_off > UINT32_MAX - 4)
         return false;
      *offset_B = (uint32_t)byte_off;
      return true;
   }

   const nir_intrinsic_instr *load = intrinsic(value.def, nir_intrinsic_load_deref);
   if (!plain_load(load))
      return false;

   const nir_deref_instr *field = deref(load->src[0], nir_deref_type_struct,
                                        nir_var_mem_push_const);
   if (!field || !glsl_type_is_scalar(field->type) ||
       (glsl_get_base_type(field->type) != GLSL_TYPE_INT &&
        glsl_get_base_type(field->type) != GLSL_TYPE_UINT))
      return false;

   const nir_deref_instr *var = deref(field->parent, nir_deref_type_var,
                                      nir_var_mem_push_const);
   if (!var)
      return false;

   int offset = glsl_get_struct_field_offset(var->type, field->strct.index);
   if (offset < 0 || (offset & 3))
      return false;

   *offset_B = (uint32_t)offset;
   return true;
}

static bool
same_with_local(nir_scalar a, nir_scalar b, const uint16_t local_size[3])
{
   a = strip_zero_add(a);
   b = strip_zero_add(b);
   if (!a.def || !b.def)
      return false;
   if (a.def == b.def && a.comp == b.comp)
      return true;

   unsigned ax_a = 0, ax_b = 0;
   if (sysval_id_axis(a, SYSTEM_VALUE_GLOBAL_INVOCATION_ID, local_size, &ax_a) &&
       sysval_id_axis(b, SYSTEM_VALUE_GLOBAL_INVOCATION_ID, local_size, &ax_b))
      return ax_a == ax_b;

   uint32_t a_off = 0, b_off = 0;
   if (push_count(a, &a_off) && push_count(b, &b_off))
      return a_off == b_off;

   return false;
}

struct buffer_access {
   const nir_intrinsic_instr *index;
   const nir_intrinsic_instr *descriptor;
   nir_scalar element;
   struct hk_nir_clamp_buffer binding;
};

static bool
same_binding(struct hk_nir_clamp_buffer a, struct hk_nir_clamp_buffer b)
{
   return a.set == b.set && a.binding == b.binding && a.offset_B == b.offset_B;
}

static bool
buffer(nir_src pointer, struct buffer_access *out)
{
   const nir_deref_instr *element = deref(pointer, nir_deref_type_array,
                                          nir_var_mem_ssbo);
   if (!element || !glsl_type_is_scalar(element->type) ||
       !glsl_type_is_float(element->type))
      return false;

   nir_scalar index =
      nir_scalar_chase_movs(nir_get_scalar(element->arr.index.ssa, 0));
   if (index.def->bit_size != 32)
      return false;

   const nir_deref_instr *field = deref(element->parent, nir_deref_type_struct,
                                        nir_var_mem_ssbo);
   if (!field || !glsl_type_is_array(field->type) ||
       glsl_get_explicit_stride(field->type) != 4)
      return false;

   const nir_deref_instr *cast = deref(field->parent, nir_deref_type_cast,
                                       nir_var_mem_ssbo);
   if (!cast)
      return false;

   int offset = glsl_get_struct_field_offset(cast->type, field->strct.index);
   if (offset < 0 || (offset & 3))
      return false;

   const nir_intrinsic_instr *descriptor =
      intrinsic(cast->parent.ssa, nir_intrinsic_load_vulkan_descriptor);
   if (!descriptor ||
       nir_intrinsic_desc_type(descriptor) != nir_descriptor_type_storage_buffer)
      return false;

   const nir_intrinsic_instr *resource =
      intrinsic(descriptor->src[0].ssa, nir_intrinsic_vulkan_resource_index);
   if (!resource ||
       nir_intrinsic_desc_type(resource) != nir_descriptor_type_storage_buffer ||
       !nir_src_is_const(resource->src[0]) ||
       nir_src_as_uint(resource->src[0]) != 0)
      return false;

   *out = (struct buffer_access){
      .index = resource,
      .descriptor = descriptor,
      .element = index,
      .binding = {
         .set = nir_intrinsic_desc_set(resource),
         .binding = nir_intrinsic_binding(resource),
         .offset_B = (uint32_t)offset,
      },
   };
   return true;
}

static bool
match_clamp_const_op(nir_scalar s, nir_op op, nir_scalar *var_out,
                     uint32_t *bits_out, uint32_t *fp_ctrl_out)
{
   s = nir_scalar_chase_movs(s);
   if (!scalar_op(s, op))
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   nir_scalar c = {0}, v = {0};
   if (nir_scalar_is_const(b) && !nir_scalar_is_const(a)) {
      v = a;
      c = b;
   } else if (nir_scalar_is_const(a) && !nir_scalar_is_const(b)) {
      v = b;
      c = a;
   } else {
      return false;
   }
   uint32_t bits = nir_scalar_as_uint(c);
   if (!isfinite(uif(bits)))
      return false;
   *var_out = v;
   *bits_out = bits;
   *fp_ctrl_out = nir_scalar_as_alu(s)->fp_math_ctrl;
   return true;
}

/* Match `gx < count` (polarity == true) or `gx >= count` (polarity == false),
 * including `inot` wrappers and signed/unsigned comparisons. */
static bool
match_1d_bounds_cond(nir_scalar cond, const uint16_t local_size[3],
                     bool want_in_bounds, uint32_t *count_offset_B)
{
   cond = nir_scalar_chase_movs(cond);
   while (scalar_op(cond, nir_op_inot)) {
      want_in_bounds = !want_in_bounds;
      cond = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   }

   if (want_in_bounds) {
      if (!scalar_op(cond, nir_op_ilt) && !scalar_op(cond, nir_op_ult))
         return false;
   } else {
      if (!scalar_op(cond, nir_op_ige) && !scalar_op(cond, nir_op_uge))
         return false;
   }

   nir_scalar lhs = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(cond, 1);
   return is_global_id(lhs, local_size, 0) && push_count(rhs, count_offset_B);
}

bool
hk_nir_match_pointwise_clamp(const nir_shader *shader,
                             struct hk_nir_pointwise_clamp *match)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable ||
       !shader->info.workgroup_size[0] ||
       shader->info.workgroup_size[1] != 1 ||
       shader->info.workgroup_size[2] != 1)
      return false;

   nir_function_impl *impl = NULL;
   nir_foreach_function_impl(candidate, shader) {
      if (impl || !candidate->function->is_entrypoint)
         return false;
      impl = candidate;
   }
   if (!impl)
      return false;

   const uint16_t *ls = shader->info.workgroup_size;
   const nir_intrinsic_instr *store = NULL;
   const nir_intrinsic_instr *read = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
         case nir_instr_type_load_const:
         case nir_instr_type_deref:
            continue;
         case nir_instr_type_intrinsic:
            break;
         default:
            return false;
         }

         const nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
         switch (op->intrinsic) {
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_push_constant:
            break;
         case nir_intrinsic_load_deref: {
            const nir_deref_instr *ptr =
               nir_def_is_deref(op->src[0].ssa)
                  ? nir_def_as_deref(op->src[0].ssa)
                  : NULL;
            if (!ptr)
               return false;
            if (nir_deref_mode_is(ptr, nir_var_mem_ssbo)) {
               if (read || !plain_load(op) || op->def.bit_size != 32 ||
                   op->def.num_components != 1)
                  return false;
               read = op;
            } else if (nir_deref_mode_is(ptr, nir_var_mem_push_const)) {
               uint32_t off;
               if (!push_count(nir_get_scalar((nir_def *)&op->def, 0), &off))
                  return false;
            } else if (nir_deref_mode_is(ptr, nir_var_system_value)) {
               unsigned axis;
               if (!sysval_id_axis(nir_get_scalar((nir_def *)&op->def, 0),
                                   SYSTEM_VALUE_GLOBAL_INVOCATION_ID, ls, &axis))
                  return false;
            } else {
               return false;
            }
            break;
         }
         case nir_intrinsic_store_deref:
            if (store || op->num_components != 1 ||
                op->src[1].ssa->bit_size != 32 ||
                nir_intrinsic_write_mask(op) != 1 ||
                (nir_intrinsic_access(op) &
                 ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS)))
               return false;
            store = op;
            break;
         default:
            return false;
         }
      }
   }
   if (!store || !read)
      return false;

   /* Store must be guarded by a single top-level if checking gx < count (then)
    * or gx >= count (else), with the other arm empty. */
   nir_cf_node *parent = store->instr.block->cf_node.parent;
   if (parent->type != nir_cf_node_if || parent->parent != &impl->cf_node)
      return false;
   nir_if *guard = nir_cf_node_as_if(parent);

   bool store_in_then = false;
   if (store->instr.block == nir_if_first_then_block(guard) &&
       nir_cf_node_is_last(&store->instr.block->cf_node) &&
       exec_list_is_empty(&nir_if_first_else_block(guard)->instr_list) &&
       nir_cf_node_is_last(&nir_if_first_else_block(guard)->cf_node)) {
      store_in_then = true;
   } else if (store->instr.block == nir_if_first_else_block(guard) &&
              nir_cf_node_is_last(&store->instr.block->cf_node) &&
              exec_list_is_empty(&nir_if_first_then_block(guard)->instr_list) &&
              nir_cf_node_is_last(&nir_if_first_then_block(guard)->cf_node)) {
      store_in_then = false;
   } else {
      return false;
   }

   /* The input load must be inside the same guarded block so it cannot fault
    * on out-of-bounds invocations. */
   if (read->instr.block != store->instr.block)
      return false;

   uint32_t count_offset_B = 0;
   if (!match_1d_bounds_cond(source(guard->condition), ls, store_in_then,
                             &count_offset_B))
      return false;

   struct buffer_access in, out;
   if (!buffer(read->src[0], &in) || !buffer(store->src[0], &out))
      return false;
   if (!is_global_id(in.element, ls, 0) || !is_global_id(out.element, ls, 0))
      return false;

   /* Match lower-only fmax(x, lo), upper-only fmin(x, hi),
    * fmin(fmax(x, lo), hi), or fmax(fmin(x, hi), lo). */
   struct hk_nir_clamp_epilogue ep = {
      .float_controls_execution_mode = shader->info.float_controls_execution_mode,
   };
   nir_scalar stored = source(store->src[1]);
   nir_scalar inner = {0}, base = {0};

   if (match_clamp_const_op(stored, nir_op_fmin, &inner,
                            &ep.upper_bits, &ep.min_fp_math_ctrl)) {
      ep.has_upper = true;
      if (match_clamp_const_op(inner, nir_op_fmax, &base,
                               &ep.lower_bits, &ep.max_fp_math_ctrl)) {
         ep.has_lower = true;
         ep.upper_first = false;
      } else {
         base = inner;
      }
   } else if (match_clamp_const_op(stored, nir_op_fmax, &inner,
                                   &ep.lower_bits, &ep.max_fp_math_ctrl)) {
      ep.has_lower = true;
      if (match_clamp_const_op(inner, nir_op_fmin, &base,
                               &ep.upper_bits, &ep.min_fp_math_ctrl)) {
         ep.has_upper = true;
         ep.upper_first = true;
      } else {
         base = inner;
      }
   } else {
      return false;
   }

   base = nir_scalar_chase_movs(base);
   if (base.def != &read->def || base.comp != 0)
      return false;

   if (!hk_nir_clamp_epilogue_valid(&ep))
      return false;

   *match = (struct hk_nir_pointwise_clamp){
      .input = in.binding,
      .output = out.binding,
      .count_offset_B = count_offset_B,
      .epilogue = ep,
      .local_size_x = shader->info.workgroup_size[0],
   };
   return true;
}

static nir_if *
block_if(nir_block *block)
{
   nir_cf_node *next = nir_cf_node_next(&block->cf_node);
   return next && next->type == nir_cf_node_if ? nir_cf_node_as_if(next) : NULL;
}

static bool
match_mul_pair(nir_scalar s, nir_scalar a, nir_scalar b, const uint16_t ls[3])
{
   s = nir_scalar_chase_movs(s);
   if (!scalar_op(s, nir_op_imul))
      return false;
   nir_scalar x = nir_scalar_chase_alu_src(s, 0);
   nir_scalar y = nir_scalar_chase_alu_src(s, 1);
   return (same_with_local(x, a, ls) && same_with_local(y, b, ls)) ||
          (same_with_local(x, b, ls) && same_with_local(y, a, ls));
}

static bool
match_div_op(nir_scalar s, nir_scalar num, nir_scalar den, const uint16_t ls[3])
{
   s = nir_scalar_chase_movs(s);
   if (!scalar_op(s, nir_op_idiv) && !scalar_op(s, nir_op_udiv))
      return false;
   return same_with_local(nir_scalar_chase_alu_src(s, 0), num, ls) &&
          same_with_local(nir_scalar_chase_alu_src(s, 1), den, ls);
}

static bool
match_rem_op(nir_scalar s, nir_scalar num, nir_scalar den, const uint16_t ls[3])
{
   s = nir_scalar_chase_movs(s);
   if (scalar_op(s, nir_op_irem) || scalar_op(s, nir_op_imod) ||
       scalar_op(s, nir_op_umod)) {
      return same_with_local(nir_scalar_chase_alu_src(s, 0), num, ls) &&
             same_with_local(nir_scalar_chase_alu_src(s, 1), den, ls);
   }
   /* Also match num - (num / den) * den or num + -((num / den) * den). */
   if (scalar_op(s, nir_op_isub)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (same_with_local(a, num, ls) && scalar_op(b, nir_op_imul)) {
         nir_scalar m0 = nir_scalar_chase_alu_src(b, 0);
         nir_scalar m1 = nir_scalar_chase_alu_src(b, 1);
         return (match_div_op(m0, num, den, ls) && same_with_local(m1, den, ls)) ||
                (match_div_op(m1, num, den, ls) && same_with_local(m0, den, ls));
      }
   }
   if (scalar_op(s, nir_op_iadd)) {
      for (unsigned side = 0; side < 2; ++side) {
         nir_scalar a = nir_scalar_chase_alu_src(s, side);
         nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1 - side));
         if (same_with_local(a, num, ls) && scalar_op(b, nir_op_ineg)) {
            nir_scalar p = nir_scalar_chase_movs(nir_scalar_chase_alu_src(b, 0));
            if (scalar_op(p, nir_op_imul)) {
               nir_scalar m0 = nir_scalar_chase_alu_src(p, 0);
               nir_scalar m1 = nir_scalar_chase_alu_src(p, 1);
               if ((match_div_op(m0, num, den, ls) && same_with_local(m1, den, ls)) ||
                   (match_div_op(m1, num, den, ls) && same_with_local(m0, den, ls)))
                  return true;
            }
         }
      }
   }
   return false;
}

static bool
match_ge_x_term(nir_scalar s, nir_scalar gx, nir_scalar width,
                nir_scalar channels, const uint16_t ls[3], bool is_lt)
{
   s = nir_scalar_chase_movs(s);
   if (is_lt) {
      if (!scalar_op(s, nir_op_ilt) && !scalar_op(s, nir_op_ult))
         return false;
   } else {
      if (!scalar_op(s, nir_op_ige) && !scalar_op(s, nir_op_uge))
         return false;
   }
   nir_scalar lhs = nir_scalar_chase_alu_src(s, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(s, 1);
   if (same_with_local(lhs, gx, ls) && match_mul_pair(rhs, width, channels, ls))
      return true;
   if (match_div_op(lhs, gx, channels, ls) && same_with_local(rhs, width, ls))
      return true;
   return false;
}

static bool
match_ge_y_term(nir_scalar s, nir_scalar gy, uint32_t *height_offset,
                const uint16_t ls[3], bool is_lt)
{
   s = nir_scalar_chase_movs(s);
   if (is_lt) {
      if (!scalar_op(s, nir_op_ilt) && !scalar_op(s, nir_op_ult))
         return false;
   } else {
      if (!scalar_op(s, nir_op_ige) && !scalar_op(s, nir_op_uge))
         return false;
   }
   return same_with_local(nir_scalar_chase_alu_src(s, 0), gy, ls) &&
          push_count(nir_scalar_chase_alu_src(s, 1), height_offset);
}

static bool
store_guard(nir_if *exit, bool body_in_then, nir_scalar gx, nir_scalar gy,
            nir_scalar width, nir_scalar channels, uint32_t *height_offset,
            const uint16_t ls[3])
{
   nir_scalar condition = source(exit->condition);
   bool want_in_bounds = body_in_then;
   while (scalar_op(condition, nir_op_inot)) {
      want_in_bounds = !want_in_bounds;
      condition = nir_scalar_chase_movs(nir_scalar_chase_alu_src(condition, 0));
   }

   /* Collapsed post-preprocess form:
    *   out-of-bounds: ior(gx >= width*channels, gy >= height)
    *   in-bounds:     iand(gx < width*channels, gy < height) */
   if (!want_in_bounds && scalar_op(condition, nir_op_ior)) {
      nir_scalar a = nir_scalar_chase_alu_src(condition, 0);
      nir_scalar b = nir_scalar_chase_alu_src(condition, 1);
      if (match_ge_x_term(a, gx, width, channels, ls, false) &&
          match_ge_y_term(b, gy, height_offset, ls, false))
         return true;
      if (match_ge_x_term(b, gx, width, channels, ls, false) &&
          match_ge_y_term(a, gy, height_offset, ls, false))
         return true;
   }
   if (want_in_bounds && scalar_op(condition, nir_op_iand)) {
      nir_scalar a = nir_scalar_chase_alu_src(condition, 0);
      nir_scalar b = nir_scalar_chase_alu_src(condition, 1);
      if (match_ge_x_term(a, gx, width, channels, ls, true) &&
          match_ge_y_term(b, gy, height_offset, ls, true))
         return true;
      if (match_ge_x_term(b, gx, width, channels, ls, true) &&
          match_ge_y_term(a, gy, height_offset, ls, true))
         return true;
   }

   /* Pre-preprocess short-circuit phi form. */
   if (!nir_def_is_phi(condition.def))
      return false;

   const nir_phi_instr *phi = nir_def_as_phi(condition.def);
   if (nir_cf_node_next(&phi->instr.block->cf_node) != &exit->cf_node)
      return false;
   nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
   if (!prev || prev->type != nir_cf_node_if)
      return false;

   nir_if *test = nir_cf_node_as_if(prev);
   nir_scalar test_cond = source(test->condition);

   if (!want_in_bounds) {
      if (!scalar_op(test_cond, nir_op_inot))
         return false;
      nir_scalar outside_x = nir_scalar_chase_alu_src(test_cond, 0);
      if (!match_ge_x_term(outside_x, gx, width, channels, ls, false))
         return false;
      bool x = false, y = false;
      nir_foreach_phi_src(src, phi) {
         nir_scalar val = source(src->src);
         if (src->pred == nir_if_last_else_block(test) &&
             same_with_local(val, outside_x, ls)) {
            x = true;
         } else if (src->pred == nir_if_last_then_block(test) &&
                    match_ge_y_term(val, gy, height_offset, ls, false)) {
            y = true;
         } else {
            return false;
         }
      }
      return x && y;
   } else {
      if (!match_ge_x_term(test_cond, gx, width, channels, ls, true))
         return false;
      bool x = false, y = false;
      nir_foreach_phi_src(src, phi) {
         nir_scalar val = source(src->src);
         if (src->pred == nir_if_last_else_block(test) &&
             (same_with_local(val, test_cond, ls) || constant(val, 0))) {
            x = true;
         } else if (src->pred == nir_if_last_then_block(test) &&
                    match_ge_y_term(val, gy, height_offset, ls, true)) {
            y = true;
         } else {
            return false;
         }
      }
      return x && y;
   }
}

static bool
bounded_loop(nir_loop *loop, nir_scalar *index, nir_scalar *limit,
             uint32_t *limit_offset, const uint16_t ls[3])
{
   nir_block *head = nir_loop_first_block(loop);
   nir_if *branch = block_if(head);
   if (!branch || nir_loop_has_continue_construct(loop))
      return false;

   /* Accept either:
    *   if (counter < bound) {} else { break; }
    *   if (counter >= bound) { break; } else {} */
   nir_scalar test = source(branch->condition);
   bool break_in_else = false;
   nir_block *else_blk = nir_if_first_else_block(branch);
   nir_block *then_blk = nir_if_first_then_block(branch);
   nir_instr *else_term = nir_block_last_instr(else_blk);
   nir_instr *then_term = nir_block_last_instr(then_blk);

   if (nir_cf_node_is_last(&else_blk->cf_node) && else_term &&
       else_term->type == nir_instr_type_jump &&
       nir_instr_as_jump(else_term)->type == nir_jump_break) {
      break_in_else = true;
   } else if (nir_cf_node_is_last(&then_blk->cf_node) && then_term &&
              then_term->type == nir_instr_type_jump &&
              nir_instr_as_jump(then_term)->type == nir_jump_break) {
      break_in_else = false;
   } else {
      return false;
   }

   bool want_lt = break_in_else;
   while (scalar_op(test, nir_op_inot)) {
      want_lt = !want_lt;
      test = nir_scalar_chase_movs(nir_scalar_chase_alu_src(test, 0));
   }

   nir_scalar counter, bound;
   if (want_lt) {
      if (!scalar_op(test, nir_op_ilt))
         return false;
      counter = nir_scalar_chase_alu_src(test, 0);
      bound = nir_scalar_chase_alu_src(test, 1);
   } else {
      if (!scalar_op(test, nir_op_ige))
         return false;
      counter = nir_scalar_chase_alu_src(test, 0);
      bound = nir_scalar_chase_alu_src(test, 1);
   }

   if (!nir_def_is_phi(counter.def) || nir_def_instr(counter.def)->block != head ||
       !push_count(bound, limit_offset))
      return false;

   const nir_phi_instr *phi = nir_def_as_phi(counter.def);
   nir_cf_node *entry = nir_cf_node_prev(&loop->cf_node);
   if (!entry || entry->type != nir_cf_node_block || counter.def->bit_size != 32)
      return false;
   bool enters_at_zero = false, advances = false;
   nir_foreach_phi_src(src, phi) {
      nir_scalar val = source(src->src);
      if (src->pred == nir_cf_node_as_block(entry) && constant(val, 0)) {
         enters_at_zero = true;
      } else if (src->pred == nir_loop_last_block(loop) &&
                 scalar_op(val, nir_op_iadd) &&
                 ((same_with_local(nir_scalar_chase_alu_src(val, 0), counter, ls) &&
                   constant(nir_scalar_chase_alu_src(val, 1), 1)) ||
                  (same_with_local(nir_scalar_chase_alu_src(val, 1), counter, ls) &&
                   constant(nir_scalar_chase_alu_src(val, 0), 1)))) {
         advances = true;
      } else {
         return false;
      }
   }
   *index = counter;
   *limit = bound;
   return enters_at_zero && advances;
}

static bool
storage_read(const nir_intrinsic_instr *load, unsigned desc_comps,
             struct buffer_access *access)
{
   return plain_load(load) && load->def.bit_size == 32 &&
          load->def.num_components == 1 && buffer(load->src[0], access) &&
          access->descriptor->def.num_components == desc_comps &&
          access->descriptor->def.bit_size == 32;
}

static bool
in_then(nir_cf_node *node, const nir_if *branch)
{
   while (node && node->parent != &branch->cf_node)
      node = node->parent;
   if (!node)
      return false;

   foreach_list_typed(nir_cf_node, child, node, &branch->then_list) {
      if (node == child)
         return true;
   }
   return false;
}

static bool
in_else(nir_cf_node *node, const nir_if *branch)
{
   while (node && node->parent != &branch->cf_node)
      node = node->parent;
   if (!node)
      return false;

   foreach_list_typed(nir_cf_node, child, node, &branch->else_list) {
      if (node == child)
         return true;
   }
   return false;
}

/* Match dominating bounds check `0 <= coord < extent` (either short-circuit
 * phi, ior(coord < 0, coord >= extent), uge(coord, extent), or the negated
 * iand(coord >= 0, coord < extent) / ult(coord, extent)). */
static bool
coordinate_guard(nir_if *branch, bool load_in_then, nir_scalar coordinate,
                 uint32_t *extent_offset, const uint16_t ls[3])
{
   nir_scalar condition = source(branch->condition);
   bool want_in_bounds = load_in_then;
   while (scalar_op(condition, nir_op_inot)) {
      want_in_bounds = !want_in_bounds;
      condition = nir_scalar_chase_movs(nir_scalar_chase_alu_src(condition, 0));
   }

   /* Unsigned single-comparison form (produced by nir_opt_algebraic from
    * coord < 0 || coord >= extent when extent >= 0):
    *   out-of-bounds: uge(coord, extent)
    *   in-bounds:     ult(coord, extent) */
   if (!want_in_bounds && scalar_op(condition, nir_op_uge)) {
      if (same_with_local(nir_scalar_chase_alu_src(condition, 0), coordinate, ls) &&
          push_count(nir_scalar_chase_alu_src(condition, 1), extent_offset))
         return true;
   }
   if (want_in_bounds && scalar_op(condition, nir_op_ult)) {
      if (same_with_local(nir_scalar_chase_alu_src(condition, 0), coordinate, ls) &&
          push_count(nir_scalar_chase_alu_src(condition, 1), extent_offset))
         return true;
   }

   /* Collapsed ior(ilt(coord, 0), ige(coord, extent)) */
   if (!want_in_bounds && scalar_op(condition, nir_op_ior)) {
      nir_scalar a = nir_scalar_chase_alu_src(condition, 0);
      nir_scalar b = nir_scalar_chase_alu_src(condition, 1);
      for (unsigned side = 0; side < 2; ++side) {
         nir_scalar lo = side ? b : a;
         nir_scalar hi = side ? a : b;
         if (scalar_op(lo, nir_op_ilt) &&
             same_with_local(nir_scalar_chase_alu_src(lo, 0), coordinate, ls) &&
             constant(nir_scalar_chase_alu_src(lo, 1), 0) &&
             scalar_op(hi, nir_op_ige) &&
             same_with_local(nir_scalar_chase_alu_src(hi, 0), coordinate, ls) &&
             push_count(nir_scalar_chase_alu_src(hi, 1), extent_offset))
            return true;
      }
   }

   /* Collapsed iand(ige(coord, 0), ilt(coord, extent)) */
   if (want_in_bounds && scalar_op(condition, nir_op_iand)) {
      nir_scalar a = nir_scalar_chase_alu_src(condition, 0);
      nir_scalar b = nir_scalar_chase_alu_src(condition, 1);
      for (unsigned side = 0; side < 2; ++side) {
         nir_scalar lo = side ? b : a;
         nir_scalar hi = side ? a : b;
         if (scalar_op(lo, nir_op_ige) &&
             same_with_local(nir_scalar_chase_alu_src(lo, 0), coordinate, ls) &&
             constant(nir_scalar_chase_alu_src(lo, 1), 0) &&
             scalar_op(hi, nir_op_ilt) &&
             same_with_local(nir_scalar_chase_alu_src(hi, 0), coordinate, ls) &&
             push_count(nir_scalar_chase_alu_src(hi, 1), extent_offset))
            return true;
      }
   }

   /* Pre-preprocess short-circuit phi form. */
   if (!nir_def_is_phi(condition.def))
      return false;

   const nir_phi_instr *phi = nir_def_as_phi(condition.def);
   if (nir_cf_node_next(&phi->instr.block->cf_node) != &branch->cf_node)
      return false;
   nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
   if (!prev || prev->type != nir_cf_node_if)
      return false;
   nir_if *test = nir_cf_node_as_if(prev);
   nir_scalar test_cond = source(test->condition);

   if (!want_in_bounds) {
      if (!scalar_op(test_cond, nir_op_inot))
         return false;
      nir_scalar negative = nir_scalar_chase_alu_src(test_cond, 0);
      if (!scalar_op(negative, nir_op_ilt) ||
          !same_with_local(nir_scalar_chase_alu_src(negative, 0), coordinate, ls) ||
          !constant(nir_scalar_chase_alu_src(negative, 1), 0))
         return false;

      bool checked_min = false, checked_max = false;
      nir_foreach_phi_src(src, phi) {
         nir_scalar val = source(src->src);
         if (src->pred == nir_if_last_else_block(test) &&
             same_with_local(val, negative, ls)) {
            checked_min = true;
         } else if (src->pred == nir_if_last_then_block(test) &&
                    scalar_op(val, nir_op_ige) &&
                    same_with_local(nir_scalar_chase_alu_src(val, 0), coordinate, ls) &&
                    push_count(nir_scalar_chase_alu_src(val, 1), extent_offset)) {
            checked_max = true;
         } else {
            return false;
         }
      }
      return checked_min && checked_max;
   } else {
      if (!scalar_op(test_cond, nir_op_ige) ||
          !same_with_local(nir_scalar_chase_alu_src(test_cond, 0), coordinate, ls) ||
          !constant(nir_scalar_chase_alu_src(test_cond, 1), 0))
         return false;

      bool checked_min = false, checked_max = false;
      nir_foreach_phi_src(src, phi) {
         nir_scalar val = source(src->src);
         if (src->pred == nir_if_last_else_block(test) &&
             (same_with_local(val, test_cond, ls) || constant(val, 0))) {
            checked_min = true;
         } else if (src->pred == nir_if_last_then_block(test) &&
                    scalar_op(val, nir_op_ilt) &&
                    same_with_local(nir_scalar_chase_alu_src(val, 0), coordinate, ls) &&
                    push_count(nir_scalar_chase_alu_src(val, 1), extent_offset)) {
            checked_max = true;
         } else {
            return false;
         }
      }
      return checked_min && checked_max;
   }
}

static bool
coordinate_is_bounded(nir_block *load, nir_scalar coord, uint32_t *offset,
                      const uint16_t ls[3])
{
   for (nir_cf_node *node = load->cf_node.parent; node; node = node->parent) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *branch = nir_cf_node_as_if(node);
      if (in_else(&load->cf_node, branch) &&
          coordinate_guard(branch, false, coord, offset, ls))
         return true;
      if (in_then(&load->cf_node, branch) &&
          coordinate_guard(branch, true, coord, offset, ls))
         return true;
   }
   return false;
}

/* Match (y*width)*channels + x*channels + channel OR (y*width + x)*channels + channel,
 * in any commutative order. */
static bool
row_major_read(nir_scalar index, nir_scalar channel, nir_scalar channels,
               nir_scalar *x, nir_scalar *y, nir_scalar *width,
               const uint16_t ls[3])
{
   index = nir_scalar_chase_movs(index);
   if (!scalar_op(index, nir_op_iadd))
      return false;

   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar ch = nir_scalar_chase_alu_src(index, side);
      nir_scalar xy = nir_scalar_chase_movs(nir_scalar_chase_alu_src(index, 1 - side));
      if (!same_with_local(ch, channel, ls))
         continue;

      /* Form A: (y*width)*channels + x*channels */
      if (scalar_op(xy, nir_op_iadd)) {
         for (unsigned s2 = 0; s2 < 2; ++s2) {
            nir_scalar row = nir_scalar_chase_movs(nir_scalar_chase_alu_src(xy, s2));
            nir_scalar col = nir_scalar_chase_movs(nir_scalar_chase_alu_src(xy, 1 - s2));
            if (!scalar_op(row, nir_op_imul) || !scalar_op(col, nir_op_imul))
               continue;
            nir_scalar r_other = {0}, c_other = {0};
            if (same_with_local(nir_scalar_chase_alu_src(row, 1), channels, ls))
               r_other = nir_scalar_chase_movs(nir_scalar_chase_alu_src(row, 0));
            else if (same_with_local(nir_scalar_chase_alu_src(row, 0), channels, ls))
               r_other = nir_scalar_chase_movs(nir_scalar_chase_alu_src(row, 1));
            else
               continue;

            if (same_with_local(nir_scalar_chase_alu_src(col, 1), channels, ls))
               c_other = nir_scalar_chase_movs(nir_scalar_chase_alu_src(col, 0));
            else if (same_with_local(nir_scalar_chase_alu_src(col, 0), channels, ls))
               c_other = nir_scalar_chase_movs(nir_scalar_chase_alu_src(col, 1));
            else
               continue;

            if (scalar_op(r_other, nir_op_imul)) {
               *x = c_other;
               *y = nir_scalar_chase_alu_src(r_other, 0);
               *width = nir_scalar_chase_alu_src(r_other, 1);
               return true;
            }
         }
      }

      /* Form B: (y*width + x) * channels */
      if (scalar_op(xy, nir_op_imul)) {
         nir_scalar sum = {0};
         if (same_with_local(nir_scalar_chase_alu_src(xy, 1), channels, ls))
            sum = nir_scalar_chase_movs(nir_scalar_chase_alu_src(xy, 0));
         else if (same_with_local(nir_scalar_chase_alu_src(xy, 0), channels, ls))
            sum = nir_scalar_chase_movs(nir_scalar_chase_alu_src(xy, 1));
         else
            continue;

         if (scalar_op(sum, nir_op_iadd)) {
            for (unsigned s2 = 0; s2 < 2; ++s2) {
               nir_scalar mul = nir_scalar_chase_movs(nir_scalar_chase_alu_src(sum, s2));
               nir_scalar xterm = nir_scalar_chase_movs(nir_scalar_chase_alu_src(sum, 1 - s2));
               if (scalar_op(mul, nir_op_imul)) {
                  *x = xterm;
                  *y = nir_scalar_chase_alu_src(mul, 0);
                  *width = nir_scalar_chase_alu_src(mul, 1);
                  return true;
               }
            }
         }
      }
   }
   return false;
}

static bool
after_loop_test(nir_block *load, nir_loop *loop)
{
   nir_cf_node *child = &load->cf_node;
   while (child && child->parent != &loop->cf_node)
      child = child->parent;
   if (!child)
      return false;

   nir_if *test = block_if(nir_loop_first_block(loop));
   for (nir_cf_node *node = nir_cf_node_next(&test->cf_node);
        node; node = nir_cf_node_next(node)) {
      if (node == child)
         return true;
   }
   return false;
}

struct depthwise_vec4_ctx {
   nir_scalar in_x, in_y;
   nir_intrinsic_instr *role_loads[3];
   uint32_t role_res_type[3];
};

static bool
match_dw_coord(nir_scalar coord, nir_scalar out_coord, nir_scalar k_idx,
               uint32_t *pad_off, uint32_t *stride_off, uint32_t *dil_off,
               const uint16_t ls[3])
{
   /* Match (pad + out_coord * stride) + dil * k_idx in any associative/commutative
    * 3-term sum form. */
   coord = nir_scalar_chase_movs(coord);
   if (!scalar_op(coord, nir_op_iadd))
      return false;

   nir_scalar terms[3];
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(coord, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(coord, 1));

   for (unsigned split = 0; split < 2; ++split) {
      nir_scalar pair = split ? b : a;
      nir_scalar single = split ? a : b;
      if (!scalar_op(pair, nir_op_iadd))
         continue;
      terms[0] = nir_scalar_chase_movs(nir_scalar_chase_alu_src(pair, 0));
      terms[1] = nir_scalar_chase_movs(nir_scalar_chase_alu_src(pair, 1));
      terms[2] = single;

      bool got_p = false, got_o = false, got_k = false;
      uint32_t p_off = 0, st_off = 0, d_off = 0;
      for (unsigned t = 0; t < 3; ++t) {
         uint32_t tmp;
         if (!got_p && push_count(terms[t], &tmp)) {
            p_off = tmp;
            got_p = true;
         } else if (scalar_op(terms[t], nir_op_imul)) {
            nir_scalar m0 = nir_scalar_chase_alu_src(terms[t], 0);
            nir_scalar m1 = nir_scalar_chase_alu_src(terms[t], 1);
            if (!got_o && same_with_local(m0, out_coord, ls) && push_count(m1, &tmp)) {
               st_off = tmp;
               got_o = true;
            } else if (!got_o && same_with_local(m1, out_coord, ls) && push_count(m0, &tmp)) {
               st_off = tmp;
               got_o = true;
            } else if (!got_k && same_with_local(m0, k_idx, ls) && push_count(m1, &tmp)) {
               d_off = tmp;
               got_k = true;
            } else if (!got_k && same_with_local(m1, k_idx, ls) && push_count(m0, &tmp)) {
               d_off = tmp;
               got_k = true;
            }
         }
      }
      if (got_p && got_o && got_k) {
         *pad_off = p_off;
         *stride_off = st_off;
         *dil_off = d_off;
         return true;
      }
   }
   return false;
}

static bool
match_dw_accum(nir_scalar val, nir_intrinsic_instr *in_ld, nir_intrinsic_instr *wt_ld,
               nir_intrinsic_instr *bias_ld, struct hk_nir_depthwise_output *res,
               const uint16_t ls[3])
{
   val = nir_scalar_chase_movs(val);
   nir_scalar outer_sum = {0};
   uint32_t hb_off = UINT32_MAX;
   uint32_t bias_fp_ctrl = 0;

   if (!bias_ld) {
      /* Biasless depthwise: stored value is outer_sum directly. */
      outer_sum = val;
   } else {
      /* Biased depthwise: either if (has_bias == 1 / != 0) phi or bcsel. */
      nir_scalar with_bias = {0};
      nir_scalar bcond = {0};

      if (nir_def_is_phi(val.def) && !val.comp) {
         nir_phi_instr *bias_phi = nir_def_as_phi(val.def);
         nir_cf_node *prev = nir_cf_node_prev(&bias_phi->instr.block->cf_node);
         if (!prev || prev->type != nir_cf_node_if)
            return false;
         nir_if *bias_if = nir_cf_node_as_if(prev);
         bcond = source(bias_if->condition);
         nir_foreach_phi_src(ps, bias_phi) {
            if (ps->pred == nir_if_last_then_block(bias_if))
               with_bias = source(ps->src);
            else if (ps->pred == nir_if_last_else_block(bias_if))
               outer_sum = source(ps->src);
            else
               return false;
         }
      } else if (scalar_op(val, nir_op_bcsel)) {
         bcond = nir_scalar_chase_movs(nir_scalar_chase_alu_src(val, 0));
         with_bias = nir_scalar_chase_movs(nir_scalar_chase_alu_src(val, 1));
         outer_sum = nir_scalar_chase_movs(nir_scalar_chase_alu_src(val, 2));
      } else if (scalar_op(val, nir_op_fadd)) {
         nir_alu_instr *badd = nir_def_as_alu(val.def);
         nir_scalar wa = nir_scalar_chase_movs(nir_scalar_chase_alu_src(val, 0));
         nir_scalar wb = nir_scalar_chase_movs(nir_scalar_chase_alu_src(val, 1));
         if (wb.def == &bias_ld->def && !wb.comp)
            outer_sum = wa;
         else if (wa.def == &bias_ld->def && !wa.comp)
            outer_sum = wb;
         else
            return false;
         bias_fp_ctrl = badd->fp_math_ctrl;
         goto have_outer_sum;
      } else {
         return false;
      }

      bool flipped = false;
      while (scalar_op(bcond, nir_op_inot)) {
         flipped = !flipped;
         bcond = nir_scalar_chase_movs(nir_scalar_chase_alu_src(bcond, 0));
      }
      if (flipped) {
         nir_scalar tmp = with_bias;
         with_bias = outer_sum;
         outer_sum = tmp;
      }

      bool cond_ok = false;
      if (scalar_op(bcond, nir_op_ieq)) {
         nir_scalar ba = nir_scalar_chase_alu_src(bcond, 0);
         nir_scalar bb = nir_scalar_chase_alu_src(bcond, 1);
         if ((push_count(ba, &hb_off) && constant(bb, 1)) ||
             (push_count(bb, &hb_off) && constant(ba, 1)))
            cond_ok = true;
         else if ((push_count(ba, &hb_off) && constant(bb, 0)) ||
                  (push_count(bb, &hb_off) && constant(ba, 0))) {
            nir_scalar tmp = with_bias;
            with_bias = outer_sum;
            outer_sum = tmp;
            cond_ok = true;
         }
      } else if (scalar_op(bcond, nir_op_ine)) {
         nir_scalar ba = nir_scalar_chase_alu_src(bcond, 0);
         nir_scalar bb = nir_scalar_chase_alu_src(bcond, 1);
         if ((push_count(ba, &hb_off) && constant(bb, 0)) ||
             (push_count(bb, &hb_off) && constant(ba, 0)))
            cond_ok = true;
      }
      if (!cond_ok)
         return false;

      if (!outer_sum.def || !with_bias.def || !scalar_op(with_bias, nir_op_fadd))
         return false;
      nir_alu_instr *badd = nir_def_as_alu(with_bias.def);
      nir_scalar wa = nir_scalar_chase_alu_src(with_bias, 0);
      nir_scalar wb = nir_scalar_chase_alu_src(with_bias, 1);
      if (!((same_with_local(wa, outer_sum, ls) && wb.def == &bias_ld->def) ||
            (same_with_local(wb, outer_sum, ls) && wa.def == &bias_ld->def)))
         return false;
      bias_fp_ctrl = badd->fp_math_ctrl;
   have_outer_sum:;
   }

   /* Trace outer_sum (outer loop header phi) -> init 0.0f and backedge phi chain
    * down to inner loop phi -> single fadd(inner_sum, fmul(in_ld, wt_ld)). */
   if (!outer_sum.def || !nir_def_is_phi(outer_sum.def))
      return false;
   nir_phi_instr *ophi = nir_def_as_phi(outer_sum.def);
   bool saw_zero = false;
   nir_scalar o_next = {0};
   nir_foreach_phi_src(ps, ophi) {
      nir_scalar s = source(ps->src);
      if (constant(s, 0))
         saw_zero = true;
      else if (!o_next.def)
         o_next = s;
      else
         return false;
   }
   if (!saw_zero || !o_next.def)
      return false;

   nir_scalar cur = o_next;
   nir_phi_instr *iphi = NULL;
   for (unsigned depth = 0; depth < 6 && cur.def && nir_def_is_phi(cur.def); ++depth) {
      nir_phi_instr *p = nir_def_as_phi(cur.def);
      nir_scalar other = {0};
      bool has_outer = false;
      unsigned nsrcs = 0;
      nir_foreach_phi_src(ps, p) {
         nsrcs++;
         nir_scalar s = source(ps->src);
         if (same_with_local(s, outer_sum, ls))
            has_outer = true;
         else if (!other.def)
            other = s;
         else
            return false;
      }
      if (nsrcs != 2 || !has_outer || !other.def)
         return false;
      if (p->instr.block->cf_node.parent->type == nir_cf_node_loop &&
          p->instr.block == nir_loop_first_block(nir_cf_node_as_loop(p->instr.block->cf_node.parent))) {
         iphi = p;
         cur = other;
         break;
      }
      cur = other;
   }
   if (!iphi || !cur.def)
      return false;
   nir_scalar inner_sum = nir_get_scalar(&iphi->def, 0);

   for (unsigned depth = 0; depth < 6 && cur.def && nir_def_is_phi(cur.def); ++depth) {
      nir_phi_instr *p = nir_def_as_phi(cur.def);
      nir_scalar other = {0};
      bool has_inner = false;
      unsigned nsrcs = 0;
      nir_foreach_phi_src(ps, p) {
         nsrcs++;
         nir_scalar s = source(ps->src);
         if (same_with_local(s, inner_sum, ls))
            has_inner = true;
         else if (!other.def)
            other = s;
         else
            return false;
      }
      if (nsrcs != 2 || !has_inner || !other.def)
         return false;
      cur = other;
   }

   if (!cur.def)
      return false;
   if (scalar_op(cur, nir_op_fadd)) {
      nir_alu_instr *add = nir_def_as_alu(cur.def);
      nir_scalar aa = nir_scalar_chase_alu_src(cur, 0);
      nir_scalar ab = nir_scalar_chase_alu_src(cur, 1);
      nir_scalar prod = {0};
      if (same_with_local(aa, inner_sum, ls))
         prod = ab;
      else if (same_with_local(ab, inner_sum, ls))
         prod = aa;
      else
         return false;
      if (!scalar_op(prod, nir_op_fmul))
         return false;
      nir_alu_instr *mul = nir_def_as_alu(prod.def);
      nir_scalar ma = nir_scalar_chase_alu_src(prod, 0);
      nir_scalar mb = nir_scalar_chase_alu_src(prod, 1);
      if (!((ma.def == &in_ld->def && mb.def == &wt_ld->def) ||
            (mb.def == &in_ld->def && ma.def == &wt_ld->def)))
         return false;
      res->fmul_fp_ctrl = mul->fp_math_ctrl;
      res->fadd_fp_ctrl = add->fp_math_ctrl;
   } else if (scalar_op(cur, nir_op_ffma)) {
      nir_alu_instr *fma = nir_def_as_alu(cur.def);
      nir_scalar ma = nir_scalar_chase_alu_src(cur, 0);
      nir_scalar mb = nir_scalar_chase_alu_src(cur, 1);
      nir_scalar mc = nir_scalar_chase_alu_src(cur, 2);
      if (!same_with_local(mc, inner_sum, ls) ||
          !((ma.def == &in_ld->def && mb.def == &wt_ld->def) ||
            (mb.def == &in_ld->def && ma.def == &wt_ld->def)))
         return false;
      res->fmul_fp_ctrl = fma->fp_math_ctrl;
      res->fadd_fp_ctrl = fma->fp_math_ctrl;
   } else {
      return false;
   }

   res->has_bias_offset_B = hb_off;
   res->bias_fadd_fp_ctrl = bias_fp_ctrl;
   return true;
}

static bool
read_footprints(struct hk_nir_depthwise_output *result,
                const struct buffer_access *reads,
                nir_intrinsic_instr *const *read_ops,
                nir_block *const *blocks, unsigned read_count,
                const nir_if *guard, bool body_in_then,
                nir_loop *outer, nir_loop *inner,
                nir_scalar kx, nir_scalar ky, nir_scalar kw,
                nir_scalar channel, nir_scalar channels,
                const uint16_t ls[3],
                struct depthwise_vec4_ctx *vctx)
{
   bool have[3] = {0};
   for (unsigned i = 0; i < read_count; ++i) {
      bool inside_guard = body_in_then ? in_then(&blocks[i]->cf_node, guard)
                                       : in_else(&blocks[i]->cf_node, guard);
      if (!inside_guard)
         return false;

      if (same_with_local(reads[i].element, channel, ls)) {
         if (have[2])
            return false;
         have[2] = true;
         result->reads[2] = reads[i].binding;
         if (vctx) {
            vctx->role_loads[2] = read_ops[i];
            vctx->role_res_type[2] = nir_intrinsic_resource_type(
               nir_def_as_intrinsic(reads[i].descriptor->src[0].ssa));
         }
         continue;
      }

      nir_scalar x, y, width;
      if (!row_major_read(reads[i].element, channel, channels, &x, &y, &width, ls))
         return false;

      /* Weight read: y == ky, x == kx, width == kw (or y/width commuted in mul). */
      bool is_weight = false;
      if (after_loop_test(blocks[i], outer) && after_loop_test(blocks[i], inner) &&
          same_with_local(x, kx, ls)) {
         if ((same_with_local(y, ky, ls) && same_with_local(width, kw, ls)) ||
             (same_with_local(y, kw, ls) && same_with_local(width, ky, ls)))
            is_weight = true;
      }
      if (is_weight) {
         if (have[1])
            return false;
         have[1] = true;
         result->reads[1] = reads[i].binding;
         if (vctx) {
            vctx->role_loads[1] = read_ops[i];
            vctx->role_res_type[1] = nir_intrinsic_resource_type(
               nir_def_as_intrinsic(reads[i].descriptor->src[0].ssa));
         }
         continue;
      }

      /* Input read: x and y are bounded by push constants iw, ih. Note y*iw may
       * commute in row_major_read so try both (y, width) orientations. */
      uint32_t iw_off = 0, ih_off = 0;
      nir_scalar in_y = y;
      bool is_input = false;
      if (push_count(width, &iw_off) &&
          coordinate_is_bounded(blocks[i], x, &iw_off, ls) &&
          coordinate_is_bounded(blocks[i], y, &ih_off, ls)) {
         /* Confirm xs bound matches widths push offset. */
         uint32_t x_bound_off = 0;
         if (coordinate_is_bounded(blocks[i], x, &x_bound_off, ls) &&
             x_bound_off == iw_off) {
            is_input = true;
            in_y = y;
         }
      }
      if (!is_input && push_count(y, &iw_off)) {
         uint32_t x_bound_off = 0;
         if (coordinate_is_bounded(blocks[i], x, &x_bound_off, ls) &&
             x_bound_off == iw_off &&
             coordinate_is_bounded(blocks[i], width, &ih_off, ls)) {
            is_input = true;
            in_y = width;
         }
      }

      if (is_input) {
         if (have[0])
            return false;
         have[0] = true;
         result->reads[0] = reads[i].binding;
         result->input_width_offset_B = iw_off;
         result->input_height_offset_B = ih_off;
         if (vctx) {
            vctx->in_x = x;
            vctx->in_y = in_y;
            vctx->role_loads[0] = read_ops[i];
            vctx->role_res_type[0] = nir_intrinsic_resource_type(
               nir_def_as_intrinsic(reads[i].descriptor->src[0].ssa));
         }
         continue;
      }

      return false;
   }
   result->has_bias_buffer = have[2];
   return have[0] && have[1] && (read_count == 2 ? !have[2] : have[2]);
}

/* Match depthwise output index:
 *   Form 1: ((gy*width)*channels + (ox*channels)) + chan  (or factored (gy*width + ox)*channels + chan)
 *   Form 2: (gy*width)*channels + gx  or  gy*(width*channels) + gx  (algebraic opt simplified ox*c+chan -> gx)
 */
static bool
match_dw_output_index(nir_scalar idx, const uint16_t ls[3],
                      nir_scalar *gx_out, nir_scalar *gy_out,
                      nir_scalar *width_out, nir_scalar *channels_out,
                      uint32_t *width_off_out, uint32_t *channels_off_out,
                      nir_scalar *ox_out, nir_scalar *chan_out,
                      bool *have_ox_chan_out)
{
   idx = nir_scalar_chase_movs(idx);
   if (!scalar_op(idx, nir_op_iadd))
      return false;

   /* Form 1: row_major_read(idx, chan, channels, &ox, &gy, &width) where
    * chan = rem(gx, channels) and ox = div(gx, channels). */
   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar cand_ch = nir_scalar_chase_movs(nir_scalar_chase_alu_src(idx, side));
      nir_scalar gx = {0}, channels = {0};
      uint32_t ch_off = 0;

      if (scalar_op(cand_ch, nir_op_irem) || scalar_op(cand_ch, nir_op_imod) ||
          scalar_op(cand_ch, nir_op_umod)) {
         gx = nir_scalar_chase_alu_src(cand_ch, 0);
         channels = nir_scalar_chase_alu_src(cand_ch, 1);
      } else if (scalar_op(cand_ch, nir_op_isub)) {
         gx = nir_scalar_chase_alu_src(cand_ch, 0);
         nir_scalar sub = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cand_ch, 1));
         if (scalar_op(sub, nir_op_imul)) {
            nir_scalar m0 = nir_scalar_chase_alu_src(sub, 0);
            nir_scalar m1 = nir_scalar_chase_alu_src(sub, 1);
            if (push_count(m1, &ch_off) && match_div_op(m0, gx, m1, ls))
               channels = m1;
            else if (push_count(m0, &ch_off) && match_div_op(m1, gx, m0, ls))
               channels = m0;
         }
      }
      if (!channels.def || !is_global_id(gx, ls, 0) || !push_count(channels, &ch_off))
         continue;

      nir_scalar ox, gy, w;
      if (!row_major_read(idx, cand_ch, channels, &ox, &gy, &w, ls))
         continue;
      if (!match_div_op(ox, gx, channels, ls))
         continue;

      uint32_t w_off = 0;
      if (is_global_id(gy, ls, 1) && push_count(w, &w_off)) {
         *gx_out = gx; *gy_out = gy; *width_out = w; *channels_out = channels;
         *width_off_out = w_off; *channels_off_out = ch_off;
         *ox_out = ox; *chan_out = cand_ch; *have_ox_chan_out = true;
         return true;
      }
      if (is_global_id(w, ls, 1) && push_count(gy, &w_off)) {
         *gx_out = gx; *gy_out = w; *width_out = gy; *channels_out = channels;
         *width_off_out = w_off; *channels_off_out = ch_off;
         *ox_out = ox; *chan_out = cand_ch; *have_ox_chan_out = true;
         return true;
      }
   }

   /* Form 2: (gy * width) * channels + gx  OR  gy * (width * channels) + gx */
   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar gx = nir_scalar_chase_movs(nir_scalar_chase_alu_src(idx, side));
      nir_scalar row = nir_scalar_chase_movs(nir_scalar_chase_alu_src(idx, 1 - side));
      if (!is_global_id(gx, ls, 0) || !scalar_op(row, nir_op_imul))
         continue;

      /* Flatten 3-factor product a * b * c into (gy, p0, p1). */
      nir_scalar r0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(row, 0));
      nir_scalar r1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(row, 1));
      for (unsigned split = 0; split < 2; ++split) {
         nir_scalar pair = split ? r1 : r0;
         nir_scalar single = split ? r0 : r1;
         if (!scalar_op(pair, nir_op_imul))
            continue;
         nir_scalar f[3] = {
            nir_scalar_chase_movs(nir_scalar_chase_alu_src(pair, 0)),
            nir_scalar_chase_movs(nir_scalar_chase_alu_src(pair, 1)),
            single,
         };
         nir_scalar gy = {0}, p0 = {0}, p1 = {0};
         uint32_t off0 = 0, off1 = 0;
         unsigned np = 0;
         for (unsigned k = 0; k < 3; ++k) {
            uint32_t tmp;
            if (!gy.def && is_global_id(f[k], ls, 1))
               gy = f[k];
            else if (push_count(f[k], &tmp)) {
               if (np == 0) { p0 = f[k]; off0 = tmp; np++; }
               else if (np == 1) { p1 = f[k]; off1 = tmp; np++; }
            }
         }
         if (gy.def && np == 2) {
            /* Caller will disambiguate which of p0/p1 is channels vs width via
             * the div/rem in the reads. Return p0 as width, p1 as channels
             * tentatively and set have_ox_chan_out = false. */
            *gx_out = gx; *gy_out = gy;
            *width_out = p0; *channels_out = p1;
            *width_off_out = off0; *channels_off_out = off1;
            *have_ox_chan_out = false;
            return true;
         }
      }
   }

   return false;
}

static bool
find_div_rem_in_shader(nir_function_impl *impl, nir_scalar gx,
                       nir_scalar channels, const uint16_t ls[3],
                       nir_scalar *ox_out, nir_scalar *chan_out)
{
   nir_scalar ox = {0}, ch = {0};
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *a = nir_instr_as_alu(instr);
         if (a->def.num_components != 1 || a->def.bit_size != 32)
            continue;
         nir_scalar s = nir_get_scalar(&a->def, 0);
         if (!ox.def && match_div_op(s, gx, channels, ls))
            ox = s;
         if (!ch.def && match_rem_op(s, gx, channels, ls))
            ch = s;
      }
   }
   if (!ox.def || !ch.def)
      return false;
   *ox_out = ox;
   *chan_out = ch;
   return true;
}

static bool
match_depthwise(const nir_shader *shader, struct hk_nir_depthwise_output *match,
                nir_address_format storage_format)
{
   const unsigned desc_comps = storage_format_components(storage_format);
   if (!desc_comps ||
       shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable || !shader->info.workgroup_size[0] ||
       !shader->info.workgroup_size[1] || shader->info.workgroup_size[2] != 1 ||
       (shader->info.num_ssbos != 3 && shader->info.num_ssbos != 4))
      return false;

   nir_function_impl *impl = NULL;
   nir_foreach_function_impl(candidate, shader) {
      if (impl || !candidate->function->is_entrypoint)
         return false;
      impl = candidate;
   }
   if (!impl)
      return false;

   const uint16_t *ls = shader->info.workgroup_size;
   const nir_intrinsic_instr *store = NULL;
   struct buffer_access read_access[3];
   nir_intrinsic_instr *read_ops[3] = {0};
   nir_block *read_blocks[3];
   unsigned read_count = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
         case nir_instr_type_load_const:
         case nir_instr_type_deref:
         case nir_instr_type_phi:
            continue;
         case nir_instr_type_jump:
            if (nir_instr_as_jump(instr)->type != nir_jump_break)
               return false;
            continue;
         case nir_instr_type_intrinsic:
            break;
         default:
            return false;
         }

         nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
         switch (op->intrinsic) {
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_push_constant:
            break;
         case nir_intrinsic_load_deref: {
            const nir_deref_instr *ptr =
               nir_def_is_deref(op->src[0].ssa)
                  ? nir_def_as_deref(op->src[0].ssa)
                  : NULL;
            if (!ptr)
               return false;
            if (nir_deref_mode_is(ptr, nir_var_mem_ssbo)) {
               if (read_count == 3 ||
                   !storage_read(op, desc_comps, &read_access[read_count]))
                  return false;
               read_ops[read_count] = op;
               read_blocks[read_count++] = block;
            } else if (nir_deref_mode_is(ptr, nir_var_mem_push_const)) {
               uint32_t offset;
               if (!push_count(nir_get_scalar((nir_def *)&op->def, 0), &offset))
                  return false;
            } else if (nir_deref_mode_is(ptr, nir_var_system_value)) {
               unsigned axis;
               if (!sysval_id_axis(nir_get_scalar((nir_def *)&op->def, 0),
                                   SYSTEM_VALUE_GLOBAL_INVOCATION_ID, ls, &axis))
                  return false;
            } else {
               return false;
            }
            break;
         }
         case nir_intrinsic_store_deref:
            if (store || op->num_components != 1 ||
                op->src[1].ssa->bit_size != 32 ||
                nir_intrinsic_write_mask(op) != 1 ||
                (nir_intrinsic_access(op) &
                 ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS)))
               return false;
            store = op;
            break;
         default:
            return false;
         }
      }
   }

   if (!store || (read_count != 2 && read_count != 3))
      return false;

   struct buffer_access out;
   if (!buffer(store->src[0], &out) ||
       out.descriptor->def.num_components != desc_comps ||
       out.descriptor->def.bit_size != 32)
      return false;

   struct hk_nir_depthwise_output result = {
      .has_bias_offset_B = UINT32_MAX,
      .float_controls_execution_mode = shader->info.float_controls_execution_mode,
   };

   nir_scalar gx = {0}, gy = {0}, width = {0}, channels = {0};
   nir_scalar divided = {0}, chan = {0};
   bool have_ox_chan = false;
   if (!match_dw_output_index(out.element, ls, &gx, &gy, &width, &channels,
                              &result.width_offset_B, &result.channels_offset_B,
                              &divided, &chan, &have_ox_chan))
      return false;

   if (!have_ox_chan) {
      if (!find_div_rem_in_shader(impl, gx, channels, ls, &divided, &chan)) {
         /* Try swapping width and channels! */
         nir_scalar tmp_s = width;
         width = channels;
         channels = tmp_s;
         uint32_t tmp_o = result.width_offset_B;
         result.width_offset_B = result.channels_offset_B;
         result.channels_offset_B = tmp_o;
         if (!find_div_rem_in_shader(impl, gx, channels, ls, &divided, &chan))
            return false;
      }
   }

   nir_cf_node *owner = store->instr.block->cf_node.parent;
   if (owner->type != nir_cf_node_if || owner->parent != &impl->cf_node)
      return false;
   nir_if *guard = nir_cf_node_as_if(owner);

   bool body_in_then = false;
   if (store->instr.block == nir_if_last_else_block(guard) &&
       exec_list_is_empty(&nir_if_first_then_block(guard)->instr_list) &&
       nir_cf_node_is_last(&nir_if_first_then_block(guard)->cf_node)) {
      body_in_then = false;
   } else if (store->instr.block == nir_if_last_then_block(guard) &&
              exec_list_is_empty(&nir_if_first_else_block(guard)->instr_list) &&
              nir_cf_node_is_last(&nir_if_first_else_block(guard)->cf_node)) {
      body_in_then = true;
   } else {
      return false;
   }

   if (!store_guard(guard, body_in_then, gx, gy, width, channels,
                    &result.height_offset_B, ls))
      return false;

   struct exec_list *body_list = body_in_then ? &guard->then_list : &guard->else_list;
   nir_cf_node *outer = NULL;
   nir_scalar ky, kh;
   foreach_list_typed(nir_cf_node, node, node, body_list) {
      if (node->type != nir_cf_node_loop)
         continue;
      if (outer || !bounded_loop(nir_cf_node_as_loop(node), &ky, &kh,
                                 &result.kernel_height_offset_B, ls))
         return false;
      outer = node;
   }
   if (!outer)
      return false;

   nir_cf_node *inner = NULL;
   nir_scalar kx, kw;
   nir_foreach_block(block, impl) {
      nir_cf_node *parent = block->cf_node.parent;
      if (parent->type != nir_cf_node_loop ||
          block != nir_loop_first_block(nir_cf_node_as_loop(parent)) ||
          parent == outer)
         continue;
      nir_cf_node *ancestor = parent->parent;
      while (ancestor && ancestor != outer)
         ancestor = ancestor->parent;
      if (inner || !ancestor ||
          !bounded_loop(nir_cf_node_as_loop(parent), &kx, &kw,
                        &result.kernel_width_offset_B, ls))
         return false;
      inner = parent;
   }

   struct depthwise_vec4_ctx vctx = {0};
   if (!inner ||
       !read_footprints(&result, read_access, read_ops, read_blocks, read_count,
                        guard, body_in_then,
                        nir_cf_node_as_loop(outer), nir_cf_node_as_loop(inner),
                        kx, ky, kw, chan, channels, ls, &vctx))
      return false;

   result.output = out.binding;
   result.desc_comps = desc_comps;
   nir_intrinsic_instr *out_res = nir_def_as_intrinsic(out.descriptor->src[0].ssa);
   result.out_res_type = nir_intrinsic_resource_type(out_res);
   for (unsigned i = 0; i < 3; ++i) {
      result.read_res_type[i] = vctx.role_res_type[i];
      result.local_size[i] = shader->info.workgroup_size[i];
   }

   if (match_dw_coord(vctx.in_x, divided, kx, &result.pad_x_offset_B,
                      &result.stride_x_offset_B, &result.dil_x_offset_B, ls) &&
       match_dw_coord(vctx.in_y, gy, ky, &result.pad_y_offset_B,
                      &result.stride_y_offset_B, &result.dil_y_offset_B, ls) &&
       match_dw_accum(source(store->src[1]), vctx.role_loads[0],
                      vctx.role_loads[1], vctx.role_loads[2], &result, ls)) {
      uint32_t offs[14] = {
         result.width_offset_B, result.height_offset_B, result.channels_offset_B,
         result.input_width_offset_B, result.input_height_offset_B,
         result.kernel_width_offset_B, result.kernel_height_offset_B,
         result.stride_x_offset_B, result.stride_y_offset_B,
         result.dil_x_offset_B, result.dil_y_offset_B,
         result.pad_x_offset_B, result.pad_y_offset_B,
         (result.has_bias_buffer && result.has_bias_offset_B != UINT32_MAX)
            ? result.has_bias_offset_B : 0,
      };
      unsigned noffs =
         (result.has_bias_buffer && result.has_bias_offset_B != UINT32_MAX) ? 14 : 13;
      uint32_t max_b = 0;
      bool aligned = true;
      for (unsigned i = 0; i < noffs; ++i) {
         if ((offs[i] & 3) || offs[i] + 4 > 64)
            aligned = false;
         max_b = MAX2(max_b, offs[i] + 4);
      }
      if (aligned && result.output.offset_B == 0 &&
          result.reads[0].offset_B == 0 && result.reads[1].offset_B == 0 &&
          (!result.has_bias_buffer || result.reads[2].offset_B == 0)) {
         result.push_bytes = max_b;
         result.vec4_valid = true;
      }
   }

   *match = result;
   return true;
}

bool
hk_nir_match_depthwise_output(const nir_shader *shader,
                              nir_address_format storage_format,
                              struct hk_nir_depthwise_output *match)
{
   return match_depthwise(shader, match, storage_format);
}

/* Match gy * R for 1 <= R <= 16 (gy itself for R=1, ishl(gy, log2(R)), or
 * imul(gy, R)). */
static bool
match_gy_times_r(nir_scalar s, const uint16_t ls[3], uint32_t *r_out)
{
   s = strip_zero_add(s);
   if (is_global_id(s, ls, 1)) {
      *r_out = 1;
      return true;
   }
   if (scalar_op(s, nir_op_ishl)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (is_global_id(a, ls, 1) && nir_scalar_is_const(b)) {
         uint32_t sh = nir_scalar_as_uint(b);
         if (sh >= 1 && sh <= 4) {
            *r_out = 1u << sh;
            return true;
         }
      }
   }
   if (scalar_op(s, nir_op_imul)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (is_global_id(a, ls, 1) && nir_scalar_is_const(b)) {
         uint32_t r = nir_scalar_as_uint(b);
         if (r >= 1 && r <= 16) {
            *r_out = r;
            return true;
         }
      }
      if (is_global_id(b, ls, 1) && nir_scalar_is_const(a)) {
         uint32_t r = nir_scalar_as_uint(a);
         if (r >= 1 && r <= 16) {
            *r_out = r;
            return true;
         }
      }
   }
   return false;
}

/* Match gy * R + j for 1 <= R <= 16, 0 <= j < R (handling iadd and, when R is
 * a power of two, ior after algebraic opt). */
static bool
match_gemm_row(nir_scalar s, const uint16_t ls[3], uint32_t *r_out, uint32_t *j_out)
{
   s = strip_zero_add(s);
   uint32_t r = 0;
   if (match_gy_times_r(s, ls, &r)) {
      *r_out = r;
      *j_out = 0;
      return true;
   }
   const bool is_add = scalar_op(s, nir_op_iadd);
   const bool is_or = scalar_op(s, nir_op_ior);
   if (!is_add && !is_or)
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(s, 0);
   nir_scalar b = nir_scalar_chase_alu_src(s, 1);
   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar base = side ? b : a;
      nir_scalar c = side ? a : b;
      if (match_gy_times_r(base, ls, &r) && nir_scalar_is_const(c)) {
         uint32_t j = nir_scalar_as_uint(c);
         if (j < r && (is_add || util_is_power_of_two_nonzero(r))) {
            *r_out = r;
            *j_out = j;
            return true;
         }
      }
   }
   return false;
}

/* Match row * n + col where row == gy*R + j, col == gid.x, n == push[n_off]. */
static bool
match_gemm_c_index(nir_scalar idx, const uint16_t ls[3],
                   uint32_t *r_out, uint32_t *j_out, uint32_t *n_offset_B)
{
   idx = nir_scalar_chase_movs(idx);
   if (!scalar_op(idx, nir_op_iadd))
      return false;
   for (unsigned side = 0; side < 2; ++side) {
      nir_scalar mul = nir_scalar_chase_movs(nir_scalar_chase_alu_src(idx, side));
      nir_scalar col = nir_scalar_chase_alu_src(idx, 1 - side);
      if (!is_global_id(col, ls, 0) || !scalar_op(mul, nir_op_imul))
         continue;
      nir_scalar a = nir_scalar_chase_alu_src(mul, 0);
      nir_scalar b = nir_scalar_chase_alu_src(mul, 1);
      uint32_t r, j, n_off;
      if (match_gemm_row(a, ls, &r, &j) && push_count(b, &n_off)) {
         *r_out = r; *j_out = j; *n_offset_B = n_off;
         return true;
      }
      if (match_gemm_row(b, ls, &r, &j) && push_count(a, &n_off)) {
         *r_out = r; *j_out = j; *n_offset_B = n_off;
         return true;
      }
   }
   return false;
}

/* Extract the two predicates of an `a && b` branch condition (either `iand`
 * or a short-circuit phi after `if (a) { b }`). */
static bool
match_guard_cond_pair(nir_if *branch, const uint16_t ls[3],
                      nir_scalar *p0_out, nir_scalar *p1_out)
{
   nir_scalar cond = source(branch->condition);
   if (scalar_op(cond, nir_op_iand)) {
      *p0_out = nir_scalar_chase_alu_src(cond, 0);
      *p1_out = nir_scalar_chase_alu_src(cond, 1);
      return true;
   }
   if (!nir_def_is_phi(cond.def))
      return false;
   const nir_phi_instr *phi = nir_def_as_phi(cond.def);
   if (nir_cf_node_next(&phi->instr.block->cf_node) != &branch->cf_node)
      return false;
   nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
   if (!prev || prev->type != nir_cf_node_if)
      return false;
   nir_if *test = nir_cf_node_as_if(prev);
   nir_scalar first = source(test->condition);
   nir_scalar p1 = {0};
   bool saw_false = false, saw_second = false;
   nir_foreach_phi_src(src, phi) {
      nir_scalar val = source(src->src);
      if (src->pred == nir_if_last_else_block(test) &&
          (same_with_local(val, first, ls) || constant(val, 0))) {
         saw_false = true;
      } else if (src->pred == nir_if_last_then_block(test)) {
         p1 = val;
         saw_second = true;
      } else {
         return false;
      }
   }
   if (!saw_false || !saw_second)
      return false;
   *p0_out = first;
   *p1_out = p1;
   return true;
}

/* Match row < m && col < n guard (iand or short-circuit phi). */
static bool
match_gemm_store_guard(nir_if *branch, const uint16_t ls[3],
                       uint32_t expected_r, uint32_t expected_j,
                       uint32_t expected_n_off, uint32_t *m_offset_B)
{
   nir_scalar p0 = {0}, p1 = {0};
   if (!match_guard_cond_pair(branch, ls, &p0, &p1))
      return false;
   if ((!scalar_op(p0, nir_op_ilt) && !scalar_op(p0, nir_op_ult)) ||
       (!scalar_op(p1, nir_op_ilt) && !scalar_op(p1, nir_op_ult)))
      return false;

   nir_scalar pairs[2][2] = {
      { nir_scalar_chase_alu_src(p0, 0), nir_scalar_chase_alu_src(p0, 1) },
      { nir_scalar_chase_alu_src(p1, 0), nir_scalar_chase_alu_src(p1, 1) },
   };
   bool got_row = false, got_col = false;
   uint32_t m_off = 0;
   for (unsigned i = 0; i < 2; ++i) {
      uint32_t r, j, off;
      if (is_global_id(pairs[i][0], ls, 0) &&
          push_count(pairs[i][1], &off) && off == expected_n_off) {
         got_col = true;
      } else if (match_gemm_row(pairs[i][0], ls, &r, &j) &&
                 r == expected_r && j == expected_j &&
                 push_count(pairs[i][1], &off)) {
         m_off = off;
         got_row = true;
      }
   }
   if (!got_row || !got_col)
      return false;
   *m_offset_B = m_off;
   return true;
}

static bool
match_bias_mode_if(const nir_if *branch, uint32_t *bias_mode_off,
                   uint32_t *mode_out)
{
   nir_scalar cond = source(branch->condition);
   if (!scalar_op(cond, nir_op_ieq))
      return false;
   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   uint32_t off, mode;
   if (push_count(a, &off) && nir_scalar_is_const(b)) {
      mode = nir_scalar_as_uint(b);
   } else if (push_count(b, &off) && nir_scalar_is_const(a)) {
      mode = nir_scalar_as_uint(a);
   } else {
      return false;
   }
   if (mode > 2)
      return false;
   *bias_mode_off = off;
   *mode_out = mode;
   return true;
}

/* Find the dominating `if (push[bias_mode_off] == mode)` whose then-arm holds
 * `node`. Returns false if `node` is not inside such a bias_mode arm. */
static bool
find_bias_mode_arm(nir_cf_node *node, uint32_t *bias_mode_off, uint32_t *mode_out)
{
   for (nir_cf_node *cur = node; cur && cur->parent; cur = cur->parent) {
      if (cur->parent->type != nir_cf_node_if)
         continue;
      nir_if *branch = nir_cf_node_as_if(cur->parent);
      if (!in_then(cur, branch))
         continue;
      if (match_bias_mode_if(branch, bias_mode_off, mode_out))
         return true;
   }
   return false;
}

/* Extract push-constant stride from a linear SSBO index `(... * stride) + ...`. */
static bool
find_linear_stride_push_rec(nir_scalar idx, uint32_t *stride_off, unsigned depth)
{
   idx = nir_scalar_chase_movs(idx);
   if (scalar_op(idx, nir_op_imul)) {
      if (push_count(nir_scalar_chase_alu_src(idx, 1), stride_off))
         return true;
      if (push_count(nir_scalar_chase_alu_src(idx, 0), stride_off))
         return true;
      return false;
   }
   if (!depth || !scalar_op(idx, nir_op_iadd))
      return false;
   return find_linear_stride_push_rec(nir_scalar_chase_alu_src(idx, 0), stride_off, depth - 1) ||
          find_linear_stride_push_rec(nir_scalar_chase_alu_src(idx, 1), stride_off, depth - 1);
}

static bool
find_linear_stride_push(nir_scalar idx, uint32_t *stride_off)
{
   return find_linear_stride_push_rec(idx, stride_off, 4);
}

static bool
is_local_id_axis(nir_scalar s, const uint16_t ls[3], unsigned axis)
{
   unsigned a = 0;
   return sysval_id_axis(s, SYSTEM_VALUE_LOCAL_INVOCATION_ID, ls, &a) && a == axis;
}

static bool
is_const_u32(nir_scalar s, uint32_t expected)
{
   s = nir_scalar_chase_movs(s);
   return nir_scalar_is_const(s) && s.def->bit_size == 32 &&
          nir_scalar_as_uint(s) == expected;
}

static bool
is_lid_y_mul8(nir_scalar s, const uint16_t ls[3])
{
   s = nir_scalar_chase_movs(s);
   if (scalar_op(s, nir_op_imul)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      return (is_local_id_axis(a, ls, 1) && is_const_u32(b, 8)) ||
             (is_local_id_axis(b, ls, 1) && is_const_u32(a, 8));
   }
   if (scalar_op(s, nir_op_ishl)) {
      return is_local_id_axis(nir_scalar_chase_alu_src(s, 0), ls, 1) &&
             is_const_u32(nir_scalar_chase_alu_src(s, 1), 3);
   }
   return false;
}

static bool
match_lid_row8(nir_scalar s, const uint16_t ls[3], uint32_t *j_out)
{
   s = nir_scalar_chase_movs(s);
   if (is_lid_y_mul8(s, ls)) {
      *j_out = 0;
      return true;
   }
   if (scalar_op(s, nir_op_iadd)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (is_lid_y_mul8(a, ls) && nir_scalar_is_const(b) &&
          b.def->bit_size == 32 && nir_scalar_as_uint(b) < 8) {
         *j_out = (uint32_t)nir_scalar_as_uint(b);
         return true;
      }
      if (is_lid_y_mul8(b, ls) && nir_scalar_is_const(a) &&
          a.def->bit_size == 32 && nir_scalar_as_uint(a) < 8) {
         *j_out = (uint32_t)nir_scalar_as_uint(a);
         return true;
      }
   }
   return false;
}

static bool
match_mul32(nir_scalar s, nir_scalar *base_out)
{
   s = nir_scalar_chase_movs(s);
   if (scalar_op(s, nir_op_imul)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (is_const_u32(b, 32)) {
         *base_out = a;
         return true;
      }
      if (is_const_u32(a, 32)) {
         *base_out = b;
         return true;
      }
   } else if (scalar_op(s, nir_op_ishl)) {
      if (is_const_u32(nir_scalar_chase_alu_src(s, 1), 5)) {
         *base_out = nir_scalar_chase_alu_src(s, 0);
         return true;
      }
   }
   return false;
}

static bool
is_phi_scalar(nir_scalar s, const nir_phi_instr *phi)
{
   s = nir_scalar_chase_movs(s);
   return phi && s.def == &phi->def && s.comp == 0;
}

static bool
is_tile_k_base(nir_scalar s, const nir_phi_instr *tile_phi,
               const nir_phi_instr *k_idx_phi)
{
   s = nir_scalar_chase_movs(s);
   if (k_idx_phi && is_phi_scalar(s, k_idx_phi))
      return true;
   nir_scalar base;
   if (tile_phi && match_mul32(s, &base) && is_phi_scalar(base, tile_phi))
      return true;
   return false;
}

static bool
flatten_iadd_terms(nir_scalar s, nir_scalar *terms, unsigned *count,
                   unsigned max_terms, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (depth > 0 && scalar_op(s, nir_op_iadd)) {
      return flatten_iadd_terms(nir_scalar_chase_alu_src(s, 0), terms, count,
                                max_terms, depth - 1) &&
             flatten_iadd_terms(nir_scalar_chase_alu_src(s, 1), terms, count,
                                max_terms, depth - 1);
   }
   if (*count >= max_terms)
      return false;
   terms[(*count)++] = s;
   return true;
}

static bool
match_k_plus_31(nir_scalar s, uint32_t k_off)
{
   s = nir_scalar_chase_movs(s);
   uint32_t off;
   if (scalar_op(s, nir_op_isub)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (is_const_u32(b, 1) && scalar_op(a, nir_op_iadd)) {
         nir_scalar a0 = nir_scalar_chase_alu_src(a, 0);
         nir_scalar a1 = nir_scalar_chase_alu_src(a, 1);
         if ((push_count(a0, &off) && off == k_off && is_const_u32(a1, 32)) ||
             (push_count(a1, &off) && off == k_off && is_const_u32(a0, 32)))
            return true;
      }
      return false;
   }
   nir_scalar terms[4];
   unsigned n = 0;
   if (!flatten_iadd_terms(s, terms, &n, 4, 3))
      return false;
   bool got_k = false;
   int64_t csum = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (!got_k && push_count(terms[i], &off) && off == k_off) {
         got_k = true;
      } else if (nir_scalar_is_const(terms[i]) && terms[i].def->bit_size == 32) {
         csum += (int32_t)nir_scalar_as_uint(terms[i]);
      } else {
         return false;
      }
   }
   return got_k && csum == 31;
}

static bool
match_ceil_div_32(nir_scalar s, uint32_t k_off)
{
   s = nir_scalar_chase_movs(s);
   if (scalar_op(s, nir_op_idiv) || scalar_op(s, nir_op_udiv)) {
      return is_const_u32(nir_scalar_chase_alu_src(s, 1), 32) &&
             match_k_plus_31(nir_scalar_chase_alu_src(s, 0), k_off);
   }
   if (scalar_op(s, nir_op_ishr) || scalar_op(s, nir_op_ushr)) {
      if (!is_const_u32(nir_scalar_chase_alu_src(s, 1), 5))
         return false;
      nir_scalar x = nir_scalar_chase_alu_src(s, 0);
      if (match_k_plus_31(x, k_off))
         return true;
      /* Lowered signed idiv by 32: (x + ((x >> 31) & 31)) >> 5. */
      if (scalar_op(x, nir_op_iadd)) {
         nir_scalar a = nir_scalar_chase_alu_src(x, 0);
         nir_scalar b = nir_scalar_chase_alu_src(x, 1);
         for (unsigned flip = 0; flip < 2; ++flip) {
            nir_scalar xv = flip ? b : a;
            nir_scalar bias = flip ? a : b;
            if (!match_k_plus_31(xv, k_off))
               continue;
            if (scalar_op(bias, nir_op_iand)) {
               nir_scalar b0 = nir_scalar_chase_alu_src(bias, 0);
               nir_scalar b1 = nir_scalar_chase_alu_src(bias, 1);
               nir_scalar sh = is_const_u32(b1, 31) ? b0 : (is_const_u32(b0, 31) ? b1 : (nir_scalar){0});
               if (sh.def && (scalar_op(sh, nir_op_ishr) || scalar_op(sh, nir_op_ushr)) &&
                   is_const_u32(nir_scalar_chase_alu_src(sh, 1), 31) &&
                   match_k_plus_31(nir_scalar_chase_alu_src(sh, 0), k_off))
                  return true;
            } else if (scalar_op(bias, nir_op_ushr)) {
               nir_scalar b0 = nir_scalar_chase_alu_src(bias, 0);
               nir_scalar b1 = nir_scalar_chase_alu_src(bias, 1);
               if (is_const_u32(b1, 27) && scalar_op(b0, nir_op_ishr) &&
                   is_const_u32(nir_scalar_chase_alu_src(b0, 1), 31) &&
                   match_k_plus_31(nir_scalar_chase_alu_src(b0, 0), k_off))
                  return true;
            }
         }
      }
   }
   /* Lowered signed idiv by 32 via isign(x) * (iabs(x) >> 5), where isign(x)
    * may be lowered to imin(imax(x, -1), 1) or imax(imin(x, 1), -1). */
   if (scalar_op(s, nir_op_imul)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      for (unsigned flip = 0; flip < 2; ++flip) {
         nir_scalar sgn = flip ? b : a;
         nir_scalar mag = flip ? a : b;
         if (!scalar_op(mag, nir_op_ushr) ||
             !is_const_u32(nir_scalar_chase_alu_src(mag, 1), 5))
            continue;
         nir_scalar abs_x = nir_scalar_chase_alu_src(mag, 0);
         if (!scalar_op(abs_x, nir_op_iabs))
            continue;
         nir_scalar x = nir_scalar_chase_alu_src(abs_x, 0);
         if (!match_k_plus_31(x, k_off))
            continue;
         bool sgn_ok = false;
         if (scalar_op(sgn, nir_op_isign)) {
            sgn_ok = match_k_plus_31(nir_scalar_chase_alu_src(sgn, 0), k_off);
         } else if (scalar_op(sgn, nir_op_imin)) {
            nir_scalar m0 = nir_scalar_chase_alu_src(sgn, 0);
            nir_scalar m1 = nir_scalar_chase_alu_src(sgn, 1);
            nir_scalar inner = is_const_u32(m1, 1) ? m0 : (is_const_u32(m0, 1) ? m1 : (nir_scalar){0});
            if (inner.def && scalar_op(inner, nir_op_imax)) {
               nir_scalar n0 = nir_scalar_chase_alu_src(inner, 0);
               nir_scalar n1 = nir_scalar_chase_alu_src(inner, 1);
               nir_scalar xv = is_const_u32(n1, UINT32_MAX) ? n0 :
                               (is_const_u32(n0, UINT32_MAX) ? n1 : (nir_scalar){0});
               sgn_ok = xv.def && match_k_plus_31(xv, k_off);
            }
         } else if (scalar_op(sgn, nir_op_imax)) {
            nir_scalar m0 = nir_scalar_chase_alu_src(sgn, 0);
            nir_scalar m1 = nir_scalar_chase_alu_src(sgn, 1);
            nir_scalar inner = is_const_u32(m1, UINT32_MAX) ? m0 : (is_const_u32(m0, UINT32_MAX) ? m1 : (nir_scalar){0});
            if (inner.def && scalar_op(inner, nir_op_imin)) {
               nir_scalar n0 = nir_scalar_chase_alu_src(inner, 0);
               nir_scalar n1 = nir_scalar_chase_alu_src(inner, 1);
               nir_scalar xv = is_const_u32(n1, 1) ? n0 :
                               (is_const_u32(n0, 1) ? n1 : (nir_scalar){0});
               sgn_ok = xv.def && match_k_plus_31(xv, k_off);
            }
         }
         if (sgn_ok)
            return true;
      }
   }
   if (scalar_op(s, nir_op_bcsel)) {
      /* bcsel(x < 0, (x + 31) >> 5, x >> 5) */
      nir_scalar c = nir_scalar_chase_alu_src(s, 0);
      nir_scalar t = nir_scalar_chase_alu_src(s, 1);
      nir_scalar f = nir_scalar_chase_alu_src(s, 2);
      if (scalar_op(c, nir_op_ilt) &&
          is_const_u32(nir_scalar_chase_alu_src(c, 1), 0) &&
          match_k_plus_31(nir_scalar_chase_alu_src(c, 0), k_off) &&
          scalar_op(f, nir_op_ishr) &&
          is_const_u32(nir_scalar_chase_alu_src(f, 1), 5) &&
          match_k_plus_31(nir_scalar_chase_alu_src(f, 0), k_off) &&
          scalar_op(t, nir_op_ishr) &&
          is_const_u32(nir_scalar_chase_alu_src(t, 1), 5))
         return true;
   }
   return false;
}

struct shared_access_idx {
   const nir_variable *var;
   bool is_2d;
   nir_scalar linear;
   nir_scalar row2d;
   nir_scalar col2d;
};

static bool
match_shared_array_access(const nir_deref_instr *d, struct shared_access_idx *out)
{
   if (!d || !nir_deref_mode_is(d, nir_var_mem_shared) ||
       d->deref_type != nir_deref_type_array)
      return false;
   const nir_deref_instr *parent = nir_deref_instr_parent(d);
   if (!parent)
      return false;
   if (parent->deref_type == nir_deref_type_var && parent->var) {
      const nir_variable *var = parent->var;
      if (!glsl_type_is_array(var->type) ||
          glsl_get_array_element(var->type) != glsl_float_type() ||
          glsl_get_length(var->type) < 1024)
         return false;
      *out = (struct shared_access_idx){
         .var = var, .is_2d = false, .linear = source(d->arr.index),
      };
      return true;
   }
   if (parent->deref_type == nir_deref_type_array) {
      const nir_deref_instr *gp = nir_deref_instr_parent(parent);
      if (!gp || gp->deref_type != nir_deref_type_var || !gp->var)
         return false;
      const nir_variable *var = gp->var;
      if (!glsl_type_is_array(var->type) || glsl_get_length(var->type) < 32)
         return false;
      const struct glsl_type *inner = glsl_get_array_element(var->type);
      if (!glsl_type_is_array(inner) ||
          glsl_get_array_element(inner) != glsl_float_type() ||
          glsl_get_length(inner) != 32)
         return false;
      *out = (struct shared_access_idx){
         .var = var, .is_2d = true,
         .row2d = source(parent->arr.index),
         .col2d = source(d->arr.index),
      };
      return true;
   }
   return false;
}

static bool
is_lid_y_mul256(nir_scalar s, const uint16_t ls[3])
{
   s = nir_scalar_chase_movs(s);
   if (scalar_op(s, nir_op_ishl)) {
      return is_local_id_axis(nir_scalar_chase_alu_src(s, 0), ls, 1) &&
             is_const_u32(nir_scalar_chase_alu_src(s, 1), 8);
   }
   if (scalar_op(s, nir_op_imul)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      return (is_local_id_axis(a, ls, 1) && is_const_u32(b, 256)) ||
             (is_local_id_axis(b, ls, 1) && is_const_u32(a, 256));
   }
   return false;
}

static bool
match_shared_tile_idx(const struct shared_access_idx *acc, const uint16_t ls[3],
                      uint32_t *j_out)
{
   if (acc->is_2d) {
      return is_local_id_axis(acc->col2d, ls, 0) &&
             match_lid_row8(acc->row2d, ls, j_out);
   }
   nir_scalar terms[6];
   unsigned n = 0;
   if (!flatten_iadd_terms(acc->linear, terms, &n, 6, 4))
      return false;
   bool got_lx = false, got_row_mul32 = false, got_ly256 = false;
   uint32_t row_j = 0;
   int64_t csum = 0;
   for (unsigned i = 0; i < n; ++i) {
      nir_scalar row;
      if (!got_lx && is_local_id_axis(terms[i], ls, 0)) {
         got_lx = true;
      } else if (!got_row_mul32 && !got_ly256 &&
                 match_mul32(terms[i], &row) && match_lid_row8(row, ls, &row_j)) {
         got_row_mul32 = true;
      } else if (!got_row_mul32 && !got_ly256 && is_lid_y_mul256(terms[i], ls)) {
         got_ly256 = true;
      } else if (nir_scalar_is_const(terms[i]) && terms[i].def->bit_size == 32) {
         csum += (int32_t)nir_scalar_as_uint(terms[i]);
      } else {
         return false;
      }
   }
   if (!got_lx)
      return false;
   if (got_row_mul32) {
      if (csum != 0)
         return false;
      *j_out = row_j;
      return true;
   }
   if (got_ly256) {
      if (csum < 0 || csum >= 256 || (csum % 32) != 0)
         return false;
      *j_out = (uint32_t)(csum / 32);
      return true;
   }
   return false;
}

static bool
match_tile_k_plus_lid_row(nir_scalar s, const uint16_t ls[3],
                          const nir_phi_instr *tile_phi,
                          const nir_phi_instr *k_idx_phi, uint32_t expected_j)
{
   nir_scalar terms[6];
   unsigned n = 0;
   if (!flatten_iadd_terms(s, terms, &n, 6, 4))
      return false;
   bool got_k = false, got_lid_y = false;
   int64_t csum = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (!got_k && is_tile_k_base(terms[i], tile_phi, k_idx_phi)) {
         got_k = true;
      } else if (!got_lid_y && is_lid_y_mul8(terms[i], ls)) {
         got_lid_y = true;
      } else if (nir_scalar_is_const(terms[i]) && terms[i].def->bit_size == 32) {
         csum += (int32_t)nir_scalar_as_uint(terms[i]);
      } else {
         return false;
      }
   }
   return got_k && got_lid_y && csum == (int64_t)expected_j;
}

static bool
match_a_ssbo_idx(nir_scalar idx, const uint16_t ls[3], uint32_t k_off,
                 const nir_phi_instr *tile_phi, const nir_phi_instr *k_idx_phi,
                 uint32_t j)
{
   nir_scalar terms[6];
   unsigned n = 0;
   if (!flatten_iadd_terms(idx, terms, &n, 6, 4))
      return false;
   bool got_row_k = false, got_tile_k = false, got_lid_x = false;
   for (unsigned i = 0; i < n; ++i) {
      if (is_const_u32(terms[i], 0))
         continue;
      if (!got_tile_k && is_tile_k_base(terms[i], tile_phi, k_idx_phi)) {
         got_tile_k = true;
      } else if (!got_lid_x && is_local_id_axis(terms[i], ls, 0)) {
         got_lid_x = true;
      } else if (!got_row_k && scalar_op(terms[i], nir_op_imul)) {
         nir_scalar a = nir_scalar_chase_alu_src(terms[i], 0);
         nir_scalar b = nir_scalar_chase_alu_src(terms[i], 1);
         uint32_t r, rj, off;
         if ((match_gemm_row(a, ls, &r, &rj) && r == 8 && rj == j &&
              push_count(b, &off) && off == k_off) ||
             (match_gemm_row(b, ls, &r, &rj) && r == 8 && rj == j &&
              push_count(a, &off) && off == k_off))
            got_row_k = true;
         else
            return false;
      } else {
         return false;
      }
   }
   return got_row_k && got_tile_k && got_lid_x;
}

static bool
match_b_ssbo_idx(nir_scalar idx, const uint16_t ls[3], uint32_t n_off,
                 const nir_phi_instr *tile_phi, const nir_phi_instr *k_idx_phi,
                 uint32_t j)
{
   nir_scalar terms[4];
   unsigned n = 0;
   if (!flatten_iadd_terms(idx, terms, &n, 4, 3))
      return false;
   bool got_row_n = false, got_col = false;
   for (unsigned i = 0; i < n; ++i) {
      if (is_const_u32(terms[i], 0))
         continue;
      if (!got_col && is_global_id(terms[i], ls, 0)) {
         got_col = true;
      } else if (!got_row_n && scalar_op(terms[i], nir_op_imul)) {
         nir_scalar a = nir_scalar_chase_alu_src(terms[i], 0);
         nir_scalar b = nir_scalar_chase_alu_src(terms[i], 1);
         uint32_t off;
         if ((match_tile_k_plus_lid_row(a, ls, tile_phi, k_idx_phi, j) &&
              push_count(b, &off) && off == n_off) ||
             (match_tile_k_plus_lid_row(b, ls, tile_phi, k_idx_phi, j) &&
              push_count(a, &off) && off == n_off))
            got_row_n = true;
         else
            return false;
      } else {
         return false;
      }
   }
   return got_row_n && got_col;
}

static bool
match_gemm_shared_guard(nir_if *branch, bool is_a, const uint16_t ls[3],
                        uint32_t m_off, uint32_t n_off, uint32_t k_off,
                        const nir_phi_instr *tile_phi,
                        const nir_phi_instr *k_idx_phi, uint32_t j)
{
   nir_scalar p0 = {0}, p1 = {0};
   if (!match_guard_cond_pair(branch, ls, &p0, &p1))
      return false;
   if ((!scalar_op(p0, nir_op_ilt) && !scalar_op(p0, nir_op_ult)) ||
       (!scalar_op(p1, nir_op_ilt) && !scalar_op(p1, nir_op_ult)))
      return false;

   nir_scalar pairs[2][2] = {
      { nir_scalar_chase_alu_src(p0, 0), nir_scalar_chase_alu_src(p0, 1) },
      { nir_scalar_chase_alu_src(p1, 0), nir_scalar_chase_alu_src(p1, 1) },
   };
   bool got0 = false, got1 = false;
   for (unsigned i = 0; i < 2; ++i) {
      uint32_t off, r, rj;
      if (is_a) {
         if (match_gemm_row(pairs[i][0], ls, &r, &rj) && r == 8 && rj == j &&
             push_count(pairs[i][1], &off) && off == m_off) {
            got0 = true;
         } else if (push_count(pairs[i][1], &off) && off == k_off) {
            nir_scalar t[4];
            unsigned tn = 0;
            bool tk = false, lx = false;
            if (flatten_iadd_terms(pairs[i][0], t, &tn, 4, 3)) {
               bool ok = true;
               for (unsigned ti = 0; ti < tn; ++ti) {
                  if (is_const_u32(t[ti], 0))
                     continue;
                  else if (!tk && is_tile_k_base(t[ti], tile_phi, k_idx_phi))
                     tk = true;
                  else if (!lx && is_local_id_axis(t[ti], ls, 0))
                     lx = true;
                  else
                     ok = false;
               }
               if (ok && tk && lx)
                  got1 = true;
            }
         }
      } else {
         if (is_global_id(pairs[i][0], ls, 0) &&
             push_count(pairs[i][1], &off) && off == n_off) {
            got0 = true;
         } else if (match_tile_k_plus_lid_row(pairs[i][0], ls, tile_phi, k_idx_phi, j) &&
                    push_count(pairs[i][1], &off) && off == k_off) {
            got1 = true;
         }
      }
   }
   return got0 && got1;
}

static bool
match_dot_step(nir_scalar acc_in, nir_scalar acc_prev,
               const nir_variable *s_a, const nir_variable *s_b,
               const uint16_t ls[3], uint32_t j,
               const nir_phi_instr *l_phi, uint32_t *kk_out)
{
   acc_in = nir_scalar_chase_movs(acc_in);
   nir_scalar op_a = {0}, op_b = {0};
   if (scalar_op(acc_in, nir_op_ffma)) {
      const nir_alu_instr *alu = nir_def_as_alu(acc_in.def);
      if (alu->fp_math_ctrl != 0)
         return false;
      nir_scalar s2 = nir_scalar_chase_alu_src(acc_in, 2);
      if (s2.def != acc_prev.def || s2.comp != acc_prev.comp)
         return false;
      op_a = nir_scalar_chase_alu_src(acc_in, 0);
      op_b = nir_scalar_chase_alu_src(acc_in, 1);
   } else if (scalar_op(acc_in, nir_op_fadd)) {
      const nir_alu_instr *alu = nir_def_as_alu(acc_in.def);
      if (alu->fp_math_ctrl != 0)
         return false;
      nir_scalar s0 = nir_scalar_chase_alu_src(acc_in, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(acc_in, 1);
      nir_scalar prod = {0};
      if (s0.def == acc_prev.def && s0.comp == acc_prev.comp)
         prod = s1;
      else if (s1.def == acc_prev.def && s1.comp == acc_prev.comp)
         prod = s0;
      else
         return false;
      if (!scalar_op(prod, nir_op_fmul) || nir_def_as_alu(prod.def)->fp_math_ctrl != 0)
         return false;
      op_a = nir_scalar_chase_alu_src(prod, 0);
      op_b = nir_scalar_chase_alu_src(prod, 1);
   } else {
      return false;
   }

   const nir_intrinsic_instr *ld0 = intrinsic(op_a.def, nir_intrinsic_load_deref);
   const nir_intrinsic_instr *ld1 = intrinsic(op_b.def, nir_intrinsic_load_deref);
   if (!ld0 || !ld1 || !nir_def_is_deref(ld0->src[0].ssa) ||
       !nir_def_is_deref(ld1->src[0].ssa))
      return false;
   struct shared_access_idx acc0 = {0}, acc1 = {0};
   if (!match_shared_array_access(nir_def_as_deref(ld0->src[0].ssa), &acc0) ||
       !match_shared_array_access(nir_def_as_deref(ld1->src[0].ssa), &acc1))
      return false;
   const struct shared_access_idx *a_acc = NULL, *b_acc = NULL;
   if (acc0.var == s_a && acc1.var == s_b) {
      a_acc = &acc0;
      b_acc = &acc1;
   } else if (acc0.var == s_b && acc1.var == s_a) {
      a_acc = &acc1;
      b_acc = &acc0;
   } else {
      return false;
   }

   /* Match A: s_a[lid_y * 8 + j, kk] */
   bool a_has_const_k = false;
   uint32_t a_k_const = 0;
   if (a_acc->is_2d) {
      uint32_t aj = 0;
      if (!match_lid_row8(a_acc->row2d, ls, &aj) || aj != j)
         return false;
      if (l_phi) {
         if (!is_phi_scalar(a_acc->col2d, l_phi))
            return false;
      } else {
         nir_scalar ck = nir_scalar_chase_movs(a_acc->col2d);
         if (!nir_scalar_is_const(ck) || ck.def->bit_size != 32)
            return false;
         a_k_const = (uint32_t)nir_scalar_as_uint(ck);
         a_has_const_k = true;
      }
   } else {
      nir_scalar terms[6];
      unsigned n = 0;
      if (!flatten_iadd_terms(a_acc->linear, terms, &n, 6, 4))
         return false;
      bool got_row_mul32 = false, got_ly256 = false, got_l = false;
      int64_t csum = 0;
      for (unsigned i = 0; i < n; ++i) {
         nir_scalar row;
         uint32_t aj = 0;
         if (l_phi && !got_l && is_phi_scalar(terms[i], l_phi)) {
            got_l = true;
         } else if (!got_row_mul32 && !got_ly256 &&
                    match_mul32(terms[i], &row) &&
                    match_lid_row8(row, ls, &aj) && aj == j) {
            got_row_mul32 = true;
         } else if (!got_row_mul32 && !got_ly256 && is_lid_y_mul256(terms[i], ls)) {
            got_ly256 = true;
         } else if (nir_scalar_is_const(terms[i]) && terms[i].def->bit_size == 32) {
            csum += (int32_t)nir_scalar_as_uint(terms[i]);
         } else {
            return false;
         }
      }
      if (!got_row_mul32 && !got_ly256)
         return false;
      if (l_phi) {
         if (!got_l)
            return false;
         if (got_row_mul32 && csum != 0)
            return false;
         if (got_ly256 && csum != (int64_t)j * 32)
            return false;
      } else {
         int64_t k_val = got_row_mul32 ? csum : (csum - (int64_t)j * 32);
         if (k_val < 0 || k_val >= 32)
            return false;
         a_k_const = (uint32_t)k_val;
         a_has_const_k = true;
      }
   }

   /* Match B: s_b[kk, lid_x] */
   uint32_t b_k_const = 0;
   if (b_acc->is_2d) {
      if (!is_local_id_axis(b_acc->col2d, ls, 0))
         return false;
      if (l_phi) {
         if (!is_phi_scalar(b_acc->row2d, l_phi))
            return false;
         return true;
      }
      nir_scalar ck = nir_scalar_chase_movs(b_acc->row2d);
      if (!nir_scalar_is_const(ck) || ck.def->bit_size != 32)
         return false;
      b_k_const = (uint32_t)nir_scalar_as_uint(ck);
   } else {
      nir_scalar terms[4];
      unsigned n = 0;
      if (!flatten_iadd_terms(b_acc->linear, terms, &n, 4, 3))
         return false;
      bool got_lx = false, got_l32 = false;
      int64_t csum = 0;
      for (unsigned i = 0; i < n; ++i) {
         nir_scalar kk;
         if (!got_lx && is_local_id_axis(terms[i], ls, 0)) {
            got_lx = true;
         } else if (l_phi && !got_l32 && match_mul32(terms[i], &kk) &&
                    is_phi_scalar(kk, l_phi)) {
            got_l32 = true;
         } else if (nir_scalar_is_const(terms[i]) && terms[i].def->bit_size == 32) {
            csum += (int32_t)nir_scalar_as_uint(terms[i]);
         } else {
            return false;
         }
      }
      if (!got_lx)
         return false;
      if (l_phi)
         return got_l32 && csum == 0;
      if (csum < 0 || (csum % 32) != 0 || (csum / 32) >= 32)
         return false;
      b_k_const = (uint32_t)(csum / 32);
   }

   if (!a_has_const_k || a_k_const != b_k_const || a_k_const >= 32)
      return false;
   if (kk_out)
      *kk_out = a_k_const;
   return true;
}

static unsigned
count_cf_loops(const struct exec_list *list)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop) {
         n += 1 + count_cf_loops(&nir_cf_node_as_loop(node)->body);
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         n += count_cf_loops(&nif->then_list);
         n += count_cf_loops(&nif->else_list);
      }
   }
   return n;
}

static bool
match_loop_iv_less_than(nir_loop *loop, const nir_phi_instr *iv,
                        bool match_const_bound, uint32_t const_bound,
                        bool match_ceil32_k, uint32_t k_off,
                        bool match_raw_k,
                        const nir_instr **exit_break_out)
{
   nir_block *hdr = nir_loop_first_block(loop);
   nir_cf_node *next = nir_cf_node_next(&hdr->cf_node);
   if (!next || next->type != nir_cf_node_if)
      return false;
   nir_if *br = nir_cf_node_as_if(next);
   nir_block *tb = nir_if_first_then_block(br);
   nir_block *eb = nir_if_first_else_block(br);
   bool then_break =
      exec_list_is_singular(&br->then_list) &&
      !exec_list_is_empty(&tb->instr_list) &&
      nir_block_last_instr(tb)->type == nir_instr_type_jump &&
      nir_instr_as_jump(nir_block_last_instr(tb))->type == nir_jump_break;
   bool else_break =
      exec_list_is_singular(&br->else_list) &&
      !exec_list_is_empty(&eb->instr_list) &&
      nir_block_last_instr(eb)->type == nir_instr_type_jump &&
      nir_instr_as_jump(nir_block_last_instr(eb))->type == nir_jump_break;
   if (then_break == else_break)
      return false;

   const nir_instr *break_instr =
      then_break ? nir_block_last_instr(tb) : nir_block_last_instr(eb);

   nir_scalar cond = source(br->condition);
   if (then_break && scalar_op(cond, nir_op_inot)) {
      cond = nir_scalar_chase_alu_src(cond, 0);
      then_break = false;
      else_break = true;
   }

   nir_scalar lhs = {0}, rhs = {0};
   if (else_break) {
      if (!scalar_op(cond, nir_op_ilt) && !scalar_op(cond, nir_op_ult))
         return false;
      lhs = nir_scalar_chase_alu_src(cond, 0);
      rhs = nir_scalar_chase_alu_src(cond, 1);
   } else {
      if (!scalar_op(cond, nir_op_ige) && !scalar_op(cond, nir_op_uge))
         return false;
      lhs = nir_scalar_chase_alu_src(cond, 0);
      rhs = nir_scalar_chase_alu_src(cond, 1);
   }
   if (!is_phi_scalar(lhs, iv))
      return false;
   bool bound_ok = false;
   if (match_const_bound)
      bound_ok = is_const_u32(rhs, const_bound);
   else if (match_ceil32_k)
      bound_ok = match_ceil_div_32(rhs, k_off);
   else if (match_raw_k) {
      uint32_t off;
      bound_ok = push_count(rhs, &off) && off == k_off;
   }
   if (!bound_ok)
      return false;
   if (exit_break_out)
      *exit_break_out = break_instr;
   return true;
}

static bool
is_wg_shared_barrier(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   const nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
   return op->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(op) == SCOPE_WORKGROUP &&
          (nir_intrinsic_memory_modes(op) & nir_var_mem_shared);
}

/* Match a leaf C-store value against one of:
 *   bias_kind == 0: acc_phi[j] + bias[row_j]
 *   bias_kind == 1: acc_phi[j] + bias[col]
 *   bias_kind == 2: acc_phi[j] (no bias)
 */
static const nir_phi_instr *
match_c_store_bias_leaf(nir_scalar sv, const nir_block *k_hdr,
                        const uint16_t ls[3], uint32_t j,
                        bool has_bias_buffer,
                        struct hk_nir_clamp_buffer bias_binding,
                        unsigned desc_comps, int *bias_kind_out)
{
   sv = nir_scalar_chase_movs(sv);
   if (nir_def_is_phi(sv.def) && nir_def_instr(sv.def)->block == k_hdr) {
      *bias_kind_out = 2;
      return nir_def_as_phi(sv.def);
   }
   if (scalar_op(sv, nir_op_fadd) &&
       nir_def_as_alu(sv.def)->fp_math_ctrl == 0 && has_bias_buffer) {
      for (unsigned s = 0; s < 2; ++s) {
         nir_scalar op0 = nir_scalar_chase_alu_src(sv, s);
         nir_scalar op1 = nir_scalar_chase_alu_src(sv, 1 - s);
         if (!nir_def_is_phi(op0.def) || nir_def_instr(op0.def)->block != k_hdr)
            continue;
         const nir_intrinsic_instr *bld =
            intrinsic(op1.def, nir_intrinsic_load_deref);
         struct buffer_access bacc;
         if (!bld || !storage_read(bld, desc_comps, &bacc) ||
             !same_binding(bacc.binding, bias_binding))
            continue;
         uint32_t br = 0, bj = 0;
         if (is_global_id(bacc.element, ls, 0)) {
            *bias_kind_out = 1;
            return nir_def_as_phi(op0.def);
         }
         if (match_gemm_row(bacc.element, ls, &br, &bj) && br == 8 && bj == j) {
            *bias_kind_out = 0;
            return nir_def_as_phi(op0.def);
         }
      }
   }
   return NULL;
}

/* Recursively verify a non-split C-store value that merges bias_mode arms via
 * phis: every then-arm of `bias_mode == m` must produce the exact matching
 * bias kind (0=row, 1=col, 2=none), and every fallthrough else-leaf must be
 * plain acc_phi (kind 2). */
static const nir_phi_instr *
extract_dyn_bias_phi(nir_scalar sv, const nir_block *src_block,
                     const nir_block *k_hdr, const uint16_t ls[3], uint32_t j,
                     bool has_bias_buffer,
                     struct hk_nir_clamp_buffer bias_binding,
                     uint32_t bias_mode_off, unsigned desc_comps,
                     uint8_t *seen_modes, unsigned depth)
{
   sv = nir_scalar_chase_movs(sv);
   if (depth > 4)
      return NULL;
   if (nir_def_is_phi(sv.def) && nir_def_instr(sv.def)->block != k_hdr) {
      nir_phi_instr *p = nir_def_as_phi(sv.def);
      nir_cf_node *prev = nir_cf_node_prev(&p->instr.block->cf_node);
      uint32_t if_off = 0, if_mode = 0;
      if (!prev || prev->type != nir_cf_node_if ||
          !match_bias_mode_if(nir_cf_node_as_if(prev), &if_off, &if_mode) ||
          if_off != bias_mode_off)
         return NULL;

      const nir_phi_instr *common = NULL;
      nir_foreach_phi_src(src, p) {
         const nir_phi_instr *sub = extract_dyn_bias_phi(
            source(src->src), src->pred, k_hdr, ls, j, has_bias_buffer,
            bias_binding, bias_mode_off, desc_comps, seen_modes, depth + 1);
         if (!sub || (common && common != sub))
            return NULL;
         common = sub;
      }
      return common;
   }

   int kind = -1;
   const nir_phi_instr *acc = match_c_store_bias_leaf(
      sv, k_hdr, ls, j, has_bias_buffer, bias_binding, desc_comps, &kind);
   if (!acc)
      return NULL;

   uint32_t arm_off = 0, arm_mode = 0;
   if (find_bias_mode_arm(&((nir_block *)src_block)->cf_node, &arm_off,
                          &arm_mode)) {
      if (arm_off != bias_mode_off || kind != (int)arm_mode)
         return NULL;
      *seen_modes |= (uint8_t)(1u << arm_mode);
   } else {
      /* Fallthrough else arm corresponds to no-bias (mode 2). */
      if (kind != 2)
         return NULL;
      *seen_modes |= (uint8_t)(1u << 2);
   }
   return acc;
}

static bool
is_shared_load(const nir_intrinsic_instr *op)
{
   return op->intrinsic == nir_intrinsic_load_deref &&
          nir_def_is_deref(op->src[0].ssa) &&
          nir_deref_mode_is(nir_def_as_deref(op->src[0].ssa),
                            nir_var_mem_shared);
}

/* True if a shared-memory load lies outside the CF subtree `node`. */
static bool
has_shared_load_outside(const nir_function_impl *impl, const nir_cf_node *node)
{
   nir_foreach_block(blk, (nir_function_impl *)impl) {
      const nir_cf_node *cur = &blk->cf_node;
      while (cur && cur != node)
         cur = cur->parent;
      if (cur)
         continue;
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_intrinsic &&
             is_shared_load(nir_instr_as_intrinsic(instr)))
            return true;
      }
   }
   return false;
}

/* The cmat alternate writes every in-bounds C element exactly once per
 * dispatch, so a certified C store may only be predicated by what the
 * alternate reproduces: the per-element `row < M && col < N` guard (guarded
 * form: the store's immediate parent, then-arm) and, for stores split per
 * bias_mode arm, the `bias_mode == mode` dispatch chain (the then-arm of
 * exactly one `bias_mode == mode` if, plus the else-arms of `bias_mode == m`
 * ifs with m != mode). Any other enclosing if or loop, at any depth up to
 * the function body, can skip or repeat the store in the original. */
static bool
c_store_cf_accounted(const nir_intrinsic_instr *store, const uint16_t ls[3],
                     uint32_t j, uint32_t n_off, bool is_guarded,
                     uint32_t m_off, bool stores_split_by_bias_mode,
                     uint32_t bias_mode_off, uint32_t mode)
{
   nir_cf_node *store_block = &store->instr.block->cf_node;
   bool seen_guard = false, seen_arm = false;

   for (nir_cf_node *cur = store_block;
        cur->parent && cur->parent->type != nir_cf_node_function;
        cur = cur->parent) {
      if (cur->parent->type != nir_cf_node_if)
         return false;
      nir_if *branch = nir_cf_node_as_if(cur->parent);
      const bool then_arm = in_then(cur, branch);

      uint32_t guard_m_off = UINT32_MAX;
      if (is_guarded && cur == store_block && then_arm &&
          match_gemm_store_guard(branch, ls, 8, j, n_off, &guard_m_off) &&
          guard_m_off == m_off) {
         seen_guard = true;
         continue;
      }

      uint32_t if_off = 0, if_mode = 0;
      if (stores_split_by_bias_mode &&
          match_bias_mode_if(branch, &if_off, &if_mode) &&
          if_off == bias_mode_off) {
         if (then_arm && !seen_arm && if_mode == mode) {
            seen_arm = true;
            continue;
         }
         if (!then_arm && if_mode != mode)
            continue;
      }
      return false;
   }

   return seen_guard == is_guarded && seen_arm == stores_split_by_bias_mode;
}

/* Certify that a matched 32x4x1, 8-row/thread GEMM producer actually executes
 * a standard 32x32 shared-memory tiled row-major FP32 GEMM K-loop
 * `C[row, col] = sum_{k=0..K-1} A[row*K + k] * B[k*N + col] (+ bias)` with
 * default FP32 math controls.
 *
 * Floating-point exactness requirement: the certified original executes a
 * sequential FP32 ffma (or fmul+fadd) chain per output element under default
 * Vulkan math controls, which permit FMA contraction but NOT reassociation or
 * reordering of the summation. Replacement by AGX `f32matmad8x8` therefore
 * relies on hardware `f32matmad8x8` being bit-exact with that sequential FP32
 * FMA accumulation order (including NaN/Inf propagation, overflow, and
 * denormals). */
static bool
certify_gemm_row_major_k_loop(const nir_function_impl *impl,
                              const uint16_t ls[3],
                              const nir_intrinsic_instr *const *c_stores,
                              const struct buffer_access *c_access,
                              unsigned num_c_stores,
                              struct hk_nir_clamp_buffer a_binding,
                              struct hk_nir_clamp_buffer b_binding,
                              bool has_bias_buffer,
                              struct hk_nir_clamp_buffer bias_binding,
                              bool stores_split_by_bias_mode,
                              bool has_bias_mode, uint32_t bias_mode_off,
                              uint8_t bias_mode_mask,
                              bool is_guarded,
                              uint32_t m_off, uint32_t n_off, uint32_t k_off,
                              unsigned desc_comps)
{
   /* 1. Require exactly one top-level K-tile loop in the entrypoint body. */
   nir_loop *k_loop = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_loop) {
         if (k_loop)
            return false;
         k_loop = nir_cf_node_as_loop(node);
      }
   }
   if (!k_loop)
      return false;

   nir_block *k_pre = nir_cf_node_as_block(nir_cf_node_prev(&k_loop->cf_node));
   nir_block *k_hdr = nir_loop_first_block(k_loop);
   nir_block *k_back = nir_loop_last_block(k_loop);

   /* 2. Map every C store (rows j = 0..7) back to its outer-loop acc_phi[j]
    *    and enforce the exact bias convention (0=row, 1=col, 2=none) on both
    *    split-store and non-split paths. */
   const nir_phi_instr *acc_phi[8] = {0};
   for (unsigned i = 0; i < num_c_stores; ++i) {
      uint32_t r = 0, j = 0, store_n_off = 0;
      if (!match_gemm_c_index(c_access[i].element, ls, &r, &j, &store_n_off) ||
          r != 8 || j >= 8 || store_n_off != n_off)
         return false;

      const nir_phi_instr *phi = NULL;
      uint32_t arm_mode = 0;
      if (stores_split_by_bias_mode) {
         uint32_t arm_off = 0;
         int kind = -1;
         if (!find_bias_mode_arm(&c_stores[i]->instr.block->cf_node, &arm_off,
                                 &arm_mode) ||
             arm_off != bias_mode_off)
            return false;
         phi = match_c_store_bias_leaf(source(c_stores[i]->src[1]), k_hdr, ls, j,
                                       has_bias_buffer, bias_binding, desc_comps,
                                       &kind);
         if (!phi || kind != (int)arm_mode)
            return false;
      } else if (!has_bias_mode) {
         int want_kind = -1;
         if (!has_bias_buffer)
            want_kind = 2;
         else if (bias_mode_mask == (1u << 0))
            want_kind = 0;
         else if (bias_mode_mask == (1u << 1))
            want_kind = 1;
         else
            return false;
         int kind = -1;
         phi = match_c_store_bias_leaf(source(c_stores[i]->src[1]), k_hdr, ls, j,
                                       has_bias_buffer, bias_binding, desc_comps,
                                       &kind);
         if (!phi || kind != want_kind)
            return false;
      } else {
         uint8_t seen_modes = 0;
         phi = extract_dyn_bias_phi(source(c_stores[i]->src[1]),
                                    c_stores[i]->instr.block, k_hdr, ls, j,
                                    has_bias_buffer, bias_binding, bias_mode_off,
                                    desc_comps, &seen_modes, 0);
         if (!phi || seen_modes != bias_mode_mask)
            return false;
      }

      if (!c_store_cf_accounted(c_stores[i], ls, j, n_off, is_guarded, m_off,
                                stores_split_by_bias_mode, bias_mode_off,
                                arm_mode))
         return false;

      if (acc_phi[j] && acc_phi[j] != phi)
         return false;
      acc_phi[j] = phi;
   }

   for (unsigned j = 0; j < 8; ++j) {
      if (!acc_phi[j])
         return false;
      for (unsigned k = 0; k < j; ++k) {
         if (acc_phi[k] == acc_phi[j])
            return false;
      }
      /* Pre-header accumulator must initialize to +0.0f. */
      bool pre_zero = false, got_back = false;
      nir_foreach_phi_src(src, acc_phi[j]) {
         if (src->pred == k_pre)
            pre_zero = is_const_u32(source(src->src), 0x00000000u);
         else if (src->pred == k_back)
            got_back = true;
         else
            return false;
      }
      if (!pre_zero || !got_back)
         return false;
   }

   /* 3. Classify outer-loop induction variables (tile_phi +1, k_idx_phi +32)
    *    and verify the K-tile exit bound. */
   const nir_phi_instr *tile_phi = NULL;
   const nir_phi_instr *k_idx_phi = NULL;
   nir_foreach_phi(phi, k_hdr) {
      bool is_acc = false;
      for (unsigned j = 0; j < 8; ++j) {
         if (acc_phi[j] == phi)
            is_acc = true;
      }
      if (is_acc)
         continue;
      nir_scalar pre_v = {0}, back_v = {0};
      bool got_pre = false, got_back = false;
      nir_foreach_phi_src(src, phi) {
         if (src->pred == k_pre) {
            pre_v = source(src->src);
            got_pre = true;
         } else if (src->pred == k_back) {
            back_v = source(src->src);
            got_back = true;
         } else {
            return false;
         }
      }
      if (!got_pre || !got_back || !is_const_u32(pre_v, 0) ||
          !scalar_op(back_v, nir_op_iadd))
         return false;
      nir_scalar a = nir_scalar_chase_alu_src(back_v, 0);
      nir_scalar b = nir_scalar_chase_alu_src(back_v, 1);
      nir_scalar step = {0};
      if (is_phi_scalar(a, phi))
         step = b;
      else if (is_phi_scalar(b, phi))
         step = a;
      else
         return false;
      if (is_const_u32(step, 1)) {
         if (tile_phi)
            return false;
         tile_phi = phi;
      } else if (is_const_u32(step, 32)) {
         if (k_idx_phi)
            return false;
         k_idx_phi = phi;
      } else {
         return false;
      }
   }
   if (!tile_phi && !k_idx_phi)
      return false;

   const nir_instr *k_exit_break = NULL;
   bool exit_ok = false;
   if (tile_phi &&
       match_loop_iv_less_than(k_loop, tile_phi, false, 0, true, k_off, false,
                               &k_exit_break))
      exit_ok = true;
   else if (k_idx_phi &&
            match_loop_iv_less_than(k_loop, k_idx_phi, false, 0, false, k_off, true,
                                    &k_exit_break))
      exit_ok = true;
   if (!exit_ok || !k_exit_break)
      return false;

   /* 4. Partition k_loop body into Phase 1 (before first WG barrier),
    *    Phase 2 (between first and second WG barriers), and tail. */
   unsigned barrier_stage = 0; /* 0 = Phase 1, 1 = Phase 2, 2 = after barrier_2 */
   nir_loop *dot_loop = NULL;
   const nir_variable *s_a = NULL, *s_b = NULL;
   uint8_t a_load_mask = 0, a_zero_mask = 0;
   uint8_t b_load_mask = 0, b_zero_mask = 0;

   foreach_list_typed(nir_cf_node, cf, node, &k_loop->body) {
      if (cf->type == nir_cf_node_loop) {
         if (barrier_stage != 1 || dot_loop)
            return false;
         dot_loop = nir_cf_node_as_loop(cf);
         continue;
      }

      /* Walk every block inside this top-level CF node of k_loop. */
      nir_block *start = (cf->type == nir_cf_node_block)
         ? nir_cf_node_as_block(cf)
         : nir_cf_node_cf_tree_first(cf);
      nir_block *end = (cf->type == nir_cf_node_block)
         ? start
         : nir_cf_node_cf_tree_last(cf);

      for (nir_block *blk = start; blk; blk = (blk == end) ? NULL : nir_block_cf_tree_next(blk)) {
         nir_foreach_instr(instr, blk) {
            if (is_wg_shared_barrier(instr)) {
               if (cf->type != nir_cf_node_block)
                  return false;
               if (barrier_stage < 2)
                  barrier_stage++;
               else
                  return false;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
            if (is_shared_load(op)) {
               /* Tile reads belong to Phase 2 only. The dot_loop form ties
                * its loads to l_phi; the unrolled form uses constant columns
                * and would otherwise also accept a read before barrier_1. */
               if (barrier_stage != 1)
                  return false;
               continue;
            }
            if (op->intrinsic != nir_intrinsic_store_deref)
               continue;
            if (!nir_def_is_deref(op->src[0].ssa))
               return false;
            const nir_deref_instr *dst = nir_def_as_deref(op->src[0].ssa);
            /* No SSBO stores allowed inside the K-loop, and shared stores only in Phase 1. */
            if (!nir_deref_mode_is(dst, nir_var_mem_shared) || barrier_stage != 0)
               return false;
            if (op->num_components != 1 || op->src[1].ssa->bit_size != 32 ||
                nir_intrinsic_write_mask(op) != 1)
               return false;

            struct shared_access_idx sh_acc = {0};
            uint32_t j = 0;
            if (!match_shared_array_access(dst, &sh_acc) ||
                !match_shared_tile_idx(&sh_acc, ls, &j) || j >= 8)
               return false;
            const nir_variable *var = sh_acc.var;

            nir_scalar val = source(op->src[1]);
            const nir_intrinsic_instr *ld = intrinsic(val.def, nir_intrinsic_load_deref);
            bool is_zero = is_const_u32(val, 0x00000000u);

            if (!is_guarded) {
               if (cf->type != nir_cf_node_block || !ld)
                  return false;
               struct buffer_access acc;
               if (!storage_read(ld, desc_comps, &acc))
                  return false;
               if (same_binding(acc.binding, a_binding) &&
                   match_a_ssbo_idx(acc.element, ls, k_off, tile_phi, k_idx_phi, j)) {
                  if (s_a && s_a != var)
                     return false;
                  s_a = var;
                  if (a_load_mask & (1u << j))
                     return false;
                  a_load_mask |= (uint8_t)(1u << j);
               } else if (same_binding(acc.binding, b_binding) &&
                          match_b_ssbo_idx(acc.element, ls, n_off, tile_phi, k_idx_phi, j)) {
                  if (s_b && s_b != var)
                     return false;
                  s_b = var;
                  if (b_load_mask & (1u << j))
                     return false;
                  b_load_mask |= (uint8_t)(1u << j);
               } else {
                  return false;
               }
            } else {
               /* Guarded: either direct if/else stores, or phi(load, 0.0f) store. */
               if (cf->type == nir_cf_node_if) {
                  nir_if *br = nir_cf_node_as_if(cf);
                  /* The store must sit directly in an arm of the guard `br`.
                   * A store inside a nested `if` in either arm carries a
                   * predicate that match_gemm_shared_guard() never sees.
                   * Nested control flow next to a direct store cannot skip
                   * it: the only jumps allowed are the loop header breaks
                   * (checked below). */
                  if (blk->cf_node.parent != &br->cf_node)
                     return false;
                  bool in_t = in_then(&blk->cf_node, br);
                  if (in_t && ld) {
                     struct buffer_access acc;
                     if (!storage_read(ld, desc_comps, &acc))
                        return false;
                     if (same_binding(acc.binding, a_binding) &&
                         match_a_ssbo_idx(acc.element, ls, k_off, tile_phi, k_idx_phi, j) &&
                         match_gemm_shared_guard(br, true, ls, m_off, n_off, k_off,
                                                 tile_phi, k_idx_phi, j)) {
                        if (s_a && s_a != var)
                           return false;
                        s_a = var;
                        if (a_load_mask & (1u << j))
                           return false;
                        a_load_mask |= (uint8_t)(1u << j);
                     } else if (same_binding(acc.binding, b_binding) &&
                                match_b_ssbo_idx(acc.element, ls, n_off, tile_phi, k_idx_phi, j) &&
                                match_gemm_shared_guard(br, false, ls, m_off, n_off, k_off,
                                                        tile_phi, k_idx_phi, j)) {
                        if (s_b && s_b != var)
                           return false;
                        s_b = var;
                        if (b_load_mask & (1u << j))
                           return false;
                        b_load_mask |= (uint8_t)(1u << j);
                     } else {
                        return false;
                     }
                  } else if (!in_t && is_zero) {
                     if (match_gemm_shared_guard(br, true, ls, m_off, n_off, k_off,
                                                 tile_phi, k_idx_phi, j)) {
                        if (s_a && s_a != var)
                           return false;
                        s_a = var;
                        if (a_zero_mask & (1u << j))
                           return false;
                        a_zero_mask |= (uint8_t)(1u << j);
                     } else if (match_gemm_shared_guard(br, false, ls, m_off, n_off, k_off,
                                                        tile_phi, k_idx_phi, j)) {
                        if (s_b && s_b != var)
                           return false;
                        s_b = var;
                        if (b_zero_mask & (1u << j))
                           return false;
                        b_zero_mask |= (uint8_t)(1u << j);
                     } else {
                        return false;
                     }
                  } else {
                     return false;
                  }
               } else if (cf->type == nir_cf_node_block && nir_def_is_phi(val.def)) {
                  const nir_phi_instr *vphi = nir_def_as_phi(val.def);
                  nir_cf_node *prev = nir_cf_node_prev(&vphi->instr.block->cf_node);
                  if (!prev || prev->type != nir_cf_node_if)
                     return false;
                  nir_if *br = nir_cf_node_as_if(prev);
                  nir_scalar tv = {0}, ev = {0};
                  bool gt = false, ge = false;
                  nir_foreach_phi_src(ps, vphi) {
                     if (ps->pred == nir_if_last_then_block(br)) {
                        tv = source(ps->src);
                        gt = true;
                     } else if (ps->pred == nir_if_last_else_block(br)) {
                        ev = source(ps->src);
                        ge = true;
                     } else {
                        return false;
                     }
                  }
                  const nir_intrinsic_instr *tld = intrinsic(tv.def, nir_intrinsic_load_deref);
                  if (!gt || !ge || !tld || !is_const_u32(ev, 0x00000000u))
                     return false;
                  struct buffer_access acc;
                  if (!storage_read(tld, desc_comps, &acc))
                     return false;
                  if (same_binding(acc.binding, a_binding) &&
                      match_a_ssbo_idx(acc.element, ls, k_off, tile_phi, k_idx_phi, j) &&
                      match_gemm_shared_guard(br, true, ls, m_off, n_off, k_off,
                                              tile_phi, k_idx_phi, j)) {
                     if (s_a && s_a != var)
                        return false;
                     s_a = var;
                     if ((a_load_mask | a_zero_mask) & (1u << j))
                        return false;
                     a_load_mask |= (uint8_t)(1u << j);
                     a_zero_mask |= (uint8_t)(1u << j);
                  } else if (same_binding(acc.binding, b_binding) &&
                             match_b_ssbo_idx(acc.element, ls, n_off, tile_phi, k_idx_phi, j) &&
                             match_gemm_shared_guard(br, false, ls, m_off, n_off, k_off,
                                                     tile_phi, k_idx_phi, j)) {
                     if (s_b && s_b != var)
                        return false;
                     s_b = var;
                     if ((b_load_mask | b_zero_mask) & (1u << j))
                        return false;
                     b_load_mask |= (uint8_t)(1u << j);
                     b_zero_mask |= (uint8_t)(1u << j);
                  } else {
                     return false;
                  }
               } else {
                  return false;
               }
            }
         }
      }
   }

   if (barrier_stage != 2 || !s_a || !s_b || s_a == s_b ||
       a_load_mask != 0xffu || b_load_mask != 0xffu)
      return false;
   if (has_shared_load_outside(impl, &k_loop->cf_node))
      return false;
   if (is_guarded && (a_zero_mask != 0xffu || b_zero_mask != 0xffu))
      return false;

   /* 5. Verify Phase 2 dot-product accumulation for all 8 rows j = 0..7. */
   if (dot_loop) {
      nir_block *d_pre = nir_cf_node_as_block(nir_cf_node_prev(&dot_loop->cf_node));
      nir_block *d_hdr = nir_loop_first_block(dot_loop);
      nir_block *d_back = nir_loop_last_block(dot_loop);

      /* Reject any stores or barriers inside dot_loop. */
      nir_block *d_end = nir_loop_last_block(dot_loop);
      for (nir_block *blk = d_hdr; blk;
           blk = (blk == d_end) ? NULL : nir_block_cf_tree_next(blk)) {
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
               if (op->intrinsic == nir_intrinsic_store_deref ||
                   op->intrinsic == nir_intrinsic_barrier)
                  return false;
            }
         }
      }

      /* Map outer acc_phi[j] back-edge -> inner dot_loop phi in_acc[j]. */
      const nir_phi_instr *in_acc[8] = {0};
      for (unsigned j = 0; j < 8; ++j) {
         nir_scalar k_next = {0};
         nir_foreach_phi_src(src, acc_phi[j]) {
            if (src->pred == k_back)
               k_next = source(src->src);
         }
         if (!k_next.def || !nir_def_is_phi(k_next.def) ||
             nir_def_instr(k_next.def)->block != d_hdr)
            return false;
         in_acc[j] = nir_def_as_phi(k_next.def);
      }

      /* Find inner index l_phi: 0 .. 31, step +1. */
      const nir_phi_instr *l_phi = NULL;
      nir_foreach_phi(phi, d_hdr) {
         bool is_acc = false;
         for (unsigned j = 0; j < 8; ++j) {
            if (in_acc[j] == phi)
               is_acc = true;
         }
         if (is_acc)
            continue;
         nir_scalar pre_v = {0}, back_v = {0};
         bool gp = false, gb = false;
         nir_foreach_phi_src(src, phi) {
            if (src->pred == d_pre) {
               pre_v = source(src->src);
               gp = true;
            } else if (src->pred == d_back) {
               back_v = source(src->src);
               gb = true;
            } else {
               return false;
            }
         }
         if (!gp || !gb || !is_const_u32(pre_v, 0) ||
             !scalar_op(back_v, nir_op_iadd))
            return false;
         nir_scalar a = nir_scalar_chase_alu_src(back_v, 0);
         nir_scalar b = nir_scalar_chase_alu_src(back_v, 1);
         if (!((is_phi_scalar(a, phi) && is_const_u32(b, 1)) ||
               (is_phi_scalar(b, phi) && is_const_u32(a, 1))))
            return false;
         if (l_phi)
            return false;
         l_phi = phi;
      }
      const nir_instr *dot_exit_break = NULL;
      if (!l_phi ||
          !match_loop_iv_less_than(dot_loop, l_phi, true, 32, false, 0, false,
                                   &dot_exit_break) ||
          !dot_exit_break)
         return false;

      /* Require no extra loops anywhere and no break other than the two loop
       * header exits (rejects any early `if (...) break;` in K- or dot-loop). */
      if (count_cf_loops(&impl->body) != 2u)
         return false;
      nir_foreach_block(blk, (nir_function_impl *)impl) {
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_jump &&
                (nir_instr_as_jump(instr)->type != nir_jump_break ||
                 (instr != k_exit_break && instr != dot_exit_break)))
               return false;
         }
      }

      for (unsigned j = 0; j < 8; ++j) {
         nir_scalar pre_v = {0}, back_v = {0};
         bool gp = false, gb = false;
         nir_foreach_phi_src(src, in_acc[j]) {
            if (src->pred == d_pre) {
               pre_v = source(src->src);
               gp = true;
            } else if (src->pred == d_back) {
               back_v = source(src->src);
               gb = true;
            } else {
               return false;
            }
         }
         if (!gp || !gb || !is_phi_scalar(pre_v, acc_phi[j]))
            return false;
         if (!match_dot_step(back_v, nir_get_scalar((nir_def *)&in_acc[j]->def, 0),
                             s_a, s_b, ls, j, l_phi, NULL))
            return false;
      }
      return true;
   }

   /* Require K-loop is the only loop and its header exit is the only break. */
   if (count_cf_loops(&impl->body) != 1u)
      return false;
   nir_foreach_block(blk, (nir_function_impl *)impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_jump &&
             (nir_instr_as_jump(instr)->type != nir_jump_break ||
              instr != k_exit_break))
            return false;
      }
   }

   /* Unrolled 32-step dot-product fallback (no inner loop). */
   for (unsigned j = 0; j < 8; ++j) {
      nir_scalar cur = {0};
      nir_foreach_phi_src(src, acc_phi[j]) {
         if (src->pred == k_back)
            cur = source(src->src);
      }
      uint32_t seen_kk = 0;
      for (unsigned step = 0; step < 32; ++step) {
         cur = nir_scalar_chase_movs(cur);
         nir_scalar prev = {0};
         if (scalar_op(cur, nir_op_ffma)) {
            prev = nir_scalar_chase_alu_src(cur, 2);
         } else if (scalar_op(cur, nir_op_fadd)) {
            nir_scalar s0 = nir_scalar_chase_alu_src(cur, 0);
            nir_scalar s1 = nir_scalar_chase_alu_src(cur, 1);
            prev = scalar_op(s1, nir_op_fmul) ? s0 : s1;
         } else {
            return false;
         }
         uint32_t kk = 0;
         if (!match_dot_step(cur, prev, s_a, s_b, ls, j, NULL, &kk) ||
             (seen_kk & (1u << kk)))
            return false;
         seen_kk |= (1u << kk);
         cur = prev;
      }
      if (seen_kk != 0xffffffffu || !is_phi_scalar(cur, acc_phi[j]))
         return false;
   }
   return true;
}

bool
hk_nir_match_gemm_output(const nir_shader *shader,
                         nir_address_format storage_format,
                         struct hk_nir_gemm_output *match)
{
   const unsigned desc_comps = storage_format_components(storage_format);
   if (!desc_comps ||
       shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable || !shader->info.workgroup_size[0] ||
       !shader->info.workgroup_size[1] || shader->info.workgroup_size[2] != 1 ||
       (shader->info.num_ssbos != 3 && shader->info.num_ssbos != 4))
      return false;

   nir_function_impl *impl = NULL;
   nir_foreach_function_impl(candidate, shader) {
      if (impl || !candidate->function->is_entrypoint)
         return false;
      impl = candidate;
   }
   if (!impl)
      return false;

   const uint16_t *ls = shader->info.workgroup_size;
   const nir_intrinsic_instr *c_stores[48]; /* up to 3 arms * 16 rows */
   struct buffer_access c_access[48];
   unsigned num_c_stores = 0;

   struct hk_nir_clamp_buffer read_bindings[4];
   uint32_t read_res_types[4] = {0};
   uint32_t read_stride_off[4] = {0};
   bool read_has_stride[4] = {0};
   bool read_is_row_idx[4] = {0};
   bool read_is_col_idx[4] = {0};
   uint32_t read_bias_arm_off[4] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
   uint8_t read_bias_arm_mask[4] = {0};
   bool read_has_unguarded_bias[4] = {0};
   bool read_bias_arm_inconsistent[4] = {0};
   unsigned num_read_bindings = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
         case nir_instr_type_load_const:
         case nir_instr_type_deref:
         case nir_instr_type_phi:
            continue;
         case nir_instr_type_jump:
            if (nir_instr_as_jump(instr)->type != nir_jump_break)
               return false;
            continue;
         case nir_instr_type_intrinsic:
            break;
         default:
            return false;
         }

         nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
         switch (op->intrinsic) {
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_push_constant:
            break;
         case nir_intrinsic_barrier:
            if ((nir_intrinsic_execution_scope(op) != SCOPE_NONE &&
                 nir_intrinsic_execution_scope(op) != SCOPE_WORKGROUP) ||
                (nir_intrinsic_memory_modes(op) & ~nir_var_mem_shared))
               return false;
            break;
         case nir_intrinsic_load_deref: {
            if (!nir_def_is_deref(op->src[0].ssa))
               return false;
            const nir_deref_instr *ptr = nir_def_as_deref(op->src[0].ssa);
            if (nir_deref_mode_is(ptr, nir_var_mem_shared)) {
               if (!plain_load(op))
                  return false;
               break;
            }
            if (nir_deref_mode_is(ptr, nir_var_mem_ssbo)) {
               struct buffer_access acc;
               if (!storage_read(op, desc_comps, &acc))
                  return false;
               unsigned slot = num_read_bindings;
               for (unsigned i = 0; i < num_read_bindings; ++i) {
                  if (same_binding(read_bindings[i], acc.binding)) {
                     slot = i;
                     break;
                  }
               }
               if (slot == num_read_bindings) {
                  if (num_read_bindings >= ARRAY_SIZE(read_bindings))
                     return false;
                  read_bindings[num_read_bindings] = acc.binding;
                  read_res_types[num_read_bindings] = nir_intrinsic_resource_type(
                     nir_def_as_intrinsic(acc.descriptor->src[0].ssa));
                  num_read_bindings++;
               }
               uint32_t s_off;
               if (!read_has_stride[slot] &&
                   find_linear_stride_push(acc.element, &s_off)) {
                  read_has_stride[slot] = true;
                  read_stride_off[slot] = s_off;
               }
               uint32_t tmp_r, tmp_j;
               const bool is_col = is_global_id(acc.element, ls, 0);
               const bool is_row = !is_col && match_gemm_row(acc.element, ls, &tmp_r, &tmp_j);
               if (is_col)
                  read_is_col_idx[slot] = true;
               else if (is_row)
                  read_is_row_idx[slot] = true;
               if (is_col || is_row) {
                  uint32_t b_off = 0, b_mode = 0;
                  if (find_bias_mode_arm(&op->instr.block->cf_node, &b_off, &b_mode)) {
                     if ((is_row && b_mode != 0) || (is_col && b_mode != 1)) {
                        read_bias_arm_inconsistent[slot] = true;
                     } else if (read_bias_arm_off[slot] == UINT32_MAX) {
                        read_bias_arm_off[slot] = b_off;
                        read_bias_arm_mask[slot] |= (uint8_t)(1u << b_mode);
                     } else if (read_bias_arm_off[slot] == b_off) {
                        read_bias_arm_mask[slot] |= (uint8_t)(1u << b_mode);
                     } else {
                        read_bias_arm_inconsistent[slot] = true;
                     }
                  } else {
                     read_has_unguarded_bias[slot] = true;
                  }
               }
            } else if (nir_deref_mode_is(ptr, nir_var_mem_push_const)) {
               uint32_t offset;
               if (!push_count(nir_get_scalar((nir_def *)&op->def, 0), &offset))
                  return false;
            } else if (nir_deref_mode_is(ptr, nir_var_system_value)) {
               unsigned axis;
               if (!sysval_id_axis(nir_get_scalar((nir_def *)&op->def, 0),
                                   SYSTEM_VALUE_GLOBAL_INVOCATION_ID, ls, &axis) &&
                   !sysval_id_axis(nir_get_scalar((nir_def *)&op->def, 0),
                                   SYSTEM_VALUE_LOCAL_INVOCATION_ID, ls, &axis))
                  return false;
            } else {
               return false;
            }
            break;
         }
         case nir_intrinsic_store_deref: {
            if (!nir_def_is_deref(op->src[0].ssa))
               return false;
            const nir_deref_instr *ptr = nir_def_as_deref(op->src[0].ssa);
            if (nir_deref_mode_is(ptr, nir_var_mem_shared)) {
               if (nir_intrinsic_access(op) &
                   ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS))
                  return false;
               break;
            }
            if (!nir_deref_mode_is(ptr, nir_var_mem_ssbo))
               return false;
            if (num_c_stores >= ARRAY_SIZE(c_stores) ||
                op->num_components != 1 || op->src[1].ssa->bit_size != 32 ||
                nir_intrinsic_write_mask(op) != 1 ||
                (nir_intrinsic_access(op) &
                 ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS)))
               return false;
            struct buffer_access acc;
            if (!buffer(op->src[0], &acc) ||
                acc.descriptor->def.num_components != desc_comps ||
                acc.descriptor->def.bit_size != 32)
               return false;
            c_stores[num_c_stores] = op;
            c_access[num_c_stores] = acc;
            num_c_stores++;
            break;
         }
         default:
            return false;
         }
      }
   }

   if (num_c_stores == 0 || (num_read_bindings != 2 && num_read_bindings != 3))
      return false;

   const struct hk_nir_clamp_buffer out_binding = c_access[0].binding;
   for (unsigned i = 1; i < num_c_stores; ++i) {
      if (!same_binding(c_access[i].binding, out_binding))
         return false;
   }
   for (unsigned i = 0; i < num_read_bindings; ++i) {
      if (same_binding(read_bindings[i], out_binding))
         return false;
   }

   uint32_t rows_per_thread = 0;
   uint32_t n_off = UINT32_MAX, m_off = UINT32_MAX, bias_mode_off = UINT32_MAX;
   bool have_guarded = false, is_guarded = false;
   bool have_bias_mode = false, has_bias_mode = false;
   uint8_t bias_mode_mask = 0;
   uint32_t row_mask[3] = {0};
   uint32_t flat_row_mask = 0;
   bool have_bias_binding = false;
   struct hk_nir_clamp_buffer bias_binding = {0};

   for (unsigned i = 0; i < num_c_stores; ++i) {
      uint32_t r, j, store_n_off;
      if (!match_gemm_c_index(c_access[i].element, ls, &r, &j, &store_n_off))
         return false;
      if (rows_per_thread == 0)
         rows_per_thread = r;
      else if (rows_per_thread != r)
         return false;

      if (n_off == UINT32_MAX)
         n_off = store_n_off;
      else if (n_off != store_n_off)
         return false;

      uint32_t arm_off = 0, mode = 0;
      const bool in_arm = find_bias_mode_arm(&c_stores[i]->instr.block->cf_node,
                                             &arm_off, &mode);
      if (!have_bias_mode) {
         have_bias_mode = true;
         has_bias_mode = in_arm;
      } else if (has_bias_mode != in_arm) {
         return false;
      }

      if (has_bias_mode) {
         if (bias_mode_off == UINT32_MAX)
            bias_mode_off = arm_off;
         else if (bias_mode_off != arm_off)
            return false;
         if (row_mask[mode] & (1u << j))
            return false;
         row_mask[mode] |= (1u << j);
         bias_mode_mask |= (uint8_t)(1u << mode);
      } else {
         if (flat_row_mask & (1u << j))
            return false;
         flat_row_mask |= (1u << j);
      }

      /* Check optional per-element bounds guard `row < m && col < n`. */
      nir_cf_node *parent = c_stores[i]->instr.block->cf_node.parent;
      uint32_t store_m_off = UINT32_MAX;
      bool guarded = false;
      if (parent->type == nir_cf_node_if &&
          in_then(&c_stores[i]->instr.block->cf_node, nir_cf_node_as_if(parent)) &&
          match_gemm_store_guard(nir_cf_node_as_if(parent), ls,
                                 rows_per_thread, j, n_off, &store_m_off)) {
         guarded = true;
      }
      if (!have_guarded) {
         have_guarded = true;
         is_guarded = guarded;
         m_off = store_m_off;
      } else if (is_guarded != guarded || (guarded && m_off != store_m_off)) {
         return false;
      }

      /* Identify bias read binding from bias-adding store values. */
      nir_scalar sv = source(c_stores[i]->src[1]);
      if (scalar_op(sv, nir_op_fadd)) {
         for (unsigned s = 0; s < 2; ++s) {
            nir_scalar opnd = nir_scalar_chase_alu_src(sv, s);
            const nir_intrinsic_instr *ld =
               intrinsic(opnd.def, nir_intrinsic_load_deref);
            struct buffer_access bacc;
            if (ld && storage_read(ld, desc_comps, &bacc)) {
               if (!have_bias_binding) {
                  bias_binding = bacc.binding;
                  have_bias_binding = true;
               } else if (!same_binding(bias_binding, bacc.binding)) {
                  return false;
               }
            }
         }
      }
   }

   if (!rows_per_thread || rows_per_thread > 16)
      return false;
   const uint32_t full_rows = (1u << rows_per_thread) - 1u;

   if (has_bias_mode) {
      if (!bias_mode_mask)
         return false;
      for (unsigned m = 0; m < 3; ++m) {
         if ((bias_mode_mask & (1u << m)) && row_mask[m] != full_rows)
            return false;
      }
   } else {
      if (flat_row_mask != full_rows)
         return false;
   }

   /* Identify A (stride k), B (stride n), and optional bias binding. */
   int a_slot = -1, b_slot = -1, bias_slot = -1;
   for (unsigned i = 0; i < num_read_bindings; ++i) {
      if (have_bias_binding && same_binding(read_bindings[i], bias_binding)) {
         bias_slot = (int)i;
      } else if (read_has_stride[i] && read_stride_off[i] == n_off) {
         if (b_slot != -1)
            return false;
         b_slot = (int)i;
      } else if (read_has_stride[i]) {
         if (a_slot != -1)
            return false;
         a_slot = (int)i;
      } else if (bias_slot == -1 && num_read_bindings == 3) {
         bias_slot = (int)i;
      }
   }

   if (a_slot < 0 || b_slot < 0 || !read_has_stride[a_slot])
      return false;
   const bool has_bias_buffer = (bias_slot >= 0);
   if (num_read_bindings == 3 && !has_bias_buffer)
      return false;

   const bool stores_split_by_bias_mode = has_bias_mode;

   /* Enforce bias_mode load consistency unconditionally on both split-store
    * and non-split paths. */
   if (has_bias_buffer) {
      if (read_bias_arm_inconsistent[bias_slot] ||
          (read_has_unguarded_bias[bias_slot] && read_bias_arm_mask[bias_slot]))
         return false;
      if (stores_split_by_bias_mode) {
         if (read_has_unguarded_bias[bias_slot])
            return false;
         if (read_bias_arm_mask[bias_slot] &&
             (read_bias_arm_off[bias_slot] != bias_mode_off ||
              (read_bias_arm_mask[bias_slot] & ~bias_mode_mask) != 0))
            return false;
      }
   }

   /* If there is a bias buffer and stores themselves were not split per
    * bias_mode arm, either the bias loads are guarded by bias_mode arms
    * (with a fallthrough no-bias mode 2) or the bias load is unconditional
    * (indexed exclusively by row -> mode 0, or col -> mode 1). */
   if (!has_bias_mode && has_bias_buffer) {
      if (read_bias_arm_mask[bias_slot]) {
         has_bias_mode = true;
         bias_mode_off = read_bias_arm_off[bias_slot];
         bias_mode_mask = (uint8_t)(read_bias_arm_mask[bias_slot] | (1u << 2));
      } else if (read_is_row_idx[bias_slot] && !read_is_col_idx[bias_slot]) {
         bias_mode_mask = (1u << 0);
      } else if (read_is_col_idx[bias_slot] && !read_is_row_idx[bias_slot]) {
         bias_mode_mask = (1u << 1);
      } else {
         return false;
      }
   }
   /* If bias_mode arms 0 or 1 exist, a bias buffer must exist. */
   if (has_bias_mode && (bias_mode_mask & 0x3u) && !has_bias_buffer)
      return false;

   const uint32_t k_off = read_stride_off[a_slot];
   uint32_t max_push = MAX2(n_off, k_off) + 4;
   if (is_guarded)
      max_push = MAX2(max_push, m_off + 4);
   if (has_bias_mode)
      max_push = MAX2(max_push, bias_mode_off + 4);

   const uint32_t out_res_type = nir_intrinsic_resource_type(
      nir_def_as_intrinsic(c_access[0].descriptor->src[0].ssa));
   const struct hk_nir_clamp_buffer bias_b =
      has_bias_buffer ? read_bindings[bias_slot] : (struct hk_nir_clamp_buffer){0};

   const bool cmat_ok =
      !shader->info.float_controls_execution_mode &&
      ls[0] == 32 && ls[1] == 4 && ls[2] == 1 &&
      rows_per_thread == 8 &&
      desc_comps >= 3 && desc_comps <= 5 &&
      max_push <= 64 &&
      (n_off & 3) == 0 && (k_off & 3) == 0 &&
      (!is_guarded || (m_off & 3) == 0) &&
      (!has_bias_mode || (bias_mode_off & 3) == 0) &&
      ((out_binding.offset_B | read_bindings[a_slot].offset_B |
        read_bindings[b_slot].offset_B | bias_b.offset_B) & 3) == 0 &&
      (is_guarded || (read_bindings[a_slot].offset_B & 15) == 0) &&
      certify_gemm_row_major_k_loop(impl, ls, c_stores, c_access, num_c_stores,
                                    read_bindings[a_slot], read_bindings[b_slot],
                                    has_bias_buffer, bias_b,
                                    stores_split_by_bias_mode, has_bias_mode,
                                    bias_mode_off, bias_mode_mask, is_guarded,
                                    m_off, n_off, k_off, desc_comps);

   *match = (struct hk_nir_gemm_output){
      .output = out_binding,
      .reads = {
         read_bindings[a_slot],
         read_bindings[b_slot],
         bias_b,
      },
      .m_offset_B = is_guarded ? m_off : UINT32_MAX,
      .n_offset_B = n_off,
      .k_offset_B = k_off,
      .bias_mode_offset_B = has_bias_mode ? bias_mode_off : UINT32_MAX,
      .push_bytes = max_push,
      .out_res_type = out_res_type,
      .read_res_type = {
         read_res_types[a_slot],
         read_res_types[b_slot],
         has_bias_buffer ? read_res_types[bias_slot] : 0,
      },
      .desc_comps = desc_comps,
      .local_size = {
         shader->info.workgroup_size[0],
         shader->info.workgroup_size[1],
         shader->info.workgroup_size[2],
      },
      .rows_per_thread = rows_per_thread,
      .float_controls_execution_mode = shader->info.float_controls_execution_mode,
      .bias_mode_mask = bias_mode_mask,
      .has_bias_mode = has_bias_mode,
      .has_bias_buffer = has_bias_buffer,
      .is_guarded = is_guarded,
      .cmat_valid = cmat_ok,
   };
   return true;
}

static bool
read_positive_push(const struct hk_nir_direct_shape *launch,
                   uint32_t offset_B, uint32_t *out)
{
   if (!launch->push || offset_B > launch->push_size_B ||
       launch->push_size_B - offset_B < 4)
      return false;
   int32_t value;
   memcpy(&value, (const uint8_t *)launch->push + offset_B, sizeof(value));
   if (value <= 0)
      return false;
   *out = (uint32_t)value;
   return true;
}

static bool
read_u32_push(const struct hk_nir_direct_shape *launch,
              uint32_t offset_B, uint32_t *out)
{
   if (!launch->push || offset_B > launch->push_size_B ||
       launch->push_size_B - offset_B < 4)
      return false;
   memcpy(out, (const uint8_t *)launch->push + offset_B, sizeof(*out));
   return true;
}

static bool
dispatch_covers(const struct hk_nir_direct_shape *launch,
                const uint32_t local_size[3], const uint64_t active[3])
{
   if (!hk_nir_dispatch_zero_base(launch->base_group))
      return false;
   for (unsigned i = 0; i < 3; ++i) {
      if (!local_size[i])
         return false;
      uint64_t threads = (uint64_t)launch->groups[i] * local_size[i];
      if (threads < active[i] || threads > INT32_MAX)
         return false;
   }
   return true;
}

static bool
footprint_end(uint32_t offset, uint32_t width, uint32_t height,
              uint32_t channels, uint64_t *end)
{
   uint64_t row = (uint64_t)width * channels;
   if (row > INT32_MAX || row * height > INT32_MAX)
      return false;
   *end = offset + 4 * row * height;
   /* SSBO 32-bit byte offset limit. */
   return *end <= UINT32_MAX;
}

bool
hk_nir_depthwise_producer_footprint(const struct hk_nir_depthwise_output *producer,
                                    const struct hk_nir_direct_shape *producer_launch,
                                    uint64_t producer_ends[4], uint64_t *out_elems_out)
{
   uint32_t iw, ih, ow, oh, kw, kh, channels;
   if (!read_positive_push(producer_launch, producer->input_width_offset_B, &iw) ||
       !read_positive_push(producer_launch, producer->input_height_offset_B, &ih) ||
       !read_positive_push(producer_launch, producer->width_offset_B, &ow) ||
       !read_positive_push(producer_launch, producer->height_offset_B, &oh) ||
       !read_positive_push(producer_launch, producer->kernel_width_offset_B, &kw) ||
       !read_positive_push(producer_launch, producer->kernel_height_offset_B, &kh) ||
       !read_positive_push(producer_launch, producer->channels_offset_B, &channels))
      return false;

   uint64_t p[4] = {0};
   if (!footprint_end(producer->reads[0].offset_B, iw, ih, channels, &p[0]) ||
       !footprint_end(producer->reads[1].offset_B, kw, kh, channels, &p[1]) ||
       !footprint_end(producer->output.offset_B, ow, oh, channels, &p[3]))
      return false;

   if (producer->has_bias_buffer) {
      if (!footprint_end(producer->reads[2].offset_B, 1, 1, channels, &p[2]))
         return false;
   } else {
      p[2] = 0;
   }

   uint64_t x = (uint64_t)ow * channels;
   if (!dispatch_covers(producer_launch, producer->local_size,
                        (uint64_t[]){x, oh, 1}))
      return false;

   memcpy(producer_ends, p, sizeof(p));
   if (out_elems_out)
      *out_elems_out = x * oh;
   return true;
}

bool
hk_nir_depthwise_clamp_footprint(const struct hk_nir_depthwise_output *producer,
                                 const struct hk_nir_pointwise_clamp *consumer,
                                 const struct hk_nir_direct_shape *producer_launch,
                                 const struct hk_nir_direct_shape *consumer_launch,
                                 uint64_t producer_ends[4], uint64_t consumer_ends[2])
{
   uint32_t count;
   uint64_t out_elems = 0;
   if (!hk_nir_depthwise_producer_footprint(producer, producer_launch,
                                            producer_ends, &out_elems) ||
       !read_positive_push(consumer_launch, consumer->count_offset_B, &count))
      return false;

   uint64_t c[2] = {0};
   if (!footprint_end(consumer->input.offset_B, count, 1, 1, &c[0]) ||
       !footprint_end(consumer->output.offset_B, count, 1, 1, &c[1]))
      return false;

   const uint32_t consumer_local[] = {consumer->local_size_x, 1, 1};
   if (count != out_elems || consumer_launch->groups[1] != 1 ||
       !dispatch_covers(consumer_launch, consumer_local,
                        (uint64_t[]){count, 1, 1}))
      return false;

   memcpy(consumer_ends, c, sizeof(c));
   return true;
}

bool
hk_nir_gemm_producer_footprint(const struct hk_nir_gemm_output *producer,
                               const struct hk_nir_direct_shape *producer_launch,
                               uint64_t producer_ends[4],
                               uint32_t *m_out, uint32_t *n_out)
{
   if (!producer->rows_per_thread || producer->rows_per_thread > 16)
      return false;

   uint32_t m, n, k;
   if (!read_positive_push(producer_launch, producer->n_offset_B, &n) ||
       !read_positive_push(producer_launch, producer->k_offset_B, &k))
      return false;

   uint32_t active_bias_mode = 2; /* 2 = no bias read */
   if (producer->has_bias_mode) {
      if (!read_u32_push(producer_launch, producer->bias_mode_offset_B, &active_bias_mode) ||
          active_bias_mode > 2 || !(producer->bias_mode_mask & (1u << active_bias_mode)))
         return false;
   } else if (producer->has_bias_buffer) {
      if (producer->bias_mode_mask == (1u << 0))
         active_bias_mode = 0;
      else if (producer->bias_mode_mask == (1u << 1))
         active_bias_mode = 1;
      else
         return false;
   }

   if (producer->is_guarded) {
      if (!read_positive_push(producer_launch, producer->m_offset_B, &m))
         return false;
   } else {
      uint64_t tx = (uint64_t)producer_launch->groups[0] * producer->local_size[0];
      uint64_t ty = (uint64_t)producer_launch->groups[1] * producer->local_size[1];
      uint64_t rows = ty * producer->rows_per_thread;
      if (tx != n || rows == 0 || rows > INT32_MAX)
         return false;
      m = (uint32_t)rows;
   }

   uint64_t p[4] = {0};
   if (!footprint_end(producer->reads[0].offset_B, m, k, 1, &p[0]) ||
       !footprint_end(producer->reads[1].offset_B, k, n, 1, &p[1]) ||
       !footprint_end(producer->output.offset_B, m, n, 1, &p[3]))
      return false;

   if (active_bias_mode == 2 || !producer->has_bias_buffer) {
      p[2] = 0;
   } else {
      const uint32_t bias_len = active_bias_mode == 0 ? m : n;
      if (!footprint_end(producer->reads[2].offset_B, bias_len, 1, 1, &p[2]))
         return false;
   }

   const uint64_t active_y = DIV_ROUND_UP((uint64_t)m, producer->rows_per_thread);
   if (!dispatch_covers(producer_launch, producer->local_size,
                        (uint64_t[]){n, active_y, 1}))
      return false;

   memcpy(producer_ends, p, sizeof(p));
   if (m_out)
      *m_out = m;
   if (n_out)
      *n_out = n;
   return true;
}

bool
hk_nir_gemm_clamp_footprint(const struct hk_nir_gemm_output *producer,
                            const struct hk_nir_pointwise_clamp *consumer,
                            const struct hk_nir_direct_shape *producer_launch,
                            const struct hk_nir_direct_shape *consumer_launch,
                            uint64_t producer_ends[4], uint64_t consumer_ends[2])
{
   uint32_t m = 0, n = 0, count;
   if (!hk_nir_gemm_producer_footprint(producer, producer_launch,
                                       producer_ends, &m, &n) ||
       !read_positive_push(consumer_launch, consumer->count_offset_B, &count))
      return false;

   uint64_t c[2] = {0};
   if (!footprint_end(consumer->input.offset_B, count, 1, 1, &c[0]) ||
       !footprint_end(consumer->output.offset_B, count, 1, 1, &c[1]))
      return false;

   const uint32_t consumer_local[] = {consumer->local_size_x, 1, 1};
   if ((uint64_t)count != (uint64_t)m * n || consumer_launch->groups[1] != 1 ||
       !dispatch_covers(consumer_launch, consumer_local,
                        (uint64_t[]){count, 1, 1}))
      return false;

   memcpy(consumer_ends, c, sizeof(c));
   return true;
}

static nir_def *
apply_clamp_epilogue(nir_builder *b, nir_def *val,
                     const struct hk_nir_clamp_epilogue *ep)
{
   unsigned saved_ctrl = b->fp_math_ctrl;
   if (ep->has_lower && ep->has_upper) {
      if (ep->upper_first) {
         b->fp_math_ctrl = ep->min_fp_math_ctrl;
         val = nir_fmin(b, val, nir_imm_float(b, uif(ep->upper_bits)));
         b->fp_math_ctrl = ep->max_fp_math_ctrl;
         val = nir_fmax(b, val, nir_imm_float(b, uif(ep->lower_bits)));
      } else {
         b->fp_math_ctrl = ep->max_fp_math_ctrl;
         val = nir_fmax(b, val, nir_imm_float(b, uif(ep->lower_bits)));
         b->fp_math_ctrl = ep->min_fp_math_ctrl;
         val = nir_fmin(b, val, nir_imm_float(b, uif(ep->upper_bits)));
      }
   } else if (ep->has_lower) {
      b->fp_math_ctrl = ep->max_fp_math_ctrl;
      val = nir_fmax(b, val, nir_imm_float(b, uif(ep->lower_bits)));
   } else if (ep->has_upper) {
      b->fp_math_ctrl = ep->min_fp_math_ctrl;
      val = nir_fmin(b, val, nir_imm_float(b, uif(ep->upper_bits)));
   }
   b->fp_math_ctrl = saved_ctrl;
   return val;
}

nir_shader *
hk_nir_splice_epilogue(const nir_shader *producer,
                       struct hk_nir_clamp_buffer output,
                       const struct hk_nir_clamp_epilogue *epilogue)
{
   if (!producer || producer->info.stage != MESA_SHADER_COMPUTE ||
       !hk_nir_clamp_epilogue_valid(epilogue) ||
       producer->info.float_controls_execution_mode !=
          epilogue->float_controls_execution_mode)
      return NULL;

   nir_shader *copy = nir_shader_clone(NULL, producer);
   if (!copy)
      return NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(copy);
   if (!impl) {
      ralloc_free(copy);
      return NULL;
   }
   nir_foreach_function_impl(other, copy) {
      if (other != impl) {
         ralloc_free(copy);
         return NULL;
      }
   }

   nir_intrinsic_instr *stores[64];
   unsigned num_stores = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call) {
            ralloc_free(copy);
            return NULL;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
         if (op->intrinsic == nir_intrinsic_barrier) {
            if ((nir_intrinsic_execution_scope(op) != SCOPE_NONE &&
                 nir_intrinsic_execution_scope(op) != SCOPE_WORKGROUP) ||
                (nir_intrinsic_memory_modes(op) & ~nir_var_mem_shared)) {
               ralloc_free(copy);
               return NULL;
            }
            continue;
         }
         if (op->intrinsic != nir_intrinsic_store_deref) {
            if (!nir_intrinsic_has_semantic(op, NIR_INTRINSIC_CAN_ELIMINATE)) {
               ralloc_free(copy);
               return NULL;
            }
            continue;
         }
         if (!nir_def_is_deref(op->src[0].ssa)) {
            ralloc_free(copy);
            return NULL;
         }
         const nir_deref_instr *ptr = nir_def_as_deref(op->src[0].ssa);
         if (nir_deref_mode_is(ptr, nir_var_mem_shared))
            continue;
         struct buffer_access acc;
         if (!nir_deref_mode_is(ptr, nir_var_mem_ssbo) ||
             num_stores >= ARRAY_SIZE(stores) ||
             op->num_components != 1 || op->src[1].ssa->bit_size != 32 ||
             nir_intrinsic_write_mask(op) != 1 ||
             (nir_intrinsic_access(op) &
              ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS)) ||
             !buffer(op->src[0], &acc) ||
             !same_binding(acc.binding, output)) {
            ralloc_free(copy);
            return NULL;
         }
         stores[num_stores++] = op;
      }
   }

   if (num_stores == 0) {
      ralloc_free(copy);
      return NULL;
   }

   for (unsigned i = 0; i < num_stores; ++i) {
      nir_builder b = nir_builder_at(nir_before_instr(&stores[i]->instr));
      nir_def *clamped = apply_clamp_epilogue(&b, stores[i]->src[1].ssa, epilogue);
      nir_src_rewrite(&stores[i]->src[1], clamped);
   }
   nir_progress(true, impl, nir_metadata_control_flow);
   return copy;
}

static nir_def *
dw_vec4_addr(nir_builder *b, struct hk_nir_clamp_buffer buf, uint32_t desc_comps,
             uint32_t res_type, nir_def *idx4, enum gl_access_qualifier access)
{
   if (buf.offset_B)
      idx4 = nir_iadd_imm(b, idx4, buf.offset_B / 16);
   nir_def *resource = nir_vulkan_resource_index(
      b, desc_comps, 32, nir_imm_int(b, 0),
      .desc_type = nir_descriptor_type_storage_buffer, .resource_type = res_type,
      .desc_set = buf.set, .binding = buf.binding);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      b, desc_comps, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   const glsl_type *vec = glsl_vector_type(GLSL_TYPE_FLOAT, 4);
   const glsl_type *array_type = glsl_array_type(vec, 0, 16);
   nir_deref_instr *array = nir_build_deref_cast_with_alignment(
      b, descriptor, nir_var_mem_ssbo, array_type, 0, 16, 0);
   nir_deref_instr *element = nir_build_deref_array(b, array, idx4);
   element->arr.in_bounds = true;
   return nir_deref_buffer_address(b, 1, 64, &element->def, .access = access);
}

static nir_def *
dw_push(nir_builder *b, uint32_t offset_B, uint32_t push_bytes)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset_B),
                                 .base = 0, .range = push_bytes, .align_mul = 4);
}

static nir_def *
dw_fma4(nir_builder *b, nir_def *acc, nir_def *in_v, nir_def *wt_v,
        const struct hk_nir_depthwise_output *info)
{
   b->fp_math_ctrl = info->fmul_fp_ctrl;
   nir_def *prod = nir_fmul(b, in_v, wt_v);
   b->fp_math_ctrl = info->fadd_fp_ctrl;
   /* Put prod first so algebraic FMA fusion contracts prod (not an unrolled
    * tap-0 acc that folded fadd(0, r0*w0) -> r0*w0), matching rolled loop order. */
   nir_def *sum = nir_fadd(b, prod, acc);
   b->fp_math_ctrl = 0;
   return sum;
}

static nir_def *
dw_load4(nir_builder *b, const struct hk_nir_depthwise_output *info,
         unsigned read_idx, nir_def *idx4)
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   return nir_load_global(
      b, 4, 32,
      dw_vec4_addr(b, info->reads[read_idx], info->desc_comps,
                   info->read_res_type[read_idx], idx4, read),
      .align_mul = 16, .access = read);
}

static void
emit_dw_3x3_s1_1x2(nir_builder *b, const struct hk_nir_depthwise_output *info,
                   nir_def *in_base0, nir_def *ch4, nir_def *c4,
                   nir_def *row_stride4, nir_def *zero_v4,
                   nir_def **a0_out, nir_def **a1_out)
{
   nir_def *zero_i = nir_imm_int(b, 0);
   nir_def *c4_2 = nir_ishl_imm(b, c4, 1);
   nir_def *c4_3 = nir_iadd(b, c4_2, c4);

   nir_loop *ky_lp = nir_push_loop(b);
   ky_lp->control = nir_loop_control_dont_unroll;
   nir_block *ky_pre = nir_cf_node_as_block(nir_cf_node_prev(&ky_lp->cf_node));
   nir_block *ky_hdr = nir_loop_first_block(ky_lp);

   nir_phi_instr *ky_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
   nir_phi_instr_add_src(ky_phi, ky_pre, zero_i);

   nir_phi_instr *r0_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r0_phi->instr, &r0_phi->def, 4, 32);
   nir_phi_instr_add_src(r0_phi, ky_pre, zero_v4);

   nir_phi_instr *r1_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r1_phi->instr, &r1_phi->def, 4, 32);
   nir_phi_instr_add_src(r1_phi, ky_pre, zero_v4);

   nir_phi_instr *in_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&in_phi->instr, &in_phi->def, 1, 32);
   nir_phi_instr_add_src(in_phi, ky_pre, in_base0);

   nir_phi_instr *wt_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&wt_phi->instr, &wt_phi->def, 1, 32);
   nir_phi_instr_add_src(wt_phi, ky_pre, ch4);

   nir_if *ky_done = nir_push_if(b, nir_ige_imm(b, &ky_phi->def, 3));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, ky_done);

   nir_def *in_row = &in_phi->def;
   nir_def *wt_row = &wt_phi->def;
   nir_def *a0 = &r0_phi->def;
   nir_def *a1 = &r1_phi->def;

   /* Stream 1 horizontal tap at a time so peak live vec4s stay low (1024 tpc). */
   nir_def *r0 = dw_load4(b, info, 0, in_row);
   nir_def *r1 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4));
   nir_def *w0 = dw_load4(b, info, 1, wt_row);
   a0 = dw_fma4(b, a0, r0, w0, info);
   a1 = dw_fma4(b, a1, r1, w0, info);

   nir_def *r2 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4_2));
   nir_def *w1 = dw_load4(b, info, 1, nir_iadd(b, wt_row, c4));
   a0 = dw_fma4(b, a0, r1, w1, info);
   a1 = dw_fma4(b, a1, r2, w1, info);

   nir_def *r3 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4_3));
   nir_def *w2 = dw_load4(b, info, 1, nir_iadd(b, wt_row, c4_2));
   a0 = dw_fma4(b, a0, r2, w2, info);
   a1 = dw_fma4(b, a1, r3, w2, info);

   nir_block *ky_back = nir_loop_last_block(ky_lp);
   nir_phi_instr_add_src(ky_phi, ky_back, nir_iadd_imm(b, &ky_phi->def, 1));
   nir_phi_instr_add_src(r0_phi, ky_back, a0);
   nir_phi_instr_add_src(r1_phi, ky_back, a1);
   nir_phi_instr_add_src(in_phi, ky_back, nir_iadd(b, in_row, row_stride4));
   nir_phi_instr_add_src(wt_phi, ky_back, nir_iadd(b, wt_row, c4_3));
   nir_instr_insert(nir_before_block(ky_hdr), &ky_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &r0_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &r1_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &in_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &wt_phi->instr);
   nir_pop_loop(b, ky_lp);

   *a0_out = &r0_phi->def;
   *a1_out = &r1_phi->def;
}

static void
emit_dw_3x3_s2_1x2(nir_builder *b, const struct hk_nir_depthwise_output *info,
                   nir_def *in_base0, nir_def *ch4, nir_def *c4,
                   nir_def *row_stride4, nir_def *zero_v4,
                   nir_def **a0_out, nir_def **a1_out)
{
   nir_def *zero_i = nir_imm_int(b, 0);
   nir_def *c4_2 = nir_ishl_imm(b, c4, 1);
   nir_def *c4_3 = nir_iadd(b, c4_2, c4);
   nir_def *c4_4 = nir_ishl_imm(b, c4, 2);

   nir_loop *ky_lp = nir_push_loop(b);
   ky_lp->control = nir_loop_control_dont_unroll;
   nir_block *ky_pre = nir_cf_node_as_block(nir_cf_node_prev(&ky_lp->cf_node));
   nir_block *ky_hdr = nir_loop_first_block(ky_lp);

   nir_phi_instr *ky_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
   nir_phi_instr_add_src(ky_phi, ky_pre, zero_i);

   nir_phi_instr *r0_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r0_phi->instr, &r0_phi->def, 4, 32);
   nir_phi_instr_add_src(r0_phi, ky_pre, zero_v4);

   nir_phi_instr *r1_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r1_phi->instr, &r1_phi->def, 4, 32);
   nir_phi_instr_add_src(r1_phi, ky_pre, zero_v4);

   nir_phi_instr *in_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&in_phi->instr, &in_phi->def, 1, 32);
   nir_phi_instr_add_src(in_phi, ky_pre, in_base0);

   nir_phi_instr *wt_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&wt_phi->instr, &wt_phi->def, 1, 32);
   nir_phi_instr_add_src(wt_phi, ky_pre, ch4);

   nir_if *ky_done = nir_push_if(b, nir_ige_imm(b, &ky_phi->def, 3));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, ky_done);

   nir_def *in_row = &in_phi->def;
   nir_def *wt_row = &wt_phi->def;
   nir_def *a0 = &r0_phi->def;
   nir_def *a1 = &r1_phi->def;

   /* Stream 1 horizontal tap at a time (reusing r2 across tap 0 and tap 2). */
   nir_def *r0 = dw_load4(b, info, 0, in_row);
   nir_def *r2 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4_2));
   nir_def *w0 = dw_load4(b, info, 1, wt_row);
   a0 = dw_fma4(b, a0, r0, w0, info);
   a1 = dw_fma4(b, a1, r2, w0, info);

   nir_def *r1 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4));
   nir_def *r3 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4_3));
   nir_def *w1 = dw_load4(b, info, 1, nir_iadd(b, wt_row, c4));
   a0 = dw_fma4(b, a0, r1, w1, info);
   a1 = dw_fma4(b, a1, r3, w1, info);

   nir_def *r4 = dw_load4(b, info, 0, nir_iadd(b, in_row, c4_4));
   nir_def *w2 = dw_load4(b, info, 1, nir_iadd(b, wt_row, c4_2));
   a0 = dw_fma4(b, a0, r2, w2, info);
   a1 = dw_fma4(b, a1, r4, w2, info);

   nir_block *ky_back = nir_loop_last_block(ky_lp);
   nir_phi_instr_add_src(ky_phi, ky_back, nir_iadd_imm(b, &ky_phi->def, 1));
   nir_phi_instr_add_src(r0_phi, ky_back, a0);
   nir_phi_instr_add_src(r1_phi, ky_back, a1);
   nir_phi_instr_add_src(in_phi, ky_back, nir_iadd(b, in_row, row_stride4));
   nir_phi_instr_add_src(wt_phi, ky_back, nir_iadd(b, wt_row, c4_3));
   nir_instr_insert(nir_before_block(ky_hdr), &ky_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &r0_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &r1_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &in_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &wt_phi->instr);
   nir_pop_loop(b, ky_lp);

   *a0_out = &r0_phi->def;
   *a1_out = &r1_phi->def;
}

static void
emit_dw_kx_streamed(nir_builder *b, const struct hk_nir_depthwise_output *info,
                    nir_def *kw, nir_def *kh, unsigned stride_x,
                    nir_def *in_base0, nir_def *ch4, nir_def *c4,
                    nir_def *dy_step4, nir_def *wt_row4,
                    nir_def *zero_v4, nir_def **a0_out, nir_def **a1_out)
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   nir_def *zero_i = nir_imm_int(b, 0);
   nir_def *sx_off = (stride_x == 2) ? nir_ishl_imm(b, c4, 1) : NULL;

   nir_loop *ky_lp = nir_push_loop(b);
   ky_lp->control = nir_loop_control_dont_unroll;
   nir_block *ky_pre = nir_cf_node_as_block(nir_cf_node_prev(&ky_lp->cf_node));
   nir_block *ky_hdr = nir_loop_first_block(ky_lp);

   nir_phi_instr *ky_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
   nir_phi_instr_add_src(ky_phi, ky_pre, zero_i);

   nir_phi_instr *r0_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r0_phi->instr, &r0_phi->def, 4, 32);
   nir_phi_instr_add_src(r0_phi, ky_pre, zero_v4);

   nir_phi_instr *r1_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r1_phi->instr, &r1_phi->def, 4, 32);
   nir_phi_instr_add_src(r1_phi, ky_pre, zero_v4);

   nir_phi_instr *in_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&in_phi->instr, &in_phi->def, 1, 32);
   nir_phi_instr_add_src(in_phi, ky_pre, in_base0);

   nir_phi_instr *wt_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&wt_phi->instr, &wt_phi->def, 1, 32);
   nir_phi_instr_add_src(wt_phi, ky_pre, ch4);

   nir_if *ky_done = nir_push_if(b, nir_ige(b, &ky_phi->def, kh));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, ky_done);

   nir_def *iv_init = (stride_x == 1)
      ? nir_load_global(b, 4, 32,
           dw_vec4_addr(b, info->reads[0], info->desc_comps, info->read_res_type[0],
                        &in_phi->def, read),
           .align_mul = 16, .access = read)
      : zero_v4;

   nir_loop *kx_lp = nir_push_loop(b);
   kx_lp->control = nir_loop_control_dont_unroll;
   nir_block *kx_pre = nir_cf_node_as_block(nir_cf_node_prev(&kx_lp->cf_node));
   nir_block *kx_hdr = nir_loop_first_block(kx_lp);

   nir_phi_instr *kx_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&kx_phi->instr, &kx_phi->def, 1, 32);
   nir_phi_instr_add_src(kx_phi, kx_pre, zero_i);

   nir_phi_instr *kr0_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&kr0_phi->instr, &kr0_phi->def, 4, 32);
   nir_phi_instr_add_src(kr0_phi, kx_pre, &r0_phi->def);

   nir_phi_instr *kr1_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&kr1_phi->instr, &kr1_phi->def, 4, 32);
   nir_phi_instr_add_src(kr1_phi, kx_pre, &r1_phi->def);

   nir_phi_instr *kin_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&kin_phi->instr, &kin_phi->def, 1, 32);
   nir_phi_instr_add_src(kin_phi, kx_pre, &in_phi->def);

   nir_phi_instr *kwt_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&kwt_phi->instr, &kwt_phi->def, 1, 32);
   nir_phi_instr_add_src(kwt_phi, kx_pre, &wt_phi->def);

   nir_phi_instr *kiv_phi = NULL;
   if (stride_x == 1) {
      kiv_phi = nir_phi_instr_create(b->shader);
      nir_def_init(&kiv_phi->instr, &kiv_phi->def, 4, 32);
      nir_phi_instr_add_src(kiv_phi, kx_pre, iv_init);
   }

   nir_if *kx_done = nir_push_if(b, nir_ige(b, &kx_phi->def, kw));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, kx_done);

   nir_def *wv = nir_load_global(b, 4, 32,
      dw_vec4_addr(b, info->reads[1], info->desc_comps, info->read_res_type[1],
                   &kwt_phi->def, read),
      .align_mul = 16, .access = read);
   nir_def *iv0 = (stride_x == 1)
      ? &kiv_phi->def
      : nir_load_global(b, 4, 32,
           dw_vec4_addr(b, info->reads[0], info->desc_comps, info->read_res_type[0],
                        &kin_phi->def, read),
           .align_mul = 16, .access = read);
   nir_def *nr0 = dw_fma4(b, &kr0_phi->def, iv0, wv, info);
   nir_def *kin_nxt = nir_iadd(b, &kin_phi->def, c4);
   nir_def *p1_in = (stride_x == 1) ? kin_nxt : nir_iadd(b, &kin_phi->def, sx_off);
   nir_def *iv1 = nir_load_global(b, 4, 32,
      dw_vec4_addr(b, info->reads[0], info->desc_comps, info->read_res_type[0],
                   p1_in, read),
      .align_mul = 16, .access = read);
   nir_def *nr1 = dw_fma4(b, &kr1_phi->def, iv1, wv, info);

   nir_block *kx_back = nir_loop_last_block(kx_lp);
   nir_phi_instr_add_src(kx_phi, kx_back, nir_iadd_imm(b, &kx_phi->def, 1));
   nir_phi_instr_add_src(kr0_phi, kx_back, nr0);
   nir_phi_instr_add_src(kr1_phi, kx_back, nr1);
   nir_phi_instr_add_src(kin_phi, kx_back, kin_nxt);
   nir_phi_instr_add_src(kwt_phi, kx_back, nir_iadd(b, &kwt_phi->def, c4));
   nir_instr_insert(nir_before_block(kx_hdr), &kx_phi->instr);
   nir_instr_insert(nir_before_block(kx_hdr), &kr0_phi->instr);
   nir_instr_insert(nir_before_block(kx_hdr), &kr1_phi->instr);
   nir_instr_insert(nir_before_block(kx_hdr), &kin_phi->instr);
   nir_instr_insert(nir_before_block(kx_hdr), &kwt_phi->instr);
   if (stride_x == 1) {
      nir_phi_instr_add_src(kiv_phi, kx_back, iv1);
      nir_instr_insert(nir_before_block(kx_hdr), &kiv_phi->instr);
   }
   nir_pop_loop(b, kx_lp);

   nir_block *ky_back = nir_loop_last_block(ky_lp);
   nir_phi_instr_add_src(ky_phi, ky_back, nir_iadd_imm(b, &ky_phi->def, 1));
   nir_phi_instr_add_src(r0_phi, ky_back, &kr0_phi->def);
   nir_phi_instr_add_src(r1_phi, ky_back, &kr1_phi->def);
   nir_phi_instr_add_src(in_phi, ky_back, nir_iadd(b, &in_phi->def, dy_step4));
   nir_phi_instr_add_src(wt_phi, ky_back, nir_iadd(b, &wt_phi->def, wt_row4));
   nir_instr_insert(nir_before_block(ky_hdr), &ky_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &r0_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &r1_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &in_phi->instr);
   nir_instr_insert(nir_before_block(ky_hdr), &wt_phi->instr);
   nir_pop_loop(b, ky_lp);

   *a0_out = &r0_phi->def;
   *a1_out = &r1_phi->def;
}

static nir_def *
dual_push_u64(nir_builder *b, uint32_t offset_B, uint32_t push_bytes)
{
   nir_def *lo = dw_push(b, offset_B, push_bytes);
   nir_def *hi = dw_push(b, offset_B + 4, push_bytes);
   return nir_pack_64_2x32_split(b, lo, hi);
}

nir_shader *
hk_nir_build_depthwise_vec4(const nir_shader *producer,
                            const struct hk_nir_depthwise_output *info,
                            const struct hk_nir_clamp_epilogue *epilogue,
                            enum hk_compute_pair_dual_write dual_write)
{
   if (!producer || !info || !info->vec4_valid ||
       info->desc_comps < 3 || info->desc_comps > 5)
      return NULL;
   if (dual_write != HK_COMPUTE_PAIR_DUAL_NONE &&
       dual_write != HK_COMPUTE_PAIR_DUAL_COPY)
      return NULL;
   if (epilogue &&
       (!hk_nir_clamp_epilogue_valid(epilogue) ||
        producer->info.float_controls_execution_mode !=
           epilogue->float_controls_execution_mode))
      return NULL;

   const uint32_t dual_off = hk_dual_write_push_offset(info->push_bytes);
   const uint32_t total_push_bytes =
      dual_write == HK_COMPUTE_PAIR_DUAL_COPY
         ? dual_off + sizeof(struct hk_dual_copy_push)
         : info->push_bytes;
   if (total_push_bytes > 256u)
      return NULL;

   nir_shader *nir = nir_shader_create(NULL, MESA_SHADER_COMPUTE,
                                       producer->options);
   if (!nir)
      return NULL;
   nir->info.name = ralloc_asprintf(nir, "%s_dw_vec4%s%s",
                                    producer->info.name ? producer->info.name : "cs",
                                    epilogue ? "_clamp" : "",
                                    dual_write == HK_COMPUTE_PAIR_DUAL_COPY ? "_dcopy" : "");
   /* 3D grid (c4, ceil(out_w/2), out_h): 8 contiguous ch4 vec4s (128B full
    * cache line) along X x 8 ox_pairs along Y for spatial L1 reuse. */
   nir->info.workgroup_size[0] = 8;
   nir->info.workgroup_size[1] = 8;
   nir->info.workgroup_size[2] = 2;
   nir->info.workgroup_size_variable = false;
   nir->info.num_ssbos = producer->info.num_ssbos;
   nir->info.float_controls_execution_mode = producer->info.float_controls_execution_mode;
   nir->num_uniforms = total_push_bytes;

   nir_function *fn = nir_function_create(nir, "main");
   fn->is_entrypoint = true;
   nir_function_impl *impl = nir_function_impl_create(fn);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   const uint32_t pb = total_push_bytes;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *ch4 = nir_channel(&b, gid, 0);
   nir_def *ox_pair = nir_channel(&b, gid, 1);
   nir_def *oy = nir_channel(&b, gid, 2);

   nir_def *in_w = dw_push(&b, info->input_width_offset_B, pb);
   nir_def *in_h = dw_push(&b, info->input_height_offset_B, pb);
   nir_def *out_w = dw_push(&b, info->width_offset_B, pb);
   nir_def *out_h = dw_push(&b, info->height_offset_B, pb);
   nir_def *kw = dw_push(&b, info->kernel_width_offset_B, pb);
   nir_def *kh = dw_push(&b, info->kernel_height_offset_B, pb);
   nir_def *sx = dw_push(&b, info->stride_x_offset_B, pb);
   nir_def *sy = dw_push(&b, info->stride_y_offset_B, pb);
   nir_def *dx = dw_push(&b, info->dil_x_offset_B, pb);
   nir_def *dy = dw_push(&b, info->dil_y_offset_B, pb);
   nir_def *px = dw_push(&b, info->pad_x_offset_B, pb);
   nir_def *py = dw_push(&b, info->pad_y_offset_B, pb);
   nir_def *ch = dw_push(&b, info->channels_offset_B, pb);
   nir_def *has_bias =
      (info->has_bias_buffer && info->has_bias_offset_B != UINT32_MAX)
      ? dw_push(&b, info->has_bias_offset_B, pb)
      : NULL;

   nir_def *c4 = nir_ushr_imm(&b, ch, 2);
   nir_def *out_pairs = nir_ushr_imm(&b, nir_iadd_imm(&b, out_w, 1), 1);

   nir_if *active = nir_push_if(&b, nir_iand(&b, nir_iand(&b, nir_ult(&b, ch4, c4),
                                                              nir_ult(&b, ox_pair, out_pairs)),
                                                 nir_ult(&b, oy, out_h)));
   {
      nir_def *km1_dx = nir_imul(&b, nir_iadd_imm(&b, kw, -1), dx);
      nir_def *km1_dy = nir_imul(&b, nir_iadd_imm(&b, kh, -1), dy);
      nir_def *row_stride4 = nir_imul(&b, in_w, c4);
      nir_def *wt_row4 = nir_imul(&b, kw, c4);
      nir_def *dx_step4 = nir_imul(&b, dx, c4);
      nir_def *dy_step4 = nir_imul(&b, dy, row_stride4);
      nir_def *sx_step4 = nir_imul(&b, sx, c4);

      nir_def *ox0_chk = nir_ishl_imm(&b, ox_pair, 1);
      nir_def *ix0_chk = nir_iadd(&b, nir_imul(&b, ox0_chk, sx), px);
      nir_def *iy0_chk = nir_iadd(&b, nir_imul(&b, oy, sy), py);
      nir_def *in_y_ok = nir_iand(&b, nir_ige_imm(&b, iy0_chk, 0),
                                      nir_ilt(&b, nir_iadd(&b, iy0_chk, km1_dy), in_h));
      nir_def *in_x_ok = nir_iand(&b, nir_iand(&b, nir_ult(&b, nir_iadd_imm(&b, ox0_chk, 1), out_w),
                                                   nir_ige_imm(&b, ix0_chk, 0)),
                                      nir_ilt(&b, nir_iadd(&b, nir_iadd(&b, ix0_chk, sx), km1_dx), in_w));
      nir_def *is_interior2 = nir_iand(&b, in_y_ok, in_x_ok);

      nir_def *zero_f = nir_imm_float(&b, 0.0f);
      nir_def *zero_v4 = nir_vec4(&b, zero_f, zero_f, zero_f, zero_f);

      nir_def *acc0_int = NULL, *acc1_int = NULL;
      nir_def *acc0_brd = NULL, *acc1_brd = NULL;
      nir_if *int_if = nir_push_if(&b, is_interior2);
      {
         nir_def *in_base0 =
            nir_iadd(&b, nir_iadd(&b, nir_imul(&b, iy0_chk, row_stride4),
                                      nir_imul(&b, ix0_chk, c4)), ch4);
         nir_def *is_3x3_d1 = nir_iand(&b,
            nir_iand(&b, nir_ieq_imm(&b, kw, 3), nir_ieq_imm(&b, kh, 3)),
            nir_iand(&b, nir_ieq_imm(&b, dx, 1), nir_ieq_imm(&b, dy, 1)));
         nir_def *is_s1_d1 = nir_iand(&b, nir_ieq_imm(&b, dx, 1), nir_ieq_imm(&b, sx, 1));
         nir_def *a0_s1 = NULL, *a1_s1 = NULL;
         nir_def *a0_g = NULL, *a1_g = NULL;
         nir_if *s1_if = nir_push_if(&b, is_s1_d1);
         {
            nir_def *a0_3 = NULL, *a1_3 = NULL, *a0_k = NULL, *a1_k = NULL;
            nir_if *k3_if = nir_push_if(&b, is_3x3_d1);
            {
               emit_dw_3x3_s1_1x2(&b, info, in_base0, ch4, c4, row_stride4, zero_v4,
                                  &a0_3, &a1_3);
            }
            nir_push_else(&b, k3_if);
            {
               emit_dw_kx_streamed(&b, info, kw, kh, 1, in_base0, ch4, c4, dy_step4, wt_row4,
                                   zero_v4, &a0_k, &a1_k);
            }
            nir_pop_if(&b, k3_if);
            a0_s1 = nir_if_phi(&b, a0_3, a0_k);
            a1_s1 = nir_if_phi(&b, a1_3, a1_k);
         }
         nir_push_else(&b, s1_if);
         {
            nir_def *is_s2_d1 = nir_iand(&b, nir_ieq_imm(&b, dx, 1), nir_ieq_imm(&b, sx, 2));
            nir_def *a0_s2 = NULL, *a1_s2 = NULL;
            nir_def *a0_gen = NULL, *a1_gen = NULL;
            nir_if *s2_if = nir_push_if(&b, is_s2_d1);
            {
               nir_def *a0_3 = NULL, *a1_3 = NULL, *a0_k = NULL, *a1_k = NULL;
               nir_if *k3_if = nir_push_if(&b, is_3x3_d1);
               {
                  emit_dw_3x3_s2_1x2(&b, info, in_base0, ch4, c4, row_stride4, zero_v4,
                                     &a0_3, &a1_3);
               }
               nir_push_else(&b, k3_if);
               {
                  emit_dw_kx_streamed(&b, info, kw, kh, 2, in_base0, ch4, c4, dy_step4, wt_row4,
                                      zero_v4, &a0_k, &a1_k);
               }
               nir_pop_if(&b, k3_if);
               a0_s2 = nir_if_phi(&b, a0_3, a0_k);
               a1_s2 = nir_if_phi(&b, a1_3, a1_k);
            }
            nir_push_else(&b, s2_if);
            {
               nir_def *zero_i = nir_imm_int(&b, 0);
               nir_loop *oy_lp = nir_push_loop(&b);
               oy_lp->control = nir_loop_control_dont_unroll;
               nir_block *oy_pre = nir_cf_node_as_block(nir_cf_node_prev(&oy_lp->cf_node));
               nir_block *oy_hdr = nir_loop_first_block(oy_lp);

               nir_phi_instr *ky_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
               nir_phi_instr_add_src(ky_phi, oy_pre, zero_i);

               nir_phi_instr *oy_a0_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&oy_a0_phi->instr, &oy_a0_phi->def, 4, 32);
               nir_phi_instr_add_src(oy_a0_phi, oy_pre, zero_v4);

               nir_phi_instr *oy_a1_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&oy_a1_phi->instr, &oy_a1_phi->def, 4, 32);
               nir_phi_instr_add_src(oy_a1_phi, oy_pre, zero_v4);

               nir_phi_instr *oy_in_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&oy_in_phi->instr, &oy_in_phi->def, 1, 32);
               nir_phi_instr_add_src(oy_in_phi, oy_pre, in_base0);

               nir_phi_instr *oy_wt_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&oy_wt_phi->instr, &oy_wt_phi->def, 1, 32);
               nir_phi_instr_add_src(oy_wt_phi, oy_pre, ch4);

               nir_if *oy_done = nir_push_if(&b, nir_ige(&b, &ky_phi->def, kh));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, oy_done);

               nir_loop *ix_lp = nir_push_loop(&b);
               ix_lp->control = nir_loop_control_dont_unroll;
               nir_block *ix_pre = nir_cf_node_as_block(nir_cf_node_prev(&ix_lp->cf_node));
               nir_block *ix_hdr = nir_loop_first_block(ix_lp);

               nir_phi_instr *kx_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&kx_phi->instr, &kx_phi->def, 1, 32);
               nir_phi_instr_add_src(kx_phi, ix_pre, zero_i);

               nir_phi_instr *ix_a0_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&ix_a0_phi->instr, &ix_a0_phi->def, 4, 32);
               nir_phi_instr_add_src(ix_a0_phi, ix_pre, &oy_a0_phi->def);

               nir_phi_instr *ix_a1_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&ix_a1_phi->instr, &ix_a1_phi->def, 4, 32);
               nir_phi_instr_add_src(ix_a1_phi, ix_pre, &oy_a1_phi->def);

               nir_phi_instr *ix_in_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&ix_in_phi->instr, &ix_in_phi->def, 1, 32);
               nir_phi_instr_add_src(ix_in_phi, ix_pre, &oy_in_phi->def);

               nir_phi_instr *ix_wt_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&ix_wt_phi->instr, &ix_wt_phi->def, 1, 32);
               nir_phi_instr_add_src(ix_wt_phi, ix_pre, &oy_wt_phi->def);

               nir_if *ix_done = nir_push_if(&b, nir_ige(&b, &kx_phi->def, kw));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, ix_done);

               nir_def *wv = nir_load_global(&b, 4, 32,
                  dw_vec4_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                               &ix_wt_phi->def, read),
                  .align_mul = 16, .access = read);
               nir_def *iv0 = nir_load_global(&b, 4, 32,
                  dw_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                               &ix_in_phi->def, read),
                  .align_mul = 16, .access = read);
               nir_def *a0_next = dw_fma4(&b, &ix_a0_phi->def, iv0, wv, info);
               nir_def *iv1 = nir_load_global(&b, 4, 32,
                  dw_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                               nir_iadd(&b, &ix_in_phi->def, sx_step4), read),
                  .align_mul = 16, .access = read);
               nir_def *a1_next = dw_fma4(&b, &ix_a1_phi->def, iv1, wv, info);

               nir_block *ix_back = nir_loop_last_block(ix_lp);
               nir_phi_instr_add_src(kx_phi, ix_back, nir_iadd_imm(&b, &kx_phi->def, 1));
               nir_phi_instr_add_src(ix_a0_phi, ix_back, a0_next);
               nir_phi_instr_add_src(ix_a1_phi, ix_back, a1_next);
               nir_phi_instr_add_src(ix_in_phi, ix_back, nir_iadd(&b, &ix_in_phi->def, dx_step4));
               nir_phi_instr_add_src(ix_wt_phi, ix_back, nir_iadd(&b, &ix_wt_phi->def, c4));
               nir_instr_insert(nir_before_block(ix_hdr), &kx_phi->instr);
               nir_instr_insert(nir_before_block(ix_hdr), &ix_a0_phi->instr);
               nir_instr_insert(nir_before_block(ix_hdr), &ix_a1_phi->instr);
               nir_instr_insert(nir_before_block(ix_hdr), &ix_in_phi->instr);
               nir_instr_insert(nir_before_block(ix_hdr), &ix_wt_phi->instr);
               nir_pop_loop(&b, ix_lp);

               nir_block *oy_back = nir_loop_last_block(oy_lp);
               nir_phi_instr_add_src(ky_phi, oy_back, nir_iadd_imm(&b, &ky_phi->def, 1));
               nir_phi_instr_add_src(oy_a0_phi, oy_back, &ix_a0_phi->def);
               nir_phi_instr_add_src(oy_a1_phi, oy_back, &ix_a1_phi->def);
               nir_phi_instr_add_src(oy_in_phi, oy_back, nir_iadd(&b, &oy_in_phi->def, dy_step4));
               nir_phi_instr_add_src(oy_wt_phi, oy_back, nir_iadd(&b, &oy_wt_phi->def, wt_row4));
               nir_instr_insert(nir_before_block(oy_hdr), &ky_phi->instr);
               nir_instr_insert(nir_before_block(oy_hdr), &oy_a0_phi->instr);
               nir_instr_insert(nir_before_block(oy_hdr), &oy_a1_phi->instr);
               nir_instr_insert(nir_before_block(oy_hdr), &oy_in_phi->instr);
               nir_instr_insert(nir_before_block(oy_hdr), &oy_wt_phi->instr);
               nir_pop_loop(&b, oy_lp);
               a0_gen = &oy_a0_phi->def;
               a1_gen = &oy_a1_phi->def;
            }
            nir_pop_if(&b, s2_if);
            a0_g = nir_if_phi(&b, a0_s2, a0_gen);
            a1_g = nir_if_phi(&b, a1_s2, a1_gen);
         }
         nir_pop_if(&b, s1_if);
         acc0_int = nir_if_phi(&b, a0_s1, a0_g);
         acc1_int = nir_if_phi(&b, a1_s1, a1_g);
      }
      nir_push_else(&b, int_if);
      {
         /* Recompute border coordinates inside the border branch so they are
          * not kept live in registers across the hot interior loop. */
         nir_def *ox0_b = nir_ishl_imm(&b, ox_pair, 1);
         nir_def *has_p1_b = nir_ult(&b, nir_iadd_imm(&b, ox0_b, 1), out_w);
         nir_def *ix0_b = nir_iadd(&b, nir_imul(&b, ox0_b, sx), px);
         nir_def *ix1_b = nir_iadd(&b, ix0_b, sx);
         nir_def *iy0_b = nir_iadd(&b, nir_imul(&b, oy, sy), py);

         nir_def *zero_i = nir_imm_int(&b, 0);
         nir_loop *oy_lp = nir_push_loop(&b);
         oy_lp->control = nir_loop_control_dont_unroll;
         nir_block *oy_pre = nir_cf_node_as_block(nir_cf_node_prev(&oy_lp->cf_node));
         nir_block *oy_hdr = nir_loop_first_block(oy_lp);

         nir_phi_instr *ky_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
         nir_phi_instr_add_src(ky_phi, oy_pre, zero_i);

         nir_phi_instr *oy_a0_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&oy_a0_phi->instr, &oy_a0_phi->def, 4, 32);
         nir_phi_instr_add_src(oy_a0_phi, oy_pre, zero_v4);

         nir_phi_instr *oy_a1_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&oy_a1_phi->instr, &oy_a1_phi->def, 4, 32);
         nir_phi_instr_add_src(oy_a1_phi, oy_pre, zero_v4);

         nir_if *oy_done = nir_push_if(&b, nir_ige(&b, &ky_phi->def, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, oy_done);

         nir_def *iy = nir_iadd(&b, iy0_b, nir_imul(&b, dy, &ky_phi->def));
         nir_def *y_a0 = NULL, *y_a1 = NULL;
         nir_if *y_ok = nir_push_if(&b, nir_ult(&b, iy, in_h));
         {
            nir_def *in_row4 = nir_iadd(&b, nir_imul(&b, iy, row_stride4), ch4);
            nir_def *wt_y4 = nir_iadd(&b, nir_imul(&b, &ky_phi->def, wt_row4), ch4);

            nir_loop *ix_lp = nir_push_loop(&b);
            ix_lp->control = nir_loop_control_dont_unroll;
            nir_block *ix_pre = nir_cf_node_as_block(nir_cf_node_prev(&ix_lp->cf_node));
            nir_block *ix_hdr = nir_loop_first_block(ix_lp);

            nir_phi_instr *kx_phi = nir_phi_instr_create(b.shader);
            nir_def_init(&kx_phi->instr, &kx_phi->def, 1, 32);
            nir_phi_instr_add_src(kx_phi, ix_pre, zero_i);

            nir_phi_instr *ix_a0_phi = nir_phi_instr_create(b.shader);
            nir_def_init(&ix_a0_phi->instr, &ix_a0_phi->def, 4, 32);
            nir_phi_instr_add_src(ix_a0_phi, ix_pre, &oy_a0_phi->def);

            nir_phi_instr *ix_a1_phi = nir_phi_instr_create(b.shader);
            nir_def_init(&ix_a1_phi->instr, &ix_a1_phi->def, 4, 32);
            nir_phi_instr_add_src(ix_a1_phi, ix_pre, &oy_a1_phi->def);

            nir_if *ix_done = nir_push_if(&b, nir_ige(&b, &kx_phi->def, kw));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, ix_done);

            nir_def *kdx = nir_imul(&b, dx, &kx_phi->def);
            nir_def *ix_a = nir_iadd(&b, ix0_b, kdx);
            nir_def *ix_b = nir_iadd(&b, ix1_b, kdx);
            nir_def *ok0 = nir_ult(&b, ix_a, in_w);
            nir_def *ok1 = nir_iand(&b, has_p1_b, nir_ult(&b, ix_b, in_w));

            nir_def *any_a0 = NULL, *any_a1 = NULL;
            nir_if *any_ok = nir_push_if(&b, nir_ior(&b, ok0, ok1));
            {
               nir_def *wt_idx4 = nir_iadd(&b, wt_y4, nir_imul(&b, &kx_phi->def, c4));
               nir_def *wv = nir_load_global(&b, 4, 32,
                  dw_vec4_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                               wt_idx4, read),
                  .align_mul = 16, .access = read);

               nir_def *hit0 = NULL;
               nir_if *if0 = nir_push_if(&b, ok0);
               {
                  nir_def *in_idx4 = nir_iadd(&b, in_row4, nir_imul(&b, ix_a, c4));
                  nir_def *iv0 = nir_load_global(&b, 4, 32,
                     dw_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                                  in_idx4, read),
                     .align_mul = 16, .access = read);
                  hit0 = dw_fma4(&b, &ix_a0_phi->def, iv0, wv, info);
               }
               nir_push_else(&b, if0);
               nir_pop_if(&b, if0);
               any_a0 = nir_if_phi(&b, hit0, &ix_a0_phi->def);

               nir_def *hit1 = NULL;
               nir_if *if1 = nir_push_if(&b, ok1);
               {
                  nir_def *in_idx4 = nir_iadd(&b, in_row4, nir_imul(&b, ix_b, c4));
                  nir_def *iv1 = nir_load_global(&b, 4, 32,
                     dw_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                                  in_idx4, read),
                     .align_mul = 16, .access = read);
                  hit1 = dw_fma4(&b, &ix_a1_phi->def, iv1, wv, info);
               }
               nir_push_else(&b, if1);
               nir_pop_if(&b, if1);
               any_a1 = nir_if_phi(&b, hit1, &ix_a1_phi->def);
            }
            nir_push_else(&b, any_ok);
            nir_pop_if(&b, any_ok);
            nir_def *ix_a0_next = nir_if_phi(&b, any_a0, &ix_a0_phi->def);
            nir_def *ix_a1_next = nir_if_phi(&b, any_a1, &ix_a1_phi->def);

            nir_block *ix_back = nir_loop_last_block(ix_lp);
            nir_phi_instr_add_src(kx_phi, ix_back, nir_iadd_imm(&b, &kx_phi->def, 1));
            nir_phi_instr_add_src(ix_a0_phi, ix_back, ix_a0_next);
            nir_phi_instr_add_src(ix_a1_phi, ix_back, ix_a1_next);
            nir_instr_insert(nir_before_block(ix_hdr), &kx_phi->instr);
            nir_instr_insert(nir_before_block(ix_hdr), &ix_a0_phi->instr);
            nir_instr_insert(nir_before_block(ix_hdr), &ix_a1_phi->instr);
            nir_pop_loop(&b, ix_lp);
            y_a0 = &ix_a0_phi->def;
            y_a1 = &ix_a1_phi->def;
         }
         nir_push_else(&b, y_ok);
         nir_pop_if(&b, y_ok);
         nir_def *oy_a0_next = nir_if_phi(&b, y_a0, &oy_a0_phi->def);
         nir_def *oy_a1_next = nir_if_phi(&b, y_a1, &oy_a1_phi->def);

         nir_block *oy_back = nir_loop_last_block(oy_lp);
         nir_phi_instr_add_src(ky_phi, oy_back, nir_iadd_imm(&b, &ky_phi->def, 1));
         nir_phi_instr_add_src(oy_a0_phi, oy_back, oy_a0_next);
         nir_phi_instr_add_src(oy_a1_phi, oy_back, oy_a1_next);
         nir_instr_insert(nir_before_block(oy_hdr), &ky_phi->instr);
         nir_instr_insert(nir_before_block(oy_hdr), &oy_a0_phi->instr);
         nir_instr_insert(nir_before_block(oy_hdr), &oy_a1_phi->instr);
         nir_pop_loop(&b, oy_lp);
         acc0_brd = &oy_a0_phi->def;
         acc1_brd = &oy_a1_phi->def;
      }
      nir_pop_if(&b, int_if);
      nir_def *acc0 = nir_if_phi(&b, acc0_int, acc0_brd);
      nir_def *acc1 = nir_if_phi(&b, acc1_int, acc1_brd);

      nir_def *val0 = acc0, *val1 = acc1;
      if (info->has_bias_buffer) {
         if (has_bias) {
            nir_def *b_a0 = NULL, *b_a1 = NULL;
            nir_if *b_if = nir_push_if(&b, nir_ine_imm(&b, has_bias, 0));
            {
               nir_def *ba = dw_vec4_addr(&b, info->reads[2], info->desc_comps,
                                          info->read_res_type[2], ch4, read);
               nir_def *bv = nir_load_global(&b, 4, 32, ba, .align_mul = 16, .access = read);
               b.fp_math_ctrl = info->bias_fadd_fp_ctrl;
               b_a0 = nir_fadd(&b, acc0, bv);
               b_a1 = nir_fadd(&b, acc1, bv);
               b.fp_math_ctrl = 0;
            }
            nir_push_else(&b, b_if);
            nir_pop_if(&b, b_if);
            val0 = nir_if_phi(&b, b_a0, acc0);
            val1 = nir_if_phi(&b, b_a1, acc1);
         } else {
            nir_def *ba = dw_vec4_addr(&b, info->reads[2], info->desc_comps,
                                       info->read_res_type[2], ch4, read);
            nir_def *bv = nir_load_global(&b, 4, 32, ba, .align_mul = 16, .access = read);
            b.fp_math_ctrl = info->bias_fadd_fp_ctrl;
            val0 = nir_fadd(&b, acc0, bv);
            val1 = nir_fadd(&b, acc1, bv);
            b.fp_math_ctrl = 0;
         }
      }

      if (epilogue) {
         val0 = apply_clamp_epilogue(&b, val0, epilogue);
         val1 = apply_clamp_epilogue(&b, val1, epilogue);
      }

      nir_def *ox0 = nir_ishl_imm(&b, ox_pair, 1);
      nir_def *has_p1 = nir_ult(&b, nir_iadd_imm(&b, ox0, 1), out_w);
      nir_def *out_idx0 =
         nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, oy, out_w), ox0), c4), ch4);
      nir_def *dst2_va =
         dual_write == HK_COMPUTE_PAIR_DUAL_COPY
            ? dual_push_u64(&b, dual_off + offsetof(struct hk_dual_copy_push, dst2_va_lo), pb)
            : NULL;

      nir_def *out_idx1 = nir_iadd(&b, out_idx0, c4);
      nir_def *oa0 = dw_vec4_addr(&b, info->output, info->desc_comps,
                                  info->out_res_type, out_idx0, ACCESS_NON_READABLE);
      nir_store_global(&b, val0, oa0, .align_mul = 16, .write_mask = 0xf,
                       .access = ACCESS_NON_READABLE);
      nir_if *st1 = nir_push_if(&b, has_p1);
      {
         nir_def *oa1 = dw_vec4_addr(&b, info->output, info->desc_comps,
                                     info->out_res_type, out_idx1,
                                     ACCESS_NON_READABLE);
         nir_store_global(&b, val1, oa1, .align_mul = 16, .write_mask = 0xf,
                          .access = ACCESS_NON_READABLE);
      }
      nir_pop_if(&b, st1);

      if (dst2_va) {
         nir_def *da0 = nir_iadd(&b, dst2_va, nir_u2u64(&b, nir_ishl_imm(&b, out_idx0, 4)));
         nir_store_global(&b, val0, da0, .align_mul = 16, .write_mask = 0xf,
                          .access = ACCESS_NON_READABLE);
         nir_if *st2 = nir_push_if(&b, has_p1);
         {
            nir_def *da1 = nir_iadd(&b, dst2_va, nir_u2u64(&b, nir_ishl_imm(&b, out_idx1, 4)));
            nir_store_global(&b, val1, da1, .align_mul = 16, .write_mask = 0xf,
                             .access = ACCESS_NON_READABLE);
         }
         nir_pop_if(&b, st2);
      }
   }
   nir_pop_if(&b, active);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}

static nir_def *
gemm_f32_addr(nir_builder *b, struct hk_nir_clamp_buffer buf, uint32_t desc_comps,
              uint32_t res_type, nir_def *idx, enum gl_access_qualifier access)
{
   if (buf.offset_B)
      idx = nir_iadd_imm(b, idx, buf.offset_B / 4);
   nir_def *resource = nir_vulkan_resource_index(
      b, desc_comps, 32, nir_imm_int(b, 0),
      .desc_type = nir_descriptor_type_storage_buffer, .resource_type = res_type,
      .desc_set = buf.set, .binding = buf.binding);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      b, desc_comps, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   const glsl_type *array_type = glsl_array_type(glsl_float_type(), 0, 4);
   nir_deref_instr *array = nir_build_deref_cast_with_alignment(
      b, descriptor, nir_var_mem_ssbo, array_type, 0, 4, 0);
   nir_deref_instr *element = nir_build_deref_array(b, array, idx);
   element->arr.in_bounds = true;
   return nir_deref_buffer_address(b, 1, 64, &element->def, .access = access);
}

static nir_def *
gemm_vec4_addr(nir_builder *b, struct hk_nir_clamp_buffer buf, uint32_t desc_comps,
               uint32_t res_type, nir_def *idx4, enum gl_access_qualifier access)
{
   if (buf.offset_B)
      idx4 = nir_iadd_imm(b, idx4, buf.offset_B / 16);
   nir_def *resource = nir_vulkan_resource_index(
      b, desc_comps, 32, nir_imm_int(b, 0),
      .desc_type = nir_descriptor_type_storage_buffer, .resource_type = res_type,
      .desc_set = buf.set, .binding = buf.binding);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      b, desc_comps, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   const glsl_type *array_type = glsl_array_type(glsl_vec4_type(), 0, 16);
   nir_deref_instr *array = nir_build_deref_cast_with_alignment(
      b, descriptor, nir_var_mem_ssbo, array_type, 0, 16, 0);
   nir_deref_instr *element = nir_build_deref_array(b, array, idx4);
   element->arr.in_bounds = true;
   return nir_deref_buffer_address(b, 1, 64, &element->def, .access = access);
}

struct hk_gemm_dual_ctx {
   enum hk_compute_pair_dual_write mode;
   nir_def *dst2_va;
   nir_def *simple_pad;
   nir_def *in_h;
   nir_def *in_w;
   struct hk_nir_fast_udiv in_w_div;
   nir_def *out_h;
   nir_def *out_w;
   nir_def *pad_n0;
   nir_def *pad_h0;
   nir_def *pad_w0;
   nir_def *pad_c0;
};

static void
emit_pad4_zero_xrange(nir_builder *b, nir_def *dst2_va, nir_def *oy,
                      nir_def *x_start, nir_def *x_end, nir_def *out_w,
                      nir_def *n, nir_def *col, unsigned comps)
{
   nir_def *zero_f = nir_imm_float(b, 0.0f);
   nir_def *zero_v = comps == 2 ? nir_vec2(b, zero_f, zero_f) : zero_f;
   nir_def *row_y = nir_imul(b, oy, out_w);
   nir_loop *xlp = nir_push_loop(b);
   xlp->control = nir_loop_control_dont_unroll;
   nir_block *xpre = nir_cf_node_as_block(nir_cf_node_prev(&xlp->cf_node));
   nir_block *xhdr = nir_loop_first_block(xlp);
   nir_phi_instr *xphi = nir_phi_instr_create(b->shader);
   nir_def_init(&xphi->instr, &xphi->def, 1, 32);
   nir_phi_instr_add_src(xphi, xpre, x_start);
   nir_def *ox = &xphi->def;
   nir_if *xbrk = nir_push_if(b, nir_uge(b, ox, x_end));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, xbrk);
   nir_def *d_idx = nir_iadd(b, nir_imul(b, nir_iadd(b, row_y, ox), n), col);
   nir_def *da = nir_iadd(b, dst2_va, nir_u2u64(b, nir_ishl_imm(b, d_idx, 2)));
   nir_store_global(b, zero_v, da, .align_mul = 4,
                    .write_mask = comps == 2 ? 3 : 1, .access = ACCESS_NON_READABLE);
   nir_def *xnext = nir_iadd_imm(b, ox, 1);
   nir_block *xback = nir_cursor_current_block(b->cursor);
   nir_phi_instr_add_src(xphi, xback, xnext);
   nir_instr_insert(nir_before_block(xhdr), &xphi->instr);
   nir_pop_loop(b, xlp);
}

static void
emit_pad4_zero_yrange_x(nir_builder *b, nir_def *dst2_va,
                        nir_def *y_start, nir_def *y_end,
                        nir_def *ox_mid, nir_def *is_x0, nir_def *is_xlast,
                        nir_def *pad_w0, nir_def *in_w, nir_def *out_w,
                        nir_def *n, nir_def *col, unsigned comps)
{
   nir_def *zero_f = nir_imm_float(b, 0.0f);
   nir_def *zero_v = comps == 2 ? nir_vec2(b, zero_f, zero_f) : zero_f;
   nir_loop *ylp = nir_push_loop(b);
   ylp->control = nir_loop_control_dont_unroll;
   nir_block *ypre = nir_cf_node_as_block(nir_cf_node_prev(&ylp->cf_node));
   nir_block *yhdr = nir_loop_first_block(ylp);
   nir_phi_instr *yphi = nir_phi_instr_create(b->shader);
   nir_def_init(&yphi->instr, &yphi->def, 1, 32);
   nir_phi_instr_add_src(yphi, ypre, y_start);
   nir_def *oy = &yphi->def;
   nir_if *ybrk = nir_push_if(b, nir_uge(b, oy, y_end));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, ybrk);

   nir_def *d_mid = nir_iadd(b, nir_imul(b, nir_iadd(b, nir_imul(b, oy, out_w), ox_mid), n), col);
   nir_store_global(b, zero_v,
                    nir_iadd(b, dst2_va, nir_u2u64(b, nir_ishl_imm(b, d_mid, 2))),
                    .align_mul = 4, .write_mask = comps == 2 ? 3 : 1,
                    .access = ACCESS_NON_READABLE);

   nir_if *lx = nir_push_if(b, is_x0);
   emit_pad4_zero_xrange(b, dst2_va, oy, nir_imm_int(b, 0), pad_w0, out_w, n, col, comps);
   nir_pop_if(b, lx);

   nir_if *rx = nir_push_if(b, is_xlast);
   emit_pad4_zero_xrange(b, dst2_va, oy, nir_iadd(b, pad_w0, in_w), out_w, out_w, n, col, comps);
   nir_pop_if(b, rx);

   nir_def *ynext = nir_iadd_imm(b, oy, 1);
   nir_block *yback = nir_cursor_current_block(b->cursor);
   nir_phi_instr_add_src(yphi, yback, ynext);
   nir_instr_insert(nir_before_block(yhdr), &yphi->instr);
   nir_pop_loop(b, ylp);
}

static void
emit_pad4_simple_border(nir_builder *b, const struct hk_gemm_dual_ctx *dual,
                        nir_def *iy, nir_def *ix, nir_def *n, nir_def *col,
                        unsigned comps)
{
   nir_def *is_x0 = nir_ieq_imm(b, ix, 0);
   nir_def *is_xlast = nir_ieq(b, ix, nir_iadd_imm(b, dual->in_w, -1));
   nir_def *is_y0 = nir_ieq_imm(b, iy, 0);
   nir_def *is_ylast = nir_ieq(b, iy, nir_iadd_imm(b, dual->in_h, -1));
   nir_def *on_edge = nir_ior(b, nir_ior(b, is_x0, is_xlast), nir_ior(b, is_y0, is_ylast));
   nir_if *ef = nir_push_if(b, on_edge);
   {
      nir_def *oy_mid = nir_iadd(b, iy, dual->pad_h0);
      nir_def *ox_mid = nir_iadd(b, ix, dual->pad_w0);
      nir_if *lx = nir_push_if(b, is_x0);
      emit_pad4_zero_xrange(b, dual->dst2_va, oy_mid, nir_imm_int(b, 0),
                            dual->pad_w0, dual->out_w, n, col, comps);
      nir_pop_if(b, lx);

      nir_if *rx = nir_push_if(b, is_xlast);
      emit_pad4_zero_xrange(b, dual->dst2_va, oy_mid,
                            nir_iadd(b, dual->pad_w0, dual->in_w), dual->out_w,
                            dual->out_w, n, col, comps);
      nir_pop_if(b, rx);

      nir_if *ty = nir_push_if(b, is_y0);
      emit_pad4_zero_yrange_x(b, dual->dst2_va, nir_imm_int(b, 0), dual->pad_h0,
                              ox_mid, is_x0, is_xlast, dual->pad_w0, dual->in_w,
                              dual->out_w, n, col, comps);
      nir_pop_if(b, ty);

      nir_if *by = nir_push_if(b, is_ylast);
      emit_pad4_zero_yrange_x(b, dual->dst2_va,
                              nir_iadd(b, dual->pad_h0, dual->in_h), dual->out_h,
                              ox_mid, is_x0, is_xlast, dual->pad_w0, dual->in_w,
                              dual->out_w, n, col, comps);
      nir_pop_if(b, by);
   }
   nir_pop_if(b, ef);
}

static void
emit_pad4_general_scatter_scalar(nir_builder *b, const struct hk_gemm_dual_ctx *dual,
                                 nir_def *iy, nir_def *ix, nir_def *n,
                                 nir_def *col, nir_def *val)
{
   nir_def *oy64 = nir_iadd(b, nir_u2u64(b, iy), nir_i2i64(b, dual->pad_h0));
   nir_def *ox64 = nir_iadd(b, nir_u2u64(b, ix), nir_i2i64(b, dual->pad_w0));
   nir_def *oc64 = nir_iadd(b, nir_u2u64(b, col), nir_i2i64(b, dual->pad_c0));
   nir_def *ok = nir_iand(b,
      nir_ieq_imm(b, dual->pad_n0, 0),
      nir_iand(b,
         nir_iand(b, nir_ige_imm(b, oy64, 0), nir_ilt(b, oy64, nir_u2u64(b, dual->out_h))),
         nir_iand(b,
            nir_iand(b, nir_ige_imm(b, ox64, 0), nir_ilt(b, ox64, nir_u2u64(b, dual->out_w))),
            nir_iand(b, nir_ige_imm(b, oc64, 0), nir_ilt(b, oc64, nir_u2u64(b, n))))));
   nir_if *sf = nir_push_if(b, ok);
   {
      nir_def *pix64 = nir_iadd(b, nir_imul(b, oy64, nir_u2u64(b, dual->out_w)), ox64);
      nir_def *idx64 = nir_iadd(b, nir_imul(b, pix64, nir_u2u64(b, n)), oc64);
      nir_def *da = nir_iadd(b, dual->dst2_va, nir_ishl_imm(b, idx64, 2));
      nir_store_global(b, val, da, .align_mul = 4, .write_mask = 1,
                       .access = ACCESS_NON_READABLE);
   }
   nir_pop_if(b, sf);
}


static void
emit_gemm_cmat_store_fast(nir_builder *b, const struct hk_nir_gemm_output *info,
                          const struct hk_nir_clamp_epilogue *epilogue,
                          const struct hk_gemm_dual_ctx *dual,
                          unsigned n_m_tiles, unsigned n_n_tiles,
                          nir_def *n, nir_def *bias_mode,
                          nir_def *c_row0, nir_def *c_row1, nir_def *c_col0,
                          nir_def *acc0[4], nir_def *acc1[4])
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   nir_def *c_idx0 = nir_iadd(b, nir_imul(b, c_row0, n), c_col0);
   nir_def *c_idx1 = (n_m_tiles == 2) ? nir_iadd(b, nir_imul(b, c_row1, n), c_col0) : NULL;

   const bool dyn_row = info->has_bias_buffer && info->has_bias_mode &&
                        (info->bias_mode_mask & (1u << 0));
   const bool dyn_col = info->has_bias_buffer && info->has_bias_mode &&
                        (info->bias_mode_mask & (1u << 1));
   const bool fixed_row = info->has_bias_buffer && !info->has_bias_mode &&
                          info->bias_mode_mask == (1u << 0);
   const bool fixed_col = info->has_bias_buffer && !info->has_bias_mode &&
                          info->bias_mode_mask == (1u << 1);

   nir_def *zero_f = nir_imm_float(b, 0.0f);
   nir_def *is_row_bias = dyn_row ? nir_ieq_imm(b, bias_mode, 0) : NULL;
   nir_def *is_col_bias = dyn_col ? nir_ieq_imm(b, bias_mode, 1) : NULL;

   nir_def *rb0_v2 = NULL, *rb1_v2 = NULL;
   if (fixed_row) {
      nir_def *rb0 = nir_load_global(b, 1, 32,
         gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                       c_row0, read),
         .align_mul = 4, .access = read);
      rb0_v2 = nir_vec2(b, rb0, rb0);
      if (n_m_tiles == 2) {
         nir_def *rb1 = nir_load_global(b, 1, 32,
            gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                          c_row1, read),
            .align_mul = 4, .access = read);
         rb1_v2 = nir_vec2(b, rb1, rb1);
      }
   } else if (dyn_row) {
      nir_def *rb0 = NULL, *rb1 = NULL;
      nir_if *rb_if = nir_push_if(b, is_row_bias);
      {
         rb0 = nir_load_global(b, 1, 32,
            gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                          c_row0, read),
            .align_mul = 4, .access = read);
         if (n_m_tiles == 2) {
            rb1 = nir_load_global(b, 1, 32,
               gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                             c_row1, read),
               .align_mul = 4, .access = read);
         }
      }
      nir_push_else(b, rb_if);
      nir_pop_if(b, rb_if);
      rb0 = nir_if_phi(b, rb0, zero_f);
      if (n_m_tiles == 2)
         rb1 = nir_if_phi(b, rb1, zero_f);
      rb0_v2 = nir_vec2(b, rb0, rb0);
      if (n_m_tiles == 2)
         rb1_v2 = nir_vec2(b, rb1, rb1);
   }

   nir_def *v0_arr[4] = {0}, *v1_arr[4] = {0};
   for (unsigned t = 0; t < n_n_tiles; ++t) {
      nir_def *v0 = acc0[t];
      nir_def *v1 = (n_m_tiles == 2) ? acc1[t] : NULL;
      if (fixed_row) {
         v0 = nir_fadd(b, v0, rb0_v2);
         if (n_m_tiles == 2)
            v1 = nir_fadd(b, v1, rb1_v2);
      } else if (fixed_col) {
         nir_def *cb = nir_load_global(b, 2, 32,
            gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                          nir_iadd_imm(b, c_col0, 8 * t), read),
            .align_mul = 4, .access = read);
         v0 = nir_fadd(b, v0, cb);
         if (n_m_tiles == 2)
            v1 = nir_fadd(b, v1, cb);
      } else if (dyn_row || dyn_col) {
         if (dyn_row) {
            nir_def *v0_rb = NULL, *v1_rb = NULL;
            nir_def *v0_nc = NULL, *v1_nc = NULL;
            nir_if *m0_if = nir_push_if(b, is_row_bias);
            {
               v0_rb = nir_fadd(b, v0, rb0_v2);
               if (n_m_tiles == 2)
                  v1_rb = nir_fadd(b, v1, rb1_v2);
            }
            nir_push_else(b, m0_if);
            {
               if (dyn_col) {
                  nir_def *v0_cb = NULL, *v1_cb = NULL;
                  nir_if *m1_if = nir_push_if(b, is_col_bias);
                  {
                     nir_def *cb = nir_load_global(b, 2, 32,
                        gemm_f32_addr(b, info->reads[2], info->desc_comps,
                                      info->read_res_type[2],
                                      nir_iadd_imm(b, c_col0, 8 * t), read),
                        .align_mul = 4, .access = read);
                     v0_cb = nir_fadd(b, v0, cb);
                     if (n_m_tiles == 2)
                        v1_cb = nir_fadd(b, v1, cb);
                  }
                  nir_push_else(b, m1_if);
                  nir_pop_if(b, m1_if);
                  v0_nc = nir_if_phi(b, v0_cb, v0);
                  if (n_m_tiles == 2)
                     v1_nc = nir_if_phi(b, v1_cb, v1);
               } else {
                  v0_nc = v0;
                  v1_nc = v1;
               }
            }
            nir_pop_if(b, m0_if);
            v0 = nir_if_phi(b, v0_rb, v0_nc);
            if (n_m_tiles == 2)
               v1 = nir_if_phi(b, v1_rb, v1_nc);
         } else {
            nir_def *v0_cb = NULL, *v1_cb = NULL;
            nir_if *m1_if = nir_push_if(b, is_col_bias);
            {
               nir_def *cb = nir_load_global(b, 2, 32,
                  gemm_f32_addr(b, info->reads[2], info->desc_comps,
                                info->read_res_type[2],
                                nir_iadd_imm(b, c_col0, 8 * t), read),
                  .align_mul = 4, .access = read);
               v0_cb = nir_fadd(b, v0, cb);
               if (n_m_tiles == 2)
                  v1_cb = nir_fadd(b, v1, cb);
            }
            nir_push_else(b, m1_if);
            nir_pop_if(b, m1_if);
            v0 = nir_if_phi(b, v0_cb, v0);
            if (n_m_tiles == 2)
               v1 = nir_if_phi(b, v1_cb, v1);
         }
      }

      if (epilogue) {
         v0 = apply_clamp_epilogue(b, v0, epilogue);
         if (n_m_tiles == 2)
            v1 = apply_clamp_epilogue(b, v1, epilogue);
      }

      const enum gl_access_qualifier out_acc =
         (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_PAD4) ? 0 : ACCESS_NON_READABLE;
      nir_def *ci0 = nir_iadd_imm(b, c_idx0, 8 * t);
      nir_store_global(b, v0,
         gemm_f32_addr(b, info->output, info->desc_comps, info->out_res_type,
                       ci0, out_acc),
         .align_mul = 4, .write_mask = 3, .access = out_acc);
      if (n_m_tiles == 2) {
         nir_def *ci1 = nir_iadd_imm(b, c_idx1, 8 * t);
         nir_store_global(b, v1,
            gemm_f32_addr(b, info->output, info->desc_comps, info->out_res_type,
                          ci1, out_acc),
            .align_mul = 4, .write_mask = 3, .access = out_acc);
      }
      v0_arr[t] = v0;
      v1_arr[t] = v1;
   }

   if (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_COPY) {
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         nir_def *ci0 = nir_iadd_imm(b, c_idx0, 8 * t);
         nir_store_global(b, v0_arr[t],
            nir_iadd(b, dual->dst2_va, nir_u2u64(b, nir_ishl_imm(b, ci0, 2))),
            .align_mul = 4, .write_mask = 3, .access = ACCESS_NON_READABLE);
      }
      if (n_m_tiles == 2) {
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            nir_def *ci1 = nir_iadd_imm(b, c_idx1, 8 * t);
            nir_store_global(b, v1_arr[t],
               nir_iadd(b, dual->dst2_va, nir_u2u64(b, nir_ishl_imm(b, ci1, 2))),
               .align_mul = 4, .write_mask = 3, .access = ACCESS_NON_READABLE);
         }
      }
   } else if (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_PAD4) {
      nir_if *spf = nir_push_if(b, dual->simple_pad);
      {
         nir_def *pad_iy0 = NULL, *pad_ix0 = NULL;
         hk_nir_fast_udivmod(b, &dual->in_w_div, c_row0, &pad_iy0, &pad_ix0);
         nir_def *pad_row0 = nir_imul(
            b,
            nir_iadd(b, nir_imul(b, nir_iadd(b, pad_iy0, dual->pad_h0), dual->out_w),
                     nir_iadd(b, pad_ix0, dual->pad_w0)),
            n);
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            nir_def *di0 = nir_iadd(b, pad_row0, nir_iadd_imm(b, c_col0, 8 * t));
            nir_store_global(b, v0_arr[t],
               nir_iadd(b, dual->dst2_va, nir_u2u64(b, nir_ishl_imm(b, di0, 2))),
               .align_mul = 4, .write_mask = 3, .access = ACCESS_NON_READABLE);
         }
         if (n_m_tiles == 2) {
            nir_def *pad_iy1 = NULL, *pad_ix1 = NULL;
            hk_nir_fast_udivmod(b, &dual->in_w_div, c_row1, &pad_iy1, &pad_ix1);
            nir_def *pad_row1 = nir_imul(
               b,
               nir_iadd(b, nir_imul(b, nir_iadd(b, pad_iy1, dual->pad_h0), dual->out_w),
                        nir_iadd(b, pad_ix1, dual->pad_w0)),
               n);
            for (unsigned t = 0; t < n_n_tiles; ++t) {
               nir_def *di1 = nir_iadd(b, pad_row1, nir_iadd_imm(b, c_col0, 8 * t));
               nir_store_global(b, v1_arr[t],
                  nir_iadd(b, dual->dst2_va, nir_u2u64(b, nir_ishl_imm(b, di1, 2))),
                  .align_mul = 4, .write_mask = 3, .access = ACCESS_NON_READABLE);
            }
         }
      }
      nir_pop_if(b, spf);
   }
}

static void
emit_gemm_cmat_store_guarded_row(nir_builder *b, const struct hk_nir_gemm_output *info,
                                 const struct hk_nir_clamp_epilogue *epilogue,
                                 const struct hk_gemm_dual_ctx *dual,
                                 unsigned n_n_tiles,
                                 nir_def *m, nir_def *n, nir_def *bias_mode,
                                 nir_def *c_row, nir_def *c_col0, nir_def *acc[4])
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   const bool dyn_row = info->has_bias_buffer && info->has_bias_mode &&
                        (info->bias_mode_mask & (1u << 0));
   const bool dyn_col = info->has_bias_buffer && info->has_bias_mode &&
                        (info->bias_mode_mask & (1u << 1));
   const bool fixed_row = info->has_bias_buffer && !info->has_bias_mode &&
                          info->bias_mode_mask == (1u << 0);
   const bool fixed_col = info->has_bias_buffer && !info->has_bias_mode &&
                          info->bias_mode_mask == (1u << 1);

   nir_if *r_if = nir_push_if(b, nir_ult(b, c_row, m));
   {
      nir_def *zero_f = nir_imm_float(b, 0.0f);
      nir_def *is_row_bias = dyn_row ? nir_ieq_imm(b, bias_mode, 0) : NULL;
      nir_def *is_col_bias = dyn_col ? nir_ieq_imm(b, bias_mode, 1) : NULL;
      nir_def *rb = NULL;
      if (fixed_row) {
         rb = nir_load_global(b, 1, 32,
            gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                          c_row, read),
            .align_mul = 4, .access = read);
      } else if (dyn_row) {
         nir_def *rb_ld = NULL;
         nir_if *rb_if = nir_push_if(b, is_row_bias);
         {
            rb_ld = nir_load_global(b, 1, 32,
               gemm_f32_addr(b, info->reads[2], info->desc_comps, info->read_res_type[2],
                             c_row, read),
               .align_mul = 4, .access = read);
         }
         nir_push_else(b, rb_if);
         nir_pop_if(b, rb_if);
         rb = nir_if_phi(b, rb_ld, zero_f);
      }

      nir_def *pad_row = NULL;
      if (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_PAD4) {
         nir_def *iy = NULL, *ix = NULL;
         hk_nir_fast_udivmod(b, &dual->in_w_div, c_row, &iy, &ix);
         pad_row = nir_imul(
            b,
            nir_iadd(b, nir_imul(b, nir_iadd(b, iy, dual->pad_h0), dual->out_w),
                     nir_iadd(b, ix, dual->pad_w0)),
            n);
      }
      nir_def *c_idx0 = nir_iadd(b, nir_imul(b, c_row, n), c_col0);
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         for (unsigned lc = 0; lc < 2; ++lc) {
            const uint32_t off = 8 * t + lc;
            nir_def *col_k = nir_iadd_imm(b, c_col0, off);
            nir_if *c_if = nir_push_if(b, nir_ult(b, col_k, n));
            {
               nir_def *v = nir_channel(b, acc[t], lc);
               if (fixed_row) {
                  v = nir_fadd(b, v, rb);
               } else if (fixed_col) {
                  nir_def *cb = nir_load_global(b, 1, 32,
                     gemm_f32_addr(b, info->reads[2], info->desc_comps,
                                   info->read_res_type[2], col_k, read),
                     .align_mul = 4, .access = read);
                  v = nir_fadd(b, v, cb);
               } else if (dyn_row || dyn_col) {
                  if (dyn_row) {
                     nir_def *v_rb = NULL, *v_nc = NULL;
                     nir_if *m0_if = nir_push_if(b, is_row_bias);
                     {
                        v_rb = nir_fadd(b, v, rb);
                     }
                     nir_push_else(b, m0_if);
                     {
                        if (dyn_col) {
                           nir_def *v_cb = NULL;
                           nir_if *m1_if = nir_push_if(b, is_col_bias);
                           {
                              nir_def *cb = nir_load_global(b, 1, 32,
                                 gemm_f32_addr(b, info->reads[2], info->desc_comps,
                                               info->read_res_type[2], col_k, read),
                                 .align_mul = 4, .access = read);
                              v_cb = nir_fadd(b, v, cb);
                           }
                           nir_push_else(b, m1_if);
                           nir_pop_if(b, m1_if);
                           v_nc = nir_if_phi(b, v_cb, v);
                        } else {
                           v_nc = v;
                        }
                     }
                     nir_pop_if(b, m0_if);
                     v = nir_if_phi(b, v_rb, v_nc);
                  } else {
                     nir_def *v_cb = NULL;
                     nir_if *m1_if = nir_push_if(b, is_col_bias);
                     {
                        nir_def *cb = nir_load_global(b, 1, 32,
                           gemm_f32_addr(b, info->reads[2], info->desc_comps,
                                         info->read_res_type[2], col_k, read),
                           .align_mul = 4, .access = read);
                        v_cb = nir_fadd(b, v, cb);
                     }
                     nir_push_else(b, m1_if);
                     nir_pop_if(b, m1_if);
                     v = nir_if_phi(b, v_cb, v);
                  }
               }
               if (epilogue)
                  v = apply_clamp_epilogue(b, v, epilogue);
               const enum gl_access_qualifier out_acc =
                  (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_PAD4) ? 0 : ACCESS_NON_READABLE;
               nir_def *ci = nir_iadd_imm(b, c_idx0, off);
               nir_store_global(b, v,
                  gemm_f32_addr(b, info->output, info->desc_comps, info->out_res_type,
                                ci, out_acc),
                  .align_mul = 4, .write_mask = 1, .access = out_acc);
               if (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_COPY) {
                  nir_store_global(b, v,
                     nir_iadd(b, dual->dst2_va, nir_u2u64(b, nir_ishl_imm(b, ci, 2))),
                     .align_mul = 4, .write_mask = 1, .access = ACCESS_NON_READABLE);
               } else if (dual && dual->mode == HK_COMPUTE_PAIR_DUAL_PAD4) {
                  nir_if *spf = nir_push_if(b, dual->simple_pad);
                  {
                     nir_def *di = nir_iadd(b, pad_row, col_k);
                     nir_store_global(b, v,
                        nir_iadd(b, dual->dst2_va, nir_u2u64(b, nir_ishl_imm(b, di, 2))),
                        .align_mul = 4, .write_mask = 1, .access = ACCESS_NON_READABLE);
                  }
                  nir_pop_if(b, spf);
               }
            }
            nir_pop_if(b, c_if);
         }
      }
   }
   nir_pop_if(b, r_if);
}

nir_shader *
hk_nir_build_gemm_cmat(const nir_shader *producer,
                       const struct hk_nir_gemm_output *info,
                       uint32_t tile_n,
                       const struct hk_nir_clamp_epilogue *epilogue,
                       enum hk_compute_pair_dual_write dual_write)
{
   if (!producer || !info || !info->cmat_valid ||
       (tile_n != 16 && tile_n != 32 && tile_n != 64))
      return NULL;
   if (dual_write != HK_COMPUTE_PAIR_DUAL_NONE &&
       dual_write != HK_COMPUTE_PAIR_DUAL_COPY &&
       dual_write != HK_COMPUTE_PAIR_DUAL_PAD4)
      return NULL;
   if (epilogue &&
       (!hk_nir_clamp_epilogue_valid(epilogue) ||
        producer->info.float_controls_execution_mode !=
           epilogue->float_controls_execution_mode))
      return NULL;

   const uint32_t dual_off = hk_dual_write_push_offset(info->push_bytes);
   const uint32_t total_push_bytes =
      dual_write == HK_COMPUTE_PAIR_DUAL_COPY
         ? dual_off + sizeof(struct hk_dual_copy_push)
         : (dual_write == HK_COMPUTE_PAIR_DUAL_PAD4
               ? dual_off + sizeof(struct hk_dual_pad4_push)
               : info->push_bytes);
   if (total_push_bytes > 256u)
      return NULL;

   nir_shader *nir =
      nir_shader_create(NULL, MESA_SHADER_COMPUTE,
                        producer->options);
   if (!nir)
      return NULL;
   const char *dual_sfx =
      dual_write == HK_COMPUTE_PAIR_DUAL_COPY ? "_dcopy"
      : (dual_write == HK_COMPUTE_PAIR_DUAL_PAD4 ? "_dpad4" : "");
   nir->info.name = ralloc_asprintf(nir, "%s_gemm_cmat%u%s%s%s",
                                    producer->info.name ? producer->info.name : "cs",
                                    tile_n, info->is_guarded ? "" : "_ug",
                                    epilogue ? "_clamp" : "", dual_sfx);

   /* All three variants use 128 threads (4 subgroups) and 4 KiB XOR-swizzled
    * shared A while loading B directly from global memory:
    *   tile_n == 16: 32x16, K_step=32 (4x1 SGs, 1x2 acc tiles/SG)
    *   tile_n == 32: 64x32, K_step=16 (4x1 SGs, 2x4 acc tiles/SG)
    *   tile_n == 64: 32x64, K_step=32 (2x2 SGs, 2x4 acc tiles/SG) */
   const bool k32 = (tile_n != 32);
   const uint32_t tile_m = hk_gemm_cmat_tile_m(tile_n);
   const unsigned tile_m_shift = k32 ? 5 : 6;
   const unsigned tile_n_shift = (tile_n == 64) ? 6 : (tile_n == 32 ? 5 : 4);
   const unsigned k_step = k32 ? 32 : 16;
   const unsigned k_step_shift = k32 ? 5 : 4;
   const unsigned n_k_slices = k32 ? 4 : 2;
   const unsigned a_half_shift = k32 ? 4 : 5;
   const unsigned a_c4_mask = k32 ? 7 : 3;
   const unsigned a_c4_shift = k32 ? 3 : 2;
   const unsigned a_rd_row8_off = k32 ? 128 : 64;
   const unsigned n_m_tiles = (tile_n == 16) ? 1 : 2;
   const unsigned n_n_tiles = (tile_n == 16) ? 2 : 4;

   nir_variable *a_sh_var = nir_variable_create(
      nir, nir_var_mem_shared, glsl_array_type(glsl_float_type(), 1024, 4),
      "gemm_a_sh");
   nir->info.workgroup_size[0] = 128;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   nir->info.api_subgroup_size = 32;
   nir->info.min_subgroup_size = 32;
   nir->info.max_subgroup_size = 32;
   nir->info.num_ssbos = producer->info.num_ssbos;
   nir->info.float_controls_execution_mode = producer->info.float_controls_execution_mode;
   nir->num_uniforms = total_push_bytes;

   nir_function *fn = nir_function_create(nir, "main");
   fn->is_entrypoint = true;
   nir_function_impl *impl = nir_function_impl_create(fn);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   nir_def *wg_id = nir_load_workgroup_id(&b);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *sg = nir_load_subgroup_id(&b);
   nir_def *lane = nir_load_subgroup_invocation(&b);

   nir_def *n = dw_push(&b, info->n_offset_B, total_push_bytes);
   nir_def *k = dw_push(&b, info->k_offset_B, total_push_bytes);
   nir_def *bias_mode = (info->has_bias_buffer && info->has_bias_mode)
      ? dw_push(&b, info->bias_mode_offset_B, total_push_bytes)
      : NULL;
   nir_def *m = info->is_guarded
      ? dw_push(&b, info->m_offset_B, total_push_bytes)
      : NULL;

   nir_def *wg_col = nir_ishl_imm(&b, nir_channel(&b, wg_id, 0), tile_n_shift);
   nir_def *wg_row = nir_ishl_imm(&b, nir_channel(&b, wg_id, 1), tile_m_shift);

   nir_def *sg_row = (tile_n == 16)
      ? nir_ishl_imm(&b, sg, 3)
      : (tile_n == 32 ? nir_ishl_imm(&b, sg, 4)
                      : nir_ishl_imm(&b, nir_ushr_imm(&b, sg, 1), 4));
   nir_def *sg_col = (tile_n == 64)
      ? nir_ishl_imm(&b, nir_iand_imm(&b, sg, 1), 5)
      : nir_imm_int(&b, 0);

   nir_def *row_in_8 = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, lane, 1), 3),
                                   nir_iand_imm(&b, nir_ushr_imm(&b, lane, 2), 4));
   nir_def *col_in_8 = nir_ior(&b, nir_ishl_imm(&b, nir_iand_imm(&b, lane, 1), 1),
                                   nir_iand_imm(&b, nir_ushr_imm(&b, lane, 1), 4));

   nir_deref_instr *a_sh_base = nir_build_deref_var(&b, a_sh_var);
   nir_deref_instr *a_sh_f32 = nir_build_deref_cast_with_alignment(
      &b, &a_sh_base->def, nir_var_mem_shared,
      glsl_array_type(glsl_float_type(), 0, 4), 0, 4, 0);
   nir_deref_instr *a_sh_v2 = nir_build_deref_cast_with_alignment(
      &b, &a_sh_base->def, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_FLOAT, 2), 0, 8), 0, 8, 0);
   nir_deref_instr *a_sh_v4 = nir_build_deref_cast_with_alignment(
      &b, &a_sh_base->def, nir_var_mem_shared,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_FLOAT, 4), 0, 16), 0, 16, 0);

   /* Precompute loop-invariant u16 XOR-swizzled 4 KiB shared A indices:
    *   k_step=32 (32x32): sh_idx4 = (r << 3) | (c4 ^ (r & 7)), row+16 is +128
    *   k_step=16 (64x16): sh_idx4 = (r << 2) | (c4 ^ (r & 3)), row+32 is +128 */
   nir_def *a_c4_pre = nir_iand_imm(&b, lid, a_c4_mask);
   nir_def *a_r0_pre = nir_ushr_imm(&b, lid, a_c4_shift);
   nir_def *a_sh_idx4_0_16 = nir_u2u16(
      &b, nir_ior(&b, nir_ishl_imm(&b, a_r0_pre, a_c4_shift),
                      nir_ixor(&b, a_c4_pre, nir_iand_imm(&b, a_r0_pre, a_c4_mask))));

   nir_def *a_rd_r0 = nir_iadd(&b, sg_row, row_in_8);
   nir_def *a_rd_c4 = nir_ixor(&b, nir_ushr_imm(&b, col_in_8, 2),
                                   nir_iand_imm(&b, a_rd_r0, a_c4_mask));
   nir_def *a_rd_idx2_0_16 = nir_u2u16(
      &b, nir_ior(&b, nir_ior(&b, nir_ishl_imm(&b, a_rd_r0, a_c4_shift + 1),
                                  nir_ishl_imm(&b, a_rd_c4, 1)),
                      nir_iand_imm(&b, nir_ushr_imm(&b, col_in_8, 1), 1)));

   nir_def *k4 = NULL, *a_g_base4_0 = NULL, *a_row_half_stride4 = NULL;
   if (!info->is_guarded) {
      k4 = nir_ushr_imm(&b, k, 2);
      a_g_base4_0 = nir_iadd(&b, nir_imul(&b, nir_iadd(&b, wg_row, a_r0_pre), k4),
                                 a_c4_pre);
      a_row_half_stride4 = nir_ishl_imm(&b, k4, a_half_shift);
   }

   nir_def *b_col0 = nir_iadd(&b, nir_iadd(&b, wg_col, sg_col), col_in_8);
   nir_def *b_k0_base = nir_iadd(&b, nir_imul(&b, row_in_8, n), b_col0);
   nir_def *b_stride8 = nir_ishl_imm(&b, n, 3);
   nir_def *b_stride_kstep = nir_ishl_imm(&b, n, k_step_shift);

   nir_def *tile_m_full = NULL, *tile_n_full = NULL;
   if (info->is_guarded) {
      tile_m_full = nir_iand(&b, nir_uge_imm(&b, m, tile_m),
                                 nir_uge(&b, nir_iadd_imm(&b, m, -(int32_t)tile_m), wg_row));
      tile_n_full = nir_iand(&b, nir_uge_imm(&b, n, tile_n),
                                 nir_uge(&b, nir_iadd_imm(&b, n, -(int32_t)tile_n), wg_col));
   }

   nir_def *zero_i = nir_imm_int(&b, 0);
   nir_def *zero_f = nir_imm_float(&b, 0.0f);
   nir_def *zero_v2 = nir_vec2(&b, zero_f, zero_f);
   nir_def *zero_v4 = nir_vec4(&b, zero_f, zero_f, zero_f, zero_f);

   nir_loop *k_lp = nir_push_loop(&b);
   k_lp->control = nir_loop_control_dont_unroll;
   nir_block *k_pre = nir_cf_node_as_block(nir_cf_node_prev(&k_lp->cf_node));
   nir_block *k_hdr = nir_loop_first_block(k_lp);

   nir_phi_instr *kb_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&kb_phi->instr, &kb_phi->def, 1, 32);
   nir_phi_instr_add_src(kb_phi, k_pre, zero_i);

   nir_phi_instr *b_idx_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&b_idx_phi->instr, &b_idx_phi->def, 1, 32);
   nir_phi_instr_add_src(b_idx_phi, k_pre, b_k0_base);

   nir_phi_instr *a0_phi[4] = {0}, *a1_phi[4] = {0};
   for (unsigned t = 0; t < n_n_tiles; ++t) {
      a0_phi[t] = nir_phi_instr_create(b.shader);
      nir_def_init(&a0_phi[t]->instr, &a0_phi[t]->def, 2, 32);
      nir_phi_instr_add_src(a0_phi[t], k_pre, zero_v2);

      if (n_m_tiles == 2) {
         a1_phi[t] = nir_phi_instr_create(b.shader);
         nir_def_init(&a1_phi[t]->instr, &a1_phi[t]->def, 2, 32);
         nir_phi_instr_add_src(a1_phi[t], k_pre, zero_v2);
      }
   }

   nir_def *k_base = &kb_phi->def;
   nir_def *b_idx_cur = &b_idx_phi->def;
   nir_if *k_done = nir_push_if(&b, nir_ige(&b, k_base, k));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, k_done);

   nir_def *a_sh_idx4_0 = nir_u2u32(&b, a_sh_idx4_0_16);
   nir_def *k_full = NULL;

   /* Phase 1: Stage tile_m x k_step tile of A into 4 KiB XOR-swizzled shared
    * memory (128 threads x 2 vec4s = 256 vec4s = 1024 floats). */
   if (!info->is_guarded) {
      nir_def *kb4 = nir_ushr_imm(&b, k_base, 2);
      nir_def *a_idx4 = nir_iadd(&b, a_g_base4_0, kb4);
      nir_def *va0 = nir_load_global(
         &b, 4, 32,
         gemm_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                        a_idx4, read),
         .align_mul = 16, .access = read);
      nir_def *va1 = nir_load_global(
         &b, 4, 32,
         gemm_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                        nir_iadd(&b, a_idx4, a_row_half_stride4), read),
         .align_mul = 16, .access = read);
      nir_store_deref(&b, nir_build_deref_array(&b, a_sh_v4, a_sh_idx4_0), va0, 0xf);
      nir_store_deref(&b, nir_build_deref_array(&b, a_sh_v4,
                      nir_iadd_imm(&b, a_sh_idx4_0, 128)), va1, 0xf);
   } else {
      k_full = nir_iand(&b, nir_uge_imm(&b, k, k_step),
                            nir_uge(&b, nir_iadd_imm(&b, k, -(int32_t)k_step), k_base));
      nir_def *fast_a = nir_iand(&b, tile_m_full, k_full);
      nir_if *fa_if = nir_push_if(&b, fast_a);
      {
         /* Float-indexed addressing works for any K (including K % 4 != 0)
          * because every element of A is 4-byte aligned. Recompute row/col
          * from lid inside fast_a to keep K-loop live GPRs <= 103. */
         nir_def *lid_f = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
         nir_def *a_r0_f = nir_ushr_imm(&b, lid_f, a_c4_shift);
         nir_def *a_c0_f = nir_ishl_imm(&b, nir_iand_imm(&b, lid_f, a_c4_mask), 2);
         nir_def *a_idx = nir_iadd(&b, nir_imul(&b, nir_iadd(&b, wg_row, a_r0_f), k),
                                       nir_iadd(&b, k_base, a_c0_f));
         nir_def *va0 = nir_load_global(
            &b, 4, 32,
            gemm_f32_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                          a_idx, read),
            .align_mul = 4, .access = read);
         nir_def *va1 = nir_load_global(
            &b, 4, 32,
            gemm_f32_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0],
                          nir_iadd(&b, a_idx, nir_ishl_imm(&b, k, a_half_shift)), read),
            .align_mul = 4, .access = read);
         nir_store_deref(&b, nir_build_deref_array(&b, a_sh_v4, a_sh_idx4_0), va0, 0xf);
         nir_store_deref(&b, nir_build_deref_array(&b, a_sh_v4,
                         nir_iadd_imm(&b, a_sh_idx4_0, 128)), va1, 0xf);
      }
      nir_push_else(&b, fa_if);
      {
         nir_def *lid_s = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
         nir_def *a_r0 = nir_ushr_imm(&b, lid_s, a_c4_shift);
         nir_def *a_c0 = nir_ishl_imm(&b, nir_iand_imm(&b, lid_s, a_c4_mask), 2);
         nir_def *gk_base = nir_iadd(&b, k_base, a_c0);
         nir_def *k_vec4_ok = nir_iand(&b, nir_uge_imm(&b, k, 4),
                                           nir_uge(&b, nir_iadd_imm(&b, k, -4), gk_base));
         nir_def *k_any_ok = nir_ult(&b, gk_base, k);

         nir_loop *rr_lp = nir_push_loop(&b);
         rr_lp->control = nir_loop_control_dont_unroll;
         nir_block *rr_pre = nir_cf_node_as_block(nir_cf_node_prev(&rr_lp->cf_node));
         nir_block *rr_hdr = nir_loop_first_block(rr_lp);
         nir_phi_instr *rr_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&rr_phi->instr, &rr_phi->def, 1, 32);
         nir_phi_instr_add_src(rr_phi, rr_pre, zero_i);
         nir_def *rr_v = &rr_phi->def;
         nir_if *rr_done = nir_push_if(&b, nir_ige_imm(&b, rr_v, 2));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, rr_done);

         nir_def *ar = nir_iadd(&b, a_r0, nir_ishl_imm(&b, rr_v, a_half_shift));
         nir_def *gr = nir_iadd(&b, wg_row, ar);
         nir_def *r_ok = nir_ult(&b, gr, m);
         nir_def *sh_i = nir_iadd(&b, a_sh_idx4_0, nir_ishl_imm(&b, rr_v, 7));
         nir_if *v4_if = nir_push_if(&b, nir_iand(&b, r_ok, k_vec4_ok));
         {
            nir_def *g_idx = nir_iadd(&b, nir_imul(&b, gr, k), gk_base);
            nir_def *va_ld = nir_load_global(
               &b, 4, 32,
               gemm_f32_addr(&b, info->reads[0], info->desc_comps,
                             info->read_res_type[0], g_idx, read),
               .align_mul = 4, .access = read);
            nir_store_deref(&b, nir_build_deref_array(&b, a_sh_v4, sh_i), va_ld, 0xf);
         }
         nir_push_else(&b, v4_if);
         {
            nir_if *tail_if = nir_push_if(&b, nir_iand(&b, r_ok, k_any_ok));
            {
               nir_def *g_row_base = nir_iadd(&b, nir_imul(&b, gr, k), gk_base);
               nir_def *sh_base = nir_ishl_imm(&b, sh_i, 2);
               nir_loop *cc_lp = nir_push_loop(&b);
               cc_lp->control = nir_loop_control_dont_unroll;
               nir_block *cc_pre = nir_cf_node_as_block(nir_cf_node_prev(&cc_lp->cf_node));
               nir_block *cc_hdr = nir_loop_first_block(cc_lp);
               nir_phi_instr *cc_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&cc_phi->instr, &cc_phi->def, 1, 32);
               nir_phi_instr_add_src(cc_phi, cc_pre, zero_i);
               nir_if *cc_done = nir_push_if(&b, nir_ige_imm(&b, &cc_phi->def, 4));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, cc_done);

               nir_def *gk = nir_iadd(&b, gk_base, &cc_phi->def);
               nir_def *v_ld = NULL;
               nir_if *ok_if = nir_push_if(&b, nir_ult(&b, gk, k));
               {
                  v_ld = nir_load_global(&b, 1, 32,
                     gemm_f32_addr(&b, info->reads[0], info->desc_comps,
                                   info->read_res_type[0],
                                   nir_iadd(&b, g_row_base, &cc_phi->def), read),
                     .align_mul = 4, .access = read);
               }
               nir_push_else(&b, ok_if);
               nir_pop_if(&b, ok_if);
               nir_def *av = nir_if_phi(&b, v_ld, zero_f);
               nir_store_deref(&b, nir_build_deref_array(&b, a_sh_f32,
                               nir_iadd(&b, sh_base, &cc_phi->def)), av, 1);

               nir_block *cc_back = nir_loop_last_block(cc_lp);
               nir_phi_instr_add_src(cc_phi, cc_back, nir_iadd_imm(&b, &cc_phi->def, 1));
               nir_instr_insert(nir_before_block(cc_hdr), &cc_phi->instr);
               nir_pop_loop(&b, cc_lp);
            }
            nir_push_else(&b, tail_if);
            {
               nir_store_deref(&b, nir_build_deref_array(&b, a_sh_v4, sh_i), zero_v4, 0xf);
            }
            nir_pop_if(&b, tail_if);
         }
         nir_pop_if(&b, v4_if);

         nir_block *rr_back = nir_cursor_current_block(b.cursor);
         nir_phi_instr_add_src(rr_phi, rr_back, nir_iadd_imm(&b, rr_v, 1));
         nir_instr_insert(nir_before_block(rr_hdr), &rr_phi->instr);
         nir_pop_loop(&b, rr_lp);
      }
      nir_pop_if(&b, fa_if);
   }

   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_modes = nir_var_mem_shared,
               .memory_semantics = NIR_MEMORY_ACQ_REL);

   /* Phase 2: n_k_slices (2 or 4) K-slices of 8 via rolled dont_unroll loop
    * (keeps GPRs <= 63 for 1024t occupancy by preventing cross-slice B-load
    * hoisting). */
   nir_def *a_rd_idx2_0 = nir_u2u32(&b, a_rd_idx2_0_16);
   nir_def *next0[4] = {0}, *next1[4] = {0};

   if (!info->is_guarded) {
      nir_loop *ks_lp = nir_push_loop(&b);
      ks_lp->control = nir_loop_control_dont_unroll;
      nir_block *ks_pre = nir_cf_node_as_block(nir_cf_node_prev(&ks_lp->cf_node));
      nir_block *ks_hdr = nir_loop_first_block(ks_lp);

      nir_phi_instr *ks_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&ks_phi->instr, &ks_phi->def, 1, 32);
      nir_phi_instr_add_src(ks_phi, ks_pre, zero_i);

      nir_phi_instr *bks_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&bks_phi->instr, &bks_phi->def, 1, 32);
      nir_phi_instr_add_src(bks_phi, ks_pre, b_idx_cur);

      nir_phi_instr *s0_phi[4] = {0}, *s1_phi[4] = {0};
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         s0_phi[t] = nir_phi_instr_create(b.shader);
         nir_def_init(&s0_phi[t]->instr, &s0_phi[t]->def, 2, 32);
         nir_phi_instr_add_src(s0_phi[t], ks_pre, &a0_phi[t]->def);
         if (n_m_tiles == 2) {
            s1_phi[t] = nir_phi_instr_create(b.shader);
            nir_def_init(&s1_phi[t]->instr, &s1_phi[t]->def, 2, 32);
            nir_phi_instr_add_src(s1_phi[t], ks_pre, &a1_phi[t]->def);
         }
      }

      nir_def *ks_v = &ks_phi->def;
      nir_def *b_ks = &bks_phi->def;
      nir_if *ks_done = nir_push_if(&b, nir_ige_imm(&b, ks_v, n_k_slices));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, ks_done);

      nir_def *a_idx0 = nir_ixor(&b, a_rd_idx2_0, nir_ishl_imm(&b, ks_v, 2));
      nir_def *left0 = nir_load_deref(&b, nir_build_deref_array(&b, a_sh_v2, a_idx0));
      nir_def *left1 = (n_m_tiles == 2)
         ? nir_load_deref(&b, nir_build_deref_array(&b, a_sh_v2,
                                                    nir_iadd_imm(&b, a_idx0, a_rd_row8_off)))
         : NULL;
      nir_def *right_v[4] = {0};
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         nir_def *bi = t ? nir_iadd_imm(&b, b_ks, 8 * t) : b_ks;
         right_v[t] = nir_load_global(
            &b, 2, 32,
            gemm_f32_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                          bi, read),
            .align_mul = 4, .access = read);
      }
      nir_def *ft0[4] = {0}, *ft1[4] = {0};
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         ft0[t] = nir_cmat_muladd_agx(&b, left0, right_v[t], &s0_phi[t]->def,
            .src_type = nir_type_float32, .src_type2 = nir_type_float32);
         if (n_m_tiles == 2) {
            ft1[t] = nir_cmat_muladd_agx(&b, left1, right_v[t], &s1_phi[t]->def,
               .src_type = nir_type_float32, .src_type2 = nir_type_float32);
         }
      }

      nir_block *ks_back = nir_loop_last_block(ks_lp);
      nir_phi_instr_add_src(ks_phi, ks_back, nir_iadd_imm(&b, ks_v, 1));
      nir_phi_instr_add_src(bks_phi, ks_back, nir_iadd(&b, b_ks, b_stride8));
      nir_instr_insert(nir_before_block(ks_hdr), &ks_phi->instr);
      nir_instr_insert(nir_before_block(ks_hdr), &bks_phi->instr);
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         nir_phi_instr_add_src(s0_phi[t], ks_back, ft0[t]);
         nir_instr_insert(nir_before_block(ks_hdr), &s0_phi[t]->instr);
         if (n_m_tiles == 2) {
            nir_phi_instr_add_src(s1_phi[t], ks_back, ft1[t]);
            nir_instr_insert(nir_before_block(ks_hdr), &s1_phi[t]->instr);
         }
      }
      nir_pop_loop(&b, ks_lp);
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         next0[t] = &s0_phi[t]->def;
         if (n_m_tiles == 2)
            next1[t] = &s1_phi[t]->def;
      }
   } else {
      nir_def *b_full = nir_iand(&b, tile_n_full, k_full);
      nir_def *ft0_out[4] = {0}, *ft1_out[4] = {0};
      nir_def *ge0_out[4] = {0}, *ge1_out[4] = {0};
      nir_if *bf_if = nir_push_if(&b, b_full);
      {
         nir_loop *ks_lp = nir_push_loop(&b);
         ks_lp->control = nir_loop_control_dont_unroll;
         nir_block *ks_pre = nir_cf_node_as_block(nir_cf_node_prev(&ks_lp->cf_node));
         nir_block *ks_hdr = nir_loop_first_block(ks_lp);

         nir_phi_instr *ks_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&ks_phi->instr, &ks_phi->def, 1, 32);
         nir_phi_instr_add_src(ks_phi, ks_pre, zero_i);

         nir_phi_instr *bks_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&bks_phi->instr, &bks_phi->def, 1, 32);
         nir_phi_instr_add_src(bks_phi, ks_pre, b_idx_cur);

         nir_phi_instr *s0_phi[4] = {0}, *s1_phi[4] = {0};
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            s0_phi[t] = nir_phi_instr_create(b.shader);
            nir_def_init(&s0_phi[t]->instr, &s0_phi[t]->def, 2, 32);
            nir_phi_instr_add_src(s0_phi[t], ks_pre, &a0_phi[t]->def);
            if (n_m_tiles == 2) {
               s1_phi[t] = nir_phi_instr_create(b.shader);
               nir_def_init(&s1_phi[t]->instr, &s1_phi[t]->def, 2, 32);
               nir_phi_instr_add_src(s1_phi[t], ks_pre, &a1_phi[t]->def);
            }
         }

         nir_def *ks_v = &ks_phi->def;
         nir_def *b_ks = &bks_phi->def;
         nir_if *ks_done = nir_push_if(&b, nir_ige_imm(&b, ks_v, n_k_slices));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ks_done);

         nir_def *a_idx0 = nir_ixor(&b, a_rd_idx2_0, nir_ishl_imm(&b, ks_v, 2));
         nir_def *left0 = nir_load_deref(&b, nir_build_deref_array(&b, a_sh_v2, a_idx0));
         nir_def *left1 = (n_m_tiles == 2)
            ? nir_load_deref(&b, nir_build_deref_array(&b, a_sh_v2,
                                                       nir_iadd_imm(&b, a_idx0, a_rd_row8_off)))
            : NULL;
         nir_def *right_v[4] = {0};
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            nir_def *bi = t ? nir_iadd_imm(&b, b_ks, 8 * t) : b_ks;
            right_v[t] = nir_load_global(
               &b, 2, 32,
               gemm_f32_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                             bi, read),
               .align_mul = 4, .access = read);
         }
         nir_def *ft0[4] = {0}, *ft1[4] = {0};
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            ft0[t] = nir_cmat_muladd_agx(&b, left0, right_v[t], &s0_phi[t]->def,
               .src_type = nir_type_float32, .src_type2 = nir_type_float32);
            if (n_m_tiles == 2) {
               ft1[t] = nir_cmat_muladd_agx(&b, left1, right_v[t], &s1_phi[t]->def,
                  .src_type = nir_type_float32, .src_type2 = nir_type_float32);
            }
         }

         nir_block *ks_back = nir_loop_last_block(ks_lp);
         nir_phi_instr_add_src(ks_phi, ks_back, nir_iadd_imm(&b, ks_v, 1));
         nir_phi_instr_add_src(bks_phi, ks_back, nir_iadd(&b, b_ks, b_stride8));
         nir_instr_insert(nir_before_block(ks_hdr), &ks_phi->instr);
         nir_instr_insert(nir_before_block(ks_hdr), &bks_phi->instr);
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            nir_phi_instr_add_src(s0_phi[t], ks_back, ft0[t]);
            nir_instr_insert(nir_before_block(ks_hdr), &s0_phi[t]->instr);
            if (n_m_tiles == 2) {
               nir_phi_instr_add_src(s1_phi[t], ks_back, ft1[t]);
               nir_instr_insert(nir_before_block(ks_hdr), &s1_phi[t]->instr);
            }
         }
         nir_pop_loop(&b, ks_lp);
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            ft0_out[t] = &s0_phi[t]->def;
            if (n_m_tiles == 2)
               ft1_out[t] = &s1_phi[t]->def;
         }
      }
      nir_push_else(&b, bf_if);
      {
         nir_def *lane_e = nir_load_subgroup_invocation(&b);
         nir_def *sg_e = nir_load_subgroup_id(&b);
         nir_def *sg_col_e = (tile_n == 64)
            ? nir_ishl_imm(&b, nir_iand_imm(&b, sg_e, 1), 5)
            : zero_i;
         nir_def *row8_e = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, lane_e, 1), 3),
                                       nir_iand_imm(&b, nir_ushr_imm(&b, lane_e, 2), 4));
         nir_def *col8_e = nir_ior(&b, nir_ishl_imm(&b, nir_iand_imm(&b, lane_e, 1), 1),
                                       nir_iand_imm(&b, nir_ushr_imm(&b, lane_e, 1), 4));
         nir_def *b_col_e = nir_iadd(&b, nir_iadd(&b, wg_col, sg_col_e), col8_e);
         nir_def *k_ge8 = nir_uge_imm(&b, k, 8);
         nir_def *k_m8 = nir_iadd_imm(&b, k, -8);

         nir_loop *ks_lp = nir_push_loop(&b);
         ks_lp->control = nir_loop_control_dont_unroll;
         nir_block *ks_pre = nir_cf_node_as_block(nir_cf_node_prev(&ks_lp->cf_node));
         nir_block *ks_hdr = nir_loop_first_block(ks_lp);

         nir_phi_instr *ks_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&ks_phi->instr, &ks_phi->def, 1, 32);
         nir_phi_instr_add_src(ks_phi, ks_pre, zero_i);

         nir_phi_instr *bks_phi = nir_phi_instr_create(b.shader);
         nir_def_init(&bks_phi->instr, &bks_phi->def, 1, 32);
         nir_phi_instr_add_src(bks_phi, ks_pre, b_idx_cur);

         nir_phi_instr *s0_phi[4] = {0}, *s1_phi[4] = {0};
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            s0_phi[t] = nir_phi_instr_create(b.shader);
            nir_def_init(&s0_phi[t]->instr, &s0_phi[t]->def, 2, 32);
            nir_phi_instr_add_src(s0_phi[t], ks_pre, &a0_phi[t]->def);
            if (n_m_tiles == 2) {
               s1_phi[t] = nir_phi_instr_create(b.shader);
               nir_def_init(&s1_phi[t]->instr, &s1_phi[t]->def, 2, 32);
               nir_phi_instr_add_src(s1_phi[t], ks_pre, &a1_phi[t]->def);
            }
         }

         nir_def *ks_v = &ks_phi->def;
         nir_def *b_ks = &bks_phi->def;
         nir_def *ks_k = nir_iadd(&b, k_base, nir_ishl_imm(&b, ks_v, 3));
         nir_if *ks_done = nir_push_if(&b, nir_ior(&b, nir_ige_imm(&b, ks_v, n_k_slices),
                                                       nir_ige(&b, ks_k, k)));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ks_done);

         nir_def *a_idx0 = nir_ixor(&b, a_rd_idx2_0, nir_ishl_imm(&b, ks_v, 2));
         nir_def *left0 = nir_load_deref(&b, nir_build_deref_array(&b, a_sh_v2, a_idx0));
         nir_def *left1 = (n_m_tiles == 2)
            ? nir_load_deref(&b, nir_build_deref_array(&b, a_sh_v2,
                                                       nir_iadd_imm(&b, a_idx0, a_rd_row8_off)))
            : NULL;

         /* Fast per-slice path when all 8 K-rows and all N columns of this
          * slice are in-bounds. */
         nir_def *slice_fast = nir_iand(&b, tile_n_full,
                                            nir_iand(&b, k_ge8, nir_uge(&b, k_m8, ks_k)));
         nir_def *sf0[4] = {0}, *sf1[4] = {0};
         nir_def *sg0[4] = {0}, *sg1[4] = {0};
         nir_if *sf_if = nir_push_if(&b, slice_fast);
         {
            nir_def *right_v[4] = {0};
            for (unsigned t = 0; t < n_n_tiles; ++t) {
               nir_def *bi = t ? nir_iadd_imm(&b, b_ks, 8 * t) : b_ks;
               right_v[t] = nir_load_global(
                  &b, 2, 32,
                  gemm_f32_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                                bi, read),
                  .align_mul = 4, .access = read);
            }
            for (unsigned t = 0; t < n_n_tiles; ++t) {
               sf0[t] = nir_cmat_muladd_agx(&b, left0, right_v[t], &s0_phi[t]->def,
                  .src_type = nir_type_float32, .src_type2 = nir_type_float32);
               if (n_m_tiles == 2) {
                  sf1[t] = nir_cmat_muladd_agx(&b, left1, right_v[t], &s1_phi[t]->def,
                     .src_type = nir_type_float32, .src_type2 = nir_type_float32);
               }
            }
         }
         nir_push_else(&b, sf_if);
         {
            nir_def *bk_row = nir_iadd(&b, ks_k, row8_e);
            nir_def *row_ok = nir_ult(&b, bk_row, k);
            for (unsigned t = 0; t < n_n_tiles; ++t) {
               nir_def *bc = t ? nir_iadd_imm(&b, b_col_e, 8 * t) : b_col_e;
               nir_def *bi = t ? nir_iadd_imm(&b, b_ks, 8 * t) : b_ks;
               nir_def *v2_ld = NULL, *v1_ld = NULL;
               nir_if *pair_ok = nir_push_if(&b, nir_iand(&b, row_ok, nir_ult(&b, nir_iadd_imm(&b, bc, 1), n)));
               {
                  v2_ld = nir_load_global(
                     &b, 2, 32,
                     gemm_f32_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                                   bi, read),
                     .align_mul = 4, .access = read);
               }
               nir_push_else(&b, pair_ok);
               {
                  nir_def *e0 = NULL;
                  nir_if *c0_ok = nir_push_if(&b, nir_iand(&b, row_ok, nir_ult(&b, bc, n)));
                  {
                     e0 = nir_load_global(
                        &b, 1, 32,
                        gemm_f32_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1],
                                      bi, read),
                        .align_mul = 4, .access = read);
                  }
                  nir_push_else(&b, c0_ok);
                  nir_pop_if(&b, c0_ok);
                  nir_def *e0_v = nir_if_phi(&b, e0, zero_f);
                  v1_ld = nir_vec2(&b, e0_v, zero_f);
               }
               nir_pop_if(&b, pair_ok);
               nir_def *right = nir_if_phi(&b, v2_ld, v1_ld);
               sg0[t] = nir_cmat_muladd_agx(&b, left0, right, &s0_phi[t]->def,
                  .src_type = nir_type_float32, .src_type2 = nir_type_float32);
               if (n_m_tiles == 2) {
                  sg1[t] = nir_cmat_muladd_agx(&b, left1, right, &s1_phi[t]->def,
                     .src_type = nir_type_float32, .src_type2 = nir_type_float32);
               }
            }
         }
         nir_pop_if(&b, sf_if);

         nir_def *ge0[4] = {0}, *ge1[4] = {0};
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            ge0[t] = nir_if_phi(&b, sf0[t], sg0[t]);
            if (n_m_tiles == 2)
               ge1[t] = nir_if_phi(&b, sf1[t], sg1[t]);
         }

         nir_block *ks_back = nir_loop_last_block(ks_lp);
         nir_phi_instr_add_src(ks_phi, ks_back, nir_iadd_imm(&b, ks_v, 1));
         nir_phi_instr_add_src(bks_phi, ks_back, nir_iadd(&b, b_ks, b_stride8));
         nir_instr_insert(nir_before_block(ks_hdr), &ks_phi->instr);
         nir_instr_insert(nir_before_block(ks_hdr), &bks_phi->instr);
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            nir_phi_instr_add_src(s0_phi[t], ks_back, ge0[t]);
            nir_instr_insert(nir_before_block(ks_hdr), &s0_phi[t]->instr);
            if (n_m_tiles == 2) {
               nir_phi_instr_add_src(s1_phi[t], ks_back, ge1[t]);
               nir_instr_insert(nir_before_block(ks_hdr), &s1_phi[t]->instr);
            }
         }
         nir_pop_loop(&b, ks_lp);
         for (unsigned t = 0; t < n_n_tiles; ++t) {
            ge0_out[t] = &s0_phi[t]->def;
            if (n_m_tiles == 2)
               ge1_out[t] = &s1_phi[t]->def;
         }
      }
      nir_pop_if(&b, bf_if);
      for (unsigned t = 0; t < n_n_tiles; ++t) {
         next0[t] = nir_if_phi(&b, ft0_out[t], ge0_out[t]);
         if (n_m_tiles == 2)
            next1[t] = nir_if_phi(&b, ft1_out[t], ge1_out[t]);
      }
   }

   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_modes = nir_var_mem_shared,
               .memory_semantics = NIR_MEMORY_ACQ_REL);

   nir_block *k_back = nir_loop_last_block(k_lp);
   nir_phi_instr_add_src(kb_phi, k_back, nir_iadd_imm(&b, k_base, k_step));
   nir_phi_instr_add_src(b_idx_phi, k_back, nir_iadd(&b, b_idx_cur, b_stride_kstep));
   nir_instr_insert(nir_before_block(k_hdr), &kb_phi->instr);
   nir_instr_insert(nir_before_block(k_hdr), &b_idx_phi->instr);
   for (unsigned t = 0; t < n_n_tiles; ++t) {
      nir_phi_instr_add_src(a0_phi[t], k_back, next0[t]);
      nir_instr_insert(nir_before_block(k_hdr), &a0_phi[t]->instr);
      if (n_m_tiles == 2) {
         nir_phi_instr_add_src(a1_phi[t], k_back, next1[t]);
         nir_instr_insert(nir_before_block(k_hdr), &a1_phi[t]->instr);
      }
   }
   nir_pop_loop(&b, k_lp);

   /* Phase 3: Direct-C writeback + optional bias + optional clamp epilogue. */
   nir_def *acc0[4] = {0}, *acc1[4] = {0};
   for (unsigned t = 0; t < n_n_tiles; ++t) {
      acc0[t] = &a0_phi[t]->def;
      if (n_m_tiles == 2)
         acc1[t] = &a1_phi[t]->def;
   }
   nir_def *sg_ep = nir_load_subgroup_id(&b);
   nir_def *lane_ep = nir_load_subgroup_invocation(&b);
   nir_def *sg_row_ep = (tile_n == 16)
      ? nir_ishl_imm(&b, sg_ep, 3)
      : (tile_n == 32 ? nir_ishl_imm(&b, sg_ep, 4)
                      : nir_ishl_imm(&b, nir_ushr_imm(&b, sg_ep, 1), 4));
   nir_def *sg_col_ep = (tile_n == 64)
      ? nir_ishl_imm(&b, nir_iand_imm(&b, sg_ep, 1), 5)
      : nir_imm_int(&b, 0);
   nir_def *row8_ep = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, lane_ep, 1), 3),
                                  nir_iand_imm(&b, nir_ushr_imm(&b, lane_ep, 2), 4));
   nir_def *col8_ep = nir_ior(&b, nir_ishl_imm(&b, nir_iand_imm(&b, lane_ep, 1), 1),
                                  nir_iand_imm(&b, nir_ushr_imm(&b, lane_ep, 1), 4));
   nir_def *c_row0 = nir_iadd(&b, nir_iadd(&b, wg_row, sg_row_ep), row8_ep);
   nir_def *c_col0 = nir_iadd(&b, nir_iadd(&b, wg_col, sg_col_ep), col8_ep);
   nir_def *c_row1 = (n_m_tiles == 2) ? nir_iadd_imm(&b, c_row0, 8) : NULL;

   struct hk_gemm_dual_ctx dual = {.mode = dual_write};
   const struct hk_gemm_dual_ctx *dual_ptr = NULL;
   if (dual_write == HK_COMPUTE_PAIR_DUAL_COPY) {
      dual.dst2_va = dual_push_u64(
         &b, dual_off + offsetof(struct hk_dual_copy_push, dst2_va_lo), total_push_bytes);
      dual_ptr = &dual;
   } else if (dual_write == HK_COMPUTE_PAIR_DUAL_PAD4) {
      dual.dst2_va = dual_push_u64(
         &b, dual_off + offsetof(struct hk_dual_pad4_push, dst2_va_lo), total_push_bytes);
      nir_def *pad_va = dual_push_u64(
         &b, dual_off + offsetof(struct hk_dual_pad4_push, pad_va_lo), total_push_bytes);
      dual.in_h = dw_push(&b, dual_off + offsetof(struct hk_dual_pad4_push, in_h), total_push_bytes);
      dual.in_w = dw_push(&b, dual_off + offsetof(struct hk_dual_pad4_push, in_w), total_push_bytes);
      dual.in_w_div = hk_nir_fast_udiv_prepare(&b, dual.in_w);
      dual.out_h = dw_push(&b, dual_off + offsetof(struct hk_dual_pad4_push, out_h), total_push_bytes);
      dual.out_w = dw_push(&b, dual_off + offsetof(struct hk_dual_pad4_push, out_w), total_push_bytes);
      nir_def *pidx[4];
      nir_def *pval[4];
      for (unsigned d = 0; d < 4; ++d) {
         pidx[d] = dw_push(&b, dual_off + offsetof(struct hk_dual_pad4_push, pad_idx) + 4 * d,
                           total_push_bytes);
         pval[d] = nir_load_global(
            &b, 1, 32, nir_iadd(&b, pad_va, nir_u2u64(&b, nir_ishl_imm(&b, pidx[d], 2))),
            .align_mul = 4, .access = read);
      }
      dual.pad_n0 = pval[0];
      dual.pad_h0 = pval[1];
      dual.pad_w0 = pval[2];
      dual.pad_c0 = pval[3];
      nir_def *nc_ok =
         nir_iand(&b, nir_ieq_imm(&b, dual.pad_n0, 0), nir_ieq_imm(&b, dual.pad_c0, 0));
      nir_def *h_ok = nir_iand(
         &b, nir_ige_imm(&b, dual.pad_h0, 0),
         nir_iand(&b, nir_uge(&b, dual.out_h, dual.in_h),
                  nir_uge(&b, nir_isub(&b, dual.out_h, dual.in_h), dual.pad_h0)));
      nir_def *w_ok = nir_iand(
         &b, nir_ige_imm(&b, dual.pad_w0, 0),
         nir_iand(&b, nir_uge(&b, dual.out_w, dual.in_w),
                  nir_uge(&b, nir_isub(&b, dual.out_w, dual.in_w), dual.pad_w0)));
      dual.simple_pad = nir_iand(&b, nc_ok, nir_iand(&b, h_ok, w_ok));
      dual_ptr = &dual;
   }

   if (!info->is_guarded) {
      emit_gemm_cmat_store_fast(&b, info, epilogue, dual_ptr, n_m_tiles, n_n_tiles,
                                n, bias_mode, c_row0, c_row1, c_col0, acc0, acc1);
   } else {
      nir_if *cf_if = nir_push_if(&b, nir_iand(&b, tile_m_full, tile_n_full));
      {
         emit_gemm_cmat_store_fast(&b, info, epilogue, dual_ptr, n_m_tiles, n_n_tiles,
                                   n, bias_mode, c_row0, c_row1, c_col0, acc0, acc1);
      }
      nir_push_else(&b, cf_if);
      {
         emit_gemm_cmat_store_guarded_row(&b, info, epilogue, dual_ptr, n_n_tiles,
                                          m, n, bias_mode, c_row0, c_col0, acc0);
         if (n_m_tiles == 2) {
            emit_gemm_cmat_store_guarded_row(&b, info, epilogue, dual_ptr, n_n_tiles,
                                             m, n, bias_mode, c_row1, c_col0, acc1);
         }
      }
      nir_pop_if(&b, cf_if);
   }

   if (dual_write == HK_COMPUTE_PAIR_DUAL_PAD4) {
      /* Emit PAD4 border zeroing (simple_pad) and rare general scatter fallback
       * ONCE after all accumulators are dead, avoiding per-tile code duplication
       * and register-pressure bloat across the store loops. Recompute output
       * coordinates here so they stay dead across the store loops. */
      nir_def *wg_p = nir_load_workgroup_id(&b);
      nir_def *sg_p = nir_load_subgroup_id(&b);
      nir_def *lane_p = nir_load_subgroup_invocation(&b);
      nir_def *wg_row_p = nir_ishl_imm(&b, nir_channel(&b, wg_p, 1), tile_m_shift);
      nir_def *wg_col_p = nir_ishl_imm(&b, nir_channel(&b, wg_p, 0), tile_n_shift);
      nir_def *sg_row_p = (tile_n == 16)
         ? nir_ishl_imm(&b, sg_p, 3)
         : (tile_n == 32 ? nir_ishl_imm(&b, sg_p, 4)
                         : nir_ishl_imm(&b, nir_ushr_imm(&b, sg_p, 1), 4));
      nir_def *sg_col_p = (tile_n == 64)
         ? nir_ishl_imm(&b, nir_iand_imm(&b, sg_p, 1), 5)
         : nir_imm_int(&b, 0);
      nir_def *row8_p = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, lane_p, 1), 3),
                                    nir_iand_imm(&b, nir_ushr_imm(&b, lane_p, 2), 4));
      nir_def *col8_p = nir_ior(&b, nir_ishl_imm(&b, nir_iand_imm(&b, lane_p, 1), 1),
                                    nir_iand_imm(&b, nir_ushr_imm(&b, lane_p, 1), 4));
      c_row0 = nir_iadd(&b, nir_iadd(&b, wg_row_p, sg_row_p), row8_p);
      c_col0 = nir_iadd(&b, nir_iadd(&b, wg_col_p, sg_col_p), col8_p);
      c_row1 = (n_m_tiles == 2) ? nir_iadd_imm(&b, c_row0, 8) : NULL;

      nir_if *pad_if = nir_push_if(&b, dual.simple_pad);
      {
         nir_loop *mi_lp = NULL;
         nir_block *mi_hdr = NULL;
         nir_phi_instr *mi_phi = NULL;
         nir_def *c_row_cur = c_row0;
         if (n_m_tiles == 2) {
            mi_lp = nir_push_loop(&b);
            mi_lp->control = nir_loop_control_dont_unroll;
            nir_block *mi_pre = nir_cf_node_as_block(nir_cf_node_prev(&mi_lp->cf_node));
            mi_hdr = nir_loop_first_block(mi_lp);
            mi_phi = nir_phi_instr_create(b.shader);
            nir_def_init(&mi_phi->instr, &mi_phi->def, 1, 32);
            nir_phi_instr_add_src(mi_phi, mi_pre, zero_i);
            nir_if *mi_done = nir_push_if(&b, nir_ige_imm(&b, &mi_phi->def, 2));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, mi_done);
            c_row_cur = nir_bcsel(&b, nir_ieq_imm(&b, &mi_phi->def, 0), c_row0, c_row1);
         }

         nir_if *mr_if = info->is_guarded ? nir_push_if(&b, nir_ult(&b, c_row_cur, m)) : NULL;
         {
            nir_def *iy = NULL, *ix = NULL;
            hk_nir_fast_udivmod(&b, &dual.in_w_div, c_row_cur, &iy, &ix);
            nir_def *on_edge = nir_ior(
               &b,
               nir_ior(&b, nir_ieq_imm(&b, ix, 0),
                       nir_ieq(&b, ix, nir_iadd_imm(&b, dual.in_w, -1))),
               nir_ior(&b, nir_ieq_imm(&b, iy, 0),
                       nir_ieq(&b, iy, nir_iadd_imm(&b, dual.in_h, -1))));
            nir_if *eb_if = nir_push_if(&b, on_edge);
            {
               if (!info->is_guarded) {
                  nir_loop *tlp = nir_push_loop(&b);
                  tlp->control = nir_loop_control_dont_unroll;
                  nir_block *tpre = nir_cf_node_as_block(nir_cf_node_prev(&tlp->cf_node));
                  nir_block *thdr = nir_loop_first_block(tlp);
                  nir_phi_instr *tphi = nir_phi_instr_create(b.shader);
                  nir_def_init(&tphi->instr, &tphi->def, 1, 32);
                  nir_phi_instr_add_src(tphi, tpre, zero_i);
                  nir_if *tdone = nir_push_if(&b, nir_ige_imm(&b, &tphi->def, n_n_tiles));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, tdone);

                  nir_def *col_t = nir_iadd(&b, c_col0, nir_ishl_imm(&b, &tphi->def, 3));
                  emit_pad4_simple_border(&b, &dual, iy, ix, n, col_t, 2);

                  nir_block *tback = nir_cursor_current_block(b.cursor);
                  nir_phi_instr_add_src(tphi, tback, nir_iadd_imm(&b, &tphi->def, 1));
                  nir_instr_insert(nir_before_block(thdr), &tphi->instr);
                  nir_pop_loop(&b, tlp);
               } else {
                  nir_loop *slp = nir_push_loop(&b);
                  slp->control = nir_loop_control_dont_unroll;
                  nir_block *spre = nir_cf_node_as_block(nir_cf_node_prev(&slp->cf_node));
                  nir_block *shdr = nir_loop_first_block(slp);
                  nir_phi_instr *sphi = nir_phi_instr_create(b.shader);
                  nir_def_init(&sphi->instr, &sphi->def, 1, 32);
                  nir_phi_instr_add_src(sphi, spre, zero_i);
                  nir_if *sdone = nir_push_if(&b, nir_ige_imm(&b, &sphi->def, 2 * n_n_tiles));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, sdone);

                  nir_def *col_s = nir_iadd(
                     &b, c_col0,
                     nir_iadd(&b, nir_ishl_imm(&b, nir_ushr_imm(&b, &sphi->def, 1), 3),
                              nir_iand_imm(&b, &sphi->def, 1)));
                  nir_if *cn_if = nir_push_if(&b, nir_ult(&b, col_s, n));
                  emit_pad4_simple_border(&b, &dual, iy, ix, n, col_s, 1);
                  nir_pop_if(&b, cn_if);

                  nir_block *sback = nir_cursor_current_block(b.cursor);
                  nir_phi_instr_add_src(sphi, sback, nir_iadd_imm(&b, &sphi->def, 1));
                  nir_instr_insert(nir_before_block(shdr), &sphi->instr);
                  nir_pop_loop(&b, slp);
               }
            }
            nir_pop_if(&b, eb_if);
         }
         if (mr_if)
            nir_pop_if(&b, mr_if);

         if (mi_lp) {
            nir_block *mi_back = nir_cursor_current_block(b.cursor);
            nir_phi_instr_add_src(mi_phi, mi_back, nir_iadd_imm(&b, &mi_phi->def, 1));
            nir_instr_insert(nir_before_block(mi_hdr), &mi_phi->instr);
            nir_pop_loop(&b, mi_lp);
         }
      }
      nir_push_else(&b, pad_if);
      {
         /* Rare fallback when pad_b is negative, crops, or shifts N/C: read
          * back this invocation's own just-written output elements from out1
          * (same-invocation program-order RAW, no barrier or extra shmem) and
          * zero every output element outside the input box via a disjoint
          * grid-stride sweep (no cross-WG synchronization needed). */
         nir_loop *mi_lp = NULL;
         nir_block *mi_hdr = NULL;
         nir_phi_instr *mi_phi = NULL;
         nir_def *c_row_cur = c_row0;
         if (n_m_tiles == 2) {
            mi_lp = nir_push_loop(&b);
            mi_lp->control = nir_loop_control_dont_unroll;
            nir_block *mi_pre = nir_cf_node_as_block(nir_cf_node_prev(&mi_lp->cf_node));
            mi_hdr = nir_loop_first_block(mi_lp);
            mi_phi = nir_phi_instr_create(b.shader);
            nir_def_init(&mi_phi->instr, &mi_phi->def, 1, 32);
            nir_phi_instr_add_src(mi_phi, mi_pre, zero_i);
            nir_if *mi_done = nir_push_if(&b, nir_ige_imm(&b, &mi_phi->def, 2));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, mi_done);
            c_row_cur = nir_bcsel(&b, nir_ieq_imm(&b, &mi_phi->def, 0), c_row0, c_row1);
         }
         nir_if *mr_if = info->is_guarded ? nir_push_if(&b, nir_ult(&b, c_row_cur, m)) : NULL;
         {
            nir_def *iy = NULL, *ix = NULL;
            hk_nir_fast_udivmod(&b, &dual.in_w_div, c_row_cur, &iy, &ix);
            nir_def *row_base = nir_imul(&b, c_row_cur, n);

            nir_loop *slp = nir_push_loop(&b);
            slp->control = nir_loop_control_dont_unroll;
            nir_block *spre = nir_cf_node_as_block(nir_cf_node_prev(&slp->cf_node));
            nir_block *shdr = nir_loop_first_block(slp);
            nir_phi_instr *sphi = nir_phi_instr_create(b.shader);
            nir_def_init(&sphi->instr, &sphi->def, 1, 32);
            nir_phi_instr_add_src(sphi, spre, zero_i);
            nir_if *sdone = nir_push_if(&b, nir_ige_imm(&b, &sphi->def, 2 * n_n_tiles));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, sdone);

            nir_def *col_s = nir_iadd(
               &b, c_col0,
               nir_iadd(&b, nir_ishl_imm(&b, nir_ushr_imm(&b, &sphi->def, 1), 3),
                        nir_iand_imm(&b, &sphi->def, 1)));
            nir_if *cn_if = info->is_guarded ? nir_push_if(&b, nir_ult(&b, col_s, n)) : NULL;
            {
               nir_def *v = nir_load_global(
                  &b, 1, 32,
                  gemm_f32_addr(&b, info->output, info->desc_comps, info->out_res_type,
                                nir_iadd(&b, row_base, col_s), 0),
                  .align_mul = 4);
               emit_pad4_general_scatter_scalar(&b, &dual, iy, ix, n, col_s, v);
            }
            if (cn_if)
               nir_pop_if(&b, cn_if);

            nir_block *sback = nir_cursor_current_block(b.cursor);
            nir_phi_instr_add_src(sphi, sback, nir_iadd_imm(&b, &sphi->def, 1));
            nir_instr_insert(nir_before_block(shdr), &sphi->instr);
            nir_pop_loop(&b, slp);
         }
         if (mr_if)
            nir_pop_if(&b, mr_if);

         if (mi_lp) {
            nir_block *mi_back = nir_cursor_current_block(b.cursor);
            nir_phi_instr_add_src(mi_phi, mi_back, nir_iadd_imm(&b, &mi_phi->def, 1));
            nir_instr_insert(nir_before_block(mi_hdr), &mi_phi->instr);
            nir_pop_loop(&b, mi_lp);
         }

         nir_def *m_eff = info->is_guarded ? m : nir_imul(&b, dual.in_h, dual.in_w);
         nir_def *gx = nir_ushr_imm(&b, nir_iadd_imm(&b, n, (int32_t)tile_n - 1), tile_n_shift);
         nir_def *gy = nir_ushr_imm(&b, nir_iadd_imm(&b, m_eff, (int32_t)tile_m - 1), tile_m_shift);
         nir_def *wg_id_fb = nir_load_workgroup_id(&b);
         nir_def *lid_fb = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
         nir_def *wg_lin = nir_iadd(&b, nir_imul(&b, nir_channel(&b, wg_id_fb, 1), gx),
                                    nir_channel(&b, wg_id_fb, 0));
         nir_def *tid0 = nir_iadd(&b, nir_ishl_imm(&b, wg_lin, 7), lid_fb);
         nir_def *stride = nir_ishl_imm(&b, nir_imul(&b, gx, gy), 7);
         nir_def *out_total = nir_imul(&b, nir_imul(&b, dual.out_h, dual.out_w), n);
         nir_def *zero_fb = nir_imm_float(&b, 0.0f);

         nir_loop *zlp = nir_push_loop(&b);
         zlp->control = nir_loop_control_dont_unroll;
         nir_block *zpre = nir_cf_node_as_block(nir_cf_node_prev(&zlp->cf_node));
         nir_block *zhdr = nir_loop_first_block(zlp);
         nir_phi_instr *zphi = nir_phi_instr_create(b.shader);
         nir_def_init(&zphi->instr, &zphi->def, 1, 32);
         nir_phi_instr_add_src(zphi, zpre, tid0);
         nir_def *idx = &zphi->def;

         nir_if *zbrk = nir_push_if(&b, nir_uge(&b, idx, out_total));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, zbrk);

         nir_def *oc = nir_umod(&b, idx, n);
         nir_def *pix = nir_udiv(&b, idx, n);
         nir_def *ox = nir_umod(&b, pix, dual.out_w);
         nir_def *oy = nir_udiv(&b, pix, dual.out_w);

         nir_def *iy64 = nir_isub(&b, nir_u2u64(&b, oy), nir_i2i64(&b, dual.pad_h0));
         nir_def *ix64 = nir_isub(&b, nir_u2u64(&b, ox), nir_i2i64(&b, dual.pad_w0));
         nir_def *ic64 = nir_isub(&b, nir_u2u64(&b, oc), nir_i2i64(&b, dual.pad_c0));
         nir_def *in_box = nir_iand(&b,
            nir_ieq_imm(&b, dual.pad_n0, 0),
            nir_iand(&b,
               nir_iand(&b, nir_ige_imm(&b, iy64, 0), nir_ilt(&b, iy64, nir_u2u64(&b, dual.in_h))),
               nir_iand(&b,
                  nir_iand(&b, nir_ige_imm(&b, ix64, 0), nir_ilt(&b, ix64, nir_u2u64(&b, dual.in_w))),
                  nir_iand(&b, nir_ige_imm(&b, ic64, 0), nir_ilt(&b, ic64, nir_u2u64(&b, n))))));

         nir_if *zf = nir_push_if(&b, nir_inot(&b, in_box));
         {
            nir_store_global(&b, zero_fb,
               nir_iadd(&b, dual.dst2_va, nir_u2u64(&b, nir_ishl_imm(&b, idx, 2))),
               .align_mul = 4, .write_mask = 1, .access = ACCESS_NON_READABLE);
         }
         nir_pop_if(&b, zf);

         nir_def *idx_next = nir_iadd(&b, idx, stride);
         nir_block *zback = nir_cursor_current_block(b.cursor);
         nir_phi_instr_add_src(zphi, zback, idx_next);
         nir_instr_insert(nir_before_block(zhdr), &zphi->instr);
         nir_pop_loop(&b, zlp);
      }
      nir_pop_if(&b, pad_if);
   }

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   nir_validate_shader(nir, "hk_nir_build_gemm_cmat");
   return nir;
}
