/* SPDX-License-Identifier: MIT */
#include "hk_nir_match_pointwise_clamp.h"

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

/* Match base_workgroup_id[axis] * local_size[axis] (scalar or vector imul/ishl,
 * or identity when local_size == 1).  nir_lower_compute_system_values emits
 * vec3 imul(load_base_workgroup_id, vec3(workgroup_size)) even when an axis
 * has local_size == 1, before any scalarization pass runs. */
static bool
match_base_wg_scaled(nir_scalar s, unsigned axis, uint32_t local_size)
{
   s = nir_scalar_chase_movs(s);
   if (!s.def || !local_size)
      return false;

   if (local_size == 1 && s.comp == axis && s.def->bit_size == 32 &&
       intrinsic(s.def, nir_intrinsic_load_base_workgroup_id) != NULL)
      return true;

   if (alu_op(s, nir_op_imul)) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (a.comp == axis && a.def->bit_size == 32 &&
          intrinsic(a.def, nir_intrinsic_load_base_workgroup_id) &&
          constant(b, local_size))
         return true;
      if (b.comp == axis && b.def->bit_size == 32 &&
          intrinsic(b.def, nir_intrinsic_load_base_workgroup_id) &&
          constant(a, local_size))
         return true;
   }

   if (util_is_power_of_two_nonzero(local_size) && alu_op(s, nir_op_ishl)) {
      unsigned shift = ffs(local_size) - 1;
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
      if (a.comp == axis && a.def->bit_size == 32 &&
          intrinsic(a.def, nir_intrinsic_load_base_workgroup_id) &&
          constant(b, shift))
         return true;
   }

   return false;
}

/* Recognizes a component of GlobalInvocationID or LocalInvocationID on either
 * pre-preprocess NIR (load_deref of nir_var_system_value) or post-preprocess
 * NIR (load_global_invocation_id [+ load_base_workgroup_id * local_size],
 * load_local_invocation_id). */
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
      if (value.comp < 3 &&
          intrinsic(value.def, nir_intrinsic_load_global_invocation_id)) {
         *axis_out = value.comp;
         return true;
      }
      if (local_size && alu_op(value, nir_op_iadd)) {
         nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(value, 0));
         nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(value, 1));
         if (a.comp < 3 &&
             intrinsic(a.def, nir_intrinsic_load_global_invocation_id) &&
             match_base_wg_scaled(b, a.comp, local_size[a.comp])) {
            *axis_out = a.comp;
            return true;
         }
         if (b.comp < 3 &&
             intrinsic(b.def, nir_intrinsic_load_global_invocation_id) &&
             match_base_wg_scaled(a, b.comp, local_size[b.comp])) {
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

/* Match row < m && col < n guard (iand or short-circuit phi). */
static bool
match_gemm_store_guard(nir_if *branch, const uint16_t ls[3],
                       uint32_t expected_r, uint32_t expected_j,
                       uint32_t expected_n_off, uint32_t *m_offset_B)
{
   nir_scalar cond = source(branch->condition);
   nir_scalar p0 = {0}, p1 = {0};
   bool have_pair = false;

   if (scalar_op(cond, nir_op_iand)) {
      p0 = nir_scalar_chase_alu_src(cond, 0);
      p1 = nir_scalar_chase_alu_src(cond, 1);
      have_pair = true;
   } else if (nir_def_is_phi(cond.def)) {
      const nir_phi_instr *phi = nir_def_as_phi(cond.def);
      if (nir_cf_node_next(&phi->instr.block->cf_node) != &branch->cf_node)
         return false;
      nir_cf_node *prev = nir_cf_node_prev(&phi->instr.block->cf_node);
      if (!prev || prev->type != nir_cf_node_if)
         return false;
      nir_if *test = nir_cf_node_as_if(prev);
      nir_scalar first = source(test->condition);
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
      if (saw_false && saw_second) {
         p0 = first;
         have_pair = true;
      }
   }
   if (!have_pair)
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
      nir_scalar cond = source(branch->condition);
      if (!scalar_op(cond, nir_op_ieq))
         continue;
      nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
      nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
      uint32_t off, mode;
      if (push_count(a, &off) && nir_scalar_is_const(b)) {
         mode = nir_scalar_as_uint(b);
      } else if (push_count(b, &off) && nir_scalar_is_const(a)) {
         mode = nir_scalar_as_uint(a);
      } else {
         continue;
      }
      if (mode <= 2) {
         *bias_mode_off = off;
         *mode_out = mode;
         return true;
      }
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
   uint32_t read_stride_off[4] = {0};
   bool read_has_stride[4] = {0};
   bool read_is_row_idx[4] = {0};
   bool read_is_col_idx[4] = {0};
   uint32_t read_bias_arm_off[4] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
   uint8_t read_bias_arm_mask[4] = {0};
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
                  read_bindings[num_read_bindings++] = acc.binding;
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
                  } else if (read_bias_arm_mask[slot]) {
                     read_bias_arm_inconsistent[slot] = true;
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

   /* If there is a bias buffer and stores themselves were not split per
    * bias_mode arm, either the bias loads are guarded by bias_mode arms
    * (with a fallthrough no-bias mode 2) or the bias load is unconditional
    * (indexed exclusively by row -> mode 0, or col -> mode 1). */
   if (!has_bias_mode && has_bias_buffer) {
      if (read_bias_arm_inconsistent[bias_slot])
         return false;
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

   *match = (struct hk_nir_gemm_output){
      .output = out_binding,
      .reads = {
         read_bindings[a_slot],
         read_bindings[b_slot],
         has_bias_buffer ? read_bindings[bias_slot] : (struct hk_nir_clamp_buffer){0},
      },
      .m_offset_B = is_guarded ? m_off : UINT32_MAX,
      .n_offset_B = n_off,
      .k_offset_B = read_stride_off[a_slot],
      .bias_mode_offset_B = has_bias_mode ? bias_mode_off : UINT32_MAX,
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
   for (unsigned i = 0; i < 3; ++i) {
      if (launch->base_group[i] || !local_size[i])
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
hk_nir_depthwise_clamp_footprint(const struct hk_nir_depthwise_output *producer,
                                 const struct hk_nir_pointwise_clamp *consumer,
                                 const struct hk_nir_direct_shape *producer_launch,
                                 const struct hk_nir_direct_shape *consumer_launch,
                                 uint64_t producer_ends[4], uint64_t consumer_ends[2])
{
   uint32_t iw, ih, ow, oh, kw, kh, channels, count;
   if (!read_positive_push(producer_launch, producer->input_width_offset_B, &iw) ||
       !read_positive_push(producer_launch, producer->input_height_offset_B, &ih) ||
       !read_positive_push(producer_launch, producer->width_offset_B, &ow) ||
       !read_positive_push(producer_launch, producer->height_offset_B, &oh) ||
       !read_positive_push(producer_launch, producer->kernel_width_offset_B, &kw) ||
       !read_positive_push(producer_launch, producer->kernel_height_offset_B, &kh) ||
       !read_positive_push(producer_launch, producer->channels_offset_B, &channels) ||
       !read_positive_push(consumer_launch, consumer->count_offset_B, &count))
      return false;

   uint64_t p[4] = {0}, c[2] = {0};
   if (!footprint_end(producer->reads[0].offset_B, iw, ih, channels, &p[0]) ||
       !footprint_end(producer->reads[1].offset_B, kw, kh, channels, &p[1]) ||
       !footprint_end(producer->output.offset_B, ow, oh, channels, &p[3]) ||
       !footprint_end(consumer->input.offset_B, count, 1, 1, &c[0]) ||
       !footprint_end(consumer->output.offset_B, count, 1, 1, &c[1]))
      return false;

   if (producer->has_bias_buffer) {
      if (!footprint_end(producer->reads[2].offset_B, 1, 1, channels, &p[2]))
         return false;
   } else {
      p[2] = 0;
   }

   uint64_t x = (uint64_t)ow * channels;
   const uint32_t consumer_local[] = {consumer->local_size_x, 1, 1};
   if (count != x * oh || consumer_launch->groups[1] != 1 ||
       !dispatch_covers(producer_launch, producer->local_size,
                        (uint64_t[]){x, oh, 1}) ||
       !dispatch_covers(consumer_launch, consumer_local,
                        (uint64_t[]){count, 1, 1}))
      return false;

   memcpy(producer_ends, p, sizeof(p));
   memcpy(consumer_ends, c, sizeof(c));
   return true;
}

bool
hk_nir_gemm_clamp_footprint(const struct hk_nir_gemm_output *producer,
                            const struct hk_nir_pointwise_clamp *consumer,
                            const struct hk_nir_direct_shape *producer_launch,
                            const struct hk_nir_direct_shape *consumer_launch,
                            uint64_t producer_ends[4], uint64_t consumer_ends[2])
{
   if (!producer->rows_per_thread || producer->rows_per_thread > 16)
      return false;

   uint32_t m, n, k, count;
   if (!read_positive_push(producer_launch, producer->n_offset_B, &n) ||
       !read_positive_push(producer_launch, producer->k_offset_B, &k) ||
       !read_positive_push(consumer_launch, consumer->count_offset_B, &count))
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

   uint64_t p[4] = {0}, c[2] = {0};
   if (!footprint_end(producer->reads[0].offset_B, m, k, 1, &p[0]) ||
       !footprint_end(producer->reads[1].offset_B, k, n, 1, &p[1]) ||
       !footprint_end(producer->output.offset_B, m, n, 1, &p[3]) ||
       !footprint_end(consumer->input.offset_B, count, 1, 1, &c[0]) ||
       !footprint_end(consumer->output.offset_B, count, 1, 1, &c[1]))
      return false;

   if (active_bias_mode == 2 || !producer->has_bias_buffer) {
      p[2] = 0;
   } else {
      const uint32_t bias_len = active_bias_mode == 0 ? m : n;
      if (!footprint_end(producer->reads[2].offset_B, bias_len, 1, 1, &p[2]))
         return false;
   }

   const uint64_t active_y = DIV_ROUND_UP((uint64_t)m, producer->rows_per_thread);
   const uint32_t consumer_local[] = {consumer->local_size_x, 1, 1};
   if ((uint64_t)count != (uint64_t)m * n || consumer_launch->groups[1] != 1 ||
       !dispatch_covers(producer_launch, producer->local_size,
                        (uint64_t[]){n, active_y, 1}) ||
       !dispatch_covers(consumer_launch, consumer_local,
                        (uint64_t[]){count, 1, 1}))
      return false;

   memcpy(producer_ends, p, sizeof(p));
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
   nir_def *sum = nir_fadd(b, acc, prod);
   b->fp_math_ctrl = 0;
   return sum;
}

nir_shader *
hk_nir_build_depthwise_vec4(const nir_shader *producer,
                            const struct hk_nir_depthwise_output *info,
                            const struct hk_nir_clamp_epilogue *epilogue)
{
   if (!producer || !info || !info->vec4_valid ||
       info->desc_comps < 3 || info->desc_comps > 5)
      return NULL;
   if (epilogue &&
       (!hk_nir_clamp_epilogue_valid(epilogue) ||
        producer->info.float_controls_execution_mode !=
           epilogue->float_controls_execution_mode))
      return NULL;

   nir_shader *nir = nir_shader_create(NULL, MESA_SHADER_COMPUTE,
                                       producer->options);
   if (!nir)
      return NULL;
   nir->info.name = ralloc_asprintf(nir, "%s_dw_vec4%s",
                                    producer->info.name ? producer->info.name : "cs",
                                    epilogue ? "_clamp" : "");
   nir->info.workgroup_size[0] = 32;
   nir->info.workgroup_size[1] = 4;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   nir->info.num_ssbos = producer->info.num_ssbos;
   nir->info.float_controls_execution_mode = producer->info.float_controls_execution_mode;

   nir_function *fn = nir_function_create(nir, "main");
   fn->is_entrypoint = true;
   nir_function_impl *impl = nir_function_impl_create(fn);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   const uint32_t pb = info->push_bytes;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

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

   nir_def *c4 = nir_ishr_imm(&b, ch, 2);
   nir_def *out_pairs = nir_ishr_imm(&b, nir_iadd_imm(&b, out_w, 1), 1);
   nir_def *total_x4 = nir_imul(&b, out_pairs, c4);
   nir_def *in_bounds = nir_iand(&b, nir_ilt(&b, gx, total_x4), nir_ilt(&b, gy, out_h));

   nir_if *active = nir_push_if(&b, in_bounds);
   {
      nir_def *pair_idx = nir_udiv(&b, gx, c4);
      nir_def *ch4 = nir_isub(&b, gx, nir_imul(&b, pair_idx, c4));
      nir_def *ox0 = nir_ishl_imm(&b, pair_idx, 1);
      nir_def *ox1 = nir_iadd_imm(&b, ox0, 1);
      nir_def *has_p1 = nir_ilt(&b, ox1, out_w);

      nir_def *ix0 = nir_iadd(&b, px, nir_imul(&b, ox0, sx));
      nir_def *ix1 = nir_iadd(&b, ix0, sx);
      nir_def *iy0 = nir_iadd(&b, py, nir_imul(&b, gy, sy));
      nir_def *ix_max = nir_iadd(&b, ix1, nir_imul(&b, nir_iadd_imm(&b, kw, -1), dx));
      nir_def *iy_max = nir_iadd(&b, iy0, nir_imul(&b, nir_iadd_imm(&b, kh, -1), dy));
      nir_def *interior = nir_iand(&b,
         nir_iand(&b, nir_ige_imm(&b, ix0, 0), nir_ilt(&b, ix_max, in_w)),
         nir_iand(&b, nir_ige_imm(&b, iy0, 0), nir_ilt(&b, iy_max, in_h)));
      interior = nir_iand(&b, interior, has_p1);

      nir_def *zero_i = nir_imm_int(&b, 0);
      nir_def *zero4 = nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f);
      nir_def *kw_c4 = nir_imul(&b, kw, c4);
      nir_def *iw_c4 = nir_imul(&b, in_w, c4);
      nir_def *dx_c4 = nir_imul(&b, dx, c4);
      nir_def *sx_c4 = nir_imul(&b, sx, c4);
      nir_def *dy_iw_c4 = nir_imul(&b, dy, iw_c4);

      nir_def *acc0_int = NULL, *acc1_int = NULL;
      nir_def *acc0_brd = NULL, *acc1_brd = NULL;

      nir_if *int_if = nir_push_if(&b, interior);
      {
         nir_def *base_in0 = nir_iadd(&b,
            nir_imul(&b, nir_iadd(&b, nir_imul(&b, iy0, in_w), ix0), c4), ch4);
         nir_loop *ylp = nir_push_loop(&b);
         nir_block *pre_y = nir_cf_node_as_block(nir_cf_node_prev(&ylp->cf_node));
         nir_block *yhdr = nir_loop_first_block(ylp);

         nir_phi_instr *ky_phi = nir_phi_instr_create(nir);
         nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
         nir_phi_instr_add_src(ky_phi, pre_y, zero_i);

         nir_phi_instr *in_y_phi = nir_phi_instr_create(nir);
         nir_def_init(&in_y_phi->instr, &in_y_phi->def, 1, 32);
         nir_phi_instr_add_src(in_y_phi, pre_y, base_in0);

         nir_phi_instr *wt_y_phi = nir_phi_instr_create(nir);
         nir_def_init(&wt_y_phi->instr, &wt_y_phi->def, 1, 32);
         nir_phi_instr_add_src(wt_y_phi, pre_y, ch4);

         nir_phi_instr *a0_y_phi = nir_phi_instr_create(nir);
         nir_def_init(&a0_y_phi->instr, &a0_y_phi->def, 4, 32);
         nir_phi_instr_add_src(a0_y_phi, pre_y, zero4);

         nir_phi_instr *a1_y_phi = nir_phi_instr_create(nir);
         nir_def_init(&a1_y_phi->instr, &a1_y_phi->def, 4, 32);
         nir_phi_instr_add_src(a1_y_phi, pre_y, zero4);

         nir_if *ybrk = nir_push_if(&b, nir_ige(&b, &ky_phi->def, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ybrk);

         nir_loop *xlp = nir_push_loop(&b);
         nir_block *pre_x = nir_cf_node_as_block(nir_cf_node_prev(&xlp->cf_node));
         nir_block *xhdr = nir_loop_first_block(xlp);

         nir_phi_instr *kx_phi = nir_phi_instr_create(nir);
         nir_def_init(&kx_phi->instr, &kx_phi->def, 1, 32);
         nir_phi_instr_add_src(kx_phi, pre_x, zero_i);

         nir_phi_instr *in_x_phi = nir_phi_instr_create(nir);
         nir_def_init(&in_x_phi->instr, &in_x_phi->def, 1, 32);
         nir_phi_instr_add_src(in_x_phi, pre_x, &in_y_phi->def);

         nir_phi_instr *wt_x_phi = nir_phi_instr_create(nir);
         nir_def_init(&wt_x_phi->instr, &wt_x_phi->def, 1, 32);
         nir_phi_instr_add_src(wt_x_phi, pre_x, &wt_y_phi->def);

         nir_phi_instr *a0_x_phi = nir_phi_instr_create(nir);
         nir_def_init(&a0_x_phi->instr, &a0_x_phi->def, 4, 32);
         nir_phi_instr_add_src(a0_x_phi, pre_x, &a0_y_phi->def);

         nir_phi_instr *a1_x_phi = nir_phi_instr_create(nir);
         nir_def_init(&a1_x_phi->instr, &a1_x_phi->def, 4, 32);
         nir_phi_instr_add_src(a1_x_phi, pre_x, &a1_y_phi->def);

         nir_if *xbrk = nir_push_if(&b, nir_ige(&b, &kx_phi->def, kw));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, xbrk);

         nir_def *wa = dw_vec4_addr(&b, info->reads[1], info->desc_comps,
                                    info->read_res_type[1], &wt_x_phi->def, read);
         nir_def *wv = nir_load_global(&b, 4, 32, wa, .align_mul = 16, .access = read);

         nir_def *ia0 = dw_vec4_addr(&b, info->reads[0], info->desc_comps,
                                     info->read_res_type[0], &in_x_phi->def, read);
         nir_def *iv0 = nir_load_global(&b, 4, 32, ia0, .align_mul = 16, .access = read);

         nir_def *ia1 = dw_vec4_addr(&b, info->reads[0], info->desc_comps,
                                     info->read_res_type[0],
                                     nir_iadd(&b, &in_x_phi->def, sx_c4), read);
         nir_def *iv1 = nir_load_global(&b, 4, 32, ia1, .align_mul = 16, .access = read);

         nir_def *na0 = dw_fma4(&b, &a0_x_phi->def, iv0, wv, info);
         nir_def *na1 = dw_fma4(&b, &a1_x_phi->def, iv1, wv, info);

         nir_block *xback = nir_loop_last_block(xlp);
         nir_phi_instr_add_src(kx_phi, xback, nir_iadd_imm(&b, &kx_phi->def, 1));
         nir_phi_instr_add_src(in_x_phi, xback, nir_iadd(&b, &in_x_phi->def, dx_c4));
         nir_phi_instr_add_src(wt_x_phi, xback, nir_iadd(&b, &wt_x_phi->def, c4));
         nir_phi_instr_add_src(a0_x_phi, xback, na0);
         nir_phi_instr_add_src(a1_x_phi, xback, na1);
         nir_instr_insert(nir_before_block(xhdr), &kx_phi->instr);
         nir_instr_insert(nir_before_block(xhdr), &in_x_phi->instr);
         nir_instr_insert(nir_before_block(xhdr), &wt_x_phi->instr);
         nir_instr_insert(nir_before_block(xhdr), &a0_x_phi->instr);
         nir_instr_insert(nir_before_block(xhdr), &a1_x_phi->instr);
         nir_pop_loop(&b, xlp);

         nir_block *yback = nir_loop_last_block(ylp);
         nir_phi_instr_add_src(ky_phi, yback, nir_iadd_imm(&b, &ky_phi->def, 1));
         nir_phi_instr_add_src(in_y_phi, yback, nir_iadd(&b, &in_y_phi->def, dy_iw_c4));
         nir_phi_instr_add_src(wt_y_phi, yback, nir_iadd(&b, &wt_y_phi->def, kw_c4));
         nir_phi_instr_add_src(a0_y_phi, yback, &a0_x_phi->def);
         nir_phi_instr_add_src(a1_y_phi, yback, &a1_x_phi->def);
         nir_instr_insert(nir_before_block(yhdr), &ky_phi->instr);
         nir_instr_insert(nir_before_block(yhdr), &in_y_phi->instr);
         nir_instr_insert(nir_before_block(yhdr), &wt_y_phi->instr);
         nir_instr_insert(nir_before_block(yhdr), &a0_y_phi->instr);
         nir_instr_insert(nir_before_block(yhdr), &a1_y_phi->instr);
         nir_pop_loop(&b, ylp);

         acc0_int = &a0_y_phi->def;
         acc1_int = &a1_y_phi->def;
      }
      nir_push_else(&b, int_if);
      {
         nir_loop *oy_lp = nir_push_loop(&b);
         nir_block *pre_oy = nir_cf_node_as_block(nir_cf_node_prev(&oy_lp->cf_node));
         nir_block *oy_hdr = nir_loop_first_block(oy_lp);

         nir_phi_instr *ky_phi = nir_phi_instr_create(nir);
         nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
         nir_phi_instr_add_src(ky_phi, pre_oy, zero_i);

         nir_phi_instr *oy_a0_phi = nir_phi_instr_create(nir);
         nir_def_init(&oy_a0_phi->instr, &oy_a0_phi->def, 4, 32);
         nir_phi_instr_add_src(oy_a0_phi, pre_oy, zero4);

         nir_phi_instr *oy_a1_phi = nir_phi_instr_create(nir);
         nir_def_init(&oy_a1_phi->instr, &oy_a1_phi->def, 4, 32);
         nir_phi_instr_add_src(oy_a1_phi, pre_oy, zero4);

         nir_if *ybrk = nir_push_if(&b, nir_ige(&b, &ky_phi->def, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ybrk);

         nir_def *iy = nir_iadd(&b, iy0, nir_imul(&b, &ky_phi->def, dy));
         nir_def *iy_ok = nir_ult(&b, iy, in_h);
         nir_def *y_a0 = NULL, *y_a1 = NULL;
         nir_if *y_ok = nir_push_if(&b, iy_ok);
         {
            nir_def *in_row4 = nir_iadd(&b, nir_imul(&b, iy, iw_c4), ch4);
            nir_def *wt_row4 = nir_iadd(&b, nir_imul(&b, &ky_phi->def, kw_c4), ch4);
            nir_loop *ix_lp = nir_push_loop(&b);
            nir_block *pre_ix = nir_cf_node_as_block(nir_cf_node_prev(&ix_lp->cf_node));
            nir_block *ix_hdr = nir_loop_first_block(ix_lp);

            nir_phi_instr *kx_phi = nir_phi_instr_create(nir);
            nir_def_init(&kx_phi->instr, &kx_phi->def, 1, 32);
            nir_phi_instr_add_src(kx_phi, pre_ix, zero_i);

            nir_phi_instr *ix_a0_phi = nir_phi_instr_create(nir);
            nir_def_init(&ix_a0_phi->instr, &ix_a0_phi->def, 4, 32);
            nir_phi_instr_add_src(ix_a0_phi, pre_ix, &oy_a0_phi->def);

            nir_phi_instr *ix_a1_phi = nir_phi_instr_create(nir);
            nir_def_init(&ix_a1_phi->instr, &ix_a1_phi->def, 4, 32);
            nir_phi_instr_add_src(ix_a1_phi, pre_ix, &oy_a1_phi->def);

            nir_if *xbrk = nir_push_if(&b, nir_ige(&b, &kx_phi->def, kw));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xbrk);

            nir_def *kdx = nir_imul(&b, &kx_phi->def, dx);
            nir_def *ix_a = nir_iadd(&b, ix0, kdx);
            nir_def *ix_b = nir_iadd(&b, ix1, kdx);
            nir_def *ok0 = nir_ult(&b, ix_a, in_w);
            nir_def *ok1 = nir_iand(&b, has_p1, nir_ult(&b, ix_b, in_w));

            nir_def *any_a0 = NULL, *any_a1 = NULL;
            nir_if *any_ok = nir_push_if(&b, nir_ior(&b, ok0, ok1));
            {
               nir_def *wt_idx4 = nir_iadd(&b, wt_row4, nir_imul(&b, &kx_phi->def, c4));
               nir_def *wv = nir_load_global(&b, 4, 32,
                  dw_vec4_addr(&b, info->reads[1], info->desc_comps, info->read_res_type[1], wt_idx4, read),
                  .align_mul = 16, .access = read);

               nir_def *hit0 = NULL;
               nir_if *if0 = nir_push_if(&b, ok0);
               {
                  nir_def *in_idx4 = nir_iadd(&b, in_row4, nir_imul(&b, ix_a, c4));
                  nir_def *iv0 = nir_load_global(&b, 4, 32,
                     dw_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0], in_idx4, read),
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
                     dw_vec4_addr(&b, info->reads[0], info->desc_comps, info->read_res_type[0], in_idx4, read),
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

      nir_def *val0 = acc0;
      nir_def *val1 = acc1;
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

      nir_def *out_idx0 =
         nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, gy, out_w), ox0), c4), ch4);
      nir_def *oa0 = dw_vec4_addr(&b, info->output, info->desc_comps,
                                  info->out_res_type, out_idx0, ACCESS_NON_READABLE);
      nir_store_global(&b, val0, oa0, .align_mul = 16, .write_mask = 0xf,
                       .access = ACCESS_NON_READABLE);

      nir_if *st1 = nir_push_if(&b, has_p1);
      {
         nir_def *oa1 = dw_vec4_addr(&b, info->output, info->desc_comps,
                                     info->out_res_type, nir_iadd(&b, out_idx0, c4),
                                     ACCESS_NON_READABLE);
         nir_store_global(&b, val1, oa1, .align_mul = 16, .write_mask = 0xf,
                          .access = ACCESS_NON_READABLE);
      }
      nir_pop_if(&b, st1);
   }
   nir_pop_if(&b, active);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}
