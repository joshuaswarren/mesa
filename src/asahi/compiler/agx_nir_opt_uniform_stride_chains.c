/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "util/u_dynarray.h"

struct stride_chain {
   nir_alu_instr *add;
   nir_scalar base;
   nir_scalar stride;
   unsigned step;
   unsigned root;
   unsigned max_step;
};

struct hoist_ctx {
   BITSET_WORD *can_move;
   unsigned nonuniform_cf_nesting;
   bool no_textures;
};

static bool
can_move_src(nir_src *src, void *state)
{
   struct hoist_ctx *ctx = state;
   return BITSET_TEST(ctx->can_move, src->ssa->index);
}

static bool
can_move_srcs(nir_instr *instr, struct hoist_ctx *ctx)
{
   return nir_foreach_src(instr, can_move_src, ctx);
}

/* Match nir_opt_preamble's movability rules under agx_nir_opt_preamble's
 * options (drawid_uniform = true, subgroup_size_uniform = true,
 * load_workgroup_size_allowed = false, no_textures = agx3c), so we only
 * rewrite v_j = base + (j * s) when j * s can actually hoist into the
 * preamble instead of adding main-shader multiplies. */
static bool
can_move_intrinsic(nir_intrinsic_instr *instr, struct hoist_ctx *ctx)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_work_dim:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_ray_launch_size:
   case nir_intrinsic_load_is_indexed_draw:
   case nir_intrinsic_load_viewport_scale:
   case nir_intrinsic_load_user_clip_plane:
   case nir_intrinsic_load_viewport_x_scale:
   case nir_intrinsic_load_viewport_y_scale:
   case nir_intrinsic_load_viewport_z_scale:
   case nir_intrinsic_load_viewport_offset:
   case nir_intrinsic_load_viewport_x_offset:
   case nir_intrinsic_load_viewport_y_offset:
   case nir_intrinsic_load_viewport_z_offset:
   case nir_intrinsic_load_blend_const_color_a_float:
   case nir_intrinsic_load_blend_const_color_b_float:
   case nir_intrinsic_load_blend_const_color_g_float:
   case nir_intrinsic_load_blend_const_color_r_float:
   case nir_intrinsic_load_blend_const_color_rgba:
   case nir_intrinsic_load_blend_const_color_aaaa8888_unorm:
   case nir_intrinsic_load_blend_const_color_rgba8888_unorm:
   case nir_intrinsic_load_line_width:
   case nir_intrinsic_load_aa_line_width:
   case nir_intrinsic_load_vbo_base_agx:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_draw_id:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_num_subgroups:
      return true;

   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ubo_vec4:
   case nir_intrinsic_load_push_constant:
   case nir_intrinsic_get_ubo_size:
   case nir_intrinsic_get_ssbo_size:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_load_deref:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_preamble:
   case nir_intrinsic_load_constant:
   case nir_intrinsic_load_sample_pos_from_id:
   case nir_intrinsic_load_kernel_input:
   case nir_intrinsic_image_levels:
   case nir_intrinsic_image_deref_levels:
   case nir_intrinsic_bindless_image_levels:
   case nir_intrinsic_image_heap_levels:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_bindless_image_samples:
   case nir_intrinsic_image_heap_samples:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_image_heap_size:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_load_ssbo_address:
   case nir_intrinsic_load_constant_agx:
   case nir_intrinsic_bindless_image_agx:
   case nir_intrinsic_bindless_sampler_agx:
      return can_move_srcs(&instr->instr, ctx);

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_samples_identical:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_image_heap_load:
      if (ctx->no_textures)
         return false;
      FALLTHROUGH;
   case nir_intrinsic_load_global_bounded:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_global_offset:
   case nir_intrinsic_load_agx:
      return (nir_intrinsic_access(instr) & ACCESS_CAN_REORDER) &&
             can_move_srcs(&instr->instr, ctx);

   default:
      return false;
   }
}

static bool
can_move_instr(nir_instr *instr, struct hoist_ctx *ctx)
{
   if (ctx->nonuniform_cf_nesting > 0 &&
       instr->type != nir_instr_type_phi && !nir_instr_can_speculate(instr))
      return false;

   switch (instr->type) {
   case nir_instr_type_tex: {
      if (ctx->no_textures)
         return false;
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (nir_tex_instr_has_implicit_derivative(tex) &&
          tex->op != nir_texop_tex)
         return false;
      return can_move_srcs(instr, ctx);
   }

   case nir_instr_type_alu:
      return can_move_srcs(instr, ctx);

   case nir_instr_type_intrinsic:
      return can_move_intrinsic(nir_instr_as_intrinsic(instr), ctx);

   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;

   case nir_instr_type_deref: {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      if (deref->deref_type == nir_deref_type_var) {
         return deref->modes == nir_var_uniform ||
                deref->modes == nir_var_mem_ubo;
      }
      return can_move_srcs(instr, ctx);
   }

   case nir_instr_type_phi: {
      nir_cf_node *prev = nir_cf_node_prev(&instr->block->cf_node);
      if (!prev || prev->type != nir_cf_node_if)
         return false;
      nir_if *nif = nir_cf_node_as_if(prev);
      return BITSET_TEST(ctx->can_move, nif->condition.ssa->index) &&
             can_move_srcs(instr, ctx);
   }

   default:
      return false;
   }
}

static void
mark_hoistable_cf_list(struct hoist_ctx *ctx, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, block) {
            nir_def *def = nir_instr_def(instr);
            if (!def)
               continue;
            if (can_move_instr(instr, ctx))
               BITSET_SET(ctx->can_move, def->index);
         }
         break;
      }

      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         const bool uniform =
            BITSET_TEST(ctx->can_move, nif->condition.ssa->index);
         if (!uniform)
            ctx->nonuniform_cf_nesting++;
         mark_hoistable_cf_list(ctx, &nif->then_list);
         mark_hoistable_cf_list(ctx, &nif->else_list);
         if (!uniform)
            ctx->nonuniform_cf_nesting--;
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         ctx->nonuniform_cf_nesting++;
         mark_hoistable_cf_list(ctx, &loop->body);
         ctx->nonuniform_cf_nesting--;
         break;
      }

      default:
         UNREACHABLE("invalid cf node");
      }
   }
}

/* Flatten serial uniform-stride 32-bit addition chains in one block:
 *
 *    v1 = base + s; v2 = v1 + s; ...; vk = v_{k-1} + s
 *
 * becomes base + (2 * s), ..., base + (k * s) when s is non-constant,
 * preamble-hoistable, and k >= 3. Both forms are exact 32-bit modular sums;
 * requiring s to be preamble-hoistable ensures the uniform multiples j * s
 * move into the preamble instead of adding main-shader multiplies, leaving
 * every strided address to compute independently in one add. */
static bool
opt_block_add_chains(nir_builder *b, nir_block *block,
                     const BITSET_WORD *can_move,
                     struct util_dynarray *chains)
{
   util_dynarray_clear(chains);
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_iadd || alu->def.bit_size != 32 ||
          alu->def.num_components != 1)
         continue;

      const unsigned n =
         util_dynarray_num_elements(chains, struct stride_chain);
      if (n >= 64)
         break;

      nir_scalar def = nir_get_scalar(&alu->def, 0);
      for (unsigned m = 0; m < 2; ++m) {
         if (nir_src_is_divergent(&alu->src[m].src) ||
             !nir_src_is_divergent(&alu->src[1 - m].src))
            continue;
         nir_scalar s = nir_scalar_chase_alu_src(def, m);
         if (nir_scalar_is_const(s) || !BITSET_TEST(can_move, s.def->index))
            continue;
         nir_scalar p = nir_scalar_chase_alu_src(def, 1 - m);

         struct stride_chain *arr = chains->data;
         int prev = -1;
         for (unsigned i = n; i-- > 0;) {
            if (nir_scalar_equal(p, nir_get_scalar(&arr[i].add->def, 0)) &&
                nir_scalar_equal(s, arr[i].stride)) {
               prev = (int)i;
               break;
            }
         }

         struct stride_chain entry;
         if (prev >= 0 && arr[prev].step < 16) {
            entry.add = alu;
            entry.base = arr[prev].base;
            entry.stride = s;
            entry.step = arr[prev].step + 1;
            entry.root = arr[prev].root;
            entry.max_step = entry.step;
            if (entry.step > arr[entry.root].max_step)
               arr[entry.root].max_step = entry.step;
         } else {
            entry.add = alu;
            entry.base = p;
            entry.stride = s;
            entry.step = 1;
            entry.root = n;
            entry.max_step = 1;
         }
         util_dynarray_append(chains, entry);
         break;
      }
   }

   const unsigned n = util_dynarray_num_elements(chains, struct stride_chain);
   struct stride_chain *arr = chains->data;
   bool progress = false;
   for (unsigned i = 0; i < n; ++i) {
      if (arr[i].step < 2 || arr[arr[i].root].max_step < 3)
         continue;
      b->cursor = nir_before_instr(&arr[i].add->instr);
      nir_def *base = nir_mov_scalar(b, arr[i].base);
      nir_def *stride = nir_mov_scalar(b, arr[i].stride);
      nir_def *offset = nir_imul_imm(b, stride, arr[i].step);
      nir_def_replace(&arr[i].add->def, nir_iadd(b, base, offset));
      progress = true;
   }
   return progress;
}

bool
agx_nir_opt_uniform_stride_chains(nir_shader *shader, bool no_textures)
{
   nir_divergence_analysis(shader);
   bool any = false;
   struct util_dynarray chains;
   util_dynarray_init(&chains, NULL);
   nir_foreach_function_impl(impl, shader) {
      nir_index_ssa_defs(impl);
      BITSET_WORD *can_move =
         calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
      struct hoist_ctx hctx = {
         .can_move = can_move,
         .no_textures = no_textures,
      };
      mark_hoistable_cf_list(&hctx, &impl->body);

      nir_builder b = nir_builder_create(impl);
      bool progress = false;
      nir_foreach_block(block, impl)
         progress |= opt_block_add_chains(&b, block, can_move, &chains);
      free(can_move);
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }
   util_dynarray_fini(&chains);
   return any;
}
