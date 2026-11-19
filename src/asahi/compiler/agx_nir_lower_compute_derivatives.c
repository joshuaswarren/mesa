/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_debug.h"
#include "agx_nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"

static nir_def *
quad_delta(nir_builder *b, nir_def *v, bool vertical)
{
   const unsigned bit_size = v->bit_size;
   nir_def *x = bit_size == 32 ? v : nir_f2f32(b, v);
   nir_def *other =
      vertical ? nir_quad_swap_vertical(b, x) : nir_quad_swap_horizontal(b, x);
   nir_def *d = nir_fsub(b, other, x);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *high = nir_i2b(b, nir_iand_imm(b, lane, vertical ? 2 : 1));
   nir_def *r = nir_bcsel(b, high, nir_fneg(b, d), d);
   return bit_size == 32 ? r : nir_f2fN(b, r, bit_size);
}

static nir_def *
quad_derivative(nir_builder *b, nir_def *v, bool vertical)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < v->num_components; ++c)
      comps[c] = quad_delta(b, nir_channel(b, v, c), vertical);
   return nir_vec(b, comps, v->num_components);
}

nir_def *
agx_nir_quad_ddx(nir_builder *b, nir_def *v)
{
   return quad_derivative(b, v, false);
}

nir_def *
agx_nir_quad_ddy(nir_builder *b, nir_def *v)
{
   return quad_derivative(b, v, true);
}

static bool
lower_derivative(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   bool vertical;
   switch (intr->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
      vertical = false;
      break;
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
      vertical = true;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *r = quad_derivative(b, intr->src[0].ssa, vertical);
   nir_def_replace(&intr->def, r);
   return true;
}

static bool
lower_implicit_sample(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_MS ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_RECT)
      return false;

   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned grad_comps = tex->coord_components - (tex->is_array ? 1 : 0);
   nir_def *c = nir_trim_vector(b, coord, grad_comps);
   nir_def *ddx = agx_nir_quad_ddx(b, c);
   nir_def *ddy = agx_nir_quad_ddy(b, c);

   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias) {
      nir_def *scale = nir_fexp2(b, nir_f2f32(b, bias));
      ddx = nir_fmul(b, ddx, nir_f2fN(b, scale, ddx->bit_size));
      ddy = nir_fmul(b, ddy, nir_f2fN(b, scale, ddy->bit_size));
   }

   nir_tex_instr_add_src(tex, nir_tex_src_ddx, ddx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, ddy);
   tex->op = nir_texop_txd;
   return true;
}

bool
agx_nir_lower_compute_derivatives(nir_shader *s)
{
   assert(mesa_shader_stage_uses_workgroup(s->info.stage) &&
          s->info.derivative_group != DERIVATIVE_GROUP_NONE);

   bool progress = false;
   if (!debug_get_bool_option("ASAHI_G17P_COMPUTE_HW_DERIVATIVE_DIAGNOSTIC",
                              false)) {
      NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_derivative,
               nir_metadata_control_flow, NULL);
   }
   NIR_PASS(progress, s, nir_shader_tex_pass, lower_implicit_sample,
            nir_metadata_control_flow, NULL);
   return progress;
}

static nir_def *
workgroup_size(nir_builder *b, nir_shader *s)
{
   if (s->info.workgroup_size_variable)
      return nir_load_workgroup_size(b);
   return nir_imm_ivec3(b, s->info.workgroup_size[0],
                        s->info.workgroup_size[1], s->info.workgroup_size[2]);
}

static bool
lower_quad_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_shader *s = data;
   const unsigned bit_size = intr->def.bit_size;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_invocation_index: {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *id = nir_load_local_invocation_id(b);
      nir_def *size = workgroup_size(b, s);
      nir_def *index = nir_iadd(
         b, nir_channel(b, id, 0),
         nir_imul(b, nir_channel(b, size, 0),
                  nir_iadd(b, nir_channel(b, id, 1),
                           nir_imul(b, nir_channel(b, size, 1),
                                    nir_channel(b, id, 2)))));
      nir_def_replace(&intr->def, nir_u2uN(b, index, bit_size));
      return true;
   }
   case nir_intrinsic_load_global_invocation_id: {
      b->cursor = nir_before_instr(&intr->instr);
      /* Only a compute dispatch has a base workgroup. */
      nir_def *group = nir_load_workgroup_id(b);
      if (s->info.stage == MESA_SHADER_COMPUTE)
         group = nir_iadd(b, group, nir_load_base_workgroup_id(b, 32));
      nir_def *global = nir_iadd(b, nir_imul(b, group, workgroup_size(b, s)),
                                 nir_load_local_invocation_id(b));
      nir_def_replace(&intr->def, nir_u2uN(b, global, bit_size));
      return true;
   }
   default:
      return false;
   }
}

bool
agx_nir_lower_quad_derivative_ids(nir_shader *s)
{
   assert(mesa_shader_stage_uses_workgroup(s->info.stage) &&
          s->info.derivative_group == DERIVATIVE_GROUP_QUADS);
   return nir_shader_intrinsics_pass(s, lower_quad_id,
                                     nir_metadata_control_flow, s);
}
