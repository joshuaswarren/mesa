/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir_ray_query.h"

#include <string.h>

#include "asahi/lib/agx_rt_result.h"

static bool
committed_result_layout(nir_ray_query_value value, uint32_t column,
                        struct agx_nir_ray_query_result_layout *out)
{
   if (column != 0)
      return false;

   switch (value) {
   case nir_ray_query_value_intersection_type:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_TYPE,
         .components = 1,
         .storage_bit_size = 8,
         .kind = AGX_NIR_RAY_QUERY_RESULT_U32,
      };
      return true;
   case nir_ray_query_value_intersection_t:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_T,
         .components = 1,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_F32,
      };
      return true;
   case nir_ray_query_value_intersection_geometry_index:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_GEOMETRY_ID,
         .components = 1,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_U32,
      };
      return true;
   case nir_ray_query_value_intersection_primitive_index:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_PRIMITIVE_ID,
         .components = 1,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_U32,
      };
      return true;
   case nir_ray_query_value_intersection_barycentrics:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_BARYCENTRICS,
         .components = 2,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_F32,
      };
      return true;
   case nir_ray_query_value_intersection_front_face:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_FRONT_FACE,
         .components = 1,
         .storage_bit_size = 8,
         .kind = AGX_NIR_RAY_QUERY_RESULT_BOOL,
      };
      return true;
   default:
      /* Everything else -- including committed instance id and user instance
       * id, whose accessors resolve through the candidate record rather than
       * a single committed field -- has no exact ray-query offset. */
      return false;
   }
}

static bool
committed_prerequisites_for_value(
   enum agx_chip chip, nir_ray_query_value value, uint32_t column,
   struct agx_nir_ray_query_committed_prerequisites *out)
{
   if (out == NULL)
      return false;

   memset(out, 0, sizeof(*out));

   struct agx_nir_ray_query_result_layout result = {0};
   if (!committed_result_layout(value, column, &result))
      return false;

   const enum agx_isa_encoding intersect_encoding =
      agx3_intersect_ray_witnessed_chip_encoding(chip);
   const enum agx_isa_encoding continue_encoding =
      agx3_continue_ray_witnessed_chip_encoding(chip);
   const enum agx_isa_encoding inc_stk_encoding =
      agx3_inc_stk_witnessed_chip_encoding(chip);

   if (intersect_encoding == AGX_ISA_ENCODING_UNKNOWN ||
       intersect_encoding != continue_encoding ||
       intersect_encoding != inc_stk_encoding)
      return false;

   struct agx3_encoded_intersect_ray intersect;
   struct agx3_encoded_continue_ray_fixed_first_site_witness continue_first;
   struct agx3_encoded_inc_stk_fixed_witness inc_stk;

   if (!agx3_pack_intersect_ray(intersect_encoding, 0, &intersect) ||
       !agx3_pack_continue_ray_fixed_first_site_witness(continue_encoding,
                                                        &continue_first) ||
       !agx3_pack_inc_stk_fixed_witness(inc_stk_encoding, &inc_stk))
      return false;

   *out = (struct agx_nir_ray_query_committed_prerequisites){
      .encoding = intersect_encoding,
      .query_state_size = AGX_RT_RAY_QUERY_STATE_SIZE,
      .intersect_ray_witness_size = intersect.length,
      .continue_ray_first_site_witness_size = continue_first.length,
      .inc_stk_witness_size = inc_stk.length,
      .result = result,
   };
   return true;
}

bool
agx_nir_ray_query_committed_load_prerequisites(
   enum agx_chip chip, const nir_intrinsic_instr *load,
   struct agx_nir_ray_query_committed_prerequisites *out)
{
   if (out == NULL)
      return false;

   memset(out, 0, sizeof(*out));

   if (load == NULL || load->intrinsic != nir_intrinsic_rq_load ||
       !nir_intrinsic_committed(load))
      return false;

   return committed_prerequisites_for_value(chip,
                                            nir_intrinsic_ray_query_value(load),
                                            nir_intrinsic_column(load), out);
}

/* Gen1 packet math from native 25G83 query.next; shared arithmetic ported
 * from the M5 helper, with Gen1 control fields and a 32-byte core. */
#include "compiler/nir/nir_builder.h"

void
agx_nir_gen1_ray_core_words(nir_builder *b, nir_def *origin,
                                nir_def *direction,
                                nir_def *t_max, nir_def *bounds_min,
                                nir_def *bounds_max, nir_def *cull_mode,
                                nir_def *flag16, nir_def *words[2])
{
   nir_def *o[3], *d[3], *far[3];
   for (unsigned i = 0; i < 3; ++i) {
      o[i] = nir_channel(b, origin, i);
      d[i] = nir_channel(b, direction, i);
      nir_def *lo = nir_fsub(b, o[i], nir_channel(b, bounds_min, i));
      nir_def *hi = nir_fsub(b, o[i], nir_channel(b, bounds_max, i));
      /* One-ULP outward bias, as in the captured native woop setup. */
      lo = nir_fmul(b, lo, nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), lo),
                                   nir_imm_float(b, 0x1.fffffcp-1f),
                                   nir_imm_float(b, 0x1.000002p0f)));
      hi = nir_fmul(b, hi, nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), hi),
                                   nir_imm_float(b, 0x1.000002p0f),
                                   nir_imm_float(b, 0x1.fffffcp-1f)));
      far[i] = nir_fmax(b, nir_fabs(b, lo), nir_fabs(b, hi));
   }
   nir_def *y_major = nir_flt(b, nir_fabs(b, d[0]), nir_fabs(b, d[1]));
   nir_def *xy_major = nir_bcsel(b, y_major, d[1], d[0]);
   nir_def *z_major = nir_flt(b, nir_fabs(b, xy_major), nir_fabs(b, d[2]));
   nir_def *major = nir_bcsel(b, z_major, nir_imm_int(b, 2),
                             nir_b2i32(b, y_major));
   nir_def *major_dir = nir_bcsel(b, z_major, d[2], xy_major);
   nir_def *major_far = nir_bcsel(b, z_major, far[2],
                                 nir_bcsel(b, y_major, far[1], far[0]));
   nir_def *other_far = nir_bcsel(
      b, z_major, nir_fmax(b, far[0], far[1]),
      nir_fmax(b, far[2], nir_bcsel(b, y_major, far[0], far[1])));
   nir_def *epsilon = nir_fmul_imm(b, nir_fadd(b, major_far, other_far), 0x1.4p-22f);
   nir_def *quantized = nir_iand_imm(
      b, nir_ushr_imm(b, nir_iadd_imm(b, epsilon, 131071), 17), 0x3fff);
   nir_def *u = nir_bcsel(b, nir_ieq_imm(b, major, 2), nir_imm_int(b, 0),
                         nir_iadd_imm(b, major, 1));
   nir_def *v = nir_bcsel(b, nir_ieq_imm(b, major, 0), nir_imm_int(b, 2),
                         nir_iadd_imm(b, major, -1));
   nir_def *negative = nir_flt(b, major_dir, nir_imm_float(b, 0));
   nir_def *sx = nir_bcsel(b, negative, v, u);
   nir_def *sy = nir_bcsel(b, negative, u, v);
   nir_def *control = nir_ior(
      b, major, nir_ior(b, nir_ishl_imm(b, quantized, 2),
         nir_ior(b, nir_ishl_imm(b, nir_b2i32(b, flag16), 16),
            nir_ior(b, nir_ishl_imm(b, cull_mode, 18),
               nir_ior(b, nir_ishl_imm(b, sx, 20), nir_ishl_imm(b, sy, 22))))));
   words[0] = nir_vec4(b, o[0], d[0], o[1], d[1]);
   words[1] = nir_vec4(b, o[2], d[2], t_max, control);
}

nir_def *
agx_nir_gen1_trace_opaque_with_status(nir_builder *b, nir_def *bvh, nir_def *request,
                         nir_def *query_base, nir_def *lane,
                         nir_def *launch, unsigned context_uniform, unsigned status_uniform)
{
   nir_intersect_ray_gen1_agx(b, bvh, request, launch, nir_imm_int(b,0),
                             .base=context_uniform);
   return agx_nir_gen1_wait_traversal(b, query_base, lane, status_uniform);
}

nir_def *
agx_nir_gen1_wait_traversal(nir_builder *b, nir_def *query_base, nir_def *lane,
                           unsigned status_uniform)
{
   nir_variable *completed = nir_local_variable_create(b->impl,
      glsl_bool_type(), "rq_traversal_completed");
   nir_store_var(b, completed, nir_imm_true(b), 1);
   nir_def *pending = nir_load_scs_base_agx(b,1,16,
      nir_iadd(b,query_base,nir_ishl_imm(b,lane,3)),
      .base=263,.format=PIPE_FORMAT_R8_UINT,.component=4,.flags=8);
   nir_push_if(b,nir_ine_imm(b,nir_iand_imm(b,pending,64),0));
   nir_def *status_address=nir_iadd(b,query_base,nir_ishl_imm(b,lane,3));
   status_address=nir_iadd(b,status_address,
                          nir_load_preamble(b,1,32,.base=status_uniform));
   /* A continue can suspend while RayCore traverses a large overlapping TLAS.
    * Poll count is not an instance/node limit or a completion indication.
    * Never return an unfinished result as a miss or reuse its live context.
    * The kernel's job timeout bounds a stuck traversal, as for other GPU work.
    */
   nir_push_loop(b);
   nir_def *status=nir_load_scs_base_agx(b,1,16,status_address,
      .base=7,.format=PIPE_FORMAT_R8_UINT,.component=4,.flags=4);
   nir_store_scs_base_agx(b,nir_iand_imm(b,status,~64u),
      status_address,.base=7,.format=PIPE_FORMAT_R8_UINT,.component=4);
   /* G17P Metal publishes the zero-extended SR59 resident ID to staging4. */
   nir_continue_ray_gen1_agx(b, nir_load_scs_thread_index_agx(b));
   status=nir_load_scs_base_agx(b,1,16,status_address,
      .base=7,.format=PIPE_FORMAT_R8_UINT,.component=4,.flags=8);
   nir_def *done = nir_ieq_imm(b,nir_iand_imm(b,status,64),0);
   nir_store_var(b, completed, done, 1);
   nir_break_if(b, done);
   nir_pop_loop(b,NULL);
   nir_pop_if(b,NULL);
   return nir_load_var(b, completed);
}
