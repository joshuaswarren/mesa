#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_nir_gen1_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
static void
standard_ray_query(unsigned count, unsigned array, bool instances, bool candidates,
                   bool wrapper)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "Gen1 standard ray query");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = count + wrapper;
   nir_variable *queries[8];
   nir_def *rqs[8];
   assert(count && count <= 8);
   const struct glsl_type *array_type = array == 2 ?
      glsl_array_type(glsl_array_type(glsl_uint64_t_type(), 4, 0), 2, 0) :
      glsl_array_type(glsl_uint64_t_type(), count, 0);
   nir_variable *query_array = array ? nir_local_variable_create(
      b.impl, array_type, "queries") : NULL;
   if (query_array)
      query_array->data.ray_query = true;
   for (unsigned i = 0; i < count; ++i) {
      nir_variable *query = array ? query_array :
         nir_local_variable_create(b.impl, glsl_uint64_t_type(), "query");
      query->data.ray_query = true;
      queries[i] = query;
      nir_deref_instr *deref = nir_build_deref_var(&b, query);
      if (array) {
         nir_def *index = nir_iand_imm(&b,
            nir_iadd_imm(&b, nir_load_local_invocation_index(&b), i), count - 1);
         if (array == 2) {
            deref = nir_build_deref_array(&b, deref, nir_ushr_imm(&b, index, 2));
            index = nir_iand_imm(&b, index, 3);
         }
         deref = nir_build_deref_array(&b, deref, index);
      }
      nir_def *rq = &deref->def;
      rqs[i] = rq;
      nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10002000000ull),
                        candidates ? nir_ishl_imm(&b,nir_iand_imm(&b,nir_load_local_invocation_index(&b),1),1) : nir_imm_int(&b, 1), nir_imm_int(&b, 255),
                        nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0),
                        nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   }
   nir_variable *wrapper_query = NULL;
   if (wrapper) {
      assert(count == 1 && !array);
      wrapper_query = nir_local_variable_create(
         b.impl, glsl_uint64_t_type(), "wrapper_query");
      wrapper_query->data.ray_query = true;
      nir_deref_instr *target = nir_build_deref_var(&b, wrapper_query);
      nir_store_deref(&b, target,
                     nir_load_deref(&b, nir_build_deref_var(&b, queries[0])), 1);
      rqs[0] = &target->def;
   }
   for (unsigned i = count; i-- > 0;) {
      nir_push_loop(&b);
      nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rqs[i])));
      if(candidates) {
         nir_def *candidate_id=nir_rq_load(&b,1,32,rqs[i],
            .ray_query_value=nir_ray_query_value_intersection_instance_id,.committed=false);
         nir_def *candidate_addr=nir_rq_load(&b,1,64,rqs[i],
            .ray_query_value=nir_ray_query_value_intersection_instance_address,.committed=false);
         nir_store_global(&b,candidate_id,nir_imm_int64(&b,0x10003020000ull),.align_mul=4,.write_mask=1);
         nir_store_global(&b,candidate_addr,nir_imm_int64(&b,0x10003020008ull),.align_mul=8,.write_mask=1);
         nir_rq_confirm_intersection(&b,rqs[i]);
      }
      nir_pop_loop(&b, NULL);
   }
   if (wrapper) {
      nir_deref_instr *target = nir_build_deref_var(&b, queries[0]);
      nir_store_deref(&b, target,
                     nir_load_deref(&b, nir_build_deref_var(&b, wrapper_query)), 1);
      rqs[0] = &target->def;
   }
   nir_def *rq = rqs[0];
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type, .committed = true);
   nir_def *distance = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_t, .committed = true);
   nir_store_global(&b, nir_vec2(&b, type, distance),
                     nir_imm_int64(&b, 0x10003010000ull),
                     .align_mul = 8, .write_mask = 3);
   if (instances) {
      nir_def *id=nir_rq_load(&b,1,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_instance_id,.committed=true);
      nir_def *custom=nir_rq_load(&b,1,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_instance_custom_index,.committed=true);
      nir_def *addr=nir_rq_load(&b,1,64,rq,
         .ray_query_value=nir_ray_query_value_intersection_instance_address,.committed=true);
      nir_def *sbt=nir_rq_load(&b,1,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_instance_sbt_index,.committed=true);
      nir_store_global(&b,nir_vec2(&b,id,custom),nir_imm_int64(&b,0x10003010008ull),.align_mul=8,.write_mask=3);
      nir_store_global(&b,addr,nir_imm_int64(&b,0x10003010010ull),.align_mul=8,.write_mask=1);
      nir_store_global(&b,sbt,nir_imm_int64(&b,0x10003010018ull),.align_mul=4,.write_mask=1);
   }
   if(candidates) {
      nir_def *object_origin=nir_rq_load(&b,3,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_object_ray_origin,.committed=true);
      nir_def *object_direction=nir_rq_load(&b,3,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_object_ray_direction,.committed=true);
      nir_def *object_to_world=nir_rq_load(&b,3,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_object_to_world,.committed=true,.column=3);
      nir_def *sbt=nir_rq_load(&b,1,32,rq,
         .ray_query_value=nir_ray_query_value_intersection_instance_sbt_index,.committed=true);
      nir_store_global(&b,object_origin,nir_imm_int64(&b,0x10003030000ull),.align_mul=4,.write_mask=7);
      nir_store_global(&b,object_direction,nir_imm_int64(&b,0x10003030010ull),.align_mul=4,.write_mask=7);
      nir_store_global(&b,object_to_world,nir_imm_int64(&b,0x10003030020ull),.align_mul=4,.write_mask=7);
      nir_store_global(&b,sbt,nir_imm_int64(&b,0x10003030030ull),.align_mul=4,.write_mask=1);
   }
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = candidates ? 118 : 124, .bvh_uniform = 96,
      .status_uniform = candidates ? 128 : instances ? 130 : 128,
      .instance_traversal=instances,.candidates=candidates,
      .launch_header = candidates ? 0x210000 : 0x110000,
   };
   struct agx_gen1_ray_query_layout bad_layout = layout;
   bad_layout.payload_uniform++;
   assert(!agx_nir_lower_gen1_ray_query(b.shader, &bad_layout));
   assert(queries[0]->data.ray_query && b.shader->info.ray_queries == count + wrapper);
   assert(agx_nir_lower_gen1_ray_query(b.shader, &layout));
   assert(!b.shader->info.ray_queries);
   if (wrapper) {
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      unsigned bounds_reads = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_global &&
                intr->num_components == 3 &&
                nir_src_is_const(intr->src[0]) &&
                nir_src_as_uint(intr->src[0]) == 0x10002000050ull)
               bounds_reads++;
         }
      }
      assert(bounds_reads);
   }
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            assert(!nir_intrinsic_is_ray_query(intr->intrinsic));
            if (intr->intrinsic == nir_intrinsic_store_scs_base_agx &&
                nir_intrinsic_base(intr) == 7) {
               /* Native Gen1 clears pending bit6, without forcing bit4. */
               assert(nir_def_instr(intr->src[0].ssa)->type == nir_instr_type_alu);
               nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(intr->src[0].ssa));
               assert(alu->op == nir_op_iand);
               assert(nir_src_is_const(alu->src[1].src));
               assert(nir_src_as_uint(alu->src[1].src) == 0xffbf);
            }
            if (intr->intrinsic == nir_intrinsic_store_scs_uniform_agx &&
                nir_intrinsic_component(intr) == 2 &&
                nir_intrinsic_base(intr) >= 256 && nir_intrinsic_base(intr) <= 288) {
               /* The native G17P packet and GPU register probe establish
                * twelve dwords (48 bytes) of saved context per resident warp. */
               nir_instr *index = nir_def_instr(intr->src[2].ssa);
               assert(index->type == nir_instr_type_alu);
               nir_alu_instr *mul = nir_instr_as_alu(index);
               assert(mul->op == nir_op_imul);
               assert(nir_src_is_const(mul->src[1].src));
               assert(nir_src_as_uint(mul->src[1].src) == 12);
            }
            if (intr->intrinsic == nir_intrinsic_load_scs_base_agx &&
                nir_intrinsic_base(intr) == 263)
               assert(nir_intrinsic_flags(intr) == 8);
            if (intr->intrinsic == nir_intrinsic_continue_ray_gen1_agx)
               assert(intr->src[0].ssa->bit_size == 32);
         }
      }
   }
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17P, .encoding = AGX_ISA_ENCODING_AGX3B},
      .no_preamble = true,
      .has_scratch = true,
   };
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(b.shader, &key, &part));
   assert(part.info.uses_ray_traversal && part.info.main_size);
   free(part.binary);
   ralloc_free(b.shader);
   printf("Gen1_STANDARD_RAY_QUERY_PASS count=%u array=%u candidates=%u wrapper=%u\n", count, array, candidates, wrapper);
}

int main(void) {
 setbuf(stdout, NULL);
 glsl_type_singleton_init_or_ref();
 standard_ray_query(1, 0, false, false, false);
 standard_ray_query(2, 0, false, false, false);
 standard_ray_query(8, 0, false, false, false);
 standard_ray_query(4, 1, false, false, false);
 standard_ray_query(8, 1, false, false, false);
 standard_ray_query(8, 2, false, false, false);
 standard_ray_query(1, 0, true, false, false);
 standard_ray_query(8, 2, true, false, false);
 standard_ray_query(1, 0, true, true, false);
 standard_ray_query(1, 0, true, true, true);
 glsl_type_singleton_decref();
 return 0;
}
