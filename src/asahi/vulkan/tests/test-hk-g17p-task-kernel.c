/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_shader.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
 fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static void
payload_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   const unsigned comp = glsl_get_bit_size(type) / 8;
   *size = comp * glsl_get_vector_elements(type);
   *align = comp;
}

/* The task shader of dEQP-VK.mesh_shader.ext.api.*.with_task_shader:
 *
 *    struct TaskData { uint blockNumber; uint blockRow; };
 *    taskPayloadSharedEXT TaskData td;
 *    td.blockNumber = gl_DrawID; td.blockRow = gl_WorkGroupID.x;
 *    EmitMeshTasksEXT(1, 1, 1);
 *
 * with the payload I/O lowered as hk_lower_nir() does.
 */
static nir_shader *
build_task_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TASK,
      &agx_nir_options, "task payload");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;

   const struct glsl_struct_field fields[] = {
      {.type = glsl_uint_type(), .name = "blockNumber", .location = -1},
      {.type = glsl_uint_type(), .name = "blockRow", .location = -1},
   };
   nir_variable *td = nir_variable_create(b.shader, nir_var_mem_task_payload,
      glsl_struct_type(fields, 2, "TaskData", false), "td");
   nir_deref_instr *var = nir_build_deref_var(&b, td);
   nir_store_deref(&b, nir_build_deref_struct(&b, var, 0),
                   nir_load_draw_id(&b), 1);
   nir_store_deref(&b, nir_build_deref_struct(&b, var, 1),
                   nir_channel(&b, nir_load_workgroup_id(&b), 0), 1);
   nir_launch_mesh_workgroups_with_payload_deref(&b, nir_imm_ivec3(&b, 1, 1, 1),
                                                 &var->def);

   NIR_PASS(_, b.shader, nir_lower_vars_to_explicit_types,
            nir_var_mem_task_payload, payload_var_info);
   NIR_PASS(_, b.shader, nir_lower_explicit_io, nir_var_mem_task_payload,
            nir_address_format_32bit_offset);
   nir_validate_shader(b.shader, "task shader input");
   return b.shader;
}

static unsigned
count_payload_accesses(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_task_payload:
            case nir_intrinsic_store_task_payload:
            case nir_intrinsic_task_payload_atomic:
            case nir_intrinsic_task_payload_atomic_swap:
            case nir_intrinsic_launch_mesh_workgroups:
               n++;
               break;
            default:
               break;
            }
         }
      }
   }
   return n;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   nir_shader *s = build_task_shader();
   unsigned payload_vars = 0;
   nir_foreach_variable_with_modes(v, s, nir_var_mem_task_payload)
      payload_vars++;
   CHECK(payload_vars == 1);
   CHECK(s->info.task_payload_size == 8);

   hk_g17p_lower_task_kernel(s);

   /* A compute shader may not declare task payload, and no task payload
    * access may remain. */
   CHECK(s->info.stage == MESA_SHADER_COMPUTE);
   nir_foreach_variable_with_modes(v, s, nir_var_mem_task_payload)
      CHECK(!"task payload variable in a compute shader");
   CHECK(count_payload_accesses(s) == 0);
   nir_validate_shader(s, "after hk_g17p_lower_task_kernel");

   /* The driver's next step (mesa#590 failed here, at the first validated
    * pass, nir_lower_load_const_to_scalar). */
   agx_preprocess_nir_native(s, AGX_CHIP_G17P);
   nir_validate_shader(s, "after agx_preprocess_nir_native");

   ralloc_free(s);
   glsl_type_singleton_decref();
   printf("G17P task kernel lowering: 1 passed\n");
   return 0;
}
