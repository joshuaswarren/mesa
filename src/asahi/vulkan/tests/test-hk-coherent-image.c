/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set_layout.h"
#include "hk_native_descriptor.h"
#include "hk_shader.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "vk_pipeline.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
 fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

struct counts {
   unsigned image_loads, image_stores, global_loads, global_stores;
   unsigned coherent_global;
};

static struct counts
count(nir_shader *s)
{
   struct counts c = {0};
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_bindless_image_load:
               c.image_loads++;
               break;
            case nir_intrinsic_bindless_image_store:
               c.image_stores++;
               break;
            case nir_intrinsic_load_global:
               c.global_loads++;
               c.coherent_global +=
                  !!(nir_intrinsic_access(intr) & ACCESS_COHERENT);
               break;
            case nir_intrinsic_store_global:
               c.global_stores++;
               c.coherent_global +=
                  !!(nir_intrinsic_access(intr) & ACCESS_COHERENT);
               break;
            default:
               break;
            }
         }
      }
   }
   return c;
}

/* One coherent and one plain load and store of an r32ui 2D storage image.
 * With the multi-cluster AGX2 option only the coherent pair becomes device
 * memory accesses, and those stay coherent; without it nothing does. */
static void
check(bool coherent_images_in_memory, enum pipe_format format)
{
   const unsigned generation = 8;
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(generation);
   CHECK(native.stride);

   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = generation;
   layout->binding_count = 1;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .array_size = 1, .stride = native.stride, .offset = 0,
   };
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "coherent image");
   const struct glsl_type *type =
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT);
   nir_variable *var = nir_variable_create(b.shader, nir_var_image, type,
                                           "img");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = format;

   nir_def *img = &nir_build_deref_var(&b, var)->def;
   nir_def *coord = nir_imm_ivec4(&b, 1, 2, 0, 0);
   nir_def *sample = nir_undef(&b, 1, 32);
   nir_def *lod = nir_imm_int(&b, 0);

   nir_def *plain = nir_image_deref_load(&b, 4, 32, img, coord, sample, lod,
      .image_dim = GLSL_SAMPLER_DIM_2D, .format = format,
      .access = ACCESS_RESTRICT, .dest_type = nir_type_uint32);
   nir_def *coherent = nir_image_deref_load(&b, 4, 32, img, coord, sample,
      lod, .image_dim = GLSL_SAMPLER_DIM_2D, .format = format,
      .access = ACCESS_COHERENT, .dest_type = nir_type_uint32);
   nir_image_deref_store(&b, img, coord, sample, plain, lod,
      .image_dim = GLSL_SAMPLER_DIM_2D, .format = format,
      .access = ACCESS_RESTRICT, .src_type = nir_type_uint32);
   nir_image_deref_store(&b, img, coord, sample, coherent, lod,
      .image_dim = GLSL_SAMPLER_DIM_2D, .format = format,
      .access = ACCESS_COHERENT, .src_type = nir_type_uint32);

   CHECK(hk_nir_lower_descriptors(b.shader, &vk_robustness_disabled, 1,
      layouts, false, false, 0, false, false, false,
      coherent_images_in_memory, false, false));
   nir_validate_shader(b.shader, "after hk_nir_lower_descriptors");

   const bool memory = coherent_images_in_memory &&
                       format == PIPE_FORMAT_R32_UINT;
   struct counts c = count(b.shader);
   CHECK(c.image_loads == (memory ? 1 : 2));
   CHECK(c.image_stores == (memory ? 1 : 2));
   CHECK(c.global_loads == (memory ? 1 : 0));
   CHECK(c.global_stores == (memory ? 1 : 0));
   CHECK(c.coherent_global == (memory ? 2 : 0));

   ralloc_free(b.shader);
   free(layout);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check(false, PIPE_FORMAT_R32_UINT);
   check(true, PIPE_FORMAT_R32_UINT);
   /* Texels that are not their memory words keep the texture/PBE path. */
   check(true, PIPE_FORMAT_R8G8B8A8_UINT);
   glsl_type_singleton_decref();
   return 0;
}
