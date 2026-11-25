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

/* The M5 store window record (hk_native_descriptor.h). */
static void
check_record(void)
{
   uint32_t rec[4];

   /* dEQP-VK.sparse_resources.shader_intrinsics 2d_array 512x256x6, r32ui,
    * level 9 (1x1): 6 << 9 = 3072 slices. A page-aligned layer of a full
    * 512x256 r32ui chain is 704 KiB; the level is one 128-byte line. */
   const uint64_t stride_B = 704 * 1024;
   CHECK(hk_pack_m5_store_window(stride_B, 128, 4, 1, 1, 1, 1, 6, rec) ==
         (5 * stride_B / 4 + 32 + HK_M5_STORE_WINDOW_WIDTH - 1) /
            HK_M5_STORE_WINDOW_WIDTH);
   CHECK(rec[0] == stride_B / 4);
   CHECK(rec[1] == 0);
   CHECK(rec[2] == (1 | (1 << 16)));
   CHECK(rec[3] == 6);

   /* cube_array 256x256, 6 cubes (36 faces), level 6 (4x4 with 4x4 tiles). */
   CHECK(hk_pack_m5_store_window(stride_B, 128, 4, 4, 4, 4, 4, 36, rec));
   CHECK(rec[1] == (2 | (2 << 4)));
   CHECK(rec[2] == (4 | (4 << 16)));
   CHECK(rec[3] == 36);

   /* A 32x16 tile of a 16-byte format: non-square tiles keep both logs. */
   CHECK(hk_pack_m5_store_window(1 << 20, 1 << 16, 16, 64, 64, 32, 16, 3,
                                 rec));
   CHECK(rec[0] == (1 << 20) / 16);
   CHECK(rec[1] == (5 | (4 << 4)));

   /* The window must reach every layer: 2^29 elements at most. */
   CHECK(!hk_pack_m5_store_window((uint64_t)1 << 28, 1 << 20, 1, 64, 64, 16,
                                  16, 3, rec));
   CHECK(hk_pack_m5_store_window((uint64_t)1 << 26, 1 << 20, 1, 64, 64, 16,
                                 16, 3, rec));
   /* Layer strides must be whole elements, tiles powers of two. */
   CHECK(!hk_pack_m5_store_window(4100, 64, 8, 4, 4, 4, 4, 3, rec));
   CHECK(!hk_pack_m5_store_window(4096, 64, 4, 4, 4, 3, 4, 3, rec));
   CHECK(!hk_pack_m5_store_window(4096, 64, 4, 4, 4, 4, 4, 0, rec));

   /* A one-layer view of a compressed image: a 2D PBE. Stores see a layer
    * count of one and the flag; atomics see a zero layer stride. */
   CHECK(hk_pack_m5_store_2d(256, 128, rec));
   CHECK(rec[0] == 0);
   CHECK(rec[1] == HK_M5_STORE_2D);
   CHECK(rec[2] == (256 | (128 << 16)));
   CHECK(rec[3] == 1);
   CHECK(!hk_pack_m5_store_2d(65536, 1, rec));
   CHECK(!hk_pack_m5_store_2d(1, 0, rec));
   /* The flag is clear of the window's tile logs. */
   CHECK(hk_pack_m5_store_window(1 << 20, 1 << 16, 16, 64, 64, 1 << 15,
                                 1 << 15, 3, rec));
   CHECK(!(rec[1] & HK_M5_STORE_2D));
}

struct counts {
   unsigned image_stores, image_stores_2d, image_atomics, global_atomics;
   unsigned ifs;
};

static struct counts
count(nir_shader *s)
{
   struct counts c = {0};
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *prev = nir_cf_node_prev(&block->cf_node);
         if (prev && prev->type == nir_cf_node_if)
            c.ifs++;
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_bindless_image_store:
               c.image_stores++;
               if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_2D &&
                   !nir_intrinsic_image_array(intr))
                  c.image_stores_2d++;
               break;
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_bindless_image_atomic_swap:
               c.image_atomics++;
               break;
            case nir_intrinsic_global_atomic:
            case nir_intrinsic_global_atomic_swap:
               c.global_atomics++;
               break;
            default:
               break;
            }
         }
      }
   }
   return c;
}

/* One store and one atomic on an image of the given dimension. With the
 * window lowering, layered ones check the record: a 2D store through the
 * window and the native store or atomic. Everything else is unchanged. */
static void
check_lowering(enum glsl_sampler_dim dim, bool array, bool window)
{
   const unsigned generation = 10;
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
      .array_size = 4, .stride = HK_G17P_STORAGE_STRIDE, .offset = 0,
   };
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "store window");
   const struct glsl_type *type =
      glsl_image_type(dim, array, GLSL_TYPE_UINT);
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image, glsl_array_type(type, 4, 0), "img");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;

   nir_deref_instr *arr = nir_build_deref_var(&b, var);
   nir_def *img = &nir_build_deref_array(&b, arr,
      nir_load_local_invocation_index(&b))->def;
   nir_def *coord = nir_load_global_invocation_id(&b, 32);
   coord = nir_pad_vector(&b, coord, 4);
   nir_def *sample = nir_undef(&b, 1, 32);
   nir_def *lod = nir_imm_int(&b, 0);

   nir_image_deref_store(&b, img, coord, sample, nir_imm_ivec4(&b, 1, 0, 0, 1),
      lod, .image_dim = dim, .image_array = array,
      .format = PIPE_FORMAT_R32_UINT, .src_type = nir_type_uint32);
   nir_def *old = nir_image_deref_atomic(&b, 32, img, coord, sample,
      nir_imm_int(&b, 1), .image_dim = dim, .image_array = array,
      .format = PIPE_FORMAT_R32_UINT, .atomic_op = nir_atomic_op_iadd);
   nir_store_global(&b, old, nir_imm_int64(&b, 0x10004000000ull),
      .align_mul = 4, .write_mask = 1);

   CHECK(hk_nir_lower_descriptors(b.shader, &vk_robustness_disabled, 1,
      layouts, false, false, 0, false, false, false, false, window, false));
   nir_validate_shader(b.shader, "after hk_nir_lower_descriptors");

   const bool layered = dim == GLSL_SAMPLER_DIM_CUBE || array;
   const bool lowered = window && layered;
   struct counts c = count(b.shader);
   CHECK(c.image_stores == (lowered ? 2 : 1));
   CHECK(c.image_stores_2d ==
         (lowered ? 1 : (dim == GLSL_SAMPLER_DIM_2D && !array)));
   CHECK(c.image_atomics == 1);
   CHECK(c.global_atomics == (lowered ? 1 : 0));
   /* Per lowered op: the record check and the bounds check. */
   CHECK(c.ifs == (lowered ? 4 : 0));

   ralloc_free(b.shader);
   free(layout);
}

static void
check_m5_chip_predicates(void)
{
   CHECK(agx_chip_g17_m5(AGX_CHIP_G17G));
   CHECK(agx_chip_g17_m5(AGX_CHIP_G17S));
   CHECK(agx_chip_g17_m5(AGX_CHIP_G17C));
   CHECK(!agx_chip_g17_m5(AGX_CHIP_G17P));
   CHECK(!agx_chip_g17_m5(AGX_CHIP_G18G));

   CHECK(agx_chip_g17p_or_m5(AGX_CHIP_G17P));
   CHECK(agx_chip_g17p_or_m5(AGX_CHIP_G17G));
   CHECK(agx_chip_g17p_or_m5(AGX_CHIP_G17S));
   CHECK(agx_chip_g17p_or_m5(AGX_CHIP_G17C));
   CHECK(!agx_chip_g17p_or_m5(AGX_CHIP_G16G));
   CHECK(!agx_chip_g17p_or_m5(AGX_CHIP_G18G));
}

int
main(void)
{
   check_record();
   check_m5_chip_predicates();

   glsl_type_singleton_init_or_ref();
   for (unsigned w = 0; w < 2; ++w) {
      check_lowering(GLSL_SAMPLER_DIM_2D, true, w);
      check_lowering(GLSL_SAMPLER_DIM_CUBE, false, w);
      check_lowering(GLSL_SAMPLER_DIM_CUBE, true, w);
      check_lowering(GLSL_SAMPLER_DIM_1D, true, w);
      check_lowering(GLSL_SAMPLER_DIM_2D, false, w);
      check_lowering(GLSL_SAMPLER_DIM_3D, false, w);
   }
   glsl_type_singleton_decref();
   return 0;
}
