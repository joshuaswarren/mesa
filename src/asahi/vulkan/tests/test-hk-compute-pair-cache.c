/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A depthwise or GEMM compute shader fuses with the clamp after it at
 * submission by compiling a variant from its producer state: its NIR before
 * hk_lower_nir(), its compile options and its set layouts. A shader that
 * comes from a pipeline cache or a shader binary must bring that state with
 * it, or it is never fused. Round-trips descriptor set layouts and producers
 * through their serializers and checks that the result compiles exactly what
 * the original does: the same NIR, options, and layouts by the hash the
 * shader cache keys them with.
 */
#undef NDEBUG
#include "hk_cmd_buffer.h"
#include "hk_compute_pair.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"

#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_bo.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct hk_descriptor_set_layout *
create_layout(struct hk_device *dev, VkDescriptorSetLayoutCreateFlags flags,
              const VkDescriptorSetLayoutBinding *bindings,
              uint32_t binding_count)
{
   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = flags,
      .bindingCount = binding_count,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout handle = VK_NULL_HANDLE;
   assert(hk_CreateDescriptorSetLayout(hk_device_to_handle(dev), &info, NULL,
                                       &handle) == VK_SUCCESS);
   return hk_descriptor_set_layout_from_handle(handle);
}

static void
check_layout_equal(const struct hk_descriptor_set_layout *a,
                   const struct hk_descriptor_set_layout *b)
{
   assert(a->vk.flags == b->vk.flags);
   assert(a->vk.dynamic_descriptor_count == b->vk.dynamic_descriptor_count);
   assert(!memcmp(a->vk.blake3, b->vk.blake3, sizeof(a->vk.blake3)));
   assert(a->descriptor_generation == b->descriptor_generation);
   assert(a->native_sampler_indices == b->native_sampler_indices);
   assert(a->non_variable_descriptor_buffer_size ==
          b->non_variable_descriptor_buffer_size);
   assert(a->max_buffer_size == b->max_buffer_size);
   assert(a->binding_count == b->binding_count);
   for (uint32_t i = 0; i < a->binding_count; ++i) {
      assert(a->binding[i].type == b->binding[i].type);
      assert(a->binding[i].flags == b->binding[i].flags);
      assert(a->binding[i].array_size == b->binding[i].array_size);
      assert(a->binding[i].offset == b->binding[i].offset);
      assert(a->binding[i].stride == b->binding[i].stride);
      assert(a->binding[i].dynamic_buffer_index ==
             b->binding[i].dynamic_buffer_index);
      assert(b->binding[i].immutable_samplers == NULL);
   }
}

static struct hk_descriptor_set_layout *
round_trip_layout(struct hk_device *dev,
                  const struct hk_descriptor_set_layout *layout)
{
   struct blob blob;
   blob_init(&blob);
   assert(hk_descriptor_set_layout_serialize(&blob, layout));
   assert(!blob.out_of_memory);

   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct hk_descriptor_set_layout *copy =
      hk_descriptor_set_layout_deserialize(dev, &reader);
   assert(copy && reader.current == reader.end && !reader.overrun);
   check_layout_equal(layout, copy);

   /* A blob cut short holds no layout. */
   blob_reader_init(&reader, blob.data, blob.size - 1);
   assert(hk_descriptor_set_layout_deserialize(dev, &reader) == NULL);

   blob_finish(&blob);
   return copy;
}

/* A compute shader writing three storage buffers' worth of loads to a
 * fourth, in the form hk_compute_pair_producer_create() takes. */
static nir_shader *
build_producer_nir(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "compute_pair_producer");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.num_ssbos = 4;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *offset = nir_imul_imm(&b, id, 4);
   nir_def *sum = nir_imm_float(&b, 0.0);
   for (unsigned i = 0; i < 3; ++i) {
      nir_def *v = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, i), offset,
                                 .align_mul = 4);
      sum = nir_fadd(&b, sum, v);
   }
   nir_store_ssbo(&b, sum, nir_imm_int(&b, 3), offset, .align_mul = 4);
   return b.shader;
}

static void
nir_bytes(const nir_shader *nir, struct blob *out)
{
   blob_init(out);
   nir_serialize(out, nir, false);
   assert(!out->out_of_memory);
}

static void
check_producer_equal(const struct hk_compute_pair_producer *a,
                     const struct hk_compute_pair_producer *b)
{
   assert(!memcmp(&a->info, &b->info, sizeof(a->info)));
   assert(a->flags == b->flags);
   assert(!memcmp(&a->robustness, &b->robustness, sizeof(a->robustness)));
   assert(a->next_stage_mask == b->next_stage_mask);
   assert(a->features == b->features);
   assert(a->set_layout_count == b->set_layout_count);
   for (uint32_t s = 0; s < a->set_layout_count; ++s) {
      assert(!a->set_layouts[s] == !b->set_layouts[s]);
      if (a->set_layouts[s])
         check_layout_equal(vk_to_hk_descriptor_set_layout(a->set_layouts[s]),
                            vk_to_hk_descriptor_set_layout(b->set_layouts[s]));
   }
   assert(b->variant_count == 0);

   struct blob na, nb;
   nir_bytes(a->nir, &na);
   nir_bytes(b->nir, &nb);
   assert(na.size == nb.size && !memcmp(na.data, nb.data, na.size));
   blob_finish(&na);
   blob_finish(&nb);
}

static void
check_producer(struct hk_device *dev, struct hk_descriptor_set_layout *set0,
               struct hk_descriptor_set_layout *set2,
               struct hk_descriptor_set_layout *sampled)
{
   const struct vk_pipeline_robustness_state robustness = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT,
      .null_uniform_buffer_descriptor = true,
      .null_storage_buffer_descriptor = true,
   };
   /* Set 1 is unused: a hole in the pipeline layout. */
   struct vk_descriptor_set_layout *layouts[3] = {&set0->vk, NULL, &set2->vk};
   nir_shader *nir = build_producer_nir();
   const struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_COMPUTE,
      .flags = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
      .next_stage_mask = 0,
      .nir = nir,
      .robustness = &robustness,
      .set_layout_count = ARRAY_SIZE(layouts),
      .set_layouts = layouts,
   };
   const struct hk_compute_pair_info pair = {
      .version = HK_COMPUTE_PAIR_INFO_VERSION,
      .role = HK_COMPUTE_PAIR_DEPTHWISE,
   };

   struct hk_compute_pair_producer *p =
      hk_compute_pair_producer_create(dev, &info, 0x15, &pair, nir);
   assert(p);

   struct blob blob;
   blob_init(&blob);
   assert(hk_compute_pair_producer_serialize(&blob, p));
   assert(!blob.out_of_memory);

   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct hk_compute_pair_producer *q =
      hk_compute_pair_producer_deserialize(dev, &reader, &pair);
   assert(q && reader.current == reader.end && !reader.overrun);
   check_producer_equal(p, q);

   /* The restored producer writes the same bytes again. */
   struct blob again;
   blob_init(&again);
   assert(hk_compute_pair_producer_serialize(&again, q));
   assert(again.size == blob.size && !memcmp(again.data, blob.data, blob.size));
   blob_finish(&again);

   /* Only depthwise and GEMM shaders have producers. */
   const struct hk_compute_pair_info clamp = {
      .version = HK_COMPUTE_PAIR_INFO_VERSION,
      .role = HK_COMPUTE_PAIR_CLAMP,
   };
   blob_reader_init(&reader, blob.data, blob.size);
   assert(hk_compute_pair_producer_deserialize(dev, &reader, &clamp) == NULL);
   struct hk_compute_pair_producer *c =
      hk_compute_pair_producer_create(dev, &info, 0x15, &clamp, nir);
   assert(c);
   struct blob none;
   blob_init(&none);
   assert(!hk_compute_pair_producer_serialize(&none, c));
   blob_finish(&none);
   hk_compute_pair_producer_unref(c);

   /* A blob cut short anywhere before its NIR holds no producer, whether
    * the cut falls in its options, in a set layout or between them.
    * (nir_deserialize() takes only what nir_serialize() wrote: no cut goes
    * into the NIR.) */
   struct blob nir_only;
   nir_bytes(p->nir, &nir_only);
   assert(nir_only.size + 16 < blob.size);
   for (size_t size = 0; size + nir_only.size + 16 < blob.size; ++size) {
      blob_reader_init(&reader, blob.data, size);
      assert(hk_compute_pair_producer_deserialize(dev, &reader, &pair) == NULL);
   }
   blob_finish(&nir_only);

   hk_compute_pair_producer_unref(q);
   blob_finish(&blob);

   /* A producer over a layout with immutable samplers is not written. */
   layouts[1] = &sampled->vk;
   struct hk_compute_pair_producer *s =
      hk_compute_pair_producer_create(dev, &info, 0x15, &pair, nir);
   assert(s);
   blob_init(&blob);
   assert(!hk_compute_pair_producer_serialize(&blob, s));
   blob_finish(&blob);
   hk_compute_pair_producer_unref(s);

   hk_compute_pair_producer_unref(p);
   ralloc_free(nir);
}

static void
check_chip(enum agx_chip chip)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   assert(pdev && dev);
   pdev->dev.chip = chip;
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);

   /* The set of a depthwise convolution: input, weights, bias, output. */
   const VkDescriptorSetLayoutBinding storage[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   /* Holes, arrays, dynamic buffers and images. */
   const VkDescriptorSetLayoutBinding mixed[] = {
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 2, VK_SHADER_STAGE_ALL, NULL},
      {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3, VK_SHADER_STAGE_ALL, NULL},
      {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, NULL},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, VK_SHADER_STAGE_ALL, NULL},
      {7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_ALL, NULL},
   };

   struct hk_descriptor_set_layout *set0 = create_layout(dev, 0, storage, 4);
   struct hk_descriptor_set_layout *set2 =
      create_layout(dev, 0, mixed, ARRAY_SIZE(mixed));
   struct hk_descriptor_set_layout *push = create_layout(
      dev, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, storage, 4);

   struct hk_descriptor_set_layout *layouts[] = {set0, set2, push};
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      struct hk_descriptor_set_layout *copy = round_trip_layout(dev, layouts[i]);
      vk_descriptor_set_layout_unref(&dev->vk, &copy->vk);
   }

   /* A layout with an immutable sampler. */
   struct hk_rc_sampler hw = {.index = 1};
   struct hk_sampler sampler = {
      .plane_count = 1,
      .planes = {{.hw = &hw}},
   };
   vk_object_base_init(NULL, &sampler.vk.base, VK_OBJECT_TYPE_SAMPLER);
   const VkSampler sampler_handle = hk_sampler_to_handle(&sampler);
   const VkDescriptorSetLayoutBinding immutable[] = {
      {0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_ALL, &sampler_handle},
   };
   struct hk_descriptor_set_layout *sampled =
      create_layout(dev, 0, immutable, 1);
   struct blob blob;
   blob_init(&blob);
   assert(!hk_descriptor_set_layout_serialize(&blob, sampled));
   assert(blob.size == 0);
   blob_finish(&blob);

   check_producer(dev, set0, set2, sampled);

   vk_descriptor_set_layout_unref(&dev->vk, &sampled->vk);
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i)
      vk_descriptor_set_layout_unref(&dev->vk, &layouts[i]->vk);
   free(dev);
   free(pdev);
}

static void
check_fused_m5_dispatch(enum agx_chip chip, uint32_t clusters, uint32_t dies,
                        bool expect_ok, uint32_t expected_tail)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_shader *s = calloc(1, sizeof(*s));
   struct hk_linked_shader *linked = calloc(1, sizeof(*linked));
   assert(pdev && dev && s && linked);

   pdev->dev.chip = chip;
   pdev->vk.properties.maxComputeWorkGroupInvocations = 1024;
   dev->vk.physical = &pdev->vk;
   dev->dev.chip = chip;
   dev->dev.shader_base = UINT64_C(0x10000000000);
   dev->dev.params.num_clusters_total = clusters;
   dev->dev.params.num_dies = dies;

   struct agx_va shader_va = {
      .addr = dev->dev.shader_base + 0x400,
      .size_B = 0x100,
   };
   struct agx_bo shader_bo = {
      .va = &shader_va,
   };
   linked->b.bo = &shader_bo;
   linked->b.size = 64;
   linked->b.nr_gprs = 8;

   s->only_linked = linked;
   s->info.stage = MESA_SHADER_COMPUTE;
   s->b.info.stage = MESA_SHADER_COMPUTE;
   s->b.info.workgroup_size[0] = 64;
   s->b.info.workgroup_size[1] = 1;
   s->b.info.workgroup_size[2] = 1;
   s->b.info.agx3_staging_known = true;
   s->b.info.agx3_staging_halfs = 2;

   struct hk_compute_pair_dispatch p = {
      .exec_local = {64, 1, 1},
      .exec_threads = {64, 1, 1},
      .root_va = UINT64_C(0x200000000),
   };

   uint8_t cs_bytes[HK_CS_SPACE_PADDING + 256];
   memset(cs_bytes, 0, sizeof(cs_bytes));
   struct hk_cs cs = {
      .type = HK_CS_CDM,
      .start = cs_bytes,
      .current = cs_bytes,
      .end = cs_bytes + sizeof(cs_bytes),
   };

   uint8_t ioto_buf[512];
   size_t ioto_used = 0;
   const bool ok = hk_emit_fused_compute_dispatch(
      dev, &cs, s, &p, ioto_buf, dev->dev.shader_base + 0x8000,
      sizeof(ioto_buf), &ioto_used);
   assert(ok == expect_ok);
   if (expect_ok) {
      assert(ioto_used > 0);
      assert((const uint8_t *)cs.current >=
             (const uint8_t *)cs.start + sizeof(uint32_t));
      uint32_t tail = 0;
      memcpy(&tail, (const uint8_t *)cs.current - sizeof(tail), sizeof(tail));
      assert(tail == expected_tail);
   }

   free(linked);
   free(s);
   free(dev);
   free(pdev);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const enum agx_chip chips[] = {
      AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17D,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i)
      check_chip(chips[i]);
   glsl_type_singleton_decref();

   check_fused_m5_dispatch(AGX_CHIP_G17G, 2, 1, true, 0x60000160);
   check_fused_m5_dispatch(AGX_CHIP_G17G, 2, 2, true, 0x60000160);
   check_fused_m5_dispatch(AGX_CHIP_G17D, 8, 2, true, 0x6000017c);
   check_fused_m5_dispatch(AGX_CHIP_G17G, 0, 1, false, 0);
   check_fused_m5_dispatch(AGX_CHIP_G17D, 0, 2, false, 0);

   printf("PASS: compute-pair producers and their set layouts round-trip on "
          "%zu chips; fused M5 dispatch verified\n",
          ARRAY_SIZE(chips));
   return 0;
}
