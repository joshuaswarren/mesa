/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_pipeline_binary import of a ray tracing pipeline. An exported
 * binary must create the pipeline again, with the same group handles. A
 * binary whose shader groups name a shader past the pipeline's stages, or a
 * shader of a stage the group's slot does not take, must be refused, even
 * with its key recomputed: the key is the binary's unkeyed BLAKE3, so an
 * application can always reseal its edit.
 *
 * Requires the no-op DRM shim and a device with native ray tracing pipeline
 * binaries. Creates pipelines and binaries; never records or submits work.
 * Exits 77 (skip) when the device lacks the features.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "hk_ray_tracing_core.h"
#include "compiler/shader_info.h"
#include "util/mesa-blake3.h"
#include "hk_vulkan_rt_binary_groups_shaders.h"

#define REQUIRE(x)                                                             \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#define VK_CHECK(x)                                                            \
   do {                                                                        \
      VkResult r_ = (x);                                                       \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #x, r_);                                  \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

/* hk_rt_binary_serialize(): magic, version, UUID, binary key, stage and
 * group counts, stage mask, executable kind, stack sizes, the pipeline stack
 * size and then the groups, every field 4-byte aligned. */
#define BIN_MAGIC        0x42524b48u
#define BIN_STAGES_B     (8 + VK_UUID_SIZE + 32)
#define BIN_GROUPS_N_B   (BIN_STAGES_B + 4)
#define BIN_GROUPS_B     (BIN_STAGES_B + 16 + sizeof(struct hk_rt_stack_sizes) + 4)

enum { STAGE_RGEN, STAGE_RMISS, STAGE_RCHIT, STAGE_RAHIT, STAGE_COUNT };
enum { GROUP_RGEN, GROUP_MISS, GROUP_HIT, GROUP_COUNT };

static VkDevice dev;
static VkPipelineLayout layout;
static VkShaderModule modules[STAGE_COUNT];
static uint32_t handle_size;
static unsigned failures;

#define GET(name) static PFN_##name name##_
GET(vkCreateRayTracingPipelinesKHR);
GET(vkGetRayTracingShaderGroupHandlesKHR);
GET(vkGetRayTracingShaderGroupStackSizeKHR);
GET(vkCreatePipelineBinariesKHR);
GET(vkGetPipelineBinaryDataKHR);
GET(vkDestroyPipelineBinaryKHR);
#undef GET

struct binary {
   uint8_t *data;
   size_t size;
};

static VkShaderModule
module(const uint32_t *code, size_t size)
{
   const VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule m;
   VK_CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

/* Create the pipeline from SPIR-V, or from one binary when it is given. */
static VkResult
create_pipeline(VkPipelineBinaryKHR binary, VkPipeline *pipeline)
{
   static const VkShaderStageFlagBits kinds[STAGE_COUNT] = {
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      VK_SHADER_STAGE_MISS_BIT_KHR,
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
   };
   VkPipelineShaderStageCreateInfo stages[STAGE_COUNT];
   for (unsigned i = 0; i < STAGE_COUNT; ++i) {
      stages[i] = (VkPipelineShaderStageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = kinds[i],
         .module = modules[i],
         .pName = "main",
      };
   }
   const VkRayTracingShaderGroupCreateInfoKHR groups[GROUP_COUNT] = {
      {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
         .generalShader = STAGE_RGEN,
         .closestHitShader = VK_SHADER_UNUSED_KHR,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR,
      },
      {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
         .generalShader = STAGE_RMISS,
         .closestHitShader = VK_SHADER_UNUSED_KHR,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR,
      },
      {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
         .generalShader = VK_SHADER_UNUSED_KHR,
         .closestHitShader = STAGE_RCHIT,
         .anyHitShader = STAGE_RAHIT,
         .intersectionShader = VK_SHADER_UNUSED_KHR,
      },
   };
   const VkPipelineBinaryInfoKHR binaries = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
      .binaryCount = 1,
      .pPipelineBinaries = &binary,
   };
   const VkPipelineCreateFlags2CreateInfoKHR flags = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
      .pNext = binary ? &binaries : NULL,
      .flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR,
   };
   const VkRayTracingPipelineCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext = &flags,
      .stageCount = binary ? 0 : STAGE_COUNT,
      .pStages = binary ? NULL : stages,
      .groupCount = binary ? 0 : GROUP_COUNT,
      .pGroups = binary ? NULL : groups,
      .maxPipelineRayRecursionDepth = 1,
      .layout = layout,
   };
   *pipeline = VK_NULL_HANDLE;
   return vkCreateRayTracingPipelinesKHR_(dev, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                          1, &ci, NULL, pipeline);
}

static struct binary
export_binary(VkPipeline pipeline)
{
   const VkPipelineBinaryCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
      .pipeline = pipeline,
   };
   VkPipelineBinaryKHR handle;
   VkPipelineBinaryHandlesInfoKHR handles = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
      .pipelineBinaryCount = 1,
      .pPipelineBinaries = &handle,
   };
   VK_CHECK(vkCreatePipelineBinariesKHR_(dev, &ci, NULL, &handles));
   REQUIRE(handles.pipelineBinaryCount == 1);

   const VkPipelineBinaryDataInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
      .pipelineBinary = handle,
   };
   VkPipelineBinaryKeyKHR key = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
   };
   struct binary b = {0};
   VK_CHECK(vkGetPipelineBinaryDataKHR_(dev, &info, &key, &b.size, NULL));
   b.data = malloc(b.size);
   REQUIRE(b.data);
   VK_CHECK(vkGetPipelineBinaryDataKHR_(dev, &info, &key, &b.size, b.data));
   vkDestroyPipelineBinaryKHR_(dev, handle, NULL);
   return b;
}

/* Import binary data under the key an application would compute for it. */
static VkResult
import(const struct binary *b, VkPipeline *pipeline)
{
   VkPipelineBinaryKeyKHR key = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
      .keySize = BLAKE3_OUT_LEN,
   };
   _mesa_blake3_compute(b->data, b->size, key.key);
   const VkPipelineBinaryDataKHR data = {
      .dataSize = b->size,
      .pData = b->data,
   };
   const VkPipelineBinaryKeysAndDataKHR keys = {
      .binaryCount = 1,
      .pPipelineBinaryKeys = &key,
      .pPipelineBinaryData = &data,
   };
   const VkPipelineBinaryCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
      .pKeysAndDataInfo = &keys,
   };
   VkPipelineBinaryKHR handle;
   VkPipelineBinaryHandlesInfoKHR handles = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
      .pipelineBinaryCount = 1,
      .pPipelineBinaries = &handle,
   };
   VK_CHECK(vkCreatePipelineBinariesKHR_(dev, &ci, NULL, &handles));
   const VkResult result = create_pipeline(handle, pipeline);
   vkDestroyPipelineBinaryKHR_(dev, handle, NULL);
   return result;
}

static void
expect(const char *what, bool ok)
{
   printf("%s %s\n", ok ? "PASS" : "FAIL", what);
   if (!ok)
      failures++;
}

/* Edit one stage index of one group in a copy of b and import it. */
static void
expect_refused(const char *what, const struct binary *b, unsigned group,
               size_t field, uint32_t index)
{
   struct binary t = {malloc(b->size), b->size};
   REQUIRE(t.data);
   memcpy(t.data, b->data, b->size);
   memcpy(t.data + BIN_GROUPS_B + group * sizeof(struct hk_ray_tracing_group) +
             field,
          &index, sizeof(index));
   VkPipeline pipeline;
   const VkResult result = import(&t, &pipeline);
   if (result == VK_SUCCESS)
      vkDestroyPipeline(dev, pipeline, NULL);
   printf("INFO %s: result %d\n", what, result);
   expect(what, result != VK_SUCCESS && pipeline == VK_NULL_HANDLE);
   free(t.data);
}

static uint32_t
read32(const struct binary *b, size_t offset)
{
   uint32_t v;
   REQUIRE(offset + sizeof(v) <= b->size);
   memcpy(&v, b->data + offset, sizeof(v));
   return v;
}

/* The byte holding shader_info::stage (a bitfield) in the stage's
 * serialized NIR: after the groups, each stage is a u32 size and its NIR,
 * which nir_serialize() starts with an index-table size, flags, the strings
 * the flags name and then the raw shader_info. */
static size_t
nir_stage_byte(const struct binary *b, unsigned stage)
{
   struct shader_info probe;
   memset(&probe, 0, sizeof(probe));
   probe.stage = 0x5a;
   size_t in_info = sizeof(probe);
   for (size_t i = 0; i < sizeof(probe); ++i) {
      if (((const uint8_t *)&probe)[i] == 0x5a) {
         REQUIRE(in_info == sizeof(probe));
         in_info = i;
      }
   }
   REQUIRE(in_info < sizeof(probe));

   size_t offset = BIN_GROUPS_B + GROUP_COUNT * sizeof(struct hk_ray_tracing_group);
   for (unsigned i = 0;; ++i) {
      offset = (offset + 3) & ~(size_t)3;
      const uint32_t size = read32(b, offset);
      const size_t nir = offset + 4;
      if (i == stage) {
         const uint32_t flags = read32(b, nir + 4);
         size_t at = nir + 8;
         /* NIR_SERIALIZE_SHADER_NAME, _LABEL and _SPEC each add a string. */
         for (unsigned bit = 0; bit < 4; ++bit) {
            if (bit == 2 || !(flags & (1u << bit)))
               continue;
            while (at < b->size && b->data[at])
               ++at;
            ++at;
         }
         REQUIRE(at + in_info < nir + size);
         return at + in_info;
      }
      offset = nir + size;
   }
}

int
main(void)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   VkResult result = vkEnumeratePhysicalDevices(instance, &count, &physical);
   REQUIRE((result == VK_SUCCESS || result == VK_INCOMPLETE) && count == 1);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(physical, &props);
   fprintf(stderr, "Device: %s\n", props.deviceName);

   VkPhysicalDevicePipelineBinaryFeaturesKHR binary_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR,
   };
   VkPhysicalDeviceMaintenance5FeaturesKHR m5_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .pNext = &binary_features,
   };
   VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .pNext = &m5_features,
   };
   VkPhysicalDeviceVulkan12Features v12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &rt_features,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &v12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!rt_features.rayTracingPipeline || !binary_features.pipelineBinaries ||
       !m5_features.maintenance5 || !v12.bufferDeviceAddress) {
      printf("SKIP: no ray tracing pipeline binaries on %s\n",
             props.deviceName);
      return 77;
   }

   VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &rt_props,
   };
   vkGetPhysicalDeviceProperties2(physical, &props2);
   handle_size = rt_props.shaderGroupHandleSize;

   /* Enable only what the test uses. */
   v12 = (VkPhysicalDeviceVulkan12Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &rt_features,
      .bufferDeviceAddress = true,
   };
   rt_features = (VkPhysicalDeviceRayTracingPipelineFeaturesKHR){
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .pNext = &m5_features,
      .rayTracingPipeline = true,
   };
   m5_features.maintenance5 = true;
   binary_features.pipelineBinaries = true;
   binary_features.pNext = NULL;

   const float priority = 1;
   const VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *exts[] = {
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
      VK_KHR_PIPELINE_BINARY_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &v12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = sizeof(exts) / sizeof(exts[0]),
      .ppEnabledExtensionNames = exts,
   };
   VK_CHECK(vkCreateDevice(physical, &dci, NULL, &dev));
#define GET(name)                                                              \
   name##_ = (PFN_##name)vkGetDeviceProcAddr(dev, #name);                      \
   REQUIRE(name##_)
   GET(vkCreateRayTracingPipelinesKHR);
   GET(vkGetRayTracingShaderGroupHandlesKHR);
   GET(vkGetRayTracingShaderGroupStackSizeKHR);
   GET(vkCreatePipelineBinariesKHR);
   GET(vkGetPipelineBinaryDataKHR);
   GET(vkDestroyPipelineBinaryKHR);
#undef GET

   const VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VK_CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   modules[STAGE_RGEN] = module(rt_rgen, sizeof(rt_rgen));
   modules[STAGE_RMISS] = module(rt_rmiss, sizeof(rt_rmiss));
   modules[STAGE_RCHIT] = module(rt_rchit, sizeof(rt_rchit));
   modules[STAGE_RAHIT] = module(rt_rahit, sizeof(rt_rahit));

   VkPipeline compiled;
   VK_CHECK(create_pipeline(VK_NULL_HANDLE, &compiled));
   struct binary b = export_binary(compiled);
   REQUIRE(read32(&b, 0) == BIN_MAGIC);
   REQUIRE(read32(&b, BIN_STAGES_B) == STAGE_COUNT);
   REQUIRE(read32(&b, BIN_GROUPS_N_B) == GROUP_COUNT);
   REQUIRE(BIN_GROUPS_B + GROUP_COUNT * sizeof(struct hk_ray_tracing_group) <=
           b.size);
   printf("INFO binary: %zu bytes\n", b.size);

   /* Round trip: the binary creates the pipeline again, with the same group
    * handles and stack sizes. */
   uint8_t handles[2][GROUP_COUNT * 64];
   REQUIRE(handle_size * GROUP_COUNT <= sizeof(handles[0]));
   VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR_(
      dev, compiled, 0, GROUP_COUNT, handle_size * GROUP_COUNT, handles[0]));
   VkPipeline loaded;
   result = import(&b, &loaded);
   expect("unmodified binary creates the pipeline", result == VK_SUCCESS);
   if (result == VK_SUCCESS) {
      VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR_(
         dev, loaded, 0, GROUP_COUNT, handle_size * GROUP_COUNT, handles[1]));
      expect("loaded pipeline has the same group handles",
             !memcmp(handles[0], handles[1], handle_size * GROUP_COUNT));
      bool same = true;
      for (unsigned g = 0; g < GROUP_COUNT; ++g) {
         for (unsigned s = VK_SHADER_GROUP_SHADER_GENERAL_KHR;
              s <= VK_SHADER_GROUP_SHADER_INTERSECTION_KHR; ++s) {
            same &= vkGetRayTracingShaderGroupStackSizeKHR_(dev, compiled, g, s) ==
                    vkGetRayTracingShaderGroupStackSizeKHR_(dev, loaded, g, s);
         }
      }
      expect("loaded pipeline has the same group stack sizes", same);
      vkDestroyPipeline(dev, loaded, NULL);
   }

   /* Dropping the any-hit shader leaves a valid triangles hit group. */
   {
      struct binary t = {malloc(b.size), b.size};
      REQUIRE(t.data);
      memcpy(t.data, b.data, b.size);
      const uint32_t unused = VK_SHADER_UNUSED_KHR;
      memcpy(t.data + BIN_GROUPS_B +
                GROUP_HIT * sizeof(struct hk_ray_tracing_group) +
                offsetof(struct hk_ray_tracing_group, ahit_index),
             &unused, sizeof(unused));
      result = import(&t, &loaded);
      expect("hit group without its any-hit shader loads", result == VK_SUCCESS);
      if (result == VK_SUCCESS)
         vkDestroyPipeline(dev, loaded, NULL);
      free(t.data);
   }

   const size_t recursive = offsetof(struct hk_ray_tracing_group, recursive_index);
   const size_t ahit = offsetof(struct hk_ray_tracing_group, ahit_index);
   const size_t isec = offsetof(struct hk_ray_tracing_group, isec_index);
   const uint32_t past[] = {STAGE_COUNT, STAGE_COUNT + 1,
                            VK_SHADER_UNUSED_KHR - 1};
   for (unsigned i = 0; i < 3; ++i) {
      char what[96];
      snprintf(what, sizeof(what), "raygen group shader %u", past[i]);
      expect_refused(what, &b, GROUP_RGEN, recursive, past[i]);
      snprintf(what, sizeof(what), "hit group closest hit %u", past[i]);
      expect_refused(what, &b, GROUP_HIT, recursive, past[i]);
      snprintf(what, sizeof(what), "hit group any hit %u", past[i]);
      expect_refused(what, &b, GROUP_HIT, ahit, past[i]);
      snprintf(what, sizeof(what), "hit group intersection %u", past[i]);
      expect_refused(what, &b, GROUP_HIT, isec, past[i]);
   }

   /* A stored shader whose NIR claims a stage that is not a ray tracing
    * stage, or no stage at all, is refused before its stage is used. */
   {
      const size_t at = nir_stage_byte(&b, STAGE_RGEN);
      REQUIRE(b.data[at] == MESA_SHADER_RAYGEN);
      const uint8_t bad[] = {MESA_SHADER_COMPUTE, MESA_SHADER_FRAGMENT,
                             MESA_SHADER_KERNEL, 200, 0xff};
      for (unsigned i = 0; i < ARRAY_SIZE(bad); ++i) {
         struct binary t = {malloc(b.size), b.size};
         REQUIRE(t.data);
         memcpy(t.data, b.data, b.size);
         t.data[at] = bad[i];
         VkPipeline pipeline;
         result = import(&t, &pipeline);
         if (result == VK_SUCCESS)
            vkDestroyPipeline(dev, pipeline, NULL);
         char what[64];
         snprintf(what, sizeof(what), "raygen shader NIR stage %u", bad[i]);
         printf("INFO %s: result %d\n", what, result);
         expect(what, result != VK_SUCCESS);
         free(t.data);
      }
   }

   /* In range, but a shader of a stage the slot does not take. */
   expect_refused("miss group names the any-hit shader", &b, GROUP_MISS,
                  recursive, STAGE_RAHIT);
   expect_refused("any-hit slot names the raygen shader", &b, GROUP_HIT, ahit,
                  STAGE_RGEN);
   expect_refused("intersection slot names the closest-hit shader", &b,
                  GROUP_HIT, isec, STAGE_RCHIT);

   free(b.data);
   vkDestroyPipeline(dev, compiled, NULL);
   for (unsigned i = 0; i < STAGE_COUNT; ++i)
      vkDestroyShaderModule(dev, modules[i], NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%s: %u failure(s)\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
