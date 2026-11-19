/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Native G17P ray-tracing state follows the logical device's enabled
 * features. The kernel-facing VA, BO and bind calls are replaced; the
 * production device-state initialization and shader cache key run as in the
 * driver. No DRM device or GPU is used.
 */
#include "hk_acceleration_structure.h"
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

/* The driver headers see the same NDEBUG as the linked driver. */
#undef NDEBUG
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RQ_CODE_ALIAS_VA UINT64_C(0x10010000000)
#define RQ_CODE_BYTES    11108352u
#define RQ_CODE_MAPS     630u

static unsigned failures;
static unsigned va_allocs, va_live, bo_creates, bo_live, binds, unbinds;
static unsigned code_bos, alias_attempts;
static bool fail_code_alias;
static uint64_t next_va = UINT64_C(0x2000000000);

#define T(name, cond)                                                          \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL: %s\n", name);                                  \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

__typeof__(agx_va_alloc) __wrap_agx_va_alloc;
__typeof__(agx_va_free) __wrap_agx_va_free;
__typeof__(agx_bo_create) __wrap_agx_bo_create;
__typeof__(agx_bo_unreference) __wrap_agx_bo_unreference;
__typeof__(agx_bo_bind) __wrap_agx_bo_bind;
__typeof__(__vk_errorf) __wrap___vk_errorf;

struct agx_va *
__wrap_agx_va_alloc(struct agx_device *dev, uint64_t size_B, uint64_t align_B,
                    enum agx_va_flags flags, uint64_t fixed_va)
{
   ++va_allocs;
   if (fixed_va == RQ_CODE_ALIAS_VA) {
      ++alias_attempts;
      if (fail_code_alias)
         return NULL;
   }
   struct agx_va *va = calloc(1, sizeof(*va));
   assert(va);
   va->flags = flags;
   va->size_B = size_B;
   va->addr = fixed_va ? fixed_va : next_va;
   if (!fixed_va)
      next_va += ALIGN_POT(size_B, UINT64_C(0x4000));
   ++va_live;
   return va;
}

void
__wrap_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind)
{
   if (!va)
      return;
   assert(va_live);
   --va_live;
   free(va);
}

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   ++bo_creates;
   if (label && !strcmp(label, "Gen1 ray query programs"))
      ++code_bos;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   assert(bo);
   bo->size = size;
   bo->flags = flags;
   bo->dev = dev;
   bo->refcnt = 1;
   bo->label = label;
   bo->_map = calloc(1, size);
   assert(bo->_map);
   bo->va = calloc(1, sizeof(*bo->va));
   assert(bo->va);
   bo->va->addr = next_va;
   bo->va->size_B = size;
   next_va += ALIGN_POT((uint64_t)size, UINT64_C(0x4000));
   ++bo_live;
   return bo;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   assert(bo_live);
   --bo_live;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

int
__wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t address,
                   size_t size, uint64_t offset, uint32_t flags)
{
   if (flags & DRM_ASAHI_BIND_UNBIND)
      ++unbinds;
   else
      ++binds;
   return 0;
}

VkResult
__wrap___vk_errorf(const void *object, VkResult result, const char *file,
                   int line, const char *format, ...)
{
   return result;
}

static void
reset_counts(void)
{
   va_allocs = bo_creates = binds = unbinds = code_bos = alias_attempts = 0;
}

static struct hk_device *
new_device(enum agx_chip chip, uint64_t kernel_features,
           const struct vk_features *features)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   assert(dev);
   dev->dev.chip = chip;
   dev->dev.params.features = kernel_features;
   if (features)
      dev->vk.enabled_features = *features;
   return dev;
}

static bool
rt_state_absent(const struct hk_device *dev)
{
   return !dev->g17p_native_as && !dev->gen1_native_rq &&
          !dev->gen1_rq_instances && !dev->gen1_rq_candidates &&
          !dev->gen1_native_fragment_rq && !dev->gen1_native_graphics_rq &&
          !dev->gen1_rq_code && !dev->gen1_rq_ift && !dev->gen1_rq_state &&
          !dev->gen1_as_query_bank && !dev->gen1_vertex_directory &&
          !dev->gen1_as_lock_ready && !dev->accel_struct_args.has_update;
}

/* What hk_gen1_rq_finish() releases; the profile flags stay as they were. */
static bool
rt_resources_released(const struct hk_device *dev)
{
   return !dev->gen1_native_rq && !dev->gen1_rq_code && !dev->gen1_rq_ift &&
          !dev->gen1_rq_state && !dev->gen1_rq_code_alias &&
          !dev->gen1_rq_maps && !dev->gen1_rq_user_image &&
          !dev->gen1_rq_bound_maps && !dev->gen1_as_query_bank &&
          !dev->gen1_vertex_directory && !dev->gen1_as_metadata &&
          !dev->gen1_as_addresses && !dev->gen1_as_lock_ready &&
          !dev->gen1_as_ids && !dev->gen1_vertex_tables &&
          bo_live == 0 && va_live == 0;
}

/* Every member of the ray-tracing feature family, by offset. */
static const struct {
   const char *name;
   size_t offset;
} rt_features[] = {
#define F(x) {#x, offsetof(struct vk_features, x)}
   F(accelerationStructure),
   F(accelerationStructureCaptureReplay),
   F(accelerationStructureIndirectBuild),
   F(accelerationStructureHostCommands),
   F(descriptorBindingAccelerationStructureUpdateAfterBind),
   F(rayQuery),
   F(rayTraversalPrimitiveCulling),
   F(rayTracingPipeline),
   F(rayTracingPipelineShaderGroupHandleCaptureReplay),
   F(rayTracingPipelineShaderGroupHandleCaptureReplayMixed),
   F(rayTracingPipelineTraceRaysIndirect),
   F(pipelineLibraryGroupHandles),
   F(rayTracingMaintenance1),
   F(rayTracingPipelineTraceRaysIndirect2),
   F(rayTracingPositionFetch),
   F(rayTracingInvocationReorder),
#undef F
};

static struct vk_features
only(size_t offset)
{
   struct vk_features f;
   memset(&f, 0, sizeof(f));
   *(bool *)((uint8_t *)&f + offset) = true;
   return f;
}

/* Features with no ray-tracing meaning that an ordinary graphics client
 * enables. */
static struct vk_features
ordinary_features(void)
{
   struct vk_features f;
   memset(&f, 0, sizeof(f));
   f.robustBufferAccess = true;
   f.geometryShader = true;
   f.tessellationShader = true;
   f.samplerAnisotropy = true;
   f.bufferDeviceAddress = true;
   f.descriptorIndexing = true;
   f.timelineSemaphore = true;
   f.dynamicRendering = true;
   f.synchronization2 = true;
   return f;
}

static const uint64_t rt_kernel = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                                  DRM_ASAHI_FEATURE_RENDER_RAY_QUERY;

static void
test_feature_predicate(void)
{
   struct vk_features none;
   memset(&none, 0, sizeof(none));
   T("no feature enables ray tracing", !hk_rt_features_enabled(&none));
   struct vk_features ordinary = ordinary_features();
   T("ordinary features do not enable ray tracing",
     !hk_rt_features_enabled(&ordinary));
   for (unsigned i = 0; i < ARRAY_SIZE(rt_features); ++i) {
      struct vk_features f = only(rt_features[i].offset);
      if (!hk_rt_features_enabled(&f)) {
         fprintf(stderr, "FAIL: %s alone does not enable ray tracing\n",
                 rt_features[i].name);
         ++failures;
      }
   }
}

/* A device without ray-tracing features allocates and binds nothing for it
 * and leaves every flag that selects the render SCS reservation clear. */
static void
test_device_without_rt(const char *name, enum agx_chip chip, uint64_t kernel,
                       const struct vk_features *features)
{
   reset_counts();
   struct hk_device *dev = new_device(chip, kernel, features);
   const VkResult result = hk_device_init_accel_struct_state(dev);
   char what[160];
   snprintf(what, sizeof(what), "%s: init succeeds", name);
   T(what, result == VK_SUCCESS);
   snprintf(what, sizeof(what), "%s: no native ray-tracing state", name);
   T(what, rt_state_absent(dev));
   snprintf(what, sizeof(what), "%s: no VA, BO or bind", name);
   T(what, va_allocs == 0 && bo_creates == 0 && binds == 0);
   snprintf(what, sizeof(what), "%s: queue reserves no render SCS", name);
   T(what, !dev->gen1_native_graphics_rq && !dev->gen1_native_fragment_rq);
   snprintf(what, sizeof(what), "%s: software AS layout", name);
   T(what, dev->accel_struct_args.bvh_bounds_offset != 0x50 &&
              !dev->accel_struct_args.morton_phase_dispatch);
   hk_gen1_rq_finish(dev);
   snprintf(what, sizeof(what), "%s: finish touches nothing", name);
   T(what, unbinds == 0 && rt_resources_released(dev) &&
              rt_state_absent(dev));
   free(dev);
}

/* Each ray-tracing feature alone reaches the native ray query image. The
 * fixed code alias is refused here, so initialization fails at its first
 * kernel-facing step and unwinds without leaks. */
static void
test_each_feature_requests_native_state(void)
{
   fail_code_alias = true;
   for (unsigned i = 0; i < ARRAY_SIZE(rt_features); ++i) {
      reset_counts();
      struct vk_features f = only(rt_features[i].offset);
      struct hk_device *dev = new_device(AGX_CHIP_G17P, rt_kernel, &f);
      const VkResult result = hk_device_init_accel_struct_state(dev);
      if (result != VK_ERROR_INITIALIZATION_FAILED || alias_attempts != 1 ||
          !dev->g17p_native_as || !rt_resources_released(dev)) {
         fprintf(stderr,
                 "FAIL: %s alone: result %d, alias attempts %u, live %u/%u\n",
                 rt_features[i].name, result, alias_attempts, bo_live,
                 va_live);
         ++failures;
      }
      free(dev);
   }
   fail_code_alias = false;
}

/* With ray tracing enabled, initialization creates the full native state,
 * exactly as every G17P device with a capable kernel used to. */
static void
test_device_with_rt(const char *name, const struct vk_features *features)
{
   reset_counts();
   struct hk_device *dev = new_device(AGX_CHIP_G17P, rt_kernel, features);
   const VkResult result = hk_device_init_accel_struct_state(dev);
   char what[160];
   snprintf(what, sizeof(what), "%s: init succeeds", name);
   T(what, result == VK_SUCCESS);
   snprintf(what, sizeof(what), "%s: native profile selected", name);
   T(what, dev->g17p_native_as && dev->gen1_native_rq &&
              dev->gen1_rq_instances && dev->gen1_rq_candidates &&
              dev->gen1_native_fragment_rq && dev->gen1_native_graphics_rq);
   snprintf(what, sizeof(what), "%s: candidate SCS layout", name);
   T(what, hk_gen1_ray_query_scs_layout(dev) ==
              UINT64_C(0x0130000060010070));
   snprintf(what, sizeof(what), "%s: ray query image and code binds", name);
   T(what, code_bos == 1 && dev->gen1_rq_code &&
              dev->gen1_rq_code->size == RQ_CODE_BYTES &&
              dev->gen1_rq_bound_maps == RQ_CODE_MAPS &&
              binds >= RQ_CODE_MAPS && alias_attempts == 1);
   snprintf(what, sizeof(what), "%s: user image and AS tables", name);
   T(what, dev->gen1_rq_ift && dev->gen1_rq_state && dev->gen1_as_lock_ready &&
              dev->gen1_vertex_directory && dev->gen1_as_metadata &&
              dev->gen1_as_addresses);
   snprintf(what, sizeof(what), "%s: native AS layout", name);
   T(what, dev->accel_struct_args.bvh_bounds_offset == 0x50 &&
              dev->accel_struct_args.has_update &&
              dev->accel_struct_args.morton_phase_dispatch);
   const unsigned bound = binds;
   hk_gen1_rq_finish(dev);
   snprintf(what, sizeof(what), "%s: finish releases everything", name);
   T(what, rt_resources_released(dev) && unbinds >= RQ_CODE_MAPS &&
              unbinds <= bound);
   free(dev);
}

static void
hash(struct hk_physical_device *pdev, const struct vk_features *features,
     VkShaderStageFlags stages, blake3_hash out)
{
   hk_device_shader_ops.hash_state(&pdev->vk, NULL, features, stages, out);
}

/* Shader cache keys separate the two kinds of device on G17P. Every
 * ray-tracing feature selects the same compile, so all share one key. */
static void
test_cache_key(void)
{
   static const VkShaderStageFlags stages[] = {
      VK_SHADER_STAGE_COMPUTE_BIT,
      VK_SHADER_STAGE_GEOMETRY_BIT,
      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
   };
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(pdev);
   pdev->dev.chip = AGX_CHIP_G17P;
   pdev->dev.params.features = rt_kernel;

   struct vk_features none, ordinary = ordinary_features();
   memset(&none, 0, sizeof(none));
   for (unsigned s = 0; s < ARRAY_SIZE(stages); ++s) {
      blake3_hash plain, plain_again, rq, ordinary_hash, ordinary_rq;
      hash(pdev, &none, stages[s], plain);
      hash(pdev, &none, stages[s], plain_again);
      T("key is deterministic", !memcmp(plain, plain_again, sizeof(plain)));

      struct vk_features f = only(offsetof(struct vk_features, rayQuery));
      hash(pdev, &f, stages[s], rq);
      T("ray tracing changes the key", memcmp(plain, rq, sizeof(plain)));

      hash(pdev, &ordinary, stages[s], ordinary_hash);
      struct vk_features ordinary_with_rq = ordinary;
      ordinary_with_rq.rayQuery = true;
      hash(pdev, &ordinary_with_rq, stages[s], ordinary_rq);
      T("ray tracing changes an ordinary key",
        memcmp(ordinary_hash, ordinary_rq, sizeof(ordinary_hash)));

      for (unsigned i = 0; i < ARRAY_SIZE(rt_features); ++i) {
         blake3_hash other;
         f = only(rt_features[i].offset);
         hash(pdev, &f, stages[s], other);
         if (memcmp(rq, other, sizeof(rq))) {
            fprintf(stderr, "FAIL: %s keys differently from rayQuery\n",
                    rt_features[i].name);
            ++failures;
         }
      }
   }
   free(pdev);
}

int
main(void)
{
   unsetenv("ASAHI_G17P_RT_DISABLE");
   unsetenv("ASAHI_G17P_RQ_LIVE_AS_TABLE");

   test_feature_predicate();

   struct vk_features none = {0};
   struct vk_features ordinary = ordinary_features();
   struct vk_features rq = only(offsetof(struct vk_features, rayQuery));
   test_device_without_rt("G17P, no features", AGX_CHIP_G17P, rt_kernel,
                          &none);
   test_device_without_rt("G17P, ordinary features", AGX_CHIP_G17P,
                          rt_kernel, &ordinary);
   test_device_without_rt("G17P, ray query on a kernel without it",
                          AGX_CHIP_G17P,
                          DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS, &rq);
   test_device_without_rt("G13, ray query", AGX_CHIP_G13G, rt_kernel, &rq);

   test_each_feature_requests_native_state();

   test_device_with_rt("ray query", &rq);
   struct vk_features as =
      only(offsetof(struct vk_features, accelerationStructure));
   test_device_with_rt("acceleration structure", &as);
   struct vk_features pipeline =
      only(offsetof(struct vk_features, rayTracingPipeline));
   test_device_with_rt("ray tracing pipeline", &pipeline);
   struct vk_features ordinary_rq = ordinary;
   ordinary_rq.rayQuery = true;
   ordinary_rq.accelerationStructure = true;
   test_device_with_rt("ordinary features with ray query", &ordinary_rq);

   setenv("ASAHI_G17P_RT_DISABLE", "1", 1);
   test_device_with_rt("ray query ignores the retired environment opt-out", &rq);
   unsetenv("ASAHI_G17P_RT_DISABLE");

   test_cache_key();

   /* A noborder run must never replace border-enabled disk-cache entries. */
   unsetenv("HK_PERFTEST");
   uint64_t normal_env = hk_codegen_env_hash();
   setenv("HK_PERFTEST", "noborder", 1);
   T("noborder has a separate disk-cache identity",
     normal_env != hk_codegen_env_hash());
   unsetenv("HK_PERFTEST");
   T("normal disk-cache identity returns", normal_env == hk_codegen_env_hash());

   if (failures) {
      fprintf(stderr, "%u failure(s)\n", failures);
      return 1;
   }
   printf("hk_rt_on_demand: all checks passed\n");
   return 0;
}
