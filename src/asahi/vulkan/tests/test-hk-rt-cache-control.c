/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* CPU control of the production RT creation entry point.  Translation and
 * megakernel construction are real; only GPU shader compilation/upload is
 * replaced (with a serializable stand-in, so pipeline cache entries can be
 * written and read back).  These checks do not execute shaders or establish
 * conformance.
 */
#include "spirv/spirv.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "util/blob.h"
#include "util/set.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_ray_tracing_core.h"
#include "hk_ray_tracing_pipeline.h"
#include "hk_shader.h"
#include "hk_rt_recursion_spv.h"
#include "nir.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_instance.h"
#include "vk_nir.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"

#include <inttypes.h>
#include <stdio.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)
#define FAIL  VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
#define LIB   VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
#define EARLY VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT

static unsigned translations, backends, shader_destroys, live_nir;
static unsigned last_instrs, last_calls, last_max_block;
static void (*inspect_nir)(nir_shader *nir);
static bool capture_last_nir;
static nir_shader *captured_nir;
static nir_shader *tracked_nir[64];
static VkResult backend_result = VK_SUCCESS;
static unsigned fail_translation;

VkResult __real_vk_pipeline_shader_stage_to_nir(
   struct vk_device *, VkPipelineCreateFlags2KHR,
   const VkPipelineShaderStageCreateInfo *, const struct spirv_to_nir_options *,
   const nir_shader_compiler_options *, void *, nir_shader **);

VkResult
__wrap_vk_pipeline_shader_stage_to_nir(
   struct vk_device *dev, VkPipelineCreateFlags2KHR flags,
   const VkPipelineShaderStageCreateInfo *stage,
   const struct spirv_to_nir_options *spirv_options,
   const nir_shader_compiler_options *nir_options, void *ctx, nir_shader **out);

VkResult
__wrap_vk_pipeline_shader_stage_to_nir(
   struct vk_device *dev, VkPipelineCreateFlags2KHR flags,
   const VkPipelineShaderStageCreateInfo *stage,
   const struct spirv_to_nir_options *spirv_options,
   const nir_shader_compiler_options *nir_options, void *ctx, nir_shader **out)
{
   translations++;
   if (fail_translation == translations)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = __real_vk_pipeline_shader_stage_to_nir(
      dev, flags, stage, spirv_options, nir_options, ctx, out);
   if (result == VK_SUCCESS) {
      unsigned i;
      for (i = 0; i < ARRAY_SIZE(tracked_nir); i++) {
         if (!tracked_nir[i]) {
            tracked_nir[i] = *out;
            live_nir++;
            break;
         }
      }
      CHECK(i < ARRAY_SIZE(tracked_nir));
   }
   return result;
}

void __real_ralloc_free(void *);
void
__wrap_ralloc_free(void *ptr);

void
__wrap_ralloc_free(void *ptr)
{
   if (ptr) {
      for (unsigned i = 0; i < ARRAY_SIZE(tracked_nir); i++) {
         if (tracked_nir[i] == ptr) {
            tracked_nir[i] = NULL;
            live_nir--;
         }
      }
   }
   __real_ralloc_free(ptr);
}

static unsigned restored;

static bool
fake_serialize(struct vk_device *device, const struct vk_shader *shader,
               struct blob *blob)
{
   blob_write_uint32(blob, 0x5348524b);
   return !blob->out_of_memory;
}
static const struct vk_shader_ops fake_shader_ops = {.serialize = fake_serialize};

static VkResult
fake_deserialize(struct vk_device *device, struct blob_reader *blob,
                 uint32_t version, const VkAllocationCallbacks *alloc,
                 struct vk_shader **out)
{
   CHECK(blob_read_uint32(blob) == 0x5348524b);
   restored++;
   struct hk_api_shader *shader = calloc(1, sizeof(*shader));
   CHECK(shader);
   shader->vk.ops = &fake_shader_ops;
   *out = &shader->vk;
   return VK_SUCCESS;
}
static const struct vk_device_shader_ops fake_device_ops = {
   .deserialize = fake_deserialize};

VkResult
__wrap_hk_compile_shader(struct hk_device *dev,
                         struct vk_shader_compile_info *info,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features,
                         const VkAllocationCallbacks *alloc,
                         struct hk_api_shader **out);

VkResult
__wrap_hk_compile_shader(struct hk_device *dev,
                         struct vk_shader_compile_info *info,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features,
                         const VkAllocationCallbacks *alloc,
                         struct hk_api_shader **out)
{
   backends++;
   /* What the last megakernel holds: every call must be inlined. */
   last_instrs = last_calls = last_max_block = 0;
   nir_foreach_function_impl(impl, info->nir) {
      nir_foreach_block(block, impl) {
         unsigned n = 0;
         nir_foreach_instr(instr, block) {
            n++;
            last_calls += instr->type == nir_instr_type_call;
         }
         last_instrs += n;
         last_max_block = MAX2(last_max_block, n);
      }
   }
   if (inspect_nir)
      inspect_nir(info->nir);
   if (capture_last_nir) {
      ralloc_free(captured_nir);
      captured_nir = info->nir;
   } else {
      ralloc_free(info->nir);
   }
   if (backend_result != VK_SUCCESS)
      return backend_result;
   *out = calloc(1, sizeof(**out));
   CHECK(*out);
   (*out)->vk.ops = &fake_shader_ops;
   return VK_SUCCESS;
}

void
__wrap_hk_api_shader_destroy(struct vk_device *dev, struct vk_shader *shader,
                             const VkAllocationCallbacks *alloc);

void
__wrap_hk_api_shader_destroy(struct vk_device *dev, struct vk_shader *shader,
                             const VkAllocationCallbacks *alloc)
{
   shader_destroys++;
   free(container_of(shader, struct hk_api_shader, vk));
}

/* Empty raygen SPIR-V, assembled from the public SPIR-V opcodes. */
static const uint32_t raygen[] = {
   SpvMagicNumber,
   0x00010400,
   0,
   5,
   0,
   (2 << 16) | SpvOpCapability,
   SpvCapabilityShader,
   (2 << 16) | SpvOpCapability,
   SpvCapabilityRayTracingKHR,
   (6 << 16) | SpvOpExtension,
   0x5f565053,
   0x5f52484b,
   0x5f796172,
   0x63617274,
   0x00676e69,
   (3 << 16) | SpvOpMemoryModel,
   SpvAddressingModelLogical,
   SpvMemoryModelGLSL450,
   (5 << 16) | SpvOpEntryPoint,
   SpvExecutionModelRayGenerationKHR,
   3,
   0x6e69616d,
   0,
   (2 << 16) | SpvOpTypeVoid,
   1,
   (3 << 16) | SpvOpTypeFunction,
   2,
   1,
   (5 << 16) | SpvOpFunction,
   1,
   3,
   SpvFunctionControlMaskNone,
   2,
   (2 << 16) | SpvOpLabel,
   4,
   (1 << 16) | SpvOpReturn,
   (1 << 16) | SpvOpFunctionEnd,
};

static struct vk_instance instance;
static struct hk_device dev;
static struct hk_physical_device pdev;
static VkDevice device;
static struct vk_pipeline_layout *layout;
static VkShaderModule module;
static VkPipelineShaderStageCreateInfo stage;
static VkRayTracingShaderGroupCreateInfoKHR group = {
   .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
   .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
   .generalShader = 0,
   .closestHitShader = VK_SHADER_UNUSED_KHR,
   .anyHitShader = VK_SHADER_UNUSED_KHR,
   .intersectionShader = VK_SHADER_UNUSED_KHR,
};
static unsigned pipeline_allocs, pipeline_frees;

static void *VKAPI_CALL
alloc_pipeline(void *data, size_t size, size_t alignment,
               VkSystemAllocationScope scope)
{
   void *ptr = NULL;
   if (posix_memalign(&ptr, MAX2(alignment, sizeof(void *)), size) != 0)
      return NULL;
   pipeline_allocs++;
   return ptr;
}
static void VKAPI_CALL
free_pipeline(void *data, void *ptr)
{
   if (ptr)
      pipeline_frees++;
   free(ptr);
}
static const VkAllocationCallbacks alloc = {
   .pfnAllocation = alloc_pipeline,
   .pfnFree = free_pipeline,
};

static void VKAPI_CALL
get_properties(VkPhysicalDevice physical, VkPhysicalDeviceProperties *props)
{
   memset(props, 0, sizeof(*props));
}

static VkRayTracingPipelineCreateInfoKHR
info(VkPipelineCreateFlags flags)
{
   return (VkRayTracingPipelineCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags = flags,
      .stageCount = 1,
      .pStages = &stage,
      .groupCount = 1,
      .pGroups = &group,
      .maxPipelineRayRecursionDepth = 1,
      .layout = vk_pipeline_layout_to_handle(layout),
   };
}

static void
destroy(VkPipeline pipeline)
{
   if (pipeline) {
      struct hk_ray_tracing_pipeline *p =
         hk_ray_tracing_pipeline_from_handle(pipeline);
      p->base.ops->destroy(&dev.vk, &p->base, &alloc);
   }
}

static VkResult
create(VkPipelineCache cache, const VkRayTracingPipelineCreateInfoKHR *ci,
       VkPipeline *pipeline)
{
   *pipeline = (VkPipeline)(uintptr_t)0xbad;
   return hk_CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, cache, 1, ci,
                                          &alloc, pipeline);
}

static void
check_refusal(VkPipelineCache cache, VkRayTracingPipelineCreateInfoKHR ci)
{
   VkPipeline p;
   unsigned t = translations, b = backends, n = live_nir;
   unsigned a = pipeline_allocs, f = pipeline_frees, refs = layout->ref_cnt;
   VkResult result = create(cache, &ci, &p);
   if (result != VK_PIPELINE_COMPILE_REQUIRED || translations != t ||
       backends != b)
      fprintf(stderr, "flag oracle: result=%d, translations=%u, backends=%u\n",
              result, translations - t, backends - b);
   CHECK(result == VK_PIPELINE_COMPILE_REQUIRED);
   CHECK(p == VK_NULL_HANDLE);
   CHECK(translations == t && backends == b && live_nir == n);
   CHECK(layout->ref_cnt == refs);
   CHECK(pipeline_allocs - a == pipeline_frees - f);
}

/* A pipeline cache entry satisfies VK_PIPELINE_COMPILE_REQUIRED: no stage
 * translation, no backend compile. */
static void
check_hit(VkPipelineCache cache, VkRayTracingPipelineCreateInfoKHR ci)
{
   VkPipeline p;
   unsigned t = translations, b = backends;
   unsigned a = pipeline_allocs, f = pipeline_frees, refs = layout->ref_cnt;
   VkResult result = create(cache, &ci, &p);
   if (result != VK_SUCCESS || translations != t || backends != b)
      fprintf(stderr, "cache oracle: result=%d, translations=%u, backends=%u\n",
              result, translations - t, backends - b);
   CHECK(result == VK_SUCCESS && p != VK_NULL_HANDLE);
   CHECK(translations == t && backends == b);
   destroy(p);
   CHECK(layout->ref_cnt == refs);
   CHECK(pipeline_allocs - a == pipeline_frees - f);
}

static struct hk_ray_tracing_pipeline *
rt(VkPipeline p)
{
   return hk_ray_tracing_pipeline_from_handle(p);
}

/* Minimal concrete evaluator for pure NIR SSA DAGs emitted by
 * hk_compile_rt_kernel() (launch-ID reconstruction, OOB guard, SBT entry
 * address arithmetic, and group-handle dispatch conditions). */
struct nir_eval_env {
   uint32_t wg[3];
   uint32_t lid;
   uint32_t sbt_index;
   VkTraceRaysIndirectCommand2KHR cmd;
   uint64_t root_addr;
   uint64_t cmd_addr;
   const uint8_t *hit_sbt;
   size_t hit_sbt_size;
};

static uint64_t
nir_eval_mask(unsigned bit_size)
{
   return bit_size >= 64 ? ~UINT64_C(0) : ((UINT64_C(1) << bit_size) - 1);
}

static int64_t
nir_eval_sext(uint64_t v, unsigned bit_size)
{
   v &= nir_eval_mask(bit_size);
   if (bit_size < 64 && (v & (UINT64_C(1) << (bit_size - 1))))
      v |= ~nir_eval_mask(bit_size);
   return (int64_t)v;
}

static uint64_t
nir_eval_def(const nir_def *def, unsigned comp, const struct nir_eval_env *env);

static uint64_t
nir_eval_alu_src(const nir_alu_src *src, unsigned comp,
                 const struct nir_eval_env *env)
{
   return nir_eval_def(src->src.ssa, src->swizzle[comp], env);
}

static uint64_t
nir_eval_mem_read(const struct nir_eval_env *env, uint64_t addr,
                  unsigned bytes)
{
   if (addr == env->root_addr + hk_root_descriptor_offset(cs.trace_rays_addr)) {
      CHECK(bytes == 8);
      return env->cmd_addr;
   }
   if (addr >= env->cmd_addr &&
       addr + bytes <= env->cmd_addr + sizeof(env->cmd)) {
      uint64_t out = 0;
      memcpy(&out, (const uint8_t *)&env->cmd + (addr - env->cmd_addr), bytes);
      return out;
   }
   if (env->hit_sbt && addr >= env->cmd.hitShaderBindingTableAddress &&
       addr + bytes <= env->cmd.hitShaderBindingTableAddress + env->hit_sbt_size) {
      uint64_t out = 0;
      memcpy(&out,
             env->hit_sbt + (addr - env->cmd.hitShaderBindingTableAddress),
             bytes);
      return out;
   }
   fprintf(stderr, "unexpected nir_eval_mem_read at 0x%" PRIx64 " (%u B)\n",
           addr, bytes);
   abort();
}

static uint64_t
nir_eval_def(const nir_def *def, unsigned comp, const struct nir_eval_env *env)
{
   const nir_instr *instr = nir_def_instr(def);
   const unsigned bs = def->bit_size;
   const uint64_t mask = nir_eval_mask(bs);

   switch (instr->type) {
   case nir_instr_type_load_const: {
      const nir_load_const_instr *lc = nir_instr_as_load_const(instr);
      return nir_const_value_as_uint(lc->value[comp], bs) & mask;
   }
   case nir_instr_type_phi:
      /* Software traversal carries sbt_index across loop iterations in a phi. */
      return env->sbt_index & mask;
   case nir_instr_type_intrinsic: {
      const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_workgroup_id:
         CHECK(comp < 3);
         return env->wg[comp];
      case nir_intrinsic_load_local_invocation_index:
         return env->lid;
      case nir_intrinsic_load_root_agx:
         return env->root_addr;
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_global: {
         const uint64_t base = nir_eval_def(intr->src[0].ssa, 0, env);
         const unsigned bytes = bs / 8;
         return nir_eval_mem_read(env, base + comp * bytes, bytes) & mask;
      }
      case nir_intrinsic_rq_load: {
         const nir_ray_query_value val = nir_intrinsic_ray_query_value(intr);
         if (val == nir_ray_query_value_intersection_instance_sbt_index)
            return env->sbt_index & mask;
         return 0;
      }
      default:
         fprintf(stderr, "unsupported intrinsic in nir_eval_def: %s\n",
                 nir_intrinsic_infos[intr->intrinsic].name);
         abort();
      }
   }
   case nir_instr_type_alu: {
      const nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_mov:
         return nir_eval_alu_src(&alu->src[0], comp, env) & mask;
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
         return nir_eval_alu_src(&alu->src[comp], 0, env) & mask;
      case nir_op_iadd:
         return (nir_eval_alu_src(&alu->src[0], comp, env) +
                 nir_eval_alu_src(&alu->src[1], comp, env)) & mask;
      case nir_op_isub:
         return (nir_eval_alu_src(&alu->src[0], comp, env) -
                 nir_eval_alu_src(&alu->src[1], comp, env)) & mask;
      case nir_op_imul:
         return (nir_eval_alu_src(&alu->src[0], comp, env) *
                 nir_eval_alu_src(&alu->src[1], comp, env)) & mask;
      case nir_op_umul_2x32_64:
         return (uint64_t)(uint32_t)nir_eval_alu_src(&alu->src[0], comp, env) *
                (uint64_t)(uint32_t)nir_eval_alu_src(&alu->src[1], comp, env);
      case nir_op_ishl: {
         const uint64_t a = nir_eval_alu_src(&alu->src[0], comp, env);
         const unsigned s = nir_eval_alu_src(&alu->src[1], comp, env) & (bs - 1);
         return (a << s) & mask;
      }
      case nir_op_ushr: {
         const uint64_t a = nir_eval_alu_src(&alu->src[0], comp, env) & mask;
         const unsigned s = nir_eval_alu_src(&alu->src[1], comp, env) & (bs - 1);
         return (a >> s) & mask;
      }
      case nir_op_iand:
         return (nir_eval_alu_src(&alu->src[0], comp, env) &
                 nir_eval_alu_src(&alu->src[1], comp, env)) & mask;
      case nir_op_ior:
         return (nir_eval_alu_src(&alu->src[0], comp, env) |
                 nir_eval_alu_src(&alu->src[1], comp, env)) & mask;
      case nir_op_ixor:
         return (nir_eval_alu_src(&alu->src[0], comp, env) ^
                 nir_eval_alu_src(&alu->src[1], comp, env)) & mask;
      case nir_op_inot:
         return (~nir_eval_alu_src(&alu->src[0], comp, env)) & mask;
      case nir_op_ieq: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         const uint64_t sm = nir_eval_mask(sbs);
         return ((nir_eval_alu_src(&alu->src[0], comp, env) & sm) ==
                 (nir_eval_alu_src(&alu->src[1], comp, env) & sm)) ? 1 : 0;
      }
      case nir_op_ine: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         const uint64_t sm = nir_eval_mask(sbs);
         return ((nir_eval_alu_src(&alu->src[0], comp, env) & sm) !=
                 (nir_eval_alu_src(&alu->src[1], comp, env) & sm)) ? 1 : 0;
      }
      case nir_op_ult: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         const uint64_t sm = nir_eval_mask(sbs);
         return ((nir_eval_alu_src(&alu->src[0], comp, env) & sm) <
                 (nir_eval_alu_src(&alu->src[1], comp, env) & sm)) ? 1 : 0;
      }
      case nir_op_uge: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         const uint64_t sm = nir_eval_mask(sbs);
         return ((nir_eval_alu_src(&alu->src[0], comp, env) & sm) >=
                 (nir_eval_alu_src(&alu->src[1], comp, env) & sm)) ? 1 : 0;
      }
      case nir_op_ilt: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         return (nir_eval_sext(nir_eval_alu_src(&alu->src[0], comp, env), sbs) <
                 nir_eval_sext(nir_eval_alu_src(&alu->src[1], comp, env), sbs))
                   ? 1 : 0;
      }
      case nir_op_ige: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         return (nir_eval_sext(nir_eval_alu_src(&alu->src[0], comp, env), sbs) >=
                 nir_eval_sext(nir_eval_alu_src(&alu->src[1], comp, env), sbs))
                   ? 1 : 0;
      }
      case nir_op_bcsel:
         return (nir_eval_alu_src(&alu->src[0], comp, env) & 1)
                   ? (nir_eval_alu_src(&alu->src[1], comp, env) & mask)
                   : (nir_eval_alu_src(&alu->src[2], comp, env) & mask);
      case nir_op_u2u64:
      case nir_op_u2u32:
      case nir_op_u2u16:
      case nir_op_u2u8: {
         const unsigned sbs = alu->src[0].src.ssa->bit_size;
         return (nir_eval_alu_src(&alu->src[0], comp, env) &
                 nir_eval_mask(sbs)) & mask;
      }
      case nir_op_pack_64_2x32:
         return (nir_eval_alu_src(&alu->src[0], 0, env) & UINT32_MAX) |
                ((nir_eval_alu_src(&alu->src[0], 1, env) & UINT32_MAX) << 32);
      case nir_op_pack_64_2x32_split:
         return (nir_eval_alu_src(&alu->src[0], comp, env) & UINT32_MAX) |
                ((nir_eval_alu_src(&alu->src[1], comp, env) & UINT32_MAX) << 32);
      case nir_op_unpack_64_2x32: {
         const uint64_t v = nir_eval_alu_src(&alu->src[0], 0, env);
         return (comp == 0 ? v : (v >> 32)) & UINT32_MAX;
      }
      case nir_op_unpack_64_2x32_split_x:
         return nir_eval_alu_src(&alu->src[0], comp, env) & UINT32_MAX;
      case nir_op_unpack_64_2x32_split_y:
         return (nir_eval_alu_src(&alu->src[0], comp, env) >> 32) & UINT32_MAX;
      case nir_op_b2i32:
      case nir_op_b2i64:
         return (nir_eval_alu_src(&alu->src[0], comp, env) & 1) ? 1 : 0;
      case nir_op_ubfe: {
         const uint64_t v = nir_eval_alu_src(&alu->src[0], comp, env);
         const unsigned off = nir_eval_alu_src(&alu->src[1], comp, env) & 63;
         const unsigned bits = nir_eval_alu_src(&alu->src[2], comp, env) & 63;
         return bits == 0 ? 0 : ((v >> off) & nir_eval_mask(bits)) & mask;
      }
      default:
         fprintf(stderr, "unsupported ALU op in nir_eval_def: %s\n",
                 nir_op_infos[alu->op].name);
         abort();
      }
   }
   default:
      fprintf(stderr, "unsupported instr type %d in nir_eval_def\n",
              instr->type);
      abort();
   }
}

/* Verify that the real megakernel NIR emitted by hk_compile_rt_kernel()
 * retires the exact partial-tile OOB lanes and passes the exact in-bounds
 * lanes across heights 1, 2, 4, and 5 when dispatched with
 * hk_rt_launch_group_count(). */
static void
inspect_tiled_launch_oob(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_if *oob_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if) {
         oob_if = nir_cf_node_as_if(node);
         break;
      }
   }
   CHECK(oob_if != NULL);

   /* Determine whether the active body lives in `then` or `else` (after
    * nir_lower_returns + nir_opt_if, the condition polarity may flip). */
   unsigned then_instrs = 0, else_instrs = 0;
   foreach_list_typed(nir_cf_node, cf, node, &oob_if->then_list) {
      nir_foreach_block_in_cf_node(block, cf)
         nir_foreach_instr(instr, block)
            then_instrs++;
   }
   foreach_list_typed(nir_cf_node, cf, node, &oob_if->else_list) {
      nir_foreach_block_in_cf_node(block, cf)
         nir_foreach_instr(instr, block)
            else_instrs++;
   }
   CHECK(then_instrs != else_instrs);
   const bool active_on_true = then_instrs > else_instrs;

   const uint32_t cases[][3] = {
      {1, 1, 1}, {7, 1, 1}, {33, 1, 1},
      {1, 2, 1}, {7, 2, 1}, {8, 2, 1}, {9, 2, 1}, {9, 2, 2},
      {1, 4, 1}, {7, 4, 1}, {8, 4, 1}, {9, 4, 1}, {9, 4, 2},
      {1, 5, 1}, {7, 5, 1}, {8, 5, 1}, {9, 5, 1}, {9, 5, 2},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const uint32_t w = cases[i][0], h = cases[i][1], d = cases[i][2];
      uint32_t groups[3];
      hk_rt_launch_group_count(w, h, d, groups);
      struct nir_eval_env env = {
         .root_addr = UINT64_C(0x100000),
         .cmd_addr = UINT64_C(0x200000),
         .cmd = { .width = w, .height = h, .depth = d },
      };
      const size_t total = (size_t)w * (size_t)h * (size_t)d;
      const size_t launched = (size_t)groups[0] * groups[1] * groups[2] *
                              HK_RT_WORKGROUP_SIZE_X;
      size_t active = 0, retired = 0;
      for (uint32_t gz = 0; gz < groups[2]; ++gz) {
         for (uint32_t gy = 0; gy < groups[1]; ++gy) {
            for (uint32_t gx = 0; gx < groups[0]; ++gx) {
               for (uint32_t lid = 0; lid < HK_RT_WORKGROUP_SIZE_X; ++lid) {
                  env.wg[0] = gx;
                  env.wg[1] = gy;
                  env.wg[2] = gz;
                  env.lid = lid;
                  const bool cond =
                     (nir_eval_def(oob_if->condition.ssa, 0, &env) & 1) != 0;
                  if (cond == active_on_true)
                     active++;
                  else
                     retired++;
               }
            }
         }
      }
      CHECK(active == total);
      CHECK(retired == launched - total);
   }
}

static bool
cf_list_contains_node(const struct exec_list *list, const nir_cf_node *target)
{
   foreach_list_typed(nir_cf_node, n, node, list) {
      if (n == target)
         return true;
   }
   return false;
}

static unsigned
find_xor_magic(nir_shader *nir, uint32_t magic, nir_alu_instr **first_out)
{
   unsigned count = 0;
   if (first_out)
      *first_out = NULL;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_ixor)
               continue;
            for (unsigned s = 0; s < 2; ++s) {
               if (nir_src_is_const(alu->src[s].src) &&
                   nir_src_as_uint(alu->src[s].src) == magic) {
                  if (first_out && !*first_out)
                     *first_out = alu;
                  count++;
                  break;
               }
            }
         }
      }
   }
   return count;
}

static uint32_t
read_handle_index(const uint8_t *handles, unsigned group)
{
   const uint8_t *h = handles + group * HK_RAY_TRACING_GROUP_HANDLE_SIZE;
   uint32_t idx = 0;
   memcpy(&idx, h, sizeof(idx));
   for (unsigned b = sizeof(idx); b < HK_RAY_TRACING_GROUP_HANDLE_SIZE; ++b)
      CHECK(h[b] == 0);
   return idx;
}

/* Verify that shared hit groups with distinct SBT records merge into a single
 * call site across all chips (software traversal on G13G, native traversal on
 * G17P and M5/G17G), that the merged guard accepts every sharing group's
 * handle and rejects unrelated handles, and that each group still reads its
 * own distinct SBT record without aliasing. */
static void
test_shared_stage_call_sites(void)
{
   static const struct {
      const uint32_t *code;
      size_t size;
      VkShaderStageFlagBits stage;
   } spv[] = {
      {hk_rt_recursion_rgen_spv, sizeof(hk_rt_recursion_rgen_spv),
       VK_SHADER_STAGE_RAYGEN_BIT_KHR},
      {hk_rt_sbt_chit_spv, sizeof(hk_rt_sbt_chit_spv),
       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
      {hk_rt_recursion_ahit_spv, sizeof(hk_rt_recursion_ahit_spv),
       VK_SHADER_STAGE_ANY_HIT_BIT_KHR},
      {hk_rt_recursion_miss_spv, sizeof(hk_rt_recursion_miss_spv),
       VK_SHADER_STAGE_MISS_BIT_KHR},
      {hk_rt_isec_spv, sizeof(hk_rt_isec_spv),
       VK_SHADER_STAGE_INTERSECTION_BIT_KHR},
   };
   VkShaderModule mods[ARRAY_SIZE(spv)];
   VkPipelineShaderStageCreateInfo stages[ARRAY_SIZE(spv)];
   for (unsigned i = 0; i < ARRAY_SIZE(spv); ++i) {
      VkShaderModuleCreateInfo mi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = spv[i].size,
         .pCode = spv[i].code,
      };
      CHECK(vk_common_CreateShaderModule(device, &mi, NULL, &mods[i]) ==
            VK_SUCCESS);
      stages[i] = (VkPipelineShaderStageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = spv[i].stage,
         .module = mods[i],
         .pName = "main",
      };
   }

   /* Place miss (group 2) between hit group A (group 1) and hit groups B/C
    * (groups 3, 4) so the sharing groups' handle indices are non-adjacent. */
   VkRayTracingShaderGroupCreateInfoKHR tri_groups[5] = {
      group, group, group, group, group,
   };
   tri_groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
   tri_groups[1].generalShader = VK_SHADER_UNUSED_KHR;
   tri_groups[1].closestHitShader = 1;
   tri_groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
   tri_groups[2].generalShader = 3;
   tri_groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
   tri_groups[3].generalShader = VK_SHADER_UNUSED_KHR;
   tri_groups[3].closestHitShader = 1;
   tri_groups[3].anyHitShader = 2;
   tri_groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
   tri_groups[4].generalShader = VK_SHADER_UNUSED_KHR;
   tri_groups[4].closestHitShader = 1;
   tri_groups[4].anyHitShader = 2;

   /* Procedural hit groups: P1 and P2 share both (isec, ahit=unused) and must
    * merge into one intersection call site; P3 shares isec with a different
    * any-hit shader and must keep a second intersection call site, while all
    * three still share one closest-hit call site. */
   VkRayTracingShaderGroupCreateInfoKHR proc_groups[4] = {
      group, group, group, group,
   };
   for (unsigned g = 1; g < 4; ++g) {
      proc_groups[g].type =
         VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
      proc_groups[g].generalShader = VK_SHADER_UNUSED_KHR;
      proc_groups[g].closestHitShader = 1;
      proc_groups[g].intersectionShader = 4;
      proc_groups[g].anyHitShader = g == 3 ? 2 : VK_SHADER_UNUSED_KHR;
   }

   const struct {
      enum agx_chip chip;
      bool native_rq;
   } configs[] = {
      {AGX_CHIP_G13G, false},
      {AGX_CHIP_G17P, true},
      {AGX_CHIP_G17G, true},
   };
   const enum agx_chip saved_chip = dev.dev.chip;
   const bool saved_native = dev.gen1_native_rq;
   const bool saved_instances = dev.gen1_rq_instances;

   for (unsigned c = 0; c < ARRAY_SIZE(configs); ++c) {
      dev.dev.chip = configs[c].chip;
      dev.gen1_native_rq = configs[c].native_rq;
      dev.gen1_rq_instances = configs[c].native_rq;

      VkRayTracingPipelineCreateInfoKHR ci = info(0);
      ci.stageCount = 4;
      ci.pStages = stages;
      ci.groupCount = ARRAY_SIZE(tri_groups);
      ci.pGroups = tri_groups;
      ci.maxPipelineRayRecursionDepth = 1;

      VkPipeline p = VK_NULL_HANDLE;
      capture_last_nir = true;
      CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
      capture_last_nir = false;
      CHECK(captured_nir != NULL);

      uint8_t handles[ARRAY_SIZE(tri_groups) * HK_RAY_TRACING_GROUP_HANDLE_SIZE];
      CHECK(hk_GetRayTracingShaderGroupHandlesKHR(
               device, p, 0, ARRAY_SIZE(tri_groups), sizeof(handles),
               handles) == VK_SUCCESS);
      const uint32_t hA = read_handle_index(handles, 1);
      const uint32_t hMiss = read_handle_index(handles, 2);
      const uint32_t hB = read_handle_index(handles, 3);
      const uint32_t hC = read_handle_index(handles, 4);
      CHECK(hA != hMiss && hA != hB && hA != hC && hB != hC &&
            hB != hMiss && hC != hMiss);

      /* Shared closest-hit must appear exactly once across all three hit groups. */
      nir_alu_instr *chit_xor = NULL;
      CHECK(find_xor_magic(captured_nir, 0x51a7c0deu, &chit_xor) == 1);
      CHECK(chit_xor != NULL);

      /* Build a hit SBT with 6 records (stride 64): records 0..2 hold the three
       * sharing hit groups with distinct SBT payload tags; records 3..5 hold
       * miss, zero, and an unrelated handle. */
      enum { SBT_STRIDE = 64, SBT_RECORDS = 6 };
      uint8_t sbt[SBT_RECORDS * SBT_STRIDE];
      memset(sbt, 0, sizeof(sbt));
      const uint32_t rec_handles[SBT_RECORDS] = {
         hA, hB, hC, hMiss, 0u, hC + 17u,
      };
      const uint32_t rec_tags[SBT_RECORDS] = {
         0x11110001u, 0x22220002u, 0x33330003u, 0x44440004u, 0u, 0x55550005u,
      };
      for (unsigned r = 0; r < SBT_RECORDS; ++r) {
         memcpy(sbt + r * SBT_STRIDE, &rec_handles[r], sizeof(uint32_t));
         memcpy(sbt + r * SBT_STRIDE + HK_RAY_TRACING_GROUP_HANDLE_SIZE,
                &rec_tags[r], sizeof(uint32_t));
      }

      struct nir_eval_env env = {
         .root_addr = UINT64_C(0x100000),
         .cmd_addr = UINT64_C(0x200000),
         .cmd = {
            .hitShaderBindingTableAddress = UINT64_C(0x10000000),
            .hitShaderBindingTableStride = SBT_STRIDE,
            .width = 8, .height = 4, .depth = 1,
         },
         .hit_sbt = sbt,
         .hit_sbt_size = sizeof(sbt),
      };

      /* Each sharing group's SBT index must read its own distinct SBT record
       * through the single merged closest-hit body (zero aliasing). */
      for (unsigned r = 0; r < 3; ++r) {
         env.sbt_index = r;
         const uint32_t got = (uint32_t)nir_eval_def(&chit_xor->def, 0, &env);
         CHECK(got == (rec_tags[r] ^ 0x51a7c0deu));
      }

      /* Walk to the enclosing nir_if guarding the merged closest-hit body and
       * verify it accepts hA, hB, hC and rejects hMiss, 0, and hC + 17. */
      nir_cf_node *child = &chit_xor->instr.block->cf_node;
      nir_cf_node *parent = child->parent;
      CHECK(parent && parent->type == nir_cf_node_if);
      nir_if *guard = nir_cf_node_as_if(parent);
      const bool in_then = cf_list_contains_node(&guard->then_list, child);
      CHECK(in_then || cf_list_contains_node(&guard->else_list, child));
      for (unsigned r = 0; r < SBT_RECORDS; ++r) {
         env.sbt_index = r;
         const bool cond =
            (nir_eval_def(guard->condition.ssa, 0, &env) & 1) != 0;
         const bool entered = in_then ? cond : !cond;
         CHECK(entered == (r < 3));
      }

      ralloc_free(captured_nir);
      captured_nir = NULL;
      destroy(p);

      /* Procedural hit groups: (isec, no-ahit) merges P1+P2 into 1 copy;
       * (isec, ahit) for P3 remains a 2nd distinct copy; chit merges to 1. */
      ci.stageCount = ARRAY_SIZE(stages);
      ci.pStages = stages;
      ci.groupCount = ARRAY_SIZE(proc_groups);
      ci.pGroups = proc_groups;
      capture_last_nir = true;
      CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
      capture_last_nir = false;
      CHECK(captured_nir != NULL);
      CHECK(find_xor_magic(captured_nir, 0x51a7c0deu, NULL) == 1);
      CHECK(find_xor_magic(captured_nir, 0x15ec0de1u, NULL) == 2);
      ralloc_free(captured_nir);
      captured_nir = NULL;
      destroy(p);
   }

   dev.dev.chip = saved_chip;
   dev.gen1_native_rq = saved_native;
   dev.gen1_rq_instances = saved_instances;
   for (unsigned i = 0; i < ARRAY_SIZE(mods); ++i)
      vk_common_DestroyShaderModule(device, mods[i], NULL);
   puts("PASS: shared hit groups merge stage call sites with distinct SBT "
        "records across G13G, G17P and M5");
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   vk_object_base_init(NULL, &instance.base, VK_OBJECT_TYPE_INSTANCE);
   instance.base.client_visible = true;
   list_inithead(&instance.debug_utils.callbacks);
   list_inithead(&instance.debug_report.callbacks);
   vk_object_base_init(&dev.vk, &dev.vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(NULL, &pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   instance.app_info.api_version = VK_API_VERSION_1_4;
   pdev.vk.instance = &instance;
   dev.vk.physical = &pdev.vk;
   dev.vk.alloc = *vk_default_allocator();
   dev.dev.chip = AGX_CHIP_G17P;
   dev.vk.robustness_state = (struct vk_pipeline_robustness_state){
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   dev.vk.enabled_features.rayTracingPipeline = true;
   dev.vk.enabled_features.shaderModuleIdentifier = true;
   dev.vk.enabled_features.shaderInt64 = true;
   pdev.vk.supported_features = dev.vk.enabled_features;
   pdev.vk.dispatch_table.GetPhysicalDeviceProperties = get_properties;
   pdev.vk.properties.subgroupSize = 32;
   pdev.vk.properties.minSubgroupSize = 32;
   pdev.vk.properties.maxSubgroupSize = 32;
   dev.vk.shader_ops = &fake_device_ops;
   device = hk_device_to_handle(&dev);
   layout = vk_pipeline_layout_zalloc(
      &dev.vk, sizeof(*layout),
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      });
   CHECK(layout);
   VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(raygen),
      .pCode = raygen,
   };
   CHECK(vk_common_CreateShaderModule(device, &module_info, NULL, &module) ==
         VK_SUCCESS);
   stage = (VkPipelineShaderStageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      .module = module,
      .pName = "main",
   };
   struct vk_pipeline_cache *empty =
      vk_pipeline_cache_create(&dev.vk,
                               &(struct vk_pipeline_cache_create_info){
                                  .force_enable = true,
                                  .skip_disk_cache = true,
                               },
                               NULL);
   struct vk_pipeline_cache *populated =
      vk_pipeline_cache_create(&dev.vk,
                               &(struct vk_pipeline_cache_create_info){
                                  .force_enable = true,
                                  .skip_disk_cache = true,
                               },
                               NULL);
   CHECK(empty && populated);
   const char key[] = "unrelated raw cache entry";
   struct vk_raw_data_cache_object *object = vk_raw_data_cache_object_create(
      &dev.vk, key, sizeof(key), raygen, sizeof(raygen));
   CHECK(object);
   vk_pipeline_cache_object_unref(
      &dev.vk, vk_pipeline_cache_add_object(populated, &object->base));
   CHECK(empty->object_cache->entries == 0);
   CHECK(populated->object_cache->entries == 1);
   VkPipelineCache caches[] = {VK_NULL_HANDLE,
                               vk_pipeline_cache_to_handle(empty),
                               vk_pipeline_cache_to_handle(populated)};
   VkPipelineShaderStageModuleIdentifierCreateInfoEXT identifier = {
      .sType =
         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT,
      .identifierSize = 1,
      .pIdentifier = (const uint8_t *)"x",
   };
   VkPipeline p;
   for (unsigned c = 0; c < ARRAY_SIZE(caches); c++) {
      /* caches[0] is VK_NULL_HANDLE, and this device has no internal cache. */
      const bool cached = c != 0;
      for (unsigned inline_code = 0; inline_code < 2; inline_code++) {
         stage.module = inline_code ? VK_NULL_HANDLE : module;
         stage.pNext = inline_code ? &module_info : NULL;
         /* Inline code hashes like its module: the second pass is warm. */
         const bool warm = cached && inline_code;
         if (warm)
            check_hit(caches[c], info(FAIL));
         else
            check_refusal(caches[c], info(FAIL));
         check_refusal(caches[c], info(FAIL | LIB));
         VkRayTracingPipelineCreateInfoKHR ci = info(0);
         unsigned t = translations, b = backends, r = restored;
         CHECK(create(caches[c], &ci, &p) == VK_SUCCESS);
         CHECK(translations == t + !warm && backends == b + !warm);
         CHECK(restored == r + warm);
         destroy(p);
         CHECK(live_nir == 0);
         /* Warming a pipeline cache creates an RT cache hit. */
         VkPipelineCreateFlags2CreateInfoKHR flags2 = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
            .flags = FAIL,
         };
         if (cached) {
            check_hit(caches[c], info(FAIL));
            ci.pNext = &flags2;
            check_hit(caches[c], ci);
         } else {
            check_refusal(caches[c], info(FAIL));
            ci.pNext = &flags2;
            check_refusal(caches[c], ci);
         }
      }
      stage.module = VK_NULL_HANDLE;
      stage.pNext = &identifier;
      check_refusal(caches[c], info(FAIL));
      check_refusal(caches[c], info(FAIL | LIB));
      if (cached) {
         /* The module's identifier names the cached stage. */
         VkShaderModuleIdentifierEXT id = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_IDENTIFIER_EXT};
         vk_common_GetShaderModuleIdentifierEXT(device, module, &id);
         VkPipelineShaderStageModuleIdentifierCreateInfoEXT module_id = {
            .sType =
               VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT,
            .identifierSize = id.identifierSize,
            .pIdentifier = id.identifier,
         };
         stage.pNext = &module_id;
         check_hit(caches[c], info(FAIL));
         /* Anything else that changes the code misses. */
         stage.module = module;
         stage.pNext = NULL;
         const VkSpecializationMapEntry entry = {.size = sizeof(uint32_t)};
         const uint32_t value = 1;
         const VkSpecializationInfo spec = {.mapEntryCount = 1,
                                            .pMapEntries = &entry,
                                            .dataSize = sizeof(value),
                                            .pData = &value};
         stage.pSpecializationInfo = &spec;
         check_refusal(caches[c], info(FAIL));
         stage.pSpecializationInfo = NULL;
         VkRayTracingPipelineCreateInfoKHR deeper = info(FAIL);
         deeper.maxPipelineRayRecursionDepth = 2;
         check_refusal(caches[c], deeper);
         const VkRayTracingShaderGroupCreateInfoKHR other_group = {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = 0,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
         };
         const VkRayTracingShaderGroupCreateInfoKHR two_groups[] = {group,
                                                                    other_group};
         VkRayTracingPipelineCreateInfoKHR wider = info(FAIL);
         wider.groupCount = 2;
         wider.pGroups = two_groups;
         check_refusal(caches[c], wider);
         VkPipelineCreateFlags2CreateInfoKHR capture = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
            .flags = FAIL |
                     VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
         };
         VkRayTracingPipelineCreateInfoKHR ir = info(0);
         ir.pNext = &capture;
         check_refusal(caches[c], ir);
      }
   }
   puts("PASS: warm caches hit by module, inline code and module identifier; "
        "changed code misses");
   /* The nonempty-ID cases without FAIL here are defensive function controls,
    * not valid application calls.  A module or inline code takes precedence
    * over an identifier in the translator.  A zero
    * identifier size with code must retain the same ordinary creation path.
    */
   for (unsigned size = 0; size < 2; size++) {
      identifier.identifierSize = size;
      for (unsigned inline_code = 0; inline_code < 2; inline_code++) {
         module_info.pNext = &identifier;
         stage.module = inline_code ? VK_NULL_HANDLE : module;
         stage.pNext = inline_code ? (const void *)&module_info : &identifier;
         check_refusal(VK_NULL_HANDLE, info(FAIL));
         VkRayTracingPipelineCreateInfoKHR code_ci = info(LIB);
         unsigned t = translations;
         CHECK(create(VK_NULL_HANDLE, &code_ci, &p) == VK_SUCCESS);
         CHECK(translations == t + 1);
         destroy(p);
         CHECK(live_nir == 0);
      }
   }
   identifier.identifierSize = 1;
   module_info.pNext = NULL;
   puts(
      "PASS: module/inline/identifier; null/empty/populated cache; warm call; flags2");

   stage.module = module;
   stage.pNext = NULL;
   VkRayTracingPipelineCreateInfoKHR ci = info(LIB);
   VkPipeline library;
   CHECK(create(VK_NULL_HANDLE, &ci, &library) == VK_SUCCESS);
   struct hk_ray_tracing_pipeline *lib =
      hk_ray_tracing_pipeline_from_handle(library);
   CHECK(lib->stages[0]->ref_cnt == 1 && live_nir == 1);
   VkPipelineLibraryCreateInfoKHR imports = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
      .libraryCount = 1,
      .pLibraries = &library,
   };
   ci = info(FAIL | LIB);
   ci.stageCount = ci.groupCount = 0;
   ci.pLibraryInfo = &imports;
   unsigned t = translations, b = backends;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
   CHECK(translations == t && backends == b && lib->stages[0]->ref_cnt == 2);
   destroy(p);
   CHECK(lib->stages[0]->ref_cnt == 1);
   ci.flags = FAIL;
   check_refusal(VK_NULL_HANDLE, ci);
   CHECK(lib->stages[0]->ref_cnt == 1); /* failure releases imported NIR */
   ci.flags = 0;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
   CHECK(translations == t && backends == b + 1);
   destroy(p);
   CHECK(lib->stages[0]->ref_cnt == 1);
   ci.flags = FAIL | LIB;
   ci.stageCount = ci.groupCount = 1;
   check_refusal(VK_NULL_HANDLE, ci); /* new local code plus imported NIR */
   puts("PASS: library reuse and executable link compilation boundary");

   /* A later stage failure must release earlier translated stages. */
   VkPipelineShaderStageCreateInfo stages[2] = {stage, stage};
   stages[1].module = VK_NULL_HANDLE;
   stages[1].pNext = &identifier;
   ci = info(LIB);
   ci.stageCount = 2;
   ci.pStages = stages;
   unsigned n = live_nir, refs = layout->ref_cnt;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_PIPELINE_COMPILE_REQUIRED);
   CHECK(p == VK_NULL_HANDLE && live_nir == n && layout->ref_cnt == refs);
   stages[1] = stage;
   fail_translation = translations + 2;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_OUT_OF_HOST_MEMORY);
   CHECK(p == VK_NULL_HANDLE && live_nir == n && layout->ref_cnt == refs);
   fail_translation = 0;
   backend_result = VK_ERROR_OUT_OF_HOST_MEMORY;
   ci = info(0);
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_OUT_OF_HOST_MEMORY);
   CHECK(p == VK_NULL_HANDLE && live_nir == n && layout->ref_cnt == refs);
   backend_result = VK_SUCCESS;
   puts("PASS: partial stage, translation failure and backend failure cleanup");

   for (unsigned early = 0; early < 2; early++) {
      VkRayTracingPipelineCreateInfoKHR batch[] = {
         info(LIB),
         info(FAIL | (early ? EARLY : 0)),
         info(LIB),
      };
      VkPipeline pipelines[3] = {(VkPipeline)1, (VkPipeline)1, (VkPipeline)1};
      t = translations;
      /* No cache, so the FAIL entry cannot hit. */
      CHECK(hk_CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, caches[0],
                                            3, batch, &alloc, pipelines) ==
            VK_PIPELINE_COMPILE_REQUIRED);
      CHECK(pipelines[0] != VK_NULL_HANDLE && pipelines[1] == VK_NULL_HANDLE);
      CHECK((pipelines[2] == VK_NULL_HANDLE) == (early != 0));
      CHECK(translations == t + (early ? 1 : 2));
      for (unsigned i = 0; i < 3; i++)
         destroy(pipelines[i]);
   }
   puts(
      "PASS: mixed batches, success preservation, EARLY_RETURN and skipped null handles");
   /* CTS misc.shaders_from_lib creates stage-only libraries and supplies
    * groups when linking, with recursion depth 10 in all three creates. */
   for (unsigned depth = 10; depth <= HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH;
        depth += HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH - 10) {
      VkPipeline libs[2];
      ci = info(LIB);
      ci.maxPipelineRayRecursionDepth = depth;
      ci.groupCount = 0;
      ci.pGroups = NULL;
      CHECK(create(VK_NULL_HANDLE, &ci, &libs[0]) == VK_SUCCESS);
      CHECK(create(VK_NULL_HANDLE, &ci, &libs[1]) == VK_SUCCESS);
      VkPipelineLibraryCreateInfoKHR imports = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
         .libraryCount = 2, .pLibraries = libs,
      };
      ci = info(0);
      ci.maxPipelineRayRecursionDepth = depth;
      ci.stageCount = 0;
      ci.pStages = NULL;
      ci.pLibraryInfo = &imports;
      CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
      struct hk_ray_tracing_pipeline *linked =
         hk_ray_tracing_pipeline_from_handle(p);
      CHECK(linked->stage_count == 2 && linked->group_count == 1);
      destroy(libs[0]);
      destroy(libs[1]);
      destroy(p);
   }
   ci = info(0);
   ci.maxPipelineRayRecursionDepth = HK_G17P_MAX_REQUESTED_RAY_RECURSION_DEPTH + 1;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(p == VK_NULL_HANDLE);
   puts("PASS: stage-only libraries, link-time groups, legacy-client depth 10/31 and overflow");

   /* A library loaded from a cache takes the handles of the latest identical
    * library on this device. An executable linking libraries hits only while
    * their handles are the ones compiled into the entry. */
   {
      /* `library` above is still alive with its stage NIR. */
      const unsigned live0 = live_nir;
      struct vk_pipeline_cache *lc = vk_pipeline_cache_create(
         &dev.vk,
         &(struct vk_pipeline_cache_create_info){.force_enable = true,
                                                 .skip_disk_cache = true},
         NULL);
      CHECK(lc);
      VkPipelineCache lch = vk_pipeline_cache_to_handle(lc);
      VkPipeline l1, l2, l3;
      ci = info(LIB);
      CHECK(create(lch, &ci, &l1) == VK_SUCCESS);
      ci = info(FAIL | LIB);
      t = translations;
      CHECK(create(lch, &ci, &l2) == VK_SUCCESS && translations == t);
      CHECK(rt(l2)->groups[0].handle.index == rt(l1)->groups[0].handle.index);
      VkPipelineLibraryCreateInfoKHR link = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
         .libraryCount = 1,
         .pLibraries = &l1,
      };
      ci = info(0);
      ci.stageCount = ci.groupCount = 0;
      ci.pLibraryInfo = &link;
      b = backends;
      CHECK(create(lch, &ci, &p) == VK_SUCCESS && backends == b + 1);
      CHECK(rt(p)->groups[0].handle.index == rt(l1)->groups[0].handle.index);
      destroy(p);
      link.pLibraries = &l2;
      ci.flags = FAIL;
      check_hit(lch, ci);
      ci = info(LIB);
      CHECK(create(VK_NULL_HANDLE, &ci, &l3) == VK_SUCCESS);
      CHECK(rt(l3)->groups[0].handle.index != rt(l1)->groups[0].handle.index);
      link.pLibraries = &l3;
      ci = info(FAIL);
      ci.stageCount = ci.groupCount = 0;
      ci.pLibraryInfo = &link;
      check_refusal(lch, ci);
      destroy(l1);
      destroy(l2);
      destroy(l3);
      vk_pipeline_cache_destroy(lc, NULL);
      CHECK(live_nir == live0);
   }
   puts("PASS: cached library handles and executable keys over linked handles");
   /* M1-M3 retain the legacy limit even for the same library request. */
   dev.dev.chip = AGX_CHIP_G13G;
   ci = info(LIB);
   ci.maxPipelineRayRecursionDepth = HK_MAX_RAY_RECURSION_DEPTH + 1;
   t = translations;
   CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(p == VK_NULL_HANDLE && translations == t);
   dev.dev.chip = AGX_CHIP_G17P;

   /* Exercise the real G17P loader and its SPECTPR spill frame. The nested
    * callable CTS has a 256-element local array and reserves 31 frames. */
   struct agx3_g17p_compute_program_desc desc = {
      .program_va = 0x100000, .shader_va = 0x200000,
      .shader_binary_size = 64, .nr_gprs = 32,
      .compiler_d_valid = true, .compiler_d = 64,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   CHECK(agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &desc.launch_sources));
   const uint32_t frames[] = {0, 8192, 31 * 1088, UINT16_MAX * 4u};
   for (unsigned i = 0; i < ARRAY_SIZE(frames); ++i) {
      desc.scratch_size = frames[i];
      struct agx3_g17p_compute_program program;
      CHECK(agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &program));
      uint8_t spectpr[AGX_IOTO_SPECTPR_SIZE];
      CHECK(agx_ioto_g17p_compiler_spectpr_pack(frames[i], desc.compiler_d,
                                               spectpr, sizeof(spectpr)));
      bool found = false;
      for (unsigned j = 0; j + sizeof(spectpr) <= program.length; ++j)
         found |= !memcmp(program.bytes + j, spectpr, sizeof(spectpr));
      CHECK(found);
   }
   desc.scratch_size = UINT16_MAX * 4u + 4u;
   struct agx3_g17p_compute_program program;
   CHECK(!agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &program));
   desc.scratch_size = 8193;
   CHECK(!agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, &program));
   puts("PASS: G17P callable-size spill frames and encoding bounds");

   /* Continuation mode: raygen, closest-hit and miss trace from every level,
    * with an any-hit and a recursive callable. Above depth 3 every stage and
    * the traversal appear once, so the code size does not grow with the
    * depth, and the stack grows by one closest-hit/miss frame and record per
    * level. */
   {
      static const struct {
         const uint32_t *code;
         size_t size;
         VkShaderStageFlagBits stage;
      } spv[] = {
         {hk_rt_recursion_rgen_spv, sizeof(hk_rt_recursion_rgen_spv),
          VK_SHADER_STAGE_RAYGEN_BIT_KHR},
         {hk_rt_recursion_chit_spv, sizeof(hk_rt_recursion_chit_spv),
          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
         {hk_rt_recursion_miss_spv, sizeof(hk_rt_recursion_miss_spv),
          VK_SHADER_STAGE_MISS_BIT_KHR},
         {hk_rt_recursion_ahit_spv, sizeof(hk_rt_recursion_ahit_spv),
          VK_SHADER_STAGE_ANY_HIT_BIT_KHR},
         {hk_rt_recursion_call_spv, sizeof(hk_rt_recursion_call_spv),
          VK_SHADER_STAGE_CALLABLE_BIT_KHR},
      };
      VkShaderModule mods[ARRAY_SIZE(spv)];
      VkPipelineShaderStageCreateInfo stages[ARRAY_SIZE(spv)];
      for (unsigned i = 0; i < ARRAY_SIZE(spv); ++i) {
         VkShaderModuleCreateInfo mi = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spv[i].size,
            .pCode = spv[i].code,
         };
         CHECK(vk_common_CreateShaderModule(device, &mi, NULL, &mods[i]) ==
               VK_SUCCESS);
         stages[i] = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = spv[i].stage,
            .module = mods[i],
            .pName = "main",
         };
      }
      VkRayTracingShaderGroupCreateInfoKHR groups[] = {group, group, group, group};
      groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      groups[1].generalShader = VK_SHADER_UNUSED_KHR;
      groups[1].closestHitShader = 1;
      groups[1].anyHitShader = 3;
      groups[2].generalShader = 2;
      groups[3].generalShader = 4;

      const uint32_t depths[] = {3, 4, 8, 31};
      uint32_t stack[ARRAY_SIZE(depths)], instrs[ARRAY_SIZE(depths)], cm = 0;
      for (unsigned d = 0; d < ARRAY_SIZE(depths); ++d) {
         ci = info(0);
         ci.stageCount = ARRAY_SIZE(stages);
         ci.pStages = stages;
         ci.groupCount = ARRAY_SIZE(groups);
         ci.pGroups = groups;
         ci.maxPipelineRayRecursionDepth = depths[d];
         b = backends;
         CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
         CHECK(backends == b + 1 && last_calls == 0);
         struct hk_ray_tracing_pipeline *rp =
            hk_ray_tracing_pipeline_from_handle(p);
         stack[d] = rp->stack_size;
         instrs[d] = last_instrs;
         if (depths[d] > HK_RT_INLINE_MAX_RAY_RECURSION_DEPTH) {
            CHECK(rp->stack_sizes.closest_hit >= HK_RAY_HIT_ATTRIBS_SIZE);
            cm = ((MAX2(rp->stack_sizes.closest_hit, rp->stack_sizes.miss) +
                   15) & ~15u) + HK_RT_RETURN_RECORD_SIZE;
         }
         fprintf(stderr, "continuation test: depth %u: %u instructions, "
                 "stack %u bytes\n", depths[d], last_instrs, stack[d]);
         destroy(p);
      }
      CHECK(stack[3] - stack[2] == 23 * cm && stack[2] - stack[1] == 4 * cm);
      /* Group handles and the stack limit are constants that differ between
       * the pipelines, so folding may differ by a few instructions; a copy
       * per depth level would be orders of magnitude more. */
#define NEAR(a, b) ((a) <= (b) + (b) / 50 && (b) <= (a) + (a) / 50)
      CHECK(NEAR(instrs[1], instrs[2]) && NEAR(instrs[1], instrs[3]));
#undef NEAR

      /* Many callable groups: continuations find a group's entry and a part
       * by balanced search, so no block grows with the number of groups. */
      unsigned max_block[2];
      const unsigned callables[2] = {8, 48};
      for (unsigned k = 0; k < 2; ++k) {
         const unsigned n = 3 + callables[k];
         VkPipelineShaderStageCreateInfo many[3 + 48];
         VkRayTracingShaderGroupCreateInfoKHR many_groups[3 + 48];
         for (unsigned i = 0; i < n; ++i) {
            /* raygen, closest-hit, miss, then the callables */
            many[i] = stages[i < 3 ? i : 4];
            many_groups[i] = i < 3 ? groups[i] : groups[3];
            if (i >= 3)
               many_groups[i].generalShader = i;
         }
         many_groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
         ci = info(0);
         ci.stageCount = n;
         ci.pStages = many;
         ci.groupCount = n;
         ci.pGroups = many_groups;
         ci.maxPipelineRayRecursionDepth = 4;
         CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS && last_calls == 0);
         max_block[k] = last_max_block;
         fprintf(stderr, "continuation test: %u callable groups: largest block "
                 "%u instructions\n", callables[k], max_block[k]);
         destroy(p);
      }
      CHECK(max_block[1] <= max_block[0] + 16);

      struct hk_rt_stack_sizes sizes = {
         .raygen = 40, .closest_hit = 72, .miss = 20, .any_hit = 32,
         .intersection = 48, .callable = 100,
      };
      uint32_t leaf;
      CHECK(hk_rt_continuation_stack_size(&sizes, 5, false, 0, &leaf) ==
            32 + (48 + 32) + 5 * (80 + 32) + 112 && leaf == 112);
      CHECK(hk_rt_continuation_stack_size(&sizes, 5, true, 0, &leaf) ==
            32 + (48 + 32) + 5 * (80 + 32) + 31 * (112 + 32) + 112);
      CHECK(hk_rt_continuation_stack_size(&sizes, 5, false, 72, &leaf) ==
            32 + (48 + 32) + 5 * (80 + 32) + 160 && leaf == 80 + 80);
      /* Verify real compiled NIR launch-ID reconstruction and OOB guard across
       * partial edge tiles at heights 1, 2, 4, and 5. */
      ci = info(0);
      ci.stageCount = ARRAY_SIZE(stages);
      ci.pStages = stages;
      ci.groupCount = ARRAY_SIZE(groups);
      ci.pGroups = groups;
      ci.maxPipelineRayRecursionDepth = 1;
      inspect_nir = inspect_tiled_launch_oob;
      CHECK(create(VK_NULL_HANDLE, &ci, &p) == VK_SUCCESS);
      inspect_nir = NULL;
      destroy(p);
      puts("PASS: compiled NIR launch-ID OOB guard covers partial edge tiles at "
           "heights 1, 2, 4 and 5");
      test_shared_stage_call_sites();

      for (unsigned i = 0; i < ARRAY_SIZE(mods); ++i)
         vk_common_DestroyShaderModule(device, mods[i], NULL);
      puts("PASS: continuation mode above depth 3: no calls, code size "
           "independent of depth, linear stack");
   }

   destroy(library);
   CHECK(live_nir == 0 && pipeline_allocs == pipeline_frees);
   CHECK(layout->ref_cnt == 1);
   /* Every compiled or restored shader is destroyed, less the injected
    * backend failure. */
   CHECK(shader_destroys == backends - 1 + restored);
   hk_rt_device_finish(&dev);
   vk_pipeline_layout_unref(&dev.vk, layout);
   vk_common_DestroyShaderModule(device, module, NULL);
   vk_pipeline_cache_destroy(empty, NULL);
   vk_pipeline_cache_destroy(populated, NULL);
   vk_object_base_finish(&dev.vk.base);
   vk_object_base_finish(&pdev.vk.base);
   vk_object_base_finish(&instance.base);
   glsl_type_singleton_decref();
   puts(
      "PASS: all pipeline allocations, stage NIR, layout and shader references released");
   return 0;
}
