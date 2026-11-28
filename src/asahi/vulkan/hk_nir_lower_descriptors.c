/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "pipe/p_defines.h"
#include "util/u_debug.h"
#include "util/set.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"
#include "agx_abi.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_sampler.h"
#include "hk_sampler_index.h"
#include "hk_native_descriptor.h"
#include "hk_shader.h"
#include "hk_nir_sampler_bias.h"
#include "layout/layout.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"
#include "vk_pipeline.h"

struct lower_descriptors_ctx {
   const struct hk_descriptor_set_layout *set_layouts[HK_MAX_SETS];

   bool clamp_desc_array_bounds;
   bool sparse_buffer_residency;
   /* Unbound pages are unmapped and soft faults read them as zero, so only
    * stores and atomics consult the residency map. */
   bool sparse_soft_unmapped;
   /* With sparse_soft_unmapped: the stores whose binding the shader also
    * reads. */
   struct set *sparse_guarded_stores;
   nir_address_format ubo_addr_format;
   nir_address_format ssbo_addr_format;
   uint64_t sparse_ro_offset;
   unsigned descriptor_generation;
   bool scaled_sampling;
   /* Whether the hardware sampler word applies the API LOD bias. */
   bool sampler_has_lod_bias;
   bool d3d_nearest_mip;
   /* Coherent image loads and stores access memory directly instead of the
    * texture cache and PBE (see image_access_needs_memory_path()). */
   bool coherent_images_in_memory;
   /* Layered stores and atomics check the M5 store window record
    * (lower_m5_store_window()). */
   bool m5_store_window;
   /* AGX3D hardware LOD queries do not support array or cube textures;
    * lower fragment array/cube nir_texop_lod from derivatives and sampler
    * descriptor fields. */
   bool lower_array_cube_lod_query;
   mesa_shader_stage stage;
};

static bool
storage_address_has_map(const struct lower_descriptors_ctx *ctx)
{
   return ctx->ssbo_addr_format == nir_address_format_64bit_global_32bit_offset_map ||
          ctx->ssbo_addr_format == nir_address_format_64bit_bounded_global_map;
}

static nir_def *
storage_address(nir_builder *b, nir_def *desc,
                const struct lower_descriptors_ctx *ctx)
{
   nir_def *parts[] = {nir_channel(b, desc, 0), nir_channel(b, desc, 1),
      nir_channel(b, desc, 2), nir_imm_int(b, 0),
      ctx->sparse_buffer_residency ? nir_channel(b, desc, 3) : nir_imm_int(b, 0)};
   return nir_vec(b, parts, 5);
}

static const struct hk_descriptor_set_binding_layout *
get_binding_layout(uint32_t set, uint32_t binding,
                   const struct lower_descriptors_ctx *ctx)
{
   assert(set < HK_MAX_SETS);
   assert(ctx->set_layouts[set] != NULL);

   const struct hk_descriptor_set_layout *set_layout = ctx->set_layouts[set];

   assert(binding < set_layout->binding_count);
   return &set_layout->binding[binding];
}

static bool
layout_has_binding(uint32_t set, uint32_t binding, uint32_t set_layout_count,
                   struct vk_descriptor_set_layout *const *set_layouts)
{
   return set < set_layout_count && set_layouts[set] != NULL &&
          binding <
             vk_to_hk_descriptor_set_layout(set_layouts[set])->binding_count;
}

/* Every descriptor the shader reaches must exist in the pipeline layout.
 * Vulkan makes the opposite invalid usage, but descriptor lowering would
 * dereference the missing set layout and kill the process. */
bool
hk_nir_layout_has_descriptors(const nir_shader *nir, uint32_t set_layout_count,
                              struct vk_descriptor_set_layout *const *set_layouts)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_vulkan_resource_index &&
                   !layout_has_binding(nir_intrinsic_desc_set(intr),
                                       nir_intrinsic_binding(intr),
                                       set_layout_count, set_layouts))
                  return false;
            } else if (instr->type == nir_instr_type_deref) {
               const nir_deref_instr *deref = nir_instr_as_deref(instr);
               if (deref->deref_type != nir_deref_type_var ||
                   !(deref->var->data.mode & (nir_var_uniform | nir_var_image)))
                  continue;
               const struct glsl_type *type = glsl_without_array(deref->type);
               if ((glsl_type_is_image(type) || glsl_type_is_sampler(type) ||
                    glsl_type_is_texture(type)) &&
                   !layout_has_binding(deref->var->data.descriptor_set,
                                       deref->var->data.binding,
                                       set_layout_count, set_layouts))
                  return false;
            }
         }
      }
   }
   return true;
}

static nir_def *
load_speculatable(nir_builder *b, unsigned num_components, unsigned bit_size,
                  nir_def *addr, unsigned align)
{
   return nir_load_global_constant(b, num_components, bit_size, addr,
                                   .align_mul = align,
                                   .access = ACCESS_CAN_SPECULATE);
}

static nir_def *
load_root(nir_builder *b, unsigned num_components, unsigned bit_size,
          nir_def *offset, unsigned align)
{
   nir_def *addr = nir_iadd(b, nir_load_root_agx(b), nir_u2u64(b, offset));

   return load_speculatable(b, num_components, bit_size, addr, align);
}

static bool
lower_load_constant(nir_builder *b, nir_intrinsic_instr *load,
                    const struct lower_descriptors_ctx *ctx)
{
   assert(load->intrinsic == nir_intrinsic_load_constant);

   unsigned root_offset;
   switch (ctx->stage) {
   case MESA_SHADER_COMPUTE:
      root_offset = hk_root_descriptor_offset(cs.constant_data_addr);
      break;
   case MESA_SHADER_VERTEX:
      root_offset = hk_root_descriptor_offset(draw.vs_constant_data_addr);
      break;
   case MESA_SHADER_FRAGMENT:
      root_offset = hk_root_descriptor_offset(draw.fs_constant_data_addr);
      break;
   default:
      return false;
   }

   uint32_t base = nir_intrinsic_base(load);

   b->cursor = nir_before_instr(&load->instr);

   nir_def *constant_data =
      load_root(b, 1, 64, nir_imm_int(b, root_offset), 8);
   nir_def *offset = nir_iadd_imm(b, load->src[0].ssa, base);
   nir_def *addr = nir_iadd(b, constant_data, nir_u2u64(b, offset));
   nir_def *data = nir_load_global_constant(
      b, load->def.num_components, load->def.bit_size, addr,
      .align_mul = nir_intrinsic_align_mul(load),
      .align_offset = nir_intrinsic_align_offset(load),
      .access = ACCESS_CAN_SPECULATE);

   nir_def_rewrite_uses(&load->def, data);

   return true;
}

static nir_def *
load_dynamic_buffer_start(nir_builder *b, uint32_t set,
                          const struct lower_descriptors_ctx *ctx)
{
   int dynamic_buffer_start_imm = 0;
   for (uint32_t s = 0; s < set; s++) {
      if (ctx->set_layouts[s] == NULL) {
         dynamic_buffer_start_imm = -1;
         break;
      }

      dynamic_buffer_start_imm += ctx->set_layouts[s]->vk.dynamic_descriptor_count;
   }

   if (dynamic_buffer_start_imm >= 0) {
      return nir_imm_int(b, dynamic_buffer_start_imm);
   } else {
      uint32_t root_offset =
         hk_root_descriptor_offset(set_dynamic_buffer_start) + set;

      return nir_u2u32(b, load_root(b, 1, 8, nir_imm_int(b, root_offset), 1));
   }
}

static nir_def *
load_descriptor(nir_builder *b, unsigned num_components, unsigned bit_size,
                uint32_t set, uint32_t binding, nir_def *index,
                unsigned offset_B, bool buffer_binding,
                const struct lower_descriptors_ctx *ctx)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      get_binding_layout(set, binding, ctx);
   /* A mutable descriptor read as a storage buffer holds the same record as a
    * storage buffer descriptor; past it are the bytes of whatever it held
    * before. */
   const bool map_address = storage_address_has_map(ctx) &&
      (binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
       binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
       (binding_layout->type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
        num_components == 5));
   const unsigned load_components = map_address ? 4 : num_components;

   if (ctx->clamp_desc_array_bounds) {
      if (nir_def_is_const(index)) {
         index = nir_imm_int(b, MIN2(nir_src_as_uint(nir_src_for_ssa(index)),
                                     binding_layout->array_size - 1));
      } else {
         index =
            nir_umin(b, index, nir_imm_int(b, binding_layout->array_size - 1));
      }
   }

   switch (binding_layout->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
      /* Statically indexed non-partially-bound dynamic descriptors are valid
       * (or null with size == 0) on every path, matching static buffers below.
       */
      const bool always_valid =
         ctx->descriptor_generation == AGX_DESCRIPTOR_GENERATION_9 &&
         nir_def_is_const(index) &&
         !(binding_layout->flags &
           (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT));

      /* Get the index in the root descriptor table dynamic_buffers array. */
      nir_def *dynamic_buffer_start = load_dynamic_buffer_start(b, set, ctx);

      index = nir_iadd(b, index,
                       nir_iadd_imm(b, dynamic_buffer_start,
                                    binding_layout->dynamic_buffer_index));

      nir_def *root_desc_offset = nir_iadd_imm(
         b, nir_amul_imm(b, index, sizeof(struct hk_buffer_address)),
         hk_root_descriptor_offset(dynamic_buffers));

      assert(num_components == (map_address ? 5 : 4) && bit_size == 32);
      nir_def *desc = load_root(b, 4, 32, root_desc_offset, 16);
      if (buffer_binding || always_valid) {
         nir_intrinsic_instr *desc_intr = nir_def_as_intrinsic(desc);
         nir_intrinsic_set_access(
            desc_intr, nir_intrinsic_access(desc_intr) |
                          (always_valid ? ACCESS_BUFFER_DESCRIPTOR_AGX : 0) |
                          (buffer_binding ? ACCESS_BUFFER_BINDING_AGX : 0));
      }

      if (map_address)
         return storage_address(b, desc, ctx);

      if (binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC &&
          ctx->sparse_buffer_residency &&
          (ctx->descriptor_generation == 9 || ctx->descriptor_generation == 10))
         return desc;

      /* We know a priori that the .w component (offset) is zero. */
      return nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 3);
   }

   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
      nir_def *base_addr = nir_iadd_imm(b, nir_load_descriptor_set_agx(b, set),
                                        binding_layout->offset);

      assert(binding_layout->stride == 1);
      const uint32_t binding_size = binding_layout->array_size;

      /* Convert it to nir_address_format_64bit_bounded_global */
      assert(num_components == (map_address ? 5 : 4) && bit_size == 32);
      return nir_vec4(b, nir_unpack_64_2x32_split_x(b, base_addr),
                      nir_unpack_64_2x32_split_y(b, base_addr),
                      nir_imm_int(b, binding_size), nir_imm_int(b, 0));
   }

   default: {
      assert(binding_layout->stride > 0);
      nir_def *desc_ubo_offset =
         nir_iadd_imm(b, nir_amul_imm(b, index, binding_layout->stride),
                      binding_layout->offset + offset_B);

      unsigned desc_align_mul = (1 << (ffs(binding_layout->stride) - 1));
      desc_align_mul = MIN2(desc_align_mul, 16);
      unsigned desc_align_offset = binding_layout->offset + offset_B;
      desc_align_offset %= desc_align_mul;

      /* Unless it may be partly bound, a statically used descriptor is valid
       * on every path, so G17P can hoist constant reads through it past the
       * branches that guard them, see agx_nir_opt_preamble.
       */
      const bool always_valid =
         ctx->descriptor_generation == AGX_DESCRIPTOR_GENERATION_9 &&
         nir_def_is_const(index) &&
         (binding_layout->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
          binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) &&
         !(binding_layout->flags &
           (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT));

      nir_def *desc;
      nir_def *set_addr = nir_load_descriptor_set_agx(b, set);
      desc = nir_load_global_constant_offset(
         b, load_components, bit_size, set_addr, desc_ubo_offset,
         .align_mul = desc_align_mul, .align_offset = desc_align_offset,
         .access = ACCESS_CAN_SPECULATE |
                   (always_valid ? ACCESS_BUFFER_DESCRIPTOR_AGX : 0) |
                   (buffer_binding ? ACCESS_BUFFER_BINDING_AGX : 0));

      if (binding_layout->type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          map_address)
         return storage_address(b, desc, ctx);

      if (binding_layout->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
          binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
         assert(num_components == (map_address ? 5 : 4) && bit_size == 32);
         if (map_address)
            return storage_address(b, desc, ctx);

         if (!(binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
               ctx->sparse_buffer_residency &&
               (ctx->descriptor_generation == 9 ||
                ctx->descriptor_generation == 10))) {
            /* We know a priori that the .w component (offset) is zero. */
            desc = nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 3);
         }
      }
      return desc;
   }
   }
}

static bool
is_idx_intrin(nir_intrinsic_instr *intrin)
{
   while (intrin->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
      intrin = nir_src_as_intrinsic(intrin->src[0]);
      if (intrin == NULL)
         return false;
   }

   return intrin->intrinsic == nir_intrinsic_vulkan_resource_index;
}

static nir_def *
load_descriptor_for_idx_intrin(nir_builder *b, nir_intrinsic_instr *intrin,
                               const struct lower_descriptors_ctx *ctx)
{
   nir_def *index = nir_imm_int(b, 0);

   /* load_descriptor tells a constant index from a dynamic one. */
   b->constant_fold_alu = true;
   while (intrin->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
      index = nir_iadd(b, index, intrin->src[1].ssa);
      intrin = nir_src_as_intrinsic(intrin->src[0]);
   }

   assert(intrin->intrinsic == nir_intrinsic_vulkan_resource_index);
   uint32_t set = nir_intrinsic_desc_set(intrin);
   uint32_t binding = nir_intrinsic_binding(intrin);
   index = nir_iadd(b, index, intrin->src[0].ssa);
   b->constant_fold_alu = false;

   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);

   /* A buffer descriptor is a vec4 (address, size, ...); an acceleration
    * structure is a bare 64-bit BVH address, and rq_initialize consumes it
    * as one.  Loading it as a vec4 would both read past the descriptor and
    * hand the ray-query lowering the wrong-shaped value. */
   if (desc_type == nir_descriptor_type_acceleration_structure)
      return load_descriptor(b, 1, 64, set, binding, index, 0, false, ctx);

   const unsigned components =
      desc_type == nir_descriptor_type_storage_buffer &&
      storage_address_has_map(ctx) ? 5 : 4;
   nir_def *desc =
      load_descriptor(b, components, 32, set, binding, index, 0, true, ctx);
   if (desc_type == nir_descriptor_type_uniform_buffer &&
       !nir_scalar_is_const(nir_scalar_resolved(desc, 3))) {
      /* Every uniform buffer descriptor (including mutable slots written as
       * VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) leaves the .w offset word zero. */
      desc = nir_vector_insert_imm(b, desc, nir_imm_int(b, 0), 3);
   }
   return desc;
}

static bool
try_lower_load_vulkan_descriptor(nir_builder *b, nir_intrinsic_instr *intrin,
                                 const struct lower_descriptors_ctx *ctx)
{
   ASSERTED const nir_descriptor_type desc_type =
      nir_intrinsic_desc_type(intrin);
   b->cursor = nir_before_instr(&intrin->instr);

   nir_intrinsic_instr *idx_intrin = nir_src_as_intrinsic(intrin->src[0]);
   if (idx_intrin == NULL || !is_idx_intrin(idx_intrin)) {
      assert(desc_type == nir_descriptor_type_storage_buffer);
      return false;
   }

   nir_def *desc = load_descriptor_for_idx_intrin(b, idx_intrin, ctx);

   nir_def_rewrite_uses(&intrin->def, desc);

   return true;
}

static bool
_lower_sysval_to_root_table(nir_builder *b, nir_intrinsic_instr *intrin,
                            uint32_t root_table_offset)
{
   b->cursor = nir_instr_remove(&intrin->instr);
   assert((root_table_offset & 3) == 0 && "aligned");

   nir_def *val = load_root(b, intrin->def.num_components, intrin->def.bit_size,
                            nir_imm_int(b, root_table_offset), 4);

   nir_def_rewrite_uses(&intrin->def, val);

   return true;
}

#define lower_sysval_to_root_table(b, intrin, member)                          \
   _lower_sysval_to_root_table(b, intrin, hk_root_descriptor_offset(member))

static bool
lower_load_push_constant(nir_builder *b, nir_intrinsic_instr *load,
                         const struct lower_descriptors_ctx *ctx)
{
   const uint32_t push_region_offset = hk_root_descriptor_offset(push);
   const uint32_t base = nir_intrinsic_base(load);

   b->cursor = nir_before_instr(&load->instr);

   nir_def *offset =
      nir_iadd_imm(b, load->src[0].ssa, push_region_offset + base);

   nir_def *val = load_root(b, load->def.num_components, load->def.bit_size,
                            offset, load->def.bit_size / 8);

   nir_def_rewrite_uses(&load->def, val);

   return true;
}

static void
get_resource_deref_binding(nir_builder *b, nir_deref_instr *deref,
                           uint32_t *set, uint32_t *binding, nir_def **index)
{
   if (deref->deref_type == nir_deref_type_array) {
      *index = deref->arr.index.ssa;
      deref = nir_deref_instr_parent(deref);
   } else {
      *index = nir_imm_int(b, 0);
   }

   assert(deref->deref_type == nir_deref_type_var);
   nir_variable *var = deref->var;

   *set = var->data.descriptor_set;
   *binding = var->data.binding;
}

static nir_def *
load_resource_deref_desc(nir_builder *b, unsigned num_components,
                         unsigned bit_size, nir_deref_instr *deref,
                         unsigned offset_B,
                         const struct lower_descriptors_ctx *ctx)
{
   uint32_t set, binding;
   nir_def *index;
   get_resource_deref_binding(b, deref, &set, &binding, &index);
   return load_descriptor(b, num_components, bit_size, set, binding, index,
                          offset_B, false, ctx);
}

static nir_def *
load_image_handle(nir_builder *b, const struct lower_descriptors_ctx *ctx,
                  nir_src src, unsigned offset_B)

{
   uint32_t set, binding;
   nir_def *index;
   get_resource_deref_binding(b, nir_src_as_deref(src), &set, &binding, &index);

   const struct hk_descriptor_set_binding_layout *binding_layout =
      get_binding_layout(set, binding, ctx);

   if (ctx->clamp_desc_array_bounds) {
      if (nir_def_is_const(index)) {
         index = nir_imm_int(b, MIN2(nir_src_as_uint(nir_src_for_ssa(index)),
                                     binding_layout->array_size - 1));
      } else {
         index =
            nir_umin(b, index, nir_imm_int(b, binding_layout->array_size - 1));
      }
   }

   assert(binding_layout->stride > 0);
   nir_def *desc_offs_B =
      nir_iadd_imm(b, nir_amul_imm(b, index, binding_layout->stride),
                   binding_layout->offset + offset_B);

   return nir_bindless_image_agx(b, desc_offs_B, .desc_set = set);
}

/*
 * On GPUs with several clusters and the AGX2 cache operations (G13 and G14
 * Pro/Max/Ultra), PBE stores and texture-path loads are not coherent between
 * clusters: with the full device-scope release/acquire barrier train, an
 * image texel written on one cluster can still read stale on another (the
 * message-passing memory-model tests fail there and pass on the
 * single-cluster G14G). Device memory accesses that are encoded coherent do
 * reach the other clusters. So coherent and volatile image loads and stores
 * (ACCESS_COHERENT also marks the accesses that MakeAvailable/MakeVisible
 * cover, see nir_lower_memory_model) access the texel in memory directly,
 * through the address computation that image atomics use. Only plain
 * 32-bit-channel formats, whose texels are their memory words, take this
 * path; storage images are never compressed.
 */
static bool
image_access_needs_memory_path(nir_intrinsic_instr *intr,
                               const struct lower_descriptors_ctx *ctx,
                               enum pipe_format *format_out)
{
   if (!ctx->coherent_images_in_memory)
      return false;

   if (intr->intrinsic != nir_intrinsic_image_deref_load &&
       intr->intrinsic != nir_intrinsic_image_deref_store)
      return false;

   if (!(nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE)))
      return false;

   switch (nir_intrinsic_image_dim(intr)) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_3D:
      break;
   default:
      return false;
   }

   enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_NONE) {
      nir_variable *var = nir_intrinsic_get_var(intr, 0);
      if (var)
         format = var->data.image.format;
   }

   switch (format) {
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
      *format_out = format;
      return true;
   default:
      return false;
   }
}

static void
lower_image_load_from_memory(nir_builder *b, nir_intrinsic_instr *intr,
                             enum pipe_format format,
                             const struct lower_descriptors_ctx *ctx)
{
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(ctx->descriptor_generation);
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool array = nir_intrinsic_image_array(intr);
   const unsigned coord_comps =
      glsl_get_sampler_dim_coordinate_components(dim) + (array ? 1 : 0);
   const unsigned channels = util_format_get_nr_components(format);
   const enum gl_access_qualifier access = nir_intrinsic_access(intr);

   nir_def *pbe = load_image_handle(b, ctx, intr->src[0], native.pbe);
   nir_def *texture = load_image_handle(b, ctx, intr->src[0], native.texture);
   nir_def *coord = nir_trim_vector(b, intr->src[1].ssa, coord_comps);

   /* Out-of-bounds texels read as zero, as robust image loads do. */
   nir_def *size = nir_bindless_image_size(b, coord_comps, 32, texture,
                                           nir_imm_int(b, 0),
                                           .image_dim = dim,
                                           .image_array = array);
   nir_def *in_bounds = nir_ball(b, nir_ult(b, coord, size));
   nir_def *zero = nir_imm_zero(b, channels, 32);

   nir_push_if(b, in_bounds);
   nir_def *address = nir_bindless_image_texel_address(
      b, 64, pbe, intr->src[1].ssa, intr->src[2].ssa, .image_dim = dim,
      .image_array = array, .format = format, .access = access);
   nir_def *texel = nir_load_global(b, channels, 32, address, .align_mul = 4,
                                    .access = access & (ACCESS_COHERENT |
                                                        ACCESS_VOLATILE));
   nir_pop_if(b, NULL);
   texel = nir_if_phi(b, texel, zero);

   /* Missing channels read as 0, 0, 1. */
   const bool is_float =
      nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) ==
      nir_type_float;
   nir_def *chans[4];
   for (unsigned c = 0; c < 4; ++c) {
      if (c < channels)
         chans[c] = nir_channel(b, texel, c);
      else if (c == 3)
         chans[c] = is_float ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      else
         chans[c] = nir_imm_int(b, 0);
   }

   nir_def_replace(&intr->def,
                   nir_vec(b, chans, intr->def.num_components));
}

static void
lower_image_store_to_memory(nir_builder *b, nir_intrinsic_instr *intr,
                            enum pipe_format format,
                            const struct lower_descriptors_ctx *ctx)
{
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(ctx->descriptor_generation);
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool array = nir_intrinsic_image_array(intr);
   const unsigned coord_comps =
      glsl_get_sampler_dim_coordinate_components(dim) + (array ? 1 : 0);
   const unsigned channels = util_format_get_nr_components(format);
   const enum gl_access_qualifier access = nir_intrinsic_access(intr);

   nir_def *pbe = load_image_handle(b, ctx, intr->src[0], native.pbe);
   nir_def *texture = load_image_handle(b, ctx, intr->src[0], native.texture);
   nir_def *coord = nir_trim_vector(b, intr->src[1].ssa, coord_comps);

   /* Out-of-bounds stores are dropped, as robust image stores are. */
   nir_def *size = nir_bindless_image_size(b, coord_comps, 32, texture,
                                           nir_imm_int(b, 0),
                                           .image_dim = dim,
                                           .image_array = array);
   nir_push_if(b, nir_ball(b, nir_ult(b, coord, size)));
   nir_def *address = nir_bindless_image_texel_address(
      b, 64, pbe, intr->src[1].ssa, intr->src[2].ssa, .image_dim = dim,
      .image_array = array, .format = format, .access = access);
   nir_store_global(b, nir_trim_vector(b, intr->src[3].ssa, channels),
                    address, .align_mul = 4,
                    .write_mask = BITFIELD_MASK(channels),
                    .access = access & (ACCESS_COHERENT | ACCESS_VOLATILE));
   nir_pop_if(b, NULL);

   /* Later texture-path reads of the image in this invocation must see the
    * store, as after an image atomic. */
   if (!(access & ACCESS_NON_READABLE))
      nir_fence_mem_to_tex_agx(b);

   nir_instr_remove(&intr->instr);
}

/* Interleave the low 16 bits of v with zeros: bit i moves to bit 2i. */
static nir_def *
spread_bits_16(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 8)), 0x00ff00ff);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 4)), 0x0f0f0f0f);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 2)), 0x33333333);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 1)), 0x55555555);
   return v;
}

/*
 * The M5 and M5 Pro (G17G, G17S) store to a layer of an array as a 3D store,
 * which addresses at most HK_M5_STORE_MAX_SLICES slices. When a storage
 * view's level needs more, its PBE is a linear window over that level of
 * every layer of the view, and the descriptor's store window record says so
 * (hk_native_descriptor.h). Layered stores check the record's layer count
 * and atomics its layer stride: at zero (every other descriptor, and the null
 * descriptor) they take the native path; otherwise they store through the
 * window at the texel's element index, or update the texel in memory. The
 * index is the tiled address arithmetic of image atomics
 * (libagx_twiddle_coordinates_gen9) plus the layer times the layer stride.
 * Texels outside the view are dropped, as the native store drops them.
 *
 * A one-layer view of a compressed image has a 2D PBE instead, which its
 * record marks (HK_M5_STORE_2D): layered stores store to it at (x, y), with
 * the same bounds check, so layers other than 0 are dropped. Its record's
 * word 0 is zero, so atomics keep the native op.
 *
 * Block stores (image_store_block_agx) are not windowed: on M5 nothing emits
 * them for an application storage view (meta copies take the tiled or
 * graphics paths), and a window PBE would take them as linear.
 */
static bool
lower_m5_store_window(nir_builder *b, nir_intrinsic_instr *intr,
                      const struct lower_descriptors_ctx *ctx)
{
   if (!ctx->m5_store_window)
      return false;

   const bool atomic = intr->intrinsic == nir_intrinsic_image_deref_atomic ||
                       intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
   if (intr->intrinsic != nir_intrinsic_image_deref_store && !atomic)
      return false;

   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool is_1d = dim == GLSL_SAMPLER_DIM_1D;
   if (dim != GLSL_SAMPLER_DIM_CUBE &&
       !(nir_intrinsic_image_array(intr) &&
         (is_1d || dim == GLSL_SAMPLER_DIM_2D)))
      return false;

   uint32_t set, binding;
   nir_def *unused_index;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   get_resource_deref_binding(b, deref, &set, &binding, &unused_index);
   if (get_binding_layout(set, binding, ctx)->stride <
       HK_M5_STORE_WINDOW_OFFSET + HK_M5_STORE_WINDOW_SIZE)
      return false;

   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(ctx->descriptor_generation);
   nir_def *pbe = load_image_handle(b, ctx, intr->src[0], native.pbe);
   nir_def *window = load_resource_deref_desc(
      b, 4, 32, deref, HK_M5_STORE_WINDOW_OFFSET, ctx);

   nir_def *coord = intr->src[1].ssa;
   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = is_1d ? nir_imm_int(b, 0) : nir_channel(b, coord, 1);
   nir_def *z = nir_channel(b, coord, is_1d ? 1 : 2);

   nir_def *layer_stride_el = nir_channel(b, window, 0);
   nir_def *layers = nir_channel(b, window, 3);
   nir_def *window_result = NULL;
   nir_push_if(b, nir_ine_imm(b, atomic ? layer_stride_el : layers, 0));
   {
      nir_def *tiles = nir_channel(b, window, 1);
      nir_def *size = nir_channel(b, window, 2);
      nir_def *width = nir_iand_imm(b, size, 0xffff);
      nir_def *height = nir_ushr_imm(b, size, 16);
      nir_def *in_bounds =
         nir_iand(b, nir_iand(b, nir_ult(b, x, width), nir_ult(b, y, height)),
                  nir_ult(b, z, layers));
      nir_def *zero = atomic ? nir_imm_zero(b, 1, intr->def.bit_size) : NULL;

      nir_push_if(b, in_bounds);
      nir_def *tile_w = nir_ishl(b, nir_imm_int(b, 1),
                                 nir_iand_imm(b, tiles, 0xf));
      nir_def *tile_h = nir_ishl(b, nir_imm_int(b, 1),
                                 nir_ubitfield_extract_imm(b, tiles, 4, 4));
      nir_def *mask_w = nir_iadd_imm(b, tile_w, -1);
      nir_def *mask_h = nir_iadd_imm(b, tile_h, -1);
      nir_def *aligned_width =
         nir_iand(b, nir_iadd(b, width, mask_w), nir_inot(b, mask_w));
      nir_def *within =
         nir_ior(b, spread_bits_16(b, nir_iand(b, x, mask_w)),
                 nir_ishl_imm(b, spread_bits_16(b, nir_iand(b, y, mask_h)), 1));
      nir_def *index = nir_iadd(
         b, nir_imul(b, nir_iand(b, y, nir_inot(b, mask_h)), aligned_width),
         nir_iadd(b, nir_imul(b, nir_iand(b, x, nir_inot(b, mask_w)), tile_h),
                  within));
      index = nir_iadd(b, index, nir_imul(b, z, layer_stride_el));

      nir_def *in_window = NULL;
      if (atomic) {
         /* The linear PBE keeps its address >> 4 in the low 40 bits of its
          * second 64-bit word (agx_pbe_agx3_2d_pack()). */
         nir_def *pbe_word1 = load_resource_deref_desc(
            b, 1, 64, deref, native.pbe + 8, ctx);
         nir_def *base = nir_ishl_imm(
            b, nir_iand_imm(b, pbe_word1, UINT64_C(0xffffffffff)), 4);
         const unsigned bit_size = intr->def.bit_size;
         nir_def *address = nir_iadd(
            b, base, nir_imul_imm(b, nir_u2u64(b, index), bit_size / 8));
         const nir_atomic_op op = nir_intrinsic_atomic_op(intr);
         const enum gl_access_qualifier access = nir_intrinsic_access(intr);
         if (intr->intrinsic == nir_intrinsic_image_deref_atomic_swap) {
            in_window = nir_global_atomic_swap(
               b, bit_size, address, intr->src[3].ssa, intr->src[4].ssa,
               .atomic_op = op, .access = access);
         } else {
            in_window = nir_global_atomic(b, bit_size, address,
                                          intr->src[3].ssa, .atomic_op = op,
                                          .access = access);
         }
         b->shader->info.use_lowered_image_to_global = true;
         /* Later texture-path reads of the image in this invocation must see
          * the update, as after a native image atomic (fence_image()). */
         if (!(access & ACCESS_NON_READABLE))
            nir_fence_mem_to_tex_agx(b);
      } else {
         nir_def *store_2d = nir_test_mask(b, tiles, HK_M5_STORE_2D);
         nir_def *window_coord = nir_vec4(
            b,
            nir_bcsel(b, store_2d, x,
                      nir_iand_imm(b, index, HK_M5_STORE_WINDOW_WIDTH - 1)),
            nir_bcsel(b, store_2d, y,
                      nir_ushr_imm(b, index,
                                   util_logbase2(HK_M5_STORE_WINDOW_WIDTH))),
            nir_imm_int(b, 0), nir_imm_int(b, 0));
         nir_intrinsic_instr *store =
            nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
         nir_builder_instr_insert(b, &store->instr);
         nir_rewrite_image_intrinsic(store, pbe,
                                     nir_image_intrinsic_type_bindless);
         nir_src_rewrite(&store->src[1], window_coord);
         nir_intrinsic_set_image_dim(store, GLSL_SAMPLER_DIM_2D);
         nir_intrinsic_set_image_array(store, false);
      }
      nir_pop_if(b, NULL);
      if (atomic)
         window_result = nir_if_phi(b, in_window, zero);
   }
   nir_push_else(b, NULL);
   nir_intrinsic_instr *native_op =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
   nir_builder_instr_insert(b, &native_op->instr);
   nir_rewrite_image_intrinsic(native_op, pbe,
                               nir_image_intrinsic_type_bindless);
   nir_pop_if(b, NULL);

   if (atomic) {
      nir_def_replace(&intr->def,
                      nir_if_phi(b, window_result, &native_op->def));
   } else {
      nir_instr_remove(&intr->instr);
   }
   return true;
}

static bool
lower_image_intrin(nir_builder *b, nir_intrinsic_instr *intr,
                   const struct lower_descriptors_ctx *ctx)
{
   b->cursor = nir_before_instr(&intr->instr);

   if (lower_m5_store_window(b, intr, ctx))
      return true;

   enum pipe_format memory_format;
   if (image_access_needs_memory_path(intr, ctx, &memory_format)) {
      if (intr->intrinsic == nir_intrinsic_image_deref_load)
         lower_image_load_from_memory(b, intr, memory_format, ctx);
      else
         lower_image_store_to_memory(b, intr, memory_format, ctx);
      return true;
   }

   /* Reads and queries use the texture descriptor; writes and atomics PBE. */
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(ctx->descriptor_generation);
   assert(native.stride);
   unsigned offs;
   if (intr->intrinsic != nir_intrinsic_image_deref_load &&
       intr->intrinsic != nir_intrinsic_image_deref_sparse_load &&
       intr->intrinsic != nir_intrinsic_image_deref_size &&
       intr->intrinsic != nir_intrinsic_image_deref_samples) {

      offs = native.pbe;
   } else {
      offs = native.texture;
   }

   nir_rewrite_image_intrinsic(intr,
                               load_image_handle(b, ctx, intr->src[0], offs),
                               nir_image_intrinsic_type_bindless);

   return true;
}

static VkQueryPipelineStatisticFlagBits
translate_pipeline_stat_bit(enum pipe_statistics_query_index pipe)
{
   switch (pipe) {
   case PIPE_STAT_QUERY_IA_VERTICES:
      return VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT;
   case PIPE_STAT_QUERY_IA_PRIMITIVES:
      return VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT;
   case PIPE_STAT_QUERY_VS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT;
   case PIPE_STAT_QUERY_GS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT;
   case PIPE_STAT_QUERY_GS_PRIMITIVES:
      return VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT;
   case PIPE_STAT_QUERY_C_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
   case PIPE_STAT_QUERY_C_PRIMITIVES:
      return VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
   case PIPE_STAT_QUERY_PS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
   case PIPE_STAT_QUERY_HS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT;
   case PIPE_STAT_QUERY_DS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT;
   case PIPE_STAT_QUERY_CS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
   case PIPE_STAT_QUERY_TS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT;
   case PIPE_STAT_QUERY_MS_INVOCATIONS:
      return VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
   default:
      UNREACHABLE("invalid statistic");
   }
}

struct lower_uvs_args {
   mesa_shader_stage sw_stage;
   unsigned nr_vbos;
   unsigned uniform_base;
};

static bool
lower_uvs_index(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const struct lower_uvs_args *args = data;

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_uvs_index_agx: {
      gl_varying_slot slot = nir_intrinsic_io_semantics(intrin).location;
      unsigned offset = hk_root_descriptor_offset(draw.uvs_index[slot]);
      b->cursor = nir_instr_remove(&intrin->instr);

      nir_def *val = load_root(b, 1, 8, nir_imm_int(b, offset), 1);
      nir_def_rewrite_uses(&intrin->def, nir_u2u16(b, val));
      return true;
   }

   case nir_intrinsic_load_shader_part_tests_zs_agx:
      return lower_sysval_to_root_table(b, intrin, draw.no_epilog_discard);

   case nir_intrinsic_load_clip_z_coeff:
      return lower_sysval_to_root_table(b, intrin, clip_z_coeff);

   case nir_intrinsic_load_api_sample_mask_agx:
      return lower_sysval_to_root_table(b, intrin, draw.api_sample_mask);

   case nir_intrinsic_load_alpha_to_coverage_agx:
      return lower_sysval_to_root_table(b, intrin, draw.alpha_to_coverage);

   case nir_intrinsic_load_sample_positions_agx:
      return lower_sysval_to_root_table(b, intrin, draw.ppp_multisamplectl);

   case nir_intrinsic_load_sample_positions_hi_agx:
      return lower_sysval_to_root_table(b, intrin, ppp_multisamplectl_hi);

   case nir_intrinsic_load_depth_never_agx:
      return lower_sysval_to_root_table(b, intrin, draw.force_never_in_shader);

   case nir_intrinsic_load_geometry_param_buffer_poly:
      return lower_sysval_to_root_table(b, intrin, draw.geometry_params);

   case nir_intrinsic_load_vs_outputs_poly:
      return lower_sysval_to_root_table(b, intrin, draw.vertex_outputs);

   case nir_intrinsic_load_tess_param_buffer_poly:
      return lower_sysval_to_root_table(b, intrin, draw.tess_params);

   case nir_intrinsic_load_vertex_param_buffer_poly:
      if (args->sw_stage == MESA_SHADER_VERTEX) {
         b->cursor = nir_instr_remove(&intrin->instr);

         unsigned base = args->uniform_base +
                         AGX_ABI_VUNI_VERTEX_PARAMS(args->nr_vbos);
         nir_def *val = nir_load_preamble(b, 1, 64, .base = base);
         nir_def_rewrite_uses(&intrin->def, val);

         return true;
      } else {
         return lower_sysval_to_root_table(b, intrin, draw.vertex_params);
      }

   case nir_intrinsic_load_rasterization_stream:
      return lower_sysval_to_root_table(b, intrin, draw.rasterization_stream);

   case nir_intrinsic_load_vbo_stride_agx:
      return lower_sysval_to_root_table(
         b, intrin, draw.attrib_strides[nir_src_as_uint(intrin->src[0])]);

   case nir_intrinsic_load_is_first_fan_agx: {
      unsigned offset = hk_root_descriptor_offset(draw.provoking);
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *val = load_root(b, 1, 16, nir_imm_int(b, offset), 2);
      nir_def_rewrite_uses(&intrin->def, nir_ieq_imm(b, val, 1));
      return true;
   }

   case nir_intrinsic_load_provoking_last: {
      unsigned offset = hk_root_descriptor_offset(draw.provoking);
      b->cursor = nir_instr_remove(&intrin->instr);
      nir_def *val = load_root(b, 1, 16, nir_imm_int(b, offset), 2);
      nir_def_rewrite_uses(&intrin->def, nir_b2b32(b, nir_ieq_imm(b, val, 2)));
      return true;
   }

   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_draw_id: {
      b->cursor = nir_instr_remove(&intrin->instr);

      /* Mesh programs have no vertex uniform ABI; the draw root carries
       * the draw index of each vkCmdDrawMeshTasks* draw. */
      if ((args->sw_stage == MESA_SHADER_MESH ||
           args->sw_stage == MESA_SHADER_TASK) &&
          intrin->intrinsic == nir_intrinsic_load_draw_id) {
         unsigned offset = hk_root_descriptor_offset(mesh_draw_id);
         nir_def *val = load_root(b, 1, 32, nir_imm_int(b, offset), 4);
         nir_def_rewrite_uses(&intrin->def, nir_u2uN(b, val, intrin->def.bit_size));
         return true;
      }

      unsigned base = args->uniform_base +
                      AGX_ABI_VUNI_FIRST_VERTEX(args->nr_vbos);
      unsigned size = 32;

      if (intrin->intrinsic == nir_intrinsic_load_base_instance) {
         base = args->uniform_base +
                AGX_ABI_VUNI_BASE_INSTANCE(args->nr_vbos);
      } else if (intrin->intrinsic == nir_intrinsic_load_draw_id) {
         base = args->uniform_base + AGX_ABI_VUNI_DRAW_ID(args->nr_vbos);
         size = 16;
      }

      nir_def *val = nir_load_preamble(b, 1, size, .base = base);
      nir_def_rewrite_uses(&intrin->def,
                           nir_u2uN(b, val, intrin->def.bit_size));
      return true;
   }

   case nir_intrinsic_load_stat_query_address_poly: {
      b->cursor = nir_instr_remove(&intrin->instr);

      unsigned off1 = hk_root_descriptor_offset(draw.pipeline_stats);
      unsigned off2 = hk_root_descriptor_offset(draw.pipeline_stats_flags);
      static_assert(hk_root_descriptor_offset(draw.pipeline_stats) ==
                       304 + 8 * HK_MAX_SETS,
                    "pipeline-statistics compiler oracle ABI");
      static_assert(hk_root_descriptor_offset(draw.pipeline_stats_flags) ==
                       312 + 8 * HK_MAX_SETS,
                    "pipeline-statistics compiler oracle ABI");

      nir_def *base = load_root(b, 1, 64, nir_imm_int(b, off1), 8);
      /* Keep the VkFlags field at its native 32-bit width. Besides preserving
       * every public statistic bit, G17P has an exact live-source U32 popcount
       * form for the compact report index. A U16 load instead leaves this
       * value live across both the popcount and presence test and requires an
       * unwitnessed cached-half popcount. */
      nir_def *flags = load_root(b, 1, 32, nir_imm_int(b, off2), 4);

      unsigned query = nir_intrinsic_base(intrin);
      VkQueryPipelineStatisticFlagBits bit = translate_pipeline_stat_bit(query);

      /* Prefix sum to find the compacted offset */
      nir_def *idx = nir_bit_count(b, nir_iand_imm(b, flags, bit - 1));
      nir_def *addr = nir_iadd(
         b, base, nir_amul_imm(b, nir_u2u64(b, idx), sizeof(uint64_t)));

      /* The above returns garbage if the query isn't actually enabled, handle
       * that case.
       *
       * TODO: Optimize case where we *know* the query is present?
       */
      nir_def *present = nir_ine_imm(b, nir_iand_imm(b, flags, bit), 0);

      /* Sometimes we insert a GS internally, it should not contribute to GS
       * statistics. This is not strictly needed for Vulkan but vkd3d-proton
       * tests it and we should avoid the surprising behaviour.
       */
      if (query == PIPE_STAT_QUERY_GS_INVOCATIONS ||
          query == PIPE_STAT_QUERY_GS_PRIMITIVES) {

         unsigned api_gs_offset = hk_root_descriptor_offset(draw.api_gs);
         nir_def *api_gs =
            load_root(b, 1, 16, nir_imm_int(b, api_gs_offset), 4);

         present = nir_iand(b, present, nir_ine_imm(b, api_gs, 0));
      }

      addr = nir_bcsel(b, present, addr,
                       nir_imm_int64(b, AGX_SCRATCH_PAGE_ADDRESS));

      nir_def_rewrite_uses(&intrin->def, addr);
      return true;
   }

   default:
      return false;
   }
}

bool
hk_lower_uvs_index(nir_shader *s, mesa_shader_stage sw_stage, unsigned nr_vbos,
                   unsigned uniform_base)
{
   struct lower_uvs_args args = {
      .sw_stage = sw_stage,
      .nr_vbos = nr_vbos,
      .uniform_base = uniform_base,
   };
   return nir_shader_intrinsics_pass(s, lower_uvs_index,
                                     nir_metadata_control_flow, &args);
}

static bool
try_lower_intrin(nir_builder *b, nir_intrinsic_instr *intrin,
                 const struct lower_descriptors_ctx *ctx)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_constant:
      return lower_load_constant(b, intrin, ctx);

   case nir_intrinsic_load_vulkan_descriptor:
      return try_lower_load_vulkan_descriptor(b, intrin, ctx);

   case nir_intrinsic_load_workgroup_size:
      UNREACHABLE("Should have been lowered by nir_lower_cs_intrinsics()");

   case nir_intrinsic_load_base_workgroup_id:
      return lower_sysval_to_root_table(b, intrin, cs.base_group);

   case nir_intrinsic_load_push_constant:
      return lower_load_push_constant(b, intrin, ctx);

   case nir_intrinsic_load_view_index:
      return lower_sysval_to_root_table(b, intrin, draw.view_index);

   case nir_intrinsic_load_printf_buffer_address:
      return lower_sysval_to_root_table(b, intrin, printf_buffer_address);

   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_store_block_agx:
      return lower_image_intrin(b, intrin, ctx);

   case nir_intrinsic_load_image_descriptor_word_agx: {
      b->cursor = nir_instr_remove(&intrin->instr);
      const struct hk_storage_image_descriptor_layout native =
         hk_storage_image_descriptor_layout(ctx->descriptor_generation);
      nir_def *word = load_resource_deref_desc(
         b, 1, 32, nir_src_as_deref(intrin->src[0]), native.texture, ctx);
      nir_def_rewrite_uses(&intrin->def, word);
      return true;
   }

   case nir_intrinsic_load_texture_descriptor_word_agx: {
      /* First word of the (plane 0) texture descriptor of a sampled image. */
      b->cursor = nir_instr_remove(&intrin->instr);
      const struct hk_sampled_image_descriptor_layout native =
         hk_sampled_image_descriptor_layout(ctx->descriptor_generation);
      nir_def *word = load_resource_deref_desc(
         b, 1, 32, nir_src_as_deref(intrin->src[0]), native.texture, ctx);
      nir_def_rewrite_uses(&intrin->def, word);
      return true;
   }

   case nir_intrinsic_load_num_workgroups: {
      b->cursor = nir_instr_remove(&intrin->instr);

      /* Mesh and task shaders run under the graphics root, whose cs arm
       * overlays the draw arm: read the draw's own grid instead. With a task
       * stage this is the API task grid, which only the task stage reads. */
      if (b->shader->info.stage == MESA_SHADER_MESH ||
          b->shader->info.stage == MESA_SHADER_TASK) {
         unsigned offset = hk_root_descriptor_offset(mesh_grid_addr);
         nir_def *ptr = load_root(b, 1, 64, nir_imm_int(b, offset), 8);
         nir_def_rewrite_uses(&intrin->def, load_speculatable(b, 3, 32, ptr, 4));
         return true;
      }

      unsigned offset = hk_root_descriptor_offset(cs.group_count_addr);
      nir_def *ptr = load_root(b, 1, 64, nir_imm_int(b, offset), 4);
      nir_def *val = load_speculatable(b, 3, 32, ptr, 4);

      nir_def_rewrite_uses(&intrin->def, val);
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex,
          const struct lower_descriptors_ctx *ctx);

static nir_def *
compute_lod_lambda(nir_builder *b, nir_tex_instr *tex, nir_def *size,
                   bool fragment)
{
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   /* nir_texop_lod coordinates never carry an array layer component. */
   const unsigned comps = tex->coord_components;
   nir_def *c = nir_f2f32(b, nir_trim_vector(b, tex->src[coord_idx].src.ssa,
                                             comps));
   nir_def *dx = fragment ? nir_ddx(b, c) : agx_nir_quad_ddx(b, c);
   nir_def *dy = fragment ? nir_ddy(b, c) : agx_nir_quad_ddy(b, c);

   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      nir_def *abs_p = nir_fabs(b, c);
      nir_def *abs_x = nir_channel(b, abs_p, 0);
      nir_def *abs_y = nir_channel(b, abs_p, 1);
      nir_def *abs_z = nir_channel(b, abs_p, 2);
      nir_def *cond_z = nir_fge(b, abs_z, nir_fmax(b, abs_x, abs_y));
      nir_def *cond_y = nir_fge(b, abs_y, nir_fmax(b, abs_x, abs_z));
      const unsigned yzx[3] = {1, 2, 0};
      const unsigned xzy[3] = {0, 2, 1};
      nir_def *q = nir_bcsel(b, cond_z, c,
                             nir_bcsel(b, cond_y, nir_swizzle(b, c, xzy, 3),
                                       nir_swizzle(b, c, yzx, 3)));
      nir_def *dqdx = nir_bcsel(b, cond_z, dx,
                                nir_bcsel(b, cond_y, nir_swizzle(b, dx, xzy, 3),
                                          nir_swizzle(b, dx, yzx, 3)));
      nir_def *dqdy = nir_bcsel(b, cond_z, dy,
                                nir_bcsel(b, cond_y, nir_swizzle(b, dy, xzy, 3),
                                          nir_swizzle(b, dy, yzx, 3)));
      nir_def *recip = nir_frcp(b, nir_channel(b, q, 2));
      nir_def *qxy = nir_trim_vector(b, q, 2);
      nir_def *fdx = nir_fmul(
         b, recip,
         nir_fsub(b, nir_trim_vector(b, dqdx, 2),
                  nir_fmul(b, qxy, nir_fmul(b, nir_channel(b, dqdx, 2), recip))));
      nir_def *fdy = nir_fmul(
         b, recip,
         nir_fsub(b, nir_trim_vector(b, dqdy, 2),
                  nir_fmul(b, qxy, nir_fmul(b, nir_channel(b, dqdy, 2), recip))));
      nir_def *m = nir_fmax(b, nir_fdot(b, fdx, fdx), nir_fdot(b, fdy, fdy));
      nir_def *l = nir_channel(b, size, 0);
      return nir_fadd_imm(
         b, nir_fmul_imm(b, nir_flog2(b, nir_fmul(b, nir_fmul(b, l, l), m)), 0.5),
         -1.0);
   }

   nir_def *s = nir_trim_vector(b, size, comps);
   nir_def *sx = nir_fmul(b, dx, s);
   nir_def *sy = nir_fmul(b, dy, s);
   nir_def *rho2 = nir_fmax(b, nir_fdot(b, sx, sx), nir_fdot(b, sy, sy));
   return nir_fmul_imm(b, nir_flog2(b, rho2), 0.5);
}

static bool
lower_compute_lod_query(nir_builder *b, nir_tex_instr *tex,
                        const struct lower_descriptors_ctx *ctx)
{
   /* Task and mesh shaders with a derivative group take the compute
    * derivatives (meshAndTaskShaderDerivatives). AGX3D fragment shaders
    * also use this path for array/cube LOD queries, which hardware SMP
    * does not encode. */
   const bool compute_like =
      mesa_shader_stage_uses_workgroup(b->shader->info.stage) &&
      b->shader->info.derivative_group != DERIVATIVE_GROUP_NONE;
   const bool frag_array_cube =
      b->shader->info.stage == MESA_SHADER_FRAGMENT &&
      ctx->lower_array_cube_lod_query &&
      (tex->is_array || tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE);
   if (tex->op != nir_texop_lod ||
       (!hk_descriptor_generation_is_agx2(ctx->descriptor_generation) &&
        ctx->descriptor_generation != 9 && ctx->descriptor_generation != 10) ||
       (!compute_like && !frag_array_cube))
      return false;

   const int sampler_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (sampler_idx < 0 || coord_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   const unsigned size_comps =
      (tex->sampler_dim == GLSL_SAMPLER_DIM_1D ? 1
       : tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2) +
      (tex->is_array ? 1 : 0);
   nir_def *size = nir_build_texture_query(b, tex, nir_texop_txs, size_comps,
                                           nir_type_int32, false, true);
   nir_def *levels = nir_build_texture_query(b, tex, nir_texop_query_levels, 1,
                                             nir_type_int32, false, false);
   lower_tex(b, nir_instr_as_tex(nir_def_instr(size)), ctx);
   lower_tex(b, nir_instr_as_tex(nir_def_instr(levels)), ctx);
   b->cursor = nir_before_instr(&tex->instr);

   nir_deref_instr *sampler = nir_src_as_deref(tex->src[sampler_idx].src);
   const int plane_idx = nir_tex_instr_src_index(tex, nir_tex_src_plane);
   const uint32_t plane =
      plane_idx >= 0 ? nir_src_as_uint(tex->src[plane_idx].src) : 0;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(ctx->descriptor_generation);
   const uint64_t plane_offset_B = plane * native.stride;

   nir_def *word, *api_bias;
   if (hk_descriptor_generation_is_agx2(ctx->descriptor_generation)) {
      nir_def *published = load_resource_deref_desc(
         b, 1, 16, sampler, plane_offset_B + native.sampler_index, ctx);
      nir_def *published32 = nir_u2u32(b, published);
      nir_def *index = nir_bcsel(
         b, nir_uge_imm(b, published32, HK_SAMPLER_AGX2_BIAS),
         nir_iadd_imm(b, published32, -(int)HK_SAMPLER_AGX2_BIAS),
         nir_imm_int(b, 0));
      api_bias = nir_f2f32(
         b, load_resource_deref_desc(b, 1, 16, sampler,
                                     plane_offset_B + native.lod_bias, ctx));
      nir_def *heap = load_root(
         b, 1, 64,
         nir_imm_int(b, hk_root_descriptor_offset(sampler_heap_address)), 8);
      nir_def *addr = nir_iadd(
         b, heap,
         nir_imul_imm(b, nir_u2u64(b, index),
                      agx_chip_sampler_heap_element_size(AGX_CHIP_G13G)));
      word = load_speculatable(b, 1, 32, addr, 8);
   } else {
      /* Both Gen9 and Gen10 descriptors store the unbiased sampler indices and
       * float API bias in sw.g17p (overlapping sw.sampler), while the hardware
       * sampler state word lives in the 32-byte-per-entry native sampler heap. */
      static_assert(AGX_SAMPLER_GEN5_LENGTH <= HK_NATIVE_SAMPLER_MAX_ELEMENT_SIZE,
                    "Gen5 sampler fits in native heap slot");
      nir_def *index = load_resource_deref_desc(
         b, 1, 16, sampler, plane_offset_B + native.sampler_index, ctx);
      api_bias = load_resource_deref_desc(
         b, 1, 32, sampler,
         plane_offset_B + native.sampler +
            offsetof(struct hk_sampled_image_descriptor_sw, g17p.lod_bias),
         ctx);
      nir_def *heap = load_root(
         b, 1, 64,
         nir_imm_int(b, hk_root_descriptor_offset(sampler_heap_address)), 8);
      nir_def *addr = nir_iadd(
         b, heap,
         nir_imul_imm(b, nir_u2u64(b, index), HK_NATIVE_SAMPLER_MAX_ELEMENT_SIZE));
      word = load_speculatable(b, 1, 32, addr, 8);
   }
   nir_def *min_lod =
      nir_fmul_imm(b, nir_u2f32(b, nir_iand_imm(b, word, 0x3ff)), 1.0 / 64.0);
   nir_def *max_lod = nir_fmul_imm(
      b, nir_u2f32(b, nir_iand_imm(b, nir_ushr_imm(b, word, 10), 0x3ff)),
      1.0 / 64.0);
   nir_def *mip_filter = nir_iand_imm(b, nir_ushr_imm(b, word, 27), 3);

   nir_def *lambda = nir_fadd(
      b, compute_lod_lambda(b, tex, nir_i2f32(b, size), frag_array_cube),
      api_bias);
   nir_def *clamped = nir_fclamp(b, lambda, min_lod, max_lod);
   nir_def *nearest =
      nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, clamped, 0.5)), -1.0);
   nir_def *level = nir_bcsel(
      b, nir_ieq_imm(b, mip_filter, AGX_MIP_FILTER_NEAREST), nearest,
      nir_bcsel(b, nir_ieq_imm(b, mip_filter, AGX_MIP_FILTER_LINEAR), clamped,
                nir_imm_float(b, 0.0)));
   nir_def *last = nir_i2f32(b, nir_iadd_imm(b, levels, -1));
   level = nir_fclamp(b, level, nir_imm_float(b, 0.0), last);

   /* The nearest-mip rounding bias picks the level but is not API bias. */
   if (ctx->d3d_nearest_mip) {
      lambda = nir_bcsel(b, nir_ieq_imm(b, mip_filter, AGX_MIP_FILTER_NEAREST),
                         nir_fadd_imm(b, lambda, -HK_D3D_NEAREST_MIP_BIAS),
                         lambda);
   }

   nir_def *result = nir_bcsel(b, nir_ieq_imm(b, levels, 0),
                               nir_imm_zero(b, 2, 32),
                               nir_vec2(b, level, lambda));
   if (tex->def.bit_size != 32)
      result = nir_f2fN(b, result, tex->def.bit_size);
   nir_def_replace(&tex->def, result);
   return true;
}

/*
 * A Gen4 sampler word has no LOD-bias field, so NIR's sampler-bias lowering
 * hands a LOD query the API bias as an instruction operand. G15's native
 * query does not apply that operand correctly: a null texture reports a
 * nonzero LOD. Query without it, add the bias to the query's unclamped LOD
 * and select the accessed level from the sampler's LOD range and mip filter,
 * as lower_compute_lod_query does. A view without levels (a null descriptor)
 * reports zero.
 */
static void
lower_gen4_biased_lod_query(nir_builder *b, nir_tex_instr *tex,
                            const struct lower_descriptors_ctx *ctx)
{
   if (tex->op != nir_texop_lod || ctx->descriptor_generation != 9 ||
       ctx->sampler_has_lod_bias ||
       nir_tex_instr_src_index(tex, nir_tex_src_bias) < 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref) < 0)
      return;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *bias = nir_f2f32(b, nir_steal_tex_src(tex, nir_tex_src_bias));

   nir_def *levels = nir_build_texture_query(b, tex, nir_texop_query_levels, 1,
                                             nir_type_int32, false, false);
   lower_tex(b, nir_instr_as_tex(nir_def_instr(levels)), ctx);
   b->cursor = nir_before_instr(&tex->instr);

   const int sampler_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   nir_deref_instr *sampler = nir_src_as_deref(tex->src[sampler_idx].src);
   const int plane_idx = nir_tex_instr_src_index(tex, nir_tex_src_plane);
   const uint32_t plane =
      plane_idx >= 0 ? nir_src_as_uint(tex->src[plane_idx].src) : 0;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(ctx->descriptor_generation);

   nir_def *index = load_resource_deref_desc(
      b, 1, 16, sampler, plane * native.stride + native.sampler_index, ctx);
   nir_def *heap = load_root(
      b, 1, 64, nir_imm_int(b, hk_root_descriptor_offset(sampler_heap_address)),
      8);
   nir_def *addr = nir_iadd(
      b, heap,
      nir_imul_imm(b, nir_u2u64(b, index),
                   agx_chip_sampler_heap_element_size(AGX_CHIP_G15S)));
   nir_def *word = load_speculatable(b, 1, 32, addr, 8);
   nir_def *min_lod =
      nir_fmul_imm(b, nir_u2f32(b, nir_iand_imm(b, word, 0x3ff)), 1.0 / 64.0);
   nir_def *max_lod = nir_fmul_imm(
      b, nir_u2f32(b, nir_iand_imm(b, nir_ushr_imm(b, word, 10), 0x3ff)),
      1.0 / 64.0);
   nir_def *mip_filter = nir_iand_imm(b, nir_ushr_imm(b, word, 27), 3);

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *lambda =
      nir_fadd(b, nir_f2f32(b, nir_channel(b, &tex->def, 1)), bias);
   nir_def *clamped = nir_fclamp(b, lambda, min_lod, max_lod);
   nir_def *nearest =
      nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, clamped, 0.5)), -1.0);
   nir_def *level = nir_bcsel(
      b, nir_ieq_imm(b, mip_filter, AGX_MIP_FILTER_NEAREST), nearest,
      nir_bcsel(b, nir_ieq_imm(b, mip_filter, AGX_MIP_FILTER_LINEAR), clamped,
                nir_imm_float(b, 0.0)));
   nir_def *last = nir_i2f32(b, nir_iadd_imm(b, levels, -1));
   level = nir_fclamp(b, level, nir_imm_float(b, 0.0), last);

   nir_def *result = nir_bcsel(b, nir_ieq_imm(b, levels, 0),
                               nir_imm_zero(b, 2, 32),
                               nir_vec2(b, level, lambda));
   if (tex->def.bit_size != 32)
      result = nir_f2fN(b, result, tex->def.bit_size);
   nir_def_rewrite_uses_after(&tex->def, result);
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex,
          const struct lower_descriptors_ctx *ctx)
{
   if (lower_compute_lod_query(b, tex, ctx))
      return true;

   lower_gen4_biased_lod_query(b, tex, ctx);

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *texture = nir_steal_tex_src(tex, nir_tex_src_texture_deref);
   nir_def *sampler = nir_steal_tex_src(tex, nir_tex_src_sampler_deref);
   if (!texture) {
      assert(!sampler);
      return false;
   }

   nir_def *plane_ssa = nir_steal_tex_src(tex, nir_tex_src_plane);
   const uint32_t plane =
      plane_ssa ? nir_src_as_uint(nir_src_for_ssa(plane_ssa)) : 0;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(ctx->descriptor_generation);
   assert(native.stride);
   const uint64_t plane_offset_B = plane * native.stride;

   /* LOD bias is passed in the descriptor set, rather than embedded into
    * the sampler descriptor. There's no spot in the hardware descriptor,
    * plus this saves on precious sampler heap spots.
    */
   if (tex->op == nir_texop_lod_bias) {
      /* G17P carries API LOD bias in its native sampler word. Avoid a
       * redundant software load, including for dynamically indexed sets.
       * A Gen4 sampler has no such field, so the shader loads the bias. */
      if ((ctx->descriptor_generation == 9 ||
           ctx->descriptor_generation == 10) &&
          ctx->sampler_has_lod_bias &&
          !(tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS)) {
         nir_def_replace(&tex->def, nir_imm_float16(b, 0.0));
         return true;
      }
      const bool combined =
         tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS;
      unsigned offs = combined
         ? native.sampler + offsetof(struct hk_sampled_image_descriptor_sw,
                                     g17p.lod_bias)
         : native.lod_bias;

      nir_def *bias = load_resource_deref_desc(
         b, 1, combined ? 32 : 16,
         nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_def_replace(&tex->def, bias);
      return true;
   }

   if (tex->op == nir_texop_image_min_lod_agx) {
      assert(tex->dest_type == nir_type_float16 ||
             tex->dest_type == nir_type_uint16);

      unsigned offs = tex->dest_type == nir_type_float16
                         ? native.min_lod
                         : native.min_lod_uint;

      nir_def *min = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(texture)),
         plane_offset_B + offs, ctx);

      nir_def_replace(&tex->def, min);
      return true;
   }

   if (tex->op == nir_texop_has_custom_border_color_agx) {
      unsigned offs = native.clamp_0_sampler_index;

      nir_def *res = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_def_replace(&tex->def, nir_ige_imm(b, res, 0));
      return true;
   }

   if (tex->op == nir_texop_custom_border_color_agx) {
      unsigned offs = native.border;

      nir_def *border = load_resource_deref_desc(
         b, 4, 32, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);

      nir_alu_type T = nir_alu_type_get_base_type(tex->dest_type);
      nir_def *kind = NULL;
      if (ctx->descriptor_generation == 9 && T == nir_type_float) {
         /* An integer border (INT_OPAQUE_*, INT_CUSTOM) sampled as float
          * names the texel value, so it is converted rather than
          * reinterpreted; on a scaled view that is before the scale
          * division below. */
         nir_def *clamp_0 = load_resource_deref_desc(
            b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
            plane_offset_B + native.clamp_0_sampler_index, ctx);
         nir_def *is_int = nir_ine_imm(
            b, nir_iand_imm(b, clamp_0, HK_BORDER_IS_INT_BIT), 0);
         nir_def *as_int = nir_u2f32(b, border);

         /* Only a scaled view has a nonzero scaled_conversion kind. Without
          * sampleable scaled formats it is zero in every valid descriptor. */
         if (ctx->scaled_sampling) {
            /* Look up the image independently: separate samplers have no
             * format. */
            kind = load_resource_deref_desc(
               b, 1, 16, nir_src_as_deref(nir_src_for_ssa(texture)),
               plane_offset_B + native.texture + 32 +
                  offsetof(struct hk_sampled_image_descriptor_sw, scaled_conversion), ctx);
            as_int = nir_bcsel(b, nir_ieq_imm(b, nir_iand_imm(b, kind, 3), 1),
                               nir_i2f32(b, border), as_int);
         }
         border = nir_bcsel(b, is_int, as_int, border);
      }
      if (kind && !(tex->backend_flags & AGX_TEXTURE_FLAG_BORDER_UNFILTERED)) {
         /* Scaled samples rescale the zero/one border probes as well as
          * interior texels. Divide only their interpolation coefficient. */
         nir_def *normalized = nir_ieq_imm(b, nir_iand_imm(b, kind, 3), 3);
         nir_def *reciprocal = nir_bcsel(b,
            nir_ine_imm(b, nir_iand_imm(b, kind, 4), 0),
            nir_imm_float(b, 1.0 / 65535.0), nir_imm_float(b, 1.0 / 255.0));
         reciprocal = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, kind, 128), 0),
                                  nir_imm_float(b, 1.0 / 1023.0), reciprocal);
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c) {
            nir_def *scaled = nir_iand(b, normalized,
               nir_ine_imm(b, nir_iand_imm(b, kind, BITFIELD_BIT(3 + c)), 0));
            channels[c] = nir_fmul(b, nir_channel(b, border, c),
               nir_bcsel(b, scaled,
                  nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, kind, BITFIELD_BIT(8 + c)), 0),
                                nir_imm_float(b, 1.0 / 3.0), reciprocal),
                  nir_imm_float(b, 1)));
         }
         border = nir_vec(b, channels, 4);
      }
      /* See hk_border_swizzle_word(). A combined image sampler applied the
       * mapping when it was written. */
      if (ctx->descriptor_generation == 9 && texture != sampler) {
         nir_def *word = load_resource_deref_desc(
            b, 1, 32, nir_src_as_deref(nir_src_for_ssa(texture)),
            plane_offset_B + native.texture + 32 +
               offsetof(struct hk_sampled_image_descriptor_sw, border), ctx);
         nir_def *known = nir_ine_imm(b, nir_iand_imm(b, word, BITFIELD_BIT(31)), 0);
         nir_def *one = T == nir_type_float ? nir_imm_float(b, 1.0f)
                                            : nir_imm_int(b, 1);
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c) {
            nir_def *code = nir_iand_imm(b, nir_ushr_imm(b, word, 3 * c), 7);
            nir_def *value = nir_bcsel(b, nir_ieq_imm(b, code, 5), one,
                                       nir_imm_int(b, 0));
            for (unsigned i = 0; i < 4; ++i) {
               value = nir_bcsel(b, nir_ieq_imm(b, code, i),
                                 nir_channel(b, border, i), value);
            }
            channels[c] = nir_bcsel(b, known, value, nir_channel(b, border, c));
         }
         border = nir_vec(b, channels, 4);
      }

      border = nir_convert_to_bit_size(b, border, T, tex->def.bit_size);

      nir_def_replace(&tex->def, border);
      return true;
   }

   {
      unsigned offset_B = plane_offset_B + native.texture;

      nir_def *handle =
         load_image_handle(b, ctx, nir_src_for_ssa(texture), offset_B);

      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, handle);
   }

   if (sampler != NULL) {
      unsigned offs = native.sampler_index;
      bool clamp_to_0 = tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0;

      if (clamp_to_0) {
         offs = native.clamp_0_sampler_index;
      }

      if ((ctx->descriptor_generation == 9 ||
           ctx->descriptor_generation == 10) &&
          (tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS)) {
         offs = native.sampler +
                (clamp_to_0
                    ? offsetof(struct hk_sampled_image_descriptor_sw,
                               g17p.unbiased_clamp_0_sampler_index)
                    : offsetof(struct hk_sampled_image_descriptor_sw,
                               g17p.unbiased_sampler_index));
      }

      nir_def *index = load_resource_deref_desc(
         b, 1, 16, nir_src_as_deref(nir_src_for_ssa(sampler)),
         plane_offset_B + offs, ctx);
      if (clamp_to_0)
         index = nir_iand_imm(b, index, HK_BORDER_IS_INT_BIT - 1);

      if (!clamp_to_0) {
         uint32_t set, binding;
         nir_def *idx;
         get_resource_deref_binding(b,
                                    nir_src_as_deref(nir_src_for_ssa(sampler)),
                                    &set, &binding, &idx);

         const struct hk_descriptor_set_binding_layout *binding_layout =
            get_binding_layout(set, binding, ctx);

         if (ctx->clamp_desc_array_bounds)
            idx =
               nir_umin(b, idx, nir_imm_int(b, binding_layout->array_size - 1));

         assert(binding_layout->stride > 0);
         nir_def *desc_offs_B = nir_iadd_imm(
            b, nir_amul_imm(b, idx, binding_layout->stride),
            binding_layout->offset + plane_offset_B + native.sampler);

         index =
            nir_bindless_sampler_agx(b, desc_offs_B, index, .desc_set = set);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, index);
   }

   /* The HK descriptor selection is complete; do not leak a private flag
    * into shared native texture-shape validation. */
   tex->backend_flags &= ~HK_TEXTURE_FLAG_COMBINED_BIAS;
   return true;
}

static bool
try_lower_descriptors_instr(nir_builder *b, nir_instr *instr, void *_data)
{
   const struct lower_descriptors_ctx *ctx = _data;

   switch (instr->type) {
   case nir_instr_type_tex:
      return lower_tex(b, nir_instr_as_tex(instr), ctx);
   case nir_instr_type_intrinsic:
      return try_lower_intrin(b, nir_instr_as_intrinsic(instr), ctx);
   default:
      return false;
   }
}

static bool
lower_ssbo_resource_index(nir_builder *b, nir_intrinsic_instr *intrin,
                          const struct lower_descriptors_ctx *ctx)
{
   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   uint32_t set = nir_intrinsic_desc_set(intrin);
   uint32_t binding = nir_intrinsic_binding(intrin);
   nir_def *index = intrin->src[0].ssa;

   const struct hk_descriptor_set_binding_layout *binding_layout =
      get_binding_layout(set, binding, ctx);

   nir_def *binding_addr;
   uint32_t binding_stride;
   switch (binding_layout->type) {
   case VK_DESCRIPTOR_TYPE_MUTABLE_EXT:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      nir_def *set_addr = nir_load_descriptor_set_agx(b, set);
      binding_addr = nir_iadd_imm(b, set_addr, binding_layout->offset);
      binding_stride = binding_layout->stride;
      break;
   }

   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
      const uint32_t root_desc_addr_offset =
         hk_root_descriptor_offset(root_desc_addr);

      nir_def *root_desc_addr =
         load_root(b, 1, 64, nir_imm_int(b, root_desc_addr_offset), 8);

      nir_def *dynamic_buffer_start =
         nir_iadd_imm(b, load_dynamic_buffer_start(b, set, ctx),
                      binding_layout->dynamic_buffer_index);

      nir_def *dynamic_binding_offset =
         nir_iadd_imm(b,
                      nir_amul_imm(b, dynamic_buffer_start,
                                   sizeof(struct hk_buffer_address)),
                      hk_root_descriptor_offset(dynamic_buffers));

      binding_addr =
         nir_iadd(b, root_desc_addr, nir_u2u64(b, dynamic_binding_offset));
      binding_stride = sizeof(struct hk_buffer_address);
      break;
   }

   default:
      UNREACHABLE("Not an SSBO descriptor");
   }

   /* Tuck the stride in the top 8 bits of the binding address */
   assert(binding_stride <= UINT8_MAX);
   binding_addr = nir_ior_imm(b, binding_addr, (uint64_t)binding_stride << 56);

   const uint32_t binding_size = binding_layout->array_size * binding_stride;
   nir_def *offset_in_binding = nir_amul_imm(b, index, binding_stride);

   nir_def *addr = nir_vec4(b, nir_unpack_64_2x32_split_x(b, binding_addr),
                            nir_unpack_64_2x32_split_y(b, binding_addr),
                            nir_imm_int(b, binding_size), offset_in_binding);

   if (storage_address_has_map(ctx)) {
      nir_def *parts[] = {nir_channel(b, addr, 0), nir_channel(b, addr, 1),
         nir_channel(b, addr, 2), nir_channel(b, addr, 3), nir_imm_int(b, 0)};
      addr = nir_vec(b, parts, 5);
   }
   nir_def_rewrite_uses(&intrin->def, addr);

   return true;
}

static bool
lower_ssbo_resource_reindex(nir_builder *b, nir_intrinsic_instr *intrin,
                            const struct lower_descriptors_ctx *ctx)
{
   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *addr = intrin->src[0].ssa;
   nir_def *index = intrin->src[1].ssa;

   nir_def *addr_high32 = nir_channel(b, addr, 1);
   nir_def *stride = nir_ushr_imm(b, addr_high32, 24);
   nir_def *offset = nir_imul(b, index, stride);

   addr = nir_build_addr_iadd(b, addr, ctx->ssbo_addr_format, nir_var_mem_ssbo,
                              offset);
   nir_def_rewrite_uses(&intrin->def, addr);

   return true;
}

static bool
lower_load_ssbo_descriptor(nir_builder *b, nir_intrinsic_instr *intrin,
                           const struct lower_descriptors_ctx *ctx)
{
   const nir_descriptor_type desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type != nir_descriptor_type_storage_buffer)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *addr = intrin->src[0].ssa;

   nir_def *desc;
   switch (ctx->ssbo_addr_format) {
   case nir_address_format_64bit_global_32bit_offset:
   case nir_address_format_64bit_global_32bit_offset_map: {
      nir_def *base = nir_pack_64_2x32(b, nir_trim_vector(b, addr, 2));
      nir_def *offset = nir_channel(b, addr, 3);
      /* Mask off the binding stride */
      base = nir_iand_imm(b, base, BITFIELD64_MASK(56));
      desc = nir_load_global_constant_offset(
         b, 4, 32, base, offset, .align_mul = 16, .align_offset = 0,
         .access = ACCESS_CAN_SPECULATE | ACCESS_BUFFER_BINDING_AGX);
      break;
   }

   case nir_address_format_64bit_bounded_global:
   case nir_address_format_64bit_bounded_global_map: {
      nir_def *base = nir_pack_64_2x32(b, nir_trim_vector(b, addr, 2));
      nir_def *size = nir_channel(b, addr, 2);
      nir_def *offset = nir_channel(b, addr, 3);
      /* Mask off the binding stride */
      base = nir_iand_imm(b, base, BITFIELD64_MASK(56));
      desc = nir_load_global_constant_bounded(
         b, 4, 32, base, offset, size, .align_mul = 16, .align_offset = 0,
         .access = ACCESS_CAN_SPECULATE | ACCESS_BUFFER_BINDING_AGX);
      break;
   }

   default:
      UNREACHABLE("Unknown address mode");
   }

   if (storage_address_has_map(ctx))
      desc = storage_address(b, desc, ctx);
   nir_def_rewrite_uses(&intrin->def, desc);

   return true;
}

static bool
lower_ssbo_descriptor(nir_builder *b, nir_intrinsic_instr *intr, void *_data)
{
   const struct lower_descriptors_ctx *ctx = _data;

   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      return lower_ssbo_resource_index(b, intr, ctx);
   case nir_intrinsic_vulkan_resource_reindex:
      return lower_ssbo_resource_reindex(b, intr, ctx);
   case nir_intrinsic_load_vulkan_descriptor:
      return lower_load_ssbo_descriptor(b, intr, ctx);
   default:
      return false;
   }
}

static bool
is_buffer_deref_access(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_deref:
   case nir_intrinsic_load_deref_block_intel:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_store_deref_block_intel:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
   case nir_intrinsic_deref_buffer_array_length:
      return true;
   default:
      return false;
   }
}

static bool
is_buffer_deref_atomic(nir_intrinsic_op op)
{
   return op == nir_intrinsic_deref_atomic ||
          op == nir_intrinsic_deref_atomic_swap;
}

/* The (set, binding) of a storage-buffer deref before descriptor lowering,
 * or -1 if it cannot be identified. */
static int64_t
ssbo_deref_binding(nir_deref_instr *deref)
{
   while (deref->deref_type != nir_deref_type_cast) {
      deref = nir_deref_instr_parent(deref);
      if (!deref)
         return -1;
   }

   nir_intrinsic_instr *desc = nir_src_as_intrinsic(deref->parent);
   if (!desc || desc->intrinsic != nir_intrinsic_load_vulkan_descriptor)
      return -1;

   nir_intrinsic_instr *index = nir_src_as_intrinsic(desc->src[0]);
   while (index && index->intrinsic == nir_intrinsic_vulkan_resource_reindex)
      index = nir_src_as_intrinsic(index->src[0]);
   if (!index || index->intrinsic != nir_intrinsic_vulkan_resource_index)
      return -1;

   return ((int64_t)nir_intrinsic_desc_set(index) << 32) |
          nir_intrinsic_binding(index);
}

static bool
binding_in(const struct util_dynarray *bindings, int64_t binding)
{
   util_dynarray_foreach(bindings, int64_t, b) {
      if (*b == binding)
         return true;
   }
   return false;
}

/* A store dropped by an unmapped page stays visible in the cache to later
 * reads of the same address, so a store needs a residency guard when the
 * shader can read its binding back. Collect those stores before descriptor
 * lowering hides the binding. */
static void
collect_sparse_guarded_stores(nir_shader *nir, struct set *stores)
{
   struct util_dynarray read = UTIL_DYNARRAY_INIT;
   bool read_unknown = false;

   for (unsigned pass = 0; pass < 2; ++pass) {
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               const bool load =
                  intr->intrinsic == nir_intrinsic_load_deref ||
                  intr->intrinsic == nir_intrinsic_load_deref_block_intel ||
                  intr->intrinsic == nir_intrinsic_deref_atomic ||
                  intr->intrinsic == nir_intrinsic_deref_atomic_swap;
               const bool store =
                  intr->intrinsic == nir_intrinsic_store_deref ||
                  intr->intrinsic == nir_intrinsic_store_deref_block_intel;
               if (!load && !store)
                  continue;

               nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
               if (!deref || !(deref->modes & nir_var_mem_ssbo))
                  continue;

               const int64_t binding = ssbo_deref_binding(deref);
               if (pass == 0 && load) {
                  if (binding < 0)
                     read_unknown = true;
                  else if (!binding_in(&read, binding))
                     util_dynarray_append(&read, binding);
               } else if (pass == 1 && store &&
                          (read_unknown || binding < 0 ||
                           binding_in(&read, binding))) {
                  _mesa_set_add(stores, intr);
               }
            }
         }
      }
   }

   util_dynarray_fini(&read);
}

static bool
lower_sparse_buffer_deref(nir_builder *b, nir_intrinsic_instr *intr,
                          void *_data)
{
   const struct lower_descriptors_ctx *ctx = _data;
   if (!is_buffer_deref_access(intr->intrinsic))
      return false;

   const bool is_load = intr->intrinsic == nir_intrinsic_load_deref ||
                        intr->intrinsic == nir_intrinsic_load_deref_block_intel;
   const bool is_store =
      intr->intrinsic == nir_intrinsic_store_deref ||
      intr->intrinsic == nir_intrinsic_store_deref_block_intel;
   if (ctx->sparse_soft_unmapped &&
       (is_load ||
        (is_store && !_mesa_set_search(ctx->sparse_guarded_stores, intr))))
      return false;

   nir_deref_instr *leaf = nir_src_as_deref(intr->src[0]);
   if (!leaf || !(leaf->modes & nir_var_mem_ssbo))
      return false;

   nir_deref_path path;
   nir_deref_path_init(&path, leaf, NULL);
   nir_deref_instr *root = path.path[0];
   if (root->deref_type != nir_deref_type_cast ||
       root->parent.ssa->num_components != (storage_address_has_map(ctx) ? 5 : 4) ||
       root->parent.ssa->bit_size != 32) {
      nir_deref_path_finish(&path);
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *addr = root->parent.ssa;
   nir_def *base = nir_pack_64_2x32(b, nir_trim_vector(b, addr, 2));
   nir_def *resident = NULL;

   if (is_load || is_buffer_deref_atomic(intr->intrinsic) ||
       (is_store && ctx->sparse_soft_unmapped)) {
      static_assert(AIL_PAGESIZE == (1u << 14), "update shader lookup");

      nir_def *map_page = nir_channel(b, addr, storage_address_has_map(ctx) ? 4 : 3);
      nir_def *has_map = nir_ine_imm(b, map_page, 0);

      /* Ordinary buffers have no residency map. Keep their loads on the
       * direct path instead of executing the sparse address translation and
       * reading the zero page for every access. */
      nir_if *map_if = nir_push_if(b, has_map);
      nir_def *map_base =
         nir_ishl_imm(b, nir_u2u64(b, map_page), 14);
      nir_def *safe_map_base = nir_bcsel(
         b, has_map, map_base, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      nir_def *resource_base = nir_load_global_constant(
         b, 1, 64,
         nir_iadd_imm(b, safe_map_base,
                      offsetof(struct hk_sparse_buffer_map_header,
                               resource_base)),
         .align_mul = alignof(uint64_t),
         .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
      nir_def *page_count = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, safe_map_base,
                      offsetof(struct hk_sparse_buffer_map_header,
                               page_count)),
         .align_mul = alignof(uint32_t),
         .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
      nir_def *byte_offset =
         nir_build_deref_offset(b, leaf, glsl_get_natural_size_align_bytes);
      if (storage_address_has_map(ctx))
         byte_offset = nir_iadd(b, byte_offset, nir_channel(b, addr, 3));
      nir_def *rw_base = nir_iand_imm(b, base, ~ctx->sparse_ro_offset);
      nir_def *access_addr =
         nir_iadd(b, rw_base, nir_u2u64(b, byte_offset));
      nir_def *resource_offset = nir_isub(b, access_addr, resource_base);
      nir_def *page = nir_ushr_imm(b, resource_offset, 14);
      nir_def *page_in_range = nir_ult(b, page, nir_u2u64(b, page_count));
      nir_def *valid_map = nir_iand(b, has_map, page_in_range);
      nir_def *word = nir_ushr_imm(b, page, 5);
      nir_def *map_addr = nir_iadd_imm(
         b, nir_iadd(b, map_base,
                     nir_imul_imm(b, word, sizeof(uint32_t))),
         sizeof(struct hk_sparse_buffer_map_header));
      nir_def *safe_map_addr = nir_bcsel(
         b, valid_map, map_addr, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      nir_def *resident_word = nir_load_global_constant(
         b, 1, 32, safe_map_addr, .align_mul = sizeof(uint32_t),
         .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
      /* A variable MASK, not `(word >> (page & 31)) & 1`: that shape folds
       * into a one-bit ubitfield_extract whose offset never gets AGX's shift
       * mask back, so a buffer past the first bitmap word (512 KB) reads its
       * residency from the raw page number.  See the same note in
       * agx_nir_lower_texture.c's lower_sparse_buffer_texture(). */
      nir_def *resident_bit = nir_iand(
         b, resident_word,
         nir_ishl(b, nir_imm_int(b, 1),
                  nir_iand_imm(b, nir_u2u32(b, page), 31)));
      resident = nir_ior(
         b, nir_inot(b, has_map),
         nir_iand(b, page_in_range, nir_ine_imm(b, resident_bit, 0)));

      if (is_load) {
         nir_def *zero_base = nir_ior_imm(b, base, ctx->sparse_ro_offset);
         base = nir_bcsel(b, resident, base, zero_base);
      }

      nir_def *mapped_base = base;
      nir_def *mapped_resident = resident;
      nir_push_else(b, map_if);
      nir_def *direct_base =
         nir_pack_64_2x32(b, nir_trim_vector(b, addr, 2));
      nir_def *direct_resident = nir_imm_true(b);
      nir_pop_if(b, map_if);
      base = nir_if_phi(b, mapped_base, direct_base);
      resident = nir_if_phi(b, mapped_resident, direct_resident);
   }

   nir_def *lowered_addr =
      nir_vec4(b, nir_unpack_64_2x32_split_x(b, base),
               nir_unpack_64_2x32_split_y(b, base), nir_channel(b, addr, 2),
               nir_imm_int(b, 0));
   if (storage_address_has_map(ctx)) {
      nir_def *parts[] = {nir_channel(b, lowered_addr, 0),
         nir_channel(b, lowered_addr, 1), nir_channel(b, lowered_addr, 2),
         nir_channel(b, addr, 3), nir_imm_int(b, 0)};
      lowered_addr = nir_vec(b, parts, 5);
   }
   nir_deref_instr *lowered = nir_build_deref_cast_with_alignment(
      b, lowered_addr, root->modes, root->type, root->cast.ptr_stride,
      root->cast.align_mul, root->cast.align_offset);

   for (unsigned i = 1; path.path[i] != NULL; ++i)
      lowered = nir_build_deref_follower(b, lowered, path.path[i]);

   nir_src_rewrite(&intr->src[0], &lowered->def);
   nir_deref_path_finish(&path);

   if (is_store && resident) {
      nir_if *nif = nir_push_if(b, resident);
      nir_instr_remove(&intr->instr);
      nir_builder_instr_insert(b, &intr->instr);
      nir_pop_if(b, nif);
   }

   if (is_buffer_deref_atomic(intr->intrinsic)) {
      assert(resident != NULL);
      const bool return_value = !nir_def_is_unused(&intr->def);
      nir_def old_result;

      if (return_value) {
         old_result = intr->def;
         list_replace(&intr->def.uses, &old_result.uses);
         nir_def_init(&intr->instr, &intr->def, old_result.num_components,
                      old_result.bit_size);
      }

      nir_if *nif = nir_push_if(b, resident);
      nir_instr_remove(&intr->instr);
      nir_builder_instr_insert(b, &intr->instr);

      if (return_value) {
         nir_push_else(b, nif);
         nir_def *zero =
            nir_imm_zero(b, old_result.num_components, old_result.bit_size);
         nir_pop_if(b, nif);
         nir_def *result = nir_if_phi(b, &intr->def, zero);
         result->divergent = old_result.divergent;
         nir_def_rewrite_uses(&old_result, result);
      } else {
         nir_pop_if(b, nif);
      }
   }

   return true;
}

bool
hk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts,
                         bool sparse_buffer_residency,
                         bool sparse_soft_unmapped,
                         uint64_t sparse_ro_offset, bool scaled_sampling,
                         bool sampler_has_lod_bias, bool d3d_nearest_mip,
                         bool coherent_images_in_memory,
                         bool m5_store_window,
                         bool lower_array_cube_lod_query)
{
   struct lower_descriptors_ctx ctx = {
      .stage = nir->info.stage,
      .coherent_images_in_memory = coherent_images_in_memory,
      .m5_store_window = m5_store_window,
      .lower_array_cube_lod_query = lower_array_cube_lod_query,
      .scaled_sampling = scaled_sampling,
      .sampler_has_lod_bias = sampler_has_lod_bias,
      .d3d_nearest_mip = d3d_nearest_mip,
      .sparse_buffer_residency = sparse_buffer_residency,
      .sparse_soft_unmapped = sparse_soft_unmapped,
      .sparse_ro_offset = sparse_ro_offset,
      .clamp_desc_array_bounds =
         rs->storage_buffers !=
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||

         rs->uniform_buffers !=
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||

         rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,

      .ssbo_addr_format = hk_buffer_addr_format(
         rs->storage_buffers, rs->null_storage_buffer_descriptor, false),
      .ubo_addr_format = hk_ubo_addr_format(rs->uniform_buffers),
   };

   assert(set_layout_count <= HK_MAX_SETS);
   for (uint32_t s = 0; s < set_layout_count; s++) {
      if (set_layouts[s] != NULL) {
         ctx.set_layouts[s] = vk_to_hk_descriptor_set_layout(set_layouts[s]);
         const unsigned generation =
            ctx.set_layouts[s]->descriptor_generation;
         assert(!ctx.descriptor_generation ||
                ctx.descriptor_generation == generation);
         ctx.descriptor_generation = generation;
      }
   }

   ctx.ssbo_addr_format = hk_buffer_addr_format(
      rs->storage_buffers, rs->null_storage_buffer_descriptor,
      ctx.descriptor_generation == AGX_DESCRIPTOR_GENERATION_10);

   /* First lower everything but complex SSBOs, then lower complex SSBOs.
    *
    * TODO: See if we can unify this, not sure if the fast path matters on
    * Apple. This is inherited from NVK.
    */
   const nir_metadata preserved =
      ctx.coherent_images_in_memory || ctx.m5_store_window
         ? nir_metadata_none : nir_metadata_control_flow;
   const bool lower_sparse =
      ctx.sparse_buffer_residency &&
      (ctx.descriptor_generation == 9 || ctx.descriptor_generation == 10);
   bool pass_sparse_opt = false;
   if (lower_sparse && ctx.sparse_soft_unmapped) {
      /* Loads that only feed a partial store back, and dead loads, must not
       * make their binding look read. */
      pass_sparse_opt |= nir_opt_copy_prop_vars(nir);
      pass_sparse_opt |= nir_opt_dead_write_vars(nir);
      pass_sparse_opt |= nir_opt_copy_prop(nir);
      pass_sparse_opt |= nir_opt_dce(nir);
      ctx.sparse_guarded_stores = _mesa_pointer_set_create(NULL);
      collect_sparse_guarded_stores(nir, ctx.sparse_guarded_stores);
   }

   bool pass_lower_descriptors = nir_shader_instructions_pass(
      nir, try_lower_descriptors_instr, preserved, &ctx);

   bool pass_lower_ssbo = nir_shader_intrinsics_pass(
      nir, lower_ssbo_descriptor, nir_metadata_control_flow, &ctx);

   bool pass_sparse_buffers = false;
   if (lower_sparse) {
      pass_sparse_buffers = nir_shader_intrinsics_pass(
         nir, lower_sparse_buffer_deref, nir_metadata_none, &ctx);
   }
   if (ctx.sparse_guarded_stores)
      _mesa_set_destroy(ctx.sparse_guarded_stores, NULL);

   return pass_sparse_opt || pass_lower_descriptors || pass_lower_ssbo ||
          pass_sparse_buffers;
}
