/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <stdio.h>
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_compiler.h"
#include "agx_debug.h"

#include "nir.h"
#include "nir_builder.h"

#include <stdio.h>

/* Pre-raster graphics stages have no implicit derivatives. Their Poly
 * compute variants use the same native explicit-LOD texture operations, and
 * so does the VDM-launched G17P mesh program. */
static bool
g17p_explicit_lod_stage(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY ||
          stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_TESS_EVAL ||
          stage == MESA_SHADER_MESH;
}

/* M5 TA programs (API mesh and object/task) have no fragment derivatives and
 * take the same heap-indexed texture/image operands as compute. */
static bool
m5_ta_stage(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_MESH || stage == MESA_SHADER_TASK;
}

/* agx_nir_lower_g17p_indexed_samplers() is all-or-nothing per shader: one
 * unadmitted texture or image instruction and NOTHING in that shader gets a
 * heap index, so every texture instruction in it then reaches the packer with
 * a 32-bit bindless handle it has no encoding for.  A refusal census therefore
 * measures shaders, not instructions, and says nothing about which shape
 * actually caused it.  Name the instruction that did.
 */
static bool
g17p_decline(const char *why, const nir_tex_instr *tex,
             const nir_intrinsic_instr *intr, mesa_shader_stage stage)
{
   if (!(agx_compiler_debug & AGX_DBG_SHADERS))
      return false;

   if (tex) {
      fprintf(stderr,
              "G17P indexed-sampler decline (%s): stage=%u op=%u dim=%u "
              "array=%u shadow=%u sparse=%u coords=%u srcs=%u dest_type=%u "
              "comps=%u bits=%u flags=0x%x tex_idx=%u smp_idx=%u\n",
              why, stage, tex->op, tex->sampler_dim, tex->is_array,
              tex->is_shadow, tex->is_sparse, tex->coord_components,
              tex->num_srcs, tex->dest_type, tex->def.num_components,
              tex->def.bit_size, tex->backend_flags, tex->texture_index,
              tex->sampler_index);
   } else if (intr) {
      fprintf(stderr,
              "G17P indexed-sampler decline (%s): stage=%u image_load dim=%u "
              "array=%u comps=%u bits=%u fmt=%u access=0x%x coordc=%u "
              "coordb=%u lod_const=%u ms_const=%u handle_const=%u\n",
              why, stage, nir_intrinsic_image_dim(intr),
              nir_intrinsic_image_array(intr), intr->def.num_components,
              intr->def.bit_size, (unsigned)nir_intrinsic_format(intr),
              nir_intrinsic_access(intr),
              nir_src_num_components(intr->src[1]),
              nir_src_bit_size(intr->src[1]),
              nir_src_is_const(intr->src[2]),
              nir_src_is_const(intr->src[3]),
              nir_src_as_intrinsic(intr->src[0]) &&
                 nir_src_is_const(nir_src_as_intrinsic(intr->src[0])->src[0]));
   }
   return false;
}

static bool
g17p_sample_shape(const nir_tex_instr *tex, bool m5, bool g17p)
{
   const bool query_lod = tex->op == nir_texop_lod;
   const int backend_coord =
      nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   const unsigned coords = tex->coord_components == 0 &&
      tex->op == nir_texop_tg4 && backend_coord >= 0
         ? nir_src_num_components(tex->src[backend_coord].src)
         : tex->coord_components;
   const bool three_d = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   const bool cube_array = cube && tex->is_array;
   const bool offset_array = tex->is_array && !cube &&
      (nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_backend2) >= 0);
   const int level_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int bias_src = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   /* Native G17P depthcube, depthcube_array and depth2d_array level-zero
    * comparisons retain the ordinary face/layer staging and select SMP
    * f139=2. Keep descriptor legalization aligned with the binary packer. */
   const bool shadow_level_zero = tex->is_shadow &&
      tex->op == nir_texop_txl && level_src >= 0 &&
      nir_src_is_const(tex->src[level_src].src) &&
      nir_src_as_float(tex->src[level_src].src) == 0.0;
   /* A per-lane level or bias is published beside the operand its dimension
    * already owns -- the layer for a noncube array (agx3_tex_stage_array_lod),
    * CEM's face for a cube (agx3_tex_stage_cube_lod) -- so it is the runtime
    * value, not a constant, that the comparison forms admit. A constant level
    * or bias would instead have to share SMP's immediate f29/f43 slot with
    * that same operand, which is the collision agx3_emit_texture_sample()
    * declines. */
   const bool register_level = tex->op == nir_texop_txl && level_src >= 0 &&
      !nir_src_is_const(tex->src[level_src].src);
   const bool register_bias = tex->op == nir_texop_txb && bias_src >= 0 &&
      !nir_src_is_const(tex->src[bias_src].src);

   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
       tex->op != nir_texop_tg4 && !query_lod) ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D && !three_d && !cube) ||
       /* 3D takes the implicit, immediate-bias and per-lane LOD shapes the
        * 2D dimension takes; only gradient3d, which stages six scalars
        * rather than four, has no witness. */
       (three_d &&
        (tex->is_array || tex->is_shadow ||
         (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
          tex->op != nir_texop_txl && tex->op != nir_texop_txd && !query_lod))) ||
       /* A cube comparison publishes its reference through the same CEM
        * bracket the colour cube uses, with implicit or explicit zero LOD.
        * A colour cube also takes an explicit per-lane level, on the
        * plain cube exactly as on the cube array -- Apple's own
        * t.sample(s, dir, level(reg)) for texturecube<float> is
        *
        *   05800cb8b0000003000010000100
        *
        * which is the instruction agx3_tex_encode_sample() already emits for
        * AGX3_TEX_LOD_REGISTER on a cube, field for field. Admitting only the
        * array form
        * here was what refused textureLod(samplerCube, dir, lod) -- the
        * roughness-indexed probe lookup every image-based lighting shader
        * does, SuperTuxKart's IBL.frag included.
        *
        * A cube-array comparison takes a per-lane bias beside its face and
        * layer, as the packer stages it: a Gen4 sampler has no bias field,
        * so G15 hands every depth cube-array texture() its sampler bias this
        * way (dEQP-VK.texture.shadow.cube_array). A constant bias or a bias
        * with a minimum stays closed. */
       /*
        * On the M5 the cube-array comparison also takes a per-lane level: the
        * packer stages it beside the face and the layer, as it stages the
        * cube comparison's level beside the face (agx3_emit_texture_sample).
        * It is the form compute shaders with derivative groups reach, whose
        * texture() agx_nir_lower_compute_derivatives() turns into txl
        * (dEQP-VK.glsl.texture_functions.texture.samplercubearrayshadow_compute).
        * Refusing it here sent the whole shader to the packer as a software
        * heap handle, which has no AGX3 encoding.
        */
       (cube &&
        (tex->op != nir_texop_tex && tex->op != nir_texop_tg4 && !query_lod &&
         !((!tex->is_shadow || shadow_level_zero ||
            (!cube_array && level_src >= 0 &&
             !nir_src_is_const(tex->src[level_src].src)) ||
            (m5 && cube_array && register_level)) &&
            tex->op == nir_texop_txl) &&
         !(tex->op == nir_texop_txb &&
           (!tex->is_shadow ||
            nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx) >= 0 ||
            register_bias) &&
           !(tex->is_shadow && cube_array && !register_bias)))) ||
       /* Sparse residency rides on the ordinary sample: the encoder sets
        * three extra fields and the instruction writes one more register.
        * A depth comparison and a LOD query have no sparse witness. */
       (tex->is_sparse && (tex->is_shadow || query_lod)) ||
       coords != (query_lod ? (three_d || cube ? 3 : 2) :
                                cube_array ? 4 :
                                (tex->is_array || three_d || cube) ? 3 : 2) ||
       (tex->backend_flags & ~(AGX_TEXTURE_FLAG_CLAMP_TO_0 |
                               AGX_TEXTURE_FLAG_NO_CLAMP)) != 0 ||
       tex->backend_flags == (AGX_TEXTURE_FLAG_CLAMP_TO_0 |
                              AGX_TEXTURE_FLAG_NO_CLAMP) ||
       (tex->dest_type != nir_type_float32 &&
        tex->dest_type != nir_type_uint32 &&
        tex->dest_type != nir_type_int32) ||
       tex->def.num_components !=
          ((query_lod ? 2 : (tex->is_shadow && tex->op != nir_texop_tg4) ? 1 : 4) +
           (tex->is_sparse ? 1 : 0)) ||
       (query_lod && tex->dest_type != nir_type_float32) ||
       tex->def.bit_size != 32)
      return false;

   const int packed_bias_min =
      nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx);
   const int minimum = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);

   if (tex->is_shadow && !query_lod) {
      /* The depth comparison is orthogonal to the LOD shape: Apple's
       * depth2d sample_compare with level(k), level(reg), bias(k), bias(reg)
       * and gradient2d() each differ from the corresponding colour sample by
       * f143 alone, and gather_compare is the ordinary gather plus f143.
       * See the witness table in agx3_tex_core.c. The array and cube
       * comparisons use separately packed layer / CEM operands and admit
       * only implicit LOD, the native explicit-zero form above, or a
       * clamped bias or gradient, which publish their minimum beside the
       * layer or in a packed register. */
      const int comparator =
         nir_tex_instr_src_index(tex, nir_tex_src_comparator);
      if (comparator < 0 ||
          nir_src_num_components(tex->src[comparator].src) != 1 ||
          nir_src_bit_size(tex->src[comparator].src) != 32 ||
          (tex->is_array && !offset_array && tex->op != nir_texop_tex &&
           tex->op != nir_texop_tg4 &&
           !(tex->op == nir_texop_txb && packed_bias_min >= 0) &&
           /* GLSL's remaining comparison LOD shapes for an array depth
            * sampler: textureLod and texture(..., bias) on
            * sampler1DArrayShadow, and textureGrad on sampler1DArrayShadow
            * and sampler2DArrayShadow -- a gradient now with or without a
            * minimum. Each stages its scalar in a word the packer has
            * emitted since the colour forms landed: the level or bias beside
            * the layer, the gradient in banks 8..14 with the layer still on
            * PACK_SMPIR bank 6. Only this gate held them back, and refusing
            * them here does not fall back to anything -- the pass is
            * all-or-nothing per shader, so every texture instruction in the
            * shader then reached the packer as a software-heap bindless
            * handle, which has no AGX3 encoding, and the pipeline failed
            * with VK_ERROR_UNKNOWN. */
           !register_level && !register_bias &&
           /* G17P also takes a constant level or bias: the backend keeps
            * it in a register (agx_optimizer_inline_imm), so it is staged
            * beside the layer like a per-lane one. */
           !(g17p && !cube &&
             ((tex->op == nir_texop_txl && level_src >= 0) ||
              (tex->op == nir_texop_txb && bias_src >= 0))) &&
           tex->op != nir_texop_txd &&
           !shadow_level_zero))
         return false;
   }

   const bool has_minimum = packed_bias_min >= 0 || minimum >= 0;
   const bool has_bias = packed_bias_min >= 0 ||
      nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0;
   const bool has_lod = nir_tex_instr_src_index(tex, nir_tex_src_lod) >= 0;
   const int ddx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
   const int ddy = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
   const bool has_grad = ddx >= 0 && ddy >= 0;
   /* Constant coordinate offsets use one immediate nibble per coordinate in
    * SMP (f43/f51/f141=1). Runtime offsets keep their component vector for
    * PACK_SMPIR and select the staged form with f141=2; this includes native
    * gathers as well as ordinary 2D samples. */
   int offset_src = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   const bool backend_offset = offset_src < 0;
   if (backend_offset)
      offset_src = nir_tex_instr_src_index(tex, nir_tex_src_backend2);
   const bool has_offset = offset_src >= 0;
   const bool packed_offset =
      has_offset && backend_offset &&
      nir_src_num_components(tex->src[offset_src].src) == 1;
   const bool staged_packed_offset =
      packed_offset && !nir_src_is_const(tex->src[offset_src].src) &&
      (tex->op == nir_texop_tg4 || tex->is_array || has_lod || has_bias ||
       has_minimum);
   const uint32_t constant_offset =
      has_offset && nir_src_is_const(tex->src[offset_src].src)
         ? nir_src_as_uint(tex->src[offset_src].src)
         : 0;
   /* lower_regular_texture() keeps G17P gather X/Y as two 16-bit lanes so a
    * runtime value can feed PACK_SMPIR directly.  A constant therefore has
    * its signed nibbles in bits 0..3 and 16..19 (for example, CTS offset
    * (+8,+8) is 0x00080008), not in the compact SMP byte used after
    * agx_emit_tex() folds it.  Admit exactly that generated representation;
    * the backend later maps it to (y << 4) | x. */
   /* C's texture lowering (generation 10) packs gather offsets like any
    * other constant offset: compact (y << 4) | x. */
   const bool valid_constant_offset =
      tex->op == nir_texop_tg4 && !m5
         ? (constant_offset & ~0x000f000fu) == 0
         : constant_offset <= (three_d ? 0xfffu : 0xffu);
   /* Native bank-6 minimum operands are grounded for 2D, 2D-array and 3D
    * fragment samples, and for biased cube samples. The shared texture lowering has already combined a
    * bias/minimum pair into one f16x2 source, or left the gradient minimum
    * as a scalar half. Preserve that source while selecting native handles.
    * A 2D runtime offset carries the bias or gradient minimum in its packed
    * bank-6 word; an unpacked component offset would overwrite it. A runtime
    * 3D offset is witnessed beside a bias only. */
   if (has_minimum &&
       ((cube && packed_bias_min < 0) ||
        (three_d && has_offset && packed_bias_min < 0 &&
         !nir_src_is_const(tex->src[offset_src].src)) ||
        (has_offset && !three_d && !packed_offset &&
         !nir_src_is_const(tex->src[offset_src].src)) ||
        (packed_bias_min >= 0 &&
         (tex->op != nir_texop_txb || minimum >= 0 ||
          nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 ||
          nir_src_num_components(tex->src[packed_bias_min].src) != 1 ||
          nir_src_bit_size(tex->src[packed_bias_min].src) != 32)) ||
        (minimum >= 0 &&
         (tex->op != nir_texop_txd ||
          nir_src_num_components(tex->src[minimum].src) != 1 ||
          nir_src_bit_size(tex->src[minimum].src) != 16))))
      return false;
   if (query_lod && has_offset)
      return false;
   if (has_offset &&
       (nir_src_num_components(tex->src[offset_src].src) !=
           (packed_offset ? 1u : three_d ? 3u : 2u) ||
        (!packed_offset && three_d) ||
        (packed_offset && !staged_packed_offset &&
         (!nir_src_is_const(tex->src[offset_src].src) ||
           !valid_constant_offset)) ||
         cube))
      return false;

   const unsigned expected_srcs =
      (has_grad ? 5 : ((has_bias || has_lod) ? 4 : 3)) +
      (tex->is_shadow && !query_lod) +
      has_offset + (minimum >= 0);
   if (tex->num_srcs != expected_srcs ||
       has_lod != (tex->op == nir_texop_txl) ||
       has_bias != (tex->op == nir_texop_txb) ||
       has_grad != (tex->op == nir_texop_txd))
      return false;

   /* Metal's gather takes no LOD argument, so no gather carries one. The
    * array layer, the coordinate offset and the depth comparison all
    * compose with it. Cube gathers use the existing CEM coordinate path;
    * 3D gathers remain unsupported. */
   if (tex->op == nir_texop_tg4 && (three_d || tex->component > 3))
      return false;

   /* An explicit gradient is four 32-bit scalars staged into their own
    * operand banks. G17P array gradients name the independent dynamic layer
    * register directly in the sample instruction. */
   const unsigned grad_components = three_d ? 3u : 2u;
   if (has_grad &&
       (nir_src_num_components(tex->src[ddx].src) != grad_components ||
        nir_src_num_components(tex->src[ddy].src) != grad_components ||
        nir_src_bit_size(tex->src[ddx].src) != 32 ||
        nir_src_bit_size(tex->src[ddy].src) != 32))
      return false;

   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   return coord >= 0 &&
          nir_src_num_components(tex->src[coord].src) == coords &&
          nir_src_bit_size(tex->src[coord].src) == 32 &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
}

static bool
g17p_handle_binding(const nir_tex_instr *tex, nir_tex_src_type source,
                    nir_intrinsic_op intrinsic, uint8_t *set,
                    uint32_t *offset_B)
{
   const int index = nir_tex_instr_src_index(tex, source);
   if (index < 0)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(tex->src[index].src);
   if (!handle || handle->intrinsic != intrinsic ||
       nir_intrinsic_desc_set(handle) > UINT8_MAX ||
       !nir_src_is_const(handle->src[0]))
      return false;

   *set = nir_intrinsic_desc_set(handle);
   *offset_B = nir_src_as_uint(handle->src[0]);
   return true;
}

static bool
g17p_handle_offset(const nir_tex_instr *tex, nir_tex_src_type source,
                   nir_intrinsic_op intrinsic, uint32_t *offset_B)
{
   uint8_t set;
   return g17p_handle_binding(tex, source, intrinsic, &set, offset_B) &&
          set == 0;
}

static bool
g17p_is_set_address(nir_def *def, uint8_t set)
{
   nir_scalar address = nir_scalar_resolved(def, 0);
   nir_intrinsic_instr *load =
      nir_def_as_intrinsic_or_null(address.def);

   return address.comp == 0 && load &&
          load->intrinsic == nir_intrinsic_load_descriptor_set_agx &&
          nir_intrinsic_desc_set(load) == set;
}

static bool
g17p_is_constant_scalar(nir_scalar offset, uint32_t expected)
{
   offset = nir_scalar_chase_movs(offset);
   if (nir_scalar_is_const(offset))
      return nir_scalar_as_uint(offset) == expected;

   nir_alu_instr *convert = nir_def_as_alu_or_null(offset.def);
   if (offset.comp != 0 || !convert || convert->op != nir_op_u2u64)
      return false;

   return g17p_is_constant_scalar(nir_scalar_chase_alu_src(offset, 0),
                                  expected);
}

static bool
g17p_is_constant_offset(nir_def *def, uint32_t expected)
{
   return g17p_is_constant_scalar(nir_get_scalar(def, 0), expected);
}

/* Match only the driver-generated software LOD-bias field belonging to this
 * sampled descriptor. API samplers carry the same bias in their G17P native
 * sampler word, so this zero-valued software load must not become a txb.
 * A shader with no bias source at all has nothing to classify and passes;
 * g17p_sample_shape() has already tied the presence of the source to txb.
 */
static bool
g17p_is_sampler_bias(const nir_tex_instr *tex, uint8_t set,
                     uint32_t expected_offset_B)
{
   const int index = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (index < 0)
      return tex->op != nir_texop_txb;

   nir_intrinsic_instr *load = nir_src_as_intrinsic(tex->src[index].src);
   if (!load || load->def.num_components != 1 || load->def.bit_size != 16)
      return false;

   if (load->intrinsic == nir_intrinsic_load_global_constant_offset ||
       load->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      return g17p_is_set_address(load->src[0].ssa, set) &&
             g17p_is_constant_offset(load->src[1].ssa, expected_offset_B);
   }

   if (load->intrinsic != nir_intrinsic_load_global_constant)
      return false;

   nir_scalar address = nir_scalar_resolved(load->src[0].ssa, 0);
   nir_alu_instr *add = nir_def_as_alu_or_null(address.def);
   if (address.comp != 0 || !add || add->op != nir_op_iadd)
      return false;

   nir_scalar lhs = nir_scalar_chase_alu_src(address, 0);
   nir_scalar rhs = nir_scalar_chase_alu_src(address, 1);

   return (g17p_is_set_address(lhs.def, set) &&
           g17p_is_constant_scalar(rhs, expected_offset_B)) ||
          (g17p_is_set_address(rhs.def, set) &&
           g17p_is_constant_scalar(lhs, expected_offset_B));
}

/* The native texture/sampler table prefix is separate from the fragment
 * push-root ABI. Residual descriptor accesses need their own lowering, but a
 * root load used by push constants is valid alongside this sampled resource. */
static bool
g17p_has_descriptor_access(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_bindless_image_agx ||
                op == nir_intrinsic_bindless_sampler_agx ||
                op == nir_intrinsic_load_descriptor_set_agx)
               return true;
         }
      }
   }

   return false;
}

/* Array-coordinate legalization reads the texture descriptor to clamp the
 * API layer before the sample.  The combined-sampler fragment ABI already
 * publishes that exact 32-byte descriptor pointer in preamble halves 0..3;
 * retaining the original descriptor-set handle as well would require a
 * second, unavailable root.  Replace only compiler-generated descriptor
 * crawls of the selected texture with the native pointer.  Any independent
 * handle/use remains below and makes the specialization fail closed. */
static bool
g17p_rewrite_combined_texture_crawls(nir_shader *shader, uint8_t set,
                                     uint32_t texture_offset_B)
{
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
            if (load->intrinsic !=
                nir_intrinsic_load_from_texture_handle_agx)
               continue;

            nir_intrinsic_instr *handle = nir_src_as_intrinsic(load->src[0]);
            if (!handle ||
                handle->intrinsic != nir_intrinsic_bindless_image_agx ||
                nir_intrinsic_desc_set(handle) != set ||
                !nir_src_is_const(handle->src[0]) ||
                nir_src_as_uint(handle->src[0]) != texture_offset_B)
               return false;

            b.cursor = nir_before_instr(instr);
            nir_def_replace(&load->def,
                            nir_load_preamble(&b, 1, 64, .base = 0));
         }
      }
   }

   return true;
}

/* The checks of agx_nir_lower_g17p_combined_sampler() made before it
 * changes anything: the shader's one texture instruction, a sample of the
 * supported shape, through a Gen9 sampled descriptor. */
static nir_tex_instr *
g17p_combined_sampler_candidate(const nir_shader *shader,
                                uint32_t *texture_offset_B,
                                uint32_t *sampler_state_offset_B, bool g17p)
{
   if (!shader || (shader->info.stage != MESA_SHADER_FRAGMENT &&
                   shader->info.stage != MESA_SHADER_COMPUTE))
      return NULL;

   nir_tex_instr *sample = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            if (sample ||
                !g17p_sample_shape(nir_instr_as_tex(instr), false, g17p))
               return NULL;
            sample = nir_instr_as_tex(instr);
         }
      }
   }

   if (!sample)
      return NULL;

   if (!g17p_handle_offset(sample, nir_tex_src_texture_handle,
                           nir_intrinsic_bindless_image_agx,
                           texture_offset_B) ||
       !g17p_handle_offset(sample, nir_tex_src_sampler_handle,
                           nir_intrinsic_bindless_sampler_agx,
                           sampler_state_offset_B))
      return NULL;

   /* Gen9 sampled descriptors contain one 32-byte texture followed by the
    * eight-byte native sampler word and its 16-bit global-heap index.
    */
   if ((*texture_offset_B & 7) != 0 ||
       *sampler_state_offset_B != *texture_offset_B + 32)
      return NULL;

   return sample;
}

bool
agx_nir_g17p_combined_sampler_candidate(const nir_shader *shader, bool g17p)
{
   uint32_t texture_offset_B, sampler_state_offset_B;
   return g17p_combined_sampler_candidate(shader, &texture_offset_B,
                                          &sampler_state_offset_B,
                                          g17p) != NULL;
}

bool
agx_nir_lower_g17p_combined_sampler(
   nir_shader *shader, struct agx_g17p_combined_sampler_binding *binding,
   bool g17p)
{
   uint32_t texture_offset_B, sampler_state_offset_B;
   nir_tex_instr *sample =
      binding ? g17p_combined_sampler_candidate(shader, &texture_offset_B,
                                                &sampler_state_offset_B, g17p)
              : NULL;
   if (!sample)
      return false;

   /* The Gen9 software LOD bias is at byte 12 of sampler state, and the
    * native sampler word already applies it, so that load -- and only that
    * load -- is dropped. A genuine API bias is a different value entirely:
    * keep it, and keep the sample a txb, so the hardware applies both the
    * descriptor bias and the shader's. */
   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   if (sample->op == nir_texop_txb &&
       g17p_is_sampler_bias(sample, 0, sampler_state_offset_B + 12)) {
      nir_steal_tex_src(sample, nir_tex_src_bias);
      sample->op = nir_texop_tex;
   }
   sample->texture_index = 0;
   /* HAL200's bounded fragment ABI binds independent texture/sampler table
    * pointers. Each table is rebased to the selected API resource, so both
    * hardware indices are zero; the legacy texel-fetch reservation is not
    * an extra entry in this sampler table.
    */
   sample->sampler_index = 0;

   if (!g17p_rewrite_combined_texture_crawls(shader, 0, texture_offset_B))
      return false;

   NIR_PASS(_, shader, nir_opt_dce);
   /* The initial fragment ABI replaces the whole descriptor root. Compute
    * reserves the same native texture/sampler pointer prefix but may keep
    * unrelated storage-image/SSBO descriptors in the ordinary root. */
   if (shader->info.stage == MESA_SHADER_FRAGMENT &&
       g17p_has_descriptor_access(shader))
      return false;

   *binding = (struct agx_g17p_combined_sampler_binding){
      .texture_offset_B = texture_offset_B,
      .sampler_index_offset_B = sampler_state_offset_B + 8,
   };
   return true;
}

bool
agx_nir_lower_g17g_combined_sampler(
   nir_shader *shader, struct agx_g17p_combined_sampler_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            if (tex->op != nir_texop_txl ||
                tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
                tex->is_shadow || tex->dest_type != nir_type_float32 ||
                lod < 0 || !nir_src_is_const(tex->src[lod].src) ||
                nir_src_as_float(tex->src[lod].src) != 0.0)
               return false;
         }
      }
   }

   /* Both generations place the sampler word at +32 and its heap index at
    * +40. The common pass validates a single static set-0 resource and
    * rewrites only that resource's descriptor crawls. No API LOD bias is
    * removed here: the bounded M5 shape has an explicit zero level. */
   return agx_nir_lower_g17p_combined_sampler(shader, binding, false);
}

/* Ordinary native integer-coordinate read. The descriptor selects float,
 * signed, or unsigned texel interpretation; the result is a 32-bit register
 * value in each case. Input attachments arrive as 2D-array reads. */
static bool
g17p_indexed_fetch_shape(const nir_tex_instr *tex)
{
   const bool multisample = tex->op == nir_texop_txf_ms;
   const bool three_d = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);

   /* Buffer textures are lowered to a 2D hardware texture before this pass.
    * That lowering deliberately preserves coord_components == 1 while its
    * backend coordinate is the remapped vec2.  Admit exactly that canonical
    * mismatch alongside ordinary 2D/2D-array fetches.
    */
   const bool lowered_buffer =
      tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array &&
      tex->coord_components == 1 && coord >= 0 &&
      tex->src[coord].src.ssa->num_components == 2;

   if ((tex->op != nir_texop_txf && !multisample) ||
       tex->sampler_dim !=
          (multisample ? GLSL_SAMPLER_DIM_MS
                       : three_d ? GLSL_SAMPLER_DIM_3D
                                 : GLSL_SAMPLER_DIM_2D) ||
       (three_d && tex->is_array) ||
       tex->is_shadow || tex->texture_index ||
       /* No sparse multisample read is witnessed. */
       (tex->is_sparse && multisample) ||
       tex->sampler_index || tex->def.bit_size != 32 ||
       tex->def.num_components != (tex->is_sparse ? 5u : 4u) ||
       (tex->dest_type != nir_type_float32 && tex->dest_type != nir_type_int32 &&
        tex->dest_type != nir_type_uint32) ||
       (!lowered_buffer &&
        tex->coord_components != (tex->is_array || three_d ? 3 : 2)) ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_NO_CLAMP))
      return false;
   const int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int sample =
      nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   /* The backend stages a dynamic uint level independently of the texture
    * result for 2D, array, and 3D fetches. Keep that SSA value through descriptor
    * lowering; leaving the old base/byte-offset handle in place is not an
    * encoding of the native texture heap index. The fetch instruction's 3D
    * selector is orthogonal to the integer-LOD PACK_SMPIR transaction. */
   const bool valid_lod = lod < 0 ||
      (nir_src_num_components(tex->src[lod].src) == 1 &&
       (nir_src_bit_size(tex->src[lod].src) == 16 ||
        nir_src_bit_size(tex->src[lod].src) == 32));
   const unsigned lowered_coord_components =
      coord >= 0 ? nir_src_num_components(tex->src[coord].src) : 0;
   return handle >= 0 && coord >= 0 &&
      (multisample
          ? (tex->num_srcs == (lod >= 0 ? 3 : 2) && sample < 0 &&
             (lod < 0 || (nir_src_is_const(tex->src[lod].src) &&
                          nir_src_as_uint(tex->src[lod].src) == 0)) &&
             /* agx_nir_lower_texture packs sample and optional array layer
              * into the third 32-bit backend coordinate. */
             lowered_coord_components == 3)
          : (tex->num_srcs == (lod >= 0 ? 3 : 2) && sample < 0 && valid_lod)) &&
      nir_src_bit_size(tex->src[coord].src) == 32 &&
      (multisample || lowered_coord_components ==
                         (lowered_buffer ? 2 : tex->coord_components));
}

/* Descriptor lowering normally wraps the software heap index in a bindless
 * sampler. The custom-border zero-clamp sample already carries the companion
 * index directly: its descriptor load and nonnegative runtime branch must
 * remain intact. Both indices include the same AGX2 reservation of 28. */
static nir_def *
g17p_sample_heap_index(const nir_tex_instr *tex)
{
   const int index = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (index < 0)
      return NULL;

   nir_src handle = tex->src[index].src;
   nir_intrinsic_instr *sampler = nir_src_as_intrinsic(handle);
   nir_def *value = NULL;
   if (sampler && sampler->intrinsic == nir_intrinsic_bindless_sampler_agx)
      value = sampler->src[1].ssa;
   else if (tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0)
      value = handle.ssa;

   return value && value->num_components == 1 && value->bit_size == 16
             ? value : NULL;
}

/* Read-only storage images use the same native texture heap and texel-fetch
 * sampler as ordinary indexed txf. The software descriptor tail supplies the
 * heap index; the original handle remains live for robust descriptor bounds
 * loads, which are independent of the hardware fetch operand. */
static bool
g17p_indexed_image_fetch_shape(const nir_intrinsic_instr *intr)
{
   /* No format condition.  This predicate only decides whether the image's
    * descriptor handle is replaced by a heap index; the rewrite is
    * format-agnostic and agx3_emit_texture_fetch() never reads I->format --
    * its only uses of the field are in agx3_emit_storage_image_write(), a
    * different instruction.  Apple agrees: metal-nt -arch applegpu_g17p emits
    * BYTE-IDENTICAL code for texture2d<float>, texture2d<uint> and
    * texture2d<int> read(), one variable per kernel,
    *
    *   05 80 0c b8 b0 00 17 01 00 00 00 00 01 00
    *
    * so the component type lives in the descriptor and the fetch has no field
    * for it.  The old PIPE_FORMAT_R8G8B8A8_UNORM equality was the format of
    * the kernel this was first witnessed against, not a constraint, and
    * because the pass is all-or-nothing per shader it took every OTHER
    * texture instruction in the shader down with it: an r32ui or r32i
    * imageLoad -- what every compute shader and every WebGPU storage texture
    * uses -- refused the whole shader.
    */
   /* Admit exactly the dimensions agx3_emit_texture_fetch() encodes for an
    * image load: 2D, 2D array and 3D. Vulkan storage cube views are packed as
    * 2D-array descriptors by hk_image_view, and agx_emit_image_load() already
    * maps both cube forms onto AGX_DIM_2D_ARRAY. They therefore use the same
    * qualified fetch as an ordinary array image; rejecting the original NIR
    * dimension here only leaves its unencodable software descriptor handle in
    * place. Multisample uses the emitter's native sample/layer packing and
    * may have a nonzero or dynamic sample index. 1D is also a 2D fetch:
    * lower_1d_image() rewrites the dimension before emit, so the indexed
    * handle feeds the same 2D/2D-array encoding. */
   const enum glsl_sampler_dim image_dim = nir_intrinsic_image_dim(intr);
   const bool ms = image_dim == GLSL_SAMPLER_DIM_MS;
   /* OpImageSparseRead arrives as bindless_image_sparse_load: the same fetch
    * with one more destination component for the residency code.  Without
    * this the pass leaves the software heap-index handle in place and the
    * packer refuses the shape it cannot encode. */
   const bool sparse = intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;
   if ((intr->intrinsic != nir_intrinsic_bindless_image_load && !sparse) ||
       (sparse && ms) ||
      !(ms || image_dim == GLSL_SAMPLER_DIM_2D ||
        image_dim == GLSL_SAMPLER_DIM_1D ||
        image_dim == GLSL_SAMPLER_DIM_CUBE ||
        (image_dim == GLSL_SAMPLER_DIM_3D &&
         !nir_intrinsic_image_array(intr))) ||
       intr->def.num_components < 1 ||
       intr->def.num_components > (sparse ? 5u : 4u) ||
       (sparse && intr->def.num_components != 5) ||
       intr->def.bit_size != 32 ||
       nir_src_num_components(intr->src[1]) != 4 ||
       nir_src_bit_size(intr->src[1]) != 32 ||
       (!ms && (!nir_src_is_const(intr->src[2]) ||
                nir_src_as_uint(intr->src[2]) != 0)) ||
      !nir_src_is_const(intr->src[3]) ||
      nir_src_as_uint(intr->src[3]) != 0)
      return false;

   /* A load cannot be truly non-readable: nir_opt_access stamps
    * ACCESS_NON_READABLE on loads of write-only-declared images (the meta
    * copy's destination read-back) and on a module's only buffer read, and
    * reading a write-only image is undefined-behavior-legal to compile. The
    * fetch rewrite is access-agnostic, so such loads lower like any other.
    *
    * Coherent and volatile reads take the same fetch.  G17P's texture read
    * has no coherency field: metal-nt -arch applegpu_g17p compiles a
    * read_write texture's read() after a write() to the same smp form as an
    * access::read texture's read(), differing only in the staged coordinate
    * halves and the result register,
    *
    *   texture2d<int, access::read>        05800cb8b0001701000000000100
    *   texture2d<int, access::read_write>  05800cb9b0001701000000000200
    *
    * and the preceding store's fence already retires the write before any
    * later texture instruction is issued. */

   /* The descriptor offset does not have to be a constant.  The rewrite below
    * is pure SSA arithmetic -- nir_iadd_imm() onto the offset, then one
    * nir_load_global_constant() with .align_mul = 4 -- and runs the same for a
    * dynamically indexed descriptor as for a constant one.  The texture path
    * (g17p_indexed_fetch_shape) never required constness in the first place,
    * so requiring it here was an asymmetry, not a constraint.
    *
    * Keep the 8-alignment assertion wherever it can still be evaluated.  It is
    * stronger than the load needs; every descriptor stride the layout produces
    * is a multiple of 8, so base + index * stride + 64 stays 8-aligned for a
    * dynamic index too.
    */
   nir_intrinsic_instr *image = nir_src_as_intrinsic(intr->src[0]);
   return image && image->intrinsic == nir_intrinsic_bindless_image_agx &&
          nir_intrinsic_desc_set(image) <= UINT8_MAX &&
          (!nir_src_is_const(image->src[0]) ||
           (nir_src_as_uint(image->src[0]) & 7) == 0);
}

/* A constant descriptor is inside its mapped set, so its heap index may be read
 * ahead of the sample, out of a branch or loop into the constant program. This
 * GPU has no soft faults, and a dynamic index is only known to be in range where
 * the sample really runs: `if (i >= 0) ...` must not read element -1 per draw.
 */
static enum gl_access_qualifier
heap_index_access(const nir_intrinsic_instr *image)
{
   return nir_src_is_const(image->src[0]) ? ACCESS_CAN_SPECULATE : 0;
}

/* Native indexed samples consume zero-extended heap indices, loaded by the
 * GPU from the descriptor set. Keep descriptor-set roots alive so updates
 * after command recording and dynamically selected array elements are visible.
 */
static bool
lower_indexed_samplers(nir_shader *shader, int native32_set, bool m5,
                       bool g17p)
{
   if (!shader || (!g17p_explicit_lod_stage(shader->info.stage) &&
                   shader->info.stage != MESA_SHADER_FRAGMENT &&
                   shader->info.stage != MESA_SHADER_COMPUTE &&
                   !(m5 && m5_ta_stage(shader->info.stage))))
      return false;

   unsigned samples = 0, image_reads = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                   intr->intrinsic ==
                      nir_intrinsic_bindless_image_sparse_load) {
                  if (native32_set >= 0 || !g17p_indexed_image_fetch_shape(intr))
                     return g17p_decline("image_fetch_shape", NULL, intr,
                                         shader->info.stage);
                  image_reads++;
               }
               continue;
            }
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            const bool fetch = g17p_indexed_fetch_shape(tex);
            /* txb and txd reach the packer as the same sample instruction
             * with a different LOD mode; the heap-indexed descriptor operand
             * is orthogonal to it. */
            /* Explicit levels and gradients need no fragment derivatives.
             * Gather uses the packer's existing non-fragment level-zero form.
             * Ordinary implicit samples and biases still remain closed. */
            if (!fetch && (g17p_explicit_lod_stage(shader->info.stage) ||
                           (m5 && m5_ta_stage(shader->info.stage))) &&
                tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
                tex->op != nir_texop_tg4)
               return g17p_decline("vertex_implicit_sample", tex, NULL,
                                   shader->info.stage);
            if (!fetch && !g17p_sample_shape(tex, m5, g17p))
               return g17p_decline("sample_shape", tex, NULL,
                                   shader->info.stage);
            nir_intrinsic_instr *image = nir_src_as_intrinsic(
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_handle)].src);
            if (!image || image->intrinsic != nir_intrinsic_bindless_image_agx)
               return g17p_decline("handle_not_bindless", tex, NULL,
                                   shader->info.stage);
            if (native32_set >= 0 &&
                (nir_intrinsic_desc_set(image) != native32_set ||
                 nir_src_num_components(image->src[0]) != 1 ||
                 nir_src_bit_size(image->src[0]) != 32 ||
                 (nir_src_is_const(image->src[0]) &&
                  (nir_src_as_uint(image->src[0]) & 31))))
               return g17p_decline("native32_texture_handle", tex, NULL,
                                   shader->info.stage);
            const int sampler_src =
               nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
            if (!fetch && native32_set >= 0 &&
                (sampler_src < 0 ||
                 nir_src_num_components(tex->src[sampler_src].src) != 1 ||
                 (nir_src_bit_size(tex->src[sampler_src].src) != 16 &&
                  nir_src_bit_size(tex->src[sampler_src].src) != 32 &&
                  nir_src_bit_size(tex->src[sampler_src].src) != 64) ||
                 nir_src_as_intrinsic(tex->src[sampler_src].src) ||
                 (nir_src_is_const(tex->src[sampler_src].src) &&
                  nir_src_as_uint(tex->src[sampler_src].src) > 16)))
               return g17p_decline("native32_sampler_index", tex, NULL,
                                   shader->info.stage);
            if (!fetch && native32_set < 0 && !g17p_sample_heap_index(tex))
               return g17p_decline("no_sampler_heap_index", tex, NULL,
                                   shader->info.stage);
            samples++;
         }
      }
   }
   if (!samples && !image_reads) {
      /* A shader with no texture instruction at all is the ordinary case and
       * says nothing; reporting it once per compile buries the shaders that
       * did have something to lower and did not get it. Report only when the
       * shader has texture instructions this pass declined to count. */
      if (agx_compiler_debug & AGX_DBG_SHADERS) {
         unsigned tex = 0;
         nir_foreach_function_impl(impl, shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block)
                  tex += instr->type == nir_instr_type_tex;
            }
         }
         if (tex) {
            fprintf(stderr,
                    "G17P indexed samplers: %u texture instructions in stage "
                    "%u, none admitted\n",
                    tex, (unsigned)shader->info.stage);
         }
      }
      return false;
   }

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_bindless_image_load &&
                   intr->intrinsic !=
                      nir_intrinsic_bindless_image_sparse_load)
                  continue;

               b.cursor = nir_before_instr(instr);
               nir_intrinsic_instr *image =
                  nir_src_as_intrinsic(intr->src[0]);
               assert(image &&
                      image->intrinsic == nir_intrinsic_bindless_image_agx);
               nir_def *set = nir_load_descriptor_set_agx(
                  &b, nir_intrinsic_desc_set(image));
               nir_def *offset = nir_iadd_imm(
                  &b, image->src[0].ssa,
                  AGX_G17P_STORAGE_HEAP_INDEX_OFFSET);
               nir_def *address =
                  nir_iadd(&b, set, nir_u2u64(&b, offset));
               nir_def *texture_index = nir_load_global_constant(
                  &b, 1, 32, address, .align_mul = 4,
                  .access = heap_index_access(image));
               texture_index = nir_u2u64(&b, texture_index);
               nir_def *native_handle = nir_bindless_image_agx(
                  &b, texture_index, .desc_set = 0);
               nir_src_rewrite(&intr->src[0], native_handle);
               continue;
            }
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            b.cursor = nir_before_instr(instr);
            nir_intrinsic_instr *image = nir_src_as_intrinsic(
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_handle)].src);
            const bool fetch = tex->op == nir_texop_txf ||
                               tex->op == nir_texop_txf_ms;
            nir_def *sampler = fetch ? NULL : native32_set >= 0
               ? tex->src[nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle)].src.ssa
               : g17p_sample_heap_index(tex);

            /* texture2d::read takes a uint level on AGX3.  The shared AGX2
             * texture lowering narrows every NIR LOD source to 16 bits, but
             * G17P's integer PACK_SMPIR form consumes a full 32-bit scalar.
             * Leaving the upper half unwritten makes the native sampler see
             * an indeterminate, usually out-of-range level and return zero.
             * Restore the API integer width before AIR is formed. */
            if (tex->op == nir_texop_txf) {
               const int lod =
                  nir_tex_instr_src_index(tex, nir_tex_src_lod);
               if (lod < 0) {
                  nir_tex_instr_add_src(tex, nir_tex_src_lod,
                                        nir_imm_intN_t(&b, 0, 32));
               } else {
                  nir_def *level = tex->src[lod].src.ssa;
                  nir_src_rewrite(&tex->src[lod].src,
                                  nir_src_is_const(tex->src[lod].src)
                                     ? nir_imm_intN_t(
                                          &b, nir_src_as_uint(tex->src[lod].src), 32)
                                     : nir_u2u32(&b, level));
               }
            }

            nir_def *texture_index;
            if (native32_set >= 0) {
               texture_index = nir_ushr_imm(&b, image->src[0].ssa, 5);
            } else {
               nir_def *set = nir_load_descriptor_set_agx(&b, nir_intrinsic_desc_set(image));
               /* Descriptor offsets are 32-bit and valid sets fit far below
                * 4GiB. Add the field offset before widening so native LDDEV can
                * consume base+byte-index without a separate 64-bit base add. */
               nir_def *offset = nir_iadd_imm(&b, image->src[0].ssa,
                                             AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET);
               nir_def *address = nir_iadd(&b, set, nir_u2u64(&b, offset));
               texture_index = nir_load_global_constant(
                  &b, 1, 32, address, .align_mul = 4,
                  .access = heap_index_access(image));
            }
            /* G17P Vulkan descriptors already name the native zero-based
             * sampler heap. The packed native32 path also uses raw indices.
             * Untouched/null descriptors select reserved entry zero. Preserve
             * native32's explicitly admitted wider sampler sources here. */
            texture_index = nir_u2u64(&b, texture_index);
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, texture_index);
            if (!fetch) {
               nir_def *sampler_index = nir_u2u64(
                  &b, native32_set >= 0 ? nir_u2u32(&b, sampler) : sampler);
               nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
               nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, sampler_index);
            }
            /* Fetch uses native sampler0, reserved in the same device heap
             * that ordinary samples use, so mixed reads share one prefix. */
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}

bool
agx_nir_lower_g17g_indexed_samplers(nir_shader *shader)
{
   if (!shader || shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            /* The shared lowering also supports storage-image fetches on
             * G17P. M5 currently only has the array-sample operand ABI. */
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_bindless_image_load)
               return false;
            if (instr->type != nir_instr_type_tex)
               continue;
            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            if (tex->op != nir_texop_txl ||
                tex->sampler_dim != GLSL_SAMPLER_DIM_2D || !tex->is_array ||
                tex->is_shadow || tex->dest_type != nir_type_float32 ||
                lod < 0 || !nir_src_is_const(tex->src[lod].src) ||
                nir_src_as_float(tex->src[lod].src) != 0.0)
               return false;
         }
      }
   }
   /* Gen10 publishes the same software heap-index tail and reserved sampler
    * prefix as Gen9. The native opcode/staging differences stay in the M5
    * encoder; this shared lowering only translates API handles to indices. */
   return agx_nir_lower_g17p_indexed_samplers(shader, false);
}

bool
agx_nir_lower_g17p_indexed_samplers(nir_shader *shader, bool g17p)
{
   return lower_indexed_samplers(shader, -1, false, g17p);
}


/* G17G multisample texel fetches: fragment, compute and vertex. The vertex
 * form is the fragment form with sampler index 0 (port/vmswit v_ms_f, v_ms_u,
 * v_msa_f, v_ms_k against f_ms_f). */
static bool
g17g_ms_read_stage(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_FRAGMENT || stage == MESA_SHADER_COMPUTE ||
          stage == MESA_SHADER_VERTEX;
}

bool
agx_nir_lower_g17g_indexed_reads(nir_shader *shader)
{
   if (!shader || (!g17p_explicit_lod_stage(shader->info.stage) &&
                   shader->info.stage != MESA_SHADER_FRAGMENT &&
                   shader->info.stage != MESA_SHADER_COMPUTE &&
                   !m5_ta_stage(shader->info.stage)))
      return false;

   unsigned reads = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                   intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {
                  if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS &&
                      !g17g_ms_read_stage(shader->info.stage))
                     return false;
                  if ((nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D &&
                       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_3D &&
                       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_CUBE &&
                       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS) ||
                      intr->def.bit_size != 32)
                     return false;
                  reads++;
               }
            }
            if (instr->type != nir_instr_type_tex)
               continue;
            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            const bool multisample = tex->sampler_dim == GLSL_SAMPLER_DIM_MS;
            const bool query_lod = tex->op == nir_texop_lod &&
               shader->info.stage == MESA_SHADER_FRAGMENT &&
               tex->dest_type == nir_type_float32 && tex->def.bit_size == 32 &&
               (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE);
            if (multisample && !g17g_ms_read_stage(shader->info.stage))
               return false;
            if ((!multisample && !query_lod &&
                 tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
                 tex->sampler_dim != GLSL_SAMPLER_DIM_3D &&
                 tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE))
               return false;

            /* C uses the common heap indices but its own SMP selectors and
             * staging. Ordinary 2D/array/3D/cube samples share the supported
             * LOD forms and interleaved gradient banks. Minimum LOD uses
             * typed publishers shared with the dimension's layer/face.
             * Runtime offsets retain their generation-specific requirements. */
            const int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
            const int packed_offset =
               nir_tex_instr_src_index(tex, nir_tex_src_backend2);
            const bool constant_offset =
               (offset < 0 || nir_src_is_const(tex->src[offset].src)) &&
               (packed_offset < 0 ||
                nir_src_is_const(tex->src[packed_offset].src));
            const bool ms_fetch = multisample &&
               g17p_indexed_fetch_shape(tex) && offset < 0 && packed_offset < 0;
            const bool fetch = ms_fetch || (!multisample &&
               tex->op == nir_texop_txf && lod >= 0 &&
               (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_3D) &&
               offset < 0 && packed_offset < 0 &&
               (tex->dest_type == nir_type_uint32 ||
                tex->dest_type == nir_type_float32 ||
                tex->dest_type == nir_type_int32));
            const bool minimum =
               nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 ||
               nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx) >= 0;
            /* Constant offsets are SMP immediates on C as on B: native G17G
             * (texwit3) carries {f43, f51, f141} on 2D, 2D-array, 3D,
             * comparison and gather samples alike. Cubes have no offset. */
            /* Runtime offsets: the generation-10 lowering keeps X/Y separate
             * (a component vector, or a 2x16 pair for a gather) on non-array
             * 2D samples without a register LOD/bias/minimum; the packer
             * publishes them with PACK_SMPIR into bank 6 (texwit3/4 d_*). */
            const bool runtime_offset = !constant_offset &&
               offset < 0 && packed_offset >= 0 && !tex->is_array &&
               tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !minimum &&
               nir_src_num_components(tex->src[packed_offset].src) ==
                  (tex->op == nir_texop_tg4 ? 1u : 2u) &&
               (lod < 0 || nir_src_is_const(tex->src[lod].src)) &&
               (nir_tex_instr_src_index(tex, nir_tex_src_bias) < 0 ||
                nir_src_is_const(tex->src[nir_tex_instr_src_index(
                   tex, nir_tex_src_bias)].src));
            /* A 2D or 2D-array sample whose runtime offset shares bank 6
             * with its layer or register LOD/bias arrives as G17P's complete
             * 32-bit word (one component), staged by COPY; a comparison's
             * reference rides in front of it (texwit6 c_arr, c_lvlr,
             * g_cmp_arr). */
            const bool packed_runtime_offset = !constant_offset &&
               offset < 0 && packed_offset >= 0 &&
               (tex->op != nir_texop_tg4 || tex->is_array) &&
               (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                (tex->sampler_dim == GLSL_SAMPLER_DIM_3D &&
                 nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx) >= 0)) &&
               nir_src_num_components(tex->src[packed_offset].src) == 1;
            const bool sample = !multisample &&
               (constant_offset || runtime_offset || packed_runtime_offset) &&
               (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE ||
                (offset < 0 && packed_offset < 0)) &&
               (tex->dest_type == nir_type_float32 ||
                tex->dest_type == nir_type_uint32 ||
                tex->dest_type == nir_type_int32) &&
               (tex->op == nir_texop_tex || tex->op == nir_texop_txl ||
                tex->op == nir_texop_txb || tex->op == nir_texop_tg4 ||
                (tex->op == nir_texop_txd &&
                 tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE));
            if (!fetch && !sample && !query_lod)
               return false;
            reads++;
         }
      }
   }
   return reads > 0 && lower_indexed_samplers(shader, -1, true, false);
}

bool
agx_nir_lower_g17p_native32_samplers(nir_shader *shader, uint8_t texture_set)
{
   return lower_indexed_samplers(shader, texture_set, false, false);
}

static bool
g17p_txf_shape(const nir_tex_instr *tex)
{
   if (tex->op != nir_texop_txf ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_shadow ||
       tex->is_sparse || (tex->coord_components != 2 &&
                          tex->coord_components != 3) ||
       tex->is_array != (tex->coord_components == 3) ||
       (tex->backend_flags & ~AGX_TEXTURE_FLAG_NO_CLAMP) != 0 ||
       tex->dest_type != nir_type_uint32 || tex->def.num_components != 4 ||
       tex->def.bit_size != 32 || tex->texture_index != 0 ||
       tex->sampler_index != 0)
      return false;

   const int handle =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);

   if (handle < 0 || lod < 0 || coord < 0 || tex->num_srcs != 3 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) >= 0 ||
       !nir_src_is_const(tex->src[lod].src) ||
       nir_src_as_uint(tex->src[lod].src) != 0)
      return false;

   return nir_src_num_components(tex->src[coord].src) ==
             tex->coord_components &&
          nir_src_bit_size(tex->src[coord].src) == 32;
}

static bool
g17p_has_bindless_texture_access(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_bindless_image_agx ||
                op == nir_intrinsic_bindless_sampler_agx)
               return true;
         }
      }
   }

   return false;
}

/* A robustness bounds check lowers imageSize() into a descriptor crawl before
 * storage-image specialization runs. The load/store handles are removed by
 * the specialization, but that read-only crawl and image atomics still need
 * their bindless handles for address calculation. Admit only the exact paired
 * texture/PBE descriptors and those two consumers; any independently indexed
 * image or sampler access still fails closed. */
static bool
g17p_has_unsupported_storage_descriptor_access(const nir_shader *shader,
                                                uint32_t texture_offset_B,
                                                uint32_t pbe_offset_B)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx)
               return true;
            if (intr->intrinsic != nir_intrinsic_bindless_image_agx)
               continue;

            if (nir_intrinsic_desc_set(intr) != 0 ||
                !nir_src_is_const(intr->src[0]))
               return true;

            const uint32_t offset_B = nir_src_as_uint(intr->src[0]);
            if (offset_B != texture_offset_B && offset_B != pbe_offset_B)
               return true;

            nir_foreach_use(use, &intr->def) {
               nir_instr *parent = nir_src_use_instr(use);
               if (parent->type != nir_instr_type_intrinsic)
                  return true;

               nir_intrinsic_instr *user = nir_instr_as_intrinsic(parent);
               const bool descriptor_address =
                  user->intrinsic ==
                     nir_intrinsic_load_from_texture_handle_agx &&
                  user->def.num_components == 1 && user->def.bit_size == 64;
               const bool image_atomic =
                  offset_B == pbe_offset_B &&
                  (user->intrinsic == nir_intrinsic_bindless_image_atomic ||
                   user->intrinsic ==
                      nir_intrinsic_bindless_image_atomic_swap);
               if (!descriptor_address && !image_atomic)
                  return true;
            }
         }
      }
   }

   return false;
}

bool
agx_nir_lower_g17p_txf(nir_shader *shader,
                        struct agx_g17p_txf_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_tex_instr *fetch = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            if (fetch || !g17p_txf_shape(nir_instr_as_tex(instr)))
               return false;
            fetch = nir_instr_as_tex(instr);
         }
      }
   }

   if (!fetch)
      return false;

   uint32_t texture_offset_B;
   if (!g17p_handle_offset(fetch, nir_tex_src_texture_handle,
                           nir_intrinsic_bindless_image_agx,
                           &texture_offset_B) ||
       (texture_offset_B & 7) != 0)
      return false;

   nir_steal_tex_src(fetch, nir_tex_src_texture_handle);
   fetch->texture_index = 0;

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_txf_binding){
      .texture_offset_B = texture_offset_B,
   };
   return true;
}

static bool
g17p_storage_handle_binding(const nir_intrinsic_instr *intr, uint8_t *set,
                            uint32_t *offset_B)
{
   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
       nir_intrinsic_desc_set(handle) > UINT8_MAX ||
       !nir_src_is_const(handle->src[0]))
      return false;

   *set = nir_intrinsic_desc_set(handle);
   *offset_B = nir_src_as_uint(handle->src[0]);
   return true;
}

static bool
g17p_storage_handle_offset(const nir_intrinsic_instr *intr,
                           uint32_t *offset_B)
{
   uint8_t set;
   return g17p_storage_handle_binding(intr, &set, offset_B) && set == 0;
}

bool
agx_nir_lower_g17p_indexed_storage(
   nir_shader *shader, struct agx_g17p_indexed_storage_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_intrinsic_instr *store = NULL;
   bool has_buffer_fetch = false;
   uint8_t pbe_set = 0;
   uint32_t pbe_offset_B = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               /* Buffer textures are lowered to a 2D txf whose coordinate is
                * carried in backend1; ordinary 2D texelFetch keeps coord. */
               has_buffer_fetch |=
                  tex->op == nir_texop_txf &&
                  nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               const enum pipe_format format = nir_intrinsic_format(intr);
               const bool r32 = format == PIPE_FORMAT_R32_UINT;
               if (store ||
                   nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D ||
                   nir_intrinsic_image_array(intr) ||
                   (format != PIPE_FORMAT_R8G8B8A8_UNORM &&
                    format != PIPE_FORMAT_R10G10B10A2_UNORM && !r32) ||
                   nir_src_num_components(intr->src[1]) != 4 ||
                   nir_src_bit_size(intr->src[1]) != 32 ||
                   (nir_src_num_components(intr->src[3]) != 4 &&
                    !(r32 && nir_src_num_components(intr->src[3]) == 1)) ||
                   nir_src_bit_size(intr->src[3]) != 32 ||
                   !nir_src_is_const(intr->src[2]) ||
                   nir_src_as_uint(intr->src[2]) != 0 ||
                   !nir_src_is_const(intr->src[4]) ||
                   nir_src_as_uint(intr->src[4]) != 0 ||
                   (nir_intrinsic_access(intr) &
                    (ACCESS_COHERENT | ACCESS_VOLATILE)) ||
                   !g17p_storage_handle_binding(intr, &pbe_set,
                                                 &pbe_offset_B) ||
                   (pbe_offset_B & 7) != 0)
                  return false;
               store = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                       intr->intrinsic ==
                          nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic ==
                          nir_intrinsic_bindless_image_atomic_swap ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   if (!store)
      return false;

   /* The indexed texture path occupies the first native state pair with the
    * texture/sampler heap roots.  A normalized immediate-state write after a
    * lowered texel-buffer fetch currently resolves state 1 as empty on G17P,
    * producing an all-zero destination.  Keep the working bindless PBE form
    * for this mixed shape; integer state 1 and straight-line sampled-storage
    * retain their independently witnessed specialized paths. */
   if (has_buffer_fetch &&
       (nir_intrinsic_format(store) == PIPE_FORMAT_R8G8B8A8_UNORM ||
        nir_intrinsic_format(store) == PIPE_FORMAT_R10G10B10A2_UNORM))
      return false;

   nir_builder b = nir_builder_at(nir_before_instr(&store->instr));
   nir_src_rewrite(&store->src[0], nir_imm_intN_t(&b, 1, 16));
   store->intrinsic = nir_intrinsic_image_store;
   nir_progress(true, nir_shader_get_entrypoint(shader),
                nir_metadata_control_flow);
   NIR_PASS(_, shader, nir_opt_dce);

   *binding = (struct agx_g17p_indexed_storage_binding){
      .pbe_offset_B = pbe_offset_B,
      .pbe_set = pbe_set,
   };
   return true;
}

/* any_format (G18G): the write PBE converts from its own descriptor format,
 * so a store of any colour format may take the state-0 route
 * (agx_nir_lower_g18g_storage_image). */
static bool
g17p_storage_image_shape(const nir_intrinsic_instr *intr, bool load,
                         bool any_format)
{
   const enum pipe_format format = nir_intrinsic_format(intr);
   const bool format_ok = format == PIPE_FORMAT_R8G8B8A8_UINT ||
                          format == PIPE_FORMAT_R32_UINT ||
                          (any_format && format != PIPE_FORMAT_NONE);
   /* G18G (any_format): the texture/PBE descriptors carry the dimension,
    * so 2D arrays, cubes and 3D take the same state-0 route. */
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool g18_dim_ok =
      any_format &&
      ((dim == GLSL_SAMPLER_DIM_2D) || (dim == GLSL_SAMPLER_DIM_CUBE) ||
       (dim == GLSL_SAMPLER_DIM_3D && !nir_intrinsic_image_array(intr)));
   const bool plain_2d =
      dim == GLSL_SAMPLER_DIM_2D && !nir_intrinsic_image_array(intr);
   if (intr->intrinsic != (load ? nir_intrinsic_bindless_image_load
                                : nir_intrinsic_bindless_image_store) ||
       (nir_intrinsic_access(intr) & ACCESS_BUFFER_AGX) ||
       (!plain_2d && !g18_dim_ok) || !format_ok ||
       nir_src_num_components(intr->src[1]) != 4 ||
       nir_src_bit_size(intr->src[1]) != 32 ||
       !nir_src_is_const(intr->src[2]) ||
       nir_src_as_uint(intr->src[2]) != 0)
      return false;

   if (load) {
      return intr->def.num_components == 4 && intr->def.bit_size == 32 &&
             /* G18G also loads 2D arrays, cubes and 3D images through
              * the table: the texture descriptor carries the dimension. */
             (plain_2d || g18_dim_ok) &&
             nir_src_is_const(intr->src[3]) &&
             nir_src_as_uint(intr->src[3]) == 0 &&
             !(nir_intrinsic_access(intr) &
               (ACCESS_COHERENT | ACCESS_VOLATILE));
   }

   return nir_src_num_components(intr->src[3]) == 4 &&
          nir_src_bit_size(intr->src[3]) == 32 &&
          nir_src_is_const(intr->src[4]) &&
          nir_src_as_uint(intr->src[4]) == 0 &&
          !(nir_intrinsic_access(intr) & ACCESS_VOLATILE);
}

static bool
g17p_storage_atomic_shape(const nir_intrinsic_instr *intr,
                          uint32_t pbe_offset_B)
{
   const bool atomic =
      intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
      intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
   const enum pipe_format format = nir_intrinsic_format(intr);

   uint32_t atomic_pbe_offset_B;
   return atomic && nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_2D &&
          !nir_intrinsic_image_array(intr) &&
          (format == PIPE_FORMAT_R32_UINT ||
           format == PIPE_FORMAT_R32_SINT) &&
          nir_src_num_components(intr->src[1]) == 4 &&
          nir_src_bit_size(intr->src[1]) == 32 &&
          nir_src_is_const(intr->src[2]) &&
          nir_src_as_uint(intr->src[2]) == 0 &&
          !(nir_intrinsic_access(intr) & ACCESS_VOLATILE) &&
          g17p_storage_handle_offset(intr, &atomic_pbe_offset_B) &&
          atomic_pbe_offset_B == pbe_offset_B;
}

static bool
g17p_lower_storage_image(nir_shader *shader,
                         struct agx_g17p_storage_image_binding *binding,
                         bool any_format)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_intrinsic_instr *load = NULL, *store = NULL;
   unsigned atomics = 0;
   uint32_t texture_offset_B = 0, pbe_offset_B = 0;
   unsigned impl_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            /* The single-image state-0 route predates the general indexed
             * storage ABI. It is correct for an isolated image roundtrip, but
             * a shader memory barrier does not order its native state-0 image
             * accesses at all -- the pairing of the barrier with buffer or
             * global memory was never what mattered.
             *
             * Measured on G17P with dEQP-VK.memory_model.write_after_read:
             * the image + buffer-guard variants pass only because the check
             * below already routed them to the indexed lowering, and forcing
             * those same 80 passing cases back onto the state-0 route fails
             * all 80, deterministically, with nothing else changed. The
             * image + workgroup-guard variants name only image and shared
             * memory, so they stayed on the state-0 route and fail.
             *
             * The barrier instructions the two routes emit are identical, so
             * WHY the native state-0 pair is unordered is not understood.
             * Only that it is. Decline any barrier that names the image and
             * leave ordered images to the indexed lowering. */
            if (intr->intrinsic == nir_intrinsic_barrier) {
               const nir_variable_mode modes =
                  nir_intrinsic_memory_modes(intr);
               if (modes & nir_var_image)
                  return false;
            }
            if (intr->intrinsic == nir_intrinsic_bindless_image_load) {
               if (load || !g17p_storage_image_shape(intr, true, any_format) ||
                   !g17p_storage_handle_offset(intr, &texture_offset_B))
                  return false;
               load = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               if (store ||
                   !g17p_storage_image_shape(intr, false, any_format) ||
                   !g17p_storage_handle_offset(intr, &pbe_offset_B))
                  return false;
               store = intr;
            } else if (intr->intrinsic ==
                          nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic ==
                          nir_intrinsic_bindless_image_atomic_swap) {
               uint32_t atomic_pbe_offset_B;
               if (!g17p_storage_handle_offset(intr,
                                                &atomic_pbe_offset_B))
                  return false;
               if (atomics++ == 0)
                  pbe_offset_B = atomic_pbe_offset_B;
               else if (pbe_offset_B != atomic_pbe_offset_B)
                  return false;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   /* A Gen9 storage descriptor is texture32 followed by PBE32. A shader with
    * only PBE stores still needs the same native state-0 pair; derive the
    * unused texture address from the PBE handle instead of leaving imgwci
    * state0 unbound. Atomics keep the PBE handle live because they lower to a
    * raw texel address, while the paired imageStore uses native state 0. */
   if (impl_count != 1 || !store || (pbe_offset_B & 7) != 0)
      return false;

   uint32_t store_pbe_offset_B;
   if (!g17p_storage_handle_offset(store, &store_pbe_offset_B) ||
       store_pbe_offset_B != pbe_offset_B)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                 intr->intrinsic ==
                    nir_intrinsic_bindless_image_atomic_swap) &&
                !g17p_storage_atomic_shape(intr, pbe_offset_B))
               return false;
         }
      }
   }

   /* The texture record sits one 32-byte record before the PBE (asserted in
    * hk_descriptor_set.h). */
   if (load) {
      if ((texture_offset_B & 7) != 0 ||
          pbe_offset_B != texture_offset_B + 32)
         return false;
   } else {
      if (pbe_offset_B < 32)
         return false;
      texture_offset_B = pbe_offset_B - 32;
   }

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if ((!load || instr != &load->instr) && instr != &store->instr)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0],
                            nir_imm_intN_t(&b, 0, 16));
            intr->intrinsic = load && intr == load
                                 ? nir_intrinsic_image_load
                                 : nir_intrinsic_image_store;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_unsupported_storage_descriptor_access(
          shader, texture_offset_B, pbe_offset_B))
      return false;

   *binding = (struct agx_g17p_storage_image_binding){
      .texture_offset_B = texture_offset_B,
      .pbe_offset_B = pbe_offset_B,
      .write_only = load == NULL && atomics == 0,
   };
   return true;
}

bool
agx_nir_lower_g17p_storage_image(nir_shader *shader,
                                 struct agx_g17p_storage_image_binding *binding)
{
   return g17p_lower_storage_image(shader, binding, false);
}

bool
agx_nir_lower_g17p_gallium_storage_write(nir_shader *shader,
                                          unsigned pbe_state)
{
   if (!shader || shader->info.stage != MESA_SHADER_COMPUTE ||
       BITSET_LAST_BIT(shader->info.textures_used) != 0 ||
       BITSET_LAST_BIT(shader->info.images_used) != 1)
      return false;

   nir_intrinsic_instr *store = NULL;
   unsigned impl_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_store: {
               const enum pipe_format format = nir_intrinsic_format(intr);
               const unsigned data_components =
                  nir_src_num_components(intr->src[3]);
               if (store ||
                   (format != PIPE_FORMAT_R32_UINT &&
                    format != PIPE_FORMAT_R32_SINT) ||
                   nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D ||
                   nir_intrinsic_image_array(intr) ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) != pbe_state ||
                   nir_src_num_components(intr->src[1]) != 4 ||
                   nir_src_bit_size(intr->src[1]) != 32 ||
                   (data_components != 1 && data_components != 4) ||
                   nir_src_bit_size(intr->src[3]) != 32 ||
                   !nir_src_is_const(intr->src[2]) ||
                   nir_src_as_uint(intr->src[2]) != 0 ||
                   !nir_src_is_const(intr->src[4]) ||
                   nir_src_as_uint(intr->src[4]) != 0 ||
                   (nir_intrinsic_access(intr) &
                    (ACCESS_COHERENT | ACCESS_VOLATILE)))
                  return false;
               store = intr;
               break;
            }
            case nir_intrinsic_image_load:
            case nir_intrinsic_image_sparse_load:
            case nir_intrinsic_image_atomic:
            case nir_intrinsic_image_atomic_swap:
            case nir_intrinsic_image_size:
            case nir_intrinsic_image_samples:
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_bindless_image_store:
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_bindless_image_atomic_swap:
            case nir_intrinsic_bindless_image_size:
            case nir_intrinsic_bindless_image_samples:
               return false;
            default:
               break;
            }
         }
      }
   }

   if (impl_count != 1 || !store)
      return false;

   nir_builder b = nir_builder_at(nir_before_instr(&store->instr));
   nir_src_rewrite(&store->src[0], nir_imm_intN_t(&b, 0, 16));
   nir_progress(true, nir_shader_get_entrypoint(shader),
                nir_metadata_control_flow);
   NIR_PASS(_, shader, nir_opt_dce);
   return true;
}

static bool
g17p_sampled_storage_store_shape(const nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_bindless_image_store &&
          nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_2D &&
          !nir_intrinsic_image_array(intr) &&
          nir_intrinsic_format(intr) == PIPE_FORMAT_R8G8B8A8_UNORM &&
          nir_src_num_components(intr->src[1]) == 4 &&
          nir_src_bit_size(intr->src[1]) == 32 &&
          nir_src_num_components(intr->src[3]) == 4 &&
          nir_src_bit_size(intr->src[3]) == 32 &&
          nir_src_is_const(intr->src[2]) &&
          nir_src_as_uint(intr->src[2]) == 0 &&
          nir_src_is_const(intr->src[4]) &&
          nir_src_as_uint(intr->src[4]) == 0 &&
          !(nir_intrinsic_access(intr) &
            (ACCESS_COHERENT | ACCESS_VOLATILE));
}

bool
agx_nir_lower_g17p_sampled_storage(
   nir_shader *shader, struct agx_g17p_sampled_storage_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   nir_tex_instr *sample = NULL;
   nir_intrinsic_instr *store = NULL;
   unsigned impl_count = 0, block_count = 0;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         block_count++;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               if (sample ||
                   !g17p_sample_shape(nir_instr_as_tex(instr), false, false))
                  return false;
               sample = nir_instr_as_tex(instr);
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
               if (store || !g17p_sampled_storage_store_shape(intr))
                  return false;
               store = intr;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic ||
                       intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap ||
                       intr->intrinsic == nir_intrinsic_bindless_image_size ||
                       intr->intrinsic == nir_intrinsic_bindless_image_samples) {
               return false;
            }
         }
      }
   }

   /* This protocol is derived from Apple's straight-line kernel with one
    * explicit level-zero sample feeding one rgba8 write.  Keep all control
    * flow and independent data transforms outside the qualified shape. */
   if (impl_count != 1 || block_count != 1 || !sample || !store ||
       store->src[3].ssa != &sample->def || sample->op != nir_texop_txl)
      return false;

   const int lod = nir_tex_instr_src_index(sample, nir_tex_src_lod);
   if (lod < 0 || !nir_src_is_const(sample->src[lod].src) ||
       nir_src_as_uint(sample->src[lod].src) != 0)
      return false;

   uint8_t texture_set, sampler_set, pbe_set;
   uint32_t texture_offset_B, sampler_state_offset_B, storage_offset_B;
   if (!g17p_handle_binding(sample, nir_tex_src_texture_handle,
                            nir_intrinsic_bindless_image_agx, &texture_set,
                            &texture_offset_B) ||
       !g17p_handle_binding(sample, nir_tex_src_sampler_handle,
                            nir_intrinsic_bindless_sampler_agx, &sampler_set,
                            &sampler_state_offset_B) ||
       !g17p_storage_handle_binding(store, &pbe_set, &storage_offset_B) ||
       texture_set != sampler_set || (texture_offset_B & 7) != 0 ||
       sampler_state_offset_B != texture_offset_B + 32 ||
       !g17p_is_sampler_bias(sample, texture_set,
                             sampler_state_offset_B + 12) ||
       (storage_offset_B & 7) != 0)
      return false;

   nir_steal_tex_src(sample, nir_tex_src_texture_handle);
   nir_steal_tex_src(sample, nir_tex_src_sampler_handle);
   if (sample->op == nir_texop_txb) {
      nir_steal_tex_src(sample, nir_tex_src_bias);
      sample->op = nir_texop_tex;
   }
   sample->texture_index = 0;
   sample->sampler_index = 0;

   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(shader));
   b.cursor = nir_before_instr(&store->instr);
   nir_src_rewrite(&store->src[0], nir_imm_intN_t(&b, 1, 16));
   store->intrinsic = nir_intrinsic_image_store;

   NIR_PASS(_, shader, nir_opt_dce);
   if (g17p_has_bindless_texture_access(shader))
      return false;

   *binding = (struct agx_g17p_sampled_storage_binding){
      .texture_offset_B = texture_offset_B,
      .sampler_index_offset_B = sampler_state_offset_B + 8,
      .pbe_offset_B = storage_offset_B,
      .texture_set = texture_set,
      .pbe_set = pbe_set,
   };
   return true;
}

bool
agx_nir_lower_g18g_storage_image(struct nir_shader *shader,
                                 struct agx_g17p_storage_image_binding *binding)
{
   const bool progress = g17p_lower_storage_image(shader, binding, true);
   if (!progress || binding->write_only)
      return progress;

   /* Read-write: Gallium's G18G form. One table holds the
    * image's texture descriptor (entry 0, read by image_load) and its PBE
    * (entry 1, written by image_store); the hk storage descriptor already
    * keeps them adjacent (PBE = texture + 32). Point the store at entry 1. */
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_image_store)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0], nir_imm_intN_t(&b, 1, 16));
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   return true;
}

/* G18G, several storage images: one native state table spanning their
 * descriptors. hk stores a storage image as texture32 then PBE32 in its
 * descriptor set, and the G18G state-0 route reads its table straight from
 * set memory (entry 0 = the bound address). With the table based at the
 * lowest texture descriptor, image i loads through entry (texture_i - base) /
 * 32 and stores through the entry after it, the read-write form
 * agx_nir_lower_g18g_storage_image already uses for one image. Loads and
 * stores only (hk lowers image atomics to global atomics before this pass),
 * every image in one set; anything else declines. Runs only after the
 * single-image route declined. */
#define G18G_MAX_STORAGE_IMAGES 8

static bool
g18g_storage_image_key(const nir_intrinsic_instr *intr, bool load,
                       uint32_t *key, uint8_t *set)
{
   uint32_t offset_B;
   const bool shape = g17p_storage_image_shape(intr, load, true);
   /* One table, so every image must be in one set (any set). */
   if (!shape || !g17p_storage_handle_binding(intr, set, &offset_B) ||
       *set >= 8 || (offset_B & 31) != 0)
      return false;
   if (!load) {
      if (offset_B < 32)
         return false;
      offset_B -= 32;
   }
   *key = offset_B;
   return true;
}

bool
agx_nir_lower_g18g_storage_images(
   struct nir_shader *shader, struct agx_g17p_storage_image_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   uint32_t keys[G18G_MAX_STORAGE_IMAGES];
   unsigned key_count = 0, impl_count = 0;
   int table_set = -1;
   bool ok = true;
   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               ok = false;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_barrier:
               /* As the single-image route: state-0 image accesses are not
                * ordered by a barrier that names image memory. */
               if (nir_intrinsic_memory_modes(intr) & nir_var_image)
                  ok = false;
               break;
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_store: {
               uint32_t key;
               uint8_t set;
               if (!g18g_storage_image_key(
                      intr,
                      intr->intrinsic == nir_intrinsic_bindless_image_load,
                      &key, &set) ||
                   (table_set >= 0 && table_set != set)) {
                  ok = false;
                  break;
               }
               table_set = set;
               unsigned k;
               for (k = 0; k < key_count && keys[k] != key; ++k)
                  ;
               if (k == key_count) {
                  if (key_count == G18G_MAX_STORAGE_IMAGES)
                     ok = false;
                  else
                     keys[key_count++] = key;
               }
               break;
            }
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_bindless_image_atomic_swap:
               /* hk lowers image atomics to global atomics before this pass;
                * one still here is not a shape this table route knows. */
               ok = false;
               break;
            case nir_intrinsic_bindless_image_size:
            case nir_intrinsic_bindless_image_texel_address: {
               /* The bounds check and texel
                * address of a lowered atomic read the image's descriptor
                * words directly; they need no table entry, only the set. */
               uint8_t set;
               uint32_t offset_B;
               if (!g17p_storage_handle_binding(intr, &set, &offset_B) ||
                   (table_set >= 0 && table_set != set))
                  ok = false;
               else
                  table_set = set;
               break;
            }
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_bindless_image_samples:
            case nir_intrinsic_bindless_sampler_agx:
               ok = false;
               break;
            default:
               break;
            }
         }
      }
   }
   /* Runs only after the single-image route declined, so one image is
    * fine here too: in another set, or in set 0 beside descriptor reads of
    * other images (lowered atomics) that the single route refuses. */
   if (!ok || impl_count != 1 || table_set < 0 || key_count < 1)
      return false;

   uint32_t base = keys[0], top = keys[0];
   for (unsigned k = 1; k < key_count; ++k) {
      base = MIN2(base, keys[k]);
      top = MAX2(top, keys[k]);
   }
   /* Keys are 32-aligned descriptor offsets: equal keys are one image. A
    * key of one image can not fall on another image's PBE (descriptors do
    * not overlap), so entries never alias. Bound the table size. */
   if (top - base > 32 * 1024)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool load =
               intr->intrinsic == nir_intrinsic_bindless_image_load;
            if (!load && intr->intrinsic != nir_intrinsic_bindless_image_store)
               continue;
            uint32_t key;
            uint8_t set;
            if (!g18g_storage_image_key(intr, load, &key, &set) ||
                set != table_set)
               return false;
            const unsigned entry = (key - base) / 32 + (load ? 0 : 1);
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0], nir_imm_intN_t(&b, entry, 16));
            intr->intrinsic =
               load ? nir_intrinsic_image_load : nir_intrinsic_image_store;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   NIR_PASS(_, shader, nir_opt_dce);

   /* Every descriptor access must now be gone, except reads of a table
    * image's own descriptor words (the array layer clamp of the robust
    * image_load lowering), which address the table's set directly. */
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx)
               return false;
            if (intr->intrinsic != nir_intrinsic_bindless_image_agx)
               continue;
            if (nir_intrinsic_desc_set(intr) != (unsigned)table_set ||
                !nir_src_is_const(intr->src[0]))
               return false;
            nir_foreach_use(use, &intr->def) {
               nir_instr *parent = nir_src_use_instr(use);
               if (parent->type != nir_instr_type_intrinsic)
                  return false;
               nir_intrinsic_instr *user = nir_instr_as_intrinsic(parent);
               const bool descriptor_address =
                  user->intrinsic ==
                     nir_intrinsic_load_from_texture_handle_agx &&
                  user->def.num_components == 1 && user->def.bit_size == 64;
               const bool address_use =
                  user->intrinsic == nir_intrinsic_bindless_image_size ||
                  user->intrinsic == nir_intrinsic_bindless_image_texel_address;
               if (!descriptor_address && !address_use)
                  return false;
            }
         }
      }
   }

   *binding = (struct agx_g17p_storage_image_binding){
      .texture_offset_B = base,
      .pbe_offset_B = base + 32,
      .write_only = false,
      .table_size_B = top - base + 64,
      .set = (uint8_t)table_set,
   };
   return true;
}

/* G18G, sampled textures and storage images in one compute shader: per-dispatch
 * native tables. G18G has one texture-state table (storage images read their
 * texture at entry k and write through their PBE at k + 1, the read-write form
 * above) and one sampler table (the image write uses sampler entry 0, the
 * G18G write-state record). Here both tables are built at dispatch from the
 * bound descriptor sets: entries 2i / 2i + 1 are storage image i's texture and
 * PBE, the sampled textures follow, and sampler entries 1.. are the shader's
 * samplers (copied from the sampler heap; entry 0 stays the write state).
 * Every texture, sampler and image access becomes an immediate table index.
 * Constant descriptor offsets only; anything else declines. */
static int
g18g_table_find(const struct agx_g18g_table_entry *entries, unsigned count,
                uint8_t set, uint32_t offset_B)
{
   for (unsigned i = 0; i < count; ++i) {
      if (entries[i].set == set && entries[i].offset_B == offset_B)
         return (int)i;
   }
   return -1;
}

static bool
g18g_const_handle(nir_src *src, nir_intrinsic_op op, uint8_t *set,
                  uint32_t *offset_B)
{
   nir_intrinsic_instr *h = nir_src_as_intrinsic(*src);
   if (!h || h->intrinsic != op || nir_intrinsic_desc_set(h) >= 8 ||
       !nir_src_is_const(h->src[0]))
      return false;
   *set = nir_intrinsic_desc_set(h);
   *offset_B = nir_src_as_uint(h->src[0]);
   return true;
}

static bool
g18g_table_add(struct agx_g18g_table_entry *entries, unsigned *count,
               unsigned max, uint8_t set, uint32_t offset_B)
{
   if (g18g_table_find(entries, *count, set, offset_B) >= 0)
      return true;
   if (*count == max)
      return false;
   entries[(*count)++] = (struct agx_g18g_table_entry){set, offset_B};
   return true;
}

bool
agx_nir_lower_g18g_combined_tables(struct nir_shader *shader,
                                   struct agx_g18g_table_binding *binding)
{
   if (!shader || !binding || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   struct agx_g18g_table_entry images[AGX_G18G_TABLE_MAX_TEXTURES / 2];
   struct agx_g18g_table_entry textures[AGX_G18G_TABLE_MAX_TEXTURES];
   struct agx_g18g_table_entry samplers[AGX_G18G_TABLE_MAX_SAMPLERS - 1];
   unsigned image_count = 0, texture_count = 0, sampler_count = 0;
   unsigned impl_count = 0, image_ops = 0;
   bool ok = true;

   nir_foreach_function_impl(impl, shader) {
      impl_count++;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               uint8_t set;
               uint32_t offset_B;
               const int ti =
                  nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
               const int si =
                  nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
               if (ti < 0 ||
                   nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) >=
                      0 ||
                   nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >=
                      0 ||
                   !g18g_const_handle(&tex->src[ti].src,
                                      nir_intrinsic_bindless_image_agx, &set,
                                      &offset_B) ||
                   (offset_B & 7) != 0 ||
                   !g18g_table_add(textures, &texture_count,
                                   AGX_G18G_TABLE_MAX_TEXTURES, set,
                                   offset_B)) {
                  ok = false;
                  continue;
               }
               if (si >= 0 &&
                   (!g18g_const_handle(&tex->src[si].src,
                                       nir_intrinsic_bindless_sampler_agx, &set,
                                       &offset_B) ||
                    !g18g_table_add(samplers, &sampler_count,
                                    AGX_G18G_TABLE_MAX_SAMPLERS - 1, set,
                                    offset_B)))
                  ok = false;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_barrier:
               if (nir_intrinsic_memory_modes(intr) & nir_var_image)
                  ok = false;
               break;
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_store: {
               uint32_t key;
               uint8_t set;
               if (!g18g_storage_image_key(
                      intr,
                      intr->intrinsic == nir_intrinsic_bindless_image_load,
                      &key, &set) ||
                   !g18g_table_add(images, &image_count,
                                   AGX_G18G_TABLE_MAX_TEXTURES / 2, set, key))
                  ok = false;
               image_ops++;
               break;
            }
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_bindless_image_atomic_swap:
            case nir_intrinsic_bindless_image_samples:
               ok = false;
               break;
            default:
               break;
            }
         }
      }
   }
   /* Also storage images alone, when the set-0/one-set routes declined
    * (images spread over several sets): each entry names its own set. */
   if (!ok || impl_count != 1 || !image_ops ||
       2 * image_count + texture_count > AGX_G18G_TABLE_MAX_TEXTURES)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               uint8_t set;
               uint32_t offset_B;
               const int ti =
                  nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
               if (!g18g_const_handle(&tex->src[ti].src,
                                      nir_intrinsic_bindless_image_agx, &set,
                                      &offset_B))
                  return false;
               const int t =
                  g18g_table_find(textures, texture_count, set, offset_B);
               unsigned sampler_entry = 0;
               const int si =
                  nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
               if (si >= 0) {
                  if (!g18g_const_handle(&tex->src[si].src,
                                         nir_intrinsic_bindless_sampler_agx,
                                         &set, &offset_B))
                     return false;
                  sampler_entry = 1 + g18g_table_find(samplers, sampler_count,
                                                      set, offset_B);
               }
               nir_steal_tex_src(tex, nir_tex_src_texture_handle);
               if (si >= 0)
                  nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
               tex->texture_index = 2 * image_count + t;
               tex->sampler_index = sampler_entry;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool load =
               intr->intrinsic == nir_intrinsic_bindless_image_load;
            if (!load && intr->intrinsic != nir_intrinsic_bindless_image_store)
               continue;
            uint32_t key;
            uint8_t set;
            if (!g18g_storage_image_key(intr, load, &key, &set))
               return false;
            const int i = g18g_table_find(images, image_count, set, key);
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0],
                            nir_imm_intN_t(&b, 2 * i + (load ? 0 : 1), 16));
            intr->intrinsic =
               load ? nir_intrinsic_image_load : nir_intrinsic_image_store;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   NIR_PASS(_, shader, nir_opt_dce);

   /* What is left of the handles may only read descriptor words directly. */
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >=
                      0 ||
                   nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >=
                      0)
                  return false;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx)
               return false;
            if (intr->intrinsic != nir_intrinsic_bindless_image_agx)
               continue;
            nir_foreach_use(use, &intr->def) {
               nir_instr *parent = nir_src_use_instr(use);
               if (parent->type != nir_instr_type_intrinsic)
                  return false;
               nir_intrinsic_instr *user = nir_instr_as_intrinsic(parent);
               if (!(user->intrinsic ==
                        nir_intrinsic_load_from_texture_handle_agx &&
                     user->def.num_components == 1 &&
                     user->def.bit_size == 64) &&
                   user->intrinsic != nir_intrinsic_bindless_image_size &&
                   user->intrinsic !=
                      nir_intrinsic_bindless_image_texel_address)
                  return false;
            }
         }
      }
   }

   struct agx_g18g_table_binding out = {0};
   for (unsigned i = 0; i < image_count; ++i) {
      out.textures[out.texture_count++] = images[i];
      out.textures[out.texture_count++] =
         (struct agx_g18g_table_entry){images[i].set, images[i].offset_B + 32};
   }
   for (unsigned i = 0; i < texture_count; ++i)
      out.textures[out.texture_count++] = textures[i];
   /* The handle names the descriptor's sampler word; the published heap
    * index sits 8 bytes after it (hk_sampled_image_descriptor_layout). */
   for (unsigned i = 0; i < sampler_count; ++i)
      out.samplers[out.sampler_count++] = (struct agx_g18g_table_entry){
         samplers[i].set, samplers[i].offset_B + 8};
   *binding = out;
   return true;
}
