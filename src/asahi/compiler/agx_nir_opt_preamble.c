/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2021 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "agx_compiler.h"
#include "util/bitset.h"
#include "util/set.h"
#include "util/u_dynarray.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_opcodes.h"

static nir_preamble_class
preamble_class(nir_def *def)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic_or_null(def);
   if (!intr || (nir_intrinsic_has_desc_set(intr) &&
                 nir_intrinsic_desc_set(intr) >= 32 /* encoding restriction */))
      return nir_preamble_class_general;

   if (intr->intrinsic == nir_intrinsic_bindless_image_agx)
      return nir_preamble_class_image;
   else if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx)
      return nir_preamble_class_sampler;
   else
      return nir_preamble_class_general;
}

static void
def_size(nir_def *def, unsigned *size, unsigned *align,
         nir_preamble_class *class)
{
   unsigned bit_size = MAX2(def->bit_size, 16);

   *size = (bit_size * def->num_components) / 16;
   *align = bit_size / 16;
   *class = preamble_class(def);
}

static void
def_size_g17p(nir_def *def, unsigned *size, unsigned *align,
              nir_preamble_class *class)
{
   def_size(def, size, align, class);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic_or_null(def);
   if (!intr || intr->intrinsic != nir_intrinsic_bindless_image_agx ||
       intr->src[0].ssa->bit_size != 64)
      return;

   /* G17P cannot export texture state, so its native image descriptor is
    * carried in general uniforms. The opaque token is 32 bits; reserve the
    * 64-bit payload that lower_store_preamble actually exports instead.
    * Image-class slot units and the other encodings stay unchanged.
    */
   *class = nir_preamble_class_general;
   *size = *align = 4;
}

/* The AGX3C constant-program export is a 32-bit uniform COPY: a lone half
 * is published as one pair-aligned copy whose upper half is padding (Apple
 * widens scalar half preamble results the same way), and no half-width
 * copy-to-uniform form is witnessed. Give a scalar 16-bit value its own
 * aligned pair so the export never lands on an odd half. */
/* A half result takes a whole word, see widen_half_exports(). */
static void
def_size_agx3b(nir_def *def, unsigned *size, unsigned *align,
               nir_preamble_class *class)
{
   def_size_g17p(def, size, align, class);
   if (*class == nir_preamble_class_general && def->num_components == 1 &&
       def->bit_size <= 16) {
      *size = 2;
      *align = 2;
   }
}

static void
def_size_agx3c(nir_def *def, unsigned *size, unsigned *align,
               nir_preamble_class *class)
{
   def_size_g17p(def, size, align, class);
   if (*class == nir_preamble_class_general && def->num_components == 1 &&
       def->bit_size <= 16) {
      *size = 2;
      *align = 2;
   }
}

static bool
all_uses_float(nir_def *def)
{
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use))
         return false;

      nir_instr *use_instr = nir_src_use_instr(use);
      if (use_instr->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *use_alu = nir_instr_as_alu(use_instr);
      unsigned src_index = ~0;
      for (unsigned i = 0; i < nir_op_infos[use_alu->op].num_inputs; i++) {
         if (&use_alu->src[i].src == use) {
            src_index = i;
            break;
         }
      }

      assert(src_index != ~0);
      nir_alu_type src_type = nir_alu_type_get_base_type(
         nir_op_infos[use_alu->op].input_types[src_index]);

      if (src_type != nir_type_float)
         return false;

      /* No float modifiers on G13 */
      if (use_alu->op == nir_op_fmax || use_alu->op == nir_op_fmin)
         return false;
   }

   return true;
}

static float
alu_cost(nir_alu_instr *alu)
{
   /* TODO: Model 64-bit better */
   if (alu->def.bit_size == 64)
      return 10.0f;

   switch (alu->op) {
   case nir_op_fsat:
   case nir_op_f2fmp:
   case nir_op_f2f16:
   case nir_op_f2f16_rtne:
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_iadd:
   case nir_op_inot:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_feq:
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_fneu:
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_ilt:
   case nir_op_ige:
   case nir_op_ult:
   case nir_op_uge:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_isub:
   case nir_op_ineg:
   case nir_op_bcsel:
   case nir_op_b2b1:
   case nir_op_b2b32:
   case nir_op_b2i8:
   case nir_op_b2i16:
   case nir_op_b2i32:
   case nir_op_b2f16:
   case nir_op_b2f32:
   case nir_op_i2i32:
   case nir_op_i2i16:
   case nir_op_u2u32:
   case nir_op_u2u16:
   case nir_op_u2u8:
   case nir_op_i2i8:
   case nir_op_iadd_sat:
   case nir_op_isub_sat:
   case nir_op_uadd_sat:
   case nir_op_usub_sat:
   case nir_op_iabs:
      /* SCIB */
      return 1.0;

   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_bit_count:
   case nir_op_bitfield_reverse:
   case nir_op_ufind_msb:
   case nir_op_imul:
   case nir_op_imadshl_agx:
   case nir_op_imsubshl_agx:
   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr:
   case nir_op_flog2:
   case nir_op_fexp2:
   case nir_op_extr_agx:
   case nir_op_ubitfield_extract:
   case nir_op_f2i8:
   case nir_op_f2i16:
   case nir_op_f2i32:
   case nir_op_f2u8:
   case nir_op_f2u16:
   case nir_op_f2u32:
   case nir_op_i2fmp:
   case nir_op_i2f16:
   case nir_op_i2f32:
   case nir_op_u2fmp:
   case nir_op_u2f16:
   case nir_op_u2f32:
   case nir_op_interleave:
      /* IC */
      return 4.0;

   case nir_op_frcp:
      /* IC */
      return 6.0;

   case nir_op_frsq:
      /* IC */
      return 8.0;

   case nir_op_fsqrt:
      /* IC + F32 */
      return 8.5;

   case nir_op_imul_high:
   case nir_op_umul_high:
   case nir_op_imul_2x32_64:
   case nir_op_umul_2x32_64:
      /* IC */
      return 8.0;

   case nir_op_fsin_agx:
      /* 2 IC + 1 F32 in parallel */
      return 8.5;

   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_f2f32:
      /* Float source modifiers will be propagated */
      return all_uses_float(&alu->def) ? 0.0 : 1.0;

   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_pack_32_2x16_split:
   case nir_op_pack_64_2x32_split:
   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y:
   case nir_op_unpack_32_2x16_split_x:
   case nir_op_unpack_32_2x16_split_y:
   case nir_op_extract_i8:
   case nir_op_extract_u8:
   case nir_op_extract_i16:
   case nir_op_extract_u16:
      /* We optimistically assume that moves get coalesced */
      return 0.0;

   default:
      /* Shrug */
      return 2.0;
   }
}

static float
instr_cost(nir_instr *instr, const void *data)
{
   switch (instr->type) {
   case nir_instr_type_intrinsic:
      switch (nir_instr_as_intrinsic(instr)->intrinsic) {
      case nir_intrinsic_load_global:
      case nir_intrinsic_load_agx:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_constant_agx:
      case nir_intrinsic_load_ubo:
         return 10.0;
      case nir_intrinsic_ddx:
      case nir_intrinsic_ddx_fine:
      case nir_intrinsic_ddx_coarse:
      case nir_intrinsic_ddy:
      case nir_intrinsic_ddy_fine:
      case nir_intrinsic_ddy_coarse:
         return 1.0;
      case nir_intrinsic_bindless_image_agx:
      case nir_intrinsic_bindless_sampler_agx:
         /* It's worth promoting even with a constant source, but it doesn't
          * turn into instructions so should be less than any other normal
          * instruction... But just enough to get over the image rewrite_cost.
          */
         return 2.5;
      default:
         /* Assume it's a sysval or something */
         return 0.0;
      }

   case nir_instr_type_tex:
      /* Texturing involes lots of memory bandwidth */
      return 20.0;

   case nir_instr_type_alu:
      return alu_cost(nir_instr_as_alu(instr));

   default:
      return 1.0;
   }
}

/* Uniform slots run out in big shaders. Code under an if may never run, so
 * divide its claim on them per level: otherwise the many small loads of a
 * cold branch outbid the vectors every fragment reads, and the shader spills
 * to hold those. The rewrite cost scales the same way, so this reorders the
 * candidates without dropping any while slots are free.
 *
 * data is the set of ifs that stay branches (surviving_branches), NULL for no
 * weighting. An if that the later nir_opt_peephole_select turns into selects
 * runs for every invocation that reaches it, so it does not count. A loop
 * leaves the weight as is. */
static bool
is_branch(const struct set *branches, nir_cf_node *node)
{
   return node && _mesa_set_search(branches, node);
}

static float
execution_weight(nir_block *block, const struct set *branches)
{
   float weight = 1.0f;
   if (!branches)
      return weight;
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (is_branch(branches, node))
         weight /= 8.0f;
   }
   return weight;
}

static void
collect_ifs(struct exec_list *list, struct util_dynarray *ifs)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         util_dynarray_append(ifs, nif);
         collect_ifs(&nif->then_list, ifs);
         collect_ifs(&nif->else_list, ifs);
      } else if (node->type == nir_cf_node_loop) {
         collect_ifs(&nir_cf_node_as_loop(node)->body, ifs);
      }
   }
}

/* The ifs of the main program that are still branches once flatten, the
 * nir_opt_peephole_select that runs after the constant program is formed,
 * has seen it. Found by forming an unweighted constant program on a clone and
 * flattening that, so the rule is the select pass's own. Without flatten,
 * every if is a branch. */
static struct set *
surviving_branches(nir_shader *nir, const nir_opt_preamble_options *options,
                   const unsigned *sizes,
                   const nir_opt_peephole_select_options *flatten)
{
   struct set *branches = _mesa_pointer_set_create(NULL);
   struct util_dynarray ifs, clone_ifs, left;
   util_dynarray_init(&ifs, NULL);
   util_dynarray_init(&clone_ifs, NULL);
   util_dynarray_init(&left, NULL);
   collect_ifs(&nir_shader_get_entrypoint(nir)->body, &ifs);

   if (flatten && ifs.size) {
      unsigned used[nir_preamble_num_classes];
      memcpy(used, sizes, sizeof(used));

      nir_shader *clone = nir_shader_clone(NULL, nir);
      collect_ifs(&nir_shader_get_entrypoint(clone)->body, &clone_ifs);
      nir_opt_preamble(clone, options, used);
      nir_opt_peephole_select(clone, flatten);
      collect_ifs(&nir_shader_get_entrypoint(clone)->body, &left);

      /* Both walks see the same ifs in the same order, and the select pass
       * creates no ifs. */
      struct set *standing = _mesa_pointer_set_create(NULL);
      util_dynarray_foreach(&left, nir_if *, nif)
         _mesa_set_add(standing, *nif);
      for (unsigned i = 0; i < util_dynarray_num_elements(&ifs, nir_if *);
           ++i) {
         if (_mesa_set_search(standing, *util_dynarray_element(
                                           &clone_ifs, nir_if *, i)))
            _mesa_set_add(branches,
                          *util_dynarray_element(&ifs, nir_if *, i));
      }
      _mesa_set_destroy(standing, NULL);
      ralloc_free(clone);
   } else {
      util_dynarray_foreach(&ifs, nir_if *, nif)
         _mesa_set_add(branches, *nif);
   }

   util_dynarray_fini(&ifs);
   util_dynarray_fini(&clone_ifs);
   util_dynarray_fini(&left);
   return branches;
}

static float
instr_cost_g17p(nir_instr *instr, const void *data)
{
   float weight = execution_weight(instr->block, data);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_fabs && alu->def.bit_size <= 32)
         return weight;
   }
   return instr_cost(instr, data) * weight;
}

/* On AGX3C, native uniform-operand float instructions never fold fabs(u), and
 * only fold fneg(u) on fadd and the addend (src 2) of ffma. Give fabs(u) and
 * non-foldable fneg(u) non-zero cost so loop-invariant modified uniforms are
 * precomputed once in the constant program instead of copied and negated per
 * thread or per loop iteration. */
static bool
agx3c_fneg_all_uses_fold_uniform(const nir_def *def)
{
   bool any = false;
   nir_foreach_use_including_if(use, def) {
      any = true;
      if (nir_src_is_if(use))
         return false;
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op == nir_op_fadd)
         continue;
      if (alu->op == nir_op_ffma && use == &alu->src[2].src)
         continue;
      return false;
   }
   return any;
}

static float
instr_cost_agx3c(nir_instr *instr, const void *data)
{
   float weight = execution_weight(instr->block, data);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->def.bit_size <= 32) {
         if (alu->op == nir_op_fabs)
            return weight;
         if (alu->op == nir_op_fneg &&
             !agx3c_fneg_all_uses_fold_uniform(&alu->def))
            return weight;
      }
   }
   return instr_cost(instr, data) * weight;
}

/* G17P's vertex output store reads its 16-bit index from the uniform file
 * (agx_instr_accepts_uniform), so an index computed in the preamble -- an
 * unlinked vertex shader's per-draw slot plus the component -- costs the
 * main program nothing. */
static bool
g17p_uniform_uvs_index(const nir_src *use)
{
   nir_instr *parent = nir_src_use_instr(use);
   if (parent->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);
   return intr->intrinsic == nir_intrinsic_store_uvs_agx &&
          use == &intr->src[1] && use->ssa->bit_size == 16 &&
          use->ssa->num_components == 1;
}

static float
rewrite_cost_impl(nir_def *def, bool g17p)
{
   bool mov_needed = false, vectorizable = true;
   nir_foreach_use(use, def) {
      nir_instr *parent_instr = nir_src_use_instr(use);
      if (g17p && g17p_uniform_uvs_index(use)) {
         continue;
      } else if (parent_instr->type == nir_instr_type_tex) {
         /* TODO: Maybe check the source index, but biases can be uniform */
         break;
      } else if (parent_instr->type == nir_instr_type_phi) {
         /* Assume we'd eat a move anyway */
      } else if (parent_instr->type != nir_instr_type_alu) {
         mov_needed = true;
         vectorizable = false;
         break;
      } else {
         nir_alu_instr *alu = nir_instr_as_alu(parent_instr);
         if (alu->op == nir_op_vec2 || alu->op == nir_op_vec3 ||
             alu->op == nir_op_vec4) {
            mov_needed = true;
            break;
         } else if (alu->op == nir_op_mov) {
            mov_needed = true;
            vectorizable = false;
         } else {
            /* Assume for non-moves that the const is folded into the src */
         }
      }
   }

   return mov_needed ? ((float)(def->num_components * def->bit_size) /
                        (vectorizable ? 32.0 : 16.0))
                     : 0;
}

static float
rewrite_cost(nir_def *def, const void *data)
{
   return rewrite_cost_impl(def, false);
}

static float
rewrite_cost_g17p(nir_def *def, const void *data)
{
   /* A phi is worth what its arms are, one if deeper than it sits. */
   nir_block *block = nir_def_block(def);
   float weight = execution_weight(block, data);
   if (data && nir_def_instr(def)->type == nir_instr_type_phi &&
       is_branch(data, nir_cf_node_prev(&block->cf_node)))
      weight /= 8.0f;

   return rewrite_cost_impl(def, true) * weight;
}

/* AGX3C publishes constant-program results with 32-bit uniform copies only
 * (agx3_pack_binary), so a narrow vector has no export form there: keep it
 * out of the preamble and let its scalar or 32-bit uses be hoisted instead. */
static float
rewrite_cost_agx3c(nir_def *def, const void *data)
{
   if (def->bit_size < 32 && def->num_components > 1)
      return 1e30f;
   return rewrite_cost_impl(def, true);
}

static bool
avoid_instr(const nir_instr *instr, const void *data)
{
   return false;
}

static const nir_opt_preamble_options preamble_options = {
   .drawid_uniform = true,
   .subgroup_size_uniform = true,
   /* not supported in hardware */
   .load_workgroup_size_allowed = false,
   .def_size = def_size,
   .instr_cost_cb = instr_cost,
   .rewrite_cost_cb = rewrite_cost,
   .avoid_instr_cb = avoid_instr,

   /* Custom-border emulation and DXVK's per-sample descriptor reads repeat
    * one uniform expression at every guarded sample site; give them one slot
    * so the storage is left for the remaining descriptor work.
    */
   .share_equal_values = true,

   /* hardware size is 512, but it's polite to leave some wiggle room to push
    * hot constants so we don't end up rematerializing all over the place.
    * 480 seems to be a sweetspot, based on a few minutes of shader-db.
    */
   .preamble_storage_size[nir_preamble_class_general] = 480,

   /* We have at least 32 texture state registers. TODO: check for more? */
   .preamble_storage_size[nir_preamble_class_image] = 32,

   /* We have at least 16 sampler state registers. TODO: check for more? */
   .preamble_storage_size[nir_preamble_class_sampler] = 16,
};

struct preamble_image {
   int16_t heap;
   uint8_t bit_size;
};

/*
 * Bindless image handles can't be stored to uniforms, so we move them back to
 * the main shader. Effectively un-optimizing the preamble.
 */
static bool
lower_store_preamble(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct preamble_image *images = data;
   if (intr->intrinsic != nir_intrinsic_store_preamble ||
       nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;

   images[nir_intrinsic_base(intr)] = (struct preamble_image) {
      .heap = nir_intrinsic_desc_set(handle),
      .bit_size = handle->src[0].ssa->bit_size,
   };
   nir_src_rewrite(&intr->src[0], handle->src[0].ssa);
   return true;
}

static bool
lower_preamble(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
      /* Rematerialize bindless_sampler_agx before store_preamble with only the
       * byte offset (first source), not the sampler index.
       */
      nir_foreach_use_safe(use, &intr->def) {
         nir_instr *parent = nir_src_use_instr(use);
         if (parent->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *pintr = nir_instr_as_intrinsic(parent);
         if (pintr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(pintr) != nir_preamble_class_sampler)
            continue;

         b->cursor = nir_before_src(use);
         nir_def *repl =
            nir_bindless_sampler_agx(b, intr->src[0].ssa, nir_undef(b, 1, 16),
                                     .desc_set = nir_intrinsic_desc_set(intr));
         nir_src_rewrite(use, repl);
      }

      /* Replace other uses with just the sampler index. */
      nir_def_replace(&intr->def, intr->src[1].ssa);
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   struct preamble_image *images = data;
   b->cursor = nir_after_instr(&intr->instr);

   unsigned base = nir_intrinsic_base(intr);
   nir_def *new_ = NULL;
   bool ts = nir_intrinsic_preamble_class(intr) == nir_preamble_class_image;
   bool ss = nir_intrinsic_preamble_class(intr) == nir_preamble_class_sampler;
   if (nir_intrinsic_preamble_class(intr) == nir_preamble_class_general &&
       images[base].heap >= 0) {
      /* Reload the exported descriptor payload, then recreate the opaque
       * token for its users. The token's own width is not the payload width.
       */
      intr->def.bit_size = images[base].bit_size;
      new_ = nir_bindless_image_agx(b, &intr->def,
                                   .desc_set = images[base].heap);
   }

   nir_foreach_use_safe(use, &intr->def) {
      nir_instr *parent = nir_src_use_instr(use);

      if (parent->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *pintr = nir_instr_as_intrinsic(parent);

         if (ts) {
            nir_rewrite_image_intrinsic(pintr, nir_imm_intN_t(b, base / 2, 16),
                                        nir_image_intrinsic_type_default);
         } else if (new_ != NULL &&
                    pintr->intrinsic != nir_intrinsic_bindless_image_agx) {
            nir_src_rewrite(use, new_);
         }
      } else if (parent->type == nir_instr_type_tex) {
         nir_tex_instr *tex = nir_instr_as_tex(parent);
         nir_tex_src *src = (nir_tex_src *)use;

         if (src->src_type == nir_tex_src_sampler_handle && ss) {
            nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
            tex->sampler_index = base;
         } else if (src->src_type == nir_tex_src_texture_handle && ts) {
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            tex->texture_index = base / 2;
         } else if (src->src_type == nir_tex_src_texture_handle) {
            assert(new_ != NULL);
            nir_src_rewrite(use, new_);
         }
      }
   }

   return true;
}

/*
 * The flag a preamble store exports: the 1-bit value itself, or, once
 * widen_half_exports() has run, the bool of its word bcsel(flag, 1, 0), which
 * sets *word.
 */
static nir_scalar
flag_store_value(nir_intrinsic_instr *store, bool *word)
{
   nir_def *value = store->src[0].ssa;
   *word = false;
   if (value->num_components != 1)
      return (nir_scalar){NULL, 0};
   if (value->bit_size == 1)
      return nir_get_scalar(value, 0);

   nir_alu_instr *sel = nir_def_as_alu_or_null(value);
   if (value->bit_size == 32 && sel && sel->op == nir_op_bcsel &&
       nir_src_bit_size(sel->src[0].src) == 1 &&
       /* ALU sources are swizzled: a constant source can be a vector of
        * which the select reads one component. */
       nir_src_is_const(sel->src[1].src) &&
       nir_src_comp_as_uint(sel->src[1].src, sel->src[1].swizzle[0]) == 1 &&
       nir_src_is_const(sel->src[2].src) &&
       nir_src_comp_as_uint(sel->src[2].src, sel->src[2].swizzle[0]) == 0) {
      *word = true;
      return nir_get_scalar(sel->src[0].src.ssa, sel->src[0].swizzle[0]);
   }
   return (nir_scalar){NULL, 0};
}

/*
 * A preamble boolean that the main program only selects on is exported as the
 * half 1.0 or 0.0 and compared against zero where it is used. This is Apple's
 * G17P shape for a select on a constant-program flag: the constant program
 * converts the flag once, and every select reads the uniform directly as its
 * comparand (agx3_alu_fcmp_select, f42 = 3), instead of copying the 0/1 half
 * into a register for each integer select.
 *
 * With @branches, a flag the main program also branches on is exported the
 * same way when its store is already a word with zeroes on top
 * (widen_half_exports(), G17P): the word is nonzero exactly when the flag is
 * set, and a branch compares that aligned word against zero in place, as
 * Apple's loop guard does (agx_lower_uniform_sources.c, uniform_imm_compare),
 * instead of copying the half into a register for each branch. Called again
 * after nir_opt_peephole_select, which turns most early flag branches into
 * selects, so each use takes the form it ends up with.
 */
bool
agx_nir_export_select_flags_as_halves(nir_shader *nir, bool branches)
{
   nir_function_impl *main = nir_shader_get_entrypoint(nir);
   if (!main->preamble)
      return false;
   nir_function_impl *pre = nir_shader_get_preamble(nir);

   BITSET_DECLARE(flag, 512);
   BITSET_DECLARE(other, 512);
   BITSET_DECLARE(branch, 512);
   BITSET_DECLARE(word, 512);
   BITSET_ZERO(flag);
   BITSET_ZERO(other);
   BITSET_ZERO(branch);
   BITSET_ZERO(word);

   nir_foreach_block(block, main) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
            continue;
         const unsigned base = nir_intrinsic_base(intr);
         const unsigned halves =
            MAX2(intr->def.bit_size, 16) * intr->def.num_components / 16;
         if (base + halves > 512)
            return false;

         bool selects_only = intr->def.bit_size == 1 &&
                             intr->def.num_components == 1;
         nir_foreach_use_including_if(use, &intr->def) {
            if (!selects_only)
               break;
            if (nir_src_is_if(use)) {
               selects_only = branches;
               BITSET_SET(branch, base);
               continue;
            }
            nir_instr *parent = nir_src_use_instr(use);
            selects_only =
               parent->type == nir_instr_type_alu &&
               nir_instr_as_alu(parent)->op == nir_op_bcsel &&
               use == &nir_instr_as_alu(parent)->src[0].src;
         }
         if (selects_only)
            BITSET_SET(flag, base);
         else
            BITSET_SET_RANGE(other, base, base + halves - 1);
      }
   }

   nir_foreach_block(block, pre) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
            continue;
         const unsigned base = nir_intrinsic_base(intr);
         bool is_word;
         if (base >= 512)
            continue;
         if (!flag_store_value(intr, &is_word).def)
            BITSET_SET(other, base);
         else if (is_word)
            BITSET_SET(word, base);
      }
   }

   BITSET_ANDNOT(flag, flag, other);
   /* A branch reads the whole word: only a word with zeroes on top. */
   BITSET_ANDNOT(branch, branch, word);
   BITSET_ANDNOT(flag, flag, branch);
   if (BITSET_IS_EMPTY(flag))
      return false;

   nir_foreach_block(block, pre) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general ||
             nir_intrinsic_base(intr) >= 512 ||
             !BITSET_TEST(flag, nir_intrinsic_base(intr)))
            continue;
         bool is_word;
         nir_scalar value = flag_store_value(intr, &is_word);
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_def *half = nir_bcsel(&b, nir_mov_scalar(&b, value),
                                   nir_imm_float16(&b, 1.0),
                                   nir_imm_float16(&b, 0.0));
         nir_src_rewrite(&intr->src[0], is_word ? nir_u2u32(&b, half) : half);
      }
   }

   nir_foreach_block(block, main) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general ||
             !BITSET_TEST(flag, nir_intrinsic_base(intr)) ||
             intr->def.bit_size != 1 /* or the word added below */)
            continue;
         nir_builder b = nir_builder_at(nir_after_instr(instr));
         nir_def *word_value = NULL;
         nir_foreach_use_including_if_safe(use, &intr->def) {
            if (!nir_src_is_if(use))
               continue;
            if (!word_value) {
               word_value = nir_load_preamble(&b, 1, 32,
                                              .base = nir_intrinsic_base(intr));
            }
            b.cursor = nir_before_src(use);
            nir_src_rewrite(use, nir_ine_imm(&b, word_value, 0));
         }
         b.cursor = nir_after_instr(instr);
         intr->def.bit_size = 16;
         nir_def *set = nir_fneu_imm(&b, &intr->def, 0.0);
         nir_def_rewrite_uses_after(&intr->def, set);
      }
   }

   nir_progress(true, pre, nir_metadata_control_flow);
   nir_progress(true, main, nir_metadata_control_flow);
   return true;
}

/*
 * A constant read of a buffer under non-uniform control flow would serve the
 * constant program as well as any other, but hoisting it makes the read happen
 * for fragments that never take the branch. Without soft faults that needs
 * care: a binding may be smaller than the block, as long as the branch that
 * reads past it never runs, and reading past the end of the allocation faults.
 *
 * The descriptor says how much is bound, so read the zero page instead when the
 * load would leave the binding. That costs nothing on a path that really runs,
 * and it is added only to the loads that do move: a load left in the main
 * program stays under its branch, as it was.
 */
static nir_intrinsic_instr *
speculative_descriptor(const nir_intrinsic_instr *load, unsigned *end)
{
   if ((load->intrinsic != nir_intrinsic_load_agx &&
        load->intrinsic != nir_intrinsic_load_constant_agx) ||
       !(nir_intrinsic_access(load) & ACCESS_CAN_REORDER) ||
       !nir_src_is_const(load->src[1]) || load->def.bit_size != 32 ||
       nir_intrinsic_format(load) != PIPE_FORMAT_R32_UINT)
      return NULL;

   nir_alu_instr *pack = nir_def_as_alu_or_null(load->src[0].ssa);
   if (!pack || pack->op != nir_op_pack_64_2x32_split ||
       pack->src[0].src.ssa != pack->src[1].src.ssa ||
       pack->src[0].swizzle[0] != 0 || pack->src[1].swizzle[0] != 1)
      return NULL;

   nir_intrinsic_instr *desc = nir_def_as_intrinsic_or_null(pack->src[0].src.ssa);
   if (!desc || desc->intrinsic != nir_intrinsic_load_constant_agx ||
       desc->def.num_components != 4 ||
       !(nir_intrinsic_access(desc) & ACCESS_BUFFER_DESCRIPTOR_AGX))
      return NULL;

   uint64_t index = nir_src_as_uint(load->src[1]);
   uint64_t bytes = (index * 4 << nir_intrinsic_base(load)) +
                    load->def.num_components * 4;
   *end = bytes;
   return bytes <= AGX_ZERO_PAGE_SIZE ? desc : NULL;
}

/* A load that moves takes its pointer with it, so one that then finds no slot
 * leaves the main program rebuilding that pointer from the descriptor. Allow
 * only what fits in the storage left free by everything that moves anyway.
 */
static bool
allow_speculative_load(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   unsigned *free_halves = data;
   const unsigned halves = load->def.num_components * 2;
   unsigned end;
   if (load->instr.block->cf_node.parent->type == nir_cf_node_function ||
       !speculative_descriptor(load, &end) || halves > *free_halves)
      return false;

   *free_halves -= halves;
   nir_intrinsic_set_access(
      load, nir_intrinsic_access(load) | ACCESS_CAN_SPECULATE);
   return true;
}

static unsigned
free_preamble_halves(const nir_shader *nir,
                     const nir_opt_preamble_options *options,
                     const unsigned *sizes)
{
   unsigned used[nir_preamble_num_classes];
   memcpy(used, sizes, sizeof(used));

   nir_shader *clone = nir_shader_clone(NULL, nir);
   nir_opt_preamble(clone, options, used);
   ralloc_free(clone);

   const unsigned total =
      options->preamble_storage_size[nir_preamble_class_general];
   return used[nir_preamble_class_general] < total
             ? total - used[nir_preamble_class_general]
             : 0;
}

/* What stayed in the main program must not be flattened past its branch now. */
static bool
disallow_speculative_load(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   if ((load->intrinsic != nir_intrinsic_load_agx &&
        load->intrinsic != nir_intrinsic_load_constant_agx) ||
       !(nir_intrinsic_access(load) & ACCESS_CAN_REORDER))
      return false;

   nir_intrinsic_set_access(
      load, nir_intrinsic_access(load) & ~ACCESS_CAN_SPECULATE);
   return true;
}

static bool
guard_speculative_load(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   unsigned end;
   nir_intrinsic_instr *desc = speculative_descriptor(load, &end);
   if (!desc || !(nir_intrinsic_access(load) & ACCESS_CAN_SPECULATE))
      return false;

   b->cursor = nir_before_instr(&load->instr);
   nir_def *bound = nir_channel(b, &desc->def, 2);
   nir_src_rewrite(&load->src[0],
                   nir_bcsel(b, nir_uge_imm(b, bound, end), load->src[0].ssa,
                             nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS)));
   return true;
}

/* A G17P constant program writes uniforms a word at a time, from an aligned
 * register pair. A half exported from the top of a pair, or into the top of a
 * word, cannot be packed, and that refusal costs the shader its whole constant
 * program. So export every half result as a word with zeroes on top, in the
 * slot def_size_agx3b() reserved; the main program still reads the low half.
 */
static bool
widen_half_exports(nir_shader *nir)
{
   nir_function_impl *pre = nir_shader_get_preamble(nir);
   bool progress = false;

   nir_foreach_block(block, pre) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
            continue;
         nir_def *value = intr->src[0].ssa;
         if (value->num_components != 1 || value->bit_size > 16)
            continue;
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_src_rewrite(&intr->src[0],
                         value->bit_size == 1
                            ? nir_bcsel(&b, value, nir_imm_int(&b, 1),
                                        nir_imm_int(&b, 0))
                            : nir_u2u32(&b, value));
         progress = true;
      }
   }

   nir_foreach_function_impl(impl, nir)
      nir_progress(impl == pre && progress, impl, nir_metadata_control_flow);
   return progress;
}

bool
agx_nir_opt_preamble(nir_shader *nir, unsigned *sizes, bool g17p,
                     bool agx3c,
                     const nir_opt_peephole_select_options *flatten)
{
   bool progress = false;
   nir_opt_preamble_options options = preamble_options;
   /* Native M5 compiler specimens leave texture operations in main even
    * for uniform coordinates. Keep their dependent expressions there too;
    * invariant ALU and buffer loads remain eligible for the CProg. */
   options.no_textures = agx3c;
   if (g17p) {
      /* The G17P uniform fabs cost is not an M5 witness; keep the
       * generic instruction cost there. */
      options.def_size = agx3c ? def_size_agx3c : def_size_agx3b;
      options.instr_cost_cb = agx3c ? instr_cost_agx3c : instr_cost_g17p;
      options.rewrite_cost_cb = agx3c ? rewrite_cost_agx3c : rewrite_cost_g17p;
   }
   /* Every AGX3B chip: the uniform file it fills is the encoding's. */
   struct set *branches = NULL;
   if (g17p && !agx3c) {
      branches = surviving_branches(nir, &options, sizes, flatten);
      options.cb_data = branches;
   }
   bool speculative = false;
   if (g17p && !agx3c && nir->info.stage == MESA_SHADER_FRAGMENT) {
      unsigned free_halves = free_preamble_halves(nir, &options, sizes);
      NIR_PASS(speculative, nir, nir_shader_intrinsics_pass,
               allow_speculative_load, nir_metadata_all, &free_halves);
   }

   NIR_PASS(progress, nir, nir_opt_preamble, &options, sizes);

   if (speculative) {
      nir_function_intrinsics_pass(nir_shader_get_entrypoint(nir),
                                   disallow_speculative_load, nir_metadata_all,
                                   NULL);
      if (progress) {
         nir_function_intrinsics_pass(nir_shader_get_preamble(nir),
                                      guard_speculative_load,
                                      nir_metadata_control_flow, NULL);
      }
   }

   if (progress && g17p)
      NIR_PASS(_, nir, agx_nir_export_select_flags_as_halves, false);
   if (progress && g17p && !agx3c)
      NIR_PASS(_, nir, widen_half_exports);

   struct preamble_image heap[512];
   memset(heap, ~0, sizeof(heap));

   if (progress) {
      nir_function_intrinsics_pass(nir_shader_get_preamble(nir),
                                   lower_store_preamble,
                                   nir_metadata_control_flow, heap);
   }

   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_preamble,
            nir_metadata_control_flow, heap);

   _mesa_set_destroy(branches, NULL);
   return progress;
}

/* General preamble exports are read through load_preamble, or implicitly as
 * the base of a bindless image/sampler token. Image/sampler-class exports use
 * separate native state and are deliberately left alone.
 */
bool
agx_nir_remove_unused_preamble_exports(nir_shader *nir, unsigned reserved)
{
   if (!nir_shader_get_entrypoint(nir)->preamble)
      return false;

   bool read[512] = {false};
   for (unsigned h = 0; h < MIN2(reserved, ARRAY_SIZE(read)); h++)
      read[h] = true;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned base, size;
            if (intr->intrinsic == nir_intrinsic_load_preamble &&
                nir_intrinsic_preamble_class(intr) == nir_preamble_class_general) {
               base = nir_intrinsic_base(intr);
               size = MAX2(intr->def.bit_size, 16) * intr->def.num_components / 16;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
               base = nir_intrinsic_desc_set(intr);
               size = 4;
            } else {
               continue;
            }
            if (base >= ARRAY_SIZE(read) || size > ARRAY_SIZE(read) - base)
               return false;
            for (unsigned h = 0; h < size; h++)
               read[base + h] = true;
         }
      }
   }

   bool progress = false;
   nir_function_impl *impl = nir_shader_get_preamble(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
            continue;
         unsigned base = nir_intrinsic_base(intr);
         nir_def *value = intr->src[0].ssa;
         unsigned size = MAX2(value->bit_size, 16) * value->num_components / 16;
         if (base >= ARRAY_SIZE(read) || size > ARRAY_SIZE(read) - base)
            continue;
         bool live = false;
         for (unsigned h = 0; h < size; h++)
            live |= read[base + h];
         if (!live) {
            nir_instr_remove(instr);
            progress = true;
         }
      }
   }
   nir_foreach_function_impl(other, nir) {
      nir_progress(other == impl && progress, other, nir_metadata_control_flow);
   }
   return progress;
}

/* Native CProgs export a whole uniform word. When its two live halves are
 * assigned in different control-flow blocks, defer only that pair's exports
 * until the preamble finishes. Locals lowered to SSA preserve each path's
 * last assignment, with undef on paths which never write a half. The pack
 * preserves the defined neighbor even on such a path; do not initialize the
 * unwritten half or change any main-program load or uniform allocation.
 */
bool
agx_nir_coalesce_preamble_half_exports(nir_shader *nir, unsigned reserved,
                                     unsigned allocated)
{
   if (!nir_shader_get_entrypoint(nir)->preamble)
      return false;
   nir_function_impl *impl = nir_shader_get_preamble(nir);

   if (reserved >= allocated || allocated > 512)
      return false;

   bool written[512] = {false};
   bool blocked[256] = {false};
   nir_block *first_block[256] = {NULL};
   bool cross_block[256] = {false};
   for (unsigned h = 0; h < MIN2(reserved, ARRAY_SIZE(written)); h++)
      blocked[h / 2] = true;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump) {
            nir_jump_type type = nir_instr_as_jump(instr)->type;
            if (type != nir_jump_break && type != nir_jump_continue)
               return false;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         unsigned base, size;
         if (intr->intrinsic == nir_intrinsic_store_preamble) {
            if (nir_intrinsic_preamble_class(intr) !=
                nir_preamble_class_general)
               continue;
            nir_def *value = intr->src[0].ssa;
            base = nir_intrinsic_base(intr);
            size = MAX2(value->bit_size, 16) * value->num_components / 16;
            if (base >= ARRAY_SIZE(written) ||
                size > ARRAY_SIZE(written) - base)
               return false;
            if (value->num_components == 1 &&
                (value->bit_size == 1 || value->bit_size == 16)) {
               written[base] = true;
               if (first_block[base / 2] && first_block[base / 2] != block)
                  cross_block[base / 2] = true;
               else
                  first_block[base / 2] = block;
               continue;
            }
         } else if (intr->intrinsic == nir_intrinsic_load_preamble &&
                    nir_intrinsic_preamble_class(intr) ==
                       nir_preamble_class_general) {
            base = nir_intrinsic_base(intr);
            size = MAX2(intr->def.bit_size, 16) * intr->def.num_components / 16;
         } else if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                    intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
            base = nir_intrinsic_desc_set(intr);
            size = 4;
         } else {
            /* Do not move exports through an unaccounted-for side effect. */
            if (!(nir_intrinsic_infos[intr->intrinsic].flags &
                  NIR_INTRINSIC_CAN_ELIMINATE))
               return false;
            continue;
         }
         if (base >= ARRAY_SIZE(written) || size > ARRAY_SIZE(written) - base)
            return false;
         for (unsigned h = 0; h < size; h++)
            blocked[(base + h) / 2] = true;
      }
   }

   nir_variable *halves[512] = {NULL};
   bool progress = false;
   for (unsigned base = 0; base < ARRAY_SIZE(written); base += 2) {
      if (base < reserved || base + 1 >= allocated || blocked[base / 2] ||
          !cross_block[base / 2] || !written[base] || !written[base + 1])
         continue;
      for (unsigned h = 0; h < 2; h++)
         halves[base + h] = nir_local_variable_create(
            impl, glsl_uint16_t_type(), "preamble_half");
      progress = true;
   }
   if (!progress)
      return false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
            continue;
         unsigned base = nir_intrinsic_base(intr);
         if (!halves[base])
            continue;
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_def *value = intr->src[0].ssa;
         /* AGX represents scalar NIR booleans as the half values 0 or 1. */
         if (value->bit_size == 1)
            value = nir_bcsel(&b, value, nir_imm_intN_t(&b, 1, 16),
                              nir_imm_intN_t(&b, 0, 16));
         nir_store_var(&b, halves[base], value, 1);
         nir_instr_remove(instr);
      }
   }
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   for (unsigned base = 0; base < ARRAY_SIZE(halves); base += 2) {
      if (!halves[base])
         continue;
      nir_def *pair =
         nir_pack_32_2x16_split(&b, nir_load_var(&b, halves[base]),
                                nir_load_var(&b, halves[base + 1]));
      nir_store_preamble(&b, pair, .base = base,
                         .preamble_class = nir_preamble_class_general);
   }
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_lower_vars_to_ssa(nir);
   return true;
}


static unsigned
if_depth(const nir_instr *instr)
{
   unsigned depth = 0;
   for (nir_cf_node *n = instr->block->cf_node.parent; n; n = n->parent)
      depth += n->type == nir_cf_node_if;
   return depth;
}

/* If this 64-bit value is a sample's texture or sampler handle, or with
 * load_bases a load's base address, the least if-nesting depth of those uses;
 * otherwise UINT_MAX.
 */
static unsigned
sample_handle_depth(const nir_def *def, bool load_bases)
{
   unsigned depth = UINT_MAX;
   if (def->bit_size != 64 || def->num_components != 1)
      return depth;

   nir_foreach_use(use, def) {
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);
         if (load_bases && use == &intr->src[0] &&
             (intr->intrinsic == nir_intrinsic_load_constant_agx ||
              intr->intrinsic == nir_intrinsic_load_agx))
            depth = MIN2(depth, if_depth(parent));
         continue;
      }
      if (parent->type != nir_instr_type_tex)
         continue;
      nir_tex_instr *tex = nir_instr_as_tex(parent);
      const unsigned s = (nir_tex_src *)use - tex->src;
      switch (tex->src[s].src_type) {
      case nir_tex_src_texture_offset:
      case nir_tex_src_texture_handle:
      case nir_tex_src_sampler_offset:
      case nir_tex_src_sampler_handle:
         depth = MIN2(depth, if_depth(parent));
         break;
      default:
         break;
      }
   }
   return depth;
}

struct preamble_slot {
   uint16_t base, size, align, pad_size;
   bool handle;
   unsigned depth; /* least if-depth of a sample using it as a handle */
   uint16_t new_base;
};

static int
find_slot(struct preamble_slot *slots, unsigned n, unsigned half)
{
   for (unsigned i = 0; i < n; ++i) {
      if (half >= slots[i].base && half < slots[i].base + slots[i].size)
         return i;
   }
   return -1;
}

/*
 * nir_opt_preamble lays general exports out by benefit, so bindless texture
 * and sampler handles of a shader with many exports can land above the range
 * a sample reads directly, and each sample then copies its handles into
 * registers first. With load_bases, G17P device loads count too: they read
 * their base from the same range, and a base above it costs two copies per
 * load. Move every export that is used as a handle to the bottom of the general
 * range [reserved, exports_end), keeping the others in order behind them.
 * Uniforms from exports_end to the allocated size, if any, are copied in by the
 * launch and stay where they are. Updates the allocated size when alignment
 * grows the layout.
 */
bool
agx_nir_place_preamble_handles_low(nir_shader *nir, unsigned reserved,
                                   unsigned exports_end,
                                   unsigned *allocated_inout, bool load_bases)
{
   const unsigned allocated = *allocated_inout;
   if (!nir_shader_get_entrypoint(nir)->preamble || allocated > 512 ||
       exports_end > allocated)
      return false;
   const unsigned limit = exports_end < allocated ? exports_end : 512;

   struct preamble_slot slots[512];
   unsigned n = 0;
   bool any_high_handle = false;

   /* Every general export is one slot, sized and aligned like its value. */
   nir_function_impl *preamble = nir_shader_get_preamble(nir);
   nir_foreach_block(block, preamble) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
            continue;
         const nir_def *value = intr->src[0].ssa;
         const unsigned base = nir_intrinsic_base(intr);
         const unsigned align = MAX2(value->bit_size, 16) / 16;
         const unsigned size = align * value->num_components;
         /* A lone half is written as a 32-bit copy whose other half must be
          * padding (agx3_pack_binary), so it keeps its pair to itself. */
         const unsigned slot_align = MAX2(align, 2);
         const unsigned slot_size = ALIGN_POT(size, 2);
         if (base < reserved || base + size > exports_end)
            return false;

         const int s = find_slot(slots, n, base);
         if (s >= 0) {
            /* The same slot written again (one value per branch). */
            if (slots[s].base != base || slots[s].size != size)
               return false;
            continue;
         }
         if (find_slot(slots, n, base + size - 1) >= 0)
            return false;
         slots[n++] = (struct preamble_slot){
            .base = base, .size = size, .align = slot_align,
            .pad_size = slot_size, .depth = UINT_MAX};
      }
   }

   /* Every read must fall inside one slot; handles are four aligned halves. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned base, size;
            unsigned depth = UINT_MAX;
            if (intr->intrinsic == nir_intrinsic_load_preamble &&
                nir_intrinsic_preamble_class(intr) ==
                   nir_preamble_class_general) {
               base = nir_intrinsic_base(intr);
               size = MAX2(intr->def.bit_size, 16) * intr->def.num_components / 16;
               depth = sample_handle_depth(&intr->def, load_bases);
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
               base = nir_intrinsic_desc_set(intr);
               size = 4;
               depth = if_depth(&intr->instr);
            } else {
               continue;
            }
            if (base < reserved || base >= exports_end)
               continue;
            const int s = find_slot(slots, n, base);
            if (s < 0 || base + size > slots[s].base + slots[s].size)
               return false;
            if (depth != UINT_MAX) {
               if (base != slots[s].base || slots[s].size != 4)
                  return false;
               slots[s].handle = true;
               slots[s].align = 4;
               slots[s].depth = MIN2(slots[s].depth, depth);
               any_high_handle |= base > AGX3_SMP_HANDLE_MAX_HALF;
            }
         }
      }
   }

   if (!any_high_handle)
      return false;

   /* Handles first, those of the least conditional samples first, then
    * everything else in its current order. */
   unsigned max_depth = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (slots[i].handle)
         max_depth = MAX2(max_depth, slots[i].depth);
   }
   unsigned next = reserved;
   for (unsigned depth = 0; depth <= max_depth + 1; ++depth) {
      for (unsigned i = 0; i < n; ++i) {
         if (slots[i].handle ? slots[i].depth != depth : depth <= max_depth)
            continue;
         next = ALIGN_POT(next, slots[i].align);
         slots[i].new_base = next;
         next += slots[i].pad_size;
      }
   }
   /* Alignment can grow the layout by a few halves; stay below the launch's
    * copies and in the hardware's uniform file. */
   if (next > limit)
      return false;
   *allocated_inout = MAX2(allocated, next);

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool general =
               (intr->intrinsic == nir_intrinsic_load_preamble ||
                intr->intrinsic == nir_intrinsic_store_preamble) &&
               nir_intrinsic_preamble_class(intr) == nir_preamble_class_general;
            if (general) {
               const unsigned base = nir_intrinsic_base(intr);
               if (base < reserved || base >= exports_end)
                  continue;
               const struct preamble_slot *s = &slots[find_slot(slots, n, base)];
               nir_intrinsic_set_base(intr, s->new_base + (base - s->base));
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
               const unsigned base = nir_intrinsic_desc_set(intr);
               if (base < reserved || base >= exports_end)
                  continue;
               nir_intrinsic_set_desc_set(intr,
                  slots[find_slot(slots, n, base)].new_base);
            }
         }
      }
      nir_progress(true, impl, nir_metadata_all);
   }

   return true;
}
