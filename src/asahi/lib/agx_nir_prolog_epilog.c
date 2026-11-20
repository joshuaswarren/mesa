/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gallium/include/pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/nir/poly_nir.h"
#include "util/format/u_formats.h"
#include "agx_abi.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_lower_blend.h"
#include "shader_enums.h"

static bool drop_unit_range_source_clamps(nir_shader *s);

/* Mesa's "every sample" spelling, shared with agx_nir_lower_tilebuffer. */
#define ALL_SAMPLES 0xFF

/*
 * Insert code into a fragment shader to lower polygon stipple. The stipple is
 * passed in a sideband, rather than requiring a texture binding. This is
 * simpler for drivers to integrate and might be more efficient.
 */
static bool
agx_nir_lower_poly_stipple(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   /* Insert at the beginning for performance. */
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   /* The stipple coordinate is defined at the window coordinate mod 32. It's
    * reversed along the X-axis to simplify the driver, hence the NOT.
    */
   nir_def *raw = nir_u2u32(b, nir_load_pixel_coord(b));
   nir_def *coord = nir_umod_imm(
      b,
      nir_vec2(b, nir_inot(b, nir_channel(b, raw, 0)), nir_channel(b, raw, 1)),
      32);

   /* Extract the column from the packed bitfield */
   nir_def *pattern = nir_load_polygon_stipple_agx(b, nir_channel(b, coord, 1));
   nir_def *bit = nir_ubitfield_extract(b, pattern, nir_channel(b, coord, 0),
                                        nir_imm_int(b, 1));

   /* Discard fragments where the pattern is 0 */
   nir_demote_if(b, nir_ieq_imm(b, bit, 0));
   s->info.fs.uses_discard = true;

   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

static int
map_vs_part_uniform(nir_intrinsic_instr *intr, unsigned nr_attribs,
                    unsigned uniform_base)
{
   int uniform;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vbo_base_agx:
      uniform = AGX_ABI_VUNI_VBO_BASE(nir_src_as_uint(intr->src[0]));
      break;

   case nir_intrinsic_load_attrib_clamp_agx:
      uniform =
         AGX_ABI_VUNI_VBO_CLAMP(nr_attribs, nir_src_as_uint(intr->src[0]));
      break;

   case nir_intrinsic_load_first_vertex:
      uniform = AGX_ABI_VUNI_FIRST_VERTEX(nr_attribs);
      break;

   case nir_intrinsic_load_base_instance:
      uniform = AGX_ABI_VUNI_BASE_INSTANCE(nr_attribs);
      break;

   case nir_intrinsic_load_vertex_param_buffer_poly:
      uniform = AGX_ABI_VUNI_VERTEX_PARAMS(nr_attribs);
      break;

   default:
      return -1;
   }

   return uniform_base + uniform;
}

static int
map_fs_part_uniform(nir_intrinsic_instr *intr,
                    const struct agx_fs_epilog_key *key)
{
   const unsigned blend_base = key && key->blend_constant_base
                                  ? key->blend_constant_base
                                  : AGX_ABI_FUNI_BLEND_R;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_blend_const_color_r_float:
      return blend_base;

   case nir_intrinsic_load_blend_const_color_g_float:
      return blend_base + 2;

   case nir_intrinsic_load_blend_const_color_b_float:
      return blend_base + 4;

   case nir_intrinsic_load_blend_const_color_a_float:
      return blend_base + 6;

   default:
      return -1;
   }
}

static bool
lower_native_emrt_handle(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   const struct agx_fs_epilog_key *key = data;
   nir_def *index = intr->src[0].ssa;
   /* API bindings precede Gallium's private spill records. Only those
    * records expand from two logical handles to three native slots. */
   unsigned base = key->native_spill_generation == 10
                      ? key->link.rt_spill_base : 0;
   index = nir_iadd_imm(b, index, -(int)base);
   nir_def *rt = nir_ushr_imm(b, index, 1);
   nir_def *offset = nir_iadd(b, nir_imul_imm(b, rt, AGX_G17P_EMRT_DESCRIPTOR_SIZE),
                            nir_ishl_imm(b, nir_iand_imm(b, index, 1), 5));
   offset = nir_iadd_imm(b, offset, base * 32);
   nir_def_replace(&intr->def, nir_bindless_image_agx(b, offset, .desc_set = 0));
   return true;
}

/* G17G spills: the handles of the descriptor crawls left after
 * agx_nir_rebind_spilled_rts(), at the stride of the 32-byte Gen10
 * descriptors rather than the AGX2 AGX_TEXTURE_LENGTH. */
static bool
lower_gen10_texture_handle(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offset = nir_imul_imm(b, nir_u2u32(b, intr->src[0].ssa), 32);
   nir_def_replace(&intr->def, nir_bindless_image_agx(b, offset, .desc_set = 0));
   return true;
}

static bool
lower_native_emrt_uniforms(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_fs_epilog_key *key = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
      nir_def *value = intr->src[3].ssa;
      nir_alu_type type = nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));
      if (value->bit_size < 32) {
         value = type == nir_type_float ? nir_f2f32(b, value) :
                 type == nir_type_int ? nir_i2i32(b, value) : nir_u2u32(b, value);
         nir_src_rewrite(&intr->src[3], value);
         nir_intrinsic_set_src_type(intr, type | 32);
      }
      /* G17P's existing bindless store consumes native 32-byte PBE records,
       * not byte offsets. Descriptor crawls retain their byte handles. */
      nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
      assert(handle && handle->intrinsic == nir_intrinsic_bindless_image_agx);
      nir_src_rewrite(&intr->src[0], nir_bindless_image_agx(
         b, nir_ushr_imm(b, handle->src[0].ssa, 5),
         .desc_set = key->emrt_heap_base));
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
      assert(nir_intrinsic_desc_set(intr) == 0);
      nir_def_replace(&intr->def,
         nir_load_preamble(b, 1, 64, .base = key->emrt_heap_base));
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
      nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
      assert(handle && handle->intrinsic == nir_intrinsic_bindless_image_agx);
      nir_def_replace(&intr->def, nir_iadd(b,
         nir_load_preamble(b, 1, 64, .base = key->emrt_heap_base),
         nir_u2u64(b, handle->src[0].ssa)));
      return true;
   }
   return false;
}

static bool
lower_non_monolithic_uniforms(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   int unif;
   if (b->shader->info.stage == MESA_SHADER_VERTEX) {
      const struct agx_vs_prolog_key *layout = data;
      unif = map_vs_part_uniform(intr,
                                 DIV_ROUND_UP(BITSET_LAST_BIT(
                                                 layout->component_mask),
                                              4),
                                 layout->uniform_base);
   } else {
      unif = map_fs_part_uniform(intr, data);
   }

   if (unif >= 0) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *load = nir_load_preamble(b, 1, intr->def.bit_size, .base = unif);
      nir_def_rewrite_uses(&intr->def, load);
      return true;
   } else if (intr->intrinsic == nir_intrinsic_load_texture_handle_agx) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *offs =
         nir_imul_imm(b, nir_u2u32(b, intr->src[0].ssa), AGX_TEXTURE_LENGTH);
      nir_def_rewrite_uses(&intr->def,
                           nir_bindless_image_agx(b, offs, .desc_set = 0));
      return true;
   } else {
      return false;
   }
}

bool
agx_nir_lower_non_monolithic_uniforms_at(nir_shader *nir, unsigned nr,
                                         unsigned uniform_base)
{
   struct agx_vs_prolog_key layout = {.uniform_base = uniform_base};
   if (nr)
      BITSET_SET_COUNT(layout.component_mask, 0, nr * 4);

   return nir_shader_intrinsics_pass(nir, lower_non_monolithic_uniforms,
                                     nir_metadata_control_flow, &layout);
}

bool
agx_nir_lower_non_monolithic_uniforms(nir_shader *nir, unsigned nr)
{
   return agx_nir_lower_non_monolithic_uniforms_at(nir, nr, 0);
}

static bool
lower_adjacency(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_vs_prolog_key *key = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_load_vertex_id)
      return false;

   nir_def *id = nir_load_vertex_id(b);

   if (key->adjacency == MESA_PRIM_LINES_ADJACENCY) {
      id = poly_map_to_line_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      id = poly_map_to_tri_strip_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      id = poly_map_to_line_strip_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_TRIANGLES_ADJACENCY) {
      /* Sequence (0, 2, 4), (6, 8, 10), ... */
      id = nir_imul_imm(b, id, 2);
   } else {
      UNREACHABLE("unknown");
   }

   id = poly_nir_load_vertex_id(b, id);

   nir_def_replace(&intr->def, id);
   return true;
}

/* A software VS fetches for the draw's instance, not the dispatch's. */
static bool
lower_sw_instance_offset(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_instance_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *id = nir_channel(b, nir_load_global_invocation_id(b, 32), 1);
   nir_def_replace(&intr->def,
                   nir_iadd(b, id, poly_nir_load_instance_offset(b)));
   return true;
}

static void
agx_nir_vs_prolog_impl(nir_builder *b, const void *key_, bool native_raw)
{
   const struct agx_vs_prolog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_VERTEX;
   b->shader->info.name = "VS prolog";

   /* First, construct a passthrough shader reading each attribute and exporting
    * the value. We also need to export vertex/instance ID in their usual regs.
    */
   if (!key->static_vi) {
      unsigned i = 0;
      nir_def *vec = NULL;
      unsigned vec_idx = ~0;
      BITSET_FOREACH_SET(i, key->component_mask, AGX_MAX_ATTRIBS * 4) {
         unsigned a = i / 4;
         unsigned c = i % 4;

         if (vec_idx != a) {
            vec = nir_load_input(b, 4, 32, nir_imm_int(b, 0), .base = a);
            vec_idx = a;
         }

         nir_export_agx(b, nir_channel(b, vec, c),
                        .base = AGX_ABI_VIN_ATTRIB(i));
      }
   }

   if (!key->hw) {
      nir_export_agx(b, nir_channel(b, nir_load_global_invocation_id(b, 32), 0),
                     .base = AGX_ABI_VIN_VERTEX_ID_ZERO_BASE);
   }

   nir_export_agx(b, nir_load_vertex_id(b), .base = AGX_ABI_VIN_VERTEX_ID);

   /* With an instance offset the exported ID stays the dispatch's: the main
    * shader lays its outputs out by it, and adds the offset to the ID it
    * reads itself.
    */
   const bool instance_offset = !key->hw && key->sw_instance_offset;
   nir_export_agx(b,
                  instance_offset
                     ? nir_channel(b, nir_load_global_invocation_id(b, 32), 1)
                     : nir_load_instance_id(b),
                  .base = AGX_ABI_VIN_INSTANCE_ID);

   /* Now lower the resulting program using the key */
   if (!key->static_vi) {
      if (native_raw)
         agx_nir_lower_vbo_raw(b->shader, key->attribs, key->robustness, false);
      else
         agx_nir_lower_vbo(b->shader, key->attribs, key->robustness, false);

      /* Clean up redundant vertex ID loads */
      if (!key->hw || key->adjacency) {
         NIR_PASS(_, b->shader, nir_opt_cse);
         NIR_PASS(_, b->shader, nir_opt_dce);
      }
   }

   if (!key->hw) {
      b->cursor = nir_before_impl(nir_shader_get_entrypoint(b->shader));
      /* Only the fetches still read the instance ID here. */
      if (instance_offset)
         nir_shader_intrinsics_pass(b->shader, lower_sw_instance_offset,
                                    nir_metadata_control_flow, NULL);
      poly_nir_lower_sw_vs(b->shader);
   } else if (key->adjacency) {
      nir_shader_intrinsics_pass(b->shader, lower_adjacency,
                                 nir_metadata_control_flow, (void *)key);
   }
   nir_inline_sysval(b->shader, nir_intrinsic_load_index_size_poly,
                     key->sw_index_size_B);

   /* Finally, lower uniforms according to our ABI */
   nir_shader_intrinsics_pass(b->shader, lower_non_monolithic_uniforms,
                              nir_metadata_control_flow, (void *)key);
   b->shader->info.io_lowered = true;
}

void
agx_nir_vs_prolog(nir_builder *b, const void *key)
{
   agx_nir_vs_prolog_impl(b, key, false);
}

void
agx_nir_vs_prolog_raw(nir_builder *b, const void *key)
{
   agx_nir_vs_prolog_impl(b, key, true);
}

/*
 * Map a lowered load_input to its slot in the vertex input ABI.
 *
 * The prolog publishes every attribute component as a full 32-bit value in
 * its own pair of half-registers: agx_nir_vs_prolog_impl() fetches with
 * nir_load_input(b, 4, 32, ...) and exports each channel at
 * AGX_ABI_VIN_ATTRIB(4 * attrib + comp) = 2 * (8 + i), two halves apart.
 * struct agx_vs_prolog_key carries no bit sizes at all -- only a component
 * mask in these same 32-bit slot units -- so the prolog cannot publish
 * anything narrower even in principle.
 *
 * A main shader may still declare a narrower input, since we advertise
 * VK_KHR_16bit_storage's storageInputOutput16. That does not change the slot
 * arithmetic: lowered NIR IO keeps nir_intrinsic_component() in 32-bit slot
 * components even for 16-bit loads (nir_validate asserts component +
 * num_components <= 4), and the only way two 16-bit values share one
 * component is nir_io_semantics::high_16bits, which nir_lower_io never sets
 * and which is produced for varyings alone (nir_lower_mediump's 16-bit
 * varying slots and nir_opt_vectorize_io, neither of which we run). So one
 * declared component is still exactly one 32-bit ABI slot, and all the main
 * shader has to do is narrow the value the prolog handed it.
 */
static unsigned
vs_input_abi_slot(nir_intrinsic_instr *intr)
{
   ASSERTED nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   assert(!sem.high_16bits && "VS inputs never share a 32-bit slot");
   assert(intr->def.bit_size <= 32 && "64-bit attributes need two ABI slots");

   unsigned idx = nir_src_as_uint(intr->src[0]) + nir_intrinsic_base(intr);
   unsigned comp = nir_intrinsic_component(intr);
   assert(comp + intr->def.num_components <= 4);

   return 4 * idx + comp;
}

static bool
gather_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned base = vs_input_abi_slot(intr);

   b->cursor = nir_before_instr(&intr->instr);
   BITSET_WORD *comps_read = data;
   nir_component_mask_t mask = nir_def_components_read(&intr->def);

   u_foreach_bit(c, mask) {
      assert(base + c < AGX_MAX_ATTRIBS * 4 &&
             "attribute slot beyond component mask capacity");
      BITSET_SET(comps_read, base + c);
   }

   return false;
}

bool
agx_nir_gather_vs_inputs(nir_shader *s, BITSET_WORD *attrib_components_read)
{
   return nir_shader_intrinsics_pass(
      s, gather_inputs, nir_metadata_control_flow, attrib_components_read);
}

static bool
lower_input_to_prolog(nir_builder *b, nir_intrinsic_instr *intr, void *_data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned base = vs_input_abi_slot(intr);

   b->cursor = nir_before_instr(&intr->instr);

   /* Always read the ABI slot at its published width. Loading at the narrow
    * width would be silently wrong rather than merely slow: agx_load_exported
    * strides the base by the DESTINATION size, so a 16-bit load would take
    * component 1 out of component 0's high half.
    */
   nir_def *val = nir_load_exported_agx(b, intr->def.num_components, 32,
                                        .base = AGX_ABI_VIN_ATTRIB(base));

   if (intr->def.bit_size != 32) {
      /* The prolog's vertex fetch already converted the attribute into the
       * 32-bit slot with the format's own numeric type (agx_nir_lower_vbo
       * emits f2f32/i2i32/u2u32), so narrowing by the declared base type
       * recovers the value exactly.
       */
      nir_alu_type base_type =
         nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));

      assert((base_type == nir_type_float || base_type == nir_type_int ||
              base_type == nir_type_uint) &&
             "narrow vertex input without a numeric type");

      val = nir_convert_to_bit_size(b, val, base_type, intr->def.bit_size);
   }

   nir_def_replace(&intr->def, val);
   return true;
}

bool
agx_nir_lower_vs_input_to_prolog(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_input_to_prolog,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_active_samples_to_register(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *id =
      nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

   nir_def_replace(&intr->def, id);
   return true;
}

static bool
lower_tests_zs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *value = data;
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, *value ? 0xFF : 0, 16));
   return true;
}

static bool
lower_tests_zs(nir_shader *s, bool value)
{
   if (!s->info.fs.uses_discard)
      return false;

   return nir_shader_intrinsics_pass(s, lower_tests_zs_intr,
                                     nir_metadata_control_flow, &value);
}

static inline bool
blend_uses_2src(struct agx_blend_rt_key rt)
{
   assert(rt.advanced_blend == false);
   const struct agx_blend_standard blend = agx_unpack_blend_standard(rt.mode);

   enum pipe_blendfactor factors[] = {
      blend.rgb_src_factor,
      blend.rgb_dst_factor,
      blend.alpha_src_factor,
      blend.alpha_dst_factor,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      switch (factors[i]) {
      case PIPE_BLENDFACTOR_SRC1_COLOR:
      case PIPE_BLENDFACTOR_SRC1_ALPHA:
      case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
         return true;
      default:
         break;
      }
   }

   return false;
}

static void
copy_colour(nir_builder *b, const struct agx_fs_epilog_key *key,
            unsigned out_rt, unsigned in_loc, bool dual_src)
{
   unsigned size = (key->link.size_32 & BITFIELD_BIT(in_loc)) ? 32 : 16;

   struct agx_g17p_colour_layout layout;
   agx_g17p_colour_layout_init(&layout, key->link.loc_written,
                               key->link.size_32, key->link.colour_comps,
                               key->link.packed_colour);

   /* Packed colours hold only the components the main fragment exported.
    * Everything past them is undefined in the API too, so read what is there
    * and leave the rest undefined rather than reading the next location's
    * registers -- a location with no components of its own shares its base
    * with the next one. The ABI layout keeps all four components and is
    * unchanged. */
   const unsigned nr = layout.comps[in_loc];
   nir_def *value = NULL;

   if (nr > 0) {
      value = nir_load_exported_agx(
         b, nr, size, .base = agx_g17p_sample_colour_base(&layout, 0, in_loc));
   }

   if (nr < 4) {
      nir_def *comps[4];

      for (unsigned c = 0; c < 4; ++c)
         comps[c] = c < nr ? nir_channel(b, value, c) : nir_undef(b, 1, size);

      value = nir_vec(b, comps, 4);
   }

   if (key->link.loc0_w_1 && in_loc == 0) {
      value =
         nir_vector_insert_imm(b, value, nir_imm_floatN_t(b, 1.0, size), 3);
   }

   if (key->smooth_line_alpha && in_loc == 0 && !dual_src) {
      nir_def *alpha = nir_fmul_imm(b, nir_channel(b, value, 3), 0.5);
      value = nir_vector_insert_imm(b, value, alpha, 3);
   }

   nir_store_output(b, value, nir_imm_int(b, 0),
                    .io_semantics.location = FRAG_RESULT_DATA0 + out_rt,
                    .io_semantics.dual_source_blend_index = dual_src,
                    .src_type = nir_type_float | size);
}

/* Blend in the API format, then restore the wider physical store shape.
 * Padding is never observable through a public view or image transfer. */
static bool
pad_blended_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_fs_epilog_key *key = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < FRAG_RESULT_DATA0 || sem.location > FRAG_RESULT_DATA7)
      return false;

   unsigned rt = sem.location - FRAG_RESULT_DATA0;
   if (key->blend_formats[rt] == PIPE_FORMAT_NONE &&
       key->scaled_formats[rt] == PIPE_FORMAT_NONE)
      return false;

   unsigned comps = util_format_get_nr_components(key->rt_formats[rt]);
   if (intr->num_components >= comps)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[0], nir_pad_vector(b, intr->src[0].ssa, comps));
   intr->num_components = comps;
   return true;
}

/* Scaled Vulkan attachments have floating-point shader semantics and integer
 * physical storage. Convert after blending and before tilebuffer lowering. */
static bool
convert_scaled_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_fs_epilog_key *key = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < FRAG_RESULT_DATA0 || sem.location > FRAG_RESULT_DATA7)
      return false;

   const unsigned rt = sem.location - FRAG_RESULT_DATA0;
   const enum pipe_format format = key->scaled_formats[rt];
   if (format == PIPE_FORMAT_NONE)
      return false;

   const struct util_format_description *desc =
      util_format_description(format);
   const bool is_signed = desc->channel[0].type == UTIL_FORMAT_TYPE_SIGNED;
   unsigned component_bits[4];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      /* Meta fragment shaders may carry a fourth value into a three-component
       * attachment. Constant format swizzles need no special range, so retain
       * the format's uniform channel width for that otherwise-unobservable
       * lane. Packed formats map all four logical components to real channels. */
      const unsigned channel = desc->swizzle[c] < PIPE_SWIZZLE_0
                                  ? desc->swizzle[c]
                                  : 0;
      if (desc->channel[channel].normalized ||
          desc->channel[channel].type != desc->channel[0].type)
         return false;

      component_bits[c] = desc->channel[channel].size;
      if (component_bits[c] == 0 || component_bits[c] >= 32)
         return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *upper_components[4];
   nir_def *lower_components[4];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      const unsigned bits = component_bits[c];
      nir_def *limit = nir_imm_float(b, 1u << bits);
      nir_def *signed_limit = nir_fmul_imm(b, limit, 0.5);
      upper_components[c] = nir_fadd_imm(
         b, is_signed ? signed_limit : limit, -1);
      lower_components[c] =
         is_signed ? nir_fneg(b, signed_limit) : nir_imm_float(b, 0);
   }

   nir_def *upper = nir_vec(b, upper_components, intr->num_components);
   nir_def *lower = nir_vec(b, lower_components, intr->num_components);
   nir_def *rounded = nir_fround_even(b, intr->src[0].ssa);
   nir_def *clamped = nir_fmin(b, nir_fmax(b, rounded, lower), upper);
   nir_src_rewrite(&intr->src[0],
                   is_signed ? nir_f2i32(b, clamped)
                             : nir_f2u32(b, clamped));
   nir_intrinsic_set_src_type(intr, is_signed ? nir_type_int32
                                               : nir_type_uint32);
   return true;
}

static void
agx_nir_fs_epilog_native(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS epilog";

   struct agx_tilebuffer_layout tib =
      key->g18_native_mixed
         ? agx_build_tilebuffer_layout_g18_mixed(key->rt_formats,
                                                 ARRAY_SIZE(key->rt_formats))
      : key->rt0_resident
         ? agx_build_tilebuffer_layout_rt0_resident(
              key->rt_formats, ARRAY_SIZE(key->rt_formats), key->nr_samples,
              key->max_bytes_per_tile, key->max_samples_per_tile)
         : key->native_imageblock
              ? agx_build_tilebuffer_layout_resident(
                   key->rt_formats, ARRAY_SIZE(key->rt_formats),
                   key->nr_samples, true, key->max_sample_size_B,
                   key->max_bytes_per_tile, key->max_samples_per_tile,
                   key->resident_mask)
              : agx_build_tilebuffer_layout_with_limits_resident(
                   key->rt_formats, ARRAY_SIZE(key->rt_formats),
                   key->nr_samples, true, key->max_sample_size_B,
                   key->max_bytes_per_tile, key->max_samples_per_tile,
                   key->resident_mask);

   unsigned rt_order[AGX_MAX_RENDER_TARGETS];
   for (unsigned rt = 0; rt < ARRAY_SIZE(rt_order); ++rt)
      rt_order[rt] = rt;

   /* G18's grounded multisample protocol loads resident partitions in
    * ascending order. Sharing the private descriptor ABI does not select
    * Gen9's descending-load transaction order. */
   if (key->native_spills && key->native_spill_generation != 10 &&
       agx_tilebuffer_spills(&tib)) {
      /* Spill stores introduce conditional image accesses. Emit those first,
       * then the resident attachments, so every sample finishes its tile
       * stores in one block. These outputs address distinct attachments;
       * sorting before blend lowering keeps each destination load with its
       * own store and does not move accesses across sample iterations. */
      for (unsigned i = 0; i < ARRAY_SIZE(rt_order); ++i) {
         for (unsigned j = i + 1; j < ARRAY_SIZE(rt_order); ++j) {
            unsigned a = rt_order[i], c = rt_order[j];
            if ((!tib.spilled[a] && tib.spilled[c]) ||
                (!tib.spilled[a] && !tib.spilled[c] &&
                 agx_tilebuffer_offset_B(&tib, a) <
                    agx_tilebuffer_offset_B(&tib, c)))
               SWAP(rt_order[i], rt_order[j]);
         }
      }
   }

   /* Import masks before the colour stores: alpha-to-coverage is inserted
    * before those stores and must be dominated by the imported write mask. */
   nir_def *sample_id = NULL, *write_samples = NULL;
   if (key->link.sample_shading) {
      sample_id =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_SAMPLE_MASK);
   }

   if (key->link.sample_mask_after_force_early) {
      write_samples =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   }

   if (key->defer_api_sample_mask) {
      nir_def *api_mask = nir_imm_intN_t(b, key->api_sample_mask, 16);
      write_samples = write_samples ? nir_iand(b, write_samples, api_mask)
                                    : api_mask;
   }

   /* First, construct a passthrough shader reading each colour and outputting
    * the value.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(key->remap); ++i) {
      unsigned rt = rt_order[i];
      int location = key->remap[rt];

      /* A write-disabled attachment stores its own tile value back; blending
       * cannot express this for pure-integer formats (nir_lower_blend skips
       * them: dynamic_rendering.*.local_read.interaction_with_color_write_
       * enable on r32_uint). */
      if ((key->preserve_rt & BITFIELD_BIT(rt)) &&
          key->rt_formats[rt] != PIPE_FORMAT_NONE) {
         nir_io_semantics sem = {
            .location = FRAG_RESULT_DATA0 + rt,
            .num_slots = 1,
            .fb_fetch_output = true,
         };
         b->shader->info.outputs_read |= BITFIELD64_BIT(sem.location);
         b->shader->info.fs.uses_fbfetch_output = true;
         nir_def *dst = nir_load_output(b, 4, 32, nir_imm_int(b, 0),
                                        .dest_type = nir_type_float32,
                                        .io_semantics = sem);
         nir_store_output(b, dst, nir_imm_int(b, 0),
                          .io_semantics.location = FRAG_RESULT_DATA0 + rt,
                          .src_type = nir_type_float32);
         continue;
      }

      /* Negative remaps indicate the attachment isn't written. */
      if (location >= 0 && key->link.loc_written & BITFIELD_BIT(location)) {
         copy_colour(b, key, rt, location, false);

         /* If this render target uses dual source blending, also copy the dual
          * source colour. While the copy_colour above is needed even for
          * missing attachments to handle alpha-to-coverage, this copy is only
          * for blending so should be suppressed for missing attachments to keep
          * the assert from blowing up on OpenGL.
          */
         if (!key->blend.rt[rt].advanced_blend &&
             blend_uses_2src(key->blend.rt[rt]) &&
             key->rt_formats[rt] != PIPE_FORMAT_NONE) {

            assert(location == 0);
            copy_colour(b, key, rt, 1, true);
         }
      }
   }

   /* Now lower the resulting program using the key */
   if (key->force_small_tile)
      tib.tile_size = 16 * 16;

   bool force_translucent = false;
   nir_lower_blend_options opts = {
      .scalar_blend_const = true,
      .logicop_enable = key->blend.logicop_enable,
      .logicop_func = key->blend.logicop_func,
   };

   static_assert(ARRAY_SIZE(opts.rt) == 8, "max RTs out of sync");

   for (unsigned i = 0; i < 8; ++i) {
      if (key->blend.rt[i].advanced_blend) {
         const struct agx_blend_advanced blend =
            agx_unpack_blend_advanced(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->blend_formats[i] != PIPE_FORMAT_NONE
                         ? key->blend_formats[i]
                         : key->rt_formats[i],
            .advanced_blend = true,
            .colormask = key->blend.rt[i].colormask,
            .blend_mode = blend.op,
            .src_premultiplied = blend.src_premultiplied,
            .dst_premultiplied = blend.dst_premultiplied,
            .overlap = blend.overlap,
         };
      } else {
         const struct agx_blend_standard blend =
            agx_unpack_blend_standard(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->blend_formats[i] != PIPE_FORMAT_NONE
                         ? key->blend_formats[i]
                         : key->rt_formats[i],

            .rgb.src_factor = blend.rgb_src_factor,
            .rgb.dst_factor = blend.rgb_dst_factor,
            .rgb.func = blend.rgb_func,

            .alpha.src_factor = blend.alpha_src_factor,
            .alpha.dst_factor = blend.alpha_dst_factor,
            .alpha.func = blend.alpha_func,

            .colormask = key->blend.rt[i].colormask,
         };
      }
   }

   /* It's more efficient to use masked stores (with
    * agx_nir_lower_tilebuffer) than to emulate colour masking with
    * nir_lower_blend.
    */
   uint8_t colormasks[8] = {0};

   for (unsigned i = 0; i < 8; ++i) {
      if (key->rt_formats[i] == PIPE_FORMAT_NONE)
         continue;

      if (key->blend.rt[i].colormask == 0) {
         /* Writing nothing needs no masked store, so this costs nothing even
          * where the tilebuffer cannot mask -- and it must go this way. Left
          * to nir_lower_blend, a mask of zero deletes the store outright, and
          * a covered fragment that stores nothing loses the tile it was drawn
          * over: agx_nir_lower_tilebuffer preserves the destination instead.
          */
         colormasks[i] = 0;
         opts.rt[i].colormask = (uint8_t)BITFIELD_MASK(4);
      } else if (agx_tilebuffer_supports_mask(&tib, i)) {
         colormasks[i] = key->blend.rt[i].colormask;
         opts.rt[i].colormask = (uint8_t)BITFIELD_MASK(4);
      } else {
         colormasks[i] = (uint8_t)BITFIELD_MASK(4);
      }

      /* If not all bound RTs are fully written to, we need to force
       * translucent pass type. agx_nir_lower_tilebuffer will take
       * care of this for its own colormasks input.
       */
      unsigned comps = util_format_get_nr_components(key->rt_formats[i]);
      if ((opts.rt[i].colormask & BITFIELD_MASK(comps)) !=
          BITFIELD_MASK(comps)) {
         force_translucent = true;
      }
   }

   /* Alpha-to-coverage must be lowered before alpha-to-one */
   const bool discards_before_a2c = b->shader->info.fs.uses_discard;
   if (key->blend.alpha_to_coverage && !key->a2c_in_main)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_coverage, false,
               NULL);

   /* Native late SampleMask writes cross the main/epilog split in a register.
    * Combine them with alpha coverage before the one test-triggering update,
    * as Apple's G17P sample-mask + alpha-coverage fragment does. Applying the
    * mask only to the tile store would still update depth for killed samples.
    * Early-tested shaders must not run the tests again.
    *
    * At two and four samples the G17P depth/stencil write names the surviving
    * samples in its own per-sample coverage words, so it already excludes the
    * killed ones and needs no coverage update of its own. Issuing one anyway
    * makes the hardware run the depth/stencil test on each surviving sample
    * twice -- measured as four INCREMENT_AND_WRAP steps where the API asked
    * for two, on renderpasses.*.multisample.d32_sfloat_s8_uint.samples_4. At
    * one sample the write's word has no mask field at all, so there the kill
    * is the only thing that can hold the test back, as
    * fragment_operations.early_fragment.discard_no_early_fragment_tests_stencil
    * depends on. A deferred epilog is cloned per sample after this point and
    * coalesces its kills itself (agx_nir_hoist_g17p_zs_writes), so leave that
    * shape alone. */
   const bool zs_carries_write_samples =
      (key->link.write_z || key->link.write_s) && key->nr_samples > 1 &&
      !key->defer_monolithic_msaa;

   /* M5 carries alpha coverage the same way. Native G17G multisample
    * fragments that write depth (native G17G compiler zs_depth_4x_a2c,
    * zs_cdepth_4x_a2c and zs_depth_2x_a2c) evaluate it into each sample's
    * DEPTHF coverage word and issue no MOVMSK; the colour store follows
    * unmasked. A separate kill ahead of the words would be a second coverage
    * update. Sample-rate epilogs keep the kill: they are lowered per sample
    * below. */
   nir_def *zs_alpha_coverage = NULL;
   if (key->m5_coverage && zs_carries_write_samples &&
       key->blend.alpha_to_coverage && !key->link.sample_shading) {
      nir_foreach_block(block, b->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_demote_samples)
               continue;
            nir_builder mask_b = nir_builder_at(nir_before_instr(instr));
            nir_def *covered =
               nir_u2u16(&mask_b, nir_inot(&mask_b, intr->src[0].ssa));
            zs_alpha_coverage = zs_alpha_coverage
               ? nir_iand(&mask_b, zs_alpha_coverage, covered) : covered;
            nir_instr_remove(instr);
         }
      }
      b->shader->info.fs.uses_discard = discards_before_a2c;
   }

   /* already_ran_zs says the main shader has already triggered the
    * depth/stencil tests, so the epilog must not trigger them again.  It does
    * not say the main shader's coverage has been applied: a forced-early
    * shader runs the tests at its start with only the API mask
    * (hk_g17p_apply_api_mask_to_force_early) and exports the coverage its
    * discards and SampleMask writes produced in FOUT_WRITE_SAMPLES for us.
    * Gating this block on the flag dropped that mask on the floor and stored
    * colour for killed samples -- query_pool.discard.early.*.{discard,
    * sample_mask} saw white where the discard should have left the target
    * untouched.
    *
    * Applying it here does not re-run the tests: lower_tests_zs() below
    * substitutes 0 for load_shader_part_tests_zs_agx when already_ran_zs, and
    * run_tests_after_last_discard() then builds `sample_mask killed, ~killed`,
    * which kills without testing.  So keep the two decisions separate --
    * always honour the exported coverage, and let the flag govern only
    * whether the kill also triggers the tests. */
   bool single_sample_coverage_applied = false;
   if (write_samples) {
      bool kills = false;
      nir_foreach_block(block, b->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_demote_samples)
               continue;
            nir_builder mask_b = nir_builder_at(nir_before_instr(instr));
            nir_def *killed = nir_ixor(
               &mask_b, write_samples, nir_imm_intN_t(&mask_b, ALL_SAMPLES, 16));
            killed = nir_u2uN(&mask_b, killed, intr->src[0].ssa->bit_size);
            nir_src_rewrite(&intr->src[0],
                            nir_ior(&mask_b, intr->src[0].ssa, killed));
            kills = true;
         }
      }
      if (!kills && !zs_carries_write_samples) {
         nir_instr *imported = nir_def_instr(write_samples);
         nir_builder mask_b = nir_builder_at(nir_after_instr(imported));

         /* Two builders sharing one insertion point emit in reverse order.
          * `b` is parked on the imported mask itself whenever no colour was
          * copied, so without carrying it past the kill everything `b`
          * appends next -- the depth/stencil write below among it -- is
          * threaded in *ahead* of the kill, and a coverage update after the
          * write that triggered the tests is far too late. */
         const bool b_shares_cursor = b->cursor.option == nir_cursor_after_instr &&
                                      b->cursor.instr == imported;

         nir_demote_samples(&mask_b,
                            nir_ixor(&mask_b, write_samples,
                                      nir_imm_intN_t(&mask_b, ALL_SAMPLES, 16)));
         if (b_shares_cursor)
            b->cursor = mask_b.cursor;

         kills = true;
      }

      if (kills) {
         b->shader->info.fs.uses_discard = true;
         single_sample_coverage_applied =
            key->m5_coverage && key->nr_samples == 1;
      }
   }

   /* Depth/stencil writes must be deferred until after all discards,
    * particularly alpha-to-coverage.
    */
   if (key->link.write_z || key->link.write_s) {
      nir_def *zs_samples =
         write_samples ? nir_iand(b, write_samples,
                                  nir_load_active_samples_agx(b))
                       : nir_imm_intN_t(b, 0xFF, 16);
      if (zs_alpha_coverage)
         zs_samples = nir_iand(b, zs_samples, zs_alpha_coverage);
      nir_def *depth =
         nir_load_exported_agx(b, 1, 32, .base = AGX_ABI_FOUT_Z);

      nir_store_zs_agx(
         b, zs_samples, depth,
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_S),
         .base = (key->link.write_z ? 1 : 0) | (key->link.write_s ? 2 : 0));

      if (key->link.write_z)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);

      if (key->link.write_s)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_STENCIL);
   }

   /* Alpha-to-one must be lowered before blending */
   if (key->blend.alpha_to_one)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_one);

   NIR_PASS(_, b->shader, nir_lower_blend, &opts);
   if (key->colour_unit_range)
      NIR_PASS(_, b->shader, drop_unit_range_source_clamps);
   nir_shader_intrinsics_pass(b->shader, convert_scaled_output,
                              nir_metadata_control_flow, (void *)key);
   nir_shader_intrinsics_pass(b->shader, pad_blended_output,
                              nir_metadata_control_flow, (void *)key);

   /* Native1130 (run 1139): Apple's fragments for a G18 mixed-format tile
    * neither load nor store a target whose write mask selects none of its
    * channels -- 39 such targets in 16 of the 39 workloads; the fragment of
    * the one whose four targets are all masked is the 512 entry and its final
    * release alone, which is what an empty epilog emits. The tile keeps the
    * background's value. (A packed R11G11B10F target with an alpha-only mask
    * still reaches lower_tilebuffer with a nonzero mask and keeps its
    * read-modify-write, as Apple's does.) */
   const bool preserve_zero_mask =
      key->preserve_zero_mask && !key->g18_native_mixed;
   /* A single-sample tile store already observes the kill above. Passing
    * the same dynamic mask to the store would require a per-sample MRT
    * transaction. Image spills still need their explicit predicate.
    */
   nir_def *store_samples =
      single_sample_coverage_applied && !agx_tilebuffer_spills(&tib)
         ? NULL : write_samples;
   if (write_samples && !store_samples) {
      /* A masked colour store still requires a translucent pass. */
      for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt)
         force_translucent |= tib.logical_format[rt] != PIPE_FORMAT_NONE;
   }
   unsigned rt_spill = key->link.rt_spill_base;
   NIR_PASS(_, b->shader, agx_nir_lower_tilebuffer_with_conversions, &tib,
            colormasks, &rt_spill, store_samples, &force_translucent,
            preserve_zero_mask, key->lower_srgb, key->lower_r11g11b10,
            key->g17p_tile_formats, key->native_epilog);
   const bool native_spills = key->native_spills && agx_tilebuffer_spills(&tib);
   const unsigned spill_generation = native_spills
      ? (key->native_spill_generation ? key->native_spill_generation : 9) : 0;
   if (native_spills) {
      NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_native_emrt_handle,
               nir_metadata_control_flow, (void *)key);
   }
   /* G17G's spilled attachments use its Gen10 descriptors. */
   const unsigned generation =
      native_spills ? spill_generation : (key->rt0_resident && agx_tilebuffer_spills(&tib))
                             ? 10 : 0;
   NIR_PASS(_, b->shader, agx_nir_lower_texture_for_generation, generation);
   NIR_PASS(_, b->shader, agx_nir_lower_multisampled_image_store_for_generation,
            generation);
   if (key->rt0_resident && generation == 10) {
      NIR_PASS(_, b->shader, agx_nir_rebind_spilled_rts);
      NIR_PASS(_, b->shader, nir_shader_intrinsics_pass,
               lower_gen10_texture_handle, nir_metadata_control_flow, NULL);
   }

   /* If the API shader runs once per sample, then the epilog runs once per
    * sample as well, so we need to lower our code to run for a single sample.
    *
    * If the API shader runs once per pixel, then the epilog runs once per
    * pixel. So we run through the monolithic MSAA lowering, which wraps the
    * epilog in the sample loop if needed. This localizes sample shading
    * to the epilog, when sample shading is not used but blending is.
    */
   if (key->link.sample_shading) {
      /* Lower the resulting discards. Done in agx_nir_lower_monolithic_msaa for
       * the pixel shaded path. Must be done before agx_nir_lower_to_per_sample
       * to avoid duplicating tests.
       */
      if (b->shader->info.fs.uses_discard) {
         if (key->m5_coverage)
            NIR_PASS(_, b->shader, agx_nir_lower_sample_mask_m5);
         else
            NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
      }

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_lower_fs_active_samples_to_register);

      /* Ensure the sample ID is preserved in register. We do this late since it
       * has to go in the last block, and the above passes might add control
       * flow when lowering.
       */
      b->cursor = nir_after_impl(b->impl);
      nir_export_agx(b, sample_id, .base = AGX_ABI_FIN_SAMPLE_MASK);
   } else if (!key->defer_monolithic_msaa) {
      /* Zero-mask preservation can introduce tile loads after blend lowering.
       * Those loads must retain each sample's previous value. Pure stores can
       * broadcast the pixel-shaded colour to the live coverage mask without a
       * sample loop. Preserve any sample-shading request already set above. */
      if (key->sample_loop_mrt) {
         bool reads_tile = false;
         nir_foreach_block(block, b->impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_load_local_pixel_agx)
                  reads_tile = true;
            }
         }
         /* A single unwritten attachment also needs its own previous value
          * at every sample. Broadcasting one loaded value would overwrite
          * the other samples even though the API shader writes no colour. */
         b->shader->info.fs.uses_sample_shading |= reads_tile;
      }
      if (key->m5_coverage)
         NIR_PASS(_, b->shader, agx_nir_lower_monolithic_msaa_m5, key->nr_samples);
      else
         NIR_PASS(_, b->shader, agx_nir_lower_monolithic_msaa, key->nr_samples);
   } else if (b->shader->info.fs.uses_discard) {
      /* G17P can keep a pixel-shaded multisample blend as one tile
       * load/store pair carrying the live coverage register. Lower the
       * alpha discard while the epilog's test intrinsics are still available;
       * doing it after this function leaves a newly introduced test query
       * past divergence analysis. */
      if (key->m5_coverage)
         NIR_PASS(_, b->shader, agx_nir_lower_sample_mask_m5);
      else
         NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
   }

   /* M5 writes whole-pixel depth or stencil alone in one covered-sample
    * DEPTHF loop, as native G17G does. A sample-rate epilog writes its own
    * sample instead. A blended epilog's write qualifies once the MSAA
    * lowering has kept it ahead of the sample loop; one left inside that
    * loop does not run once and stays as it is. */
   if (key->m5_coverage && !key->link.sample_shading &&
       !key->defer_monolithic_msaa)
      NIR_PASS(_, b->shader, agx_nir_lower_m5_zs_sample_loop, key->nr_samples);

   if (native_spills) {
      /* The generated loop has exactly two or four iterations. G17P's MRT
       * transaction is specialized for those sample groups; do not let the
       * general cost heuristic leave a dynamic loop around the spill crawl. */
      foreach_list_typed(nir_cf_node, node, node, &b->impl->body) {
         if (node->type == nir_cf_node_loop)
            nir_cf_node_as_loop(node)->control = nir_loop_control_unroll;
      }
      NIR_PASS(_, b->shader, nir_opt_constant_folding);
      NIR_PASS(_, b->shader, agx_nir_lower_g17p_indexed_samplers, false);
      NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_native_emrt_uniforms,
               nir_metadata_control_flow, (void *)key);
   }

   /* Finally, lower uniforms according to our ABI */
   nir_shader_intrinsics_pass(b->shader, lower_non_monolithic_uniforms,
                              nir_metadata_control_flow, (void *)key);

   /* There is no shader part after the epilog, so we're always responsible for
    * running our own tests, unless the fragment shader forced early tests.
    */
   NIR_PASS(_, b->shader, lower_tests_zs, !key->link.already_ran_zs);

   b->shader->info.io_lowered = true;
   b->shader->info.fs.uses_fbfetch_output |= force_translucent;
   b->shader->info.fs.uses_sample_shading = key->link.sample_shading;
   /* The epilog runs inside the interlock its main opened by waiting for
    * the read class alone (a framebuffer fetch); agx_compile.c seeds the
    * epilog's waited classes from this. */
   b->shader->info.fs.pixel_interlock_ordered =
      (key->main_pixel_held & 0xc) == 0x8;
}

void
agx_nir_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   if (!key->g17p_tile_formats && !key->native_epilog)
      agx_nir_fs_epilog_agx2(b, key);
   else
      agx_nir_fs_epilog_native(b, key);
}

/*
 * AGX_G17P_UNIT_RANGE_BLEND.
 *
 * Apple's inline blend for the throughput fragment shader has no source
 * clamp: its compiler knows float(v & 63) / 255 lies in [0, 1]
 * (native fragment 0x106..0x1b4). Honeykrisp's split epilog cannot see the
 * main shader's values, so nir_lower_blend clamps every UNORM blend source
 * (four f32add.sat per RGBA8 pixel). The main shader proves the bound here;
 * the epilog key carries it; the epilog then drops fsat of the exported
 * colour. Only fsat whose operand is the exported colour itself is removed.
 */
static bool
unit_bound(nir_shader *s, struct hash_table *ht, nir_scalar v, double *lo,
           double *hi, unsigned depth)
{
   if (depth > 24)
      return false;

   v = nir_scalar_chase_movs(v);
   if (nir_scalar_is_const(v)) {
      if (v.def->bit_size != 16 && v.def->bit_size != 32)
         return false;
      const double c = nir_scalar_as_float(v);
      if (isnan(c) || isinf(c))
         return false;
      *lo = *hi = c;
      return true;
   }

   if (!nir_scalar_is_alu(v))
      return false;

   const nir_op op = nir_scalar_alu_op(v);
   double a0, a1, b0, b1, c0, c1;
   switch (op) {
   case nir_op_fsat:
      *lo = 0.0;
      *hi = 1.0;
      return true;

   case nir_op_u2f32:
   case nir_op_u2f16: {
      const uint32_t ub = nir_unsigned_upper_bound(
         s, ht, nir_scalar_chase_alu_src(v, 0));
      /* Keep the conversion exact so the bound is not rounded up. */
      if (ub > (op == nir_op_u2f16 ? 2048u : (1u << 24)))
         return false;
      *lo = 0.0;
      *hi = ub;
      return true;
   }

   case nir_op_f2f32:
   case nir_op_f2f16:
      return unit_bound(s, ht, nir_scalar_chase_alu_src(v, 0), lo, hi,
                        depth + 1);

   case nir_op_fmul:
   case nir_op_fmulz:
   case nir_op_ffma:
   case nir_op_ffmaz: {
      if (!unit_bound(s, ht, nir_scalar_chase_alu_src(v, 0), &a0, &a1,
                      depth + 1) ||
          !unit_bound(s, ht, nir_scalar_chase_alu_src(v, 1), &b0, &b1,
                      depth + 1))
         return false;
      /* Only non-negative factors: the product bound is then its corners
       * and no sign or zero-times-infinity case arises. */
      if (a0 < 0.0 || b0 < 0.0)
         return false;
      *lo = a0 * b0;
      *hi = a1 * b1;
      if (op == nir_op_ffma || op == nir_op_ffmaz) {
         if (!unit_bound(s, ht, nir_scalar_chase_alu_src(v, 2), &c0, &c1,
                         depth + 1))
            return false;
         *lo += c0;
         *hi += c1;
      }
      return true;
   }

   case nir_op_fadd:
      if (!unit_bound(s, ht, nir_scalar_chase_alu_src(v, 0), &a0, &a1,
                      depth + 1) ||
          !unit_bound(s, ht, nir_scalar_chase_alu_src(v, 1), &b0, &b1,
                      depth + 1))
         return false;
      *lo = a0 + b0;
      *hi = a1 + b1;
      return true;

   case nir_op_fmin:
   case nir_op_fmax:
      if (!unit_bound(s, ht, nir_scalar_chase_alu_src(v, 0), &a0, &a1,
                      depth + 1) ||
          !unit_bound(s, ht, nir_scalar_chase_alu_src(v, 1), &b0, &b1,
                      depth + 1))
         return false;
      *lo = op == nir_op_fmin ? MIN2(a0, b0) : MAX2(a0, b0);
      *hi = op == nir_op_fmin ? MIN2(a1, b1) : MAX2(a1, b1);
      return true;

   case nir_op_bcsel:
      if (!unit_bound(s, ht, nir_scalar_chase_alu_src(v, 1), &a0, &a1,
                      depth + 1) ||
          !unit_bound(s, ht, nir_scalar_chase_alu_src(v, 2), &b0, &b1,
                      depth + 1))
         return false;
      *lo = MIN2(a0, b0);
      *hi = MAX2(a1, b1);
      return true;

   default:
      return false;
   }
}

bool
agx_nir_fs_colour_outputs_unit_range(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   struct hash_table *ht = _mesa_pointer_hash_table_create(NULL);
   bool any = false, all = true;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (sem.location != FRAG_RESULT_COLOR &&
                sem.location < FRAG_RESULT_DATA0)
               continue;

            nir_def *vec = intr->src[0].ssa;
            if (vec->bit_size != 16 && vec->bit_size != 32) {
               all = false;
               continue;
            }
            u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
               double lo, hi;
               any = true;
               /* Margin covers f16/f32 rounding of each bounded step. */
               if (!unit_bound(s, ht, nir_get_scalar(vec, c), &lo, &hi, 0) ||
                   lo < 0.0 || hi > 0.99)
                  all = false;
            }
         }
      }
   }
   _mesa_hash_table_destroy(ht, NULL);
   return any && all;
}

static bool
drop_unit_range_source_clamp(nir_builder *b, nir_alu_instr *alu,
                             UNUSED void *data)
{
   if (alu->op != nir_op_fsat)
      return false;

   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      nir_scalar src = nir_scalar_chase_movs(
         nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[c]));
      if (!nir_def_is_intrinsic(src.def) ||
          nir_scalar_intrinsic_op(src) != nir_intrinsic_load_exported_agx)
         return false;
   }

   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, nir_mov_alu(b, alu->src[0],
                                          alu->def.num_components));
   return true;
}

static bool
drop_unit_range_source_clamps(nir_shader *s)
{
   return nir_shader_alu_pass(s, drop_unit_range_source_clamp,
                              nir_metadata_control_flow, NULL);
}

struct lower_epilog_ctx {
   struct agx_fs_epilog_link_info *info;
   nir_variable *masked_samples;
   bool preserve_early_demote;
   bool preserve_memory_demote;
   bool preserve_alpha_one;
};

static bool
lower_output_to_epilog(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_epilog_ctx *ctx = data;
   struct agx_fs_epilog_link_info *info = ctx->info;

   if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
      assert(nir_src_as_uint(intr->src[0]) == 0xff && "msaa not yet lowered");
      b->cursor = nir_instr_remove(&intr->instr);

      unsigned base = nir_intrinsic_base(intr);
      info->write_z = !!(base & 1);
      info->write_s = !!(base & 2);

      if (info->write_z)
         nir_export_agx(b, intr->src[1].ssa, .base = AGX_ABI_FOUT_Z);

      if (info->write_s)
         nir_export_agx(b, intr->src[2].ssa, .base = AGX_ABI_FOUT_S);

      return true;
   }

   if (intr->intrinsic == nir_intrinsic_demote_samples &&
       (b->shader->info.fs.early_fragment_tests ||
        (ctx->preserve_memory_demote && b->shader->info.writes_memory) ||
        (ctx->preserve_early_demote &&
         !b->shader->info.fs.uses_sample_shading &&
         (b->shader->info.outputs_written &
          BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK))))) {

      /* G17P carries the discard mask across the split for both early and
       * late tests. LSTCOL consumes it directly, and late DEPTHF must use it
       * too: MOVMSK does not make a split epilog's reconstructed depth word
       * retain the main shader's discard predicate. Late per-pixel SampleMask
       * outputs must also combine with alpha coverage before triggering tests.
       * Keep ordinary discard-only shaders on their existing lowering. */

      if (!ctx->masked_samples) {
         b->cursor = nir_before_impl(nir_shader_get_entrypoint(b->shader));

         ctx->masked_samples =
            nir_local_variable_create(b->impl, glsl_uint16_t_type(), NULL);

         nir_store_var(b, ctx->masked_samples, nir_imm_intN_t(b, 0xFF, 16),
                       nir_component_mask(1));
      }

      b->cursor = nir_before_instr(&intr->instr);

      nir_def *mask = nir_load_var(b, ctx->masked_samples);
      /* The demoted-sample mask arrives as 16-bit from the sample-mask
       * lowering but as 32-bit when it comes straight from gl_SampleMask
       * (fragment_operations.early_fragment.samplemask_*). Coverage is eight
       * bits, so narrow it before combining with the 16-bit accumulator. */
      nir_def *demoted = nir_u2uN(b, intr->src[0].ssa, 16);
      nir_def *mask_2 = nir_ixor(b, demoted, nir_imm_intN_t(b, 0xff, 16));

      mask = nir_iand(b, mask, mask_2);
      nir_store_var(b, ctx->masked_samples, mask, nir_component_mask(1));

      nir_instr_remove(&intr->instr);
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   /* Fix up gl_FragColor */
   if (sem.location == FRAG_RESULT_COLOR) {
      sem.location = FRAG_RESULT_DATA0;
      info->broadcast_rt0 = true;
   }

   /* We don't use the epilog for sample mask writes */
   if (sem.location < FRAG_RESULT_DATA0)
      return false;

   /* Determine the ABI location. Dual source blending aliases a second
    * render target, so get that out of the way now.
    */
   unsigned loc = sem.location - FRAG_RESULT_DATA0;
   loc += nir_src_as_uint(intr->src[1]);

   if (sem.dual_source_blend_index) {
      assert(loc == 0);
      loc = 1;
   }

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *vec = intr->src[0].ssa;

   info->loc_written |= BITFIELD_BIT(loc);

   if (vec->bit_size == 32)
      info->size_32 |= BITFIELD_BIT(loc);
   else
      assert(vec->bit_size == 16);

   uint32_t one_f = (vec->bit_size == 32 ? fui(1.0) : _mesa_float_to_half(1.0));
   unsigned comp = nir_intrinsic_component(intr);

   /* Record the whole write mask, including a W the loc0_w_1 case elides
    * below: the epilog materializes that component into the loaded vector, so
    * the slot has to be there to insert it into. */
   info->colour_comps |=
      (uint32_t)((nir_intrinsic_write_mask(intr) << comp) & 0xf) << (4 * loc);

   u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
      nir_scalar s = nir_scalar_resolved(vec, c);
      /* G17P's mixed-width MRT epilog cannot materialize the elided alpha
       * through AGX2's 16-bit immediate move. Keep it in the ordinary export
       * ABI, which is also what native multi-target shaders use. */
      if (!ctx->preserve_alpha_one && loc == 0 && c == 3 &&
          nir_scalar_is_const(s) &&
          nir_scalar_as_uint(s) == one_f) {

         info->loc0_w_1 = true;
      } else {
         unsigned stride = vec->bit_size / 16;

         nir_export_agx(b, nir_channel(b, vec, c),
                        .base = AGX_ABI_FOUT_COLOUR(loc) + (comp + c) * stride);
      }
   }

   return true;
}

static bool
lower_fs_output_to_epilog(nir_shader *s, struct agx_fs_epilog_link_info *out,
                          bool preserve_early_demote, bool preserve_memory_demote,
                          bool preserve_alpha_one)
{
   struct lower_epilog_ctx ctx = {
      .info = out,
      .preserve_early_demote = preserve_early_demote,
      .preserve_memory_demote = preserve_memory_demote,
      .preserve_alpha_one = preserve_alpha_one,
   };

   nir_shader_intrinsics_pass(s, lower_output_to_epilog,
                              nir_metadata_control_flow, &ctx);

   if (ctx.masked_samples) {
      nir_builder b =
         nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));

      nir_export_agx(&b, nir_load_var(&b, ctx.masked_samples),
                     .base = AGX_ABI_FOUT_WRITE_SAMPLES);
      out->sample_mask_after_force_early = true;

      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, s, nir_lower_vars_to_ssa);
         NIR_PASS(progress, s, nir_opt_dce);
      } while (progress);
   }

   out->sample_shading = s->info.fs.uses_sample_shading;
   return true;
}

bool
agx_nir_lower_fs_output_to_epilog(nir_shader *s,
                                  struct agx_fs_epilog_link_info *out)
{
   return lower_fs_output_to_epilog(s, out, false, false, false);
}

bool
agx_nir_lower_fs_output_to_epilog_g17p(
   nir_shader *s, struct agx_fs_epilog_link_info *out)
{
   return agx_nir_lower_fs_output_to_epilog_for_device(s, AGX_CHIP_G17P, out);
}

bool
agx_nir_lower_fs_output_to_epilog_for_device(
   nir_shader *s, enum agx_chip chip, struct agx_fs_epilog_link_info *out)
{
   return lower_fs_output_to_epilog(
      s, out, agx_chip_native_or_m5(chip),
      agx_chip_native_agx3(chip), agx_chip_native_agx3(chip));
}

static bool
pack_colour_export(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_g17p_colour_layout *layout = data;

   if (intr->intrinsic != nir_intrinsic_export_agx &&
       intr->intrinsic != nir_intrinsic_load_exported_agx)
      return false;

   /* Depth, stencil and the masks live below the first colour. */
   const unsigned base = nir_intrinsic_base(intr);
   if (base < AGX_ABI_FOUT_COLOUR(0))
      return false;

   const unsigned loc = (base - AGX_ABI_FOUT_COLOUR(0)) / 8;
   assert(loc < 8 && "colour exports name a render target");

   /* Only the location's base moves; a component's place inside it does not */
   const unsigned half = base - AGX_ABI_FOUT_COLOUR(loc);
   nir_intrinsic_set_base(
      intr, agx_g17p_sample_colour_base(layout, 0, loc) + half);

   return true;
}

/*
 * Move a fragment's colour exports onto the packed layout, which gives each
 * location the components it writes instead of the ABI's four. Record it, so
 * the epilog reads the registers the main fragment wrote.
 */
bool
agx_nir_pack_g17p_colour_exports(nir_shader *s,
                                 struct agx_fs_epilog_link_info *link)
{
   struct agx_g17p_colour_layout layout;
   agx_g17p_colour_layout_init(&layout, link->loc_written, link->size_32,
                               link->colour_comps, true);

   link->packed_colour = true;

   return nir_shader_intrinsics_pass(s, pack_colour_export,
                                     nir_metadata_control_flow, &layout);
}

/*
 * What lower_fs_output_to_epilog() would record for this shader, without
 * lowering it. A caller deciding whether a fragment's per-sample exports can
 * fit the register file needs the layout before the split happens.
 *
 * Anything this cannot account for is reported as a full four-component
 * 32-bit location, which is the ABI's own slot, so the answer is never
 * smaller than what the lowering goes on to produce.
 */
void
agx_nir_gather_fs_colour_exports(const nir_shader *s_, uint8_t *loc_written,
                                 uint8_t *size_32, uint32_t *colour_comps)
{
   nir_shader *s = (nir_shader *)s_;

   *loc_written = 0;
   *size_32 = 0;
   *colour_comps = 0;

   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;

            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            unsigned location = sem.location;

            if (location == FRAG_RESULT_COLOR)
               location = FRAG_RESULT_DATA0;

            /* Sample mask and depth/stencil do not go through the epilog */
            if (location < FRAG_RESULT_DATA0)
               continue;

            if (!nir_src_is_const(intr->src[1])) {
               /* The lowering requires a constant index. Give up exactly */
               *loc_written = BITFIELD_MASK(8);
               *size_32 = BITFIELD_MASK(8);
               *colour_comps = ~0u;
               return;
            }

            unsigned loc = (location - FRAG_RESULT_DATA0) +
                           nir_src_as_uint(intr->src[1]);

            if (sem.dual_source_blend_index)
               loc = 1;

            if (loc >= 8)
               continue;

            *loc_written |= BITFIELD_BIT(loc);

            if (intr->src[0].ssa->bit_size == 32)
               *size_32 |= BITFIELD_BIT(loc);

            *colour_comps |=
               (uint32_t)((nir_intrinsic_write_mask(intr)
                           << nir_intrinsic_component(intr)) &
                          0xf)
               << (4 * loc);
         }
      }
   }
}


static bool
lower_single_active_sample(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, 1, intr->def.bit_size));
   return true;
}

bool
agx_nir_lower_g17p_1x_sample_shading(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   const bool sample_shading = s->info.fs.uses_sample_shading;
   nir_lower_single_sampled_options options = {
      .lower_sample_mask_in = true,
   };
   bool progress = nir_lower_single_sampled(s, &options);

   /* Gallium lowers sample-rate helper checks before the raster sample count
    * is known. Their active-sample mask names the sample-loop iteration, not
    * hardware coverage. With a single sample the sole iteration is bit0;
    * leaving it dynamic would import a register from a loop we just removed.
    * Keep the separate load_sample_mask in those checks so helper lanes and
    * samples rejected by the hardware still cannot perform side effects. */
   progress |= nir_shader_intrinsics_pass(s, lower_single_active_sample,
                                          nir_metadata_control_flow, NULL);

   /* With one rasterization sample, per-sample and per-pixel invocation are
    * identical. Leaving this set makes agx_fast_link append its AGX2 sample
    * loop header/footer (including 0x0088 STOP) to AGX3 code, whereas G17P's
    * grounded terminator is 0x0000000e. */
   s->info.fs.uses_sample_shading = false;
   return progress || sample_shading;
}

bool
agx_nir_lower_fs_active_samples_to_register(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_active_samples_to_register,
                                     nir_metadata_control_flow, NULL);
}

static bool
agx_nir_lower_stats_fs(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   nir_push_if(b, nir_inot(b, nir_load_helper_invocation(b, 1)));
   nir_def *samples = nir_bit_count(b, nir_load_sample_mask_in(b));
   unsigned query = PIPE_STAT_QUERY_PS_INVOCATIONS;

   nir_def *addr = nir_load_stat_query_address_poly(b, .base = query);
   nir_global_atomic(b, 32, addr, samples, .atomic_op = nir_atomic_op_iadd);

   nir_pop_if(b, NULL);
   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

/* A prolog that does not own Z/S tests only kills its targeted samples.
 * Its generated LIVE complement is disjoint from TARGET, so canonicalize
 * LIVE to zero before source provenance is lost during later lowering. */
static bool
lower_prolog_kill_only(nir_builder *b, nir_intrinsic_instr *intr,
                      UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[1],
                   nir_imm_intN_t(b, 0, intr->src[1].ssa->bit_size));
   return true;
}

void
agx_nir_fs_prolog(nir_builder *b, const void *key_)
{
   const struct agx_fs_prolog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS prolog";

   /* First, insert code for any emulated features */
   if (key->api_sample_mask != 0xff) {
      /* Kill samples that are NOT covered by the mask */
      nir_demote_samples(b, nir_imm_intN_t(b, key->api_sample_mask ^ 0xff, 16));
      b->shader->info.fs.uses_discard = true;
   }

   if (key->statistics) {
      NIR_PASS(_, b->shader, agx_nir_lower_stats_fs);
   }

   if (key->cull_distance_size) {
      NIR_PASS(_, b->shader, agx_nir_lower_cull_distance_fs,
               key->cull_distance_size);
   }

   if (key->polygon_stipple) {
      NIR_PASS(_, b->shader, agx_nir_lower_poly_stipple);
   }

   /* A read-only early-reject prolog has no coverage to change: it only asks
    * the depth/stencil unit to test the complete incoming mask.  Ordinarily a
    * prolog runs tests while lowering its API-mask or cull-distance demotes;
    * make the all-live form explicit so a later shader-part discard can stay
    * deferred without also deferring rejection. */
   if (key->run_zs_tests && !b->shader->info.fs.uses_discard) {
      nir_def *all = nir_imm_intN_t(b, 0xff, 16);
      nir_sample_mask_agx(b, all, all);
   }

   /* Then, lower the prolog */
   NIR_PASS(_, b->shader, agx_nir_lower_discard_zs_emit);
   NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass,
            lower_non_monolithic_uniforms, nir_metadata_control_flow, NULL);
   NIR_PASS(_, b->shader, lower_tests_zs, key->run_zs_tests);
   /* Exact on every generation: without the tests, TARGET and LIVE are
    * disjoint, so each update only kills. The backend (agx_compile.c) can
    * only prove "no Z/S tests" from a constant LIVE; a register LIVE gets the
    * testing form (wait_pix, and the testing MOVMSK on AGX3). This follows
    * from sample_mask semantics, not from a chip. */
   if (!key->run_zs_tests)
      NIR_PASS(_, b->shader, nir_shader_intrinsics_pass,
               lower_prolog_kill_only, nir_metadata_control_flow, NULL);

   b->shader->info.io_lowered = true;
}

void
agx_nir_fs_epilog_g17p(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   if (key->link.sample_shading) {
      /* The main fragment exported every sample's colours. Build the per-pixel
       * epilog, then store and blend each sample from its own colours. */
      struct agx_fs_epilog_key pixel = *key;
      pixel.link.sample_shading = false;
      pixel.defer_monolithic_msaa = true;
      agx_nir_fs_epilog_native(b, &pixel);

      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, key->link.loc_written,
                                  key->link.size_32, key->link.colour_comps,
                                  key->link.packed_colour);

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_unroll_g17p_samples_layout,
               key->nr_samples, &layout);
      bool opt_progress;
      do {
         opt_progress = false;
         NIR_PASS(opt_progress, b->shader, nir_opt_constant_folding);
         NIR_PASS(opt_progress, b->shader, nir_opt_algebraic);
         /* M5 folds dead branches so its sample-rate coverage updates
          * coalesce ahead of the colour stores. Other parts keep their
          * qualified epilog shape. */
         if (key->m5_coverage)
            NIR_PASS(opt_progress, b->shader, nir_opt_dead_cf);
      } while (opt_progress);
      NIR_PASS(_, b->shader, agx_nir_hoist_g17p_zs_writes);
   } else {
      /* G17P's tile operations carry the live coverage register. With
       * alpha-to-coverage, Apple's 2x/4x blended fragments therefore issue
       * one masked tile load/store pair rather than cloning the blend once
       * per sample. Keep the ordinary per-sample lowering for every other
       * pixel-shaded epilog, and for spilled targets: their memory stores
       * address each sample by its ID, which only the sample loop provides. */
      struct agx_fs_epilog_key pixel = *key;
      pixel.defer_monolithic_msaa = key->blend.alpha_to_coverage &&
                                    key->nr_samples > 1 && !key->native_spills;
      agx_nir_fs_epilog_native(b, &pixel);
   }

   NIR_PASS(_, b->shader, agx_nir_lower_g17p_tile_samples, key->nr_samples);
}

/* Both API frontends use the same bounded register ABI for an unrolled main
 * and its epilog. Keep layout selection and the register-file limit together.
 */
bool
agx_nir_lower_g17p_sample_shading_file(nir_shader *nir,
                                       struct agx_fs_epilog_link_info *link,
                                       unsigned samples, unsigned registers,
                                       bool eight_samples)
{
   if (samples != 2 && samples != 4 && !(eight_samples && samples == 8))
      return false;

   struct agx_g17p_colour_layout layout;
   agx_g17p_colour_layout_init(&layout, link->loc_written, link->size_32,
                               link->colour_comps, false);
   if (agx_g17p_sample_exports_end(&layout, samples) > registers) {
      NIR_PASS(_, nir, agx_nir_pack_g17p_colour_exports, link);
      agx_g17p_colour_layout_init(&layout, link->loc_written, link->size_32,
                                  link->colour_comps, true);
   }
   if (agx_g17p_sample_exports_end(&layout, samples) > registers)
      return false;

   NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
   NIR_PASS(_, nir, agx_nir_unroll_g17p_samples_layout, samples, &layout);
   nir->info.fs.uses_sample_shading = false;
   return true;
}

/* G17P: two or four samples within its 192-half register file. */
bool
agx_nir_lower_g17p_sample_shading(nir_shader *nir,
                                  struct agx_fs_epilog_link_info *link,
                                  unsigned samples)
{
   return agx_nir_lower_g17p_sample_shading_file(nir, link, samples,
                                                 AGX3B_NUM_REGS, false);
}
