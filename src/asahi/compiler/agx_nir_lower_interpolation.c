/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/shader_enums.h"
#include "agx_compile.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * In AGX, the values of fragment shader inputs are represented as coefficient
 * vectors <A, B, C>, which are dotted with <x, y, 1> to perform interpolation.
 * x and y are relative to the tile. In other words, A and B are the
 * screen-space partial derivatives of the input, and C is the value at the
 * corner of the tile.
 *
 * For some interpolation modes, the dot product happens in the iterator
 * hardware. Other modes are implemented in this file, by lowering to math on
 * the coefficient vectors.
 */

/* XXX: It's not clear what this is for, but seems necessary */
static nir_def *
cf_valid(nir_builder *b, nir_def *cf)
{
   nir_def *bit = nir_ieq_imm(b, nir_iand_imm(b, nir_channel(b, cf, 0), 1), 0);

   /* XXX: Apple's compiler actually checks that the significand is nonzero and
    * the exponent is 0 or 1. This is probably a typo -- it doesn't make any
    * logical sense.  Presumably they just meant to check for denorms, so let's
    * do that. Either way the tests pass.
    */
   nir_def *cf01 = nir_trim_vector(b, cf, 2);

   /* Reduce the per-channel check to one boolean: interpolate_at_offset
    * selects a scalar with it. A vec2 condition made bcsel produce a vec2,
    * which nir_vec hid for wider loads but a single-component
    * interpolateAtSample/Offset result stored it as-is and failed NIR
    * validation (pipeline.*.misc.interpolate_at_sample_no_sample_shading).
    */
   return nir_ior(b, bit, nir_ball(b, nir_fisnormal(b, cf01)));
}

static nir_def *
cf_constant_marker(nir_builder *b, nir_def *cf)
{
   /* Native HAL200 constant planes store unscaled C, with a nonzero
    * significand and exponent 0 or 1 in A. Test the native marker directly
    * so an ordinary coefficient with its low bit set remains a plane. */
   nir_def *a = nir_channel(b, cf, 0);
   nir_def *exponent_small =
      nir_ult_imm(b, nir_iand_imm(b, a, 0x7f800000), 0x01000000);
   nir_def *significand = nir_ine_imm(b, nir_iand_imm(b, a, 0x007fffff), 0);
   return nir_iand(b, exponent_small, significand);
}

static nir_def *
interpolate_at_offset(nir_builder *b, nir_def *cf, nir_def *offset,
                      bool perspective, bool native)
{
   /* Get the coordinate of the pixel within the tile.
    *
    * On G17P the pixel-coordinate special register disagrees with the
    * interpolated fragment position once the framebuffer is multisampled, so
    * every offset interpolation lands somewhere other than the pixel it is
    * being evaluated for.  Measured on hardware: sourcing the coordinate from
    * the register fails all 28 cases of
    * dEQP-VK.pipeline.*.multisample_interpolation.offset_interpolate_at_pixel_center
    * at 2x and 4x, and sourcing it from the interpolated position passes all
    * 28 with no other multisample interpolation case changing.  Only that
    * family compares this path against a hardware-interpolated value; the
    * others compare two results that share the error, which is why they never
    * caught it.
    *
    * What the register actually holds under multisampling is NOT established.
    * Sample-unit readings were tested and refuted: halving X, halving Y and
    * halving both still fail all 28 cases at both sample counts, so it is not
    * a power-of-two scale.  Until it is understood, take the coordinate from
    * the position varying, which is the value the hardware interpolator
    * itself agrees with.  Gated on the same flag the native constant-plane
    * marker uses, which hk sets for AGX_CHIP_G17P; AGX2 is conformant on the
    * register path and keeps it, as do the host-test callers.
    */
   nir_def *tile_offs;
   if (native) {
      nir_def *fc = nir_trim_vector(b, nir_load_frag_coord(b), 2);
      tile_offs = nir_u2u16(b, nir_umod_imm(b, nir_f2u32(b, nir_ffloor(b, fc)), 32));
   } else {
      nir_def *pixel_coords = nir_load_pixel_coord(b);
      tile_offs = nir_umod_imm(b, pixel_coords, 32);
   }

   /* Convert to float, getting the center of the pixel */
   nir_def *center = nir_fadd_imm(b, nir_u2f32(b, tile_offs), 0.5);

   /* Calculate the location to interpolate. offset is defined relative to the
    * center of the pixel and is a float.
    */
   nir_def *pos = nir_fadd(b, center, nir_f2f32(b, offset));

   /* Interpolate with the given coefficients */
   nir_def *interp = nir_ffma_weak(b, nir_channel(b, pos, 1), nir_channel(b, cf, 1),
                                   nir_channel(b, cf, 2));

   interp = nir_ffma_weak(b, nir_channel(b, pos, 0), nir_channel(b, cf, 0), interp);

   /* Divide by RHW. This load will be lowered recursively. */
   if (perspective) {
      nir_def *bary = nir_load_barycentric_at_offset(
         b, 32, offset, .interp_mode = INTERP_MODE_NOPERSPECTIVE);

      nir_def *rhw = nir_load_interpolated_input(
         b, 1, 32, bary, nir_imm_int(b, 0), .component = 3,
         .io_semantics = {
            .location = VARYING_SLOT_POS,
            .num_slots = 1,
         });

      interp = nir_fdiv(b, interp, rhw);
   }

   if (native)
      return nir_bcsel(b, cf_constant_marker(b, cf), nir_channel(b, cf, 2),
                       interp);

   /* Replace invalid interpolations with the constant channel */
   return nir_bcsel(b, cf_valid(b, cf), interp, nir_channel(b, cf, 2));
}

static nir_def *
interpolate_flat(nir_builder *b, nir_def *coefficients)
{
   /* Same value anywhere, so just take the constant (affine) component. For
    * triangle fans with the first provoking vertex, the CF layout is slightly
    * different. I am unsure why, but Apple does the same and the bcsel is
    * required for corrctness.
    */
   return nir_bcsel(b, nir_load_is_first_fan_agx(b),
                    nir_channel(b, coefficients, 1),
                    nir_channel(b, coefficients, 2));
}

static enum glsl_interp_mode
interp_mode_for_load(nir_intrinsic_instr *load)
{
   if (load->intrinsic == nir_intrinsic_load_input)
      return INTERP_MODE_FLAT;
   else
      return nir_intrinsic_interp_mode(nir_src_as_intrinsic(load->src[0]));
}

static bool
needs_lower(const nir_instr *instr, UNUSED const void *_)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);

   /* at_offset barycentrics need to be lowered */
   if (load->intrinsic == nir_intrinsic_load_interpolated_input) {
      return (nir_src_as_intrinsic(load->src[0])->intrinsic ==
              nir_intrinsic_load_barycentric_at_offset);
   }

   /* Flat shading always lowered */
   return (load->intrinsic == nir_intrinsic_load_input);
}

static nir_def *
interpolate_channel(nir_builder *b, nir_intrinsic_instr *load,
                    unsigned channel, bool layer_as_varying)
{
   nir_def *coefficients = nir_load_coefficients_agx(
      b, nir_get_io_offset_src(load)->ssa,
      .component = nir_intrinsic_component(load) + channel,
      .interp_mode = interp_mode_for_load(load),
      .io_semantics = nir_intrinsic_io_semantics(load));

   if (load->intrinsic == nir_intrinsic_load_input) {
      nir_def *flat;

      if (nir_intrinsic_io_semantics(load).location == VARYING_SLOT_LAYER &&
          !layer_as_varying)
         flat = nir_load_layer_id(b);
      else
         flat = interpolate_flat(b, coefficients);

      /* A coefficient register is 32-bit, but storageInputOutput16 permits a
       * logical 16-bit fragment input.  agx_nir_lower_uvs.c widened the value
       * type-correctly before ST_VARY, so narrow it back the same way here.
       * The interpolated path does its own narrowing in agx_compile.c; this is
       * the flat half of the same contract.
       */
      if (flat->bit_size != load->def.bit_size) {
         assert(load->def.bit_size == 16 && flat->bit_size == 32);

         flat = nir_convert_to_bit_size(
            b, flat, nir_alu_type_get_base_type(nir_intrinsic_dest_type(load)),
            load->def.bit_size);
      }

      return flat;
   } else {
      nir_intrinsic_instr *bary = nir_src_as_intrinsic(load->src[0]);

      nir_def *interp = interpolate_at_offset(
         b, coefficients, bary->src[0].ssa,
         nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE,
         layer_as_varying);

      return nir_f2fN(b, interp, load->def.bit_size);
   }
}

static nir_def *
lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   const bool layer_as_varying = *(const bool *)data;

   /* Each component is loaded separated */
   nir_def *values[NIR_MAX_VEC_COMPONENTS] = {NULL};
   for (unsigned i = 0; i < intr->def.num_components; ++i) {
      values[i] = interpolate_channel(b, intr, i, layer_as_varying);
   }

   return nir_vec(b, values, intr->def.num_components);
}

bool
agx_nir_lower_interpolation(nir_shader *s, bool layer_as_varying)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   return nir_shader_lower_instructions(s, needs_lower, lower,
                                        &layer_as_varying);
}
