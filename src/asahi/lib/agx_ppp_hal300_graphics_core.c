/* SPDX-License-Identifier: MIT */
#include "agx_ppp_hal300_graphics_core.h"
#include "agx_vdm_state_core.h"
#include <string.h>

/* The j873 G18G packer, unchanged; chip is AGX_CHIP_G18G. */
static bool
agx_ppp_hal300_graphics_core_pack_g18g(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out)
{
   /* Every field below but lines was grounded on G17G. The J873 G18G
    * captures reproduce byte for byte for each shape they contain --
    * triangles, points, lines, the three pass types, depth less/always with
    * writes, stencil control, two- and four-component smooth varyings --
    * including the stencil reference byte recovered from the G18 setters.
    * The shapes grounded
    * only on G17G (flat, noperspective, colour-free, depth output, clip
    * distances, culling and winding) are carried to G18G on the strength of
    * that shared layout, without a G18G capture of their own. */
   if (!agx_chip_hal300_records(chip) || !fields || !out ||
       !fields->fragment_program || fields->cull_mode > 3 ||
       fields->fragment_pass > (chip == AGX_CHIP_G18G ? 3 : 2) ||
       fields->flat_shading_vertex > 2 ||
       (fields->flat_shading_vertex && chip != AGX_CHIP_G18G))
      return false;

   if (fields->general_varyings) {
      /* Field widths: a 7-bit binding count, 8-bit class counts and a 4-byte
       * VIT alignment. OUTPUT_SELECT is held to what Gallium emits: clip
       * planes 0..7, varyings, point size and fragcoord Z. Native597 carries
       * the same G18 records with viewport/layer outputs: viewport is bit 19;
       * layer requires both bits 19 and 20. */
      if (chip != AGX_CHIP_G18G || fields->cf_binding_count > 0x7f ||
          ((fields->varying_counts_32 | fields->varying_counts_16) >> 24) ||
          (fields->output_select & ~UINT32_C(0x003d00ff)) ||
          ((fields->output_select & UINT32_C(0x00180000)) ==
             UINT32_C(0x00100000)) ||
          fields->uvs_words < 4 || fields->uvs_words > 0xff ||
          !fields->cf_binding_count != !fields->varying_table ||
          (fields->varying_table & 3) ||
          fields->varying_table >= (UINT64_C(1) << 48))
         return false;
   } else {
      if (fields->vertex_varying_components > 8 ||
          fields->fragment_varying_components >
             fields->vertex_varying_components ||
          fields->needs_primitive_id)
         return false;

      if (fields->fragment_varying_components) {
         if (!fields->varying_table ||
             (fields->varying_table & 63) ||
             fields->varying_table >= (UINT64_C(1) << 48) ||
             !fields->cf_binding_count || fields->cf_binding_count > 3 ||
             (fields->interpolation != AGX_PPP_HAL300_SMOOTH &&
              fields->cf_binding_count > 2))
            return false;
      } else if (fields->varying_table || fields->cf_binding_count ||
                 fields->interpolation != AGX_PPP_HAL300_SMOOTH) {
         return false;
      }
      if (fields->interpolation > AGX_PPP_HAL300_LINEAR)
         return false;

      /* Only the interpolated colour-free shape is witnessed. */
      if (fields->no_color_attachment &&
          fields->interpolation == AGX_PPP_HAL300_FLAT &&
          fields->fragment_varying_components)
         return false;
   }
   const bool psiz = fields->general_varyings
                        ? (fields->output_select & 0x40000) != 0
                        : fields->point_size_output;

   /* Only the witnessed point shape: Apple's point draw exports a point
    * size, and a point without one has no capture behind it. Lines are
    * captured on G18G only, without a point size. */
   if (fields->object_type > AGX_PPP_HAL300_LINES ||
       (fields->object_type == AGX_PPP_HAL300_POINTS && !psiz) ||
       (fields->object_type == AGX_PPP_HAL300_LINES &&
        (chip != AGX_CHIP_G18G || psiz)))
      return false;

   if (fields->line_width_valid &&
       (chip != AGX_CHIP_G18G ||
        fields->object_type != AGX_PPP_HAL300_LINES || fields->line_width > 0xff))
      return false;

   /* Rasterizer discard is known on G18G only; G17G has no witness. */
   if (fields->rasterizer_discard && chip != AGX_CHIP_G18G)
      return false;
   /* The written-nothing ISP word is witnessed on G18G only. */
   if (fields->fragment_writes_nothing && chip != AGX_CHIP_G18G)
      return false;
   if (fields->depth_bias_enable &&
       (chip != AGX_CHIP_G18G || fields->object_type != AGX_PPP_HAL300_TRIANGLES))
      return false;

   if ((fields->occlusion_query_mode != 0 &&
        fields->occlusion_query_mode != 2 &&
        fields->occlusion_query_mode != 3) ||
       fields->occlusion_query_index >= 32768 ||
       (chip != AGX_CHIP_G18G &&
        (fields->occlusion_query_mode || fields->occlusion_query_index)))
      return false;

   /* never .. always; see the header's decode table. An inactive depth unit
    * carries no comparison of its own. */
   if (fields->depth_compare > 7 ||
       (!fields->depth_active && (fields->depth_compare || fields->depth_write)))
      return false;

   /* Retain the captured bit 18 for depth-disabled draws. On G18, stencil
    * testing independently requires BOTH bits 18 and 19, including when depth
    * testing is enabled. A depth/stencil state with either stencil face
    * enabled selects 0xc0000, and binding that state puts both bits into
    * primary ISP control. Setting only bit 19 silently loses stencil
    * tests/updates on combined depth/stencil draws; stencil-only draws already
    * had bit 18 by accident.
    */
   const uint8_t compare = fields->depth_active ? fields->depth_compare : 7;
   const bool write = fields->depth_active && fields->depth_write;
   const bool depth_inactive = !fields->depth_active;
   const bool points = fields->object_type == AGX_PPP_HAL300_POINTS;
   const bool lines = fields->object_type == AGX_PPP_HAL300_LINES;
   const uint32_t control = 0x00000200 | (points || lines ? 0x04000000 : 0) |
                            (fields->occlusion_query_mode << 14) |
                            (fields->scissor_enable ? 0x00010000 : 0) |
                            (fields->depth_bias_enable ? 0x00020000 : 0) |
                            (depth_inactive ? 0x00040000 : 0) |
                            (fields->stencil_active
                                ? (chip == AGX_CHIP_G18G ? 0x000c0000
                                                        : 0x00080000)
                                : 0);
   const uint32_t line_width = fields->line_width_valid ? fields->line_width : 0xf;
   const uint32_t face = ((uint32_t)compare << 24) |
                         (write ? 0 : 0x00200000) | (line_width << 8);

   /* The ISP may resolve depth before the shader runs only while the shader
    * cannot change it; a fragment that writes gl_FragDepth drops both of
    * these bits. See fragment_writes_depth in the header. */
   const uint32_t early_test =
      (fields->fragment_writes_depth ? 0x07200000 : 0x07e00000) |
      (points ? 0x40000000 : 0) | (lines ? 0x10000000 : 0);

   struct agx_vdm_state_vertex_pipeline_pair program;
   if (!agx_vdm_state_vertex_pipeline_pair_pack(
          chip, fields->fragment_program,
          fields->fragment_writes_depth ? 0x40 : 0, &program))
      return false;

   /* M5 native VDM 0x1000018000 references these payloads at offsets
    * 0x34/3c/44/4c/54/5c/6c/74/84. HAL300 has different selectors from
    * HAL200 and groups front/back face before front/back stencil. The
    * OUTPUT record is exactly five words, with no overlapping face header.
    * Viewport and scissor-base updates are independently emitted by caller.
    */
   /* Native color/flat-depth captures on M5/25G76 independently validate
    * counts32 = 4/0x400, absolute 48-bit VIT addresses, and the D32 LESS/write
    * face group. The VIT count occupies bits16..22 above address bits32..47.
    * The fragment program still uses an Ioto pair, not this address format.
    */
   /* General varyings: R1 is the linker's VARYING_COUNTS pair, R4 word 1
    * its OUTPUT_SELECT and word 4 the vertex output total (Apple builds both
    * from the same per-class counts). */
   const uint32_t counts_32 =
      fields->general_varyings
         ? fields->varying_counts_32
         : fields->fragment_varying_components <<
              (fields->no_color_attachment
                  ? 16
                  : fields->interpolation == AGX_PPP_HAL300_FLAT ? 8 : 0);
   const uint32_t output_select =
      fields->general_varyings
         ? fields->output_select
         : (fields->point_size_output ? 0x40000 : 0) |
              (fields->vertex_varying_components ? 0x10000 : 0) |
              (fields->clip_distance_words ? 0x1 : 0);
   const uint32_t uvs_words =
      fields->general_varyings
         ? fields->uvs_words
         : 4 + (fields->point_size_output ? 1 : 0) +
              fields->clip_distance_words + fields->vertex_varying_components;
   const uint32_t records[][7] = {
      {0x10100000, counts_32,
       fields->general_varyings ? fields->varying_counts_16 : 0, 0, 0},
      {0x01000000, (uint32_t)(fields->varying_table >> 32) |
                      (fields->cf_binding_count << 16),
       (uint32_t)fields->varying_table, 0, 0, 0, 0},
      {0x00000001, 0x000007aa, 0xe2e00000, 0xf0c00000,
       0x00023020, 0x0001ffff},
      {0x0c280000, output_select, 0, 0, uvs_words},
      {0x020000e0, control,
       face | fields->front_stencil_reference,
       face | fields->back_stencil_reference,
       fields->stencil_active ? fields->front_stencil : 0x0e000000,
       fields->stencil_active ? fields->back_stencil : 0x0e000000,
       fields->occlusion_query_index << 17},
      /* On HAL300 (M5) the fragment pass type lives in bits 29..31 of this
       * secondary record's word1, and stays there unchanged across
       * render-state emission. They do not belong in the primary face record,
       * unlike the HAL200 layout. */
      {0x02000062, 0x00000200 | (fields->fragment_pass << 29) |
                      (fields->disable_tri_merging || points || lines
                          ? 0x04000000
                          : 0),
       early_test, early_test,
       fields->no_color_attachment || fields->fragment_writes_nothing
          ? 0 : 0x0001ffff},
      /* W clamp and cull-2; bit 17 of the cull-2 word is rasterizer
       * discard (see agx_ppp_hal300_graphics_fields::rasterizer_discard)
       * and bit 12 a hardware primitive ID read by the fragment shader. */
      {0x00840002, 0,
       (fields->rasterizer_discard ? 0x00020080 : 0x80) |
          (fields->needs_primitive_id ? 0x1000 : 0)},
      /* Bits 7..8 are the provoking vertex selector plus one. */
      {0x00800000, 0x400 | ((fields->flat_shading_vertex + 1) << 7) |
                      fields->cull_mode |
                      ((uint32_t)fields->front_face_ccw << 16)},
      {0x00000400, 0, 0},
   };
   const uint8_t lengths[] = {20, 28, 24, 20, 28, 20, 12, 8, 12};
   struct agx_ppp_hal300_graphics_core staged = {0};
   unsigned offset = 0;
   for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
      staged.offsets[i] = offset;
      staged.lengths[i] = lengths[i];
      for (unsigned j = 0; j < lengths[i]; ++j)
         staged.bytes[offset + j] = records[i][j / 4] >> (8 * (j % 4));
      offset += lengths[i];
   }

   /* Native fragment record uses the same full 48-bit Ioto pair as VDM
    * vertex state. The captured 01000000/00005c1c points to 0x10000170700.
    * Metal's cull mode and front-facing winding map to cull bits 0..1 and
    * winding bit 16 above. */
   memcpy(staged.bytes + staged.offsets[AGX_PPP_HAL300_PROGRAM_RECORD] +
             AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_OFFSET,
          program.bytes, AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_SIZE);
   *out = staged;
   return true;
}

bool
agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out)
{
   if (chip == AGX_CHIP_G18G)
      return agx_ppp_hal300_graphics_core_pack_g18g(chip, fields, out);

   if (!agx_chip_g17_m5(chip) || !fields || !out ||
       !fields->fragment_program || fields->cull_mode > 3 ||
       fields->clip_distance_words > 8 || fields->viewport_count > 16 ||
       fields->fragment_pass > 3 || fields->flat_shading_vertex > 2)
      return false;

   const uint64_t ti_spill_quads =
      ((uint64_t)fields->fragment_ti_spill_B + 3) >> 2;
   const uint64_t ti_staging_quads =
      ((((uint64_t)fields->fragment_ti_staging_B + 3) >> 2) + 1) &
      ~UINT64_C(1);
   if (ti_spill_quads > 0x3ffff || ti_staging_quads > 0xffe ||
       ((ti_spill_quads || ti_staging_quads) && !agx_chip_g17_m5(chip)))
      return false;

   if (fields->general_varyings) {
      /* HAL300's native VSStateConfig and VIT producer retain the shared
       * all-f32 UVS/CF ABI: 128 total UVS words and 127 coefficients. */
      const unsigned smooth = fields->varying_counts_32 & 0xff;
      const unsigned flat = (fields->varying_counts_32 >> 8) & 0xff;
      const unsigned linear = (fields->varying_counts_32 >> 16) & 0xff;
      const uint32_t expected_output_select =
         (fields->point_size_output ? 0x40000 : 0) |
         (fields->layer_output ? 0x180000 : 0) |
         ((fields->vertex_varying_components || fields->cf_binding_count)
             ? 0x10000 : 0) |
         (fields->fragment_reads_depth ? 0x200000 : 0) |
         ((UINT32_C(1) << fields->clip_distance_words) - 1);
      if (fields->vertex_varying_components > 124 ||
          fields->fragment_varying_components >
             fields->vertex_varying_components ||
          fields->varying_counts_16 || (fields->varying_counts_32 >> 24) ||
          smooth + flat + linear != fields->vertex_varying_components ||
          fields->uvs_words > 128 ||
          fields->uvs_words != 4 + fields->vertex_varying_components +
             fields->point_size_output + fields->layer_output +
             fields->clip_distance_words ||
          fields->output_select != expected_output_select ||
          !fields->cf_binding_count || fields->cf_binding_count > 127 ||
          fields->cf_register_count < fields->cf_binding_count ||
          fields->cf_register_count > 127 ||
          !fields->varying_table || (fields->varying_table & 63) ||
          fields->varying_table >= (UINT64_C(1) << 48))
         return false;
   } else {
      if (fields->vertex_varying_components >
             ((fields->rasterizer_discard ||
               !fields->fragment_varying_components) ? 124 : 8) ||
          4 + fields->point_size_output + fields->layer_output +
             fields->clip_distance_words + fields->vertex_varying_components > 128 ||
          fields->fragment_varying_components >
             fields->vertex_varying_components || fields->needs_primitive_id)
         return false;

      if (fields->fragment_varying_components || fields->fragment_reads_depth ||
          fields->fragment_reads_reciprocal_w) {
         if (!fields->varying_table ||
             (fields->varying_table & 63) ||
             fields->varying_table >= (UINT64_C(1) << 48) ||
             !fields->cf_binding_count || fields->cf_binding_count > 3 ||
             (fields->interpolation != AGX_PPP_HAL300_SMOOTH &&
              fields->cf_binding_count > 2 &&
              !fields->fragment_reads_reciprocal_w))
            return false;
      } else if (fields->varying_table || fields->cf_binding_count ||
                 fields->interpolation != AGX_PPP_HAL300_SMOOTH) {
         return false;
      }
      if (fields->interpolation > AGX_PPP_HAL300_LINEAR)
         return false;

      if (fields->no_color_attachment &&
          fields->interpolation == AGX_PPP_HAL300_FLAT &&
          fields->fragment_varying_components)
         return false;
   }
   const bool psiz = fields->general_varyings
                        ? (fields->output_select & 0x40000) != 0
                        : fields->point_size_output;

   if (fields->object_type > AGX_PPP_HAL300_LINES ||
       (fields->object_type == AGX_PPP_HAL300_POINTS && !psiz &&
        !fields->rasterizer_discard))
      return false;

   if ((fields->visibility_mode != 0 && fields->visibility_mode != 2 &&
        fields->visibility_mode != 3) ||
       fields->visibility_index >= (1u << 15))
      return false;

   if (fields->line_width > 0xff || fields->polygon_mode > 2 ||
       fields->fragment_depth_mode > 3 || fields->depth_compare > 7 ||
       (!fields->depth_active && (fields->depth_compare || fields->depth_write)))
      return false;

   const uint8_t compare = fields->depth_active ? fields->depth_compare : 7;
   const bool write = fields->depth_active && fields->depth_write;
   const bool depth_inactive = !fields->depth_active;
   const bool points = fields->object_type == AGX_PPP_HAL300_POINTS;
   const bool lines = fields->object_type == AGX_PPP_HAL300_LINES;
   const uint32_t control = 0x00000200 | (points || lines ? 0x04000000 : 0) |
                            (fields->scissor_enable ? 0x00010000 : 0) |
                            (fields->depth_bias_enable ? 0x00020000 : 0) |
                            ((depth_inactive || fields->stencil_active) ? 0x00040000 : 0) |
                            (fields->stencil_active ? 0x00080000 : 0) |
                            /* The native visibility-result-mode setter
                             * sets bits 15:14 of this word (0, 2 counting,
                             * 3 boolean) and offset / 8 into bits 31:17 of
                             * the word 0x14 bytes after it. */
                            ((uint32_t)fields->visibility_mode << 14);
   const uint32_t visibility_query =
      fields->visibility_mode ? (uint32_t)fields->visibility_index << 17 : 0;
   const uint32_t face = ((uint32_t)compare << 24) |
                         (write ? 0 : 0x00200000) | (fields->line_width << 8) |
                         (fields->polygon_mode << 18);

   /* On HAL300 a draw with no fragment program disables tag writes, clears
    * fragment output mask and selects depth mode 3. Rasterizer discard
    * selects that inert fragment state while the VDM still runs the VS. */
   const uint32_t early_test =
      0x07200000 |
      ((fields->rasterizer_discard ? 3 : fields->fragment_depth_mode) << 22) |
      (points ? 0x40000000 : 0) | (lines ? 0x10000000 : 0);

   struct agx_vdm_state_vertex_pipeline_pair program;

   const uint32_t program_template = agx_chip_g17_m5(chip)
      ? (uint32_t)(ti_spill_quads << 3)
      : fields->fragment_writes_depth ? 0x40 : 0;
   if (!agx_vdm_state_vertex_pipeline_pair_pack(
          chip, fields->fragment_program, program_template, &program))
      return false;

   const uint32_t counts_32 =
      fields->general_varyings
         ? fields->varying_counts_32
         : fields->fragment_varying_components <<
              (fields->no_color_attachment
                  ? 16
                  : fields->interpolation == AGX_PPP_HAL300_FLAT ? 8 : 0);
   const uint32_t output_select =
      fields->general_varyings
         ? fields->output_select
         : (fields->point_size_output ? 0x40000 : 0) |
              (fields->layer_output ? 0x180000 : 0) |
              /* System Z/1-W bindings also require coefficient production.
               * Keep the shared linker's OUTPUT_SELECT.varyings contract. */
              ((fields->vertex_varying_components || fields->cf_binding_count)
                  ? 0x10000 : 0) |
              (fields->fragment_reads_depth ? 0x200000 : 0) |
              ((UINT32_C(1) << fields->clip_distance_words) - 1);
   const uint32_t uvs_words =
      fields->general_varyings
         ? fields->uvs_words
         : 4 + (fields->point_size_output ? 1 : 0) +
              (fields->layer_output ? 1 : 0) +
              fields->clip_distance_words + fields->vertex_varying_components;
   const uint32_t records[][7] = {
      {0x10100000, counts_32,
       fields->general_varyings ? fields->varying_counts_16 : 0, 0, 0},
      {0x01000000, (uint32_t)(fields->varying_table >> 32) |
                      (fields->cf_binding_count << 16),
       (uint32_t)fields->varying_table, (uint32_t)ti_staging_quads, 0, 0, 0},
      {0x00000001, 0x000007aa, 0xe2e00000, 0xf0c00000,
       0x00023020, 0x0001ffff},
      {0x0c280000, output_select, 0,
       fields->viewport_count ? fields->viewport_count - 1 : 0, uvs_words},
      {0x020000e0, control,
       face | fields->front_stencil_reference,
       face | fields->back_stencil_reference,
       fields->stencil_active ? fields->front_stencil : 0x0e000000,
       fields->stencil_active ? fields->back_stencil : 0x0e000000,
       visibility_query},

      /* Bit 21 is the tag-write disable that a draw with no fragment
       * program selects; a linked fragment that writes no tile needs it too. */
      {0x02000062, 0x00000200 |
                      (fields->rasterizer_discard ? 0x00200000
                                                : fields->fragment_pass << 29) |
                      (fields->tag_write_disable ? 0x00200000 : 0) |
                      (fields->disable_tri_merging || fields->polygon_mode || points || lines
                          ? 0x04000000
                          : 0),
       early_test, early_test,
       (fields->no_color_attachment || fields->no_color_writes ||
        fields->rasterizer_discard)
          ? 0 : 0x0001ffff},

      {0x00840002, 0,
       (fields->rasterizer_discard ? 0x00020080 : 0x80) |
          (fields->needs_primitive_id ? 0x1000 : 0)},

      {0x00800000, (fields->depth_clip_disable ? 0x800 : 0x400) |
                      ((fields->flat_shading_vertex + 1) << 7) |
                      fields->cull_mode |
                      ((uint32_t)fields->front_face_ccw << 16) |
                      (fields->primitive_msaa ? 0x8000 : 0)},
      {0x00000400, 0, 0},
   };
   const uint8_t lengths[] = {20, 28, 24, 20, 28, 20, 12, 8, 12};
   struct agx_ppp_hal300_graphics_core staged = {0};
   unsigned offset = 0;
   for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
      staged.offsets[i] = offset;
      staged.lengths[i] = lengths[i];
      for (unsigned j = 0; j < lengths[i]; ++j)
         staged.bytes[offset + j] = records[i][j / 4] >> (8 * (j % 4));
      offset += lengths[i];
   }

   memcpy(staged.bytes + staged.offsets[AGX_PPP_HAL300_PROGRAM_RECORD] +
             AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_OFFSET,
          program.bytes, AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_SIZE);
   *out = staged;
   return true;
}

bool
agx_ppp_hal300_graphics_core_patch_program(
   enum agx_chip chip, struct agx_ppp_hal300_graphics_core *core,
   uint64_t fragment_program)
{
   if (!core || !fragment_program)
      return false;

   /* The pair's template bits belong to the layout the core was packed with
    * (the fragment TI spill on the G17G family, the depth-write bit on G18G),
    * so keep the ones already in the record and replace only the address. */
   uint8_t *pair_bytes = core->bytes +
                         core->offsets[AGX_PPP_HAL300_PROGRAM_RECORD] +
                         AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_OFFSET;
   uint64_t old_program;
   uint32_t raw_template;
   struct agx_vdm_state_vertex_pipeline_pair program;
   if (!agx_vdm_state_vertex_pipeline_pair_unpack(
          chip, pair_bytes, AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_SIZE,
          &old_program, &raw_template) ||
       !agx_vdm_state_vertex_pipeline_pair_pack(chip, fragment_program,
                                                raw_template, &program))
      return false;

   _Static_assert(
      sizeof(program.bytes) == AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_SIZE,
      "HAL300 fragment program pair size mismatch");
   memcpy(pair_bytes, program.bytes, AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_SIZE);
   return true;
}
