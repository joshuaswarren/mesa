/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "agx_chip.h"

#define AGX_PPP_HAL300_GRAPHICS_RECORDS 9u
#define AGX_PPP_HAL300_GRAPHICS_BYTES 172u

struct agx_ppp_hal300_graphics_fields {
   uint64_t fragment_program;
   uint64_t varying_table;
   unsigned cf_binding_count;

   uint32_t fragment_ti_spill_B;
   uint32_t fragment_ti_staging_B;
   unsigned cull_mode;

   /* On G18, primary ISP control bit 16 is set when the draw uses the
    * separately bound scissor buffer. */
   bool scissor_enable;
   /* On G18, primary ISP control bit 17 is set for a biased triangle. The bias
    * record itself is separately bound by the depth-bias/scissor block. */
   bool depth_bias_enable;
   bool depth_clip_disable;
   unsigned polygon_mode;
   bool primitive_msaa;
   /* Encoded 4:4 fixed-point line width, with the hardware off-by-one. */
   unsigned line_width;

   bool no_color_attachment;
   bool no_color_writes;
   /*
    * G18G only (the G17G-family parts use no_color_writes): set when colour
    * attachments are bound but the linked fragment program writes nothing
    * to them: no tile store, memory write or coverage write (the linker's
    * no_op). Only the ISP record's last word moves, to the colour-free zero.
    * Native capture 1130 (J873, 26A428): of 39 mixed-format MRT renders, the
    * one whose four colour masks are all empty runs a fragment of only the
    * pixel-transaction bracket and carries zero there, and each of the 38
    * that store a target carries a nonzero word. With 0x0001ffff, G18G
    * replaces every covered pixel of every attachment with zero at end of
    * tile although the fragment stored no lane. The varying count keeps the
    * colour-render position; that render has no varyings, so it witnesses
    * neither placement.
    */
   bool fragment_writes_nothing;

   /* ISP pass type: 0 opaque, 1 translucent, 2 punch through, and on G18G
    * 3 translucent punch through. Binding the fragment program sets all
    * three pass bits in the primary ISP word, and the secondary ISP word
    * keeps them unchanged. */
   unsigned fragment_pass;
   bool disable_tri_merging;

   bool fragment_writes_depth;
   /* Native conservative-depth mode after early-test/side-effect policy. */
   unsigned fragment_depth_mode;
   bool fragment_reads_depth;
   /* 1/W uses the existing position coefficient, with no user UVS word. */
   bool fragment_reads_reciprocal_w;
   bool front_face_ccw;
   bool point_size_output;
   bool layer_output;

   unsigned clip_distance_words;

   enum agx_ppp_hal300_object_type {
      AGX_PPP_HAL300_TRIANGLES = 0,
      AGX_PPP_HAL300_POINTS = 1,
      AGX_PPP_HAL300_LINES = 2,
   } object_type;

   /* G18G's J873 "lines" render (sessions 20260925-a/b) against its colour
    * render in the same session moves exactly the words points move, except
    * that the early-test words gain 0x10000000 where points set 0x40000000;
    * lines export no point size and keep the colour render's UVS record.
    *
    * On G18 the line width goes into both PPP face words, each carrying
    * (width * 16) - 1 in bits 8..17, clamped to 0xff for lines. On G18G an
    * absent field (line_width_valid clear) preserves width 1; an explicit
    * width is admitted only for G18 lines. */
   bool line_width_valid;

   unsigned vertex_varying_components;
   unsigned fragment_varying_components;

   enum agx_ppp_hal300_interpolation {
      AGX_PPP_HAL300_SMOOTH = 0,
      AGX_PPP_HAL300_FLAT = 1,
      AGX_PPP_HAL300_LINEAR = 2,
   } interpolation;

   /* G18 requires 0xc0000 in the primary ISP control whenever stencil is
    * active, even with depth testing enabled. Setting the stencil reference
    * value, for both faces or for each face separately, replaces exactly
    * the low byte of the two face words. */
   bool depth_active;
   uint8_t depth_compare;
   bool depth_write;
   bool stencil_active;

   uint8_t front_stencil_reference;
   uint8_t back_stencil_reference;
   uint32_t front_stencil;
   uint32_t back_stencil;

   /*
    * GL_RASTERIZER_DISCARD, as Metal's rasterizationEnabled = NO with a
    * fragment function bound: the W-clamp/cull-2 record's third word gains bit
    * 17 (0x80 -> 0x00020080). The bit is a pipeline property and every draw
    * with that pipeline carries the word unchanged; the cull record keeps its
    * bit 17 clear. The other half of the state is the VDM vertex-program
    * variant (agx_vdm_state_update_draw_info::rasterization_enabled).
    */
   bool rasterizer_discard;
   /* FRAGMENT_CONTROL bit 21 of the linked fragment: no part writes the
    * tile, so the fragment must not take the visibility tag. */
   bool tag_write_disable;

   /* Checked all-f32 shared VIT/UVS layout, including mixed interpolation.
    *
    * G18G general varyings: R1, R2 and R4 take the linker's words unchanged.
    * R1 is built from the compiler's per-class counts, AGX2 VARYING_COUNTS for
    * 32- and 16-bit varyings, and R4 word 1 is AGX2 OUTPUT_SELECT, vertex
    * half | fragment half; word 4 is the vertex output total. When set,
    * vertex/fragment_varying_components, interpolation, point_size_output and
    * clip_distance_words are ignored. */
   bool general_varyings;
   unsigned cf_register_count;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   uint32_t output_select;
   uint32_t uvs_words;

   /* Cull-2 bit 12: the fragment shader reads a hardware primitive ID
    * (a type-7 iteration in the fragment program sets it). */
   bool needs_primitive_id;

   /* Visibility (occlusion) result: AGX_VISIBILITY_MODE_* (0, 2 counting, 3
    * boolean) and the 64-bit slot index in the render's result buffer. */
   uint8_t visibility_mode;
   uint16_t visibility_index;

   /* G18G only (the G17G-family parts use visibility_mode/index above):
    * Metal's setVisibilityResultMode:offset: on G18 writes mode 0/2/3 into
    * primary ISP bits 14..15 and the 15-bit query slot into the primary
    * record's last word, bits 17..31. Slot units are uint64_t results; the
    * largest query offset is 0x3fff8. Each draw carries both fields as set.
    * Zero defaults preserve the non-query path.
    */
   unsigned occlusion_query_mode;
   unsigned occlusion_query_index;

   /* Cull word bits 7..8 hold the provoking vertex selector plus one: 0 first,
    * 2 last (the J873 captures carry 0x480 for first and 0x580 for last, gears
    * 0x10582). */
   unsigned flat_shading_vertex;

   /* Zero retains the single-viewport default for standalone builders. */
   unsigned viewport_count;
};

#define AGX_PPP_HAL300_STENCIL_KEEP_ALWAYS UINT32_C(0x0e000000)
#define AGX_PPP_HAL300_STENCIL_REPLACE_ALWAYS UINT32_C(0x0e020000)
#define AGX_PPP_HAL300_STENCIL_KEEP_EQUAL UINT32_C(0x04000000)

struct agx_ppp_hal300_graphics_core {
   uint8_t bytes[AGX_PPP_HAL300_GRAPHICS_BYTES];
   uint16_t offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS];
   uint8_t lengths[AGX_PPP_HAL300_GRAPHICS_RECORDS];
};

/* G17G-family parts (agx3c) and G18G. G18G takes its own packer, which reads
 * the fields its J873 captures ground and ignores the G17G-family-only ones
 * (fragment_ti_*, depth_clip_disable, polygon_mode, primitive_msaa,
 * no_color_writes, fragment_depth_mode, fragment_reads_*, layer_output,
 * tag_write_disable, cf_register_count, visibility_*, viewport_count).
 * On failure out is untouched. */
bool agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out);

/* Record index and fragment-program pair layout inside the HAL300 PPP update. */
#define AGX_PPP_HAL300_PROGRAM_RECORD             1
#define AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_OFFSET 16
#define AGX_PPP_HAL300_PROGRAM_RECORD_PAIR_SIZE   8

/* Replace the fragment program address in a core packed by
 * agx_ppp_hal300_graphics_core_pack() for the same chip, keeping the pair's
 * template bits and every other byte, so it matches a full pack of the same
 * fields with only fragment_program changed. Returns false, leaving core
 * untouched, if the address cannot be encoded. */
bool agx_ppp_hal300_graphics_core_patch_program(
   enum agx_chip chip, struct agx_ppp_hal300_graphics_core *core,
   uint64_t fragment_program);
