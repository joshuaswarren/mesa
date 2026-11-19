/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Honeykrisp bridge for the exact bounded AGX3 Ioto program composer.
 */

#pragma once

#include "asahi/compiler/agx3_ioto_uniform_program.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx_ioto_program_core.h"
#include "util/format/u_format.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct agx_tilebuffer_layout;

bool hk_agx3_format_needs_shader_resolve(enum pipe_format format);

/* Describe a nonempty compiler preamble uploaded with its whole binary.
 * Checks both CPU and executable extents. The caller keeps the immutable CPU
 * binary alive during composition and the BO alive through execution. This
 * selects the address-independent G17P branch, but does not qualify the
 * preamble's register/resource ABI. Leaves output unchanged on failure. */
bool hk_agx3_g17p_constant_program_from_binary(
   const void *binary, size_t binary_size, size_t offset, size_t size,
   uint64_t allocation_va, size_t allocation_size,
   struct agx_ioto_constant_program *output);

struct hk_agx3_ioto_program_desc {
   uint64_t program_va;
   const void *body;
   size_t body_size;
   struct agx_ioto_state_tail tail;
   uint16_t profile_payload0;
   uint16_t profile_payload1;
};

bool hk_agx3_ioto_program_generate(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   void *output, size_t capacity, size_t *written);

bool hk_agx3_ioto_program_generate_no_pixrel(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written);

struct hk_agx3_g17p_graphics_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   const struct agx3_ioto_uniform_range *uniform_ranges;
   size_t uniform_range_count;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   uint32_t compiler_d;
   bool fragment;
   bool writes_sample_mask;
   uint8_t colour_classes;
   struct agx_ioto_g17p_imageblock imageblock;
   bool preloaded_root;
   bool patchable;
   struct agx_ioto_constant_program constant_program;
   const struct agx3_cprog_inputs *cprog_inputs;
};

#define HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE 6144u

/*
 * The number of independently addressed uniform ranges a single graphics
 * program can publish. Aliases the composer's capacity so the driver and
 * agx3_g17p_graphics_program_bound() can never disagree about it.
 *
 * The worst case is a vertex program, which can request all nineteen:
 *
 *   1. vertex attribute bases              nr_attribs != 0
 *   2. vertex attribute clamps             nr_attribs != 0
 *   3. draw parameters                     uses_base_param
 *   4. draw ID                             uses_draw_id
 *   5. texture/sampler heap pointers       agx3_indexed_sampler_valid
 *   6. root descriptor                     shader/prolog root || set_count
 *   7. promoted constants                  rodata.size_16 != 0
 *   8-19. uniform DMA spans, two each      agx3_uniform_dma_count
 *
 * The combined-sampler range is not in that list: it is fragment-only, and
 * a fragment program's own worst case is seventeen. hk_cmd_buffer.c static-asserts
 * this derivation against the collector.
 */
/* Honeykrisp has a fixed nineteen-range collector and a matching cache key.
 * Gallium needs a larger budget in the shared program composer. */
#define HK_AGX3_G17P_GRAPHICS_MAX_RANGES 19u

bool hk_agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);

bool hk_agx3_g17p_graphics_program_desc_bound(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   size_t *size);

struct hk_agx3_g17p_graphics_program {
   uint8_t bytes[HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool hk_agx3_g17g_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);
bool hk_agx3_g17g_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program);

bool hk_agx3_g17p_graphics_program_template_generate(
   enum agx_chip chip,
   const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program);

bool hk_agx3_g17p_graphics_program_template_relocate(
   enum agx_chip chip,
   const struct hk_agx3_g17p_graphics_program_desc *template_desc,
   const struct hk_agx3_g17p_graphics_program_desc *target_desc,
   const struct agx3_g17p_graphics_program_template *program,
   void *output, size_t capacity, uint16_t *written);

struct hk_agx3_g17p_combined_sampler_heap {
   uint64_t descriptor_set_va;
   uint64_t sampler_heap_va;
   uint32_t texture_offset_B;
   uint16_t sampler_index; /* Raw G17P heap index; zero is reserved. */
   uint16_t sampler_count;
   uint16_t sampler_element_size_B;
};

/* Resolve two raw table pointers for the bounded texture0/sampler0 fragment
 * ABI. Upload these qwords into command-owned memory, then load its 16 bytes
 * into uniform halves 0..7. Descriptors themselves remain in their tables;
 * copying inline descriptor bytes to those uniforms is not this ABI.
 */
bool hk_agx3_g17p_combined_sampler_pointers(
   const struct hk_agx3_g17p_combined_sampler_heap *heap,
   uint64_t pointers[2]);

struct hk_agx3_g17p_sampled_storage_heap {
   struct hk_agx3_g17p_combined_sampler_heap sampled;
   uint64_t pbe_descriptor_set_va;
   uint64_t write_control_va;
   uint32_t pbe_offset_B;
};

/* Resolve Apple's exact four-slot compute texture0/sample0 -> write-texture1
 * tuple: sampled texture, sampler, destination PBE, write-control pointer. */
bool hk_agx3_g17p_sampled_storage_pointers(
   const struct hk_agx3_g17p_sampled_storage_heap *heap,
   uint64_t pointers[4]);

struct hk_agx3_g17p_indexed_storage_heap {
   uint64_t texture_heap_va;
   uint64_t sampler_heap_va;
   uint64_t pbe_descriptor_set_va;
   uint64_t write_control_va;
   uint32_t pbe_offset_B;
};

bool hk_agx3_g17p_indexed_storage_pointers(
   const struct hk_agx3_g17p_indexed_storage_heap *heap,
   uint64_t pointers[4]);

static inline bool
hk_agx3_g17p_graphics_needs_root(bool shader_uses_root,
                                 bool prolog_uses_root, uint8_t set_count)
{
   return shader_uses_root || prolog_uses_root || set_count != 0;
}

/* Load one root descriptor followed by every descriptor-set pointer named by
 * the shader's pipeline layout. */
bool hk_agx3_g17p_root_range(uint16_t start_halfs, uint8_t set_count,
                             uint64_t root_va,
                             struct agx3_ioto_uniform_range *range);

/* Publish promoted constants through G17P's dword-granular Ioto loader.
 * Compiler binaries pad the rodata blob before the aligned code entry, so an
 * odd final half may be covered by loading the following padding half. */
bool hk_agx3_g17p_rodata_range(uint16_t base_halfs, uint16_t size_halfs,
                               uint64_t bo_va, uint32_t offset_B,
                               uint64_t bo_size_B,
                               struct agx3_ioto_uniform_range *range);

/* Load the four fragment blend constants at an even uniform-half base. The
 * standard ABI uses half4; HAL200 may relocate the epilog range when native
 * texture/sampler table pointers occupy halves0..7. */
bool hk_agx3_g17p_fragment_blend_constants_range(
   uint16_t start_halfs, uint64_t blend_constants_va,
   struct agx3_ioto_uniform_range *range);

/* Build the bounded G17P vertex or fragment Ioto program.  A zero-range
 * fragment uses the valid ordinary-FS PIXREL suffix; a zero-range vertex uses
 * the distinct compute/vertex profile suffix. Up to six dword-aligned
 * absolute uniform ranges add the CProg loads used by vertex attributes,
 * draw parameters, descriptor roots, promoted constants, or the bounded
 * native texture/sampler pair above. */
bool hk_agx3_g17p_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program);

/* Classify the ordinary non-multiview, one-layer fast path. */
bool hk_agx3_g17p_single_layer_rendering(uint32_t view_mask,
                                        uint32_t layer_count);

/* Admit the bounded single-target, layered and exactly witnessed two-target
 * physical tilebuffer layouts. */
bool hk_agx3_g17p_rt0_tile_layout_supported(
   struct agx_tilebuffer_layout *tib);

/* Admit the unresolved multisample colour store. Requires the caller to have
 * confirmed the named one-boot hardware probe: the end-of-tile program and
 * the multisample PBE descriptor are both byte-witnessed from Apple's own
 * encoders, but no G17P silicon has executed the pair. */
bool hk_agx3_g17p_multisample_store_supported(struct agx_tilebuffer_layout *tib,
                                              unsigned color_attachment_count,
                                              bool store_source,
                                              unsigned target_samples);

bool hk_agx3_g17p_direct_color_resolve_supported(
   struct agx_tilebuffer_layout *tib, unsigned color_attachment_count,
   bool has_resolve_target, bool average_resolve, bool store_source);

#define HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE 6144u

#define HK_AGX3_G17P_MAX_ROOT_SETS            16u

/* Resolve a packed precompiled-kernel argument block to the dword-granular
 * upload and uniform-range sizes required by G17P's Ioto loader. */
bool hk_agx3_g17p_precompiled_data_layout(size_t data_size_B,
                                          uint32_t *upload_size_B,
                                          uint16_t *uniform_size_halfs);

struct hk_agx3_g17p_compute_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   const struct agx3_ioto_uniform_range *uniform_ranges;
   size_t uniform_range_count;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   uint32_t compiler_d;
   uint32_t threadgroup_memory_size_B;
   struct agx3_cdm_direct_template_sources launch_sources;

   /* Which program kind the LoadShader record announces.  Zero, the default,
    * is the direct-dispatch kind used by every direct launch.  A native
    * indirect pair announces two different kinds -- see
    * AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1 and AGX3_G17P_GRID_HELPER_LDSHDR_F1 --
    * so both records of that pair must set this explicitly.  Only the three
    * witnessed values are accepted; anything else fails the compose. */
   enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1;

   /* Optional native constant program and its TPS inputs. */
   struct agx_ioto_constant_program constant_program;
   const struct agx3_cprog_inputs *cprog_inputs;
};

struct hk_agx3_g17p_compute_program {
   uint8_t bytes[HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
   uint16_t length;
   struct agx3_compute_program launch;
};

bool hk_agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_compute_program_desc *desc,
   struct hk_agx3_g17p_compute_program *program);

bool hk_agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

bool hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink);

/* Byte bridge for the native main/grid-helper indirect record pair.  Helper
 * program and resource construction remain caller-owned. */
bool hk_agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink);

bool hk_agx3_g17s_minimal_compute_program_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program);

bool hk_agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

bool hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

bool hk_agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

#ifdef __cplusplus
}
#endif
