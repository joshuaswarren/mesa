/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx3_ioto_uniform_program.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
#include "asahi/lib/agx_ioto_state_core.h"

#define AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE 6144u
/* Software composer capacity, not a hardware field: the Ioto loader bounds a
 * program by its total load chunks and body size, never by how many ranges
 * produced them. So this only has to cover every range the driver's collector
 * can emit, and a value below that refuses a legal program fail-closed.
 *
 * Honeykrisp's collector is fixed-shape and needs at most seven for a vertex
 * program: VBO bases, VBO clamps, draw parameters, draw ID, indexed
 * texture/sampler heaps, descriptor root, and promoted constants
 * (hk_cmd_buffer.c static-asserts that against
 * HK_AGX3_G17P_GRAPHICS_MAX_RANGES, which aliases this value).
 *
 * The GL driver's collector is not fixed-shape: agx_nir_layout_uniforms()
 * pushes one range per contiguous run of system values of one element size,
 * so the count depends on which sysvals a shader reads. Firefox's WebRender
 * vertex shaders on the J700 need eight or nine -- the texture/sampler
 * pointer pair, VBO bases, VBO clamps, draw parameters, the stage table's
 * texture and UBO bases, the UBO size, two 16-bit root sysvals, and for some
 * variants promoted constants -- and a budget of seven refused every one of
 * them ("no G17P vertex program could be built for this draw"), leaving GPU
 * WebRender black. Thirty-two covers the GL collector's realistic worst case
 * (every root and stage-table field a vertex shader can read, each split by
 * element size) with room to spare; the chunk and body bounds below remain
 * the real limits and are still enforced per program. */
#define AGX3_G17P_GRAPHICS_MAX_RANGES 32u
#define AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS 128u
#define AGX3_G17P_GRAPHICS_ROOT_PREFIX_MAX_SIZE \
   (2u * AGX_IOTO_LOAD_LDIMM_MAX_SIZE)
#define AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE \
   AGX_IOTO_G17P_EXTERNAL_LAUNCHER_MAX_SIZE
#define AGX3_G17P_GRAPHICS_ROOT_BASE_HALF 0u
#define AGX3_G17P_GRAPHICS_ROOT_GPR_DWORDS 2u
/* The launcher carries its 64-bit root through LDSHDR in the final four
 * CProg halves.  External setup is used only when the linked shader does not
 * consume this range; every other shader stays on the qualified inline form. */
#define AGX3_G17P_GRAPHICS_EXTERNAL_ROOT_UNIFORM_HALF 508u

struct agx3_g17p_graphics_program_desc {
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

   /* Persistent setup programs receive their command-buffer-owned root base
    * in r0:r1 from a compact launcher. Each range address is then an offset
    * of a source-VA qword in that root, and setup performs the execution-time
    * second-level dereference. Ordinary programs leave this false and
    * materialize every absolute range themselves. */
   bool preloaded_root;

   /* Shared memory in bytes for a compute, mesh or object launch. On G18G
    * the SPECLM carries it as byte size f18 plus enable f24, the vector
    * Apple's G18 reduction kernel publishes (00d50001... = f18 0x100, f24 1
    * for 256 B). */
   uint32_t threadgroup_memory_size_B;

   /* Address-independent shape (agx3_ioto_uniform_program_desc::patchable):
    * every range materializes its own base with fixed-size LDIMMs, so the
    * recorded relocations can be patched for other addresses. */
   bool patchable;

   /* Executed after resource publication and the terminal state protocol.
    * program_va must name the actual first byte for relative external
    * branches; absolute G17P branches can be copied unchanged to another
    * allocation. The caller owns executable lifetime and launch
    * resource accounting; a zero extent preserves the absent END suffix. */
   struct agx_ioto_constant_program constant_program;
   const struct agx3_cprog_inputs *cprog_inputs;

   const struct agx_ioto_preshader_desc *preshader;

   /* M5: the cluster count the USC window's code-container lead was built
    * for, the kernel's num_clusters_total (agx3_build_m5_container_lead()).
    * The program's profile loader points at that lead's helper-table group
    * (agx3_m5_container_group_offset()); an M5 program is refused without
    * it. Other paths ignore it. */
   unsigned usc_clusters;
};

struct agx3_g17p_graphics_program {
   uint8_t bytes[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool agx3_g17g_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);
bool agx3_g17g_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);
/* G17S-only native TA mesh/object program composition. Shader entry ABI and
 * resources must already be supplied; this does not advertise mesh support. */
bool agx3_g17g_mesh_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   bool object, struct agx3_g17p_graphics_program *program);
bool agx3_g17g_compute_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);
bool agx3_g17g_tile_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   bool store, struct agx3_g17p_graphics_program *program);

/* G18G only. The G18G layer's programs: the G17G-family (agx3c) GPU1 state
 * and resource program, composed with G18G's imageblock admission, as
 * canonical GPU1 output the caller re-encodes as agx3d. The other GPUs keep
 * agx3_g17g_*_program_generate() above. */

/* Scratch-free vertex and fragment programs. */
bool agx3_g18g_graphics_program_generate(
   const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);

/* G18 compute's static TPS frame. Produces the canonical GPU1 program for
 * subsequent agx3d re-encoding; the kernel must own the matching pool before
 * execution. Other stages and G17G retain their separate scratch-free APIs. */
bool agx3_g18g_compute_program_generate(
   const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);

/* G18 vertex/fragment programs with a static TPS frame. Native1054 programs
 * the 16-byte-aligned frame in both SPECTPR words, in four-byte units, for
 * both stages. Canonical GPU1 output for agx3d re-encoding; the caller must
 * report the frame in the render's drm_asahi_g18_render_resources so the
 * kernel sizes the shared TA/3D pool. Tile programs stay scratch-free. */
bool agx3_g18g_graphics_program_generate_tps(
   const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);

/* No-preamble internal tile shaders: EOT has no PIXREL, background has the
 * distinct release mask. Uniform ranges describe the compiled shader ABI. */
bool agx3_g18g_tile_program_generate(
   const struct agx3_g17p_graphics_program_desc *desc, bool store,
   struct agx3_g17p_graphics_program *program);

/* G18G general HAL300 tile shaders (any admitted M5 imageblock shape), emitted
 * as single-cluster G17G canonical GPU1 so agx3_transcode_agx3d_ioto() doubles
 * the SR14 helper base from 0x100 to the witnessed G18G 0x200 base regardless
 * of the host device's cluster count. */
bool agx3_g18g_general_tile_program_generate(
   const struct agx3_g17p_graphics_program_desc *desc, bool store,
   struct agx3_g17p_graphics_program *program);

struct agx3_g17p_graphics_program_template {
   struct agx3_g17p_graphics_program program;
   struct agx_ioto_load_relocation
      relocations[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   uint16_t relocation_count;
   uint16_t constant_suffix_offset;
};



bool agx3_g17g_graphics_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program);

/* G18G: agx3_g18g_graphics_program_generate(), or _tps() for a program with
 * a static TPS frame, recording where it materializes each range's base as
 * agx3_g17g_graphics_program_template_generate() does. Canonical GPU1 output
 * for agx3d re-encoding (agx3_reencode_template_create()). */
bool agx3_g18g_graphics_program_template_generate(
   const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program);

/* Capture the exact address materializations of the normal M5 compute
 * program. Relocations refer to bytes of the final composed program. */
bool agx3_g17g_compute_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program);

bool agx3_g17g_tile_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   bool store, struct agx3_g17p_graphics_program_template *program);

enum agx3_g17p_rt0_bg_op {
   AGX3_G17P_RT0_BG_CLEAR,
   AGX3_G17P_RT0_BG_RELOAD,
   AGX3_G17P_RT0_BG_NONE,
   /* Reload the leading targets and clear the rest. */
   AGX3_G17P_RT0_BG_CLEAR_RELOAD,
};

#define AGX3_G17P_RT0_CLEAR_PUSH_HALFS 12u
#define AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS 14u
#define AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS 18u
#define AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS 20u
#define AGX3_G17P_TRIPLE_CLEAR_PUSH_HALFS 28u
#define AGX3_G17P_QUAD_CLEAR_PUSH_HALFS 36u
/* Colour targets a mixed-format program stores. */
#define AGX3_G17P_MRT_MAX_TARGETS 8u
/* Eight mixed-format targets, each clear value in its own eight halves. */
#define AGX3_G17P_OCTA_CLEAR_PUSH_HALFS 68u

bool agx3_g17p_rt0_bg_resource_spec(enum agx_chip chip,
                                    enum agx3_g17p_rt0_bg_op op,
                                    uint64_t *resource_spec);

bool agx3_g17p_rt0_bg_shader_resources_match(enum agx3_g17p_rt0_bg_op op,
                                             unsigned push_count,
                                             bool uses_txf);

struct agx3_g17p_rt0_bg_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   uint64_t clear_value_va;
   uint64_t texture_pointer_va;
   uint64_t sampler_pointer_va;
   /* Optional x0,y0,x1,y1 at uniform halves8..15 for partial resolves. */
   uint64_t resolve_bounds_va;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   uint16_t staging_halfs;
   uint16_t clear_push_halfs;
   /* CLEAR_RELOAD: uniform half of the first cleared target's value, which
    * clear_value_va addresses. The reloaded targets precede it. */
   uint16_t clear_start_halfs;
   bool staging_known;
   enum agx3_g17p_rt0_bg_op op;
   struct agx_ioto_g17p_imageblock imageblock;
};

struct agx3_g17p_rt0_bg_program {
   uint8_t bytes[AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool agx3_g17p_rt0_bg_program_generate(
   enum agx_chip chip, const struct agx3_g17p_rt0_bg_program_desc *desc,
   struct agx3_g17p_rt0_bg_program *program);

struct agx3_g17p_rt0_eot_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   uint64_t pbe_argument_va;
   uint8_t imageblock_offset_B;
   uint8_t attachment_count;
   uint8_t imageblock_offset1_B;
   uint8_t imageblock_offset2_B;
   uint8_t imageblock_offset3_B;
   uint8_t imageblock_offset4_B;
   uint8_t imageblock_offset5_B;
   uint8_t imageblock_offset6_B;
   uint8_t imageblock_offset7_B;
   /* Two to eight targets of arbitrary formats, each at its own offset. */
   bool mixed_mrt;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   uint16_t staging_halfs;
   struct agx_ioto_g17p_imageblock imageblock;
};

struct agx3_g17p_rt0_eot_program {
   uint8_t bytes[AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE];
   uint16_t length;
};

bool agx3_g17p_rt0_eot_program_generate(
   enum agx_chip chip, const struct agx3_g17p_rt0_eot_program_desc *desc,
   struct agx3_g17p_rt0_eot_program *program);

bool agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);

bool agx3_g17p_graphics_program_bound_with_preshader(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, const struct agx_ioto_preshader_desc *preshader,
   size_t *size);

bool agx3_g17p_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program);

/* Generate an immutable program plus exact LDIMM relocation sites, then copy
 * it for another descriptor whose absolute ranges preserve the same loader
 * cache topology.  Relocation fails closed if any instruction changes size. */
bool agx3_g17p_graphics_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program);

bool agx3_g17p_graphics_program_template_relocate(
   enum agx_chip chip,
   const struct agx3_g17p_graphics_program_desc *template_desc,
   const struct agx3_g17p_graphics_program_desc *target_desc,
   const struct agx3_g17p_graphics_program_template *program,
   void *output, size_t capacity, uint16_t *written);

/* Write directly into command-buffer-private, unpublished storage. Unlike the
 * transactional object API above, output may be modified when this returns
 * false; `written` is unchanged. The caller must not publish the storage until
 * success. */
bool agx3_g17p_graphics_program_write_unpublished(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   void *output, size_t capacity, size_t *written);

/* HAL200 native texture/sampler pointers occupy uniform halves 0..7.  Select
 * the first even eight-half range after every main-shader uniform and rodata
 * range for a fragment blend epilog which also uses that resource prefix.
 * The output is transactional on failure. */
bool agx3_g17p_fragment_blend_constant_base(enum agx_chip chip,
                                             unsigned occupied_halfs,
                                             uint16_t *base);

struct agx3_g17p_graphics_root_prefix {
   uint8_t bytes[AGX3_G17P_GRAPHICS_ROOT_PREFIX_MAX_SIZE];
   uint8_t length;
};

struct agx3_g17p_graphics_external_launcher_template {
   uint8_t bytes[AGX3_G17P_GRAPHICS_EXTERNAL_LAUNCHER_MAX_SIZE];
   uint8_t length;
   uint8_t root_prefix_length;
   uint8_t branch_offset;
};

bool agx3_g17p_graphics_program_desc_bound(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   size_t *size);

bool agx3_g17p_graphics_root_prefix_generate(
   enum agx_chip chip, uint64_t root_va,
   struct agx3_g17p_graphics_root_prefix *prefix);

bool agx3_g17p_graphics_external_setup_suffix_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   void *output, size_t capacity, size_t *written);

bool agx3_g17p_graphics_external_launcher_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   uint64_t program_va, uint64_t root_va, uint64_t setup_va, void *output,
   size_t capacity, size_t *written);

bool agx3_g17p_graphics_external_launcher_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   uint64_t setup_va,
   struct agx3_g17p_graphics_external_launcher_template *launcher);

bool agx3_g17p_graphics_external_launcher_template_relocate(
   enum agx_chip chip,
   const struct agx3_g17p_graphics_external_launcher_template *launcher,
   uint64_t program_va, uint64_t root_va, uint64_t setup_va, void *output,
   size_t capacity, size_t *written);
