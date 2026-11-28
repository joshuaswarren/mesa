/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact bounded join between an AGX3 Ioto compute program and its native
 * direct CDM launch record.
 */

#pragma once

#include "agx3_cdm_direct_launch_semantic_core.h"
#include "agx_ioto_program_core.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_compute_program_desc {
   enum agx_chip chip;
   uint64_t program_va;

   const void *body;
   size_t body_size;
   struct agx_ioto_state_tail tail;
   struct agx_ioto_profile_control profile;
   struct agx_ioto_preshader_desc preshader;

   struct agx3_cdm_direct_template_sources launch_sources;
};

/* Runtime-built G17P Ioto program joined to the native A18 Pro CDM pointer
 * form.  The Ioto bytes remain frontend-owned; this object binds their exact
 * allocation to the launch metadata used below. */
struct agx3_g17p_native_compute_program_desc {
   enum agx_chip chip;
   uint64_t program_va;
   uint64_t shader_va;
   struct agx3_cdm_direct_template_sources launch_sources;
   /* M5: every launch's ordering record takes memory scope
    * (agx_cdm_ordering_memory_scope(), the device's choice). */
   bool ordering_memory_scope;
};

struct agx3_g17s_minimal_compute_program_desc {
   uint64_t program_va;
   uint64_t shader_va;

   /* Exact ProgramVariantESLState UniformData/loadFromUserBuffer inputs for
    * the witnessed root-uniform pointer.  These names preserve the Apple
    * call ABI: destination/count are the values at UniformData +0x30/+0x38,
    * and temporary_base is loadFromUserBuffer's final argument. */
   struct {
      uint32_t user_buffer_binding;
      uint32_t byte_offset;
      uint32_t uniform_data_destination;
      uint32_t uniform_data_count;
      uint32_t temporary_base;
   } root_uniform;

   struct agx3_cdm_direct_template_sources launch_sources;
};

/* Flat, pointer-free state needed after the Ioto bytes have been uploaded. */
struct agx3_compute_program {
   enum agx_chip chip;
   uint64_t program_va;
   uint64_t shader_va;
   struct agx3_cdm_direct_template launch_template;

   /* Set only after a runtime G17P Ioto program has been composed in the
    * allocation named by program_va. */
   bool g17p_native_compute_live_bound;
   /* Set only by agx3_g17g_native_compute_program_bind, the G18G path. */
   bool g17g_native_compute_live_bound;
   bool m5_native_compute_live_bound;
   /* agx3_g17p_native_compute_program_desc::ordering_memory_scope */
   bool ordering_memory_scope;

   /* Only the live G17S minimal-resource preparation path sets this.  Keep
    * research composers and the zero-relocation witness out of a command
    * stream even when their launch metadata is otherwise encodable. */
   bool g17s_minimal_live_admitted;
};

/* Caller-owned CPU mapping of one GPU allocation.  Apple writes the dynamic
 * state-loader program at a 64-byte-aligned mapped address, then puts that
 * mapping's corresponding GPU address into the CDM launch in 64-byte units.
 * Keeping both views in one object prevents a successfully composed program
 * from being launched through a different allocation. */
struct agx3_compute_program_upload {
   void *cpu;
   size_t capacity;
   uint64_t gpu_va;
};

#define AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE \
   (AGX3_CDM_DIRECT_LAUNCH_SIZE + 4u)
#define AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE \
   (AGX3_CDM_DIRECT_LAUNCH_SIZE + 4u)
#define AGX3_G17G_NATIVE_COMPUTE_DISPATCH_SIZE \
   (AGX3_CDM_DIRECT_LAUNCH_SIZE + 4u)
/* Mode 2 consumes both global and local sizes from six indirect u32 words. */
#define AGX3_G17P_NATIVE_INDIRECT_MODE UINT32_C(0x10000000)
/* Six words where the direct record has ten: mode 2 replaces the three global
 * and three local size words with one 48-bit address. */
#define AGX3_G17P_NATIVE_INDIRECT_LAUNCH_SIZE 24u
#define AGX3_G17P_NATIVE_INDIRECT_MAIN_SIZE \
   (AGX3_G17P_NATIVE_INDIRECT_LAUNCH_SIZE + 4u)
#define AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE \
   (AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + \
    AGX3_G17P_NATIVE_INDIRECT_MAIN_SIZE)
/* The pair plus the flush agx_chip_indirect_geometry_flush() places between
 * the helper and the launch. */
#define AGX3_NATIVE_INDIRECT_COMPUTE_DISPATCH_MAX_SIZE \
   (AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE + 4u)

static inline size_t
agx3_native_indirect_compute_dispatch_size(enum agx_chip chip)
{
   return AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE +
          (agx_chip_indirect_geometry_flush(chip) ? 4u : 0u);
}

/* Caller-owned command-stream span.  A successful append writes one exact
 * direct launch followed by the AGX3 encoding's strong flags=5 barrier. */
struct agx3_compute_dispatch_sink {
   uint8_t *data;
   size_t capacity;
   size_t length;
};

struct agx3_compute_direct_grid {
   enum agx3_cdm_direct_grid_source source;
   uint32_t size[3];
   uint32_t local_size[3];
   uint8_t dimensions;

   /* Valid only for the aggregate G15X identities accepted by the lower
    * launch codec.  It must remain false for G17P/A18 Pro. */
   bool g15x_has_multiple_agcs;
};

/* The direct helper runs first and writes geometry. The compact application
 * launch then consumes it. Each program supplies its own launch template. */
struct agx3_g17p_native_indirect_dispatch {
   const struct agx3_compute_program *helper_program;
   uint64_t geometry_va;
};

/*
 * Compose the Ioto program and prepare its native launch state as one
 * transaction.  output, *written and *program are unchanged on every
 * failure.  This does not infer either ShaderInfo value from legacy Mesa USC
 * counts, allocate GPU memory, submit work, or admit an AGX3 device.
 */
bool agx3_compute_program_prepare(const struct agx3_compute_program_desc *desc,
                                  void *output, size_t capacity,
                                  size_t *written,
                                  struct agx3_compute_program *program);

/* The captured M5 direct CDM record uses the same native program-pointer
 * descriptor. Firmware must still deliver this stream through M5 SKSM. */
bool agx3_g17g_native_compute_program_bind(
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program);
bool agx3_g17g_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink);

/* G18G indirect launches. Mode 2 reads six u32 at grid_va: three global thread
 * counts, then three local sizes. Mode 1 reads three threadgroup counts at
 * groups_va and carries the local size in the record. Both append the caller's
 * barrier. */
#define AGX3_G17G_NATIVE_INDIRECT_GROUPS_MODE UINT32_C(0x08000000)
#define AGX3_G17G_NATIVE_INDIRECT_LOCAL_MODE  UINT32_C(0x10000000)
#define AGX3_G17G_NATIVE_INDIRECT_LOCAL_LAUNCH_SIZE  24u
#define AGX3_G17G_NATIVE_INDIRECT_GROUPS_LAUNCH_SIZE 36u
bool agx3_g17g_native_indirect_compute_launch_append(
   const struct agx3_compute_program *program, uint64_t grid_va,
   uint8_t barrier_flags, struct agx3_compute_dispatch_sink *sink);
bool agx3_g17g_native_indirect_groups_compute_launch_append(
   const struct agx3_compute_program *program, uint64_t groups_va,
   const uint32_t local_size[3], uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink);

/* Bind a successfully composed runtime Ioto program to its native direct CDM
 * launch state.  No bytes are generated here. */
bool agx3_g17p_native_compute_program_bind(
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program);

bool agx3_m5_native_compute_program_bind(
   enum agx_chip chip,
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program);

bool agx3_m5_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink);

/* local_size is NULL for an indirect-local grid and non-NULL for group counts. */
bool agx3_m5_native_indirect_dispatch_append(
   const struct agx3_compute_program *program, uint64_t grid_va,
   const uint32_t local_size[3], uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink);

/* Append one direct G17P launch and the flags=0 barrier used by the
 * output-positive native compute stream.  Direct global or threadgroup counts
 * are accepted; indirect dispatch uses a different helper-program contract. */
bool agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

/* Same native dispatch with a caller-selected canonical post-dispatch
 * barrier. The plain entry point preserves the minimal flags-0 form. */
bool agx3_g17p_native_compute_dispatch_append_with_barrier(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink);

/* Append one standalone 24-byte indirect launch and its barrier.  `geometry_va`
 * addresses six u32 the GPU reads for itself: three global THREAD counts then
 * three local sizes.  Use this when the producer of that grid is already its
 * own dispatch earlier in the stream -- GL's geometry and tessellation setup
 * kernels are dispatched that way, behind a full barrier -- so there is no
 * helper program to couple into the same record pair. */
bool agx3_g17p_native_indirect_compute_launch_append(
   const struct agx3_compute_program *program, uint64_t geometry_va,
   uint8_t barrier_flags, struct agx3_compute_dispatch_sink *sink);

/* Append the helper/indirect-application record pair used by G17P indirect
 * compute.  The caller owns the helper program, its resources, and geometry
 * storage; this encoder only publishes their already-established GPU VAs. */
bool agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink);

/* Join the exact 194-byte G17S minimal Ioto witness to its native direct-launch
 * template.  The root-uniform source must match Apple's independently traced
 * buffer(1), offset-zero load into temporary base zero.  Its result is loaded
 * at half24 and copied to CProg half8.  This template form intentionally
 * retains Apple's zero runtime relocations and does not admit a live frontend
 * path. */
bool agx3_g17s_minimal_compute_program_prepare(
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program);

/* Exact mapped-allocation handoff for the G17S witness.  Both cpu and gpu_va
 * must identify the beginning of the same 64-byte-aligned allocation, and
 * gpu_va must equal desc->program_va.  No sibling target is admitted.  The
 * mapped bytes, *written and *program are unchanged on every failure.  This
 * does not patch the live gather or root-uniform addresses. */
bool agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

/* G17S-only live-resource handoff.  This is the exact direct-enqueue contract
 * recovered for the minimal witness: the mapped program allocation is joined
 * to the secondary gather allocation VA and UserComputeArgumentTable GPU VA
 * entry 1 and the constructor-owned absolute UniformData source VA.
 * loadImmediateUserBuffer passes the same kind-3 Register zero to the pointer
 * load and the leading COPY pair, so no separate hardware entry seed is
 * required.  The mapping, *written and *program are unchanged on failure. */
bool agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

bool agx3_g17s_minimal_compute_dispatch_ready(enum agx_chip chip);

/* Append the live-admitted program's canonical 40-byte direct launch and the
 * exact four-byte strong post-dispatch barrier.  sink and sink->length are
 * unchanged on every failure; sibling chips and non-live program provenance
 * are rejected. */
bool agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

/*
 * Encode a canonical 40-byte direct launch whose 48-bit pipeline pointer and
 * template are taken exclusively from a prepared program.  packed is cleared
 * on every failure.
 */
bool agx3_compute_program_pack_direct_launch(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE]);

#ifdef __cplusplus
}
#endif
