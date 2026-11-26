/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx3_ioto_uniform_program.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
#include "asahi/lib/agx3_compute_program_core.h"

#define AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE 6144u
#define AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS 128u
/* Largest thread-private spill frame a G17P program publishes, in bytes.
 * applegpu_g17p holding 60 live uint4 temporaries publishes 1584. */
#define AGX3_G17P_MAX_SPILL_B 4096u
/* SPECTPR has a 16-bit count of four-byte units. RT callable stacks can
 * exceed 8 KiB (31 frames of a callable with a local array). Keep admission
 * consistent with agx_ioto_g17p_compiler_spectpr_pack(). */
#define AGX3_G17P_MAX_COMPUTE_SPILL_B (UINT16_MAX * 4u)

struct agx3_g17p_compute_program_desc {
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
   /* Bytes per pixel of the kernel's imageblock, or 0 without one. A compute
    * imageblock is a 32x32 tile of one sample per pixel, the tile a 32x32
    * workgroup stages for its block store. */
   uint16_t imageblock_stride_B;
   struct agx3_cdm_direct_template_sources launch_sources;
   enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1;
   const struct agx_ioto_preshader_desc *preshader;

   /* Optional native constant program, run by the loader after the uniform
    * ranges are loaded, and the inputs it imports through its TPS frame.
    * Its thread-invariant resources belong in launch_sources. */
   struct agx_ioto_constant_program constant_program;
   const struct agx3_cprog_inputs *cprog_inputs;

   /* Optional: generate address-independent bytes (see
    * agx3_ioto_uniform_program_desc::patchable) and record where each range
    * base is materialized. Every relocation lies before the first COPY, at
    * the same offset in the composed program. */
   bool patchable;
   struct agx_ioto_load_relocation *relocations;
   size_t relocation_capacity;
   size_t *relocation_count;
};

struct agx3_g17p_compute_program {
   uint8_t bytes[AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
   uint16_t length;
   struct agx3_compute_program launch;
};

bool agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct agx3_g17p_compute_program_desc *desc,
   struct agx3_g17p_compute_program *program);
