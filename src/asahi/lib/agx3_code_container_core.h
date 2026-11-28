/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* G18 uses four AGX3D helper tables in a larger leading block. */
#define AGX3_G18G_CONTAINER_LEAD_SIZE 0x640u
bool agx3_build_g18g_container_lead(void *image, size_t capacity);

/* Write the 0x40-byte program immediately before a G17P container entry.
 * Apple's HAL200 compiler emits distinct vertex/compute, plain-fragment, and
 * sampled-fragment programs.  sampled_fragment is valid only for fragment. */
#define AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE 0x40u
void agx3_build_container_constant_program(
   uint8_t image[AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE], bool fragment,
   bool sampled_fragment);

/* A CODE block has its size word 0x80 bytes before the main entry and a
 * 0x40-byte constant program immediately before it. Blocks can be packed in
 * a shared code allocation; the independent 0x340-byte allocation lead is
 * deliberately not part of this helper. */
#define AGX3_CODE_BLOCK_HEADER_SIZE 0x40u
#define AGX3_CODE_BLOCK_ENTRY_OFFSET 0x80u
bool agx3_build_m5_container_lead(void *image, size_t capacity,
                                  unsigned clusters);

/* The byte offset, from the USC window base, of the helper-table group an M5
 * stage calls in the lead agx3_build_m5_container_lead() builds for this many
 * clusters: one 0x100-byte table per cluster in each of two groups, every
 * stage but vertex at clusters * 0x100 and vertex at twice that. The
 * profile loader of every M5 program (agx_ioto_m5_resource_program_compose_
 * classes()) points there, so the lead and the loaders take the cluster
 * count the kernel reports from one place. 0 for a count the lead refuses.
 */
static inline uint32_t
agx3_m5_container_group_offset(unsigned clusters, bool vertex)
{
   if (!clusters || clusters > (UINT32_MAX - 0x40u) / 0x300u)
      return 0;
   return (vertex ? 2u : 1u) * clusters * 0x100u;
}

size_t agx3_code_block_size(size_t main_span);
/* The fragment stop constant program in the G18G (agx3d) encoding. */
void agx3_build_container_stop_program_agx3d(
   uint8_t image[AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE]);

bool agx3_build_code_block(void *image, size_t capacity, const void *main,
                          size_t main_size, size_t main_span,
                          const uint8_t *constant_program,
                          bool fragment);
