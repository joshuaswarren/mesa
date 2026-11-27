/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/* First byte of a shader binary that its own code upload must contain.
 *
 * Draws and dispatches run a shader's main program from its fast-linked
 * copies (hk_fast_link()), never from the shader's own upload, which only
 * serves the constant program (preamble) and the promoted constants
 * (rodata). The compiler places rodata before the main and the preamble
 * after it, so without rodata everything before the preamble is a main that
 * is never executed from there. Returns binary_size when nothing is needed,
 * otherwise an offset rounded down to the upload's alignment, so every kept
 * byte keeps its alignment.
 */
static inline uint32_t
hk_shader_upload_start(uint32_t binary_size, bool has_preamble,
                       uint32_t preamble_offset, uint32_t rodata_offset,
                       uint32_t rodata_size_16, uint32_t alignment)
{
   uint32_t start = binary_size;

   if (has_preamble && preamble_offset < start)
      start = preamble_offset;
   if (rodata_size_16 && rodata_offset < start)
      start = rodata_offset;

   if (start == binary_size)
      return binary_size;

   return start - (start % alignment);
}
