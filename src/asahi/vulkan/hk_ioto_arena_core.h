/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bump-allocate small G17P Ioto programs (64-byte aligned) out of 128 KiB
 * chunks (matching HK_CMD_BO_SIZE) instead of per-program BOs, which would
 * waste 32 KiB (page + guard) of the 4 GiB USC exec window per program.
 */
#define HK_IOTO_ARENA_CHUNK_SIZE (128u * 1024u)

struct hk_ioto_arena_span {
   /* Byte offset of the program within the chunk. */
   uint32_t offset;

   /* Bump cursor once the program has been placed. */
   uint32_t next_offset;
};

/*
 * Size of the chunk to request so that a program of `program_size` bytes at
 * `align` fits. Returns 0 for an unrepresentable request.
 */
size_t hk_ioto_arena_chunk_size(size_t program_size, uint32_t align);

/*
 * Place `program_size` bytes at `align` in a chunk of `capacity` bytes whose
 * bump cursor is at `cursor`. Returns false -- leaving *span untouched -- when
 * the program does not fit, which is the caller's signal to start a new chunk.
 */
bool hk_ioto_arena_place(uint32_t cursor, uint32_t capacity,
                         size_t program_size, uint32_t align,
                         struct hk_ioto_arena_span *span);

#ifdef __cplusplus
}
#endif
