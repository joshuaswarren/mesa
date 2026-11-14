/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include "util/simple_mtx.h"
#include "agx3_graphics_program.h"
#include "agx3_transcode.h"

/*
 * G18G re-encodings of the M5 generators' Ioto programs (agx3c to agx3d), and
 * a cache of them: see agx3_reencode_cache.c.
 */

struct agx3_reencode_entry;

struct agx3_reencode_cache {
   simple_mtx_t lock;
   /* Most recently used first */
   struct agx3_reencode_entry *entries;
   unsigned count;
};

#define AGX3_REENCODE_CACHE_INITIALIZER {SIMPLE_MTX_INITIALIZER, NULL, 0}

/* The most entries a cache keeps: the least recently used one goes first */
#define AGX3_REENCODE_CACHE_LIMIT 128

enum agx3_reencode_result {
   /* Not a program the cache takes: re-encode it in full */
   AGX3_REENCODE_UNCACHEABLE,
   /* program is its agx3d form, from an entry */
   AGX3_REENCODE_HIT,
   /* program is its agx3d form, re-encoded in full, and an entry made if
    * the program allows one */
   AGX3_REENCODE_MISS,
   /* The transcoder refused program, which is unchanged (why, at) */
   AGX3_REENCODE_REFUSED,
};

/* Re-encode program as agx3d in place, in full. False, with program
 * unchanged, if the transcoder refuses it or its agx3d form is too long. */
bool agx3_reencode_program(struct agx3_g17p_graphics_program *program,
                           const struct agx3_transcode_ioto *ioto,
                           const char **why, unsigned *at);

/* Re-encode program as agx3d in place through cache */
enum agx3_reencode_result
agx3_reencode_cached(struct agx3_reencode_cache *cache,
                     struct agx3_g17p_graphics_program *program,
                     const struct agx3_transcode_ioto *ioto, const char **why,
                     unsigned *at);

/* Free every entry of cache */
void agx3_reencode_cache_clear(struct agx3_reencode_cache *cache);
