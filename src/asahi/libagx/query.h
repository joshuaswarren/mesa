/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"

#pragma once

struct libagx_xfb_counter_copy {
   DEVICE(uint32_t) dest[4];
   DEVICE(uint32_t) src[4];
};

struct libagx_imm_write {
   DEVICE(uint32_t) address;
   uint32_t value;
   /* i386 aligns uint64_t to 4 bytes in structs: without this word the host
    * packs 12-byte elements while libagx_write_u32s reads 16-byte ones. */
   uint32_t pad;
};
static_assert(sizeof(struct libagx_imm_write) == 16,
              "struct libagx_imm_write must match the device layout");

/* One timestamp libagx_copy_timestamps copies into another query. */
struct libagx_timestamp_copy {
   DEVICE(uint64_t) dest;
   DEVICE(uint64_t) src;
};
static_assert(sizeof(struct libagx_timestamp_copy) == 16,
              "struct libagx_timestamp_copy must match the device layout");

#define LIBAGX_QUERY_UNAVAILABLE (uint64_t)((int64_t)-1)

#define LIBAGX_QUERY_INDEX_UNUSED ((uint16_t)0xffff)
