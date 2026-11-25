/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded HAL300 PPP depth-bias/scissor records and base block.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH        12u
#define AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH           16u
#define AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_LENGTH 20u

#define AGX_PPP_HAL300_DEPTH_BIAS_BASE_SELECTOR (UINT32_C(1) << 8)
#define AGX_PPP_HAL300_SCISSOR_BASE_SELECTOR    (UINT32_C(1) << 9)
#define AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_HEADER                         \
   (AGX_PPP_HAL300_DEPTH_BIAS_BASE_SELECTOR |                                  \
    AGX_PPP_HAL300_SCISSOR_BASE_SELECTOR)

struct agx_ppp_hal300_depth_bias_record_fields {
   uint32_t depth_bias_raw;
   uint32_t slope_scale_raw;
   uint32_t clamp_raw;
};

struct agx_ppp_hal300_scissor_record_fields {
   uint16_t max_x;
   uint16_t min_x;
   uint16_t max_y;
   uint16_t min_y;
   uint32_t min_z_raw;
   uint32_t max_z_raw;
};

struct agx_ppp_hal300_depth_bias_scissor_bases_fields {
   uint64_t depth_bias_base;
   uint64_t scissor_base;
};

struct agx_ppp_hal300_depth_bias_record {
   uint8_t bytes[AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH];
   uint8_t length;
};

struct agx_ppp_hal300_scissor_record {
   uint8_t bytes[AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH];
   uint8_t length;
};

struct agx_ppp_hal300_depth_bias_scissor_bases {
   uint8_t bytes[AGX_PPP_HAL300_DEPTH_BIAS_SCISSOR_BASES_LENGTH];
   uint8_t length;
};

/*
 * HAL300 PPP depth-bias (3 x f32) and scissor (4 x u16, 2 x f32) records plus
 * the 5-word base-address reference block (header 0x300, followed by 48-bit
 * 4-byte-aligned depth-bias and scissor bases).
 *
 * Admitted: G17G/X (HAL300) and G18G (J873 native renders); closed: G15/G16
 * (legacy packed-index form) and G17P/G17A.
 */
bool agx_ppp_hal300_depth_bias_scissor_core_supported(enum agx_chip chip);

bool agx_ppp_hal300_depth_bias_record_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_bias_record_fields *fields,
   struct agx_ppp_hal300_depth_bias_record *out);

bool agx_ppp_hal300_depth_bias_record_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_bias_record_fields *fields);

bool agx_ppp_hal300_scissor_record_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_scissor_record_fields *fields,
   struct agx_ppp_hal300_scissor_record *out);

bool agx_ppp_hal300_scissor_record_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_scissor_record_fields *fields);

bool agx_ppp_hal300_depth_bias_scissor_bases_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields *fields,
   struct agx_ppp_hal300_depth_bias_scissor_bases *out);

bool agx_ppp_hal300_depth_bias_scissor_bases_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_bias_scissor_bases_fields *fields);

#ifdef __cplusplus
}
#endif
