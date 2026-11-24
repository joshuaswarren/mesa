/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 device memory encoders (`lddev`/`stdev`):
 * address = base64 (uniform quad or GPR pair) + (index << shift) +
 * sext16(byte_offset). Loads carry a scoreboard slot and wait mask; stores
 * signal no slot.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Element width. 64-bit is two 32-bit elements with mask 3; the caller splits
 * it, so only 8/16/32 appear here. */
enum agx3_mem_width {
   AGX3_MEM_W8 = 0,
   AGX3_MEM_W16 = 1,
   AGX3_MEM_W32 = 2,
};

enum agx3_mem_index {
   AGX3_MEM_IDX_IMM = 0, /* f23 = 1 */
   AGX3_MEM_IDX_REG = 1, /* f23 = 2 */
};

#define AGX3_MEM_MAX_LENGTH 16u

struct agx3_mem_desc {
   uint16_t data;   /* dest (load) / value (store) register, half units */
   enum agx3_mem_width width;
   uint8_t mask;    /* 1..15 element enables */
   uint8_t shift;   /* 0..4 index scale */

   bool base_is_uniform;
   uint16_t base;   /* uniform half (4-aligned) or GPR half (2-aligned) */
   bool base_kill;  /* GPR base only */

   enum agx3_mem_index idx_kind;
   /* Immediate index: either the signed byte offset (byte form) or a scaled
    * element index (scaled form). */
   bool idx_scaled;
   int32_t byte_off;  /* byte form: -32768..32767 */
   uint32_t elem_idx; /* scaled form: 0..65535 */
   /* Register index, plus byte_off added after scaling it. */
   uint16_t idx_reg;
   bool idx_signed;
   bool idx_kill;

   uint8_t slot;       /* load only: 1..6 (agx3b) / 1..8 (agx3c) */
   uint16_t wait_mask; /* bit i sets wait slot i */

   /* Apple marks the final LDDEV of a program, and only that one: AGX3b with
    * f79=2 (vertex corpus), AGX3c with f82=2, which only the 14-byte ep2/ep5
    * forms carry (every metal-nt G17G kernel, vertex, fragment and preamble
    * in gpu/m5-*-reference; a two-load kernel emits the 8-byte form then the
    * marked one). No store witness carries either. */
   bool last_load;
   /* Atomic/coherent access qualifier. Apple's AGX3 encoders expose this as
    * f77 on loads and f75 on stores. */
   bool coherent;

   /* AGX3b f79=2. Apple's vertex corpus sets this on the final LDDEV and only
    * the final LDDEV; no AGX3c or store witness carries it. */
   bool agx3b_last_load;
};

bool agx3_mem_encode_load(enum agx_isa_encoding enc,
                          const struct agx3_mem_desc *d,
                          uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len);

/* Encode one exact Apple LDDEV encoding point.  AGX3b exposes only ep0;
 * AGX3c exposes ep0..ep5.  This is the form-explicit primitive used by
 * command-stream builders which have already reproduced Apple's canonical
 * form selection. */
bool agx3_mem_encode_load_form(enum agx_isa_encoding enc,
                               const struct agx3_mem_desc *d, uint8_t ep,
                               uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len);
bool agx3_mem_encode_store(enum agx_isa_encoding enc,
                           const struct agx3_mem_desc *d,
                           uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif
