/* SPDX-License-Identifier: MIT */
#pragma once

#include "compiler/nir/nir.h"

enum {
   AGX_GEMM_WIDTH = 32,
   AGX_GEMM_SUBGROUPS = 4,
   AGX_GEMM_ROWS_PER_LANE = 8,
   AGX_GEMM_TILE_WORDS = AGX_GEMM_WIDTH * AGX_GEMM_WIDTH,
};

struct agx_gemm_match {
   nir_loop *loop;
   nir_instr *ready;
   nir_instr *consumed;
   uint32_t a_bytes;
   uint32_t b_bytes;
   nir_phi_instr *initial[AGX_GEMM_ROWS_PER_LANE];
   nir_alu_instr *fma[AGX_GEMM_WIDTH][AGX_GEMM_ROWS_PER_LANE];
   nir_phi_instr *exit[AGX_GEMM_ROWS_PER_LANE];
};

/*
 * Match a 32x32 tile of FP32 dot products in explicitly lowered AGX3 shared
 * memory. The caller supplies a key with fixed 32-lane AGX3C subgroups.
 * Success certifies the shared tile, CFG and recurrence. The IR is unchanged
 * apart from analysis metadata.
 */
bool agx_nir_match_gemm(nir_shader *shader, struct agx_gemm_match *match);

/* Apply the conservative shared-memory epilogue to a successful match. */
bool agx_nir_lower_gemm(nir_shader *shader, const struct agx_gemm_match *match);
