/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 subgroup shuffle, ballot, and reduction encoders. Unsupported operand
 * kinds, conditions, widths, prefix operations, and marker profiles fail closed.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_SUBGROUP_MAX_LENGTH 16u

enum agx3_subgroup_shuffle_op {
   AGX3_SUBGROUP_SHUFFLE_BROADCAST = 0,
   AGX3_SUBGROUP_SHUFFLE_XOR,
   AGX3_SUBGROUP_SHUFFLE_UP,
   AGX3_SUBGROUP_SHUFFLE_DOWN,
   AGX3_SUBGROUP_SHUFFLE_NUM,
};

enum agx3_subgroup_lane_kind {
   /* An immediate lane index, in f29 with the class selector f37 = 1. Apple
    * emits this for the bounded SIMD and quad witnesses accepted below. */
   AGX3_SUBGROUP_LANE_IMMEDIATE = 1,
   AGX3_SUBGROUP_LANE_REGISTER = 2,
   AGX3_SUBGROUP_LANE_UNIFORM = 3,
};

struct agx3_subgroup_shuffle_desc {
   bool quad;
   enum agx3_subgroup_shuffle_op op;
   enum agx3_subgroup_lane_kind lane_kind;
   uint16_t dest;
   uint16_t src;
   uint16_t lane;
   uint16_t wait_mask;
   /* f11, the one-bit general-destination hint. Only consulted on the
    * immediate-lane path; the register and uniform paths keep their own
    * corpus constants pinned. */
   bool dest_hint;
   /* The G17P immediate quad form uses f25 while the source remains live and
    * f27 on its final read.  This is ignored by the older SIMD witnesses. */
   bool src_live;

   /* Register-lane quad broadcast must retain a lane read again later. */
   bool lane_live;

   /* 16-bit data (f10 = f24 = 1), dest and src in half units. Grounded for
    * the SIMD broadcast with an immediate lane and for every SIMD shuffle with
    * a register lane. */
   bool half;

   /* Lifetime hints of the G17P SIMD register-lane forms, at either width;
    * the lane is read as 16 bits from the half in `lane`. */
   bool src_cache;
   bool src_discard;
   bool lane_cache;
   bool lane_discard;
};

enum agx3_subgroup_ballot_type {
   AGX3_SUBGROUP_BALLOT_INTEGER = 0,
   AGX3_SUBGROUP_BALLOT_FLOAT,
};

struct agx3_subgroup_ballot_desc {
   bool quad;
   enum agx3_subgroup_ballot_type type;
   uint16_t dest;
   uint16_t src;
   /* The G17P SIMD integer form can compare against a 16-bit immediate. */
   bool compare_immediate;
   bool compare_register;
   uint16_t compare_value;
   bool compare_half;
   bool compare_cache;
   bool compare_discard;
   bool src_cache;
   /* The immediate equal-to-zero form also takes a 16-bit source register
    * (f24 = 1), as Apple's simd_broadcast_first() compares the 16-bit active
    * thread index against zero. */
   bool src_half;
   /* A 16-bit destination (f10 = 1): the quad ballot of a 16- or 32-bit
    * register against zero, the only such form witnessed. It carries its
    * source wait in the f46 bank. */
   bool dest_half;
   bool dest_cache;
   bool src_live;
   uint8_t condition;
   bool invert;
   uint16_t wait_mask;
};

enum agx3_subgroup_reduce_op {
   AGX3_SUBGROUP_REDUCE_AND = 0,
   AGX3_SUBGROUP_REDUCE_FADD,
   AGX3_SUBGROUP_REDUCE_OR,
   AGX3_SUBGROUP_REDUCE_FMUL,
   AGX3_SUBGROUP_REDUCE_XOR,
   AGX3_SUBGROUP_REDUCE_FMIN,
   AGX3_SUBGROUP_REDUCE_FMAX,
   AGX3_SUBGROUP_REDUCE_IADD,
   AGX3_SUBGROUP_REDUCE_SMIN,
   AGX3_SUBGROUP_REDUCE_SMAX,
   AGX3_SUBGROUP_REDUCE_UMIN,
   AGX3_SUBGROUP_REDUCE_UMAX,
   AGX3_SUBGROUP_REDUCE_NUM,
};

struct agx3_subgroup_reduce_desc {
   bool quad;
   bool prefix;
   enum agx3_subgroup_reduce_op op;
   uint16_t dest;
   uint16_t src;
   uint16_t wait_mask;
   /* f11, the general-destination hint. */
   bool dest_hint;
   /* Float reduction source liveness: false sets f29=1 (last use), true sets
    * f27=1 (source stays live). */
   bool src_live;

   /* 16-bit destination and source, in half units (AGX3b only). */
   bool half;
};

bool agx3_subgroup_encode_active_mask(uint16_t dest, bool dest_cache,
                                      uint8_t wait_mask,
                                       uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                       uint8_t *length);
bool agx3_subgroup_encode_quad_ballot_imm(uint16_t dest, bool dest_cache,
                                          uint16_t value, uint8_t wait_mask,
                                          uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                          uint8_t *length);

bool agx3_subgroup_encode_shuffle(enum agx_isa_encoding enc,
                                  const struct agx3_subgroup_shuffle_desc *desc,
                                  uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                  uint8_t *length);

/* G18G only (AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d afterwards): AGX3c
 * quad shuffles from a register lane take the register-lane form with the
 * quad selectors. */
bool
agx3_subgroup_encode_shuffle_g18g(enum agx_isa_encoding enc,
                                  const struct agx3_subgroup_shuffle_desc *d,
                                  uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                  uint8_t *length);

/* The recovered ballot subset compares a register against zero or, on G17P,
 * performs an unsigned ==, <, or > comparison against a 16-bit immediate. */
bool agx3_subgroup_encode_ballot(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_ballot_desc *desc,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length);

/* G18G only (AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d afterwards): the
 * AGX3c immediate ballot of a source that stays live carries f25 as the
 * G18G layer emits it. */
bool
agx3_subgroup_encode_ballot_g18g(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_ballot_desc *d,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length);

/* AGX3c `simdiballot` forms with no comparand register-against-constant:
 * simd_ballot(true), and an unsigned less-than of two 32-bit registers whose
 * last use this is. See agx3_subgroup_core.c for the witnesses. */
bool agx3_subgroup_encode_ballot_true_gpu1(uint16_t dest, bool dest_cache,
                                           uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                           uint8_t *length);
bool agx3_subgroup_encode_ballot_ult_reg_gpu1(
   uint16_t dest, bool dest_cache, uint16_t src0, uint16_t src1,
   uint8_t out[AGX3_SUBGROUP_MAX_LENGTH], uint8_t *length);

bool agx3_subgroup_encode_reduce(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_reduce_desc *desc,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length);

/* G18G only (AGX_ISA_ENCODING_AGX3C, re-encoded as agx3d afterwards): the
 * source-lifetime hints of the AGX3c SIMD integer reductions as the G18G
 * layer emits them (native999). */
bool
agx3_subgroup_encode_reduce_g18g(enum agx_isa_encoding enc,
                                 const struct agx3_subgroup_reduce_desc *d,
                                 uint8_t out[AGX3_SUBGROUP_MAX_LENGTH],
                                 uint8_t *length);

#ifdef __cplusplus
}
#endif
