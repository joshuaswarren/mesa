#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

enum agx3_matrix_input_type {
   AGX3_MATRIX_INPUT_F16 = 1,
   AGX3_MATRIX_INPUT_F32 = 2,
   AGX3_MATRIX_INPUT_BF16 = 4,
};

/* matrix_type bit of an 8x8 matmad with an f16 accumulator and result */
#define AGX3_MATRIX_ACCUM_F16 (1u << 8)

struct agx3_matrix_gemm16_desc {
   uint16_t dest;
   uint16_t a;
   uint16_t b;
   enum agx3_matrix_input_type input_type;
   bool dest_hint;
   bool transpose_b;
   bool field33;
   uint8_t field40_mask;
   bool zero_accumulator;
   bool transpose_a;
   bool preserve_a;
   bool preserve_b;
   uint8_t wait_mask;
   enum agx3_matrix_input_type b_input_type;
};

bool agx3_matrix_encode_gemm16(enum agx_isa_encoding encoding,
                              const struct agx3_matrix_gemm16_desc *desc,
                              uint8_t out[10], uint8_t *length);

struct agx3_matrix_mma8_desc {
   uint16_t dest;
   uint16_t a;
   uint16_t b;
   uint16_t c;
   bool c_zero;
   bool dest_cache;
   bool cache[3];
   bool discard[3];
   uint8_t wait_mask;
   enum agx3_matrix_input_type input_type;
   /* f16 C and result (f16matmad8x8, AGX3B), f16 A and B */
   bool f16_accumulate;
};

bool agx3_matrix_encode_mma8(enum agx_isa_encoding encoding,
                            const struct agx3_matrix_mma8_desc *desc,
                            uint8_t out[14], uint8_t *length);

#ifdef __cplusplus
}
#endif
