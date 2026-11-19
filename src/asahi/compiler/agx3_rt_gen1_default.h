#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define AGX3_RT_GEN1_DEFAULT_BODY_MAX_SIZE 512u
#define AGX3_RT_GEN1_DEFAULT_CONSTANTS_SIZE 16u
#define AGX3_RT_GEN1_RUNTIME_BASE UINT64_C(0x10010000000)
#define AGX3_RT_GEN1_RUNTIME_SLOT_STRIDE UINT64_C(0x200000)
#define AGX3_RT_GEN1_RUNTIME_MAP_BYTES UINT64_C(0x4000)
#define AGX3_RT_GEN1_RUNTIME_SLOTS 210u
#define AGX3_RT_GEN1_RUNTIME_MAP_ROWS (3u * AGX3_RT_GEN1_RUNTIME_SLOTS)
#define AGX3_RT_GEN1_RUNTIME_IMAGE_BYTES 11108352u
#define AGX3_RT_GEN1_IFT_BYTES 0x4000u
bool agx3_rt_gen1_default_body_generate(uint8_t out[AGX3_RT_GEN1_DEFAULT_BODY_MAX_SIZE], size_t *length);
void agx3_rt_gen1_default_constants(uint8_t out[AGX3_RT_GEN1_DEFAULT_CONSTANTS_SIZE]);
bool agx3_rt_gen1_default_entry_generate(uint64_t slot_body_va, uint8_t out[512], size_t *length);
bool agx3_rt_gen1_runtime_code_map(uint64_t rows[AGX3_RT_GEN1_RUNTIME_MAP_ROWS][3]);
void agx3_rt_gen1_runtime_ift(uint8_t out[AGX3_RT_GEN1_IFT_BYTES]);
bool agx3_rt_gen1_stub_body_generate(unsigned slot, uint8_t *out, size_t capacity, size_t *length);
