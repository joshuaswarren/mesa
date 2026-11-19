#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define AGX3_RT_GEN1_TRIANGLE_SLOT 101u
bool agx3_rt_gen1_triangle_body_generate(uint8_t *out, size_t capacity, size_t *length);
void agx3_rt_gen1_triangle_constants(const uint8_t **data, size_t *length);
