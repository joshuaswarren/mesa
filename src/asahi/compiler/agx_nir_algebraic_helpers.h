/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_search.h"

/* nir_algebraic condition: the source is the scalar result of an
 * intrinsic, such as a single value loaded from memory, rather than one lane
 * of a vector (a four-byte load is one word on AGX3B, fold_w8_word_unpack())
 * or a value taken apart from a wider one by ALU instructions. */
static inline bool
is_scalar_intrinsic_result(UNUSED const nir_search_state *state,
                           const nir_alu_instr *instr, unsigned src,
                           UNUSED unsigned num_components,
                           const uint8_t *swizzle)
{
   nir_scalar s = nir_scalar_chase_movs(
      nir_get_scalar(instr->src[src].src.ssa, swizzle[0]));
   return nir_scalar_is_intrinsic(s) && s.def->num_components == 1;
}
