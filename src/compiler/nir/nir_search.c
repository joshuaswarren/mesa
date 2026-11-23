/*
 * Copyright © 2014 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir_search.h"
#include <inttypes.h>
#include "util/half_float.h"
#include "util/u_call_once.h"
#include "nir_builder.h"
#include "nir_opcodes.h"
#include "nir_worklist.h"

/* This should be the same as nir_search_max_comm_ops in nir_algebraic.py. */
#define NIR_SEARCH_MAX_COMM_OPS 8

struct match_variable {
   nir_def *def;
   uint8_t swizzle[NIR_MAX_VEC_COMPONENTS];
};

struct match_state {
   unsigned fp_math_ctrl;
   uint8_t comm_op_direction;
   unsigned variables_seen;

   /* Used for running the automaton on newly-constructed instructions. */
   struct util_dynarray *states;
   const struct per_op_table *pass_op_table;
   const nir_algebraic_table *table;

   struct match_variable variables[NIR_SEARCH_MAX_VARIABLES];
   const nir_search_state *state;
};

static bool
match_expression(const nir_algebraic_table *table, const nir_search_expression *expr, nir_alu_instr *instr,
                 unsigned num_components, const uint8_t *swizzle,
                 struct match_state *state);
static bool
nir_algebraic_automaton_alu(nir_alu_instr *alu, struct util_dynarray *states,
                           const struct per_op_table *tbl)
{
   const nir_op op = alu->op;
   if (tbl->num_filtered_states == 0)
      return false;

   /* Calculate the index into the transition table. Note the index
    * calculated must match the iteration order of Python's
    * itertools.product(), which was used to emit the transition
    * table.
    */
   uint16_t *states_data = states->data;
   unsigned index = 0;
   if (tbl->filter) {
      const unsigned num_inputs = nir_op_infos[op].num_inputs;
      const unsigned num_filtered = tbl->num_filtered_states;
      const uint16_t *filter = tbl->filter;
      if (num_inputs == 1) {
         index = filter[states_data[alu->src[0].src.ssa->index]];
      } else if (num_inputs == 2) {
         index = filter[states_data[alu->src[0].src.ssa->index]] * num_filtered +
                 filter[states_data[alu->src[1].src.ssa->index]];
      } else {
         for (unsigned i = 0; i < num_inputs; i++) {
            index = index * num_filtered +
                    filter[states_data[alu->src[i].src.ssa->index]];
         }
      }
   }

   uint16_t *state = &states_data[alu->def.index];
   const uint16_t next = tbl->table[index];
   if (*state != next) {
      *state = next;
      return true;
   }
   return false;
}

static bool
nir_algebraic_automaton(nir_instr *instr, struct util_dynarray *states,
                        const struct per_op_table *pass_op_table);

static const uint8_t identity_swizzle[NIR_MAX_VEC_COMPONENTS] = {
   0,
   1,
   2,
   3,
   4,
   5,
   6,
   7,
   8,
   9,
   10,
   11,
   12,
   13,
   14,
   15,
};

static bool
nir_op_matches_search_op(nir_op nop, uint16_t sop)
{
   if (sop <= nir_last_opcode)
      return nop == sop;

#define MATCH_FCONV_CASE(op)           \
   case nir_search_op_##op:            \
      return nop == nir_op_##op##16 || \
             nop == nir_op_##op##32 || \
             nop == nir_op_##op##64;

#define MATCH_ICONV_CASE(op)           \
   case nir_search_op_##op:            \
      return nop == nir_op_##op##8 ||  \
             nop == nir_op_##op##16 || \
             nop == nir_op_##op##32 || \
             nop == nir_op_##op##64;

   switch (sop) {
      MATCH_FCONV_CASE(i2f)
      MATCH_FCONV_CASE(u2f)
      MATCH_FCONV_CASE(f2f)
      MATCH_ICONV_CASE(f2u)
      MATCH_ICONV_CASE(f2i)
      MATCH_ICONV_CASE(u2u)
      MATCH_ICONV_CASE(i2i)
      MATCH_FCONV_CASE(b2f)
      MATCH_ICONV_CASE(b2i)
   default:
      UNREACHABLE("Invalid nir_search_op");
   }

#undef MATCH_FCONV_CASE
#undef MATCH_ICONV_CASE
}

static uint16_t
nir_search_op_for_nir_op_slow(nir_op nop)
{
#define MATCH_FCONV_CASE(op) \
   case nir_op_##op##16:     \
   case nir_op_##op##32:     \
   case nir_op_##op##64:     \
      return nir_search_op_##op;

#define MATCH_ICONV_CASE(op) \
   case nir_op_##op##8:      \
   case nir_op_##op##16:     \
   case nir_op_##op##32:     \
   case nir_op_##op##64:     \
      return nir_search_op_##op;

   switch (nop) {
      MATCH_FCONV_CASE(i2f)
      MATCH_FCONV_CASE(u2f)
      MATCH_FCONV_CASE(f2f)
      MATCH_ICONV_CASE(f2u)
      MATCH_ICONV_CASE(f2i)
      MATCH_ICONV_CASE(u2u)
      MATCH_ICONV_CASE(i2i)
      MATCH_FCONV_CASE(b2f)
      MATCH_ICONV_CASE(b2i)
   default:
      return nop;
   }

#undef MATCH_FCONV_CASE
#undef MATCH_ICONV_CASE
}

static uint16_t nir_op_to_search_op[nir_num_opcodes];
static once_flag nir_search_op_once = ONCE_FLAG_INIT;

static void
init_nir_search_op_table(void)
{
   for (unsigned i = 0; i < nir_num_opcodes; i++)
      nir_op_to_search_op[i] = nir_search_op_for_nir_op_slow((nir_op)i);
}

uint16_t
nir_search_op_for_nir_op(nir_op nop)
{
   call_once(&nir_search_op_once, init_nir_search_op_table);
   return nir_op_to_search_op[nop];
}

static nir_op
nir_op_for_search_op(uint16_t sop, unsigned bit_size)
{
   if (sop <= nir_last_opcode)
      return sop;

#define RET_FCONV_CASE(op)                \
   case nir_search_op_##op:               \
      switch (bit_size) {                 \
      case 16:                            \
         return nir_op_##op##16;          \
      case 32:                            \
         return nir_op_##op##32;          \
      case 64:                            \
         return nir_op_##op##64;          \
      default:                            \
         UNREACHABLE("Invalid bit size"); \
      }

#define RET_ICONV_CASE(op)                \
   case nir_search_op_##op:               \
      switch (bit_size) {                 \
      case 8:                             \
         return nir_op_##op##8;           \
      case 16:                            \
         return nir_op_##op##16;          \
      case 32:                            \
         return nir_op_##op##32;          \
      case 64:                            \
         return nir_op_##op##64;          \
      default:                            \
         UNREACHABLE("Invalid bit size"); \
      }

   switch (sop) {
      RET_FCONV_CASE(i2f)
      RET_FCONV_CASE(u2f)
      RET_FCONV_CASE(f2f)
      RET_ICONV_CASE(f2u)
      RET_ICONV_CASE(f2i)
      RET_ICONV_CASE(u2u)
      RET_ICONV_CASE(i2i)
      RET_FCONV_CASE(b2f)
      RET_ICONV_CASE(b2i)
   default:
      UNREACHABLE("Invalid nir_search_op");
   }

#undef RET_FCONV_CASE
#undef RET_ICONV_CASE
}

static bool
match_value(const nir_algebraic_table *table,
            const nir_search_value *value, nir_alu_instr *instr, unsigned src,
            unsigned num_components, const uint8_t *swizzle,
            struct match_state *state)
{
   /* If the value has a specific bit size and it doesn't match, bail */
   if (value->bit_size > 0 &&
       nir_src_bit_size(instr->src[src].src) != value->bit_size)
      return false;

   /* If the source is an explicitly sized source, then we need to reset
    * both the number of components and the swizzle.
    */
   if (nir_op_infos[instr->op].input_sizes[src] != 0) {
      num_components = nir_op_infos[instr->op].input_sizes[src];
      swizzle = identity_swizzle;
   }

   switch (value->type) {
   case nir_search_value_expression: {
      if (!nir_src_is_alu(instr->src[src].src))
         return false;

      uint8_t new_swizzle[NIR_MAX_VEC_COMPONENTS];
      if (num_components == 1) {
         new_swizzle[0] = instr->src[src].swizzle[swizzle[0]];
      } else {
         for (unsigned i = 0; i < num_components; ++i)
            new_swizzle[i] = instr->src[src].swizzle[swizzle[i]];
      }

      return match_expression(table, nir_search_value_as_expression(value),
                              nir_def_as_alu(instr->src[src].src.ssa),
                              num_components, new_swizzle, state);
   }

   case nir_search_value_variable: {
      const nir_search_variable *var = nir_search_value_as_variable(value);
      assert(var->variable < NIR_SEARCH_MAX_VARIABLES);

      if (var->is_constant &&
          !nir_src_is_const(instr->src[src].src))
         return false;

      const unsigned var_bit = 1u << var->variable;
      if (var->cond_index == -1) {
         if (state->variables_seen & var_bit) {
            const struct match_variable *existing = &state->variables[var->variable];
            if (existing->def != instr->src[src].src.ssa)
               return false;

            if (num_components == 1)
               return existing->swizzle[0] == instr->src[src].swizzle[swizzle[0]];

            for (unsigned i = 0; i < num_components; ++i) {
               if (existing->swizzle[i] != instr->src[src].swizzle[swizzle[i]])
                  return false;
            }
            return true;
         } else {
            state->variables_seen |= var_bit;
            struct match_variable *dst = &state->variables[var->variable];
            dst->def = instr->src[src].src.ssa;

            if (num_components == 1) {
               dst->swizzle[0] = instr->src[src].swizzle[swizzle[0]];
            } else {
               for (unsigned i = 0; i < num_components; ++i)
                  dst->swizzle[i] = instr->src[src].swizzle[swizzle[i]];
            }
            memset(dst->swizzle + num_components, 0,
                   NIR_MAX_VEC_COMPONENTS - num_components);
            return true;
         }
      }

      uint8_t new_swizzle[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < num_components; ++i)
         new_swizzle[i] = instr->src[src].swizzle[swizzle[i]];

      if (!table->variable_cond[var->cond_index](state->state, instr,
                                                 src, num_components, new_swizzle))
         return false;

      if (state->variables_seen & var_bit) {
         if (state->variables[var->variable].def != instr->src[src].src.ssa)
            return false;

         return !memcmp(state->variables[var->variable].swizzle, new_swizzle, num_components);
      } else {
         state->variables_seen |= var_bit;
         struct match_variable *dst = &state->variables[var->variable];
         dst->def = instr->src[src].src.ssa;
         memcpy(dst->swizzle, new_swizzle, num_components);
         memset(dst->swizzle + num_components, 0, NIR_MAX_VEC_COMPONENTS - num_components);
         return true;
      }
   }

   case nir_search_value_constant: {
      const nir_search_constant *const_val = nir_search_value_as_constant(value);

      if (!nir_src_is_const(instr->src[src].src))
         return false;

      switch (const_val->type) {
      case nir_type_float: {
         nir_load_const_instr *const load =
            nir_def_as_load_const(instr->src[src].src.ssa);

         /* There are 8-bit and 1-bit integer types, but there are no 8-bit or
          * 1-bit float types.  This prevents potential assertion failures in
          * nir_src_comp_as_float.
          */
         if (load->def.bit_size < 16)
            return false;

         for (unsigned i = 0; i < num_components; ++i) {
            double val = nir_src_comp_as_float(instr->src[src].src,
                                               instr->src[src].swizzle[swizzle[i]]);
            if (val != const_val->data.d)
               return false;

            /* The comparison above does not check the sign bit for 0.0,
             * so do it manually.
             */
            if ((dui(val) == 0) != (dui(const_val->data.d) == 0))
               return false;
         }
         return true;
      }

      case nir_type_int:
      case nir_type_uint:
      case nir_type_bool: {
         unsigned bit_size = nir_src_bit_size(instr->src[src].src);
         uint64_t mask = u_uintN_max(bit_size);
         for (unsigned i = 0; i < num_components; ++i) {
            uint64_t val = nir_src_comp_as_uint(instr->src[src].src,
                                                instr->src[src].swizzle[swizzle[i]]);
            if ((val & mask) != (const_val->data.u & mask))
               return false;
         }
         return true;
      }

      default:
         UNREACHABLE("Invalid alu source type");
      }
   }

   default:
      UNREACHABLE("Invalid search value type");
   }
}

static inline bool
match_expression_sources(const nir_algebraic_table *table,
                         const nir_search_expression *expr, nir_alu_instr *instr,
                         unsigned num_components, const uint8_t *swizzle,
                         struct match_state *state)
{
   /* If this is a commutative expression and it's one of the first few, look
    * up its direction for the current search operation.  We'll use that value
    * to possibly flip the sources for the match.
    */
   unsigned comm_op_flip =
      !!(state->comm_op_direction & expr->comm_expr_mask);

   const unsigned num_inputs = nir_op_infos[instr->op].num_inputs;
   bool matched = true;
   for (unsigned i = 0; i < num_inputs; i++) {
      /* If src1 of the search expression is a constant, check that first since it's faster. */
      unsigned src_idx = i < 2 ? i ^ expr->src1_is_const : i;

      /* 2src_commutative instructions that have 3 sources are only commutative
       * in the first two sources.  Source 2 is always source 2.
       */
      if (!match_value(table, &state->table->values[expr->srcs[src_idx]].value, instr,
                       i < 2 ? src_idx ^ comm_op_flip : src_idx,
                       num_components, swizzle, state)) {
         matched = false;
         break;
      }
   }

   return matched;
}

static bool
match_expression(const nir_algebraic_table *table, const nir_search_expression *expr, nir_alu_instr *instr,
                 unsigned num_components, const uint8_t *swizzle,
                 struct match_state *state)
{
   if (expr->cond_index != -1 && !table->expression_cond[expr->cond_index](instr))
      return false;

   if (!nir_op_matches_search_op(instr->op, expr->opcode))
      return false;

   if (expr->value.bit_size > 0 &&
       instr->def.bit_size != expr->value.bit_size)
      return false;

   if (expr->fp_math_ctrl_exclude & instr->fp_math_ctrl)
      return false;

   state->fp_math_ctrl |= instr->fp_math_ctrl;

   assert(nir_op_infos[instr->op].num_inputs > 0);

   /* If we have an explicitly sized destination, we can only handle the
    * identity swizzle.  While dot(vec3(a, b, c).zxy) is a valid
    * expression, we don't have the information right now to propagate that
    * swizzle through.  We can only properly propagate swizzles if the
    * instruction is vectorized.
    *
    * The only exception is swizzle, for which we have a special condition,
    * so that we can do pack64_2x32_split(unpack(a).x, unpack(a).y) --> a.
    */
   if (expr->swizzle >= 0) {
      if (num_components != 1 || swizzle[0] != expr->swizzle)
         return false;
   } else {
      if (nir_op_infos[instr->op].output_size != 0) {
         if (memcmp(swizzle, identity_swizzle, num_components))
            return false;
      }
   }

   return match_expression_sources(table, expr, instr, num_components,
                                   swizzle, state);
}

static unsigned
replace_bitsize(const nir_search_value *value, unsigned search_bitsize,
                struct match_state *state)
{
   if (value->bit_size > 0)
      return value->bit_size;
   if (value->bit_size < 0)
      return state->variables[-value->bit_size - 1].def->bit_size;
   return search_bitsize;
}

static nir_alu_src
construct_value(nir_builder *build,
                const nir_search_value *value,
                unsigned num_components, unsigned search_bitsize,
                struct match_state *state,
                nir_instr *instr)
{
   switch (value->type) {
   case nir_search_value_expression: {
      const nir_search_expression *expr = nir_search_value_as_expression(value);
      unsigned dst_bit_size = replace_bitsize(value, search_bitsize, state);
      nir_op op = nir_op_for_search_op(expr->opcode, dst_bit_size);

      if (nir_op_infos[op].output_size != 0)
         num_components = nir_op_infos[op].output_size;

      nir_alu_instr *alu = nir_alu_instr_create(build->shader, op);
      nir_def_init(&alu->instr, &alu->def, num_components,
                   dst_bit_size);

      /* We have no way of knowing what values in a given search expression
       * map to a particular replacement value.  Therefore, if the
       * expression we are replacing has any exact values, the entire
       * replacement should be exact.
       */
      alu->fp_math_ctrl = nir_op_valid_fp_math_ctrl(op, state->fp_math_ctrl | expr->fp_math_ctrl_add);

      for (unsigned i = 0; i < nir_op_infos[op].num_inputs; i++) {
         /* If the source is an explicitly sized source, then we need to reset
          * the number of components to match.
          */
         if (nir_op_infos[alu->op].input_sizes[i] != 0)
            num_components = nir_op_infos[alu->op].input_sizes[i];

         alu->src[i] = construct_value(build, &state->table->values[expr->srcs[i]].value,
                                       num_components, search_bitsize,
                                       state, instr);
      }

      /* Immediately try to constant-fold the expression, in order to allow
       * for more expressions to be matched within a single pass.
       */
      nir_def *def = &alu->def;
      nir_def *const_expr = nir_try_constant_fold_alu(build, alu);
      if (const_expr) {
         nir_instr_free(&alu->instr);
         def = const_expr;
         if (nir_def_is_alu(def)) {
            /* The instruction got folded into bcsel of two constants. */
            nir_alu_instr *bcsel = nir_def_as_alu(def);
            assert(bcsel->op == nir_op_bcsel);
            util_dynarray_append_typed(state->states, uint16_t, 0);
            nir_algebraic_automaton(nir_def_instr(bcsel->src[1].src.ssa), state->states, state->pass_op_table);
            util_dynarray_append_typed(state->states, uint16_t, 0);
            nir_algebraic_automaton(nir_def_instr(bcsel->src[2].src.ssa), state->states, state->pass_op_table);
         }
      } else {
         nir_builder_instr_insert(build, &alu->instr);
      }
      assert(def->index ==
             util_dynarray_num_elements(state->states, uint16_t));
      util_dynarray_append_typed(state->states, uint16_t, 0);
      nir_algebraic_automaton(nir_def_instr(def), state->states, state->pass_op_table);

      nir_alu_src val;
      val.src = nir_src_for_ssa(def);
      if (expr->swizzle < 0)
         memcpy(val.swizzle, identity_swizzle, sizeof(val.swizzle));
      else
         memset(val.swizzle, expr->swizzle, sizeof(val.swizzle));

      return val;
   }

   case nir_search_value_variable: {
      const nir_search_variable *var = nir_search_value_as_variable(value);
      assert(state->variables_seen & (1 << var->variable));

      nir_alu_src val;
      val.src = nir_src_for_ssa(state->variables[var->variable].def);
      assert(!var->is_constant);

      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
         val.swizzle[i] = state->variables[var->variable].swizzle[var->swizzle[i]];

      return val;
   }

   case nir_search_value_constant: {
      const nir_search_constant *c = nir_search_value_as_constant(value);
      unsigned bit_size = replace_bitsize(value, search_bitsize, state);

      nir_def *cval;
      switch (c->type) {
      case nir_type_float:
         cval = nir_imm_floatN_t(build, c->data.d, bit_size);
         break;

      case nir_type_int:
      case nir_type_uint:
         cval = nir_imm_intN_t(build, c->data.i, bit_size);
         break;

      case nir_type_bool:
         cval = nir_imm_boolN_t(build, c->data.u, bit_size);
         break;

      default:
         UNREACHABLE("Invalid alu source type");
      }

      assert(cval->index ==
             util_dynarray_num_elements(state->states, uint16_t));
      util_dynarray_append_typed(state->states, uint16_t, 0);
      nir_algebraic_automaton(nir_def_instr(cval), state->states,
                              state->pass_op_table);

      nir_alu_src val;
      val.src = nir_src_for_ssa(cval);
      memset(val.swizzle, 0, sizeof val.swizzle);

      return val;
   }

   default:
      UNREACHABLE("Invalid search value type");
   }
}

UNUSED static void
dump_value(const nir_algebraic_table *table, const nir_search_value *val)
{
   switch (val->type) {
   case nir_search_value_constant: {
      const nir_search_constant *sconst = nir_search_value_as_constant(val);
      switch (sconst->type) {
      case nir_type_float:
         fprintf(stderr, "%f", sconst->data.d);
         break;
      case nir_type_int:
         fprintf(stderr, "%" PRId64, sconst->data.i);
         break;
      case nir_type_uint:
         fprintf(stderr, "0x%" PRIx64, sconst->data.u);
         break;
      case nir_type_bool:
         fprintf(stderr, "%s", sconst->data.u != 0 ? "True" : "False");
         break;
      default:
         UNREACHABLE("bad const type");
      }
      break;
   }

   case nir_search_value_variable: {
      const nir_search_variable *var = nir_search_value_as_variable(val);
      if (var->is_constant)
         fprintf(stderr, "#");
      fprintf(stderr, "%c", var->variable + 'a');
      break;
   }

   case nir_search_value_expression: {
      const nir_search_expression *expr = nir_search_value_as_expression(val);
      fprintf(stderr, "(");
      if (expr->fp_math_ctrl_exclude & nir_fp_exact)
         fprintf(stderr, "~");
      switch (expr->opcode) {
#define CASE(n)            \
   case nir_search_op_##n: \
      fprintf(stderr, #n); \
      break;
         CASE(b2f)
         CASE(b2i)
         CASE(i2i)
         CASE(u2u)
         CASE(f2f)
         CASE(f2i)
         CASE(f2u)
         CASE(i2f)
         CASE(u2f)
#undef CASE
      default:
         fprintf(stderr, "%s", nir_op_infos[expr->opcode].name);
      }

      unsigned num_srcs = 1;
      if (expr->opcode <= nir_last_opcode)
         num_srcs = nir_op_infos[expr->opcode].num_inputs;

      for (unsigned i = 0; i < num_srcs; i++) {
         fprintf(stderr, " ");
         dump_value(table, &table->values[expr->srcs[i]].value);
      }

      fprintf(stderr, ")");
      break;
   }
   }

   if (val->bit_size > 0)
      fprintf(stderr, "@%d", val->bit_size);
}

static void
add_uses_to_worklist(nir_instr *instr,
                     nir_instr_worklist *worklist,
                     struct util_dynarray *states,
                     const struct per_op_table *pass_op_table)
{
   nir_def *def = nir_instr_def(instr);

   nir_foreach_use_safe(use_src, def) {
      if (nir_algebraic_automaton(nir_src_use_instr(use_src), states, pass_op_table))
         nir_instr_worklist_push_tail(worklist, nir_src_use_instr(use_src));
   }
}

static inline bool
state_has_transforms(const nir_algebraic_table *table, uint16_t xform_idx)
{
   return table->transforms[table->transform_offsets[xform_idx]].condition_offset !=
          ~0u;
}

static void
nir_algebraic_update_automaton(nir_instr *new_instr,
                               nir_instr_worklist *algebraic_worklist,
                               struct util_dynarray *states,
                               const nir_algebraic_table *table)
{
   const struct per_op_table *pass_op_table = table->pass_op_table;

   nir_instr_worklist automaton_worklist;
   nir_instr_worklist_init(&automaton_worklist);

   /* Walk through the tree of uses of our new instruction's SSA value,
    * recursively updating the automaton state until it stabilizes.
    */
   add_uses_to_worklist(new_instr, &automaton_worklist, states, pass_op_table);

   nir_instr *instr;
   while ((instr = nir_instr_worklist_pop_head(&automaton_worklist))) {
      if (instr->type == nir_instr_type_alu) {
         const uint16_t xform_idx =
            ((const uint16_t *)states->data)[nir_instr_as_alu(instr)->def.index];
         if (state_has_transforms(table, xform_idx))
            nir_instr_worklist_push_tail(algebraic_worklist, instr);
      }
      add_uses_to_worklist(instr, &automaton_worklist, states, pass_op_table);
   }

   nir_instr_worklist_fini(&automaton_worklist);
}

static bool
nir_replace_instr(nir_builder *build, nir_alu_instr *instr,
                  const nir_search_state *search_state,
                  struct util_dynarray *states,
                  const nir_algebraic_table *table,
                  const nir_search_expression *search,
                  const nir_search_value *replace,
                  nir_instr_worklist *algebraic_worklist,
                  struct exec_list *dead_instrs)
{
   struct match_state state;
   state.state = search_state;
   state.pass_op_table = table->pass_op_table;
   state.table = table;

   STATIC_ASSERT(sizeof(state.comm_op_direction) * 8 >= NIR_SEARCH_MAX_COMM_OPS);
   STATIC_ASSERT(sizeof(search->comm_expr_mask) * 8 >= NIR_SEARCH_MAX_COMM_OPS);

   unsigned comm_expr_combinations =
      1 << MIN2(search->comm_exprs, NIR_SEARCH_MAX_COMM_OPS);

   const bool direct_root = search->cond_index == -1;
   if (direct_root) {
      if (!nir_op_matches_search_op(instr->op, search->opcode) ||
          (search->value.bit_size > 0 &&
           instr->def.bit_size != search->value.bit_size) ||
          (search->fp_math_ctrl_exclude & instr->fp_math_ctrl) ||
          (search->swizzle >= 0 &&
           (instr->def.num_components != 1 || search->swizzle != 0)))
         return false;
      assert(nir_op_infos[instr->op].num_inputs > 0);
   }
   const unsigned root_fp_math_ctrl = nir_fp_fast_math |
      (direct_root ? instr->fp_math_ctrl : 0);

   bool found = false;
   for (unsigned comb = 0; comb < comm_expr_combinations; comb++) {
      /* The bitfield of directions is just the current iteration.  Hooray for
       * binary.
       */
      state.comm_op_direction = comb;
      state.fp_math_ctrl = root_fp_math_ctrl;
      state.variables_seen = 0;

      const bool matched = direct_root
         ? match_expression_sources(table, search, instr,
                                    instr->def.num_components,
                                    identity_swizzle, &state)
         : match_expression(table, search, instr,
                            instr->def.num_components,
                            identity_swizzle, &state);
      if (matched) {
         found = true;
         break;
      }
   }
   if (!found)
      return false;

#if 0
   fprintf(stderr, "matched: ");
   dump_value(table, &search->value);
   fprintf(stderr, " -> ");
   dump_value(table, replace);
   fprintf(stderr, " ssa_%d\n", instr->def.index);
#endif

   /* If the instruction at the root of the expression tree being replaced is
    * a unary operation, insert the replacement instructions at the location
    * of the source of the unary operation.  Otherwise, insert the replacement
    * instructions at the location of the expression tree root.
    *
    * For the unary operation case, this is done to prevent some spurious code
    * motion that can dramatically extend live ranges.  Imagine an expression
    * like -(A+B) where the addtion and the negation are separated by flow
    * control and thousands of instructions.  If this expression is replaced
    * with -A+-B, inserting the new instructions at the site of the negation
    * could extend the live range of A and B dramtically.  This could increase
    * register pressure and cause spilling.
    *
    * It may well be that moving instructions around is a good thing, but
    * keeping algebraic optimizations and code motion optimizations separate
    * seems safest.
    */
   nir_alu_instr *const src_instr = nir_src_as_alu(instr->src[0].src);
   if (src_instr != NULL &&
       (instr->op == nir_op_fneg || instr->op == nir_op_fabs ||
        instr->op == nir_op_ineg || instr->op == nir_op_iabs ||
        instr->op == nir_op_inot)) {
      /* Insert new instructions *after*.  Otherwise a hypothetical
       * replacement fneg(X) -> fabs(X) would insert the fabs() instruction
       * before X!  This can also occur for things like fneg(X.wzyx) -> X.wzyx
       * in vector mode.  A move instruction to handle the swizzle will get
       * inserted before X.
       *
       * This manifested in a single OpenGL ES 2.0 CTS vertex shader test on
       * older Intel GPU that use vector-mode vertex processing.
       */
      build->cursor = nir_after_instr(&src_instr->instr);
   } else {
      build->cursor = nir_before_instr(&instr->instr);
   }

   state.states = states;

   nir_alu_src val = construct_value(build, replace,
                                     instr->def.num_components,
                                     instr->def.bit_size,
                                     &state, &instr->instr);

   /* Note that NIR builder will elide the MOV if it's a no-op, which may
    * allow more work to be done in a single pass through algebraic.
    */
   nir_def *mov = nir_def_rewrite_uses_with_alu_src(build, &instr->def, val);

   if (mov) {
      util_dynarray_append_typed(states, uint16_t, 0);
      nir_algebraic_automaton(nir_def_instr(mov), states, table->pass_op_table);
   }

   /* Recurse through the uses updating the automaton's state. */
   nir_algebraic_update_automaton(nir_def_instr(val.src.ssa), algebraic_worklist,
                                  states, table);

   /* Nothing uses the instr any more, so drop it out of the program.  Note
    * that the instr may be in the worklist still, so we can't free it
    * directly.
    */
   assert(instr->instr.pass_flags == 0);
   instr->instr.pass_flags = 1;
   nir_instr_remove(&instr->instr);
   exec_list_push_tail(dead_instrs, &instr->instr.node);

   return true;
}

static bool
nir_algebraic_automaton(nir_instr *instr, struct util_dynarray *states,
                        const struct per_op_table *pass_op_table)
{
   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      uint16_t search_op = nir_op_to_search_op[alu->op];
      return nir_algebraic_automaton_alu(alu, states,
                                        &pass_op_table[search_op]);
   }

   case nir_instr_type_load_const: {
      nir_load_const_instr *load_const = nir_instr_as_load_const(instr);
      uint16_t *state = util_dynarray_element(states, uint16_t,
                                              load_const->def.index);
      if (*state != CONST_STATE) {
         *state = CONST_STATE;
         return true;
      }
      return false;
   }

   default:
      return false;
   }
}

static bool
nir_algebraic_instr(nir_builder *build, nir_instr *instr,
                    const nir_search_state *state,
                    const bool *condition_flags,
                    const nir_algebraic_table *table,
                    struct util_dynarray *states,
                    nir_instr_worklist *worklist,
                    struct exec_list *dead_instrs)
{

   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);

   int xform_idx = *util_dynarray_element(states, uint16_t,
                                          alu->def.index);
   for (const struct transform *xform = &table->transforms[table->transform_offsets[xform_idx]];
        xform->condition_offset != ~0;
        xform++) {
      if (condition_flags[xform->condition_offset] &&
          nir_replace_instr(build, alu, state, states, table,
                            &table->values[xform->search].expression,
                            &table->values[xform->replace].value, worklist, dead_instrs)) {
         nir_invalidate_fp_analysis_state(state->range_ht);
         if (state->numlsb_ht->entries)
            _mesa_hash_table_clear(state->numlsb_ht, NULL);
         return true;
      }
   }

   return false;
}

bool
nir_algebraic_impl(nir_function_impl *impl,
                   const bool *condition_flags,
                   const nir_algebraic_table *table)
{
   bool progress = false;

   nir_builder build = nir_builder_create(impl);

   /* Note: it's important here that we're allocating a zeroed array, since
    * state 0 is the default state, which means we don't have to visit
    * anything other than constants and ALU instructions.
    */
   struct util_dynarray states = { 0 };
   if (!util_dynarray_resize(&states, uint16_t, impl->ssa_alloc)) {
      return nir_no_progress(impl);
   }
   memset(states.data, 0, states.size);

   nir_instr_worklist worklist;
   nir_instr_worklist_init(&worklist);

   const bool state0_has_transforms = state_has_transforms(table, 0);
   call_once(&nir_search_op_once, init_nir_search_op_table);

   /* Walk top-to-bottom setting up the automaton state. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         instr->pass_flags = 0;
         if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            const uint16_t op = nir_op_to_search_op[alu->op];
            const struct per_op_table *tbl = &table->pass_op_table[op];
            if (tbl->num_filtered_states == 0 && !state0_has_transforms)
               continue;
            nir_algebraic_automaton_alu(alu, &states, tbl);
            const uint16_t xform_idx =
               ((const uint16_t *)states.data)[alu->def.index];
            if (state_has_transforms(table, xform_idx))
               nir_instr_worklist_push_tail(&worklist, instr);
         } else if (instr->type == nir_instr_type_load_const) {
            nir_load_const_instr *load_const = nir_instr_as_load_const(instr);
            ((uint16_t *)states.data)[load_const->def.index] = CONST_STATE;
         }
      }
   }

   if (nir_instr_worklist_is_empty(&worklist)) {
      nir_instr_worklist_fini(&worklist);
      util_dynarray_fini(&states);
      return nir_no_progress(impl);
   }

   nir_fp_analysis_state range_ht = nir_create_fp_analysis_state(impl);

   struct hash_table numlsb_ht;
   _mesa_pointer_hash_table_init(&numlsb_ht, NULL);

   nir_search_state state;
   state.range_ht = &range_ht;
   state.numlsb_ht = &numlsb_ht;

   /* Put our instrs in the worklist such that we're popping the last instr
    * first.  This will encourage us to match the biggest source patterns when
    * possible.
    */
   assert(worklist.instr_vec.tail == 0);
   const unsigned initial_count = nir_instr_worklist_length(&worklist);
   nir_instr **initial = worklist.instr_vec.data;
   for (unsigned i = 0; i < initial_count / 2; ++i) {
      nir_instr *tmp = initial[i];
      initial[i] = initial[initial_count - 1 - i];
      initial[initial_count - 1 - i] = tmp;
   }

   struct exec_list dead_instrs;
   exec_list_make_empty(&dead_instrs);

   nir_instr *instr;
   while ((instr = nir_instr_worklist_pop_head(&worklist))) {
      /* The worklist can have an instr pushed to it multiple times if it was
       * the src of multiple instrs that also got optimized, so make sure that
       * we don't try to re-optimize an instr we already handled.
       */
      if (instr->pass_flags)
         continue;

      progress |= nir_algebraic_instr(&build, instr,
                                      &state, condition_flags,
                                      table, &states, &worklist, &dead_instrs);
   }

   nir_instr_free_list(&dead_instrs);

   nir_instr_worklist_fini(&worklist);
   _mesa_hash_table_fini(&numlsb_ht, NULL);
   nir_free_fp_analysis_state(&range_ht);
   util_dynarray_fini(&states);

   return nir_progress(progress, impl, nir_metadata_control_flow);
}
