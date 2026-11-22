/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"

#define SHARED_PAIR_WINDOW 64

struct shared_address {
   nir_scalar index;
   uint32_t byte_offset;
};

static bool
plain_shared_load(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
   return load->intrinsic == nir_intrinsic_load_shared &&
          nir_intrinsic_access(load) == 0;
}

/* A lowered AGX3 shared address is BASE in bytes plus a 16-bit index in
 * scalar elements. Compare (dynamic index, constant byte offset) only after
 * proving that the original addition and narrowing do not wrap.
 */
static bool
shared_address32(nir_shader *shader, struct hash_table *ranges,
                 nir_instr *instr, struct shared_address *address)
{
   if (!plain_shared_load(instr))
      return false;

   nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
   const int base = nir_intrinsic_base(load);
   if (load->def.bit_size != 32 || load->def.num_components != 1 ||
       load->src[0].ssa->bit_size != 16 ||
       nir_intrinsic_align_mul(load) < 4 ||
       nir_intrinsic_align_offset(load) % 4 || base < 0 || base % 4)
      return false;

   nir_scalar index = nir_scalar_resolved(load->src[0].ssa, 0);
   if (nir_scalar_is_alu(index) && nir_scalar_alu_op(index) == nir_op_u2u16)
      index = nir_scalar_chase_alu_src(index, 0);

   uint64_t add = 0;
   uint64_t upper = 0;
   if (nir_scalar_is_const(index)) {
      add = nir_scalar_as_uint(index);
      index = (nir_scalar){0};
   } else {
      if (index.def->bit_size != 16 && index.def->bit_size != 32)
         return false;

      if (nir_scalar_is_alu(index) && nir_scalar_alu_op(index) == nir_op_iadd) {
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar term = nir_scalar_chase_alu_src(index, i);
            if (nir_scalar_is_const(term)) {
               add = nir_scalar_as_uint(term);
               index = nir_scalar_chase_alu_src(index, 1 - i);
               break;
            }
         }
      }

      upper = nir_unsigned_upper_bound(shader, ranges, index);
   }

   if (upper + add > UINT16_MAX)
      return false;

   const uint64_t byte_offset = base + 4 * add;
   if (byte_offset + 4 * upper + 4 > shader->info.shared_size)
      return false;

   *address = (struct shared_address){ index, byte_offset };
   return true;
}

static bool
pair_reads(nir_builder *b, struct hash_table *ranges, nir_instr *begin,
           nir_instr *end)
{
   bool progress = false;
   for (nir_instr *low = nir_instr_next(begin); low != end;) {
      nir_instr *next = nir_instr_next(low);
      struct shared_address first;
      if (shared_address32(b->shader, ranges, low, &first)) {
         unsigned distance = 0;
         for (nir_instr *high = next; high != end && distance < SHARED_PAIR_WINDOW;
              high = nir_instr_next(high), ++distance) {
            struct shared_address second;
            if (!shared_address32(b->shader, ranges, high, &second) ||
                !nir_scalar_equal(first.index, second.index) ||
                second.byte_offset != first.byte_offset + 4)
               continue;

            nir_intrinsic_instr *lo = nir_instr_as_intrinsic(low);
            nir_intrinsic_instr *hi = nir_instr_as_intrinsic(high);
            b->cursor = nir_before_instr(low);
            nir_def *pair =
               nir_load_shared(b, 2, 32, lo->src[0].ssa,
                               .base = nir_intrinsic_base(lo),
                               .align_mul = nir_intrinsic_align_mul(lo),
                               .align_offset = nir_intrinsic_align_offset(lo));
            nir_def_rewrite_uses(&lo->def, nir_channel(b, pair, 0));
            nir_def_rewrite_uses(&hi->def, nir_channel(b, pair, 1));

            if (next == high)
               next = nir_instr_next(high);
            nir_instr_remove(high);
            nir_instr_remove(low);
            progress = true;
            break;
         }
      }
      low = next;
   }

   return progress;
}

/* Track only a single convergent block between two workgroup fences. All
 * participating lanes execute both reads and no lane writes the allocation.
 * Keep the first load in place and move only its adjacent word up to it.
 */
bool
agx_nir_opt_shared_vec2(nir_shader *shader)
{
   if (!shader->info.shared_element_index_agx || !shader->info.shared_size)
      return false;

   struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
   bool any = false;
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool progress = false;
      nir_foreach_block(block, impl) {
         if (block->divergent)
            continue;

         nir_instr *begin = NULL;
         nir_foreach_instr(instr, block) {
            if (agx_nir_is_shared_phase_barrier(instr)) {
               if (begin)
                  progress |= pair_reads(&b, ranges, begin, instr);
               begin = instr;
            } else if (instr->type != nir_instr_type_alu &&
                       instr->type != nir_instr_type_load_const &&
                       !plain_shared_load(instr)) {
               begin = NULL;
            }
         }
      }
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }
   _mesa_hash_table_destroy(ranges, NULL);
   return any;
}
