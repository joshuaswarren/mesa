/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2021 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"

UNUSED static void
print_copy(const struct agx_copy *cp)
{
   printf("%sr%u = ", cp->dest_mem ? "m" : "", cp->dest);
   agx_print_index(cp->src, false, stdout);
   printf("\n");
}

UNUSED static void
print_copies(const struct agx_copy *copies, unsigned nr)
{
   printf("[\n");

   for (unsigned i = 0; i < nr; ++i) {
      printf("  ");
      print_copy(&copies[i]);
   }

   printf("]\n");
}

/*
 * Emits code for
 *
 *    for (int i = 0; i < n; ++i)
 *       registers[dests[i]] = registers[srcs[i]];
 *
 * ...with all copies happening in parallel.
 *
 * That is, emit machine instructions equivalent to a parallel copy. This is
 * used to lower not only parallel copies but also collects and splits, which
 * also have parallel copy semantics.
 *
 * We only handles register-register copies, not general agx_index sources. This
 * suffices for its internal use for register allocation.
 */

static void
do_copy(agx_builder *b, const struct agx_copy *copy)
{
   agx_index dst = copy->dest_mem
                      ? agx_memory_register(copy->dest, copy->src.size)
                      : agx_register(copy->dest, copy->src.size);

   const bool via_scratch =
      copy->dest_mem && (copy->src.memory ||
                         copy->src.type == AGX_INDEX_IMMEDIATE ||
                         copy->src.type == AGX_INDEX_UNIFORM);

   if (via_scratch) {
      /* Memory is written from a register only. Memory-memory copies, and
       * immediate or uniform phi sources of spilled phis (agx_spill.c), are
       * lowered to a register write and a register-memory copy, using a
       * reserved scratch register. Keep AGX3's r1 available for a full-width
       * divergent-shuffle index when required.
       */
      const bool agx3_shuffle =
         b->shader->any_quad_divergent_shuffle &&
         b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
      agx_index scratch_reg =
         agx_register(agx3_shuffle ? 4 : 2, copy->src.size);
      if (copy->src.type == AGX_INDEX_IMMEDIATE)
         agx_mov_imm_to(b, scratch_reg, copy->src.value);
      else
         agx_mov_to(b, scratch_reg, copy->src);
      agx_mov_to(b, dst, scratch_reg);
   } else if (copy->src.type == AGX_INDEX_IMMEDIATE) {
      agx_mov_imm_to(b, dst, copy->src.value);
   } else {
      agx_mov_to(b, dst, copy->src);
   }
}

static void
do_swap(agx_builder *b, const struct agx_copy *copy)
{
   assert(copy->src.type == AGX_INDEX_REGISTER && "only GPRs are swapped");

   if (copy->dest == copy->src.value)
      return;

   agx_index x = copy->dest_mem
                    ? agx_memory_register(copy->dest, copy->src.size)
                    : agx_register(copy->dest, copy->src.size);
   agx_index y = copy->src;
   assert(x.memory == y.memory);

   /* A copy's abs/neg is a float modifier composed into the move. AGX3's XOR
    * swap has no field for one, so swap the registers themselves and apply
    * the modifier to the moved value afterwards.
    */
   const bool agx3_modified =
      b->shader->key &&
      b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
      (y.abs || y.neg);
   if (agx3_modified)
      y.abs = y.neg = false;

   /* Memory-memory swaps lowered here, GPR swaps lowered later */
   if (x.memory) {
      agx_index temp1 = agx_register(4, copy->src.size);
      agx_index temp2 = agx_register(6, copy->src.size);

      agx_mov_to(b, temp1, x);
      agx_mov_to(b, temp2, y);
      agx_mov_to(b, y, temp1);
      agx_mov_to(b, x, temp2);
   } else {
      agx_swap(b, x, y);
   }

   if (agx3_modified) {
      agx_index modified = x;
      modified.abs = copy->src.abs;
      modified.neg = copy->src.neg;
      agx_mov_to(b, x, modified);
   }
}

#define PCOPY_STACK_REGS    AGX_NUM_REGS
#define PCOPY_STACK_ENTRIES 128

struct copy_ctx {
   /* Number of copies being processed and allocated capacity */
   unsigned entry_count;
   unsigned entry_cap;

   /* Number of modeled registers tracked in the tables below */
   unsigned reg_count;

   /* For each physreg, the number of pending copy entries that use it as a
    * source. Once this drops to zero, then the physreg is unblocked and can
    * be moved to.
    */
   unsigned *physreg_use_count;

   /* For each physreg, the pending copy_entry that uses it as a dest. */
   struct agx_copy **physreg_dest;

   struct agx_copy *entries;
};

static bool
entry_blocked(struct agx_copy *entry, struct copy_ctx *ctx)
{
   for (unsigned i = 0; i < agx_size_align_16(entry->src.size); i++) {
      if (ctx->physreg_use_count[entry->dest + i] != 0)
         return true;
   }

   return false;
}

static bool
is_real(struct agx_copy *entry)
{
   return entry->src.type == AGX_INDEX_REGISTER &&
          entry->dest_mem == entry->src.memory;
}

/* TODO: Generalize to other bit sizes */
static void
split_32bit_copy(struct copy_ctx *ctx, struct agx_copy *entry)
{
   assert(!entry->done);
   assert(is_real(entry));
   assert(agx_size_align_16(entry->src.size) == 2);
   assert(ctx->entry_count < ctx->entry_cap);
   assert(entry->dest + 1 < ctx->reg_count);
   struct agx_copy *new_entry = &ctx->entries[ctx->entry_count++];

   new_entry->dest = entry->dest + 1;
   new_entry->dest_mem = entry->dest_mem;
   new_entry->src = entry->src;
   new_entry->src.value += 1;
   new_entry->done = false;
   entry->src.size = AGX_SIZE_16;
   new_entry->src.size = AGX_SIZE_16;
   ctx->physreg_dest[entry->dest + 1] = new_entry;
}

static void
agx_emit_parallel_copies_for_class(agx_builder *b, struct agx_copy *copies,
                                   unsigned num_copies, bool cls,
                                   agx_index temp)
{
   /* A GPR temporary lets a cycle be broken with one extra move: save the
    * value a blocked copy is about to overwrite, retarget the copies that read
    * it, and the cycle becomes a path for step 1.  Only GPR copies use it;
    * memory copies keep the swap path. */
   const bool use_temp = !cls && !agx_is_null(temp) &&
                         temp.type == AGX_INDEX_REGISTER && !temp.memory;

   /* Count matching entries (splitting 64-bit copies into two 32-bit chunks)
    * and find the tight register bound so GPR copies stay on a small stack
    * scratch buffer and high-slot memory copies only size to the slots used.
    */
   unsigned num_entries = 0;
   unsigned reg_count = 0;

   for (unsigned i = 0; i < num_copies; ++i) {
      const struct agx_copy *copy = &copies[i];
      if (copy->dest_mem != cls)
         continue;

      const unsigned width = agx_size_align_16(copy->src.size);
      assert(copy->dest + width <= AGX_NUM_MODELED_REGS);
      num_entries += (copy->src.size == AGX_SIZE_64) ? 2 : 1;
      reg_count = MAX2(reg_count, copy->dest + width);

      if (copy->src.type == AGX_INDEX_REGISTER &&
          copy->dest_mem == copy->src.memory) {
         assert(copy->src.value + width <= AGX_NUM_MODELED_REGS);
         reg_count = MAX2(reg_count, copy->src.value + width);
      }
   }

   if (num_entries == 0)
      return;

   if (use_temp)
      reg_count = MAX2(reg_count, MIN2(temp.value + 3, (unsigned)AGX_NUM_REGS));

   /* Each 32-bit entry can split at most once into two 16-bit entries. */
   const unsigned entry_cap = num_entries * 2;

   unsigned stack_use_count[PCOPY_STACK_REGS];
   struct agx_copy *stack_dest[PCOPY_STACK_REGS];
   struct agx_copy stack_entries[PCOPY_STACK_ENTRIES];
   void *heap_buf = NULL;

   struct copy_ctx ctx_storage = {
      .entry_count = num_entries,
      .entry_cap = entry_cap,
      .reg_count = reg_count,
   };
   struct copy_ctx *ctx = &ctx_storage;

   if (reg_count <= PCOPY_STACK_REGS && entry_cap <= PCOPY_STACK_ENTRIES) {
      memset(stack_use_count, 0, reg_count * sizeof(stack_use_count[0]));
      memset(stack_dest, 0, reg_count * sizeof(stack_dest[0]));
      ctx->physreg_use_count = stack_use_count;
      ctx->physreg_dest = stack_dest;
      ctx->entries = stack_entries;
   } else {
      const size_t dest_bytes = reg_count * sizeof(ctx->physreg_dest[0]);
      const size_t entries_bytes = entry_cap * sizeof(ctx->entries[0]);
      const size_t use_bytes = reg_count * sizeof(ctx->physreg_use_count[0]);
      static_assert(_Alignof(struct agx_copy) <= _Alignof(struct agx_copy *),
                    "entries naturally aligned after pointer table");
      static_assert(_Alignof(unsigned) <= _Alignof(struct agx_copy),
                    "use_count naturally aligned after entries");

      heap_buf = calloc(1, dest_bytes + entries_bytes + use_bytes);
      if (heap_buf == NULL)
         UNREACHABLE("failed to allocate parallel-copy context");

      uint8_t *p = heap_buf;
      ctx->physreg_dest = (struct agx_copy **)p;
      p += dest_bytes;
      ctx->entries = (struct agx_copy *)p;
      p += entries_bytes;
      ctx->physreg_use_count = (unsigned *)p;
   }

   /* Lower away 64-bit copies to 32-bit chunks directly into ctx->entries,
    * since we don't have 64-bit ALU so we always want to split.
    */
   unsigned out_idx = 0;
   for (unsigned i = 0; i < num_copies; ++i) {
      struct agx_copy copy = copies[i];
      if (copy.dest_mem != cls)
         continue;

      copy.done = false;
      if (copy.src.size == AGX_SIZE_64) {
         copy.src.size = AGX_SIZE_32;
         ctx->entries[out_idx++] = copy;

         if (copy.src.type == AGX_INDEX_IMMEDIATE) {
            static_assert(sizeof(copy.src.value) * 8 == 32, "known size");
            copy.src.value = 0;
         } else {
            assert(copy.src.type == AGX_INDEX_REGISTER ||
                   copy.src.type == AGX_INDEX_UNIFORM);

            copy.src.value += 2;
         }

         copy.dest += 2;
         ctx->entries[out_idx++] = copy;
      } else {
         ctx->entries[out_idx++] = copy;
      }
   }
   assert(out_idx == ctx->entry_count);

   for (unsigned i = 0; i < ctx->entry_count; i++) {
      struct agx_copy *entry = &ctx->entries[i];

      for (unsigned j = 0; j < agx_size_align_16(entry->src.size); j++) {
         if (is_real(entry))
            ctx->physreg_use_count[entry->src.value + j]++;

         /* Copies should not have overlapping destinations. */
         assert(!ctx->physreg_dest[entry->dest + j]);
         ctx->physreg_dest[entry->dest + j] = entry;
      }
   }

   /* Try to vectorize aligned 16-bit copies to use 32-bit operations instead */
   for (unsigned i = 0; i < ctx->entry_count; i++) {
      struct agx_copy *entry = &ctx->entries[i];
      if (entry->src.size != AGX_SIZE_16)
         continue;

      if ((entry->dest & 1) || (entry->src.value & 1))
         continue;

      if (entry->src.type != AGX_INDEX_UNIFORM &&
          entry->src.type != AGX_INDEX_REGISTER)
         continue;

      unsigned next_dest = entry->dest + 1;
      if (next_dest >= ctx->reg_count)
         continue;

      struct agx_copy *next_copy = ctx->physreg_dest[next_dest];
      if (!next_copy)
         continue;

      assert(next_copy->dest == next_dest && "data structure invariant");
      assert(next_copy->src.size == AGX_SIZE_16 && "unaligned copy");

      if (next_copy->src.type != entry->src.type)
         continue;

      if (next_copy->src.value != (entry->src.value + 1))
         continue;

      /* Vectorize the copies */
      ctx->physreg_dest[next_dest] = entry;
      entry->src.size = AGX_SIZE_32;
      next_copy->done = true;
   }

   bool progress = true;
   while (progress) {
      progress = false;

      /* Step 1: resolve paths in the transfer graph. This means finding
       * copies whose destination aren't blocked by something else and then
       * emitting them, continuing this process until every copy is blocked
       * and there are only cycles left.
       *
       * TODO: We should note that src is also available in dest to unblock
       * cycles that src is involved in.
       */

      for (unsigned i = 0; i < ctx->entry_count; i++) {
         struct agx_copy *entry = &ctx->entries[i];
         if (!entry->done && !entry_blocked(entry, ctx)) {
            entry->done = true;
            progress = true;
            do_copy(b, entry);
            for (unsigned j = 0; j < agx_size_align_16(entry->src.size); j++) {
               if (is_real(entry))
                  ctx->physreg_use_count[entry->src.value + j]--;
               ctx->physreg_dest[entry->dest + j] = NULL;
            }
         }
      }

      if (progress)
         continue;

      /* Step 2: Find partially blocked copies and split them. In the
       * mergedregs case, we can 32-bit copies which are only blocked on one
       * 16-bit half, and splitting them helps get things moving.
       *
       * We can skip splitting copies if the source isn't a register,
       * however, because it does not unblock anything and therefore doesn't
       * contribute to making forward progress with step 1. These copies
       * should still be resolved eventually in step 1 because they can't be
       * part of a cycle.
       */
      for (unsigned i = 0; i < ctx->entry_count; i++) {
         struct agx_copy *entry = &ctx->entries[i];
         if (entry->done || (agx_size_align_16(entry->src.size) != 2))
            continue;

         if (((ctx->physreg_use_count[entry->dest] == 0 ||
               ctx->physreg_use_count[entry->dest + 1] == 0)) &&
             is_real(entry)) {
            split_32bit_copy(ctx, entry);
            progress = true;
         }
      }

      if (progress || !use_temp)
         continue;

      /* Step 2b: only cycles remain.  Break one through the temporary: the
       * value in a blocked copy's destination is saved, the copy is emitted,
       * and every copy reading that destination now reads the temporary.
       * The cycle is then a path, so step 1 resumes.  Costs k + 1 moves for a
       * k-element cycle instead of 3 (k - 1) XORs.
       */
      for (unsigned i = 0; i < ctx->entry_count; i++) {
         struct agx_copy *entry = &ctx->entries[i];
         if (entry->done)
            continue;

         assert(is_real(entry));
         if (entry->dest == entry->src.value) {
            entry->done = true;
            progress = true;
            break;
         }

         const unsigned width = agx_size_align_16(entry->src.size);
         agx_index saved = agx_register(temp.value, entry->src.size);
         agx_index victim = agx_register(entry->dest, entry->src.size);
         agx_mov_to(b, saved, victim);

         do_copy(b, entry);
         entry->done = true;
         for (unsigned j = 0; j < width; j++) {
            ctx->physreg_use_count[entry->src.value + j]--;
            ctx->physreg_dest[entry->dest + j] = NULL;
         }

         /* Copies whose source lay (wholly or partly) inside the destination
          * now read the saved value.  A 32-bit reader straddling the
          * destination boundary is split first so each half can be redirected
          * independently. */
         if (width == 1) {
            for (unsigned j = 0; j < ctx->entry_count; j++) {
               struct agx_copy *blocking = &ctx->entries[j];
               if (blocking->done || !is_real(blocking))
                  continue;
               if (blocking->src.value <= entry->dest &&
                   blocking->src.value + 1 >= entry->dest &&
                   agx_size_align_16(blocking->src.size) == 2)
                  split_32bit_copy(ctx, blocking);
            }
         }

         for (unsigned j = 0; j < ctx->entry_count; j++) {
            struct agx_copy *blocking = &ctx->entries[j];
            if (blocking->done || !is_real(blocking))
               continue;
            if (blocking->src.value >= entry->dest &&
                blocking->src.value < entry->dest + width) {
               const unsigned offset = blocking->src.value - entry->dest;
               /* The temporary is live until the retargeted copies drain,
                * which step 1 does before the next cycle break: the cycle is
                * now a path whose every copy becomes unblocked in order. */
               ctx->physreg_use_count[blocking->src.value]--;
               blocking->src.value = temp.value + offset;
               ctx->physreg_use_count[blocking->src.value]++;
            }
         }

         progress = true;
         break;
      }
   }

   /* Step 3: resolve cycles through swapping.
    *
    * At this point, the transfer graph should consist of only cycles.
    * The reason is that, given any physreg n_1 that's the source of a
    * remaining entry, it has a destination n_2, which (because every
    * copy is blocked) is the source of some other copy whose destination
    * is n_3, and so we can follow the chain until we get a cycle. If we
    * reached some other node than n_1:
    *
    *  n_1 -> n_2 -> ... -> n_i
    *          ^             |
    *          |-------------|
    *
    *  then n_2 would be the destination of 2 copies, which is illegal
    *  (checked above in an assert). So n_1 must be part of a cycle:
    *
    *  n_1 -> n_2 -> ... -> n_i
    *  ^                     |
    *  |---------------------|
    *
    *  and this must be only cycle n_1 is involved in, because any other
    *  path starting from n_1 would also have to end in n_1, resulting in
    *  a node somewhere along the way being the destination of 2 copies
    *  when the 2 paths merge.
    *
    *  The way we resolve the cycle is through picking a copy (n_1, n_2)
    *  and swapping n_1 and n_2. This moves n_1 to n_2, so n_2 is taken
    *  out of the cycle:
    *
    *  n_1 -> ... -> n_i
    *  ^              |
    *  |--------------|
    *
    *  and we can keep repeating this until the cycle is empty.
    */

   for (unsigned i = 0; i < ctx->entry_count; i++) {
      struct agx_copy *entry = &ctx->entries[i];
      if (entry->done)
         continue;

      assert(is_real(entry));

      /* catch trivial copies */
      if (entry->dest == entry->src.value) {
         entry->done = true;
         continue;
      }

      do_swap(b, entry);

      /* Split any blocking copies whose sources are only partially
       * contained within our destination.
       */
      if (agx_size_align_16(entry->src.size) == 1) {
         for (unsigned j = 0; j < ctx->entry_count; j++) {
            struct agx_copy *blocking = &ctx->entries[j];

            if (blocking->done)
               continue;

            if (blocking->src.value <= entry->dest &&
                blocking->src.value + 1 >= entry->dest &&
                agx_size_align_16(blocking->src.size) == 2) {
               split_32bit_copy(ctx, blocking);
            }
         }
      }

      /* Update sources of blocking copies.
       *
       * Note: at this point, every blocking copy's source should be
       * contained within our destination.
       */
      for (unsigned j = 0; j < ctx->entry_count; j++) {
         struct agx_copy *blocking = &ctx->entries[j];
         if (blocking->src.value >= entry->dest &&
             blocking->src.value <
                entry->dest + agx_size_align_16(entry->src.size)) {
            blocking->src.value =
               entry->src.value + (blocking->src.value - entry->dest);
         }
      }

      entry->done = true;
   }

   free(heap_buf);
}

void
agx_emit_parallel_copies_temp(agx_builder *b, struct agx_copy *copies,
                              unsigned num_copies, agx_index temp)
{
   /* Emit copies fo reach register class separately because we don't have
    * register class awareness in the parallel copy lowering data structure.
    */
   if (num_copies == 0)
      return;

   agx_emit_parallel_copies_for_class(b, copies, num_copies, false, temp);
   agx_emit_parallel_copies_for_class(b, copies, num_copies, true, temp);
}

void
agx_emit_parallel_copies(agx_builder *b, struct agx_copy *copies,
                         unsigned num_copies)
{
   agx_emit_parallel_copies_temp(b, copies, num_copies, agx_null());
}
