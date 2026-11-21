/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/bitset.h"
#include "util/macros.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_qsort.h"
#include "util/u_debug.h"

#include "agx3_copy_core.h"
#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_opcodes.h"
#include "shader_enums.h"

/* The eight scalars collected into an operand of a 16x16x16 matrix
 * multiply-accumulate, as the phi webs they belong to. The instruction takes
 * each operand from an aligned run of registers, so the webs want consecutive
 * registers of one such run: then the collect, the phis around it and the
 * copy into the accumulator the instruction overwrites are all in place.
 */
struct matrix_group {
   uint32_t root[8];

   /* Registers (16-bit halves) of each member */
   uint8_t size;
};

/* SSA-based register allocator */
struct phi_web_node {
   /* Parent index, or circular for root */
   uint32_t parent;

   /* If root, assigned register, or ~0 if no register assigned. */
   uint16_t reg;
   bool assigned;

   /* Rank, at most log2(n) so need ~5-bits */
   uint8_t rank;
};
static_assert(sizeof(struct phi_web_node) == 8, "packed");

static unsigned
phi_web_find(struct phi_web_node *web, unsigned x)
{
   if (web[x].parent == x) {
      /* Root */
      return x;
   } else {
      /* Search up the tree */
      unsigned root = x;
      while (web[root].parent != root)
         root = web[root].parent;

      /* Compress path. Second pass ensures O(1) memory usage. */
      while (web[x].parent != x) {
         unsigned temp = web[x].parent;
         web[x].parent = root;
         x = temp;
      }

      return root;
   }
}

static void
phi_web_union(struct phi_web_node *web, unsigned x, unsigned y)
{
   x = phi_web_find(web, x);
   y = phi_web_find(web, y);

   if (x == y)
      return;

   /* Union-by-rank: ensure x.rank >= y.rank */
   if (web[x].rank < web[y].rank) {
      SWAP(x, y);
   }

   web[y].parent = x;

   /* Increment rank if necessary */
   if (web[x].rank == web[y].rank) {
      web[x].rank++;
   }
}

struct ra_ctx {
   agx_context *shader;
   agx_block *block;
   agx_instr *instr;
   uint16_t *ssa_to_reg;
   uint8_t *ncomps;
   uint8_t *ncomps_unrounded;
   enum agx_size *sizes;
   enum ra_class *classes;
   BITSET_WORD *visited;
   BITSET_WORD *used_regs[RA_CLASSES];

   /* SSA values read by some instruction; see AGX_OPCODE_ENTER_QUERY_M5. */
   const BITSET_WORD *dest_read;

   /* Were any sources killed early this instruction? We assert this is not true
    * when shuffling.
    */
   bool early_killed;
   bool transient_texture_used;

   /* Maintained while assigning registers. Count of registers required, i.e.
    * the maximum register assigned + 1.
    */
   unsigned *count[RA_CLASSES];

   /* For affinities */
   agx_instr **src_to_collect_phi;
   BITSET_WORD *early_scalar_export;
   struct phi_web_node *phi_web;

   /* Matrix operand groups, and for each phi web root its group plus one
    * (zero for none).
    */
   const struct matrix_group *matrix_groups;
   const uint16_t *web_matrix_group;

   /* If bit i of used_regs is set, and register i is the first consecutive
    * register holding an SSA value, then reg_to_ssa[i] is the SSA index of the
    * value currently in register  i.
    *
    * Only for GPRs. We can add reg classes later if we have a use case.
    */
   uint32_t reg_to_ssa[AGX_NUM_REGS];

   /* Maximum number of registers that RA is allowed to use */
   unsigned bound[RA_CLASSES];

   /* SSA values that must begin in G17P's corpus-grounded low store bank. */
   BITSET_WORD *low_gpr;
   unsigned tile_gpr_limit;

   /* SSA values read by G17P's seven-bit texture COPY/CEM source fields. */
   BITSET_WORD *texture_gpr;

   /* SSA values read by G17P's 192-half f32log2 operand domain. */
   BITSET_WORD *log2_gpr;

   /* SSA coordinates that must stay above CEM's fixed scratch register bank
    * or a fully occupied MRT store bank. */
   BITSET_WORD *high_gpr;

   /* Values that must be allocated below half 128: coordinates of a
    * cube-array sample with a register LOD (the fused layer/face/LOD
    * PACK_SMPIR operand carries a seven-bit layer register). */
   BITSET_WORD *tex_low128;
};

/*
 * RA treats the nesting counter, the divergent shuffle temporary, and the
 * spiller temporaries as alive throughout if used anywhere. This could be
 * optimized. Using a single power-of-two reserved region at the start ensures
 * these registers are never shuffled.
 */
static unsigned
reserved_size(agx_context *ctx)
{
   return agx_reserved_registers(ctx);
}

UNUSED static void
print_reg_file(struct ra_ctx *rctx, FILE *fp)
{
   unsigned reserved = reserved_size(rctx->shader);

   /* Dump the contents */
   for (unsigned i = reserved; i < rctx->bound[RA_GPR]; ++i) {
      if (BITSET_TEST(rctx->used_regs[RA_GPR], i)) {
         uint32_t ssa = rctx->reg_to_ssa[i];
         unsigned n = rctx->ncomps[ssa];
         fprintf(fp, "h%u...%u: %u\n", i, i + n - 1, ssa);
         i += (n - 1);
      }
   }
   fprintf(fp, "\n");

   /* Dump a visualization of the sizes to understand what live range
    * splitting is up against.
    */
   for (unsigned i = 0; i < rctx->bound[RA_GPR]; ++i) {
      /* Space out 16-bit vec4s */
      if (i && (i % 4) == 0) {
         fprintf(fp, " ");
      }

      if (i < reserved) {
         fprintf(fp, "-");
      } else if (BITSET_TEST(rctx->used_regs[RA_GPR], i)) {
         uint32_t ssa = rctx->reg_to_ssa[i];
         unsigned n = rctx->ncomps[ssa];
         for (unsigned j = 0; j < n; ++j) {
            assert(n < 10);
            fprintf(fp, "%u", n);
         }

         i += (n - 1);
      } else {
         fprintf(fp, ".");
      }
   }
   fprintf(fp, "\n\n");
}

enum agx_size
agx_split_width(const agx_instr *I)
{
   enum agx_size width = ~0;

   agx_foreach_dest(I, d) {
      if (I->dest[d].type == AGX_INDEX_NULL)
         continue;
      else if (width != ~0)
         assert(width == I->dest[d].size);
      else
         width = I->dest[d].size;
   }

   assert(width != ~0 && "should have been DCE'd");
   return width;
}

/* SSA values read by any instruction (including phis). */
static BITSET_WORD *
agx_values_read(agx_context *ctx)
{
   BITSET_WORD *read = calloc(BITSET_WORDS(ctx->alloc), sizeof(BITSET_WORD));

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         BITSET_SET(read, I->src[s].value);
   }

   return read;
}

/*
 * Calculate register demand in 16-bit registers, while gathering widths and
 * classes. Becuase we allocate in SSA, this calculation is exact in
 * linear-time. Depends on liveness information.
 */
unsigned
agx_calc_register_demand(agx_context *ctx, bool remat)
{
   /* Print detailed demand calculation, helpful to debug spilling */
   bool debug = false;

   if (debug) {
      agx_print_shader(ctx, stdout);
   }

   uint8_t *widths = calloc(ctx->alloc, sizeof(uint8_t));
   enum ra_class *classes = calloc(ctx->alloc, sizeof(enum ra_class));
   BITSET_WORD *dest_read = agx_values_read(ctx);

   agx_foreach_instr_global(ctx, I) {
      if (remat && (I->op == AGX_OPCODE_MOV_IMM ||
                    agx_can_rematerialize_uniform(ctx, I)))
         continue;

      agx_foreach_ssa_dest(I, d) {
         unsigned v = I->dest[d].value;
         assert(widths[v] == 0 && "broken SSA");
         /* Round up vectors for easier live range splitting */
         widths[v] = util_next_power_of_two(agx_index_size_16(I->dest[d]));
         classes[v] = ra_class_for_index(I->dest[d]);
      }
   }

   /* Calculate demand at the start of each block based on live-in, then update
    * for each instruction processed. Calculate rolling maximum.
    */
   unsigned max_demand = 0;

   agx_foreach_block(ctx, block) {
      unsigned demand = reserved_size(ctx);

      /* Everything live-in */
      U_SPARSE_BITSET_FOREACH_SET(&block->live_in, i) {
         if (classes[i] == RA_GPR)
            demand += widths[i];
      }

      max_demand = MAX2(demand, max_demand);

      /* To handle late-kill sources, this counter tracks the number of
       * registers to be freed after 1 extra instruction.
       */
      unsigned late_kill_count = 0;

      if (debug) {
         printf("\n");
      }

      agx_foreach_instr_in_block(block, I) {
         /* Phis happen in parallel and are already accounted for in the live-in
          * set, just skip them so we don't double count.
          */
         if (I->op == AGX_OPCODE_PHI)
            continue;

         if (remat && (I->op == AGX_OPCODE_MOV_IMM ||
                       agx_can_rematerialize_uniform(ctx, I)))
            continue;

         if (debug) {
            printf("%u: ", demand);
            agx_print_instr(I, stdout);
         }

         if (I->op == AGX_OPCODE_PRELOAD) {
            unsigned size = agx_size_align_16(I->src[0].size);
            max_demand = MAX2(max_demand, I->src[0].value + size);
         } else if (I->op == AGX_OPCODE_EXPORT) {
            unsigned size = agx_size_align_16(I->src[0].size);
            max_demand = MAX2(max_demand, I->imm + size);
         }

         /* Handle late-kill registers from last instruction */
         demand -= late_kill_count;
         late_kill_count = 0;

         /* Late-kill sources the first time we see them. This simplifies RA. We
          * could optimize to early-kill in some situations if we wanted.
          */
         agx_foreach_src(I, s) {
            if (!I->src[s].kill)
               continue;
            assert(I->src[s].type == AGX_INDEX_NORMAL);
            if (ra_class_for_index(I->src[s]) != RA_GPR)
               continue;

            bool skip = false;

            for (unsigned backwards = 0; backwards < s; ++backwards) {
               if (agx_is_equiv(I->src[backwards], I->src[s])) {
                  skip = true;
                  break;
               }
            }

            if (!skip)
               late_kill_count += widths[I->src[s].value];
         }

         /* Make destinations live */
         agx_foreach_ssa_dest(I, d) {
            if (ra_class_for_index(I->dest[d]) != RA_GPR)
               continue;

            /* Live range splits allocate at power-of-two granularity. Round up
             * destination sizes (temporarily) to powers-of-two.
             */
            unsigned real_width = widths[I->dest[d].value];
            unsigned pot_width = util_next_power_of_two(real_width);

            demand += pot_width;
            late_kill_count += (pot_width - real_width);

            /* An unread M5 query-entry result dies right after its write. */
            if (I->op == AGX_OPCODE_ENTER_QUERY_M5 && !BITSET_TEST(dest_read, I->dest[d].value))
               late_kill_count += real_width;
         }

         max_demand = MAX2(demand, max_demand);
      }

      demand -= late_kill_count;
   }

   free(dest_read);
   free(widths);
   free(classes);
   return max_demand;
}

static bool
find_regs_simple(struct ra_ctx *rctx, enum ra_class cls, unsigned count,
                 unsigned align, unsigned minimum, unsigned limit,
                 unsigned *out)
{
   for (unsigned reg = ALIGN_POT(minimum, align); reg + count <= limit;
        reg += align) {
      if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count)) {
         *out = reg;
         return true;
      }
   }

   return false;
}

static unsigned
dest_limit(const struct ra_ctx *rctx, const agx_instr *I, unsigned d)
{
   enum ra_class cls = ra_class_for_index(I->dest[d]);
   unsigned limit = rctx->bound[cls];
   /* G17P TEX encodes a seven-bit half-register destination, for image
    * reads as for texture reads. Coordinate and other staged sources are
    * bounded separately through texture_gpr below. AGX3C's destination is
    * eight bits: G17G samples and reads land at halves 148 and 174 under
    * register pressure (port/rawit k_tex_hi, k_read_hi). G18G keeps the
    * seven-bit bound its mixed descriptor samples were qualified with. */
   if ((rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        agx_is_g18g(&rctx->shader->key->dev)) &&
       (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
        I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD))
      limit = MIN2(limit, 128);
   /* A move from the uniform file packs to a compact COPY on G17P and
    * G18G, which names a seven-bit destination like the half COPY. Other
    * instructions can still allocate the full register file. */
   if ((rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        agx_is_g18g(&rctx->shader->key->dev)) &&
       I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM)
      limit = MIN2(limit, AGX3_COPY_REG16_DEST_HALVES);
   if (rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       BITSET_TEST(rctx->low_gpr, I->dest[d].value))
      limit = MIN2(limit, rctx->tile_gpr_limit);
   if ((rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       ra_class_for_index(I->dest[d]) == RA_GPR &&
       BITSET_TEST(rctx->texture_gpr, I->dest[d].value))
      limit = MIN2(limit, 128);
   /* The G18G stream packer accepts direct coordinate bases through half
    * 60. Keep the complete coordinate vector in the low bank, including
    * after an overlapping TEX destination evicts a still-live value. */
   if (agx_is_g18g(&rctx->shader->key->dev) &&
       ra_class_for_index(I->dest[d]) == RA_GPR &&
       BITSET_TEST(rctx->texture_gpr, I->dest[d].value))
      limit = MIN2(limit, 64);
   if (rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       ra_class_for_index(I->dest[d]) == RA_GPR &&
       (I->op == AGX_OPCODE_LOG2 ||
        BITSET_TEST(rctx->log2_gpr, I->dest[d].value)))
      limit = MIN2(limit, 192);
   if (rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       ra_class_for_index(I->dest[d]) == RA_GPR &&
       rctx->tex_low128 &&
       BITSET_TEST(rctx->tex_low128, I->dest[d].value))
      limit = MIN2(limit, 128);
   return limit;
}

static unsigned
dest_minimum(const struct ra_ctx *rctx, const agx_instr *I, unsigned d)
{
   if (agx_isa_is_agx3(rctx->shader->key->dev.encoding) &&
       ra_class_for_index(I->dest[d]) == RA_GPR &&
       BITSET_TEST(rctx->high_gpr, I->dest[d].value))
      return 16;

   return 0;
}

static bool
ssa_has_bank_constraint(const struct ra_ctx *rctx, unsigned ssa)
{
   if (rctx->shader->key->dev.chip != AGX_CHIP_G17P ||
       rctx->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return BITSET_TEST(rctx->low_gpr, ssa) ||
             BITSET_TEST(rctx->texture_gpr, ssa) ||
             BITSET_TEST(rctx->log2_gpr, ssa) ||
             BITSET_TEST(rctx->high_gpr, ssa);

   /* A bank restriction cannot prevent relocation when the complete register
    * file fits inside that bank. In particular, pinning a texture coordinate
    * in a 112-half allocation can strand the only partly free eight-half
    * region, even though every possible new home satisfies its 128-half
    * encoding limit. Keep the restriction for genuinely narrower banks.
    */
   const unsigned bound = rctx->bound[RA_GPR];
   return (BITSET_TEST(rctx->low_gpr, ssa) && bound > rctx->tile_gpr_limit) ||
          (BITSET_TEST(rctx->texture_gpr, ssa) && bound > 128) ||
          (BITSET_TEST(rctx->log2_gpr, ssa) && bound > 192) ||
          BITSET_TEST(rctx->high_gpr, ssa);
}

/* The interval a bank-constrained value may live in: the same bounds
 * dest_limit() and dest_minimum() gave it when it was defined. */
static void
ssa_bank_range(const struct ra_ctx *rctx, unsigned ssa, unsigned *floor,
               unsigned *limit)
{
   const bool agx3b =
      rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B;
   *floor = BITSET_TEST(rctx->high_gpr, ssa) ? 16 : 0;
   *limit = ~0u;
   if (agx3b && BITSET_TEST(rctx->low_gpr, ssa))
      *limit = MIN2(*limit, rctx->tile_gpr_limit);
   if (BITSET_TEST(rctx->texture_gpr, ssa))
      *limit = MIN2(*limit, 128);
   if (agx3b && BITSET_TEST(rctx->log2_gpr, ssa))
      *limit = MIN2(*limit, 192);
}

/* Whether a value's bank limit narrows the current register file, mirroring
 * dest_limit(): the low-bank and log2 bounds are G17P (AGX3B) encodings, and
 * the seven-bit texture/COPY bound means nothing in a file of 128 halves or
 * fewer, which is what compute shaders get once the workgroup bounds their
 * registers. */
static bool
ssa_has_binding_bank_constraint(const struct ra_ctx *rctx, unsigned ssa)
{
   const bool agx3b =
      rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B;
   const unsigned bound = rctx->bound[RA_GPR];

   return (agx3b && BITSET_TEST(rctx->low_gpr, ssa) &&
           bound > rctx->tile_gpr_limit) ||
          (BITSET_TEST(rctx->texture_gpr, ssa) && bound > 128) ||
          (agx3b && BITSET_TEST(rctx->log2_gpr, ssa) && bound > 192) ||
          BITSET_TEST(rctx->high_gpr, ssa);
}

static bool
region_has_bank_constrained_value(const struct ra_ctx *rctx, unsigned base,
                                  unsigned size, bool binding_only)
{
   for (unsigned reg = reserved_size(rctx->shader);
        reg < rctx->bound[RA_GPR];) {
      if (!BITSET_TEST(rctx->used_regs[RA_GPR], reg)) {
         ++reg;
         continue;
      }

      unsigned ssa = rctx->reg_to_ssa[reg];
      unsigned width = rctx->ncomps[ssa];
      assert(width && rctx->ssa_to_reg[ssa] == reg);
      if (reg < base + size && reg + width > base &&
          (binding_only ? ssa_has_binding_bank_constraint(rctx, ssa)
                        : ssa_has_bank_constraint(rctx, ssa)))
         return true;

      reg += width;
   }

   return false;
}

/*
 * Search the register file for the best contiguous aligned region of the given
 * size to evict when shuffling registers. The region must not contain any
 * register marked in the passed bitset.
 *
 * As a hint, this also takes in the set of registers from killed sources passed
 * to this instruction. These should be deprioritized, since they are more
 * expensive to use (extra moves to shuffle the contents away).
 *
 * Precondition: such a region exists.
 *
 * Postcondition: at least one register in the returned region is already free.
 */
static unsigned
find_best_region_to_evict_impl(struct ra_ctx *rctx, unsigned size,
                               BITSET_WORD *already_evicted, unsigned cap,
                               unsigned floor, bool honor_banks, bool strict)
{
   assert(util_is_power_of_two_or_zero(size) && "precondition");
   assert((rctx->bound[RA_GPR] % size) == 0 &&
          "register file size must be aligned to the maximum vector size");

   /* Useful for testing RA */
   bool invert = false;

   unsigned best_base = ~0;
   unsigned best_moves = invert ? 0 : ~0;
   unsigned bound = MIN2(cap, rctx->bound[RA_GPR]);

   for (unsigned base = 0; base + size <= bound; base += size) {
      /* The first k registers are preallocated and unevictable, so must be
       * skipped. By itself, this does not pose a problem. We are allocating n
       * registers, but this region has at most n-k free.  Since there are at
       * least n free registers total, there is at least k free registers
       * outside this region. Choose any such free register. The region
       * containing it has at most n-1 occupied registers. In the worst case,
       * n-k of those registers are are moved to the beginning region and the
       * remaining (n-1)-(n-k) = k-1 registers are moved to the k-1 free
       * registers in other regions, given there are k free registers total.
       * These recursive shuffles work out because everything is power-of-two
       * sized and naturally aligned, so the sizes shuffled are strictly
       * descending. So, we do not need extra registers to handle "single
       * region" unevictability.
       */
      if (base < reserved_size(rctx->shader) || base < floor)
         continue;

      /* Do not evict the same register multiple times. It's not necessary since
       * we're just shuffling, there are enough free registers elsewhere.
       */
      if (BITSET_TEST_COUNT(already_evicted, base, size))
         continue;

      /* Keep constrained values in their assigned bank across live-range
       * splitting. Otherwise a later destination can relocate a valid TEX
       * coordinate above the seven-bit COPY source range. */
      if (honor_banks && region_has_bank_constrained_value(rctx, base, size, false))
         continue;

      /* Estimate the number of moves required if we pick this region */
      unsigned moves = 0;
      bool any_free = false;

      for (unsigned reg = base; reg < base + size; ++reg) {
         /* We need a move for each blocked register (TODO: we only need a
          * single move for 32-bit pairs, could optimize to use that instead.)
          */
         if (BITSET_TEST(rctx->used_regs[RA_GPR], reg))
            moves++;
         else
            any_free = true;
      }

      /* Pick the region requiring fewest moves as a heuristic. Regions with no
       * free registers are skipped, since the recursive splitting algorithm
       * requires at least one free register.
       */
      if (any_free && ((moves < best_moves) ^ invert)) {
         best_moves = moves;
         best_base = base;
      }
   }

   /* A capped search is a preference, not a requirement: the caller only
    * narrows the file so a half victim's relocation stays inside the seven-bit
    * COPY destination.  If nothing fits down there, fall back to the whole
    * file rather than failing allocation outright. */
   if (best_base == ~0u && bound < rctx->bound[RA_GPR] && !strict)
      return find_best_region_to_evict_impl(rctx, size, already_evicted, ~0u,
                                            floor, honor_banks, strict);

   return best_base;
}

/*
 * `cap` and `floor` bound where the region may start; with `strict` they are
 * requirements (a bank-constrained value's interval), otherwise a preference.
 */
static unsigned
find_best_region_to_evict(struct ra_ctx *rctx, unsigned size,
                          BITSET_WORD *already_evicted, unsigned cap,
                          unsigned floor, bool strict)
{
   /* Prefer not to disturb bank-constrained values. */
   unsigned base = find_best_region_to_evict_impl(
      rctx, size, already_evicted, cap, floor, true, strict);

   /* They are not immovable, though: the free halves around them can all be
    * unreachable even though demand fits the file (CS2 fills G17P's low bank
    * with alternating pinned 32-bit scalars and the high bank solid). Evict
    * them too; the caller re-places each one inside its own bank. */
   if (base == ~0u) {
      base = find_best_region_to_evict_impl(rctx, size, already_evicted, cap,
                                            floor, false, strict);
   }

   /* Still nothing legal: finish allocation consistently anywhere and refuse
    * the result, so agx_compile_function_nir() recompiles with spill headroom.
    * Returning ~0 here used to index the used-register bitset half a gigabyte
    * out of bounds in release builds. */
   if (base == ~0u) {
      rctx->shader->ra_failed = true;
      base = find_best_region_to_evict_impl(rctx, size, already_evicted, ~0u, 0,
                                            false, false);
   }

   /* Every region was already evicted for this value: there is nothing left
    * to finish the allocation with. Leave RA now (agx_ra_impl() frees its
    * state); the caller sees ra_failed and recompiles with a smaller spill
    * budget. Continuing used to assert here in debug builds and index out of
    * bounds in release builds (geometry shader of
    * spirv_assembly.instruction.graphics.16bit_storage.struct_mixed_types.uniform_geom
    * on M5: 248 registers, every 2-aligned region evicted). */
   if (base == ~0u && rctx->shader->ra_abort) {
      assert(rctx->shader->ra_failed);
      longjmp(*(jmp_buf *)rctx->shader->ra_abort, 1);
   }

   assert(base < rctx->bound[RA_GPR] &&
          "not enough registers (should have spilled already)");
   return base;
}

static void
set_ssa_to_reg(struct ra_ctx *rctx, unsigned ssa, unsigned reg)
{
   enum ra_class cls = rctx->classes[ssa];
   *(rctx->count[cls]) = MAX2(*(rctx->count[cls]), reg + rctx->ncomps[ssa]);

   rctx->ssa_to_reg[ssa] = reg;

   if (cls == RA_GPR) {
      rctx->reg_to_ssa[reg] = ssa;
   }
}

/*
 * Insert parallel copies to move an SSA variable `var` to a new register
 * `new_reg`. This may require scalarizing.
 */
static void
insert_copy(struct ra_ctx *rctx, struct util_dynarray *copies, unsigned new_reg,
            unsigned var)
{
   enum agx_size size = rctx->sizes[var];
   unsigned align = agx_size_align_16(size);

   for (unsigned i = 0; i < rctx->ncomps[var]; i += align) {
      struct agx_copy copy = {
         .dest = new_reg + i,
         .src = agx_register(rctx->ssa_to_reg[var] + i, size),
      };

      assert((copy.dest % align) == 0 && "new dest must be aligned");
      assert((copy.src.value % align) == 0 && "src must be aligned");
      util_dynarray_append(copies, copy);
   }
}

/* Chooses a region of `count` halves inside [minimum, limit) that
 * assign_regs_by_copying() can free by relocating whole values.  Returns false
 * when no such region exists, which the caller retries with the half-victim
 * restriction lifted before it gives up. */
static bool
find_constrained_region(struct ra_ctx *rctx, unsigned count, unsigned minimum,
                        unsigned limit, bool avoid_half_copies,
                        unsigned *base_out, unsigned *span_out)
{
   unsigned best = ~0u, best_moves = ~0u, best_span = 0;

   /* agx3_copy_encode_reg16() refuses a destination at or above half 128, so a
    * half victim is only awkward on a register file that actually reaches that
    * far; below it the restriction rejects regions to avoid a copy that cannot
    * exist.  Where it does apply it is a preference: find_best_region_to_evict()
    * keeps the relocation encodable, and the caller retries without it. */
   bool avoid_half = avoid_half_copies &&
                     rctx->bound[RA_GPR] > AGX3_COPY_REG16_DEST_HALVES;

   /* RA vectors are naturally aligned powers of two, at most 16 halves.
    * A result may overlap a larger live vector. Evict that whole vector,
    * rather than interpreting one of its interior halves as another SSA. */
   for (unsigned span = count; span <= 16; span *= 2) {
      for (unsigned base = 0; base + span <= limit; base += span) {
         if (base < reserved_size(rctx->shader))
            continue;
         if (base < minimum)
            continue;
         unsigned moves = 0;
         bool valid = true;
         for (unsigned reg = reserved_size(rctx->shader); reg < limit;) {
            if (!BITSET_TEST(rctx->used_regs[RA_GPR], reg)) {
               ++reg;
               continue;
            }
            unsigned ssa = rctx->reg_to_ssa[reg];
            unsigned width = rctx->ncomps[ssa];
            assert(width && rctx->ssa_to_reg[ssa] == reg);
            if (reg < base + span && reg + width > base) {
               /* G17P's grounded half COPY has a seven-bit destination. A
                * constrained result shuffle cannot preserve a half victim in
                * the high bank, so choose another initial victim region. */
               if (ssa_has_bank_constraint(rctx, ssa) ||
                   (avoid_half && rctx->sizes[ssa] == AGX_SIZE_16)) {
                  valid = false;
                  break;
               }
               if (reg < base || reg + width > base + span) {
                  valid = false;
                  break;
               }
               moves += width / agx_size_align_16(rctx->sizes[ssa]);
            }
            reg += width;
         }
         if (valid && moves < best_moves) {
            best = base;
            best_moves = moves;
            best_span = span;
         }
      }
      /* Prefer the smallest whole-value region before minimizing copies.
       * Otherwise a cheap wider victim can leave all free halves trapped
       * in the reserved prefix and the result's partially occupied block,
       * with no movable region large enough to relocate that victim. */
      if (best != ~0u)
         break;
   }

   if (best >= limit || !best_span)
      return false;

   *span_out = best_span;
   *base_out = best;
   return true;
}

static unsigned
assign_regs_by_copying(struct ra_ctx *rctx, agx_index dest, const agx_instr *I,
                       struct util_dynarray *copies, unsigned minimum,
                       unsigned limit)
{
   BITSET_DECLARE(clobbered, AGX_NUM_REGS) = {0};
   assert(dest.type == AGX_INDEX_NORMAL);

   /* Initialize the worklist with the variable we're assigning */
   unsigned blocked_vars[16] = {dest.value};
   size_t nr_blocked = 1;

   if (minimum > 0 || limit < rctx->bound[RA_GPR]) {
      unsigned count = rctx->ncomps[dest.value], span, base;
      bool agx3b =
         I != NULL && agx_isa_is_agx3(rctx->shader->key->dev.encoding);

      /* Prefer a region with no half victim: relocating one is bounded by the
       * seven-bit COPY destination.  When the constrained window holds nothing
       * else -- eight-target MRT packs the low bank solid -- take the half
       * victim anyway and let find_best_region_to_evict() keep its new home
       * under AGX3_COPY_REG16_DEST_HALVES. */
      bool found =
         find_constrained_region(rctx, count, minimum, limit, agx3b, &base,
                                 &span) ||
         (agx3b && find_constrained_region(rctx, count, minimum, limit, false,
                                           &base, &span));

      /* The tile-store bank (agx3_tile_store_bank()) is sized per colour
       * target, one live payload each. A fragment can hold two payloads for
       * the same target at once -- a vec4 stored on one path while another
       * path's vec4 is still live, as dEQP-VK.glsl.opaque_type_indexing.
       * sampler.*.fragment does -- and with the reserved prefix a 16-half
       * bank then holds only one of them. The bank is a native placement
       * preference, not an encoding bound (agx3_tile_value_ok() takes any
       * half below 256), and Apple's own single-target fragments store from
       * halves up to 31 (id_blend_4x stores RGBA8 from half 28). Widen to
       * the 32- and then the 64-half bank native multi-target fragments use
       * before giving up, instead of aborting the compile. Only a shader
       * that would otherwise abort here takes this path. */
      for (unsigned wide = 32; !found && agx3b && wide <= 64; wide *= 2) {
         const bool tile_bank =
            BITSET_TEST(rctx->low_gpr, dest.value) &&
            limit == rctx->tile_gpr_limit;
         if (!tile_bank || limit >= wide || wide > rctx->bound[RA_GPR])
            continue;
         found = find_constrained_region(rctx, count, minimum, wide, true,
                                         &base, &span) ||
                 find_constrained_region(rctx, count, minimum, wide, false,
                                         &base, &span);
      }

      if (!found) {
         /* No legal placement exists. Never write through an unset base --
          * it feeds BITSET_SET_COUNT() and reg_to_ssa[] -- and never emit the
          * illegal placement either: record the failure, let the ordinary
          * eviction below place the value anywhere so allocation finishes
          * consistently, and let agx_compile_function_nir() refuse the
          * shader through its normal error path. */
         rctx->shader->ra_failed = true;
         goto unconstrained;
      }
      nr_blocked = 0;
      for (unsigned reg = base; reg < base + span;) {
         if (!BITSET_TEST(rctx->used_regs[RA_GPR], reg)) {
            ++reg;
            continue;
         }
         unsigned ssa = rctx->reg_to_ssa[reg];
         unsigned width = rctx->ncomps[ssa];
         assert(rctx->ssa_to_reg[ssa] == reg && reg + width <= base + span);
         assert(nr_blocked < ARRAY_SIZE(blocked_vars));
         blocked_vars[nr_blocked++] = ssa;
         BITSET_CLEAR_COUNT(rctx->used_regs[RA_GPR], reg, width);
         reg += width;
      }
      /* Only reserve the actual result. Freed excess halves of a larger
       * victim remain available to the shuffle. This leaves enough total
       * free space even when the low bank started completely full. Copies
       * execute in parallel before I, preserving every victim's old value. */
      BITSET_SET_COUNT(rctx->used_regs[RA_GPR], base, count);
      BITSET_SET_COUNT(clobbered, base, count);
      set_ssa_to_reg(rctx, dest.value, base);
   }

unconstrained:
   while (nr_blocked > 0) {
      /* Grab the largest var. TODO: Consider not writing O(N^2) code. */
      uint32_t ssa = ~0, nr = 0, chosen_idx = ~0;
      for (unsigned i = 0; i < nr_blocked; ++i) {
         uint32_t this_ssa = blocked_vars[i];
         uint32_t this_nr = rctx->ncomps[this_ssa];

         if (this_nr > nr) {
            nr = this_nr;
            ssa = this_ssa;
            chosen_idx = i;
         }
      }

      assert(ssa != ~0 && nr > 0 && "must have found something");
      assert(chosen_idx < nr_blocked && "must have found something");

      /* Pop it from the work list by swapping in the last element */
      blocked_vars[chosen_idx] = blocked_vars[--nr_blocked];

      /* We need to shuffle some variables to make room. Look for a range of
       * the register file that is partially blocked.
       */
      /* Every iteration but the first emits insert_copy() for this value.  A
       * 16-bit copy has a seven-bit destination, so its new home must stay
       * below AGX3_COPY_REG16_DEST_HALVES on a file that reaches that far. */
      unsigned cap = ~0u;
      if (ssa != dest.value && rctx->sizes[ssa] == AGX_SIZE_16 &&
          agx_isa_is_agx3(rctx->shader->key->dev.encoding))
         cap = AGX3_COPY_REG16_DEST_HALVES;

      /* An evicted bank-constrained value must land back inside its bank. */
      unsigned floor = 0;
      bool strict = ssa != dest.value && ssa_has_bank_constraint(rctx, ssa);
      if (strict) {
         unsigned bank_limit;
         ssa_bank_range(rctx, ssa, &floor, &bank_limit);
         cap = MIN2(cap, bank_limit);
      }

      unsigned new_reg =
         find_best_region_to_evict(rctx, nr, clobbered, cap, floor, strict);

      /* No region can be freed for this value. Never index the register file
       * with the failure value: refuse the shader through
       * agx_compile_function_nir()'s normal error path instead, and hand back
       * an in-range register so allocation can finish. */
      if (new_reg >= rctx->bound[RA_GPR] ||
          new_reg + nr > rctx->bound[RA_GPR]) {
         rctx->shader->ra_failed = true;
         return reserved_size(rctx->shader);
      }

      /* Blocked registers need to get reassigned. Add them to the worklist. */
      for (unsigned i = 0; i < nr; ++i) {
         if (BITSET_TEST(rctx->used_regs[RA_GPR], new_reg + i)) {
            unsigned blocked_reg = new_reg + i;
            uint32_t blocked_ssa = rctx->reg_to_ssa[blocked_reg];
            uint32_t blocked_nr = rctx->ncomps[blocked_ssa];

            assert(blocked_nr >= 1 && "must be assigned");

            blocked_vars[nr_blocked++] = blocked_ssa;
            assert(
               rctx->ssa_to_reg[blocked_ssa] == blocked_reg &&
               "variable must start within the range, since vectors are limited");

            for (unsigned j = 0; j < blocked_nr; ++j) {
               assert(
                  BITSET_TEST(rctx->used_regs[RA_GPR], new_reg + i + j) &&
                  "variable is allocated contiguous and vectors are limited, "
                  "so evicted in full");
            }

            /* Skip to the next variable */
            i += blocked_nr - 1;
         }
      }

      /* We are going to allocate to this range, so it is now fully used. Mark
       * it as such so we don't reassign here later.
       */
      BITSET_SET_COUNT(rctx->used_regs[RA_GPR], new_reg, nr);

      /* The first iteration is special: it is the original allocation of a
       * variable. All subsequent iterations pick a new register for a blocked
       * variable. For those, copy the blocked variable to its new register.
       */
      if (ssa != dest.value) {
         insert_copy(rctx, copies, new_reg, ssa);
      }

      /* Mark down the set of clobbered registers, so that killed sources may be
       * handled correctly later.
       */
      BITSET_SET_COUNT(clobbered, new_reg, nr);

      /* Update bookkeeping for this variable */
      set_ssa_to_reg(rctx, ssa, new_reg);
   }

   return rctx->ssa_to_reg[dest.value];
}

static int
sort_by_size(const void *a_, const void *b_, void *sizes_)
{
   const enum agx_size *sizes = sizes_;
   const unsigned *a = a_, *b = b_;

   return sizes[*b] - sizes[*a];
}

/*
 * When shuffling registers to assign a phi destination, we can't simply insert
 * the required moves before the phi, since phis happen in parallel along the
 * edge. Instead, there are two cases:
 *
 * 1. The source of the copy is the destination of a phi. Since we are
 *    emitting shuffle code, there will be no more reads of that destination
 *    with the old register. Since the phis all happen in parallel and writes
 *    precede reads, there was no previous read of that destination either. So
 *    the old destination is dead. Just replace the phi's destination with the
 *    moves's destination instead.
 *
 * 2. Otherwise, the source of the copy is a live-in value, since it's
 *    live when assigning phis at the start of a block but it is not a phi.
 *    If we move in parallel with the phi, the phi will still read the correct
 *    old register regardless and the destinations can't alias. So, insert a phi
 *    to do the copy in parallel along the incoming edges.
 *
 * The copies are one parallel move that insert_copy() scalarized: a vector
 * value moves as one copy per component, padding included. Each copy is
 * therefore matched against the whole register range a phi defined before
 * this shuffle, and that phi is moved once. Matching only a phi's base
 * register moved a vector phi with its first component and then failed to
 * recognise the rest, which went down the live-in path: a second phi wrote
 * the same registers again from whatever the predecessor held there -- a
 * duplicate parallel-copy destination, or without assertions a clobbered
 * value. Phis not yet given a register (the one being allocated and those
 * after it that were not assigned up front) cannot hold a moved value.
 * agx_as_register() of one is an assertion failure, and without assertions a
 * comparison against a stale register field that can silently drop a copy.
 */
static void
agx_emit_moves_before_phi(struct ra_ctx *rctx, agx_block *block,
                          const struct agx_copy *copies, unsigned nr_copies)
{
   struct phi_def {
      agx_instr *phi;
      unsigned base, width, new_base;
      bool moved;
   };
   struct util_dynarray defs = UTIL_DYNARRAY_INIT;

   agx_foreach_phi_in_block(block, phi) {
      const agx_index d = phi->dest[0];
      struct phi_def def = {.phi = phi};

      if (d.memory)
         continue;

      if (d.type == AGX_INDEX_REGISTER) {
         /* A move inserted by case 2 for an earlier shuffle of this block */
         def.base = d.value;
         def.width = agx_index_size_16(d);
      } else if (d.type == AGX_INDEX_NORMAL && d.has_reg) {
         def.base = d.reg;
         def.width = MAX2(rctx->ncomps[d.value], agx_index_size_16(d));
      } else {
         continue;
      }

      util_dynarray_append(&defs, def);
   }

   for (unsigned i = 0; i < nr_copies; ++i) {
      const struct agx_copy *copy = &copies[i];
      assert(!copy->dest_mem && !copy->src.memory && "no memory shuffles");
      assert(copy->src.type == AGX_INDEX_REGISTER);

      struct phi_def *match = NULL;
      util_dynarray_foreach(&defs, struct phi_def, def) {
         if (copy->src.value >= def->base &&
             copy->src.value < def->base + def->width) {
            match = def;
            break;
         }
      }

      if (match) {
         const unsigned offset = copy->src.value - match->base;
         assert(copy->dest >= offset);
         const unsigned new_base = copy->dest - offset;

         assert((!match->moved || match->new_base == new_base) &&
                "components of one value move together");
         match->new_base = new_base;
         match->moved = true;
         continue;
      }

      /* There wasn't such a phi, so it's live-in. Insert a phi instead. */
      agx_builder b = agx_init_builder(rctx->shader, agx_before_block(block));

      agx_instr *phi = agx_phi_to(&b, agx_register_like(copy->dest, copy->src),
                                  agx_num_predecessors(block));
      assert(!copy->src.kill);

      agx_foreach_src(phi, s) {
         phi->src[s] = copy->src;
      }
   }

   util_dynarray_foreach(&defs, struct phi_def, def) {
      if (!def->moved)
         continue;

      if (def->phi->dest[0].type == AGX_INDEX_REGISTER)
         def->phi->dest[0].value = def->new_base;
      else
         def->phi->dest[0].reg = def->new_base;
   }

   util_dynarray_fini(&defs);
}

static bool
pick_transient_texture_region(struct ra_ctx *rctx, agx_instr *I, unsigned d,
                              unsigned count, unsigned standard, unsigned *out);

/*
 * G17P texture and image results, texture coordinates and moves from the
 * uniform file name seven-bit half-register fields, so they live in halves
 * 0..127, as do M5 texture coordinates, sin/cos operands and its other
 * bank-constrained values. Lowest first, every other value lands there too,
 * and in a shader at the limit of the register file each constrained
 * definition then evicts an unconstrained value from the low bank. The moved
 * values sit in different registers on the two sides of the next merge, which
 * costs a copy per value at the join.
 *
 * Once a shader spills, every register up to the bound is in use somewhere,
 * so where a value goes no longer changes the register count. Put values
 * without a bank constraint above the low bank there, within the registers
 * already used, unless they feed a low-bank collect.
 */
#define G17P_LOW_BANK_HALVES 128

/* Register budget, in halves, of M5 compute kernels that use native ray
 * traversal. Value from 7c91dddf53f (the M5 userspace snapshot), measured on
 * M5 hardware with the full-resolution RT workload: 160 halves was faster
 * than 128 and 192. Not derived; the snapshot records no numbers.
 */
#define AGX_M5_RT_REGISTER_BUDGET 160u

static bool
prefer_high_bank(const struct ra_ctx *rctx, const agx_instr *I, unsigned d,
                 enum ra_class cls, unsigned minimum, unsigned limit)
{
   const enum agx_chip chip = rctx->shader->key->dev.chip;
   if ((chip != AGX_CHIP_G17P && !agx_chip_g17_m5(chip)) ||
       !rctx->shader->has_spill_pcopy_reserved || cls != RA_GPR ||
       minimum != 0 || limit != rctx->bound[cls] ||
       limit <= G17P_LOW_BANK_HALVES ||
       (chip == AGX_CHIP_G17P &&
        ssa_has_bank_constraint(rctx, I->dest[d].value)))
      return false;

   const agx_instr *use = rctx->src_to_collect_phi[I->dest[d].value];
   return !use || use->op != AGX_OPCODE_COLLECT ||
          dest_limit(rctx, use, 0) > G17P_LOW_BANK_HALVES;
}

static unsigned
find_regs(struct ra_ctx *rctx, agx_instr *I, unsigned dest_idx, unsigned count,
          unsigned align)
{
   unsigned reg;
   assert(count == align);

   enum ra_class cls = ra_class_for_index(I->dest[dest_idx]);
   unsigned minimum = dest_minimum(rctx, I, dest_idx);
   unsigned limit = dest_limit(rctx, I, dest_idx);

   bool simple = prefer_high_bank(rctx, I, dest_idx, cls, minimum, limit) &&
                 find_regs_simple(rctx, cls, count, align,
                                  G17P_LOW_BANK_HALVES,
                                  MIN2(*rctx->count[cls], limit), &reg);
   if (!simple)
      simple = find_regs_simple(rctx, cls, count, align, minimum, limit, &reg);
   unsigned transient;
   if (pick_transient_texture_region(rctx, I, dest_idx, count,
                                    simple ? reg : ~0u, &transient)) {
      rctx->transient_texture_used = true;
      return transient;
   }
   if (simple) {
      return reg;
   } else {
      assert(!rctx->early_killed && "no live range splits with early kill");
      assert(cls == RA_GPR && "no memory live range splits");

      struct util_dynarray copies = UTIL_DYNARRAY_INIT;

      reg = assign_regs_by_copying(rctx, I->dest[dest_idx], I, &copies,
                                   minimum, limit);

      /* Insert the necessary copies. Phis need special handling since we can't
       * insert instructions before the phi.
       */
      if (I->op == AGX_OPCODE_PHI) {
         agx_emit_moves_before_phi(
            rctx, rctx->block, copies.data,
            util_dynarray_num_elements(&copies, struct agx_copy));
      } else {
         agx_builder b = agx_init_builder(rctx->shader, agx_before_instr(I));
         agx_emit_parallel_copies(
            &b, copies.data,
            util_dynarray_num_elements(&copies, struct agx_copy));
      }

      util_dynarray_fini(&copies);

      /* assign_regs asserts this is cleared, so clear to be reassigned */
      BITSET_CLEAR_COUNT(rctx->used_regs[cls], reg, count);
      return reg;
   }
}

static uint32_t
search_ssa_to_reg_out(struct ra_ctx *ctx, struct agx_block *blk,
                      enum ra_class cls, unsigned ssa)
{
   for (unsigned reg = 0; reg < ctx->bound[cls]; ++reg) {
      if (blk->reg_to_ssa_out[cls][reg] == ssa)
         return reg;
   }

   UNREACHABLE("variable not defined in block");
}

/*
 * Loop over live-in values at the start of the block and mark their registers
 * as in-use. We process blocks in dominance order, so this handles everything
 * but loop headers.
 *
 * For loop headers, this handles the forward edges but not the back edge.
 * However, that's okay: we don't want to reserve the registers that are
 * defined within the loop, because then we'd get a contradiction. Instead we
 * leave them available and then they become fixed points of a sort.
 */
static void
reserve_live_in(struct ra_ctx *rctx)
{
   /* If there are no predecessors, there is nothing live-in */
   unsigned nr_preds = agx_num_predecessors(rctx->block);
   if (nr_preds == 0)
      return;

   agx_builder b =
      agx_init_builder(rctx->shader, agx_before_block(rctx->block));

   U_SPARSE_BITSET_FOREACH_SET(&rctx->block->live_in, i) {
      /* Skip values defined in loops when processing the loop header */
      if (!BITSET_TEST(rctx->visited, i))
         continue;

      unsigned base;
      enum ra_class cls = rctx->classes[i];
      enum agx_size size = rctx->sizes[i];

      /* We need to use the unrounded channel count, since the extra padding
       * will be uninitialized and would fail RA validation.
       */
      unsigned channels = rctx->ncomps_unrounded[i] / agx_size_align_16(size);

      /* If we split live ranges, the variable might be defined differently at
       * the end of each predecessor. Join them together with a phi inserted at
       * the start of the block.
       */
      if (nr_preds > 1) {
         /* We'll fill in the destination after, to coalesce one of the moves */
         agx_instr *phi = agx_phi_to(&b, agx_null(), nr_preds);

         agx_foreach_predecessor(rctx->block, pred) {
            unsigned pred_idx = agx_predecessor_index(rctx->block, *pred);

            phi->src[pred_idx] = agx_get_vec_index(i, size, channels);
            phi->src[pred_idx].memory = cls == RA_MEM;

            if ((*pred)->reg_to_ssa_out[cls] == NULL) {
               /* If this is a loop header, we don't know where the register
                * will end up. So, we create a phi conservatively but don't fill
                * it in until the end of the loop. Stash in the information
                * we'll need to fill in the real register later.
                */
               assert(rctx->block->loop_header);
            } else {
               /* Otherwise, we can build the phi now */
               phi->src[pred_idx].reg =
                  search_ssa_to_reg_out(rctx, *pred, cls, i);
               phi->src[pred_idx].has_reg = true;
            }
         }

         /* Pick the phi destination to coalesce a move. Predecessor ordering is
          * stable, so this means all live-in values get their registers from a
          * particular predecessor. That means that such a register allocation
          * is valid here, because it was valid in the predecessor.
          */
         assert(phi->src[0].has_reg && "not loop source");
         phi->dest[0] = phi->src[0];
         base = phi->dest[0].reg;
      } else {
         /* If we don't emit a phi, there is already a unique register */
         assert(nr_preds == 1);

         agx_block **pred = util_dynarray_begin(&rctx->block->predecessors);
         /* TODO: Flip logic to eliminate the search */
         base = search_ssa_to_reg_out(rctx, *pred, cls, i);
      }

      set_ssa_to_reg(rctx, i, base);

      for (unsigned j = 0; j < rctx->ncomps[i]; ++j) {
         BITSET_SET(rctx->used_regs[cls], base + j);
      }
   }
}

/* The first web of a matrix operand to get a register picks the run of
 * registers for all eight (see struct matrix_group).
 */
static void
place_matrix_group(struct ra_ctx *rctx, unsigned root, unsigned minimum,
                   unsigned limit)
{
   if (!rctx->web_matrix_group || !rctx->web_matrix_group[root] ||
       rctx->phi_web[root].assigned)
      return;

   const struct matrix_group *group =
      &rctx->matrix_groups[rctx->web_matrix_group[root] - 1];
   const unsigned run = 8 * group->size;
   unsigned base;

   if (!find_regs_simple(rctx, RA_GPR, run, run, minimum, limit, &base))
      return;

   for (unsigned k = 0; k < 8; ++k) {
      struct phi_web_node *node = &rctx->phi_web[group->root[k]];
      if (!node->assigned) {
         node->reg = base + k * group->size;
         node->assigned = true;
      }
   }
}

static void
assign_regs(struct ra_ctx *rctx, agx_index v, unsigned reg)
{
   enum ra_class cls = ra_class_for_index(v);
   assert(reg < rctx->bound[cls] && "must not overflow register file");
   assert(v.type == AGX_INDEX_NORMAL && "only SSA gets registers allocated");
   set_ssa_to_reg(rctx, v.value, reg);

   assert(!BITSET_TEST(rctx->visited, v.value) && "SSA violated");
   BITSET_SET(rctx->visited, v.value);

   unsigned nr = rctx->ncomps[v.value];
   assert(nr >= 1);
   assert(!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, nr) &&
          "no interference");

   BITSET_SET_COUNT(rctx->used_regs[cls], reg, nr);

   /* Phi webs need to remember which register they're assigned to. A web of a
    * matrix operand remembers its place in the operand's run instead, if one
    * is free: this value may be elsewhere (a component of a vector load, in
    * place), but the phis it reaches then gather the operand where the
    * instruction wants it.
    */
   const unsigned root = phi_web_find(rctx->phi_web, v.value);
   struct phi_web_node *node = &rctx->phi_web[root];

   if (cls == RA_GPR)
      place_matrix_group(rctx, root, 0, rctx->bound[cls]);

   if (!node->assigned) {
      node->reg = reg;
      node->assigned = true;
   }
}

static void
agx_set_sources(struct ra_ctx *rctx, agx_instr *I)
{
   assert(I->op != AGX_OPCODE_PHI);

   agx_foreach_ssa_src(I, s) {
      assert(BITSET_TEST(rctx->visited, I->src[s].value) && "no phis");

      I->src[s].reg = rctx->ssa_to_reg[I->src[s].value];
      I->src[s].has_reg = true;
   }
}

static void
agx_set_dests(struct ra_ctx *rctx, agx_instr *I)
{
   agx_foreach_ssa_dest(I, s) {
      I->dest[s].reg = rctx->ssa_to_reg[I->dest[s].value];
      I->dest[s].has_reg = true;
   }
}

static unsigned
affinity_base_of_collect(struct ra_ctx *rctx, agx_instr *collect, unsigned src)
{
   unsigned src_reg = rctx->ssa_to_reg[collect->src[src].value];
   unsigned src_offset = src * agx_size_align_16(collect->src[src].size);

   if (src_reg >= src_offset)
      return src_reg - src_offset;
   else
      return ~0;
}

static bool
try_coalesce_with(struct ra_ctx *rctx, agx_index ssa, unsigned count,
                  bool may_be_unvisited, unsigned *out)
{
   assert(ssa.type == AGX_INDEX_NORMAL);
   if (!BITSET_TEST(rctx->visited, ssa.value)) {
      assert(may_be_unvisited);
      return false;
   }

   unsigned base = rctx->ssa_to_reg[ssa.value];
   enum ra_class cls = ra_class_for_index(ssa);

   if (BITSET_TEST_COUNT(rctx->used_regs[cls], base, count))
      return false;

   assert(base + count <= rctx->bound[cls] && "invariant");
   *out = base;
   return true;
}

static bool
pick_collect_affinity(struct ra_ctx *rctx, agx_instr *I, unsigned d,
                      unsigned count, unsigned align, unsigned *out)
{
   agx_index idx = I->dest[d];
   agx_instr *collect = rctx->src_to_collect_phi[idx.value];
   if (!collect || collect->op != AGX_OPCODE_COLLECT)
      return false;
   const enum ra_class cls = ra_class_for_index(idx);
   const unsigned minimum = dest_minimum(rctx, I, d);
   const unsigned limit = dest_limit(rctx, I, d);
   unsigned our_source = ~0u;
   agx_foreach_ssa_src(collect, s) {
      if (agx_is_equiv(collect->src[s], idx)) {
         our_source = s;
         break;
      }
   }
   assert(our_source < collect->nr_srcs);
   agx_foreach_ssa_src(collect, s) {
      if (!BITSET_TEST(rctx->visited, collect->src[s].value))
         continue;
      unsigned base = affinity_base_of_collect(rctx, collect, s);
      if (base >= rctx->bound[cls])
         continue;
      unsigned reg = base + our_source * align;
      if (reg < minimum || reg + count > limit || reg % align)
         continue;
      if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count)) {
         *out = reg;
         return true;
      }
   }
   unsigned collect_align = rctx->ncomps[collect->dest[0].value];
   unsigned offset = our_source * align;
   for (unsigned base = 0; base + collect->nr_srcs * align <= limit;
        base += collect_align) {
      if (base + offset >= minimum &&
          !BITSET_TEST_COUNT(rctx->used_regs[cls], base, collect->nr_srcs * align)) {
         *out = base + offset;
         return true;
      }
   }
   if (collect_align > align) {
      for (unsigned reg = offset; reg + collect_align <= limit;
           reg += collect_align) {
         if (reg >= minimum &&
             !BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count)) {
            *out = reg;
            return true;
         }
      }
   }
   return false;
}

/*
 * A destination with a G17P bank constraint (e.g. a colour tile-store payload,
 * limited to the tile-store bank by agx3_tile_store_bank()) used to skip every
 * affinity heuristic and take the lowest free region. For the ordinary packed
 * RGBA8 payload, `collect(fltnorm, fltnorm)` whose halves already sit in an
 * aligned pair inside the bank, that forced a parallel copy: one extra MOV in
 * front of every LSTCOL, on the fragment's critical tail. Apple's G17P
 * fragments store the FLTNORM pair directly: FLTNORM writes one half of the
 * payload pair in place, then PIXWAIT, then LSTCOL stores the pair, no MOV.
 *
 * Try the ordinary collect/move affinity first, but accept it only when the
 * whole region is free, aligned, and inside [dest_minimum, dest_limit). This
 * never evicts or shuffles anything, so it cannot fail where find_regs()
 * would have succeeded; otherwise fall through to find_regs() unchanged.
 */
static bool
pick_bank_affinity(struct ra_ctx *rctx, agx_instr *I, unsigned d,
                   unsigned count, unsigned align, unsigned *out)
{
   if (!agx_isa_is_agx3(rctx->shader->key->dev.encoding))
      return false;

   const enum ra_class cls = ra_class_for_index(I->dest[d]);
   const unsigned minimum = dest_minimum(rctx, I, d);
   const unsigned limit = dest_limit(rctx, I, d);

   if (I->op == AGX_OPCODE_COLLECT && d == 0) {
      agx_foreach_ssa_src(I, s) {
         if (ra_class_for_index(I->src[s]) != cls ||
             !BITSET_TEST(rctx->visited, I->src[s].value))
            continue;

         const unsigned base = affinity_base_of_collect(rctx, I, s);
         if (base == ~0u || (base % align) || base < minimum ||
             base + count > limit || base + count > rctx->bound[cls])
            continue;

         if (!BITSET_TEST_COUNT(rctx->used_regs[cls], base, count)) {
            *out = base;
            return true;
         }
      }
   }

   if (I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_NORMAL &&
       I->src[0].memory == I->dest[0].memory &&
       I->src[0].size == I->dest[0].size &&
       BITSET_TEST(rctx->visited, I->src[0].value)) {
      const unsigned base = rctx->ssa_to_reg[I->src[0].value];
      if (!(base % align) && base >= minimum && base + count <= limit &&
          base + count <= rctx->bound[cls] &&
          !BITSET_TEST_COUNT(rctx->used_regs[cls], base, count)) {
         *out = base;
         return true;
      }
   }

   return pick_collect_affinity(rctx, I, d, count, align, out);
}

static bool
pick_phi_edge_affinity(struct ra_ctx *rctx, agx_instr *I, unsigned d,
                       unsigned count, unsigned align, unsigned *out)
{
   if (I->op != AGX_OPCODE_PHI ||
       !agx_isa_is_agx3(rctx->shader->key->dev.encoding))
      return false;
   enum ra_class cls = ra_class_for_index(I->dest[d]);
   unsigned first = ~0u;
   agx_foreach_ssa_src(I, s) {
      if (!I->src[s].has_reg || ra_class_for_index(I->src[s]) != cls)
         continue;
      unsigned reg = I->src[s].reg;
      if (reg % align || reg < dest_minimum(rctx, I, d) ||
          reg + count > dest_limit(rctx, I, d) ||
          BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count))
         continue;
      bool private = true;
      agx_foreach_phi_in_block(rctx->block, other) {
         if (other == I)
            continue;
         agx_foreach_ssa_src(other, edge) {
            agx_index src = other->src[edge];
            if (ra_class_for_index(src) != cls)
               continue;
            agx_index own = I->src[edge];
            bool identity = own.type == AGX_INDEX_NORMAL && own.has_reg &&
                            ra_class_for_index(own) == cls && own.reg == reg;
            if (identity)
               continue;
            if (!src.has_reg ||
                (src.reg < reg + count &&
                 reg < src.reg + rctx->ncomps[other->dest[0].value])) {
               private = false;
               break;
            }
         }
         if (!private)
            break;
      }
      if (private)
         first = MIN2(first, reg);
   }
   if (first == ~0u)
      return false;
   *out = first;
   return true;
}

/*
 * A phi whose incoming values all arrive in the same register needs no copy on
 * any edge when it is allocated there as well. The phi web only remembers the
 * register of its first allocated member, which need not be where the incoming
 * values ended up, so this takes precedence over the web. Loop headers do not
 * know their back edge yet and keep following the web.
 */
static bool
pick_phi_source_agreement(struct ra_ctx *rctx, agx_instr *I, unsigned *out)
{
   if (I->op != AGX_OPCODE_PHI ||
       (rctx->shader->key->dev.chip != AGX_CHIP_G17P &&
        !agx_chip_g17_m5(rctx->shader->key->dev.chip)))
      return false;

   enum ra_class cls = ra_class_for_index(I->dest[0]);
   unsigned count = rctx->ncomps[I->dest[0].value];
   unsigned reg = ~0u;

   agx_foreach_ssa_src(I, s) {
      if (!I->src[s].has_reg || ra_class_for_index(I->src[s]) != cls ||
          (reg != ~0u && I->src[s].reg != reg))
         return false;

      reg = I->src[s].reg;
   }

   if (reg == ~0u || (reg % count) || reg < dest_minimum(rctx, I, 0) ||
       reg + count > dest_limit(rctx, I, 0) ||
       BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count))
      return false;

   *out = reg;
   return true;
}

/*
 * The incoming values of a block's phis die on the edge, so their registers
 * are free while the phis are assigned. In a block with many phis, the lowest
 * free register is then usually where the incoming value of a phi assigned
 * later sits, and taking it forces that phi off its own input. When a phi has
 * no affinity left to follow, prefer a register no unassigned phi of the block
 * is receiving.
 */
static bool
pick_phi_unclaimed(struct ra_ctx *rctx, agx_instr *I, unsigned d,
                   unsigned count, unsigned align, unsigned *out)
{
   if (I->op != AGX_OPCODE_PHI ||
       (rctx->shader->key->dev.chip != AGX_CHIP_G17P &&
        !agx_chip_g17_m5(rctx->shader->key->dev.chip)))
      return false;

   enum ra_class cls = ra_class_for_index(I->dest[d]);
   BITSET_DECLARE(claimed, AGX_NUM_MODELED_REGS) = {0};
   bool any = false;

   agx_foreach_phi_in_block(rctx->block, other) {
      if (other == I || other->dest[0].type != AGX_INDEX_NORMAL ||
          other->dest[0].has_reg)
         continue;

      agx_foreach_ssa_src(other, s) {
         agx_index src = other->src[s];
         if (src.has_reg && ra_class_for_index(src) == cls) {
            BITSET_SET_COUNT(claimed, src.reg, rctx->ncomps[src.value]);
            any = true;
         }
      }
   }

   if (!any)
      return false;

   for (unsigned reg = ALIGN_POT(dest_minimum(rctx, I, d), align);
        reg + count <= dest_limit(rctx, I, d); reg += align) {
      if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count) &&
          !BITSET_TEST_COUNT(claimed, reg, count)) {
         *out = reg;
         return true;
      }
   }

   return false;
}

static unsigned
pick_regs(struct ra_ctx *rctx, agx_instr *I, unsigned d)
{
   agx_index idx = I->dest[d];
   enum ra_class cls = ra_class_for_index(idx);
   assert(idx.type == AGX_INDEX_NORMAL);

   unsigned count = rctx->ncomps[idx.value];
   assert(count >= 1);

   unsigned align = count;

   /* Scalar byte loads address the low half of a word. Keep this physical
    * requirement through allocation, including byte-copy helper loops. */
   if (I->op == AGX_OPCODE_DEVICE_LOAD && I->format == AGX_FORMAT_I8 &&
       idx.size == AGX_SIZE_16 && I->mask == 1 &&
       (rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C))
      return find_regs(rctx, I, d, count, MAX2(align, 2));

   /* AGX3b atomicdev is physically two-address: field16 supplies the update
    * value and receives the returned old value.  The legalization pass gives
    * every returning scalar atomic a last-use value temporary.  Its region was
    * freed by early-kill immediately above, so assign the destination there
    * exactly.  If the source is unexpectedly still live, fall through and let
    * the fail-closed packer reject the untied result. */
   if (I->op == AGX_OPCODE_ATOMIC && d == 0 &&
       rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       I->atomic_opc != AGX_ATOMIC_OPC_CMPXCHG &&
       I->src[0].type == AGX_INDEX_NORMAL && !I->src[0].memory &&
       I->src[0].size == idx.size) {
      unsigned out;
      if (try_coalesce_with(rctx, I->src[0], count, false, &out))
         return out;
   }

   if (dest_minimum(rctx, I, d) > 0 ||
       dest_limit(rctx, I, d) < rctx->bound[cls]) {
      unsigned out;
      if (pick_bank_affinity(rctx, I, d, count, align, &out) ||
          pick_phi_edge_affinity(rctx, I, d, count, align, &out) ||
          pick_phi_unclaimed(rctx, I, d, count, align, &out))
         return out;
      return find_regs(rctx, I, d, count, align);
   }

   /* Try to allocate entire phi webs compatibly */
   unsigned phi_idx = phi_web_find(rctx->phi_web, idx.value);

   if (cls == RA_GPR) {
      place_matrix_group(rctx, phi_idx, dest_minimum(rctx, I, d),
                         dest_limit(rctx, I, d));
   }

   if (rctx->phi_web[phi_idx].assigned) {
      unsigned reg = rctx->phi_web[phi_idx].reg;
      if ((reg % align) == 0 && reg + align < rctx->bound[cls] &&
          !BITSET_TEST_COUNT(rctx->used_regs[cls], reg, align))
         return reg;
   }

   /* Try to allocate moves compatibly with their sources */
   if (I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_NORMAL &&
       I->src[0].memory == I->dest[0].memory &&
       I->src[0].size == I->dest[0].size) {

      unsigned out;
      if (try_coalesce_with(rctx, I->src[0], count, false, &out))
         return out;
   }

   /* Try to allocate phis compatibly with their sources */
   if (I->op == AGX_OPCODE_PHI) {
      unsigned edge_reg;
      if (pick_phi_edge_affinity(rctx, I, d, count, align, &edge_reg))
         return edge_reg;
      agx_foreach_ssa_src(I, s) {
         /* Loop headers have phis with a source preceding the definition */
         bool may_be_unvisited = rctx->block->loop_header;

         unsigned out;
         if (try_coalesce_with(rctx, I->src[s], count, may_be_unvisited, &out))
            return out;
      }
   }

   /* Try to allocate collects compatibly with their sources */
   if (I->op == AGX_OPCODE_COLLECT) {
      agx_foreach_ssa_src(I, s) {
         assert(BITSET_TEST(rctx->visited, I->src[s].value) &&
                "registers assigned in an order compatible with dominance "
                "and this is not a phi node, so we have assigned a register");

         unsigned base = affinity_base_of_collect(rctx, I, s);
         if (base >= rctx->bound[cls] || (base + count) > rctx->bound[cls])
            continue;

         /* Unaligned destinations can happen when dest size > src size */
         if (base % align)
            continue;

         if (!BITSET_TEST_COUNT(rctx->used_regs[cls], base, count))
            return base;
      }
   }

   /* Try to coalesce scalar exports */
   agx_instr *collect_phi = rctx->src_to_collect_phi[idx.value];
   if (collect_phi && collect_phi->op == AGX_OPCODE_EXPORT) {
      unsigned reg = collect_phi->imm;

      if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, align) &&
          (reg % align) == 0) {
         unsigned lower;
         if (BITSET_TEST(rctx->early_scalar_export, idx.value) &&
             reg + count > *rctx->count[cls] &&
             find_regs_simple(rctx, cls, count, align, dest_minimum(rctx, I, d),
                              reg, &lower))
            return lower;
         return reg;
      }
   }

   /* Try to coalesce vector exports */
   if (collect_phi && collect_phi->op == AGX_OPCODE_SPLIT) {
      if (collect_phi->dest[0].type == AGX_INDEX_NORMAL) {
         agx_instr *exp = rctx->src_to_collect_phi[collect_phi->dest[0].value];
         if (exp && exp->op == AGX_OPCODE_EXPORT) {
            unsigned reg = exp->imm;

            if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, align) &&
                (reg % align) == 0)
               return reg;
         }
      }
   }

   unsigned collect_reg;
   if (pick_collect_affinity(rctx, I, d, count, align, &collect_reg))
      return collect_reg;

   /* Try to allocate phi sources compatibly with their phis */
   if (collect_phi && collect_phi->op == AGX_OPCODE_PHI) {
      agx_instr *phi = collect_phi;
      unsigned out;

      agx_foreach_ssa_src(phi, s) {
         if (try_coalesce_with(rctx, phi->src[s], count, true, &out))
            return out;
      }

      /* If we're in a loop, we may have already allocated the phi. Try that. */
      if (phi->dest[0].has_reg) {
         unsigned base = phi->dest[0].reg;

         if (base + count <= rctx->bound[cls] &&
             !BITSET_TEST_COUNT(rctx->used_regs[cls], base, count))
            return base;
      }
   }

   unsigned unclaimed;
   if (pick_phi_unclaimed(rctx, I, d, count, align, &unclaimed))
      return unclaimed;

   /* Default to any contiguous sequence of registers */
   return find_regs(rctx, I, d, count, align);
}

static void
kill_source(struct ra_ctx *rctx, const agx_instr *I, unsigned s)
{
   enum ra_class cls = ra_class_for_index(I->src[s]);
   unsigned reg = rctx->ssa_to_reg[I->src[s].value];
   unsigned count = rctx->ncomps[I->src[s].value];

   assert(I->op != AGX_OPCODE_PHI && "phis don't use .kill");
   assert(count >= 1);

   BITSET_CLEAR_COUNT(rctx->used_regs[cls], reg, count);
}

static void
try_kill_early_sources(struct ra_ctx *rctx, const agx_instr *I,
                       unsigned first_source, unsigned last_source,
                       unsigned region_end, unsigned region_base)
{
   unsigned dest_size = util_next_power_of_two(rctx->ncomps[I->dest[0].value]);
   unsigned dest_end = region_base + dest_size;

   /* We can only early-kill a region if we can trivially allocate the
    * destination to it. That way we never shuffle killed sources.
    *
    * To ensure that, the region must be aligned and cover the destination.
    */
   if (region_base == region_end || region_base < dest_minimum(rctx, I, 0) ||
       dest_end > dest_limit(rctx, I, 0) ||
       (rctx->ssa_to_reg[I->src[first_source].value] & (dest_size - 1)) ||
       ((region_end < dest_end) &&
        BITSET_TEST_RANGE(rctx->used_regs[RA_GPR], region_end, dest_end)))
      return;

   for (unsigned s = first_source; s <= last_source; ++s) {
      if (I->src[s].kill && !I->src[s].memory) {
         kill_source(rctx, I, s);
         rctx->early_killed = true;
         I->src[s].kill = false;
      }
   }
}

static bool
transient_texture_program(const agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble || ctx->any_cf ||
       ctx->scratch_size_B || list_length(&ctx->blocks) != 1)
      return false;
   bool terminal_tile = false;
   agx_foreach_instr_global(ctx, I) {
      const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
      if (info->can_reorder && info->can_eliminate &&
          info->schedule_class == AGX_SCHEDULE_CLASS_NONE)
         continue;
      switch (I->op) {
      case AGX_OPCODE_PRELOAD:
      case AGX_OPCODE_UNIT_TEST:
      case AGX_OPCODE_EXPORT:
      case AGX_OPCODE_INTERP_AGX3:
      case AGX_OPCODE_RHW_INTERP_AGX3:
      case AGX_OPCODE_GET_SR:
      case AGX_OPCODE_PUBLISH_TEXTURE_AGX3:
      case AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3:
         break;
      case AGX_OPCODE_ST_TILE:
      case AGX_OPCODE_SIGNAL_PIX:
         terminal_tile = true;
         break;
      case AGX_OPCODE_WAIT_PIX:
      case AGX_OPCODE_STOP:
         break;
      case AGX_OPCODE_TEXTURE_SAMPLE:
         if (terminal_tile || I->sparse || I->shadow || I->offset || I->query_lod ||
             I->gather != AGX_GATHER_NONE ||
             (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_CUBE) ||
             I->lod_mode != AGX_LOD_MODE_AUTO_LOD)
            return false;
         break;
      default:
         return false;
      }
   }
   return true;
}

static bool
texture_region_retired(struct ra_ctx *rctx, agx_instr *I, unsigned reg,
                       unsigned count)
{
   agx_foreach_instr_in_block(rctx->block, producer) {
      if (producer == I)
         break;
      if (!agx3_is_async(producer))
         continue;
      agx_foreach_ssa_dest(producer, d) {
         agx_index value = producer->dest[d];
         if (value.memory || !value.has_reg ||
             value.reg >= reg + count ||
             reg >= value.reg + rctx->ncomps[value.value])
            continue;
         BITSET_WORD *alias = BITSET_CALLOC(rctx->shader->alloc);
         if (!alias)
            return false;
         BITSET_SET(alias, value.value);
         bool retired = false;
         for (struct list_head *node = producer->link.next; node != &I->link;
              node = node->next) {
            agx_instr *use = list_entry(node, agx_instr, link);
            bool reads = false;
            agx_foreach_ssa_src(use, q)
               reads |= BITSET_TEST(alias, use->src[q].value);
            if (!reads)
               continue;
            bool wiring = use->op == AGX_OPCODE_SPLIT ||
                          use->op == AGX_OPCODE_COLLECT ||
                          use->op == AGX_OPCODE_PHI;
            if (use->op == AGX_OPCODE_MOV && use->nr_dests == 1 &&
                use->nr_srcs == 1 && use->dest[0].has_reg &&
                use->src[0].has_reg && use->dest[0].reg == use->src[0].reg &&
                use->dest[0].size == use->src[0].size &&
                !use->src[0].abs && !use->src[0].neg)
               wiring = true;
            if (use->op == AGX_OPCODE_EXPORT && use->src[0].has_reg &&
                use->src[0].reg == use->imm)
               wiring = true;
            if (wiring) {
               agx_foreach_ssa_dest(use, q)
                  BITSET_SET(alias, use->dest[q].value);
            } else {
               retired = true;
               break;
            }
         }
         free(alias);
         if (!retired)
            return false;
      }
   }
   return true;
}

static bool
texture_tail_cost(struct ra_ctx *rctx, agx_instr *I, agx_instr *split,
                  unsigned reg, unsigned *peak, unsigned *copies)
{
   struct ra_ctx model = *rctx;
   BITSET_DECLARE(used, AGX_NUM_REGS);
   memcpy(used, rctx->used_regs[RA_GPR], sizeof(used));
   model.used_regs[RA_GPR] = used;
   model.ssa_to_reg = mem_dup(rctx->ssa_to_reg,
                             rctx->shader->alloc * sizeof(uint16_t));
   model.visited = mem_dup(rctx->visited,
                           BITSET_BYTES(rctx->shader->alloc));
   model.phi_web = mem_dup(rctx->phi_web,
                           rctx->shader->alloc * sizeof(struct phi_web_node));
   unsigned footprint = *rctx->count[RA_GPR], moves = 0;
   model.count[RA_GPR] = &footprint;
   model.transient_texture_used = true;
   bool ok = false;
   if (!model.ssa_to_reg || !model.visited || !model.phi_web)
      goto done;
   assign_regs(&model, I->dest[0], reg);
   agx_foreach_ssa_src(I, q) {
      if (I->src[q].kill && !I->src[q].memory)
         BITSET_CLEAR_COUNT(used, model.ssa_to_reg[I->src[q].value],
                            model.ncomps[I->src[q].value]);
   }
   for (struct list_head *node = &split->link;
        node != &rctx->block->instructions; node = node->next) {
      agx_instr *next = list_entry(node, agx_instr, link);
      agx_instr proxy = *next;
      agx_instr *proxy_ptr = &proxy;
      agx_index sources[256], destinations[256];
      memcpy(sources, next->src, next->nr_srcs * sizeof(agx_index));
      memcpy(destinations, next->dest, next->nr_dests * sizeof(agx_index));
      proxy.src = sources;
      proxy.dest = destinations;
      model.instr = &proxy;
      agx_foreach_ssa_src(proxy_ptr, q) {
         if (proxy.src[q].memory ||
             !BITSET_TEST(model.visited, proxy.src[q].value))
            goto done;
      }
      if (proxy.op == AGX_OPCODE_SPLIT && proxy.src[0].kill) {
         unsigned base = model.ssa_to_reg[proxy.src[0].value];
         unsigned width = agx_size_align_16(agx_split_width(&proxy));
         BITSET_CLEAR_COUNT(used, base, model.ncomps[proxy.src[0].value]);
         agx_foreach_ssa_dest(proxy_ptr, q)
            assign_regs(&model, proxy.dest[q], base + q * width);
         continue;
      }
      if (proxy.op == AGX_OPCODE_PRELOAD || proxy.op == AGX_OPCODE_PHI ||
          proxy.agx3_cube_scalar_coords || proxy.agx3_split_coords ||
          proxy.op == AGX_OPCODE_FFS ||
          proxy.op == AGX_OPCODE_BFEIL || proxy.op == AGX_OPCODE_BITOP ||
          proxy.op == AGX_OPCODE_IADD)
         goto done;
      unsigned source_regs[256];
      agx_foreach_ssa_src(proxy_ptr, q)
         source_regs[q] = model.ssa_to_reg[proxy.src[q].value];
      model.early_killed = false;
      if (proxy.nr_dests == 1 && proxy.dest[0].type == AGX_INDEX_NORMAL) {
         unsigned first_source = 0, end = 0, base = 0;
         agx_foreach_ssa_src(proxy_ptr, q) {
            if (!proxy.src[q].kill)
               continue;
            unsigned current = source_regs[q];
            if (base == end || current != end) {
               try_kill_early_sources(&model, &proxy, first_source, q, end, base);
               first_source = q;
               base = current;
            }
            end = current + model.ncomps[proxy.src[q].value];
         }
         if (proxy.nr_srcs)
            try_kill_early_sources(&model, &proxy, first_source,
                                   proxy.nr_srcs - 1, end, base);
      }
      agx_foreach_ssa_dest(proxy_ptr, q) {
         unsigned count = model.ncomps[proxy.dest[q].value], available;
         if (proxy.dest[q].memory ||
             !find_regs_simple(&model, RA_GPR, count, count,
                               dest_minimum(&model, &proxy, q),
                               dest_limit(&model, &proxy, q), &available))
            goto done;
         unsigned chosen = pick_regs(&model, &proxy, q);
         assign_regs(&model, proxy.dest[q], chosen);
      }
      if (proxy.op == AGX_OPCODE_COLLECT) {
         unsigned base = model.ssa_to_reg[proxy.dest[0].value];
         unsigned width = agx_size_align_16(proxy.dest[0].size);
         agx_foreach_src(proxy_ptr, q) {
            if (proxy.src[q].type == AGX_INDEX_UNDEF || agx_is_null(proxy.src[q]))
               continue;
            if (proxy.src[q].type != AGX_INDEX_NORMAL)
               goto done;
            moves += source_regs[q] != base + q * width;
         }
      } else if (proxy.op == AGX_OPCODE_SPLIT) {
         unsigned base = source_regs[0];
         unsigned width = agx_size_align_16(agx_split_width(&proxy));
         agx_foreach_ssa_dest(proxy_ptr, q)
            moves += model.ssa_to_reg[proxy.dest[q].value] != base + q * width;
      } else if (proxy.op == AGX_OPCODE_EXPORT) {
         moves += proxy.src[0].type != AGX_INDEX_NORMAL ||
                  source_regs[0] != proxy.imm;
         footprint = MAX2(footprint, proxy.imm +
                         agx_size_align_16(proxy.src[0].size));
      } else if (proxy.op == AGX_OPCODE_MOV) {
         if (proxy.nr_dests != 1 || proxy.dest[0].type != AGX_INDEX_NORMAL)
            goto done;
         moves += proxy.src[0].type != AGX_INDEX_NORMAL ||
                  proxy.src[0].size != proxy.dest[0].size ||
                  model.ssa_to_reg[proxy.dest[0].value] != source_regs[0];
      }
      agx_foreach_ssa_src(proxy_ptr, q) {
         if (proxy.src[q].kill)
            BITSET_CLEAR_COUNT(used, source_regs[q],
                               model.ncomps[proxy.src[q].value]);
      }
   }
   ok = true;
   *peak = footprint;
   *copies = moves;
done:
   free(model.ssa_to_reg);
   free(model.visited);
   free(model.phi_web);
   return ok;
}

static bool
pick_transient_texture_region(struct ra_ctx *rctx, agx_instr *I, unsigned d,
                              unsigned count, unsigned standard, unsigned *out)
{
   if (rctx->transient_texture_used ||
       I->op != AGX_OPCODE_TEXTURE_SAMPLE || count != 8 || d ||
       I->dest[0].size != AGX_SIZE_32 ||
       I->link.next == &rctx->block->instructions)
      return false;
   agx_instr *split = list_entry(I->link.next, agx_instr, link);
   if (split->op != AGX_OPCODE_SPLIT || split->nr_srcs != 1 ||
       split->nr_dests != 4 || !split->src[0].kill ||
       !agx_is_equiv(split->src[0], I->dest[0]) ||
       agx_split_width(split) != AGX_SIZE_32)
      return false;
   agx_foreach_ssa_dest(split, q) {
      if (split->dest[q].memory || rctx->ncomps[split->dest[q].value] != 2)
         return false;
   }
   unsigned remaining = 0;
   agx_foreach_ssa_dest(split, q)
      remaining |= 1u << q;
   for (struct list_head *node = split->link.next;
        remaining && node != &rctx->block->instructions; node = node->next) {
      agx_instr *next = list_entry(node, agx_instr, link);
      agx_foreach_ssa_dest(next, q) {
         if (!next->dest[q].memory &&
             rctx->ncomps[next->dest[q].value] > 2)
            return false;
      }
      agx_foreach_ssa_src(next, q) {
         if (!next->src[q].kill)
            continue;
         agx_foreach_ssa_dest(split, c) {
            if (agx_is_equiv(next->src[q], split->dest[c]))
               remaining &= ~(1u << c);
         }
      }
   }
   if (remaining)
      return false;
   unsigned reg;
   if (!find_regs_simple(rctx, RA_GPR, count, 2, dest_minimum(rctx, I, d),
                         dest_limit(rctx, I, d), &reg) || reg >= standard ||
       !transient_texture_program(rctx->shader) ||
       !texture_region_retired(rctx, I, reg, count))
      return false;
   for (unsigned base = ROUND_DOWN_TO(reg, count); base < reg + count;
        base += count) {
      if (!BITSET_TEST_COUNT(rctx->used_regs[RA_GPR], base, count)) {
         return false;
      }
   }
   struct ra_ctx model = *rctx;
   BITSET_DECLARE(used, AGX_NUM_REGS);
   memcpy(used, rctx->used_regs[RA_GPR], sizeof(used));
   model.used_regs[RA_GPR] = used;
   model.ssa_to_reg = mem_dup(rctx->ssa_to_reg,
                             rctx->shader->alloc * sizeof(uint16_t));
   model.visited = mem_dup(rctx->visited,
                           BITSET_BYTES(rctx->shader->alloc));
   model.phi_web = mem_dup(rctx->phi_web,
                           rctx->shader->alloc * sizeof(struct phi_web_node));
   unsigned footprint = *rctx->count[RA_GPR];
   model.count[RA_GPR] = &footprint;
   bool ok = false;
   if (!model.ssa_to_reg || !model.visited || !model.phi_web)
      goto done;
   assign_regs(&model, I->dest[0], reg);
   agx_foreach_ssa_src(I, q) {
      if (I->src[q].kill && !I->src[q].memory)
         BITSET_CLEAR_COUNT(used, model.ssa_to_reg[I->src[q].value],
                            model.ncomps[I->src[q].value]);
   }
   BITSET_CLEAR_COUNT(used, reg, count);
   agx_foreach_ssa_dest(split, q)
      assign_regs(&model, split->dest[q], reg + 2 * q);
   struct list_head *node = split->link.next;
   for (unsigned step = 0; step < 8 && node != &rctx->block->instructions;
        ++step, node = node->next) {
      agx_instr *next = list_entry(node, agx_instr, link);
      if ((next->op != AGX_OPCODE_FADD && next->op != AGX_OPCODE_FMUL &&
           next->op != AGX_OPCODE_FFMA) || next->nr_dests != 1 ||
          next->dest[0].type != AGX_INDEX_NORMAL || next->dest[0].memory ||
          model.ncomps[next->dest[0].value] != 2)
         break;
      bool reads = false, simple = true;
      agx_foreach_ssa_src(next, q) {
         simple &= !next->src[q].memory &&
            model.ncomps[next->src[q].value] == 2;
         agx_foreach_ssa_dest(split, c)
            reads |= agx_is_equiv(next->src[q], split->dest[c]);
      }
      if (!simple)
         break;
      model.instr = next;
      agx_foreach_ssa_src(next, q) {
         if (next->src[q].kill)
            BITSET_CLEAR_COUNT(used, model.ssa_to_reg[next->src[q].value], 2);
      }
      if (reads) {
         agx_instr *export =
            model.src_to_collect_phi[next->dest[0].value];
         if (export && export->op == AGX_OPCODE_EXPORT &&
             export->imm < reg + count && reg < export->imm + 2 &&
             !(standard < export->imm + 2 &&
               export->imm < standard + count) &&
             BITSET_TEST_COUNT(used, export->imm, 2))
            break;
         ok = true;
         break;
      }
      unsigned available;
      if (!find_regs_simple(&model, RA_GPR, 2, 2,
                            dest_minimum(&model, next, 0),
                            dest_limit(&model, next, 0), &available))
         break;
      unsigned chosen = pick_regs(&model, next, 0);
      if (chosen < reg + count && reg < chosen + 2)
         break;
      assign_regs(&model, next->dest[0], chosen);
   }
done:
   free(model.ssa_to_reg);
   free(model.visited);
   free(model.phi_web);
   if (ok) {
      unsigned candidate_peak, baseline_peak, candidate_copies, baseline_copies;
      ok = standard != ~0u &&
         texture_tail_cost(rctx, I, split, reg, &candidate_peak,
                           &candidate_copies) &&
         texture_tail_cost(rctx, I, split, standard, &baseline_peak,
                           &baseline_copies) &&
         candidate_peak <= baseline_peak && candidate_copies <= baseline_copies;
   }
   if (ok) {
      *out = reg;
   }
   return ok;
}

/** Assign registers to SSA values in a block. */

static void
agx_ra_assign_local(struct ra_ctx *rctx)
{
   BITSET_DECLARE(used_regs_gpr, AGX_NUM_REGS) = {0};
   BITSET_DECLARE(used_regs_mem, AGX_NUM_MODELED_REGS) = {0};
   uint16_t *ssa_to_reg = calloc(rctx->shader->alloc, sizeof(uint16_t));

   agx_block *block = rctx->block;
   rctx->used_regs[RA_GPR] = used_regs_gpr;
   rctx->used_regs[RA_MEM] = used_regs_mem;
   rctx->ssa_to_reg = ssa_to_reg;

   reserve_live_in(rctx);

   /* Force the nesting counter r0l live throughout shaders using control flow.
    * This could be optimized (sync with agx_calc_register_demand).
    */
   if (rctx->shader->any_cf)
      BITSET_SET(used_regs_gpr, 0);

   /* Keep the divergent-shuffle scratch live throughout the shader. AGX2
    * uses r0h; AGX3 reserves the full r1 after r0's nesting counter. */
   if (rctx->shader->any_quad_divergent_shuffle) {
      assert(rctx->shader->any_cf);
      if (rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
         BITSET_SET(used_regs_gpr, 1);
      else
         BITSET_SET_COUNT(used_regs_gpr, 0, 4);
   }

   /* Match the CEM prefix counted by demand, spilling and shuffling. */
   if (rctx->shader->agx3_cem_low_scratch)
      BITSET_SET_COUNT(used_regs_gpr, 0, reserved_size(rctx->shader));

   /* Reserve bottom registers as temporaries for parallel copy lowering */
   if (rctx->shader->has_spill_pcopy_reserved) {
      BITSET_SET_COUNT(used_regs_gpr, 0, 8);
   }

   /* Assign phis whose incoming values all arrive in one free register first,
    * so that no phi before them in the block can take that register. The loop
    * below skips them.
    */
   agx_foreach_phi_in_block(block, phi) {
      unsigned reg;
      if (phi->dest[0].type == AGX_INDEX_NORMAL && !phi->dest[0].has_reg &&
          pick_phi_source_agreement(rctx, phi, &reg)) {
         assign_regs(rctx, phi->dest[0], reg);
         agx_set_dests(rctx, phi);
      }
   }

   agx_foreach_instr_in_block(block, I) {
      rctx->instr = I;

      /* Optimization: if a split contains the last use of a vector, the split
       * can be removed by assigning the destinations overlapping the source.
       */
      /* A killed vector may live above a sampler operand's register bank.
       * A channel used by that sampler must take the regular parallel-copy
       * path if it cannot inherit the vector's physical register. */
      bool coalesce_split = I->op == AGX_OPCODE_SPLIT && I->src[0].kill;
      if (coalesce_split &&
          rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
         const unsigned base = ssa_to_reg[I->src[0].value];
         const unsigned width = agx_size_align_16(agx_split_width(I));
         agx_foreach_ssa_dest(I, d) {
            const unsigned reg = base + d * width;
            if (reg < dest_minimum(rctx, I, d) ||
                reg + width > dest_limit(rctx, I, d)) {
               coalesce_split = false;
               break;
            }
         }
      }
      if (coalesce_split) {
         assert(ra_class_for_index(I->src[0]) == RA_GPR);
         unsigned reg = ssa_to_reg[I->src[0].value];
         unsigned width = agx_size_align_16(agx_split_width(I));

         agx_foreach_dest(I, d) {
            assert(ra_class_for_index(I->dest[0]) == RA_GPR);

            /* Free up the source */
            unsigned offset_reg = reg + (d * width);
            BITSET_CLEAR_COUNT(used_regs_gpr, offset_reg, width);

            /* Assign the destination where the source was */
            if (!agx_is_null(I->dest[d]))
               assign_regs(rctx, I->dest[d], offset_reg);
         }

         unsigned trail = rctx->ncomps[I->src[0].value] - (I->nr_dests * width);
         BITSET_CLEAR_COUNT(used_regs_gpr, reg + (I->nr_dests * width), trail);

         agx_set_sources(rctx, I);
         agx_set_dests(rctx, I);
         continue;
      } else if (I->op == AGX_OPCODE_PRELOAD) {
         /* We must coalesce all preload moves */
         assert(I->dest[0].size == I->src[0].size);
         assert(I->src[0].type == AGX_INDEX_REGISTER);

         /* r1l specifically is a preloaded register. It is reserved during
          * demand calculations to ensure we don't need live range shuffling of
          * spilling temporaries. But we can still preload to it. So if it's
          * reserved, just free it. It'll be fine.
          */
         if (I->src[0].value == 2) {
            BITSET_CLEAR(rctx->used_regs[RA_GPR], 2);
         }

         assign_regs(rctx, I->dest[0], I->src[0].value);
         agx_set_dests(rctx, I);
         continue;
      }

      if (I->agx3_cube_scalar_coords && I->nr_dests == 1 &&
          I->dest[0].type == AGX_INDEX_NORMAL) {
         unsigned released[3], nr_released = 0;
         for (unsigned c = 0; c < 3; ++c) {
            unsigned src = c ? c + 5 : 0;
            if (I->src[src].type != AGX_INDEX_NORMAL || !I->src[src].kill)
               continue;
            bool duplicate = false;
            for (unsigned n = 0; n < nr_released; ++n)
               duplicate |= agx_is_equiv(I->src[src], I->src[released[n]]);
            if (duplicate)
               continue;
            kill_source(rctx, I, src);
            released[nr_released++] = src;
         }
         unsigned count = rctx->ncomps[I->dest[0].value], reg;
         bool found =
            nr_released && find_regs_simple(rctx, RA_GPR, count, count,
                                            dest_minimum(rctx, I, 0),
                                            dest_limit(rctx, I, 0), &reg);
         if (found) {
            reg = pick_regs(rctx, I, 0);
            assign_regs(rctx, I->dest[0], reg);
            agx_foreach_ssa_src(I, src) {
               if (!I->src[src].kill)
                  continue;
               bool already = false;
               for (unsigned n = 0; n < nr_released; ++n)
                  already |= agx_is_equiv(I->src[src], I->src[released[n]]);
               if (!already)
                  kill_source(rctx, I, src);
            }
            agx_set_sources(rctx, I);
            agx_set_dests(rctx, I);
            continue;
         }
         for (unsigned n = 0; n < nr_released; ++n) {
            unsigned src = released[n];
            BITSET_SET_COUNT(used_regs_gpr, ssa_to_reg[I->src[src].value],
                             rctx->ncomps[I->src[src].value]);
         }
      }

      /* AGX3b device atomics return the old value in-place over their value
       * operand.  When that operand dies here, assign the destination to its
       * exact base before the generic allocator can choose a different free
       * region.  This also handles cmpxchg: its two-channel
       * [replacement, compare] source dies as a unit while the scalar return
       * occupies the first channel.  Scalar returned atomics were given a
       * private value above, so their source is guaranteed to die here.
       */
      if (agx_isa_is_agx3(rctx->shader->key->dev.encoding) &&
          I->op == AGX_OPCODE_ATOMIC && I->nr_dests == 1 &&
          !agx_is_null(I->dest[0]) && I->src[0].type == AGX_INDEX_NORMAL &&
          I->src[0].kill) {
         const unsigned reg = ssa_to_reg[I->src[0].value];

         kill_source(rctx, I, 0);
         assign_regs(rctx, I->dest[0], reg);

         for (unsigned s = 1; s < I->nr_srcs; ++s) {
            if (I->src[s].type == AGX_INDEX_NORMAL && I->src[s].kill)
               kill_source(rctx, I, s);
         }

         agx_set_sources(rctx, I);
         agx_set_dests(rctx, I);
         continue;
      }

      if (I->op == AGX_OPCODE_MATMAD16_AGX3) {
         if (I->nr_dests != 1 || I->nr_srcs != 3 ||
             I->src[2].type != AGX_INDEX_NORMAL || I->src[2].memory ||
             !I->src[2].kill || rctx->ncomps[I->src[2].value] != 16 ||
             rctx->ncomps[I->dest[0].value] != 16) {
            rctx->shader->ra_failed = true;
            return;
         }
         const unsigned reg = ssa_to_reg[I->src[2].value];
         kill_source(rctx, I, 2);
         assign_regs(rctx, I->dest[0], reg);
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_NORMAL && I->src[s].kill)
               kill_source(rctx, I, s);
         }
         agx_set_sources(rctx, I);
         agx_set_dests(rctx, I);
         continue;
      }

      /* Search for regions of contiguous killed sources to early-kill. */
      rctx->early_killed = false;

      /* G17P FT1's completed native form requires distinct source and
       * destination registers. Keep its last-use source occupied until after
       * destination allocation so generic early-kill cannot coalesce them. */
      const bool g17p_ft1 =
         agx_isa_is_agx3(rctx->shader->key->dev.encoding) &&
         I->op == AGX_OPCODE_FFS;

      /* A G17P register-boundary SRL may overwrite its value when that value
       * dies, as in Apple's native tied witness, but it must not overwrite
       * the boundary operand itself. Keep a killed boundary allocated until
       * the destination has been placed. */
      const bool g17p_srl_boundary =
         agx_isa_is_agx3(rctx->shader->key->dev.encoding) &&
         I->op == AGX_OPCODE_BFEIL && I->nr_srcs == 3 &&
         I->src[2].type == AGX_INDEX_NORMAL && I->src[2].kill;

      /* Apple's G17P register-subtract witnesses and both exact passing SWAR
       * shaders keep the destination distinct from their inputs. The compact
       * tied shape has not been independently qualified, so allocate the
       * result before freeing either source for the register-register shape
       * the packer maps to ISUB.
       *
       * AGX3B only. Native G17G emits isub32 both tied and with a distinct
       * destination (port/cwit4 k_sub_*), and widening this rule to AGX3C
       * made tessellation.invariance.{inner,outer}_triangle_set.*_fractional_
       * even_spacing fail on M5 (Pass with the rule off; bisected to this
       * hunk on hardware). Which allocation in the 48 changed precompiled
       * kernels or the runtime shaders causes it is not yet known. */
      const bool g17p_reg_isub =
         rctx->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         I->op == AGX_OPCODE_IADD && I->nr_srcs == 2 && I->shift == 0 &&
         I->src[0].type == AGX_INDEX_NORMAL &&
         I->src[1].type == AGX_INDEX_NORMAL && !I->src[0].abs &&
         !I->src[0].neg && !I->src[1].abs && I->src[1].neg;

      /* The same destructive restriction applies to G17P's register AND.
       * The failing compact form reused src1 (the mask) for the destination;
       * both the fully staged shader (distinct from both) and the keepalive
       * discriminator (tied only to src0) are exact on hardware. */
      const bool g17p_reg_and =
         agx_isa_is_agx3(rctx->shader->key->dev.encoding) &&
         I->op == AGX_OPCODE_BITOP && I->nr_srcs == 2 &&
         (I->truth_table & 0xfu) == 0x8u &&
         I->src[0].type == AGX_INDEX_NORMAL &&
         I->src[1].type == AGX_INDEX_NORMAL;

      if (I->nr_dests == 1 && !agx_is_null(I->dest[0]) &&
          !g17p_ft1 && !g17p_srl_boundary &&
          !g17p_reg_isub && !g17p_reg_and) {
         unsigned first_src = 0;
         unsigned end = 0;
         unsigned start = 0;

         agx_foreach_ssa_src(I, s) {
            if (I->src[s].kill && !I->src[s].memory) {
               unsigned reg = rctx->ssa_to_reg[I->src[s].value];

               if (start == end || end != reg) {
                  try_kill_early_sources(rctx, I, first_src, s, end, start);
                  first_src = s;
                  start = reg;
               }

               end = reg + rctx->ncomps[I->src[s].value];
            }
         }

         try_kill_early_sources(rctx, I, first_src, I->nr_srcs - 1, end, start);
      }

      /* Next, assign destinations one at a time. This is always legal
       * because of the SSA form.
       */
      agx_foreach_ssa_dest(I, d) {
         if (I->op == AGX_OPCODE_PHI && I->dest[d].has_reg)
            continue;

         assign_regs(rctx, I->dest[d], pick_regs(rctx, I, d));
      }

      /* The M5 query entry writes its result register even when nothing reads
       * it. It still needs a real register (so nothing live is clobbered),
       * which is free again immediately after the instruction.
       */
      if (I->op == AGX_OPCODE_ENTER_QUERY_M5 && I->nr_dests == 1 &&
          !BITSET_TEST(rctx->dest_read, I->dest[0].value)) {
         enum ra_class cls = ra_class_for_index(I->dest[0]);
         BITSET_CLEAR_COUNT(rctx->used_regs[cls],
                            rctx->ssa_to_reg[I->dest[0].value],
                            rctx->ncomps[I->dest[0].value]);
      }

      /* Free late-killed sources */
      agx_foreach_ssa_src(I, s) {
         if (I->src[s].kill) {
            kill_source(rctx, I, s);
         }
      }

      /* Phi sources are special. Set in the corresponding predecessors */
      if (I->op != AGX_OPCODE_PHI)
         agx_set_sources(rctx, I);

      agx_set_dests(rctx, I);
   }

   for (unsigned i = 0; i < RA_CLASSES; ++i) {
      block->reg_to_ssa_out[i] =
         malloc(rctx->bound[i] * sizeof(*block->reg_to_ssa_out[i]));

      /* Initialize with sentinel so we don't have unused regs mapping to r0 */
      memset(block->reg_to_ssa_out[i], 0xFF,
             rctx->bound[i] * sizeof(*block->reg_to_ssa_out[i]));
   }

   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, i) {
      block->reg_to_ssa_out[rctx->classes[i]][rctx->ssa_to_reg[i]] = i;
   }

   /* Also set the sources for the phis in our successors, since that logically
    * happens now (given the possibility of live range splits, etc)
    */
   agx_foreach_successor(block, succ) {
      unsigned pred_idx = agx_predecessor_index(succ, block);

      agx_foreach_phi_in_block(succ, phi) {
         if (phi->src[pred_idx].type == AGX_INDEX_NORMAL &&
             !phi->src[pred_idx].has_reg) {
            /* This source needs a fixup */
            unsigned value = phi->src[pred_idx].value;
            phi->src[pred_idx].reg = rctx->ssa_to_reg[value];
            phi->src[pred_idx].has_reg = true;
         }
      }
   }

   free(rctx->ssa_to_reg);
}

/*
 * Lower phis to parallel copies at the logical end of a given block. If a block
 * needs parallel copies inserted, a successor of the block has a phi node. To
 * have a (nontrivial) phi node, a block must have multiple predecessors. So the
 * edge from the block to the successor (with phi) is not the only edge entering
 * the successor. Because the control flow graph has no critical edges, this
 * edge must therefore be the only edge leaving the block, so the block must
 * have only a single successor.
 */
static void
agx_insert_parallel_copies(agx_context *ctx, agx_block *block)
{
   bool any_succ = false;
   unsigned nr_phi = 0;

   /* Phi nodes logically happen on the control flow edge, so parallel copies
    * are added at the end of the predecessor */
   agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));

   agx_foreach_successor(block, succ) {
      assert(nr_phi == 0 && "control flow graph has a critical edge");

      agx_foreach_phi_in_block(succ, phi) {
         assert(!any_succ && "control flow graph has a critical edge");
         nr_phi += agx_channels(phi->dest[0]);
      }

      any_succ = true;

      /* Nothing to do if there are no phi nodes */
      if (nr_phi == 0)
         continue;

      unsigned pred_index = agx_predecessor_index(succ, block);

      /* Create a parallel copy lowering all the phi nodes */
      struct agx_copy *copies = calloc(sizeof(*copies), nr_phi);

      unsigned i = 0;

      agx_foreach_phi_in_block(succ, phi) {
         agx_index dest = phi->dest[0];
         agx_index src = phi->src[pred_index];

         if (src.type == AGX_INDEX_IMMEDIATE)
            src.size = dest.size;

         assert(dest.type == AGX_INDEX_REGISTER);
         assert(dest.size == src.size);

         /* Scalarize the phi, since the parallel copy lowering doesn't handle
          * vector phis. While we scalarize phis in NIR, we can generate vector
          * phis from spilling so must take care.
          */
         for (unsigned c = 0; c < agx_channels(phi->dest[0]); ++c) {
            agx_index src_ = src;
            unsigned offs = c * agx_size_align_16(src.size);

            if (src.type != AGX_INDEX_IMMEDIATE) {
               assert(src.type == AGX_INDEX_UNIFORM ||
                      src.type == AGX_INDEX_REGISTER);
               src_.value += offs;
               src_.channels_m1 = 1 - 1;
            }

            assert(i < nr_phi);
            copies[i++] = (struct agx_copy){
               .dest = dest.value + offs,
               .dest_mem = dest.memory,
               .src = src_,
            };
         }
      }

      /* A register pair holding no value live out of this block and touched
       * by no copy is free on this edge: the successor's live-ins are exactly
       * the phi destinations plus this block's live-outs.  Stay inside the
       * allocated file so the register count is unchanged. */
      agx_index temp = agx_null();
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          block->reg_to_ssa_out[RA_GPR]) {
         const uint32_t *out = block->reg_to_ssa_out[RA_GPR];
         const unsigned top = ctx->max_reg + 1;
         for (unsigned reg = ALIGN_POT(reserved_size(ctx), 2);
              reg + 2 <= top; reg += 2) {
            if (out[reg] != ~0u || out[reg + 1] != ~0u)
               continue;

            bool touched = false;
            for (unsigned j = 0; j < nr_phi && !touched; ++j) {
               const unsigned w = agx_size_align_16(copies[j].src.size);
               if (!copies[j].dest_mem && copies[j].dest < reg + 2 &&
                   copies[j].dest + w > reg)
                  touched = true;
               if (copies[j].src.type == AGX_INDEX_REGISTER &&
                   !copies[j].src.memory && copies[j].src.value < reg + 2 &&
                   copies[j].src.value + w > reg)
                  touched = true;
            }
            if (touched)
               continue;

            temp = agx_register(reg, AGX_SIZE_32);
            break;
         }
      }

      agx_emit_parallel_copies_temp(&b, copies, nr_phi, temp);

      free(copies);
   }
}

static void
lower_exports(agx_context *ctx)
{
   struct agx_copy copies[AGX_NUM_REGS];
   unsigned nr = 0;
   agx_block *block = agx_exit_block(ctx);

   agx_foreach_instr_in_block_safe(block, I) {
      if (I->op != AGX_OPCODE_EXPORT)
         continue;

      assert(agx_channels(I->src[0]) == 1 && "scalarized in frontend");
      assert(nr < ARRAY_SIZE(copies));

      copies[nr++] = (struct agx_copy){
         .dest = I->imm,
         .src = I->src[0],
      };

      /* The export itself is now trivial, reflect that for correct last-use
       * tracking later.
       */
      I->src[0] = agx_register_like(I->imm, I->src[0]);

      /* We cannot use fewer registers than we export */
      ctx->max_reg =
         MAX2(ctx->max_reg, I->imm + agx_size_align_16(I->src[0].size));
   }

   agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));
   agx_emit_parallel_copies(&b, copies, nr);
}

/* AGX3b device atomics return the old value by overwriting the value operand.
 * Give every scalar returning atomic a private sacrificial value before
 * liveness is computed.  This is required when a source-level value (commonly
 * a CSE'd constant) remains live after the atomic.  cmpxchg already arrives as
 * a private two-channel collect built specifically for the instruction.
 */
static void
agx_lower_agx3b_atomic_returns(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_ATOMIC || I->nr_dests != 1 ||
          agx_is_null(I->dest[0]) || I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_replace_src(I, 0, agx_mov(&b, I->src[0]));
   }
}

/* The GPR halves a G17P fragment's tile stores read their payloads from. */
unsigned
agx3_tile_store_bank(agx_context *ctx)
{
   /* Native raw32 stores need simultaneously live payloads. Their
    * exact ordinary/BG witnesses use GPR halves0..31 with f13=0. This expands
    * only the admitted two-to-four RGBA32 or four R32 payload bank. */
   unsigned quad_stores = 0, quad_width = 0, quad_mask = 0;
   BITSET_DECLARE(tile_offsets, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   /* Halves of payload the stores need live together, counted once per colour
    * target: a target stored for several samples reuses one register vector. */
   unsigned payload_halves = 0, payload_align = 1;
   bool tile_loads = false;
   bool quad_tile =
      agx_isa_is_agx3(ctx->key->dev.encoding) &&
      ctx->stage == MESA_SHADER_FRAGMENT &&
      (ctx->key->secondary || ctx->key->fs.ignore_tib_dependencies) &&
      list_length(&ctx->blocks) == 1;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_ST_TILE) {
         if (I->pixel_offset < AGX_G17P_MAX_BYTES_PER_SAMPLE) {
            if (!BITSET_TEST(tile_offsets, I->pixel_offset) &&
                I->src[0].type == AGX_INDEX_NORMAL) {
               unsigned halves = agx_index_size_16(I->src[0]);
               payload_halves += halves;
               payload_align = MAX2(payload_align, halves);
            }
            BITSET_SET(tile_offsets, I->pixel_offset);
         }
         unsigned width = I->mask == 15 ? 16 : I->mask == 1 ? 4 : 0;
         quad_tile &= width && (!quad_width || quad_width == width) &&
            I->format == AGX_FORMAT_I32 && !I->explicit_coords &&
            I->nr_srcs == 3 && I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == 0xff && agx_is_null(I->src[2]);
         quad_width = width;
         if (!width || I->pixel_offset % width || I->pixel_offset / width >= 4 ||
             (quad_mask & BITFIELD_BIT(I->pixel_offset / width))) {
            quad_tile = false;
         } else {
            quad_mask |= BITFIELD_BIT(I->pixel_offset / width);
         }
         quad_stores++;
      } else if (I->op == AGX_OPCODE_LD_TILE) {
         quad_tile = false;
         tile_loads = true;
      }
   }
   const bool raw32_bank = quad_tile && quad_stores >= 2 && quad_stores <= 4 &&
      (quad_width == 16 || quad_stores == 4) &&
      quad_mask == BITFIELD_MASK(quad_stores);
   /* Apple's store-only fragments of three or more targets and multi-target
    * BG builtins of other formats store every width from the same bank:
    * halves 0..31 for up to four targets and 0..63 for more. Its blending
    * fragments of three or more targets do too: three RGBA32Float targets
    * store from halves0..23, eight RGBA16Float targets from halves0..31, and
    * five from0..19. Per-sample native witnesses
    * use that bank even with fewer targets: id_blend_4x stores RGBA8 from
    * half28, and half2_sample4 stores SINT16/F16 from halves28..0. */
   const unsigned targets = BITSET_COUNT(tile_offsets);

   /* Native G17P conditional image-write plus two-target MRT uses half24 for
    * its F16 tile payload, and its memory epilog owns the high-bank form, so
    * the ordered eMRT path stores from halves 0..31.
    *
    * That bank does not hold four 128-bit payloads.  Each is eight halves and
    * naturally aligned, and the reserved prefix sits inside the first slot, so
    * halves 0..7 cannot carry one: the bank's real capacity is three, and RA
    * aborted with `no constrained region for count=8 minimum=0 limit=32` on
    * dEQP-VK.renderpasses.*.multisample_resolve.r32g32b32a32_uint.
    * max_attachments_8_samples_2, whose eMRT epilog stores four RGBA32 targets
    * at two samples -- halves 8..31 each held an unevictable payload and the
    * only free region began at the reserved half.
    *
    * So size the bank by what the stores actually need, prefix included, and
    * widen to the same halves 0..63 the plain MRT bank below already uses past
    * four targets rather than inventing a third bank. */
   if (ctx->key->g17p_emrt && ctx->agx3_mrt_tile_ops_ordered &&
       list_length(&ctx->blocks) > 1) {
      const unsigned prefix = ALIGN_POT(reserved_size(ctx), payload_align);
      return prefix + payload_halves > 32 ? 64 : 32;
   }

   const bool mrt_bank =
      agx_isa_is_agx3(ctx->key->dev.encoding) &&
      ctx->stage == MESA_SHADER_FRAGMENT &&
      (!tile_loads || targets >= 3) &&
      targets >= (ctx->key->g17p_mrt_bg ? 2 : 3) &&
      targets <= AGX_MAX_RTS &&
      (ctx->key->secondary || ctx->key->g17p_mrt_bg);
   const bool sample_bank =
      agx_isa_is_agx3(ctx->key->dev.encoding) &&
      ctx->stage == MESA_SHADER_FRAGMENT && ctx->key->secondary &&
      ctx->agx3_mrt_tile_ops_ordered && ctx->key->fs.nr_samples > 1 &&
      targets >= 2;
   return mrt_bank && targets > 4 ? 64
          : raw32_bank || mrt_bank || sample_bank ? 32 : 16;
}

/* The packer places cube CEM scratch above every allocated operand.
 * Reserve that storage before spilling/RA, including the extra packed layer
 * word used by bias+minimum. Cube-array face staging and the register LOD
 * operand of a plain cube sample both have seven-bit source fields.
 * Keep the allocation budget aligned to the largest SSA vector, so rounding
 * demand for live-range splitting cannot exceed the reduced file. */
static unsigned
agx3_cube_register_budget(agx_context *ctx)
{
   unsigned budget = AGX3B_NUM_REGS;
   unsigned alignment = 8;
   bool shared_cem = false, packed_layer = false;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         alignment = MAX2(alignment,
                         util_next_power_of_two(agx_index_size_16(I->dest[d])));
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || I->nr_srcs < 6 ||
          (I->dim != AGX_DIM_CUBE && I->dim != AGX_DIM_CUBE_ARRAY))
         continue;

      /* A plain cube sample stages a register level or bias with CEM's face
       * in a form that encodes the face's high bit, so only the level needs
       * a seven-bit register, and RA keeps it there (texture_gpr). A cube
       * array stages its face in a seven-bit field, from its own CEM scratch
       * (agx3_reserve_cube_scratch) where it has one. The bias with
       * minimum also places a packed word above the allocation. */
      const bool bias_min = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
      /* With its own scratch, nothing of a cube sample sits above the
       * allocation but a cube array's packed layer word. Its budget keeps
       * the shader at 576 threads: Counter-Strike 2's register-heavy
       * cube-array material passes ran faster at 177 halves than at 185
       * (512 threads), though they spill a little more. */
      const bool own_scratch = !I->agx3_cube_scalar_coords && I->nr_srcs == 7;
      budget = MIN2(budget,
                    ((I->dim == AGX_DIM_CUBE_ARRAY && !own_scratch) || bias_min)
                       ? 112 : own_scratch ? AGX3B_NUM_REGS - 16
                                           : AGX3B_NUM_REGS - 8);
      shared_cem |= !own_scratch;
      packed_layer |= own_scratch && I->src[1].type != AGX_INDEX_IMMEDIATE &&
                      (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                       I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS);
   }
   /* The shared CEM scratch and a cube array's prepacked layer word both sit
    * above the allocation, below half 191. */
   if (shared_cem && packed_layer)
      budget = MIN2(budget, AGX3B_NUM_REGS - 16);
   return ROUND_DOWN_TO(budget, alignment);
}

static bool
pair_peer_publication_consumed_early(agx_context *ctx, unsigned value)
{
   if (list_length(&ctx->blocks) != 1)
      return false;
   bool found = false;
   agx_block *block = list_first_entry(&ctx->blocks, agx_block, link);
   agx_foreach_instr_in_block(block, P) {
      bool reads = false;
      agx_foreach_ssa_src(P, q)
         reads |= P->src[q].value == value;
      const struct agx_opcode_info *info = &agx_opcodes_info[P->op];
      if (!reads || (info->can_reorder && info->can_eliminate &&
                     info->schedule_class == AGX_SCHEDULE_CLASS_NONE))
         continue;
      if (!agx_is_texture_publication(P))
         return false;
      bool consumed = false;
      for (struct list_head *node = P->link.next; node != &block->instructions;
           node = node->next) {
         agx_instr *sample = list_entry(node, agx_instr, link);
         if (agx_texture_resource(sample) != P->imm || sample->nr_dests != 1 ||
             sample->dest[0].type != AGX_INDEX_NORMAL)
            continue;
         BITSET_WORD *result = BITSET_CALLOC(ctx->alloc);
         if (!result)
            return false;
         BITSET_SET(result, sample->dest[0].value);
         for (struct list_head *next_node = sample->link.next;
              next_node != &block->instructions; next_node = next_node->next) {
            agx_instr *I = list_entry(next_node, agx_instr, link);
            if (I->op == AGX_OPCODE_TEXTURE_SAMPLE)
               break;
            bool uses = false;
            agx_foreach_ssa_src(I, q)
               uses |= BITSET_TEST(result, I->src[q].value);
            if (!uses)
               continue;
            if (I->op == AGX_OPCODE_SPLIT || I->op == AGX_OPCODE_COLLECT) {
               agx_foreach_ssa_dest(I, d)
                  BITSET_SET(result, I->dest[d].value);
               continue;
            }
            const struct agx_opcode_info *use_info = &agx_opcodes_info[I->op];
            consumed = use_info->can_reorder && use_info->can_eliminate &&
                       use_info->schedule_class == AGX_SCHEDULE_CLASS_NONE;
            break;
         }
         free(result);
         break;
      }
      if (!consumed)
         return false;
      found = true;
   }
   return found;
}

/* Vectors need whole aligned slots for live range splitting, so the usable
 * register file is a multiple of the largest vector, at least 8 halves.
 */
static unsigned
ra_file_alignment(agx_context *ctx)
{
   unsigned alignment = 8;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         alignment = MAX2(alignment,
                          util_next_power_of_two(agx_index_size_16(I->dest[d])));
   }
   return alignment;
}

/* AGX2 declares a register count with every program, and the hardware splits
 * its register file statically: how many threads fit follows from that count
 * (agx_occupancy_for_register_count). The AGX3B and AGX3C encodings declare no
 * register count, in the program or its launch; their on-chip storage is
 * allocated as threads use it. Spending registers there costs no fixed
 * occupancy step, so the budget is the register file alone. AGX3D programs
 * declare their register count per dispatch again (registers_16 in the G18
 * dispatch and render resources) and keep the split.
 */
static bool
agx_has_static_register_split(const agx_context *ctx)
{
   return !agx_isa_is_agx3(ctx->key->dev.encoding) || ctx->key->dev.emit_agx3d;
}

static void
agx_ra_impl(agx_context *ctx, bool disable_remat)
{
   ctx->uniforms_written = agx_has_uniform_store(ctx);
   agx_lower_agx3b_atomic_returns(ctx);

   /* CEM has narrower destination fields than the M5 general register file.
    * Own its four words (magnitude/face and two coordinates) in the low bank,
    * reusing an axis word for packed array/LOD after the finalizers.
    * Do not cap unrelated live values at 192:
    * they remain eligible for all 252 usable halves. The bank starts beyond
    * the eight halves used by CF, shuffle and spill parallel-copy lowering.
    * Register allocation and the rematerializing spiller both reserve it. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY))
            ctx->agx3_cem_low_scratch = true;
      }
   }

   bool force_spilling =
      (agx_compiler_debug & AGX_DBG_SPILL) && ctx->key->has_scratch;

   /* Determine maximum possible registers. We won't exceed this! */
   unsigned max_possible_regs = AGX_NUM_REGS;

   /* Compute shaders need to have their entire workgroup together, so with a
    * static register split our register usage is bounded by the workgroup
    * size. Without one, a 1024-thread workgroup using the whole file runs
    * correctly, with its barriers, and faster than one spilling to fit an
    * occupancy step.
    */
   if (mesa_shader_stage_is_compute(ctx->stage) &&
       agx_has_static_register_split(ctx)) {
      unsigned threads_per_workgroup;

      /* If we don't know the workgroup size, worst case it. TODO: Optimize
       * this, since it'll decimate opencl perf.
       */
      if (ctx->nir->info.workgroup_size_variable) {
         threads_per_workgroup = 1024;
      } else {
         threads_per_workgroup = ctx->nir->info.workgroup_size[0] *
                                 ctx->nir->info.workgroup_size[1] *
                                 ctx->nir->info.workgroup_size[2];
      }

      max_possible_regs =
         agx_max_registers_for_occupancy(threads_per_workgroup);
   }

   /* G17P's register file is 192 halves. Halves 192..255 still fit its
    * 8-bit register fields but are not general registers: a copy into half
    * 204 crashed the GPU co-processor. applegpu_g17p holding 60 live float4
    * or uint4 temporaries never names a register above half 191 and spills
    * the rest to thread-private storage, so bound the file there and spill
    * the same way.
    */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
      max_possible_regs = MIN2(max_possible_regs, AGX3B_NUM_REGS);

   /* G18G's eight-bit operand fields also encode r126/r127, but those names
    * do not behave as ordinary data registers. Physical scalar readbacks
    * from r124/r125 succeed while stores from r126/r127 leave their output
    * untouched; a vec4 spanning them loses the entire store. Native G18G
    * pressure kernels reach r125 and spill the remaining live values.
    * Keep the unavailable halves out of allocation, including spill temps.
    * G18G uses the G17G allocator before the final AGX3D transcode.
    */
   if (ctx->key->dev.emit_agx3d) {
      /* Round before deciding whether to spill. Rounding only at assignment
       * could leave demand above the aligned file after the spill pass. */
      max_possible_regs = MIN2(max_possible_regs,
                               ROUND_DOWN_TO(AGX3D_NUM_REGS,
                                             ra_file_alignment(ctx)));
   }

   /* Native G17G pressure shaders spill at 126 dwords. Direct M5 tests of
    * IADD/EXTR followed by STDEV agree: pair 250 works, pair 252 does not.
    * Align the usable file before spilling as live-range splitting requires
    * whole vector slots; rounding only at final assignment is too late.
    */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      max_possible_regs = MIN2(max_possible_regs,
                               ROUND_DOWN_TO(AGX3C_NUM_REGS,
                                             ra_file_alignment(ctx)));
   }

   /* Large native-query kernels benefit from leaving registers available for
    * more resident work, even when this introduces ordinary scratch spills.
    * Ordinary compute, helpers, preambles and AGX3B retain their existing
    * budgets.
    */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       ctx->stage == MESA_SHADER_COMPUTE && !ctx->is_preamble &&
       !ctx->key->is_helper && ctx->out->uses_ray_traversal) {
      max_possible_regs = MIN2(max_possible_regs, AGX_M5_RT_REGISTER_BUDGET);
   }

   /* Match the CEM scratch extent reserved by agx3_pack_binary. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
      max_possible_regs = MIN2(max_possible_regs,
                               agx3_cube_register_budget(ctx));

   /* A multisample-array read packs its normalized layer beside another
    * operand, and the packer reserves one transient word for it immediately
    * above every register this allocation used -- the same "reserve above the
    * allocation" shape as the CEM scratch, and the same failure: a fragment
    * that reaches half 190 leaves nowhere to put the word and agx3_pack_binary
    * refuses the whole shader. Measured on
    * dEQP-VK.rasterization.rasterization_order_attachment_access.format_float.
    * attachments_8_samples_4.*, whose eight input attachments at four samples
    * allocate the entire file: observed=191 max_reg=191 base=192.
    *
    * Budget the word inside the file. ROUND_DOWN_TO(reg_file_alignment) below
    * only lowers this further, so 184 leaves the word at half 186 at the very
    * highest under either 8- or 16-half alignment. A fragment of this shape
    * that used halves 184..191 now spills them instead; one that did not, and
    * every fragment without such a read, keeps the whole file.
    */
   if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
       ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble) {
      agx_foreach_instr_global(ctx, I) {
         if ((I->op == AGX_OPCODE_TEXTURE_LOAD ||
              I->op == AGX_OPCODE_IMAGE_LOAD) &&
             I->dim == AGX_DIM_2D_MS_ARRAY) {
            max_possible_regs = MIN2(max_possible_regs, 184);
            break;
         }
      }
   }

   if (force_spilling) {
      /* Even when testing spilling, we need enough room for preloaded/exported
       * regs.
       */
      unsigned d = 24;
      unsigned max_ncomps = 8;

      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_PRELOAD) {
            unsigned size = agx_size_align_16(I->src[0].size);
            d = MAX2(d, I->src[0].value + size);
         } else if (I->op == AGX_OPCODE_EXPORT) {
            unsigned size = agx_size_align_16(I->src[0].size);
            d = MAX2(d, I->imm + size);
         } else if (I->op == AGX_OPCODE_IMAGE_WRITE) {
            /* vec4 source + vec4 coordinates + bindless handle + reserved */
            d = MAX2(d, 26);
         } else if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
                    (I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
                     I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN)) {
            /* as above but with big gradient */
            d = MAX2(d, 36);
         }

         agx_foreach_ssa_dest(I, v) {
            max_ncomps = MAX2(max_ncomps, agx_index_size_16(I->dest[v]));
         }
      }

      max_possible_regs = ALIGN_POT(d, util_next_power_of_two(max_ncomps));
   } else if (ctx->key->is_helper) {
      /* The helper program is unspillable and has a limited register file */
      max_possible_regs = 32;
   }

   /* A retry after a failed allocation spills down to a smaller file, leaving
    * the rest free for live-range splitting around bank-constrained values. */
   const unsigned uncut_regs = max_possible_regs;
   const bool budget_cut =
      ctx->ra_budget_cut && ctx->key->has_scratch && !ctx->key->is_helper;
   if (budget_cut)
      max_possible_regs =
         ROUND_DOWN_TO(max_possible_regs - MIN2(ctx->ra_budget_cut,
                                                max_possible_regs / 2), 16);

   /* Calculate the demand. We'll use it to determine if we need to spill and to
    * bound register assignment.
    */
   agx_compute_liveness(ctx);
   unsigned effective_demand = agx_calc_register_demand(ctx, false);
   bool spilling = (effective_demand > max_possible_regs);
   bool remat = false;
   const unsigned full_budget = max_possible_regs;

   /* If we need multiple waves, see if we can rematerialize constants to save
    * waves. If we only have a single wave regardless, this is pointless.
    * Without a static register split there is no wave to save.
    */
   if (!disable_remat && agx_has_static_register_split(ctx) &&
       effective_demand > agx_round_registers(1) && !spilling) {
      unsigned effective_demand_remat = agx_calc_register_demand(ctx, true);

      /* Worst-case assume we need 6 16-bit registers for constants, for a
       * four-source cmpsel where 3 sources are 32-bit constants. Rounded to
       * ensure live-range splitting works. The remat-only spiller also keeps
       * RA's reserved low bank free (four halves on AGX3, two before), so the
       * slack must grow with it or the remat-only pass cannot meet its budget
       * (ssbo.phys.layout.random.8bit.nested_structs_arrays.20 asserted
       * "must have remat something").
       */
      const unsigned remat_slack =
         4 + MAX2(2, agx_reserved_registers(ctx));
      if ((effective_demand_remat + remat_slack) < effective_demand) {
         /* Assignment rounds the file down to the vector alignment, so the
          * target must already be aligned. Aligning to 16 regardless never
          * reached the 104-half (1024-thread) or 136-half occupancy steps.
          */
         unsigned alignment = ra_file_alignment(ctx);
         unsigned l = ROUND_DOWN_TO(
            agx_round_registers(
               ALIGN_POT(effective_demand_remat + remat_slack, alignment)),
            alignment);

         /* Only rematerialize if it actually lets us save a wave */
         if (agx_round_registers(l) <
             agx_round_registers(ALIGN_POT(effective_demand, alignment))) {
            remat = true;
            max_possible_regs = MIN2(l, max_possible_regs);
         }
      }
   }

   const bool ssa_renumbered = spilling || remat;
   if (spilling || remat) {
      assert((remat || ctx->key->has_scratch) &&
             "internal shaders are unspillable");

      agx_spill(ctx, max_possible_regs, remat);

      if (agx_is_g18g(&ctx->key->dev)) {
         /* Rematerialization changes instruction lifetimes. Measure the
          * actual result before deciding whether the optional wave-saving
          * budget held.
          */
         agx_compute_liveness(ctx);
         effective_demand = agx_calc_register_demand(ctx, false);

         if (remat && (ctx->remat_fallback_spilled ||
                       effective_demand > max_possible_regs)) {
            /* The reduced budget is only an occupancy optimization.
             * Constants reintroduced at their uses can exceed that estimate
             * even without a scratch spill. Retain the rematerialized
             * program and allocate under the original hardware budget,
             * spilling only if needed.
             */
            max_possible_regs = full_budget;
            remat = false;
            spilling = ctx->remat_fallback_spilled ||
                       effective_demand > max_possible_regs;
            if (effective_demand > max_possible_regs) {
               agx_spill(ctx, max_possible_regs, false);
               agx_compute_liveness(ctx);
               effective_demand = agx_calc_register_demand(ctx, false);
            }
         }
      } else {
         if (remat && ctx->remat_fallback_spilled) {
            /* The wave-saving remat-only pass could not meet its estimated
             * budget and spilled to scratch; finish under the full budget. */
            max_possible_regs = full_budget;
            remat = false;
            spilling = true;
            agx_compute_liveness(ctx);
            if (agx_calc_register_demand(ctx, false) > max_possible_regs)
               agx_spill(ctx, max_possible_regs, false);
         }

         /* After spilling, recalculate liveness and demand */
         agx_compute_liveness(ctx);
         effective_demand = agx_calc_register_demand(ctx, false);
      }

      /* The resulting program can now be assigned registers */
      assert(effective_demand <= max_possible_regs && "spiller post-condition");
   }

   /* The cut only lowers the spill target; assignment gets the whole file. */
   if (budget_cut)
      max_possible_regs = uncut_regs;

   unsigned original_hints =
      util_dynarray_num_elements(&ctx->collect_split_hints, agx_instr *);
   if (ssa_renumbered || ctx->scratch_size_B) {
      agx_foreach_instr_global(ctx, I) {
         /* Split sample coordinates take no pair hint (see
          * agx3_split_texture_coordinates()). */
         if ((!I->agx3_cube_scalar_coords && !I->agx3_split_coords) ||
             I->nr_srcs != 8 ||
             (I->agx3_split_coords && I->op != AGX_OPCODE_TEXTURE_LOAD))
            continue;
         if (!ctx->collect_split_hints.data)
            util_dynarray_init(&ctx->collect_split_hints, ctx);
         const unsigned n = agx_is_null(I->src[7]) ? 2 : 3;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_instr *hint = agx_alloc_instr(&b, AGX_OPCODE_COLLECT, 1, n);
         hint->dest[0] = agx_vec_temp(ctx, AGX_SIZE_32, n);
         hint->src[0] = I->src[0];
         hint->src[1] = I->src[6];
         if (n == 3)
            hint->src[2] = I->src[7];
         util_dynarray_append(&ctx->collect_split_hints, hint);
      }
   }

   /* Record all phi webs. First initialize the union-find data structure with
    * all SSA defs in their own singletons, then union together anything related
    * by a phi. The resulting union-find structure will be the webs.
    */
   struct phi_web_node *phi_web = calloc(ctx->alloc, sizeof(*phi_web));
   for (unsigned i = 0; i < ctx->alloc; ++i) {
      phi_web[i].parent = i;
   }

   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi) {
         agx_foreach_ssa_src(phi, s) {
            phi_web_union(phi_web, phi->dest[0].value, phi->src[s].value);
         }
      }
   }

   /* Group the phi webs that make up the operands of matrix multiplies */
   struct util_dynarray matrix_groups;
   util_dynarray_init(&matrix_groups, NULL);
   uint16_t *web_matrix_group = NULL;
   bool any_matrix = false;
   agx_foreach_instr_global(ctx, I)
      any_matrix |= I->op == AGX_OPCODE_MATMAD16_AGX3;

   if (any_matrix) {
      agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
      web_matrix_group = calloc(ctx->alloc, sizeof(*web_matrix_group));

      agx_foreach_instr_global(ctx, I) {
         agx_foreach_ssa_dest(I, d)
            defs[I->dest[d].value] = I;
      }

      agx_foreach_instr_global(ctx, I) {
         if (I->op != AGX_OPCODE_MATMAD16_AGX3)
            continue;

         agx_foreach_ssa_src(I, s) {
            /* The accumulator is a copy (agx_lower_agx3_matrix_sources) */
            agx_instr *def = defs[I->src[s].value];
            if (def && def->op == AGX_OPCODE_MOV &&
                def->src[0].type == AGX_INDEX_NORMAL)
               def = defs[def->src[0].value];

            if (!def || def->op != AGX_OPCODE_COLLECT || def->nr_srcs != 8)
               continue;

            struct matrix_group group = {0};
            bool valid = true;
            for (unsigned k = 0; k < 8 && valid; ++k) {
               const agx_index src = def->src[k];
               valid = src.type == AGX_INDEX_NORMAL && !src.memory &&
                       agx_channels(src) == 1 &&
                       src.size == def->src[0].size &&
                       (src.size == AGX_SIZE_16 || src.size == AGX_SIZE_32);
               if (!valid)
                  break;

               group.root[k] = phi_web_find(phi_web, src.value);
               valid = !web_matrix_group[group.root[k]];
               for (unsigned j = 0; j < k; ++j)
                  valid &= group.root[j] != group.root[k];
            }
            if (!valid)
               continue;

            group.size = agx_size_align_16(def->src[0].size);
            util_dynarray_append(&matrix_groups, group);
            const unsigned id = util_dynarray_num_elements(
               &matrix_groups, struct matrix_group);
            if (id > UINT16_MAX) {
               (void)util_dynarray_pop(&matrix_groups, struct matrix_group);
               continue;
            }
            for (unsigned k = 0; k < 8; ++k)
               web_matrix_group[group.root[k]] = id;
         }
      }

      free(defs);
   }

   uint8_t *ncomps = calloc(ctx->alloc, sizeof(uint8_t));
   uint8_t *ncomps_unrounded = calloc(ctx->alloc, sizeof(uint8_t));
   enum ra_class *classes = calloc(ctx->alloc, sizeof(enum ra_class));
   agx_instr **src_to_collect_phi = calloc(ctx->alloc, sizeof(agx_instr *));
   enum agx_size *sizes = calloc(ctx->alloc, sizeof(enum agx_size));
   BITSET_WORD *visited = BITSET_CALLOC(ctx->alloc);
   BITSET_WORD *low_gpr = BITSET_CALLOC(ctx->alloc);
   unsigned *split_source = calloc(ctx->alloc, sizeof(unsigned));
   BITSET_WORD *texture_gpr = BITSET_CALLOC(ctx->alloc);
   BITSET_WORD *log2_gpr = BITSET_CALLOC(ctx->alloc);
   BITSET_WORD *high_gpr = BITSET_CALLOC(ctx->alloc);
   BITSET_WORD *tex_low128 = BITSET_CALLOC(ctx->alloc);
   unsigned max_ncomps = 1;
   unsigned *definition_order = calloc(ctx->alloc, sizeof(unsigned));
   BITSET_WORD *early_scalar_export = BITSET_CALLOC(ctx->alloc);
   BITSET_WORD *effectful_use = BITSET_CALLOC(ctx->alloc);
   unsigned instruction_order = 0;

   const unsigned tile_gpr_limit = agx3_tile_store_bank(ctx);

   agx_foreach_instr_global(ctx, I) {
      instruction_order++;
      const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
      if (!info->can_reorder || !info->can_eliminate ||
          info->schedule_class != AGX_SCHEDULE_CLASS_NONE) {
         agx_foreach_ssa_src(I, q)
            BITSET_SET(effectful_use, I->src[q].value);
      }
      /* The G17P funop encoder and fused quadrant selects use the bounded
       * seven-bit operand forms. Constrain only their SSA webs; unrelated
       * values retain the complete register file. */
      if ((ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (I->op == AGX_OPCODE_SINC_AGX3 ||
           I->op == AGX_OPCODE_BITTEST_SELECT_AGX3 ||
           (I->op == AGX_OPCODE_TRUNC && I->nr_dests == 1 &&
            I->dest[0].size == AGX_SIZE_32) ||
           I->op == AGX_OPCODE_DFDX || I->op == AGX_OPCODE_DFDY)) ||
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
           I->op == AGX_OPCODE_SINC_AGX3)) {
         agx_foreach_ssa_src(I, s) {
            BITSET_SET(texture_gpr, I->src[s].value);
            /* A split channel takes its vector's register. */
            if (split_source[I->src[s].value])
               BITSET_SET(texture_gpr, split_source[I->src[s].value] - 1);
         }
         agx_foreach_ssa_dest(I, d)
            BITSET_SET(texture_gpr, I->dest[d].value);
      }

      /* A split channel takes its vector's register, so a tile store of one
       * channel constrains the whole vector: a per-sample reload of a scalar
       * target stores channel 0 of each texel fetch. */
      if (I->op == AGX_OPCODE_SPLIT && I->src[0].type == AGX_INDEX_NORMAL) {
         agx_foreach_ssa_dest(I, d)
            split_source[I->dest[d].value] = I->src[0].value + 1;
      }

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_ST_TILE && I->src[0].type == AGX_INDEX_NORMAL) {
         BITSET_SET(low_gpr, I->src[0].value);
         if (split_source[I->src[0].value])
            BITSET_SET(low_gpr, split_source[I->src[0].value] - 1);
      }

      /* Explicit publications own the coordinate GPR constraints even when
       * the following sample no longer exposes those operands to RA.
       */
      if ((ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
          agx_is_texture_publication(I)) {
         /* Scalar 3D publication can use the full M5 source register field.
          * Keep every other publication within the original low-bank range.
          */
         bool wide_scalar = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
            ctx->stage == MESA_SHADER_FRAGMENT &&
            I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3 &&
            I->nr_srcs == 3 && !agx_is_null(I->src[2]);
         for (unsigned s = 0; s < I->nr_srcs && wide_scalar; ++s)
            wide_scalar &= !agx_is_null(I->src[s]) &&
                           I->src[s].size == AGX_SIZE_32;
         if (!wide_scalar) {
            agx_foreach_ssa_src(I, s)
               BITSET_SET(texture_gpr, I->src[s].value);
         }
      }

      /* G18G has its own direct-coordinate bound. Mark the SSA web so a
       * low TEX destination cannot relocate its still-live coordinates
       * into an unencodable high bank. Other operands keep their limits. */
      if (agx_is_g18g(&ctx->key->dev) &&
          (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
           I->op == AGX_OPCODE_TEXTURE_LOAD ||
           I->op == AGX_OPCODE_IMAGE_LOAD) &&
          I->src[0].type == AGX_INDEX_NORMAL)
         BITSET_SET(texture_gpr, I->src[0].value);

      /* Other sampled coordinates and CEM retain their low-bank source
       * constraints. Gradient scalars and shadow references also remain
       * within the existing qualified range. */
      if ((ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
          (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
           I->op == AGX_OPCODE_TEXTURE_LOAD ||
           I->op == AGX_OPCODE_IMAGE_LOAD)) {
         /* Split M5 fetch and sample coordinates stage x and y through the
          * wide scalar COPY, which reads the whole register file; the sample
          * index or layer keeps PACK_SMPIR's seven-bit source. */
         if (I->src[0].type == AGX_INDEX_NORMAL && !I->agx3_split_coords)
            BITSET_SET(texture_gpr, I->src[0].value);
         if (I->agx3_cube_scalar_coords) {
            for (unsigned s = 6; s < 8; ++s) {
               if (I->src[s].type == AGX_INDEX_NORMAL)
                  BITSET_SET(texture_gpr, I->src[s].value);
            }
         }
         if (I->agx3_split_coords && I->src[7].type == AGX_INDEX_NORMAL)
            BITSET_SET(texture_gpr, I->src[7].value);

         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
              I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN ||
              I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN) &&
             I->src[1].type == AGX_INDEX_NORMAL)
            BITSET_SET(texture_gpr, I->src[1].value);

         /* M5 PACK_SMPIR level/bias operands retain a seven-bit source,
          * even though SMP can return into the complete register file. */
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
             I->src[1].type == AGX_INDEX_NORMAL)
            BITSET_SET(texture_gpr, I->src[1].value);

         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->shadow &&
             I->src[5].type == AGX_INDEX_NORMAL)
            BITSET_SET(texture_gpr, I->src[5].value);

         /* A cube sample's register level or bias is a seven-bit operand of
          * the staging PACK_SMPIR (agx3_tex_stage_cube_lod). */
         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             (I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY) &&
             I->src[1].type == AGX_INDEX_NORMAL)
            BITSET_SET(texture_gpr, I->src[1].value);

         /* A cube array's own CEM scratch holds its face, a seven-bit operand
          * of the staging PACK_SMPIR (agx3_tex_stage_cube_array_layer). */
         if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
             I->dim == AGX_DIM_CUBE_ARRAY && I->nr_srcs == 7 &&
             I->src[6].type == AGX_INDEX_NORMAL)
            BITSET_SET(texture_gpr, I->src[6].value);
      }

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3 &&
          I->src[0].type == AGX_INDEX_NORMAL)
         BITSET_SET(texture_gpr, I->src[0].value);

      /* imgwci stages its data and coordinates through the same seven-bit
       * COPY sources and names its PBE index register in a seven-bit field.
       */
      if ((ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
          I->op == AGX_OPCODE_IMAGE_WRITE) {
         for (unsigned s = 0; s < I->nr_srcs; ++s) {
            if ((s == 0 || s == 1 || s == 4) &&
                I->src[s].type == AGX_INDEX_NORMAL)
               BITSET_SET(texture_gpr, I->src[s].value);
         }
      }

      /* Both the result and source operands of G17P f32log2 are bounded to
       * halves 0..191 by the recovered native form. dest_limit handles the
       * instruction result directly; mark the defining SSA of its source so
       * high register pressure cannot produce a post-RA refusal. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_LOG2 && I->src[0].type == AGX_INDEX_NORMAL)
         BITSET_SET(log2_gpr, I->src[0].value);

      /* agx3_tex_stage_cube_array_lod encodes the layer register (coords + 6
       * halves) in seven bits; keep such coordinate vectors below half 128. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_TEXTURE_SAMPLE && I->dim == AGX_DIM_CUBE_ARRAY &&
          I->src[0].type == AGX_INDEX_NORMAL && I->nr_srcs > 1 &&
          I->src[1].type == AGX_INDEX_NORMAL &&
          (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS))
         BITSET_SET(tex_low128, I->src[0].value);

      /* Record collects/phis so we can coalesce when assigning */
      if (I->op == AGX_OPCODE_COLLECT || I->op == AGX_OPCODE_PHI ||
          I->op == AGX_OPCODE_EXPORT || I->op == AGX_OPCODE_SPLIT) {
         agx_foreach_ssa_src(I, s) {
            src_to_collect_phi[I->src[s].value] = I;
         }
      }

      agx_foreach_ssa_dest(I, d) {
         unsigned v = I->dest[d].value;
         assert(ncomps[v] == 0 && "broken SSA");
         /* Round up vectors for easier live range splitting */
         ncomps_unrounded[v] = agx_index_size_16(I->dest[d]);
         ncomps[v] = util_next_power_of_two(ncomps_unrounded[v]);
         sizes[v] = I->dest[d].size;
         definition_order[v] = instruction_order;
         classes[v] = ra_class_for_index(I->dest[d]);

         max_ncomps = MAX2(max_ncomps, ncomps[v]);
      }
   }

   {
      unsigned hint_index = 0;
      util_dynarray_foreach(&ctx->collect_split_hints, agx_instr *, hint) {
         if (hint_index++ < original_hints &&
             (ssa_renumbered || ctx->scratch_size_B))
            continue;
         agx_instr *collect = *hint;
         unsigned dest = collect->dest[0].value;
         unsigned width = agx_size_align_16(collect->dest[0].size);
         unsigned span = collect->nr_srcs * width;
         if (ncomps[dest] || !span || span > 16)
            continue;
         bool valid = true;
         agx_foreach_src(collect, s) {
            agx_index src = collect->src[s];
            if (src.type == AGX_INDEX_UNDEF || agx_is_null(src))
               continue;
            if (src.type == AGX_INDEX_NORMAL && !ncomps[src.value]) {
               collect->src[s] = agx_undef(src.size);
               continue;
            }
            valid &= src.type == AGX_INDEX_NORMAL && !src.memory &&
               src.size == collect->dest[0].size && agx_channels(src) == 1 &&
               !src.abs && !src.neg && ncomps[src.value] == width;
         }
         if (!valid)
            continue;
         unsigned last_unexported = 0, last_unexported_source = 0;
         agx_foreach_ssa_src(collect, s) {
            unsigned value = collect->src[s].value;
            if (!src_to_collect_phi[value] ||
                src_to_collect_phi[value]->op != AGX_OPCODE_EXPORT) {
               last_unexported = MAX2(last_unexported, definition_order[value]);
               last_unexported_source = s;
            }
         }
         ncomps[dest] = util_next_power_of_two(span);
         agx_foreach_ssa_src(collect, s) {
            unsigned value = collect->src[s].value;
            agx_instr *affinity = src_to_collect_phi[value];
            if (affinity && affinity->op == AGX_OPCODE_EXPORT) {
               if (s > 0 && definition_order[value] < last_unexported &&
                   (s < last_unexported_source ||
                    (collect->nr_srcs == 2 &&
                     (!BITSET_TEST(effectful_use, collect->src[0].value) ||
                      pair_peer_publication_consumed_early(
                         ctx, collect->src[0].value)))))
                  BITSET_SET(early_scalar_export, value);
            } else if (!affinity || affinity->op == AGX_OPCODE_COLLECT) {
               src_to_collect_phi[value] = collect;
            }
         }
      }
   }

   /* The ordered eight-target RGBA8 reload keeps sixteen packed halves live
    * until its terminal tile stores. Its shared texture coordinates are also
    * bank-constrained, so the shuffler cannot evict them from that low bank.
    * Place those coordinates above it from the outset, within their existing
    * seven-bit COPY domain. This does not relax the tile or texture encodings.
    */
   if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
       ctx->agx3_mrt_tile_ops_ordered) {
      unsigned tile_halfs = 0;
      for (unsigned ssa = 0; ssa < ctx->alloc; ++ssa) {
         if (BITSET_TEST(low_gpr, ssa))
            tile_halfs += ncomps[ssa];
      }
      if (tile_halfs >= 16) {
         for (unsigned ssa = 0; ssa < ctx->alloc; ++ssa) {
            if (BITSET_TEST(texture_gpr, ssa) && !BITSET_TEST(low_gpr, ssa))
               BITSET_SET(high_gpr, ssa);
         }
      }
   }

   /* For live range splitting to work properly, ensure the register file is
    * aligned to the larger vector size. Most of the time, this is a no-op since
    * the largest vector size is usually 128-bit and the register file is
    * naturally 128-bit aligned. However, this is required for correctness with
    * 3D textureGrad, which can have a source vector of length 6x32-bit,
    * rounding up to 256-bit and requiring special accounting here.
    */
   unsigned reg_file_alignment = MAX2(max_ncomps, 8);
   assert(util_is_power_of_two_nonzero(reg_file_alignment));
   assert(reg_file_alignment <= 16 && "max size");

   unsigned demand = ALIGN_POT(effective_demand, reg_file_alignment);
   assert(demand <= max_possible_regs && "Invariant");

   /* Round up the demand to the maximum number of registers we can use without
    * affecting occupancy. This reduces live range splitting.
    */
   unsigned max_regs = agx_occupancy_for_register_count(demand).max_registers;
   if (ctx->key->is_helper || force_spilling)
      max_regs = max_possible_regs;

   /* Occupancy rounding must not reach past a smaller register file. */
   max_regs = MIN2(max_regs, max_possible_regs);

   max_regs = ROUND_DOWN_TO(max_regs, reg_file_alignment);

   /* Or, we can bound tightly for debugging */
   if (agx_compiler_debug & AGX_DBG_DEMAND)
      max_regs = ALIGN_POT(MAX2(demand, 12), reg_file_alignment);

   /* ...but not too tightly */
   assert((max_regs % reg_file_alignment) == 0 && "occupancy limits aligned");
   assert(max_regs >= (6 * 2) && "space for vertex shader preloading");
   assert(max_regs <= max_possible_regs);

   unsigned reg_count = 0, mem_slot_count = 0;

   /* Assign registers in dominance-order. This coincides with source-order due
    * to a NIR invariant, so we do not need special handling for this.
    */
   BITSET_WORD *dest_read = agx_values_read(ctx);

   /* Everything assign_local reads below was allocated above, and nothing the
    * cleanup reads is changed after this point, so it survives a longjmp. */
   jmp_buf ra_abort;
   if (setjmp(ra_abort)) {
      ctx->ra_abort = NULL;
      free(dest_read);
      goto cleanup;
   }
   ctx->ra_abort = &ra_abort;

   agx_foreach_block(ctx, block) {
      agx_ra_assign_local(&(struct ra_ctx){
         .shader = ctx,
         .block = block,
         .dest_read = dest_read,
         .src_to_collect_phi = src_to_collect_phi,
         .early_scalar_export = early_scalar_export,
         .phi_web = phi_web,
         .matrix_groups = matrix_groups.data,
         .web_matrix_group = web_matrix_group,
         .ncomps = ncomps,
         .ncomps_unrounded = ncomps_unrounded,
         .sizes = sizes,
         .classes = classes,
         .visited = visited,
         .low_gpr = low_gpr,
         .tile_gpr_limit = tile_gpr_limit,
         .texture_gpr = texture_gpr,
         .log2_gpr = log2_gpr,
         .high_gpr = high_gpr,
         .tex_low128 = tex_low128,
         .bound[RA_GPR] = max_regs,
         .bound[RA_MEM] = AGX_NUM_MODELED_REGS,
         .count[RA_GPR] = &reg_count,
         .count[RA_MEM] = &mem_slot_count,
      });
   }

   ctx->ra_abort = NULL;
   free(dest_read);

   ctx->max_reg = reg_count ? (reg_count - 1) : 0;
   ctx->spill_base_B = ctx->scratch_size_B;
   ctx->scratch_size_B += mem_slot_count * 2;

   /* Vertex shaders preload the vertex/instance IDs (r5, r6) even if the shader
    * don't use them. Account for that so the preload doesn't clobber GPRs.
    * Hardware tessellation eval shaders preload patch/instance IDs there.
    */
   if (ctx->nir->info.stage == MESA_SHADER_VERTEX ||
       ctx->nir->info.stage == MESA_SHADER_TESS_EVAL)
      ctx->max_reg = MAX2(ctx->max_reg, 6 * 2);

   assert(ctx->max_reg <= max_regs);

   /* Validate RA after assigning registers just before lowering SSA */
   agx_validate_ra(ctx);

   agx_foreach_instr_global_safe(ctx, ins) {
      /* Lower away SSA */
      agx_foreach_ssa_dest(ins, d) {
         ins->dest[d] =
            agx_replace_index(ins->dest[d], agx_as_register(ins->dest[d]));
      }

      agx_foreach_ssa_src(ins, s) {
         agx_replace_src(ins, s, agx_as_register(ins->src[s]));
      }

      /* Native AGX3b returned add/cmpxchg/xchg marks its consumed in-place
       * value discardable, while the standalone min/max return form does not.
       * RA has made the operand private and coalesced it with the destination
       * above. Set the physical hint only after SSA lowering, which
       * intentionally strips SSA liveness flags. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          ins->op == AGX_OPCODE_ATOMIC && ins->nr_dests == 1 &&
          !agx_is_null(ins->dest[0]) &&
          ins->atomic_opc != AGX_ATOMIC_OPC_UMIN &&
          ins->atomic_opc != AGX_ATOMIC_OPC_IMIN &&
          ins->atomic_opc != AGX_ATOMIC_OPC_UMAX &&
          ins->atomic_opc != AGX_ATOMIC_OPC_IMAX)
         ins->src[0].discard = true;

      /* Lower away RA pseudo-instructions */
      agx_builder b = agx_init_builder(ctx, agx_after_instr(ins));

      if (ins->op == AGX_OPCODE_COLLECT) {
         assert(ins->dest[0].type == AGX_INDEX_REGISTER);
         assert(!ins->dest[0].memory);

         unsigned base = ins->dest[0].value;
         unsigned width = agx_size_align_16(ins->src[0].size);

         struct agx_copy *copies = alloca(sizeof(copies[0]) * ins->nr_srcs);
         unsigned n = 0;

         /* Move the sources */
         agx_foreach_src(ins, i) {
            if (agx_is_null(ins->src[i]) || ins->src[i].type == AGX_INDEX_UNDEF)
               continue;
            assert(ins->src[i].size == ins->src[0].size);

            assert(n < ins->nr_srcs);
            copies[n++] = (struct agx_copy){
               .dest = base + (i * width),
               .src = ins->src[i],
            };
         }

         agx_emit_parallel_copies(&b, copies, n);
         agx_remove_instruction(ins);
         continue;
      } else if (ins->op == AGX_OPCODE_SPLIT) {
         assert(ins->src[0].type == AGX_INDEX_REGISTER ||
                ins->src[0].type == AGX_INDEX_UNIFORM);

         struct agx_copy copies[8];
         assert(ins->nr_dests <= ARRAY_SIZE(copies));

         unsigned n = 0;
         unsigned width = agx_size_align_16(agx_split_width(ins));

         /* Move the sources */
         agx_foreach_dest(ins, i) {
            if (ins->dest[i].type != AGX_INDEX_REGISTER)
               continue;

            assert(!ins->dest[i].memory);

            agx_index src = ins->src[0];
            src.size = ins->dest[i].size;
            src.channels_m1 = 0;
            src.value += (i * width);

            assert(n < ARRAY_SIZE(copies));
            copies[n++] = (struct agx_copy){
               .dest = ins->dest[i].value,
               .src = src,
            };
         }

         /* Lower away */
         agx_builder b = agx_init_builder(ctx, agx_after_instr(ins));
         agx_emit_parallel_copies(&b, copies, n);
         agx_remove_instruction(ins);
         continue;
      }
   }

   /* Insert parallel copies lowering phi nodes and exports */
   agx_foreach_block(ctx, block) {
      agx_insert_parallel_copies(ctx, block);
   }

   lower_exports(ctx);

   agx_foreach_instr_global_safe(ctx, I) {
      switch (I->op) {
      /* Pseudoinstructions for RA must be removed now */
      case AGX_OPCODE_PHI:
      case AGX_OPCODE_PRELOAD:
         agx_remove_instruction(I);
         break;

      /* Coalesced moves can be removed */
      case AGX_OPCODE_MOV:
         if (I->src[0].type == AGX_INDEX_REGISTER &&
             I->dest[0].size == I->src[0].size &&
             I->src[0].value == I->dest[0].value &&
             I->src[0].memory == I->dest[0].memory) {

            assert(I->dest[0].type == AGX_INDEX_REGISTER);
            agx_remove_instruction(I);
         }
         break;

      default:
         break;
      }
   }

   if (spilling)
      agx_lower_spill(ctx);

cleanup:
   agx_foreach_block(ctx, block) {
      for (unsigned i = 0; i < ARRAY_SIZE(block->reg_to_ssa_out); ++i) {
         free(block->reg_to_ssa_out[i]);
         block->reg_to_ssa_out[i] = NULL;
      }
   }

   free(phi_web);
   util_dynarray_fini(&matrix_groups);
   free(web_matrix_group);
   free(low_gpr);
   free(split_source);
   free(texture_gpr);
   free(log2_gpr);
   free(high_gpr);
   free(tex_low128);
   free(src_to_collect_phi);
   free(definition_order);
   free(early_scalar_export);
   free(effectful_use);
   free(ncomps);
   free(ncomps_unrounded);
   free(sizes);
   free(classes);
   free(visited);
}

void
agx_ra(agx_context *ctx)
{
   agx_ra_impl(ctx, false);
}

/* Retry from a fresh compiler context, retaining the original full budget. */
void
agx_ra_without_remat(agx_context *ctx)
{
   agx_ra_impl(ctx, true);
}
