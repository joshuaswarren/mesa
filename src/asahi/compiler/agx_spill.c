/*
 * Copyright 2023-2024 Alyssa Rosenzweig
 * Copyright 2023-2024 Valve Corporation
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "util/bitset.h"
#include "util/ralloc.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "util/u_qsort.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/*
 * An implementation of "Register Spilling and Live-Range Splitting for SSA-Form
 * Programs" by Braun and Hack.
 */

/*
 * Next-use distances are logically in ℤ ∪ {∞}. Modeled as saturating uint32 and
 * referred to as dist_t.
 *
 * next_uses represents a next-use map. This is a sparse data structure mapping
 * variable names to next-use dist_t's. Variables with no later use (infinite
 * next-use distance) are not stored explicitly, making the time/space
 * requirements O(live variables). This is important for performance and memory
 * usage on big shaders with many blocks.
 *
 * Backed by a compact open-addressed table with power-of-two capacity.
 * Distances decrease monotonically during fixed-point propagation, so entries
 * are never removed and no tombstones are needed.
 */
#define DIST_INFINITY (UINT32_MAX)
typedef uint32_t dist_t;

static inline dist_t
dist_sum(dist_t A, dist_t B)
{
   return (A + B < A) ? DIST_INFINITY : (A + B);
}

struct next_use_entry {
   uint32_t key; /* UINT32_MAX for empty */
   dist_t dist;
};

struct next_uses {
   void *memctx;
   struct next_use_entry *entries;
   uint32_t mask;
   uint32_t count;
};

static inline uint32_t
next_use_hash(uint32_t key)
{
   uint32_t h = key * 2654435761u;
   return h ^ (h >> 16);
}

static void
init_next_uses(struct next_uses *nu, void *memctx)
{
   nu->memctx = memctx;
   nu->entries = NULL;
   nu->mask = 0;
   nu->count = 0;
}

static void
next_uses_grow(struct next_uses *nu)
{
   uint32_t old_cap = nu->entries ? nu->mask + 1 : 0;
   uint32_t new_cap = old_cap ? old_cap * 2 : 16;
   uint32_t new_mask = new_cap - 1;
   struct next_use_entry *new_entries =
      ralloc_array(nu->memctx, struct next_use_entry, new_cap);
   memset(new_entries, 0xff, new_cap * sizeof(*new_entries));

   for (uint32_t i = 0; i < old_cap; ++i) {
      uint32_t k = nu->entries[i].key;
      if (k != UINT32_MAX) {
         uint32_t idx = next_use_hash(k) & new_mask;
         while (new_entries[idx].key != UINT32_MAX)
            idx = (idx + 1) & new_mask;
         new_entries[idx] = nu->entries[i];
      }
   }

   ralloc_free(nu->entries);
   nu->entries = new_entries;
   nu->mask = new_mask;
}

static inline bool
set_next_use_min(struct next_uses *nu, unsigned node, dist_t dist)
{
   assert(node != UINT32_MAX);
   assert(dist != DIST_INFINITY);

   if (!nu->entries || nu->count + 1 > ((nu->mask + 1) >> 2) * 3)
      next_uses_grow(nu);

   uint32_t mask = nu->mask;
   uint32_t idx = next_use_hash(node) & mask;
   for (;;) {
      struct next_use_entry *e = &nu->entries[idx];
      if (e->key == node) {
         if (dist < e->dist) {
            e->dist = dist;
            return true;
         }
         return false;
      }
      if (e->key == UINT32_MAX) {
         e->key = node;
         e->dist = dist;
         nu->count++;
         return true;
      }
      idx = (idx + 1) & mask;
   }
}

static inline dist_t
search_next_uses(const struct next_uses *nu, unsigned node)
{
   if (!nu->entries)
      return DIST_INFINITY;

   uint32_t mask = nu->mask;
   uint32_t idx = next_use_hash(node) & mask;
   for (;;) {
      const struct next_use_entry *e = &nu->entries[idx];
      if (e->key == node)
         return e->dist;
      if (e->key == UINT32_MAX)
         return DIST_INFINITY;
      idx = (idx + 1) & mask;
   }
}

#define foreach_next_use(nu, node, dist_)                                      \
   for (uint32_t _nu_i = 0, _nu_cap = (nu)->entries ? (nu)->mask + 1 : 0;      \
        _nu_i < _nu_cap; ++_nu_i)                                              \
      if ((nu)->entries[_nu_i].key != UINT32_MAX)                              \
         for (uint32_t _terminator = 1, node = (nu)->entries[_nu_i].key,       \
                       UNUSED dist_ = (nu)->entries[_nu_i].dist;               \
              _terminator != 0; _terminator = 0)

static uint32_t
instr_cycles(const agx_instr *I)
{
   return 1;
}

struct spill_block {
   /* Set of values available in the register file at the end */
   unsigned W_exit[AGX_NUM_REGS];
   unsigned nW_exit;

   unsigned W_entry[AGX_NUM_REGS];
   unsigned nW_entry;

   /* Set of live-out spilled values at the end of the block */
   unsigned *S_exit;
   unsigned nS_exit;

   unsigned *S_entry;
   unsigned nS_entry;

   /* Estimate */
   uint32_t cycles;

   /* Next-use maps at the start/end of the block */
   struct next_uses next_use_in;
   struct next_uses next_use_out;
};

struct candidate;

struct spill_ctx {
   void *memctx;
   agx_context *shader;
   agx_block *block;

   /* Set of values currently available in the register file */
   BITSET_WORD *W;

   /* |W| = Current register pressure */
   unsigned nW;

   /* Local IPs of next-use */
   dist_t *next_uses;

   /* Scratch array (size n, all DIST_INFINITY between calls) for
    * calculate_local_next_use(). */
   dist_t *local_nu;

   /* Reusable scratch buffers for compute_w_entry() and insert_coupling_code(). */
   unsigned *freq;
   struct candidate *candidates;
   BITSET_WORD *pred_W;
   BITSET_WORD *pred_S;
   BITSET_WORD *phi_any_dests;
   BITSET_WORD *phi_raw_dests;

   /* Current local IP relative to the start of the block */
   uint32_t ip;

   /* Set of live values that have been spilled. Contrary to the paper, this
    * is not a subset of W: the definition in the paper is bogus.
    */
   BITSET_WORD *S;

   /* Widths of vectors */
   uint8_t *channels;
   enum agx_size *size;

   /* Mapping of rematerializable values to their definitions, or NULL for nodes
    * that are not materializable.
    */
   agx_instr **remat;

   /* AGX3 instructions the spiller may repeat to reload a value (see
    * can_repeat_alu and can_remat_load), indexed by the node holding their
    * result or one component of it, and that component. NULL for other nodes.
    */
   agx_instr **repeat;
   uint8_t *repeat_comp;

   /* Maximum register pressure allowed */
   unsigned k;

   /* Number of variables */
   unsigned n;

   /* Information on blocks indexed in source order */
   struct spill_block *blocks;

   /* Base memory index reserved for spilled indices */
   unsigned spill_base;

   /* If true, we cannot use scratch and must only rematerialize. Values that
    * cannot be rematerialized then stay in registers even when that exceeds
    * k (see keep_in_registers()).
    */
   bool remat_only;
};

static inline struct spill_block *
spill_block(struct spill_ctx *ctx, agx_block *block)
{
   return &ctx->blocks[block->index];
}

/* Calculate the register demand of a node. This is rounded up to a power-of-two
 * to match the equivalent calculations in RA.
 */
static unsigned
node_size(struct spill_ctx *ctx, unsigned node)
{
   return util_next_power_of_two(ctx->channels[node]) *
          agx_size_align_16(ctx->size[node]);
}

/*
 * Map a control flow edge to a block. Assumes no critical edges.
 */
static agx_block *
agx_edge_to_block(agx_block *pred, agx_block *succ)
{
   /* End of predecessor is unique if there's a single successor */
   if (agx_num_successors(pred) == 1)
      return pred;

   /* The predecessor has multiple successors, meaning this is not the only
    * edge leaving the predecessor. Therefore, it is the only edge entering
    * the successor (otherwise the edge would be critical), so the start of
    * the successor is unique.
    */
   assert(agx_num_predecessors(succ) == 1 && "critical edge detected");
   return succ;
}

/*
 * Get a cursor to insert along a control flow edge: either at the start of the
 * successor or the end of the predecessor. This relies on the control flow
 * graph having no critical edges.
 */
static agx_cursor
agx_along_edge(agx_block *pred, agx_block *succ)
{
   agx_block *to = agx_edge_to_block(pred, succ);

   if (to == pred)
      return agx_after_block_logical(pred);
   else
      return agx_before_block(succ);
}

static inline agx_index
agx_index_as_mem(agx_index idx, unsigned mem_base)
{
   assert(idx.type == AGX_INDEX_NORMAL);
   assert(!idx.memory);
   idx.memory = true;
   idx.value = mem_base + idx.value;
   return idx;
}

static unsigned
chase_mem_index(agx_index ref, unsigned mem_base)
{
   assert(ref.type == AGX_INDEX_NORMAL);
   return ref.memory ? (ref.value - mem_base) : ref.value;
}

static agx_index
reconstruct_index(struct spill_ctx *ctx, unsigned node)
{
   return agx_get_vec_index(node, ctx->size[node], ctx->channels[node]);
}

bool
agx_has_uniform_store(const agx_context *ctx)
{
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_UNIFORM_STORE)
         return true;
   }
   return false;
}

/*
 * A spilled copy of a uniform word can be made again at its use: the AGX3B
 * and AGX3C copy forms read any even uniform pair into any 32-bit register.
 * This covers the AGX3 word copy only; AGX2 reads uniforms with its own bitop
 * and keeps its spill behaviour. Not on G18G (AGX3C re-encoded as agx3d,
 * emit_agx3d): a rematerialized copy is placed under register pressure, where
 * its destination is often at or above half 128, and agx3d copies into those
 * destinations have not been verified to execute correctly (the direct agx3d
 * encoder, agx3_copy_encode_uniform_to_reg32, only emits lower ones).
 */
bool
agx_can_rematerialize_uniform(const agx_context *ctx, const agx_instr *I)
{
   return ctx->key &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
            !agx_is_g18g(&ctx->key->dev))) &&
          !ctx->is_preamble && !ctx->uniforms_written &&
          I->op == AGX_OPCODE_MOV && I->nr_dests == 1 && I->nr_srcs == 1 &&
          I->dest[0].type == AGX_INDEX_NORMAL && !I->dest[0].memory &&
          I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
          I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
          I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
          I->src[0].value < 512 && !(I->src[0].value & 1) && !I->src[0].abs &&
          !I->src[0].neg;
}

static bool
can_remat(agx_context *ctx, agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
   case AGX_OPCODE_GET_SR:
      return true;
   default:
      return agx_can_rematerialize_uniform(ctx, I);
   }
}

static agx_instr *
remat_to(agx_builder *b, agx_index dst, struct spill_ctx *ctx, unsigned node)
{
   agx_instr *I = ctx->remat[node];
   assert(can_remat(b->shader, I));

   switch (I->op) {
   case AGX_OPCODE_MOV:
      return agx_mov_to(b, dst, I->src[0]);
   case AGX_OPCODE_MOV_IMM:
      return agx_mov_imm_to(b, dst, I->imm);
   case AGX_OPCODE_GET_SR:
      return agx_get_sr_to(b, dst, I->sr);
   default:
      UNREACHABLE("invalid remat");
   }
}

/*
 * On AGX3, a spilled value can often be computed again instead of filled from
 * its spill slot: by repeating the device load that produced it, where one
 * load replaces a fill per component, or the arithmetic instruction that did.
 * Once no fill reads the slot, DCE removes the spill stores too.
 *
 * Repeating must yield the same value anywhere the value is live. Arithmetic
 * qualifies if it reads nothing but its operands. A load qualifies if it reads
 * memory that nothing writes (memory_can_reorder, from nir_opt_access, which
 * allows rematerializing such reads). Register operands must be in the
 * register file at the reload; the spiller's register file sets decide that,
 * so the reload never extends a live range.
 */
static unsigned
remat_load_elem_bytes(const agx_instr *L)
{
   switch (L->format) {
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return 2;
   case AGX_FORMAT_I32:
      return 4;
   default:
      return 0;
   }
}

static bool
can_remat_load(const agx_context *ctx, const agx_instr *L)
{
   if (!ctx->key || !agx_isa_is_agx3(ctx->key->dev.encoding) ||
       L->op != AGX_OPCODE_DEVICE_LOAD || !L->memory_can_reorder ||
       L->memory_is_packed || ctx->is_preamble || L->nr_dests != 1 ||
       L->dest[0].type != AGX_INDEX_NORMAL || L->dest[0].memory ||
       L->mask != BITFIELD_MASK(agx_channels(L->dest[0])))
      return false;

   /* One memory element per register component */
   if (remat_load_elem_bytes(L) != agx_size_align_16(L->dest[0].size) * 2)
      return false;

   agx_foreach_src(L, s) {
      const agx_index src = L->src[s];

      if (src.type == AGX_INDEX_UNIFORM) {
         if (ctx->uniforms_written)
            return false;
      } else if (src.type == AGX_INDEX_NORMAL) {
         if (src.memory)
            return false;
      } else if (src.type != AGX_INDEX_IMMEDIATE) {
         return false;
      }
   }

   return true;
}

static bool
can_repeat_alu(const agx_context *ctx, const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_FFMA:
   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
   case AGX_OPCODE_FCMPSEL:
   case AGX_OPCODE_ICMPSEL:
   case AGX_OPCODE_IADD:
   case AGX_OPCODE_IMAD:
   case AGX_OPCODE_ASR:
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_EXTR:
   case AGX_OPCODE_BITOP:
   case AGX_OPCODE_CONVERT:
   case AGX_OPCODE_FLTNORM_AGX3:
   case AGX_OPCODE_NORMFLT_AGX3:
      break;
   default:
      return false;
   }

   /* A texture coordinate producer is paired with its sample */
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) || ctx->is_preamble ||
       I->nr_dests != 1 || I->dest[0].type != AGX_INDEX_NORMAL ||
       I->dest[0].memory || I->agx3_tex_coord_stage ||
       I->agx3_tex_coord_component || I->agx3_tex_coord_start)
      return false;

   agx_foreach_src(I, s) {
      const agx_index src = I->src[s];

      if (src.type == AGX_INDEX_UNIFORM) {
         if (ctx->uniforms_written)
            return false;
      } else if (src.type == AGX_INDEX_NORMAL) {
         if (src.memory)
            return false;
      } else if (src.type != AGX_INDEX_IMMEDIATE &&
                 src.type != AGX_INDEX_NULL) {
         return false;
      }
   }

   return true;
}

/* Byte offset of a load of `comp` onwards, if it is still encodable */
static bool
remat_load_offset(const agx_instr *L, unsigned comp, int16_t *offset)
{
   if (L->op != AGX_OPCODE_DEVICE_LOAD) {
      *offset = 0;
      return true;
   }

   const unsigned bytes = remat_load_elem_bytes(L);
   const int32_t off = L->memory_byte_offset + (int32_t)(comp * bytes);

   if (off > INT16_MAX)
      return false;

   /* An immediate index folds into the offset, which must then fit too */
   if (comp && L->src[1].type == AGX_INDEX_IMMEDIATE) {
      const unsigned shift = L->shift +
         (L->memory_index_is_bytes ? 0 : util_logbase2(bytes));
      const int64_t total = ((int64_t)L->src[1].value << shift) + off;

      if (total < INT16_MIN || total > INT16_MAX)
         return false;
   }

   *offset = off;
   return true;
}

static bool
srcs_in_regs(const agx_instr *I, const BITSET_WORD *regs)
{
   agx_foreach_ssa_src(I, s) {
      if (!BITSET_TEST(regs, I->src[s].value))
         return false;
   }

   return true;
}

/*
 * Whether node can be reloaded by repeating its instruction with the
 * registers in regs.
 *
 * Repeated arithmetic may also make one of its operands again when that
 * operand has left the register file: a constant that can be rematerialized,
 * or repeatable arithmetic over registers in the file. A rotated sample
 * offset ffma(c, k1, -(s * k2)) is the typical case: the product dies at the
 * ffma, so only s and c stay live. The operand goes to a new value that dies
 * at the repeated instruction, which needs room for one more value of its
 * size while the pair runs. If so, *operand is that operand's node,
 * otherwise ~0.
 */
static bool
can_reload_by_repeat(struct spill_ctx *ctx, unsigned node,
                     const BITSET_WORD *regs, unsigned room, unsigned *operand)
{
   agx_instr *L = ctx->repeat[node];
   int16_t offset;

   *operand = ~0;

   if (!L || !remat_load_offset(L, ctx->repeat_comp[node], &offset))
      return false;

   agx_foreach_ssa_src(L, s) {
      const unsigned v = L->src[s].value;
      if (BITSET_TEST(regs, v) || v == *operand)
         continue;

      /* At most one operand: a constant, or arithmetic over registers in
       * the file
       */
      const agx_instr *J = ctx->repeat[v];
      if (L->op == AGX_OPCODE_DEVICE_LOAD || *operand != ~0 ||
          node_size(ctx, v) > room)
         return false;

      if (!ctx->remat[v] &&
          (!J || J->op == AGX_OPCODE_DEVICE_LOAD || !srcs_in_regs(J, regs)))
         return false;

      *operand = v;
   }

   return true;
}

/* Whether repeating node, with its operand computed again if any, reads v */
static bool
repeat_reads(const struct spill_ctx *ctx, unsigned node, unsigned operand,
             unsigned v)
{
   const agx_instr *L = ctx->repeat[node];

   agx_foreach_ssa_src(L, s) {
      if (L->src[s].value == v && v != operand)
         return true;
   }

   if (operand != ~0 && !ctx->remat[operand]) {
      const agx_instr *J = ctx->repeat[operand];

      agx_foreach_ssa_src(J, s) {
         if (J->src[s].value == v)
            return true;
      }
   }

   return false;
}

static agx_instr *
repeat_to(agx_builder *b, const agx_instr *L, agx_index dst)
{
   agx_instr *I = agx_alloc_instr(b, L->op, 1, L->nr_srcs);
   agx_index *dest = I->dest, *src = I->src;
   memcpy(I, L, sizeof(*I));
   I->dest = dest;
   I->src = src;
   I->nr_dests = 1;
   memcpy(src, L->src, sizeof(*src) * L->nr_srcs);

   /* Sources are renamed to their reaching copies by SSA repair */
   dest[0] = dst;
   agx_builder_insert(&b->cursor, I);
   return I;
}

static void
reload_by_repeat(agx_builder *b, struct spill_ctx *ctx, unsigned node,
                 unsigned operand)
{
   const agx_instr *L = ctx->repeat[node];
   const agx_index dst = reconstruct_index(ctx, node);
   int16_t offset;
   ASSERTED bool ok = remat_load_offset(L, ctx->repeat_comp[node], &offset);
   assert(ok);

   /* The operand gets a value of its own, so its old live range is not
    * extended and SSA repair has nothing to rename for it.
    */
   agx_index tmp = agx_null();
   if (operand != ~0) {
      tmp = agx_temp_like(ctx->shader, reconstruct_index(ctx, operand));

      if (ctx->remat[operand])
         remat_to(b, tmp, ctx, operand);
      else
         repeat_to(b, ctx->repeat[operand], tmp);
   }

   agx_instr *I = repeat_to(b, L, dst);

   if (operand != ~0) {
      agx_foreach_ssa_src(I, s) {
         if (I->src[s].value == operand)
            I->src[s].value = tmp.value;
      }
   }

   if (L->op == AGX_OPCODE_DEVICE_LOAD) {
      I->mask = BITFIELD_MASK(agx_channels(dst));
      I->memory_byte_offset = offset;

      /* Fragment loads at immediate indices may need reordering after RA */
      ctx->shader->spilled_by_load = true;
   }
}

static void
record_spill(struct spill_ctx *ctx)
{
   if (!ctx->shader->key->has_scratch)
      ctx->shader->ra_failed = true;
   if (ctx->remat_only)
      ctx->shader->remat_fallback_spilled = true;
   ctx->shader->has_spill_pcopy_reserved = true;
}

static void
insert_spill(agx_builder *b, struct spill_ctx *ctx, unsigned node)
{
   if (!ctx->remat[node]) {
      /* Values that cannot be rematerialized go to scratch. Only the full
       * spill pass evicts them: the remat-only pass keeps them in registers
       * (keep_in_registers()). */
      record_spill(ctx);

      agx_index idx = reconstruct_index(ctx, node);
      agx_mov_to(b, agx_index_as_mem(idx, ctx->spill_base), idx);
   }
}

static void
insert_reload(struct spill_ctx *ctx, agx_block *block, agx_cursor cursor,
              unsigned node, bool by_repeat, unsigned operand)
{
   agx_builder b = agx_init_builder(ctx->shader, cursor);
   agx_index idx = reconstruct_index(ctx, node);

   /* Reloading breaks SSA, but agx_repair_ssa will repair */
   if (ctx->remat[node]) {
      remat_to(&b, idx, ctx, node);
   } else if (by_repeat) {
      reload_by_repeat(&b, ctx, node, operand);
   } else {
      agx_mov_to(&b, idx, agx_index_as_mem(idx, ctx->spill_base));
   }
}

/* Insert into the register file */
static void
insert_W(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   assert(!BITSET_TEST(ctx->W, v));

   BITSET_SET(ctx->W, v);
   ctx->nW += node_size(ctx, v);
}

/* Remove from the register file */
static void
remove_W(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   assert(BITSET_TEST(ctx->W, v));

   BITSET_CLEAR(ctx->W, v);
   ctx->nW -= node_size(ctx, v);
}

static void
remove_W_if_present(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);

   if (BITSET_TEST(ctx->W, v))
      remove_W(ctx, v);
}

struct candidate {
   unsigned node;
   dist_t dist;
};

/*
 * The remat-only pass is an occupancy optimization: it lowers pressure towards
 * a smaller budget by rematerializing constants, but the program already fits
 * the full register file without it. Its budget is only an estimate, so it may
 * be unreachable by rematerialization alone. Never move a value that cannot be
 * rematerialized out of the register file to meet it: keep the value and let
 * W exceed k. Register allocation measures the result and allocates it under
 * the full budget when the estimate was missed (agx_ra()). The pressure of the
 * result never exceeds the original program's, since every rematerialized
 * value is recreated inside the live range of the value it replaces.
 *
 * Any other choice introduces scratch traffic that the program without
 * rematerialization does not need.
 */
static bool
keep_in_registers(const struct spill_ctx *ctx, unsigned node)
{
   return ctx->remat_only && !ctx->remat[node];
}

static int
cmp_dist(const void *left_, const void *right_, void *ctx_)
{
   struct spill_ctx *ctx = ctx_;
   const struct candidate *left = left_;
   const struct candidate *right = right_;

   /* We assume that rematerializing - even before every instruction - is
    * cheaper than spilling. As long as one of the nodes is rematerializable
    * (with distance > 0), we choose it over spilling. Within a class of nodes
    * (rematerializable or not), compare by next-use-distance, breaking ties
    * deterministically by descending SSA index so spilling decisions do not
    * depend on hash-table iteration order or qsort instability and prefer
    * keeping more recently defined inner values over older outer values.
    */
   bool remat_left = ctx->remat[left->node] != NULL && left->dist > 0;
   bool remat_right = ctx->remat[right->node] != NULL && right->dist > 0;

   if (remat_left != remat_right)
      return remat_left ? 1 : -1;
   else if (left->dist != right->dist)
      return (left->dist > right->dist) - (left->dist < right->dist);
   else
      return (right->node > left->node) - (right->node < left->node);
}

/*
 * Limit the register file W to maximum size m by evicting registers.
 */
static ATTRIBUTE_NOINLINE void
limit(struct spill_ctx *ctx, agx_instr *I, unsigned m)
{
   /* Nothing to do if we're already below the limit */
   if (ctx->nW <= m)
      return;

   /* Gather candidates for eviction. Note that next_uses gives IPs whereas
    * cmp_dist expects relative distances. This requires us to subtract ctx->ip
    * to ensure that cmp_dist works properly. Even though logically it shouldn't
    * affect the sorted order, practically this matters for correctness with
    * rematerialization. See the dist=0 test in cmp_dist.
    */
   struct candidate *candidates = alloca(ctx->nW * sizeof(struct candidate));
   unsigned j = 0;

   int i;
   BITSET_FOREACH_SET(i, ctx->W, ctx->n) {
      assert(j < ctx->nW);

      candidates[j++] = (struct candidate){
         .node = i,
         .dist = ctx->next_uses[i] - ctx->ip,
      };
   }

   /* Sort by next-use distance */
   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   /* Evict what doesn't fit */
   unsigned new_weight = 0;

   for (i = 0; i < j; ++i) {
      unsigned v = candidates[i].node;
      unsigned comps = node_size(ctx, v);

      if ((new_weight + comps) <= m) {
         new_weight += comps;
      } else if (keep_in_registers(ctx, v) &&
                 candidates[i].dist < DIST_INFINITY) {
         new_weight += comps;
      } else {
         /* Insert a spill if we haven't spilled before and there is
          * another use
          */
         if (!BITSET_TEST(ctx->S, v) && candidates[i].dist < DIST_INFINITY) {
            agx_builder b = agx_init_builder(ctx->shader, agx_before_instr(I));
            insert_spill(&b, ctx, v);
            BITSET_SET(ctx->S, v);
         }

         remove_W(ctx, v);

         /* We keep going in case we can pack in a scalar */
      }
   }
}

/*
 * Insert coupling code on block boundaries. This must ensure:
 *
 *    - anything live-in we expect to have spilled is spilled
 *    - anything live-in we expect to have filled is filled
 *    - phi sources are spilled if the destination is spilled
 *    - phi sources are filled if the destination is not spilled
 *
 * The latter two requirements ensure correct pressure calculations for phis.
 */
static ATTRIBUTE_NOINLINE void
insert_coupling_code(struct spill_ctx *ctx, agx_block *pred, agx_block *succ)
{
   struct spill_block *sp = spill_block(ctx, pred);
   struct spill_block *ss = spill_block(ctx, succ);

   BITSET_WORD *pred_W = ctx->pred_W;
   BITSET_WORD *pred_S = ctx->pred_S;
   BITSET_WORD *phi_any_dests = ctx->phi_any_dests;
   BITSET_WORD *phi_raw_dests = ctx->phi_raw_dests;
   BITSET_WORD *edge_regs = ctx->W;

   for (unsigned i = 0; i < sp->nW_exit; ++i)
      BITSET_SET(pred_W, sp->W_exit[i]);

   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_SET(pred_S, sp->S_exit[i]);

   /* Values in registers on both sides of the edge, where a reload may read
    * them without extending their live ranges. Nothing else is in a register
    * along the edge, so what the start of succ does not use is free there.
    * There is none when the entry exceeds k, which a remat-only pass allows.
    */
   unsigned entry_size = 0;
   for (unsigned j = 0; j < ss->nW_entry; ++j) {
      unsigned v = ss->W_entry[j];
      entry_size += node_size(ctx, v);
      if (BITSET_TEST(pred_W, v))
         BITSET_SET(edge_regs, v);
   }

   const unsigned room = entry_size < ctx->k ? ctx->k - entry_size : 0;

   agx_foreach_phi_in_block(succ, phi) {
      unsigned chased = chase_mem_index(phi->dest[0], ctx->spill_base);
      if (chased < ctx->n)
         BITSET_SET(phi_any_dests, chased);
      if (phi->dest[0].value < ctx->n)
         BITSET_SET(phi_raw_dests, phi->dest[0].value);
   }

   unsigned s = agx_predecessor_index(succ, pred);

   agx_foreach_phi_in_block(succ, I) {
      if (!I->dest[0].memory)
         continue;

      agx_builder b =
         agx_init_builder(ctx->shader, agx_before_function(ctx->shader));

      /* An immediate or uniform phi source needs no memory variable of its
       * own: parallel copy lowering writes it to the phi's slot on the edge,
       * through the scratch register a memory-memory copy uses
       * (agx_lower_parallel_copy.c do_copy()). A variable made at the start
       * of the program would instead stay live, in a slot of its own, until
       * the phi, and the edge would copy it from memory to memory. That is
       * a fill per traversal of the edge where a uniform read suffices.
       *
       * A uniform the program writes must still be read at the start.
       */
      if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
          (I->src[s].type == AGX_INDEX_UNIFORM &&
           !ctx->shader->uniforms_written)) {
         record_spill(ctx);
         continue;
      }

      /* Copy other uniform phi sources to memory variables at the start of
       * the program, where pressure is zero and hence the copy is legal.
       */
      if (I->src[s].type != AGX_INDEX_NORMAL) {
         assert(I->src[s].type == AGX_INDEX_UNIFORM);

         agx_index mem = agx_temp_like(ctx->shader, I->dest[0]);
         assert(mem.memory);

         agx_index gpr = agx_temp_like(ctx->shader, I->dest[0]);
         gpr.memory = false;

         if (I->src[s].type == AGX_INDEX_IMMEDIATE)
            agx_mov_imm_to(&b, gpr, I->src[s].value);
         else
            agx_mov_to(&b, gpr, I->src[s]);

         agx_mov_to(&b, mem, gpr);
         record_spill(ctx);
         I->src[s] = mem;
         continue;
      }

      bool spilled = BITSET_TEST(pred_S, I->src[s].value);

      if (!spilled) {
         /* Spill the phi source. TODO: avoid redundant spills here */
         agx_builder b =
            agx_init_builder(ctx->shader, agx_after_block_logical(pred));

         insert_spill(&b, ctx, I->src[s].value);
      }

      if (ctx->remat[I->src[s].value]) {
         unsigned node = I->src[s].value;
         agx_index idx = reconstruct_index(ctx, node);
         agx_index tmp = agx_temp_like(ctx->shader, idx);

         remat_to(&b, tmp, ctx, node);
         agx_mov_to(&b, agx_index_as_mem(idx, ctx->spill_base), tmp);
         record_spill(ctx);
      }

      /* Use the spilled version */
      I->src[s] = agx_index_as_mem(I->src[s], ctx->spill_base);
   }

   /* Anything assumed to be spilled at the start of succ must be spilled along
    * all edges.
    */
   for (unsigned i = 0; i < ss->nS_entry; ++i) {
      unsigned v = ss->S_entry[i];

      /* We handle spilling phi destinations separately */
      if (BITSET_TEST(pred_S, v) || BITSET_TEST(phi_any_dests, v))
         continue;

      agx_builder b = agx_init_builder(ctx->shader, agx_along_edge(pred, succ));
      insert_spill(&b, ctx, v);
   }

   /* Variables in W at the start of succ must be defined along the edge. */
   for (unsigned i = 0; i < ss->nW_entry; ++i) {
      unsigned node = ss->W_entry[i];

      /* Variables live at the end of the predecessor and phis are defined along
       * the edge. */
      if (BITSET_TEST(pred_W, node) || BITSET_TEST(phi_raw_dests, node))
         continue;

      /* Otherwise, inserting a reload defines the variable along the edge */
      agx_block *reload_block = agx_edge_to_block(pred, succ);
      unsigned operand;
      bool by_repeat =
         can_reload_by_repeat(ctx, node, edge_regs, room, &operand);
      insert_reload(ctx, reload_block, agx_along_edge(pred, succ), node,
                    by_repeat, operand);
   }

   agx_foreach_phi_in_block(succ, I) {
      if (I->dest[0].memory)
         continue;

      /* Treat immediate/uniform phi sources as registers for pressure
       * accounting and phi lowering purposes. Parallel copy lowering can handle
       * a copy from a immediate/uniform to a register, but not from an
       * immediate/uniform directly to memory.
       */
      if (I->src[s].type != AGX_INDEX_NORMAL) {
         assert(I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].type == AGX_INDEX_UNIFORM);

         continue;
      }

      /* Fill the phi source in the predecessor */
      if (!BITSET_TEST(pred_W, I->src[s].value)) {
         agx_block *reload_block = agx_edge_to_block(pred, succ);
         unsigned operand;
         bool by_repeat = can_reload_by_repeat(ctx, I->src[s].value, edge_regs,
                                               room, &operand);
         insert_reload(ctx, reload_block, agx_along_edge(pred, succ),
                       I->src[s].value, by_repeat, operand);
      }

      /* Leave as-is for the GPR version */
      assert(!I->src[s].memory);
   }

   /* Restore scratch bitsets sparsely for the next edge. */
   for (unsigned i = 0; i < sp->nW_exit; ++i)
      BITSET_CLEAR(pred_W, sp->W_exit[i]);

   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_CLEAR(pred_S, sp->S_exit[i]);

   for (unsigned j = 0; j < ss->nW_entry; ++j)
      BITSET_CLEAR(edge_regs, ss->W_entry[j]);

   agx_foreach_phi_in_block(succ, phi) {
      unsigned chased = chase_mem_index(phi->dest[0], ctx->spill_base);
      if (chased < ctx->n)
         BITSET_CLEAR(phi_any_dests, chased);
      if (phi->dest[0].value < ctx->n)
         BITSET_CLEAR(phi_raw_dests, phi->dest[0].value);
   }
}

/*
 * Produce an array of next-use IPs relative to the start of the block. This is
 * an array of dist_t scalars, representing the next-use IP of each SSA dest
 * (right-to-left) and SSA source (left-to-right) of each instruction in the
 * block (bottom-to-top). Its size equals the # of SSA sources in the block.
 */
static ATTRIBUTE_NOINLINE void
calculate_local_next_use(struct spill_ctx *ctx, struct util_dynarray *out)
{
   struct spill_block *sb = spill_block(ctx, ctx->block);
   unsigned ip = sb->cycles;
   dist_t *nu = ctx->local_nu;

   *out = UTIL_DYNARRAY_INIT;

   foreach_next_use(&sb->next_use_out, i, dist) {
      assert(i < ctx->n);
      nu[i] = dist_sum(ip, dist);
   }

   agx_foreach_instr_in_block_rev(ctx->block, I) {
      ip -= instr_cycles(I);

      if (I->op != AGX_OPCODE_PHI) {
         agx_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;
            assert(v < ctx->n);

            util_dynarray_append(out, nu[v]);
            nu[v] = ip;
         }

         agx_foreach_ssa_dest_rev(I, d) {
            unsigned v = I->dest[d].value;
            assert(v < ctx->n);

            util_dynarray_append(out, nu[v]);
         }
      }
   }

   assert(ip == 0 && "cycle counting is consistent");

   /* Restore touched entries to DIST_INFINITY for the next block. */
   foreach_next_use(&sb->next_use_out, i, _) {
      nu[i] = DIST_INFINITY;
   }
   agx_foreach_instr_in_block(ctx->block, I) {
      if (I->op != AGX_OPCODE_PHI) {
         agx_foreach_ssa_src(I, s) {
            nu[I->src[s].value] = DIST_INFINITY;
         }
      }
   }
}

static agx_cursor
agx_before_instr_logical(agx_block *block, agx_instr *I)
{
   if (I->op == AGX_OPCODE_EXPORT) {
      agx_instr *first = agx_first_instr(block);

      while (I != first && I->op == AGX_OPCODE_EXPORT) {
         I = agx_prev_op(I);
      }

      if (I == first && first->op == AGX_OPCODE_EXPORT) {
         return agx_before_block(block);
      } else {
         return agx_after_instr(I);
      }
   } else {
      return agx_before_instr(I);
   }
}

/*
 * Insert spills/fills for a single basic block, following Belady's algorithm.
 * Corresponds to minAlgorithm from the paper.
 */
static ATTRIBUTE_NOINLINE void
min_algorithm(struct spill_ctx *ctx)
{
   struct spill_block *sblock = spill_block(ctx, ctx->block);
   struct util_dynarray local_next_ip;
   calculate_local_next_use(ctx, &local_next_ip);

   /* next_uses gives the distance from the start of the block, so prepopulate
    * with next_use_in.
    */
   foreach_next_use(&sblock->next_use_in, key, dist) {
      assert(key < ctx->n);
      ctx->next_uses[key] = dist;
   }

   dist_t *next_ips = util_dynarray_element(&local_next_ip, dist_t, 0);
   unsigned next_use_cursor =
      util_dynarray_num_elements(&local_next_ip, dist_t);

   /* Iterate each instruction in forward order */
   agx_foreach_instr_in_block(ctx->block, I) {
      assert((ctx->nW <= ctx->k ||
              (ctx->remat_only && ctx->nW <= AGX_NUM_REGS)) &&
             "invariant");

      /* Debug to check against our RA demand calculations */
      if (0) {
         printf("%u: ", ctx->nW);
         agx_print_instr(I, stdout);
      }

      /* Phis are special since they happen along the edge. When we initialized
       * W and S, we implicitly chose which phis are spilled. So, here we just
       * need to rewrite the phis to write into memory.
       *
       * Phi sources are handled later.
       */
      if (I->op == AGX_OPCODE_PHI) {
         if (!BITSET_TEST(ctx->W, I->dest[0].value)) {
            I->dest[0] = agx_index_as_mem(I->dest[0], ctx->spill_base);
         }

         ctx->ip += instr_cycles(I);
         continue;
      }

      /* Any source that is not in W needs to be reloaded. Gather the set R of
       * such values.
       */
      unsigned R[AGX_MAX_NORMAL_SOURCES];
      unsigned nR = 0;

      agx_foreach_ssa_src(I, s) {
         unsigned node = I->src[s].value;
         if (BITSET_TEST(ctx->W, node))
            continue;

         /* Mark this variable as needing a reload. */
         assert(node < ctx->n);
         assert(BITSET_TEST(ctx->S, node) && "must have been spilled");
         assert(nR < ARRAY_SIZE(R) && "maximum source count");
         R[nR++] = node;

         /* The inserted reload will add the value to the register file. */
         insert_W(ctx, node);
      }

      /* Limit W to make space for the sources we just added */
      limit(ctx, I, ctx->k);

      agx_foreach_ssa_dest(I, d) {
         assert(next_use_cursor >= 1);
         unsigned next_ip = next_ips[--next_use_cursor];

         if (next_ip == DIST_INFINITY)
            remove_W_if_present(ctx, I->dest[d].value);
         else
            ctx->next_uses[I->dest[d].value] = next_ip;
      }

      /* Count how many registers we need for destinations. Because of
       * SSA form, destinations are unique.
       */
      unsigned dest_size = 0;
      agx_foreach_ssa_dest(I, d) {
         dest_size += node_size(ctx, I->dest[d].value);
      }

      /* Limit W to make space for the destinations. */
      limit(ctx, I, ctx->k - dest_size);

      /* W now holds what is in registers at I. A repeated instruction may
       * read anything there except another repeated reload, which it would
       * have to follow. The room left for I's destinations is free until I,
       * so an operand computed again for a repeat may use it. Reloads for an
       * export are hoisted above every export of the group, whose sources
       * are all live there, so they do not get that room, and there is none
       * while W exceeds k, which a remat-only pass allows.
       */
      const unsigned room =
         I->op == AGX_OPCODE_EXPORT || ctx->nW >= ctx->k ? 0 : ctx->k - ctx->nW;
      bool by_repeat[AGX_MAX_NORMAL_SOURCES];
      unsigned operand[AGX_MAX_NORMAL_SOURCES];
      for (unsigned i = 0; i < nR; ++i) {
         by_repeat[i] =
            can_reload_by_repeat(ctx, R[i], ctx->W, room, &operand[i]);
      }

      for (unsigned i = 0; i < nR; ++i) {
         for (unsigned j = 0; j < nR && by_repeat[i]; ++j) {
            if (by_repeat[j] && repeat_reads(ctx, R[i], operand[i], R[j]))
               by_repeat[i] = false;
         }
      }

      /* Destinations are now in the register file */
      agx_foreach_ssa_dest(I, d) {
         insert_W(ctx, I->dest[d].value);
      }

      /* The M5 query entry's result may be unread; it then dies immediately
       * and must not occupy W or be spilled.
       */
      if (I->op == AGX_OPCODE_ENTER_QUERY_M5 && I->nr_dests == 1 &&
          ctx->next_uses[I->dest[0].value] == DIST_INFINITY)
         remove_W_if_present(ctx, I->dest[0].value);

      /* Update next-use distances for this instruction. Unlike the paper, we
       * prune dead values from W as we go. This doesn't affect correctness, but
       * it speeds up limit() on average.
       *
       * This happens after the above limit() calls to model sources as
       * late-kill. This is conservative and could be improved, but it matches
       * how we currently estimate register demand.
       */
      agx_foreach_ssa_src_rev(I, s) {
         assert(next_use_cursor >= 1);

         unsigned next_ip = next_ips[--next_use_cursor];
         assert((next_ip == DIST_INFINITY) == I->src[s].kill);

         if (I->src[s].kill) {
            remove_W_if_present(ctx, I->src[s].value);
         } else {
            ctx->next_uses[I->src[s].value] = next_ip;
         }
      }

      /* Add reloads for the sources in front of the instruction. We need to be
       * careful around exports, hoisting the reloads to before all exports.
       *
       * This is legal since all exports happen in parallel and all registers
       * are dead after the exports. The register file
       * must be big enough for everything exported, so it must be big enough
       * for all the reloaded values right before the parallel exports.
       */
      for (unsigned pass = 0; pass < 2; ++pass) {
         for (unsigned i = 0; i < nR; ++i) {
            if (by_repeat[i] == (pass == 1)) {
               insert_reload(ctx, ctx->block,
                             agx_before_instr_logical(ctx->block, I), R[i],
                             by_repeat[i], operand[i]);
            }
         }
      }

      ctx->ip += instr_cycles(I);
   }

   assert(next_use_cursor == 0 && "exactly sized");

   int i;
   BITSET_FOREACH_SET(i, ctx->W, ctx->n)
      sblock->W_exit[sblock->nW_exit++] = i;

   unsigned nS = __bitset_count(ctx->S, BITSET_WORDS(ctx->n));
   sblock->S_exit = ralloc_array(ctx->memctx, unsigned, nS);

   BITSET_FOREACH_SET(i, ctx->S, ctx->n)
      sblock->S_exit[sblock->nS_exit++] = i;

   assert(nS == sblock->nS_exit);
   util_dynarray_fini(&local_next_ip);
}

/*
 * Section 4.2 of the paper. A value live into a loop that nothing in the loop
 * reads only needs a register again after the loop. Keeping it in one at the
 * header pays off only if the loop never needs that register: once the body
 * runs out of registers it spills the value, and the back edge then has to
 * reload it into the register the header expects, so the value is spilled and
 * reloaded on every iteration instead of once around the loop.
 *
 * Structured control flow lays a loop out contiguously from its header to its
 * last back edge. Mark the values read in the loop and return the registers
 * the loop's own peak pressure leaves for the others, which are live through
 * the loop. Rematerializable values are evicted first, so they count only
 * where an instruction reads or writes them.
 */
static unsigned
loop_room_for_live_through(struct spill_ctx *ctx, BITSET_WORD *used)
{
   agx_block *header = ctx->block;
   unsigned last = header->index;
   agx_foreach_predecessor(header, pred) {
      last = MAX2(last, (*pred)->index);
   }

   agx_foreach_block_from(ctx->shader, header, block) {
      if (block->index > last)
         break;

      agx_foreach_instr_in_block(block, I) {
         agx_foreach_ssa_src(I, s) {
            /* Header phis read their entry sources before the loop */
            if (I->op == AGX_OPCODE_PHI && block == header) {
               agx_block **pred = util_dynarray_element(
                  &header->predecessors, agx_block *, s);
               if ((*pred)->index < header->index)
                  continue;
            }

            if (I->src[s].value < ctx->n)
               BITSET_SET(used, I->src[s].value);
         }
      }
   }

   BITSET_WORD *through = BITSET_CALLOC(ctx->n);
   U_SPARSE_BITSET_FOREACH_SET(&header->live_in, v) {
      if (v < ctx->n && !BITSET_TEST(used, v))
         BITSET_SET(through, v);
   }

   BITSET_WORD *live = BITSET_CALLOC(ctx->n);
   unsigned peak = 0;

   agx_foreach_block_from(ctx->shader, header, block) {
      if (block->index > last)
         break;

      memset(live, 0, BITSET_BYTES(ctx->n));
      unsigned count = 0;
      U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v) {
         if (v < ctx->n && !BITSET_TEST(through, v) && !ctx->remat[v]) {
            BITSET_SET(live, v);
            count += node_size(ctx, v);
         }
      }

      agx_foreach_instr_in_block_rev(block, I) {
         unsigned here = count;
         agx_foreach_ssa_dest(I, d) {
            const unsigned v = I->dest[d].value;
            if (v >= ctx->n)
               continue;

            if (BITSET_TEST(live, v)) {
               BITSET_CLEAR(live, v);
               count -= node_size(ctx, v);
            } else {
               here += node_size(ctx, v);
            }
         }

         if (I->op != AGX_OPCODE_PHI) {
            agx_foreach_ssa_src(I, s) {
               const unsigned v = I->src[s].value;
               if (v >= ctx->n || BITSET_TEST(live, v) ||
                   BITSET_TEST(through, v))
                  continue;

               here += node_size(ctx, v);
               if (!ctx->remat[v]) {
                  BITSET_SET(live, v);
                  count += node_size(ctx, v);
               }
            }
         }

         peak = MAX2(peak, here);
      }
   }

   free(live);
   free(through);
   return ctx->k > peak ? ctx->k - peak : 0;
}

/*
 * Hack's thesis sorts the live-in set (+ destinations of phis) by next-use
 * distance and takes as much as fits. On G17P and M5, values read in the loop
 * are taken first that way, and values live through the loop then fill only
 * the room the loop leaves them (section 4.2 of the paper). A remat-only pass
 * must not spill, so it keeps the thesis heuristic.
 */
static ATTRIBUTE_NOINLINE void
compute_w_entry_loop_header(struct spill_ctx *ctx)
{
   agx_block *block = ctx->block;
   struct spill_block *sb = spill_block(ctx, block);

   unsigned nP = u_sparse_bitset_count(&block->live_in);
   struct candidate *candidates = ctx->candidates;
   unsigned j = 0;

   foreach_next_use(&sb->next_use_in, i, dist) {
      assert(j < nP);
      candidates[j++] = (struct candidate){.node = i, .dist = dist};
   }

   assert(j == nP);

   /* Sort by next-use distance */
   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   BITSET_WORD *used = NULL;
   unsigned room = 0;
   if (!ctx->remat_only) {
      used = BITSET_CALLOC(ctx->n);
      room = loop_room_for_live_through(ctx, used);
   }

   /* Take as much as we can, values read in the loop first */
   unsigned through = 0;
   for (unsigned pass = 0; pass < (used ? 2 : 1); ++pass) {
      for (unsigned i = 0; i < j; ++i) {
         unsigned node = candidates[i].node;
         unsigned comps = node_size(ctx, node);

         if (used && BITSET_TEST(used, node) != (pass == 0))
            continue;

         if (used && pass == 1) {
            if (through + comps > room)
               continue;

            through += comps;
         }

         if ((ctx->nW + comps) <= ctx->k || keep_in_registers(ctx, node)) {
            insert_W(ctx, node);
            sb->W_entry[sb->nW_entry++] = node;
         }
      }
   }

   assert(ctx->nW <= ctx->k ||
          (ctx->remat_only && ctx->nW <= AGX_NUM_REGS));
   free(used);
}

/*
 * Compute W_entry for a block. Section 4.2 in the paper.
 */
static ATTRIBUTE_NOINLINE void
compute_w_entry(struct spill_ctx *ctx)
{
   agx_block *block = ctx->block;
   struct spill_block *sb = spill_block(ctx, block);

   /* Nothing to do for start blocks */
   if (agx_num_predecessors(block) == 0)
      return;

   /* Loop headers have a different heuristic */
   if (block->loop_header) {
      compute_w_entry_loop_header(ctx);
      return;
   }

   /* Usual blocks follow */
   unsigned *freq = ctx->freq;

   /* Record what's written at the end of each predecessor */
   agx_foreach_predecessor(ctx->block, P) {
      struct spill_block *sp = spill_block(ctx, *P);

      for (unsigned i = 0; i < sp->nW_exit; ++i) {
         unsigned v = sp->W_exit[i];
         freq[v]++;
      }
   }

   struct candidate *candidates = ctx->candidates;
   unsigned j = 0;

   /* Variables that are in all predecessors are assumed in W_entry. Phis and
    * variables in some predecessors are scored by next-use. W_entry lists
    * them all, so the coupling code knows every register live into the block.
    */
   foreach_next_use(&sb->next_use_in, i, dist) {
      if (freq[i] == agx_num_predecessors(ctx->block)) {
         insert_W(ctx, i);
         sb->W_entry[sb->nW_entry++] = i;
      } else if (freq[i]) {
         candidates[j++] = (struct candidate){.node = i, .dist = dist};
      }
   }

   agx_foreach_phi_in_block(ctx->block, I) {
      bool all_found = true;

      agx_foreach_predecessor(ctx->block, pred) {
         struct spill_block *sp = spill_block(ctx, *pred);
         bool found = false;

         agx_index src = I->src[agx_predecessor_index(ctx->block, *pred)];
         if (src.type != AGX_INDEX_NORMAL)
            continue;

         unsigned v = src.value;
         for (unsigned i = 0; i < sp->nW_exit; ++i) {
            if (sp->W_exit[i] == v) {
               found = true;
               break;
            }
         }

         all_found &= found;
      }

      const dist_t dist =
         search_next_uses(&sb->next_use_in, I->dest[0].value);

      /* Heuristic: if any phi source is spilled, spill the whole phi. This is
       * suboptimal, but it massively reduces pointless fill/spill chains with
       * massive phi webs. A remat-only pass never spills: the missing sources
       * are constants, rematerialized along their edges.
       *
       * A phi this block reads is the exception. Spilled, it is filled before
       * that read on every path into the block, and its sources still in
       * registers are spilled on their edges. In registers, only its spilled
       * sources are filled, each on its own edge, so that costs no more
       * memory traffic and frees the slot. Whether it fits is left to the
       * next-use order below, like any other candidate.
       */
      if (!all_found && !ctx->remat_only && dist >= sb->cycles)
         continue;

      candidates[j++] = (struct candidate){
         .node = I->dest[0].value,
         .dist = dist,
      };
   }

   /* Sort by next-use distance */
   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   /* Take as much as we can */
   for (unsigned i = 0; i < j; ++i) {
      unsigned node = candidates[i].node;
      unsigned comps = node_size(ctx, node);

      if ((ctx->nW + comps) <= ctx->k || keep_in_registers(ctx, node)) {
         insert_W(ctx, node);
         sb->W_entry[sb->nW_entry++] = node;
      }
   }

   assert((ctx->nW <= ctx->k ||
           (ctx->remat_only && ctx->nW <= AGX_NUM_REGS)) &&
          "invariant");

   /* Restore touched freq[] entries sparsely for the next block. */
   agx_foreach_predecessor(ctx->block, P) {
      struct spill_block *sp = spill_block(ctx, *P);

      for (unsigned i = 0; i < sp->nW_exit; ++i)
         freq[sp->W_exit[i]] = 0;
   }
}

/*
 * We initialize S with the union of S at the exit of (forward edge)
 * predecessors and the complement of W, intersected with the live-in set. The
 * former propagates S forward. The latter ensures we spill along the edge when
 * a live value is not selected for the entry W.
 */
static ATTRIBUTE_NOINLINE void
compute_s_entry(struct spill_ctx *ctx)
{
   unsigned v;

   agx_foreach_predecessor(ctx->block, pred) {
      struct spill_block *sp = spill_block(ctx, *pred);

      for (unsigned i = 0; i < sp->nS_exit; ++i) {
         v = sp->S_exit[i];

         if (u_sparse_bitset_test(&ctx->block->live_in, v))
            BITSET_SET(ctx->S, v);
      }
   }

   U_SPARSE_BITSET_FOREACH_SET(&ctx->block->live_in, v) {
      if (!BITSET_TEST(ctx->W, v))
         BITSET_SET(ctx->S, v);
   }

   /* Copy ctx->S to S_entry for later look-ups with coupling code */
   struct spill_block *sb = spill_block(ctx, ctx->block);
   unsigned nS = __bitset_count(ctx->S, BITSET_WORDS(ctx->n));
   sb->S_entry = ralloc_array(ctx->memctx, unsigned, nS);

   int i;
   BITSET_FOREACH_SET(i, ctx->S, ctx->n)
      sb->S_entry[sb->nS_entry++] = i;

   assert(sb->nS_entry == nS);
}

struct block_local_defs {
   uint32_t *defined;
   unsigned n_defined;
};

/* Braun & Hack Section 4.1: penalize loop-exit edges so spilling inside a hot
 * loop evicts outer/post-loop values ahead of loop-carried inner values. */
#define AGX_SPILL_LOOP_EXIT_PENALTY 10000u
#define AGX_SPILL_MAX_DIST (DIST_INFINITY / 2u)

static inline dist_t
dist_add_clamped(dist_t d, uint32_t penalty)
{
   if (d == DIST_INFINITY)
      return DIST_INFINITY;
   dist_t sum = dist_sum(d, penalty);
   return MIN2(sum, AGX_SPILL_MAX_DIST);
}

static ATTRIBUTE_NOINLINE void
global_next_use_distances(agx_context *ctx, void *memctx,
                          struct spill_block *blocks)
{
   u_worklist worklist;
   u_worklist_init(&worklist, ctx->num_blocks, NULL);

   /* Compute structured loop-nesting depth per block: for each loop header h,
    * find its furthest back-edge predecessor and count the loop once even when
    * multiple continue edges target the same header. */
   uint32_t *loop_depth = calloc(MAX2(ctx->num_blocks, 1), sizeof(*loop_depth));
   uint32_t *loop_end = calloc(MAX2(ctx->num_blocks, 1), sizeof(*loop_end));
   agx_foreach_block(ctx, block) {
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index &&
             block->index < ctx->num_blocks) {
            loop_end[succ->index] = MAX2(loop_end[succ->index],
                                         (uint32_t)block->index + 1u);
         }
      }
   }
   for (unsigned h = 0; h < ctx->num_blocks; ++h) {
      for (unsigned i = h; i < loop_end[h]; ++i)
         loop_depth[i]++;
   }
   free(loop_end);

   BITSET_WORD *defined = BITSET_CALLOC(ctx->alloc);
   BITSET_WORD *phi_dests = BITSET_CALLOC(ctx->alloc);
   struct block_local_defs *local_defs =
      calloc(ctx->num_blocks, sizeof(*local_defs));
   struct util_dynarray def_scratch = UTIL_DYNARRAY_INIT;

   /* Precompute each block's local first uses (seeded directly into
    * next_use_in) and defined variable list once before the worklist loop. */
   agx_foreach_block(ctx, block) {
      struct spill_block *sb = &blocks[block->index];

      init_next_uses(&sb->next_use_in, memctx);
      init_next_uses(&sb->next_use_out, memctx);

      util_dynarray_clear(&def_scratch);
      uint32_t cycle = 0;

      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_PHI) {
            cycle += instr_cycles(I);
            continue;
         }

         agx_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;
            assert(v < ctx->alloc);
            if (!BITSET_TEST(defined, v))
               set_next_use_min(&sb->next_use_in, v, cycle);
         }

         agx_foreach_ssa_dest(I, d) {
            uint32_t v = I->dest[d].value;
            assert(v < ctx->alloc);
            if (!BITSET_TEST(defined, v)) {
               BITSET_SET(defined, v);
               util_dynarray_append(&def_scratch, v);
            }
         }

         cycle += instr_cycles(I);
      }

      sb->cycles = cycle;
      unsigned n_def = util_dynarray_num_elements(&def_scratch, uint32_t);
      if (n_def) {
         local_defs[block->index].n_defined = n_def;
         local_defs[block->index].defined = malloc(n_def * sizeof(uint32_t));
         memcpy(local_defs[block->index].defined,
                util_dynarray_begin(&def_scratch), n_def * sizeof(uint32_t));
         for (unsigned i = 0; i < n_def; ++i)
            BITSET_CLEAR(defined, local_defs[block->index].defined[i]);
      }

      agx_worklist_push_head(&worklist, block);
   }

   util_dynarray_fini(&def_scratch);

   /* Iterate the work list in reverse order since liveness is backwards */
   while (!u_worklist_is_empty(&worklist)) {
      agx_block *blk = agx_worklist_pop_head(&worklist);
      struct spill_block *sb = &blocks[blk->index];
      const struct block_local_defs *ld = &local_defs[blk->index];

      for (unsigned i = 0; i < ld->n_defined; ++i)
         BITSET_SET(defined, ld->defined[i]);

      /* Apply transfer function: local first uses are already in next_use_in
       * with cycle < sb->cycles, so taking the minimum preserves them while
       * incorporating monotonically decreasing next_use_out distances. */
      foreach_next_use(&sb->next_use_out, node, dist) {
         if (!BITSET_TEST(defined, node)) {
            dist_t d = dist_sum(dist, sb->cycles);
            if (d != DIST_INFINITY)
               set_next_use_min(&sb->next_use_in, node, d);
         }
      }

      for (unsigned i = 0; i < ld->n_defined; ++i)
         BITSET_CLEAR(defined, ld->defined[i]);

      /* Propagate the live in of the successor (blk) to the live out of
       * predecessors.
       *
       * Phi nodes are logically on the control flow edge and act in parallel.
       * Skip phi destinations from next_use_in and make each predecessor's
       * corresponding phi source live with distance 0.
       */
      bool has_phis = false;
      agx_foreach_phi_in_block(blk, I) {
         assert(I->dest[0].type == AGX_INDEX_NORMAL);
         BITSET_SET(phi_dests, I->dest[0].value);
         has_phis = true;
      }

      agx_foreach_predecessor(blk, pred) {
         struct spill_block *sp = &blocks[(*pred)->index];
         bool progress = false;
         uint32_t exit_penalty = 0;
         if (loop_depth[(*pred)->index] > loop_depth[blk->index]) {
            const uint32_t delta =
               MIN2(loop_depth[(*pred)->index] - loop_depth[blk->index], 100u);
            exit_penalty = delta * AGX_SPILL_LOOP_EXIT_PENALTY;
         }

         if (has_phis) {
            foreach_next_use(&sb->next_use_in, node, dist) {
               if (!BITSET_TEST(phi_dests, node) &&
                   set_next_use_min(&sp->next_use_out, node,
                                    dist_add_clamped(dist, exit_penalty)))
                  progress = true;
            }

            unsigned pred_idx = agx_predecessor_index(blk, *pred);
            agx_foreach_phi_in_block(blk, I) {
               agx_index operand = I->src[pred_idx];
               if (operand.type == AGX_INDEX_NORMAL &&
                   set_next_use_min(&sp->next_use_out, operand.value,
                                    exit_penalty))
                  progress = true;
            }
         } else {
            foreach_next_use(&sb->next_use_in, node, dist) {
               if (set_next_use_min(&sp->next_use_out, node,
                                    dist_add_clamped(dist, exit_penalty)))
                  progress = true;
            }
         }

         if (progress)
            agx_worklist_push_tail(&worklist, *pred);
      }

      if (has_phis) {
         agx_foreach_phi_in_block(blk, I)
            BITSET_CLEAR(phi_dests, I->dest[0].value);
      }
   }

   for (unsigned i = 0; i < ctx->num_blocks; ++i)
      free(local_defs[i].defined);
   free(local_defs);
   free(loop_depth);
   free(phi_dests);
   free(defined);
   u_worklist_fini(&worklist);
}

static ATTRIBUTE_NOINLINE void
validate_next_use_info(UNUSED agx_context *ctx,
                       UNUSED struct spill_block *blocks)
{
#ifndef NDEBUG
   agx_foreach_block(ctx, blk) {
      struct spill_block *sb = &blocks[blk->index];

      /* Invariant: next-use distance is finite iff the node is live */
      U_SPARSE_BITSET_FOREACH_SET(&blk->live_in, i)
         assert(search_next_uses(&sb->next_use_in, i) < DIST_INFINITY);

      U_SPARSE_BITSET_FOREACH_SET(&blk->live_out, i)
         assert(search_next_uses(&sb->next_use_out, i) < DIST_INFINITY);

      foreach_next_use(&sb->next_use_in, i, _)
         assert(u_sparse_bitset_test(&blk->live_in, i));

      foreach_next_use(&sb->next_use_out, i, _)
         assert(u_sparse_bitset_test(&blk->live_out, i));
   }
#endif
}

/* SSA repair after rematerialization may merge different register copies of
 * one literal. Prove the entire scalar PHI/MOV web, including loop backedges,
 * before replacing a PHI. No partially-known input is assumed constant. */
struct constant_phi_edge {
   unsigned parent;
   int next;
};

static bool
constant_phi_scalar(agx_index x)
{
   return x.type == AGX_INDEX_NORMAL && !x.memory && !x.has_reg &&
          !x.abs && !x.neg && agx_channels(x) == 1 &&
          (x.size == AGX_SIZE_16 || x.size == AGX_SIZE_32);
}

static bool
constant_phi_transparent(const agx_instr *I)
{
   if (!I || I->nr_dests != 1 || !constant_phi_scalar(I->dest[0]) ||
       I->saturate || I->invert_cond)
      return false;

   if (I->op == AGX_OPCODE_MOV_IMM)
      return I->nr_srcs == 0;

   if ((I->op != AGX_OPCODE_MOV || I->nr_srcs != 1) &&
       (I->op != AGX_OPCODE_PHI || I->nr_srcs == 0))
      return false;

   agx_foreach_src(I, s) {
      if (!constant_phi_scalar(I->src[s]) ||
          I->src[s].size != I->dest[0].size)
         return false;
   }

   return true;
}

static bool
constant_phi_prove(agx_instr **defs, const int *parents,
                   const struct constant_phi_edge *edges, unsigned n,
                   unsigned start, BITSET_WORD *visited, BITSET_WORD *grounded,
                   unsigned *nodes, unsigned *work, uint64_t *literal)
{
   memset(visited, 0, BITSET_BYTES(n));
   memset(grounded, 0, BITSET_BYTES(n));
   unsigned count = 1, read = 0, roots = 0;
   bool have_literal = false;
   enum agx_size size = defs[start]->dest[0].size;
   nodes[0] = start;
   BITSET_SET(visited, start);

   while (read < count) {
      unsigned v = nodes[read++];
      const agx_instr *I = defs[v];
      if (!constant_phi_transparent(I) || I->dest[0].size != size)
         return false;

      if (I->op == AGX_OPCODE_MOV_IMM) {
         uint64_t value = I->imm &
            (size == AGX_SIZE_16 ? UINT64_C(0xffff) : UINT64_C(0xffffffff));
         if (have_literal && value != *literal)
            return false;
         have_literal = true;
         *literal = value;
         BITSET_SET(grounded, v);
         work[roots++] = v;
         continue;
      }

      agx_foreach_src(I, s) {
         unsigned src = I->src[s].value;
         if (src >= n || !defs[src])
            return false;
         if (!BITSET_TEST(visited, src)) {
            BITSET_SET(visited, src);
            nodes[count++] = src;
         }
      }
   }

   if (!have_literal)
      return false;

   /* Every node must have a path to a literal. Otherwise a root-free cycle
    * could have contributed an unknown input beside a known literal. Traverse
    * reverse edges from the actual literal definitions, rather than guessing
    * that an unresolved PHI input agrees with an already-known one. */
   unsigned proven = roots;
   read = 0;
   while (read < proven) {
      unsigned v = work[read++];
      for (int e = parents[v]; e >= 0; e = edges[e].next) {
         unsigned parent = edges[e].parent;
         if (BITSET_TEST(visited, parent) &&
             !BITSET_TEST(grounded, parent)) {
            BITSET_SET(grounded, parent);
            work[proven++] = parent;
         }
      }
   }

   return proven == count;
}

bool
agx_fold_agx3_spill_constant_phis(agx_context *ctx)
{
   if (!ctx->key || !agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->is_preamble || !ctx->alloc)
      return false;

   void *memctx = ralloc_context(NULL);
   unsigned n = ctx->alloc;
   agx_instr **defs = rzalloc_array(memctx, agx_instr *, n);
   int *parents = ralloc_array(memctx, int, n);
   memset(parents, 0xff, n * sizeof(*parents));
   struct util_dynarray edges;
   util_dynarray_init(&edges, memctx);

   agx_foreach_instr_global(ctx, I) {
      if (I->nr_dests == 1 && I->dest[0].type == AGX_INDEX_NORMAL &&
          !I->dest[0].memory && !I->dest[0].has_reg)
         defs[I->dest[0].value] = I;

      if (!constant_phi_transparent(I) || I->op == AGX_OPCODE_MOV_IMM)
         continue;

      agx_foreach_src(I, s) {
         unsigned src = I->src[s].value;
         if (src >= n)
            continue;
         struct constant_phi_edge edge = {
            .parent = I->dest[0].value, .next = parents[src],
         };
         parents[src] = util_dynarray_num_elements(&edges, struct constant_phi_edge);
         util_dynarray_append(&edges, edge);
      }
   }

   BITSET_WORD *visited = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   BITSET_WORD *grounded = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   unsigned *nodes = ralloc_array(memctx, unsigned, n);
   unsigned *work = ralloc_array(memctx, unsigned, n);
   bool progress = false;

   agx_foreach_block(ctx, block) {
      /* Preserve the ELSE/PHI prefix: literals execute after all surviving
       * parallel PHIs, never between them. */
      agx_cursor cursor = agx_after_block(block);
      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_PHI && I->op != AGX_OPCODE_ELSE_ICMP &&
             I->op != AGX_OPCODE_ELSE_FCMP) {
            cursor = agx_before_instr(I);
            break;
         }
      }
      agx_builder b = agx_init_builder(ctx, cursor);
      agx_foreach_phi_in_block_safe(block, phi) {
         uint64_t literal = 0;
         if (constant_phi_transparent(phi) &&
             constant_phi_prove(defs, parents, util_dynarray_begin(&edges), n,
                                phi->dest[0].value, visited, grounded,
                                nodes, work, &literal)) {
            agx_mov_imm_to(&b, phi->dest[0], literal);
            agx_remove_instruction(phi);
            progress = true;
         }
      }
   }

   ralloc_free(memctx);
   return progress;
}

/*
 * Reachability-based dead code elimination after spilling.
 *
 * Unlike agx_dce(), this removes dead loop-header PHIs and dead PHI cycles.
 * Spilling with repeatable ALU/load rematerialization or constant-PHI folding
 * can eliminate every fill of a spilled value, making its spill store dead and
 * leaving a loop-header PHI inserted by agx_repair_ssa() with zero uses. Such
 * a dead loop-header PHI is excluded from liveness and demand accounting, yet
 * RA still assigns it a register that is never killed, causing register leaks
 * and spurious RA failures.
 */
static void
agx_spill_dce(agx_context *ctx)
{
   if (!ctx->alloc)
      return;

   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   uint32_t *worklist = malloc(ctx->alloc * sizeof(*worklist));
   BITSET_WORD *live = BITSET_CALLOC(ctx->alloc);
   unsigned head = 0, tail = 0;

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         defs[I->dest[d].value] = I;
      }
   }

   agx_foreach_instr_global(ctx, I) {
      const bool rt_completion =
         (I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3) &&
         I->scs_rt_completion;
      if (agx_opcodes_info[I->op].can_eliminate &&
          !I->agx3_tex_coord_stage && !rt_completion)
         continue;

      agx_foreach_ssa_src(I, s) {
         unsigned v = I->src[s].value;
         if (!BITSET_TEST(live, v)) {
            BITSET_SET(live, v);
            worklist[tail++] = v;
         }
      }
   }

   while (head < tail) {
      unsigned v = worklist[head++];
      agx_instr *I = defs[v];
      if (!I)
         continue;

      agx_foreach_ssa_dest(I, d) {
         defs[I->dest[d].value] = NULL;
      }

      agx_foreach_ssa_src(I, s) {
         unsigned src = I->src[s].value;
         if (!BITSET_TEST(live, src)) {
            BITSET_SET(live, src);
            worklist[tail++] = src;
         }
      }
   }

   agx_foreach_instr_global_safe(ctx, I) {
      const bool rt_completion =
         (I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
          I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3) &&
         I->scs_rt_completion;
      if (!agx_opcodes_info[I->op].can_eliminate ||
          I->agx3_tex_coord_stage || rt_completion)
         continue;

      bool needed = false;
      agx_foreach_ssa_dest(I, d) {
         if (BITSET_TEST(live, I->dest[d].value)) {
            needed = true;
            break;
         }
      }

      if (!needed)
         agx_remove_instruction(I);
   }

   free(defs);
   free(worklist);
   free(live);
}

void
agx_spill(agx_context *ctx, unsigned k, bool remat_only)
{
   ctx->uniforms_written = agx_has_uniform_store(ctx);
   void *memctx = ralloc_context(NULL);

   /* Spilling needs extra registers for memory-memory swaps. Rematerializing
    * inserts none, but must still leave room for the registers RA reserves at
    * the bottom of the file, which is four for an AGX3 divergent shuffle.
    */
   k -= remat_only ? MAX2(agx_reserved_registers(ctx), 2)
                   : MAX2(agx_reserved_registers(ctx), 8);

   uint8_t *channels = rzalloc_array(memctx, uint8_t, ctx->alloc);
   dist_t *next_uses = rzalloc_array(memctx, dist_t, ctx->alloc);
   enum agx_size *sizes = rzalloc_array(memctx, enum agx_size, ctx->alloc);
   agx_instr **remat = rzalloc_array(memctx, agx_instr *, ctx->alloc);
   agx_instr **repeat = rzalloc_array(memctx, agx_instr *, ctx->alloc);
   uint8_t *repeat_comp = rzalloc_array(memctx, uint8_t, ctx->alloc);

   agx_foreach_instr_global(ctx, I) {
      if (can_remat(ctx, I))
         remat[I->dest[0].value] = I;

      /* Definitions precede uses in block order, so a split of a load sees
       * the load first.
       */
      if (can_remat_load(ctx, I) || can_repeat_alu(ctx, I)) {
         repeat[I->dest[0].value] = I;
      } else if (I->op == AGX_OPCODE_SPLIT &&
                 I->src[0].type == AGX_INDEX_NORMAL && !I->src[0].memory &&
                 repeat[I->src[0].value] &&
                 repeat[I->src[0].value]->op == AGX_OPCODE_DEVICE_LOAD &&
                 repeat[I->src[0].value]->dest[0].value == I->src[0].value) {
         agx_foreach_ssa_dest(I, d) {
            if (I->dest[d].size == I->src[0].size &&
                agx_channels(I->dest[d]) == 1) {
               repeat[I->dest[d].value] = repeat[I->src[0].value];
               repeat_comp[I->dest[d].value] = d;
            }
         }
      }

      /* Measure vectors */
      agx_foreach_ssa_dest(I, d) {
         assert(sizes[I->dest[d].value] == 0 && "broken SSA");
         assert(channels[I->dest[d].value] == 0 && "broken SSA");

         sizes[I->dest[d].value] = I->dest[d].size;
         channels[I->dest[d].value] = agx_channels(I->dest[d]);
      }
   }

   struct spill_block *blocks =
      rzalloc_array(memctx, struct spill_block, ctx->num_blocks);

   /* Step 1. Compute global next-use distances */
   global_next_use_distances(ctx, memctx, blocks);
   validate_next_use_info(ctx, blocks);

   /* Reserve a memory variable for every regular variable */
   unsigned n = ctx->alloc;
   ctx->alloc *= 2;

   BITSET_WORD *W = ralloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   BITSET_WORD *S = ralloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   dist_t *local_nu = ralloc_array(memctx, dist_t, n);
   memset(local_nu, 0xff, n * sizeof(dist_t));
   unsigned *freq = rzalloc_array(memctx, unsigned, n);
   struct candidate *candidates = ralloc_array(memctx, struct candidate, n);

   agx_foreach_block(ctx, block) {
      memset(W, 0, BITSET_BYTES(n));
      memset(S, 0, BITSET_BYTES(n));

      struct spill_ctx sctx = {
         .memctx = memctx,
         .shader = ctx,
         .n = n,
         .channels = channels,
         .size = sizes,
         .remat = remat,
         .repeat = repeat,
         .repeat_comp = repeat_comp,
         .next_uses = next_uses,
         .local_nu = local_nu,
         .freq = freq,
         .candidates = candidates,
         .block = block,
         .blocks = blocks,
         .k = k,
         .W = W,
         .S = S,
         .spill_base = n,
         .remat_only = remat_only,
      };

      compute_w_entry(&sctx);
      compute_s_entry(&sctx);
      min_algorithm(&sctx);
   }

   /* Now that all blocks are processed separately, stitch it together */
   memset(W, 0, BITSET_BYTES(n));
   BITSET_WORD *pred_W = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   BITSET_WORD *pred_S = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   BITSET_WORD *phi_any_dests =
      rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   BITSET_WORD *phi_raw_dests =
      rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));

   agx_foreach_block(ctx, block) {
      struct spill_ctx sctx = {
         .memctx = memctx,
         .shader = ctx,
         .n = n,
         .channels = channels,
         .size = sizes,
         .remat = remat,
         .repeat = repeat,
         .repeat_comp = repeat_comp,
         .pred_W = pred_W,
         .pred_S = pred_S,
         .phi_any_dests = phi_any_dests,
         .phi_raw_dests = phi_raw_dests,
         .block = block,
         .blocks = blocks,
         .k = k,
         .W = W,
         .S = S,
         .spill_base = n,
         .remat_only = remat_only,
      };

      agx_foreach_predecessor(block, pred) {
         /* After spilling phi sources, insert coupling code */
         insert_coupling_code(&sctx, *pred, block);
      }
   }

   ralloc_free(memctx);

   /* Spilling breaks SSA, so we need to repair before validating */
   agx_repair_ssa(ctx);
   agx_fold_agx3_spill_constant_phis(ctx);

   /* Remat and constant-PHI folding can leave dead spill stores and dead
    * loop-header PHIs whose backedge reloads are also dead. */
   agx_spill_dce(ctx);
   agx_validate(ctx, "Spilling");

   /* Remat can introduce dead code. When every reload of a spilled split
    * component repeats its device load, the spill store is dead, and with it
    * the only read of that split destination. The spiller stopped counting
    * the component at the store, but RA counts an unread destination until
    * the end of its block, so null unread destinations as the partial DCE
    * ahead of RA does, or the program can exceed the spill budget.
    */
   agx_dce(ctx, true);

   /* Any memory value that survives spilling is a stack slot: RA must keep
    * its parallel-copy scratch registers to lower memory copies after
    * allocation, and a remat-only pass that left one behind has spilled to
    * scratch. record_spill() sets both flags at each store this pass inserts
    * (cdfdf3baf3c added the phi-edge ones), so this normally changes nothing;
    * it makes the RA contract follow the program itself, not the bookkeeping
    * of every path that can create a memory value. That is generic register
    * allocation logic, so it holds on every generation.
    */
   bool memory = false;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         memory |= I->dest[d].memory;
      agx_foreach_ssa_src(I, s)
         memory |= I->src[s].memory;
   }

   if (memory) {
      if (!ctx->key->has_scratch)
         ctx->ra_failed = true;
      ctx->has_spill_pcopy_reserved = true;
      ctx->remat_fallback_spilled |= remat_only;
   }
}
