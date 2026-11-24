/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure */

#include <stdio.h>
#include <string.h>
#include "util/bitset.h"
#include "util/dag.h"
#include "agx_debug.h"
#include "util/sparse_bitset.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx3_fixed_dependencies.h"
#include "agx_opcodes.h"

struct sched_use_entry {
   struct sched_node *node;
   int32_t next;
   uint8_t size16;
};

struct sched_ctx {
   /* Dependency graph */
   struct dag *dag;

   /* Dense live set for the current block */
   BITSET_WORD *live;

   /* Incremental pressure delta tables indexed by SSA value */
   struct sched_node **def_node;
   uint8_t *def_size;
   int32_t *use_head;
   struct util_dynarray *use_pool;
   unsigned touched_start;
   unsigned touched_end;
};

struct sched_node {
   struct dag_node dag;

   /* Instruction this node represents */
   agx_instr *instr;

   /* Original position within the block (ILP scheduler only) */
   unsigned index;

   /* Cached pressure delta for the current live set */
   int32_t delta;
};

static void
add_dep(struct sched_node *a, struct sched_node *b)
{
   assert(a != b && "no self-dependencies");

   if (a && b)
      dag_add_edge(&a->dag, &b->dag, 0);
}

static void
serialize(struct sched_node *a, struct sched_node **b)
{
   add_dep(a, *b);
   *b = a;
}

static bool
native_uniform_write(const agx_instr *I)
{
   return I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
          I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
          I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3;
}

static void
add_fixed_dependency(void *data, uint32_t before, uint32_t after)
{
   struct sched_node **nodes = data;
   /* The pressure scheduler walks dependents before their dependencies. */
   add_dep(nodes[after], nodes[before]);
}

static bool
uniform_access(struct util_dynarray *accesses, unsigned instruction,
               unsigned first, unsigned count, bool write)
{
   if (!count || first >= AGX_NUM_UNIFORMS ||
       count > AGX_NUM_UNIFORMS - first)
      return false;
   struct agx3_fixed_access access = {
      .instruction = instruction, .first = first, .count = count, .write = write,
   };
   util_dynarray_append(accesses, access);
   return true;
}

static bool
add_uniform_dependencies(struct util_dynarray *nodes, void *memctx)
{
   struct util_dynarray accesses;
   util_dynarray_init(&accesses, memctx);
   unsigned instruction = 0;
   util_dynarray_foreach(nodes, struct sched_node *, node) {
      const agx_instr *I = (*node)->instr;
      agx_foreach_src(I, s) {
         const agx_index src = I->src[s];
         if (src.type == AGX_INDEX_UNIFORM &&
             !uniform_access(&accesses, instruction, src.value,
                              agx_index_size_16(src), false))
            return false;
      }
      if (native_uniform_write(I)) {
         if (I->imm > AGX_NUM_UNIFORMS - 2 ||
             !uniform_access(&accesses, instruction, I->imm, 2, true))
            return false;
      } else if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         if (I->nr_srcs != 2 || I->src[1].type != AGX_INDEX_IMMEDIATE)
            return false;
         const unsigned stride = agx_size_align_16(I->src[0].size);
         const unsigned base = I->src[1].value;
         if (base >= AGX_NUM_UNIFORMS || I->mask > 15)
            return false;
         for (unsigned c = 0; c < 4; ++c) {
            if ((I->mask & (1u << c)) &&
                !uniform_access(&accesses, instruction, base + c * stride,
                                  stride, true))
               return false;
         }
      }
      ++instruction;
   }
   return agx3_fixed_dependencies(accesses.data,
      util_dynarray_num_elements(&accesses, struct agx3_fixed_access),
      AGX_NUM_UNIFORMS, add_fixed_dependency, nodes->data);
}

struct sched_dag_state {
   struct sched_node **last_write;
   unsigned start;
   unsigned end;
};

enum sched_mem_space {
   SCHED_MEM_DEVICE = 0,
   SCHED_MEM_LOCAL = 1,
   SCHED_MEM_STACK = 2,
   SCHED_MEM_NUM = 3,
};

static enum sched_mem_space
instr_mem_space(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
   case AGX_OPCODE_LOCAL_ATOMIC:
      return SCHED_MEM_LOCAL;
   case AGX_OPCODE_STACK_ADJUST:
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
      return SCHED_MEM_STACK;
   default:
      return SCHED_MEM_DEVICE;
   }
}

static struct dag *
create_dag(agx_context *ctx, agx_block *block, void *memctx, bool native_uniforms,
           struct util_dynarray *all_nodes, struct sched_dag_state *state,
           bool reorder_loads)
{
   struct dag *dag = dag_create(ctx);
   struct util_dynarray nodes;
   util_dynarray_init(&nodes, memctx);

   if (!state->last_write) {
      state->last_write = calloc(ctx->alloc, sizeof(*state->last_write));
   } else if (state->start < state->end) {
      memset(state->last_write + state->start, 0,
             (state->end - state->start) * sizeof(*state->last_write));
   }
   state->start = ctx->alloc;
   state->end = 0;
   struct sched_node **last_write = state->last_write;
   struct sched_node *coverage = NULL;
   struct sched_node *preload = NULL;

   /* Last memory load/store per disjoint address space (device, threadgroup
    * local, private stack).  Global device memory, on-chip local memory, and
    * per-thread stack cannot alias each other, so a local_store must not
    * serialize an independent device_load (and vice versa).  Barriers
    * serialize all spaces. */
   struct sched_node *memory_load[SCHED_MEM_NUM] = {NULL};
   struct sched_node *memory_store[SCHED_MEM_NUM] = {NULL};

   /* Last unified-vertex-store write, to keep UVS writeout in program order.
    *
    * AGX3's UVSW is not a plain indexed write.  agx3_pack_binary.c copies
    * each value into the staging file first and then emits the UVSW against
    * it, and outside Apple's batched protocols the staging half and the
    * scoreboard slot a store uses are picked from the store's position in
    * the emission order (`next_uvs_slot` round-robin, `staging_half =
    * 2 * (stage_slot - 1)`).  That fallback is what an unlinked vertex
    * shader takes, because a run-time output index disqualifies every
    * batched form.  Permuting the stores therefore permutes which of them
    * share a staging generation, and the packer's slot bookkeeping does not
    * model those generations, so the resulting hazards are invisible to it:
    * on G17P this rendered a random subset of a tessellated+geometry draw's
    * primitives per run.  st_vary carries no schedule class, so nothing else
    * stops the reorder.  The stores sit at the end of the shader and each
    * one already depends on its own value, so ordering them among themselves
    * costs the scheduler essentially nothing (measured: identical
    * instruction, GPR and spill counts; 10 bytes smaller code on the shader
    * that exposed the bug).  AGX2 addresses the UVS directly and is left
    * alone.
    */
   struct sched_node *vary_store = NULL;
   const enum agx_isa_encoding enc = ctx->key->dev.encoding;
   const bool order_vary_stores =
      agx_encoding_is_agx3(enc);

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      dag_init_node(dag, &node->dag);
      if (native_uniforms)
         util_dynarray_append(&nodes, node);
      if (all_nodes) {
         node->index = util_dynarray_num_elements(all_nodes,
                                                  struct sched_node *);
         util_dynarray_append(all_nodes, node);
      }

      /* Reads depend on writes, no other hazards in SSA */
      agx_foreach_ssa_src(I, s) {
         add_dep(node, last_write[I->src[s].value]);
      }

      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         const unsigned dest = I->dest[d].value;
         last_write[dest] = node;
         state->start = MIN2(state->start, dest);
         state->end = MAX2(state->end, dest + 1);
      }

      /* Classify the instruction and add dependencies according to the class */
      enum agx_schedule_class dep = agx_opcodes_info[I->op].schedule_class;
      assert(dep != AGX_SCHEDULE_CLASS_INVALID && "invalid instruction seen");

      /* No store of the shader can write a reorderable load's memory
       * (memory_can_reorder), so the copy schedule may drop its memory-order
       * edges, exactly like texture_load's. Other schedules keep them: the
       * bottom-up pressure order would otherwise sink such loads next to
       * their uses, shortening the latency they already hide. */
      if (reorder_loads && I->op == AGX_OPCODE_IMAGE_LOAD &&
          I->memory_can_reorder)
         dep = AGX_SCHEDULE_CLASS_NONE;

      bool barrier = dep == AGX_SCHEDULE_CLASS_BARRIER;
      bool discards =
         I->op == AGX_OPCODE_SAMPLE_MASK || I->op == AGX_OPCODE_ZS_EMIT ||
         I->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
         I->op == AGX_OPCODE_ZS_EMIT_AGX3;

      if (barrier) {
         for (unsigned sp = 0; sp < SCHED_MEM_NUM; ++sp) {
            serialize(node, &memory_load[sp]);
            serialize(node, &memory_store[sp]);
         }
      } else {
         const enum sched_mem_space sp = instr_mem_space(I);

         if (dep == AGX_SCHEDULE_CLASS_STORE)
            add_dep(node, memory_load[sp]);
         else if (dep == AGX_SCHEDULE_CLASS_ATOMIC)
            serialize(node, &memory_load[sp]);

         if (dep == AGX_SCHEDULE_CLASS_LOAD ||
             dep == AGX_SCHEDULE_CLASS_STORE ||
             dep == AGX_SCHEDULE_CLASS_ATOMIC)
            serialize(node, &memory_store[sp]);
      }

      if (dep == AGX_SCHEDULE_CLASS_COVERAGE || barrier)
         serialize(node, &coverage);

      /* Make sure externally visible side effects happen before a discard */
      if (discards)
         add_dep(node, memory_store[SCHED_MEM_DEVICE]);

      if (order_vary_stores && I->op == AGX_OPCODE_ST_VARY)
         serialize(node, &vary_store);

      if (dep == AGX_SCHEDULE_CLASS_PRELOAD)
         serialize(node, &preload);
      else
         add_dep(node, preload);
   }

   if (native_uniforms && !add_uniform_dependencies(&nodes, memctx))
      return NULL;
   return dag;
}

/*
 * Calculate the change in register pressure from scheduling a given
 * instruction. Equivalently, calculate the difference in the number of live
 * registers before and after the instruction, given the live set after the
 * instruction. This calculation follows immediately from the dataflow
 * definition of liveness:
 *
 *      live_in = (live_out - KILL) + GEN
 */
static signed
calculate_pressure_delta(const agx_instr *I, struct u_sparse_bitset *live)
{
   signed delta = 0;

   /* Destinations must be unique */
   agx_foreach_ssa_dest(I, d) {
      if (u_sparse_bitset_test(live, I->dest[d].value))
         delta -= agx_index_size_16(I->dest[d]);
   }

   agx_foreach_ssa_src(I, src) {
      /* Filter duplicates */
      bool dupe = false;

      for (unsigned i = 0; i < src; ++i) {
         if (agx_is_equiv(I->src[i], I->src[src])) {
            dupe = true;
            break;
         }
      }

      if (!dupe && !u_sparse_bitset_test(live, I->src[src].value))
         delta += agx_index_size_16(I->src[src]);
   }

   return delta;
}

static signed
calculate_pressure_delta_dense(const agx_instr *I, const BITSET_WORD *live)
{
   signed delta = 0;

   /* Destinations must be unique */
   agx_foreach_ssa_dest(I, d) {
      if (BITSET_TEST(live, I->dest[d].value))
         delta -= agx_index_size_16(I->dest[d]);
   }

   agx_foreach_ssa_src(I, src) {
      /* Filter duplicates */
      bool dupe = false;

      for (unsigned i = 0; i < src; ++i) {
         if (agx_is_equiv(I->src[i], I->src[src])) {
            dupe = true;
            break;
         }
      }

      if (!dupe && !BITSET_TEST(live, I->src[src].value))
         delta += agx_index_size_16(I->src[src]);
   }

   return delta;
}

static inline void
sched_liveness_update_dense(BITSET_WORD *live, const agx_instr *I)
{
   agx_foreach_ssa_dest(I, d)
      BITSET_CLEAR(live, I->dest[d].value);

   agx_foreach_ssa_src(I, s)
      BITSET_SET(live, I->src[s].value);
}

static inline void
sched_adjust_deltas(struct sched_ctx *s, unsigned v, int32_t sign)
{
   struct sched_node *def = s->def_node[v];
   if (def)
      def->delta += sign * (int32_t)s->def_size[v];

   const struct sched_use_entry *entries = s->use_pool->data;
   for (int32_t idx = s->use_head[v]; idx >= 0; idx = entries[idx].next)
      entries[idx].node->delta += sign * (int32_t)entries[idx].size16;
}

static inline void
sched_apply_instr(struct sched_ctx *s, const agx_instr *I)
{
   agx_foreach_ssa_dest(I, d) {
      const unsigned v = I->dest[d].value;
      if (BITSET_TEST(s->live, v)) {
         BITSET_CLEAR(s->live, v);
         sched_adjust_deltas(s, v, +1);
      }
   }

   agx_foreach_ssa_src(I, src) {
      const unsigned v = I->src[src].value;
      if (!BITSET_TEST(s->live, v)) {
         BITSET_SET(s->live, v);
         sched_adjust_deltas(s, v, -1);
      }
   }
}

/*
 * Choose the next instruction, bottom-up. For now we use a simple greedy
 * heuristic: choose the instruction that has the best effect on liveness, while
 * hoisting sample_mask.
 */
static struct sched_node *
choose_instr(struct sched_ctx *s)
{
   int32_t min_delta = INT32_MAX;
   struct sched_node *best = NULL;

   list_for_each_entry(struct sched_node, n, &s->dag->heads, dag.link) {
      /* Heuristic: hoist sample_mask/zs_emit. This allows depth/stencil tests
       * to run earlier, and potentially to discard the entire quad invocation
       * earlier, reducing how much redundant fragment shader we run.
       *
       * Since we schedule backwards, we make that happen by only choosing
       * sample_mask when all other instructions have been exhausted.
       */
      if (n->instr->op == AGX_OPCODE_SAMPLE_MASK ||
          n->instr->op == AGX_OPCODE_SAMPLE_MASK_AGX3 ||
          n->instr->op == AGX_OPCODE_ZS_EMIT ||
          n->instr->op == AGX_OPCODE_ZS_EMIT_AGX3) {

         if (!best) {
            best = n;
            assert(min_delta == INT32_MAX);
         }

         continue;
      }

      /* Heuristic: sink wait_pix to increase parallelism. Since wait_pix does
       * not read or write registers, this has no effect on pressure.
       */
      if (n->instr->op == AGX_OPCODE_WAIT_PIX)
         return n;

      assert(n->delta == calculate_pressure_delta_dense(n->instr, s->live));
      int32_t delta = n->delta;

      if (delta < min_delta) {
         best = n;
         min_delta = delta;
      }
   }

   return best;
}

static void
pressure_schedule_block(agx_context *ctx, agx_block *block, struct sched_ctx *s,
                        struct util_dynarray *all_nodes)
{
   const unsigned nr_dag =
      util_dynarray_num_elements(all_nodes, struct sched_node *);
   if (nr_dag <= 1)
      return;

   /* off by a constant, that's ok */
   signed pressure = 0;
   signed orig_max_pressure = 0;

   memset(s->live, 0, BITSET_BYTES(ctx->alloc));
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v)
      BITSET_SET(s->live, v);

   agx_foreach_instr_in_block_rev(block, I) {
      pressure += calculate_pressure_delta_dense(I, s->live);
      orig_max_pressure = MAX2(pressure, orig_max_pressure);
      sched_liveness_update_dense(s->live, I);
   }

   memset(s->live, 0, BITSET_BYTES(ctx->alloc));
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v)
      BITSET_SET(s->live, v);

   /* Initialize per-node deltas and incremental def/use tables */
   if (s->touched_start < s->touched_end) {
      const unsigned span = s->touched_end - s->touched_start;
      memset(s->def_node + s->touched_start, 0, span * sizeof(*s->def_node));
      memset(s->use_head + s->touched_start, -1, span * sizeof(*s->use_head));
   }
   s->touched_start = ctx->alloc;
   s->touched_end = 0;
   util_dynarray_clear(s->use_pool);

   util_dynarray_foreach(all_nodes, struct sched_node *, np) {
      struct sched_node *node = *np;
      const agx_instr *I = node->instr;
      int32_t delta = 0;

      agx_foreach_ssa_dest(I, d) {
         const unsigned v = I->dest[d].value;
         const uint8_t sz = agx_index_size_16(I->dest[d]);
         s->def_node[v] = node;
         s->def_size[v] = sz;
         s->touched_start = MIN2(s->touched_start, v);
         s->touched_end = MAX2(s->touched_end, v + 1);
         if (BITSET_TEST(s->live, v))
            delta -= sz;
      }

      agx_foreach_ssa_src(I, src) {
         bool dupe = false;
         for (unsigned i = 0; i < src; ++i) {
            if (agx_is_equiv(I->src[i], I->src[src])) {
               dupe = true;
               break;
            }
         }
         if (dupe)
            continue;

         const unsigned v = I->src[src].value;
         const uint8_t sz = agx_index_size_16(I->src[src]);
         const int32_t entry_idx =
            util_dynarray_num_elements(s->use_pool, struct sched_use_entry);
         struct sched_use_entry entry = {
            .node = node,
            .next = s->use_head[v],
            .size16 = sz,
         };
         util_dynarray_append(s->use_pool, entry);
         s->use_head[v] = entry_idx;
         s->touched_start = MIN2(s->touched_start, v);
         s->touched_end = MAX2(s->touched_end, v + 1);
         if (!BITSET_TEST(s->live, v))
            delta += sz;
      }

      node->delta = delta;
   }

   /* off by a constant, that's ok */
   signed max_pressure = 0;
   pressure = 0;

   struct sched_node **schedule = malloc(nr_dag * sizeof(struct sched_node *));
   unsigned nr_ins = 0;

   while (!list_is_empty(&s->dag->heads)) {
      struct sched_node *node = choose_instr(s);
      assert(node->delta == calculate_pressure_delta_dense(node->instr, s->live));
      pressure += node->delta;
      max_pressure = MAX2(pressure, max_pressure);
      dag_prune_head(s->dag, &node->dag);

      schedule[nr_ins++] = node;
      sched_apply_instr(s, node->instr);
   }

   /* Bail if it looks like it's worse */
   if (max_pressure >= orig_max_pressure) {
      free(schedule);
      return;
   }

   /* Apply the schedule */
   for (unsigned i = 0; i < nr_ins; ++i) {
      agx_remove_instruction(schedule[i]->instr);
      list_add(&schedule[i]->instr->link, &block->instructions);
   }

   free(schedule);
}

void
agx_pressure_schedule(agx_context *ctx)
{
   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);
   struct sched_dag_state dag_state = {0};

   bool native_uniforms = false;
   agx_foreach_instr_global(ctx, I) {
      native_uniforms |= native_uniform_write(I) ||
                         I->op == AGX_OPCODE_UNIFORM_STORE;
   }

   BITSET_DECLARE(stack_live, 4096);
   BITSET_WORD *live =
      ctx->alloc <= 4096 ? stack_live : BITSET_CALLOC(ctx->alloc);
   struct sched_node **def_node = calloc(ctx->alloc, sizeof(*def_node));
   uint8_t *def_size = malloc(ctx->alloc * sizeof(*def_size));
   int32_t *use_head = malloc(ctx->alloc * sizeof(*use_head));
   memset(use_head, -1, ctx->alloc * sizeof(*use_head));
   struct util_dynarray use_pool = UTIL_DYNARRAY_INIT;
   struct util_dynarray all_nodes = UTIL_DYNARRAY_INIT;

   struct sched_ctx sctx = {
      .live = live,
      .def_node = def_node,
      .def_size = def_size,
      .use_head = use_head,
      .use_pool = &use_pool,
      .touched_start = ctx->alloc,
      .touched_end = 0,
   };

   agx_foreach_block(ctx, block) {
      if (list_is_empty(&block->instructions) ||
          list_is_singular(&block->instructions))
         continue;

      util_dynarray_clear(&all_nodes);
      sctx.dag = create_dag(ctx, block, memctx, native_uniforms, &all_nodes,
                            &dag_state, false);
      /* Preserve the original order if a complete dependency graph cannot be
       * built (invalid range or workspace allocation failure). */
      if (!sctx.dag)
         continue;

      pressure_schedule_block(ctx, block, &sctx, &all_nodes);
   }

   /* Clean up after liveness analysis */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   util_dynarray_fini(&all_nodes);
   util_dynarray_fini(&use_pool);
   free(use_head);
   free(def_size);
   free(def_node);
   if (live != stack_live)
      free(live);
   free(dag_state.last_write);
   ralloc_free(memctx);
}

/*
 * Latency list scheduling for G17P.
 *
 * Measured on Neo, Mesa's ALU-bound loops (compute128, heavy fragment) retire
 * exactly one instruction per lane per clock on 5 cores x 128 lanes, while
 * Apple's code for the same 13-instruction loop retires 1.13-1.20. Apple's
 * compiler interleaves the independent induction-variable chain (i + c, i++,
 * i == n) between the dependent xorshift steps; Mesa's pressure scheduler
 * leaves the dependent chain back to back. This pass is a latency-aware
 * top-down list scheduler over the same dependency DAG: at each step it
 * issues the ready instruction whose operands are available soonest, breaking
 * ties by the longest remaining critical path. It never adds or removes
 * instructions and keeps every ordering edge of create_dag(). A block whose
 * register pressure would rise by more than AGX_ILP_SLACK halves keeps its
 * original order.
 *
 * Latencies are a model, not measured G17P numbers: AGX_ILP_ALU_LATENCY for
 * ALU results, 4x that for memory, texture and tile loads.
 */
#define AGX_ILP_ALU_LATENCY 4
#define AGX_ILP_SLACK 2
/* Register pressure a G17P constant-program schedule may reach (of 192). */
#define AGX_ILP_CPROG_HALVES 128
/* A G17P fragment block with at least this many samples issues them ahead of
 * their consumers; hoisting the samples of a three-sample fragment program
 * was measured slower (Subnautica pass 74272). */
#define AGX_ILP_FS_MIN_SAMPLES 4
/* Register halves such a schedule may add to the block's pressure, and the
 * forward cap on the halves its own definitions hold. */
#define AGX_ILP_FS_SAMPLE_SLACK 16
#define AGX_ILP_FS_SAMPLE_HALVES 88
/* Samples a fragment block may issue ahead of the first unissued one. */
#define AGX_ILP_FS_HOIST_SAMPLES 6

/* Whether an AGX3B fragment sample or texel fetch can allocate disjoint
 * staging banks in agx3_pack_binary.c and genuinely overlap in flight.
 * Non-relocatable forms (register-descriptor 2D_ARRAY samples, 3D or cube
 * samples, gradient/register-LOD/dynamic-offset samples, and multisample or
 * register-LOD fetches) stay pinned to staging bank 0, where hoisting only
 * inflates GPR pressure and forces serializing result-wait barriers between
 * consecutive samples. */
static bool
ilp_agx3b_fragment_sample_relocatable(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE: {
      if (I->nr_srcs < 5 || I->query_lod || (I->offset && !I->offset_xyz))
         return false;

      const bool lod_ok =
         I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
         ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
           I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
          I->src[1].type == AGX_INDEX_IMMEDIATE);
      if (!lod_ok)
         return false;

      if (I->dim == AGX_DIM_2D)
         return true;

      if (I->dim == AGX_DIM_2D_ARRAY) {
         return (I->src[3].type == AGX_INDEX_UNIFORM ||
                 I->src[3].type == AGX_INDEX_IMMEDIATE) &&
                (I->src[4].type == AGX_INDEX_UNIFORM ||
                 I->src[4].type == AGX_INDEX_IMMEDIATE);
      }

      return false;
   }
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
      if (I->nr_srcs < 2 ||
          (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY &&
           I->dim != AGX_DIM_3D))
         return false;

      return !(I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
               I->src[1].type != AGX_INDEX_IMMEDIATE);
   default:
      return false;
   }
}

/* A sample for the sample-issue schedule. AGX3 encodes a storage-image read as
 * the same native SMP as a texture fetch (Apple's texture.read), so it has
 * the same latency to hide and is batched the same way. */
static bool
ilp_is_sample(const agx_context *ctx, const agx_instr *I)
{
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       ctx->stage == MESA_SHADER_FRAGMENT)
      return ilp_agx3b_fragment_sample_relocatable(I);

   return I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
          I->op == AGX_OPCODE_TEXTURE_LOAD ||
          (agx_isa_is_agx3(ctx->key->dev.encoding) &&
           I->op == AGX_OPCODE_IMAGE_LOAD);
}

/* What the sample-issue schedule moves up: samples, and with `loads` device
 * and threadgroup local loads too (see agx_ilp_schedule). */
static bool
ilp_is_hoisted(const agx_context *ctx, const agx_instr *I, bool loads)
{
   return ilp_is_sample(ctx, I) ||
          (loads && (I->op == AGX_OPCODE_DEVICE_LOAD ||
                     I->op == AGX_OPCODE_LOCAL_LOAD));
}

static unsigned
ilp_latency(const agx_instr *I, unsigned alu, unsigned load)
{
   switch (I->op) {
   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ATOMIC:
   case AGX_OPCODE_LOCAL_ATOMIC:
      return load;
   default:
      return alu;
   }
}

static signed
ilp_max_pressure(struct sched_node **order, unsigned n,
                 struct u_sparse_bitset *live_end,
                 struct u_sparse_bitset *scratch)
{
   u_sparse_bitset_dup(scratch, live_end);
   signed pressure = 0, max_pressure = 0;
   for (unsigned i = n; i-- > 0;) {
      pressure += calculate_pressure_delta(order[i]->instr, scratch);
      max_pressure = MAX2(pressure, max_pressure);
      agx_liveness_ins_update(scratch, order[i]->instr);
   }
   return max_pressure;
}

/*
 * Loop-exit compare placement.
 *
 * On Apple's own loop, moving the `icmp` next to its
 * `cndlt` keeps only 38% of Apple's advantage over Mesa, and Mesa's order
 * (counter work at the end) keeps 8%; Apple's order puts the counter's
 * update and compare `sll, xor, srl, c+i, i++, icmp, xor, iadd, iadd, xor,
 * x+=, cndlt` exactly five instructions ahead of the CND.
 *
 * Rule, applied to the order the ILP scheduler produced for a loop block
 * whose trailing control flow is an integer compare T:
 *  - the cone of T = the block's instructions T's sources depend on; it must
 *    be small (<= 3) and depend only on values from outside the block (the
 *    counter increment, not the loop's main chain);
 *  - readers = cone-independent instructions with no in-block dependency
 *    that read a value the cone also reads from outside the block (the old
 *    counter, e.g. c+i), kept just ahead of the cone so both uses of the old
 *    value end together;
 *  - everything else keeps its relative order, and the group [readers, cone]
 *    goes where `distance` of those instructions still follow it, or earlier
 *    if one of them uses a group result. The packer's early compare then
 *    emits T's compare right after the cone.
 * This is placement by dependency and distance, not a pattern: nothing
 * depends on the shift/xor opcodes. Pressure is checked by the caller. */
#define AGX_EXIT_COMPARE_DISTANCE 5

static bool
ilp_defines(const agx_instr *I, unsigned ssa)
{
   agx_foreach_ssa_dest(I, d) {
      if (I->dest[d].value == ssa)
         return true;
   }
   return false;
}

/* Whether a DAG edge from `user` to `producer` carries data. The others only
 * order the two (memory order, uniform-file access), and the later one may
 * issue on the next cycle. */
static bool
ilp_reads_result(const agx_instr *user, const agx_instr *producer)
{
   agx_foreach_ssa_src(user, s) {
      if (ilp_defines(producer, user->src[s].value))
         return true;
   }
   return false;
}

/* Values produced by a non-phi instruction of this block. Phis stay at the
 * top and their values count as loop inputs. */
static bool
ilp_block_value(struct sched_node **nodes, unsigned n, unsigned ssa)
{
   for (unsigned k = 0; k < n; ++k) {
      if (nodes[k]->instr->op != AGX_OPCODE_PHI &&
          ilp_defines(nodes[k]->instr, ssa))
         return true;
   }
   return false;
}

static bool
ilp_place_exit_compare(agx_block *block, struct sched_node **nodes,
                       struct sched_node **order, unsigned n,
                       const struct util_dynarray *dependents, void *memctx)
{
   const unsigned distance = AGX_EXIT_COMPARE_DISTANCE;

   agx_instr *T = agx_last_instr(block);
   if (!T || (T->op != AGX_OPCODE_IF_ICMP && T->op != AGX_OPCODE_WHILE_ICMP &&
              T->op != AGX_OPCODE_BREAK_IF_ICMP))
      return false;

   bool *cone = rzalloc_array(memctx, bool, n);
   bool *group = rzalloc_array(memctx, bool, n);
   if (!cone || !group)
      return false;

   unsigned cone_n = 0;
   agx_foreach_ssa_src(T, s) {
      for (unsigned i = 0; i < n; ++i) {
         if (!cone[i] && nodes[i]->instr->op != AGX_OPCODE_PHI &&
             ilp_defines(nodes[i]->instr, T->src[s].value)) {
            cone[i] = true;
            cone_n++;
         }
      }
   }
   if (cone_n == 0)
      return false;

   /* Close over in-block dependencies other than phis. */
   for (bool grew = true; grew;) {
      grew = false;
      for (unsigned i = 0; i < n; ++i) {
         if (!cone[i])
            continue;
         util_dynarray_foreach(&nodes[i]->dag.edges, struct dag_edge, edge) {
            const struct sched_node *child =
               (const struct sched_node *)edge->child;
            if (child->instr->op != AGX_OPCODE_PHI && !cone[child->index]) {
               cone[child->index] = true;
               cone_n++;
               grew = true;
            }
         }
      }
   }
   if (cone_n > 3)
      return false;

   /* Readers: other instructions whose only in-block dependencies are phis
    * and which read a loop input the cone also reads (the old counter). */
   unsigned readers = 0;
   for (unsigned j = 0; j < n; ++j) {
      group[j] = cone[j];
      if (cone[j] || nodes[j]->instr->op == AGX_OPCODE_PHI)
         continue;
      bool independent = true;
      util_dynarray_foreach(&nodes[j]->dag.edges, struct dag_edge, edge) {
         const struct sched_node *child = (const struct sched_node *)edge->child;
         independent &= child->instr->op == AGX_OPCODE_PHI;
      }
      if (!independent)
         continue;
      bool shares = false;
      agx_foreach_ssa_src(nodes[j]->instr, sj) {
         const unsigned v = nodes[j]->instr->src[sj].value;
         if (ilp_block_value(nodes, n, v))
            continue;
         for (unsigned i = 0; i < n && !shares; ++i) {
            if (!cone[i])
               continue;
            agx_foreach_ssa_src(nodes[i]->instr, si)
               shares |= nodes[i]->instr->src[si].value == v;
         }
      }
      if (shares && readers < 2) {
         group[j] = true;
         readers++;
      }
   }

   struct sched_node **rest = rzalloc_array(memctx, struct sched_node *, n);
   struct sched_node **grp = rzalloc_array(memctx, struct sched_node *, n);
   struct sched_node **out = rzalloc_array(memctx, struct sched_node *, n);
   if (!rest || !grp || !out)
      return false;
   unsigned nr = 0, ng = 0, phis = 0;
   /* Readers first, then the cone, each in the scheduled order. */
   for (unsigned pass = 0; pass < 2; ++pass) {
      for (unsigned i = 0; i < n; ++i) {
         const unsigned idx = order[i]->index;
         if (group[idx] && cone[idx] == (pass == 1))
            grp[ng++] = order[i];
      }
   }
   for (unsigned i = 0; i < n; ++i) {
      if (!group[order[i]->index]) {
         phis += order[i]->instr->op == AGX_OPCODE_PHI;
         rest[nr++] = order[i];
      }
   }

   const unsigned movable = nr - phis;
   unsigned p = phis + (movable > distance ? movable - distance : 0);
   for (unsigned q = 0; q < nr; ++q) {
      bool uses_group = false;
      for (unsigned g = 0; g < ng && !uses_group; ++g) {
         util_dynarray_foreach(&dependents[grp[g]->index], unsigned, d)
            uses_group |= *d == rest[q]->index;
      }
      if (uses_group) {
         p = MIN2(p, q);
         break;
      }
   }
   /* Phis stay first; a group user among them means no valid placement. */
   if (p < phis)
      return false;

   unsigned k = 0;
   for (unsigned i = 0; i < p; ++i)
      out[k++] = rest[i];
   for (unsigned i = 0; i < ng; ++i)
      out[k++] = grp[i];
   for (unsigned i = p; i < nr; ++i)
      out[k++] = rest[i];
   assert(k == n);

   bool changed = false;
   for (unsigned i = 0; i < n; ++i) {
      changed |= out[i] != order[i];
      order[i] = out[i];
   }
   return changed;
}

/*
 * Pixel-wait placement.
 *
 * WAIT_PIX has no SSA result; the list scheduler above gives it an ALU
 * latency and ranks it by the critical path to the tile access it guards, so
 * it tends to be hoisted as early as its barrier edges allow. In the colour
 * block of the light fragment it placed the class-4 (write) wait twelve
 * instructions ahead of LSTCOL, and the class-8 (read) wait three ahead of
 * LLDCOL, so the in-order warp could block on the pixel protocol before
 * independent arithmetic. Apple's G17P fragments put each wait immediately
 * ahead of the tile access it orders (the class-8 wait right before LLDCOL,
 * the class-4 wait right before LSTCOL), which is also what the pressure
 * scheduler's own "sink wait_pix" heuristic does.
 *
 * Move each WAIT_PIX down to just before its first dependent in the chosen
 * order. It only moves later, never past anything that depends on it, and
 * every instruction it depends on stays earlier, so all DAG edges hold.
 */
static void
ilp_sink_pixel_waits(struct sched_node **order, unsigned n,
                     const struct util_dynarray *dependents)
{
   for (unsigned p = 0; p < n; ++p) {
      if (order[p]->instr->op != AGX_OPCODE_WAIT_PIX)
         continue;

      const unsigned w = order[p]->index;
      unsigned q = p + 1;
      for (; q < n; ++q) {
         bool depends = false;
         util_dynarray_foreach(&dependents[w], unsigned, d)
            depends |= *d == order[q]->index;
         if (depends)
            break;
      }

      /* No dependent inside the block: leave it where it is. */
      if (q == n || q == p + 1)
         continue;

      struct sched_node *wait = order[p];
      memmove(&order[p], &order[p + 1], (q - 1 - p) * sizeof(order[0]));
      order[q - 1] = wait;
      /* The instruction now at p has not been examined yet. */
      --p;
   }
}

enum ilp_alu_pipe {
   ILP_PIPE_OTHER = 0,
   ILP_PIPE_F16,
   ILP_PIPE_F32,
   ILP_PIPE_INT,
};

static enum ilp_alu_pipe
ilp_instr_pipe(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
      return ILP_PIPE_F16;
   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_FFMA:
      return ILP_PIPE_F32;
   case AGX_OPCODE_IADD:
   case AGX_OPCODE_IMAD:
   case AGX_OPCODE_BITOP:
   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_EXTR:
   case AGX_OPCODE_ASR:
   case AGX_OPCODE_ICMPSEL:
      return ILP_PIPE_INT;
   default:
      return ILP_PIPE_OTHER;
   }
}

static bool
ilp_schedule_block(agx_context *ctx, agx_block *block, void *memctx,
                   bool native_uniforms, unsigned alu_latency,
                   unsigned load_latency, signed slack, signed cap,
                   const uint8_t *size16, bool place_exit_compare,
                   bool order_edges_free, signed fwd_cap,
                   unsigned hoist_samples, bool hoist_past_writes,
                   bool hoist_loads, bool loads_in_flight, bool interleave_alu,
                   struct sched_dag_state *dag_state)
{
   struct util_dynarray all;
   util_dynarray_init(&all, memctx);
   struct dag *dag = create_dag(ctx, block, memctx, native_uniforms, &all,
                                dag_state, hoist_past_writes);
   if (!dag)
      return false;

   const unsigned n = util_dynarray_num_elements(&all, struct sched_node *);
   if (n < 3)
      return false;
   struct sched_node **nodes = all.data;

   unsigned *ndeps = rzalloc_array(memctx, unsigned, n);
   unsigned *height = rzalloc_array(memctx, unsigned, n);
   unsigned *ready = rzalloc_array(memctx, unsigned, n);
   bool *has_alu_dep = rzalloc_array(memctx, bool, n);
   bool *done = rzalloc_array(memctx, bool, n);
   struct util_dynarray *dependents =
      rzalloc_array(memctx, struct util_dynarray, n);
   struct sched_node **order = rzalloc_array(memctx, struct sched_node *, n);
   if (!ndeps || !height || !ready || !has_alu_dep || !done || !dependents ||
       !order)
      return false;

   for (unsigned i = 0; i < n; ++i)
      util_dynarray_init(&dependents[i], memctx);

   /* Edges run from a dependent (parent) to its dependency (child). */
   for (unsigned i = 0; i < n; ++i) {
      util_dynarray_foreach(&nodes[i]->dag.edges, struct dag_edge, edge) {
         const struct sched_node *child = (const struct sched_node *)edge->child;
         ndeps[i]++;
         util_dynarray_append(&dependents[child->index], i);
      }
   }

   /* Dependents always follow their dependencies in the original order, so
    * a reverse sweep computes the critical-path height. */
   for (unsigned i = n; i-- > 0;) {
      const unsigned lat =
         ilp_latency(nodes[i]->instr, alu_latency, load_latency);
      unsigned h = lat;
      util_dynarray_foreach(&dependents[i], unsigned, d) {
         const bool data = !order_edges_free ||
            ilp_reads_result(nodes[*d]->instr, nodes[i]->instr);
         h = MAX2(h, (data ? lat : 1) + height[*d]);
      }
      height[i] = h;
   }

   /* With hoist_samples, the order is the original one except that each
    * sample, together with the instructions it depends on, moves up as far as
    * its dependencies allow, at most hoist_samples samples ahead of the
    * first unissued one. first_sample[i] is the ordinal of the first sample
    * that node i feeds (itself for a sample). */
   unsigned *first_sample = NULL;
   unsigned issued_samples = 0;
   /* With hoist_past_writes the window instead starts at the number of
    * memory writes already issued: a copy loop then keeps hoist_samples
    * reads in flight ahead of the write that consumes the oldest one. */
   unsigned issued_writes = 0;
   if (hoist_samples) {
      first_sample = rzalloc_array(memctx, unsigned, n);
      if (!first_sample)
         return false;
      unsigned ordinal = 0;
      for (unsigned i = 0; i < n; ++i) {
         first_sample[i] =
            ilp_is_hoisted(ctx, nodes[i]->instr, hoist_loads) ? ordinal++ : ~0u;
      }
      for (unsigned i = n; i-- > 0;) {
         util_dynarray_foreach(&dependents[i], unsigned, d)
            first_sample[i] = MIN2(first_sample[i], first_sample[*d]);
      }
   }

   /* With a register cap, track pressure forward: a value defined in this
    * block occupies its halves from its definition until its last use here,
    * or to the end when it is live out. Values from earlier blocks are not
    * counted, so the cap applies to what this block adds. While the cap is
    * reached, prefer instructions that define nothing (stores) and then the
    * smallest definitions; the final check below still measures the result
    * exactly. */
   unsigned *uses_left = NULL;
   bool *dies = NULL;
   signed live = 0;
   if (fwd_cap > 0) {
      uses_left = rzalloc_array(memctx, unsigned, ctx->alloc);
      dies = rzalloc_array(memctx, bool, ctx->alloc);
      if (!uses_left || !dies)
         return false;
      for (unsigned i = 0; i < n; ++i) {
         agx_foreach_ssa_src(nodes[i]->instr, si)
            uses_left[nodes[i]->instr->src[si].value]++;
         agx_foreach_ssa_dest(nodes[i]->instr, d) {
            dies[nodes[i]->instr->dest[d].value] =
               !u_sparse_bitset_test(&block->live_out,
                                     nodes[i]->instr->dest[d].value);
         }
      }
      /* Trailing control flow keeps its operands alive. */
      agx_foreach_instr_in_block(block, I) {
         if (!instr_after_logical_end(I))
            continue;
         agx_foreach_ssa_src(I, si)
            dies[I->src[si].value] = false;
      }
   }

   /* With loads_in_flight, the window starts at the number of hoisted loads
    * whose results something scheduled has already read, so it bounds the
    * loads in flight. Counting the loads issued instead lets every issued
    * load extend the window, which hoists all of the block's loads to its
    * top. */
   const bool in_flight = hoist_samples && loads_in_flight;
   unsigned consumed_loads = 0;
   int *load_node = NULL;
   bool *consumed = NULL;
   if (in_flight) {
      load_node = ralloc_array(memctx, int, MAX2(ctx->alloc, 1));
      consumed = rzalloc_array(memctx, bool, n);
      if (!load_node || !consumed)
         return false;
      for (unsigned v = 0; v < ctx->alloc; ++v)
         load_node[v] = -1;
      for (unsigned i = 0; i < n; ++i) {
         if (ilp_is_hoisted(ctx, nodes[i]->instr, hoist_loads)) {
            agx_foreach_ssa_dest(nodes[i]->instr, d)
               load_node[nodes[i]->instr->dest[d].value] = i;
         }
      }
   }

   unsigned cycle = 0;
   bool changed = false;
   enum ilp_alu_pipe prev_pipe = ILP_PIPE_OTHER;
   for (unsigned step = 0; step < n; ++step) {
      int best = -1;
      unsigned best_t = 0;
      signed best_grow = 0;
      bool best_fits = true;
      for (unsigned i = 0; i < n; ++i) {
         if (done[i] || ndeps[i])
            continue;
         const unsigned t = MAX2(ready[i], cycle);
         signed grow = 0;
         if (fwd_cap > 0) {
            agx_foreach_ssa_dest(nodes[i]->instr, d)
               grow += agx_index_size_16(nodes[i]->instr->dest[d]);
         }
         const bool fits = fwd_cap <= 0 || grow == 0 || live + grow <= fwd_cap;
         bool better;
         if (best < 0)
            better = true;
         else if (fits != best_fits)
            better = fits;
         else if (!fits && grow != best_grow)
            better = grow < best_grow;
         else if (hoist_samples) {
            const unsigned window =
               (hoist_past_writes ? issued_writes
                : in_flight       ? consumed_loads
                                  : issued_samples) +
               hoist_samples;
            const bool hoist_i = first_sample[i] < window;
            const bool hoist_b = first_sample[best] < window;
            better = hoist_i != hoist_b ? hoist_i : i < (unsigned)best;
         } else if (!interleave_alu) {
            better = t < best_t || (t == best_t && height[i] > height[best]);
         } else if (t != best_t) {
            better = t < best_t;
         } else if (height[i] > height[best] + alu_latency ||
                    height[best] > height[i] + alu_latency) {
            /* Keep strictly longer dependency chains ahead of shorter work so
             * the critical path never stalls behind unrelated leaves. */
            better = height[i] > height[best];
         } else {
            /* Among ready instructions within one ALU latency of each other:
             * 1. Alternate ALU execution pipes (F16 / F32 / integer) so
             *    mixed-precision loops interleave functional units and avoid
             *    long homogeneous bursts that evict shared cached operands.
             * 2. Prefer consuming a ready in-block result over launching an
             *    unbounded wave of independent producers, keeping producer-to-
             *    consumer distance within the hardware register cache. */
            const enum ilp_alu_pipe pi = ilp_instr_pipe(nodes[i]->instr);
            const enum ilp_alu_pipe pb = ilp_instr_pipe(nodes[best]->instr);
            const bool alt_i =
               prev_pipe != ILP_PIPE_OTHER && pi != ILP_PIPE_OTHER &&
               pi != prev_pipe;
            const bool alt_b =
               prev_pipe != ILP_PIPE_OTHER && pb != ILP_PIPE_OTHER &&
               pb != prev_pipe;
            const bool dep_i = has_alu_dep[i];
            const bool dep_b = has_alu_dep[best];
            if (alt_i != alt_b)
               better = alt_i;
            else if (dep_i != dep_b)
               better = dep_i;
            else if (dep_i && ready[i] != ready[best])
               better = ready[i] < ready[best];
            else
               better = height[i] > height[best];
         }
         if (better) {
            best = i;
            best_t = t;
            best_grow = grow;
            best_fits = fits;
         }
      }
      if (best < 0)
         return false; /* cycle in the DAG: leave the block alone */

      prev_pipe = ilp_instr_pipe(nodes[best]->instr);

      if (fwd_cap > 0) {
         const agx_instr *B = nodes[best]->instr;
         live += best_grow;
         agx_foreach_ssa_src(B, si) {
            const unsigned v = B->src[si].value;
            if (--uses_left[v] == 0 && dies[v])
               live -= agx_index_size_16(B->src[si]);
         }
         agx_foreach_ssa_dest(B, d) {
            const unsigned v = B->dest[d].value;
            if (uses_left[v] == 0 && dies[v])
               live -= agx_index_size_16(B->dest[d]);
         }
      }

      done[best] = true;
      order[step] = nodes[best];
      issued_samples += ilp_is_hoisted(ctx, nodes[best]->instr, hoist_loads);
      if (in_flight) {
         agx_foreach_ssa_src(nodes[best]->instr, si) {
            const int l = load_node[nodes[best]->instr->src[si].value];
            if (l >= 0 && !consumed[l]) {
               consumed[l] = true;
               consumed_loads++;
            }
         }
      }
      issued_writes += nodes[best]->instr->op == AGX_OPCODE_IMAGE_WRITE ||
                       nodes[best]->instr->op == AGX_OPCODE_DEVICE_STORE;
      changed |= (unsigned)best != step;
      cycle = best_t + 1;
      const bool best_is_phi =
         interleave_alu && nodes[best]->instr->op == AGX_OPCODE_PHI;
      const unsigned lat =
         best_is_phi ? 0 : ilp_latency(nodes[best]->instr, alu_latency, load_latency);
      util_dynarray_foreach(&dependents[best], unsigned, d) {
         const bool data = !order_edges_free ||
            ilp_reads_result(nodes[*d]->instr, nodes[best]->instr);
         ndeps[*d]--;
         ready[*d] = MAX2(ready[*d], best_t + (data ? lat : 1));
         if (interleave_alu && !best_is_phi &&
             ilp_reads_result(nodes[*d]->instr, nodes[best]->instr))
            has_alu_dep[*d] = true;
      }
   }
   /* The ILP order itself counts as a change from the source order. */
   for (unsigned i = 0; i < n; ++i)
      changed |= order[i]->index != i;
   if (place_exit_compare &&
       ilp_place_exit_compare(block, nodes, order, n, dependents, memctx)) {
      changed = false;
      for (unsigned i = 0; i < n; ++i)
         changed |= order[i]->index != i;
   }
   if (agx_isa_is_agx3(ctx->key->dev.encoding)) {
      ilp_sink_pixel_waits(order, n, dependents);
      changed = false;
      for (unsigned i = 0; i < n; ++i)
         changed |= order[i]->index != i;
   }
   if (!changed)
      return false;

   /* Live set at the logical end: walk back over the trailing control-flow
    * and export instructions, which stay where they are. */
   struct u_sparse_bitset live_end, scratch;
   u_sparse_bitset_init(&live_end, ctx->alloc, memctx);
   u_sparse_bitset_init(&scratch, ctx->alloc, memctx);
   u_sparse_bitset_dup(&live_end, &block->live_out);
   agx_instr *first_after = NULL;
   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I)) {
         first_after = I;
         break;
      }
   }
   if (first_after) {
      agx_foreach_instr_in_block_rev(block, I) {
         agx_liveness_ins_update(&live_end, I);
         if (I == first_after)
            break;
      }
   }

   /* Both peaks are relative to the values live at the logical end. The cap
    * bounds the registers the block needs, which includes those values: they
    * hold their registers through the whole block. */
   const signed before = ilp_max_pressure(nodes, n, &live_end, &scratch);
   const signed after = ilp_max_pressure(order, n, &live_end, &scratch);
   signed live_past_end = 0;
   if (cap > 0) {
      U_SPARSE_BITSET_FOREACH_SET(&live_end, v)
         live_past_end += size16[v];
   }
   if (after > before + slack && (cap <= 0 || live_past_end + after > cap))
      return false;

   for (unsigned i = 0; i < n; ++i)
      agx_remove_instruction(order[i]->instr);
   for (unsigned i = 0; i < n; ++i) {
      if (first_after)
         list_addtail(&order[i]->instr->link, &first_after->link);
      else
         list_addtail(&order[i]->instr->link, &block->instructions);
   }
   return true;
}

/* Only pure-ALU loop bodies are interleaved. Offline, scheduling the
 * crossos bw_copy grid-stride loops (vector device loads and stores) grew
 * that kernel from 23 to 39 GPR halves; the xorshift loops the issue-rate gap
 * was measured on contain no memory access. */
static bool
ilp_block_has_memory(const agx_block *block)
{
   agx_foreach_instr_in_block(block, I) {
      switch (I->op) {
      case AGX_OPCODE_DEVICE_LOAD:
      case AGX_OPCODE_DEVICE_STORE:
      case AGX_OPCODE_ATOMIC:
      case AGX_OPCODE_LOCAL_LOAD:
      case AGX_OPCODE_LOCAL_STORE:
      case AGX_OPCODE_LOCAL_ATOMIC:
      case AGX_OPCODE_TEXTURE_LOAD:
      case AGX_OPCODE_TEXTURE_SAMPLE:
      case AGX_OPCODE_IMAGE_LOAD:
      case AGX_OPCODE_IMAGE_WRITE:
      case AGX_OPCODE_STACK_LOAD:
      case AGX_OPCODE_STACK_STORE:
      case AGX_OPCODE_LD_TILE:
      case AGX_OPCODE_ST_TILE:
      case AGX_OPCODE_BLOCK_IMAGE_STORE:
         return true;
      default:
         break;
      }
   }
   return false;
}

/* A block that writes images or device memory. On M5 such a block whose
 * reads are reorderable samples is a copy loop: Mesa's pressure order puts
 * each write right after its read, so the program waits on one read at a
 * time, while Apple's compiler issues several reads ahead of their writes
 * (native725 vs_copy: four SMP, then four IMGWCI per iteration). */
static bool
ilp_block_writes_memory(const agx_block *block)
{
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IMAGE_WRITE || I->op == AGX_OPCODE_DEVICE_STORE)
         return true;
   }
   return false;
}

static unsigned
ilp_block_samples(const agx_context *ctx, const agx_block *block)
{
   unsigned samples = 0;
   agx_foreach_instr_in_block(block, I)
      samples += ilp_is_sample(ctx, I);
   return samples;
}

static unsigned
ilp_block_compute_loads(const agx_block *block)
{
   unsigned loads = 0;
   agx_foreach_instr_in_block(block, I)
      loads += I->op == AGX_OPCODE_DEVICE_LOAD ||
               I->op == AGX_OPCODE_LOCAL_LOAD;
   return loads;
}

/* True when a block emits multiple ST_TILE stores to the same pixel_offset and
 * at least one of those stores carries a sample-dependent payload. On AGX3B,
 * agx3_tile_store_bank() counts each pixel_offset once under the assumption
 * that per-sample stores to one colour target are serialized and reuse one
 * register vector in r0..r7; hoisting independent sample loads across those
 * stores keeps multiple sample payloads live simultaneously and can push
 * later ST_TILE sources past the 16-half bank. */
static bool
ilp_block_has_multi_sample_tile_stores(const agx_context *ctx,
                                       const agx_block *block, void *memctx)
{
   BITSET_WORD *sample_dep =
      rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(MAX2(ctx->alloc, 1)));
   if (!sample_dep)
      return true;

   BITSET_DECLARE(seen_offsets, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   BITSET_DECLARE(sample_offsets, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   bool result = false;

   agx_foreach_instr_in_block(block, I) {
      bool dep = ilp_is_sample(ctx, I);
      if (!dep) {
         agx_foreach_ssa_src(I, s) {
            if (BITSET_TEST(sample_dep, I->src[s].value)) {
               dep = true;
               break;
            }
         }
      }
      if (dep) {
         agx_foreach_ssa_dest(I, d)
            BITSET_SET(sample_dep, I->dest[d].value);
      }

      if (I->op != AGX_OPCODE_ST_TILE)
         continue;

      const bool uses_sample = I->src[0].type == AGX_INDEX_NORMAL &&
                               BITSET_TEST(sample_dep, I->src[0].value);
      if (I->pixel_offset >= AGX_G17P_MAX_BYTES_PER_SAMPLE) {
         if (uses_sample) {
            result = true;
            break;
         }
         continue;
      }

      if (BITSET_TEST(seen_offsets, I->pixel_offset) &&
          (uses_sample || BITSET_TEST(sample_offsets, I->pixel_offset))) {
         result = true;
         break;
      }

      BITSET_SET(seen_offsets, I->pixel_offset);
      if (uses_sample)
         BITSET_SET(sample_offsets, I->pixel_offset);
   }

   ralloc_free(sample_dep);
   return result;
}

/* Register halves the block's sample results occupy together. */
static unsigned
ilp_block_sample_halves(const agx_context *ctx, const agx_block *block)
{
   unsigned halves = 0;
   agx_foreach_instr_in_block(block, I) {
      if (!ilp_is_sample(ctx, I))
         continue;
      agx_foreach_ssa_dest(I, d)
         halves += agx_index_size_16(I->dest[d]);
   }
   return halves;
}

/* Reads a copy block keeps in flight ahead of the write consuming the oldest
 * one: Apple's vs_copy issues four SMP before its four IMGWCI. */
#define AGX_ILP_M5_COPY_SAMPLES 4

/* Samples a geometry-side M5 loop block needs for the sample-issue schedule.
 * Its pressure order otherwise places each sample right before its first use,
 * so the block waits for them one at a time; Apple issues a loop body's
 * samples ahead of the consumers (native robust_loop witness). */
#define AGX_ILP_M5_LOOP_MIN_SAMPLES 2

static bool
ilp_m5_copy_block(const agx_context *ctx, const agx_block *block)
{
   return ilp_block_samples(ctx, block) >= AGX_ILP_FS_MIN_SAMPLES &&
          ilp_block_writes_memory(block);
}

/* A fully unrolled copy loop addresses each texel with its own constant
 * coordinate, defined once ahead of the loop and so live across the whole
 * body: 64 constants held 128 of the 150 halves of the 64x64 vs_copy loop,
 * leaving no registers within its occupancy bucket for reads in flight.
 * Rematerialize such constants in the copy block, ahead of their first use
 * there (Apple computes these coordinates in the loop, with IADD). */
static bool
ilp_m5_sink_constants(agx_context *ctx, agx_block *block,
                      agx_instr **defs, agx_block **def_blocks, void *memctx)
{
   unsigned *remap = rzalloc_array(memctx, unsigned, MAX2(ctx->alloc, 1));
   if (!remap)
      return false;

   const unsigned old_alloc = ctx->alloc;
   bool progress = false;
   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I))
         break;
      if (I->op == AGX_OPCODE_PHI)
         continue;
      agx_foreach_ssa_src(I, s) {
         const unsigned v = I->src[s].value;
         if (v >= old_alloc || !defs[v] || def_blocks[v] == block ||
             defs[v]->op != AGX_OPCODE_MOV_IMM ||
             (defs[v]->dest[0].size != AGX_SIZE_16 &&
              defs[v]->dest[0].size != AGX_SIZE_32))
            continue;
         if (!remap[v]) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index t = agx_temp_like(ctx, defs[v]->dest[0]);
            agx_mov_imm_to(&b, t, defs[v]->imm);
            remap[v] = t.value;
         }
         agx_index replacement = I->src[s];
         replacement.value = remap[v];
         I->src[s] = replacement;
         progress = true;
      }
   }
   return progress;
}

/* Schedule only a small, explicit arithmetic/tile domain. Schedule-class
 * NONE alone is not a purity guarantee (e.g. DOORBELL and texture ops).
 * SSA operands expose all data dependencies; physical register accesses and
 * uniform writes stay on the original schedule. */
static bool
ilp_single_tile_blend(const agx_context *ctx, const agx_block *block)
{
   if (ctx->stage != MESA_SHADER_FRAGMENT)
      return false;

   unsigned loads = 0, stores = 0, load_target = 0, store_target = 0;
   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I))
         break;
      switch (I->op) {
      case AGX_OPCODE_LD_TILE:
         ++loads;
         load_target = I->colour_target;
         break;
      case AGX_OPCODE_ST_TILE:
         ++stores;
         store_target = I->colour_target;
         break;
      case AGX_OPCODE_PHI:
      case AGX_OPCODE_WAIT_PIX:
      case AGX_OPCODE_MOV:
      case AGX_OPCODE_MOV_IMM:
      case AGX_OPCODE_SPLIT:
      case AGX_OPCODE_COLLECT:
      case AGX_OPCODE_BITOP:
      case AGX_OPCODE_BFI:
      case AGX_OPCODE_BFEIL:
      case AGX_OPCODE_IADD:
      case AGX_OPCODE_CONVERT:
      case AGX_OPCODE_FADD:
      case AGX_OPCODE_FMUL:
      case AGX_OPCODE_FFMA:
      case AGX_OPCODE_FLTNORM_AGX3:
      case AGX_OPCODE_NORMFLT_AGX3:
         break;
      default:
         return false;
      }
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_NORMAL &&
             I->dest[d].type != AGX_INDEX_NULL)
            return false;
      }
      agx_foreach_src(I, src) {
         if (I->src[src].type == AGX_INDEX_REGISTER)
            return false;
      }
   }
   return loads == 1 && stores == 1 && load_target == store_target;
}

/* M5 compute sample-issue budget: register halves kept free below an
 * occupancy bucket for allocation alignment, and the register file size. */
#define AGX_ILP_M5_OCCUPANCY_MARGIN 8
#define AGX_ILP_M5_MAX_HALVES 256
/* Halves a program needs free below the top of its occupancy step for the
 * in-flight device-load windows (at least this many). */
#define AGX_ILP_M5_LOAD_HEADROOM 16

/* Register count up to which an M5 compute block may grow to keep device
 * loads in flight, for a program whose estimated peak is program_peak halves,
 * or 0 when such blocks keep the issue-order schedule. The loads stay inside
 * the occupancy step the program is in, AGX_ILP_M5_OCCUPANCY_MARGIN below its
 * top, as register allocation lands a few halves above the estimate. A
 * program with fewer than AGX_ILP_M5_LOAD_HEADROOM halves free below the top
 * of its step keeps the issue-order schedule, since allocation can then cross
 * into the next step even when the estimate does not grow; so does a program
 * in the last step, where crossing the top spills.
 */
int
agx_ilp_m5_device_load_cap(int program_peak)
{
   if (program_peak + AGX_ILP_M5_OCCUPANCY_MARGIN > AGX_ILP_M5_MAX_HALVES)
      return 0;

   const int step_top =
      agx_occupancy_for_register_count(program_peak).max_registers;
   if (program_peak + AGX_ILP_M5_LOAD_HEADROOM > step_top ||
       step_top >= AGX_ILP_M5_MAX_HALVES)
      return 0;

   return step_top - AGX_ILP_M5_OCCUPANCY_MARGIN;
}

/* Absolute peak register pressure of a block in source order, in halves:
 * its live-out values plus the relative peak walking back from them. */
static signed
ilp_block_peak(agx_context *ctx, agx_block *block, const uint8_t *size16,
               void *memctx)
{
   struct u_sparse_bitset live;
   u_sparse_bitset_init(&live, ctx->alloc, memctx);
   u_sparse_bitset_dup(&live, &block->live_out);
   signed out = 0;
   U_SPARSE_BITSET_FOREACH_SET(&live, v)
      out += size16[v];
   signed pressure = 0, peak = 0;
   agx_foreach_instr_in_block_rev(block, I) {
      pressure += calculate_pressure_delta(I, &live);
      peak = MAX2(peak, pressure);
      agx_liveness_ins_update(&live, I);
   }
   return out + peak;
}

static signed
ilp_static_compute_cap(const agx_context *ctx)
{
   unsigned tpw = (!ctx->nir || ctx->nir->info.workgroup_size_variable)
                     ? 1024
                     : ctx->nir->info.workgroup_size[0] *
                          ctx->nir->info.workgroup_size[1] *
                          ctx->nir->info.workgroup_size[2];
   if (!tpw)
      tpw = 1024;
   unsigned static_max = agx_max_registers_for_occupancy(tpw);
   return static_max > AGX_ILP_M5_OCCUPANCY_MARGIN
             ? (signed)(static_max - AGX_ILP_M5_OCCUPANCY_MARGIN)
             : 0;
}

/* Highest occupancy step inside AGX3B's 192-half register file (also matching
 * the 2D_MS_ARRAY cap in RA). */
#define AGX_ILP_AGX3B_MAX_HALVES 184

static int
agx_ilp_agx3b_fragment_max_halves(const agx_context *ctx)
{
   /* Respect narrower AGX3B cube-sample register budgets (matching
    * agx3_cube_register_budget in agx_register_allocate.c) so sample hoisting
    * cannot push a cube shader above its reduced register file. Because
    * agx3_reserve_cube_scratch runs after scheduling, cap cube-array and
    * bias-min samples conservatively at 112 halves and plain cube samples at
    * 160 halves (highest occupancy step inside the 176-half budget). */
   int max_halves = AGX_ILP_AGX3B_MAX_HALVES;
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || I->nr_srcs < 6 ||
          (I->dim != AGX_DIM_CUBE && I->dim != AGX_DIM_CUBE_ARRAY))
         continue;

      if (I->dim == AGX_DIM_CUBE_ARRAY ||
          I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN)
         max_halves = MIN2(max_halves, 112);
      else
         max_halves = MIN2(max_halves, 160);
   }
   return max_halves;
}

static signed
agx_ilp_agx3b_fragment_cap(agx_context *ctx, uint8_t **size16_out, void *memctx)
{
   uint8_t *size16 = *size16_out;
   if (!size16) {
      size16 = rzalloc_array(memctx, uint8_t, MAX2(ctx->alloc, 1));
      if (!size16)
         return 0;
      agx_foreach_instr_global(ctx, I) {
         agx_foreach_ssa_dest(I, d)
            size16[I->dest[d].value] = agx_index_size_16(I->dest[d]);
      }
      *size16_out = size16;
   }

   signed program_peak = 0;
   agx_foreach_block(ctx, block)
      program_peak = MAX2(program_peak,
                          ilp_block_peak(ctx, block, size16, memctx));

   const int max_halves = agx_ilp_agx3b_fragment_max_halves(ctx);
   if (program_peak + AGX_ILP_M5_OCCUPANCY_MARGIN > max_halves)
      return 0;

   return agx_occupancy_for_register_count(
             program_peak + AGX_ILP_M5_OCCUPANCY_MARGIN).max_registers -
          AGX_ILP_M5_OCCUPANCY_MARGIN;
}

void
agx_ilp_schedule(agx_context *ctx)
{
   const unsigned alu_latency = AGX_ILP_ALU_LATENCY;

   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);
   struct sched_dag_state dag_state = {0};

   bool native_uniforms = false;
   bool agx3b_frag_cap_ready = false;
   signed agx3b_frag_cap = 0;
   agx_foreach_instr_global(ctx, I) {
      native_uniforms |= native_uniform_write(I) ||
                         I->op == AGX_OPCODE_UNIFORM_STORE;
   }

   /* Loop bodies, single-tile blend blocks and constant programs: that is
    * where Apple's interleaving was measured to pay. Other straight-line code
    * keeps the pressure schedule (offline, scheduling it grew some meta
    * shaders' registers).
    * In structured AGX control flow a loop spans the source-order interval
    * from its header to the block holding the back edge. */
   unsigned nr_blocks = 0;
   agx_foreach_block(ctx, block)
      nr_blocks = MAX2(nr_blocks, block->index + 1);
   bool *in_loop = rzalloc_array(memctx, bool, MAX2(nr_blocks, 1));
   if (!in_loop) {
      ralloc_free(memctx);
      return;
   }
   agx_foreach_block(ctx, block) {
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index) {
            for (unsigned i = succ->index; i <= block->index; ++i)
               in_loop[i] = true;
         }
      }
   }

   const bool is_agx3 = agx_isa_is_agx3(ctx->key->dev.encoding);
   const bool m5 = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C;
   const bool m5_compute =
      m5 && ctx->stage == MESA_SHADER_COMPUTE && !ctx->is_preamble;
   const bool shared_compute_loads =
      !ctx->is_preamble && ctx->stage == MESA_SHADER_COMPUTE &&
      !ctx->key->dev.emit_agx3d;
   /* M5 compute uses only the sample-issue schedule below, as Apple's
    * compute kernels issue independent samples back to back (svgfVariance:
    * nine SMP in a row, then the accumulation).
    *
    * Other M5 geometry-side stages (vertex and its lowered relatives) take
    * the same occupancy-capped sample-issue schedule, but only for blocks
    * that also write memory (copy loops); everything else keeps the shared
    * AGX3 policy below. */
   const bool m5_copy = m5 && !ctx->is_preamble &&
      ctx->stage != MESA_SHADER_COMPUTE &&
      ctx->stage != MESA_SHADER_FRAGMENT;
   /* Copy blocks the M5 sample-issue schedule newly covers: every such block
    * of those other stages, and compute loop bodies. */
   bool *copy_mode = rzalloc_array(memctx, bool, MAX2(nr_blocks, 1));
   if (!copy_mode) {
      ralloc_free(memctx);
      return;
   }
   bool any_copy = false;
   if (m5_copy || m5_compute) {
      agx_foreach_block(ctx, block) {
         if (block->index < nr_blocks &&
             (m5_copy || in_loop[block->index]) &&
             ilp_m5_copy_block(ctx, block)) {
            copy_mode[block->index] = true;
            any_copy = true;
         }
      }
   }
   if (any_copy) {
      agx_instr **defs = rzalloc_array(memctx, agx_instr *, MAX2(ctx->alloc, 1));
      agx_block **def_blocks =
         rzalloc_array(memctx, agx_block *, MAX2(ctx->alloc, 1));
      if (!defs || !def_blocks) {
         ralloc_free(memctx);
         return;
      }
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            agx_foreach_ssa_dest(I, d) {
               defs[I->dest[d].value] = I;
               def_blocks[I->dest[d].value] = block;
            }
         }
      }
      bool sunk = false;
      agx_foreach_block(ctx, block) {
         if (block->index < nr_blocks && copy_mode[block->index])
            sunk |= ilp_m5_sink_constants(ctx, block, defs, def_blocks, memctx);
      }
      if (sunk) {
         agx_dce(ctx, true);
         agx_compute_liveness(ctx);
      }
   }

   /* Geometry-side M5 loop blocks with several samples and no memory writes:
    * after agx_nir_opt_sink_loop_sample_uses the robustness2 sampled_image
    * 3D vertex loop ends in a block holding two samples and the consumers of
    * all twelve; the pressure order placed those two samples late, behind
    * the consumers of the earlier ten, each completing on its own. */
   bool *loop_sample_mode = rzalloc_array(memctx, bool, MAX2(nr_blocks, 1));
   if (!loop_sample_mode) {
      ralloc_free(memctx);
      return;
   }
   if (m5_copy) {
      agx_foreach_block(ctx, block) {
         if (block->index < nr_blocks && in_loop[block->index] &&
             !copy_mode[block->index] &&
             ilp_block_samples(ctx, block) >= AGX_ILP_M5_LOOP_MIN_SAMPLES)
            loop_sample_mode[block->index] = true;
      }
   }

   /* M5 fragments: the same occupancy-capped growth as M5 compute where it
    * exceeds the G17P fragment slack below. */
   const bool m5_fragment = m5 && !ctx->is_preamble &&
      ctx->stage == MESA_SHADER_FRAGMENT;

   uint8_t *size16 = NULL;
   signed m5_occupancy_cap = 0, m5_load_cap = 0;
   if (shared_compute_loads || m5_copy || m5_fragment || ctx->is_preamble) {
      size16 = rzalloc_array(memctx, uint8_t, MAX2(ctx->alloc, 1));
      if (!size16) {
         ralloc_free(memctx);
         return;
      }
      agx_foreach_instr_global(ctx, I) {
         agx_foreach_ssa_dest(I, d)
            size16[I->dest[d].value] = agx_index_size_16(I->dest[d]);
      }
   }
   if (shared_compute_loads || m5_copy || m5_fragment) {
      signed program_peak = 0;
      agx_foreach_block(ctx, block)
         program_peak = MAX2(program_peak,
                             ilp_block_peak(ctx, block, size16, memctx));
      /* Keep a margin for register-allocation alignment. */
      if (program_peak + AGX_ILP_M5_OCCUPANCY_MARGIN <= AGX_ILP_M5_MAX_HALVES) {
         m5_occupancy_cap = agx_occupancy_for_register_count(
            program_peak + AGX_ILP_M5_OCCUPANCY_MARGIN).max_registers -
            AGX_ILP_M5_OCCUPANCY_MARGIN;
         m5_load_cap = agx_ilp_m5_device_load_cap(program_peak);
         if (!is_agx3 && mesa_shader_stage_is_compute(ctx->stage)) {
            signed static_cap = ilp_static_compute_cap(ctx);
            m5_occupancy_cap = MIN2(m5_occupancy_cap, static_cap);
            m5_load_cap = MIN2(m5_load_cap, static_cap);
         }
      }
   }

   agx_foreach_block(ctx, block) {
      if (block->index >= nr_blocks)
         continue;
      /* M5 compute: only the occupancy-capped sample-issue schedule. Other
       * stages and generations hoist compute device/local loads inside their
       * occupancy step, while pure-ALU compute loops still take ALU ILP
       * scheduling below. */
      const bool copy = copy_mode[block->index];
      const bool loop_samples = loop_sample_mode[block->index];
      /* Device and local loads too, in and outside loops: the pressure order
       * places each right before its first use, so the block waits for one
       * load at a time, as it would for samples. A copy block keeps its own
       * schedule. Exclude G18G (AGX3C with emit_agx3d). */
      const bool compute_loads = shared_compute_loads && !copy;
      const unsigned loads = compute_loads ? ilp_block_compute_loads(block) : 0;
      const unsigned samples = ilp_block_samples(ctx, block);
      const bool take_compute_hoist =
         (m5_compute || copy || loop_samples) ||
         (compute_loads && (!in_loop[block->index] || loads) &&
          samples + loads >= AGX_ILP_FS_MIN_SAMPLES);
      if (take_compute_hoist) {
         if ((!in_loop[block->index] || copy || loop_samples || loads) &&
             samples + loads >=
                (loop_samples ? AGX_ILP_M5_LOOP_MIN_SAMPLES
                              : AGX_ILP_FS_MIN_SAMPLES)) {
            /* Let the block's peak grow only up to the largest register
             * count with the occupancy the whole program already has, so
             * batching never costs threads; a program at or past the last
             * bucket (the spilling raygens) may not grow at all. */
            const signed peak = ilp_block_peak(ctx, block, size16, memctx);
            signed growth = m5_occupancy_cap > peak
                               ? m5_occupancy_cap - peak : 0;
            /* A loop block's samples otherwise complete one after another
             * on every iteration. Let the block hold all of its sample
             * results in flight, plus the fragment sample-issue slack for
             * their operands, even past the occupancy bucket (the
             * robustness2 3D vertex loop's last block: two vec4 results,
             * 21 halves of growth). */
            if (loop_samples) {
               growth = MAX2(growth, ilp_block_sample_halves(ctx, block) +
                                        AGX_ILP_FS_SAMPLE_SLACK);
            }
            if (loads && m5_load_cap > 0) {
               /* Prefer to hoist a bounded window of loads ahead of the oldest
                * one no instruction has read yet, whichever first fits the
                * program's occupancy step. This ranks candidates; register
                * pressure and scoreboard waits remain what keeps the schedule
                * safe. */
               static const unsigned windows[] = {4, 2, 1};
               const signed load_growth =
                  m5_load_cap > peak ? m5_load_cap - peak : 0;
               for (unsigned w = 0; w < ARRAY_SIZE(windows); ++w) {
                  if (ilp_schedule_block(ctx, block, memctx, native_uniforms,
                                         alu_latency, 16 * alu_latency,
                                         load_growth, 0, NULL, false, true,
                                         AGX_ILP_FS_SAMPLE_HALVES, windows[w],
                                         false, true, true, false, &dag_state))
                     break;
               }
            } else {
               ilp_schedule_block(ctx, block, memctx, native_uniforms,
                                  alu_latency, 16 * alu_latency, growth, 0,
                                  NULL, false, true, AGX_ILP_FS_SAMPLE_HALVES,
                                  copy ? AGX_ILP_M5_COPY_SAMPLES
                                       : AGX_ILP_FS_HOIST_SAMPLES,
                                  copy, loads > 0, false, false, &dag_state);
            }
         }
         continue;
      }
      /* Treat every ordered memory operation and barrier as memory across all
       * generations so pure-ALU loop scheduling never reorders protocol/helper
       * barriers, stack adjustments, or RT polling blocks. */
      bool ordered_memory = false;
      agx_foreach_instr_in_block(block, I) {
         const enum agx_schedule_class cls =
            agx_opcodes_info[I->op].schedule_class;
         ordered_memory |= cls == AGX_SCHEDULE_CLASS_LOAD ||
                           cls == AGX_SCHEDULE_CLASS_STORE ||
                           cls == AGX_SCHEDULE_CLASS_ATOMIC ||
                           cls == AGX_SCHEDULE_CLASS_BARRIER;
      }
      if (in_loop[block->index] && !ilp_block_has_memory(block) &&
          !ordered_memory) {
         ilp_schedule_block(ctx, block, memctx, native_uniforms, alu_latency,
                            4 * alu_latency, AGX_ILP_SLACK, 0, NULL, is_agx3,
                            false, 0, 0, false, false, false, is_agx3 && !m5,
                            &dag_state);
      } else if (!in_loop[block->index] && ilp_single_tile_blend(ctx, block)) {
         /* Hardware-qualified color-only budget; leave loop pressure alone. */
         ilp_schedule_block(ctx, block, memctx, native_uniforms, alu_latency,
                            4 * alu_latency, 8, 0, NULL, is_agx3, false, 0, 0,
                            false, false, false, false, &dag_state);
      } else if (!in_loop[block->index] && !ctx->is_preamble &&
                 ctx->stage == MESA_SHADER_FRAGMENT &&
                 /* On AGX3B, agx3_tile_store_bank() assumes per-sample ST_TILE
                  * stores to the same pixel_offset are serialized in r0..r7;
                  * skip sample hoisting when a block stores multiple samples
                  * to one target (background reloads and unrolled MSAA FS). */
                 !(ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                   ilp_block_has_multi_sample_tile_stores(ctx, block, memctx)) &&
                 ilp_block_samples(ctx, block) >= AGX_ILP_FS_MIN_SAMPLES) {
         /* The pressure schedule places each sample's consumers right after
          * it, so the program waits for one sample before issuing the next.
          * Apple's fragment programs issue independent samples back to back
          * and consume the results afterwards: move each sample and what it
          * depends on up, and keep the pressure order for everything else. */
         signed slack = AGX_ILP_FS_SAMPLE_SLACK;
         if (m5_fragment) {
            const signed peak = ilp_block_peak(ctx, block, size16, memctx);
            slack = MAX2(slack, m5_occupancy_cap - peak);
         } else if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
            if (!agx3b_frag_cap_ready) {
               agx3b_frag_cap =
                  agx_ilp_agx3b_fragment_cap(ctx, &size16, memctx);
               agx3b_frag_cap_ready = true;
            }
            if (size16) {
               const signed peak = ilp_block_peak(ctx, block, size16, memctx);
               slack = MAX2(slack, agx3b_frag_cap - peak);
            }
         }
         ilp_schedule_block(ctx, block, memctx, native_uniforms, alu_latency,
                            16 * alu_latency, slack, 0, NULL, false,
                            true, AGX_ILP_FS_SAMPLE_HALVES,
                            AGX_ILP_FS_HOIST_SAMPLES, false, false, false,
                            false, &dag_state);
      } else if (!in_loop[block->index] && ctx->is_preamble) {
         /* A constant program's device loads are mostly independent, but the
          * pressure schedule places each right before its uniform store, so
          * the program waits on one load at a time: 85 loads with about 50
          * separate waits in Subnautica's scene fragment. Apple's constant
          * programs issue their loads in batches of up to about twenty before
          * storing. Its registers are counted separately from the main
          * program's (preshader_register_count), so pressure is bounded only
          * to keep allocation comfortably inside the file.
          *
          * On static-register-split compute shaders (AGX2, AGX3D), the split
          * caps preamble registers to
          * agx_max_registers_for_occupancy(threads_per_workgroup) (104 halves
          * at 1024 threads); stay strictly under that limit so preambles never
          * spill into legacy scratch. */
         signed cprog_cap = AGX_ILP_CPROG_HALVES;
         signed cprog_slack = AGX_ILP_SLACK;
         if ((!is_agx3 || ctx->key->dev.emit_agx3d) &&
             mesa_shader_stage_is_compute(ctx->stage)) {
            signed static_cap = ilp_static_compute_cap(ctx);
            if (static_cap < cprog_cap) {
               cprog_cap = static_cap;
               cprog_slack = 0;
            }
         }
         if (cprog_cap > 16) {
            ilp_schedule_block(ctx, block, memctx, native_uniforms, alu_latency,
                               16 * alu_latency, cprog_slack,
                               cprog_cap, size16, false, true,
                               cprog_cap - 16, 0, false, false, false,
                               false, &dag_state);
         }
      }
   }

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   free(dag_state.last_write);
   ralloc_free(memctx);
}
