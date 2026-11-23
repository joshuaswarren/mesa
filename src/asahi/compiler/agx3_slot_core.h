/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 scoreboard slot assignment.
 *
 * AGX3 has no standalone wait instruction and no AGX2-style two-slot scoreboard.
 * Instead every async result (device/threadgroup load, texture sample, atomic,
 * stack fill) is tagged with a scoreboard slot, and every later instruction that
 * consumes such a result carries a wait-bit mask naming the slots it must first
 * synchronise. A slot waited once is satisfied for all later instructions until
 * it is reallocated ("waited once = satisfied globally"). ALU-to-ALU
 * dependencies are hardware-interlocked and need no wait.
 *
 * This is the encoding-independent algorithm -- which async op gets which slot,
 * and which slots each instruction waits -- expressed over a normalized
 * instruction list so it is testable on its own. The wait-bit field ids per
 * mnemonic (agx3_wait_field_map) and the actual deposit are the caller's.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Normal register dependencies occupy tokens 0..255. The next eight tokens
 * represent compound sample staging groups (eight halves per group), followed
 * by the 256 physical aliases used by explicit native resource values. These
 * are distinct from GPR allocation and the older implicit-coordinate token.
 * Static Mesa stack
 * offsets are 16-bit byte immediates; their 32768 addressable half-registers
 * occupy a separate token range so async copy/rld ordering is exact. The final
 * token holds the one bounded implicit-coordinate staging transaction selected
 * by the pre-RA fragment sink. */
#define AGX3_SLOT_GPR_TOKENS 256u
#define AGX3_SLOT_STAGING_TOKEN_BASE AGX3_SLOT_GPR_TOKENS
#define AGX3_SLOT_STAGING_GROUPS 8u
#define AGX3_SLOT_RESOURCE_TOKEN_BASE \
   (AGX3_SLOT_STAGING_TOKEN_BASE + AGX3_SLOT_STAGING_GROUPS)
#define AGX3_SLOT_RESOURCE_HALVES 256u
#define AGX3_SLOT_CPROG_INPUT_TOKEN \
   (AGX3_SLOT_RESOURCE_TOKEN_BASE + AGX3_SLOT_RESOURCE_HALVES)
#define AGX3_SLOT_TPS_TOKEN_BASE \
   (AGX3_SLOT_CPROG_INPUT_TOKEN + 1u)
#define AGX3_SLOT_TPS_HALF_TOKENS 32768u
#define AGX3_SLOT_TEX_COORD_TOKEN                                        \
   (AGX3_SLOT_TPS_TOKEN_BASE + AGX3_SLOT_TPS_HALF_TOKENS)
#define AGX3_SLOT_DYNAMIC_TPS_TOKEN (AGX3_SLOT_TEX_COORD_TOKEN + 1u)
#define AGX3_SLOT_NUM_TOKENS (AGX3_SLOT_DYNAMIC_TPS_TOKEN + 1u)

/* A 3D gradient+minimum sample reads six coordinate halves, fourteen LOD
 * halves, and two four-half descriptor operands. Keep the full dependency
 * list; truncating it can drop a late descriptor load's RAW wait. */
#define AGX3_SLOT_MAX_REGS 32u

/* How emission treats the scoreboard at an instruction. Slot assignment is
 * linear, so emission retires every outstanding generation before a
 * draining control-flow boundary (agx3_slot_cf_kind() in
 * agx3_pack_binary.c). Assignment starts the code after such a boundary from
 * an empty scoreboard, and agx3_slot_retire_at_cf() reports the drains. */
enum agx3_slot_cf {
   AGX3_SLOT_CF_NONE = 0,
   /* Emission retires everything outstanding before this instruction. */
   AGX3_SLOT_CF_DRAIN,
};

/*
 * A drain's retire carrier runs under the exec mask current where it is
 * emitted, and an instruction no lane executes does not reliably wait. So the
 * carrier at the end of an if arm (its else or pop_exec) retires only what
 * that arm issued: a generation issued before the if, under the enclosing
 * mask, may still be outstanding if the arm had no lanes. After an arm ends,
 * assignment therefore resumes from the state at the if, and consumers of
 * generations issued before it wait for them themselves; each consumer only
 * runs with lanes, where its wait holds.
 */

struct agx3_slot_instr {
   bool is_async;   /* produces a scoreboard-tracked result */
   bool drain;      /* retire preceding async operations before a fence */
   uint8_t cf;      /* enum agx3_slot_cf */
   /* Structured control flow: exec levels this instruction opens (an if or
    * a loop push) and closes (pop_exec). cf_arm marks an opened level whose
    * arm keeps results in flight (cf == AGX3_SLOT_CF_NONE); cf_else ends one
    * arm of the innermost level and starts the next. */
   uint8_t cf_push, cf_pop;
   bool cf_arm, cf_else;
   /* This asynchronous instruction reads its sources when it retires rather
    * than when it issues, so redefining one of them first changes what it
    * transfers. Set it for device stores, whose payload nothing else tracks:
    * a store has no destination, so the ordinary result bookkeeping below
    * never sees those registers. Loads latch their address at issue and keep
    * this clear, which the native corpus requires. */
   bool latches_srcs;
   /* Selected sources read through completion, independently of ordinary
    * issue-time GPR inputs. Native SMP uses this for its staging group while
    * its descriptor operands retain the existing issue-time contract. */
   uint32_t late_src_mask;
   uint8_t preferred_slot; /* 0=round-robin, otherwise required slot */
   /* 0 = any slot. Otherwise the round-robin choice is limited to slots
    * 1..max_slot, for an encoding that can only name those slots. */
   uint8_t max_slot;
   /* Optional source index plus one whose completed producer slot may be
    * recycled for this event. This is a preference, never a new wait: the
    * source's actual generation must already be synchronized by dataflow.
    * Distinct events remain distinct even when they use the same slot.
    */
   uint8_t coalesce_src;
   /* The round-robin slot may be reissued while an earlier generation of the
    * same kind is still live there, since the instruction cannot wait for it:
    * Apple reissues fragment coefficient-load slots without any wait. */
   bool reissues_live_slot;
   /* Slots used and fully consumed by this instruction's multi-op encoding.
    * A live prior generation is waited before the internal producer reuses
    * the slot; its register aliases are then synchronously complete. */
   uint16_t transient_slots;
   uint8_t nr_dsts; /* dependency tokens this instruction writes */
   uint8_t nr_srcs; /* dependency tokens this instruction reads */
   uint16_t dst[AGX3_SLOT_MAX_REGS];
   uint16_t src[AGX3_SLOT_MAX_REGS];
   /* Compatible asynchronous producers can share a native completion queue.
    * Zero keeps the one-generation policy. The count is bounded separately
    * from the number of physical scoreboard slots. */
   uint8_t join_class;
   uint8_t join_limit;
   bool prefer_join;
};

struct agx3_slot_result {
   uint8_t slot;       /* 1..nslots for async instructions, 0 otherwise */
   uint16_t wait_mask; /* bit (s-1) set => wait scoreboard slot s */
};

/* An asynchronous producer outside this instruction region. For example,
 * the native loader publishes constant-program imports on slot 6. Reads,
 * overwrites, drains and reuse of that slot must retire the entry generation
 * by the same rules as a producer within the region. */
struct agx3_slot_entry {
   uint16_t token;
   uint8_t slot;
};

bool agx3_slot_assign_with_entry(const struct agx3_slot_instr *instrs, unsigned n,
                                 unsigned nslots,
                                 const struct agx3_slot_entry *entry,
                                 unsigned entry_count,
                                 struct agx3_slot_result *out);

/* A still-pending producer before the first instruction in this region.
 * Tokens name its written results, not async reads of an input payload.
 * Ranges must be nonempty, disjoint and inside the dependency-token domain.
 * Several ranges may share one slot; waiting it retires all of them. */
struct agx3_slot_live_range {
   uint16_t first;
   uint16_t count;
   uint8_t slot;
};

/*
 * Assign slots and waits for one straight-line instruction region, which may
 * span fallthrough basic blocks. `n` is the instruction count; `nslots` is 6
 * (agx3b) or 8 (agx3c). Writes one result per instruction. Returns false on a
 * malformed input (too many regs, nslots out of range, null pointers).
 *
 * GPR tokens are half-register indices; static TPS tokens use the disjoint
 * range above. A value that is not currently the live result of an async op
 * contributes no wait. The default slot policy is round-robin. A caller may
 * request a corpus-witnessed slot for one async instruction with
 * preferred_slot; dependencies then follow that actual slot normally.
 * transient_slots models a compound encoding which publishes an internal
 * staging value on a slot and consumes it before the logical instruction
 * ends. It may not overlap the instruction's result slot.
 */
bool agx3_slot_assign(const struct agx3_slot_instr *instrs, unsigned n,
                      unsigned nslots, struct agx3_slot_result *out);

/*
 * After assignment: remove any wait on a generation that a control-flow
 * drain (instrs[i].cf) already retired, until a later producer reuses the
 * slot. entry_wait holds the slots pending on entry to the region. When
 * `drained` is not NULL, drained[i] is set when the drain before instrs[i]
 * retires a generation instrs[i] does not read; such a drain costs latency
 * that state carried across the branch would not.
 */
void agx3_slot_retire_at_cf(const struct agx3_slot_instr *instrs, unsigned n,
                            uint16_t entry_wait, struct agx3_slot_result *res,
                            bool *drained);

/* Import pending results into the same scoreboard ledger as instructions.
 * For example, the G17P loader publishes packed constant-program inputs in
 * TPS halves 8..8+input_count on slot 6 before the shader starts. This models
 * their RAW/WAW dependencies without a wait on every subsequent input read.
 * A NULL live pointer is valid only with live_count == 0, which preserves
 * agx3_slot_assign exactly. Invalid live ranges leave out unchanged. */
bool agx3_slot_assign_with_live(
   const struct agx3_slot_instr *instrs, unsigned n, unsigned nslots,
   const struct agx3_slot_live_range *live, unsigned live_count,
   struct agx3_slot_result *out);

#ifdef __cplusplus
}
#endif
