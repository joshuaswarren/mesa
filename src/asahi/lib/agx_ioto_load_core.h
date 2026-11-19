/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded host-side model of Apple's AGX3 EslBuilder device-load scheduler.
 *
 * This core owns no memory and is deliberately not a production enablement
 * gate.  LDIMM, allocator, cache, ordering, two-phase finish and G15-style
 * NOP padding are encoded here.  The still-anonymous LDDEV/COPY operands are
 * supplied as exact native records by callbacks and validated before append.
 * The grounded sources are loadAbsolute, SCS, and direct ABI-buffer bases;
 * indirect/user/gather and texture/sampler/image ownership remain excluded.
 */

#pragma once

#include "agx_chip.h"
#include "agx_ioto_core.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_IOTO_LOAD_LDIMM_MAX_SIZE      8u
#define AGX_IOTO_LOAD_INLINE_LDDEV14_SIZE 14u
#define AGX_IOTO_LOAD_GPU1_LDDEV_MAX_SIZE 16u
#define AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE  10u
#define AGX_IOTO_LOAD_GPU1_COPY_MAX_SIZE  8u
#define AGX_IOTO_LOAD_NOP_SIZE            2u
#define AGX_IOTO_LOAD_GPU0_ROUND_NOP_SIZE 8u

#define AGX_IOTO_CPROG_UNIFORM_HALFS 512u
#define AGX_IOTO_CPROG_UNIFORM_WORDS (AGX_IOTO_CPROG_UNIFORM_HALFS / 64u)

/* Live source uniform halves of a compiled preamble. Known-empty differs
 * from missing metadata. This describes inputs, not a qualified entry ABI. */
struct agx_ioto_cprog_inputs {
   uint64_t uniform_halfs[AGX_IOTO_CPROG_UNIFORM_WORDS];
   bool known;
};

/* G17P default TPS input layout: source indices are packed in increasing
 * order at TPS byte 16. Offset is relative to that payload, for
 * agx3_ioto_pack_cprog_copy(). Output arguments are unchanged on failure. */
bool agx_ioto_cprog_input_offset(enum agx_chip chip,
                                const struct agx_ioto_cprog_inputs *inputs,
                                uint16_t source_half, uint16_t *offset_bytes);

enum agx_ioto_load_encoding {
   AGX_IOTO_LOAD_ENCODING_UNSUPPORTED = 0,
   AGX_IOTO_LOAD_ENCODING_GPU0,
   AGX_IOTO_LOAD_ENCODING_GPU1,
};

struct agx_ioto_load_sink {
   uint8_t *data;
   size_t capacity;
   size_t length;
};

/* Apple Register objects compare by the tuple documented on this structure. */
struct agx_ioto_load_register {
   uint16_t half;
   uint32_t type;
   int32_t value;
};

struct agx_ioto_inflight_load {
   struct agx_ioto_load_register data;
   struct agx_ioto_load_register base;
   int16_t byte_offset;
   uint8_t count_dwords;
   uint16_t native_slot_half;
   uint8_t lane_mask;
   uint16_t cprog_dest_half;

   /* Optional native constant-program TPS publication, independent of the
    * ordinary uniform lane mask. Selected halves compact from this base. */
   uint8_t cprog_input_mask;
   uint16_t cprog_input_tps_half;

   uint32_t sequence;
   uint8_t signal_slot;
   bool processed;
};

struct agx_ioto_load_cache_entry {
   uint64_t address;
   uint16_t base_half;
};

/* One absolute address materialized by an LDIMM low/high pair.  Consumers
 * may relocate that pair only after proving the load schedule is unchanged. */
struct agx_ioto_load_relocation {
   size_t offset;
   uint64_t address;
   uint16_t dest_half;
   uint8_t low_length;
   uint8_t high_length;
};

struct agx_ioto_load_workspace {
   struct agx_ioto_inflight_load *loads;
   size_t capacity;
};

/* GPU0 does not expose AGXIotoInstruction_LDDEV.  Its EslBuilder writes the
 * AGX3b lddev instruction inline as one fixed 14-byte record. */
enum agx_ioto_load_raw_kind {
   AGX_IOTO_LOAD_RAW_INVALID = 0,
   AGX_IOTO_GPU0_INLINE_LDDEV14,
   AGX_IOTO_LOAD_RAW_LDDEV_EP0,
   AGX_IOTO_LOAD_RAW_LDDEV_EP1,
   AGX_IOTO_LOAD_RAW_LDDEV_EP2,
   AGX_IOTO_LOAD_RAW_LDDEV_EP3,
   AGX_IOTO_LOAD_RAW_LDDEV_EP4,
   AGX_IOTO_LOAD_RAW_LDDEV_EP5,
   AGX_IOTO_LOAD_RAW_COPY_EP0,
   AGX_IOTO_LOAD_RAW_COPY_EP1,
};

struct agx_ioto_load_raw_record {
   enum agx_ioto_load_raw_kind kind;
   uint8_t length;
   uint8_t bytes[AGX_IOTO_LOAD_GPU1_LDDEV_MAX_SIZE];
};

/*
 * Callbacks are the narrow fail-closed boundary around fields whose native
 * semantic names are not yet established.  A callback must return one exact
 * target-native record.  The core validates its length and recovered fixed
 * bits before appending it transactionally.
 */
struct agx_ioto_load_emit_ops {
   bool (*pack_load)(void *context, enum agx_chip chip,
                     const struct agx_ioto_inflight_load *load,
                     struct agx_ioto_load_raw_record *record,
                     uint8_t *signal_slot);
   bool (*pack_copy)(void *context, enum agx_chip chip,
                     const struct agx_ioto_inflight_load *load, uint8_t dword,
                     uint8_t lane_code,
                     struct agx_ioto_load_raw_record *record);
   /* Optional copies from the same loaded registers after the ordinary
    * uniform copy. Both callbacks must be supplied together. The bound
    * callback is pure; extra copies are validated and appended within the
    * same transaction. Currently qualified only for default G17P TPS inputs. */
   bool (*extra_copy_bound)(void *context, enum agx_chip chip,
                            const struct agx_ioto_inflight_load *load,
                            size_t *bytes);
   bool (*pack_extra_copies)(void *context, enum agx_chip chip,
                             const struct agx_ioto_inflight_load *load,
                             uint8_t dword,
                             struct agx_ioto_load_raw_record records[2],
                             uint8_t *count);
   void *context;

   /* Only the in-tree native packer may set this. Its callbacks construct
    * records from already range-checked fields and their exhaustive tests
    * compare every supported form against the independent composer. Custom
    * callbacks remain fully decoded and validated before publication. */
   bool trusted_records;

   /* Zero, one or two additional COPY records after the uniform COPY. The
    * ordinary COPY must preserve source halves consumed by these records. */
   bool (*pack_cprog_copy)(void *context, enum agx_chip chip,
                          const struct agx_ioto_inflight_load *load,
                          uint8_t dword,
                          struct agx_ioto_load_raw_record records[2],
                          uint8_t *count);
};

struct agx_ioto_load_core {
   enum agx_chip chip;
   enum agx_ioto_load_encoding encoding;
   struct agx_ioto_load_sink sink;

   /* First COPY in the final completed round. The emitter knows this exact
    * record boundary; consumers must not decode the finished body to recover
    * information already established transactionally here. */
   size_t copy_suffix_offset;

   struct agx_ioto_inflight_load *loads;
   size_t load_count;
   size_t load_capacity;

   struct agx_ioto_load_cache_entry *cache;
   size_t cache_count;
   size_t cache_capacity;

   /* A launcher may materialize one immutable base before branching into a
    * persistent load program.  Keep that base reserved and re-seed it at
    * every complete round boundary; unlike ordinary address-cache entries it
    * is part of the caller ABI and is not owned by the round allocator. */
   uint64_t fixed_base_address;
   uint16_t fixed_base_half;
   bool fixed_base_valid;

   struct agx_ioto_load_relocation *relocations;
   size_t relocation_count;
   size_t relocation_capacity;

   uint32_t next_sequence;
   uint8_t initial_gpr_dwords;
   uint8_t current_gpr_dwords;
   bool fast_path_enabled;
   /* Materialize every base with the fixed 8-byte LDIMM form, so a recorded
    * relocation can be rewritten with any address. */
   bool long_ldimm;
   bool gpu0_round_nops;
   /* Apple's fixed SCS base register (half-register 8) is initialized lazily
    * and survives finishRound().  ABI buffer slot 4 aliases that register.
    * The native builder relies on caller ordering for the alias, so the
    * bounded public model tracks both directions and rejects unsafe mixing. */
   bool scs_base_loaded;
   bool scs_base_ever_loaded;
   bool abi4_load_pending;
};

struct agx_ioto_load_absolute {
   uint32_t native_slot;
   uint8_t count_dwords;
   uint64_t address;
   uint8_t lane_mask;
   uint16_t cprog_dest_half;
   uint8_t cprog_input_mask;
   uint16_t cprog_input_tps_half;
};

#define AGX_IOTO_LOAD_SCS_BASE_HALF 8u

#define AGX_IOTO_LOAD_SCS_BASE      UINT32_C(0x6000)

/* The only ABI-buffer base selectors an AGX3 state load from the ABI buffer
 * accepts.  Values are native dword registers; device loads consume the
 * corresponding value * 2 half-register. */
enum agx_ioto_abi_buffer_slot {
   AGX_IOTO_ABI_BUFFER_SLOT_2 = 2,
   AGX_IOTO_ABI_BUFFER_SLOT_4 = 4,
   AGX_IOTO_ABI_BUFFER_SLOT_6 = 6,
   AGX_IOTO_ABI_BUFFER_SLOT_8 = 8,
   AGX_IOTO_ABI_BUFFER_SLOT_50 = 50,
   AGX_IOTO_ABI_BUFFER_SLOT_52 = 52,
   AGX_IOTO_ABI_BUFFER_SLOT_54 = 54,
};

struct agx_ioto_load_from_fixed {
   uint32_t native_slot;
   uint8_t count_dwords;
   int16_t byte_offset;
   uint8_t lane_mask;
   uint16_t cprog_dest_half;
};

struct agx_ioto_load_allocation {
   uint16_t temp_half;
   uint8_t reserved_dwords;
   bool fast_path;
};

enum agx_ioto_load_alloc_result {
   AGX_IOTO_LOAD_ALLOC_INVALID = 0,
   AGX_IOTO_LOAD_ALLOC_OK,
   AGX_IOTO_LOAD_ALLOC_NEEDS_ROUND,
};

enum agx_ioto_load_encoding agx_ioto_load_target_encoding(enum agx_chip chip);

bool agx_ioto_load_core_init(struct agx_ioto_load_core *core,
                             enum agx_chip chip, struct agx_ioto_load_sink sink,
                             struct agx_ioto_inflight_load *loads,
                             size_t load_capacity,
                             struct agx_ioto_load_cache_entry *cache,
                             size_t cache_capacity, uint8_t initial_gpr_dwords,
                             bool fast_path_enabled, bool gpu0_round_nops);

/* Register a 64-bit base already materialized by the caller.  Its four
 * half-registers must lie wholly inside the initial reserved GPR prefix.
 * The address cache is seeded transactionally and remains seeded across
 * finishRound(true). */
bool agx_ioto_load_core_set_fixed_base(struct agx_ioto_load_core *core,
                                       uint64_t address,
                                       uint16_t base_half);

/* Drop every materialized base except the fixed one: later loads materialize
 * their own. Used to give each patchable uniform range its own base. */
void agx_ioto_load_core_forget_bases(struct agx_ioto_load_core *core);

/* The 8-byte LDIMM form, valid for any immediate, as a little-endian
 * 64-bit word: the destination fields, and the immediate fields that can be
 * ORed into an encoding of the same destination with a zero immediate. */
static inline uint64_t
agx_ioto_load_ldimm_long_dest(uint16_t dest_half)
{
   return (uint64_t)(UINT16_C(0x800c) | (((dest_half >> 1) & 0xf) << 4)) |
          (uint64_t)(UINT16_C(0x0002) | ((dest_half & 1) << 2) |
                     (((dest_half >> 5) & 3) << 6) |
                     (((dest_half >> 7) & 1) << 3)) << 16;
}

static inline uint64_t
agx_ioto_load_ldimm_long_immediate(uint32_t immediate)
{
   return (uint64_t)((immediate & 0x7f) << 8) |
          (uint64_t)(((immediate >> 25) & 0x7f) << 9) << 16 |
          (uint64_t)((((immediate >> 7) & 0xf) << 1) |
                     (((immediate >> 11) & 3) << 10)) << 32 |
          (uint64_t)(((immediate >> 13) & 0xf) |
                     (((immediate >> 17) & 0xff) << 4)) << 48;
}

static inline void
agx_ioto_load_pack_ldimm_long(uint16_t dest_half, uint32_t immediate,
                              uint8_t out[AGX_IOTO_LOAD_LDIMM_MAX_SIZE])
{
   const uint64_t word = agx_ioto_load_ldimm_long_dest(dest_half) |
                         agx_ioto_load_ldimm_long_immediate(immediate);
   for (unsigned i = 0; i < 8; ++i)
      out[i] = (uint8_t)(word >> (8 * i));
}

bool agx_ioto_load_pack_ldimm(enum agx_chip chip, uint16_t dest_half,
                              uint32_t immediate,
                              uint8_t out[AGX_IOTO_LOAD_LDIMM_MAX_SIZE],
                              uint8_t *length);

void agx_ioto_load_pack_nop(uint8_t hazard_mask,
                            uint8_t out[AGX_IOTO_LOAD_NOP_SIZE]);

/* Pure allocator step.  NEEDS_ROUND and INVALID leave core unchanged. */
enum agx_ioto_load_alloc_result
agx_ioto_load_allocator_reserve(struct agx_ioto_load_core *core,
                                uint8_t requested_dwords,
                                uint16_t cprog_dest_half, uint32_t lane_mask,
                                struct agx_ioto_load_allocation *allocation);

/* Exact Apple comparator: {key(base), key(data)}. */
int agx_ioto_load_compare(const struct agx_ioto_inflight_load *a,
                          const struct agx_ioto_inflight_load *b);

/* Maximum ordinary load/uniform-copy bytes for the current pending set.
 * Use the with_ops variant when optional input copies are enabled. */
bool agx_ioto_load_finish_worst_case_size(const struct agx_ioto_load_core *core,
                                          bool complete_round,
                                          size_t *required);

/* Includes optional input copies. NULL ops retains the original bound. */
bool agx_ioto_load_finish_worst_case_size_with_ops(
   const struct agx_ioto_load_core *core, bool complete_round,
   const struct agx_ioto_load_emit_ops *ops, size_t *required);

/* false emits/marks only loads; true also resets allocation, emits the GPU0
 * four-NOP pad when configured, emits copies, and consumes pending loads. */
bool agx_ioto_load_finish_round(struct agx_ioto_load_core *core,
                                bool complete_round,
                                const struct agx_ioto_load_emit_ops *ops,
                                struct agx_ioto_load_workspace workspace);

/* Queue one compiler-native absolute load.  All logical state and sink length
 * roll back on failure.  Counts above four are rejected here because Apple's
 * queued lane mask is one byte; allocator boundary coverage uses reserve(). */
bool agx_ioto_load_absolute(struct agx_ioto_load_core *core,
                            const struct agx_ioto_load_absolute *load,
                            const struct agx_ioto_load_emit_ops *ops,
                            struct agx_ioto_load_workspace workspace,
                            struct agx_ioto_load_allocation *allocation);

bool agx_ioto_load_from_scs(struct agx_ioto_load_core *core,
                            const struct agx_ioto_load_from_fixed *load,
                            const struct agx_ioto_load_emit_ops *ops,
                            struct agx_ioto_load_workspace workspace,
                            struct agx_ioto_load_allocation *allocation);

bool agx_ioto_load_from_abi_buffer(struct agx_ioto_load_core *core,
                                   enum agx_ioto_abi_buffer_slot source,
                                   const struct agx_ioto_load_from_fixed *load,
                                   const struct agx_ioto_load_emit_ops *ops,
                                   struct agx_ioto_load_workspace workspace,
                                   struct agx_ioto_load_allocation *allocation);

#ifdef __cplusplus
}
#endif
