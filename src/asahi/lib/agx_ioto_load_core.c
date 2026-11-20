/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ioto_load_core.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static unsigned
cprog_popcount(uint64_t bits)
{
   unsigned count = 0;
   while (bits) {
      bits &= bits - 1;
      ++count;
   }
   return count;
}

bool
agx_ioto_cprog_input_offset(enum agx_chip chip,
                            const struct agx_ioto_cprog_inputs *inputs,
                            uint16_t source_half, uint16_t *offset_bytes)
{
   if (chip != AGX_CHIP_G17P || inputs == NULL || !inputs->known ||
       offset_bytes == NULL || source_half >= AGX_IOTO_CPROG_UNIFORM_HALFS)
      return false;

   const unsigned word = source_half / 64, bit = source_half % 64;
   if (!(inputs->uniform_halfs[word] & (UINT64_C(1) << bit)))
      return false;

   unsigned preceding = 0;
   for (unsigned i = 0; i < word; ++i)
      preceding += cprog_popcount(inputs->uniform_halfs[i]);
   preceding += cprog_popcount(inputs->uniform_halfs[word] &
                               ((UINT64_C(1) << bit) - 1));
   *offset_bytes = 2 * preceding;
   return true;
}

static uint16_t
load_u16(const uint8_t *bytes)
{
   return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static void
store_u16(uint8_t *bytes, uint16_t value)
{
   bytes[0] = (uint8_t)value;
   bytes[1] = (uint8_t)(value >> 8);
}

enum agx_ioto_load_encoding
agx_ioto_load_target_encoding(enum agx_chip chip)
{
   switch (agx_chip_ioto_generation(chip)) {
   case AGX_IOTO_GENERATION_GPU0:
      return AGX_IOTO_LOAD_ENCODING_GPU0;
   case AGX_IOTO_GENERATION_GPU1:
      return AGX_IOTO_LOAD_ENCODING_GPU1;
   default:
      return AGX_IOTO_LOAD_ENCODING_UNSUPPORTED;
   }
}

static uint8_t
allocator_limit(const struct agx_ioto_load_core *core)
{
   return core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 ? 97 : 127;
}

bool
agx_ioto_load_core_init(struct agx_ioto_load_core *core, enum agx_chip chip,
                        struct agx_ioto_load_sink sink,
                        struct agx_ioto_inflight_load *loads,
                        size_t load_capacity,
                        struct agx_ioto_load_cache_entry *cache,
                        size_t cache_capacity, uint8_t initial_gpr_dwords,
                        bool fast_path_enabled, bool gpu0_round_nops)
{
   if (core == NULL || (sink.capacity != 0 && sink.data == NULL) ||
       sink.length > sink.capacity || (load_capacity != 0 && loads == NULL) ||
       (cache_capacity != 0 && cache == NULL))
      return false;

   const enum agx_ioto_load_encoding encoding =
      agx_ioto_load_target_encoding(chip);
   if (encoding == AGX_IOTO_LOAD_ENCODING_UNSUPPORTED)
      return false;

   const uint8_t limit = encoding == AGX_IOTO_LOAD_ENCODING_GPU0 ? 97 : 127;
   if (initial_gpr_dwords >= limit)
      return false;

   memset(core, 0, sizeof(*core));
   core->chip = chip;
   core->encoding = encoding;
   core->sink = sink;
   core->loads = loads;
   core->load_capacity = load_capacity;
   core->cache = cache;
   core->cache_capacity = cache_capacity;
   core->initial_gpr_dwords = initial_gpr_dwords;
   core->current_gpr_dwords = initial_gpr_dwords;
   core->fast_path_enabled = fast_path_enabled;
   core->gpu0_round_nops =
      encoding == AGX_IOTO_LOAD_ENCODING_GPU0 && gpu0_round_nops;
   return true;
}

bool
agx_ioto_load_core_set_fixed_base(struct agx_ioto_load_core *core,
                                  uint64_t address, uint16_t base_half)
{
   if (!core || core->fixed_base_valid || core->load_count ||
       core->cache_count || !core->cache || !core->cache_capacity ||
       (address & 3) || (base_half & 1) ||
       (uint32_t)base_half + 4u > (uint32_t)core->initial_gpr_dwords * 2u)
      return false;

   core->fixed_base_address = address;
   core->fixed_base_half = base_half;
   core->fixed_base_valid = true;
   core->cache[0] = (struct agx_ioto_load_cache_entry){
      .address = address,
      .base_half = base_half,
   };
   core->cache_count = 1;
   return true;
}

bool
agx_ioto_load_pack_ldimm(enum agx_chip chip, uint16_t dest_half,
                         uint32_t immediate,
                         uint8_t out[AGX_IOTO_LOAD_LDIMM_MAX_SIZE],
                         uint8_t *length)
{
   if (out == NULL || length == NULL || (dest_half & 1) != 0 ||
       dest_half >= 256)
      return false;

   const enum agx_ioto_load_encoding encoding =
      agx_ioto_load_target_encoding(chip);
   if (encoding == AGX_IOTO_LOAD_ENCODING_UNSUPPORTED)
      return false;

   memset(out, 0, AGX_IOTO_LOAD_LDIMM_MAX_SIZE);
   if (dest_half < 32 && immediate < 128) {
      out[0] =
         (uint8_t)((encoding == AGX_IOTO_LOAD_ENCODING_GPU0 ? 0x0c : 0x04) |
                   ((dest_half >> 1) << 4));
      out[1] = (uint8_t)immediate;
      *length = 2;
      return true;
   }

   agx_ioto_load_pack_ldimm_long(dest_half, immediate, out);
   *length = 8;
   return true;
}

void
agx_ioto_load_core_forget_bases(struct agx_ioto_load_core *core)
{
   core->cache_count = 0;
}

void
agx_ioto_load_pack_nop(uint8_t hazard_mask, uint8_t out[AGX_IOTO_LOAD_NOP_SIZE])
{
   const uint16_t word =
      UINT16_C(0x0006) | (uint16_t)((hazard_mask & 0x3f) << 10);
   store_u16(out, word);
}

static bool
full_fast_mask(uint8_t dwords, uint32_t *mask)
{
   if (dwords > 16)
      return false;

   const unsigned bits = 2u * dwords;
   *mask = bits == 32 ? UINT32_MAX : ((UINT32_C(1) << bits) - 1);
   return true;
}

/* Pure allocator step.  NEEDS_ROUND and INVALID leave core unchanged. */
enum agx_ioto_load_alloc_result
agx_ioto_load_allocator_reserve(struct agx_ioto_load_core *core,
                                uint8_t requested_dwords,
                                uint16_t cprog_dest_half, uint32_t lane_mask,
                                struct agx_ioto_load_allocation *allocation)
{
   if (allocation != NULL)
      memset(allocation, 0, sizeof(*allocation));
   if (core == NULL || allocation == NULL || requested_dwords == 0 ||
       core->encoding == AGX_IOTO_LOAD_ENCODING_UNSUPPORTED)
      return AGX_IOTO_LOAD_ALLOC_INVALID;

   const uint8_t n = requested_dwords > 2 ? requested_dwords : 2;
   const uint8_t limit = allocator_limit(core);
   if ((unsigned)core->initial_gpr_dwords + n >= limit)
      return AGX_IOTO_LOAD_ALLOC_INVALID;

   uint32_t expected_mask;
   if (core->fast_path_enabled && full_fast_mask(n, &expected_mask) &&
       lane_mask == expected_mask &&
       (unsigned)(cprog_dest_half / 2) + 2u * n <= 76u) {
      allocation->temp_half = (uint16_t)(cprog_dest_half / 2 + 20);
      allocation->reserved_dwords = n;
      allocation->fast_path = true;
      return AGX_IOTO_LOAD_ALLOC_OK;
   }

   if ((unsigned)core->current_gpr_dwords + n >= limit)
      return AGX_IOTO_LOAD_ALLOC_NEEDS_ROUND;

   allocation->temp_half = (uint16_t)(2u * core->current_gpr_dwords);
   allocation->reserved_dwords = n;
   core->current_gpr_dwords = (uint8_t)(core->current_gpr_dwords + n);
   return AGX_IOTO_LOAD_ALLOC_OK;
}

static unsigned
register_class(uint32_t type)
{
   static const uint8_t table[] = {2, 2, 2, 7, 7, 7, 4, 4};
   return type >= 2 && type <= 9 ? table[type - 2] : 3;
}

static unsigned
register_subclass(uint32_t type)
{
   static const uint8_t table[] = {2, 1, 2, 3, 1, 2, 3, 1, 2};
   return type >= 1 && type <= 9 ? table[type - 1] : 1;
}

static int
compare_u32(uint32_t a, uint32_t b)
{
   return a < b ? -1 : a > b ? 1 : 0;
}

static int
compare_i32(int32_t a, int32_t b)
{
   return a < b ? -1 : a > b ? 1 : 0;
}

static int
compare_register(const struct agx_ioto_load_register *a,
                 const struct agx_ioto_load_register *b)
{
   int c = compare_u32(register_class(a->type), register_class(b->type));
   if (c != 0)
      return c;
   c = compare_u32(a->half, b->half);
   if (c != 0)
      return c;
   c = compare_u32(register_subclass(a->type), register_subclass(b->type));
   if (c != 0)
      return c;
   return compare_i32(a->value, b->value);
}

/* Exact Apple comparator: {key(base), key(data)}. */
int
agx_ioto_load_compare(const struct agx_ioto_inflight_load *a,
                      const struct agx_ioto_inflight_load *b)
{
   int c = compare_register(&a->base, &b->base);
   return c != 0 ? c : compare_register(&a->data, &b->data);
}

static void
sort_loads(struct agx_ioto_inflight_load *loads, size_t count)
{
   for (size_t i = 1; i < count; ++i) {
      struct agx_ioto_inflight_load value = loads[i];
      size_t j = i;
      while (j != 0 && agx_ioto_load_compare(&value, &loads[j - 1]) < 0) {
         loads[j] = loads[j - 1];
         --j;
      }
      loads[j] = value;
   }
}

static bool
add_size(size_t *total, size_t add)
{
   if (SIZE_MAX - *total < add)
      return false;
   *total += add;
   return true;
}

/* Maximum bytes a finish call may append for the current pending set. */
bool
agx_ioto_load_finish_worst_case_size(const struct agx_ioto_load_core *core,
                                     bool complete_round, size_t *required)
{
   if (required != NULL)
      *required = 0;
   if (core == NULL || required == NULL)
      return false;

   size_t total = 0;
   const size_t load_size = core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0
                               ? AGX_IOTO_LOAD_INLINE_LDDEV14_SIZE
                               : AGX_IOTO_LOAD_GPU1_LDDEV_MAX_SIZE;
   const size_t copy_size = core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0
                               ? AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE
                               : AGX_IOTO_LOAD_GPU1_COPY_MAX_SIZE;

   for (size_t i = 0; i < core->load_count; ++i) {
      if (!core->loads[i].processed && !add_size(&total, load_size))
         return false;
      if (complete_round) {
         uint8_t mask = core->loads[i].lane_mask;
         unsigned tps = core->loads[i].cprog_input_tps_half;
         for (uint8_t dword = 0; dword < core->loads[i].count_dwords;
              ++dword, mask >>= 2) {
            if ((mask & 3) != 0 && !add_size(&total, copy_size))
               return false;
            unsigned inputs = (core->loads[i].cprog_input_mask >> (2 * dword)) & 3;
            unsigned halves = (inputs & 1) + (inputs >> 1);
            unsigned copies = inputs == 3 && !(tps & 1) ? 1 : halves;
            if (!add_size(&total, copies * copy_size))
               return false;
            tps += halves;
         }
      }
   }

   if (complete_round && core->gpu0_round_nops &&
       !add_size(&total, AGX_IOTO_LOAD_GPU0_ROUND_NOP_SIZE))
      return false;
   *required = total;
   return true;
}

bool
agx_ioto_load_finish_worst_case_size_with_ops(
   const struct agx_ioto_load_core *core, bool complete_round,
   const struct agx_ioto_load_emit_ops *ops, size_t *required)
{
   size_t total;
   if (required != NULL)
      *required = 0;
   if (required == NULL ||
       !agx_ioto_load_finish_worst_case_size(core, complete_round, &total))
      return false;

   if (ops != NULL &&
       ((ops->extra_copy_bound == NULL) != (ops->pack_extra_copies == NULL)))
      return false;
   if (ops != NULL && ops->pack_extra_copies != NULL) {
      if (core->chip != AGX_CHIP_G17P || core->fast_path_enabled ||
          !core->gpu0_round_nops)
         return false;
      if (complete_round) {
         for (size_t i = 0; i < core->load_count; ++i) {
            size_t extra;
            if (!ops->extra_copy_bound(ops->context, core->chip, &core->loads[i],
                                       &extra) ||
                extra > 2u * core->loads[i].count_dwords *
                           AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE ||
                !add_size(&total, extra))
               return false;
         }
      }
   }
   *required = total;
   return true;
}

static enum agx_ioto_kind
raw_ioto_kind(enum agx_ioto_load_raw_kind kind)
{
   switch (kind) {
   case AGX_IOTO_LOAD_RAW_LDDEV_EP0:
      return AGX_IOTO_KIND_LDDEV_EP0;
   case AGX_IOTO_LOAD_RAW_LDDEV_EP1:
      return AGX_IOTO_KIND_LDDEV_EP1;
   case AGX_IOTO_LOAD_RAW_LDDEV_EP2:
      return AGX_IOTO_KIND_LDDEV_EP2;
   case AGX_IOTO_LOAD_RAW_LDDEV_EP3:
      return AGX_IOTO_KIND_LDDEV_EP3;
   case AGX_IOTO_LOAD_RAW_LDDEV_EP4:
      return AGX_IOTO_KIND_LDDEV_EP4;
   case AGX_IOTO_LOAD_RAW_LDDEV_EP5:
      return AGX_IOTO_KIND_LDDEV_EP5;
   case AGX_IOTO_LOAD_RAW_COPY_EP0:
      return AGX_IOTO_KIND_COPY_EP0;
   case AGX_IOTO_LOAD_RAW_COPY_EP1:
      return AGX_IOTO_KIND_COPY_EP1;
   default:
      return AGX_IOTO_KIND_INVALID;
   }
}

static bool
raw_record_valid(const struct agx_ioto_load_core *core,
                 const struct agx_ioto_load_raw_record *raw, bool is_load,
                 bool trusted)
{
   if (raw == NULL)
      return false;

   if (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 && is_load) {
      return raw->kind == AGX_IOTO_GPU0_INLINE_LDDEV14 &&
             raw->length == AGX_IOTO_LOAD_INLINE_LDDEV14_SIZE &&
             (load_u16(raw->bytes) & UINT16_C(0x07ff)) == UINT16_C(0x0067);
   }

   if (is_load) {
      if (core->encoding != AGX_IOTO_LOAD_ENCODING_GPU1 ||
          raw->kind < AGX_IOTO_LOAD_RAW_LDDEV_EP0 ||
          raw->kind > AGX_IOTO_LOAD_RAW_LDDEV_EP5)
         return false;
   } else if (raw->kind != AGX_IOTO_LOAD_RAW_COPY_EP0 &&
              raw->kind != AGX_IOTO_LOAD_RAW_COPY_EP1) {
      return false;
   }

   if (trusted) {
      if (is_load) {
         switch (raw->kind) {
         case AGX_IOTO_LOAD_RAW_LDDEV_EP0:
            return raw->length == 8;
         case AGX_IOTO_LOAD_RAW_LDDEV_EP1:
         case AGX_IOTO_LOAD_RAW_LDDEV_EP3:
            return raw->length == 10;
         case AGX_IOTO_LOAD_RAW_LDDEV_EP2:
            return raw->length == 14;
         case AGX_IOTO_LOAD_RAW_LDDEV_EP4:
            return raw->length == 12;
         case AGX_IOTO_LOAD_RAW_LDDEV_EP5:
            return raw->length == 16;
         default:
            return false;
         }
      }

      if (raw->kind == AGX_IOTO_LOAD_RAW_COPY_EP0)
         return raw->length == 4;
      if (raw->kind == AGX_IOTO_LOAD_RAW_COPY_EP1)
         return raw->length ==
                (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 ? 10 : 8);
      return false;
   }

   const enum agx_ioto_kind kind = raw_ioto_kind(raw->kind);
   struct agx_ioto_record validated;
   size_t consumed = 0;
   return kind != AGX_IOTO_KIND_INVALID &&
          agx_ioto_validate_kind(core->chip, kind, raw->bytes, raw->length,
                                 &validated, &consumed) &&
          consumed == raw->length;
}

static bool
append_bytes(struct agx_ioto_load_core *core, const uint8_t *bytes,
             size_t length)
{
   if (length > core->sink.capacity - core->sink.length)
      return false;
   memcpy(&core->sink.data[core->sink.length], bytes, length);
   core->sink.length += length;
   return true;
}

static struct agx_ioto_inflight_load *
find_sequence(struct agx_ioto_load_core *core, uint32_t sequence)
{
   for (size_t i = 0; i < core->load_count; ++i) {
      if (core->loads[i].sequence == sequence)
         return &core->loads[i];
   }
   return NULL;
}

bool
agx_ioto_load_finish_round(struct agx_ioto_load_core *core, bool complete_round,
                           const struct agx_ioto_load_emit_ops *ops,
                           struct agx_ioto_load_workspace workspace)
{
   if (core == NULL || ops == NULL || ops->pack_load == NULL ||
       (complete_round && ops->pack_copy == NULL) ||
       workspace.capacity < core->load_count ||
       (core->load_count != 0 && workspace.loads == NULL))
      return false;

   size_t required;
   if (!agx_ioto_load_finish_worst_case_size_with_ops(core, complete_round, ops,
                                                    &required) ||
       required > core->sink.capacity - core->sink.length)
      return false;

   const size_t old_length = core->sink.length;
   const size_t old_copy_suffix_offset = core->copy_suffix_offset;
   if (core->load_count != 0) {
      memcpy(workspace.loads, core->loads,
             core->load_count * sizeof(workspace.loads[0]));
   }
   sort_loads(workspace.loads, core->load_count);

   for (size_t i = 0; i < core->load_count; ++i) {
      struct agx_ioto_inflight_load *load = &workspace.loads[i];
      if (load->processed)
         continue;

      struct agx_ioto_load_raw_record raw;
      uint8_t slot = 0;
      memset(&raw, 0, sizeof(raw));
      if (!ops->pack_load(ops->context, core->chip, load, &raw, &slot) ||
          slot == 0 ||
          slot > (core->encoding == AGX_IOTO_LOAD_ENCODING_GPU0 ? 6 : 8) ||
          !raw_record_valid(core, &raw, true, ops->trusted_records) ||
          !append_bytes(core, raw.bytes, raw.length)) {
         core->sink.length = old_length;
         return false;
      }
      load->signal_slot = slot;
      load->processed = true;
   }

   if (complete_round && core->gpu0_round_nops) {
      static const uint8_t hazard_for_value[6] = {1, 2, 4, 8, 16, 32};
      uint8_t hazard = 0;
      for (size_t i = 0; i < core->load_count; ++i) {
         const int32_t value = workspace.loads[i].data.value;
         if (value >= 0 && value < 6)
            hazard |= hazard_for_value[value];
      }

      uint8_t nop[AGX_IOTO_LOAD_NOP_SIZE];
      agx_ioto_load_pack_nop(hazard, nop);
      if (!append_bytes(core, nop, sizeof(nop)))
         goto fail;
      agx_ioto_load_pack_nop(0, nop);
      for (unsigned i = 0; i < 3; ++i) {
         if (!append_bytes(core, nop, sizeof(nop)))
            goto fail;
      }
   }

   if (complete_round) {
      core->copy_suffix_offset = core->sink.length;
      for (size_t i = 0; i < core->load_count; ++i) {
         const struct agx_ioto_inflight_load *load = &workspace.loads[i];
         uint8_t mask = load->lane_mask;
         for (uint8_t dword = 0; dword < load->count_dwords;
              ++dword, mask >>= 2) {
            const uint8_t lane_code = mask & 3;
            if (lane_code == 0)
               continue;

            struct agx_ioto_load_raw_record raw;
            memset(&raw, 0, sizeof(raw));
            if (!ops->pack_copy(ops->context, core->chip, load, dword,
                                lane_code, &raw) ||
                !raw_record_valid(core, &raw, false,
                                  ops->trusted_records) ||
                !append_bytes(core, raw.bytes, raw.length))
               goto fail;

            if ((load->cprog_input_mask >> (2 * dword)) & 3) {
               struct agx_ioto_load_raw_record extra[2] = {0};
               uint8_t count = 0;
               if (!ops->pack_cprog_copy ||
                   !ops->pack_cprog_copy(ops->context, core->chip, load,
                                          dword, extra, &count) ||
                   count == 0 || count > 2)
                  goto fail;
               for (unsigned e = 0; e < count; ++e) {
                  if (!raw_record_valid(core, &extra[e], false,
                                         ops->trusted_records) ||
                      !append_bytes(core, extra[e].bytes, extra[e].length))
                     goto fail;
               }
            }
            if (ops->pack_extra_copies != NULL) {
               struct agx_ioto_load_raw_record extra[2] = {0};
               uint8_t count = 0;
               if (!ops->pack_extra_copies(ops->context, core->chip, load,
                                           dword, extra, &count) || count > 2)
                  goto fail;
               for (unsigned e = 0; e < count; ++e) {
                  if (!raw_record_valid(core, &extra[e], false, ops->trusted_records) ||
                      !append_bytes(core, extra[e].bytes, extra[e].length))
                     goto fail;
               }
            }
         }
      }

      core->current_gpr_dwords = core->initial_gpr_dwords;
      core->load_count = 0;
      core->abi4_load_pending = false;
      /*
       * Apple destroys both EslBuilder address maps at a complete round
       * boundary.  In particular, absolute-address base registers may be
       * reused immediately after current_gpr_dwords is reset.
       */
      if (core->fixed_base_valid) {
         assert(core->cache_capacity > 0);
         core->cache[0] = (struct agx_ioto_load_cache_entry){
            .address = core->fixed_base_address,
            .base_half = core->fixed_base_half,
         };
         core->cache_count = 1;
      } else {
         core->cache_count = 0;
      }
   } else {
      for (size_t i = 0; i < core->load_count; ++i) {
         struct agx_ioto_inflight_load *original =
            find_sequence(core, workspace.loads[i].sequence);
         if (original == NULL)
            goto fail;
         original->processed = workspace.loads[i].processed;
         original->signal_slot = workspace.loads[i].signal_slot;
      }
   }

   return true;

fail:
   core->sink.length = old_length;
   core->copy_suffix_offset = old_copy_suffix_offset;
   return false;
}

static size_t
cache_lower_bound(const struct agx_ioto_load_core *core, uint64_t address)
{
   size_t lo = 0, hi = core->cache_count;
   while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      if (core->cache[mid].address < address)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo;
}

static bool
cache_lookup(const struct agx_ioto_load_core *core, uint64_t address,
             size_t *insert, uint16_t *base_half, int16_t *delta)
{
   *insert = cache_lower_bound(core, address);
   if (*insert < core->cache_count && core->cache[*insert].address == address) {
      *base_half = core->cache[*insert].base_half;
      *delta = 0;
      return true;
   }
   if (*insert == 0)
      return false;

   const struct agx_ioto_load_cache_entry *previous = &core->cache[*insert - 1];
   const uint64_t difference = address - previous->address;
   if (difference > INT16_MAX)
      return false;
   *base_half = previous->base_half;
   *delta = (int16_t)difference;
   return true;
}

bool
agx_ioto_load_absolute(struct agx_ioto_load_core *core,
                       const struct agx_ioto_load_absolute *load,
                       const struct agx_ioto_load_emit_ops *ops,
                       struct agx_ioto_load_workspace workspace,
                       struct agx_ioto_load_allocation *allocation)
{
   if (allocation != NULL)
      memset(allocation, 0, sizeof(*allocation));
   if (core == NULL || load == NULL || allocation == NULL ||
       load->count_dwords == 0 || load->count_dwords > 4 ||
       load->native_slot > UINT16_MAX / 2 || load->lane_mask == 0 ||
       core->next_sequence == UINT32_MAX)
      return false;

   if (load->cprog_input_mask &&
       ((!agx_chip_native_preamble(core->chip) &&
         !agx_chip_g17_m5(core->chip)) ||
        !ops || !ops->pack_cprog_copy ||
        (load->cprog_input_mask & ~load->lane_mask) ||
        (load->cprog_input_mask >> (2 * load->count_dwords)) ||
        load->cprog_input_tps_half < 8 || load->cprog_input_tps_half >= 520))
      return false;
   if (load->cprog_input_mask) {
      unsigned inputs = 0;
      for (unsigned h = 0; h < 8; ++h)
         inputs += (load->cprog_input_mask >> h) & 1;
      if (inputs > 520u - load->cprog_input_tps_half)
         return false;
   }

   const uint8_t n = load->count_dwords > 2 ? load->count_dwords : 2;
   const uint8_t valid_lane_bits = (uint8_t)((1u << (2u * n)) - 1);
   if ((load->lane_mask & ~valid_lane_bits) != 0)
      return false;

   /*
    * Reserve the 64-bit base register AND the loaded data separately.
    *
    * An absolute load needs two disjoint register regions: two halves' worth
    * of 32-bit LDIMM slots holding the address (`temp_half`, `temp_half + 2`),
    * and `count_dwords` dwords receiving the LDDEV result.  This function used
    * to hand `allocation->temp_half` to BOTH -- `queued->base.half` and
    * `queued->data.half` were literally the same register -- so the LDDEV
    * wrote its result over the address it had just been given.  For the
    * Honeykrisp root-descriptor load that is r0:r1 holding 0x2ffff6c540 and a
    * four-dword LDDEV landing on halves 0..7, i.e. the base is destroyed by
    * the very load that reads it, and the loaded value is itself a pointer the
    * shader then dereferences.
    *
    * Reserving `count_dwords + 2` and putting the base at the bottom of the
    * reservation with the data two dwords above it makes the two disjoint.
    * This is correct whether or not the hardware happened to tolerate the
    * overlay (a read-then-write LDDEV would), so it is safe either way, and it
    * matches every witness: neither the G17P add3 preamble (data at halves
    * 0x24/0x2c/0x30) nor the G17S minimal witness (0x14/0x16/0x18) ever lands
    * loaded data on its address registers.
    */
   if (load->count_dwords > UINT8_MAX - 2)
      return false;
   const uint8_t requested_dwords = (uint8_t)(load->count_dwords + 2u);

   /* Predict the allocation without mutating caller-owned state. */
   struct agx_ioto_load_core predicted_core = *core;
   enum agx_ioto_load_alloc_result result = agx_ioto_load_allocator_reserve(
      &predicted_core, requested_dwords, load->cprog_dest_half,
      load->lane_mask, allocation);
   if (result == AGX_IOTO_LOAD_ALLOC_INVALID)
      return false;

   const bool needs_round = result == AGX_IOTO_LOAD_ALLOC_NEEDS_ROUND;
   if (needs_round) {
      predicted_core.current_gpr_dwords = core->initial_gpr_dwords;
      result = agx_ioto_load_allocator_reserve(
         &predicted_core, requested_dwords, load->cprog_dest_half,
         load->lane_mask, allocation);
      if (result != AGX_IOTO_LOAD_ALLOC_OK)
         return false;
   }

   /* Base at the bottom of the reservation, data two dwords (four halves)
    * above it. */
   const uint16_t base_temp_half = allocation->temp_half;
   if (base_temp_half > UINT16_MAX - 4)
      return false;
   const uint16_t data_temp_half = (uint16_t)(base_temp_half + 4u);

   /*
    * Apple performs rollover before consulting the address map and destroys
    * that map in finishRound(true).  A post-rollover request is therefore an
    * unconditional cache miss.
    */
   size_t cache_insert = 0;
   uint16_t base_half = 0;
   int16_t delta = 0;
   bool cache_hit = false;
   if (!needs_round) {
      cache_hit = cache_lookup(core, load->address, &cache_insert,
                               &base_half, &delta);
   } else if (core->fixed_base_valid &&
              load->address >= core->fixed_base_address &&
              load->address - core->fixed_base_address <= INT16_MAX) {
      /* Ordinary round-owned bases are invalid after finishRound(true), but
       * the launcher-owned prefix is deliberately reserved and re-seeded. */
      base_half = core->fixed_base_half;
      delta = (int16_t)(load->address - core->fixed_base_address);
      cache_hit = true;
   }
   if (!cache_hit &&
       (core->cache_capacity == 0 ||
        (!needs_round && core->cache_count == core->cache_capacity)))
      return false;
   if (!cache_hit && core->relocations &&
       core->relocation_count == core->relocation_capacity)
      return false;

   uint8_t low[AGX_IOTO_LOAD_LDIMM_MAX_SIZE],
      high[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   uint8_t low_length = 0, high_length = 0;
   if (!cache_hit && core->long_ldimm) {
      agx_ioto_load_pack_ldimm_long(base_temp_half, (uint32_t)load->address,
                                    low);
      agx_ioto_load_pack_ldimm_long((uint16_t)(base_temp_half + 2),
                                    (uint32_t)(load->address >> 32), high);
      low_length = high_length = 8;
   } else if (!cache_hit &&
              (!agx_ioto_load_pack_ldimm(core->chip, base_temp_half,
                                         (uint32_t)load->address, low,
                                         &low_length) ||
               !agx_ioto_load_pack_ldimm(
                  core->chip, (uint16_t)(base_temp_half + 2),
                  (uint32_t)(load->address >> 32), high, &high_length)))
      return false;

   size_t finish_size = 0;
   if (needs_round &&
       (!agx_ioto_load_finish_worst_case_size_with_ops(core, true, ops, &finish_size) ||
        workspace.capacity < core->load_count || ops == NULL))
      return false;
   size_t append_size = finish_size;
   if (!add_size(&append_size, low_length) ||
       !add_size(&append_size, high_length))
      return false;
   if (append_size > core->sink.capacity - core->sink.length)
      return false;
   if ((needs_round ? 1 : core->load_count + 1) > core->load_capacity)
      return false;

   const size_t old_length = core->sink.length;
   const size_t old_load_count = core->load_count;
   const size_t old_cache_count = core->cache_count;
   const size_t old_relocation_count = core->relocation_count;
   const uint8_t old_current = core->current_gpr_dwords;
   const uint32_t old_next_sequence = core->next_sequence;

   if (needs_round && !agx_ioto_load_finish_round(core, true, ops, workspace))
      return false;

   core->current_gpr_dwords = predicted_core.current_gpr_dwords;

   if (!cache_hit) {
      const size_t relocation_offset = core->sink.length;
      if (!append_bytes(core, low, low_length) ||
          !append_bytes(core, high, high_length))
         goto rollback;

      if (core->relocations) {
         core->relocations[core->relocation_count++] =
            (struct agx_ioto_load_relocation){
               .offset = relocation_offset,
               .address = load->address,
               .dest_half = base_temp_half,
               .low_length = low_length,
               .high_length = high_length,
            };
      }

      /* A complete round clears ordinary entries.  A launcher-owned fixed
       * base is immediately re-seeded and remains the first sorted entry. */
      if (needs_round)
         cache_insert = cache_lower_bound(core, load->address);

      memmove(&core->cache[cache_insert + 1], &core->cache[cache_insert],
              (core->cache_count - cache_insert) * sizeof(core->cache[0]));
      core->cache[cache_insert].address = load->address;
      core->cache[cache_insert].base_half = base_temp_half;
      ++core->cache_count;
      base_half = base_temp_half;
      delta = 0;
   }

   struct agx_ioto_inflight_load *queued = &core->loads[core->load_count++];
   memset(queued, 0, sizeof(*queued));
   queued->data.half = data_temp_half;
   queued->data.type = 3;
   queued->data.value = 1;
   queued->base.half = base_half;
   queued->base.type = 4;
   queued->base.value = 8;
   queued->byte_offset = delta;
   queued->count_dwords = load->count_dwords;
   queued->native_slot_half = (uint16_t)(2u * load->native_slot);
   queued->lane_mask = load->lane_mask;
   queued->cprog_dest_half = load->cprog_dest_half;
   queued->cprog_input_mask = load->cprog_input_mask;
   queued->cprog_input_tps_half = load->cprog_input_tps_half;
   queued->sequence = core->next_sequence++;
   return true;

rollback:
   core->sink.length = old_length;
   core->load_count = old_load_count;
   core->cache_count = old_cache_count;
   core->relocation_count = old_relocation_count;
   core->current_gpr_dwords = old_current;
   core->next_sequence = old_next_sequence;
   return false;
}

static bool
abi_buffer_base_half(enum agx_ioto_abi_buffer_slot source, uint16_t *half)
{
   switch (source) {
   case AGX_IOTO_ABI_BUFFER_SLOT_2:
   case AGX_IOTO_ABI_BUFFER_SLOT_4:
   case AGX_IOTO_ABI_BUFFER_SLOT_6:
   case AGX_IOTO_ABI_BUFFER_SLOT_8:
   case AGX_IOTO_ABI_BUFFER_SLOT_50:
   case AGX_IOTO_ABI_BUFFER_SLOT_52:
   case AGX_IOTO_ABI_BUFFER_SLOT_54:
      *half = (uint16_t)(2u * (unsigned)source);
      return true;
   default:
      return false;
   }
}

static bool
load_from_fixed(struct agx_ioto_load_core *core,
                const struct agx_ioto_load_from_fixed *load, uint16_t base_half,
                bool is_scs, enum agx_ioto_abi_buffer_slot abi_source,
                const struct agx_ioto_load_emit_ops *ops,
                struct agx_ioto_load_workspace workspace,
                struct agx_ioto_load_allocation *allocation)
{
   if (allocation != NULL)
      memset(allocation, 0, sizeof(*allocation));
   if (core == NULL || load == NULL || allocation == NULL ||
       load->count_dwords == 0 || load->count_dwords > 4 ||
       load->native_slot > UINT16_MAX / 2 || load->lane_mask == 0 ||
       core->next_sequence == UINT32_MAX)
      return false;

   /* half-register 8 is both Apple's lazily materialized SCS base and ABI
    * buffer selector 4.  The native builder only flips its cache byte; it
    * neither flushes a queued ABI4 load before LDIMM nor restores ABI4 after
    * LDIMM.  Keep both caller-ordering hazards closed until that contract is
    * grounded at a live producer. */
   if ((is_scs && core->abi4_load_pending) ||
       (!is_scs && abi_source == AGX_IOTO_ABI_BUFFER_SLOT_4 &&
        core->scs_base_ever_loaded))
      return false;

   const uint8_t n = load->count_dwords > 2 ? load->count_dwords : 2;
   const uint8_t valid_lane_bits = (uint8_t)((1u << (2u * n)) - 1u);
   if ((load->lane_mask & ~valid_lane_bits) != 0)
      return false;

   struct agx_ioto_load_core predicted_core = *core;
   enum agx_ioto_load_alloc_result result = agx_ioto_load_allocator_reserve(
      &predicted_core, load->count_dwords, load->cprog_dest_half,
      load->lane_mask, allocation);
   if (result == AGX_IOTO_LOAD_ALLOC_INVALID)
      return false;

   const bool needs_round = result == AGX_IOTO_LOAD_ALLOC_NEEDS_ROUND;
   if (needs_round) {
      predicted_core.current_gpr_dwords = core->initial_gpr_dwords;
      result = agx_ioto_load_allocator_reserve(
         &predicted_core, load->count_dwords, load->cprog_dest_half,
         load->lane_mask, allocation);
      if (result != AGX_IOTO_LOAD_ALLOC_OK)
         return false;
   }

   uint8_t scs_base[AGX_IOTO_LOAD_LDIMM_MAX_SIZE];
   uint8_t scs_base_length = 0;
   if (is_scs && !core->scs_base_loaded &&
       !agx_ioto_load_pack_ldimm(core->chip, AGX_IOTO_LOAD_SCS_BASE_HALF,
                                 AGX_IOTO_LOAD_SCS_BASE, scs_base,
                                 &scs_base_length))
      return false;

   size_t finish_size = 0;
   if (needs_round &&
       (!agx_ioto_load_finish_worst_case_size_with_ops(core, true, ops, &finish_size) ||
        workspace.capacity < core->load_count || ops == NULL))
      return false;
   size_t append_size = finish_size;
   if (!add_size(&append_size, scs_base_length) ||
       append_size > core->sink.capacity - core->sink.length ||
       (needs_round ? 1 : core->load_count + 1) > core->load_capacity)
      return false;

   const struct agx_ioto_load_core before = *core;
   if (needs_round && !agx_ioto_load_finish_round(core, true, ops, workspace))
      return false;

   core->current_gpr_dwords = predicted_core.current_gpr_dwords;
   if (scs_base_length != 0 && !append_bytes(core, scs_base, scs_base_length))
      goto rollback;

   struct agx_ioto_inflight_load *queued = &core->loads[core->load_count++];
   memset(queued, 0, sizeof(*queued));
   queued->data.half = allocation->temp_half;
   queued->data.type = 3;
   queued->data.value = 1;
   queued->base.half = base_half;
   queued->base.type = 4;
   queued->base.value = 8;
   queued->byte_offset = load->byte_offset;
   queued->count_dwords = load->count_dwords;
   queued->native_slot_half = (uint16_t)(2u * load->native_slot);
   queued->lane_mask = load->lane_mask;
   queued->cprog_dest_half = load->cprog_dest_half;
   queued->sequence = core->next_sequence++;

   if (is_scs) {
      core->scs_base_loaded = true;
      core->scs_base_ever_loaded = true;
   } else if (abi_source == AGX_IOTO_ABI_BUFFER_SLOT_4) {
      core->scs_base_loaded = false;
      core->abi4_load_pending = true;
   }
   return true;

rollback:
   *core = before;
   memset(allocation, 0, sizeof(*allocation));
   return false;
}

bool
agx_ioto_load_from_scs(struct agx_ioto_load_core *core,
                       const struct agx_ioto_load_from_fixed *load,
                       const struct agx_ioto_load_emit_ops *ops,
                       struct agx_ioto_load_workspace workspace,
                       struct agx_ioto_load_allocation *allocation)
{
   return load_from_fixed(core, load, AGX_IOTO_LOAD_SCS_BASE_HALF, true,
                          (enum agx_ioto_abi_buffer_slot)0, ops, workspace,
                          allocation);
}

bool
agx_ioto_load_from_abi_buffer(struct agx_ioto_load_core *core,
                              enum agx_ioto_abi_buffer_slot source,
                              const struct agx_ioto_load_from_fixed *load,
                              const struct agx_ioto_load_emit_ops *ops,
                              struct agx_ioto_load_workspace workspace,
                              struct agx_ioto_load_allocation *allocation)
{
   uint16_t base_half;
   if (!abi_buffer_base_half(source, &base_half)) {
      if (allocation != NULL)
         memset(allocation, 0, sizeof(*allocation));
      return false;
   }
   return load_from_fixed(core, load, base_half, false, source, ops, workspace,
                          allocation);
}
