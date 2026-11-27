/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "../agx_ioto_load_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         return false;                                                         \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static const enum agx_chip gpu0_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
};

static const enum agx_chip gpu1_targets[] = {
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,
   AGX_CHIP_G17D,
};

static const uint8_t gpu0_lddev14[] = {
   0x67, 0x00, 0x54, 0x00, 0x01, 0x01, 0x20,
   0x00, 0x51, 0x01, 0x00, 0x40, 0x46, 0x00,
};
static const uint8_t gpu1_lddev0[] = {
   0x0f, 0x04, 0x03, 0x00, 0x18, 0x04, 0x90, 0x20,
};
static const uint8_t gpu0_copy1[] = {
   0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x60, 0x00,
};
static const uint8_t gpu1_copy1[] = {
   0x23, 0x00, 0x07, 0x02, 0x42, 0x20, 0xa0, 0x02,
};

struct emit_context {
   unsigned loads;
   unsigned copies;
   unsigned fail_load;
   unsigned fail_copy;
   uint32_t order[16];
};

static bool
pack_load(void *opaque, enum agx_chip chip,
          const struct agx_ioto_inflight_load *load,
          struct agx_ioto_load_raw_record *record, uint8_t *signal_slot)
{
   struct emit_context *ctx = opaque;
   ++ctx->loads;
   if (ctx->loads == ctx->fail_load)
      return false;
   ctx->order[ctx->loads - 1] = load->sequence;
   *signal_slot = 1;

   if (agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0) {
      record->kind = AGX_IOTO_GPU0_INLINE_LDDEV14;
      record->length = sizeof(gpu0_lddev14);
      memcpy(record->bytes, gpu0_lddev14, sizeof(gpu0_lddev14));
   } else {
      record->kind = AGX_IOTO_LOAD_RAW_LDDEV_EP0;
      record->length = sizeof(gpu1_lddev0);
      memcpy(record->bytes, gpu1_lddev0, sizeof(gpu1_lddev0));
   }
   return true;
}

static bool
pack_copy(void *opaque, enum agx_chip chip,
          const struct agx_ioto_inflight_load *load, uint8_t dword,
          uint8_t lane_code, struct agx_ioto_load_raw_record *record)
{
   struct emit_context *ctx = opaque;
   (void)load;
   (void)dword;
   (void)lane_code;
   ++ctx->copies;
   if (ctx->copies == ctx->fail_copy)
      return false;

   record->kind = AGX_IOTO_LOAD_RAW_COPY_EP1;
   if (agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0) {
      record->length = sizeof(gpu0_copy1);
      memcpy(record->bytes, gpu0_copy1, sizeof(gpu0_copy1));
   } else {
      record->length = sizeof(gpu1_copy1);
      memcpy(record->bytes, gpu1_copy1, sizeof(gpu1_copy1));
   }
   return true;
}

static struct agx_ioto_load_emit_ops
ops(struct emit_context *ctx)
{
   return (struct agx_ioto_load_emit_ops){
      .pack_load = pack_load,
      .pack_copy = pack_copy,
      .context = ctx,
   };
}

struct fixture {
   uint8_t bytes[8192];
   struct agx_ioto_inflight_load loads[128];
   struct agx_ioto_inflight_load scratch[128];
   struct agx_ioto_load_cache_entry cache[128];
   struct agx_ioto_load_core core;
};

static bool
fixture_init(struct fixture *f, enum agx_chip chip, uint8_t initial, bool fast,
             bool nops)
{
   memset(f, 0, sizeof(*f));
   return agx_ioto_load_core_init(
      &f->core, chip,
      (struct agx_ioto_load_sink){f->bytes, sizeof(f->bytes), 0}, f->loads,
      ARRAY_SIZE(f->loads), f->cache, ARRAY_SIZE(f->cache), initial, fast,
      nops);
}

static struct agx_ioto_load_workspace
workspace(struct fixture *f)
{
   return (struct agx_ioto_load_workspace){f->scratch, ARRAY_SIZE(f->scratch)};
}

static bool
test_targets(void)
{
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      CHECK(agx_ioto_load_target_encoding(gpu0_targets[i]) ==
            AGX_IOTO_LOAD_ENCODING_GPU0);
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      CHECK(agx_ioto_load_target_encoding(gpu1_targets[i]) ==
            AGX_IOTO_LOAD_ENCODING_GPU1);
   }
   CHECK(agx_ioto_load_target_encoding(AGX_CHIP_G14G) ==
         AGX_IOTO_LOAD_ENCODING_UNSUPPORTED);
   CHECK(agx_ioto_load_target_encoding(AGX_CHIP_G15P) ==
         AGX_IOTO_LOAD_ENCODING_UNSUPPORTED);
   return true;
}

static bool
test_ldimm_nop(void)
{
   uint8_t bytes[8], length;
   static const uint8_t long_witness[] = {
      0x0c, 0xf8, 0x02, 0x12, 0x18, 0x08, 0xa2, 0x01,
   };
   CHECK(
      agx_ioto_load_pack_ldimm(AGX_CHIP_G15G, 0, 0x12345678, bytes, &length));
   CHECK(length == sizeof(long_witness));
   CHECK(memcmp(bytes, long_witness, sizeof(long_witness)) == 0);
   CHECK(
      agx_ioto_load_pack_ldimm(AGX_CHIP_G17G, 0, 0x12345678, bytes, &length));
   CHECK(memcmp(bytes, long_witness, sizeof(long_witness)) == 0);
   CHECK(agx_ioto_load_pack_ldimm(AGX_CHIP_G15G, 0, 0x20, bytes, &length));
   CHECK(length == 2 && bytes[0] == 0x0c && bytes[1] == 0x20);
   CHECK(agx_ioto_load_pack_ldimm(AGX_CHIP_G17G, 0, 0x20, bytes, &length));
   CHECK(length == 2 && bytes[0] == 0x04 && bytes[1] == 0x20);
   CHECK(!agx_ioto_load_pack_ldimm(AGX_CHIP_G17G, 1, 0, bytes, &length));
   static const uint8_t scs_base[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
   };
   CHECK(agx_ioto_load_pack_ldimm(AGX_CHIP_G15G, AGX_IOTO_LOAD_SCS_BASE_HALF,
                                  AGX_IOTO_LOAD_SCS_BASE, bytes, &length));
   CHECK(length == sizeof(scs_base) &&
         memcmp(bytes, scs_base, sizeof(scs_base)) == 0);
   CHECK(agx_ioto_load_pack_ldimm(AGX_CHIP_G17G, AGX_IOTO_LOAD_SCS_BASE_HALF,
                                  AGX_IOTO_LOAD_SCS_BASE, bytes, &length));
   CHECK(length == sizeof(scs_base) &&
         memcmp(bytes, scs_base, sizeof(scs_base)) == 0);

   uint8_t nop[2];
   agx_ioto_load_pack_nop(0x2a, nop);
   CHECK(nop[0] == 0x06 && nop[1] == 0xa8);
   agx_ioto_load_pack_nop(0, nop);
   CHECK(nop[0] == 0x06 && nop[1] == 0x00);
   return true;
}

static struct agx_ioto_load_from_fixed
fixed_load(uint16_t destination)
{
   return (struct agx_ioto_load_from_fixed){
      .native_slot = 7,
      .count_dwords = 2,
      .byte_offset = 0x123,
      .lane_mask = 0xf,
      .cprog_dest_half = destination,
   };
}

static bool
test_direct_sources(void)
{
   static const uint8_t scs_base[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
   };
   static const enum agx_ioto_abi_buffer_slot slots[] = {
      AGX_IOTO_ABI_BUFFER_SLOT_2,  AGX_IOTO_ABI_BUFFER_SLOT_4,
      AGX_IOTO_ABI_BUFFER_SLOT_6,  AGX_IOTO_ABI_BUFFER_SLOT_8,
      AGX_IOTO_ABI_BUFFER_SLOT_50, AGX_IOTO_ABI_BUFFER_SLOT_52,
      AGX_IOTO_ABI_BUFFER_SLOT_54,
   };
   struct emit_context ctx = {0};
   struct agx_ioto_load_emit_ops emit = ops(&ctx);
   struct fixture f;

   const enum agx_chip representatives[] = {AGX_CHIP_G15G, AGX_CHIP_G17G};
   for (size_t i = 0; i < ARRAY_SIZE(representatives); ++i) {
      CHECK(fixture_init(&f, representatives[i], 0, false, true));
      struct agx_ioto_load_from_fixed load = fixed_load(40);
      struct agx_ioto_load_allocation allocation;
      CHECK(agx_ioto_load_from_scs(&f.core, &load, &emit, workspace(&f),
                                   &allocation));
      CHECK(allocation.temp_half == 0 && allocation.reserved_dwords == 2);
      CHECK(f.core.sink.length == sizeof(scs_base) &&
            memcmp(f.bytes, scs_base, sizeof(scs_base)) == 0);
      CHECK(f.core.scs_base_loaded && f.core.load_count == 1);
      CHECK(f.core.loads[0].data.half == 0 &&
            f.core.loads[0].base.half == AGX_IOTO_LOAD_SCS_BASE_HALF &&
            f.core.loads[0].base.type == 4 && f.core.loads[0].base.value == 8 &&
            f.core.loads[0].byte_offset == 0x123 &&
            f.core.loads[0].native_slot_half == 14 &&
            f.core.loads[0].lane_mask == 0xf &&
            f.core.loads[0].cprog_dest_half == 40);

      load.cprog_dest_half = 44;
      CHECK(agx_ioto_load_from_scs(&f.core, &load, &emit, workspace(&f),
                                   &allocation));
      CHECK(f.core.sink.length == sizeof(scs_base));
      CHECK(f.core.loads[1].data.half == 4 &&
            f.core.loads[1].base.half == AGX_IOTO_LOAD_SCS_BASE_HALF);
   }

   for (size_t i = 0; i < ARRAY_SIZE(slots); ++i) {
      CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
      struct agx_ioto_load_from_fixed load = fixed_load(40);
      struct agx_ioto_load_allocation allocation;
      CHECK(agx_ioto_load_from_abi_buffer(&f.core, slots[i], &load, &emit,
                                          workspace(&f), &allocation));
      CHECK(f.core.sink.length == 0 && f.core.load_count == 1);
      CHECK(f.core.loads[0].base.half == 2u * (unsigned)slots[i]);
      CHECK(!f.core.scs_base_loaded && !f.core.scs_base_ever_loaded);
      CHECK(f.core.abi4_load_pending ==
            (slots[i] == AGX_IOTO_ABI_BUFFER_SLOT_4));
   }

   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
   struct agx_ioto_load_from_fixed load = fixed_load(40);
   struct agx_ioto_load_allocation allocation = {
      .temp_half = UINT16_MAX,
      .reserved_dwords = UINT8_MAX,
      .fast_path = true,
   };
   CHECK(!agx_ioto_load_from_abi_buffer(&f.core,
                                        (enum agx_ioto_abi_buffer_slot)3, &load,
                                        &emit, workspace(&f), &allocation));
   CHECK(f.core.load_count == 0 && f.core.sink.length == 0 &&
         allocation.temp_half == 0 && allocation.reserved_dwords == 0 &&
         !allocation.fast_path);

   CHECK(agx_ioto_load_from_abi_buffer(&f.core, AGX_IOTO_ABI_BUFFER_SLOT_4,
                                       &load, &emit, workspace(&f),
                                       &allocation));
   CHECK(f.core.abi4_load_pending && !f.core.scs_base_ever_loaded);
   load.cprog_dest_half = 44;
   CHECK(!agx_ioto_load_from_scs(&f.core, &load, &emit, workspace(&f),
                                 &allocation));
   CHECK(f.core.sink.length == 0 && f.core.load_count == 1 &&
         f.core.abi4_load_pending);
   CHECK(agx_ioto_load_finish_round(&f.core, true, &emit, workspace(&f)));
   CHECK(!f.core.abi4_load_pending);
   CHECK(agx_ioto_load_from_scs(&f.core, &load, &emit, workspace(&f),
                                &allocation));
   CHECK(f.core.sink.length >= sizeof(scs_base) && f.core.scs_base_loaded &&
         f.core.scs_base_ever_loaded);
   CHECK(!agx_ioto_load_from_abi_buffer(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_4, &load, &emit, workspace(&f),
      &allocation));
   CHECK(agx_ioto_load_from_abi_buffer(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_2, &load, &emit, workspace(&f),
      &allocation));

   const size_t old_loads = f.core.load_count;
   const uint8_t old_current = f.core.current_gpr_dwords;
   load.lane_mask = 0x80;
   CHECK(!agx_ioto_load_from_scs(&f.core, &load, &emit, workspace(&f),
                                 &allocation));
   CHECK(f.core.load_count == old_loads &&
         f.core.current_gpr_dwords == old_current);

   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
   f.core.sink.capacity = sizeof(scs_base) - 1;
   load = fixed_load(40);
   CHECK(!agx_ioto_load_from_scs(&f.core, &load, &emit, workspace(&f),
                                 &allocation));
   CHECK(f.core.sink.length == 0 && f.core.load_count == 0 &&
         f.core.current_gpr_dwords == 0 && !f.core.scs_base_loaded);
   return true;
}

static bool
test_allocator_boundaries(void)
{
   struct fixture f;
   struct agx_ioto_load_allocation a;
   CHECK(fixture_init(&f, AGX_CHIP_G15G, 0, false, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 2, 40, 0xf, &a) ==
         AGX_IOTO_LOAD_ALLOC_OK);
   CHECK(a.temp_half == 0 && a.reserved_dwords == 2 &&
         f.core.current_gpr_dwords == 2);
   CHECK(fixture_init(&f, AGX_CHIP_G15G, 0, false, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 96, 0, 0, &a) ==
         AGX_IOTO_LOAD_ALLOC_OK);
   CHECK(f.core.current_gpr_dwords == 96);
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 2, 0, 0, &a) ==
         AGX_IOTO_LOAD_ALLOC_NEEDS_ROUND);
   CHECK(f.core.current_gpr_dwords == 96);
   CHECK(fixture_init(&f, AGX_CHIP_G15G, 0, false, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 97, 0, 0, &a) ==
         AGX_IOTO_LOAD_ALLOC_INVALID);

   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 126, 0, 0, &a) ==
         AGX_IOTO_LOAD_ALLOC_OK);
   CHECK(f.core.current_gpr_dwords == 126);
   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 127, 0, 0, &a) ==
         AGX_IOTO_LOAD_ALLOC_INVALID);
   return true;
}

static bool
test_fast_path(void)
{
   struct fixture f;
   struct agx_ioto_load_allocation a;
   CHECK(fixture_init(&f, AGX_CHIP_G17G, 7, true, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 2, 40, 0xf, &a) ==
         AGX_IOTO_LOAD_ALLOC_OK);
   CHECK(a.fast_path && a.temp_half == 40 && f.core.current_gpr_dwords == 7);
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 2, 40, 0x7, &a) ==
         AGX_IOTO_LOAD_ALLOC_OK);
   CHECK(!a.fast_path && a.temp_half == 14 && f.core.current_gpr_dwords == 9);
   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, true, false));
   CHECK(agx_ioto_load_allocator_reserve(&f.core, 2, 146, 0xf, &a) ==
         AGX_IOTO_LOAD_ALLOC_OK);
   CHECK(!a.fast_path);
   return true;
}

static bool
queue_absolute(struct fixture *f, const struct agx_ioto_load_emit_ops *emit,
               uint64_t address, uint16_t dest)
{
   const struct agx_ioto_load_absolute load = {
      .native_slot = 3,
      .count_dwords = 2,
      .address = address,
      .lane_mask = 0xf,
      .cprog_dest_half = dest,
   };
   struct agx_ioto_load_allocation allocation;
   return agx_ioto_load_absolute(&f->core, &load, emit, workspace(f),
                                 &allocation);
}

static bool
test_cache_and_rollback(void)
{
   struct fixture f;
   struct emit_context ctx = {0};
   struct agx_ioto_load_emit_ops emit = ops(&ctx);
   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x123456789abc0000), 40));
   const size_t first_length = f.core.sink.length;
   CHECK(first_length == 16 && f.core.cache_count == 1 &&
         f.core.load_count == 1);
   CHECK(f.core.loads[0].byte_offset == 0);
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x123456789abc7fff), 44));
   CHECK(f.core.sink.length == first_length && f.core.cache_count == 1);
   CHECK(f.core.loads[1].byte_offset == 32767);
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x123456789abc8000), 48));
   CHECK(f.core.sink.length == first_length + 16 && f.core.cache_count == 2);
   CHECK(f.core.loads[2].byte_offset == 0);

   struct fixture full;
   CHECK(fixture_init(&full, AGX_CHIP_G17G, 0, false, false));
   full.core.cache_capacity = 1;
   CHECK(queue_absolute(&full, &emit, UINT64_C(0x10000), 40));
   const size_t full_length = full.core.sink.length;
   const size_t full_loads = full.core.load_count;
   const uint8_t full_current = full.core.current_gpr_dwords;
   CHECK(!queue_absolute(&full, &emit, UINT64_C(0x20000), 44));
   CHECK(full.core.sink.length == full_length &&
         full.core.load_count == full_loads && full.core.cache_count == 1 &&
         full.core.current_gpr_dwords == full_current);

   struct fixture small;
   CHECK(fixture_init(&small, AGX_CHIP_G17G, 0, false, false));
   small.core.sink.capacity = 15;
   memset(small.bytes, 0xa5, sizeof(small.bytes));
   CHECK(!queue_absolute(&small, &emit, UINT64_C(0x123456789abc0000), 40));
   CHECK(small.core.sink.length == 0 && small.core.cache_count == 0 &&
         small.core.load_count == 0 && small.core.current_gpr_dwords == 0);
   return true;
}

static bool
test_comparator(void)
{
   struct agx_ioto_inflight_load a = {
      .base = {10, 4, 8},
      .data = {20, 3, 1},
   };
   struct agx_ioto_inflight_load b = a;
   b.base.half = 12;
   CHECK(agx_ioto_load_compare(&a, &b) < 0);
   b = a;
   b.data.half = 18;
   CHECK(agx_ioto_load_compare(&a, &b) > 0);
   b = a;
   b.base.type = 5;
   CHECK(agx_ioto_load_compare(&a, &b) < 0);
   b = a;
   b.base.value = 9;
   CHECK(agx_ioto_load_compare(&a, &b) < 0);
   return true;
}

static bool
test_two_phase_and_nops(void)
{
   struct fixture f;
   struct emit_context ctx = {0};
   struct agx_ioto_load_emit_ops emit = ops(&ctx);
   CHECK(fixture_init(&f, AGX_CHIP_G15G, 0, false, true));
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x20000), 40));
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x10000), 44));
   /* Exercise the native tuple comparator, independent of queue order. */
   f.core.loads[0].base.half = 8;
   f.core.loads[1].base.half = 4;
   const size_t ldimms = f.core.sink.length;
   CHECK(agx_ioto_load_finish_round(&f.core, false, &emit, workspace(&f)));
   CHECK(ctx.loads == 2 && ctx.order[0] == 1 && ctx.order[1] == 0);
   CHECK(f.core.load_count == 2 && f.core.loads[0].processed &&
         f.core.loads[1].processed);
   const size_t after_loads = f.core.sink.length;
   CHECK(after_loads == ldimms + 2 * sizeof(gpu0_lddev14));
   CHECK(agx_ioto_load_finish_round(&f.core, false, &emit, workspace(&f)));
   CHECK(f.core.sink.length == after_loads && ctx.loads == 2);
   CHECK(agx_ioto_load_finish_round(&f.core, true, &emit, workspace(&f)));
   CHECK(f.core.load_count == 0 && f.core.current_gpr_dwords == 0);
   CHECK(f.core.cache_count == 0);
   CHECK(f.bytes[after_loads + 0] == 0x06 && f.bytes[after_loads + 1] == 0x08);
   CHECK(f.bytes[after_loads + 2] == 0x06 && f.bytes[after_loads + 3] == 0x00);
   CHECK(ctx.copies == 4);
   return true;
}

static bool
test_finish_rollback(void)
{
   struct fixture f;
   struct emit_context ctx = {0};
   struct agx_ioto_load_emit_ops emit = ops(&ctx);
   CHECK(fixture_init(&f, AGX_CHIP_G17G, 0, false, false));
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x10000), 40));
   const size_t old_length = f.core.sink.length;
   const uint8_t old_current = f.core.current_gpr_dwords;
   ctx.fail_copy = 2;
   CHECK(!agx_ioto_load_finish_round(&f.core, true, &emit, workspace(&f)));
   CHECK(f.core.sink.length == old_length && f.core.load_count == 1 &&
         f.core.current_gpr_dwords == old_current &&
         !f.core.loads[0].processed);

   memset(&ctx, 0, sizeof(ctx));
   emit = ops(&ctx);
   ctx.fail_load = 1;
   CHECK(!agx_ioto_load_finish_round(&f.core, false, &emit, workspace(&f)));
   CHECK(f.core.sink.length == old_length && !f.core.loads[0].processed);
   return true;
}

static bool
test_rollover(void)
{
   struct fixture f;
   struct emit_context ctx = {0};
   struct agx_ioto_load_emit_ops emit = ops(&ctx);
   /* Each absolute load now reserves count+2 dwords: two for the 64-bit
    * base and count for the loaded data, which must not overlap.  Start
    * two dwords lower so this still exercises the rollover boundary
    * rather than the allocator's hard limit. */
   CHECK(fixture_init(&f, AGX_CHIP_G15G, 92, false, false));
   f.core.cache_capacity = 1;
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x10000), 40));
   CHECK(f.core.current_gpr_dwords == 96 && f.core.load_count == 1);
   const size_t first_length = f.core.sink.length;
   CHECK(f.core.cache_count == 1);
   CHECK(queue_absolute(&f, &emit, UINT64_C(0x10000), 44));
   CHECK(ctx.loads == 1 && ctx.copies == 2);
   CHECK(f.core.load_count == 1 && f.core.current_gpr_dwords == 96 &&
         f.core.cache_count == 1);
   /* The same address is re-materialized after complete-round cache clear. */
   CHECK(f.core.sink.length ==
         first_length + sizeof(gpu0_lddev14) + 2 * sizeof(gpu0_copy1) + 16);
   return true;
}

static bool
test_cohort_parity(void)
{
   uint8_t reference[8];
   uint8_t reference_length = 0;
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      uint8_t bytes[8], length;
      CHECK(agx_ioto_load_pack_ldimm(gpu0_targets[i], 0, 42, bytes, &length));
      if (i == 0) {
         memcpy(reference, bytes, length);
         reference_length = length;
      } else {
         CHECK(length == reference_length &&
               memcmp(bytes, reference, length) == 0);
      }
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      uint8_t bytes[8], length;
      CHECK(agx_ioto_load_pack_ldimm(gpu1_targets[i], 0, 42, bytes, &length));
      if (i == 0) {
         memcpy(reference, bytes, length);
         reference_length = length;
      } else {
         CHECK(length == reference_length &&
               memcmp(bytes, reference, length) == 0);
      }
   }
   return true;
}

static bool
test_cprog_input_layout(void)
{
   /* Sparse native texture inputs [8,9,10,11] are packed into TPS halves
    * [8,9,10,11], rather than retaining their source uniform indices. */
   struct agx_ioto_cprog_inputs inputs = {.uniform_halfs = {0xf00}, .known = true};
   for (unsigned h = 8; h < 12; ++h) {
      uint16_t offset;
      CHECK(agx_ioto_cprog_input_offset(AGX_CHIP_G17P, &inputs, h, &offset));
      CHECK(offset == (h - 8) * 2);
   }

   /* Exercise rank across every bitmap-word boundary, including bit 63.
    * A sequential enumeration is independent of the popcount lookup. */
   for (unsigned pattern = 0; pattern < 3; ++pattern) {
      memset(inputs.uniform_halfs, 0, sizeof(inputs.uniform_halfs));
      for (unsigned h = 0; h < AGX_IOTO_CPROG_UNIFORM_HALFS; ++h) {
         bool selected = pattern == 0 || (pattern == 1 ? h % 3 == 1 :
                                            h % 64 == 0 || h % 64 == 63);
         if (selected)
            inputs.uniform_halfs[h / 64] |= UINT64_C(1) << (h % 64);
      }
      unsigned expected = 0;
      for (unsigned h = 0; h < AGX_IOTO_CPROG_UNIFORM_HALFS; ++h) {
         uint16_t offset = 0xbeef;
         bool selected = (inputs.uniform_halfs[h / 64] >> (h % 64)) & 1;
         CHECK(agx_ioto_cprog_input_offset(AGX_CHIP_G17P, &inputs, h, &offset) == selected);
         CHECK(offset == (selected ? 2 * expected : 0xbeef));
         expected += selected;
      }
   }

   uint16_t offset = 0xabcd;
   inputs.known = false;
   CHECK(!agx_ioto_cprog_input_offset(AGX_CHIP_G17P, &inputs, 0, &offset));
   inputs.known = true;
   CHECK(!agx_ioto_cprog_input_offset(AGX_CHIP_G17P, &inputs, 512, &offset));
   CHECK(!agx_ioto_cprog_input_offset(AGX_CHIP_G17P, NULL, 0, &offset));
   CHECK(!agx_ioto_cprog_input_offset(AGX_CHIP_G17P, &inputs, 0, NULL));
   CHECK(offset == 0xabcd);
   memset(inputs.uniform_halfs, 0, sizeof(inputs.uniform_halfs));
   CHECK(!agx_ioto_cprog_input_offset(AGX_CHIP_G17P, &inputs, 0, &offset));
   CHECK(offset == 0xabcd);
   return true;
}

int
main(void)
{
   if (!test_targets() || !test_ldimm_nop() || !test_allocator_boundaries() ||
       !test_fast_path() || !test_direct_sources() ||
       !test_cache_and_rollback() || !test_comparator() ||
       !test_two_phase_and_nops() || !test_finish_rollback() ||
       !test_rollover() || !test_cohort_parity() || !test_cprog_input_layout())
      return 1;

   printf("agx_ioto_load_core: %u checks passed\n", checks);
   return 0;
}
