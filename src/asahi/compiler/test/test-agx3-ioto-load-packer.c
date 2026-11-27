/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

#include "agx3_ioto_load_packer.h"
#include "agx3_binder.h"

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

struct fixture {
   uint8_t bytes[1024];
   struct agx_ioto_inflight_load loads[16];
   struct agx_ioto_inflight_load scratch[16];
   struct agx_ioto_load_cache_entry cache[16];
   struct agx_ioto_load_core core;
};

static bool
fixture_init(struct fixture *f, enum agx_chip chip, bool gpu0_nops)
{
   memset(f, 0, sizeof(*f));
   return agx_ioto_load_core_init(
      &f->core, chip,
      (struct agx_ioto_load_sink){f->bytes, sizeof(f->bytes), 0}, f->loads,
      ARRAY_SIZE(f->loads), f->cache, ARRAY_SIZE(f->cache), 0, false,
      gpu0_nops);
}

static struct agx_ioto_load_workspace
workspace(struct fixture *f)
{
   return (struct agx_ioto_load_workspace){f->scratch, ARRAY_SIZE(f->scratch)};
}

static struct agx_ioto_inflight_load
absolute_record(int16_t offset)
{
   return (struct agx_ioto_inflight_load){
      .data = {.half = 40, .type = 3, .value = 1},
      .base = {.half = 44, .type = 4, .value = 8},
      .byte_offset = offset,
      .count_dwords = 1,
      .native_slot_half = 20,
      .lane_mask = 3,
      .cprog_dest_half = 40,
      .signal_slot = 1,
   };
}

static bool
test_target_gate(void)
{
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      CHECK(agx3_ioto_load_absolute_packer_ready(gpu0_targets[i]));
      CHECK(agx3_ioto_load_packer_init(&packer, gpu0_targets[i]));
      CHECK(packer.encoding == AGX_IOTO_LOAD_ENCODING_GPU0);
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      CHECK(agx3_ioto_load_absolute_packer_ready(gpu1_targets[i]));
      CHECK(agx3_ioto_load_packer_init(&packer, gpu1_targets[i]));
      CHECK(packer.encoding == AGX_IOTO_LOAD_ENCODING_GPU1);
   }
   CHECK(!agx3_ioto_load_absolute_packer_ready(AGX_CHIP_G14G));
   CHECK(!agx3_ioto_load_absolute_packer_ready(AGX_CHIP_G15P));
   CHECK(!agx3_ioto_load_packer_init(&packer, AGX_CHIP_UNKNOWN));
   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17P));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.trusted_records);
   return true;
}

static bool
pack_load(enum agx_chip chip, struct agx_ioto_inflight_load *load,
          struct agx_ioto_load_raw_record *raw, uint8_t *slot)
{
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (!agx3_ioto_load_packer_init(&packer, chip))
      return false;
   agx3_ioto_load_packer_ops(&packer, &ops);
   return ops.pack_load(ops.context, chip, load, raw, slot);
}

static bool
test_load_records(void)
{
   static const uint8_t gpu0_expected[] = {
      0x67, 0x00, 0x54, 0x28, 0x16, 0x00, 0x00,
      0x00, 0x51, 0x00, 0x00, 0x40, 0x22, 0x00,
   };
   static const uint8_t gpu1_ep0[] = {
      0x4f, 0x2c, 0x43, 0x00, 0x80, 0x04, 0x00, 0x20,
   };
   static const uint8_t gpu1_ep2[] = {
      0x4f, 0x2c, 0x43, 0x00, 0x80, 0x04, 0x00,
      0xa0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
   };

   struct agx_ioto_inflight_load load = absolute_record(0);
   struct agx_ioto_load_raw_record raw;
   uint8_t slot = 0;
   for (size_t i = 0; i < ARRAY_SIZE(gpu0_targets); ++i) {
      CHECK(pack_load(gpu0_targets[i], &load, &raw, &slot));
      CHECK(raw.kind == AGX_IOTO_GPU0_INLINE_LDDEV14);
      CHECK(raw.length == sizeof(gpu0_expected));
      CHECK(memcmp(raw.bytes, gpu0_expected, sizeof(gpu0_expected)) == 0);
      CHECK(slot == 2);
   }
   for (size_t i = 0; i < ARRAY_SIZE(gpu1_targets); ++i) {
      CHECK(pack_load(gpu1_targets[i], &load, &raw, &slot));
      CHECK(raw.kind == AGX_IOTO_LOAD_RAW_LDDEV_EP0);
      CHECK(raw.length == sizeof(gpu1_ep0));
      CHECK(memcmp(raw.bytes, gpu1_ep0, sizeof(gpu1_ep0)) == 0);
   }

   load.byte_offset = 256;
   CHECK(pack_load(AGX_CHIP_G17G, &load, &raw, &slot));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_LDDEV_EP2);
   CHECK(raw.length == sizeof(gpu1_ep2));
   CHECK(memcmp(raw.bytes, gpu1_ep2, sizeof(gpu1_ep2)) == 0);

   load.byte_offset = -1;
   CHECK(pack_load(AGX_CHIP_G17G, &load, &raw, &slot));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_LDDEV_EP2);
   load.data.type = 2;
   CHECK(!pack_load(AGX_CHIP_G17G, &load, &raw, &slot));
   return true;
}

static bool
test_copy_records(void)
{
   static const uint8_t gpu0_dword0[] = {0xab, 0x28, 0x09, 0x04};
   static const uint8_t gpu0_dword1[] = {0xbb, 0x2a, 0x09, 0x04};
   static const uint8_t gpu1_dword1[] = {0xbb, 0x2a, 0x09, 0x04};

   struct agx_ioto_inflight_load load = absolute_record(0);
   load.count_dwords = 2;
   struct agx_ioto_load_raw_record raw;
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;

   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G15G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G15G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP0 &&
         raw.length == sizeof(gpu0_dword0));
   CHECK(memcmp(raw.bytes, gpu0_dword0, sizeof(gpu0_dword0)) == 0);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G15G, &load, 1, 3, &raw));
   CHECK(memcmp(raw.bytes, gpu0_dword1, sizeof(gpu0_dword1)) == 0);

   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP0 && raw.length == 4);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 1, 3, &raw));
   CHECK(memcmp(raw.bytes, gpu1_dword1, sizeof(gpu1_dword1)) == 0);
   CHECK(!ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 2, 3, &raw));
   CHECK(!ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 0, &raw));

   /* Native G17G EOT Ioto, macOS 25G76, offsets 0xde and 0xe2: a
    * two-dword pointer moves from halves 36/38 to adjacent halves 0/2.
    * An earlier COPY already consumed the load's wait in this sequence.
    */
   load.native_slot_half = 0;
   load.data.half = 36;
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 3, &raw));
   CHECK(raw.length == 4 &&
         memcmp(raw.bytes, (uint8_t[]){0x0b, 0x24, 0x09, 0x04}, 4) == 0);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 1, 3, &raw));
   CHECK(raw.length == 4 &&
         memcmp(raw.bytes, (uint8_t[]){0x1b, 0x26, 0x09, 0x04}, 4) == 0);

   load.native_slot_half = 150;
   load.data.half = 190;
   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G15G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G15G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP1 && raw.length == 10);
   CHECK(memcmp(raw.bytes,
                (uint8_t[]){0xbb, 0x3f, 0x0f, 0x00, 0x42, 0x11, 0x00, 0x00,
                            0x00, 0x00},
                10) == 0);

   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17G));
   agx3_ioto_load_packer_ops(&packer, &ops);
   CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17G, &load, 0, 3, &raw));
   CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP1 && raw.length == 8);

   /* The production AGX3B wide-copy specialization must remain exactly the
    * same encoder as the authoritative table composer for every register
    * value it admits. */
   static const struct agx3_field_kv baseline[] = {
      {9, 3}, {10, 2}, {23, 2}, {24, 2}, {27, 1},
   };
   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17P));
   agx3_ioto_load_packer_ops(&packer, &ops);
   load.count_dwords = 1;
   for (unsigned destination = 128; destination < 256; ++destination) {
      for (unsigned source = 0; source < 256; ++source) {
         const struct agx3_field_kv overrides[] = {
            {3, destination}, {16, source},
         };
         uint8_t expected[32] = {0}, expected_length = 0;
         load.native_slot_half = destination;
         load.data.half = source;
         CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17P, &load, 0, 3, &raw));
         CHECK(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_copy, 1,
                                 baseline, ARRAY_SIZE(baseline), overrides,
                                 ARRAY_SIZE(overrides), expected,
                                 &expected_length));
         CHECK(raw.kind == AGX_IOTO_LOAD_RAW_COPY_EP1);
         CHECK(raw.length == expected_length);
         CHECK(memcmp(raw.bytes, expected, expected_length) == 0);
      }
   }
   return true;
}

static bool
queue_two_dwords(struct fixture *f)
{
   const struct agx_ioto_load_absolute request = {
      .native_slot = 10,
      .count_dwords = 2,
      .address = 0x20,
      .lane_mask = 0x0f,
      .cprog_dest_half = 40,
   };
   struct agx_ioto_load_allocation allocation;
   return agx3_ioto_load_absolute_native(&f->core, &request, workspace(f),
                                         &allocation) &&
          allocation.temp_half == 0;
}

static bool
test_native_rounds(void)
{
   static const uint8_t gpu0_tail[] = {
      0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00, 0x59, 0x00,
      0x00, 0x40, 0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00,
      0x06, 0x00, 0xab, 0x04, 0x09, 0x04, 0xbb, 0x06, 0x09, 0x04,
   };
   static const uint8_t gpu1_tail[] = {
      0x2f, 0x00, 0x03, 0x00, 0xa0, 0x0c, 0x00, 0x20,
   };

   struct fixture f;
   CHECK(fixture_init(&f, AGX_CHIP_G15G, true));
   CHECK(queue_two_dwords(&f));
   CHECK(f.core.sink.length == 4);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == 4 + sizeof(gpu0_tail));
   CHECK(memcmp(&f.bytes[4], gpu0_tail, sizeof(gpu0_tail)) == 0);

   CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
   CHECK(queue_two_dwords(&f));
   CHECK(agx3_ioto_load_finish_round_native(&f.core, false, workspace(&f)));
   CHECK(f.core.sink.length == 12);
   CHECK(f.core.loads[0].processed && f.core.loads[0].signal_slot == 1);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == 4 + sizeof(gpu1_tail) + 8);
   CHECK(memcmp(&f.bytes[4], gpu1_tail, sizeof(gpu1_tail)) == 0);
   CHECK(f.core.load_count == 0 && f.core.cache_count == 0);

   CHECK(fixture_init(&f, AGX_CHIP_G15G, false));
   CHECK(!queue_two_dwords(&f));
   return true;
}

static bool
test_native_direct_sources(void)
{
   static const uint8_t scs_base[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
   };
   static const uint8_t gpu0_scs[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x67, 0x00,
      0x54, 0x00, 0x04, 0x00, 0x00, 0x00, 0x59, 0x60, 0x24, 0x40,
      0x22, 0x00, 0x06, 0x08, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
      0x7b, 0x00, 0x09, 0x04, 0x8b, 0x02, 0x09, 0x04,
   };
   static const uint8_t gpu1_scs[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x0f, 0x08,
      0x03, 0x00, 0xa0, 0x0c, 0x60, 0xa4, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x01,
   };
   static const uint8_t gpu1_abi_slot2[] = {
      0x0f, 0x04, 0x03, 0x00, 0xa0, 0x0c, 0x60, 0xa4, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x01,
   };
   static const uint8_t gpu1_abi_slot54[] = {
      0x0f, 0x6c, 0x03, 0x00, 0xa0, 0x0c, 0x60, 0xa4, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x01,
   };
   static const enum agx_ioto_abi_buffer_slot abi_slots[] = {
      AGX_IOTO_ABI_BUFFER_SLOT_2,  AGX_IOTO_ABI_BUFFER_SLOT_4,
      AGX_IOTO_ABI_BUFFER_SLOT_6,  AGX_IOTO_ABI_BUFFER_SLOT_8,
      AGX_IOTO_ABI_BUFFER_SLOT_50, AGX_IOTO_ABI_BUFFER_SLOT_52,
      AGX_IOTO_ABI_BUFFER_SLOT_54,
   };
   const struct agx_ioto_load_from_fixed request = {
      .native_slot = 7,
      .count_dwords = 2,
      .byte_offset = 0x123,
      .lane_mask = 0xf,
      .cprog_dest_half = 40,
   };
   struct agx_ioto_load_allocation allocation;
   struct fixture f;

   CHECK(fixture_init(&f, AGX_CHIP_G15G, true));
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(allocation.temp_half == 0 && f.core.sink.length == sizeof(scs_base));
   CHECK(memcmp(f.bytes, scs_base, sizeof(scs_base)) == 0);
   CHECK(f.core.loads[0].base.half == AGX_IOTO_LOAD_SCS_BASE_HALF);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == sizeof(gpu0_scs));
   CHECK(memcmp(f.bytes, gpu0_scs, sizeof(gpu0_scs)) == 0);
   CHECK(f.core.scs_base_loaded);
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(f.core.sink.length == sizeof(gpu0_scs));
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == sizeof(gpu0_scs) * 2 - sizeof(scs_base));
   CHECK(memcmp(&f.bytes[sizeof(gpu0_scs)], &gpu0_scs[sizeof(scs_base)],
                sizeof(gpu0_scs) - sizeof(scs_base)) == 0);

   CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(memcmp(f.bytes, scs_base, sizeof(scs_base)) == 0);
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(f.core.sink.length == sizeof(gpu1_scs) + 8);
   CHECK(memcmp(f.bytes, gpu1_scs, sizeof(gpu1_scs)) == 0);

   for (size_t i = 0; i < ARRAY_SIZE(abi_slots); ++i) {
      CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
      CHECK(agx3_ioto_load_from_abi_buffer_native(
         &f.core, abi_slots[i], &request, workspace(&f), &allocation));
      CHECK(f.core.loads[0].base.half == 2u * (unsigned)abi_slots[i]);
      CHECK(!f.core.scs_base_loaded && !f.core.scs_base_ever_loaded);
      CHECK(f.core.abi4_load_pending ==
            (abi_slots[i] == AGX_IOTO_ABI_BUFFER_SLOT_4));
      CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
      CHECK(!f.core.abi4_load_pending);
      CHECK(f.core.sink.length == sizeof(gpu1_abi_slot2) + 8);
      if (i == 0)
         CHECK(memcmp(f.bytes, gpu1_abi_slot2, sizeof(gpu1_abi_slot2)) == 0);
      if (i + 1 == ARRAY_SIZE(abi_slots))
         CHECK(memcmp(f.bytes, gpu1_abi_slot54, sizeof(gpu1_abi_slot54)) == 0);
   }

   CHECK(fixture_init(&f, AGX_CHIP_G17G, false));
   CHECK(agx3_ioto_load_from_abi_buffer_native(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_4, &request, workspace(&f),
      &allocation));
   CHECK(!agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                         &allocation));
   CHECK(agx3_ioto_load_finish_round_native(&f.core, true, workspace(&f)));
   CHECK(agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                        &allocation));
   CHECK(!agx3_ioto_load_from_abi_buffer_native(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_4, &request, workspace(&f),
      &allocation));

   CHECK(fixture_init(&f, AGX_CHIP_G15G, false));
   CHECK(!agx3_ioto_load_from_scs_native(&f.core, &request, workspace(&f),
                                         &allocation));
   CHECK(!agx3_ioto_load_from_abi_buffer_native(
      &f.core, AGX_IOTO_ABI_BUFFER_SLOT_2, &request, workspace(&f),
      &allocation));
   return true;
}

static bool
test_mem_form_bounds(void)
{
   struct agx3_mem_desc desc = {
      .data = 40,
      .width = AGX3_MEM_W32,
      .mask = 1,
      .base = 44,
      .idx_kind = AGX3_MEM_IDX_IMM,
      .byte_off = 0,
      .slot = 1,
   };
   uint8_t bytes[AGX3_MEM_MAX_LENGTH], length;
   CHECK(agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3B, &desc, 0, bytes,
                                   &length));
   CHECK(!agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3B, &desc, 1, bytes,
                                    &length));
   for (uint8_t ep = 0; ep <= 5; ++ep)
      CHECK(agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3C, &desc, ep, bytes,
                                      &length));
   CHECK(!agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3C, &desc, 6, bytes,
                                    &length));
   return true;
}

static bool
test_cprog_inputs(void)
{
   /* Native G17P copyGprToCprog oracle, driver SHA-256 666816a37045b6dc...
    * Source Register(type=3, half=10, value=1), encoder +0x100 clear.
    * These cover packed partial masks and the odd-TPS-half split. */
   static const struct {
      uint16_t destination;
      uint8_t mask, length;
      uint8_t bytes[AGX3_IOTO_CPROG_COPY_MAX_SIZE];
   } native[] = {
      {0, 1, 10, {0x43, 0x0a, 0x0f, 0x00, 0x22, 0x40, 0x00, 0x14, 0x00, 0x00}},
      {0, 2, 10, {0x43, 0x0a, 0x0f, 0x00, 0x2a, 0x40, 0x00, 0x14, 0x00, 0x00}},
      {0, 3, 10, {0x4b, 0x0b, 0x0f, 0x00, 0x22, 0x40, 0x00, 0x14, 0x00, 0x00}},
      {2, 3, 20, {0x43, 0x0a, 0x0f, 0x00, 0x26, 0x40, 0x00, 0x14, 0x00, 0x00,
                  0x53, 0x0a, 0x0f, 0x00, 0x2a, 0x40, 0x00, 0x14, 0x00, 0x00}},
      {4, 3, 10, {0x5b, 0x0b, 0x0f, 0x00, 0x22, 0x40, 0x00, 0x14, 0x00, 0x00}},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(native); ++i) {
      struct agx3_ioto_cprog_copy_output out;
      const struct agx3_ioto_cprog_copy_desc d = {
         .destination_bytes = native[i].destination,
         .source_half = 10,
         .lane_mask = native[i].mask,
         .source_wait_slot = 1,
      };
      CHECK(agx3_ioto_pack_cprog_copy(AGX_CHIP_G17P, &d, &out));
      CHECK(out.length == native[i].length);
      CHECK(out.written_bytes == (native[i].mask == 3 ? 4 : 2));
      CHECK(memcmp(out.bytes, native[i].bytes, sizeof(out.bytes)) == 0);
   }

   const struct agx3_ioto_cprog_copy_desc invalid[] = {
      {1, 0, 3, -1}, {0, 1, 3, -1}, {0, 256, 3, -1},
      {0, 0, 0, -1}, {0, 0, 4, -1}, {0, 0, 3, -2}, {0, 0, 3, 6},
   };
   struct agx3_ioto_cprog_copy_output out, before;
   memset(&out, 0xa5, sizeof(out));
   before = out;
   for (unsigned i = 0; i < ARRAY_SIZE(invalid); ++i) {
      CHECK(!agx3_ioto_pack_cprog_copy(AGX_CHIP_G17P, &invalid[i], &out));
      CHECK(memcmp(&out, &before, sizeof(out)) == 0);
   }

   const struct agx3_ioto_cprog_copy_desc boundary = {65534, 254, 3, -1};
   CHECK(!agx3_ioto_pack_cprog_copy(AGX_CHIP_G17S, &boundary, &out));
   CHECK(!agx3_ioto_pack_cprog_copy(AGX_CHIP_G13G, &boundary, &out));
   CHECK(!agx3_ioto_pack_cprog_copy(AGX_CHIP_G17P, NULL, &out));
   CHECK(memcmp(&out, &before, sizeof(out)) == 0);
   CHECK(!agx3_ioto_pack_cprog_copy(AGX_CHIP_G17P, &boundary, NULL));
   CHECK(agx3_ioto_pack_cprog_copy(AGX_CHIP_G17P, &boundary, &out));
   CHECK(out.length == 20 && out.written_bytes == 4);
   return true;
}

static bool
compare_gpu0_load(struct agx_ioto_inflight_load load)
{
   const struct agx3_mem_desc desc = {
      .data = load.data.half, .width = AGX3_MEM_W32,
      .mask = (1u << load.count_dwords) - 1, .base = load.base.half,
      .idx_kind = AGX3_MEM_IDX_IMM, .byte_off = load.byte_offset, .slot = 2,
   };
   uint8_t expected[AGX3_MEM_MAX_LENGTH], length = 0, slot = 0;
   struct agx_ioto_load_raw_record raw;
   bool reference = agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3B,
                                             &desc, 0, expected, &length);
   CHECK(pack_load(AGX_CHIP_G17P, &load, &raw, &slot) == reference);
   if (reference) {
      CHECK(raw.length == length && slot == 2);
      CHECK(memcmp(raw.bytes, expected, length) == 0);
   }
   return true;
}

static bool
logical_state_unchanged(const struct fixture *f, const struct fixture *before)
{
   return memcmp(&f->core, &before->core, sizeof(f->core)) == 0 &&
          memcmp(f->loads, before->loads, sizeof(f->loads)) == 0 &&
          memcmp(f->cache, before->cache, sizeof(f->cache)) == 0 &&
          memcmp(f->bytes, before->bytes, before->core.sink.length) == 0;
}

static bool
test_cprog_rounds(void)
{
   /* Ready-source copyGprToCprog bytes from native G17P 360.34.5,
    * driver SHA-256 666816a37045b6dc... . The ordinary uniform COPY must
    * leave its source live when the following TPS COPY still reads it. */
   static const struct {
      uint64_t mask;
      unsigned extra_bound;
      uint8_t length;
      uint8_t copies[38];
   } native[] = {
      {0, 0, 8,
       {0xab, 0x04, 0x09, 0x04, 0xbb, 0x06, 0x09, 0x04}},
      {UINT64_C(1) << 20, 10, 18,
       {0xab, 0x04, 0x01, 0x04,
        0x43, 0x04, 0x0f, 0x00, 0x22, 0x00, 0x00, 0x14, 0x00, 0x00,
        0xbb, 0x06, 0x09, 0x04}},
      {UINT64_C(3) << 20, 20, 18,
       {0xab, 0x04, 0x01, 0x04,
        0x4b, 0x05, 0x0f, 0x00, 0x22, 0x00, 0x00, 0x14, 0x00, 0x00,
        0xbb, 0x06, 0x09, 0x04}},
      {(UINT64_C(1) << 20) | (UINT64_C(3) << 22), 30, 38,
       {0xab, 0x04, 0x01, 0x04,
        0x43, 0x04, 0x0f, 0x00, 0x22, 0x00, 0x00, 0x14, 0x00, 0x00,
        0xbb, 0x06, 0x01, 0x04,
        0x43, 0x06, 0x0f, 0x00, 0x26, 0x00, 0x00, 0x14, 0x00, 0x00,
        0x53, 0x06, 0x0f, 0x00, 0x2a, 0x00, 0x00, 0x14, 0x00, 0x00}},
   };
   struct fixture baseline;
   CHECK(fixture_init(&baseline, AGX_CHIP_G17P, true));
   CHECK(queue_two_dwords(&baseline));
   CHECK(agx3_ioto_load_finish_round_native(&baseline.core, true,
                                           workspace(&baseline)));

   for (unsigned n = 0; n < ARRAY_SIZE(native); ++n) {
      /* Exercise both a single finish and an earlier load-only finish. */
      for (unsigned staged = 0; staged < 2; ++staged) {
         struct fixture f;
         struct agx3_ioto_load_packer packer;
         struct agx_ioto_load_emit_ops ops;
         const struct agx_ioto_cprog_inputs inputs = {
            .known = true, .uniform_halfs = {native[n].mask},
         };
         CHECK(fixture_init(&f, AGX_CHIP_G17P, true));
         CHECK(queue_two_dwords(&f));
         CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17P));
         packer.cprog_inputs = &inputs;
         agx3_ioto_load_packer_ops(&packer, &ops);
         if (staged) {
            CHECK(agx_ioto_load_finish_round(&f.core, false, &ops,
                                              workspace(&f)));
            CHECK(f.core.sink.length == 18 && f.core.loads[0].processed);
         }
         size_t ordinary, bound;
         CHECK(agx_ioto_load_finish_worst_case_size(&f.core, true, &ordinary));
         CHECK(agx_ioto_load_finish_worst_case_size_with_ops(
            &f.core, true, &ops, &bound));
         CHECK(bound == ordinary + native[n].extra_bound);

         f.core.sink.capacity = f.core.sink.length + bound - 1;
         struct fixture before = f;
         CHECK(!agx_ioto_load_finish_round(&f.core, true, &ops, workspace(&f)));
         CHECK(logical_state_unchanged(&f, &before));

         f.core.sink.capacity++;
         CHECK(agx_ioto_load_finish_round(&f.core, true, &ops, workspace(&f)));
         CHECK(f.core.sink.length == 26 + native[n].length);
         /* Address setup, LDDEV and the existing four NOPs are unchanged. */
         CHECK(memcmp(f.bytes, baseline.bytes, 26) == 0);
         CHECK(memcmp(f.bytes + 26, native[n].copies, native[n].length) == 0);
         CHECK(f.core.load_count == 0 && f.core.cache_count == 0);
         CHECK(f.core.current_gpr_dwords == f.core.initial_gpr_dwords);
      }
   }
   return true;
}

static bool
reject_extra_copy(void *context, enum agx_chip chip,
                  const struct agx_ioto_inflight_load *load, uint8_t dword,
                  struct agx_ioto_load_raw_record records[2], uint8_t *count)
{
   (void)context;
   (void)chip;
   (void)load;
   (void)dword;
   (void)records;
   (void)count;
   return false;
}

static bool
invalid_extra_copy(void *context, enum agx_chip chip,
                   const struct agx_ioto_inflight_load *load, uint8_t dword,
                   struct agx_ioto_load_raw_record records[2], uint8_t *count)
{
   (void)context;
   (void)chip;
   (void)load;
   (void)dword;
   /* An LDDEV must never be accepted in the optional COPY stream. */
   records[0] = (struct agx_ioto_load_raw_record){
      .kind = AGX_IOTO_GPU0_INLINE_LDDEV14, .length = 14,
   };
   *count = 1;
   return true;
}

static bool
test_cprog_round_failures(void)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      struct fixture f;
      struct agx3_ioto_load_packer packer;
      struct agx_ioto_load_emit_ops ops;
      struct agx_ioto_cprog_inputs inputs = {
         .known = true, .uniform_halfs = {UINT64_C(1) << 20},
      };
      CHECK(fixture_init(&f, AGX_CHIP_G17P, true));
      CHECK(queue_two_dwords(&f));
      CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17P));
      packer.cprog_inputs = &inputs;
      agx3_ioto_load_packer_ops(&packer, &ops);
      switch (mode) {
      case 0: inputs.known = false; break;
      case 1: f.core.fast_path_enabled = true; break;
      case 2: f.core.gpu0_round_nops = false; break;
      case 3: ops.extra_copy_bound = NULL; break;
      case 4: ops.pack_extra_copies = NULL; break;
      case 5: ops.pack_extra_copies = reject_extra_copy; break;
      case 6: ops.pack_extra_copies = invalid_extra_copy; break;
      case 7: f.core.loads[0].lane_mask = 1; break;
      case 8: f.core.chip = packer.chip = AGX_CHIP_G15G; break;
      }
      struct fixture before = f;
      CHECK(!agx_ioto_load_finish_round(&f.core, true, &ops, workspace(&f)));
      CHECK(logical_state_unchanged(&f, &before));
      if (mode == 5 || mode == 6) {
         /* These fail after the ordinary COPY was emitted, so exercise
          * rollback after partial work, not just input validation. */
         CHECK(memcmp(f.bytes + 26, (uint8_t[]){0xab, 0x04, 0x01, 0x04}, 4) == 0);
      }
   }
   return true;
}

static bool
test_gpu0_load_specializations(void)
{
   struct agx_ioto_inflight_load load = absolute_record(0);
   /* Every signed offset at every load width; register fields then varied
    * independently across the full accepted/rejected boundary. */
   for (unsigned count = 1; count <= 4; ++count) {
      load.count_dwords = count;
      for (int offset = -32768; offset <= 32767; ++offset) {
         load.byte_offset = offset;
         CHECK(compare_gpu0_load(load));
      }
      for (unsigned data = 0; data <= 256; ++data) {
         for (unsigned base = 0; base <= 192; ++base) {
            load.data.half = data;
            load.base.half = base;
            load.byte_offset = (int16_t)(data * 193 + base * 79);
            CHECK(compare_gpu0_load(load));
         }
      }
   }

   return true;
}

static bool
test_gpu0_copy_specializations(bool source_kill)
{
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   CHECK(agx3_ioto_load_packer_init(&packer, AGX_CHIP_G17P));
   struct agx_ioto_cprog_inputs inputs = {.known = true};
   for (unsigned i = 0; i < ARRAY_SIZE(inputs.uniform_halfs); ++i)
      inputs.uniform_halfs[i] = UINT64_MAX;
   if (!source_kill)
      packer.cprog_inputs = &inputs;
   agx3_ioto_load_packer_ops(&packer, &ops);
   const struct agx3_field_kv baseline[] = {
      {9, 3}, {10, 2}, {23, 2}, {24, 2}, {27, source_kill},
   };
   /* Exhaust the specialized region and the surrounding extended-form
    * fallback, including the existing ep0 alias at destination 512. */
   for (unsigned dst = 0; dst <= (source_kill ? 513 : 510); ++dst) {
      for (unsigned src = 0; src <= 256; ++src) {
         const struct agx3_field_kv fields[] = {{3, dst}, {16, src}};
         uint8_t expected[32], length = 0;
         bool reference = false;
         unsigned ep;
         for (ep = 0; ep < 2; ++ep) {
            reference = agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_copy, ep,
               baseline, ARRAY_SIZE(baseline), fields, ARRAY_SIZE(fields),
               expected, &length);
            if (reference)
               break;
         }
         struct agx_ioto_inflight_load load = absolute_record(0);
         load.data.half = src;
         load.native_slot_half = dst;
         struct agx_ioto_load_raw_record raw;
         CHECK(ops.pack_copy(ops.context, AGX_CHIP_G17P, &load, 0, 3, &raw)
               == reference);
         if (reference) {
            CHECK(raw.length == length);
            CHECK(raw.kind == (ep == 0 ? AGX_IOTO_LOAD_RAW_COPY_EP0 :
                                         AGX_IOTO_LOAD_RAW_COPY_EP1));
            CHECK(memcmp(raw.bytes, expected, length) == 0);
         }
      }
   }
   return true;
}

int
main(void)
{
   if (!test_target_gate() || !test_load_records() || !test_copy_records() ||
       !test_native_rounds() || !test_native_direct_sources() ||
       !test_mem_form_bounds() || !test_cprog_inputs() || !test_cprog_rounds() ||
       !test_cprog_round_failures() || !test_gpu0_load_specializations() ||
       !test_gpu0_copy_specializations(false) ||
       !test_gpu0_copy_specializations(true))
      return 1;
   printf("agx3_ioto_load_packer: %u checks passed\n", checks);
   return 0;
}
