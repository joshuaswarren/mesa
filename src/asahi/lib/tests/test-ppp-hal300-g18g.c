/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Mesa's HAL300 record packers against Apple's G18G records.
 *
 * Every record below comes from the J873 native Metal captures (Apple M6,
 * macOS 26A428, runs/mini-26A428/sessions-20260925-{a,b}) through
 * g18g_capture_vectors.h. For each of the thirteen oracle modes this packs,
 * with AGX_CHIP_G18G, what Mesa would write for the same draw and requires
 * Apple's bytes back:
 *
 *   - the 28-byte VDM State Update, the nine PPP graphics records, the
 *     direct Index List and every Stream Terminate;
 *   - the depth/stencil block, viewport, W-clamp/cull2 and depth-bias/scissor
 *     base records, which are packed from their unpacked fields;
 *   - the compute-add CDM launch, barrier and terminate;
 *   - the colour target's PBE and texture descriptors and the texture-sample
 *     sampler, which settle the descriptor and sampler generations;
 *   - the ZLS pixel dimensions the kernel command carries.
 *
 * The draw state for each mode is what the oracle program asked Metal for
 * (the native Metal capture oracle, oracle.m): program and varying-table addresses
 * are the capture's own. Two things are deliberately not modelled and are
 * checked as exceptions: the three-record prefix Apple emits ahead of the
 * VDM State Update, which Mesa does not emit, and bit 0 of the stencil
 * always/replace draw's face words, which the M5 capture carries too.
 */

#include "asahi/genxml/agx3_cmdstream.h"
#include "agx_pack.h"

#include "../agx3_cdm_decode_core.h"
#include "../agx_pbe_agx3_core.h"
#include "../agx_ppp_decode_core.h"
#include "../agx_ppp_hal300_depth_bias_scissor_core.h"
#include "../agx_ppp_hal300_depth_stencil_core.h"
#include "../agx_ppp_hal300_graphics_core.h"
#include "../agx_ppp_viewport_payload_core.h"
#include "../agx_ppp_w_clamp_cull2_core.h"
#include "../agx_vdm_barrier_core.h"
#include "../agx_vdm_decode_core.h"
#include "../agx_vdm_index_list_core.h"
#include "../agx_vdm_state_update_core.h"
#include "../agx_vdm_stream_link_core.h"
#include "../agx_vdm_stream_terminate_core.h"
#include "../agx_vdm_tessellate_core.h"
#include "../agx_zls_pixels.h"
#include "g18g_capture_vectors.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, what, #x);     \
         failures++;                                                           \
      }                                                                        \
   } while (0)

#define MAX_BYTES 256

static void
store_words(const uint32_t *words, unsigned count, uint8_t *bytes)
{
   for (unsigned i = 0; i < count; ++i) {
      bytes[4 * i + 0] = (uint8_t)words[i];
      bytes[4 * i + 1] = (uint8_t)(words[i] >> 8);
      bytes[4 * i + 2] = (uint8_t)(words[i] >> 16);
      bytes[4 * i + 3] = (uint8_t)(words[i] >> 24);
   }
}

static bool
equal_words(const uint8_t *bytes, unsigned length, const uint32_t *words,
            unsigned count)
{
   uint8_t expected[MAX_BYTES];

   if (length != count * 4 || length > sizeof(expected))
      return false;

   store_words(words, count, expected);
   return memcmp(bytes, expected, length) == 0;
}

/* The draw state of one oracle render, as Metal was asked for it. */
struct g18g_render {
   const char *name;
   uint64_t vertex_program;
   uint32_t vertex_outputs;
   uint8_t primitive;
   uint32_t count;
   uint32_t starts[2];
   unsigned draws;
   bool zls;
   struct agx_ppp_hal300_graphics_fields graphics;
   /* Only the stencil render draws twice with different state. */
   uint32_t second_stencil;
};

/* Every oracle vertex shader exports a smooth colour or texture coordinate
 * that its fragment shader reads through two CF bindings. */
#define G18G_DRAW(fs, vt, components, ...)                                     \
   {                                                                           \
      .fragment_program = (fs), .varying_table = (vt), .cf_binding_count = 2,  \
      .vertex_varying_components = (components),                              \
      .fragment_varying_components = (components), __VA_ARGS__                 \
   }

static const struct g18g_render renders[] = {
   {"color", 0x10000158000, 0x808, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000168780, 0x100000007c0, 4), 0},
   {"depth-less", 0x100001a0000, 0x808, 6, 3, {0, 3}, 2, true,
    G18G_DRAW(0x100001b0780, 0x100000007c0, 4, .depth_active = true,
              .depth_compare = 1, .depth_write = true), 0},
   {"depth-always", 0x100001a0000, 0x808, 6, 3, {0, 3}, 2, true,
    G18G_DRAW(0x100001b0780, 0x100000007c0, 4, .depth_active = true,
              .depth_compare = 7, .depth_write = true), 0},
   /* Draw A writes reference 2 with always/replace; draw B tests it with
    * equal/keep. */
   {"stencil", 0x100001b8000, 0x808, 6, 3, {0, 3}, 2, true,
    G18G_DRAW(0x100001c8780, 0x100000007c0, 4, .stencil_active = true,
              .front_stencil = 0x0e02ffff, .back_stencil = 0x0e02ffff),
    0x0400ff00},
   {"msaa4", 0x10000230000, 0x808, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000240900, 0x100000007c0, 4), 0},
   {"r32f", 0x10000158000, 0x808, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000168780, 0x10000000780, 4), 0},
   {"r8ui", 0x10000128000, 0x808, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000138780, 0x10000000780, 4), 0},
   {"points", 0x10000158000, 0x909, 0, 3, {0}, 1, false,
    G18G_DRAW(0x10000168780, 0x100000007c0, 4, .point_size_output = true,
              .object_type = AGX_PPP_HAL300_POINTS), 0},
   {"lines", 0x10000158000, 0x808, 1, 6, {6}, 1, false,
    G18G_DRAW(0x10000168780, 0x100000007c0, 4,
              .object_type = AGX_PPP_HAL300_LINES), 0},
   {"blend-add", 0x10000158000, 0x808, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000168780, 0x10000000800, 4, .fragment_pass = 1), 0},
   {"discard", 0x10000158000, 0x808, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000168780, 0x10000000800, 4, .fragment_pass = 2,
              .disable_tri_merging = true), 0},
   {"texture-sample", 0x10000158000, 0x606, 6, 3, {0}, 1, false,
    G18G_DRAW(0x10000168780, 0x100000007c0, 2), 0},
};

/* The VDM offsets of the PPP records Mesa's graphics core packs, in its
 * record order, and of the records the other packers cover. They are the
 * same in every render; test_vdm_decode asserts the pointers. */
static const uint32_t graphics_offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS] = {
   0x34, 0x3c, 0x44, 0x4c, 0x54, 0x5c, 0x6c, 0x74, 0x84,
};
#define PREFIX_DEPTH_STENCIL_OFFSET 0x00
#define PREFIX_A_OFFSET             0x08
#define PREFIX_B_OFFSET             0x10
#define VDM_STATE_OFFSET            0x18
#define VIEWPORT_OFFSET             0x64
#define DEPTH_BIAS_SCISSOR_OFFSET   0x7c

static const struct g18g_capture_ppp *
find_ppp(const struct g18g_capture_mode *mode, uint32_t vdm_offset)
{
   for (unsigned i = 0; i < mode->ppp_count; ++i) {
      if (mode->ppp[i].vdm_offset == vdm_offset)
         return &mode->ppp[i];
   }

   return NULL;
}

static const struct g18g_capture_mode *
find_mode(const char *name)
{
   for (unsigned i = 0;
        i < sizeof(g18g_capture_modes) / sizeof(g18g_capture_modes[0]); ++i) {
      if (!strcmp(g18g_capture_modes[i].name, name))
         return &g18g_capture_modes[i];
   }

   return NULL;
}

/* Walk the whole VDM stream with the G18G decoder, recording where the Index
 * Lists are, and return the number of draws. G17G must accept every record
 * but the Index List. */
static unsigned
test_vdm_decode(const struct g18g_capture_mode *mode,
                const struct g18g_render *render, uint32_t *index_lists)
{
   const char *what = mode->name;
   uint8_t bytes[MAX_BYTES];
   const size_t size = mode->stream_words * 4;
   size_t offset = 0;
   unsigned ppp = 0, draws = 0;
   bool terminated = false;

   CHECK(size <= sizeof(bytes));
   if (size > sizeof(bytes))
      return 0;

   store_words(mode->stream, mode->stream_words, bytes);

   while (offset < size && !terminated) {
      struct agx_vdm_decoded_record record, g17g;
      if (!agx_vdm_decode_one(AGX_CHIP_G18G, bytes + offset, size - offset,
                              &record)) {
         CHECK(!"G18G refuses a captured VDM record");
         return 0;
      }

      const bool g17g_ok = agx_vdm_decode_one(AGX_CHIP_G17G, bytes + offset,
                                              size - offset, &g17g);

      switch (record.kind) {
      case AGX_VDM_RECORD_PPP_STATE_UPDATE:
         CHECK(ppp < mode->ppp_count);
         CHECK(mode->ppp[ppp].vdm_offset == offset);
         CHECK(record.data.ppp_state_update.pointer == mode->ppp[ppp].address);
         CHECK(record.data.ppp_state_update.size_words ==
               mode->ppp[ppp].size_words);
         CHECK(g17g_ok && g17g.kind == record.kind);
         ppp++;
         break;
      case AGX_VDM_RECORD_STATE_UPDATE:
         CHECK(offset == VDM_STATE_OFFSET);
         CHECK(g17g_ok && g17g.kind == record.kind);
         break;
      case AGX_VDM_RECORD_INDEX_LIST_NONINDEXED: {
         const struct agx_vdm_index_list_direct_nonindexed_info *info =
            &record.data.index_list_nonindexed;
         CHECK(record.length == 20);
         CHECK(draws < render->draws);
         CHECK(info->primitive == render->primitive);
         CHECK(info->count == render->count);
         CHECK(info->instances == 1);
         CHECK(info->start == render->starts[draws < 2 ? draws : 0]);
         if (draws < 2)
            index_lists[draws] = offset;
         draws++;
         break;
      }
      case AGX_VDM_RECORD_STREAM_TERMINATE:
         CHECK(g17g_ok && g17g.kind == record.kind);
         terminated = true;
         break;
      default:
         CHECK(!"unexpected VDM record");
         return 0;
      }

      offset += record.length;
   }

   CHECK(terminated && offset == size);
   CHECK(ppp == mode->ppp_count);
   CHECK(draws == render->draws);
   return draws;
}

static void
test_ppp_decode(const struct g18g_capture_mode *mode)
{
   const char *what = mode->name;
   unsigned counts[AGX_PPP_DECODED_RECORD_HAL300_DEPTH_BIAS_SCISSOR_BASES + 1] =
      {0};

   for (unsigned i = 0; i < mode->ppp_count; ++i) {
      uint8_t bytes[MAX_BYTES];
      struct agx_ppp_decoded_record record;
      const size_t size = mode->ppp[i].size_words * 4;

      CHECK(size <= sizeof(bytes));
      if (size > sizeof(bytes))
         continue;

      store_words(mode->ppp[i].words, mode->ppp[i].size_words, bytes);
      if (agx_ppp_decode_one(AGX_CHIP_G18G, bytes, size, &record)) {
         CHECK(record.length == size);
         counts[record.kind]++;
      }
   }

   CHECK(counts[AGX_PPP_DECODED_RECORD_VIEWPORT] == 1);
   CHECK(counts[AGX_PPP_DECODED_RECORD_W_CLAMP_CULL2] == 1);
   CHECK(counts[AGX_PPP_DECODED_RECORD_HAL300_DEPTH_BIAS_SCISSOR_BASES] == 1);
   CHECK(counts[AGX_PPP_DECODED_RECORD_HAL300_DEPTH_STENCIL] == 2);
}

/* Pack one draw's nine graphics records and compare them with the capture.
 * `skip_face_bit0` masks the one bit this packer has no field for. */
static void
test_graphics(const struct g18g_capture_mode *mode,
              const struct agx_ppp_hal300_graphics_fields *fields,
              const uint32_t *offsets, unsigned first, unsigned last,
              bool skip_face_bit0)
{
   const char *what = mode->name;
   struct agx_ppp_hal300_graphics_core g18g;

   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, fields, &g18g));

   for (unsigned i = first; i <= last; ++i) {
      const struct g18g_capture_ppp *ppp = find_ppp(mode, offsets[i]);
      CHECK(ppp != NULL);
      if (!ppp)
         continue;

      uint32_t expected[16];
      CHECK(ppp->size_words * 4 == g18g.lengths[i]);
      if (ppp->size_words > 16)
         continue;

      memcpy(expected, ppp->words, ppp->size_words * 4);
      if (skip_face_bit0 && i == 4) {
         /* The always/replace face words: 0x07200f01 on the J873 (and on
          * M5), 0x07200f00 from the packer. */
         CHECK(expected[2] == 0x07200f01 && expected[3] == 0x07200f01);
         expected[2] &= ~1u;
         expected[3] &= ~1u;
      }

      CHECK(equal_words(g18g.bytes + g18g.offsets[i], g18g.lengths[i],
                        expected, ppp->size_words));
   }
}

static void
test_render(const struct g18g_render *render)
{
   const char *what = render->name;
   const struct g18g_capture_mode *mode = find_mode(render->name);
   CHECK(mode != NULL && !mode->compute);
   if (!mode || mode->compute)
      return;

   uint32_t index_lists[2] = {0};
   if (test_vdm_decode(mode, render, index_lists) != render->draws)
      return;

   test_ppp_decode(mode);

   /* VDM State Update. The capture is Metal's first-vertex convention, the
    * raw selector 0; GL's default last vertex is 2. */
   const struct agx_vdm_state_update_draw_info draw = {
      .source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled = true,
      .raw_provoking_selector = 0,
      .template_info = {
         .pipeline = render->vertex_program,
         .raw_vs_output_count_0 = render->vertex_outputs,
         .raw_active_context_selector = UINT32_MAX,
      },
   };
   struct agx_vdm_state_update state, state_g17g;
   CHECK(agx_vdm_state_update_draw_pack(AGX_CHIP_G18G, &draw, &state));
   CHECK(equal_words(state.bytes, state.length,
                     &mode->stream[VDM_STATE_OFFSET / 4],
                     AGX_VDM_STATE_UPDATE_LENGTH / 4));
   CHECK(agx_vdm_state_update_draw_pack(AGX_CHIP_G17G, &draw, &state_g17g));
   CHECK(memcmp(&state, &state_g17g, sizeof(state)) == 0);

   /* The nine graphics records of the first (or only) draw. */
   test_graphics(mode, &render->graphics, graphics_offsets, 0,
                 AGX_PPP_HAL300_GRAPHICS_RECORDS - 1,
                 render->second_stencil != 0);

   /* The stencil render's second draw re-emits only the face and ISP
    * records, between its two Index Lists. */
   if (render->second_stencil) {
      struct agx_ppp_hal300_graphics_fields second = render->graphics;
      second.front_stencil = second.back_stencil =
         render->second_stencil;
      const uint32_t offsets[AGX_PPP_HAL300_GRAPHICS_RECORDS] = {
         [4] = 0xa0,
         [5] = 0xa8,
      };
      CHECK(index_lists[0] < 0xa0 && 0xa8 < index_lists[1]);
      test_graphics(mode, &second, offsets, 4, 5, false);
   }

   /* Direct non-indexed Index Lists: the tailed form, one per draw. */
   for (unsigned d = 0; d < render->draws; ++d) {
      struct agx_vdm_index_list_direct_words words;
      CHECK(agx_vdm_index_list_direct_pack_words_unk2(
         AGX_CHIP_G18G, render->primitive, false, false, 0, 0, render->count,
         1, render->starts[d], 0, 7, &words));
      CHECK(words.count == 5);
      CHECK(memcmp(words.words, &mode->stream[index_lists[d] / 4],
                   words.count * 4) == 0);
   }

   /* Stream Terminate. */
   struct agx_vdm_stream_terminate_core terminate;
   CHECK(agx_vdm_stream_terminate_core_pack(AGX_CHIP_G18G, &terminate));
   CHECK(equal_words(terminate.bytes, terminate.length,
                     &mode->stream[mode->stream_words - 1], 1));

   /* The depth/stencil block, both in Apple's prefix and in the draw. */
   const struct agx_ppp_hal300_depth_stencil_fields depth_stencil = {
      .fragment_control_raw = 0x000007aa,
      .fragment_front_face_raw = 0xe2e00000,
      .fragment_back_face_raw = 0xf0c00000,
      .fragment_front_stencil_raw = 0x00023020,
      .fragment_back_stencil_raw = 0x0001ffff,
   };
   struct agx_ppp_hal300_depth_stencil_core ds;
   CHECK(agx_ppp_hal300_depth_stencil_core_pack(AGX_CHIP_G18G, &depth_stencil,
                                                &ds));
   const uint32_t ds_offsets[] = {PREFIX_DEPTH_STENCIL_OFFSET, 0x44};
   for (unsigned i = 0; i < 2; ++i) {
      const struct g18g_capture_ppp *ppp = find_ppp(mode, ds_offsets[i]);
      CHECK(ppp && equal_words(ds.bytes, ds.length, ppp->words,
                               ppp->size_words));
   }

   /* Viewport, repacked from its unpacked fields, and only as Gen7. */
   const struct g18g_capture_ppp *vp = find_ppp(mode, VIEWPORT_OFFSET);
   CHECK(vp != NULL);
   if (vp) {
      uint8_t bytes[MAX_BYTES];
      struct agx_ppp_viewport_payload_fields fields;
      struct agx_ppp_viewport_payload_core packed;
      store_words(vp->words, vp->size_words, bytes);
      CHECK(agx_ppp_viewport_payload_core_unpack(
         AGX_CHIP_G18G, bytes, vp->size_words * 4, &fields));
      CHECK(fields.viewport_count == 1);
      CHECK(agx_ppp_viewport_payload_core_pack(AGX_CHIP_G18G, &fields,
                                               &packed));
      CHECK(equal_words(packed.bytes, packed.length, vp->words,
                        vp->size_words));
      CHECK(!agx_ppp_viewport_payload_core_unpack(
         AGX_CHIP_G17P, bytes, vp->size_words * 4, &fields));
   }

   /* W-clamp/cull2: the Gen7 default witness. */
   const struct agx_ppp_w_clamp_cull2_fields wc = {.cull2_raw = 0x80};
   struct agx_ppp_w_clamp_cull2_core wc_packed;
   CHECK(agx_ppp_w_clamp_cull2_core_pack(AGX_CHIP_G18G, &wc, &wc_packed));
   const struct g18g_capture_ppp *wc_ppp = find_ppp(mode, graphics_offsets[6]);
   CHECK(wc_ppp && equal_words(wc_packed.bytes, wc_packed.length,
                               wc_ppp->words, wc_ppp->size_words));

   /* Depth-bias/scissor bases, repacked from their unpacked fields. */
   const struct g18g_capture_ppp *db =
      find_ppp(mode, DEPTH_BIAS_SCISSOR_OFFSET);
   CHECK(db != NULL);
   if (db) {
      uint8_t bytes[MAX_BYTES];
      struct agx_ppp_hal300_depth_bias_scissor_bases_fields fields;
      struct agx_ppp_hal300_depth_bias_scissor_bases packed;
      store_words(db->words, db->size_words, bytes);
      CHECK(agx_ppp_hal300_depth_bias_scissor_bases_unpack(
         AGX_CHIP_G18G, bytes, db->size_words * 4, &fields));
      CHECK(agx_ppp_hal300_depth_bias_scissor_bases_pack(AGX_CHIP_G18G,
                                                         &fields, &packed));
      CHECK(equal_words(packed.bytes, packed.length, db->words,
                        db->size_words));
   }

   /* Apple's prefix: two 33-word records Mesa does not emit. They are
    * checked only for shape so a regenerated vector set that drops them is
    * noticed here. */
   const struct g18g_capture_ppp *pa = find_ppp(mode, PREFIX_A_OFFSET);
   const struct g18g_capture_ppp *pb = find_ppp(mode, PREFIX_B_OFFSET);
   CHECK(pa && pa->size_words == 33 && pa->words[0] == 0x13d03762);
   CHECK(pb && pb->size_words == 33 && pb->words[0] == 0x13d03760);

   /* The J873 kernel command publishes 0x00ff00ff for every 256x256 render
    * with depth or stencil storage (kcmd +0x554) and zero otherwise. */
   const uint64_t storage = render->zls ? 0x100000c8000 : 0;
   CHECK(agx_zls_pixels(AGX_CHIP_G18G, 256, 256, storage, 0, 0, 0) ==
         (render->zls ? 0x00ff00ffu : 0));
}

static void
test_compute(void)
{
   const char *what = "compute-add";
   const struct g18g_capture_mode *mode = find_mode("compute-add");
   CHECK(mode != NULL && mode->compute);
   if (!mode)
      return;

   uint8_t bytes[MAX_BYTES];
   const size_t size = mode->stream_words * 4;
   CHECK(size == AGX3_CDM_LAUNCH_SIZE + AGX3_CDM_BARRIER_SIZE +
                    AGX3_CDM_STREAM_TERMINATE_SIZE);
   store_words(mode->stream, mode->stream_words, bytes);

   struct agx3_cdm_launch_info launch;
   CHECK(agx3_cdm_launch_unpack(AGX_CHIP_G18G, bytes, size, &launch));
   CHECK(launch.global_size[0] == 4096 && launch.local_size[0] == 64);
   uint8_t packed[AGX3_CDM_LAUNCH_SIZE];
   CHECK(agx3_cdm_launch_pack(&launch, packed));
   CHECK(memcmp(packed, bytes, AGX3_CDM_LAUNCH_SIZE) == 0);

   struct agx3_cdm_barrier_info barrier;
   CHECK(agx3_cdm_barrier_unpack(AGX_CHIP_G18G, bytes + AGX3_CDM_LAUNCH_SIZE,
                                 AGX3_CDM_BARRIER_SIZE, &barrier));
   CHECK(barrier.flags == 0);
   uint8_t barrier_packed[AGX3_CDM_BARRIER_SIZE];
   CHECK(agx3_cdm_barrier_pack(&barrier, barrier_packed));
   CHECK(memcmp(barrier_packed, bytes + AGX3_CDM_LAUNCH_SIZE,
                AGX3_CDM_BARRIER_SIZE) == 0);

   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   CHECK(agx3_cdm_stream_terminate_pack(AGX_CHIP_G18G, terminate));
   CHECK(memcmp(terminate,
                bytes + AGX3_CDM_LAUNCH_SIZE + AGX3_CDM_BARRIER_SIZE,
                AGX3_CDM_STREAM_TERMINATE_SIZE) == 0);

   /* The whole stream decodes, and exactly as G17G decodes it. */
   size_t offset = 0;
   unsigned records = 0;
   while (offset < size) {
      struct agx3_cdm_decoded_record record, g17g;
      if (!agx3_cdm_decode_one(AGX_CHIP_G18G, bytes + offset, size - offset,
                               &record)) {
         CHECK(!"G18G refuses a captured CDM record");
         return;
      }
      CHECK(agx3_cdm_decode_one(AGX_CHIP_G17G, bytes + offset, size - offset,
                                &g17g));
      CHECK(g17g.kind == record.kind && g17g.length == record.length);
      offset += record.length;
      records++;
   }
   CHECK(records == 3 && offset == size);

   /* These families remain ungrounded. Indirect launch is grounded
    * separately for G18G. */
   CHECK(!agx3_cdm_secondary_barrier_target_grounded(AGX_CHIP_G18G));
   CHECK(!agx3_cdm_stream_link_target_grounded(AGX_CHIP_G18G));
   struct agx3_cdm_indirect_launch_info indirect_info = {
      .chip = AGX_CHIP_G18G,
      .word0_payload = 0x10080000,
      .pipeline = UINT64_C(0x100fff58400),
      .opaque_word3 = 0x40000001,
      .indirect_parameters = UINT64_C(0x10000123440),
   };
   static const uint32_t indirect_words[] = {
      0x10080000, 0x01000000, 0x03ffd610, 0x40000001, 0x00000100, 0x00123440,
   };
   uint8_t indirect_packed[AGX3_CDM_INDIRECT_LAUNCH_SIZE];
   CHECK(agx3_cdm_indirect_launch_pack(&indirect_info, indirect_packed));
   CHECK(equal_words(indirect_packed, sizeof(indirect_packed), indirect_words,
                     sizeof(indirect_words) / sizeof(indirect_words[0])));
   indirect_info.pipeline |= 1;
   CHECK(!agx3_cdm_indirect_launch_pack(&indirect_info, indirect_packed));

   /* The flag-free barrier and the flags=5 strong flush are captured:
    * every other flag value is refused on G18G in both directions,
    * although G17G admits all of them.  flags=5 is grounded from rel24b+
    * chained launches (hk_cmd_dispatch.c). */
   for (unsigned flags = 1; flags <= 0xf; ++flags) {
      if (flags == 5)
         continue;
      CHECK(!agx3_cdm_barrier_flags_grounded(AGX_CHIP_G18G, flags));
      CHECK(agx3_cdm_barrier_flags_grounded(AGX_CHIP_G17G, flags));

      const struct agx3_cdm_barrier_info flagged = {
         .chip = AGX_CHIP_G18G,
         .flags = flags,
      };
      CHECK(!agx3_cdm_barrier_pack(&flagged, barrier_packed));

      uint8_t word[AGX3_CDM_BARRIER_SIZE];
      const uint32_t flagged_word = agx3_cdm_barrier_word(flags);
      store_words(&flagged_word, 1, word);
      CHECK(!agx3_cdm_barrier_unpack(AGX_CHIP_G18G, word, sizeof(word),
                                     &barrier));
      CHECK(agx3_cdm_barrier_unpack(AGX_CHIP_G17G, word, sizeof(word),
                                    &barrier) &&
            barrier.flags == flags);
   }

   /* The flags=5 strong flush is grounded on G18G hardware. */
   CHECK(agx3_cdm_barrier_flags_grounded(AGX_CHIP_G18G, 5));
   const struct agx3_cdm_barrier_info strong = {
      .chip = AGX_CHIP_G18G,
      .flags = 5,
   };
   CHECK(agx3_cdm_barrier_pack(&strong, barrier_packed));
   static const uint32_t strong_words[] = {UINT32_C(0x60001560)};
   CHECK(equal_words(barrier_packed, sizeof(barrier_packed), strong_words, 1));
   CHECK(agx3_cdm_barrier_unpack(AGX_CHIP_G18G, barrier_packed,
                                 sizeof(barrier_packed), &barrier) &&
         barrier.flags == 5);

   /* No capture has a substream barrier, even a flag-free one. */
   const struct agx3_cdm_substream_barrier_info substream = {
      .chip = AGX_CHIP_G18G,
      .flags = 0,
      .substream = 0,
   };
   uint8_t substream_packed[AGX3_CDM_BARRIER_SIZE];
   CHECK(!agx3_cdm_substream_barrier_pack(&substream, substream_packed));
   struct agx3_cdm_substream_barrier_info substream_g17g = substream;
   substream_g17g.chip = AGX_CHIP_G17G;
   CHECK(agx3_cdm_substream_barrier_pack(&substream_g17g, substream_packed));
   struct agx3_cdm_substream_barrier_info substream_out;
   CHECK(!agx3_cdm_substream_barrier_unpack(AGX_CHIP_G18G, substream_packed,
                                            sizeof(substream_packed),
                                            &substream_out));

   /* Nor link padding, although it shares the launch record's family on
    * every earlier generation. */
   CHECK(!agx3_cdm_link_padding_target_grounded(AGX_CHIP_G18G));
   CHECK(agx3_cdm_link_padding_target_grounded(AGX_CHIP_G17G));
   const struct agx3_cdm_link_padding_info padding = {
      .chip = AGX_CHIP_G18G,
      .count = 1,
   };
   uint8_t padding_packed[AGX3_CDM_LINK_PADDING_SIZE];
   CHECK(!agx3_cdm_link_padding_pack(&padding, padding_packed));
   struct agx3_cdm_link_padding_info padding_g17g = padding;
   padding_g17g.chip = AGX_CHIP_G17G;
   CHECK(agx3_cdm_link_padding_pack(&padding_g17g, padding_packed));
   struct agx3_cdm_link_padding_info padding_out;
   CHECK(!agx3_cdm_link_padding_unpack(AGX_CHIP_G18G, padding_packed,
                                       sizeof(padding_packed), &padding_out));
   struct agx3_cdm_decoded_record padding_record;
   CHECK(!agx3_cdm_decode_one(AGX_CHIP_G18G, padding_packed,
                              sizeof(padding_packed), &padding_record));
}

static void
test_descriptors(void)
{
   const char *what = "descriptors";

   /* The colour target's PBE descriptor (root 27 +0x620 of the colour
    * render) and the same image's texture descriptor (root 27 +0x20):
    * linear RGBA8 256x256 at 0x10000058000, stride 1024. */
   const uint32_t pbe_words[8] = {0xffe40a02, 0x00007f80, 0x00005800,
                                  0x0003f010};
   const uint32_t texture_words[8] = {0xf6880a02, 0x0007f80f, 0x00005800,
                                      0x000fc010};
   uint8_t bytes[32];

   store_words(pbe_words, 8, bytes);
   struct agx_pbe_agx3_2d_fields pbe;
   struct agx_pbe_agx3_descriptor repacked;
   CHECK(agx_pbe_agx3_2d_unpack(AGX_PBE_AGX3_GENERATION_10, bytes,
                                sizeof(bytes), &pbe));
   CHECK(pbe.buffer == 0x10000058000 && pbe.width == 256 &&
         pbe.height == 256);
   CHECK(agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_10, &pbe, &repacked));
   CHECK(repacked.length == sizeof(bytes) &&
         memcmp(repacked.bytes, bytes, sizeof(bytes)) == 0);

   store_words(texture_words, 8, bytes);
   struct AGX_TEXTURE_GEN10 texture;
   uint32_t texture_repacked[8] = {0};
   CHECK(AGX_TEXTURE_GEN10_unpack(stderr, bytes, &texture));
   CHECK(texture.width == 256 && texture.height == 256);
   AGX_TEXTURE_GEN10_pack(texture_repacked, &texture);
   CHECK(memcmp(texture_repacked, texture_words, sizeof(texture_words)) == 0);

   /* The texture-sample render's sampler: Metal's default descriptor with
    * nearest filters and clamp-to-edge, element 2 of the 32-byte sampler
    * heap (root 2 +0x40). Only Gen5 represents its maximum LOD of 15. */
   const uint32_t sampler_words[2] = {0x000f0000, 0x00000780};
   uint32_t sampler_repacked[2] = {0};
   store_words(sampler_words, 2, bytes);

   struct AGX_SAMPLER_GEN5 gen5;
   CHECK(AGX_SAMPLER_GEN5_unpack(stderr, bytes, &gen5));
   CHECK(gen5.maximum_lod == 15.0f && gen5.minimum_lod == 0.0f);
   AGX_SAMPLER_GEN5_pack(sampler_repacked, &gen5);
   CHECK(memcmp(sampler_repacked, sampler_words, sizeof(sampler_words)) == 0);

   struct AGX_SAMPLER_HAL200 hal200;
   CHECK(AGX_SAMPLER_HAL200_unpack(stderr, bytes, &hal200));
   memset(sampler_repacked, 0, sizeof(sampler_repacked));
   AGX_SAMPLER_HAL200_pack(sampler_repacked, &hal200);
   CHECK(sampler_repacked[0] == 0x000e0000);
}

static void
test_closed_forms(void)
{
   const char *what = "closed forms";
   struct agx_vdm_index_list_direct_words words;
   struct agx_vdm_index_list_indirect_words indirect;

   /* Direct indexed draws are captured: Apple's words for three u16
    * triangles and for a restarted nine-index u16 strip, both with indices at
    * 0x10000018300 (sessions-20260925-g1 indexed-u16 and indexed-strip). */
   static const uint32_t apple_u16_triangles[] = {
      0x69f20600, 0x00018300, 3, 1, 0, 7, 0xf, 1,
   };
   static const uint32_t apple_u16_strip[] = {
      0x69f30900, 0x00018300, 9, 1, 0, 7, 0xf, 1,
   };
   CHECK(agx_vdm_index_list_direct_pack_words_unk2(
      AGX_CHIP_G18G, 6, true, false, AGX_VDM_INDEX_LIST_INDEX_SIZE_U16,
      0x10000018300, 3, 1, 0, 0xf, 7, &words));
   CHECK(words.count == 8 &&
         memcmp(words.words, apple_u16_triangles,
                sizeof(apple_u16_triangles)) == 0);
   CHECK(agx_vdm_index_list_direct_pack_words_unk2(
      AGX_CHIP_G18G, 9, true, true, AGX_VDM_INDEX_LIST_INDEX_SIZE_U16,
      0x10000018300, 9, 1, 0, 0xf, 7, &words));
   CHECK(words.count == 8 &&
         memcmp(words.words, apple_u16_strip, sizeof(apple_u16_strip)) == 0);

   /* G18-INDIRECT-DBIAS-PBE.md section 1 grounds indirect draw words;
    * G18-RASTERIZER-DISCARD.md section 7 grounds the VDM barrier. */
   CHECK(agx_vdm_index_list_direct_pack_words(
      AGX_CHIP_G17G, 6, true, false, AGX_VDM_INDEX_LIST_INDEX_SIZE_U16,
      0x10000100000, 3, 1, 0, 0, &words));
   static const uint32_t apple_indirect_triangles[] = {
      0x6c040600, 0x00000100, 0x00100000, 7,
   };
   CHECK(agx_vdm_index_list_indirect_pack_words_unk2(
       AGX_CHIP_G18G, 6, false, false, 0, 0, 0x10000100000, 0, 0, 7, &indirect));
   CHECK(indirect.count == 4 &&
         memcmp(indirect.words, apple_indirect_triangles,
                sizeof(apple_indirect_triangles)) == 0);
   CHECK(!agx_vdm_index_list_indirect_pack_words_unk2(
      AGX_CHIP_G18G, 6, false, false, 0, 0, 0x10000100001, 0, 0, 7, &indirect));
   struct agx_vdm_barrier_core vdm_barrier;
   /* Apple's word lacks the USC cache invalidate (bit 3) that we add. */
   static const uint8_t apple_vdm_barrier[] = {0x60, 0x09, 0x00, 0x21};
   static const uint8_t packed_vdm_barrier[] = {0x68, 0x09, 0x00, 0x21};
   CHECK(agx_vdm_barrier_core_pack(AGX_CHIP_G18G, false, &vdm_barrier));
   CHECK(vdm_barrier.length == sizeof(packed_vdm_barrier) &&
         memcmp(vdm_barrier.bytes, packed_vdm_barrier,
                sizeof(packed_vdm_barrier)) == 0);
   CHECK(agx_vdm_barrier_core_unpack(AGX_CHIP_G18G, apple_vdm_barrier,
                                     sizeof(apple_vdm_barrier)));
   CHECK(agx_vdm_barrier_core_unpack(AGX_CHIP_G18G, packed_vdm_barrier,
                                     sizeof(packed_vdm_barrier)));
   CHECK(!agx_vdm_barrier_core_unpack(AGX_CHIP_G18G, apple_vdm_barrier,
                                     sizeof(apple_vdm_barrier) - 1));
   /* Stream links and tessellation still lack G18 evidence. */
   CHECK(!agx_vdm_stream_link_core_supported(AGX_CHIP_G18G));
   CHECK(!agx_vdm_tessellate_direct_target_supported(AGX_CHIP_G18G));

   /* Lines are a G18G shape; a point size on a line has no capture. */
   struct agx_ppp_hal300_graphics_fields lines = {
      .fragment_program = 0x10000168780,
      .object_type = AGX_PPP_HAL300_LINES,
      .point_size_output = true,
   };
   struct agx_ppp_hal300_graphics_core out;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &lines, &out));
   lines.point_size_output = false;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &lines, &out));
   lines.object_type = (enum agx_ppp_hal300_object_type)3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &lines, &out));
}

int
main(void)
{
   const char *what = "vectors";

   CHECK(sizeof(g18g_capture_modes) / sizeof(g18g_capture_modes[0]) == 13);
   CHECK(sizeof(renders) / sizeof(renders[0]) == 12);

   for (unsigned i = 0; i < sizeof(renders) / sizeof(renders[0]); ++i)
      test_render(&renders[i]);

   test_compute();
   test_descriptors();
   test_closed_forms();

   if (failures) {
      fprintf(stderr, "G18G HAL300 records: %u of %u checks failed\n",
              failures, checks);
      return 1;
   }

   printf("G18G HAL300 records: all %u checks passed\n", checks);
   return 0;
}
