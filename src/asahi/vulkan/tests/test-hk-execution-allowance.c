/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_execution_allowance.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_cmd_buffer.h"
#undef NDEBUG
#include <assert.h>

/* The production merge copies or links bytes before combining metadata. A
 * synthetic link emitter keeps this a CPU-only test of both merge paths.
 */
static unsigned links;
void *
hk_cmd_buffer_emit_cs_jump(struct hk_cmd_buffer *cmd, void *current,
                          uint64_t target, bool vdm)
{
   assert(cmd == NULL && !vdm);
   links++;
   return (uint8_t *)current + AGX_CDM_STREAM_LINK_LENGTH;
}

/* Merged streams here record no attachments; with more than one merge call
 * the compiler keeps hk_cs_merge_cdm() out of line, with its attachment loop. */
bool
hk_drm_attachment_set_add(struct hk_drm_attachment_set *set, uint64_t pointer,
                          uint64_t size)
{
   assert(!"no attachments are recorded");
   return false;
}

static void
check_merge(bool first, bool second, bool linked)
{
   uint8_t a_bytes[HK_CS_SPACE_PADDING + 128] = {0}, b_bytes[32] = {0};
   struct hk_cs a = {
      .type = HK_CS_CDM, .start = a_bytes, .current = a_bytes + 8,
      .end = a_bytes + sizeof(a_bytes), .as_build = first,
      .stream_linked = linked,
   };
   struct hk_cs b = {
      .type = HK_CS_CDM, .start = b_bytes, .current = b_bytes + 8,
      .end = b_bytes + sizeof(b_bytes), .as_build = second,
   };
   const unsigned before = links;
   hk_cs_merge_cdm(&a, &b);
   assert(a.as_build == (first || second));
   assert(b.as_build == second);
   assert(links == before + linked);
}

/* Memory scope across hk_cs_merge_cdm(): a stream whose trailing strong flush
 * has memory scope must keep it when the next stream's opening barrier is
 * folded into that record, and memory_flush_end must stay exact so a later
 * merge or cache flush sees the right tail. */
#define LAUNCH_BYTES 8
static const uint32_t plain_flush = UINT32_C(0x60001560);
static const uint32_t memory_flush = UINT32_C(0x60001560) |
                                     AGX3_CDM_BARRIER_FLUSH_MEMORY;

static uint32_t
word_at(const void *p)
{
   return agx3_cmdstream_load_le32((const uint8_t *)p);
}

/* A stream holding a launch and then a strong flush, with or without memory
 * scope, whose markers point past that flush. */
static void
init_tail(struct hk_cs *cs, uint8_t *bytes, size_t size, bool memory)
{
   memset(bytes, 0xaa, LAUNCH_BYTES);
   agx3_cmdstream_store_le32(bytes + LAUNCH_BYTES,
                             memory ? memory_flush : plain_flush);
   *cs = (struct hk_cs){
      .type = HK_CS_CDM, .start = bytes,
      .current = bytes + LAUNCH_BYTES + AGX3_CDM_BARRIER_SIZE,
      .end = bytes + size,
   };
   cs->strong_flush_end = cs->current;
   cs->memory_flush_end = memory ? cs->current : NULL;
}

/* A stream that opens with a strong flush and, when launch is set, has a
 * launch after it. */
static void
init_opener(struct hk_cs *cs, uint8_t *bytes, size_t size, bool memory,
            bool launch)
{
   agx3_cmdstream_store_le32(bytes, memory ? memory_flush : plain_flush);
   if (launch)
      memset(bytes + AGX3_CDM_BARRIER_SIZE, 0xbb, LAUNCH_BYTES);
   *cs = (struct hk_cs){
      .type = HK_CS_CDM, .start = bytes,
      .current = bytes + AGX3_CDM_BARRIER_SIZE + (launch ? LAUNCH_BYTES : 0),
      .end = bytes + size,
      .leading_barrier_end = bytes + AGX3_CDM_BARRIER_SIZE,
   };
   if (!launch) {
      cs->strong_flush_end = cs->current;
      cs->memory_flush_end = memory ? cs->current : NULL;
   }
}

static void
check_merge_memory_scope(void)
{
   uint8_t a_bytes[HK_CS_SPACE_PADDING + 256], b_bytes[64];
   struct hk_cs a, b;
   uint8_t *tail = a_bytes + LAUNCH_BYTES;

   assert(agx3_cdm_barrier_word(5) == plain_flush);

   /* A memory-scoped tail folded with a plain opening barrier keeps its
    * scope. b's launch then ends the stream, so no memory flush is last. */
   init_tail(&a, a_bytes, sizeof(a_bytes), true);
   init_opener(&b, b_bytes, sizeof(b_bytes), false, true);
   hk_cs_merge_cdm(&a, &b);
   assert(word_at(tail) == memory_flush);
   assert(a.current == tail + AGX3_CDM_BARRIER_SIZE + LAUNCH_BYTES);
   assert(word_at(tail + AGX3_CDM_BARRIER_SIZE) == 0xbbbbbbbb);
   assert(a.memory_flush_end == NULL);

   /* A barrier-only second stream folds away entirely: the scoped record
    * stays last, and both markers still point just past it. */
   init_tail(&a, a_bytes, sizeof(a_bytes), true);
   init_opener(&b, b_bytes, sizeof(b_bytes), false, false);
   hk_cs_merge_cdm(&a, &b);
   assert(word_at(tail) == memory_flush);
   assert(a.current == tail + AGX3_CDM_BARRIER_SIZE);
   assert(a.memory_flush_end == a.current);
   assert(a.strong_flush_end == a.current);

   /* Repeated merges: more barrier-only streams keep the scoped record and
    * its marker; a stream with a launch then clears the marker, and a later
    * plain barrier is appended as its own record without memory scope. */
   for (unsigned i = 0; i < 3; ++i) {
      init_opener(&b, b_bytes, sizeof(b_bytes), false, false);
      hk_cs_merge_cdm(&a, &b);
      assert(word_at(tail) == memory_flush);
      assert(a.current == tail + AGX3_CDM_BARRIER_SIZE);
      assert(a.memory_flush_end == a.current);
   }
   init_opener(&b, b_bytes, sizeof(b_bytes), false, true);
   hk_cs_merge_cdm(&a, &b);
   assert(word_at(tail) == memory_flush);
   assert(a.memory_flush_end == NULL);
   uint8_t *const after_launch = a.current;
   init_opener(&b, b_bytes, sizeof(b_bytes), false, false);
   hk_cs_merge_cdm(&a, &b);
   assert(a.current == after_launch + AGX3_CDM_BARRIER_SIZE);
   assert(word_at(after_launch) == plain_flush);
   assert(a.memory_flush_end == NULL);

   /* A memory-scoped opener folded onto a plain tail brings its scope, and
    * its marker maps onto the record that absorbed it. */
   init_tail(&a, a_bytes, sizeof(a_bytes), false);
   init_opener(&b, b_bytes, sizeof(b_bytes), true, false);
   hk_cs_merge_cdm(&a, &b);
   assert(word_at(tail) == memory_flush);
   assert(a.current == tail + AGX3_CDM_BARRIER_SIZE);
   assert(a.memory_flush_end == a.current);

   /* Without a trailing barrier there is no fold: b is copied whole and its
    * own scoped record and marker follow it. */
   init_tail(&a, a_bytes, sizeof(a_bytes), false);
   a.current = a_bytes + LAUNCH_BYTES;
   a.strong_flush_end = NULL;
   init_opener(&b, b_bytes, sizeof(b_bytes), true, false);
   hk_cs_merge_cdm(&a, &b);
   assert(word_at(a_bytes + LAUNCH_BYTES) == memory_flush);
   assert(a.current == a_bytes + LAUNCH_BYTES + AGX3_CDM_BARRIER_SIZE);
   assert(a.memory_flush_end == a.current);
}

static void
check_as_contracts(void)
{
   struct agx_device dev = {
      .chip = AGX_CHIP_G17P,
      .params = {.chip_id = 0x6040, .gpu_generation = 16, .gpu_variant = 'S'},
   };
   /* Exercise every combination: AS construction cannot imply traversal,
    * and bounded execution alone cannot select the extended-prior ABI. */
   for (unsigned bits = 0; bits < 8; ++bits) {
      dev.params.features =
         ((bits & 1) ? DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS : 0) |
         ((bits & 2) ? DRM_ASAHI_FEATURE_RENDER_RAY_QUERY : 0) |
         ((bits & 4) ? DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD : 0);
      assert(hk_g16s_as_enabled(&dev) == ((bits & 5) == 5));
      assert(hk_g17p_rt_enabled(&dev) == ((bits & 3) == 3));
      assert(hk_gen1_as_enabled(&dev) ==
             (((bits & 5) == 5) || ((bits & 3) == 3)));
   }
   dev.params.features = UINT64_MAX;
   dev.params.chip_id = 0x8140;
   assert(!hk_g16s_as_enabled(&dev) && hk_g17p_rt_enabled(&dev));
   dev.params.chip_id = 0x6040;
   dev.params.gpu_variant = 'G';
   assert(!hk_g16s_as_enabled(&dev));
   dev.params.gpu_variant = 'S';
   dev.params.gpu_generation = 17;
   assert(!hk_g16s_as_enabled(&dev));
   dev.params.gpu_generation = 16;
   for (enum agx_chip chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip) {
      dev.chip = chip;
      if (chip != AGX_CHIP_G17P)
         assert(!hk_g16s_as_enabled(&dev) && !hk_g17p_rt_enabled(&dev));
   }
}

int main(void)
{
   check_as_contracts();
   for (enum agx_chip chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip)
      assert(hk_shader_binary_version(chip) ==
             (chip == AGX_CHIP_G17P || chip == AGX_CHIP_G17G ||
              chip == AGX_CHIP_G17S ? 36 :
              chip <= AGX_CHIP_G15P ? 36 : 34));
   assert(hk_as_build_flags(AGX_CHIP_G17P, 0, true) == 0);
   assert(hk_as_build_flags(AGX_CHIP_G17P, DRM_ASAHI_FEATURE_RENDER_RAY_QUERY, true) == 0);
   assert(hk_as_build_flags(AGX_CHIP_G17P, DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD, false) == 0);
   assert(hk_as_build_flags(AGX_CHIP_G17P, DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD, true) ==
          DRM_ASAHI_COMPUTE_AS_BUILD);
   assert(hk_as_build_flags(AGX_CHIP_G17P, UINT64_MAX, true) == DRM_ASAHI_COMPUTE_AS_BUILD);
   /* Even a future classic kernel advertising every bit keeps zero flags. */
   for (enum agx_chip chip = AGX_CHIP_G13G; chip <= AGX_CHIP_G17A; ++chip)
      if (chip != AGX_CHIP_G17P)
         assert(hk_as_build_flags(chip, UINT64_MAX, true) == 0);
   for (unsigned a = 0; a < 2; ++a)
      for (unsigned b = 0; b < 2; ++b)
         for (unsigned link = 0; link < 2; ++link)
            check_merge(a, b, link);
   check_merge_memory_scope();
   return 0;
}
