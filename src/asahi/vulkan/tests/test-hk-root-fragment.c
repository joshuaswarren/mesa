/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_cmd_buffer.h"

static uint64_t checks;
#define CHECK(condition)                                                       \
   do {                                                                        \
      ++checks;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "line %d: %s\n", __LINE__, #condition);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

/* Independent reference: select by comparing the actual published roots. */
static uint64_t
fresh_select(struct hk_graphics_state *gfx, uint8_t reads,
             uint32_t required_size)
{
   const uint8_t hidden = HK_ROOT_READS_ADDRESS | HK_ROOT_READS_VERTEX_INPUT |
                          HK_ROOT_READS_DYNAMIC_BUFFERS;
   const uint32_t full = sizeof(struct hk_root_descriptor_table);
   const uint32_t req = required_size ? MIN2(required_size, full) : full;
   const uint32_t cur_size = gfx->root_size ? MIN2(gfx->root_size, full) : full;
   const uint32_t prev_size =
      gfx->fs_root_size ? MIN2(gfx->fs_root_size, full) : full;
   if (gfx->fs_root && gfx->fs_root != gfx->root && gfx->root_cpu &&
       prev_size >= req && !(reads & hidden) &&
       hk_root_matches_for_fragment(gfx->fs_root_cpu, prev_size, gfx->root_cpu,
                                    cur_size) &&
       (!(reads & HK_ROOT_READS_PUSH) ||
        hk_root_matches_push(gfx->fs_root_cpu, prev_size, gfx->root_cpu,
                             cur_size)))
      return gfx->fs_root;
   gfx->fs_root = gfx->root;
   gfx->fs_root_cpu = gfx->root_cpu;
   gfx->fs_root_size = gfx->root_size;
   return gfx->root;
}

/* The root steps hk_g17p_flush_dynamic_state_fast() takes for a draw that
 * changes nothing but what the test marked, with a push-only vertex launch
 * that succeeds whenever it is tried. draw_fits stands for the draw's own
 * terms of push_launch_fits: a direct draw, a vertex shader without a
 * preamble, and a fragment shader that reads neither push constants nor the
 * root's address. */
enum fast_root_step {
   FAST_ROOT_KEEP,        /* early return: the published launch is current */
   FAST_ROOT_PUSH_LAUNCH, /* vertex launch over push constants, root kept */
   FAST_ROOT_UPLOAD,      /* full root upload */
};

static enum fast_root_step
fast_draw_root(struct hk_descriptor_state *d, bool draw_fits)
{
   const bool push_launch_fits = hk_root_push_only_dirty(d) && draw_fits;
   if (hk_root_launch_current(d, push_launch_fits))
      return FAST_ROOT_KEEP;
   if (push_launch_fits && hk_root_push_only_dirty(d)) {
      hk_root_mark_push_launched(d);
      return FAST_ROOT_PUSH_LAUNCH;
   }
   hk_root_mark_clean(d);
   return FAST_ROOT_UPLOAD;
}

static void
select_and_check_size(struct hk_graphics_state *fresh,
                      struct hk_graphics_state *cached, uint8_t reads,
                      uint32_t required_size, bool memo, bool shadow)
{
   CHECK(fresh_select(fresh, reads, required_size) ==
         hk_agx3_fragment_root_select(cached, reads, required_size, memo,
                                      shadow));
   CHECK(fresh->fs_root == cached->fs_root);
   CHECK(fresh->fs_root_cpu == cached->fs_root_cpu);
   CHECK(fresh->fs_root_size == cached->fs_root_size);
}

static void
select_and_check(struct hk_graphics_state *fresh,
                 struct hk_graphics_state *cached, uint8_t reads, bool memo,
                 bool shadow)
{
   select_and_check_size(fresh, cached, reads, 0, memo, shadow);
}

/* Publication uses a mutable cached CPU source and an immutable GPU copy.
 * Poison skipped dynamic bytes and bytes beyond root_size so accidental
 * observation cannot pass. */
static void
publish_size(struct hk_graphics_state *fresh, struct hk_graphics_state *cached,
             struct hk_root_descriptor_table *gpu,
             const struct hk_root_descriptor_table *source,
             unsigned dynamic_end, uint32_t root_size, uint64_t va,
             bool shadow)
{
   const size_t head =
      MIN2((size_t)root_size,
           hk_root_descriptor_offset(dynamic_buffers) +
              dynamic_end * sizeof(source->dynamic_buffers[0]));
   const size_t tail = hk_root_descriptor_offset(set_dynamic_buffer_start);
   memset(gpu, 0xa7, sizeof(*gpu));
   if (shadow)
      hk_root_fragment_snapshot_publish(cached, gpu, source, root_size);
   memcpy(gpu, source, head);
   if (tail < root_size)
      memcpy((uint8_t *)gpu + tail, (const uint8_t *)source + tail,
             root_size - tail);
   fresh->root = cached->root = va;
   fresh->root_cpu = cached->root_cpu = gpu;
   fresh->root_size = cached->root_size = root_size;
   if (shadow) {
      size_t out = 0;
      for (size_t b = 0; b < sizeof(*gpu); ++b) {
         if (!hk_root_reads_of(b, b + 1)) {
            const uint8_t expected =
               b < root_size ? ((const uint8_t *)gpu)[b] : 0;
            CHECK(cached->root_fragment_snapshot.bytes[out] == expected);
            ++out;
         }
      }
      CHECK(out == HK_ROOT_FRAGMENT_VISIBLE_SIZE);
      const size_t push_off = hk_root_descriptor_offset(push);
      for (size_t i = 0; i < HK_MAX_PUSH_SIZE; ++i) {
         const size_t b = push_off + i;
         const uint8_t expected =
            b < root_size ? ((const uint8_t *)gpu)[b] : 0;
         CHECK(cached->root_fragment_snapshot.push[i] == expected);
      }
   }
}

static void
publish(struct hk_graphics_state *fresh, struct hk_graphics_state *cached,
        struct hk_root_descriptor_table *gpu,
        const struct hk_root_descriptor_table *source, unsigned dynamic_end,
        uint64_t va, bool shadow)
{
   publish_size(fresh, cached, gpu, source, dynamic_end, sizeof(*gpu), va,
                shadow);
}

int
main(void)
{
   struct hk_graphics_state *fresh = calloc(1, sizeof(*fresh));
   struct hk_graphics_state *cached = calloc(1, sizeof(*cached));
   struct hk_root_descriptor_table source, gpu[35];
   CHECK(fresh && cached);

   for (unsigned mode = 0; mode < 4; ++mode) {
      const bool memo = mode & 1, shadow = mode & 2;
      for (unsigned end = 0; end <= HK_MAX_DYNAMIC_BUFFERS; ++end) {
         for (unsigned byte = 0; byte < sizeof(source); ++byte) {
            memset(fresh, 0, sizeof(*fresh));
            memset(cached, 0, sizeof(*cached));
            memset(&source, 0, sizeof(source));
            publish(fresh, cached, &gpu[0], &source, end, 0x1000, shadow);
            select_and_check(fresh, cached, 0, memo, shadow);
            const struct hk_root_fragment_snapshot prior =
               cached->fs_root_fragment_snapshot;
            const struct hk_root_descriptor_table *prior_cpu =
               cached->fs_root_fragment_snapshot_cpu;
            ((uint8_t *)&source)[byte] ^= 0x5a;
            publish(fresh, cached, &gpu[1], &source, end, 0x2000, shadow);
            for (unsigned reads = 0; reads < 16; ++reads) {
               /* Each mask must decide between the old and current roots.
                * A previous hidden-read selection must not pre-adopt it. */
               fresh->fs_root = cached->fs_root = 0x1000;
               fresh->fs_root_cpu = cached->fs_root_cpu = &gpu[0];
               fresh->fs_root_size = cached->fs_root_size = sizeof(source);
               cached->fs_root_checked_root = 0;
               cached->fs_root_checked_push_match = false;
               cached->fs_root_fragment_snapshot = prior;
               cached->fs_root_fragment_snapshot_cpu = prior_cpu;
               select_and_check(fresh, cached, reads, memo, shadow);
               select_and_check(fresh, cached, reads, memo, shadow);
            }
         }
      }

      /* Cross-shader memo safety: a non-push FS reusing fs_root across a
       * push-constant change must not satisfy a subsequent push-reading FS on
       * the same root; conversely a push-reading FS reusing fs_root across a
       * vertex-input change must not stale-cache push equality for a later
       * root whose push constants differ. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      source.push[0] = 0x33;
      publish(fresh, cached, &gpu[1], &source, 0, 0x2000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      CHECK(cached->fs_root == 0x1000);
      select_and_check(fresh, cached, HK_ROOT_READS_PUSH, memo, shadow);
      CHECK(cached->fs_root == 0x2000);

      /* Push-reading FS reuses across a vertex-input change, then a new root
       * changes push[]: non-push FS reuses, push-reading FS must adopt. */
      source.draw.attrib_base[0] = 0xdeadbeef0000ull;
      publish(fresh, cached, &gpu[2], &source, 0, 0x3000, shadow);
      select_and_check(fresh, cached, HK_ROOT_READS_PUSH, memo, shadow);
      CHECK(cached->fs_root == 0x2000);
      select_and_check(fresh, cached, 0, memo, shadow);
      CHECK(cached->fs_root == 0x2000);
      source.push[12] = 0x77;
      publish(fresh, cached, &gpu[3], &source, 0, 0x4000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      CHECK(cached->fs_root == 0x2000);
      select_and_check(fresh, cached, HK_ROOT_READS_PUSH, memo, shadow);
      CHECK(cached->fs_root == 0x4000);

      /* Trimmed HAL200 roots: reuse holds when fs_root_size covers the
       * shader's required_size, and is rejected when a later shader requires
       * bytes beyond the trimmed upload (even if those bytes are zero). */
      const uint32_t push_off = hk_root_descriptor_offset(push);
      const uint32_t push16 = push_off + 16;
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      source.sets[0] = 0x12340000;
      source.push[0] = 1;
      publish_size(fresh, cached, &gpu[0], &source, 0, push16, 0x1000, shadow);
      select_and_check_size(fresh, cached, 0, push_off, memo, shadow);
      CHECK(cached->fs_root == 0x1000);
      source.push[0] = 2;
      publish_size(fresh, cached, &gpu[1], &source, 0, push16, 0x2000, shadow);
      select_and_check_size(fresh, cached, 0, push_off, memo, shadow);
      CHECK(cached->fs_root == 0x1000);
      /* Growing to a shader that needs the full root must not reuse push16. */
      publish_size(fresh, cached, &gpu[2], &source, 0, sizeof(source), 0x3000,
                   shadow);
      select_and_check_size(fresh, cached, 0, sizeof(source), memo, shadow);
      CHECK(cached->fs_root == 0x3000);

      /* Every publication-time global and the complete tail are visible. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      for (unsigned globals = 0; globals < 32; ++globals) {
         source.root_desc_addr = 0x100000000ull + globals;
         source.printf_buffer_address = globals & 1 ? 0x12340000 : 0;
         source.sampler_heap_address = globals & 2 ? 0x23450000 : 0;
         source.physical_sparse_map = globals & 4 ? 0x34560000 : 0;
         source.physical_sparse_base = globals & 8 ? 0x45670000 : 0;
         source.physical_sparse_size = globals & 16 ? 0x100000 : 0;
         publish(fresh, cached, &gpu[globals], &source, 0,
                 0x3000 + globals * 0x1000, shadow);
         select_and_check(fresh, cached, 0, memo, shadow);
      }

      /* Mutable descriptor restoration cannot change a published snapshot. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      publish(fresh, cached, &gpu[1], &source, 0, 0x2000, shadow);
      memset(&source, 0x6d, sizeof(source));
      select_and_check(fresh, cached, 0, memo, shadow);
      select_and_check(fresh, cached, 15, memo, shadow);

      /* Failed publication retains the old CPU copy while its GPU VA is 0. */
      source.push[0] ^= 1;
      publish(fresh, cached, &gpu[2], &source, 64, 0x3000, shadow);
      fresh->root = cached->root = 0;
      select_and_check(fresh, cached, 0, memo, shadow);
      select_and_check(fresh, cached, 15, memo, shadow);
      cached->root_fragment_snapshot_cpu = NULL;
      select_and_check(fresh, cached, 0, memo, shadow);

      /* Missing current/prior metadata must use the fresh comparison. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      source.sets[0] = 1;
      publish(fresh, cached, &gpu[1], &source, 0, 0x2000, shadow);
      cached->root_fragment_snapshot_cpu = NULL;
      select_and_check(fresh, cached, 0, memo, shadow);
      publish(fresh, cached, &gpu[2], &source, 0, 0x3000, shadow);
      cached->fs_root_fragment_snapshot_cpu = NULL;
      select_and_check(fresh, cached, 0, memo, shadow);

      /* Command-buffer reset permits both pointer and VA reuse. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      fresh->root_cpu = cached->root_cpu = NULL;
      fresh->root = cached->root = 0;
      select_and_check(fresh, cached, 0, memo, shadow);
   }
   /* Root change tracking: push-only until any other change. */
   {
      struct hk_descriptor_state *d = calloc(1, sizeof(*d));
      CHECK(d != NULL);
      CHECK(!hk_root_dirty(d) && !hk_root_push_only_dirty(d));

      hk_root_mark_push_dirty(d);
      CHECK(hk_root_dirty(d) && hk_root_push_only_dirty(d));
      hk_root_mark_push_dirty(d);
      CHECK(hk_root_push_only_dirty(d));

      /* Any other change makes the root fully dirty, and a later push does
       * not make it push-only again. */
      hk_root_mark_dirty(d);
      CHECK(hk_root_dirty(d) && !hk_root_push_only_dirty(d));
      hk_root_mark_push_dirty(d);
      CHECK(hk_root_dirty(d) && !hk_root_push_only_dirty(d));

      hk_root_mark_clean(d);
      CHECK(!hk_root_dirty(d) && !hk_root_push_only_dirty(d));

      /* A change before the push keeps it fully dirty too. */
      hk_root_mark_dirty(d);
      hk_root_mark_push_dirty(d);
      CHECK(!hk_root_push_only_dirty(d));
      hk_root_mark_clean(d);
      hk_root_mark_push_dirty(d);
      hk_root_mark_clean(d);
      CHECK(!hk_root_dirty(d));
      free(d);
   }

   /* A push-only vertex launch is current for the next draw that changes
    * nothing, while the root itself stays dirty for every other reader. */
   {
      struct hk_descriptor_state *d = calloc(1, sizeof(*d));
      CHECK(d != NULL);
      CHECK(fast_draw_root(d, true) == FAST_ROOT_KEEP);

      /* Push once, draw three times: one push-only launch, then the early
       * return, as after a full upload. */
      hk_root_mark_push_dirty(d);
      CHECK(!hk_root_launch_current(d, true));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_PUSH_LAUNCH);
      CHECK(hk_root_dirty(d) && hk_root_push_only_dirty(d));
      CHECK(hk_root_launch_current(d, true));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_KEEP);
      CHECK(fast_draw_root(d, true) == FAST_ROOT_KEEP);
      CHECK(hk_root_dirty(d) && hk_root_push_only_dirty(d));

      /* New push constants need a new push-only launch. */
      hk_root_mark_push_dirty(d);
      CHECK(!hk_root_launch_current(d, true));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_PUSH_LAUNCH);
      CHECK(fast_draw_root(d, true) == FAST_ROOT_KEEP);

      /* New descriptors need a full upload. The next push launches
       * push-only again. */
      hk_root_mark_dirty(d);
      CHECK(!hk_root_launch_current(d, true) && !hk_root_push_only_dirty(d));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_UPLOAD);
      CHECK(!hk_root_dirty(d));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_KEEP);
      hk_root_mark_push_dirty(d);
      CHECK(fast_draw_root(d, true) == FAST_ROOT_PUSH_LAUNCH);
      CHECK(fast_draw_root(d, true) == FAST_ROOT_KEEP);

      /* A draw that needs the full root (an indirect draw, a fragment
       * shader reading push constants, or any path but the fast one) still
       * finds it dirty and uploads it. */
      CHECK(hk_root_dirty(d));
      CHECK(!hk_root_launch_current(d, false));
      CHECK(fast_draw_root(d, false) == FAST_ROOT_UPLOAD);
      CHECK(!hk_root_dirty(d) && hk_root_launch_current(d, false));

      /* Any upload, through any path, ends the push-only launch, and so
       * does a push that follows another change. */
      hk_root_mark_push_dirty(d);
      CHECK(fast_draw_root(d, true) == FAST_ROOT_PUSH_LAUNCH);
      hk_root_mark_clean(d);
      hk_root_mark_push_dirty(d);
      CHECK(!hk_root_launch_current(d, true));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_PUSH_LAUNCH);
      hk_root_mark_dirty(d);
      hk_root_mark_push_dirty(d);
      CHECK(!hk_root_launch_current(d, true) && !hk_root_push_only_dirty(d));
      CHECK(fast_draw_root(d, true) == FAST_ROOT_UPLOAD);
      free(d);
   }

   printf("root fragment: %" PRIu64 " checks passed\n", checks);
   free(fresh);
   free(cached);
   return 0;
}
