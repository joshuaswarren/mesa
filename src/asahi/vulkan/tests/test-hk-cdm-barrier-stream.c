/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Stream-level CPU-only test for AGX2 CDM barrier narrowing, sparse-site
 * patching, and stream-open gating through libhk. */
#undef NDEBUG
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hk_barrier.h"
#include "hk_cmd_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_device.h"

static bool force_next_chunk_switch;
static uint8_t alt_chunk[0x4000];

void __real_hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd,
                                   struct hk_cs *cs, size_t space);
void __wrap_hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd,
                                   struct hk_cs *cs, size_t space);

void
__wrap_hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                              size_t space)
{
   if (force_next_chunk_switch && space >= 0x2000) {
      force_next_chunk_switch = false;
      cs->current = alt_chunk;
      cs->end = alt_chunk + sizeof(alt_chunk);
      cs->stream_linked = true;
      return;
   }
   __real_hk_ensure_cs_has_space(cmd, cs, space);
}

static uint32_t
barrier_record(uint32_t mask)
{
   return mask | (AGX_CDM_BLOCK_TYPE_BARRIER << 29);
}

struct fixture {
   struct hk_device dev;
   struct hk_cmd_pool pool;
   struct hk_cmd_buffer cmd;
   struct hk_cs cs;
   uint8_t stream[0x8000];
};

static void
fixture_init(struct fixture *f, enum agx_chip chip, uint32_t gpu_gen,
             uint32_t perftest)
{
   memset(f, 0, sizeof(*f));
   force_next_chunk_switch = false;

   f->dev.dev.chip = chip;
   f->dev.dev.params.gpu_generation = gpu_gen;
   f->dev.perftest = perftest;
   f->dev.cdm_barrier_default_mask =
      agx2_cdm_barrier_default_word(chip, gpu_gen);
   u_rwlock_init(&f->dev.host_alias.patch_lock);

   f->pool.vk.base.device = &f->dev.vk;
   f->cmd.vk.base.device = &f->dev.vk;
   f->cmd.vk.pool = &f->pool.vk;
   f->cmd.sparse_sites = UTIL_DYNARRAY_INIT;

   f->cs.cmd = &f->cmd;
   f->cs.type = HK_CS_CDM;
   f->cs.start = f->stream;
   f->cs.current = f->stream;
   f->cs.end = f->stream + sizeof(f->stream);
   f->cmd.current_cs.cs = &f->cs;
}

static void
fixture_fini(struct fixture *f)
{
   util_dynarray_fini(&f->cmd.sparse_sites);
   u_rwlock_destroy(&f->dev.host_alias.patch_lock);
}

static bool
sparse_site_recorded(const struct hk_cmd_buffer *cmd, const uint32_t *word)
{
   util_dynarray_foreach(&cmd->sparse_sites, struct hk_sparse_site, site) {
      if (site->word == word)
         return true;
   }
   return false;
}

static VkDependencyInfo
make_ssbo_dep(VkMemoryBarrier2 *mb)
{
   *mb = (VkMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
   };
   return (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = mb,
   };
}

static VkDependencyInfo
make_broad_dep(VkMemoryBarrier2 *mb)
{
   *mb = (VkMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
   };
   return (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = mb,
   };
}

static struct agx_shader_info
make_compute_shader_info(bool uses_images)
{
   struct agx_shader_info info = {
      .stage = MESA_SHADER_COMPUTE,
      .texture_state_count = uses_images ? 1 : 0,
   };
   return info;
}

static uint32_t *
emit_launch(struct fixture *f, struct agx_shader_info *info)
{
   hk_dispatch_with_usc(&f->dev, &f->cs, info, 0x1000, agx_1d(64),
                        agx_workgroup(64, 1, 1));
   return (uint32_t *)f->cs.current - 1;
}

static uint32_t *
emit_precomp_launch(struct fixture *f)
{
   struct agx_cdm_launch_word_0_packed launch = {0};
   f->cmd.state.cs.cdm_barrier_full_needed = true;
   f->cmd.state.cs.cdm_stream_has_images = true;
   hk_dispatch_with_usc_launch(&f->dev, &f->cs, launch, 0x1000, agx_1d(64),
                               agx_workgroup(64, 1, 1), false);
   return (uint32_t *)f->cs.current - 1;
}

/* Pure SSBO barrier between two AGX2 launches narrows the barrier word between
 * them in place to 0x160 while leaving the second launch's tail at 0x1ff, and
 * late submit-time sparse patching still upgrades both records to 0xfffff. */
static void
test_pure_ssbo_narrows_between_launches(enum agx_chip chip, uint32_t gen)
{
   struct fixture f;
   fixture_init(&f, chip, gen, 0);

   struct agx_shader_info pure = make_compute_shader_info(false);
   uint32_t *b1 = emit_launch(&f, &pure);
   void *after_l1 = f.cs.current;

   assert(*b1 == barrier_record(0x1ffu));
   assert(f.cs.strong_flush_end == after_l1);
   assert(sparse_site_recorded(&f.cmd, b1));

   VkMemoryBarrier2 mb;
   VkDependencyInfo dep = make_ssbo_dep(&mb);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep));
   assert(f.cmd.state.cs.cdm_barrier_pending);
   assert(!f.cmd.state.cs.cdm_barrier_full_needed);

   uint32_t *b2 = emit_launch(&f, &pure);
   assert(*b1 == barrier_record(AGX_CDM_BARRIER_SSBO_WORD));
   assert(*b2 == barrier_record(0x1ffu));
   assert(f.cs.strong_flush_end == f.cs.current);
   assert(f.cs.stats.cmds == 2);
   assert(f.cs.stats.flushes == 2);
   assert(!f.cmd.state.cs.cdm_barrier_pending);
   assert(sparse_site_recorded(&f.cmd, b1));
   assert(sparse_site_recorded(&f.cmd, b2));
   assert((uint8_t *)b2 > (uint8_t *)after_l1);

   /* Late sparse creation upgrades both the narrowed 0x160 record and the
    * trailing 0x1ff record to the full word at submit. */
   struct vk_command_buffer *cmds[] = { &f.cmd.vk };
   hk_submit_patch_sparse_sites(&f.dev, cmds, 1);
   assert(*b1 == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   assert(*b2 == barrier_record(AGX_CDM_BARRIER_FULL_WORD));

   fixture_fini(&f);
}

/* Even if the second launch triggers a chunk switch, the preceding launch's
 * trailing barrier in the old chunk is still narrowed to 0x160. */
static void
test_pure_ssbo_narrows_before_chunk_switch(void)
{
   struct fixture f;
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);

   struct agx_shader_info pure = make_compute_shader_info(false);
   uint32_t *b1 = emit_launch(&f, &pure);
   assert(*b1 == barrier_record(0x1ffu));

   VkMemoryBarrier2 mb;
   VkDependencyInfo dep = make_ssbo_dep(&mb);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep));

   force_next_chunk_switch = true;
   uint32_t *b2 = emit_launch(&f, &pure);
   assert(f.cs.stream_linked);
   assert(*b1 == barrier_record(AGX_CDM_BARRIER_SSBO_WORD));
   assert(*b2 == barrier_record(0x1ffu));

   fixture_fini(&f);
}

/* Without a pure SSBO barrier, or with broad/image/merged barriers, the
 * barrier between launches stays at the conservative 0x1ff default word. */
static void
test_conservative_cases_keep_default_word(void)
{
   struct fixture f;
   struct agx_shader_info pure = make_compute_shader_info(false);
   struct agx_shader_info img = make_compute_shader_info(true);
   VkMemoryBarrier2 mb_ssbo, mb_broad;
   VkDependencyInfo dep_ssbo = make_ssbo_dep(&mb_ssbo);
   VkDependencyInfo dep_broad = make_broad_dep(&mb_broad);

   /* 1. No barrier between two launches. */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   uint32_t *b1 = emit_launch(&f, &pure);
   emit_launch(&f, &pure);
   assert(*b1 == barrier_record(0x1ffu));
   fixture_fini(&f);

   /* 2. Broad ALL_COMMANDS barrier between two launches. */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   b1 = emit_launch(&f, &pure);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_broad));
   assert(f.cmd.state.cs.cdm_barrier_full_needed);
   emit_launch(&f, &pure);
   assert(*b1 == barrier_record(0x1ffu));
   fixture_fini(&f);

   /* 3. Pure SSBO barrier merged with a broader barrier before launch 2. */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   b1 = emit_launch(&f, &pure);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_ssbo));
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_broad));
   emit_launch(&f, &pure);
   assert(*b1 == barrier_record(0x1ffu));
   fixture_fini(&f);

   /* 4. Image producer -> pure SSBO barrier -> pure consumer. */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   b1 = emit_launch(&f, &img);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_ssbo));
   emit_launch(&f, &pure);
   assert(*b1 == barrier_record(0x1ffu));
   fixture_fini(&f);

   /* 5. Pure producer -> pure SSBO barrier -> image consumer. */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   b1 = emit_launch(&f, &pure);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_ssbo));
   emit_launch(&f, &img);
   assert(*b1 == barrier_record(0x1ffu));
   fixture_fini(&f);

   /* 6. Pure producer -> pure SSBO barrier -> stream close (no launch 2). */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   b1 = emit_launch(&f, &pure);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_ssbo));
   hk_cdm_end_stream_flush(&f.cmd, &f.cs);
   assert(*b1 == barrier_record(0x1ffu));
   assert(f.cs.stats.flushes == 1);
   assert(!f.cmd.state.cs.cdm_barrier_pending);
   assert(!f.cmd.state.cs.cdm_stream_has_images);
   fixture_fini(&f);
}

/* Full-word parts, sparse devices, and fullcdmbarrier never narrow. */
static void
test_full_word_and_overrides(void)
{
   struct fixture f;
   struct agx_shader_info pure = make_compute_shader_info(false);
   VkMemoryBarrier2 mb;
   VkDependencyInfo dep = make_ssbo_dep(&mb);

   /* G14G keeps full 0xfffff default word; never narrows to 0x160. */
   fixture_init(&f, AGX_CHIP_G14G, 14, 0);
   uint32_t *b1 = emit_launch(&f, &pure);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep));
   emit_launch(&f, &pure);
   assert(*b1 == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   fixture_fini(&f);

   /* Sparse resource created before launch 2 inhibits narrowing. */
   fixture_init(&f, AGX_CHIP_G14X, 14, 0);
   b1 = emit_launch(&f, &pure);
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep));
   p_atomic_set(&f.dev.sparse_used, 1);
   uint32_t *b2 = emit_launch(&f, &pure);
   assert(*b1 == barrier_record(0x1ffu));
   assert(*b2 == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   fixture_fini(&f);

   /* fullcdmbarrier ends the stream at the barrier and emits 0xfffff. */
   fixture_init(&f, AGX_CHIP_G13X, 13, HK_PERF_FULLCDMBARRIER);
   b1 = emit_launch(&f, &pure);
   assert(hk_cdm_record_pipeline_barrier(&f.cmd, &dep));
   assert(!f.cmd.state.cs.cdm_barrier_pending);
   assert(*b1 == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   fixture_fini(&f);
}

/* Non-AGX2 command streams (G18G/HAL300) and open graphics batches must end
 * the compute stream at pipeline barriers instead of keeping it open. */
static void
test_stream_open_limits(void)
{
   struct fixture f;
   VkMemoryBarrier2 mb;
   VkDependencyInfo dep = make_ssbo_dep(&mb);

   /* G18G uses HAL300 command streams outside agx_chip_native_or_m5(). */
   fixture_init(&f, AGX_CHIP_G18G, 18, 0);
   f.cs.stats.cmds = 1;
   assert(hk_cdm_record_pipeline_barrier(&f.cmd, &dep));
   assert(!f.cmd.state.cs.cdm_barrier_pending);
   fixture_fini(&f);

   /* Open graphics batch on AGX2 must also end the compute stream. */
   fixture_init(&f, AGX_CHIP_G13X, 13, 0);
   struct hk_cs gfx_cs = {0};
   f.cs.stats.cmds = 1;
   f.cmd.current_cs.gfx = &gfx_cs;
   assert(hk_cdm_record_pipeline_barrier(&f.cmd, &dep));
   assert(f.cmd.state.cs.cdm_barrier_full_needed);
   fixture_fini(&f);
}

static void
test_defer_flush_g13(enum agx_chip chip)
{
   struct fixture f;
   struct agx_shader_info pure = make_compute_shader_info(false);
   VkMemoryBarrier2 mb_ssbo, mb_broad;
   VkDependencyInfo dep_ssbo = make_ssbo_dep(&mb_ssbo);
   VkDependencyInfo dep_broad = make_broad_dep(&mb_broad);

   /* 1. Two independent launches: zero barriers between them, 0x1ff at close. */
   fixture_init(&f, chip, 13, 0);
   emit_launch(&f, &pure);
   assert(f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 0);
   size_t len1 = (uint8_t *)f.cs.current - f.stream;
   emit_launch(&f, &pure);
   assert(f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 0);
   assert((size_t)((uint8_t *)f.cs.current - f.stream) == 2 * len1);
   hk_cdm_end_stream_flush(&f.cmd, &f.cs);
   assert(!f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 1);
   uint32_t *end_w = (uint32_t *)f.cs.current - 1;
   assert(*end_w == barrier_record(0x1ffu));
   assert(sparse_site_recorded(&f.cmd, end_w));
   fixture_fini(&f);

   /* 2. Pure SSBO barrier between two launches emits 0x160; stream close emits
    * 0x1ff; late sparse patching upgrades both to 0xfffff. */
   fixture_init(&f, chip, 13, 0);
   emit_launch(&f, &pure);
   void *after_l1 = f.cs.current;
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_ssbo));
   emit_launch(&f, &pure);
   uint32_t *mid_w = (uint32_t *)after_l1;
   assert(*mid_w == barrier_record(0x160u));
   assert(sparse_site_recorded(&f.cmd, mid_w));
   assert(f.cs.cdm_flush_pending);
   hk_cdm_end_stream_flush(&f.cmd, &f.cs);
   assert(!f.cs.cdm_flush_pending);
   uint32_t *tail_w = (uint32_t *)f.cs.current - 1;
   assert(*tail_w == barrier_record(0x1ffu));
   assert(sparse_site_recorded(&f.cmd, tail_w));
   p_atomic_set(&f.dev.sparse_used, 1);
   struct vk_command_buffer *cmds[] = {&f.cmd.vk};
   hk_submit_patch_sparse_sites(&f.dev, cmds, 1);
   assert(*mid_w == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   assert(*tail_w == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   fixture_fini(&f);

   /* 3. Broad barrier between two launches emits 0x1ff. */
   fixture_init(&f, chip, 13, 0);
   emit_launch(&f, &pure);
   after_l1 = f.cs.current;
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_broad));
   emit_launch(&f, &pure);
   mid_w = (uint32_t *)after_l1;
   assert(*mid_w == barrier_record(0x1ffu));
   assert(f.cs.cdm_flush_pending);
   fixture_fini(&f);

   /* 4. Deferred launch followed by in_meta launch flushes 0x1ff before meta
    * and meta emits trailing 0x1ff. */
   fixture_init(&f, chip, 13, 0);
   emit_launch(&f, &pure);
   after_l1 = f.cs.current;
   f.cmd.in_meta = true;
   uint32_t *meta_tail = emit_launch(&f, &pure);
   f.cmd.in_meta = false;
   mid_w = (uint32_t *)after_l1;
   assert(*mid_w == barrier_record(0x1ffu));
   assert(*meta_tail == barrier_record(0x1ffu));
   assert(!f.cs.cdm_flush_pending);
   fixture_fini(&f);

   /* 5. Internal precomp launches (e.g. query reset -> query copy on the same
    * stream) never defer: each emits its own trailing 0x1ff flush. */
   fixture_init(&f, chip, 13, 0);
   uint32_t *p1_tail = emit_precomp_launch(&f);
   assert(*p1_tail == barrier_record(0x1ffu));
   assert(!f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 1);
   uint32_t *p2_tail = emit_precomp_launch(&f);
   assert(*p2_tail == barrier_record(0x1ffu));
   assert(!f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 2);
   fixture_fini(&f);

   /* 6. Deferred app dispatch followed by precomp drains 0x1ff before precomp
    * and emits trailing 0x1ff after precomp. */
   fixture_init(&f, chip, 13, 0);
   emit_launch(&f, &pure);
   assert(f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 0);
   after_l1 = f.cs.current;
   p1_tail = emit_precomp_launch(&f);
   mid_w = (uint32_t *)after_l1;
   assert(*mid_w == barrier_record(0x1ffu));
   assert(*p1_tail == barrier_record(0x1ffu));
   assert(!f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 2);
   fixture_fini(&f);

   /* 7. Deferred app dispatch -> pure SSBO barrier -> precomp must upgrade the
    * pending barrier to full 0x1ff (not 0x160) and flush after precomp. */
   fixture_init(&f, chip, 13, 0);
   emit_launch(&f, &pure);
   after_l1 = f.cs.current;
   assert(!hk_cdm_record_pipeline_barrier(&f.cmd, &dep_ssbo));
   p1_tail = emit_precomp_launch(&f);
   mid_w = (uint32_t *)after_l1;
   assert(*mid_w == barrier_record(0x1ffu));
   assert(*p1_tail == barrier_record(0x1ffu));
   assert(!f.cs.cdm_flush_pending);
   assert(f.cs.stats.flushes == 2);
   fixture_fini(&f);

   /* 8. fullcdmbarrier disables deferral and emits 0xfffff per launch. */
   fixture_init(&f, chip, 13, HK_PERF_FULLCDMBARRIER);
   uint32_t *b1 = emit_launch(&f, &pure);
   assert(*b1 == barrier_record(AGX_CDM_BARRIER_FULL_WORD));
   assert(!f.cs.cdm_flush_pending);
   fixture_fini(&f);
}

int
main(void)
{
   test_pure_ssbo_narrows_between_launches(AGX_CHIP_G14X, 14);
   test_pure_ssbo_narrows_before_chunk_switch();
   test_conservative_cases_keep_default_word();
   test_full_word_and_overrides();
   test_stream_open_limits();
   test_defer_flush_g13(AGX_CHIP_G13G);
   test_defer_flush_g13(AGX_CHIP_G13X);
   printf("hk_cdm_barrier_stream: all checks passed\n");
   return 0;
}
