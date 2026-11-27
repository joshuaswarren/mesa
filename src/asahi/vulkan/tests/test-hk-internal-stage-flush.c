/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_shader.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* hk_dispatch_with_local_size() with the launches and the barrier replaced by
 * a recorder: each call appends one letter, so a test reads back the order of
 * what the helper emitted. M = M5 native launch, P = G17P native launch,
 * G = G18G native launch, U = legacy USC launch, R = barrier space reserved,
 * F = cache flush. */
static char events[64];

static void
note(char c)
{
   size_t n = strlen(events);
   assert(n + 1 < sizeof(events));
   events[n] = c;
}

void
hk_dispatch_m5_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                      struct hk_shader *s, struct agx_grid grid,
                      struct agx_workgroup local_size)
{
   (void)cmd, (void)cs, (void)s, (void)grid, (void)local_size;
   note('M');
}

void
hk_dispatch_g17g_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s, struct agx_grid grid,
                        struct agx_workgroup local_size)
{
   (void)cmd, (void)cs, (void)s, (void)grid, (void)local_size;
   note('G');
}

void
hk_dispatch_g17p_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s, struct agx_grid grid,
                        struct agx_workgroup local_size)
{
   (void)cmd, (void)cs, (void)s, (void)grid, (void)local_size;
   note('P');
}

void
hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                     struct agx_shader_info *info, uint32_t usc,
                     struct agx_grid grid, struct agx_workgroup local_size)
{
   (void)dev, (void)cs, (void)info, (void)usc, (void)grid, (void)local_size;
   note('U');
}

uint64_t
hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                    struct hk_linked_shader *linked)
{
   (void)cmd, (void)s, (void)linked;
   return 0x1000;
}

void
hk_reserve_scratch_slow(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s)
{
   (void)cmd, (void)cs, (void)s;
}

void
hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs, size_t size)
{
   assert(cmd == cs->cmd);
   assert(size == AGX3_CDM_BARRIER_SIZE);
   note('R');
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   assert((struct hk_device *)cs->cmd->vk.base.device == dev);
   note('F');
}

/* Only the legacy path's oversized-upload refusal reports an error. */
VkResult
__vk_errorf(const void *obj, VkResult error, const char *file, int line,
            const char *format, ...)
{
   (void)obj, (void)file, (void)line, (void)format;
   abort();
   return error;
}

static const char *
dispatch(struct hk_cmd_buffer *cmd, enum agx_chip chip, mesa_shader_stage stage,
         unsigned instrs)
{
   struct hk_device *dev = (struct hk_device *)cmd->vk.base.device;
   dev->dev.chip = chip;

   struct hk_shader *s = calloc(1, sizeof(*s));
   assert(s);
   s->b.info.stats.instrs = instrs;
   s->info.stage = stage;

   struct hk_cs cs = {.cmd = cmd, .type = HK_CS_CDM};
   memset(events, 0, sizeof(events));
   hk_dispatch_with_local_size(cmd, &cs, s, agx_1d(64),
                               agx_workgroup(2, 1, 1));
   free(s);
   return events;
}

static void
expect(struct hk_cmd_buffer *cmd, enum agx_chip chip, mesa_shader_stage stage,
       unsigned instrs, const char *want)
{
   const char *got = dispatch(cmd, chip, stage, instrs);
   if (strcmp(got, want)) {
      fprintf(stderr, "chip %u stage %u: emitted \"%s\", expected \"%s\"\n",
              chip, stage, got, want);
      abort();
   }
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   assert(dev && cmd);
   cmd->vk.base.device = (struct vk_device *)dev;

   static const mesa_shader_stage internal[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL,
      MESA_SHADER_GEOMETRY,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(internal); ++i) {
      /* Software graphics stages hand their outputs to the next launch
       * through memory: every M5 part and G17P flush after the launch. */
      expect(cmd, AGX_CHIP_G17G, internal[i], 8, "MRF");
      expect(cmd, AGX_CHIP_G17S, internal[i], 8, "MRF");
      expect(cmd, AGX_CHIP_G17C, internal[i], 8, "MRF");
      expect(cmd, AGX_CHIP_G17D, internal[i], 8, "MRF");
      expect(cmd, AGX_CHIP_G17P, internal[i], 8, "PRF");

      /* G18G is unchanged: the launch alone. */
      expect(cmd, AGX_CHIP_G18G, internal[i], 8, "G");

      /* Legacy launches flush inside hk_dispatch_with_usc(). */
      expect(cmd, AGX_CHIP_G13G, internal[i], 8, "U");

      /* An empty shader launches nothing and needs no flush. */
      expect(cmd, AGX_CHIP_G17D, internal[i], 1, "");
   }

   /* Application compute keeps its own barrier policy. */
   expect(cmd, AGX_CHIP_G17D, MESA_SHADER_COMPUTE, 8, "M");
   expect(cmd, AGX_CHIP_G17S, MESA_SHADER_COMPUTE, 8, "M");
   expect(cmd, AGX_CHIP_G17P, MESA_SHADER_COMPUTE, 8, "P");
   expect(cmd, AGX_CHIP_G13G, MESA_SHADER_COMPUTE, 8, "U");

   free(cmd);
   free(dev);
   printf("hk_internal_stage_flush: pass\n");
   return 0;
}
