/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"
#include "hk_acceleration_structure.h"
#include "asahi/lib/agx_scs_layout.h"
#include "util/u_debug.h"
#include <string.h>

/* The generated runtime writes traversal state with stscs, into the kernel's
 * per-kick SCS allocation. The user image supplies only global-load data;
 * mutable AS tables and launch contexts have their own allocations. Keep a
 * command-buffer reference just as we do for the runtime code and IFT.
 * External diagnostic runtimes retain private, fully initialized storage.
 */
uint64_t
hk_gen1_query_state(struct hk_cmd_buffer *cmd,
                    const struct agx_scs_layout_gen2 *scs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->gen1_rq_state) {
      hk_cmd_buffer_retain_bo(cmd, dev->gen1_rq_state);
      return dev->gen1_rq_state->va->addr;
   }

   const uint64_t size = MAX2(UINT64_C(0x60000),
                             ALIGN_POT((uint64_t)scs->end[12], 0x4000));
   struct agx_ptr state = hk_pool_alloc(cmd, size, 0x4000);
   if (!state.cpu)
      return 0;

   memcpy(state.cpu, dev->gen1_rq_user_image, 0x60000);
   if (size > 0x60000)
      memset((uint8_t *)state.cpu + 0x60000, 0, size - 0x60000);
   if (dev->gen1_rq_instances)
      agx_scs_layout_gen1_parameters(scs, (uint8_t *)state.cpu + 0x4000);
   return state.gpu;
}

/* Upload the 512-byte uniform block a Gen1 native ray query reads, for both
 * compute and graphics launches: the generated runtime's uniforms rebased
 * onto this kick's state image, the AS-table and candidate constants, and the
 * caller's launch context (context_words words, at byte 352). Graphics
 * launches also publish the packed SCS layout through a profile pointer.
 */
bool
hk_gen1_upload_rq_uniforms(struct hk_cmd_buffer *cmd, uint32_t *context,
                           unsigned context_words, bool profile,
                           struct agx3_ioto_uniform_range *range)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint64_t packed = hk_gen1_ray_query_scs_layout(dev);
   struct agx_scs_layout_gen2 scs;
   if (!agx_scs_layout_gen1_init(packed, &scs)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   const uint64_t state = hk_gen1_query_state(cmd, &scs);
   struct agx_ptr u=hk_pool_alloc(cmd,512,16);
   if(!state || !u.cpu)return false;
   memset(u.cpu,0,512);memcpy(u.cpu,dev->gen1_rq_uniforms,320);
   for(unsigned i=0;i<74;i+=2) {
      uint64_t value;memcpy(&value,(uint8_t*)u.cpu+4*i,8);
      if(value>=UINT64_C(0x20d0000000) && value<UINT64_C(0x20d0060000)) {
         value+=state-UINT64_C(0x20d0000000);
         memcpy((uint8_t*)u.cpu+4*i,&value,8);
      }
   }
   uint64_t table=state+0x50000,ctx=u.gpu+352;
   if (dev->gen1_rq_instances) {
      table=hk_gen1_upload_as_table(cmd);
      if (!table) return false;
      const uint32_t constants[]={0x3f7ffffe,0x3f800001,0x6000,0x8000,
         0xfe00001,0x1000,0x100,0x10002,0x1ffff,0x20000,0x400000,
         0x1f000000,0x34a00000,0x40000000,0xfffeff00};
      memcpy((uint8_t *)u.cpu+118*2,constants,sizeof(constants));
      uint32_t *uw=u.cpu;
      uw[106/2]=88;
      uw[48/2]=scs.end[6];uw[50/2]=scs.end[11];
      uw[52/2]=packed;uw[54/2]=packed>>32;
      memcpy((uint8_t *)u.cpu+32*2,&table,8);
      context[2]=packed;context[3]=packed>>32;
      context[20]=scs.end[5];
   }
   if(dev->gen1_rq_candidates) {
      const uint32_t constants[]={0x3fffffff,0x3f7ffffe,0x3f800001,
         0x6000,0x8000,0x10000,0x60400,0x3ffd0fe0,0x914348a1,
         0x100,0x10002,0x1ffff,0x20000,0xfffff,0x400000,
         0x1f000001,0x34a00000,0xfe000000,0xfe1ffffe,0xfffe0000,0xfffeff00};
      memcpy((uint8_t*)u.cpu+110*2,constants,sizeof(constants));
      ((uint32_t*)u.cpu)[104/2]=304;
   }
   if (profile) {
      struct agx_ptr p=hk_pool_alloc(cmd,sizeof(packed),8);
      if (!p.cpu) return false;
      memcpy(p.cpu,&packed,sizeof(packed));
      context[9]=0;
      memcpy(context+10,&p.gpu,sizeof(p.gpu));
      context[14]=context[15]=0;
      context[26]=context[27]=0;
   }
   memcpy(context+12,&table,8);
   memcpy((uint8_t*)u.cpu+352,context,context_words*sizeof(*context));
   memcpy((uint8_t*)u.cpu+320,&ctx,8);memcpy((uint8_t*)u.cpu+328,&table,8);
   *range=(struct agx3_ioto_uniform_range){
      .start_halfs=0,.size_halfs=168,.address=u.gpu,
   };
   hk_cmd_buffer_retain_bo(cmd,dev->gen1_rq_code);
   hk_cmd_buffer_retain_bo(cmd,dev->gen1_rq_ift);
   return true;
}

bool
hk_gen1_upload_fragment_query_state(struct hk_cmd_buffer *cmd, bool vertex_stage,
                                    struct agx3_ioto_uniform_range *range)
{
   struct hk_device *dev=hk_cmd_buffer_device(cmd);
   if (!dev->gen1_native_rq || !dev->gen1_native_fragment_rq ||
       (vertex_stage && !dev->gen1_native_graphics_rq) ||
       cmd->state.gfx.render.tilebuffer.nr_samples > 1) {
      vk_command_buffer_set_error(&cmd->vk,VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   uint32_t context[30]={[17]=0x6f};
   if (!hk_gen1_upload_rq_uniforms(cmd, context, ARRAY_SIZE(context), true,
                                   range))
      return false;
   if (!cmd->current_cs.gfx) return false;
   if (vertex_stage)
      cmd->current_cs.gfx->gen1_native_vertex_rq=true;
   else
      cmd->current_cs.gfx->gen1_native_fragment_rq=true;
   return true;
}
