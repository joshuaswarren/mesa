#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Execute extracted robust-index kernels and abort consumers on the CPU.

Usage: python3 robust-index-exhaustion.py --output /path/to/evidence
Requires clang and pthreads. No Vulkan device is opened. Subgroup lanes execute
serially, but independent workgroups compete on real C11 atomic reservations.
This tests source semantics, not OpenCL lowering, GPU ordering or coherency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

# The exact prior implementation is the parent of the commit that made
# shadow exhaustion report device loss; --base overrides it.
BASE = None
BASE_SUBJECT = "hk: report robust index shadow exhaustion as device loss"
ROOT = Path(__file__).resolve().parents[4]


def base_revision():
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H",
         "--fixed-strings", f"--grep={BASE_SUBJECT}"]
    ).decode().strip()
    if not commit:
        raise SystemExit(f"cannot find {BASE_SUBJECT!r}; pass --base")
    return commit + "^"


def block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    pos, depth = opening + 1, 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]


def source(path, revision=None):
    if revision:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{revision}:{path}"]
        ).decode()
    return (ROOT / path).read_text()


PREFIX = r'''
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#define global
#define constant const
#define GLOBAL
#define CONST const
#define KERNEL(x) void
#define align(x,a) (((x)+(a)-1)&~((a)-1))
typedef unsigned uint; typedef unsigned short ushort; typedef unsigned char uchar;
enum agx_index_size { AGX_INDEX_SIZE_U8, AGX_INDEX_SIZE_U16, AGX_INDEX_SIZE_U32 };
enum agx_primitive { AGX_PRIMITIVE_POINTS, AGX_PRIMITIVE_LINES,
 AGX_PRIMITIVE_LINE_STRIP, AGX_PRIMITIVE_LINE_LOOP, AGX_PRIMITIVE_TRIANGLES,
 AGX_PRIMITIVE_TRIANGLE_STRIP, AGX_PRIMITIVE_TRIANGLE_FAN };
enum agx_chip { AGX_CHIP_G13G, AGX_CHIP_G14G, AGX_CHIP_G17P };
#define AGX_ZERO_PAGE_ADDRESS 0x1000ull
struct agx_grid { uint count[3]; };
struct agx_draw { struct agx_grid b; uint64_t index_buffer;
 uint64_t index_buffer_range_B; uint start,index_bias,start_instance;
 bool restart; enum agx_index_size index_size; bool indexed; };
struct poly_heap { uchar *base; uint bottom,size; };
typedef struct {uint indexCount,instanceCount,firstIndex; int32_t vertexOffset;
 uint firstInstance;} VkDrawIndexedIndirectCommand;
static _Thread_local uint lane, broadcast_value, group, emissions;
static _Thread_local struct agx_draw captured;
static uint get_sub_group_local_id(void) {return lane;}
static uint get_group_id(uint dim) {assert(dim==0); return group;}
static uint sub_group_broadcast(uint x,uint from) {
 assert(from==0); if(lane==0) broadcast_value=x; return broadcast_value;
}
static void agx_vdm_draw_robust_index(uint32_t *vdm,enum agx_chip chip,
 struct agx_draw draw,enum agx_primitive topology,uint32_t restart_index,uint state,
 bool emitted,uint64_t jump) {
 (void)vdm;(void)chip;(void)topology; assert(restart_index==0xffff);
 assert(state==12 && emitted && jump==64);
 captured=draw; emissions++;
}
static struct agx_grid agx_3d(uint x,uint y,uint z) {return (struct agx_grid){{x,y,z}};}
'''

OLD_STUB = (
    "enum agx_primitive topology,uint32_t restart_index,uint state,\n"
    " bool emitted,uint64_t jump) {\n"
    " (void)vdm;(void)chip;(void)topology; assert(restart_index==0xffff);\n",
    "enum agx_primitive topology,uint state,bool emitted,uint64_t jump) {\n"
    " (void)vdm;(void)chip;(void)topology;\n",
)

TESTS = r'''
static uint status_words[4];
static void repair_addr(struct poly_heap *h, VkDrawIndexedIndirectCommand *cmd,
 uintptr_t in_addr,uint64_t range,uint width,uint topology,uint chip,bool batch) {
 struct libagx_robust_index_draw jobs[2] = {{0}, {.vdm=0,
  .command=(uintptr_t)cmd,.empty_draw_jump=64,.draw_state_offset=12,.render_state_emitted=1}};
 emissions=0; group=1;
 for(lane=0;lane<32;lane++) {
  if(batch) libagx_draw_robust_index_batch(jobs,h,FAILURE_ARG in_addr,range,true,0xffff,chip,topology,width);
  else libagx_draw_robust_index(NULL,h,FAILURE_ARG cmd,in_addr,range,true,0xffff,chip,topology,12,1,64,width);
 }
 assert(emissions==1);
}
static void repair(struct poly_heap *h, VkDrawIndexedIndirectCommand *cmd,
 uchar *in,uint64_t range,uint width,uint topology,uint chip,bool batch) {
 repair_addr(h,cmd,(uintptr_t)in,range,width,topology,chip,batch);
}
static void reset_status(void) {status_words[0]=8;status_words[1]=0;
 status_words[2]=0xaabbccdd;status_words[3]=0x11223344;}
static void check_status_canaries(void) {assert(status_words[0]==8);
 assert(status_words[2]==0xaabbccdd && status_words[3]==0x11223344);}
static void matrix(bool old_or_mutation) {
 uint cases=0;
 for(uint batch=0;batch<2;batch++) for(uint chip=0;chip<3;chip++)
 for(uint size=0;size<3;size++) for(uint topology=0;topology<7;topology++) {
  uint w=1u<<size; uchar in[1024],storage[2048];
  for(uint i=0;i<sizeof(in);i++) in[i]=(i*37+11)&255;
  memset(storage,0xcd,sizeof(storage));reset_status();
  struct poly_heap h={storage,16,sizeof(storage)};
  VkDrawIndexedIndirectCommand cmd={64,3,1,-17,11};
  repair(&h,&cmd,in,2*w,size,topology,chip,batch);
  uint expected=topology==AGX_PRIMITIVE_POINTS?64:33;
  assert(captured.b.count[0]==expected && captured.b.count[1]==3);
  assert(captured.index_bias==(uint)-17 && captured.start_instance==11);
  assert(captured.restart && captured.start==0 && captured.index_buffer_range_B==expected*w);
  assert(h.bottom==16+align(expected*w,16) && !status_words[1]);
  for(uint i=0;i<expected*w;i++) assert(storage[16+i]==(i<w?in[w+i]:0));
  assert(storage[15]==0xcd && storage[16+expected*w]==0xcd);
  check_status_canaries();

  // Exact reviewed pressure case, independently requiring output OR failure.
  memset(storage,0xcd,sizeof(storage));reset_status();h.bottom=16;h.size=32;
  repair(&h,&cmd,in,2*w,size,topology,chip,batch);
  printf("pressure batch=%u chip=%u width=%u topology=%u count=%u abort=%u bottom=%u\n",
    batch,chip,w,topology,captured.b.count[0],status_words[1],h.bottom);fflush(stdout);
  assert(captured.b.count[0]==expected || status_words[1]!=0);
  assert(h.bottom==16 && status_words[1]==1);
  assert(captured.b.count[0]==0 && captured.index_buffer==AGX_ZERO_PAGE_ADDRESS);
  for(uint i=0;i<sizeof(storage);i++) assert(storage[i]==0xcd);
  check_status_canaries();

  // Success cannot clear the device-wide execution failure.
  h.size=sizeof(storage);
  repair(&h,&cmd,in,2*w,size,topology,chip,batch);assert(status_words[1]==1);

  // In bounds, count=0, instances=0, fully OOB: no shadow, even with no heap.
  for(uint mode=0;mode<5;mode++) {
   reset_status();h.bottom=h.size=0;
   cmd=(VkDrawIndexedIndirectCommand){64,3,1,-17,11};uint range=2*w;
   if(mode==0) range=1024;
   if(mode==1) cmd.indexCount=0;
   if(mode==2) {cmd.instanceCount=0;cmd.indexCount=UINT32_MAX;}
   if(mode==3) cmd.firstIndex=UINT32_MAX;
   if(mode==4) range=0;
   repair(&h,&cmd,in,range,size,topology,chip,batch);
   assert(!status_words[1] && h.bottom==0);
   assert(captured.b.count[0]==cmd.indexCount && captured.b.count[1]==cmd.instanceCount);
   if(mode>=3) assert(captured.index_buffer==AGX_ZERO_PAGE_ADDRESS && captured.start==0);
   check_status_canaries();
  }
  // Reservation boundary: exact fit succeeds; one byte short fails unchanged.
  cmd=(VkDrawIndexedIndirectCommand){64,1,1,0,0};
  uint span=align(expected*w,16);
  for(uint short_by=0;short_by<2;short_by++) {
   reset_status();h.bottom=16;h.size=16+span-short_by;
   repair(&h,&cmd,in,2*w,size,topology,chip,batch);
   assert(h.bottom==(short_by?16:16+span));assert(status_words[1]==short_by);
  }
  // UINT32_MAX-size point shadows cannot be represented/aligned, for all widths.
  reset_status();h.bottom=16;h.size=UINT32_MAX;cmd.indexCount=UINT32_MAX;
  repair(&h,&cmd,in,2*w,size,AGX_PRIMITIVE_POINTS,chip,batch);
  assert(status_words[1]==1 && h.bottom==16 && !captured.b.count[0]);
  reset_status();h.bottom=UINT32_MAX-15;cmd.indexCount=64;
  repair(&h,&cmd,in,2*w,size,AGX_PRIMITIVE_POINTS,chip,batch);
  assert(status_words[1]==1 && h.bottom==UINT32_MAX-15);
  // 4 GiB and 4 GiB + 4 bindings must keep full 64-bit ranges without
  // wrapping to 0 (zero-page fast path) or 4 (truncated bounds).
  reset_status();h.bottom=h.size=0;
  cmd=(VkDrawIndexedIndirectCommand){64,3,1,-17,11};
  repair(&h,&cmd,in,UINT64_C(0x100000000),size,topology,chip,batch);
  assert(!status_words[1] && h.bottom==0);
  assert(captured.index_buffer==(uintptr_t)in && captured.index_buffer_range_B==UINT64_C(0x100000000));
  assert(captured.start==1 && captured.b.count[0]==64);
  reset_status();h.bottom=h.size=0;
  repair(&h,&cmd,in,UINT64_C(0x100000004),size,topology,chip,batch);
  assert(!status_words[1] && h.bottom==0);
  assert(captured.index_buffer==(uintptr_t)in && captured.index_buffer_range_B==UINT64_C(0x100000004));
  assert(captured.start==1 && captured.b.count[0]==64);
  // 4 GiB + 4 with U32 firstIndex=0x40000000 leaves 1 index available; requesting 2 shadows safely.
  // Form the synthetic base in uintptr_t arithmetic so no out-of-bounds pointer subtraction occurs.
  if(size==AGX_INDEX_SIZE_U32) {
   reset_status();h.bottom=16;h.size=sizeof(storage);
   cmd=(VkDrawIndexedIndirectCommand){2,1,0x40000000u,0,0};
   uintptr_t synth_base=(uintptr_t)in-(uintptr_t)(UINT64_C(1)<<32);
   repair_addr(&h,&cmd,synth_base,UINT64_C(0x100000004),size,topology,chip,batch);
   assert(!status_words[1] && captured.start==0 && captured.index_buffer!=synth_base);
   assert(captured.index_buffer_range_B==2*w);
   for(uint i=0;i<w;i++) assert(storage[16+i]==in[i]);
   for(uint i=w;i<2*w;i++) assert(storage[16+i]==0);
  }
  cases++;
 }
 printf("PASS %u matrix cases (each includes pressure, success, sticky, 5 noalloc, boundaries and overflow)\n",cases);
 (void)old_or_mutation;
}
#define THREADS 32
static uint ready,go;
static pthread_mutex_t start_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t start_cond=PTHREAD_COND_INITIALIZER;
struct worker {struct poly_heap *heap; uint size,chip,batch; uint64_t result; bool success;};
static void *compete(void *arg) {
 struct worker *p=arg;uchar in[16];memset(in,0x47,sizeof(in));
 VkDrawIndexedIndirectCommand cmd={64,1,1,0,0};
 assert(!pthread_mutex_lock(&start_lock));ready++;
 assert(!pthread_cond_broadcast(&start_cond));
 while(!go) assert(!pthread_cond_wait(&start_cond,&start_lock));
 assert(!pthread_mutex_unlock(&start_lock));
 repair(p->heap,&cmd,in,2*(1u<<p->size),p->size,AGX_PRIMITIVE_POINTS,p->chip,p->batch);
 p->result=captured.index_buffer;p->success=captured.b.count[0]==64;
 return NULL;
}
static void parallel(void) {
 for(uint batch=0;batch<2;batch++) for(uint chip=0;chip<3;chip++) for(uint size=0;size<3;size++) {
  uchar storage[4096];memset(storage,0xcd,sizeof(storage));uint span=64*(1u<<size);
  struct poly_heap h={storage,16,16+7*span};reset_status();
  ready=go=0;
  struct worker workers[THREADS];pthread_t threads[THREADS];
  for(uint i=0;i<THREADS;i++) {
   workers[i]=(struct worker){.heap=&h,.size=size,.chip=chip,.batch=batch};
   assert(!pthread_create(&threads[i],NULL,compete,&workers[i]));
  }
  assert(!pthread_mutex_lock(&start_lock));
  while(ready!=THREADS) assert(!pthread_cond_wait(&start_cond,&start_lock));
  go=1;assert(!pthread_cond_broadcast(&start_cond));
  assert(!pthread_mutex_unlock(&start_lock));
  uint winners=0;
  for(uint i=0;i<THREADS;i++) assert(!pthread_join(threads[i],NULL));
  for(uint i=0;i<THREADS;i++) if(workers[i].success) {
   winners++;uint off=workers[i].result-(uintptr_t)storage;
   assert(off>=16 && off+span<=h.size && (off-16)%span==0);
   for(uint j=0;j<i;j++) if(workers[j].success) assert(workers[j].result!=workers[i].result);
   for(uint b=0;b<span;b++) assert(storage[off+b]==(b<(1u<<size)?0x47:0));
  } else assert(workers[i].result==AGX_ZERO_PAGE_ADDRESS);
  assert(winners==7 && h.bottom==h.size && status_words[1]==1);
  assert(storage[15]==0xcd && storage[h.size]==0xcd);check_status_canaries();
 }
 puts("PASS 18 concurrent reservation cases, 32 workgroups each, 7 disjoint winners");
}
int main(int argc,char **argv) {
 if(argc==5) {
  uint batch=atoi(argv[2]),chip=atoi(argv[3]),size=atoi(argv[4]),w=1u<<size;
  uchar in[16]={0},storage[128];memset(storage,0xcd,sizeof(storage));reset_status();
  struct poly_heap h={storage,16,32};VkDrawIndexedIndirectCommand cmd={64,1,1,0,0};
  repair(&h,&cmd,in,2*w,size,AGX_PRIMITIVE_POINTS,chip,batch);
  printf("exact pressure batch=%u chip=%u width=%u requested=64 output=%u abort=%u bottom=%u\n",
   batch,chip,w,captured.b.count[0],status_words[1],h.bottom);fflush(stdout);
  assert(captured.b.count[0]==64 || status_words[1]!=0);return 0;
 }
 matrix(false);parallel();return 0;
}
'''


def kernels(revision=None, mutation=None):
    h = source("src/asahi/libagx/libagx_dgc.h", revision)
    cl = source("src/asahi/libagx/draws.cl", revision)
    poly = source("src/poly/poly_heap_math.h", revision)
    code = PREFIX + "\n" + block(h, "struct libagx_robust_index_draw {") + ";\n"
    for marker in ["static inline unsigned\nagx_indices_to_B(",
                   "static inline struct agx_draw\nagx_draw_indexed("]:
        code += block(h, marker) + "\n"
    code += block(poly, "static inline bool\npoly_heap_allocation_fits(") + "\n"
    for name, typ in [("available", "uint32_t"), ("copy", "void"),
                      ("clamped_count", "uint32_t"), ("shadow_sizes", "bool")]:
        code += block(h, f"static inline {typ}\nagx_robust_index_{name}(") + "\n"
    for marker in ["static void\nlibagx_draw_robust_index_impl(",
                   "KERNEL(32)\nlibagx_draw_robust_index(",
                   "KERNEL(32)\nlibagx_draw_robust_index_batch("]:
        code += block(cl, marker) + "\n"
    if mutation == "no-abort":
        code = code.replace("atomic_store((volatile global atomic_uint *)(printf_buffer + 1), 1);", "(void)printf_buffer;")
    if mutation == "no-empty-guard":
        code = code.replace("cmd->instanceCount != 0 && ", "")
    if mutation == "drop-points":
        code = code.replace("if (topology == AGX_PRIMITIVE_POINTS)", "if (false)")
    code += "#define FAILURE_ARG " + ("status_words," if "printf_buffer" in cl else "") + "\n"
    tests = TESTS
    if "uint32_t restart_index" not in cl:
        # Kernels from before they took the custom primitive restart index.
        code = code.replace(OLD_STUB[0], OLD_STUB[1])
        tests = tests.replace(",true,0xffff,", ",true,")
    return code + tests


API_PREFIX = r'''
#include <stddef.h>
#include <errno.h>
#define VKAPI_ATTR
#define VKAPI_CALL
#define MESA_TRACE_FUNC() ((void)0)
#define unlikely(x) (x)
#define container_of(p,t,m) ((t*)((char*)(p)-offsetof(t,m)))
#define VK_FROM_HANDLE(t,n,h) struct t *n=(struct t*)(h)
#define STACK_ARRAY(t,n,c) t n[c]
#define STACK_ARRAY_FINISH(n) ((void)(n))
#define VK_QUERY_RESULT_WAIT_BIT 1
#define VK_SEMAPHORE_WAIT_ANY_BIT 1
#define VK_SEMAPHORE_TYPE_TIMELINE 1
#define DRM_ASAHI_FEATURE_VM_STATUS 1
#define DRM_ASAHI_PARAM_GROUP_VM_STATUS 1
#define DRM_IOCTL_ASAHI_GET_PARAMS 1
#define HK_QUERY_TIMEOUT 2000000000ull
#define VK_SYNC_FEATURE_CPU_WAIT 1
#define vk_foreach_queue(q,d) for(struct vk_queue *q=(d)->queue;q;q=NULL)
typedef int VkResult;typedef void *VkDevice,*VkFence,*VkSemaphore,*VkQueryPool,*VkQueue;
typedef uint32_t VkBool32,VkQueryResultFlags;typedef uint64_t VkDeviceSize,VkPipelineStageFlags2;
enum {VK_SUCCESS=0,VK_NOT_READY=1,VK_TIMEOUT=2,VK_ERROR_DEVICE_LOST=-4};
enum vk_sync_wait_flags {VK_SYNC_WAIT_COMPLETE=1,VK_SYNC_WAIT_ANY=2};
struct vk_sync {int unused;};struct vk_sync_type {int features;};
struct vk_sync_wait {struct vk_sync *sync;uint64_t stage_mask,wait_value;};
struct vk_device;
struct vk_device_dispatch_table {VkResult (*QueueWaitIdle)(VkQueue);};
struct vk_device {bool lost;FILE *debug_output;VkResult (*check_status)(struct vk_device*);
 void *physical;struct vk_queue *queue;struct vk_device_dispatch_table dispatch_table;};
struct u_printf_ctx {int lock;void *bo;uint32_t *map;};
struct hk_device {struct vk_device vk;struct {int32_t *word;} vm_status_mirror;
 struct {uint vm_id;int fd;struct {uint features;} params;struct u_printf_ctx printf;} dev;};
struct drm_asahi_vm_status {uint vm_id;int error;};
struct drm_asahi_get_params {uint param_group;uintptr_t pointer;uint size;};
struct vk_fence {int unused;};struct vk_semaphore {uint type;};
struct vk_queue {struct {struct vk_device *device;} base;};
struct hk_query_report {uint64_t value;};struct hk_query_pool {struct hk_query_report report;bool available;};
typedef struct {uint32_t flags,semaphoreCount;const VkSemaphore *pSemaphores;
 const uint64_t *pValues;} VkSemaphoreWaitInfo;
static bool vk_device_is_lost(struct vk_device *d) {return d->lost;}
static bool vk_device_is_lost_no_report(struct vk_device *d) {return d->lost;}
static VkResult vk_device_set_lost(struct vk_device *d,const char *fmt,...) {d->lost=true;(void)fmt;return VK_ERROR_DEVICE_LOST;}
static int drmIoctl(int fd,int req,void *arg) {(void)fd;(void)req;(void)arg;assert(false);return -1;}
static void simple_mtx_init(int *lock,int type) {(void)lock;(void)type;}
static void simple_mtx_lock(int *lock) {(void)lock;}
static void simple_mtx_unlock(int *lock) {(void)lock;}
#define mtx_plain 0
static void u_printf(FILE *out,const char *buf,size_t size,void *info,unsigned n) {
 (void)out;(void)buf;(void)size;(void)info;(void)n;assert(false);
}
static uint32_t *completion_abort;
static bool fail_on_complete;
static VkResult transport_result=VK_SUCCESS;
static void complete(void) {if(fail_on_complete) *completion_abort=1;}
static VkResult vk_common_GetFenceStatus(VkDevice d,VkFence f) {(void)d;(void)f;complete();return transport_result;}
static VkResult vk_common_GetSemaphoreCounterValue(VkDevice d,VkSemaphore s,uint64_t *v) {(void)d;(void)s;*v=1;complete();return transport_result;}
static uint64_t os_time_get_absolute_timeout(uint64_t n) {return n;}
static uint64_t os_time_get_nano(void) {static uint64_t time;return ++time;}
static struct vk_sync *vk_fence_get_active_sync(struct vk_fence *f) {(void)f;return NULL;}
static struct vk_sync *vk_semaphore_get_active_sync(struct vk_semaphore *s) {(void)s;return NULL;}
static VkResult vk_sync_wait_many(struct vk_device*d,uint n,struct vk_sync_wait*w,enum vk_sync_wait_flags f,uint64_t t) {
 (void)d;(void)n;(void)w;(void)f;(void)t;complete();return transport_result;
}
static const struct vk_sync_type *get_cpu_wait_type(void*p) {static struct vk_sync_type type={1};(void)p;return &type;}
static VkResult vk_sync_create(struct vk_device*d,const struct vk_sync_type*t,uint f,uint64_t v,struct vk_sync**s) {
 (void)d;(void)t;(void)f;(void)v;*s=NULL;return VK_SUCCESS;
}
static VkResult vk_queue_signal_sync(struct vk_queue*q,struct vk_sync*s,uint64_t v) {(void)q;(void)s;(void)v;return VK_SUCCESS;}
static VkResult vk_sync_wait(struct vk_device*d,struct vk_sync*s,uint64_t v,enum vk_sync_wait_flags f,uint64_t t) {
 (void)d;(void)s;(void)v;(void)f;(void)t;complete();return transport_result;
}
static void vk_sync_destroy(struct vk_device*d,struct vk_sync*s) {(void)d;(void)s;}
static VkQueue vk_queue_to_handle(struct vk_queue*q) {return q;}
static bool hk_query_is_available(struct hk_device*d,struct hk_query_pool*p,uint q) {(void)d;(void)q;complete();return p->available;}
static const struct hk_query_report *hk_query_report_map(struct hk_device*d,struct hk_query_pool*p,uint q) {(void)d;(void)q;return &p->report;}
static uint hk_reports_per_query(struct hk_query_pool*p) {(void)p;return 1;}
static bool hk_query_write_cpu_result(void*dst,const uint64_t*src,uint n,bool avail,VkQueryResultFlags flags) {
 (void)n;(void)flags;*(uint64_t*)dst=*src;return avail;
}
'''
API_TESTS = r'''
int main(void) {
 uint32_t words[4]={8,0,0xaabbccdd,0x11223344};
 struct hk_device d={0};d.vk.debug_output=stderr;d.vk.check_status=hk_check_status;
 u_printf_init(&d.dev.printf,NULL,words);completion_abort=&words[1];
 struct vk_queue queue={.base={.device=&d.vk}};d.vk.queue=&queue;
 d.vk.dispatch_table.QueueWaitIdle=vk_common_QueueWaitIdle;
 struct vk_fence fence={0};VkFence f=&fence;
 struct vk_semaphore semaphore={.type=VK_SEMAPHORE_TYPE_TIMELINE};VkSemaphore s=&semaphore;
 uint64_t counter=0,value=1,result_value=0;
 VkSemaphoreWaitInfo wait={.semaphoreCount=1,.pSemaphores=&s,.pValues=&value};
 struct hk_query_pool pool={.report={123},.available=true};
 for(uint api=0;api<7;api++) for(uint failure=0;failure<2;failure++) {
  words[1]=0;d.vk.lost=false;fail_on_complete=failure;transport_result=VK_SUCCESS;
  VkResult got;
  switch(api) {
  case 0: got=hk_GetFenceStatus(&d,&fence);break;
  case 1: got=hk_GetSemaphoreCounterValue(&d,&semaphore,&counter);break;
  case 2: got=vk_common_WaitForFences(&d,1,&f,1,UINT64_MAX);break;
  case 3: got=vk_common_WaitSemaphores(&d,&wait,UINT64_MAX);break;
  case 4: got=vk_common_QueueWaitIdle(&queue);break;
  case 5: got=vk_common_DeviceWaitIdle(&d);break;
  default: got=hk_GetQueryPoolResults(&d,&pool,0,1,sizeof(result_value),&result_value,8,0);break;
  }
  printf("api=%u failure=%u result=%d sticky=%u\n",api,failure,got,d.vk.lost);fflush(stdout);
  assert(got==(failure?VK_ERROR_DEVICE_LOST:VK_SUCCESS));assert(d.vk.lost==failure);
  if(failure) {fail_on_complete=false;assert(vk_device_check_status(&d.vk)==VK_ERROR_DEVICE_LOST);}
  assert(words[0]==8 && words[2]==0xaabbccdd && words[3]==0x11223344);
 }
 // A not-ready poll preserves VK_NOT_READY if healthy and surfaces abort if set.
 d.vk.lost=false;words[1]=0;transport_result=VK_NOT_READY;fail_on_complete=false;
 assert(hk_GetFenceStatus(&d,&fence)==VK_NOT_READY);
 fail_on_complete=true;assert(hk_GetFenceStatus(&d,&fence)==VK_ERROR_DEVICE_LOST);
 // Both available+WAIT and unavailable query routes must propagate failure.
 for(uint available=0;available<2;available++) {
  d.vk.lost=false;words[1]=0;fail_on_complete=true;pool.available=available;
  assert(hk_GetQueryPoolResults(&d,&pool,0,1,sizeof(result_value),&result_value,8,VK_QUERY_RESULT_WAIT_BIT)==VK_ERROR_DEVICE_LOST);
 }
 d.vk.lost=false;words[1]=0;fail_on_complete=false;pool.available=false;
 assert(hk_GetQueryPoolResults(&d,&pool,0,1,sizeof(result_value),&result_value,8,0)==VK_NOT_READY);
 puts("PASS actual abort/status callbacks and 7 completion APIs, sticky repeat and not-ready polling");
 return 0;
}
'''


def api_code(old_query=False):
    code = PREFIX.split("enum agx_index_size")[0] + API_PREFIX
    util = source("src/util/u_printf.h")
    for name in ["u_printf_init", "u_printf_with_ctx", "u_printf_check_abort"]:
        code += block(util, ("static inline bool\n" if name.endswith("abort") else
                             "static inline void\n") + name + "(") + "\n"
    code += block(source("src/vulkan/runtime/vk_debug_utils.c"),
                  "VkResult\nvk_check_printf_status(") + "\n"
    code += block(source("src/vulkan/runtime/vk_device.h"),
                  "static inline VkResult\nvk_device_check_status(") + "\n"
    device = source("src/asahi/vulkan/hk_device.c")
    code += block(device, "static VkResult\nhk_check_status(") + "\n"
    for name in ["hk_GetFenceStatus", "hk_GetSemaphoreCounterValue"]:
        code += block(device, "VKAPI_ATTR VkResult VKAPI_CALL\n" + name + "(") + "\n"
    for path, name in [("vk_fence.c", "WaitForFences"), ("vk_semaphore.c", "WaitSemaphores"),
                       ("vk_queue.c", "QueueWaitIdle"), ("vk_device.c", "DeviceWaitIdle")]:
        code += block(source("src/vulkan/runtime/" + path),
                      "VKAPI_ATTR VkResult VKAPI_CALL\nvk_common_" + name + "(") + "\n"
    query = source("src/asahi/vulkan/hk_query_pool.c", BASE if old_query else None)
    code += block(query, "static VkResult\nhk_query_wait_for_available(") + "\n"
    code += block(query, "VKAPI_ATTR VkResult VKAPI_CALL\nhk_GetQueryPoolResults(") + "\n"
    return code + API_TESTS


def end_to_end_code(revision=None):
    repair = kernels(revision).replace("int main(", "int repair_probe_main(")
    api = api_code().replace("int main(", "int api_probe_main(")
    return repair + api + r'''
int main(void) {
 struct hk_device d={0};d.vk.debug_output=stderr;d.vk.check_status=hk_check_status;
 u_printf_init(&d.dev.printf,NULL,status_words);
 fail_on_complete=false;transport_result=VK_SUCCESS;completion_abort=&status_words[1];
 for(uint batch=0;batch<2;batch++) for(uint chip=0;chip<3;chip++) for(uint size=0;size<3;size++) {
  uchar in[16]={0},storage[128];memset(storage,0xcd,sizeof(storage));
  struct poly_heap h={storage,16,32};VkDrawIndexedIndirectCommand cmd={64,1,1,0,0};
  reset_status();d.vk.lost=false;
  repair(&h,&cmd,in,2*(1u<<size),size,AGX_PRIMITIVE_POINTS,chip,batch);
  VkResult result=hk_GetFenceStatus(&d,NULL);
  printf("composed batch=%u chip=%u width=%u output=%u abort=%u VkResult=%d\n",
   batch,chip,1u<<size,captured.b.count[0],status_words[1],result);fflush(stdout);
  assert(result==VK_ERROR_DEVICE_LOST && d.vk.lost && h.bottom==16);
 }
 puts("PASS 18 actual repair-to-Vulkan-status composition cases");return 0;
}
'''


def run(output, label, code, expected, args=()):
    cfile, exe = output / (label + ".c"), output / label
    cfile.write_text(code)
    compile_cmd = ["clang", "-std=c11", "-O1", "-g", "-pthread",
                   "-fsanitize=address,undefined", str(cfile), "-o", str(exe)]
    compiled = subprocess.run(compile_cmd, capture_output=True, text=True)
    (output / (label + "-compile.log")).write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(f"{label}: compile failed: {compiled.stderr}")
    result = subprocess.run([str(exe), *args], capture_output=True, text=True)
    (output / (label + ".log")).write_text(result.stdout + result.stderr)
    row = dict(name=label, exit=result.returncode, expected_pass=expected,
               source_sha256=hashlib.sha256(code.encode()).hexdigest())
    assert (result.returncode == 0) == expected, row
    return row



def source_contracts():
    draw = source("src/asahi/vulkan/hk_cmd_draw.c")
    single = "".join(block(draw, "static void\nhk_emit_robust_index(").split())
    batch = "".join(block(draw, "static void\nhk_flush_robust_index_batch(").split())
    assert "target,hk_heap(cmd),dev->dev.printf_buffer_address,draw.b.ptr" in single
    assert "batch->jobs.gpu,hk_heap(cmd),dev->dev.printf_buffer_address,draw->index_buffer" in batch
    assert "AGX_BARRIER_ALL|AGX_PREGFX" in single
    assert "AGX_BARRIER_ALL|AGX_PREGFX" in batch
    # Heap reclamation cannot clear the sticky device abort word.
    assert "printf" not in block(draw, "static uint64_t\nhk_heap(")
    paths = ["src/asahi/libagx/draws.cl", "src/asahi/vulkan/hk_cmd_draw.c",
             "src/asahi/vulkan/hk_device.c", "src/asahi/vulkan/hk_query_pool.c",
             "src/asahi/lib/agx_device.c", "src/compiler/nir/nir_lower_printf.c",
             "src/util/u_printf.h", "src/vulkan/runtime/vk_debug_utils.c",
             "src/vulkan/runtime/vk_device.h", "src/vulkan/runtime/vk_device.c",
             "src/vulkan/runtime/vk_fence.c", "src/vulkan/runtime/vk_semaphore.c",
             "src/vulkan/runtime/vk_queue.c"]
    return {path: hashlib.sha256(source(path).encode()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", help="revision with the prior behavior")
    args = parser.parse_args()
    global BASE
    BASE = args.base or base_revision()
    args.output.mkdir(parents=True, exist_ok=True)
    contracts = source_contracts()
    (args.output / "source-contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")
    rows = []
    for label, rev, mutation, expected in [
        ("candidate", None, None, True), ("exact-base", BASE, None, False),
        ("no-abort-control", None, "no-abort", False),
        ("drop-points-control", None, "drop-points", False),
        ("no-empty-guard-control", None, "no-empty-guard", False),
    ]:
        rows.append(run(args.output, label, kernels(rev, mutation), expected))
    rows.append(run(args.output, "repair-to-api", end_to_end_code(), True))
    rows.append(run(args.output, "exact-base-to-api", end_to_end_code(BASE), False))
    rows.append(run(args.output, "api-contract", api_code(), True))
    rows.append(run(args.output, "old-query-control", api_code(True), False))
    for batch in range(2):
        for chip in range(3):
            for width in range(3):
                label = f"exact-base-pressure-b{batch}-c{chip}-w{1 << width}"
                rows.append(run(args.output, label, kernels(BASE), False,
                                ["pressure", str(batch), str(chip), str(width)]))
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
