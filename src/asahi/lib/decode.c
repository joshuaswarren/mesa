/*
 * Copyright 2017-2019 Alyssa Rosenzweig
 * Copyright 2017-2019 Connor Abbott
 * Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <ctype.h>
#include <memory.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "util/os_misc.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include <sys/mman.h>
#include <agx_pack.h>

#include "asahi/isa/disasm.h"
#include "agx3_cdm_decode_core.h"
#include "agx3_cdm_direct_launch_semantic_core.h"
#include "agx_ioto_core.h"
#include "agx_m5_va.h"
#include "agx_ppp_decode_core.h"
#include "agx_va_runtime_hole.h"
#include "agx_vdm_decode_core.h"
#include "util/u_hexdump.h"
#include "decode.h"

struct libagxdecode_config lib_config;

static void
agx_disassemble(void *_code, size_t maxlen, FILE *fp)
{
   bool errors = agx2_disassemble(_code, maxlen, fp);
   assert(!errors);
}

FILE *agxdecode_dump_stream;

struct agxdecode_ctx {
   struct util_dynarray mmap_array;
   uint64_t shader_base;
};

static uint64_t
decode_usc(struct agxdecode_ctx *ctx, uint64_t addr)
{
   return ctx->shader_base + addr;
}

struct agxdecode_ctx *
agxdecode_new_context(uint64_t shader_base)
{
   struct agxdecode_ctx *ctx = calloc(1, sizeof(struct agxdecode_ctx));
   ctx->shader_base = shader_base;
   return ctx;
}

void
agxdecode_destroy_context(struct agxdecode_ctx *ctx)
{
   free(ctx);
}

static struct agx_bo *
agxdecode_find_mapped_gpu_mem_containing(struct agxdecode_ctx *ctx,
                                         uint64_t addr)
{
   util_dynarray_foreach(&ctx->mmap_array, struct agx_bo, it) {
      if (it->va && addr >= it->va->addr && (addr - it->va->addr) < it->size)
         return it;
   }

   return NULL;
}

static struct agx_bo *
agxdecode_find_handle(struct agxdecode_ctx *ctx, unsigned handle)
{
   util_dynarray_foreach(&ctx->mmap_array, struct agx_bo, it) {
      if (it->handle == handle)
         return it;
   }

   return NULL;
}

static size_t
_agxdecode_grab_mapped(struct agxdecode_ctx *ctx, uint64_t gpu_va, void **buf,
                       int line, const char *filename)
{
   if (lib_config.read_gpu_mem)
      UNREACHABLE("you'll have to figure it out.");

   const struct agx_bo *mem =
      agxdecode_find_mapped_gpu_mem_containing(ctx, gpu_va);

   if (!mem) {
      fprintf(stderr, "Access to unknown memory %" PRIx64 " in %s:%d\n", gpu_va,
              filename, line);
      fflush(agxdecode_dump_stream);
      *buf = NULL;
      return 0;
   }

   uint32_t offset = gpu_va - mem->va->addr;

   *buf = mem->_map + offset;
   return mem->size - offset;
}

static size_t
__agxdecode_fetch_gpu_mem(struct agxdecode_ctx *ctx, const struct agx_bo *mem,
                          uint64_t gpu_va, size_t size, void *buf, int line,
                          const char *filename)
{
   if (lib_config.read_gpu_mem)
      return lib_config.read_gpu_mem(gpu_va, size, buf);

   if (!mem)
      mem = agxdecode_find_mapped_gpu_mem_containing(ctx, gpu_va);

   if (!mem) {
      fprintf(stderr, "Access to unknown memory %" PRIx64 " in %s:%d\n", gpu_va,
              filename, line);
      fflush(agxdecode_dump_stream);
      assert(0);
   }

   assert(mem);

   if (size + (gpu_va - mem->va->addr) > mem->size) {
      fprintf(stderr,
              "Overflowing to unknown memory %" PRIx64
              " of size %zu (max size %zu) in %s:%d\n",
              gpu_va, size, (size_t)(mem->size - (gpu_va - mem->va->addr)),
              filename, line);
      fflush(agxdecode_dump_stream);
      assert(0);
   }

   memcpy(buf, mem->_map + gpu_va - mem->va->addr, size);

   return size;
}

#define agxdecode_fetch_gpu_mem(ctx, gpu_va, size, buf)                        \
   __agxdecode_fetch_gpu_mem(ctx, NULL, gpu_va, size, buf, __LINE__, __FILE__)

#define agxdecode_grab_mapped(ctx, gpu_va, buf)                                \
   _agxdecode_grab_mapped(ctx, gpu_va, buf, __LINE__, __FILE__)

#define agxdecode_fetch_gpu_array(ctx, gpu_va, buf)                            \
   agxdecode_fetch_gpu_mem(ctx, gpu_va, sizeof(buf), buf)

/* Helpers for parsing the cmdstream */

#define DUMP_UNPACKED(T, var, str)                                             \
   {                                                                           \
      agxdecode_log(str);                                                      \
      agx_print(agxdecode_dump_stream, T, var, 2);                             \
   }

#define DUMP_CL(T, cl, str)                                                    \
   {                                                                           \
      agx_unpack(agxdecode_dump_stream, cl, T, temp);                          \
      DUMP_UNPACKED(T, temp, str "\n");                                        \
   }

#define DUMP_FIELD(struct, fmt, field)                                         \
   {                                                                           \
      fprintf(agxdecode_dump_stream, #field " = " fmt "\n", struct->field);    \
   }

#define agxdecode_log(str) fputs(str, agxdecode_dump_stream)
#define agxdecode_msg(str) fprintf(agxdecode_dump_stream, "// %s", str)

typedef struct drm_asahi_params_global decoder_params;

/* Keep command-stream selection on the kernel's explicit generation/variant
 * identity.  Cluster count cannot distinguish A18 Pro (G17P/AGX3b) from M5
 * (G17G/AGX3c), and must never be used as an architecture proxy.  This local
 * form avoids making the standalone decoder library depend on libasahi_lib;
 * agx_chip_from_params() is the equivalent device-open mapping. */
enum agx_chip
agxdecode_chip_from_params(const decoder_params *params)
{
   if (!params)
      return AGX_CHIP_UNKNOWN;

   switch (params->gpu_generation) {
   case 13:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G13G;
      case 'S':
      case 'C':
      case 'D': return AGX_CHIP_G13X;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 14:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G14G;
      case 'S':
      case 'C':
      case 'D': return AGX_CHIP_G14X;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 15:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G15P;
      case 'G': return AGX_CHIP_G15G;
      case 'S': return AGX_CHIP_G15S;
      case 'C': return AGX_CHIP_G15C;
      case 'D': return AGX_CHIP_G15D;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 16:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G16P;
      case 'G': return AGX_CHIP_G16G;
      case 'S':
         /* As agx_chip_from_params(): the M4 Pro on the G17 runtime is G17P. */
         if (params->chip_id == 0x6040 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200)
            return AGX_CHIP_G17P;
         return AGX_CHIP_G16S;
      case 'C':
      case 'X':
         /* As agx_chip_from_params(): the M4 Max on the G17 runtime is G17P. */
         if (params->chip_id == 0x6041 &&
             params->usc_generation == AGX_ISA_AGX3 &&
             params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200)
            return AGX_CHIP_G17P;
         return params->gpu_variant == 'C' ? AGX_CHIP_G16C : AGX_CHIP_UNKNOWN;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 17:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G17P;
      case 'A': return AGX_CHIP_G17A;
      case 'G': return AGX_CHIP_G17G;
      case 'S': return AGX_CHIP_G17S;
      case 'C': return AGX_CHIP_G17C;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 18:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G18G;
      default: return AGX_CHIP_UNKNOWN;
      }
   default:
      return AGX_CHIP_UNKNOWN;
   }
}

static bool
agxdecode_chip_uses_agx3(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
   case AGX_CHIP_G18G:
      return true;
   default:
      return false;
   }
}

static const char *
agxdecode_chip_name(enum agx_chip chip)
{
   switch (chip) {
   case AGX_CHIP_G15G: return "G15G";
   case AGX_CHIP_G15S: return "G15S";
   case AGX_CHIP_G15C: return "G15C";
   case AGX_CHIP_G15D: return "G15D";
   case AGX_CHIP_G16P: return "G16P";
   case AGX_CHIP_G16G: return "G16G";
   case AGX_CHIP_G16S: return "G16S";
   case AGX_CHIP_G16C: return "G16C";
   case AGX_CHIP_G17P: return "G17P";
   case AGX_CHIP_G17A: return "G17A";
   case AGX_CHIP_G17G: return "G17G";
   case AGX_CHIP_G17S: return "G17S";
   case AGX_CHIP_G17C: return "G17C";
   case AGX_CHIP_G18G: return "G18G";
   default: return "unknown";
   }
}

/* Abstraction for command stream parsing */
typedef unsigned (*decode_cmd)(struct agxdecode_ctx *ctx, const uint8_t *map,
                               size_t available, uint64_t *link, bool verbose,
                               decoder_params *params, void *data);

#define STATE_DONE (0xFFFFFFFFu)
#define STATE_LINK (0xFFFFFFFEu)
#define STATE_CALL (0xFFFFFFFDu)
#define STATE_RET  (0xFFFFFFFCu)
#define STATE_ERROR (0xFFFFFFFBu)

/* Every refill, not just the first one, must stay inside a tracked BO: a
 * link, call, return or refill can land in the last buffer-full of a BO, and
 * fetching a whole buffer there overflows it (an assert, or a read past the
 * mapping in a release build). Decoder robustness, independent of the GPU. */
static int
agxdecode_fetch_stream(struct agxdecode_ctx *ctx, uint64_t va, void *buffer,
                       size_t size)
{
   if (!lib_config.read_gpu_mem) {
      struct agx_bo *alloc = agxdecode_find_mapped_gpu_mem_containing(ctx, va);
      if (!alloc)
         return 0;
      size = MIN2(size, alloc->size - (va - alloc->va->addr));
   }

   return agxdecode_fetch_gpu_mem(ctx, va, size, buffer);
}

static void
agxdecode_stateful(struct agxdecode_ctx *ctx, uint64_t va, const char *label,
                   decode_cmd decoder, bool verbose, decoder_params *params,
                   void *data)
{
   uint64_t stack[16];
   unsigned sp = 0;

   uint8_t buf[1024];
   size_t size = sizeof(buf);
   if (!lib_config.read_gpu_mem) {
      struct agx_bo *alloc = agxdecode_find_mapped_gpu_mem_containing(ctx, va);
      if (!alloc) {
         fprintf(agxdecode_dump_stream,
                 "%s (%" PRIx64 ") is not mapped; skipping decode\n", label,
                 va);
         return;
      }
      fprintf(agxdecode_dump_stream, "%s (%" PRIx64 ", handle %u)\n", label, va,
              alloc->handle);
      size = MIN2(size, alloc->size - (va - alloc->va->addr));
   } else {
      fprintf(agxdecode_dump_stream, "%s (%" PRIx64 ")\n", label, va);
   }
   fflush(agxdecode_dump_stream);

   int len = agxdecode_fetch_stream(ctx, va, buf, size);

   int left = len;
   uint8_t *map = buf;
   uint64_t link = 0;

   fflush(agxdecode_dump_stream);

   while (left) {
      if (len <= 0) {
         fprintf(agxdecode_dump_stream, "!! Failed to read GPU memory\n");
         fflush(agxdecode_dump_stream);
         return;
      }

      unsigned count =
         decoder(ctx, map, (size_t)left, &link, verbose, params, data);

      /* If we fail to decode, default to a hexdump (don't hang), without
       * reading past the bytes we have. */
      if (count == 0) {
         count = MIN2(8, left);
         u_hexdump(agxdecode_dump_stream, map, count, false);
      }

      fflush(agxdecode_dump_stream);
      if (count == STATE_DONE) {
         break;
      } else if (count == STATE_LINK) {
         fprintf(agxdecode_dump_stream, "Linking to 0x%" PRIx64 "\n\n", link);
         va = link;
         left = len = agxdecode_fetch_stream(ctx, va, buf, sizeof(buf));
         map = buf;
      } else if (count == STATE_CALL) {
         fprintf(agxdecode_dump_stream,
                 "Calling 0x%" PRIx64 " (return = 0x%" PRIx64 ")\n\n", link,
                 va + 8);
         assert(sp < ARRAY_SIZE(stack));
         stack[sp++] = va + 8;
         va = link;
         left = len = agxdecode_fetch_stream(ctx, va, buf, sizeof(buf));
         map = buf;
      } else if (count == STATE_RET) {
         assert(sp > 0);
         va = stack[--sp];
         fprintf(agxdecode_dump_stream, "Returning to 0x%" PRIx64 "\n\n", va);
         left = len = agxdecode_fetch_stream(ctx, va, buf, sizeof(buf));
         map = buf;
      } else if (count == STATE_ERROR) {
         fprintf(agxdecode_dump_stream,
                 "Stopping at a non-canonical command record\n");
         break;
      } else {
         if (count > (unsigned)left) {
            fprintf(agxdecode_dump_stream, "Truncated command record\n");
            break;
         }
         va += count;
         map += count;
         left -= count;

         if (left < 512 && len == sizeof(buf)) {
            left = len = agxdecode_fetch_stream(ctx, va, buf, sizeof(buf));
            map = buf;
         }
      }
   }
}

static void
agxdecode_texture_pbe(struct agxdecode_ctx *ctx, const void *map)
{
   struct AGX_TEXTURE tex;
   struct AGX_PBE pbe;

   bool valid_texture = AGX_TEXTURE_unpack(NULL, map, &tex);
   bool valid_pbe = AGX_PBE_unpack(NULL, map, &pbe);

   /* Try to guess if it's texture or PBE */
   valid_texture &=
      tex.swizzle_r <= AGX_CHANNEL_0 && tex.swizzle_g <= AGX_CHANNEL_0 &&
      tex.swizzle_b <= AGX_CHANNEL_0 && tex.swizzle_a <= AGX_CHANNEL_0;

   if (valid_texture && !valid_pbe) {
      DUMP_CL(TEXTURE, map, "Texture");
   } else if (valid_pbe && !valid_texture) {
      DUMP_CL(PBE, map, "PBE");
   } else {
      if (!valid_texture) {
         assert(!valid_pbe);
         fprintf(agxdecode_dump_stream, "XXX: invalid texture/PBE\n");
      }

      DUMP_CL(TEXTURE, map, "Texture");
      DUMP_CL(PBE, map, "PBE");
   }
}

static unsigned
agxdecode_usc(struct agxdecode_ctx *ctx, const uint8_t *map,
              UNUSED size_t available, UNUSED uint64_t *link,
              UNUSED bool verbose,
              decoder_params *params, UNUSED void *data)
{
   enum agx_sampler_states *sampler_states = data;
   enum agx_usc_control type = map[0];

   bool extended_samplers =
      (sampler_states != NULL) &&
      (((*sampler_states) == AGX_SAMPLER_STATES_8_EXTENDED) ||
       ((*sampler_states) == AGX_SAMPLER_STATES_16_EXTENDED));

#define USC_CASE(name, human)                                                  \
   case AGX_USC_CONTROL_##name: {                                              \
      DUMP_CL(USC_##name, map, human);                                         \
      return AGX_USC_##name##_LENGTH;                                          \
   }

   switch (type) {
   case AGX_USC_CONTROL_NO_PRESHADER: {
      DUMP_CL(USC_NO_PRESHADER, map, "No preshader");
      return STATE_DONE;
   }

   case AGX_USC_CONTROL_PRESHADER: {
      agx_unpack(agxdecode_dump_stream, map, USC_PRESHADER, ctrl);
      DUMP_UNPACKED(USC_PRESHADER, ctrl, "Preshader\n");

      void *buf;
      size_t size =
         agxdecode_grab_mapped(ctx, decode_usc(ctx, ctrl.code), &buf);

      if (size)
         agx_disassemble(buf, size, agxdecode_dump_stream);
      return STATE_DONE;
   }

   case AGX_USC_CONTROL_SHADER: {
      agx_unpack(agxdecode_dump_stream, map, USC_SHADER, ctrl);
      DUMP_UNPACKED(USC_SHADER, ctrl, "Shader\n");

      agxdecode_log("\n");
      void *buf;
      size_t size =
         agxdecode_grab_mapped(ctx, decode_usc(ctx, ctrl.code), &buf);

      if (size)
         agx_disassemble(buf, size, agxdecode_dump_stream);
      agxdecode_log("\n");

      return AGX_USC_SHADER_LENGTH;
   }

   case AGX_USC_CONTROL_SAMPLER: {
      agx_unpack(agxdecode_dump_stream, map, USC_SAMPLER, temp);
      DUMP_UNPACKED(USC_SAMPLER, temp, "Sampler state\n");

      size_t stride =
         AGX_SAMPLER_LENGTH + (extended_samplers ? AGX_BORDER_LENGTH : 0);
      uint8_t *samp = alloca(stride * temp.count);

      agxdecode_fetch_gpu_mem(ctx, temp.buffer, stride * temp.count, samp);

      for (unsigned i = 0; i < temp.count; ++i) {
         DUMP_CL(SAMPLER, samp, "Sampler");
         samp += AGX_SAMPLER_LENGTH;

         if (extended_samplers) {
            DUMP_CL(BORDER, samp, "Border");
            samp += AGX_BORDER_LENGTH;
         }
      }

      return AGX_USC_SAMPLER_LENGTH;
   }

   case AGX_USC_CONTROL_TEXTURE: {
      agx_unpack(agxdecode_dump_stream, map, USC_TEXTURE, temp);
      DUMP_UNPACKED(USC_TEXTURE, temp, "Texture state\n");

      uint8_t buf[AGX_TEXTURE_LENGTH * temp.count];
      uint8_t *tex = buf;

      agxdecode_fetch_gpu_array(ctx, temp.buffer, buf);

      /* Note: samplers only need 8 byte alignment? */
      for (unsigned i = 0; i < temp.count; ++i) {
         fprintf(agxdecode_dump_stream, "ts%u: \n", temp.start + i);
         agxdecode_texture_pbe(ctx, tex);

         tex += AGX_TEXTURE_LENGTH;
      }

      return AGX_USC_TEXTURE_LENGTH;
   }

   case AGX_USC_CONTROL_UNIFORM: {
      agx_unpack(agxdecode_dump_stream, map, USC_UNIFORM, temp);
      DUMP_UNPACKED(USC_UNIFORM, temp, "Uniform\n");

      uint8_t buf[2 * temp.size_halfs];
      agxdecode_fetch_gpu_array(ctx, temp.buffer, buf);
      u_hexdump(agxdecode_dump_stream, buf, 2 * temp.size_halfs, false);

      return AGX_USC_UNIFORM_LENGTH;
   }

   case AGX_USC_CONTROL_UNIFORM_HIGH: {
      agx_unpack(agxdecode_dump_stream, map, USC_UNIFORM_HIGH, temp);
      DUMP_UNPACKED(USC_UNIFORM_HIGH, temp, "Uniform (high)\n");

      uint8_t buf[2 * temp.size_halfs];
      agxdecode_fetch_gpu_array(ctx, temp.buffer, buf);
      u_hexdump(agxdecode_dump_stream, buf, 2 * temp.size_halfs, false);

      return AGX_USC_UNIFORM_HIGH_LENGTH;
   }

      USC_CASE(FRAGMENT_PROPERTIES, "Fragment properties");
      USC_CASE(SHARED, "Shared");
      USC_CASE(REGISTERS, "Registers");

   default:
      fprintf(agxdecode_dump_stream, "Unknown USC control type: %u\n", type);
      u_hexdump(agxdecode_dump_stream, map, 8, false);
      return 8;
   }

#undef USC_CASE
}

#define PPP_PRINT(map, header_name, struct_name, human)                        \
   if (hdr.header_name) {                                                      \
      if (((map + AGX_##struct_name##_LENGTH) > (base + size))) {              \
         fprintf(agxdecode_dump_stream, "Buffer overrun in PPP update\n");     \
         return;                                                               \
      }                                                                        \
      DUMP_CL(struct_name, map, human);                                        \
      map += AGX_##struct_name##_LENGTH;                                       \
      fflush(agxdecode_dump_stream);                                           \
   }

static void
agxdecode_record(struct agxdecode_ctx *ctx, uint64_t va, size_t size,
                 bool verbose, decoder_params *params)
{
   uint8_t buf[size];
   uint8_t *base = buf;
   uint8_t *map = base;

   agxdecode_fetch_gpu_array(ctx, va, buf);

   agx_unpack(agxdecode_dump_stream, map, PPP_HEADER, hdr);
   map += AGX_PPP_HEADER_LENGTH;

   PPP_PRINT(map, fragment_control, FRAGMENT_CONTROL, "Fragment control");
   PPP_PRINT(map, fragment_control_2, FRAGMENT_CONTROL, "Fragment control 2");
   PPP_PRINT(map, fragment_front_face, FRAGMENT_FACE, "Front face");
   PPP_PRINT(map, fragment_front_face_2, FRAGMENT_FACE_2, "Front face 2");
   PPP_PRINT(map, fragment_front_stencil, FRAGMENT_STENCIL, "Front stencil");
   PPP_PRINT(map, fragment_back_face, FRAGMENT_FACE, "Back face");
   PPP_PRINT(map, fragment_back_face_2, FRAGMENT_FACE_2, "Back face 2");
   PPP_PRINT(map, fragment_back_stencil, FRAGMENT_STENCIL, "Back stencil");
   PPP_PRINT(map, depth_bias_scissor, DEPTH_BIAS_SCISSOR, "Depth bias/scissor");

   if (hdr.region_clip) {
      if (((map + (AGX_REGION_CLIP_LENGTH * hdr.viewport_count)) >
           (base + size))) {
         fprintf(agxdecode_dump_stream, "Buffer overrun in PPP update\n");
         return;
      }

      for (unsigned i = 0; i < hdr.viewport_count; ++i) {
         DUMP_CL(REGION_CLIP, map, "Region clip");
         map += AGX_REGION_CLIP_LENGTH;
         fflush(agxdecode_dump_stream);
      }
   }

   if (hdr.viewport) {
      if (((map + AGX_VIEWPORT_CONTROL_LENGTH +
            (AGX_VIEWPORT_LENGTH * hdr.viewport_count)) > (base + size))) {
         fprintf(agxdecode_dump_stream, "Buffer overrun in PPP update\n");
         return;
      }

      DUMP_CL(VIEWPORT_CONTROL, map, "Viewport control");
      map += AGX_VIEWPORT_CONTROL_LENGTH;

      for (unsigned i = 0; i < hdr.viewport_count; ++i) {
         DUMP_CL(VIEWPORT, map, "Viewport");
         map += AGX_VIEWPORT_LENGTH;
         fflush(agxdecode_dump_stream);
      }
   }

   PPP_PRINT(map, w_clamp, W_CLAMP, "W clamp");
   PPP_PRINT(map, output_select, OUTPUT_SELECT, "Output select");
   PPP_PRINT(map, varying_counts_32, VARYING_COUNTS, "Varying counts 32");
   PPP_PRINT(map, varying_counts_16, VARYING_COUNTS, "Varying counts 16");
   PPP_PRINT(map, cull, CULL, "Cull");
   PPP_PRINT(map, cull_2, CULL_2, "Cull 2");

   if (hdr.fragment_shader) {
      agx_unpack(agxdecode_dump_stream, map, FRAGMENT_SHADER_WORD_0, frag_0);
      agx_unpack(agxdecode_dump_stream, map + 4, FRAGMENT_SHADER_WORD_1,
                 frag_1);
      agx_unpack(agxdecode_dump_stream, map + 8, FRAGMENT_SHADER_WORD_2,
                 frag_2);
      agxdecode_stateful(ctx, decode_usc(ctx, frag_1.pipeline),
                         "Fragment pipeline", agxdecode_usc, verbose, params,
                         &frag_0.sampler_state_register_count);

      if (frag_2.cf_bindings) {
         uint8_t buf[128];
         uint8_t *cf = buf;

         agxdecode_fetch_gpu_array(ctx, decode_usc(ctx, frag_2.cf_bindings),
                                   buf);
         u_hexdump(agxdecode_dump_stream, cf, 128, false);

         DUMP_CL(CF_BINDING_HEADER, cf, "Coefficient binding header:");
         cf += AGX_CF_BINDING_HEADER_LENGTH;

         for (unsigned i = 0; i < frag_0.cf_binding_count; ++i) {
            DUMP_CL(CF_BINDING, cf, "Coefficient binding:");
            cf += AGX_CF_BINDING_LENGTH;
         }
      }

      DUMP_CL(FRAGMENT_SHADER_WORD_0, map, "Fragment shader word 0");
      DUMP_CL(FRAGMENT_SHADER_WORD_1, map + 4, "Fragment shader word 1");
      DUMP_CL(FRAGMENT_SHADER_WORD_2, map + 8, "Fragment shader word 2");
      DUMP_CL(FRAGMENT_SHADER_WORD_3, map + 12, "Fragment shader word 3");
      map += 16;
   }

   PPP_PRINT(map, occlusion_query, FRAGMENT_OCCLUSION_QUERY, "Occlusion query");
   PPP_PRINT(map, occlusion_query_2, FRAGMENT_OCCLUSION_QUERY_2,
             "Occlusion query 2");
   PPP_PRINT(map, output_unknown, OUTPUT_UNKNOWN, "Output unknown");
   PPP_PRINT(map, output_size, OUTPUT_SIZE, "Output size");
   PPP_PRINT(map, varying_word_2, VARYING_2, "Varying word 2");

   /* PPP print checks we don't read too much, now check we read enough */
   assert(map == (base + size) && "invalid size of PPP update");
}

static unsigned
agxdecode_ioto(UNUSED struct agxdecode_ctx *ctx, const uint8_t *map,
               size_t available, UNUSED uint64_t *link, bool verbose,
               decoder_params *params, UNUSED void *data)
{
   const enum agx_chip chip = agxdecode_chip_from_params(params);
   if (chip == AGX_CHIP_G18G) {
      const char *display = NULL;
      int length = agx3d_instr_length(map, available, &display);
      if (length <= 0) {
         fprintf(agxdecode_dump_stream, "Unknown or truncated G18 Ioto instruction\n");
         return STATE_ERROR;
      }

      /* ISA decoders read a padded instruction, not an unaligned byte stream. */
      BITSET_WORD instruction[6] = {0};
      if ((unsigned)length > sizeof(instruction))
         return STATE_ERROR;
      memcpy(instruction, map, length);
      agx3d_disassemble_instr(instruction, agxdecode_dump_stream, 0, length,
                              verbose);
      return display && !strcmp(display, "end") ? STATE_DONE : (unsigned)length;
   }

   struct agx_ioto_program_record record;
   size_t consumed = 0;

   if (!agx_ioto_decode_program_record(chip, map, available, &record,
                                       &consumed)) {
      fprintf(agxdecode_dump_stream,
              "Unknown or non-canonical AGX3 Ioto record for %s\n",
              agxdecode_chip_name(chip));
      return STATE_ERROR;
   }

   switch (record.kind) {
   case AGX_IOTO_PROGRAM_RECORD_IOTO:
      fprintf(agxdecode_dump_stream, "AGX3 Ioto record kind %u (%zu bytes)\n",
              (unsigned)record.ioto.kind, consumed);
      break;
   case AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV:
      fprintf(agxdecode_dump_stream,
              "AGX3 Ioto inline device load (%zu bytes)\n", consumed);
      break;
   case AGX_IOTO_PROGRAM_RECORD_TERMINATE:
      agxdecode_log("AGX3 Ioto program terminate\n");
      return STATE_DONE;
   case AGX_IOTO_PROGRAM_RECORD_INVALID:
   default:
      return STATE_ERROR;
   }

   if (verbose)
      u_hexdump(agxdecode_dump_stream, record.bytes, consumed, false);
   return (unsigned)consumed;
}

static unsigned
agxdecode_cdm_agx3(struct agxdecode_ctx *ctx, const uint8_t *map,
                   size_t available, uint64_t *link, bool verbose,
                   decoder_params *params, enum agx_chip chip)
{
   struct agx3_cdm_decoded_record record;

   if (available < sizeof(uint32_t))
      return STATE_ERROR;

   /* Type zero has three witnessed lengths (24/36/40 bytes), and word 0's
    * mode field is what separates them -- 0 direct, 1 the nine-word indirect
    * form, 2 the compact one -- so the record decoder can size any of them
    * without guessing.  The direct SEMANTIC unpack is a stricter thing: it
    * also demands word 3 be a canonical dimension word.  The native compute
    * path derives word 3 from the program VA instead, which hardware settled
    * (see agx3_g17p_native_compute_launch_pack), so a mismatch there is a
    * note about an unresolved field, not a reason to abandon the stream. */
   if ((map[3] >> 5) == AGX_CDM_BLOCK_TYPE_LAUNCH &&
       ((map[3] >> 3) & 3) == 0) {
      struct agx3_cdm_direct_launch_info direct;
      if (!agx3_cdm_direct_launch_unpack(chip, map, available, &direct)) {
         fprintf(agxdecode_dump_stream,
                 "Note: AGX3 launch is not the canonical direct form for %s\n",
                 agxdecode_chip_name(chip));
         u_hexdump(agxdecode_dump_stream, map,
                   MIN2(available, (size_t)AGX3_CDM_DIRECT_LAUNCH_SIZE), false);
      }
   }

   if (!agx3_cdm_decode_one(chip, map, available, &record)) {
      fprintf(agxdecode_dump_stream,
              "Unknown or non-canonical AGX3 CDM record for %s\n",
              agxdecode_chip_name(chip));
      return STATE_ERROR;
   }

   switch (record.kind) {
   case AGX3_CDM_RECORD_LAUNCH:
      fprintf(agxdecode_dump_stream,
              "AGX3 Compute\n"
              "  Word 0 payload = 0x%08" PRIx32 "\n"
              "  Word 1 payload = 0x%08" PRIx32 "\n"
              "  Pipeline = 0x%012" PRIx64 "\n"
              "  Opaque word 3 = 0x%08" PRIx32 "\n"
              "  Global size = (%" PRIu32 ", %" PRIu32 ", %" PRIu32 ")\n"
              "  Local size = (%" PRIu32 ", %" PRIu32 ", %" PRIu32 ")\n",
              record.data.launch.word0_payload,
              record.data.launch.word1_payload, record.data.launch.pipeline,
              record.data.launch.opaque_word3,
              record.data.launch.global_size[0],
              record.data.launch.global_size[1],
              record.data.launch.global_size[2],
              record.data.launch.local_size[0],
              record.data.launch.local_size[1],
              record.data.launch.local_size[2]);
      if (record.data.launch.pipeline != 0)
         agxdecode_stateful(ctx, record.data.launch.pipeline, "Ioto pipeline",
                            agxdecode_ioto, verbose, params, NULL);
      return record.length;

   case AGX3_CDM_RECORD_INDIRECT_LAUNCH_36:
   case AGX3_CDM_RECORD_INDIRECT_LAUNCH:
      fprintf(agxdecode_dump_stream,
              "AGX3 Compute (indirect, %u bytes)\n"
              "  Word 0 payload = 0x%08" PRIx32 "\n"
              "  Word 1 payload = 0x%08" PRIx32 "\n"
              "  Pipeline = 0x%012" PRIx64 "\n"
              "  Opaque word 3 = 0x%08" PRIx32 "\n"
              "  Word 4 payload = 0x%04" PRIx16 "\n"
              "  Parameters = 0x%012" PRIx64 "\n",
              (unsigned)record.length,
              record.data.indirect_launch.word0_payload,
              record.data.indirect_launch.word1_payload,
              record.data.indirect_launch.pipeline,
              record.data.indirect_launch.opaque_word3,
              record.data.indirect_launch.word4_payload,
              record.data.indirect_launch.parameters);
      if (record.kind == AGX3_CDM_RECORD_INDIRECT_LAUNCH_36) {
         fprintf(agxdecode_dump_stream,
                 "  Trailing = (0x%08" PRIx32 ", 0x%08" PRIx32
                 ", 0x%08" PRIx32 ")\n",
                 record.data.indirect_launch.trailing_words[0],
                 record.data.indirect_launch.trailing_words[1],
                 record.data.indirect_launch.trailing_words[2]);
      }
      if (record.data.indirect_launch.pipeline != 0)
         agxdecode_stateful(ctx, record.data.indirect_launch.pipeline,
                            "Ioto pipeline", agxdecode_ioto, verbose, params,
                            NULL);
      return record.length;

   case AGX3_CDM_RECORD_STREAM_LINK:
      fprintf(agxdecode_dump_stream,
              "AGX3 Stream Link\n  Target = 0x%012" PRIx64
              "\n  With return = %s\n",
              record.data.stream_link.target,
              record.data.stream_link.with_return ? "true" : "false");
      *link = record.data.stream_link.target;
      return record.data.stream_link.with_return ? STATE_CALL : STATE_LINK;

   case AGX3_CDM_RECORD_LINK_PADDING:
      fprintf(agxdecode_dump_stream,
              "AGX3 Link Padding\n  Count = %u\n",
              record.data.link_padding.count);
      return record.length;

   case AGX3_CDM_RECORD_BARRIER:
      fprintf(agxdecode_dump_stream,
              "AGX3 Barrier\n  Flags = 0x%x\n", record.data.barrier.flags);
      return record.length;

   case AGX3_CDM_RECORD_SSTT_FLUSH_INVALIDATE:
      agxdecode_log("AGX3 SSTT Flush Invalidate\n");
      return record.length;

   case AGX3_CDM_RECORD_SUBSTREAM_BARRIER:
      fprintf(agxdecode_dump_stream,
              "AGX3 Substream Barrier\n  Flags = 0x%x\n  Substream = %u\n",
              record.data.substream_barrier.flags,
              record.data.substream_barrier.substream);
      return record.length;

   case AGX3_CDM_RECORD_SECONDARY_BARRIER:
      agxdecode_log("AGX3 Secondary Barrier\n");
      return record.length;

   case AGX3_CDM_RECORD_STREAM_TERMINATE:
      agxdecode_log("AGX3 Stream Terminate\n");
      return STATE_DONE;

   case AGX3_CDM_RECORD_UNKNOWN:
   default:
      return STATE_ERROR;
   }
}

static unsigned
agxdecode_cdm(struct agxdecode_ctx *ctx, const uint8_t *map,
              UNUSED size_t available, uint64_t *link, bool verbose,
              decoder_params *params, UNUSED void *data)
{
   const enum agx_chip chip = agxdecode_chip_from_params(params);
   if (agxdecode_chip_uses_agx3(chip))
      return agxdecode_cdm_agx3(ctx, map, available, link, verbose, params,
                                chip);

   /* Bits 29-31 contain the block type */
   enum agx_cdm_block_type block_type = (map[3] >> 5);

   switch (block_type) {
   case AGX_CDM_BLOCK_TYPE_LAUNCH: {
      size_t length =
         AGX_CDM_LAUNCH_WORD_0_LENGTH + AGX_CDM_LAUNCH_WORD_1_LENGTH;

#define CDM_PRINT(STRUCT_NAME, human)                                          \
   do {                                                                        \
      DUMP_CL(CDM_##STRUCT_NAME, map, human);                                  \
      map += AGX_CDM_##STRUCT_NAME##_LENGTH;                                   \
      length += AGX_CDM_##STRUCT_NAME##_LENGTH;                                \
   } while (0);

      agx_unpack(agxdecode_dump_stream, map + 0, CDM_LAUNCH_WORD_0, hdr0);
      agx_unpack(agxdecode_dump_stream, map + 4, CDM_LAUNCH_WORD_1, hdr1);

      agxdecode_stateful(ctx, decode_usc(ctx, hdr1.pipeline), "Pipeline",
                         agxdecode_usc, verbose, params,
                         &hdr0.sampler_state_register_count);
      DUMP_UNPACKED(CDM_LAUNCH_WORD_0, hdr0, "Compute\n");
      DUMP_UNPACKED(CDM_LAUNCH_WORD_1, hdr1, "Compute\n");
      map += 8;

      /* Added in G14X */
      if (params->gpu_generation >= 14 && params->num_clusters_total > 1)
         CDM_PRINT(UNK_G14X, "Unknown G14X");

      switch (hdr0.mode) {
      case AGX_CDM_MODE_DIRECT:
         CDM_PRINT(GLOBAL_SIZE, "Global size");
         CDM_PRINT(LOCAL_SIZE, "Local size");
         break;
      case AGX_CDM_MODE_INDIRECT_GLOBAL:
         CDM_PRINT(INDIRECT, "Indirect buffer");
         CDM_PRINT(LOCAL_SIZE, "Local size");
         break;
      case AGX_CDM_MODE_INDIRECT_LOCAL:
         CDM_PRINT(INDIRECT, "Indirect buffer");
         break;
      default:
         fprintf(agxdecode_dump_stream, "Unknown CDM mode: %u\n", hdr0.mode);
         break;
      }

      return length;
   }

   case AGX_CDM_BLOCK_TYPE_STREAM_LINK: {
      agx_unpack(agxdecode_dump_stream, map, CDM_STREAM_LINK, hdr);
      DUMP_UNPACKED(CDM_STREAM_LINK, hdr, "Stream Link\n");
      *link = hdr.target_lo | (((uint64_t)hdr.target_hi) << 32);
      return hdr.with_return ? STATE_CALL : STATE_LINK;
   }

   case AGX_CDM_BLOCK_TYPE_STREAM_TERMINATE: {
      DUMP_CL(CDM_STREAM_TERMINATE, map, "Stream Terminate");
      return STATE_DONE;
   }

   case AGX_CDM_BLOCK_TYPE_STREAM_RETURN: {
      DUMP_CL(CDM_STREAM_RETURN, map, "Stream Return");
      return STATE_RET;
   }

   case AGX_CDM_BLOCK_TYPE_BARRIER: {
      DUMP_CL(CDM_BARRIER, map, "Barrier");
      return AGX_CDM_BARRIER_LENGTH;
   }

   default:
      fprintf(agxdecode_dump_stream, "Unknown CDM block type: %u\n",
              block_type);
      u_hexdump(agxdecode_dump_stream, map, 8, false);
      return 8;
   }
}

static void
agxdecode_ppp_agx3(struct agxdecode_ctx *ctx, enum agx_chip chip, uint64_t va,
                   size_t size, bool verbose)
{
   uint8_t payload[AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH];
   if (size == 0 || size > sizeof(payload)) {
      fprintf(agxdecode_dump_stream,
              "AGX3 PPP payload size %zu is outside the bounded grammar\n",
              size);
      return;
   }

   if (!lib_config.read_gpu_mem &&
       !agxdecode_find_mapped_gpu_mem_containing(ctx, va)) {
      fprintf(agxdecode_dump_stream,
              "AGX3 PPP payload 0x%012" PRIx64 " is not mapped\n", va);
      return;
   }

   const int read = agxdecode_fetch_gpu_mem(ctx, va, size, payload);
   if (read < 0 || (size_t)read != size) {
      fprintf(agxdecode_dump_stream,
              "Failed to read AGX3 PPP payload at 0x%012" PRIx64 "\n", va);
      return;
   }

   struct agx_ppp_decoded_record record;
   if (!agx_ppp_decode_one(chip, payload, size, &record)) {
      fprintf(agxdecode_dump_stream,
              "Unsupported or non-canonical AGX3 PPP payload for %s "
              "(%zu bytes)\n",
              agxdecode_chip_name(chip), size);
      if (verbose)
         u_hexdump(agxdecode_dump_stream, payload, size, false);
      return;
   }

   switch (record.kind) {
   case AGX_PPP_DECODED_RECORD_W_CLAMP_CULL2:
      fprintf(agxdecode_dump_stream,
              "AGX3 PPP W-clamp/Cull-2\n"
              "  W-clamp raw = 0x%08" PRIx32 "\n"
              "  Cull-2 raw = 0x%08" PRIx32 "\n",
              record.data.w_clamp_cull2.w_clamp_raw,
              record.data.w_clamp_cull2.cull2_raw);
      break;
   case AGX_PPP_DECODED_RECORD_VIEWPORT:
      fprintf(agxdecode_dump_stream,
              "AGX3 PPP Viewports\n"
              "  Count = %u\n"
              "  Control raw = 0x%08" PRIx32 "\n",
              record.data.viewport.viewport_count,
              record.data.viewport.viewport_control_raw);
      break;
   case AGX_PPP_DECODED_RECORD_HAL300_DEPTH_STENCIL:
      fprintf(agxdecode_dump_stream,
              "AGX3 HAL300 PPP Depth/Stencil\n"
              "  Fragment control raw = 0x%08" PRIx32 "\n"
              "  Front face raw = 0x%08" PRIx32 "\n"
              "  Back face raw = 0x%08" PRIx32 "\n"
              "  Front stencil raw = 0x%08" PRIx32 "\n"
              "  Back stencil raw = 0x%08" PRIx32 "\n",
              record.data.hal300_depth_stencil.fragment_control_raw,
              record.data.hal300_depth_stencil.fragment_front_face_raw,
              record.data.hal300_depth_stencil.fragment_back_face_raw,
              record.data.hal300_depth_stencil.fragment_front_stencil_raw,
              record.data.hal300_depth_stencil.fragment_back_stencil_raw);
      break;
   case AGX_PPP_DECODED_RECORD_HAL300_DEPTH_BIAS_SCISSOR_BASES:
      fprintf(agxdecode_dump_stream,
              "AGX3 HAL300 PPP Depth-bias/Scissor bases\n"
              "  Depth-bias base = 0x%012" PRIx64 "\n"
              "  Scissor base = 0x%012" PRIx64 "\n",
              record.data.hal300_depth_bias_scissor_bases.depth_bias_base,
              record.data.hal300_depth_bias_scissor_bases.scissor_base);
      break;
   case AGX_PPP_DECODED_RECORD_INVALID:
   default:
      break;
   }
}

/* G17P and G18G carry PPP pointers as offsets from their render-context base.
 * Resolve that address only for decoder memory fetches; keep the raw packet
 * value visible so diagnostics cannot accidentally hide an encoder bug. */
static uint64_t
agxdecode_ppp_fetch_address(enum agx_chip chip, uint64_t wire_pointer)
{
   if (chip == AGX_CHIP_G18G && wire_pointer < (UINT64_C(1) << 39))
      return AGX_M5_DRIVER_BASE + wire_pointer;

   if (agx_chip_native_agx3(chip) &&
       wire_pointer < AGX_G17P_VDM_USER_SIZE)
      return AGX_G17P_VDM_USER_BASE + wire_pointer;

   return wire_pointer;
}

static unsigned
agxdecode_vdm_agx3(struct agxdecode_ctx *ctx, const uint8_t *map,
                   size_t available, uint64_t *link, bool verbose,
                   decoder_params *params, enum agx_chip chip)
{
   struct agx_vdm_decoded_record record;

   if (!agx_vdm_decode_one(chip, map, available, &record)) {
      fprintf(agxdecode_dump_stream,
              "Unknown or non-canonical AGX3 VDM record for %s\n",
              agxdecode_chip_name(chip));
      return STATE_ERROR;
   }

   switch (record.kind) {
   case AGX_VDM_RECORD_PPP_STATE_UPDATE: {
      const uint64_t fetch_pointer = agxdecode_ppp_fetch_address(
         chip, record.data.ppp_state_update.pointer);

      fprintf(agxdecode_dump_stream,
              "AGX3 PPP State Update\n"
              "  Wire pointer = 0x%012" PRIx64 "\n"
              "  Fetch pointer = 0x%012" PRIx64 "\n"
              "  Size = %" PRIu32 " words\n",
              record.data.ppp_state_update.pointer,
              fetch_pointer,
              record.data.ppp_state_update.size_words);
      agxdecode_ppp_agx3(ctx, chip, fetch_pointer,
                         (size_t)record.data.ppp_state_update.size_words * 4,
                         verbose);
      return record.length;
   }

   case AGX_VDM_RECORD_BARRIER:
      agxdecode_log("AGX3 VDM Barrier\n");
      return record.length;

   case AGX_VDM_RECORD_STATE_UPDATE:
      fprintf(agxdecode_dump_stream,
              "AGX3 VDM State Update\n"
              "  Vertex shader word 0 = 0x%08" PRIx32 "\n"
              "  Pipeline = 0x%012" PRIx64 "\n"
              "  Pipeline template = 0x%08" PRIx32 "\n"
              "  Vertex outputs = { 0x%08" PRIx32 ", 0x%08" PRIx32
              " }\n"
              "  Vertex unknown = 0x%08" PRIx32 "\n",
              record.data.state_update.raw_vertex_shader_word0,
              record.data.state_update.pipeline,
              record.data.state_update.raw_pipeline_template,
              record.data.state_update.raw_vertex_outputs[0],
              record.data.state_update.raw_vertex_outputs[1],
              record.data.state_update.raw_vertex_unknown);
      if (record.data.state_update.pipeline != 0)
         agxdecode_stateful(ctx, record.data.state_update.pipeline,
                            "Vertex Ioto pipeline", agxdecode_ioto, verbose,
                            params, NULL);
      return record.length;

   case AGX_VDM_RECORD_INDEX_LIST_NONINDEXED:
      fprintf(agxdecode_dump_stream,
              "AGX3 Direct Non-indexed Draw\n"
              "  Primitive = %u\n"
              "  Count = %" PRIu32 "\n"
              "  Instances = %" PRIu32 "\n"
              "  Start = %" PRIu32 "\n",
              record.data.index_list_nonindexed.primitive,
              record.data.index_list_nonindexed.count,
              record.data.index_list_nonindexed.instances,
              record.data.index_list_nonindexed.start);
      return record.length;

   case AGX_VDM_RECORD_INDEX_LIST_INDEXED:
      fprintf(agxdecode_dump_stream,
              "AGX3 Direct Indexed Draw\n"
              "  Primitive = %u\n"
              "  Restart = %s\n"
              "  Index size = %u\n"
              "  Index buffer = 0x%012" PRIx64 "\n"
              "  Count = %" PRIu32 "\n"
              "  Instances = %" PRIu32 "\n"
              "  Start = %" PRIu32 "\n"
              "  Index buffer size word = 0x%08" PRIx32 "\n",
              record.data.index_list_indexed.primitive,
              record.data.index_list_indexed.restart_enable ? "true" :
                                                              "false",
              record.data.index_list_indexed.index_size,
              record.data.index_list_indexed.index_buffer,
              record.data.index_list_indexed.count,
              record.data.index_list_indexed.instances,
              record.data.index_list_indexed.start,
              record.data.index_list_indexed.index_buffer_size_word);
      return record.length;

   case AGX_VDM_RECORD_INDEX_LIST_INDIRECT_NONINDEXED:
      fprintf(agxdecode_dump_stream,
              "AGX3 Indirect Non-indexed Draw\n"
              "  Primitive = %u\n"
              "  Indirect buffer = 0x%012" PRIx64 "\n",
              record.data.index_list_indirect_nonindexed.primitive,
              record.data.index_list_indirect_nonindexed.indirect_buffer);
      return record.length;

   case AGX_VDM_RECORD_INDEX_LIST_INDIRECT_INDEXED:
      fprintf(agxdecode_dump_stream,
              "AGX3 Indirect Indexed Draw\n"
              "  Primitive = %u\n"
              "  Restart = %s\n"
              "  Index size = %u\n"
              "  Index buffer = 0x%012" PRIx64 "\n"
              "  Indirect buffer = 0x%012" PRIx64 "\n"
              "  Restart index = 0x%08" PRIx32 "\n"
              "  Index buffer size word = 0x%08" PRIx32 "\n",
              record.data.index_list_indirect_indexed.primitive,
              record.data.index_list_indirect_indexed.restart_enable
                 ? "true"
                 : "false",
              record.data.index_list_indirect_indexed.index_size,
              record.data.index_list_indirect_indexed.index_buffer,
              record.data.index_list_indirect_indexed.indirect_buffer,
              record.data.index_list_indirect_indexed.restart_index,
              record.data.index_list_indirect_indexed.index_buffer_size_word);
      return record.length;

   case AGX_VDM_RECORD_STREAM_LINK:
      fprintf(agxdecode_dump_stream,
              "AGX3 VDM Stream Link\n  Target = 0x%012" PRIx64 "\n",
              record.data.stream_link.target);
      *link = record.data.stream_link.target;
      return STATE_LINK;

   case AGX_VDM_RECORD_STREAM_TERMINATE:
      agxdecode_log("AGX3 VDM Stream Terminate\n");
      return STATE_DONE;

   case AGX_VDM_RECORD_TESSELLATE_DIRECT:
      fprintf(agxdecode_dump_stream,
              "AGX3 Direct Tessellate\n"
              "  Factor buffer = 0x%012" PRIx64 "\n"
              "  Patch count = %" PRIu32 "\n"
              "  Instance count = %" PRIu32 "\n"
              "  Base patch = %" PRIu32 "\n"
              "  Base instance = %" PRIu32 "\n"
              "  Instance stride word = 0x%08" PRIx32 "\n"
              "  Factor buffer size minus one = 0x%08" PRIx32 "\n",
              record.data.tessellate_direct.factor_buffer,
              record.data.tessellate_direct.patch_count,
              record.data.tessellate_direct.instance_count,
              record.data.tessellate_direct.base_patch,
              record.data.tessellate_direct.base_instance,
              record.data.tessellate_direct.instance_stride_word,
              record.data.tessellate_direct.factor_buffer_size_minus_one);
      return record.length;

   case AGX_VDM_RECORD_TESSELLATE_INDIRECT:
      fprintf(agxdecode_dump_stream,
              "AGX3 Indirect Tessellate\n"
              "  Factor buffer = 0x%012" PRIx64 "\n"
              "  Indirect buffer = 0x%012" PRIx64 "\n"
              "  Instance stride word = 0x%08" PRIx32 "\n"
              "  Factor buffer size minus one = 0x%08" PRIx32 "\n",
              record.data.tessellate_indirect.factor_buffer,
              record.data.tessellate_indirect.indirect_buffer,
              record.data.tessellate_indirect.instance_stride_word,
              record.data.tessellate_indirect.factor_buffer_size_minus_one);
      return record.length;

   case AGX_VDM_RECORD_TESSELLATION_STATE:
      fprintf(agxdecode_dump_stream,
              "AGX3 Tessellation State\n"
              "  Partition = %u\n"
              "  Counter-clockwise = %u\n"
              "  Maximum factor = %u\n"
              "  Quads = %u\n"
              "  Float factors = %u\n"
              "  Step function = %u\n"
              "  Scale = 0x%04x\n",
              record.data.tessellation_state.partition,
              record.data.tessellation_state.counterclockwise,
              record.data.tessellation_state.max_factor,
              record.data.tessellation_state.quads,
              record.data.tessellation_state.float_factors,
              record.data.tessellation_state.step,
              record.data.tessellation_state.scale_half);
      return record.length;

   case AGX_VDM_RECORD_UNKNOWN:
   default:
      return STATE_ERROR;
   }
}

static unsigned
agxdecode_vdm(struct agxdecode_ctx *ctx, const uint8_t *map,
              size_t available, uint64_t *link, bool verbose,
              decoder_params *params, UNUSED void *data)
{
   const enum agx_chip chip = agxdecode_chip_from_params(params);
   if (agxdecode_chip_uses_agx3(chip))
      return agxdecode_vdm_agx3(ctx, map, available, link, verbose, params,
                                chip);

   /* Bits 29-31 contain the block type */
   enum agx_vdm_block_type block_type = (map[3] >> 5);

   switch (block_type) {
   case AGX_VDM_BLOCK_TYPE_BARRIER: {
      agx_unpack(agxdecode_dump_stream, map, VDM_BARRIER, hdr);
      DUMP_UNPACKED(VDM_BARRIER, hdr, "Barrier\n");
      return hdr.returns ? STATE_RET : AGX_VDM_BARRIER_LENGTH;
   }

   case AGX_VDM_BLOCK_TYPE_PPP_STATE_UPDATE: {
      agx_unpack(agxdecode_dump_stream, map, PPP_STATE, cmd);

      uint64_t address = (((uint64_t)cmd.pointer_hi) << 32) | cmd.pointer_lo;

      if (!lib_config.read_gpu_mem) {
         struct agx_bo *mem =
            agxdecode_find_mapped_gpu_mem_containing(ctx, address);

         if (!mem) {
            DUMP_UNPACKED(PPP_STATE, cmd, "Non-existent record (XXX)\n");
            return AGX_PPP_STATE_LENGTH;
         }
      }

      agxdecode_record(ctx, address, cmd.size_words * 4, verbose, params);
      return AGX_PPP_STATE_LENGTH;
   }

   case AGX_VDM_BLOCK_TYPE_VDM_STATE_UPDATE: {
      size_t length = AGX_VDM_STATE_LENGTH;
      agx_unpack(agxdecode_dump_stream, map, VDM_STATE, hdr);
      map += AGX_VDM_STATE_LENGTH;

#define VDM_PRINT(header_name, STRUCT_NAME, human)                             \
   if (hdr.header_name##_present) {                                            \
      DUMP_CL(VDM_STATE_##STRUCT_NAME, map, human);                            \
      map += AGX_VDM_STATE_##STRUCT_NAME##_LENGTH;                             \
      length += AGX_VDM_STATE_##STRUCT_NAME##_LENGTH;                          \
   }

      VDM_PRINT(restart_index, RESTART_INDEX, "Restart index");

      /* If word 1 is present but word 0 is not, fallback to compact samplers */
      enum agx_sampler_states sampler_states = 0;

      if (hdr.vertex_shader_word_0_present) {
         agx_unpack(agxdecode_dump_stream, map, VDM_STATE_VERTEX_SHADER_WORD_0,
                    word_0);
         sampler_states = word_0.sampler_state_register_count;
      }

      VDM_PRINT(vertex_shader_word_0, VERTEX_SHADER_WORD_0,
                "Vertex shader word 0");

      if (hdr.vertex_shader_word_1_present) {
         agx_unpack(agxdecode_dump_stream, map, VDM_STATE_VERTEX_SHADER_WORD_1,
                    word_1);
         fprintf(agxdecode_dump_stream, "Pipeline %X\n",
                 (uint32_t)word_1.pipeline);
         agxdecode_stateful(ctx, decode_usc(ctx, word_1.pipeline), "Pipeline",
                            agxdecode_usc, verbose, params, &sampler_states);
      }

      VDM_PRINT(vertex_shader_word_1, VERTEX_SHADER_WORD_1,
                "Vertex shader word 1");
      VDM_PRINT(vertex_outputs, VERTEX_OUTPUTS, "Vertex outputs");
      VDM_PRINT(tessellation, TESSELLATION, "Tessellation");
      VDM_PRINT(vertex_unknown, VERTEX_UNKNOWN, "Vertex unknown");
      VDM_PRINT(tessellation_scale, TESSELLATION_SCALE, "Tessellation scale");

#undef VDM_PRINT
      return hdr.tessellation_scale_present ? length : ALIGN_POT(length, 8);
   }

   case AGX_VDM_BLOCK_TYPE_INDEX_LIST: {
      size_t length = AGX_INDEX_LIST_LENGTH;
      agx_unpack(agxdecode_dump_stream, map, INDEX_LIST, hdr);
      DUMP_UNPACKED(INDEX_LIST, hdr, "Index List\n");
      map += AGX_INDEX_LIST_LENGTH;

#define IDX_PRINT(header_name, STRUCT_NAME, human)                             \
   if (hdr.header_name##_present) {                                            \
      DUMP_CL(INDEX_LIST_##STRUCT_NAME, map, human);                           \
      map += AGX_INDEX_LIST_##STRUCT_NAME##_LENGTH;                            \
      length += AGX_INDEX_LIST_##STRUCT_NAME##_LENGTH;                         \
   }

      IDX_PRINT(index_buffer, BUFFER_LO, "Index buffer");
      IDX_PRINT(index_count, COUNT, "Index count");
      IDX_PRINT(instance_count, INSTANCES, "Instance count");
      IDX_PRINT(start, START, "Start");
      IDX_PRINT(indirect_buffer, INDIRECT_BUFFER, "Indirect buffer");
      IDX_PRINT(index_buffer_size, BUFFER_SIZE, "Index buffer size");

#undef IDX_PRINT
      return length;
   }

   case AGX_VDM_BLOCK_TYPE_STREAM_LINK: {
      agx_unpack(agxdecode_dump_stream, map, VDM_STREAM_LINK, hdr);
      DUMP_UNPACKED(VDM_STREAM_LINK, hdr, "Stream Link\n");
      *link = hdr.target_lo | (((uint64_t)hdr.target_hi) << 32);
      return hdr.with_return ? STATE_CALL : STATE_LINK;
   }

   case AGX_VDM_BLOCK_TYPE_STREAM_TERMINATE: {
      DUMP_CL(VDM_STREAM_TERMINATE, map, "Stream Terminate");
      return STATE_DONE;
   }

   case AGX_VDM_BLOCK_TYPE_TESSELLATE: {
      size_t length = AGX_VDM_TESSELLATE_LENGTH;
      agx_unpack(agxdecode_dump_stream, map, VDM_TESSELLATE, hdr);
      DUMP_UNPACKED(VDM_TESSELLATE, hdr, "Tessellate List\n");
      map += AGX_VDM_TESSELLATE_LENGTH;

#define TESS_PRINT(header_name, STRUCT_NAME, human)                            \
   if (hdr.header_name##_present) {                                            \
      DUMP_CL(VDM_TESSELLATE_##STRUCT_NAME, map, human);                       \
      map += AGX_VDM_TESSELLATE_##STRUCT_NAME##_LENGTH;                        \
      length += AGX_VDM_TESSELLATE_##STRUCT_NAME##_LENGTH;                     \
   }

      TESS_PRINT(factor_buffer, FACTOR_BUFFER, "Factor buffer");
      TESS_PRINT(patch_count, PATCH_COUNT, "Patch");
      TESS_PRINT(instance_count, INSTANCE_COUNT, "Instance count");
      TESS_PRINT(base_patch, BASE_PATCH, "Base patch");
      TESS_PRINT(base_instance, BASE_INSTANCE, "Base instance");
      TESS_PRINT(instance_stride, INSTANCE_STRIDE, "Instance stride");
      TESS_PRINT(indirect, INDIRECT, "Indirect");
      TESS_PRINT(factor_buffer_size, FACTOR_BUFFER_SIZE, "Factor buffer size");

#undef TESS_PRINT
      return length;
   }

   default:
      fprintf(agxdecode_dump_stream, "Unknown VDM block type: %u\n",
              block_type);
      u_hexdump(agxdecode_dump_stream, map, 8, false);
      return 8;
   }
}

static void
agxdecode_sampler_heap(struct agxdecode_ctx *ctx, uint64_t heap, unsigned count)
{
   if (!heap)
      return;

   struct agx_sampler_packed samp[1024];
   agxdecode_fetch_gpu_array(ctx, heap, samp);

   for (unsigned i = 0; i < count; ++i) {
      bool nonzero = false;
      for (unsigned j = 0; j < ARRAY_SIZE(samp[i].opaque); ++j) {
         nonzero |= samp[i].opaque[j] != 0;
      }

      if (nonzero) {
         fprintf(agxdecode_dump_stream, "Heap sampler %u\n", i);

         agx_unpack(agxdecode_dump_stream, samp + i, SAMPLER, temp);
         agx_print(agxdecode_dump_stream, SAMPLER, temp, 2);
      }
   }
}

static void
agxdecode_helper(struct agxdecode_ctx *ctx, const char *prefix, uint64_t helper)
{
   if (helper & 1) {
      fprintf(agxdecode_dump_stream, "%s helper program:\n", prefix);
      uint8_t buf[1024];
      agx_disassemble(
         buf, agxdecode_fetch_gpu_array(ctx, decode_usc(ctx, helper & ~1), buf),
         agxdecode_dump_stream);
   }
}

static void
agxdecode_drm_cmd_render(struct agxdecode_ctx *ctx,
                         struct drm_asahi_params_global *params,
                         struct drm_asahi_cmd_render *c, bool verbose)
{
   const enum agx_chip chip = agxdecode_chip_from_params(params);
   decode_cmd graphics_program_decoder =
      agx_chip_native_agx3(chip) || chip == AGX_CHIP_G18G ? agxdecode_ioto
                                                          : agxdecode_usc;

   DUMP_FIELD(c, "%x", flags);
   DUMP_FIELD(c, "0x%llx", vdm_ctrl_stream_base);
   agxdecode_stateful(ctx, c->vdm_ctrl_stream_base, "Encoder", agxdecode_vdm,
                      verbose, params, NULL);
   DUMP_FIELD(c, "0x%x", ppp_ctrl);
   DUMP_FIELD(c, "0x%llx", ppp_multisamplectl);
   DUMP_CL(ZLS_CONTROL, &c->zls_ctrl, "ZLS Control");
   DUMP_FIELD(c, "0x%llx", depth.base);
   DUMP_FIELD(c, "0x%llx", depth.comp_base);
   DUMP_FIELD(c, "%u", depth.stride);
   DUMP_FIELD(c, "%u", depth.comp_stride);
   DUMP_FIELD(c, "0x%llx", stencil.base);
   DUMP_FIELD(c, "0x%llx", stencil.comp_base);
   DUMP_FIELD(c, "%u", stencil.stride);
   DUMP_FIELD(c, "%u", stencil.comp_stride);
   DUMP_FIELD(c, "0x%llx", isp_scissor_base);
   DUMP_FIELD(c, "0x%llx", isp_dbias_base);
   DUMP_FIELD(c, "%d", width_px);
   DUMP_FIELD(c, "%d", height_px);
   DUMP_FIELD(c, "%d", layers);
   DUMP_FIELD(c, "%d", samples);
   DUMP_FIELD(c, "%d", sample_size_B);
   DUMP_FIELD(c, "%d", utile_width_px);
   DUMP_FIELD(c, "%d", utile_height_px);
   DUMP_FIELD(c, "0x%x", bg.usc);
   DUMP_FIELD(c, "0x%x", bg.rsrc_spec);
   agxdecode_stateful(ctx, decode_usc(ctx, c->bg.usc & ~0x7), "Load pipeline",
                      graphics_program_decoder, verbose, params, NULL);
   DUMP_FIELD(c, "0x%x", eot.usc);
   DUMP_FIELD(c, "0x%x", eot.rsrc_spec);
   agxdecode_stateful(ctx, decode_usc(ctx, c->eot.usc & ~0x7), "Store pipeline",
                      graphics_program_decoder, verbose, params, NULL);
   DUMP_FIELD(c, "0x%x", partial_bg.usc);
   DUMP_FIELD(c, "0x%x", partial_bg.rsrc_spec);
   agxdecode_stateful(ctx, decode_usc(ctx, c->partial_bg.usc & ~0x7),
                      "Partial reload pipeline", graphics_program_decoder,
                      verbose, params, NULL);
   DUMP_FIELD(c, "0x%x", partial_eot.usc);
   DUMP_FIELD(c, "0x%x", partial_eot.rsrc_spec);
   agxdecode_stateful(ctx, decode_usc(ctx, c->partial_eot.usc & ~0x7),
                      "Partial store pipeline", graphics_program_decoder,
                      verbose, params, NULL);

   DUMP_FIELD(c, "0x%x", isp_zls_pixels);
   DUMP_FIELD(c, "0x%x", isp_bgobjdepth);
   DUMP_FIELD(c, "0x%x", isp_bgobjvals);

   agxdecode_sampler_heap(ctx, c->sampler_heap, c->sampler_count);

   agxdecode_helper(ctx, "Vertex", c->vertex_helper.binary);
   agxdecode_helper(ctx, "Fragment", c->fragment_helper.binary);
}

static void
agxdecode_drm_cmd_compute(struct agxdecode_ctx *ctx,
                          struct drm_asahi_params_global *params,
                          struct drm_asahi_cmd_compute *c, bool verbose)
{
   DUMP_FIELD(c, "%x", flags);
   DUMP_FIELD(c, "0x%llx", cdm_ctrl_stream_base);
   agxdecode_stateful(ctx, c->cdm_ctrl_stream_base, "Encoder", agxdecode_cdm,
                      verbose, params, NULL);

   agxdecode_sampler_heap(ctx, c->sampler_heap, c->sampler_count);
   agxdecode_helper(ctx, "Compute", c->helper.binary);
}

static void
agxdecode_drm_attachments(const char *name, struct drm_asahi_attachment *atts,
                          size_t size)
{
   fprintf(agxdecode_dump_stream, "%s attachments:\n", name);
   unsigned count = size / sizeof(struct drm_asahi_attachment);
   for (unsigned i = 0; i < count; i++) {
      DUMP_FIELD((&atts[i]), "0x%llx", size);
      DUMP_FIELD((&atts[i]), "0x%llx", pointer);
   }
}

void
agxdecode_drm_cmdbuf(struct agxdecode_ctx *ctx,
                     struct drm_asahi_params_global *params,
                     struct util_dynarray *cmdbuf, bool verbose)
{
   agxdecode_dump_file_open();

   for (unsigned offs = 0; offs < cmdbuf->size;) {
      struct drm_asahi_cmd_header *header =
         (void *)((uint8_t *)cmdbuf->data) + offs;
      offs += sizeof(*header);
      void *data = (void *)((uint8_t *)cmdbuf->data) + offs;

      if (header->cmd_type == DRM_ASAHI_CMD_RENDER) {
         agxdecode_drm_cmd_render(ctx, params, data, verbose);
      } else if (header->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         agxdecode_drm_cmd_compute(ctx, params, data, verbose);
      } else if (header->cmd_type == DRM_ASAHI_SET_VERTEX_ATTACHMENTS) {
         agxdecode_drm_attachments("Vertex", data, header->size);
      } else if (header->cmd_type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS) {
         agxdecode_drm_attachments("Fragment", data, header->size);
      } else if (header->cmd_type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS) {
         agxdecode_drm_attachments("Compute", data, header->size);
      } else {
         UNREACHABLE("Invalid command type");
      }

      offs += header->size;
   }
}

static void
chip_id_to_params(decoder_params *params, uint32_t chip_id)
{
   switch (chip_id) {
   case 0x6000 ... 0x6002:
      *params = (decoder_params){
         .gpu_generation = 13,
         .gpu_variant = "SCD"[chip_id & 15],
         .chip_id = chip_id,
         .num_clusters_total = 2 << (chip_id & 15),
      };
      break;
   case 0x6020 ... 0x6022:
      *params = (decoder_params){
         .gpu_generation = 14,
         .gpu_variant = "SCD"[chip_id & 15],
         .chip_id = chip_id,
         .num_clusters_total = 2 << (chip_id & 15),
      };
      break;
   case 0x8112:
      *params = (decoder_params){
         .gpu_generation = 14,
         .gpu_variant = 'G',
         .chip_id = chip_id,
         .num_clusters_total = 1,
      };
      break;
   case 0x8152:
      *params = (decoder_params){
         .gpu_generation = 18,
         .gpu_variant = 'G',
         .chip_id = chip_id,
         .num_clusters_total = 2,
      };
      break;
   case 0x8103:
   default:
      *params = (decoder_params){
         .gpu_generation = 13,
         .gpu_variant = 'G',
         .chip_id = chip_id,
         .num_clusters_total = 1,
      };
      break;
   }
}

void
agxdecode_cmdstream(struct agxdecode_ctx *ctx, unsigned cmdbuf_handle,
                    unsigned map_handle, bool verbose)
{
   agxdecode_dump_file_open();

   struct agx_bo *cmdbuf = agxdecode_find_handle(ctx, cmdbuf_handle);
   assert(cmdbuf != NULL && "nonexistent command buffer");

   struct drm_asahi_params_global params;

   chip_id_to_params(&params, 0x8103);

   uint32_t *map = cmdbuf->_map;
   uint64_t encoder = (uint64_t)map[14] | ((uint64_t)(map[15]) << 32);

   if (map[13] == 3) {
      agxdecode_stateful(ctx, encoder, "Encoder", agxdecode_cdm, verbose,
                         &params, NULL);
   } else {
      agxdecode_stateful(ctx, encoder, "Encoder", agxdecode_vdm, verbose,
                         &params, NULL);
   }
}

void
agxdecode_track_alloc(struct agxdecode_ctx *ctx, struct agx_bo *alloc)
{
   util_dynarray_foreach(&ctx->mmap_array, struct agx_bo, it) {
      if (it->handle == alloc->handle) {
         /* PRIME can hand the same shared BO back to a client repeatedly.
          * Keep one current decoder view instead of aborting the application
          * for an otherwise valid, idempotent import. */
         *it = *alloc;
         return;
      }
   }

   util_dynarray_append(&ctx->mmap_array, *alloc);
}

void
agxdecode_track_free(struct agxdecode_ctx *ctx, struct agx_bo *bo)
{
   util_dynarray_foreach(&ctx->mmap_array, struct agx_bo, it) {
      if (it->handle == bo->handle) {
         memset(it, 0, sizeof(*it));
         return;
      }
   }
}

static int agxdecode_dump_frame_count = 0;

void
agxdecode_dump_file_open(void)
{
   if (agxdecode_dump_stream)
      return;

   /* This does a os_get_option every frame, so it is possible to use
    * os_set_option to change the base at runtime.
    */
   const char *dump_file_base =
      os_get_option("AGXDECODE_DUMP_FILE") ?: "agxdecode.dump";
   if (!strcmp(dump_file_base, "stderr"))
      agxdecode_dump_stream = stderr;
   else {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "%s.%ld.%04d", dump_file_base,
               (long)getpid(), agxdecode_dump_frame_count);
      printf("agxdecode: dump command stream to file %s\n", buffer);
      agxdecode_dump_stream = fopen(buffer, "w");
      if (!agxdecode_dump_stream) {
         fprintf(stderr,
                 "agxdecode: failed to open command stream log file %s\n",
                 buffer);
      }
   }
}

static void
agxdecode_dump_file_close(void)
{
   if (agxdecode_dump_stream && agxdecode_dump_stream != stderr) {
      fclose(agxdecode_dump_stream);
      agxdecode_dump_stream = NULL;
   }
}

void
agxdecode_next_frame(void)
{
   agxdecode_dump_file_close();
   agxdecode_dump_frame_count++;
}

void
agxdecode_close(void)
{
   agxdecode_dump_file_close();
}

static ssize_t
libagxdecode_writer(void *cookie, const char *buffer, size_t size)
{
   return lib_config.stream_write(buffer, size);
}

#if defined(_GNU_SOURCE) && !DETECT_OS_ANDROID
static cookie_io_functions_t funcs = {.write = libagxdecode_writer};
#endif

static decoder_params lib_params;

void
libagxdecode_init(struct libagxdecode_config *config)
{
#if defined(_GNU_SOURCE) && !DETECT_OS_ANDROID
   lib_config = *config;
   agxdecode_dump_stream = fopencookie(NULL, "w", funcs);

   chip_id_to_params(&lib_params, config->chip_id);
#else
   /* fopencookie is a glibc extension */
   UNREACHABLE("libagxdecode only available with glibc");
#endif
}

void
libagxdecode_vdm(struct agxdecode_ctx *ctx, uint64_t addr, const char *label,
                 bool verbose)
{
   agxdecode_stateful(ctx, addr, label, agxdecode_vdm, verbose, &lib_params,
                      NULL);
}

void
libagxdecode_cdm(struct agxdecode_ctx *ctx, uint64_t addr, const char *label,
                 bool verbose)
{
   agxdecode_stateful(ctx, addr, label, agxdecode_cdm, verbose, &lib_params,
                      NULL);
}
void
libagxdecode_usc(struct agxdecode_ctx *ctx, uint64_t addr, const char *label,
                 bool verbose)
{
   agxdecode_stateful(ctx, addr, label, agxdecode_usc, verbose, &lib_params,
                      NULL);
}
void
libagxdecode_shutdown(void)
{
   agxdecode_dump_file_close();
}
