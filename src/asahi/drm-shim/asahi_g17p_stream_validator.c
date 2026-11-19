/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi_g17p_stream_validator.h"

#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_ioto_program_core.h"
#include "asahi/lib/agx_g17p_draw_state_upload.h"
#include "asahi/lib/agx_ppp_viewport_payload_core.h"
#include "asahi/lib/agx_ppp_w_clamp_cull2_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "asahi/lib/agx3_cdm_decode_core.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_vdm_decode_core.h"
#include "asahi/lib/agx_vdm_hal200_bind_tail_core.h"
#include "asahi/isa/disasm.h"
#include "drm-uapi/asahi_drm.h"

#include <stdio.h>
#include <string.h>

#define G17P_CONTEXT_BASE UINT64_C(0x1000000000)
#define G17P_USC_BASE UINT64_C(0x10000000000)
#define MAX_RECORDS 65536
#define MAX_IOTO_PROGRAM_SIZE 6144
#define G17P_CDM_DIRECT_LAUNCH_SIZE 40

static uint32_t
load_le32(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t
load_le16(const uint8_t *p)
{
   return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint64_t
compact_address(uint64_t address)
{
   return address < G17P_CONTEXT_BASE ? G17P_CONTEXT_BASE + address : address;
}

static uint64_t
decode_pipeline_pair(const uint8_t pair[8])
{
   const uint32_t high = load_le32(pair);
   const uint32_t low = load_le32(pair + 4);

   /* The high word's low 22 bits are the stage-program template, not address.
    * A fragment with a constant program carries its thread-invariant spill
    * quads in bits 3..20 (agx_ppp_hal200_fragment_ti_resources, the native
    * fragment record layout); nothing sets the remaining template bits. */
   if (high & UINT32_C(0x003fffff) & ~UINT32_C(0x001ffff8))
      return 0;

   return ((uint64_t)(high >> 22) << 38) | ((uint64_t)low << 6);
}

static uint64_t
load_le64(const uint8_t *p)
{
   return (uint64_t)load_le32(p) | ((uint64_t)load_le32(p + 4) << 32);
}

/*
 * The 44-bit surface address of an AGX3 texture heap element.
 *
 * cmdbuf.xml "Texture Gen9"/"Texture Gen10": <field name="Address" size="44"
 * start="64" type="address" modifier="shr(4)"/>.  So bits 64..107 of the
 * 32-byte descriptor, times sixteen.
 *
 * This pointer never appears in the command stream.  The stream only names
 * the descriptor; the texture unit dereferences the address inside it at draw
 * time.  That makes it the one GPU-dereferenced pointer a command-stream walk
 * has historically been unable to see -- and it is exactly the pointer a
 * never-written or recycled descriptor set aims at unmapped memory.
 */
static uint64_t
agx3_texture_descriptor_surface(const uint8_t bytes[32])
{
   const uint64_t low = load_le32(bytes + 8);
   const uint64_t high = load_le32(bytes + 12) & UINT32_C(0xfff);

   return (low | (high << 32)) << 4;
}

/*
 * A descriptor that is entirely zero is the reserved null texture, not a bug:
 * hk_device.c pins heap entry zero to it and every null descriptor path in
 * hk_descriptor_set.c writes it deliberately.  Its surface address is zero,
 * which is never dereferenced.
 */
static bool
agx3_descriptor_is_blank(const uint8_t bytes[32])
{
   for (unsigned i = 0; i < 32; ++i) {
      if (bytes[i])
         return false;
   }

   return true;
}

static bool
fail_at(uint64_t address, const char *reason)
{
   fprintf(stderr, "shim: invalid G17P command stream at %#llx: %s\n",
           (unsigned long long)address, reason);
   return false;
}

/*
 * A record the decoder understands but this oracle does not model.
 *
 * This validator's accept-set was grown from streams the Gallium driver was
 * observed to emit.  Honeykrisp legitimately emits records outside that set --
 * tessellation and geometry lower through TESSELLATE_DIRECT, and the compute
 * side uses the substream/secondary barriers and link padding.  Refusing them
 * would report a gap in the ORACLE as a bug in the DRIVER, which is the one
 * misclassification that makes this whole harness untrustworthy.
 *
 * These records carry a decoded length, so the rest of the stream is still
 * checked: say plainly that the oracle is the thing that fell short, and skip.
 */
static void
oracle_gap_at(uint64_t address, const char *what)
{
   fprintf(stderr,
           "shim: G17P oracle gap at %#llx: %s is decodable but not modelled "
           "by this validator -- NOT a driver bug, stream not rejected\n",
           (unsigned long long)address, what);
}

/*
 * A record the decoder could not identify at all.  Unlike the above there is
 * no length, so validation cannot continue.  The cause is genuinely ambiguous
 * -- a corrupt stream and a decoder that has never seen this opcode look
 * identical from here -- so do not let it masquerade as a proven driver bug.
 */
/*
 * An ambiguous observation that does NOT stop the walk.  Some checks can spot
 * something odd without being able to prove it wrong -- a payload that reads
 * like an unowned pointer pair, say, when the same range legitimately carries
 * push constants, divisors and indirect arguments.  Report the shape so a
 * human can judge it, and keep validating.
 */
static void
note_indeterminate_at(uint64_t address, const char *what)
{
   fprintf(stderr,
           "shim: G17P indeterminate at %#llx: %s -- not by itself evidence of "
           "a driver bug; stream not rejected\n",
           (unsigned long long)address, what);
}

static bool
fail_indeterminate_at(uint64_t address, const char *what)
{
   fprintf(stderr,
           "shim: G17P undecodable record at %#llx: %s -- INDETERMINATE "
           "(stream corruption or a decoder gap); not by itself evidence of a "
           "driver bug\n",
           (unsigned long long)address, what);
   return false;
}

static bool
read_owned(const struct asahi_g17p_stream_memory *memory, uint64_t address,
           void *dst, size_t size)
{
   return memory->owned(memory->data, address, size) &&
          memory->read(memory->data, address, dst, size);
}

/* Walk agx3b code at address with the generated decoder up to the 0xe stop,
 * bounded by available. On success *end is the offset just past the stop. */
static bool
code_reaches_stop(const struct asahi_g17p_stream_memory *memory,
                  uint64_t address, size_t available, size_t *end)
{
   FILE *sink = memory->verbose ? stderr : fopen("/dev/null", "w");
   if (!sink)
      return fail_at(address, "could not open disassembly sink");

   size_t offset = 0;
   bool valid = false;
   while (offset < available) {
      uint8_t bytes[64] = {0};
      const size_t readable =
         available - offset < sizeof(bytes) ? available - offset
                                            : sizeof(bytes);
      if (!memory->read(memory->data, address + offset, bytes, readable) ||
          readable < 2)
         break;

      const signed decoded = agx3b_disassemble_instr(
         (BITSET_WORD *)bytes, sink, (unsigned)offset, readable, false);
      const size_t length = decoded < 0 ? (size_t)-decoded : (size_t)decoded;
      /* The generated decoder reports a negative length for an instruction
       * carrying not-yet-named bits, while still giving its authoritative
       * byte length. That is usable for a bounded walk; zero is not. */
      if (decoded == 0 || !length || length > readable)
         break;

      if (length == 4 && load_le32(bytes) == UINT32_C(0x0000000e)) {
         valid = true;
         offset += length;
         break;
      }
      offset += length;
   }

   if (!memory->verbose)
      fclose(sink);
   *end = offset;
   return valid;
}

static bool
validate_shader_blob(const struct asahi_g17p_stream_memory *memory,
                     uint64_t address)
{
   size_t available;
   if ((address & 1) || !memory->span ||
       !memory->span(memory->data, address, &available) || available < 4)
      return fail_at(address, "shader entry is unaligned or unowned");

   /* LDSHDR names +0x3c0 inside Apple's executable container. The dword at
    * entry-0x80 is the exact 0x80 framing plus 64-byte-padded main span; use
    * it instead of the page-rounded BO mapping as the executable bound. */
   uint8_t container_head[4], block_head[4];
   if (address < 0x3c0 ||
       !read_owned(memory, address - 0x3c0, container_head,
                   sizeof(container_head)) ||
       !read_owned(memory, address - 0x80, block_head, sizeof(block_head)) ||
       load_le32(container_head) != AGX3_CONTAINER_LEAD_SIZE)
      return fail_at(address, "shader does not own a G17P code container");
   const uint32_t block_size = load_le32(block_head);
   if (block_size < 0x80 || (block_size & 63) || block_size - 0x80 > available)
      return fail_at(address, "shader container has an invalid code span");
   available = block_size - 0x80;

   /* The container's own span bounds the walk. A fixed 16 KiB bound used to
    * reject complete fragment shaders past it -- 16bit_storage
    * struct_mixed_types links a 27986-byte fragment -- as having no end. */
   size_t offset;
   if (!code_reaches_stop(memory, address, available, &offset))
      return fail_at(address, "shader has no decodable terminal end");
   for (; offset < available; ++offset) {
      uint8_t byte;
      if (!memory->read(memory->data, address + offset, &byte, 1) || byte)
         return fail_at(address + offset,
                        "shader container has nonzero trailing code");
   }
   return true;
}

static bool
validate_ioto_program(const struct asahi_g17p_stream_memory *memory,
                      uint64_t address, bool fragment, bool nested_tables,
                      struct asahi_g17p_stream_stats *stats)
{
   size_t available;
   if ((address & 63) || !memory->span ||
       !memory->span(memory->data, address, &available) || available < 4)
      return fail_at(address, "Ioto program is unaligned or unowned");

   available = MIN2(available, MAX_IOTO_PROGRAM_SIZE);
   uint64_t cursor = address;
   unsigned branches = 0;
   bool seen_speclm = false, seen_ldshdr = false, seen_spectpr = false;
   bool seen_pixrel = false;
   uint32_t immediates[128] = {0};
   bool immediate_valid[128] = {0};
   for (unsigned i = 0; i < 2048 && cursor - address < available; ++i) {
      uint8_t bytes[AGX_IOTO_MAX_RECORD_SIZE] = {0};
      const size_t remain = available - (cursor - address);
      const size_t readable = MIN2(remain, sizeof(bytes));
      if (!memory->read(memory->data, cursor, bytes, readable))
         return fail_at(cursor, "Ioto record is unreadable");

      struct agx_ioto_program_record record;
      size_t consumed;
      if (!agx_ioto_decode_program_record(AGX_CHIP_G17P, bytes, readable,
                                           &record, &consumed) ||
          !consumed || consumed > readable) {
         /* Once the state tail is complete, the program may continue into
          * its constant program instead of terminating: hk's
          * graphics_constant_suffix_replace swaps exactly this program's
          * terminator for the shader's preamble, inline when it fits
          * (AGX_IOTO_PRESHADER_INLINE_LIMIT). That is ordinary compiled
          * code, a superset of the Ioto records, ending in its own stop;
          * walk the remainder with the shader decoder. */
         if (seen_speclm && seen_ldshdr && seen_spectpr &&
             fragment == seen_pixrel) {
            size_t end;
            if (code_reaches_stop(memory, cursor, remain, &end))
               return true;
            return fail_at(cursor,
                           "inline constant program has no decodable stop");
         }

         fprintf(stderr, "shim: Ioto bytes");
         for (size_t j = 0; j < readable; ++j)
            fprintf(stderr, " %02x", bytes[j]);
         fprintf(stderr, "\n");
         /* A record the Ioto decoder cannot identify is genuinely ambiguous:
          * a truncated program and a decoder that has never been given this
          * payload look identical from here, and the second has already
          * happened once (every MRT release descriptor was undecodable while
          * the driver was emitting the natively witnessed value). Validation
          * cannot continue either way, but do not call it a driver bug. */
         return fail_at(cursor,
                        "Ioto record is undecodable -- INDETERMINATE (a "
                        "truncated program or a decoder gap); not by itself "
                        "evidence of a driver bug");
      }

      if (record.kind == AGX_IOTO_PROGRAM_RECORD_TERMINATE) {
         if (!seen_speclm || !seen_ldshdr || !seen_spectpr ||
             fragment != seen_pixrel)
            return fail_at(cursor, "incomplete Ioto graphics state tail");
         return true;
      }

      if (record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO) {
         switch (record.ioto.kind) {
         case AGX_IOTO_KIND_LDIMM_EP0:
         case AGX_IOTO_KIND_LDIMM_EP1: {
            const uint16_t w0 = load_le16(bytes);
            unsigned dest;
            uint32_t value;
            if (record.ioto.kind == AGX_IOTO_KIND_LDIMM_EP0) {
               dest = (w0 >> 4) & 0xf;
               value = bytes[1];
            } else {
               const uint16_t w1 = load_le16(bytes + 2);
               const uint16_t w2 = load_le16(bytes + 4);
               const uint16_t w3 = load_le16(bytes + 6);
               dest = ((w0 >> 4) & 0xf) | (((w1 >> 6) & 3) << 4) |
                      (((w1 >> 3) & 1) << 6);
               value = ((uint32_t)(w0 >> 8) & 0x7f) |
                       (((uint32_t)w2 >> 1) & 0xf) << 7 |
                       (((uint32_t)w2 >> 10) & 3) << 11 |
                       ((uint32_t)w3 & 0xf) << 13 |
                       (((uint32_t)w3 >> 4) & 0xff) << 17 |
                       (((uint32_t)w1 >> 9) & 0x7f) << 25;
            }
            if (dest >= ARRAY_SIZE(immediates))
               return fail_at(cursor, "Ioto LDIMM destination is invalid");
            immediates[dest] = value;
            immediate_valid[dest] = true;

            /* The bounded absolute-load builder always materializes a 64-bit
             * source as consecutive low/high dwords. Validate it as soon as
             * the high half arrives, independently of the producer's range
             * list. A non-fragment EOT program is the one no-nested-table
             * form: dwords 2/3 hold its PBE argument address, while 4/5 are
             * the two MRT imageblock offsets and must not be mistaken for an
             * address. */
            if (dest && immediate_valid[dest - 1]) {
               const uint64_t source =
                  immediates[dest - 1] | ((uint64_t)value << 32);
               const bool absolute_source = fragment || nested_tables ||
                                            dest == 3;
               if (absolute_source && source > UINT32_MAX) {
                  if (!memory->owned(memory->data, source, 4))
                     return fail_at(cursor, "Ioto load source is unowned");

                  uint8_t nested[16];
                  if (nested_tables &&
                      memory->read(memory->data, source, nested,
                                   sizeof(nested))) {
                     const uint64_t a = load_le64(nested);
                     const uint64_t b = load_le64(nested + 8);
                     /* These sixteen bytes are a uniform payload. Nothing here
                      * knows what the encoder put in them: for a sampling
                      * stage it is the texture/sampler pointer pair, but the
                      * same range carries push constants, attribute divisors
                      * and indirect draw arguments, and the driver's range
                      * list -- which does know -- is not visible from a
                      * command-stream walk.
                      *
                      * This used to guess, on nothing more than both halves
                      * landing in (2^32, 2^48), and REJECT the stream when the
                      * guessed targets were unmapped. On G17P the user heap
                      * starts at 0x4000000000, so every legitimate address in
                      * any payload satisfies that range and so does plenty of
                      * ordinary data -- which is why the rejections clustered
                      * on indirect, instanced and divisor draws while simple
                      * draws, whose payload really is a pointer pair, passed.
                      *
                      * A check that cannot tell a pointer from an integer must
                      * not fail a stream. Report the shape and walk on. */
                     const bool pointer_shaped =
                        a > UINT32_MAX && b > UINT32_MAX &&
                        a < (UINT64_C(1) << 48) && b < (UINT64_C(1) << 48);
                     if (pointer_shaped &&
                         (!memory->owned(memory->data, a, 32) ||
                          !memory->owned(memory->data, b, 32))) {
                        char what[160];
                        snprintf(what, sizeof(what),
                                 "Ioto payload reads as an unowned pointer "
                                 "pair (%#llx, %#llx), but this range also "
                                 "carries push constants, divisors and "
                                 "indirect arguments",
                                 (unsigned long long)a, (unsigned long long)b);
                        note_indeterminate_at(source, what);
                     }
                     /* One level deeper than the pair itself: when the
                      * first half really is a texture descriptor, the surface
                      * it names must be owned.  A never-written descriptor set
                      * -- a PARTIALLY_BOUND binding, an unwritten array
                      * element, a range recycled out of a destroyed set --
                      * decodes here as a live-looking pointer into memory this
                      * VM has never mapped, and on G17P that is a halting
                      * GMMU read fault rather than a soft-faulted zero.
                      *
                      * Report by default and reject only when the caller says
                      * the payload is known to be a resource pair, because the
                      * same bytes legitimately carry push constants. */
                     if (memory->strict_descriptors && pointer_shaped &&
                         memory->owned(memory->data, a, 32)) {
                        uint8_t descriptor[32] = {0};
                        if (memory->read(memory->data, a, descriptor,
                                         sizeof(descriptor))) {
                           const uint64_t surface =
                              agx3_texture_descriptor_surface(descriptor);
                           if (!agx3_descriptor_is_blank(descriptor) &&
                               surface != 0 &&
                               !memory->owned(memory->data, surface, 16)) {
                              char what[192];
                              snprintf(what, sizeof(what),
                                       "texture descriptor at %#llx names "
                                       "unowned surface %#llx -- an unwritten "
                                       "or recycled descriptor set reads "
                                       "exactly like this",
                                       (unsigned long long)a,
                                       (unsigned long long)surface);
                              note_indeterminate_at(source, what);
                              if (memory->strict_descriptors > 1)
                                 return fail_at(source, what);
                           }
                        }
                     }
                     if (memory->verbose && pointer_shaped) {
                        uint8_t texture[32] = {0}, sampler[32] = {0};
                        if (memory->read(memory->data, a, texture,
                                         sizeof(texture)) &&
                            memory->read(memory->data, b, sampler,
                                         sizeof(sampler))) {
                           fprintf(stderr,
                                   "shim: G17P resource pair texture=%#llx "
                                   "sampler=%#llx texture-bytes=",
                                   (unsigned long long)a,
                                   (unsigned long long)b);
                           for (unsigned j = 0; j < sizeof(texture); ++j)
                              fprintf(stderr, "%02x", texture[j]);
                           fprintf(stderr, " sampler-bytes=");
                           for (unsigned j = 0; j < sizeof(sampler); ++j)
                              fprintf(stderr, "%02x", sampler[j]);
                           fprintf(stderr, "\n");
                        }
                     }
                  }
               }
            }
            break;
         }
         case AGX_IOTO_KIND_SPECLM_EP0:
            if (seen_speclm || seen_ldshdr || seen_spectpr)
               return fail_at(cursor, "misordered Ioto SPECLM");
            seen_speclm = true;
            break;
         case AGX_IOTO_KIND_LDSHDR_EP0: {
            if (!seen_speclm || seen_ldshdr || seen_spectpr || consumed != 10)
               return fail_at(cursor, "misordered Ioto LDSHDR");
            const uint16_t w1 = load_le16(bytes + 2);
            const uint16_t w2 = load_le16(bytes + 4);
            const uint16_t w3 = load_le16(bytes + 6);
            const uint16_t w4 = load_le16(bytes + 8);
            const uint64_t shader =
               ((uint64_t)((w1 >> 7) & 0x1ff) << 6) |
               ((uint64_t)w2 << 15) | ((uint64_t)w3 << 31) |
               ((uint64_t)(w4 & 1) << 47);
            if ((shader & 63) || !validate_shader_blob(memory, shader))
               return fail_at(cursor, "Ioto LDSHDR shader is not owned");
            if (fragment)
               stats->fragment_code = shader;
            seen_ldshdr = true;
            break;
         }
         case AGX_IOTO_KIND_SPECTPR_EP0:
            if (!seen_ldshdr || seen_spectpr)
               return fail_at(cursor, "misordered Ioto SPECTPR");
            seen_spectpr = true;
            break;
         case AGX_IOTO_KIND_PIXREL_EP0:
            if (!fragment || !seen_spectpr || seen_pixrel)
               return fail_at(cursor, "invalid Ioto PIXREL");
            seen_pixrel = true;
            break;
         case AGX_IOTO_KIND_B_EP0: {
            /* An unconditional branch ends this linear run: the bytes after
             * it are padding or another program, so a straight walk past it
             * reads garbage. hk emits two:
             *
             *  - relative: the external-setup graphics launcher
             *    (agx_ioto_g17p_external_setup_launcher_compose_trusted) jumps to
             *    the persistent setup suffix holding the body and the
             *    terminator. That suffix is Ioto; keep walking it, carrying
             *    the state-tail and immediate tracking across.
             *
             *  - absolute: the constant program (agx_ioto_constant_program_
             *    emit, hk_agx3_g17p_constant_program_from_binary) replaces
             *    the program's terminator with a jump to the shader's
             *    preamble. That is ordinary compiled code ending in its own
             *    stop, not Ioto; the program is complete at the jump.
             *
             * GPU0 B: bits 25..71 hold the halfword target and bit 24
             * selects an absolute target; clear, the target is a
             * displacement from this record (agx_ioto_preshader_pack_branch
             * and agx_ioto_constant_program_emit). */
            if (consumed != 10 || ++branches > 4)
               return fail_at(cursor, "Ioto branch chain is unbounded");
            uint64_t field = 0;
            for (unsigned b = 0; b < 6; ++b)
               field |= (uint64_t)bytes[3 + b] << (8 * b);
            const bool absolute = field & 1;
            field &= ~UINT64_C(1);
            const uint64_t target =
               absolute ? field
                        : (uint64_t)((int64_t)cursor +
                                     ((int64_t)(field << 16) >> 16));
            size_t target_available;
            if (target >= (UINT64_C(1) << 48) ||
                !memory->span(memory->data, target, &target_available) ||
                target_available < 4)
               return fail_at(cursor, "Ioto branch target is unowned");

            if (absolute) {
               if (!seen_speclm || !seen_ldshdr || !seen_spectpr ||
                   fragment != seen_pixrel)
                  return fail_at(cursor,
                                 "incomplete Ioto graphics state tail");
               size_t end;
               if (!code_reaches_stop(memory, target, target_available,
                                      &end))
                  return fail_at(target,
                                 "constant program has no decodable stop");
               return true;
            }

            address = target;
            available = MIN2(target_available, MAX_IOTO_PROGRAM_SIZE);
            cursor = target;
            continue;
         }
         default:
            if (seen_speclm && !seen_spectpr)
               return fail_at(cursor, "foreign record inside Ioto state tail");
            break;
         }
      }
      cursor += consumed;
   }

   return fail_at(cursor, "Ioto program has no bounded terminator");
}

static bool
validate_compact_compute_launch(
   const struct asahi_g17p_stream_memory *memory, uint64_t address,
   const uint8_t bytes[AGX3_CDM_INDIRECT_LAUNCH_SIZE])
{
   struct agx3_cdm_indirect_launch_info launch;
   if (!agx3_cdm_indirect_launch_unpack(
          AGX_CHIP_G17P, bytes, AGX3_CDM_INDIRECT_LAUNCH_SIZE, &launch) ||
       (launch.word0_payload >> 27) != 2 ||
       launch.indirect_word4_payload != 0 || !launch.pipeline ||
       (launch.pipeline & 63) ||
       launch.opaque_word3 !=
          (UINT32_C(0x40000000) | ((launch.pipeline >> 40) & 0xff)) ||
       !launch.indirect_parameters || (launch.indirect_parameters & 3) ||
       !memory->owned(memory->data, launch.indirect_parameters, 6 * 4))
      return fail_at(address, "invalid compact indirect compute launch");

   struct asahi_g17p_stream_stats shader_stats = {0};
   if (!validate_ioto_program(memory, launch.pipeline, false, true,
                              &shader_stats))
      return fail_at(address, "compact compute launch Ioto program is invalid");

   return true;
}

bool
asahi_g17p_compute_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_compute *compute,
   struct asahi_g17p_compute_stats *stats)
{
   struct asahi_g17p_compute_stats local = {0};
   bool need_barrier = false;
   /* need_barrier was set by a direct launch. Honeykrisp's batched
    * tessellation stages (hk_tess_batch.h) put the direct launches of one
    * stage back to back; anything else after a launch still needs a barrier
    * first. */
   bool direct_tail = false;
   bool followed_link = false;

   if (stats)
      memset(stats, 0, sizeof(*stats));
   if (!memory || !memory->read || !memory->owned || !memory->span ||
       !compute || compute->flags || !compute->cdm_ctrl_stream_base ||
       compute->cdm_ctrl_stream_end <= compute->cdm_ctrl_stream_base ||
       compute->cdm_ctrl_stream_end - compute->cdm_ctrl_stream_base >
          SIZE_MAX ||
       !memory->owned(memory->data, compute->cdm_ctrl_stream_base,
                      (size_t)(compute->cdm_ctrl_stream_end -
                               compute->cdm_ctrl_stream_base)))
      return fail_at(compute ? compute->cdm_ctrl_stream_base : 0,
                     "compute stream bounds are invalid");

   uint64_t cursor = compute->cdm_ctrl_stream_base;
   for (unsigned iteration = 0; iteration < MAX_RECORDS; ++iteration) {
      size_t span;
      if (!memory->span(memory->data, cursor, &span) || span < 4)
         return fail_at(cursor, "compute record is not owned");

      if (!followed_link && cursor >= compute->cdm_ctrl_stream_end)
         return fail_at(cursor, "compute stream ended without terminate");

      const size_t available = followed_link
                                  ? span
                                  : MIN2(span, (size_t)(
                                                  compute->cdm_ctrl_stream_end -
                                                  cursor));
      uint8_t bytes[G17P_CDM_DIRECT_LAUNCH_SIZE] = {0};
      const size_t readable = MIN2(available, sizeof(bytes));
      if (!read_owned(memory, cursor, bytes, readable))
         return fail_at(cursor, "compute record is not owned");
      if (memory->verbose) {
         fprintf(stderr, "shim: G17P CDM %#llx:",
                 (unsigned long long)cursor);
         for (size_t i = 0; i < readable; ++i)
            fprintf(stderr, " %02x", bytes[i]);
         fprintf(stderr, "\n");
      }

      /* Mode 2 is a compact application launch reading six geometry words.
       * The preceding geometry producer is an ordinary direct launch. Bit 23
       * is part of the resource template and does not select indirect mode. */
      const uint32_t header = agx3_cmdstream_load_le32(bytes);
      if ((header >> 29) == AGX_CDM_BLOCK_TYPE_LAUNCH &&
          ((header >> 27) & 3) == 2) {
         if (need_barrier || available < AGX3_CDM_INDIRECT_LAUNCH_SIZE ||
             !validate_compact_compute_launch(memory, cursor, bytes))
            return fail_at(cursor, "invalid compact compute launch ordering");

         local.launches++;
         need_barrier = true;
         direct_tail = false;
         cursor += AGX3_CDM_INDIRECT_LAUNCH_SIZE;
         continue;
      }

      struct agx3_cdm_decoded_record record;
      if (!agx3_cdm_decode_one(AGX_CHIP_G17P, bytes, readable, &record) ||
          !record.length || record.length > available)
         return fail_at(cursor, "unknown or truncated AGX3 compute record");

      switch (record.kind) {
      case AGX3_CDM_RECORD_LAUNCH: {
         if ((need_barrier && !direct_tail) ||
             (record.data.launch.word0_payload >> 27) != 0 ||
             (record.data.launch.pipeline & 63) ||
             !record.data.launch.pipeline ||
             record.length != G17P_CDM_DIRECT_LAUNCH_SIZE ||
             record.data.launch.opaque_word3 !=
                (UINT32_C(0x40000000) |
                 ((record.data.launch.pipeline >> 40) & 0xff)))
            return fail_at(cursor, "compute launch ordering is invalid");

         for (unsigned axis = 0; axis < 3; ++axis) {
            if (!record.data.launch.global_size[axis] ||
                !record.data.launch.local_size[axis])
               return fail_at(cursor, "compute launch has a zero dimension");
         }

         struct asahi_g17p_stream_stats shader_stats = {0};
         if (!validate_ioto_program(memory, record.data.launch.pipeline,
                                    false, true, &shader_stats))
            return fail_at(cursor, "compute launch Ioto program is invalid");

         local.launches++;
         local.unbarriered_launches += need_barrier;
         local.two_dimensional_launches +=
            record.data.launch.global_size[1] > 1 &&
            record.data.launch.global_size[2] == 1;
         need_barrier = true;
         direct_tail = true;
         break;
      }

      case AGX3_CDM_RECORD_BARRIER:
         if (record.data.barrier.flags != 0 &&
             record.data.barrier.flags != 0x5)
            return fail_at(cursor, "compute barrier is not canonical");

         if (!need_barrier) {
            /* Immediate post-graphics writes begin with the strong cache
             * flush which makes the preceding render visible.  It is an
             * ordered stream prologue, not a launch terminator. */
            if (local.launches != 0 ||
                local.prologue_barriers != 0 ||
                record.data.barrier.flags != 0x5)
               return fail_at(cursor, "compute barrier is not canonical");

            local.prologue_barriers++;
            local.strong_barriers++;
            break;
         }

         local.barriers++;
         local.strong_barriers += record.data.barrier.flags == 0x5;
         need_barrier = false;
         break;

      case AGX3_CDM_RECORD_SSTT_FLUSH_INVALIDATE:
         /* HAL200 emits SSTT maintenance in a command encoder of its own.
          * Combining it with shader launches would leave ordering ambiguous. */
         if (need_barrier || local.launches || local.sstt_flushes)
            return fail_at(cursor, "SSTT flush-invalidate ordering is invalid");

         local.sstt_flushes++;
         break;

      case AGX3_CDM_RECORD_STREAM_TERMINATE:
         /* A stream whose only record is the opening strong flush is a real
          * shape, not an empty stream: hk_CmdPipelineBarrier2() emits exactly
          * that for an API-declared HOST_READ dependency when the command
          * buffer records no other compute work, and it is what makes a
          * preceding fragment store visible to the CPU (Mesa 6aae441547b,
          * qualified on hardware).  It reaches submission alone whenever the
          * neighbouring control streams are render ones, so it cannot be
          * merged into a compute stream that has launches. */
         if (need_barrier ||
             (!local.launches && !local.sstt_flushes &&
              !local.prologue_barriers) ||
             (!followed_link &&
              cursor + record.length != compute->cdm_ctrl_stream_end))
            return fail_at(cursor, "compute terminator is not final");
         local.terminated = true;
         if (stats)
            *stats = local;
         return true;

      case AGX3_CDM_RECORD_STREAM_LINK: {
         const uint64_t target = record.data.stream_link.target;
         if (need_barrier || record.data.stream_link.with_return || (target & 3) ||
             !memory->owned(memory->data, target, 4))
            return fail_at(cursor, "compute stream link is invalid");

         cursor = target;
         followed_link = true;
         continue;
      }

      case AGX3_CDM_RECORD_LINK_PADDING:
         oracle_gap_at(cursor, "CDM link padding");
         break;

      case AGX3_CDM_RECORD_SUBSTREAM_BARRIER:
         oracle_gap_at(cursor, "CDM substream barrier");
         break;

      case AGX3_CDM_RECORD_SECONDARY_BARRIER:
         oracle_gap_at(cursor, "CDM secondary barrier");
         break;

      case AGX3_CDM_RECORD_UNKNOWN:
      default:
         return fail_indeterminate_at(cursor, "unrecognised CDM record");
      }

      cursor += record.length;
   }

   return fail_at(cursor, "compute record limit reached before terminate");
}

static uint64_t
render_resource_spec(const struct drm_asahi_bg_eot *program, uint32_t high)
{
   return program->rsrc_spec | ((uint64_t)high << 32);
}

static bool
validate_render_aux_program(const struct asahi_g17p_stream_memory *memory,
                            const struct drm_asahi_bg_eot *program,
                            const char *name, bool fragment,
                            bool nested_tables)
{
   if ((program->usc & 0xf) != 4) {
      fprintf(stderr, "shim: G17P %s USC tag is not 4\n", name);
      return false;
   }

   const uint64_t address = G17P_USC_BASE | (program->usc & ~UINT32_C(0xf));
   struct asahi_g17p_stream_stats stats = {0};
   if (!validate_ioto_program(memory, address, fragment, nested_tables,
                              &stats)) {
      fprintf(stderr, "shim: G17P %s Ioto program %#llx is invalid\n", name,
              (unsigned long long)address);
      return false;
   }

   return true;
}

bool
asahi_g17p_render_aux_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_render *render)
{
   if (!memory || !render)
      return false;

   const uint64_t bg =
      render_resource_spec(&render->bg, render->bg_rsrc_spec_hi);
   const uint64_t partial_bg = render_resource_spec(
      &render->partial_bg, render->bg_partial_rsrc_spec_hi);
   const uint64_t eot =
      render_resource_spec(&render->eot, render->bg_eot_rsrc_spec_hi);
   const uint64_t partial_eot = render_resource_spec(
      &render->partial_eot, render->bg_eot_partial_rsrc_spec_hi);
   const uint32_t high = render->bg_rsrc_spec_hi |
                         render->bg_partial_rsrc_spec_hi |
                         render->bg_eot_rsrc_spec_hi |
                         render->bg_eot_partial_rsrc_spec_hi;
   if (!!high != !!(render->flags & DRM_ASAHI_RENDER_RSRC_SPEC_HI))
      return fail_at(0, "render resource-spec high flag disagrees with payload");

   const uint64_t bg_resources =
      AGX_IOTO_G17P_BG_RESOURCE_SPEC & ((UINT64_C(1) << 47) - 1);
   const uint64_t bg_writes = bg_resources | (UINT64_C(0x1ffff) << 47);
   if ((bg != bg_resources && bg != bg_writes) ||
       (partial_bg != bg_resources && partial_bg != bg_writes) ||
       (eot != AGX_IOTO_G17P_EOT_RESOURCE_SPEC && eot != bg_resources) ||
       (partial_eot != AGX_IOTO_G17P_EOT_RESOURCE_SPEC &&
        partial_eot != bg_resources))
      return fail_at(0, "render auxiliary resource spec is not canonical");

   return validate_render_aux_program(memory, &render->bg, "background",
                                      true, true) &&
          validate_render_aux_program(memory, &render->partial_bg,
                                      "partial background", true, true) &&
          validate_render_aux_program(memory, &render->eot, "end-of-tile",
                                      eot != AGX_IOTO_G17P_EOT_RESOURCE_SPEC,
                                      false) &&
          validate_render_aux_program(memory, &render->partial_eot,
                                      "partial end-of-tile",
                                      partial_eot !=
                                         AGX_IOTO_G17P_EOT_RESOURCE_SPEC,
                                      false);
}

static bool
validate_slot(const struct asahi_g17p_stream_memory *memory, unsigned slot,
              uint64_t address, size_t size,
              struct asahi_g17p_stream_stats *stats)
{
   uint8_t bytes[AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH];
   if (size > sizeof(bytes) || !read_owned(memory, address, bytes, size))
      return fail_at(address, "bind slot has no owning VM mapping");

   struct agx_ppp_hal200_graphics_fields graphics;
   struct agx_ppp_viewport_payload_fields viewport;
   struct agx_ppp_w_clamp_cull2_fields w_cull2;

   switch (slot) {
   case 0: { /* Varyings */
      if (size != 20 || load_le32(bytes) != AGX_G17P_VARYING_RECORD_WORD0)
         return fail_at(address, "invalid 20-byte varyings slot");
      if (((load_le32(bytes + 4) | load_le32(bytes + 8)) &
           UINT32_C(0xff000000)) || load_le32(bytes + 12) ||
          load_le32(bytes + 16))
         return fail_at(address, "varyings slot has reserved bits");
      return true;
   }
   case 1: { /* Fragment program and VIT */
      if (size != 28 || load_le32(bytes) != AGX_PPP_HAL200_FS_HEADER ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_FS, bytes, size,
             &graphics))
         return fail_at(address, "invalid 28-byte fragment slot");

      const uint64_t vit =
         ((uint64_t)(load_le32(bytes + 4) & UINT32_C(0xffff)) << 32) |
         load_le32(bytes + 8);
      const uint64_t program = decode_pipeline_pair(bytes + 16);
      if (memory->verbose)
         fprintf(stderr,
                 "shim: G17P FS slot=%#llx VIT=%#llx program=%#llx\n",
                 (unsigned long long)address, (unsigned long long)vit,
                 (unsigned long long)program);
      if ((vit & 63) || (program & 63) || load_le32(bytes + 12) ||
          load_le32(bytes + 24) ||
          (vit && !memory->owned(memory->data, vit, 4)) || !program ||
          !validate_ioto_program(memory, program, true, true, stats)) {
         fprintf(stderr,
                 "shim: G17P fragment slot %#llx has unowned VIT %#llx or "
                 "program %#llx\n",
                 (unsigned long long)address, (unsigned long long)vit,
                 (unsigned long long)program);
         return false;
      }
      stats->varying_fs |= vit != 0;
      stats->no_varying_fs |= vit == 0;
      return true;
   }
   case 2: {
      if (size != 28 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_OUTPUT, bytes, size,
             &graphics) ||
          /* The low sixteen bits are the eight clip-distance enables and
           * their companion select bits. agx_ppp_hal200_output_select_pack()
           * bounds the live producer to this same field; rejecting it here
           * made valid XFB clip/cull streams look malformed. */
          (graphics.words[0] & ~UINT32_C(0x003dffff)) ||
          !!(graphics.words[0] & UINT32_C(0x00080000)) !=
             !!(graphics.words[0] & UINT32_C(0x00100000)) ||
          graphics.words[1] ||
          graphics.words[2] > 15 ||
          (graphics.words[3] & UINT32_C(0xff00ff00)) ||
          graphics.words[4] != AGX_PPP_HAL200_PRIMARY_FACE_HEADER ||
          (graphics.words[5] & UINT32_C(0x00040200)) !=
             UINT32_C(0x00040200) ||
          (graphics.words[5] & ~UINT32_C(0x040fc200)))
         return fail_at(address, "output slot has non-canonical semantics");
      return true;
   }
   case 3: {
      if (size != 28 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, bytes,
             size, &graphics))
         return fail_at(address, "invalid primary-face slot");
      stats->depth_enabled |= ((graphics.words[1] >> 24) & 7) != 7 ||
                              !(graphics.words[1] & (UINT32_C(1) << 21));
      const bool stencil_enabled =
         graphics.words[2] != UINT32_C(0x0e000000) ||
         graphics.words[4] != UINT32_C(0x0e000000);
      stats->stencil_enabled |= stencil_enabled;
      stats->stencil_state_updates += stencil_enabled;
      stats->primary_tri_merge_disable |=
         (graphics.words[0] & (UINT32_C(1) << 26)) != 0;
      return true;
   }
   case 4: {
      if (size != 20 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, bytes,
             size, &graphics) || graphics.words[1] != graphics.words[2] ||
          graphics.words[3] != UINT32_C(0x0001ffff) ||
          (graphics.words[0] & ~UINT32_C(0x64200200)) ||
          (graphics.words[0] & UINT32_C(0x200)) == 0)
         return fail_at(address, "secondary-face slot is non-canonical");
      stats->blend_enabled |= (graphics.words[0] >> 29) != 0;
      stats->secondary_tri_merge_disable |=
         (graphics.words[0] & (UINT32_C(1) << 26)) != 0;
      return true;
   }
   case 5:
      if (!agx_ppp_viewport_payload_core_unpack(
             AGX_CHIP_G17P, bytes, size, &viewport))
         return fail_at(address, "invalid viewport slot");
      stats->viewport_count = viewport.viewport_count;
      if (memory->verbose && viewport.viewport_count > 1) {
         fprintf(stderr,
                 "shim: viewport[1] tx=%#x sx=%#x ty=%#x sy=%#x\n",
                 viewport.viewports[1].translate_x,
                 viewport.viewports[1].scale_x,
                 viewport.viewports[1].translate_y,
                 viewport.viewports[1].scale_y);
      }
      stats->multi_viewport |=
         viewport.viewport_count == 16 &&
         viewport.viewports[1].translate_x == UINT32_C(0x41b80000) &&
         viewport.viewports[1].scale_x == UINT32_C(0x41200000) &&
         viewport.viewports[1].translate_y == UINT32_C(0x42180000) &&
         viewport.viewports[1].scale_y == UINT32_C(0xc1100000);
      stats->two_viewport |=
         viewport.viewport_count == 2 &&
         viewport.viewports[0].translate_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].translate_y == UINT32_C(0x41800000) &&
         viewport.viewports[0].scale_y == UINT32_C(0x41800000) &&
         viewport.viewports[1].translate_x == UINT32_C(0x41c00000) &&
         viewport.viewports[1].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[1].translate_y == UINT32_C(0x41800000) &&
         viewport.viewports[1].scale_y == UINT32_C(0x41800000);
      stats->four_viewport |=
         viewport.viewport_count == 4 &&
         viewport.viewports[0].translate_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].translate_y == UINT32_C(0x41000000) &&
         viewport.viewports[0].scale_y == UINT32_C(0x41000000) &&
         viewport.viewports[1].translate_x == UINT32_C(0x41c00000) &&
         viewport.viewports[1].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[1].translate_y == UINT32_C(0x41000000) &&
         viewport.viewports[1].scale_y == UINT32_C(0x41000000) &&
         viewport.viewports[2].translate_x == UINT32_C(0x41000000) &&
         viewport.viewports[2].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[2].translate_y == UINT32_C(0x41c00000) &&
         viewport.viewports[2].scale_y == UINT32_C(0x41000000) &&
         viewport.viewports[3].translate_x == UINT32_C(0x41c00000) &&
         viewport.viewports[3].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[3].translate_y == UINT32_C(0x41c00000) &&
         viewport.viewports[3].scale_y == UINT32_C(0x41000000);
      stats->viewport_changed |=
         viewport.viewports[0].translate_x == UINT32_C(0x41d80000) &&
         viewport.viewports[0].scale_x == UINT32_C(0x41c00000) &&
         viewport.viewports[0].translate_y == UINT32_C(0x41c80000) &&
         viewport.viewports[0].scale_y == UINT32_C(0x41a00000);
      return true;
   case 6:
      return size == 12 && agx_ppp_w_clamp_cull2_core_unpack(
                              AGX_CHIP_G17P, bytes, size, &w_cull2);
   case 7:
      if (size != 8 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_CULL, bytes, size,
             &graphics) || (graphics.words[0] & ~UINT32_C(0x00030d83)))
         return fail_at(address, "cull slot has reserved bits");
      stats->cull_enabled |= (graphics.words[0] & 3) != 0;
      stats->rasterizer_discard |=
         (graphics.words[0] & (UINT32_C(1) << 17)) != 0;
      return true;
   default:
      return false;
   }
}

static bool
validate_bind_tail(const struct asahi_g17p_stream_memory *memory,
                   uint64_t address,
                   struct asahi_g17p_stream_stats *stats)
{
   uint32_t controls[8] = {
      0x700, 0x500, 0x700, 0x500, 0, 0x300, 0x200, 0x200,
   };
   size_t lengths[8] = {20, 28, 28, 28, 20, 0, 12, 8};
   uint8_t bytes[AGX_VDM_HAL200_BIND_TAIL_LENGTH];
   struct agx_vdm_hal200_bind_pair decoded[8];

   if (!read_owned(memory, address, bytes, sizeof(bytes)) ||
       load_le32(bytes) != UINT32_C(0x00000500) ||
       !agx_vdm_hal200_bind_tail_unpack(AGX_CHIP_G17P, bytes, sizeof(bytes),
                                        decoded))
      return fail_at(address, "invalid 68-byte HAL200 bind tail");

   /* The complete viewport payload is immediately followed by the varying
    * record. Their independently addressed bind slots therefore make the
    * variable 1..16 viewport length self-describing. */
   if (decoded[0].context_offset <= decoded[5].context_offset)
      return fail_at(address, "viewport/varying slots are not ordered");
   lengths[5] = decoded[0].context_offset - decoded[5].context_offset;
   if (lengths[5] < AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH ||
       lengths[5] > AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH)
      return fail_at(address, "viewport slot length is out of range");
   const unsigned viewport_count = (lengths[5] - 8) / 32;

   /* Pair 4 sizes the viewport payload in words. A stale one-viewport size
    * on a longer payload leaves the tiler parsing a truncated PPP record. */
   controls[AGX_VDM_HAL200_BIND_VIEWPORT_CONTROL_PAIR] =
      agx_vdm_hal200_bind_viewport_control(lengths[5]);

   uint8_t output_record[AGX_PPP_HAL200_OUTPUT_LENGTH];
   uint8_t primary_record[AGX_PPP_HAL200_PRIMARY_FACE_LENGTH];
   uint8_t secondary_record[AGX_PPP_HAL200_SECONDARY_FACE_LENGTH];
   uint8_t cull_record[AGX_PPP_HAL200_CULL_LENGTH];
   if (!read_owned(memory, compact_address(decoded[2].context_offset),
                   output_record, sizeof(output_record)) ||
       !read_owned(memory, compact_address(decoded[3].context_offset),
                   primary_record, sizeof(primary_record)) ||
       !read_owned(memory, compact_address(decoded[4].context_offset),
                   secondary_record, sizeof(secondary_record)) ||
       !read_owned(memory, compact_address(decoded[7].context_offset),
                   cull_record, sizeof(cull_record)) ||
       load_le32(output_record + 12) != viewport_count - 1)
      return fail_at(address, "output/viewport counts disagree");

   if (!!(load_le32(output_record + 24) & (UINT32_C(1) << 26)) !=
       !!(load_le32(primary_record + 4) & (UINT32_C(1) << 26)))
      return fail_at(address, "output/primary triangle-merge controls disagree");

   /* SECONDARY bit 21 is tag-write-disable. Rasterizer discard also sets it,
    * but a depth/stencil-only fragment pass legitimately disables colour tag
    * writes without discarding rasterization. CULL bit 17 is the definitive
    * discard control; when it is set the companion tag-write bit is required. */
   const bool tag_write_disable =
      (load_le32(secondary_record + 4) & (UINT32_C(1) << 21)) != 0;
   const bool rasterizer_discard =
      (load_le32(cull_record + 4) & (UINT32_C(1) << 17)) != 0;
   if (rasterizer_discard && !tag_write_disable)
      return fail_at(address, "rasterizer discard leaves tag writes enabled");

   /* OUTPUT bits 19 and 20 are the paired viewport/layer UVS selector.  A
    * multiview stream is not complete merely because it exposes two image
    * layers: the vertex ViewIndex must also reach this native output path. */
   stats->layer_output |=
      (load_le32(output_record + 4) & UINT32_C(0x00180000)) ==
      UINT32_C(0x00180000);

   for (unsigned i = 0; i < 8; ++i) {
      /* Fixed controls and lengths are intentionally independent of the
       * producer's slot construction. */
      if (decoded[i].control != controls[i])
         return fail_at(address + 4 + i * 8, "invalid bind-slot control");

      const uint64_t slot = compact_address(decoded[i].context_offset);
      if (!validate_slot(memory, i, slot, lengths[i], stats))
         return false;
   }

   return true;
}

static bool
validate_initial_ppp(const struct asahi_g17p_stream_memory *memory,
                     uint64_t address, uint32_t size_words)
{
   uint8_t bytes[12];
   struct agx_ppp_w_clamp_cull2_fields state;

   address = compact_address(address);
   if (size_words != 3 || !read_owned(memory, address, bytes, sizeof(bytes)) ||
       !agx_ppp_w_clamp_cull2_core_unpack(AGX_CHIP_G17P, bytes, sizeof(bytes),
                                          &state))
      return fail_at(address, "invalid initial 12-byte PPP state");

   return true;
}

bool
asahi_g17p_stream_validate(const struct asahi_g17p_stream_memory *memory,
                           const struct asahi_g17p_stream_config *config,
                           struct asahi_g17p_stream_stats *stats)
{
   struct asahi_g17p_stream_stats local = {0};
   bool pending_selector = false;
   bool have_state = false;
   bool tessellation_state_seen = false;
   bool restart_seen = false;
   /* Honeykrisp re-emits the restart State Update only when the restart
    * index changes, and forgets it where the driver does: after an indirect
    * packet (indexed ones write their own) and at a stream link, which also
    * covers an empty robust-index draw's jump. A non-empty robust-index draw
    * carries its own record. */
   bool restart_held = false;
   /* Cleared by every draw body, set by every barrier record.  A state
    * republication seen while this is false is a draw installing fresh VDM
    * pointers over the ones an earlier draw in this same stream may still be
    * dereferencing. */
   bool barrier_since_last_draw = false;
   const uint64_t root = config ? config->root : 0;
   uint64_t cursor = root;

   if (stats)
      memset(stats, 0, sizeof(*stats));
   if (!config || !memory || !memory->read || !memory->owned ||
       !memory->owned(memory->data, root, 4))
      return fail_at(root, "root is not owned");

   for (unsigned iteration = 0; iteration < MAX_RECORDS; ++iteration) {
      uint8_t bytes[68];
      if (!read_owned(memory, cursor, bytes, 4))
         return fail_at(cursor, "record header is not owned");

      if (pending_selector) {
         const uint32_t selector_offset = load_le32(bytes);
         const uint64_t selector = compact_address(selector_offset);
         uint8_t indices[8];
         /* Zero is not a selector. Every CPU-side packer refuses it outright
          * (agx_vdm_hal200_{direct,indexed,indirect}_draw_pack all reject
          * draw_state_context_offset == 0), but a GPU-side record builder can
          * still be handed 0 -- hk_cmd_draw.c's robust-index path initialises
          * draw_state_offset to 0 and only fills it when the native bind tail
          * was emitted. Say that plainly: compact_address(0) resolves to the
          * render context base, which is mapped, so the walk would otherwise
          * read the context BO and report a meaningless flag mismatch. */
         if (selector_offset == 0)
            return fail_at(cursor,
                           "draw record carries a null draw-state selector "
                           "offset, which every CPU-side packer refuses");
         if ((selector_offset & 3) ||
             !read_owned(memory, selector, indices, sizeof(indices)))
            return fail_at(cursor, "draw-state selector is not owned");
         if (load_le32(indices) != AGX_G17P_DRAW_STATE_SELECTOR_WORD0) {
            /* The expectation is the encoder's own constant, not a restated
             * one: agx_g17p_ppp_indices_pack() and its base variant are the
             * only writers of this record and both emit it unconditionally.
             *
             * Print what was actually read. A mismatch here is either a driver
             * that stopped emitting the selector the encoder defines, or a
             * walk that resolved something which is not a selector at all, and
             * a bare "flags are invalid" cannot tell those apart -- which is
             * why a batch of these rejections could not be adjudicated. */
            fprintf(stderr,
                    "shim: G17P draw-state selector at %#llx has word0 %#x, "
                    "encoder emits %#x\n",
                    (unsigned long long)selector, load_le32(indices),
                    AGX_G17P_DRAW_STATE_SELECTOR_WORD0);
            return fail_at(selector, "draw-state selector flags are invalid");
         }
         if (local.viewport_count == 0 ||
             local.viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS)
            return fail_at(selector, "selector has no bounded viewport state");
         const uint64_t scissor_base =
            config->scissor_base + (uint64_t)load_le16(indices + 4) * 16;
         const uint64_t depth_bias_base =
            config->depth_bias_base + (uint64_t)load_le16(indices + 6) * 12;
         for (unsigned i = 0; i < local.viewport_count; ++i) {
            const uint64_t scissor = scissor_base + (uint64_t)i * 16;
            const uint64_t depth_bias = depth_bias_base + (uint64_t)i * 12;
            uint8_t scissor_record[16], depth_bias_record[12];
            if (!read_owned(memory, scissor, scissor_record,
                            sizeof(scissor_record)) ||
                !read_owned(memory, depth_bias, depth_bias_record,
                            sizeof(depth_bias_record)))
               return fail_at(selector,
                              "selected raster array is not fully owned");
            if (load_le16(scissor_record) < load_le16(scissor_record + 2) ||
                load_le16(scissor_record + 4) < load_le16(scissor_record + 6))
               return fail_at(scissor,
                              "selected scissor extents are inverted");
            if (i == 0) {
               local.scissor_changed |= load_le16(scissor_record) == 30 &&
                                        load_le16(scissor_record + 2) == 7 &&
                                        load_le16(scissor_record + 4) == 28 &&
                                        load_le16(scissor_record + 6) == 9;
            }
         }
         pending_selector = false;
         cursor += 4;
         continue;
      }

      if (!read_owned(memory, cursor, bytes, sizeof(bytes)))
         return fail_at(cursor, "record crosses its owning mapping");

      /* HAL200 bind state is independent of the retained vertex program.
       * PPP-only updates and clean indirect draws need a bind tail/selector
       * without a preceding VS update. Do not decode its first bind pair as
       * the separate initial 12-byte PPP record. */
      if (have_state && load_le32(bytes) == UINT32_C(0x00000500)) {
         if (!validate_bind_tail(memory, cursor, &local))
            return false;
         cursor += AGX_VDM_HAL200_BIND_TAIL_LENGTH;
         pending_selector = true;
         continue;
      }

      struct agx_vdm_decoded_record record;
      if (!agx_vdm_decode_one(AGX_CHIP_G17P, bytes, sizeof(bytes), &record)) {
         if (load_le32(bytes) == UINT32_C(0x40000001)) {
            if (restart_seen)
               return fail_at(cursor, "duplicate indexed restart record");
            restart_seen = true;
            restart_held = true;
            cursor += 8;
            continue;
         }
         return fail_at(cursor, "unknown or non-canonical AGX3 record");
      }

      switch (record.kind) {
      case AGX_VDM_RECORD_BARRIER:
         local.barriers++;
         local.inter_draw_barriers += local.draws > 0;
         barrier_since_last_draw = true;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_PPP_STATE_UPDATE:
         if (!validate_initial_ppp(memory,
                                   record.data.ppp_state_update.pointer,
                                   record.data.ppp_state_update.size_words))
            return false;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_STATE_UPDATE:
         if (memory->verbose)
            fprintf(stderr, "shim: G17P VS program=%#llx\n",
                    (unsigned long long)record.data.state_update.pipeline);
         if (!validate_ioto_program(memory,
                                    record.data.state_update.pipeline, false,
                                    true,
                                    &local)) {
            fprintf(stderr,
                    "shim: G17P vertex program %#llx at VDM %#llx is "
                    "unowned\n",
                    (unsigned long long)record.data.state_update.pipeline,
                    (unsigned long long)cursor);
            return false;
         }
         cursor += record.length;
         if (!validate_bind_tail(memory, cursor, &local))
            return false;
         cursor += AGX_VDM_HAL200_BIND_TAIL_LENGTH;
         pending_selector = true;
         have_state = true;
         local.state_updates++;
         if (local.draws > 0 && !barrier_since_last_draw)
            local.unbarriered_state_republications++;
         break;

      case AGX_VDM_RECORD_INDEX_LIST_NONINDEXED:
         if (!have_state || restart_seen)
            return fail_at(cursor, "non-indexed draw has invalid state order");
         local.draws++;
         barrier_since_last_draw = false;
         if (record.data.index_list_nonindexed.instances >
             local.max_instance_count)
            local.max_instance_count =
               record.data.index_list_nonindexed.instances;
         local.primitive_mask |=
            UINT32_C(1) << record.data.index_list_nonindexed.primitive;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_INDEX_LIST_INDEXED: {
         const struct agx_vdm_index_list_direct_indexed_info *draw =
            &record.data.index_list_indexed;
         const size_t index_size =
            draw->index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U8  ? 1
            : draw->index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U16 ? 2
                                                                  : 4;
         const uint64_t bytes_needed = (uint64_t)draw->count * index_size;
         if (!have_state || !(restart_seen || restart_held) ||
             bytes_needed > SIZE_MAX ||
             !memory->owned(memory->data, draw->index_buffer,
                            (size_t)bytes_needed))
            return fail_at(cursor, "indexed draw buffer is not owned");
         restart_seen = false;
         local.draws++;
         barrier_since_last_draw = false;
         local.indexed_draws++;
         local.primitive_mask |= UINT32_C(1) << draw->primitive;
         local.index_size_mask |= UINT8_C(1) << draw->index_size;
         local.restart_enabled |= draw->restart_enable;
         local.restart_disabled |= !draw->restart_enable;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_STREAM_LINK: {
         const uint64_t compact_target = record.data.stream_link.target;
         uint64_t target;
         if (!compact_target || (compact_target & 3) ||
             !agx_g17p_vdm_payload_expand(compact_target, 4, &target) ||
             target == cursor || target == config->root ||
             !memory->owned(memory->data, target, 4))
            return fail_at(cursor, "stream-link target is invalid or unowned");
         cursor = target;
         local.links++;
         restart_held = false;
         break;
      }

      case AGX_VDM_RECORD_STREAM_TERMINATE:
         /* Empty VDM streams are valid for clear-only render passes, whose
          * useful work is performed by the background/EOT programs. */
         if (pending_selector || restart_seen)
            return fail_at(cursor, "stream terminates with incomplete draw");
         local.clear_only = local.draws == 0;
         local.body_only = local.draws > local.state_updates;
         if (stats)
            *stats = local;
         return true;

      case AGX_VDM_RECORD_INDEX_LIST_INDIRECT_NONINDEXED: {
         const uint64_t indirect_buffer =
            record.data.index_list_indirect_nonindexed.indirect_buffer;
         if (!have_state || restart_seen ||
             !memory->owned(memory->data, indirect_buffer, 16))
            return fail_at(cursor, "indirect draw arguments are not owned");
         local.draws++;
         barrier_since_last_draw = false;
         local.primitive_mask |= UINT32_C(1) <<
                                 record.data.index_list_indirect_nonindexed.primitive;
         local.indirect_nonindexed = true;
         restart_held = false;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_INDEX_LIST_INDIRECT_INDEXED: {
         const struct agx_vdm_index_list_indirect_indexed_info *draw =
            &record.data.index_list_indirect_indexed;
         const size_t index_range =
            ((size_t)draw->index_buffer_size_word + 1) * 4;
         if (!have_state || restart_seen ||
             !memory->owned(memory->data, draw->indirect_buffer, 20) ||
             !memory->owned(memory->data, draw->index_buffer, index_range))
            return fail_at(cursor,
                           "indexed-indirect buffers are not owned");
         local.draws++;
         barrier_since_last_draw = false;
         local.indexed_draws++;
         local.primitive_mask |= UINT32_C(1) << draw->primitive;
         local.index_size_mask |= UINT8_C(1) << draw->index_size;
         local.restart_enabled |= draw->restart_enable;
         local.restart_disabled |= !draw->restart_enable;
         local.indirect_indexed = true;
         restart_held = false;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_TESSELLATION_STATE:
         if (!have_state || restart_seen)
            return fail_at(cursor, "tessellation state has invalid order");
         tessellation_state_seen = true;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_TESSELLATE_DIRECT: {
         const struct agx_vdm_tessellate_direct_info *draw =
            &record.data.tessellate_direct;
         const uint64_t factor_bytes =
            ((uint64_t)draw->factor_buffer_size_minus_one + 1) * 4;
         if (!have_state || restart_seen || !tessellation_state_seen ||
             draw->patch_count == 0 || draw->instance_count == 0 ||
             factor_bytes > SIZE_MAX ||
             !memory->owned(memory->data, draw->factor_buffer,
                            (size_t)factor_bytes))
            return fail_at(cursor,
                           "tessellate draw has no state or unowned factors");
         local.draws++;
         local.tessellated_draws++;
         barrier_since_last_draw = false;
         if (draw->instance_count > local.max_instance_count)
            local.max_instance_count = draw->instance_count;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_TESSELLATE_INDIRECT: {
         const struct agx_vdm_tessellate_indirect_info *draw =
            &record.data.tessellate_indirect;
         const uint64_t factor_bytes =
            ((uint64_t)draw->factor_buffer_size_minus_one + 1) * 4;
         if (!have_state || restart_seen || !tessellation_state_seen ||
             factor_bytes > SIZE_MAX ||
             !memory->owned(memory->data, draw->factor_buffer,
                            (size_t)factor_bytes) ||
             !memory->owned(memory->data, draw->indirect_buffer, 16))
            return fail_at(cursor, "indirect tessellate draw has no state or "
                                   "unowned factors or arguments");
         local.draws++;
         local.tessellated_draws++;
         barrier_since_last_draw = false;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_UNKNOWN:
      default:
         return fail_indeterminate_at(cursor, "unrecognised VDM record");
      }
   }

   return fail_at(cursor, "record limit reached before terminate");
}
